"""Potential maxima <-> void minima correspondence (full-box slices + stacked profiles).

Method- and simulation-agnostic via dtfelib/pipeline:

    python3 plot/plot_phi_delta.py                       # TNG50-4-Dark, all config snapshots
    python3 plot/plot_phi_delta.py --method dtfe
    python3 plot/plot_phi_delta.py --sim TNG50-3-Dark --snap 50
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sfft
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from dtfelib.figures import save_plot_to_multiple_paths
from dtfelib import make_parser
import config
from dtfelib import figures as style
from dtfelib import pipeline

style.apply()

OUTPUT_DIR = config.figures_path('extrema')

# Grid geometry defaults (refreshed per snapshot from the FieldSet metadata).
FIELD_RESOLUTION = config.FIELD_RESOLUTION
BOX_SIZE = config.BOX_SIZE
CELL_SIZE = config.CELL_SIZE
AXIS_UNITS = config.AXIS_UNITS
DPI = config.DPI


def _set_geometry(fs):
    """Take grid size / box size from the loaded fields instead of config constants."""
    global FIELD_RESOLUTION, BOX_SIZE, CELL_SIZE
    FIELD_RESOLUTION = fs.grid_n
    BOX_SIZE = fs.meta.box_mpc
    CELL_SIZE = BOX_SIZE / FIELD_RESOLUTION

MATCH_DIST_CELLS = config.EXTREMA_MATCH_DISTANCE_CELLS
NULL_DRAWS = config.EXTREMA_NULL_DRAWS
PROFILE_RMAX_MPC = config.PHI_PROFILE_RMAX_MPC
PROFILE_NBINS = config.PHI_PROFILE_NBINS

SLICE_TOLERANCE_CELLS = config.FOOTPRINT_SIZE // 2


def match_statistics(phi_coords, void_tree, n_voids):
    dist_cells, idx = void_tree.query(phi_coords, k=1)
    matched = dist_cells <= MATCH_DIST_CELLS
    return {
        'distances_cells': dist_cells,
        'void_index': idx,
        'matched': matched,
        'match_fraction': float(np.mean(matched)) if len(matched) else 0.0,
        'completeness': float(len(np.unique(idx[matched])) / n_voids)
        if n_voids else 0.0,
        'median_sep_mpc': float(np.median(dist_cells) * CELL_SIZE)
        if len(dist_cells) else np.nan,
    }


def null_statistics(n_phi, void_tree, rng):
    pts = rng.uniform(0, FIELD_RESOLUTION, size=(NULL_DRAWS * n_phi, 3))
    d, _ = void_tree.query(pts, k=1)
    d = d.reshape(NULL_DRAWS, n_phi)
    fractions = np.mean(d <= MATCH_DIST_CELLS, axis=1)
    medians_mpc = np.median(d, axis=1) * CELL_SIZE
    lo_f, hi_f = np.percentile(fractions, [2.5, 97.5])
    lo_m, hi_m = np.percentile(medians_mpc, [2.5, 97.5])
    return {'frac_lo': lo_f, 'frac_hi': hi_f, 'med_lo': lo_m, 'med_hi': hi_m}


def _tophat_window(k_times_r):
    x = k_times_r
    out = np.ones_like(x)
    nz = x > 1e-6
    xs = x[nz]
    out[nz] = 3.0 * (np.sin(xs) - xs * np.cos(xs)) / xs ** 3
    return out


def stacked_profiles(delta, phi_coords, rng):
    n = delta.shape[0]
    radii_mpc = np.linspace(PROFILE_RMAX_MPC / PROFILE_NBINS,
                            PROFILE_RMAX_MPC, PROFILE_NBINS)
    radii_cells = radii_mpc / CELL_SIZE

    kx = (2 * np.pi * np.fft.fftfreq(n)).astype(np.float32)[:, None, None]
    ky = (2 * np.pi * np.fft.fftfreq(n)).astype(np.float32)[None, :, None]
    kz = (2 * np.pi * np.fft.rfftfreq(n)).astype(np.float32)[None, None, :]
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    fk = sfft.rfftn(delta.astype(np.float32, copy=False), workers=-1)

    n_phi = len(phi_coords)
    pc = np.asarray(phi_coords, dtype=int) % n
    null_idx = rng.integers(0, n, size=(NULL_DRAWS, n_phi, 3))

    sphere_data = np.empty(PROFILE_NBINS)
    sphere_null = np.empty((PROFILE_NBINS, NULL_DRAWS))
    for j, R in enumerate(radii_cells):
        field = sfft.irfftn(fk * _tophat_window(kmag * R), s=delta.shape,
                            workers=-1)
        sphere_data[j] = float(np.mean(
            field[pc[:, 0], pc[:, 1], pc[:, 2]]))
        sphere_null[j] = field[null_idx[..., 0], null_idx[..., 1],
                               null_idx[..., 2]].mean(axis=1)
        del field

    vol = radii_cells ** 3
    def to_shells(spheres):
        shells = np.empty_like(spheres)
        shells[0] = spheres[0]
        dv = vol[1:] - vol[:-1]
        if spheres.ndim == 2:
            shells[1:] = (vol[1:, None] * spheres[1:]
                          - vol[:-1, None] * spheres[:-1]) / dv[:, None]
        else:
            shells[1:] = (vol[1:] * spheres[1:]
                          - vol[:-1] * spheres[:-1]) / dv
        return shells

    shell_data = to_shells(sphere_data)
    shell_null = to_shells(sphere_null)
    lo = np.percentile(shell_null, 2.5, axis=1)
    hi = np.percentile(shell_null, 97.5, axis=1)
    return radii_mpc, shell_data, lo, hi


def significant_radius(radii_mpc, profile, lo):
    below = profile < lo
    if not below[0]:
        return np.nan
    k = 0
    while k + 1 < len(below) and below[k + 1]:
        k += 1
    return float(radii_mpc[k])


def plot_profile(radii, profile, lo, hi, redshift, output_path):
    fig, ax = plt.subplots(figsize=style.FIGSIZE['single'])
    if config.SHOW_NULL_BANDS:
        ax.fill_between(radii, lo, hi, color='0.55', alpha=0.35, lw=0,
                        label='95% random-position null')
    ax.plot(radii, profile, 'o-', lw=2.5, color='darkblue', markersize=4,
            label=r'$\langle\delta(r)\rangle$ around $\phi$ maxima')
    ax.axhline(0.0, color='black', linestyle='--', alpha=0.4, linewidth=1.2)
    ax.set_xlabel('Distance from potential maximum [Mpc]')
    ax.set_ylabel(r'Shell-averaged density contrast $\langle\delta\rangle$')
    ax.grid(alpha=0.25, linestyle=':', linewidth=1.0)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    save_plot_to_multiple_paths(fig, output_path, dpi=DPI,
                                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _phi_contour_levels(phi_slice, num=12):
    lo, hi = np.percentile(phi_slice, [1, 99])
    if not np.isfinite(hi - lo) or hi <= lo:
        return np.linspace(float(np.min(phi_slice)),
                           float(np.max(phi_slice)), num)
    return np.linspace(lo, hi, num)


def plot_max_slices(delta, phi, phi_coords, phi_values, void_coords,
                    snapshot, redshift, output_dir):
    n = delta.shape[0]
    phi_coords = np.asarray(phi_coords, dtype=int) % n
    void_coords = np.asarray(void_coords, dtype=int)
    if len(phi_coords) == 0:
        return
    top = int(np.argmax(phi_values))
    extent = [0, BOX_SIZE, 0, BOX_SIZE]
    xs = np.linspace(0, BOX_SIZE, n)

    def _markers(ax, vx, vy, cx, cy):
        ax.scatter(vx, vy, s=14, c='yellow', marker='o', edgecolors='black',
                   linewidths=0.9, alpha=0.95, zorder=5)
        ax.scatter([cx], [cy], marker='*', s=280, c='white',
                   edgecolors='black', linewidths=1.4, zorder=6)

    for i, c in enumerate(phi_coords):
        dsl = delta[:, :, c[2]]
        psl = phi[:, :, c[2]]
        dnorm = style.field_norm('delta', dsl)
        pnorm = style.field_norm('potential', psl)

        rel_z = (void_coords[:, 2] - c[2] + n // 2) % n - n // 2
        near = np.abs(rel_z) <= SLICE_TOLERANCE_CELLS
        vx = void_coords[near, 0] * CELL_SIZE
        vy = void_coords[near, 1] * CELL_SIZE
        cx, cy = c[0] * CELL_SIZE, c[1] * CELL_SIZE

        fig, (axd, axp) = plt.subplots(1, 2, figsize=(14, 6.6))

        imd = axd.imshow(dsl.T, origin='lower', cmap=style.CMAP['delta'],
                         norm=dnorm, extent=extent)
        axd.contour(xs, xs, dsl.T,
                    levels=style.delta_contour_levels(dsl, dnorm),
                    colors='black', alpha=0.6, linewidths=0.5)
        _markers(axd, vx, vy, cx, cy)
        style.set_title(axd, 'Density field')
        axd.set_xlabel(f'X [{AXIS_UNITS}]')
        axd.set_ylabel(f'Y [{AXIS_UNITS}]')
        cbd = fig.colorbar(imd, ax=axd, fraction=0.046, pad=0.04,
                           ticks=style.norm_ticks(dnorm))
        cbd.set_label(r'Density contrast $\delta$')
        axd.set_aspect('equal')
        axd.set_xlim(0, BOX_SIZE)
        axd.set_ylim(0, BOX_SIZE)

        imp = axp.imshow(psl.T, origin='lower', cmap=style.CMAP['potential'],
                         norm=pnorm, extent=extent)
        axp.contour(xs, xs, psl.T, levels=_phi_contour_levels(psl),
                    colors='black', alpha=0.5, linewidths=0.5)
        _markers(axp, vx, vy, cx, cy)
        style.set_title(axp, 'Gravitational potential')
        axp.set_xlabel(f'X [{AXIS_UNITS}]')
        axp.set_ylabel(f'Y [{AXIS_UNITS}]')
        cbp = fig.colorbar(imp, ax=axp, fraction=0.046, pad=0.04)
        cbp.set_label(r'Gravitational potential $\phi$')
        axp.set_aspect('equal')
        axp.set_xlim(0, BOX_SIZE)
        axp.set_ylim(0, BOX_SIZE)

        style.set_suptitle(
            fig, f'Potential maximum {i + 1} (z={redshift:.2f})')
        plt.tight_layout()

        stem = f'phi_max_slice_snap{snapshot}_z{redshift:.2f}_m{i:02d}.png'
        save_plot_to_multiple_paths(fig, output_dir / stem, dpi=DPI,
                                    bbox_inches='tight', facecolor='white')
        if i == top:
            save_plot_to_multiple_paths(
                fig, output_dir / f'phi_top_hill_snap{snapshot}'
                                  f'_z{redshift:.2f}.png',
                dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)


def _redshift_axis(ax):
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 3, 6, 11, 21])
    ax.set_xticklabels(['0', '1', '2', '5', '10', '20'])
    ax.invert_xaxis()
    ax.set_xlabel('Redshift $z$')


def plot_trend(zs, values, null_lo, null_hi, ylabel, fname, ref_line=None):
    fig, ax = plt.subplots(figsize=style.FIGSIZE['single'])
    x = 1.0 + np.asarray(zs)
    if config.SHOW_NULL_BANDS:
        ax.fill_between(x, null_lo, null_hi, color='0.55', alpha=0.35, lw=0,
                        label='95% random-position null')
    ax.plot(x, values, 'o-', lw=2.5, color='steelblue', label='Measured')
    if ref_line is not None:
        ax.axhline(ref_line, color='black', linestyle='--', alpha=0.4,
                   linewidth=1.2)
    _redshift_axis(ax)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle=':', linewidth=1.0)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    save_plot_to_multiple_paths(fig, Path(OUTPUT_DIR) / fname, dpi=DPI,
                                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_profile_evolution(profile_rows):
    rows = [r for r in profile_rows
            if r['snapshot'] in config.PANEL_SNAPSHOTS]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=style.FIGSIZE['single'])
    for r in sorted(rows, key=lambda q: -q['z']):
        line, = ax.plot(r['radii'], r['profile'], 'o-', lw=2,
                        markersize=3.5, label=f"z = {r['z']:.2f}")
        if config.SHOW_NULL_BANDS:
            ax.fill_between(r['radii'], r['lo'], r['hi'],
                            color=line.get_color(), alpha=0.10, lw=0)
    ax.axhline(0.0, color='black', linestyle='--', alpha=0.4, linewidth=1.2)
    ax.set_xlabel('Distance from potential maximum [Mpc]')
    ax.set_ylabel(r'Shell-averaged $\langle\delta\rangle$')
    ax.grid(alpha=0.25, linestyle=':', linewidth=1.0)
    ax.legend(framealpha=0.9,
              title='Shaded: 95% null' if config.SHOW_NULL_BANDS else None)
    plt.tight_layout()
    save_plot_to_multiple_paths(
        fig, Path(OUTPUT_DIR) / 'phi_delta_profile_evolution.png',
        dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def analyze_snapshot(snapshot, redshift, output_root, sim, method):
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")
    output_dir = output_root / f"snapshot_{snapshot}_z{redshift:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        p = pipeline.products(snapshot, redshift, sim=sim, method=method)
        _set_geometry(p.fs)
        cat = p.voids()
        maxima = p.phi_maxima()
        phi_coords = np.asarray(maxima['coords'], dtype=int)
        phi_values = np.asarray(maxima['values'], dtype=float)
        void_coords = np.asarray(cat['coords'], dtype=int)
        n_phi, n_voids = len(phi_coords), len(void_coords)
        print(f"  phi maxima: {n_phi}, voids: {n_voids}")
        if n_phi == 0 or n_voids == 0:
            print("  Empty population; skipping")
            return None

        delta = p.delta_smoothed
        rng = np.random.default_rng(config.CORRELATION_SEED + int(snapshot))

        void_tree = cKDTree(void_coords.astype(float),
                            boxsize=FIELD_RESOLUTION)
        stats = match_statistics(phi_coords.astype(float), void_tree, n_voids)
        null = null_statistics(n_phi, void_tree, rng)
        print(f"  match fraction (r < {MATCH_DIST_CELLS * CELL_SIZE:.1f} "
              f"Mpc): {stats['match_fraction']:.3f} "
              f"(null {null['frac_lo']:.3f}-{null['frac_hi']:.3f})")
        print(f"  median separation: {stats['median_sep_mpc']:.2f} Mpc "
              f"(null {null['med_lo']:.2f}-{null['med_hi']:.2f})")

        pc = phi_coords % FIELD_RESOLUTION
        delta_at_max = delta[pc[:, 0], pc[:, 1], pc[:, 2]]
        frac_neg = float(np.mean(delta_at_max < 0))
        if n_phi >= 3:
            spear_r, spear_p = spearmanr(phi_values, delta_at_max)
        else:
            spear_r, spear_p = np.nan, np.nan
        print(f"  <delta> at maxima: {np.mean(delta_at_max):+.3f} "
              f"(fraction underdense: {frac_neg:.2f}; "
              f"Spearman r = {spear_r:+.2f})")

        print(f"  Stacked profiles ({PROFILE_NBINS} shells to "
              f"{PROFILE_RMAX_MPC:.0f} Mpc, {NULL_DRAWS} null stacks)")
        radii, profile, lo, hi = stacked_profiles(delta, phi_coords, rng)
        r_sig = significant_radius(radii, profile, lo)
        print(f"  profile significant out to r = {r_sig:.1f} Mpc"
              if np.isfinite(r_sig) else "  profile not significant at r->0")
        plot_profile(radii, profile, lo, hi, redshift,
                     output_dir / f'phi_profile_snap{snapshot}'
                                  f'_z{redshift:.2f}.png')

        plot_max_slices(delta, p.phi, phi_coords, phi_values, void_coords,
                        snapshot, redshift, output_dir)
        p.release()

        return {'snapshot': snapshot, 'z': redshift, 'n_phi': n_phi,
                'n_voids': n_voids, 'stats': stats, 'null': null,
                'delta_at_max': float(np.mean(delta_at_max)),
                'frac_neg': frac_neg, 'radii': radii, 'profile': profile,
                'lo': lo, 'hi': hi, 'r_sig': r_sig,
                'spear_r': spear_r, 'spear_p': spear_p}
    except FileNotFoundError:
        print(f"  skipping snapshot {snapshot} (no {method} fields)")
        return None
    except Exception as e:
        print(f"  Error analyzing snapshot {snapshot}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = make_parser("Potential maxima <-> void minima correspondence "
                         "(full-box slices, stacked profiles, redshift trends).")
    parser.set_defaults(snap=None)   # default: run every snapshot in config's table
    args = parser.parse_args()

    print("Potential maxima <-> void minima (full-box slices + profiles)")
    print(f"  sim: {args.sim}; method: {args.method}")
    print(f"  match radius: {MATCH_DIST_CELLS:g} cells "
          f"({MATCH_DIST_CELLS * CELL_SIZE:.2f} Mpc); "
          f"null draws: {NULL_DRAWS}")
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    items = config.snapshot_items()
    if args.snap is not None:
        want = f"{args.snap:03d}"
        items = [(s, z) for s, z in items if s == want]
        if not items:
            print(f"snapshot {want} not in config.SNAPSHOT_TO_REDSHIFT; nothing to do")
            return

    rows = []
    for snapshot, redshift in items:
        r = analyze_snapshot(snapshot, redshift, output_root,
                             sim=args.sim, method=args.method)
        if r is not None:
            rows.append(r)
    if not rows:
        print("No results; aborting")
        return

    zs = [r['z'] for r in rows]
    plot_trend(zs, [r['stats']['match_fraction'] for r in rows],
               [r['null']['frac_lo'] for r in rows],
               [r['null']['frac_hi'] for r in rows],
               'Match fraction of φ maxima',
               'phi_delta_match_fraction_evolution.png')
    plot_trend(zs, [r['stats']['median_sep_mpc'] for r in rows],
               [r['null']['med_lo'] for r in rows],
               [r['null']['med_hi'] for r in rows],
               'Median nearest-neighbour separation [Mpc]',
               'phi_delta_separation_evolution.png')
    plot_profile_evolution(rows)

    header = (f"{'z':>7} {'n_phi':>6} {'n_void':>7} {'frac':>6} "
              f"{'null95%':>15} {'med_sep':>8} {'null95%':>15} "
              f"{'<d@max>':>8} {'f(d<0)':>7} {'r_sig':>6} "
              f"{'spear_r':>8} {'p':>9}")
    lines = [f"phi-void correspondence v3 "
             f"(match radius {MATCH_DIST_CELLS * CELL_SIZE:.2f} Mpc, "
             f"{NULL_DRAWS} null draws; r_sig = radius out to which the "
             f"stacked profile stays below its null band; spear_r = "
             f"Spearman rank corr. of phi height vs delta at the maxima, "
             f"all maxima)",
             header, '-' * len(header)]
    for r in rows:
        s, nl = r['stats'], r['null']
        lines.append(
            f"{r['z']:7.2f} {r['n_phi']:6d} {r['n_voids']:7d} "
            f"{s['match_fraction']:6.3f} "
            f"{nl['frac_lo']:7.3f}-{nl['frac_hi']:7.3f} "
            f"{s['median_sep_mpc']:8.2f} "
            f"{nl['med_lo']:7.2f}-{nl['med_hi']:7.2f} "
            f"{r['delta_at_max']:+8.3f} {r['frac_neg']:7.2f} "
            f"{r['r_sig']:6.1f} {r['spear_r']:8.3f} {r['spear_p']:9.2e}")
    table = output_root / 'phi_delta_stats.txt'
    table.write_text('\n'.join(lines) + '\n')
    print(f"\nTable written: {table}")
    print(f"Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
