"""Shear magnitude and triaxiality slice maps + per-snapshot statistics.

Method-, snapshot- and simulation-agnostic via the shared dtfelib.FieldSet layer:

    python3 plot/plot_shear_triaxial.py                    # TNG50-4-Dark, full series, auto method
    python3 plot/plot_shear_triaxial.py --method dtfe
    python3 plot/plot_shear_triaxial.py --snap 99          # single snapshot only
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from dtfelib import fields as dtfe
from dtfelib.figures import save_plot_to_multiple_paths
import config
from dtfelib import figures as style
from dtfelib import make_parser
from dtfelib.io import FieldSet

style.apply()

OUTPUT_DIR = config.figures_path('shear')

SMOOTHING_SIGMA = config.SMOOTHING_SIGMA_CELLS
MAPS_SIGMA = config.SHEAR_MAPS_SIGMA

DPI = config.DPI

SNAPSHOT_TO_REDSHIFT = config.SNAPSHOT_TO_REDSHIFT

SLICE_PLANES = config.SLICE_PLANES


def smooth_components(field, sigma):
    if sigma <= 0:
        return field
    out = np.empty_like(field)
    for c in range(field.shape[-1]):
        out[..., c] = dtfe.smooth_field(field[..., c], sigma=sigma)
    return out


def shear_tensor_slice_from_5comp(velShear_field, slice_dim, index=None):
    comp = [dtfe.extract_2d_slice(velShear_field[..., c], slice_dim, index)
            for c in range(5)]
    Sxx, Sxy, Sxz, Syy, Syz = comp
    Szz = -(Sxx + Syy)
    sh = Sxx.shape
    S = np.zeros(sh + (3, 3), dtype=velShear_field.dtype)
    S[..., 0, 0] = Sxx
    S[..., 0, 1] = S[..., 1, 0] = Sxy
    S[..., 0, 2] = S[..., 2, 0] = Sxz
    S[..., 1, 1] = Syy
    S[..., 1, 2] = S[..., 2, 1] = Syz
    S[..., 2, 2] = Szz
    return S


def shear_tensor_slice_from_grad9(velGrad_field, slice_dim, index=None):
    comp = [dtfe.extract_2d_slice(velGrad_field[..., c], slice_dim, index)
            for c in range(9)]
    sh = comp[0].shape
    G = np.zeros(sh + (3, 3), dtype=velGrad_field.dtype)
    for c in range(9):
        G[..., c // 3, c % 3] = comp[c]
    S = 0.5 * (G + np.swapaxes(G, -2, -1))
    tr = np.trace(S, axis1=-2, axis2=-1) / 3.0
    for d in range(3):
        S[..., d, d] -= tr
    return S


def triaxiality_from_eigenvalues(evals):
    s1, s2, s3 = evals[..., 2], evals[..., 1], evals[..., 0]
    den = s1 - s3
    T = np.zeros_like(den)
    ok = np.abs(den) > 1e-12
    T[ok] = (s1[ok] - s2[ok]) / den[ok]
    return T


def visualize_shear_magnitude(shear_slice, slice_dim, redshift, box_size, save_path, norm):
    extent = [0, box_size, 0, box_size]
    fig, ax = plt.subplots(figsize=style.FIGSIZE['single'])
    plot_data = np.clip(shear_slice.T, norm.vmin, None)
    im = ax.imshow(plot_data, origin='lower', cmap=style.CMAP['shear'],
                   norm=norm, extent=extent)

    plane_info = SLICE_PLANES[slice_dim]
    style.set_title(ax, f'Shear Magnitude (z={redshift:.2f})')
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [Mpc]")
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [Mpc]")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Shear Magnitude [km/s/Mpc]')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    save_plot_to_multiple_paths(fig, save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def visualize_triaxiality(T_slice, slice_dim, redshift, box_size, save_path):
    extent = [0, box_size, 0, box_size]
    fig, ax = plt.subplots(figsize=style.FIGSIZE['single'])
    im = ax.imshow(T_slice.T, origin='lower', cmap=style.CMAP['triaxiality'],
                   norm=style.norm_bounded(field='triaxiality'), extent=extent)

    plane_info = SLICE_PLANES[slice_dim]
    style.set_title(ax, f'Triaxiality (z={redshift:.2f})')
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [Mpc]")
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [Mpc]")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Triaxiality')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    save_plot_to_multiple_paths(fig, save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def compute_snapshot_slices(fs):
    if fs.has('gradient'):
        print("  Loading 9-component velocity gradient field")
        field = fs.load('gradient')
        tensor_slice = shear_tensor_slice_from_grad9
        source = '9-component gradient'
    elif fs.has('shear'):
        print("  Loading 5-component velocity shear field")
        field = fs.load('shear')
        tensor_slice = shear_tensor_slice_from_5comp
        source = '5-component shear'
    else:
        print("  No velocity shear/gradient fields found")
        return None

    def plane_products(smoothed_field):
        planes = {}
        for slice_dim in (0, 1, 2):
            S = tensor_slice(smoothed_field, slice_dim)
            magnitude = np.sqrt(np.sum(S ** 2, axis=(-2, -1)))
            evals = np.linalg.eigvalsh(
                S.reshape(-1, 3, 3)).reshape(S.shape[:2] + (3,))
            planes[slice_dim] = {
                'magnitude': magnitude,
                'triaxiality': triaxiality_from_eigenvalues(evals),
            }
        return planes

    out = {'source': source}

    print(f"  Smoothing components for statistics "
          f"(sigma = {SMOOTHING_SIGMA} cells)")
    out['stats_planes'] = plane_products(smooth_components(field, SMOOTHING_SIGMA))

    if MAPS_SIGMA == SMOOTHING_SIGMA:
        out['map_planes'] = out['stats_planes']
    else:
        print(f"  Smoothing components for maps (sigma = {MAPS_SIGMA} cells)")
        out['map_planes'] = plane_products(smooth_components(field, MAPS_SIGMA))
    return out


def main():
    parser = make_parser("Shear magnitude and triaxiality slice maps + statistics.")
    # For this series script --snap restricts the run to ONE snapshot; the default
    # (no --snap) keeps the original behavior of processing the whole redshift series.
    snap_action = parser._option_string_actions['--snap']
    snap_action.default = None
    snap_action.help = "process only this snapshot (default: full snapshot series)"
    args = parser.parse_args()

    if args.snap is not None:
        snapshots = [f"{args.snap:03d}"]
    else:
        snapshots = list(SNAPSHOT_TO_REDSHIFT)

    print("Starting shear analysis")
    print(f"Simulation: {args.sim} | method: {args.method}")
    print(f"Processing {len(snapshots)} snapshots")

    results = {}
    for snapshot in snapshots:
        z_cfg = SNAPSHOT_TO_REDSHIFT.get(snapshot)
        z_txt = f" (z={z_cfg:.2f})" if z_cfg is not None else ""
        print(f"\nProcessing snapshot {snapshot}{z_txt}")
        try:
            fs = FieldSet(args.data_root / args.sim / f"snapdir_{snapshot}",
                          method=args.method, averaged=not args.raw)
            r = compute_snapshot_slices(fs)
        except FileNotFoundError:
            print(f"  skipping snapshot {snapshot} (no {args.method} fields)")
            continue
        except ValueError as e:
            # e.g. snapdir holds BOTH ps_output.* and output.* fields under --method auto
            print(f"  Error: {e}")
            continue
        except Exception as e:
            print(f"  Failed: {e}")
            continue
        if r is not None:
            r['redshift'] = z_cfg if z_cfg is not None else fs.meta.redshift
            r['box_mpc'] = fs.meta.box_mpc
            r['smoothing_mpc'] = SMOOTHING_SIGMA * fs.cell_mpc
            results[snapshot] = r

    if not results:
        print("No snapshots processed")
        return

    all_mag = np.concatenate([r['map_planes'][d]['magnitude'].ravel()
                              for r in results.values() for d in (0, 1, 2)])
    norm = style.norm_positive_log(data=all_mag, field='shear')
    print(f"\nShared shear map scale (sigma = {MAPS_SIGMA} cells): "
          f"[{norm.vmin:.3g}, {norm.vmax:.3g}] km/s/Mpc")

    stats_lines = [
        f"{'z':>7} {'snap':>5} {'mean|S|':>10} {'p99.5|S|':>10} {'max|S|':>10} "
        f"{'<T>':>7} {'std(T)':>7}",
    ]
    stats_lines.append('-' * len(stats_lines[0]))

    for snapshot, r in results.items():
        redshift = r['redshift']
        box_size = r['box_mpc']
        output_path = Path(OUTPUT_DIR) / f'snapshot_{snapshot}_z{redshift:.2f}'

        for slice_dim in (0, 1, 2):
            plane_name = SLICE_PLANES[slice_dim]['name']
            plane_dir = output_path / plane_name
            plane_dir.mkdir(parents=True, exist_ok=True)
            pl = r['map_planes'][slice_dim]
            visualize_shear_magnitude(
                pl['magnitude'], slice_dim, redshift, box_size,
                plane_dir / f'shear_magnitude_{plane_name}_z{redshift:.2f}.png',
                norm)
            visualize_triaxiality(
                pl['triaxiality'], slice_dim, redshift, box_size,
                plane_dir / f'triaxiality_{plane_name}_z{redshift:.2f}.png')

        mag = np.concatenate([r['stats_planes'][d]['magnitude'].ravel()
                              for d in (0, 1, 2)])
        T = np.concatenate([r['stats_planes'][d]['triaxiality'].ravel()
                            for d in (0, 1, 2)])
        stats_lines.append(
            f"{redshift:7.2f} {snapshot:>5} {np.mean(mag):10.1f} "
            f"{np.percentile(mag, 99.5):10.1f} {np.max(mag):10.1f} "
            f"{np.mean(T):7.3f} {np.std(T):7.3f}")

        summary_file = output_path / f'shear_analysis_summary_z{redshift:.2f}.txt'
        with open(summary_file, 'w') as f:
            f.write("Shear Analysis Summary (mid-plane slices)\n")
            f.write(f"Snapshot: {snapshot}, Redshift: {redshift:.2f}\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Data source: {r['source']}\n")
            f.write(f"Statistics smoothing: {SMOOTHING_SIGMA} cells "
                    f"({r['smoothing_mpc']:.2f} Mpc); "
                    f"maps drawn at {MAPS_SIGMA} cells (visualization)\n\n")
            f.write("Shear magnitude [km/s/Mpc]:\n")
            f.write(f"  Mean: {np.mean(mag):.3f}\n")
            f.write(f"  Std: {np.std(mag):.3f}\n")
            f.write(f"  99.5th percentile: {np.percentile(mag, 99.5):.3f}\n")
            f.write(f"  Max: {np.max(mag):.3f}\n\n")
            f.write("Triaxiality:\n")
            f.write(f"  Mean: {np.mean(T):.3f}\n")
            f.write(f"  Std: {np.std(T):.3f}\n")

    stats_file = Path(OUTPUT_DIR) / 'shear_stats.txt'
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.write_text('\n'.join(stats_lines) + '\n')
    print(f"\nShear statistics written: {stats_file}")
    print(f"Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
