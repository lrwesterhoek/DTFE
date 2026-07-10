import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path
#!/usr/bin/env python3

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from dtfelib.figures import save_plot_to_multiple_paths
from dtfelib import make_parser
import config
from dtfelib import figures as style
from dtfelib import pipeline

style.apply()

OUTPUT_DIR = config.figures_path('correlation')

BOX_SIZE = config.BOX_SIZE
DPI = config.DPI

N_EVAL = config.CORRELATION_N_EVAL
MIN_PAIRS = config.CORRELATION_MIN_PAIRS
N_PERM = config.CORRELATION_N_PERMUTATIONS
BAND_LO, BAND_HI = config.CORRELATION_BAND_PERCENTILES
PERM_BLOCK = 100
SMOOTH_SIGMA_EVAL = 2.0


def void_sample(cat):
    bbks = np.asarray(cat['bbks_params'], dtype=float)
    e, p = bbks[:, 0], bbks[:, 1]
    valid = np.isfinite(e) & np.isfinite(p) & (e >= 0) & (np.abs(p) <= e)

    pos = np.asarray(cat['positions_mpc'], dtype=float)[valid]
    evals = np.asarray(cat['eigenvalues'], dtype=float)[valid]
    evecs = np.asarray(cat['eigenvectors'], dtype=float)[valid]
    e, p = e[valid], p[valid]

    idx = np.argmin(np.abs(evals), axis=1)
    major = evecs[np.arange(len(evecs)), :, idx]
    major /= np.linalg.norm(major, axis=1, keepdims=True)

    return pos, e, p, major


def pair_geometry(pos):
    n = len(pos)
    ii, jj = np.triu_indices(n, k=1)
    d = pos[jj] - pos[ii]
    d -= BOX_SIZE * np.round(d / BOX_SIZE)
    return ii, jj, np.linalg.norm(d, axis=1)


def build_windows(dist):
    r_max = config.correlation_max_sep()
    r_eval = np.linspace(0.0, r_max, N_EVAL)
    w0 = r_max / 8.0
    members, counts = [], []
    for r in r_eval:
        w = w0 * (1.0 + r / r_max)
        sel = np.where((dist >= max(0.0, r - w / 2)) &
                       (dist <= min(r_max, r + w / 2)))[0]
        members.append(sel)
        counts.append(len(sel))
    return r_eval, members, np.array(counts)


def pearson_curves(mark_rows, ii, jj, members):
    B = mark_rows.shape[0]
    out = np.full((B, len(members)), np.nan)
    for k, idx in enumerate(members):
        if len(idx) < MIN_PAIRS:
            continue
        mi = mark_rows[:, ii[idx]]
        mj = mark_rows[:, jj[idx]]
        mi = mi - mi.mean(axis=1, keepdims=True)
        mj = mj - mj.mean(axis=1, keepdims=True)
        si = mi.std(axis=1)
        sj = mj.std(axis=1)
        cov = (mi * mj).mean(axis=1)
        ok = (si > 1e-10) & (sj > 1e-10)
        out[ok, k] = np.clip(cov[ok] / (si[ok] * sj[ok]), -1.0, 1.0)
    return out


def mcf_curves(mark_rows, ii, jj, members):
    B = mark_rows.shape[0]
    out = np.full((B, len(members)), np.nan)
    mbar2 = mark_rows.mean(axis=1) ** 2
    for k, idx in enumerate(members):
        if len(idx) < MIN_PAIRS:
            continue
        prod = (mark_rows[:, ii[idx]] * mark_rows[:, jj[idx]]).mean(axis=1)
        out[:, k] = prod / mbar2
    return out


def alignment_curves(perm_rows, A, ii, jj, members):
    B = perm_rows.shape[0]
    out = np.full((B, len(members)), np.nan)
    for k, idx in enumerate(members):
        if len(idx) < MIN_PAIRS:
            continue
        a = A[perm_rows[:, ii[idx]], perm_rows[:, jj[idx]]]
        out[:, k] = 2.0 * a.mean(axis=1) - 1.0
    return out


def smooth_rows(rows):
    valid = np.isfinite(rows)
    vals = np.where(valid, rows, 0.0)
    num = gaussian_filter1d(vals, SMOOTH_SIGMA_EVAL, axis=1, mode='nearest')
    den = gaussian_filter1d(valid.astype(float), SMOOTH_SIGMA_EVAL,
                            axis=1, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        sm = num / den
    sm[~valid] = np.nan
    return sm


def curve_with_band(estimator, data_rows, perm_source, ii, jj, members):
    curve = smooth_rows(estimator(data_rows, ii, jj, members))[0]

    perm_curves = []
    for b0 in range(0, N_PERM, PERM_BLOCK):
        block = perm_source(b0, min(b0 + PERM_BLOCK, N_PERM))
        perm_curves.append(smooth_rows(estimator(block, ii, jj, members)))
    perm_curves = np.vstack(perm_curves)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        lo = np.nanpercentile(perm_curves, BAND_LO, axis=0)
        hi = np.nanpercentile(perm_curves, BAND_HI, axis=0)
    return curve, lo, hi


def significant_ranges(r, curve, lo, hi):
    sig = np.isfinite(curve) & np.isfinite(lo) & ((curve > hi) | (curve < lo))
    ranges, start = [], None
    for k, s in enumerate(sig):
        if s and start is None:
            start = r[k]
        elif not s and start is not None:
            ranges.append((start, r[k - 1]))
            start = None
    if start is not None:
        ranges.append((start, r[-1]))
    return ranges


def plot_panel(r, curve, lo, hi, color, label, ylabel, null_level, ylim,
               output_path):
    fig, ax = plt.subplots(figsize=style.FIGSIZE['single'])

    if config.SHOW_NULL_BANDS:
        ax.fill_between(r, lo, hi, color='0.55', alpha=0.35, lw=0,
                        label='95\\% null band' if plt.rcParams['text.usetex']
                        else '95% null band')
    ax.plot(r, curve, '-', color=color, linewidth=2.5, label=label, alpha=0.9)

    valid = np.where(np.isfinite(curve))[0]
    if len(valid):
        step = max(1, len(valid) // 15)
        pts = valid[::step]
        ax.scatter(r[pts], curve[pts], c=color, s=40, marker='o',
                   edgecolor='white', linewidth=1.2, alpha=0.9, zorder=5)

    ax.axhline(y=null_level, color='black', linestyle='--', alpha=0.4,
               linewidth=1.2)
    ax.set_xlabel('Distance [Mpc]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=1.0)
    ax.legend(framealpha=0.9)
    ax.set_xlim(0.0, config.correlation_max_sep())
    ax.set_ylim(*ylim)

    plt.tight_layout()
    save_plot_to_multiple_paths(fig, output_path, dpi=DPI,
                                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def process_snapshot(snapshot, redshift, args):
    p = pipeline.products(snapshot, redshift, sim=args.sim, method=args.method)
    redshift = p.redshift
    if redshift is None:
        print(f"\nskipping snapshot {snapshot} (unknown redshift)")
        return None
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")
    output_dir = Path(OUTPUT_DIR) / f"snapshot_{snapshot}_z{redshift:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cat = p.voids()
        p.release()

        pos, e_marks, p_marks, major = void_sample(cat)
        n = len(pos)
        if n < 10:
            print(f"  Warning: only {n} valid voids; skipping")
            return False
        print(f"  Sample: {n} voids (full catalog, BBKS validity cuts)")

        ii, jj, dist = pair_geometry(pos)
        r_eval, members, counts = build_windows(dist)
        n_masked = int(np.sum(counts < MIN_PAIRS))
        print(f"  Pairs: {len(dist)}; windows: {N_EVAL} "
              f"({n_masked} masked, < {MIN_PAIRS} pairs)")

        rng = np.random.default_rng(config.CORRELATION_SEED + int(snapshot))
        perm_rows = np.stack([rng.permutation(n) for _ in range(N_PERM)])

        ylim_sym = (-config.CORRELATION_YLIM, config.CORRELATION_YLIM)
        ylim_m = (1.0 - config.CORRELATION_M_YLIM,
                  1.0 + config.CORRELATION_M_YLIM)
        results = {}

        for marks, stem, color, label in (
                (e_marks, 'ellipticity_correlation', 'green',
                 'Ellipticity (BBKS $e$)'),
                (p_marks, 'prolateness_correlation', 'purple',
                 'Prolateness (BBKS $p$)')):
            curve, lo, hi = curve_with_band(
                pearson_curves, marks[None, :],
                lambda a, b: marks[perm_rows[a:b]], ii, jj, members)
            results[stem] = (curve, lo, hi, 0.0)
            plot_panel(r_eval, curve, lo, hi, color, label, r'$\rho(r)$',
                       0.0, ylim_sym,
                       output_dir / f'{stem}_snap{snapshot}_z{redshift:.2f}.png')

        curve, lo, hi = curve_with_band(
            mcf_curves, e_marks[None, :],
            lambda a, b: e_marks[perm_rows[a:b]], ii, jj, members)
        results['marked_M_ellipticity'] = (curve, lo, hi, 1.0)
        plot_panel(r_eval, curve, lo, hi, 'darkorange',
                   'Marked correlation (BBKS $e$)', r'$M(r)$', 1.0, ylim_m,
                   output_dir / f'marked_M_ellipticity_snap{snapshot}'
                                f'_z{redshift:.2f}.png')

        A = np.abs(major @ major.T)
        identity = np.arange(n)[None, :]
        curve, lo, hi = curve_with_band(
            lambda rows, *a: alignment_curves(rows, A, ii, jj, members),
            identity, lambda a, b: perm_rows[a:b], ii, jj, members)
        results['orientation_correlation'] = (curve, lo, hi, 0.0)
        plot_panel(r_eval, curve, lo, hi, 'red', 'Orientation',
                   r'$\xi_{\mathrm{orient}}(r)$', 0.0, ylim_sym,
                   output_dir / f'orientation_correlation_snap{snapshot}'
                                f'_z{redshift:.2f}.png')

        lines = [
            "Marked correlation analysis summary",
            f"Snapshot {snapshot}, z = {redshift:.2f}",
            "=" * 60, "",
            f"Population: full void catalog, BBKS validity cuts",
            f"  Voids: {n}   Pairs: {len(dist)}",
            f"  Mean ellipticity e: {np.mean(e_marks):.3f} "
            f"+/- {np.std(e_marks):.3f}",
            f"  Mean prolateness p: {np.mean(p_marks):.3f} "
            f"+/- {np.std(p_marks):.3f}", "",
            f"Windows: {N_EVAL} over 0 - "
            f"{config.correlation_max_sep():.1f} Mpc",
            f"  Pairs per window: min {counts.min()}, "
            f"median {int(np.median(counts))}, max {counts.max()}",
            f"  Masked windows (< {MIN_PAIRS} pairs): {n_masked}", "",
            f"Null band: central 95% of {N_PERM} mark shuffles "
            f"(seed {config.CORRELATION_SEED})", "",
            "Separations OUTSIDE the null band (significant):",
        ]
        for stem, (curve, lo, hi, _null) in results.items():
            ranges = significant_ranges(r_eval, curve, lo, hi)
            txt = ', '.join(f"{a:.1f}-{b:.1f} Mpc" for a, b in ranges) \
                if ranges else 'none'
            lines.append(f"  {stem:28s}: {txt}")
        (output_dir / f'analysis_summary_snap{snapshot}_z{redshift:.2f}.txt'
         ).write_text('\n'.join(lines) + '\n')

        print(f"  Output saved to: {output_dir}")
        return True

    except FileNotFoundError:
        print(f"  skipping snapshot {snapshot} (no {args.method} fields)")
        return None
    except Exception as e:
        print(f"  Error processing snapshot {snapshot}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = make_parser(
        "Marked correlation analysis (BBKS marks, permutation null bands).")
    for action in parser._actions:
        if action.dest == 'snap':
            action.default = None
            action.help = ("process only this snapshot "
                           "(default: all snapshots in the config table)")
    args = parser.parse_args()

    if args.snap is not None:
        snap_id = f"{args.snap:03d}"
        items = [(snap_id, config.get_redshift(snap_id))]
    else:
        items = config.snapshot_items()

    print("Marked correlation analysis (BBKS marks, permutation null bands)")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing {len(items)} snapshots "
          f"(sim {args.sim}, method {args.method})")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    processed, skipped, failed = 0, 0, []
    for snapshot, redshift in items:
        ok = process_snapshot(snapshot, redshift, args)
        if ok:
            processed += 1
        elif ok is None:
            skipped += 1
        else:
            failed.append((snapshot, redshift))

    print(f"\n{'=' * 60}")
    print(f"Successfully processed: {processed}/"
          f"{len(items)} snapshots ({skipped} skipped)")
    for snapshot, redshift in failed:
        print(f"  FAILED: snapshot {snapshot} (z={redshift:.2f})")
    print(f"Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
