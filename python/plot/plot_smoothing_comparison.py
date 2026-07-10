"""Void-catalog robustness against the Gaussian smoothing scale.

Method-, snapshot- and simulation-agnostic via the shared pipeline/FieldSet layer:

    python3 plot/plot_smoothing_comparison.py                  # TNG50-4-Dark, panel snapshots, auto method
    python3 plot/plot_smoothing_comparison.py --method dtfe
    python3 plot/plot_smoothing_comparison.py --snap 99        # single snapshot only
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dtfelib import fields as dtfe
from dtfelib.figures import save_plot_to_multiple_paths
import config
from dtfelib import figures as style
from dtfelib import pipeline
from dtfelib import FieldSet, make_parser

style.apply()

OUTPUT_DIR = config.figures_path('smoothing_comparison')

SIGMAS = list(config.SMOOTHING_COMPARISON_SIGMAS)
PANEL_SNAPSHOTS = list(config.PANEL_SNAPSHOTS)

DPI = config.DPI


def catalog_statistics(cat):
    ax = np.asarray(cat['axis_ratios'])
    bbks = np.asarray(cat['bbks_params'])

    valid_ax = ~np.isnan(ax).any(axis=1)
    valid_bbks = ~np.isnan(bbks).any(axis=1)
    if valid_bbks.any():
        e, p = bbks[valid_bbks, 0], bbks[valid_bbks, 1]
        valid_phys = (e >= 0) & (p >= -e) & (p <= e)
        e, p = e[valid_phys], p[valid_phys]
    else:
        e = p = np.array([])

    return {
        'n_voids': int(len(cat['coords'])),
        'mean_ca': float(np.mean(ax[valid_ax, 1])) if valid_ax.any() else np.nan,
        'mean_e': float(np.mean(e)) if e.size else np.nan,
        'mean_p': float(np.mean(p)) if p.size else np.nan,
        'prolate_frac': float(100 * np.sum(p > 0.05) / len(p)) if p.size else np.nan,
    }


def catalog_for(snapshot, sigma, args):
    """Void catalog for one (snapshot, sigma). Returns (catalog, fieldset)."""
    if sigma == config.SMOOTHING_SIGMA_CELLS:
        prod = pipeline.products(snapshot, sim=args.sim, method=args.method)
        return prod.voids(), prod.fs

    fs = FieldSet(args.data_root / args.sim / f"snapdir_{snapshot}",
                  method=args.method)
    density = fs.density(units='mean')
    delta_s = dtfe.smooth_field(dtfe.calculate_density_contrast(density),
                                sigma=sigma)
    hessian = pipeline.hessian_from_field(delta_s)
    return pipeline.build_void_catalog(delta_s, hessian), fs


def plot_statistic(rows, redshifts, snapshots, cell_mpc, key, ylabel, fname):
    fig, ax = plt.subplots(figsize=style.FIGSIZE['single'])

    for snap in snapshots:
        pts = [(s * cell_mpc, rows[(snap, s)][key])
               for s in SIGMAS if (snap, s) in rows]
        if not pts:
            continue
        z = redshifts[snap]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker='o', lw=2, label=f'z = {z:.2f}')

    ax.axvline(config.SMOOTHING_SIGMA_CELLS * cell_mpc,
               color='gray', ls='--', alpha=0.6,
               label='analysis scale')
    ax.set_xlabel(r'Smoothing scale $\sigma$ [Mpc]')
    ax.set_ylabel(ylabel)
    if key == 'n_voids':
        ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    save_plot_to_multiple_paths(fig, Path(OUTPUT_DIR) / fname,
                                dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = make_parser("Void-catalog robustness against the Gaussian smoothing scale.")
    # For this series script --snap restricts the run to ONE snapshot; the default
    # (no --snap) keeps the original behavior of processing the panel snapshots.
    snap_action = parser._option_string_actions['--snap']
    snap_action.default = None
    snap_action.help = "process only this snapshot (default: panel snapshot series)"
    args = parser.parse_args()

    snapshots = [f"{args.snap:03d}"] if args.snap is not None else PANEL_SNAPSHOTS

    print("Smoothing-robustness comparison")
    print(f"  simulation: {args.sim} | method: {args.method}")
    print(f"  sigmas: {SIGMAS} cells")
    print(f"  snapshots: {snapshots}")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    rows = {}
    redshifts = {}
    cell_mpc = None
    for snap in snapshots:
        z_cfg = config.get_redshift(snap)
        z_cfg = float('nan') if z_cfg is None else z_cfg
        for sigma in SIGMAS:
            print(f"  snapshot {snap} (z={z_cfg:.2f}), sigma={sigma:g} cells")
            try:
                cat, fs = catalog_for(snap, sigma, args)
            except FileNotFoundError:
                print(f"    skipping snapshot {snap} (no {args.method} fields)")
                continue
            except Exception as e:
                print(f"    FAILED: {e}")
                continue
            cell_mpc = fs.cell_mpc
            redshifts[snap] = fs.meta.redshift
            rows[(snap, sigma)] = catalog_statistics(cat)
            print(f"    N={rows[(snap, sigma)]['n_voids']}, "
                  f"<e>={rows[(snap, sigma)]['mean_e']:.3f}, "
                  f"<p>={rows[(snap, sigma)]['mean_p']:.3f}")

    if not rows:
        print("No results; aborting")
        return

    plot_statistic(rows, redshifts, snapshots, cell_mpc, 'n_voids',
                   'Number of voids', 'smoothing_n_voids.png')
    plot_statistic(rows, redshifts, snapshots, cell_mpc, 'mean_e',
                   r'Mean ellipticity $\langle e \rangle$', 'smoothing_mean_e.png')
    plot_statistic(rows, redshifts, snapshots, cell_mpc, 'mean_p',
                   r'Mean prolateness $\langle p \rangle$', 'smoothing_mean_p.png')
    plot_statistic(rows, redshifts, snapshots, cell_mpc, 'prolate_frac',
                   'Prolate fraction ($p > 0.05$) [%]', 'smoothing_prolate_fraction.png')
    plot_statistic(rows, redshifts, snapshots, cell_mpc, 'mean_ca',
                   r'Mean axis ratio $\langle c/a \rangle$', 'smoothing_mean_ca.png')

    lines = [f"{'z':>7} {'snap':>5} {'sigma':>6} {'N_voids':>8} "
             f"{'<c/a>':>7} {'<e>':>7} {'<p>':>7} {'prolate%':>9}"]
    lines.append('-' * len(lines[0]))
    for snap in snapshots:
        z = redshifts.get(snap)
        for sigma in SIGMAS:
            r = rows.get((snap, sigma))
            if r is None:
                continue
            lines.append(f"{z:7.2f} {snap:>5} {sigma:6g} {r['n_voids']:8d} "
                         f"{r['mean_ca']:7.3f} {r['mean_e']:7.3f} "
                         f"{r['mean_p']:7.3f} {r['prolate_frac']:9.1f}")
    table = Path(OUTPUT_DIR) / 'smoothing_comparison.txt'
    table.write_text('\n'.join(lines) + '\n')
    print(f"\nTable written: {table}")
    print(f"Output saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
