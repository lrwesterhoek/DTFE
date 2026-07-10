"""Hessian eigenvalue slice maps + per-snapshot statistics + evolution summary.

Method-, snapshot- and simulation-agnostic via the shared pipeline/FieldSet layer:

    python3 plot/plot_eigenvalues.py                     # TNG50-4-Dark, full series, auto method
    python3 plot/plot_eigenvalues.py --method dtfe
    python3 plot/plot_eigenvalues.py --snap 99           # single snapshot only
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from dtfelib.figures import save_plot_to_multiple_paths
import config
from dtfelib import figures as style
from dtfelib import pipeline
from dtfelib import make_parser

style.apply()

OUTPUT_DIR = config.figures_path('eigenvalues')

AXIS_UNITS = config.AXIS_UNITS

SLICE_PLANES_TO_PLOT = [0, 1, 2]

SNAPSHOT_TO_REDSHIFT = config.SNAPSHOT_TO_REDSHIFT

DPI = config.DPI

SLICE_PLANES = config.SLICE_PLANES


def visualize_eigenvalue_slice(eigenvalue_slice, field_name, slice_dim, box_size, redshift=None,
                               save_path=None, colormap=style.CMAP['divergence']):
    extent = [0, box_size, 0, box_size]

    norm = style.norm_signed_log(
        data=eigenvalue_slice, field='eigenvalue',
        vmax=pipeline.series_vmax('eigenvalue', eigenvalue_slice))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(eigenvalue_slice.T, origin='lower', cmap=colormap,
                    norm=norm, extent=extent)

    plane_info = SLICE_PLANES[slice_dim]
    title = field_name
    if redshift is not None:
        title += f' (z={redshift:.2f})'

    style.set_title(ax, title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]")
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(field_name)

    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        save_plot_to_multiple_paths(fig, save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def process_snapshot(snapshot, redshift, sim, method):

    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")

    snapshot_output_dir = Path(OUTPUT_DIR) / f"snapshot_{snapshot}_z{redshift:.2f}"
    snapshot_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        p = pipeline.products(snapshot, redshift, sim=sim, method=method)
        box_size = p.fs.meta.box_mpc
        eig_slices = p.eigen_slices()
        p.release()

        stats_row = {'snapshot': snapshot, 'z': redshift}
        lines = [f"Eigenvalue statistics (cached slices, all 3 planes)",
                 f"Snapshot {snapshot}, z = {redshift:.2f}", "=" * 50]
        for k in ('lambda1', 'lambda2', 'lambda3'):
            v = np.concatenate([np.ravel(eig_slices[d][k]) for d in eig_slices])
            v = v[np.isfinite(v)]
            p05, p50, p995 = np.percentile(v, [0.5, 50, 99.5])
            abs995 = float(np.percentile(np.abs(v), 99.5))
            stats_row[k] = abs995
            lines.append(f"{k}: p0.5={p05:.3e}  median={p50:.3e}  "
                         f"p99.5={p995:.3e}  |.|p99.5={abs995:.3e}")
        (snapshot_output_dir / f'eigenvalue_stats_z{redshift:.2f}.txt'
         ).write_text('\n'.join(lines) + '\n')

        for slice_dim in SLICE_PLANES_TO_PLOT:
            plane_name = SLICE_PLANES[slice_dim]['name']
            plane_dir = snapshot_output_dir / plane_name

            print(f"  Creating {plane_name} visualizations...")

            es = eig_slices[slice_dim]
            lambda1_slice = es['lambda1']
            lambda2_slice = es['lambda2']
            lambda3_slice = es['lambda3']
            derived_fields = {'trace': es['trace'], 'determinant': es['det']}

            visualize_eigenvalue_slice(
                lambda1_slice, "λ₁ (Smallest Eigenvalue)", slice_dim, box_size, redshift=redshift,
                save_path=plane_dir / f"lambda1_{plane_name}_z{redshift:.2f}.png"
            )

            visualize_eigenvalue_slice(
                lambda2_slice, "λ₂ (Middle Eigenvalue)", slice_dim, box_size, redshift=redshift,
                save_path=plane_dir / f"lambda2_{plane_name}_z{redshift:.2f}.png"
            )

            visualize_eigenvalue_slice(
                lambda3_slice, "λ₃ (Largest Eigenvalue)", slice_dim, box_size, redshift=redshift,
                save_path=plane_dir / f"lambda3_{plane_name}_z{redshift:.2f}.png"
            )

            visualize_eigenvalue_slice(
                derived_fields['trace'], "Trace (∇²δ)", slice_dim, box_size, redshift=redshift,
                save_path=plane_dir / f"trace_{plane_name}_z{redshift:.2f}.png"
            )

            visualize_eigenvalue_slice(
                derived_fields['determinant'], "Determinant", slice_dim, box_size, redshift=redshift,
                save_path=plane_dir / f"determinant_{plane_name}_z{redshift:.2f}.png"
            )

        return stats_row

    except FileNotFoundError:
        print(f"  skipping snapshot {snapshot} (no {method} fields)")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_evolution_summary(rows):
    rows = [r for r in rows if r is not None]
    if len(rows) < 2:
        return
    print("\nCreating evolution summary")
    rows.sort(key=lambda r: -r['z'])
    zp1 = np.array([1.0 + r['z'] for r in rows])

    fig, ax = plt.subplots(figsize=style.FIGSIZE['single'])
    for k, fmt, label in (('lambda1', 'ro-', 'λ₁ (smallest)'),
                          ('lambda2', 'go-', 'λ₂ (middle)'),
                          ('lambda3', 'bo-', 'λ₃ (largest)')):
        ax.plot(zp1, [r[k] for r in rows], fmt, label=label, markersize=4)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 3, 6, 11, 21])
    ax.set_xticklabels(['0', '1', '2', '5', '10', '20'])
    ax.invert_xaxis()
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('Robust eigenvalue amplitude $|\\lambda|_{p99.5}$')
    style.set_title(ax, 'Evolution of Hessian Eigenvalue Amplitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_multiple_paths(
        fig, Path(OUTPUT_DIR) / 'eigenvalue_amplitude_evolution.png',
        dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    header = (f"{'z':>7} {'snap':>5} {'|l1|p99.5':>12} {'|l2|p99.5':>12} "
              f"{'|l3|p99.5':>12}")
    lines = ["Robust eigenvalue amplitudes per snapshot "
             "(|.| 99.5th percentile over the three cached slices)",
             header, '-' * len(header)]
    for r in rows:
        lines.append(f"{r['z']:7.2f} {r['snapshot']:>5} "
                     f"{r['lambda1']:12.3e} {r['lambda2']:12.3e} "
                     f"{r['lambda3']:12.3e}")
    (Path(OUTPUT_DIR) / 'eigenvalue_stats.txt'
     ).write_text('\n'.join(lines) + '\n')
    print(f"  Table written: {Path(OUTPUT_DIR) / 'eigenvalue_stats.txt'}")

def main():

    parser = make_parser("Hessian eigenvalue slice maps, statistics and evolution summary.")
    # For this series script --snap restricts the run to ONE snapshot; the default
    # (no --snap) keeps the original behavior of processing the whole redshift series.
    snap_action = parser._option_string_actions['--snap']
    snap_action.default = None
    snap_action.help = "process only this snapshot (default: full snapshot series)"
    args = parser.parse_args()

    if args.snap is not None:
        snap_id = f"{args.snap:03d}"
        z = config.get_redshift(snap_id)
        if z is None:
            try:
                z = pipeline.products(snap_id, sim=args.sim, method=args.method).redshift
            except FileNotFoundError:
                z = None
        if z is None:
            print(f"skipping snapshot {snap_id} (no {args.method} fields)")
            return
        snapshot_items = [(snap_id, z)]
    else:
        snapshot_items = list(SNAPSHOT_TO_REDSHIFT.items())

    print("Starting eigenvalue analysis")
    print(f"Simulation: {args.sim} | method: {args.method}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing {len(snapshot_items)} snapshots")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    processed_snapshots = 0
    failed_snapshots = []
    rows = []

    for snapshot, redshift in snapshot_items:
        row = process_snapshot(snapshot, redshift, args.sim, args.method)
        if row is not None:
            rows.append(row)
            processed_snapshots += 1
        else:
            failed_snapshots.append((snapshot, redshift))

    if processed_snapshots > 1:
        try:
            create_evolution_summary(rows)
        except Exception as e:
            print(f"\nError creating evolution summary: {e}")

    print(f"\nCompleted: {processed_snapshots}/{len(snapshot_items)} snapshots processed")

    if failed_snapshots:
        print(f"Failed snapshots: {', '.join(f'{s} (z={z:.2f})' for s, z in failed_snapshots)}")

    print(f"Output saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
