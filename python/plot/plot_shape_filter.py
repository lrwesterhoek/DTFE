"""Void ellipsoid shape analysis (axis ratios, BBKS e/p) per snapshot.

    python3 plot/plot_shape_filter.py                     # all snapshots, TNG50-4-Dark, auto method
    python3 plot/plot_shape_filter.py --snap 99 --method dtfe
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import minimum_filter
from dtfelib import fields as dtfe
from dtfelib.figures import save_plot_to_multiple_paths
import config
from dtfelib import figures as style
from dtfelib import pipeline
from dtfelib import make_parser

style.apply()

OUTPUT_DIR = config.figures_path('void_shapes')

SMOOTHING_SIGMA = config.SMOOTHING_SIGMA_CELLS
FOOTPRINT_SIZE = config.FOOTPRINT_SIZE


DPI = config.DPI

SNAPSHOT_TO_REDSHIFT = config.SNAPSHOT_TO_REDSHIFT


def plot_axis_ratios(axis_ratios, density_values=None, redshift=None, save_path=None):
    if len(axis_ratios) == 0:
        print("  No data to plot for axis ratios")
        return
        
    b_a = axis_ratios[:, 0]
    c_a = axis_ratios[:, 1]
    
    plt.figure(figsize=(8, 7))

    if density_values is not None:
        dnorm = style.norm_signed_log(field='delta',
                                      vmax=pipeline.series_vmax('delta', density_values))
        sc = plt.scatter(b_a, c_a, c=density_values, cmap=style.CMAP['delta'],
                         norm=dnorm, s=30, alpha=0.7)
        cbar = plt.colorbar(sc)
        cbar.set_label('Density Contrast δ')
    else:
        plt.scatter(b_a, c_a, color='blue', s=30, alpha=0.7)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)

    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.xlabel('b/a (Intermediate-to-Major Axis Ratio)')
    plt.ylabel('c/a (Minor-to-Major Axis Ratio)')

    title = 'Shape Distribution of Void Ellipsoids'
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    style.set_title(plt.gca(), title, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_plot_to_multiple_paths(plt.gcf(), save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_bbks_parameters(bbks_params, density_values=None, redshift=None, save_path=None):
    if len(bbks_params) == 0:
        print("  No data to plot for BBKS parameters")
        return
        
    e_values = bbks_params[:, 0]
    p_values = bbks_params[:, 1]
    
    plt.figure(figsize=(8, 7))

    if density_values is not None:
        dnorm = style.norm_signed_log(field='delta',
                                      vmax=pipeline.series_vmax('delta', density_values))
        sc = plt.scatter(e_values, p_values, c=density_values, cmap=style.CMAP['delta'],
                         norm=dnorm, s=30, alpha=0.7)
        cbar = plt.colorbar(sc)
        cbar.set_label('Density Contrast δ')
    else:
        plt.scatter(e_values, p_values, color='blue', s=30, alpha=0.7)

    plt.axhline(y=0, color='k', linestyle='--', alpha=0.4)

    plt.xlim(0, 0.55)
    plt.ylim(-0.55, 0.55)
    plt.xlabel('e (Ellipticity)')
    plt.ylabel('p (Prolateness)')

    title = 'BBKS Shape Parameters of Void Ellipsoids'
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    style.set_title(plt.gca(), title, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_plot_to_multiple_paths(plt.gcf(), save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def _plot_histogram(values, xlabel, color, bins, save_path, redshift=None,
                    mark_zero=False):
    fig, ax = plt.subplots(figsize=style.FIGSIZE['single'])
    ax.hist(values, bins=bins, color=color, alpha=0.7, edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')

    mean_v = np.mean(values)
    ax.axvline(x=mean_v, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_v:.3f}')
    if mark_zero:
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.legend()

    title = f'Distribution of {xlabel}'
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    style.set_title(ax, title)

    plt.tight_layout()
    save_plot_to_multiple_paths(fig, save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_bbks_histograms(bbks_params, redshift=None, save_path=None):
    if len(bbks_params) == 0:
        print("  No data to plot for BBKS histograms")
        return

    valid = ~np.isnan(bbks_params).any(axis=1)
    e_values = bbks_params[valid, 0]
    p_values = bbks_params[valid, 1]
    stem = Path(save_path)

    if len(e_values) > 0:
        e_bins = np.linspace(0.0, 0.55, 23)
        _plot_histogram(e_values, 'e (Ellipticity)', 'blue', e_bins,
                        stem.with_name(stem.stem + '_e.png'), redshift)

    if len(p_values) > 0:
        p_bins = np.linspace(-0.55, 0.55, 23)
        _plot_histogram(p_values, 'p (Prolateness)', 'green', p_bins,
                        stem.with_name(stem.stem + '_p.png'), redshift,
                        mark_zero=True)

def process_snapshot(snapshot, redshift, sim=None, method=None):
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")

    output_path = Path(OUTPUT_DIR) / f'snapshot_{snapshot}_z{redshift:.2f}'
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        p = pipeline.products(snapshot, redshift, sim=sim, method=method)
        cat = p.voids()
        grid_n = p.fs.grid_n
        p.release()

        if len(cat['coords']) == 0:
            print("No valid minima found")
            return

        print(f"Found {len(cat['coords'])} valid minima")

        minima_indices = cat['coords']
        axis_ratios = cat['axis_ratios']
        bbks_params = cat['bbks_params']
        density_values = cat['delta_values']

        ax_v = axis_ratios[~np.isnan(axis_ratios).any(axis=1)]
        bbks_v = bbks_params[~np.isnan(bbks_params).any(axis=1)]

        if len(ax_v) > 0 and len(bbks_v) > 0:
            print(f"  Mean b/a: {np.mean(ax_v[:, 0]):.3f} ± {np.std(ax_v[:, 0]):.3f}")
            print(f"  Mean c/a: {np.mean(ax_v[:, 1]):.3f} ± {np.std(ax_v[:, 1]):.3f}")
            print(f"  Mean ellipticity: {np.mean(bbks_v[:, 0]):.3f} ± {np.std(bbks_v[:, 0]):.3f}")
            print(f"  Mean prolateness: {np.mean(bbks_v[:, 1]):.3f} ± {np.std(bbks_v[:, 1]):.3f}")

            prolate_pct = 100 * np.sum(bbks_v[:, 1] > 0.05) / len(bbks_v)
            oblate_pct = 100 * np.sum(bbks_v[:, 1] < -0.05) / len(bbks_v)
            print(f"  Prolate: {prolate_pct:.1f}%, Oblate: {oblate_pct:.1f}%")
        
        print("Creating visualizations")
        plot_axis_ratios(axis_ratios, density_values, redshift=redshift, 
                        save_path=output_path / f'void_axis_ratios_z{redshift:.2f}.png')
        
        plot_bbks_parameters(bbks_params, density_values, redshift=redshift, 
                           save_path=output_path / f'void_bbks_params_z{redshift:.2f}.png')
        
        plot_bbks_histograms(bbks_params, redshift=redshift, 
                           save_path=output_path / f'void_bbks_histograms_z{redshift:.2f}.png')
        
        summary_file = output_path / f'void_shape_analysis_summary_z{redshift:.2f}.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Void Shape Analysis Summary\n")
            f.write(f"Snapshot: {snapshot}, Redshift: {redshift:.2f}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  Grid resolution: {grid_n}³\n")
            f.write(f"  Footprint size: {FOOTPRINT_SIZE}\n")
            f.write(f"  Smoothing sigma: {SMOOTHING_SIGMA}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total minima found: {len(minima_indices)}\n")
            f.write(f"  Valid BBKS params (e/p population): {len(bbks_v)}\n")
            f.write(f"  Valid axis ratios (all lambda > 0): {len(ax_v)}\n")

            if len(ax_v) > 0 and len(bbks_v) > 0:
                f.write(f"  Mean axis ratios:\n")
                f.write(f"    b/a: {np.mean(ax_v[:, 0]):.3f} ± {np.std(ax_v[:, 0]):.3f}\n")
                f.write(f"    c/a: {np.mean(ax_v[:, 1]):.3f} ± {np.std(ax_v[:, 1]):.3f}\n")
                f.write(f"  BBKS parameters:\n")
                f.write(f"    Mean ellipticity (e): {np.mean(bbks_v[:, 0]):.3f} ± {np.std(bbks_v[:, 0]):.3f}\n")
                f.write(f"    Mean prolateness (p): {np.mean(bbks_v[:, 1]):.3f} ± {np.std(bbks_v[:, 1]):.3f}\n")
                prolate_pct = 100 * np.sum(bbks_v[:, 1] > 0.05) / len(bbks_v)
                oblate_pct = 100 * np.sum(bbks_v[:, 1] < -0.05) / len(bbks_v)
                f.write(f"    Prolate shapes (p > 0.05): {prolate_pct:.1f}%\n")
                f.write(f"    Oblate shapes (p < -0.05): {oblate_pct:.1f}%\n")
                f.write(f"  Density statistics:\n")
                f.write(f"    Mean density at voids: {np.mean(density_values):.3f}\n")
                f.write(f"    Std density at voids: {np.std(density_values):.3f}\n")
                f.write(f"    Min density at voids: {np.min(density_values):.3f}\n")
                f.write(f"    Max density at voids: {np.max(density_values):.3f}\n")
        
        print(f"Completed snapshot {snapshot}")

    except FileNotFoundError:
        print(f"skipping snapshot {snapshot} (no {method or 'auto'} fields)")
    except Exception as e:
        print(f"Error processing snapshot {snapshot}: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    parser = make_parser("Void ellipsoid shape analysis (axis ratios, BBKS e/p) per snapshot. "
                         "By default iterates ALL snapshots in config.SNAPSHOT_TO_REDSHIFT; "
                         "pass --snap to restrict to one.")
    parser.set_defaults(snap=None)   # series script: default = all snapshots, not just 099
    for _a in parser._actions:
        if _a.dest == "snap":
            _a.help = "process only this snapshot (default: all snapshots in config)"
    args = parser.parse_args()

    if args.snap is not None:
        snap_id = f"{int(args.snap):03d}"
        redshift = config.get_redshift(snap_id)
        if redshift is None:
            print(f"Unknown snapshot {snap_id} (not in config.SNAPSHOT_TO_REDSHIFT)")
            return
        items = [(snap_id, redshift)]
    else:
        items = list(SNAPSHOT_TO_REDSHIFT.items())

    print(f"Starting void shape analysis")
    print(f"Processing {len(items)} snapshots")

    processed_count = 0
    failed_snapshots = []

    for snapshot, redshift in items:
        try:
            process_snapshot(snapshot, redshift, sim=args.sim, method=args.method)
            processed_count += 1
        except Exception as e:
            print(f"Failed to process snapshot {snapshot} (z={redshift:.2f}): {str(e)}")
            failed_snapshots.append((snapshot, redshift))

    print(f"\nAnalysis complete")
    print(f"Successfully processed: {processed_count}/{len(items)} snapshots")
    
    if failed_snapshots:
        print(f"Failed snapshots: {len(failed_snapshots)}")
        for snapshot, redshift in failed_snapshots:
            print(f"  Snapshot {snapshot} (z={redshift:.2f})")
    
    print(f"\nOutput saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
