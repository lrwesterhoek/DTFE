import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.animation as animation
from scipy.ndimage import minimum_filter
from dtfelib import fields as dtfe
from dtfelib.figures import save_plot_to_multiple_paths
import config
from dtfelib import figures as style
from dtfelib import pipeline
from dtfelib import make_parser

style.apply()

OUTPUT_DIR = config.figures_path('void_analysis')

FOOTPRINT_SIZE = config.FOOTPRINT_SIZE
SMOOTHING_SIGMA = config.SMOOTHING_SIGMA_CELLS

SNAPSHOT_TO_REDSHIFT = config.SNAPSHOT_TO_REDSHIFT

DPI = config.DPI

def get_redshift(snapshot_id):
    return SNAPSHOT_TO_REDSHIFT.get(snapshot_id)

def get_distinct_colors(n_colors):
    if n_colors <= 10:
        cmap = mpl.colormaps['tab10']
        colors = [cmap(i) for i in range(n_colors)]
    elif n_colors <= 12:
        cmap = mpl.colormaps['Set3']
        colors = [cmap(i) for i in range(n_colors)]
    else:
        cmap1 = mpl.colormaps['tab10']
        cmap2 = mpl.colormaps['Set3']
        colors = []
        for i in range(n_colors):
            if i < 10:
                colors.append(cmap1(i))
            else:
                colors.append(cmap2((i-10) % 12))
    return colors

def filter_valid_data(data_array, parameter_type='axis_ratios'):
    if len(data_array) == 0:
        return np.array([]).reshape(0, 2), np.array([])
        
    if parameter_type == 'axis_ratios':
        valid_nan = ~(np.isnan(data_array[:, 0]) | np.isnan(data_array[:, 1]))
        
        if np.any(valid_nan):
            valid_data = data_array[valid_nan]
            valid_physical = np.logical_and(
                np.logical_and(valid_data[:, 0] >= 0, valid_data[:, 0] <= 1),
                np.logical_and(valid_data[:, 1] >= 0, valid_data[:, 1] <= 1)
            )
            valid_physical = np.logical_and(valid_physical, valid_data[:, 1] <= valid_data[:, 0])
            
            if np.any(valid_physical):
                return valid_data[valid_physical], valid_physical
        
        return np.array([]).reshape(0, 2), np.array([])
            
    elif parameter_type == 'bbks_params':
        valid_nan = ~(np.isnan(data_array[:, 0]) | np.isnan(data_array[:, 1]))
        
        if np.any(valid_nan):
            valid_data = data_array[valid_nan]
            e_values = valid_data[:, 0]
            p_values = valid_data[:, 1]
            
            valid_physical = np.logical_and(e_values >= 0, 
                                          np.logical_and(p_values >= -e_values, p_values <= e_values))
            
            if np.any(valid_physical):
                return valid_data[valid_physical], valid_physical
        
        return np.array([]).reshape(0, 2), np.array([])
    
    return data_array, np.ones(len(data_array), dtype=bool)

def bbks_theoretical_distribution(e, p, gamma=0.6):
    in_range = np.logical_and(p >= -e, p <= e)
    distribution = np.zeros_like(e, dtype=float)
    distribution[in_range] = e[in_range] * (e[in_range]**2 - p[in_range]**2) * \
                            np.exp(-5/(2*(1-gamma**2)) * ((3*e[in_range])**2 + p[in_range]**2))
    return distribution

def process_snapshot(snapshot, redshift, sim, method):

    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")

    snapshot_output_dir = Path(OUTPUT_DIR) / f"snapshot_{snapshot}_z{redshift:.2f}"
    snapshot_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        p = pipeline.products(snapshot, redshift, sim=sim, method=method)
        cat = p.voids()
        p.release()

        if len(cat['coords']) == 0:
            print(f"  Warning: No valid minima found after filtering")
            return None

        print(f"  {len(cat['coords'])} voids passing the {config.VOID_EIGENVALUE_CRITERION!r} eigenvalue criterion")

        minima_indices = cat['coords']
        eigenvalues = cat['eigenvalues']
        shape_params = {'axis_ratios': cat['axis_ratios'],
                        'bbks_params': cat['bbks_params']}
        density_values = cat['delta_values']
        
        summary_file = snapshot_output_dir / f'void_shape_summary_z{redshift:.2f}.txt'
        
        axis_ratios = shape_params['axis_ratios']
        bbks_params = shape_params['bbks_params']
        
        valid_axis_mask = ~(np.isnan(axis_ratios[:, 0]) | np.isnan(axis_ratios[:, 1]))
        valid_bbks_mask = ~(np.isnan(bbks_params[:, 0]) | np.isnan(bbks_params[:, 1]))
        
        with open(summary_file, 'w') as f:
            f.write(f"Void Shape Analysis Summary\n")
            f.write(f"Snapshot: {snapshot}, Redshift: {redshift:.2f}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  Grid resolution: {p.fs.grid_n}³\n")
            f.write(f"  Footprint size: {FOOTPRINT_SIZE}\n")
            f.write(f"  Smoothing sigma: {SMOOTHING_SIGMA} cells ({config.SMOOTHING_SIGMA_MPC:.2f} Mpc)\n")
            f.write(f"  Eigenvalue criterion: {config.VOID_EIGENVALUE_CRITERION}\n")
            f.write(f"  Method: FFT-based Hessian\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total minima found: {len(minima_indices)}\n")
            f.write(f"  Valid axis ratios: {np.sum(valid_axis_mask)}\n")
            f.write(f"  Valid BBKS params: {np.sum(valid_bbks_mask)}\n")
            
            if np.any(valid_axis_mask):
                valid_axis_ratios = axis_ratios[valid_axis_mask]
                f.write(f"  Mean axis ratios:\n")
                f.write(f"    b/a: {np.mean(valid_axis_ratios[:, 0]):.3f} ± {np.std(valid_axis_ratios[:, 0]):.3f}\n")
                f.write(f"    c/a: {np.mean(valid_axis_ratios[:, 1]):.3f} ± {np.std(valid_axis_ratios[:, 1]):.3f}\n")
            
            if np.any(valid_bbks_mask):
                valid_bbks_params = bbks_params[valid_bbks_mask]
                f.write(f"  BBKS parameters:\n")
                f.write(f"    Mean ellipticity (e): {np.mean(valid_bbks_params[:, 0]):.3f} ± {np.std(valid_bbks_params[:, 0]):.3f}\n")
                f.write(f"    Mean prolateness (p): {np.mean(valid_bbks_params[:, 1]):.3f} ± {np.std(valid_bbks_params[:, 1]):.3f}\n")
                prolate_percentage = 100 * np.sum(valid_bbks_params[:, 1] > 0.05) / len(valid_bbks_params)
                oblate_percentage = 100 * np.sum(valid_bbks_params[:, 1] < -0.05) / len(valid_bbks_params)
                f.write(f"    Prolate shapes (p > 0.05): {prolate_percentage:.1f}%\n")
                f.write(f"    Oblate shapes (p < -0.05): {oblate_percentage:.1f}%\n")
        
        return {
            'minima_indices': minima_indices,
            'minima_eigenvalues': eigenvalues,
            'axis_ratios': shape_params['axis_ratios'],
            'bbks_params': shape_params['bbks_params'],
            'density_values': density_values
        }

    except FileNotFoundError:
        label = method if method != 'auto' else 'DTFE/PS-DTFE'
        print(f"  skipping snapshot {snapshot} (no {label} fields)")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def plot_contour_axis_ratios(snapshots_data, snapshot_ids, output_path=None):
    plt.figure(figsize=(12, 9))
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    
    redshift_data = []
    for i, snapshot_id in enumerate(snapshot_ids):
        z = get_redshift(snapshot_id)
        if z is not None and snapshots_data[i] is not None:
            redshift_data.append((z, snapshot_id, snapshots_data[i]))
    
    redshift_data.sort(reverse=True)
    
    if not redshift_data:
        return
    
    colors = get_distinct_colors(len(redshift_data))
    
    x_grid = np.linspace(0, 1.0, 200)
    y_grid = np.linspace(0, 1.0, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    mask = Y <= X
    
    legend_elements = []
    for i, (redshift, snapshot_id, snapshot_data) in enumerate(redshift_data):
        axis_ratios = snapshot_data['axis_ratios']
        valid_data, _ = filter_valid_data(axis_ratios, 'axis_ratios')

        print(f"  z={redshift:.2f}: {len(valid_data)} valid axis ratio points")

        if len(valid_data) < 10:
            print(f"    Skipping - too few points (need at least 10)")
            continue

        b_a = valid_data[:, 0]
        c_a = valid_data[:, 1]

        print(f"    b/a range: [{np.min(b_a):.3f}, {np.max(b_a):.3f}]")
        print(f"    c/a range: [{np.min(c_a):.3f}, {np.max(c_a):.3f}]")

        try:
            kde = gaussian_kde([b_a, c_a], bw_method='scott')
            positions = np.vstack([X.ravel(), Y.ravel()])
            valid_indices = np.where(mask.ravel())[0]

            Z = np.zeros(X.size)
            Z[valid_indices] = kde(positions[:, valid_indices])
            Z = Z.reshape(X.shape)

            Z_masked = np.ma.array(Z, mask=~mask)
            z_min, z_max = Z_masked.min(), Z_masked.max()

            print(f"    KDE z_min={z_min:.6f}, z_max={z_max:.6f}")

            if z_max > z_min:
                Z_norm = (Z_masked - z_min) / (z_max - z_min)
                positive_Z = Z_norm.compressed()
                positive_Z = positive_Z[positive_Z > 0]

                if len(positive_Z) > 0:
                    color = colors[i]
                    levels = np.percentile(positive_Z, [25, 50, 68, 85, 95])
                    print(f"    Plotting {len(levels)} contour levels")
                    plt.contour(X, Y, Z_norm, levels=levels, colors=[color], linewidths=2.0, alpha=0.8)
                    plt.contourf(X, Y, Z_norm, levels=levels, colors=[color], alpha=0.15)

                    mean_b_a = np.mean(b_a)
                    mean_c_a = np.mean(c_a)
                    print(f"    Plotting mean at ({mean_b_a:.3f}, {mean_c_a:.3f})")
                    plt.scatter(mean_b_a, mean_c_a, color=color, s=120, edgecolor='black',
                              marker='o', linewidths=2.5, zorder=10)
                else:
                    print(f"    Warning: No positive Z values, only plotting mean")
                    color = colors[i]
                    mean_b_a = np.mean(b_a)
                    mean_c_a = np.mean(c_a)
                    plt.scatter(mean_b_a, mean_c_a, color=color, s=120, edgecolor='black',
                              marker='o', linewidths=2.5, zorder=10)
            else:
                print(f"    Warning: z_max <= z_min, skipping")

            legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=2,
                                          label=f'z = {redshift:.2f}'))
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel('b/a (Intermediate-to-Major Axis Ratio)')
    plt.ylabel('c/a (Minor-to-Major Axis Ratio)')
    style.set_title(plt.gca(), 'Evolution of Void Shape Distribution', fontsize=14)
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='lower right', title='Redshift')
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        save_plot_to_multiple_paths(plt.gcf(), output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_contour_bbks(snapshots_data, snapshot_ids, output_path=None):
    plt.figure(figsize=(12, 9))
    
    redshift_data = []
    for i, snapshot_id in enumerate(snapshot_ids):
        z = get_redshift(snapshot_id)
        if z is not None and snapshots_data[i] is not None:
            redshift_data.append((z, snapshot_id, snapshots_data[i]))
    
    redshift_data.sort(reverse=True)
    
    if not redshift_data:
        return
    
    colors = get_distinct_colors(len(redshift_data))
    
    max_e = 0.5
    for _, _, snapshot_data in redshift_data:
        bbks_params = snapshot_data['bbks_params']
        valid_data, _ = filter_valid_data(bbks_params, 'bbks_params')
        if len(valid_data) > 0:
            max_e = max(max_e, np.max(valid_data[:, 0]))
    
    max_e *= 1.1
    
    plt.xlim(0, max_e)
    plt.ylim(-max_e, max_e)
    
    e_grid = np.linspace(0, max_e, 200)
    p_grid = np.linspace(-max_e, max_e, 400)
    E, P = np.meshgrid(e_grid, p_grid)
    triangle_mask = np.logical_and(P >= -E, P <= E)
    
    gamma = 0.6
    Z_theory = bbks_theoretical_distribution(E, P, gamma)
    Z_theory_masked = np.ma.array(Z_theory, mask=~triangle_mask)
    
    if Z_theory_masked.max() > Z_theory_masked.min():
        Z_theory_norm = (Z_theory_masked - Z_theory_masked.min()) / (Z_theory_masked.max() - Z_theory_masked.min())
    else:
        Z_theory_norm = Z_theory_masked
    
    theory_levels = np.percentile(Z_theory_norm.compressed()[Z_theory_norm.compressed() > 0], 
                                 [50, 68, 85, 95, 99])
    plt.contour(E, P, Z_theory_norm, levels=theory_levels, colors=['gray'], 
               linestyles=['--'], alpha=0.5, linewidths=1.0)
    
    legend_elements = []
    for i, (redshift, snapshot_id, snapshot_data) in enumerate(redshift_data):
        bbks_params = snapshot_data['bbks_params']
        valid_data, _ = filter_valid_data(bbks_params, 'bbks_params')

        print(f"  z={redshift:.2f}: {len(valid_data)} valid BBKS points")

        if len(valid_data) < 10:
            print(f"    Skipping - too few points (need at least 10)")
            continue

        e_values = valid_data[:, 0]
        p_values = valid_data[:, 1]

        print(f"    e range: [{np.min(e_values):.3f}, {np.max(e_values):.3f}]")
        print(f"    p range: [{np.min(p_values):.3f}, {np.max(p_values):.3f}]")

        prolate_percentage = 100 * np.sum(p_values > 0.05) / len(p_values)
        oblate_percentage = 100 * np.sum(p_values < -0.05) / len(p_values)

        try:
            kde = gaussian_kde([e_values, p_values], bw_method='scott')
            positions = np.vstack([E.ravel(), P.ravel()])
            valid_indices = np.where(triangle_mask.ravel())[0]

            Z = np.zeros(E.size)
            Z[valid_indices] = kde(positions[:, valid_indices])
            Z = Z.reshape(E.shape)

            Z_masked = np.ma.array(Z, mask=~triangle_mask)
            z_min, z_max = Z_masked.min(), Z_masked.max()

            print(f"    KDE z_min={z_min:.6f}, z_max={z_max:.6f}")

            if z_max > z_min:
                Z_norm = (Z_masked - z_min) / (z_max - z_min)
                positive_Z = Z_norm.compressed()
                positive_Z = positive_Z[positive_Z > 0]

                if len(positive_Z) > 0:
                    color = colors[i]
                    levels = np.percentile(positive_Z, [25, 50, 68, 85, 95])
                    print(f"    Plotting {len(levels)} contour levels")
                    plt.contour(E, P, Z_norm, levels=levels, colors=[color], linewidths=2.0, alpha=0.8)
                    plt.contourf(E, P, Z_norm, levels=levels, colors=[color], alpha=0.15)

                    mean_e = np.mean(e_values)
                    mean_p = np.mean(p_values)
                    print(f"    Plotting mean at ({mean_e:.3f}, {mean_p:.3f})")
                    plt.scatter(mean_e, mean_p, color=color, s=120, edgecolor='black',
                              marker='o', linewidths=2.5, zorder=10)
                else:
                    print(f"    Warning: No positive Z values, only plotting mean")
                    color = colors[i]
                    mean_e = np.mean(e_values)
                    mean_p = np.mean(p_values)
                    plt.scatter(mean_e, mean_p, color=color, s=120, edgecolor='black',
                              marker='o', linewidths=2.5, zorder=10)
            else:
                print(f"    Warning: z_max <= z_min, skipping")

            legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=2,
                                        label=f'z = {redshift:.2f}: Pro={prolate_percentage:.1f}%, Obl={oblate_percentage:.1f}%'))
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    plt.plot([0, max_e], [0, max_e], 'k-', alpha=0.3)
    plt.plot([0, max_e], [0, -max_e], 'k-', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.4)
    
    plt.xlim(0, max_e)
    plt.ylim(-max_e, max_e)
    plt.xlabel('e (Ellipticity)')
    plt.ylabel('p (Prolateness)')
    style.set_title(plt.gca(), 'Evolution of Void Shape Distribution', fontsize=14)
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='lower right', title='Redshift')
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        save_plot_to_multiple_paths(plt.gcf(), output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compute_snapshot_statistics(snapshots_data, snapshot_ids):
    rows = []
    for i, snapshot_id in enumerate(snapshot_ids):
        data = snapshots_data[i]
        z = get_redshift(snapshot_id)
        if data is None or z is None:
            continue
        ax, _ = filter_valid_data(data['axis_ratios'], 'axis_ratios')
        bbks, _ = filter_valid_data(data['bbks_params'], 'bbks_params')
        if len(ax) == 0 or len(bbks) == 0:
            continue
        e, p = bbks[:, 0], bbks[:, 1]
        rows.append({
            'z': z, 'snapshot': snapshot_id, 'n_voids': len(bbks),
            'mean_ca': np.mean(ax[:, 1]), 'mean_ba': np.mean(ax[:, 0]),
            'std_ca': np.std(ax[:, 1]), 'std_ba': np.std(ax[:, 0]),
            'mean_e': np.mean(e), 'mean_p': np.mean(p),
            'std_e': np.std(e), 'std_p': np.std(p),
            'prolate_frac': 100 * np.sum(p > 0.05) / len(p),
        })
    rows.sort(key=lambda r: -r['z'])
    return rows


def write_summary_table(rows, output_path):
    header = (f"{'z':>7} {'snap':>5} {'N_voids':>8} {'<c/a>':>7} {'s(c/a)':>7} "
              f"{'<b/a>':>7} {'s(b/a)':>7} {'<e>':>7} {'s(e)':>7} "
              f"{'<p>':>7} {'s(p)':>7} {'prolate%':>9}")
    lines = [header, '-' * len(header)]
    for r in rows:
        lines.append(f"{r['z']:7.2f} {r['snapshot']:>5} {r['n_voids']:8d} "
                     f"{r['mean_ca']:7.3f} {r['std_ca']:7.3f} "
                     f"{r['mean_ba']:7.3f} {r['std_ba']:7.3f} "
                     f"{r['mean_e']:7.3f} {r['std_e']:7.3f} "
                     f"{r['mean_p']:7.3f} {r['std_p']:7.3f} {r['prolate_frac']:9.1f}")
    Path(output_path).write_text('\n'.join(lines) + '\n')
    print(f"  Summary table written: {output_path}")


def plot_distribution_evolution(snapshots_data, snapshot_ids, column, xlabel, output_path):
    plt.figure(figsize=style.FIGSIZE['large'])

    redshift_data = []
    for i, snapshot_id in enumerate(snapshot_ids):
        z = get_redshift(snapshot_id)
        if z is not None and snapshots_data[i] is not None:
            redshift_data.append((z, snapshot_id, snapshots_data[i]))
    redshift_data.sort(reverse=True)
    if not redshift_data:
        return

    colors = get_distinct_colors(len(redshift_data))
    x = np.linspace(0, 1, 400)
    for i, (z, snapshot_id, data) in enumerate(redshift_data):
        valid, _ = filter_valid_data(data['axis_ratios'], 'axis_ratios')
        if len(valid) < 10:
            continue
        vals = valid[:, column]
        try:
            kde = gaussian_kde(vals, bw_method='scott')
            plt.plot(x, kde(x), color=colors[i], lw=2, label=f'z = {z:.2f}')
        except Exception as err:
            print(f"    KDE failed for z={z:.2f}: {err}")
            continue

    plt.xlabel(xlabel)
    plt.ylabel('Probability density')
    plt.xlim(0, 1)
    plt.legend(title='Redshift', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_plot_to_multiple_paths(plt.gcf(), output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _evolution_panel(zp1, series, ylabel, output_path):
    fig, ax = plt.subplots(figsize=style.FIGSIZE['single'])
    for y, spread, sem, kwargs in series:
        y = np.asarray(y, dtype=float)
        if sem is not None:
            container = ax.errorbar(zp1, y, yerr=np.asarray(sem, dtype=float),
                                    lw=2, capsize=2, **kwargs)
            line = container[0]
        else:
            line, = ax.plot(zp1, y, lw=2, **kwargs)
        if spread is not None:
            s = np.asarray(spread, dtype=float)
            ax.fill_between(zp1, y - s, y + s, color=line.get_color(),
                            alpha=0.15, lw=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('$1 + z$')
    if any(kw.get('label') for *_, kw in series):
        ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    ax.invert_xaxis()
    plt.tight_layout()
    save_plot_to_multiple_paths(fig, output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_shape_evolution_summary(rows, output_dir):
    if len(rows) < 2:
        return
    zp1 = np.array([1 + r['z'] for r in rows])
    output_dir = Path(output_dir)

    def sem(std_key):
        return [r[std_key] / np.sqrt(r['n_voids']) for r in rows]

    _evolution_panel(zp1, [
        ([r['mean_e'] for r in rows], [r['std_e'] for r in rows],
         sem('std_e'), dict(marker='o', label='ellipticity $e$')),
        ([r['mean_p'] for r in rows], [r['std_p'] for r in rows],
         sem('std_p'), dict(marker='s', label='prolateness $p$')),
    ], 'BBKS shape parameters', output_dir / 'evolution_bbks_parameters.png')

    _evolution_panel(zp1, [
        ([r['mean_ca'] for r in rows], [r['std_ca'] for r in rows],
         sem('std_ca'), dict(marker='o', label='$c/a$')),
        ([r['mean_ba'] for r in rows], [r['std_ba'] for r in rows],
         sem('std_ba'), dict(marker='s', label='$b/a$')),
    ], 'Axis ratios', output_dir / 'evolution_axis_ratios.png')

    prolate_sem = [100.0 * np.sqrt((r['prolate_frac'] / 100.0)
                                   * (1.0 - r['prolate_frac'] / 100.0)
                                   / r['n_voids']) for r in rows]
    _evolution_panel(zp1, [
        ([r['prolate_frac'] for r in rows], None, prolate_sem,
         dict(marker='o', color='tab:red')),
    ], 'Prolate fraction ($p > 0.05$) [%]', output_dir / 'evolution_prolate_fraction.png')

    _evolution_panel(zp1, [
        ([r['n_voids'] for r in rows], None, None,
         dict(marker='o', color='tab:gray')),
    ], 'Number of voids', output_dir / 'evolution_void_count.png')


def create_evolution_summary(snapshots_data, snapshot_ids):

    print("\nCreating evolution summary plots")

    valid_data = []
    for i, snapshot_id in enumerate(snapshot_ids):
        if snapshots_data[i] is not None:
            redshift = get_redshift(snapshot_id)
            if redshift is not None:
                valid_data.append((redshift, snapshot_id, snapshots_data[i]))

    if len(valid_data) < 2:
        print("  Need at least 2 valid snapshots")
        return

    valid_data.sort(reverse=True)
    valid_snapshot_ids = [item[1] for item in valid_data]
    valid_snapshots_data = [item[2] for item in valid_data]

    axis_ratios_contour_path = Path(OUTPUT_DIR) / 'axis_ratios_contour.png'
    bbks_contour_path = Path(OUTPUT_DIR) / 'bbks_contour.png'

    plot_contour_axis_ratios(valid_snapshots_data, valid_snapshot_ids, output_path=axis_ratios_contour_path)
    plot_contour_bbks(valid_snapshots_data, valid_snapshot_ids, output_path=bbks_contour_path)

    plot_distribution_evolution(valid_snapshots_data, valid_snapshot_ids, 1,
                                'c/a (Minor-to-Major Axis Ratio)',
                                Path(OUTPUT_DIR) / 'c_a_distribution.png')
    plot_distribution_evolution(valid_snapshots_data, valid_snapshot_ids, 0,
                                'b/a (Intermediate-to-Major Axis Ratio)',
                                Path(OUTPUT_DIR) / 'b_a_distribution.png')

    rows = compute_snapshot_statistics(valid_snapshots_data, valid_snapshot_ids)
    plot_shape_evolution_summary(rows, Path(OUTPUT_DIR))
    write_summary_table(rows, Path(OUTPUT_DIR) / 'void_shape_table.txt')

def main():

    parser = make_parser("Void shape (axis-ratio / BBKS) contour evolution across snapshots.")
    args = parser.parse_args()

    print("Starting void shape evolution analysis")
    print(f"Data directory: {args.data_root / args.sim}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing {len(SNAPSHOT_TO_REDSHIFT)} snapshots")
    print(f"Method: FFT-based Hessian computation")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    processed_snapshots = 0
    failed_snapshots = []
    snapshots_data = []
    snapshot_ids = []

    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        result = process_snapshot(snapshot, redshift, args.sim, args.method)
        if result is not None:
            snapshots_data.append(result)
            snapshot_ids.append(snapshot)
            processed_snapshots += 1
        else:
            snapshots_data.append(None)
            snapshot_ids.append(snapshot)
            failed_snapshots.append((snapshot, redshift))
    
    if processed_snapshots > 1:
        try:
            create_evolution_summary(snapshots_data, snapshot_ids)
        except Exception as e:
            print(f"\nError creating evolution summary: {e}")
    
    print(f"\nCompleted: {processed_snapshots}/{len(SNAPSHOT_TO_REDSHIFT)} snapshots processed")
    
    if failed_snapshots:
        print(f"Failed snapshots: {', '.join(f'{s} (z={z:.2f})' for s, z in failed_snapshots)}")
    
    print(f"Output saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
