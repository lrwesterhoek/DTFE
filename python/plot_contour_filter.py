"""
Void Shape Analysis Tool

Identifies voids in density fields using local minima detection, computes
Hessian eigenvalues to characterize void shapes, and creates evolution plots
across redshift.
"""

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
from pathlib import Path
from scipy.ndimage import minimum_filter
import dtfe_functions as dtfe
from dtfe_shared import save_plot_to_multiple_paths

# ============================================================================
# Configuration Section
# ============================================================================

BASE_DATA_DIR = "output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/void_analysis"

FIELD_RESOLUTION = 512
FOOTPRINT_SIZE = 11
SMOOTHING_SIGMA = 10.0

# Snapshot to redshift mapping
SNAPSHOT_TO_REDSHIFT = dtfe.SNAPSHOT_TO_REDSHIFT

DPI = 300

# ============================================================================

def get_redshift(snapshot_id):
    """Convert snapshot ID to redshift."""
    return SNAPSHOT_TO_REDSHIFT.get(snapshot_id)

def get_distinct_colors(n_colors):
    """Get n distinct colors for plotting."""
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
    """Filter data to remove NaN values and apply physical constraints."""
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
    """Calculate theoretical BBKS distribution."""
    in_range = np.logical_and(p >= -e, p <= e)
    distribution = np.zeros_like(e, dtype=float)
    distribution[in_range] = e[in_range] * (e[in_range]**2 - p[in_range]**2) * \
                            np.exp(-5/(2*(1-gamma**2)) * ((3*e[in_range])**2 + p[in_range]**2))
    return distribution

def process_snapshot(snapshot, redshift):
    """Process a single snapshot and return shape parameters."""
    
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")
    
    snapshot_dir = Path(BASE_DATA_DIR) / f"snapdir_{snapshot}"
    
    if not snapshot_dir.exists():
        print(f"  Warning: Directory not found")
        return None
    
    density_file = snapshot_dir / 'output.a_den'
    if not density_file.exists():
        print(f"  Warning: Density file not found")
        return None
    
    snapshot_output_dir = Path(OUTPUT_DIR) / f"snapshot_{snapshot}_z{redshift:.2f}"
    snapshot_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)
        
        print(f"  Loading density field")
        density_field = dtfe.load_binary_field(str(density_file), field_shape, num_components=1)
        delta_field = dtfe.calculate_density_contrast(density_field)
        delta_field = dtfe.smooth_field(delta_field, sigma=SMOOTHING_SIGMA)
        
        print(f"  Computing Hessian using FFT")
        hessian_components = dtfe.calculate_hessian_fft(delta_field)
        
        print(f"  Finding minima and computing eigenvalues")
        minima_mask, minima_coords = dtfe.find_local_minima(delta_field, footprint_size=FOOTPRINT_SIZE)
        print(f"  Found {len(minima_coords)} local minima")

        eigenvalues, eigenvectors = dtfe.compute_eigenvalues_at_minima(hessian_components, minima_coords)
        minima_coords, eigenvalues = dtfe.filter_minima_by_eigenvalues(minima_coords, eigenvalues, positive_only=True)

        if len(minima_coords) == 0:
            print(f"  Warning: No valid minima found after filtering")
            return None

        print(f"  Found {len(minima_coords)} voids with positive eigenvalues")

        shape_params = dtfe.calculate_shape_parameters(eigenvalues)
        minima_indices = minima_coords
        density_values = np.array([delta_field[tuple(idx)] for idx in minima_indices])
        
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
            f.write(f"  Grid resolution: {FIELD_RESOLUTION}³\n")
            f.write(f"  Footprint size: {FOOTPRINT_SIZE}\n")
            f.write(f"  Smoothing sigma: {SMOOTHING_SIGMA}\n")
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
                prolate_percentage = 100 * np.sum(valid_bbks_params[:, 1] > 0) / len(valid_bbks_params)
                oblate_percentage = 100 * np.sum(valid_bbks_params[:, 1] < 0) / len(valid_bbks_params)
                f.write(f"    Prolate shapes: {prolate_percentage:.1f}%\n")
                f.write(f"    Oblate shapes: {oblate_percentage:.1f}%\n")
        
        return {
            'minima_indices': minima_indices,
            'minima_eigenvalues': eigenvalues,
            'axis_ratios': shape_params['axis_ratios'],
            'bbks_params': shape_params['bbks_params'],
            'density_values': density_values
        }
        
    except Exception as e:
        print(f"  Error: {e}")
        return None

def plot_contour_axis_ratios(snapshots_data, snapshot_ids, output_path=None):
    """Create contour plot of axis ratio evolution."""
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
    plt.xlabel('b/a (Intermediate-to-Major Axis Ratio)', fontsize=12)
    plt.ylabel('c/a (Minor-to-Major Axis Ratio)', fontsize=12)
    plt.title('Evolution of Void Shape Distribution', fontsize=14)
    
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
    """Create contour plot of BBKS parameter evolution."""
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

        prolate_percentage = 100 * np.sum(p_values > 0) / len(p_values)
        oblate_percentage = 100 * np.sum(p_values < 0) / len(p_values)

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
    plt.xlabel('e (Ellipticity)', fontsize=12)
    plt.ylabel('p (Prolateness)', fontsize=12)
    plt.title('Evolution of Void Shape Distribution', fontsize=14)
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='lower right', title='Redshift')
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        save_plot_to_multiple_paths(plt.gcf(), output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_parameter_distribution(snapshots_data, snapshot_ids, parameter_name, 
                                parameter_label, output_path=None, xlim=None):
    """Create stacked histogram showing parameter evolution across redshifts."""
    
    print(f"\nCreating {parameter_name} distribution plot")
    
    redshift_data = []
    for i, snapshot_id in enumerate(snapshot_ids):
        z = get_redshift(snapshot_id)
        if z is not None and snapshots_data[i] is not None:
            redshift_data.append((z, snapshot_id, snapshots_data[i]))
    
    if not redshift_data:
        print(f"  No valid data for {parameter_name}")
        return
    
    redshift_data.sort(reverse=True)
    colors = get_distinct_colors(len(redshift_data))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    all_values = []
    all_means = []
    valid_redshifts = []
    
    for z, snapshot_id, snapshot_data in redshift_data:
        if parameter_name == 'c/a':
            values = snapshot_data['axis_ratios'][:, 1]
        elif parameter_name == 'b/a':
            values = snapshot_data['axis_ratios'][:, 0]
        elif parameter_name == 'ellipticity':
            values = snapshot_data['bbks_params'][:, 0]
        elif parameter_name == 'prolateness':
            values = snapshot_data['bbks_params'][:, 1]
        else:
            continue
        
        values = values[~np.isnan(values)]
        
        if len(values) > 0:
            all_values.append(values)
            all_means.append(np.mean(values))
            valid_redshifts.append(z)
    
    if len(all_values) == 0:
        print(f"  No valid values found")
        return
    
    if xlim is None:
        all_data = np.concatenate(all_values)
        data_min, data_max = np.percentile(all_data, [1, 99])
        margin = (data_max - data_min) * 0.1
        xlim = (data_min - margin, data_max + margin)
    
    bins = np.linspace(xlim[0], xlim[1], 50)
    
    ax.hist(all_values, bins=bins, stacked=True, 
            color=colors[:len(all_values)], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    legend_elements = []
    for i, (z, mean_val) in enumerate(zip(valid_redshifts, all_means)):
        ax.axvline(mean_val, color=colors[i], linestyle='--', linewidth=2, alpha=0.8)
        legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=3,
                                     label=f'z = {z:.2f}, Mean: {mean_val:.3f}'))
    
    ax.set_xlim(xlim)
    ax.set_xlabel(parameter_label, fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'Evolution of {parameter_label} Distribution', fontsize=14)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        save_plot_to_multiple_paths(fig, output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_all_distribution_plots(snapshots_data, snapshot_ids):
    """Generate all four parameter distribution plots."""
    
    print("\nCreating parameter distribution plots")
    
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
    
    distributions = [
        ('c/a', 'c/a Ratio', (0.0, 1.0)),
        ('b/a', 'b/a Ratio', (0.0, 1.0)),
        ('ellipticity', 'Ellipticity (e)', None),
        ('prolateness', 'Prolateness (p)', None)
    ]
    
    for param_name, param_label, xlim in distributions:
        output_path = Path(OUTPUT_DIR) / f'{param_name.replace("/", "_")}_distribution.png'
        plot_parameter_distribution(valid_snapshots_data, valid_snapshot_ids, 
                                   param_name, param_label, output_path, xlim)

def create_evolution_summary(snapshots_data, snapshot_ids):
    """Create comprehensive evolution summary plots."""
    
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
    
    create_all_distribution_plots(valid_snapshots_data, valid_snapshot_ids)

def main():
    """Process all snapshots for void shape evolution analysis."""
    
    print("Starting void shape evolution analysis")
    print(f"Data directory: {BASE_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing {len(SNAPSHOT_TO_REDSHIFT)} snapshots")
    print(f"Method: FFT-based Hessian computation")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    processed_snapshots = 0
    failed_snapshots = []
    snapshots_data = []
    snapshot_ids = []
    
    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        result = process_snapshot(snapshot, redshift)
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