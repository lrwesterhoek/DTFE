"""
Void Shape Analysis Tool

Identifies voids in density fields and characterizes their shapes using
Hessian eigenvalue analysis. Computes axis ratios and BBKS shape parameters
across multiple redshifts.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import minimum_filter
from pathlib import Path
import dtfe_functions as dtfe
from dtfe_shared import save_plot_to_multiple_paths

# ============================================================================
# Configuration Section
# ============================================================================

BASE_DATA_DIR = "output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/void_shape_analysis"

FIELD_RESOLUTION = 512
SMOOTHING_SIGMA = 10.0
FOOTPRINT_SIZE = 11
POSITIVE_EIGENVALUES_ONLY = True

DPI = 300

SNAPSHOT_TO_REDSHIFT = dtfe.SNAPSHOT_TO_REDSHIFT

# ============================================================================

def plot_axis_ratios(axis_ratios, density_values=None, redshift=None, save_path=None):
    """Create scatter plot of b/a vs c/a axis ratios."""
    if len(axis_ratios) == 0:
        print("  No data to plot for axis ratios")
        return
        
    b_a = axis_ratios[:, 0]
    c_a = axis_ratios[:, 1]
    
    plt.figure(figsize=(8, 7))
    
    if density_values is not None:
        sc = plt.scatter(b_a, c_a, c=density_values, cmap='viridis', s=30, alpha=0.7)
        cbar = plt.colorbar(sc)
        cbar.set_label('Density Contrast δ', fontsize=10)
    else:
        plt.scatter(b_a, c_a, color='blue', s=30, alpha=0.7)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.xlabel('b/a (Intermediate-to-Major Axis Ratio)', fontsize=11)
    plt.ylabel('c/a (Minor-to-Major Axis Ratio)', fontsize=11)
    
    title = 'Shape Distribution of Void Ellipsoids'
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    plt.title(title, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_plot_to_multiple_paths(plt.gcf(), save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_bbks_parameters(bbks_params, density_values=None, redshift=None, save_path=None):
    """Create scatter plot of BBKS e vs p parameters."""
    if len(bbks_params) == 0:
        print("  No data to plot for BBKS parameters")
        return
        
    e_values = bbks_params[:, 0]
    p_values = bbks_params[:, 1]
    
    plt.figure(figsize=(8, 7))
    
    if density_values is not None:
        sc = plt.scatter(e_values, p_values, c=density_values, cmap='viridis', s=30, alpha=0.7)
        cbar = plt.colorbar(sc)
        cbar.set_label('Density Contrast δ', fontsize=10)
    else:
        plt.scatter(e_values, p_values, color='blue', s=30, alpha=0.7)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.4)
    
    max_e = max(0.5, np.max(e_values) * 1.1) if len(e_values) > 0 else 0.5
    max_abs_p = max(0.5, np.max(np.abs(p_values)) * 1.1) if len(p_values) > 0 else 0.5
    plt.xlim(0, max_e)
    plt.ylim(-max_abs_p, max_abs_p)
    plt.xlabel('e (Ellipticity)', fontsize=11)
    plt.ylabel('p (Prolateness)', fontsize=11)
    
    title = 'BBKS Shape Parameters of Void Ellipsoids'
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    plt.title(title, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_plot_to_multiple_paths(plt.gcf(), save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_bbks_histograms(bbks_params, redshift=None, save_path=None):
    """Create histograms of BBKS e and p parameters."""
    if len(bbks_params) == 0:
        print("  No data to plot for BBKS histograms")
        return
        
    e_values = bbks_params[:, 0]
    p_values = bbks_params[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if len(e_values) > 0:
        e_bins = np.linspace(0, np.max(e_values) * 1.1, 20)
        ax1.hist(e_values, bins=e_bins, color='blue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('e (Ellipticity)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        
        title1 = 'Distribution of Ellipticity (e)'
        if redshift is not None:
            title1 += f' (z={redshift:.2f})'
        ax1.set_title(title1, fontsize=12)
        
        mean_e = np.mean(e_values)
        ax1.axvline(x=mean_e, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Mean: {mean_e:.3f}')
        ax1.legend()
    
    if len(p_values) > 0:
        p_bins = np.linspace(np.min(p_values) * 1.1, np.max(p_values) * 1.1, 20)
        ax2.hist(p_values, bins=p_bins, color='green', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('p (Prolateness)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        
        title2 = 'Distribution of Prolateness (p)'
        if redshift is not None:
            title2 += f' (z={redshift:.2f})'
        ax2.set_title(title2, fontsize=12)
        
        mean_p = np.mean(p_values)
        ax2.axvline(x=mean_p, color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean: {mean_p:.3f}')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_plot_to_multiple_paths(plt.gcf(), save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def process_snapshot(snapshot, redshift):
    """Process a single snapshot for void shape analysis."""
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")
    
    data_path = Path(BASE_DATA_DIR) / f'snapdir_{snapshot}'
    density_file = data_path / 'output.a_den'
    
    if not data_path.exists():
        print(f"Path does not exist: {data_path}")
        return
    
    if not density_file.exists():
        print(f"Density file not found: {density_file}")
        return
    
    output_path = Path(OUTPUT_DIR) / f'snapshot_{snapshot}_z{redshift:.2f}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Loading density field")
        field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)
        density_field = dtfe.load_binary_field(density_file, field_shape, num_components=1,
                                           dtype=np.float32, try_infer_shape=True)
        
        print("Computing density contrast")
        delta_field = dtfe.calculate_density_contrast(density_field)
        delta_field = dtfe.smooth_field(delta_field, sigma=SMOOTHING_SIGMA)

        print("Computing Hessian using FFT")
        hessian_components = dtfe.calculate_hessian_fft(delta_field)

        print("Finding minima and computing eigenvalues")
        minima_mask, minima_coords = dtfe.find_local_minima(delta_field, footprint_size=FOOTPRINT_SIZE)
        eigenvalues, eigenvectors = dtfe.compute_eigenvalues_at_minima(hessian_components, minima_coords)
        minima_coords, eigenvalues = dtfe.filter_minima_by_eigenvalues(
            minima_coords, eigenvalues, positive_only=POSITIVE_EIGENVALUES_ONLY
        )

        if len(minima_coords) == 0:
            print("No valid minima found")
            return

        print(f"Found {len(minima_coords)} valid minima")

        minima_indices = minima_coords  # For compatibility

        print("Calculating shape parameters")
        shape_params = dtfe.calculate_shape_parameters(eigenvalues)
        axis_ratios = shape_params['axis_ratios']
        bbks_params = shape_params['bbks_params']
        
        density_values = np.array([delta_field[tuple(idx)] for idx in minima_indices])
        
        if len(axis_ratios) > 0:
            print(f"  Mean b/a: {np.mean(axis_ratios[:, 0]):.3f} ± {np.std(axis_ratios[:, 0]):.3f}")
            print(f"  Mean c/a: {np.mean(axis_ratios[:, 1]):.3f} ± {np.std(axis_ratios[:, 1]):.3f}")
            print(f"  Mean ellipticity: {np.mean(bbks_params[:, 0]):.3f} ± {np.std(bbks_params[:, 0]):.3f}")
            print(f"  Mean prolateness: {np.mean(bbks_params[:, 1]):.3f} ± {np.std(bbks_params[:, 1]):.3f}")
            
            prolate_pct = 100 * np.sum(bbks_params[:, 1] > 0) / len(bbks_params)
            oblate_pct = 100 * np.sum(bbks_params[:, 1] < 0) / len(bbks_params)
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
            f.write(f"  Grid resolution: {FIELD_RESOLUTION}³\n")
            f.write(f"  Footprint size: {FOOTPRINT_SIZE}\n")
            f.write(f"  Smoothing sigma: {SMOOTHING_SIGMA}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total minima found: {len(minima_indices)}\n")
            
            if len(axis_ratios) > 0:
                f.write(f"  Mean axis ratios:\n")
                f.write(f"    b/a: {np.mean(axis_ratios[:, 0]):.3f} ± {np.std(axis_ratios[:, 0]):.3f}\n")
                f.write(f"    c/a: {np.mean(axis_ratios[:, 1]):.3f} ± {np.std(axis_ratios[:, 1]):.3f}\n")
                f.write(f"  BBKS parameters:\n")
                f.write(f"    Mean ellipticity (e): {np.mean(bbks_params[:, 0]):.3f} ± {np.std(bbks_params[:, 0]):.3f}\n")
                f.write(f"    Mean prolateness (p): {np.mean(bbks_params[:, 1]):.3f} ± {np.std(bbks_params[:, 1]):.3f}\n")
                prolate_pct = 100 * np.sum(bbks_params[:, 1] > 0) / len(bbks_params)
                oblate_pct = 100 * np.sum(bbks_params[:, 1] < 0) / len(bbks_params)
                f.write(f"    Prolate shapes: {prolate_pct:.1f}%\n")
                f.write(f"    Oblate shapes: {oblate_pct:.1f}%\n")
                f.write(f"  Density statistics:\n")
                f.write(f"    Mean density at voids: {np.mean(density_values):.3f}\n")
                f.write(f"    Std density at voids: {np.std(density_values):.3f}\n")
                f.write(f"    Min density at voids: {np.min(density_values):.3f}\n")
                f.write(f"    Max density at voids: {np.max(density_values):.3f}\n")
        
        print(f"Completed snapshot {snapshot}")
        
    except Exception as e:
        print(f"Error processing snapshot {snapshot}: {str(e)}")
        import traceback
        traceback.print_exc()

def create_evolution_summary():
    """Create summary plot showing evolution of void shape properties across redshifts."""
    print("\nCreating evolution summary")
    
    redshifts = []
    n_voids = []
    mean_b_a = []
    mean_c_a = []
    mean_ellipticity = []
    mean_prolateness = []
    prolate_fractions = []
    oblate_fractions = []
    
    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        summary_file = Path(OUTPUT_DIR) / f'snapshot_{snapshot}_z{redshift:.2f}' / f'void_shape_analysis_summary_z{redshift:.2f}.txt'
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    content = f.read()

                import re
                n_voids_match = re.search(r'Total minima found: (\d+)', content)
                b_a_match = re.search(r'b/a: ([\d.]+) ±', content)
                c_a_match = re.search(r'c/a: ([\d.]+) ±', content)
                ellipticity_match = re.search(r'Mean ellipticity \(e\): ([-\d.]+) ±', content)
                prolateness_match = re.search(r'Mean prolateness \(p\): ([-\d.]+) ±', content)
                prolate_match = re.search(r'Prolate shapes: ([\d.]+)%', content)
                oblate_match = re.search(r'Oblate shapes: ([\d.]+)%', content)
                
                if all([n_voids_match, b_a_match, c_a_match, ellipticity_match, 
                       prolateness_match, prolate_match, oblate_match]):
                    redshifts.append(redshift)
                    n_voids.append(int(n_voids_match.group(1)))
                    mean_b_a.append(float(b_a_match.group(1)))
                    mean_c_a.append(float(c_a_match.group(1)))
                    mean_ellipticity.append(float(ellipticity_match.group(1)))
                    mean_prolateness.append(float(prolateness_match.group(1)))
                    prolate_fractions.append(float(prolate_match.group(1)))
                    oblate_fractions.append(float(oblate_match.group(1)))
                    
            except Exception as e:
                print(f"  Warning: Could not read summary for z={redshift:.2f}: {e}")
    
    if len(redshifts) > 0:
        sorted_idx = np.argsort(redshifts)
        redshifts = np.array(redshifts)[sorted_idx]
        n_voids = np.array(n_voids)[sorted_idx]
        mean_b_a = np.array(mean_b_a)[sorted_idx]
        mean_c_a = np.array(mean_c_a)[sorted_idx]
        mean_ellipticity = np.array(mean_ellipticity)[sorted_idx]
        mean_prolateness = np.array(mean_prolateness)[sorted_idx]
        prolate_fractions = np.array(prolate_fractions)[sorted_idx]
        oblate_fractions = np.array(oblate_fractions)[sorted_idx]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        ax1.plot(redshifts, n_voids, 'ko-', markersize=4)
        ax1.set_xlabel('Redshift z')
        ax1.set_ylabel('Number of Voids')
        ax1.set_title('Evolution of Void Count')
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        
        ax2.plot(redshifts, mean_b_a, 'ro-', label='b/a (intermediate/major)', markersize=4)
        ax2.plot(redshifts, mean_c_a, 'bo-', label='c/a (minor/major)', markersize=4)
        ax2.set_xlabel('Redshift z')
        ax2.set_ylabel('Mean Axis Ratio')
        ax2.set_title('Evolution of Mean Axis Ratios')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()
        
        ax3.plot(redshifts, mean_ellipticity, 'go-', label='Ellipticity (e)', markersize=4)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(redshifts, mean_prolateness, 'mo-', label='Prolateness (p)', markersize=4)
        ax3.set_xlabel('Redshift z')
        ax3.set_ylabel('Mean Ellipticity (e)', color='g')
        ax3_twin.set_ylabel('Mean Prolateness (p)', color='m')
        ax3.set_title('Evolution of BBKS Parameters')
        ax3.grid(True, alpha=0.3)
        ax3.invert_xaxis()
        ax3_twin.invert_xaxis()
        
        ax4.plot(redshifts, prolate_fractions, 'ro-', label='Prolate (p > 0)', markersize=4)
        ax4.plot(redshifts, oblate_fractions, 'bo-', label='Oblate (p < 0)', markersize=4)
        ax4.set_xlabel('Redshift z')
        ax4.set_ylabel('Fraction (%)')
        ax4.set_title('Evolution of Shape Type Fractions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.invert_xaxis()
        
        plt.tight_layout()
        
        evolution_file = Path(OUTPUT_DIR) / 'void_shape_evolution_summary.png'
        save_plot_to_multiple_paths(fig, evolution_file, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Evolution summary saved")
        
        evolution_data_file = Path(OUTPUT_DIR) / 'void_shape_evolution_data.txt'
        with open(evolution_data_file, 'w') as f:
            f.write("Redshift Evolution of Void Shape Properties\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Redshift':>8} {'N_voids':>8} {'<b/a>':>8} {'<c/a>':>8} {'<e>':>8} {'<p>':>8} {'Prolate%':>9} {'Oblate%':>8}\n")
            f.write("-" * 70 + "\n")
            for i in range(len(redshifts)):
                f.write(f"{redshifts[i]:8.2f} {n_voids[i]:8d} {mean_b_a[i]:8.3f} {mean_c_a[i]:8.3f} "
                       f"{mean_ellipticity[i]:8.3f} {mean_prolateness[i]:8.3f} "
                       f"{prolate_fractions[i]:8.1f} {oblate_fractions[i]:8.1f}\n")
    else:
        print("  Warning: No valid summary data found")

def main():
    """Main function to process all snapshots for void shape analysis."""
    print(f"Starting void shape analysis")
    print(f"Processing {len(SNAPSHOT_TO_REDSHIFT)} snapshots")
    
    processed_count = 0
    failed_snapshots = []
    
    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        try:
            process_snapshot(snapshot, redshift)
            processed_count += 1
        except Exception as e:
            print(f"Failed to process snapshot {snapshot} (z={redshift:.2f}): {str(e)}")
            failed_snapshots.append((snapshot, redshift))
    
    if processed_count > 1:
        try:
            create_evolution_summary()
        except Exception as e:
            print(f"Failed to create evolution summary: {str(e)}")
    
    print(f"\nAnalysis complete")
    print(f"Successfully processed: {processed_count}/{len(SNAPSHOT_TO_REDSHIFT)} snapshots")
    
    if failed_snapshots:
        print(f"Failed snapshots: {len(failed_snapshots)}")
        for snapshot, redshift in failed_snapshots:
            print(f"  Snapshot {snapshot} (z={redshift:.2f})")
    
    print(f"\nOutput saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()