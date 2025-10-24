"""
Velocity Divergence-Density Analysis Tool

Analyzes the relationship between velocity divergence and density contrast
across multiple redshifts. Compares measured slopes with theoretical predictions
from linear perturbation theory.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter
from pathlib import Path
import dtfe_functions as dtfe
from dtfe_shared import save_plot_to_multiple_paths

# ============================================================================
# Configuration Section
# ============================================================================

BASE_DATA_DIR = "output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/veldiv_density_analysis"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7  # Mpc
AXIS_UNITS = "Mpc"
VELOCITY_UNITS = "km/s"
LINEAR_REGIME_RANGE = (-0.5, 1.0)
SMOOTHING_SIGMA = 0

# Cosmology parameters (TNG50-3-Dark)
H0 = 67.74  # km/s/Mpc
OMEGA_M0 = 0.3089
OMEGA_LAMBDA0 = 0.6911

SAMPLE_SIZE = 100000
THRESHOLD = 1e-3

# Which slice planes to visualize? (0=YZ, 1=XZ, 2=XY)
SLICE_PLANES_TO_PLOT = [0, 1, 2]

DPI = 300

SNAPSHOT_TO_REDSHIFT = dtfe.SNAPSHOT_TO_REDSHIFT

# ============================================================================

SLICE_PLANES = {
    0: {'name': 'yz_plane', 'axis_labels': ('Y', 'Z')},
    1: {'name': 'xz_plane', 'axis_labels': ('X', 'Z')},
    2: {'name': 'xy_plane', 'axis_labels': ('X', 'Y')}
}

# ============================================================================

def load_fields(base_path, full_field_shape, try_infer_shape=True):
    """Load density and velocity divergence fields."""
    density_pattern = str(base_path / 'output.a_den')
    velDiv_pattern = str(base_path / 'output.a_velDiv')
    
    density_field = dtfe.load_binary_field(density_pattern, full_field_shape, num_components=1,
                                        dtype=np.float32, try_infer_shape=try_infer_shape)
    velDiv_field = dtfe.load_binary_field(velDiv_pattern, full_field_shape, num_components=1,
                                      dtype=np.float32, try_infer_shape=try_infer_shape)
    
    return density_field, velDiv_field

def process_fields(density_field, velDiv_field, a_scale, sigma=0):
    """Process fields: compute density contrast and apply smoothing."""
    mean_density = np.mean(density_field)
    delta_field = (density_field - mean_density) / mean_density
    delta_field = dtfe.smooth_field(delta_field, sigma=sigma)
    velDiv_field = dtfe.smooth_field(velDiv_field, sigma=sigma)
    velDiv_field *= np.sqrt(a_scale)
    return delta_field, velDiv_field

def compute_regression(delta_field, velDiv_field, threshold=1e-3, delta_range=None):
    """Compute linear regression slope between delta and velocity divergence."""
    delta_flat = delta_field.flatten()
    velDiv_flat = velDiv_field.flatten()
    
    mask = np.abs(delta_flat) > threshold
    
    if delta_range is not None:
        delta_min, delta_max = delta_range
        range_mask = (delta_flat >= delta_min) & (delta_flat <= delta_max)
        mask = mask & range_mask
    
    delta_reg = delta_flat[mask]
    velDiv_reg = velDiv_flat[mask]
    
    slope_reg = np.sum(delta_reg * velDiv_reg) / np.sum(delta_reg**2)
    return slope_reg, delta_reg, velDiv_reg

def sample_scatter_data(delta_reg, velDiv_reg, sample_size=100000, delta_range=None):
    """Sample data for scatter plot."""
    if delta_range is not None:
        delta_min, delta_max = delta_range
        mask = (delta_reg >= delta_min) & (delta_reg <= delta_max)
        delta_reg = delta_reg[mask]
        velDiv_reg = velDiv_reg[mask]
    
    n_points = delta_reg.size
    if n_points > sample_size:
        idx = np.random.choice(n_points, size=sample_size, replace=False)
        return delta_reg[idx], velDiv_reg[idx]
    
    return delta_reg, velDiv_reg

def setup_symmetric_normalizations(delta_slice, velDiv_slice, residual_slice):
    """Setup symmetric normalizations centered at zero for all fields."""
    delta_max = np.nanmax(np.abs(delta_slice))
    velDiv_max = np.nanmax(np.abs(velDiv_slice))
    residual_max = np.nanmax(np.abs(residual_slice))
    
    delta_norm = colors.Normalize(vmin=-delta_max, vmax=delta_max)
    velDiv_norm = colors.Normalize(vmin=-velDiv_max, vmax=velDiv_max)
    residual_norm = colors.Normalize(vmin=-residual_max, vmax=residual_max)
    
    return delta_norm, velDiv_norm, residual_norm

def create_density_plot(delta_slice, slice_dim, box_size, delta_norm, redshift, output_path):
    """Create density contrast field plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    im = ax.imshow(delta_slice.T, origin='lower', cmap='seismic', norm=delta_norm,
                   extent=[0, box_size, 0, box_size])

    plane_info = SLICE_PLANES[slice_dim]
    title = "Density Contrast δ"
    if redshift is not None:
        title += f" (z={redshift:.2f})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("δ", fontsize=12)

    ax.set_aspect('equal')
    plt.tight_layout()

    plane_name = plane_info['name']
    filepath = output_path / f"density_contrast_{plane_name}_z{redshift:.2f}.png"
    save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def create_velocity_divergence_plot(velDiv_slice, slice_dim, box_size, velDiv_norm, redshift, output_path):
    """Create velocity divergence field plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    im = ax.imshow(velDiv_slice.T, origin='lower', cmap='seismic', norm=velDiv_norm,
                   extent=[0, box_size, 0, box_size])

    plane_info = SLICE_PLANES[slice_dim]
    title = "Velocity Divergence"
    if redshift is not None:
        title += f" (z={redshift:.2f})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{VELOCITY_UNITS}/{AXIS_UNITS}", fontsize=12)

    ax.set_aspect('equal')
    plt.tight_layout()

    plane_name = plane_info['name']
    filepath = output_path / f"velocity_divergence_{plane_name}_z{redshift:.2f}.png"
    save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def create_linear_fit_plot(delta_sample, velDiv_sample, slope_reg, slope_theory, 
                          delta_range, redshift, output_path):
    """Create linear fit scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.scatter(delta_sample, velDiv_sample, s=1, alpha=0.5, color='lightblue')
    
    delta_min, delta_max = delta_range
    x_fit = np.linspace(delta_min, delta_max, 100)
    
    ax.plot(x_fit, slope_reg * x_fit, color="red", linewidth=2, 
            label=f"Linear Fit: {slope_reg:.2f}")
    
    ax.plot(x_fit, slope_theory * x_fit, color="green", linestyle="--", linewidth=2, 
            label=f"Theory: {slope_theory:.2f}")
    
    ax.set_xlabel("Density Contrast δ")
    ax.set_ylabel("Velocity Divergence (km/s/Mpc)")
    
    title = "Velocity Divergence vs Density Contrast δ"
    if redshift is not None:
        title += f" (z={redshift:.2f})"
    ax.set_title(title, fontsize=14)
    
    ax.set_xlim([delta_min, delta_max])
    
    y_range = max(abs(slope_reg * delta_max), abs(slope_theory * delta_max)) * 1.5
    ax.set_ylim([-y_range, y_range])
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    filepath = output_path / f"linear_fit_z{redshift:.2f}.png"
    save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def create_residual_plot(residual_slice, slice_dim, box_size, residual_norm, redshift, output_path):
    """Create residual field plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    im = ax.imshow(residual_slice.T, origin='lower', cmap='seismic', norm=residual_norm,
                   extent=[0, box_size, 0, box_size])

    plane_info = SLICE_PLANES[slice_dim]
    title = "Residual Field (∇·v - slope×δ)"
    if redshift is not None:
        title += f" (z={redshift:.2f})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{VELOCITY_UNITS}/{AXIS_UNITS}", fontsize=12)

    ax.set_aspect('equal')
    plt.tight_layout()

    plane_name = plane_info['name']
    filepath = output_path / f"residual_field_{plane_name}_z{redshift:.2f}.png"
    save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def process_snapshot(snapshot, redshift):
    """Process single snapshot for velocity divergence-density analysis."""
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")
    
    data_path = Path(BASE_DATA_DIR) / f'snapdir_{snapshot}'
    
    if not data_path.exists():
        print(f"Path does not exist: {data_path}")
        return
    
    density_pattern = str(data_path / 'output.a_den')
    velDiv_pattern = str(data_path / 'output.a_velDiv')
    
    if not glob.glob(density_pattern):
        print(f"No density files found")
        return
    if not glob.glob(velDiv_pattern):
        print(f"No velocity divergence files found")
        return
    
    output_path = Path(OUTPUT_DIR) / f'snapshot_{snapshot}_z{redshift:.2f}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Computing cosmological parameters")
        cosmo_params = dtfe.get_cosmology_params(redshift, H0, OMEGA_M0, OMEGA_LAMBDA0)
        a_scale = cosmo_params['a']
        H_z = cosmo_params['H_z']
        f_growth = cosmo_params['f_growth']
        slope_theory = cosmo_params['slope_theory']
        
        print("Loading fields")
        field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)
        density_field, velDiv_field = load_fields(data_path, field_shape)
        delta_field, velDiv_field = process_fields(density_field, velDiv_field, a_scale, SMOOTHING_SIGMA)
        
        print("Computing regression slopes")
        full_slope_reg, full_delta_reg, full_velDiv_reg = compute_regression(
            delta_field, velDiv_field, THRESHOLD
        )
        
        slope_reg, delta_reg, velDiv_reg = compute_regression(
            delta_field, velDiv_field, THRESHOLD, delta_range=LINEAR_REGIME_RANGE
        )
        
        print(f"  Theoretical slope: {slope_theory:.2f} km/s/Mpc")
        print(f"  Measured slope: {slope_reg:.2f} km/s/Mpc")
        print(f"  Difference: {(slope_reg - slope_theory) / slope_theory * 100:.1f}%")
        
        delta_sample, velDiv_sample = sample_scatter_data(
            delta_reg, velDiv_reg, SAMPLE_SIZE, delta_range=LINEAR_REGIME_RANGE
        )

        residual_field = velDiv_field - slope_reg * delta_field

        print("Creating plots for all planes")
        for slice_dim in SLICE_PLANES_TO_PLOT:
            plane_name = SLICE_PLANES[slice_dim]['name']
            plane_dir = output_path / plane_name

            print(f"  Creating {plane_name} visualizations...")

            delta_slice = dtfe.extract_2d_slice(delta_field, slice_dim)
            velDiv_slice = dtfe.extract_2d_slice(velDiv_field, slice_dim)
            residual_slice = dtfe.extract_2d_slice(residual_field, slice_dim)

            delta_norm, velDiv_norm, residual_norm = setup_symmetric_normalizations(
                delta_slice, velDiv_slice, residual_slice
            )

            create_density_plot(delta_slice, slice_dim, BOX_SIZE, delta_norm, redshift, plane_dir)
            create_velocity_divergence_plot(velDiv_slice, slice_dim, BOX_SIZE, velDiv_norm, redshift, plane_dir)
            create_residual_plot(residual_slice, slice_dim, BOX_SIZE, residual_norm, redshift, plane_dir)

        # Create linear fit plot once (not plane-specific)
        create_linear_fit_plot(delta_sample, velDiv_sample, slope_reg, slope_theory,
                             LINEAR_REGIME_RANGE, redshift, output_path)

        # Get statistics from the first plane for summary
        delta_slice_stats = dtfe.extract_2d_slice(delta_field, 0)
        velDiv_slice_stats = dtfe.extract_2d_slice(velDiv_field, 0)
        residual_slice_stats = dtfe.extract_2d_slice(residual_field, 0)

        summary_file = output_path / f'analysis_summary_z{redshift:.2f}.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Velocity Divergence-Density Analysis Summary\n")
            f.write(f"Snapshot: {snapshot}, Redshift: {redshift:.2f}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Cosmological Parameters:\n")
            f.write(f"  Scale factor a = {a_scale:.4f}\n")
            f.write(f"  H(z) = {H_z:.2f} km/s/Mpc\n")
            f.write(f"  f(z) = {f_growth:.3f}\n")
            f.write(f"  Theoretical slope = {slope_theory:.2f} km/s/Mpc\n\n")
            f.write(f"Regression Results:\n")
            f.write(f"  Full-range slope = {full_slope_reg:.2f} km/s/Mpc\n")
            f.write(f"  Linear regime slope = {slope_reg:.2f} km/s/Mpc\n")
            f.write(f"  Difference from theory = {(slope_reg - slope_theory) / slope_theory * 100:.1f}%\n\n")
            f.write(f"Field Statistics (from YZ plane slice):\n")
            f.write(f"  Delta range: [{np.min(delta_slice_stats):.3f}, {np.max(delta_slice_stats):.3f}]\n")
            f.write(f"  VelDiv range: [{np.min(velDiv_slice_stats):.1f}, {np.max(velDiv_slice_stats):.1f}] km/s/Mpc\n")
            f.write(f"  Residual range: [{np.min(residual_slice_stats):.1f}, {np.max(residual_slice_stats):.1f}] km/s/Mpc\n")
        
        print(f"Completed snapshot {snapshot}")
        
    except Exception as e:
        print(f"Error processing snapshot {snapshot}: {str(e)}")
        import traceback
        traceback.print_exc()

def create_evolution_summary():
    """Create summary plot showing evolution of key parameters across redshifts."""
    print("\nCreating evolution summary")
    
    redshifts = []
    theoretical_slopes = []
    measured_slopes = []
    differences = []
    
    output_path = Path(OUTPUT_DIR)
    
    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        summary_file = output_path / f'snapshot_{snapshot}_z{redshift:.2f}' / f'analysis_summary_z{redshift:.2f}.txt'
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    content = f.read()

                import re
                theory_match = re.search(r'Theoretical slope = ([-\d.]+) km/s/Mpc', content)
                linear_match = re.search(r'Linear regime slope = ([-\d.]+) km/s/Mpc', content)
                diff_match = re.search(r'Difference from theory = ([-\d.]+)%', content)
                
                if theory_match and linear_match and diff_match:
                    redshifts.append(redshift)
                    theoretical_slopes.append(float(theory_match.group(1)))
                    measured_slopes.append(float(linear_match.group(1)))
                    differences.append(float(diff_match.group(1)))
                    
            except Exception as e:
                print(f"  Warning: Could not read summary for z={redshift:.2f}: {e}")
    
    if len(redshifts) > 0:
        sorted_idx = np.argsort(redshifts)
        redshifts = np.array(redshifts)[sorted_idx]
        theoretical_slopes = np.array(theoretical_slopes)[sorted_idx]
        measured_slopes = np.array(measured_slopes)[sorted_idx]
        differences = np.array(differences)[sorted_idx]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(redshifts, theoretical_slopes, 'g--', label='Theory', linewidth=2)
        ax1.plot(redshifts, measured_slopes, 'ro-', label='Measured', markersize=4)
        ax1.set_xlabel('Redshift z')
        ax1.set_ylabel('Slope (km/s/Mpc)')
        ax1.set_title('Evolution of Velocity Divergence-Density Slope')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        
        ax2.plot(redshifts, differences, 'bo-', markersize=4)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Redshift z')
        ax2.set_ylabel('Difference from Theory (%)')
        ax2.set_title('Relative Difference: (Measured - Theory) / Theory × 100%')
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()
        
        plt.tight_layout()
        
        evolution_file = output_path / 'slope_evolution_summary.png'
        save_plot_to_multiple_paths(fig, evolution_file, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        
        evolution_data_file = output_path / 'evolution_data.txt'
        with open(evolution_data_file, 'w') as f:
            f.write("Redshift Evolution of Velocity Divergence-Density Relationship\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Redshift':>8} {'Theory':>10} {'Measured':>10} {'Diff(%)':>8}\n")
            f.write("-" * 40 + "\n")
            for z, th, meas, diff in zip(redshifts, theoretical_slopes, measured_slopes, differences):
                f.write(f"{z:8.2f} {th:10.2f} {meas:10.2f} {diff:8.1f}\n")
        
        print("Evolution summary created")
    else:
        print("  Warning: No valid summary data found")

def main():
    """Main function to process all snapshots."""
    print(f"Starting velocity divergence-density analysis")
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