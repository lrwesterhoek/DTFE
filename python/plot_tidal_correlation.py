"""
Conditional Probability Mapping Analysis for Phi-Delta Correspondence

Computes P(δ minimum | φ maximum) in local neighborhoods across multiple redshifts.
Generates separate plots for xy, xz, yz planes along with combined overview plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter, maximum_filter
from pathlib import Path
import dtfe_functions as dtfe
from dtfe_shared import save_plot_to_multiple_paths

# ============================================================================
# Configuration Section
# ============================================================================

BASE_DATA_DIR = "output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/tidal_correlation"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7  # Mpc
AXIS_UNITS = "Mpc"

WINDOW_SIZE = 16
OVERLAP = 0.5
MAX_MATCH_DISTANCE = 5.0
SMOOTHING_SIGMA = 10.0

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

def solve_poisson_phi(density_field, box_size, G=1.0, rho_bar=None):
    """Solve Poisson's equation in Fourier space: ∇²φ = 4πG ρ_bar δ"""
    if rho_bar is None:
        rho_bar = np.mean(density_field)
    
    delta = (density_field - rho_bar) / rho_bar
    delta_k = np.fft.fftn(delta)
    
    nx, ny, nz = density_field.shape
    L = box_size
    kx = np.fft.fftfreq(nx, d=L/nx) * 2*np.pi
    ky = np.fft.fftfreq(ny, d=L/ny) * 2*np.pi
    kz = np.fft.fftfreq(nz, d=L/nz) * 2*np.pi
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = np.inf
    
    phi_k = -4*np.pi*G*rho_bar * delta_k / k2
    phi = np.real(np.fft.ifftn(phi_k))
    
    return phi

def find_nearest_matches(phi_coords, delta_coords, max_distance=5.0):
    """Find nearest matches between phi maxima and delta minima."""
    if len(phi_coords) == 0 or len(delta_coords) == 0:
        return [], [], []
    
    matched_pairs = []
    phi_indices = []
    delta_indices = []
    
    for i, phi_coord in enumerate(phi_coords):
        distances = np.linalg.norm(delta_coords - phi_coord, axis=1)
        nearest_idx = np.argmin(distances)
        min_distance = distances[nearest_idx]
        
        if min_distance <= max_distance:
            matched_pairs.append({
                'phi_idx': i,
                'delta_idx': nearest_idx,
                'distance': min_distance,
                'phi_coord': phi_coord,
                'delta_coord': delta_coords[nearest_idx]
            })
            phi_indices.append(i)
            delta_indices.append(nearest_idx)
    
    return matched_pairs, phi_indices, delta_indices

def count_matched_pairs_in_window(phi_window, delta_window, max_distance=3.0):
    """Count matched pairs in local window."""
    phi_local_max = maximum_filter(phi_window, size=3, mode='constant')
    phi_maxima_mask = (phi_window == phi_local_max) & (phi_window > np.percentile(phi_window, 95))
    phi_coords = np.array(np.where(phi_maxima_mask)).T
    
    delta_local_min = minimum_filter(delta_window, size=3, mode='constant')
    delta_minima_mask = (delta_window == delta_local_min) & (delta_window < np.percentile(delta_window, 5))
    delta_coords = np.array(np.where(delta_minima_mask)).T
    
    if len(phi_coords) == 0 or len(delta_coords) == 0:
        return 0
    
    matched_pairs, _, _ = find_nearest_matches(phi_coords, delta_coords, max_distance)
    return len(matched_pairs)

def conditional_probability_analysis(phi_field, delta_field, window_size=16, overlap=0.5):
    """Compute P(δ minimum | φ maximum) in local neighborhoods across volume."""
    step_size = int(window_size * (1 - overlap))
    prob_map = np.zeros_like(phi_field)
    count_map = np.zeros_like(phi_field)
    
    phi_smooth = dtfe.smooth_field(phi_field, sigma=SMOOTHING_SIGMA)
    delta_smooth = dtfe.smooth_field(delta_field, sigma=SMOOTHING_SIGMA)
    
    total_windows = 0
    for i in range(0, phi_field.shape[0] - window_size, step_size):
        for j in range(0, phi_field.shape[1] - window_size, step_size):
            for k in range(0, phi_field.shape[2] - window_size, step_size):
                total_windows += 1
    
    processed_windows = 0
    for i in range(0, phi_field.shape[0] - window_size, step_size):
        for j in range(0, phi_field.shape[1] - window_size, step_size):
            for k in range(0, phi_field.shape[2] - window_size, step_size):
                
                phi_window = phi_smooth[i:i+window_size, j:j+window_size, k:k+window_size]
                delta_window = delta_smooth[i:i+window_size, j:j+window_size, k:k+window_size]

                maxima_mask, _ = dtfe.find_local_maxima(phi_window, footprint_size=3)
                phi_max_count = np.sum(maxima_mask)
                minima_mask, _ = dtfe.find_local_minima(delta_window, footprint_size=3)
                delta_min_count = np.sum(minima_mask)
                matched_count = count_matched_pairs_in_window(phi_window, delta_window)
                
                if phi_max_count > 0:
                    local_prob = matched_count / phi_max_count
                else:
                    local_prob = 0.0
                
                prob_map[i:i+window_size, j:j+window_size, k:k+window_size] += local_prob
                count_map[i:i+window_size, j:j+window_size, k:k+window_size] += 1
                
                processed_windows += 1
                
                if processed_windows % 200 == 0:
                    progress = (processed_windows / total_windows) * 100
                    print(f"  Progress: {progress:.1f}%")
    
    prob_map = np.divide(prob_map, count_map, out=np.zeros_like(prob_map), where=count_map!=0)
    
    return prob_map, count_map

def create_plane_directories(output_dir):
    """Create subdirectories for each plane."""
    plane_dirs = {}
    for slice_dim in SLICE_PLANES_TO_PLOT:
        plane_name = SLICE_PLANES[slice_dim]['name']
        plane_dir = output_dir / plane_name
        plane_dir.mkdir(parents=True, exist_ok=True)
        plane_dirs[slice_dim] = plane_dir
    return plane_dirs

def plot_tidal_phi_field(phi_field, redshift, box_size, plane_dirs):
    """Plot tidal phi field slices for all planes."""
    extent = [0, box_size, 0, box_size]
    phi_smooth = dtfe.smooth_field(phi_field, sigma=2.0)

    for slice_dim in SLICE_PLANES_TO_PLOT:
        plane_info = SLICE_PLANES[slice_dim]
        plane_name = plane_info['name']

        # Use dtfe.extract_2d_slice for proper alignment
        phi_slice = dtfe.extract_2d_slice(phi_smooth, slice_dim)

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        im = ax.imshow(phi_slice.T, cmap='RdBu_r', origin='lower',
                      extent=extent, aspect='equal')

        ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
        ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
        ax.set_title(f"Tidal Gravitational Potential φ (z={redshift:.2f})", fontsize=14)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Gravitational Potential φ', fontsize=12)

        mean_phi = np.mean(phi_slice)
        max_phi = np.max(phi_slice)
        min_phi = np.min(phi_slice)
        std_phi = np.std(phi_slice)

        stats_text = f"Mean: {mean_phi:.2e}\nMax: {max_phi:.2e}\nMin: {min_phi:.2e}\nStd: {std_phi:.2e}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=11)

        ax.set_aspect('equal')
        plt.tight_layout()

        filename = f"tidal_phi_{plane_name}_z{redshift:.2f}.png"
        filepath = plane_dirs[slice_dim] / filename
        save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()

def plot_probability_mapping_by_plane(prob_map, redshift, box_size, plane_dirs):
    """Plot conditional probability mapping for all planes."""
    extent = [0, box_size, 0, box_size]

    for slice_dim in SLICE_PLANES_TO_PLOT:
        plane_info = SLICE_PLANES[slice_dim]
        plane_name = plane_info['name']

        # Use dtfe.extract_2d_slice for proper alignment
        prob_slice = dtfe.extract_2d_slice(prob_map, slice_dim)

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        im = ax.imshow(prob_slice.T, cmap='hot', vmin=0, vmax=1,
                      origin='lower', extent=extent, aspect='equal')

        ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
        ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
        ax.set_title(f"P(δ minimum | φ maximum) (z={redshift:.2f})", fontsize=14)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Conditional Probability', fontsize=12)

        mean_prob = np.mean(prob_slice)
        max_prob = np.max(prob_slice)
        std_prob = np.std(prob_slice)

        stats_text = f"Mean: {mean_prob:.3f}\nMax: {max_prob:.3f}\nStd: {std_prob:.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=11)

        ax.set_aspect('equal')
        plt.tight_layout()

        filename = f"prob_map_{plane_name}_z{redshift:.2f}.png"
        filepath = plane_dirs[slice_dim] / filename
        save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()

def create_combined_overview_plots(prob_map, phi_field, redshift, box_size, output_dir):
    """Create combined overview plots showing all planes together."""
    extent = [0, box_size, 0, box_size]
    phi_smooth = dtfe.smooth_field(phi_field, sigma=2.0)

    # Extract slices for all planes using dtfe.extract_2d_slice
    prob_slices = []
    phi_slices = []
    for slice_dim in SLICE_PLANES_TO_PLOT:
        prob_slices.append(dtfe.extract_2d_slice(prob_map, slice_dim))
        phi_slices.append(dtfe.extract_2d_slice(phi_smooth, slice_dim))

    # Plot probability maps
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    for i, slice_dim in enumerate(SLICE_PLANES_TO_PLOT):
        plane_info = SLICE_PLANES[slice_dim]
        im = axes[i].imshow(prob_slices[i].T, cmap='hot', vmin=0, vmax=1,
                          origin='lower', extent=extent, aspect='equal')

        axes[i].set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
        axes[i].set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
        axes[i].set_title(f"{plane_info['name'].replace('_', ' ').title()}", fontsize=13)
        axes[i].set_aspect('equal')

        cbar = plt.colorbar(im, ax=axes[i], label='Probability', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

    fig.suptitle(f"Conditional Probability: P(δ minimum | φ maximum) (z={redshift:.2f})", fontsize=14)

    plt.tight_layout()

    prob_filename = f"prob_map_combined_z{redshift:.2f}.png"
    prob_filepath = output_dir / prob_filename
    save_plot_to_multiple_paths(fig, prob_filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    # Plot phi fields
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    for i, slice_dim in enumerate(SLICE_PLANES_TO_PLOT):
        plane_info = SLICE_PLANES[slice_dim]
        im = axes[i].imshow(phi_slices[i].T, cmap='RdBu_r',
                          origin='lower', extent=extent, aspect='equal')

        axes[i].set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
        axes[i].set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
        axes[i].set_title(f"{plane_info['name'].replace('_', ' ').title()}", fontsize=13)
        axes[i].set_aspect('equal')

        cbar = plt.colorbar(im, ax=axes[i], label='φ', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

    fig.suptitle(f"Tidal Gravitational Potential φ (z={redshift:.2f})", fontsize=14)

    plt.tight_layout()

    phi_filename = f"tidal_phi_combined_z{redshift:.2f}.png"
    phi_filepath = output_dir / phi_filename
    save_plot_to_multiple_paths(fig, phi_filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

def analyze_snapshot(snapshot, redshift, plane_dirs, output_dir):
    """Analyze single snapshot."""
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")
    
    data_path = Path(BASE_DATA_DIR) / f'snapdir_{snapshot}'
    density_file = data_path / 'output.a_den'
    
    if not density_file.exists():
        print(f"Density file not found: {density_file}")
        return False
    
    try:
        print("Loading density field")
        field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)
        density_field = dtfe.load_binary_field(density_file, field_shape, num_components=1, dtype=np.float32, try_infer_shape=True)
        
        print("Computing density contrast")
        rho_bar = np.mean(density_field)
        delta = (density_field - rho_bar) / rho_bar
        
        print("Computing gravitational potential")
        phi = solve_poisson_phi(density_field, BOX_SIZE, G=1.0)
        
        print("Computing conditional probability mapping")
        prob_map, count_map = conditional_probability_analysis(phi, delta, WINDOW_SIZE, OVERLAP)
        
        print("Creating tidal phi field plots")
        plot_tidal_phi_field(phi, redshift, BOX_SIZE, plane_dirs)
        
        print("Creating probability mapping plots")
        plot_probability_mapping_by_plane(prob_map, redshift, BOX_SIZE, plane_dirs)
        
        print("Creating combined overview plots")
        create_combined_overview_plots(prob_map, phi, redshift, BOX_SIZE, output_dir)
        
        print(f"Completed snapshot {snapshot}")
        return True
        
    except Exception as e:
        print(f"Error analyzing snapshot {snapshot}: {str(e)}")
        return False

def main():
    """Main function to analyze all snapshots."""
    print(f"Starting conditional probability mapping analysis")
    print(f"Processing {len(SNAPSHOT_TO_REDSHIFT)} snapshots")
    print(f"Window size: {WINDOW_SIZE}, Overlap: {OVERLAP*100:.0f}%")
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    plane_dirs = create_plane_directories(output_path)
    
    successful_analyses = []
    failed_analyses = []
    
    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        success = analyze_snapshot(snapshot, redshift, plane_dirs, output_path)
        
        if success:
            successful_analyses.append((snapshot, redshift))
        else:
            failed_analyses.append((snapshot, redshift))
    
    print(f"\nAnalysis complete")
    print(f"Total snapshots: {len(SNAPSHOT_TO_REDSHIFT)}")
    print(f"Successful: {len(successful_analyses)}")
    print(f"Failed: {len(failed_analyses)}")
    
    if successful_analyses:
        print(f"\nSuccessfully analyzed snapshots:")
        for snapshot, redshift in successful_analyses:
            print(f"  Snapshot {snapshot} (z={redshift:.2f})")
    
    if failed_analyses:
        print(f"\nFailed to analyze snapshots:")
        for snapshot, redshift in failed_analyses:
            print(f"  Snapshot {snapshot} (z={redshift:.2f})")
    
    print(f"\nOutput saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()