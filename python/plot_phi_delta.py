"""
Direct Extrema Correlation Analysis for Phi-Delta Correspondence

Analyzes spatial correlation between gravitational potential maxima and 
density contrast minima through exact extrema detection and optimal pairing.
Eliminates windowing artifacts for precise geometric characterization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from pathlib import Path
import dtfe_functions as dtfe
from dtfe_shared import save_plot_to_multiple_paths

# ============================================================================
# Configuration Section
# ============================================================================

BASE_DATA_DIR = "output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/extrema_correlation"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7  # Mpc
AXIS_UNITS = "Mpc"

# Extrema detection parameters
PHI_PERCENTILE_THRESHOLD = 95  # Top 5% of phi values
DELTA_PERCENTILE_THRESHOLD = 5  # Bottom 5% of delta values
FOOTPRINT_SIZE = 3  # Local extrema detection neighborhood

# Matching parameters
MAX_MATCH_DISTANCE = 5.0  # Maximum separation for valid pairing (grid units)
SMOOTHING_SIGMA = 2.0  # Gaussian smoothing before extrema detection

# Visualization parameters
SLICE_PLANES_TO_PLOT = [0, 1, 2]  # 0=YZ, 1=XZ, 2=XY
SLICE_BUFFER = 2  # Tolerance for including extrema near slice plane
DPI = 300

SNAPSHOT_TO_REDSHIFT = dtfe.SNAPSHOT_TO_REDSHIFT

# ============================================================================

SLICE_PLANES = {
    0: {'name': 'yz_plane', 'axis_labels': ('Y', 'Z'), 'axes': (1, 2)},
    1: {'name': 'xz_plane', 'axis_labels': ('X', 'Z'), 'axes': (0, 2)},
    2: {'name': 'xy_plane', 'axis_labels': ('X', 'Y'), 'axes': (0, 1)}
}

# ============================================================================
# Core Analysis Functions
# ============================================================================

def solve_poisson_phi(density_field, box_size, G=1.0, rho_bar=None):
    """
    Solve Poisson's equation in Fourier space: ∇²φ = 4πG ρ̄ δ
    
    Parameters:
        density_field: 3D density array
        box_size: Physical box size in Mpc
        G: Gravitational constant (normalized to 1.0)
        rho_bar: Mean density (computed if not provided)
    
    Returns:
        phi: Gravitational potential field
    """
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
    k2[0,0,0] = np.inf  # Avoid division by zero at DC component
    
    phi_k = -4*np.pi*G*rho_bar * delta_k / k2
    phi = np.real(np.fft.ifftn(phi_k))
    
    return phi

def detect_global_extrema(field, percentile_threshold, detect_maxima=True, 
                         footprint_size=3):
    """
    Identify field extrema using morphological filtering.
    
    Parameters:
        field: 3D scalar field
        percentile_threshold: Intensity threshold for significance
        detect_maxima: If True, detect maxima; if False, detect minima
        footprint_size: Neighborhood size for local extrema detection
    
    Returns:
        coords: Nx3 array of extrema coordinates
        values: N array of field values at extrema
        mask: Binary mask of extrema locations
    """
    if detect_maxima:
        extrema_filter = maximum_filter(field, size=footprint_size, mode='wrap')
        is_extremum = (field == extrema_filter)
        threshold = np.percentile(field, percentile_threshold)
        significant_extrema = is_extremum & (field >= threshold)
    else:
        extrema_filter = minimum_filter(field, size=footprint_size, mode='wrap')
        is_extremum = (field == extrema_filter)
        threshold = np.percentile(field, percentile_threshold)
        significant_extrema = is_extremum & (field <= threshold)
    
    coords = np.array(np.where(significant_extrema)).T
    values = field[significant_extrema]
    
    return coords, values, significant_extrema

def match_extrema_pairs(phi_coords, delta_coords, box_size, max_distance=5.0):
    """
    Compute optimal pairing between phi maxima and delta minima.
    
    Uses k-d tree with periodic boundary conditions for efficient nearest
    neighbor search in cosmological simulation box.
    
    Parameters:
        phi_coords: Mx3 array of phi maxima coordinates
        delta_coords: Nx3 array of delta minima coordinates
        box_size: Simulation box size for periodic boundaries
        max_distance: Maximum separation for valid pairing
    
    Returns:
        matched_pairs: Dictionary containing pairing information
        all_distances: Array of all nearest-neighbor distances
    """
    if len(phi_coords) == 0 or len(delta_coords) == 0:
        return None, np.array([])
    
    # Build k-d tree with periodic boundary conditions
    tree = cKDTree(delta_coords, boxsize=box_size)
    distances, indices = tree.query(phi_coords, k=1)
    
    # Filter by distance threshold
    valid_matches = distances <= max_distance
    
    matched_pairs = {
        'phi_coords': phi_coords[valid_matches],
        'delta_coords': delta_coords[indices[valid_matches]],
        'distances': distances[valid_matches],
        'phi_indices': np.where(valid_matches)[0],
        'delta_indices': indices[valid_matches]
    }
    
    return matched_pairs, distances

def compute_correlation_metrics(matched_pairs, phi_coords, delta_coords):
    """
    Quantify strength and significance of extrema correspondence.
    
    Returns:
        metrics: Dictionary of statistical measures
    """
    n_phi = len(phi_coords)
    n_delta = len(delta_coords)
    
    if matched_pairs is None or len(matched_pairs['distances']) == 0:
        return {
            'n_phi_maxima': n_phi,
            'n_delta_minima': n_delta,
            'n_matched': 0,
            'match_fraction': 0.0,
            'mean_distance': np.nan,
            'median_distance': np.nan,
            'distance_std': np.nan,
            'completeness_phi': 0.0,
            'completeness_delta': 0.0
        }
    
    n_matched = len(matched_pairs['distances'])
    
    metrics = {
        'n_phi_maxima': n_phi,
        'n_delta_minima': n_delta,
        'n_matched': n_matched,
        'match_fraction': n_matched / n_phi if n_phi > 0 else 0.0,
        'mean_distance': np.mean(matched_pairs['distances']),
        'median_distance': np.median(matched_pairs['distances']),
        'distance_std': np.std(matched_pairs['distances']),
        'completeness_phi': n_matched / n_phi if n_phi > 0 else 0.0,
        'completeness_delta': n_matched / n_delta if n_delta > 0 else 0.0
    }
    
    return metrics

# ============================================================================
# Visualization Functions: Option A - Spatial Overlay
# ============================================================================

def plot_extrema_overlay(phi_field, delta_field, matched_pairs, 
                         slice_dim, redshift, box_size, output_dir):
    """
    Visualize spatial distribution of extrema and their pairings.
    
    Creates side-by-side plots showing phi and delta fields with overlaid
    extrema locations and connecting lines between matched pairs.
    """
    plane_info = SLICE_PLANES[slice_dim]
    plane_name = plane_info['name']
    axis_labels = plane_info['axis_labels']
    axes_indices = plane_info['axes']
    
    slice_idx = phi_field.shape[slice_dim] // 2
    
    # Extract 2D slices
    phi_slice = dtfe.extract_2d_slice(phi_field, slice_dim)
    delta_slice = dtfe.extract_2d_slice(delta_field, slice_dim)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    extent = [0, box_size, 0, box_size]
    
    # Left panel: Phi field with maxima
    im1 = axes[0].imshow(phi_slice.T, cmap='RdBu_r', origin='lower',
                         extent=extent, aspect='equal')
    axes[0].set_xlabel(f"{axis_labels[0]} [{AXIS_UNITS}]", fontsize=12)
    axes[0].set_ylabel(f"{axis_labels[1]} [{AXIS_UNITS}]", fontsize=12)
    axes[0].set_title(f'Gravitational Potential φ (z={redshift:.2f})', fontsize=13)
    
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('φ', fontsize=11)
    
    # Right panel: Delta field with minima
    im2 = axes[1].imshow(delta_slice.T, cmap='viridis', origin='lower',
                         extent=extent, aspect='equal')
    axes[1].set_xlabel(f"{axis_labels[0]} [{AXIS_UNITS}]", fontsize=12)
    axes[1].set_ylabel(f"{axis_labels[1]} [{AXIS_UNITS}]", fontsize=12)
    axes[1].set_title(f'Density Contrast δ (z={redshift:.2f})', fontsize=13)
    
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('δ', fontsize=11)
    
    # Overlay extrema if matched pairs exist
    if matched_pairs is not None and len(matched_pairs['distances']) > 0:
        # Filter extrema near this slice
        in_slice = np.abs(matched_pairs['phi_coords'][:, slice_dim] - slice_idx) < SLICE_BUFFER
        
        if np.any(in_slice):
            phi_2d = matched_pairs['phi_coords'][in_slice][:, list(axes_indices)]
            delta_2d = matched_pairs['delta_coords'][in_slice][:, list(axes_indices)]
            
            # Convert grid coordinates to physical coordinates
            grid_to_phys = box_size / phi_field.shape[0]
            phi_2d_phys = phi_2d * grid_to_phys
            delta_2d_phys = delta_2d * grid_to_phys
            
            # Plot on phi panel
            axes[0].scatter(phi_2d_phys[:, 0], phi_2d_phys[:, 1],
                           c='cyan', s=100, marker='o', edgecolors='black',
                           linewidths=1.5, label='φ maxima', alpha=0.9, zorder=5)
            axes[0].scatter(delta_2d_phys[:, 0], delta_2d_phys[:, 1],
                           c='yellow', s=100, marker='s', edgecolors='black',
                           linewidths=1.5, label='δ minima', alpha=0.9, zorder=5)
            
            # Draw connecting lines
            for i in range(len(phi_2d_phys)):
                axes[0].plot([phi_2d_phys[i, 0], delta_2d_phys[i, 0]],
                            [phi_2d_phys[i, 1], delta_2d_phys[i, 1]],
                            'w-', alpha=0.5, linewidth=1.5, zorder=4)
            
            axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            # Plot on delta panel
            axes[1].scatter(delta_2d_phys[:, 0], delta_2d_phys[:, 1],
                           c='yellow', s=100, marker='s', edgecolors='black',
                           linewidths=1.5, label='δ minima', alpha=0.9, zorder=5)
            axes[1].legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    for ax in axes:
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    filename = f"extrema_overlay_{plane_name}_z{redshift:.2f}.png"
    filepath = output_dir / plane_name / filename
    save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

# ============================================================================
# Visualization Functions: Option B - Distance Distribution
# ============================================================================

def plot_distance_distribution(all_distances, matched_pairs, max_distance, 
                               redshift, output_dir):
    """
    Characterize spatial separation between extrema populations.
    
    Generates histogram and cumulative distribution of nearest-neighbor
    distances to quantify correlation strength statistically.
    """
    if matched_pairs is None or len(matched_pairs['distances']) == 0:
        print(f"  No matched pairs found for distance distribution at z={redshift:.2f}")
        return
    
    matched_distances = matched_pairs['distances']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Histogram of matched pair distances
    n_bins = min(50, len(matched_distances) // 10 + 1)
    axes[0].hist(matched_distances, bins=n_bins, density=True,
                alpha=0.75, color='steelblue', edgecolor='black', linewidth=1.2)
    
    mean_dist = np.mean(matched_distances)
    median_dist = np.median(matched_distances)
    
    axes[0].axvline(mean_dist, color='red', linestyle='--', 
                   linewidth=2.5, label=f'Mean: {mean_dist:.2f}')
    axes[0].axvline(median_dist, color='orange', linestyle='--', 
                   linewidth=2.5, label=f'Median: {median_dist:.2f}')
    
    axes[0].set_xlabel('Separation Distance [grid units]', fontsize=12)
    axes[0].set_ylabel('Probability Density', fontsize=12)
    axes[0].set_title(f'Distribution of φ-δ Extrema Separations (z={redshift:.2f})', 
                     fontsize=13)
    axes[0].legend(fontsize=11, framealpha=0.95)
    axes[0].grid(alpha=0.3, linestyle=':', linewidth=0.8)
    
    # Right panel: Cumulative distribution function
    sorted_dist = np.sort(matched_distances)
    cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
    
    axes[1].plot(sorted_dist, cdf, linewidth=2.5, color='darkgreen', 
                label='Matched pairs CDF')
    axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    axes[1].axvline(max_distance, color='red', linestyle='--', 
                   label=f'Threshold: {max_distance}', linewidth=2)
    
    # Mark quartiles
    q25_idx = int(0.25 * len(sorted_dist))
    q75_idx = int(0.75 * len(sorted_dist))
    axes[1].plot([sorted_dist[q25_idx], sorted_dist[q75_idx]], [0.25, 0.75],
                'o', color='purple', markersize=8, label='Q1-Q3', zorder=5)
    
    axes[1].set_xlabel('Separation Distance [grid units]', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].set_title(f'CDF of Matched Pair Distances (z={redshift:.2f})', fontsize=13)
    axes[1].legend(fontsize=11, framealpha=0.95)
    axes[1].grid(alpha=0.3, linestyle=':', linewidth=0.8)
    axes[1].set_xlim(0, max(sorted_dist[-1] * 1.1, max_distance * 1.2))
    
    plt.tight_layout()
    
    filename = f"distance_distribution_z{redshift:.2f}.png"
    filepath = output_dir / filename
    save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

# ============================================================================
# Visualization Functions: Option C - Field Value Correlation
# ============================================================================

def plot_field_value_correlation(phi_values, delta_values, matched_pairs, 
                                redshift, output_dir):
    """
    Examine relationship between extrema field intensities.
    
    Tests whether stronger phi maxima preferentially pair with deeper
    delta minima through scatter plot and correlation coefficient.
    """
    if matched_pairs is None or len(matched_pairs['distances']) == 0:
        print(f"  No matched pairs found for field correlation at z={redshift:.2f}")
        return
    
    # Extract values for matched pairs
    matched_phi = phi_values[matched_pairs['phi_indices']]
    matched_delta = delta_values[matched_pairs['delta_indices']]
    matched_distances = matched_pairs['distances']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left panel: Scatter with distance coloring
    scatter = axes[0].scatter(matched_phi, matched_delta,
                             c=matched_distances, cmap='plasma',
                             s=60, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    # Compute Pearson correlation
    corr, p_value = pearsonr(matched_phi, matched_delta)
    
    axes[0].set_xlabel('φ Value at Maxima', fontsize=12)
    axes[0].set_ylabel('δ Value at Paired Minima', fontsize=12)
    axes[0].set_title(f'Field Intensity Correlation (z={redshift:.2f})\n' + 
                     f'Pearson r = {corr:.4f}, p = {p_value:.2e}', fontsize=13)
    axes[0].grid(alpha=0.3, linestyle=':', linewidth=0.8)
    
    cbar = plt.colorbar(scatter, ax=axes[0], label='Pair Separation [grid units]')
    cbar.ax.tick_params(labelsize=10)
    
    # Add regression line if correlation is significant
    if p_value < 0.05:
        z = np.polyfit(matched_phi, matched_delta, 1)
        p = np.poly1d(z)
        x_line = np.linspace(matched_phi.min(), matched_phi.max(), 100)
        axes[0].plot(x_line, p(x_line), 'r--', linewidth=2.5, 
                    label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.3f}', alpha=0.8)
        axes[0].legend(fontsize=10, framealpha=0.95)
    
    # Right panel: Hexbin density plot
    hexbin = axes[1].hexbin(matched_phi, matched_delta, gridsize=40,
                           cmap='YlOrRd', mincnt=1, alpha=0.8,
                           edgecolors='black', linewidths=0.3)
    
    axes[1].set_xlabel('φ Value at Maxima', fontsize=12)
    axes[1].set_ylabel('δ Value at Paired Minima', fontsize=12)
    axes[1].set_title(f'Density Distribution (z={redshift:.2f})', fontsize=13)
    axes[1].grid(alpha=0.3, linestyle=':', linewidth=0.8)
    
    cbar2 = plt.colorbar(hexbin, ax=axes[1], label='Pair Count per Bin')
    cbar2.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    filename = f"field_correlation_z{redshift:.2f}.png"
    filepath = output_dir / filename
    save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

# ============================================================================
# Combined Overview Visualization
# ============================================================================

def create_combined_overview(phi_field, delta_field, matched_pairs, 
                            redshift, box_size, output_dir):
    """
    Generate multi-panel overview showing all three slice planes together.
    """
    extent = [0, box_size, 0, box_size]
    
    # Extract slices for all planes
    phi_slices = []
    delta_slices = []
    for slice_dim in SLICE_PLANES_TO_PLOT:
        phi_slices.append(dtfe.extract_2d_slice(phi_field, slice_dim))
        delta_slices.append(dtfe.extract_2d_slice(delta_field, slice_dim))
    
    # Create figure with two rows (phi and delta)
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    
    # Top row: Phi fields
    for i, slice_dim in enumerate(SLICE_PLANES_TO_PLOT):
        plane_info = SLICE_PLANES[slice_dim]
        
        im = axes[0, i].imshow(phi_slices[i].T, cmap='RdBu_r', origin='lower',
                              extent=extent, aspect='equal')
        axes[0, i].set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=11)
        axes[0, i].set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=11)
        axes[0, i].set_title(f"{plane_info['name'].replace('_', ' ').title()}", fontsize=12)
        axes[0, i].set_aspect('equal')
        
        cbar = plt.colorbar(im, ax=axes[0, i], label='φ', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
    
    # Bottom row: Delta fields
    for i, slice_dim in enumerate(SLICE_PLANES_TO_PLOT):
        plane_info = SLICE_PLANES[slice_dim]
        
        im = axes[1, i].imshow(delta_slices[i].T, cmap='viridis', origin='lower',
                              extent=extent, aspect='equal')
        axes[1, i].set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=11)
        axes[1, i].set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=11)
        axes[1, i].set_title(f"{plane_info['name'].replace('_', ' ').title()}", fontsize=12)
        axes[1, i].set_aspect('equal')
        
        cbar = plt.colorbar(im, ax=axes[1, i], label='δ', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
    
    fig.suptitle(f'Gravitational Potential and Density Contrast Fields (z={redshift:.2f})', 
                fontsize=15, y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    filename = f"combined_fields_z{redshift:.2f}.png"
    filepath = output_dir / filename
    save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def create_output_directories(output_dir):
    """Create directory structure for outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plane_dirs = {}
    for slice_dim in SLICE_PLANES_TO_PLOT:
        plane_name = SLICE_PLANES[slice_dim]['name']
        plane_dir = output_path / plane_name
        plane_dir.mkdir(parents=True, exist_ok=True)
        plane_dirs[slice_dim] = plane_dir
    
    return output_path, plane_dirs

def analyze_snapshot(snapshot, redshift, plane_dirs, output_dir):
    """
    Execute complete analysis pipeline for single snapshot.
    
    Workflow:
    1. Load density field and compute gravitational potential
    2. Detect global extrema in phi and delta fields
    3. Match extrema pairs using nearest-neighbor algorithm
    4. Generate all visualization types (A, B, C)
    5. Compute and report correlation metrics
    """
    print(f"\n{'='*70}")
    print(f"Processing snapshot {snapshot} (z={redshift:.2f})")
    print(f"{'='*70}")
    
    data_path = Path(BASE_DATA_DIR) / f'snapdir_{snapshot}'
    density_file = data_path / 'output.a_den'
    
    if not density_file.exists():
        print(f"ERROR: Density file not found: {density_file}")
        return False
    
    try:
        # Load density field
        print("Loading density field...")
        field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)
        density_field = dtfe.load_binary_field(density_file, field_shape, 
                                              num_components=1, dtype=np.float32,
                                              try_infer_shape=True)
        
        # Compute density contrast
        print("Computing density contrast δ...")
        rho_bar = np.mean(density_field)
        delta = (density_field - rho_bar) / rho_bar
        
        # Compute gravitational potential
        print("Solving Poisson equation for φ...")
        phi = solve_poisson_phi(density_field, BOX_SIZE, G=1.0)
        
        # Apply smoothing before extrema detection
        print(f"Applying Gaussian smoothing (σ={SMOOTHING_SIGMA})...")
        phi_smooth = dtfe.smooth_field(phi, sigma=SMOOTHING_SIGMA)
        delta_smooth = dtfe.smooth_field(delta, sigma=SMOOTHING_SIGMA)
        
        # Detect extrema
        print(f"Detecting phi maxima (top {100-PHI_PERCENTILE_THRESHOLD}%)...")
        phi_coords, phi_values, phi_mask = detect_global_extrema(
            phi_smooth, PHI_PERCENTILE_THRESHOLD, 
            detect_maxima=True, footprint_size=FOOTPRINT_SIZE
        )
        print(f"  Found {len(phi_coords)} phi maxima")
        
        print(f"Detecting delta minima (bottom {DELTA_PERCENTILE_THRESHOLD}%)...")
        delta_coords, delta_values, delta_mask = detect_global_extrema(
            delta_smooth, DELTA_PERCENTILE_THRESHOLD,
            detect_maxima=False, footprint_size=FOOTPRINT_SIZE
        )
        print(f"  Found {len(delta_coords)} delta minima")
        
        # Match extrema pairs
        print(f"Matching extrema pairs (max distance = {MAX_MATCH_DISTANCE} grid units)...")
        matched_pairs, all_distances = match_extrema_pairs(
            phi_coords, delta_coords, FIELD_RESOLUTION, MAX_MATCH_DISTANCE
        )
        
        # Compute correlation metrics
        metrics = compute_correlation_metrics(matched_pairs, phi_coords, delta_coords)
        
        print(f"\nCorrelation Statistics:")
        print(f"  Total φ maxima:        {metrics['n_phi_maxima']}")
        print(f"  Total δ minima:        {metrics['n_delta_minima']}")
        print(f"  Matched pairs:         {metrics['n_matched']}")
        print(f"  Match fraction:        {metrics['match_fraction']:.3f}")
        print(f"  Mean separation:       {metrics['mean_distance']:.2f} grid units")
        print(f"  Median separation:     {metrics['median_distance']:.2f} grid units")
        print(f"  Separation std dev:    {metrics['distance_std']:.2f} grid units")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # Option A: Spatial overlay for each plane
        print("  Creating spatial overlay plots (Option A)...")
        for slice_dim in SLICE_PLANES_TO_PLOT:
            plot_extrema_overlay(phi_smooth, delta_smooth, matched_pairs,
                               slice_dim, redshift, BOX_SIZE, output_dir)
        
        # Option B: Distance distribution
        print("  Creating distance distribution plots (Option B)...")
        plot_distance_distribution(all_distances, matched_pairs, MAX_MATCH_DISTANCE,
                                  redshift, output_dir)
        
        # Option C: Field value correlation
        print("  Creating field value correlation plots (Option C)...")
        plot_field_value_correlation(phi_values, delta_values, matched_pairs,
                                    redshift, output_dir)
        
        # Combined overview
        print("  Creating combined overview plots...")
        create_combined_overview(phi_smooth, delta_smooth, matched_pairs,
                                redshift, BOX_SIZE, output_dir)
        
        print(f"\n✓ Completed snapshot {snapshot}")
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR analyzing snapshot {snapshot}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main execution function for extrema correlation analysis.
    
    Processes all snapshots defined in SNAPSHOT_TO_REDSHIFT dictionary,
    generating comprehensive visualization outputs for each redshift.
    """
    print("\n" + "="*70)
    print(" Direct Extrema Correlation Analysis")
    print(" Phi-Delta Correspondence in Cosmological Simulations")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Field resolution:      {FIELD_RESOLUTION}³")
    print(f"  Box size:              {BOX_SIZE} Mpc")
    print(f"  Phi threshold:         Top {100-PHI_PERCENTILE_THRESHOLD}%")
    print(f"  Delta threshold:       Bottom {DELTA_PERCENTILE_THRESHOLD}%")
    print(f"  Footprint size:        {FOOTPRINT_SIZE}³ voxels")
    print(f"  Max match distance:    {MAX_MATCH_DISTANCE} grid units")
    print(f"  Smoothing sigma:       {SMOOTHING_SIGMA}")
    print(f"  Total snapshots:       {len(SNAPSHOT_TO_REDSHIFT)}")
    
    # Create output directories
    output_path, plane_dirs = create_output_directories(OUTPUT_DIR)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    
    # Process all snapshots
    successful_analyses = []
    failed_analyses = []
    
    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        success = analyze_snapshot(snapshot, redshift, plane_dirs, output_path)
        
        if success:
            successful_analyses.append((snapshot, redshift))
        else:
            failed_analyses.append((snapshot, redshift))
    
    # Summary report
    print("\n" + "="*70)
    print(" Analysis Summary")
    print("="*70)
    print(f"Total snapshots processed:  {len(SNAPSHOT_TO_REDSHIFT)}")
    print(f"Successful analyses:        {len(successful_analyses)}")
    print(f"Failed analyses:            {len(failed_analyses)}")
    
    if successful_analyses:
        print(f"\nSuccessfully analyzed:")
        for snapshot, redshift in successful_analyses:
            print(f"  ✓ Snapshot {snapshot:3d} (z={redshift:.2f})")
    
    if failed_analyses:
        print(f"\nFailed to analyze:")
        for snapshot, redshift in failed_analyses:
            print(f"  ✗ Snapshot {snapshot:3d} (z={redshift:.2f})")
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()