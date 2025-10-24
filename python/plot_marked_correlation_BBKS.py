"""
Marked Correlation Analysis for Void Ellipsoids

Computes spatial correlations of void shape parameters (ellipticity, prolateness,
orientation) using adaptive binning and parameter selection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, minimum_filter
from scipy.stats import scoreatpercentile
import warnings
from pathlib import Path
import dtfe_functions as dtfe
from dtfe_shared import save_plot_to_multiple_paths
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration Section
# ============================================================================

BASE_DATA_DIR = "output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/correlation_analysis"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7  # Mpc
SMOOTHING_SIGMA = 10.0
FOOTPRINT_SIZE = 3

# Correlation analysis parameters
MAX_CORRELATION_RANGE = 51.7  # Mpc - maximum distance for correlation analysis
MIN_CORRELATION_RANGE = 0.0   # Mpc - minimum distance for correlation analysis

# Snapshot to redshift mapping
SNAPSHOT_TO_REDSHIFT = dtfe.SNAPSHOT_TO_REDSHIFT

DPI = 300

# ============================================================================

class EllipsoidProperties:
    """Extract and store ellipsoid properties from Hessian eigendata."""
    
    def __init__(self, eigenvalue_info, box_size, grid_size):
        self.box_size = box_size
        self.grid_size = grid_size
        self.cell_size = box_size / grid_size
        self.positions = []
        self.semi_axes = []
        self.orientations = []
        self.aspect_ratios = []
        self.ellipticities = []
        self.prolateness = []
        self.valid_indices = []
        
        self._extract_properties(eigenvalue_info)
    
    def _extract_properties(self, eigenvalue_info):
        """Extract ellipsoid properties from eigenvalue information."""
        for i, info in enumerate(eigenvalue_info):
            coords = np.array(info['coords_3d'])
            eigenvals = info['eigenvalues']
            eigenvecs = info['eigenvectors']
            
            if np.any(np.abs(eigenvals) < 1e-10):
                continue
            
            if np.any(eigenvals < -1e-8):
                continue
            
            eigenvals_abs = np.abs(eigenvals) + 1e-12
            sorted_indices = np.argsort(eigenvals_abs)[::-1]
            eigenvals_sorted = eigenvals_abs[sorted_indices]
            eigenvecs_sorted = eigenvecs[:, sorted_indices]
            
            scale_factor = 2.0 * self.cell_size
            semi_axes_raw = scale_factor / np.sqrt(eigenvals_sorted)
            semi_axes = np.sort(semi_axes_raw)[::-1]
            
            min_size = 0.1
            max_size = 10.0
            
            if np.min(semi_axes) < min_size or np.max(semi_axes) > max_size:
                continue
            
            if semi_axes[0] / semi_axes[2] > 10.0:
                continue
            
            position = coords * self.cell_size
            
            a, b, c = semi_axes
            
            aspect_ratio_1 = b / a
            aspect_ratio_2 = c / a
            ellipticity = 1 - c / a
            
            if a**2 - c**2 > 1e-12:
                prolateness = (a**2 - b**2) / (a**2 - c**2)
            else:
                prolateness = 0
            
            self.positions.append(position)
            self.semi_axes.append(semi_axes)
            self.orientations.append(eigenvecs_sorted)
            self.aspect_ratios.append([aspect_ratio_1, aspect_ratio_2])
            self.ellipticities.append(ellipticity)
            self.prolateness.append(prolateness)
            self.valid_indices.append(i)
        
        self.positions = np.array(self.positions)
        self.semi_axes = np.array(self.semi_axes)
        self.orientations = np.array(self.orientations)
        self.aspect_ratios = np.array(self.aspect_ratios)
        self.ellipticities = np.array(self.ellipticities)
        self.prolateness = np.array(self.prolateness)

class AdaptiveParams:
    """Adaptive parameter selection for correlation analysis."""
    
    def __init__(self, distances, n_ellipsoids, box_size):
        self.distances = distances
        self.n_ellipsoids = n_ellipsoids
        self.n_pairs = len(distances)
        self.box_size = box_size
        self.stats = self._compute_stats()
        
    def _compute_stats(self):
        """Compute distance distribution statistics."""
        if len(self.distances) == 0:
            return {}
            
        stats = {
            'min': np.min(self.distances),
            'max': np.max(self.distances),
            'mean': np.mean(self.distances),
            'median': np.median(self.distances),
            'q25': scoreatpercentile(self.distances, 25),
            'q75': scoreatpercentile(self.distances, 75),
            'q90': scoreatpercentile(self.distances, 90),
            'q95': scoreatpercentile(self.distances, 95)
        }
        
        stats['typical_scale'] = stats['median']
        stats['max_scale'] = min(stats['q90'], self.box_size / 4.0)
        
        return stats
    
    def get_range(self, correlation_type='marked'):
        """Determine optimal distance range - limited to physically reasonable correlation scales."""
        # Use configured range for void correlations
        min_r = MIN_CORRELATION_RANGE
        max_r = MAX_CORRELATION_RANGE

        if len(self.distances) > 0:
            d_min = np.min(self.distances)
            d_max = np.max(self.distances)
            median_dist = np.median(self.distances)
            print(f"    Distance stats: min={d_min:.2f}, median={median_dist:.2f}, max={d_max:.2f} Mpc")

        print(f"    Using correlation range: {min_r:.2f} - {max_r:.2f} Mpc")
        return min_r, max_r
    
    def get_bins(self, min_r, max_r, target_pairs=5):
        """Determine optimal binning for smooth correlation function."""
        if len(self.distances) == 0:
            return 25, 2, np.linspace(MIN_CORRELATION_RANGE, MAX_CORRELATION_RANGE, 26)

        in_range = np.sum((self.distances >= min_r) & (self.distances <= max_r))

        # Use linear binning for smoother, more interpretable correlations
        n_bins = 30  # Fixed number for consistency

        # Use linear bins which work better with sliding window approach
        r_bins = np.linspace(min_r, max_r, n_bins + 1)

        print(f"    Using {n_bins} bins from {min_r:.2f} to {max_r:.2f} Mpc ({in_range} pairs in range)")
        return n_bins, 2, r_bins

class CorrelationAnalysis:
    """Enhanced correlation analysis with adaptive parameters."""
    
    def __init__(self, ellipsoid_props, adaptive=True):
        self.props = ellipsoid_props
        self.distances = None
        self.pair_indices = None
        self.adaptive = adaptive
        self._compute_distances()
        
        if self.adaptive and len(self.props.positions) > 1:
            self.params = AdaptiveParams(self.distances, len(self.props.positions), 
                                       self.props.box_size)
        else:
            self.params = None
    
    def _compute_distances(self):
        """Compute pairwise distances between ellipsoids."""
        if len(self.props.positions) < 2:
            self.distances = np.array([])
            self.pair_indices = np.array([])
            return

        positions = self.props.positions
        n_points = len(positions)

        distances = []
        pair_indices = []

        for i in range(n_points):
            for j in range(i + 1, n_points):
                diff = positions[j] - positions[i]
                diff = diff - self.props.box_size * np.round(diff / self.props.box_size)
                dist = np.linalg.norm(diff)
                distances.append(dist)
                pair_indices.append([i, j])

        self.distances = np.array(distances)
        self.pair_indices = np.array(pair_indices)

        # Print diagnostic info
        if len(self.distances) > 0:
            print(f"    Distance distribution: min={np.min(self.distances):.2f}, "
                  f"median={np.median(self.distances):.2f}, max={np.max(self.distances):.2f} Mpc")
            print(f"    Pairs in [0-2] Mpc: {np.sum(self.distances < 2)}, "
                  f"[2-4] Mpc: {np.sum((self.distances >= 2) & (self.distances < 4))}, "
                  f"[4-6] Mpc: {np.sum((self.distances >= 4) & (self.distances < 6))}")
    
    def get_analysis_params(self, correlation_type='marked'):
        """Get analysis parameters (adaptive or default)."""
        if self.params is None or not self.adaptive:
            # Default parameters limited to reasonable correlation scale
            return {
                'min_r': MIN_CORRELATION_RANGE,
                'max_r': MAX_CORRELATION_RANGE,
                'n_bins': 25,
                'min_pairs': 2,
                'r_bins': np.linspace(MIN_CORRELATION_RANGE, MAX_CORRELATION_RANGE, 26)  # Linear bins for smoother curves
            }

        min_r, max_r = self.params.get_range(correlation_type)
        n_bins, min_pairs, r_bins = self.params.get_bins(min_r, max_r)

        return {
            'min_r': min_r,
            'max_r': max_r,
            'n_bins': n_bins,
            'min_pairs': min_pairs,
            'r_bins': r_bins
        }
    
    def _compute_correlation(self, marks, r_bins, min_pairs):
        """
        Compute marked correlation using Pearson correlation coefficient.

        This is more statistically robust than the simple product-mean approach,
        as it properly accounts for variance and gives values in [-1, 1].
        """
        if len(marks) < 2:
            return np.zeros(len(r_bins) - 1), np.zeros(len(r_bins) - 1), np.zeros(len(r_bins) - 1)

        if len(self.distances) == 0:
            return np.zeros(len(r_bins) - 1), np.zeros(len(r_bins) - 1), np.zeros(len(r_bins) - 1)

        # Use sliding window approach for smooth correlation function
        # Create dense evaluation points
        n_eval_points = 60
        r_eval = np.linspace(r_bins[0], r_bins[-1], n_eval_points)

        # Adaptive window width - wider for longer distances
        window_width_base = (r_bins[-1] - r_bins[0]) / 8.0

        correlations_eval = []
        counts_eval = []

        for r_center in r_eval:
            # Adaptive window: wider at larger separations
            window_width = window_width_base * (1.0 + r_center / r_bins[-1])

            r_min = max(r_bins[0], r_center - window_width / 2)
            r_max = min(r_bins[-1], r_center + window_width / 2)

            mask = (self.distances >= r_min) & (self.distances <= r_max)
            window_pairs = self.pair_indices[mask]

            if len(window_pairs) >= 3:
                marks_i = marks[window_pairs[:, 0]]
                marks_j = marks[window_pairs[:, 1]]

                # Pearson correlation coefficient
                # More robust than simple mean(m_i * m_j) / mean(m)^2
                if np.std(marks_i) > 1e-10 and np.std(marks_j) > 1e-10:
                    # Compute covariance
                    mean_i = np.mean(marks_i)
                    mean_j = np.mean(marks_j)
                    cov = np.mean((marks_i - mean_i) * (marks_j - mean_j))
                    std_i = np.std(marks_i)
                    std_j = np.std(marks_j)

                    correlation = cov / (std_i * std_j)
                    correlation = np.clip(correlation, -1.0, 1.0)
                else:
                    correlation = 0.0

                correlations_eval.append(correlation)
                counts_eval.append(len(window_pairs))
            else:
                correlations_eval.append(0.0)
                counts_eval.append(0)

        correlations_eval = np.array(correlations_eval)
        counts_eval = np.array(counts_eval)

        # Apply moderate smoothing (less aggressive since Pearson is already robust)
        from scipy.ndimage import gaussian_filter1d
        valid_mask = counts_eval > 0
        if np.sum(valid_mask) > 3:
            correlations_smooth = gaussian_filter1d(correlations_eval, sigma=2.0, mode='nearest')
        else:
            correlations_smooth = correlations_eval

        # Bin centers for output (linear spacing)
        bin_centers = (r_bins[:-1] + r_bins[1:]) / 2

        # Interpolate smoothed correlation onto bin centers
        correlations = np.interp(bin_centers, r_eval, correlations_smooth)

        # Estimate pair counts for each output bin
        pair_counts = []
        for i in range(len(r_bins) - 1):
            mask = (self.distances >= r_bins[i]) & (self.distances < r_bins[i+1])
            pair_counts.append(np.sum(mask))
        pair_counts = np.array(pair_counts)

        return bin_centers, correlations, pair_counts
    
    def ellipticity_correlation(self):
        """Compute ellipticity correlation."""
        params = self.get_analysis_params('marked')
        return self._compute_correlation(self.props.ellipticities, 
                                       params['r_bins'], params['min_pairs'])
    
    def prolateness_correlation(self):
        """Compute prolateness correlation."""
        params = self.get_analysis_params('marked')
        return self._compute_correlation(self.props.prolateness, 
                                       params['r_bins'], params['min_pairs'])
    
    def orientation_correlation(self):
        """
        Compute orientation correlation using alignment measure.

        Uses the alignment parameter: |cos(theta)| where theta is the angle
        between major axes. This is better than using raw angles because:
        - It's invariant to 180 degree rotations (ellipsoids have no head/tail)
        - Values in [0,1]: 1 = aligned, 0 = perpendicular
        - More physically meaningful for void shapes
        """
        if len(self.props.orientations) < 2:
            return np.array([]), np.array([]), np.array([])

        params = self.get_analysis_params('orientation')
        r_bins = params['r_bins']

        # Create dense evaluation points
        n_eval_points = 60
        r_eval = np.linspace(r_bins[0], r_bins[-1], n_eval_points)
        window_width_base = (r_bins[-1] - r_bins[0]) / 8.0

        correlations_eval = []
        counts_eval = []

        for r_center in r_eval:
            window_width = window_width_base * (1.0 + r_center / r_bins[-1])
            r_min = max(r_bins[0], r_center - window_width / 2)
            r_max = min(r_bins[-1], r_center + window_width / 2)

            mask = (self.distances >= r_min) & (self.distances <= r_max)
            window_pairs = self.pair_indices[mask]

            if len(window_pairs) >= 3:
                alignments = []
                for idx_i, idx_j in window_pairs:
                    # Get major axes (first eigenvector)
                    axis_i = self.props.orientations[idx_i][:, 0]
                    axis_j = self.props.orientations[idx_j][:, 0]

                    # Compute alignment: |cos(theta)|
                    # (absolute value because ellipsoids have no direction)
                    cos_theta = np.abs(np.dot(axis_i, axis_j))
                    cos_theta = np.clip(cos_theta, 0.0, 1.0)

                    alignments.append(cos_theta)

                # Mean alignment in this distance bin
                mean_alignment = np.mean(alignments)

                # Convert to correlation:
                # Random orientations give <|cos(theta)|> = 0.5
                # So correlation = (alignment - 0.5) / 0.5
                # This gives: 1 = perfect alignment, 0 = random, -1 = perpendicular
                correlation = (mean_alignment - 0.5) / 0.5
                correlation = np.clip(correlation, -1.0, 1.0)

                correlations_eval.append(correlation)
                counts_eval.append(len(window_pairs))
            else:
                correlations_eval.append(0.0)
                counts_eval.append(0)

        correlations_eval = np.array(correlations_eval)
        counts_eval = np.array(counts_eval)

        # Apply smoothing
        from scipy.ndimage import gaussian_filter1d
        valid_mask = counts_eval > 0
        if np.sum(valid_mask) > 3:
            correlations_smooth = gaussian_filter1d(correlations_eval, sigma=2.0, mode='nearest')
        else:
            correlations_smooth = correlations_eval

        # Bin centers for output
        bin_centers = (r_bins[:-1] + r_bins[1:]) / 2

        # Interpolate onto bin centers
        correlations = np.interp(bin_centers, r_eval, correlations_smooth)

        # Estimate pair counts
        pair_counts = []
        for i in range(len(r_bins) - 1):
            mask = (self.distances >= r_bins[i]) & (self.distances < r_bins[i+1])
            pair_counts.append(np.sum(mask))
        pair_counts = np.array(pair_counts)

        return bin_centers, correlations, pair_counts

def plot_shape_correlations(analysis_results, param_info, ellipsoid_props, output_path):
    """
    Plot ellipticity and prolateness correlations together.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Extract data
    r_ellip, corr_ellip, counts_ellip = analysis_results.get('ellipticity_correlation', ([], [], []))
    r_prolate, corr_prolate, counts_prolate = analysis_results.get('prolateness_correlation', ([], [], []))
    
    # Plot ellipticity
    if len(r_ellip) > 0:
        ax.plot(r_ellip, corr_ellip, '-', color='green', linewidth=3.0, 
                label='Ellipticity', alpha=0.9)
        marker_step = max(1, len(r_ellip) // 12)
        marker_indices = np.arange(0, len(r_ellip), marker_step)
        ax.scatter(r_ellip[marker_indices], corr_ellip[marker_indices],
                  c='green', s=60, marker='o', edgecolor='white',
                  linewidth=1.5, alpha=0.9, zorder=5)
    
    # Plot prolateness
    if len(r_prolate) > 0:
        ax.plot(r_prolate, corr_prolate, '-', color='purple', linewidth=3.0,
                label='Prolateness', alpha=0.9)
        marker_step = max(1, len(r_prolate) // 12)
        marker_indices = np.arange(0, len(r_prolate), marker_step)
        ax.scatter(r_prolate[marker_indices], corr_prolate[marker_indices],
                  c='purple', s=60, marker='s', edgecolor='white',
                  linewidth=1.5, alpha=0.9, zorder=5)
    
    # Reference line and styling
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.set_xlabel('Distance [Mpc]', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Correlation $\xi(r)$', fontsize=13, fontweight='bold')
    ax.set_title('Void Shape Correlations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=1.0)
    ax.legend(fontsize=12, framealpha=0.9, loc='best')
    
    ax.set_xlim(MIN_CORRELATION_RANGE, MAX_CORRELATION_RANGE + 0.5)
    
    # Set symmetric y-axis limits
    all_corr = []
    if len(r_ellip) > 0:
        all_corr.extend(corr_ellip)
    if len(r_prolate) > 0:
        all_corr.extend(corr_prolate)
    
    if len(all_corr) > 0 and np.max(np.abs(all_corr)) > 0.01:
        y_max = max(0.3, np.max(np.abs(all_corr)) * 1.3)
        ax.set_ylim(-y_max, y_max)
    else:
        ax.set_ylim(-0.3, 0.3)
    
    plt.tight_layout()
    save_plot_to_multiple_paths(fig, output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return fig


def plot_orientation_correlation(analysis_results, param_info, ellipsoid_props, output_path):
    """
    Plot orientation correlation separately.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Extract data
    r_orient, corr_orient, counts_orient = analysis_results.get('orientation_correlation', ([], [], []))
    
    # Plot orientation
    if len(r_orient) > 0:
        ax.plot(r_orient, corr_orient, '-', color='red', linewidth=3.0,
                label='Orientation', alpha=0.9)
        marker_step = max(1, len(r_orient) // 12)
        marker_indices = np.arange(0, len(r_orient), marker_step)
        ax.scatter(r_orient[marker_indices], corr_orient[marker_indices],
                  c='red', s=60, marker='^', edgecolor='white',
                  linewidth=1.5, alpha=0.9, zorder=5)
    
    # Reference line and styling
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.set_xlabel('Distance [Mpc]', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Correlation $\xi(r)$', fontsize=13, fontweight='bold')
    ax.set_title('Void Orientation Correlation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=1.0)
    ax.legend(fontsize=12, framealpha=0.9, loc='best')
    
    ax.set_xlim(MIN_CORRELATION_RANGE, MAX_CORRELATION_RANGE + 0.5)
    
    # Set symmetric y-axis limits
    if len(r_orient) > 0 and np.max(np.abs(corr_orient)) > 0.01:
        y_max = max(0.3, np.max(np.abs(corr_orient)) * 1.3)
        ax.set_ylim(-y_max, y_max)
    else:
        ax.set_ylim(-0.5, 0.5)
    
    plt.tight_layout()
    save_plot_to_multiple_paths(fig, output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return fig


def determine_dynamic_parameters(num_ellipsoids, redshift):
    """
    Determine adaptive parameters based on number of ellipsoids and redshift.

    Parameters
    ----------
    num_ellipsoids : int
        Number of valid ellipsoids found
    redshift : float
        Redshift of snapshot

    Returns
    -------
    dict
        Dictionary with smoothing_sigma and footprint_size
    """
    # Use constant smoothing sigma - no adaptation
    smoothing_sigma = SMOOTHING_SIGMA
    footprint_size = FOOTPRINT_SIZE

    # Only adjust footprint based on number density
    if num_ellipsoids < 100:
        # Too few - decrease footprint to find more
        footprint_size = max(7, footprint_size - 2)
    elif num_ellipsoids > 1000:
        # Too many - increase footprint to be more selective
        footprint_size = min(15, footprint_size + 2)

    # No redshift-based smoothing adjustment - keep constant

    return {
        'smoothing_sigma': smoothing_sigma,
        'footprint_size': footprint_size
    }


def process_snapshot(snapshot, redshift):
    """Process a single snapshot for correlation analysis."""

    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")

    snapshot_dir = Path(BASE_DATA_DIR) / f"snapdir_{snapshot}"
    output_dir = Path(OUTPUT_DIR) / f"snapshot_{snapshot}_z{redshift:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)

    density_file = snapshot_dir / 'output.a_den'

    if not density_file.exists():
        print(f"  Warning: Density file not found")
        return False

    field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)

    try:
        # Initial pass with default parameters to get ellipsoid count
        print(f"  Loading and processing density field")
        density_field = dtfe.load_binary_field(str(density_file), field_shape, num_components=1)
        density_contrast = dtfe.calculate_density_contrast(density_field)
        contrast_smoothed = dtfe.smooth_field(density_contrast, sigma=SMOOTHING_SIGMA)

        print(f"  Finding 3D minima (initial pass)")
        minima_mask_3d, minima_coords_3d = dtfe.find_local_minima(contrast_smoothed, footprint_size=FOOTPRINT_SIZE)
        initial_count = len(minima_coords_3d)

        # Determine dynamic parameters
        dynamic_params = determine_dynamic_parameters(initial_count, redshift)
        smoothing_sigma = dynamic_params['smoothing_sigma']
        footprint_size = dynamic_params['footprint_size']

        print(f"  Initial minima: {initial_count}")
        print(f"  Adaptive parameters: sigma={smoothing_sigma:.1f}, footprint={footprint_size}")

        # Reprocess with adaptive parameters if they differ significantly
        if abs(smoothing_sigma - SMOOTHING_SIGMA) > 1.0 or footprint_size != FOOTPRINT_SIZE:
            contrast_smoothed = dtfe.smooth_field(density_contrast, sigma=smoothing_sigma)
            minima_mask_3d, minima_coords_3d = dtfe.find_local_minima(contrast_smoothed, footprint_size=footprint_size)
            print(f"  Reprocessed minima: {len(minima_coords_3d)}")

        print(f"  Computing Hessian and eigenvalues")
        hessian_components = dtfe.calculate_hessian_fft(contrast_smoothed)
        eigenvalues, eigenvectors = dtfe.compute_eigenvalues_at_minima(hessian_components, minima_coords_3d)
        eigenvalue_info = [
            {'coords_3d': tuple(minima_coords_3d[i]),
             'eigenvalues': eigenvalues[i],
             'eigenvectors': eigenvectors[i]}
            for i in range(len(minima_coords_3d))
        ]

        print(f"  Extracting ellipsoid properties")
        ellipsoid_props = EllipsoidProperties(eigenvalue_info, box_size=BOX_SIZE, grid_size=FIELD_RESOLUTION)

        # If we don't have enough ellipsoids, try with original parameters
        if len(ellipsoid_props.positions) < 10 and (smoothing_sigma != SMOOTHING_SIGMA or footprint_size != FOOTPRINT_SIZE):
            print(f"  Only found {len(ellipsoid_props.positions)} ellipsoids, retrying with default parameters...")
            contrast_smoothed = dtfe.smooth_field(density_contrast, sigma=SMOOTHING_SIGMA)
            minima_mask_3d, minima_coords_3d = dtfe.find_local_minima(contrast_smoothed, footprint_size=FOOTPRINT_SIZE)
            print(f"  Retry found {len(minima_coords_3d)} minima")

            hessian_components = dtfe.calculate_hessian_fft(contrast_smoothed)
            eigenvalues, eigenvectors = dtfe.compute_eigenvalues_at_minima(hessian_components, minima_coords_3d)
            eigenvalue_info = [
                {'coords_3d': tuple(minima_coords_3d[i]),
                 'eigenvalues': eigenvalues[i],
                 'eigenvectors': eigenvectors[i]}
                for i in range(len(minima_coords_3d))
            ]
            ellipsoid_props = EllipsoidProperties(eigenvalue_info, box_size=BOX_SIZE, grid_size=FIELD_RESOLUTION)
            smoothing_sigma = SMOOTHING_SIGMA
            footprint_size = FOOTPRINT_SIZE

        if len(ellipsoid_props.positions) < 2:
            print(f"  Warning: Insufficient ellipsoids for correlation analysis ({len(ellipsoid_props.positions)} found)")
            return False

        print(f"  Found {len(ellipsoid_props.positions)} valid ellipsoids")

        print(f"  Computing correlation functions")
        correlation_analysis = CorrelationAnalysis(ellipsoid_props, adaptive=True)

        param_info = {
            'marked': correlation_analysis.get_analysis_params('marked'),
            'orientation': correlation_analysis.get_analysis_params('orientation')
        }

        analysis_results = {}

        r_centers, ellip_corr, ellip_counts = correlation_analysis.ellipticity_correlation()
        analysis_results['ellipticity_correlation'] = (r_centers, ellip_corr, ellip_counts)

        r_centers, prolate_corr, prolate_counts = correlation_analysis.prolateness_correlation()
        analysis_results['prolateness_correlation'] = (r_centers, prolate_corr, prolate_counts)

        r_centers, orient_corr, orient_counts = correlation_analysis.orientation_correlation()
        analysis_results['orientation_correlation'] = (r_centers, orient_corr, orient_counts)

        # Create separate plots
        print(f"  Creating shape correlation plot")
        shape_output = output_dir / f'shape_correlations_snap{snapshot}_z{redshift:.2f}.png'
        plot_shape_correlations(analysis_results, param_info, ellipsoid_props, shape_output)

        print(f"  Creating orientation correlation plot")
        orient_output = output_dir / f'orientation_correlation_snap{snapshot}_z{redshift:.2f}.png'
        plot_orientation_correlation(analysis_results, param_info, ellipsoid_props, orient_output)

        # Save summary statistics
        summary_file = output_dir / f'correlation_summary_z{redshift:.2f}.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Void Correlation Analysis Summary\n")
            f.write(f"Snapshot: {snapshot}, Redshift: {redshift:.2f}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  Smoothing sigma: {smoothing_sigma:.1f}\n")
            f.write(f"  Footprint size: {footprint_size}\n")
            f.write(f"  Grid resolution: {FIELD_RESOLUTION}^3\n\n")
            f.write(f"Results:\n")
            f.write(f"  Ellipsoids analyzed: {len(ellipsoid_props.positions)}\n")
            f.write(f"  Mean ellipticity: {np.mean(ellipsoid_props.ellipticities):.3f} +/- {np.std(ellipsoid_props.ellipticities):.3f}\n")
            f.write(f"  Mean prolateness: {np.mean(ellipsoid_props.prolateness):.3f} +/- {np.std(ellipsoid_props.prolateness):.3f}\n")

        print(f"  Output saved to: {output_dir}")
        return True

    except Exception as e:
        print(f"  Error processing snapshot {snapshot}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main analysis function - processes all snapshots."""

    print("Starting adaptive ellipsoid correlation analysis")
    print(f"Data directory: {BASE_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing {len(SNAPSHOT_TO_REDSHIFT)} snapshots")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    processed_count = 0
    failed_snapshots = []

    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        if process_snapshot(snapshot, redshift):
            processed_count += 1
        else:
            failed_snapshots.append((snapshot, redshift))

    print(f"\n{'='*60}")
    print(f"Analysis complete")
    print(f"Successfully processed: {processed_count}/{len(SNAPSHOT_TO_REDSHIFT)} snapshots")

    if failed_snapshots:
        print(f"Failed snapshots: {len(failed_snapshots)}")
        for snapshot, redshift in failed_snapshots:
            print(f"  Snapshot {snapshot} (z={redshift:.2f})")

    print(f"\nOutput saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()