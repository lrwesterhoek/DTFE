"""
3D Void Shape Analysis with Ellipse Fitting

Identifies 3D local minima in density fields, computes Hessian eigenvalues,
and fits ellipses to visualize void shapes on 2D slices.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
import dtfe_functions as dtfe
from dtfe_shared import save_plot_to_multiple_paths

# ============================================================================
# Configuration Section
# ============================================================================

BASE_DATA_DIR = "output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/void_analysis_plots"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7  # Mpc
SLICE_DIM = 0  # Which dimension to slice (0=YZ, 1=XZ, 2=XY)

# Analysis parameters
SMOOTHING_SIGMA = 10.0
FOOTPRINT_SIZE = 11  # Larger = fewer, more separated minima
SCALE_FACTOR = 0.1  # Controls ellipse size
DENSITY_THRESHOLD = -0.1  # Only keep deep voids
USE_NMS = False  # Additional non-maximum suppression

# Snapshot to redshift mapping
SNAPSHOT_TO_REDSHIFT = dtfe.SNAPSHOT_TO_REDSHIFT

DPI = 600  # High resolution for PNG output

# ============================================================================

def project_3d_minima_to_slice(minima_3d, slice_dim, slice_index, tolerance=5):
    """Project 3D minima coordinates to 2D slice."""
    if len(minima_3d) == 0:
        return np.array([]).reshape(0, 2)
    
    slice_distances = np.abs(minima_3d[:, slice_dim] - slice_index)
    close_minima = minima_3d[slice_distances <= tolerance]
    
    if len(close_minima) == 0:
        return np.array([]).reshape(0, 2)
    
    if slice_dim == 0:
        projected = close_minima[:, [1, 2]]
    elif slice_dim == 1:
        projected = close_minima[:, [0, 2]]
    else:
        projected = close_minima[:, [0, 1]]
    
    return projected

def merge_close_minima(minima_2d, contrast_slice, min_separation=8):
    """Merge minima that are too close together."""
    if len(minima_2d) == 0:
        return minima_2d
    
    densities = np.array([contrast_slice[int(m[0]), int(m[1])] for m in minima_2d])
    sorted_indices = np.argsort(densities)
    
    kept_minima = []
    
    for idx in sorted_indices:
        current_minimum = minima_2d[idx]
        
        too_close = False
        for kept_minimum in kept_minima:
            distance = np.sqrt((current_minimum[0] - kept_minimum[0])**2 + 
                             (current_minimum[1] - kept_minimum[1])**2)
            if distance < min_separation:
                too_close = True
                break
        
        if not too_close:
            kept_minima.append(current_minimum)
    
    return np.array(kept_minima) if kept_minima else np.array([]).reshape(0, 2)

def create_density_contrast_transform():
    """Create symmetric logarithmic transformation for density contrast visualization."""
    def transform_contrast(delta):
        # Symmetric log transformation that handles both positive and negative values
        # sign(x) * log10(1 + |x|) gives smooth transition through zero
        return np.sign(delta) * np.log10(1 + np.abs(delta))

    return transform_contrast

def create_custom_colorbar(im, ax, contrast_slice, transform_func, vmin=None, vmax=None):
    """Create custom colorbar with original density contrast values."""
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25)

    contrast_transformed = transform_func(contrast_slice)

    # Use provided vmin/vmax if available (from clipping), otherwise use data range
    if vmin is None or vmax is None:
        vmin = contrast_transformed.min()
        vmax = contrast_transformed.max()

    n_ticks = 11
    tick_positions = np.linspace(vmin, vmax, n_ticks)

    min_contrast, max_contrast = contrast_slice.min(), contrast_slice.max()
    test_deltas = np.linspace(min_contrast, max_contrast, 10000)
    test_transformed = transform_func(test_deltas)

    original_values = []
    for tick_pos in tick_positions:
        idx = np.argmin(np.abs(test_transformed - tick_pos))
        original_values.append(test_deltas[idx])

    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f'{val:.2f}' if abs(val) > 0.01 else f'{val:.3f}' for val in original_values])
    cbar.set_label('Density Contrast δ = (ρ - ρ̄)/ρ̄', fontsize=14, labelpad=15)

    return cbar

def create_contour_levels(contrast_slice, transform_func, num_contours=20):
    """Create contour levels with more emphasis on void regions."""
    min_contrast = contrast_slice.min()
    max_contrast = contrast_slice.max()

    # More contours in void regions (negative delta) where we have more interest
    # 75% of contours for voids, 25% for overdensities
    num_void_contours = int(num_contours * 0.75)
    num_overdense_contours = num_contours - num_void_contours

    void_levels = np.linspace(min_contrast, -0.01, num_void_contours)
    overdense_levels = np.linspace(0.01, max_contrast, num_overdense_contours)
    all_levels = np.concatenate([void_levels, [0], overdense_levels])

    return transform_func(all_levels)

def extract_2d_hessian_from_3d(eigenvalues_3d, eigenvectors_3d, slice_dim):
    """Extract 2D Hessian components for the slice plane from 3D Hessian."""
    if slice_dim == 0:
        plane_indices = [1, 2]
    elif slice_dim == 1:
        plane_indices = [0, 2]
    else:
        plane_indices = [0, 1]
    
    sorted_indices = np.argsort(np.abs(eigenvalues_3d))[::-1]
    eigenvalues_sorted = eigenvalues_3d[sorted_indices]
    eigenvectors_sorted = eigenvectors_3d[:, sorted_indices]
    
    eigenvals_2d = eigenvalues_sorted[:2]
    eigenvecs_2d = eigenvectors_sorted[plane_indices, :2]
    
    if np.any(np.abs(eigenvals_2d) < 1e-12):
        eigenvals_2d = np.abs(eigenvals_2d) + 1e-12
    
    return eigenvals_2d, eigenvecs_2d

def create_ellipse_from_hessian(eigenvals_2d, eigenvecs_2d, scale_factor=2.0):
    """Create ellipse parameters from 2D Hessian eigendata."""
    min_eigenval = np.min(np.abs(eigenvals_2d))
    max_eigenval = np.max(np.abs(eigenvals_2d))
    
    if min_eigenval < 1e-15:
        return None
    
    condition_number = max_eigenval / min_eigenval
    if condition_number > 10000:
        return None
    
    semi_axes = scale_factor / np.sqrt(np.abs(eigenvals_2d))
    
    max_axis = np.max(semi_axes)
    if max_axis > 50.0:
        return None
    
    min_axis = np.min(semi_axes)
    if min_axis < 0.0001:
        return None
    
    angle = np.arctan2(eigenvecs_2d[1, 0], eigenvecs_2d[0, 0])
    
    return semi_axes[0], semi_axes[1], np.degrees(angle)

def add_periodic_ellipse(ax, center_x, center_y, width, height, angle, box_size, **kwargs):
    """Add an ellipse with periodic boundary conditions."""
    ellipse = Ellipse((center_x, center_y), width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    
    half_width = width / 2
    half_height = height / 2
    
    rad_angle = np.radians(angle)
    cos_a = np.cos(rad_angle)
    sin_a = np.sin(rad_angle)
    
    extent_x = abs(half_width * cos_a) + abs(half_height * sin_a)
    extent_y = abs(half_width * sin_a) + abs(half_height * cos_a)
    
    wrap_positions = []
    
    if center_x - extent_x < 0:
        wrap_positions.append((box_size, 0))
    if center_x + extent_x > box_size:
        wrap_positions.append((-box_size, 0))
    if center_y - extent_y < 0:
        wrap_positions.append((0, box_size))
    if center_y + extent_y > box_size:
        wrap_positions.append((0, -box_size))
    
    if (center_x - extent_x < 0) and (center_y - extent_y < 0):
        wrap_positions.append((box_size, box_size))
    if (center_x + extent_x > box_size) and (center_y - extent_y < 0):
        wrap_positions.append((-box_size, box_size))
    if (center_x - extent_x < 0) and (center_y + extent_y > box_size):
        wrap_positions.append((box_size, -box_size))
    if (center_x + extent_x > box_size) and (center_y + extent_y > box_size):
        wrap_positions.append((-box_size, -box_size))
    
    for dx, dy in wrap_positions:
        wrapped_ellipse = Ellipse((center_x + dx, center_y + dy), width, height, 
                                 angle=angle, **kwargs)
        ax.add_patch(wrapped_ellipse)

def plot_density_contrast_log_scale(contrast_slice, minima_2d, slice_dim, redshift, box_size):
    """Plot density contrast slice with minima marked."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    extent = [0, box_size, 0, box_size]

    # Axis labels based on slice dimension
    axis_labels = {
        0: ('Y [Mpc]', 'Z [Mpc]'),  # YZ plane
        1: ('X [Mpc]', 'Z [Mpc]'),  # XZ plane
        2: ('X [Mpc]', 'Y [Mpc]')   # XY plane
    }
    xlabel, ylabel = axis_labels[slice_dim]

    transform_func = create_density_contrast_transform()
    contrast_transformed = transform_func(contrast_slice)
    contrast_plot = contrast_transformed.T

    # Get vmin/vmax from transformed data for colorbar
    vmin, vmax = contrast_transformed.min(), contrast_transformed.max()

    im = ax.imshow(contrast_plot, origin='lower', cmap='coolwarm',
                   extent=extent, alpha=0.95)

    nx, ny = contrast_slice.shape
    nx_up, ny_up = contrast_plot.shape
    x_up = np.linspace(0, box_size, ny_up)
    y_up = np.linspace(0, box_size, nx_up)
    X_up, Y_up = np.meshgrid(x_up, y_up)

    transformed_levels = create_contour_levels(contrast_slice, transform_func)
    ax.contour(X_up, Y_up, contrast_plot, levels=transformed_levels,
            colors='black', alpha=0.7, linewidths=0.6)

    if len(minima_2d) > 0:
        x_coords = minima_2d[:, 0] * (box_size / nx)
        y_coords = minima_2d[:, 1] * (box_size / ny)

        ax.scatter(x_coords, y_coords, s=10, c='yellow', marker='o',
                  edgecolors='black', linewidths=1, alpha=0.95)

    create_custom_colorbar(im, ax, contrast_slice, transform_func, vmin=vmin, vmax=vmax)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f'3D minima: z={redshift:.2f}', fontsize=16)

    plt.tight_layout()
    return fig, ax

def plot_density_contrast_with_ellipses(contrast_slice, minima_2d, eigenvalue_info_slice,
                                       slice_dim, redshift, box_size, scale_factor):
    """Plot density contrast with ellipses fitted to minima."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    extent = [0, box_size, 0, box_size]

    # Axis labels based on slice dimension
    axis_labels = {
        0: ('Y [Mpc]', 'Z [Mpc]'),  # YZ plane
        1: ('X [Mpc]', 'Z [Mpc]'),  # XZ plane
        2: ('X [Mpc]', 'Y [Mpc]')   # XY plane
    }
    xlabel, ylabel = axis_labels[slice_dim]

    transform_func = create_density_contrast_transform()
    contrast_transformed = transform_func(contrast_slice)
    contrast_plot = contrast_transformed.T

    # Get vmin/vmax from transformed data for colorbar
    vmin, vmax = contrast_transformed.min(), contrast_transformed.max()

    im = ax.imshow(contrast_plot, origin='lower', cmap='coolwarm',
                   extent=extent, alpha=0.95)

    nx, ny = contrast_slice.shape
    x = np.linspace(0, box_size, ny)
    y = np.linspace(0, box_size, nx)
    X, Y = np.meshgrid(x, y)

    transformed_levels = create_contour_levels(contrast_slice, transform_func)
    ax.contour(X, Y, contrast_plot, levels=transformed_levels,
               colors='black', alpha=0.7, linewidths=0.6)

    if len(minima_2d) > 0:
        x_coords = minima_2d[:, 0] * (box_size / nx)
        y_coords = minima_2d[:, 1] * (box_size / ny)

        ax.scatter(x_coords, y_coords, s=10, c='yellow', marker='o',
                  edgecolors='black', linewidths=1, alpha=0.95, zorder=10)

    plotted_count = 0
    filtered_count = 0

    for minima_2d_coord in minima_2d:
        center_x = minima_2d_coord[0] * (box_size / nx)
        center_y = minima_2d_coord[1] * (box_size / ny)

        matching_info = None
        for info in eigenvalue_info_slice:
            coords_3d = info['coords_3d']

            if slice_dim == 0:
                projected_coord = [coords_3d[1], coords_3d[2]]
            elif slice_dim == 1:
                projected_coord = [coords_3d[0], coords_3d[2]]
            else:
                projected_coord = [coords_3d[0], coords_3d[1]]

            if (np.abs(projected_coord[0] - minima_2d_coord[0]) < 0.5 and
                np.abs(projected_coord[1] - minima_2d_coord[1]) < 0.5):
                matching_info = info
                break

        if matching_info is not None:
            eigenvals_3d = matching_info['eigenvalues']
            eigenvecs_3d = matching_info['eigenvectors']

            eigenvals_2d, eigenvecs_2d = extract_2d_hessian_from_3d(
                eigenvals_3d, eigenvecs_3d, slice_dim)

            ellipse_params = create_ellipse_from_hessian(
                eigenvals_2d, eigenvecs_2d, scale_factor=scale_factor)

            if ellipse_params is not None:
                width, height, angle = ellipse_params

                add_periodic_ellipse(
                    ax, center_x, center_y, width, height, angle, box_size,
                    facecolor='cyan', edgecolor='blue', alpha=0.3,
                    linewidth=1.5, zorder=5
                )
                plotted_count += 1
            else:
                filtered_count += 1

    print(f"  Ellipses plotted: {plotted_count}, filtered: {filtered_count}")

    create_custom_colorbar(im, ax, contrast_slice, transform_func, vmin=vmin, vmax=vmax)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f'3D Void Shapes: z={redshift:.2f}', fontsize=16)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)

    plt.tight_layout()
    return fig, ax

def process_single_snapshot(snapshot, redshift):
    """Process a single snapshot and generate plots."""
    
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")
    
    snapshot_dir = Path(BASE_DATA_DIR) / f"snapdir_{snapshot}"
    output_dir = Path(OUTPUT_DIR) / f"void_shapes_sigma{SMOOTHING_SIGMA}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    density_file = snapshot_dir / 'output.a_den'
    
    if not density_file.exists():
        print(f"  Warning: Density file not found")
        return False
    
    field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)
    
    try:
        print(f"  Loading and smoothing density field")
        density_field = dtfe.load_binary_field(str(density_file), field_shape, num_components=1)
        density_contrast = dtfe.calculate_density_contrast(density_field)
        contrast_smoothed = dtfe.smooth_field(density_contrast, sigma=SMOOTHING_SIGMA)

        print(f"  Finding 3D minima")
        minima_mask_3d, minima_coords_3d = dtfe.find_local_minima(
            contrast_smoothed, footprint_size=FOOTPRINT_SIZE)
        
        print(f"  Computing Hessian and eigenvalues")
        hessian_components = dtfe.calculate_hessian_fft(contrast_smoothed)
        eigenvalues, eigenvectors = dtfe.compute_eigenvalues_at_minima(hessian_components, minima_coords_3d)
        eigenvalue_info = [
            {'coords_3d': tuple(minima_coords_3d[i]),
             'eigenvalues': eigenvalues[i],
             'eigenvectors': eigenvectors[i]}
            for i in range(len(minima_coords_3d))
        ]

        # Process all 3 slice dimensions
        slice_names = ['YZ', 'XZ', 'XY']

        for slice_dim in range(3):
            slice_name = slice_names[slice_dim]
            print(f"  Processing {slice_name} plane (slice_dim={slice_dim})")

            # Create plane-specific output directory
            plane_output_dir = output_dir / slice_name
            plane_output_dir.mkdir(parents=True, exist_ok=True)

            slice_index = contrast_smoothed.shape[slice_dim] // 2
            contrast_slice = dtfe.extract_2d_slice(contrast_smoothed, slice_dim, slice_index)

            minima_2d = project_3d_minima_to_slice(minima_coords_3d, slice_dim, slice_index, tolerance=5)

            def filter_eigenvalue_info_for_slice(eigenvalue_info, slice_dim, slice_index, tolerance=5):
                filtered_info = []
                for info in eigenvalue_info:
                    coords_3d = info['coords_3d']
                    if abs(coords_3d[slice_dim] - slice_index) <= tolerance:
                        filtered_info.append(info)
                return filtered_info

            eigenvalue_info_slice = filter_eigenvalue_info_for_slice(
                eigenvalue_info, slice_dim, slice_index, tolerance=5)

            filtered_minima_2d = []

            for minima_2d_coord in minima_2d:
                matching_info = None
                for info in eigenvalue_info_slice:
                    coords_3d = info['coords_3d']

                    if slice_dim == 0:
                        projected_coord = [coords_3d[1], coords_3d[2]]
                    elif slice_dim == 1:
                        projected_coord = [coords_3d[0], coords_3d[2]]
                    else:
                        projected_coord = [coords_3d[0], coords_3d[1]]

                    if (np.abs(projected_coord[0] - minima_2d_coord[0]) < 0.5 and
                        np.abs(projected_coord[1] - minima_2d_coord[1]) < 0.5):
                        matching_info = info
                        break

                if matching_info is not None:
                    coords_3d = matching_info['coords_3d']
                    i_grid, j_grid, k_grid = coords_3d

                    try:
                        if slice_dim == 0:
                            density_contrast_val = contrast_slice[j_grid, k_grid]
                        elif slice_dim == 1:
                            density_contrast_val = contrast_slice[i_grid, k_grid]
                        else:
                            density_contrast_val = contrast_slice[i_grid, j_grid]

                        if density_contrast_val < DENSITY_THRESHOLD:
                            filtered_minima_2d.append(minima_2d_coord)
                    except IndexError:
                        continue

            filtered_minima_2d = np.array(filtered_minima_2d) if filtered_minima_2d else np.array([]).reshape(0, 2)

            final_minima_2d = filtered_minima_2d
            if USE_NMS and len(filtered_minima_2d) > 0:
                min_separation = 8
                final_minima_2d = merge_close_minima(filtered_minima_2d, contrast_slice, min_separation=min_separation)

            print(f"    Found {len(final_minima_2d)} voids in {slice_name} plane")
            print(f"    Creating plots for {slice_name} plane")

            fig1, ax1 = plot_density_contrast_log_scale(contrast_slice, final_minima_2d, slice_dim, redshift, BOX_SIZE)

            fig2, ax2 = plot_density_contrast_with_ellipses(
                contrast_slice, final_minima_2d, eigenvalue_info_slice,
                slice_dim, redshift, BOX_SIZE, SCALE_FACTOR
            )

            output_path_1 = plane_output_dir / f"void_analysis_z{redshift:.2f}_snap{snapshot}_basic.png"
            output_path_2 = plane_output_dir / f"void_analysis_z{redshift:.2f}_snap{snapshot}_ellipses.png"

            save_plot_to_multiple_paths(fig1, output_path_1, dpi=DPI, bbox_inches='tight', facecolor='white')
            save_plot_to_multiple_paths(fig2, output_path_2, dpi=DPI, bbox_inches='tight', facecolor='white')

            plt.close(fig1)
            plt.close(fig2)
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    """Process all snapshots for 3D void shape analysis."""
    
    print("Starting 3D void shape analysis")
    print(f"Data directory: {BASE_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}/void_shapes_sigma{SMOOTHING_SIGMA}")
    print(f"Smoothing sigma: {SMOOTHING_SIGMA}")
    print(f"Footprint size: {FOOTPRINT_SIZE}")
    print(f"Processing {len(SNAPSHOT_TO_REDSHIFT)} snapshots")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    successful_count = 0
    failed_snapshots = []
    
    sorted_snapshots = sorted(SNAPSHOT_TO_REDSHIFT.items(), 
                             key=lambda x: x[1], reverse=True)
    
    for snapshot, redshift in sorted_snapshots:
        if process_single_snapshot(snapshot, redshift):
            successful_count += 1
        else:
            failed_snapshots.append((snapshot, redshift))
    
    print(f"\nCompleted: {successful_count}/{len(SNAPSHOT_TO_REDSHIFT)} snapshots processed")
    
    if failed_snapshots:
        print(f"Failed snapshots: {', '.join(f'{s} (z={z:.2f})' for s, z in failed_snapshots)}")
    
    print(f"Total plots generated: {successful_count * 2}")
    print(f"Output saved to: {OUTPUT_DIR}/void_shapes_sigma{SMOOTHING_SIGMA}/")

if __name__ == "__main__":
    main()