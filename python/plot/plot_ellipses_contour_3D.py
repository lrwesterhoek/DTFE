"""3D void-shape maps: deep-void minima and Hessian ellipses on delta contour slices.

Method-, snapshot- and simulation-agnostic via dtfelib/pipeline:

    python3 plot/plot_ellipses_contour_3D.py                  # whole series, auto method
    python3 plot/plot_ellipses_contour_3D.py --snap 99 --method dtfe
    python3 plot/plot_ellipses_contour_3D.py --sim TNG50-3-Dark
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from dtfelib import fields as dtfe
from dtfelib.figures import save_plot_to_multiple_paths
import config
from dtfelib import figures as style
from dtfelib import pipeline
from dtfelib import make_parser, DATA_ROOT

style.apply()

OUTPUT_DIR = config.figures_path('ellipses')

SLICE_DIM = 0

SMOOTHING_SIGMA = config.SMOOTHING_SIGMA_CELLS
FOOTPRINT_SIZE = config.FOOTPRINT_SIZE
SCALE_FACTOR = 0.5
USE_NMS = False

SNAPSHOT_TO_REDSHIFT = config.SNAPSHOT_TO_REDSHIFT

DPI = config.DPI


def project_3d_minima_to_slice(minima_3d, slice_dim, slice_index, tolerance=5):
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

def create_contour_levels(contrast_slice, norm, num_contours=20):
    return style.delta_contour_levels(contrast_slice, norm, num_contours)

def extract_2d_hessian_from_3d(eigenvalues_3d, eigenvectors_3d, slice_dim):
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
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    extent = [0, box_size, 0, box_size]

    axis_labels = {
        0: ('Y [Mpc]', 'Z [Mpc]'),
        1: ('X [Mpc]', 'Z [Mpc]'),
        2: ('X [Mpc]', 'Y [Mpc]')
    }
    xlabel, ylabel = axis_labels[slice_dim]

    norm = style.field_norm('delta', contrast_slice)
    contrast_plot = contrast_slice.T

    im = ax.imshow(contrast_plot, origin='lower', cmap=style.CMAP['delta'],
                   norm=norm, extent=extent, alpha=0.95)

    nx, ny = contrast_slice.shape
    nx_up, ny_up = contrast_plot.shape
    x_up = np.linspace(0, box_size, ny_up)
    y_up = np.linspace(0, box_size, nx_up)
    X_up, Y_up = np.meshgrid(x_up, y_up)

    contour_levels = create_contour_levels(contrast_slice, norm)
    ax.contour(X_up, Y_up, contrast_plot, levels=contour_levels,
            colors='black', alpha=0.7, linewidths=0.6)

    if len(minima_2d) > 0:
        x_coords = minima_2d[:, 0] * (box_size / nx)
        y_coords = minima_2d[:, 1] * (box_size / ny)

        ax.scatter(x_coords, y_coords, s=10, c='yellow', marker='o',
                  edgecolors='black', linewidths=1, alpha=0.95)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25,
                        ticks=style.norm_ticks(norm))
    cbar.set_label('Density Contrast δ = (ρ - ρ̄)/ρ̄', labelpad=15)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style.set_title(ax, f'3D minima: z={redshift:.2f}', fontsize=16)

    plt.tight_layout()
    return fig, ax

def plot_density_contrast_with_ellipses(contrast_slice, minima_2d, eigenvalue_info_slice,
                                       slice_dim, redshift, box_size, scale_factor):
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    extent = [0, box_size, 0, box_size]

    axis_labels = {
        0: ('Y [Mpc]', 'Z [Mpc]'),
        1: ('X [Mpc]', 'Z [Mpc]'),
        2: ('X [Mpc]', 'Y [Mpc]')
    }
    xlabel, ylabel = axis_labels[slice_dim]

    norm = style.field_norm('delta', contrast_slice)
    contrast_plot = contrast_slice.T

    im = ax.imshow(contrast_plot, origin='lower', cmap=style.CMAP['delta'],
                   norm=norm, extent=extent, alpha=0.95)

    nx, ny = contrast_slice.shape
    x = np.linspace(0, box_size, ny)
    y = np.linspace(0, box_size, nx)
    X, Y = np.meshgrid(x, y)

    contour_levels = create_contour_levels(contrast_slice, norm)
    ax.contour(X, Y, contrast_plot, levels=contour_levels,
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

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25,
                        ticks=style.norm_ticks(norm))
    cbar.set_label('Density Contrast δ = (ρ - ρ̄)/ρ̄', labelpad=15)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style.set_title(ax, f'3D Void Shapes: z={redshift:.2f}', fontsize=16)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)

    plt.tight_layout()
    return fig, ax

def process_single_snapshot(snapshot, redshift, args):

    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")

    output_dir = Path(OUTPUT_DIR) / f"void_shapes_sigma{SMOOTHING_SIGMA}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        p = pipeline.products(snapshot, sim=args.sim, method=args.method)
        grid_n = p.fs.grid_n
        box_size = p.fs.meta.box_mpc
        redshift = p.redshift
        cat = p.voids()
        delta_slices = p.delta_slices()
        p.release()

        deep = np.asarray(cat['deep'], dtype=bool)
        minima_coords_3d = cat['coords'][deep]
        evals_deep = cat['eigenvalues'][deep]
        evecs_deep = cat['eigenvectors'][deep]
        print(f"  Deep voids (delta < {config.DEEP_VOID_THRESHOLD}): "
              f"{len(minima_coords_3d)} / {len(cat['coords'])}")
        eigenvalue_info = [
            {'coords_3d': tuple(minima_coords_3d[i]),
             'eigenvalues': evals_deep[i],
             'eigenvectors': evecs_deep[i]}
            for i in range(len(minima_coords_3d))
        ]

        slice_names = ['YZ', 'XZ', 'XY']

        for slice_dim in range(3):
            slice_name = slice_names[slice_dim]
            print(f"  Processing {slice_name} plane (slice_dim={slice_dim})")

            plane_output_dir = output_dir / slice_name
            plane_output_dir.mkdir(parents=True, exist_ok=True)

            slice_index = grid_n // 2
            contrast_slice = delta_slices[slice_dim]

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

            final_minima_2d = minima_2d
            if USE_NMS and len(minima_2d) > 0:
                final_minima_2d = merge_close_minima(minima_2d, contrast_slice,
                                                     min_separation=8)

            print(f"    Found {len(final_minima_2d)} voids in {slice_name} plane")
            print(f"    Creating plots for {slice_name} plane")

            fig1, ax1 = plot_density_contrast_log_scale(contrast_slice, final_minima_2d, slice_dim, redshift, box_size)

            fig2, ax2 = plot_density_contrast_with_ellipses(
                contrast_slice, final_minima_2d, eigenvalue_info_slice,
                slice_dim, redshift, box_size, SCALE_FACTOR
            )

            output_path_1 = plane_output_dir / f"void_analysis_z{redshift:.2f}_snap{snapshot}_basic.png"
            output_path_2 = plane_output_dir / f"void_analysis_z{redshift:.2f}_snap{snapshot}_ellipses.png"

            save_plot_to_multiple_paths(fig1, output_path_1, dpi=DPI, bbox_inches='tight', facecolor='white')
            save_plot_to_multiple_paths(fig2, output_path_2, dpi=DPI, bbox_inches='tight', facecolor='white')

            plt.close(fig1)
            plt.close(fig2)
        
        return True

    except FileNotFoundError:
        print(f"  skipping snapshot {snapshot} (no {args.method} fields)")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():

    parser = make_parser("3D void shapes: deep-void minima and Hessian ellipses on delta contour slices.")
    for action in parser._actions:   # series script: default is the WHOLE snapshot series
        if action.dest == "snap":
            action.default = None
            action.help = "process only this snapshot (default: all snapshots in the series)"
    args = parser.parse_args()

    print("Starting 3D void shape analysis")
    print(f"Data directory: {DATA_ROOT / args.sim} (method: {args.method})")
    print(f"Output directory: {OUTPUT_DIR}/void_shapes_sigma{SMOOTHING_SIGMA}")
    print(f"Smoothing sigma: {SMOOTHING_SIGMA}")
    print(f"Footprint size: {FOOTPRINT_SIZE}")

    if args.snap is None:
        sorted_snapshots = sorted(SNAPSHOT_TO_REDSHIFT.items(),
                                 key=lambda x: x[1], reverse=True)
    else:
        key = f"{args.snap:03d}"
        z = config.get_redshift(key)
        sorted_snapshots = [(key, z if z is not None else float('nan'))]

    print(f"Processing {len(sorted_snapshots)} snapshots")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    successful_count = 0
    failed_snapshots = []

    for snapshot, redshift in sorted_snapshots:
        if process_single_snapshot(snapshot, redshift, args):
            successful_count += 1
        else:
            failed_snapshots.append((snapshot, redshift))

    print(f"\nCompleted: {successful_count}/{len(sorted_snapshots)} snapshots processed")
    
    if failed_snapshots:
        print(f"Failed snapshots: {', '.join(f'{s} (z={z:.2f})' for s, z in failed_snapshots)}")
    
    print(f"Total plots generated: {successful_count * 2}")
    print(f"Output saved to: {OUTPUT_DIR}/void_shapes_sigma{SMOOTHING_SIGMA}/")

if __name__ == "__main__":
    main()
