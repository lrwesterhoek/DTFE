"""
3D Surface Visualization of Density Contrast Fields

Creates elevated 3D surface plots from density slices, where the height
and color represent the density contrast values.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import dtfe_functions as dtfe
from dtfe_shared import save_plot_to_multiple_paths

# Configuration
BASE_DATA_DIR = "output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/3d_surface_plots"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7  # Mpc
SLICE_DIM = 2  # 0=YZ, 1=XZ, 2=XY

# Smoothing and visualization parameters
SMOOTHING_SIGMA = 10.0
HEIGHT_SCALE = 5.0  # Exaggerates vertical features
DOWNSAMPLE_FACTOR = 4  # Reduces mesh density for cleaner look
STRIDE = 2  # Controls wireframe density

SNAPSHOT_TO_REDSHIFT = dtfe.SNAPSHOT_TO_REDSHIFT
DPI = 300


def create_3d_surface_plot(contrast_slice, redshift, box_size, height_scale=5.0, 
                          downsample=4, stride=2):
    """
    Generate a 3D surface plot from a 2D density contrast slice.
    
    The surface height represents the density contrast value, while color
    provides additional visual information about over/underdensities.
    """
    # Downsample for performance and visual clarity
    if downsample > 1:
        ny, nx = contrast_slice.shape
        new_ny, new_nx = ny // downsample, nx // downsample
        
        # Use block averaging for smoother downsampling
        y_indices = np.linspace(0, ny - 1, new_ny, dtype=int)
        x_indices = np.linspace(0, nx - 1, new_nx, dtype=int)
        contrast_downsampled = contrast_slice[np.ix_(y_indices, x_indices)]
    else:
        contrast_downsampled = contrast_slice
    
    ny, nx = contrast_downsampled.shape
    
    # Create coordinate grids in physical units
    x = np.linspace(0, box_size, nx)
    y = np.linspace(0, box_size, ny)
    X, Y = np.meshgrid(x, y)
    
    # Z values represent the density contrast
    Z = contrast_downsampled * height_scale
    
    # Set up the figure with good viewing angle
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors to density contrast values
    norm = plt.Normalize(vmin=contrast_downsampled.min(), 
                        vmax=contrast_downsampled.max())
    colors = cm.coolwarm(norm(contrast_downsampled))
    
    # Create the surface with colored faces
    surf = ax.plot_surface(X, Y, Z, facecolors=colors, 
                          rstride=stride, cstride=stride,
                          linewidth=0.5,
                          alpha=0.9, antialiased=True, 
                          shade=True)
    
    # Add black wireframe overlay for better visibility
    ax.plot_wireframe(X, Y, Z, rstride=stride*2, cstride=stride*2,
                     color='black', linewidth=0.3, alpha=0.4)
    
    # Viewing angle (elevation, azimuth)
    ax.view_init(elev=35, azim=45)
    
    # Labels depend on slice orientation
    axis_labels = {
        0: ('Y [Mpc]', 'Z [Mpc]'),
        1: ('X [Mpc]', 'Z [Mpc]'),
        2: ('X [Mpc]', 'Y [Mpc]')
    }
    xlabel, ylabel = axis_labels[SLICE_DIM]
    
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_zlabel(f'Density Contrast × {height_scale:.1f}', fontsize=12, labelpad=10)
    
    # Set equal aspect ratio for X and Y (not Z, since we're exaggerating height)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Density Contrast δ', fontsize=12)
    
    # Title with redshift info
    ax.set_title(f'Density Field Structure (z={redshift:.2f})', 
                fontsize=14, pad=20)
    
    # Clean up tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    return fig, ax


def create_multiple_views(contrast_slice, redshift, box_size, height_scale=5.0):
    """
    Create a figure with three different viewing angles of the same slice.
    
    This might help show the full 3D structure from multiple perspectives.
    """
    downsample = DOWNSAMPLE_FACTOR
    stride = STRIDE
    
    # Downsample
    if downsample > 1:
        ny, nx = contrast_slice.shape
        new_ny, new_nx = ny // downsample, nx // downsample
        y_indices = np.linspace(0, ny - 1, new_ny, dtype=int)
        x_indices = np.linspace(0, nx - 1, new_nx, dtype=int)
        contrast_downsampled = contrast_slice[np.ix_(y_indices, x_indices)]
    else:
        contrast_downsampled = contrast_slice
    
    ny, nx = contrast_downsampled.shape
    x = np.linspace(0, box_size, nx)
    y = np.linspace(0, box_size, ny)
    X, Y = np.meshgrid(x, y)
    Z = contrast_downsampled * height_scale
    
    # Three viewing angles
    viewing_angles = [
        (35, 45, 'View 1'),
        (35, 135, 'View 2'),
        (35, 225, 'View 3')
    ]
    
    fig = plt.figure(figsize=(18, 6))
    
    norm = plt.Normalize(vmin=contrast_downsampled.min(), 
                        vmax=contrast_downsampled.max())
    colors = cm.coolwarm(norm(contrast_downsampled))
    
    for idx, (elev, azim, title) in enumerate(viewing_angles, 1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        
        ax.plot_surface(X, Y, Z, facecolors=colors,
                       rstride=stride, cstride=stride,
                       linewidth=0.5,
                       alpha=0.9, antialiased=True, shade=True)
        
        # Add wireframe overlay
        ax.plot_wireframe(X, Y, Z, rstride=stride*2, cstride=stride*2,
                         color='black', linewidth=0.3, alpha=0.4)
        
        ax.view_init(elev=elev, azim=azim)
        
        axis_labels = {
            0: ('Y [Mpc]', 'Z [Mpc]'),
            1: ('X [Mpc]', 'Z [Mpc]'),
            2: ('X [Mpc]', 'Y [Mpc]')
        }
        xlabel, ylabel = axis_labels[SLICE_DIM]
        
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_zlabel(f'δ × {height_scale:.1f}', fontsize=10)
        ax.set_title(f'{title} (elev={elev}°, azim={azim}°)', fontsize=11)
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
    
    fig.suptitle(f'Multiple Viewing Angles (z={redshift:.2f})', 
                fontsize=14, y=0.98)
    plt.tight_layout()
    
    return fig


def process_snapshot(snapshot, redshift):
    """Process a single snapshot and create 3D surface visualizations."""
    
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")
    
    snapshot_dir = Path(BASE_DATA_DIR) / f"snapdir_{snapshot}"
    output_dir = Path(OUTPUT_DIR) / f"sigma{SMOOTHING_SIGMA}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    density_file = snapshot_dir / 'output.a_den'
    
    if not density_file.exists():
        print(f"  Density file not found: {density_file}")
        return False
    
    field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)
    
    try:
        print(f"  Loading density field...")
        density_field = dtfe.load_binary_field(str(density_file), field_shape, 
                                              num_components=1)
        
        print(f"  Computing density contrast...")
        density_contrast = dtfe.calculate_density_contrast(density_field)
        
        print(f"  Smoothing (sigma={SMOOTHING_SIGMA})...")
        contrast_smoothed = dtfe.smooth_field(density_contrast, sigma=SMOOTHING_SIGMA)
        
        # Extract slice
        slice_index = contrast_smoothed.shape[SLICE_DIM] // 2
        contrast_slice = dtfe.extract_2d_slice(contrast_smoothed, SLICE_DIM, slice_index)
        
        print(f"  Creating 3D surface plot...")
        fig1, ax1 = create_3d_surface_plot(
            contrast_slice, redshift, BOX_SIZE, 
            height_scale=HEIGHT_SCALE,
            downsample=DOWNSAMPLE_FACTOR,
            stride=STRIDE
        )
        
        # Save single view
        output_path_1 = output_dir / f"surface_3d_z{redshift:.2f}_snap{snapshot}.png"
        save_plot_to_multiple_paths(fig1, output_path_1, dpi=DPI, 
                                   bbox_inches='tight', facecolor='white')
        plt.close(fig1)
        
        print(f"  Creating multi-view plot...")
        fig2 = create_multiple_views(contrast_slice, redshift, BOX_SIZE, 
                                     height_scale=HEIGHT_SCALE)
        
        # Save multi-view
        output_path_2 = output_dir / f"surface_3d_multiview_z{redshift:.2f}_snap{snapshot}.png"
        save_plot_to_multiple_paths(fig2, output_path_2, dpi=DPI, 
                                   bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        
        print(f"  Saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"  Error processing snapshot: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Process all snapshots and create 3D surface visualizations."""
    
    print("Starting 3D surface visualization pipeline")
    print(f"Data directory: {BASE_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}/sigma{SMOOTHING_SIGMA}")
    print(f"Height scale factor: {HEIGHT_SCALE}")
    print(f"Downsample factor: {DOWNSAMPLE_FACTOR}")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = []
    
    # Process snapshots from high to low redshift
    sorted_snapshots = sorted(SNAPSHOT_TO_REDSHIFT.items(), 
                             key=lambda x: x[1], reverse=True)
    
    for snapshot, redshift in sorted_snapshots:
        if process_snapshot(snapshot, redshift):
            successful += 1
        else:
            failed.append((snapshot, redshift))
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {successful}/{len(SNAPSHOT_TO_REDSHIFT)} snapshots")
    
    if failed:
        print(f"Failed snapshots: {', '.join(f'{s} (z={z:.2f})' for s, z in failed)}")
    
    print(f"Output saved to: {OUTPUT_DIR}/sigma{SMOOTHING_SIGMA}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()