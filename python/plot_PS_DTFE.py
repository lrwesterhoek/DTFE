"""
PS-DTFE Field Visualization Tool

Loads binary field data from PS-DTFE output (density, stream count) and creates
2D slice visualizations across different planes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ============================================================================
# Configuration Section
# ============================================================================

BASE_DATA_DIR = "/Users/luukw/DTFE/output/TNG50-3-Dark/snapdir_099"
OUTPUT_DIR = "python/figures/ps_dtfe"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7  # Mpc
AXIS_UNITS = "Mpc"

REDSHIFT = 0.00

# Which slice planes to visualize? (0=YZ, 1=XZ, 2=XY)
SLICE_PLANES_TO_PLOT = [0, 1, 2]

# Which fields to process?
PROCESS_DENSITY = True
PROCESS_STREAMS = True

# Visualization settings
GAUSSIAN_SMOOTHING_SIGMA = 5.0
DPI = 300

# ============================================================================

SLICE_PLANES = {
    0: {'name': 'yz_plane', 'axis_labels': ('Y', 'Z')},
    1: {'name': 'xz_plane', 'axis_labels': ('X', 'Z')},
    2: {'name': 'xy_plane', 'axis_labels': ('X', 'Y')}
}

# ============================================================================
# File Loading
# ============================================================================

def load_binary_field(binary_file, field_shape, num_components=1, dtype=np.float32):
    """Generic loader for binary field data."""
    data = np.fromfile(binary_file, dtype=dtype)
    expected_size = np.prod(field_shape) * num_components

    if data.size != expected_size:
        total_elements = data.size / num_components
        cube_root = round(total_elements ** (1/3))

        if cube_root**3 * num_components == data.size:
            field_shape = (cube_root, cube_root, cube_root)
        else:
            raise ValueError(
                f"File {binary_file} has {data.size} elements, "
                f"but expected {expected_size} for shape {field_shape}"
            )

    if num_components == 1:
        return data.reshape(field_shape)
    else:
        return data.reshape(field_shape + (num_components,))

# ============================================================================
# Slice Extraction
# ============================================================================

def extract_slice(field, slice_dim=2):
    """Extract a 2D slice from a 3D field at the midpoint."""
    idx = field.shape[slice_dim] // 2
    slices = [slice(None)] * 3
    slices[slice_dim] = idx
    return field[tuple(slices)]

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_density(density_field, slice_dim, box_size, redshift=None, save_path=None):
    """Create a PS-DTFE density field visualization."""
    dens_slice = extract_slice(density_field, slice_dim).T

    # PS-DTFE density can have zeros (no stream) and negatives shouldn't occur
    # but handle gracefully
    positive = dens_slice[dens_slice > 0]
    if positive.size == 0:
        print(f"    Warning: No positive density values in slice (dim={slice_dim})")
        return
    vmin = max(np.min(positive), 1e-6)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(cmap(0))
    cmap.set_under(cmap(0))
    im = ax.imshow(
        dens_slice, origin='lower', cmap=cmap,
        norm=colors.LogNorm(vmin=vmin, vmax=dens_slice.max()),
        extent=[0, box_size, 0, box_size]
    )

    plane_info = SLICE_PLANES[slice_dim]
    title = "PS-DTFE Density"
    if redshift is not None:
        title += f" (z={redshift:.2f})"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Density [{AXIS_UNITS}$^{{-3}}$]", fontsize=12)

    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_streams(stream_field, slice_dim, box_size, redshift=None, save_path=None):
    """Create a stream count visualization."""
    stream_slice = extract_slice(stream_field, slice_dim).T

    max_streams = int(stream_slice.max())
    if max_streams == 0:
        print(f"    Warning: No streams in slice (dim={slice_dim})")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    if max_streams <= 1:
        # Binary: 0 or 1 stream
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(cmap(0))
        im = ax.imshow(
            stream_slice, origin='lower', cmap=cmap,
            extent=[0, box_size, 0, box_size],
            vmin=0, vmax=1
        )
    else:
        # Multi-stream: use log scale for better contrast
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(cmap(0))
        cmap.set_under(cmap(0))
        im = ax.imshow(
            stream_slice, origin='lower', cmap=cmap,
            norm=colors.LogNorm(vmin=0.5, vmax=max(max_streams, 2)),
            extent=[0, box_size, 0, box_size]
        )

    plane_info = SLICE_PLANES[slice_dim]
    title = "Stream Count (PS-DTFE)"
    if redshift is not None:
        title += f" (z={redshift:.2f})"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of streams", fontsize=12)

    # Print stream statistics
    nonzero = stream_slice[stream_slice > 0]
    multi = stream_slice[stream_slice > 1]
    total = stream_slice.size
    ax.text(
        0.02, 0.98,
        f"max={max_streams}, multi-stream={100*multi.size/total:.1f}%",
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_density_comparison(density_field, stream_field, slice_dim, box_size,
                            redshift=None, save_path=None):
    """Side-by-side density and stream count for a single slice."""
    dens_slice = extract_slice(density_field, slice_dim).T
    stream_slice = extract_slice(stream_field, slice_dim).T

    positive = dens_slice[dens_slice > 0]
    if positive.size == 0:
        return
    vmin_dens = max(np.min(positive), 1e-6)
    max_streams = int(stream_slice.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plane_info = SLICE_PLANES[slice_dim]

    # Density panel
    cmap_dens = plt.cm.plasma.copy()
    cmap_dens.set_bad(cmap_dens(0))
    cmap_dens.set_under(cmap_dens(0))
    im1 = ax1.imshow(
        dens_slice, origin='lower', cmap=cmap_dens,
        norm=colors.LogNorm(vmin=vmin_dens, vmax=dens_slice.max()),
        extent=[0, box_size, 0, box_size]
    )
    ax1.set_title("PS-DTFE Density", fontsize=14)
    ax1.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax1.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
    ax1.set_aspect('equal')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label=f"Density [{AXIS_UNITS}$^{{-3}}$]")

    # Stream count panel
    if max_streams > 1:
        cmap_str = plt.cm.inferno.copy()
        cmap_str.set_bad(cmap_str(0))
        cmap_str.set_under(cmap_str(0))
        im2 = ax2.imshow(
            stream_slice, origin='lower', cmap=cmap_str,
            norm=colors.LogNorm(vmin=0.5, vmax=max(max_streams, 2)),
            extent=[0, box_size, 0, box_size]
        )
    else:
        cmap_str = plt.cm.viridis.copy()
        cmap_str.set_bad(cmap_str(0))
        im2 = ax2.imshow(
            stream_slice, origin='lower', cmap=cmap_str,
            extent=[0, box_size, 0, box_size],
            vmin=0, vmax=max(max_streams, 1)
        )
    ax2.set_title("Stream Count", fontsize=14)
    ax2.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax2.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
    ax2.set_aspect('equal')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Number of streams")

    suptitle = f"PS-DTFE: {plane_info['name']}"
    if redshift is not None:
        suptitle += f" (z={redshift:.2f})"
    fig.suptitle(suptitle, fontsize=16, y=1.02)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# ============================================================================
# Main Processing
# ============================================================================

def main():
    """Load PS-DTFE output and create visualizations."""

    data_dir = Path(BASE_DATA_DIR)
    field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)

    file_paths = {
        'density': data_dir / 'ps_output.den',
        'streams': data_dir / 'ps_output.streams',
    }

    print("PS-DTFE Field Visualization")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Grid resolution: {FIELD_RESOLUTION}^3")

    fields = {}

    if PROCESS_DENSITY:
        p = file_paths['density']
        if p.exists():
            print(f"  Loading density: {p}")
            dens = load_binary_field(str(p), field_shape)
            if GAUSSIAN_SMOOTHING_SIGMA > 0:
                dens = gaussian_filter(dens, sigma=GAUSSIAN_SMOOTHING_SIGMA, mode='wrap')
            fields['density'] = dens
            print(f"    min={dens.min():.4e}, max={dens.max():.4e}, mean={dens.mean():.4e}")
        else:
            print(f"  Warning: {p} not found")

    if PROCESS_STREAMS:
        p = file_paths['streams']
        if p.exists():
            print(f"  Loading stream count: {p}")
            streams = load_binary_field(str(p), field_shape)
            if GAUSSIAN_SMOOTHING_SIGMA > 0:
                streams = gaussian_filter(streams, sigma=GAUSSIAN_SMOOTHING_SIGMA, mode='wrap')
            fields['streams'] = streams
            print(f"    min={streams.min():.0f}, max={streams.max():.0f}, "
                  f"mean={streams.mean():.2f}")
            multi = np.sum(streams > 1)
            total = streams.size
            print(f"    multi-stream voxels: {multi} ({100*multi/total:.2f}%)")
        else:
            print(f"  Warning: {p} not found")

    if not fields:
        print("\nNo data files found. Nothing to plot.")
        return

    output_base = Path(OUTPUT_DIR)

    for slice_dim in SLICE_PLANES_TO_PLOT:
        plane_name = SLICE_PLANES[slice_dim]['name']
        plane_dir = output_base / plane_name
        print(f"\n  Creating {plane_name} visualizations...")

        if 'density' in fields:
            save_path = plane_dir / f"ps_density_{plane_name}_z{REDSHIFT:.2f}.png"
            plot_density(fields['density'], slice_dim, BOX_SIZE, REDSHIFT, save_path)
            print(f"    Saved density plot")

        if 'streams' in fields:
            save_path = plane_dir / f"ps_streams_{plane_name}_z{REDSHIFT:.2f}.png"
            plot_streams(fields['streams'], slice_dim, BOX_SIZE, REDSHIFT, save_path)
            print(f"    Saved stream count plot")

        if 'density' in fields and 'streams' in fields:
            save_path = plane_dir / f"ps_comparison_{plane_name}_z{REDSHIFT:.2f}.png"
            plot_density_comparison(
                fields['density'], fields['streams'], slice_dim,
                BOX_SIZE, REDSHIFT, save_path
            )
            print(f"    Saved comparison plot")

    print(f"\nDone! Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
