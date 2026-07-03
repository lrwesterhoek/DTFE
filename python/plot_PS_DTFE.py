
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter
from pathlib import Path

BASE_DATA_DIR = "/Users/luukw/output/TNG50-4-Dark/snapdir_099"
OUTPUT_DIR = "python/figures/ps_dtfe"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7
AXIS_UNITS = "Mpc"

REDSHIFT = 0.00

# Use the volume-averaged ('_a') PS-DTFE fields. Both the averaged and unaveraged fields are
# mass-conserving (each tetrahedron deposits its full mass onto the grid); the '_a' fields resolve
# each cell with an nSub^3 sub-sample grid (default 27 points), so they are smoother and better
# resolved at caustics. Set False to use the coarser single-sample-per-cell fields.
USE_AVERAGED = True

SLICE_PLANES_TO_PLOT = [0, 1, 2]

PROCESS_DENSITY = True
PROCESS_STREAMS = True
PROCESS_DISPERSION = True   # velocity dispersion trace + tensor magnitude
PROCESS_VELOCITY = True     # velocity magnitude |v|
PROCESS_WEB = True          # T-web/V-web classification + eigenvalue maps

GAUSSIAN_SMOOTHING_SIGMA = 1.0
DPI = 300

SLICE_PLANES = {
    0: {'name': 'yz_plane', 'axis_labels': ('Y', 'Z')},
    1: {'name': 'xz_plane', 'axis_labels': ('X', 'Z')},
    2: {'name': 'xy_plane', 'axis_labels': ('X', 'Y')}
}

def load_binary_field(binary_file, field_shape, num_components=1, dtype=np.float32):
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

def extract_slice(field, slice_dim=2):
    idx = field.shape[slice_dim] // 2
    slices = [slice(None)] * 3
    slices[slice_dim] = idx
    return field[tuple(slices)]

def plot_density(density_field, slice_dim, box_size, redshift=None, save_path=None):
    dens_slice = extract_slice(density_field, slice_dim).T

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
    stream_slice = extract_slice(stream_field, slice_dim).T

    max_streams = int(stream_slice.max())
    if max_streams == 0:
        print(f"    Warning: No streams in slice (dim={slice_dim})")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    if max_streams <= 1:
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(cmap(0))
        im = ax.imshow(
            stream_slice, origin='lower', cmap=cmap,
            extent=[0, box_size, 0, box_size],
            vmin=0, vmax=1
        )
    else:
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
    dens_slice = extract_slice(density_field, slice_dim).T
    stream_slice = extract_slice(stream_field, slice_dim).T

    positive = dens_slice[dens_slice > 0]
    if positive.size == 0:
        return
    vmin_dens = max(np.min(positive), 1e-6)
    max_streams = int(stream_slice.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plane_info = SLICE_PLANES[slice_dim]

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

def plot_scalar_log(field, slice_dim, box_size, title, cbar_label, redshift=None, save_path=None):
    """Log-scale slice plot for positive scalar fields (dispersion trace, tensor magnitude)."""
    fslice = extract_slice(field, slice_dim).T
    positive = fslice[fslice > 0]
    if positive.size == 0:
        print(f"    Warning: no positive values in slice (dim={slice_dim})")
        return
    vmin = np.percentile(positive, 1)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    cmap = plt.get_cmap('magma')
    im = ax.imshow(
        fslice, origin='lower', cmap=cmap,
        norm=colors.LogNorm(vmin=vmin, vmax=fslice.max()),
        extent=[0, box_size, 0, box_size]
    )
    plane_info = SLICE_PLANES[slice_dim]
    full_title = f"{title} (z={redshift:.2f})" if redshift is not None else title
    ax.set_title(full_title, fontsize=16)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)


WEB_CLASS_NAMES = ['void', 'wall', 'filament', 'node']
WEB_CLASS_COLORS = ['#1a1a40', '#3b7ea1', '#e8843c', '#f7e463']   # void/wall/filament/node

def plot_web_classification(field, slice_dim, box_size, title, redshift=None, save_path=None):
    """Discrete cosmic-web classification map (0=void, 1=wall, 2=filament, 3=node)."""
    wslice = extract_slice(field, slice_dim).T
    cmap = colors.ListedColormap(WEB_CLASS_COLORS)
    norm = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(wslice, origin='lower', cmap=cmap, norm=norm,
                   extent=[0, box_size, 0, box_size], interpolation='nearest')
    plane_info = SLICE_PLANES[slice_dim]
    full_title = f"{title} (z={redshift:.2f})" if redshift is not None else title
    ax.set_title(full_title, fontsize=16)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(WEB_CLASS_NAMES)
    # volume fractions of this slice, printed on the panel
    fracs = [100.0 * np.mean(wslice == k) for k in range(4)]
    ax.text(0.02, 0.98, "  ".join(f"{n}: {f:.1f}%" for n, f in zip(WEB_CLASS_NAMES, fracs)),
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)


def plot_web_eigenvalues(eig_field, slice_dim, box_size, title, redshift=None, save_path=None):
    """Triptych of the (descending-sorted) eigenvalue maps, diverging colormap, symlog scale."""
    eslice = extract_slice(eig_field, slice_dim)          # (N, N, 3)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    plane_info = SLICE_PLANES[slice_dim]
    for i, ax in enumerate(axes):
        lam = eslice[..., i].T
        vmax = np.percentile(np.abs(lam), 99.5)
        if vmax <= 0: vmax = 1.0
        norm = colors.SymLogNorm(linthresh=vmax / 1e3, vmin=-vmax, vmax=vmax, base=10)
        im = ax.imshow(lam, origin='lower', cmap='RdBu_r', norm=norm,
                       extent=[0, box_size, 0, box_size])
        ax.set_title(rf"$\lambda_{i+1}$", fontsize=14)
        ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=11)
        if i == 0:
            ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    full_title = f"{title} (z={redshift:.2f})" if redshift is not None else title
    fig.suptitle(full_title, fontsize=16)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)


def main():

    data_dir = Path(BASE_DATA_DIR)
    field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)

    # 'ps_output.a_*' = volume-averaged (clean); 'ps_output.*' = raw nSub=1 point samples (alias).
    prefix = 'ps_output.a_' if USE_AVERAGED else 'ps_output.'
    file_paths = {
        'density': data_dir / (prefix + 'den'),
        'streams': data_dir / (prefix + 'streams'),
        'velDisp': data_dir / (prefix + 'velDisp'),
        'velDispTensor': data_dir / (prefix + 'velDispTensor'),
        'vel': data_dir / (prefix + 'vel'),
        # T-web comes from python/compute_tweb.py (tidal tensor of the density grid); the old
        # C++ 'velTweb' output was removed -- it duplicated the V-web.
        'velTweb': data_dir / 'ps_output.py_tweb',
        'velTwebEig': data_dir / 'ps_output.py_twebEig',
        'velVweb': data_dir / (prefix + 'velVweb'),
        'velVwebEig': data_dir / (prefix + 'velVwebEig'),
    }

    print("PS-DTFE Field Visualization")
    print(f"Using {'volume-averaged _a' if USE_AVERAGED else 'UNAVERAGED nSub=1 (aliased)'} fields")
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

    if PROCESS_DISPERSION:
        p = file_paths['velDisp']
        if p.exists():
            print(f"  Loading velocity dispersion (trace): {p}")
            disp = load_binary_field(str(p), field_shape)
            fields['velDisp'] = disp
            print(f"    min={disp.min():.4e}, max={disp.max():.4e}, mean={disp.mean():.4e}")
        else:
            print(f"  Warning: {p} not found")
        p = file_paths['velDispTensor']
        if p.exists():
            print(f"  Loading dispersion tensor (6 comps): {p}")
            tens = load_binary_field(str(p), field_shape, num_components=6)
            # Frobenius norm of the symmetric tensor (off-diagonals xy,xz,yz counted twice)
            xx, xy, xz, yy, yz, zz = [tens[..., i] for i in range(6)]
            fields['velDispMag'] = np.sqrt(xx**2 + yy**2 + zz**2 + 2*(xy**2 + xz**2 + yz**2))
            del tens
        else:
            print(f"  Warning: {p} not found")

    if PROCESS_VELOCITY:
        p = file_paths['vel']
        if p.exists():
            print(f"  Loading velocity (3 comps): {p}")
            vel = load_binary_field(str(p), field_shape, num_components=3)
            fields['velMag'] = np.sqrt(vel[..., 0]**2 + vel[..., 1]**2 + vel[..., 2]**2)
            del vel
            m = fields['velMag']
            print(f"    |v|: min={m.min():.4g}, max={m.max():.4g}, mean={m.mean():.4g} km/s")
        else:
            print(f"  Warning: {p} not found")

    if PROCESS_WEB:
        for key, label in [('velTweb', 'T-web'), ('velVweb', 'V-web')]:
            p = file_paths[key]
            if p.exists():
                print(f"  Loading {label} classification: {p}")
                web = load_binary_field(str(p), field_shape)
                fields[key] = web
                fr = [100.0 * np.mean(np.rint(web) == k) for k in range(4)]
                print("    volume fractions: " + "  ".join(
                    f"{n}={f:.1f}%" for n, f in zip(['void','wall','filament','node'], fr)))
            else:
                print(f"  Warning: {p} not found")
        for key, label in [('velTwebEig', 'T-web eigenvalues'), ('velVwebEig', 'V-web eigenvalues')]:
            p = file_paths[key]
            if p.exists():
                print(f"  Loading {label} (3 comps): {p}")
                fields[key] = load_binary_field(str(p), field_shape, num_components=3)
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

        if 'velDisp' in fields:
            save_path = plane_dir / f"ps_velDisp_{plane_name}_z{REDSHIFT:.2f}.png"
            plot_scalar_log(fields['velDisp'], slice_dim, BOX_SIZE,
                            "PS-DTFE Velocity Dispersion", r"Tr $\sigma^2$  [(km/s)$^2$]",
                            REDSHIFT, save_path)
            print(f"    Saved velocity dispersion plot")

        if 'velDispMag' in fields:
            save_path = plane_dir / f"ps_velDispTensor_{plane_name}_z{REDSHIFT:.2f}.png"
            plot_scalar_log(fields['velDispMag'], slice_dim, BOX_SIZE,
                            "PS-DTFE Dispersion Tensor", r"$|\sigma^2|$  [(km/s)$^2$]",
                            REDSHIFT, save_path)
            print(f"    Saved dispersion tensor plot")

        if 'velMag' in fields:
            save_path = plane_dir / f"ps_velMag_{plane_name}_z{REDSHIFT:.2f}.png"
            plot_scalar_log(fields['velMag'], slice_dim, BOX_SIZE,
                            "PS-DTFE Velocity Magnitude", r"$|v|$  [km/s]",
                            REDSHIFT, save_path)
            print(f"    Saved velocity magnitude plot")

        if 'velTweb' in fields:
            save_path = plane_dir / f"ps_tweb_{plane_name}_z{REDSHIFT:.2f}.png"
            plot_web_classification(fields['velTweb'], slice_dim, BOX_SIZE,
                                    "PS-DTFE T-web Classification (tidal)", REDSHIFT, save_path)
            print(f"    Saved T-web classification plot")

        if 'velTwebEig' in fields:
            save_path = plane_dir / f"ps_twebEig_{plane_name}_z{REDSHIFT:.2f}.png"
            plot_web_eigenvalues(fields['velTwebEig'], slice_dim, BOX_SIZE,
                                 "PS-DTFE T-web Eigenvalues (tidal)", REDSHIFT, save_path)
            print(f"    Saved T-web eigenvalue plot")

        if 'velVweb' in fields:
            save_path = plane_dir / f"ps_vweb_{plane_name}_z{REDSHIFT:.2f}.png"
            plot_web_classification(fields['velVweb'], slice_dim, BOX_SIZE,
                                    "PS-DTFE V-web Classification", REDSHIFT, save_path)
            print(f"    Saved V-web classification plot")

        if 'velVwebEig' in fields:
            save_path = plane_dir / f"ps_vwebEig_{plane_name}_z{REDSHIFT:.2f}.png"
            plot_web_eigenvalues(fields['velVwebEig'], slice_dim, BOX_SIZE,
                                 "PS-DTFE V-web Eigenvalues", REDSHIFT, save_path)
            print(f"    Saved V-web eigenvalue plot")

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
