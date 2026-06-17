
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter
from pathlib import Path

BASE_DATA_DIR = "/Users/luukw/output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/dtfe"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7
AXIS_UNITS = "Mpc"
VELOCITY_UNITS = "km/s"

SNAPSHOT_TO_REDSHIFT = {
    '000': 20.05,
    '004': 10.00,
    '008': 8.01,
    '013': 6.01,
    '017': 5.00,
    '021': 4.01,
    '025': 3.01,
    '033': 2.00,
    '040': 1.50,
    '050': 1.00,
    '067': 0.50,
    '072': 0.40,
    '078': 0.30,
    '084': 0.20,
    '091': 0.10,
    '099': 0.00
}

SLICE_PLANES_TO_PLOT = [0, 1, 2]

PROCESS_DENSITY = True
PROCESS_VELOCITY = True
PROCESS_DIVERGENCE = True
PROCESS_SHEAR = True

GAUSSIAN_SMOOTHING_SIGMA = 0.0
VELOCITY_QUIVER_STEP = 8
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

def extract_velocity_slice(velocity_field, slice_dim=2):
    idx = velocity_field.shape[slice_dim] // 2
    
    if slice_dim == 0:
        slice_data = velocity_field[idx, :, :]
        U, V = slice_data[..., 1], slice_data[..., 2]
    elif slice_dim == 1:
        slice_data = velocity_field[:, idx, :]
        U, V = slice_data[..., 0], slice_data[..., 2]
    else:
        slice_data = velocity_field[:, :, idx]
        U, V = slice_data[..., 0], slice_data[..., 1]
    
    ny, nx = U.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    return X, Y, U, V

def plot_density(density_field, slice_dim, box_size, redshift=None, save_path=None):
    dens_slice = extract_slice(density_field, slice_dim).T
    vmin = max(np.min(dens_slice[dens_slice > 0]), 1e-6)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        dens_slice, origin='lower', cmap='plasma',
        norm=colors.LogNorm(vmin=vmin, vmax=dens_slice.max()),
        extent=[0, box_size, 0, box_size]
    )
    
    plane_info = SLICE_PLANES[slice_dim]
    title = "Density Field"
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

def plot_velocity(velocity_field, slice_dim, box_size, quiver_step, 
                 redshift=None, save_path=None):
    X, Y, U, V = extract_velocity_slice(velocity_field, slice_dim)
    U, V = U.T, V.T
    
    ny, nx = U.shape
    pixel_size = box_size / nx
    X_mesh, Y_mesh = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    X_phys, Y_phys = X_mesh * pixel_size, Y_mesh * pixel_size
    
    X_q = X_phys[::quiver_step, ::quiver_step]
    Y_q = Y_phys[::quiver_step, ::quiver_step]
    U_q = U[::quiver_step, ::quiver_step]
    V_q = V[::quiver_step, ::quiver_step]
    
    mag = np.sqrt(U_q**2 + V_q**2)
    scale = np.percentile(mag, 95) * 0.5
    
    fig, ax = plt.subplots(figsize=(8, 7))
    Q = ax.quiver(
        X_q, Y_q, U_q, V_q, mag,
        angles='xy', scale_units='xy', scale=scale,
        pivot='middle', cmap='viridis', alpha=0.8, width=0.003
    )
    
    plane_info = SLICE_PLANES[slice_dim]
    title = "Velocity Field"
    if redshift is not None:
        title += f" (z={redshift:.2f})"
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    
    cbar = fig.colorbar(Q, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Velocity [{VELOCITY_UNITS}]", fontsize=12)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_divergence(div_field, slice_dim, box_size, redshift=None, save_path=None):
    div_slice = extract_slice(div_field, slice_dim).T
    vmin, vmax = np.percentile(div_slice, [1, 99])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        div_slice, origin='lower', cmap='RdBu_r',
        extent=[0, box_size, 0, box_size],
        vmin=vmin, vmax=vmax
    )
    
    plane_info = SLICE_PLANES[slice_dim]
    title = "Velocity Divergence"
    if redshift is not None:
        title += f" (z={redshift:.2f})"
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"∇·v [{VELOCITY_UNITS}/{AXIS_UNITS}]", fontsize=12)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_shear(shear_field, slice_dim, box_size, redshift=None, save_path=None):
    σ_xx = shear_field[..., 0]
    σ_xy = shear_field[..., 1]
    σ_xz = shear_field[..., 2]
    σ_yy = shear_field[..., 3]
    σ_yz = shear_field[..., 4]
    σ_zz = -(σ_xx + σ_yy)
    
    shear_mag = np.sqrt(
        σ_xx**2 + σ_yy**2 + σ_zz**2 +
        2*(σ_xy**2 + σ_xz**2 + σ_yz**2)
    )
    
    shear_slice = extract_slice(shear_mag, slice_dim).T
    vmin = max(np.min(shear_slice[shear_slice > 0]), 1e-6)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        shear_slice, origin='lower', cmap='plasma',
        norm=colors.LogNorm(vmin=vmin, vmax=shear_slice.max()),
        extent=[0, box_size, 0, box_size]
    )
    
    plane_info = SLICE_PLANES[slice_dim]
    title = "Velocity Shear"
    if redshift is not None:
        title += f" (z={redshift:.2f})"
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"|σ| [{VELOCITY_UNITS}/{AXIS_UNITS}]", fontsize=12)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def process_snapshot(snapshot, redshift):
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")
    
    snapshot_dir = Path(BASE_DATA_DIR) / f"snapdir_{snapshot}"
    
    if not snapshot_dir.exists():
        print(f"  Warning: Directory not found: {snapshot_dir}")
        return False
    
    field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)
    
    file_paths = {
        'density': snapshot_dir / 'output.a_den',
        'velocity': snapshot_dir / 'output.a_vel',
        'divergence': snapshot_dir / 'output.a_velDiv',
        'shear': snapshot_dir / 'output.a_velShear'
    }
    
    fields = {}
    
    try:
        if PROCESS_DENSITY:
            dens = load_binary_field(str(file_paths['density']), field_shape, num_components=1)
            if GAUSSIAN_SMOOTHING_SIGMA > 0:
                dens = gaussian_filter(dens, sigma=GAUSSIAN_SMOOTHING_SIGMA, mode='wrap')
            fields['density'] = dens
        
        if PROCESS_VELOCITY:
            fields['velocity'] = load_binary_field(
                str(file_paths['velocity']), field_shape, num_components=3
            )
        
        if PROCESS_DIVERGENCE:
            fields['divergence'] = load_binary_field(
                str(file_paths['divergence']), field_shape, num_components=1
            )
        
        if PROCESS_SHEAR:
            fields['shear'] = load_binary_field(
                str(file_paths['shear']), field_shape, num_components=5
            )
        
    except FileNotFoundError as e:
        print(f"  Error: Missing files for snapshot {snapshot}")
        return False
    except Exception as e:
        print(f"  Error: Failed to load data for snapshot {snapshot}: {e}")
        return False
    
    output_base = Path(OUTPUT_DIR) / f"snapshot_{snapshot}_z{redshift:.2f}"
    
    for slice_dim in SLICE_PLANES_TO_PLOT:
        plane_name = SLICE_PLANES[slice_dim]['name']
        plane_dir = output_base / plane_name
        
        print(f"  Creating {plane_name} visualizations...")
        
        try:
            if 'density' in fields:
                save_path = plane_dir / f"density_{plane_name}_z{redshift:.2f}.png"
                plot_density(fields['density'], slice_dim, BOX_SIZE, redshift, save_path)
            
            if 'velocity' in fields:
                save_path = plane_dir / f"velocity_{plane_name}_z{redshift:.2f}.png"
                plot_velocity(fields['velocity'], slice_dim, BOX_SIZE, 
                            VELOCITY_QUIVER_STEP, redshift, save_path)
            
            if 'divergence' in fields:
                save_path = plane_dir / f"divergence_{plane_name}_z{redshift:.2f}.png"
                plot_divergence(fields['divergence'], slice_dim, BOX_SIZE, redshift, save_path)
            
            if 'shear' in fields:
                save_path = plane_dir / f"shear_{plane_name}_z{redshift:.2f}.png"
                plot_shear(fields['shear'], slice_dim, BOX_SIZE, redshift, save_path)
                
        except Exception as e:
            print(f"  Error: Failed to create plots for {plane_name}: {e}")
            continue
    
    return True

def main():
    
    fields_to_process = []
    if PROCESS_DENSITY: fields_to_process.append("density")
    if PROCESS_VELOCITY: fields_to_process.append("velocity")
    if PROCESS_DIVERGENCE: fields_to_process.append("divergence")
    if PROCESS_SHEAR: fields_to_process.append("shear")
    
    print("Starting DTFE field visualization")
    print(f"Data directory: {BASE_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing {len(SNAPSHOT_TO_REDSHIFT)} snapshots")
    print(f"Fields: {', '.join(fields_to_process)}")
    
    success_count = 0
    failed = []
    
    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        if process_snapshot(snapshot, redshift):
            success_count += 1
        else:
            failed.append((snapshot, redshift))
    
    print(f"\nCompleted: {success_count}/{len(SNAPSHOT_TO_REDSHIFT)} snapshots processed")
    
    if failed:
        print(f"Failed snapshots: {', '.join(f'{s} (z={z:.2f})' for s, z in failed)}")
    
    plots_per_snapshot = len(SLICE_PLANES_TO_PLOT) * len(fields_to_process)
    total_plots = success_count * plots_per_snapshot
    print(f"Total plots created: {total_plots}")
    print(f"Output saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
