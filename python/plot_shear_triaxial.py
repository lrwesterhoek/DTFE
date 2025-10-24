"""
Shear Analysis Tool

Analyzes velocity shear tensors from cosmological simulations. Computes eigenvalues,
triaxiality, and vorticity. Supports both 5-component traceless shear and 9-component
full velocity gradient tensors.
"""

import os
import glob
import numpy as np
from numpy.linalg import eigvals
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
from numba import njit, prange
from pathlib import Path
import dtfe_functions as dtfe
from dtfe_shared import save_plot_to_multiple_paths

# ============================================================================
# Configuration Section
# ============================================================================

BASE_DATA_DIR = "output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/shear_analysis"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7  # Mpc
SMOOTHING_SIGMA = 0

DPI = 300

SNAPSHOT_TO_REDSHIFT = dtfe.SNAPSHOT_TO_REDSHIFT

SLICE_PLANES = {
    0: {'name': 'yz_plane', 'axis_labels': ('Y', 'Z')},
    1: {'name': 'xz_plane', 'axis_labels': ('X', 'Z')},
    2: {'name': 'xy_plane', 'axis_labels': ('X', 'Y')}
}

# ============================================================================

def load_multiple_binary_files(file_pattern, full_field_shape, combine_method="sum",
                               num_components=5, dtype=np.float32, **kwargs):
    """Generic function to load and combine multiple binary files."""
    files = sorted(glob.glob(file_pattern))
    n_files = len(files)

    if n_files == 0:
        raise FileNotFoundError(f"No files found matching pattern {file_pattern}")

    if num_components == 1:
        full_field = np.zeros(full_field_shape, dtype=dtype)
    else:
        full_field = np.zeros(full_field_shape + (num_components,), dtype=dtype)

    for i, file in enumerate(files):
        field_part = dtfe.load_binary_field(file, full_field_shape, num_components=num_components,
                                           dtype=dtype, **kwargs)
        full_field += field_part
    
    if combine_method == "average":
        full_field /= n_files
    
    return full_field

def reconstruct_shear_matrix_from_5comp(velShear_field):
    """
    Reconstruct full 3x3 trace-free shear tensor from 5-component field.
    Components ordered as: σ11, σ12, σ13, σ22, σ23
    """
    Sxx = velShear_field[..., 0]
    Sxy = velShear_field[..., 1]
    Sxz = velShear_field[..., 2]
    Syy = velShear_field[..., 3]
    Syz = velShear_field[..., 4]
    Szz = -(Sxx + Syy)

    shape_3d = velShear_field.shape[:3]
    S = np.zeros(shape_3d + (3, 3), dtype=velShear_field.dtype)
    S[..., 0, 0] = Sxx
    S[..., 0, 1] = Sxy
    S[..., 1, 0] = Sxy
    S[..., 0, 2] = Sxz
    S[..., 2, 0] = Sxz
    S[..., 1, 1] = Syy
    S[..., 1, 2] = Syz
    S[..., 2, 1] = Syz
    S[..., 2, 2] = Szz
    
    return S

def reconstruct_full_gradient_matrix(velGrad_field):
    """Reconstruct full 3x3 velocity gradient tensor from 9-component field."""
    shape_3d = velGrad_field.shape[:3]
    grad_tensor = np.zeros(shape_3d + (3, 3), dtype=velGrad_field.dtype)
    
    grad_tensor[..., 0, 0] = velGrad_field[..., 0]
    grad_tensor[..., 0, 1] = velGrad_field[..., 1]
    grad_tensor[..., 0, 2] = velGrad_field[..., 2]
    grad_tensor[..., 1, 0] = velGrad_field[..., 3]
    grad_tensor[..., 1, 1] = velGrad_field[..., 4]
    grad_tensor[..., 1, 2] = velGrad_field[..., 5]
    grad_tensor[..., 2, 0] = velGrad_field[..., 6]
    grad_tensor[..., 2, 1] = velGrad_field[..., 7]
    grad_tensor[..., 2, 2] = velGrad_field[..., 8]
    
    return grad_tensor

def decompose_velocity_gradient(grad_tensor):
    """Decompose velocity gradient into symmetric strain rate and antisymmetric vorticity."""
    strain_rate = 0.5 * (grad_tensor + np.transpose(grad_tensor, axes=(0, 1, 2, 4, 3)))
    vorticity = 0.5 * (grad_tensor - np.transpose(grad_tensor, axes=(0, 1, 2, 4, 3)))
    return strain_rate, vorticity

def compute_traceless_shear(strain_rate):
    """Compute traceless part of strain rate tensor."""
    trace = np.trace(strain_rate, axis1=-2, axis2=-1)
    traceless_shear = strain_rate.copy()
    traceless_shear[..., 0, 0] -= trace / 3.0
    traceless_shear[..., 1, 1] -= trace / 3.0
    traceless_shear[..., 2, 2] -= trace / 3.0
    return traceless_shear

@njit(parallel=True)
def compute_triaxiality(eigenvals):
    """Compute triaxiality parameter T = (λ2 - λ1)/(λ3 - λ1)."""
    nx, ny, nz = eigenvals.shape[:3]
    T = np.zeros((nx, ny, nz), dtype=eigenvals.dtype)
    epsilon = 1e-12

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Extract and sort the 3 eigenvalues at this point
                ev1, ev2, ev3 = eigenvals[i, j, k, 0], eigenvals[i, j, k, 1], eigenvals[i, j, k, 2]

                # Manual sort of 3 values (ascending order)
                if ev1 > ev2:
                    ev1, ev2 = ev2, ev1
                if ev2 > ev3:
                    ev2, ev3 = ev3, ev2
                if ev1 > ev2:
                    ev1, ev2 = ev2, ev1

                # Now: ev1 <= ev2 <= ev3
                lambda1, lambda2, lambda3 = ev1, ev2, ev3

                numerator = lambda2 - lambda1
                denominator = lambda3 - lambda1

                # Safe division: avoid divide-by-zero
                if abs(denominator) > epsilon:
                    T[i, j, k] = numerator / denominator
                else:
                    T[i, j, k] = 0.0

    return T

def visualize_shear_magnitude(shear_tensor, slice_dim=2, box_size=35, redshift=None, save_path=None):
    """Visualize shear magnitude as 2D slice."""
    shear_magnitude = np.sqrt(np.sum(shear_tensor**2, axis=(-2, -1)))
    shear_slice = dtfe.extract_2d_slice(shear_magnitude, slice_dim).T

    extent = [0, box_size, 0, box_size]

    # Use percentile-based normalization for better contrast
    vmin = np.percentile(shear_slice, 1)
    vmax = np.percentile(shear_slice, 99)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        shear_slice, origin='lower', cmap='plasma',
        vmin=vmin, vmax=vmax, extent=extent
    )
    
    plane_info = SLICE_PLANES[slice_dim]
    title = 'Shear Magnitude'
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [Mpc]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [Mpc]", fontsize=12)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Shear Magnitude [km/s/Mpc]', fontsize=12)
    
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    if save_path:
        save_plot_to_multiple_paths(fig, save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def visualize_triaxiality(T_field, slice_dim=2, box_size=35, redshift=None, save_path=None):
    """Visualize triaxiality field as 2D slice."""
    T_slice = dtfe.extract_2d_slice(T_field, slice_dim).T

    extent = [0, box_size, 0, box_size]

    # Use percentile-based normalization for better contrast
    vmin = np.percentile(T_slice, 1)
    vmax = np.percentile(T_slice, 99)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(T_slice, origin='lower', cmap='viridis',
                   vmin=vmin, vmax=vmax, extent=extent)
    
    plane_info = SLICE_PLANES[slice_dim]
    title = 'Triaxiality'
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [Mpc]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [Mpc]", fontsize=12)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Triaxiality', fontsize=12)
    
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    if save_path:
        save_plot_to_multiple_paths(fig, save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def process_snapshot(snapshot, redshift):
    """Process single snapshot for comprehensive shear analysis."""
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")
    
    data_path = Path(BASE_DATA_DIR) / f'snapdir_{snapshot}'
    
    if not data_path.exists():
        print(f"Path does not exist: {data_path}")
        return
    
    field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)
    
    shear5_pattern = str(data_path / 'output.a_velShear')
    grad9_pattern = str(data_path / 'output.a_velGrad')
    velDiv_pattern = str(data_path / 'output.a_velDiv')
    
    has_shear5 = bool(glob.glob(shear5_pattern))
    has_grad9 = bool(glob.glob(grad9_pattern))
    has_velDiv = bool(glob.glob(velDiv_pattern))
    
    if not (has_shear5 or has_grad9):
        print("No velocity shear/gradient files found")
        return
    
    if not has_velDiv:
        print("No velocity divergence files found")
        return
    
    output_path = Path(OUTPUT_DIR) / f'snapshot_{snapshot}_z{redshift:.2f}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Loading velocity divergence field")
        velDiv_field = load_multiple_binary_files(
            velDiv_pattern, field_shape, combine_method="sum",
            num_components=1, try_infer_shape=True
        )

        if has_grad9:
            print("Loading 9-component velocity gradient field")
            velGrad_field = load_multiple_binary_files(
                grad9_pattern, field_shape, combine_method="average",
                num_components=9, try_infer_shape=True
            )

            print("Reconstructing velocity gradient tensor")
            grad_tensor = reconstruct_full_gradient_matrix(velGrad_field)

            print("Decomposing into strain rate and vorticity")
            strain_rate, vorticity = decompose_velocity_gradient(grad_tensor)

            print("Computing traceless shear from strain rate")
            shear_tensor = compute_traceless_shear(strain_rate)
        else:
            print("Loading 5-component velocity shear field")
            velShear_field = load_multiple_binary_files(
                shear5_pattern, field_shape, combine_method="sum",
                num_components=5, try_infer_shape=True
            )
            
            print("Reconstructing traceless shear tensor")
            shear_tensor = reconstruct_shear_matrix_from_5comp(velShear_field)
            vorticity = None
            strain_rate = None
        
        print("Computing eigenvalues of shear tensor")
        # Compute eigenvalues using standard library
        nx, ny, nz = shear_tensor.shape[:3]
        reshaped = shear_tensor.reshape(-1, 3, 3)
        eigenvals_flat = np.linalg.eigvalsh(reshaped)
        eigenvals = eigenvals_flat.reshape(nx, ny, nz, 3)
        
        print("Computing triaxiality field")
        T_field = compute_triaxiality(eigenvals)
        T_field_smoothed = dtfe.smooth_field(T_field, sigma=SMOOTHING_SIGMA)
        
        for slice_dim in [0, 1, 2]:
            plane_name = SLICE_PLANES[slice_dim]['name']
            print(f"Creating {plane_name} visualizations")
            
            plane_output_path = output_path / plane_name
            plane_output_path.mkdir(parents=True, exist_ok=True)
            
            visualize_shear_magnitude(
                shear_tensor,
                slice_dim=slice_dim,
                box_size=BOX_SIZE,
                redshift=redshift,
                save_path=plane_output_path / f'shear_magnitude_{plane_name}_z{redshift:.2f}.png'
            )
            
            visualize_triaxiality(
                T_field_smoothed,
                slice_dim=slice_dim,
                box_size=BOX_SIZE,
                redshift=redshift,
                save_path=plane_output_path / f'triaxiality_{plane_name}_z{redshift:.2f}.png'
            )
        
        summary_file = output_path / f'shear_analysis_summary_z{redshift:.2f}.txt'
        shear_magnitude = np.sqrt(np.sum(shear_tensor**2, axis=(-2, -1)))
        eig_flat = eigenvals.flatten()
        
        with open(summary_file, 'w') as f:
            f.write(f"Shear Analysis Summary\n")
            f.write(f"Snapshot: {snapshot}, Redshift: {redshift:.2f}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Data Source: {'9-component gradient' if has_grad9 else '5-component shear'}\n\n")
            f.write(f"Shear Statistics:\n")
            f.write(f"  Mean magnitude: {np.mean(shear_magnitude):.3f} km/s/Mpc\n")
            f.write(f"  Std magnitude: {np.std(shear_magnitude):.3f} km/s/Mpc\n")
            f.write(f"  Max magnitude: {np.max(shear_magnitude):.3f} km/s/Mpc\n\n")
            f.write(f"Eigenvalue Statistics:\n")
            f.write(f"  Mean eigenvalue: {np.mean(eig_flat):.3f} km/s/Mpc\n")
            f.write(f"  Std eigenvalue: {np.std(eig_flat):.3f} km/s/Mpc\n")
            f.write(f"  Min eigenvalue: {np.min(eig_flat):.3f} km/s/Mpc\n")
            f.write(f"  Max eigenvalue: {np.max(eig_flat):.3f} km/s/Mpc\n\n")
            f.write(f"Triaxiality Statistics:\n")
            f.write(f"  Mean triaxiality: {np.mean(T_field):.3f}\n")
            f.write(f"  Std triaxiality: {np.std(T_field):.3f}\n")
            f.write(f"  Min triaxiality: {np.min(T_field):.3f}\n")
            f.write(f"  Max triaxiality: {np.max(T_field):.3f}\n")
            
            if vorticity is not None:
                vort_magnitude = np.sqrt(np.sum(vorticity**2, axis=(-2, -1)))
                f.write(f"\nVorticity Statistics:\n")
                f.write(f"  Mean magnitude: {np.mean(vort_magnitude):.3f} km/s/Mpc\n")
                f.write(f"  Std magnitude: {np.std(vort_magnitude):.3f} km/s/Mpc\n")
                f.write(f"  Max magnitude: {np.max(vort_magnitude):.3f} km/s/Mpc\n")
        
        print(f"Completed snapshot {snapshot}")
        
    except Exception as e:
        print(f"Error processing snapshot {snapshot}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to process all snapshots for shear analysis."""
    print(f"Starting shear analysis")
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
    
    print(f"\nAnalysis complete")
    print(f"Successfully processed: {processed_count}/{len(SNAPSHOT_TO_REDSHIFT)} snapshots")
    
    if failed_snapshots:
        print(f"Failed snapshots: {len(failed_snapshots)}")
        for snapshot, redshift in failed_snapshots:
            print(f"  Snapshot {snapshot} (z={redshift:.2f})")
    
    print(f"\nOutput saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()