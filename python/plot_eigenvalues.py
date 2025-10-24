"""
Hessian Eigenvalue Analysis Tool

Computes the Hessian matrix of density fields using FFT, calculates eigenvalues
to identify critical points (maxima, minima, saddles), and creates visualizations
of the cosmic web structure.
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from pathlib import Path
import dtfe_functions as dtfe
from dtfe_shared import save_plot_to_multiple_paths

# ============================================================================
# Configuration Section
# ============================================================================

BASE_DATA_DIR = "output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/eigenvalue_analysis"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7  # Mpc
SMOOTHING_SIGMA = 10.0
AXIS_UNITS = "Mpc"

# Which slice planes to visualize? (0=YZ, 1=XZ, 2=XY)
SLICE_PLANES_TO_PLOT = [0, 1, 2]

# Snapshot to redshift mapping
SNAPSHOT_TO_REDSHIFT = dtfe.SNAPSHOT_TO_REDSHIFT

DPI = 300

# ============================================================================

SLICE_PLANES = {
    0: {'name': 'yz_plane', 'axis_labels': ('Y', 'Z')},
    1: {'name': 'xz_plane', 'axis_labels': ('X', 'Z')},
    2: {'name': 'xy_plane', 'axis_labels': ('X', 'Y')}
}

# ============================================================================

def calculate_eigenvalue_fields_for_slice(hessian_components, slice_dim=0, slice_index=None):
    """Calculate eigenvalue fields for a 2D slice using dtfe.extract_2d_slice."""
    hxx = hessian_components['hxx']
    hxy = hessian_components['hxy']
    hxz = hessian_components['hxz']
    hyy = hessian_components['hyy']
    hyz = hessian_components['hyz']
    hzz = hessian_components['hzz']

    # Use dtfe.extract_2d_slice to ensure proper alignment
    hxx_slice = dtfe.extract_2d_slice(hxx, slice_dim, slice_index)
    hxy_slice = dtfe.extract_2d_slice(hxy, slice_dim, slice_index)
    hxz_slice = dtfe.extract_2d_slice(hxz, slice_dim, slice_index)
    hyy_slice = dtfe.extract_2d_slice(hyy, slice_dim, slice_index)
    hyz_slice = dtfe.extract_2d_slice(hyz, slice_dim, slice_index)
    hzz_slice = dtfe.extract_2d_slice(hzz, slice_dim, slice_index)

    ny, nz = hxx_slice.shape

    lambda1_slice = np.zeros((ny, nz))
    lambda2_slice = np.zeros((ny, nz))
    lambda3_slice = np.zeros((ny, nz))

    for i in range(ny):
        for j in range(nz):
            H = np.array([
                [hxx_slice[i, j], hxy_slice[i, j], hxz_slice[i, j]],
                [hxy_slice[i, j], hyy_slice[i, j], hyz_slice[i, j]],
                [hxz_slice[i, j], hyz_slice[i, j], hzz_slice[i, j]]
            ])

            try:
                eigenvalues, _ = eigh(H)
                lambda1_slice[i, j] = eigenvalues[0]
                lambda2_slice[i, j] = eigenvalues[1]
                lambda3_slice[i, j] = eigenvalues[2]
            except np.linalg.LinAlgError:
                lambda1_slice[i, j] = np.nan
                lambda2_slice[i, j] = np.nan
                lambda3_slice[i, j] = np.nan

    return lambda1_slice, lambda2_slice, lambda3_slice

def calculate_derived_fields_for_slice(lambda1_slice, lambda2_slice, lambda3_slice):
    """Calculate derived fields based on eigenvalues."""
    trace_slice = lambda1_slice + lambda2_slice + lambda3_slice
    determinant_slice = lambda1_slice * lambda2_slice * lambda3_slice
    
    mask = (lambda1_slice != 0)
    ellipticity_slice = np.zeros_like(lambda1_slice)
    ellipticity_slice[mask] = lambda3_slice[mask] / lambda1_slice[mask] - 1
    
    shape_parameter_slice = np.zeros_like(trace_slice)
    nonzero_mask = (np.abs(trace_slice) > 1e-10)
    shape_parameter_slice[nonzero_mask] = trace_slice[nonzero_mask] / np.abs(trace_slice[nonzero_mask])
    
    signature_slice = np.zeros_like(lambda1_slice, dtype=int)
    signature_slice += (lambda1_slice > 0).astype(int)
    signature_slice += (lambda2_slice > 0).astype(int)
    signature_slice += (lambda3_slice > 0).astype(int)
    
    return {
        'trace': trace_slice,
        'determinant': determinant_slice,
        'ellipticity': ellipticity_slice,
        'shape_parameter': shape_parameter_slice,
        'classification': signature_slice
    }

def visualize_eigenvalue_slice(eigenvalue_slice, field_name, slice_dim, box_size, redshift=None,
                               save_path=None, colormap='RdBu_r'):
    """Visualize a 2D eigenvalue field."""
    extent = [0, box_size, 0, box_size]

    vmin = np.percentile(eigenvalue_slice, 5)
    vmax = np.percentile(eigenvalue_slice, 95)
    if vmin < 0 and vmax > 0:
        vlim = max(abs(vmin), abs(vmax))
        vmin, vmax = -vlim, vlim

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(eigenvalue_slice.T, origin='lower', cmap=colormap,
                    vmin=vmin, vmax=vmax, extent=extent)

    plane_info = SLICE_PLANES[slice_dim]
    title = field_name
    if redshift is not None:
        title += f' (z={redshift:.2f})'

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(field_name, fontsize=12)

    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        save_plot_to_multiple_paths(fig, save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def process_snapshot(snapshot, redshift):
    """Process a single snapshot for eigenvalue analysis."""

    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")

    snapshot_dir = Path(BASE_DATA_DIR) / f"snapdir_{snapshot}"

    if not snapshot_dir.exists():
        print(f"  Warning: Directory not found")
        return False

    density_file = snapshot_dir / 'output.a_den'
    if not density_file.exists():
        print(f"  Warning: Density file not found")
        return False

    snapshot_output_dir = Path(OUTPUT_DIR) / f"snapshot_{snapshot}_z{redshift:.2f}"
    snapshot_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)

        print(f"  Loading and processing density field")
        density_field = dtfe.load_binary_field(str(density_file), field_shape, num_components=1)
        delta_field = dtfe.calculate_density_contrast(density_field)
        delta_field = dtfe.smooth_field(delta_field, sigma=SMOOTHING_SIGMA)

        print(f"  Computing Hessian matrix")
        hessian_components = dtfe.calculate_hessian_fft(delta_field)

        for slice_dim in SLICE_PLANES_TO_PLOT:
            plane_name = SLICE_PLANES[slice_dim]['name']
            plane_dir = snapshot_output_dir / plane_name

            print(f"  Creating {plane_name} visualizations...")

            print(f"    Calculating eigenvalues for {plane_name}")
            lambda1_slice, lambda2_slice, lambda3_slice = calculate_eigenvalue_fields_for_slice(
                hessian_components, slice_dim
            )

            derived_fields = calculate_derived_fields_for_slice(lambda1_slice, lambda2_slice, lambda3_slice)

            visualize_eigenvalue_slice(
                lambda1_slice, "λ₁ (Smallest Eigenvalue)", slice_dim, BOX_SIZE, redshift=redshift,
                save_path=plane_dir / f"lambda1_{plane_name}_z{redshift:.2f}.png"
            )

            visualize_eigenvalue_slice(
                lambda2_slice, "λ₂ (Middle Eigenvalue)", slice_dim, BOX_SIZE, redshift=redshift,
                save_path=plane_dir / f"lambda2_{plane_name}_z{redshift:.2f}.png"
            )

            visualize_eigenvalue_slice(
                lambda3_slice, "λ₃ (Largest Eigenvalue)", slice_dim, BOX_SIZE, redshift=redshift,
                save_path=plane_dir / f"lambda3_{plane_name}_z{redshift:.2f}.png"
            )

            visualize_eigenvalue_slice(
                derived_fields['trace'], "Trace (∇²δ)", slice_dim, BOX_SIZE, redshift=redshift,
                save_path=plane_dir / f"trace_{plane_name}_z{redshift:.2f}.png"
            )

            visualize_eigenvalue_slice(
                derived_fields['determinant'], "Determinant", slice_dim, BOX_SIZE, redshift=redshift,
                save_path=plane_dir / f"determinant_{plane_name}_z{redshift:.2f}.png"
            )

        return True

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_evolution_summary():
    """Create summary plot of eigenvalue evolution across redshift."""
    
    print("\nCreating evolution summary")
    
    redshifts = []
    mean_lambda1 = []
    mean_lambda2 = []
    mean_lambda3 = []
    maxima_fractions = []
    minima_fractions = []
    
    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        summary_file = Path(OUTPUT_DIR) / f'snapshot_{snapshot}_z{redshift:.2f}' / f'eigenvalue_summary_z{redshift:.2f}.txt'
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    content = f.read()
                
                import re
                lambda1_match = re.search(r'λ₁ range: \[([-e\d.]+), ([-e\d.]+)\]', content)
                lambda2_match = re.search(r'λ₂ range: \[([-e\d.]+), ([-e\d.]+)\]', content)
                lambda3_match = re.search(r'λ₃ range: \[([-e\d.]+), ([-e\d.]+)\]', content)
                maxima_match = re.search(r'Maxima: \d+ \(([\d.]+)%\)', content)
                minima_match = re.search(r'Minima: \d+ \(([\d.]+)%\)', content)
                
                if lambda1_match and lambda2_match and lambda3_match and maxima_match and minima_match:
                    redshifts.append(redshift)
                    mean_lambda1.append((float(lambda1_match.group(1)) + float(lambda1_match.group(2))) / 2)
                    mean_lambda2.append((float(lambda2_match.group(1)) + float(lambda2_match.group(2))) / 2)
                    mean_lambda3.append((float(lambda3_match.group(1)) + float(lambda3_match.group(2))) / 2)
                    maxima_fractions.append(float(maxima_match.group(1)))
                    minima_fractions.append(float(minima_match.group(1)))
                    
            except Exception:
                continue
    
    if len(redshifts) > 0:
        sorted_indices = np.argsort(redshifts)
        redshifts = np.array(redshifts)[sorted_indices]
        mean_lambda1 = np.array(mean_lambda1)[sorted_indices]
        mean_lambda2 = np.array(mean_lambda2)[sorted_indices]
        mean_lambda3 = np.array(mean_lambda3)[sorted_indices]
        maxima_fractions = np.array(maxima_fractions)[sorted_indices]
        minima_fractions = np.array(minima_fractions)[sorted_indices]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(redshifts, mean_lambda1, 'ro-', label='λ₁ (smallest)', markersize=4)
        ax1.plot(redshifts, mean_lambda2, 'go-', label='λ₂ (middle)', markersize=4)
        ax1.plot(redshifts, mean_lambda3, 'bo-', label='λ₃ (largest)', markersize=4)
        ax1.set_xlabel('Redshift z')
        ax1.set_ylabel('Eigenvalue Range Centers')
        ax1.set_title('Evolution of Hessian Eigenvalues')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        
        ax2.plot(redshifts, maxima_fractions, 'ro-', label='Maxima', markersize=4)
        ax2.plot(redshifts, minima_fractions, 'bo-', label='Minima', markersize=4)
        ax2.set_xlabel('Redshift z')
        ax2.set_ylabel('Fraction (%)')
        ax2.set_title('Evolution of Critical Point Types')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()
        
        plt.tight_layout()
        
        evolution_file = Path(OUTPUT_DIR) / 'eigenvalue_evolution_summary.png'
        save_plot_to_multiple_paths(fig, evolution_file, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

def main():
    """Process all snapshots for eigenvalue analysis."""
    
    print("Starting eigenvalue analysis")
    print(f"Data directory: {BASE_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing {len(SNAPSHOT_TO_REDSHIFT)} snapshots")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    processed_snapshots = 0
    failed_snapshots = []
    
    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        if process_snapshot(snapshot, redshift):
            processed_snapshots += 1
        else:
            failed_snapshots.append((snapshot, redshift))
    
    if processed_snapshots > 1:
        try:
            create_evolution_summary()
        except Exception as e:
            print(f"\nError creating evolution summary: {e}")
    
    print(f"\nCompleted: {processed_snapshots}/{len(SNAPSHOT_TO_REDSHIFT)} snapshots processed")
    
    if failed_snapshots:
        print(f"Failed snapshots: {', '.join(f'{s} (z={z:.2f})' for s, z in failed_snapshots)}")
    
    print(f"Output saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()