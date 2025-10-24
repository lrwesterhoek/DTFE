"""
DTFE Shared Library

Common functions for DTFE (Delaunay Tessellation Field Estimator) analysis scripts.
Provides utilities for loading binary data, computing density contrasts, Hessian matrices,
eigenvalues, and other field operations.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter
from numba import njit, prange
import math
from pathlib import Path

# ============================================================================
# Binary File Loading Functions
# ============================================================================

def load_binary_field(binary_file, field_shape, num_components=1, dtype=np.float32, try_infer_shape=False):
    """
    Generic loader for binary field data.

    Parameters
    ----------
    binary_file : str or Path
        Path to binary file
    field_shape : tuple
        Expected 3D shape (nx, ny, nz)
    num_components : int
        Number of components per grid point (1 for scalar, 3 for vector, etc.)
    dtype : numpy dtype
        Data type of binary file
    try_infer_shape : bool
        If True, attempt to infer shape from file size when mismatch occurs

    Returns
    -------
    ndarray
        Reshaped field data. Shape is field_shape for scalar, field_shape + (num_components,) for multi-component
    """
    data = np.fromfile(binary_file, dtype=dtype)
    expected_size = np.prod(field_shape) * num_components

    if data.size != expected_size:
        msg = (f"Data size in {binary_file} is {data.size} elements, "
               f"but expected {expected_size} for shape {field_shape} with {num_components} component(s).")

        if try_infer_shape:
            total_elements = data.size / num_components
            cube_root = round(total_elements ** (1/3))

            if cube_root**3 * num_components == data.size:
                print(f"Warning: {msg}")
                print(f"Inferred cubic shape: ({cube_root}, {cube_root}, {cube_root})")
                field_shape = (cube_root, cube_root, cube_root)
            else:
                raise ValueError(msg)
        else:
            raise ValueError(msg)

    if num_components == 1:
        return data.reshape(field_shape)
    else:
        return data.reshape(field_shape + (num_components,))


def load_binary_density(binary_file, field_shape, dtype=np.float32, try_infer_shape=False):
    """Load binary density file (scalar field)."""
    return load_binary_field(binary_file, field_shape, num_components=1,
                            dtype=dtype, try_infer_shape=try_infer_shape)


def load_binary_velocity(binary_file, field_shape, dtype=np.float32, try_infer_shape=False):
    """Load binary velocity file (3-component vector field)."""
    return load_binary_field(binary_file, field_shape, num_components=3,
                            dtype=dtype, try_infer_shape=try_infer_shape)


def load_binary_velDiv(binary_file, field_shape, dtype=np.float32, try_infer_shape=False):
    """Load binary velocity divergence file (scalar field)."""
    return load_binary_field(binary_file, field_shape, num_components=1,
                            dtype=dtype, try_infer_shape=try_infer_shape)


def load_binary_velShear(binary_file, field_shape, dtype=np.float32, try_infer_shape=False):
    """Load binary velocity shear file (5-component symmetric trace-free tensor)."""
    return load_binary_field(binary_file, field_shape, num_components=5,
                            dtype=dtype, try_infer_shape=try_infer_shape)


def load_binary_velGrad(binary_file, field_shape, dtype=np.float32, try_infer_shape=False):
    """Load binary velocity gradient file (9-component tensor)."""
    return load_binary_field(binary_file, field_shape, num_components=9,
                            dtype=dtype, try_infer_shape=try_infer_shape)


# ============================================================================
# Density Contrast Functions
# ============================================================================

def calculate_density_contrast(density_field):
    """
    Calculate density contrast: delta = (rho - <rho>) / <rho>

    Parameters
    ----------
    density_field : ndarray
        Density field

    Returns
    -------
    ndarray
        Density contrast field
    """
    mean_density = np.mean(density_field)
    if mean_density == 0:
        return density_field
    return (density_field - mean_density) / mean_density


# ============================================================================
# Slice Extraction Functions
# ============================================================================

def extract_2d_slice(field, slice_dim=2, slice_index=None):
    """
    Extract a 2D slice from a 3D field.

    Parameters
    ----------
    field : ndarray
        3D field array
    slice_dim : int
        Dimension to slice (0=YZ plane, 1=XZ plane, 2=XY plane)
    slice_index : int, optional
        Index along slice dimension. If None, uses midpoint.

    Returns
    -------
    ndarray
        2D slice of the field
    """
    if slice_index is None:
        slice_index = field.shape[slice_dim] // 2

    if slice_dim == 0:
        return field[slice_index, :, :]
    elif slice_dim == 1:
        return field[:, slice_index, :]
    elif slice_dim == 2:
        return field[:, :, slice_index]
    else:
        raise ValueError("slice_dim must be 0, 1, or 2")


def extract_velocity_slice(velocity_field, slice_dim=2):
    """
    Extract velocity components for a 2D slice.

    Parameters
    ----------
    velocity_field : ndarray
        3D velocity field with shape (nx, ny, nz, 3)
    slice_dim : int
        Dimension to slice (0=YZ, 1=XZ, 2=XY)

    Returns
    -------
    X, Y : ndarray
        Meshgrid coordinates
    U, V : ndarray
        Velocity components in the slice plane
    """
    idx = velocity_field.shape[slice_dim] // 2

    if slice_dim == 0:  # YZ plane
        slice_data = velocity_field[idx, :, :]
        U, V = slice_data[..., 1], slice_data[..., 2]
    elif slice_dim == 1:  # XZ plane
        slice_data = velocity_field[:, idx, :]
        U, V = slice_data[..., 0], slice_data[..., 2]
    else:  # XY plane
        slice_data = velocity_field[:, :, idx]
        U, V = slice_data[..., 0], slice_data[..., 1]

    ny, nx = U.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    return X, Y, U, V


# ============================================================================
# Smoothing Functions
# ============================================================================

def smooth_field(field, sigma, mode='wrap'):
    """
    Apply Gaussian smoothing to a field.

    Parameters
    ----------
    field : ndarray
        Input field
    sigma : float
        Standard deviation for Gaussian kernel
    mode : str
        Mode for handling boundaries (default: 'wrap' for periodic)

    Returns
    -------
    ndarray
        Smoothed field
    """
    if sigma <= 0:
        return field
    return gaussian_filter(field, sigma=sigma, mode=mode)


# ============================================================================
# FFT-Based Derivative Functions
# ============================================================================

def calculate_gradient_fft(field):
    """
    Calculate gradient using FFT.

    Parameters
    ----------
    field : ndarray
        3D scalar field

    Returns
    -------
    grad_x, grad_y, grad_z : ndarray
        Gradient components
    """
    nx, ny, nz = field.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny)
    kz = 2 * np.pi * np.fft.fftfreq(nz)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    field_fft = np.fft.fftn(field)

    grad_x_fft = 1j * KX * field_fft
    grad_y_fft = 1j * KY * field_fft
    grad_z_fft = 1j * KZ * field_fft

    grad_x = np.real(np.fft.ifftn(grad_x_fft))
    grad_y = np.real(np.fft.ifftn(grad_y_fft))
    grad_z = np.real(np.fft.ifftn(grad_z_fft))

    return grad_x, grad_y, grad_z


def calculate_gradient_magnitude(grad_x, grad_y, grad_z):
    """Calculate gradient magnitude from components."""
    return np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)


def calculate_hessian_fft_full(field):
    """
    Calculate full Hessian matrix for entire field using FFT.
    Use this when you need Hessian at ALL points (e.g., eigenvalue fields).

    Parameters
    ----------
    field : ndarray
        3D scalar field

    Returns
    -------
    dict
        Dictionary with keys 'hxx', 'hxy', 'hxz', 'hyy', 'hyz', 'hzz'
        Each value is a 3D array of the same shape as input
    """
    nx, ny, nz = field.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny)
    kz = 2 * np.pi * np.fft.fftfreq(nz)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    field_fft = np.fft.fftn(field)

    hxx = np.real(np.fft.ifftn(-(KX * KX) * field_fft))
    hxy = np.real(np.fft.ifftn(-(KX * KY) * field_fft))
    hxz = np.real(np.fft.ifftn(-(KX * KZ) * field_fft))
    hyy = np.real(np.fft.ifftn(-(KY * KY) * field_fft))
    hyz = np.real(np.fft.ifftn(-(KY * KZ) * field_fft))
    hzz = np.real(np.fft.ifftn(-(KZ * KZ) * field_fft))

    return {
        'hxx': hxx, 'hxy': hxy, 'hxz': hxz,
        'hyy': hyy, 'hyz': hyz, 'hzz': hzz
    }


def calculate_hessian_at_points(field, coordinates):
    """
    Calculate Hessian matrix at specific points only using FFT.
    More efficient when you only need Hessian at sparse locations (e.g., minima).

    Parameters
    ----------
    field : ndarray
        3D scalar field
    coordinates : ndarray
        Array of shape (N, 3) with grid coordinates

    Returns
    -------
    list of dict
        List of dictionaries, each containing:
        - 'coords': tuple of (i, j, k)
        - 'hessian': 3x3 Hessian matrix at that point
        - 'eigenvalues': sorted eigenvalues
        - 'eigenvectors': corresponding eigenvectors
    """
    # First compute full Hessian components (still need FFT for all)
    hessian_components = calculate_hessian_fft_full(field)

    # Extract only at specified points
    hxx = hessian_components['hxx']
    hxy = hessian_components['hxy']
    hxz = hessian_components['hxz']
    hyy = hessian_components['hyy']
    hyz = hessian_components['hyz']
    hzz = hessian_components['hzz']

    results = []
    for coord in coordinates:
        i, j, k = int(coord[0]), int(coord[1]), int(coord[2])

        H = np.array([
            [hxx[i, j, k], hxy[i, j, k], hxz[i, j, k]],
            [hxy[i, j, k], hyy[i, j, k], hyz[i, j, k]],
            [hxz[i, j, k], hyz[i, j, k], hzz[i, j, k]]
        ])

        try:
            eigenvals, eigenvecs = np.linalg.eigh(H)
            results.append({
                'coords': tuple(coord),
                'hessian': H,
                'eigenvalues': eigenvals,
                'eigenvectors': eigenvecs
            })
        except np.linalg.LinAlgError:
            # Skip points with singular matrices
            continue

    return results


# ============================================================================
# Eigenvalue Computation (Optimized)
# ============================================================================

@njit
def eigenvalues_3x3_sym(a, b, c, d, e, f):
    """
    Compute eigenvalues of symmetric 3x3 matrix using analytical solution.
    Matrix is: [[a, d, f], [d, b, e], [f, e, c]]

    Parameters
    ----------
    a, b, c : float
        Diagonal elements
    d, e, f : float
        Off-diagonal elements

    Returns
    -------
    eig1, eig2, eig3 : float
        Three eigenvalues
    """
    q = (a + b + c) / 3.0
    a_ = a - q
    b_ = b - q
    c_ = c - q
    p2 = a_ * a_ + b_ * b_ + c_ * c_ + 2.0 * (d * d + e * e + f * f)
    p = math.sqrt(p2 / 6.0)

    if p == 0.0:
        return q, q, q

    detA_q = a_ * b_ * c_ + 2.0 * d * e * f - a_ * e * e - b_ * f * f - c_ * d * d
    r = detA_q / (2.0 * p * p * p)

    if r < -1.0:
        r = -1.0
    elif r > 1.0:
        r = 1.0

    theta = math.acos(r) / 3.0
    eig1 = q + 2.0 * p * math.cos(theta)
    eig3 = q + 2.0 * p * math.cos(theta + (2.0 * math.pi / 3.0))
    eig2 = 3.0 * q - eig1 - eig3

    return eig1, eig2, eig3


@njit(parallel=True)
def compute_eigenvalue_field(hessian_components):
    """
    Compute eigenvalues for each point in field (optimized with Numba).
    Use for full-field eigenvalue analysis.

    Parameters
    ----------
    hessian_components : dict
        Dictionary with 'hxx', 'hxy', 'hxz', 'hyy', 'hyz', 'hzz' arrays

    Returns
    -------
    ndarray
        Array of shape (nx, ny, nz, 3) with eigenvalues at each point
    """
    hxx = hessian_components['hxx']
    hxy = hessian_components['hxy']
    hxz = hessian_components['hxz']
    hyy = hessian_components['hyy']
    hyz = hessian_components['hyz']
    hzz = hessian_components['hzz']

    nx, ny, nz = hxx.shape
    eigenvals = np.empty((nx, ny, nz, 3), dtype=hxx.dtype)

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                a = hxx[i, j, k]
                d = hxy[i, j, k]
                f = hxz[i, j, k]
                b = hyy[i, j, k]
                e = hyz[i, j, k]
                c = hzz[i, j, k]

                eig1, eig2, eig3 = eigenvalues_3x3_sym(a, b, c, d, e, f)
                eigenvals[i, j, k, 0] = eig1
                eigenvals[i, j, k, 1] = eig2
                eigenvals[i, j, k, 2] = eig3

    return eigenvals


# ============================================================================
# Critical Point Detection
# ============================================================================

def find_local_minima(field, footprint_size=3, mode='wrap'):
    """
    Find local minima in a field.

    Parameters
    ----------
    field : ndarray
        Input field (2D or 3D)
    footprint_size : int
        Size of the local neighborhood
    mode : str
        Boundary mode for filter

    Returns
    -------
    minima_mask : ndarray
        Boolean mask of minima locations
    minima_coords : ndarray
        Array of coordinates where minima occur, shape (N, ndim)
    """
    local_min = minimum_filter(field, size=footprint_size, mode=mode)
    minima_mask = (field == local_min)
    minima_coords = np.array(np.where(minima_mask)).T
    return minima_mask, minima_coords


def find_local_maxima(field, footprint_size=3, mode='wrap'):
    """
    Find local maxima in a field.

    Parameters
    ----------
    field : ndarray
        Input field (2D or 3D)
    footprint_size : int
        Size of the local neighborhood
    mode : str
        Boundary mode for filter

    Returns
    -------
    maxima_mask : ndarray
        Boolean mask of maxima locations
    maxima_coords : ndarray
        Array of coordinates where maxima occur, shape (N, ndim)
    """
    local_max = maximum_filter(field, size=footprint_size, mode=mode)
    maxima_mask = (field == local_max)
    maxima_coords = np.array(np.where(maxima_mask)).T
    return maxima_mask, maxima_coords


def find_critical_points_with_hessian(field, hessian_components, gradient_threshold=0.1,
                                     min_distance=3, mode='wrap'):
    """
    Find critical points using gradient magnitude and classify using Hessian.

    Parameters
    ----------
    field : ndarray
        3D scalar field
    hessian_components : dict
        Hessian components from calculate_hessian_fft_full
    gradient_threshold : float
        Maximum gradient magnitude for critical points
    min_distance : int
        Minimum distance between critical points
    mode : str
        Boundary mode

    Returns
    -------
    dict
        Dictionary with 'maxima', 'minima', 'saddle1', 'saddle2' masks
    """
    # Compute gradient
    grad_x, grad_y, grad_z = calculate_gradient_fft(field)
    grad_mag = calculate_gradient_magnitude(grad_x, grad_y, grad_z)

    # Normalize gradient
    grad_max = np.max(grad_mag)
    if grad_max > 0:
        grad_norm = grad_mag / grad_max
    else:
        grad_norm = grad_mag

    # Find points with small gradient
    critical_mask = grad_norm < gradient_threshold

    # Apply minimum distance filter
    if min_distance > 0:
        inverted_grad = grad_max - grad_mag
        local_maxima = maximum_filter(inverted_grad, size=min_distance, mode=mode) == inverted_grad
        critical_mask = critical_mask & local_maxima

    # Classify using Hessian eigenvalues
    critical_indices = np.where(critical_mask)
    num_critical = len(critical_indices[0])

    maxima_mask = np.zeros_like(critical_mask, dtype=bool)
    minima_mask = np.zeros_like(critical_mask, dtype=bool)
    saddle1_mask = np.zeros_like(critical_mask, dtype=bool)
    saddle2_mask = np.zeros_like(critical_mask, dtype=bool)

    # Extract Hessian components
    hxx = hessian_components['hxx']
    hxy = hessian_components['hxy']
    hxz = hessian_components['hxz']
    hyy = hessian_components['hyy']
    hyz = hessian_components['hyz']
    hzz = hessian_components['hzz']

    # Process in batches for efficiency
    batch_size = 10000
    for start_idx in range(0, num_critical, batch_size):
        end_idx = min(start_idx + batch_size, num_critical)

        i_batch = critical_indices[0][start_idx:end_idx]
        j_batch = critical_indices[1][start_idx:end_idx]
        k_batch = critical_indices[2][start_idx:end_idx]

        batch_size_actual = len(i_batch)

        hessian_batch = np.zeros((batch_size_actual, 3, 3))
        hessian_batch[:, 0, 0] = hxx[i_batch, j_batch, k_batch]
        hessian_batch[:, 0, 1] = hxy[i_batch, j_batch, k_batch]
        hessian_batch[:, 0, 2] = hxz[i_batch, j_batch, k_batch]
        hessian_batch[:, 1, 0] = hxy[i_batch, j_batch, k_batch]
        hessian_batch[:, 1, 1] = hyy[i_batch, j_batch, k_batch]
        hessian_batch[:, 1, 2] = hyz[i_batch, j_batch, k_batch]
        hessian_batch[:, 2, 0] = hxz[i_batch, j_batch, k_batch]
        hessian_batch[:, 2, 1] = hyz[i_batch, j_batch, k_batch]
        hessian_batch[:, 2, 2] = hzz[i_batch, j_batch, k_batch]

        eigenvals_batch = np.linalg.eigvals(hessian_batch)

        positive_count = np.sum(eigenvals_batch > 0, axis=1)
        negative_count = np.sum(eigenvals_batch < 0, axis=1)

        for idx in range(batch_size_actual):
            i, j, k = i_batch[idx], j_batch[idx], k_batch[idx]
            pos_count = positive_count[idx]
            neg_count = negative_count[idx]

            if pos_count == 3 and neg_count == 0:
                minima_mask[i, j, k] = True
            elif pos_count == 0 and neg_count == 3:
                maxima_mask[i, j, k] = True
            elif pos_count == 2 and neg_count == 1:
                saddle1_mask[i, j, k] = True
            elif pos_count == 1 and neg_count == 2:
                saddle2_mask[i, j, k] = True

    return {
        'maxima': maxima_mask,
        'minima': minima_mask,
        'saddle1': saddle1_mask,
        'saddle2': saddle2_mask
    }


# ============================================================================
# Cosmology Functions
# ============================================================================

def get_cosmology_params(z, H0=67.74, Omega_m0=0.3089, Omega_Lambda0=0.6911):
    """
    Compute cosmological parameters at given redshift.

    Parameters
    ----------
    z : float
        Redshift
    H0 : float
        Hubble constant at z=0 in km/s/Mpc
    Omega_m0 : float
        Matter density parameter at z=0
    Omega_Lambda0 : float
        Dark energy density parameter at z=0

    Returns
    -------
    dict
        Dictionary containing:
        - 'a': scale factor
        - 'H_z': Hubble parameter at z
        - 'Omega_m_z': Matter density parameter at z
        - 'f_growth': Growth rate
        - 'slope_theory': Theoretical velocity divergence-density slope
    """
    a_scale = 1.0 / (1.0 + z)
    H_z = H0 * np.sqrt(Omega_m0 * (1 + z)**3 + Omega_Lambda0)
    Omega_m_z = Omega_m0 * (1 + z)**3 / (Omega_m0 * (1 + z)**3 + Omega_Lambda0)
    f_growth = Omega_m_z**0.55
    slope_theory = -a_scale * H_z * f_growth

    return {
        'a': a_scale,
        'H_z': H_z,
        'Omega_m_z': Omega_m_z,
        'f_growth': f_growth,
        'slope_theory': slope_theory
    }


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_output_dir(path):
    """Ensure output directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


# Common snapshot-to-redshift mapping for TNG50-3-Dark
SNAPSHOT_TO_REDSHIFT = {
    '000': 20.05, '004': 10.00, '008': 8.01, '013': 6.01,
    '017': 5.00, '021': 4.01, '025': 3.01, '033': 2.00,
    '040': 1.50, '050': 1.00, '067': 0.50, '072': 0.40,
    '078': 0.30, '084': 0.20, '091': 0.10, '099': 0.00
}

# Common slice plane configurations
SLICE_PLANES = {
    0: {'name': 'yz_plane', 'axis_labels': ('Y', 'Z')},
    1: {'name': 'xz_plane', 'axis_labels': ('X', 'Z')},
    2: {'name': 'xy_plane', 'axis_labels': ('X', 'Y')}
}

# Additional save path for all figures
ADDITIONAL_SAVE_PATH = '/Users/luukw/Documents/Rijksuniversiteit Groningen/Physics/Year 3/Bachelor Research Project PH/Fate_of_Cosmic_Voids__Collapse_or_Expansion_in_a_Dark_Universe/Figures'


# ============================================================================
# Plot Saving Functions
# ============================================================================

def save_plot_to_multiple_paths(fig, primary_path, dpi=300, **kwargs):
    """
    Save a matplotlib figure to both the primary path and an additional path.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    primary_path : str or Path
        The primary save path
    dpi : int
        DPI for saving
    **kwargs : additional keyword arguments
        Additional arguments passed to savefig (e.g., bbox_inches='tight')
    """
    import matplotlib.pyplot as plt

    # Ensure primary directory exists and save
    primary_path = Path(primary_path)
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(primary_path, dpi=dpi, **kwargs)

    # Construct additional path maintaining relative structure from 'figures'
    primary_str = str(primary_path)

    # Find the figures directory in the path
    if '/figures/' in primary_str or '\\figures\\' in primary_str:
        # Extract everything after 'figures/'
        parts = primary_str.replace('\\', '/').split('/figures/')
        if len(parts) > 1:
            relative_path = parts[1]
        else:
            # Fallback: just use the filename
            relative_path = primary_path.name
    else:
        # If 'figures' not in path, just use the filename
        relative_path = primary_path.name

    # Create additional save path
    additional_path = Path(ADDITIONAL_SAVE_PATH) / relative_path
    additional_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(additional_path, dpi=dpi, **kwargs)
