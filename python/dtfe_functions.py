"""
DTFE Functions Library

Common functions for DTFE (Delaunay Tessellation Field Estimator) analysis scripts.
Provides utilities for loading binary data, computing density contrasts, Hessian matrices,
eigenvalues, and other field operations.

All functions use standard library implementations (numpy, scipy) for maximum
compatibility and maintainability.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter
from pathlib import Path

# ============================================================================
# Binary File Loading Functions
# ============================================================================

def load_binary_field(binary_file, field_shape, num_components=1, dtype=np.float32, try_infer_shape=False):
    """
    Universal loader for all binary field data types.

    This function handles all types of binary files: density (1 component),
    velocity (3 components), velocity divergence (1 component), velocity shear
    (5 components), and velocity gradient (9 components).

    Parameters
    ----------
    binary_file : str or Path
        Path to binary file
    field_shape : tuple
        Expected 3D shape (nx, ny, nz)
    num_components : int
        Number of components per grid point:
        - 1 for scalar fields (density, divergence)
        - 3 for vector fields (velocity)
        - 5 for symmetric trace-free tensors (shear)
        - 9 for full tensors (velocity gradient)
    dtype : numpy dtype
        Data type of binary file (default: np.float32)
    try_infer_shape : bool
        If True, attempt to infer shape from file size when mismatch occurs

    Returns
    -------
    ndarray
        Reshaped field data.
        - Shape is field_shape for scalar fields
        - Shape is field_shape + (num_components,) for multi-component fields
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


def calculate_hessian_fft(field, coordinates=None):
    """
    Calculate Hessian matrix using FFT - unified function for both full field and sparse points.

    This function replaces the old calculate_hessian_fft_full() and calculate_hessian_at_points().
    It computes the full Hessian field via FFT, then optionally extracts only specific points.

    Parameters
    ----------
    field : ndarray
        3D scalar field
    coordinates : ndarray or None
        Optional array of shape (N, 3) with grid coordinates.
        - If None: returns full Hessian field (dict of 3D arrays)
        - If provided: returns Hessian info at specified points only (list of dicts)

    Returns
    -------
    dict or list
        If coordinates is None:
            Dictionary with keys 'hxx', 'hxy', 'hxz', 'hyy', 'hyz', 'hzz'
            Each value is a 3D array of the same shape as input field

        If coordinates provided:
            List of dictionaries, each containing:
            - 'coords': tuple of (i, j, k)
            - 'hessian': 3x3 Hessian matrix at that point
            - 'eigenvalues': sorted eigenvalues
            - 'eigenvectors': corresponding eigenvectors
    """
    # Always compute full Hessian components via FFT
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

    hessian_components = {
        'hxx': hxx, 'hxy': hxy, 'hxz': hxz,
        'hyy': hyy, 'hyz': hyz, 'hzz': hzz
    }

    # If no coordinates specified, return full field
    if coordinates is None:
        return hessian_components

    # Otherwise, extract at specified points
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
# Eigenvalue Computation
# ============================================================================

def compute_eigenvalue_field(hessian_components):
    """
    Compute eigenvalues for each point in field using standard numpy.linalg.

    This function uses numpy's standard eigenvalue solver which is well-tested
    and maintained by the NumPy community.

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

    # Construct full Hessian matrices for all points
    hessian_matrices = np.zeros((nx, ny, nz, 3, 3), dtype=hxx.dtype)
    hessian_matrices[:, :, :, 0, 0] = hxx
    hessian_matrices[:, :, :, 0, 1] = hxy
    hessian_matrices[:, :, :, 0, 2] = hxz
    hessian_matrices[:, :, :, 1, 0] = hxy  # Symmetric
    hessian_matrices[:, :, :, 1, 1] = hyy
    hessian_matrices[:, :, :, 1, 2] = hyz
    hessian_matrices[:, :, :, 2, 0] = hxz  # Symmetric
    hessian_matrices[:, :, :, 2, 1] = hyz  # Symmetric
    hessian_matrices[:, :, :, 2, 2] = hzz

    # Compute eigenvalues for all points
    # Reshape to 2D for vectorized eigenvalue computation
    reshaped = hessian_matrices.reshape(-1, 3, 3)
    eigenvals_flat = np.linalg.eigvalsh(reshaped)  # Use eigvalsh for symmetric matrices
    eigenvals = eigenvals_flat.reshape(nx, ny, nz, 3)

    return eigenvals


def compute_eigenvalues_at_minima(hessian_components, minima_coords):
    """
    Compute eigenvalues and eigenvectors at specified minima using standard library.

    Parameters
    ----------
    hessian_components : dict
        Dict with 'hxx', 'hxy', 'hxz', 'hyy', 'hyz', 'hzz' arrays
    minima_coords : ndarray
        Array of shape (N, 3) with coordinates

    Returns
    -------
    eigenvalues : list of ndarray
        List of eigenvalue arrays (sorted, shape=(3,)) for each minimum
    eigenvectors : list of ndarray
        List of eigenvector matrices (shape=(3,3)) for each minimum
    """
    hxx = hessian_components['hxx']
    hxy = hessian_components['hxy']
    hxz = hessian_components['hxz']
    hyy = hessian_components['hyy']
    hyz = hessian_components['hyz']
    hzz = hessian_components['hzz']

    eigenvalues = []
    eigenvectors = []

    for coord in minima_coords:
        i, j, k = int(coord[0]), int(coord[1]), int(coord[2])

        H = np.array([
            [hxx[i, j, k], hxy[i, j, k], hxz[i, j, k]],
            [hxy[i, j, k], hyy[i, j, k], hyz[i, j, k]],
            [hxz[i, j, k], hyz[i, j, k], hzz[i, j, k]]
        ])

        try:
            evals, evecs = np.linalg.eigh(H)
            eigenvalues.append(evals)
            eigenvectors.append(evecs)
        except np.linalg.LinAlgError:
            eigenvalues.append(np.array([np.nan, np.nan, np.nan]))
            eigenvectors.append(np.eye(3) * np.nan)

    return eigenvalues, eigenvectors


def filter_minima_by_eigenvalues(minima_coords, eigenvalues, positive_only=True):
    """
    Filter minima based on eigenvalue signs.

    Parameters
    ----------
    minima_coords : ndarray
        Coordinates of minima (N, 3)
    eigenvalues : list of ndarray
        Eigenvalues for each minimum
    positive_only : bool
        If True, keep only minima with all positive eigenvalues

    Returns
    -------
    filtered_coords : ndarray
        Filtered coordinates
    filtered_eigenvalues : list
        Corresponding eigenvalues
    """
    valid_indices = []

    for i, evals in enumerate(eigenvalues):
        if np.any(np.isnan(evals)):
            continue

        if positive_only:
            if np.all(evals > 0):
                valid_indices.append(i)
        else:
            valid_indices.append(i)

    if len(valid_indices) == 0:
        return np.array([]).reshape(0, 3), []

    filtered_coords = minima_coords[valid_indices]
    filtered_eigenvalues = [eigenvalues[i] for i in valid_indices]

    return filtered_coords, filtered_eigenvalues


def calculate_shape_parameters(eigenvalues):
    """
    Calculate void shape parameters from eigenvalues.
    FIXED VERSION: Corrected eigenvalue-axis relationship and BBKS formulas.

    Parameters
    ----------
    eigenvalues : list of ndarray
        List of eigenvalue arrays for each void

    Returns
    -------
    dict
        'axis_ratios': array of shape (N, 2) with [b/a, c/a] ratios
        'bbks_params': array of shape (N, 2) with [ellipticity e, prolateness p]
    """
    n_voids = len(eigenvalues)

    if n_voids == 0:
        return {
            'axis_ratios': np.array([]).reshape(0, 2),
            'bbks_params': np.array([]).reshape(0, 2)
        }

    axis_ratios = np.zeros((n_voids, 2))
    bbks_params = np.zeros((n_voids, 2))

    for i, evals in enumerate(eigenvalues):
        # Sort eigenvalues: lambda1 <= lambda2 <= lambda3
        evals_sorted = np.sort(evals)
        lambda1, lambda2, lambda3 = evals_sorted

        # Skip if any eigenvalue is too small
        if lambda1 <= 1e-12 or lambda2 <= 1e-12 or lambda3 <= 1e-12:
            axis_ratios[i] = [np.nan, np.nan]
            bbks_params[i] = [np.nan, np.nan]
            continue

        # Calculate axis lengths 
        # For voids, larger eigenvalues = more constrained directions (smaller axes)
        # a >= b >= c by convention
        a = 1.0 / np.sqrt(lambda1)  # Major axis (from smallest eigenvalue)
        b = 1.0 / np.sqrt(lambda2)  # Intermediate axis
        c = 1.0 / np.sqrt(lambda3)  # Minor axis (from largest eigenvalue)
        
        # Calculate axis ratios
        b_a = b / a  # Intermediate-to-major axis ratio
        c_a = c / a  # Minor-to-major axis ratio
        
        # Validate axis ratios (should be in [0,1] and c/a <= b/a <= 1)
        if not (0 <= c_a <= b_a <= 1.0):
            axis_ratios[i] = [np.nan, np.nan]
            bbks_params[i] = [np.nan, np.nan]
            continue
        
        # Store axis ratios in correct order: [b/a, c/a]
        axis_ratios[i] = [b_a, c_a]
        
        # Calculate BBKS shape parameters using the correct formulas
        trace = lambda1 + lambda2 + lambda3
        
        if trace <= 0:
            bbks_params[i] = [np.nan, np.nan]
            continue
        
        # e parameter (ellipticity) - difference between largest and smallest eigenvalues
        e = (lambda3 - lambda1) / (2 * trace)
        
        # p parameter (prolateness)
        p = (lambda1 - 2 * lambda2 + lambda3) / (2 * trace)
        
        # Store BBKS parameters
        bbks_params[i] = [e, p]

    return {
        'axis_ratios': axis_ratios,
        'bbks_params': bbks_params
    }


# ============================================================================
# Critical Point Detection (Modular Functions)
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


def find_gradient_critical_points(grad_magnitude, gradient_threshold=0.1, min_distance=3, mode='wrap'):
    """
    Find points with small gradient magnitude (potential critical points).

    This is a modular function that can be combined with classify_critical_points_morse().

    Parameters
    ----------
    grad_magnitude : ndarray
        Gradient magnitude field
    gradient_threshold : float
        Maximum normalized gradient magnitude for critical points (0-1)
    min_distance : int
        Minimum distance between critical points
    mode : str
        Boundary mode for filter

    Returns
    -------
    critical_mask : ndarray
        Boolean mask of potential critical points
    """
    # Normalize gradient
    grad_max = np.max(grad_magnitude)
    if grad_max > 0:
        grad_norm = grad_magnitude / grad_max
    else:
        grad_norm = grad_magnitude

    # Find points with small gradient
    critical_mask = grad_norm < gradient_threshold

    # Apply minimum distance filter
    if min_distance > 0:
        inverted_grad = grad_max - grad_magnitude
        local_maxima = maximum_filter(inverted_grad, size=min_distance, mode=mode) == inverted_grad
        critical_mask = critical_mask & local_maxima

    return critical_mask


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
    '000': 20.05,
    '017': 5.00,
    '050': 1.00,
    '099': 0.00
}

# Common slice plane configurations
SLICE_PLANES = {
    0: {'name': 'yz_plane', 'axis_labels': ('Y', 'Z')},
    1: {'name': 'xz_plane', 'axis_labels': ('X', 'Z')},
    2: {'name': 'xy_plane', 'axis_labels': ('X', 'Y')}
}
