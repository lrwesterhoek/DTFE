"""Generic grid-field utilities (smoothing, slices, minima, shape parameters, cosmology).

Plain numpy/scipy only -- no config or I/O; callers pass everything in. Import as
`from dtfelib import fields as dtfe` (the historical alias used by the plot scripts).
"""

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter
from pathlib import Path


def calculate_density_contrast(density_field):
    mean_density = np.mean(density_field)
    if mean_density == 0:
        return density_field
    return (density_field - mean_density) / mean_density


def extract_2d_slice(field, slice_dim=2, slice_index=None):
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


def smooth_field(field, sigma, mode='wrap'):
    if sigma <= 0:
        return field
    return gaussian_filter(field, sigma=sigma, mode=mode)


def calculate_shape_parameters(eigenvalues):
    # Vectorized over voids; keeps the original per-void semantics bit-for-bit (native
    # eigenvalue dtype throughout, float64 output rows, NaN for trace <= 1e-12 /
    # lambda1 <= 1e-12 / a failed 0 <= c/a <= b/a <= 1 sanity gate). Covered by the
    # reference-equivalence test in tests/py_dtfelib_test.py.
    n_voids = len(eigenvalues)

    if n_voids == 0:
        return {
            'axis_ratios': np.array([]).reshape(0, 2),
            'bbks_params': np.array([]).reshape(0, 2)
        }

    ev = np.sort(np.asarray(eigenvalues), axis=1)   # ascending: lambda1 <= lambda2 <= lambda3
    l1, l2, l3 = ev[:, 0], ev[:, 1], ev[:, 2]
    trace = l1 + l2 + l3

    bbks_params = np.full((n_voids, 2), np.nan)
    ok = trace > 1e-12
    bbks_params[ok, 0] = (l3[ok] - l1[ok]) / (2 * trace[ok])
    bbks_params[ok, 1] = (l1[ok] - 2 * l2[ok] + l3[ok]) / (2 * trace[ok])

    axis_ratios = np.full((n_voids, 2), np.nan)
    pos = np.where(l1 > 1e-12)[0]
    a = 1.0 / np.sqrt(l1[pos])
    b = 1.0 / np.sqrt(l2[pos])
    c = 1.0 / np.sqrt(l3[pos])
    b_a = b / a
    c_a = c / a
    good = (c_a >= 0) & (c_a <= b_a) & (b_a <= 1.0)
    axis_ratios[pos[good], 0] = b_a[good]
    axis_ratios[pos[good], 1] = c_a[good]

    return {
        'axis_ratios': axis_ratios,
        'bbks_params': bbks_params
    }


def find_local_minima(field, footprint_size=3, mode='wrap'):
    local_min = minimum_filter(field, size=footprint_size, mode=mode)
    minima_mask = (field == local_min)
    minima_coords = np.array(np.where(minima_mask)).T
    return minima_mask, minima_coords


def get_cosmology_params(z, H0=67.74, Omega_m0=0.3089, Omega_Lambda0=0.6911):
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


def ensure_output_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
