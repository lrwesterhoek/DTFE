"""Generic grid-field utilities (smoothing, slices, minima, shape parameters, cosmology, and the
mean-field velocity derivatives: mean_field_derivatives, the V-web of the mean flow for PS-DTFE grids; and
the same on one plane in a single pass over the grid, mean_field_plane, for the launcher's Explore).

Plain numpy/scipy only -- no config or I/O; callers pass everything in. Import as
`from dtfelib import fields as dtfe` (the historical alias used by the plot scripts).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import (gaussian_filter, gaussian_filter1d, maximum_filter, maximum_filter1d, minimum_filter,
                           minimum_filter1d)
from pathlib import Path

# The separable filters below run slab-threaded: scipy's ndimage filters are single-threaded, and a
# 512^3 smoothing at sigma 10 takes ~11 s on one core (the 'environment' step: ~100 of its 140 s per box). Each 1-D
# pass is split into slabs along ANOTHER axis, so every filtered line lies whole inside one slab and is the same
# float computation as scipy's own pass; the passes go in scipy's order (axis 0, 1, 2) with the same intermediate
# dtype, so the result is bit-identical to gaussian_filter / minimum_filter (tests/py_dtfelib_test.py). Fields
# below _THREAD_MIN elements keep the plain call (threads would cost more than they save).
_THREAD_MIN = 1 << 21
_THREADS = max(1, min(8, os.cpu_count() or 1))


def _separable_threaded(fn1d, field, args, kwargs):
    """scipy's separable filter (fn1d along axis 0, then 1, ...; the output's dtype = the input's, as scipy keeps
    it), each 1-D pass threaded over slabs along another axis."""
    field = np.asarray(field)
    out = np.empty(field.shape, dtype=field.dtype)
    src = field
    with ThreadPoolExecutor(_THREADS) as ex:
        for ax in range(field.ndim):
            split = 1 if ax == 0 else 0                   # slabs along an axis this pass does not filter
            n = field.shape[split]
            edges = np.linspace(0, n, min(_THREADS, n) + 1).astype(int)

            def one(lo_hi, ax=ax, split=split, src=src):
                sl = [slice(None)] * field.ndim
                sl[split] = slice(*lo_hi)
                sl = tuple(sl)
                fn1d(src[sl], *args, axis=ax, output=out[sl], **kwargs)

            list(ex.map(one, [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]))
            src = out                                      # in place from the second pass on, as scipy does
    return out


def pseudo_phase_space_density(density, dispersion, eps=1e-30):
    """Q = rho / sigma^3, the standard scalar proxy for the phase-space density.

    The true distribution function of a collisionless sheet is singular -- f(x,v) is a sum of
    delta functions, one per stream -- so nothing can tabulate it on a grid. Two things are
    computable instead, and they are not the same:

      * this Q (Taylor & Navarro 2001), formed from the EXACT multi-stream density and velocity
        dispersion that PS-DTFE deposits, so it inherits the real multi-stream structure;
      * the standard binary's '--approxPSD', f ~ rho(x) * g(v), which builds a second Delaunay
        tessellation in VELOCITY space and multiplies the two densities. That assumes the local
        distribution factorises into a spatial and a velocity part, which is exactly what fails
        in a multi-stream region.

    Comparing the two is therefore a direct measurement of how badly the factorisation assumption
    breaks down where streams cross; see the recipe in demo/COMMANDS.md.

    Parameters
    ----------
    density : array
        Multi-stream density, in whatever units the caller wants Q expressed in
        (FieldSet.density(units="mean") for rho/rho_bar).
    dispersion : array
        Velocity dispersion sigma (the '.velDisp' TRACE field is sigma^2; pass its square root,
        or use FieldSet-derived sigma). Cells with sigma below `eps` return NaN rather than a
        divergent Q -- cold single-stream regions have sigma == 0 by construction.
    """
    import numpy as np
    rho = np.asarray(density, dtype=np.float64)
    sig = np.asarray(dispersion, dtype=np.float64)
    out = np.full(np.broadcast(rho, sig).shape, np.nan, dtype=np.float64)
    ok = sig > eps
    np.divide(rho, sig ** 3, out=out, where=ok)
    return out


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
    if np.ndim(sigma) == 0 and np.ndim(field) >= 2 and np.size(field) >= _THREAD_MIN and sigma > 1e-15 \
            and np.asarray(field).dtype.kind == 'f':
        return _separable_threaded(gaussian_filter1d, field, (float(sigma),), {"mode": mode})
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


def box_filter(field, size, mode='wrap', maximum=False):
    """minimum_filter (maximum_filter when maximum) of field over a box of the given size, slab-threaded on a big
    field: a scalar size is a box footprint, which scipy runs as separable 1-D passes, so threaded is identical
    (find_local_minima, and the critical points and potential maxima of pipeline.py)."""
    if np.ndim(size) == 0 and int(size) > 1 and np.ndim(field) >= 2 and np.size(field) >= _THREAD_MIN:
        return _separable_threaded(maximum_filter1d if maximum else minimum_filter1d, field, (int(size),),
                                   {"mode": mode})
    return (maximum_filter if maximum else minimum_filter)(field, size=size, mode=mode)


def find_local_minima(field, footprint_size=3, mode='wrap'):
    local_min = box_filter(field, footprint_size, mode=mode)
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


# ---------------------------------------------------------------- mean-field velocity derivatives
# THE SAME QUANTITY FOR BOTH ESTIMATORS. Standard DTFE stores the derivatives of its one single-valued velocity
# field: 'a_velGrad' is the cell average of the gradient of the linear interpolant, the gradient of the MEAN
# velocity grad<v>. PS-DTFE stores, in a multi-stream cell, the stream average of each stream's own gradient
# <grad v> (dtfelib/io.py, the note above FIELDS): its divergence has a box mean of +40 km/s/Mpc at TNG100-3-Dark
# z=1 (standard DTFE: 0, by Gauss) and its V-web finds 1.8% filaments against standard DTFE's 6.6%.
# The functions below give a PS output the standard quantity -- the derivatives of the velocity GRID
# itself ('a_vel', the stream-mean velocity: volume-weighted under --ps-volume-weighted, the production convention,
# else the mass-weighted mean) -- by spectral (FFT) differentiation on the periodic box, in the binary's conventions:
#   gradient          9 components, c = a*3 + b holds dv_a/dx_b (field_computation.h velocityGradient writes
#                     [j*3+i] = dv_j/dx_i; gui/explore.py reads it so); axis 0 of the grid is x, as the T-web's FFT
#   divergence        dv_x/dx + dv_y/dy + dv_z/dz
#   shear             the traceless symmetric part (g_ab + g_ba)/2 - delta_ab div/3, 5 components in the order
#                     xx, xy, xz, yy, yz (web_fields.cc velocityShear; zz = -(xx + yy))
#   vorticity         (g_ab - g_ba)/2 for ab = xy, xz, yz (web_fields.cc velocityVorticity)
#   vweb_eigenvalues  the eigenvalues, DESCENDING, of Sigma_ab = -sqrt(a)/(2 H0) (g_ab + g_ba), H0 = 100 h km/s/Mpc
#                     and g in Gadget u-units per Mpc (web_fields.cc computeWebClassification: the Hoffman et al. 2012
#                     V-web; sqrt(a) takes u-units to peculiar km/s), by the binary's Cardano formula in double
#   vweb              the number of those eigenvalues above lambda_th: 0 void, 1 wall, 2 filament, 3 node
# Units: those of the velocity grid per Mpc of a box_mpc box -- pass the RAW u-unit grid (FieldSet.load(...,
# scaled=False)) for the binary's on-disk numbers; FieldSet.mean_field() then applies load()'s sqrt(a). No smoothing:
# the binary's V-web classifies the raw cell gradient too (lambda_th = 0.3 in the run scripts).
# WHY SPECTRAL: for a band-limited field the FFT derivative of the cell AVERAGES is exactly the cell average of the
# derivative (the box filter commutes with d/dx), and the cell average is what an '_a' grid holds. The odd derivative
# of the Nyquist mode is set to zero (its sine vanishes at every grid point). Measured on a synthetic Zel'dovich field
# (32^3 particles, standard binary with --exact-average, grids 32/64/128^3): applied to the binary's own 'a_vel' it
# reproduces its 'a_velDiv' with correlation 0.998/0.995/0.998 (rms difference 7/10/6% of the rms) and its V-web
# label in 98.5/96.6/97.8% of the cells; central second-order differences do worse (0.992/0.995/0.997, 95.8/96.6/
# 97.4%). The rest is the interpolant's power above the grid's Nyquist frequency, which no derivative of the cell
# averages can recover. tests/py_meanfield_test.py repeats that comparison against the binary.
# MEMORY: float32 throughout for a float32 grid (float64 for a DOUBLE=1 grid); a velocity component's spectrum is
# made when first needed and dropped after its last use (the divergence holds one at a time, the shear three), the
# V-web keeps its six tensor components and reduces them to eigenvalues in slabs. mean_field_peak_bytes() estimates
# the peak: at 512^3 float32 the divergence ~3.2 GB, the V-web (labels and eigenvalues together) ~6.5 GB, the
# 9-component gradient ~7.5 GB (the 9 float64 components a naive version would hold: 9.7 GB, before any FFT).
MEAN_FIELD_NAMES = ("gradient", "divergence", "shear", "vorticity", "vweb", "vweb_eigenvalues")
_MF_NCOMP = {"gradient": 9, "divergence": 1, "shear": 5, "vorticity": 3, "vweb": 1, "vweb_eigenvalues": 3}
_VWEB_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))     # symmetricEigenvalues3x3's argument order
_EIG_SLAB_CELLS = 1 << 20     # cells per eigenvalue slab: Cardano's float64 temporaries then hold ~0.4-0.8 GB (measured)


def _velgrad_terms(name: str) -> list:
    """The output components of 'name' as linear combinations of g_ab = dv_a/dx_b: one list of (a, b, coef) per
    component, in the binary's component order. 'vweb_tensor' = g_ab + g_ba over _VWEB_PAIRS (the V-web's
    -sqrt(a)/(2 H0) is applied with the eigenvalues)."""
    if name == "gradient":
        return [[(a, b, 1.0)] for a in range(3) for b in range(3)]
    if name == "divergence":
        return [[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]]
    if name == "shear":
        out = []
        for i in range(2):
            for j in range(i, 3):
                if i == j:          # g_ii - div/3
                    out.append([(d, d, 2.0 / 3.0 if d == i else -1.0 / 3.0) for d in range(3)])
                else:
                    out.append([(i, j, 0.5), (j, i, 0.5)])
        return out
    if name == "vorticity":
        return [[(i, j, 0.5), (j, i, -0.5)] for i in range(3) for j in range(i + 1, 3)]
    if name == "vweb_tensor":
        return [[(a, a, 2.0)] if a == b else [(a, b, 1.0), (b, a, 1.0)] for a, b in _VWEB_PAIRS]
    raise KeyError(f"no mean-field derivative {name!r}; known: {', '.join(MEAN_FIELD_NAMES)}")


def _wavenumbers(n: int, box_mpc: float, half: bool) -> np.ndarray:
    """Angular wavenumbers [rad/Mpc] of one axis in the FFT's order (numpy fftfreq / rfftfreq), the Nyquist mode
    zeroed: the first derivative of the real cosine at Nyquist is a sine that vanishes at every grid point."""
    m = np.fft.rfftfreq(n, 1.0 / n) if half else np.fft.fftfreq(n, 1.0 / n)
    k = (2.0 * np.pi / float(box_mpc)) * m
    if n % 2 == 0:
        k[np.abs(m) == n // 2] = 0.0
    return k


def symmetric_eigenvalues3(a00, a01, a02, a11, a12, a22) -> np.ndarray:
    """Eigenvalues of the symmetric 3x3 matrices [[a00 a01 a02] [a01 a11 a12] [a02 a12 a22]], DESCENDING along a new
    last axis, in float64: the binary's symmetricEigenvalues3x3 (web_fields.cc, Cardano's formula in double, the
    acos argument clamped to [-1, 1]; a matrix within 1e-15 of a multiple of the identity returns its diagonal),
    vectorized. Inputs broadcast; any float dtype."""
    a00, a01, a02, a11, a12, a22 = (np.asarray(x, dtype=np.float64) for x in (a00, a01, a02, a11, a12, a22))
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    q = (a00 + a11 + a22) / 3.0
    b00, b11, b22 = a00 - q, a11 - q, a22 - q
    p = np.sqrt((b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1) / 6.0)
    iso = p < 1.0e-15
    inv_p = 1.0 / np.where(iso, 1.0, p)
    B00, B01, B02, B11, B12, B22 = b00 * inv_p, a01 * inv_p, a02 * inv_p, b11 * inv_p, a12 * inv_p, b22 * inv_p
    det = B00 * (B11 * B22 - B12 * B12) - B01 * (B01 * B22 - B12 * B02) + B02 * (B01 * B12 - B11 * B02)
    phi = np.arccos(np.clip(det / 2.0, -1.0, 1.0)) / 3.0
    e0 = q + 2.0 * p * np.cos(phi)
    e2 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    e1 = 3.0 * q - e0 - e2                                  # the trace identity, as the binary
    ev = np.stack([np.where(iso, a00, e0), np.where(iso, a11, e1), np.where(iso, a22, e2)], axis=-1)
    return -np.sort(-ev, axis=-1)                           # descending (the binary's three swaps)


def _vweb_norm(scale_factor, hubble_h) -> float:
    """The V-web's -sqrt(a)/(2 H0), H0 = 100 h, with the binary's fallbacks (a or h None or <= 0: 1)."""
    sqrt_a = float(np.sqrt(scale_factor)) if scale_factor is not None and scale_factor > 0 else 1.0
    h0 = 100.0 * float(hubble_h) if hubble_h is not None and hubble_h > 0 else 1.0
    return -sqrt_a / (2.0 * h0)


def classify_web(eigenvalues, lambda_th: float) -> np.ndarray:
    """The web label of web_fields.cc classifyWeb: how many eigenvalues (last axis) exceed lambda_th -- 0 void, 1 wall,
    2 filament, 3 node -- as float32, the binary's on-disk type of the '.velVweb' / '.tweb' grids."""
    return (np.asarray(eigenvalues) > float(lambda_th)).sum(axis=-1).astype(np.float32)


def _mf_jobs(names) -> list:
    """The inverse FFTs of mean_field_derivatives(names), in order: (result name, component index or None for a
    one-component result, terms). The V-web's names share one 'vweb_tensor' job set, last."""
    jobs, seen = [], set()
    for nm in names:
        if nm in ("vweb", "vweb_eigenvalues") or nm in seen:
            continue
        seen.add(nm)
        terms = _velgrad_terms(nm)
        jobs += [(nm, None if len(terms) == 1 else c, t) for c, t in enumerate(terms)]
    if any(nm in ("vweb", "vweb_eigenvalues") for nm in names):
        jobs += [("vweb_tensor", c, t) for c, t in enumerate(_velgrad_terms("vweb_tensor"))]
    return jobs


def _mf_max_spectra(jobs) -> int:
    """The most velocity spectra alive at once under mean_field_derivatives' make-late, drop-after-last-use rule."""
    uses = [sum(1 for _, _, t in jobs for a, _, _ in t if a == comp) for comp in range(3)]
    alive, peak = set(), 0
    for _, _, terms in jobs:
        for a, _, _ in terms:
            alive.add(a)
            peak = max(peak, len(alive))
            uses[a] -= 1
            if uses[a] == 0:
                alive.discard(a)
    return peak


def mean_field_peak_bytes(n: int, names, itemsize: int = 4) -> int:
    """The peak memory of mean_field_derivatives(names) on an n^3 grid of itemsize-byte floats, an estimate: the
    larger of its two stages -- the FFTs (the velocity spectra alive at once, two complex work buffers, one copied-out
    velocity component and the inverse FFT's output, the results, the V-web's six tensor components) and the V-web's
    eigenvalue pass (the tensor, the results, the float64 temporaries of one slab). Checked against the process's
    peak footprint at 256^3 and 320^3 (within 15%); at 512^3 the FFT stage dominates."""
    names = (names,) if isinstance(names, str) else tuple(names)
    cells = int(n) ** 3
    spec = int(n) * int(n) * (int(n) // 2 + 1) * 2 * itemsize
    spectra = _mf_max_spectra(_mf_jobs(names))
    plain = sum(_MF_NCOMP[nm] for nm in names if nm not in ("vweb", "vweb_eigenvalues")) * cells * itemsize
    web = any(nm in ("vweb", "vweb_eigenvalues") for nm in names)
    tensor = 6 * cells * itemsize if web else 0
    fft_stage = spectra * spec + 2 * spec + 2 * cells * itemsize + plain + tensor
    if not web:
        return fft_stage
    web_out = (3 * cells * itemsize if "vweb_eigenvalues" in names else 0) + (cells * 4 if "vweb" in names else 0)
    # the slab's temporaries: ~48 float64 arrays of numpy's own, about as much again in the allocator's caches
    # (RSS measured at 256^3 and 320^3: 0.4 GB above numpy's traced peak, whatever the grid)
    return max(fft_stage, plain + tensor + web_out + 96 * min(cells, _EIG_SLAB_CELLS) * 8)


def mean_field_derivatives(velocity, box_mpc: float, names=("divergence",), *, scale_factor: float | None = 1.0,
                           hubble_h: float | None = None, lambda_th: float = 0.3, workers: int | None = None) -> dict:
    """{name: grid} of the MEAN-FIELD velocity derivatives (the section comment above): the spectral gradient of the
    periodic (N,N,N,3) 'velocity' grid of a box_mpc box (h-free Mpc), in the binary's component order and in the
    velocity's units per Mpc. names: any of MEAN_FIELD_NAMES; 'vweb'/'vweb_eigenvalues' need scale_factor (a: the
    sqrt(a) of u-units; None or <= 0 = 1, the binary's fallback) and hubble_h (H0 = 100 h; None or <= 0 = 1, the
    binary's fallback) and use lambda_th. 'velocity' may be a read-only memmap: the components are copied out one at
    a time. Shapes: (N,N,N) for divergence/vweb, (N,N,N,c) otherwise; float32 for a float32 grid (float64 for float64)."""
    from scipy import fft as sfft

    names = (names,) if isinstance(names, str) else tuple(names)
    for nm in names:
        if nm not in MEAN_FIELD_NAMES:
            raise KeyError(f"no mean-field derivative {nm!r}; known: {', '.join(MEAN_FIELD_NAMES)}")
    shape = tuple(np.shape(velocity))
    if len(shape) != 4 or shape[3] != 3 or not (shape[0] == shape[1] == shape[2]):
        raise ValueError(f"mean_field_derivatives needs an (N,N,N,3) velocity grid, got {shape}")
    n = shape[0]
    real = np.float64 if np.dtype(getattr(velocity, "dtype", np.float32)) == np.float64 else np.float32
    cplx = np.complex128 if real == np.float64 else np.complex64
    w = _THREADS if workers is None else int(workers)
    nh = n // 2 + 1
    ik = [(1j * _wavenumbers(n, box_mpc, False)).astype(cplx).reshape(n, 1, 1),
          (1j * _wavenumbers(n, box_mpc, False)).astype(cplx).reshape(1, n, 1),
          (1j * _wavenumbers(n, box_mpc, True)).astype(cplx).reshape(1, 1, nh)]

    # the jobs (_mf_jobs); the V-web's tensor goes into a (6,N,N,N) work array
    jobs = _mf_jobs(names)
    want_web = any(nm == "vweb_tensor" for nm, _, _ in jobs)
    out = {}
    for nm, c, _ in jobs:
        if nm != "vweb_tensor" and nm not in out:
            k = _MF_NCOMP[nm]
            out[nm] = np.empty((n, n, n) if k == 1 else (n, n, n, k), dtype=real)
    tensor = np.empty((6, n, n, n), dtype=real) if want_web else None

    # each velocity component's spectrum is made when first needed and dropped after its last use (_mf_max_spectra:
    # 'divergence' alone holds one at a time, the V-web two, the shear three)
    uses = [sum(1 for _, _, t in jobs for a, _, _ in t if a == comp) for comp in range(3)]
    spec = [None, None, None]
    acc = np.empty((n, n, nh), dtype=cplx)
    tmp = np.empty((n, n, nh), dtype=cplx)
    for nm, c, terms in jobs:
        acc.fill(0)
        for a, b, coef in terms:
            if spec[a] is None:
                spec[a] = sfft.rfftn(np.ascontiguousarray(velocity[..., a], dtype=real), workers=w)
            np.multiply(spec[a], ik[b] * cplx(coef), out=tmp)
            acc += tmp
            uses[a] -= 1
            if uses[a] == 0:
                spec[a] = None
        res = sfft.irfftn(acc, s=(n, n, n), workers=w, overwrite_x=True).astype(real, copy=False)   # acc is refilled
        if nm == "vweb_tensor":
            tensor[c] = res
        elif c is None:
            out[nm][...] = res
        else:
            out[nm][..., c] = res
        del res
    del acc, tmp, spec

    if want_web:
        norm = _vweb_norm(scale_factor, hubble_h)          # Sigma = norm (g + g^T): tensor holds g + g^T already
        eig = np.empty((n, n, n, 3), dtype=real) if "vweb_eigenvalues" in names else None
        lab = np.empty((n, n, n), dtype=np.float32) if "vweb" in names else None
        step = max(1, _EIG_SLAB_CELLS // (n * n))
        for s in range(0, n, step):
            e = min(n, s + step)
            ev = symmetric_eigenvalues3(*(norm * tensor[i, s:e].astype(np.float64) for i in range(6)))
            if eig is not None:
                eig[s:e] = ev
            if lab is not None:
                lab[s:e] = classify_web(ev.astype(real), lambda_th)
        del tensor
        if eig is not None:
            out["vweb_eigenvalues"] = eig
        if lab is not None:
            out["vweb"] = lab
    return out


# ---------------------------------------------------------------- the same derivatives on ONE plane
# The launcher's Explore shows a PS output's divergence / shear / vorticity / gradient / V-web as the mean field
# (the figure scripts' default) one slice at a time, beside a running job whose memory budget it must leave alone:
# mean_field_derivatives' whole-cube FFTs peak at 3.2-7.5 GB on a 512^3 grid. The spectral derivative is separable:
# d/dx_b of the 3D transform multiplies by i k_b alone, so it is the 1-D spectral derivative along axis b of every
# line of the grid -- a periodic convolution with the fixed kernel K = irfft(i k) (the 1-D wavenumbers with the
# Nyquist mode zeroed, _wavenumbers), (D v)_i = sum_j K[(i - j) mod n] v_j. On the plane 'index' normal to 'axis'
# the in-plane derivatives are then 1-D FFTs of the plane itself, and the normal one the kernel's weighted sum over
# the line through each pixel: one pass over the grid in slabs along axis 0, holding one slab (16 MB by default) and
# a few planes -- about 100 MB in all at 512^3 (the slab, its float64 copy in the contraction; 64 MB slabs: 350 MB,
# no faster). The same linear operator as the cube's, so the same numbers to float rounding (measured 2e-7 of the
# largest value on random, not band-limited, float32 grids of even and odd n, every axis; tests/
# py_meanfield_test.py repeats it for the six names). Cost: one read of the velocity grid per plane (0.3-0.6 s at
# 512^3 from memory, 0.6-1.0 s from a file in the page cache; measured on a loaded machine).
def mean_field_gradient_plane(velocity, box_mpc: float, axis: int, index: int, *, slab_bytes: int = 16 << 20):
    """g[..., a, b] = dv_a/dx_b, float64 (N, N, 3, 3), on the plane 'index' normal to 'axis' of the periodic
    (N,N,N,3) 'velocity' grid of a box_mpc box: exactly mean_field_derivatives' spectral gradient there (its
    'gradient' cut as np.take(cube, index, axis), reshaped to 3 x 3), from one pass over the grid (the section
    comment above). 'velocity': anything with .shape, .dtype and [s:e] slabs along axis 0 -- an array, a memmap,
    or a reader that does plain reads (the launcher never maps a file: a page of an unplugged drive is a SIGBUS);
    slabs of about slab_bytes (16 MB: ~100 MB in all at 512^3). Units: the velocity's per Mpc."""
    shape = tuple(int(v) for v in np.shape(velocity))
    if len(shape) != 4 or shape[3] != 3 or not (shape[0] == shape[1] == shape[2]):
        raise ValueError(f"mean_field_gradient_plane needs an (N,N,N,3) velocity grid, got {shape}")
    n, axis, index = shape[0], int(axis), int(index)
    if axis not in (0, 1, 2) or not 0 <= index < n:
        raise ValueError(f"plane {index} along axis {axis} is outside an {n}^3 grid")
    rest = [a for a in range(3) if a != axis]
    k = _wavenumbers(n, box_mpc, True)
    w = np.fft.irfft(1j * k, n)[(index - np.arange(n)) % n]         # K[(index - j) mod n], j = 0 .. n-1
    item = np.dtype(getattr(velocity, "dtype", np.float32)).itemsize
    step = max(1, int(slab_bytes) // (n * n * 3 * item))
    plane = np.empty((n, n, 3))
    normal = np.zeros((n, n, 3))
    for s in range(0, n, step):
        e = min(n, s + step)
        slab = np.asarray(velocity[s:e])
        if axis == 0:
            normal += np.tensordot(w[s:e], slab, axes=(0, 0))
            if s <= index < e:
                plane[...] = slab[index - s]
        elif axis == 1:
            normal[s:e] = np.tensordot(slab, w, axes=(1, 0))
            plane[s:e] = slab[:, index]
        else:
            normal[s:e] = np.tensordot(slab, w, axes=(2, 0))
            plane[s:e] = slab[:, :, index]
        del slab
    g = np.empty((n, n, 3, 3))
    g[..., axis] = normal
    for pa, b in enumerate(rest):                          # the plane's axis pa is the grid's axis b
        ik = (1j * k).reshape((-1, 1, 1) if pa == 0 else (1, -1, 1))
        g[..., b] = np.fft.irfft(np.fft.rfft(plane, axis=pa) * ik, n, axis=pa)
    return g


def mean_field_from_gradient(g, names=("divergence",), *, real=np.float32, scale_factor: float | None = 1.0,
                             hubble_h: float | None = None, lambda_th: float = 0.3) -> dict:
    """{name: array} of mean_field_derivatives' results from a velocity gradient g[..., a, b] = dv_a/dx_b (any
    leading shape; mean_field_gradient_plane's plane): the same combinations of the components (_velgrad_terms), each
    rounded to 'real' as the cube's inverse FFTs are (float32 for a float32 grid), the V-web's tensor likewise
    before its eigenvalues (the same normalisation, Cardano and labels). Shapes: g's leading shape, with the
    components last where there are several."""
    names = (names,) if isinstance(names, str) else tuple(names)
    for nm in names:
        if nm not in MEAN_FIELD_NAMES:
            raise KeyError(f"no mean-field derivative {nm!r}; known: {', '.join(MEAN_FIELD_NAMES)}")
    g = np.asarray(g, dtype=np.float64)
    real = np.float64 if np.dtype(real) == np.float64 else np.float32

    def combine(terms):
        return sum(coef * g[..., a, b] for a, b, coef in terms).astype(real)

    out = {}
    for nm in names:
        if nm in ("vweb", "vweb_eigenvalues") or nm in out:
            continue
        comps = [combine(t) for t in _velgrad_terms(nm)]
        out[nm] = comps[0] if len(comps) == 1 else np.stack(comps, axis=-1)
    if any(nm in ("vweb", "vweb_eigenvalues") for nm in names):
        norm = _vweb_norm(scale_factor, hubble_h)
        tensor = [combine(t) for t in _velgrad_terms("vweb_tensor")]
        ev = symmetric_eigenvalues3(*(norm * c.astype(np.float64) for c in tensor))
        if "vweb_eigenvalues" in names:
            out["vweb_eigenvalues"] = ev.astype(real)
        if "vweb" in names:
            out["vweb"] = classify_web(ev.astype(real), lambda_th)
    return {nm: out[nm] for nm in names}


def mean_field_plane(velocity, box_mpc: float, names, axis: int, index: int, *, scale_factor: float | None = 1.0,
                     hubble_h: float | None = None, lambda_th: float = 0.3, slab_bytes: int = 16 << 20) -> dict:
    """{name: plane} of mean_field_derivatives(velocity, box_mpc, names, ...) on the one plane 'index' normal to
    'axis' -- np.take(result, index, axis) to rounding, the same dtype and component order -- in one pass over the
    grid (mean_field_gradient_plane: one slab in memory, never the cube's FFTs)."""
    real = np.float64 if np.dtype(getattr(velocity, "dtype", np.float32)) == np.float64 else np.float32
    g = mean_field_gradient_plane(velocity, box_mpc, axis, index, slab_bytes=slab_bytes)
    return mean_field_from_gradient(g, names, real=real, scale_factor=scale_factor, hubble_h=hubble_h,
                                    lambda_th=lambda_th)
