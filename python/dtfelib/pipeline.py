
import contextlib
import json
import os
import zipfile
import zlib
import numpy as np
from pathlib import Path

import config
from .io import FieldSet
from .cli import require_sim, sim_dir
from . import fields as dtfe

try:
    from scipy import fft as _sfft
    def _rfftn(a):
        return _sfft.rfftn(a, workers=-1)
    def _irfftn(a, shape):
        return _sfft.irfftn(a, s=shape, workers=-1)
except ImportError:
    def _rfftn(a):
        return np.fft.rfftn(a)
    def _irfftn(a, shape):
        return np.fft.irfftn(a, s=shape)


def _k_vectors(shape):
    nx, ny, nz = shape
    kx = (2 * np.pi * np.fft.fftfreq(nx)).astype(np.float32)[:, None, None]
    ky = (2 * np.pi * np.fft.fftfreq(ny)).astype(np.float32)[None, :, None]
    kz = (2 * np.pi * np.fft.rfftfreq(nz)).astype(np.float32)[None, None, :]
    return kx, ky, kz


def hessian_from_field(field):
    shape = field.shape
    fk = _rfftn(field.astype(np.float32, copy=False))
    kx, ky, kz = _k_vectors(shape)
    out = {}
    for name, ka, kb in (('hxx', kx, kx), ('hxy', kx, ky), ('hxz', kx, kz),
                         ('hyy', ky, ky), ('hyz', ky, kz), ('hzz', kz, kz)):
        out[name] = _irfftn(-(ka * kb) * fk, shape).astype(np.float32)
    return out


def gradient_magnitude_from_field(field):
    shape = field.shape
    fk = _rfftn(field.astype(np.float32, copy=False))
    kx, ky, kz = _k_vectors(shape)
    g2 = None
    for ka in (kx, ky, kz):
        g = _irfftn(1j * ka * fk, shape)
        g2 = g * g if g2 is None else g2 + g * g
    return np.sqrt(g2).astype(np.float32)


def potential_from_delta(delta, box_size, rho_bar, G=1.0):
    shape = delta.shape
    n = shape[0]
    dk = _rfftn(delta.astype(np.float32, copy=False))
    d = box_size / n
    kx = (2 * np.pi * np.fft.fftfreq(shape[0], d=d)).astype(np.float32)[:, None, None]
    ky = (2 * np.pi * np.fft.fftfreq(shape[1], d=d)).astype(np.float32)[None, :, None]
    kz = (2 * np.pi * np.fft.rfftfreq(shape[2], d=d)).astype(np.float32)[None, None, :]
    k2 = kx * kx + ky * ky + kz * kz
    k2[0, 0, 0] = np.inf
    phi = _irfftn(-4 * np.pi * G * rho_bar * dk / k2, shape)
    return phi.astype(np.float32)


def eigh_at_coords(hessian, coords):
    coords = np.asarray(coords, dtype=int)
    if coords.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3, 3))
    i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]
    H = np.empty((len(coords), 3, 3), dtype=np.float64)
    H[:, 0, 0] = hessian['hxx'][i, j, k]
    H[:, 0, 1] = H[:, 1, 0] = hessian['hxy'][i, j, k]
    H[:, 0, 2] = H[:, 2, 0] = hessian['hxz'][i, j, k]
    H[:, 1, 1] = hessian['hyy'][i, j, k]
    H[:, 1, 2] = H[:, 2, 1] = hessian['hyz'][i, j, k]
    H[:, 2, 2] = hessian['hzz'][i, j, k]
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs


def hessian_slice(hessian, slice_dim, index=None):
    comp = {}
    for name, arr in hessian.items():
        comp[name] = dtfe.extract_2d_slice(arr, slice_dim, index)
    sh = comp['hxx'].shape
    H = np.empty(sh + (3, 3), dtype=np.float64)
    H[..., 0, 0] = comp['hxx']
    H[..., 0, 1] = H[..., 1, 0] = comp['hxy']
    H[..., 0, 2] = H[..., 2, 0] = comp['hxz']
    H[..., 1, 1] = comp['hyy']
    H[..., 1, 2] = H[..., 2, 1] = comp['hyz']
    H[..., 2, 2] = comp['hzz']
    evals = np.linalg.eigvalsh(H.reshape(-1, 3, 3)).reshape(sh + (3,))
    return {
        'lambda1': evals[..., 0].astype(np.float32),
        'lambda2': evals[..., 1].astype(np.float32),
        'lambda3': evals[..., 2].astype(np.float32),
        'trace': evals.sum(axis=-1).astype(np.float32),
        'det': evals.prod(axis=-1).astype(np.float32),
    }


def find_critical_points(delta_s, hessian, grad_mag):
    # the three box filters run slab-threaded (dtfe.box_filter, bit-identical to scipy's serial minimum_filter /
    # maximum_filter): at 512^3 serial filters are ~3 passes of several seconds each
    fp = config.FOOTPRINT_SIZE
    minima_mask = (delta_s == dtfe.box_filter(delta_s, fp, mode='wrap'))
    maxima_mask = (delta_s == dtfe.box_filter(delta_s, fp, mode='wrap', maximum=True))

    gmax = float(np.max(grad_mag))
    gnorm = grad_mag / gmax if gmax > 0 else grad_mag
    cand = gnorm < config.GRADIENT_THRESHOLD
    inv = gmax - grad_mag
    cand &= (inv == dtfe.box_filter(inv, fp, mode='wrap', maximum=True))
    cand &= ~minima_mask
    cand &= ~maxima_mask

    out = {}
    for name, mask in (('minima', minima_mask), ('maxima', maxima_mask)):
        coords = np.argwhere(mask)
        evals, _ = eigh_at_coords(hessian, coords)
        out[name] = {'coords': coords, 'values': delta_s[mask], 'eigenvalues': evals}

    coords = np.argwhere(cand)
    evals, _ = eigh_at_coords(hessian, coords)
    nneg = (evals < 0).sum(axis=1)
    vals = delta_s[cand]
    for name, n in (('saddle1', 1), ('saddle2', 2)):
        sel = nneg == n
        out[name] = {'coords': coords[sel], 'values': vals[sel], 'eigenvalues': evals[sel]}

    out['field_mean'] = float(np.mean(delta_s))
    out['field_std'] = float(np.std(delta_s))
    return out


R_V_STEP_SIGMA = 0.2      # radial step of the R_v search, in smoothing lengths (the crossing is interpolated)
R_V_MAX_SIGMA = 24.0      # how far out it looks (capped at half the box)
# Why 24 (15 = 32.4 Mpc on TNG100): at 15 sigma 17-25% of the underdense minima of TNG100-3-Dark and
# TNG50-3-Dark (z = 0, 1, 3, 5; 10-cell smoothing) have NO crossing within reach (NaN, 88-104 per catalogue) or cross
# in its last sigma, 8-17% of the well-resolved ones -- the R_v tail is cut, not measured. Same count at 20 / 24 / 25
# sigma / L/2: 5-8% / 1.1-2.6% / 0.6-1.8% / 0.4-1.2% (well-resolved 4-8% / 0-2.9% / 0-2% / 0-2%). Half the 512^3 box is
# 25.6 sigma at 10 cells; 24 is the largest whole cap a sigma short of it (shells beyond L/2 meet the centre's periodic
# images; within the last sigma the smoothed field already feels them). The distinct-void median crossing R_v
# (Mpc, 15 -> 24 sigma) at z = 0 / 1 / 3 / 5: TNG100 6.94 -> 7.42 / 7.59 -> 7.85 / 6.82 -> 7.64 / 6.27 -> 6.79,
# TNG50 2.82 -> 2.88 / 3.41 -> 3.98 / 3.68 -> 3.97 / 3.13 -> 3.73, and 6-13% more distinct voids (TNG100 z = 0: 77 ->
# 83) as fewer deep voids lack a radius. It is in r_v_settings, so a catalogue measured otherwise is re-measured once
# (~15 s and 2.4 GB per 512^3 snapshot: reading and smoothing the grid; the shells themselves 1-2 s). The half-box
# clause has a sigma's margin of its own (r_v_reach_cells: min(cap sigma, L/2 - sigma); min(cap sigma, L/2) would let
# a large-sigma search run right up to the centre's periodic images: TNG50 at 2.1625 Mpc = 21.4 cells would reach
# exactly 256 cells, not 234.6). The margin changes nothing while cap sigma <= L/2 - sigma, i.e. sigma <=
# 10.24 cells at 512^3 (the production 10 included), so the catalogue records the reach it searched (r_v_settings'
# sixth element) instead of a new algorithm number: a cache without one is served when the current reach equals the
# margin-free rule's and re-measured only where the margin binds (catalog_r_v_ok).
R_V_DIRS = 128            # Fibonacci directions per shell
R_V_DELTA08 = -0.8        # the enclosed contrast of the 'delta08' radius: shell crossing of the spherical model
                          # (nonlinear -0.8 <-> linear -2.71; Sheth & van de Weygaert 2004)
# The R_v ALGORITHM, bumped by hand when the measurement changes (a cache records it, r_v_settings, and is re-measured
# when it differs): 1 = the first up-crossing alone, the loop stopping at it; 2 = the
# full shell matrix out to the search range, giving the crossing (bit-identical to 1), delta08 and ridge.
R_V_ALGORITHM = 2


def r_v_settings():
    """What a catalogue's radii were measured with, as it records them ('r_v_settings'): [step in sigma, reach in
    sigma, directions, algorithm, delta08 threshold]. Read at call time, so a changed module constant counts. A
    catalogue's record carries a sixth element, the reach in cells it searched
    (r_v_record); these five are the settings every other reader keys on (exports, the fate links)."""
    return np.array([R_V_STEP_SIGMA, R_V_MAX_SIGMA, R_V_DIRS, R_V_ALGORITHM, R_V_DELTA08], dtype=np.float64)


def r_v_reach_cells(sigma_cells, n, half_box_margin=True):
    """How far out the R_v search looks, in cells: min(R_V_MAX_SIGMA sigma, L/2 - sigma) on an n^3 grid (sigma
    floored at a cell, as the step is), never below 0. half_box_margin=False is the margin-free rule,
    min(R_V_MAX_SIGMA sigma, L/2): what a record without a reach was searched with (catalog_r_v_ok)."""
    s = max(float(sigma_cells), 1.0)
    return max(min(R_V_MAX_SIGMA * s, 0.5 * float(n) - (s if half_box_margin else 0.0)), 0.0)


def r_v_record(sigma_cells, n):
    """The record a newly measured catalogue carries: r_v_settings() plus the reach in cells it searched
    (r_v_reach_cells for its smoothing and grid)."""
    return np.append(r_v_settings(), r_v_reach_cells(sigma_cells, n))


def catalog_r_v_ok(cat, sigma_cells=None, n=None):
    """Whether a catalogue's radii were measured with the current settings (r_v_settings) out to the current reach
    (r_v_reach_cells for sigma_cells / n: config's smoothing and grid by default, the namespace the catalogue lives
    in) and it holds all three (catalog.R_V_DEFINITIONS). A cache without a record: not ok,
    re-measured once (else a changed R_V_MAX_SIGMA or algorithm would leave every cache with stale
    radii and NaNs). A record of the five settings alone (no reach) is ok
    when the margin-free half-box rule reaches as far as the current one does -- every sigma <= 10.24 cells at 512^3,
    the production
    smoothing included -- so the margin re-measures only the catalogues it changes."""
    if 'r_v_settings' not in cat or any(k not in cat for k in ('r_v_cells', 'r_v_delta08_cells', 'r_v_ridge_cells')):
        return False
    have = np.asarray(cat['r_v_settings'], dtype=np.float64).reshape(-1)
    want = r_v_settings()
    if have.shape[0] not in (want.shape[0], want.shape[0] + 1) or not np.array_equal(have[:want.shape[0]], want):
        return False
    sigma_cells = config.SMOOTHING_SIGMA_CELLS if sigma_cells is None else sigma_cells
    n = config.FIELD_RESOLUTION if n is None else n
    reach = r_v_reach_cells(sigma_cells, n)
    if have.shape[0] == want.shape[0]:
        return r_v_reach_cells(sigma_cells, n, half_box_margin=False) == reach
    return float(have[-1]) == reach


def r_v_stale_reason(cat, sigma_cells=None, n=None):
    """Why catalog_r_v_ok says no, in a few words for the log (None when it says yes)."""
    if catalog_r_v_ok(cat, sigma_cells, n):
        return None
    if 'r_v_settings' not in cat:
        return "has no record of its R_v settings"
    have = np.asarray(cat['r_v_settings'], dtype=np.float64).reshape(-1)
    want = r_v_settings()
    if have.shape[0] in (want.shape[0], want.shape[0] + 1) and np.array_equal(have[:want.shape[0]], want) \
            and all(k in cat for k in ('r_v_cells', 'r_v_delta08_cells', 'r_v_ridge_cells')):
        sigma_cells = config.SMOOTHING_SIGMA_CELLS if sigma_cells is None else sigma_cells
        n = config.FIELD_RESOLUTION if n is None else n
        old = float(have[-1]) if have.shape[0] > want.shape[0] else r_v_reach_cells(sigma_cells, n, False)
        return (f"was searched out to {old:.4g} cells, today's reach is {r_v_reach_cells(sigma_cells, n):.4g} "
                f"(L/2 less a sigma)")
    if have.shape[0] in (want.shape[0], want.shape[0] + 1) and np.array_equal(have[:want.shape[0]], want):
        return "lacks one of the three radii"
    return f"was measured with R_v settings {have[:want.shape[0]].tolist()}, not today's {want.tolist()}"


def _enclosed_contrast(radii, shells):
    """Delta(<r) at every radius of 'radii' (r_0 = 0 first) from the shell means 'shells' (rows = voids, column 0 =
    the value at r = 0): delta taken piecewise LINEAR in r between the shells and r^2 dr integrated exactly over each
    piece, so a constant profile gives Delta = that constant (a trapezoid on r^2 delta gives 1.5x it at the first
    shell). Column 0 is the central value itself."""
    a, b = radii[:-1], radii[1:]
    va, vb = shells[:, :-1], shells[:, 1:]
    d3 = (b ** 3 - a ** 3) / 3.0
    slope = (vb - va) / (b - a)
    piece = va * d3 + slope * ((b ** 4 - a ** 4) / 4.0 - a * d3)          # integral of delta(r) r^2 over [a, b]
    out = np.empty_like(shells)
    out[:, 0] = shells[:, 0]
    out[:, 1:] = 3.0 * np.cumsum(piece, axis=1) / radii[1:] ** 3
    return out


def void_radii_cells(delta_s, coords, sigma_cells):
    """The three measured radii per minimum, in cells, from ONE matrix of shell means of the SMOOTHED delta
    (trilinear, R_V_DIRS directions about the minimum's cell centre, every R_V_STEP_SIGMA sigma out to R_V_MAX_SIGMA
    sigma or half the box less a sigma, r_v_reach_cells; column 0 the minimum itself at r = 0)
    (catalog.R_V_DEFINITIONS):
      'r_v_cells'          the first up-crossing of 0 (the mean), linearly interpolated from the last negative shell
                           -- void_radius_cells, bit for bit;
      'r_v_delta08_cells'  where the enclosed contrast Delta(<R) (_enclosed_contrast) first reaches R_V_DELTA08 going
                           outward, interpolated likewise; NaN when the minimum itself is not below it (no sphere
                           about it is that empty: on a field smoothed at sigma this is the common case for shallow
                           minima) or it is not reached within reach;
      'r_v_ridge_cells'    the radius of the MAXIMUM of the shell mean over the search range (the compensation ridge),
                           refined by a parabola through it and its neighbours; NaN when the maximum is the outermost
                           shell (the profile still rising: the range truncated it, no ridge was seen).
    All three NaN for a minimum that is not underdense (delta_s >= 0). The question behind delta08 and the ridge:
    is the z < 3 shrinkage of the crossing R_v a property of the voids or the first-crossing rule latching onto an
    inner wall? A void whose crossing R_v falls while delta08 and the ridge hold is the latter."""
    from .profiles import fibonacci_sphere, shell_means          # here: profiles imports catalog, which imports this module
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    out = {k: np.full(len(coords), np.nan) for k in ('r_v_cells', 'r_v_delta08_cells', 'r_v_ridge_cells')}
    if not len(coords):
        return out
    n = delta_s.shape[0]
    step = R_V_STEP_SIGMA * max(float(sigma_cells), 1.0)
    radii = np.arange(1, int(r_v_reach_cells(sigma_cells, n) / step) + 1) * step   # L/2 less a sigma
    v0 = np.asarray(delta_s[coords[:, 0], coords[:, 1], coords[:, 2]], dtype=np.float64)
    todo = np.nonzero(v0 < 0)[0]                                # an overdense 'minimum' has no underdense region
    if not todo.size or not radii.size:
        return out
    dirs = fibonacci_sphere(R_V_DIRS)
    k = radii.size
    centers = np.repeat((coords[todo] + 0.5) / n, k, axis=0)   # (void, shell) pairs, void-major
    shells = np.empty((todo.size, k + 1))
    shells[:, 0] = v0[todo]
    shells[:, 1:] = shell_means(delta_s, centers, np.tile(radii, todo.size), dirs).reshape(todo.size, k)
    r_all = np.concatenate([[0.0], radii])

    def first_up(prof, level):
        # the first column >= level after column 0 (which is below it), interpolated from the column before
        hit = prof[:, 1:] >= level
        rows = np.nonzero(hit.any(axis=1) & (prof[:, 0] < level))[0]
        j = np.argmax(hit[rows], axis=1) + 1
        pr, pv, v = r_all[j - 1], prof[rows, j - 1], prof[rows, j]
        res = np.full(len(prof), np.nan)
        res[rows] = pr + (r_all[j] - pr) * (level - pv) / (v - pv)
        return res

    out['r_v_cells'][todo] = first_up(shells, 0.0)
    out['r_v_delta08_cells'][todo] = first_up(_enclosed_contrast(r_all, shells), R_V_DELTA08)
    j = np.argmax(shells[:, 1:], axis=1) + 1                    # the ridge: a shell, never the minimum itself
    inner = j < k                                               # the outermost shell is no ridge (truncated)
    jj, rows = j[inner], np.nonzero(inner)[0]
    f0, f1, f2 = shells[rows, jj - 1], shells[rows, jj], shells[rows, jj + 1]
    curv = f0 - 2.0 * f1 + f2
    off = np.where(curv < 0, 0.5 * (f0 - f2) / np.where(curv < 0, curv, -1.0), 0.0)
    ridge = np.full(todo.size, np.nan)
    ridge[rows] = r_all[jj] + np.clip(off, -0.5, 0.5) * step
    out['r_v_ridge_cells'][todo] = ridge
    return out


def void_radius_cells(delta_s, coords, sigma_cells):
    """R_v per minimum, in cells (NaN when there is none): the radius at which the spherically averaged SMOOTHED
    density contrast around the minimum first rises to 0 -- the underdense region the void is, MEASURED. The
    ellipsoid's 2 / sqrt|lambda| is where a QUADRATIC rises by a fixed 2 whatever the void's depth, ~1.6x the radius
    where the stacked density reaches the mean, v_r turns to infall and the single-stream fraction bottoms out
    (TNG100 z = 0). Shell means of delta_s (trilinear, R_V_DIRS directions, centred on the minimum's
    cell centre) every R_V_STEP_SIGMA sigma out to R_V_MAX_SIGMA sigma or half the box; the crossing linearly
    interpolated from the last negative shell (from r = 0: the minimum itself). Defined at every redshift (delta_s
    < 0 at a void's minimum); NaN for a minimum that is not underdense or never reaches 0 within reach. The
    'crossing' of void_radii_cells (one shell matrix gives all three definitions)."""
    return void_radii_cells(delta_s, coords, sigma_cells)['r_v_cells']


def distinct_voids(coords, delta_values, r_v_cells, n_grid, r_eff_cells=None):
    """Which of the given voids are DISTINCT: deepest first, a void whose centre lies inside the region a deeper void
    already kept claims is a sub-minimum of the same underdense region (periodic distances, cells). With r_eff_cells
    (catalog.distinct_mask always passes it) each void CLAIMS the sphere of min(R_v, R_eff) (a missing one ignored),
    so a void without R_v still takes part with its R_eff, and the tail of the deepest voids, whose R_v reaches 1.2-3
    R_eff (84th percentile R_v/R_eff 1.2-1.5), does not swallow 20-27% of the sample (R_v spheres alone, TNG300 z=0:
    11 voids with R_v > 10 sigma drop 99 of 365). Without it the claim is the R_v sphere alone, and a void without
    R_v is not distinct and claims no region. Meant for the WELL-RESOLVED voids (catalog.distinct_mask runs it over
    all of them, then narrows the result to the caller's sample):
    the smoothed field's minima within an 11-cell footprint often share one void -- 63% of the TNG100 z=0
    resolved voids have another centre within R_eff -- while over ALL minima a few deep ones that fail the cut,
    with R_v up to 14 sigma, would swallow three quarters of the resolved sample (24 of 103 left)."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    r_v = np.asarray(r_v_cells, dtype=np.float64).reshape(-1)
    if r_eff_cells is not None:
        r_v = np.fmin(r_v, np.asarray(r_eff_cells, dtype=np.float64).reshape(-1))   # fmin: a NaN side is ignored
    keep = np.zeros(len(coords), dtype=bool)
    kept_c, kept_r = np.zeros((0, 3)), np.zeros(0)
    for i in np.argsort(np.asarray(delta_values, dtype=np.float64), kind="stable"):
        if not np.isfinite(r_v[i]):
            continue
        if kept_r.size:
            d = coords[i] - kept_c
            d -= n_grid * np.rint(d / n_grid)
            if (np.sqrt((d * d).sum(axis=1)) < kept_r).any():
                continue
        keep[i] = True
        kept_c, kept_r = np.vstack([kept_c, coords[i]]), np.append(kept_r, r_v[i])
    return keep


def build_void_catalog(delta_s, hessian):
    fp = config.FOOTPRINT_SIZE
    _, coords = dtfe.find_local_minima(delta_s, footprint_size=fp)
    evals, evecs = eigh_at_coords(hessian, coords)

    keep = ~np.isnan(evals).any(axis=1)
    crit = config.VOID_EIGENVALUE_CRITERION
    if crit == 'positive':
        keep &= (evals > 0).all(axis=1)
    elif crit == 'trace':
        keep &= evals.sum(axis=1) > 0
    coords, evals, evecs = coords[keep], evals[keep], evecs[keep]

    shapes = dtfe.calculate_shape_parameters(list(evals))
    dvals = delta_s[coords[:, 0], coords[:, 1], coords[:, 2]]

    cat = {
        'coords': coords,
        'eigenvalues': evals,
        'eigenvectors': evecs,
        'axis_ratios': shapes['axis_ratios'],
        'bbks_params': shapes['bbks_params'],
        'delta_values': dvals,
        'deep': dvals < config.DEEP_VOID_THRESHOLD,
    }
    cat.update(_ellipsoid_fits(coords, evals, evecs))
    # the MEASURED radii (crossing, delta08 and ridge): in cells, frame-free like coords, so a refit
    # keeps them; with the settings they were measured with (catalog_r_v_ok)
    cat.update(void_radii_cells(delta_s, coords, config.SMOOTHING_SIGMA_CELLS))
    cat['r_v_settings'] = r_v_record(config.SMOOTHING_SIGMA_CELLS, delta_s.shape[0])   # with the reach searched
    cat['cell_mpc'] = np.float64(config.CELL_SIZE)      # the FRAME the positions and semi-axes are in
    cat['cuts_mpc'] = _cuts_array()                     # the cuts (in Mpc) the well_resolved flag was taken with
    cat['deep_threshold'] = np.float64(config.DEEP_VOID_THRESHOLD)   # the threshold 'deep' was taken at
    return cat


def set_smoothing(cells=None, mpc=None):
    """Set the smoothing length for this process: in cells (a script's --smooth: the footprint stays as it is, the
    thesis's r512_s5_f11 / s20_f11 caches rely on that) or in Mpc (--smooth-mpc: the same physical scale in every
    box, the footprint following it -- config.footprint_for). Keeps config.SMOOTHING_SIGMA_MPC in step. In cells the
    footprint is the PRODUCTION one even after an Mpc setting or under DTFE_SIGMA_MPC (else '--smooth 10' would give
    r512_s10_f5, neither the thesis's s10_f11 nor the Mpc run). Returns sigma
    in cells. Neither given (or <= 0): nothing changes."""
    if cells and mpc:
        raise ValueError("give the smoothing in cells or in Mpc, not both")
    if mpc and float(mpc) > 0:
        config.SMOOTHING_SIGMA_CELLS = round(float(mpc) / float(config.CELL_SIZE), 9)
        config.FOOTPRINT_SIZE = config.footprint_for(config.SMOOTHING_SIGMA_CELLS)
    elif cells and float(cells) > 0:
        config.SMOOTHING_SIGMA_CELLS = float(cells)
        config.FOOTPRINT_SIZE = int(config.PRODUCTION_SMOOTHING[1])     # cells: the production footprint, whatever was set
    config.SMOOTHING_SIGMA_MPC = float(config.SMOOTHING_SIGMA_CELLS) * float(config.CELL_SIZE)
    return config.SMOOTHING_SIGMA_CELLS


def add_smoothing_mpc_arg(parser):
    """The void scripts' --smooth-mpc (beside make_parser's --smooth in cells)."""
    parser.add_argument("--smooth-mpc", type=float, default=0.0, metavar="MPC",
                        help="the void catalogue's smoothing length in Mpc, the same physical scale in every box (the "
                             "minima footprint follows); instead of --smooth in cells (also: DTFE_SIGMA_MPC)")


def apply_smoothing_args(args, tag="pipeline"):
    """A void script's --smooth (cells) or --smooth-mpc into set_smoothing, said when it changes anything."""
    cells, mpc = getattr(args, "smooth", 0.0) or 0.0, getattr(args, "smooth_mpc", 0.0) or 0.0
    if cells > 0 and mpc > 0:
        raise SystemExit("give --smooth (cells) or --smooth-mpc, not both")
    if cells > 0 or mpc > 0:
        set_smoothing(cells=cells or None, mpc=mpc or None)
        print(f"[{tag}] smoothing: {config.SMOOTHING_SIGMA_CELLS:g} cells = {config.SMOOTHING_SIGMA_MPC:.3g} Mpc, "
              f"footprint {config.FOOTPRINT_SIZE} (cache namespace {_param_hash()})")
    return config.SMOOTHING_SIGMA_CELLS


def smoothing_tag() -> str:
    """'' at the production smoothing (config.PRODUCTION_SMOOTHING: 10 cells, footprint 11), else a tag for file
    and folder names -- 's2.1625Mpc_f5' -- so a --smooth / --smooth-mpc run never overwrites the default products.
    Six significant digits: '.3g' would make 2.16 and 2.1625 Mpc one tag over two different caches.
    A non-default void gate (DTFE_VOID_GATE, void_gate) adds 'gatenone': 'gatenone' alone at the
    production smoothing, 's2.1625Mpc_f5_gatenone' with both -- so every caller that keys names (and the thesis
    mirror's refusal) on this tag keeps the ungated products apart with no change of its own."""
    parts = []
    sig0, fp0 = config.PRODUCTION_SMOOTHING
    if (float(config.SMOOTHING_SIGMA_CELLS), int(config.FOOTPRINT_SIZE)) != (float(sig0), int(fp0)):
        parts.append(f"s{smoothing_length_mpc():.6g}Mpc_f{int(config.FOOTPRINT_SIZE)}")
    if void_gate() != 'ellipsoid':
        parts.append(f"gate{void_gate()}")
    return "_".join(parts)


def smoothing_length_mpc():
    """The smoothing length sigma in Mpc as set NOW (config.SMOOTHING_SIGMA_CELLS x CELL_SIZE; a script's --smooth
    changes it after import) -- the unit of the void cut, the profile bins, the abundance radii and the skeleton's
    cut-out. sigma = 0 (no smoothing: an r512_s0 cache exists) counts as ONE cell, so none of them collapses to 0."""
    return max(float(config.SMOOTHING_SIGMA_CELLS), 1.0) * float(config.CELL_SIZE)


def ellipsoid_cuts_mpc():
    """config.ELLIPSOID_CUT_RULES in Mpc for the current simulation and smoothing, read at CALL time (the plot
    scripts' --smooth sets config.SMOOTHING_SIGMA_CELLS after import, so an import-time value would hold the
    gate at 100 cells): min_axis_mpc = rule x cell, max_axis_mpc = min(rule x sigma, rule x box), the ratio."""
    r, cell = config.ELLIPSOID_CUT_RULES, float(config.CELL_SIZE)
    return {'min_axis_mpc': float(r['min_axis_cells']) * cell,
            'max_axis_mpc': min(float(r['max_axis_sigma']) * smoothing_length_mpc(),
                                float(r['max_axis_box_frac']) * float(config.BOX_SIZE)),
            'max_axis_ratio': float(r['max_axis_ratio'])}


VOID_GATES = ('ellipsoid', 'none')


def void_gate():
    """The void sample gate for this process (config.VOID_GATE, DTFE_VOID_GATE): 'ellipsoid' (default:
    well_resolved is the ellipsoid cut) or 'none' (the max-axis clause dropped, gate_view). Read at call time."""
    g = str(getattr(config, 'VOID_GATE', 'ellipsoid') or 'ellipsoid').strip().lower()
    if g not in VOID_GATES:
        raise ValueError(f"DTFE_VOID_GATE must be one of {VOID_GATES}, got {g!r}")
    return g


def gate_cuts_mpc():
    """The cuts the returned catalogue's well_resolved is taken with under void_gate(): ellipsoid_cuts_mpc(), with
    max_axis_mpc = inf under 'none'."""
    cuts = ellipsoid_cuts_mpc()
    if void_gate() == 'none':
        cuts['max_axis_mpc'] = float('inf')
    return cuts


def gate_view(cat):
    """The catalogue as the samples see it under void_gate(). 'ellipsoid': 'cat' itself. 'none' (under the gate
    the R_v abundance and profiles inherit the 10-sigma max-axis cut, which drops a void because its CURVATURE
    length is large, not because its R_v is unresolved -- about 77-79% of the z = 0 minima): a copy whose ellipsoid
    fits, well_resolved and 'cuts_mpc' are re-derived without the max-axis clause (min axis and axis ratio kept), so
    every sample built on well_resolved -- the R_v ones, and the R_eff / shape ones too -- is ungated, with no change
    in its script. NEVER saved: the cache keeps the production flags, so processes with either gate share it; the
    outputs are kept apart by smoothing_tag()'s 'gatenone'. Marked 'void_gate' = 'none'."""
    if void_gate() == 'ellipsoid':
        return cat
    out = dict(cat)
    cuts = gate_cuts_mpc()
    out.update(_ellipsoid_fits(np.asarray(cat['coords']), np.asarray(cat['eigenvalues']), np.asarray(cat['eigenvectors']),
                               cuts=cuts))
    out['cuts_mpc'] = np.array([cuts['min_axis_mpc'], cuts['max_axis_mpc'], cuts['max_axis_ratio']], dtype=np.float64)
    out['void_gate'] = np.str_('none')
    return out


def _cuts_array():
    c = ellipsoid_cuts_mpc()
    return np.array([c['min_axis_mpc'], c['max_axis_mpc'], c['max_axis_ratio']], dtype=np.float64)


def catalog_cuts_ok(cat):
    """Whether a catalogue's well_resolved and deep flags were taken with the current cuts (ellipsoid_cuts_mpc) and
    DEEP_VOID_THRESHOLD (a cache without the records: not ok, so it is re-derived once and
    then carries them)."""
    return ('cuts_mpc' in cat and np.allclose(np.asarray(cat['cuts_mpc'], dtype=np.float64), _cuts_array())
            and 'deep_threshold' in cat and float(cat['deep_threshold']) == float(config.DEEP_VOID_THRESHOLD))


def snapshot_frame(products):
    """(cell_mpc, box_mpc, grid_n, source) of a snapshot: its header's box over its grid -- THE truth about
    the frame -- which must be config.CELL_SIZE (the DTFE_SIM box over FIELD_RESOLUTION), the frame every
    catalogue is built, cut and exported in; else ValueError, BEFORE any cache is read or written
    (with DTFE_SIM unset a refit would turn a correct TNG100 cache into TNG50's frame). It needs the
    snapshot header: FieldSet raises FileNotFoundError without a combined file, so 'source' is always
    'header' (kept in the tuple for the provenance)."""
    fs = products.fs
    n = int(fs.grid_n)
    box = float(fs.meta.box_mpc)
    cell = box / n
    if abs(cell - config.CELL_SIZE) > 1e-6 * max(cell, config.CELL_SIZE):
        why = []                                # the hint by cause: only advice that would change something
        if products.sim != config.SIMULATION:
            why.append(f"run with DTFE_SIM={products.sim}")
        if n != config.FIELD_RESOLUTION:
            why.append(f"set config.FIELD_RESOLUTION = {n} (this grid's; hard-coded in python/config.py) or regrid at "
                       f"{config.FIELD_RESOLUTION}")
        if not why:
            why.append(f"the header's box {box:g} Mpc is not config.BOX_SIZE {config.BOX_SIZE:g} "
                       f"(config.SIMULATION_BOX_MPC['{config.SIMULATION}'] or the box read for it)")
        why = " and ".join(why)
        raise ValueError(f"the snapshot's cell is {cell:.6g} Mpc (box {box:g} Mpc / {n}), the catalogue's frame "
                         f"config.CELL_SIZE is {config.CELL_SIZE:.6g} (DTFE_SIM={config.SIMULATION}: box "
                         f"{config.BOX_SIZE:g} / {config.FIELD_RESOLUTION}): {why}")
    return cell, box, n, "header"


def adopt_snapshot_grid(products):
    """Make config's frame THIS snapshot's (its header's box over its grid n) for a tool that shows one snapshot --
    the launcher's void outlines -- keeping the smoothing length in Mpc (the footprint follows, set_smoothing): a
    256^3 or 1024^3 run, or a simulation config does not pin, gets its catalogue at the same physical sigma, cached
    under its own namespace (r{n}_...). Returns (cell_mpc, box_mpc, n). NOT for the analysis scripts: their frame is
    DTFE_SIM's on purpose, and snapshot_frame refuses a grid that is not in it."""
    fs = products.fs
    n, box = int(fs.grid_n), float(fs.meta.box_mpc)
    if n != int(config.FIELD_RESOLUTION) or abs(box - float(config.BOX_SIZE)) > 1e-6 * box:
        sigma_mpc = float(config.SMOOTHING_SIGMA_CELLS) * float(config.CELL_SIZE)   # not smoothing_length_mpc's floor
        config.FIELD_RESOLUTION, config.BOX_SIZE, config.CELL_SIZE = n, box, box / n
        set_smoothing(mpc=sigma_mpc)                 # sigma 0 (no smoothing) stays 0 cells
    return float(config.CELL_SIZE), box, n


def refit_catalog(cat):
    """The ellipsoid fits of a cached catalogue re-derived in THIS frame with THESE cuts: positions_mpc,
    the semi-axes, orientations, ellipticity, prolateness and well_resolved all follow from the cached
    minima (coords) and Hessian eigen-decomposition (in cell units: frame-free), so a catalogue built
    under another DTFE_SIM's cell, or before a change of the cuts (ellipsoid_cuts_mpc), is put right without the
    field (18 s and 10 GB at 512^3 for the whole build; this is milliseconds). 'deep' is retaken from the
    cached minima against config.DEEP_VOID_THRESHOLD; the frame, the cuts and the threshold are recorded."""
    out = dict(cat)
    out.update(_ellipsoid_fits(np.asarray(cat['coords']), np.asarray(cat['eigenvalues']), np.asarray(cat['eigenvectors'])))
    out['deep'] = np.asarray(cat['delta_values']) < config.DEEP_VOID_THRESHOLD
    out['cell_mpc'] = np.float64(config.CELL_SIZE)
    out['cuts_mpc'] = _cuts_array()
    out['deep_threshold'] = np.float64(config.DEEP_VOID_THRESHOLD)
    return out


def catalog_cell_mpc(cat):
    """The cell size a cached catalogue was built with: its 'cell_mpc' entry, else recovered EXACTLY from
    positions_mpc = coords x cell (the first void with a non-zero index); None for an empty catalogue."""
    if 'cell_mpc' in cat:
        return float(cat['cell_mpc'])
    coords = np.asarray(cat['coords'])
    pos = np.asarray(cat['positions_mpc'], dtype=np.float64)
    nz = np.argwhere(coords > 0)
    if not len(nz):
        return None
    i, k = nz[0]
    return float(pos[i, k] / coords[i, k])


def catalog_frame_ok(cat, cell_mpc=None, rtol=1e-5):
    """Whether a catalogue's frame is 'cell_mpc' (default config.CELL_SIZE); True for an empty one."""
    c = catalog_cell_mpc(cat)
    want = float(config.CELL_SIZE if cell_mpc is None else cell_mpc)
    return c is None or abs(c - want) <= rtol * max(want, c)


def ellipsoid_fit_one(evals, evecs, cell):
    """Cut-free ellipsoid fit of ONE void from its Hessian eigen-decomposition.

    The single source of the fit formulas: semi-axes 2*cell/sqrt(|lambda|) ordered a>=b>=c,
    orientation columns in the same order (major axis first), ell = 1-c/a and
    prol = (a^2-b^2)/(a^2-c^2). Used by _ellipsoid_fits (which then applies the
    ellipsoid_cuts_mpc resolution gate) and by the void tracker (which reports sub-resolution
    epochs explicitly instead of dropping them). Returns None for degenerate eigenvalues."""
    ev = np.asarray(evals, dtype=np.float64)
    if np.any(np.abs(ev) < 1e-10):
        return None
    ev_abs = np.abs(ev) + 1e-12
    order = np.argsort(ev_abs)
    axes = 2.0 * cell / np.sqrt(ev_abs[order])
    a, b, c = axes
    return {
        'semi_axes': axes,
        'orientation': np.asarray(evecs)[:, order],
        'ell': float(1 - c / a),
        'prol': float((a * a - b * b) / (a * a - c * c)) if (a * a - c * c) > 1e-12 else 0.0,
    }


def _ellipsoid_fits(coords, evals, evecs, cuts=None):
    # Batched twin of ellipsoid_fit_one (same formulas, thresholds and argsort tie order --
    # keep them in lockstep) + the ellipsoid_cuts_mpc() resolution gate (or 'cuts': gate_view's), vectorized over voids.
    # Bit-equivalence with the per-void path is covered by tests/py_dtfelib_test.py.
    cell = config.CELL_SIZE
    cuts = ellipsoid_cuts_mpc() if cuts is None else cuts
    n = len(coords)
    valid = np.zeros(n, dtype=bool)
    semi_axes = np.zeros((n, 3), dtype=np.float32)
    orient = np.zeros((n, 3, 3), dtype=np.float32)
    ell = np.zeros(n, dtype=np.float32)
    prol = np.zeros(n, dtype=np.float32)

    if n:
        ev = np.asarray(evals, dtype=np.float64).reshape(n, 3)
        ev_abs = np.abs(ev) + 1e-12
        order = np.argsort(ev_abs, axis=1)
        axes = 2.0 * cell / np.sqrt(np.take_along_axis(ev_abs, order, axis=1))
        ok = (~(np.abs(ev) < 1e-10).any(axis=1)
              & ~(axes.min(axis=1) < cuts['min_axis_mpc'])
              & ~(axes.max(axis=1) > cuts['max_axis_mpc'])
              & ~(axes[:, 0] / axes[:, 2] > cuts['max_axis_ratio']))
        idx = np.where(ok)[0]
        valid[idx] = True
        semi_axes[idx] = axes[idx]
        orient[idx] = np.take_along_axis(np.asarray(evecs).reshape(n, 3, 3),
                                         order[:, None, :], axis=2)[idx]
        a, b, c = axes[idx, 0], axes[idx, 1], axes[idx, 2]
        ell[idx] = 1 - c / a
        d2 = a * a - c * c
        p = np.zeros(idx.size)
        pm = d2 > 1e-12
        p[pm] = (a * a - b * b)[pm] / d2[pm]
        prol[idx] = p

    return {
        'well_resolved': valid,
        'semi_axes': semi_axes,
        'orientations': orient,
        'fit_ellipticity': ell,
        'fit_prolateness': prol,
        'positions_mpc': coords.astype(np.float32) * cell,
    }


def _param_hash():
    return (f"r{config.FIELD_RESOLUTION}_s{config.SMOOTHING_SIGMA_CELLS:g}"
            f"_f{config.FOOTPRINT_SIZE}_{config.VOID_EIGENVALUE_CRITERION}"
            f"_g{config.GRADIENT_THRESHOLD:g}_v2")


def cache_dir():
    d = Path(config.CACHE_DIR) / _param_hash()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_npz(path, **arrays):
    """Write a cache whole or not at all: into a sibling temp file, then os.replace onto 'path'. Written
    in place, a build killed mid-save (a memory-pressure SIGKILL, the launcher's void helper stopped when the output
    changes, Ctrl-C) or two processes saving the same snapshot would leave a cut zip that every later np.load refuses.
    The
    temp name ends in .npz (numpy appends one otherwise) and carries the pid, so two writers never share it."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


class SnapshotProducts:
    """Per-snapshot derived products (delta, hessian, phi, voids, ...) with disk caching.

    Backed by dtfelib.FieldSet, so it works for BOTH estimators: pass method='ps' or
    method='dtfe' (default 'auto' = whichever exists in the snapshot directory). Density
    enters all derived products in MEAN units (rho/rho_bar) for either method, so every
    downstream quantity is method-independent by construction. Caches are namespaced by
    (sim, [prefix,] method, snapshot) -- caches keyed by snapshot alone are never
    reused, and an alternate OUTPUT_PREFIX set never shares caches with the primary one.
    """

    def __init__(self, snapshot, redshift=None, sim=None, method=None, prefix=None, data_root=None):
        self.snapshot = f"{int(snapshot):03d}"   # accepts 99, '99' or '099'
        self.sim = require_sim(sim)              # --sim / DTFE_SIM: there is no default simulation
        self.method = method or "auto"
        self.prefix = prefix                     # alternate OUTPUT_PREFIX set, see FieldSet
        self.data_root = data_root               # None = DATA_ROOT (DTFE_DATA_ROOT); a script's --data-root
        self._fs = None
        self._redshift = redshift
        self._ram = {}
        self._stamp = None                       # (size, mtime_ns) of the density grid, taken BEFORE it is read

    @property
    def fs(self):
        if self._fs is None:
            self._fs = FieldSet(sim_dir(self.sim, self.data_root) / f"snapdir_{self.snapshot}",
                                method=self.method, prefix=self.prefix)
        return self._fs

    @property
    def data_dir(self):
        return self.fs.snapdir

    @property
    def redshift(self):
        if self._redshift is None:
            try:
                self._redshift = self.fs.meta.redshift
            except FileNotFoundError:          # no snapshot file: fall back to the config table
                self._redshift = config.get_redshift(self.snapshot)
        return self._redshift

    @property
    def density_raw(self):
        """Density in MEAN units (rho/rho_bar) -- identical semantics for ps and dtfe."""
        if 'density_raw' not in self._ram:
            self._ram['density_raw'] = self.fs.density(units='mean')
        return self._ram['density_raw']

    @property
    def density_mean(self):
        return float(np.mean(self.density_raw))

    @property
    def delta_smoothed(self):
        if 'delta_s' not in self._ram:
            delta = dtfe.calculate_density_contrast(self.density_raw)
            self._ram['delta_s'] = dtfe.smooth_field(
                delta, sigma=config.SMOOTHING_SIGMA_CELLS)
        return self._ram['delta_s']

    @property
    def hessian(self):
        if 'hessian' not in self._ram:
            self._ram['hessian'] = hessian_from_field(self.delta_smoothed)
        return self._ram['hessian']

    @property
    def phi(self):
        if 'phi' not in self._ram:
            self._ram['phi'] = potential_from_delta(
                self.delta_smoothed, self.fs.meta.box_mpc, self.density_mean)
        return self._ram['phi']

    def release(self):
        self._ram.clear()


    def _cpath(self, name):
        # an alternate OUTPUT_PREFIX set gets its own cache namespace -- A/B validation
        # against e.g. ps_mw.* must never reuse (or poison) the primary ps_output caches
        tag = "" if self.prefix is None else f"{str(self.prefix).rstrip('.')}_"
        return cache_dir() / f"{self.sim}_{tag}{self.fs.method}_{self.snapshot}_{name}.npz"

    def _grid_path(self):
        """The density grid every product derives from; None for a cache-only test double (no FieldSet paths)."""
        fp = getattr(self.fs, "_field_path", None)
        return None if fp is None else fp("density")

    def _grid_stamp(self):
        """(size, mtime_ns) of the density grid, taken once -- the first time a cache is looked up, i.e. BEFORE
        the grid is read for a build, so a grid rewritten during the build cannot pass as the one built from.
        Empty when there is no grid to stamp."""
        if self._stamp is None:
            self._stamp = np.zeros(0, dtype=np.int64)
            grid = self._grid_path()
            if grid is not None:
                try:
                    st = grid.stat()
                    self._stamp = np.array([st.st_size, st.st_mtime_ns], dtype=np.int64)
                except OSError:
                    pass
        return self._stamp

    def _save(self, p, arrays):
        """Write a product's cache with the stamp of the grid it came from."""
        _save_npz(p, **dict(arrays, grid_stamp=self._grid_stamp()))

    def _cached(self, name):
        """The cache file of product 'name' when it may be used: present AND from the density grid on disk now.
        Every product is a function of that grid, and the cache key holds only the parameters (a rerun that
        rewrites the grids would otherwise go unnoticed). A cache
        carries the grid's (size, mtime_ns) and must match it exactly (a copied cache, a future
        grid mtime, a grid rewritten mid-build: none fools an equality); a cache without the stamp is
        judged by modification time (older than the grid = stale). A stale cache is rebuilt, and says so; None
        then (and when there is no cache). So is an UNREADABLE one (a cut zip: BadZipFile / EOFError)."""
        p = self._cpath(name)
        if not p.exists():
            return None
        stamp = self._grid_stamp()
        if stamp.size == 0:
            return p                            # no grid to compare with (a cache-only test double): use it
        try:
            with np.load(p) as z:
                have = z["grid_stamp"] if "grid_stamp" in z.files else None
        except (zipfile.BadZipFile, EOFError, OSError, KeyError, ValueError, zlib.error) as e:
            print(f"    [pipeline] cached {p.name} is unreadable ({type(e).__name__}: {e}): rebuilding it")
            return None
        if have is not None and have.size:
            stale = not np.array_equal(have, stamp)
            why = "was built from another version of the grid"
        else:
            stale = int(stamp[1]) > p.stat().st_mtime_ns
            why = f"is older than {self._grid_path().name} (the grid was rewritten since)"
        if stale:
            print(f"    [pipeline] cached {p.name} {why}: rebuilding it")
            return None
        return p

    def voids(self):
        """The void catalogue. From the cache, whose minima and eigenvalues (cell units) never go stale; the
        ellipsoid fits on top of them are re-derived (refit_catalog, milliseconds, no field) and the cache
        rewritten when their FRAME is not config.CELL_SIZE -- the positions and semi-axes scale with the
        cell of the DTFE_SIM simulation at build time, which a script's --sim does not change: a
        TNG100-3-Dark catalogue built under TNG50's frame (positions up to 51.6 Mpc in a
        110.7 Mpc box) reads as hundreds of 'resolved' voids where the right frame leaves a handful --
        or when the cuts (ellipsoid_cuts_mpc) changed since it was built (they are not in the cache's parameter
        hash). Says so. A cache whose radii were measured with other settings (catalog_r_v_ok) gets them re-measured
        from the smoothed field. The cache always
        holds the PRODUCTION gate; DTFE_VOID_GATE=none is applied to the returned dict only (gate_view)."""
        snapshot_frame(self)                    # the header's frame must be config's: refuse before any cache is touched
        p = self._cpath('voids')
        if self._cached('voids') is not None:
            with np.load(p) as z:
                cat = {k: z[k] for k in z.files}
            remeasured = False
            # the reach depends on the grid's n: the snapshot's own (snapshot_frame passed: its cell is config's)
            n_grid = int(self.fs.grid_n)
            if not catalog_r_v_ok(cat, n=n_grid) and self._grid_path() is not None:
                # the radii need the smoothed field, which a refit does not have; the minima are this grid's (the
                # stamp passed), so only the radii are re-measured: no Hessian, saved once with the rest
                why = r_v_stale_reason(cat, n=n_grid)
                print(f"    [pipeline] cached void catalogue {p.name} {why}: re-measuring R_v (crossing, delta08, ridge)")
                field = self.delta_smoothed
                cat.update(void_radii_cells(field, cat['coords'], config.SMOOTHING_SIGMA_CELLS))
                cat['r_v_settings'] = r_v_record(config.SMOOTHING_SIGMA_CELLS, field.shape[0])
                remeasured = True
            frame_ok, cuts_ok = catalog_frame_ok(cat), catalog_cuts_ok(cat)
            if frame_ok and cuts_ok:
                if remeasured:
                    self._save(p, cat)
                return gate_view(cat)
            why = []
            if not frame_ok:
                why.append(f"built with cell {catalog_cell_mpc(cat):.5g} Mpc, not this frame's {config.CELL_SIZE:.5g} "
                           f"(DTFE_SIM={config.SIMULATION})")
            if not cuts_ok:
                why.append("its ellipsoid cuts or deep threshold are not config's" if 'cuts_mpc' in cat
                           else "it carries no record of its ellipsoid cuts")
            print(f"    [pipeline] cached void catalogue {p.name}: {'; '.join(why)}: re-deriving its fits")
            cat = refit_catalog(cat)
            self._save(p, cat)
            return gate_view(cat)
        cat = build_void_catalog(self.delta_smoothed, self.hessian)
        self._save(p, cat)
        return gate_view(cat)

    def critical_points(self):
        p = self._cpath('critical')
        if self._cached('critical') is not None:
            with np.load(p) as z:
                flat = {k: z[k] for k in z.files}
        else:
            cp = find_critical_points(self.delta_smoothed, self.hessian,
                                      gradient_magnitude_from_field(self.delta_smoothed))
            flat = {}
            for t in ('minima', 'maxima', 'saddle1', 'saddle2'):
                for k in ('coords', 'values', 'eigenvalues'):
                    flat[f"{t}_{k}"] = cp[t][k]
            flat['field_mean'] = np.array(cp['field_mean'])
            flat['field_std'] = np.array(cp['field_std'])
            self._save(p, flat)
        out = {'field_mean': float(flat['field_mean']),
               'field_std': float(flat['field_std'])}
        for t in ('minima', 'maxima', 'saddle1', 'saddle2'):
            out[t] = {k: flat[f"{t}_{k}"] for k in ('coords', 'values', 'eigenvalues')}
        return out

    def eigen_slices(self):
        p = self._cpath('eigslices')
        if self._cached('eigslices') is not None:
            with np.load(p) as z:
                return {d: {k: z[f"{d}_{k}"] for k in
                            ('lambda1', 'lambda2', 'lambda3', 'trace', 'det')}
                        for d in (0, 1, 2)}
        out, flat = {}, {}
        for d in (0, 1, 2):
            out[d] = hessian_slice(self.hessian, d)
            for k, v in out[d].items():
                flat[f"{d}_{k}"] = v
        self._save(p, flat)
        return out

    def delta_slices(self):
        p = self._cpath('deltaslices')
        if self._cached('deltaslices') is not None:
            with np.load(p) as z:
                return {d: z[str(d)] for d in (0, 1, 2)}
        out = {d: dtfe.extract_2d_slice(self.delta_smoothed, d).astype(np.float32)
               for d in (0, 1, 2)}
        self._save(p, {str(d): v for d, v in out.items()})
        return out

    def phi_slices(self):
        p = self._cpath('phislices')
        if self._cached('phislices') is not None:
            with np.load(p) as z:
                return {d: z[str(d)] for d in (0, 1, 2)}
        out = {d: dtfe.extract_2d_slice(self.phi, d).astype(np.float32)
               for d in (0, 1, 2)}
        self._save(p, {str(d): v for d, v in out.items()})
        return out

    def phi_maxima(self):
        p = self._cpath('phimaxima')
        if self._cached('phimaxima') is not None:
            with np.load(p) as z:
                return {k: z[k] for k in z.files}
        mask = (self.phi == dtfe.box_filter(self.phi, config.FOOTPRINT_SIZE, mode='wrap', maximum=True))
        coords = np.argwhere(mask)
        cat = {'coords': coords, 'values': self.phi[mask]}
        self._save(p, cat)
        return cat


    def warm(self, verbose=True):
        steps = [('voids', self.voids), ('critical_points', self.critical_points),
                 ('eigen_slices', self.eigen_slices), ('delta_slices', self.delta_slices),
                 ('phi_slices', self.phi_slices), ('phi_maxima', self.phi_maxima)]
        for name, fn in steps:
            if verbose:
                print(f"    {name}")
            fn()
        return self


def products(snapshot, redshift=None, sim=None, method=None, prefix=None, data_root=None):
    """The derived products of one snapshot. data_root: a script's --data-root (default: DATA_ROOT, so
    DTFE_DATA_ROOT); the caches stay keyed by (sim, prefix, method, snapshot), as they are for a root
    set through the environment."""
    return SnapshotProducts(snapshot, redshift, sim=sim, method=method, prefix=prefix, data_root=data_root)


def _limits_path(sim=None):
    # one file PER SIMULATION: cache_dir() is hashed on the parameters only, so with one global file
    # compute runs for two simulations would max-merge, a TNG300 series getting
    # the TNG50 amplitudes (sigma is in cells: a different Mpc smoothing, a different scale)
    return cache_dir() / f'global_limits_{require_sim(sim)}.json'


def save_global_limits(limits, sim=None):
    with open(_limits_path(sim), 'w') as f:
        json.dump(limits, f, indent=2)


def _snapshot_limits_path(sim=None):
    return cache_dir() / f'snapshot_limits_{require_sim(sim)}.json'


def load_snapshot_limits(sim=None):
    """{snapshot id ('099'): {field: amplitude}} -- what analyze.py compute measured per snapshot of
    'sim' (default: the DTFE_SIM simulation)."""
    p = _snapshot_limits_path(sim)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def canonical_snapshot_id(snap):
    """'99', 99, '099' -> '099' (config.SNAPSHOT_TO_REDSHIFT's keys); other text unchanged."""
    s = str(snap)
    return f"{int(s):03d}" if s.isdigit() else s


def record_limits(updates, canonical=None, sim=None):
    """analyze.py compute's bookkeeping. 'updates' = {snapshot: {field: amplitude}} for the snapshots
    just computed of simulation 'sim' (default DTFE_SIM). They go into the per-snapshot store
    (snapshot_limits_<sim>.json, beside global_limits_<sim>.json under the same parameter-hashed cache
    dir, so a change of smoothing or criterion invalidates both); the global limits are then the MAXIMUM
    over every stored snapshot of the canonical series, not over
    the snapshots of one run (a run on a subset, or on an unknown id, would shrink or empty the
    cross-epoch limits every later plot uses). Returns (limits, covered, missing): the limits written
    (None when nothing was computed: the global file is left alone), the canonical snapshots with an
    entry, and those still without one."""
    canonical = [canonical_snapshot_id(s) for s in (canonical if canonical is not None
                                                     else config.SNAPSHOT_TO_REDSHIFT)]
    store = load_snapshot_limits(sim)
    for snap, amp in (updates or {}).items():
        vals = {k: float(v) for k, v in amp.items() if v is not None and np.isfinite(v)}
        if vals:
            store[canonical_snapshot_id(snap)] = vals
    covered = [s for s in canonical if s in store]
    missing = [s for s in canonical if s not in store]
    if not updates:
        return None, covered, missing
    with open(_snapshot_limits_path(sim), 'w') as f:
        json.dump(store, f, indent=2)
    limits = {}
    for s in covered:
        for field, v in store[s].items():
            limits[field] = max(limits.get(field, v), v)
    save_global_limits(limits, sim)
    return limits, covered, missing


def load_global_limits(sim=None):
    p = _limits_path(sim)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def series_vmax(field, fallback_data=None, sim=None):
    """The cross-epoch colour amplitude of 'field' for simulation 'sim' (default DTFE_SIM): a fixed
    config.FIELD_LIMITS entry, else what analyze.py compute recorded for that simulation, else the
    robust maximum of 'fallback_data' (None without it)."""
    fixed = config.FIELD_LIMITS.get(field)
    if fixed is not None:
        return float(fixed)
    gl = load_global_limits(sim)
    if field in gl and gl[field] is not None:
        v = gl[field]
        return float(v if not isinstance(v, (list, tuple)) else v[1])
    from . import figures as style
    return style.robust_vmax(fallback_data) if fallback_data is not None else None
