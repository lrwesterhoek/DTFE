
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import minimum_filter, maximum_filter

import config
from .io import FieldSet
from .cli import DATA_ROOT, DEFAULT_SIM, sim_dir
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
    fp = config.FOOTPRINT_SIZE
    minima_mask = (delta_s == minimum_filter(delta_s, size=fp, mode='wrap'))
    maxima_mask = (delta_s == maximum_filter(delta_s, size=fp, mode='wrap'))

    gmax = float(np.max(grad_mag))
    gnorm = grad_mag / gmax if gmax > 0 else grad_mag
    cand = gnorm < config.GRADIENT_THRESHOLD
    inv = gmax - grad_mag
    cand &= (inv == maximum_filter(inv, size=fp, mode='wrap'))
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
R_V_MAX_SIGMA = 15.0      # how far out it looks (capped at half the box)
R_V_DIRS = 128            # Fibonacci directions per shell


def void_radius_cells(delta_s, coords, sigma_cells):
    """R_v per minimum, in cells (NaN when there is none): the radius at which the spherically averaged SMOOTHED
    density contrast around the minimum first rises to 0 -- the underdense region the void is, MEASURED. The
    ellipsoid's 2 / sqrt|lambda| is where a QUADRATIC rises by a fixed 2 whatever the void's depth, ~1.6x the radius
    where the stacked density reaches the mean, v_r turns to infall and the single-stream fraction bottoms out
    (TNG100 z = 0, 2026-10-07). Shell means of delta_s (trilinear, R_V_DIRS directions, centred on the minimum's
    cell centre) every R_V_STEP_SIGMA sigma out to R_V_MAX_SIGMA sigma or half the box; the crossing linearly
    interpolated from the last negative shell (from r = 0: the minimum itself). Defined at every redshift (delta_s
    < 0 at a void's minimum); NaN for a minimum that is not underdense or never reaches 0 within reach."""
    from .profiles import fibonacci_sphere, shell_means          # here: profiles imports catalog, which imports this module
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    out = np.full(len(coords), np.nan)
    if not len(coords):
        return out
    n = delta_s.shape[0]
    step = R_V_STEP_SIGMA * max(float(sigma_cells), 1.0)
    radii = np.arange(1, int(min(R_V_MAX_SIGMA * max(float(sigma_cells), 1.0), 0.5 * n) / step) + 1) * step
    centers = (coords + 0.5) / n
    dirs = fibonacci_sphere(R_V_DIRS)
    prev_v = np.asarray(delta_s[coords[:, 0], coords[:, 1], coords[:, 2]], dtype=np.float64)
    prev_r = np.zeros(len(coords))
    todo = np.nonzero(prev_v < 0)[0]                            # an overdense 'minimum' has no underdense region
    for r in radii:
        if not todo.size:
            break
        v = shell_means(delta_s, centers[todo], np.full(todo.size, r), dirs)
        hit = v >= 0
        i = todo[hit]
        out[i] = prev_r[i] + (r - prev_r[i]) * (-prev_v[i]) / (v[hit] - prev_v[i])
        stay = todo[~hit]
        prev_v[stay], prev_r[stay] = v[~hit], r
        todo = stay
    return out


def distinct_voids(coords, delta_values, r_v_cells, n_grid, r_eff_cells=None):
    """Which of the given voids are DISTINCT: deepest first, a void whose centre lies inside the R_v sphere of a
    deeper void already kept is a sub-minimum of the same underdense region (periodic distances, cells). A void
    without R_v is not distinct and claims no region -- unless r_eff_cells is given: then each void CLAIMS the
    region of min(R_v, R_eff) (a missing one ignored), so a void without R_v still takes part with its R_eff, and the
    tail of the deepest voids, whose R_v reaches 1.2-3 R_eff (84th percentile R_v/R_eff 1.2-1.5), no longer swallows
    20-27% of the sample (review 2026-10-07: TNG300 z=0, 11 voids with R_v > 10 sigma dropped 99 of 365). Meant for
    the voids of ONE sample (catalog.distinct_mask):
    the smoothed field's minima within an 11-cell footprint often share one void -- 63% of the TNG100 z=0
    resolved voids had another centre within R_eff -- while over ALL minima a few deep ones that fail the cut,
    with R_v up to 14 sigma, would swallow three quarters of the resolved sample (2026-10-07: 24 of 103 left)."""
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
    # the MEASURED radius (2026-10-07): in cells, frame-free like coords, so a refit keeps it
    cat['r_v_cells'] = void_radius_cells(delta_s, coords, config.SMOOTHING_SIGMA_CELLS)
    cat['cell_mpc'] = np.float64(config.CELL_SIZE)      # the FRAME the positions and semi-axes are in (2026-10-05)
    cat['cuts_mpc'] = _cuts_array()                     # the cuts (in Mpc) the well_resolved flag was taken with
    cat['deep_threshold'] = np.float64(config.DEEP_VOID_THRESHOLD)   # the threshold 'deep' was taken at
    return cat


def set_smoothing(cells=None, mpc=None):
    """Set the smoothing length for this process: in cells (a script's --smooth: the footprint stays as it is, the
    thesis's r512_s5_f11 / s20_f11 caches rely on that) or in Mpc (--smooth-mpc: the same physical scale in every
    box, the footprint following it -- config.footprint_for). Keeps config.SMOOTHING_SIGMA_MPC in step (it went
    stale under --smooth before). In cells the footprint is the PRODUCTION one even after an Mpc setting or under
    DTFE_SIGMA_MPC (else '--smooth 10' gave r512_s10_f5, neither the thesis's s10_f11 nor the Mpc run). Returns sigma
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
    Six significant digits: '.3g' made 2.16 and 2.1625 Mpc one tag over two different caches (review 2026-10-07)."""
    sig0, fp0 = config.PRODUCTION_SMOOTHING
    if (float(config.SMOOTHING_SIGMA_CELLS), int(config.FOOTPRINT_SIZE)) == (float(sig0), int(fp0)):
        return ""
    return f"s{smoothing_length_mpc():.6g}Mpc_f{int(config.FOOTPRINT_SIZE)}"


def smoothing_length_mpc():
    """The smoothing length sigma in Mpc as set NOW (config.SMOOTHING_SIGMA_CELLS x CELL_SIZE; a script's --smooth
    changes it after import) -- the unit of the void cut, the profile bins, the abundance radii and the skeleton's
    cut-out. sigma = 0 (no smoothing: an r512_s0 cache exists) counts as ONE cell, so none of them collapses to 0."""
    return max(float(config.SMOOTHING_SIGMA_CELLS), 1.0) * float(config.CELL_SIZE)


def ellipsoid_cuts_mpc():
    """config.ELLIPSOID_CUT_RULES in Mpc for today's simulation and smoothing, read at CALL time (the plot
    scripts' --smooth sets config.SMOOTHING_SIGMA_CELLS after import, so an import-time value would hold the
    gate at 100 cells): min_axis_mpc = rule x cell, max_axis_mpc = min(rule x sigma, rule x box), the ratio."""
    r, cell = config.ELLIPSOID_CUT_RULES, float(config.CELL_SIZE)
    return {'min_axis_mpc': float(r['min_axis_cells']) * cell,
            'max_axis_mpc': min(float(r['max_axis_sigma']) * smoothing_length_mpc(),
                                float(r['max_axis_box_frac']) * float(config.BOX_SIZE)),
            'max_axis_ratio': float(r['max_axis_ratio'])}


def _cuts_array():
    c = ellipsoid_cuts_mpc()
    return np.array([c['min_axis_mpc'], c['max_axis_mpc'], c['max_axis_ratio']], dtype=np.float64)


def catalog_cuts_ok(cat):
    """Whether a catalogue's well_resolved and deep flags were taken with today's cuts (ellipsoid_cuts_mpc) and
    DEEP_VOID_THRESHOLD (a cache without the records predates them: not ok, so it is re-derived once and
    then carries them)."""
    return ('cuts_mpc' in cat and np.allclose(np.asarray(cat['cuts_mpc'], dtype=np.float64), _cuts_array())
            and 'deep_threshold' in cat and float(cat['deep_threshold']) == float(config.DEEP_VOID_THRESHOLD))


def snapshot_frame(products):
    """(cell_mpc, box_mpc, grid_n, source) of a snapshot: its header's box over its grid -- THE truth about
    the frame -- which must be config.CELL_SIZE (the DTFE_SIM box over FIELD_RESOLUTION), the frame every
    catalogue is built, cut and exported in; else ValueError, BEFORE any cache is read or written (review
    2026-10-05: with DTFE_SIM unset the refit turned a correct TNG100 cache into TNG50's frame). It needs the
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


def _ellipsoid_fits(coords, evals, evecs):
    # Batched twin of ellipsoid_fit_one (same formulas, thresholds and argsort tie order --
    # keep them in lockstep) + the ellipsoid_cuts_mpc() resolution gate, vectorized over voids.
    # Bit-equivalence with the per-void path is covered by tests/py_dtfelib_test.py.
    cell = config.CELL_SIZE
    cuts = ellipsoid_cuts_mpc()
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
    np.savez_compressed(path, **arrays)


class SnapshotProducts:
    """Per-snapshot derived products (delta, hessian, phi, voids, ...) with disk caching.

    Backed by dtfelib.FieldSet, so it works for BOTH estimators: pass method='ps' or
    method='dtfe' (default 'auto' = whichever exists in the snapshot directory). Density
    enters all derived products in MEAN units (rho/rho_bar) for either method, so every
    downstream quantity is method-independent by construction. Caches are namespaced by
    (sim, [prefix,] method, snapshot) -- old caches keyed by snapshot alone are simply not
    reused, and an alternate OUTPUT_PREFIX set never shares caches with the primary one.
    """

    def __init__(self, snapshot, redshift=None, sim=None, method=None, prefix=None, data_root=None):
        self.snapshot = f"{int(snapshot):03d}"   # accepts 99, '99' or '099'
        self.sim = sim or DEFAULT_SIM
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

    def _field_shape(self):
        n = self.fs.grid_n
        return (n, n, n)

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
        Every product is a function of that grid, and the cache key holds only the parameters: the 2026-10-03/04
        rerun rewrote the grids under July's TNG300 caches (5% of the minima moved) and nothing noticed. A cache
        carries the grid's (size, mtime_ns) since 2026-10-07 and must match it exactly (a copied cache, a future
        grid mtime, a grid rewritten mid-build: none fools an equality); an older cache without the stamp is
        judged by modification time (older than the grid = stale). A stale cache is rebuilt, and says so; None
        then (and when there is no cache)."""
        p = self._cpath(name)
        if not p.exists():
            return None
        stamp = self._grid_stamp()
        if stamp.size == 0:
            return p                            # no grid to compare with (a cache-only test double): use it
        with np.load(p) as z:
            have = z["grid_stamp"] if "grid_stamp" in z.files else None
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
        cell of the DTFE_SIM simulation at build time, which a script's --sim does not change: the
        TNG100-3-Dark catalogues of July were built under TNG50's frame (positions up to 51.6 Mpc in a
        110.7 Mpc box) and read as hundreds of 'resolved' voids where the right frame leaves a handful --
        or when the cuts (ellipsoid_cuts_mpc) changed since (they are not in the cache's parameter hash, so a
        retuned cut used to need the caches deleted). Says so. (2026-10-05)"""
        snapshot_frame(self)                    # the header's frame must be config's: refuse before any cache is touched
        p = self._cpath('voids')
        if self._cached('voids') is not None:
            with np.load(p) as z:
                cat = {k: z[k] for k in z.files}
            if 'r_v_cells' not in cat and self._grid_path() is not None:
                # R_v needs the smoothed field, which a refit does not have: rebuild (once per cache, ~18 s at 512^3)
                print(f"    [pipeline] cached void catalogue {p.name} has no measured radius R_v (2026-10-07): rebuilding it")
                cat = build_void_catalog(self.delta_smoothed, self.hessian)
                self._save(p, cat)
                return cat
            frame_ok, cuts_ok = catalog_frame_ok(cat), catalog_cuts_ok(cat)
            if frame_ok and cuts_ok:
                return cat
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
            return cat
        cat = build_void_catalog(self.delta_smoothed, self.hessian)
        self._save(p, cat)
        return cat

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
        mask = (self.phi == maximum_filter(self.phi, size=config.FOOTPRINT_SIZE,
                                           mode='wrap'))
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
    # one file PER SIMULATION (2026-10-05 review): cache_dir() is hashed on the parameters only, and
    # compute runs for two simulations used to max-merge into one global file, so a TNG300 series got
    # the TNG50 amplitudes (sigma is in cells: a different Mpc smoothing, a different scale)
    return cache_dir() / f'global_limits_{sim or DEFAULT_SIM}.json'


def save_global_limits(limits, sim=None):
    with open(_limits_path(sim), 'w') as f:
        json.dump(limits, f, indent=2)


def _snapshot_limits_path(sim=None):
    return cache_dir() / f'snapshot_limits_{sim or DEFAULT_SIM}.json'


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
    over every stored snapshot of the canonical series. Before 2026-10-05 compute wrote the maximum over
    the snapshots of that one run: a run on a subset, or on an unknown id, shrank or emptied the
    cross-epoch limits every later plot used. Returns (limits, covered, missing): the limits written
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
