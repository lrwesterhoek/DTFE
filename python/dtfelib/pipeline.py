
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
    cat['cell_mpc'] = np.float64(config.CELL_SIZE)      # the FRAME the positions and semi-axes are in (2026-10-05)
    cat['cuts_mpc'] = _cuts_array()                     # the ELLIPSOID_CUTS the well_resolved flag was taken with
    cat['deep_threshold'] = np.float64(config.DEEP_VOID_THRESHOLD)   # the threshold 'deep' was taken at
    return cat


def _cuts_array():
    c = config.ELLIPSOID_CUTS
    return np.array([c['min_axis_mpc'], c['max_axis_mpc'], c['max_axis_ratio']], dtype=np.float64)


def catalog_cuts_ok(cat):
    """Whether a catalogue's well_resolved and deep flags were taken with today's config.ELLIPSOID_CUTS and
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
    under another DTFE_SIM's cell, or before a change of config.ELLIPSOID_CUTS, is put right without the
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
    ELLIPSOID_CUTS resolution gate) and by the void tracker (which reports sub-resolution
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
    # keep them in lockstep) + the ELLIPSOID_CUTS resolution gate, vectorized over voids.
    # Bit-equivalence with the per-void path is covered by tests/py_dtfelib_test.py.
    cell = config.CELL_SIZE
    cuts = config.ELLIPSOID_CUTS
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

    def voids(self):
        """The void catalogue. From the cache, whose minima and eigenvalues (cell units) never go stale; the
        ellipsoid fits on top of them are re-derived (refit_catalog, milliseconds, no field) and the cache
        rewritten when their FRAME is not config.CELL_SIZE -- the positions and semi-axes scale with the
        cell of the DTFE_SIM simulation at build time, which a script's --sim does not change: the
        TNG100-3-Dark catalogues of July were built under TNG50's frame (positions up to 51.6 Mpc in a
        110.7 Mpc box) and read as hundreds of 'resolved' voids where the right frame leaves a handful --
        or when config.ELLIPSOID_CUTS changed since (the cuts are not in the cache's parameter hash, so a
        retuned cut used to need the caches deleted). Says so. (2026-10-05)"""
        snapshot_frame(self)                    # the header's frame must be config's: refuse before any cache is touched
        p = self._cpath('voids')
        if p.exists():
            with np.load(p) as z:
                cat = {k: z[k] for k in z.files}
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
            _save_npz(p, **cat)
            return cat
        cat = build_void_catalog(self.delta_smoothed, self.hessian)
        _save_npz(p, **cat)
        return cat

    def critical_points(self):
        p = self._cpath('critical')
        if p.exists():
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
            _save_npz(p, **flat)
        out = {'field_mean': float(flat['field_mean']),
               'field_std': float(flat['field_std'])}
        for t in ('minima', 'maxima', 'saddle1', 'saddle2'):
            out[t] = {k: flat[f"{t}_{k}"] for k in ('coords', 'values', 'eigenvalues')}
        return out

    def eigen_slices(self):
        p = self._cpath('eigslices')
        if p.exists():
            with np.load(p) as z:
                return {d: {k: z[f"{d}_{k}"] for k in
                            ('lambda1', 'lambda2', 'lambda3', 'trace', 'det')}
                        for d in (0, 1, 2)}
        out, flat = {}, {}
        for d in (0, 1, 2):
            out[d] = hessian_slice(self.hessian, d)
            for k, v in out[d].items():
                flat[f"{d}_{k}"] = v
        _save_npz(p, **flat)
        return out

    def delta_slices(self):
        p = self._cpath('deltaslices')
        if p.exists():
            with np.load(p) as z:
                return {d: z[str(d)] for d in (0, 1, 2)}
        out = {d: dtfe.extract_2d_slice(self.delta_smoothed, d).astype(np.float32)
               for d in (0, 1, 2)}
        _save_npz(p, **{str(d): v for d, v in out.items()})
        return out

    def phi_slices(self):
        p = self._cpath('phislices')
        if p.exists():
            with np.load(p) as z:
                return {d: z[str(d)] for d in (0, 1, 2)}
        out = {d: dtfe.extract_2d_slice(self.phi, d).astype(np.float32)
               for d in (0, 1, 2)}
        _save_npz(p, **{str(d): v for d, v in out.items()})
        return out

    def phi_maxima(self):
        p = self._cpath('phimaxima')
        if p.exists():
            with np.load(p) as z:
                return {k: z[k] for k in z.files}
        mask = (self.phi == maximum_filter(self.phi, size=config.FOOTPRINT_SIZE,
                                           mode='wrap'))
        coords = np.argwhere(mask)
        cat = {'coords': coords, 'values': self.phi[mask]}
        _save_npz(p, **cat)
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
