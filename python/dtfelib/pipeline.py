
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import minimum_filter, maximum_filter, gaussian_filter

import config
from .io import FieldSet
from .cli import DATA_ROOT, DEFAULT_SIM
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
    return cat


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

    def __init__(self, snapshot, redshift=None, sim=None, method=None, prefix=None):
        self.snapshot = f"{int(snapshot):03d}"   # accepts 99, '99' or '099'
        self.sim = sim or DEFAULT_SIM
        self.method = method or "auto"
        self.prefix = prefix                     # alternate OUTPUT_PREFIX set, see FieldSet
        self._fs = None
        self._redshift = redshift
        self._ram = {}

    @property
    def fs(self):
        if self._fs is None:
            self._fs = FieldSet(DATA_ROOT / self.sim / f"snapdir_{self.snapshot}",
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
        p = self._cpath('voids')
        if p.exists():
            with np.load(p) as z:
                return {k: z[k] for k in z.files}
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


def products(snapshot, redshift=None, sim=None, method=None):
    return SnapshotProducts(snapshot, redshift, sim=sim, method=method)


def _limits_path():
    return cache_dir() / 'global_limits.json'


def save_global_limits(limits):
    with open(_limits_path(), 'w') as f:
        json.dump(limits, f, indent=2)


def load_global_limits():
    p = _limits_path()
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def series_vmax(field, fallback_data=None):
    fixed = config.FIELD_LIMITS.get(field)
    if fixed is not None:
        return float(fixed)
    gl = load_global_limits()
    if field in gl and gl[field] is not None:
        v = gl[field]
        return float(v if not isinstance(v, (list, tuple)) else v[1])
    from . import figures as style
    return style.robust_vmax(fallback_data) if fallback_data is not None else None
