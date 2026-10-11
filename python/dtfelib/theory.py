"""Linear-theory expectations for the void statistics (2026-10-11): the void size function and the shapes of the
minima of a Gaussian random field, to draw beside what the catalogues measure.

    from dtfelib import theory
    dn = theory.void_size_function(r_mpc, z=0.0)                  # dn/dlnR [Mpc^-3], the Vdn model
    q = theory.minima_shape_percentiles(nu=-delta_min / s0, gamma=theory.bbks_gamma(sigma_mpc)[0])

UNITS: h-free like the rest of the code base -- R in Mpc (comoving), k in 1/Mpc, n in Mpc^-3.

COSMOLOGY. config.COSMOLOGY, TNG's Planck 2015 (Planck Collaboration 2016, XIII): Omega_m 0.3089, Omega_L 0.6911,
Omega_b 0.0486, h 0.6774, sigma_8 0.8159, n_s 0.9667. The linear spectrum is spectra.linear_pk's: the Eisenstein &
Hu (1998) no-wiggle transfer function normalised to sigma_8, times D(z)^2, D the exact growth factor of flat LambdaCDM
(D(0) = 1). This module adds the vectorised top-hat sigma(R, z) the size function needs and the Gaussian-smoothed
spectral moments the BBKS shapes need; spectra stays the one place the spectrum itself is written.

THE SIZE FUNCTION. Sheth & van de Weygaert (2004, SvdW): a void is a region whose linear density, smoothed on R_L,
first crosses the shell-crossing threshold DELTA_V (linear -2.71, the -0.8 of a nonlinear spherical void) without
having crossed the collapse threshold DELTA_C = 1.686 on a larger scale (the void-in-cloud term). The first-crossing
distribution of the two-barrier random walk in ln(1/sigma) (their eq 4, as Jennings, Li & Hu 2013 eq 8 write it):

    f_lnsigma(sigma) = 2 sum_j exp(-(j pi x)^2 / 2) j pi x^2 sin(j pi D),  D = |dv| / (dc + |dv|),  x = D sigma / |dv|

summed to convergence (small x, where the sum converges slowly, takes its exact one-barrier limit
sqrt(2/pi) (|dv|/sigma) exp(-dv^2 / 2 sigma^2), within 1e-6 of the sum below SMALL_X; tests/py_population_test.py
checks both). A linear void of R_L has the nonlinear radius R = R_L (1 + dv_NL)^(-1/3) = 1.71 R_L. Two ways to count:

  * 'svdw' : dn/dlnR = f(sigma) / V(R_L) dln(1/sigma)/dlnR_L     (SvdW: number conserved through the expansion)
  * 'vdn'  : dn/dlnR = f(sigma) / V(R)   dln(1/sigma)/dlnR_L     (Jennings, Li & Hu 2013 eq 12: VOLUME conserved;
             the volume fraction in voids is then the walks' probability, at most dc / (dc + |dv|) = 0.38, where
             SvdW's grows to 1.71^3 times that and fills more than the box)

both at R = 1.71 R_L, sigma = sigma(R_L, z) with a top-hat window. Jennings et al. find Vdn the one that fits
simulated voids; it is the default. Each threshold is a module constant.

MINIMA SHAPES (Bardeen, Bond, Kaiser & Szalay 1986, BBKS). A minimum of the smoothed delta is a peak of -delta. With
lambda_1 >= lambda_2 >= lambda_3 > 0 the Hessian's eigenvalues there (fields.calculate_shape_parameters writes the
same e and p) and x = (lambda_1 + lambda_2 + lambda_3) / sigma_2, the shape distribution at fixed x is BBKS eq 7.6,

    P(e, p | x) = 3^2 5^(5/2) / sqrt(2 pi) x^8 / f(x) e (e^2 - p^2) (1 - 2p) ((1 + p)^2 - 9 e^2) exp(-5x^2 (3e^2 + p^2)/2)

on 0 <= e <= 1/4, -e <= p <= e and 1/4 <= e <= 1/2, 3e - 1 <= p <= e, with f(x) of eq A15, and the curvature of a
peak of height nu = -delta_min / sigma_0 is distributed as f(x) exp(-(x - gamma nu)^2 / (2 (1 - gamma^2))) (eq A14),
gamma = sigma_1^2 / (sigma_0 sigma_2) from the spectrum smoothed with the pipeline's Gaussian (window
exp(-k^2 sigma^2 / 2), pipeline.delta_smoothed). Over ALL minima (nu integrated out) x goes as f(x) exp(-x^2/2), so
their shapes are the same for every spectrum: the spectrum enters through nu only.
"""

from __future__ import annotations

import math

import numpy as np

import config
from . import spectra

DELTA_V_LINEAR = -2.71        # SvdW's linear shell-crossing threshold (Sheth & van de Weygaert 2004)
DELTA_V_NONLINEAR = -0.8      # the nonlinear density contrast of a void at shell crossing
DELTA_C = 1.686               # the linear collapse threshold (the void-in-cloud barrier)
SMALL_X = 0.1                 # below this x the two-barrier sum takes its one-barrier limit (error < 1e-6)

_trapz = getattr(np, "trapezoid", None) or np.trapz          # NumPy 1.26 (CI's) has only trapz
_K = np.geomspace(1e-5, 1e3, 6000)                            # 1/Mpc: the sigma(R) integration grid (ln k)


def cosmology() -> dict:
    """The cosmology every function here uses by default: config.COSMOLOGY with h."""
    return dict(config.COSMOLOGY, h=config.HUBBLE_H)


def growth(z, cosmo: dict | None = None):
    """D(z), D(0) = 1: spectra.growth_factor (flat LambdaCDM, exact integral)."""
    return spectra.growth_factor(z, cosmo)


def transfer(k_mpc, cosmo: dict | None = None):
    """T(k), k in 1/Mpc: spectra.transfer_eh98 (Eisenstein & Hu 1998, no wiggles)."""
    return spectra.transfer_eh98(k_mpc, cosmo)


_PK0 = {}


def _pk0(cosmo: dict | None = None) -> np.ndarray:
    """P_lin(k, z = 0) on _K, cached per cosmology (the sigma_8 normalisation is a 20000-point integral)."""
    c = cosmo or dict(config.COSMOLOGY)
    key = tuple(sorted(c.items()))
    if key not in _PK0:
        _PK0[key] = spectra.linear_pk(_K, 0.0, c)
    return _PK0[key]


def sigma_tophat(r_mpc, z: float = 0.0, cosmo: dict | None = None) -> np.ndarray:
    """sigma(R, z) of the linear density in top-hat spheres of radius R [Mpc]: sqrt(int P W^2 k^3 dlnk / 2 pi^2)
    times D(z). Vectorised over R; sigma_tophat(8 / h) = sigma_8."""
    r = np.atleast_1d(np.asarray(r_mpc, dtype=np.float64))
    integrand = _pk0(cosmo)[None, :] * spectra._tophat(np.outer(r, _K)) ** 2 * _K[None, :] ** 3 / (2.0 * np.pi ** 2)
    s = np.sqrt(_trapz(integrand, np.log(_K), axis=1)) * float(growth(z, cosmo))
    return s if np.ndim(r_mpc) else float(s[0])


def spectral_moments(sigma_g_mpc: float, z: float = 0.0, cosmo: dict | None = None) -> tuple:
    """(sigma_0, sigma_1, sigma_2) of the linear field smoothed with a Gaussian of 'sigma_g_mpc' (the pipeline's
    smoothing length): sigma_j^2 = int k^(2j) P(k) exp(-k^2 R^2) k^3 dlnk / 2 pi^2 (BBKS eq 4.6a)."""
    base = _pk0(cosmo) * np.exp(-(_K * float(sigma_g_mpc)) ** 2) * _K ** 3 / (2.0 * np.pi ** 2)
    d = float(growth(z, cosmo))
    return tuple(float(np.sqrt(_trapz(base * _K ** (2 * j), np.log(_K)))) * d for j in (0, 1, 2))


def bbks_gamma(sigma_g_mpc: float, cosmo: dict | None = None) -> tuple:
    """(gamma, R_*) of the Gaussian-smoothed linear spectrum: gamma = sigma_1^2 / (sigma_0 sigma_2) (the width of
    the spectrum, 0 < gamma < 1; independent of z in linear theory), R_* = sqrt(3) sigma_1 / sigma_2 [Mpc]."""
    s0, s1, s2 = spectral_moments(sigma_g_mpc, 0.0, cosmo)
    return s1 * s1 / (s0 * s2), math.sqrt(3.0) * s1 / s2


# ---------------------------------------------------------------- the void size function

def svdw_multiplicity(sigma, delta_v: float = DELTA_V_LINEAR, delta_c: float = DELTA_C) -> np.ndarray:
    """f_lnsigma(sigma): the two-barrier first-crossing distribution in ln(1/sigma) (module docstring). delta_c
    = inf (or 0 < delta_c not finite) is the one-barrier limit."""
    s = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    dv = abs(float(delta_v))
    one = np.sqrt(2.0 / np.pi) * (dv / s) * np.exp(-0.5 * (dv / s) ** 2)
    if not np.isfinite(delta_c):
        return one if np.ndim(sigma) else float(one[0])
    dd = dv / (float(delta_c) + dv)
    x = dd * s / dv
    out = one.copy()
    big = x >= SMALL_X
    if big.any():
        xb = x[big]
        jmax = int(np.ceil(12.0 / (np.pi * xb.min()))) + 2           # the last term below exp(-72)
        j = np.arange(1, jmax + 1, dtype=np.float64)[None, :]
        terms = np.exp(-0.5 * (j * np.pi * xb[:, None]) ** 2) * j * np.pi * xb[:, None] ** 2 * np.sin(j * np.pi * dd)
        out[big] = 2.0 * terms.sum(axis=1)
    out = np.clip(out, 0.0, None)                                    # the sum's last-digit noise at large x
    return out if np.ndim(sigma) else float(out[0])


def linear_radius_ratio(delta_v_nl: float = DELTA_V_NONLINEAR) -> float:
    """R / R_L of a void at shell crossing: (1 + delta_v_NL)^(-1/3) (1.71 for -0.8)."""
    return (1.0 + float(delta_v_nl)) ** (-1.0 / 3.0)


def void_size_function(r_mpc, z: float = 0.0, model: str = "vdn", delta_v: float = DELTA_V_LINEAR,
                       delta_c: float = DELTA_C, delta_v_nl: float = DELTA_V_NONLINEAR,
                       cosmo: dict | None = None) -> np.ndarray:
    """dn/dlnR [Mpc^-3] of voids of nonlinear radius R [Mpc] at redshift z: 'vdn' (Jennings, Li & Hu 2013, the
    default) or 'svdw' (Sheth & van de Weygaert 2004); see the module docstring."""
    if model not in ("vdn", "svdw"):
        raise ValueError(f"model must be 'vdn' or 'svdw', got {model!r}")
    r = np.atleast_1d(np.asarray(r_mpc, dtype=np.float64))
    r_l = r / linear_radius_ratio(delta_v_nl)
    eps = 1e-3                                                       # dln(1/sigma)/dlnR_L, centred in ln R
    s = sigma_tophat(r_l, z, cosmo)
    sp, sm = sigma_tophat(r_l * math.exp(eps), z, cosmo), sigma_tophat(r_l * math.exp(-eps), z, cosmo)
    dlninv = -(np.log(sp) - np.log(sm)) / (2.0 * eps)
    vol = 4.0 / 3.0 * np.pi * (r if model == "vdn" else r_l) ** 3
    out = svdw_multiplicity(s, delta_v, delta_c) / vol * dlninv
    return out if np.ndim(r_mpc) else float(out[0])


def void_cumulative(r_mpc, z: float = 0.0, r_max_mpc: float | None = None, **kw) -> np.ndarray:
    """n(>R) [Mpc^-3]: dn/dlnR integrated in ln R from each R to r_max_mpc (default 100 R_max(r_mpc), where the
    abundance is exp-small)."""
    r = np.atleast_1d(np.asarray(r_mpc, dtype=np.float64))
    top = float(r_max_mpc) if r_max_mpc else 100.0 * float(r.max())
    grid = np.geomspace(float(r.min()), top, 1200)
    dn = void_size_function(grid, z, **kw)
    lg = np.log(grid)
    seg = 0.5 * (dn[1:] + dn[:-1]) * np.diff(lg)
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])       # n(> grid[i])
    out = np.interp(np.log(r), lg, tail)
    return out if np.ndim(r_mpc) else float(out[0])


# ---------------------------------------------------------------- BBKS minima shapes

def bbks_f(x) -> np.ndarray:
    """BBKS eq A15: the curvature factor f(x) of the peak number density (f -> x^3 - 3x at large x)."""
    x = np.asarray(x, dtype=np.float64)
    a = np.sqrt(2.5)
    return ((x ** 3 - 3.0 * x) / 2.0 * (_erf(a * x) + _erf(a * x / 2.0))
            + np.sqrt(2.0 / (5.0 * np.pi)) * ((31.0 * x * x / 4.0 + 1.6) * np.exp(-5.0 * x * x / 8.0)
                                              + (x * x / 2.0 - 1.6) * np.exp(-2.5 * x * x)))


def _erf(x):
    from scipy.special import erf
    return erf(x)


def _ep_grid(n: int = 241):
    """The (e, p) grid of the shape pdfs (cell centres; e in (0, 1/2), p in (-1/2, 1/2)) and BBKS's eigenvalue
    weight e (e^2 - p^2) (1 - 2p) ((1 + p)^2 - 9e^2) on its domain, 0 outside it."""
    e = (np.arange(n) + 0.5) * (0.5 / n)
    p = -0.5 + (np.arange(2 * n) + 0.5) * (0.5 / n)
    E, P = np.meshgrid(e, p, indexing="ij")
    dom = (P <= E) & (np.where(E <= 0.25, P >= -E, P >= 3.0 * E - 1.0))
    w = np.where(dom, E * (E * E - P * P) * (1.0 - 2.0 * P) * ((1.0 + P) ** 2 - 9.0 * E * E), 0.0)
    return e, p, np.clip(w, 0.0, None), (0.5 / n) ** 2


def bbks_ep_pdf(x: float, n: int = 241):
    """P(e, p | x) of BBKS eq 7.6 on the _ep_grid (e, p, pdf), the analytic normalisation (3^2 5^(5/2) /
    sqrt(2 pi) x^8 / f(x)): its grid integral is 1 to the grid's accuracy (the test of f)."""
    e, p, w, _da = _ep_grid(n)
    E, P = np.meshgrid(e, p, indexing="ij")
    pdf = 9.0 * 5.0 ** 2.5 / math.sqrt(2.0 * math.pi) * x ** 8 / float(bbks_f(x)) * w \
        * np.exp(-2.5 * x * x * (3.0 * E * E + P * P))
    return e, p, pdf


_X = np.linspace(0.02, 8.0, 400)        # the curvature grid of the mixtures
_MARG = {}


def _marginals(n: int = 241):
    """Per x of _X: the e- and p-marginals of P(e, p | x), each normalised on its grid (cached)."""
    if n not in _MARG:
        e, p, w, da = _ep_grid(n)
        E2, P2 = np.meshgrid(e, p, indexing="ij")
        q = 3.0 * E2 * E2 + P2 * P2
        me, mp = np.zeros((_X.size, e.size)), np.zeros((_X.size, p.size))
        for i, x in enumerate(_X):
            j = w * np.exp(-2.5 * x * x * q)
            tot = j.sum()
            if tot > 0:
                me[i], mp[i] = j.sum(axis=1) / tot, j.sum(axis=0) / tot
        _MARG[n] = (e, p, me, mp)
    return _MARG[n]


def peak_curvature_pdf(nu=None, gamma: float = 0.5) -> np.ndarray:
    """The curvature x of peaks on _X, normalised: at height nu, f(x) exp(-(x - gamma nu)^2 / 2(1 - gamma^2))
    (BBKS eq A14); nu None: all peaks, f(x) exp(-x^2 / 2). An array of nu gives their MIXTURE (each peak one
    weight), the sample's own distribution of heights."""
    f = np.clip(bbks_f(_X), 0.0, None)
    if nu is None:
        w = f * np.exp(-0.5 * _X ** 2)
    else:
        nus = np.atleast_1d(np.asarray(nu, dtype=np.float64))
        nus = nus[np.isfinite(nus)]
        if not nus.size:
            return np.full(_X.size, np.nan)
        g = float(gamma)
        rows = f[None, :] * np.exp(-0.5 * (_X[None, :] - g * nus[:, None]) ** 2 / (1.0 - g * g))
        rows /= np.maximum(rows.sum(axis=1, keepdims=True), 1e-300)
        w = rows.mean(axis=0)
    return w / w.sum()


def minima_shape_pdf(nu=None, gamma: float = 0.5) -> dict:
    """The pdfs of e and of p of the minima of a Gaussian random field (as minima_shape_percentiles): {'e', 'pdf_e',
    'p', 'pdf_p'}, each pdf a density on its grid (integral 1)."""
    e, p, me, mp = _marginals()
    wx = peak_curvature_pdf(nu, gamma)
    pe, pp = wx @ me, wx @ mp
    return {"e": e, "pdf_e": pe / (e[1] - e[0]), "p": p, "pdf_p": pp / (p[1] - p[0])}


def minima_shape_percentiles(nu=None, gamma: float = 0.5, pct=(16, 50, 84)) -> dict:
    """The percentiles 'pct' of the BBKS ellipticity e and prolateness p of the minima of a Gaussian random field:
    {'e': array, 'p': array, 'e_mean', 'p_mean'}. nu (= -delta_min / sigma_0 per minimum, an array) conditions on the
    sample's heights with the spectrum's gamma; None takes all minima (spectrum-independent, the module docstring)."""
    e, p, me, mp = _marginals()
    wx = peak_curvature_pdf(nu, gamma)
    if not np.isfinite(wx).all():
        nan = np.full(len(pct), np.nan)
        return {"e": nan, "p": nan, "e_mean": np.nan, "p_mean": np.nan}
    pe, pp = wx @ me, wx @ mp

    def quantiles(grid, pdf):
        step = grid[1] - grid[0]
        cdf = np.concatenate([[0.0], np.cumsum(pdf)])
        edges = np.concatenate([[grid[0] - step / 2], grid + step / 2])
        return np.interp(np.asarray(pct, float) / 100.0, cdf / cdf[-1], edges)
    return {"e": quantiles(e, pe), "p": quantiles(p, pp),
            "e_mean": float((e * pe).sum() / pe.sum()), "p_mean": float((p * pp).sum() / pp.sum())}
