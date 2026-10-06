"""Density power spectra (survey item 25, 2026-10-05): P(k) of a grid, and the linear-theory P(k) of the
simulation's cosmology to compare it with.

    from dtfelib import spectra
    ps = spectra.power_spectrum(fs.density(units="mean") - 1.0, fs.meta.box_mpc)   # k [1/Mpc], P [Mpc^3]
    pl = spectra.linear_pk(ps["k"], z=fs.meta.redshift)                            # Mpc^3, the same k

UNITS: h-free, like every length in this code base -- k in 1/Mpc (comoving), P(k) in Mpc^3. The transfer
function's own variables (Eisenstein & Hu 1998 want k in h/Mpc) and sigma_8's radius (8/h Mpc = 11.8 Mpc
h-free) are converted INSIDE those functions only.

THE ESTIMATOR. delta sampled on N^3 cells of a box L: with numpy's unnormalised transform
delta_k = sum_x delta(x) e^{-i k x}, the estimate is P(k) = V / N^6 |delta_k|^2 (V = L^3), averaged over
the modes of each |k| bin; a mode of the rfft half-cube counts twice (its conjugate is not stored) except
on the self-conjugate planes k_z = 0 and k_z = N/2, which count once; k = 0 (the mean) is left out. The
bins are the fundamental mode's multiples (k_f = 2 pi / L, edges at (i + 1/2) k_f) up to the Nyquist
k_Ny = pi N / L, or 'nbins' logarithmic ones. 'modes' per bin is the Hermitian-weighted count: the sample
variance of a bin's P is about P sqrt(2 / modes). THERE IS NO ASSIGNMENT WINDOW TO DECONVOLVE: a DTFE or
PS-DTFE grid is the field averaged over cells (or point-sampled), not a CIC mass assignment, and the cell
average damps power towards k_Ny (a sinc^2 per axis); the grid's sampling aliases the power above about
k_Ny / 2 back down, so read the comparison with linear theory below that.

THE LINEAR SPECTRUM. P_lin(k, z) = A k^n_s T(k)^2 D(z)^2 with T the Eisenstein & Hu (1998) no-wiggle
transfer function (their eqs 26-31: the baryon suppression through the sound horizon and alpha_Gamma, no
acoustic oscillations -- the TNG box resolves k down to 2 pi / 110 Mpc, the wiggles sit below), A fixed by
config.COSMOLOGY's sigma_8 at z = 0 (the top-hat integral at R = 8/h Mpc), and D the exact LambdaCDM growth
factor D(a) proportional to H(a) integral_0^a da' / (a' H(a'))^3, D(1) = 1 (growth_factor); growth_rate is
its logarithmic derivative, f = dlnD/dlna (within 0.5% of Omega_m(z)^0.55).
"""

from __future__ import annotations

import numpy as np

import config

try:
    import scipy.fft as _sfft

    def _rfftn(a):
        return _sfft.rfftn(a, workers=-1)
except ImportError:                                     # pragma: no cover
    def _rfftn(a):
        return np.fft.rfftn(a)

_COSMO = dict(config.COSMOLOGY)
_trapz = getattr(np, "trapezoid", None) or np.trapz          # NumPy 1.26 (Ubuntu's python3-numpy) has only trapz
_H = config.HUBBLE_H
T_CMB = 2.7255


def power_spectrum(delta, box_mpc: float, nbins: int | None = None, kmax: float | None = None) -> dict:
    """P(k) of the cubic grid 'delta' (N,N,N) in a periodic box of 'box_mpc': {'k': the mean |k| of each
    bin [1/Mpc], 'P': [Mpc^3], 'modes': the Hermitian-weighted mode count, 'edges', 'k_f', 'k_nyquist',
    'box_mpc', 'n'}. Linear bins of k_f up to 'kmax' (default k_Ny) or, with 'nbins', that many
    logarithmic bins from k_f to kmax. See the module docstring."""
    d = np.asarray(delta)
    n = d.shape[0]
    if d.ndim != 3 or d.shape != (n, n, n):
        raise ValueError(f"power_spectrum wants a cubic (N,N,N) grid, got {d.shape}")
    box = float(box_mpc)
    vol = box ** 3
    k_f = 2.0 * np.pi / box
    k_ny = np.pi * n / box
    kmax = float(kmax) if kmax is not None else k_ny
    dk = _rfftn(d.astype(np.float32, copy=False))
    p3 = (dk.real.astype(np.float64) ** 2 + dk.imag.astype(np.float64) ** 2) * (vol / float(n) ** 6)
    del dk
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=box / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=box / n)
    kmag = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
    w = np.full(kz.shape, 2.0)                      # a stored half-cube mode stands for its conjugate too ...
    w[0] = 1.0                                      # ... except on the self-conjugate planes
    if n % 2 == 0:
        w[-1] = 1.0
    weights = np.broadcast_to(w[None, None, :], kmag.shape).copy()
    weights[0, 0, 0] = 0.0                          # k = 0: the mean, not a mode of the fluctuations
    if nbins is None:
        edges = k_f * (np.arange(0, int(np.ceil(kmax / k_f)) + 1) + 0.5)
    else:
        edges = np.geomspace(k_f * 0.5, kmax, int(nbins) + 1)
    which = np.digitize(kmag.ravel(), edges) - 1
    ok = (which >= 0) & (which < len(edges) - 1)
    wf, pf, kf = weights.ravel()[ok], p3.ravel()[ok], kmag.ravel()[ok]
    idx = which[ok]
    modes = np.bincount(idx, weights=wf, minlength=len(edges) - 1)
    psum = np.bincount(idx, weights=wf * pf, minlength=len(edges) - 1)
    ksum = np.bincount(idx, weights=wf * kf, minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(modes > 0, psum / modes, np.nan)
        k = np.where(modes > 0, ksum / modes, np.nan)
    return {"k": k, "P": P, "modes": modes, "edges": edges, "k_f": k_f, "k_nyquist": k_ny,
            "box_mpc": box, "n": int(n)}


# ---------------------------------------------------------------- linear theory

def _e_of_a(a: float, om: float, ol: float) -> float:
    """H(a) / H0 for flat-ish LambdaCDM with the curvature term of 1 - om - ol."""
    return np.sqrt(om / a ** 3 + ol + (1.0 - om - ol) / a ** 2)


def growth_factor(z, cosmo: dict | None = None) -> np.ndarray:
    """D(z), the linear growth factor normalised to D(z=0) = 1: D(a) proportional to
    H(a) integral_0^a da' / (a' H(a'))^3 (exact for LambdaCDM with the cosmological constant)."""
    c = cosmo or _COSMO
    om, ol = c["Omega_m"], c["Omega_Lambda"]
    zz = np.atleast_1d(np.asarray(z, dtype=np.float64))

    def raw(a):
        x = np.linspace(1e-8, a, 4000)
        f = 1.0 / (x * _e_of_a(x, om, ol)) ** 3
        return _e_of_a(a, om, ol) * _trapz(f, x)
    d0 = raw(1.0)
    out = np.array([raw(1.0 / (1.0 + v)) / d0 for v in zz])
    return out if np.ndim(z) else out[0]


def growth_rate(z, cosmo: dict | None = None, eps: float = 1e-3) -> np.ndarray:
    """f(z) = dlnD / dlna by a centred difference of growth_factor in ln a."""
    zz = np.atleast_1d(np.asarray(z, dtype=np.float64))
    a = 1.0 / (1.0 + zz)
    ap, am = a * np.exp(eps), a * np.exp(-eps)
    dp, dm = growth_factor(1.0 / ap - 1.0, cosmo), growth_factor(1.0 / am - 1.0, cosmo)
    out = (np.log(dp) - np.log(dm)) / (2.0 * eps)
    return out if np.ndim(z) else out[0]


def transfer_eh98(k_mpc, cosmo: dict | None = None, h: float = _H, t_cmb: float = T_CMB) -> np.ndarray:
    """Eisenstein & Hu (1998) no-wiggle transfer function T(k), k in 1/Mpc (h-free). The sound horizon s
    (their eq 26, in Mpc) and alpha_Gamma (eq 31) carry the baryons' suppression into Gamma_eff (eq 30),
    whose k s is k in 1/Mpc times s in Mpc (Eisenstein's own TFnowiggles converts to 1/Mpc first; a k in
    h/Mpc there put the suppression 1/h too far in k -- review 2026-10-05); only q (eq 28) is written with
    k in h/Mpc. T from q (eq 29)."""
    c = cosmo or _COSMO
    om, ob = c["Omega_m"], c["Omega_b"]
    omh2, obh2 = om * h * h, ob * h * h
    theta = t_cmb / 2.7
    k = np.asarray(k_mpc, dtype=np.float64)                               # 1 / Mpc
    kh = k / h                                                            # h / Mpc: eq 28's q only
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * obh2 ** 0.75)   # Mpc
    alpha = 1.0 - 0.328 * np.log(431.0 * omh2) * ob / om + 0.38 * np.log(22.3 * omh2) * (ob / om) ** 2
    gamma_eff = om * h * (alpha + (1.0 - alpha) / (1.0 + (0.43 * k * s) ** 4))     # eq 30: k [1/Mpc] s [Mpc]
    q = kh * theta ** 2 / gamma_eff
    l0 = np.log(2.0 * np.e + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return l0 / (l0 + c0 * q * q)


def _tophat(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    nz = x > 1e-6
    out[nz] = 3.0 * (np.sin(x[nz]) - x[nz] * np.cos(x[nz])) / x[nz] ** 3
    return out


def sigma_r(pk_fn, r_mpc: float) -> float:
    """sigma(R) = sqrt( integral P(k) W^2(kR) k^2 dk / (2 pi^2) ) with a top-hat W, R in Mpc, from a
    callable P(k) [Mpc^3] of k [1/Mpc]; integrated in ln k over 1e-5 .. 1e3 / Mpc."""
    k = np.geomspace(1e-5, 1e3, 20000)
    integrand = pk_fn(k) * _tophat(k * r_mpc) ** 2 * k ** 3 / (2.0 * np.pi ** 2)
    return float(np.sqrt(_trapz(integrand, np.log(k))))


def linear_amplitude(cosmo: dict | None = None, h: float = _H) -> float:
    """A in P_lin(k, z = 0) = A k^n_s T(k)^2 [Mpc^3], fixed by sigma_8 at R = 8/h Mpc."""
    c = cosmo or _COSMO
    unnorm = lambda k: k ** c["n_s"] * transfer_eh98(k, c, h) ** 2       # noqa: E731
    return (c["sigma8"] / sigma_r(unnorm, 8.0 / h)) ** 2


def linear_pk(k_mpc, z: float = 0.0, cosmo: dict | None = None, h: float = _H) -> np.ndarray:
    """The linear-theory P(k) [Mpc^3] at redshift z for k [1/Mpc]: A k^n_s T(k)^2 D(z)^2."""
    c = cosmo or _COSMO
    k = np.asarray(k_mpc, dtype=np.float64)
    return linear_amplitude(c, h) * k ** c["n_s"] * transfer_eh98(k, c, h) ** 2 * float(growth_factor(z, c)) ** 2
