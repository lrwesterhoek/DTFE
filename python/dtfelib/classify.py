"""Void-in-void versus void-in-cloud (2026-10-11): each distinct resolved void classified by the mean density contrast
ENCLOSED within ENV_FACTOR x R_v of its centre, from the raw density.

    from dtfelib import pipeline, classify
    p = pipeline.products("099", 0.0)
    env = classify.classify_voids(p)          # per void: Delta(<2 R_v), 'S' / 'R', R_v, depth, shape
    env["vic"].mean()                         # the void-in-cloud fraction

THE RULE (Ceccarelli, Paz, Lares, Padilla & Lambas 2013, MNRAS 434, 1435): Delta(<2 R_v) > 0 -- the compensation
ridge outweighs the void, the void sits in an overdense region and is expected to shrink or be crushed: S-type,
VOID-IN-CLOUD (the 'cloud-in-void' / 'void-in-cloud' language of Sheth & van de Weygaert 2004); Delta(<2 R_v) <= 0
-- underdense out to twice its radius, a void inside a larger void that keeps expanding: R-type, VOID-IN-VOID.
ENV_FACTOR (2.0) and VIC_THRESHOLD (0.0) are module constants: change them here, the scripts tag a non-default factor.

THE MEASUREMENT. Delta(<X R) = 3 / X^3 int_0^X (rho/rho_bar - 1) x^2 dx, x = r / R, from the shell means of the RAW
density rho/rho_bar (the catalogue's minima come from the smoothed delta, which would smear the ridge away) on
SHELLS shells out to exactly ENV_FACTOR R_v, sampled by profiles.profile_grids (Fibonacci directions, trilinear,
centred on the void's cell CENTRE, (i + 0.5) / N); the integral as in profiles.linear_v_r: the profile linear between
the shells on a fine grid, flat inside the first one. A void whose outer shell would reach beyond half the box is
dropped and counted (profile_grids' wrap rule: in a periodic box it would sample its own interior).

THE SAMPLE is the profiles': every well-resolved void with a measured R_v (catalog.r_v_mpc), only the DISTINCT ones
(catalog.distinct_mask) unless distinct=False. Nothing is cached: the density read dominates (0.5 GB at 512^3) and
the shell sampling takes seconds; the figure script stores the per-void table in its own npz.
"""

from __future__ import annotations

import numpy as np

from . import catalog
from .profiles import profile_grids

ENV_FACTOR = 2.0            # Delta(< ENV_FACTOR x R_v) decides the type (Ceccarelli et al. 2013)
VIC_THRESHOLD = 0.0         # Delta above it: void-in-cloud (S-type); at or below: void-in-void (R-type)
SHELLS = 40                 # shells between 0 and ENV_FACTOR R_v
DIRS = 300                  # Fibonacci directions per shell
TYPES = ("R", "S")          # void-in-void, void-in-cloud
TYPE_NAMES = {"R": "void-in-void", "S": "void-in-cloud"}


def shell_bins(factor: float = ENV_FACTOR, n: int = SHELLS) -> np.ndarray:
    """The shell radii in units of R: n evenly spaced, the last exactly 'factor'."""
    return np.linspace(float(factor) / n, float(factor), int(n))


def enclosed_contrast(bins, density_profile) -> np.ndarray:
    """Delta(<x) = 3 / x^3 int_0^x (rho - 1) x'^2 dx' at every shell x of 'bins', for one profile (nb,) or many
    (n, nb) of rho/rho_bar. The profile linear between shells (a fine-grid trapezoid: the coarse shells' own one
    overestimates x^2 at the first step) and flat inside the first shell -- profiles.linear_v_r's integral."""
    x = np.asarray(bins, dtype=np.float64)
    d = np.atleast_2d(np.asarray(density_profile, dtype=np.float64)) - 1.0
    xf = np.linspace(x[0], x[-1], max(4000, 50 * len(x)))
    out = np.empty_like(d)
    for i, row in enumerate(d):
        df = np.interp(xf, x, row)
        inc = 0.5 * (df[1:] * xf[1:] ** 2 + df[:-1] * xf[:-1] ** 2) * np.diff(xf)
        integral = np.concatenate([[0.0], np.cumsum(inc)])
        out[i] = 3.0 * (row[0] * x[0] ** 3 / 3.0 + np.interp(x, xf, integral)) / x ** 3
    return out if np.ndim(density_profile) > 1 else out[0]


def classify_grid(density, centers_frac, r_v_cells, factor: float = ENV_FACTOR, shells: int = SHELLS,
                  m: int = DIRS) -> dict:
    """Delta(< factor x R_v) and the type of every void given as (centre fraction (n, 3), R_v in cells (n,)) on the
    rho/rho_bar grid 'density': {'delta_enclosed': (n,) NaN where dropped, 'vic': (n,) bool, 'type': (n,) 'S'/'R'/'',
    'kept': (n,) bool, 'dropped': k, 'bins', 'profile': (n, shells) rho/rho_bar}. Dropped: no finite positive R_v,
    or the outer shell beyond half the box."""
    bins = shell_bins(factor, shells)
    r = np.asarray(r_v_cells, dtype=np.float64).reshape(-1)
    prof = profile_grids({"density": density}, centers_frac, r, bins=bins, m=m)
    n = len(r)
    delta = np.full(n, np.nan)
    profile = np.full((n, len(bins)), np.nan)
    kept = np.zeros(n, dtype=bool)
    idx = prof["index"]
    kept[idx] = True
    if idx.size:
        profile[idx] = prof["density"]
        delta[idx] = enclosed_contrast(bins, prof["density"])[:, -1]
    vic = kept & (delta > VIC_THRESHOLD)
    typ = np.where(~kept, "", np.where(vic, "S", "R"))
    return {"delta_enclosed": delta, "vic": vic, "type": typ, "kept": kept, "dropped": int((~kept).sum()),
            "bins": bins, "profile": profile, "factor": float(factor)}


def void_sample(cat, n_grid: int, distinct: bool = True, definition: str = "crossing") -> np.ndarray:
    """The catalogue rows classified: well resolved with a measured R_v ('definition': catalog.R_V_DEFINITIONS), the
    distinct ones unless distinct=False (profiles.void_profiles' sample)."""
    sel = np.asarray(cat["well_resolved"]).astype(bool) & np.isfinite(catalog.r_v_mpc(cat, definition))
    if distinct:
        sel = catalog.distinct_mask(cat, sel, int(n_grid))
    return np.nonzero(sel)[0]


def classify_catalog(cat, density, cell_mpc: float, factor: float = ENV_FACTOR, distinct: bool = True,
                     definition: str = "crossing") -> dict:
    """classify_grid on a catalogue's sample (void_sample) with the density grid it was built from: one row per void
    kept (the wrap rule drops the rest, counted in 'dropped') with 'index' (catalogue row), 'delta_enclosed', 'vic',
    'type', 'r_v_mpc', 'delta_c' (central smoothed delta, the depth), 'e', 'p' (BBKS), 'b_over_a', 'c_over_a',
    'r_eff_mpc', 'center_frac'; plus 'n_sample', 'dropped', 'factor', 'bins', 'profile', 'definition' (which R_v:
    'crossing', the catalogue's R_v, by default; a catalogue without the definition's key gives no void)."""
    n_grid = int(density.shape[0])
    rows = void_sample(cat, n_grid, distinct, definition)
    coords = np.asarray(cat["coords"])[rows]
    r_v = catalog.r_v_mpc(cat, definition)[rows]
    centers = (coords + 0.5) / n_grid
    g = classify_grid(density, centers, r_v / float(cell_mpc), factor=factor)
    k = g["kept"]
    bbks = np.asarray(cat["bbks_params"], dtype=np.float64).reshape(-1, 2)[rows]
    ratios = np.asarray(cat["axis_ratios"], dtype=np.float64).reshape(-1, 2)[rows]
    return {"index": rows[k], "delta_enclosed": g["delta_enclosed"][k], "vic": g["vic"][k], "type": g["type"][k],
            "r_v_mpc": r_v[k], "delta_c": np.asarray(cat["delta_values"], dtype=np.float64)[rows][k],
            "e": bbks[k, 0], "p": bbks[k, 1], "b_over_a": ratios[k, 0], "c_over_a": ratios[k, 1],
            "r_eff_mpc": catalog.r_eff_mpc(cat)[rows][k], "center_frac": centers[k], "profile": g["profile"][k],
            "bins": g["bins"], "n_sample": int(len(rows)), "dropped": g["dropped"], "factor": float(factor),
            "distinct": bool(distinct), "definition": str(definition)}


def classify_voids(products, factor: float = ENV_FACTOR, distinct: bool = True, definition: str = "crossing") -> dict:
    """classify_catalog on one snapshot (pipeline.SnapshotProducts): its catalogue (the frame checked by
    catalog.frame_of) and its raw density rho/rho_bar. Adds 'cell_mpc', 'box_mpc', 'n_grid', 'method', 'prefix'."""
    cell, box, n_grid, _src = catalog.frame_of(products)
    cat = products.voids()
    catalog.assert_frame(cat, cell)
    out = classify_catalog(cat, products.density_raw, cell, factor=factor, distinct=distinct, definition=definition)
    out.update(cell_mpc=float(cell), box_mpc=float(box), n_grid=int(n_grid), method=str(products.fs.method),
               prefix=str(products.fs.prefix))
    return out


def vic_fraction(vic, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(fraction, 16th, 84th percentile of n_boot bootstrap resamples) of a boolean array; NaNs when empty."""
    v = np.asarray(vic, dtype=bool)
    if not v.size:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, v.size, size=(int(n_boot), v.size))].mean(axis=1)
    return float(v.mean()), float(np.percentile(boot, 16)), float(np.percentile(boot, 84))
