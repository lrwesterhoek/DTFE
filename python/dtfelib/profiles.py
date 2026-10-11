"""Void-centred stacked profiles in r / R (survey item 8, 2026-10-05): the density, the radial
velocity and the single-stream fraction around the catalogue's voids, averaged over spherical shells,
stacked in bins of R with bootstrap bands. R is the MEASURED void radius R_v by default (2026-10-07:
pipeline.void_radius_cells, where the spherically averaged smoothed density contrast first rises to 0), or the
ellipsoid's R_eff (radius='r_eff'; with distinct=False and the old bins the profiles before 2026-10-07): R_eff is a
curvature length, ~1.75x R_v
(median R_v/R_eff 0.53-0.65 in TNG100 and TNG300 at z = 0, 1, 5), and in the R_eff stacks the density reached
the mean, v_r turned to infall and the single-stream fraction bottomed out all at 0.5-0.7 R_eff -- at R_v.
In the R_v stacks the raw density crossing the mean at 1.0 R_v is the definition of R_v (the first up-crossing of
the SMOOTHED profile), a check of the smoothing against the raw density, not a finding (2026-10-11); the infall
radius, the compensation ridge and the single-stream fraction are independent of it.

    from dtfelib import pipeline, profiles
    p = pipeline.products("099", 0.0)
    prof = profiles.void_profiles(p)                           # per void: rho/rho_bar, v_r, single-stream fraction
    st = profiles.stack(prof, edges_mpc=(0, 2, 4, 8))          # per R_eff bin: mean, 16-84% bootstrap band, count

WHAT IS SAMPLED. Every well-resolved void of the catalogue (its ellipsoid fit within config.ELLIPSOID_CUT_RULES;
sample='deep' narrows to a central delta below config.DEEP_VOID_THRESHOLD, the tracking sample) with a radius,
by default only the DISTINCT ones of that sample (catalog.distinct_mask: deepest first, none whose centre lies
inside the sphere of min(R_v, R_eff) of a deeper resolved void, its R_eff when it has no R_v -- neighbouring minima
of one void would be stacked twice), is taken as a sphere of
radius R (R_v from catalog.r_v_mpc, or R_eff = (a b c)^(1/3) from catalog.r_eff_mpc) around its catalogue cell
-- the cell's CENTRE, (i + 0.5) / N in environment.sample_grid's convention; the catalogue's positions_mpc
are the cells' corners i x cell, half a cell apart, far below any R_eff. On each shell r = x R_eff (x in
'bins'), M points of a Fibonacci sphere sample
  * the RAW density rho/rho_bar (fs.density(units='mean')): the catalogue's minima come from the delta
    smoothed with config.SMOOTHING_SIGMA_CELLS, which would smear the compensation ridge away;
  * the velocity (peculiar km/s; FieldSet.load applies the sqrt(a) factor, so the grid is read into RAM:
    at 1024^3 that is 12.9 GB) as v_r = v . n, POSITIVE = OUTFLOW. The void's bulk velocity is a dipole
    over the sphere and cancels in the shell mean, so nothing is subtracted;
  * the single-stream mask (io.single_stream_mask: |s - 1| <= STREAM_TOL and no hidden-streams bit --
    the rule FieldSet.velocity_single_stream and webstreams share), by nearest cell, whose shell mean is
    the single-stream FRACTION of the shell (phase-space grids only: None under the standard DTFE),
by trilinear interpolation (the mask: nearest cell) through sample_grid, vectorised per shell over every
void. A void whose outermost shell (max(bins) R_eff) would reach beyond half the box is dropped and
counted ('dropped'): in a periodic box it would sample its own interior through the wrap.

STACKING. stack() bins the voids by R (fixed edges in Mpc, comparable across snapshots), and gives
per bin the mean profile over its voids, the 16th-84th percentile band of 'n_boot' bootstrap resamples of
the voids (NaN with fewer than two voids), and the count; a bin below 'min_count' voids is left NaN.

THE LINEAR-THEORY OVERLAY (optional, linear_v_r): from the stacked density itself, the mean overdensity
within r, Delta(<r) = 3 / x^3 * integral_0^x (rho/rho_bar - 1) x'^2 dx' (trapezoid over the shells, the
profile taken flat inside the first shell), gives v_r(r) = -(1/3) f(z) H(z) a r Delta(<r) in km/s with r
comoving (fields.get_cosmology_params: f = Omega_m(z)^0.55, H in km/s/Mpc). What the measured v_r panel
would be if the voids expanded linearly: the comparison is the point of the figure.

THE INTERIOR SINGLE-STREAM FRACTION (2026-10-11, interior_single_fraction): per void, the fraction of the
cells whose centres lie within x R of the void's cell centre (periodic) that are single-stream, for x in
INTERIOR_X = (0.5, 1.0): a VOLUME fraction counted cell by cell, not the x^2-weighted integral of the shells
(whose first shell sits at 0.1 R). void_profiles returns it as 'single_interior' (n_voids, len(INTERIOR_X)).

THE UNIVERSAL PROFILE (2026-10-11, hsw_delta / fit_hsw / fit_stack): Hamaus, Sutter & Wandelt (2014, PRL 112,
251302) delta(r) = delta_c (1 - (r/r_s)^alpha) / (1 + (r/R_v)^beta), with r in units of the stack's own R
(x = r/R, so R_v = 1 and r_s is r_s/R_v), fitted by scipy's curve_fit to a stack's mean rho/rho_bar - 1 with
the stack's scatter as the error of each shell (std / sqrt(n) over its voids, absolute_sigma). The shells of one
stack are correlated (the same voids), so chi2/dof is a guide, not a probability.

THE R_V DEFINITION (contract C3, 2026-10-11): radius='r_v' takes catalog.r_v_mpc(cat, definition) with
definition in catalog.R_V_DEFINITIONS ('crossing' = today's R_v, 'delta08', 'ridge'); a catalogue without that
key (all NaN) falls back to 'crossing' and says so.
"""

from __future__ import annotations

import numpy as np

from . import catalog
from .environment import sample_grid
from .io import single_stream_mask

DEFAULT_BINS = tuple(np.round(np.linspace(0.1, 3.0, 30), 4).tolist())   # r / R_eff of the shells
DEFAULT_EDGES = (0.0, 2.0, 4.0, 8.0, 16.0)                               # R_eff bins in Mpc (stack()'s default)
DEFAULT_EDGES_SIGMA = {"r_v": (0.0, 2.5, 3.5, 5.0),     # the scripts' default bins, in smoothing lengths (+ the wrap
                       "r_eff": (0.0, 5.0, 6.0, 7.0)}   # cap); R_v of distinct resolved voids: 16/50/84% ~ 2 / 3.3 / 7.5 sigma
RADII = ("r_v", "r_eff")
DEFAULT_M = 400                                                          # points per shell
INTERIOR_X = (0.5, 1.0)                                                  # interior_single_fraction's spheres, in R
# curve_fit's start and bounds for (delta_c, r_s/R_v, alpha, beta): HSW 2014's fits sit near (-0.8, 0.9, 2, 8)
HSW_P0 = (-0.8, 0.9, 2.0, 8.0)
HSW_BOUNDS = ((-1.0, 0.05, 0.1, 0.5), (0.0, 5.0, 20.0, 40.0))
HSW_PARAMS = ("delta_c", "r_s", "alpha", "beta")


def default_edges_mpc(sigma_mpc: float, box_mpc: float, radius: str = "r_v") -> tuple:
    """Bin edges in R for a stack in this simulation: DEFAULT_EDGES_SIGMA[radius] x sigma, closed at the wrap
    rule's cap L/6 (3 R <= L/2: a void beyond it is dropped anyway). Both radii scale with the smoothing (R_eff is a
    curvature length, 5-7 sigma for most voids passing the 10-sigma cut; R_v of the distinct ones 2-7.5 sigma), in
    EVERY simulation, while the old Mpc edges 0-2-4-8-16 put no TNG300 void in any bin (2026-10-07)."""
    if not float(sigma_mpc) > 0:
        raise ValueError(f"the smoothing length must be positive, got {sigma_mpc!r} (pipeline.smoothing_length_mpc)")
    if radius not in DEFAULT_EDGES_SIGMA:
        raise ValueError(f"radius must be one of {RADII}, got {radius!r}")
    cap = float(box_mpc) / 6.0
    edges = [k * float(sigma_mpc) for k in DEFAULT_EDGES_SIGMA[radius] if k * float(sigma_mpc) < cap]
    return tuple(edges + [cap])


def fibonacci_sphere(m: int) -> np.ndarray:
    """(m, 3) unit vectors spread evenly over the sphere (the Fibonacci lattice)."""
    i = np.arange(m, dtype=np.float64) + 0.5
    y = 1.0 - 2.0 * i / m
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([r * np.cos(phi), y, r * np.sin(phi)], axis=1)


def shell_means(grid: np.ndarray, centers_frac: np.ndarray, radii_cells: np.ndarray, dirs: np.ndarray,
                radial: bool = False, mode: str = "trilinear", chunk: int = 2_000_000) -> np.ndarray:
    """The mean over the directions 'dirs' (M, 3) of 'grid' sampled at centre + radius * dir, for every
    (centre, radius) pair: (n,) for a scalar grid; for a (N,N,N,3) grid with radial=True the mean of the
    radial component v . dir. 'centers_frac' (n, 3) in [0, 1), 'radii_cells' (n,) in cells. Chunked so
    that at most 'chunk' points are sampled at once."""
    n_grid = grid.shape[0]
    centers = np.asarray(centers_frac, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii_cells, dtype=np.float64).reshape(-1)
    m = dirs.shape[0]
    out = np.empty(len(radii), dtype=np.float64)
    step = max(1, chunk // m)
    for a in range(0, len(radii), step):
        b = min(a + step, len(radii))
        pts = centers[a:b, None, :] + (radii[a:b, None, None] * dirs[None, :, :]) / n_grid
        vals = sample_grid(grid, pts.reshape(-1, 3), mode=mode)
        if radial:
            vals = np.einsum("ij,ij->i", vals.reshape(-1, 3), np.broadcast_to(dirs, (b - a, m, 3)).reshape(-1, 3))
        out[a:b] = vals.reshape(b - a, m).mean(axis=1)
    return out


def profile_grids(grids: dict, centers_frac, r_eff_cells, bins=DEFAULT_BINS, m: int = DEFAULT_M) -> dict:
    """The shell profiles of the voids given as (centre fraction, R_eff in cells) on the grids
    {'density': (N,N,N) rho/rho_bar, 'velocity': (N,N,N,3) km/s or None, 'single': (N,N,N) bool or None}:
    {'bins': x, 'density' | 'v_r' | 'single': (n_voids, n_bins) or None, 'dropped': k, 'index': kept rows}.
    Voids whose outermost shell reaches beyond half the box are dropped (the periodic wrap)."""
    bins = np.asarray(bins, dtype=np.float64)
    centers = np.asarray(centers_frac, dtype=np.float64).reshape(-1, 3)
    r_eff = np.asarray(r_eff_cells, dtype=np.float64).reshape(-1)
    n_grid = grids["density"].shape[0]
    keep = np.isfinite(r_eff) & (r_eff > 0) & (bins.max() * r_eff <= 0.5 * n_grid)
    index = np.nonzero(keep)[0]
    centers, r_eff = centers[keep], r_eff[keep]
    dirs = fibonacci_sphere(m)
    out = {"bins": bins, "index": index, "dropped": int((~keep).sum()), "m": int(m),
           "density": np.full((len(r_eff), len(bins)), np.nan), "v_r": None, "single": None}
    vel, single = grids.get("velocity"), grids.get("single")
    if vel is not None:
        out["v_r"] = np.full((len(r_eff), len(bins)), np.nan)
    if single is not None:
        out["single"] = np.full((len(r_eff), len(bins)), np.nan)
    if not len(r_eff):
        return out
    for j, x in enumerate(bins):
        radii = x * r_eff
        out["density"][:, j] = shell_means(grids["density"], centers, radii, dirs)
        if vel is not None:
            out["v_r"][:, j] = shell_means(vel, centers, radii, dirs, radial=True)
        if single is not None:
            out["single"][:, j] = shell_means(single.astype(np.float32), centers, radii, dirs, mode="ngp")
    return out


def interior_single_fraction(single, coords, radii_cells, x=INTERIOR_X) -> np.ndarray:
    """(n_voids, len(x)): per void, the fraction of the cells whose CENTRES lie within x * R of the void's cell
    centre (periodic; 'coords' (n, 3) integer cells, 'radii_cells' (n,) R in cells) that are True in 'single'
    (N,N,N) bool -- the interior single-stream VOLUME fraction (2026-10-11). The void's centre is its cell's centre,
    so the offsets to the other cells' centres are whole cells and the centre cell itself is always counted.
    NaN for a void without a finite positive radius."""
    single = np.asarray(single)
    n_grid = single.shape[0]
    xs = np.asarray(x, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    radii = np.asarray(radii_cells, dtype=np.float64).reshape(-1)
    out = np.full((len(radii), len(xs)), np.nan)
    for i, (c, r) in enumerate(zip(coords, radii)):
        if not (np.isfinite(r) and r > 0):
            continue
        h = int(np.floor(xs.max() * r))
        h = min(h, (n_grid - 1) // 2)                       # never the same cell twice through the wrap
        d = np.arange(-h, h + 1)
        sub = single[np.ix_((c[0] + d) % n_grid, (c[1] + d) % n_grid, (c[2] + d) % n_grid)]
        d2 = d[:, None, None] ** 2 + d[None, :, None] ** 2 + d[None, None, :] ** 2
        for j, xv in enumerate(xs):
            sel = d2 <= (xv * r) ** 2 * (1.0 + 1e-12)       # a cell exactly on the sphere counts as inside
            out[i, j] = float(sub[sel].mean())
    return out


def r_v_by_definition(cat, definition: str = "crossing", say=print) -> tuple[np.ndarray, str]:
    """(R_v in Mpc per catalogue row, the definition used): catalog.r_v_mpc(cat, definition) (contract C3,
    2026-10-11). A catalogue without that definition's key (r_v_<definition>_cells: built before the definitions),
    or a catalog module without it, falls back to 'crossing', today's R_v, said through 'say'. A key that is there
    but NaN everywhere is an answer, not a gap: delta08 is undefined at z >= 3 in TNG100 (the enclosed contrast
    never reaches -0.8), and a fallback there would mix two definitions on one axis."""
    if definition == "crossing":
        return catalog.r_v_mpc(cat), "crossing"
    known = getattr(catalog, "R_V_DEFINITIONS", ("crossing",))
    r = None
    key = getattr(catalog, "R_V_KEYS", {}).get(definition, f"r_v_{definition}_cells")
    if definition in known and key in cat:
        try:
            r = np.asarray(catalog.r_v_mpc(cat, definition=definition), dtype=np.float64)
        except TypeError:                                   # a catalog.r_v_mpc from before the definitions
            r = None
    if r is None:
        say(f"  R_v definition {definition!r} not in this catalogue (rebuild it with the R_v definitions); "
            f"using 'crossing'")
        return catalog.r_v_mpc(cat), "crossing"
    return r, definition


def void_profiles(products, bins=DEFAULT_BINS, m: int = DEFAULT_M, sample: str = "resolved", radius: str = "r_v",
                  distinct: bool = True, definition: str = "crossing", interior=INTERIOR_X) -> dict:
    """profile_grids() on one snapshot's grids and catalogue (pipeline.SnapshotProducts): the voids of the
    'sample' ('resolved': well_resolved; 'deep': well_resolved and deep) with a finite 'radius' ('r_v', the
    measured catalog.r_v_mpc under 'definition' (contract C3), or 'r_eff', catalog.r_eff_mpc), only the distinct
    ones of them unless distinct=False, the frame checked by catalog.frame_of. Adds 'r_mpc' (the radius the shells
    are in), 'r_v_mpc', 'r_eff_mpc', 'radius', 'distinct', 'delta_c', 'sample', 'cell_mpc', 'n_grid', 'definition',
    and on phase-space grids 'single_interior' (interior_single_fraction at 'interior' x R) and 'single_box' (the
    whole box's single-stream fraction, the reference for it)."""
    cat = products.voids()
    cell, box, n_grid, _source = catalog.frame_of(products)
    catalog.assert_frame(cat, cell)
    if radius not in RADII:
        raise ValueError(f"radius must be one of {RADII}, got {radius!r}")
    resolved = np.asarray(cat["well_resolved"]).astype(bool)
    if sample == "deep":
        resolved &= np.asarray(cat["deep"]).astype(bool)
    elif sample != "resolved":
        raise ValueError(f"sample must be 'resolved' or 'deep', got {sample!r}")
    r_v_all, definition = r_v_by_definition(cat, definition) if radius == "r_v" else (catalog.r_v_mpc(cat), "crossing")
    r_eff_all = catalog.r_eff_mpc(cat)
    r_all = r_v_all if radius == "r_v" else r_eff_all
    n_sample = int(resolved.sum())
    resolved &= np.isfinite(r_all)
    n_no_radius = n_sample - int(resolved.sum())             # said, not silent: they are the larger, deeper ones
    n_with = int(resolved.sum())
    if distinct:
        resolved = catalog.distinct_mask(cat, resolved, n_grid)
    n_overlapping = n_with - int(resolved.sum())
    rows = np.nonzero(resolved)[0]
    coords = np.asarray(cat["coords"])[rows]
    r_mpc = r_all[rows]
    fs = products.fs
    grids = {"density": fs.density(units="mean"), "velocity": None, "single": None}
    if fs.has("velocity"):
        grids["velocity"] = fs.load("velocity", mode="ram")
    if fs.method == "ps" and fs.has("streams"):
        hidden = fs.load("hidden_streams") if fs.has("hidden_streams") else None
        grids["single"] = single_stream_mask(fs.load("streams"), hidden)
    out = profile_grids(grids, (coords + 0.5) / n_grid, r_mpc / cell, bins=bins, m=m)
    local = out["index"]                        # rows of the sample kept by profile_grids (the wrap rule)
    out["index"] = rows[local]                  # -> rows of the catalogue
    out["r_mpc"] = r_mpc[local]
    out["r_v_mpc"] = r_v_all[rows[local]]
    out["r_eff_mpc"] = r_eff_all[rows[local]]
    out["radius"], out["distinct"] = radius, bool(distinct)
    out["n_no_radius"], out["n_overlapping"] = n_no_radius, n_overlapping
    out["delta_c"] = np.asarray(cat["delta_values"], dtype=np.float64)[rows[local]]
    out.update(sample=sample, cell_mpc=float(cell), box_mpc=float(box), n_grid=int(n_grid), method=str(fs.method),
               definition=definition, interior_x=np.asarray(interior, dtype=np.float64))
    out["single_interior"], out["single_box"] = None, None
    if grids["single"] is not None:
        out["single_interior"] = interior_single_fraction(grids["single"], coords[local], r_mpc[local] / cell, interior)
        out["single_box"] = float(grids["single"].mean())
    return out


def _band(values: np.ndarray, n_boot: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """16th and 84th percentiles over 'n_boot' bootstrap means of the rows of 'values' (n, nb)."""
    n = len(values)
    if n < 2:
        return np.full(values.shape[1], np.nan), np.full(values.shape[1], np.nan)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = np.nanmean(values[draws], axis=1)
    return np.nanpercentile(means, 16, axis=0), np.nanpercentile(means, 84, axis=0)


def stack(prof: dict, edges_mpc=DEFAULT_EDGES, n_boot: int = 200, seed: int = 0, min_count: int = 5) -> dict:
    """The profiles stacked in bins of their radius (prof['r_mpc'], else 'r_eff_mpc'): {'edges', 'count': (nbin,), 'bins': x, and per field
    ('density', 'v_r', 'single' when present) {'mean', 'lo', 'hi'}: (nbin, nb)} -- see the module docstring."""
    edges = np.asarray(edges_mpc, dtype=np.float64)
    r_eff = np.asarray(prof["r_mpc"] if "r_mpc" in prof else prof["r_eff_mpc"], dtype=np.float64)   # the shells' radius
    nb = len(prof["bins"])
    nbin = len(edges) - 1
    rng = np.random.default_rng(seed)
    out = {"edges": edges, "bins": np.asarray(prof["bins"]), "count": np.zeros(nbin, dtype=np.int64),
           "n_boot": int(n_boot), "min_count": int(min_count)}
    fields = [k for k in ("density", "v_r", "single") if prof.get(k) is not None]
    for k in fields:
        out[k] = {"mean": np.full((nbin, nb), np.nan), "lo": np.full((nbin, nb), np.nan), "hi": np.full((nbin, nb), np.nan)}
    for i in range(nbin):
        sel = (r_eff >= edges[i]) & (r_eff < edges[i + 1])
        out["count"][i] = int(sel.sum())
        if out["count"][i] < min_count:
            continue
        for k in fields:
            v = np.asarray(prof[k])[sel]
            out[k]["mean"][i] = np.nanmean(v, axis=0)
            out[k]["lo"][i], out[k]["hi"][i] = _band(v, n_boot, rng)
    return out


def linear_v_r(bins, density_mean, r_eff_mpc: float, z: float) -> np.ndarray:
    """Linear-theory v_r (km/s, positive = outflow) at the shells x = r / R_eff from a stacked density
    profile rho/rho_bar(x): v_r = -(1/3) f(z) H(z) a r Delta(<r), Delta(<r) the mean overdensity inside r
    (the profile taken flat inside the first shell; r comoving Mpc). See the module docstring."""
    from . import fields as dtfe
    x = np.asarray(bins, dtype=np.float64)
    d = np.asarray(density_mean, dtype=np.float64) - 1.0
    # integral_{x0}^{x} d x'^2 dx' on a fine grid (the profile linear between shells): the coarse shells'
    # own trapezoid overestimates x^2 by 7% at the first step
    xf = np.linspace(x[0], x[-1], max(4000, 50 * len(x)))
    df = np.interp(xf, x, d)
    inc = 0.5 * (df[1:] * xf[1:] ** 2 + df[:-1] * xf[:-1] ** 2) * np.diff(xf)
    integral = np.concatenate([[0.0], np.cumsum(inc)])
    inner = d[0] * x[0] ** 3 / 3.0                           # flat inside the first shell
    cum = 3.0 * (inner + np.interp(x, xf, integral)) / x ** 3
    c = dtfe.get_cosmology_params(z)
    return -(1.0 / 3.0) * c["f_growth"] * c["H_z"] * c["a"] * (x * r_eff_mpc) * cum


def hsw_delta(x, delta_c, r_s, alpha, beta):
    """The universal void profile of Hamaus, Sutter & Wandelt (2014): delta(r) = delta_c (1 - (r/r_s)^alpha) /
    (1 + (r/R_v)^beta), in x = r / R_v (so R_v = 1 and r_s is r_s / R_v)."""
    x = np.asarray(x, dtype=np.float64)
    return delta_c * (1.0 - (x / r_s) ** alpha) / (1.0 + x ** beta)


def stack_scatter(prof: dict, edges_mpc, key: str = "density", min_count: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """(mean, error) per radius bin of stack(): the mean over the bin's voids and its error std / sqrt(n) per shell
    (the stack's scatter, ddof=1), both (nbin, nb); NaN in a bin below 'min_count' voids. fit_stack's weights."""
    edges = np.asarray(edges_mpc, dtype=np.float64)
    r = np.asarray(prof["r_mpc"] if "r_mpc" in prof else prof["r_eff_mpc"], dtype=np.float64)
    v_all = np.asarray(prof[key], dtype=np.float64)
    nb = v_all.shape[1] if v_all.ndim == 2 else len(prof["bins"])
    mean = np.full((len(edges) - 1, nb), np.nan)
    err = np.full((len(edges) - 1, nb), np.nan)
    for i in range(len(edges) - 1):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        if sel.sum() < max(int(min_count), 2):
            continue
        v = v_all[sel]
        n = np.sum(np.isfinite(v), axis=0)
        mean[i] = np.nanmean(v, axis=0)
        err[i] = np.nanstd(v, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, err


def fit_hsw(bins, density_mean, density_err, x_range=(0.0, np.inf)) -> dict:
    """curve_fit of hsw_delta to a stacked rho/rho_bar (density_mean, its error density_err per shell, at x =
    'bins' in r / R_v; shells outside 'x_range' left out): {'params': {delta_c, r_s, alpha, beta}, 'errors': the
    same (sqrt of the covariance's diagonal, absolute_sigma), 'chi2_dof', 'n_shells', 'ok', 'at_bound' (the
    parameters on a bound of HSW_BOUNDS), 'constrained' (ok, none at a bound, every error below its value: the
    largest R_v bins put delta_c on -1 with alpha ~ 0.2, where the first shell at 0.1 R_v leaves the centre free)}. The errors are floored
    at 1e-3 of the profile's range so a noiseless shell does not get infinite weight. A fit that fails (too few
    shells, no convergence) is ok=False with NaN parameters, never an exception."""
    from scipy.optimize import curve_fit
    x = np.asarray(bins, dtype=np.float64)
    y = np.asarray(density_mean, dtype=np.float64) - 1.0
    e = np.asarray(density_err, dtype=np.float64)
    use = np.isfinite(y) & np.isfinite(e) & (x >= x_range[0]) & (x <= x_range[1])
    nan = {k: float("nan") for k in HSW_PARAMS}
    out = {"params": dict(nan), "errors": dict(nan), "chi2_dof": float("nan"), "n_shells": int(use.sum()), "ok": False,
           "at_bound": [], "constrained": False}
    if use.sum() <= len(HSW_PARAMS):
        return out
    x, y, e = x[use], y[use], e[use]
    e = np.maximum(e, 1e-3 * max(float(np.ptp(y)), 1e-6))
    p0 = list(HSW_P0)
    p0[0] = float(np.clip(y[0], HSW_BOUNDS[0][0] + 1e-3, HSW_BOUNDS[1][0] - 1e-3))   # the centre's own depth
    try:
        popt, pcov = curve_fit(hsw_delta, x, y, p0=p0, sigma=e, absolute_sigma=True, bounds=HSW_BOUNDS, maxfev=20000)
    except (RuntimeError, ValueError):
        return out
    perr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf)) if np.all(np.isfinite(pcov)) else np.full(4, np.nan)
    chi2 = float(np.sum(((hsw_delta(x, *popt) - y) / e) ** 2))
    out.update(params=dict(zip(HSW_PARAMS, map(float, popt))), errors=dict(zip(HSW_PARAMS, map(float, perr))),
               chi2_dof=chi2 / max(len(x) - len(HSW_PARAMS), 1), ok=True)
    lo, hi = np.asarray(HSW_BOUNDS[0]), np.asarray(HSW_BOUNDS[1])
    tol = 1e-4 * (hi - lo)
    out["at_bound"] = [k for k, v, a, b, t in zip(HSW_PARAMS, popt, lo, hi, tol) if v - a <= t or b - v <= t]
    out["constrained"] = bool(not out["at_bound"] and np.all(np.isfinite(perr)) and np.all(perr < np.abs(popt)))
    return out


def fit_stack(prof: dict, edges_mpc, min_count: int = 5, x_range=(0.0, np.inf)) -> list:
    """fit_hsw per radius bin of the profiles 'prof' (stack_scatter's mean and error): one dict per bin with
    'edges' (lo, hi Mpc), 'count' and 'mean_r_mpc' added (a bin below 'min_count' voids: ok=False)."""
    edges = np.asarray(edges_mpc, dtype=np.float64)
    r = np.asarray(prof["r_mpc"] if "r_mpc" in prof else prof["r_eff_mpc"], dtype=np.float64)
    mean, err = stack_scatter(prof, edges, "density", min_count)
    fits = []
    for i in range(len(edges) - 1):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        f = fit_hsw(prof["bins"], mean[i], err[i], x_range) if np.isfinite(mean[i]).any() else \
            {"params": {k: float("nan") for k in HSW_PARAMS}, "errors": {k: float("nan") for k in HSW_PARAMS},
             "chi2_dof": float("nan"), "n_shells": 0, "ok": False, "at_bound": [], "constrained": False}
        f.update(edges=[float(edges[i]), float(edges[i + 1])], count=int(sel.sum()),
                 mean_r_mpc=float(np.mean(r[sel])) if sel.any() else float("nan"))
        fits.append(f)
    return fits
