"""Void-in-void versus void-in-cloud (2026-10-11): each distinct resolved void classified as S-type, VOID-IN-CLOUD
(being squeezed by an overdense surrounding, expected to shrink or be crushed), or R-type, VOID-IN-VOID (inside a
larger underdense region, expanding), by one of CRITERIA; every criterion's measure is reported for each void.

    from dtfelib import pipeline, classify
    p = pipeline.products("099", 0.0)
    env = classify.classify_voids(p)          # per void: the shell's v_r, Delta(<3 R_v) smoothed, Delta(<2 R_v) raw
    env["vic"].mean()                         # the void-in-cloud fraction under the chosen criterion
    classify.agreement(env)                   # the fraction of voids two criteria label the same

THE CRITERIA (classify.CRITERIA, DEFAULT_CRITERION = 'velocity'; CRITERION_SHELL gives each one's shell in R_v):
  'velocity'          (default, 2026-10-11) the mean peculiar RADIAL velocity of the shell between 2 and 3 R_v is
                      negative: the surroundings beyond the compensation ridge are falling IN, the void is being
                      squeezed. This is the dynamical statement Sheth & van de Weygaert (2004, MNRAS 350, 517) make
                      about the void-in-cloud process, measured directly instead of inferred from the density.
  'density_smoothed'  the enclosed contrast of the SMOOTHED density (config.SMOOTHING_SIGMA_CELLS, the catalogue's
                      own field) within 3 R_v is positive: the large-scale environment is overdense.
  'density_raw'       the rule of Ceccarelli, Paz, Lares, Padilla & Lambas (2013, MNRAS 434, 1435): the enclosed
                      contrast of the RAW density within ENV_FACTOR (2) R_v is positive. Kept for comparison: the
                      1-2 R_v shell carries the halos of the walls (the mean profile of the S voids reaches rho ~2 at
                      2 R_v), so it measures the walls more than the larger-scale environment.
WHY THE VELOCITY BY DEFAULT: the velocity in the 2-3 R_v shell, beyond the ridge, responds to the mass of the whole
region inside it (linear theory: v_r = -(1/3) f H a r Delta(<r)), which smooths over individual walls and haloes,
and it says directly whether the void is being squeezed. Measured (TNG100-3-Dark / TNG50-3-Dark, 2026-10-11): it
labels 80-94% of the voids (R_v <= L/6) void-in-cloud, against 40-46% of random centres with the same radii (whose
mean flux is ~0), agrees with the smoothed Delta(<3 R_v) on 87-95% of the voids and with the raw rule on 70-88%,
and its S voids are the smaller ones at every z and the shallower ones at z >= 1 (Sheth & van de Weygaert's
prediction), where the raw rule shows no clear split. The voids beyond the 3 R_v wrap rule, the largest ones, are
mostly void-in-void by the raw rule: the fraction holds for R_v <= L/6, not for the whole population.

THE MEASUREMENTS. All three sample spherical shells around the void's cell CENTRE ((i + 0.5) / N) with
profiles.profile_grids (Fibonacci directions, trilinear, periodic through environment.sample_grid):
  * Delta(<X R) = 3 / X^3 int_0^X (rho/rho_bar - 1) x^2 dx, x = r / R, from SHELLS shell means out to X R_v
    (enclosed_contrast: the profile linear between the shells on a fine grid, flat inside the first one, the
    integral of profiles.linear_v_r), of the raw density (X = ENV_FACTOR) and of 1 + the smoothed delta (X = 3);
  * the shell velocity: v_r = v . r_hat (POSITIVE = OUTFLOW) on VEL_SHELLS shells from 2 to 3 R_v, VOLUME weighted
    across the shells, v_shell = int v_r x^2 dx / int x^2 dx (trapezoid). The velocity is the grid as stored (its
    per-cell weighting is PS-DTFE's), read with FieldSet.load, which applies the sqrt(a) factor -> peculiar km/s;
    the void's bulk velocity is a dipole over each shell and cancels in the mean, as in profiles.py. 'v_shell_norm'
    = v_shell / (H(z) a R_v), R_v comoving Mpc, H in km/s/Mpc (fields.get_cosmology_params, linear_v_r's factor):
    linear theory gives v_shell_norm = -(1/3) f(z) x Delta(<x) averaged over the shell, so it reads directly as
    the strength of the squeeze (-0.04 for Delta(<2.5 R_v) = +0.1 at z = 0, f = 0.52).
A void whose outer shell would reach beyond half the box is dropped for that measure (profile_grids' wrap rule: in a
periodic box it would sample its own interior); the rows returned are those the CHOSEN criterion keeps, and another
criterion's measure is NaN where its own (larger) shell is dropped.

THE SAMPLE is the profiles': every well-resolved void with a measured R_v (catalog.r_v_mpc), only the DISTINCT ones
(catalog.distinct_mask) unless distinct=False. Nothing is cached: the grid reads dominate (density 0.5 GB, velocity
1.6 GB at 512^3) and the shell sampling takes seconds; the figure script stores the per-void table in its own npz.
classify_grid / classify_catalog keep the raw-density defaults of the first version (2026-10-11 morning): the default
criterion is the velocity at the snapshot level, classify_voids, which has the FieldSet to read the velocity from.
"""

from __future__ import annotations

import numpy as np

from . import catalog
from .profiles import fibonacci_sphere, profile_grids, shell_means

ENV_FACTOR = 2.0            # Delta(< ENV_FACTOR x R_v) decides the type (Ceccarelli et al. 2013)
VIC_THRESHOLD = 0.0         # Delta above it: void-in-cloud (S-type); at or below: void-in-void (R-type)
SHELLS = 40                 # shells between 0 and ENV_FACTOR R_v
DIRS = 300                  # Fibonacci directions per shell
TYPES = ("R", "S")          # void-in-void, void-in-cloud
TYPE_NAMES = {"R": "void-in-void", "S": "void-in-cloud"}
CRITERIA = ("velocity", "density_smoothed", "density_raw")
DEFAULT_CRITERION = "velocity"
# each criterion's shell in units of R_v: (inner, outer); density_raw's outer radius is the 'factor' argument
CRITERION_SHELL = {"velocity": (2.0, 3.0), "density_smoothed": (0.0, 3.0), "density_raw": (0.0, ENV_FACTOR)}
CRITERION_LABEL = {"velocity": "mean v_r(2-3 R_v) < 0",
                   "density_smoothed": "smoothed Delta(<3 R_v) > 0",
                   "density_raw": "raw Delta(<2 R_v) > 0"}
VEL_SHELLS = 11             # shells from 2 to 3 R_v for the velocity (0.1 R_v apart)


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


def velocity_shell_grid(velocity, centers_frac, r_v_cells, inner: float = 2.0, outer: float = 3.0,
                        shells: int = VEL_SHELLS, m: int = DIRS) -> dict:
    """The volume-weighted mean radial velocity of the shell between 'inner' and 'outer' x R_v around every void
    given as (centre fraction (n, 3), R_v in cells (n,)) on the (N,N,N,3) velocity grid (the caller's units, km/s
    from FieldSet.load): {'v_shell': (n,) NaN where dropped, 'kept', 'dropped', 'bins' (shells' x), 'v_r' (n, shells)}.
    v_shell = int v_r x^2 dx / int x^2 dx over the shells (trapezoid; POSITIVE = OUTFLOW). Dropped: no finite
    positive R_v, or the outer shell beyond half the box (profile_grids' wrap rule)."""
    bins = np.linspace(float(inner), float(outer), int(shells))
    r = np.asarray(r_v_cells, dtype=np.float64).reshape(-1)
    centers = np.asarray(centers_frac, dtype=np.float64).reshape(-1, 3)
    n_grid = velocity.shape[0]
    # profile_grids' wrap rule, applied here because its own call would also sample a scalar 'density' grid
    kept = np.isfinite(r) & (r > 0) & (bins.max() * r <= 0.5 * n_grid)
    v = np.full(len(r), np.nan)
    v_r = np.full((len(r), len(bins)), np.nan)
    if kept.any():
        dirs = fibonacci_sphere(m)
        for j, x in enumerate(bins):
            v_r[kept, j] = shell_means(velocity, centers[kept], x * r[kept], dirs, radial=True)
        w = bins ** 2
        tw = np.zeros(len(bins))                     # trapezoid weights of int f x^2 dx (numpy 1.26 and 2.x alike)
        tw[1:] += 0.5 * np.diff(bins)
        tw[:-1] += 0.5 * np.diff(bins)
        v[kept] = (v_r[kept] * (w * tw)).sum(axis=1) / (w * tw).sum()
    return {"v_shell": v, "kept": kept, "dropped": int((~kept).sum()), "bins": bins, "v_r": v_r}


def measure_grid(density, centers_frac, r_v_mpc, cell_mpc: float, factor: float = ENV_FACTOR, *, velocity=None,
                 density_smoothed=None, redshift=None, criterion: str = "density_raw") -> dict:
    """Every criterion's measure for the voids given as (centre fraction (n, 3), R_v in Mpc (n,)) on the grids handed
    in (see classify_catalog), the type by 'criterion': (n,) arrays 'delta_enclosed', 'delta_smoothed_enclosed',
    'v_shell_kms', 'v_shell_norm' (NaN where not measured or dropped), 'vic_<c>' for every criterion, 'kept' (the
    chosen criterion's wrap rule), 'vic', 'type' ('S' / 'R', '' where not kept); plus 'dropped', 'bins', 'profile'
    (raw rho/rho_bar out to factor R_v), 'shell' (the criterion's (inner, outer) in R_v), 'measured'."""
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    if criterion == "velocity" and velocity is None:
        raise ValueError("the velocity criterion needs the velocity grid (FieldSet.load('velocity'))")
    if criterion == "density_smoothed" and density_smoothed is None:
        raise ValueError("the density_smoothed criterion needs the smoothed density (1 + pipeline delta_smoothed)")
    if velocity is not None and redshift is None:
        raise ValueError("the velocity measure needs the redshift (v_shell_norm is in units of H(z) a R_v)")
    r_v = np.asarray(r_v_mpc, dtype=np.float64).reshape(-1)
    r_cells = r_v / float(cell_mpc)
    g = classify_grid(density, centers_frac, r_cells, factor=factor)
    n = len(r_v)
    kept = {"density_raw": g["kept"]}
    sm = np.full(n, np.nan)
    if density_smoothed is not None:
        gs = classify_grid(density_smoothed, centers_frac, r_cells, factor=CRITERION_SHELL["density_smoothed"][1])
        sm, kept["density_smoothed"] = gs["delta_enclosed"], gs["kept"]
    v_kms, v_norm = np.full(n, np.nan), np.full(n, np.nan)
    if velocity is not None:
        lo_, hi_ = CRITERION_SHELL["velocity"]
        gv = velocity_shell_grid(velocity, centers_frac, r_cells, inner=lo_, outer=hi_)
        from .fields import get_cosmology_params
        c = get_cosmology_params(float(redshift))
        v_kms, kept["velocity"] = gv["v_shell"], gv["kept"]
        v_norm = v_kms / (c["H_z"] * c["a"] * r_v)
    # each criterion's verdict; NaN (dropped or not measured) is never void-in-cloud
    vics = {"velocity": np.isfinite(v_kms) & (v_kms < 0.0),
            "density_smoothed": np.isfinite(sm) & (sm > VIC_THRESHOLD),
            "density_raw": g["vic"]}
    # a void the wrap rule keeps but whose measure is NaN (a NaN grid cell on its shells) is dropped and counted,
    # never typed: NaN would otherwise read as 'not void-in-cloud'
    measure = {"velocity": v_kms, "density_smoothed": sm, "density_raw": g["delta_enclosed"]}[criterion]
    k = kept[criterion] & np.isfinite(measure)
    vic = vics[criterion] & k
    shell = (CRITERION_SHELL[criterion][0],
             float(factor) if criterion == "density_raw" else CRITERION_SHELL[criterion][1])
    out = {"delta_enclosed": g["delta_enclosed"], "delta_smoothed_enclosed": sm, "v_shell_kms": v_kms,
           "v_shell_norm": v_norm, "kept": k, "vic": vic, "type": np.where(~k, "", np.where(vic, "S", "R")),
           "dropped": int((~k).sum()), "bins": g["bins"], "profile": g["profile"], "shell": shell,
           "measured": tuple(c for c in CRITERIA if c in kept), "criterion": criterion}
    for c in CRITERIA:
        out[f"vic_{c}"] = vics[c]
    return out


def classify_catalog(cat, density, cell_mpc: float, factor: float = ENV_FACTOR, distinct: bool = True,
                     definition: str = "crossing", *, velocity=None, density_smoothed=None, redshift=None,
                     criterion: str = "density_raw") -> dict:
    """Every criterion's measure for a catalogue's sample (void_sample) on the grids handed in, the type by
    'criterion': the raw density rho/rho_bar 'density' (always: Delta(< factor R_v), 'delta_enclosed'), the smoothed
    rho/rho_bar 'density_smoothed' (Delta(<3 R_v), 'delta_smoothed_enclosed'), the (N,N,N,3) peculiar 'velocity' in
    km/s (the 2-3 R_v shell's 'v_shell_kms', and 'v_shell_norm' in units of H(z) a R_v, which needs 'redshift').
    A measure without its grid is NaN; asking for a criterion whose grid is missing raises ValueError (no silent
    fallback to another criterion). The default criterion here is the raw density, the grid every caller passes;
    classify_voids, which reads the snapshot's grids itself, defaults to DEFAULT_CRITERION (the velocity).
    One row per void the CHOSEN criterion keeps (its wrap rule; the rest counted in 'dropped') with 'index'
    (catalogue row), 'delta_enclosed', 'delta_smoothed_enclosed', 'v_shell_kms', 'v_shell_norm', 'vic' and 'type'
    (the criterion's), 'vic_<criterion>' for every criterion (False where its measure is NaN), 'r_v_mpc', 'delta_c'
    (central smoothed delta, the depth), 'e', 'p' (BBKS), 'b_over_a', 'c_over_a', 'r_eff_mpc', 'center_frac';
    plus 'n_sample', 'dropped', 'factor', 'bins', 'profile' (raw rho/rho_bar out to factor R_v), 'definition',
    'criterion', 'shell' (its (inner, outer) in R_v), 'measured' (the criteria that had their grid)."""
    n_grid = int(density.shape[0])
    rows = void_sample(cat, n_grid, distinct, definition)
    coords = np.asarray(cat["coords"])[rows]
    r_v = catalog.r_v_mpc(cat, definition)[rows]
    centers = (coords + 0.5) / n_grid
    m = measure_grid(density, centers, r_v, cell_mpc, factor=factor, velocity=velocity,
                     density_smoothed=density_smoothed, redshift=redshift, criterion=criterion)
    k = m["kept"]
    bbks = np.asarray(cat["bbks_params"], dtype=np.float64).reshape(-1, 2)[rows]
    ratios = np.asarray(cat["axis_ratios"], dtype=np.float64).reshape(-1, 2)[rows]
    out = {"index": rows[k], "delta_enclosed": m["delta_enclosed"][k],
           "delta_smoothed_enclosed": m["delta_smoothed_enclosed"][k], "v_shell_kms": m["v_shell_kms"][k],
           "v_shell_norm": m["v_shell_norm"][k], "vic": m["vic"][k], "type": m["type"][k],
           "r_v_mpc": r_v[k], "delta_c": np.asarray(cat["delta_values"], dtype=np.float64)[rows][k],
           "e": bbks[k, 0], "p": bbks[k, 1], "b_over_a": ratios[k, 0], "c_over_a": ratios[k, 1],
           "r_eff_mpc": catalog.r_eff_mpc(cat)[rows][k], "center_frac": centers[k], "profile": m["profile"][k],
           "bins": m["bins"], "n_sample": int(len(rows)), "dropped": m["dropped"], "factor": float(factor),
           "distinct": bool(distinct), "definition": str(definition), "criterion": criterion, "shell": m["shell"],
           "measured": m["measured"]}
    for c in CRITERIA:
        out[f"vic_{c}"] = m[f"vic_{c}"][k]
    return out


def classify_voids(products, factor: float = ENV_FACTOR, distinct: bool = True, definition: str = "crossing",
                   criterion: str = DEFAULT_CRITERION) -> dict:
    """classify_catalog on one snapshot (pipeline.SnapshotProducts): its catalogue (the frame checked by
    catalog.frame_of), its raw density rho/rho_bar, 1 + its smoothed delta, and its velocity (FieldSet.load:
    peculiar km/s, read into RAM, dropped on return; skipped when the snapshot has none and the criterion is not
    the velocity). Every criterion's measure is filled; 'criterion' decides the type. Adds 'cell_mpc', 'box_mpc',
    'n_grid', 'method', 'prefix', 'redshift'."""
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    cell, box, n_grid, _src = catalog.frame_of(products)
    cat = products.voids()
    catalog.assert_frame(cat, cell)
    fs = products.fs
    has_vel = bool(getattr(fs, "has", lambda name: False)("velocity"))
    if criterion == "velocity" and not has_vel:
        raise ValueError(f"no velocity grid in {getattr(fs, 'snapdir', '?')}: the velocity criterion needs one "
                         f"(choose criterion='density_smoothed' or 'density_raw')")
    velocity = fs.load("velocity", mode="ram") if has_vel else None
    z = float(products.redshift)
    out = classify_catalog(cat, products.density_raw, cell, factor=factor, distinct=distinct, definition=definition,
                           velocity=velocity, density_smoothed=products.delta_smoothed + 1.0, redshift=z,
                           criterion=criterion)
    del velocity
    out.update(cell_mpc=float(cell), box_mpc=float(box), n_grid=int(n_grid), method=str(fs.method),
               prefix=str(fs.prefix), redshift=z)
    return out


def agreement(out: dict) -> dict:
    """{(a, b): (fraction of the voids that criteria a and b label the same, the number compared)} for every pair of
    the criteria measured in a classify_catalog / classify_voids result ('measured'), over the voids where both
    measures are finite. (NaN, 0) when no void has both."""
    finite = {"velocity": np.isfinite(np.asarray(out["v_shell_kms"], dtype=float)),
              "density_smoothed": np.isfinite(np.asarray(out["delta_smoothed_enclosed"], dtype=float)),
              "density_raw": np.isfinite(np.asarray(out["delta_enclosed"], dtype=float))}
    meas = [c for c in CRITERIA if c in out.get("measured", CRITERIA)]
    res = {}
    for i, a in enumerate(meas):
        for b in meas[i + 1:]:
            both = finite[a] & finite[b]
            nb = int(both.sum())
            same = np.asarray(out[f"vic_{a}"], dtype=bool)[both] == np.asarray(out[f"vic_{b}"], dtype=bool)[both]
            res[(a, b)] = (float(same.mean()) if nb else float("nan"), nb)
    return res


def vic_fraction(vic, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(fraction, 16th, 84th percentile of n_boot bootstrap resamples) of a boolean array; NaNs when empty."""
    v = np.asarray(vic, dtype=bool)
    if not v.size:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, v.size, size=(int(n_boot), v.size))].mean(axis=1)
    return float(v.mean()), float(np.percentile(boot, 16)), float(np.percentile(boot, 84))
