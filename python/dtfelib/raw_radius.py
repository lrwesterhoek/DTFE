"""The void radius of the size-function theory measured on the RAW density: where the ENCLOSED density
contrast Delta(<R) around a void centre first reaches DELTA_V_NONLINEAR = -0.8 going outward, on the unsmoothed DTFE
rho/rho_bar -- 'delta08raw', beside catalog's 'delta08' (the same rule on the SMOOTHED field).

    from dtfelib import pipeline, raw_radius
    p = pipeline.products("099")
    cat = p.voids()
    r = raw_radius.raw_radius_mpc(p, cat)          # Mpc per catalogue row (NaN: undefined or not measured)

WHY. The size function (theory.void_size_function, Vdn) counts spheres whose enclosed contrast has reached the
shell-crossing -0.8. On the smoothed field that radius follows the smoothing length at every sigma tried (TNG100,
10/5/3/2 cells: median 0.7-1.06 sigma) and measured/Vdn stays at 1e-3-1e-2: the -0.8 sphere of a smoothed minimum is a
smoothing length. The raw field keeps the void's own emptiness (rho/rho_bar ~ 0.1 inside TNG voids), so the -0.8 sphere
about a centre can be a void size -- if it does not scale with sigma; plot_void_population.py measures that (notes in
its docstring).

THE MEASUREMENT. Centres: the catalogue's minima of the smoothed field, at their cell CENTRES ((i + 0.5) / N, as
profiles and pipeline.void_radii_cells). Around each, the raw delta = rho/rho_bar - 1 is averaged over spherical shells
(profiles.shell_means: RAW_DIRS Fibonacci directions, trilinear, periodic) every RAW_STEP_CELLS cells, column 0 the
raw value AT the centre (trilinear at a cell centre = that cell), and Delta(<R) integrated with delta piecewise linear
in r (pipeline._enclosed_contrast, exact for a constant profile). The radius is the first UP-crossing of -0.8 going
outward, linearly interpolated between the two shells around it: the enclosed contrast must have been below -0.8
first. NaN when it never goes below -0.8 within reach (a centre in a filament wisp: no sphere about it is that empty)
or never comes back up (still below at the reach, 'capped'). The rule differs from catalog's delta08 in one place on
purpose: there the minimum itself must be below the level; on the raw field a single cell is noise (one tetrahedron
can put a wisp on the centre) while the enclosed contrast a cell further out is already -0.9.

STEP AND REACH ARE IN CELLS, NOT SIGMA. Step RAW_STEP_CELLS (half a cell: the raw field is cell-resolved) and reach
RAW_REACH_FRAC x N (0.45 of the box, the R_v search range's own margin below L/2: pipeline.R_V_MAX_SIGMA is 24 of the
25.6 sigma to L/2 at 10 cells). The catalogue's R_v search range is in smoothing lengths (24 sigma = 72 cells at 3
cells), so with it a radius measured at 3 cells could be cut where the same void at 10 cells is not: the test 'does the
radius follow sigma' would be confounded by the reach. A void is grown shell block by shell block and dropped as soon
as it has crossed, so the large reach costs only for the few that never cross.

ONLY THE ROWS ASKED FOR are measured (default: the catalogue's well_resolved rows -- the size function's sample is a
subset; TNG300 at 3.66 cells has tens of thousands of minima); the rest stay NaN and 'measured' says which.

THE CACHE. raw_radius_mpc keeps the radii beside the void catalogue, in the catalogue's own cache namespace (one per
smoothing): '<sim>_<method>_<snap>_voids_raw08.npz' (products._cpath), keyed on the density grid's stamp (size,
mtime_ns), a hash of the catalogue's minima (its identity: a rebuilt catalogue with other minima re-measures), and
settings() (step, reach, directions, threshold, ALGORITHM). NOT on pipeline.r_v_settings(): the crossing R_v's search
range changes the distinct set, not these radii. Rows not yet measured are measured and added; written atomically
(pipeline._save_npz: temp file, then rename).

RE-CENTRING (raw_radii(..., recentre=True)). The catalogue's centres are minima of the SMOOTHED field; the uncentred
fixed-R size function follows the smoothing (TNG300 z = 0: 0.84 / 0.78 / 0.58 of Vdn at 3.66 / 3 / 2 cells), which a
small-sigma minimum sitting off the raw void's centre would explain (the raw -0.8 sphere about it too small).
The option moves each centre to the MINIMUM OF THE RAW DENSITY smoothed by RC_SMOOTH_CELLS (a Gaussian in CELLS, so it
brings no sigma back; periodic) inside a ball of RC_SEARCH_FRAC x the void's own uncentred raw radius R0 about the
catalogue centre, then measures the -0.8 radius there (raw_radii_cells, unchanged); up to RC_ITER passes, each
searching about the last centre with the radius measured there, until no centre moves. WHY THIS RULE and
not 'the centre that maximises the -0.8 radius': that optimises the measured quantity over the noise and makes every
radius larger by construction, so a higher abundance afterwards would prove nothing; the minimum is the void's own
centre, chosen without looking at the radius. WHY 0.5 R0 and not 0.5 R_eff: ungated (the size function's sample),
R_eff is uncapped for the large voids and a ball of half of it can reach the next void; inside the R0 sphere the
enclosed contrast is already below -0.8, so the new centre stays in the same empty region, and R0 has no sigma in it.
A void with no R0 because its centre sits in a wisp ('never_below', the case re-centring could rescue) searches
RC_SEARCH_FRAC x the catalogue's crossing R_v instead (gate-independent, unlike R_eff), and so does a 'capped' one
(no R0 either); a void with neither stays put. Cached as '<...>_voids_raw08rc.npz' (NAME_RC) beside the uncentred file, with its own settings
(settings_rc: settings() + the four re-centring numbers) and the re-centred cells and offsets per row.
MEASURED (TNG100/TNG300/TNG50, z = 0, 0.5, 1; plot_void_population.py FIXED_R comments): the centres move by a median
0.9-1.6 sigma in every box, but the -0.8 sphere about the raw minimum is SMALLER for 70-88% of the voids (median R/R0
0.81-0.93): the emptiest point is not the centre of the largest empty sphere. It is therefore an option, not the
fixed-R default (TNG300's smoothing dependence mostly goes, but its z = 0 count halves and TNG100 z = 0 breaks).
"""

from __future__ import annotations

import hashlib

import numpy as np

from . import theory
from .environment import sample_grid
from .profiles import fibonacci_sphere, shell_means

RAW_STEP_CELLS = 0.5        # radial step in cells (the raw field is cell-resolved; the crossing is interpolated)
RAW_REACH_FRAC = 0.45       # reach as a fraction of the box (N cells); the R_v search range's margin below L/2
RAW_DIRS = 256              # Fibonacci directions per shell: twice R_V_DIRS, the raw field is noisier than the smoothed
RAW_LEVEL = theory.DELTA_V_NONLINEAR    # -0.8: the enclosed contrast of a void at shell crossing
RAW_BLOCK = 24              # shells per outward block (voids that crossed are dropped between blocks)
# Bumped by hand when the measurement changes (a cache records it in settings() and is re-measured otherwise):
# 1 = the first up-crossing after the first point below the level.
ALGORITHM = 1
NAME = "voids_raw08"        # the cache's product name (products._cpath(NAME))
# Re-centring (raw_radii(..., recentre=True); module docstring):
RC_SMOOTH_CELLS = 1.0       # the light Gaussian on the raw density before its minimum is taken (cells, not sigma)
RC_SEARCH_FRAC = 0.5        # the search ball: this x the uncentred raw radius R0 (x the crossing R_v without an R0)
RC_ITER = 8                 # at most this many search passes; each searches about the last centre with the radius
                            # measured there, and a centre that does not move has converged
RC_ALGORITHM = 1            # bumped by hand when the re-centring changes (recorded in settings_rc)
NAME_RC = "voids_raw08rc"   # its cache's product name


def settings(step_cells=None, reach_frac=None, m=None, level=None) -> np.ndarray:
    """What the radii are measured with, as a cache records it: [step in cells, reach as a fraction of N,
    directions, level, ALGORITHM]. Read at call time, so a changed module constant counts."""
    return np.array([RAW_STEP_CELLS if step_cells is None else step_cells,
                     RAW_REACH_FRAC if reach_frac is None else reach_frac,
                     RAW_DIRS if m is None else m, RAW_LEVEL if level is None else level, ALGORITHM],
                    dtype=np.float64)


def raw_radii_cells(rho_mean, coords, step_cells: float = None, reach_cells: float = None, m: int = None,
                    level: float = None, block: int = RAW_BLOCK) -> dict:
    """The enclosed -0.8 radius in CELLS about each of 'coords' ((n, 3) integer cells; their centres) on the raw
    'rho_mean' (N,N,N) rho/rho_bar (module docstring): {'r_cells': (n,), 'never_below': (n,) bool (Delta(<R) never
    below 'level' within reach), 'capped': (n,) bool (below at the reach, no crossing), 'min_enclosed': (n,) the
    lowest Delta(<R) seen}. NaN radius for both kinds of failure."""
    from .pipeline import _enclosed_contrast          # here: pipeline imports catalog, which imports profiles
    step = float(RAW_STEP_CELLS if step_cells is None else step_cells)
    level = float(RAW_LEVEL if level is None else level)
    m = int(RAW_DIRS if m is None else m)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    n = int(rho_mean.shape[0])
    reach = float(RAW_REACH_FRAC * n if reach_cells is None else reach_cells)
    nv = len(coords)
    out = {"r_cells": np.full(nv, np.nan), "never_below": np.zeros(nv, bool), "capped": np.zeros(nv, bool),
           "min_enclosed": np.full(nv, np.nan)}
    if not nv:
        return out
    radii = np.arange(1, int(np.floor(reach / step + 1e-9)) + 1) * step
    dirs = fibonacci_sphere(m)
    centers = (coords + 0.5) / n
    # column 0: the raw delta AT the centre; carried per void: the last radius, its Delta(<R), the integral of
    # delta r^2 dr so far (Delta = 3 I / R^3), the last shell value (delta is linear between shells)
    last_r = np.zeros(nv)
    last_v = np.asarray(sample_grid(rho_mean, centers), dtype=np.float64) - 1.0
    last_D = last_v.copy()
    integ = np.zeros(nv)
    mind = last_D.copy()
    active = np.arange(nv)
    for a in range(0, radii.size, int(block)):
        if not active.size:
            break
        rb = radii[a:a + int(block)]
        k = rb.size
        cen = np.repeat(centers[active], k, axis=0)                # (void, shell) pairs, void-major
        sh = shell_means(rho_mean, cen, np.tile(rb, active.size), dirs).reshape(active.size, k) - 1.0
        # _enclosed_contrast from the previous radius: its columns 1.. are 3 int_{r_prev}^{r} delta r^2 dr / r^3
        rr = np.concatenate([[0.0], rb])                           # placeholder r_0, replaced per void below
        prof = np.concatenate([last_v[active, None], sh], axis=1)
        D = np.empty((active.size, k + 1))
        D[:, 0] = last_D[active]
        # one call per distinct previous radius (every active void shares it: all grow on the same shells)
        rr[0] = last_r[active[0]]
        part = _enclosed_contrast(rr, prof)[:, 1:] * rr[1:] ** 3 / 3.0   # int_{r_prev}^{r_j} delta r^2 dr
        tot = integ[active, None] + part
        D[:, 1:] = 3.0 * tot / rr[1:] ** 3
        mind[active] = np.minimum(mind[active], D[:, 1:].min(axis=1))
        up = (D[:, :-1] < level) & (D[:, 1:] >= level)             # an up-crossing between columns j-1 and j
        hit = up.any(axis=1)
        if hit.any():
            rows = np.nonzero(hit)[0]
            j = np.argmax(up[rows], axis=1) + 1
            rlo = rr[j - 1]                                        # rr[0] = the previous block's last radius
            dlo, dhi = D[rows, j - 1], D[rows, j]
            out["r_cells"][active[rows]] = rlo + (rr[j] - rlo) * (level - dlo) / (dhi - dlo)
        last_r[active] = rb[-1]
        last_v[active] = sh[:, -1]
        last_D[active] = D[:, -1]
        integ[active] = tot[:, -1]
        active = active[~hit]
    out["min_enclosed"] = mind
    rest = ~np.isfinite(out["r_cells"])
    out["never_below"] = rest & ~(mind < level)
    out["capped"] = rest & (mind < level)
    return out


def settings_rc() -> np.ndarray:
    """settings() + [RC_SMOOTH_CELLS, RC_SEARCH_FRAC, RC_ITER, RC_ALGORITHM]: what the re-centred radii are measured
    with (read at call time)."""
    return np.concatenate([settings(), np.array([RC_SMOOTH_CELLS, RC_SEARCH_FRAC, RC_ITER, RC_ALGORITHM],
                                                dtype=np.float64)])


def _ball_offsets(rad: float) -> np.ndarray:
    """Integer cell offsets (k, 3) with |offset| <= rad, nearest first (argmin then keeps the nearest of tied
    minima: a flat empty patch does not throw the centre to the ball's edge)."""
    m = int(np.floor(rad + 1e-9))
    g = np.arange(-m, m + 1)
    o = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    d2 = (o * o).sum(axis=1)
    keep = d2 <= rad * rad + 1e-9
    o, d2 = o[keep], d2[keep]
    return o[np.argsort(d2, kind="stable")]


def recentre_cells(field, coords, search_cells) -> np.ndarray:
    """The cell of the minimum of 'field' (N,N,N; the lightly smoothed raw density) within a ball of 'search_cells'
    (per void, cells) about each of 'coords' ((n, 3) integer cells), periodic. A void with a search radius below one
    cell or NaN stays where it is. Returns (n, 3) int64 cells."""
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    search = np.asarray(search_cells, dtype=np.float64).reshape(-1)
    n = int(field.shape[0])
    out = coords.copy()
    rad = np.where(np.isfinite(search), np.minimum(search, 0.25 * n), 0.0)    # a quarter box: the ball stays small
    cache = {}
    for i in np.nonzero(rad >= 1.0)[0]:
        key = round(float(rad[i]), 1)                # 0.1 cell: the ball's cells barely change below that
        if key not in cache:
            cache[key] = _ball_offsets(key)
        cell = (coords[i] + cache[key]) % n
        out[i] = cell[int(np.argmin(field[cell[:, 0], cell[:, 1], cell[:, 2]]))]
    return out


def recentred_radii_cells(rho_mean, coords, r0_cells, fallback_cells=None, smooth_cells: float = None,
                          search_frac: float = None, n_iter: int = None, smoothed=None) -> dict:
    """The raw -0.8 radius about RE-CENTRED centres (module docstring, RE-CENTRING): each of 'coords' moves to the
    minimum of the raw 'rho_mean' smoothed by 'smooth_cells' within search_frac x its uncentred radius 'r0_cells'
    (NaN: search_frac x 'fallback_cells', the crossing R_v), then raw_radii_cells measures there; up to n_iter
    passes, the later ones searching about the previous centre with its radius (where it has none, the previous ball),
    until no centre moves. 'smoothed' passes the smoothed grid in (tests; else made here). Returns raw_radii_cells' dict + 'coords_rc' (n, 3) cells,
    'offset_cells' (periodic distance from the catalogue centre), 'rule' (0 not moved: no radius to search with, 1 R0,
    2 the fallback: no R0, never below or capped)."""
    from .fields import smooth_field
    smooth_cells = float(RC_SMOOTH_CELLS if smooth_cells is None else smooth_cells)
    search_frac = float(RC_SEARCH_FRAC if search_frac is None else search_frac)
    n_iter = int(RC_ITER if n_iter is None else n_iter)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    n = int(rho_mean.shape[0])
    r0 = np.asarray(r0_cells, dtype=np.float64).reshape(-1)
    fb = np.full(r0.size, np.nan) if fallback_cells is None else np.asarray(fallback_cells, np.float64).reshape(-1)
    rule = np.where(np.isfinite(r0), 1, np.where(np.isfinite(fb), 2, 0)).astype(np.int8)
    search = search_frac * np.where(rule == 1, r0, np.where(rule == 2, fb, np.nan))
    if smoothed is None:
        smoothed = smooth_field(np.asarray(rho_mean, dtype=np.float32), smooth_cells, mode="wrap")
    cur = coords.copy()
    active = np.nonzero(rule > 0)[0]
    for _ in range(max(n_iter, 1)):
        if not active.size:
            break
        new = recentre_cells(smoothed, cur[active], search[active])
        moved = (new != cur[active]).any(axis=1)
        active, new = active[moved], new[moved]
        if not active.size:
            break
        cur[active] = new
        # a centre that moved searches again about itself with the radius measured there (none: the same ball):
        # the ball is half a radius that is too small when the centre is off, so one pass moves it only part way
        # (a top-hat void with the centre 6 cells off: 2 cells per pass); a centre that did not move has converged
        r = raw_radii_cells(rho_mean, cur[active])["r_cells"]
        search[active] = np.where(np.isfinite(r), search_frac * r, search[active])
    got = raw_radii_cells(rho_mean, cur)
    d = (cur - coords).astype(np.float64)
    d -= n * np.rint(d / n)
    got["coords_rc"] = cur
    got["offset_cells"] = np.sqrt((d * d).sum(axis=1))
    got["rule"] = rule
    return got


def coords_hash(coords) -> str:
    """The catalogue's identity for the cache: a SHA-1 of its minima (int64 cells, row order)."""
    c = np.ascontiguousarray(np.asarray(coords, dtype=np.int64))
    return hashlib.sha1(c.tobytes() + str(c.shape).encode()).hexdigest()


def _key_ok(z, stamp, chash, want) -> bool:
    return ("settings" in z and "coords_sha1" in z and "grid_stamp" in z
            and np.array_equal(np.asarray(z["settings"], np.float64), want)
            and str(z["coords_sha1"]) == chash
            and np.array_equal(np.asarray(z["grid_stamp"], np.int64), np.asarray(stamp, np.int64)))


def raw_radii(products, cat, rows=None, say=print, recentre=False) -> dict:
    """raw_radii_cells for the catalogue 'cat' of 'products' (pipeline.SnapshotProducts; its grids, frame and cache
    namespace) at the rows 'rows' (bool mask or indices; default cat['well_resolved']), cached (module docstring):
    {'r_cells', 'r_mpc', 'never_below', 'capped', 'measured', 'min_enclosed'}, every array per catalogue row (NaN /
    False where not measured). recentre=True: the radii about the RE-CENTRED centres (module docstring),
    cached apart (NAME_RC), plus 'coords_rc' (the catalogue's own cell where not measured), 'offset_cells', 'rule'."""
    if recentre:
        return _raw_radii_rc(products, cat, rows, say)
    from . import catalog, pipeline
    cell, _box, n_grid, _src = catalog.frame_of(products)
    coords = np.asarray(cat["coords"], dtype=np.int64)
    nrow = len(coords)
    want_rows = np.zeros(nrow, bool)
    if rows is None:
        want_rows[:] = np.asarray(cat["well_resolved"]).astype(bool)
    else:
        rows = np.asarray(rows)
        if rows.dtype == bool:
            want_rows[:] = rows
        else:
            want_rows[rows.astype(np.int64)] = True
    want = settings()
    stamp = np.asarray(products._grid_stamp(), dtype=np.int64)
    chash = coords_hash(coords)
    path = products._cpath(NAME)
    keys = ("r_cells", "never_below", "capped", "measured", "min_enclosed")
    have = None
    if path.exists():
        try:
            with np.load(path) as z:
                if _key_ok(z, stamp, chash, want) and all(k in z for k in keys):
                    have = {k: np.array(z[k]) for k in keys}
                else:
                    say(f"    [raw_radius] cached {path.name} is for another grid, catalogue or settings: re-measuring")
        except Exception as e:                            # a cut or foreign file: re-measured, said
            say(f"    [raw_radius] cached {path.name} is unreadable ({type(e).__name__}): re-measuring")
    if have is None or len(have["r_cells"]) != nrow:
        have = {"r_cells": np.full(nrow, np.nan), "never_below": np.zeros(nrow, bool), "capped": np.zeros(nrow, bool),
                "measured": np.zeros(nrow, bool), "min_enclosed": np.full(nrow, np.nan)}
    todo = np.nonzero(want_rows & ~have["measured"])[0]
    if todo.size:
        # the grid the products already hold: raw_radii is called while a fresh voids() build still has
        # its density_raw (that same fs.density(units='mean')), and fs.density would read a second 0.5 GB copy from
        # disk.
        # Not the property: on a cached catalogue it would load the grid into the products until release()
        rho = getattr(products, "_ram", {}).get("density_raw")
        if rho is None:
            rho = products.fs.density(units="mean")
        if rho.shape[0] != n_grid:
            raise ValueError(f"density grid is {rho.shape[0]}^3, the catalogue's frame {n_grid}^3")
        got = raw_radii_cells(rho, coords[todo], reach_cells=RAW_REACH_FRAC * n_grid)
        del rho
        for k in ("r_cells", "never_below", "capped", "min_enclosed"):
            have[k][todo] = got[k]
        have["measured"][todo] = True
        pipeline._save_npz(path, settings=want, coords_sha1=np.array(chash), grid_stamp=stamp, **have)
    out = dict(have)
    out["r_mpc"] = have["r_cells"] * float(cell)
    return out


def _raw_radii_rc(products, cat, rows, say) -> dict:
    """raw_radii(..., recentre=True): the uncentred radii first (their own cache: R0 sets the search ball), then the
    re-centred ones, cached as NAME_RC under the same key as the uncentred file but settings_rc()."""
    from . import catalog, pipeline
    base = raw_radii(products, cat, rows, say)                  # R0 (cached); 'measured' = the rows asked for
    cell, _box, n_grid, _src = catalog.frame_of(products)
    coords = np.asarray(cat["coords"], dtype=np.int64)
    nrow = len(coords)
    want_rows = np.asarray(base["measured"], dtype=bool)
    if rows is not None:
        sel = np.zeros(nrow, bool)
        rows = np.asarray(rows)
        sel[rows if rows.dtype == bool else rows.astype(np.int64)] = True
        want_rows = want_rows & sel
    elif "well_resolved" in cat:
        want_rows = want_rows & np.asarray(cat["well_resolved"]).astype(bool)
    want = settings_rc()
    stamp = np.asarray(products._grid_stamp(), dtype=np.int64)
    chash = coords_hash(coords)
    path = products._cpath(NAME_RC)
    keys = ("r_cells", "never_below", "capped", "measured", "min_enclosed", "coords_rc", "offset_cells", "rule")
    have = None
    if path.exists():
        try:
            with np.load(path) as z:
                if _key_ok(z, stamp, chash, want) and all(k in z for k in keys):
                    have = {k: np.array(z[k]) for k in keys}
                else:
                    say(f"    [raw_radius] cached {path.name} is for another grid, catalogue or settings: re-measuring")
        except Exception as e:                            # a cut or foreign file: re-measured, said
            say(f"    [raw_radius] cached {path.name} is unreadable ({type(e).__name__}): re-measuring")
    if have is None or len(have["r_cells"]) != nrow:
        have = {"r_cells": np.full(nrow, np.nan), "never_below": np.zeros(nrow, bool), "capped": np.zeros(nrow, bool),
                "measured": np.zeros(nrow, bool), "min_enclosed": np.full(nrow, np.nan), "coords_rc": coords.copy(),
                "offset_cells": np.full(nrow, np.nan), "rule": np.zeros(nrow, np.int8)}
    todo = np.nonzero(want_rows & ~have["measured"])[0]
    if todo.size:
        rho = getattr(products, "_ram", {}).get("density_raw")
        if rho is None:
            rho = products.fs.density(units="mean")
        if rho.shape[0] != n_grid:
            raise ValueError(f"density grid is {rho.shape[0]}^3, the catalogue's frame {n_grid}^3")
        fb = np.asarray(cat["r_v_cells"], dtype=np.float64)[todo] if "r_v_cells" in cat else None
        got = recentred_radii_cells(rho, coords[todo], np.asarray(base["r_cells"])[todo], fb)
        del rho
        for k in ("r_cells", "never_below", "capped", "min_enclosed", "coords_rc", "offset_cells", "rule"):
            have[k][todo] = got[k]
        have["measured"][todo] = True
        pipeline._save_npz(path, settings=want, coords_sha1=np.array(chash), grid_stamp=stamp, **have)
    out = dict(have)
    out["r_mpc"] = have["r_cells"] * float(cell)
    out["r0_mpc"] = np.asarray(base["r_cells"]) * float(cell)
    return out


def raw_radius_mpc(products, cat, rows=None, say=print) -> np.ndarray:
    """The 'delta08raw' radius in Mpc per catalogue row (raw_radii): NaN where undefined or not measured."""
    return raw_radii(products, cat, rows, say)["r_mpc"]
