"""Resolution and estimator convergence of the void statistics (2026-10-11; plot/plot_void_convergence.py draws it).

Are the R_v distribution, the distinct-void abundance and the stacked profiles a property of the voids, or of the
grid / particle mass / field estimator they were measured with? Three comparisons, all at ONE physical smoothing
length (the void catalogue's sigma in Mpc), each with the pipeline's own definitions (pipeline.build_void_catalog:
minima, Hessian, ellipsoid fits and gate, the three R_v; catalog.distinct_mask: the abundance sample):

  * GRID resolution: the snapshot's density grid (rho / rho_bar, a cell AVERAGE for both estimators) block-averaged
    by a factor f (512^3 -> 256^3, 128^3) is exactly the cell average on the coarser grid, so a catalogue built from
    it (coarse_catalog, at the same sigma in Mpc, the footprint following it as for --smooth-mpc) is what a coarser
    run would give from the same particles. Matched void by void with the native catalogue (match).
  * MASS resolution: different boxes (TNG50-3 / TNG100-3 / TNG300-3) at a common sigma; distribution level only
    (different initial conditions), from plot_void_crossbox.py's per-box products.
  * ESTIMATOR: DTFE against PS-DTFE on the same snapshot (both grids present), matched void by void (match).

Matching uses dtfelib.voids.match_catalog_void (periodic nearest centre), within a radius in smoothing lengths;
it is done both ways, so the reference voids WITHOUT a partner (lost by the test) and the test voids without one
(born of it) are both counted, and 'mutual' marks pairs that are each other's nearest.

The coarse catalogue cannot go through SnapshotProducts.voids(): every catalogue function reads the frame (cell,
grid, sigma in cells, footprint) from config at call time and voids() refuses a grid that is not config's frame.
frame() sets that frame for the coarse grid and restores it whatever happens; the coarse catalogue is cached under
the coarse frame's own namespace (pipeline.cache_dir() inside frame(): 'r256_s5_f5_...'), named '..._voids_block<f>'
so it never poses as a real 256^3 run's, with the native density grid's stamp (rebuilt when that grid changes).
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

import numpy as np

import config
from . import pipeline, catalog, profiles
from . import fields as dtfe
from .voids import match_catalog_void

FRAME_KEYS = ("FIELD_RESOLUTION", "CELL_SIZE", "SMOOTHING_SIGMA_CELLS", "SMOOTHING_SIGMA_MPC", "FOOTPRINT_SIZE")
DEFAULT_FACTORS = (2, 4)          # 512^3 -> 256^3, 128^3
DEFAULT_MATCH_SIGMA = 1.0         # match radius in smoothing lengths: distinct voids lie >= min(R_v, R_eff) ~ 2-7 sigma
                                  # apart, while a coarser grid moves a minimum by about a coarse cell (<= 0.4 sigma at f=4)


@contextlib.contextmanager
def frame(n_grid: int, box_mpc: float, sigma_mpc: float):
    """config's catalogue frame set to a grid of n_grid cells over box_mpc, smoothed at sigma_mpc (in cells of THAT
    grid; footprint config.footprint_for, as pipeline.set_smoothing(mpc=...)); everything restored on exit."""
    saved = {k: getattr(config, k) for k in FRAME_KEYS}
    try:
        config.FIELD_RESOLUTION = int(n_grid)
        config.CELL_SIZE = float(box_mpc) / int(n_grid)
        config.SMOOTHING_SIGMA_CELLS = round(float(sigma_mpc) / config.CELL_SIZE, 9)
        config.FOOTPRINT_SIZE = config.footprint_for(config.SMOOTHING_SIGMA_CELLS)
        config.SMOOTHING_SIGMA_MPC = float(sigma_mpc)
        yield
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def block_average(grid, factor: int) -> np.ndarray:
    """The mean over factor^3 blocks of a (N,N,N) or (N,N,N,C) grid (N divisible by factor), as float32 (summed in
    float64): a cell-average grid's coarse cell average, its mean preserved."""
    g = np.asarray(grid)
    f = int(factor)
    if f == 1:
        return g
    n = g.shape[0]
    if n % f or g.shape[1] != n or g.shape[2] != n:
        raise ValueError(f"block_average: a cubic grid divisible by {f}, got {g.shape}")
    m = n // f
    tail = g.shape[3:]
    return g.reshape((m, f, m, f, m, f) + tail).mean(axis=(1, 3, 5), dtype=np.float64).astype(np.float32)


VELOCITY_WEIGHTINGS = ("volume", "mass")
_VOLUME_FLAG = re.compile(r"(^|\s)--ps-volume-weighted(\s|$)", re.M)


def velocity_weighting(products) -> tuple[str, str]:
    """How the velocity grid of 'products' (a SnapshotProducts) was deposited, and how that is known: ('volume' |
    'mass', why). 2026-10-11: the coarse velocity must be the mean a coarser run of the SAME deposit gives -- the
    plain block mean for a volume-weighted grid, sum rho v / sum rho for a mass-weighted one -- and the production
    PS-DTFE grids are volume-weighted (runspec's default, the TNG runlogs), so mass-weighting them all compared
    the native v_r stack with coarse ones built another way. Standard DTFE's cell-averaged velocity is the volume
    average. A PS-DTFE run says it in its run log ('<prefix>runlog', the binary's command line: the header does
    not record it); without the log the production convention (volume) is assumed, and said."""
    fs = products.fs
    if fs.method != "ps":
        return "volume", "standard DTFE: the volume average"
    log = Path(fs.snapdir) / f"{fs.prefix}runlog"
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return "volume", f"no {log.name}: the production convention (--ps-volume-weighted) assumed"
    if _VOLUME_FLAG.search(text):
        return "volume", f"{log.name}: --ps-volume-weighted"
    return "mass", f"{log.name}: no --ps-volume-weighted (the binary's mass-weighted default)"


def block_average_velocity(velocity, density, factor: int, weighting: str = "volume") -> np.ndarray:
    """The coarse cell's velocity from a (N,N,N,3) velocity grid as a coarser run of the same deposit gives it:
    'volume' (a volume-weighted deposit, the production convention: velocity_weighting) the plain block mean;
    'mass' the MASS-weighted block mean (sum rho v / sum rho over each block; the plain mean where a block holds
    no mass)."""
    if weighting not in VELOCITY_WEIGHTINGS:
        raise ValueError(f"block_average_velocity: weighting must be one of {VELOCITY_WEIGHTINGS}, got {weighting!r}")
    f = int(factor)
    if f == 1:
        return np.asarray(velocity)
    if weighting == "volume":
        return block_average(velocity, f)
    rho = np.asarray(density, dtype=np.float32)
    mass = block_average(rho, f).astype(np.float64)
    mom = block_average(np.asarray(velocity, dtype=np.float32) * rho[..., None], f).astype(np.float64)
    plain = block_average(velocity, f)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(mass[..., None] > 0, mom / mass[..., None], plain)
    return out.astype(np.float32)


def build_catalog(density_mean, box_mpc: float, sigma_mpc: float, gate: bool = True) -> dict:
    """The void catalogue of a density grid (rho / rho_bar) of any size, at sigma_mpc: pipeline.build_void_catalog
    in that grid's frame (frame()), 'n_grid' and 'sigma_mpc' recorded; with gate the process's void gate applied
    (pipeline.gate_view; the production flags otherwise, which is what a cache keeps)."""
    rho = np.asarray(density_mean)
    n = int(rho.shape[0])
    with frame(n, box_mpc, sigma_mpc):
        delta_s = dtfe.smooth_field(dtfe.calculate_density_contrast(rho), sigma=config.SMOOTHING_SIGMA_CELLS)
        cat = pipeline.build_void_catalog(delta_s, pipeline.hessian_from_field(delta_s))
        if gate:
            cat = pipeline.gate_view(cat)
    cat["n_grid"] = np.int64(n)
    cat["sigma_mpc"] = np.float64(sigma_mpc)
    return cat


def gated(cat, box_mpc: float):
    """pipeline.gate_view of a catalogue in ITS frame (the cuts are in that grid's cells and sigma)."""
    with frame(int(cat["n_grid"]), box_mpc, float(cat["sigma_mpc"])):
        return pipeline.gate_view(cat)


def coarse_cache_path(products, factor: int, box_mpc: float, sigma_mpc: float, n_native: int) -> Path:
    """Where the factor-f block-averaged catalogue of 'products' is cached: the coarse frame's namespace."""
    tag = "" if products.prefix is None else f"{str(products.prefix).rstrip('.')}_"
    with frame(int(n_native) // int(factor), box_mpc, sigma_mpc):
        return pipeline.cache_dir() / f"{products.sim}_{tag}{products.fs.method}_{products.snapshot}_voids_block{int(factor)}.npz"


def coarse_catalog(products, factor: int, sigma_mpc: float, density=None, log=print) -> dict:
    """The catalogue of 'products' (a SnapshotProducts) block-averaged by 'factor', at sigma_mpc: from its cache when
    that carries the native density grid's stamp, else built (build_catalog) and cached. 'density': the native
    rho / rho_bar if already in memory (else products.density_raw)."""
    cell, box, n, _src = catalog.frame_of(products)
    p = coarse_cache_path(products, factor, box, sigma_mpc, n)
    stamp = products._grid_stamp()
    if p.exists():
        try:
            with np.load(p) as z:
                cat = {k: z[k] for k in z.files}
            with frame(int(n) // int(factor), box, sigma_mpc):              # radii and cuts as pipeline judges them
                current = pipeline.catalog_r_v_ok(cat) and pipeline.catalog_cuts_ok(cat)
            if stamp.size and np.array_equal(cat.get("grid_stamp", np.zeros(0)), stamp) and current:
                return gated(cat, box)
            log(f"    [convergence] cached {p.name} is from another version of the grid, the R_v settings or the cuts: "
                f"rebuilding it")
        except Exception as e:                                          # an unreadable cache: rebuilt, as pipeline's
            log(f"    [convergence] cached {p.name} is unreadable ({type(e).__name__}): rebuilding it")
    rho = products.density_raw if density is None else density
    cat = build_catalog(block_average(rho, factor), box, sigma_mpc, gate=False)    # the cache keeps the production gate
    pipeline._save_npz(p, **dict(cat, grid_stamp=stamp))
    return gated(cat, box)


def distinct_rows(cat, n_grid: int, definition: str = "crossing") -> np.ndarray:
    """Rows of the abundance sample: well resolved, with an R_v under 'definition', distinct (catalog.distinct_mask)
    -- plot_void_population.snapshot_stats's 'counted'."""
    r = catalog.r_v_mpc(cat, definition)
    sel = np.asarray(cat["well_resolved"]).astype(bool) & np.isfinite(r)
    return np.nonzero(catalog.distinct_mask(cat, sel, int(n_grid)))[0]


def centres_frac(cat, rows, n_grid: int) -> np.ndarray:
    return (np.asarray(cat["coords"], dtype=np.float64)[rows] + 0.5) / int(n_grid)


def match(ref_cat, ref_rows, ref_n: int, test_cat, test_rows, test_n: int, box_mpc: float, max_dist_mpc: float) -> dict:
    """Void-by-void matching of the reference sample (ref_rows of ref_cat, an n_grid ref_n catalogue) to the test
    sample, by centre (voids.match_catalog_void: periodic nearest, within max_dist_mpc). Per reference void:
    'test' (row of test_cat or -1), 'dist_mpc' (to the nearest test centre, matched or not), 'mutual' (the test void's
    nearest reference void is this one), the two crossing R_v and central deltas ('r_v_ref', 'r_v_test', NaN when
    unmatched; 'delta_ref', 'delta_test'); and 'test_matched' (bool per test row: some reference void matched it)."""
    ref_rows, test_rows = np.asarray(ref_rows, int), np.asarray(test_rows, int)
    out = {"ref": ref_rows, "test": np.full(ref_rows.size, -1), "dist_mpc": np.full(ref_rows.size, np.nan),
           "mutual": np.zeros(ref_rows.size, bool), "test_rows": test_rows,
           "test_matched": np.zeros(test_rows.size, bool), "max_dist_mpc": float(max_dist_mpc)}
    r_ref, r_test = catalog.r_v_mpc(ref_cat), catalog.r_v_mpc(test_cat)
    out["r_v_ref"] = r_ref[ref_rows]
    out["r_v_test"] = np.full(ref_rows.size, np.nan)
    out["delta_ref"] = np.asarray(ref_cat["delta_values"], np.float64)[ref_rows]
    out["delta_test"] = np.full(ref_rows.size, np.nan)
    if not ref_rows.size or not test_rows.size:
        return out
    sub_test = {"coords": np.asarray(test_cat["coords"])[test_rows]}
    sub_ref = {"coords": np.asarray(ref_cat["coords"])[ref_rows]}
    lim_t = float(max_dist_mpc) / float(box_mpc)
    c_ref, c_test = centres_frac(ref_cat, ref_rows, ref_n), centres_frac(test_cat, test_rows, test_n)
    # match_catalog_void reads each side's cell coords on its OWN grid (fractional centres): grids may differ
    for i, c in enumerate(c_ref):
        j, d = match_catalog_void(sub_test, c, int(test_n), lim_t)
        out["dist_mpc"][i] = d * float(box_mpc)
        if j is None:
            continue
        out["test"][i] = test_rows[j]
        out["test_matched"][j] = True
        out["r_v_test"][i] = r_test[test_rows[j]]
        out["delta_test"][i] = float(np.asarray(test_cat["delta_values"])[test_rows[j]])
        back, _ = match_catalog_void(sub_ref, c_test[j], int(ref_n), np.inf)
        out["mutual"][i] = back == i
    return out


def match_summary(m: dict, sigma_mpc: float) -> dict:
    """The numbers of one match: counts, the matched and unmatched fractions both ways, the R_v ratio test / ref
    (16/50/84%), the centre offset (median, in Mpc and sigma), the mutual fraction."""
    ok = np.asarray(m["test"]) >= 0
    ratio = np.asarray(m["r_v_test"])[ok] / np.asarray(m["r_v_ref"])[ok]
    ratio = ratio[np.isfinite(ratio)]
    off = np.asarray(m["dist_mpc"])[ok]
    n_ref, n_test = int(ok.size), int(np.asarray(m["test_matched"]).size)
    return {"n_ref": n_ref, "n_test": n_test, "n_matched": int(ok.sum()), "n_mutual": int(np.asarray(m["mutual"]).sum()),
            "frac_ref_unmatched": float(1.0 - ok.mean()) if n_ref else np.nan,
            "frac_test_unmatched": float(1.0 - np.asarray(m["test_matched"]).mean()) if n_test else np.nan,
            "ratio_pct": np.percentile(ratio, (16, 50, 84)) if ratio.size else np.full(3, np.nan),
            "offset_med_mpc": float(np.median(off)) if off.size else np.nan,
            "offset_med_sigma": float(np.median(off)) / float(sigma_mpc) if off.size else np.nan,
            "delta_diff_med": float(np.nanmedian(np.asarray(m["delta_test"])[ok] - np.asarray(m["delta_ref"])[ok]))
            if ok.any() else np.nan}


def stacked_profile(density, cat, rows, n_grid: int, box_mpc: float, edges_mpc, velocity=None,
                    m: int = profiles.DEFAULT_M, n_boot: int = 200, bins=profiles.DEFAULT_BINS) -> dict:
    """profiles.profile_grids on the given grids (rho / rho_bar, optional velocity; the grid of THIS resolution) for
    the voids 'rows' of 'cat' in r / R_v (crossing), stacked over 'edges_mpc' (profiles.stack): what
    profiles.void_profiles + stack give for a SnapshotProducts, for a grid that is not one."""
    r = catalog.r_v_mpc(cat)[rows]
    cell = float(box_mpc) / int(n_grid)
    grids = {"density": density, "velocity": velocity, "single": None}
    prof = profiles.profile_grids(grids, centres_frac(cat, rows, n_grid), r / cell, bins=bins, m=m)
    prof["r_mpc"] = r[prof["index"]]
    st = profiles.stack(prof, edges_mpc=edges_mpc, n_boot=n_boot)
    st["profiled"], st["dropped"] = len(prof["index"]), int(prof["dropped"])
    return st
