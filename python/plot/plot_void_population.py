"""POPULATION-level void statistics across cosmic time: counts, sizes, shapes, depths.

Complements plot_void_tracking.py (deep-dives on individual voids) with the population
answer: what did ALL voids do between z=20 and z=0? Uses the same pipeline catalogs
(pipeline.SnapshotProducts.voids(): smoothing, FFT Hessian, footprint, criterion, BBKS
shapes, ellipsoid cuts identical to the correlation analyses) and the same sample
definitions:

  * 'bbks'      : the correlation sample -- finite e/p with e >= 0 and |p| <= e
                  (identical to void_sample() in plot_marked_correlation_BBKS.py)
  * 'resolved'  : well_resolved (passes config.ELLIPSOID_CUT_RULES) -- sizes/orientation defined
  * 'deep'      : well_resolved AND central delta < DEEP_VOID_THRESHOLD -- the tracking sample

Outputs (figures/void_population/<sim>/):
  population_evolution.png    counts, R_eff and the distinct voids' R_v, e, p, central delta vs redshift
  population_distributions.png  R_eff / e / p / delta histograms at config.PANEL_SNAPSHOTS
  population_abundance.png    n(> R_v) and dn/dlnR_v per Mpc^3 (distinct well-resolved voids, Poisson errors) at
                              config.PANEL_SNAPSHOTS -- with --other-estimator also the other estimator's
                              catalogue, dashed -- and R_v against the central delta at the latest one
  population_size_function.png  the void size function (2026-10-11): measured dn/dlnR_v and n(>R_v) of the distinct
                              voids at the panel snapshots against the Vdn model at each z (dtfelib.theory: Sheth &
                              van de Weygaert 2004 in the volume-conserving form of Jennings, Li & Hu 2013, delta_v
                              -2.71 linear / -0.8 nonlinear, delta_c 1.686, EH98 sigma(R) at TNG's Planck 2015
                              sigma_8, D(z) of flat LambdaCDM), and their ratio. R_v is --size-rv's definition
                              ('delta08' by default: where the ENCLOSED Delta(<R) reaches -0.8, the model's own
                              radius; 'crossing', said, while the catalogue lacks it). Measured on its OWN
                              catalogue, smoothed less (--size-smooth, default 5 cells; 2026-10-11): at the production
                              10 cells the delta08 radius lies below sigma and exists for only 31-47% of the distinct
                              voids. That png is named population_size_function_ss<cells>.png
                              (untagged only with --size-smooth 0: the figures' own catalogue); a non-default
                              --size-smooth also tags population_stats '_ss<cells>'.
                              --size-rv delta08raw (2026-10-11): the same -0.8 rule on the RAW density about the
                              same centres (dtfelib.raw_radius, cached beside the catalogue), png and npz tagged
                              '_delta08raw'. Defined for 73-100% of the distinct voids (vs 1-45%), TNG100 and TNG300
                              agree at 2.1625 Mpc, measured/Vdn 0.04-0.17 at the median; but the sample's median
                              radius still follows the smoothing (TNG100 z = 0: 4.4 / 2.4 / 1.5 Mpc at 10 / 5 / 3
                              cells) and the ellipsoid gate cuts the large voids, so NEITHER radius makes the
                              comparison a test of Vdn and 'delta08' stays the default (the figure says why;
                              SIZE_FUNCTION_CAVEAT holds the numbers). Ungated (DTFE_VOID_GATE=none), the raw
                              abundance at fixed R above ~3 sigma converges with the smoothing (measured/Vdn at
                              8 Mpc, z = 0: 0.8-0.9 at 10, 5 and 3 cells)
  population_size_function_fixedR.png (+ .npz)  --size-fixed-r ONLY (2026-10-11; DTFE_VOID_GATE=none required, written
                              into the 'gatenone' folder): the size function at FIXED radius -- n(>R) and dn/dlnR of
                              the raw -0.8 radius of the distinct ungated voids from catalogues at several smoothings
                              (--fixed-r-smooth, default 10 5 3 cells; --fixed-r-smooth-mpc in Mpc), only over the
                              common complete range (3 x the largest sigma up to the radius with 5 voids above it in
                              the scarcest smoothing, at least x1.15 wide; --fixed-r-range sets it), against Vdn with
                              Poisson and 3^3 sub-cube bootstrap errors, the ratio per bin and over the range, its
                              scatter across smoothings and a verdict per z (z = 0, 0.5, 1, 2; the smoothings compared
                              on the PAIRED bootstrap, 2026-10-11). Non-default smoothings or a range tag the name:
                              _c3-2_m2.1625, _R6.5-9. RESULT (TNG100 at 10/5/3 cells, TNG300 at 2.1625 Mpc/3/2 cells):
                              at z = 0 TNG100 0.84-1.01 of Vdn (consistent, no significant difference between its
                              smoothings), TNG300 0.58-0.84 (4% errors, still following the smoothing). Above Vdn at
                              z >= 0.5 ONLY at the larger smoothings (z = 0.5: TNG100 10 cells 2.16, TNG300 2.1625 Mpc
                              / 3 cells 1.69 / 1.42; z = 1 TNG300 3.55 / 2.40); at the smallest smoothing in each box
                              the excess is gone at z = 0.5 (TNG100 5 / 3 cells 0.90 / 1.17, TNG300 2 cells 0.95), so
                              the high-z excess depends on the smoothing, and the smoothings disagree at every z but
                              TNG100 z = 0 (numbers below FIXED_R_SNAPS).
                              --fixed-r-recentre (2026-10-12): each void re-centred on the raw density first
                              (dtfelib.raw_radius RE-CENTRING), png/npz tagged '_rc'; --fixed-r-distinct raw: the
                              distinct voids by their raw spheres (largest first, no overlap), tagged '_draw'. Neither
                              is the default (measured below FIXED_R_SNAPS: re-centring removes most of TNG300's
                              smoothing dependence but halves the z = 0 count and breaks TNG100 z = 0)
The figures are in the house style (dtfelib.figures: serif, config.DPI, titles only with config.SHOW_TITLES) and
mirrored like the thesis figures (2026-10-11). The counts of resolved and distinct voids against z are set by the
ellipsoid gate (the 10 sigma max-axis cut turns on as the voids grow): the evolution figure draws the minima with an
R_v (no gate) beside them and says so.
Those names are PS-DTFE's default grids (ps_output.); another estimator or --prefix adds '_<method>' / '_<prefix>',
--keep-overlaps '_overlaps', to every name (2026-10-08: a --method dtfe run replaced the PS-DTFE files), and a
non-default smoothing writes into its own folder.

Examples:
  python3 plot/plot_void_population.py --sim TNG50-3-Dark --method dtfe
  python3 plot/plot_void_population.py --sim TNG50-4-Dark --method ps
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import contextlib
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import config
from dtfelib import pipeline, catalog, theory, raw_radius
from dtfelib import make_parser, use_data_root
from dtfelib import figures as style
from dtfelib import environment as env
from dtfelib import groupcat as gc

FIGURE_ROOT = Path(config.LOCAL_FIGURES_ROOT) / "void_population"
PCT = (16, 50, 84)   # band percentiles
SIZE_FUNCTION_RV = "delta08"     # the R_v definition of the size function (catalog.R_V_DEFINITIONS, contract C3): the
                                 # theory's radius is where the enclosed contrast reaches DELTA_V_NONLINEAR = -0.8
RAW_RV = "delta08raw"            # the enclosed -0.8 radius on the RAW density about the same centres (dtfelib.raw_radius,
                                 # 2026-10-11): not a catalogue key, measured here (cached beside the catalogue)
SIZE_RV_CHOICES = ("crossing", "delta08", "ridge", RAW_RV)
# Why the size function is NOT a test of Vdn with either -0.8 radius, said on the figure under the definition drawn
# (2026-10-11, rawradius; TNG100-3-Dark z = 0, 0.5, 1, 2 and TNG300-3-Dark z = 0, 1, catalogues at 10 / 5 / 3 cells):
#   delta08    (smoothed field) a smoothing length: median 0.7-1.06 sigma at 10/5/3/2 cells, defined for 1-45% of the
#              distinct voids, measured/Vdn 1e-3-1e-2.
#   delta08raw (raw density, dtfelib.raw_radius) defined for 73-100% of the distinct voids (z <= 1: 94-100%), TNG100
#              and TNG300 agree at the common 2.1625 Mpc (z = 0 median 4.40 / 4.37 Mpc, measured/Vdn 0.15 / 0.13;
#              z = 1 3.16 / 3.19 Mpc, 0.10 / 0.11), measured/Vdn 0.04-0.17 at the median (z <= 1). BUT the sample's
#              median still follows the smoothing that found the centres: TNG100 z = 0 4.40 / 2.44 / 1.53 Mpc at
#              10 / 5 / 3 cells (d ln R / d ln sigma 0.88; TNG300 0.69), because each smoothing finds voids down to
#              ~2 sigma and the ellipsoid gate (max semi-axis 10 sigma) removes the large ones. So the default stays
#              'delta08' (the rule: delta08raw only if it is defined for >= 50% AND does not scale with sigma).
#              Ungated (DTFE_VOID_GATE=none) the abundance at fixed R above ~3 sigma converges: measured/Vdn at
#              R = 8 Mpc, z = 0, 0.81 / 0.79 / 0.91 at 10 / 5 / 3 cells -- the one place it is close to a test.
SIZE_FUNCTION_CAVEAT = {
    "delta08": "not a test of Vdn: the smoothed-field -0.8 radius is a smoothing length (median 0.7-1.1 sigma at "
               "10, 5, 3, 2 cells)",
    RAW_RV: "not a test of Vdn: the sample's median radius follows the smoothing that found the centres "
            "(TNG100 z = 0: 4.4, 2.4, 1.5 Mpc at 10, 5, 3 cells) and the ellipsoid gate removes the large voids; "
            "ungated, measured/Vdn at 8 Mpc (z = 0) is 0.8-0.9 at 10, 5 and 3 cells",
}
SIZE_FUNCTION_MODEL = "vdn"      # theory.void_size_function's model ('vdn' or 'svdw')
RATIO_HALF_WIDTH = 0.25          # the measured/theory ratio at the median R_v: counts in ln R within +-0.25 of it
RATIO_SHOWN = (1e-3, 1e3)        # the ratio panel's range (a high-z ratio is astronomically large: printed, not drawn)
SIZE_SMOOTH_CELLS = 5.0          # the size function's catalogue smoothing in cells (--size-smooth, 2026-10-11): at the
                                 # production 10 cells the delta08 radius (median 1.56 Mpc TNG100 z = 0) is 0.7 sigma,
                                 # below the smoothing that sets it, and defined for 31% of the distinct voids. Half
                                 # the smoothing keeps sigma ~5 mean interparticle spacings (TNG100 0.24 Mpc = 1.1
                                 # cells), far from particle noise, with the production footprint (11 cells: no new
                                 # noise sub-minima), and gives 3.6x the voids (91 vs 25 at z = 0, 38% defined).
                                 # MEASURED (TNG100, 10/5/3/2 cells): the median delta08 radius follows sigma (0.7,
                                 # 1.06, 1.03, 0.91 sigma) and measured/Vdn stays 1e-3-1e-2: the -0.8 sphere of a
                                 # minimum is a smoothing length, not a converged void size
MIRROR = True                    # into the thesis Figures/ tree as well (dtfelib.figures; DTFE_NO_MIRROR turns it off)
# The size function at FIXED radius (--size-fixed-r, 2026-10-11, sizefn): the raw -0.8 radius (dtfelib.raw_radius) of
# the distinct UNGATED voids from catalogues at several smoothings, compared only where every smoothing is complete. The
# lead (rawradius, same day): ungated, the abundance at fixed R above ~3 sigma stopped depending on the smoothing and
# measured/Vdn at 8 Mpc, z = 0 was 0.8-0.9 at 10, 5 and 3 cells; this figure makes that a test per z. MEASURED
# (measured/Vdn over the common range from 6.49 Mpc; +- the larger of Poisson and bootstrap; the smoothings compared on
# the PAIRED sub-cube bootstrap, chi^2 for 2 dof, 95% point 6.0 -- rerun 2026-10-11 after review-science):
#   TNG100 10 / 5 / 3 cells: z = 0 0.93 / 0.84 / 1.01 +- 0.15 (paired chi^2 1.1: no significant difference; consistent
#     with Vdn); z = 0.5 2.16 / 0.90 / 1.17 +- 0.3-0.4 (paired chi^2 14.3: disagree); z = 1 and 2 no common range
#     (18 / 8 / 3 and 0 voids above 6.49 Mpc).
#   TNG300 2.1625 Mpc / 3 / 2 cells (1.77 / 1.18 Mpc; at 10 / 5 / 3 cells 3 sigma is 17.7 Mpc: no range at any z):
#     z = 0 0.84 / 0.78 / 0.58 +- 0.03-0.04, z = 0.5 1.69 / 1.42 / 0.95, z = 1 3.55 / 2.40 / 1.59, z = 2 no range
#     (paired chi^2 80 / 119 / 112: disagree at every z).
#   Until 2026-10-11 the last dn/dlnR bin of the scarcest smoothing also counted the void at r_hi (np.histogram's closed
#   last bin): its per-bin ratio was up to 2x high (TNG100 z = 0.5, 5 cells: 1.68, now 0.84); the whole-range ratios
#   above never included it.
#   Cross-box at 2.1625 Mpc, R 6.5-9 Mpc: z = 0 0.81 +- 0.15 / 0.78 +- 0.05, z = 0.5 2.32 +- 0.40 / 1.64 +- 0.08, z = 1
#     4.9 +- 1.3 / 3.55 +- 0.22 (TNG100 / TNG300).
# So the fixed-R abundance converges with sigma only to TNG100's 15%: at TNG300's 4% the smaller the smoothing the
# FEWER voids above a fixed R at every z (untested guess: a large void's deepest minimum at small sigma sits off
# its centre, so the raw -0.8 sphere about it is smaller). The excess above Vdn at z >= 0.5 is itself
# smoothing-dependent, NOT one factor: it holds at the larger smoothings (z = 0.5 TNG100 10 cells 2.16, TNG300
# 2.1625 Mpc / 3 cells 1.69 / 1.42; z = 1 TNG300 3.55 / 2.40, rising with z) but is gone at the smallest smoothing in
# each box at z = 0.5 (TNG100 5 / 3 cells 0.90 / 1.17, TNG300 2 cells 0.95; at z = 1 TNG300 2 cells is still 1.59).
# THE OFF-CENTRE GUESS TESTED (2026-10-12, recentre; z = 0 / 0.5 / 1, same smoothings, paired chi^2 in brackets):
#   Mechanism, from matching the voids above r_lo of the largest smoothing to the smaller one's minima inside their
#   raw spheres (TNG300 z = 0, 765 voids at 2.1625 Mpc; at 2 cells): 311 still a distinct void above r_lo, 243 only
#   minima with SMALLER raw spheres (the off-centre case), 208 a minimum with a large sphere that catalog.distinct_mask
#   dropped for a deeper, smaller one beside it. So the guess holds for about half of the deficit.
#   --fixed-r-recentre (minimum of the 1-cell-smoothed raw density within 0.5 R0, iterated): the centres move by a
#   median 0.9-1.6 sigma in every box, but the -0.8 sphere about the raw minimum is SMALLER for 70-88% of the voids
#   (median R/R0 0.81-0.93): the emptiest point is not the centre of the largest empty sphere.
#     TNG300 0.45/0.47/0.51 [7.4], 0.88/0.89/0.89 [0.1], 1.60/1.45/1.43 [5.9]   (was [80], [119], [112])
#     TNG100 0.53/0.84/1.07 [10.3] at z = 0 (was 0.93/0.84/1.01 [1.1]); no common range at z >= 0.5
#     TNG50  0.38/0.47/0.59 [4.2], 0.58/0.68/0.84 [4.7], 0.70/0.58/1.22 [11.8] (was z = 0 0.46/0.56/0.75)
#   It maps every catalogue onto nearly the same raw minima, so TNG300's smoothing dependence mostly goes -- but the
#   amplitude drops (z = 0 half of Vdn), TNG100 z = 0 loses its agreement and TNG50 z = 0 moves away from 1. Against
#   the bar set before measuring (TNG300 z = 0 stops rejecting, TNG100 z = 0 stays consistent, TNG50 z = 0 moves
#   toward 1) it fails all three: an option, not the default. One pass instead of up to 8 gives the same ratios
#   (TNG300 z = 0 0.47/0.48/0.51 with chi^2 3.9, so it passes the first bar; the other two fail the same way).
#   --fixed-r-distinct raw (largest raw sphere first, no overlap): TNG50 z = 0 0.93/0.84/0.87 [0.7] (consistent with
#   Vdn), TNG100 z = 0 1.45/1.32/1.32 [1.0] (converged, 1.36x Vdn), TNG300 1.58/1.37/1.07 [316] and z >= 0.5 2-5x Vdn
#   with the smoothings apart: no fix for TNG300 either. With the centre rule instead of no overlap it was 2-4x Vdn.
#   So the fixed-R abundance depends on WHICH point stands for a void and WHICH of a region's minima count, at the
#   level of the TNG300 smoothing dependence; neither choice here removes it while keeping the amplitude.
FIXED_R_RECENTRE = False                 # re-centre each void on the raw density before its -0.8 sphere is grown
                                         # (raw_radius RE-CENTRING; --fixed-r-recentre / --fixed-r-no-recentre;
                                         # the png/npz say the non-default choice: '_rc' / '_norc')
FIXED_R_DISTINCT = "catalogue"           # which voids are distinct: 'catalogue' (catalog.distinct_mask, the smoothed
                                         # field's deepest minimum first) or 'raw' (raw_sphere_distinct, the largest raw
                                         # -0.8 sphere first); --fixed-r-distinct, the non-default tags '_draw' / '_dcat'
FIXED_R_DISTINCT_CHOICES = ("catalogue", "raw")
FIXED_R_SNAPS = (99, 67, 50, 33)         # z = 0, 0.5, 1, 2 (--panel-snaps overrides)
FIXED_R_SMOOTH_CELLS = (10.0, 5.0, 3.0)  # the catalogues' smoothings (--fixed-r-smooth; --fixed-r-smooth-mpc in Mpc)
FIXED_R_MIN_SIGMA = 3.0                  # the common range starts at this many LARGEST sigma: below it the voids of the
                                         # most smoothed catalogue are missing (each smoothing finds voids down to ~2 sigma)
FIXED_R_MIN_COUNT = 5                    # ... and ends where the scarcest smoothing still has this many voids above R:
                                         # beyond, the box holds too few voids for a comparison (the box-limited end)
FIXED_R_MIN_WIDTH = 1.15                 # ... and spans at least this factor in R, else it is empty (TNG300 z = 2 gave
                                         # 6.49-6.59 Mpc with 2 and 5 voids: a 'range' that tests nothing; TNG100
                                         # z = 0.5, 6.49-7.85 Mpc with 10-24 voids, still tests)
FIXED_R_BINS = 6                         # ln R bins across the common range
FIXED_R_NSUB = 3                         # bootstrap: the box cut into NSUB^3 sub-cubes, resampled with replacement
FIXED_R_NBOOT = 500                      # (sample variance of a clustered population, beside Poisson's sqrt N)
FIXED_R_SEED = 20261011
FIXED_R_CONSISTENT = 2.0                 # 'consistent with Vdn': |ratio - 1| within this many errors at every smoothing


def r_v_definition(cat, definition="crossing"):
    """(R_v in Mpc per catalogue row, the definition used): catalog.r_v_mpc under 'definition' (contract C3). A
    catalogue without that key, or a catalog module without the definitions, gives 'crossing' (today's R_v)
    instead; the caller says so. A catalogue WITH the key keeps it even when every row is NaN: at high z no minimum
    of the smoothed field is as empty as -0.8, so there are no such voids, which is a result, not a missing key."""
    key = getattr(catalog, "R_V_KEYS", {}).get(definition)
    if definition != "crossing" and key is not None and key in cat:
        try:
            return np.asarray(catalog.r_v_mpc(cat, definition=definition), dtype=np.float64), definition
        except TypeError:                          # a catalog.r_v_mpc from before the definitions
            pass
    return catalog.r_v_mpc(cat), "crossing"


def ratio_at(r_v, box_mpc, r0, z, half=RATIO_HALF_WIDTH, model=SIZE_FUNCTION_MODEL):
    """measured / theory dn/dlnR in the ln R window r0 exp(+-half): (ratio, Poisson error, count). The theory is
    averaged over the same window (it falls by decades across a wide bin)."""
    r = np.asarray(r_v, dtype=np.float64)
    r = r[np.isfinite(r)]
    if not (np.isfinite(r0) and r0 > 0 and np.isfinite(z)):
        return np.nan, np.nan, 0
    n = int(((r >= r0 * math.exp(-half)) & (r < r0 * math.exp(half))).sum())
    grid = r0 * np.exp(np.linspace(-half, half, 41))
    th = np.float64(np.mean(theory.void_size_function(grid, z, model)))    # 0 at high z: an inf ratio, not a raise
    meas = n / (float(box_mpc) ** 3 * 2.0 * half)
    return meas / th, math.sqrt(max(n, 1)) / (float(box_mpc) ** 3 * 2.0 * half) / th, n


def bbks_valid(cat):
    """The correlation scripts' validity mask (plot_marked_correlation_BBKS.void_sample)."""
    e, p = cat["bbks_params"][:, 0], cat["bbks_params"][:, 1]
    return np.isfinite(e) & np.isfinite(p) & (e >= 0) & (np.abs(p) <= e)


def snapshot_stats(cat, n_grid=None, keep_overlaps=False, size_rv=SIZE_FUNCTION_RV, raw=None):
    """Per-snapshot population statistics from one pipeline void catalog. The abundance sample (2026-10-07): the
    resolved voids with a measured radius R_v (catalog.r_v_mpc), only the DISTINCT ones (catalog.distinct_mask) unless
    keep_overlaps -- neighbouring minima of one void counted once. The size function's sample (2026-10-11): the same
    with R_v under 'size_rv' (r_v_definition; '_r_v_sf', its definition in 'rv_sf_definition'). 'n_rv_all' counts
    every minimum with an R_v, no ellipsoid gate and no de-duplication: the reference beside the gated counts.
    size_rv 'delta08raw' (2026-10-11) takes its radii from 'raw' (raw_radius.raw_radii of this catalogue: the
    enclosed -0.8 radius on the RAW density, which the catalogue does not hold); without it the size function's
    sample is the crossing one, said by the figure as for a catalogue without a definition."""
    valid = bbks_valid(cat)
    resolved = cat["well_resolved"].astype(bool)
    deep = resolved & (cat["delta_values"] < config.DEEP_VOID_THRESHOLD)

    e, p = cat["bbks_params"][:, 0], cat["bbks_params"][:, 1]
    r_eff = catalog.r_eff_mpc(cat)          # (a b c)^(1/3) where well resolved, NaN elsewhere: the one definition
    r_v = catalog.r_v_mpc(cat)              # the measured radius (NaN for a catalogue built before it)
    with_rv = resolved & np.isfinite(r_v)
    counted = with_rv if keep_overlaps else catalog.distinct_mask(cat, with_rv, int(n_grid or config.FIELD_RESOLUTION))
    # said, not silent (review 2026-10-07): the resolved voids without R_v (no crossing within 15 sigma: the larger,
    # deeper ones -- the large-R end of n(>R_v) is low by them) and those inside a deeper void's region
    if size_rv == "crossing" or (size_rv == RAW_RV and raw is None):
        r_sf, sf_def, sf_sel = r_v, "crossing", counted
    elif size_rv == RAW_RV:
        r_sf, sf_def = np.asarray(raw["r_mpc"], dtype=np.float64), RAW_RV
        with_sf = resolved & np.isfinite(r_sf)
        sf_sel = with_sf if keep_overlaps else catalog.distinct_mask(cat, with_sf, int(n_grid or config.FIELD_RESOLUTION))
    else:
        r_sf, sf_def = r_v_definition(cat, size_rv)
        with_sf = resolved & np.isfinite(r_sf)
        sf_sel = with_sf if keep_overlaps else catalog.distinct_mask(cat, with_sf, int(n_grid or config.FIELD_RESOLUTION))

    def pct(x, sel):
        x = np.asarray(x, dtype=float)[sel]
        x = x[np.isfinite(x)]
        return np.percentile(x, PCT) if x.size else np.full(3, np.nan)

    return {
        "n_total": int(len(cat["coords"])),
        "n_bbks": int(valid.sum()),
        "n_resolved": int(resolved.sum()),
        "n_deep": int(deep.sum()),
        "n_counted": int(counted.sum()),
        "n_no_rv": int((resolved & ~np.isfinite(r_v)).sum()),
        "n_overlapping": int(with_rv.sum() - counted.sum()),
        "n_rv_all": int(np.isfinite(r_v).sum()),
        "n_sf": int(sf_sel.sum()),
        # the fraction of the distinct (crossing R_v) voids whose size_rv radius exists (2026-10-11): delta08 is NaN
        # where no sphere about the minimum is as empty as -0.8
        "frac_sf_defined": float(np.isfinite(r_sf[counted]).mean()) if counted.any() else float("nan"),
        "rv_sf_definition": sf_def,
        # delta08raw only (2026-10-11): of the distinct (crossing R_v) voids, those whose raw enclosed contrast never
        # went below -0.8 within reach (a wisp on the centre) and those still below at the reach (no crossing)
        "n_sf_never_below": int(np.asarray(raw["never_below"])[counted].sum()) if sf_def == RAW_RV else 0,
        "n_sf_capped": int(np.asarray(raw["capped"])[counted].sum()) if sf_def == RAW_RV else 0,
        "r_eff_pct": pct(r_eff, resolved),
        "r_v_pct": pct(r_v, counted),
        "e_pct": pct(e, valid),
        "p_pct": pct(p, valid),
        "delta_pct": pct(cat["delta_values"], np.ones(len(e), bool)),
        # raw samples for the distribution and abundance figures
        "_r_eff": r_eff[resolved & np.isfinite(r_eff)],
        "_delta_resolved": np.asarray(cat["delta_values"], dtype=float)[resolved & np.isfinite(r_eff)],
        "_r_v": r_v[counted],
        "_delta_counted": np.asarray(cat["delta_values"], dtype=float)[counted],
        "_e": e[valid], "_p": p[valid],
        "_delta_valid": np.asarray(cat["delta_values"], dtype=float)[valid],
        "_r_v_sf": r_sf[sf_sel],
        "_delta": np.asarray(cat["delta_values"], dtype=float),
    }


def size_function_figure(stats, panel, ztab, box, args, methods, path) -> dict:
    """population_size_function.png (2026-10-11): the measured dn/dlnR_v and n(>R_v) of the distinct voids (Poisson
    errors) at the panel snapshots against theory.void_size_function at each z (solid lines), and measured / theory;
    the ratio at each snapshot's median R_v printed and returned. The arrays for population_stats.npz."""
    sig = pipeline.smoothing_length_mpc()
    edges = np.geomspace(0.3 * sig, 30.0 * sig, 16)
    mid = np.sqrt(edges[1:] * edges[:-1])
    radii = np.geomspace(0.3 * sig, 30.0 * sig, 40)
    fine = np.geomspace(0.3 * sig, 30.0 * sig, 200)
    defs = sorted({stats[s]["rv_sf_definition"] for s in panel})
    if args.size_rv not in defs:                   # contract C3: said, not silent, until the catalogues carry it
        print(f"  size function: R_v definition {args.size_rv!r} is not in these catalogues: using "
              f"{'/'.join(defs) or 'crossing'} (rebuild them once void-core's R_v definitions are in)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(panel), 1)))
    out = {"size_function_rv": "/".join(defs) or args.size_rv, "size_function_model": SIZE_FUNCTION_MODEL,
           "size_function_edges_mpc": edges, "size_function_radii_mpc": radii, "size_function_theory_r_mpc": fine}
    for c_, s in zip(colors, panel):
        st, z = stats[s], float(ztab.get(s, np.nan))
        rv = st["_r_v_sf"]
        n_cum, err, cnt = catalog.void_abundance(rv, box[s], radii)
        dn, derr, dcnt = catalog.void_dn_dlnr(rv, box[s], edges)
        lab = f"z={z:.2f} (n={rv.size})"
        ok, okd = cnt > 0, dcnt > 0
        axes[0].errorbar(radii[ok], n_cum[ok], yerr=err[ok], fmt="o", ms=3, color=c_, capsize=2, label=lab)
        axes[1].errorbar(mid[okd], dn[okd], yerr=derr[okd], fmt="o", ms=3, color=c_, capsize=2, label=lab)
        med = float(np.median(rv)) if rv.size else np.nan
        out[f"size_function_{s}_frac_defined"] = float(st.get("frac_sf_defined", np.nan))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratio, rerr, rn = ratio_at(rv, box[s], med, z)
        out.update({f"size_function_{s}_dn_dlnr": dn, f"size_function_{s}_dn_dlnr_err": derr,
                    f"size_function_{s}_n_cum": n_cum, f"size_function_{s}_n_cum_err": err,
                    f"size_function_{s}_median_rv_mpc": med, f"size_function_{s}_ratio_at_median": ratio,
                    f"size_function_{s}_ratio_at_median_err": rerr})
        if not np.isfinite(z):
            continue                               # no redshift: no theory to draw
        th = theory.void_size_function(fine, z, SIZE_FUNCTION_MODEL)
        th_cum = theory.void_cumulative(radii, z, model=SIZE_FUNCTION_MODEL)
        out[f"size_function_{s}_theory_dn_dlnr"] = th
        # at high z the model's voids are exp-rare (sigma << |delta_v|: no region has shell-crossed), so its curve
        # falls through any axis: drawn only where it is within the axes' 1e-12 Mpc^-3 floor
        axes[0].plot(radii, np.where(th_cum > 1e-12, th_cum, np.nan), color=c_, lw=1.2)
        axes[1].plot(fine, np.where(th > 1e-12, th, np.nan), color=c_, lw=1.2)
        th_mid = np.array([np.mean(theory.void_size_function(np.geomspace(lo, hi, 21), z, SIZE_FUNCTION_MODEL))
                           for lo, hi in zip(edges[:-1], edges[1:])])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            rr, re_ = dn / th_mid, derr / th_mid
        okr = okd & np.isfinite(rr) & (rr > RATIO_SHOWN[0]) & (rr < RATIO_SHOWN[1])
        if okr.any():
            axes[2].errorbar(mid[okr], rr[okr], yerr=re_[okr], fmt="o-", ms=3, lw=1.0, color=c_, capsize=2,
                             label=f"z={z:.2f}: {ratio:.3g} +- {rerr:.2g} at median {med:.3g} Mpc")
        else:
            axes[2].plot([], [], "o", color=c_, label=f"z={z:.2f}: {ratio:.3g} (off the axis)")
        if st["rv_sf_definition"] == RAW_RV:
            out[f"size_function_{s}_n_never_below"] = int(st.get("n_sf_never_below", 0))
            out[f"size_function_{s}_n_capped"] = int(st.get("n_sf_capped", 0))
        print(f"  size function z={z:.2f}: {rv.size} distinct voids, median R_v {med:.3g} Mpc "
              f"({out['size_function_rv']}, defined for {st.get('frac_sf_defined', np.nan):.1%} of the distinct voids; "
              f"sigma {sig:.3g} Mpc); measured/{SIZE_FUNCTION_MODEL} there {ratio:.3g} +- {rerr:.2g} ({rn} voids "
              f"within exp(+-{RATIO_HALF_WIDTH}))")
    cut = pipeline.ellipsoid_cuts_mpc()["max_axis_mpc"]
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$R_v$ [Mpc]")
        ax.grid(alpha=0.3, which="both")
        ax.axvline(cut, color="0.4", lw=0.8, ls=":")
    axes[0].set_yscale("log"); axes[1].set_yscale("log"); axes[2].set_yscale("log")
    axes[2].set_ylim(*RATIO_SHOWN)
    for ax in axes[:2]:
        lo, hi = ax.get_ylim()
        ax.set_ylim(max(lo, 1e-12), max(hi, 1e-11))
    axes[0].set_ylabel(r"$n(>R_v)$ [Mpc$^{-3}$]")
    axes[1].set_ylabel(r"$dn/d\ln R_v$ [Mpc$^{-3}$]")
    axes[2].set_ylabel("measured / theory")
    axes[2].axhline(1.0, color="gray", lw=0.7, ls=":")
    axes[0].legend(fontsize=7, title="points: measured, lines: Vdn" if SIZE_FUNCTION_MODEL == "vdn" else
                   "points: measured, lines: SvdW", title_fontsize=7)
    axes[2].legend(fontsize=7)
    # where the sample is cut, said on the figure: the ellipsoid gate's max semi-axis (dotted) bounds R_eff ~ 1.75 R_v,
    # and the resolved voids without an R_v (the larger, deeper ones) are missing from the large-R end
    no_rv = sum(stats[s]["n_no_rv"] for s in panel)
    axes[1].text(0.02, 0.03, f"catalogue smoothed at sigma = {config.SMOOTHING_SIGMA_CELLS:g} cells = {sig:.3g} Mpc\n"
                 f"dotted: max semi-axis cut {cut:.3g} Mpc; {no_rv} resolved voids without R_v not counted"
                 + (f"\nR_v: {out['size_function_rv']} ({args.size_rv} not in the catalogue)"
                    if args.size_rv not in defs else f"\nR_v: {out['size_function_rv']}")
                 + ("\nraw density, step "
                    f"{raw_radius.RAW_STEP_CELLS:g} cells, reach {raw_radius.RAW_REACH_FRAC:g} L; defined for "
                    + ", ".join(f"{stats[s].get('frac_sf_defined', np.nan):.0%}" for s in panel)
                    + "\nof the distinct voids" if RAW_RV in defs else ""),
                 transform=axes[1].transAxes, fontsize=7, color="0.3")
    # why the comparison is not a test (2026-10-11), under the panels: inside an axis the long lines squeezed the
    # layout
    caveat = "\n".join(SIZE_FUNCTION_CAVEAT[d] for d in defs if d in SIZE_FUNCTION_CAVEAT)
    if caveat:
        fig.text(0.5, 0.01, caveat, ha="center", va="bottom", fontsize=8, color="0.25", wrap=True)
    style.set_suptitle(fig, f"{args.sim}: void size function ({'/'.join(methods)}, distinct well-resolved voids) "
                            f"against {SIZE_FUNCTION_MODEL.upper()} (delta_v = {theory.DELTA_V_LINEAR}, "
                            f"delta_c = {theory.DELTA_C})")
    fig.tight_layout(rect=(0, 0.07 if caveat else 0, 1, 1))
    style.save_plot_to_multiple_paths(fig, path, dpi=config.DPI, mirror=MIRROR)
    plt.close(fig)
    return out


@contextlib.contextmanager
def smoothing_cells(cells=None, mpc=None):
    """The process's smoothing set to 'cells' (pipeline.set_smoothing in cells: the production footprint) or to 'mpc'
    (its footprint following it, as --smooth-mpc) for the block, restored after it whatever happens -- the size
    function's catalogue (2026-10-11)."""
    saved = (config.SMOOTHING_SIGMA_CELLS, config.FOOTPRINT_SIZE, config.SMOOTHING_SIGMA_MPC)
    try:
        pipeline.set_smoothing(cells=cells, mpc=mpc)
        yield
    finally:
        config.SMOOTHING_SIGMA_CELLS, config.FOOTPRINT_SIZE, config.SMOOTHING_SIGMA_MPC = saved


def size_smooth_active(size_smooth) -> bool:
    """Whether --size-smooth asks for a catalogue other than the one the rest of the figures use (0: the same)."""
    ss = float(size_smooth or 0.0)
    return ss > 0 and not math.isclose(ss, float(config.SMOOTHING_SIGMA_CELLS), rel_tol=1e-9)


def size_smooth_tag(size_smooth) -> str:
    """'' at the default --size-smooth, else '_ss<cells>' ('_ss0': the figures' own catalogue): the stats npz's name,
    so another smoothing never overwrites the default's (the png is tagged '_ss<cells>' whenever size_smooth_active)."""
    ss = float(size_smooth or 0.0)
    return "" if math.isclose(ss, SIZE_SMOOTH_CELLS) else f"_ss{ss:g}"


def raw_for(prod, cat, size_rv):
    """raw_radius.raw_radii of the catalogue's well-resolved rows when the size function asks for 'delta08raw',
    else None (2026-10-11); called while 'prod' still holds its grids."""
    return raw_radius.raw_radii(prod, cat) if size_rv == RAW_RV else None


def size_function_stats(args, snaps, keep_overlaps=False) -> dict:
    """snapshot_stats of the panel snapshots on the --size-smooth catalogue (its own cache namespace,
    r512_s<cells>_f11_...; built on first use, ~20 s and ~7 GB per 512^3 snapshot); to be called inside
    smoothing_cells(args.size_smooth). A snapshot that cannot be read is said and left out."""
    out = {}
    for s in snaps:
        prod = pipeline.products(s, sim=args.sim, method=args.method, prefix=args.prefix, data_root=args.data_root)
        try:
            _cell, _box, n_, _src = catalog.frame_of(prod)
            cat = prod.voids()
            raw = raw_for(prod, cat, args.size_rv)
        except (OSError, ValueError) as e:
            print(f"  size function: snap {s:3d} at {config.SMOOTHING_SIGMA_CELLS:g} cells left out: {e}")
            continue
        finally:
            prod.release()
        out[s] = snapshot_stats(cat, n_, keep_overlaps, args.size_rv, raw)
    return out


# ---------------------------------------------------------------- the size function at fixed R (--size-fixed-r)

def fixed_r_smoothings(cells=None, mpc=None) -> list:
    """The fixed-R figure's smoothings as [(kind, value, sigma_mpc)], kind 'cells' or 'mpc', in this simulation's frame.
    Neither list given: FIXED_R_SMOOTH_CELLS; either given: only those. A smoothing within 0.1% of an earlier one is
    the same catalogue scale (TNG100's 2.1625 Mpc is 10.0002 cells) and dropped, said: counted twice it would shrink
    the scatter across smoothings."""
    if cells is None and mpc is None:
        cells = FIXED_R_SMOOTH_CELLS
    out = []
    for kind, vals in (("cells", cells or ()), ("mpc", mpc or ())):
        for v in vals:
            v = float(v)
            if v <= 0:
                raise SystemExit(f"the fixed-R smoothings must be > 0, got {v:g} {kind}")
            sig = max(v, 1.0) * float(config.CELL_SIZE) if kind == "cells" else max(v, float(config.CELL_SIZE))
            same = [o for o in out if math.isclose(o[2], sig, rel_tol=1e-3)]
            if same:
                print(f"  size function at fixed R: {v:g} {kind} = {sig:.4g} Mpc is the catalogue of {same[0][1]:g} "
                      f"{same[0][0]} ({same[0][2]:.4g} Mpc): counted once")
                continue
            out.append((kind, v, sig))
    if not out:
        raise SystemExit("the fixed-R size function needs at least one smoothing")
    return sorted(out, key=lambda o: -o[2])          # largest sigma first: the legend reads in the smoothing's order


def fixed_r_label(kind, value, sig) -> str:
    return (f"{value:g} cells = {sig:.3g} Mpc" if kind == "cells"
            else f"{sig:.4g} Mpc = {sig / float(config.CELL_SIZE):.3g} cells")


def _drop_same_centre(sel, centres, depth) -> tuple:
    """(sel without the rows whose re-centred cell another selected row already holds -- the deeper by 'depth' kept
    --, how many were dropped)."""
    sel = np.asarray(sel, dtype=bool).copy()
    rows = np.nonzero(sel)[0]
    rows = rows[np.argsort(np.asarray(depth, dtype=np.float64)[rows], kind="stable")]
    seen, ndup = set(), 0
    for i in rows:
        key = tuple(int(v) for v in centres[i])
        if key in seen:
            sel[i] = False
            ndup += 1
        else:
            seen.add(key)
    return sel, ndup


def raw_sphere_distinct(centres, r_cells, sample, n_grid) -> np.ndarray:
    """The distinct voids by their RAW -0.8 spheres (2026-10-12, --fixed-r-distinct raw): among the 'sample' rows
    with a radius, the largest sphere first, none that OVERLAPS a sphere already kept (centre distance below the sum
    of the radii, periodic) -- the non-overlapping spheres the volume-conserving Vdn counts (Jennings, Li & Hu 2013
    clean their spherical voids the same way). The catalogue rule (catalog.distinct_mask) keeps the DEEPEST smoothed
    minimum of a region with its claim min(R_v, R_eff); at a small sigma a large void holds several minima and the
    deepest is often not the one with the largest raw sphere (TNG300 z = 0, the 765 voids above 6.49 Mpc at 2.1625
    Mpc: at 2 cells 208 held a minimum with R above 6.49 Mpc that the catalogue rule dropped). The centre rule
    (pipeline.distinct_voids: drop only a centre INSIDE a kept sphere) was tried first and kept 3-4x Vdn: spheres
    overlapping by up to their whole radius were all counted."""
    r_cells = np.asarray(r_cells, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    rows = np.nonzero(np.asarray(sample, dtype=bool) & np.isfinite(r_cells))[0]
    rows = rows[np.argsort(-r_cells[rows], kind="stable")]
    out = np.zeros(r_cells.size, dtype=bool)
    kc, kr, nk = np.empty((rows.size, 3)), np.empty(rows.size), 0
    for i in rows:
        if nk:
            d = centres[i] - kc[:nk]
            d -= n_grid * np.rint(d / n_grid)
            if ((d * d).sum(axis=1) < (kr[:nk] + r_cells[i]) ** 2).any():
                continue
        out[i] = True
        kc[nk], kr[nk] = centres[i], r_cells[i]
        nk += 1
    return out


def fixed_r_samples(args, snaps, smooths, recentre=None, distinct=None) -> tuple:
    """The fixed-R size function's samples, per smoothing of 'smooths' (fixed_r_smoothings): the raw -0.8 radius
    (raw_radius.raw_radii, measured and cached on first use in that smoothing's cache namespace) of the DISTINCT voids
    among the well-resolved ones with a raw radius, under the process's gate (main refuses anything but
    DTFE_VOID_GATE=none), and their centres in Mpc. Returns (entries, box per snapshot, (method, prefix) per
    snapshot); each entry {'kind', 'value', 'sigma_mpc', 'label', 'namespace', 'r': {snap: Mpc}, 'pos': {snap: (n, 3)
    Mpc}, 'n_resolved', 'n_never_below', 'n_capped', 'n_dup': {snap: int}, 'rc': {snap: offsets and radii}}. A
    snapshot that cannot be read is said and left out of that smoothing. recentre (default FIXED_R_RECENTRE;
    --fixed-r-recentre / --fixed-r-no-recentre, 2026-10-12): the radii about the RE-CENTRED centres
    (raw_radius.raw_radii(recentre=True)), the positions the re-centred ones, duplicates of one centre dropped.
    distinct (default FIXED_R_DISTINCT; --fixed-r-distinct): 'catalogue' (catalog.distinct_mask) or 'raw'
    (raw_sphere_distinct: by the raw spheres, largest first)."""
    recentre = FIXED_R_RECENTRE if recentre is None else bool(recentre)
    distinct = FIXED_R_DISTINCT if distinct is None else str(distinct)
    entries, box, used = [], {}, {}
    for kind, value, _sig in smooths:
        with smoothing_cells(**{kind: value}):
            sig = pipeline.smoothing_length_mpc()
            ent = {"kind": kind, "value": float(value), "sigma_mpc": float(sig), "label": fixed_r_label(kind, value, sig),
                   "namespace": pipeline._param_hash(), "r": {}, "pos": {}, "n_resolved": {}, "n_never_below": {},
                   "n_capped": {}, "n_dup": {}, "rc": {}}
            print(f"  size function at fixed R: catalogues at {ent['label']} (cache namespace {ent['namespace']})")
            for s in snaps:
                prod = pipeline.products(s, sim=args.sim, method=args.method, prefix=args.prefix,
                                         data_root=args.data_root)
                try:
                    cell, box_s, n_, _src = catalog.frame_of(prod)
                    cat = prod.voids()
                    raw = raw_radius.raw_radii(prod, cat, recentre=recentre)
                    used[s] = (prod.fs.method, prod.fs.prefix)
                except (OSError, ValueError) as e:
                    print(f"  size function at fixed R: snap {s:3d} at {ent['label']} left out: {e}")
                    continue
                finally:
                    prod.release()
                box[s] = float(box_s)
                resolved = np.asarray(cat["well_resolved"]).astype(bool)
                r = np.asarray(raw["r_mpc"], dtype=np.float64)
                centres = np.asarray(raw["coords_rc"] if recentre else cat["coords"], dtype=np.int64)
                if distinct == "raw":
                    sel = raw_sphere_distinct(centres, r / float(cell), resolved, int(n_))
                else:
                    sel = catalog.distinct_mask(cat, resolved & np.isfinite(r), int(n_))
                if recentre and distinct != "raw":           # 'raw' already keeps one of a shared centre
                    # two minima of one void can re-centre onto the same raw minimum (2026-10-12): counted once, the
                    # deeper catalogue row kept; the number dropped is said and saved
                    sel, ent["n_dup"][s] = _drop_same_centre(sel, centres, np.asarray(cat["delta_values"]))
                    r0 = np.asarray(raw["r0_mpc"], dtype=np.float64)[sel]
                    off = np.asarray(raw["offset_cells"], dtype=np.float64)[sel]
                    ent["rc"][s] = {"offset_cells": off, "offset_sigma": off * float(cell) / float(sig),
                                    "r0_mpc": r0, "r_mpc": r[sel]}
                    grown = np.isfinite(r0) & (r[sel] > r0 * (1 + 1e-9))
                    print(f"    snap {s:3d}: re-centred {int(sel.sum())} distinct voids: median offset "
                          f"{np.median(off):.2f} cells = {np.median(off) * float(cell) / float(sig):.2f} sigma, "
                          f"median R/R0 {np.nanmedian(r[sel] / r0):.3f}, radius grew for {grown.mean():.0%}, "
                          f"{ent['n_dup'][s]} duplicate centres dropped, {int((~np.isfinite(r0)).sum())} had no R0")
                ent["r"][s] = r[sel]
                ent["pos"][s] = (centres[sel].astype(np.float64) + 0.5) * float(cell)
                ent["n_resolved"][s] = int(resolved.sum())
                ent["n_never_below"][s] = int(np.asarray(raw["never_below"])[resolved].sum())
                ent["n_capped"][s] = int(np.asarray(raw["capped"])[resolved].sum())
        entries.append(ent)
    return entries, box, used


def fixed_r_range(samples, sigmas, box_mpc, r_range=None) -> tuple:
    """The common complete range (r_lo, r_hi) in Mpc of the radius samples (one per smoothing) found at 'sigmas':
    r_lo = FIXED_R_MIN_SIGMA x the LARGEST sigma, r_hi = the smallest over the smoothings of the radius with
    FIXED_R_MIN_COUNT voids at or above it, at most the raw reach (raw_radius.RAW_REACH_FRAC L), empty when narrower
    than FIXED_R_MIN_WIDTH. 'r_range' (lo, hi) replaces both (a cross-box comparison on one range). (nan, nan) when
    it is empty."""
    if r_range is not None:
        lo, hi = float(r_range[0]), float(r_range[1])
    else:
        lo = FIXED_R_MIN_SIGMA * float(max(sigmas))
        tops = []
        for r in samples:
            r = np.sort(np.asarray(r, dtype=np.float64)[np.isfinite(r)])
            tops.append(r[-FIXED_R_MIN_COUNT] if r.size >= FIXED_R_MIN_COUNT else -np.inf)
        hi = min(min(tops), raw_radius.RAW_REACH_FRAC * float(box_mpc))
        if hi < FIXED_R_MIN_WIDTH * lo:
            return np.nan, np.nan
    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else (np.nan, np.nan)


def bootstrap_counts(r, pos, box_mpc, edges, radii, nsub=None, nboot=None, seed=None) -> dict:
    """Sub-cube bootstrap of the counts of radii 'r' (centres 'pos', Mpc) in the bins 'edges', above each of 'radii'
    and in [edges[0], edges[-1]): the box cut into nsub^3 cubes, nsub^3 of them drawn with replacement per draw (the
    sample variance of a clustered population, which Poisson's sqrt N leaves out). {'hist': (nboot, nbins), 'cum':
    (nboot, nradii), 'tot': (nboot,)}: counts per draw, for a standard deviation across the draws. The bins are
    half-open like 'tot' (2026-10-11): edges[-1] is a sample radius (fixed_r_range's r_hi) and np.histogram's closed
    last bin counted that void in 'hist' but not in 'tot'. The draws depend only on (nsub, nboot, seed), so calls with
    the same three resample the same sub-cubes: fixed_r_statistics pairs the smoothings' draws on that."""
    nsub = int(nsub or FIXED_R_NSUB)
    nboot = int(nboot or FIXED_R_NBOOT)
    r = np.asarray(r, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    edges, radii = np.asarray(edges, dtype=np.float64), np.asarray(radii, dtype=np.float64)
    ncube = nsub ** 3
    ijk = np.clip(np.floor(np.mod(pos, float(box_mpc)) / float(box_mpc) * nsub).astype(np.int64), 0, nsub - 1)
    cube = (ijk[:, 0] * nsub + ijk[:, 1]) * nsub + ijk[:, 2]
    hist = np.zeros((ncube, edges.size - 1))
    cum = np.zeros((ncube, radii.size))
    tot = np.zeros(ncube)
    for c in range(ncube):
        rc = r[cube == c]
        hist[c] = np.histogram(rc[rc < edges[-1]], bins=edges)[0]
        cum[c] = (rc[:, None] > radii[None, :]).sum(axis=0)
        tot[c] = ((rc >= edges[0]) & (rc < edges[-1])).sum()
    rng = np.random.default_rng(FIXED_R_SEED if seed is None else seed)
    draws = rng.integers(0, ncube, size=(nboot, ncube))
    mult = np.stack([np.bincount(d, minlength=ncube) for d in draws]).astype(np.float64)    # (nboot, ncube)
    return {"hist": mult @ hist, "cum": mult @ cum, "tot": mult @ tot}


def fixed_r_statistics(entries, box, ztab, snaps, r_range=None, model=SIZE_FUNCTION_MODEL) -> dict:
    """The fixed-R comparison per snapshot (2026-10-11): the common range (fixed_r_range), per smoothing n(>R) and
    dn/dlnR with Poisson and sub-cube bootstrap errors, the bin-averaged model (theory.void_dn_dlnr_binned), the ratio
    per bin and ONE ratio per smoothing over the whole range (count / the model's count there,
    theory.void_number_between), and their scatter across the smoothings; whether the smoothings differ is tested on the
    paired bootstrap (paired_difference_test, 2026-10-11). Every count is half-open [lo, hi). {snap: {...}} (empty
    range: 'empty' True)."""
    out = {}
    for s in snaps:
        have = [e for e in entries if s in e["r"]]
        z = float(ztab.get(s, np.nan))
        if not have or s not in box:
            continue
        b = float(box[s])
        vol = b ** 3
        lo, hi = fixed_r_range([e["r"][s] for e in have], [e["sigma_mpc"] for e in have], b, r_range)
        lo0 = FIXED_R_MIN_SIGMA * max(e["sigma_mpc"] for e in have) if r_range is None else float(r_range[0])
        st = {"z": z, "box_mpc": b, "r_lo": lo, "r_hi": hi, "labels": [e["label"] for e in have],
              "sigma_mpc": np.array([e["sigma_mpc"] for e in have]), "empty": not np.isfinite(lo), "per": [],
              # the voids above the range's lower end per smoothing: said when the range is empty (why no test)
              "n_above_lo": np.array([int((np.asarray(e["r"][s]) >= lo0).sum()) for e in have]), "lo0": lo0}
        out[s] = st
        if st["empty"] or not np.isfinite(z):
            st["empty"] = True
            continue
        edges = np.geomspace(lo, hi, FIXED_R_BINS + 1)
        mid = np.sqrt(edges[1:] * edges[:-1])
        radii = np.geomspace(lo, hi, 24)
        dln = np.log(edges[1:] / edges[:-1])
        th_bin = theory.void_dn_dlnr_binned(edges, z, model=model)
        th_cum = theory.void_cumulative(radii, z, model=model)
        th_tot = theory.void_number_between(lo, hi, z, model=model)
        st.update(edges=edges, mid=mid, radii=radii, theory_dn_dlnr=th_bin, theory_n_cum=th_cum, theory_total=th_tot)
        boot_ratio = []
        for e in have:
            r = e["r"][s]
            n_cum, err, cnt = catalog.void_abundance(r, b, radii)
            # half-open [lo, hi) like n_total, the bootstrap and Vdn's number (2026-10-11): hi is a sample radius
            # (the scarcest smoothing's 5th largest) that np.histogram's closed last bin counted, one void too many
            dn, derr, dcnt = catalog.void_dn_dlnr(r[r < hi], b, edges)
            bs = bootstrap_counts(r, e["pos"][s], b, edges, radii)
            n_tot = int(((r >= lo) & (r < hi)).sum())
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                ratio_bin = dn / th_bin
                p = {"label": e["label"], "sigma_mpc": e["sigma_mpc"], "n_cum": n_cum, "n_cum_err": err,
                     "n_cum_boot": bs["cum"].std(axis=0) / vol, "count_cum": cnt,
                     "dn_dlnr": dn, "dn_dlnr_err": derr, "dn_dlnr_boot": bs["hist"].std(axis=0) / (vol * dln),
                     "count_bin": dcnt, "ratio_bin": ratio_bin, "ratio_bin_err": derr / th_bin,
                     "ratio_bin_boot": bs["hist"].std(axis=0) / (vol * dln) / th_bin, "n_total": n_tot,
                     "n_distinct": int(r.size), "n_resolved": int(e["n_resolved"].get(s, 0)),
                     "n_never_below": int(e["n_never_below"].get(s, 0)), "n_capped": int(e["n_capped"].get(s, 0)),
                     "n_expected": vol * th_tot, "ratio": n_tot / (vol * th_tot), "ratio_err": math.sqrt(max(n_tot, 1)) / (vol * th_tot),
                     "ratio_boot": float(bs["tot"].std()) / (vol * th_tot)}
            st["per"].append(p)
            boot_ratio.append(bs["tot"] / (vol * th_tot))        # this smoothing's whole-range ratio per draw
        rat = np.array([p["ratio"] for p in st["per"]])
        errs = np.array([max(p["ratio_err"], p["ratio_boot"]) for p in st["per"]])
        st["ratio_mean"] = float(np.mean(rat))
        st["ratio_scatter"] = float(np.std(rat)) if rat.size > 1 else 0.0       # across the smoothings
        st["ratio_err_typical"] = float(np.median(errs))
        st["consistent"] = bool(np.all(np.abs(rat - 1.0) <= FIXED_R_CONSISTENT * errs))
        # 'converged' = no significant difference: the ratios differ by no more than the PAIRED sub-cube bootstrap
        # allows (2026-10-11, review-science): they are catalogues of one density field that largely count the same
        # voids, and the sample variance in 'errs' is common to all of them and cancels in their differences. A chi^2 on
        # independent 'errs' therefore overstates the spread of the differences and is biased toward 'agree' (it was
        # called conservative here: the wrong direction). bootstrap_counts drew the same sub-cubes for every
        # smoothing, so the differences per draw carry only what the catalogues do NOT share. The old chi^2 stays
        # in the npz ('chi2') for reference only.
        st["chi2"] = float(np.sum((rat - rat.mean()) ** 2 / np.maximum(errs, 1e-300) ** 2))
        st["chi2_95"] = float(_chi2_95(rat.size - 1)) if rat.size > 1 else np.inf
        st["paired_chi2"], st["paired_max_sigma"] = paired_difference_test(rat, np.array(boot_ratio))
        st["converged"] = bool(st["paired_chi2"] <= st["chi2_95"])
        # a forced range (--fixed-r-range) can hold almost no voids: 0 of an expected 0.04 is 'consistent' and says
        # nothing, so the verdict is withheld below FIXED_R_MIN_COUNT voids in any smoothing
        st["too_few"] = bool(min(p["n_total"] for p in st["per"]) < FIXED_R_MIN_COUNT)
    return out


def paired_difference_test(rat, boot) -> tuple:
    """(chi^2 for n - 1 dof, the largest pairwise difference in sigma) of the ratios 'rat' (n smoothings) against
    their PAIRED bootstrap draws 'boot' (n, nboot) -- the same resamples for every smoothing (2026-10-11): the
    contrasts rat[k] - rat[0] with the covariance of their draws. A contrast the draws do not vary is 0 sigma when
    the ratios agree there exactly (identical catalogues) and infinite when they do not. (0, 0) for one smoothing."""
    rat = np.asarray(rat, dtype=np.float64)
    boot = np.asarray(boot, dtype=np.float64).reshape(rat.size, -1)
    if rat.size < 2:
        return 0.0, 0.0
    tol = 1e-9 * max(float(np.max(np.abs(rat))), 1e-300)

    def _sig(d, var):
        return 0.0 if abs(d) <= tol else (abs(d) / math.sqrt(var) if var > tol ** 2 else np.inf)

    d = rat[1:] - rat[0]
    db = boot[1:] - boot[0]
    cov = np.atleast_2d(np.cov(db))
    lam, vec = np.linalg.eigh(cov)
    proj = vec.T @ d
    chi2 = 0.0
    for lm, pr in zip(lam, proj):
        s_ = _sig(pr, lm)
        chi2 = np.inf if not np.isfinite(s_) else chi2 + s_ * s_
    pair = max(_sig(rat[k] - rat[j], float(np.var(boot[k] - boot[j], ddof=1)))
               for k in range(rat.size) for j in range(k))
    return float(chi2), float(pair)


def _chi2_95(dof: int) -> float:
    from scipy.stats import chi2
    return chi2.ppf(0.95, dof)


def fixed_r_verdict(st) -> str:
    """One plain line per snapshot: the ratio per smoothing, its scatter, whether it is consistent with the model."""
    if st.get("empty"):
        return (f"z={st['z']:.2f}: no common complete range (from {FIXED_R_MIN_SIGMA:g} x the largest sigma to "
                f">= {FIXED_R_MIN_COUNT} voids in every smoothing, at least x{FIXED_R_MIN_WIDTH:g} wide; voids above "
                f"{st.get('lo0', np.nan):.3g} Mpc: "
                + ", ".join(str(int(n)) for n in st.get("n_above_lo", ())) + "): no test")
    err = [max(p["ratio_err"], p["ratio_boot"]) for p in st["per"]]
    each = ", ".join(f"{p['ratio']:.2f} +- {e:.2f} ({p['n_total']} voids, Vdn {p['n_expected']:.3g})"
                     for p, e in zip(st["per"], err))
    off = sum(abs(p["ratio"] - 1.0) > FIXED_R_CONSISTENT * e for p, e in zip(st["per"], err))
    verdict = ("CONSISTENT with Vdn at every smoothing" if st["consistent"]
               else f"NOT consistent with Vdn (beyond {FIXED_R_CONSISTENT:g} errors at {off} of {len(st['per'])} "
                    "smoothings)")
    if st.get("too_few"):
        verdict = f"too few voids for a test (below {FIXED_R_MIN_COUNT} in a smoothing)"
    # worded from the PAIRED bootstrap test (2026-10-11): a non-rejection is 'no significant difference', not proof
    # that the smoothings have converged
    conv = "one smoothing: no convergence test" if len(st["per"]) < 2 else \
        ("no significant difference between the smoothings" if st["converged"] else "smoothings disagree") + \
        f" (paired bootstrap chi2 {st['paired_chi2']:.1f} for {len(st['per']) - 1} dof, largest pair " \
        f"{st['paired_max_sigma']:.1f} sigma; scatter {st['ratio_scatter']:.2f}, error {st['ratio_err_typical']:.2f})"
    return (f"z={st['z']:.2f}, R {st['r_lo']:.3g}-{st['r_hi']:.3g} Mpc: measured/Vdn {each}; mean "
            f"{st['ratio_mean']:.2f}; {conv}; {verdict}")


def fixed_r_figure(fstats, snaps, sim, methods, path, model=SIZE_FUNCTION_MODEL, recentre=False) -> dict:
    """population_size_function_fixedR*.png (2026-10-11): one row per snapshot -- n(>R), dn/dlnR (points: Poisson
    errors, bands: sub-cube bootstrap) of each smoothing's ungated raw-radius sample against the model (black), and
    measured / model per bin with the one-number ratio of each smoothing in the legend -- only over the common
    complete range. The verdict per z under the panels. Returns the arrays for the npz."""
    rows = [s for s in snaps if s in fstats]
    fig, axes = plt.subplots(max(len(rows), 1), 3, figsize=(15, 3.6 * max(len(rows), 1) + 1.0), squeeze=False)
    nsm = max([len(fstats[s]["per"]) for s in rows] + [1])
    colors = plt.cm.plasma(np.linspace(0.1, 0.8, nsm))
    out = {"fixed_r_model": model, "fixed_r_snaps": np.array(rows, dtype=np.int64),
           "fixed_r_min_sigma": FIXED_R_MIN_SIGMA, "fixed_r_min_count": FIXED_R_MIN_COUNT,
           "fixed_r_nsub": FIXED_R_NSUB, "fixed_r_nboot": FIXED_R_NBOOT}
    verdicts = []
    for row, s in zip(axes, rows):
        st = fstats[s]
        verdicts.append(fixed_r_verdict(st))
        print("  size function at fixed R: " + verdicts[-1])
        out.update({f"fixed_r_{s}_n_above_lo": st["n_above_lo"], f"fixed_r_{s}_z": st["z"],
                    f"fixed_r_{s}_r_lo": st["r_lo"], f"fixed_r_{s}_r_hi": st["r_hi"],
                    f"fixed_r_{s}_labels": np.array(st["labels"]), f"fixed_r_{s}_sigma_mpc": st["sigma_mpc"]})
        zl = f"z={st['z']:.2f}"
        if st["empty"]:
            for ax in row:
                ax.text(0.5, 0.5, f"{zl}: no common complete range", ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="0.3")
                ax.set_xticks([]); ax.set_yticks([])
            continue
        out.update({f"fixed_r_{s}_edges_mpc": st["edges"], f"fixed_r_{s}_radii_mpc": st["radii"],
                    f"fixed_r_{s}_theory_dn_dlnr": st["theory_dn_dlnr"], f"fixed_r_{s}_theory_n_cum": st["theory_n_cum"],
                    f"fixed_r_{s}_ratio_mean": st["ratio_mean"], f"fixed_r_{s}_ratio_scatter": st["ratio_scatter"],
                    f"fixed_r_{s}_consistent": st["consistent"], f"fixed_r_{s}_converged": st["converged"],
                    f"fixed_r_{s}_chi2": st["chi2"], f"fixed_r_{s}_too_few": st["too_few"],
                    f"fixed_r_{s}_paired_chi2": st["paired_chi2"],
                    f"fixed_r_{s}_paired_max_sigma": st["paired_max_sigma"]})
        for k, (p, c_) in enumerate(zip(st["per"], colors)):
            for key in ("n_cum", "n_cum_err", "n_cum_boot", "dn_dlnr", "dn_dlnr_err", "dn_dlnr_boot", "ratio_bin",
                        "ratio_bin_err", "ratio_bin_boot", "ratio", "ratio_err", "ratio_boot", "n_total", "n_expected",
                        "n_distinct", "n_resolved", "n_never_below", "n_capped"):
                out[f"fixed_r_{s}_{k}_{key}"] = p[key]
            ok, okd = p["count_cum"] > 0, p["count_bin"] > 0
            f = 1.0 + 0.012 * (k - (nsm - 1) / 2.0)                 # a small x offset per smoothing: bars apart
            row[0].errorbar(st["radii"][ok] * f, p["n_cum"][ok], yerr=p["n_cum_err"][ok], fmt="o", ms=3, color=c_,
                            capsize=2, label=f"{p['label']} (n={p['n_total']})")
            row[0].fill_between(st["radii"], np.clip(p["n_cum"] - p["n_cum_boot"], 1e-30, None),
                                p["n_cum"] + p["n_cum_boot"], color=c_, alpha=0.15, lw=0)
            row[1].errorbar(st["mid"][okd] * f, p["dn_dlnr"][okd], yerr=p["dn_dlnr_err"][okd], fmt="o", ms=3,
                            color=c_, capsize=2)
            row[1].fill_between(st["mid"], np.clip(p["dn_dlnr"] - p["dn_dlnr_boot"], 1e-30, None),
                                p["dn_dlnr"] + p["dn_dlnr_boot"], color=c_, alpha=0.15, lw=0)
            err = np.maximum(p["ratio_bin_err"], p["ratio_bin_boot"])
            row[2].errorbar(st["mid"][okd] * f, p["ratio_bin"][okd], yerr=err[okd], fmt="o-", ms=3, lw=1.0, color=c_,
                            capsize=2, label=f"{p['ratio']:.2f} +- {max(p['ratio_err'], p['ratio_boot']):.2f}")
        row[0].plot(st["radii"], st["theory_n_cum"], color="k", lw=1.2, label=model.upper())
        row[1].stairs(st["theory_dn_dlnr"], st["edges"], color="k", lw=1.2)
        row[2].axhline(1.0, color="k", lw=0.8)
        row[2].axhspan(st["ratio_mean"] - st["ratio_scatter"], st["ratio_mean"] + st["ratio_scatter"], color="0.5",
                       alpha=0.15, lw=0)
        for ax in row:
            ax.set_xscale("log")
            ax.set_xlim(st["r_lo"] * 0.95, st["r_hi"] * 1.05)
            # the range is often under a decade: plain numbers on the minor ticks, else one '10^1' is all the axis says
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(FormatStrFormatter("%g"))
            ax.tick_params(axis="x", which="minor", labelsize=7)
            ax.grid(alpha=0.3, which="both")
            ax.set_xlabel(r"raw $-0.8$ radius $R$ [Mpc]" + (" (re-centred)" if recentre else ""))
        row[0].set_yscale("log"); row[1].set_yscale("log")
        # the y range from the points and the model only: a zero bin's band or error bar reaches 0 and a log axis
        # followed it down to 1e-30
        for ax, keys, th in ((row[0], "n_cum", st["theory_n_cum"]), (row[1], "dn_dlnr", st["theory_dn_dlnr"])):
            v = np.concatenate([np.asarray(th, dtype=float)] + [np.asarray(p[keys], dtype=float) for p in st["per"]])
            v = v[np.isfinite(v) & (v > 0)]
            if v.size:
                ax.set_ylim(v.min() / 3.0, v.max() * 3.0)
        row[0].set_ylabel(r"$n(>R)$ [Mpc$^{-3}$]")
        row[1].set_ylabel(r"$dn/d\ln R$ [Mpc$^{-3}$]")
        row[2].set_ylabel(f"measured / {model.upper()}")
        row[0].legend(fontsize=7, title=zl, title_fontsize=8)
        row[2].legend(fontsize=7, title="whole range (band: scatter)", title_fontsize=7)
    fig.text(0.5, 0.005, "\n".join(verdicts) + f"\nungated distinct voids; points: Poisson, bands: {FIXED_R_NSUB}^3 "
             f"sub-cube bootstrap; range from {FIXED_R_MIN_SIGMA:g} x the largest sigma to >= {FIXED_R_MIN_COUNT} "
             "voids above R in every smoothing; the smoothings share one density field, so they are compared on the "
             "paired bootstrap (the same sub-cube draws for each)",
             ha="center", va="bottom", fontsize=7, color="0.25")
    style.set_suptitle(fig, f"{sim}: void size function at fixed R, raw -0.8 radius"
                            f"{' about re-centred centres' if recentre else ''}, ungated ({'/'.join(methods)}) "
                            f"against {model.upper()}")
    fig.tight_layout(rect=(0, 0.035 * (len(verdicts) + 2) / max(len(rows), 1), 1, 1))
    style.save_plot_to_multiple_paths(fig, path, dpi=config.DPI, mirror=MIRROR)
    plt.close(fig)
    out["fixed_r_verdicts"] = np.array(verdicts)
    return out


def fixed_r_tag(smooths, r_range=None) -> str:
    """The png/npz name tag of a fixed-R run: '' for the default smoothings, else '_c10-5-3' (cells) / '_m2.1625'
    (Mpc), and '_R<lo>-<hi>' for a given range -- runs with other smoothings never overwrite each other."""
    cells = [v for k, v, _ in smooths if k == "cells"]
    mpc = [v for k, v, _ in smooths if k == "mpc"]
    tag = "" if (not mpc and tuple(cells) == tuple(FIXED_R_SMOOTH_CELLS)) else \
        (("_c" + "-".join(f"{v:g}" for v in cells)) if cells else "") + \
        (("_m" + "-".join(f"{v:g}" for v in mpc)) if mpc else "")
    if r_range is not None:
        tag += f"_R{float(r_range[0]):g}-{float(r_range[1]):g}"
    return tag


def fixed_r_main(args, ztab) -> Path:
    """--size-fixed-r: only the fixed-R size function figure and its npz (2026-10-11). Refuses unless
    DTFE_VOID_GATE=none: gated, the ellipsoid gate truncates the large voids differently at each smoothing (TNG100
    z = 0, R = 3 Mpc: 0.015 / 0.108 / 0 of Vdn at 10 / 5 / 3 cells), so there is nothing to converge."""
    if pipeline.void_gate() != "none":
        raise SystemExit("the size function at fixed R is defined on the UNGATED voids: run it as "
                         f"DTFE_VOID_GATE=none DTFE_SIM={args.sim} python3 plot/plot_void_population.py --size-fixed-r")
    smooths = fixed_r_smoothings(args.fixed_r_smooth, args.fixed_r_smooth_mpc)
    snaps = [int(s) for s in (args.panel_snaps or FIXED_R_SNAPS)]
    recentre = FIXED_R_RECENTRE if getattr(args, "fixed_r_recentre", None) is None else bool(args.fixed_r_recentre)
    distinct = getattr(args, "fixed_r_distinct", None) or FIXED_R_DISTINCT
    entries, box, used = fixed_r_samples(args, snaps, smooths, recentre, distinct)
    if not used:
        raise SystemExit(f"no {args.method} field grids for {args.sim} at snapshots {snaps} -- nothing to analyse")
    fstats = fixed_r_statistics(entries, box, ztab, snaps, args.fixed_r_range)
    methods = sorted({m for m, _p in used.values()})
    default_prefix = {"ps": "ps_output.", "dtfe": "output."}
    grids = "+".join(sorted({m if p == default_prefix.get(m) else p.rstrip(".") for m, p in used.values()}))
    name_tag = "" if grids == "ps" else f"_{grids}"
    with smoothing_cells(cells=config.PRODUCTION_SMOOTHING[0]):
        outdir = FIGURE_ROOT / args.sim / pipeline.smoothing_tag()        # 'gatenone': the ungated products' folder
    outdir.mkdir(parents=True, exist_ok=True)
    tag = name_tag + fixed_r_tag(smooths, args.fixed_r_range)
    if recentre != FIXED_R_RECENTRE:
        tag += "_rc" if recentre else "_norc"
    if distinct != FIXED_R_DISTINCT:
        tag += "_draw" if distinct == "raw" else "_dcat"
    path = outdir / f"population_size_function_fixedR{tag}.png"
    arrs = fixed_r_figure(fstats, snaps, args.sim, methods, path, recentre=recentre)
    # the re-centring's own numbers per smoothing and snapshot (2026-10-12): how far the centres moved, in cells and
    # in that smoothing's sigma, the radius before (R0) and after, the duplicate centres dropped
    for k, e in enumerate(entries):
        for s_, d in e["rc"].items():
            for key, v in d.items():
                arrs[f"fixed_r_rc_{s_}_{k}_{key}"] = np.asarray(v)
        for s_, v in e["n_dup"].items():
            arrs[f"fixed_r_rc_{s_}_{k}_n_dup"] = int(v)
    arrs["fixed_r_recentre"] = bool(recentre)
    arrs["fixed_r_distinct"] = np.str_(distinct)
    if recentre:
        arrs["fixed_r_rc_settings"] = raw_radius.settings_rc()
    if fstats and all(st["empty"] for st in fstats.values()):     # TNG300 at 10 / 5 / 3 cells: 3 sigma = 17.7 Mpc
        print("  size function at fixed R: no z has a common range -- the largest smoothing sets the lower end; try "
              f"smaller ones, e.g. fixed-r-smooth-mpc {config.CROSSBOX_SMOOTHING_MPC:g} with fixed-r-smooth 3 2")
    np.savez_compressed(outdir / f"population_size_function_fixedR{tag}.npz",
                        fixed_r_smoothing_kind=np.array([k for k, _v, _s in smooths]),
                        fixed_r_smoothing_value=np.array([v for _k, v, _s in smooths]),
                        fixed_r_smoothing_mpc=np.array([e["sigma_mpc"] for e in entries]),
                        fixed_r_namespaces=np.array([e["namespace"] for e in entries]),
                        box_mpc=np.array([box.get(s, np.nan) for s in snaps]), **arrs)
    print(f"-> {path} (+ .npz)")
    return path


def main():
    parser = make_parser("Population-level void statistics across all snapshots "
                         "(same catalogs and sample cuts as the correlation analyses).")
    parser.add_argument("--panel-snaps", type=int, nargs="+", default=None,
                        help="snapshots for the distribution overlays "
                             f"(default: config.PANEL_SNAPSHOTS = {config.PANEL_SNAPSHOTS})")
    parser.add_argument("--other-estimator", action="store_true",
                        help="abundance figure: overlay the OTHER estimator's catalogue (dtfe beside ps or the "
                             "reverse) where its grids exist -- computed into the cache when not there yet")
    parser.add_argument("--keep-overlaps", action="store_true",
                        help="abundance: count every resolved void with a radius, not only the distinct ones (none "
                             "whose centre lies inside a deeper void's min(R_v, R_eff))")
    parser.add_argument("--size-rv", choices=SIZE_RV_CHOICES, default=SIZE_FUNCTION_RV,
                        help="the R_v definition of the size function figure (default delta08: where the enclosed "
                             "contrast of the smoothed field reaches -0.8, the theory's radius; crossing while a "
                             "catalogue lacks it; delta08raw: the same on the raw density about the same centres, "
                             "measured and cached on first use)")
    parser.add_argument("--size-smooth", type=float, default=SIZE_SMOOTH_CELLS, metavar="CELLS",
                        help=f"the size function's own catalogue smoothing in cells (default {SIZE_SMOOTH_CELLS:g}: at "
                             "the production 10 cells the delta08 radius is below sigma and mostly undefined); 0 = "
                             "the catalogue of the other figures")
    parser.add_argument("--size-fixed-r", action="store_true",
                        help="make ONLY the size function at fixed radius (population_size_function_fixedR.png + npz): "
                             "the raw -0.8 radius of the ungated distinct voids from catalogues at several smoothings, "
                             f"over the radii where every one is complete ({FIXED_R_MIN_SIGMA:g} x the largest sigma up "
                             f"to {FIXED_R_MIN_COUNT} voids), against Vdn with Poisson and bootstrap errors, at z = 0, "
                             "0.5, 1, 2 (panel-snaps overrides); needs DTFE_VOID_GATE=none")
    parser.add_argument("--fixed-r-smooth", type=float, nargs="*", default=None, metavar="CELLS",
                        help="fixed R: the catalogues' smoothings in cells (default "
                             f"{' '.join(f'{c:g}' for c in FIXED_R_SMOOTH_CELLS)}; with fixed-r-smooth-mpc only "
                             "the ones given)")
    parser.add_argument("--fixed-r-smooth-mpc", type=float, nargs="*", default=None, metavar="MPC",
                        help="fixed R: smoothings in Mpc, the same scale in every box (e.g. the cross-box "
                             f"{config.CROSSBOX_SMOOTHING_MPC:g}); the png is tagged with the smoothings")
    rc = parser.add_mutually_exclusive_group()
    rc.add_argument("--fixed-r-recentre", dest="fixed_r_recentre", action="store_true", default=None,
                    help="fixed R: re-centre each void on the raw density (its minimum after a 1-cell smoothing, "
                         "within half the void's raw radius) before the -0.8 sphere is grown; cached apart"
                         + (" (the default)" if FIXED_R_RECENTRE else ""))
    rc.add_argument("--fixed-r-no-recentre", dest="fixed_r_recentre", action="store_false",
                    help="fixed R: the catalogue's own centres (the minima of the smoothed field)"
                         + ("" if FIXED_R_RECENTRE else " (the default)"))
    parser.add_argument("--fixed-r-distinct", choices=FIXED_R_DISTINCT_CHOICES, default=None,
                        help="fixed R: which voids are distinct: catalogue (the deepest smoothed minimum of a region "
                             "first) or raw (the largest raw -0.8 sphere first, none overlapping a kept one); "
                             f"default {FIXED_R_DISTINCT}")
    parser.add_argument("--fixed-r-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="fixed R: the radius range in Mpc instead of the common complete one (one range for a "
                             "cross-box comparison)")
    pipeline.add_smoothing_mpc_arg(parser)
    args = parser.parse_args()
    style.apply()
    pipeline.apply_smoothing_args(args, "void-population")

    use_data_root(args.data_root)      # --data-root for the trees, group catalogues and ladder too (2026-10-05)
    ztab = gc.redshift_table(args.sim)
    if args.size_fixed_r:              # only that figure (2026-10-11): its own smoothings, snapshots and gate
        fixed_r_main(args, ztab)
        return
    ladder = sorted({int(d.name.split("_")[1]) for d in env.sim_dir(args.sim).glob("snapdir_*")})

    stats, method_used, prefix_used, box, other, refused = {}, {}, {}, {}, {}, []
    panel_want = [int(s) for s in (args.panel_snaps or config.PANEL_SNAPSHOTS)]
    for s in ladder:
        prod = pipeline.products(s, sim=args.sim, method=args.method, prefix=args.prefix, data_root=args.data_root)
        try:
            cell_, box[s], _n, _src = catalog.frame_of(prod)    # the header's frame must be config's (DTFE_SIM): else refused
            cat = prod.voids()
            method_used[s], prefix_used[s] = prod.fs.method, prod.fs.prefix
            # the raw radii only where the size function is drawn from THIS catalogue (the panel, no --size-smooth):
            # each is a raw density grid read
            raw = (raw_for(prod, cat, args.size_rv)
                   if s in panel_want and not size_smooth_active(args.size_smooth) else None)
        except OSError as e:                           # no grids (FileNotFoundError: silent, as before) or an unreadable file
            if not isinstance(e, FileNotFoundError):
                print(f"  snap {s:3d}: unreadable, left out: {e}")
            continue
        except ValueError as e:                        # the frame (or a bad grid): say so, the snapshot is left out
            print(f"  snap {s:3d}: REFUSED: {e}")
            refused.append(s)
            continue
        finally:
            prod.release()
        style.note_source(prod.fs)                     # the thesis mirror's guard reads which grids these are
        stats[s] = snapshot_stats(cat, _n, args.keep_overlaps, args.size_rv, raw)
        if args.other_estimator and s in panel_want:   # only the panel snapshots are drawn: no catalogue for the rest
            oth = pipeline.products(s, sim=args.sim, method="dtfe" if method_used[s] == "ps" else "ps",
                                    data_root=args.data_root)
            try:
                other[s] = snapshot_stats(oth.voids(), _n, args.keep_overlaps, args.size_rv)
            except (FileNotFoundError, ValueError):
                pass
            finally:
                oth.release()

    if not stats:
        raise SystemExit(f"no {args.method} field grids for {args.sim} -- nothing to analyse"
                         + (f" ({len(refused)} snapshot(s) refused, see above)" if refused else ""))
    snaps = sorted(stats)
    z = np.array([ztab.get(s, np.nan) for s in snaps])
    methods = sorted(set(method_used.values()))
    print(f"{args.sim} ({'/'.join(methods)}): population statistics at {len(snaps)} snapshots, "
          f"z = {np.nanmax(z):.2f} .. {np.nanmin(z):.2f}")
    for s in snaps:
        st = stats[s]
        print(f"  snap {s:3d} (z={ztab.get(s, float('nan')):6.2f}): total={st['n_total']:5d} "
              f"bbks={st['n_bbks']:5d} resolved={st['n_resolved']:4d} deep={st['n_deep']:4d}  "
              f"R_eff={st['r_eff_pct'][1]:5.2f} Mpc  e={st['e_pct'][1]:.3f}  "
              f"p={st['p_pct'][1]:+.3f}  delta={st['delta_pct'][1]:+.3f}")

    outdir = FIGURE_ROOT / args.sim / pipeline.smoothing_tag()     # a non-default sigma: its own folder
    outdir.mkdir(parents=True, exist_ok=True)
    # the names say what they hold (2026-10-08): untagged for PS-DTFE's default grids, as every file on the T7 is;
    # another estimator, a --prefix or --keep-overlaps used to overwrite those (as the pk / profile files are tagged)
    default_prefix = {"ps": "ps_output.", "dtfe": "output."}         # io's: an estimator's own grids
    grids = "+".join(sorted({m if prefix_used[s] == default_prefix.get(m) else prefix_used[s].rstrip(".")
                             for s, m in method_used.items()}))
    name_tag = ("" if grids == "ps" else f"_{grids}") + ("_overlaps" if args.keep_overlaps else "")

    # ---- evolution figure ---------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(8, 15), sharex=True)
    ax = axes[0]
    for key, lab, c_, ls in (("n_total", "all catalog minima", "0.6", "o-"),
                             ("n_rv_all", r"minima with an $R_v$ (no ellipsoid gate)", "0.35", "s:"),
                             ("n_bbks", "BBKS-valid (correlation sample)", "tab:blue", "o-"),
                             ("n_resolved", "well-resolved", "tab:orange", "o-"),
                             ("n_deep", "deep + resolved (tracking sample)", "tab:red", "o-")):
        ax.plot(z, [stats[s][key] for s in snaps], ls, ms=4, color=c_, label=lab)
    ax.set_yscale("log"); ax.set_ylabel("void count")
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, hi_ * 4.0)                    # room above the curves for the caveat
    ax.legend(fontsize=7, loc="lower center", ncol=2)
    # the caveat (survey GAP, 2026-10-11): the well-resolved count rises and falls with the max-axis cut (10 sigma)
    # turning on as voids grow, while the minima stay nearly flat -- not voids forming at z > 5 and dying since
    ax.text(0.02, 0.97, "well-resolved counts follow the ellipsoid gate (max semi-axis "
            f"{pipeline.ellipsoid_cuts_mpc()['max_axis_mpc']:.3g} Mpc), not void formation",
            transform=ax.transAxes, fontsize=7, color="0.3", va="top")
    style.set_title(ax, f"{args.sim}: void population evolution ({'/'.join(methods)}, "
                        f"$\\sigma$={config.SMOOTHING_SIGMA_CELLS:g} cells)")

    for ax, key, lab, c_ in ((axes[1], "r_eff_pct", r"$R_{\rm eff}$ [Mpc] (well-resolved)", "tab:purple"),
                             (axes[2], "e_pct", "BBKS ellipticity e (BBKS-valid)", "tab:blue"),
                             (axes[3], "delta_pct", r"central $\delta$ (all minima)", "tab:red")):
        lo, med, hi = (np.array([stats[s][key][i] for s in snaps]) for i in range(3))
        ax.plot(z, med, "o-", ms=4, color=c_, label=f"median {lab}")
        ax.fill_between(z, lo, hi, color=c_, alpha=0.2, label="16-84%")
        ax.set_ylabel(lab); ax.legend(fontsize=8)
    # the measured radius beside R_eff (~1.75 R_v, a curvature length): the abundance sample's R_v (2026-10-08)
    lo, med, hi = (np.array([stats[s]["r_v_pct"][i] for s in snaps]) for i in range(3))
    rv_lab = r"$R_v$ [Mpc] (" + ("" if args.keep_overlaps else "distinct ") + "well-resolved)"
    axes[1].plot(z, med, "s--", ms=4, color="tab:green", label=f"median {rv_lab}")
    axes[1].fill_between(z, lo, hi, color="tab:green", alpha=0.15)
    axes[1].set_ylabel("void radius [Mpc]"); axes[1].legend(fontsize=8)
    # overlay prolateness on the ellipticity panel
    lo, med, hi = (np.array([stats[s]["p_pct"][i] for s in snaps]) for i in range(3))
    axes[2].plot(z, med, "s--", ms=4, color="tab:orange", label="median p")
    axes[2].fill_between(z, lo, hi, color="tab:orange", alpha=0.15)
    axes[2].axhline(0, color="gray", lw=0.7, ls=":")
    grf = theory.minima_shape_percentiles(None)      # all minima of a Gaussian field: the same for every spectrum
    axes[2].axhline(grf["e"][1], color="tab:blue", lw=0.9, ls="-.", alpha=0.7, label="Gaussian field, all minima: median e")
    axes[2].axhline(grf["p"][1], color="tab:orange", lw=0.9, ls="-.", alpha=0.7, label="Gaussian field: median p")
    axes[2].legend(fontsize=8)
    axes[3].set_xlabel("redshift z")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[-1].invert_xaxis()
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, outdir / f"population_evolution{name_tag}.png", dpi=config.DPI, mirror=MIRROR)
    plt.close(fig)

    # ---- distribution figure ------------------------------------------------------------
    panel = args.panel_snaps or [int(s) for s in config.PANEL_SNAPSHOTS]
    panel = [s for s in panel if s in stats]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    specs = (("_r_eff", axes[0, 0], r"$R_{\rm eff}$ [Mpc]", {}),
             ("_e", axes[0, 1], "BBKS ellipticity e", {"range": (0, 0.6)}),
             ("_p", axes[1, 0], "BBKS prolateness p", {"range": (-0.4, 0.4)}),
             ("_delta", axes[1, 1], r"central $\delta$", {"range": (-1, 0.2)}))
    for key, ax, lab, hkw in specs:
        for s in panel:
            x = stats[s][key]
            if x.size:
                ax.hist(x, bins=30, density=True, histtype="step", lw=1.6,
                        label=f"z={ztab.get(s, float('nan')):.2f} (n={x.size})", **hkw)
        ax.set_xlabel(lab); ax.set_ylabel("pdf"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    style.set_suptitle(fig, f"{args.sim}: void population distributions ({'/'.join(methods)})")
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, outdir / f"population_distributions{name_tag}.png", dpi=config.DPI,
                                      mirror=MIRROR)
    plt.close(fig)

    # ---- abundance figure (survey item 10): n(> R_eff), dn/dlnR per Mpc^3, R_eff against the central delta
    # the abundance in the MEASURED radius R_v of the distinct voids (2026-10-07; R_eff, ~1.75 R_v, is a curvature
    # length); radii in smoothing lengths, 0.3-30 sigma: fixed Mpc radii saturated n(>R) in TNG300
    sig = pipeline.smoothing_length_mpc()          # sigma = 0 counts as one cell
    radii = np.geomspace(0.3 * sig, 30.0 * sig, 40)
    edges = np.geomspace(0.3 * sig, 30.0 * sig, 16)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(panel), 1)))
    abund = {}
    for c_, s in zip(colors, panel):
        zlab = f"z={ztab.get(s, float('nan')):.2f}"
        for st_, ls, tag in ((stats[s], "-", ""), (other.get(s), "--", " (other estimator)")):
            if st_ is None:
                continue
            n_cum, err, cnt = catalog.void_abundance(st_["_r_v"], box[s], radii)
            dn, derr, dcnt = catalog.void_dn_dlnr(st_["_r_v"], box[s], edges)
            abund[f"{s}{'_other' if tag else ''}"] = (n_cum, err, dn, derr)
            ok = cnt > 0
            axes[0].errorbar(radii[ok], n_cum[ok], yerr=err[ok], color=c_, ls=ls, lw=1.4, capsize=2,
                             label=f"{zlab}{tag} (n={cnt[0]})")
            mid = np.sqrt(edges[1:] * edges[:-1])
            okd = dcnt > 0
            axes[1].errorbar(mid[okd], dn[okd], yerr=derr[okd], color=c_, ls=ls, lw=1.4, capsize=2, label=f"{zlab}{tag}")
    for ax, ylab in ((axes[0], r"$n(>R_v)$ [Mpc$^{-3}$]"), (axes[1], r"$dn/d\ln R_v$ [Mpc$^{-3}$]")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$R_v$ [Mpc]"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    last = max(panel, key=lambda s: -ztab.get(s, 0.0)) if panel else None
    if last is not None and stats[last]["_r_v"].size:
        r_res = stats[last]["_r_v"]
        d_res = stats[last]["_delta_counted"]
        hb = axes[2].hexbin(r_res, d_res, gridsize=30, xscale="log", bins="log", cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=axes[2], label="voids per cell (log)")
        axes[2].axhline(config.DEEP_VOID_THRESHOLD, color="tab:red", lw=0.8, ls=":", label="deep-void threshold")
        axes[2].set_xlabel(r"$R_v$ [Mpc]"); axes[2].set_ylabel(r"central $\delta$ (smoothed)")
        style.set_title(axes[2], f"z={ztab.get(last, float('nan')):.2f}, {'' if args.keep_overlaps else 'distinct '}"
                                 "well-resolved voids")
        axes[2].legend(fontsize=7, loc="lower right")
    style.set_suptitle(fig, f"{args.sim}: void abundance in R_v ({'/'.join(methods)}, "
                            f"{'' if args.keep_overlaps else 'distinct '}well-resolved voids, box {box[snaps[-1]]:.0f} Mpc)")
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, outdir / f"population_abundance{name_tag}.png", dpi=config.DPI, mirror=MIRROR)
    plt.close(fig)

    sf_tag = "" if args.size_rv == SIZE_FUNCTION_RV else f"_{args.size_rv}"     # a non-default R_v: its own npz
    # the png is untagged only for 'delta08', whatever the default (2026-10-11): an untagged size-function png has
    # always meant the smoothed-field delta08 radius, and a name must not change meaning
    rv_png = "" if args.size_rv == "delta08" else f"_{args.size_rv}"
    # the png is tagged whenever its catalogue is not the other figures' ('_ss5' at the default: the mirrored untagged
    # png was the production-smoothing one this morning, and a name must not change meaning); the npz, which holds
    # every figure's statistics, only off the default (2026-10-11)
    ss_png = f"_ss{float(args.size_smooth):g}" if size_smooth_active(args.size_smooth) else ""
    sf_path = outdir / f"population_size_function{name_tag}{rv_png}{ss_png}.png"
    sf_tag += size_smooth_tag(args.size_smooth)
    if size_smooth_active(args.size_smooth):
        # the size function on its own, less smoothed catalogue (2026-10-11); its axes, cut and labels in that sigma
        with smoothing_cells(args.size_smooth):
            print(f"  size function: catalogue at {config.SMOOTHING_SIGMA_CELLS:g} cells = "
                  f"{pipeline.smoothing_length_mpc():.3g} Mpc (cache namespace {pipeline._param_hash()})")
            sf_stats = size_function_stats(args, panel, args.keep_overlaps)
            sf = size_function_figure(sf_stats, [s for s in panel if s in sf_stats], ztab, box, args, methods, sf_path)
            sf["size_function_smooth_cells"] = float(config.SMOOTHING_SIGMA_CELLS)
            sf["size_function_smooth_mpc"] = float(pipeline.smoothing_length_mpc())
    else:
        sf = size_function_figure(stats, panel, ztab, box, args, methods, sf_path)
        sf["size_function_smooth_cells"] = float(config.SMOOTHING_SIGMA_CELLS)
        sf["size_function_smooth_mpc"] = float(pipeline.smoothing_length_mpc())

    np.savez_compressed(
        outdir / f"population_stats{name_tag}{sf_tag}.npz",
        snaps=np.array(snaps), redshift=z, box_mpc=np.array([box[s] for s in snaps]),
        method=np.array([method_used[s] for s in snaps]), prefix=np.array([prefix_used[s] for s in snaps]),
        abundance_radii_mpc=radii, abundance_edges_mpc=edges,
        **{k: np.array([stats[s][k] for s in snaps])
           for k in ("n_total", "n_bbks", "n_resolved", "n_deep", "n_counted", "n_no_rv", "n_overlapping",
                     "r_eff_pct", "r_v_pct", "e_pct", "p_pct", "delta_pct")},
        abundance_radius="r_v", abundance_distinct=not args.keep_overlaps,
        n_rv_all=np.array([stats[s]["n_rv_all"] for s in snaps]),
        **sf,
        **{f"abundance_{key}_{part}": np.asarray(vals[i])
           for key, vals in abund.items() for i, part in enumerate(("n_cum", "n_cum_err", "dn_dlnr", "dn_dlnr_err"))})
    print(f"-> {outdir}/population_evolution{name_tag}.png, population_distributions{name_tag}.png, "
          f"population_abundance{name_tag}.png, {sf_path.name}, "
          f"population_stats{name_tag}{sf_tag}.npz")


if __name__ == "__main__":
    main()
