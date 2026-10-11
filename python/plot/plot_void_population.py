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
                              --size-smooth also tags population_stats '_ss<cells>'
  population_stats.npz        every per-snapshot statistic, the abundance curves included
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

import config
from dtfelib import pipeline, catalog, theory
from dtfelib import make_parser, use_data_root
from dtfelib import figures as style
from dtfelib import environment as env
from dtfelib import groupcat as gc

FIGURE_ROOT = Path(config.LOCAL_FIGURES_ROOT) / "void_population"
PCT = (16, 50, 84)   # band percentiles
SIZE_FUNCTION_RV = "delta08"     # the R_v definition of the size function (catalog.R_V_DEFINITIONS, contract C3): the
                                 # theory's radius is where the enclosed contrast reaches DELTA_V_NONLINEAR = -0.8
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


def snapshot_stats(cat, n_grid=None, keep_overlaps=False, size_rv=SIZE_FUNCTION_RV):
    """Per-snapshot population statistics from one pipeline void catalog. The abundance sample (2026-10-07): the
    resolved voids with a measured radius R_v (catalog.r_v_mpc), only the DISTINCT ones (catalog.distinct_mask) unless
    keep_overlaps -- neighbouring minima of one void counted once. The size function's sample (2026-10-11): the same
    with R_v under 'size_rv' (r_v_definition; '_r_v_sf', its definition in 'rv_sf_definition'). 'n_rv_all' counts
    every minimum with an R_v, no ellipsoid gate and no de-duplication: the reference beside the gated counts."""
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
    if size_rv == "crossing":
        r_sf, sf_def, sf_sel = r_v, "crossing", counted
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
                    if args.size_rv not in defs else f"\nR_v: {out['size_function_rv']}"),
                 transform=axes[1].transAxes, fontsize=7, color="0.3")
    style.set_suptitle(fig, f"{args.sim}: void size function ({'/'.join(methods)}, distinct well-resolved voids) "
                            f"against {SIZE_FUNCTION_MODEL.upper()} (delta_v = {theory.DELTA_V_LINEAR}, "
                            f"delta_c = {theory.DELTA_C})")
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, path, dpi=config.DPI, mirror=MIRROR)
    plt.close(fig)
    return out


@contextlib.contextmanager
def smoothing_cells(cells):
    """The process's smoothing set to 'cells' (pipeline.set_smoothing in cells: the production footprint) for the
    block, restored after it whatever happens -- the size function's catalogue (2026-10-11)."""
    saved = (config.SMOOTHING_SIGMA_CELLS, config.FOOTPRINT_SIZE, config.SMOOTHING_SIGMA_MPC)
    try:
        pipeline.set_smoothing(cells=cells)
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
        except (OSError, ValueError) as e:
            print(f"  size function: snap {s:3d} at {config.SMOOTHING_SIGMA_CELLS:g} cells left out: {e}")
            continue
        finally:
            prod.release()
        out[s] = snapshot_stats(cat, n_, keep_overlaps, args.size_rv)
    return out


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
    parser.add_argument("--size-rv", choices=("crossing", "delta08", "ridge"), default=SIZE_FUNCTION_RV,
                        help="the R_v definition of the size function figure (default delta08: where the enclosed "
                             "contrast reaches -0.8, the theory's radius; crossing while a catalogue lacks it)")
    parser.add_argument("--size-smooth", type=float, default=SIZE_SMOOTH_CELLS, metavar="CELLS",
                        help=f"the size function's own catalogue smoothing in cells (default {SIZE_SMOOTH_CELLS:g}: at "
                             "the production 10 cells the delta08 radius is below sigma and mostly undefined); 0 = "
                             "the catalogue of the other figures")
    pipeline.add_smoothing_mpc_arg(parser)
    args = parser.parse_args()
    style.apply()
    pipeline.apply_smoothing_args(args, "void-population")

    use_data_root(args.data_root)      # --data-root for the trees, group catalogues and ladder too (2026-10-05)
    ztab = gc.redshift_table(args.sim)
    ladder = sorted({int(d.name.split("_")[1]) for d in env.sim_dir(args.sim).glob("snapdir_*")})

    stats, method_used, prefix_used, box, other, refused = {}, {}, {}, {}, {}, []
    panel_want = [int(s) for s in (args.panel_snaps or config.PANEL_SNAPSHOTS)]
    for s in ladder:
        prod = pipeline.products(s, sim=args.sim, method=args.method, prefix=args.prefix, data_root=args.data_root)
        try:
            cell_, box[s], _n, _src = catalog.frame_of(prod)    # the header's frame must be config's (DTFE_SIM): else refused
            cat = prod.voids()
            method_used[s], prefix_used[s] = prod.fs.method, prod.fs.prefix
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
        stats[s] = snapshot_stats(cat, _n, args.keep_overlaps, args.size_rv)
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

    sf_tag = "" if args.size_rv == SIZE_FUNCTION_RV else f"_{args.size_rv}"     # a non-default R_v: its own names (png, npz)
    # the png is tagged whenever its catalogue is not the other figures' ('_ss5' at the default: the mirrored untagged
    # png was the production-smoothing one this morning, and a name must not change meaning); the npz, which holds
    # every figure's statistics, only off the default (2026-10-11)
    ss_png = f"_ss{float(args.size_smooth):g}" if size_smooth_active(args.size_smooth) else ""
    sf_path = outdir / f"population_size_function{name_tag}{sf_tag}{ss_png}.png"
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
