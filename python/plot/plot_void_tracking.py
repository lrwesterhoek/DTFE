"""Track VOIDS through cosmic time: size, BBKS shape (e, p) and orientation evolution.

CONSISTENCY: the per-snapshot void properties come from pipeline.SnapshotProducts.voids()
-- the exact same cached catalogs (same smoothing, FFT Hessian, footprint, eigenvalue
criterion, BBKS shapes and ellipsoid fits) that feed the correlation analyses. This script
adds only the TIME LINKING:

  1. Reference voids at --snap = the deepest well-resolved entries of the pipeline catalog.
  2. IDENTITY across time from the MERGER TREES: subhalos within --sel-factor x (max
     semi-axis) of the centre are walked back along their SubLink main branches; the centre
     moves with the median periodic displacement of the surviving tracers. Before the
     earliest snapshot the branches reach (trees thin out at z >~ 10), the centre is FROZEN
     at its earliest tracked position -- proto-voids barely move in comoving coordinates --
     so the full ladder down to z=20.05 is covered; those epochs are flagged and drawn as
     open symbols.
  3. At every snapshot with field grids the tracked centre is matched to the nearest
     catalog void (within --match-factor x R_eff) and its size (the MEASURED radius R_v --
     where the smoothed density around the minimum rises back to the mean -- and the ellipsoid's
     semi-axes and R_eff), BBKS e/p, ellipsoid-fit ell/prol, central delta and major-axis
     orientation are read from that catalog entry. R_v is the size drawn (--radius r_eff for the
     ellipsoid's, 1.5-1.9x larger); like the fit, it is read whether or not the void passes the cut.

Works identically for --method dtfe and ps.  Examples:
  python3 plot/plot_void_tracking.py --sim TNG50-3-Dark --method dtfe
  python3 plot/plot_void_tracking.py --sim TNG50-3-Dark --n-voids 10
Outputs: figures/void_tracking/<sim>/void<K>_evolution.png, voids_summary.png and one .npz
per void with every tracked quantity. Those names are PS-DTFE's default grids (ps_output.) from snapshot 99; another
estimator or --prefix adds '_<method>' / '_<prefix>', another --snap '_snap<N>', --radius r_eff '_reff' to the
figures (2026-10-08: a --method dtfe or --snap run replaced the default files), and the void<K> files of this set
beyond the voids written now are removed (a smaller --n-voids, or a void skipped, left an earlier run's beside them).
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config
from dtfelib import pipeline
from dtfelib import catalog as C
from dtfelib import make_parser, use_data_root
from dtfelib.cli import DEFAULT_SNAP
from dtfelib.trees import TreeSet
from dtfelib import environment as env
from dtfelib import voids as V
from dtfelib import groupcat as gc

FIGURE_ROOT = Path(config.LOCAL_FIGURES_ROOT) / "void_tracking"


def plot_finite(ax, x, y, *fmt, **kw):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return ax.plot(x[ok], y[ok], *fmt, **kw)


def unit_cos(v, w):
    n = np.linalg.norm(v) * np.linalg.norm(w)
    return np.abs(float(np.dot(v, w) / n)) if n > 0 else np.nan


def catalog_at(snap, args):
    """Pipeline void catalog, grid size and grids (method, prefix) for one snapshot; Nones if no fields on disk.
    A ValueError -- the snapshot's frame is not config's (DTFE_SIM is not --sim), or a bad grid -- is raised to the
    caller, which says it: it used to read as 'no field grids' (2026-10-08)."""
    prod = pipeline.products(snap, sim=args.sim, method=args.method, prefix=args.prefix, data_root=args.data_root)
    try:
        cat = prod.voids()
        n = prod.fs.grid_n
        grids = (prod.fs.method, prod.fs.prefix)
    except FileNotFoundError:
        return None, None, None
    finally:
        prod.release()
    return cat, n, grids


def main():
    def extra(p):
        p.add_argument("--n-voids", type=int, default=5,
                       help="number of (deepest, well-resolved) voids to track (default 5)")
        p.add_argument("--sel-factor", type=float, default=1.5,
                       help="tracer selection radius in units of the void's max semi-axis (default 1.5)")
        p.add_argument("--match-factor", type=float, default=0.75,
                       help="max centre-to-catalog-void match distance in units of the reference R_eff (default 0.75)")
        p.add_argument("--min-mass", type=float, default=0.0,
                       help="minimum tracer subhalo mass [1e10 Msun/h] (default 0 = all tree subhalos)")
        p.add_argument("--max-tracers", type=int, default=200,
                       help="cap on tracers per void (default 200, nearest first)")
        p.add_argument("--min-tracers", type=int, default=3,
                       help="freeze the centre when fewer branches survive (default 3)")
        p.add_argument("--radius", choices=("r_v", "r_eff"), default="r_v",
                       help="the size drawn and summarized: r_v, the measured void radius (default), or r_eff, "
                            "the ellipsoid fit's (abc)^(1/3) (the matching always uses R_eff)")
    parser = make_parser("Track void size / BBKS shape (e,p) / orientation along merger-tree tracers, "
                         "using the pipeline's own void catalogs at every snapshot.")
    extra(parser)
    pipeline.add_smoothing_mpc_arg(parser)
    args = parser.parse_args()
    # keep the pipeline's cache/param hash in charge of consistency: an explicit --smooth / --smooth-mpc changes
    # SMOOTHING_SIGMA_CELLS (and, in Mpc, the footprint) exactly the way a pipeline run with that value would
    pipeline.apply_smoothing_args(args, "void-tracking")

    # ---- 1. reference-snapshot void catalog (the pipeline's own) ------------------------
    try:
        cat0, n0, grids0 = catalog_at(args.snap, args)
    except ValueError as e:                    # the frame guard (or a bad grid): its own message, not 'no grids'
        raise SystemExit(f"snapshot {args.snap} of {args.sim}: REFUSED: {e}")
    if cat0 is None:
        raise SystemExit(f"no {args.method} field grids at snapshot {args.snap} of {args.sim}")
    good = np.nonzero(cat0["well_resolved"] & cat0["deep"])[0]
    if not good.size:
        raise SystemExit("no deep, well-resolved voids in the pipeline catalog")
    order = good[np.argsort(cat0["delta_values"][good])]
    picked = order[:args.n_voids]
    print(f"{args.sim} snap {args.snap} ({args.method}): pipeline catalog has {len(cat0['coords'])} voids, "
          f"{good.size} deep+well-resolved; tracking the {picked.size} deepest")

    # ---- 2. tracer pool from the merger trees ------------------------------------------
    use_data_root(args.data_root)      # --data-root for the trees, group catalogues and ladder too (2026-10-05)
    ts = TreeSet(args.sim)
    box_ckpc_h = env.box_ckpc_h(args.sim, args.snap)   # SubLink positions are ckpc/h
    pool = ts.positions_at(args.snap, min_mass=args.min_mass)
    pool_frac = V.wrap_frac(pool["pos_ckpc_h"] / box_ckpc_h)
    print(f"tracer pool: {pool_frac.shape[0]} tree subhalos at snap {args.snap}")

    ztab = gc.redshift_table(args.sim)
    ladder = sorted({int(d.name.split("_")[1]) for d in env.sim_dir(args.sim).glob("snapdir_*")})

    # per-snapshot pipeline catalogs (cached on disk by the pipeline itself)
    catalogs = {}
    for s in ladder:
        if s > args.snap:
            continue
        try:
            cat, n, _grids = catalog_at(s, args)
        except ValueError as e:                # said, as plot_void_population does, not dropped silently
            print(f"  snap {s:3d}: REFUSED: {e}")
            continue
        if cat is not None:
            catalogs[s] = (cat, n)
    print(f"pipeline void catalogs available at {len(catalogs)} snapshots: {sorted(catalogs)}")

    outdir = FIGURE_ROOT / args.sim / pipeline.smoothing_tag()     # a non-default sigma: its own folder
    outdir.mkdir(parents=True, exist_ok=True)
    args.tag = _set_tag(args, grids0)
    summary = []

    for k, i0 in enumerate(picked):
        center0 = (cat0["coords"][i0] + 0.5) / n0
        semi0 = cat0["semi_axes"][i0]
        r_eff0 = float(np.prod(semi0) ** (1.0 / 3.0))
        r_v0 = float(C.r_v_mpc(cat0)[i0])                 # NaN: no R_v (a catalogue older than 2026-10-07)
        ref_axis = cat0["orientations"][i0][:, 0]
        r_sel_frac = args.sel_factor * semi0.max() / config.BOX_SIZE
        idx = V.select_tracers(pool_frac, center0, r_sel_frac, args.max_tracers)
        print(f"\nvoid {k}: centre frac {np.round(center0,3)}  delta_min {cat0['delta_values'][i0]:.2f}  "
              f"R_v {r_v0:.2f} Mpc  R_eff {r_eff0:.2f} Mpc  e {cat0['bbks_params'][i0,0]:.3f}  p {cat0['bbks_params'][i0,1]:+.3f}"
              f"  -> {idx.size} tracers within {r_sel_frac*config.BOX_SIZE:.1f} Mpc")
        if idx.size < args.min_tracers:
            print("   too few tracers; skipping (raise --sel-factor or lower --min-mass)")
            continue

        branches = V.tracer_branches(ts, args.snap, pool["subfind_id"][idx], box_ckpc_h)
        centers = V.track_center(branches, args.snap, center0, args.min_tracers)
        earliest = min(centers)
        n_tr = {s: sum(1 for b in branches.values() if s in b) for s in centers}
        print(f"   {len(branches)} branches walked; tree-tracked centre over snaps "
              f"{earliest}..{max(centers)}; frozen-identity fallback below snap {earliest}")

        # ---- 3. read the void from the PIPELINE CATALOG at every field snapshot --------
        snaps = sorted(set(catalogs) | set(centers), reverse=True)
        t = {q: np.full(len(snaps), np.nan) for q in
             ("r_v", "r_eff", "e", "p", "ell", "prol", "delta_min", "axis_cos",
              "a", "b", "c", "match_dist_mpc")}
        t["snap"] = np.array(snaps)
        t["redshift"] = np.array([ztab.get(s, np.nan) for s in snaps])
        t["n_tracers"] = np.array([n_tr.get(s, 0) for s in snaps], dtype=float)
        t["tree_identity"] = np.array([s >= earliest for s in snaps])
        t["well_resolved"] = np.zeros(len(snaps), dtype=bool)
        match_frac = args.match_factor * r_eff0 / config.BOX_SIZE
        missed = []
        for i, s in enumerate(snaps):
            if s not in catalogs:
                continue
            cat, n = catalogs[s]
            center = centers.get(s, centers[earliest])       # frozen identity before `earliest`
            j, dist = V.match_catalog_void(cat, center, n, match_frac)
            if j is None:
                missed.append((s, dist))
                continue
            t["well_resolved"][i] = bool(cat["well_resolved"][j])
            t["e"][i], t["p"][i] = cat["bbks_params"][j]
            t["delta_min"][i] = cat["delta_values"][j]
            t["match_dist_mpc"][i] = dist * config.BOX_SIZE
            t["r_v"][i] = C.r_v_mpc(cat)[j]                      # the measured radius, cut or no cut
            # sizes/orientation from the SAME fit formula as the catalog, but without the
            # resolution gate (config.ELLIPSOID_CUT_RULES): shallow high-z proto-voids exceed its max axis
            # and would otherwise vanish from the track; well_resolved records the gate.
            fit = pipeline.ellipsoid_fit_one(cat["eigenvalues"][j], cat["eigenvectors"][j],
                                             config.BOX_SIZE / n)
            if fit is not None:
                a, b, c = fit["semi_axes"]
                t["a"][i], t["b"][i], t["c"][i] = a, b, c
                t["r_eff"][i] = float((a * b * c) ** (1.0 / 3.0))
                t["ell"][i] = fit["ell"]
                t["prol"][i] = fit["prol"]
                t["axis_cos"][i] = unit_cos(fit["orientation"][:, 0], ref_axis)

        got = np.isfinite(t["e"])
        zs = t["redshift"][got]
        print(f"   matched to catalog voids at {int(got.sum())}/{len(catalogs)} field snapshots "
              f"(z = {np.nanmax(zs):.2f} .. {np.nanmin(zs):.2f})"
              + (f"; no match within {args.match_factor}*R_eff at snaps {[s for s,_ in missed]}"
                 if missed else ""))

        plot_void(t, k, args, outdir, r_eff0)
        np.savez_compressed(outdir / f"void{k}_track{args.tag}.npz", **t, center0_frac=center0,
                            catalog_index0=i0, method=grids0[0], prefix=grids0[1], ref_snap=args.snap)
        summary.append((k, t))

    if summary:
        plot_summary(summary, args, outdir)
    _remove_stale(outdir, args, {k for k, _t in summary})
    print("Done.")


def _set_tag(args, grids):
    """The name tag of this run's set (2026-10-08): '' for PS-DTFE's default grids from snapshot 99 (every file on the
    T7); else '_<method>' or '_<prefix>', and '_snap<N>'. The radius has its own suffix on the figures."""
    method, prefix = grids
    default_prefix = {"ps": "ps_output.", "dtfe": "output."}         # io's: an estimator's own grids
    g = method if prefix == default_prefix.get(method) else prefix.rstrip(".")
    return ("" if g == "ps" else f"_{g}") + ("" if args.snap == DEFAULT_SNAP else f"_snap{args.snap}")


def _remove_stale(outdir, args, written):
    """Remove this set's void<K> files (same tag; the figures also the same radius) that this run did not write: a
    smaller --n-voids, or a void skipped for too few tracers, left an earlier run's files beside the new ones. A run
    that tracked no void at all (a mis-set --sel-factor or --min-mass) leaves the earlier set alone, and says so."""
    if not written:
        print("  no void tracked: the earlier files of this set are left as they are")
        return
    fig = re.compile(rf"void(\d+)_evolution{re.escape(args.tag + _radius_suffix(args))}\.png")
    npz = re.compile(rf"void(\d+)_track{re.escape(args.tag)}\.npz")
    for f in outdir.iterdir():
        m = fig.fullmatch(f.name) or npz.fullmatch(f.name)
        if m and int(m.group(1)) not in written:
            f.unlink()
            print(f"  removed {f.name} (an earlier run's void, not tracked now)")


def _open_markers(ax, x, y, tree_identity, **kw):
    """Overlay open circles where the identity is frozen (pre-tree epochs)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    frz = ~np.asarray(tree_identity) & np.isfinite(x) & np.isfinite(y)
    if frz.any():
        ax.scatter(x[frz], y[frz], facecolors="none", edgecolors=kw.pop("color", "k"),
                   s=90, zorder=5, **kw)


def _sub_res_markers(ax, x, y, well_resolved, **kw):
    """Overlay open diamonds where the fit fails the resolution gate (config.ELLIPSOID_CUT_RULES)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    sub = ~np.asarray(well_resolved) & np.isfinite(x) & np.isfinite(y)
    if sub.any():
        ax.scatter(x[sub], y[sub], facecolors="none", edgecolors=kw.pop("color", "k"),
                   marker="D", s=70, zorder=5, **kw)


def _radius_suffix(args):
    """The figures drawn in R_eff get their own names (the .npz holds both radii, so it has one)."""
    return "" if args.radius == "r_v" else "_reff"


def plot_void(t, k, args, outdir, r_eff0):
    x = t["redshift"]
    xlabel = "redshift z"
    if not np.isfinite(x).any():
        x, xlabel = t["snap"].astype(float), "snapshot"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    main_q, other_q = ("r_v", "r_eff") if args.radius == "r_v" else ("r_eff", "r_v")
    labels = {"r_v": r"$R_v$ (measured) [Mpc]", "r_eff": r"$R_{\rm eff}=(abc)^{1/3}$ [Mpc]"}
    plot_finite(ax1, x, t[main_q], "o-", color="tab:purple", label=labels[main_q])
    _open_markers(ax1, x, t[main_q], t["tree_identity"], color="tab:purple")
    _sub_res_markers(ax1, x, t[main_q], t["well_resolved"], color="tab:purple")
    plot_finite(ax1, x, t[other_q], ":", color="tab:purple", lw=1.0, alpha=0.7, label=labels[other_q])
    for q, c_, lab in (("a", "0.6", "a"), ("c", "0.8", "c")):
        plot_finite(ax1, x, t[q], "--", color=c_, lw=0.9, label=f"semi-axis {lab}")
    ax1b = ax1.twinx()
    plot_finite(ax1b, x, t["delta_min"], "s--", color="tab:red", ms=4, label=r"central $\delta$")
    ax1b.axhline(config.DEEP_VOID_THRESHOLD, color="tab:red", lw=0.7, ls=":", alpha=0.6)
    ax1.set_ylabel("size [Mpc]"); ax1b.set_ylabel(r"central $\delta$")
    ax1.set_title(f"{args.sim}  void {k} (snap {args.snap}, {args.method}) — size / shape / orientation\n"
                  "pipeline-catalog measurements; open circles: frozen identity, "
                  "open diamonds: below the resolution cut")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")

    plot_finite(ax2, x, t["e"], "o-", label="BBKS ellipticity e")
    plot_finite(ax2, x, t["p"], "s-", label="BBKS prolateness p")
    plot_finite(ax2, x, t["ell"], "--", alpha=0.5, label="ellipsoid-fit 1-c/a")
    _open_markers(ax2, x, t["e"], t["tree_identity"], color="tab:blue")
    _open_markers(ax2, x, t["p"], t["tree_identity"], color="tab:orange")
    ax2.axhline(0, color="gray", lw=0.7, ls=":")
    ax2.set_ylabel("shape"); ax2.legend(fontsize=9)

    plot_finite(ax3, x, t["axis_cos"], "o-", color="tab:green",
                label=r"major axis $|\cos\theta|$ vs snap %d" % args.snap)
    _open_markers(ax3, x, t["axis_cos"], t["tree_identity"], color="tab:green")
    _sub_res_markers(ax3, x, t["axis_cos"], t["well_resolved"], color="tab:green")
    ax3b = ax3.twinx()
    plot_finite(ax3b, x, t["n_tracers"], "-", color="0.6", lw=0.9, label="surviving tracers")
    ax3.set_ylim(0, 1.05); ax3.set_ylabel("orientation drift"); ax3b.set_ylabel("tracers")
    ax3.set_xlabel(xlabel)
    h1, l1 = ax3.get_legend_handles_labels(); h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower left")

    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
    if xlabel.startswith("redshift"):
        ax3.invert_xaxis()
    fig.tight_layout()
    fig.savefig(outdir / f"void{k}_evolution{args.tag}{_radius_suffix(args)}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary, args, outdir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    for k, t in summary:
        x = t["redshift"] if np.isfinite(t["redshift"]).any() else t["snap"].astype(float)
        c = f"C{k % 10}"                 # one colour per void in both panels (the cycle gave e and p different ones)
        plot_finite(ax1, x, t[args.radius], "o-", ms=3, alpha=0.8, color=c, label=f"void {k}")
        plot_finite(ax2, x, t["e"], "o-", ms=3, alpha=0.8, color=c, label=f"void {k}")
        plot_finite(ax2, x, t["p"], "s--", ms=3, alpha=0.5, color=c)
    ax1.set_ylabel(r"$R_v$ (measured) [Mpc]" if args.radius == "r_v" else r"$R_{\rm eff}$ [Mpc]"); ax1.legend(fontsize=8)
    ax1.set_title(f"{args.sim} — tracked void evolution ({args.method}, pipeline catalogs); "
                  "solid=e, dashed=p below")
    ax2.axhline(0, color="gray", lw=0.7, ls=":")
    ax2.set_ylabel("BBKS e (solid), p (dashed)")
    ax2.set_xlabel("redshift z"); ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    ax2.invert_xaxis()
    fig.tight_layout()
    fig.savefig(outdir / f"voids_summary{args.tag}{_radius_suffix(args)}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
