"""Resolution and estimator convergence of the void statistics: are the R_v distribution, the distinct-void
abundance and the stacked profiles properties of the voids, or of the grid, the particle mass, the field estimator?

Three rows per redshift (dtfelib.convergence holds the machinery; every definition is the pipeline's: minima,
the void gate of DTFE_VOID_GATE (best none: rows (a) and (b) are R_v statistics), crossing R_v, the
distinct sample of catalog.distinct_mask, profiles.profile_grids / stack):

  (a) GRID      the snapshot's density grid block-averaged by --factors (512^3 -> 256^3, 128^3: exactly the cell
                average a coarser deposit gives), voids found at the SAME sigma in Mpc (the footprint following it),
                matched void by void with the native catalogue within --match-sigma smoothing lengths: the R_v ratio
                coarse / native of the matched voids, their centre offset, the fractions unmatched both ways, the
                abundance dn/dlnR_v and the stacked density in r / R_v (R_v in --rv-range smoothing lengths)
  (b) MASS      TNG50-3 / TNG100-3 / TNG300-3 at one common sigma (plot_void_crossbox.py's per-box .npz files, read
                only, those of this run's gate; --crossbox-dir, --crossbox-smooth-mpc): dn/dlnR_v, the median R_v
                against z and the stacked density at z. Different initial conditions: a comparison of distributions,
                cosmic variance included
  (c) ESTIMATOR DTFE against PS-DTFE on the same snapshot (output.* AND ps_output.* grids), void by void: R_v ratio
                DTFE / PS-DTFE, central delta of both, abundance; and a per-void table (txt)
A panel that cannot be drawn (no crossbox products, no snapshot with both estimators' grids) says so on the figure
and in the log, and the rest is drawn.

Outputs (figures/void_convergence/<sim>/[<smoothing tag>/]):
  convergence_<set>_z<z>.png             the three rows at one redshift
  convergence_<set>.txt                  the numbers (per z: per factor, the masses, the estimators)
  convergence_<set>.npz                  the same as arrays
  estimator_voids_<snap>[_<set>].txt     (c)'s void-by-void table when both grids exist
<set> is the method for its default grids, else the --prefix (set_name); with --prefix, row (c)'s side of
--method reads that set too. The coarse velocity is block-averaged as the run deposited it (volume or mass weighted:
convergence.velocity_weighting, from the run log; named in the txt and npz).
The coarse catalogues are cached under their own frame's namespace (python/cache/r256_s5_f5_.../..._voids_block2.npz).

Examples:
  DTFE_SIM=TNG100-3-Dark python3 plot/plot_void_convergence.py --sim TNG100-3-Dark
  DTFE_SIM=TNG100-3-Dark python3 plot/plot_void_convergence.py --sim TNG100-3-Dark --z 0 --factors 2 4 8
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dtfelib import pipeline, catalog, profiles, convergence as cv
from dtfelib import make_parser, use_data_root
from dtfelib import figures as style
from dtfelib import environment as env
from dtfelib import groupcat as gc

FIGURE_ROOT = Path(config.LOCAL_FIGURES_ROOT) / "void_convergence"
CROSSBOX_ROOT = Path(config.LOCAL_FIGURES_ROOT) / "void_crossbox"
DEFAULT_Z = (0.0, 1.0)
DEFAULT_RV_RANGE = (2.5, 5.0)        # the stacks' R_v range in smoothing lengths (plot_void_crossbox's)
CROSSBOX_SIGMA_MPC = config.CROSSBOX_SMOOTHING_MPC          # plot_void_crossbox.DEFAULT_SIGMA_MPC
Z_TOLERANCE = 0.05
PER_VOLUME = 1.0e6                   # densities per (100 Mpc)^3


def set_name(method, prefix) -> str:
    """The grid set a run reads: the method for its own default grids ('ps_output.' / 'output.'), else
    the prefix -- lagrangian.link_tag's and plot_void_population's name_tag rule. The output names carry it, so a
    --prefix run never replaces the default set's figure, txt and npz."""
    default_prefix = {"ps": "ps_output.", "dtfe": "output."}
    if not prefix or str(prefix).rstrip(".") + "." == default_prefix.get(method):
        return method
    return str(prefix).rstrip(".")


def snap_near(ladder, ztab, z):
    """The snapshot of 'ladder' nearest z within Z_TOLERANCE, or None."""
    best = min(ladder, key=lambda s: abs(float(ztab.get(s, np.inf)) - z), default=None)
    return best if best is not None and abs(float(ztab.get(best, np.inf)) - z) <= Z_TOLERANCE else None


def grid_rows(prod, args, factors, log=print) -> dict:
    """(a) at one snapshot: the native catalogue and the block-averaged ones, matched; abundance and stacks."""
    cell, box, n, _src = catalog.frame_of(prod)
    sigma = pipeline.smoothing_length_mpc()
    ref = prod.voids()
    rho = prod.density_raw
    vel = prod.fs.load("velocity", mode="ram") if (args.velocity and prod.fs.has("velocity")) else None
    ref_rows = cv.distinct_rows(ref, n)
    lo, hi = (k * sigma for k in args.rv_range)
    # the coarse velocity the way the run deposited it: the production PS-DTFE grids are volume-weighted, and a
    # mass-weighted block mean would make the coarse v_r stacks another quantity than the native one
    weighting, why = cv.velocity_weighting(prod)
    if vel is not None:
        log(f"  velocity: {weighting}-weighted block means ({why})")
    out = {"box": float(box), "n": int(n), "sigma": sigma, "levels": [],      # no velocity (--no-velocity): none said
           "velocity_weighting": weighting if vel is not None else "", "velocity_weighting_why": why}
    st = cv.stacked_profile(rho, ref, ref_rows, n, box, [lo, hi], velocity=vel, m=args.points, n_boot=args.n_boot)
    out["levels"].append({"n_grid": n, "factor": 1, "r_v": catalog.r_v_mpc(ref)[ref_rows], "stack": st,
                          "n_minima": len(ref["coords"]), "n_resolved": int(np.asarray(ref["well_resolved"]).sum()),
                          "summary": None, "match": None})
    for f in factors:
        if n % f:
            log(f"  factor {f}: {n} is not divisible by it, left out")
            continue
        test = cv.coarse_catalog(prod, f, sigma, density=rho, log=log)
        nt = n // f
        rows = cv.distinct_rows(test, nt)
        m = cv.match(ref, ref_rows, n, test, rows, nt, box, args.match_sigma * sigma)
        summ = cv.match_summary(m, sigma)
        rho_c = cv.block_average(rho, f)
        vel_c = cv.block_average_velocity(vel, rho, f, weighting) if vel is not None else None
        st = cv.stacked_profile(rho_c, test, rows, nt, box, [lo, hi], velocity=vel_c, m=args.points, n_boot=args.n_boot)
        out["levels"].append({"n_grid": nt, "factor": f, "r_v": catalog.r_v_mpc(test)[rows], "stack": st,
                              "n_minima": len(test["coords"]), "n_resolved": int(np.asarray(test["well_resolved"]).sum()),
                              "summary": summ, "match": m})
        pc = summ["ratio_pct"]
        log(f"  {nt}^3 (x{f}): {len(test['coords'])} minima, {rows.size} distinct (native {ref_rows.size}); matched "
            f"{summ['n_matched']} ({summ['n_mutual']} mutual), native unmatched {summ['frac_ref_unmatched']:.1%}, coarse "
            f"unmatched {summ['frac_test_unmatched']:.1%}; R_v ratio {pc[1]:.3f} ({pc[0]:.3f}-{pc[2]:.3f}), offset "
            f"{summ['offset_med_mpc']:.3g} Mpc = {summ['offset_med_sigma']:.2f} sigma")
    return out


def estimator_rows(s, args, log=print):
    """(c) at one snapshot: PS-DTFE (reference) against DTFE, matched; None (said) without both grids. The side of
    the run's --method reads the run's --prefix set (else a
    --prefix run's row (a) and row (c) would come from different sets); 'sets' names the two."""
    method, prefix = getattr(args, "method", "ps"), getattr(args, "prefix", None)
    pfx = {m: (prefix if m == method else None) for m in ("ps", "dtfe")}
    ps = pipeline.products(s, sim=args.sim, method="ps", prefix=pfx["ps"], data_root=args.data_root)
    dt = pipeline.products(s, sim=args.sim, method="dtfe", prefix=pfx["dtfe"], data_root=args.data_root)
    sets = {m: set_name(m, pfx[m]) for m in ("ps", "dtfe")}
    try:
        try:
            cell, box, n, _ = catalog.frame_of(ps)
            cell_d, box_d, n_d, _ = catalog.frame_of(dt)
        except (FileNotFoundError, OSError, ValueError) as e:
            log(f"  estimator: snapshot {s} lacks one estimator's grids ({type(e).__name__}; sets {sets['ps']} and "
                f"{sets['dtfe']}): row (c) not drawn")
            return None
        if n != n_d:
            log(f"  estimator: grids differ ({n}^3 PS-DTFE, {n_d}^3 DTFE): row (c) not drawn")
            return None
        sigma = pipeline.smoothing_length_mpc()
        ref, test = ps.voids(), dt.voids()
        ref_rows, rows = cv.distinct_rows(ref, n), cv.distinct_rows(test, n)
        m = cv.match(ref, ref_rows, n, test, rows, n, box, args.match_sigma * sigma)
        return {"box": float(box), "n": int(n), "match": m, "summary": cv.match_summary(m, sigma),
                "r_v_ps": catalog.r_v_mpc(ref)[ref_rows], "r_v_dtfe": catalog.r_v_mpc(test)[rows],
                "pos_ps": np.asarray(ref["coords"], float)[ref_rows] * cell, "snap": int(s), "sets": sets}
    finally:
        ps.release()
        dt.release()


def estimator_table(est, path):
    """(c)'s void-by-void table: one line per distinct PS-DTFE void."""
    m = est["match"]
    sets = est.get("sets", {"ps": "ps", "dtfe": "dtfe"})
    lines = [f"# {path.stem}: distinct PS-DTFE voids (grid set {sets['ps']}) matched to distinct DTFE voids (grid set "
             f"{sets['dtfe']}) within {m['max_dist_mpc']:.4g} Mpc",
             "# ps_row dtfe_row  x_mpc   y_mpc   z_mpc   r_v_ps  r_v_dtfe  ratio   offset_mpc  delta_ps  delta_dtfe  mutual"]
    for i in range(m["ref"].size):
        x = est["pos_ps"][i]
        ratio = m["r_v_test"][i] / m["r_v_ref"][i] if m["test"][i] >= 0 else np.nan
        lines.append(f"{m['ref'][i]:7d} {m['test'][i]:8d} {x[0]:7.3f} {x[1]:7.3f} {x[2]:7.3f} {m['r_v_ref'][i]:7.3f} "
                     f"{m['r_v_test'][i]:8.3f} {ratio:7.3f} {m['dist_mpc'][i]:10.3f} {m['delta_ref'][i]:9.4f} "
                     f"{m['delta_test'][i]:10.4f} {int(m['mutual'][i]):6d}")
    path.write_text("\n".join(lines) + "\n")


def load_crossbox(args):
    """(b): the crossbox per-box results at the common sigma UNDER THIS PROCESS'S VOID GATE (plot_void_crossbox's own
    file tag, sigma_tag: '_gatenone' under DTFE_VOID_GATE=none); [] (said) when none is there. The
    gate matters: rows (a) and (b) are R_v statistics, best taken ungated, so an ungated run must find
    the ungated files."""
    import plot_void_crossbox as cb
    d = Path(args.crossbox_dir).expanduser()
    tag = cb.sigma_tag(args.crossbox_smooth_mpc)
    out = []
    for p in sorted(d.glob(f"crossbox_*_{args.method}_{tag}.npz")):
        try:
            out.append(cb.load(p))
        except (OSError, ValueError, KeyError) as e:
            print(f"  crossbox: {p.name} unreadable ({e}), left out")
    if not out:
        print(f"  crossbox: no per-box products in {d} at sigma {args.crossbox_smooth_mpc:g} Mpc under the "
              f"{pipeline.void_gate()} gate (*_{tag}.npz): row (b) not drawn (run plot_void_crossbox.py first with the "
              f"same DTFE_VOID_GATE, or give --crossbox-dir)")
    return out


def no_panel(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=8, color="0.3", wrap=True)
    ax.set_xticks([]); ax.set_yticks([])


def mass_summary(results, z):
    """(b) at z per box: (sim, snap, n distinct, median R_v, 16, 84, dn/dlnR_v-ready R_v array)."""
    import plot_void_crossbox as cb
    rows = []
    for r in results:
        zz = np.asarray(r["redshift"], float)
        if not np.isfinite(zz).any() or np.nanmin(np.abs(zz - z)) > Z_TOLERANCE:
            continue
        snap = int(r["snaps"][np.nanargmin(np.abs(zz - z))])
        rv = cb.r_v_at(r, snap)
        rows.append((str(r["sim"]), snap, rv, float(r["box_mpc"])))
    return rows


def draw(z, grid, est, cross, args, sigma, path):
    fig, axes = plt.subplots(3, 4, figsize=(19, 13))
    edges = np.geomspace(0.3 * sigma, 30.0 * sigma, 16)
    mid = np.sqrt(edges[1:] * edges[:-1])
    colors = plt.cm.viridis(np.linspace(0.05, 0.8, 4))
    # ---- (a) grid resolution
    if grid is None:
        for ax in axes[0]:
            no_panel(ax, f"(a) no grids at z = {z:g}")
    else:
        for c, lev in zip(colors, grid["levels"]):
            lab = f"{lev['n_grid']}$^3$" + ("" if lev["factor"] == 1 else f" (x{lev['factor']} block mean)")
            s = lev["summary"]
            if s is not None:
                m = lev["match"]
                ok = m["test"] >= 0
                ratio = m["r_v_test"][ok] / m["r_v_ref"][ok]
                axes[0, 0].hist(ratio[np.isfinite(ratio)], bins=np.linspace(0.4, 1.6, 31), histtype="step", lw=1.5,
                                color=c, label=f"{lab}: {s['ratio_pct'][1]:.3f}, {s['n_matched']}/{s['n_ref']} matched")
                axes[0, 1].hist(m["dist_mpc"][ok] / sigma, bins=np.linspace(0, args.match_sigma, 21), histtype="step",
                                lw=1.5, color=c, label=f"{lab}: unmatched {s['frac_ref_unmatched']:.0%} native, "
                                                       f"{s['frac_test_unmatched']:.0%} coarse")
            dn, derr, cnt = catalog.void_dn_dlnr(lev["r_v"], grid["box"], edges)
            ok = cnt > 0
            axes[0, 2].errorbar(mid[ok], dn[ok] * PER_VOLUME, yerr=derr[ok] * PER_VOLUME, fmt="o-", ms=3, capsize=2,
                                color=c, label=f"{lab} (n={lev['r_v'].size})")
            st = lev["stack"]
            mean = st["density"]["mean"][0]
            if np.isfinite(mean).any():
                axes[0, 3].plot(st["bins"], mean, color=c, lw=1.5, label=f"{lab} ({int(st['count'][0])} voids)")
                axes[0, 3].fill_between(st["bins"], st["density"]["lo"][0], st["density"]["hi"][0], color=c, alpha=0.2)
            else:
                axes[0, 3].plot([], [], color=c, ls=":", label=f"{lab}: too few voids")
        axes[0, 0].axvline(1.0, color="gray", lw=0.7, ls=":")
        axes[0, 0].set_xlabel(r"$R_v$(coarse) / $R_v$(native), matched voids"); axes[0, 0].set_ylabel("voids")
        axes[0, 1].set_xlabel(r"centre offset / $\sigma$"); axes[0, 1].set_ylabel("voids")
        axes[0, 2].set_xscale("log"); axes[0, 2].set_yscale("log")
        axes[0, 2].set_xlabel(r"$R_v$ [Mpc]"); axes[0, 2].set_ylabel(r"$dn/d\ln R_v$ per $(100\,$Mpc$)^3$")
        axes[0, 3].axhline(1.0, color="gray", lw=0.7, ls=":")
        axes[0, 3].set_xlabel(r"$r / R_v$"); axes[0, 3].set_ylabel(r"$\rho/\bar\rho$ (stacked)")
        style.set_title(axes[0, 0], f"(a) grid resolution, same particles, sigma {sigma:.4g} Mpc")
        style.set_title(axes[0, 3], f"R_v {args.rv_range[0]:g}-{args.rv_range[1]:g} sigma")
        for ax in axes[0]:
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
    # ---- (b) mass resolution (other boxes, common sigma)
    rows = mass_summary(cross, z) if cross else []
    if not rows:
        why = "no crossbox products" if not cross else f"no box has a snapshot at z = {z:g}"
        for ax in axes[1]:
            no_panel(ax, f"(b) mass resolution not drawn: {why}\n(plot_void_crossbox.py at sigma "
                         f"{args.crossbox_smooth_mpc:g} Mpc; --crossbox-dir)")
    else:
        sig_b = float(cross[0]["sigma_mpc"])
        e_b = np.geomspace(0.3 * sig_b, 30.0 * sig_b, 16)
        m_b = np.sqrt(e_b[1:] * e_b[:-1])
        bc = dict(zip([r["sim"] for r in cross], plt.cm.tab10(np.arange(len(cross)))))
        for sim, snap, rv, box in rows:
            dn, derr, cnt = catalog.void_dn_dlnr(rv, box, e_b)
            ok = cnt > 0
            axes[1, 0].errorbar(m_b[ok], dn[ok] * PER_VOLUME, yerr=derr[ok] * PER_VOLUME, fmt="o-", ms=3, capsize=2,
                                color=bc[sim], label=f"{sim} (n={rv.size})")
            if rv.size:
                axes[1, 1].hist(rv / sig_b, bins=np.linspace(0, 10, 26), density=True, histtype="step", lw=1.5,
                                color=bc[sim], label=f"{sim}: median {np.median(rv):.3g} Mpc")
        for r in cross:
            zz = np.asarray(r["redshift"], float)
            ok = np.isfinite(zz)
            pc = np.asarray(r["r_v_pct"], float)
            axes[1, 2].plot(zz[ok], pc[ok, 1], "o-", ms=4, color=bc[r["sim"]], label=str(r["sim"]))
            axes[1, 2].fill_between(zz[ok], pc[ok, 0], pc[ok, 2], color=bc[r["sim"]], alpha=0.12)
            i = int(np.argmin(np.abs(np.asarray(r["pz_target"], float) - z)))
            if abs(float(r["pz_target"][i]) - z) < 1e-9 and np.isfinite(r["pz_density_mean"][i]).any():
                axes[1, 3].plot(r["pz_bins"], r["pz_density_mean"][i], color=bc[r["sim"]], lw=1.5,
                                label=f"{r['sim']} ({int(r['pz_count'][i])} voids)")
                axes[1, 3].fill_between(r["pz_bins"], r["pz_density_lo"][i], r["pz_density_hi"][i], color=bc[r["sim"]],
                                        alpha=0.2)
        axes[1, 0].set_xscale("log"); axes[1, 0].set_yscale("log")
        axes[1, 0].set_xlabel(r"$R_v$ [Mpc]"); axes[1, 0].set_ylabel(r"$dn/d\ln R_v$ per $(100\,$Mpc$)^3$")
        axes[1, 1].set_xlabel(r"$R_v / \sigma$"); axes[1, 1].set_ylabel("pdf")
        axes[1, 2].set_xlabel("redshift z"); axes[1, 2].set_ylabel(r"median $R_v$ [Mpc] (16-84%)")
        axes[1, 2].invert_xaxis()
        axes[1, 3].axhline(1.0, color="gray", lw=0.7, ls=":")
        axes[1, 3].set_xlabel(r"$r / R_v$"); axes[1, 3].set_ylabel(r"$\rho/\bar\rho$ (stacked)")
        style.set_title(axes[1, 0], f"(b) particle mass: boxes at sigma {sig_b:.5g} Mpc (different ICs)")
        for ax in axes[1]:
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
    # ---- (c) estimator
    if est is None:
        for ax in axes[2]:
            no_panel(ax, f"(c) estimator not drawn: no snapshot at z = {z:g}\nwith both DTFE (output.*) and "
                         "PS-DTFE (ps_output.*) grids")
    else:
        m, s = est["match"], est["summary"]
        ok = m["test"] >= 0
        ratio = m["r_v_test"][ok] / m["r_v_ref"][ok]
        axes[2, 0].hist(ratio[np.isfinite(ratio)], bins=np.linspace(0.4, 1.6, 31), histtype="step", lw=1.5,
                        color="tab:blue", label=f"median {s['ratio_pct'][1]:.3f}, {s['n_matched']}/{s['n_ref']} matched")
        axes[2, 0].axvline(1.0, color="gray", lw=0.7, ls=":")
        axes[2, 0].set_xlabel(r"$R_v$(DTFE) / $R_v$(PS-DTFE), matched voids")
        axes[2, 1].scatter(m["delta_ref"][ok], m["delta_test"][ok], s=8, color="tab:blue")
        lim = [min(-1.0, np.nanmin(m["delta_ref"]) if m["delta_ref"].size else -1.0), 0.0]
        axes[2, 1].plot(lim, lim, color="gray", lw=0.7, ls=":")
        axes[2, 1].set_xlabel(r"central $\delta$ PS-DTFE"); axes[2, 1].set_ylabel(r"central $\delta$ DTFE")
        for rv, lab, c in ((est["r_v_ps"], "PS-DTFE", "tab:blue"), (est["r_v_dtfe"], "DTFE", "tab:orange")):
            dn, derr, cnt = catalog.void_dn_dlnr(rv, est["box"], edges)
            okd = cnt > 0
            axes[2, 2].errorbar(mid[okd], dn[okd] * PER_VOLUME, yerr=derr[okd] * PER_VOLUME, fmt="o-", ms=3, capsize=2,
                                color=c, label=f"{lab} (n={rv.size})")
        axes[2, 2].set_xscale("log"); axes[2, 2].set_yscale("log")
        axes[2, 2].set_xlabel(r"$R_v$ [Mpc]"); axes[2, 2].set_ylabel(r"$dn/d\ln R_v$ per $(100\,$Mpc$)^3$")
        axes[2, 3].hist(m["dist_mpc"][ok] / sigma, bins=np.linspace(0, args.match_sigma, 21), histtype="step", lw=1.5,
                        color="tab:blue", label=f"unmatched: {s['frac_ref_unmatched']:.0%} PS, {s['frac_test_unmatched']:.0%} DTFE")
        axes[2, 3].set_xlabel(r"centre offset / $\sigma$")
        sets = est.get("sets", {"ps": "ps", "dtfe": "dtfe"})
        style.set_title(axes[2, 0], f"(c) estimator, snapshot {est['snap']} (grid sets {sets['ps']} / {sets['dtfe']})")
        for ax in axes[2]:
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
    style.set_suptitle(fig, f"{args.sim}: void statistics against grid, particle mass and estimator, z = {z:g}")
    fig.tight_layout()
    style.save_figure(fig, path, dpi=config.DPI)
    plt.close(fig)


def summary_lines(z, snap, grid, est, cross, sigma):
    L = [f"z = {z:g} (snapshot {snap}), sigma {sigma:.6g} Mpc"]
    if grid is None:
        L.append("  (a) not measured: no grids, or the snapshot refused (see the log)")
    else:
        if grid.get("velocity_weighting"):
            L.append(f"  (a) coarse velocity: {grid['velocity_weighting']}-weighted block means "
                     f"({grid.get('velocity_weighting_why', '')})")
        L.append("  (a) grid    n_grid  minima  resolved  distinct  median R_v  matched  mutual  native_unmatched  "
                 "coarse_unmatched  R_v ratio 16/50/84       offset [Mpc] [sigma]")
        for lev in grid["levels"]:
            s = lev["summary"]
            med = np.median(lev["r_v"]) if lev["r_v"].size else np.nan
            row = (f"           {lev['n_grid']:6d} {lev['n_minima']:7d} {lev['n_resolved']:9d} {lev['r_v'].size:9d} "
                   f"{med:10.3f}")
            if s is not None:
                pc = s["ratio_pct"]
                row += (f" {s['n_matched']:8d} {s['n_mutual']:7d} {s['frac_ref_unmatched']:17.3f} "
                        f"{s['frac_test_unmatched']:17.3f}  {pc[0]:.3f} {pc[1]:.3f} {pc[2]:.3f}"
                        f"       {s['offset_med_mpc']:.3f} {s['offset_med_sigma']:.3f}")
            L.append(row)
            st = lev["stack"]
            mean = st["density"]["mean"][0]
            if np.isfinite(mean).any():
                vr = st["v_r"]["mean"][0] if "v_r" in st else None
                L.append(f"             stack: {int(st['count'][0])} voids, rho/rho_bar at r/R_v = 0.1: {mean[0]:.3f}, "
                         f"max {np.nanmax(mean):.3f} at {st['bins'][int(np.nanargmax(mean))]:.2f}"
                         + (f"; v_r max {np.nanmax(vr):.1f} km/s at {st['bins'][int(np.nanargmax(vr))]:.2f}"
                            if vr is not None and np.isfinite(vr).any() else ""))
    rows = mass_summary(cross, z) if cross else []
    if not rows:
        L.append("  (b) not drawn: no crossbox products at this z")
    for sim, s, rv, box in rows:
        pc = np.percentile(rv, (16, 50, 84)) if rv.size else np.full(3, np.nan)
        L.append(f"  (b) {sim:14s} snap {s:3d}: {rv.size:5d} distinct, {rv.size / box ** 3 * PER_VOLUME:7.1f} per "
                 f"(100 Mpc)^3, R_v {pc[0]:.3f} {pc[1]:.3f} {pc[2]:.3f} Mpc")
    if est is None:
        L.append("  (c) not drawn: no snapshot with both estimators' grids")
    else:
        s = est["summary"]
        pc = s["ratio_pct"]
        sets = est.get("sets", {"ps": "ps", "dtfe": "dtfe"})
        if sets != {"ps": "ps", "dtfe": "dtfe"}:
            L.append(f"  (c) grid sets: PS-DTFE {sets['ps']}, DTFE {sets['dtfe']}")
        L.append(f"  (c) PS-DTFE {s['n_ref']} vs DTFE {s['n_test']} distinct: matched {s['n_matched']} ({s['n_mutual']} "
                 f"mutual), unmatched {s['frac_ref_unmatched']:.3f} / {s['frac_test_unmatched']:.3f}, R_v ratio "
                 f"DTFE/PS {pc[0]:.3f} {pc[1]:.3f} {pc[2]:.3f}, central delta DTFE - PS {s['delta_diff_med']:+.4f}")
    return L


def main(argv=None):
    p = make_parser("Resolution (grid, particle mass) and estimator convergence of the void statistics.")
    p.add_argument("--z", type=float, nargs="+", default=list(DEFAULT_Z), help="the redshifts (default 0 1)")
    p.add_argument("--factors", type=int, nargs="+", default=list(cv.DEFAULT_FACTORS),
                   help="block-averaging factors of the grid (default 2 4: 512 -> 256, 128 cells)")
    p.add_argument("--match-sigma", type=float, default=cv.DEFAULT_MATCH_SIGMA,
                   help=f"match radius in smoothing lengths (default {cv.DEFAULT_MATCH_SIGMA:g})")
    p.add_argument("--rv-range", type=float, nargs=2, default=list(DEFAULT_RV_RANGE), metavar=("LO", "HI"),
                   help="the stacks' R_v range in smoothing lengths (default 2.5 5)")
    p.add_argument("--points", type=int, default=profiles.DEFAULT_M, help="points per profile shell")
    p.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples of the stacks")
    p.add_argument("--no-velocity", dest="velocity", action="store_false",
                   help="skip the velocity grid (the stacks' v_r, in the txt and npz, is then left out; saves 1.6 GB "
                        "at 512 cubed)")
    p.add_argument("--crossbox-dir", type=Path, default=CROSSBOX_ROOT,
                   help=f"where plot_void_crossbox.py's per-box .npz files are (default {CROSSBOX_ROOT})")
    p.add_argument("--crossbox-smooth-mpc", type=float, default=CROSSBOX_SIGMA_MPC,
                   help=f"their common sigma in Mpc (default {CROSSBOX_SIGMA_MPC})")
    p.add_argument("--out", type=Path, default=None, help=f"output folder (default {FIGURE_ROOT}/<sim>)")
    pipeline.add_smoothing_mpc_arg(p)
    args = p.parse_args(argv)
    style.apply()
    pipeline.apply_smoothing_args(args, "void-convergence")
    method = "ps" if args.method == "auto" else args.method
    args.method = method
    name = set_name(method, args.prefix)                 # 'ps' / 'dtfe' for the default grids: untagged names
    tag = "" if name == method else f"_{name}"
    use_data_root(args.data_root)
    ztab = gc.redshift_table(args.sim)
    ladder = sorted({int(d.name.split("_")[1]) for d in env.sim_dir(args.sim, args.data_root).glob("snapdir_*")})
    sigma = pipeline.smoothing_length_mpc()
    out_dir = (args.out or FIGURE_ROOT / args.sim / pipeline.smoothing_tag()).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cross = load_crossbox(args)
    lines, arrays, written = [f"{args.sim} ({name}), sigma {sigma:.6g} Mpc = {config.SMOOTHING_SIGMA_CELLS:g} cells, "
                              f"factors {args.factors}, match within {args.match_sigma:g} sigma"], {}, []
    for z in args.z:
        snap = snap_near(ladder, ztab, z)
        grid = est = None
        print(f"\n== z = {z:g}: snapshot {snap}")
        if snap is not None:
            prod = pipeline.products(snap, sim=args.sim, method=method, prefix=args.prefix, data_root=args.data_root)
            try:
                grid = grid_rows(prod, args, args.factors)
            except FileNotFoundError as e:
                print(f"  (a): no {method} grids at snapshot {snap} ({e})")
            except ValueError as e:                    # the header's frame is not config's (DTFE_SIM): refused, said
                print(f"  (a): snapshot {snap} REFUSED: {e}")
            finally:
                prod.release()
            est = estimator_rows(snap, args)
            if est is not None:
                tab = out_dir / f"estimator_voids_{snap:03d}{tag}.txt"
                estimator_table(est, tab)
                written.append(tab)
        lines += summary_lines(z, snap, grid, est, cross, sigma)
        zt = f"z{z:.2f}"
        if grid is not None:
            if grid.get("velocity_weighting"):
                arrays[f"{zt}_velocity_weighting"] = np.array(grid["velocity_weighting"])
            for lev in grid["levels"]:
                k = f"{zt}_n{lev['n_grid']}"
                arrays[f"{k}_r_v"] = lev["r_v"]
                arrays[f"{k}_stack_density"] = lev["stack"]["density"]["mean"][0]
                if "v_r" in lev["stack"]:
                    arrays[f"{k}_stack_v_r"] = lev["stack"]["v_r"]["mean"][0]
                if lev["summary"] is not None:
                    arrays[f"{k}_ratio_pct"] = lev["summary"]["ratio_pct"]
                    arrays[f"{k}_unmatched"] = np.array([lev["summary"]["frac_ref_unmatched"],
                                                         lev["summary"]["frac_test_unmatched"]])
                    arrays[f"{k}_r_v_matched"] = np.stack([lev["match"]["r_v_ref"], lev["match"]["r_v_test"]])
        if est is not None:
            arrays[f"{zt}_estimator_r_v_matched"] = np.stack([est["match"]["r_v_ref"], est["match"]["r_v_test"]])
        path = out_dir / f"convergence_{name}_{zt}.png"
        draw(z, grid, est, cross, args, sigma, path)
        written.append(path)
    print("\n" + "\n".join(lines))
    table = out_dir / f"convergence_{name}.txt"
    table.write_text("\n".join(lines) + "\n")
    np.savez_compressed(out_dir / f"convergence_{name}.npz", sigma_mpc=sigma, factors=np.array(args.factors),
                        match_sigma=args.match_sigma, grid_set=np.array(name), **arrays)
    print(f"-> {out_dir}: {', '.join(q.name for q in written)}, {table.name}, convergence_{name}.npz")


if __name__ == "__main__":
    main()
