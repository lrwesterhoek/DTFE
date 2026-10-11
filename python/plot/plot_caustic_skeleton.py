"""The caustic skeleton of a '--ps-caustics' run: walls, filaments and nodes from the caustic_class grid.

For every snapshot with a caustic_class grid: dtfelib.skeleton's summary (the exclusive collapse classes'
volume and mass fractions, the fold / cusp / swallowtail / umbilic indicator fractions, and per cumulative
class -- sheet k >= 1, line k >= 2, node k = 3 -- the periodic components: count, the largest one's share,
the top sizes) as JSON + text, and a figure: the class map of the middle slice (k = 0 .. 3 with the fold
outlined), the component size distributions, and -- with --voids, from the pipeline's void catalogue --
the fold fraction of each distinct void's shells (--shells, default 0.8-1.2 R_v: around the wall, R_v being the
measured void radius; --radius r_eff for the ellipsoid's, ~1.75 R_v, with 1.0-1.3) and its distance to the nearest
sheet in units of that radius (searched within --cutout cells, default 10 smoothing lengths). The cusp and swallowtail rows are '--' unless the grid carries bit 7/8 (only --ps-caustic-cusps on
the CPU deposit sets them); a parity-only grid (--ps-linear-deposit --ps-gpu: the fold flag without the
collapse bits) is refused. On the T7 only TNG100-3-Dark z = 0 carries one (2026-10-08: its ps_caustic.* grids,
read with --prefix ps_caustic); the other boxes and snapshots need a caustics rerun.

Names (2026-10-08): caustic_skeleton_z<z>.* for the defaults; a non-default option adds its tag, so a rerun with
another one no longer overwrites them: '_<prefix>' (not ps_output), '_c6' / '_c18' (--connectivity), and with
--voids '_reff' (--radius r_eff), '_shells<a>-<b>-...' (not the radius's default shells), '_overlaps'
(--keep-overlaps), '_cut<N>' (--cutout) and the smoothing tag. A run without --voids removes the set's _voids.npz.

The control (2026-10-11): with --voids the same metrics are measured in --null-draws (default 50) realisations of
random spheres, one per void at a random cell with that void's radius, and the voids' median fold / sheet shell
fraction and wall distance are reported against the null's mean +- std with a z-score (JSON 'null', the text, and the
grey / orange bands of the void panel); --min-component N / --min-component-mass F keep only the sheet components of
at least N cells / that fraction of the box's mass for the void metrics and the null ('sheet_filter' in the JSON).
--sheet-min-density RHO keeps only the sheet cells denser than RHO x the mean (applied first: the k >= 1 sheet
percolates through the voids themselves). Their non-default values tag the set: '_null<K>', '_seed<S>', '_min<N>',
'_minm<F>', '_dens<RHO>'. --all-caustic runs every snapshot
that has a caustic_class grid under the prefix.

    python3 plot/plot_caustic_skeleton.py --sim TNG100-3-Dark --snap 99 --voids
    python3 plot/plot_caustic_skeleton.py --sim TNG50-4-Dark --snaps 99 50 --connectivity 6
    python3 plot/plot_caustic_skeleton.py --sim TNG100-3-Dark --prefix ps_caustic --all-caustic --voids --min-component 1000
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import config
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from dtfelib import make_parser, FieldSet, sim_dir, skeleton as sk, catalog, pipeline
from dtfelib import figures as style

OUTPUT_DIR = Path(config.LOCAL_FIGURES_ROOT) / "caustic_skeleton"
CLASS_COLORS = ["#f2f2f2", "#d9e6f2", "#f2b134", "#e4572e", "#7b1e3b"]   # none, k=0, k=1, k=2, k=3
MIRROR = True                   # into the thesis Figures tree too; off under --out (main sets it)


def summary_text(summ, title=""):
    lines = [title, "=" * len(title)] if title else []
    f = summ["fractions"]
    lines.append(f"cell {summ['cell_mpc']:.4f} Mpc, connectivity {summ['connectivity']}")
    lines.append(f"{'class':12s} {'volume':>8s} {'mass':>8s}")

    def pct(v):
        return "   --  " if v is None else f"{100 * v:7.3f}%"
    for name, v in f["exclusive"].items():
        lines.append(f"{name:12s} {pct(v['volume']):>8s} {pct(v['mass']):>8s}")
    cusps = summ.get("cusp_bits_present", True)
    for name, v in f["indicator"].items():
        if name in ("cusp", "swallowtail") and not cusps:
            lines.append(f"{name:12s} {'--':>8s} {'--':>8s}")       # not computed, not zero
        else:
            lines.append(f"{name:12s} {pct(v['volume']):>8s} {pct(v['mass']):>8s}")
    if not cusps:
        lines.append("cusp/swallowtail: no bit 7/8 in this grid -- set only by --ps-caustic-cusps on the CPU deposit, or none flagged")
    lines.append("")
    for name, c in summ["components"].items():
        lines.append(f"{name}: {c['n']} component(s); the largest holds {100 * (c['largest_fraction'] or 0):.1f}% of the class; "
                     f"top sizes (cells): {c['top_cells'][:5]}")
    if summ.get("sheet_min_density"):
        lines.append(f"sheet for the void metrics: cells denser than {summ['sheet_min_density']:g} x the mean only")
    filt = summ.get("sheet_filter")
    if filt:
        lines.append(f"sheet for the void metrics: components of >= {filt['min_cells']} cells and >= "
                     f"{filt['min_mass_fraction']:g} of the mass only: {filt['kept']} of {filt['n']} kept, "
                     f"{filt['cells_removed']} cells removed")
    null = summ.get("null")
    if null:
        lines.append("")
        lines.append(f"random-sphere null: {null['draws']} draws (seed {null['seed']}), one sphere per void at a random "
                     f"cell with that void's radius; box fold {pct(null['box_fold_volume'])}, sheet {pct(null['box_sheet_volume'])}")
        lines.append(f"{'metric':12s} {'voids':>8s} {'null mean':>10s} {'null std':>9s} {'z':>7s}")

        def num(v, f):
            return format(v, f) if _finite(v) else "--"
        for name, m in null["metrics"].items():
            lines.append(f"{NULL_LABELS.get(name, name):12s} {num(m['void'], '8.4f'):>8s} {num(m['null_mean'], '10.4f'):>10s} "
                         f"{num(m['null_std'], '9.4f'):>9s} {num(m['z'], '+7.2f'):>7s}")
    return "\n".join(lines) + "\n"


NULL_LABELS = {"fold": "fold shell", "sheet": "sheet shell", "dist_r": "wall dist/R"}
NULL_DRAWS = 50                 # the random-sphere null's default realisations (a non-default count tags the set)


def _finite(v):
    """A number that is not None, NaN or infinite (the null's entries before and after json_safe)."""
    return v is not None and bool(np.isfinite(v))


def _median_or_none(a):
    """The median of the finite entries; None (JSON null, no RuntimeWarning) when there are none."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else None


def void_block(cc, cat, cell_c, n_grid, cutout, shells=(1.0, 1.1, 1.2, 1.3), radius="r_eff", distinct=False,
               masks=None):
    """--voids: per well-resolved void of the catalogue with that radius ('r_eff', or 'r_v' the measured one; only
    the distinct ones with distinct=True), the fold / sheet fractions of its shells (x the radius) and the distance
    to the nearest sheet (cells, and in units of the radius; NaN + bounded beyond the cut-out, which is at most half
    the grid). The keys keep their names ('dist_reff', 'r_eff_cells') whatever the radius: 'radius' says which.
    Returns (the arrays for the _voids.npz, the summary's 'voids' entry -- medians None when undefined:
    no well-resolved void, or every one beyond the cut-out). 'masks' {'fold', 'sheet'}: the grids the metrics are
    taken on (default: cc's classes; --min-component passes a filtered sheet)."""
    r_all = catalog.r_v_mpc(cat) if radius == "r_v" else catalog.r_eff_mpc(cat)
    res = np.asarray(cat["well_resolved"]).astype(bool) & np.isfinite(r_all)
    if distinct:
        res = catalog.distinct_mask(cat, res, n_grid)
    coords = np.asarray(cat["coords"])[res]
    r_eff = r_all[res] / cell_c
    sh = sk.void_shell_fractions(cc, (coords + 0.5) / n_grid, r_eff, shells=tuple(shells), masks=masks)
    sheet = masks["sheet"] if masks is not None else sk.class_masks(cc)["cumulative"]["sheet"]
    dist, bounded = sk.nearest_wall_distance(sheet, coords,
                                             cutout_cells=min(int(cutout), int(n_grid) // 2))
    voids = {"fold": sh["fold"], "sheet": sh["sheet"], "dist_cells": dist, "dist_reff": dist / r_eff, "bounded": bounded,
             "index": np.nonzero(res)[0], "r_eff_cells": r_eff, "radius": radius, "shells": np.asarray(shells, float)}
    summary = {"n": int(res.sum()), "fold_shell_median": _median_or_none(sh["fold"]),
               "dist_reff_median": _median_or_none(dist / r_eff), "beyond_cutout": int(bounded.sum())}
    return voids, summary


def plot_skeleton(cc, summ, fold, voids, title, path):
    n = cc.shape[0]
    kmax = sk.class_masks(cc)["exclusive"]
    cls = np.full((n, n, n), 0, np.int8)
    for i, name in enumerate(("k=0", "k=1", "k=2", "k=3"), start=1):
        cls[kmax[name]] = i
    ncol = 3 if voids is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.6))
    ax = axes[0]
    mid = n // 2
    im = ax.imshow(cls[:, :, mid].T, origin="lower", cmap=ListedColormap(CLASS_COLORS), vmin=-0.5, vmax=4.5, interpolation="nearest")
    ax.contour(fold[:, :, mid].T.astype(float), levels=[0.5], colors="k", linewidths=0.4)
    cb = fig.colorbar(im, ax=ax, ticks=range(5), fraction=0.046)
    cb.set_ticklabels(["none", "k=0", "k=1 wall", "k=2 filament", "k=3 node"])
    ax.set_xlabel("x [cells]"); ax.set_ylabel("y [cells]")
    style.set_title(ax, f"collapse class, slice z = {mid} (fold outlined)")
    ax = axes[1]
    for name, color in (("sheet", "#f2b134"), ("line", "#e4572e"), ("node", "#7b1e3b")):
        sizes = np.asarray(summ["components"][name]["top_cells"], dtype=float)
        if sizes.size:
            ax.plot(np.arange(1, len(sizes) + 1), sizes, "o-", color=color, label=f"{name} ({summ['components'][name]['n']} components)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("component rank"); ax.set_ylabel("size [cells]")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    style.set_title(ax, "the largest components")
    if voids is not None:
        ax = axes[2]
        ok = np.isfinite(voids["dist_reff"])
        ax.scatter(voids["fold"], voids["dist_reff"][ok] if ok.all() else np.where(ok, voids["dist_reff"], np.nan), s=8, alpha=0.6, color="#1f5fa8")
        rl = r"$R_v$" if voids.get("radius") == "r_v" else r"$R_{\rm eff}$"
        sh_ = voids.get("shells", np.array([1.0, 1.3]))
        ax.set_xlabel(f"fold fraction of the {sh_[0]:g}-{sh_[-1]:g} {rl} shells")
        ax.set_ylabel(f"distance to the nearest sheet [{rl}]")
        null = summ.get("null")
        if null:                                    # the random spheres' median, mean +- std over the draws
            mf, md = null["metrics"]["fold"], null["metrics"]["dist_r"]
            if _finite(mf["null_mean"]) and _finite(mf["null_std"]):
                ax.axvspan(mf["null_mean"] - mf["null_std"], mf["null_mean"] + mf["null_std"], color="0.5", alpha=0.25,
                           label="random spheres: median fold, mean $\\pm$ std")
            if _finite(md["null_mean"]) and _finite(md["null_std"]):
                ax.axhspan(md["null_mean"] - md["null_std"], md["null_mean"] + md["null_std"], color="tab:orange",
                           alpha=0.2, label="random spheres: median distance")
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        # 'distinct' from what was applied (summ['voids']), not from the radius: --keep-overlaps under r_v was
        # titled 'distinct' and the distinct r_eff sample was not (2026-10-08)
        distinct = bool(summ.get("voids", {}).get("distinct", False))
        style.set_title(ax, f"{len(voids['fold'])} {'distinct ' if distinct else ''}well-resolved voids "
                            f"({int((~ok).sum())} beyond the cut-out)")
    style.set_suptitle(fig, title)
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, path, mirror=MIRROR)
    plt.close(fig)


def caustic_snapshots(sim, data_root, method, prefix):
    """The snapshots of 'sim' whose snapdir holds a caustic_class grid under 'prefix' (--all-caustic)."""
    out = []
    for d in sorted(sim_dir(sim, data_root).glob("snapdir_*")):
        try:
            if FieldSet(d, method=method, prefix=prefix).has("caustic_class"):
                out.append(int(d.name.split("_")[1]))
        except (FileNotFoundError, ValueError, OSError):
            continue
    return out


def skeleton_tag(args, prefix, default_shells):
    """The file-name tag of the non-default options (2026-10-08: only the smoothing was in the name, so another
    --connectivity, --prefix, --radius, --shells, --keep-overlaps or --cutout overwrote the default set, the
    mirrored PNG included). The void options count only with --voids."""
    tag = "" if prefix.rstrip(".") == "ps_output" else f"_{prefix.rstrip('.')}"
    tag += "" if args.connectivity == 26 else f"_c{args.connectivity}"
    if args.voids:
        shells = [float(x) for x in args.shells.split(",") if x.strip()]
        tag += "" if args.radius == "r_v" else "_reff"
        if shells != [float(x) for x in default_shells.split(",")]:
            tag += "_shells" + "-".join(f"{x:g}" for x in shells)
        tag += "_overlaps" if args.keep_overlaps else ""
        tag += "" if args.cutout is None else f"_cut{args.cutout}"
        tag += "" if args.null_draws == NULL_DRAWS else f"_null{args.null_draws}"     # the control (2026-10-11)
        tag += "" if not args.null_seed else f"_seed{args.null_seed}"
        tag += "" if args.min_component <= 1 else f"_min{args.min_component}"
        tag += "" if not args.min_component_mass > 0 else f"_minm{args.min_component_mass:g}"
        tag += "" if not args.sheet_min_density > 0 else f"_dens{args.sheet_min_density:g}"
        tag += f"_{pipeline.smoothing_tag()}" if pipeline.smoothing_tag() else ""   # a non-default sigma: its own files
    return tag


def main():
    global MIRROR
    parser = make_parser("The caustic skeleton (walls, filaments, nodes) from the caustic_class grid of a --ps-caustics run.")
    parser.add_argument("--snaps", type=int, nargs="*", default=None, help="snapshots (default: --snap)")
    parser.add_argument("--connectivity", type=int, choices=(6, 18, 26), default=26)
    parser.add_argument("--voids", action="store_true", help="per void: the shell's fold fraction and the nearest sheet (pipeline catalogue)")
    parser.add_argument("--cutout", type=int, default=None,
                        help="half-width in cells of the cut-out for the wall distance (default 10 smoothing lengths: "
                             "R_eff is 5-7 sigma, so the old 32 cells left almost every void 'beyond the cut-out')")
    parser.add_argument("--radius", choices=("r_v", "r_eff"), default="r_v",
                        help="the void radius the shells and the wall distance are in: r_v, the measured one (default), "
                             "or r_eff, the ellipsoid's (~1.75 r_v)")
    parser.add_argument("--shells", default=None,
                        help="the shells' radii in units of --radius for the fold / sheet fractions, comma-separated "
                             "(default 0.8-1.2 R_v around the wall, or 1.0-1.3 R_eff)")
    parser.add_argument("--keep-overlaps", action="store_true",
                        help="--voids: every resolved void, not only the distinct ones (none whose centre lies inside "
                             "a deeper void's min(R_v, R_eff))")
    parser.add_argument("--null-draws", type=int, default=NULL_DRAWS,
                        help=f"--voids: realisations of the random-sphere null (one sphere per void, at a random cell, "
                             f"with that void's radius); the voids' medians against it as z-scores (default {NULL_DRAWS}; "
                             "0: no null)")
    parser.add_argument("--null-seed", type=int, default=0, help="the null's random seed (default 0)")
    parser.add_argument("--min-component", type=int, default=0, metavar="CELLS",
                        help="--voids: take the shells and the wall distance on sheet components of at least this many "
                             "cells only (default 0: every sheet cell)")
    parser.add_argument("--min-component-mass", type=float, default=0.0, metavar="FRACTION",
                        help="--voids: and of at least this fraction of the box's mass (default 0)")
    parser.add_argument("--sheet-min-density", type=float, default=0.0, metavar="RHO",
                        help="--voids: count only the sheet cells denser than RHO times the mean (default 0: every one); "
                             "the k >= 1 sheet percolates through the voids, its dense part is the walls")
    parser.add_argument("--all-caustic", action="store_true",
                        help="every snapshot of the simulation that has a caustic_class grid under this prefix "
                             "(instead of --snap / --snaps)")
    pipeline.add_smoothing_mpc_arg(parser)
    parser.add_argument("--out", default=None, help=f"output folder (default {OUTPUT_DIR}/<sim>)")
    args = parser.parse_args()
    style.apply()
    pipeline.apply_smoothing_args(args, "caustic-skeleton")    # --smooth / --smooth-mpc: the --voids catalogue's
    default_shells = "0.8,0.9,1.0,1.1,1.2" if args.radius == "r_v" else "1.0,1.1,1.2,1.3"
    if args.shells is None:
        args.shells = default_shells
    out_dir = Path(args.out).expanduser() if args.out else OUTPUT_DIR / args.sim
    MIRROR = args.out is None
    method = "ps" if args.method == "auto" else args.method
    snaps = args.snaps if args.snaps else [args.snap]
    if args.all_caustic:
        snaps = caustic_snapshots(args.sim, args.data_root, method, args.prefix)
        print(f"caustic_class grids ({args.prefix or 'ps_output'}) at snapshot(s) {snaps or 'none'}")
        if len(snaps) < 2:
            print("  one snapshot shows where the walls are, not when they form at R_v: a --ps-caustics rerun of more "
                  "snapshots (050 and 033, z = 1 and 2; 25-40 min each for TNG100-3-Dark at 512^3) would add that epoch")
    failed, written = [], []
    for snap in snaps:
        snapdir = sim_dir(args.sim, args.data_root) / f"snapdir_{snap:03d}"
        try:
            fs = FieldSet(snapdir, method=method, prefix=args.prefix)
            if not fs.has("caustic_class"):
                raise FileNotFoundError(f"no caustic_class grid in {snapdir} (a --ps-caustics run writes one)")
            cc = np.rint(fs.load("caustic_class")).astype(np.int32)
            sk.class_masks(cc)                              # refuses a parity-only grid (no collapse bits) here, not later
            den = fs.density(units="mean")
            z, cell = fs.meta.redshift, fs.cell_mpc
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"  snapshot {snap:03d}: {e}")
            failed.append(snap)
            continue
        summ = sk.skeleton_summary(cc, den, cell_mpc=cell, connectivity=args.connectivity)
        summ.update(sim=args.sim, snapshot=int(snap), redshift=float(z), prefix=fs.prefix)
        voids = None
        if args.voids:
            p = pipeline.products(snap, z, sim=args.sim, method=method, prefix=args.prefix, data_root=args.data_root)
            try:
                cell_c, _box, n_grid, _src = catalog.frame_of(p)
                cat = p.voids()
                cutout = args.cutout if args.cutout is not None else int(round(10 * pipeline.smoothing_length_mpc() / cell_c))
                shells = [float(x) for x in args.shells.split(",") if x.strip()]
                cm = sk.class_masks(cc)
                masks = {"fold": cm["indicator"]["fold"], "sheet": cm["cumulative"]["sheet"]}
                if args.sheet_min_density > 0:          # before the components: a dense wall, not a flake, is kept
                    masks["sheet"] = masks["sheet"] & (np.asarray(den) > args.sheet_min_density)
                    summ["sheet_min_density"] = float(args.sheet_min_density)
                if args.min_component > 1 or args.min_component_mass > 0:
                    masks["sheet"], summ["sheet_filter"] = sk.filter_components(
                        masks["sheet"], args.min_component, den, args.min_component_mass, args.connectivity)
                voids, summ["voids"] = void_block(cc, cat, cell_c, n_grid, cutout, shells=shells, radius=args.radius,
                                                  distinct=not args.keep_overlaps, masks=masks)
                if args.null_draws > 0 and len(voids["index"]):
                    cut = min(int(cutout), int(n_grid) // 2)
                    nul = sk.random_sphere_null(masks, voids["r_eff_cells"], n_grid, shells, cut,
                                                n_draws=args.null_draws, seed=args.null_seed)
                    summ["null"] = {"draws": int(args.null_draws), "seed": int(args.null_seed),
                                    "metrics": sk.null_zscores({"fold": voids["fold"], "sheet": voids["sheet"],
                                                                "dist_r": voids["dist_reff"]}, nul),
                                    "box_fold_volume": float(masks["fold"].mean()),
                                    "box_sheet_volume": float(masks["sheet"].mean()),
                                    "bounded_fraction_mean": float(np.mean(nul["bounded_fraction"]))}
                summ["voids"].update(radius=args.radius, shells=shells, distinct=not args.keep_overlaps,
                                     sigma_mpc=pipeline.smoothing_length_mpc(), footprint_cells=int(config.FOOTPRINT_SIZE))
            except (FileNotFoundError, ValueError, OSError) as e:   # asked for and not delivered: a failed snapshot, exit 1
                print(f"    voids: FAILED: {e}")
                summ["voids_error"] = str(e)
                failed.append(snap)
            finally:
                p.release()
        base = out_dir / (f"caustic_skeleton_z{z:.2f}" + skeleton_tag(args, fs.prefix, default_shells))  # suffixes APPENDED
        base.parent.mkdir(parents=True, exist_ok=True)
        title = f"{args.sim}  z = {z:.2f}  caustic skeleton (prefix {fs.prefix.rstrip('.')})"
        Path(f"{base}.json").write_text(json.dumps(sk.json_safe(summ), indent=1))   # no bare NaN in the JSON
        Path(f"{base}.txt").write_text(summary_text(summ, title))
        if voids is not None:
            np.savez_compressed(f"{base}_voids.npz", **voids)
        else:                                       # no --voids, or they failed: an earlier run's file is not this set's
            Path(f"{base}_voids.npz").unlink(missing_ok=True)
        plot_skeleton(cc, summ, sk.class_masks(cc)["indicator"]["fold"], voids, title, Path(f"{base}.png"))
        print(summary_text(summ, title))
        written.append(base)
    print(f"\n{len(written)} snapshot(s) under {out_dir}" + (f"; failed: {failed}" if failed else ""))
    raise SystemExit(1 if failed or not written else 0)


if __name__ == "__main__":
    main()
