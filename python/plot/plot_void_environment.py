"""Void-in-void versus void-in-cloud across cosmic time (2026-10-11): which voids sit in an overdense region and are
expected to shrink, and how their sizes, depths and shapes differ from the voids that keep expanding.

Per snapshot of the ladder, every distinct well-resolved void of the pipeline catalogue (the profiles' sample) is
classified by dtfelib.classify: S-type, VOID-IN-CLOUD, when the raw density contrast enclosed within 2 R_v is
positive, R-type, VOID-IN-VOID, otherwise (Ceccarelli et al. 2013, MNRAS 434, 1435; the void-in-cloud process of
Sheth & van de Weygaert 2004). R_v is the measured void radius (catalog.r_v_mpc).

Outputs (config.figures_path('void_environment')/<sim>/):
  environment_evolution.png   the void-in-cloud fraction (bootstrap 16-84% band) and the medians of R_v, the central
                              delta and the BBKS e, p per type against 1 + z
  environment_z<z>.png        per snapshot: the R_v, central delta, Delta(<2 R_v), e, p and c/a distributions of the
                              two types, and their mean density profiles out to 2 R_v
  environment_tracking.png    the voids followed by plot_void_tracking.py (its void<K>_track.npz files, read only,
                              from --tracking-dir), typed at their reference snapshot: R_v, depth, e and p along
                              their tracks; skipped with a note when no tracking files exist for the simulation
  environment_stats.npz       per snapshot the counts and fractions, and per void its catalogue row, Delta(<2 R_v),
                              type, R_v, depth and shape
Those names are for PS-DTFE's default grids; another estimator or --prefix adds '_<method>' / '_<prefix>',
--keep-overlaps '_overlaps', a --factor other than 2 '_x<factor>', and a non-default smoothing writes into its own
folder (plot_void_population's rules).

Examples:
  python3 plot/plot_void_environment.py --sim TNG100-3-Dark
  python3 plot/plot_void_environment.py --sim TNG50-3-Dark --tracking-dir "/Volumes/Samsung T7/Illustris TNG/figures/void_tracking/TNG50-3-Dark"
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dtfelib import pipeline, catalog, classify
from dtfelib import make_parser, use_data_root
from dtfelib import environment as env
from dtfelib import groupcat as gc
from dtfelib import voids as V
from dtfelib import figures as style
from dtfelib import cli

FOLDER = "void_environment"
PCT = (16, 50, 84)
TYPE_COLORS = {"R": "tab:blue", "S": "tab:red"}
MIRROR = True
# the per-void quantities stored and drawn: (key, label, histogram range or None)
QUANTITIES = (("r_v_mpc", r"$R_v$ [Mpc]", None),
              ("delta_c", r"central $\delta$ (smoothed)", (-1.0, 0.2)),
              ("delta_enclosed", r"$\Delta(<2R_v)$", None),
              ("e", "BBKS ellipticity e", (0.0, 0.6)),
              ("p", "BBKS prolateness p", (-0.4, 0.4)),
              ("c_over_a", "axis ratio c/a", (0.0, 1.0)))
TABLE_KEYS = ("index", "delta_enclosed", "vic", "r_v_mpc", "delta_c", "e", "p", "b_over_a", "c_over_a", "r_eff_mpc")


def snapshot_summary(envd, n_boot=1000, seed=0):
    """Counts, the void-in-cloud fraction with its bootstrap band, and per type the 16/50/84% of every quantity."""
    vic = np.asarray(envd["vic"], dtype=bool)
    f, lo, hi = classify.vic_fraction(vic, n_boot=n_boot, seed=seed)
    out = {"n": int(vic.size), "n_S": int(vic.sum()), "n_R": int((~vic).sum()), "f_vic": f, "f_vic_lo": lo,
           "f_vic_hi": hi, "dropped": int(envd["dropped"])}
    for key, _lab, _rng in QUANTITIES:
        for t, sel in (("R", ~vic), ("S", vic)):
            x = np.asarray(envd[key], dtype=float)[sel]
            x = x[np.isfinite(x)]
            out[f"{key}_{t}_pct"] = np.percentile(x, PCT) if x.size else np.full(3, np.nan)
    return out


def name_tag(method_used, prefix_used, keep_overlaps, factor, definition="crossing"):
    """plot_void_population's rule (untagged for PS-DTFE's default grids), plus '_x<factor>' off the default and
    '_rv<definition>' for an R_v other than the crossing one."""
    default_prefix = {"ps": "ps_output.", "dtfe": "output."}
    grids = "+".join(sorted({m if prefix_used[s] == default_prefix.get(m) else prefix_used[s].rstrip(".")
                             for s, m in method_used.items()}))
    return (("" if grids == "ps" else f"_{grids}") + ("_overlaps" if keep_overlaps else "")
            + ("" if float(factor) == classify.ENV_FACTOR else f"_x{float(factor):g}")
            + ("" if definition == "crossing" else f"_rv{definition}"))


def plot_evolution(sims, z, summ, sim, methods, factor, path):
    fig, axes = plt.subplots(4, 1, figsize=(8, 14), sharex=True)
    x = 1.0 + np.asarray(z)
    f = np.array([summ[s]["f_vic"] for s in sims])
    lo = np.array([summ[s]["f_vic_lo"] for s in sims])
    hi = np.array([summ[s]["f_vic_hi"] for s in sims])
    ax = axes[0]
    ax.plot(x, f, "o-", color="k", ms=4, label="void-in-cloud fraction")
    ax.fill_between(x, lo, hi, color="0.5", alpha=0.25, label="16-84% bootstrap")
    for s, xi in zip(sims, x):
        ax.annotate(f"{summ[s]['n']}", (xi, summ[s]["f_vic"]), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7)
    ax.set_ylim(0, 1); ax.axhline(0.5, color="0.6", lw=0.7, ls=":")
    ax.set_ylabel(rf"$f(\Delta(<{factor:g}R_v) > 0)$"); ax.legend(fontsize=8)
    for ax, key, lab in ((axes[1], "r_v_mpc", r"$R_v$ [Mpc]"), (axes[2], "delta_c", r"central $\delta$"),
                         (axes[3], "e", "BBKS e (solid), p (dashed)")):
        for t in ("R", "S"):
            pct = np.array([summ[s][f"{key}_{t}_pct"] for s in sims])
            ax.plot(x, pct[:, 1], "o-", ms=4, color=TYPE_COLORS[t], label=f"{classify.TYPE_NAMES[t]} ({t})")
            ax.fill_between(x, pct[:, 0], pct[:, 2], color=TYPE_COLORS[t], alpha=0.15)
            if key == "e":
                pp = np.array([summ[s][f"p_{t}_pct"] for s in sims])
                ax.plot(x, pp[:, 1], "s--", ms=3, color=TYPE_COLORS[t])
        ax.set_ylabel(lab); ax.legend(fontsize=8)
    axes[3].axhline(0, color="0.6", lw=0.7, ls=":")
    axes[-1].set_xscale("log"); axes[-1].set_xlabel("1 + z")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    axes[-1].invert_xaxis()
    style.set_suptitle(fig, f"{sim}: void-in-void and void-in-cloud ({'/'.join(methods)}, distinct well-resolved voids)")
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, path, mirror=MIRROR)
    plt.close(fig)


def plot_snapshot(envd, summ, z, sim, factor, path):
    vic = np.asarray(envd["vic"], dtype=bool)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    flat = axes.ravel()
    for ax, (key, lab, rng) in zip(flat, QUANTITIES):
        vals = np.asarray(envd[key], dtype=float)
        fin = vals[np.isfinite(vals)]
        edges = np.linspace(*(rng or ((fin.min(), fin.max()) if fin.size else (0.0, 1.0))), 21)
        for t, sel in (("R", ~vic), ("S", vic)):
            v = vals[sel]
            v = v[np.isfinite(v)]
            if v.size:
                ax.hist(v, bins=edges, histtype="step", lw=1.6, color=TYPE_COLORS[t],
                        label=f"{t}: n={v.size}, median {np.median(v):.3g}")
        if key == "delta_enclosed":
            ax.axvline(classify.VIC_THRESHOLD, color="k", lw=0.8, ls=":")
        ax.set_xlabel(lab.replace("2R_v", f"{factor:g}R_v")); ax.set_ylabel("voids"); ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    ax = flat[len(QUANTITIES)]
    bins = np.asarray(envd["bins"])
    for t, sel in (("R", ~vic), ("S", vic)):
        if sel.any():
            prof = np.asarray(envd["profile"])[sel]
            ax.plot(bins, np.nanmean(prof, axis=0), color=TYPE_COLORS[t], lw=1.6, label=f"{classify.TYPE_NAMES[t]}: rho")
            ax.plot(bins, np.nanmean(classify.enclosed_contrast(bins, prof), axis=0) + 1.0, color=TYPE_COLORS[t],
                    lw=1.2, ls="--", label=r"$1 + \Delta(<r)$")
    ax.axhline(1, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel(r"$r / R_v$"); ax.set_ylabel(r"$\rho/\bar\rho$ (mean over voids)"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = flat[len(QUANTITIES) + 1]
    ax.scatter(envd["r_v_mpc"], envd["delta_enclosed"], s=10, c=np.where(vic, "tab:red", "tab:blue"), alpha=0.7)
    ax.axhline(classify.VIC_THRESHOLD, color="k", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlabel(r"$R_v$ [Mpc]"); ax.set_ylabel(rf"$\Delta(<{factor:g}R_v)$"); ax.grid(alpha=0.3)
    style.set_suptitle(fig, f"{sim}  z = {z:.2f}: {summ['n_S']} void-in-cloud, {summ['n_R']} void-in-void "
                            f"(f = {summ['f_vic']:.2f}, {summ['dropped']} beyond the wrap rule)")
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, path, mirror=MIRROR)
    plt.close(fig)


def default_tracking_dir(sim):
    """plot_void_tracking's folder for 'sim' at this smoothing: under the figure root, else on the T7's figure tree
    (the tests and this wave's runs move the figure root, the tracking files stay where they were written)."""
    for root in (Path(config.LOCAL_FIGURES_ROOT), cli._T7_DATA / "figures"):
        d = root / "void_tracking" / sim / pipeline.smoothing_tag()
        if any(d.glob("void*_track*.npz")):
            return d
    return Path(config.LOCAL_FIGURES_ROOT) / "void_tracking" / sim / pipeline.smoothing_tag()


def tracking_files(tdir, tag=""):
    """The void<K>_track<tag>.npz files of one tracking set, by K."""
    pat = re.compile(rf"void(\d+)_track{re.escape(tag)}\.npz$")
    out = {}
    for f in Path(tdir).glob("void*_track*.npz"):
        m = pat.match(f.name)
        if m:
            out[int(m.group(1))] = f
    return dict(sorted(out.items()))


def classify_tracks(files, sim, method, prefix, data_root, factor):
    """Each tracked void's type at its reference snapshot: Delta(<factor R_v) around its centre there, with R_v from
    the track (its reference row) or, when that is NaN, from the catalogue void nearest to the centre (within one
    R_eff). The centre is the track's own (center0_frac): its catalog_index0 belongs to the catalogue of the day it
    was written. Returns {K: dict(track arrays, 'type', 'delta_enclosed', 'ref_snap')}; tracks it cannot type are
    left out with a note."""
    by_ref, tracks = {}, {}
    for k, f in files.items():
        with np.load(f, allow_pickle=False) as z:
            t = {key: z[key] for key in z.files}
        ref = int(t["ref_snap"]) if "ref_snap" in t else cli.DEFAULT_SNAP
        tracks[k] = t
        by_ref.setdefault(ref, []).append(k)
    out = {}
    for ref, ks in sorted(by_ref.items()):
        meth = str(tracks[ks[0]].get("method", method))
        pref = tracks[ks[0]].get("prefix")
        pref = None if pref is None or str(pref) in ("ps_output.", "output.") else str(pref)
        prod = pipeline.products(ref, sim=sim, method=meth if pref is None else "ps", prefix=pref or prefix,
                                 data_root=data_root)
        try:
            cell, _box, n, _src = catalog.frame_of(prod)
            cat = prod.voids()
            style.note_source(prod.fs)
            dens = prod.density_raw
            for k in ks:
                t = tracks[k]
                row = np.nonzero(np.asarray(t["snap"]) == ref)[0]
                r_v = float(t["r_v"][row[0]]) if row.size else float("nan")
                c0 = np.asarray(t["center0_frac"], dtype=float)
                if not np.isfinite(r_v):
                    j, _d = V.match_catalog_void(cat, c0, n, float(np.nanmax(t["r_eff"])) / (cell * n)
                                                 if np.isfinite(t["r_eff"]).any() else 0.05)
                    r_v = float(catalog.r_v_mpc(cat)[j]) if j is not None else float("nan")
                if not np.isfinite(r_v):
                    print(f"  tracked void {k}: no R_v at snapshot {ref}, not typed")
                    continue
                g = classify.classify_grid(dens, c0[None, :], [r_v / cell], factor=factor)
                if not g["kept"][0]:
                    print(f"  tracked void {k}: {factor:g} R_v reaches beyond half the box, not typed")
                    continue
                out[k] = dict(t, type=str(g["type"][0]), delta_enclosed=float(g["delta_enclosed"][0]), ref_snap=ref,
                              r_v_ref=r_v)
        except (OSError, ValueError) as e:
            print(f"  tracking: reference snapshot {ref} unusable: {e}")
        finally:
            prod.release()
    return out


def plot_tracking(typed, sim, factor, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    specs = ((axes[0, 0], "r_v", r"$R_v$ [Mpc]"), (axes[0, 1], "delta_min", r"central $\delta$"),
             (axes[1, 0], "e", "BBKS e"), (axes[1, 1], "p", "BBKS p"))
    seen = set()
    for k, t in typed.items():
        x = 1.0 + np.asarray(t["redshift"], dtype=float)
        for ax, key, _lab in specs:
            y = np.asarray(t[key], dtype=float)
            ok = np.isfinite(y) & np.isfinite(x)
            lab = None
            if (t["type"], key) not in seen and ax is axes[0, 0]:
                lab = f"{classify.TYPE_NAMES[t['type']]} ({t['type']})"
            seen.add((t["type"], key))
            ax.plot(x[ok], y[ok], "o-", ms=3, lw=1.2, color=TYPE_COLORS[t["type"]], alpha=0.8, label=lab)
        axes[0, 0].annotate(f"{k}", (x[np.isfinite(t['r_v'])][0], t["r_v"][np.isfinite(t["r_v"])][0])
                            if np.isfinite(t["r_v"]).any() else (x[0], 0), fontsize=7)
    for ax, _key, lab in specs:
        ax.set_ylabel(lab); ax.grid(alpha=0.3, which="both"); ax.set_xscale("log")
    for ax in axes[1]:
        ax.set_xlabel("1 + z")
    axes[1, 0].invert_xaxis()
    axes[0, 0].legend(fontsize=8)
    n_s = sum(t["type"] == "S" for t in typed.values())
    style.set_suptitle(fig, f"{sim}: {len(typed)} tracked voids, {n_s} void-in-cloud at the reference snapshot "
                            f"(Delta(<{factor:g} R_v) > 0)")
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, path, mirror=MIRROR)
    plt.close(fig)


def main(argv=None):
    parser = make_parser("Void-in-void versus void-in-cloud (Delta(<2 R_v)) across the snapshots, with the sizes, "
                         "depths and shapes of the two types and the tracked voids' trajectories by type.")
    parser.add_argument("--factor", type=float, default=classify.ENV_FACTOR,
                        help=f"the enclosed contrast is taken within this many R_v (default {classify.ENV_FACTOR:g}, "
                             "Ceccarelli et al. 2013)")
    parser.add_argument("--r-v-definition", choices=catalog.R_V_DEFINITIONS, default="crossing",
                        help="which R_v: crossing (the mean, default), delta08 (enclosed contrast -0.8) or ridge (the "
                             "compensation ridge); a catalogue without it is rebuilt by the pipeline or gives no void")
    parser.add_argument("--keep-overlaps", action="store_true",
                        help="every resolved void with a radius, not only the distinct ones")
    parser.add_argument("--snaps", type=int, nargs="*", default=None, help="snapshots (default: every one with grids)")
    parser.add_argument("--tracking-dir", default=None,
                        help="folder of plot_void_tracking's void<K>_track.npz files (default: void_tracking/<sim> "
                             "under the figure root, else on the T7's figure tree)")
    parser.add_argument("--tracking-tag", default="", help="the tracking set's file tag (default: the untagged set)")
    pipeline.add_smoothing_mpc_arg(parser)
    args = parser.parse_args(argv)
    style.apply()
    pipeline.apply_smoothing_args(args, "void-environment")
    use_data_root(args.data_root)
    ztab = gc.redshift_table(args.sim)
    ladder = sorted({int(d.name.split("_")[1]) for d in env.sim_dir(args.sim).glob("snapdir_*")})
    if args.snaps:
        ladder = [s for s in ladder if s in set(args.snaps)]

    envs, summ, method_used, prefix_used, zs, refused = {}, {}, {}, {}, {}, []
    for s in ladder:
        prod = pipeline.products(s, sim=args.sim, method=args.method, prefix=args.prefix, data_root=args.data_root)
        try:
            e = classify.classify_voids(prod, factor=args.factor, distinct=not args.keep_overlaps,
                                        definition=args.r_v_definition)
            method_used[s], prefix_used[s] = prod.fs.method, prod.fs.prefix
            zs[s] = float(ztab.get(s, prod.redshift))
            style.note_source(prod.fs)
        except OSError as err:                      # no grids: silent (FileNotFoundError), unreadable: said
            if not isinstance(err, FileNotFoundError):
                print(f"  snap {s:3d}: unreadable, left out: {err}")
            continue
        except ValueError as err:                   # the frame (DTFE_SIM) or a bad grid
            print(f"  snap {s:3d}: REFUSED: {err}")
            refused.append(s)
            continue
        finally:
            prod.release()
        envs[s], summ[s] = e, snapshot_summary(e)
        st = summ[s]
        print(f"  snap {s:3d} (z={zs[s]:6.2f}): {st['n']:4d} voids, {st['n_S']:4d} void-in-cloud, {st['n_R']:4d} "
              f"void-in-void: f_vic = {st['f_vic']:.3f} [{st['f_vic_lo']:.3f}, {st['f_vic_hi']:.3f}]"
              f"  median R_v R {st['r_v_mpc_R_pct'][1]:.2f} / S {st['r_v_mpc_S_pct'][1]:.2f} Mpc"
              + (f"  ({st['dropped']} beyond the wrap rule)" if st["dropped"] else ""))
    if not envs:
        raise SystemExit(f"no {args.method} field grids for {args.sim} -- nothing to classify"
                         + (f" ({len(refused)} snapshot(s) refused, see above)" if refused else ""))
    snaps = sorted(envs, key=lambda s: -zs[s])            # high z first
    methods = sorted(set(method_used.values()))
    tag = name_tag(method_used, prefix_used, args.keep_overlaps, args.factor, args.r_v_definition)
    outdir = Path(config.figures_path(FOLDER)) / args.sim / pipeline.smoothing_tag()
    outdir.mkdir(parents=True, exist_ok=True)

    plot_evolution(snaps, [zs[s] for s in snaps], summ, args.sim, methods, args.factor,
                   outdir / f"environment_evolution{tag}.png")
    for s in snaps:
        plot_snapshot(envs[s], summ[s], zs[s], args.sim, args.factor, outdir / f"environment_z{zs[s]:.2f}{tag}.png")

    tdir = Path(args.tracking_dir).expanduser() if args.tracking_dir else default_tracking_dir(args.sim)
    files = tracking_files(tdir, args.tracking_tag)
    typed = {}
    if files:
        typed = classify_tracks(files, args.sim, args.method, args.prefix, args.data_root, args.factor)
        if typed:
            plot_tracking(typed, args.sim, args.factor, outdir / f"environment_tracking{tag}.png")
            print(f"  tracked voids typed: " + ", ".join(f"{k}: {t['type']} (Delta {t['delta_enclosed']:+.3f})"
                                                       for k, t in typed.items()))
    else:
        print(f"  no tracking files in {tdir}: the trajectories by type are skipped (plot_void_tracking.py writes them)")
    if not typed:
        (outdir / f"environment_tracking{tag}.png").unlink(missing_ok=True)   # an earlier run's is not this one's

    arrays = {"snaps": np.array(snaps), "redshift": np.array([zs[s] for s in snaps]),
              "method": np.array([method_used[s] for s in snaps]), "prefix": np.array([prefix_used[s] for s in snaps]),
              "factor": np.float64(args.factor), "vic_threshold": np.float64(classify.VIC_THRESHOLD),
              "distinct": not args.keep_overlaps}
    for key in ("n", "n_S", "n_R", "f_vic", "f_vic_lo", "f_vic_hi", "dropped"):
        arrays[key] = np.array([summ[s][key] for s in snaps])
    for key, _lab, _rng in QUANTITIES:
        for t in ("R", "S"):
            arrays[f"{key}_{t}_pct"] = np.array([summ[s][f"{key}_{t}_pct"] for s in snaps])
    for s in snaps:
        for key in TABLE_KEYS:
            arrays[f"snap{s:03d}_{key}"] = np.asarray(envs[s][key])
    if typed:
        arrays["tracked_voids"] = np.array(sorted(typed))
        arrays["tracked_type"] = np.array([typed[k]["type"] for k in sorted(typed)])
        arrays["tracked_delta_enclosed"] = np.array([typed[k]["delta_enclosed"] for k in sorted(typed)])
    np.savez_compressed(outdir / f"environment_stats{tag}.npz", **arrays)
    print(f"-> {outdir}/environment_evolution{tag}.png, environment_z<z>{tag}.png, "
          + (f"environment_tracking{tag}.png, " if typed else "") + f"environment_stats{tag}.npz")


if __name__ == "__main__":
    main()
