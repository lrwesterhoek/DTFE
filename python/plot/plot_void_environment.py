"""Void-in-void versus void-in-cloud across cosmic time: which voids sit in an overdense region and are
expected to shrink, and how their sizes, depths and shapes differ from the voids that keep expanding.

Per snapshot of the ladder, every distinct well-resolved void of the pipeline catalogue (the profiles' sample) is
classified by dtfelib.classify under one of its CRITERIA (--criterion): S-type, VOID-IN-CLOUD, being squeezed,
R-type, VOID-IN-VOID, otherwise (the void-in-cloud process of Sheth & van de Weygaert 2004). R_v is the measured
void radius (catalog.r_v_mpc).
  velocity          (default) the shell between 2 and 3 R_v has net INFALL: its volume-weighted mean
                    peculiar radial velocity is negative. Its strength is drawn as v_r / (H(z) a R_v).
  density_smoothed  the smoothed density contrast enclosed within 3 R_v is positive (the large-scale environment).
  density_raw       the raw density contrast enclosed within 2 R_v is positive (Ceccarelli et al. 2013, MNRAS 434,
                    1435): kept for comparison; the 1-2 R_v shell carries the walls' halos.
The 3 R_v shell of the first two leaves out the voids with R_v > L/6 (the wrap rule), the largest and mostly
void-in-void ones: their fractions hold for the voids below that size.
Every criterion's measure is computed for every void in one pass; the figures and the npz carry the void-in-cloud
fraction under each and their agreement (the fraction of voids two criteria label the same), the type follows the
chosen one.

Outputs (config.figures_path('void_environment')/<sim>/; a default run's names end in '_velocity', see below):
  environment_evolution.png   the void-in-cloud fraction (bootstrap 16-84% band; the other criteria's fractions
                              dotted) and the medians of R_v, the central delta, the BBKS e, p and the shell's
                              v_r / (H a R_v) per type against 1 + z
  environment_z<z>.png        per snapshot: the R_v, central delta, the three criteria's measures, e, p and c/a
                              distributions of the two types, their mean density profiles out to 2 R_v, the deciding
                              measure against R_v, the shell velocity against the smoothed Delta(<3 R_v), and the
                              fractions and agreements
  environment_tracking.png    the voids followed by plot_void_tracking.py (its void<K>_track.npz files, read only,
                              from --tracking-dir), typed at their reference snapshot: R_v, depth, e and p along
                              their tracks; skipped with a note when no tracking files exist for the simulation
  environment_stats.npz       per snapshot the counts and fractions (f_vic under the chosen criterion, f_vic_<c> under
                              each, agree_<a>__<b> per pair), and per void its catalogue row, every criterion's
                              measure and verdict, type, R_v, depth and shape
Those names are for PS-DTFE's default grids and the density_raw criterion: the untagged names keep the meaning they
had before the velocity became the default, so EVERY other criterion adds '_<criterion>', the default
one too (environment_evolution_velocity.png, environment_stats_velocity.npz). Another estimator or --prefix adds
'_<method>' / '_<prefix>', --keep-overlaps '_overlaps', a --factor other than 2 '_x<factor>' (the raw criterion's
radius), and a non-default smoothing writes into its own folder (plot_void_population's rules).

Examples:
  python3 plot/plot_void_environment.py --sim TNG100-3-Dark
  python3 plot/plot_void_environment.py --sim TNG100-3-Dark --criterion density_raw
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
              ("delta_enclosed", r"raw $\Delta(<2R_v)$", None),
              ("delta_smoothed_enclosed", r"smoothed $\Delta(<3R_v)$", None),
              ("v_shell_norm", r"$\bar v_r(2$-$3R_v)\,/\,(H a R_v)$", None),
              ("e", "BBKS ellipticity e", (0.0, 0.6)),
              ("p", "BBKS prolateness p", (-0.4, 0.4)),
              ("c_over_a", "axis ratio c/a", (0.0, 1.0)))
TABLE_KEYS = ("index", "delta_enclosed", "vic", "r_v_mpc", "delta_c", "e", "p", "b_over_a", "c_over_a", "r_eff_mpc",
              "delta_smoothed_enclosed", "v_shell_kms", "v_shell_norm", "vic_velocity", "vic_density_smoothed",
              "vic_density_raw")
# the per-void quantity each criterion thresholds at 0 (the velocity: its sign; shown in units of H a R_v)
DECIDING = {"velocity": "v_shell_norm", "density_smoothed": "delta_smoothed_enclosed", "density_raw": "delta_enclosed"}
# the criterion whose files carry no tag: density_raw, the earlier default -- the untagged
# environment_*.png / environment_stats.npz (and their thesis mirrors) hold its fraction, and a name must not change
# meaning (plot_void_population's rule), so the default velocity is tagged '_velocity' like the others
UNTAGGED_CRITERION = "density_raw"
CRIT_STYLE = {"velocity": "-", "density_smoothed": "--", "density_raw": ":"}


def criterion_text(criterion, factor=classify.ENV_FACTOR):
    """The criterion and its shell in words for the figures (no option syntax): 'mean v_r(2-3 R_v) < 0' etc."""
    if criterion == "density_raw":
        return f"raw Delta(<{float(factor):g} R_v) > 0"
    return classify.CRITERION_LABEL[criterion]


def agree_key(a, b):
    return f"agree_{a}__{b}"


def snapshot_summary(envd, n_boot=1000, seed=0):
    """Counts, the void-in-cloud fraction with its bootstrap band, and per type the 16/50/84% of every quantity."""
    vic = np.asarray(envd["vic"], dtype=bool)
    f, lo, hi = classify.vic_fraction(vic, n_boot=n_boot, seed=seed)
    out = {"n": int(vic.size), "n_S": int(vic.sum()), "n_R": int((~vic).sum()), "f_vic": f, "f_vic_lo": lo,
           "f_vic_hi": hi, "dropped": int(envd["dropped"])}
    measured = tuple(envd.get("measured", (envd.get("criterion", "density_raw"),)))
    for c in classify.CRITERIA:                   # each criterion's fraction over the voids it measured
        m = np.asarray(envd.get(DECIDING[c], np.full(vic.size, np.nan)), dtype=float)
        ok = np.isfinite(m) if c in measured else np.zeros(vic.size, dtype=bool)
        v_c = np.asarray(envd.get(f"vic_{c}", np.zeros(vic.size, dtype=bool)), dtype=bool)
        out[f"f_vic_{c}"] = float(v_c[ok].mean()) if ok.any() else float("nan")
        out[f"n_{c}"] = int(ok.sum())
    pairs = classify.agreement(envd) if "measured" in envd else {}
    for i, a in enumerate(classify.CRITERIA):
        for b in classify.CRITERIA[i + 1:]:
            frac, nb = pairs.get((a, b), (float("nan"), 0))
            out[agree_key(a, b)], out[agree_key(a, b) + "_n"] = frac, nb
    for key, _lab, _rng in QUANTITIES:
        for t, sel in (("R", ~vic), ("S", vic)):
            x = np.asarray(envd[key], dtype=float)[sel]
            x = x[np.isfinite(x)]
            out[f"{key}_{t}_pct"] = np.percentile(x, PCT) if x.size else np.full(3, np.nan)
    return out


def name_tag(method_used, prefix_used, keep_overlaps, factor, definition="crossing",
             criterion=classify.DEFAULT_CRITERION):
    """plot_void_population's rule (untagged for PS-DTFE's default grids), plus '_x<factor>' off the default,
    '_rv<definition>' for an R_v other than the crossing one and '_<criterion>' for every criterion but
    UNTAGGED_CRITERION (density_raw), the default velocity included."""
    default_prefix = {"ps": "ps_output.", "dtfe": "output."}
    grids = "+".join(sorted({m if prefix_used[s] == default_prefix.get(m) else prefix_used[s].rstrip(".")
                             for s, m in method_used.items()}))
    return (("" if grids == "ps" else f"_{grids}") + ("_overlaps" if keep_overlaps else "")
            + ("" if float(factor) == classify.ENV_FACTOR else f"_x{float(factor):g}")
            + ("" if definition == "crossing" else f"_rv{definition}")
            + ("" if criterion == UNTAGGED_CRITERION else f"_{criterion}"))


def plot_evolution(sims, z, summ, sim, methods, factor, path, criterion=classify.DEFAULT_CRITERION):
    fig, axes = plt.subplots(5, 1, figsize=(8, 17), sharex=True)
    x = 1.0 + np.asarray(z)
    f = np.array([summ[s]["f_vic"] for s in sims])
    lo = np.array([summ[s]["f_vic_lo"] for s in sims])
    hi = np.array([summ[s]["f_vic_hi"] for s in sims])
    ax = axes[0]
    ax.plot(x, f, "o-", color="k", ms=4, label=f"void-in-cloud: {criterion_text(criterion, factor)}")
    ax.fill_between(x, lo, hi, color="0.5", alpha=0.25, label="16-84% bootstrap")
    for c in classify.CRITERIA:                 # the other criteria, for comparison
        fc = np.array([summ[s].get(f"f_vic_{c}", np.nan) for s in sims], dtype=float)
        if c != criterion and np.isfinite(fc).any():
            ax.plot(x, fc, "s" + CRIT_STYLE[c], color="0.45", ms=3, lw=1.0, label=criterion_text(c, factor))
    for s, xi in zip(sims, x):
        ax.annotate(f"{summ[s]['n']}", (xi, summ[s]["f_vic"]), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7)
    ax.set_ylim(0, 1); ax.axhline(0.5, color="0.6", lw=0.7, ls=":")
    ax.set_ylabel("void-in-cloud fraction"); ax.legend(fontsize=7)
    for ax, key, lab in ((axes[1], "r_v_mpc", r"$R_v$ [Mpc]"), (axes[2], "delta_c", r"central $\delta$"),
                         (axes[3], "e", "BBKS e (solid), p (dashed)"),
                         (axes[4], "v_shell_norm", r"$\bar v_r(2$-$3R_v)\,/\,(H a R_v)$")):
        for t in ("R", "S"):
            pct = np.array([summ[s].get(f"{key}_{t}_pct", np.full(3, np.nan)) for s in sims], dtype=float)
            ax.plot(x, pct[:, 1], "o-", ms=4, color=TYPE_COLORS[t], label=f"{classify.TYPE_NAMES[t]} ({t})")
            ax.fill_between(x, pct[:, 0], pct[:, 2], color=TYPE_COLORS[t], alpha=0.15)
            if key == "e":
                pp = np.array([summ[s][f"p_{t}_pct"] for s in sims])
                ax.plot(x, pp[:, 1], "s--", ms=3, color=TYPE_COLORS[t])
        ax.set_ylabel(lab); ax.legend(fontsize=8)
    axes[3].axhline(0, color="0.6", lw=0.7, ls=":")
    axes[4].axhline(0, color="0.6", lw=0.7, ls=":")
    axes[-1].set_xscale("log"); axes[-1].set_xlabel("1 + z")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    axes[-1].invert_xaxis()
    style.set_suptitle(fig, f"{sim}: void-in-void and void-in-cloud by {criterion_text(criterion, factor)} "
                            f"({'/'.join(methods)}, distinct well-resolved voids)")
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, path, mirror=MIRROR)
    plt.close(fig)


def plot_snapshot(envd, summ, z, sim, factor, path, criterion=classify.DEFAULT_CRITERION):
    vic = np.asarray(envd["vic"], dtype=bool)
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    flat = axes.ravel()
    deciding = DECIDING[criterion]
    for ax, (key, lab, rng) in zip(flat, QUANTITIES):
        vals = np.asarray(envd.get(key, np.full(vic.size, np.nan)), dtype=float)
        fin = vals[np.isfinite(vals)]
        edges = np.linspace(*(rng or ((fin.min(), fin.max()) if fin.size and np.ptp(fin) > 0 else
                                      ((fin.min() - 0.5, fin.min() + 0.5) if fin.size else (0.0, 1.0)))), 21)
        for t, sel in (("R", ~vic), ("S", vic)):
            v = vals[sel]
            v = v[np.isfinite(v)]
            if v.size:
                ax.hist(v, bins=edges, histtype="step", lw=1.6, color=TYPE_COLORS[t],
                        label=f"{t}: n={v.size}, median {np.median(v):.3g}")
        if key in DECIDING.values():
            ax.axvline(0.0, color="k", lw=1.2 if key == deciding else 0.6, ls="-" if key == deciding else ":")
        ax.set_xlabel(lab.replace("2R_v", f"{factor:g}R_v") if key == "delta_enclosed" else lab)
        ax.set_ylabel("voids"); ax.legend(fontsize=7)
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
    ax.set_xlabel(r"$r / R_v$"); ax.set_ylabel(r"raw $\rho/\bar\rho$ (mean over voids)"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = flat[len(QUANTITIES) + 1]
    colors = np.where(vic, "tab:red", "tab:blue")
    ax.scatter(envd["r_v_mpc"], envd[deciding], s=10, c=colors, alpha=0.7)
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlabel(r"$R_v$ [Mpc]")
    ax.set_ylabel(dict((k, lab) for k, lab, _r in QUANTITIES)[deciding]); ax.grid(alpha=0.3)
    ax = flat[len(QUANTITIES) + 2]
    xs = np.asarray(envd.get("delta_smoothed_enclosed", np.full(vic.size, np.nan)), dtype=float)
    ys = np.asarray(envd.get("v_shell_norm", np.full(vic.size, np.nan)), dtype=float)
    raw = np.asarray(envd.get("vic_density_raw", np.zeros(vic.size, dtype=bool)), dtype=bool)
    for mk, sel, lab in (("o", raw, "raw: void-in-cloud"), ("x", ~raw, "raw: void-in-void")):
        ok = sel & np.isfinite(xs) & np.isfinite(ys)
        if ok.any():
            ax.scatter(xs[ok], ys[ok], s=14, marker=mk, c=colors[ok], alpha=0.7, label=lab)
    ax.axhline(0.0, color="k", lw=0.8, ls=":"); ax.axvline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"smoothed $\Delta(<3R_v)$"); ax.set_ylabel(r"$\bar v_r(2$-$3R_v)\,/\,(H a R_v)$")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, title="colour: chosen type", title_fontsize=7)
    ax = flat[len(QUANTITIES) + 3]
    ax.axis("off")
    # the type's criterion stated in the panel too: the suptitle is off when config.SHOW_TITLES is
    lines = [f"z = {z:.2f}, type by {criterion_text(criterion, factor)}", "", "void-in-cloud fraction"]
    for c in classify.CRITERIA:
        lines.append(f"  {criterion_text(c, factor)}: {summ.get(f'f_vic_{c}', float('nan')):.2f}"
                     f" (n = {summ.get(f'n_{c}', 0)})" + ("  <- chosen" if c == criterion else ""))
    lines.append("agreement (same label)")
    for i, a in enumerate(classify.CRITERIA):
        for b in classify.CRITERIA[i + 1:]:
            lines.append(f"  {a} / {b}: {summ.get(agree_key(a, b), float('nan')):.2f}"
                         f" (n = {summ.get(agree_key(a, b) + '_n', 0)})")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8, transform=ax.transAxes)
    style.set_suptitle(fig, f"{sim}  z = {z:.2f}, void-in-cloud by {criterion_text(criterion, factor)}: "
                            f"{summ['n_S']} void-in-cloud, {summ['n_R']} void-in-void "
                            f"(f = {summ['f_vic']:.2f}, {summ['dropped']} beyond the wrap rule)")
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, path, mirror=MIRROR)
    plt.close(fig)


def default_tracking_dir(sim):
    """plot_void_tracking's folder for 'sim' at this smoothing: under the figure root, else on the T7's figure tree
    (the tests and some runs move the figure root, the tracking files stay where they were written)."""
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


def classify_tracks(files, sim, method, prefix, data_root, factor, criterion=classify.DEFAULT_CRITERION):
    """Each tracked void's type at its reference snapshot under 'criterion' (its measure around the void's centre
    there: the 2-3 R_v shell's v_r, the smoothed Delta(<3 R_v) or the raw Delta(<factor R_v)), with R_v from
    the track (its reference row) or, when that is NaN, from the catalogue void nearest to the centre (within one
    R_eff). The centre is the track's own (center0_frac): its catalog_index0 belongs to the catalogue of the day it
    was written. Returns {K: dict(track arrays, 'type', 'delta_enclosed' (raw), 'measure' (the criterion's),
    'ref_snap')}; tracks it cannot type are left out with a note. Only the grid the criterion needs is read."""
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
            grids = {}
            if criterion == "velocity":
                if not getattr(prod.fs, "has", lambda name: False)("velocity"):
                    raise ValueError("no velocity grid for the velocity criterion")
                grids = {"velocity": prod.fs.load("velocity", mode="ram"), "redshift": float(prod.redshift)}
            elif criterion == "density_smoothed":
                grids = {"density_smoothed": prod.delta_smoothed + 1.0}
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
                g = classify.measure_grid(dens, c0[None, :], [r_v], cell, factor=factor, criterion=criterion,
                                          **grids)
                if not g["kept"][0]:
                    print(f"  tracked void {k}: {g['shell'][1]:g} R_v reaches beyond half the box, not typed")
                    continue
                out[k] = dict(t, type=str(g["type"][0]), delta_enclosed=float(g["delta_enclosed"][0]), ref_snap=ref,
                              r_v_ref=r_v, measure=float(g[DECIDING[criterion]][0]))
        except (OSError, ValueError) as e:
            print(f"  tracking: reference snapshot {ref} unusable: {e}")
        finally:
            prod.release()
    return out


def plot_tracking(typed, sim, factor, path, criterion=classify.DEFAULT_CRITERION):
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
    axes[0, 0].legend(fontsize=8, title=f"type by {criterion_text(criterion, factor)}", title_fontsize=8)
    n_s = sum(t["type"] == "S" for t in typed.values())
    style.set_suptitle(fig, f"{sim}: {len(typed)} tracked voids, {n_s} void-in-cloud at the reference snapshot "
                            f"({criterion_text(criterion, factor)})")
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, path, mirror=MIRROR)
    plt.close(fig)


def main(argv=None):
    parser = make_parser("Void-in-void versus void-in-cloud across the snapshots (by default: net infall in the "
                         "2-3 R_v shell), with the sizes, depths and shapes of the two types, the agreement of the "
                         "criteria and the tracked voids' trajectories by type.")
    parser.add_argument("--criterion", choices=classify.CRITERIA, default=classify.DEFAULT_CRITERION,
                        help="what makes a void void-in-cloud: velocity (default; the mean peculiar radial velocity "
                             "of the 2-3 R_v shell is negative), density_smoothed (the smoothed density contrast "
                             "within 3 R_v is positive) or density_raw (the raw one within 2 R_v, Ceccarelli et al. "
                             "2013); every criterion is measured and compared, this one sets the type")
    parser.add_argument("--factor", type=float, default=classify.ENV_FACTOR,
                        help=f"the raw density criterion's radius: its enclosed contrast is taken within this many "
                             f"R_v (default {classify.ENV_FACTOR:g}, Ceccarelli et al. 2013)")
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
                                        definition=args.r_v_definition, criterion=args.criterion)
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
        print("             f_vic " + ", ".join(f"{c} {st[f'f_vic_{c}']:.3f}" for c in classify.CRITERIA)
              + "; agree " + ", ".join(f"{a}/{b} {st[agree_key(a, b)]:.3f}" for i, a in enumerate(classify.CRITERIA)
                                       for b in classify.CRITERIA[i + 1:]))
    if not envs:
        raise SystemExit(f"no {args.method} field grids for {args.sim} -- nothing to classify"
                         + (f" ({len(refused)} snapshot(s) refused, see above)" if refused else ""))
    snaps = sorted(envs, key=lambda s: -zs[s])            # high z first
    methods = sorted(set(method_used.values()))
    tag = name_tag(method_used, prefix_used, args.keep_overlaps, args.factor, args.r_v_definition, args.criterion)
    outdir = Path(config.figures_path(FOLDER)) / args.sim / pipeline.smoothing_tag()
    outdir.mkdir(parents=True, exist_ok=True)

    plot_evolution(snaps, [zs[s] for s in snaps], summ, args.sim, methods, args.factor,
                   outdir / f"environment_evolution{tag}.png", args.criterion)
    for s in snaps:
        plot_snapshot(envs[s], summ[s], zs[s], args.sim, args.factor, outdir / f"environment_z{zs[s]:.2f}{tag}.png",
                      args.criterion)

    tdir = Path(args.tracking_dir).expanduser() if args.tracking_dir else default_tracking_dir(args.sim)
    files = tracking_files(tdir, args.tracking_tag)
    typed = {}
    if files:
        typed = classify_tracks(files, args.sim, args.method, args.prefix, args.data_root, args.factor, args.criterion)
        if typed:
            plot_tracking(typed, args.sim, args.factor, outdir / f"environment_tracking{tag}.png", args.criterion)
            print(f"  tracked voids typed by {criterion_text(args.criterion, args.factor)}: "
                  + ", ".join(f"{k}: {t['type']} ({DECIDING[args.criterion]} {t['measure']:+.3f})"
                              for k, t in typed.items()))
    else:
        print(f"  no tracking files in {tdir}: the trajectories by type are skipped (plot_void_tracking.py writes them)")
    if not typed:
        (outdir / f"environment_tracking{tag}.png").unlink(missing_ok=True)   # an earlier run's is not this one's

    arrays = {"snaps": np.array(snaps), "redshift": np.array([zs[s] for s in snaps]),
              "method": np.array([method_used[s] for s in snaps]), "prefix": np.array([prefix_used[s] for s in snaps]),
              "factor": np.float64(args.factor), "vic_threshold": np.float64(classify.VIC_THRESHOLD),
              "distinct": not args.keep_overlaps, "criterion": np.str_(args.criterion),
              "criterion_shell": np.asarray(envs[snaps[0]]["shell"], dtype=np.float64)}
    pair_keys = [agree_key(a, b) + sfx for i, a in enumerate(classify.CRITERIA) for b in classify.CRITERIA[i + 1:]
                 for sfx in ("", "_n")]
    for key in ("n", "n_S", "n_R", "f_vic", "f_vic_lo", "f_vic_hi", "dropped",
                *[f"f_vic_{c}" for c in classify.CRITERIA], *[f"n_{c}" for c in classify.CRITERIA], *pair_keys):
        arrays[key] = np.array([summ[s][key] for s in snaps])
    for key, _lab, _rng in QUANTITIES:
        for t in ("R", "S"):
            arrays[f"{key}_{t}_pct"] = np.array([summ[s][f"{key}_{t}_pct"] for s in snaps])
    for s in snaps:
        for key in TABLE_KEYS:
            if key in envs[s]:
                arrays[f"snap{s:03d}_{key}"] = np.asarray(envs[s][key])
    if typed:
        arrays["tracked_voids"] = np.array(sorted(typed))
        arrays["tracked_type"] = np.array([typed[k]["type"] for k in sorted(typed)])
        arrays["tracked_delta_enclosed"] = np.array([typed[k]["delta_enclosed"] for k in sorted(typed)])
        arrays["tracked_measure"] = np.array([typed[k]["measure"] for k in sorted(typed)])
    np.savez_compressed(outdir / f"environment_stats{tag}.npz", **arrays)
    print(f"-> {outdir}/environment_evolution{tag}.png, environment_z<z>{tag}.png, "
          + (f"environment_tracking{tag}.png, " if typed else "") + f"environment_stats{tag}.npz")


if __name__ == "__main__":
    main()
