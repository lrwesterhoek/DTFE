"""The FATE of the voids: a census over every snapshot of a box (2026-10-11).

Each distinct well-resolved void (the profile / abundance sample, catalog.distinct_mask: the FOLLOWED voids) of
every snapshot of the ladder is linked to the voids of the next one by its MATTER (dtfelib.lagrangian): its member
particles -- those within R_v of its centre -- are found again by ParticleID, and the voids holding them there are
its descendants. The descendants are drawn from EVERY minimum of the next catalogue with a usable radius (2026-10-11,
lagrangian.void_pool; the followed ones with their R_v), not from the followed ones alone: at z = 0 those are a fifth
of the minima, and the morning's census counted a void whose matter sat in an unfollowed minimum as vanished.
Classes (Lagrangian linking, lagrangian.py's constants): SURVIVE (>= 50% of the members in one followed descendant),
MERGE (two or more progenitors pass >= 50% to the same followed descendant), UNRESOLVED (survive or merge, but into
a minimum that is not followed: not distinct, not well resolved or without R_v -- the void lives on, out of the
sample), SPLIT (no 50% descendant, but two or more with >= 25% each) and VANISH (no void of the next snapshot
receives these thresholds: a void-in-cloud crushed, or its matter spread thin). The thesis's Limitations section
reads the 598 -> 453 void-count decline as 'mergers and void-in-cloud collapse'; this is that census.

Outputs (figures/void_fate/<sim>/):
  fate_census.png   (a) counts per class per interval (stacked bars), (b) the fraction of each snapshot's voids
                    still holding a FOLLOWED main descendant (survive or merge; a chain ends at an unresolved
                    descendant, which is not followed further) at the last snapshot, against log(1+z),
                    (c) class fractions against the progenitor's R_v in smoothing lengths, (d) against its central
                    delta (all intervals pooled)
  fate_table.csv    one row per (interval, progenitor void): position, R_v, central delta, members, enclosed delta,
                    class, main descendant (catalogue row), whether it is followed, how many progenitors pass half
                    their matter to it, its shared fraction, the two largest over every void and the largest in a
                    followed one
  fate_census.npz   the per-interval class counts, the chained survival and the direct --ref-snap -> last link
The per-interval links (shared-particle contingency tables) are cached in the pipeline cache
(<sim>_<method>_fate_<a>_<b>.npz, rebuilt when either catalogue's voids changed, and once for the descendant pool:
lagrangian.LINK_VERSION); the classes are taken at plot
time, so a changed lagrangian.SURVIVE_FRAC / SPLIT_FRAC needs no particle read. Those names are PS-DTFE's default
grids; another estimator or --prefix adds '_<method>' / '_<prefix>', --exclusive '_exclusive' (each particle in one
void only: the nearest centre in R_v units; the default follows C6, overlapping spheres share particles), a
non-default smoothing writes into its own folder.

Examples:
  python3 python/plot/plot_void_fate.py --sim TNG50-3-Dark
  python3 python/plot/plot_void_fate.py --sim TNG100-3-Dark --ref-snap 33
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import resource
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config
from dtfelib import pipeline, catalog
from dtfelib import lagrangian as L
from dtfelib import make_parser, use_data_root
from dtfelib import environment as env
from dtfelib import groupcat as gc
from dtfelib import figures as style

FIGURE_ROOT = Path(config.figures_path("void_fate"))
MIRROR = True
CLASS_COLORS = {"survive": "tab:green", "merge": "tab:blue", "unresolved": "yellowgreen", "split": "tab:orange",
                "vanish": "0.55"}
# drawing order: the voids that live on (followed, then not), split, and the real vanishing on top
PLOT_ORDER = ("survive", "merge", "unresolved", "split", "vanish")
CLASS_LABELS = {"unresolved": "unresolved (lives on in an unfollowed minimum)"}
REF_SNAP = 33            # z = 2: the start of the direct link to the last snapshot (--ref-snap)


def peak_rss_gb() -> float:
    """This process's peak resident memory (ru_maxrss: bytes on macOS, kB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 30) if sys.platform == "darwin" else r / (1 << 20)


def name_tag(method, prefix) -> str:
    """'' for PS-DTFE's default grids (as every file on the T7), else '_<method>' or '_<prefix>' (lagrangian.link_tag:
    the link caches carry the same tag, shared with plot_void_tracking --link particles)."""
    return L.link_tag(method, prefix)


def _vanish_note(fate) -> str:
    """How the vanished voids vanished: the median largest share of their matter that is still in ONE void of the
    next snapshot (any minimum: since 2026-10-11 a void living on in an unfollowed minimum is 'unresolved', not a
    vanish), and how many have none in any."""
    v = fate["cls"] == L.CLASS_CODE["vanish"]
    if not v.any():
        return ""
    return (f"   (vanished: median largest share {np.median(fate['f_max'][v]):.2f}, "
            f"{int((fate['f_max'][v] == 0).sum())} with no member in any void)")


def zlab(z) -> str:
    """A redshift for a label: z = 2.2e-16 (snapshot 99's header) reads as 0."""
    return "0" if abs(z) < 1e-3 else f"{z:.2g}"


def class_counts(fate) -> dict:
    return {c: int((fate["cls"] == i).sum()) for i, c in enumerate(L.CLASSES)}


def class_fraction_in_bins(x, cls, edges):
    """(fraction per class per bin (4, nb), count per bin)."""
    b = np.digitize(x, edges) - 1
    ok = (b >= 0) & (b < edges.size - 1) & np.isfinite(x)
    n = np.bincount(b[ok], minlength=edges.size - 1).astype(float)
    frac = np.zeros((len(L.CLASSES), edges.size - 1))
    for i in range(len(L.CLASSES)):
        c = np.bincount(b[ok & (cls == i)], minlength=edges.size - 1)
        frac[i] = np.divide(c, n, out=np.zeros_like(n), where=n > 0)
    return frac, n


def census(snaps, args, tag):
    """Links, fates and the per-void table for consecutive snapshots of 'snaps' (Snapshot objects, ascending in
    time), plus the direct link --ref-snap -> last. Returns (intervals, direct): intervals = [(a, b, link, fate)]."""
    intervals = []
    ref = next((s for s in snaps if s.snap == args.ref_snap), None)
    direct_wanted = ref is not None and ref is not snaps[-1]
    for a, b in zip(snaps[:-1], snaps[1:]):
        print(f"  link snap {a.snap} -> {b.snap}", flush=True)
        lk = L.cached_link(a, b, tag)
        fate = L.classify(lk)
        intervals.append((a, b, lk, fate))
        a.drop_members(keep_prog=a is ref)        # the reference snapshot's progenitors are kept for the direct link
        if not (b is snaps[-1] and direct_wanted):
            b.drop_members(keep_prog=True)        # its pool has served; as the next progenitor only its followed voids
    direct = None
    if direct_wanted:
        print(f"  direct link snap {ref.snap} -> {snaps[-1].snap}", flush=True)
        lk = L.cached_link(ref, snaps[-1], tag)
        direct = (ref, snaps[-1], lk, L.classify(lk))
    for s in snaps:
        s.drop_members()
    return intervals, direct


def write_table(intervals, ztab, path, exclusive=False):
    cols = ("snap_a", "z_a", "snap_b", "z_b", "row", "x_mpc", "y_mpc", "z_mpc", "r_v_mpc", "delta_c", "n_members",
            "enclosed_delta", "fate", "desc_row", "desc_in_sample", "desc_n_main", "f_desc", "f_max", "f_second",
            "f_sample", "n_split")
    with open(path, "w") as f:
        f.write(f"# void fate per interval (plot_void_fate.py, {time.strftime('%Y-%m-%d')}); SURVIVE_FRAC "
                f"{L.SURVIVE_FRAC}, SPLIT_FRAC {L.SPLIT_FRAC}; members = particles within R_v of the centre"
                f"{' (exclusive: each to its nearest centre in R_v units)' if exclusive else ''}; "
                f"row / desc_row = pipeline catalogue rows (desc_row: any minimum of the next catalogue with a usable "
                f"radius; desc_in_sample 1 = a followed void, 0 = not, -1 = no descendant); desc_n_main = progenitors "
                f"passing >= SURVIVE_FRAC to it (>= 2: a merge); f_desc = its share, f_max / f_second = the two largest "
                f"over every void, f_sample = the largest in a followed void; enclosed_delta = members / mean - 1\n")
        f.write(",".join(cols) + "\n")
        for a, b, lk, fate in intervals:
            vol = 4.0 / 3.0 * np.pi * a.radii ** 3
            for k, row in enumerate(a.rows):
                pos = (a.cat["coords"][row] + 0.5) * a.cell
                nm = int(lk.n_prog_members[k])
                d = fate["desc"][k]
                enc = nm / (a.n_particles * vol[k]) - 1.0 if a.n_particles else np.nan
                vals = (a.snap, f"{ztab.get(a.snap, np.nan):.4f}", b.snap, f"{ztab.get(b.snap, np.nan):.4f}", int(row),
                        *(f"{v:.4f}" for v in pos), f"{a.radii[k] * a.box:.4f}",
                        f"{float(a.cat['delta_values'][row]):.4f}", nm, f"{enc:.4f}", L.CLASSES[fate["cls"][k]],
                        int(b.pool_rows[d]) if d >= 0 else -1, int(fate["in_sample"][k]) if d >= 0 else -1,
                        int(fate["n_main"][k]), f"{fate['f_desc'][k]:.4f}", f"{fate['f_max'][k]:.4f}",
                        f"{fate['f_second'][k]:.4f}", f"{fate['f_sample'][k]:.4f}", int(fate["n_split"][k]))
                f.write(",".join(str(v) for v in vals) + "\n")


def plot_census(intervals, direct, chain, ztab, args, path):
    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    ax = axes[0, 0]
    labels = [f"{zlab(ztab.get(a.snap, np.nan))}$\\to${zlab(ztab.get(b.snap, np.nan))}" for a, b, _l, _f in intervals]
    bottom = np.zeros(len(intervals))
    for c in PLOT_ORDER:
        h = np.array([(f["cls"] == L.CLASS_CODE[c]).sum() for _a, _b, _l, f in intervals], dtype=float)
        ax.bar(np.arange(len(intervals)), h, bottom=bottom, color=CLASS_COLORS[c], label=CLASS_LABELS.get(c, c),
               width=0.75)
        bottom += h
    ax.set_xticks(np.arange(len(intervals)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_xlabel("interval (redshift)")
    ax.set_ylabel("distinct resolved voids (progenitors)")
    # above the bars: 'best' put it over the middle intervals (TNG100 --exclusive, 2026-10-11), the late ones are short
    ax.set_ylim(0, bottom.max() * 1.45 if bottom.size and bottom.max() > 0 else 1.0)
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    ax = axes[0, 1]
    zs = np.array([ztab.get(s, np.nan) for s in chain["snaps"]])
    ax.plot(np.log10(1 + zs), chain["frac"], "o-", color="tab:green",
            label="followed main descendant to the last snapshot")
    for x, y, nn in zip(np.log10(1 + zs), chain["frac"], chain["n"]):
        ax.annotate(f"{nn}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    if direct is not None:
        a, b, _lk, f = direct
        frac_d = float(np.isin(f["cls"], (L.CLASS_CODE["survive"], L.CLASS_CODE["merge"])).mean()) \
            if f["cls"].size else np.nan
        ax.plot(np.log10(1 + ztab.get(a.snap, np.nan)), frac_d, "s", mfc="none", color="tab:blue", ms=9,
                label=f"direct link z={zlab(ztab.get(a.snap, np.nan))} $\\to$ {zlab(ztab.get(b.snap, np.nan))}")
    ax.set_xlabel(r"$\log_{10}(1+z)$ of the progenitor snapshot")
    ax.set_ylabel("fraction surviving (survive or merge)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)

    sig = pipeline.smoothing_length_mpc()
    rv = np.concatenate([a.radii * a.box / sig for a, _b, _l, _f in intervals])
    dc = np.concatenate([np.asarray(a.cat["delta_values"], float)[a.rows] for a, _b, _l, _f in intervals])
    cls = np.concatenate([f["cls"] for _a, _b, _l, f in intervals])
    for ax, x, edges, xlab in ((axes[1, 0], rv, np.geomspace(max(np.nanmin(rv), 1e-3), np.nanmax(rv) * 1.0001, 8)
                                if rv.size else np.array([0.0, 1.0]), r"progenitor $R_v$ / $\sigma$ (smoothing length)"),
                               (axes[1, 1], dc, np.linspace(np.nanmin(dc), np.nanmax(dc) * (1 - 1e-9) + 1e-9, 8)
                                if dc.size else np.array([0.0, 1.0]), r"progenitor central $\delta$ (smoothed)")):
        frac, n = class_fraction_in_bins(x, cls, edges)
        mid = np.sqrt(edges[1:] * edges[:-1]) if ax is axes[1, 0] else 0.5 * (edges[1:] + edges[:-1])
        bottom = np.zeros(mid.size)
        width = np.diff(edges) * 0.9
        for c in PLOT_ORDER:
            i = L.CLASS_CODE[c]
            ax.bar(edges[:-1] + 0.05 * np.diff(edges), frac[i], width=width, bottom=bottom, align="edge",
                   color=CLASS_COLORS[c], label=c)
            bottom += frac[i]
        for m, nn in zip(mid, n):
            if nn:
                ax.annotate(f"{int(nn)}", (m, 1.0), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8)
        if ax is axes[1, 0]:
            ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel("class fraction (all intervals)")
        ax.set_ylim(0, 1.08)
    style.set_suptitle(fig, f"{args.sim}: fate of the distinct resolved voids (Lagrangian particle links"
                            f"{', exclusive members' if getattr(args, 'exclusive', False) else ''})")
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, path, mirror=MIRROR)
    plt.close(fig)


def chained_survival(intervals):
    """For each snapshot's voids: the fraction whose main-descendant chain (survive or merge into a FOLLOWED void
    each step: lagrangian.chain_survival) reaches the last snapshot. {'snaps', 'frac', 'n'}."""
    snaps, frac, n = [], [], []
    for i in range(len(intervals)):
        fates = [f for _a, _b, _l, f in intervals[i:]]
        alive = L.chain_survival(fates)
        snaps.append(intervals[i][0].snap)
        frac.append(float(alive.mean()) if alive.size else np.nan)
        n.append(int(alive.size))
    return {"snaps": np.array(snaps), "frac": np.array(frac), "n": np.array(n)}


def main(argv=None):
    global MIRROR
    parser = make_parser("Fate census of the voids: link every snapshot's distinct resolved voids to every void of "
                         "the next snapshot by their member particles (survive, merge, unresolved, split, vanish).")
    parser.add_argument("--snaps", type=int, nargs="+", default=None,
                        help="snapshots to link, in time order (default: every snapshot of the ladder with a "
                             "combined snapshot file and field grids)")
    parser.add_argument("--ref-snap", type=int, default=REF_SNAP,
                        help=f"also link this snapshot directly to the last one (default {REF_SNAP}, z = 2)")
    parser.add_argument("--exclusive", action="store_true",
                        help="give a particle inside several voids' spheres to the one whose centre is nearest in "
                             "units of its R_v (a partition of the matter: overlapping high-z voids then cannot both "
                             "pass half their matter to one descendant); files tagged '_exclusive'")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the figure, table and npz into this folder instead (nothing goes into the "
                             "thesis tree)")
    pipeline.add_smoothing_mpc_arg(parser)
    args = parser.parse_args(argv)
    pipeline.apply_smoothing_args(args, "void-fate")
    MIRROR = args.out is None
    use_data_root(args.data_root)
    ztab = gc.redshift_table(args.sim)
    ladder = args.snaps or sorted({int(d.name.split("_")[1]) for d in env.sim_dir(args.sim).glob("snapdir_*")})

    snaps = []
    for s in sorted(ladder):
        try:
            vs = L.VoidSnapshot(args.sim, s, args.method, args.prefix, args.data_root, exclusive=args.exclusive)
            style.note_source(vs.fs)
            snaps.append(vs)
        except FileNotFoundError as e:
            print(f"  snap {s:3d}: left out ({e})")
        except ValueError as e:                  # the frame guard: DTFE_SIM is not --sim, or a bad grid
            print(f"  snap {s:3d}: REFUSED: {e}")
    if len(snaps) < 2:
        raise SystemExit(f"fewer than two snapshots of {args.sim} with a combined file and void catalogue: "
                         "nothing to link")
    tag = name_tag(snaps[-1].method, snaps[-1].prefix)
    print(f"{args.sim}: linking {len(snaps)} snapshots {[s.snap for s in snaps]} "
          f"(followed voids per snapshot {[int(s.rows.size) for s in snaps]}, descendant pool "
          f"{[int(s.pool_rows.size) for s in snaps]})", flush=True)

    intervals, direct = census(snaps, args, tag)
    chain = chained_survival(intervals)
    for a, b, _lk, fate in intervals:
        cc = class_counts(fate)
        print(f"  z {ztab.get(a.snap, np.nan):5.2f} -> {ztab.get(b.snap, np.nan):5.2f} (snap {a.snap:3d} -> {b.snap:3d}): "
              f"{a.rows.size:4d} voids: " + "  ".join(f"{c} {cc[c]}" for c in L.CLASSES)
              + _vanish_note(fate))
    for s, fr, nn in zip(chain["snaps"], chain["frac"], chain["n"]):
        print(f"  voids of snap {s:3d} (z={ztab.get(s, np.nan):5.2f}) with a followed main-descendant chain to snap "
              f"{snaps[-1].snap}: {fr:.3f} of {nn}")
    if direct is not None:
        cc = class_counts(direct[3])
        print(f"  direct link snap {direct[0].snap} -> {direct[1].snap}: " + "  ".join(f"{c} {cc[c]}" for c in L.CLASSES))

    tag_out = tag + ("_exclusive" if args.exclusive else "")
    outdir = args.out or (FIGURE_ROOT / args.sim / pipeline.smoothing_tag())
    outdir.mkdir(parents=True, exist_ok=True)
    plot_census(intervals, direct, chain, ztab, args, outdir / f"fate_census{tag_out}.png")
    write_table(intervals, ztab, outdir / f"fate_table{tag_out}.csv", args.exclusive)
    counts = np.array([[class_counts(f)[c] for c in L.CLASSES] for _a, _b, _l, f in intervals])
    np.savez_compressed(
        outdir / f"fate_census{tag_out}.npz", classes=np.array(L.CLASSES),
        snap_a=np.array([a.snap for a, *_ in intervals]), snap_b=np.array([b.snap for _a, b, *_ in intervals]),
        z_a=np.array([ztab.get(a.snap, np.nan) for a, *_ in intervals]),
        z_b=np.array([ztab.get(b.snap, np.nan) for _a, b, *_ in intervals]),
        counts=counts, chain_snaps=chain["snaps"], chain_frac=chain["frac"], chain_n=chain["n"],
        survive_frac=L.SURVIVE_FRAC, split_frac=L.SPLIT_FRAC, sim=args.sim, exclusive=args.exclusive,
        link_version=L.LINK_VERSION,
        **({} if direct is None else {"direct_snaps": np.array([direct[0].snap, direct[1].snap]),
                                      "direct_counts": np.array([class_counts(direct[3])[c] for c in L.CLASSES])}))
    print(f"-> {outdir}/fate_census{tag_out}.png, fate_table{tag_out}.csv, fate_census{tag_out}.npz  "
          f"(peak memory {peak_rss_gb():.1f} GB)")


if __name__ == "__main__":
    main()
