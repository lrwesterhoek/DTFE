"""Stream multiplicity (and caustic class, when the run wrote one) by cosmic-web environment.

For every snapshot: the table of dtfelib.webstreams for the T-web and the V-web of a PS-DTFE output
-- per class (void, wall, filament, node) its share of the box in volume and mass, and within the
class the stream-multiplicity bins in both weightings -- as JSON + text, and a stacked-bar figure;
with several snapshots, the single-stream fraction of each class against redshift.

NOT in analyze.py's PLOT_SCRIPTS: that series is the standard-DTFE core set, this needs the
phase-space grids (streams, hidden_streams; caustic_class is optional).

Names: web_streams_<web>_z<z>.* and single_stream_vs_z_<web>.png for the default grids (ps_output.,
averaged) over every snapshot of the simulation on disk; another --prefix or --raw adds '_<prefix>' / '_raw', and a
series over fewer snapshots '_snaps<a>-<b>-...', so none of them overwrites the default files.

    python3 plot/plot_web_streams.py --sim TNG100-3-Dark --snaps 99 67 50 40
    python3 plot/plot_web_streams.py --sim TNG100-3-Dark --snap 99 --web tweb --raw

THE V-WEB is the MEAN-FIELD one by default: classified from the FFT gradient of the PS velocity grid
with the binary's normalisation and the run's lambda_th (FieldSet.load('vweb', mean_field=True)), the V-web standard
DTFE stores and the literature means. The binary's own PS V-web classifies the stream average of each stream's
gradient <grad v>, in which the inverted stream inside a fold has a POSITIVE divergence, so fold cells read as
'void' (dtfelib/io.py, the note above FIELDS) -- the very cells this table counts as multi-stream. '--ps-gradient
stored' tables it anyway, under the name tag '_stored' (web_streams_vweb_stored_z<z>.*, single_stream_vs_z_vweb_stored
...), so the two never overwrite each other; the table's 'vweb_source' says which it holds. The T-web is the
binary's either way.
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import config
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dtfelib import make_parser, FieldSet, sim_dir, add_ps_gradient_arg, mean_field_wanted
from dtfelib import figures as style
from dtfelib import webstreams as ws

OUTPUT_DIR = Path(config.LOCAL_FIGURES_ROOT) / "web_streams"
WEBS = {"tweb": ("T-web", "tweb"), "vweb": ("V-web", "vweb")}
BIN_COLORS = ["#bbbbbb", "#1f5fa8", "#7fb2e5", "#f2b134", "#e4572e", "#7b1e3b"]


def plot_table(table, title, path):
    """Stacked bars of the stream bins per class: volume-weighted left, mass-weighted right (when
    the density was available), with the class's share of the box under its name."""
    names = [n for n in table["classes"] if n != "other"]
    panels = ["volume"] + (["mass"] if table["mass_weighted"] else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), squeeze=False)
    for ax, w in zip(axes[0], panels):
        bottom = np.zeros(len(names))
        for j, b in enumerate(ws.STREAM_BINS):
            vals = np.array([(table["classes"][n]["stream_bins"][b][w] or 0.0) for n in names])
            ax.bar(names, vals, bottom=bottom, color=BIN_COLORS[j], label=b, width=0.7)
            bottom += vals
        share = "volume_fraction" if w == "volume" else "mass_fraction"
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{n}\n{100 * (table['classes'][n][share] or 0):.1f}%" for n in names], fontsize=8)
        ax.set_xlabel(f"class (share of the {w} below)", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"fraction of the class ({w}-weighted)")
        style.set_title(ax, f"{w}-weighted")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="multiplicity s")
    style.set_suptitle(fig, title)
    fig.tight_layout()
    style.save_figure(fig, path)
    plt.close(fig)


def plot_vs_redshift(rows, web_name, path):
    """The single-stream fraction of each class against redshift (volume-weighted, solid; mass-weighted, dashed)."""
    rows = sorted(rows, key=lambda r: -r["z"])
    zs = [r["z"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    colors = {"void": "#1a1a2e", "wall": "#b8a24a", "filament": "#d4563e", "node": "#7a7a7a"}
    for name in ("void", "wall", "filament", "node"):
        v = [ws.single_stream_fraction(r["table"], "volume").get(name, np.nan) for r in rows]
        ax.plot(zs, v, "-o", color=colors[name], label=name)
        if all(r["table"]["mass_weighted"] for r in rows):
            m = [ws.single_stream_fraction(r["table"], "mass").get(name, np.nan) for r in rows]
            ax.plot(zs, m, "--", color=colors[name], alpha=0.7)
    ax.invert_xaxis()
    ax.set_xlabel("redshift z")
    ax.set_ylabel("single-stream fraction of the class")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(title="solid: volume, dashed: mass", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    style.set_title(ax, f"{web_name}: single-stream fraction by environment")
    fig.tight_layout()
    style.save_figure(fig, path)
    plt.close(fig)


def grid_tag(fs):
    """'' for the default grids (ps_output., volume-averaged); else '_<prefix>' and/or '_raw'."""
    prefix = fs.prefix.rstrip(".")
    return ("" if prefix == "ps_output" else f"_{prefix}") + ("" if fs.averaged else "_raw")


def web_tag(key, mean_field=True):
    """'_stored' for the V-web of the stored PS gradient grids ('--ps-gradient stored'), else ''."""
    return "_stored" if key == "vweb" and not mean_field else ""


def process(fs, sim, snap, webs, series, problems, out_dir, mean_field=True):
    """The tables and figures of one snapshot: each web's table goes into series[web] AS SOON AS it is made
    (a later web's failure cannot lose it), a missing grid or a failure is noted in 'problems' as
    (snapshot, web label, what); returns the number of tables written. mean_field: the V-web is the mean-field
    one (module docstring); False = the stored PS grid."""
    z = fs.meta.redshift
    done = 0
    if not fs.has("streams"):
        print(f"  snapshot {snap:03d}: no streams grid (method '{fs.method}', prefix {fs.prefix}); skipped")
        problems.append((snap, "streams", "no streams grid"))
        return done
    streams = fs.load("streams", mode="memmap")
    hidden = fs.load("hidden_streams", mode="memmap") if fs.has("hidden_streams") else None
    density = fs.load("density", mode="memmap") if fs.has("density") else None
    caustic = fs.load("caustic_class", mode="memmap") if fs.has("caustic_class") else None
    print(f"  snapshot {snap:03d} (z={z:.2f}): {fs.grid_n}^3 cells, hidden streams {'yes' if hidden is not None else 'no'}, "
          f"caustic class {'yes' if caustic is not None else 'no'}")
    for key in webs:
        label, field = WEBS[key]
        mf = mean_field and key == "vweb"                    # the V-web of the mean flow, derived from the velocity grid
        if not fs.has(field, mean_field=mf):
            print(f"    {label}: no '{'velocity' if mf else field}' grid"
                  f"{' to derive the mean-field V-web from' if mf else ''}; skipped")
            problems.append((snap, label, "no grid"))
            continue
        try:
            web = fs.load(field, mode="memmap", mean_field=mf)
            table = ws.environment_table(web, streams, hidden=hidden, density=density, caustic_class=caustic)
            table["sim"], table["snapshot"], table["redshift"], table["web"] = sim, int(snap), float(z), label
            table["prefix"], table["averaged"] = fs.prefix, fs.averaged
            if key == "vweb":
                table["vweb_source"] = "mean-field" if mf else "stored"
            title = f"{sim}  z = {z:.2f}  {label}{' (stored PS gradient)' if web_tag(key, mean_field) else ''}  " \
                    f"({'averaged' if fs.averaged else 'raw'} streams, prefix {fs.prefix.rstrip('.')})"
            base = out_dir / f"web_streams_{key}{grid_tag(fs)}{web_tag(key, mean_field)}_z{z:.2f}"     # '.png' etc. APPENDED: with_suffix would eat '.50'
            ws.save_table(table, base, title)
            plot_table(table, title, Path(f"{base}.png"))
        except (OSError, ValueError, MemoryError) as e:      # a truncated or unreadable grid (OSError incl. missing): this web only
            print(f"    {label}: FAILED: {e}")
            problems.append((snap, label, f"failed: {e}"))
            continue
        print(ws.table_text(table, title))
        series[key].append({"z": float(z), "table": table})
        done += 1
    return done


def main():
    parser = make_parser("Stream multiplicity and caustic class by T-web / V-web environment (PS-DTFE grids).")
    parser.add_argument("--snaps", type=int, nargs="*", default=None, help="snapshots (default: --snap)")
    parser.add_argument("--web", choices=("both", "tweb", "vweb"), default="both")
    parser.add_argument("--out", default=None, help=f"output folder (default {OUTPUT_DIR}/<sim>)")
    add_ps_gradient_arg(parser)
    args = parser.parse_args()
    style.apply()
    out_dir = Path(args.out).expanduser() if args.out else OUTPUT_DIR / args.sim
    snaps = args.snaps if args.snaps else [args.snap]
    webs = ["tweb", "vweb"] if args.web == "both" else [args.web]
    method = "ps" if args.method == "auto" else args.method
    series = {k: [] for k in webs}
    failed, problems = [], []                   # failed: no FieldSet / no streams at all; problems: per (snapshot, web)
    tag = None                                  # the grids' name tag, from the first FieldSet (one --prefix / --raw per run)
    for snap in snaps:
        try:
            fs = FieldSet(sim_dir(args.sim, args.data_root) / f"snapdir_{snap:03d}", method=method,
                          averaged=not args.raw, prefix=args.prefix)
            tag = grid_tag(fs) if tag is None else tag
            n_done = process(fs, args.sim, snap, webs, series, problems, out_dir, mean_field=mean_field_wanted(args))
        except (FileNotFoundError, ValueError, OSError) as e:    # OSError: h5py on an unreadable snapshot file
            print(f"  snapshot {snap:03d}: {e}")
            failed.append(snap)
            continue
        if not n_done:
            failed.append(snap)
    # the series' default name holds the whole ladder on disk; a subset of it gets its own name, so a
    # '--snaps 99 67 50 40' run never replaces the full series (a snapshot that failed still counts as asked)
    ladder = sorted({int(d.name.split("_")[1]) for d in sim_dir(args.sim, args.data_root).glob("snapdir_*")
                     if d.name.split("_")[1].isdigit()})
    asked = sorted(set(snaps))
    subset = "" if asked == ladder else "_snaps" + "-".join(str(s) for s in asked)
    for k, rows in series.items():
        if len(rows) > 1:
            plot_vs_redshift(rows, WEBS[k][0], out_dir / f"single_stream_vs_z_{k}{tag or ''}{web_tag(k, mean_field_wanted(args))}{subset}.png")
    done = sum(len(r) for r in series.values())
    notes = [f"{lab} {what} for {s:03d}" for s, lab, what in problems]
    print(f"\n{done} table(s) written under {out_dir}"
          + (f"; no table for snapshot(s) {failed}" if failed else "") + ("; " + "; ".join(notes) if notes else ""))
    # a missing grid under the default '--web both' is a skip (the run scripts' rule); a failure, a requested
    # web that no snapshot had, a snapshot without a table, or nothing at all is an error
    errors = failed or not done or any(what.startswith("failed") for _s, _l, what in problems) \
        or (args.web != "both" and any(what == "no grid" for _s, _l, what in problems))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
