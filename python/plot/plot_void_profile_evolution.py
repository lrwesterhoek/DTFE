"""The void profiles across cosmic time on ONE figure, and the universal-profile fits per R_v bin and redshift.

For every snapshot on disk: the shells of the DISTINCT well-resolved voids (dtfelib.profiles, as
plot_void_profiles.py), and from them
  (a) the stacked RAW rho/rho_bar against r / R_v of the voids with R_v in --rv-range (in smoothing lengths, default
      2.5-5 sigma: the same voids in sigma units at every z), one curve per redshift, colour = z;
  (b) their stacked radial velocity in units of the Hubble flow at the void's own radius, v_r / (a H(z) R_v), per
      void before stacking (nine redshifts in km/s do not share an axis), with the linear-theory prediction dashed:
      v_lin(r) = -(1/3) f(z) H(z) a r Delta(<r), f = Omega_m(z)^0.55, Delta(<r) from the stacked density
      (profiles.linear_v_r), which in these units is -(1/3) f x Delta(<x) whatever R_v is;
  (c) the interior single-stream VOLUME fraction, r < 0.5 R_v and r < R_v, against 1 + z (profiles.
      interior_single_fraction: the cells whose centres lie inside, single stream = streams == 1 and no
      hidden-streams bit, io.single_stream_mask), mean over the voids with a 16-84% bootstrap band, beside the whole
      box's single-stream fraction (phase-space grids only);
  (d) v_r / v_lin at r = 0.5 and 1 R_v against 1 + z: 1 is linear expansion.
A 16-84% bootstrap band of the voids goes with every curve. 1 + z is on a log axis: 6 of the 9 TNG snapshots sit at
z < 3. The density crossing 1 at r = R_v is the definition of R_v (the first up-crossing of the SMOOTHED profile), a
check of the smoothing against the raw density, not a finding; the infall radius, the ridge and the interior
single-stream fraction are.

THE FITS. Hamaus, Sutter & Wandelt (2014, PRL 112, 251302): delta(r) = delta_c (1 - (r/r_s)^alpha) / (1 + (r/R_v)^beta)
fitted (profiles.fit_stack: scipy curve_fit, the error of each shell the stack's scatter std / sqrt(n)) to the stacked
density of every R_v bin (--edges in Mpc; default profiles.default_edges_mpc: 0, 2.5, 3.5, 5 sigma and the wrap cap
L/6) at every redshift, with r in units of each void's own R_v. Written as a table (json + txt) and a figure of
delta_c, r_s / R_v, alpha and beta against 1 + z. The shells of one stack share its voids, so chi2/dof is a guide.

CACHE. Each snapshot's per-void profiles go into per_snapshot/profiles_<snap><tag>.npz in the output folder;
--replot reads those and redraws everything (other --rv-range, --edges, --n-boot) without touching a grid.

Outputs in figures/void_profile_evolution/<sim>/ (config.figures_path; a non-default smoothing in its own subfolder):
  profile_evolution.png, profile_evolution.npz     panels (a)-(d) and their curves
  profile_fits.png, profile_fits.json, profile_fits.txt
Standard DTFE or a --prefix adds '_dtfe' / '_<prefix>' to every name, --sample deep '_deep', --keep-overlaps
'_overlaps', an R_v definition other than 'crossing' '_rv<definition>'.

    DTFE_SIM=TNG100-3-Dark python3 plot/plot_void_profile_evolution.py --sim TNG100-3-Dark
    DTFE_SIM=TNG100-3-Dark python3 plot/plot_void_profile_evolution.py --sim TNG100-3-Dark --replot --rv-range 2,4
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dtfelib import make_parser, use_data_root, pipeline, profiles, catalog
from dtfelib import environment as env
from dtfelib import figures as style
from dtfelib import fields as dtfe

FOLDER = "void_profile_evolution"
OUTPUT_DIR = Path(config.figures_path(FOLDER))
MIRROR = True                              # --out: the user's folder, nothing goes into the thesis tree
DEFAULT_RV_RANGE = (2.5, 5.0)              # the overlay's R_v bin in smoothing lengths (the middle of DEFAULT_EDGES_SIGMA)
RATIO_X = (0.5, 1.0)                       # panel (d): v_r / v_lin at these r / R_v
DEFINITIONS = tuple(getattr(catalog, "R_V_DEFINITIONS", ("crossing", "delta08", "ridge")))
_PER_VOID = ("density", "v_r", "single", "single_interior", "r_mpc", "r_v_mpc", "r_eff_mpc", "index", "delta_c")


def grids_tag(method: str, prefix) -> str:
    """'' for PS-DTFE's default grids, else '_dtfe' / '_<prefix>' (plot_void_population's rule)."""
    if prefix:
        return "_" + str(prefix).rstrip(".")
    return "" if method in ("ps", "auto") else f"_{method}"


def options_tag(args) -> str:
    return ("_deep" if args.sample == "deep" else "") + ("_overlaps" if args.keep_overlaps else "") \
        + ("" if args.rv_definition == "crossing" else f"_rv{args.rv_definition}")


def save_snapshot(path: Path, snap: int, z: float, prof: dict, extra: dict) -> None:
    """One snapshot's per-void profiles (what --replot needs), written whole or not at all (temp + rename)."""
    arrays = {k: np.asarray(prof[k]) for k in _PER_VOID if prof.get(k) is not None}
    arrays.update(bins=np.asarray(prof["bins"]), interior_x=np.asarray(prof["interior_x"]), snap=int(snap),
                  redshift=float(z), method=prof["method"], definition=prof["definition"], sample=prof["sample"],
                  distinct=bool(prof["distinct"]), cell_mpc=prof["cell_mpc"], box_mpc=prof["box_mpc"],
                  n_grid=prof["n_grid"], points=prof["m"], dropped=prof["dropped"], n_no_radius=prof["n_no_radius"],
                  n_overlapping=prof["n_overlapping"],
                  single_box=np.nan if prof.get("single_box") is None else prof["single_box"], **extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_snapshot(path: Path) -> dict:
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    for k in ("snap", "n_grid", "points", "dropped"):
        d[k] = int(d[k])
    for k in ("redshift", "cell_mpc", "box_mpc", "single_box", "sigma_mpc"):
        if k in d:
            d[k] = float(d[k])
    for k in ("method", "definition", "sample", "prefix"):
        if k in d:
            d[k] = str(d[k])
    return d


def find_snapshots(folder: Path, tag: str) -> dict:
    """{snap: path} of the per-snapshot caches with exactly this tag (profiles_099.npz is not profiles_099_deep.npz)."""
    rx = re.compile(r"profiles_(\d{3})" + re.escape(tag) + r"\.npz$")
    out = {}
    for p in sorted(folder.glob("profiles_*.npz")) if folder.is_dir() else []:
        m = rx.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def hubble_units(z: float) -> tuple[float, float, float]:
    """(a, H(z) km/s/Mpc, f(z)) of the TNG cosmology (fields.get_cosmology_params: Planck 2015)."""
    c = dtfe.get_cosmology_params(z)
    return c["a"], c["H_z"], c["f_growth"]


def _boot_mean(values: np.ndarray, n_boot: int, rng) -> tuple[float, float, float]:
    """Mean and the 16/84% of 'n_boot' bootstrap means of a 1-D sample (NaN-free); NaN band with fewer than two."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return np.nan, np.nan, np.nan
    if v.size < 2:
        return float(v.mean()), np.nan, np.nan
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 16)), float(np.percentile(means, 84))


def evolution(per: dict, rv_mpc: tuple, n_boot: int, min_count: int) -> dict:
    """Panels (a)-(d) for each snapshot: the stack of the voids with R_v in rv_mpc = (lo, hi) Mpc. {snap: {...}}."""
    out = {}
    for s, d in sorted(per.items()):
        z = d["redshift"]
        a, h, _f = hubble_units(z)
        r = np.asarray(d["r_mpc"], dtype=np.float64)
        prof = {"bins": d["bins"], "r_mpc": r, "density": d["density"]}
        if "v_r" in d:
            prof["v_r"] = d["v_r"] / (a * h * r[:, None])           # per void, in its own Hubble flow at R_v
        if "single" in d:
            prof["single"] = d["single"]
        st = profiles.stack(prof, edges_mpc=rv_mpc, n_boot=n_boot, min_count=min_count)
        e = {"z": z, "count": int(st["count"][0]), "bins": st["bins"], "stack": st}
        if st["count"][0] >= min_count:
            dens = st["density"]["mean"][0]
            e["v_lin"] = profiles.linear_v_r(st["bins"], dens, 1.0, z) / (a * h)   # -(1/3) f x Delta(<x)
            if "v_r" in st:
                e["ratio"] = {}
                for xv in RATIO_X:
                    vl = float(np.interp(xv, st["bins"], e["v_lin"]))
                    e["ratio"][xv] = tuple(float(np.interp(xv, st["bins"], st["v_r"][k][0])) / vl
                                           for k in ("mean", "lo", "hi"))
        sel = (r >= rv_mpc[0]) & (r < rv_mpc[1])
        if "single_interior" in d and sel.sum() >= min_count:
            rng = np.random.default_rng(0)
            e["interior"] = {float(xv): _boot_mean(d["single_interior"][sel, j], n_boot, rng)
                             for j, xv in enumerate(d["interior_x"])}
        e["single_box"] = d.get("single_box", np.nan)
        out[s] = e
    return out


def all_fits(per: dict, edges: list, min_count: int) -> list:
    rows = []
    for s, d in sorted(per.items()):
        prof = {"bins": d["bins"], "r_mpc": d["r_mpc"], "density": d["density"]}
        for i, f in enumerate(profiles.fit_stack(prof, edges, min_count=min_count)):
            f.update(snap=int(s), z=float(d["redshift"]), bin=i)
            rows.append(f)
    return rows


def _z_colors(zs):
    """A colour per redshift on log10(1 + z) (most snapshots are at low z) and the mappable for its colour bar."""
    lz = np.log10(1.0 + np.asarray(zs, dtype=np.float64))
    norm = matplotlib.colors.Normalize(lz.min(), lz.max() if lz.max() > lz.min() else lz.min() + 1.0)
    cmap = plt.get_cmap("plasma_r")
    return [cmap(norm(v)) for v in lz], matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)


def plot_evolution(ev: dict, title: str, path: Path, rv_label: str, min_count: int) -> None:
    snaps = [s for s in sorted(ev, key=lambda s: -ev[s]["z"])]
    zs = [ev[s]["z"] for s in snaps]
    colors, sm = _z_colors(zs)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    (ax_d, ax_v), (ax_s, ax_r) = axes
    drawn = 0
    for c, s in zip(colors, snaps):
        e = ev[s]
        if e["count"] < min_count:
            continue
        drawn += 1
        st, x = e["stack"], e["bins"]
        lab = f"z = {e['z']:.2f} (n = {e['count']})"
        ax_d.plot(x, st["density"]["mean"][0], color=c, lw=1.4, label=lab)
        ax_d.fill_between(x, st["density"]["lo"][0], st["density"]["hi"][0], color=c, alpha=0.18, lw=0)
        if "v_r" in st:
            ax_v.plot(x, st["v_r"]["mean"][0], color=c, lw=1.4)
            ax_v.fill_between(x, st["v_r"]["lo"][0], st["v_r"]["hi"][0], color=c, alpha=0.18, lw=0)
            ax_v.plot(x, e["v_lin"], "--", color=c, lw=1.0)
    for ax, ylab, ref in ((ax_d, r"$\rho / \bar\rho$", 1.0),
                          (ax_v, r"$v_r \,/\, (a H R_v)$", 0.0)):
        ax.axhline(ref, color="gray", lw=0.7, ls=":")
        ax.axvline(1.0, color="gray", lw=0.7, ls=":")
        ax.set_xlabel(r"$r / R_v$")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    ax_v.text(0.03, 0.03, r"positive = outflow; dashed: linear theory $-\frac{1}{3} f\, x\, \Delta(<x)$ from the stacked density",
              transform=ax_v.transAxes, fontsize=7, color="0.3")
    if drawn:
        ax_d.legend(fontsize=7, title=rv_label, title_fontsize=7, loc="upper left")
    else:
        ax_d.text(0.5, 0.5, f"no snapshot with {min_count} voids in {rv_label}", transform=ax_d.transAxes,
                  ha="center", color="0.4")
    # (c) the interior single-stream fraction against 1 + z
    for xv, mk, c in ((0.5, "o", "tab:blue"), (1.0, "s", "tab:orange")):
        pts = [(1 + ev[s]["z"],) + ev[s]["interior"][xv] for s in snaps if xv in ev[s].get("interior", {})]
        if pts:
            p = np.array(pts)
            ax_s.plot(p[:, 0], p[:, 1], mk + "-", color=c, ms=4, label=f"$r < {xv:g}\\,R_v$")
            ax_s.fill_between(p[:, 0], p[:, 2], p[:, 3], color=c, alpha=0.2, lw=0)
    box = [(1 + ev[s]["z"], ev[s]["single_box"]) for s in snaps if np.isfinite(ev[s]["single_box"])]
    if box:
        b = np.array(box)
        ax_s.plot(b[:, 0], b[:, 1], "k--", lw=1.0, label="whole box")
        ax_s.legend(fontsize=8)
    else:
        ax_s.text(0.5, 0.5, "no stream-count grid (phase-space DTFE only)", transform=ax_s.transAxes, ha="center",
                  color="0.4", fontsize=8)
    ax_s.set_ylabel("single-stream volume fraction")
    ax_s.set_ylim(-0.02, 1.02)
    # (d) v_r / v_lin at fixed r / R_v
    for xv, mk, c in zip(RATIO_X, ("o", "s"), ("tab:green", "tab:red")):
        pts = [(1 + ev[s]["z"],) + ev[s]["ratio"][xv] for s in snaps if xv in ev[s].get("ratio", {})]
        if pts:
            p = np.array(pts)
            ax_r.plot(p[:, 0], p[:, 1], mk + "-", color=c, ms=4, label=f"$r = {xv:g}\\,R_v$")
            ax_r.fill_between(p[:, 0], p[:, 2], p[:, 3], color=c, alpha=0.2, lw=0)
            ax_r.legend(fontsize=8)
    ax_r.axhline(1.0, color="gray", lw=0.7, ls=":")
    ax_r.set_ylabel(r"$v_r / v_{\rm lin}$")
    for ax in (ax_s, ax_r):
        ax.set_xscale("log")
        ax.set_xlabel("1 + z")
        ax.grid(alpha=0.3, which="both")
    cb = fig.colorbar(sm, ax=[ax_d, ax_v], pad=0.01, fraction=0.03)
    cb.set_ticks(np.log10(1.0 + np.asarray(sorted(set(zs)))))
    cb.set_ticklabels([f"{z:.2g}" if z >= 0.005 else "0" for z in sorted(set(zs))])   # z = 0 is 2e-16 in the header
    cb.set_label("z")
    style.set_suptitle(fig, title)
    style.save_plot_to_multiple_paths(fig, path, dpi=config.DPI, mirror=MIRROR, bbox_inches="tight")
    plt.close(fig)


def plot_fits(rows: list, edges: list, sigma_mpc: float, title: str, path: Path) -> None:
    labels = {"delta_c": r"$\delta_c$", "r_s": r"$r_s / R_v$", "alpha": r"$\alpha$", "beta": r"$\beta$"}
    nbin = len(edges) - 1
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(nbin, 1)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    any_ok = False
    for i in range(nbin):
        sel = sorted((r for r in rows if r["bin"] == i and r["constrained"]), key=lambda r: r["z"])
        if not sel:
            continue
        any_ok = True
        zp1 = np.array([1 + r["z"] for r in sel])
        lab = f"$R_v$ {edges[i] / sigma_mpc:.3g}-{edges[i + 1] / sigma_mpc:.3g}$\\sigma$ ({edges[i]:.3g}-{edges[i + 1]:.3g} Mpc)"
        for ax, k in zip(axes.flat, profiles.HSW_PARAMS):
            ax.errorbar(zp1 * (1 + 0.012 * (i - nbin / 2)), [r["params"][k] for r in sel],
                        yerr=[r["errors"][k] for r in sel], fmt="o-", ms=4, lw=1.2, capsize=2, color=colors[i],
                        label=lab)
    for ax, k in zip(axes.flat, profiles.HSW_PARAMS):
        ax.set_ylabel(labels[k])
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
    for ax in axes[1]:
        ax.set_xlabel("1 + z")
    if any_ok:
        axes[0, 0].legend(fontsize=7)
    else:
        axes[0, 0].text(0.5, 0.5, "no constrained fit", transform=axes[0, 0].transAxes, ha="center", color="0.4")
    n_left = sum(r["ok"] and not r["constrained"] for r in rows)
    if n_left:                  # a parameter on its bound or an error above its value: in the table, not drawn
        axes[1, 1].text(0.02, 0.02, f"{n_left} fit(s) with a parameter at its bound or unconstrained: table only",
                        transform=axes[1, 1].transAxes, fontsize=7, color="0.3")
    style.set_suptitle(fig, title)
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, path, dpi=config.DPI, mirror=MIRROR)
    plt.close(fig)


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def write_tables(base: Path, meta: dict, rows: list, ev: dict) -> None:
    """profile_fits.json (strict JSON, NaN = null) and profile_fits.txt (one row per snapshot and R_v bin)."""
    single = [{"snap": int(s), "z": e["z"], "count": e["count"], "single_box": e["single_box"],
               **{f"f_ss_lt_{xv:g}Rv": dict(zip(("mean", "lo", "hi"), v)) for xv, v in e.get("interior", {}).items()},
               **{f"v_over_vlin_at_{xv:g}Rv": dict(zip(("mean", "lo", "hi"), v)) for xv, v in e.get("ratio", {}).items()}}
              for s, e in sorted(ev.items(), key=lambda t: -t[1]["z"])]
    doc = {**meta, "fits": rows, "overlay": single}
    tmp = base.with_name(base.name + ".json.tmp")
    tmp.write_text(json.dumps(_json_safe(doc), indent=1, allow_nan=False))
    tmp.replace(base.with_suffix(".json"))
    lines = [f"# {meta['sim']} {meta['method']}  HSW 2014 fits to the stacked rho/rho_bar of the {meta['what']} voids "
             f"(sigma = {meta['sigma_mpc']:.4g} Mpc, R_v definition {meta['definition']}); errors 1 sigma from "
             "curve_fit with the stack's scatter",
             f"# {'snap':>4} {'z':>6} {'R_v bin [Mpc]':>14} {'n':>4} {'delta_c':>16} {'r_s/R_v':>16} {'alpha':>16} "
             f"{'beta':>16} {'chi2/dof':>8}"]
    for r in sorted(rows, key=lambda r: (-r["z"], r["bin"])):
        def pe(k):
            return f"{r['params'][k]:7.3f} +- {r['errors'][k]:5.3f}" if r["ok"] else f"{'-':>16}"
        flag = "" if r["constrained"] or not r["ok"] else \
            ("  at bound: " + ",".join(r["at_bound"]) if r["at_bound"] else "  unconstrained")
        lines.append(f"  {r['snap']:4d} {r['z']:6.2f} {r['edges'][0]:6.2f}-{r['edges'][1]:6.2f} {r['count']:5d} "
                     + " ".join(pe(k) for k in profiles.HSW_PARAMS) + f" {r['chi2_dof']:8.2f}{flag}")
    lines.append("")
    lines.append(f"# interior single-stream volume fraction and v_r / v_lin of the overlay sample ({meta['rv_text']})")
    lines.append(f"# {'snap':>4} {'z':>6} {'n':>4} {'f_ss(<0.5Rv)':>22} {'f_ss(<Rv)':>22} {'box':>6} "
                 f"{'v/vlin(0.5Rv)':>14} {'v/vlin(Rv)':>11}")
    for s in single:
        def fr(key):
            v = s.get(key)
            return f"{v['mean']:.3f} [{v['lo']:.3f},{v['hi']:.3f}]" if v and np.isfinite(v["mean"]) else f"{'-':>22}"
        def rt(key):
            v = s.get(key)
            return f"{v['mean']:.3f}" if v and np.isfinite(v["mean"]) else "-"
        lines.append(f"  {s['snap']:4d} {s['z']:6.2f} {s['count']:4d} {fr('f_ss_lt_0.5Rv'):>22} {fr('f_ss_lt_1Rv'):>22} "
                     f"{s['single_box']:6.3f} {rt('v_over_vlin_at_0.5Rv'):>14} {rt('v_over_vlin_at_1Rv'):>11}")
    base.with_suffix(".txt").write_text("\n".join(lines) + "\n")


def _floats(text: str, what: str) -> list:
    try:
        vals = [float(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise SystemExit(f"{what} must be comma-separated numbers, got {text!r}")
    if len(vals) < 2 or any(b <= a for a, b in zip(vals, vals[1:])):
        raise SystemExit(f"{what} must be at least two increasing numbers, got {text!r}")
    return vals


def main(argv=None):
    parser = make_parser("Void profiles across cosmic time on one figure (density, v_r against linear theory, "
                         "interior single-stream fraction) and Hamaus-Sutter-Wandelt fits per R_v bin and redshift.")
    parser.add_argument("--snaps", type=int, nargs="*", default=None, help="snapshots (default: every snapdir on disk)")
    parser.add_argument("--sample", choices=("resolved", "deep"), default="resolved",
                        help="well-resolved voids, or those also deeper than config.DEEP_VOID_THRESHOLD")
    parser.add_argument("--keep-overlaps", action="store_true",
                        help="every resolved void, not only the distinct ones")
    parser.add_argument("--rv-definition", choices=DEFINITIONS, default="crossing",
                        help="the void radius: crossing (first up-crossing of the mean, default), delta08 (enclosed "
                             "contrast reaches -0.8) or ridge (the compensation ridge); one the catalogue lacks falls "
                             "back to crossing")
    parser.add_argument("--rv-range", default=",".join(f"{v:g}" for v in DEFAULT_RV_RANGE),
                        help="the overlay's R_v bin in smoothing lengths, lo,hi (default 2.5,5)")
    parser.add_argument("--edges", default=None,
                        help="R_v bin edges of the fits in Mpc, comma-separated (default 0, 2.5, 3.5, 5 sigma and L/6)")
    parser.add_argument("--points", type=int, default=profiles.DEFAULT_M, help="points per shell")
    parser.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples for the bands")
    parser.add_argument("--min-count", type=int, default=5, help="fewest voids a stack or a fit needs")
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the per-snapshot caches of an earlier run, no grid read")
    parser.add_argument("--out", default=None, help=f"output folder (default {OUTPUT_DIR}/<sim>)")
    pipeline.add_smoothing_mpc_arg(parser)
    args = parser.parse_args(argv)
    global MIRROR
    MIRROR = args.out is None
    style.apply()
    pipeline.apply_smoothing_args(args, "void-profile-evolution")
    use_data_root(args.data_root)
    sigma = pipeline.smoothing_length_mpc()
    rv_sig = _floats(args.rv_range, "--rv-range")[:2]
    rv_mpc = (rv_sig[0] * sigma, rv_sig[1] * sigma)
    edges = _floats(args.edges, "--edges") if args.edges else \
        list(profiles.default_edges_mpc(sigma, config.BOX_SIZE, "r_v"))
    out_dir = Path(args.out).expanduser() if args.out else OUTPUT_DIR / args.sim / pipeline.smoothing_tag()
    cache_dir = out_dir / "per_snapshot"
    tag = grids_tag(args.method, args.prefix) + options_tag(args)

    failed = []
    if not args.replot:
        ladder = sorted({int(d.name.split("_")[1]) for d in env.sim_dir(args.sim).glob("snapdir_*")
                         if d.name.split("_")[1].isdigit()})
        snaps = args.snaps if args.snaps else ladder
        if not snaps:
            raise SystemExit(f"no snapdir_* of {args.sim} under {args.data_root}")
        for snap in snaps:
            p = pipeline.products(snap, sim=args.sim, method=args.method, prefix=args.prefix, data_root=args.data_root)
            try:
                p.data_dir
                prof = profiles.void_profiles(p, m=args.points, sample=args.sample, radius="r_v",
                                              distinct=not args.keep_overlaps, definition=args.rv_definition)
                z = float(p.redshift)
                style.note_source(p.fs)
                method, prefix = p.fs.method, p.fs.prefix
            except (FileNotFoundError, ValueError, MemoryError, OSError) as e:
                print(f"  snapshot {snap:03d}: {e}")
                failed.append(snap)
                continue
            finally:
                p.release()
            if prof["definition"] != args.rv_definition:   # the catalogue predates the definitions: never mixed in
                print(f"  snapshot {snap:03d}: its void catalogue has no {args.rv_definition} R_v (a cache from before "
                      f"the R_v definitions; rebuilding it adds them); left out")
                failed.append(snap)
                continue
            stag = grids_tag(method, args.prefix) + options_tag(args)
            if stag != tag and args.method == "auto":       # auto found standard DTFE grids only: named by them
                tag = stag
            n_in = int(((prof["r_mpc"] >= rv_mpc[0]) & (prof["r_mpc"] < rv_mpc[1])).sum())
            fss = "" if prof.get("single_interior") is None or not len(prof["single_interior"]) else \
                (f", f_ss(<0.5 R_v) {np.nanmean(prof['single_interior'][:, 0]):.3f}, f_ss(<R_v) "
                 f"{np.nanmean(prof['single_interior'][:, 1]):.3f} (box {prof['single_box']:.3f})")
            print(f"  snapshot {snap:03d} (z={z:.2f}, {method}): {len(prof['index'])} voids profiled "
                  f"({prof['n_no_radius']} without R_v, {prof['n_overlapping']} overlapping, {prof['dropped']} dropped "
                  f"by the wrap rule), {n_in} in the overlay bin{fss}")
            save_snapshot(cache_dir / f"profiles_{snap:03d}{stag}.npz", snap, z, prof,
                          {"prefix": str(prefix or ""), "sigma_mpc": sigma, "sigma_cells": config.SMOOTHING_SIGMA_CELLS,
                           "footprint_cells": config.FOOTPRINT_SIZE})
    found = find_snapshots(cache_dir, tag)
    if not found and args.replot and args.method == "auto" and not args.prefix:
        dtfe_found = find_snapshots(cache_dir, "_dtfe" + options_tag(args))      # auto that found standard DTFE only
        if dtfe_found:
            tag, found = "_dtfe" + options_tag(args), dtfe_found
    if args.snaps:
        found = {s: q for s, q in found.items() if s in set(args.snaps)}
    if not found:
        raise SystemExit(f"no per-snapshot profiles under {cache_dir} (tag '{tag}')"
                         + ("; run without --replot first" if args.replot else ""))
    per = {s: load_snapshot(q) for s, q in found.items()}
    stale = sorted(s for s, d in per.items() if abs(d.get("sigma_mpc", sigma) - sigma) > 1e-6 * sigma)
    if stale:
        raise SystemExit(f"snapshot(s) {stale} were profiled at another smoothing; rerun without --replot")
    method = next(iter(per.values()))["method"]
    what = ("" if args.keep_overlaps else "distinct ") + args.sample
    rv_label = f"$R_v$ {rv_sig[0]:g}-{rv_sig[1]:g}$\\sigma$ ({rv_mpc[0]:.3g}-{rv_mpc[1]:.3g} Mpc)"
    ev = evolution(per, rv_mpc, args.n_boot, args.min_count)
    rows = all_fits(per, edges, args.min_count)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_evolution(ev, f"{args.sim}  {method}  {what} voids, r / R_v across z", out_dir / f"profile_evolution{tag}.png",
                   rv_label, args.min_count)
    plot_fits(rows, edges, sigma, f"{args.sim}  {method}  HSW 2014 fits per R_v bin", out_dir / f"profile_fits{tag}.png")
    meta = {"sim": args.sim, "method": method, "what": what, "sample": args.sample, "distinct": not args.keep_overlaps,
            "definition": str(next(iter(per.values()))["definition"]), "sigma_mpc": sigma,
            "edges_mpc": edges, "rv_range_sigma": rv_sig, "rv_range_mpc": list(rv_mpc), "rv_label": rv_label,
            "rv_text": f"R_v {rv_sig[0]:g}-{rv_sig[1]:g} sigma ({rv_mpc[0]:.3g}-{rv_mpc[1]:.3g} Mpc)",
            "n_boot": args.n_boot, "min_count": args.min_count, "model": "Hamaus, Sutter & Wandelt (2014): "
            "delta(r) = delta_c (1 - (r/r_s)^alpha) / (1 + (r/R_v)^beta), r_s in units of R_v",
            "snaps": sorted(per)}
    write_tables(out_dir / f"profile_fits{tag}", meta, rows, ev)
    arrays = {"snaps": np.array(sorted(per)), "redshift": np.array([per[s]["redshift"] for s in sorted(per)]),
              "bins": next(iter(per.values()))["bins"], "rv_range_mpc": np.array(rv_mpc), "edges_mpc": np.array(edges),
              "count": np.array([ev[s]["count"] for s in sorted(per)])}
    for s in sorted(per):
        st = ev[s]["stack"]
        for k in ("density", "v_r", "single"):
            if k in st:
                for part in ("mean", "lo", "hi"):
                    arrays[f"{k}_{part}_{s:03d}"] = st[k][part][0]
        if "v_lin" in ev[s]:
            arrays[f"v_lin_{s:03d}"] = ev[s]["v_lin"]
    np.savez_compressed(out_dir / f"profile_evolution{tag}.npz", **arrays)
    ok, good = sum(r["ok"] for r in rows), sum(r["constrained"] for r in rows)
    print(f"\n{len(per)} snapshot(s), {ok}/{len(rows)} fits converged, {good} constrained -> {out_dir}/profile_evolution{tag}.png, "
          f"profile_fits{tag}.png/.json/.txt" + (f"; failed: {failed}" if failed else ""))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
