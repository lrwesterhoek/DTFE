"""The void population of several boxes (TNG50 / TNG100 / TNG300) side by side at ONE physical smoothing length.

At the production smoothing (10 cells) every box smooths on its own scale, and the cross-box differences of the void
statistics were that scale (2026-10-07: at sigma = 2.1625 Mpc TNG100 and TNG300 agree on the distinct-void density to
0.3% and on R_v to 3% at z = 0, 7-11% at z = 0.5-1, the resolved fraction to 2 points). This figure makes
the comparison at a common sigma in Mpc (--smooth-mpc, default 2.1625: 10.0002 TNG100-3-Dark cells, a hair above
its production 10 cells = 2.16245 Mpc, kept because the earlier results use it; the minima footprint follows it in
each box, pipeline.set_smoothing). The definitions are the per-box scripts':

  * resolved      : well_resolved (config.ELLIPSOID_CUT_RULES, in Mpc via pipeline.ellipsoid_cuts_mpc)
  * R_v           : the measured radius (catalog.r_v_mpc: where the smoothed delta around the minimum rises to 0)
  * distinct      : the resolved voids with an R_v, deepest first, none inside a deeper one's region
                    (catalog.distinct_mask) -- plot_void_population.snapshot_stats, the abundance sample
  * profiles      : profiles.void_profiles of the distinct resolved voids in r / R_v, stacked over ONE common R_v
                    range in Mpc (--rv-range, in smoothing lengths: 2.5-5 sigma by default); the wrap rule
                    (3 R_v <= L/2) leaves a small box only its smaller voids, said in the legend

Every box runs in its OWN process (DTFE_SIM=<sim>: config's frame -- cell, box -- is fixed at import), writing
crossbox_<sim>_<method>_s<sigma>Mpc.npz; this process draws them together. Catalogues come from (and go into) the
pipeline cache under the sigma's own namespace, so a rerun is fast; --replot only redraws from the .npz files.

Outputs (figures/void_crossbox/, never mirrored into the thesis tree -- a common sigma is not the production one):
  crossbox_<method>_s<sigma>Mpc_z<z>.png   top: resolved fraction, distinct voids per (100 Mpc)^3 and median R_v
                                           against z; bottom, at z: dn/dlnR_v and the stacked density and v_r
  crossbox_<method>_s<sigma>Mpc.txt        the numbers per box and snapshot
  crossbox_<sim>_<method>_s<sigma>Mpc.npz  one per box (the worker's output)

Examples:
  python3 plot/plot_void_crossbox.py                                  # the three -3-Dark boxes, sigma 2.1625 Mpc
  python3 plot/plot_void_crossbox.py --sims TNG100-3-Dark TNG300-3-Dark --smooth-mpc 3 --z 0 1
  python3 plot/plot_void_crossbox.py --replot                         # redraw from the saved .npz files
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dtfelib.cli import DATA_ROOT

DEFAULT_SIMS = ("TNG50-3-Dark", "TNG100-3-Dark", "TNG300-3-Dark")
DEFAULT_SIGMA_MPC = 2.1625            # 10.0002 cells of TNG100-3-Dark (110.7 Mpc / 512; production 10 cells = 2.16245
                                      # Mpc): the 2026-10-07 comparison's, kept so its results stay comparable
DEFAULT_Z = (0.0, 0.5, 1.0)
DEFAULT_RV_RANGE = (2.5, 5.0)         # the common R_v range of the stacks, in smoothing lengths (profiles' upper bins)
Z_TOLERANCE = 0.05                    # a box's snapshot counts as 'at z' within this
FIGURE_ROOT = Path(config.LOCAL_FIGURES_ROOT) / "void_crossbox"
PER_VOLUME = 1.0e6                    # densities per (100 Mpc)^3


def sigma_tag(sigma_mpc: float) -> str:
    """'s2.1625Mpc': the sigma in file names, six significant digits as pipeline.smoothing_tag (whose footprint part
    differs per box: it is in each box's .npz instead)."""
    return f"s{float(sigma_mpc):.6g}Mpc"


def worker_path(out_dir: Path, sim: str, method: str, sigma_mpc: float) -> Path:
    return Path(out_dir) / f"crossbox_{sim}_{method}_{sigma_tag(sigma_mpc)}.npz"


# ---------------------------------------------------------------------------------------------------------------
# the worker: ONE box, in a process whose config frame is that box's (DTFE_SIM)
# ---------------------------------------------------------------------------------------------------------------

def measure(sim, method="ps", prefix=None, data_root=None, zs=DEFAULT_Z, rv_range_sigma=DEFAULT_RV_RANGE,
            points=None, n_boot=200, log=print) -> dict:
    """Every snapshot of 'sim' with grids: plot_void_population.snapshot_stats (counts, R_v of the distinct
    resolved voids); at the snapshots nearest 'zs' (within Z_TOLERANCE) the profiles of the distinct resolved voids
    in r/R_v, stacked over rv_range_sigma x sigma. Uses the smoothing as set NOW (pipeline.set_smoothing) and
    config's frame, which must be this sim's. Returns flat arrays for an .npz."""
    from dtfelib import pipeline, catalog, profiles, use_data_root
    from dtfelib import environment as env
    from dtfelib import groupcat as gc
    import plot_void_population as pop

    data_root = data_root or DATA_ROOT
    use_data_root(data_root)                       # the trees' redshift table and the ladder read it too
    sigma = pipeline.smoothing_length_mpc()
    ztab = gc.redshift_table(sim)
    ladder = sorted({int(d.name.split("_")[1]) for d in env.sim_dir(sim, data_root).glob("snapdir_*")})
    rows, refused = [], []
    for s in ladder:
        prod = pipeline.products(s, sim=sim, method=method, prefix=prefix, data_root=data_root)
        try:
            cell, box, n, _src = catalog.frame_of(prod)
            cat = prod.voids()
            used = prod.fs.method
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as e:            # an unreadable file, or a frame not config's: said, left out
            log(f"  {sim} snap {s:3d}: left out: {e}")
            refused.append(s)
            continue
        finally:
            prod.release()
        st = pop.snapshot_stats(cat, n)
        rows.append({"snap": s, "z": float(ztab.get(s, np.nan)), "box": float(box), "cell": float(cell),
                     "n": int(n), "method": used, **st})
    if not rows:
        raise SystemExit(f"{sim}: no {method} void catalogue at any snapshot" + (f" ({len(refused)} refused)" if refused else ""))

    out = {"sim": sim, "method": rows[-1]["method"], "sigma_mpc": float(sigma),
           "sigma_cells": float(config.SMOOTHING_SIGMA_CELLS), "footprint": int(config.FOOTPRINT_SIZE),
           "cache_namespace": pipeline._param_hash(), "box_mpc": rows[-1]["box"], "cell_mpc": rows[-1]["cell"],
           "n_grid": rows[-1]["n"], "refused": np.array(refused, dtype=int),
           "snaps": np.array([r["snap"] for r in rows]), "redshift": np.array([r["z"] for r in rows]),
           "box_per_snap": np.array([r["box"] for r in rows]),
           "r_v_pct": np.array([r["r_v_pct"] for r in rows]),
           "r_v_concat": np.concatenate([r["_r_v"] for r in rows]).astype(np.float64),
           "r_v_count": np.array([r["_r_v"].size for r in rows])}
    for k in ("n_total", "n_resolved", "n_deep", "n_counted", "n_no_rv", "n_overlapping"):
        out[k] = np.array([r[k] for r in rows])

    # ---- the stacks at the target redshifts: one common R_v range in Mpc
    lo, hi = (float(k) * sigma for k in rv_range_sigma)
    zs = [float(z) for z in zs]
    out.update(pz_target=np.array(zs), pz_rv_range_mpc=np.array([lo, hi]), pz_rv_range_sigma=np.array(rv_range_sigma, float))
    snap_of = {}
    for zt in zs:
        near = min(rows, key=lambda r: abs(r["z"] - zt) if np.isfinite(r["z"]) else np.inf)
        snap_of[zt] = near["snap"] if np.isfinite(near["z"]) and abs(near["z"] - zt) <= Z_TOLERANCE else -1
    nb = len(profiles.DEFAULT_BINS)

    def blank():
        return np.full((len(zs), nb), np.nan)
    out["pz_snap"] = np.array([snap_of[z] for z in zs])
    out["pz_redshift"] = np.full(len(zs), np.nan)
    out["pz_count"], out["pz_profiled"], out["pz_dropped"] = (np.zeros(len(zs), int) for _ in range(3))
    out["pz_bins"] = np.asarray(profiles.DEFAULT_BINS, dtype=np.float64)
    for k in ("density", "v_r", "single"):
        for part in ("mean", "lo", "hi"):
            out[f"pz_{k}_{part}"] = blank()
    for i, zt in enumerate(zs):
        s = snap_of[zt]
        if s < 0:
            log(f"  {sim}: no snapshot within {Z_TOLERANCE} of z = {zt:g}")
            continue
        prod = pipeline.products(s, sim=sim, method=method, prefix=prefix, data_root=data_root)
        try:
            prof = profiles.void_profiles(prod, m=points or profiles.DEFAULT_M, sample="resolved", radius="r_v",
                                          distinct=True)
            out["pz_redshift"][i] = prod.redshift
        except (OSError, ValueError, MemoryError) as e:
            log(f"  {sim} snap {s:3d}: no profiles: {e}")
            continue
        finally:
            prod.release()
        st = profiles.stack(prof, edges_mpc=[lo, hi], n_boot=n_boot)
        out["pz_count"][i] = int(st["count"][0])
        out["pz_profiled"][i] = len(prof["index"])
        out["pz_dropped"][i] = int(prof["dropped"])
        for k in ("density", "v_r", "single"):
            if k in st:
                for part in ("mean", "lo", "hi"):
                    out[f"pz_{k}_{part}"][i] = st[k][part][0]
        log(f"  {sim} snap {s:3d} (z={out['pz_redshift'][i]:.2f}): {out['pz_count'][i]} of {len(prof['index'])} "
            f"profiled distinct voids with R_v in {lo:.3g}-{hi:.3g} Mpc ({prof['dropped']} dropped by the wrap rule)")
    return out


def worker_main(argv):
    p = argparse.ArgumentParser(description="plot_void_crossbox.py's per-box worker (run by the script itself)")
    p.add_argument("--worker", type=Path, required=True, help="the .npz to write")
    p.add_argument("--sim", required=True)
    p.add_argument("--method", default="ps")
    p.add_argument("--prefix", default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--smooth-mpc", type=float, required=True)
    p.add_argument("--z", type=float, nargs="+", default=list(DEFAULT_Z))
    p.add_argument("--rv-range", type=float, nargs=2, default=list(DEFAULT_RV_RANGE))
    p.add_argument("--points", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=200)
    a = p.parse_args(argv)
    from dtfelib import pipeline
    if os.environ.get("DTFE_SIM", a.sim) != a.sim:
        raise SystemExit(f"the worker for {a.sim} runs under DTFE_SIM={os.environ['DTFE_SIM']}: config's frame is not its")
    pipeline.apply_smoothing_args(a, f"crossbox {a.sim}")
    res = measure(a.sim, a.method, a.prefix, a.data_root, a.z, a.rv_range, a.points, a.n_boot)
    a.worker.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.worker.with_name(a.worker.name + ".partial.npz")
    np.savez_compressed(tmp, **res)
    tmp.replace(a.worker)                          # whole or not at all: --replot never reads half a file


def run_worker(sim, args, path) -> int:
    """One box in its own process (DTFE_SIM=sim), its log passed through; the exit code."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(path), "--sim", sim, "--method", args.method,
           "--smooth-mpc", repr(float(args.smooth_mpc)), "--z", *[repr(float(z)) for z in args.z],
           "--rv-range", *[repr(float(r)) for r in args.rv_range], "--n-boot", str(args.n_boot)]
    for flag, val in (("--prefix", args.prefix), ("--data-root", args.data_root), ("--points", args.points)):
        if val is not None:
            cmd += [flag, str(val)]
    env = dict(os.environ, DTFE_SIM=sim)
    env.pop("DTFE_SIGMA_MPC", None)               # the sigma is --smooth-mpc's, whatever the shell exports
    print(f"\n=== {sim}")
    return subprocess.call(cmd, env=env)


# ---------------------------------------------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------------------------------------------

def load(path) -> dict:
    """A worker's .npz, its scalars (the sim's name, sigma, the box) as Python values."""
    with np.load(path, allow_pickle=False) as f:
        return {k: (f[k].item() if f[k].ndim == 0 else f[k]) for k in f.files}


def short(sim: str) -> str:
    return str(sim).split("-")[0]


def r_v_at(res, snap) -> np.ndarray:
    """The R_v of the distinct resolved voids at one snapshot (the worker's ragged concatenation)."""
    ends = np.cumsum(res["r_v_count"])
    i = int(np.nonzero(res["snaps"] == snap)[0][0])
    return res["r_v_concat"][ends[i] - res["r_v_count"][i]:ends[i]]


def summary_lines(results) -> list:
    lines = []
    for r in results:
        lines.append(f"{r['sim']} ({r['method']}): box {float(r['box_mpc']):.4g} Mpc, cell {float(r['cell_mpc']):.4g} Mpc, "
                     f"sigma {float(r['sigma_mpc']):.6g} Mpc = {float(r['sigma_cells']):.4g} cells, footprint "
                     f"{int(r['footprint'])} ({str(r['cache_namespace'])})")
        vol = float(r["box_mpc"]) ** 3
        lines.append("   snap      z   minima  resolved  frac   distinct  per(100Mpc)^3  R_v 16/50/84 [Mpc]")
        for j, s in enumerate(r["snaps"]):
            nt, nr, nc = int(r["n_total"][j]), int(r["n_resolved"][j]), int(r["n_counted"][j])
            pc = r["r_v_pct"][j]
            lines.append(f"   {int(s):4d} {float(r['redshift'][j]):6.2f} {nt:8d} {nr:9d} {nr / max(nt, 1):5.3f} {nc:9d} "
                         f"{nc / vol * PER_VOLUME:14.1f}  {pc[0]:5.2f} {pc[1]:5.2f} {pc[2]:5.2f}")
    return lines


def draw(results, z_target, args, out_dir) -> Path:
    from dtfelib import catalog
    sigma = float(args.smooth_mpc)
    colors = dict(zip([r["sim"] for r in results], plt.cm.tab10(np.arange(len(results)))))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    # ---- top: against redshift
    for r in results:
        c, lab = colors[r["sim"]], short(r["sim"])
        z = r["redshift"]
        ok = np.isfinite(z)
        nt, nr, nc = (np.asarray(r[k], float) for k in ("n_total", "n_resolved", "n_counted"))
        vol = np.asarray(r["box_per_snap"], float) ** 3
        axes[0, 0].plot(z[ok], (nr / np.maximum(nt, 1))[ok], "o-", ms=4, color=c, label=lab)
        axes[0, 1].errorbar(z[ok], (nc / vol * PER_VOLUME)[ok], yerr=(np.sqrt(nc) / vol * PER_VOLUME)[ok], fmt="o-",
                            ms=4, capsize=2, color=c, label=f"{lab} ({float(r['box_mpc']):.0f} Mpc box)")
        pc = np.asarray(r["r_v_pct"], float)
        axes[0, 2].plot(z[ok], pc[ok, 1], "o-", ms=4, color=c, label=lab)
        for q in (0, 2):                           # the 16/84% as thin lines: three overlapping bands hid the medians
            axes[0, 2].plot(z[ok], pc[ok, q], "--", lw=0.8, color=c, alpha=0.7)
    axes[0, 0].set_ylabel("resolved fraction of the minima")
    axes[0, 1].set_ylabel(r"distinct voids per $(100\,$Mpc$)^3$")
    axes[0, 2].set_ylabel(r"$R_v$ of the distinct voids [Mpc] (median; dashed 16, 84%)")
    for ax in axes[0]:
        ax.set_xlabel("redshift z"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.invert_xaxis()                          # high z on the left, as in the population figures
    axes[0, 1].set_yscale("log")
    # ---- bottom: at z_target
    edges = np.geomspace(0.3 * sigma, 30.0 * sigma, 16)
    mid = np.sqrt(edges[1:] * edges[:-1])
    # the range the stacks were MADE over (the .npz's: a --replot need not repeat --rv-range)
    lo_mpc, hi_mpc = (float(v) for v in results[0]["pz_rv_range_mpc"])
    lo_sig, hi_sig = (float(v) for v in results[0]["pz_rv_range_sigma"])
    abundance = False                              # anything positive to put on the log-log axes
    for r in results:
        c, lab = colors[r["sim"]], short(r["sim"])
        i = int(np.argmin(np.abs(np.asarray(r["pz_target"], float) - z_target)))
        if abs(float(r["pz_target"][i]) - z_target) > 1e-9 or int(r["pz_snap"][i]) < 0:
            for ax in axes[1]:
                ax.plot([], [], color=c, ls=":", label=f"{lab}: no snapshot at z = {z_target:g}")
            continue
        snap = int(r["pz_snap"][i])
        rv = r_v_at(r, snap)
        dn, derr, cnt = catalog.void_dn_dlnr(rv, float(r["box_mpc"]), edges)
        okd = cnt > 0
        abundance |= bool(okd.any())
        axes[1, 0].errorbar(mid[okd], dn[okd] * PER_VOLUME, yerr=derr[okd] * PER_VOLUME, fmt="o-", ms=3, capsize=2,
                            color=c, label=f"{lab} (n={rv.size})")
        x = np.asarray(r["pz_bins"], float)
        n_in = int(r["pz_count"][i])
        cap = float(r["box_mpc"]) / 6.0
        note = f"{lab}: {n_in} voids" + (f", R_v <= L/6 = {cap:.3g} Mpc" if cap < hi_mpc else "")
        for ax, k in ((axes[1, 1], "density"), (axes[1, 2], "v_r")):
            m = np.asarray(r[f"pz_{k}_mean"][i], float)
            if not np.isfinite(m).any():
                ax.plot([], [], color=c, ls=":", label=f"{note} (too few)")
                continue
            ax.plot(x, m, color=c, lw=1.6, label=note)
            ax.fill_between(x, r[f"pz_{k}_lo"][i], r[f"pz_{k}_hi"][i], color=c, alpha=0.2)
    if abundance:
        axes[1, 0].set_xscale("log"); axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel(r"$R_v$ [Mpc]"); axes[1, 0].set_ylabel(r"$dn/d\ln R_v$ per $(100\,$Mpc$)^3$")
    axes[1, 0].set_title(f"distinct resolved voids, z = {z_target:g}", fontsize=10)
    axes[1, 1].set_ylabel(r"$\rho / \bar\rho$"); axes[1, 2].set_ylabel(r"$v_r$ [km/s]  (positive = outflow)")
    for ax, base in ((axes[1, 1], 1.0), (axes[1, 2], 0.0)):
        ax.axhline(base, color="gray", lw=0.7, ls=":")
        ax.axvline(1.0, color="gray", lw=0.7, ls=":")
        ax.set_xlabel(r"$r / R_v$")
        ax.set_title(f"stacked, R_v {lo_mpc:.3g}-{hi_mpc:.3g} Mpc ({lo_sig:g}-{hi_sig:g} sigma), "
                     f"z = {z_target:g}", fontsize=10)
    for ax in axes[1]:
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    methods = "/".join(sorted({str(r["method"]) for r in results}))
    fig.suptitle(f"Voids across boxes at one smoothing length: sigma = {sigma:.6g} Mpc ({methods}; the footprint "
                 f"follows sigma in each box)")
    fig.tight_layout()
    path = out_dir / f"crossbox_{args.method}_{sigma_tag(sigma)}_z{z_target:.2f}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--worker" in argv:
        return worker_main(argv)
    p = argparse.ArgumentParser(description="Void statistics of several boxes side by side at one smoothing length "
                                            "in Mpc (resolved fraction, distinct-void density, R_v, stacked profiles).")
    p.add_argument("--sims", nargs="+", default=list(DEFAULT_SIMS), help=f"the boxes (default {' '.join(DEFAULT_SIMS)})")
    p.add_argument("--method", choices=("ps", "dtfe"), default="ps", help="the field estimator (default ps)")
    p.add_argument("--prefix", default=None, help="a grid set's output prefix (default the method's)")
    p.add_argument("--data-root", type=Path, default=None, help=f"the data root (default {DATA_ROOT})")
    p.add_argument("--smooth-mpc", type=float, default=DEFAULT_SIGMA_MPC, metavar="MPC",
                   help=f"the common smoothing length in Mpc (default {DEFAULT_SIGMA_MPC}: 10.0002 TNG100-3-Dark cells, "
                        "just above its production 10 cells = 2.16245 Mpc; the earlier results use it)")
    p.add_argument("--z", type=float, nargs="+", default=list(DEFAULT_Z),
                   help="the redshifts of the bottom row, one figure each (default 0 0.5 1)")
    p.add_argument("--rv-range", type=float, nargs=2, default=list(DEFAULT_RV_RANGE), metavar=("LO", "HI"),
                   help="the stacks' common R_v range in smoothing lengths (default 2.5 5)")
    p.add_argument("--points", type=int, default=None, help="points per profile shell (default profiles.DEFAULT_M)")
    p.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples for the profile bands")
    p.add_argument("--out", type=Path, default=None, help=f"output folder (default {FIGURE_ROOT})")
    p.add_argument("--replot", action="store_true", help="only redraw from the boxes' saved .npz files")
    args = p.parse_args(argv)
    if not args.smooth_mpc > 0:
        raise SystemExit("--smooth-mpc must be positive")
    if not 0 <= args.rv_range[0] < args.rv_range[1]:
        raise SystemExit(f"--rv-range must be 0 <= LO < HI, got {args.rv_range}")
    out_dir = (args.out or FIGURE_ROOT).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    failed, results = [], []
    for sim in args.sims:
        path = worker_path(out_dir, sim, args.method, args.smooth_mpc)
        if not args.replot and run_worker(sim, args, path) != 0:
            failed.append(sim)
            continue
        if not path.is_file():
            print(f"{sim}: no {path.name}" + ("" if args.replot else " (the worker wrote nothing)"))
            failed.append(sim)
            continue
        res = load(path)
        if args.replot and [round(float(z), 9) for z in res["pz_target"]] != [round(float(z), 9) for z in args.z]:
            print(f"{sim}: its .npz holds the stacks at z = {list(res['pz_target'])}: the bottom row follows those")
        results.append(res)
    ranges = {tuple(np.round(r["pz_rv_range_mpc"], 6)) for r in results}
    if len(ranges) > 1:
        print(f"the boxes' stacks were made over different R_v ranges {sorted(ranges)}: the titles give the first's")
    if not results:
        raise SystemExit("no box measured: nothing to draw")
    lines = summary_lines(results)
    print("\n" + "\n".join(lines))
    table = out_dir / f"crossbox_{args.method}_{sigma_tag(args.smooth_mpc)}.txt"
    table.write_text("\n".join(lines) + "\n")
    zs = sorted({round(float(z), 9) for r in results for z in r["pz_target"]})
    written = [draw(results, z, args, out_dir) for z in zs]
    print(f"\n-> {out_dir}: {', '.join(q.name for q in written)}, {table.name}"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
