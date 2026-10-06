"""Void-centred stacked profiles in r / R_eff: density, radial velocity, single-stream fraction.

For every snapshot: the shells of every well-resolved void of the pipeline's catalogue (dtfelib.profiles:
the RAW rho/rho_bar, v_r positive = outflow, the single-stream fraction on phase-space grids), stacked in
bins of R_eff with 16-84% bootstrap bands, as a two- or three-panel figure plus an npz of the stacked
curves. The dashed lines on the v_r panel are linear theory from the stacked density itself
(profiles.linear_v_r): what the voids would do if they expanded linearly.

    python3 plot/plot_void_profiles.py --sim TNG100-3-Dark --snaps 99 50
    python3 plot/plot_void_profiles.py --sim TNG50-4-Dark --snap 99 --method dtfe --sample deep --edges 0,3,6,12
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import config
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dtfelib import make_parser, pipeline, profiles
from dtfelib import figures as style

OUTPUT_DIR = Path(config.LOCAL_FIGURES_ROOT) / "void_profiles"


def plot_stack(st, prof, title, path, z, linear=True):
    """One panel per stacked field, a curve (with its band) per R_eff bin; the dashed linear-theory v_r."""
    panels = [("density", r"$\rho / \bar\rho$"), ("v_r", r"$v_r$ [km/s]  (positive = outflow)")]
    if "single" in st:
        panels.append(("single", "single-stream fraction of the shell"))
    panels = [(k, lab) for k, lab in panels if k in st]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0), squeeze=False)
    nbin = len(st["edges"]) - 1
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(nbin, 1)))
    r_eff = np.asarray(prof["r_eff_mpc"])
    for ax, (k, lab) in zip(axes[0], panels):
        for i in range(nbin):
            if st["count"][i] < st["min_count"]:
                continue
            label = f"{st['edges'][i]:g}-{st['edges'][i + 1]:g} Mpc (n = {st['count'][i]})"
            ax.plot(st["bins"], st[k]["mean"][i], color=colors[i], label=label)
            ax.fill_between(st["bins"], st[k]["lo"][i], st[k]["hi"][i], color=colors[i], alpha=0.2, lw=0)
            if k == "v_r" and linear:
                sel = (r_eff >= st["edges"][i]) & (r_eff < st["edges"][i + 1])
                v_lin = profiles.linear_v_r(st["bins"], st["density"]["mean"][i], float(np.mean(r_eff[sel])), z)
                ax.plot(st["bins"], v_lin, "--", color=colors[i], lw=1.0)
        ax.axhline(1.0 if k == "density" else 0.0, color="gray", lw=0.7, ls=":")
        ax.axvline(1.0, color="gray", lw=0.7, ls=":")
        ax.set_xlabel(r"$r / R_{\rm eff}$")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        if k == "v_r" and linear:
            ax.text(0.03, 0.03, "dashed: linear theory from the stacked density", transform=ax.transAxes, fontsize=7, color="0.3")
        if k == "single":
            ax.set_ylim(0, 1.02)
    if any(st["count"][i] >= st["min_count"] for i in range(nbin)):
        axes[0][0].legend(fontsize=7, title=r"$R_{\rm eff}$ bin")
    else:
        axes[0][0].text(0.5, 0.5, f"no R_eff bin with {st['min_count']} voids", transform=axes[0][0].transAxes, ha="center", color="0.4")
    style.set_suptitle(fig, title)
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, path)
    plt.close(fig)


def main():
    parser = make_parser("Void-centred stacked profiles in r / R_eff (density, v_r, single-stream fraction).")
    parser.add_argument("--snaps", type=int, nargs="*", default=None, help="snapshots (default: --snap)")
    parser.add_argument("--sample", choices=("resolved", "deep"), default="resolved",
                        help="well-resolved voids, or those also deeper than config.DEEP_VOID_THRESHOLD")
    parser.add_argument("--edges", default=",".join(f"{e:g}" for e in profiles.DEFAULT_EDGES),
                        help="R_eff bin edges in Mpc, comma-separated")
    parser.add_argument("--points", type=int, default=profiles.DEFAULT_M, help="points per shell")
    parser.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples for the bands")
    parser.add_argument("--no-linear", action="store_true", help="no linear-theory v_r overlay")
    args = parser.parse_args()
    style.apply()
    if args.smooth > 0:
        config.SMOOTHING_SIGMA_CELLS = args.smooth       # the catalogue's cache namespace follows, as in the other scripts
    edges = [float(e) for e in args.edges.split(",") if e.strip()]
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise SystemExit(f"--edges must be at least two increasing numbers, got {args.edges!r}")
    snaps = args.snaps if args.snaps else [args.snap]
    failed, written = [], []
    for snap in snaps:
        p = pipeline.products(snap, sim=args.sim, method=args.method, prefix=args.prefix, data_root=args.data_root)
        try:
            p.data_dir                                      # raises for a missing snapdir / no grids / no snapshot file
            prof = profiles.void_profiles(p, m=args.points, sample=args.sample)
            z = p.redshift
        except (FileNotFoundError, ValueError, MemoryError, OSError) as e:      # OSError: h5py on an unreadable file
            print(f"  snapshot {snap:03d}: {e}")
            failed.append(snap)
            continue
        finally:
            p.release()
        st = profiles.stack(prof, edges_mpc=edges, n_boot=args.n_boot)
        counts = ", ".join(f"{edges[i]:g}-{edges[i + 1]:g} Mpc: {st['count'][i]}" for i in range(len(edges) - 1))
        print(f"  snapshot {snap:03d} (z={z:.2f}, {prof['method']}): {len(prof['index'])} {args.sample} voids profiled "
              f"({prof['dropped']} dropped by the wrap rule); per R_eff bin: {counts}")
        tag = args.prefix or prof["method"]            # named by what it holds, as the pk files are
        base = OUTPUT_DIR / args.sim / f"void_profiles_{tag}_{args.sample}_z{z:.2f}"    # suffixes APPENDED
        base.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"bins": st["bins"], "edges": st["edges"], "count": st["count"], "r_eff_mpc": prof["r_eff_mpc"],
                  "index": prof["index"], "delta_c": prof["delta_c"], "redshift": z, "sample": args.sample,
                  "method": prof["method"], "dropped": prof["dropped"], "points": prof["m"],
                  # what defines R_eff and the stack (a rerun with other knobs overwrites: the house convention)
                  "prefix": args.prefix or "", "sigma_cells": config.SMOOTHING_SIGMA_CELLS, "cell_mpc": prof["cell_mpc"],
                  "box_mpc": prof["box_mpc"], "n_grid": prof["n_grid"], "n_boot": st["n_boot"], "min_count": st["min_count"],
                  "cuts_mpc": [config.ELLIPSOID_CUTS[k] for k in ("min_axis_mpc", "max_axis_mpc", "max_axis_ratio")]}
        for k in ("density", "v_r", "single"):
            if k in st:
                for part in ("mean", "lo", "hi"):
                    arrays[f"{k}_{part}"] = st[k][part]
        np.savez_compressed(f"{base}.npz", **arrays)
        title = f"{args.sim}  z = {z:.2f}  {prof['method']}  ({args.sample} voids, {len(prof['index'])})"
        plot_stack(st, prof, title, Path(f"{base}.png"), z, linear=not args.no_linear)
        written.append(base)
    print(f"\n{len(written)} snapshot(s) under {OUTPUT_DIR / args.sim}" + (f"; failed: {failed}" if failed else ""))
    raise SystemExit(1 if failed or not written else 0)


if __name__ == "__main__":
    main()
