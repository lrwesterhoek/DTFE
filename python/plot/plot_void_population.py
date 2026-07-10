"""POPULATION-level void statistics across cosmic time: counts, sizes, shapes, depths.

Complements plot_void_tracking.py (deep-dives on individual voids) with the population
answer: what did ALL voids do between z=20 and z=0? Uses the same pipeline catalogs
(pipeline.SnapshotProducts.voids(): smoothing, FFT Hessian, footprint, criterion, BBKS
shapes, ellipsoid cuts identical to the correlation analyses) and the same sample
definitions:

  * 'bbks'      : the correlation sample -- finite e/p with e >= 0 and |p| <= e
                  (identical to void_sample() in plot_marked_correlation_BBKS.py)
  * 'resolved'  : well_resolved (passes ELLIPSOID_CUTS) -- sizes/orientation defined
  * 'deep'      : well_resolved AND central delta < DEEP_VOID_THRESHOLD -- the tracking sample

Outputs (figures/void_population/<sim>/):
  population_evolution.png    counts, R_eff, e, p, central delta vs redshift
  population_distributions.png  R_eff / e / p / delta histograms at config.PANEL_SNAPSHOTS
  population_stats.npz        every per-snapshot statistic

Examples:
  python3 plot/plot_void_population.py --sim TNG50-3-Dark --method dtfe
  python3 plot/plot_void_population.py --sim TNG50-4-Dark --method ps
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config
from dtfelib import pipeline
from dtfelib import make_parser
from dtfelib import environment as env
from dtfelib import groupcat as gc

FIGURE_ROOT = Path(config.LOCAL_FIGURES_ROOT) / "void_population"
PCT = (16, 50, 84)   # band percentiles


def bbks_valid(cat):
    """The correlation scripts' validity mask (plot_marked_correlation_BBKS.void_sample)."""
    e, p = cat["bbks_params"][:, 0], cat["bbks_params"][:, 1]
    return np.isfinite(e) & np.isfinite(p) & (e >= 0) & (np.abs(p) <= e)


def snapshot_stats(cat):
    """Per-snapshot population statistics from one pipeline void catalog."""
    valid = bbks_valid(cat)
    resolved = cat["well_resolved"].astype(bool)
    deep = resolved & (cat["delta_values"] < config.DEEP_VOID_THRESHOLD)

    e, p = cat["bbks_params"][:, 0], cat["bbks_params"][:, 1]
    r_eff = np.where(resolved, np.prod(cat["semi_axes"], axis=1) ** (1.0 / 3.0), np.nan)

    def pct(x, sel):
        x = np.asarray(x, dtype=float)[sel]
        x = x[np.isfinite(x)]
        return np.percentile(x, PCT) if x.size else np.full(3, np.nan)

    return {
        "n_total": int(len(cat["coords"])),
        "n_bbks": int(valid.sum()),
        "n_resolved": int(resolved.sum()),
        "n_deep": int(deep.sum()),
        "r_eff_pct": pct(r_eff, resolved),
        "e_pct": pct(e, valid),
        "p_pct": pct(p, valid),
        "delta_pct": pct(cat["delta_values"], np.ones(len(e), bool)),
        # raw samples for the distribution figure
        "_r_eff": r_eff[resolved & np.isfinite(r_eff)],
        "_e": e[valid], "_p": p[valid],
        "_delta": np.asarray(cat["delta_values"], dtype=float),
    }


def main():
    parser = make_parser("Population-level void statistics across all snapshots "
                         "(same catalogs and sample cuts as the correlation analyses).")
    parser.add_argument("--panel-snaps", type=int, nargs="+", default=None,
                        help="snapshots for the distribution overlays "
                             f"(default: config.PANEL_SNAPSHOTS = {config.PANEL_SNAPSHOTS})")
    args = parser.parse_args()
    if args.smooth > 0:
        config.SMOOTHING_SIGMA_CELLS = args.smooth

    ztab = gc.redshift_table(args.sim)
    ladder = sorted({int(d.name.split("_")[1]) for d in (env.DATA_ROOT / args.sim).glob("snapdir_*")})

    stats, method_used = {}, {}
    for s in ladder:
        prod = pipeline.products(s, sim=args.sim, method=args.method)
        try:
            cat = prod.voids()
            method_used[s] = prod.fs.method
        except (FileNotFoundError, ValueError):
            continue
        finally:
            prod.release()
        stats[s] = snapshot_stats(cat)

    if not stats:
        raise SystemExit(f"no {args.method} field grids for {args.sim} -- nothing to analyse")
    snaps = sorted(stats)
    z = np.array([ztab.get(s, np.nan) for s in snaps])
    methods = sorted(set(method_used.values()))
    print(f"{args.sim} ({'/'.join(methods)}): population statistics at {len(snaps)} snapshots, "
          f"z = {np.nanmax(z):.2f} .. {np.nanmin(z):.2f}")
    for s in snaps:
        st = stats[s]
        print(f"  snap {s:3d} (z={ztab.get(s, float('nan')):6.2f}): total={st['n_total']:5d} "
              f"bbks={st['n_bbks']:5d} resolved={st['n_resolved']:4d} deep={st['n_deep']:4d}  "
              f"R_eff={st['r_eff_pct'][1]:5.2f} Mpc  e={st['e_pct'][1]:.3f}  "
              f"p={st['p_pct'][1]:+.3f}  delta={st['delta_pct'][1]:+.3f}")

    outdir = FIGURE_ROOT / args.sim
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- evolution figure ---------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(8, 15), sharex=True)
    ax = axes[0]
    for key, lab, c_ in (("n_total", "all catalog minima", "0.6"),
                         ("n_bbks", "BBKS-valid (correlation sample)", "tab:blue"),
                         ("n_resolved", "well-resolved", "tab:orange"),
                         ("n_deep", "deep + resolved (tracking sample)", "tab:red")):
        ax.plot(z, [stats[s][key] for s in snaps], "o-", ms=4, color=c_, label=lab)
    ax.set_yscale("log"); ax.set_ylabel("void count"); ax.legend(fontsize=8)
    ax.set_title(f"{args.sim} — void population evolution ({'/'.join(methods)}, "
                 f"$\\sigma$={config.SMOOTHING_SIGMA_CELLS:g} cells)")

    for ax, key, lab, c_ in ((axes[1], "r_eff_pct", r"$R_{\rm eff}$ [Mpc] (well-resolved)", "tab:purple"),
                             (axes[2], "e_pct", "BBKS ellipticity e (BBKS-valid)", "tab:blue"),
                             (axes[3], "delta_pct", r"central $\delta$ (all minima)", "tab:red")):
        lo, med, hi = (np.array([stats[s][key][i] for s in snaps]) for i in range(3))
        ax.plot(z, med, "o-", ms=4, color=c_, label=f"median {lab}")
        ax.fill_between(z, lo, hi, color=c_, alpha=0.2, label="16-84%")
        ax.set_ylabel(lab); ax.legend(fontsize=8)
    # overlay prolateness on the ellipticity panel
    lo, med, hi = (np.array([stats[s]["p_pct"][i] for s in snaps]) for i in range(3))
    axes[2].plot(z, med, "s--", ms=4, color="tab:orange", label="median p")
    axes[2].fill_between(z, lo, hi, color="tab:orange", alpha=0.15)
    axes[2].axhline(0, color="gray", lw=0.7, ls=":")
    axes[2].legend(fontsize=8)
    axes[3].set_xlabel("redshift z")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[-1].invert_xaxis()
    fig.tight_layout()
    fig.savefig(outdir / "population_evolution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- distribution figure ------------------------------------------------------------
    panel = args.panel_snaps or [int(s) for s in config.PANEL_SNAPSHOTS]
    panel = [s for s in panel if s in stats]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    specs = (("_r_eff", axes[0, 0], r"$R_{\rm eff}$ [Mpc]", {}),
             ("_e", axes[0, 1], "BBKS ellipticity e", {"range": (0, 0.6)}),
             ("_p", axes[1, 0], "BBKS prolateness p", {"range": (-0.4, 0.4)}),
             ("_delta", axes[1, 1], r"central $\delta$", {"range": (-1, 0.2)}))
    for key, ax, lab, hkw in specs:
        for s in panel:
            x = stats[s][key]
            if x.size:
                ax.hist(x, bins=30, density=True, histtype="step", lw=1.6,
                        label=f"z={ztab.get(s, float('nan')):.2f} (n={x.size})", **hkw)
        ax.set_xlabel(lab); ax.set_ylabel("pdf"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{args.sim} — void population distributions ({'/'.join(methods)})")
    fig.tight_layout()
    fig.savefig(outdir / "population_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        outdir / "population_stats.npz",
        snaps=np.array(snaps), redshift=z,
        **{k: np.array([stats[s][k] for s in snaps])
           for k in ("n_total", "n_bbks", "n_resolved", "n_deep",
                     "r_eff_pct", "e_pct", "p_pct", "delta_pct")})
    print(f"-> {outdir}/population_evolution.png, population_distributions.png, population_stats.npz")


if __name__ == "__main__":
    main()
