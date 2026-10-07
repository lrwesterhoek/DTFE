"""Density power spectrum P(k): the estimators' grids against linear theory.

For every snapshot: P(k) of rho/rho_bar - 1 from every estimator whose grids the snapshot has (the
standard DTFE 'output.*' and the phase-space 'ps_output.*', or the --prefix given), the linear-theory
P(k) of config.COSMOLOGY at that redshift (Eisenstein & Hu 1998 no-wiggle, sigma_8-normalised, D(z)^2;
dtfelib.spectra), and their ratios. The Nyquist k = pi N / L is marked: the grid's sampling aliases the
power above about half of it, and there is no assignment window to deconvolve (a DTFE grid is a cell
average, not a CIC assignment), so the comparison is read below k_Ny / 2. With one estimator on disk
the figure shows that one against linear theory and says so.

The lowest bins hold a handful of modes (TNG100: 3, 6, 4, 3 independent complex modes in the first four), so
ONE realisation scatters far from linear theory there: TNG100's 2 k_f bin (k = 0.113 /Mpc) sits at 0.09x
linear -- 0.3% in the chi^2 tail -- and the SAME dip is in its z = 127 initial conditions (0.10x), while the
PS-DTFE grid equals a plain particle CIC there to 0.1% (2026-10-07). Hence: the error bars are the chi^2_M/M
68% interval (not +-P sqrt(2/M), which is symmetric and far too narrow below at M = 6), the ratio panel shades
the 95% cosmic-variance band where linear theory holds (Delta^2_lin < 1, the nonlinear scale: k < ~0.15 /Mpc at
z = 0), and a
third panel divides by the
SAME realisation's early spectrum grown linearly (--ref-snap, default snapshot 0, z = 20): its phases and
amplitudes cancel, so what is left is growth (1 where linear) -- the dip disappears there. The default
binning stays 40 log bins (single |n| shells at low k); --nbins 0 gives k_f-wide linear bins.

    python3 plot/plot_pk_compare.py --sim TNG100-3-Dark --snaps 99 50
    python3 plot/plot_pk_compare.py --sim TNG50-4-Dark --snap 99 --method dtfe --nbins 30
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import config
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import chi2

from dtfelib import make_parser, FieldSet, sim_dir, spectra
from dtfelib import figures as style

OUTPUT_DIR = Path(config.LOCAL_FIGURES_ROOT) / "power_spectrum"
LABEL = {"dtfe": "DTFE", "ps": "PS-DTFE"}
MIRROR = True                   # into the thesis Figures tree too; off under --out (main sets it)


def spectra_of(snapdir, methods, prefix, nbins):
    """{method: power_spectrum dict} for the estimators with grids in snapdir; the redshift and box."""
    out, z, box = {}, None, None
    for m in methods:
        try:
            fs = FieldSet(snapdir, method=m, prefix=prefix)
            delta = fs.density(units="mean") - 1.0
            box = float(fs.meta.box_mpc)
            z = float(fs.meta.redshift)
        except (FileNotFoundError, ValueError, OSError) as e:    # OSError: h5py on an unreadable snapshot file
            print(f"    {LABEL.get(m, m)}: {e}")
            continue
        out[m] = spectra.power_spectrum(delta, box, nbins=nbins or None)     # 0: k_f-wide linear bins
        del delta
        print(f"    {LABEL.get(m, m)}: {fs.grid_n}^3 cells, k_f = {out[m]['k_f']:.4f}, k_Ny = {out[m]['k_nyquist']:.3f} /Mpc")
    return out, z, box


def same_edges(a, b) -> bool:
    """Whether two spectra share their k bins (a shape test first: --nbins 0 gives N-dependent bin counts)."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.allclose(a, b)


def chi2_interval(P, modes, lo=0.16, hi=0.84):
    """The (lo, hi) interval of the true P given the measured P of a bin of 'modes' Hermitian-weighted modes
    (P_hat / P ~ chi^2_M / M for a Gaussian field): P M / chi2.ppf(hi, M) .. P M / chi2.ppf(lo, M)."""
    m = np.asarray(modes, dtype=np.float64)
    return P * m / chi2.ppf(hi, m), P * m / chi2.ppf(lo, m)


def plot_pk(ps, z, box, title, path, labels=None, note=None, ref=None):
    """The spectra of 'ps' ({method: power_spectrum dict}) against linear theory, with each estimator's own
    k_Ny (dashed) and k_Ny/2 (dotted) when the grids differ in size (one grey pair when they agree), the
    ratio panel on a LOG axis spanning the plotted bins (a linear 0-50 cap hid the bins below k_Ny/2 and
    squashed the linear regime) with the 95% Gaussian band where linear theory holds, a note inside the axes
    (titles are off under the house style); with ref = ({method: power_spectrum dict}, z_ref) a third panel,
    P / [P_ref (D(z)/D(z_ref))^2] -- the same realisation, so its cosmic variance cancels."""
    labels = labels or LABEL
    ref_ps, z_ref = ref if ref else ({}, None)
    use_ref = z_ref is not None and abs(z_ref - z) > 1e-6 and any(
        m in ref_ps and same_edges(ref_ps[m]["edges"], ps[m]["edges"]) for m in ps)
    if z_ref is not None and abs(z_ref - z) > 1e-6 and not use_ref:
        print("    no same-realisation panel: the reference snapshot's k bins differ (another grid size?)")
    fig, axes = plt.subplots(3 if use_ref else 2, 1, figsize=(6.4, 9.0 if use_ref else 7.2), sharex=True,
                             gridspec_kw={"height_ratios": (3, 1.4, 1.4) if use_ref else (3, 1.4)})
    ax, axr = axes[0], axes[1]
    axg = axes[2] if use_ref else None
    k_nys = {m: p["k_nyquist"] for m, p in ps.items()}
    k_max = max(k_nys.values())
    k = np.geomspace(min(p["k_f"] for p in ps.values()) * 0.5, k_max * 1.5, 300)
    lin = spectra.linear_pk(k, z)
    ax.plot(k, lin, "k-", lw=1.2, label=f"linear theory, z = {z:.2f} (EH98, $\\sigma_8$ = {config.COSMOLOGY['sigma8']})")
    colors = {"dtfe": "#1f5fa8", "ps": "#e4572e"}
    ratios = []
    for i, (m, p) in enumerate(ps.items()):
        ok = p["modes"] > 0
        lo68, hi68 = chi2_interval(p["P"][ok], p["modes"][ok])
        err = np.vstack([p["P"][ok] - lo68, hi68 - p["P"][ok]])
        plin = spectra.linear_pk(p["k"][ok], z)
        r = p["P"][ok] / plin
        ratios.append(r)
        ax.errorbar(p["k"][ok], p["P"][ok], yerr=err, fmt="o-", ms=3, lw=1.0, color=colors.get(m, None), label=labels.get(m, m), capsize=0)
        axr.plot(p["k"][ok], r, "o-", ms=3, lw=1.0, color=colors.get(m, None))
        if i == 0:                              # the band of one Gaussian realisation, where the linear null holds
            mm = p["modes"][ok].astype(np.float64)
            lin_ok = p["k"][ok] ** 3 * plin / (2.0 * np.pi ** 2) < 1.0          # Delta^2_lin < 1: k < ~0.15 /Mpc at z=0
            if lin_ok.any():
                axr.fill_between(p["k"][ok][lin_ok], chi2.ppf(0.025, mm[lin_ok]) / mm[lin_ok],
                                 chi2.ppf(0.975, mm[lin_ok]) / mm[lin_ok], color="0.5", alpha=0.2, lw=0,
                                 label="95% of a Gaussian field (cosmic variance)")
                axr.legend(fontsize=6, loc="lower right")
        if use_ref and m in ref_ps and same_edges(ref_ps[m]["edges"], p["edges"]):
            g = (float(spectra.growth_factor(z)) / float(spectra.growth_factor(z_ref))) ** 2
            rp = ref_ps[m]["P"][ok] * g
            pos = rp > 0                                # a bin with no reference power has no ratio
            axg.plot(p["k"][ok][pos], p["P"][ok][pos] / rp[pos], "o-", ms=3, lw=1.0, color=colors.get(m, None))
    same = max(k_nys.values()) / min(k_nys.values()) < 1.0 + 1e-9
    for a in axes:
        if same:
            a.axvline(k_max, color="gray", lw=0.8, ls="--")
            a.axvline(k_max / 2, color="gray", lw=0.6, ls=":")
        else:
            for m, kn in k_nys.items():
                a.axvline(kn, color=colors.get(m, "gray"), lw=0.8, ls="--")
                a.axvline(kn / 2, color=colors.get(m, "gray"), lw=0.6, ls=":")
        a.set_xscale("log")
        a.grid(alpha=0.3, which="both")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P(k)$ [Mpc$^3$]")
    if same:
        ax.text(k_max, ax.get_ylim()[1] / 2.0, r" $k_{\rm Ny}$", color="gray", fontsize=8, va="top")
        ax.text(k_max / 2, ax.get_ylim()[0] * 1.5, r"$k_{\rm Ny}/2$: aliased above ", color="gray", fontsize=7, ha="right")
    else:
        for m, kn in k_nys.items():
            ax.text(kn, ax.get_ylim()[1] / 2.0, f" $k_{{\\rm Ny}}$ ({labels.get(m, m)})", color=colors.get(m, "gray"), fontsize=7, va="top")
        ax.text(0.98, 0.98, "dotted: each estimator's $k_{\\rm Ny}/2$, aliased above", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="gray")
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7, color="gray")
    ax.legend(fontsize=8, loc="lower left" if not note else "upper right")
    axr.axhline(1.0, color="k", lw=0.8)
    axr.set_ylabel("P / linear")
    if use_ref:
        axg.axhline(1.0, color="k", lw=0.8)
        axg.set_yscale("log")
        axg.set_ylabel(f"P / [P(z={z_ref:.1f}) grown]")
        axg.set_xlabel(r"$k$ [Mpc$^{-1}$] (comoving, h-free)")
        axg.text(0.02, 0.92, "same realisation: its cosmic variance cancels; 1 = linear growth", transform=axg.transAxes,
                 fontsize=7, color="gray", va="top")
    else:
        axr.set_xlabel(r"$k$ [Mpc$^{-1}$] (comoving, h-free)")
    allr = np.concatenate(ratios) if ratios else np.array([1.0])
    allr = allr[np.isfinite(allr) & (allr > 0)]
    axr.set_yscale("log")
    if allr.size:
        axr.set_ylim(max(0.05, 0.7 * float(allr.min())), 1.5 * float(allr.max()))
    style.set_suptitle(fig, title)
    fig.tight_layout()
    style.save_plot_to_multiple_paths(fig, path, mirror=MIRROR)
    plt.close(fig)


def main():
    global MIRROR
    parser = make_parser("Density power spectrum: DTFE and PS-DTFE grids against linear theory.")
    parser.add_argument("--snaps", type=int, nargs="*", default=None, help="snapshots (default: --snap)")
    parser.add_argument("--nbins", type=int, default=40,
                        help="logarithmic k bins from k_f to k_Ny (default 40; 0 = k_f-wide linear bins)")
    parser.add_argument("--ref-snap", type=int, default=0,
                        help="the early snapshot whose growth-scaled P(k) is the same-realisation reference panel "
                             "(default 0; -1 = no panel)")
    parser.add_argument("--out", default=None, help=f"output folder (default {OUTPUT_DIR}/<sim>)")
    args = parser.parse_args()
    style.apply()
    out_dir = Path(args.out).expanduser() if args.out else OUTPUT_DIR / args.sim
    MIRROR = args.out is None
    methods = ("dtfe", "ps") if args.method == "auto" and args.prefix is None else (args.method if args.method != "auto" else "ps",)
    labels = dict(LABEL)
    if args.prefix:                                 # the curve is named by what was loaded
        labels = {m: f"{LABEL.get(m, m)} ({args.prefix})" for m in LABEL}
    snaps = args.snaps if args.snaps else [args.snap]
    failed, written = [], []
    ref = None
    if args.ref_snap >= 0:
        print(f"  reference snapshot {args.ref_snap:03d}:")
        rps, rz, _ = spectra_of(sim_dir(args.sim, args.data_root) / f"snapdir_{args.ref_snap:03d}", methods, args.prefix,
                                args.nbins)
        ref = (rps, rz) if rps else None
        if ref is None:
            print(f"    no grid: no same-realisation panel")
    for snap in snaps:
        print(f"  snapshot {snap:03d}:")
        ps, z, box = spectra_of(sim_dir(args.sim, args.data_root) / f"snapdir_{snap:03d}", methods, args.prefix, args.nbins)
        if not ps:
            print(f"  snapshot {snap:03d}: no density grid of any estimator; skipped")
            failed.append(snap)
            continue
        note = None
        if len(ps) == 1 and len(methods) > 1:
            other = next(m for m in methods if m not in ps)
            print(f"    only {LABEL[next(iter(ps))]} grids here: that estimator against linear theory alone")
            note = f"no {LABEL[other]} grid for this snapshot"
        tag = args.prefix or "+".join(ps)               # the file names what it holds: a --method or --prefix rerun
        base = out_dir / f"pk_{tag}_z{z:.2f}"           # used to overwrite the default file (suffixes APPENDED)
        base.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"redshift": z, "box_mpc": box, "k_linear": np.geomspace(1e-3, 10.0, 400),
                  "prefix": args.prefix or "", "methods": np.array(list(ps))}
        arrays["P_linear"] = spectra.linear_pk(arrays["k_linear"], z)
        for m, p in ps.items():
            for key in ("k", "P", "modes", "edges"):
                arrays[f"{m}_{key}"] = p[key]
            arrays[f"{m}_k_nyquist"] = p["k_nyquist"]
            if ref is not None and m in ref[0] and same_edges(ref[0][m]["edges"], p["edges"]):
                arrays[f"{m}_ref_P"] = ref[0][m]["P"]          # what the same-realisation panel divides by
        if ref is not None:
            arrays["ref_snapshot"], arrays["ref_redshift"] = args.ref_snap, ref[1]
        np.savez_compressed(f"{base}.npz", **arrays)
        title = f"{args.sim}  z = {z:.2f}  P(k): {' and '.join(LABEL[m] for m in ps)} vs linear theory"
        plot_pk(ps, z, box, title, Path(f"{base}.png"), labels=labels, note=note, ref=ref)
        written.append(base)
    print(f"\n{len(written)} snapshot(s) under {out_dir}" + (f"; failed: {failed}" if failed else ""))
    raise SystemExit(1 if failed or not written else 0)


if __name__ == "__main__":
    main()
