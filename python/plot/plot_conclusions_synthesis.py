"""Three-panel conclusions synthesis figure (void shapes, tidal alignment, potential hills).

Aggregates the redshift-series stats tables written by plot_contour_filter.py,
plot_tidal_correlation.py and plot_phi_delta.py -- it loads no raw fields itself.
The standard --sim/--snap/--method flags are accepted for pipeline uniformity, but the
upstream tables are not namespaced by them; rerun the producer scripts to change inputs.
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import re
import sys

import numpy as np
import matplotlib.pyplot as plt

from dtfelib.figures import save_plot_to_multiple_paths
from dtfelib.cli import make_parser
import config
from dtfelib import figures as style

style.apply()

OUTPUT_DIR = Path(config.figures_path('synthesis'))
DPI = config.DPI

_FLOAT = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def _numeric_rows(path, min_cols):
    rows = []
    for line in Path(path).read_text().splitlines():
        toks = line.split()
        if not toks:
            continue
        try:
            float(toks[0])
        except ValueError:
            continue
        vals = [float(x) for x in _FLOAT.findall(line)]
        if len(vals) >= min_cols:
            rows.append(vals)
    return np.array(rows, dtype=float)


def load_shape():
    a = _numeric_rows(config.figures_path('void_analysis') +
                      '/void_shape_table.txt', 12)
    return a[:, 0], a[:, 7], a[:, 9], a[:, 3]


def load_alignment():
    a = _numeric_rows(config.figures_path('tidal') +
                      '/tidal_alignment_stats.txt', 8)
    return a[:, 0], a[:, 2], a[:, 3], a[:, 4]


def load_potential():
    a = _numeric_rows(config.figures_path('extrema') +
                      '/phi_delta_stats.txt', 13)
    return a[:, 0], a[:, 9], a[:, 11]


def _redshift_axis(ax):
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 3, 6, 11, 21])
    ax.set_xticklabels(['0', '1', '2', '5', '10', '20'])
    ax.invert_xaxis()
    ax.set_xlabel('Redshift $z$')


def _panel_tag(ax, text):
    ax.text(0.04, 0.94, text, transform=ax.transAxes, va='top', ha='left',
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.25',
            facecolor='white', edgecolor='0.6', alpha=0.85))


def main():
    parser = make_parser(
        "Three-panel conclusions synthesis figure. Aggregates the stats tables written by "
        "plot_contour_filter.py, plot_tidal_correlation.py and plot_phi_delta.py; the standard "
        "--sim/--snap/--method flags are accepted for uniformity but do not select data here.")
    parser.parse_args()

    try:
        zs, e, p, ca = load_shape()
        za, align, nlo, nhi = load_alignment()
        zp, dmax, rsig = load_potential()
    except FileNotFoundError as err:
        print(f"Missing upstream stats table: {err}")
        print("Run plot_contour_filter.py, plot_tidal_correlation.py and "
              "plot_phi_delta.py first to generate them.")
        sys.exit(1)

    xs = 1.0 + zs
    xa = 1.0 + za
    xp = 1.0 + zp

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.7))

    cE, cP, cR = 'darkblue', 'firebrick', 'seagreen'
    axA.plot(xs, e, 'o-', color=cE, lw=2, ms=4, label=r'Ellipticity $\langle e\rangle$')
    axA.plot(xs, p, 's-', color=cP, lw=2, ms=4, label=r'Prolateness $\langle p\rangle$')
    axA.set_ylabel('Shape parameter')
    axA.set_ylim(0, 0.27)
    axAr = axA.twinx()
    axAr.plot(xs, ca, '^--', color=cR, lw=1.8, ms=4,
              label=r'Axis ratio $\langle c/a\rangle$')
    axAr.set_ylabel(r'$\langle c/a\rangle$', color=cR)
    axAr.tick_params(axis='y', colors=cR)
    _redshift_axis(axA)
    axA.grid(alpha=0.25, linestyle=':')
    _panel_tag(axA, '(a) Void shapes')

    if config.SHOW_NULL_BANDS:
        axB.fill_between(xa, nlo, nhi, color='0.55', alpha=0.35, lw=0,
                         label='95% pairing-shuffle null')
    axB.plot(xa, align, 'o-', color='steelblue', lw=2.5, ms=4,
             label=r'$\langle|\cos\theta|\rangle$, major axis vs tide')
    axB.axhline(0.5, color='black', ls='--', alpha=0.5, lw=1.2)
    axB.text(1.05, 0.505, 'isotropic', fontsize=8, color='0.3', va='bottom')
    axB.set_ylabel(r'Void--tide alignment $\langle|\cos\theta|\rangle$')
    axB.set_ylim(0.30, 0.74)
    _redshift_axis(axB)
    axB.grid(alpha=0.25, linestyle=':')
    _panel_tag(axB, '(b) Tidal alignment')

    cD, cS = 'darkblue', 'darkorange'
    axC.plot(xp, dmax, 'o-', color=cD, lw=2.5, ms=4,
             label=r'$\langle\delta\rangle$ at $\phi$ maxima')
    axC.axhline(0.0, color='black', ls='--', alpha=0.5, lw=1.2)
    axC.set_ylabel(r'$\langle\delta\rangle$ at potential maxima', color=cD)
    axC.tick_params(axis='y', colors=cD)
    axC.set_ylim(-1.0, 0.05)
    axCr = axC.twinx()
    axCr.plot(xp, rsig, '^--', color=cS, lw=1.8, ms=4,
              label=r'Underdense radius $r_{\mathrm{sig}}$')
    axCr.set_ylabel(r'$r_{\mathrm{sig}}$ [Mpc]', color=cS)
    axCr.tick_params(axis='y', colors=cS)
    _redshift_axis(axC)
    axC.grid(alpha=0.25, linestyle=':')
    _panel_tag(axC, '(c) Potential hills')

    handles, labels = [], []
    for ax in (axA, axAr, axB, axC, axCr):
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc='lower center', ncol=4, framealpha=0.9,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_plot_to_multiple_paths(fig, OUTPUT_DIR / 'conclusions_synthesis.png',
                                dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Wrote', OUTPUT_DIR / 'conclusions_synthesis.png')


if __name__ == '__main__':
    main()
