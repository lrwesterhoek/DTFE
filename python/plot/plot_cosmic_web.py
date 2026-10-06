"""Cosmic-web classification and eigenvalue plots (T-web / V-web / residual).

Method-, snapshot- and simulation-agnostic: all file naming, grid geometry and
metadata come from dtfelib.FieldSet.

    python3 plot/plot_cosmic_web.py                       # TNG50-4-Dark snap 99, auto method
    python3 plot/plot_cosmic_web.py --method dtfe
    python3 plot/plot_cosmic_web.py --sim TNG50-3-Dark --snap 50
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import config
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dtfelib import make_parser, snapdir, FieldSet
from dtfelib import figures as style

style.apply()                                   # the house style (serif, SHOW_TITLES, DPI): survey item 11, 2026-10-05

FIGURE_ROOT = Path(config.LOCAL_FIGURES_ROOT)
OUTPUT_DIR = FIGURE_ROOT / "cosmic_web"

SLICE_PLANES_TO_PLOT = [0, 1, 2]

PROCESS_TWEB = True
PROCESS_VWEB = True

SLICE_PLANES = config.SLICE_PLANES              # (the maps' axis labels, DPI and units are config's; the web colours figures')
WEB_LABELS = dict(enumerate(style.WEB_NAMES))


def _title(text, redshift):
    return f"{text} (z={redshift:.2f})" if redshift is not None else text


def extract_slice(field, slice_dim=2):
    idx = field.shape[slice_dim] // 2
    slices = [slice(None)] * 3
    slices[slice_dim] = idx
    return field[tuple(slices)]

def plot_classification(class_field, slice_dim, box_size, web_type,
                        redshift=None, save_path=None):
    class_slice = np.clip(np.rint(extract_slice(class_field, slice_dim).T).astype(int), 0, 3)
    cmap, norm = style.web_cmap_norm()
    unique, counts = np.unique(class_slice, return_counts=True)
    frac_text = "  ".join(f"{WEB_LABELS.get(u, '?')}: {c / class_slice.size * 100:.1f}%" for u, c in zip(unique, counts))
    style.slice_map(class_slice, box_size, slice_dim, cmap=cmap, norm=norm, interpolation='nearest',
                    cbar_ticks=[0, 1, 2, 3], cbar_ticklabels=list(style.WEB_NAMES),
                    title=_title(f"{web_type} Classification", redshift), path=save_path, footnote=frac_text)


def plot_eigenvalues(eig_field, slice_dim, box_size, web_type,
                     redshift=None, save_path=None):
    """Triptych of the eigenvalue maps: the house rule for a signed field (figures.norm_signed_log at the
    field's linthresh), the same as plot_PS_DTFE's triptych -- it was a linear (1, 99) percentile pair here."""
    panels = []
    for i, label in enumerate((r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$')):
        eig_slice = extract_slice(eig_field[..., i], slice_dim).T
        panels.append(dict(data=eig_slice, cmap=style.MAP_CMAPS['eigenvalue'], title=label, label=f"{web_type} {label}",
                           norm=style.norm_signed_log(data=eig_slice, field='eigenvalue')))   # the bar names the panel: titles are off
    style.slice_row(panels, box_size, slice_dim, path=save_path, figsize=(20, 6),
                    suptitle=_title(f"{web_type} Eigenvalues", redshift))


def plot_eigenvalue_histogram(eig_field, web_type, redshift=None, save_path=None):
    eig_labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (ax, label) in enumerate(zip(axes, eig_labels)):
        vals = eig_field[..., i].ravel()
        vmin, vmax = np.percentile(vals, [0.5, 99.5])
        bins = np.linspace(vmin, vmax, 200)
        ax.hist(vals, bins=bins, color='steelblue', alpha=0.7, density=True)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
        pos_frac = np.mean(vals > 0) * 100
        ax.text(0.95, 0.95, f"{pos_frac:.1f}% > 0", transform=ax.transAxes, ha='right', va='top')
        ax.set_xlabel(label)
        ax.set_ylabel('PDF' if i == 0 else '')
        style.set_title(ax, label)
    style.set_suptitle(fig, _title(f"{web_type} Eigenvalue Distributions", redshift), y=1.02)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)


def plot_tweb_vweb_residual(tweb_class, vweb_class, slice_dim, box_size,
                            redshift=None, save_path=None):
    tweb_slice = np.rint(extract_slice(tweb_class, slice_dim)).astype(int).T
    vweb_slice = np.rint(extract_slice(vweb_class, slice_dim)).astype(int).T
    residual = tweb_slice - vweb_slice
    web_cmap, web_norm = style.web_cmap_norm()
    res_cmap, res_norm = style.residual_cmap_norm()
    web = dict(cmap=web_cmap, norm=web_norm, interpolation='nearest', cbar_ticks=[0, 1, 2, 3], cbar_ticklabels=list(style.WEB_NAMES))
    panels = [dict(data=tweb_slice, title="T-web", label="T-web class", **web), dict(data=vweb_slice, title="V-web", label="V-web class", **web),
              dict(data=residual, cmap=res_cmap, norm=res_norm, interpolation='nearest', title="T-web − V-web", label="T-web − V-web",
                   cbar_ticks=[-3, -2, -1, 0, 1, 2, 3], cbar_ticklabels=['-3', '-2', '-1', '0', '+1', '+2', '+3'])]
    agree_frac = np.mean(residual == 0) * 100
    mean_abs = np.mean(np.abs(residual))
    style.slice_row(panels, box_size, slice_dim, path=save_path, figsize=(22, 6),
                    suptitle=_title("T-web vs V-web Comparison", redshift),
                    footnote=f"Agreement: {agree_frac:.1f}%   |  Mean |residual|: {mean_abs:.2f}")

def process_snapshot(fs, snapshot):
    redshift = fs.meta.redshift
    box_size = fs.meta.box_mpc

    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")

    web_configs = {
        'T-web': {
            'class_field': 'tweb',
            'eig_field': 'tweb_eigenvalues',
        },
        'V-web': {
            'class_field': 'vweb',
            'eig_field': 'vweb_eigenvalues',
        },
    }

    loaded = {}

    for web_type, cfg in web_configs.items():
        if web_type == 'T-web' and not PROCESS_TWEB:
            continue
        if web_type == 'V-web' and not PROCESS_VWEB:
            continue

        short = web_type.replace('-', '').lower()

        try:
            if not fs.has(cfg['class_field']):
                print(f"  Warning: {web_type} classification field not found: "
                      f"{cfg['class_field']} (method '{fs.method}')")
                continue
            class_field = fs.load(cfg['class_field'])
            print(f"  Loaded {web_type} classification: {cfg['class_field']}")
            loaded[web_type] = class_field
        except Exception as e:
            print(f"  Error loading {web_type} classification: {e}")
            continue

        try:
            if fs.has(cfg['eig_field']):
                eig_field = fs.load(cfg['eig_field'])
                print(f"  Loaded {web_type} eigenvalues: {cfg['eig_field']}")
            else:
                print(f"  Warning: {web_type} eigenvalue field not found: "
                      f"{cfg['eig_field']} (method '{fs.method}')")
                eig_field = None
        except Exception as e:
            print(f"  Error loading {web_type} eigenvalues: {e}")
            eig_field = None

        output_base = Path(OUTPUT_DIR) / f"snapshot_{snapshot}_z{redshift:.2f}"

        if eig_field is not None:
            save_path = output_base / f"{short}_eigenvalue_hist_z{redshift:.2f}.png"
            print(f"  Creating {web_type} eigenvalue histogram...")
            plot_eigenvalue_histogram(eig_field, web_type, redshift, save_path)

        for slice_dim in SLICE_PLANES_TO_PLOT:
            plane_name = SLICE_PLANES[slice_dim]['name']
            plane_dir = output_base / plane_name

            print(f"  Creating {web_type} {plane_name} visualizations...")

            save_path = plane_dir / f"{short}_classification_{plane_name}_z{redshift:.2f}.png"
            plot_classification(class_field, slice_dim, box_size, web_type, redshift, save_path)

            if eig_field is not None:
                save_path = plane_dir / f"{short}_eigenvalues_{plane_name}_z{redshift:.2f}.png"
                plot_eigenvalues(eig_field, slice_dim, box_size, web_type, redshift, save_path)

    if 'T-web' in loaded and 'V-web' in loaded:
        output_base = Path(OUTPUT_DIR) / f"snapshot_{snapshot}_z{redshift:.2f}"
        for slice_dim in SLICE_PLANES_TO_PLOT:
            plane_name = SLICE_PLANES[slice_dim]['name']
            plane_dir = output_base / plane_name

            print(f"  Creating T-web vs V-web residual {plane_name}...")
            save_path = plane_dir / f"tweb_vweb_residual_{plane_name}_z{redshift:.2f}.png"
            plot_tweb_vweb_residual(loaded['T-web'], loaded['V-web'],
                                    slice_dim, box_size, redshift, save_path)

    return True


def main():
    parser = make_parser("Cosmic-web classification and eigenvalue plots (T-web/V-web).")
    args = parser.parse_args()

    snapshot = f"{args.snap:03d}"

    try:
        fs = FieldSet(snapdir(args), method=args.method, averaged=not args.raw, prefix=args.prefix)
    except FileNotFoundError as e:
        print(f"skipping snapshot {snapshot} (no {args.method} fields): {e}")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    fields = []
    if PROCESS_TWEB: fields.append("T-web")
    if PROCESS_VWEB: fields.append("V-web")

    print("Starting Cosmic Web visualization")
    print(fs)
    print(f"Data directory: {fs.snapdir}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Averaged: {fs.averaged}")
    print(f"Fields: {', '.join(fields)}")

    if process_snapshot(fs, snapshot):
        print(f"\nCompleted: snapshot {snapshot} processed")
    else:
        print(f"\nFailed snapshot: {snapshot} (z={fs.meta.redshift:.2f})")

    print(f"Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
