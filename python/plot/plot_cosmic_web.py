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
from matplotlib.colors import ListedColormap, BoundaryNorm

from dtfelib import make_parser, snapdir, FieldSet

FIGURE_ROOT = Path(config.LOCAL_FIGURES_ROOT)
OUTPUT_DIR = FIGURE_ROOT / "cosmic_web"

AXIS_UNITS = "Mpc"

SLICE_PLANES_TO_PLOT = [0, 1, 2]

PROCESS_TWEB = True
PROCESS_VWEB = True

DPI = 300

SLICE_PLANES = {
    0: {'name': 'yz_plane', 'axis_labels': ('Y', 'Z')},
    1: {'name': 'xz_plane', 'axis_labels': ('X', 'Z')},
    2: {'name': 'xy_plane', 'axis_labels': ('X', 'Y')}
}

WEB_LABELS = {0: 'Void', 1: 'Wall', 2: 'Filament', 3: 'Node'}
WEB_COLORS = ['#1a1a2e', '#e0c97f', '#d4563e', '#f5f5dc']
WEB_CMAP = ListedColormap(WEB_COLORS)
WEB_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], WEB_CMAP.N)


def extract_slice(field, slice_dim=2):
    idx = field.shape[slice_dim] // 2
    slices = [slice(None)] * 3
    slices[slice_dim] = idx
    return field[tuple(slices)]

def plot_classification(class_field, slice_dim, box_size, web_type,
                        redshift=None, save_path=None):
    class_slice = extract_slice(class_field, slice_dim).T

    class_slice = np.rint(class_slice).astype(int)
    class_slice = np.clip(class_slice, 0, 3)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        class_slice, origin='lower', cmap=WEB_CMAP, norm=WEB_NORM,
        extent=[0, box_size, 0, box_size], interpolation='nearest'
    )

    plane_info = SLICE_PLANES[slice_dim]
    title = f"{web_type} Classification"
    if redshift is not None:
        title += f" (z={redshift:.2f})"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Void', 'Wall', 'Filament', 'Node'])

    unique, counts = np.unique(class_slice, return_counts=True)
    total = class_slice.size
    frac_text = "  ".join(
        f"{WEB_LABELS.get(u, '?')}: {c/total*100:.1f}%"
        for u, c in zip(unique, counts)
    )
    ax.text(0.5, -0.12, frac_text, transform=ax.transAxes,
            ha='center', fontsize=9, style='italic')

    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_eigenvalues(eig_field, slice_dim, box_size, web_type,
                     redshift=None, save_path=None):
    eig_labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    plane_info = SLICE_PLANES[slice_dim]

    for i, (ax, label) in enumerate(zip(axes, eig_labels)):
        eig_slice = extract_slice(eig_field[..., i], slice_dim).T
        vmin, vmax = np.percentile(eig_slice, [1, 99])

        vlim = max(abs(vmin), abs(vmax))

        im = ax.imshow(
            eig_slice, origin='lower', cmap='RdBu_r',
            extent=[0, box_size, 0, box_size],
            vmin=-vlim, vmax=vlim
        )

        subtitle = f"{label}"
        ax.set_title(subtitle, fontsize=14)
        ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=11)
        if i == 0:
            ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=11)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect('equal')

    suptitle = f"{web_type} Eigenvalues"
    if redshift is not None:
        suptitle += f" (z={redshift:.2f})"
    fig.suptitle(suptitle, fontsize=16, y=1.02)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


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
        ax.text(0.95, 0.95, f"{pos_frac:.1f}% > 0",
                transform=ax.transAxes, ha='right', va='top', fontsize=10)

        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('PDF' if i == 0 else '', fontsize=12)
        ax.set_title(label, fontsize=13)

    suptitle = f"{web_type} Eigenvalue Distributions"
    if redshift is not None:
        suptitle += f" (z={redshift:.2f})"
    fig.suptitle(suptitle, fontsize=15, y=1.02)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_tweb_vweb_residual(tweb_class, vweb_class, slice_dim, box_size,
                            redshift=None, save_path=None):
    tweb_slice = np.rint(extract_slice(tweb_class, slice_dim)).astype(int).T
    vweb_slice = np.rint(extract_slice(vweb_class, slice_dim)).astype(int).T

    residual = tweb_slice - vweb_slice

    res_colors = [
        '#08306b',
        '#2171b5',
        '#6baed6',
        '#f0f0f0',
        '#fb6a4a',
        '#cb181d',
        '#67000d',
    ]
    res_cmap = ListedColormap(res_colors)
    res_norm = BoundaryNorm(np.arange(-3.5, 4.5, 1), res_cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    plane_info = SLICE_PLANES[slice_dim]
    extent = [0, box_size, 0, box_size]

    im0 = axes[0].imshow(tweb_slice, origin='lower', cmap=WEB_CMAP, norm=WEB_NORM,
                          extent=extent, interpolation='nearest')
    axes[0].set_title("T-web", fontsize=14)
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar0.ax.set_yticklabels(['Void', 'Wall', 'Filament', 'Node'])

    im1 = axes[1].imshow(vweb_slice, origin='lower', cmap=WEB_CMAP, norm=WEB_NORM,
                          extent=extent, interpolation='nearest')
    axes[1].set_title("V-web", fontsize=14)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar1.ax.set_yticklabels(['Void', 'Wall', 'Filament', 'Node'])

    im2 = axes[2].imshow(residual, origin='lower', cmap=res_cmap, norm=res_norm,
                          extent=extent, interpolation='nearest')
    axes[2].set_title("T-web − V-web", fontsize=14)
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04,
                          ticks=[-3, -2, -1, 0, 1, 2, 3])
    cbar2.ax.set_yticklabels(['-3', '-2', '-1', '0', '+1', '+2', '+3'])

    for ax in axes:
        ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=11)
        ax.set_aspect('equal')
    axes[0].set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=11)

    agree_frac = np.mean(residual == 0) * 100
    mean_abs = np.mean(np.abs(residual))
    stat_text = f"Agreement: {agree_frac:.1f}%   |  Mean |residual|: {mean_abs:.2f}"
    fig.text(0.5, -0.02, stat_text, ha='center', fontsize=11, style='italic')

    suptitle = "T-web vs V-web Comparison"
    if redshift is not None:
        suptitle += f" (z={redshift:.2f})"
    fig.suptitle(suptitle, fontsize=16, y=1.02)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

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
