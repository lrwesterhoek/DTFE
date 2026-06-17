
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

BASE_DATA_DIR = "output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/cosmic_web"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7
AXIS_UNITS = "Mpc"

SNAPSHOT_TO_REDSHIFT = {
    '099': 0.00
}

SLICE_PLANES_TO_PLOT = [0, 1, 2]

PROCESS_TWEB = True
PROCESS_VWEB = True

AVERAGED = True

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

def load_binary_field(binary_file, field_shape, num_components=1, dtype=np.float32):
    data = np.fromfile(binary_file, dtype=dtype)
    expected_size = np.prod(field_shape) * num_components

    if data.size != expected_size:
        total_elements = data.size / num_components
        cube_root = round(total_elements ** (1/3))

        if cube_root**3 * num_components == data.size:
            field_shape = (cube_root, cube_root, cube_root)
        else:
            raise ValueError(
                f"File {binary_file} has {data.size} elements, "
                f"but expected {expected_size} for shape {field_shape}"
            )

    if num_components == 1:
        return data.reshape(field_shape)
    else:
        return data.reshape(field_shape + (num_components,))


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

def process_snapshot(snapshot, redshift):
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")

    snapshot_dir = Path(BASE_DATA_DIR) / f"snapdir_{snapshot}"

    if not snapshot_dir.exists():
        print(f"  Warning: Directory not found: {snapshot_dir}")
        return False

    field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)

    prefix = "a_vel" if AVERAGED else "vel"

    web_configs = {
        'T-web': {
            'class_file': snapshot_dir / f'cosmic_web_a.{prefix}Tweb',
            'eig_file': snapshot_dir / f'cosmic_web_a.{prefix}TwebEig',
        },
        'V-web': {
            'class_file': snapshot_dir / f'cosmic_web_a.{prefix}Vweb',
            'eig_file': snapshot_dir / f'cosmic_web_a.{prefix}VwebEig',
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
            class_field = load_binary_field(str(cfg['class_file']), field_shape, num_components=1)
            print(f"  Loaded {web_type} classification: {cfg['class_file']}")
            loaded[web_type] = class_field
        except FileNotFoundError:
            print(f"  Warning: {web_type} classification file not found: {cfg['class_file']}")
            continue
        except Exception as e:
            print(f"  Error loading {web_type} classification: {e}")
            continue

        try:
            eig_field = load_binary_field(str(cfg['eig_file']), field_shape, num_components=3)
            print(f"  Loaded {web_type} eigenvalues: {cfg['eig_file']}")
        except FileNotFoundError:
            print(f"  Warning: {web_type} eigenvalue file not found: {cfg['eig_file']}")
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
            plot_classification(class_field, slice_dim, BOX_SIZE, web_type, redshift, save_path)

            if eig_field is not None:
                save_path = plane_dir / f"{short}_eigenvalues_{plane_name}_z{redshift:.2f}.png"
                plot_eigenvalues(eig_field, slice_dim, BOX_SIZE, web_type, redshift, save_path)

    if 'T-web' in loaded and 'V-web' in loaded:
        output_base = Path(OUTPUT_DIR) / f"snapshot_{snapshot}_z{redshift:.2f}"
        for slice_dim in SLICE_PLANES_TO_PLOT:
            plane_name = SLICE_PLANES[slice_dim]['name']
            plane_dir = output_base / plane_name

            print(f"  Creating T-web vs V-web residual {plane_name}...")
            save_path = plane_dir / f"tweb_vweb_residual_{plane_name}_z{redshift:.2f}.png"
            plot_tweb_vweb_residual(loaded['T-web'], loaded['V-web'],
                                    slice_dim, BOX_SIZE, redshift, save_path)

    return True


def main():
    fields = []
    if PROCESS_TWEB: fields.append("T-web")
    if PROCESS_VWEB: fields.append("V-web")

    print("Starting Cosmic Web visualization")
    print(f"Data directory: {BASE_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Averaged: {AVERAGED}")
    print(f"Fields: {', '.join(fields)}")

    success_count = 0
    failed = []

    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        if process_snapshot(snapshot, redshift):
            success_count += 1
        else:
            failed.append((snapshot, redshift))

    print(f"\nCompleted: {success_count}/{len(SNAPSHOT_TO_REDSHIFT)} snapshots processed")

    if failed:
        print(f"Failed snapshots: {', '.join(f'{s} (z={z:.2f})' for s, z in failed)}")

    print(f"Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
