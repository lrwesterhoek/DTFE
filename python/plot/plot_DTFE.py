"""Slice-map plots of the standard-DTFE field family (density, velocity, divergence, shear)
across a snapshot series.

All file naming, grid geometry, units and metadata come from dtfelib.FieldSet; density is
plotted MEAN-NORMALIZED (rho/rho_bar, identical colorbars for both estimators;
FieldSet auto-detects the on-disk convention). --method defaults to
'dtfe' (this is the standard-DTFE plot script); pass --method ps to plot the PS-DTFE
versions of the same fields.

    python3 plot/plot_DTFE.py                          # TNG50-4-Dark, full snapshot series
    python3 plot/plot_DTFE.py --snap 99                # single snapshot
    python3 plot/plot_DTFE.py --sim TNG50-3-Dark --snaps 0 17 50 99 --smooth 1
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import config
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter

from dtfelib import make_parser, FieldSet
from dtfelib.fields import extract_2d_slice as extract_slice, extract_velocity_slice

OUTPUT_DIR = Path(config.LOCAL_FIGURES_ROOT) / "dtfe"

AXIS_UNITS = "Mpc"
VELOCITY_UNITS = "km/s"

# density colour range in rho/rho_bar, shared with plot_PS_DTFE.py so DTFE and PS-DTFE
# panels are directly comparable side by side (set either to None for the slice's own range)
DENSITY_VMIN = 1e-1
DENSITY_VMAX = 1e4

# historical TNG50 output series; snapshots without fields on disk are skipped
SNAPSHOTS = [99]

SLICE_PLANES_TO_PLOT = [0, 1, 2]

PROCESS_DENSITY = True
PROCESS_VELOCITY = True
PROCESS_DIVERGENCE = True
PROCESS_SHEAR = True

VELOCITY_QUIVER_STEP = 8
DPI = 300

SLICE_PLANES = {
    0: {'name': 'yz_plane', 'axis_labels': ('Y', 'Z')},
    1: {'name': 'xz_plane', 'axis_labels': ('X', 'Z')},
    2: {'name': 'xy_plane', 'axis_labels': ('X', 'Y')}
}

def plot_density(density_field, slice_dim, box_size, redshift=None, save_path=None):
    dens_slice = extract_slice(density_field, slice_dim).T
    vmin = DENSITY_VMIN if DENSITY_VMIN is not None else max(np.min(dens_slice[dens_slice > 0]), 1e-6)
    vmax = DENSITY_VMAX if DENSITY_VMAX is not None else dens_slice.max()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        dens_slice, origin='lower', cmap='plasma',
        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        extent=[0, box_size, 0, box_size]
    )

    plane_info = SLICE_PLANES[slice_dim]
    title = "DTFE Density"
    if redshift is not None:
        title += f" (z={redshift:.2f})"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Density $\rho/\bar{\rho}$", fontsize=12)

    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_velocity(velocity_field, slice_dim, box_size, quiver_step,
                 redshift=None, save_path=None):
    X, Y, U, V = extract_velocity_slice(velocity_field, slice_dim)
    U, V = U.T, V.T

    ny, nx = U.shape
    pixel_size = box_size / nx
    X_mesh, Y_mesh = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    X_phys, Y_phys = X_mesh * pixel_size, Y_mesh * pixel_size

    X_q = X_phys[::quiver_step, ::quiver_step]
    Y_q = Y_phys[::quiver_step, ::quiver_step]
    U_q = U[::quiver_step, ::quiver_step]
    V_q = V[::quiver_step, ::quiver_step]

    mag = np.sqrt(U_q**2 + V_q**2)
    scale = np.percentile(mag, 95) * 0.5

    fig, ax = plt.subplots(figsize=(8, 7))
    Q = ax.quiver(
        X_q, Y_q, U_q, V_q, mag,
        angles='xy', scale_units='xy', scale=scale,
        pivot='middle', cmap='viridis', alpha=0.8, width=0.003
    )

    plane_info = SLICE_PLANES[slice_dim]
    title = "DTFE Velocity"
    if redshift is not None:
        title += f" (z={redshift:.2f})"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)

    cbar = fig.colorbar(Q, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Velocity [{VELOCITY_UNITS}]", fontsize=12)

    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_divergence(div_field, slice_dim, box_size, redshift=None, save_path=None):
    div_slice = extract_slice(div_field, slice_dim).T
    vmin, vmax = np.percentile(div_slice, [1, 99])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        div_slice, origin='lower', cmap='RdBu_r',
        extent=[0, box_size, 0, box_size],
        vmin=vmin, vmax=vmax
    )

    plane_info = SLICE_PLANES[slice_dim]
    title = "DTFE Velocity Divergence"
    if redshift is not None:
        title += f" (z={redshift:.2f})"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"∇·v [{VELOCITY_UNITS}/{AXIS_UNITS}]", fontsize=12)

    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_shear(shear_field, slice_dim, box_size, redshift=None, save_path=None):
    σ_xx = shear_field[..., 0]
    σ_xy = shear_field[..., 1]
    σ_xz = shear_field[..., 2]
    σ_yy = shear_field[..., 3]
    σ_yz = shear_field[..., 4]
    σ_zz = -(σ_xx + σ_yy)

    shear_mag = np.sqrt(
        σ_xx**2 + σ_yy**2 + σ_zz**2 +
        2*(σ_xy**2 + σ_xz**2 + σ_yz**2)
    )

    shear_slice = extract_slice(shear_mag, slice_dim).T
    vmin = max(np.min(shear_slice[shear_slice > 0]), 1e-6)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        shear_slice, origin='lower', cmap='plasma',
        norm=colors.LogNorm(vmin=vmin, vmax=shear_slice.max()),
        extent=[0, box_size, 0, box_size]
    )

    plane_info = SLICE_PLANES[slice_dim]
    title = "DTFE Velocity Shear"
    if redshift is not None:
        title += f" (z={redshift:.2f})"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"|σ| [{VELOCITY_UNITS}/{AXIS_UNITS}]", fontsize=12)

    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def process_snapshot(args, snap):
    snap_dir = args.data_root / args.sim / f"snapdir_{snap:03d}"

    try:
        fs = FieldSet(snap_dir, method=args.method, averaged=not args.raw, prefix=args.prefix)
    except FileNotFoundError:
        print(f"  skipping snapshot {snap:03d} (no {args.method} fields)")
        return False

    redshift = fs.meta.redshift
    box_size = fs.meta.box_mpc
    print(f"\nProcessing snapshot {snap:03d} (z={redshift:.2f})")

    fields = {}

    try:
        if PROCESS_DENSITY and fs.has('density'):
            dens = fs.density(units='mean')
            if args.smooth > 0:
                dens = gaussian_filter(dens, sigma=args.smooth, mode='wrap')
            fields['density'] = dens

        if PROCESS_VELOCITY and fs.has('velocity'):
            fields['velocity'] = fs.load('velocity')

        if PROCESS_DIVERGENCE and fs.has('divergence'):
            fields['divergence'] = fs.load('divergence')

        if PROCESS_SHEAR and fs.has('shear'):
            fields['shear'] = fs.load('shear')

    except FileNotFoundError:
        print(f"  skipping snapshot {snap:03d} (no {args.method} fields)")
        return False
    except Exception as e:
        print(f"  Error: Failed to load data for snapshot {snap:03d}: {e}")
        return False

    if not fields:
        print(f"  skipping snapshot {snap:03d} (no {fs.method} fields)")
        return False

    output_base = OUTPUT_DIR / f"snapshot_{snap:03d}_z{redshift:.2f}"

    for slice_dim in SLICE_PLANES_TO_PLOT:
        plane_name = SLICE_PLANES[slice_dim]['name']
        plane_dir = output_base / plane_name

        print(f"  Creating {plane_name} visualizations...")

        try:
            if 'density' in fields:
                save_path = plane_dir / f"density_{plane_name}_z{redshift:.2f}.png"
                plot_density(fields['density'], slice_dim, box_size, redshift, save_path)

            if 'velocity' in fields:
                save_path = plane_dir / f"velocity_{plane_name}_z{redshift:.2f}.png"
                plot_velocity(fields['velocity'], slice_dim, box_size,
                            VELOCITY_QUIVER_STEP, redshift, save_path)

            if 'divergence' in fields:
                save_path = plane_dir / f"divergence_{plane_name}_z{redshift:.2f}.png"
                plot_divergence(fields['divergence'], slice_dim, box_size, redshift, save_path)

            if 'shear' in fields:
                save_path = plane_dir / f"shear_{plane_name}_z{redshift:.2f}.png"
                plot_shear(fields['shear'], slice_dim, box_size, redshift, save_path)

        except Exception as e:
            print(f"  Error: Failed to create plots for {plane_name}: {e}")
            continue

    return True

def main():
    parser = make_parser("Slice-map plots of the standard-DTFE fields "
                         "(density, velocity, divergence, shear) across a snapshot series.")
    parser.set_defaults(method="dtfe", snap=None)
    parser.add_argument("--snaps", type=int, nargs="+", default=None,
                        help="snapshot numbers to process (default: --snap if given, "
                             f"else the full series {SNAPSHOTS})")
    args = parser.parse_args()

    if args.snaps is not None:
        snapshots = args.snaps
    elif args.snap is not None:
        snapshots = [args.snap]
    else:
        snapshots = SNAPSHOTS

    fields_to_process = []
    if PROCESS_DENSITY: fields_to_process.append("density")
    if PROCESS_VELOCITY: fields_to_process.append("velocity")
    if PROCESS_DIVERGENCE: fields_to_process.append("divergence")
    if PROCESS_SHEAR: fields_to_process.append("shear")

    print("Starting DTFE field visualization")
    print(f"Data directory: {args.data_root / args.sim}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing {len(snapshots)} snapshots (method={args.method})")
    print(f"Fields: {', '.join(fields_to_process)}")

    success_count = 0
    failed = []

    for snap in snapshots:
        if process_snapshot(args, snap):
            success_count += 1
        else:
            failed.append(snap)

    print(f"\nCompleted: {success_count}/{len(snapshots)} snapshots processed")

    if failed:
        print(f"Failed snapshots: {', '.join(f'{s:03d}' for s in failed)}")

    plots_per_snapshot = len(SLICE_PLANES_TO_PLOT) * len(fields_to_process)
    total_plots = success_count * plots_per_snapshot
    print(f"Total plots created: {total_plots}")
    print(f"Output saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
