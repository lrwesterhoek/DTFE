"""Slice-map plots of every raw field the DTFE / PS-DTFE binaries export.

Method-, snapshot- and simulation-agnostic: all file naming, grid geometry, units and
metadata come from dtfelib.FieldSet. Density is plotted MEAN-NORMALIZED (rho/rho_bar)
for both estimators; FieldSet auto-detects the on-disk convention (old PS files are physical).

    python3 plot/plot_PS_DTFE.py                       # TNG50-4-Dark snap 99, auto method
    python3 plot/plot_PS_DTFE.py --method dtfe --smooth 1
    python3 plot/plot_PS_DTFE.py --sim TNG50-3-Dark --snap 50
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import config
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter

from dtfelib import make_fieldset

# which field families to plot (PS-only ones are skipped automatically under --method dtfe)
PROCESS_DENSITY = True
PROCESS_STREAMS = True
PROCESS_DISPERSION = True   # velocity dispersion trace + tensor magnitude
PROCESS_VELOCITY = True     # velocity magnitude |v|
PROCESS_WEB = True          # web volume-fraction stats + eigenvalue maps (no classification maps)

SLICE_PLANES_TO_PLOT = [0, 1, 2]
SLICE_PLANES = {
    0: {'name': 'yz_plane', 'axis_labels': ('Y', 'Z')},
    1: {'name': 'xz_plane', 'axis_labels': ('X', 'Z')},
    2: {'name': 'xy_plane', 'axis_labels': ('X', 'Y')}
}
AXIS_UNITS = "Mpc"
DPI = 300
FIGURE_ROOT = Path(config.LOCAL_FIGURES_ROOT)

# density colour range in rho/rho_bar, shared with plot_DTFE.py so DTFE and PS-DTFE
# panels are directly comparable side by side (set either to None for the slice's own range)
DENSITY_VMIN = 1e-1
DENSITY_VMAX = 1e4

_SMOOTH = 0.0   # set from --smooth in main()

def extract_slice(field, slice_dim=2):
    idx = field.shape[slice_dim] // 2
    slices = [slice(None)] * 3
    slices[slice_dim] = idx
    return field[tuple(slices)]

def smooth(field):
    """Plot-time Gaussian smoothing (--smooth, grid cells); identity when 0."""
    if _SMOOTH <= 0:
        return field
    return gaussian_filter(field, sigma=_SMOOTH, mode='wrap')

def smooth_components(field):
    """Per-component smoothing for (N,N,N,c) fields (before norms/magnitudes)."""
    if _SMOOTH <= 0:
        return field
    out = np.empty_like(field)
    for c in range(field.shape[-1]):
        out[..., c] = gaussian_filter(field[..., c], sigma=_SMOOTH, mode='wrap')
    return out

def plot_density(density_field, slice_dim, box_size, redshift=None, save_path=None, label="PS-DTFE"):
    dens_slice = extract_slice(density_field, slice_dim).T

    positive = dens_slice[dens_slice > 0]
    if positive.size == 0:
        print(f"    Warning: No positive density values in slice (dim={slice_dim})")
        return
    vmin = DENSITY_VMIN if DENSITY_VMIN is not None else max(np.min(positive), 1e-6)
    vmax = DENSITY_VMAX if DENSITY_VMAX is not None else dens_slice.max()

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(cmap(0))
    cmap.set_under(cmap(0))
    im = ax.imshow(
        dens_slice, origin='lower', cmap=cmap,
        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        extent=[0, box_size, 0, box_size]
    )

    plane_info = SLICE_PLANES[slice_dim]
    title = f"{label} Density"
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


def plot_streams(stream_field, slice_dim, box_size, redshift=None, save_path=None, label="PS-DTFE"):
    stream_slice = extract_slice(stream_field, slice_dim).T

    max_streams = int(stream_slice.max())
    if max_streams == 0:
        print(f"    Warning: No streams in slice (dim={slice_dim})")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    if max_streams <= 1:
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(cmap(0))
        im = ax.imshow(
            stream_slice, origin='lower', cmap=cmap,
            extent=[0, box_size, 0, box_size],
            vmin=0, vmax=1
        )
    else:
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(cmap(0))
        cmap.set_under(cmap(0))
        im = ax.imshow(
            stream_slice, origin='lower', cmap=cmap,
            norm=colors.LogNorm(vmin=0.5, vmax=max(max_streams, 2)),
            extent=[0, box_size, 0, box_size]
        )

    plane_info = SLICE_PLANES[slice_dim]
    title = f"Stream Count ({label})"
    if redshift is not None:
        title += f" (z={redshift:.2f})"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of streams", fontsize=12)

    nonzero = stream_slice[stream_slice > 0]
    multi = stream_slice[stream_slice > 1]
    total = stream_slice.size
    ax.text(
        0.02, 0.98,
        f"max={max_streams}, multi-stream={100*multi.size/total:.1f}%",
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_density_comparison(density_field, stream_field, slice_dim, box_size,
                            redshift=None, save_path=None, label="PS-DTFE"):
    dens_slice = extract_slice(density_field, slice_dim).T
    stream_slice = extract_slice(stream_field, slice_dim).T

    positive = dens_slice[dens_slice > 0]
    if positive.size == 0:
        return
    vmin_dens = DENSITY_VMIN if DENSITY_VMIN is not None else max(np.min(positive), 1e-6)
    vmax_dens = DENSITY_VMAX if DENSITY_VMAX is not None else dens_slice.max()
    max_streams = int(stream_slice.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plane_info = SLICE_PLANES[slice_dim]

    cmap_dens = plt.cm.plasma.copy()
    cmap_dens.set_bad(cmap_dens(0))
    cmap_dens.set_under(cmap_dens(0))
    im1 = ax1.imshow(
        dens_slice, origin='lower', cmap=cmap_dens,
        norm=colors.LogNorm(vmin=vmin_dens, vmax=vmax_dens),
        extent=[0, box_size, 0, box_size]
    )
    ax1.set_title(f"{label} Density", fontsize=14)
    ax1.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax1.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
    ax1.set_aspect('equal')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label=r"Density $\rho/\bar{\rho}$")

    if max_streams > 1:
        cmap_str = plt.cm.inferno.copy()
        cmap_str.set_bad(cmap_str(0))
        cmap_str.set_under(cmap_str(0))
        im2 = ax2.imshow(
            stream_slice, origin='lower', cmap=cmap_str,
            norm=colors.LogNorm(vmin=0.5, vmax=max(max_streams, 2)),
            extent=[0, box_size, 0, box_size]
        )
    else:
        cmap_str = plt.cm.viridis.copy()
        cmap_str.set_bad(cmap_str(0))
        im2 = ax2.imshow(
            stream_slice, origin='lower', cmap=cmap_str,
            extent=[0, box_size, 0, box_size],
            vmin=0, vmax=max(max_streams, 1)
        )
    ax2.set_title("Stream Count", fontsize=14)
    ax2.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax2.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
    ax2.set_aspect('equal')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Number of streams")

    suptitle = f"PS-DTFE: {plane_info['name']}"
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

def plot_scalar_log(field, slice_dim, box_size, title, cbar_label, redshift=None, save_path=None):
    """Log-scale slice plot for positive scalar fields (dispersion trace, tensor magnitude)."""
    fslice = extract_slice(field, slice_dim).T
    positive = fslice[fslice > 0]
    if positive.size == 0:
        print(f"    Warning: no positive values in slice (dim={slice_dim})")
        return
    vmin = np.percentile(positive, 1)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    cmap = plt.get_cmap('magma')
    im = ax.imshow(
        fslice, origin='lower', cmap=cmap,
        norm=colors.LogNorm(vmin=vmin, vmax=fslice.max()),
        extent=[0, box_size, 0, box_size]
    )
    plane_info = SLICE_PLANES[slice_dim]
    full_title = f"{title} (z={redshift:.2f})" if redshift is not None else title
    ax.set_title(full_title, fontsize=16)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=12)
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)


WEB_CLASS_NAMES = ['void', 'wall', 'filament', 'node']
WEB_CLASS_COLORS = ['#1a1a40', '#3b7ea1', '#e8843c', '#f7e463']   # void/wall/filament/node


def plot_web_eigenvalues(eig_field, slice_dim, box_size, title, redshift=None, save_path=None):
    """Triptych of the (descending-sorted) eigenvalue maps, diverging colormap, symlog scale."""
    eslice = extract_slice(eig_field, slice_dim)          # (N, N, 3)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    plane_info = SLICE_PLANES[slice_dim]
    for i, ax in enumerate(axes):
        lam = eslice[..., i].T
        vmax = np.percentile(np.abs(lam), 99.5)
        if vmax <= 0: vmax = 1.0
        norm = colors.SymLogNorm(linthresh=vmax / 1e3, vmin=-vmax, vmax=vmax, base=10)
        im = ax.imshow(lam, origin='lower', cmap='RdBu_r', norm=norm,
                       extent=[0, box_size, 0, box_size])
        ax.set_title(rf"$\lambda_{i+1}$", fontsize=14)
        ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]", fontsize=11)
        if i == 0:
            ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    full_title = f"{title} (z={redshift:.2f})" if redshift is not None else title
    fig.suptitle(full_title, fontsize=16)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)


def main():
    global _SMOOTH
    fs, args = make_fieldset("Slice-map plots of all DTFE/PS-DTFE raw fields.")
    _SMOOTH = args.smooth
    label = "PS-DTFE" if fs.method == "ps" else "DTFE"
    z, box = fs.meta.redshift, fs.meta.box_mpc
    out_base = FIGURE_ROOT / "fields" / args.sim / f"snap{args.snap:03d}"

    print(fs)
    print(f"plot smoothing: {_SMOOTH} cells | figures -> {out_base}")

    fields = {}

    if PROCESS_DENSITY and fs.has('density'):
        dens = smooth(fs.density(units='mean'))
        fields['density'] = dens
        print(f"  density [rho/rho_bar]: min={dens.min():.4e} max={dens.max():.4e} mean={dens.mean():.4e}")

    if PROCESS_STREAMS and fs.has('streams'):
        streams = smooth(fs.load('streams'))
        fields['streams'] = streams
        print(f"  streams: max={streams.max():.0f} mean={streams.mean():.2f} "
              f"multi-stream={100*np.mean(streams > 1):.2f}%")

    if PROCESS_DISPERSION and fs.has('dispersion'):
        fields['velDisp'] = smooth(fs.load('dispersion'))
    if PROCESS_DISPERSION and fs.has('dispersion_tensor'):
        tens = smooth_components(fs.load('dispersion_tensor'))
        xx, xy, xz, yy, yz, zz = [tens[..., i] for i in range(6)]
        fields['velDispMag'] = np.sqrt(xx**2 + yy**2 + zz**2 + 2*(xy**2 + xz**2 + yz**2))
        del tens

    if PROCESS_VELOCITY and fs.has('velocity'):
        vel = smooth_components(fs.load('velocity'))
        fields['velMag'] = np.sqrt((vel ** 2).sum(axis=-1))
        del vel
        print(f"  |v|: mean={fields['velMag'].mean():.4g} km/s")

    if PROCESS_WEB:
        for name, lab in [('tweb', 'T-web'), ('vweb', 'V-web')]:
            if fs.has(name):
                web = fs.load(name)
                fr = [100.0 * np.mean(np.rint(web) == k) for k in range(4)]
                print(f"  {lab} volume fractions: " + "  ".join(
                    f"{n}={f:.1f}%" for n, f in zip(['void', 'wall', 'filament', 'node'], fr)))
        for name, key in [('tweb_eigenvalues', 'twebEig'), ('vweb_eigenvalues', 'vwebEig')]:
            if fs.has(name):
                fields[key] = smooth_components(fs.load(name))

    if not fields:
        print("No fields found for this method/snapshot. Nothing to plot.")
        return

    for slice_dim in SLICE_PLANES_TO_PLOT:
        plane = SLICE_PLANES[slice_dim]['name']
        plane_dir = out_base / plane
        print(f"  {plane} ...")

        def path(stub):
            return plane_dir / f"{fs.method}_{stub}_{plane}_z{z:.2f}.png"

        if 'density' in fields:
            plot_density(fields['density'], slice_dim, box, z, path('density'), label=label)
        if 'streams' in fields:
            plot_streams(fields['streams'], slice_dim, box, z, path('streams'), label=label)
        if 'velDisp' in fields:
            plot_scalar_log(fields['velDisp'], slice_dim, box,
                            f"{label} Velocity Dispersion", r"Tr $\sigma^2$  [(km/s)$^2$]",
                            z, path('velDisp'))
        if 'velDispMag' in fields:
            plot_scalar_log(fields['velDispMag'], slice_dim, box,
                            f"{label} Dispersion Tensor", r"$|\sigma^2|$  [(km/s)$^2$]",
                            z, path('velDispTensor'))
        if 'velMag' in fields:
            plot_scalar_log(fields['velMag'], slice_dim, box,
                            f"{label} Velocity Magnitude", r"$|v|$  [km/s]",
                            z, path('velMag'))
        if 'twebEig' in fields:
            plot_web_eigenvalues(fields['twebEig'], slice_dim, box,
                                 f"{label} T-web Eigenvalues (tidal)", z, path('twebEig'))
        if 'vwebEig' in fields:
            plot_web_eigenvalues(fields['vwebEig'], slice_dim, box,
                                 f"{label} V-web Eigenvalues", z, path('vwebEig'))
        if 'density' in fields and 'streams' in fields:
            plot_density_comparison(fields['density'], fields['streams'], slice_dim,
                                    box, z, path('comparison'), label=label)

    print(f"Done -> {out_base}")


if __name__ == "__main__":
    main()
