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
from scipy.ndimage import gaussian_filter

from dtfelib import STREAM_TOL, make_fieldset
from dtfelib import pointeval
from dtfelib import figures as style
from dtfelib.fields import extract_2d_slice as extract_slice

# which field families to plot (PS-only ones are skipped automatically under --method dtfe)
PROCESS_DENSITY = True
PROCESS_STREAMS = True
PROCESS_DISPERSION = True   # velocity dispersion trace + tensor magnitude
PROCESS_VELOCITY = True     # velocity magnitude |v|
PROCESS_WEB = True          # web volume-fraction stats + eigenvalue maps (no classification maps)

SLICE_PLANES_TO_PLOT = [0, 1, 2]
PROCESS_POINTEVAL = True    # point-evaluated ('--sample-points') maps, when present

SLICE_PLANES = config.SLICE_PLANES              # (the maps' axis labels, DPI, units and the density range are config's)
FIGURE_ROOT = Path(config.LOCAL_FIGURES_ROOT)

_SMOOTH = 0.0   # set from --smooth in main()


def _title(text, redshift):
    return f"{text} (z={redshift:.2f})" if redshift is not None else text

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
    if not (dens_slice > 0).any():
        print(f"    Warning: No positive density values in slice (dim={slice_dim})")
        return
    norm = style.norm_log_range(dens_slice, config.DENSITY_MAP_RANGE)      # the fixed range, else the slice's own
    style.slice_map(dens_slice, box_size, slice_dim, cmap=style.MAP_CMAPS['density'], norm=norm, floor=True,
                    label=style.MAP_LABELS['density'], title=_title(f"{label} Density", redshift), path=save_path)


def plot_streams(stream_field, slice_dim, box_size, redshift=None, save_path=None, label="PS-DTFE"):
    stream_slice = extract_slice(stream_field, slice_dim).T
    # '.streams' is a float multiplicity, not an integer count (see dtfelib.STREAM_TOL): every comparison
    # carries a tolerance (figures.norm_streams), truncating with int() would read a 0.99999 cell as empty
    rule = style.norm_streams(stream_slice)
    if rule is None:
        print(f"    Warning: No streams in slice (dim={slice_dim})")
        return
    cmap, norm = rule
    max_streams = float(stream_slice.max())
    multi = stream_slice[stream_slice > 1 + STREAM_TOL]
    style.slice_map(stream_slice, box_size, slice_dim, cmap=cmap, norm=norm, floor=True,
                    label=style.MAP_LABELS['streams'], title=_title(f"Stream Count ({label})", redshift), path=save_path,
                    text=f"max={max_streams:.2f}, multi-stream={100 * multi.size / stream_slice.size:.1f}%")


def plot_density_comparison(density_field, stream_field, slice_dim, box_size,
                            redshift=None, save_path=None, label="PS-DTFE"):
    dens_slice = extract_slice(density_field, slice_dim).T
    stream_slice = extract_slice(stream_field, slice_dim).T
    if not (dens_slice > 0).any():
        return
    rule = style.norm_streams(stream_slice)
    cmap_str, norm_str = rule if rule is not None else (style.MAP_CMAPS['streams_single'], None)
    panels = [dict(data=dens_slice, cmap=style.MAP_CMAPS['density'], floor=True, title=f"{label} Density",
                   norm=style.norm_log_range(dens_slice, config.DENSITY_MAP_RANGE), label=style.MAP_LABELS['density']),
              dict(data=stream_slice, cmap=cmap_str, floor=True, title="Stream Count", label=style.MAP_LABELS['streams'],
                   norm=norm_str, vmin=0.0, vmax=max(float(stream_slice.max()), 1.0))]
    style.slice_row(panels, box_size, slice_dim, path=save_path, figsize=(16, 7),
                    suptitle=_title(f"PS-DTFE: {SLICE_PLANES[slice_dim]['name']}", redshift))

def plot_scalar_log(field, slice_dim, box_size, title, cbar_label, redshift=None, save_path=None):
    """Log-scale slice plot for positive scalar fields (dispersion trace, tensor magnitude, |v|)."""
    fslice = extract_slice(field, slice_dim).T
    positive = fslice[fslice > 0]
    if positive.size == 0:
        print(f"    Warning: no positive values in slice (dim={slice_dim})")
        return
    norm = style.norm_log_range(fslice, (float(np.percentile(positive, 1)), float(fslice.max())))
    style.slice_map(fslice, box_size, slice_dim, cmap=style.MAP_CMAPS['velDisp'], norm=norm, figsize=(10, 8.5),
                    label=cbar_label, title=_title(title, redshift), path=save_path)


WEB_CLASS_NAMES = ['void', 'wall', 'filament', 'node']      # (the web COLOURS are figures.WEB_COLORS: one definition)


def plot_web_eigenvalues(eig_field, slice_dim, box_size, title, redshift=None, save_path=None):
    """Triptych of the (descending-sorted) eigenvalue maps: the house rule for a signed field
    (figures.norm_signed_log at the field's linthresh), the same as plot_cosmic_web's triptych."""
    eslice = extract_slice(eig_field, slice_dim)          # (N, N, 3)
    panels = [dict(data=eslice[..., i].T, cmap=style.MAP_CMAPS['eigenvalue'], title=rf"$\lambda_{i+1}$",
                   label=f"{title} " + rf"$\lambda_{i+1}$",          # the bar names the panel: titles are off
                   norm=style.norm_signed_log(data=eslice[..., i].T, field='eigenvalue')) for i in range(3)]
    style.slice_row(panels, box_size, slice_dim, path=save_path, figsize=(21, 6.5), suptitle=_title(title, redshift))


# ---------------------------------------------------------------- point-evaluated maps
# The '--sample-points' maps live in dtfelib.pointeval (shared with plot_pointeval.py,
# which renders them for every snapshot of every simulation).

def main():
    global _SMOOTH
    fs, args = make_fieldset("Slice-map plots of all DTFE/PS-DTFE raw fields.",
                             extra=lambda p: p.add_argument(
                                 "--pointeval-only", action="store_true",
                                 help="render only the point-evaluated ('--sample-points') "
                                      "maps, skipping every grid load"))
    _SMOOTH = args.smooth
    label = "PS-DTFE" if fs.method == "ps" else "DTFE"
    z, box = fs.meta.redshift, fs.meta.box_mpc
    out_base = FIGURE_ROOT / "fields" / args.sim / f"snap{args.snap:03d}"

    print(fs)
    print(f"plot smoothing: {_SMOOTH} cells | figures -> {out_base}")

    if PROCESS_POINTEVAL and fs.method == "ps":
        pointeval.render_fields(fs, out_base, label=label)     # BEFORE the house style: plot_pointeval.py draws these too
    if args.pointeval_only:
        return
    style.apply()                                   # the house style for the grid maps (serif, SHOW_TITLES, DPI): survey item 11

    fields = {}

    if PROCESS_DENSITY and fs.has('density'):
        dens = smooth(fs.density(units='mean'))
        fields['density'] = dens
        print(f"  density [rho/rho_bar]: min={dens.min():.4e} max={dens.max():.4e} mean={dens.mean():.4e}")

    if PROCESS_STREAMS and fs.has('streams'):
        streams = smooth(fs.load('streams'))
        fields['streams'] = streams
        print(f"  streams: max={streams.max():.2f} mean={streams.mean():.2f} "
              f"multi-stream={100*np.mean(streams > 1 + STREAM_TOL):.2f}%")

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
