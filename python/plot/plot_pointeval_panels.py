"""Publication-resolution continuous-field maps from '--sample-points' output.

Renders the point-evaluated PS-DTFE planes written by the binary for a slab generated
with tools/make_image_plane.py: projected density (physical units, log-gray on black),
the same projection inverted (the fine sheet/tessellation structure, black on white),
and the pointwise stream count N_streams (discrete colormap). Because the points sample
the continuous piecewise-linear field, the maps are grid-free: resolution is set by the
pixel count of the plane file, limited only by the tessellation itself.

    python3 plot/plot_pointeval_panels.py --prefix <PS-DTFE output root> --plane plane_100_99.json
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

PANELS = ("density", "sheet", "streams")


def load_planes(prefix: Path, side: dict):
    shape = (side["planes"], side["nv"], side["nu"])
    n = shape[0] * shape[1] * shape[2]
    den = np.fromfile(f"{prefix}.pts_den", dtype=np.float64)
    streams = np.fromfile(f"{prefix}.pts_streams", dtype=np.int32)
    if den.size != n or streams.size != n:
        raise SystemExit(f"{prefix}.pts_* hold {den.size}/{streams.size} points, but the "
                         f"sidecar describes {n} ({shape}); wrong --plane or --prefix?")
    return den.reshape(shape), streams.reshape(shape)


def stream_cmap_norm(smax: int):
    """Discrete odd-count colormap: black for single-stream, magma steps above."""
    upper = max(5, min(int(smax) | 1, 25))            # odd cap, keeps the legend readable
    bounds = np.arange(0, upper + 2, 2)               # [0,2) = 1 stream, [2,4) = 3, ...
    steps = len(bounds) - 1
    colors = [(0, 0, 0, 1)] + [plt.get_cmap("magma")(x)
                               for x in np.linspace(0.35, 0.95, steps - 1)]
    return ListedColormap(colors), BoundaryNorm(bounds, steps), bounds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True, type=Path,
                   help="output root passed to the binary (reads <prefix>.pts_den/.pts_streams)")
    p.add_argument("--plane", required=True, type=Path, help="sidecar JSON from tools/make_image_plane.py")
    p.add_argument("-o", "--output", type=Path, default=None, help="image path (default <prefix>_pointeval_panels.png)")
    p.add_argument("--panels", default=",".join(PANELS),
                   help=f"comma list from {{{','.join(PANELS)}}} (default all three)")
    p.add_argument("--project", choices=("plane", "mean"), default="plane",
                   help="density panels: 'plane' (default) = the central plane, a crisp "
                        "cross-section of the continuous field; 'mean' = average over all "
                        "sampled planes -- a smooth projection ONLY with many planes "
                        "(>= 16): a handful of planes GHOSTS, stamping every inclined "
                        "structure once per plane with the inter-plane offset")
    p.add_argument("--vmin", type=float, default=None,
                   help="density scale min, rho/rho_bar (default: 0.1th percentile -- adapts "
                        "to the snapshot's contrast, so z=20 and z=0 both render well)")
    p.add_argument("--vmax", type=float, default=None,
                   help="density scale max, rho/rho_bar (default: 99.99th percentile)")
    p.add_argument("--dpi", type=int, default=None,
                   help="output dpi (default: native, one image pixel per sample pixel)")
    p.add_argument("--image", action="store_true",
                   help="write pixel-exact PNGs instead of the annotated figure: one file "
                        "per panel, exactly nu x nv pixels (8192 -> true 8K), no axes or "
                        "colorbars -- for zooming, posters and compositing")
    args = p.parse_args()

    panels = [s.strip() for s in args.panels.split(",") if s.strip()]
    bad = set(panels) - set(PANELS)
    if bad or not panels:
        raise SystemExit(f"unknown panels {sorted(bad)}; choose from {PANELS}")

    side = json.loads(args.plane.read_text())
    den, streams = load_planes(args.prefix, side)
    proj = den.mean(axis=0) if args.project == "mean" else den[side["planes"] // 2]
    s_mid = streams[side["planes"] // 2]                 # pointwise count on the central plane
    if args.project == "mean" and 1 < side["planes"] < 16:
        print(f"WARNING: --project mean over only {side['planes']} planes ghosts inclined "
              f"structures (one copy per plane); regenerate with --planes 16+ or use --project plane")
    extent = [side["u0"], side["u1"], side["v0"], side["v1"]]
    # percentile auto-stretch: the dynamic range grows by ~4 decades from z=20 (quasi-linear,
    # everything within [0.5, 5]) to z=0; fixed limits cannot serve both. Explicit flags win.
    vmin = args.vmin if args.vmin is not None else max(float(np.percentile(proj, 0.1)), 1e-4)
    vmax = args.vmax if args.vmax is not None else float(np.percentile(proj, 99.99))
    rho_bar = side.get("rho_bar_1e10")                   # 1e10 Msun/Mpc^3 (h-free), may be None

    print(f"density ({args.project}): min {proj.min():.3e}  max {proj.max():.3e}  mean {proj.mean():.4f} (rho/rho_bar)")
    print(f"stretch: [{vmin:.3g}, {vmax:.3g}] rho/rho_bar"
          + (" (auto percentiles)" if args.vmin is None and args.vmax is None else ""))
    print(f"streams (central plane): max {s_mid.max()}  multi-stream fraction {(s_mid > 1).mean():.4f}")

    if args.image:
        # pixel-exact export: colormap the raw sample grid directly (uint8 RGBA, one image
        # pixel per sample point), bypassing matplotlib's figure machinery entirely
        base = args.output or args.prefix.with_name(args.prefix.name + "_pointeval_panels.png")
        for name in panels:
            if name == "streams":
                cmap, norm, _ = stream_cmap_norm(s_mid.max())
                rgba = cmap(norm(s_mid), bytes=True)
            else:
                cmap = plt.get_cmap("gray" if name == "density" else "gray_r")
                rgba = cmap(LogNorm(vmin=vmin, vmax=vmax)(np.clip(proj, vmin, vmax)), bytes=True)
            out = base if (len(panels) == 1 and args.output) else \
                base.with_name(f"{base.stem}_{name}{base.suffix}")
            plt.imsave(out, rgba[::-1])          # flip: same origin='lower' as the figure
            print(f"wrote {out}  ({side['nu']} x {side['nv']} px, pixel-exact, no decorations)")
        return

    width_in = 12.0
    panel_h = width_in * (side["nv"] / side["nu"])
    fig, axes = plt.subplots(len(panels), 1, figsize=(width_in, panel_h * len(panels)),
                             sharex=True, constrained_layout=True, squeeze=False)
    axes = axes.ravel()

    for ax, name in zip(axes, panels):
        if name == "density":
            if rho_bar is not None:                      # physical colorbar like the literature
                im = ax.imshow(proj * rho_bar * 1e10, origin="lower", extent=extent, cmap="gray",
                               norm=LogNorm(vmin=vmin * rho_bar * 1e10, vmax=vmax * rho_bar * 1e10),
                               interpolation="nearest", aspect="equal")
                label = r"$\rho_{\rm DM}\ [M_\odot\,{\rm Mpc}^{-3}]$"
            else:
                im = ax.imshow(proj, origin="lower", extent=extent, cmap="gray",
                               norm=LogNorm(vmin=vmin, vmax=vmax),
                               interpolation="nearest", aspect="equal")
                label = r"$\rho/\bar\rho$"
            fig.colorbar(im, ax=ax, pad=0.01, label=label)
        elif name == "sheet":
            im = ax.imshow(proj, origin="lower", extent=extent, cmap="gray_r",
                           norm=LogNorm(vmin=vmin, vmax=vmax),
                           interpolation="nearest", aspect="equal")
            fig.colorbar(im, ax=ax, pad=0.01, label=r"$\rho/\bar\rho$")
        else:
            cmap, norm, bounds = stream_cmap_norm(s_mid.max())
            im = ax.imshow(s_mid, origin="lower", extent=extent, cmap=cmap, norm=norm,
                           interpolation="nearest", aspect="equal")
            cb = fig.colorbar(im, ax=ax, pad=0.01, label=r"$N_{\rm streams}$",
                              ticks=bounds[:-1] + 1)
            cb.set_ticklabels([str(b + 1) for b in bounds[:-1]])
        ax.set_ylabel(f"${side['v_axis']}$ [Mpc]")
    axes[-1].set_xlabel(f"${side['u_axis']}$ [Mpc]")

    out = args.output or args.prefix.with_name(args.prefix.name + "_pointeval_panels.png")
    # native resolution: the drawn axes span ~all of width_in after constrained_layout;
    # dpi = nu/width gives >= one output pixel per sampled pixel (colorbars add margin)
    dpi = args.dpi or max(200, round(side["nu"] / width_in))
    fig.savefig(out, dpi=dpi)
    print(f"wrote {out}  ({dpi} dpi, {side['nu']} x {side['nv']} pixels per panel, "
          f"slab {side['axis']} = {side['center']:g} +/- {side['thickness'] / 2:g}, "
          f"{side['planes']} planes)")


if __name__ == "__main__":
    main()
