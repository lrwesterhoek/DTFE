#!/usr/bin/env python3
"""
Plot PS-DTFE output fields (slice + transverse-averaged profile per field) from
the raw-binary grids written under <output_root>. Requires numpy, matplotlib.

Example
  python3 tests/generate_ps_test_data.py --out /tmp/ps_demo.hdf5 --n 64 --box 100 \
      --amplitude-factor 1.8 --jitter-frac 0.02 --seed 42
  ./PS-DTFE /tmp/ps_demo.hdf5 /tmp/ps_demo --grid 128 --periodic --partition 2 2 2 \
      --field density velocity dispersion density_a --input 105 --MpcUnit 1
  python3 tests/debug_plot_pancake.py /tmp/ps_demo --out /tmp/ps_demo.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def detect_grid(path):
    nbytes = os.path.getsize(path)
    for itemsize in (4, 8):
        if nbytes % itemsize:
            continue
        ncells = nbytes // itemsize
        g = round(ncells ** (1.0 / 3.0))
        if g ** 3 == ncells:
            return g
    sys.exit(f"Could not infer grid size from {path} ({nbytes} bytes).")


def load(path, grid, ncomp=1):
    if not os.path.exists(path):
        return None
    raw = np.fromfile(path, dtype=np.uint8)
    n = grid ** 3 * ncomp
    if raw.size == n * 4:
        data = raw.view(np.float32).astype(np.float64)
    elif raw.size == n * 8:
        data = raw.view(np.float64)
    else:
        print(f"  ! {os.path.basename(path)}: {raw.size} bytes != expected "
              f"{n*4}/{n*8} for grid {grid}^3 x {ncomp}; skipping")
        return None
    return data.reshape((grid, grid, grid) + ((ncomp,) if ncomp > 1 else ()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="PS-DTFE output root (the path given to PS-DTFE)")
    ap.add_argument("--grid", type=int, default=None, help="grid size N (auto-detected if omitted)")
    ap.add_argument("--axis", choices=["x", "y", "z"], default="x",
                    help="profile axis (default x = pancake displacement direction)")
    ap.add_argument("--out", default=None, help="output figure path (default <root>_plot.png)")
    args = ap.parse_args()

    root = args.root
    den_path = root + ".den"
    if not os.path.exists(den_path):
        sys.exit(f"FAIL: {den_path} not found. Pass the same <output_root> you gave PS-DTFE.")
    G = args.grid or detect_grid(den_path)
    ax = {"x": 0, "y": 1, "z": 2}[args.axis]
    other = tuple(d for d in (0, 1, 2) if d != ax)
    out = args.out or (root + "_plot.png")
    print(f"reading PS-DTFE outputs from '{root}*'  (grid {G}^3, profile along {args.axis})")

    den = load(den_path, G)
    a_den = load(root + ".a_den", G)
    streams = load(root + ".streams", G)
    a_streams = load(root + ".a_streams", G)
    vel = load(root + ".vel", G, 3)
    veldisp = load(root + ".velDisp", G)
    a_veldisp = load(root + ".a_velDisp", G)

    box = 1.0
    coord = (np.arange(G) + 0.5) / G * box
    mid = G // 2

    def slice_mid(field):
        return np.take(field, mid, axis=other[0])

    def profile(field):
        return field.mean(axis=other)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f"PS-DTFE fields  —  {os.path.basename(root)}  (grid {G}$^3$)", fontsize=14)
    extent = [0, box, 0, box]

    if den is not None:
        d = slice_mid(den); dpos = d[d > 0]
        vmin = np.percentile(dpos, 1) if dpos.size else 1e-3
        im = axes[0, 0].imshow(np.log10(np.maximum(d, vmin)).T, origin="lower",
                               extent=extent, cmap="inferno", aspect="auto")
        axes[0, 0].set_title("log$_{10}$ density (slice)"); fig.colorbar(im, ax=axes[0, 0])

    if streams is not None:
        im = axes[0, 1].imshow(slice_mid(streams).T, origin="lower", extent=extent,
                               cmap="viridis", aspect="auto")
        axes[0, 1].set_title("stream count (slice)"); fig.colorbar(im, ax=axes[0, 1])

    if veldisp is not None:
        sig = np.sqrt(np.maximum(slice_mid(veldisp), 0.0))
        im = axes[0, 2].imshow(sig.T, origin="lower", extent=extent, cmap="magma", aspect="auto")
        axes[0, 2].set_title(r"velocity dispersion $\sigma$ (slice)"); fig.colorbar(im, ax=axes[0, 2])
    else:
        axes[0, 2].set_visible(False)

    if vel is not None:
        vmag = np.sqrt((slice_mid(vel) ** 2).sum(axis=-1))
        im = axes[0, 3].imshow(vmag.T, origin="lower", extent=extent, cmap="cividis", aspect="auto")
        axes[0, 3].set_title("|velocity| (slice)"); fig.colorbar(im, ax=axes[0, 3])
    else:
        axes[0, 3].set_visible(False)

    for a in axes[0]:
        if a.get_visible():
            a.set_xlabel("x / L"); a.set_ylabel("y / L")

    if den is not None:
        axes[1, 0].plot(coord, profile(den), label="density", color="C3")
        if a_den is not None:
            axes[1, 0].plot(coord, profile(a_den), "--", label="density_a", color="C1")
        axes[1, 0].set_yscale("log"); axes[1, 0].set_title("density profile")
        axes[1, 0].set_xlabel(f"{args.axis} / L"); axes[1, 0].set_ylabel(r"$\langle\rho\rangle$")
        axes[1, 0].legend()

    if streams is not None:
        axes[1, 1].plot(coord, profile(streams), label="streams", color="C0")
        if a_streams is not None:
            axes[1, 1].plot(coord, profile(a_streams), "--", label="a_streams", color="C9")
        axes[1, 1].set_title("mean stream count profile")
        axes[1, 1].set_xlabel(f"{args.axis} / L"); axes[1, 1].set_ylabel("streams"); axes[1, 1].legend()

    if vel is not None:
        axes[1, 2].plot(coord, profile(vel[..., ax]), color="C2")
        axes[1, 2].axhline(0, color="k", lw=0.6, ls=":")
        axes[1, 2].set_title(f"velocity v$_{args.axis}$ profile")
        axes[1, 2].set_xlabel(f"{args.axis} / L"); axes[1, 2].set_ylabel(f"$v_{args.axis}$")
    else:
        axes[1, 2].set_visible(False)

    if veldisp is not None:
        axes[1, 3].plot(coord, np.sqrt(np.maximum(profile(veldisp), 0)), label=r"$\sigma$", color="C4")
        if a_veldisp is not None:
            axes[1, 3].plot(coord, np.sqrt(np.maximum(profile(a_veldisp), 0)), "--",
                            label=r"$\sigma_a$", color="C5")
        axes[1, 3].set_title(r"velocity dispersion $\sigma$ profile")
        axes[1, 3].set_xlabel(f"{args.axis} / L"); axes[1, 3].set_ylabel(r"$\sigma$"); axes[1, 3].legend()
    else:
        axes[1, 3].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
