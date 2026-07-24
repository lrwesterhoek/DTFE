"""Generate a '--sample-points' input for high-resolution slab images.

The PS-DTFE field is continuous (piecewise linear over the tessellation), so image
resolution is a *sampling* choice, not a grid property: this script writes the pixel
centres of an nu x nv image, replicated on '--planes' equally spaced planes across a
slab of '--thickness' centred at '--center' along '--axis', as the raw float64 N x 3
binary that '--sample-points' auto-detects. A JSON sidecar records the geometry and
snapshot metadata so plot/plot_pointeval_panels.py is self-configuring.

Coordinates are in the box coordinate system the binary sees: header lengths divided
by '--mpc-unit' (h-free Mpc for the merged TNG snapshots, which store h-free ckpc and
run with MpcUnit 1000; the synthetic test inputs use --mpc-unit 1).

    python3 tools/make_image_plane.py --sim TNG100-3-Dark --snap 99 \
        --axis z --center 55 --thickness 3 --planes 4 --nu 8192 -o plane_100_99

    python3 tools/make_image_plane.py --combined tests/tmp/ps_regression.hdf5 \
        --mpc-unit 1 --nu 1024 -o /tmp/plane_test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # python/ on sys.path

AXES = {"x": 0, "y": 1, "z": 2}


def read_header(combined: Path, mpc_unit: float) -> dict:
    """Box size (box units), redshift and rho_bar (1e10 Msun/Mpc^3, h-free), mirroring
    dtfelib.FieldSet._read_meta; rho_bar is None when the header lacks the mass info."""
    import h5py
    with h5py.File(combined, "r") as f:
        h = f["Header"].attrs
        box = float(h["BoxSize"]) / mpc_unit
        redshift = float(h.get("Redshift", 0.0))
        rho_bar = None
        try:
            npart = int(h["NumPart_ThisFile"][1])
            mass = float(h["MassTable"][1])
            if mass <= 0:
                mass = float(f["PartType1/Masses"][0])
            if "HFreeUnits" not in h:
                mass /= float(h["HubbleParam"])   # pre-unification files: 1e10 Msun/h
            if npart > 0 and mass > 0:
                rho_bar = npart * mass / box ** 3
        except (KeyError, IndexError):
            pass
        return {"box": box, "redshift": redshift, "rho_bar": rho_bar}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sim", help="simulation name (path via dtfelib DATA_ROOT); needs --snap")
    src.add_argument("--combined", type=Path, help="explicit combined_*.hdf5 snapshot path")
    p.add_argument("--snap", type=int, help="snapshot number (with --sim)")
    p.add_argument("--mpc-unit", type=float, default=1000.0,
                   help="input length units per box unit, the binary's --MpcUnit (default 1000)")
    p.add_argument("--axis", choices=AXES, default="z", help="projection axis (default z: image is x-y)")
    p.add_argument("--axes", default=None,
                   help="generate ONE plane file per axis, e.g. 'xyz' or 'x,y,z' -> "
                        "<stem>_x.bin/.json etc. (overrides --axis). Each axis is a SEPARATE "
                        "point-eval run: three 8192^2 planes cannot share one run (3x67M points "
                        "of records exceeds RAM), so this only writes the inputs, not one file")
    p.add_argument("--center", type=float, default=None, help="slab centre along --axis (default box/2)")
    p.add_argument("--thickness", type=float, default=2.0,
                   help="slab thickness, only used with --planes > 1 (default 2 box units)")
    p.add_argument("--planes", type=int, default=1,
                   help="sampling planes across the slab (default 1 = a single crisp "
                        "cross-section). For a smooth projected look use >= 16 with the plot's "
                        "--project mean; a HANDFUL of planes ghosts every inclined structure "
                        "(one displaced copy per plane)")
    p.add_argument("--nu", type=int, default=4096, help="pixels along the horizontal image axis (default 4096)")
    p.add_argument("--supersample", type=int, default=1, metavar="K",
                   help="evaluate KxK sub-samples per output pixel and average them (default 1 "
                        "= one point at the pixel centre). K>1 turns the map from a POINT "
                        "sample into an estimate of the pixel-AREA average, which is what the "
                        "deposit grid reports: it removes aliasing where structure is finer "
                        "than a pixel (halo cores, thin caustics). Costs K^2 sample points")
    p.add_argument("--u0", type=float, default=None, help="horizontal extent start (default 0)")
    p.add_argument("--u1", type=float, default=None, help="horizontal extent end (default box)")
    p.add_argument("--v0", type=float, default=None, help="vertical extent start (default 0)")
    p.add_argument("--v1", type=float, default=None, help="vertical extent end (default box)")
    p.add_argument("-o", "--output", required=True, help="output stem: writes <stem>.bin + <stem>.json")
    args = p.parse_args()

    if args.sim:
        if args.snap is None:
            p.error("--sim needs --snap")
        from dtfelib.cli import DATA_ROOT
        combined = DATA_ROOT / args.sim / f"snapdir_{args.snap:03d}" / f"combined_{args.snap:03d}.hdf5"
    else:
        combined = args.combined
    if not combined.is_file():
        sys.exit(f"snapshot not found: {combined}")

    meta = read_header(combined, args.mpc_unit)
    box = meta["box"]

    if args.supersample < 1:
        sys.exit("--supersample must be >= 1")
    if args.thickness <= 0 or args.planes < 1 or args.nu < 2:
        sys.exit("non-positive thickness, or degenerate planes/nu")

    if args.axes:
        axes = [a for a in args.axes.replace(",", "") if a.strip()]
        bad = [a for a in axes if a not in AXES]
        if bad:
            sys.exit(f"--axes: unknown axis/axes {bad}; use letters from x,y,z")
    else:
        axes = [args.axis]

    for axis in axes:
        # per-axis stem: <stem>_x when several axes, plain <stem> for a single one (keeps
        # the existing single-plane filenames unchanged)
        stem = Path(args.output if len(axes) == 1 else f"{args.output}_{axis}")
        write_plane(combined, meta, box, axis, args, stem)


def write_plane(combined, meta, box, axis, args, stem):
    w_axis = AXES[axis]
    u_axis, v_axis = (w_axis + 1) % 3, (w_axis + 2) % 3   # cyclic: z -> (x, y)
    u0 = 0.0 if args.u0 is None else args.u0
    u1 = box if args.u1 is None else args.u1
    v0 = 0.0 if args.v0 is None else args.v0
    v1 = box if args.v1 is None else args.v1
    center = 0.5 * box if args.center is None else args.center
    if not (u1 > u0 and v1 > v0):
        sys.exit("empty image extent (u1<=u0 or v1<=v0)")

    k = args.supersample
    nv = max(2, round(args.nu * (v1 - v0) / (u1 - u0)))    # square pixels
    # KxK sub-sample centres per output pixel: offsets (i+0.5)/K within the pixel, so the
    # mean over the block is the midpoint-rule estimate of the pixel-area average
    fu, fv = args.nu * k, nv * k
    u = u0 + (np.arange(fu) + 0.5) * (u1 - u0) / fu
    v = v0 + (np.arange(fv) + 0.5) * (v1 - v0) / fv
    if args.planes == 1:
        w = np.array([center])
    else:
        w = center + 0.5 * args.thickness * np.linspace(-1.0, 1.0, args.planes)
    w = np.mod(w, box)                                      # periodic box

    U, V = np.meshgrid(u, v)                                # (nv*K, nu*K), rows = v
    pts = np.empty((args.planes, fv, fu, 3), dtype=np.float64)
    pts[..., u_axis] = U
    pts[..., v_axis] = V
    for ip, wk in enumerate(w):        # NOT 'k': that is the supersample factor, used below
        pts[ip, ..., w_axis] = wk

    pts.reshape(-1, 3).tofile(f"{stem}.bin")
    sidecar = {
        "combined": str(combined), "mpc_unit": args.mpc_unit, "box": box,
        "redshift": meta["redshift"], "rho_bar_1e10": meta["rho_bar"],
        "axis": axis, "u_axis": "xyz"[u_axis], "v_axis": "xyz"[v_axis],
        "u0": u0, "u1": u1, "v0": v0, "v1": v1, "nu": args.nu, "nv": nv,
        "supersample": k,
        "center": center, "thickness": args.thickness, "planes": args.planes,
        "plane_coords": w.tolist(),
    }
    with open(f"{stem}.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    n = args.planes * fv * fu
    print(f"wrote {stem}.bin: {n:,} points ({n * 24 / 1e9:.2f} GB) = "
          f"{args.nu} x {nv} pixels"
          + (f" x {k}x{k} sub-samples" if k > 1 else "")
          + f" x {args.planes} planes  [axis {axis} -> "
          f"{'xyz'[u_axis]}{'xyz'[v_axis]} image]")


if __name__ == "__main__":
    main()
