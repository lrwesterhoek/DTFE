#!/usr/bin/env python3
"""
Generate a synthetic Gadget-HDF5 snapshot (input type 105) for PS-DTFE tests:
a 1-D Zel'dovich pancake, multi-streaming once A*k >= 1. Requires numpy, h5py.
"""

import argparse
import sys

import numpy as np


def build_pancake(n_side, box, amplitude_factor, jitter_frac, seed, margin_frac=0.0,
                  crossed=False):
    rng = np.random.default_rng(seed)

    if margin_frac > 0.0:
        lo, hi = margin_frac * box, (1.0 - margin_frac) * box
        spacing = (hi - lo) / n_side
        coords1d = lo + (np.arange(n_side) + 0.5) * spacing
    else:
        spacing = box / n_side
        coords1d = (np.arange(n_side) + 0.5) * spacing
    qx, qy, qz = np.meshgrid(coords1d, coords1d, coords1d, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)

    q += (rng.uniform(-1.0, 1.0, size=q.shape) * jitter_frac * spacing)
    if margin_frac == 0.0:
        q = np.mod(q, box)

    k = 2.0 * np.pi / box
    amplitude = amplitude_factor / k

    displacement = np.zeros_like(q)
    naxes = q.shape[1] if crossed else 1
    for d in range(naxes):
        displacement[:, d] = amplitude * np.sin(k * q[:, d])

    if margin_frac > 0.0:
        x = q + displacement
        if x.min() < 0.0 or x.max() > box:
            sys.exit("ERROR: non-periodic clump leaves the box; lower --amplitude-factor "
                     "or raise --margin-frac.")
    else:
        x = np.mod(q + displacement, box)

    velocity = displacement * 100.0

    return q, x, velocity


def write_snapshot(filename, q, x, velocity, box, particle_mass):
    try:
        import h5py
    except ImportError:
        sys.exit("ERROR: h5py is required to write the test snapshot "
                 "(install with: pip install h5py).")

    n = x.shape[0]
    npart = np.zeros(6, dtype=np.int32)
    npart[1] = n
    npart_total = np.zeros(6, dtype=np.uint32)
    npart_total[1] = n
    mass_table = np.zeros(6, dtype=np.float64)
    mass_table[1] = particle_mass

    with h5py.File(filename, "w") as f:
        header = f.create_group("Header")
        header.attrs.create("NumPart_ThisFile", npart)
        header.attrs.create("NumPart_Total", npart_total)
        header.attrs.create("MassTable", mass_table)
        header.attrs.create("NumFilesPerSnapshot", np.int32(1))
        header.attrs.create("BoxSize", np.float64(box))
        header.attrs.create("Time", np.float64(1.0))
        header.attrs.create("Redshift", np.float64(0.0))
        header.attrs.create("Omega0", np.float64(0.3))
        header.attrs.create("OmegaLambda", np.float64(0.7))
        header.attrs.create("HubbleParam", np.float64(0.7))

        part = f.create_group("PartType1")
        part.create_dataset("Coordinates", data=x.astype(np.float32))
        part.create_dataset("Velocities", data=velocity.astype(np.float32))
        part.create_dataset("ParticleIDs",
                            data=np.arange(1, n + 1, dtype=np.uint64))
        part.create_dataset("InitialCoordinates", data=q.astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output HDF5 filename")
    ap.add_argument("--n", type=int, default=32,
                    help="particles per side (total = n^3, default 32)")
    ap.add_argument("--box", type=float, default=100.0,
                    help="box size in Mpc (default 100)")
    ap.add_argument("--amplitude-factor", type=float, default=1.8,
                    help="A*k; >1 forces shell crossing (default 1.8)")
    ap.add_argument("--jitter-frac", type=float, default=0.05,
                    help="Lagrangian jitter as fraction of grid spacing (default 0.05)")
    ap.add_argument("--margin-frac", type=float, default=0.0,
                    help="if >0, confine particles to a centred sub-cube [m L,(1-m) L] in "
                         "vacuum and do NOT wrap -> a genuinely NON-PERIODIC clump (default 0)")
    ap.add_argument("--crossed-waves", action="store_true",
                    help="displace along ALL axes (independent sine per axis) -> a genuinely "
                         "3D multi-stream test; stream counts become {1,3,9,27}")
    ap.add_argument("--mass", type=float, default=1.0,
                    help="particle mass placed in MassTable (default 1.0)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    args = ap.parse_args()

    q, x, velocity = build_pancake(args.n, args.box, args.amplitude_factor,
                                   args.jitter_frac, args.seed, args.margin_frac,
                                   args.crossed_waves)
    write_snapshot(args.out, q, x, velocity, args.box, args.mass)

    n = x.shape[0]
    mean_density = n * args.mass / args.box ** 3
    print(f"Wrote {args.out}")
    print(f"  particles            : {n} ({args.n}^3) of type PartType1")
    print(f"  box size             : {args.box} Mpc "
          f"({'NON-periodic clump, margin %.2f' % args.margin_frac if args.margin_frac > 0 else 'periodic'})")
    print(f"  amplitude factor A*k : {args.amplitude_factor} "
          f"({'multi-stream' if args.amplitude_factor > 1 else 'single-stream'})")
    print(f"  particle mass        : {args.mass}")
    print(f"  expected mean density: {mean_density:.6g}  (= N*m/L^3)")


if __name__ == "__main__":
    main()
