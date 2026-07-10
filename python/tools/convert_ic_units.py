#!/usr/bin/env python3
"""Convert a TNG initial-conditions snapshot from ckpc/h to h-removed ckpc.

PS-DTFE matches Lagrangian positions (--lagrangianInput) to the snapshot's
Coordinates by ParticleID, so the IC file must be in the SAME units as the
merged snapshots (combined_NNN.hdf5, which python/merge_HDF5.py writes with
the little-h removed). This applies the identical convention to the ICs:

    Coordinates   -> Coordinates / h      (ckpc/h -> ckpc)
    Header BoxSize -> BoxSize / h
    Velocities, ParticleIDs, MassTable, all other attributes: unchanged

Usage:
    python3 python/convert_ic_units.py <snap_ics.hdf5> <combined_ics.hdf5>

h is taken from the input file's Header/HubbleParam.
"""

import argparse
import sys

import h5py
import numpy as np

CHUNK = 8_000_000   # particles per IO chunk (~96 MB of float32 coordinates)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="IC snapshot in ckpc/h (e.g. snap_ics.hdf5)")
    ap.add_argument("output", help="h-removed output (e.g. combined_ics.hdf5)")
    args = ap.parse_args()

    with h5py.File(args.input, "r") as src, h5py.File(args.output, "w-") as dst:
        h = float(src["Header"].attrs["HubbleParam"])
        if not (0.5 < h < 1.0):
            sys.exit(f"HubbleParam {h} looks wrong; refusing to convert.")

        # copy everything as-is first (preserves dtypes, layouts and attributes)
        for name in src:
            src.copy(name, dst)

        dst["Header"].attrs["BoxSize"] = src["Header"].attrs["BoxSize"] / h
        print(f"h = {h}")
        print(f"BoxSize: {src['Header'].attrs['BoxSize']} -> {dst['Header'].attrs['BoxSize']}")

        # rescale Coordinates in place, chunked; compute in float64, store as source dtype
        for grp in dst:
            if not grp.startswith("PartType"):
                continue
            coords = dst[grp]["Coordinates"]
            n = coords.shape[0]
            for lo in range(0, n, CHUNK):
                hi = min(lo + CHUNK, n)
                block = coords[lo:hi].astype(np.float64) / h
                coords[lo:hi] = block.astype(coords.dtype)
            print(f"{grp}: rescaled {n} Coordinates by 1/h "
                  f"(Velocities/ParticleIDs untouched)")

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
