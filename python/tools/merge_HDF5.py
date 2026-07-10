"""Merge per-snapshot TNG chunk files into a single h-free combined_NNN.hdf5.

Coordinates and the header BoxSize are divided by h (ckpc/h -> ckpc); everything else is
copied as-is. h is read from the input file's HubbleParam header (fallback: --hubble).

Usage:
    python3 merge_HDF5.py                                   # defaults below
    python3 merge_HDF5.py -d /path/to/TNG50-4-Dark 8 17 33  # merge snapshots 8, 17, 33
"""

import argparse

import h5py
import numpy as np
from pathlib import Path


SNAPSHOT_NUMBERS = [99]

NUM_SUBFILES = 4

BASE_DIR = "/Users/luukw/output/TNG100-3-Dark"

SINGLE_FILES = [
]

H_VALUE = 0.6774

DATASETS_INFO = {
    "PartType1": {
        "Coordinates": {"shape": (None, 3), "dtype": 'float32'},
        "ParticleIDs": {"shape": (None,), "dtype": 'uint64'},
        "Velocities": {"shape": (None, 3), "dtype": 'float32'},
    }
}


def merge_snapshot_files(snapshot_num, num_subfiles, base_dir, h_value, datasets_info):
    snapshot_str = f"{snapshot_num:03d}"

    snapshot_dir = Path(base_dir) / f"snapdir_{snapshot_str}"
    input_files = [snapshot_dir / f"snap_{snapshot_str}.{i}.hdf5" 
                   for i in range(num_subfiles)]
    output_file = snapshot_dir / f"combined_{snapshot_str}.hdf5"

    print(f"\n{'='*60}")
    print(f"Processing snapshot {snapshot_str}")
    print(f"Directory: {snapshot_dir}")
    print(f"Merging {num_subfiles} subfiles...")

    missing = [f for f in input_files if not f.exists()]
    if missing:
        print(f"\nProblem: Can't find {len(missing)} subfile(s):")
        for f in missing:
            print(f"  Missing: {f}")
        print("Check your BASE_DIR and NUM_SUBFILES settings.\n")
        return
    
    with h5py.File(input_files[0], 'r') as f_first:
        box_size = f_first['Header'].attrs.get('BoxSize')
        box_size_physical = box_size / h_value

    with h5py.File(output_file, 'w') as f_out:
        with h5py.File(input_files[0], 'r') as f_first:
            for group_name in ["Config", "Parameters"]:
                if group_name in f_first:
                    f_out.copy(f_first[group_name], group_name)

            if "Header" in f_first:
                f_out.copy(f_first["Header"], "Header")

        f_out["Header"].attrs["BoxSize"] = box_size_physical

        for group_name in datasets_info:
            f_out.create_group(group_name)

        out_datasets = {}
        for group_name, dsets in datasets_info.items():
            out_datasets[group_name] = {}
            group = f_out[group_name]
            for dset_name, props in dsets.items():
                initial_shape = (0,) + props["shape"][1:] if len(props["shape"]) > 1 else (0,)
                maxshape = (None,) + props["shape"][1:] if len(props["shape"]) > 1 else (None,)
                out_datasets[group_name][dset_name] = group.create_dataset(
                    dset_name, 
                    shape=initial_shape, 
                    maxshape=maxshape,
                    dtype=props["dtype"], 
                    chunks=True
                )
        
        total_counts = np.zeros(6, dtype=np.int64)

        for file_idx, fname in enumerate(input_files):
            progress = f"[{file_idx + 1}/{len(input_files)}]"
            print(f"  {progress} Reading {fname.name}...")

            try:
                with h5py.File(fname, 'r') as f_in:
                    for group_name, dsets in datasets_info.items():
                        for dset_name in dsets:
                            if group_name in f_in and dset_name in f_in[group_name]:
                                data = f_in[group_name][dset_name][...]

                                if dset_name == "Coordinates":
                                    data = data / h_value

                                out_dset = out_datasets[group_name][dset_name]
                                old_size = out_dset.shape[0]
                                new_size = old_size + data.shape[0]
                                new_shape = (new_size,) + out_dset.shape[1:]
                                out_dset.resize(new_shape)
                                out_dset[old_size:new_size, ...] = data

                                if dset_name == "ParticleIDs":
                                    particle_type_num = int(group_name[-1])
                                    total_counts[particle_type_num] += data.shape[0]
                            else:
                                print(f"    Warning: {group_name}/{dset_name} not found in this file")

            except Exception as e:
                print(f"    Error: couldn't read {fname.name} - {e}")
                continue

        header = f_out["Header"]
        header.attrs["NumPart_ThisFile"] = total_counts
        header.attrs["NumPart_Total"] = total_counts
    
    print(f"\nSaved to: {output_file}")
    print(f"  Particles merged: {total_counts[total_counts > 0]}")
    print(f"  Physical box size: {box_size_physical:.2f} kpc")
    print('='*60)


def convert_single_file(input_path, h_value, datasets_info):
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"\nSkipping {input_path} - file not found")
        return

    stem = input_path.stem
    suffix = input_path.suffix
    prefix = stem.split("_", 1)[1] if "_" in stem else stem
    output_path = input_path.parent / f"combined_{prefix}{suffix}"

    print(f"\n{'='*60}")
    print(f"Converting single file: {input_path}")
    print(f"Output: {output_path}")

    with h5py.File(input_path, 'r') as f_in:
        box_size = f_in['Header'].attrs.get('BoxSize')
        box_size_physical = box_size / h_value
        print(f"  BoxSize: {box_size:.2f} ckpc/h → {box_size_physical:.2f} ckpc")

        with h5py.File(output_path, 'w') as f_out:
            for group_name in f_in:
                if group_name not in datasets_info:
                    f_out.copy(f_in[group_name], group_name)

            f_out["Header"].attrs["BoxSize"] = box_size_physical

            for group_name, dsets in datasets_info.items():
                if group_name not in f_in:
                    continue
                grp = f_out.create_group(group_name)

                for dset_name in dsets:
                    if dset_name not in f_in[group_name]:
                        continue

                    data = f_in[group_name][dset_name][...]
                    n = data.shape[0]

                    if dset_name == "Coordinates":
                        data = (data / h_value).astype(dsets[dset_name]["dtype"])
                        print(f"  {group_name}/{dset_name}: {n} particles, divided by h={h_value}")
                    else:
                        print(f"  {group_name}/{dset_name}: {n} entries (copied as-is)")

                    grp.create_dataset(dset_name, data=data)

    print(f"  Saved: {output_path}")
    print('='*60)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("snapshots", type=int, nargs="*", default=None,
                    help=f"snapshot numbers to merge (default: {SNAPSHOT_NUMBERS})")
    ap.add_argument("-d", "--data-dir", default=BASE_DIR,
                    help=f"simulation directory holding snapdir_NNN/ (default: {BASE_DIR})")
    ap.add_argument("-n", "--num-subfiles", type=int, default=NUM_SUBFILES,
                    help=f"chunk files per snapshot (default: {NUM_SUBFILES})")
    ap.add_argument("--hubble", type=float, default=H_VALUE,
                    help=f"fallback h if the file header lacks HubbleParam (default: {H_VALUE})")
    args = ap.parse_args()
    snapshots = args.snapshots if args.snapshots else SNAPSHOT_NUMBERS

    print("\nHDF5 Snapshot Merger")
    print(f"Base directory: {args.data_dir}")
    print(f"Snapshots to process: {snapshots}")
    print(f"Single files to convert: {SINGLE_FILES}")

    for snapshot_num in snapshots:
        # h from the data itself when available; the CLI value is only a fallback
        first = Path(args.data_dir) / f"snapdir_{snapshot_num:03d}" / f"snap_{snapshot_num:03d}.0.hdf5"
        h_value = args.hubble
        if first.exists():
            with h5py.File(first, "r") as f:
                h_value = float(f["Header"].attrs.get("HubbleParam", args.hubble))
        merge_snapshot_files(
            snapshot_num,
            args.num_subfiles,
            args.data_dir,
            h_value,
            DATASETS_INFO
        )

    for single_file in SINGLE_FILES:
        convert_single_file(
            Path(args.data_dir) / single_file,
            args.hubble,
            DATASETS_INFO
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
