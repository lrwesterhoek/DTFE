"""
Merge multiple HDF5 snapshot subfiles into a single combined file.
Handles unit conversions from comoving ckpc/h to ckpc (removing the h factor).

Also supports converting single-file snapshots (like IC files) to the same
coordinate convention, so all files use consistent units.
"""

import h5py
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration Section - Adjust these for your simulation data
# ============================================================================

# Which snapshots do you want to process? (multi-subfile snapshots in snapdir_XXX/)
SNAPSHOT_NUMBERS = [0]

# How many subfiles per snapshot?
NUM_SUBFILES = 4

# Where are your snapshot directories? (relative to this script)
BASE_DIR = "output/TNG50-3-Dark"

# Single-file snapshots that just need h-factor conversion (e.g., IC files).
# Each entry is a path relative to BASE_DIR. Output will be written next to
# the input with a "combined_" prefix (e.g., snap_ics.hdf5 → combined_ics.hdf5).
SINGLE_FILES = [
    "snap_ics.hdf5",
]

# Hubble parameter for unit conversion (check your simulation parameters)
H_VALUE = 0.6774

# Modify this if you're working with gas (PartType0), stars (PartType4), etc.
DATASETS_INFO = {
    "PartType1": {  # Dark matter particles
        "Coordinates": {"shape": (None, 3), "dtype": 'float32'},
        "ParticleIDs": {"shape": (None,), "dtype": 'uint64'},
        "Velocities": {"shape": (None, 3), "dtype": 'float32'},
    }
}

# ============================================================================


def merge_snapshot_files(snapshot_num, num_subfiles, base_dir, h_value, datasets_info):
    """
    Combine multiple subfiles for a single snapshot into one HDF5 file.
    
    This reads all the subfiles (snap_XXX.0.hdf5, snap_XXX.1.hdf5, etc.),
    merges their particle data, converts coordinates to physical units,
    and writes everything to a single combined_XXX.hdf5 file.
    """
    snapshot_str = f"{snapshot_num:03d}"
    
    # Set up file paths using relative paths
    snapshot_dir = Path(base_dir) / f"snapdir_{snapshot_str}"
    input_files = [snapshot_dir / f"snap_{snapshot_str}.{i}.hdf5" 
                   for i in range(num_subfiles)]
    output_file = snapshot_dir / f"combined_{snapshot_str}.hdf5"
    
    print(f"\n{'='*60}")
    print(f"Processing snapshot {snapshot_str}")
    print(f"Directory: {snapshot_dir}")
    print(f"Merging {num_subfiles} subfiles...")
    
    # Quick sanity check - do these files actually exist?
    missing = [f for f in input_files if not f.exists()]
    if missing:
        print(f"\nProblem: Can't find {len(missing)} subfile(s):")
        for f in missing:
            print(f"  Missing: {f}")
        print("Check your BASE_DIR and NUM_SUBFILES settings.\n")
        return
    
    # Grab box size from the first file
    with h5py.File(input_files[0], 'r') as f_first:
        box_size = f_first['Header'].attrs.get('BoxSize')
        box_size_physical = box_size / h_value
    
    # Create the combined output file
    with h5py.File(output_file, 'w') as f_out:
        # Copy metadata groups (Config, Parameters, Header) from first file
        with h5py.File(input_files[0], 'r') as f_first:
            for group_name in ["Config", "Parameters"]:
                if group_name in f_first:
                    f_out.copy(f_first[group_name], group_name)
            
            if "Header" in f_first:
                f_out.copy(f_first["Header"], "Header")
        
        # Update box size to physical units (divide out that h factor)
        f_out["Header"].attrs["BoxSize"] = box_size_physical
        
        # Create empty groups and datasets that we'll fill with data
        for group_name in datasets_info:
            f_out.create_group(group_name)
        
        out_datasets = {}
        for group_name, dsets in datasets_info.items():
            out_datasets[group_name] = {}
            group = f_out[group_name]
            for dset_name, props in dsets.items():
                # Start with zero particles, resize as we go
                initial_shape = (0,) + props["shape"][1:] if len(props["shape"]) > 1 else (0,)
                maxshape = (None,) + props["shape"][1:] if len(props["shape"]) > 1 else (None,)
                out_datasets[group_name][dset_name] = group.create_dataset(
                    dset_name, 
                    shape=initial_shape, 
                    maxshape=maxshape,
                    dtype=props["dtype"], 
                    chunks=True
                )
        
        # Now loop through each subfile and append its data
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
                                
                                # Convert coordinates: ckpc/h → kpc (physical units)
                                if dset_name == "Coordinates":
                                    data = data / h_value
                                
                                # Append to output dataset
                                out_dset = out_datasets[group_name][dset_name]
                                old_size = out_dset.shape[0]
                                new_size = old_size + data.shape[0]
                                new_shape = (new_size,) + out_dset.shape[1:]
                                out_dset.resize(new_shape)
                                out_dset[old_size:new_size, ...] = data
                                
                                # Keep track of how many particles we've added
                                if dset_name == "ParticleIDs":
                                    particle_type_num = int(group_name[-1])
                                    total_counts[particle_type_num] += data.shape[0]
                            else:
                                print(f"    Warning: {group_name}/{dset_name} not found in this file")
                                
            except Exception as e:
                print(f"    Error: couldn't read {fname.name} - {e}")
                continue
        
        # Update header attributes with final particle counts
        header = f_out["Header"]
        header.attrs["NumPart_ThisFile"] = total_counts
        header.attrs["NumPart_Total"] = total_counts
    
    print(f"\nSaved to: {output_file}")
    print(f"  Particles merged: {total_counts[total_counts > 0]}")
    print(f"  Physical box size: {box_size_physical:.2f} kpc")
    print('='*60)


def convert_single_file(input_path, h_value, datasets_info):
    """
    Convert a single HDF5 file to physical coordinates (remove h factor).

    This applies the same coordinate conversion as merge_snapshot_files,
    but for files that aren't split into subfiles (e.g., IC snapshots).
    Output is written next to the input with a "combined_" prefix.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"\nSkipping {input_path} - file not found")
        return

    # Build output filename: snap_ics.hdf5 → combined_ics.hdf5
    stem = input_path.stem  # e.g., "snap_ics"
    suffix = input_path.suffix  # e.g., ".hdf5"
    prefix = stem.split("_", 1)[1] if "_" in stem else stem  # e.g., "ics"
    output_path = input_path.parent / f"combined_{prefix}{suffix}"

    print(f"\n{'='*60}")
    print(f"Converting single file: {input_path}")
    print(f"Output: {output_path}")

    with h5py.File(input_path, 'r') as f_in:
        box_size = f_in['Header'].attrs.get('BoxSize')
        box_size_physical = box_size / h_value
        print(f"  BoxSize: {box_size:.2f} ckpc/h → {box_size_physical:.2f} ckpc")

        with h5py.File(output_path, 'w') as f_out:
            # Copy metadata groups
            for group_name in f_in:
                if group_name not in datasets_info:
                    f_out.copy(f_in[group_name], group_name)

            # Update box size to physical units
            f_out["Header"].attrs["BoxSize"] = box_size_physical

            # Copy and convert particle data
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
    """Process all requested snapshots and single files."""
    print("\nHDF5 Snapshot Merger")
    print(f"Base directory: {BASE_DIR}")
    print(f"Snapshots to process: {SNAPSHOT_NUMBERS}")
    print(f"Single files to convert: {SINGLE_FILES}")

    for snapshot_num in SNAPSHOT_NUMBERS:
        merge_snapshot_files(
            snapshot_num,
            NUM_SUBFILES,
            BASE_DIR,
            H_VALUE,
            DATASETS_INFO
        )

    for single_file in SINGLE_FILES:
        convert_single_file(
            Path(BASE_DIR) / single_file,
            H_VALUE,
            DATASETS_INFO
        )

    print("\nDone!")


if __name__ == "__main__":
    main()