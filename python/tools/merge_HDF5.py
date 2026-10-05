"""Merge per-snapshot TNG chunk files into a single h-free combined_NNN.hdf5, and merge the
other downloaded chunked products so their subfiles can be deleted to save space:

  snapshots (default)  snapdir_NNN/snap_NNN.X.hdf5      -> combined_NNN.hdf5
  --groupcats          groups_NNN/fof_subhalo_tab_NNN.X.hdf5
                                                        -> combined_fof_subhalo_tab_NNN.hdf5
  --trees              Merger Trees/tree_extended.X.hdf5 -> combined_tree_extended.hdf5
                       (SubLink trees are wholly contained in a chunk and stored
                       contiguously, so concatenating chunks in order preserves the
                       row(id) = row(known) + (id - SubhaloID[known]) navigation arithmetic)
  --ics                ics.hdf5 (or snap_ics.hdf5)       -> combined_ics.hdf5 (via
                       convert_ic_units.py -- the PS-DTFE --lagrangianInput file)

UNITS: every combined_* file is h-FREE (ckpc, 1e10 Msun, ckpc km/s), one convention across
all input data. Converted quantities: snapshot Coordinates, header BoxSize and MassTable;
every groupcat/tree length, mass and spin column per the curated _H_DIVIDE map below (raw
TNG stores them as ckpc/h / 1e10 Msun/h / ckpc/h km/s). Velocities (km/s), counts, indices
and IDs carry no h. Self-describing markers record the conversion: each converted dataset
gets a 'divided_by_h' attribute (= h) and each file an 'HFreeUnits' attribute, which is how
dtfelib.groupcat/trees restore their documented RAW-unit API when reading merged files, and
how FieldSet distinguishes new headers from pre-conversion combined snapshots. A column not
in either unit map is copied raw with a LOUD warning -- classify it before trusting it.

dtfelib.groupcat / dtfelib.trees / the DTFE run scripts all PREFER the combined files when
present, so after a successful merge the chunk files are redundant: pass --delete-chunks to
remove them (only after the merged row counts verify against the header totals).
download_snapshots.sh skips re-downloading anything whose combined product exists.

h is read from the input file's HubbleParam header (fallback: --hubble).

Usage:
    python3 merge_HDF5.py                                    # snapshot defaults below
    python3 merge_HDF5.py -d /path/to/TNG50-4-Dark 8 17 33   # merge snapshots 8, 17, 33
    python3 merge_HDF5.py -d DIR --groupcats 0 50 99         # groupcats for snaps 0, 50, 99
    python3 merge_HDF5.py -d DIR --trees --ics --delete-chunks
"""

import argparse
import os
import subprocess
import sys

import h5py
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))  # python/ on sys.path, for dtfelib.cli.sim_dir


SNAPSHOT_NUMBERS = [0, 4, 17, 33, 50]

NUM_SUBFILES = 4

# default simulation dir; follows the project-wide DTFE_DATA_ROOT convention and layouts (see
# config.sh / dtfelib.cli.sim_dir), override per run with -d/--data-dir
from dtfelib.cli import sim_dir
BASE_DIR = str(sim_dir("TNG300-3-Dark"))

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

# Groupcat/tree columns whose raw TNG units carry an h (lengths ckpc/h, masses 1e10 Msun/h,
# spin ckpc/h km/s): divided by h at merge. Curated against the TNG data specification for
# the -Dark column set (groupcats and SubLink extended trees share these names).
_H_DIVIDE = {
    # lengths
    "GroupCM", "GroupPos", "Group_R_Crit200", "Group_R_Crit500", "Group_R_Mean200",
    "Group_R_TopHat200", "SubhaloCM", "SubhaloPos", "SubhaloHalfmassRad",
    "SubhaloHalfmassRadType", "SubhaloVmaxRad",
    # masses
    "GroupMass", "GroupMassType", "Group_M_Crit200", "Group_M_Crit500", "Group_M_Mean200",
    "Group_M_TopHat200", "SubhaloMass", "SubhaloMassInHalfRad", "SubhaloMassInHalfRadType",
    "SubhaloMassInMaxRad", "SubhaloMassInMaxRadType", "SubhaloMassInRad",
    "SubhaloMassInRadType", "SubhaloMassType", "Mass", "MassHistory",
    # spin
    "SubhaloSpin",
}
# h-less columns: velocities (km/s), dispersions, counts, lengths-in-particles, indices, IDs.
_H_NONE = {
    "GroupFirstSub", "GroupLen", "GroupLenType", "GroupNsubs", "GroupVel", "SubhaloGrNr",
    "SubhaloIDMostbound", "SubhaloLen", "SubhaloLenType", "SubhaloParent", "SubhaloVel",
    "SubhaloVelDisp", "SubhaloVmax", "SubhaloFlag",
    "DescendantID", "FirstProgenitorID", "FirstSubhaloInFOFGroupID", "LastProgenitorID",
    "MainLeafProgenitorID", "NextProgenitorID", "NextSubhaloInFOFGroupID", "NumParticles",
    "RootDescendantID", "SnapNum", "SubfindID", "SubhaloID", "SubhaloIDRaw", "TreeID",
}


def _sorted_chunks(directory, pattern):
    """Chunk files sorted by their trailing index -- concatenation order defines the global
    particle/row offsets, so it must match the chunk numbering."""
    return sorted(Path(directory).glob(pattern), key=lambda p: int(p.stem.split(".")[-1]))


class ChunkError(RuntimeError):
    """A chunk file that cannot be read -- a download stopped half-way, or a file that is not HDF5."""


# what HDF5 says about a file whose bytes stop early, and the size below which a file that does not
# open cannot be a chunk at all (a real chunk's header alone is many KB) -- download_snapshots.sh's
# prune_incomplete uses THE SAME two rules; keep them in step
_INCOMPLETE = ("truncated file", "file signature not found", "bad byte number")
_MIN_CHUNK_BYTES = 4096


def _is_incomplete(path, exc) -> bool:
    why = str(exc).splitlines()[0]
    if any(k in why for k in _INCOMPLETE):
        return True
    try:
        return Path(path).stat().st_size < _MIN_CHUNK_BYTES
    except OSError:
        return False


def _chunk_error(path, exc):
    why = str(exc).splitlines()[0]
    name = Path(path).name
    if _is_incomplete(path, exc):
        return ChunkError(f"Error: couldn't read {name} (an incomplete download): {why}\n"
                          "  No combined file was written. Run the download again: it removes an incomplete "
                          "chunk and fetches it anew.")
    return ChunkError(f"Error: couldn't read {name} (it cannot be read): {why}\n"
                      f"  No combined file was written. Delete {name} (or fix its permissions) and run the "
                      "download again.")


def _open_chunk(path):
    """h5py.File(path, 'r'), or a ChunkError that names the chunk and says what to do."""
    try:
        return h5py.File(path, "r")
    except OSError as e:
        raise _chunk_error(path, e) from e


def _expected_chunks(found, attr):
    """The chunk count the header of the first chunk found promises ('NumFilesPerSnapshot' for
    snapshots, 'NumFiles' for group catalogues), or None when there is no chunk or no such attribute.
    Counting the files on disk instead let a download that stopped early merge into a 'complete'
    file with missing particles (2026-10-06)."""
    if not found:
        return None
    with _open_chunk(found[0]) as f:
        v = f["Header"].attrs.get(attr) if "Header" in f else None
    return int(v) if v is not None else None


def _report_missing(input_files, n_header):
    missing = [f for f in input_files if not f.exists()]
    print(f"\nProblem: Can't find {len(missing) or 'any'} subfile(s)"
          + (f" (the header promises {n_header})" if n_header else "") + ":")
    for f in missing:
        print(f"  Missing: {f}")
    print("  No combined file was written. Run the download again (it fetches what is missing), "
          "or check -d / -n.\n")


def _partial(output_file):
    """The name a merge writes under until its totals verify: a stopped or failed merge never leaves
    a combined_*.hdf5 that the launcher and the run scripts would take for a complete snapshot."""
    partial = Path(output_file).with_name(Path(output_file).name + ".partial")
    partial.unlink(missing_ok=True)             # a merge stopped half-way last time
    return partial


def _delete_chunks(files, verified, what):
    if not verified:
        print(f"  NOT deleting {what} chunks: the merged totals did not verify.")
        return
    for f in files:
        f.unlink()
    print(f"  Deleted {len(files)} {what} chunk file(s).")


def merge_snapshot_files(snapshot_num, num_subfiles, base_dir, h_value, datasets_info,
                         delete_chunks=False):
    snapshot_str = f"{snapshot_num:03d}"

    snapshot_dir = Path(base_dir) / f"snapdir_{snapshot_str}"
    output_file = snapshot_dir / f"combined_{snapshot_str}.hdf5"

    print(f"\n{'='*60}")
    print(f"Processing snapshot {snapshot_str}")
    print(f"Directory: {snapshot_dir}")
    try:
        ok = _merge_snapshot(snapshot_str, snapshot_dir, output_file, num_subfiles, h_value, datasets_info,
                             delete_chunks)
    except ChunkError as e:
        print(f"\n{e}")
        ok = False
    print('='*60)
    return ok


def _merge_snapshot(snapshot_str, snapshot_dir, output_file, num_subfiles, h_value, datasets_info,
                    delete_chunks):
    """merge_snapshot_files' body; True when combined_NNN.hdf5 was written and verified (or is there
    already and its chunks are gone: a merge with --delete-chunks run again)."""
    partial = _partial(output_file)         # first: a merge stopped half-way may have left a big one
    found = _sorted_chunks(snapshot_dir, f"snap_{snapshot_str}.*.hdf5")
    if not found and output_file.exists():
        print(f"  {output_file.name} already exists and no chunks are left: nothing to merge")
        return True
    n_header = _expected_chunks(found, "NumFilesPerSnapshot")
    if num_subfiles:
        if n_header is not None and n_header != num_subfiles:
            print(f"  WARNING: -n {num_subfiles} given, but the header says {n_header} chunk(s) per snapshot")
        n_files = num_subfiles
    else:   # the header's count; the files on disk only when the header has none
        n_files = n_header or len(found)
    input_files = [snapshot_dir / f"snap_{snapshot_str}.{i}.hdf5" for i in range(n_files)]
    print(f"Merging {len(input_files)} subfiles...")

    if not input_files or any(not f.exists() for f in input_files):
        _report_missing(input_files, n_header)
        return False

    with _open_chunk(input_files[0]) as f_first:
        box_size = f_first['Header'].attrs.get('BoxSize')
        box_size_physical = box_size / h_value

    try:
        total_counts, expected, types = _write_snapshot(partial, input_files, h_value, datasets_info,
                                                        box_size_physical)
    except ChunkError:
        partial.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError) as e:    # the disk, not a chunk (full, gone, read-only); h5py's close raises RuntimeError
        partial.unlink(missing_ok=True)
        print(f"\nError: could not write {partial}: {e}\n  No combined file was written.")
        return False
    except Exception:                       # whatever else: never leave a .partial behind
        partial.unlink(missing_ok=True)
        raise
    verified = all(int(expected[t]) == int(total_counts[t]) for t in types)
    print(f"\n  Particles merged: {total_counts[total_counts > 0]}"
          + ("" if verified else f"  (MISMATCH vs header total {expected[sorted(types)]})"))
    if not verified:
        partial.unlink(missing_ok=True)
        print("  No combined file was written: the chunks do not hold what the header promises. Run the "
              "download again, then the merge.")
        return False
    os.replace(partial, output_file)
    print(f"Saved to: {output_file}")
    print(f"  Physical box size: {box_size_physical:.2f} kpc")
    if delete_chunks:
        _delete_chunks(input_files, verified, "snapshot")
    return True


def _write_snapshot(output_file, input_files, h_value, datasets_info, box_size_physical):
    """The merged file at 'output_file'. Returns (counts merged per type, the header's totals, the
    particle types merged). A chunk that fails to read raises ChunkError (nothing is skipped)."""
    with h5py.File(output_file, 'w') as f_out:
        with _open_chunk(input_files[0]) as f_first:
            for group_name in ["Config", "Parameters"]:
                if group_name in f_first:
                    f_out.copy(f_first[group_name], group_name)

            if "Header" in f_first:
                f_out.copy(f_first["Header"], "Header")

        f_out["Header"].attrs["BoxSize"] = box_size_physical
        # MassTable is 1e10 Msun/h in the raw chunks: convert like everything else and mark
        # the file, so FieldSet can tell h-free headers from pre-conversion combined files
        f_out["Header"].attrs["MassTable"] = np.asarray(f_out["Header"].attrs["MassTable"]) / h_value
        f_out["Header"].attrs["HFreeUnits"] = h_value

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

            with _open_chunk(fname) as f_in:
                for group_name, dsets in datasets_info.items():
                    for dset_name in dsets:
                        try:                        # a chunk truncated past its superblock fails here, not at open
                            have = group_name in f_in and dset_name in f_in[group_name]
                            data = f_in[group_name][dset_name][...] if have else None
                        except (OSError, RuntimeError) as e:
                            raise _chunk_error(fname, e) from e
                        if have:
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

        header = f_out["Header"]
        # chunk-0 global counts; TNG splits them into a low and a high 32-bit word (the -1 runs exceed 2^32)
        expected = np.array(header.attrs["NumPart_Total"], dtype=np.int64)
        if "NumPart_Total_HighWord" in header.attrs:
            expected += np.array(header.attrs["NumPart_Total_HighWord"], dtype=np.int64) << 32
            header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)   # the totals below are int64
        header.attrs["NumPart_ThisFile"] = total_counts
        header.attrs["NumPart_Total"] = total_counts
    types = {int(g[-1]) for g in datasets_info}       # a merged type with 0 particles must verify too
    return total_counts, expected, types


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
            if "MassTable" in f_out["Header"].attrs:
                f_out["Header"].attrs["MassTable"] = np.asarray(f_out["Header"].attrs["MassTable"]) / h_value
            f_out["Header"].attrs["HFreeUnits"] = h_value

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


def _concat_datasets(f_out, out_group, columns, chunk_files, h_value, group_path=None):
    """Concatenate `columns` ({name: (shape_tail, dtype)}) across `chunk_files` in order,
    dividing the _H_DIVIDE columns by h (computed in float64, stored in the source dtype;
    marker attribute 'divided_by_h' = h). Reads whole columns per chunk (chunks are at most
    a few hundred MB per column). Returns {name: rows_written}."""
    dsets, written, convert = {}, {}, {}
    for name, (tail, dtype) in columns.items():
        dsets[name] = out_group.create_dataset(name, shape=(0,) + tail,
                                               maxshape=(None,) + tail,
                                               dtype=dtype, chunks=True)
        written[name] = 0
        convert[name] = name in _H_DIVIDE
        if convert[name] and not np.issubdtype(np.dtype(dtype), np.floating):
            print(f"  WARNING: {name} is marked for h-conversion but stored as {dtype}; copied raw")
            convert[name] = False
        elif convert[name]:
            dsets[name].attrs["divided_by_h"] = h_value
        elif name not in _H_NONE:
            print(f"  WARNING: column {name} is in neither unit map -- copied RAW (ckpc/h?); "
                  f"classify it in merge_HDF5.py before trusting its values")
    for idx, path in enumerate(chunk_files):
        print(f"  [{idx + 1}/{len(chunk_files)}] {path.name}")
        with _open_chunk(path) as f_in:
            src = f_in.get(group_path) if group_path else f_in
            if src is None:
                continue        # a chunk with zero groups/subhalos has no such group at all
            for name, d in dsets.items():
                if name not in src:
                    continue    # a chunk with zero rows may omit the dataset entirely
                try:
                    data = src[name][...]
                except OSError as e:        # truncated past the superblock
                    raise _chunk_error(path, e) from e
                if convert[name]:
                    data = (data.astype(np.float64) / h_value).astype(d.dtype)
                d.resize((written[name] + data.shape[0],) + d.shape[1:])
                d[written[name]:, ...] = data
                written[name] += data.shape[0]
    return written


def _collect_columns(chunk_files, group_path=None):
    """{name: (shape_tail, dtype)} union across chunks (zero-row chunks omit datasets)."""
    columns = {}
    for path in chunk_files:
        with _open_chunk(path) as f:
            src = f.get(group_path) if group_path else f
            if src is None:
                continue
            for name, obj in src.items():
                if isinstance(obj, h5py.Dataset) and name not in columns:
                    columns[name] = (obj.shape[1:], obj.dtype)
        if columns:
            break   # the first chunk that has the group defines it; later ones only append
    return columns


def merge_groupcat_files(snapshot_num, base_dir, delete_chunks=False):
    """groups_NNN/fof_subhalo_tab_NNN.X.hdf5 -> combined_fof_subhalo_tab_NNN.hdf5, h-FREE
    (lengths/masses/spins and the header BoxSize divided by h, marker attributes set);
    dtfelib.groupcat restores its documented raw-ckpc/h API from the markers when reading."""
    snapshot_str = f"{snapshot_num:03d}"
    group_dir = Path(base_dir) / f"groups_{snapshot_str}"
    output_file = group_dir / f"combined_fof_subhalo_tab_{snapshot_str}.hdf5"
    print(f"\n{'='*60}")
    try:
        ok = _merge_groupcat(snapshot_str, group_dir, output_file, delete_chunks)
    except ChunkError as e:
        print(f"\n{e}")
        ok = False
    print('='*60)
    return ok


def _merge_groupcat(snapshot_str, group_dir, output_file, delete_chunks):
    partial = _partial(output_file)
    found = _sorted_chunks(group_dir, f"fof_subhalo_tab_{snapshot_str}.*.hdf5")
    if not found and output_file.exists():
        print(f"  {output_file.name} already exists and no chunks are left: nothing to merge")
        return True
    n_header = _expected_chunks(found, "NumFiles")
    chunk_files = ([group_dir / f"fof_subhalo_tab_{snapshot_str}.{i}.hdf5" for i in range(n_header)]
                   if n_header else found)
    print(f"Merging group catalog {snapshot_str}: {len(chunk_files)} chunk(s)")
    if not chunk_files or any(not f.exists() for f in chunk_files):
        _report_missing(chunk_files, n_header)
        return False

    try:
        rows, n_g, n_s, expected_g, expected_s = _write_groupcat(partial, chunk_files)
    except ChunkError:
        partial.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError) as e:    # the disk, not a chunk
        partial.unlink(missing_ok=True)
        print(f"\nError: could not write {partial}: {e}\n  No combined file was written.")
        return False
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    verified = (n_g == expected_g and n_s == expected_s
                and all(v in (0, n_g) for v in rows["Group"].values())
                and all(v in (0, n_s) for v in rows["Subhalo"].values()))
    print(f"  Groups: {n_g}/{expected_g}  Subhalos: {n_s}/{expected_s}  "
          f"({'verified' if verified else 'MISMATCH'})")
    if not verified:
        partial.unlink(missing_ok=True)
        print("  No combined file was written: the chunks do not hold what the header promises.")
        return False
    os.replace(partial, output_file)
    print(f"  Saved: {output_file}")
    if delete_chunks:
        _delete_chunks(chunk_files, verified, "group catalog")
    return True


def _write_groupcat(output_file, chunk_files):
    with h5py.File(output_file, "w") as f_out:
        with _open_chunk(chunk_files[0]) as f_first:
            for name in f_first:
                if name not in ("Group", "Subhalo"):
                    f_out.copy(f_first[name], name)     # Header, Config, Parameters, IDs, ...
            expected_g = int(f_first["Header"].attrs["Ngroups_Total"])
            expected_s = int(f_first["Header"].attrs["Nsubgroups_Total"])
            h_value = float(f_first["Header"].attrs["HubbleParam"])
        rows = {}
        for grp in ("Group", "Subhalo"):
            columns = _collect_columns(chunk_files, grp)
            rows[grp] = _concat_datasets(f_out, f_out.create_group(grp), columns,
                                         chunk_files, h_value, grp) if columns else {}
        n_g = max(rows["Group"].values(), default=0)
        n_s = max(rows["Subhalo"].values(), default=0)
        f_out["Header"].attrs["Ngroups_ThisFile"] = n_g
        f_out["Header"].attrs["Nsubgroups_ThisFile"] = n_s
        f_out["Header"].attrs["NumFiles"] = 1
        f_out["Header"].attrs["BoxSize"] = f_out["Header"].attrs["BoxSize"] / h_value
        f_out["Header"].attrs["HFreeUnits"] = h_value
    return rows, n_g, n_s, expected_g, expected_s


def _find_h(base_dir, fallback):
    """HubbleParam from any header on disk (tree chunks carry no cosmology header)."""
    base = Path(base_dir)
    candidates = (list(base.glob("groups_*/fof_subhalo_tab_*.hdf5"))
                  + list(base.glob("snapdir_*/combined_*.hdf5"))
                  + list(base.glob("snapdir_*/snap_*.hdf5"))
                  + list(base.glob("combined_ics.hdf5")) + list(base.glob("ics.hdf5")))
    for path in candidates:
        try:
            with h5py.File(path, "r") as f:
                return float(f["Header"].attrs["HubbleParam"])
        except (OSError, KeyError):
            continue
    print(f"  WARNING: no header with HubbleParam under {base_dir}; using h = {fallback}")
    return fallback


def merge_tree_files(base_dir, delete_chunks=False, h_fallback=H_VALUE):
    """tree_extended.X.hdf5 -> combined_tree_extended.hdf5 (same directory), h-FREE (mass,
    length and spin columns divided by h with marker attributes; dtfelib.trees restores its
    documented raw-unit API from the markers). Each SubLink tree is wholly contained in one
    chunk and stored contiguously, so concatenating the chunks in index order preserves the
    within-tree row arithmetic dtfelib.trees relies on."""
    base = Path(base_dir)
    tree_dir = None
    cands = (base / "Merger Trees", base / "postprocessing" / "trees" / "SubLink")
    for cand in cands:
        if cand.is_dir() and any(cand.glob("tree_extended.*.hdf5")):
            tree_dir = cand
            break
    print(f"\n{'='*60}")
    if tree_dir is None:
        done = next((c / "combined_tree_extended.hdf5" for c in cands if (c / "combined_tree_extended.hdf5").exists()), None)
        if done is not None:            # merged with --delete-chunks earlier: nothing left to do
            print(f"Merging trees: {done} already exists and no chunks are left: nothing to merge")
            print('='*60)
            return True
        print(f"Merging trees: no tree_extended.*.hdf5 under {base} (download_snapshots.sh -t), nothing merged")
        print('='*60)
        return False
    chunk_files = _sorted_chunks(tree_dir, "tree_extended.*.hdf5")
    output_file = tree_dir / "combined_tree_extended.hdf5"
    h_value = _find_h(base_dir, h_fallback)
    print(f"Merging SubLink trees: {len(chunk_files)} chunk(s) in {tree_dir} (h = {h_value})")
    # the tree chunks carry no count of themselves: an unreadable one is caught, a missing one is not
    partial = _partial(output_file)
    try:
        expected = 0
        for path in chunk_files:
            with _open_chunk(path) as f:
                expected += f["SubhaloID"].shape[0]
        with h5py.File(partial, "w") as f_out:
            with _open_chunk(chunk_files[0]) as f_first:
                for name, obj in f_first.items():
                    if not isinstance(obj, h5py.Dataset):
                        f_out.copy(obj, name)               # e.g. a Header group, if present
            f_out.attrs["HFreeUnits"] = h_value
            columns = _collect_columns(chunk_files)
            written = _concat_datasets(f_out, f_out, columns, chunk_files, h_value)
    except ChunkError as e:
        partial.unlink(missing_ok=True)
        print(f"\n{e}")
        print('='*60)
        return False
    except (OSError, RuntimeError) as e:    # the disk, not a chunk
        partial.unlink(missing_ok=True)
        print(f"\nError: could not write {partial}: {e}\n  No combined file was written.")
        print('='*60)
        return False
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    n = written.get("SubhaloID", 0)
    verified = n == expected and all(v == n for v in written.values())
    print(f"  Tree rows: {n}/{expected} across {len(written)} columns "
          f"({'verified' if verified else 'MISMATCH'})")
    if not verified:
        partial.unlink(missing_ok=True)
        print("  No combined file was written: the columns disagree on their row counts.")
        print('='*60)
        return False
    os.replace(partial, output_file)
    print(f"  Saved: {output_file}")
    if delete_chunks:
        _delete_chunks(chunk_files, verified, "tree")
    print('='*60)
    return True


def convert_ics(base_dir, delete_chunks=False):
    """ics.hdf5 / snap_ics.hdf5 -> combined_ics.hdf5 via convert_ic_units.py (h-free, the
    same convention as combined_NNN.hdf5 -- required for PS-DTFE ID matching)."""
    base = Path(base_dir)
    output_file = base / "combined_ics.hdf5"
    print(f"\n{'='*60}")
    if output_file.exists():
        print(f"ICs: {output_file} already exists, skipping")
        print('='*60)
        return True
    raw = next((base / n for n in ("ics.hdf5", "snap_ics.hdf5") if (base / n).exists()), None)
    if raw is None:
        print(f"ICs: no ics.hdf5 / snap_ics.hdf5 in {base} (download_snapshots.sh -i), nothing converted")
        print('='*60)
        return False
    print(f"Converting ICs: {raw} -> {output_file}")
    partial = _partial(output_file)         # the converter opens 'w-': a stale one must be gone
    rc = subprocess.run([sys.executable, str(SCRIPT_DIR / "convert_ic_units.py"),
                         str(raw), str(partial)]).returncode
    verified = rc == 0 and partial.is_file()
    if verified:                            # the final name only once the whole conversion succeeded
        os.replace(partial, output_file)
    else:
        partial.unlink(missing_ok=True)
    if delete_chunks:
        _delete_chunks([raw], verified, "IC")
    if not verified:
        print("  ICs conversion FAILED: no combined_ics.hdf5 was written")
    print('='*60)
    return verified


def fix_units(base_dir):
    """Upgrade PRE-CONVERSION combined files in place to the uniform h-free convention.

    Old combined_NNN.hdf5 / combined_ics.hdf5 already stored h-free Coordinates and BoxSize
    but copied the raw TNG MassTable (1e10 Msun/h) and carry no marker; their snapshot chunks
    are often deleted, so this fixes the header attributes in place (MassTable /= h,
    HFreeUnits set) instead of re-merging. Idempotent: marked files are skipped. Old merged
    groupcats/trees without markers store raw COLUMNS, so those are reported for a re-merge
    from their (still downloadable) chunks rather than rewritten blind."""
    base = Path(base_dir)
    fixed = skipped = 0
    targets = sorted(base.glob("snapdir_*/combined_*.hdf5"))
    targets += [p for p in (base / "combined_ics.hdf5",) if p.exists()]
    print(f"\n{'='*60}")
    print(f"fix-units: {base}")
    for path in targets:
        with h5py.File(path, "r+") as f:
            attrs = f["Header"].attrs
            if "HFreeUnits" in attrs:
                skipped += 1
                continue
            h_value = float(attrs["HubbleParam"])
            if "MassTable" in attrs:
                attrs["MassTable"] = np.asarray(attrs["MassTable"]) / h_value
            attrs["HFreeUnits"] = h_value
            fixed += 1
            print(f"  fixed {path.relative_to(base)}  (MassTable -> 1e10 Msun, marker set)")
    for path in sorted(base.glob("groups_*/combined_fof_subhalo_tab_*.hdf5")):
        with h5py.File(path, "r") as f:
            if "HFreeUnits" not in f["Header"].attrs:
                print(f"  WARNING: {path.relative_to(base)} predates the units unification "
                      f"(raw columns) -- delete it and re-merge with --groupcats")
    for sub in ("Merger Trees", "postprocessing/trees/SubLink"):
        path = base / sub / "combined_tree_extended.hdf5"
        if path.exists():
            with h5py.File(path, "r") as f:
                if "HFreeUnits" not in f.attrs:
                    print(f"  WARNING: {path.relative_to(base)} predates the units unification "
                          f"(raw columns) -- delete it and re-merge with --trees")
    print(f"  {fixed} file(s) upgraded, {skipped} already h-free")
    print('='*60)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("snapshots", type=int, nargs="*", default=None,
                    help=f"snapshot numbers to merge (default: {SNAPSHOT_NUMBERS})")
    ap.add_argument("-d", "--data-dir", default=BASE_DIR,
                    help=f"simulation directory holding snapdir_NNN/ (default: {BASE_DIR})")
    ap.add_argument("-n", "--num-subfiles", type=int, default=None,
                    help="chunk files per snapshot (default: auto-detected by globbing)")
    ap.add_argument("--hubble", type=float, default=H_VALUE,
                    help=f"fallback h if the file header lacks HubbleParam (default: {H_VALUE})")
    ap.add_argument("--snapshots-too", action="store_true",
                    help="with --groupcats/--trees/--ics: also merge the raw snapshots")
    ap.add_argument("--groupcats", action="store_true",
                    help="merge the groups_NNN chunk files of the given snapshots")
    ap.add_argument("--trees", action="store_true",
                    help="merge the SubLink tree_extended chunks (whole simulation)")
    ap.add_argument("--ics", action="store_true",
                    help="convert ics.hdf5 -> combined_ics.hdf5 (whole simulation)")
    ap.add_argument("--fix-units", action="store_true",
                    help="upgrade PRE-CONVERSION combined snapshot/ICs files in place to the "
                         "h-free convention (MassTable /h + HFreeUnits marker); idempotent")
    ap.add_argument("--delete-chunks", action="store_true",
                    help="delete the source chunk files after a VERIFIED merge")
    args = ap.parse_args()
    snapshots = args.snapshots if args.snapshots else SNAPSHOT_NUMBERS
    other_modes = args.groupcats or args.trees or args.ics or args.fix_units
    do_snapshots = ((not other_modes) or args.snapshots_too) and not args.fix_units

    print("\nHDF5 Snapshot Merger")
    print(f"Base directory: {args.data_dir}")
    if do_snapshots or args.groupcats:
        print(f"Snapshots to process: {snapshots}")

    failed = []                 # what asked for a merge and got no verified combined file
    if do_snapshots:
        for snapshot_num in snapshots:
            # h from the data itself when available; the CLI value is only a fallback
            first = Path(args.data_dir) / f"snapdir_{snapshot_num:03d}" / f"snap_{snapshot_num:03d}.0.hdf5"
            h_value = args.hubble
            if first.exists():
                try:
                    with _open_chunk(first) as f:
                        h_value = float(f["Header"].attrs.get("HubbleParam", args.hubble))
                except ChunkError as e:
                    print(f"\n{'=' * 60}\nProcessing snapshot {snapshot_num:03d}\n{e}\n{'=' * 60}")
                    failed.append(f"snapshot {snapshot_num:03d}")
                    continue
            if not merge_snapshot_files(
                snapshot_num,
                args.num_subfiles,
                args.data_dir,
                h_value,
                DATASETS_INFO,
                delete_chunks=args.delete_chunks
            ):
                failed.append(f"snapshot {snapshot_num:03d}")

        for single_file in SINGLE_FILES:
            convert_single_file(
                Path(args.data_dir) / single_file,
                args.hubble,
                DATASETS_INFO
            )

    if args.groupcats:
        for snapshot_num in snapshots:
            if not merge_groupcat_files(snapshot_num, args.data_dir, delete_chunks=args.delete_chunks):
                failed.append(f"group catalogue {snapshot_num:03d}")
    if args.trees:
        if not merge_tree_files(args.data_dir, delete_chunks=args.delete_chunks, h_fallback=args.hubble):
            failed.append("merger trees")
    if args.ics:
        if not convert_ics(args.data_dir, delete_chunks=args.delete_chunks):
            failed.append("initial conditions")
    if args.fix_units:
        fix_units(args.data_dir)

    if failed:                  # the launcher and shell chains see a failed step, not a quiet 'Done!'
        print(f"\nDone, with problems: no combined file for {', '.join(failed)}.")
        sys.exit(1)
    print("\nDone!")


if __name__ == "__main__":
    main()
