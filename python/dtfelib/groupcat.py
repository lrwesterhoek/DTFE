"""FoF/Subfind group catalogs + subhalo member particles, without offset files.

Reads the fof_subhalo_tab_NNN.X.hdf5 chunks downloaded by `download_snapshots.sh -c` into
<DATA_ROOT>/<sim>/groups_NNN/, and (for shape/orientation work) the raw snapshot chunks in
snapdir_NNN/ -- TNG snapshots store particles in FoF-group order (each group's subhalo
particles first, catalog order, then the group's "fuzz"), so a subhalo's particle range is
computable from GroupLenType / GroupNsubs / GroupFirstSub / SubhaloLenType alone:

    offset(sub s) = groupOffset(GrNr(s)) + sum SubhaloLenType[firstSub(GrNr) : s]

All returned quantities keep RAW TNG units (ckpc/h, 1e10 Msun/h) unless stated otherwise.
The merged combined_fof_subhalo_tab_NNN.hdf5 files (merge_HDF5.py --groupcats) store h-FREE
values on disk; their per-dataset 'divided_by_h' / header 'HFreeUnits' markers are used here
to restore the raw-unit API, so chunked and merged reads return the same numbers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

from .cli import DATA_ROOT

DM = 1   # PartType1 (the only species in the -Dark runs)


def groupcat_dir(sim: str, snap: int) -> Path:
    return DATA_ROOT / sim / f"groups_{snap:03d}"


def _chunks(d: Path, pattern: str) -> list[Path]:
    files = sorted(d.glob(pattern), key=lambda p: int(p.stem.split(".")[-1]))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {d} (download with download_snapshots.sh -c?)")
    return files


def _groupcat_files(sim: str, snap: int) -> list[Path]:
    """The merged catalog (merge_HDF5.py --groupcats) when present, else the raw chunks --
    the merge concatenates in chunk order, so both read identically."""
    d = groupcat_dir(sim, snap)
    combined = d / f"combined_fof_subhalo_tab_{snap:03d}.hdf5"
    if combined.exists():
        return [combined]
    return _chunks(d, f"fof_subhalo_tab_{snap:03d}.*.hdf5")


def _raw_units(dset, data):
    """Merged files store h-free values; restore the documented raw ckpc/h API."""
    factor = dset.attrs.get("divided_by_h")
    return data if factor is None else data * factor


def header(sim: str, snap: int) -> dict:
    with h5py.File(_groupcat_files(sim, snap)[0], "r") as f:
        h = f["Header"].attrs
        box = float(h["BoxSize"])
        if "HFreeUnits" in h:               # merged catalog: BoxSize on disk is h-free ckpc
            box *= float(h["HFreeUnits"])
        return {"redshift": float(h["Redshift"]), "hubble": float(h["HubbleParam"]),
                "box_ckpc_h": box, "time": float(h["Time"])}


def load(sim: str, snap: int, subhalo_fields=(), group_fields=()) -> dict:
    """Concatenate the requested Subhalo/Group columns across all catalog chunks."""
    out = {f"Subhalo/{n}": [] for n in subhalo_fields}
    out.update({f"Group/{n}": [] for n in group_fields})
    for p in _groupcat_files(sim, snap):
        with h5py.File(p, "r") as f:
            for key in out:
                grp, name = key.split("/")
                if grp in f and name in f[grp]:
                    out[key].append(_raw_units(f[grp][name], f[grp][name][:]))
    return {k.split("/")[1]: (np.concatenate(v) if v else np.empty(0)) for k, v in out.items()}


@lru_cache(maxsize=8)
def _membership(sim: str, snap: int):
    """(groupOffsets, GroupFirstSub, SubhaloGrNr, SubhaloLenType) for the DM component."""
    cat = load(sim, snap,
               subhalo_fields=("SubhaloLenType", "SubhaloGrNr"),
               group_fields=("GroupLenType", "GroupNsubs", "GroupFirstSub"))
    g_len = cat["GroupLenType"][:, DM].astype(np.int64)
    g_off = np.concatenate([[0], np.cumsum(g_len)[:-1]])
    return g_off, cat["GroupFirstSub"].astype(np.int64), \
           cat["SubhaloGrNr"].astype(np.int64), cat["SubhaloLenType"][:, DM].astype(np.int64)


def subhalo_particle_range(sim: str, snap: int, subfind_id: int) -> tuple[int, int]:
    """(global_offset, length) of the subhalo's DM particles in the group-ordered snapshot."""
    g_off, first_sub, grnr, s_len = _membership(sim, snap)
    g = grnr[subfind_id]
    first = first_sub[g]
    # Subfind guarantees a group's subhalos occupy the contiguous index range starting at
    # GroupFirstSub (verified 100% on TNG); a violation means a corrupt/partial groupcat.
    if not (0 <= first <= subfind_id):
        raise ValueError(f"{sim} snap {snap}: subhalo {subfind_id} outside its group's "
                         f"contiguous range (GroupFirstSub={first}, GrNr={g}) -- "
                         f"corrupt or incomplete group catalog download?")
    off = int(g_off[g] + s_len[first:subfind_id].sum())
    return off, int(s_len[subfind_id])


def read_subhalo_coordinates(sim: str, snap: int, subfind_id: int,
                             max_particles: int = 2_000_000) -> np.ndarray:
    """Comoving DM Coordinates (ckpc/h) of one subhalo, read across raw snapshot chunks.

    Subsamples deterministically above max_particles (plenty for a shape tensor).
    """
    off, length = subhalo_particle_range(sim, snap, subfind_id)
    if length == 0:
        return np.empty((0, 3), dtype=np.float32)

    snapdir = DATA_ROOT / sim / f"snapdir_{snap:03d}"
    # raw chunks when available; else the merged combined_NNN.hdf5 (its particles keep the
    # chunk concatenation order, so the global offsets are identical). NOTE: the combined
    # file is h-FREE (ckpc) while this function's contract is raw ckpc/h -- rescale by h.
    h_rescale = 1.0
    try:
        files = _chunks(snapdir, f"snap_{snap:03d}.*.hdf5")
    except FileNotFoundError:
        combined = snapdir / f"combined_{snap:03d}.hdf5"
        if not combined.exists():
            raise
        files = [combined]
        with h5py.File(combined, "r") as f:
            h_rescale = float(f["Header"].attrs["HubbleParam"])
    parts = []
    remaining_off, remaining = off, length
    for p in files:
        with h5py.File(p, "r") as f:
            n = int(f["Header"].attrs["NumPart_ThisFile"][DM])
            if remaining_off >= n:
                remaining_off -= n
                continue
            take = min(n - remaining_off, remaining)
            parts.append(f[f"PartType{DM}"]["Coordinates"][remaining_off:remaining_off + take])
            remaining -= take
            remaining_off = 0
            if remaining == 0:
                break
    if remaining:
        raise IOError(f"snapshot chunks of {sim} snap {snap} ended {remaining} particles early "
                      f"(incomplete download?)")
    coords = np.concatenate(parts).astype(np.float64) * h_rescale
    if coords.shape[0] > max_particles:
        step = coords.shape[0] // max_particles + 1
        coords = coords[::step]
    return coords


def shape_from_coords(coords: np.ndarray, center: np.ndarray, box_ckpc_h: float) -> dict:
    """Mass-tensor shape of an equal-mass particle set around `center` (periodic min-image).

    Returns axis lengths a>=b>=c (relative), ratios b/a, c/a, triaxiality T, and the unit
    eigenvectors (rows: major, intermediate, minor).
    """
    d = coords - np.asarray(center, dtype=np.float64)
    d -= box_ckpc_h * np.rint(d / box_ckpc_h)
    m = (d[:, :, None] * d[:, None, :]).mean(axis=0)     # 3x3 mass tensor
    evals, evecs = np.linalg.eigh(m)                     # ascending
    evals = np.clip(evals, 0, None)
    c_, b_, a_ = np.sqrt(evals)                          # axis lengths ~ sqrt(eigenvalues)
    axes = evecs.T[::-1]                                 # rows: major, intermediate, minor
    # fix sign convention so orientation vectors are comparable across snapshots
    for v in axes:
        if v[np.argmax(np.abs(v))] < 0:
            v *= -1
    T = (a_**2 - b_**2) / (a_**2 - c_**2) if a_ > c_ else np.nan
    return {"abc": (float(a_), float(b_), float(c_)),
            "b_over_a": float(b_ / a_) if a_ > 0 else np.nan,
            "c_over_a": float(c_ / a_) if a_ > 0 else np.nan,
            "triaxiality": float(T),
            "axes": axes}


# ------------------------------------------------------------------ redshift table
@lru_cache(maxsize=4)
def redshift_table(sim: str) -> dict[int, float]:
    """snap -> redshift from every header available on disk (groupcats, snapshots, combined).

    Snapshots without any file get a log-a interpolation between the nearest anchors
    (TNG outputs are near-uniform in log a, good to ~1%)."""
    root = DATA_ROOT / sim
    anchors: dict[int, float] = {}
    for d in sorted(root.glob("groups_*")):
        try:
            snap = int(d.name.split("_")[1])
            anchors[snap] = header(sim, snap)["redshift"]
        except (ValueError, FileNotFoundError):
            pass
    for d in sorted(root.glob("snapdir_*")):
        snap = int(d.name.split("_")[1])
        if snap in anchors:
            continue
        for pat in (f"snap_{snap:03d}.*.hdf5", f"combined_{snap:03d}.hdf5"):
            files = sorted(d.glob(pat))
            if files:
                with h5py.File(files[0], "r") as f:
                    anchors[snap] = float(f["Header"].attrs["Redshift"])
                break
    if len(anchors) >= 2:
        snaps = np.array(sorted(anchors))
        loga = np.log(1.0 / (1.0 + np.array([anchors[s] for s in snaps])))
        full = np.arange(snaps.min(), 100)
        interp = np.interp(full, snaps, loga)
        return {int(s): float(np.exp(-la) - 1.0) for s, la in zip(full, interp)}
    return anchors
