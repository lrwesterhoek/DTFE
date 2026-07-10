"""Sample DTFE / PS-DTFE grid fields at (halo) positions -- the merger-tree <-> field bridge.

Positions are converted to FRACTIONAL box coordinates using the raw-snapshot box size
(ckpc/h, the same frame as SubLink SubhaloPos), so sampling is independent of whichever
unit convention the field grid was produced under: the grid always spans the full box.

Works identically for both estimators via FieldSet (method='ps' | 'dtfe' | 'auto'):

    from dtfelib.environment import branch_environment
    env = branch_environment("TNG50-3-Dark", branch, method="dtfe",
                             fields=("density", "tweb", "vweb"))
    env["density"]   # rho/rho_bar at the halo position, one value per branch snapshot
    env["tweb"]      # web label 0..3 (NGP-sampled; labels are never interpolated)

Continuous fields are sampled trilinearly (periodic), categorical web labels by
nearest-grid-point. Snapshots without fields on disk yield NaN with a note in env["missing"].
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .cli import DATA_ROOT
from .io import FieldSet

# fields whose values are categorical labels: sample NGP, never interpolate
LABEL_FIELDS = {"tweb", "vweb"}


def box_ckpc_h(sim: str, snap: int) -> float:
    """Box size in ckpc/h (the SubLink/raw-snapshot frame) for `sim` at `snap`.

    Prefers a raw snapshot chunk or groupcat header (always ckpc/h); falls back to the
    merged combined_*.hdf5, which is h-free ckpc by the current convention (BoxSize * h).
    """
    root = DATA_ROOT / sim
    snapdir = root / f"snapdir_{snap:03d}"
    for pat in (f"snap_{snap:03d}.*.hdf5",):
        files = sorted(snapdir.glob(pat))
        if files:
            with h5py.File(files[0], "r") as f:
                return float(f["Header"].attrs["BoxSize"])
    gdir = root / f"groups_{snap:03d}"
    files = sorted(gdir.glob("fof_subhalo_tab_*.hdf5"))
    if files:
        with h5py.File(files[0], "r") as f:
            return float(f["Header"].attrs["BoxSize"])
    files = sorted(snapdir.glob(f"combined_{snap:03d}.hdf5"))
    if files:
        with h5py.File(files[0], "r") as f:
            h = f["Header"].attrs
            return float(h["BoxSize"]) * float(h["HubbleParam"])   # h-free ckpc -> ckpc/h
    raise FileNotFoundError(f"no header source for the box size of {sim} snap {snap}")


def sample_grid(grid: np.ndarray, frac: np.ndarray, mode: str = "trilinear") -> np.ndarray:
    """Sample a periodic (N,N,N) or (N,N,N,C) grid at fractional positions frac (M,3) in [0,1).

    mode='trilinear' (continuous fields) or 'ngp' (categorical labels).
    Grid cell i covers [i/N, (i+1)/N); cell centres sit at (i+0.5)/N.
    """
    n = grid.shape[0]
    frac = np.mod(np.asarray(frac, dtype=np.float64), 1.0)
    if mode == "ngp":
        idx = np.floor(frac * n).astype(np.int64) % n
        return grid[idx[:, 0], idx[:, 1], idx[:, 2]]
    if mode != "trilinear":
        raise ValueError(f"unknown sampling mode {mode!r}")

    # cell-centre aligned trilinear interpolation with periodic wrap
    x = frac * n - 0.5
    i0 = np.floor(x).astype(np.int64)
    w = x - i0                                  # (M,3) weights toward the upper neighbour
    i0 %= n
    i1 = (i0 + 1) % n
    out = None
    for dx, wx in ((i0[:, 0], 1 - w[:, 0]), (i1[:, 0], w[:, 0])):
        for dy, wy in ((i0[:, 1], 1 - w[:, 1]), (i1[:, 1], w[:, 1])):
            for dz, wz in ((i0[:, 2], 1 - w[:, 2]), (i1[:, 2], w[:, 2])):
                wgt = (wx * wy * wz)
                vals = grid[dx, dy, dz]
                if vals.ndim > 1:
                    wgt = wgt[:, None]
                out = vals * wgt if out is None else out + vals * wgt
    return out


def sample_fields_at(sim: str, snap: int, pos_ckpc_h: np.ndarray,
                     fields=("density", "tweb", "vweb"),
                     method: str = "auto", averaged: bool = True) -> dict:
    """Sample `fields` at positions (M,3) in ckpc/h for one snapshot.

    Density is returned in mean units (rho/rho_bar) for both estimators; other fields raw.
    Missing individual fields come back as NaN arrays (check FieldSet.has upstream if needed).
    """
    fs = FieldSet(DATA_ROOT / sim / f"snapdir_{snap:03d}", method=method, averaged=averaged)
    frac = np.asarray(pos_ckpc_h, dtype=np.float64) / box_ckpc_h(sim, snap)
    out = {"_method": fs.method, "_grid_n": fs.grid_n}
    for name in fields:
        if not fs.has(name):
            out[name] = np.full(frac.shape[0], np.nan)
            continue
        grid = fs.load(name)
        if name == "density":
            # mean units via the grid's OWN mean: mass conservation makes the volume mean
            # equal rho_bar in whatever units the file was written, so this is correct for
            # both estimators AND for grids from older unit conventions (unlike
            # FieldSet.density, which trusts the current on-disk convention).
            m = float(grid.mean())
            if m > 0:
                grid = grid / m
        out[name] = sample_grid(grid, frac, "ngp" if name in LABEL_FIELDS else "trilinear")
    return out


def branch_environment(sim: str, branch: dict, fields=("density", "tweb", "vweb"),
                       method: str = "auto", averaged: bool = True) -> dict:
    """Field environment along a SubLink branch (output of TreeSet.main_branch, which must
    include 'SubhaloPos'). One sampled value per branch snapshot; NaN where no field grids
    exist on disk. Returns {field: (n,) array, ..., "missing": [snap, ...], "method": {snap: m}}.
    """
    snaps = branch["SnapNum"].astype(int)
    pos = np.asarray(branch["SubhaloPos"], dtype=np.float64)
    out = {name: np.full(snaps.size, np.nan) for name in fields}
    out["missing"] = []
    out["method"] = {}
    for i, s in enumerate(snaps):
        try:
            smp = sample_fields_at(sim, int(s), pos[i:i + 1], fields, method, averaged)
        except (FileNotFoundError, ValueError):
            out["missing"].append(int(s))
            continue
        out["method"][int(s)] = smp["_method"]
        for name in fields:
            vec = smp[name]
            out[name][i] = float(vec[0]) if np.ndim(vec[0]) == 0 else np.nan
    return out
