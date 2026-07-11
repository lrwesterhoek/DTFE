"""SubLink merger-tree access (self-contained; no illustris_python or offset files needed).

Works directly on the `tree_extended.X.hdf5` chunks (e.g. in "<sim>/Merger Trees/"). Key ideas:

  * A subhalo at (snap, subfind_id) is located via SubhaloIDRaw = snap*10^12 + subfind_id
    by scanning the (cached) SubhaloIDRaw columns -- no offset files required.
  * Within a chunk, each tree is stored contiguously and SubhaloID is assigned sequentially
    in storage order, so the row of any tree-internal ID is
        row(id) = row(known) + (id - SubhaloID[known])
    which makes walking FirstProgenitorID links O(1) per hop (the standard SubLink trick).
  * The extended trees carry the full group-catalog columns per node (SubhaloMass, SubhaloPos,
    SubhaloHalfmassRad, SubhaloSpin, Group_R_Crit200, ...), so catalog-level tracking needs
    no separate group catalogs at all.

Units are RAW TNG tree units: lengths ckpc/h (comoving), masses 1e10 Msun/h, spins ckpc/h km/s.
(The merged combined_tree_extended.hdf5 stores h-FREE values on disk; its per-dataset
'divided_by_h' markers are applied on read so the API returns raw units either way.)

    ts = TreeSet("TNG50-3-Dark")
    branch = ts.main_branch(99, 0, fields=["SnapNum", "SubhaloMass", "SubhaloPos"])
    top = ts.subhalos_at(99, order_by="SubhaloMass", n=10)   # selection without a groupcat
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .cli import DATA_ROOT

_RAW_FACTOR = 10 ** 12   # SubhaloIDRaw = SnapNum * 1e12 + SubfindID


def _tree_dir(sim: str) -> Path:
    root = DATA_ROOT / sim
    for cand in (root / "Merger Trees", root / "postprocessing" / "trees" / "SubLink"):
        if cand.is_dir() and ((cand / "combined_tree_extended.hdf5").exists()
                              or any(cand.glob("tree_extended.*.hdf5"))):
            return cand
    raise FileNotFoundError(
        f"no SubLink tree_extended.*.hdf5 (or combined_tree_extended.hdf5) found under {root} "
        f"(looked in 'Merger Trees/' and 'postprocessing/trees/SubLink/')")


class TreeSet:
    """All SubLink tree chunks of one simulation."""

    def __init__(self, sim: str):
        self.sim = sim
        self.dir = _tree_dir(sim)
        # the merged file (merge_HDF5.py --trees) behaves as one big chunk: trees are wholly
        # contained and stored contiguously, so the row arithmetic below is unaffected
        combined = self.dir / "combined_tree_extended.hdf5"
        if combined.exists():
            self.paths = [combined]
        else:
            self.paths = sorted(self.dir.glob("tree_extended.*.hdf5"),
                                key=lambda p: int(p.stem.split(".")[-1]))
        self._files: dict[int, h5py.File] = {}
        self._id_cols: dict[tuple[int, str], np.ndarray] = {}   # (chunk, colname) -> array

    # ------------------------------------------------------------------ low level
    def _file(self, chunk: int) -> h5py.File:
        if chunk not in self._files:
            self._files[chunk] = h5py.File(self.paths[chunk], "r")
        return self._files[chunk]

    @staticmethod
    def _raw(dset, data):
        """Merged files store h-free values; restore the documented raw-unit API."""
        factor = dset.attrs.get("divided_by_h")
        return data if factor is None else data * factor

    def _read(self, chunk: int, name: str, sel=None) -> np.ndarray:
        f = self._file(chunk)
        return self._raw(f[name], f[name][:][sel] if sel is not None else f[name][:])

    def _col(self, chunk: int, name: str) -> np.ndarray:
        """Whole-column cache for the small int columns used in navigation."""
        key = (chunk, name)
        if key not in self._id_cols:
            self._id_cols[key] = self._read(chunk, name)
        return self._id_cols[key]

    # ------------------------------------------------------------------ lookup
    def find(self, snap: int, subfind_id: int) -> tuple[int, int]:
        """(chunk, row) of the tree node for subhalo `subfind_id` at snapshot `snap`."""
        raw = np.int64(snap) * _RAW_FACTOR + np.int64(subfind_id)
        for chunk in range(len(self.paths)):
            rows = np.nonzero(self._col(chunk, "SubhaloIDRaw") == raw)[0]
            if rows.size:
                return chunk, int(rows[0])
        raise KeyError(f"subhalo (snap={snap}, subfind={subfind_id}) not in any SubLink tree "
                       f"of {self.sim} (too few particles to be tracked?)")

    def positions_at(self, snap: int, min_mass: float = 0.0) -> dict:
        """SubfindID, SubhaloPos (ckpc/h) and SubhaloMass (1e10 Msun/h) of every tree
        subhalo at `snap` -- the tracer pool for region (e.g. void) tracking."""
        ids, pos, mass = [], [], []
        for chunk in range(len(self.paths)):
            sel = np.nonzero(self._col(chunk, "SnapNum") == snap)[0]
            if not sel.size:
                continue
            m = self._read(chunk, "SubhaloMass", sel)
            keep = m >= min_mass
            ids.append(self._read(chunk, "SubfindID", sel)[keep])
            pos.append(self._read(chunk, "SubhaloPos", sel)[keep])
            mass.append(m[keep])
        if not ids:
            raise KeyError(f"no subhalos at snapshot {snap} in the {self.sim} trees")
        return {"subfind_id": np.concatenate(ids).astype(np.int64),
                "pos_ckpc_h": np.concatenate(pos).astype(np.float64),
                "mass": np.concatenate(mass).astype(np.float64)}

    def subhalos_at(self, snap: int, order_by: str = "SubhaloMass", n: int = 10,
                    central_only: bool = True) -> np.ndarray:
        """SubfindIDs of the top-n subhalos at `snap`, selected FROM THE TREES
        (no group catalog needed). central_only keeps FoF centrals."""
        found = []
        for chunk in range(len(self.paths)):
            snapnum = self._col(chunk, "SnapNum")
            sel = np.nonzero(snapnum == snap)[0]
            if not sel.size:
                continue
            if central_only:
                is_central = (self._read(chunk, "FirstSubhaloInFOFGroupID", sel)
                              == self._col(chunk, "SubhaloID")[sel])
                sel = sel[is_central]
            vals = self._read(chunk, order_by, sel)
            sub = self._read(chunk, "SubfindID", sel)
            found.append(np.column_stack([vals, sub]))
        allv = np.concatenate(found)
        top = allv[np.argsort(allv[:, 0])[::-1][:n]]
        return top[:, 1].astype(np.int64)

    # ------------------------------------------------------------------ branch walk
    def _rows_to_fields(self, chunk: int, rows: np.ndarray, fields: list[str]) -> dict:
        """Read `fields` (+ SnapNum, SubfindID) at tree rows, preserving the given row order."""
        want = list(dict.fromkeys(["SnapNum", "SubfindID", *fields]))
        f = self._file(chunk)
        order = np.argsort(rows)          # h5py fancy indexing needs increasing order
        out = {}
        for name in want:
            data = self._raw(f[name], f[name][rows[order].tolist()])
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size)
            out[name] = np.asarray(data)[inv]
        return out

    def _main_branch_rows(self, chunk: int, row: int) -> np.ndarray:
        sid = self._col(chunk, "SubhaloID")
        fprog = self._col(chunk, "FirstProgenitorID")
        rows = [row]
        r = row
        while True:
            nxt = fprog[r]
            if nxt == -1:
                break
            r = r + int(nxt - sid[r])     # contiguous-tree relative indexing
            rows.append(r)
        return np.asarray(rows)

    def main_branch(self, snap: int, subfind_id: int, fields: list[str]) -> dict:
        """Main progenitor branch of (snap, subfind_id), ordered from `snap` back in time.

        Returns {field: array}, always including 'SnapNum' and 'SubfindID'.
        """
        chunk, row = self.find(snap, subfind_id)
        return self._rows_to_fields(chunk, self._main_branch_rows(chunk, row), fields)

    def descendant_branch(self, snap: int, subfind_id: int, fields: list[str]) -> dict:
        """Forward walk: (snap, subfind_id) and its descendants down to the tree root (z=0 end).

        Ordered from `snap` FORWARD in time. Same return shape as main_branch."""
        chunk, row = self.find(snap, subfind_id)
        sid = self._col(chunk, "SubhaloID")
        desc = self._col(chunk, "DescendantID")
        rows = [row]
        r = row
        while True:
            nxt = desc[r]
            if nxt == -1:
                break
            r = r + int(nxt - sid[r])
            rows.append(r)
        return self._rows_to_fields(chunk, np.asarray(rows), fields)

    # ------------------------------------------------------------------ mergers
    def mergers_along_branch(self, snap: int, subfind_id: int, min_ratio: float = 0.0,
                             mass_field: str = "SubhaloMass", ratio_at: str = "maxpast") -> dict:
        """Merger events onto the main-progenitor branch of (snap, subfind_id).

        A merger at descendant snapshot S is a secondary progenitor at S-1 (the
        FirstProgenitor's NextProgenitor chain) that vanishes into the branch.

        ratio_at='maxpast' (default, the SubLink convention of Rodriguez-Gomez et al. 2015):
        the ratio compares the two progenitors at the snapshot where the SECONDARY reached
        its maximum past mass -- infalling subhalos are tidally stripped long before they
        merge, so instantaneous ratios ('instant') classify nearly every merger as minor.
        `min_ratio` filters minor mergers (e.g. 0.1 keeps >=1:10).

        Returns {"snap" (descendant snapshot), "ratio", "sec_mass", "main_mass",
        "sec_subfind"} arrays ordered from late to early times.
        """
        if ratio_at not in ("maxpast", "instant"):
            raise ValueError("ratio_at must be 'maxpast' or 'instant'")
        chunk, row = self.find(snap, subfind_id)
        sid = self._col(chunk, "SubhaloID")
        fprog = self._col(chunk, "FirstProgenitorID")
        nprog = self._col(chunk, "NextProgenitorID")
        mass = self._read(chunk, mass_field)
        snapn = self._col(chunk, "SnapNum")
        subf = self._read(chunk, "SubfindID")

        # main-branch masses by snapshot (for the 'maxpast' ratio denominator)
        main_rows = self._main_branch_rows(chunk, row)
        main_mass_at = {int(snapn[r]): float(mass[r]) for r in main_rows}

        ev = {"snap": [], "ratio": [], "sec_mass": [], "main_mass": [], "sec_subfind": []}
        r = row
        while True:
            p = fprog[r]
            if p == -1:
                break
            pr = r + int(p - sid[r])                 # main progenitor row
            s = nprog[pr]
            while s != -1:                            # secondary progenitors = mergers
                srow = r + int(s - sid[r])
                if ratio_at == "instant":
                    m_sec, m_main = float(mass[srow]), float(mass[pr])
                else:
                    # secondary's max past mass along its own main branch, and the tracked
                    # halo's main-branch mass at that same snapshot
                    sec_rows = self._main_branch_rows(chunk, srow)
                    k = int(np.argmax(mass[sec_rows]))
                    m_sec = float(mass[sec_rows[k]])
                    s_peak = int(snapn[sec_rows[k]])
                    m_main = main_mass_at.get(s_peak, float(mass[pr]))
                ratio = m_sec / m_main if m_main > 0 else np.nan
                if np.isfinite(ratio) and ratio >= min_ratio:
                    ev["snap"].append(int(snapn[r]))  # merger completes at the descendant snap
                    ev["ratio"].append(ratio)
                    ev["sec_mass"].append(m_sec)
                    ev["main_mass"].append(m_main)
                    ev["sec_subfind"].append(int(subf[srow]))
                s = nprog[srow]
            r = pr
        return {k: np.asarray(v) for k, v in ev.items()}
