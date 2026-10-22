"""The Data tab's job, moved out of runspec.py: DataSpec (scripts/download_snapshots.sh for snapshots, group
catalogues, merger trees and ICs, then python/tools/merge_HDF5.py), with the TNG API key test and the on-disk
state of a simulation's snapshots and whole-simulation products. runspec re-exports all of it and is the module to
import; this one reads runspec's own names through it at call time (_rs.API_KEY_FILE, which a test redirects)."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import runspec as _rs


def api_key_available() -> bool:
    return bool(os.environ.get("TNG_API_KEY", "").strip()) or (_rs.API_KEY_FILE.is_file()
                                                              and os.access(_rs.API_KEY_FILE, os.R_OK))


def snapshot_state(sim_path, n: int) -> dict:
    """What is on disk for snapshot n: raw chunks and merged files, particles and group catalogue."""
    sd, gd = Path(sim_path) / f"snapdir_{n:03d}", Path(sim_path) / f"groups_{n:03d}"
    return {"chunks": len(list(sd.glob(f"snap_{n:03d}.*.hdf5"))),
            "merged": (sd / f"combined_{n:03d}.hdf5").is_file(),
            "gc_chunks": len(list(gd.glob(f"fof_subhalo_tab_{n:03d}.*.hdf5"))),
            "gc_merged": (gd / f"combined_fof_subhalo_tab_{n:03d}.hdf5").is_file()}


def sim_state(sim_path) -> dict:
    """Whole-simulation products: merger trees and initial conditions."""
    sp, td = Path(sim_path), Path(sim_path) / "Merger Trees"
    return {"tree_chunks": len(list(td.glob("tree_extended.*.hdf5"))),
            "tree_merged": (td / "combined_tree_extended.hdf5").is_file(),
            "ics_raw": (sp / "ics.hdf5").is_file() or (sp / "snap_ics.hdf5").is_file(),
            "ics_converted": (sp / "combined_ics.hdf5").is_file()}


def data_snapshots(sim_path) -> list[int]:
    """The redshift ladder plus every snapshot or group catalogue already on disk."""
    found = set(_rs.SNAPSHOT_LADDER)
    for d in list(Path(sim_path).glob("snapdir_*")) + list(Path(sim_path).glob("groups_*")):
        try:
            found.add(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(found)


@dataclass
class DataSpec:
    """Getting a simulation onto disk: download_snapshots.sh (particle snapshots, group
    catalogues, merger trees, initial conditions) and merge_HDF5.py (h-free combined files).
    Downloads run first, then merges, one step each. The API key is never on a command line."""
    data_root: str = field(default_factory=lambda: _rs.default_data_root())
    sim: str = ""
    snapshots: list[int] = field(default_factory=list)
    download_snapshots: bool = True
    download_groupcats: bool = False
    download_trees: bool = False
    download_ics: bool = False
    merge_snapshots: bool = True
    merge_groupcats: bool = False
    merge_trees: bool = False
    convert_ics: bool = False
    delete_chunks: bool = False

    def sim_path(self) -> Path:
        return _rs.sim_dir(self.sim, self.data_root)

    def _downloads(self) -> bool:
        return self.download_snapshots or self.download_groupcats or self.download_trees or self.download_ics

    def _todo(self) -> dict:
        """What is still missing on disk -- merged items are left out of every step: the download
        script would skip them anyway, and re-merging would rewrite combined_NNN.hdf5, whose new
        mtime makes every grid and slice built from it look stale to the pipeline."""
        sp = self.sim_path()
        states = {n: snapshot_state(sp, n) for n in sorted(set(self.snapshots))}
        whole = sim_state(sp)
        return {"snaps": [n for n, st in states.items() if not st["merged"]],
                "gcs": [n for n, st in states.items() if not st["gc_merged"]],
                "trees": not whole["tree_merged"], "ics": not whole["ics_converted"],
                "states": states, "whole": whole}

    def steps(self) -> list[_rs.Step]:
        todo = self._todo()
        snaps, gcs = [str(n) for n in todo["snaps"]], [str(n) for n in todo["gcs"]]
        # PY: the download script checks the chunks already on disk with h5py (an incomplete one is fetched
        # again) -- the same python the merge step runs with
        env = {"DTFE_DATA_ROOT": str(self.data_root), "PY": _rs.PYTHON}
        dl = [str(_rs.DOWNLOAD_SCRIPT)]
        out = []
        if self.download_snapshots and snaps:
            out.append(_rs.Step(f"download snapshots {' '.join(snaps)}", dl + ["-s", self.sim, *snaps], env))
        if self.download_groupcats and gcs:
            out.append(_rs.Step(f"download group catalogues {' '.join(gcs)}", dl + ["-c", "-s", self.sim, *gcs], env))
        if self.download_trees and todo["trees"]:
            out.append(_rs.Step("download merger trees", dl + ["-t", "-s", self.sim], env))
        if self.download_ics and todo["ics"]:
            out.append(_rs.Step("download initial conditions", dl + ["-i", "-s", self.sim], env))
        merge = [_rs.PYTHON, str(_rs.MERGE_TOOL), "-d", str(self.sim_path())]
        tail = ["--delete-chunks"] if self.delete_chunks else []
        # a merge runs only when its download step exited 0: half a download merged is a product with
        # missing particles (merge_HDF5.py refuses what it can detect; a missing TREE chunk it cannot)
        have = {st.label.split(" ")[1]: st.label for st in out}        # 'snapshots', 'group', 'merger', 'initial'
        if self.merge_snapshots and snaps:
            out.append(_rs.Step(f"merge snapshots {' '.join(snaps)}", merge + snaps + tail, {}, after=have.get("snapshots", "")))
        if self.merge_groupcats and gcs:
            out.append(_rs.Step(f"merge group catalogues {' '.join(gcs)}", merge + ["--groupcats", *gcs] + tail, {},
                                after=have.get("group", "")))
        if self.merge_trees and todo["trees"]:
            out.append(_rs.Step("merge merger trees", merge + ["--trees"] + tail, {}, after=have.get("merger", "")))
        if self.convert_ics and todo["ics"]:
            out.append(_rs.Step("convert initial conditions", merge + ["--ics"] + tail, {}, after=have.get("initial", "")))
        return out

    def stale_partials(self) -> list:
        """combined_*.hdf5.partial files a stopped merge left for the products about to be merged: the
        merge removes them first, so their bytes are free for it (and worth a word)."""
        todo = self._todo()
        sp = self.sim_path()
        cands = []
        if self.merge_snapshots:
            cands += [sp / f"snapdir_{n:03d}" / f"combined_{n:03d}.hdf5.partial" for n in todo["snaps"]]
        if self.merge_groupcats:
            cands += [sp / f"groups_{n:03d}" / f"combined_fof_subhalo_tab_{n:03d}.hdf5.partial" for n in todo["gcs"]]
        if self.merge_trees and todo["trees"]:
            cands.append(sp / "Merger Trees" / "combined_tree_extended.hdf5.partial")
        if self.convert_ics and todo["ics"]:
            cands.append(sp / "combined_ics.hdf5.partial")
        return [c for c in cands if c.is_file()]

    def download_bytes(self) -> tuple[int, int]:
        """(estimated raw download, estimated merged output) in bytes for the chosen snapshots,
        skipping what the script would skip (already merged) and chunks already on disk."""
        n_part = _rs.dm_particles(self.sim) or 0
        raw = merged = 0
        for n in set(self.snapshots):
            st = snapshot_state(self.sim_path(), n)
            if self.download_snapshots and not st["merged"] and not st["chunks"]:
                raw += n_part * _rs.RAW_BYTES_PER_PARTICLE
            if self.merge_snapshots and not st["merged"]:
                merged += n_part * _rs.MERGED_BYTES_PER_PARTICLE
        return raw, merged

    def problems(self) -> list[tuple[str, str]]:
        out = _rs._root_problems(self.data_root)
        if out:
            return out
        if not self.sim:
            return [("error", "no simulation named")]
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", self.sim):
            return [("error", f"not a simulation name: {self.sim!r}")]
        per_snap = (self.download_snapshots or self.download_groupcats or self.merge_snapshots
                    or self.merge_groupcats)
        whole = self.download_trees or self.download_ics or self.merge_trees or self.convert_ics
        if not (per_snap or whole):
            return out + [("error", "nothing selected to download or merge")]
        if per_snap and not self.snapshots and not whole:
            return out + [("error", "no snapshots selected")]
        todo = self._todo()
        if not self.steps():
            return out + [("error", "nothing left to do: everything selected is already merged")]
        if self.sim not in _rs.TNG_SIMULATIONS:
            out.append(("warning", f"{self.sim} is not a TNG run this GUI knows; the TNG API may not serve it"))
        if self._downloads():
            if not api_key_available():
                out.append(("error", f"no TNG API key: put it in {_rs.API_KEY_FILE} (chmod 600) or set TNG_API_KEY"))
            if shutil.which("wget") is None:
                out.append(("error", "wget is not installed (brew install wget)"))
            if not self.sim.endswith("-Dark") and self.download_snapshots:
                out.append(("warning", "full-physics snapshots also hold gas and stars: several times larger "
                                       "downloads, and only the dark matter is merged"))
        skipped = []
        if self.download_snapshots or self.merge_snapshots:
            skipped += [f"snapshot {n:03d}" for n, st in todo["states"].items() if st["merged"]]
        if self.download_groupcats or self.merge_groupcats:
            skipped += [f"groups {n:03d}" for n, st in todo["states"].items() if st["gc_merged"]]
        if (self.download_trees or self.merge_trees) and not todo["trees"]:
            skipped.append("merger trees")
        if (self.download_ics or self.convert_ics) and not todo["ics"]:
            skipped.append("initial conditions")
        if skipped:
            out.append(("warning", "already merged, left out: " + ", ".join(skipped)
                                   + " (re-merging would make their grids look stale)"))
        if self.merge_snapshots and not self.download_snapshots:
            empty = [n for n in todo["snaps"] if not todo["states"][n]["chunks"]]
            if empty:
                out.append(("warning", "no chunks to merge for snapshot " + " ".join(f"{n:03d}" for n in empty)))
        if self.merge_groupcats and not self.download_groupcats:
            empty = [n for n in todo["gcs"] if not todo["states"][n]["gc_chunks"]]
            if empty:
                out.append(("warning", "no group-catalogue chunks to merge for " + " ".join(f"{n:03d}" for n in empty)))
        if self.delete_chunks:
            out.append(("warning", "deleting the raw chunks after the merge removes "
                                   "the only copy of the fields the merge drops (potential, Subfind densities)"))
        raw, merged = self.download_bytes()
        stale = self.stale_partials()
        if stale:
            gb = sum(p.stat().st_size for p in stale) / 1e9
            out.append(("warning", f"a stopped merge left {gb:.1f} GB in " + ", ".join(p.name for p in stale)
                                   + "; the merge removes it first"))
        try:
            free = shutil.disk_usage(self.data_root).free
            free += sum(p.stat().st_size for p in stale)
        except OSError:
            free = None
        if free is not None and raw + merged > free:
            out.append(("error", f"~{(raw + merged) / 1e9:.0f} GB needed, {free / 1e9:.0f} GB free on the data disk"))
        elif free is not None and raw + merged > 0.8 * free:
            out.append(("warning", f"~{(raw + merged) / 1e9:.0f} GB of {free / 1e9:.0f} GB free on the data disk"))
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DataSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
