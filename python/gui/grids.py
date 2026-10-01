"""The data side of the GUI's Explore tab. Pure numpy (no Qt): tests/py_gui_runspec_test.py.

  OutputSet(root)          one run's grid files '<dir>/<prefix>.[a_]<suffix>': which fields exist,
                           the grid size, one slice of any field (read through a memmap, so a
                           1024^3 cube costs one plane of memory), a value at a cell
  find_outputs(...)        every output set under the data root and the custom-snapshot output directories
  colorize(plane, ...)     a 2D array -> RGB bytes with a matplotlib colormap (percentile range,
                           optional log10), plus the value range used
  server_settings(out)     how to start a dtfelib.Estimator on the snapshot behind an output (the
                           Explore tab's click-to-query), from the TNG layout or the custom-snapshot
                           job's settings sidecar ('<prefix>.gui.json')
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from dtfelib.io import FIELDS, LEGACY_SUFFIX, _VELOCITY_SCALE_EXP, _grid_shape_of  # noqa: E402

# the fields worth a map, in menu order, with their labels
FIELD_LABELS = {
    "density": "density ρ/ρ̄", "streams": "stream count", "hidden_streams": "hidden streams",
    "velocity": "velocity", "divergence": "velocity divergence", "vorticity": "vorticity",
    "shear": "shear", "gradient": "velocity gradient", "dispersion": "velocity dispersion",
    "dispersion_tensor": "dispersion tensor", "caustic": "caustic (fold) cells",
    "caustic_class": "caustic class", "tet_touch": "tetrahedra touching",
    "tweb": "T-web class", "tweb_eigenvalues": "T-web eigenvalues", "vweb": "V-web class",
    "vweb_eigenvalues": "V-web eigenvalues",
}
# what a map's values mean, where the label cannot say it (the field list's tooltips)
FIELD_HELP = {
    "streams": "How many streams overlap the cell: 1 = single-stream, 3, 5, … in folds.",
    "hidden_streams": "1 = the cell holds multi-stream volume its stream count misses (a fold "
                      "thinner than the sampling).\n2 = a tetrahedron too small to be sampled "
                      "left its mass here (usually of the same stream).\n3 = both.",
    "caustic": "1 = a fold caustic crosses the cell.",
}
# vector fields that can be drawn as arrows (their in-plane components) over the colour map
VECTOR_FIELDS = {"velocity", "vorticity"}
# which component menu a field offers; 'norm' = the vector's length
COMPONENTS = {3: ["norm", "x", "y", "z"], 5: ["1", "2", "3", "4", "5"], 6: ["xx", "xy", "xz", "yy", "yz", "zz"],
              9: ["trace (divergence)", "xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]}
# maps that are best on a log scale by default / diverging maps / category maps
LOG_DEFAULT = {"density", "dispersion", "tet_touch"}
DIVERGING = {"divergence", "vorticity", "gradient", "shear", "tweb_eigenvalues", "vweb_eigenvalues"}
CATEGORICAL = {"streams", "hidden_streams", "caustic", "caustic_class", "tweb", "vweb"}


@dataclass
class FieldFile:
    name: str                   # dtfelib field name
    path: Path
    ncomp: int
    averaged: bool


# whole fields up to this size are read into memory once (a strided z-plane read touches every
# page of the file: 1.9 s for a 512^3 grid on an external disk, every time); bigger ones go
# through a memmap per slice
CUBE_CACHE_BYTES = int(1.5e9)
_CUBES: dict[Path, np.ndarray] = {}       # module-wide, least recently used first


def _cached_cube(path: Path, dtype, shape) -> np.ndarray | None:
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if nbytes > CUBE_CACHE_BYTES:
        return None
    key = path.resolve()
    if key in _CUBES:
        _CUBES[key] = _CUBES.pop(key)     # most recently used last
        return _CUBES[key]
    while _CUBES and sum(a.nbytes for a in _CUBES.values()) + nbytes > CUBE_CACHE_BYTES:
        _CUBES.pop(next(iter(_CUBES)))
    cube = np.fromfile(path, dtype=dtype).reshape(shape)
    _CUBES[key] = cube
    return cube


class OutputSet:
    """The grid files of one run: '<dir>/<prefix>.*'."""

    def __init__(self, root, box: tuple | None = None, redshift: float | None = None, title: str = ""):
        self.root = Path(root)
        self.dir, self.prefix = self.root.parent, self.root.name
        self.title = title or f"{self.dir.name}/{self.prefix}"
        self.files: dict[str, FieldFile] = {}
        for name, (suffix, ncomp, _ps) in FIELDS.items():
            for sfx in (suffix, LEGACY_SUFFIX.get(name)):   # the current name, then a renamed one
                for averaged in (True, False):              # prefer the volume-averaged grid
                    p = self.dir / f"{self.prefix}.{'a_' if averaged else ''}{sfx}"
                    if sfx and p.is_file() and name not in self.files:
                        self.files[name] = FieldFile(name, p, ncomp, averaged)
        self.n = self.dtype = None
        for ff in self.files.values():
            try:
                self.n, self.dtype = _grid_shape_of(ff.path, ff.ncomp)
                break
            except ValueError:
                continue
        # a file whose size does not match the grid (an interrupted write) is left out
        if self.n:
            ok = {}
            for name, ff in self.files.items():
                if ff.path.stat().st_size == self.n ** 3 * ff.ncomp * self.dtype.itemsize:
                    ok[name] = ff
            self.files = ok
        self.box = box                  # ((xlo, ylo, zlo), (xhi, yhi, zhi)) in Mpc, or None
        self.redshift = redshift

    def __bool__(self):
        return bool(self.files) and bool(self.n)

    def fields(self) -> list[str]:
        return [n for n in FIELD_LABELS if n in self.files]

    @property
    def lo(self) -> np.ndarray:
        return np.zeros(3) if self.box is None else np.asarray(self.box[0], float)

    @property
    def length(self) -> np.ndarray:
        if self.box is None:
            return np.full(3, float(self.n or 1))       # unknown box: cell units
        return np.asarray(self.box[1], float) - np.asarray(self.box[0], float)

    def _memmap(self, name: str) -> np.ndarray:
        ff = self.files[name]
        shape = (self.n, self.n, self.n) + ((ff.ncomp,) if ff.ncomp > 1 else ())
        cube = _cached_cube(ff.path, self.dtype, shape)
        return cube if cube is not None else np.memmap(ff.path, dtype=self.dtype, mode="r", shape=shape)

    def _velocity_factor(self, name: str) -> float:
        exp = _VELOCITY_SCALE_EXP.get(name)
        if exp is None or self.redshift is None:
            return 1.0
        return (1.0 / (1.0 + self.redshift)) ** exp       # u-units -> peculiar km/s (FieldSet)

    def slice(self, name: str, axis: int = 2, index: int | None = None, component: str = "norm") -> np.ndarray:
        """One plane of a field, (N, N) float64, rows = the first remaining axis. Components of a
        vector/tensor field by name (COMPONENTS); 'norm' = |v|, 'trace ...' = the trace."""
        index = self.n // 2 if index is None else int(index)
        mm = self._memmap(name)
        sl = [slice(None)] * 3
        sl[axis] = index
        plane = np.array(mm[tuple(sl)], dtype=np.float64)
        del mm
        ff = self.files[name]
        if ff.ncomp > 1:
            comps = COMPONENTS.get(ff.ncomp, [str(i) for i in range(ff.ncomp)])
            if component not in comps:
                component = comps[0]
            if component == "norm":
                plane = np.sqrt((plane ** 2).sum(axis=-1))
            elif component.startswith("trace"):
                plane = plane[..., 0] + plane[..., 4] + plane[..., 8]
            else:
                idx = comps.index(component) - (1 if ff.ncomp in (3, 9) else 0)
                plane = plane[..., idx]
        return plane * self._velocity_factor(name)

    def arrow_field(self, name: str, axis: int = 2, index: int | None = None, per_side: int = 32,
                    smooth: float = 0.0) -> dict | None:
        """Arrows for a vector field on one slice, in IMAGE coordinates (the Explore image is
        plane.T[::-1]: x = the first in-plane axis rightward, y = the second, drawn upward, in
        cell units from the image's top-left corner). The two in-plane components are
        block-averaged to about 'per_side' arrows along each side; dx, dy are scaled so the
        longest arrow is 1 (times 'spacing' cells when drawn). None for a non-vector field."""
        ff = self.files.get(name)
        if ff is None or ff.ncomp != 3:
            return None
        rest = [a for a in range(3) if a != axis]
        u = self.slice(name, axis, index, "xyz"[rest[0]])      # rows = rest[0], columns = rest[1]
        v = self.slice(name, axis, index, "xyz"[rest[1]])
        if smooth > 0:
            try:
                from scipy.ndimage import gaussian_filter
                u, v = gaussian_filter(u, smooth, mode="wrap"), gaussian_filter(v, smooth, mode="wrap")
            except ImportError:
                pass
        n0, n1 = u.shape
        step = max(1, int(round(max(n0, n1) / per_side)))
        m0, m1 = n0 // step * step, n1 // step * step
        if m0 == 0 or m1 == 0:
            return None
        ub = u[:m0, :m1].reshape(m0 // step, step, m1 // step, step).mean(axis=(1, 3))
        vb = v[:m0, :m1].reshape(m0 // step, step, m1 // step, step).mean(axis=(1, 3))
        c0 = (np.arange(m0 // step) + 0.5) * step                 # block centres, plane cell units
        c1 = (np.arange(m1 // step) + 0.5) * step
        a0, a1 = np.meshgrid(c0, c1, indexing="ij")
        vmax = float(np.sqrt(ub ** 2 + vb ** 2).max())
        scale = 1.0 / vmax if vmax > 0 else 0.0
        return {"x": a0.ravel(), "y": (n1 - a1).ravel(),       # image: y grows downward
                "dx": (ub * scale).ravel(), "dy": (-vb * scale).ravel(),
                "u": ub, "v": vb, "c0": c0, "c1": c1,           # plane frame, for a matplotlib quiver
                "spacing": float(step), "vmax": vmax, "axes": (rest[0], rest[1])}

    def value(self, name: str, ijk, component: str = "norm") -> float:
        """The value of one cell (same component convention as slice)."""
        i, j, k = (int(v) for v in ijk)
        return float(self.slice(name, 0, i, component)[j, k])

    # -- coordinates
    def cell_center(self, ijk) -> np.ndarray:
        return self.lo + (np.asarray(ijk, float) + 0.5) * self.length / self.n

    def plane_point(self, axis: int, index: int, row: int, col: int) -> np.ndarray:
        """The 3D cell (i, j, k) of pixel (row, col) of slice 'index' along 'axis'."""
        rest = [a for a in range(3) if a != axis]
        ijk = [0, 0, 0]
        ijk[axis], ijk[rest[0]], ijk[rest[1]] = index, row, col
        return np.asarray(ijk)


# ---------------------------------------------------------------------- colour
_LUTS: dict[str, np.ndarray] = {}


def lut(cmap: str) -> np.ndarray:
    """(256, 3) uint8 colour table of a matplotlib colormap (a grey ramp without matplotlib)."""
    if cmap not in _LUTS:
        try:
            import matplotlib
            table = matplotlib.colormaps[cmap](np.linspace(0, 1, 256))[:, :3]
        except Exception:
            table = np.repeat(np.linspace(0, 1, 256)[:, None], 3, axis=1)
        _LUTS[cmap] = (table * 255 + 0.5).astype(np.uint8)
    return _LUTS[cmap]


def colorize(plane: np.ndarray, cmap: str = "viridis", log: bool = False,
             clip: tuple = (1.0, 99.0), symmetric: bool = False,
             vrange: tuple | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """plane -> (H, W, 3) uint8 RGB and the (vmin, vmax) shown. log10 of positive values (the rest
    at the bottom of the scale); range from the percentiles 'clip' unless 'vrange' is given;
    'symmetric' centres a diverging map on 0."""
    data = np.asarray(plane, dtype=np.float64)
    if log:
        pos = data[data > 0]
        floor = pos.min() if pos.size else 1.0
        data = np.log10(np.where(data > 0, data, floor))
    finite = data[np.isfinite(data)]
    if vrange is not None:
        vmin, vmax = vrange
    elif finite.size:
        vmin, vmax = np.percentile(finite, clip)
    else:
        vmin, vmax = 0.0, 1.0
    if symmetric:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
    if not vmax > vmin:
        vmax = vmin + 1.0
    idx = np.clip((data - vmin) / (vmax - vmin) * 255.0, 0, 255)
    idx = np.where(np.isfinite(idx), idx, 0).astype(np.uint8)
    return lut(cmap)[idx], (float(vmin), float(vmax))


# ---------------------------------------------------------------------- discovery
def _tng_box(snapdir: Path) -> tuple[tuple, float | None] | tuple[None, None]:
    """((lo, hi), redshift) of a TNG-layout snapshot from its combined file (h-free ckpc -> Mpc)."""
    files = sorted(Path(snapdir).glob("combined_[0-9][0-9][0-9].hdf5"))
    if not files:
        return None, None
    try:
        import h5py
        with h5py.File(files[0], "r") as h:
            box = float(h["Header"].attrs["BoxSize"]) / 1000.0
            z = float(h["Header"].attrs.get("Redshift", 0.0))
        return ((0.0, 0.0, 0.0), (box, box, box)), z
    except Exception:
        return None, None


def sidecar(root) -> dict | None:
    """The settings a custom-snapshot job saved beside its outputs ('<prefix>.gui.json')."""
    p = Path(str(root) + ".gui.json")
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def custom_box(settings: dict) -> tuple | None:
    """A custom-snapshot run's box in Mpc: 'box_mpc', else its --box (file units) / MpcUnit, else the
    input snapshot's header BoxSize / MpcUnit."""
    unit = float(settings.get("mpc_unit") or 1.0)
    box = settings.get("box_mpc") or []
    if len(box) != 6 and len(settings.get("box") or []) == 6:
        box = [v / unit for v in settings["box"]]
    if len(box) != 6 and settings.get("input_file"):
        try:
            import h5py
            with h5py.File(settings["input_file"], "r") as h:
                b = float(h["Header"].attrs["BoxSize"]) / unit
            box = [0.0, b] * 3
        except Exception:
            box = []
    if len(box) == 6:
        return (box[0], box[2], box[4]), (box[1], box[3], box[5])
    return None


def find_outputs(data_root, extra_dirs=()) -> list[tuple[str, OutputSet]]:
    """(menu title, OutputSet) for every run with grids: the TNG layout under the data root, then
    the custom-snapshot output directories."""
    out = []
    root = Path(data_root)
    seen = set()
    if root.is_dir():
        for pattern in ("*/snapdir_*/*.a_den", "*/*/snapdir_*/*.a_den", "*/snapdir_*/*.den", "*/*/snapdir_*/*.den"):
            for f in sorted(root.glob(pattern)):
                prefix = f.name.split(".")[0]
                key = f.parent / prefix
                if key in seen:
                    continue
                seen.add(key)
                box, z = _tng_box(f.parent)
                snap = f.parent.name.split("_")[-1]
                title = f"{f.parent.parent.name} · {snap}" + (f" (z={z:.2f})" if z is not None else "") + f" · {prefix}"
                o = OutputSet(key, box=box, redshift=z, title=title)
                if o:
                    out.append((title, o))
    for d in extra_dirs:
        for f in sorted(Path(d).glob("*.a_den")) + sorted(Path(d).glob("*.den")):
            prefix = f.name.split(".")[0]
            key = f.parent / prefix
            if key in seen:
                continue
            seen.add(key)
            s = sidecar(key) or {}
            o = OutputSet(key, box=custom_box(s), redshift=None, title=f"custom · {prefix} ({key.parent.name})")
            if o:
                out.append((o.title, o))
    return out


def machine_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 16 * 2 ** 30


def server_cost(snapshot, phase_space: bool = True, ram_bytes: int | None = None) -> dict | None:
    """What starting a query server on this snapshot will take, before it is started -- the
    binary's own serve model (auto_tune.h autoTuneServe) in round numbers: particles, memory of
    one tessellation, whether it will be split (budget 60% of RAM) and then the cache on disk
    (~150 B per cached vertex, the padded partitions hold ~4x the particles). None if unreadable."""
    try:
        import h5py
        with h5py.File(snapshot, "r") as h:
            hd = h["Header"].attrs
            n = int(np.sum(hd["NumPart_Total"])) if "NumPart_Total" in hd else int(np.sum(hd["NumPart_ThisFile"]))
    except Exception:
        return None
    ram = ram_bytes or machine_ram_bytes()
    budget = 0.6 * ram
    per_vertex = (658 if phase_space else 640) + 160
    single = n * (1.73 if phase_space else 1.2) * per_vertex + n * 64
    split = single > budget
    cache = n * 4.1 * 150 if split else n * 1.73 * 150
    return {"particles": n, "single_gb": single / 1e9, "budget_gb": budget / 1e9, "split": split,
            "cache_gb": cache / 1e9}


def server_settings(out: OutputSet) -> dict | None:
    """dtfelib.Estimator arguments for the snapshot behind an output set, or None when unknown.
    TNG layout: the snapdir's combined file (+ the simulation's converted ICs when present);
    a custom-snapshot run: the job's own settings sidecar."""
    s = sidecar(out.root)
    if s:
        if not s.get("input_file") or not Path(s["input_file"]).is_file():
            return None
        kw = {"snapshot": s["input_file"], "phase_space": s.get("estimator", "ps") == "ps",
              "periodic": bool(s.get("periodic", True)), "mpc_unit": float(s.get("mpc_unit", 1.0)),
              "input_type": int(s.get("input_type", 105))}
        if s.get("lagrangian_file"):
            kw["lagrangian_input"] = s["lagrangian_file"]
        if len(s.get("box") or []) == 6:
            kw["options"] = ["--box"] + [repr(float(v)) for v in s["box"]]
        return kw
    combined = sorted(out.dir.glob("combined_[0-9][0-9][0-9].hdf5"))
    if not combined:
        return None
    kw = {"snapshot": str(combined[0]), "phase_space": not out.prefix == "output", "periodic": True}
    ics = out.dir.parent / "combined_ics.hdf5"
    if kw["phase_space"] and ics.is_file():
        kw["lagrangian_input"] = str(ics)
    return kw
