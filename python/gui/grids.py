"""The data side of the GUI's Explore tab. Pure numpy (no Qt): tests/py_gui_runspec_test.py.

  OutputSet(root)          one run's grid files '<dir>/<prefix>.[a_]<suffix>': which fields exist,
                           the grid size, one slice of any field (read through a memmap, so a
                           1024^3 cube costs one plane of memory), a value at a cell. A 2D run's
                           grids (n^2 cells) are one plane: its slice is the whole grid
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
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from dtfelib.io import FIELDS as _IO_FIELDS, LEGACY_SUFFIX, _VELOCITY_SCALE_EXP  # noqa: E402

# the launcher's field table: dtfelib's + the per-particle scalar a run with a scalar dataset writes ('.a_scalar',
# '--field scalar_a'; dtfelib's analysis never reads it, the Explore map can)
FIELDS = {**_IO_FIELDS, "scalar": ("scalar", 1, False)}

# the fields worth a map, in menu order, with their labels
FIELD_LABELS = {
    "density": "density ρ/ρ̄", "streams": "stream count", "hidden_streams": "hidden streams",
    "velocity": "velocity", "divergence": "velocity divergence", "vorticity": "vorticity",
    "shear": "shear", "gradient": "velocity gradient", "dispersion": "velocity dispersion",
    "dispersion_tensor": "dispersion tensor", "caustic": "caustic (fold) cells",
    "caustic_class": "caustic class", "tet_touch": "tetrahedra touching",
    "tweb": "T-web class", "tweb_eigenvalues": "T-web eigenvalues", "vweb": "V-web class",
    "vweb_eigenvalues": "V-web eigenvalues", "scalar": "per-particle scalar",
}
# what a map's values mean, where the label cannot say it (the field list's tooltips)
FIELD_HELP = {
    "streams": "How many streams overlap the cell: 1 = single-stream, 3, 5, … in folds.",
    "hidden_streams": "1 = the cell holds multi-stream volume its stream count misses (a fold "
                      "thinner than the sampling).\n2 = a tetrahedron too small to be sampled "
                      "left its mass here (usually of the same stream).\n3 = both.",
    "caustic": "1 = a fold caustic crosses the cell.",
    "scalar": "The particles' own value (the run's scalar dataset, e.g. a potential), interpolated like the density.",
}
# vector fields that can be drawn as arrows (their in-plane components) over the colour map
VECTOR_FIELDS = {"velocity", "vorticity"}
# which component menu a field offers; 'norm' = the vector's length
COMPONENTS = {3: ["norm", "x", "y", "z"], 5: ["1", "2", "3", "4", "5"], 6: ["xx", "xy", "xz", "yy", "yz", "zz"],
              9: ["trace (divergence)", "xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]}
# A 2D build's grids (NO_DIM = 2, define.h): the fields whose component count differs from 3D --
# velocity 2, the gradient 2x2, the symmetric dispersion tensor (xx, xy, yy), the traceless shear
# (xx, xy), the eigenvalue pairs; the vorticity is one number (the z component of the curl)
NCOMP_2D = {"velocity": 2, "gradient": 4, "dispersion_tensor": 3, "shear": 2, "vorticity": 1,
            "tweb_eigenvalues": 2, "vweb_eigenvalues": 2}
COMPONENTS_2D = {"velocity": ["norm", "x", "y"], "gradient": ["trace (divergence)", "xx", "xy", "yx", "yy"],
                 "dispersion_tensor": ["xx", "xy", "yy"]}


def sym_diag(dim: int) -> list[int]:
    """Where the diagonal sits in a packed symmetric tensor (xx, xy, xz, yy, yz, zz / xx, xy, yy)."""
    return [i * dim - i * (i - 1) // 2 for i in range(dim)]


def ncomp_of(name: str, dim: int = 3) -> int:
    """The components per cell of a field's grid in a 'dim'-dimensional run."""
    return NCOMP_2D.get(name, FIELDS[name][1]) if dim == 2 else FIELDS[name][1]


def components(name: str, ncomp: int, dim: int = 3) -> list[str]:
    """The component menu of a field: ['value'] for a scalar; 'norm' / 'trace ...' first when the
    field has one (they are computed, not stored)."""
    if ncomp <= 1:
        return ["value"]
    if dim == 2:
        return COMPONENTS_2D.get(name, [str(i + 1) for i in range(ncomp)])
    return COMPONENTS.get(ncomp, [str(i + 1) for i in range(ncomp)])


def grid_shape(path: Path, ncomp: int, dim: int = 3) -> tuple[int, np.dtype]:
    """(N, dtype) of a raw N^dim x ncomp grid file, from its size: float32 or float64 (a DOUBLE=1
    build). dtfelib's _grid_shape_of for any dimension; ValueError when the size fits neither."""
    size = Path(path).stat().st_size
    for dt in (np.float32, np.float64):
        item = np.dtype(dt).itemsize
        n = round((size / item / ncomp) ** (1 / dim))
        if n > 0 and n ** dim * ncomp * item == size:
            return n, np.dtype(dt)
    raise ValueError(f"{path}: {size} bytes is not an N^{dim} x {ncomp} float32 or float64 grid")


def _fits(files: dict, dim: int) -> tuple[int, int | None, np.dtype | None]:
    """(how many of an output's files are N^dim grids of one N, that N, its dtype), N from the
    first file whose size fits."""
    n = dt = None
    for name, path in files.items():
        try:
            n, dt = grid_shape(path, ncomp_of(name, dim), dim)
            break
        except ValueError:
            continue
    if n is None:
        return 0, None, None
    hits = sum(1 for name, path in files.items()
               if path.stat().st_size == n ** dim * ncomp_of(name, dim) * dt.itemsize)
    return hits, n, dt


def _look_like_stream_counts(path: Path, dtype) -> bool:
    """Do a stream-count grid's first values, read as 'dtype', look like stream counts -- whole numbers
    >= 1 in most cells (dtfelib.STREAM_TOL)? Read in the other precision they do not: float32 1.0
    pairs read as float64 are ~0.0078, a float64 1.0 read as float32 is 0 and 1.875."""
    try:
        raw = np.fromfile(path, dtype=dtype, count=65536 // np.dtype(dtype).itemsize)
    except (OSError, ValueError):
        return False
    if raw.size == 0:
        return False
    whole = np.rint(raw)
    return float(np.mean(np.isfinite(raw) & (whole >= 1) & (np.abs(raw - whole) < 1e-3))) > 0.5


def _ambiguous_dim(root: Path, plane: tuple[int, np.dtype], cube: tuple[int, np.dtype]) -> int:
    """The dimension of an output whose files read equally well as an n x n plane and an m^3 cube,
    plane = (n, dtype), cube = (m, dtype): only scalar grids (density, stream counts) and no
    settings sidecar saying which. Same bytes, two readings: n^2 = m^3 in one precision (512^2 =
    64^3, 4096^2 = 256^3), or a float32 plane against a float64 cube, n^2 = 2 m^3 (256^2 = 32^3 in
    double), and the reverse (128^2 in double = 32^3 in float)."""
    if plane[1] != cube[1]:             # the readings differ in precision: a phase-space run's stream
        for sfx in ("a_streams", "streams"):            # counts are whole numbers in the right one only
            p = root.parent / f"{root.name}.{sfx}"
            if p.is_file():
                as_plane, as_cube = _look_like_stream_counts(p, plane[1]), _look_like_stream_counts(p, cube[1])
                if as_plane != as_cube:
                    return 2 if as_plane else 3
                break
        return 3                        # no evidence: the 3D reading, as before 2D outputs existed
    # One precision (a 512^2 plane or a 64^3 cube): neighbouring cells of a real grid are alike. Read flat,
    # the cube's second axis steps m values and the plane's rows step n; in the wrong reading that stride
    # lands far away (m cells along a row of the plane, or sqrt(m) cells along the cube's second axis),
    # so the true neighbour stride shows clearly the smaller differences. A field without structure
    # across them (constant, noise, one varying along the slowest axis only) cannot tell: 3D, as before.
    # Reads the first 64 rows of the plane at most (1 MB for 4096^2 in float32).
    path = next((p for p in (root.parent / f"{root.name}.{sfx}" for sfx in ("a_den", "den")) if p.is_file()), None)
    n, m = int(plane[0]), int(cube[0])
    if path is None or n <= m:
        return 3
    try:
        x = np.fromfile(path, dtype=plane[1], count=min(n * n, 64 * n)).astype(np.float64)
    except (OSError, ValueError):
        return 3
    if x.size <= n:
        return 3
    with np.errstate(invalid="ignore"):
        across_cube = np.abs(x[m:] - x[:-m])
        across_plane = np.abs(x[n:] - x[:-n])
    d_cube = float(np.nanmedian(across_cube)) if np.isfinite(across_cube).any() else 0.0
    d_plane = float(np.nanmedian(across_plane)) if np.isfinite(across_plane).any() else 0.0
    return 2 if d_plane < 0.8 * d_cube else 3
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
# through a memmap per slice. The cache holds a quarter of the machine's memory, at most 16 GB
# (DTFE_GUI_CUBE_GB overrides): the figure grid shows four or nine outputs at once, and at the old
# 1.5 GB four 512^3 cubes evicted each other, so every slice step re-read them all (75 ms each
# instead of 1 ms; measured 2026-10-06).


def _cube_cache_bytes() -> int:
    env = os.environ.get("DTFE_GUI_CUBE_GB", "")
    if env:
        try:
            return int(float(env) * 1e9)
        except ValueError:
            pass
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        ram = 8e9
    return int(min(max(0.25 * ram, 1.5e9), 16e9))


CUBE_CACHE_BYTES = _cube_cache_bytes()
# A cube above this is read whole only when a slice runs along its last axis (a memmap touches every page
# of the file for that one anyway) or once it is cached already: a single x or y plane of a 1024^3 field
# must not cost the whole 4 GB read (the 16 GB cap alone would have made it so; review 2026-10-06).
CUBE_EAGER_BYTES = min(int(2e9), CUBE_CACHE_BYTES)
# module-wide, least recently used first; the key carries the file's mtime and size, so a grid written
# again by a re-run is read again (keyed on the path alone, Explore kept showing the OLD run's values
# until the launcher restarted -- found 2026-10-05). The slice worker and the GUI thread (Save figure...)
# both come here: every look at the dict is under the lock, the disk read is not (drop_cubes() from the
# Run button must never wait on the T7).
_CUBES: dict[tuple, np.ndarray] = {}
_CUBES_LOCK = threading.Lock()
# While the launcher runs a job the cache keeps at most RUN_CUBE_BYTES: the binary's auto-tuner budgets from the
# memory free at ITS start, and Explore stepping through slices meanwhile must not refill 16 GB under it (a
# 512^3 float32 cube, 0.5 GB, still fits: stepping stays smooth). cap_cubes() is the launcher's switch.
RUN_CUBE_BYTES = int(1e9)
_CAPPED = False


def cap_cubes(on: bool) -> None:
    """The run-time cap on (a job starts: the cache is trimmed to it) or off (the job ended)."""
    global _CAPPED
    with _CUBES_LOCK:
        _CAPPED = bool(on)
        budget = _cube_budget()
        while _CUBES and sum(a.nbytes for a in _CUBES.values()) > budget:
            _CUBES.pop(next(iter(_CUBES)))


def _cube_budget() -> int:
    return min(CUBE_CACHE_BYTES, RUN_CUBE_BYTES) if _CAPPED else CUBE_CACHE_BYTES


def _cube_key(path: Path, dtype, shape) -> tuple | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (path.resolve(), st.st_mtime_ns, st.st_size, np.dtype(dtype).str, tuple(int(v) for v in shape))


def _cached_cube(path: Path, dtype, shape, axis: int | None = None) -> np.ndarray | None:
    """The whole array of a grid file, cached, or None when the caller should memmap it instead: it is
    larger than the cache, or larger than CUBE_EAGER_BYTES and not cached yet while 'axis' (the plane
    axis the caller slices next, 0 or 1 of a 3D grid) is one a memmap reads cheaply. axis None = the
    caller reads the whole array (a 2D grid) or slices along the last axis."""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    budget = _cube_budget()
    if nbytes > budget:
        return None
    key = _cube_key(path, dtype, shape)
    if key is None:
        return None
    with _CUBES_LOCK:
        cube = _CUBES.get(key)
        if cube is not None:
            _CUBES[key] = _CUBES.pop(key)     # most recently used last
            return cube
    if nbytes > min(CUBE_EAGER_BYTES, budget) and axis is not None and axis < 2:
        return None
    cube = np.fromfile(path, dtype=dtype).reshape(shape)         # outside the lock: nobody waits on the disk
    with _CUBES_LOCK:
        other = _CUBES.get(key)
        if other is not None:                 # another thread read it meanwhile: one copy, theirs
            _CUBES[key] = _CUBES.pop(key)
            return other
        for stale in [k for k in _CUBES if k[0] == key[0]]:      # the same file, another version: drop it
            _CUBES.pop(stale)
        if nbytes > _cube_budget():           # the cap came on while this was read (a job started): use it, keep none
            return cube
        while _CUBES and sum(a.nbytes for a in _CUBES.values()) + nbytes > _cube_budget():
            _CUBES.pop(next(iter(_CUBES)))
        _CUBES[key] = cube
    return cube


def _read_plane(path: Path, dtype, n: int, ncomp: int, axis: int, index: int) -> np.ndarray:
    """One plane of an (n, n, n[, ncomp]) C-order grid file by plain reads (no memmap: see OutputSet._memmap):
    axis 0 is one contiguous block, axis 1 one block per row, axis 2 the file in slabs of whole x-layers (the
    same bytes a memmap would touch), keeping one plane. Raises ValueError for a short file, OSError when
    the file cannot be read (a drive gone)."""
    dt = np.dtype(dtype)
    if not 0 <= index < n:
        raise ValueError(f"plane {index} outside 0..{n - 1}")
    row = n * ncomp                         # values per (i, j) row along the last axis
    layer = n * row                         # values per x-layer i
    out = np.empty((n, n, ncomp) if ncomp > 1 else (n, n), dtype=dt)
    with open(path, "rb") as f:
        def block(offset_values: int, count: int) -> np.ndarray:
            f.seek(offset_values * dt.itemsize)
            a = np.fromfile(f, dtype=dt, count=count)
            if a.size != count:
                raise ValueError(f"{Path(path).name} is shorter than its grid (an interrupted write?)")
            return a
        if axis == 0:
            return block(index * layer, layer).reshape(out.shape)
        if axis == 1:
            for i in range(n):
                out[i] = block(i * layer + index * row, row).reshape(out.shape[1:])
            return out
        step = max(1, int(64e6 // (layer * dt.itemsize)))      # ~64 MB of x-layers per read
        for i0 in range(0, n, step):
            k = min(step, n - i0)
            slab = block(i0 * layer, k * layer).reshape((k, n, n, ncomp) if ncomp > 1 else (k, n, n))
            out[i0:i0 + k] = slab[:, :, index]
        return out


def drop_cubes() -> int:
    """Forget every cached cube (the bytes they held). The launcher calls it when a job or a memory check
    starts: the cache is anonymous memory of the GUI process, and the binary's auto-tuner budgets from
    the memory that is free at that moment."""
    with _CUBES_LOCK:
        n = sum(a.nbytes for a in _CUBES.values())
        _CUBES.clear()
    return n


class OutputSet:
    """The grid files of one run: '<dir>/<prefix>.*'. 'dim' 2 or 3, else the settings sidecar's,
    else read off the file sizes (a 2D grid is n^2 cells, its vectors and tensors have fewer
    components: a velocity file alone tells 2D from 3D)."""

    def __init__(self, root, box: tuple | None = None, redshift: float | None = None, title: str = "",
                 dim: int | None = None, sim: str = "", snap: int | None = None):
        self.root = Path(root)
        self.dir, self.prefix = self.root.parent, self.root.name
        self.title = title or f"{self.dir.name}/{self.prefix}"
        self.sim, self.snap = sim, snap          # the TNG layout's simulation and snapshot number ("" / None otherwise)
        found: dict[str, tuple[Path, bool]] = {}
        for name, (suffix, _ncomp, _ps) in FIELDS.items():
            for sfx in (suffix, LEGACY_SUFFIX.get(name)):   # the current name, then a renamed one
                for averaged in (True, False):              # prefer the volume-averaged grid
                    p = self.dir / f"{self.prefix}.{'a_' if averaged else ''}{sfx}"
                    if sfx and p.is_file() and name not in found:
                        found[name] = (p, averaged)
        paths = {name: p for name, (p, _a) in found.items()}
        if dim not in (2, 3):
            side = sidecar(self.root) or {}
            dim = side.get("dim") if side.get("dim") in (2, 3) else None
        fit3, n3, dt3 = _fits(paths, 3)
        fit2, n2, dt2 = _fits(paths, 2)
        if dim not in (2, 3):
            dim = 3 if fit3 > fit2 else 2 if fit2 > fit3 else \
                (_ambiguous_dim(self.root, (n2, dt2), (n3, dt3)) if fit3 else 3)
        self.dim = int(dim)
        self.n, self.dtype = (n3, dt3) if self.dim == 3 else (n2, dt2)
        self.files: dict[str, FieldFile] = {name: FieldFile(name, p, ncomp_of(name, self.dim), averaged)
                                            for name, (p, averaged) in found.items()}
        # a file whose size does not match the grid (an interrupted write) is left out
        if self.n:
            self.files = {name: ff for name, ff in self.files.items()
                          if ff.path.stat().st_size == self.n ** self.dim * ff.ncomp * self.dtype.itemsize}
        else:
            self.files = {}
        self.box = box                  # ((xlo, ylo[, zlo]), (xhi, yhi[, zhi])) in Mpc, or None
        self.redshift = redshift

    def __bool__(self):
        return bool(self.files) and bool(self.n)

    def fields(self) -> list[str]:
        return [n for n in FIELD_LABELS if n in self.files]

    # lo and length always have three entries: a 2D grid is the plane z = 0 of zero thickness, so the
    # slice machinery (an axis, an index along it) treats it as the one z slice it has
    @property
    def lo(self) -> np.ndarray:
        if self.box is None:
            return np.zeros(3)
        lo = np.asarray(self.box[0], float)
        return np.append(lo, 0.0) if lo.size == 2 else lo

    @property
    def length(self) -> np.ndarray:
        if self.box is None:
            out = np.full(3, float(self.n or 1))        # unknown box: cell units
        else:
            out = np.asarray(self.box[1], float) - np.asarray(self.box[0], float)
            if out.size == 2:
                out = np.append(out, 0.0)
        if self.dim == 2:
            out[2] = 0.0
        return out

    def _memmap(self, name: str, axis: int | None = None, index: int = 0) -> np.ndarray:
        """The field's array -- the cached cube when _cached_cube() keeps one -- or, for a 3D grid, just the
        plane 'index' along 'axis', read from the file. NEVER a memmap: a mapped page of a drive unplugged
        meanwhile (the T7) kills the whole launcher with SIGBUS, which Python cannot catch; a read fails with
        an OSError the slice worker reports (2026-10-06). A 2D grid is read whole."""
        ff = self.files[name]
        shape = (self.n,) * self.dim + ((ff.ncomp,) if ff.ncomp > 1 else ())
        cube = _cached_cube(ff.path, self.dtype, shape, axis if self.dim == 3 else None)
        if cube is not None:
            return cube
        if self.dim == 2 or axis is None:
            return np.fromfile(ff.path, dtype=self.dtype).reshape(shape)
        return _read_plane(ff.path, self.dtype, self.n, ff.ncomp, int(axis), int(index))

    def _velocity_factor(self, name: str) -> float:
        exp = _VELOCITY_SCALE_EXP.get(name)
        if exp is None or self.redshift is None:
            return 1.0
        return (1.0 / (1.0 + self.redshift)) ** exp       # u-units -> peculiar km/s (FieldSet)

    def slice(self, name: str, axis: int = 2, index: int | None = None, component: str = "norm") -> np.ndarray:
        """One plane of a field, (N, N) float64, rows = the first remaining axis (a 2D grid: the whole
        grid, rows = x; axis and index are ignored). Components of a vector/tensor field by name
        (components()); 'norm' = |v|, 'trace ...' = the trace. The vorticity is the curl of the
        velocity: the binary stores the antisymmetric part of the gradient, W_ab = (dv_a/dx_b -
        dv_b/dx_a)/2 for a < b (xy, xz, yz), which this turns into the curl's x, y, z (a 2D grid: its
        z, a single number); measured on a rigid rotation (curl = 2 Omega), 2026-10-03."""
        index = self.n // 2 if index is None else int(index)
        mm = self._memmap(name, int(axis), index)
        if self.dim == 2:
            plane = np.array(mm)            # the file's precision: only the component shown is up-cast (below)
        elif mm.ndim == self.dim + (1 if self.files[name].ncomp > 1 else 0):     # the cached cube: cut the plane
            sl = [slice(None)] * 3
            sl[axis] = index
            plane = np.array(mm[tuple(sl)])
        else:                               # already the plane (read from the file)
            plane = mm
        del mm
        ff = self.files[name]
        f64 = lambda a: np.asarray(a, dtype=np.float64)        # noqa: E731
        if name == "vorticity":         # [W_xy, W_xz, W_yz] = [-w_z, w_y, -w_x] / 2 -> the curl w
            plane = -2.0 * f64(plane) if ff.ncomp == 1 else \
                2.0 * np.stack([-f64(plane[..., 2]), f64(plane[..., 1]), -f64(plane[..., 0])], -1)
        if ff.ncomp > 1:
            comps = components(name, ff.ncomp, self.dim)
            if component not in comps:
                component = comps[0]
            if component == "norm":     # component by component in float64: never the whole plane at once
                acc = np.zeros(plane.shape[:-1])
                for i in range(plane.shape[-1]):
                    c = f64(plane[..., i])
                    acc += c * c
                plane = np.sqrt(acc)
            elif component.startswith("trace"):
                d = int(round(ff.ncomp ** 0.5))
                plane = sum(f64(plane[..., i * d + i]) for i in range(d))
            else:
                idx = comps.index(component) - (1 if comps[0] == "norm" or comps[0].startswith("trace") else 0)
                plane = f64(plane[..., idx])
        return f64(plane) * self._velocity_factor(name)

    def arrow_field(self, name: str, axis: int = 2, index: int | None = None, per_side: int = 32,
                    smooth: float = 0.0) -> dict | None:
        """Arrows for a vector field on one slice, in IMAGE coordinates (the Explore image is
        plane.T[::-1]: x = the first in-plane axis rightward, y = the second, drawn upward, in
        cell units from the image's top-left corner). The two in-plane components are
        block-averaged to about 'per_side' arrows along each side; dx, dy are scaled so the
        longest arrow is 1 (times 'spacing' cells when drawn). None for a non-vector field."""
        ff = self.files.get(name)
        if ff is None or ff.ncomp != self.dim:          # a vector: one component per axis
            return None
        rest = [0, 1] if self.dim == 2 else [a for a in range(3) if a != axis]
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
        """The value of one cell (same component convention as slice); a 2D grid's cell is (i, j)."""
        if self.dim == 2:
            i, j = (int(v) for v in list(ijk)[:2])
            return float(self.slice(name, 2, 0, component)[i, j])
        i, j, k = (int(v) for v in ijk)
        return float(self.slice(name, 0, i, component)[j, k])

    # -- coordinates
    def cell_center(self, ijk) -> np.ndarray:
        return self.lo + (np.asarray(ijk, float) + 0.5) * self.length / self.n

    def plane_point(self, axis: int, index: int, row: int, col: int) -> np.ndarray:
        """The 3D cell (i, j, k) of pixel (row, col) of slice 'index' along 'axis' (a 2D grid: (i, j))."""
        if self.dim == 2:
            return np.asarray([row, col])
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


def _scaled(plane: np.ndarray, log: bool) -> np.ndarray:
    """The values a map shows: log10 of the positive ones (the rest at the bottom of the scale)."""
    data = np.asarray(plane, dtype=np.float64)
    if log:
        pos = data[data > 0]
        floor = pos.min() if pos.size else 1.0
        data = np.log10(np.where(data > 0, data, floor))
    return data


def value_range(plane: np.ndarray, log: bool = False, clip: tuple = (1.0, 99.0),
                symmetric: bool = False) -> tuple[float, float]:
    """The (vmin, vmax) colorize() would show: the percentiles 'clip' of the finite scaled values,
    'symmetric' centred on 0 (the figure grid shares such ranges between panels)."""
    data = _scaled(plane, log)
    finite = data[np.isfinite(data)]
    if finite.size:
        vmin, vmax = np.percentile(finite, clip)
    else:
        vmin, vmax = 0.0, 1.0
    if symmetric:                   # the percentiles of a map that is 0 almost everywhere are 0: then the largest
        m = max(abs(vmin), abs(vmax)) or (float(np.abs(finite).max()) if finite.size else 0.0) or 1.0   # value
        vmin, vmax = -m, m
    if not vmax > vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def colorize(plane: np.ndarray, cmap: str = "viridis", log: bool = False,
             clip: tuple = (1.0, 99.0), symmetric: bool = False,
             vrange: tuple | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """plane -> (H, W, 3) uint8 RGB and the (vmin, vmax) shown. log10 of positive values (the rest
    at the bottom of the scale); range from the percentiles 'clip' unless 'vrange' is given;
    'symmetric' centres a diverging map on 0."""
    data = _scaled(plane, log)
    if vrange is not None:
        vmin, vmax = vrange
        if symmetric:
            m = max(abs(vmin), abs(vmax)) or 1.0
            vmin, vmax = -m, m
        if not vmax > vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = value_range(plane, log, clip, symmetric)
    idx = np.clip((data - vmin) / (vmax - vmin) * 255.0, 0, 255)
    idx = np.where(np.isfinite(idx), idx, 0).astype(np.uint8)
    return lut(cmap)[idx], (float(vmin), float(vmax))


# ---------------------------------------------------------------------- ticks (the Explore axes and colour bar)
TICK_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)        # the round steps, per decade


def nice_ticks(lo: float, hi: float, n: int = 5, integer: bool = False) -> list[float]:
    """About n round values inside [lo, hi], a step of TICK_STEPS x 10^k apart (0, 20, 40 ...; 0.25, 0.5 ...).
    'integer': whole steps of 1, 2, 5 x 10^k only -- an axis in cells (0, 5, 10 on a 16-cell grid, not 0, 2.5,
    5: a cell boundary means nothing). Empty for a degenerate range."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo or n < 1:
        return []
    raw = (hi - lo) / n
    mag = 10.0 ** np.floor(np.log10(raw))
    steps = [m * mag for m in TICK_STEPS if not (integer and m == 2.5)]
    if integer:
        steps = [max(st, 1.0) for st in steps]
    step = min(steps, key=lambda st: abs(np.log(st / raw)))    # the nearest round step
    k0, k1 = int(np.ceil(lo / step - 1e-9)), int(np.floor(hi / step + 1e-9))
    digits = max(0, 2 - int(np.floor(np.log10(step))))
    return [float(round(k * step, digits)) + 0.0 for k in range(k0, k1 + 1)]   # + 0.0: no '-0'


def tick_text(v: float, ticks: list[float]) -> str:
    """A tick label with the decimals the step needs (and no more): 0.25 / 0.5 -> '0.25', '0.50'."""
    if len(ticks) > 1:
        step = float(f"{min(abs(b - a) for a, b in zip(ticks, ticks[1:])):.3g}")   # 0.000999.. is 0.001
        if step >= 1e5 or step <= 0:
            return f"{v:.3g}"
        d = next((k for k in range(13) if abs(step * 10 ** k - round(step * 10 ** k)) < 1e-6 * max(1.0, step * 10 ** k)), 12)
        return f"{v:.{d}f}"                         # the fewest decimals that show every tick exactly (2.5 -> 1)
    return f"{v:.3g}"


def nice_length(span: float, frac: float = 0.2) -> float:
    """The scale bar's round length (1, 2 or 5 x 10^k) nearest frac * span (log distance)."""
    if not (np.isfinite(span) and span > 0):
        return 0.0
    target = frac * span
    mag = 10.0 ** np.floor(np.log10(target))
    cands = [m * mag for m in (1.0, 2.0, 5.0, 10.0)]
    return float(min(cands, key=lambda c: abs(np.log(c / target))))


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def decade_text(k: int) -> str:
    """10^k as the colour bar writes it: '1', '10', '10²', '10⁻¹'."""
    return {0: "1", 1: "10"}.get(k, "10" + str(k).translate(_SUPERSCRIPT))


def colorbar_ticks(lo: float, hi: float, log: bool = False, n: int = 5) -> list[tuple[float, str]]:
    """(position in the bar's units, label) for a colour bar spanning lo..hi. On a log map (lo, hi are
    log10 values) the whole decades inside, labelled 10^k (thinned to about n); with fewer than two
    decades, round log values labelled with their linear value."""
    if log:
        ks = list(range(int(np.ceil(lo - 1e-9)), int(np.floor(hi + 1e-9)) + 1))
        if len(ks) >= 3:
            every = max(1, int(np.ceil(len(ks) / n)))
            return [(float(k), decade_text(k)) for k in ks if k % every == 0]
        # under three decades: 1, 2, 5 x 10^k as plain numbers (0.05, 0.1, 0.2), else round log values
        sub = [(np.log10(m) + k, m, k) for k in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 1) for m in (1, 2, 5)]
        sub = [(float(v), f"{m * 10.0 ** k:g}") for v, m, k in sub if lo - 1e-9 <= v <= hi + 1e-9]
        if len(sub) >= 2:
            return sub
        return [(v, f"{10.0 ** v:.3g}") for v in nice_ticks(lo, hi, n)]
    ticks = nice_ticks(lo, hi, n)
    return [(v, tick_text(v, ticks)) for v in ticks]


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


PS_PREFIX, DTFE_PREFIX = "ps_output", "output"       # the run scripts' default output prefixes


def estimator_label(o) -> str:
    """'PS-DTFE' or 'DTFE' for an output. The run's sidecar says (a custom-snapshot job records its
    estimator); else a streams grid decides -- only the phase-space binary writes one; else the prefix
    (the scripts' ps_* are phase-space). A phase-space run of the density alone has no streams grid, so
    the absence of one never means DTFE on its own. Cached on the object."""
    cached = getattr(o, "_estimator", None)
    if cached:
        return cached
    side = sidecar(o.root) or {}
    if side.get("estimator") in ("ps", "dtfe"):
        lab = "PS-DTFE" if side["estimator"] == "ps" else "DTFE"
    elif "streams" in o.files:
        lab = "PS-DTFE"
    else:
        lab = "PS-DTFE" if o.prefix.startswith("ps") else "DTFE"
    o._estimator = lab
    return lab


def type_label(o) -> str:
    """The figure grid's Type entry: the estimator, with the prefix when it is not that estimator's
    default one, so two phase-space runs of one snapshot stay apart ('PS-DTFE', 'PS-DTFE · ps_mw', 'DTFE')."""
    lab = estimator_label(o)
    default = PS_PREFIX if lab == "PS-DTFE" else DTFE_PREFIX
    return lab if o.prefix == default else f"{lab} · {o.prefix}"


def type_order(o, pref=None) -> tuple:
    """Sort key of a snapshot's outputs for the Type menu and the composer's fills: the estimator (DTFE
    first), then that estimator's default prefix, then the user's preferred one ('pref'), then the rest
    by prefix -- plain text order put ps_mw (an A/B run kept beside ps_output) before the production
    run, so 'this snapshot across outputs' compared the standard DTFE with the wrong phase-space run
    (review 2026-10-05)."""
    lab = estimator_label(o)
    default = PS_PREFIX if lab == "PS-DTFE" else DTFE_PREFIX
    return (lab, o.prefix != default, o.prefix != pref, o.prefix)


def custom_box(settings: dict) -> tuple | None:
    """A custom-snapshot run's box in Mpc: 'box_mpc', else its --box (file units) / MpcUnit, else the
    input snapshot's header BoxSize / MpcUnit. Two corners of 'dim' (the sidecar's, 3 if absent) entries."""
    unit = float(settings.get("mpc_unit") or 1.0)
    dim = settings.get("dim") if settings.get("dim") in (2, 3) else 3
    box = settings.get("box_mpc") or []
    if len(box) != 2 * dim and len(settings.get("box") or []) == 2 * dim:
        box = [v / unit for v in settings["box"]]
    if len(box) != 2 * dim and settings.get("input_file"):
        try:
            import h5py
            with h5py.File(settings["input_file"], "r") as h:
                b = float(h["Header"].attrs["BoxSize"]) / unit
            box = [0.0, b] * dim
        except Exception:
            box = []
    if len(box) == 2 * dim:
        return tuple(box[0::2]), tuple(box[1::2])
    return None


def sim_sort_key(name: str) -> tuple:
    """Natural order for simulation names: TNG50 before TNG100 before TNG300 (text sorting puts
    TNG300 first, since '3' < '5')."""
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name))


def find_outputs(data_root, extra_dirs=()) -> list[tuple[str, OutputSet]]:
    """(menu title, OutputSet) for every run with grids: the TNG layout under the data root, then
    the custom-snapshot output directories."""
    out = []
    root = Path(data_root)
    seen = set()
    if root.is_dir():
        for pattern in ("*/snapdir_*/*.a_den", "*/*/snapdir_*/*.a_den", "*/snapdir_*/*.den", "*/*/snapdir_*/*.den"):
            for f in sorted(root.glob(pattern)):
                prefix = f.stem                 # 'snap_z0.5.a_den' -> 'snap_z0.5' (no grid suffix has a dot)
                key = f.parent / prefix
                if key in seen:
                    continue
                seen.add(key)
                box, z = _tng_box(f.parent)
                snap = f.parent.name.split("_")[-1]
                title = f"{f.parent.parent.name} · {snap}" + (f" (z={z:.2f})" if z is not None else "") + f" · {prefix}"
                o = OutputSet(key, box=box, redshift=z, title=title, sim=f.parent.parent.name,
                              snap=int(snap) if snap.isdigit() else None)
                if o:
                    out.append((title, o))
    for d in extra_dirs:
        for f in sorted(Path(d).glob("*.a_den")) + sorted(Path(d).glob("*.den")):
            prefix = f.stem
            key = f.parent / prefix
            if key in seen:
                continue
            seen.add(key)
            s = sidecar(key) or {}
            o = OutputSet(key, box=custom_box(s), redshift=None, title=f"custom · {prefix} ({key.parent.name})",
                          dim=s.get("dim"))
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


# ---------------------------------------------------------------- the tessellation cache on disk
# What the query server (and any run with --tessellation-cache) leaves in its cache folder, read
# without the binary: each partition's tessellation 'tess_<hash>_p<k>.tess' starts with its full
# descriptor (tessellation_cache.h makeDescriptor), and beside it '<that>.occ' holds the partition's
# 64^3 occupancy map -- the buckets of the box its streams reach (ps_point_eval.cc).
_TESS_MAGIC = b"DTFETESS"
_OCC_MAGIC = b"DTFEOCC1"
OCC_N = 64


def _read_descriptor(f, magic: bytes) -> str | None:
    import struct
    if f.read(8) != magic:
        return None
    raw = f.read(4)
    if len(raw) != 4:
        return None
    (n,) = struct.unpack("<I", raw)
    if not 0 < n < (1 << 20):
        return None
    return f.read(n).decode("utf-8", errors="replace")


def _descriptor_fields(desc: str) -> dict:
    out = {}
    for line in desc.splitlines():
        key, _, val = line.partition("=")
        if key and key not in out:
            out[key] = val
    return out


def _canonical(path) -> str:
    """The binary's canonical_path.h: realpath, so a symlinked or relative name matches the cache key
    the binary wrote (it canonicalizes since 2026-10-02); an unresolvable path is kept as given."""
    try:
        return os.path.realpath(os.path.expanduser(str(path)))
    except OSError:
        return str(path)


def _file_stamp(path) -> str:
    """The C++ fileStamp: '<size>:<mtime seconds>'."""
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    return f"{st.st_size}:{int(st.st_mtime)}"


def cached_partition_sets(cache_dir, snapshot, lagrangian=None, periodic: bool | None = None) -> dict:
    """This snapshot's partition tessellations in a tessellation cache, by partition triple (nx, ny, nz)
    (a pair for a 2D run):
    a list of {'tess', 'occ' (Path or None), 'bytes', 'region' (6 floats), 'periodic'} per cached
    partition. Matched on the descriptor's input line (path and size:mtime stamp), the Lagrangian file
    and the periodicity -- only to choose a split and estimate a run: the binary itself compares the
    whole descriptor before it uses a file."""
    sets: dict = {}
    d = Path(cache_dir).expanduser() if cache_dir else None
    if d is None or not d.is_dir():
        return sets
    want_input = f"{_canonical(snapshot)} {_file_stamp(snapshot)}"
    for tess in sorted(d.glob("tess_*.tess")):
        try:
            with open(tess, "rb") as f:
                desc = _read_descriptor(f, _TESS_MAGIC)
        except OSError:
            continue
        if not desc:
            continue
        fl = _descriptor_fields(desc)
        if fl.get("input") != want_input:
            continue
        if lagrangian and not fl.get("lagrange", "").startswith(f"{_canonical(lagrangian)} "):
            continue
        per = fl.get("periodic", "1").strip() == "1"
        if periodic is not None and per != bool(periodic):
            continue
        part = fl.get("partition", "")
        try:
            on, triple = part.split(" ", 1)
            nums = tuple(int(v) for v in triple.split(" ")[0].strip(",").split(",") if v)
        except ValueError:
            continue
        if on != "1" or len(nums) not in (2, 3):         # a 2D run's split has two numbers
            continue
        try:
            region = [float(v) for v in fl.get("region", "").split(" ", 1)[1].strip(",").split(",")]
        except (IndexError, ValueError):
            region = []
        occ = Path(str(tess) + ".occ")
        sets.setdefault(nums, []).append({"tess": tess, "occ": occ if occ.is_file() else None,
                                          "bytes": tess.stat().st_size, "region": region, "periodic": per})
    return sets


def tess_chunked(path) -> bool:
    """A cached tessellation written with the chunked body (2026-10-02 on: inflated on several threads)."""
    try:
        with open(path, "rb") as f:
            if _read_descriptor(f, _TESS_MAGIC) is None:
                return False
            return f.read(8) == b"DTFECHK1"
    except OSError:
        return False


def globals_recorded(cache_dir, snapshot, lagrangian=None) -> bool:
    """Is this snapshot's particle-derived globals record (ps_globals_record.h) in the cache? A window
    run whose needed partitions are all cached then never reads the snapshot. Matched on the record's
    own input and Lagrangian lines (paths and size:mtime stamps); the binary compares its whole key."""
    d = Path(cache_dir).expanduser() if cache_dir else None
    if d is None or not d.is_dir():
        return False
    want_input = f"input={_canonical(snapshot)} {_file_stamp(snapshot)}"
    want_lag = f"lagrange={_canonical(lagrangian)} {_file_stamp(lagrangian)}" if lagrangian else None
    for rec in d.glob("globals_*.txt"):
        try:
            lines = rec.read_text(errors="replace").splitlines()
        except OSError:
            continue
        if want_input in lines and (want_lag is None or want_lag in lines) and "end" in lines:
            return True
    return False


def read_occupancy(path):
    """(n vertices, the 64^3 occupancy bits as a bool array [i, j, k]) of a '.occ' map, or None."""
    import struct
    try:
        with open(path, "rb") as f:
            if _read_descriptor(f, _OCC_MAGIC) is None:
                return None
            n, nvert, words = struct.unpack("<IQQ", f.read(20))
            if n != OCC_N or words not in (0, OCC_N ** 3 // 64):
                return None
            raw = np.frombuffer(f.read(8 * words), dtype="<u8")
    except (OSError, struct.error):
        return None
    if words == 0:
        return nvert, np.zeros((OCC_N,) * 3, bool)
    bits = np.unpackbits(raw.view(np.uint8), bitorder="little").astype(bool)
    return nvert, bits[: OCC_N ** 3].reshape((OCC_N,) * 3)


def occupancy_touches(occ_bits, region, periodic: bool, window) -> bool:
    """Does an occupancy map reach a window ((3, 2) in Mpc)? The binary's own test (ps_point_eval.cc
    psOccupancyTouchesWindow): the window's bucket range widened by one bucket on each side."""
    idx = []
    for d in range(3):
        lo, length = region[2 * d], region[2 * d + 1] - region[2 * d]
        w = OCC_N / length
        b0 = int(np.floor((window[d][0] - lo) * w)) - 1
        b1 = int(np.floor((window[d][1] - lo) * w)) + 1
        if periodic:
            if b1 - b0 + 1 >= OCC_N:
                b0, b1 = 0, OCC_N - 1
            idx.append(np.arange(b0, b1 + 1) % OCC_N)
        else:
            idx.append(np.arange(max(0, min(b0, OCC_N - 1)), max(0, min(b1, OCC_N - 1)) + 1))
    return bool(occ_bits[np.ix_(*idx)].any())


def find_pts_planes(o) -> list[tuple[str, Path]]:
    """The point-evaluated hi-res slices of an output: '<root>.pts_den' and every plane sidecar (tools/
    make_image_plane.py, in the simulation's folder or beside the output) whose geometry matches the file's
    size -- (label, sidecar). The '.pts_*' files carry no axis, so planes of one size (the x, y, z views)
    are all listed, each with its axis: the user picks."""
    den = Path(str(o.root) + ".pts_den")
    if not den.is_file():
        return []
    size = den.stat().st_size
    out, seen = [], set()
    for side in (sorted(o.dir.parent.glob("pointeval_plane_*.json")) + sorted(o.dir.parent.glob("hires_plane_*.json"))
                 + sorted(o.dir.glob("*plane*.json"))):
        if side in seen:
            continue
        seen.add(side)
        try:
            d = json.loads(side.read_text())
            k = int(d.get("supersample", 1))
            n = int(d["planes"]) * int(d["nu"]) * k * int(d["nv"]) * k
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if n * 8 == size:
            out.append((f"{d['nu']}×{d['nv']} {d.get('axis', '?')}-plane at {float(d.get('center', 0.0)):.1f} Mpc "
                        f"({side.stem})", side))
    return out


def server_settings(out: OutputSet, snapshot: str | None = None) -> dict | None:
    """dtfelib.Estimator arguments for the snapshot behind an output set, or None when unknown.
    TNG layout: the snapdir's combined file (+ the simulation's converted ICs when present);
    a custom-snapshot run: the job's own settings sidecar. 'snapshot': a file chosen by hand (Explore's
    'Snapshot…', when the input moved or the output has no sidecar) -- it replaces the one found."""
    s = sidecar(out.root)
    if snapshot and not Path(snapshot).is_file():
        return None
    if s:
        if snapshot:
            s = dict(s, input_file=str(snapshot))
        if not s.get("input_file") or not Path(s["input_file"]).is_file():
            return None
        dim = s.get("dim") if s.get("dim") in (2, 3) else 3
        kw = {"snapshot": s["input_file"], "phase_space": s.get("estimator", "ps") == "ps",
              "periodic": bool(s.get("periodic", True)), "mpc_unit": float(s.get("mpc_unit", 1.0)),
              "input_type": int(s.get("input_type", 105))}
        if dim == 2:
            kw["dim"] = 2               # the 2D server (Estimator: the -2d binary, 2-column points)
        if s.get("lagrangian_file"):
            kw["lagrangian_input"] = s["lagrangian_file"]
        if len(s.get("box") or []) == 2 * dim:
            kw["options"] = ["--box"] + [repr(float(v)) for v in s["box"]]
        # a non-periodic cloud's alpha shape, as the run (and its exact zoom) used it: the server's
        # answers and its cached occupancy maps then keep the same tetrahedra
        if not kw["periodic"] and kw["phase_space"] and float(s.get("alpha_shape", 3.0)) != 3.0:
            kw["options"] = list(kw.get("options", [])) + ["--ps-alpha-shape", f"{float(s['alpha_shape']):g}"]
        return kw
    combined = [Path(snapshot)] if snapshot else sorted(out.dir.glob("combined_[0-9][0-9][0-9].hdf5"))
    if not combined:
        return None
    kw = {"snapshot": str(combined[0]), "phase_space": not out.prefix == "output", "periodic": True}
    ics = combined[0].parent.parent / "combined_ics.hdf5"
    if kw["phase_space"] and ics.is_file():
        kw["lagrangian_input"] = str(ics)
    return kw
