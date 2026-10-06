"""The GUI's Explore tab: look at any run's grids without writing a plot script, and ask the
snapshot's own tessellation what is at a point.

  ExploreControls   the left pane: which output, field, component, axis and slice, colours,
                    and the point-query server (a dtfelib.Estimator on the snapshot behind the
                    output -- a composite of partitions, out of core, when one tessellation would
                    not fit in memory)
  ExploreView       the right pane: the slice (any size, scaled to fit), hover readout, colour
                    bar, and the answer for a clicked point: its streams, each with its density,
                    velocity and Lagrangian origin q(x)

The data side (grids.py) is Qt-free; this module only draws and wires. Slices load and queries
run on worker threads, so a 1024^3 z-plane or a tessellation build never freezes the window.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PySide6.QtCore import QLineF, QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFontMetrics, QImage, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSizePolicy, QSlider,
    QSpinBox, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grids as G  # noqa: E402
import runspec as rs  # noqa: E402

CMAPS = ["viridis", "magma", "inferno", "cividis", "plasma", "RdBu_r", "coolwarm", "gray", "tab10"]
AXES = ["x", "y", "z"]
# the figure grid's layouts (rows, columns): a field at several redshifts side by side, or several fields of one
# snapshot, composed as one figure (2026-10-05, Luuk's request)
GRID_LAYOUTS = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)]
GRID_NONE = ""
GRID_COLS = ("Simulation", "Redshift", "Type", "Field", "Component")      # the composer's table
_C_SIM, _C_SNAP, _C_TYPE, _C_FLD, _C_CMP = range(5)              # the output combo's first entry: an empty panel
GRID_MERGE_TIP = ("The panels touch (no gaps, shared axes, tick labels on the outer edges only), each carries a small "
                  "label with what differs between them, and the colour bars sit in one column at the right: one bar "
                  "when every panel shows the same field, one per field otherwise. Possible only when every panel "
                  "comes from a box of the same size; 'one colour range per field' is implied.")
# the Output menus' first step is a simulation, or one of these two groups (no simulation has a space
# in its name): the custom-snapshot runs, and the grid files opened by hand with the … button
GROUP_CUSTOM = "custom snapshots"
GROUP_OPENED = "opened files"


def default_cache_dir(near: str) -> Path:
    """Where the query server keeps its tessellations: beside the data they come from (a TNG
    output: the data root; your own file's output: its folder) -- that disk has the room, ~0.15 KB
    per tessellation vertex -- unless it is iCloud Drive or read-only, else the user cache."""
    root = Path(near)
    if root.is_dir() and "Mobile Documents" not in str(root.resolve()) and os.access(root, os.W_OK):
        return root / ".dtfe-tessellation-cache"
    base = Path.home() / ("Library/Caches" if sys.platform == "darwin" else ".cache")
    return base / "DTFE" / "tessellations"


# ==================================================================== the difference map (survey item 17)
COMPARE_MODES = (("A − B", "diff"), ("log₁₀ A/B", "ratio"))
NAN_GREY = (150, 150, 150)                 # a cell without a value (a ratio where A or B <= 0): not the scale's end


def compare_planes(a, b, mode: str) -> np.ndarray:
    """The difference map of two slices of the same grid: A - B, or log10(A/B) where both are positive
    (NaN elsewhere, drawn grey)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"the two outputs' slices differ in shape: {a.shape} and {b.shape}")
    if mode == "ratio":
        ok = (a > 0) & (b > 0)
        return np.where(ok, np.log10(np.where(ok, a, 1.0) / np.where(ok, b, 1.0)), np.nan)
    return a - b


# Explore field -> dtfelib.io.PointPlane field of a point-evaluated hi-res slice (2026-10-06)
PTS_FIELDS = {"density": "density", "streams": "streams", "velocity": "velocity", "dispersion": "dispersion",
              "divergence": "velDiv", "shear": "velShear", "vorticity": "velVort", "caustic": "caustic"}
PTS_MAX_SIDE = 4096                        # an 8192^2 plane is shown block-averaged to 4096^2 (screen and memory)
NO_SNAPSHOT_TEXT = "no snapshot known for this output (its input file is not beside it): choose it with Snapshot…"


def pts_plane(o, side, field: str, component: str):
    """(plane[u, v] for the map, the sidecar dict, the block factor) of a point-evaluated hi-res slice: the same
    quantity as the grid field where both exist -- the dispersion as the trace sigma^2 (the grid's '.velDisp'),
    the caustic as the fold flag, the velocity as its norm or a component; the shear and the vorticity as the
    magnitudes PointPlane derives."""
    from dtfelib.io import PointPlane
    pp = PointPlane(o.root, side, redshift=o.redshift)
    name = PTS_FIELDS.get(field)
    if name is None or name not in pp.available():
        raise ValueError(f"this hi-res slice has no {G.FIELD_LABELS.get(field, field)}"
                         + (" (it needs the velocity gradient points)" if field in ("divergence", "shear", "vorticity")
                            else ""))
    arr = pp.field(name)
    if field == "dispersion":
        arr = arr[..., 0] + arr[..., 3] + arr[..., 5]          # xx + yy + zz = sigma^2
    elif field == "caustic":
        arr = ((np.asarray(arr).astype(np.int64) & 3) == 3).astype(np.float64)
    elif arr.ndim == 3:
        comps = G.components("velocity", 3, 3)
        if component in comps[1:]:
            arr = arr[..., comps.index(component) - 1]
        else:
            arr = np.sqrt((np.asarray(arr, dtype=np.float64) ** 2).sum(axis=-1))
    arr = np.asarray(arr, dtype=np.float64)
    f = max(1, int(np.ceil(max(arr.shape) / PTS_MAX_SIDE)))
    if f > 1:                                  # block mean (a mask: any)
        nv, nu = (arr.shape[0] // f) * f, (arr.shape[1] // f) * f
        blocks = arr[:nv, :nu].reshape(nv // f, f, nu // f, f)
        arr = blocks.max(axis=(1, 3)) if field == "caustic" else blocks.mean(axis=(1, 3))
    return np.ascontiguousarray(arr.T), pp.side, f       # [u, v]: the slice map's plane[row = first in-plane axis]


def _smoothed(plane, sigma: float):
    """Gaussian smoothing in cells (periodic); unchanged without scipy."""
    if sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(plane, sigma, mode="wrap")
        except ImportError:
            pass
    return plane


# ==================================================================== workers
class SliceWorker(QObject):
    """Reads and colours one slice off the GUI thread; only the latest request is served. The figure
    grid's tiles come the same way (render_grid): coloured straight to pixels, no matplotlib."""
    done = Signal(object)       # dict: rgb, plane, vrange, request
    failed = Signal(str)
    grid_done = Signal(object)  # dict: request, rgb (the tiled composite), labels, bar, shown
    grid_failed = Signal(str)

    @Slot(object)
    def render_grid(self, req: dict):
        try:
            items = grid_items(req)
            rgb, labels, bar = grid_tiles(req, items)
            self.grid_done.emit({"request": req, "rgb": rgb, "labels": labels, "bar": bar,
                                 "shown": sum(1 for it in items if it is not None)})
        except Exception as e:
            self.grid_failed.emit(f"{type(e).__name__}: {e}")

    @Slot(object)
    def render(self, req: dict):
        try:
            out: G.OutputSet = req["output"]
            pts_side = pts_factor = None
            if req.get("pts"):                      # the point-evaluated hi-res slice in place of the grid's
                plane, pts_side, pts_factor = pts_plane(out, req["pts"], req["field"], req["component"])
                plane = _smoothed(plane, req.get("smooth", 0))
            else:
                plane = _smoothed(out.slice(req["field"], req["axis"], req["index"], req["component"]),
                                  req.get("smooth", 0))
            planes = None
            if req.get("compare") is not None:      # B read and smoothed the same way; the map is A - B or log10 A/B
                b = _smoothed(req["compare"].slice(req["field"], req["axis"], req["index"], req["component"]),
                              req.get("smooth", 0))
                planes = (plane, b)
                plane = compare_planes(plane, b, req["compare_mode"])
            # image: horizontal = the first remaining axis, vertical = the second, increasing upward
            img = np.ascontiguousarray(plane.T[::-1])
            rgb, vrange = G.colorize(img, req["cmap"], log=req["log"], clip=(req["clip"], 100 - req["clip"]),
                                     symmetric=req["symmetric"], vrange=req.get("vrange"))
            if planes is not None:
                rgb[~np.isfinite(img)] = NAN_GREY   # colorize paints NaN as vmin: on RdBu_r that would read 'A >> B'
            arrows = out.arrow_field(req["field"], req["axis"], req["index"], smooth=req.get("smooth", 0)) \
                if req.get("arrows") and not req.get("pts") else None
            self.done.emit({"rgb": np.ascontiguousarray(rgb), "plane": plane, "vrange": vrange, "request": req,
                            "arrows": arrows, "planes": planes, "pts_side": pts_side, "pts_factor": pts_factor})
        except Exception as e:                  # a partial file, a vanished disk
            self.failed.emit(f"{type(e).__name__}: {e}")


class ServerWorker(QObject):
    """A dtfelib.Estimator living on its own thread: start (the tessellation build, possibly long),
    query (milliseconds for a few points), stop. Cancel kills a build in progress."""
    started = Signal(str, object)       # (description, the server's region (3, 2) in Mpc)
    answered = Signal(object, object)   # (points, PointFields)
    zoomed = Signal(object, object)     # (zoom request, PointFields on its plane of points)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self):
        super().__init__()
        self.est = None
        self.proc = None
        self.n_partitions = 1           # the running server's split (an exact zoom reuses its cache)

    @Slot(object)
    def start(self, kw: dict):
        self.stop()
        try:
            from dtfelib import Estimator
            kw = dict(kw)
            kw["_on_spawn"] = lambda p: setattr(self, "proc", p)
            self.est = Estimator(**kw)
            self.n_partitions = int(getattr(self.est, "n_partitions", 1) or 1)
            parts = f", {self.est.n_partitions} partitions, {self.est.resident} held in memory" \
                if self.est.n_partitions > 1 else ""
            self.started.emit(f"{self.est.n_vertices:,} vertices{parts}", self.est.region.copy())
        except Exception as e:
            self.est = None
            self.failed.emit(str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__)
        finally:
            self.proc = None

    @Slot(object)
    def query(self, pts):
        if self.est is None:
            self.failed.emit("the query server is not running")
            return
        try:
            self.answered.emit(pts, self.est(pts))
        except Exception as e:
            self._request_failed("query", e)

    @Slot(object)
    def zoom(self, req: dict):
        """Every field on the tensor-product plane req['coords'] (three 1-D coordinate arrays, one of
        length 1; a 2D server: two): the zoomed region evaluated point by point, grid-free."""
        if self.est is None:
            self.failed.emit("the query server is not running")
            return
        try:
            self.zoomed.emit(req, self.est.grid(*req["coords"]))
        except Exception as e:
            self._request_failed("zoom", e)

    def _request_failed(self, what: str, e: Exception):
        """A request raised. When the server process itself has exited (killed for memory, crashed), say
        so and stop: the GUI's button then offers a new start instead of 'Stop' on a dead server (every
        further click failed the same way before 2026-10-05)."""
        proc = getattr(self.est, "_proc", None)
        code = proc.poll() if proc is not None else None
        if code is not None:
            self.failed.emit(f"{what} failed: the query server has exited (code {code}); start it again")
            self.stop()
        else:
            self.failed.emit(f"{what} failed: {e}")

    @Slot()
    def stop(self):
        self.n_partitions = 1
        if self.est is not None:
            try:
                self.est.close()
            except Exception:
                pass
            self.est = None
            self.stopped.emit()

    def cancel(self):
        """From the GUI thread: end a build in progress (its start() then reports the failure)."""
        p = self.proc
        if p is not None and p.poll() is None:
            p.kill()

    def kill(self):
        """From the GUI thread, at shutdown only: end the RUNNING server's process, so a request the
        worker thread is blocked in (a cold composite zoom, tens of seconds) returns at once with EOF
        and the thread can be joined (before 2026-10-05 the thread was dropped while running)."""
        self.cancel()
        p = getattr(self.est, "_proc", None)
        if p is not None and p.poll() is None:
            p.kill()


# ==================================================================== canvas

# the finest virtual full grid an exact zoom of a phase-space run asks for along an axis (--ps-window
# allocates the window only; the grid just sets the cell size)
WINDOW_GRID_MAX = 1 << 20

# The time an exact zoom of a phase-space run will take, from measured rates (2026-10-02, TNG100-3-Dark,
# 94M particles, from the Samsung T7, M1 Max): reading snapshot + initial conditions 16 s for 6.0 GB
# (skipped when the snapshot's globals are recorded and every needed partition is cached); per BUILT
# partition 3 s selecting its particles (all 94M scanned; a cached one skips it); 8.4 s loading a
# cached single-stream 6.0M-vertex tessellation (a 0.88 GB file) -- a chunked file loads 1.68x faster
# (measured 1.53 -> 0.91 s on the synthetic 2.1M-particle partitions); ~1.6 s for its exact deposit (3 s
# before the window-local vertex degrees halved it on the synthetic box: 3.58 -> 1.78 s); building one
# from scratch 26 s per 7M vertices (the production run) plus writing it to the cache. The binary runs
# several partitions at a time (its auto-tuner, from the free memory): the estimate assumes EST_CONCURRENCY.
EST_READ_BPS = 3.8e8
EST_SELECT_S_PER_PARTICLE = 3.2e-8
EST_LOAD_BPS = 1.05e8
EST_CHUNKED_LOAD_SPEEDUP = 1.68
EST_DEPOSIT_S_PER_VERTEX = 2.7e-7
EST_BUILD_S_PER_VERTEX = 3.7e-6
EST_SAVE_BPS = 1.5e8
EST_VERTICES_PER_BYTE = 6.0e6 / 0.884e9
EST_CONCURRENCY = 3
EXACT_CONFIRM_SECONDS = 60      # ask first when an exact zoom will take longer than this

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# zooms kept in memory (the Back button and repeated drags reuse them without a new evaluation)
ZOOM_MEMO_BYTES = 1.5e9
_REQ_LINE = re.compile(r"\[request (\d+)/(\d+)\] partition \d+(?:: (in memory|loading from the cache|building its cell index)| done)")
_REQ_DONE = re.compile(r"\[request done\] (\d+) points, (\d+) of (\d+) partitions, (\d+) loaded from the cache in ([\d.]+) s")


def _fields_bytes(f) -> int:
    """The memory a zoom's fields hold (its numpy arrays)."""
    total = 0
    for v in getattr(f, "__dict__", {}).values():
        if isinstance(v, np.ndarray):
            total += v.nbytes
        elif isinstance(v, (list, tuple)):
            total += sum(a.nbytes for a in v if isinstance(a, np.ndarray))
    return total


def _fmt_duration(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 90:
        return f"{max(1, int(round(sec / 5.0)) * 5)} s"
    if sec < 5400:
        return f"{int(round(sec / 60.0))} min"
    return f"{sec / 3600.0:.1f} h"


class ExactZoomWorker(QObject):
    """The exact zoom: runs the binary itself on the dragged region (over the whole snapshot, seconds
    to minutes) and loads the thin grid it writes. Lives on its own thread so the server keeps
    answering; reports its progress from the binary's own lines and can be cancelled (cancel(),
    from the GUI thread, kills the run: failed then reports CANCELLED)."""
    done = Signal(object, object)       # (zoom request, the fields on the region's grid)
    failed = Signal(str)
    progress = Signal(str)              # what the run is doing now, for the status line
    CANCELLED = "cancelled"

    def __init__(self):
        super().__init__()
        self.proc = None
        self.cancel_requested = False   # set from the GUI thread; cleared by the GUI before each run

    @staticmethod
    def phase(line: str, skipped: str) -> tuple[str | None, str]:
        """(the status for one line of the binary's output or None, the skipped-partitions note)."""
        if "never reach the window" in line:
            m = re.search(r"(\d+) of (\d+) partitions never reach", line)
            if m:
                skipped = f", {m.group(1)} of {m.group(2)} partitions skipped"
            return "finding the partitions that reach the region" + skipped, skipped
        if "[partitions" in line:
            m = re.search(r"\[partitions (\d+)/(\d+)", line)
            eta = re.search(r"ETA ~(.+)$", line)
            if m:
                return (f"partition {m.group(1)} of {m.group(2)}{skipped}"
                        + (f", about {eta.group(1).strip()} left" if eta else "")), skipped
        if "the snapshot is not read" in line:
            return "every partition it needs is cached: the snapshot is not read" + skipped, skipped
        if "reading the particles after all" in line:
            return "reading the snapshot (a partition it needs is not cached)", skipped
        if line.startswith("Particles deferred"):
            return "checking the tessellation cache", skipped
        if line.startswith("Reading") or "READING INPUT" in line:
            return "reading the snapshot", skipped
        if line.startswith("Writing"):
            return "writing the region", skipped
        return None, skipped

    @Slot(object)
    def run(self, req: dict):
        import shutil
        import subprocess
        tail: list[str] = []
        t0 = time.time()
        try:
            if req.get("cache_dir"):
                Path(req["cache_dir"]).mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env.pop("CLICOLOR_FORCE", None)
            self.proc = subprocess.Popen(req["cmd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, errors="replace", bufsize=1, env=env)
            if self.cancel_requested:
                self.proc.kill()
            status, skipped = "starting", ""
            for raw in self.proc.stdout:
                line = _ANSI.sub("", raw).strip()
                if not line:
                    continue
                tail = (tail + [line])[-6:]
                new, skipped = self.phase(line, skipped)
                if new and new != status:
                    status = new
                    self.progress.emit(f"{status} ({int(time.time() - t0)} s)")
            rc = self.proc.wait()
            if self.cancel_requested:
                raise RuntimeError(self.CANCELLED)
            if rc != 0:
                raise RuntimeError(f"{Path(req['cmd'][0]).name} failed ({rc}): " + " | ".join(tail[-3:]))
            self.done.emit(req, load_exact_planes(req["root"], req["grid"],
                                                  redshift=getattr(req.get("output"), "redshift", None)))
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            self.proc = None
            shutil.rmtree(Path(req["root"]).parent, ignore_errors=True)

    def cancel(self):
        """From the GUI thread: stop the run in progress (or the one about to start)."""
        self.cancel_requested = True
        p = self.proc
        if p is not None and p.poll() is None:
            p.kill()


def load_exact_planes(root, grid, redshift=None):
    """The grids an exact zoom wrote (shape 'grid', one cell thick along the slice axis; a 2D run's
    'grid' has two entries), as an object with the attributes of a dtfelib point evaluation
    (density, streams, caustic, velocity, dispersion, velocity_gradient; None when the run did not
    write them), so the zoom's renderer reads both the same way. The binary writes velocities in the
    snapshot's u-units; with the output's redshift the velocity-like planes get the same (1/(1+z))^exp
    factor the slice map (OutputSet.slice) and the point zoom (Estimator, peculiar) apply -- without
    it an exact zoom at z=2 showed velocities 1.73x the slice map's (found 2026-10-05)."""
    import types
    root = Path(root)
    shape = tuple(int(v) for v in grid)
    dim = len(shape)
    planes = {}
    for name, (sfx, _ncomp, _ps) in G.FIELDS.items():
        ncomp = G.ncomp_of(name, dim)
        for prefix in ("a_", ""):
            p = root.parent / f"{root.name}.{prefix}{sfx}"
            if not p.is_file() or name in planes:
                continue
            n = int(np.prod(shape)) * ncomp
            size = p.stat().st_size
            if n == 0 or size % n or size // n not in (4, 8):
                continue
            dt = np.float32 if size // n == 4 else np.float64
            arr = np.fromfile(p, dtype=dt)  # the file's precision: the renderer up-casts the ONE plotted component
                                            # (float64 of a 9-component gradient at 4096^2 was 2.4 GB)
            planes[name] = arr.reshape(shape + ((ncomp,) if ncomp > 1 else ()))
    if redshift is not None:
        for name in list(planes):
            exp = G._VELOCITY_SCALE_EXP.get(name)
            if exp is not None:
                planes[name] = planes[name] * (1.0 / (1.0 + float(redshift))) ** exp
    f = types.SimpleNamespace(density=planes.get("density"), streams=planes.get("streams"),
                              caustic=None, velocity=planes.get("velocity"), dispersion=planes.get("dispersion_tensor"),
                              sigma=planes.get("dispersion"), velocity_gradient=None, offsets=None,
                              scalar=planes.get("scalar"))
    if "caustic" in planes:
        f.caustic = np.rint(planes["caustic"]).astype(np.int64)
    if "gradient" in planes:   # the grid's [j*D+i] = dv_j/dx_i -> the point evaluation's [d, j] = dv_j/dx_d
        f.velocity_gradient = planes["gradient"].reshape(shape + (dim, dim)).swapaxes(-1, -2)
    if f.density is None:
        raise RuntimeError("the exact zoom wrote no density grid")
    return f

# ==================================================================== the figure grid (pure parts)
# grid_items / grid_tiles run in the slice worker's thread for the screen and on the GUI thread for a save:
# numpy on OutputSets and the request dict only, no widgets.

def grid_items(req: dict) -> list:
    """Per panel of the request: its plane on the slice and the colour range it gets (None for an empty
    panel). The range is colorize()'s (log10 where the panel is log, centred on 0 for a diverging field);
    with 'shared' or 'merged', panels of one (field, component) take the union of their ranges."""
    axis, rel, clip, smooth = req["axis"], req["rel"], req["clip"], req["smooth"]
    items = []
    for p in req["panels"]:
        if p is None:
            items.append(None)
            continue
        o, name, comp = p["output"], p["field"], p["component"]
        idx = 0 if o.dim == 2 else int(round(rel * (o.n - 1)))
        plane = o.slice(name, axis, idx, comp)
        if smooth > 0:
            try:
                from scipy.ndimage import gaussian_filter
                plane = gaussian_filter(plane, smooth, mode="wrap")
            except ImportError:
                pass
        log, cmap = p["log"], p["cmap"]
        symmetric = name in G.DIVERGING and not log
        img = np.ascontiguousarray(plane.T[::-1])          # the map's orientation: horizontal = first rest axis
        lo, hi = G.value_range(img, log, (clip, 100 - clip), symmetric)
        finite = bool(np.isfinite(plane).any() and (not log or bool((plane > 0).any())))
        rest = [a for a in range(3) if a != axis] if o.dim == 3 else [0, 1]
        ext = [o.lo[rest[0]], o.lo[rest[0]] + o.length[rest[0]], o.lo[rest[1]], o.lo[rest[1]] + o.length[rest[1]]]
        items.append({"output": o, "field": name, "component": comp, "plane": plane, "img": img, "lo": lo, "hi": hi,
                      "log": log, "cmap": cmap, "symmetric": symmetric, "ext": ext, "rest": rest, "idx": idx,
                      "finite": finite})
    if req["shared"] or req["merged"]:
        groups: dict = {}
        for it in items:
            if it is not None:
                groups.setdefault((it["field"], it["component"]), []).append(it)
        for its in groups.values():
            full = [i for i in its if i["finite"]] or its        # an all-empty panel takes the others' range
            lo, hi = min(i["lo"] for i in full), max(i["hi"] for i in full)
            if its[0]["symmetric"]:
                m = max(abs(lo), abs(hi))
                lo, hi = -m, m
            for i in its:
                i["lo"], i["hi"] = lo, hi
    return items


def panel_names(it: dict) -> tuple[str, str, str, str]:
    """(field label, component text, who, when) of a panel."""
    o, name, comp = it["output"], it["field"], it["component"]
    label = G.FIELD_LABELS.get(name, name)
    ctext = "" if comp in ("value", "") else f" [{comp}]"
    who = o.sim or o.dir.name
    when = f"z = {o.redshift:.2f}" if o.redshift is not None else (o.title.split(" · ")[1] if " · " in o.title else "")
    return label, ctext, who, when


def type_tag(o) -> str:
    """What tells two runs apart in a panel's label: the estimator with a non-default prefix
    (grids.type_label: 'DTFE', 'PS-DTFE', 'PS-DTFE · ps_mw' -- the Type column's words, not the raw
    prefix 'output' that says nothing to a reader of the saved figure); the prefix for a bare object."""
    if hasattr(o, "files") and getattr(o, "prefix", ""):
        return G.type_label(o)
    return getattr(o, "prefix", "") or ""


def grid_differences(items: list) -> dict:
    """What differs between the panels: the sets of outputs, simulations, quantities and (who, when);
    'by_prefix' = the run has to be named: outputs of one snapshot (the prefix tells them apart), or
    two estimators in the grid (a DTFE panel next to a PS-DTFE one at another redshift said nothing)."""
    live = [it for it in items if it is not None]
    outs = {str(it["output"].root) for it in live}
    moments = {panel_names(it)[2:] for it in live}
    types = {type_tag(it["output"]) for it in live}
    return {"outs": outs, "sims": {panel_names(it)[2] for it in live},
            "quantities": {(it["field"], it["component"]) for it in live},
            "by_prefix": len(outs) > len(moments) or len(types) > 1}


def panel_label(it: dict, diff: dict) -> str:
    """The short text inside a panel: only what differs between the panels (the redshift, the simulation
    when several, the field when several, the run -- its estimator and prefix -- when two runs of one
    snapshot or two estimators are shown)."""
    label, ctext, who, when = panel_names(it)
    o = it["output"]
    tag = f" · {type_tag(o)}" if diff["by_prefix"] and type_tag(o) else ""
    parts = []
    if len(diff["outs"]) > 1:
        parts.append((f"{who} · " if len(diff["sims"]) > 1 else "") + (when or who) + tag)
    if len(diff["quantities"]) > 1:
        parts.append(label + ctext)
    return "\n".join(parts) or (when or who)


def grid_tiles(req: dict, items: list, max_tile: int = 1024) -> tuple[np.ndarray, list, tuple | None]:
    """The screen's picture: every panel coloured straight to pixels (colorize, the single map's way) and
    tiled, thin white gaps unless merged, resampled to one tile size (nearest). Returns (rgb, labels,
    bar): labels = [(x, y, text)] in image pixels, bar = (cmap, vrange, label, log) when every panel shows one
    quantity on one range (shown below the picture), else None."""
    rows, cols, clip = req["rows"], req["cols"], req["clip"]
    sizes = [it["img"].shape[0] for it in items if it is not None]
    T = min(max(sizes), max_tile)
    gap = 0 if req["merged"] else 6
    out = np.full((rows * T + gap * (rows - 1), cols * T + gap * (cols - 1), 3), 255, dtype=np.uint8)
    diff = grid_differences(items)
    labels = []
    for k, it in enumerate(items):
        r, c = divmod(k, cols)
        y0, x0 = r * (T + gap), c * (T + gap)
        if it is None:
            out[y0:y0 + T, x0:x0 + T] = 238
            continue
        img = it["img"]
        if img.shape[0] != T or img.shape[1] != T:    # resample first, colour T^2 pixels, not the whole plane
            ri = (np.arange(T) * img.shape[0]) // T
            ci = (np.arange(T) * img.shape[1]) // T
            sub = img[ri][:, ci]
            if it["log"]:                             # the full plane's floor for the empty cells (colorize's
                pos = img[img > 0]                    # own would be the sample's): the same pixels as before
                sub = np.where(sub > 0, sub, pos.min() if pos.size else 1.0)
            img = sub
        rgb, _ = G.colorize(img, it["cmap"], log=it["log"], clip=(clip, 100 - clip),
                            symmetric=it["symmetric"], vrange=(it["lo"], it["hi"]))
        out[y0:y0 + T, x0:x0 + T] = rgb
        labels.append((x0 + 8, y0 + 8, panel_label(it, diff)))
    bar = None
    if len(diff["quantities"]) == 1 and (req["shared"] or req["merged"] or len(diff["outs"]) == 1):
        it0 = next(it for it in items if it is not None)
        label, ctext, _w, _t = panel_names(it0)
        bar = (it0["cmap"], (it0["lo"], it0["hi"]), ("log₁₀ " if it0["log"] else "") + label + ctext, it0["log"])
    return out, labels, bar


def grid_figure(req: dict, items: list, dpi: float = 200.0):
    """The full matplotlib figure (what Save figure… writes): axes in Mpc, per-panel titles and colour bars,
    or -- merged -- touching panels with inside labels and the bars in one column at the right, one per
    quantity. Returns (Figure, panel count)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    rows, cols, merged, axis, rel = req["rows"], req["cols"], req["merged"], req["axis"], req["rel"]
    fs = 1.0
    if merged:                              # cells shaped like the data, no gap at wspace = hspace = 0
        e0 = next(i for i in items if i)["ext"]
        aspect = (e0[3] - e0[2]) / (e0[1] - e0[0]) if e0[1] > e0[0] and e0[3] > e0[2] else 1.0
        pw, ph = 4.2, 4.2 * float(np.clip(aspect, 0.25, 4.0))
        ml, mr, mb, mt = 0.75, 1.3, 0.6, 0.55      # margins in inches: left, the bar column, bottom, top
        size_in = (pw * cols + ml + mr, ph * rows + mb + mt)
    else:
        size_in = (4.8 * cols, 4.2 * rows)
    fig = Figure(figsize=size_in, dpi=dpi)
    FigureCanvasAgg(fig)
    W, H = size_in
    if merged:
        left, right_edge, bottom, top = ml / W, 1.0 - mr / W, mb / H, 1.0 - mt / H
        gs = fig.add_gridspec(rows, cols, left=left, right=right_edge, bottom=bottom, top=top, wspace=0.0, hspace=0.0)
        axes = gs.subplots(sharex=True, sharey=True, squeeze=False)
    else:
        axes = fig.subplots(rows, cols, squeeze=False)
    diff = grid_differences(items)
    shown = 0
    bars: dict = {}                         # merged: (field, component) -> the first panel's image, in order
    for k, it in enumerate(items):
        ax = axes[k // cols][k % cols]
        if it is None:
            ax.set_axis_off()
            continue
        shown += 1
        o = it["output"]
        data = it["plane"].T
        if it["log"]:
            data = np.log10(np.where(data > 0, data, np.nan))
        hi = it["hi"] if it["hi"] > it["lo"] else it["lo"] + 1.0
        im = ax.imshow(data, origin="lower", extent=it["ext"], cmap=it["cmap"], vmin=it["lo"], vmax=hi,
                       aspect="equal", interpolation="nearest")
        label, ctext, who, when = panel_names(it)
        unit = "Mpc" if o.box is not None else "cells"
        ax.set_xlabel(f"{AXES[it['rest'][0]]} [{unit}]", fontsize=8 * fs)
        ax.set_ylabel(f"{AXES[it['rest'][1]]} [{unit}]", fontsize=8 * fs)
        ax.tick_params(labelsize=7 * fs)
        tag = f" · {type_tag(o)}" if diff["by_prefix"] and type_tag(o) else ""
        if merged:
            bars.setdefault((it["field"], it["component"]), (im, it))
            ax.text(0.03, 0.97, panel_label(it, diff), transform=ax.transAxes, va="top", ha="left",
                    fontsize=8 * fs, color="white", bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2))
        else:
            ax.set_title(f"{label}{ctext}\n{who}" + (f" · {when}" if when else "") + tag, fontsize=9 * fs)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label(("log₁₀ " if it["log"] else "") + label + ctext, fontsize=8 * fs)
            cb.ax.tick_params(labelsize=7 * fs)
    pdim = next((it["output"].dim for it in items if it), 3)
    where = f" ({req['position']})" if req.get("position") else ""
    slice_text = f"slice normal to {AXES[axis]} at {100 * rel:.0f}% of the box{where}" if pdim == 3 else ""
    if merged:
        # tick and axis labels on the outer edges only -- of the panels DRAWN: an empty panel in the bottom
        # row (three outputs in a 2 x 2) must not leave its column without an x axis
        drawn = [[axes[r][c].axison for c in range(cols)] for r in range(rows)]
        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                if not ax.axison:
                    continue
                lowest = not any(drawn[rr][c] for rr in range(r + 1, rows))
                leftmost = not any(drawn[r][cc] for cc in range(c))
                ax.tick_params(labelbottom=lowest, labelleft=leftmost)
                if not lowest:
                    ax.set_xlabel("")
                if not leftmost:
                    ax.set_ylabel("")
        n = len(bars)
        gap = (0.25 / H) if n > 1 else 0.0
        span = top - bottom
        h = (span - gap * (n - 1)) / max(n, 1)
        cb_x, cb_w = right_edge + 0.12 / W, 0.16 / W
        for i, (key, (im, it)) in enumerate(bars.items()):
            cax = fig.add_axes([cb_x, top - (i + 1) * h - i * gap, cb_w, h])
            cb = fig.colorbar(im, cax=cax)
            label, ctext, _w, _t = panel_names(it)
            text = ("log₁₀ " if it["log"] else "") + label + ("\n" + ctext.strip() if ctext else "")
            size = 8 * fs * float(np.clip(h * H / 1.6, 0.6, 1.0))   # a short bar (many of them): smaller text
            cb.set_label(text, fontsize=size)
            cb.ax.tick_params(labelsize=min(7 * fs, size))
        head = []
        if len(diff["quantities"]) == 1:
            label, ctext, _w, _t = panel_names(next(i for i in items if i))
            head.append(label + ctext)
        if len(diff["sims"]) == 1:
            head.append(next(iter(diff["sims"])))
        if slice_text:
            head.append(slice_text)
        fig.suptitle("   ·   ".join(head), fontsize=9 * fs, y=1.0 - 0.17 / H)
    else:
        if slice_text:
            fig.suptitle(slice_text, fontsize=9 * fs, y=0.995)
        fig.tight_layout()
    return fig, shown


class ImageCanvas(QLabel):
    """Shows an RGB array scaled to fit (aspect kept); reports the image pixel under the mouse."""
    hovered = Signal(int, int)       # image row, column (-1, -1 when outside)
    clicked = Signal(int, int)       # a press and release without a drag
    dragged = Signal(int, int, int, int)   # a rectangle dragged out: image rows/cols of two corners

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(240, 240)
        self._img: QImage | None = None
        self._buf = None
        self._press = None                   # where the left button went down (view coordinates)
        self._drag: QRectF | None = None     # the rectangle being dragged out (view coordinates)
        self.marker: tuple[int, int] | None = None
        self.arrows: dict | None = None     # grids.OutputSet.arrow_field(), drawn over the map
        self.labels: list | None = None     # [(x, y, text)] in image pixels: the figure grid's panel labels
        # {"x": (lo, hi, name), "y": (lo, hi, name), "unit": "Mpc"|"cells"}: the axes drawn around the image
        # (x increasing to the right, y UPWARD) and the scale bar; None for the figure grid's tiles
        self.axes: dict | None = None
        self.setText("Choose an output on the left.")
        self.setStyleSheet("color: gray")

    def set_rgb(self, rgb: np.ndarray | None):
        if rgb is None:
            self._img, self._buf, self.axes = None, None, None
            self.setPixmap(QPixmap())
            return
        self._buf = rgb                    # keep the bytes alive for the QImage
        h, w, _ = rgb.shape
        self._img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._redraw()

    def _axes_font(self):
        f = self.font()
        f.setPointSizeF(9.0)
        return f

    def _margins(self) -> tuple[float, float, float, float]:
        """(left, top, right, bottom) kept free around the image for the axes: the tick labels (as wide as
        the widest one) and the axis names."""
        a = self.axes
        if not a:
            return 0.0, 0.0, 0.0, 0.0
        fm = QFontMetrics(self._axes_font())
        whole = a["unit"] == "cells"
        yt, xt = G.nice_ticks(*a["y"][:2], integer=whole), G.nice_ticks(*a["x"][:2], integer=whole)
        yw = max((fm.horizontalAdvance(G.tick_text(v, yt)) for v in yt), default=0)
        xw = max((fm.horizontalAdvance(G.tick_text(v, xt)) for v in xt), default=0)
        h = fm.height()
        return yw + h + 12.0, h / 2 + 2.0, xw / 2 + 4.0, 2 * h + 10.0

    def _target(self) -> QRectF | None:
        """Where the image is drawn: scaled to fit (aspect kept) inside the margins the axes need."""
        if self._img is None:
            return None
        w, h = self._img.width(), self._img.height()
        left, top, right, bottom = self._margins()
        W, H = self.width() - left - right, self.height() - top - bottom
        if W < 40 or H < 40:                        # too small for axes: the whole widget
            left = top = 0.0
            W, H = self.width(), self.height()
        s = min(W / w, H / h)
        return QRectF(left + (W - w * s) / 2, top + (H - h * s) / 2, w * s, h * s)

    def _redraw(self):
        t = self._target()
        if t is None:
            return
        dpr = self.devicePixelRatioF()          # the axes' text at the screen's resolution (Retina: 2x)
        pm = QPixmap(self.size() * dpr)
        pm.setDevicePixelRatio(dpr)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        # smooth when shrinking a big slice; crisp cells when a small grid is blown up
        p.setRenderHint(QPainter.SmoothPixmapTransform, self._img.width() > t.width())
        p.drawImage(t, self._img)
        if self.arrows is not None:
            self._draw_arrows(p, t)
        if self.labels:
            self._draw_labels(p, t)
        if self.axes:
            self._draw_axes(p, t)
        if self.marker is not None:
            r, c = self.marker
            s = t.width() / self._img.width()
            x, y = t.left() + (c + 0.5) * s, t.top() + (r + 0.5) * s
            p.setPen(QPen(QColor("white"), 2))
            p.drawEllipse(QPointF(x, y), 7, 7)
            p.setPen(QPen(QColor("black"), 1))
            p.drawEllipse(QPointF(x, y), 9, 9)
        if self._drag is not None:
            p.setPen(QPen(QColor(0, 0, 0, 180), 3))
            p.drawRect(self._drag)
            p.setPen(QPen(QColor("white"), 1.2))
            p.drawRect(self._drag)
        p.end()
        self.setPixmap(pm)

    def _draw_labels(self, p: QPainter, t: QRectF):
        """White text on a translucent box at image positions (the grid's 'z = 1.50' per panel)."""
        s = t.width() / self._img.width()
        p.setRenderHint(QPainter.Antialiasing, True)
        f = p.font()
        f.setPointSizeF(10.0)
        p.setFont(f)
        fm = QFontMetrics(f)
        for x, y, text in self.labels:
            lines = text.split("\n")
            w_ = max(fm.horizontalAdvance(ln) for ln in lines) + 8
            h_ = fm.height() * len(lines) + 4
            box = QRectF(t.left() + x * s, t.top() + y * s, w_, h_)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 135))
            p.drawRoundedRect(box, 3, 3)
            p.setPen(QPen(QColor("white")))
            p.drawText(box.adjusted(4, 2, 0, 0), Qt.AlignLeft | Qt.AlignTop, text)

    def _draw_axes(self, p: QPainter, t: QRectF):
        """The axes around the image (ticks at round values, labels that would collide dropped, the axis
        names) and a scale bar of a round length in its lower-left corner (survey item 12)."""
        (x0, x1, xn), (y0, y1, yn), unit = self.axes["x"], self.axes["y"], self.axes["unit"]
        if not (x1 > x0 and y1 > y0):
            return
        f = self._axes_font()
        p.setFont(f)
        fm = QFontMetrics(f)
        h = fm.height()
        p.setRenderHint(QPainter.Antialiasing, False)
        p.setPen(QPen(self.palette().windowText().color(), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(t)
        whole = unit == "cells"                 # whole cells: no tick between two
        xt, last = G.nice_ticks(x0, x1, integer=whole), -1e9
        for v in xt:
            x = t.left() + (v - x0) / (x1 - x0) * t.width()
            p.drawLine(QPointF(x, t.bottom()), QPointF(x, t.bottom() + 4))
            text = G.tick_text(v, xt)
            w_ = fm.horizontalAdvance(text)
            if x - w_ / 2 >= last + 6:
                p.drawText(QRectF(x - w_ / 2 - 1, t.bottom() + 5, w_ + 2, h), Qt.AlignCenter, text)
                last = x + w_ / 2
        yt, last = G.nice_ticks(y0, y1, integer=whole), 1e9
        yw = max((fm.horizontalAdvance(G.tick_text(v, yt)) for v in yt), default=0)
        for v in yt:
            y = t.bottom() - (v - y0) / (y1 - y0) * t.height()
            p.drawLine(QPointF(t.left() - 4, y), QPointF(t.left(), y))
            if y + h / 2 <= last - 2:
                p.drawText(QRectF(0, y - h / 2, t.left() - 6, h), Qt.AlignRight | Qt.AlignVCenter, G.tick_text(v, yt))
                last = y - h / 2
        p.drawText(QRectF(t.left(), t.bottom() + 6 + h, t.width(), h), Qt.AlignHCenter | Qt.AlignTop, f"{xn} [{unit}]")
        p.save()
        p.translate(max(2.0, t.left() - 10 - yw - h), t.center().y())      # just left of the tick labels
        p.rotate(-90)
        p.drawText(QRectF(-t.height() / 2, 0, t.height(), h), Qt.AlignCenter, f"{yn} [{unit}]")
        p.restore()
        length = G.nice_length(x1 - x0)
        px = length / (x1 - x0) * t.width()
        if t.width() >= 120 and px >= 8:
            label = f"{length:g} {unit}"
            bx, by = t.left() + 10, t.bottom() - 10
            wide = max(px, fm.horizontalAdvance(label))
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 135))
            p.drawRoundedRect(QRectF(bx - 4, by - h - 7, wide + 8, h + 11), 3, 3)
            p.setPen(QPen(QColor("white"), 3, Qt.SolidLine, Qt.FlatCap))
            p.drawLine(QPointF(bx, by), QPointF(bx + px, by))
            p.setPen(QPen(QColor("white")))
            p.drawText(QRectF(bx, by - h - 4, wide, h), Qt.AlignLeft | Qt.AlignBottom, label)

    def _draw_arrows(self, p: QPainter, t: QRectF):
        """The vector field: one arrow per block, the longest 0.9 block long, in white over a
        dark outline so it reads on any colour map."""
        a = self.arrows
        s = t.width() / self._img.width()
        length = 0.9 * a["spacing"] * s
        lines = []
        for x, y, dx, dy in zip(a["x"], a["y"], a["dx"], a["dy"]):
            ex, ey = dx * length, dy * length
            n = (ex * ex + ey * ey) ** 0.5
            if n < 0.5:
                continue
            x0, y0 = t.left() + x * s - ex / 2, t.top() + y * s - ey / 2     # centred on the block
            x1, y1 = x0 + ex, y0 + ey
            head = min(0.35 * n, 6.0)
            ux, uy = ex / n, ey / n
            lines += [QLineF(x0, y0, x1, y1),
                      QLineF(x1, y1, x1 - head * (0.87 * ux - 0.5 * uy), y1 - head * (0.87 * uy + 0.5 * ux)),
                      QLineF(x1, y1, x1 - head * (0.87 * ux + 0.5 * uy), y1 - head * (0.87 * uy - 0.5 * ux))]
        if not lines:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(0, 0, 0, 170), 2.6, Qt.SolidLine, Qt.RoundCap))
        p.drawLines(lines)
        p.setPen(QPen(QColor("white"), 1.2, Qt.SolidLine, Qt.RoundCap))
        p.drawLines(lines)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._redraw()

    def _pixel(self, pos) -> tuple[int, int]:
        t = self._target()
        if t is None or not t.contains(pos):
            return -1, -1
        s = t.width() / self._img.width()
        return int((pos.y() - t.top()) / s), int((pos.x() - t.left()) / s)

    def _pixel_clamped(self, pos) -> tuple[int, int]:
        """Like _pixel, but a position outside the image maps to its nearest pixel."""
        t = self._target()
        s = t.width() / self._img.width()
        r = int((pos.y() - t.top()) / s)
        c = int((pos.x() - t.left()) / s)
        return max(0, min(self._img.height() - 1, r)), max(0, min(self._img.width() - 1, c))

    def mouseMoveEvent(self, e):
        if self._press is not None:
            t = self._target()
            if t is not None and (e.position() - self._press).manhattanLength() > 4:
                self._drag = QRectF(self._press, e.position()).normalized().intersected(t)
                self._redraw()
        self.hovered.emit(*self._pixel(e.position()))

    def leaveEvent(self, e):
        self.hovered.emit(-1, -1)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._img is not None:
            self._press = e.position()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        press, drag, self._press, self._drag = self._press, self._drag, None, None
        if press is None or self._img is None:
            return
        if drag is not None and drag.width() > 2 and drag.height() > 2:
            self._redraw()
            r0, c0 = self._pixel_clamped(drag.topLeft())
            r1, c1 = self._pixel_clamped(drag.bottomRight())
            self.dragged.emit(r0, c0, r1, c1)
        else:
            r, c = self._pixel(e.position())
            if r >= 0:
                self.clicked.emit(r, c)


class ColorBar(QWidget):
    """The map's colour scale (survey item 13): the gradient; ticks at round values -- whole decades,
    written 10^k, on a log map -- with the two ends as the lowest-priority labels (they say where the bar
    stops); the quantity on a row of its own, elided."""
    BAR = 14                                # the gradient's height; ticks below it, then two text rows

    def __init__(self):
        super().__init__()
        self.setFixedHeight(self.BAR + 4 + 2 * QFontMetrics(self.font()).height() + 6)
        self.cmap, self.vrange, self.label, self.log = "viridis", (0.0, 1.0), "", False

    def set(self, cmap: str, vrange, label: str, log: bool = False):
        """'label' is the full text ('log₁₀ ' included: Save image labels its log10 data with it);
        'log': the range is in log10 units, the ticks read 10^k."""
        self.cmap, self.vrange, self.label, self.log = cmap, vrange, label, bool(log)
        self.update()

    def shown_label(self) -> str:
        """The bar's own text: on a log map the ticks already read 10^k, so 'log₁₀ ' is dropped."""
        if self.log and self.label.startswith("log₁₀ "):
            return self.label[len("log₁₀ "):]
        return self.label

    def tick_labels(self, width: int | None = None) -> list[tuple[float, float, str]]:
        """(tick x, label left, label text) of every label drawn: the round ticks first, then the two ends
        where they do not collide (6 px apart at least)."""
        W = float(width if width is not None else self.width())
        fm = QFontMetrics(self.font())
        lo, hi = self.vrange
        span = (hi - lo) if hi > lo else 1.0
        placed, out = [], []

        def place(x, text, left=None):
            w_ = fm.horizontalAdvance(text)
            l_ = min(max(x - w_ / 2, 0.0), W - w_) if left is None else left
            if all(l_ + w_ + 6 <= a or l_ >= b + 6 for a, b in placed):
                placed.append((l_, l_ + w_))
                out.append((x, l_, text))
        for v, text in G.colorbar_ticks(lo, hi, self.log):
            place((v - lo) / span * W, text)
        end = (lambda v: f"{10.0 ** v:.3g}") if self.log else (lambda v: f"{v:.3g}")
        place(0.0, end(lo), left=0.0)
        place(W, end(hi), left=W - fm.horizontalAdvance(end(hi)))
        return out

    def paintEvent(self, _e):
        p = QPainter(self)
        W, B = self.width(), self.BAR
        lut = G.lut(self.cmap)
        g = QLinearGradient(0, 0, W, 0)
        for i in range(0, 256, 16):
            g.setColorAt(i / 255, QColor(*[int(v) for v in lut[i]]))
        g.setColorAt(1.0, QColor(*[int(v) for v in lut[255]]))
        p.fillRect(0, 0, W, B, g)
        p.setPen(self.palette().windowText().color())
        fm = QFontMetrics(self.font())
        h = fm.height()
        for x, left, text in self.tick_labels(W):
            if 0 < x < W:
                p.drawLine(QPointF(x, B), QPointF(x, B + 4))
            p.drawText(QRectF(left, B + 4, fm.horizontalAdvance(text) + 1, h), Qt.AlignLeft | Qt.AlignTop, text)
        p.drawText(QRectF(0, B + 6 + h, W, h), Qt.AlignHCenter | Qt.AlignTop,
                   fm.elidedText(self.shown_label(), Qt.ElideRight, W))
        p.end()


# ==================================================================== the two panes
class ExploreView(QWidget):
    """Right pane: the map, the readout and the query answer."""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        head = QHBoxLayout()
        self.title = QLabel("<b>Explore</b>")
        head.addWidget(self.title, 1)
        self.save_btn = QPushButton("Save image…")
        head.addWidget(self.save_btn)
        v.addLayout(head)
        self.canvas = ImageCanvas()
        v.addWidget(self.canvas, 1)
        self.colorbar = ColorBar()
        v.addWidget(self.colorbar)
        self.readout = QLabel(" ")
        self.readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.readout)
        self.answer_title = QLabel("")
        self.answer_title.setWordWrap(True)
        v.addWidget(self.answer_title)
        self.streams = QTableWidget(0, 6)
        self.streams.setHorizontalHeaderLabels(["stream", "ρ/ρ̄", "v (km/s)", "|v|", "came from q (Mpc)", "value"])
        self.streams.verticalHeader().setVisible(False)
        self.streams.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.streams.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.streams.horizontalHeader().setStretchLastSection(True)
        self.streams.setMaximumHeight(170)
        v.addWidget(self.streams)


class ExploreControls(QWidget):
    """Left pane. Owns the workers; talks to an ExploreView."""
    status = Signal(str)
    changed = Signal()          # a map shown or gone, or the server's phase: the menu bar follows
    # queued into the worker threads
    _to_render = Signal(object)
    _to_render_grid = Signal(object)
    _to_start = Signal(object)
    _to_query = Signal(object)
    _to_zoom = Signal(object)
    _to_exact = Signal(object)
    _to_stop = Signal()

    def __init__(self, view: ExploreView, data_root_fn, extra_dirs_fn):
        super().__init__()
        self.view = view
        self._data_root = data_root_fn          # () -> current data root
        self._extra_dirs = extra_dirs_fn        # () -> the custom-snapshot output directories
        self.outputs: list[tuple[str, G.OutputSet]] = []
        self.current: G.OutputSet | None = None
        self._plane = None
        self._planes = None                     # (A, B) of a difference map: the hover reads both
        self._arrows = None
        self._req = None
        self._pending = None
        self._busy = False
        self._server_output: G.OutputSet | None = None
        self._server_log: Path | None = None
        self._zoom: dict | None = None          # the zoomed region (see _dragged) once its fields arrived
        self._zoom_fields = None                # dtfelib PointFields on the zoom's plane of points
        self._zoom_pending = False
        self._grid_shown = False                # the figure grid is on the right pane instead of the map
        self._grid_index = None                 # the slice index the shown grid was composed at
        self._grid_busy = False                 # a tile request is with the worker; only the latest is served
        self._grid_pending = None
        self._grid_type_pref: dict = {}         # row -> (prefix, estimator) the user chose; a rebuild's fallback never replaces it

        col = QVBoxLayout(self)
        g = QGroupBox("Output")
        f = QFormLayout(g)
        self._out_form = f
        # the output to show, in two steps: the simulation (or the custom snapshots, or the files opened
        # by hand), then its snapshot by redshift (a custom run by name). A third menu appears only when
        # that snapshot has several outputs (prefixes, e.g. the phase-space and the standard run).
        self._by_root: dict[str, G.OutputSet] = {}
        self._pref_prefix = "ps_output"         # the prefix the user chose last: kept across snapshot and simulation changes
        self.sim_combo = QComboBox()
        self.sim_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.sim_combo.setMinimumContentsLength(16)
        self.sim_combo.setToolTip("The simulation whose outputs to show; the custom snapshots and the files opened "
                                  "with the … button have an entry of their own")
        self.sim_combo.currentIndexChanged.connect(self._sim_changed)
        row = QHBoxLayout()
        row.addWidget(self.sim_combo, 1)
        refresh = QToolButton(text="↻")
        refresh.setToolTip("Look for outputs again")
        refresh.clicked.connect(lambda: self.refresh_outputs())
        row.addWidget(refresh)
        open_btn = QToolButton(text="…")
        open_btn.setToolTip("Open any output: choose one of its grid files (e.g. <name>.a_den)")
        open_btn.clicked.connect(self._open_other)
        row.addWidget(open_btn)
        f.addRow("Simulation", row)
        self.snap_combo = QComboBox()
        self.snap_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.snap_combo.setMinimumContentsLength(16)
        self.snap_combo.setToolTip("The snapshot, by redshift (earliest first, today last); a custom run by its name")
        self.snap_combo.currentIndexChanged.connect(self._snap_changed)
        f.addRow("Redshift", self.snap_combo)
        self.prefix_combo = QComboBox()
        self.prefix_combo.setToolTip("This snapshot has several outputs (their file prefixes): which one to show")
        self.prefix_combo.currentIndexChanged.connect(self._prefix_changed)
        f.addRow("Output", self.prefix_combo)
        f.setRowVisible(self.prefix_combo, False)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray")
        f.addRow("", self.info)
        col.addWidget(g)

        g = QGroupBox("Map")
        f = QFormLayout(g)
        self._out_form_map = f
        self.field = QComboBox()
        self.field.currentIndexChanged.connect(self._field_changed)
        f.addRow("Field", self.field)
        self.component = QComboBox()
        self.component.currentIndexChanged.connect(self.render)
        crow = QHBoxLayout()
        crow.addWidget(self.component, 1)
        self.vectors = QCheckBox("vectors")
        self.vectors.setToolTip("Draw the in-plane components as arrows over the map")
        self.vectors.toggled.connect(self.render)
        crow.addWidget(self.vectors)
        f.addRow("Component", crow)
        ax = QHBoxLayout()
        self.axis = QComboBox()
        self.axis.addItems([f"{a} (normal)" for a in AXES])
        self.axis.setCurrentIndex(2)
        self.axis.currentIndexChanged.connect(self._axis_changed)
        ax.addWidget(self.axis)
        self.index = QSpinBox()
        self.index.valueChanged.connect(self._index_spin)
        ax.addWidget(self.index)
        f.addRow("Slice", ax)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._index_slider)
        f.addRow("", self.slider)
        self.position = QLabel("")
        self.position.setStyleSheet("color: gray")
        f.addRow("", self.position)
        cm = QHBoxLayout()
        self.cmap = QComboBox()
        self.cmap.addItems(CMAPS)
        self.cmap.currentIndexChanged.connect(self.render)
        cm.addWidget(self.cmap)
        self.log = QCheckBox("log₁₀")
        self.log.toggled.connect(self.render)
        cm.addWidget(self.log)
        f.addRow("Colours", cm)
        rg = QHBoxLayout()
        self.clip = QDoubleSpinBox(minimum=0.0, maximum=20.0, decimals=1, singleStep=0.5, value=1.0)
        self.clip.setSuffix(" % clipped at each end")
        self.clip.valueChanged.connect(self.render)
        rg.addWidget(self.clip)
        f.addRow("Range", rg)
        fx = QHBoxLayout()
        self.fixed = QCheckBox("fixed")
        self.fixed.setToolTip("Colour every map on this range (the field's own values; a log map takes positive "
                              "ones): the colours then compare between snapshots, outputs and slices")
        self.fixed.toggled.connect(self._fixed_toggled)
        fx.addWidget(self.fixed)
        self.vmin_edit, self.vmax_edit = QLineEdit(placeholderText="min"), QLineEdit(placeholderText="max")
        self._fixed_for = None              # the field the fixed range was set for: another field drops it
        self._fixed_warned = False
        for e in (self.vmin_edit, self.vmax_edit):
            e.setMaximumWidth(90)
            e.editingFinished.connect(self._fixed_edited)
            fx.addWidget(e)
        take = QToolButton(text="this map's")
        take.setToolTip("Fill in the range of the map on screen and fix it")
        take.clicked.connect(self._take_range)
        fx.addWidget(take)
        self.take_range_btn = take
        fx.addStretch(1)
        f.addRow("", fx)
        self.smooth = QDoubleSpinBox(minimum=0.0, maximum=20.0, decimals=1, singleStep=0.5)
        self.smooth.setSuffix(" cells (Gaussian)")
        self.smooth.valueChanged.connect(self.render)
        f.addRow("Smoothing", self.smooth)
        cmp_row = QHBoxLayout()
        self.compare = QComboBox()
        self.compare.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.compare.setMinimumContentsLength(12)
        self.compare.setToolTip("Show this map minus another output of the same grid (the same cells: another "
                                "estimator, prefix, precision or deposit of this snapshot, or another snapshot), "
                                "on a diverging scale centred on 0. Slice maps only: a zoom shows A alone")
        self.compare.currentIndexChanged.connect(self._compare_changed)
        cmp_row.addWidget(self.compare, 1)
        self.compare_mode = QComboBox()
        for text, mode in COMPARE_MODES:
            self.compare_mode.addItem(text, mode)
        self.compare_mode.setToolTip("A − B, or log₁₀ A/B where both are positive (grey where not): for density-like fields")
        self.compare_mode.currentIndexChanged.connect(self.render)
        cmp_row.addWidget(self.compare_mode)
        f.addRow("Compare with", cmp_row)
        self.pts_combo = QComboBox()
        self.pts_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.pts_combo.setMinimumContentsLength(12)
        self.pts_combo.setToolTip("Show the point-evaluated hi-res slice of this output ('.pts_*', the Pipeline's "
                                  "image plane: the field at each pixel centre, no grid) in place of the grid slice. "
                                  "Planes of one size are listed with their axis: the files do not say which they are")
        self.pts_combo.currentIndexChanged.connect(self._pts_changed)
        f.addRow("Hi-res slice", self.pts_combo)
        f.setRowVisible(self.pts_combo, False)
        self._pts_side = None
        col.addWidget(g)

        g = QGroupBox("Compare: a figure grid")
        v = QVBoxLayout(g)
        v.addWidget(self._note("Several maps as one figure: a field at several redshifts, or several fields of one "
                               "snapshot. Every panel takes this map's slice (the same axis, the same position across "
                               "the box), its range clip and smoothing; panels of one field share a colour range."))
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("Layout"))
        self.grid_layout = QComboBox()
        for r, c in GRID_LAYOUTS:
            self.grid_layout.addItem(f"{r} × {c}", f"{r}x{c}")     # a string: Qt matches it in findData
        self.grid_layout.setToolTip("Rows × columns of the figure")
        self.grid_layout.currentIndexChanged.connect(self._grid_layout_changed)
        lrow.addWidget(self.grid_layout)
        lrow.addWidget(QLabel("rows × columns"))
        lrow.addStretch(1)
        v.addLayout(lrow)
        self.grid_table = QTableWidget(0, len(GRID_COLS))
        self.grid_table.setHorizontalHeaderLabels(list(GRID_COLS))
        self.grid_table.verticalHeader().setVisible(False)
        self.grid_table.verticalHeader().setDefaultSectionSize(26)
        self.grid_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.grid_table.horizontalHeader().setSectionResizeMode(_C_CMP, QHeaderView.ResizeToContents)
        self.grid_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.grid_table.setMaximumHeight(9 * 26 + 30)
        self.grid_table.setToolTip("One row per panel, in reading order: the simulation, its snapshot by redshift, "
                                   "which of that snapshot's outputs (DTFE, PS-DTFE, another prefix), the field and "
                                   "the component. A row without a simulation leaves that panel blank. A row keeps "
                                   "its output type across redshifts when the snapshot has it.")
        v.addWidget(self.grid_table)
        frow = QHBoxLayout()
        self.grid_fill_z = QPushButton("This field across redshifts")
        self.grid_fill_z.setToolTip("Fill the panels with the map's field at this simulation's snapshots: the latest "
                                    "ones that have it, oldest first")
        self.grid_fill_z.clicked.connect(lambda: self.grid_fill("redshifts"))
        frow.addWidget(self.grid_fill_z)
        self.grid_fill_f = QPushButton("This snapshot across fields")
        self.grid_fill_f.setToolTip("Fill the panels with this output's fields, starting at the map's field")
        self.grid_fill_f.clicked.connect(lambda: self.grid_fill("fields"))
        frow.addWidget(self.grid_fill_f)
        self.grid_fill_t = QPushButton("This snapshot across outputs")
        self.grid_fill_t.setToolTip("Fill the panels with the map's field from every output of this snapshot: the "
                                    "standard DTFE and the phase-space runs side by side")
        self.grid_fill_t.clicked.connect(lambda: self.grid_fill("types"))
        frow.addWidget(self.grid_fill_t)
        frow.addStretch(1)
        v.addLayout(frow)
        brow = QHBoxLayout()
        self.grid_toggle = QPushButton("Show the grid")
        self.grid_toggle.setCheckable(True)
        self.grid_toggle.setToolTip("Compose the panels and show the figure in place of the map (any change to the map "
                                    "brings the map back)")
        self.grid_toggle.toggled.connect(self.show_grid)
        brow.addWidget(self.grid_toggle)
        self.grid_save_btn = QPushButton("Save figure…")
        self.grid_save_btn.setToolTip("Write the composed figure as PNG or PDF at 200 dpi")
        self.grid_save_btn.clicked.connect(self._save_grid_dialog)
        brow.addWidget(self.grid_save_btn)
        self.grid_shared = QCheckBox("one colour range per field")
        self.grid_shared.setChecked(True)
        self.grid_shared.setToolTip("Panels showing the same field share vmin/vmax (the union of their ranges), so "
                                    "the colours compare; unticked, every panel is stretched on its own")
        brow.addStretch(1)
        v.addLayout(brow)
        orow = QHBoxLayout()
        self.grid_shared.toggled.connect(lambda _on: self.show_grid(True) if self._grid_shown else None)
        orow.addWidget(self.grid_shared)
        self.grid_merge = QCheckBox("merge the axes: one colour bar")
        self.grid_merge.setToolTip(GRID_MERGE_TIP)
        self.grid_merge.toggled.connect(self._grid_merge_toggled)
        orow.addWidget(self.grid_merge)
        orow.addStretch(1)
        v.addLayout(orow)
        self.grid_state = QLabel("")
        self.grid_state.setWordWrap(True)
        self.grid_state.setStyleSheet("color: gray")
        v.addWidget(self.grid_state)
        col.addWidget(g)
        self._grid_sync_rows()

        g = QGroupBox("Streams at a point")
        v = QVBoxLayout(g)
        v.addWidget(self._note("With the server running, click the map."))
        g.setToolTip("A query server on this output's snapshot: a click lists every stream at that point, with "
                     "its density, velocity and the Lagrangian position its matter came from. A big "
                     "snapshot is served as partitions kept on disk in the cache folder.")
        cf = QFormLayout()
        self.cache_edit = QLineEdit()
        self.cache_edit.textEdited.connect(lambda _t: setattr(self, "_cache_user_set", True))
        self._cache_user_set = False
        crow = QHBoxLayout()
        crow.addWidget(self.cache_edit, 1)
        cb = QToolButton(text="…")
        cb.clicked.connect(self._browse_cache)
        crow.addWidget(cb)
        cf.addRow("Cache folder", crow)
        v.addLayout(cf)
        self.resident = QSpinBox(minimum=0, maximum=512)
        self.resident.setSpecialValueText("auto (the memory budget)")
        self.resident.setToolTip("A partitioned server keeps this many partition tessellations in memory (the rest "
                                 "stay in the cache folder and are reloaded when a click or zoom needs them). More "
                                 "keeps repeated zooms fast at the cost of memory; auto fits them to free memory")
        cf.addRow("Partitions in memory", self.resident)
        self._snapshots: dict[str, str] = {}    # output root -> a snapshot file chosen by hand
        brow = QHBoxLayout()
        self.server_btn = QPushButton("Start query server")
        self.server_btn.clicked.connect(self._server_button)
        brow.addWidget(self.server_btn)
        snap_btn = QToolButton(text="Snapshot…")
        snap_btn.setToolTip("The snapshot file this output was computed from, when the launcher cannot find it "
                            "(the input moved, or an output opened by hand): the server and the exact zoom read it")
        snap_btn.clicked.connect(self._choose_snapshot)
        brow.addWidget(snap_btn)
        self.server_state = QLabel("not running")
        self.server_state.setWordWrap(True)
        brow.addWidget(self.server_state, 1)
        v.addLayout(brow)
        col.addWidget(g)

        g = QGroupBox("Zoom")
        v = QVBoxLayout(g)
        v.addWidget(self._note("With the server running, drag a rectangle on the map: the region is evaluated "
                               "again, point by point, at the resolution below (no grid involved), and every "
                               "field of the list can be shown from that one evaluation. Drag again to go deeper. "
                               "'exact' runs the program itself on the region instead: mass-conserving cell "
                               "averages (the exact deposit) at that resolution, one slice cell thick. That is "
                               "a whole run, so slower, and needs no server."))
        zrow = QHBoxLayout()
        self.zoom_res = QSpinBox(minimum=64, maximum=4096, value=512, singleStep=64)
        self.zoom_res.setSuffix(" × same points")
        self.zoom_res.setToolTip("Points along each side of the zoomed region (512² = 262,144 points, "
                                 "a second or two on a small box)")
        zrow.addWidget(self.zoom_res)
        self.zoom_exact = QCheckBox("exact")
        self.zoom_exact.setToolTip("Run DTFE / PS-DTFE on the dragged region with the exact deposit: the region "
                                   "as a grid of that many cells across, one slice cell thick, every cell the exact "
                                   "volume average (mass is conserved). A whole run, seconds to minutes; the GPU is "
                                   "used when the program has it. A phase-space run tessellates the whole box as "
                                   "the original run did (periodic or not) and deposits only into the region, so "
                                   "the zoom shows exactly what a full run at that resolution would show there. It "
                                   "reuses the query server's tessellations on disk and skips the partitions whose "
                                   "streams never reach the region; a zoom that would take more than a minute asks "
                                   "first, with an estimate.")
        zrow.addWidget(self.zoom_exact)
        self.zoom_cancel_btn = QPushButton("Cancel")
        self.zoom_cancel_btn.setToolTip("Stop the exact zoom that is running")
        self.zoom_cancel_btn.setEnabled(False)
        self.zoom_cancel_btn.clicked.connect(self.cancel_exact_zoom)
        zrow.addWidget(self.zoom_cancel_btn)
        self.zoom_back_btn = QPushButton("Back")
        self.zoom_back_btn.setToolTip("Back to the previous zoom level, kept in memory (no new evaluation); "
                                      "at the first level, back to the slice")
        self.zoom_back_btn.setEnabled(False)
        self.zoom_back_btn.clicked.connect(self.zoom_back)
        zrow.addWidget(self.zoom_back_btn)
        self.zoom_out_btn = QPushButton("Zoom out")
        self.zoom_out_btn.setEnabled(False)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        zrow.addWidget(self.zoom_out_btn)
        zrow.addStretch(1)
        v.addLayout(zrow)
        prow = QHBoxLayout()
        self.zoom_progress = QProgressBar()
        self.zoom_progress.setTextVisible(False)
        self.zoom_progress.setMaximumHeight(10)
        self.zoom_progress.setVisible(False)
        prow.addWidget(self.zoom_progress, 1)
        self.zoom_time = QLabel("")
        self.zoom_time.setStyleSheet("color: gray")
        self.zoom_time.setToolTip("How long the zoom has been running (the last one: how long it took)")
        prow.addWidget(self.zoom_time)
        v.addLayout(prow)
        self.zoom_state = QLabel("")
        self.zoom_state.setWordWrap(True)
        self.zoom_state.setStyleSheet("color: gray")
        v.addWidget(self.zoom_state)
        col.addWidget(g)

        col.addStretch(1)

        # workers: their threads start on first use (a window that never opens Explore owns none)
        self._slice_thread = self._server_thread = None
        self._slice_worker = SliceWorker()
        self._slice_worker.done.connect(self._rendered)
        self._slice_worker.failed.connect(self._render_failed)
        self._to_render.connect(self._slice_worker.render)
        self._to_render_grid.connect(self._slice_worker.render_grid)
        self._slice_worker.grid_done.connect(self._grid_rendered)
        self._slice_worker.grid_failed.connect(self._grid_failed)
        self._server = ServerWorker()
        self._exact_worker = ExactZoomWorker()
        self._exact_thread = None
        self._to_exact.connect(self._exact_worker.run)
        self._exact_worker.done.connect(self._zoomed)
        self._exact_worker.failed.connect(self._exact_failed)
        self._exact_worker.progress.connect(self._exact_progress)
        self._exact_running = False
        # asked before a long exact zoom: None = a dialog; tests set a callable(text) -> bool
        self.confirm_exact = None
        self.server_partition = None              # tests: force the query server's split (a composite)
        # zoom timing and progress: a 0.1 s tick updates the elapsed time and, for a point zoom, follows
        # the server's '[request k/P] ...' lines in its log (--serve-progress)
        self._zoom_t0: float | None = None
        self._zoom_kind = ""
        self._zoom_log_pos = 0
        self._zoom_summary = None                 # the server's '[request done]' figures of the running zoom
        self._zoom_tick = QTimer(self, interval=100)
        self._zoom_tick.timeout.connect(self._tick_zoom)
        # earlier zoom levels (Back) and recent evaluations (a repeated drag), kept in memory
        self._zoom_history: list[tuple[dict, object]] = []
        from collections import OrderedDict
        self._zoom_memo: "OrderedDict[tuple, tuple[dict, object]]" = OrderedDict()
        self._server.started.connect(self._server_started)
        self._server.failed.connect(self._server_failed)
        self._server.answered.connect(self._answered)
        self._server.zoomed.connect(self._zoomed)
        self._server.stopped.connect(lambda: self._set_server_state("stopped"))
        self._to_start.connect(self._server.start)
        self._to_query.connect(self._server.query)
        self._to_zoom.connect(self._server.zoom)
        self._to_stop.connect(self._server.stop)
        self._server_running = self._server_starting = False
        self._shut = False                      # shutdown() ran: nothing may start a thread again
        self._build_timer = QTimer(self, interval=1000)
        self._build_timer.timeout.connect(self._poll_build)

        view.canvas.hovered.connect(self._hover)
        view.canvas.clicked.connect(self._click)
        view.canvas.dragged.connect(self._dragged)
        view.save_btn.clicked.connect(self._save_image)
        self._refresh_server_button()

    def _threads(self):
        if self._slice_thread is None:
            self._slice_thread, self._server_thread, self._exact_thread = QThread(self), QThread(self), QThread(self)
            self._slice_worker.moveToThread(self._slice_thread)
            self._server.moveToThread(self._server_thread)
            self._exact_worker.moveToThread(self._exact_thread)
            self._slice_thread.start()
            self._server_thread.start()
            self._exact_thread.start()

    @staticmethod
    def _note(text):
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet("color: gray")
        return lab

    # ---------------------------------------------------------------- outputs
    @staticmethod
    def _same(a, b) -> bool:
        """The same output, by its files' root: a refresh replaces the OutputSet objects."""
        return a is not None and b is not None and str(a.root) == str(b.root)

    @staticmethod
    def _group(o) -> str:
        """The first menu's entry an output belongs to: its simulation, else the custom snapshots,
        else the files opened by hand (marked GROUP_OPENED by _open_other)."""
        return o.sim or getattr(o, "group", GROUP_CUSTOM)

    @staticmethod
    def _snap_key(o):
        """The second menu's key of an output: the snapshot number in a simulation, else its root."""
        if o.sim:
            return o.snap if o.snap is not None else -1
        return str(o.root)

    def refresh_outputs(self, select: str | None = None):
        """Look for outputs again and show 'select' (a root), else the one shown now, else the latest
        snapshot of the first simulation. Files opened by hand stay listed."""
        keep = select or (str(self.current.root) if self.current else None)
        found = G.find_outputs(self._data_root(), self._extra_dirs())
        listed = {str(o.root) for _, o in found}
        opened = [(t, o) for t, o in self.outputs if self._group(o) == GROUP_OPENED and str(o.root) not in listed]
        self.outputs = found + opened
        self._by_root = {str(o.root): o for _, o in self.outputs}
        self._rebuild_sims()
        if self._grid_shown:                    # the map comes back below (or the empty state): not a stale picture
            self._leave_grid()
        self._grid_sync_rows()
        if not self.outputs:
            for c in (self.snap_combo, self.prefix_combo):
                c.blockSignals(True)
                c.clear()
                c.blockSignals(False)
            self._out_form.setRowVisible(self.prefix_combo, False)
            self.current = self._plane = self._arrows = None
            self.view.canvas.arrows = None
            self.info.setText("No grids yet: run the demo (Setup), a Grids job or a Custom snapshot job first.")
            self.view.canvas.set_rgb(None)
            self.view.canvas.setText("No outputs with grids found.")
            self._refresh_server_button()
            return 0
        if not (keep and self._select_output(keep)):
            self._rebuild_snaps()
            self._rebuild_prefixes()
            self._show_current()
        return len(self.outputs)

    def select_root(self, root: str):
        """Show the output whose files are '<root>.*' (the Runs browser's 'Explore')."""
        if not self._select_output(str(root)):
            self.refresh_outputs(select=str(root))

    def _rebuild_sims(self):
        """The first menu: the simulations with grids in natural order (TNG50, TNG100, TNG300), then
        the custom snapshots and the opened files when there are any. Keeps the current choice."""
        outs = [o for _, o in self.outputs]
        c = self.sim_combo
        cur = c.currentData()
        c.blockSignals(True)
        c.clear()
        for sim in sorted({o.sim for o in outs if o.sim}, key=G.sim_sort_key):
            n = len({o.snap for o in outs if o.sim == sim})
            c.addItem(sim, sim)
            c.setItemData(c.count() - 1, f"{n} snapshot{'s' if n != 1 else ''} with grids", Qt.ToolTipRole)
        for group, text in ((GROUP_CUSTOM, "Custom snapshots"), (GROUP_OPENED, "Opened files")):
            if any(self._group(o) == group for o in outs):
                c.addItem(text, group)
        i = c.findData(cur) if cur is not None else -1
        c.setCurrentIndex(i if i >= 0 else 0)
        c.blockSignals(False)

    def _rebuild_snaps(self, select=None, prefer=None):
        """The second menu for the chosen simulation: its snapshots by redshift (earliest first, so
        today is last and the default); for the custom snapshots and opened files, the runs by name.
        'select' (an OutputSet) picks its entry; else 'prefer' (a key: the snapshot shown in another
        simulation) when the menu has it."""
        grp = self.sim_combo.currentData()
        outs = [o for _, o in self.outputs if self._group(o) == grp]
        c = self.snap_combo
        c.blockSignals(True)
        c.clear()
        if grp in (GROUP_CUSTOM, GROUP_OPENED):
            label = "Run"
            for o in sorted(outs, key=lambda o: (o.dir.name.lower(), o.prefix.lower())):
                c.addItem(str(o.root) if grp == GROUP_OPENED else f"{o.prefix}  ({o.dir.name})", str(o.root))
                c.setItemData(c.count() - 1, str(o.root) + ".*", Qt.ToolTipRole)
        else:
            label = "Redshift"
            first = {}
            for o in outs:
                first.setdefault(self._snap_key(o), o)
            for key in sorted(first):
                o = first[key]
                z = f"z = {o.redshift:.2f}   ·   " if o.redshift is not None else ""
                c.addItem(z + (f"snapshot {key:03d}" if key >= 0 else "snapshot ?"), key)
        lab = self._out_form.labelForField(c)
        if lab is not None:
            lab.setText(label)
        want = self._snap_key(select) if select is not None else prefer
        i = c.findData(want) if want is not None else -1
        c.setCurrentIndex(i if i >= 0 else c.count() - 1)
        c.blockSignals(False)

    def _rebuild_prefixes(self, select=None):
        """The third menu: the outputs of the chosen snapshot by prefix, shown only when there are
        several; its current entry is the output on screen. Without 'select': the prefix the user chose
        last, else the production phase-space one (ps_output, then any ps prefix), else the first."""
        grp, key = self.sim_combo.currentData(), self.snap_combo.currentData()
        if grp in (GROUP_CUSTOM, GROUP_OPENED):
            outs = [o for _, o in self.outputs if self._group(o) == grp and str(o.root) == key]
        else:
            outs = [o for _, o in self.outputs if self._group(o) == grp and self._snap_key(o) == key]
        c = self.prefix_combo
        c.blockSignals(True)
        c.clear()
        for o in sorted(outs, key=lambda o: o.prefix):
            c.addItem(o.prefix, str(o.root))
            c.setItemData(c.count() - 1, str(o.root) + ".*", Qt.ToolTipRole)
        i = c.findData(str(select.root)) if select is not None else -1
        if i < 0:
            i = self._preferred_prefix([c.itemText(k) for k in range(c.count())])
        c.setCurrentIndex(i)
        c.blockSignals(False)
        self._out_form.setRowVisible(c, c.count() > 1)

    def _preferred_prefix(self, names: list) -> int:
        """Which of a snapshot's prefixes to show unasked: the one the user chose last, else the
        production phase-space one (ps_output, then any ps prefix), else the first."""
        for want in (self._pref_prefix, G.PS_PREFIX):
            if want in names:
                return names.index(want)
        ps = [k for k, nm in enumerate(names) if nm.startswith("ps")]
        return ps[0] if ps else 0

    def _select_output(self, root: str) -> bool:
        """Set the three menus to the output '<root>.*' and show it; False when it is not listed."""
        o = self._by_root.get(str(root))
        if o is None:
            return False
        i = self.sim_combo.findData(self._group(o))
        if i < 0:
            return False
        self.sim_combo.blockSignals(True)
        self.sim_combo.setCurrentIndex(i)
        self.sim_combo.blockSignals(False)
        self._rebuild_snaps(select=o)
        self._rebuild_prefixes(select=o)
        self._pref_prefix = o.prefix           # an output picked by name (Runs, the demo, a file): its prefix sticks
        self._show_current()
        return True

    def _current_output(self):
        root = self.prefix_combo.currentData()
        return self._by_root.get(root) if root else None

    def _show_current(self):
        o = self._current_output()
        if o is not None:
            self._show_output(o)

    def _sim_changed(self, *_):
        self._rebuild_snaps(prefer=self.snap_combo.currentData())   # the same snapshot when the simulation has it
        self._rebuild_prefixes()
        self._show_current()

    def _snap_changed(self, *_):
        self._rebuild_prefixes()
        self._show_current()

    def _prefix_changed(self, *_):
        if self.prefix_combo.currentIndex() >= 0:
            self._pref_prefix = self.prefix_combo.currentText()     # the user's choice sticks
        self._show_current()

    def _open_other(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open an output: choose one of its grid files",
                                           self._data_root(), "Grid files (*.a_den *.den *.a_vel *.vel);;All files (*)")
        if f:
            self.open_file(f)

    def open_file(self, f) -> bool:
        """Open any output by one of its grid files: an output already listed is shown where it is;
        another is listed under 'Opened files' (once) and shown."""
        root = Path(f).parent / Path(f).stem          # 'snap_z0.5.a_den' -> 'snap_z0.5' (no grid suffix has a dot)
        if str(root) in self._by_root:
            return self._select_output(str(root))
        o = G.OutputSet(root, box=G.custom_box(G.sidecar(root) or {}), title=f"opened · {root}")
        if not o:
            self.status.emit(f"no readable grids next to {f}")
            return False
        o.group = GROUP_OPENED
        self.outputs.append((o.title, o))
        self._by_root[str(o.root)] = o
        self._rebuild_sims()
        self._grid_sync_rows()                  # the composer's menus list it too (as after a refresh)
        return self._select_output(str(o.root))

    def _show_output(self, o):
        """Show an output: its field list, info line, slice controls, and the first map. The same
        output again (after a refresh: a new object for the same files) keeps the slice."""
        prev = self.current
        same_grid = self._same(prev, o) and prev.n == o.n and prev.dim == o.dim
        slice_now = self.index.value()
        self.current = o
        keep = self.field.currentData()
        self.field.blockSignals(True)
        self.field.clear()
        for name in o.fields():
            self.field.addItem(G.FIELD_LABELS[name], name)
            if name in G.FIELD_HELP:
                self.field.setItemData(self.field.count() - 1, G.FIELD_HELP[name], Qt.ToolTipRole)
        self.field.blockSignals(False)
        j = self.field.findData(keep)
        self.field.setCurrentIndex(j if j >= 0 else 0)
        if not self._cache_user_set:
            near = o.dir if G.sidecar(o.root) is not None else self._data_root()
            self.cache_edit.setText(str(default_cache_dir(str(near))))
        box = "box unknown" if o.box is None else f"box {o.length[0]:.1f} Mpc"
        grid = f"{o.n}² grid (2D)" if o.dim == 2 else f"{o.n}³ grid"
        self.info.setText(f"{grid} · {box}" + (f" · z = {o.redshift:.2f}" if o.redshift is not None else ""))
        self.info.setToolTip(str(o.root) + ".*")
        flat = o.dim == 2               # a 2D grid is its one plane: no axis, no slice to choose
        self.axis.blockSignals(True)
        if flat:
            self.axis.setCurrentIndex(2)
        self.axis.setEnabled(not flat)
        self.axis.blockSignals(False)
        for w in (self.index, self.slider):
            w.blockSignals(True)
            w.setRange(0, 0 if flat else o.n - 1)
            w.setValue(0 if flat else slice_now if same_grid else o.n // 2)
            w.setEnabled(not flat)
            w.blockSignals(False)
        self._pts_fill()                    # BEFORE the field's render: it reads the menu (this output's planes)
        self._field_changed()
        self._refresh_server_button()

    def _field_changed(self, *_):
        o = self.current
        name = self.field.currentData()
        if o is None or name is None:
            return
        if self.fixed.isChecked() and self._fixed_for not in (None, name):    # a range belongs to its field
            self.fixed.blockSignals(True)
            self.fixed.setChecked(False)
            self.fixed.blockSignals(False)
            self.status.emit(f"the fixed range was set for {G.FIELD_LABELS.get(self._fixed_for, self._fixed_for)}: off "
                             "for this field (tick 'fixed' to use it here)")
        ff = o.files[name]
        self.component.blockSignals(True)
        self.component.clear()
        self.component.addItems(G.components(name, ff.ncomp, o.dim))
        self.component.setEnabled(ff.ncomp > 1)
        self.component.blockSignals(False)
        vec = name in G.VECTOR_FIELDS and ff.ncomp == o.dim
        self.vectors.blockSignals(True)
        self.vectors.setProperty("can", vec)
        self.vectors.setEnabled(vec)
        self.vectors.setChecked(vec and name == "velocity")     # the velocity map opens as a vector field
        self.vectors.blockSignals(False)
        for w, on in ((self.log, name in G.LOG_DEFAULT),):
            w.blockSignals(True)
            w.setChecked(on)
            w.blockSignals(False)
        cmap = "RdBu_r" if name in G.DIVERGING else "tab10" if name in ("tweb", "vweb") else \
            "magma" if name in ("density", "dispersion") else "viridis"
        self.cmap.blockSignals(True)
        self.cmap.setCurrentText(cmap)
        self.cmap.blockSignals(False)
        self._compare_fill()
        self.render()

    def _axis_changed(self, *_):
        self.render()

    def _index_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self.slider.blockSignals(False)
        self.render()

    def _index_slider(self, v):
        self.index.blockSignals(True)
        self.index.setValue(v)
        self.index.blockSignals(False)
        self.render()

    def step_slice(self, delta: int):
        """The arrow keys: the next or previous slice along the current axis (stops at the box's
        edge)."""
        if self.current is not None:
            self.index.setValue(max(0, min(self.index.maximum(), self.index.value() + delta)))

    def step_field(self, delta: int):
        """Up / down: the previous or next field in the list (stops at either end)."""
        n = self.field.count()
        if n:
            self.field.setCurrentIndex(max(0, min(n - 1, self.field.currentIndex() + delta)))

    # ---------------------------------------------------------------- drawing
    def request(self) -> dict | None:
        o, name = self.current, self.field.currentData()
        if o is None or name is None:
            return None
        req = {"output": o, "field": name, "axis": self.axis.currentIndex(), "index": self.index.value(),
               "component": self.component.currentText(), "cmap": self.cmap.currentText(),
               "log": self.log.isChecked(), "clip": self.clip.value(), "smooth": self.smooth.value(),
               "symmetric": name in G.DIVERGING and not self.log.isChecked(),
               "arrows": self.vectors.isEnabled() and self.vectors.isChecked()}
        if self.pts_combo.currentData():
            req.update(pts=self.pts_combo.currentData(), arrows=False)
            if name in ("shear", "vorticity"):  # the hi-res slice holds their magnitudes (>= 0): no component, no
                req.update(component="magnitude", symmetric=False,     # scale centred on 0
                           cmap="viridis" if req["cmap"] == "RdBu_r" else req["cmap"])
        fixed = self._fixed_range(req["log"])
        self._sync_fixed(req["log"])
        if fixed is not None:
            req["vrange"] = fixed
        other = self._compare_output()
        if other is not None:                   # the difference map: its own scale, so Save image and the bar agree
            req.pop("vrange", None)
            req.update(compare=other, compare_mode=self.compare_mode.currentData(), log=False, symmetric=True,
                       cmap="RdBu_r", arrows=False)
        return req

    def render(self, *_):
        req = self.request()
        if req is None or self._shut:
            return
        if self._grid_shown:
            prev = self._req or {}
            same_view = (bool(prev) and self._same(prev.get("output"), req["output"])
                         and prev.get("field") == req["field"] and prev.get("component") == req["component"])
            if same_view:                       # the slice (arrow keys, slider), colours, range, smoothing: the grid follows
                self._req = req
                self._set_position(req)
                self._grid_update_merge()
                self._grid_refresh_soon()
                return
            self._leave_grid()                  # another field or output: the map
            req = self.request()                # ... with the comparison, which the grid left out
        self._grid_update_merge()               # the slice axis decides which box edges the panels share
        o = req["output"]
        self._set_position(req)
        if self._zoom is not None:
            z = self._zoom
            if self._same(z["output"], o) and z["axis"] == req["axis"] and z["index"] == req["index"]:
                if self._zoom_fields is not None:       # a colour/field change: no new evaluation
                    self._req = req
                    self._render_zoom()
                return
            self._clear_zoom()                          # another slice or output: back to its map
            req = self.request()                        # ... with the comparison, which the zoom left out
        if self._busy:
            self._pending = req             # only the latest request is served
            return
        self._busy = True
        self._req = req
        self._threads()
        self._to_render.emit(req)

    def _fixed_range(self, log: bool):
        """The fixed colour range in the bar's units (log10 on a log map), or None: off, unreadable, empty, or
        a log map with a bound <= 0."""
        if not self.fixed.isChecked():
            return None
        try:
            lo, hi = float(self.vmin_edit.text()), float(self.vmax_edit.text())
        except ValueError:
            return None
        if not hi > lo or (log and lo <= 0):
            return None
        return (float(np.log10(lo)), float(np.log10(hi))) if log else (lo, hi)

    def _fixed_toggled(self, on: bool):
        if on:
            self._fixed_for = self.field.currentData()
        if on and not (self.vmin_edit.text().strip() and self.vmax_edit.text().strip()):
            self._take_range()              # nothing typed yet: start from the map on screen
            return
        self.render()

    def _fixed_edited(self):
        if self.fixed.isChecked():
            self._fixed_for = self.field.currentData()
            self.render()

    def _sync_fixed(self, log: bool) -> None:
        """The clip spin is greyed only while a fixed range is IN USE (not for a range that cannot apply -- a
        log map with a bound <= 0 -- nor while the figure grid shows: the grid uses the clip); a fixed range that
        cannot apply says so once."""
        usable = self.fixed.isChecked() and self._fixed_range(log) is not None
        self.clip.setEnabled(not usable or self._grid_shown)
        if self.fixed.isChecked() and not usable:
            if not self._fixed_warned:
                self.status.emit("the fixed range does not apply to this map (a log map needs min > 0): its own "
                                 "percentile range is shown")
                self._fixed_warned = True
        else:
            self._fixed_warned = False

    def _take_range(self):
        """The map's current colour range into the fields (linear values), fixed from now on."""
        lo, hi = self.view.colorbar.vrange
        if self.view.colorbar.log:
            lo, hi = 10.0 ** lo, 10.0 ** hi
        self._fixed_for = self.field.currentData()
        self.vmin_edit.setText(f"{lo:.4g}")
        self.vmax_edit.setText(f"{hi:.4g}")
        if not self.fixed.isChecked():
            self.fixed.setChecked(True)         # _fixed_toggled renders
        else:
            self.render()

    def state(self) -> dict:
        """What the window remembers of Explore across restarts."""
        return {"root": str(self.current.root) if self.current is not None else "",
                "field": self.field.currentData(), "component": self.component.currentText(),
                "axis": self.axis.currentIndex(), "index": self.index.value(),
                "fixed": self.fixed.isChecked(), "vmin": self.vmin_edit.text(), "vmax": self.vmax_edit.text(),
                "snapshots": dict(self._snapshots), "resident": self.resident.value()}

    def restore_state(self, d: dict) -> bool:
        """Back to a remembered state when its output is still listed (selected first by refresh_outputs)."""
        if isinstance(d, dict):                 # the hand-chosen snapshots and the server's residency: for every output
            self._snapshots.update({k: v for k, v in (d.get("snapshots") or {}).items() if Path(v).is_file()})
            self.resident.setValue(int(d.get("resident", 0) or 0))
            self._refresh_server_button()       # refresh_outputs ran first: it judged the server with no snapshots
        if not d or self.current is None or str(self.current.root) != d.get("root"):
            return False
        j = self.field.findData(d.get("field"))
        if j >= 0:
            self.field.setCurrentIndex(j)
        k = self.component.findText(d.get("component", ""))
        if k >= 0:
            self.component.setCurrentIndex(k)
        if self.axis.isEnabled() and d.get("axis") in (0, 1, 2):
            self.axis.setCurrentIndex(int(d["axis"]))
        if self.index.isEnabled():
            self.index.setValue(int(d.get("index", self.index.value())))
        self.vmin_edit.setText(str(d.get("vmin", "")))
        self.vmax_edit.setText(str(d.get("vmax", "")))
        self.fixed.setChecked(bool(d.get("fixed")))
        return True

    def _canvas_axes(self) -> dict | None:
        """The axes of what the map shows (the slice, or the zoomed region): Mpc, or cells without a box."""
        g = self._geometry()
        if g is None:
            return None
        _axis, _index, rest, lo, hi, _n0, _n1 = g
        return {"x": (float(lo[0]), float(hi[0]), AXES[rest[0]]), "y": (float(lo[1]), float(hi[1]), AXES[rest[1]]),
                "unit": "Mpc" if self._req["output"].box is not None else "cells"}

    @staticmethod
    def comparable(a, b, field: str) -> bool:
        """B can be subtracted from A cell by cell: another output on the same cells (dim, grid, box) holding
        the field with the same components."""
        return (a is not None and b is not None and str(a.root) != str(b.root) and a.dim == b.dim and a.n == b.n
                and (a.box is None) == (b.box is None) and np.allclose(a.lo, b.lo) and np.allclose(a.length, b.length)
                and field in b.files and field in a.files and b.files[field].ncomp == a.files[field].ncomp)

    def _compare_fill(self):
        """The 'Compare with' menu: off, then every comparable output -- this snapshot's other runs first."""
        o, name = self.current, self.field.currentData()
        keep = self.compare.currentData()
        self.compare.blockSignals(True)
        self.compare.clear()
        self.compare.addItem("(off)", None)
        if o is not None and name is not None:
            cands = [b for _, b in self.outputs if self.comparable(o, b, name)]
            cands.sort(key=lambda b: (b.dir != o.dir, b.sim, -1 if b.snap is None else b.snap, str(b.root)))
            for b in cands:
                where = b.sim or b.dir.name
                z = f" · z = {b.redshift:.2f}" if b.redshift is not None else ""
                self.compare.addItem(f"{where}{z} · {G.type_label(b)}", str(b.root))
                self.compare.setItemData(self.compare.count() - 1, str(b.root) + ".*", Qt.ToolTipRole)
        j = self.compare.findData(keep) if keep is not None else 0
        self.compare.setCurrentIndex(max(j, 0))
        self.compare.blockSignals(False)
        signed = name in G.DIVERGING or name in ("velocity", "tweb", "vweb")
        self.compare_mode.model().item(1).setEnabled(not signed)      # log A/B: positive fields only
        if signed:
            self.compare_mode.blockSignals(True)
            self.compare_mode.setCurrentIndex(0)
            self.compare_mode.blockSignals(False)
        self._compare_sync()

    def _compare_output(self):
        """B of the difference map, or None (off, or a zoom / the figure grid on screen)."""
        root = self.compare.currentData()
        if root is None or self.is_zoomed() or self._grid_shown or self.pts_combo.currentData():
            return None
        return self._by_root.get(root)

    def _compare_sync(self):
        """The colour controls do not apply to a difference map (its own diverging scale); the menu itself
        not to a zoom or the figure grid."""
        on = self.compare.currentData() is not None
        free = not self.is_zoomed() and not self._grid_shown and not self.pts_combo.currentData()
        self.compare.setEnabled(free and self.compare.count() > 1)
        self.compare_mode.setEnabled(free and on)
        for w in (self.cmap, self.log, self.fixed, self.take_range_btn):
            w.setEnabled(not (on and free))
        self.vectors.setEnabled(self.vectors.property("can") is not False and not (on and free))

    def _pts_fill(self):
        """The hi-res slice menu for the output shown: off + every matching plane; hidden without one."""
        keep = self.pts_combo.currentData()
        planes = G.find_pts_planes(self.current) if self.current is not None else []
        self.pts_combo.blockSignals(True)
        self.pts_combo.clear()
        self.pts_combo.addItem("off: the grid slice", None)
        for label, side in planes:
            self.pts_combo.addItem(label, str(side))
        j = self.pts_combo.findData(keep) if keep else 0
        self.pts_combo.setCurrentIndex(max(j, 0))
        self.pts_combo.blockSignals(False)
        self._out_form_map.setRowVisible(self.pts_combo, bool(planes))
        self._pts_sync()

    def _pts_sync(self):
        """With the hi-res slice shown the slice axis and position do not apply (the plane is the sidecar's)."""
        on = bool(self.pts_combo.currentData())
        flat = self.current is not None and self.current.dim == 2
        for w in (self.axis, self.index, self.slider):
            w.setEnabled(not on and not flat)

    def _pts_changed(self, *_):
        if self.pts_combo.currentData() and self.is_zoomed():
            self._clear_zoom()
        self._pts_sync()
        self._compare_sync()
        self.render()

    def _compare_changed(self, *_):
        self._compare_sync()
        self.render()

    def _set_position(self, req: dict):
        o = req["output"]
        pos = o.lo[req["axis"]] + (req["index"] + 0.5) * o.length[req["axis"]] / o.n
        unit = "Mpc" if o.box is not None else "cells"
        self.position.setText("2D: the whole plane" if o.dim == 2 else f"{AXES[req['axis']]} = {pos:.2f} {unit}")

    def _rendered(self, res: dict):
        self._busy = False
        if self._pending is not None:
            req, self._pending = self._pending, None
            self.render_request(req)
        req = res["request"]
        self._plane = res["plane"]
        self._req = req
        self._arrows = res.get("arrows")
        if self._grid_shown:                    # a slice in flight when the grid was shown: keep the picture
            return
        self._planes = res.get("planes")
        self._pts_side = res.get("pts_side")
        self.view.canvas.setStyleSheet("")
        self.view.canvas.marker = None
        self.view.canvas.labels = None
        self.view.canvas.arrows = self._arrows
        self.view.canvas.axes = self._canvas_axes()      # BEFORE the image: its first drawing has the margins
        self.view.canvas.set_rgb(res["rgb"])
        label = G.FIELD_LABELS.get(req["field"], req["field"])
        comp = "" if req["component"] in ("value", "") else f" [{req['component']}]"
        o = req["output"]
        if req.get("compare") is not None:
            mode = dict((m, t) for t, m in COMPARE_MODES)[req["compare_mode"]]
            self.view.colorbar.set(req["cmap"], res["vrange"], f"{mode}: {label}{comp}")
            self.view.title.setText(f"<b>{mode}: {label}{comp}</b> — A = {o.title}, B = {req['compare'].title} — "
                                    f"{self.position.text()}")
        else:
            self.view.colorbar.set(req["cmap"], res["vrange"], ("log₁₀ " if req["log"] else "") + label + comp, req["log"])
            where = self.position.text()
            if req.get("pts") and self._pts_side is not None:
                sd, fct = self._pts_side, res.get("pts_factor") or 1
                where = (f"hi-res slice: {sd.get('axis')} = {float(sd.get('center', 0)):.2f} Mpc, {sd['nu']}×{sd['nv']} "
                         f"points" + (f" (shown {fct}×{fct}-averaged)" if fct > 1 else "") + ", point evaluation")
            self.view.title.setText(f"<b>{label}{comp}</b> — {o.title} — {where}")
        self.changed.emit()

    def render_request(self, req):
        self._busy = True
        self._to_render.emit(req)

    def _render_failed(self, msg):
        self._busy = False
        if self._pending is not None:           # a newer slice is wanted: its result decides, not this one's
            req, self._pending = self._pending, None
            self.render_request(req)
            return
        self._plane = self._arrows = None
        self.view.canvas.arrows = None
        self.view.canvas.set_rgb(None)
        self.view.canvas.setText(f"Could not read this slice: {msg}")
        self.changed.emit()

    def can_save(self) -> bool:
        return self._plane is not None or self._grid_shown

    def is_zoomed(self) -> bool:
        return self._zoom is not None and self._zoom_fields is not None

    def _geometry(self):
        """What the map shows: (axis, index, rest axes, lo and hi of the rest axes, n0, n1) -- the
        whole slice, the zoomed region at its own resolution, or a hi-res slice's plane."""
        req, plane = self._req, self._plane
        if req is None or plane is None:
            return None
        if req.get("pts") and self._pts_side is not None:
            sd = self._pts_side
            rest = [AXES.index(sd["u_axis"]), AXES.index(sd["v_axis"])]
            return (AXES.index(sd["axis"]), 0, rest, np.array([sd["u0"], sd["v0"]], float),
                    np.array([sd["u1"], sd["v1"]], float), plane.shape[0], plane.shape[1])
        o, axis = req["output"], req["axis"]
        rest = [a for a in range(3) if a != axis]
        if self.is_zoomed():
            z = self._zoom
            return (axis, req["index"], rest, np.array([z["a0"][0], z["a1"][0]]),
                    np.array([z["a0"][1], z["a1"][1]]), z["res"], z["res"])
        return axis, req["index"], rest, o.lo[rest], o.lo[rest] + o.length[rest], o.n, o.n

    def _point(self, r: int, c: int):
        """Image pixel -> (3D position, value): the centre of that pixel's cell or zoom point."""
        g = self._geometry()
        if g is None:
            return None
        axis, index, rest, lo, hi, n0, n1 = g
        plane = self._plane
        row, col = c, n1 - 1 - r              # image = plane.T[::-1]; plane rows = first remaining axis
        if not (0 <= row < n0 and 0 <= col < n1):
            return None
        o = self._req["output"]
        x = np.zeros(3)
        x[axis] = o.lo[axis] + (index + 0.5) * o.length[axis] / o.n
        if self._req.get("pts") and self._pts_side is not None:
            x[axis] = float(self._pts_side.get("center", x[axis]))      # the hi-res plane's own position
        x[rest[0]] = lo[0] + (row + 0.5) / n0 * (hi[0] - lo[0])
        x[rest[1]] = lo[1] + (col + 0.5) / n1 * (hi[1] - lo[1])
        return x, float(plane[row, col]), (row, col)

    def _cell(self, r: int, c: int):
        """Image pixel -> the 3D cell (i, j, k) of the slice (not for a zoom)."""
        req, plane = self._req, self._plane
        if req is None or plane is None or self.is_zoomed():
            return None
        n0, n1 = plane.shape                  # plane rows = first remaining axis
        row, col = c, n1 - 1 - r              # image = plane.T[::-1]
        if not (0 <= row < n0 and 0 <= col < n1):
            return None
        return req["output"].plane_point(req["axis"], req["index"], row, col), plane[row, col]

    def _hover(self, r, c):
        if r < 0 or self._grid_shown:
            self.view.readout.setText(" ")
            return
        pt = self._point(r, c)
        if pt is None:
            return
        x, value, (row, col) = pt
        o = self._req["output"]
        unit = "Mpc" if o.box is not None else "cells"
        where = f"zoom point ({row}, {col})" if self.is_zoomed() else \
            f"hi-res point ({row}, {col})" if self._req.get("pts") else \
            f"cell {tuple(int(v) for v in o.plane_point(self._req['axis'], self._req['index'], row, col))}"
        xyz = "   ".join(f"{a} = {v:.2f}" for a, v in zip(AXES[:o.dim], x))
        both = ""
        if self._planes is not None and not self.is_zoomed():
            a, b = self._planes
            both = f"   (A = {float(a[row, col]):.5g}, B = {float(b[row, col]):.5g})"
        self.view.readout.setText(f"{xyz} {unit}      {where}      value = {value:.5g}{both}")

    def _save_image(self):
        if self._grid_shown:
            self._save_grid_dialog()
            return
        if self._plane is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Save the map", str(Path.home() / "explore.png"),
                                           "PNG image (*.png);;NumPy array of the values (*.npy);;"
                                           "CSV table: x, y, value (*.csv)")
        if not f:
            return
        if Path(f).suffix.lower() in (".npy", ".csv"):
            try:
                self.status.emit(f"saved {self.save_values(f)}")
            except OSError as e:
                self.status.emit(f"could not save {f}: {e}")
            return
        req = self._req
        try:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            o = req["output"]
            data = self._plane.T
            if req["log"]:
                data = np.log10(np.where(data > 0, data, np.nan))
            _, _, rest, glo, ghi, _, _ = self._geometry()    # a hi-res slice's axes are its file's, not the menu's
            ext = [glo[0], ghi[0], glo[1], ghi[1]]
            fig, ax = plt.subplots(figsize=(7, 6))
            lo, hi = self.view.colorbar.vrange
            im = ax.imshow(data, origin="lower", extent=ext, cmap=req["cmap"], vmin=lo, vmax=hi)
            unit = "Mpc" if o.box is not None else "cells"
            ax.set_xlabel(f"{AXES[rest[0]]} [{unit}]")
            ax.set_ylabel(f"{AXES[rest[1]]} [{unit}]")
            ax.set_title(self.view.title.text().replace("<b>", "").replace("</b>", ""), fontsize=8)
            fig.colorbar(im, ax=ax, label=self.view.colorbar.label)
            arrows = getattr(self, "_arrows", None)
            if arrows is not None and arrows["vmax"] > 0:
                h = o.length / o.n                              # cell size per axis
                X, Y = np.meshgrid(o.lo[rest[0]] + arrows["c0"] * h[rest[0]],
                                   o.lo[rest[1]] + arrows["c1"] * h[rest[1]], indexing="ij")
                ax.quiver(X, Y, arrows["u"], arrows["v"], color="white", edgecolor="black", linewidth=0.3,
                          angles="xy", scale_units="xy", pivot="middle",
                          scale=arrows["vmax"] / (0.9 * arrows["spacing"] * h[rest[0]]), width=0.003)
            fig.savefig(f, dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            self.view.canvas.grab().save(f)
        self.status.emit(f"saved {f}")

    def save_values(self, path) -> str:
        """The map's VALUES (not its colours): '.npy' = the 2D array as drawn, rows along the vertical axis from
        the bottom (values[iy, ix]); '.csv' = one row per cell, 'x,y,value' at the cell centres in Mpc (cells
        without a box). The field's own units -- not log10 -- and the difference for a difference map."""
        path = Path(path)
        g = self._geometry()
        if g is None or self._plane is None:
            raise OSError("no map on screen")
        _axis, _index, rest, lo, hi, n0, n1 = g
        values = np.asarray(self._plane, dtype=np.float64).T            # [iy, ix], origin at the bottom
        if path.suffix.lower() == ".npy":
            np.save(path, values)
            return str(path)
        xs = lo[0] + (np.arange(n0) + 0.5) / n0 * (hi[0] - lo[0])
        ys = lo[1] + (np.arange(n1) + 0.5) / n1 * (hi[1] - lo[1])
        X, Y = np.meshgrid(xs, ys)
        unit = "Mpc" if self._req["output"].box is not None else "cells"
        head = f"{AXES[rest[0]]}_{unit},{AXES[rest[1]]}_{unit},value"
        np.savetxt(path, np.column_stack([X.ravel(), Y.ravel(), values.ravel()]), delimiter=",", header=head,
                   comments="", fmt="%.8g")
        return str(path)

    # ---------------------------------------------------------------- the figure grid (compare)
    def _grid_layout_changed(self, *_):
        self._grid_sync_rows()
        if self._grid_shown:
            self.show_grid(True)

    def grid_dims(self) -> tuple[int, int]:
        """The layout as (rows, columns)."""
        r, _x, c = (self.grid_layout.currentData() or "1x2").partition("x")
        return int(r), int(c)

    def grid_set_layout(self, rows: int, cols: int) -> bool:
        j = self.grid_layout.findData(f"{rows}x{cols}")
        if j >= 0:
            self.grid_layout.setCurrentIndex(j)
        return j >= 0

    def _grid_sync_rows(self):
        """One table row per panel of the layout (rows beyond it hidden, their choices kept); the output
        menus list every output known (a refresh keeps each row's choice by its root). Never recomposes
        by itself: the callers do, once."""
        rows, cols = self.grid_dims()
        want = rows * cols
        while self.grid_table.rowCount() < max(want, self.grid_table.rowCount()):
            self._grid_make_row(self.grid_table.rowCount())
        hold, self._grid_shown_hold = getattr(self, "_grid_shown_hold", False), True
        try:
            self._grid_sync_rows_inner(want)
        finally:
            self._grid_shown_hold = hold
        self._grid_update_merge()

    def _grid_sync_rows_inner(self, want: int):
        for i in range(self.grid_table.rowCount()):
            self.grid_table.setRowHidden(i, i >= want)
            self._grid_row_sims(i)

    def _grid_make_row(self, i: int):
        self.grid_table.insertRow(i)
        for c in range(len(GRID_COLS)):
            w = QComboBox()
            w.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(6 if c < _C_FLD else 8)
            self.grid_table.setCellWidget(i, c, w)
        cw = self.grid_table.cellWidget
        cw(i, _C_SIM).currentIndexChanged.connect(lambda _j, i=i: self._grid_row_snaps(i))
        cw(i, _C_SNAP).currentIndexChanged.connect(lambda _j, i=i: self._grid_row_types(i))
        cw(i, _C_TYPE).currentIndexChanged.connect(lambda _j, i=i: self._grid_type_picked(i))
        cw(i, _C_FLD).currentIndexChanged.connect(lambda _j, i=i: self._grid_components_changed(i))
        cw(i, _C_CMP).currentIndexChanged.connect(lambda _j, i=i: (self._grid_row_tips(i), self._grid_row_edited()))

    def _grid_type_picked(self, i: int):
        """The user chose a row's Type: remembered apart from the menu, so a rebuild's fallback (a snapshot
        without that run) never replaces the choice and stepping back to one that has it brings it back.
        Rebuilds block the menu's signals, so only a user's pick lands here."""
        typ = self.grid_table.cellWidget(i, _C_TYPE)
        if typ.currentData():
            self._grid_type_pref[i] = (typ.currentData(Qt.UserRole + 1), typ.currentData(Qt.UserRole + 2), typ.currentData())
        self._grid_fields_changed(i, keep_field=True)

    def _grid_row_tips(self, i: int):
        """Each cell menu's tooltip = its full text: the five columns share the pane, and a closed Type menu
        clips 'PS-DTFE · ps_vwsh' to the estimator -- the part that tells two runs apart. Type adds the root."""
        for c in range(len(GRID_COLS)):
            w = self.grid_table.cellWidget(i, c)
            if w is None:
                continue
            tip = w.currentText()
            if c == _C_TYPE and w.currentData():
                tip += f"\n{w.currentData()}.*"
            w.setToolTip(tip)

    @staticmethod
    def _grid_snap_key(o):
        """The Redshift menu's key of an output: the snapshot number in a simulation (a folder whose name
        gives none: the folder itself, so two renamed snapdirs never share one '(?)' entry); for a custom
        run the snapshot FILE it read (the sidecar's input_file: the DTFE and PS-DTFE runs of one snapshot
        meet in the Type menu whatever folder they wrote to, and two snapshots run into one folder --
        the Custom tab's default ~/DTFE-output -- stay apart); else the folder (a file opened by hand
        without a sidecar). Cached on the object (a refresh makes new objects)."""
        cached = getattr(o, "_grid_key", None)
        if cached is not None:
            return cached
        if o.sim:
            key = o.snap if o.snap is not None else str(o.dir)
        else:
            f = (G.sidecar(o.root) or {}).get("input_file")
            key = str(Path(f).expanduser().resolve()) if f else str(o.dir)
        o._grid_key = key
        return key

    def _grid_row_sims(self, i: int):
        """Row i's Simulation menu: the simulations in natural order, then the custom snapshots and the
        opened files; '—' leaves the panel blank. Keeps the row's choice, and (a refresh) its exact output."""
        sim, typ = self.grid_table.cellWidget(i, _C_SIM), self.grid_table.cellWidget(i, _C_TYPE)
        keep, keep_root = sim.currentData(), typ.currentData()
        kept = self._by_root.get(keep_root or "")
        if kept is not None:                    # a refresh re-lists an opened file under its simulation or the
            keep = self._group(kept)            # custom snapshots: the row follows its output, not its old group
        outs = [o for _, o in self.outputs]
        sim.blockSignals(True)
        sim.clear()
        sim.addItem("—", GRID_NONE)
        for name in sorted({o.sim for o in outs if o.sim}, key=G.sim_sort_key):
            sim.addItem(name, name)
        for group, text in ((GROUP_CUSTOM, "Custom snapshots"), (GROUP_OPENED, "Opened files")):
            if any(self._group(o) == group for o in outs):
                sim.addItem(text, group)
        j = sim.findData(keep) if keep else 0
        sim.setCurrentIndex(j if j >= 0 else 0)
        sim.blockSignals(False)
        self._grid_row_snaps(i, keep_root=keep_root)

    def _grid_row_snaps(self, i: int, keep_root: str | None = None):
        """Row i's Redshift menu for its simulation: the snapshots by redshift, earliest first (today
        last, the default), or the custom snapshots' and opened files' folders by name. Keeps the row's
        snapshot when the new simulation has it; 'keep_root' (a refresh, a fill) picks that output's."""
        sim, snap = self.grid_table.cellWidget(i, _C_SIM), self.grid_table.cellWidget(i, _C_SNAP)
        grp = sim.currentData()
        kept = self._by_root.get(keep_root or "")
        want = self._grid_snap_key(kept) if kept is not None else snap.currentData()
        outs = [o for _, o in self.outputs if grp and grp != GRID_NONE and self._group(o) == grp]
        snap.blockSignals(True)
        snap.clear()
        first: dict = {}
        for o in outs:
            first.setdefault(self._grid_snap_key(o), o)
        if grp in (GROUP_CUSTOM, GROUP_OPENED):
            keys = sorted(first, key=lambda k: (G.sim_sort_key(Path(k).name), k))
            labels = [Path(k).name for k in keys]
            depth = 1                           # two 'snapdir_099/combined_099.hdf5' read with as many parents as it takes
            while len(set(labels)) < len(labels) and depth < 8:
                depth += 1
                labels = [lab if Counter(labels)[lab] == 1 else "/".join(Path(k).parts[-depth:]) for lab, k in zip(labels, keys)]
            for key, lab in zip(keys, labels):
                snap.addItem(lab if len(set(labels)) == len(labels) else key, key)
                snap.setItemData(snap.count() - 1, key, Qt.ToolTipRole)
        else:                                   # snapshot numbers first, then the folders a number was not read from
            for key in sorted(first, key=lambda k: (isinstance(k, str), k if isinstance(k, int) else 0, str(k))):
                o = first[key]
                z = f"z = {o.redshift:.2f} " if o.redshift is not None else ""
                snap.addItem(z + (f"({key:03d})" if isinstance(key, int) else f"({Path(key).name})"), key)
                snap.setItemData(snap.count() - 1, f"snapshot {key:03d}" if isinstance(key, int) else str(key), Qt.ToolTipRole)
        j = snap.findData(want) if want is not None else -1
        if j < 0:                               # the default: the newest NUMBERED snapshot (today), not an unnumbered folder
            nums = [k for k in range(snap.count()) if isinstance(snap.itemData(k), int)]
            j = nums[-1] if nums else snap.count() - 1
        snap.setCurrentIndex(j)
        snap.setEnabled(snap.count() > 0)
        snap.blockSignals(False)
        self._grid_row_types(i, keep_root=keep_root)

    def _grid_row_types(self, i: int, keep_root: str | None = None):
        """Row i's Type menu: the outputs of its snapshot -- the standard DTFE, the phase-space run, other
        prefixes (grids.type_label; two runs with one label get their folder). Keeps the exact output
        ('keep_root': a refresh, a fill), else the row's remembered choice (_grid_type_pref: the user's
        pick or the fill's) by its root, else its prefix WITH its estimator, else its estimator, else its
        prefix, else the Output menus' preference (ps_output first): a row set to DTFE stays DTFE when its
        redshift changes, and a run of the same name as another estimator's (custom runs in two folders)
        comes back after a snapshot without it. Ends in the field menu, which recomposes once."""
        sim, snap, typ = (self.grid_table.cellWidget(i, c) for c in (_C_SIM, _C_SNAP, _C_TYPE))
        grp, key = sim.currentData(), snap.currentData()
        prev_prefix, prev_kind, prev_root = self._grid_type_pref.get(
            i, (typ.currentData(Qt.UserRole + 1), typ.currentData(Qt.UserRole + 2), typ.currentData()))
        outs = [o for _, o in self.outputs if grp and grp != GRID_NONE and key is not None
                and self._group(o) == grp and self._grid_snap_key(o) == key]
        outs.sort(key=lambda o: G.type_order(o, self._pref_prefix))
        labels = [G.type_label(o) for o in outs]
        dup = Counter(labels)
        if any(c > 1 for c in dup.values()):            # two runs of one name and estimator: their folders tell
            labels = [lab if dup[lab] == 1 else f"{lab} ({o.dir.name})" for lab, o in zip(labels, outs)]
            dup2 = Counter(labels)
            labels = [lab if dup2[lab] == 1 else f"{G.type_label(o)} ({o.dir})" for lab, o in zip(labels, outs)]
        typ.blockSignals(True)
        typ.clear()
        for o, lab in zip(outs, labels):
            typ.addItem(lab, str(o.root))
            k = typ.count() - 1
            typ.setItemData(k, o.prefix, Qt.UserRole + 1)
            typ.setItemData(k, G.estimator_label(o), Qt.UserRole + 2)
            typ.setItemData(k, str(o.root) + ".*", Qt.ToolTipRole)
        j = typ.findData(keep_root) if keep_root else -1
        if j < 0 and prev_root:
            j = typ.findData(prev_root)
        if j < 0 and prev_prefix and prev_kind:
            j = next((k for k, o in enumerate(outs) if o.prefix == prev_prefix and G.estimator_label(o) == prev_kind), -1)
        if j < 0 and prev_kind:
            j = next((k for k, o in enumerate(outs) if G.estimator_label(o) == prev_kind), -1)
        if j < 0 and prev_prefix:
            j = next((k for k, o in enumerate(outs) if o.prefix == prev_prefix), -1)
        if j < 0 and outs:
            j = self._preferred_prefix([o.prefix for o in outs])
        typ.setCurrentIndex(j)
        typ.setEnabled(typ.count() > 1)
        typ.blockSignals(False)
        self._grid_fields_changed(i, keep_field=True)

    def _grid_fields_changed(self, i: int, keep_field: bool = False):
        out, fld = self.grid_table.cellWidget(i, _C_TYPE), self.grid_table.cellWidget(i, _C_FLD)
        o = self._by_root.get(out.currentData() or "")
        keep = fld.currentData() if keep_field else None
        fld.blockSignals(True)
        fld.clear()
        if o is not None:
            for name in o.fields():
                fld.addItem(G.FIELD_LABELS.get(name, name), name)
        j = fld.findData(keep) if keep else -1
        fld.setCurrentIndex(j if j >= 0 else 0)
        fld.setEnabled(o is not None)
        fld.blockSignals(False)
        self._grid_components_changed(i, keep_component=keep_field)

    def _grid_components_changed(self, i: int, keep_component: bool = False):
        out, fld, cmp = (self.grid_table.cellWidget(i, c) for c in (_C_TYPE, _C_FLD, _C_CMP))
        o = self._by_root.get(out.currentData() or "")
        name = fld.currentData()
        keep = cmp.currentText() if keep_component else None
        cmp.blockSignals(True)
        cmp.clear()
        if o is not None and name in o.files:
            cmp.addItems(G.components(name, o.files[name].ncomp, o.dim))
        if keep and cmp.findText(keep) >= 0:
            cmp.setCurrentText(keep)
        cmp.setEnabled(cmp.count() > 1)
        cmp.blockSignals(False)
        self._grid_row_tips(i)
        self._grid_row_edited()

    def _grid_row_edited(self):
        self._grid_update_merge()
        if self._grid_shown:
            self.show_grid(True)

    def _grid_refresh_soon(self):
        """The shown grid again (a slice step, a colour change): a request to the worker, which serves
        only the latest one -- a held arrow key cannot pile up."""
        if self._grid_shown and not self._shut:
            self.show_grid(True)

    def _grid_merge_applicable(self) -> bool:
        """Every panel with an output comes from a box of one size along the shown axes (the merged axes and
        the shared colour bar need that); False when no panel has an output."""
        axis = self.axis.currentIndex()
        boxes = set()
        for pnl in self.grid_panels():
            if pnl is None:
                continue
            o = pnl[0]
            rest = [a for a in range(3) if a != axis] if o.dim == 3 else [0, 1]
            boxes.add((o.dim,) + tuple(round(float(v), 6) for v in np.r_[np.asarray(o.lo)[rest], np.asarray(o.length)[rest]]))
        return len(boxes) == 1

    def _grid_update_merge(self):
        """'merge the axes' is offered only for panels from boxes of one size; merged, the shared range is implied."""
        ok = self._grid_merge_applicable()
        self.grid_merge.setEnabled(ok)
        self.grid_merge.setToolTip(GRID_MERGE_TIP if ok else
                                   GRID_MERGE_TIP + "\n(greyed out: the panels come from boxes of different sizes)")
        merged = ok and self.grid_merge.isChecked()
        self.grid_shared.setEnabled(not merged)
        user = getattr(self, "_grid_shared_user", None)
        if merged and user is None:             # forced on while merged; the user's own choice comes back after
            self._grid_shared_user = self.grid_shared.isChecked()
            self.grid_shared.blockSignals(True)
            self.grid_shared.setChecked(True)
            self.grid_shared.blockSignals(False)
        elif not merged and user is not None:
            self._grid_shared_user = None
            self.grid_shared.blockSignals(True)
            self.grid_shared.setChecked(bool(user))
            self.grid_shared.blockSignals(False)

    def _grid_merge_toggled(self, _on):
        self._grid_update_merge()
        if self._grid_shown:
            self.show_grid(True)

    def grid_row(self, i: int):
        """Panel i as (OutputSet, field, component), or None for an empty panel."""
        out, fld, cmp = (self.grid_table.cellWidget(i, c) for c in (_C_TYPE, _C_FLD, _C_CMP))
        o = self._by_root.get(out.currentData() or "")
        if o is None or fld.currentData() is None:
            return None
        return o, fld.currentData(), cmp.currentText() or "value"

    def grid_set_row(self, i: int, o, field: str | None = None, component: str | None = None):
        """Row i = the output o (None: blank), its simulation, snapshot and type menus set to it, then
        the field and component when given (else the row's own, kept when o has them)."""
        sim, fld, cmp = (self.grid_table.cellWidget(i, c) for c in (_C_SIM, _C_FLD, _C_CMP))
        hold, self._grid_shown_hold = getattr(self, "_grid_shown_hold", False), True
        try:                                    # held: the cascade's recomposes (the old field, the first component) wait
            if o is not None:
                self._grid_type_pref[i] = (o.prefix, G.estimator_label(o), str(o.root))
            else:
                self._grid_type_pref.pop(i, None)
            sim.blockSignals(True)
            j = sim.findData(self._group(o)) if o is not None else 0
            sim.setCurrentIndex(j if j >= 0 else 0)
            sim.blockSignals(False)
            self._grid_row_snaps(i, keep_root=str(o.root) if o is not None else None)
            if field is not None and fld.findData(field) >= 0:
                fld.blockSignals(True)
                fld.setCurrentIndex(fld.findData(field))
                fld.blockSignals(False)
                self._grid_components_changed(i)
            if component is not None and cmp.findText(component) >= 0:
                cmp.blockSignals(True)
                cmp.setCurrentText(component)
                cmp.blockSignals(False)
                self._grid_row_tips(i)
        finally:
            self._grid_shown_hold = hold
        self._grid_row_edited()                 # once, from the final state (nothing under a fill's own hold)

    def grid_panels(self) -> list:
        rows, cols = self.grid_dims()
        return [self.grid_row(i) for i in range(rows * cols)]

    def grid_fill(self, mode: str):
        """Fill the panels from the map: 'map' = this output, field and component in every panel;
        'redshifts' = this field at this simulation's snapshots that have it, the latest ones, oldest
        first; 'fields' = this output's fields in the order of the field list, starting at the map's;
        'types' = this field from every output of this snapshot (DTFE, PS-DTFE, other prefixes)."""
        o, name, comp = self.current, self.field.currentData(), self.component.currentText()
        if o is None or name is None:
            return
        rows, cols = self.grid_dims()
        n = rows * cols
        plan: list = []
        if mode == "redshifts":
            sims = [x for _t, x in self.outputs if self._group(x) == self._group(o) and name in x.fields()]
            if o.sim:                           # one output per snapshot: the map's prefix AND estimator
                same = [x for x in sims if x.prefix == o.prefix and G.estimator_label(x) == G.estimator_label(o)]
            else:                               # custom/opened runs: the series run into THIS folder with this estimator
                same = [x for x in sims if str(x.dir) == str(o.dir) and G.estimator_label(x) == G.estimator_label(o)]
            if len(same) > 1:                   # (a custom run's prefix is its name: X in two folders is one snapshot twice)
                sims = same
            sims.sort(key=lambda x: (self._snap_key(x) if x.sim else 0, str(x.root)))
            plan = [(x, name, comp) for x in sims[-n:]]
        elif mode == "fields":
            fields = list(o.fields())
            k = fields.index(name) if name in fields else 0
            fields = fields[k:] + fields[:k]
            plan = [(o, f, None) for f in fields[:n]]
        elif mode == "types":                   # this snapshot's outputs: DTFE, PS-DTFE, other prefixes
            outs = [x for _t, x in self.outputs if self._group(x) == self._group(o)
                    and self._grid_snap_key(x) == self._grid_snap_key(o) and name in x.fields()]
            order = lambda x: G.type_order(x, self._pref_prefix)      # noqa: E731
            outs.sort(key=order)
            # the map's own run first, then the other estimator's best (its default prefix, else the preferred
            # one), then the rest: a 1x2 used to show 'output' beside ps_mw (an A/B run) while the map was on
            # ps_output -- the production run and the map's own run both dropped
            pick = [x for x in outs if self._same(x, o)]
            pick += [x for x in outs if G.estimator_label(x) != G.estimator_label(o)][:1]
            roots = {str(x.root) for x in pick}
            pick += [x for x in outs if str(x.root) not in roots]
            plan = [(x, name, comp) for x in sorted(pick[:n], key=order)]
        else:
            plan = [(o, name, comp)] * n
        with_signals = self._grid_shown
        hold, self._grid_shown_hold = getattr(self, "_grid_shown_hold", False), True
        try:
            for i in range(n):
                if i < len(plan):
                    self.grid_set_row(i, *plan[i])
                else:
                    self.grid_set_row(i, None)
        finally:
            self._grid_shown_hold = hold
        self._grid_update_merge()
        if with_signals:
            self.show_grid(True)

    def _grid_cmap(self, name: str) -> str:
        if name == self.field.currentData():
            return self.cmap.currentText()
        return "RdBu_r" if name in G.DIVERGING else "tab10" if name in ("tweb", "vweb") else \
            "magma" if name in ("density", "dispersion") else "viridis"

    def _grid_log(self, name: str) -> bool:
        return self.log.isChecked() if name == self.field.currentData() else name in G.LOG_DEFAULT

    def grid_request(self) -> dict | None:
        """Everything the grid needs, read from the widgets on the GUI thread: the panels with their colour
        rules, the slice (this map's axis and relative position), the range clip, the smoothing, the
        options. None when no panel has an output."""
        panels = self.grid_panels()
        if not any(panels) or self.current is None:
            return None
        cur = self.current
        rows, cols = self.grid_dims()
        return {"panels": [None if p is None else {"output": p[0], "field": p[1], "component": p[2],
                                                    "cmap": self._grid_cmap(p[1]), "log": self._grid_log(p[1])}
                           for p in panels],
                "rows": rows, "cols": cols, "axis": self.axis.currentIndex(),
                "rel": 0.5 if cur.n <= 1 or cur.dim == 2 else self.index.value() / (cur.n - 1),
                "index": self.index.value(), "clip": self.clip.value(), "smooth": self.smooth.value(),
                "shared": self.grid_shared.isChecked(),
                "merged": self.grid_merge.isChecked() and self._grid_merge_applicable(),
                "position": self.position.text() if cur.dim == 3 else ""}

    def show_grid(self, on: bool):
        """The toggle: the panels as tiles in place of the map (the worker colours them, like the single
        map's slice), or the map back. Shown already, a new request replaces the tiles when it arrives."""
        if getattr(self, "_grid_shown_hold", False):
            return
        if not on:
            if self._grid_shown:
                self._leave_grid()
                self.render()
            return
        if self.current is None:
            self.grid_toggle.blockSignals(True)
            self.grid_toggle.setChecked(False)
            self.grid_toggle.blockSignals(False)
            return
        if not self._grid_shown and not any(self.grid_panels()):
            self.grid_fill("map")               # turned on with nothing chosen: the map in every panel; shown and
        req = self.grid_request()               # every panel blanked by hand: the warning below, not a refill
        if req is None:
            self.grid_state.setText("⚠ no panel has an output (use a Fill button, or choose outputs in the table)")
            if self._grid_shown:
                self._leave_grid()
                self.render()
            else:
                self.grid_toggle.blockSignals(True)
                self.grid_toggle.setChecked(False)
                self.grid_toggle.blockSignals(False)
            return
        if not self._grid_shown:
            self._grid_shown = True
            self.grid_toggle.blockSignals(True)
            self.grid_toggle.setChecked(True)
            self.grid_toggle.blockSignals(False)
            self.grid_toggle.setText("Back to the map")
            self.view.streams.setVisible(False)     # the picture gets the height; a click lists no streams here
            self.view.readout.setText(" ")
            self._compare_sync()
        if self._grid_busy:
            self._grid_pending = req                # only the latest request is served
            return
        self._grid_busy = True
        self._threads()
        self._to_render_grid.emit(req)

    def _grid_rendered(self, res: dict):
        self._grid_busy = False
        if self._grid_pending is not None:
            req, self._grid_pending = self._grid_pending, None
            self._grid_busy = True
            self._to_render_grid.emit(req)
        if not self._grid_shown or self._shut:
            return
        req = res["request"]
        self.view.canvas.setStyleSheet("")
        self.view.canvas.marker = None
        self.view.canvas.arrows = None
        self.view.canvas.labels = res["labels"]
        self.view.canvas.axes = None                # tiles of several frames: the saved figure has the axes
        self.view.canvas.set_rgb(np.ascontiguousarray(res["rgb"]))
        bar = res["bar"]
        if bar is not None:                         # one quantity: its bar below, as the single map has
            cmap, vrange, label, log = bar
            self.view.colorbar.set(cmap, vrange, label, log)
            self.view.colorbar.setVisible(True)
        else:                                       # several: the saved figure carries one bar per quantity
            self.view.colorbar.setVisible(False)
        self._grid_index = req["index"]
        n, rows, cols = res["shown"], req["rows"], req["cols"]
        pos = req["position"]
        self.view.title.setText(f"<b>Figure grid</b> {rows} × {cols} — {n} panel{'s' if n != 1 else ''}"
                                + (f" — {pos}" if pos else ""))
        self.grid_state.setText(f"{n} panel{'s' if n != 1 else ''}" + (f" at {pos}" if pos else "")
                                + "; ← → step the slice for every panel, the slider and the colours apply too; "
                                  "Save figure… makes the full figure (axes, colour bars) at 200 dpi, PNG or PDF")
        self.changed.emit()

    def _grid_failed(self, msg: str):
        self.grid_state.setText(f"⚠ could not compose the grid: {msg}")
        if self._grid_pending is not None:          # a newer request waits (a panel was changed meanwhile): its
            req, self._grid_pending = self._grid_pending, None     # result decides whether the grid stays
            self._grid_busy = True
            self._to_render_grid.emit(req)
            return
        self._grid_busy = False
        if self._grid_shown:                        # a panel that cannot be read: the map, not a stale picture
            self._leave_grid()
            self.render()

    def _leave_grid(self):
        self._grid_shown = False
        self._grid_pending = None
        self.grid_toggle.blockSignals(True)
        self.grid_toggle.setChecked(False)
        self.grid_toggle.blockSignals(False)
        self.grid_toggle.setText("Show the grid")
        self.view.canvas.labels = None
        self.view.colorbar.setVisible(True)
        self.view.streams.setVisible(True)
        self._compare_sync()
        self.changed.emit()                     # the Explore menu's Figure Grid text follows

    def grid_figure(self):
        """The full matplotlib figure of the current panels (axes in Mpc, titles or inside labels, colour
        bars), composed on this thread at 200 dpi: what Save figure… writes. Returns (Figure, panel count)."""
        req = self.grid_request()
        if req is None:
            raise ValueError("no panel has an output (use a Fill button, or choose outputs in the table)")
        return grid_figure(req, grid_items(req))

    def save_grid(self, path) -> str:
        """Write the figure of the current panels as PNG or PDF; the path written."""
        fig, _n = self.grid_figure()
        fig.savefig(str(path), dpi=200, bbox_inches="tight")
        self.status.emit(f"saved {path}")
        return str(path)

    def _save_grid_dialog(self):
        if self.current is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Save the figure grid", str(Path.home() / "figure_grid.png"),
                                           "PNG (*.png);;PDF (*.pdf)")
        if not f:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.save_grid(f)
        except Exception as e:
            self.grid_state.setText(f"⚠ could not save the figure: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    # ---------------------------------------------------------------- query server
    def _browse_cache(self):
        d = QFileDialog.getExistingDirectory(self, "Tessellation cache folder (a local disk with room)",
                                             self.cache_edit.text() or str(Path.home()))
        if d:
            self.cache_edit.setText(d)
            self._cache_user_set = True

    def server_kwargs(self) -> dict | None:
        o = self.current
        if o is None:
            return None
        kw = G.server_settings(o, self._snapshots.get(str(o.root)))
        if kw is None:
            return None
        cache = Path(self.cache_edit.text().strip() or default_cache_dir(str(o.dir))).expanduser()
        ps = bool(kw.get("phase_space", True))   # the standard binary has no streams: it refuses --per-stream
        kw.update(per_stream=ps, lagrangian_positions=ps, caustics=ps, velocity_gradient=True,
                  tessellation_cache=str(cache), verbose=1, progress=True)
        if self.server_partition:
            kw["partition"] = int(self.server_partition)
        if self.resident.value() > 0:           # also for the composite the binary splits a big snapshot into
            kw["resident"] = int(self.resident.value())
        return kw

    def _choose_snapshot(self):
        o = self.current
        if o is None:
            return
        f, _ = QFileDialog.getOpenFileName(self, "The snapshot of this output", str(o.dir),
                                           "Snapshots (*.hdf5 *.h5 *.hdf);;All files (*)")
        if f:
            self._snapshots[str(o.root)] = f
            self.status.emit(f"{Path(f).name}: the snapshot of {o.title} (the server and the exact zoom read it)")
            self._refresh_server_button()

    def server_phase(self) -> str:
        return "starting" if self._server_starting else "running" if self._server_running else "stopped"

    def _refresh_server_button(self):
        phase = self.server_phase()
        if phase == "starting":
            self.server_btn.setText("Cancel")
            self.server_btn.setEnabled(True)
        elif phase == "running":
            self.server_btn.setText("Stop query server")
            self.server_btn.setEnabled(True)
        else:
            self.server_btn.setText("Start query server")
            ok = self.current is not None and G.server_settings(self.current,
                                                                 self._snapshots.get(str(self.current.root))) is not None
            self.server_btn.setEnabled(ok)
            if self.current is not None and not ok:
                self.server_state.setText(NO_SNAPSHOT_TEXT)
            elif self.server_state.text() == NO_SNAPSHOT_TEXT:      # a snapshot chosen or restored since
                self.server_state.setText("not running")
        self.changed.emit()

    def _set_server_state(self, text):
        self.server_state.setText(text)
        self._server_running = text.startswith("running")
        self._refresh_server_button()

    def _server_button(self):
        if self._server_starting:
            self._server.cancel()
            return
        if self._server_running:
            self._to_stop.emit()
            self._server_running = False
            self._refresh_server_button()
            return
        self.start_server()

    def start_server(self, confirm: bool = True):
        kw = self.server_kwargs()
        if kw is None or self._shut:
            return
        cache = Path(kw["tessellation_cache"])
        cost = G.server_cost(kw["snapshot"], kw.get("phase_space", True))
        if confirm and cost and cost["split"]:
            import shutil as _sh
            try:
                free = _sh.disk_usage(cache if cache.exists() else cache.parent).free / 1e9
            except OSError:
                free = float("nan")
            text = (f"{cost['particles']:,} particles: one tessellation would need ~{cost['single_gb']:.0f} GB, "
                    f"more than the {cost['budget_gb']:.0f} GB this machine allows a run. The server will be "
                    f"built as a composite of partitions kept on disk:\n\n"
                    f"  cache  ~{cost['cache_gb']:.0f} GB in {cache}  ({free:.0f} GB free)\n"
                    f"  first start  builds every partition, about as long as a grid run of this snapshot; "
                    f"later starts reuse the cache and take seconds.\n\nStart it?")
            if QMessageBox.question(self, "Start the query server", text) != QMessageBox.Yes:
                return
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.server_state.setText(f"cannot create the cache folder: {e}")
            return
        import tempfile
        if self._server_log is not None:            # the previous server's log: one file per start piled up
            self._server_log.unlink(missing_ok=True)
        fd, log = tempfile.mkstemp(prefix="dtfe-explore-", suffix=".log")
        os.close(fd)                                # the Estimator opens it by name; the descriptor leaked per start
        self._server_log = Path(log)
        kw["log_path"] = str(self._server_log)
        self._server_output = self.current
        # point zooms remembered from another server are dropped (a new server, perhaps other options);
        # exact zooms do not depend on the server and stay
        from collections import OrderedDict
        self._zoom_memo = OrderedDict((k, v) for k, v in self._zoom_memo.items() if k[-1])
        self._server_starting = True
        self._build_start = time.time()
        self.server_state.setText("building the tessellation…")
        self._refresh_server_button()
        self._build_timer.start()
        self._threads()
        self._to_start.emit(kw)

    def _poll_build(self):
        """Progress of a build from the server's log: the composite's '[partitions k/N]' lines."""
        if not self._server_starting or self._server_log is None:
            self._build_timer.stop()
            return
        try:
            text = self._server_log.read_text(errors="replace")[-20000:]
        except OSError:
            return
        import re
        m = re.findall(r"\[partitions (\d+)/(\d+)\]", text)
        el = time.time() - self._build_start
        msg = f"building the tessellation… {int(el)} s"
        if m:
            msg = f"building partition tessellations: {m[-1][0]}/{m[-1][1]} ({int(el)} s)"
        elif "AUTO-TUNE (serve)" in text:
            msg = f"too big for one tessellation: building a composite… {int(el)} s"
        self.server_state.setText(msg)

    def _server_started(self, info, region):
        self._server_starting = False
        self._build_timer.stop()
        o = self._server_output
        if o is not None and o.box is None and region is not None:
            # the output did not know its box: the server's is authoritative (the same coordinates)
            o.box = (tuple(region[:, 0]), tuple(region[:, 1]))
            if self._same(o, self.current):
                self.current.box = o.box            # (a refresh may have replaced the object meanwhile)
                self._show_output(self.current)
        self._set_server_state(f"running: {info}. Click the map.")
        self.status.emit(f"query server ready ({info})")

    def _server_failed(self, msg):
        if self._server_starting:
            self._server_starting = False
            self._build_timer.stop()
            self._set_server_state("failed: " + msg)
        elif self._zoom_pending:
            self._zoom_pending = False
            self._end_zoom_progress()
            self.zoom_state.setText("⚠ " + msg)
            self.changed.emit()
        else:
            self.view.answer_title.setText("⚠ " + msg)

    def _click(self, r, c):
        if self._grid_shown:                 # the grid is a picture: no cells under the mouse
            return
        pt = self._point(r, c)
        if pt is None:
            return
        self.view.canvas.marker = (r, c)
        self.view.canvas._redraw()
        x, value, (row, col) = pt
        o = self._req["output"]
        if not self._server_running or not self._same(self._server_output, o):
            what = f"hi-res point ({row}, {col})" if self._req.get("pts") else "zoom point" if self.is_zoomed() else \
                f"cell {tuple(int(v) for v in o.plane_point(self._req['axis'], self._req['index'], row, col))}"
            self.view.answer_title.setText(f"<b>{what}</b>: {value:.5g}. "
                                           "Start the query server (left) to see the streams at this point.")
            self.view.streams.setRowCount(0)
            return
        self.view.answer_title.setText("asking…")
        self._to_query.emit(np.ascontiguousarray(x[:o.dim]).reshape(1, o.dim))

    # ---------------------------------------------------------------- zoom
    def _dragged(self, r0: int, c0: int, r1: int, c1: int):
        """A rectangle dragged on the map: evaluate that region through the server on a res x res
        plane of points (inside a zoom: a deeper zoom of the same slice)."""
        if self._grid_shown:                 # the grid is a picture: no cells under the mouse
            return
        if self._req is not None and self._req.get("pts"):
            self.status.emit("the hi-res slice is already point-evaluated at its own resolution: zoom on the grid slice")
            return
        g = self._geometry()
        if g is None or self._shut:
            return
        axis, index, rest, lo, hi, n0, n1 = g
        o = self._req["output"]
        rows = sorted((c0, c1))                       # image columns = plane rows (first remaining axis)
        cols = sorted((n1 - 1 - r1, n1 - 1 - r0))     # image rows (downward) = plane columns, reversed
        a0 = (lo[0] + rows[0] / n0 * (hi[0] - lo[0]), lo[0] + (rows[1] + 1) / n0 * (hi[0] - lo[0]))
        a1 = (lo[1] + cols[0] / n1 * (hi[1] - lo[1]), lo[1] + (cols[1] + 1) / n1 * (hi[1] - lo[1]))
        res = int(self.zoom_res.value())
        depth = (self._zoom["depth"] + 1) if self.is_zoomed() else 1
        if self.zoom_exact.isChecked():
            self._exact_zoom(o, axis, index, rest, a0, a1, res, depth)
            return
        if not self._server_running or not self._same(self._server_output, o):
            self.view.answer_title.setText("Start the query server (left) to zoom: the region is then evaluated "
                                           "again, point by point, at any resolution (or tick 'exact').")
            return
        coords = [None, None, None]
        coords[axis] = np.array([o.lo[axis] + (index + 0.5) * o.length[axis] / o.n])
        coords[rest[0]] = a0[0] + (np.arange(res) + 0.5) / res * (a0[1] - a0[0])
        coords[rest[1]] = a1[0] + (np.arange(res) + 0.5) / res * (a1[1] - a1[0])
        if o.dim == 2:                                  # a 2D server: the plane's own two axes
            coords = coords[:2]
        req = {"output": o, "axis": axis, "index": index, "rest": rest, "a0": a0, "a1": a1, "res": res,
               "coords": coords, "depth": depth}
        hit = self._memo_get(self._memo_key(o, axis, index, a0, a1, res, False))
        if hit is not None:                             # the same region was evaluated a moment ago
            self._zoomed(dict(req, memo=True, seconds=0.0), hit[1])
            return
        self._zoom_pending = True
        self.zoom_state.setText(f"zoom: evaluating {res * res:,} points…")
        self.view.canvas.marker = None
        self._begin_zoom_progress("point")
        self._threads()
        self._to_zoom.emit(req)
        self.changed.emit()

    def exact_zoom_command(self, o, axis: int, index: int, rest, a0, a1, res: int):
        """The run behind an exact zoom: (command, output root, grid dims, (a0, a1)), or (None, note,
        None, None) when it cannot be built (exact_zoom_plan has the details)."""
        p = self.exact_zoom_plan(o, axis, index, rest, a0, a1, res)
        if p["cmd"] is None:
            return None, p["note"], None, None
        return p["cmd"], p["root"], p["grid"], p["edges"]

    def _exact_partition(self, o, kw: dict, side: dict, cache: Path) -> tuple[tuple | None, dict]:
        """The split an exact zoom of a phase-space run uses, and this snapshot's cached partition sets:
        the running query server's (its cache is then complete), else the most complete set in the
        cache, else the run's own split, else none (the binary's auto-tuner)."""
        d = o.dim
        sets = {t: v for t, v in G.cached_partition_sets(cache, kw["snapshot"], kw.get("lagrangian_input"),
                                                         bool(kw.get("periodic", True))).items() if len(t) == d}
        if self._server_running and self._same(self._server_output, o) and self._server.n_partitions > 1:
            n = round(self._server.n_partitions ** (1.0 / d))
            if n ** d == self._server.n_partitions:
                return (n,) * d, sets
        if sets:
            best = max(sets, key=lambda t: (len(sets[t]) / float(np.prod(t)), len(sets[t])))
            return best, sets
        if int(side.get("partition") or 0) > 0:
            n = int(side["partition"])
            return (n,) * d, sets
        return None, sets

    def exact_zoom_plan(self, o, axis: int, index: int, rest, a0, a1, res: int) -> dict:
        """Everything about the run behind an exact zoom of output 'o': the command ('cmd', None with a
        'note' when it cannot be built), its output 'root' and 'grid' dims, the region as the run will
        grid it ('edges' (a0, a1) and 'window' (3, 2) in Mpc), and for a phase-space run the tessellation
        'cache_dir', the 'partition' split and the cached partition 'sets' (for the estimate). The region
        = the dragged rectangle, one slice cell thick; its grid res x res x 1 (in the axis' place).
        Standard DTFE cuts the region from the box (regionMpc, periodic or not) and integrates the
        interpolant exactly. PS-DTFE tessellates the whole box as the original run did (periodic or not)
        and deposits only into a WINDOW of a virtual full grid whose cell is the rectangle's / res
        (--ps-window): the window's cells equal the full grid's, so the zoom is what a run at that
        resolution would show there. It reuses the tessellations the query server keeps on disk (the same
        cache folder and split), and the binary skips the partitions whose occupancy maps never reach
        the window. The window is snapped to that grid's cell edges, so the returned edges can differ
        from the drag by up to half a zoom cell. A 2D output: the same on its plane, with no slice axis
        (a res x res grid, a window and a region of two axes)."""
        import tempfile
        plan = {"cmd": None, "note": "", "root": None, "grid": None, "edges": None, "window": None,
                "ps": False, "cache_dir": None, "partition": None, "sets": {}, "snapshot": None,
                "lagrangian": None}
        kw = G.server_settings(o, self._snapshots.get(str(o.root)))
        if kw is None:
            plan["note"] = "no snapshot known for this output (its input file is not beside it)"
            return plan
        if o.box is None:
            plan["note"] = "this output's box is unknown, so the region cannot be placed"
            return plan
        ps = bool(kw.get("phase_space", True))
        periodic = bool(kw.get("periodic", True))
        side = G.sidecar(o.root) or {}
        est = "ps" if ps else "dtfe"
        precision = side.get("precision", "single") if side.get("precision") in ("single", "double") else "single"
        flat = o.dim == 2               # a 2D run: only the plane's two axes (the slice axis is z)
        binary = rs.REPO_ROOT / rs.binary_name(est, precision, o.dim)
        if not binary.is_file():
            plan["note"] = f"{binary.name} is not built"
            return plan
        mpc = float(kw["mpc_unit"]) if kw.get("mpc_unit") is not None else None   # None: the binary's default, as the server
        region = [[0.0, 0.0] for _ in range(3)]
        region[axis] = [o.lo[axis] + index * o.length[axis] / o.n, o.lo[axis] + (index + 1) * o.length[axis] / o.n]
        region[rest[0]] = [float(a0[0]), float(a0[1])]
        region[rest[1]] = [float(a1[0]), float(a1[1])]
        grid = [res, res, res]
        grid[axis] = 1
        if flat:
            grid = [res, res]
        out_dir = Path(tempfile.mkdtemp(prefix="dtfe-zoom-"))
        root = out_dir / "zoom"
        cmd = [str(binary), str(kw["snapshot"]), str(root), "--input", str(int(kw.get("input_type", 105)))]
        if mpc is not None:
            cmd += ["--MpcUnit", f"{mpc:.10g}"]
        a0n, a1n = (float(a0[0]), float(a0[1])), (float(a1[0]), float(a1[1]))
        window_mpc = [list(r) for r in region]
        if ps:
            # the virtual full grid: the run's own cells along the slice axis (the window is one of
            # them), and along the two plane axes cells of the rectangle's size / res, rounded to
            # tile the box; the window = res x res of them starting at the rectangle's corner
            n = [int(o.n)] * 3
            window = [[0.0, 0.0] for _ in range(3)]
            aligned = {}
            for a in rest:
                lo_a, hi_a = region[a]
                length = float(o.length[a])
                dx = (hi_a - lo_a) / res
                if not (dx > 0.0 and length > 0.0):
                    plan["note"] = "the dragged region has no extent"
                    shutil.rmtree(out_dir, ignore_errors=True)
                    return plan
                n_a = max(res, min(WINDOW_GRID_MAX, int(round(length / dx))))
                dxp = length / n_a
                c0 = max(0, min(n_a - res, int(round((lo_a - float(o.lo[a])) / dxp))))
                n[a] = n_a
                # the edges half a cell INSIDE the window: the program snaps them outward to the cell
                # edges, so the window is exactly cells [c0, c0+res) whatever float rounding does
                window[a] = [float(o.lo[a]) + (c0 + 0.5) * dxp, float(o.lo[a]) + (c0 + res - 0.5) * dxp]
                aligned[a] = (float(o.lo[a]) + c0 * dxp, float(o.lo[a]) + (c0 + res) * dxp)
                window_mpc[a] = list(aligned[a])
            window[axis] = [o.lo[axis] + (index + 0.25) * o.length[axis] / o.n,
                            o.lo[axis] + (index + 0.75) * o.length[axis] / o.n]
            a0n, a1n = aligned[rest[0]], aligned[rest[1]]
            if flat:
                n, window, window_mpc = n[:2], window[:2], window_mpc[:2]
            cmd += ["--grid", *[str(v) for v in n]]
            if periodic:
                cmd.append("--periodic")
            cmd += list(kw.get("options", []))                          # the custom snapshot's own --box (and alpha)
            corners = [f"{float(v):.10g}" for pair in window for v in pair]   # Mpc, as --ps-window takes
            cmd += ["--ps-window", *corners, "--field", "density_a", "velocity_a", "dispersion_a", "gradient_a",
                    "--ps-exact-deposit", "--ps-caustics"]
            if side.get("volume_weighted", True):
                cmd.append("--ps-volume-weighted")
            # chart-independent tet masses: as the custom run recorded it, or, for a TNG output (no
            # sidecar: the batch script's default, on, when the converted ICs are the Lagrangian input)
            if (bool(side.get("vertex_mass")) if side else bool(kw.get("lagrangian_input"))):
                cmd.append("--ps-vertex-mass")
            if kw.get("lagrangian_input"):
                cmd += ["--lagrangianInput", str(kw["lagrangian_input"])]
            if (not periodic and float(side.get("alpha_shape", 3.0)) != 3.0
                    and "--ps-alpha-shape" not in cmd):
                cmd += ["--ps-alpha-shape", f"{float(side['alpha_shape']):g}"]
            if rs.gpu_built("ps", precision, o.dim):
                cmd.append("--ps-gpu")
            # the query server's tessellations on disk: the same cache folder and split, so a cached
            # partition is loaded instead of rebuilt and one that never reaches the window is skipped;
            # the concurrency is the binary's (its auto-tuner sizes it by the window and the free memory)
            cache = Path(self.cache_edit.text().strip() or default_cache_dir(str(o.dir))).expanduser()
            part, sets = self._exact_partition(o, kw, side, cache)
            cmd += ["--tessellation-cache", str(cache)]
            if part is not None:
                cmd += ["--partition", *[str(int(v)) for v in part]]
            cmd += ["--verbose", "2"]                                  # the progress lines the worker follows
            plan.update(ps=True, cache_dir=str(cache), partition=part, sets=sets,
                        snapshot=str(kw["snapshot"]), lagrangian=kw.get("lagrangian_input"))
        else:
            if flat:
                region, window_mpc = region[:2], window_mpc[:2]
            cmd += ["--grid", *[str(v) for v in grid]]
            corners = [f"{float(v):.10g}" for pair in region for v in pair]     # Mpc, as --regionMpc takes
            if periodic:
                cmd.append("--periodic")
            cmd += list(kw.get("options", []))                          # the custom snapshot's own --box
            cmd += ["--regionMpc", *corners, "--field", "density_a", "velocity_a", "gradient_a", "--exact-average"]
            if side.get("scalar_dataset"):
                cmd += ["scalar_a", "scalarGradient_a", "--scalar-dataset", str(side["scalar_dataset"])]
            if rs.gpu_built("dtfe", precision, o.dim):
                cmd.append("--gpu")
            plan["snapshot"] = str(kw["snapshot"])
        plan.update(cmd=cmd, root=str(root), grid=grid, edges=(a0n, a1n), window=window_mpc)
        return plan

    def exact_zoom_estimate(self, plan: dict) -> dict | None:
        """How long the exact zoom of a phase-space run will take (None for standard DTFE, which
        triangulates only the region): {'seconds', 'n_total', 'n_skip', 'n_load', 'n_build',
        'new_gb', 'particles', 'cache_dir'}. From the cache on disk -- which partitions are cached and,
        by their occupancy maps, which reach the region -- and the measured rates above (EST_*)."""
        if not plan.get("ps") or not plan.get("snapshot"):
            return None
        if len(plan.get("grid") or ()) == 2:
            return None                 # a 2D run: small (n^2 cells, triangles), and its maps hold n^2 bits
        cost = G.server_cost(plan["snapshot"], True)
        if cost is None:
            return None
        n_part = float(cost["particles"])
        size = 0
        for f in (plan["snapshot"], plan.get("lagrangian")):
            try:
                size += os.path.getsize(f) if f else 0
            except OSError:
                pass
        read_s = size / EST_READ_BPS
        part, sets = plan.get("partition"), plan.get("sets") or {}
        recorded = G.globals_recorded(plan.get("cache_dir"), plan["snapshot"], plan.get("lagrangian"))
        n_skip = n_load = 0
        load_s = dep_vert = 0.0
        if part is not None:
            n_total = int(part[0] * part[1] * part[2])
            entries = sets.get(tuple(part), [])
            for e in entries:
                nv = e["bytes"] * EST_VERTICES_PER_BYTE
                if e["occ"] is not None:
                    occ = G.read_occupancy(e["occ"])
                    if occ is not None:
                        if e["region"] and not G.occupancy_touches(occ[1], e["region"], e["periodic"], plan["window"]):
                            n_skip += 1
                            continue
                        nv = float(occ[0]) or nv
                n_load += 1
                load_s += e["bytes"] / EST_LOAD_BPS / (EST_CHUNKED_LOAD_SPEEDUP if G.tess_chunked(e["tess"]) else 1.0)
                dep_vert += nv
            n_build = max(0, n_total - len(entries))
            build_vert = n_build * n_part / n_total * 4.1          # a padded partition: ~4x its share
        else:
            n_total, n_build = 1, 1
            build_vert = n_part * (4.1 if cost["split"] else 1.73)
        read_skipped = part is not None and recorded and n_build == 0
        if read_skipped:
            read_s = 0.0                                # the binary never reads the snapshot
        work = (n_build * n_part * EST_SELECT_S_PER_PARTICLE + load_s
                + (dep_vert + build_vert) * EST_DEPOSIT_S_PER_VERTEX + build_vert * EST_BUILD_S_PER_VERTEX
                + build_vert / EST_VERTICES_PER_BYTE / EST_SAVE_BPS)
        return {"seconds": read_s + work / EST_CONCURRENCY, "n_total": n_total, "n_skip": n_skip,
                "n_load": n_load, "n_build": n_build, "new_gb": build_vert / EST_VERTICES_PER_BYTE / 1e9,
                "particles": int(n_part), "cache_dir": plan.get("cache_dir"), "split": part is not None,
                "read_skipped": read_skipped}

    def _confirm_exact(self, est: dict) -> bool:
        """Ask before a long exact zoom; the estimate and what it is made of."""
        lines = [f"This exact zoom runs PS-DTFE over the whole snapshot again ({est['particles']:,} particles) "
                 f"and deposits only into the region.", "",
                 f"Estimated time: about {_fmt_duration(est['seconds'])}, if the program can work on "
                 f"{EST_CONCURRENCY} partitions at a time (it decides from the free memory: fewer while the "
                 f"query server holds a large snapshot, so then it takes longer).", ""]
        if est["split"]:
            reach = est["n_total"] - est["n_skip"]
            lines.append(f"  {reach} of {est['n_total']} partitions reach the region; "
                         f"the other {est['n_skip']} are skipped.")
            if est["n_load"]:
                lines.append(f"  {est['n_load']} are loaded from the tessellation cache"
                             + (", and the snapshot itself is not read (its particle totals are recorded there)."
                                if est.get("read_skipped") else "."))
            if est["n_build"]:
                lines.append(f"  {est['n_build']} have to be built first, and are kept in the cache "
                             f"(about {est['new_gb']:.0f} GB more) for later zooms and the query server.")
        else:
            lines.append(f"  No tessellation of this snapshot is cached yet: the run builds it (about "
                         f"{est['new_gb']:.0f} GB, kept in the cache for later zooms and the query server).")
        if est["n_build"]:
            lines += ["", "Starting the query server first builds the cache once; after that an exact zoom "
                          "only loads the partitions that reach the region."]
        lines += ["", f"Cache folder: {est['cache_dir']}", "", "Continue?"]
        text = "\n".join(lines)
        if self.confirm_exact is not None:
            return bool(self.confirm_exact(text))
        return QMessageBox.question(self, "Exact zoom", text) == QMessageBox.Yes

    def _exact_zoom(self, o, axis, index, rest, a0, a1, res, depth):
        if self._exact_running:
            self.zoom_state.setText("an exact zoom is still running: Cancel it to start another")
            self.changed.emit()
            return
        plan = self.exact_zoom_plan(o, axis, index, rest, a0, a1, res)
        if plan["cmd"] is None:
            self.zoom_state.setText("⚠ " + plan["note"])
            self.changed.emit()
            return
        hit = self._memo_get(self._memo_key(o, axis, index, plan["edges"][0], plan["edges"][1], res, True))
        if hit is not None:                             # the same exact zoom ran a moment ago: no run, no question
            import shutil
            shutil.rmtree(Path(plan["root"]).parent, ignore_errors=True)
            self._zoomed(dict(hit[0], depth=depth, memo=True, seconds=0.0), hit[1])
            return
        est = self.exact_zoom_estimate(plan)
        if est is not None and est["seconds"] > EXACT_CONFIRM_SECONDS and not self._confirm_exact(est):
            import shutil
            shutil.rmtree(Path(plan["root"]).parent, ignore_errors=True)
            self.zoom_state.setText("exact zoom not started")
            self.changed.emit()
            return
        a0, a1 = plan["edges"]                          # the region as the run will grid it
        req = {"output": o, "axis": axis, "index": index, "rest": rest, "a0": a0, "a1": a1, "res": res,
               "depth": depth, "exact": True, "cmd": plan["cmd"], "root": plan["root"], "grid": plan["grid"],
               "cache_dir": plan.get("cache_dir")}

        self._zoom_pending = True
        self._exact_running = True
        self._exact_worker.cancel_requested = False
        self.zoom_cancel_btn.setEnabled(True)
        self._begin_zoom_progress("exact")
        about = f", about {_fmt_duration(est['seconds'])}" if est is not None else ""
        self.zoom_state.setText(f"exact zoom: running {Path(plan['cmd'][0]).name} for the region "
                                f"({res} × {res} cells{about})…")
        self.view.canvas.marker = None
        self._threads()
        self._to_exact.emit(req)
        self.changed.emit()

    def _exact_progress(self, text: str):
        if self._exact_running and not self._shut:
            self.zoom_state.setText("exact zoom: " + text)
            m = re.search(r"partition (\d+) of (\d+)", text)
            if m:
                self.zoom_progress.setRange(0, int(m.group(2)))
                self.zoom_progress.setValue(int(m.group(1)))
            self.changed.emit()

    def _exact_done(self):
        self._exact_running = False
        self.zoom_cancel_btn.setEnabled(False)

    def _exact_failed(self, msg: str):
        self._zoom_pending = False
        self._end_zoom_progress()
        self._exact_done()
        self.zoom_state.setText("exact zoom cancelled" if msg == ExactZoomWorker.CANCELLED else "⚠ " + msg)
        self.changed.emit()

    def cancel_exact_zoom(self):
        """Stop the exact zoom that is running (its temporary output is removed)."""
        if self._exact_running:
            self._exact_worker.cancel()
            self.zoom_state.setText("exact zoom: cancelling…")
            self.changed.emit()

    def _zoomed(self, req: dict, f):
        if req.get("exact"):
            self._exact_done()
        if not req.get("memo"):
            sec = self._end_zoom_progress()
            req = dict(req, seconds=sec, summary=self._zoom_summary if not req.get("exact") else None)
            self._memo_put(self._memo_key(req["output"], req["axis"], req["index"], req["a0"], req["a1"],
                                          req["res"], bool(req.get("exact"))), req, f)
        else:
            self.zoom_time.setText("from memory")
        self._zoom_pending = False
        if self._shut or self._req is None or not self._same(req["output"], self._req["output"]):
            return
        if self._req.get("pts"):                 # a hi-res slice went on while the zoom ran: it is memoised, the
            return                               # map stays the slice (drawn in the zoom's frame it misplaced clicks)
        if self._zoom is not None and self._zoom_fields is not None and req["depth"] > self._zoom["depth"]:
            self._zoom_history.append((self._zoom, self._zoom_fields))   # Back returns here
        self._zoom, self._zoom_fields = req, f
        self.index.blockSignals(True)
        self.index.setValue(req["index"])             # the zoom belongs to this slice
        self.index.blockSignals(False)
        self._render_zoom()

    def _zoom_plane(self, name: str, component: str):
        """The zoomed field as a (res, res) plane in the slice's own layout (rows = first remaining
        axis), or None when the point evaluation has no such field. A 2D output's fields are already
        the plane (no slice axis to drop), its vectors have two components."""
        f, axis = self._zoom_fields, self._zoom["axis"]
        if self._zoom["output"].dim == 2:
            sq = lambda a: np.asarray(a, dtype=np.float64)
        else:
            sq = lambda a: np.squeeze(np.asarray(a, dtype=np.float64), axis=axis)
        if name == "density":
            return sq(f.density)
        if name == "streams":
            return None if f.streams is None else sq(f.streams)
        if name == "scalar":                    # an exact zoom of a run with a scalar dataset; the server has none
            sc = getattr(f, "scalar", None)
            return None if sc is None else sq(sc)
        if name == "caustic" and f.caustic is not None:
            return sq(((np.asarray(f.caustic) & 3) == 3).astype(np.float64))
        if name == "caustic_class" and f.caustic is not None:
            return sq(f.caustic)
        if name == "velocity":
            v = sq(f.velocity)
            if len(component) == 1 and component in "xyz"[:v.shape[-1]]:
                return v[..., "xyz".index(component)]
            return np.sqrt((v ** 2).sum(axis=-1))
        if name == "dispersion":
            if f.dispersion is None:            # an exact zoom may hold only the scalar dispersion
                sig = getattr(f, "sigma", None)
                return None if sig is None else sq(sig)
            d = sq(f.dispersion)                # packed xx, xy, xz, yy, yz, zz (2D: xx, xy, yy)
            dim = 2 if d.shape[-1] == 3 else 3
            return np.sqrt(np.maximum(sum(d[..., i] for i in G.sym_diag(dim)), 0.0))
        if name == "dispersion_tensor":
            if f.dispersion is None:
                return None
            d = sq(f.dispersion)
            comps = G.components("dispersion_tensor", d.shape[-1], 2 if d.shape[-1] == 3 else 3)
            return d[..., comps.index(component) if component in comps else 0]
        if name in ("gradient", "divergence", "vorticity", "shear") and f.velocity_gradient is not None:
            g = sq(f.velocity_gradient)                   # [.., d, j] = dv_j / dx_d
            dim = g.shape[-1]
            trace = sum(g[..., i, i] for i in range(dim))
            if name == "divergence":
                return trace
            if name == "gradient":                        # the grid's names: 'xy' = dv_x/dy
                if len(component) == 2 and all(c in "xyz"[:dim] for c in component):
                    j, d = "xyz".index(component[0]), "xyz".index(component[1])
                    return g[..., d, j]
                return trace
            if name == "vorticity":                       # the curl, as the map shows it (grids.py slice)
                if dim == 2:
                    return g[..., 0, 1] - g[..., 1, 0]    # its z: dv_y/dx - dv_x/dy
                w = np.stack([g[..., 1, 2] - g[..., 2, 1], g[..., 2, 0] - g[..., 0, 2], g[..., 0, 1] - g[..., 1, 0]], -1)
                if component in ("x", "y", "z"):
                    return w[..., "xyz".index(component)]
                return np.sqrt((w ** 2).sum(axis=-1))
            # the shear: the traceless symmetric part; the grid's components '1', '2', ... are its upper
            # triangle without the last diagonal entry, row by row (xx, xy, xz, yy, yz / 2D: xx, xy)
            sym = 0.5 * (g + np.swapaxes(g, -1, -2))
            for i in range(dim):
                sym[..., i, i] -= trace / dim
            pairs = [(a, b) for a in range(dim - 1) for b in range(a, dim)]
            k = int(component) - 1 if component.isdigit() else -1
            named = {"xx": (0, 0), "xy": (0, 1), "xz": (0, 2), "yy": (1, 1), "yz": (1, 2), "zz": (2, 2)}
            i, j = pairs[k] if 0 <= k < len(pairs) else named.get(component, (0, 0))
            return sym[..., i, j]
        return None

    def _render_zoom(self):
        req = self.request() or self._req       # zoomed: request() leaves the comparison's colours out
        self._req, z = req, self._zoom
        note = ""
        plane = self._zoom_plane(req["field"], req["component"])
        if plane is None:
            plane, note = self._zoom_plane("density", ""), " · this field has no point values: density shown"
        plane = np.ascontiguousarray(np.asarray(plane, dtype=np.float64))
        img = np.ascontiguousarray(plane.T[::-1])
        rgb, vrange = G.colorize(img, req["cmap"], log=req["log"], clip=(req["clip"], 100 - req["clip"]),
                                 symmetric=req["symmetric"], vrange=req.get("vrange"))
        self._plane, self._arrows, self._planes = plane, None, None
        self.view.canvas.setStyleSheet("")
        self.view.canvas.marker = None
        self.view.canvas.arrows = None
        self.view.canvas.axes = self._canvas_axes()
        self.view.canvas.set_rgb(np.ascontiguousarray(rgb))
        self._compare_sync()
        if self.compare.currentData() is not None:
            note += " · the comparison applies to the slice map: the zoom shows A"
        label = G.FIELD_LABELS.get(req["field"], req["field"])
        comp = "" if req["component"] in ("value", "") else f" [{req['component']}]"
        self.view.colorbar.set(req["cmap"], vrange, ("log₁₀ " if req["log"] else "") + label + comp, req["log"])
        o, rest = req["output"], z["rest"]
        unit = "Mpc" if o.box is not None else "cells"
        where = (f"{AXES[rest[0]]} {z['a0'][0]:.2f}–{z['a0'][1]:.2f}, {AXES[rest[1]]} {z['a1'][0]:.2f}–{z['a1'][1]:.2f} {unit}")
        kind = "exact zoom" if z.get("exact") else "zoom"
        self.view.title.setText(f"<b>{label}{comp}</b> — {o.title} — {kind} {z['res']}² on {where} — {self.position.text()}")
        how = "cells (exact deposit, one slice cell thick)" if z.get("exact") else "points"
        took = ""
        if z.get("memo"):
            took = "; from memory"
        elif z.get("seconds") is not None:
            took = f"; {z['seconds']:.1f} s"
            sm = z.get("summary")
            if sm and sm.get("total", 1) > 1:
                took += (f" ({sm['partitions']} of {sm['total']} partitions"
                         + (f", {sm['loaded']} loaded from disk in {sm['load_s']:.1f} s" if sm["loaded"] else ", all in memory")
                         + ")")
        self.zoom_state.setText(f"{kind} level {z['depth']}: {z['res']} × {z['res']} {how} on {where}{took}; "
                                f"drag again to go deeper" + note)
        self.zoom_out_btn.setEnabled(True)
        self.zoom_back_btn.setEnabled(True)
        self.changed.emit()

    # ---------------------------------------------------------------- zoom progress, history, memory
    @staticmethod
    def _memo_key(o, axis, index, a0, a1, res, exact: bool) -> tuple:
        """A zoom remembered for this output AS IT IS ON DISK: a re-run under the same name (other settings, a
        rebuilt binary) rewrites its grids and sidecar, and the memory must not answer for it any more."""
        r = lambda t: tuple(round(float(v), 9) for v in t)
        stamp = []
        for p in [ff.path for ff in o.files.values()] + [Path(str(o.root) + ".gui.json")]:
            try:
                st = p.stat()
                stamp.append((p.name, st.st_mtime_ns, st.st_size))
            except OSError:
                pass
        return (str(o.root), int(axis), int(index), r(a0), r(a1), int(res), tuple(sorted(stamp)), bool(exact))   # exact LAST:
                                                                                     # start_server keeps k[-1]

    def _memo_get(self, key):
        hit = self._zoom_memo.get(key)
        if hit is not None:
            self._zoom_memo.move_to_end(key)
        return hit

    def _memo_put(self, key, req: dict, f):
        self._zoom_memo[key] = (req, f)
        self._zoom_memo.move_to_end(key)
        while len(self._zoom_memo) > 1 and sum(_fields_bytes(v[1]) for v in self._zoom_memo.values()) > ZOOM_MEMO_BYTES:
            self._zoom_memo.popitem(last=False)

    def _begin_zoom_progress(self, kind: str):
        """A zoom starts: the bar (busy until the first progress line), the elapsed time, and for a
        point zoom the position in the server's log from which its progress lines are read."""
        self._zoom_t0 = time.monotonic()
        self._zoom_kind = kind
        self._zoom_summary = None
        self._zoom_log_pos = 0
        if kind == "point" and self._server_log is not None:
            try:
                self._zoom_log_pos = self._server_log.stat().st_size
            except OSError:
                pass
        self.zoom_progress.setRange(0, 0)
        self.zoom_progress.setVisible(True)
        self.zoom_time.setText("0.0 s")
        self._zoom_tick.start()

    def _end_zoom_progress(self) -> float | None:
        """The zoom has ended (result, failure or cancel): the bar goes, the time stays. Its duration."""
        self._zoom_tick.stop()
        self.zoom_progress.setVisible(False)
        if self._zoom_t0 is None:
            return None
        sec = time.monotonic() - self._zoom_t0
        self._zoom_t0 = None
        if self._zoom_kind == "point":
            self._read_request_progress()            # the final '[request done]' line
        self.zoom_time.setText(f"{sec:.1f} s")
        return sec

    def _read_request_progress(self):
        """The server's progress lines for the running point zoom: the bar over its partitions."""
        if self._server_log is None:
            return
        try:
            with open(self._server_log, "rb") as fh:
                fh.seek(self._zoom_log_pos)
                text = fh.read().decode(errors="replace")
        except OSError:
            return
        last, status = None, None
        for line in text.splitlines():
            m = _REQ_LINE.search(line)
            if m:
                k, n = int(m.group(1)), int(m.group(2))
                done = m.group(3) is None
                last = (k if done else k - 1, n)
                if not done:
                    status = (f"loading partition {k} of {n} from disk" if m.group(3) == "loading from the cache"
                              else f"indexing partition {k} of {n} (once, for faster zooms there)"
                              if m.group(3) == "building its cell index"
                              else f"evaluating partition {k} of {n}")
            d = _REQ_DONE.search(line)
            if d:
                self._zoom_summary = {"points": int(d.group(1)), "partitions": int(d.group(2)),
                                      "total": int(d.group(3)), "loaded": int(d.group(4)),
                                      "load_s": float(d.group(5))}
        if last is not None and last[1] > 1:
            self.zoom_progress.setRange(0, last[1])
            self.zoom_progress.setValue(last[0])
        if status and self._zoom_pending and not self._exact_running:
            self.zoom_state.setText(f"zoom: {status}…")

    def _tick_zoom(self):
        if self._zoom_t0 is None:
            self._zoom_tick.stop()
            return
        self.zoom_time.setText(f"{time.monotonic() - self._zoom_t0:.1f} s")
        if self._zoom_kind == "point":
            self._read_request_progress()

    def zoom_back(self):
        """One zoom level up, from memory; at the first level, back to the slice."""
        if not self._zoom_history:
            self.zoom_out()
            return
        req, f = self._zoom_history.pop()
        self._zoom, self._zoom_fields = req, f
        self._render_zoom()

    def _clear_zoom(self):
        self._zoom = self._zoom_fields = None
        self._zoom_pending = False
        self._zoom_history.clear()
        self.zoom_out_btn.setEnabled(False)
        self.zoom_back_btn.setEnabled(False)
        self.zoom_state.setText("")
        self._compare_sync()

    def zoom_out(self):
        """Back to the slice from disk (the zoom's evaluation is dropped)."""
        if self._zoom is None:
            return
        self._clear_zoom()
        self.render()
        self.changed.emit()

    def _answered(self, pts, f):
        x = pts[0]
        n = int(f.streams[0])
        caustic = ""
        if f.caustic is not None and int(f.caustic[0]) & 3 == 3:
            caustic = " · on a fold caustic"
        dim = len(x)                                    # 2 for a 2D server's points
        sigma = float(np.sqrt(max(sum(f.dispersion[0][i] for i in G.sym_diag(dim)), 0.0)))
        vec = lambda a, fmt: "(" + ", ".join(format(float(c), fmt) for c in a) + ")"
        self.view.answer_title.setText(
            f"<b>{vec(x, '.2f')} Mpc</b>: {n} stream{'s' if n != 1 else ''} · "
            f"ρ/ρ̄ = {f.density[0]:.4g} · |v̄| = {np.linalg.norm(f.velocity[0]):.1f} km/s · σ = {sigma:.1f} km/s"
            + caustic)
        t = self.view.streams
        t.setRowCount(0)
        if f.offsets is None:
            return
        sl = f.stream_slice(0)
        for k, i in enumerate(range(sl.start, sl.stop)):
            v = f.stream_velocity[i]
            q = f.stream_lagrangian[i] if f.stream_lagrangian is not None else None
            sval = f.stream_scalar[i] if f.stream_scalar is not None else None
            cells = [str(k + 1), f"{f.stream_density[i]:.4g}", vec(v, ".0f"),
                     f"{np.linalg.norm(v):.0f}",
                     "—" if q is None else vec(q, ".2f"),
                     "—" if sval is None else f"{float(np.ravel(sval)[0]):.4g}"]
            t.insertRow(k)
            for c, text in enumerate(cells):
                t.setItem(k, c, QTableWidgetItem(text))

    def shutdown(self):
        """Stop the server and the threads (the window is closing). Idempotent."""
        self._shut = True
        if self._server_log is not None:
            try:
                self._server_log.unlink(missing_ok=True)
            except OSError:
                pass
        self._zoom_tick.stop()
        self._server.kill()                            # a build OR a running request: the thread must come back
        self._exact_worker.cancel()                    # a running exact zoom would hold the thread
        if self._slice_thread is None:
            self._server.stop()
            return
        self._to_stop.emit()
        for th in (self._slice_thread, self._server_thread, self._exact_thread):
            th.quit()
            for _ in range(6):                         # up to 30 s: a thread still inside a request is never dropped
                if th.wait(5000):
                    break
        self._slice_thread = self._server_thread = self._exact_thread = None
