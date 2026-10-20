"""The Explore tab's zoom: a rectangle dragged on the map evaluated again, point by point through the query server,
or by an exact run of the program itself on the region (ExactZoomWorker on its own thread, load_exact_planes for
the grid it writes; explore.py re-exports them); ZoomMixin, the part of ExploreControls that plans, estimates and
starts a zoom, carries its window to other outputs and redshifts, draws it, and keeps the earlier levels and recent
zooms in memory."""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QMessageBox

import explore as _ex
import grids as G
import runspec as rs
from dtfelib.estimator import STANDARD_PADDING


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
                line = _ex._ANSI.sub("", raw).strip()
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
    it an exact zoom at z=2 would show velocities 1.73x the slice map's."""
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
                                            # (float64 of a 9-component gradient at 4096^2 is 2.4 GB)
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
    # f.caustic is the CLASS mask, as a point evaluation's '.pts_caustic' (the zoom's fold map masks both parity
    # bits, its class map shows it whole); the 0/1 '.caustic' grid would show no fold cell at all and a 0/1 class
    # map. The zoom's run is new: its parity bits are the geometric overlap, so the fold is exact; a '.caustic'
    # alone stands for both bits
    if "caustic_class" in planes:
        f.caustic = np.rint(planes["caustic_class"]).astype(np.int64)
    elif "caustic" in planes:
        f.caustic = np.where(np.rint(planes["caustic"]) > 0, 3, 0).astype(np.int64)
    if "gradient" in planes:   # the grid's [j*D+i] = dv_j/dx_i -> the point evaluation's [d, j] = dv_j/dx_d
        f.velocity_gradient = planes["gradient"].reshape(shape + (dim, dim)).swapaxes(-1, -2)
    if f.density is None:
        raise RuntimeError("the exact zoom wrote no density grid")
    return f


class ZoomMixin:
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
            self._drop_kept_zoom()              # a window of the user's own replaces one carried from another output
            self._exact_zoom(o, axis, index, rest, a0, a1, res, depth)
            return
        if not self._server_running or not self._same(self._server_output, o):
            self.view.answer_title.setText("Start the query server (left) to zoom: the region is then evaluated "
                                           "again, point by point, at any resolution (or tick 'exact').")
            return                              # (no zoom made: a window carried from another output stays kept)
        self._drop_kept_zoom()
        self._point_zoom(o, axis, index, rest, a0, a1, res, depth)

    def _point_zoom(self, o, axis, index, rest, a0, a1, res, depth):
        """The region a0 x a1 (Mpc, or cells without a box) of slice 'index' through the running query server, on
        a res x res plane of points; from memory when it was evaluated a moment ago."""
        coords = [None, None, None]
        coords[axis] = np.array([o.lo[axis] + (index + 0.5) * o.length[axis] / o.n])
        coords[rest[0]] = a0[0] + (np.arange(res) + 0.5) / res * (a0[1] - a0[0])
        coords[rest[1]] = a1[0] + (np.arange(res) + 0.5) / res * (a1[1] - a1[0])
        if o.dim == 2:                                  # a 2D server: the plane's own two axes
            coords = coords[:2]
        req = {"output": o, "axis": axis, "index": index, "rest": rest, "a0": a0, "a1": a1, "res": res,
               "coords": coords, "depth": depth, "stamp": self._file_stamp(o)}
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

    def _carry_zoom(self, server_started: bool = False):
        """The zoom window kept from another output or redshift (self._keep['zoom']) on the output shown
        now, the same Mpc window at the same resolution on this output's kept slice: from memory when this output's
        zoom of it was made before (switching back and forth is then instant), else by the query server when it runs
        on this output. Never by an exact run: a switch neither starts one nor asks about one (an exact zoom is a
        drag with 'exact' ticked). Otherwise the map stays the slice, the window stays kept for the next output (or
        for the server, once started here: server_started), and the zoom line and the status line say why."""
        m, o = self._keep, self.current
        w = (m or {}).get("zoom")
        if not w or o is None or self._shut or self.is_zoomed():
            return
        axis, index = self.axis.currentIndex(), self.index.value()
        (u, v), (a0, a1) = w["rest"], (w["a0"], w["a1"])
        where = f"{_ex.AXES[u]} {a0[0]:.2f}–{a0[1]:.2f}, {_ex.AXES[v]} {a1[0]:.2f}–{a1[1]:.2f} {w['unit']}"
        why = None
        if o.dim != w["dim"]:
            why = ("a 2D output shows its whole plane; the window comes back on the next 3D output" if o.dim == 2
                   else "the window was made on a 2D output's plane")
        elif w["unit"] != ("Mpc" if o.box is not None else "cells"):
            why = "this output's box is not known in the window's units"
        elif axis != w["axis"]:
            why = "the slice axis is another one here"
        elif self.pts_combo.currentData():
            why = "the hi-res slice is shown (choose 'off: the grid slice' and this output again)"
        if why is None:
            for exact in (w["exact"], not w["exact"]):
                hit = self._memo_get(self._memo_key(o, axis, index, a0, a1, w["res"], exact))
                if hit is not None:
                    self._req = self.request() or self._req     # this output's (its slice may wait behind another)
                    self._zoomed(dict(hit[0], depth=1, memo=True, seconds=0.0), hit[1])
                    return
            # never an exact run: choosing an output or stepping through redshifts must not start
            # a PS-DTFE run over the whole snapshot or ask about one; with 'exact' ticked a drag of the window does
            if self._server_running and self._same(self._server_output, o):
                self._point_zoom(o, axis, index, list(w["rest"]), a0, a1, w["res"], 1)
                return
            if server_started:
                return
            why = ("no zoom of it is in memory for this output and no query server runs on it (start one here and "
                   "the window is evaluated)")
            if self.zoom_exact.isChecked():
                why += "; choosing an output never starts an exact zoom: drag the window again for one"
        text = f"{_ex.KEPT_ZOOM_TEXT}{where} kept, not shown on this output: {why}"
        self.zoom_state.setText(text)
        self.status.emit(text)
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
        out_dir = Path(tempfile.mkdtemp(prefix=rs.ZOOM_DIR_PREFIX))     # runspec._kind tells it from a Custom run by this
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
                n_a = max(res, min(_ex.WINDOW_GRID_MAX, int(round(length / dx))))
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
            cache = Path(self.cache_edit.text().strip() or _ex.default_cache_dir(str(o.dir))).expanduser()
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
            # the padding around the region: the binary's own '--padding 5' leaves wrong values in the
            # cells near the region's faces (as near a box face: the Delaunay tetrahedra there reach the padded
            # set's hull); the query server's 15 (dtfelib.estimator.STANDARD_PADDING, run_dtfe.sh's default) unless
            # the snapshot's own options carry one. Placed before '--regionMpc': its corners are followed by '--field'
            if not any(str(v) in ("--padding", "--paddingMpc") for v in kw.get("options", [])):
                cmd += ["--padding", f"{STANDARD_PADDING:g}"]
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
        read_s = size / _ex.EST_READ_BPS
        part, sets = plan.get("partition"), plan.get("sets") or {}
        recorded = G.globals_recorded(plan.get("cache_dir"), plan["snapshot"], plan.get("lagrangian"))
        n_skip = n_load = 0
        load_s = dep_vert = 0.0
        if part is not None:
            n_total = int(part[0] * part[1] * part[2])
            entries = sets.get(tuple(part), [])
            for e in entries:
                nv = e["bytes"] * _ex.EST_VERTICES_PER_BYTE
                if e["occ"] is not None:
                    occ = G.read_occupancy(e["occ"])
                    if occ is not None:
                        if e["region"] and not G.occupancy_touches(occ[1], e["region"], e["periodic"], plan["window"]):
                            n_skip += 1
                            continue
                        nv = float(occ[0]) or nv
                n_load += 1
                load_s += e["bytes"] / _ex.EST_LOAD_BPS / (_ex.EST_CHUNKED_LOAD_SPEEDUP if G.tess_chunked(e["tess"]) else 1.0)
                dep_vert += nv
            n_build = max(0, n_total - len(entries))
            build_vert = n_build * n_part / n_total * 4.1          # a padded partition: ~4x its share
        else:
            n_total, n_build = 1, 1
            build_vert = n_part * (4.1 if cost["split"] else 1.73)
        read_skipped = part is not None and recorded and n_build == 0
        if read_skipped:
            read_s = 0.0                                # the binary never reads the snapshot
        work = (n_build * n_part * _ex.EST_SELECT_S_PER_PARTICLE + load_s
                + (dep_vert + build_vert) * _ex.EST_DEPOSIT_S_PER_VERTEX + build_vert * _ex.EST_BUILD_S_PER_VERTEX
                + build_vert / _ex.EST_VERTICES_PER_BYTE / _ex.EST_SAVE_BPS)
        return {"seconds": read_s + work / _ex.EST_CONCURRENCY, "n_total": n_total, "n_skip": n_skip,
                "n_load": n_load, "n_build": n_build, "new_gb": build_vert / _ex.EST_VERTICES_PER_BYTE / 1e9,
                "particles": int(n_part), "cache_dir": plan.get("cache_dir"), "split": part is not None,
                "read_skipped": read_skipped}

    def _confirm_exact(self, est: dict) -> bool:
        """Ask before a long exact zoom; the estimate and what it is made of."""
        lines = [f"This exact zoom runs PS-DTFE over the whole snapshot again ({est['particles']:,} particles) "
                 f"and deposits only into the region.", "",
                 f"Estimated time: about {_fmt_duration(est['seconds'])}, if the program can work on "
                 f"{_ex.EST_CONCURRENCY} partitions at a time (it decides from the free memory: fewer while the "
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
        cost = G.server_cost(plan["snapshot"], True) if plan.get("snapshot") else None
        if not self.heavy_ok("Exact zoom", "An exact zoom runs PS-DTFE over the whole snapshot again; it",
                             None if not cost else min(cost["single_gb"], cost["budget_gb"])):
            import shutil
            shutil.rmtree(Path(plan["root"]).parent, ignore_errors=True)
            self.zoom_state.setText("exact zoom not started: a DTFE run is going")
            self.changed.emit()
            return
        if est is not None and est["seconds"] > _ex.EXACT_CONFIRM_SECONDS and not self._confirm_exact(est):
            import shutil
            shutil.rmtree(Path(plan["root"]).parent, ignore_errors=True)
            self.zoom_state.setText("exact zoom not started")
            self.changed.emit()
            return
        a0, a1 = plan["edges"]                          # the region as the run will grid it
        req = {"output": o, "axis": axis, "index": index, "rest": rest, "a0": a0, "a1": a1, "res": res,
               "depth": depth, "exact": True, "cmd": plan["cmd"], "root": plan["root"], "grid": plan["grid"],
               "cache_dir": plan.get("cache_dir"), "stamp": self._file_stamp(o)}

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
        if req.get("exact") and not req.get("memo"):    # from memory: no run ended (one may be going for another output)
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
            return                               # map stays the slice (drawn in the zoom's frame it would misplace clicks)
        if (not self._same(req["output"], self.current) or req["axis"] != self.axis.currentIndex()
                or req["index"] != self.index.value()):
            # the slice changed while the zoom was evaluated: drawn, it would take the new slice's axes, scale bar
            # and clicked positions (render() saw no zoom yet). It is memoised; the map stays the slice
            self.zoom_state.setText("the zoom finished after the slice changed, so it is not shown: zoom again on "
                                    "this slice")
            self.changed.emit()
            return
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
        self.view.canvas.circles = self._void_circles()
        self.view.canvas.dashed = self._void_circles("_r_eff")
        self.view.canvas.axes = self._canvas_axes()
        self.view.canvas.set_rgb(np.ascontiguousarray(rgb))
        self._compare_sync()
        if self.compare.currentData() is not None:
            note += " · the comparison applies to the slice map: the zoom shows A"
        # a zoom is a point evaluation (or an exact run on the region): a phase-space output's velocity derivatives are
        # there the stream average, whatever the choice; standard DTFE's the containing tetrahedron's
        o = req["output"]
        ps = G.estimator_label(o) == "PS-DTFE"
        fam = "" if req["field"] not in G.GRADIENT_FIELDS else "stream" if ps else "mean"
        label = G.field_label(o, req["field"], fam)
        self.ps_gradient_note.setText("")
        self._out_form_map.setRowVisible(self.ps_gradient_note, False)
        if fam == "stream" and req.get("ps_gradient") == "mean-field":
            note += " · its velocity derivatives are the stream average (a point evaluation has no mean field)"
        comp = "" if req["component"] in ("value", "") else f" [{req['component']}]"
        self.view.colorbar.set(req["cmap"], vrange, ("log₁₀ " if req["log"] else "") + label + comp, req["log"])
        rest = z["rest"]
        kind = "exact zoom" if z.get("exact") else "zoom"

        def where(units=None):                  # the window and the slice, in Mpc or a saved figure's unit
            (u0, u1), (v0, v1), unit = z["a0"], z["a1"], "Mpc" if o.box is not None else "cells"
            if units is not None:
                (u0, unit), (u1, _), (v0, _), (v1, _) = (units(rest[0], u0), units(rest[0], u1),
                                                         units(rest[1], v0), units(rest[1], v1))
            return (f"{_ex.AXES[rest[0]]} {u0:.2f}–{u1:.2f}, {_ex.AXES[rest[1]]} {v0:.2f}–{v1:.2f} {unit} — "
                    + self._position_text(req, units))
        self._title_make = lambda units=None: f"<b>{label}{comp}</b> — {o.title} — {kind} {z['res']}² on {where(units)}"
        self.view.title.setText(self._title_make())
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
        self.zoom_state.setText(f"{kind} level {z['depth']}: {z['res']} × {z['res']} {how} on {where()}{took}; "
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
        return (str(o.root), int(axis), int(index), r(a0), r(a1), int(res), _ex.ExploreControls._file_stamp(o),
                bool(exact))                                                         # exact LAST: start_server keeps k[-1]

    @staticmethod
    def _file_stamp(o) -> tuple:
        """The output's files as they are on disk (name, mtime, size; the sidecar too): the memory's key, and
        what a zoom on screen checks before it stays beside the slice controls (render())."""
        stamp = []
        for p in [ff.path for ff in o.files.values()] + [Path(str(o.root) + ".gui.json")]:
            try:
                st = p.stat()
                stamp.append((p.name, st.st_mtime_ns, st.st_size))
            except OSError:
                pass
        return tuple(sorted(stamp))

    def _memo_get(self, key):
        hit = self._zoom_memo.get(key)
        if hit is not None:
            self._zoom_memo.move_to_end(key)
        return hit

    def _memo_put(self, key, req: dict, f):
        self._zoom_memo[key] = (req, f)
        self._zoom_memo.move_to_end(key)
        while len(self._zoom_memo) > 1 and sum(_fields_bytes(v[1]) for v in self._zoom_memo.values()) > _ex.ZOOM_MEMO_BYTES:
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
            m = _ex._REQ_LINE.search(line)
            if m:
                k, n = int(m.group(1)), int(m.group(2))
                done = m.group(3) is None
                last = (k if done else k - 1, n)
                if not done:
                    status = (f"loading partition {k} of {n} from disk" if m.group(3) == "loading from the cache"
                              else f"indexing partition {k} of {n} (once, for faster zooms there)"
                              if m.group(3) == "building its cell index"
                              else f"evaluating partition {k} of {n}")
            d = _ex._REQ_DONE.search(line)
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
        """Back to the slice from disk (the zoom's evaluation is dropped), and the whole box on the next output too
        (a zoom window kept across outputs goes)."""
        self._drop_kept_zoom()
        if self._zoom is None:
            return
        self._clear_zoom()
        self.render()
        self.changed.emit()
