"""The Explore tab's point-query server: ServerWorker, a dtfelib.Estimator on the snapshot behind an output living
on its own thread (a composite of partitions, out of core, when one tessellation would not fit in memory;
explore.py re-exports it), and ServerMixin, the part of ExploreControls that starts, follows and stops it and
answers a click on the map with the streams at that point."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem

import explore as _ex
import grids as G


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
        so and stop: the GUI's button then offers a new start instead of 'Stop' on a dead server (where
        every further click would fail the same way)."""
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
        and the thread can be joined (never dropped while running)."""
        self.cancel()
        p = getattr(self.est, "_proc", None)
        if p is not None and p.poll() is None:
            p.kill()


class ServerMixin:
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
        cache = Path(self.cache_edit.text().strip() or _ex.default_cache_dir(str(o.dir))).expanduser()
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
                self.server_state.setText(_ex.NO_SNAPSHOT_TEXT)
            elif self.server_state.text() == _ex.NO_SNAPSHOT_TEXT:      # a snapshot chosen or restored since
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
        need = None if not cost else (cost["budget_gb"] if cost["split"] else cost["single_gb"])
        if not self.heavy_ok("Start the query server", f"A query server on {Path(kw['snapshot']).name} "
                             f"({cost['particles']:,} particles)" if cost else "A query server", need):
            self.server_state.setText("not started: a DTFE run is going")
            return
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
        if self._same(o, self.current):
            self._carry_zoom(server_started=True)   # a zoom window kept from another output: evaluated here now

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
        k = self._void_pick(r, c)
        if k is not None:                    # the inspector: the whole entry of the void clicked
            self._void_picked = (self._void_cache_key(self.current), k)
            self.void_info.setText(G.void_info(self._void_result(), k))
            self.void_goto.setEnabled(not self.is_zoomed())
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
