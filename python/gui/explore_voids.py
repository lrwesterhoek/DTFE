"""The Explore tab's void outlines: VoidsMixin, the part of ExploreControls that finds a snapshot's voids (the
analysis pipeline's catalogue, through tools/void_outlines.py in a helper process, one at a time, cached per output
and smoothing), draws them over the map and the figure grid's panels, and inspects the void under the mouse or
clicked (its entry, a jump to the slice through its centre)."""

from __future__ import annotations

import json

import numpy as np
from PySide6.QtCore import QProcess, QProcessEnvironment

import explore as _ex
import grids as G
import runspec as rs


class VoidsMixin:
    # ---------------------------------------------------------------- void outlines
    @staticmethod
    def voids_applicable(o) -> str:
        """'' when the void outlines can be drawn over this output, else why not (the checkbox's tip): the
        pipeline's catalogue needs a simulation snapshot's full-box 3D grids with a density."""
        if o is None:
            return "no output shown"
        if not o.sim or o.snap is None:
            return "only for a simulation snapshot's grids (the analysis pipeline finds the voids per snapshot)"
        if o.dim != 3:
            return "only for 3D grids"
        if o.box is None or "density" not in o.files:
            return "needs the density grid and the snapshot's box"
        return ""

    def _void_cache_key(self, o, sigma: float | None = None):
        """The catalogue's key: the files' root, their grid, the density file's stamp (a rerun rewrites it) and the
        smoothing asked for (0 = production; each sigma is another catalogue)."""
        try:
            stamp = o.files["density"].path.stat().st_mtime_ns
        except (KeyError, OSError):
            stamp = None
        return str(o.root), o.n, stamp, round(self.void_sigma.value() if sigma is None else float(sigma), 6)

    def _void_reff_toggled(self, *_):
        self._voids_redraw()
        if self._grid_shown and self.voids.isChecked():
            self.show_grid(True)

    def _void_sigma_changed(self, *_):
        """Another smoothing: its own catalogue (cached by key, so going back costs nothing)."""
        self._void_picked = None
        self.void_info.setText("")
        self.void_goto.setEnabled(False)
        if self.voids.isChecked() and self.current is not None and not self.voids_applicable(self.current):
            self._voids_request(self.current)
        self._voids_redraw()
        if self._grid_shown and self.voids.isChecked():
            self.show_grid(True)

    def _voids_sync(self):
        """The checkbox follows the output: enabled where the outlines apply, else greyed out saying why."""
        why = self.voids_applicable(self.current)
        self.voids.setEnabled(not why)
        self.voids.setToolTip(_ex.VOID_TIP if not why else f"Void outlines: {why}.")
        if self._void_picked is not None and (why or self._void_picked[0] != self._void_cache_key(self.current)):
            self._void_picked = None                # another output: the inspector's void is not on it
            self.void_info.setText("")
            self.void_goto.setEnabled(False)
        if not why and self.voids.isChecked():
            self._voids_request(self.current)

    def _voids_toggled(self, on: bool):
        if on:
            for key in [k for k, v in self._void_cache.items() if "error" in v]:
                del self._void_cache[key]           # ticked again: try again (the grid's panels too)
            if self.current is not None and not self.voids_applicable(self.current):
                self._voids_request(self.current)
        else:
            self._void_queue.clear()
            self._void_picked = None
            self.void_info.setText("")
            self.void_goto.setEnabled(False)
        self._voids_redraw()
        if self._grid_shown:
            self.show_grid(True)                    # the panels' outlines follow the tick

    def _voids_request(self, o, queued: bool = False):
        """Find the voids of this output in a helper process (the pipeline's catalogue: cached, else about a
        minute), unless they are known or already being found. 'queued' (the figure grid's panels): wait for the
        search running instead of stopping it."""
        key = self._void_cache_key(o)
        if key in self._void_cache or (self._void_proc is not None and self._void_key == key):
            return
        if self._void_proc is not None:
            if queued:
                if all(self._void_cache_key(q) != key for q in self._void_queue):
                    self._void_queue.append(o)
                return
            if self._void_out is not None:          # a grid panel's search stopped for the map's: it comes back
                self._void_queue.insert(0, self._void_out)
        self._voids_stop()                          # another output's search: this one is wanted now
        n = int(o.n or 0)
        if not self.heavy_ok("Find the voids", f"Finding the voids of {o.title} builds its catalogue when it is "
                             f"not cached yet; that", 80.0 * n ** 3 / 1e9 if n else None):
            self._void_cache[key] = {"error": "not started while a DTFE run is going (tick again to retry)"}
            self.status.emit("void outlines not started: a DTFE run is going")
            self._voids_next()
            return
        root = o.dir.parent.parent                  # <data root>/<sim>/snapdir_NNN, or the family folder: both work
        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("DTFE_SIM", o.sim)
        env.insert("DTFE_DATA_ROOT", str(root))
        env.remove("DTFE_SIGMA_MPC")                # the smoothing is the --sigma-mpc below, never the shell's
        env.remove("CLICOLOR_FORCE")
        proc.setProcessEnvironment(env)
        proc.finished.connect(lambda code, _status, k=key, pr=proc: self._voids_done(k, pr, code))
        proc.errorOccurred.connect(lambda err, k=key, pr=proc: self._voids_start_failed(k, pr, err))
        self._void_proc, self._void_key, self._void_out = proc, key, o
        method = "ps" if G.estimator_label(o) == "PS-DTFE" else "dtfe"
        args = [str(_ex.VOID_TOOL), "--sim", o.sim, "--snap", str(int(o.snap)), "--method", method, "--data-root", str(root)]
        if o.prefix != ("ps_output" if method == "ps" else "output"):
            args += ["--prefix", o.prefix]          # the production set: no prefix, so the analysis scripts' cache is shared
        if key[-1] > 0:
            args += ["--sigma-mpc", repr(key[-1])]  # a common physical sigma: its own cache namespace
        proc.start(rs.PYTHON, args + (["--cache-dir", str(_ex.VOID_CACHE_DIR)] if _ex.VOID_CACHE_DIR else []))
        self.status.emit(f"finding the voids of {o.title} (the first time per snapshot: about a minute)")

    def _voids_done(self, key, proc, code: int):
        if proc is not self._void_proc:             # stopped: a newer search, or the window closing
            proc.deleteLater()
            return
        self._void_proc = self._void_key = self._void_out = None
        out = bytes(proc.readAllStandardOutput()).decode(errors="replace").strip().splitlines()
        err = bytes(proc.readAllStandardError()).decode(errors="replace").strip().splitlines()
        proc.deleteLater()
        try:
            if code != 0 or not out:
                raise ValueError(err[-1] if err else f"the helper exited with {code}")
            res = json.loads(out[-1])
            res["_centres"] = np.asarray(res["centres_mpc"], dtype=np.float64).reshape(-1, 3)
            res["_r_v"] = np.array([np.nan if v is None else v for v in res["r_v_mpc"]], dtype=np.float64)
            res["_shown"] = np.asarray(res["distinct"], dtype=bool)
            res["_r_eff"] = np.array([np.nan if v is None else v for v in res["r_eff_mpc"]], dtype=np.float64)
            self._void_cache[key] = res
            self.status.emit(f"voids: {int(res['_shown'].sum())} distinct, well-resolved voids (outlines: R_v; "
                             f"sigma {res['sigma_mpc']:.3g} Mpc)")
        except (ValueError, KeyError, TypeError) as e:
            self._void_cache[key] = {"error": str(e)}
            self.status.emit(f"no void outlines: {e}")
        self._voids_redraw()
        if self._grid_shown and self.voids.isChecked():
            self.show_grid(True)                    # a panel's catalogue arrived: its outlines
        self._voids_next()

    def _voids_next(self):
        """The next catalogue the figure grid waits for (one helper at a time)."""
        while self._void_proc is None and self._void_queue and self.voids.isChecked() and not self._shut:
            o = self._void_queue.pop(0)
            if self._void_cache_key(o) not in self._void_cache:
                self._voids_request(o)

    def _voids_start_failed(self, key, proc, err):
        """The helper never started (no 'finished' follows): said, and the next tick tries again."""
        if err != QProcess.FailedToStart or proc is not self._void_proc:
            return
        self._void_proc = self._void_key = self._void_out = None
        self._void_cache[key] = {"error": f"could not start {rs.PYTHON}: {proc.errorString()}"}
        proc.deleteLater()
        self.status.emit(f"no void outlines: could not start {rs.PYTHON}")
        self._voids_redraw()
        self._voids_next()

    def stop_void_search(self):
        """Stop a catalogue search and forget the grid's queue: for the window when a heavy job starts (a
        catalogue build costs about 80 B per cell); the next tick or output change asks again."""
        self._void_queue.clear()
        self._voids_stop()

    def _voids_stop(self):
        proc, self._void_proc, self._void_key, self._void_out = self._void_proc, None, None, None
        if proc is not None:
            proc.kill()
            proc.waitForFinished(2000)

    def _slice_position(self) -> float | None:
        """Where the shown plane lies along its normal (Mpc, or cells without a box)."""
        req = self._req
        if req is None:
            return None
        if req.get("pts") and self._pts_side is not None:
            return float(self._pts_side.get("center", 0.0))
        o, axis = req["output"], req["axis"]
        return float(o.lo[axis] + (req["index"] + 0.5) * o.length[axis] / o.n)

    def _void_result(self):
        """The catalogue of the output on the map (the void_outlines.py result), or None."""
        o = self.current
        if not self.voids.isChecked() or self._grid_shown or o is None or self.voids_applicable(o):
            return None
        res = self._void_cache.get(self._void_cache_key(o))
        if not res or "error" in res or self._req is None or self._req.get("output") is not o:
            return None
        return res

    def _void_circles(self, radius: str = "_r_v"):
        """The outlines over what the map shows now (None: off, not applicable, or not found yet); radius
        '_r_eff' = the dashed R_eff outlines (None unless ticked)."""
        res = self._void_result()
        g = self._geometry()
        if res is None or g is None or (radius == "_r_eff" and not self.void_reff.isChecked()):
            return None
        axis, _index, rest, lo, hi, n0, n1 = g
        sel = np.nonzero(res["_shown"])[0]
        circles = G.void_circles(res["_centres"][sel], res[radius][sel], float(res["box_mpc"]), axis,
                                 self._slice_position(), rest, lo, hi, n0, n1)
        return [(x, y, rx, ry, int(sel[k])) for x, y, rx, ry, k in circles]

    def _voids_redraw(self):
        c = self.view.canvas
        if c._img is None or self._grid_shown:
            return
        c.circles = self._void_circles()
        c.dashed = self._void_circles("_r_eff")
        c._redraw()

    def voids_shown(self) -> list[int]:
        """The catalogue rows outlined on the map now (R_v or R_eff), in row order: the saved CSV's voids."""
        rows = {k for circles in (self.view.canvas.circles, self.view.canvas.dashed) for *_xy, k in circles or ()}
        return sorted(rows)

    def _void_pick(self, r: int, c: int):
        """The catalogue row under image pixel (r, c), or None: the R_v outlines first, then the dashed R_eff."""
        cv = self.view.canvas
        k = G.void_at(cv.circles, c + 0.5, r + 0.5)
        k = k if k is not None else G.void_at(cv.dashed, c + 0.5, r + 0.5)
        # the hit test first: the result's key stats the density file, not wanted on every mouse move
        return k if k is not None and self._void_result() is not None else None

    def _void_goto(self):
        """The slice through the centre of the void last clicked (along the map's own axis)."""
        res, picked, o = self._void_result(), self._void_picked, self.current
        if res is None or picked is None or picked[0] != self._void_cache_key(o) or self.is_zoomed():
            return
        axis = self.axis.currentIndex()
        x = float(res["_centres"][picked[1]][axis])
        i = int(np.floor((x - float(o.lo[axis])) / (float(o.length[axis]) / o.n))) % int(o.n)
        self.index.setValue(i)

    def _panel_voids(self, o) -> dict | None:
        """A grid panel's outlines: its output's own catalogue (shown rows: centres, R_v, R_eff when ticked) when
        'voids' is ticked and it applies; a catalogue not found yet is asked for (queued, one at a time; the grid
        is recomposed when it arrives). None otherwise."""
        if not self.voids.isChecked() or self.voids_applicable(o):
            return None
        res = self._void_cache.get(self._void_cache_key(o))
        if res is None:
            self._voids_request(o, queued=True)
            return None
        if "error" in res:
            return None
        sel = np.nonzero(res["_shown"])[0]
        return {"rows": sel, "centres": res["_centres"][sel], "r_v": res["_r_v"][sel], "box": float(res["box_mpc"]),
                "r_eff": res["_r_eff"][sel] if self.void_reff.isChecked() else None}
