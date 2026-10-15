"""The GUI's Runs browser: every output set on disk, read back from its run log (results.py).

One row per run -- simulation, snapshot, output name, grid, options, when, how long, peak memory,
size on disk -- and a status that names every known output-changing fix the run predates (by
the binary's build stamp, else the run date). Actions: open the log, load the run's settings into
the Grids tab (re-run with the same settings), open it in Explore, show it in Finder, move its
outputs to the Trash (never a hard delete), and (2026-10-11) build the run's void catalogue as a queue job; the
'Voids' column says whether that catalogue is cached, stale (the grid was rewritten after it) or absent, read in a
background thread (results.void_status).

RunningNow (2026-10-11): the runs going on this machine wherever they were started (nohup, a terminal script,
this launcher), from the processes' command lines and their progress sidecars (runspec.running_now), with a Stop.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QFile, QObject, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as R  # noqa: E402

LEVEL_COLOR = {"error": QColor("#c62828"), "warning": QColor("#b26a00"), "info": QColor("gray")}
COLUMNS = ["Simulation", "Snap", "Output", "Grid", "Options", "Finished", "Wall", "Peak", "Size", "Voids", "Status"]
VOID_TEXT = {"cached": "cached", "stale": "stale", "absent": "absent", "error": "—"}
VOID_COLOR = {"cached": QColor("#2e7d32"), "stale": QColor("#b26a00"), "absent": QColor("gray"), "error": QColor("gray")}


class _Relay(QObject):
    """Carries a background thread's result to the GUI thread (a queued signal)."""
    done = Signal(object)


def _gb(x):
    return "—" if x is None else f"{x:.1f} GB"


class RunsBrowser(QWidget):
    load_settings = Signal(object)      # RunRecord -> the Grids tab
    explore = Signal(str)               # output root '<dir>/<prefix>' -> the Explore tab
    message = Signal(str)
    changed = Signal()                  # the buttons' state (the selection): the menu bar follows
    build_voids = Signal(object)        # RunRecord -> a 'Build void catalogue' queue job

    def __init__(self, data_root_fn, extra_dirs_fn, python: str | None = None):
        super().__init__()
        self._data_root, self._extra_dirs = data_root_fn, extra_dirs_fn
        self._python = python
        self.records: list[R.RunRecord] = []
        self.void_states: dict[str, tuple[str, str]] = {}      # str(record.path) -> (state, why)
        self._void_gen = 0
        self._relay = _Relay()
        self._relay.done.connect(self._voids_known)
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.filter = QLineEdit(placeholderText="filter: simulation, snapshot, output name, 'stale', 'failed'")
        self.filter.textChanged.connect(self._fill)
        top.addWidget(self.filter, 1)
        self.b_refresh = QPushButton("Refresh")
        self.b_refresh.clicked.connect(self.reload)
        top.addWidget(self.b_refresh)
        v.addLayout(top)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._show)
        self.table.cellDoubleClicked.connect(lambda *_: self._open_log())
        self.detail = QPlainTextEdit(readOnly=True)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.table)
        split.addWidget(self.detail)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        v.addWidget(split, 1)
        btns = QHBoxLayout()
        self.b_log = QPushButton("Open log")
        self.b_log.clicked.connect(self._open_log)
        self.b_load = QPushButton("Load settings")
        self.b_load.setToolTip("Put this run's settings into the Grids tab (a custom-snapshot run: the Custom tab), "
                               "to run it again")
        self.b_load.clicked.connect(lambda: self._sel() and self.load_settings.emit(self._sel()))
        self.b_explore = QPushButton("Explore")
        self.b_explore.clicked.connect(lambda: self._sel() and self.explore.emit(str(self._sel().path.parent / self._sel().prefix)))
        self.b_reveal = QPushButton("Show in Finder" if sys.platform == "darwin" else "Show folder")
        self.b_reveal.clicked.connect(self._reveal)
        self.b_trash = QPushButton("Move to Trash…")
        self.b_trash.clicked.connect(self._trash)
        self.b_voids = QPushButton("Build void catalogue")
        self.b_voids.setToolTip("Queue a job that builds this run's void catalogue into the analysis cache (about a "
                                "minute and 10 GB at 512 cubed), for the void figures and the Explore outlines")
        self.b_voids.clicked.connect(lambda: self._sel() and self.build_voids.emit(self._sel()))
        for b in (self.b_log, self.b_load, self.b_explore, self.b_voids, self.b_reveal, self.b_trash):
            btns.addWidget(b)
        btns.addStretch(1)
        self.summary = QLabel("")
        self.summary.setStyleSheet("color: gray")
        btns.addWidget(self.summary)
        v.addLayout(btns)
        self._show()

    def reload(self) -> int:
        self.records = R.scan_runs(self._data_root(), self._extra_dirs())
        self._fill()
        self._start_void_status()
        return len(self.records)

    def _start_void_status(self):
        """Ask void_cache.py, in a thread, whether each run's catalogue is cached (seconds over a whole data root;
        a newer reload's answer wins)."""
        self._void_gen += 1
        gen, recs, root = self._void_gen, list(self.records), self._data_root()
        if not any(R.void_candidate(r) for r in recs):
            return
        relay, python = self._relay, self._python

        def work():
            try:
                res = R.void_status(recs, root, python)
            except Exception as e:              # never let a helper problem take the thread down silently
                res = {str(r.path): ("error", str(e)) for r in recs if R.void_candidate(r)}
            relay.done.emit((gen, res))
        threading.Thread(target=work, daemon=True).start()

    def _voids_known(self, payload):
        gen, res = payload
        if gen != self._void_gen:
            return
        self.void_states = res
        self._fill()

    def _visible(self) -> list[R.RunRecord]:
        words = self.filter.text().lower().split()
        out = []
        for r in self.records:
            stale = R.stale_reasons(r)
            hay = " ".join([r.sim, f"{r.snap:03d}" if r.snap is not None else "", r.prefix, str(r.path),
                            "stale" if stale else "", "failed" if not r.ok else "", r.attempt, r.options_text(),
                            "voids " + self.void_states.get(str(r.path), ("", ""))[0]]).lower()
            if all(w in hay for w in words):
                out.append(r)
        return out

    def _fill(self, *_):
        keep = self._sel()
        rows = self._visible()
        self.table.setRowCount(0)
        n_stale = 0
        for k, r in enumerate(rows):
            stale = R.stale_reasons(r)
            n_stale += bool(stale)
            worst = next((lv for lv in ("error", "warning", "info") if any(s[0] == lv for s in stale)), None)
            status = ("✖ failed" if not r.ok else
                      {"error": "✖ wrong: regenerate", "warning": "⚠ stale", "info": "ℹ note"}[worst] if worst
                      else "✔ ok")
            if r.attempt == "running" and r.progress:          # C1: a run going under this name, with its sidecar
                status = "▶ " + (R.progress_text(r.progress) or "running")
            elif r.attempt == "resumable":                     # K2: killed with its checkpoint kept (2026-10-11)
                status = "⏸ resumable"
            elif r.gpu_retries:                                # G1: the log's GPU retry tally, per cause (2026-10-11)
                status += ", " + R.retries_text(r.gpu_retries, r.gpu_retries_by_cause)
            vstate, vwhy = self.void_states.get(str(r.path), ("", ""))
            voids = VOID_TEXT.get(vstate, "…" if R.void_candidate(r) and self._void_gen else "")
            cells = [r.sim or "custom", "" if r.snap is None else f"{r.snap:03d}", r.prefix,
                     f"{r.grid}{'²' if r.dim == 2 else '³'}" if r.grid else "—", r.options_text(),
                     r.finished.strftime("%Y-%m-%d %H:%M") if r.finished else "—",
                     R.format_duration(r.wall_seconds), _gb(r.footprint_gb or r.peak_rss_gb),
                     f"{r.size_bytes / 1e9:.1f} GB", voids, status]
            self.table.insertRow(k)
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                it.setData(Qt.UserRole, k)
                if c == len(cells) - 2 and vstate:
                    it.setForeground(VOID_COLOR.get(vstate, QColor("gray")))
                    it.setToolTip(vwhy)
                if c == len(cells) - 1:
                    it.setForeground(LEVEL_COLOR["warning"] if r.attempt == "resumable" else
                                     LEVEL_COLOR.get(worst, QColor("#2e7d32")) if r.ok else LEVEL_COLOR["error"])
                    if stale:
                        it.setToolTip("\n".join(m for _, m in stale))
                self.table.setItem(k, c, it)
        self._rows = rows
        self.summary.setText(f"{len(rows)} run{'s' if len(rows) != 1 else ''}"
                             + (f" · {n_stale} with outdated outputs" if n_stale else ""))
        if keep is not None and keep in rows:
            self.table.selectRow(rows.index(keep))
        self._show()

    def _sel(self) -> R.RunRecord | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        k = rows[0].row()
        return self._rows[k] if 0 <= k < len(getattr(self, "_rows", [])) else None

    def _show(self):
        self._show_detail()
        self.changed.emit()

    def _show_detail(self):
        r = self._sel()
        for b in (self.b_log, self.b_load, self.b_explore, self.b_voids, self.b_reveal, self.b_trash):
            b.setEnabled(r is not None)
        if r is None:
            self.detail.setPlainText("Every run with a log on disk: TNG runs under the data root and custom-snapshot "
                                     "runs in their output folders. Select one for its details.")
            return
        self.b_load.setEnabled((bool(r.sim) and r.estimator in ("ps", "dtfe")) or R.custom_settings(r) is not None)
        self.b_explore.setEnabled(any(p.name.endswith((".a_den", ".den")) for p in r.outputs))
        self.b_voids.setEnabled(R.void_candidate(r))
        lines = [str(r.path), ""]
        stale = R.stale_reasons(r)
        if not r.ok:
            lines.append("✖ this run did not finish (no summary, or an ERROR block in its log)")
        for lv, m in stale:
            lines.append({"error": "✖ ", "warning": "⚠ ", "info": "ℹ "}[lv] + m)
        if stale or not r.ok:
            lines.append("")
        build = f"{r.binary} {r.build_rev}, built {r.build_time:%Y-%m-%d %H:%M}" if r.build_time else \
            f"{r.binary or 'binary'} (build not recorded: logs before 2026-09-30 carry no build stamp)"
        lines += [f"build     {build}",
                  f"finished  {r.finished:%Y-%m-%d %H:%M}" if r.finished else "finished  —",
                  f"wall      {R.format_duration(r.wall_seconds)}    peak memory {_gb(r.footprint_gb)} footprint, "
                  f"{_gb(r.peak_rss_gb)} resident",
                  f"outputs   {len(r.outputs)} files, {r.size_bytes / 1e9:.2f} GB",
                  f"settings  {r.options_text() or '—'}" + (f", {r.grid}{'²' if r.dim == 2 else '³'} grid" if r.grid else "")]
        if r.gpu_retries:           # the log's closing tally (G1, 2026-10-11)
            lines.append(f"retries   {R.retries_text(r.gpu_retries, r.gpu_retries_by_cause)} of the deposit")
        elif r.gpu_retried:
            lines.append("retries   the GPU deposit was restarted (this log has no per-cause tally)")
        vstate, vwhy = self.void_states.get(str(r.path), ("", ""))
        if vstate:
            lines.append(f"voids     {vstate}" + (f" ({vwhy})" if vwhy else ""))
        if r.progress:
            lines.append(f"progress  {R.progress_text(r.progress)}")
        self.detail.setPlainText("\n".join(lines))

    def _open_log(self):
        r = self._sel()
        if r is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(r.path)))

    def _reveal(self):
        r = self._sel()
        if r is None:
            return
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(r.path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(r.path.parent)))

    def _trash(self):
        r = self._sel()
        if r is None:
            return
        # every log of the set (.runlog, .runlog.failed, .runlog.partial: 2026-10-08), or a left-behind attempt log
        # shows up as a failed run after the outputs are gone
        files = [p for p in r.outputs if p.is_file()] + [r.path] + [p for p in r.log_files() if p != r.path]
        side = Path(str(r.path.parent / r.prefix) + ".gui.json")
        if side.is_file():
            files.append(side)
        total = sum(p.stat().st_size for p in files if p.exists()) / 1e9
        names = "\n".join(p.name for p in files[:14]) + ("\n…" if len(files) > 14 else "")
        if QMessageBox.question(self, "Move to Trash",
                                f"Move these {len(files)} files ({total:.1f} GB) of {r.prefix} in\n{r.path.parent}\n"
                                f"to the Trash?\n\n{names}") != QMessageBox.Yes:
            return
        failed = [p for p in files if p.exists() and not QFile.moveToTrash(str(p))]
        self.message.emit(f"moved {len(files) - len(failed)} files to the Trash"
                          + (f"; {len(failed)} could not be moved" if failed else ""))
        self.reload()


RUNNING_COLUMNS = ["Simulation", "Snap", "Program", "pid", "Phase", "Done", "ETA", "Memory", "Elapsed", "State"]


def running_cells(e: dict) -> list[str]:
    """One 'Running now' row (runspec.running_now's dict) as the table's texts."""
    p = e.get("progress") or {}
    try:
        done = f"{int(p['done'])}/{int(p['total'])}"
        if float(p["total"]) > 0:
            done += f" ({100.0 * float(p['done']) / float(p['total']):.0f}%)"
    except (KeyError, TypeError, ValueError):
        done = "—"
    try:
        eta = R.format_duration(float(p["eta_s"])) if float(p["eta_s"]) >= 0 else "—"
    except (KeyError, TypeError, ValueError):
        eta = "—"
    gb = e.get("gb")
    if gb is None:
        try:
            gb = float(p["rss_gb"])
        except (KeyError, TypeError, ValueError):
            gb = None
    state = e.get("state", "")
    if state == "running" and not p:
        state = "running (no progress file)"
    elif state == "died":
        state = "died: the process is gone, its progress file left behind"
    retries = p.get("gpu_retries") if isinstance(p, dict) else None
    if state.startswith("running") and retries:
        try:        # per cause when the sidecar has them (G1, 2026-10-11)
            state += ", " + R.retries_text(int(retries), p.get("gpu_retries_by_cause"))
        except (TypeError, ValueError):
            state += f", {retries} GPU retries"
    return [e.get("sim") or "custom", "" if e.get("snap") is None else f"{int(e['snap']):03d}",
            e.get("name") or "", str(e.get("pid") or ""), str(p.get("phase") or "—"), done, eta, _gb(gb),
            R.format_duration(e.get("elapsed_s")) if e.get("elapsed_s") is not None else "—", state]


class RunningNow(QWidget):
    """The runs going on this machine, wherever they were started (2026-10-11): the launcher's own, a terminal
    script, a nohup'ed overnight batch. Filled by the window's background poll (runspec.running_now): simulation
    and snapshot from the binary's command line, phase, done/total, ETA and memory from its progress sidecar
    '<output>.runlog.progress' (C1), elapsed from ps. A leftover sidecar whose pid is gone reads 'died'. Stop
    sends the binary SIGTERM (its run script's trap then keeps the attempt's log as '.runlog.failed')."""
    message = Signal(str)

    def __init__(self):
        super().__init__()
        self.entries: list[dict] = []
        v = QVBoxLayout(self)
        self.table = QTableWidget(0, len(RUNNING_COLUMNS))
        self.table.setHorizontalHeaderLabels(RUNNING_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._sync)
        v.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.b_stop = QPushButton("Stop…")
        self.b_stop.setToolTip("End the selected run (SIGTERM to the program; its run script keeps the log of the "
                               "attempt as a failed one)")
        self.b_stop.clicked.connect(self._stop)
        row.addWidget(self.b_stop)
        self.note = QLabel("No DTFE or PS-DTFE run is going on this machine.")
        self.note.setStyleSheet("color: gray")
        self.note.setWordWrap(True)
        row.addWidget(self.note, 1)
        v.addLayout(row)
        self._sync()

    def show_entries(self, entries: list[dict]):
        keep = self._sel()
        self.entries = list(entries)
        self.table.setRowCount(0)
        for k, e in enumerate(self.entries):
            self.table.insertRow(k)
            for c, text in enumerate(running_cells(e)):
                it = QTableWidgetItem(text)
                if c == len(RUNNING_COLUMNS) - 1:
                    it.setForeground(LEVEL_COLOR["error"] if e.get("state") == "died" else QColor("#2e7d32"))
                if e.get("output"):
                    it.setToolTip(e["output"])
                self.table.setItem(k, c, it)
        live = sum(e.get("state") == "running" for e in self.entries)
        died = len(self.entries) - live
        self.note.setText("No DTFE or PS-DTFE run is going on this machine." if not self.entries else
                          f"{live} run{'s' if live != 1 else ''} going" + (f", {died} died" if died else "")
                          + " (refreshed every few seconds)")
        if keep is not None:
            for k, e in enumerate(self.entries):
                if e.get("pid") == keep.get("pid") and e.get("output") == keep.get("output"):
                    self.table.selectRow(k)
        self._sync()

    def _sel(self) -> dict | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        k = rows[0].row() if rows else -1
        return self.entries[k] if 0 <= k < len(self.entries) else None

    def _sync(self):
        e = self._sel()
        self.b_stop.setEnabled(e is not None and e.get("state") == "running" and bool(e.get("pid")))

    def _stop(self):
        e = self._sel()
        if e is None or e.get("state") != "running" or not e.get("pid"):
            return
        pid = int(e["pid"])
        what = f"{e.get('name') or 'the run'} (pid {pid})" + (f" on {e['sim']} snapshot {int(e['snap']):03d}"
                                                               if e.get("sim") and e.get("snap") is not None else "")
        if QMessageBox.question(self, "Stop run", f"Stop {what}?\n\nIts outputs so far are not kept as a finished "
                                "run.") != QMessageBox.Yes:
            return
        try:
            os.kill(pid, signal.SIGTERM)
            self.message.emit(f"sent stop to {what}")
        except OSError as err:
            self.message.emit(f"could not stop {what}: {err}")
