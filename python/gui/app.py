#!/usr/bin/env python3
"""Lamina, the launcher of the DTFE and PS-DTFE programs (PySide6).

Six tabs; the first five are jobs (runspec.py):
  Data       scripts/download_snapshots.sh and python/tools/merge_HDF5.py: snapshots, group
             catalogues, merger trees and initial conditions onto disk, as h-free combined files
  Grids      scripts/run_ps_dtfe.sh or run_dtfe.sh for the chosen snapshots
  Custom snapshot  the binary itself on any snapshot file (not IllustrisTNG): units, box, periodicity
             and initial positions spelled out
  Pipeline   scripts/run_ps_pipeline.sh (grids + a full-box image plane for every stale
             snapshot), then the point-evaluated figures
  Plots      the figure scripts of python/plot and analyze.py
  Explore    any run's grids as maps, and a click-to-query server on its snapshot (explore.py)
Each job shows the exact terminal commands it stands for (and exports them as a script or a
SLURM job), checks the combinations the scripts reject, and runs the commands one after another
with live progress -- now, or as a job in the queue, which runs its jobs in sequence (and waits
for other DTFE runs on the machine). The figures a run writes appear in the Figures browser, and
every run on disk in the Runs browser (runs_view.py), which also flags outputs made before an
output-changing fix. A GUI run is always one you could have typed. First launch: setup_wizard.py.

    scripts/gui.sh
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QLocale, QProcess, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidgetItem, QMainWindow,
    QMessageBox, QPushButton, QAbstractItemView, QAbstractSpinBox, QSplitter, QStackedWidget, QTabWidget, QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runspec as rs  # noqa: E402
import presets as P  # noqa: E402
import results as R  # noqa: E402
import grids as G  # noqa: E402
from app_widgets import _checkable, _grow_fields, _scroll  # noqa: E402
from app_widgets import CheckDelegate, FigureBrowser  # noqa: E402,F401  (re-exported)
from app_tabs import TabsMixin  # noqa: E402
from app_specs import SpecsMixin  # noqa: E402
from app_run import RunMixin  # noqa: E402
from app_menus import MenusMixin  # noqa: E402
from app_check import CheckMixin  # noqa: E402

ERROR_COLOR, WARN_COLOR, OK_COLOR = QColor("#c62828"), QColor("#b26a00"), QColor("#2e7d32")
INFO_COLOR = QColor("gray")
CHECK_ICON = {"error": "✖  ", "warning": "⚠  ", "ok": "✔  ", "info": "ℹ  ", "hint": "→  "}
CHECK_COLOR = {"error": ERROR_COLOR, "warning": WARN_COLOR, "ok": OK_COLOR, "info": INFO_COLOR,
               "hint": INFO_COLOR}
DATA, GRIDS, OWN, PIPELINE, PLOTS, EXPLORE = range(6)
PROBLEM_LINES = r"(?i)\berror\b|\bwarning\b|failed|mismatch|out of memory|^!! "    # the log's 'next problem'


def clock_text(sec: float) -> str:
    """A running clock: '0:07', '12:40', '2:05:09'."""
    sec = int(max(0.0, sec))
    h, m, s_ = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s_:02d}" if h else f"{m}:{s_:02d}"


def took_text(sec: float) -> str:
    """How long a job took: '7 s', '12m 40s', '2h 05m'."""
    sec = int(round(max(0.0, sec)))
    if sec < 60:
        return f"{sec} s"
    if sec < 3600:
        return f"{sec // 60}m {sec % 60:02d}s"
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m"
KIND_OF_TAB = {DATA: "data", GRIDS: "grids", OWN: "custom", PIPELINE: "pipeline", PLOTS: "plots"}
TAB_OF_KIND = {k: t for t, k in KIND_OF_TAB.items()}
TAB_NAMES = ["data", "grids", "custom", "pipeline", "plots", "explore"]
# an 'error' that only says a choice is still to be made: shown as a neutral next step, not in
# red (it still blocks Run) -- a fresh window must not open on an error
HINTS = {"no snapshots selected": "tick the snapshots to run",
         "no simulation selected": "choose a simulation",
         "no simulations selected": "tick the simulations to run",
         "no figure sets selected": "tick the figure sets to draw",
         "no simulation named": "name the simulation to download",
         "nothing selected to download or merge": "tick what to download or merge",
         "no snapshot file chosen": "choose your snapshot file",
         "nothing to run": "choose what to run"}
QUEUE_ICON = {"waiting": "○", "running": "▶", "done": "✔", "failed": "✖", "stopped": "■", "skipped": "⤼"}
QUEUE_COLOR = {"waiting": QColor("gray"), "running": QColor("#1565c0"), "done": QColor("#2e7d32"),
               "failed": QColor("#c62828"), "stopped": QColor("#b26a00"), "skipped": QColor("#b26a00")}
MERGED_COLOR, CHUNKS_COLOR = QColor("#2e7d32"), QColor("#b26a00")
PER_TEXT = {"snapshot": "one call per selected snapshot", "series": "one call for all selected snapshots",
            "sim": "one call per simulation"}


# what the plane conventions' menu adds to the binary's own help: the Runs tab's warning about older planes
PTS_CONVENTIONS_TIP = ("\n\nThe Runs tab warns about a phase-space plane made before 13 October 2026 with the vertex "
                       "masses or the volume weighting: it has the old estimators. Legacy reproduces such a plane on "
                       "purpose (its run log says so, and it is not warned about); the grids' conventions are the ones "
                       "to compare a plane with the grids.")


def _pts_conventions_combo(changed) -> QComboBox:
    """The phase-space image plane's conventions (PS-DTFE '--pts-conventions', the scripts' PTS_CONVENTIONS;
    runspec.PTS_CONVENTIONS): the grids' (the default) or legacy."""
    c = QComboBox()
    for key in rs.PTS_CONVENTIONS:
        c.addItem(rs.PTS_CONVENTIONS_TEXT[key], key)
    c.setToolTip(P.tooltip("pts_conventions") + PTS_CONVENTIONS_TIP)
    c.currentIndexChanged.connect(changed)
    return c


def _hint(level: str, msg: str) -> tuple[str, str]:
    """A selection-still-missing error as a neutral hint (see HINTS)."""
    for key, text in HINTS.items():
        if level == "error" and msg.startswith(key):
            return "hint", text + msg[len(key):]
    return level, msg


# ==================================================================== main window
class MainWindow(TabsMixin, SpecsMixin, RunMixin, MenusMixin, CheckMixin, QMainWindow):
    def __init__(self, spec: rs.RunSpec | None = None, remember: bool = True,
                 pipeline: rs.PipelineSpec | None = None, plots: rs.PlotSpec | None = None,
                 data: rs.DataSpec | None = None, settings: QSettings | None = None,
                 custom: rs.CustomSpec | None = None):
        QLocale.setDefault(QLocale.c())      # spin boxes read and show '2.00', as every label does (not the system's
        super().__init__()                   # '2,00'); BEFORE the window exists: children inherit the parent's locale
        self.setLocale(QLocale.c())
        self.setWindowTitle("Lamina")
        self.remember = remember
        self.settings = settings or QSettings("DTFE", "launcher")   # the settings' key: saved choices stay where they are
        self.spec = spec or self._restore("spec", rs.RunSpec)
        self.pipe = pipeline or self._restore("pipeline", rs.PipelineSpec)
        self.plots = plots or self._restore("plots", rs.PlotSpec)
        self.data = data or self._restore("data", rs.DataSpec)
        self.custom = custom or self._restore("custom", rs.CustomSpec)
        if custom is None and not self._restore_json("custom", None):
            # a fresh custom-snapshot job uses the GPU when built with one (the Grids and Pipeline tabs' rule,
            # rs.gpu_build_default: not an emulated one)
            self.custom.gpu = rs.gpu_build_default("ps")
        # standard DTFE's production cell average is the exact one (rs.production_deposit). A state saved by an
        # older launcher holds its start, 'sampled', for a standard-DTFE Grids or Custom job (the Monte-Carlo
        # average, never really chosen: it was the only start): moved to 'exact' ONCE; the Monte-Carlo average
        # stays selectable, and a later choice of it is kept
        if self.remember and not self._restore_json("dtfe_exact_start", False):
            for sp, given in ((self.spec, spec), (self.custom, custom)):
                if given is None and sp.estimator == "dtfe" and sp.deposit == "sampled":
                    sp.deposit = "exact"
            self.settings.setValue("dtfe_exact_start", json.dumps(True))
        # the partitions merge in index order by default (the staged merge; run_ps_dtfe.sh ORDERED_MERGE=1, the
        # binary's default). A state saved by an older launcher holds its start, off, on the Grids, Pipeline and
        # Custom tabs (never really chosen: it was the start): moved to on ONCE; unticking it later is kept
        if self.remember and not self._restore_json("ordered_merge_start", False):
            for sp, given in ((self.spec, spec), (self.pipe, pipeline), (self.custom, custom)):
                if given is None and not sp.ordered_merge:
                    sp.ordered_merge = True
            self.settings.setValue("ordered_merge_start", json.dumps(True))
        # where custom-snapshot jobs wrote outputs (the Runs browser and Explore look there too)
        self.custom_dirs: list[str] = [d for d in self._restore_json("custom_dirs", []) if isinstance(d, str)]
        self.user_presets: dict = self._restore_json("presets", {})
        self.show_commands = str(self._restore_json("show_commands", False)).lower() == "true"
        self._explore_saved: dict = self._restore_json("explore", {}) if remember else {}   # restored on first show
        self.runs_cache: list[R.RunRecord] = []            # for the time estimates
        self._runs_scanned = 0.0
        self._job_text = ""                                  # this job's output, for the failure advice
        self._tee = None                                     # the open run-log copy of the current step
        self._tee_path = ""                                  # ... its log of record ('<root>.runlog'; written as .partial)
        self._after_job = None                               # e.g. the demo: open Explore when it finishes
        self.proc: QProcess | None = None
        self._loading = False
        self._own_last_file = ""        # the custom file whose particle count last set the GPU box
        # the deposit choice per estimator, per tab ('grids', 'own'): {estimator: 'sampled' | 'exact'}, what an
        # estimator switch brings back (_estimator_switched; this session only)
        self._deposit_kept: dict[str, dict[str, str]] = {}
        self._steps: list[rs.Step] = []
        self._step_i, self._results = -1, []
        self._job_tab, self._job_start = GRIDS, None
        self._total_known = False                          # the job's snapshot count known at its start
        self._job_t0 = self._step_t0 = None               # monotonic clock of the job / its step
        self._snap_t0s: list[float] = []                   # when each snapshot of the job started (the ETA)
        self._per_snap_prior = None                        # seconds per snapshot from earlier runs, before one is done
        self._queue_est: tuple[float, float, int] | None = None   # (when, seconds, jobs without an estimate): the
                                                           # clock ticks every second, the estimate is redone every minute
        self._last_log_path = ""                           # the newest run log written beside the outputs
        self._panel_tab = GRIDS                              # the last tab with the run panel (not Explore)
        self._buffer, self._summary, self._stopping = "", "", False
        self._snaps_started = self._snaps_ok = self._total_snaps = 0
        self._dl_files = self._dl_bytes = self._problems = 0
        self._job_spec, self._job_sim, self._figs_since = None, "", None
        self.queue: list[rs.QueuedJob] = self._restore_queue()
        self._queue_running = False
        self._current_job: rs.QueuedJob | None = None
        self._wait_poll_ms = 10000          # how often a waiting queue looks for other DTFE runs again
        # memory checks: {memory key: {(sim, snapshot, plane size or None): auto-tune report}}
        self.mem_results: dict[str, dict] = {}
        self._mcheck: dict | None = None    # the check in progress
        self.opt_widgets: dict[tuple[str, str], tuple[rs.Opt, QWidget]] = {}

        self._build_menus()             # first: refresh() enables its actions
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(self._build_data(), 1)
        self.setup_btn = QPushButton("Setup…")
        self.setup_btn.setToolTip("Data folder, building the programs, the TNG API key, and the demo")
        self.setup_btn.clicked.connect(lambda: self.run_setup())
        top.addWidget(self.setup_btn, 0, Qt.AlignTop)
        lv.addLayout(top)
        import explore as X
        import runs_view as RV
        self.explore_view = X.ExploreView()
        # the Save options live in the launcher's settings when it remembers them
        self.explore = X.ExploreControls(self.explore_view, lambda: self.root_edit.text().strip(),
                                         self._custom_output_dirs, settings=self.settings if self.remember else None)
        self.explore.status.connect(self.status_message)
        self.explore.changed.connect(lambda: self._sync_explore_actions())
        # on the Explore tab, left / right step through the slices (shift: 10 at a time) and up / down
        # through the fields, wherever the focus is, except in a text or number field (those keep
        # the keys for their cursor and their value) or a table (its rows)
        self._slice_keys, self._field_keys = [], []
        for keys, delta in (("Left", -1), ("Right", 1), ("Shift+Left", -10), ("Shift+Right", 10)):
            sc = QShortcut(QKeySequence(keys), self, enabled=False)
            sc.activated.connect(lambda d=delta: self.explore.step_slice(d))
            self._slice_keys.append(sc)
        for keys, delta in (("Up", -1), ("Down", 1)):
            sc = QShortcut(QKeySequence(keys), self, enabled=False)
            sc.activated.connect(lambda d=delta: self.explore.step_field(d))
            self._field_keys.append(sc)
        QApplication.instance().focusChanged.connect(self._explore_keys)
        self.tabs = QTabWidget()
        self.tabs.addTab(_scroll(self._build_data_tab()), "Data")
        self.tabs.addTab(_scroll(self._build_grids_tab()), "Grids")
        self.tabs.addTab(_scroll(self._build_own_tab()), "Custom snapshot")
        self.tabs.addTab(_scroll(self._build_pipeline_tab()), "Pipeline")
        self.tabs.addTab(_scroll(self._build_plots_tab()), "Plots")
        self.tabs.addTab(_scroll(self.explore), "Explore")
        self.tabs.setTabToolTip(OWN, "Run DTFE / PS-DTFE on a snapshot file of your own (not IllustrisTNG)")
        self.tabs.setTabToolTip(EXPLORE, "Look at any run's grids, and click to list the streams at a point")
        lv.addWidget(self.tabs, 1)
        left.setMinimumWidth(690)           # three columns of field boxes ('velocity dispersion' ...) need it:
                                            # at 590 the third column reads 've', 'vo', 'V-'

        self.runs = RV.RunsBrowser(lambda: self.root_edit.text().strip(), self._custom_output_dirs, rs.PYTHON)
        self.runs.build_voids.connect(self.queue_void_build)
        self.running_now = RV.RunningNow()
        self.running_now.message.connect(self.status_message)
        self.runs.load_settings.connect(self._load_run_settings)
        self.runs.explore.connect(self._explore_output)
        self.runs.message.connect(self.status_message)
        self.runs.changed.connect(self._runs_changed)
        self.right = QStackedWidget()
        self.right.addWidget(self._build_run_panel())
        self.right.addWidget(self.explore_view)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.splitter = splitter
        self.resize(1400, 900)
        if remember:                        # the size, place and pane split of the last session;
            geo, split = self.settings.value("geometry"), self.settings.value("splitter")   # restoreGeometry fits
            if geo is not None:             # a window from a screen no longer attached onto the current one
                self.restoreGeometry(geo)
            if split is not None:
                splitter.restoreState(split)
        self.setAcceptDrops(True)           # a grid file, a snapshot or a folder dropped on the window

        self._load_into_widgets()
        self._read_widgets()            # the shared data (root, simulation, snapshots) into every job
        # a fresh window opens on Grids; a remembered one where it was left (by name: tab indices
        # move when tabs are added)
        name = str(self.settings.value("tab_name", "grids")) if remember else "grids"
        self.tabs.setCurrentIndex(TAB_NAMES.index(name) if name in TAB_NAMES else GRIDS)
        if self.tabs.currentIndex() != EXPLORE:
            self._panel_tab = self.tabs.currentIndex()
        self.tabs.currentChanged.connect(self._tab_changed)
        self._tab_view(self.tabs.currentIndex())
        for view in (self.snap_list, self.d_table, self.p_sims, self.fig_list):
            _checkable(view)
        _grow_fields(self)
        self._fill_snaps()
        self._refresh_queue()
        self.refresh()
        # re-check every few seconds while idle: jobs started or finished in a terminal, an external
        # disk plugged in or out
        self._poll = QTimer(self, interval=4000)
        self._poll.timeout.connect(lambda: self.proc is None and self.refresh())
        self._poll.start()
        # the memory strip and the 'Running now' panel: vm_stat, sysctl, pgrep, ps and a glob of the
        # data root's progress files, in a worker thread every 5 s (never two at once), the answer relayed back
        self.mem_strip = QLabel("")
        self.mem_strip.setStyleSheet("color: gray")
        self.mem_strip.setToolTip("Free memory (what the other programs leave), swap in use, and the DTFE / PS-DTFE "
                                  "programs running on this machine with their resident memory")
        self.statusBar().addPermanentWidget(self.mem_strip)
        self._machine_relay = RV._Relay()
        self._machine_relay.done.connect(self._machine_known)
        self._machine_busy = False
        self.machine: dict = {}             # the last poll: {'mem', 'procs', 'running'}
        self._machine_timer = QTimer(self, interval=5000)
        self._machine_timer.timeout.connect(self._poll_machine)
        self._machine_timer.start()
        self._poll_machine()

    def _job(self):
        """The job of the active tab (the Explore tab has none: the Grids job stands in)."""
        return (self.data, self.spec, self.custom, self.pipe, self.plots, self.spec)[self.tabs.currentIndex()]

    # ================================================================ discovery
    def _fill_sims(self):
        was = self._loading
        self._loading = True
        sims = rs.simulations(self.root_edit.text().strip())
        self.sim_combo.clear()
        self.sim_combo.addItems(sims)
        if self.spec.sim in sims:
            self.sim_combo.setCurrentText(self.spec.sim)
        wanted = set(self.pipe.sims) or {self.sim_combo.currentText()}
        self.p_sims.clear()
        # the run order: the pipeline's simulations as saved (the user's drag order), then the rest in natural
        # order -- TNG50 before TNG100 before TNG300 (text order runs TNG300 first)
        order = [s_ for s_ in self.pipe.sims if s_ in sims] + sorted((s_ for s_ in sims if s_ not in self.pipe.sims),
                                                                    key=G.sim_sort_key)
        for sim in order:
            it = QListWidgetItem(sim)
            it.setData(Qt.UserRole, sim)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if sim in wanted else Qt.Unchecked)
            self.p_sims.addItem(it)
        self.d_sim.clear()
        self.d_sim.addItems(sorted(set(rs.TNG_SIMULATIONS) | set(sims)))
        self.d_sim.setCurrentText(self.data.sim or self.sim_combo.currentText())
        self._fill_data_table()
        self._loading = was
        self._fill_sim_dependent()

    def _sim_path(self) -> Path | None:
        sim = self.sim_combo.currentText()
        return rs.sim_dir(sim, self.root_edit.text().strip()) if sim else None

    def _fill_sim_dependent(self):
        was = self._loading
        self._loading = True
        self._fill_snaps()
        path = self._sim_path()
        current = self.spec.slice_plane
        self.plane_combo.clear()
        self.plane_combo.addItem("none", "")
        for p in (rs.planes(path) if path else []):
            self.plane_combo.addItem(f"{p.name}  ({rs.plane_points(p)/1e6:.1f}M points)", str(p))
        self.plane_combo.setCurrentIndex(max(self.plane_combo.findData(current), 0))
        sidecars = rs.plane_sidecars(path) if path else []
        for (set_key, key), (o, w) in self.opt_widgets.items():
            if o.kind != "plane":
                continue
            current = self.plots.opt(set_key, key)
            w.clear()
            w.addItem("auto: the first plane that fits" if set_key == "pointeval" else "choose a plane", "")
            for sc in sidecars:
                w.addItem(f"{sc.stem}  ({rs.plane_points(sc.with_suffix('.bin'))/1e6:.1f}M px)", str(sc))
            w.setCurrentIndex(max(w.findData(str(current)), 0))
        self._loading = was

    def _marks(self):
        """(done test per snapdir, its mark, the third button's label) for the active tab."""
        tab = self.tabs.currentIndex()
        if tab in (DATA, OWN, EXPLORE):
            return (lambda sd: False), "", "—"
        if tab == PIPELINE:
            prefix = self.pipe.output_prefix or "ps_output"
            return (lambda sd: (sd / f"{prefix}.pts_den").is_file()), "sliced", "Sliced"
        if tab == PLOTS:
            pl = self.plots
            return (lambda sd: rs.has_grids(sd, pl.method, pl.prefix)), "grids", "With grids"
        s = self.spec
        prefix = s.output_prefix or "ps_output"
        return (lambda sd: rs.has_outputs(sd, prefix, s.estimator)), "done", "Not yet run"

    def _fill_snaps(self):
        was = self._loading
        self._loading = True
        wanted = set(self.spec.snapshots)
        done, mark, button = self._marks()
        self.mark_btn.setText(button)
        self.snap_list.clear()
        path = self._sim_path()
        for s in (rs.snapshots(path, done=done) if path else []):
            z = f"z = {s['z']:6.2f}" if s["z"] is not None else "z =    ?  "
            it = QListWidgetItem(f"{s['n']:03d}   {z}   {mark if s['done'] else ''}")
            it.setData(Qt.UserRole, s["n"])
            it.setData(Qt.UserRole + 1, s["done"])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if s["n"] in wanted else Qt.Unchecked)
            if s["done"]:
                it.setForeground(QColor("gray") if self.tabs.currentIndex() == GRIDS else OK_COLOR)
            self.snap_list.addItem(it)
        self._loading = was

    def _select_snaps(self, mode: str):
        want_done = self.tabs.currentIndex() != GRIDS      # 'Not yet run' vs 'With grids'
        self._loading = True
        for i in range(self.snap_list.count()):
            it = self.snap_list.item(i)
            on = mode == "all" or (mode == "mark" and bool(it.data(Qt.UserRole + 1)) == want_done)
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self._loading = False
        self._changed()

    # ================================================================ reactions
    def _root_changed(self):
        self.spec.data_root = self.root_edit.text().strip()
        self._fill_sims()
        if self.explore.outputs:            # Explore lists this root's outputs: follow it
            self.explore.refresh_outputs()
        self._changed()

    def _sim_changed(self, _text=""):
        if self._loading:
            return
        self.spec.snapshots = []
        self._fill_sim_dependent()
        self._changed()

    def _tab_changed(self, i):
        if i != EXPLORE:
            self._panel_tab = i
        self._fill_snaps()
        self._tab_view(i)
        self.refresh()
        self._save()

    def _tab_view(self, tab: int):
        """What the rest of the window shows for this tab: the Explore map replaces the run panel,
        and the shared simulation/snapshot rows only appear for the tabs that use them."""
        self.right.setCurrentWidget(self.explore_view if tab == EXPLORE else self.right.widget(0))
        self._explore_keys(tab=tab)
        shared = tab in (GRIDS, PLOTS)             # the Pipeline tab picks its own simulations/snapshots
        f = self._data_form
        for field in (self.sim_combo, self.snap_list, self._snap_buttons):
            f.setRowVisible(field, shared)
        if tab == EXPLORE and not self.explore.outputs:
            saved, self._explore_saved = self._explore_saved, {}
            self.explore.refresh_outputs(select=(saved or {}).get("root") or None)
            if saved:
                self.explore.restore_state(saved)       # the output, field, slice and range of last time

    def _explore_keys(self, *_, tab: int | None = None):
        """The Explore arrow keys are on only on the Explore tab. Up / down also step off while a
        number field or a table has the focus: a text field takes left / right itself, but up / down
        would otherwise never reach a spin box's value or a table's rows."""
        if not hasattr(self, "tabs"):
            return
        on = (self.tabs.currentIndex() if tab is None else tab) == EXPLORE
        for sc in self._slice_keys:
            sc.setEnabled(on)
        f = QApplication.focusWidget()
        own = f is not None and (isinstance(f, (QAbstractSpinBox, QAbstractItemView))
                                 or isinstance(f.parentWidget(), QAbstractSpinBox))
        for sc in self._field_keys:
            sc.setEnabled(on and not own)

    def _browse_root(self):
        d = QFileDialog.getExistingDirectory(self, "Data root", self.root_edit.text())
        if d:
            self.root_edit.setText(d)
            self._root_changed()

    def _browse_scratch(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Scratch directory (local, not iCloud)",
                                             edit.text() or "/private/tmp")
        if d:
            edit.setText(d)
            self._changed()

    def _marks_key(self):
        return (self.spec.estimator, self.spec.output_prefix, self.pipe.output_prefix,
                self.plots.method, self.plots.prefix)

    def _changed(self, *_):
        if self._loading:
            return
        key = self._marks_key()
        self._read_widgets()
        if self._marks_key() != key:
            self._fill_snaps()      # the marks depend on the estimator and prefixes
        self.refresh()
        self._save()

    def refresh(self):
        """Enable what the active job supports, then show its commands, checks and estimate."""
        tab = self.tabs.currentIndex()
        # ---- Grids tab
        s = self.spec
        ps = s.estimator == "ps"
        for w in (self.nsub_spin, self.ps_group, self.prefix_edit):
            w.setEnabled(ps)                # the hi-res slice works with both estimators
        self._deposit_texts(self.deposit_combo, ps)       # both estimators: standard DTFE's exact cell average
        for name, cb in self.field_checks.items():
            cb.setEnabled(ps or name in rs.DTFE_ALLOWED_FIELDS)
        self.fields_note.setText("" if ps else "Standard DTFE: the velocity dispersion is a phase-space field.")
        self.fields_note.setVisible(not ps)
        self.fields_box.sync_lambda()
        self.cu_check.setEnabled(ps and s.caustics)
        self.vg_check.setEnabled(bool(s.slice_plane))
        self.pc_combo.setEnabled(ps and bool(s.slice_plane))       # a phase-space option (standard DTFE has one way)
        self.nsub_spin.setEnabled(ps and s.deposit == "sampled")
        gpu_ok = rs.gpu_built(s.estimator, s.precision)
        self.gpu_check.setText(P.label("gpu" if ps else "gpu_dtfe") + ("" if gpu_ok else " (this build has no GPU support)"))
        self.gpu_check.setToolTip(P.tooltip("gpu") if ps else P.tooltip("gpu_dtfe", "DTFE"))
        tbb_note = "" if rs.tbb_built(s.precision) else " (this build has no TBB: ignored)"
        self.pt_check.setText(P.label("parallel_triangulation") + tbb_note)
        self.pt_check.setEnabled(ps)
        self.om_check.setEnabled(ps)                # a PS-DTFE option (standard DTFE has no partition merge order)
        self.so_check.setEnabled(True)              # both drivers checkpoint and resume
        for combo in (self.prec_combo, self.o_prec, self.p_prec):     # the double pair's state, in words
            i = combo.findData("double")
            built = rs.binary_built("ps", "double") and rs.binary_built("dtfe", "double")
            combo.setItemText(i, rs.PRECISION_TEXT["double"] + ("" if built else " (not built: Setup builds the pair)"))
        # ---- Custom snapshot tab
        c = self.custom
        cps = c.estimator == "ps"
        flat = c.dim == 2                   # the 2D programs: CPU only, sequential insertion
        for w in (self.o_lag_box, self.o_vm, self.o_vw, self.o_ca, self.o_pt, self.o_ld, self.o_window, self.o_halo):
            w.setEnabled(cps)
        for w in (self.o_sdens, self.o_pstream, self.o_pids, self.o_pc):      # the sample points' phase-space options
            w.setEnabled(cps and bool(c.sample_points))
        self.o_cu.setEnabled(cps and c.caustics)
        self.o_pvg.setEnabled(bool(c.sample_points))
        self._deposit_texts(self.o_deposit, cps)
        for name, cb in self.o_fields.items():
            cb.setEnabled(cps or name in rs.DTFE_ALLOWED_FIELDS)
        self.o_fields_box.sync_lambda()
        self.o_grid_cells.setText(("²" if flat else "³") + " cells    sub-samples per axis")
        self.o_part.setSuffix(("²" if flat else "³") + " partitions")
        self.o_grid.validator().setTop(65536 if flat else 4096)        # a 2D grid may be 65536² (4096³ in 3D)
        self.o_box.setToolTip(f"Optional: xlo xhi ylo yhi{'' if flat else ' zlo zhi'} in the file's units (all >= 0)")
        self.o_lag_path.setEnabled(cps and c.lagrangian == "separate")
        self.o_nsub.setEnabled(cps and c.deposit == "sampled")
        self.o_scalar.setEnabled(c.input_type == 105)
        self.o_pt.setText(P.label("parallel_triangulation") + (" (3D only)" if flat else "" if rs.tbb_built(c.precision)
                                                               else " (this build has no TBB: ignored)"))
        self.o_pt.setEnabled(cps and not flat)
        self.o_om.setEnabled(cps)                   # 2D runs are partitioned too
        self.o_snap_form.setRowVisible(self.o_alpha, cps and not c.periodic)
        self.o_gpu.setEnabled(not flat)
        self.o_gpu.setText(P.label("gpu" if cps else "gpu_dtfe") + (" (the 2D programs run on the CPU)" if flat else
                                                                     "" if rs.gpu_built(c.estimator, c.precision)
                                                                     else " (this build has no GPU support)"))
        self.o_gpu.setToolTip(P.tooltip("gpu") if cps else P.tooltip("gpu_dtfe", "DTFE"))
        # ---- Pipeline tab
        p = self.pipe
        self.p_render.setEnabled(not p.plan_only)
        self.p_render_box.setEnabled(p.render and not p.plan_only)
        self.p_rproject.setEnabled(p.planes > 1)              # one plane: the plane itself
        self.p_thick.setEnabled(p.planes > 1)
        self.p_super_note.setText(f"= {p.supersample ** 2} sample point{'s' if p.supersample > 1 else ''} per pixel")
        self.p_nsub.setEnabled(p.deposit == "sampled")
        self.p_cu.setEnabled(p.caustics)
        self.p_gpu.setText(P.label("gpu") + ("" if rs.gpu_built("ps", p.precision) else " (this build has no GPU support)"))
        self.p_pt.setText(P.label("parallel_triangulation")
                          + ("" if rs.tbb_built(p.precision) else " (this build has no TBB: ignored)"))
        # ---- shared data panel
        for w in (self.snap_list, self.all_btn, self.none_btn, self.mark_btn):
            w.setEnabled(tab not in (PIPELINE, DATA, OWN, EXPLORE))
        if tab == EXPLORE:
            self.stop_btn.setEnabled(self.proc is not None or self._queue_running)
            self._sync_run_actions(tab)
            self._queue_buttons()
            return

        job = self._job()
        if tab == OWN:
            steps = job.steps() if job.input_file else []
        else:
            steps = job.steps() if (tab == PIPELINE or job.sim) and rs.root_ok(job.data_root) else []
        n = len(steps)
        self.cmd_title.setText("Command" if n <= 1 else f"Commands ({n}, run in order)")
        self.cmd_view.setPlainText(rs.script_text(steps) if steps else
                                   "(choose your snapshot file)" if tab == OWN else "(choose a simulation)")
        for w in (self.export_btn, self.export_action, self.copy_action):
            w.setEnabled(bool(steps))
        self.checks.clear()
        problems = job.problems()
        if not steps and not any(level == "error" for level, _ in problems):
            problems.append(("error", "nothing to run"))
        blocked = any(level == "error" for level, _ in problems)
        lines = [_hint(level, msg) for level, msg in problems] + self._memory_lines(tab)
        for level, msg in lines:
            it = QListWidgetItem(CHECK_ICON[level] + msg)
            it.setForeground(CHECK_COLOR[level])
            self.checks.addItem(it)
        if not any(level in ("error", "warning", "hint") for level, _ in lines):
            it = QListWidgetItem("✔  ready to run")
            it.setForeground(OK_COLOR)
            self.checks.addItem(it)
        self.estimate.setText(self._estimate(tab, steps))
        self.run_btn.setEnabled(self.proc is None and not blocked and self._mcheck is None)
        self.queue_btn.setEnabled(not blocked)
        # both estimators (the standard binary's auto-tuner reports too)
        checkable = ((tab == GRIDS and bool(self.spec.snapshots)
                      and bool(self.spec.sim) and rs.root_ok(job.data_root))
                     or (tab == PIPELINE and bool(self.pipe.sims) and rs.root_ok(job.data_root))
                     or (tab == OWN and self._custom_checkable()))
        mem_ok = self._mcheck is not None or (checkable and self.proc is None)
        self.mem_btn.setVisible(tab in (GRIDS, PIPELINE, OWN))
        self.mem_btn.setText("Cancel check" if self._mcheck is not None else "Check memory")
        self.mem_btn.setEnabled(mem_ok)
        self.stop_btn.setEnabled(self.proc is not None or self._queue_running)
        self._sync_run_actions(tab)
        self._queue_buttons()

    def _estimate(self, tab: int, steps: list[rs.Step]) -> str:
        if tab == DATA:
            d = self.data
            if not d.sim or not rs.root_ok(d.data_root):
                return ""
            if not steps:
                return ""
            raw, merged = d.download_bytes()
            try:
                free = f"   ·   {rs.shutil.disk_usage(d.data_root).free / 1e9:.0f} GB free"
            except OSError:
                free = ""
            size = (f"   ·   ≈ {raw / 1e9:.1f} GB to download, {merged / 1e9:.1f} GB merged"
                    if rs.dm_particles(d.sim) and (raw or merged) else "")
            return f"{len(steps)} step{'s' if len(steps) != 1 else ''}{size}{free}"
        if tab == GRIDS:
            s = self.spec
            n = len(s.snapshots)
            est = f"full-resolution grids ≈ {s.grid_gb():.1f} GB" + ("  (float64)" if s.precision == "double" else "")
            if s.estimator == "ps" and s.slice_plane:
                est += f"   ·   slice {rs.plane_points(s.slice_plane)/1e6:.1f}M points"
            t = R.time_estimate(self._runs(), sim=s.sim, grid=s.grid, estimator=s.estimator,
                                gpu=s.gpu and rs.gpu_built(s.estimator),
                                sliced=bool(s.slice_plane), precision=s.precision)
            if t and n:
                est += (f"   ·   ≈ {R.format_duration(t[0])} per snapshot (median of {t[1]} earlier run"
                        f"{'s' if t[1] != 1 else ''}), ≈ {R.format_duration(t[0] * n)} in all")
            return (est + f"   ·   {n} snapshot{'s' if n != 1 else ''}"
                    + ("   ·   grids on scratch disk" if s.scratch_dir else ""))
        if tab == OWN:
            c = self.custom
            gb = rs.grids_gb(c.grid, c.fields if c.estimator == "ps" else rs.dtfe_fields(c.fields), c.precision)
            return (f"grids ≈ {gb:.1f} GB" + ("  (float64)" if c.precision == "double" else "")
                    + f"   ·   outputs: {c.output_root()}.*")
        if tab == PIPELINE:
            p = self.pipe
            parts = []
            for sim in p.sims:
                if rs.sim_dir(sim, p.data_root).is_dir():
                    todo, total = p.stale(sim)
                    parts.append(f"{sim}: {len(todo)} of {total} snapshots to compute")
            if p.adaptive and p.plan_key == p.memory_key():
                for sim in p.sims:
                    groups = p.adaptive_groups(sim)
                    if groups:
                        parts.append("adaptive: " + ", ".join(
                            f"{' '.join(f'{n:03d}' for n in snaps)} at {nu}²" for nu, snaps in groups.items()))
            else:
                shape = (f"plane {p.nu}²" + (f" × {p.planes} planes" if p.planes > 1 else "")
                         + (f" × {p.supersample}×{p.supersample} sub-samples" if p.supersample > 1 else "")
                         + (" in a window" if p.window.strip() else ""))
                parts.append(f"{shape} = {p.plane_points() / 1e6:.1f}M points")
            return "   ·   ".join(parts)
        return f"{len(steps)} script call{'s' if len(steps) != 1 else ''}" if steps else ""

    def closeEvent(self, event):
        if self.proc is not None:
            if QMessageBox.question(self, "Run in progress",
                                    "A run is still going. Stop it and quit?") != QMessageBox.Yes:
                event.ignore()
                return
            proc = self.proc
            self.stop_run()
            proc.waitForFinished(8000)
        if self._mcheck is not None:        # a running memory check: cancel its process tree, do not orphan it
            cproc = self._mcheck.get("proc")
            self._cancel_check()
            if cproc is not None:
                cproc.waitForFinished(2000)
        self.explore.shutdown()
        if self.remember:
            self._save()                            # Explore's state changes without a _changed()
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("splitter", self.splitter.saveState())
        event.accept()

    # ---------------------------------------------------------------- drag and drop
    def dragEnterEvent(self, event):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if urls and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if urls and urls[0].isLocalFile():
            self.open_dropped(urls[0].toLocalFile())
            event.acceptProposedAction()

    def open_dropped(self, path: str) -> str:
        """What a dropped path is for: a folder becomes the data root, a snapshot file (.hdf5) the Custom
        tab's input, a grid file opens in Explore. Returns 'root', 'custom', 'explore' or '' (says why)."""
        f = Path(path).expanduser()
        if f.is_dir():
            self.root_edit.setText(str(f))
            self._root_changed()
            self.status_message(f"data root: {f}")
            return "root"
        if f.suffix.lower() in (".hdf5", ".h5", ".hdf"):
            self.tabs.setCurrentIndex(OWN)
            self.o_file.setText(str(f))
            self._own_file_changed()
            self.status_message(f"custom snapshot: {f.name}")
            return "custom"
        if f.is_file() and self.explore.open_file(str(f)):
            self.tabs.setCurrentIndex(EXPLORE)
            return "explore"
        self.status_message(f"{f.name}: not a folder, a snapshot (.hdf5) or a grid file the launcher can show")
        return ""

    # ================================================================ helpers for the new panels
    def status_message(self, text: str):
        self.status.setText(text)

    def _poll_machine(self):
        """Start one background reading of the machine (memory, swap, DTFE processes, their progress files)."""
        if self._machine_busy:
            return
        self._machine_busy = True
        root, extra, relay = self.root_edit.text().strip(), self._custom_output_dirs(), self._machine_relay

        def work():
            res = {}
            try:
                res["mem"] = rs.memory_status()
                res["procs"] = rs.dtfe_processes(0.0)
                res["running"] = rs.running_now(rs.progress_files(root, extra), res["procs"])
            except Exception as e:          # a reading that fails shows as unknown, never kills the poll
                res["error"] = str(e)
            relay.done.emit(res)
        import threading
        threading.Thread(target=work, daemon=True).start()

    def _machine_known(self, res: dict):
        self._machine_busy = False
        self.machine = res
        try:
            self.mem_strip.setText(rs.memory_text(res.get("mem") or {}, res.get("procs")))
            self.running_now.show_entries(res.get("running") or [])
        except RuntimeError:                # the window went while the reading was under way
            pass

    def _runs(self) -> list[R.RunRecord]:
        """The run logs on disk, for the time estimates (rescanned at most every 30 s)."""
        if time.monotonic() - self._runs_scanned > 30:
            self.runs_cache = R.scan_runs(self.root_edit.text().strip(), self._custom_output_dirs())
            self._runs_scanned = time.monotonic()
        return self.runs_cache

    def _custom_output_dirs(self) -> list[str]:
        dirs = list(self.custom_dirs)
        cur = str(Path(self.custom.output_dir).expanduser()) if self.custom.output_dir else ""
        if cur and cur not in dirs:
            dirs.append(cur)
        return [d for d in dirs if Path(d).is_dir()]

    def _register_custom_output(self, spec: rs.CustomSpec, final: bool = False):
        """A custom-snapshot job is starting (or, 'final', has finished): remember its folder, and keep
        its settings beside the outputs ('<name>.gui.json') -- the Explore tab's query server reads the
        snapshot from them, the figure grid's Type column its estimator. At the start the file is written
        only when there is none yet: a re-run under the same name that is stopped or fails leaves the
        grids of the old run on disk, and its sidecar must keep describing THOSE."""
        root = spec.output_root()
        d = str(root.parent)
        if d not in self.custom_dirs:
            self.custom_dirs.append(d)
            self._save()
        side_path = Path(str(root) + ".gui.json")
        if not final and (side_path.is_file() or any(Path(str(root) + sfx).is_file() for sfx in (".a_den", ".den"))):
            return                              # grids already there (with or without a sidecar): the finish describes them
        side = spec.settings_sidecar()
        tmp = Path(str(side_path) + ".tmp")
        try:
            root.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(side, indent=1))
            os.replace(tmp, side_path)          # never a truncated sidecar
        except OSError as e:
            self.log.appendPlainText(f"!! cannot write the settings sidecar {side_path.name}: {e}")

    def _load_run_settings(self, rec: R.RunRecord):
        """The Runs browser's 'Load settings': a TNG run's options into the Grids tab; a custom-snapshot run's
        (its settings sidecar) into the Custom tab."""
        if not rec.sim:
            side = R.custom_settings(rec)
            if side is None:
                self.status_message(f"{rec.prefix}: no settings file beside its log (made before 2026-09-30?)")
                return
            self.custom = rs.CustomSpec.from_dict(side)
            self._load_into_widgets()
            self.tabs.setCurrentIndex(OWN)
            self._changed()
            self.status_message(f"loaded the settings of the custom run {rec.prefix}")
            return
        for k, v in rec.settings().items():
            if hasattr(self.spec, k):
                setattr(self.spec, k, list(v) if isinstance(v, list) else v)
        self.spec.sim = rec.sim
        self.spec.snapshots = [rec.snap] if rec.snap is not None else []
        self._load_into_widgets()
        self._fill_sim_dependent()
        self.tabs.setCurrentIndex(GRIDS)
        self._changed()
        self.status_message(f"loaded the settings of {rec.sim} {rec.snap:03d} ({rec.prefix})"
                            if rec.snap is not None else f"loaded the settings of {rec.prefix}")

    def _explore_output(self, root: str):
        self.tabs.setCurrentIndex(EXPLORE)
        self.explore.select_root(root)

    # ================================================================ persistence
    def _save(self):
        if not self.remember:
            return
        self.settings.setValue("spec", json.dumps(self.spec.to_dict()))
        self.settings.setValue("pipeline", json.dumps(self.pipe.to_dict()))
        self.settings.setValue("plots", json.dumps(self.plots.to_dict()))
        self.settings.setValue("data", json.dumps(self.data.to_dict()))
        self.settings.setValue("custom", json.dumps(self.custom.to_dict()))
        self.settings.setValue("custom_dirs", json.dumps(self.custom_dirs))
        self.settings.setValue("presets", json.dumps(self.user_presets))
        self.settings.setValue("show_commands", json.dumps(self.show_commands))
        self.settings.setValue("tab_name", TAB_NAMES[self.tabs.currentIndex()])
        if self.explore.current is not None:
            self.settings.setValue("explore", json.dumps(self.explore.state()))

    def _restore_json(self, key: str, default):
        raw = self.settings.value(key, "") if self.remember else ""
        try:
            return json.loads(raw) if raw else default
        except (ValueError, TypeError):
            return default

    def _restore(self, key: str, cls):
        raw = self.settings.value(key, "") if self.remember else ""
        try:
            return cls.from_dict(json.loads(raw)) if raw else cls()
        except (ValueError, TypeError):
            return cls()


def _start_times(pids: list[int]) -> dict[int, str] | None:
    """{pid: start time as ps prints it} for the live ones; None when ps cannot say."""
    if not pids:
        return {}
    try:
        out = subprocess.run(["ps", "-o", "pid=,lstart=", "-p", ",".join(map(str, pids))],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    times = {}
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            times[int(parts[0])] = parts[1].strip()
    return times


def _kill_leftovers(pids: list[int], started: dict[int, str] | None = None):
    """SIGKILL what is left of a stopped run's process tree: only a pid that still is the SAME process (its
    start time unchanged since Stop) -- a pid freed and reused meanwhile belongs to someone else."""
    now = _start_times(pids) if started is not None else None
    for pid in pids:
        if started is not None and now is not None and (pid not in started or now.get(pid) != started[pid]):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _descendants(pid: int) -> list[int]:
    """pid and every process below it (the script's PS-DTFE child must be stopped too)."""
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        try:
            kids = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()
        except OSError:
            kids = []
        todo.extend(int(k) for k in kids)
    return out


def smoke_report(w: MainWindow) -> str:
    """What the window's jobs would use -- for checking a launcher's environment (--smoke)."""
    def deps(py: str) -> str:
        try:
            r = subprocess.run([py, "-c", "import numpy, h5py, matplotlib, scipy"], capture_output=True, timeout=60)
            return "numpy/h5py/matplotlib/scipy ok" if r.returncode == 0 else "MISSING analysis packages"
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"not runnable ({e})"
    root = w.root_edit.text()
    return "\n".join([
        f"python3 for the scripts: {rs.PYTHON} ({deps(rs.PYTHON)})",
        f"wget: {rs.shutil.which('wget') or 'not found'}",
        f"TNG API key available: {rs.api_key_available()}",
        f"data root: {root or '(none chosen)'} ({'found' if rs.root_ok(root) else 'NOT found'}), "
        f"{w.sim_combo.count()} simulations",
        f"PS-DTFE: {'built' if (rs.REPO_ROOT / 'PS-DTFE').is_file() else 'NOT built'}, "
        f"GPU {'yes' if rs.gpu_built('ps') else 'no'}, parallel triangulation (TBB) {'yes' if rs.tbb_built() else 'no'}; "
        f"DTFE: {'built' if (rs.REPO_ROOT / 'DTFE').is_file() else 'NOT built'}, GPU {'yes' if rs.gpu_built('dtfe') else 'no'}",
        f"icon: {os.environ.get('DTFE_GUI_ICON', '(none)')}",
    ])


def _macos_app_name(name: str) -> bool:
    """Make the macOS menu bar (and its Hide / Quit items) say `name` instead of "Python".

    macOS takes that name from the main bundle's CFBundleName, and this process's main bundle is
    Python.app; overwriting the entry in the bundle's (mutable) info dictionary before the
    application object exists is enough. Plain ctypes on the Objective-C runtime, so no PyObjC.
    Returns whether the name is now set."""
    if sys.platform != "darwin":
        return False
    try:
        import ctypes
        import ctypes.util
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = objc.sel_registerName.argtypes = [ctypes.c_char_p]
        vp = ctypes.c_void_p

        def send(receiver, selector: bytes, *args, argtypes=()):
            # a typed prototype per call shape: objc_msgSend is variadic-by-convention, and on
            # arm64 the argument types must be exact
            fn = ctypes.CFUNCTYPE(vp, vp, vp, *argtypes)(("objc_msgSend", objc))
            return fn(receiver, objc.sel_registerName(selector), *args)

        def nsstring(text: str):
            return send(objc.objc_getClass(b"NSString"), b"stringWithUTF8String:", text.encode(),
                        argtypes=(ctypes.c_char_p,))

        info = send(send(objc.objc_getClass(b"NSBundle"), b"mainBundle"), b"infoDictionary")
        mutable = objc.objc_getClass(b"NSMutableDictionary")
        is_kind = ctypes.CFUNCTYPE(ctypes.c_bool, vp, vp, vp)(("objc_msgSend", objc))
        if not info or not is_kind(info, objc.sel_registerName(b"isKindOfClass:"), mutable):
            return False                  # never message an immutable dictionary: that would abort
        send(info, b"setObject:forKey:", nsstring(name), nsstring("CFBundleName"), argtypes=(vp, vp))
        return True
    except (OSError, AttributeError, TypeError):
        return False


def main():
    # the Dock and Cmd-Tab name a process after its BUNDLE, which scripts/install_gui_app.sh's GUI
    # host provides; without the host (plain interpreter) this at least fixes the menu bar
    _macos_app_name("Lamina")                   # before QApplication builds the menu bar
    app = QApplication(sys.argv)
    app.setApplicationName("Lamina")
    app.setApplicationDisplayName("Lamina")
    icon = os.environ.get("DTFE_GUI_ICON", "")          # set by the app bundle / .desktop launcher
    if icon and Path(icon).is_file():
        app.setWindowIcon(QIcon(icon))
    w = MainWindow()
    if "--smoke" in sys.argv[1:]:                        # build the window, report, quit
        print(smoke_report(w), flush=True)
        sys.exit(0)
    app.aboutToQuit.connect(w.explore.shutdown)
    w.show()
    # first launch: the setup dialog -- unless this is an existing, working installation (settings
    # remembered, data folder present, programs built), which it would only interrupt
    working = (bool(w.settings.value("spec")) and rs.root_ok(w.root_edit.text())
               and (rs.REPO_ROOT / "PS-DTFE").is_file())
    if "--setup" in sys.argv[1:] or (not w.settings.value("setup_done") and not working):
        QTimer.singleShot(300, w.run_setup)
    sys.exit(app.exec())


if __name__ == "__main__":
    # Run as a script (scripts/gui.sh), this file is '__main__'; the mixin modules import it as 'app', a second
    # copy of every name. Start the window from that one module, so the window and its mixins share one copy.
    import app
    app.main()
