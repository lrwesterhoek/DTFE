#!/usr/bin/env python3
"""DTFE / PS-DTFE launcher (PySide6).

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
from collections import Counter
import signal
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QLocale, QProcess, QProcessEnvironment, QRect, QSettings, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QActionGroup, QColor, QDesktopServices, QFont, QFontDatabase, QIcon,
                           QImageReader, QIntValidator, QKeySequence, QPixmap, QShortcut)
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QAbstractItemView, QAbstractSpinBox, QHeaderView,
    QSizePolicy, QSpinBox,
    QSplitter, QStackedWidget, QStyle, QStyledItemDelegate, QStyleFactory, QStyleOptionButton,
    QStyleOptionViewItem, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runspec as rs  # noqa: E402
import presets as P  # noqa: E402
import results as R  # noqa: E402
import grids as G  # noqa: E402

ERROR_COLOR, WARN_COLOR, OK_COLOR = QColor("#c62828"), QColor("#b26a00"), QColor("#2e7d32")
INFO_COLOR = QColor("gray")
CHECK_ICON = {"error": "✖  ", "warning": "⚠  ", "ok": "✔  ", "info": "ℹ  ", "hint": "→  "}
CHECK_COLOR = {"error": ERROR_COLOR, "warning": WARN_COLOR, "ok": OK_COLOR, "info": INFO_COLOR,
               "hint": INFO_COLOR}
DATA, GRIDS, OWN, PIPELINE, PLOTS, EXPLORE = range(6)
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


def _mono() -> QFont:
    f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    f.setPointSize(11)
    return f


def _note(text: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color: gray")
    return lab


class CheckDelegate(QStyledItemDelegate):
    """Draws the tick boxes of a checkable list or table itself. The native macOS style (Qt 6, dark
    appearance) paints the check indicator of the CURRENT row only, so every other row looked as if
    it had none. Fusion's indicator, drawn over the rectangle the view's own style reserves for it
    (so a click there still toggles the item), shows in both appearances."""
    _fusion = None

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        state = index.data(Qt.CheckStateRole)
        if state is None or not (index.flags() & Qt.ItemIsUserCheckable):
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = option.widget
        style = widget.style() if widget is not None else QApplication.style()
        area = style.subElementRect(QStyle.SubElement.SE_ItemViewItemCheckIndicator, opt, widget)
        side = min(14, area.width(), area.height())
        if side <= 0:
            return
        if CheckDelegate._fusion is None:
            CheckDelegate._fusion = QStyleFactory.create("Fusion")
        box = QStyleOptionButton()
        box.rect = QRect(area.x() + (area.width() - side) // 2, area.y() + (area.height() - side) // 2, side, side)
        box.palette = option.palette
        box.state = QStyle.StateFlag.State_On if Qt.CheckState(state) == Qt.Checked else QStyle.StateFlag.State_Off
        if option.state & QStyle.StateFlag.State_Enabled:
            box.state |= QStyle.StateFlag.State_Enabled
        painter.save()
        # clear the whole indicator column -- the item's left edge up to its text -- first: the native
        # indicator, where it is drawn, reaches a little outside the rectangle the style reports
        text = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, widget)
        strip = QRect(option.rect.left(), option.rect.top(), max(text.left() - option.rect.left(), area.width()),
                      option.rect.height())
        painter.fillRect(strip, option.palette.highlight() if option.state & QStyle.StateFlag.State_Selected
                         else option.palette.base())
        CheckDelegate._fusion.drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorCheckBox, box, painter, None)
        painter.restore()


def _checkable(view):
    """A list/table whose items carry tick boxes: our own indicator (see CheckDelegate)."""
    view.setItemDelegate(CheckDelegate(view))
    return view


def _grow_fields(root: QWidget):
    """Every form's fields take the width they are given. macOS's default form policy keeps each
    field at its size hint, which squeezed paths and names into a few characters."""
    for f in root.findChildren(QFormLayout):
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    for sb in root.findChildren(QAbstractSpinBox):      # numbers stay compact
        sb.setSizePolicy(QSizePolicy.Policy.Fixed, sb.sizePolicy().verticalPolicy())


def _scroll(inner: QWidget) -> QScrollArea:
    s = QScrollArea()
    s.setWidget(inner)
    s.setWidgetResizable(True)
    s.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return s


def _dir_row(edit: QLineEdit, on_browse) -> QHBoxLayout:
    row = QHBoxLayout()
    b = QToolButton(text="…")
    b.clicked.connect(on_browse)
    row.addWidget(edit)
    row.addWidget(b)
    return row


def _checked(lst: QListWidget) -> list:
    return [lst.item(i).data(Qt.UserRole) for i in range(lst.count())
            if lst.item(i).checkState() == Qt.Checked]


def _grid_combo(parent) -> QComboBox:
    c = QComboBox()
    c.setEditable(True)
    c.addItems(["128", "256", "512", "1024"])
    c.setValidator(QIntValidator(8, 4096, parent))
    return c


def _state_text(chunks: int, merged: bool) -> str:
    if merged:
        return "merged" + (f" (+{chunks} chunks)" if chunks else "")
    return f"{chunks} chunks" if chunks else "—"


def _int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        return 0


def _hint(level: str, msg: str) -> tuple[str, str]:
    """A selection-still-missing error as a neutral hint (see HINTS)."""
    for key, text in HINTS.items():
        if level == "error" and msg.startswith(key):
            return "hint", text + msg[len(key):]
    return level, msg


# ==================================================================== figure browser
class FigureBrowser(QWidget):
    """Figures under python/figures: the ones the last run wrote, or all of them, with a preview."""
    changed = Signal()          # the selection (what Open / Show in Finder act on): the menu bar follows

    def __init__(self):
        super().__init__()
        QImageReader.setAllocationLimit(4096)       # 8192^2 point-eval PNGs exceed Qt's 256 MB default
        self.roots: list[Path] = [rs.FIGURES_ROOT]
        self.since: float | None = None
        self._current: Path | None = None

        top = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["From the last run", "All figures"])
        self.mode.currentIndexChanged.connect(self.reload)
        self.filter = QLineEdit(placeholderText="filter: words in the path, e.g. TNG100 snap099 density")
        self.filter.textChanged.connect(self.reload)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.reload)
        top.addWidget(self.mode)
        top.addWidget(self.filter, 1)
        top.addWidget(self.refresh_btn)

        self.list = QListWidget()
        self.list.setTextElideMode(Qt.ElideMiddle)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.currentItemChanged.connect(lambda cur, _prev: self._show(cur))
        self.list.itemDoubleClicked.connect(lambda _it: self.open_current())
        self.preview = QLabel("", alignment=Qt.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview.setMinimumSize(200, 200)
        self.path_label = _note()
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        btns = QHBoxLayout()
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self.open_current)
        self.reveal_btn = QPushButton("Show in Finder")
        self.reveal_btn.clicked.connect(self.reveal_current)
        btns.addWidget(self.open_btn)
        btns.addWidget(self.reveal_btn)
        btns.addStretch(1)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.preview, 1)
        rv.addWidget(self.path_label)
        rv.addLayout(btns)
        split = QSplitter()
        split.addWidget(self.list)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

        v = QVBoxLayout(self)
        v.addLayout(top)
        v.addWidget(split, 1)
        self._show(None)

    def show_run(self, since: float, extra_roots=()) -> int:
        """List what a run started at `since` wrote; returns how many figures that is."""
        self.since = since
        self.roots = [rs.FIGURES_ROOT] + [Path(r) for r in extra_roots if Path(r) != rs.FIGURES_ROOT]
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(0)
        self.mode.blockSignals(False)
        return self.reload()

    def reload(self, *_) -> int:
        from_run = self.mode.currentIndex() == 0
        if from_run and self.since is None:
            files = []
        else:
            files = rs.figures(self.roots, self.since if from_run else None)
        words = self.filter.text().lower().split()
        if words:
            files = [p for p in files if all(w in str(p).lower() for w in words)]
        keep = str(self._current) if self._current else None
        self.list.blockSignals(True)
        self.list.clear()
        for p in files:
            rel = next((str(p.relative_to(r.resolve())) for r in self.roots
                        if p.is_relative_to(r.resolve())), str(p))
            it = QListWidgetItem(rel)
            it.setData(Qt.UserRole, str(p))
            it.setToolTip(f"{p}\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))}")
            self.list.addItem(it)
        self.list.blockSignals(False)
        row = next((i for i in range(self.list.count()) if self.list.item(i).data(Qt.UserRole) == keep), 0)
        if self.list.count():
            self.list.setCurrentRow(row)
            self._show(self.list.currentItem())
        else:
            self._show(None)
        return len(files)

    def _show(self, item):
        self._current = Path(item.data(Qt.UserRole)) if item is not None else None
        for b in (self.open_btn, self.reveal_btn):
            b.setEnabled(self._current is not None)
        self.changed.emit()
        if self._current is None:
            from_run = self.mode.currentIndex() == 0
            self.preview.setText("No figures from the last run yet." if from_run else "No figures found.")
            self.preview.setPixmap(QPixmap())
            self.path_label.setText("")
            return
        self.path_label.setText(str(self._current))
        self._render_preview()

    def _render_preview(self):
        p = self._current
        if p is None:
            return
        reader = QImageReader(str(p))
        size = reader.size()
        box = self.preview.size()
        if not reader.canRead() or not size.isValid():
            self.preview.setPixmap(QPixmap())
            self.preview.setText(f"{p.suffix.upper()[1:]}: no preview (Open shows it)")
            return
        reader.setScaledSize(size.scaled(QSize(max(box.width(), 50), max(box.height(), 50)),
                                         Qt.KeepAspectRatio))
        img = reader.read()
        self.preview.setPixmap(QPixmap.fromImage(img) if not img.isNull() else QPixmap())
        if img.isNull():
            self.preview.setText(f"could not read: {reader.errorString()}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        QTimer.singleShot(0, self._render_preview)

    def open_current(self):
        if self._current is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current)))

    def reveal_current(self):
        if self._current is not None:
            if sys.platform == "darwin":
                subprocess.run(["open", "-R", str(self._current)])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current.parent)))


# ==================================================================== main window
class MainWindow(QMainWindow):
    def __init__(self, spec: rs.RunSpec | None = None, remember: bool = True,
                 pipeline: rs.PipelineSpec | None = None, plots: rs.PlotSpec | None = None,
                 data: rs.DataSpec | None = None, settings: QSettings | None = None,
                 custom: rs.CustomSpec | None = None):
        QLocale.setDefault(QLocale.c())      # spin boxes read and show '2.00', as every label does (not the system's
        super().__init__()                   # '2,00'); BEFORE the window exists: children inherit the parent's locale
        self.setLocale(QLocale.c())
        self.setWindowTitle("DTFE / PS-DTFE launcher")
        self.remember = remember
        self.settings = settings or QSettings("DTFE", "launcher")
        self.spec = spec or self._restore("spec", rs.RunSpec)
        self.pipe = pipeline or self._restore("pipeline", rs.PipelineSpec)
        self.plots = plots or self._restore("plots", rs.PlotSpec)
        self.data = data or self._restore("data", rs.DataSpec)
        self.custom = custom or self._restore("custom", rs.CustomSpec)
        if custom is None and not self._restore_json("custom", None):
            self.custom.gpu = rs.gpu_built("ps")            # a fresh custom-snapshot job uses the GPU when built
        # where custom-snapshot jobs wrote outputs (the Runs browser and Explore look there too)
        self.custom_dirs: list[str] = [d for d in self._restore_json("custom_dirs", []) if isinstance(d, str)]
        self.user_presets: dict = self._restore_json("presets", {})
        self.show_commands = str(self._restore_json("show_commands", False)).lower() == "true"
        self.runs_cache: list[R.RunRecord] = []            # for the time estimates
        self._runs_scanned = 0.0
        self._job_text = ""                                  # this job's output, for the failure advice
        self._tee = None                                     # the open run-log copy of the current step
        self._after_job = None                               # e.g. the demo: open Explore when it finishes
        self.proc: QProcess | None = None
        self._loading = False
        self._own_last_file = ""        # the custom file whose particle count last set the GPU box
        self._steps: list[rs.Step] = []
        self._step_i, self._results = -1, []
        self._job_tab, self._job_start = GRIDS, None
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
        self.explore = X.ExploreControls(self.explore_view, lambda: self.root_edit.text().strip(),
                                         self._custom_output_dirs)
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
                                            # at 590 the third column read 've', 'vo', 'V-' (seen 2026-10-05)

        self.runs = RV.RunsBrowser(lambda: self.root_edit.text().strip(), self._custom_output_dirs)
        self.runs.load_settings.connect(self._load_run_settings)
        self.runs.explore.connect(self._explore_output)
        self.runs.message.connect(self.status_message)
        self.runs.changed.connect(lambda: self._sync_runs_actions())
        self.right = QStackedWidget()
        self.right.addWidget(self._build_run_panel())
        self.right.addWidget(self.explore_view)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.resize(1400, 900)

        self._load_into_widgets()
        self._read_widgets()            # the shared data (root, simulation, snapshots) into every job
        # a fresh window opens on Grids; a remembered one where it was left (by name: tab indices
        # moved when tabs were added)
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
        # re-check every few seconds while idle: jobs started or finished in a terminal, the T7
        # plugged in or out
        self._poll = QTimer(self, interval=4000)
        self._poll.timeout.connect(lambda: self.proc is None and self.refresh())
        self._poll.start()

    # ================================================================ shared data panel
    def _build_data(self) -> QWidget:
        g = QGroupBox("Data")
        f = QFormLayout(g)
        self._data_form = f
        self.root_edit = QLineEdit()
        self.root_edit.editingFinished.connect(self._root_changed)
        f.addRow("Data root", _dir_row(self.root_edit, self._browse_root))
        self.sim_combo = QComboBox()
        self.sim_combo.currentTextChanged.connect(self._sim_changed)
        f.addRow("Simulation", self.sim_combo)
        self.snap_list = QListWidget()
        self.snap_list.setFont(_mono())
        self.snap_list.setMinimumHeight(150)
        self.snap_list.setMaximumHeight(190)
        self.snap_list.itemChanged.connect(self._changed)
        f.addRow("Snapshots", self.snap_list)
        sel = QHBoxLayout()
        self.all_btn, self.none_btn, self.mark_btn = QPushButton("All"), QPushButton("None"), QPushButton()
        self.all_btn.clicked.connect(lambda: self._select_snaps("all"))
        self.none_btn.clicked.connect(lambda: self._select_snaps("none"))
        self.mark_btn.clicked.connect(lambda: self._select_snaps("mark"))
        for b in (self.all_btn, self.none_btn, self.mark_btn):
            sel.addWidget(b)
        sel.addStretch(1)
        self._snap_buttons = sel
        f.addRow("", sel)
        return g

    # ================================================================ Data tab
    def _build_data_tab(self) -> QWidget:
        col = QVBoxLayout()
        g = QGroupBox("Simulation")
        f = QFormLayout(g)
        self.d_sim = QComboBox()
        self.d_sim.setEditable(True)
        self.d_sim.setInsertPolicy(QComboBox.NoInsert)
        self.d_sim.activated.connect(self._data_sim_changed)
        self.d_sim.lineEdit().editingFinished.connect(self._data_sim_changed)
        f.addRow("Name", self.d_sim)
        self.d_target = _note()
        f.addRow("", self.d_target)
        col.addWidget(g)

        g = QGroupBox("Snapshots")
        v = QVBoxLayout(g)
        self.d_table = QTableWidget(0, 4)
        self.d_table.setHorizontalHeaderLabels(["Snapshot", "z", "Particles", "Group catalogue"])
        self.d_table.verticalHeader().setVisible(False)
        self.d_table.verticalHeader().setDefaultSectionSize(22)
        self.d_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.d_table.setSelectionMode(QAbstractItemView.NoSelection)
        head = self.d_table.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.Stretch)
        for c in (0, 1):                            # snapshot number and z: as wide as they need
            head.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.d_table.setMinimumHeight(230)
        self.d_table.itemChanged.connect(self._changed)
        v.addWidget(self.d_table, 1)        # grows with the window (it showed ten rows above empty space)
        sel = QHBoxLayout()
        for label, mode in (("All", "all"), ("None", "none"), ("Not merged", "missing")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, m=mode: self._select_data(m))
            sel.addWidget(b)
        self.d_other = QLineEdit(placeholderText="add snapshots, e.g. 59 72")
        self.d_other.editingFinished.connect(self._add_data_snapshots)
        sel.addWidget(self.d_other, 1)
        v.addLayout(sel)
        self.d_whole = _note()
        v.addWidget(self.d_whole)
        col.addWidget(g, 1)

        g = QGroupBox("Download from the TNG API")
        v = QVBoxLayout(g)
        self.d_dl_snap = QCheckBox("Particle snapshots")
        self.d_dl_gc = QCheckBox("Group catalogues")
        self.d_dl_tree = QCheckBox("Merger trees (whole simulation)")
        self.d_dl_ics = QCheckBox("Initial conditions (whole simulation)")
        for cb, tip in ((self.d_dl_snap, "For the ticked snapshots"),
                        (self.d_dl_gc, "For the ticked snapshots"),
                        (self.d_dl_tree, "One set for the whole simulation"),
                        (self.d_dl_ics, "One set for the whole simulation")):
            cb.setToolTip(tip)
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        col.addWidget(g)

        g = QGroupBox("Merge into combined files")
        v = QVBoxLayout(g)
        self.d_mg_snap = QCheckBox("Snapshots")
        self.d_mg_gc = QCheckBox("Group catalogues")
        self.d_mg_tree = QCheckBox("Merger trees")
        self.d_mg_ics = QCheckBox("Initial conditions")
        self.d_delete = QCheckBox("Delete the raw chunks after a verified merge")
        for cb, tip in ((self.d_mg_snap, "Writes combined_NNN.hdf5 (h-free units)"),
                        (self.d_mg_gc, "Writes combined_fof_subhalo_tab_NNN.hdf5"),
                        (self.d_mg_tree, "Writes combined_tree_extended.hdf5"),
                        (self.d_mg_ics, "Writes combined_ics.hdf5"),
                        (self.d_delete, "The chunks are the only copy of the fields the merge drops "
                                        "(potential, Subfind densities)")):
            cb.setToolTip(tip)
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        col.addWidget(g)
        # (no trailing stretch: the snapshot table above takes the room a taller window gives)
        w = QWidget()
        w.setLayout(col)
        return w

    def _data_sim_changed(self, *_):
        if self._loading:
            return
        self.data.sim = self.d_sim.currentText().strip()
        self.data.snapshots = []
        self._fill_data_table()
        self._changed()

    def _fill_data_table(self):
        was = self._loading
        self._loading = True
        sim, root = self.d_sim.currentText().strip(), self.root_edit.text().strip()
        wanted = set(self.data.snapshots)
        self.d_table.setRowCount(0)
        self.d_target.setText("")
        self.d_whole.setText("")
        if sim and Path(root).is_dir():
            sp = rs.sim_dir(sim, root)
            n_part = rs.dm_particles(sim)
            size = (f"  ·  {round(n_part ** (1 / 3))}³ particles, {n_part * rs.MERGED_BYTES_PER_PARTICLE / 1e9:.1f} GB "
                    f"per snapshot") if n_part else ""
            try:
                where = str(sp.relative_to(Path(root)))
            except ValueError:
                where = str(sp)
            self.d_target.setText(where + ("  (new folder)" if not sp.is_dir() else "") + size)
            self.d_target.setToolTip(f"{sp}\n~{n_part * rs.RAW_BYTES_PER_PARTICLE / 1e9:.1f} GB to download, "
                                     f"{n_part * rs.MERGED_BYTES_PER_PARTICLE / 1e9:.1f} GB merged, per snapshot"
                                     if n_part else str(sp))
            for n in sorted(set(rs.data_snapshots(sp)) | wanted):
                st = rs.snapshot_state(sp, n)
                z = rs.SNAPSHOT_LADDER.get(n)
                if z is None and st["merged"]:
                    z = rs._redshift(sp / f"snapdir_{n:03d}" / f"combined_{n:03d}.hdf5")
                row = self.d_table.rowCount()
                self.d_table.insertRow(row)
                first = QTableWidgetItem(f"{n:03d}")
                first.setData(Qt.UserRole, n)
                first.setData(Qt.UserRole + 1, st["merged"])
                first.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                first.setCheckState(Qt.Checked if n in wanted else Qt.Unchecked)
                cells = [first, QTableWidgetItem("?" if z is None else f"{z:.2f}"),
                         QTableWidgetItem(_state_text(st["chunks"], st["merged"])),
                         QTableWidgetItem(_state_text(st["gc_chunks"], st["gc_merged"]))]
                for col, (item, (chunks, merged)) in enumerate(zip(
                        cells, [(0, False), (0, False), (st["chunks"], st["merged"]),
                                (st["gc_chunks"], st["gc_merged"])])):
                    if col >= 2:
                        item.setForeground(MERGED_COLOR if merged else CHUNKS_COLOR if chunks else QColor("gray"))
                    if col:
                        item.setFlags(Qt.ItemIsEnabled)
                    self.d_table.setItem(row, col, item)
            w = rs.sim_state(sp)
            self.d_whole.setText(
                f"Merger trees: {_state_text(w['tree_chunks'], w['tree_merged'])}   ·   Initial conditions: "
                + ("converted" if w["ics_converted"] else "downloaded, not converted" if w["ics_raw"] else "—"))
        self._loading = was

    def _select_data(self, mode: str):
        self._loading = True
        for row in range(self.d_table.rowCount()):
            it = self.d_table.item(row, 0)
            on = mode == "all" or (mode == "missing" and not it.data(Qt.UserRole + 1))
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self._loading = False
        self._changed()

    def _add_data_snapshots(self):
        extra = {int(t) for t in self.d_other.text().replace(",", " ").split() if t.isdigit() and int(t) <= 99}
        if not extra:
            return
        self.d_other.clear()
        self._read_widgets()
        self.data.snapshots = sorted(set(self.data.snapshots) | extra)
        self._fill_data_table()
        self._changed()

    # ================================================================ presets
    def _preset_group(self, target: str) -> QGroupBox:
        """'Start from': the built-in and saved presets for the Grids ('grids') or Custom snapshot ('custom') job."""
        g = QGroupBox("Start from a preset")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        combo = QComboBox()
        combo.setToolTip("Sets the grid, fields and options below; everything stays editable")
        save = QPushButton("Save as preset…")
        delete = QPushButton("Delete")
        row.addWidget(combo, 1)
        row.addWidget(save)
        row.addWidget(delete)
        v.addLayout(row)
        about = _note()
        about.hide()
        v.addWidget(about)
        combo.activated.connect(lambda _i: self._apply_preset(target))
        save.clicked.connect(lambda: self._save_preset(target))
        delete.clicked.connect(lambda: self._delete_preset(target))
        setattr(self, f"_preset_{target}", (combo, delete, about))
        self._fill_presets(target)
        return g

    def _fill_presets(self, target: str):
        combo, delete, about = getattr(self, f"_preset_{target}")
        keep = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("choose…", "")
        for name in P.preset_names(self.user_presets):
            combo.addItem(name, name)
        combo.setCurrentIndex(max(combo.findData(keep), 0))
        combo.blockSignals(False)
        delete.setEnabled(bool(combo.currentData()) and combo.currentData() not in P.BUILTIN_PRESETS)
        about.setText("")
        about.hide()

    def _apply_preset(self, target: str):
        combo, delete, about = getattr(self, f"_preset_{target}")
        name = combo.currentData()
        delete.setEnabled(bool(name) and name not in P.BUILTIN_PRESETS)
        found = P.preset(name, self.user_presets) if name else None
        if not found:
            about.setText("")
            about.hide()
            return
        text, settings = found
        self._read_widgets()
        spec = self.spec if target == "grids" else self.custom
        planes = rs.planes(self._sim_path()) if (target == "grids" and self._sim_path()) else []
        changed = P.apply_preset(spec, settings, planes)
        if "gpu" not in settings:          # the preset leaves the GPU to the run's size (see recommended_gpu)
            n = (rs.dm_particles(spec.sim) if target == "grids" else
                 rs.hdf5_particle_count(Path(spec.input_file).expanduser()) if spec.input_file else None)
            rec = rs.recommended_gpu(spec.deposit, spec.fields, n, spec.estimator, spec.precision)
            if spec.gpu != rec:
                spec.gpu = rec
                changed.append("gpu")
        if target == "grids" and settings.get("slice_plane") == "largest" and not planes:
            text += "  (no image plane for this simulation yet: make one with the Pipeline tab)"
        about.setText(text)
        about.show()
        self._load_into_widgets()
        self._changed()
        self.status_message(f"preset '{name}': {len(changed)} setting{'s' if len(changed) != 1 else ''} changed")

    def _save_preset(self, target: str):
        name, ok = QInputDialog.getText(self, "Save as preset", "Name for the current settings:")
        name = name.strip()
        if not ok or not name:
            return
        if name in P.BUILTIN_PRESETS:
            QMessageBox.information(self, "Save as preset", f"'{name}' is a built-in preset; choose another name.")
            return
        self._read_widgets()
        self.user_presets[name] = P.capture(self.spec if target == "grids" else self.custom)
        self._save()
        for t in ("grids", "custom"):
            self._fill_presets(t)
        combo = getattr(self, f"_preset_{target}")[0]
        combo.setCurrentIndex(combo.findData(name))
        self.status_message(f"saved preset '{name}'")

    def _delete_preset(self, target: str):
        combo = getattr(self, f"_preset_{target}")[0]
        name = combo.currentData()
        if name and name in self.user_presets:
            del self.user_presets[name]
            self._save()
            for t in ("grids", "custom"):
                self._fill_presets(t)

    # ================================================================ Grids tab
    def _build_grids_tab(self) -> QWidget:
        col = QVBoxLayout()
        col.addWidget(self._preset_group("grids"))
        g = QGroupBox("Estimator and grid")
        f = QFormLayout(g)
        est = QHBoxLayout()
        self.ps_radio = QRadioButton("PS-DTFE (phase space)")
        self.dtfe_radio = QRadioButton("DTFE (standard)")
        grp = QButtonGroup(self)
        grp.addButton(self.ps_radio)
        grp.addButton(self.dtfe_radio)
        self.ps_radio.toggled.connect(self._changed)
        est.addWidget(self.ps_radio)
        est.addWidget(self.dtfe_radio)
        est.addStretch(1)
        f.addRow("Estimator", est)
        self.deposit_combo = QComboBox()
        self.deposit_combo.addItem(P.label("sampled"), "sampled")
        self.deposit_combo.addItem(P.label("exact"), "exact")
        self.deposit_combo.setItemData(0, P.tooltip("sampled"), Qt.ToolTipRole)
        self.deposit_combo.setItemData(1, P.tooltip("exact"), Qt.ToolTipRole)
        self.deposit_combo.currentIndexChanged.connect(
            lambda _i: self._deposit_changed(self.deposit_combo, self.gpu_check, self.spec.estimator))
        f.addRow("Deposit", self.deposit_combo)
        gr = QHBoxLayout()
        self.grid_combo = _grid_combo(self)
        self.grid_combo.currentTextChanged.connect(self._changed)
        gr.addWidget(self.grid_combo)
        gr.addWidget(QLabel("³ cells    sub-samples per axis"))
        self.nsub_spin = QSpinBox(minimum=1, maximum=6)
        self.nsub_spin.valueChanged.connect(self._changed)
        gr.addWidget(self.nsub_spin)
        gr.addStretch(1)
        f.addRow("Grid", gr)
        self.gpu_check = QCheckBox(P.label("gpu"))
        self.gpu_check.setToolTip(P.tooltip("gpu"))
        self.gpu_check.toggled.connect(self._changed)
        f.addRow("", self.gpu_check)
        col.addWidget(g)

        g, self.field_checks = self._fields_group()
        self.fields_note = _note()
        g.layout().addWidget(self.fields_note, 3, 0, 1, 3)
        col.addWidget(g)

        self.ps_group = QGroupBox("Phase-space options")
        v = QVBoxLayout(self.ps_group)
        self.vm_check = QCheckBox(P.label("vertex_mass"))
        self.vw_check = QCheckBox(P.label("volume_weighted"))
        self.ca_check = QCheckBox(P.label("caustics"))
        self.cu_check = QCheckBox(P.label("caustic_cusps"))
        for cb, key in ((self.vm_check, "vertex_mass"), (self.vw_check, "volume_weighted"),
                        (self.ca_check, "caustics"), (self.cu_check, "caustic_cusps")):
            cb.setToolTip(P.tooltip(key))
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        col.addWidget(self.ps_group)

        self.slice_group = QGroupBox("Hi-res slice (point evaluation)")
        f = QFormLayout(self.slice_group)
        self.plane_combo = QComboBox()
        self.plane_combo.currentIndexChanged.connect(self._changed)
        f.addRow("Sample points", self.plane_combo)
        self.vg_check = QCheckBox(P.label("slice_vel_grad"))
        self.vg_check.setToolTip(P.tooltip("slice_vel_grad"))
        self.vg_check.toggled.connect(self._changed)
        f.addRow("", self.vg_check)
        col.addWidget(self.slice_group)

        g, (self.part_spin, self.conc_spin, self.scratch_edit, self.prefix_edit,
            self.pt_check, self.prec_combo) = self._resources_group()
        col.addWidget(g)
        col.addStretch(1)
        w = QWidget()
        w.setLayout(col)
        return w

    def _deposit_changed(self, combo: QComboBox, gpu: QCheckBox, estimator: str = "ps", dim: int = 3):
        """Picking the exact deposit ticks the GPU (PS-DTFE: 2x faster than the threaded CPU deposit at
        0.26M particles, 29 s against 57 s on a 256³ grid, and more at larger sizes; standard DTFE's exact
        cell average: 6.6x, 4.1 s against 26.9 s); the box stays editable. A 2D run has no GPU."""
        if not self._loading and combo.currentData() == "exact" and rs.gpu_built(estimator, dim=dim) and not gpu.isChecked():   # (either pair has the GPU when the single one has)
            gpu.setChecked(True)            # (its toggled signal runs _changed)
            self.status_message("exact deposit: the GPU is ticked (2x faster than the CPU at 0.26M particles, "
                                "more at larger sizes)" if estimator == "ps" else
                                "exact cell averages: the GPU is ticked (6.6x faster than the CPU at 0.26M particles)")
        self._changed()

    @staticmethod
    def _deposit_texts(combo: QComboBox, ps: bool):
        """the deposit choice in the words of the estimator it applies to (the same two settings)"""
        for i, key in enumerate(("sampled", "exact")):
            k = key if ps else key + "_dtfe"
            if combo.itemText(i) != P.label(k):
                combo.setItemText(i, P.label(k))
                combo.setItemData(i, P.tooltip(k) if ps else P.tooltip(k, "DTFE"), Qt.ToolTipRole)

    def _fields_group(self, title: str = "Fields") -> tuple[QGroupBox, dict[str, QCheckBox]]:
        g = QGroupBox(title)
        grid = QGridLayout(g)
        checks: dict[str, QCheckBox] = {}
        for i, (name, label) in enumerate(rs.PS_FIELDS):
            cb = QCheckBox(label)
            cb.toggled.connect(self._changed)
            checks[name] = cb
            grid.addWidget(cb, i // 3, i % 3)
        for c in range(3):                  # equal columns: the last one must not be starved of width
            grid.setColumnStretch(c, 1)
        return g, checks

    def _resources_group(self):
        g = QGroupBox("Resources and output")
        f = QFormLayout(g)
        part = QSpinBox(minimum=0, maximum=8)
        part.setSpecialValueText("auto")
        part.setSuffix("³ partitions")
        part.valueChanged.connect(self._changed)
        f.addRow("Partition", part)
        conc = QSpinBox(minimum=0, maximum=16)
        conc.setSpecialValueText("auto")
        conc.valueChanged.connect(self._changed)
        f.addRow("Concurrent partitions", conc)
        scratch = QLineEdit()
        scratch.setPlaceholderText("none: grids in RAM")
        scratch.editingFinished.connect(self._changed)
        f.addRow("Scratch directory", _dir_row(scratch, lambda: self._browse_scratch(scratch)))
        ptri = QCheckBox(P.label("parallel_triangulation"))
        ptri.setToolTip(P.tooltip("parallel_triangulation"))
        ptri.toggled.connect(self._changed)
        f.addRow("", ptri)
        prec = QComboBox()
        for key in rs.PRECISIONS:
            prec.addItem(rs.PRECISION_TEXT[key], key)
        prec.setToolTip("Which pair of programs runs: the single-precision one (the default), or the "
                        "double-precision pair, which keeps every position, density and field in float64 from "
                        "the input read onward (TNG stores its coordinates in double) and writes float64 grids, "
                        "at about twice the memory. Setup builds the pair.")
        prec.currentIndexChanged.connect(self._changed)
        f.addRow("Precision", prec)
        prefix = QLineEdit()
        prefix.editingFinished.connect(self._changed)
        f.addRow("Output prefix", prefix)
        return g, (part, conc, scratch, prefix, ptri, prec)

    # ================================================================ Custom snapshot tab
    def _build_own_tab(self) -> QWidget:
        col = QVBoxLayout()
        g = QGroupBox("Snapshot")
        f = QFormLayout(g)
        self.o_file = QLineEdit(placeholderText="choose a snapshot file")
        self.o_file.editingFinished.connect(self._own_file_changed)
        f.addRow("File", _dir_row(self.o_file, self._browse_own_file))
        self.o_type = QComboBox()
        for t, text in rs.INPUT_TYPES.items():
            self.o_type.addItem(text, t)
        self.o_type.currentIndexChanged.connect(self._changed)
        f.addRow("Format", self.o_type)
        self.o_dim = QComboBox()
        for d in rs.DIMS:
            self.o_dim.addItem(rs.DIM_TEXT[d], d)
        self.o_dim.setToolTip("3D, or 2D for a snapshot in a plane (its coordinates have two columns): the 2D "
                              "programs DTFE-2d and PS-DTFE-2d run it, on the CPU, with every field (the 2D "
                              "T-web classes: 0 void, 1 filament, 2 node). Set from the file when you choose one "
                              "(HDF5 and text files say which).")
        self.o_dim.currentIndexChanged.connect(self._changed)
        f.addRow("Dimensions", self.o_dim)
        un = QHBoxLayout()
        self.o_unit = QDoubleSpinBox(minimum=1e-6, maximum=1e9, decimals=4, value=1.0)
        self.o_unit.setToolTip("How many of the file's length units make one Mpc: 1 when the coordinates "
                               "are in Mpc, 1000 in kpc (e.g. Gadget's usual kpc/h: 1000, and h then stays "
                               "in the lengths)")
        self.o_unit.valueChanged.connect(self._changed)
        un.addWidget(QLabel("1 Mpc ="))
        un.addWidget(self.o_unit)
        un.addWidget(QLabel("file units"))
        for text, value in (("Mpc", 1.0), ("kpc", 1000.0)):
            b = QToolButton(text=text)
            b.clicked.connect(lambda _=False, v=value: self.o_unit.setValue(v))
            un.addWidget(b)
        un.addStretch(1)
        f.addRow("Length unit", un)
        self.o_periodic = QCheckBox(P.label("periodic"))
        self.o_periodic.setToolTip(P.tooltip("periodic", "DTFE"))
        self.o_periodic.toggled.connect(self._changed)
        f.addRow("", self.o_periodic)
        self.o_alpha = QDoubleSpinBox(minimum=0.0, maximum=50.0, decimals=1, singleStep=0.5, value=3.0)
        self.o_alpha.setSuffix(" mean particle spacings")
        self.o_alpha.setSpecialValueText("whole convex hull")
        self.o_alpha.setToolTip(P.tooltip("alpha_shape"))
        self.o_alpha.valueChanged.connect(self._changed)
        f.addRow("Alpha shape", self.o_alpha)
        self.o_snap_form = f
        self.o_box = QLineEdit(placeholderText="from the file header (Gadget BoxSize)")
        self.o_box.setToolTip("Optional: xlo xhi ylo yhi zlo zhi in the file's units (all >= 0)")
        self.o_box.editingFinished.connect(self._changed)
        f.addRow("Box", self.o_box)
        col.addWidget(g)

        g = QGroupBox("Estimator")
        f = QFormLayout(g)
        est = QHBoxLayout()
        self.o_ps = QRadioButton("PS-DTFE (phase space: streams, caustics)")
        self.o_dtfe = QRadioButton("DTFE (standard)")
        grp = QButtonGroup(self)
        grp.addButton(self.o_ps)
        grp.addButton(self.o_dtfe)
        self.o_ps.toggled.connect(self._changed)
        est.addWidget(self.o_ps)
        est.addWidget(self.o_dtfe)
        est.addStretch(1)
        f.addRow("", est)
        lg = QVBoxLayout()
        self.o_lag_file = QRadioButton("Initial positions stored in the snapshot ('InitialCoordinates')")
        self.o_lag_sep = QRadioButton("Initial positions in a separate snapshot (matched by particle ID):")
        grp2 = QButtonGroup(self)
        grp2.addButton(self.o_lag_file)
        grp2.addButton(self.o_lag_sep)
        self.o_lag_file.toggled.connect(self._changed)
        lg.addWidget(self.o_lag_file)
        lg.addWidget(self.o_lag_sep)
        self.o_lag_path = QLineEdit(placeholderText="the initial-conditions snapshot (Gadget HDF5)")
        self.o_lag_path.editingFinished.connect(self._changed)
        lg.addLayout(_dir_row(self.o_lag_path, lambda: self._browse_file(self.o_lag_path, "Initial conditions")))
        self.o_lag_box = QWidget()
        self.o_lag_box.setLayout(lg)
        f.addRow("Lagrangian", self.o_lag_box)
        self.o_scalar = QLineEdit(placeholderText="none (e.g. Potential)")
        self.o_scalar.setToolTip(P.tooltip("scalar"))
        self.o_scalar.editingFinished.connect(self._changed)
        f.addRow("Per-particle value", self.o_scalar)
        col.addWidget(g)

        col.addWidget(self._preset_group("custom"))
        g = QGroupBox("Grid and fields")
        f = QFormLayout(g)
        gr = QHBoxLayout()
        self.o_grid = _grid_combo(self)
        self.o_grid.currentTextChanged.connect(self._changed)
        gr.addWidget(self.o_grid)
        self.o_grid_cells = QLabel("³ cells    sub-samples per axis")
        gr.addWidget(self.o_grid_cells)
        self.o_nsub = QSpinBox(minimum=1, maximum=6)
        self.o_nsub.valueChanged.connect(self._changed)
        gr.addWidget(self.o_nsub)
        gr.addStretch(1)
        f.addRow("Grid", gr)
        self.o_deposit = QComboBox()
        self.o_deposit.addItem(P.label("sampled"), "sampled")
        self.o_deposit.addItem(P.label("exact"), "exact")
        self.o_deposit.setItemData(0, P.tooltip("sampled"), Qt.ToolTipRole)
        self.o_deposit.setItemData(1, P.tooltip("exact"), Qt.ToolTipRole)
        self.o_deposit.currentIndexChanged.connect(lambda _i: self._deposit_changed(self.o_deposit, self.o_gpu,
                                                                                    self.custom.estimator,
                                                                                    self.custom.dim))
        f.addRow("Deposit", self.o_deposit)
        self.o_gpu = QCheckBox(P.label("gpu"))
        self.o_gpu.setToolTip(P.tooltip("gpu"))
        self.o_vm = QCheckBox(P.label("vertex_mass"))
        self.o_vm.setToolTip(P.tooltip("vertex_mass"))
        self.o_vw = QCheckBox(P.label("volume_weighted"))
        self.o_vw.setToolTip(P.tooltip("volume_weighted"))
        self.o_ca = QCheckBox(P.label("caustics"))
        self.o_ca.setToolTip(P.tooltip("caustics"))
        for cb in (self.o_gpu, self.o_vm, self.o_vw, self.o_ca):
            cb.toggled.connect(self._changed)
            f.addRow("", cb)
        col.addWidget(g)
        g, self.o_fields = self._fields_group()
        col.addWidget(g)

        g = QGroupBox("Output")
        f = QFormLayout(g)
        self.o_outdir = QLineEdit()
        self.o_outdir.editingFinished.connect(self._changed)
        f.addRow("Folder", _dir_row(self.o_outdir, lambda: self._browse_dir(self.o_outdir, "Output folder")))
        self.o_name = QLineEdit(placeholderText="the snapshot's file name")
        self.o_name.editingFinished.connect(self._changed)
        f.addRow("Name", self.o_name)
        col.addWidget(g)
        g, (self.o_part, self.o_conc, self.o_scratch, _prefix, self.o_pt, self.o_prec) = self._resources_group()
        _prefix.hide()
        g.layout().labelForField(_prefix).hide()
        col.addWidget(g)
        col.addStretch(1)
        w = QWidget()
        w.setLayout(col)
        return w

    def _browse_file(self, edit: QLineEdit, title: str):
        start = edit.text() or str(Path.home())
        f, _ = QFileDialog.getOpenFileName(self, title, start)
        if f:
            edit.setText(f)
            self._changed()

    def _browse_dir(self, edit: QLineEdit, title: str):
        d = QFileDialog.getExistingDirectory(self, title, edit.text() or str(Path.home()))
        if d:
            edit.setText(d)
            self._changed()

    def _browse_own_file(self):
        self._browse_file(self.o_file, "Your snapshot file")
        self._own_file_changed()

    def _own_file_changed(self):
        """A new file: guess its format from the name and tick the GPU when its particle count says
        the GPU deposit is the faster one (the user can still change both)."""
        f = self.o_file.text().strip()
        if not f or f == self._own_last_file:
            return
        self._own_last_file = f
        suffix = Path(f).suffix.lower()
        guess = 105 if suffix in (".hdf5", ".h5", ".hdf") else 111 if suffix in (".txt", ".dat", ".csv") else None
        if guess is not None:
            self.o_type.setCurrentIndex(max(self.o_type.findData(guess), 0))
        dim = rs.snapshot_dim(Path(f).expanduser(), guess) if guess is not None else None
        if dim is not None:                 # the file says: 2D or 3D
            self.o_dim.setCurrentIndex(max(self.o_dim.findData(dim), 0))
        if guess == 105 and Path(f).expanduser().is_file():
            self._read_widgets()
            c = self.custom
            n = rs.hdf5_particle_count(Path(f).expanduser())
            self.o_gpu.setChecked(rs.recommended_gpu(c.deposit, c.fields, n, c.estimator, c.precision, c.dim))
        self._changed()

    # ================================================================ Pipeline tab
    def _build_pipeline_tab(self) -> QWidget:
        col = QVBoxLayout()
        g = QGroupBox("Simulations")
        v = QVBoxLayout(g)
        self.p_sims = QListWidget()
        self.p_sims.setMaximumHeight(100)
        self.p_sims.itemChanged.connect(self._changed)
        v.addWidget(self.p_sims)
        v.addWidget(_note("Computes every snapshot whose slice is missing or out of date."))
        col.addWidget(g)

        g = QGroupBox("Image plane")
        g.setToolTip("A full-box slice, evaluated at every pixel (point evaluation)")
        f = QFormLayout(g)
        self.p_nu = QComboBox()
        self.p_nu.setEditable(True)
        self.p_nu.addItems(["2048", "4096", "8192", "16384"])
        self.p_nu.setValidator(QIntValidator(16, 65536, self))
        self.p_nu.currentTextChanged.connect(self._changed)
        f.addRow("Pixels across the box", self.p_nu)
        self.p_adaptive = QCheckBox("Adaptive: the largest size that fits, per snapshot")
        self.p_adaptive.setToolTip("Needs 'Check memory': it asks the binary what each snapshot needs at "
                                   "each size (halving down to 1024²) and plans the pipeline accordingly.")
        self.p_adaptive.toggled.connect(self._changed)
        f.addRow("", self.p_adaptive)
        self.p_axis = QComboBox()
        self.p_axis.addItems(["x", "y", "z"])
        self.p_axis.currentIndexChanged.connect(self._changed)
        f.addRow("Normal axis", self.p_axis)
        self.p_center = QLineEdit(placeholderText="box centre")
        self.p_center.editingFinished.connect(self._changed)
        f.addRow("Position (Mpc)", self.p_center)
        self.p_vg = QCheckBox("Velocity gradient at each point")
        self.p_vg.setToolTip("Also evaluate the velocity gradient at every sample point: the divergence, shear and "
                             "vorticity maps (PTS_VEL_GRAD)")
        self.p_vg.toggled.connect(self._changed)
        f.addRow("", self.p_vg)
        slab = QHBoxLayout()
        self.p_planes = QSpinBox(minimum=1, maximum=256)
        self.p_planes.setToolTip("Sampling planes spread across a slab, which the figures' 'slab' projection averages "
                                 "(the planes option of make_image_plane.py). 1 = one crisp zero-thickness cross-section. "
                                 "A handful of planes ghosts every inclined structure (one displaced copy per plane): "
                                 f"use 1, or at least {rs.PLANE_GHOSTING_MIN} for a smooth projection.")
        self.p_planes.valueChanged.connect(self._changed)
        self.p_thick = QDoubleSpinBox(minimum=0.01, maximum=1000.0, decimals=2, singleStep=0.5)
        self.p_thick.setSuffix(" Mpc")
        self.p_thick.setToolTip("The slab's depth along the normal axis, over which the planes are spread "
                                "(the thickness option of make_image_plane.py; one plane has no thickness)")
        self.p_thick.valueChanged.connect(self._changed)
        slab.addWidget(self.p_planes)
        slab.addWidget(QLabel("planes across"))
        slab.addWidget(self.p_thick)
        slab.addStretch(1)
        f.addRow("Slab", slab)
        sub = QHBoxLayout()
        self.p_super = QSpinBox(minimum=1, maximum=8)
        self.p_super.setToolTip("K: evaluate K×K sample points per pixel and average them (the supersample option of "
                                "make_image_plane.py). The map then estimates the pixel-AREA mean, what the deposit grid "
                                "reports, instead of a point sample: no aliasing where structure is finer than a pixel "
                                "(halo cores, thin caustics). Costs K² points.")
        self.p_super.valueChanged.connect(self._changed)
        self.p_super_note = QLabel()
        sub.addWidget(self.p_super)
        sub.addWidget(self.p_super_note)
        sub.addStretch(1)
        f.addRow("Sub-samples per pixel axis", sub)
        self.p_window = QLineEdit(placeholderText="whole box: u0 u1 v0 v1")
        self.p_window.setToolTip("Image only this part of the plane: horizontal u0..u1 and vertical v0..v1 in Mpc "
                                 "(the u0, u1, v0, v1 options of make_image_plane.py); blank = the whole box. Pixels "
                                 "stay square, so the vertical pixel count follows the window's aspect.")
        self.p_window.editingFinished.connect(self._changed)
        f.addRow("Window (Mpc)", self.p_window)
        col.addWidget(g)

        g, self.p_fields = self._fields_group("Grids")
        gr = QHBoxLayout()
        self.p_grid = _grid_combo(self)
        self.p_grid.currentTextChanged.connect(self._changed)
        gr.addWidget(QLabel("Grid"))
        gr.addWidget(self.p_grid)
        gr.addWidget(QLabel("³ cells    sub-samples per axis"))
        self.p_nsub = QSpinBox(minimum=1, maximum=6)
        self.p_nsub.setToolTip("nSub³ sample points per cell for the sampled deposit (AVG_SUBSAMPLES)")
        self.p_nsub.valueChanged.connect(self._changed)
        gr.addWidget(self.p_nsub)
        gr.addStretch(1)
        g.layout().addLayout(gr, 3, 0, 1, 3)
        dep = QHBoxLayout()
        self.p_deposit = QComboBox()
        self.p_deposit.addItem(P.label("sampled"), "sampled")
        self.p_deposit.addItem(P.label("exact"), "exact")
        self.p_deposit.setItemData(0, P.tooltip("sampled"), Qt.ToolTipRole)
        self.p_deposit.setItemData(1, P.tooltip("exact"), Qt.ToolTipRole)
        self.p_deposit.currentIndexChanged.connect(lambda _i: self._deposit_changed(self.p_deposit, self.p_gpu))
        dep.addWidget(QLabel("Deposit"))
        dep.addWidget(self.p_deposit, 1)
        g.layout().addLayout(dep, 4, 0, 1, 3)
        self.p_gpu = QCheckBox(P.label("gpu"))
        self.p_gpu.setToolTip(P.tooltip("gpu"))
        self.p_gpu.toggled.connect(self._changed)
        g.layout().addWidget(self.p_gpu, 5, 0, 1, 3)
        g.setToolTip("run_ps_pipeline.sh hands these to run_ps_dtfe.sh exactly as a Grids job does; left alone, "
                     "the scripts' own defaults are the production settings (GPU, vertex masses, volume-weighted "
                     "velocities)")
        col.addWidget(g)

        g = QGroupBox("Phase-space options")
        v = QVBoxLayout(g)
        self.p_vm = QCheckBox(P.label("vertex_mass"))
        self.p_vw = QCheckBox(P.label("volume_weighted"))
        self.p_ca = QCheckBox(P.label("caustics"))
        self.p_cu = QCheckBox(P.label("caustic_cusps"))
        for cb, key in ((self.p_vm, "vertex_mass"), (self.p_vw, "volume_weighted"),
                        (self.p_ca, "caustics"), (self.p_cu, "caustic_cusps")):
            cb.setToolTip(P.tooltip(key))
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        col.addWidget(g)

        g, (self.p_part, self.p_conc, self.p_scratch, self.p_prefix, self.p_pt, self.p_prec) = self._resources_group()
        col.addWidget(g)

        g = QGroupBox("Run")
        v = QVBoxLayout(g)
        self.p_force = QCheckBox("Recompute every snapshot, up to date or not")
        self.p_plan = QCheckBox("Plan only: show what would run")
        self.p_render = QCheckBox("Then render the point-evaluated figures")
        self.p_force.setToolTip("Ignore every freshness check (FORCE)")
        self.p_plan.setToolTip("Print the plan, compute nothing (DRY_RUN)")
        self.p_render.setToolTip("plot_pointeval.py, on this pipeline's own image plane, with the options below")
        for cb in (self.p_force, self.p_plan, self.p_render):
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        self.p_render_box = QWidget()
        rf = QFormLayout(self.p_render_box)
        rf.setContentsMargins(24, 0, 0, 0)
        self.p_rfields = QLineEdit(placeholderText="all the run wrote: " + ", ".join(rs.POINTEVAL_FIELDS))
        self.p_rfields.setToolTip("A comma-separated subset of the point-evaluated fields (the fields option of "
                                  "plot_pointeval.py); blank = every field the run wrote")
        self.p_rfields.editingFinished.connect(self._changed)
        rf.addRow("Fields", self.p_rfields)
        self.p_rproject = QComboBox()
        self.p_rproject.addItem("the central plane: a zero-thickness cross-section", "plane")
        self.p_rproject.addItem("slab: the mean over every plane of the slab", "slab")
        self.p_rproject.setToolTip("The projection option of plot_pointeval.py; 'slab' needs several planes across the "
                                   "slab (above)")
        self.p_rproject.currentIndexChanged.connect(self._changed)
        rf.addRow("Projection", self.p_rproject)
        self.p_rsmooth = QDoubleSpinBox(minimum=0.0, maximum=1000.0, decimals=1, singleStep=0.5)
        self.p_rsmooth.setSuffix(" px")
        self.p_rsmooth.setToolTip("Gaussian smoothing of every map, in output pixels (the smooth option of "
                                  "plot_pointeval.py); 0 = none. Also suppresses the smoothed gradient companions below.")
        self.p_rsmooth.valueChanged.connect(self._changed)
        rf.addRow("Smoothing", self.p_rsmooth)
        self.p_rsmoothd = QDoubleSpinBox(minimum=0.0, maximum=1000.0, decimals=1, singleStep=1.0)
        self.p_rsmoothd.setSuffix(" px")
        self.p_rsmoothd.setToolTip("The gradient maps (divergence, shear, vorticity, density gradient) are piecewise "
                                   "constant per tetrahedron and look faceted; each also gets a Gaussian-smoothed "
                                   "companion figure at this width (the smooth-derivatives option of plot_pointeval.py); "
                                   "0 = none")
        self.p_rsmoothd.valueChanged.connect(self._changed)
        rf.addRow("Gradient maps also smoothed at", self.p_rsmoothd)
        self.p_rfixed = QCheckBox("Fixed density range of the grid maps (0.1 .. 1e4) instead of a percentile stretch")
        self.p_rfixed.setToolTip("The fixed-range option of plot_pointeval.py: side-by-side comparison with the grid "
                                 "slice maps; clips ~26% of a z=0 image to the darkest colour")
        self.p_rforce = QCheckBox("Re-render figures newer than their data")
        self.p_rforce.setToolTip("The force option of plot_pointeval.py")
        for cb in (self.p_rfixed, self.p_rforce):
            cb.toggled.connect(self._changed)
            rf.addRow("", cb)
        v.addWidget(self.p_render_box)
        col.addWidget(g)
        col.addStretch(1)
        w = QWidget()
        w.setLayout(col)
        return w

    # ================================================================ Plots tab
    def _build_plots_tab(self) -> QWidget:
        col = QVBoxLayout()
        g = QGroupBox("Figure sets")
        v = QVBoxLayout(g)
        self.fig_list = QListWidget()
        self.fig_list.setMaximumHeight(200)
        for fs in rs.FIGURE_SETS:
            it = QListWidgetItem(fs.title)
            it.setData(Qt.UserRole, fs.key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.fig_list.addItem(it)
        self.fig_list.itemChanged.connect(self._changed)
        v.addWidget(self.fig_list)
        self.fig_stack = QStackedWidget()
        for fs in rs.FIGURE_SETS:
            self.fig_stack.addWidget(self._figure_page(fs))
        self.fig_list.currentRowChanged.connect(self.fig_stack.setCurrentIndex)
        v.addWidget(self.fig_stack)
        col.addWidget(g)

        g = QGroupBox("Grids to plot")
        f = QFormLayout(g)
        self.pl_method = QComboBox()
        for key, label in (("auto", "Auto: PS-DTFE where present, else DTFE"), ("ps", "PS-DTFE"),
                           ("dtfe", "Standard DTFE")):
            self.pl_method.addItem(label, key)
        self.pl_method.currentIndexChanged.connect(self._changed)
        f.addRow("Estimator", self.pl_method)
        self.pl_prefix = QLineEdit(placeholderText="default (ps_output)")
        self.pl_prefix.editingFinished.connect(self._changed)
        f.addRow("Grid prefix", self.pl_prefix)
        self.pl_smooth = QDoubleSpinBox(minimum=0.0, maximum=50.0, decimals=1, singleStep=0.5)
        self.pl_smooth.setSuffix(" cells")
        self.pl_smooth.valueChanged.connect(self._changed)
        f.addRow("Smoothing", self.pl_smooth)
        col.addWidget(g)
        col.addStretch(1)
        w = QWidget()
        w.setLayout(col)
        return w

    def _figure_page(self, fs: rs.FigureSet) -> QWidget:
        page = QWidget()
        f = QFormLayout(page)
        f.setContentsMargins(4, 4, 4, 4)
        about = QLabel(fs.about)
        about.setWordWrap(True)
        about.setToolTip(f"python/{fs.script}, {PER_TEXT[fs.per]}  →  python/figures/{fs.output}")
        f.addRow(about)
        for o in fs.opts:
            if o.kind == "bool":
                w = QCheckBox(o.label)
                w.toggled.connect(self._changed)
            elif o.kind == "int":
                w = QSpinBox(minimum=0, maximum=1_000_000)
                w.valueChanged.connect(self._changed)
            elif o.kind == "float":
                w = QDoubleSpinBox(minimum=0.0, maximum=1000.0, decimals=2, singleStep=0.5)
                w.valueChanged.connect(self._changed)
            elif o.kind == "choice":
                w = QComboBox()
                w.addItems(list(o.choices))
                w.currentIndexChanged.connect(self._changed)
            elif o.kind == "plane":
                w = QComboBox()
                w.currentIndexChanged.connect(self._changed)
            else:
                w = QLineEdit(placeholderText=o.help)
                w.editingFinished.connect(self._changed)
            if o.help:
                w.setToolTip(o.help)
            self.opt_widgets[(fs.key, o.key)] = (o, w)
            f.addRow("" if o.kind == "bool" else o.label, w)
        return page

    @staticmethod
    def _set_opt(o: rs.Opt, w: QWidget, value):
        if o.kind == "bool":
            w.setChecked(bool(value))
        elif o.kind == "int":
            w.setValue(int(value))
        elif o.kind == "float":
            w.setValue(float(value))
        elif o.kind == "choice":
            w.setCurrentText(str(value))
        elif o.kind == "plane":
            w.setCurrentIndex(max(w.findData(str(value)), 0))
        else:
            w.setText(str(value))

    @staticmethod
    def _get_opt(o: rs.Opt, w: QWidget):
        if o.kind == "bool":
            return w.isChecked()
        if o.kind in ("int", "float"):
            return w.value()
        if o.kind == "choice":
            return w.currentText()
        if o.kind == "plane":
            return w.currentData() or ""
        return w.text().strip()

    # ================================================================ run panel
    def _build_run_panel(self) -> QWidget:
        v = QVBoxLayout()
        # the exact commands a run executes: hidden unless asked for (View > Show Commands)
        self.cmd_box = QWidget()
        cb = QVBoxLayout(self.cmd_box)
        cb.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        self.cmd_title = QLabel("")
        self.cmd_title.setStyleSheet("font-weight: bold")
        head.addWidget(self.cmd_title, 1)
        copy = QPushButton("Copy")
        copy.setToolTip("Copy the commands, to run them in a terminal")
        copy.clicked.connect(self.copy_commands)
        head.addWidget(copy)
        self.export_btn = QPushButton("Export…")
        self.export_btn.setToolTip("Save the commands as a bash script, or as a SLURM job for a cluster")
        self.export_btn.clicked.connect(self.export_job)
        head.addWidget(self.export_btn)
        self.cmd_hide = QPushButton("Hide")
        self.cmd_hide.clicked.connect(lambda: self._toggle_commands(False))
        head.addWidget(self.cmd_hide)
        cb.addLayout(head)
        self.cmd_view = QPlainTextEdit(readOnly=True)
        self.cmd_view.setFont(_mono())
        self.cmd_view.setMaximumHeight(130)
        cb.addWidget(self.cmd_view)
        self.cmd_box.setVisible(self.show_commands)
        v.addWidget(self.cmd_box)

        v.addWidget(QLabel("<b>Checks</b>"))
        self.checks = QListWidget()
        self.checks.setMaximumHeight(110)
        self.checks.setWordWrap(True)
        v.addWidget(self.checks)
        self.estimate = _note()
        v.addWidget(self.estimate)

        run = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self.start_run)
        self.queue_btn = QPushButton("Add to queue")
        self.queue_btn.clicked.connect(self.enqueue)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_run)
        self.status = QLabel("Idle")
        run.addWidget(self.run_btn)
        run.addWidget(self.queue_btn)
        self.mem_btn = QPushButton("Check memory")
        self.mem_btn.setToolTip("Ask the binary's auto-tuner what this run needs per snapshot (and how it "
                                "would split the work), against this machine's memory budget: it reads each "
                                "snapshot and computes nothing")
        self.mem_btn.clicked.connect(self.check_memory)
        run.addWidget(self.mem_btn)
        run.addWidget(self.stop_btn)
        run.addWidget(self.status, 1)
        v.addLayout(run)
        prog = QHBoxLayout()
        self.snap_label = QLabel("")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.bar.setFormat("")
        self.eta = QLabel("")
        prog.addWidget(self.snap_label)
        prog.addWidget(self.bar, 1)
        prog.addWidget(self.eta)
        v.addLayout(prog)

        # after a failed job: what probably went wrong, and a button that fixes it where one can
        self.banner = QFrame()
        self.banner.setFrameShape(QFrame.StyledPanel)
        self.banner.setStyleSheet("QFrame { background: rgba(198, 40, 40, 0.08); }")
        bl = QVBoxLayout(self.banner)
        self.banner_text = QLabel("")
        self.banner_text.setWordWrap(True)
        self.banner_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bl.addWidget(self.banner_text)
        self.banner_buttons = QHBoxLayout()
        bl.addLayout(self.banner_buttons)
        self.banner.hide()
        v.addWidget(self.banner)

        self.results = QTabWidget()
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setFont(_mono())
        self.log.setMaximumBlockCount(20000)
        self.results.addTab(self.log, "Log")
        self.queue_tab = self._build_queue_tab()
        self.results.addTab(self.queue_tab, "Queue")
        self.figures = FigureBrowser()
        self.figures.changed.connect(lambda: self._sync_figures_actions())
        self.figures.mode.currentIndexChanged.connect(lambda _i: self._sync_figures_actions())
        self.results.addTab(self.figures, "Figures")
        self.results.addTab(self.runs, "Runs")
        self.results.setTabToolTip(self.results.indexOf(self.runs),
                                   "Every run on disk: settings, time, memory, size, and outputs made before a fix")
        self.results.currentChanged.connect(self._results_tab_changed)
        v.addWidget(self.results, 1)
        w = QWidget()
        w.setLayout(v)
        return w

    def _build_queue_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.q_list = QListWidget()
        self.q_list.currentRowChanged.connect(self._show_queued)
        v.addWidget(self.q_list, 2)
        btns = QHBoxLayout()
        self.q_start = QPushButton("Start queue")
        self.q_start.clicked.connect(self.start_queue)
        self.q_up, self.q_down = QPushButton("Up"), QPushButton("Down")
        self.q_up.clicked.connect(lambda: self._move_queued(-1))
        self.q_down.clicked.connect(lambda: self._move_queued(+1))
        self.q_remove, self.q_clear = QPushButton("Remove"), QPushButton("Clear finished")
        self.q_remove.clicked.connect(self._remove_queued)
        self.q_clear.clicked.connect(self._clear_finished)
        for b in (self.q_start, self.q_up, self.q_down, self.q_remove, self.q_clear):
            btns.addWidget(b)
        btns.addStretch(1)
        v.addLayout(btns)
        opts = self._restore_json("queue_options", {})
        self.q_stop_on_fail = QCheckBox("Stop the queue when a job fails (later jobs usually need its output)")
        self.q_stop_on_fail.setChecked(opts.get("stop_on_fail", True))
        self.q_wait = QCheckBox("Before a grids or pipeline job, wait while another DTFE / PS-DTFE run is going")
        self.q_wait.setChecked(opts.get("wait", True))
        self.q_notify = QCheckBox("Notify me when a job or the queue finishes (a desktop notification)")
        self.q_notify.setChecked(opts.get("notify", True))
        for cb in (self.q_stop_on_fail, self.q_wait, self.q_notify):
            cb.toggled.connect(self._save_queue)
            v.addWidget(cb)
        for cb, action in ((self.q_stop_on_fail, self.q_stop_on_fail_action), (self.q_wait, self.q_wait_action),
                           (self.q_notify, self.q_notify_action)):      # the Queue menu's copies
            action.setChecked(cb.isChecked())
            action.toggled.connect(cb.setChecked)
            cb.toggled.connect(action.setChecked)
        self.q_detail = QPlainTextEdit(readOnly=True)
        self.q_detail.setFont(_mono())
        v.addWidget(self.q_detail, 1)
        return w

    # ================================================================ spec <-> widgets
    def _load_into_widgets(self):
        self._loading = True
        s, p, pl = self.spec, self.pipe, self.plots
        self.root_edit.setText(s.data_root)
        d = self.data
        d.sim = d.sim or s.sim
        for cb, on in ((self.d_dl_snap, d.download_snapshots), (self.d_dl_gc, d.download_groupcats),
                       (self.d_dl_tree, d.download_trees), (self.d_dl_ics, d.download_ics),
                       (self.d_mg_snap, d.merge_snapshots), (self.d_mg_gc, d.merge_groupcats),
                       (self.d_mg_tree, d.merge_trees), (self.d_mg_ics, d.convert_ics),
                       (self.d_delete, d.delete_chunks)):
            cb.setChecked(on)
        self._fill_sims()
        # Grids
        (self.ps_radio if s.estimator == "ps" else self.dtfe_radio).setChecked(True)
        self.deposit_combo.setCurrentIndex(max(self.deposit_combo.findData(s.deposit), 0))
        self.grid_combo.setCurrentText(str(s.grid))
        self.nsub_spin.setValue(s.nsub)
        self.gpu_check.setChecked(s.gpu)
        for name, cb in self.field_checks.items():
            cb.setChecked(name in s.fields)
        self.vm_check.setChecked(s.vertex_mass)
        self.vw_check.setChecked(s.volume_weighted)
        self.ca_check.setChecked(s.caustics)
        self.cu_check.setChecked(s.caustic_cusps)
        self.vg_check.setChecked(s.slice_vel_grad)
        self.part_spin.setValue(s.partition)
        self.conc_spin.setValue(s.max_concurrent)
        self.scratch_edit.setText(s.scratch_dir)
        self.prefix_edit.setText(s.output_prefix)
        self.pt_check.setChecked(s.parallel_triangulation)
        self.prec_combo.setCurrentIndex(max(self.prec_combo.findData(s.precision), 0))
        # Pipeline
        self.p_nu.setCurrentText(str(p.nu))
        self.p_adaptive.setChecked(p.adaptive)
        self.p_axis.setCurrentText(p.axis)
        self.p_center.setText(p.center)
        self.p_vg.setChecked(p.vel_grad)
        self.p_planes.setValue(p.planes)
        self.p_thick.setValue(p.thickness)
        self.p_super.setValue(p.supersample)
        self.p_window.setText(p.window)
        self.p_grid.setCurrentText(str(p.grid))
        for name, cb in self.p_fields.items():
            cb.setChecked(name in p.fields)
        self.p_deposit.setCurrentIndex(max(self.p_deposit.findData(p.deposit), 0))
        self.p_nsub.setValue(p.nsub)
        self.p_gpu.setChecked(p.gpu)
        self.p_vm.setChecked(p.vertex_mass)
        self.p_vw.setChecked(p.volume_weighted)
        self.p_ca.setChecked(p.caustics)
        self.p_cu.setChecked(p.caustic_cusps)
        self.p_pt.setChecked(p.parallel_triangulation)
        self.p_part.setValue(p.partition)
        self.p_conc.setValue(p.max_concurrent)
        self.p_scratch.setText(p.scratch_dir)
        self.p_prefix.setText(p.output_prefix)
        self.p_prec.setCurrentIndex(max(self.p_prec.findData(p.precision), 0))
        self.p_force.setChecked(p.force)
        self.p_plan.setChecked(p.plan_only)
        self.p_render.setChecked(p.render)
        self.p_rfields.setText(p.render_fields)
        self.p_rproject.setCurrentIndex(max(self.p_rproject.findData(p.render_project), 0))
        self.p_rsmooth.setValue(p.render_smooth)
        self.p_rsmoothd.setValue(p.render_smooth_derivatives)
        self.p_rfixed.setChecked(p.render_fixed_range)
        self.p_rforce.setChecked(p.render_force)
        # Custom snapshot
        c = self.custom
        self.o_file.setText(c.input_file)
        self.o_type.setCurrentIndex(max(self.o_type.findData(c.input_type), 0))
        self.o_dim.setCurrentIndex(max(self.o_dim.findData(c.dim), 0))
        self.o_unit.setValue(c.mpc_unit)
        self.o_periodic.setChecked(c.periodic)
        self.o_alpha.setValue(c.alpha_shape)
        self.o_box.setText(" ".join(f"{v:g}" for v in c.box))
        (self.o_ps if c.estimator == "ps" else self.o_dtfe).setChecked(True)
        (self.o_lag_sep if c.lagrangian == "separate" else self.o_lag_file).setChecked(True)
        self.o_lag_path.setText(c.lagrangian_file)
        self.o_scalar.setText(c.scalar_dataset)
        self.o_grid.setCurrentText(str(c.grid))
        self.o_nsub.setValue(c.nsub)
        self.o_deposit.setCurrentIndex(max(self.o_deposit.findData(c.deposit), 0))
        self.o_gpu.setChecked(c.gpu)
        self.o_vm.setChecked(c.vertex_mass)
        self.o_vw.setChecked(c.volume_weighted)
        self.o_ca.setChecked(c.caustics)
        for name, cb in self.o_fields.items():
            cb.setChecked(name in c.fields)
        self.o_outdir.setText(c.output_dir)
        self.o_name.setText(c.output_name)
        self.o_part.setValue(c.partition)
        self.o_conc.setValue(c.max_concurrent)
        self.o_scratch.setText(c.scratch_dir)
        self.o_pt.setChecked(c.parallel_triangulation)
        self.o_prec.setCurrentIndex(max(self.o_prec.findData(c.precision), 0))
        self._own_last_file = c.input_file
        # Plots
        for i in range(self.fig_list.count()):
            it = self.fig_list.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in pl.sets else Qt.Unchecked)
        self.fig_list.setCurrentRow(0)
        self.pl_method.setCurrentIndex(max(self.pl_method.findData(pl.method), 0))
        self.pl_prefix.setText(pl.prefix)
        self.pl_smooth.setValue(pl.smooth)
        for (set_key, key), (o, w) in self.opt_widgets.items():
            self._set_opt(o, w, pl.opt(set_key, key))
        self._loading = False

    def _read_widgets(self):
        root, sim = self.root_edit.text().strip(), self.sim_combo.currentText()
        snaps = _checked(self.snap_list)
        s, p, pl = self.spec, self.pipe, self.plots
        for spec in (s, pl):
            spec.data_root, spec.sim, spec.snapshots = root, sim, list(snaps)
        p.data_root = root
        # Data
        d = self.data
        d.data_root, d.sim = root, self.d_sim.currentText().strip()
        d.snapshots = [self.d_table.item(r, 0).data(Qt.UserRole) for r in range(self.d_table.rowCount())
                       if self.d_table.item(r, 0).checkState() == Qt.Checked]
        d.download_snapshots, d.download_groupcats = self.d_dl_snap.isChecked(), self.d_dl_gc.isChecked()
        d.download_trees, d.download_ics = self.d_dl_tree.isChecked(), self.d_dl_ics.isChecked()
        d.merge_snapshots, d.merge_groupcats = self.d_mg_snap.isChecked(), self.d_mg_gc.isChecked()
        d.merge_trees, d.convert_ics = self.d_mg_tree.isChecked(), self.d_mg_ics.isChecked()
        d.delete_chunks = self.d_delete.isChecked()
        # Grids
        s.estimator = "ps" if self.ps_radio.isChecked() else "dtfe"
        s.deposit = self.deposit_combo.currentData() or "sampled"
        s.grid = _int(self.grid_combo.currentText())
        s.nsub = self.nsub_spin.value()
        s.gpu = self.gpu_check.isChecked()
        s.fields = [n for n, _ in rs.PS_FIELDS if self.field_checks[n].isChecked()]
        s.vertex_mass = self.vm_check.isChecked()
        s.volume_weighted = self.vw_check.isChecked()
        s.caustics = self.ca_check.isChecked()
        s.caustic_cusps = self.cu_check.isChecked() and self.ca_check.isChecked()   # a greyed box cannot be unticked
        s.slice_plane = self.plane_combo.currentData() or ""
        s.slice_vel_grad = self.vg_check.isChecked()
        s.partition = self.part_spin.value()
        s.max_concurrent = self.conc_spin.value()
        s.scratch_dir = self.scratch_edit.text().strip()
        s.output_prefix = self.prefix_edit.text().strip()
        s.parallel_triangulation = self.pt_check.isChecked()
        s.precision = self.prec_combo.currentData() or "single"
        # Pipeline
        p.sims = _checked(self.p_sims)
        p.nu = _int(self.p_nu.currentText())
        p.adaptive = self.p_adaptive.isChecked()
        p.axis = self.p_axis.currentText()
        p.center = self.p_center.text().strip()
        p.vel_grad = self.p_vg.isChecked()
        p.planes = self.p_planes.value()
        p.thickness = self.p_thick.value()
        p.supersample = self.p_super.value()
        p.window = self.p_window.text().strip()
        p.grid = _int(self.p_grid.currentText())
        p.fields = [n for n, _ in rs.PS_FIELDS if self.p_fields[n].isChecked()]
        p.deposit = self.p_deposit.currentData() or "sampled"
        p.nsub = self.p_nsub.value()
        p.gpu = self.p_gpu.isChecked()
        p.vertex_mass = self.p_vm.isChecked()
        p.volume_weighted = self.p_vw.isChecked()
        p.caustics = self.p_ca.isChecked()
        p.caustic_cusps = self.p_cu.isChecked() and self.p_ca.isChecked()
        p.parallel_triangulation = self.p_pt.isChecked()
        p.partition = self.p_part.value()
        p.max_concurrent = self.p_conc.value()
        p.scratch_dir = self.p_scratch.text().strip()
        p.output_prefix = self.p_prefix.text().strip()
        p.precision = self.p_prec.currentData() or "single"
        p.force = self.p_force.isChecked()
        p.plan_only = self.p_plan.isChecked()
        p.render = self.p_render.isChecked()
        p.render_fields = self.p_rfields.text().strip()
        p.render_project = self.p_rproject.currentData() or "plane"
        p.render_smooth = self.p_rsmooth.value()
        p.render_smooth_derivatives = self.p_rsmoothd.value()
        p.render_fixed_range = self.p_rfixed.isChecked()
        p.render_force = self.p_rforce.isChecked()
        # Custom snapshot
        c = self.custom
        c.input_file = self.o_file.text().strip()
        c.input_type = int(self.o_type.currentData() or 105)
        c.dim = int(self.o_dim.currentData() or 3)
        c.mpc_unit = self.o_unit.value()
        c.periodic = self.o_periodic.isChecked()
        c.alpha_shape = self.o_alpha.value()
        try:
            c.box = [float(t) for t in self.o_box.text().replace(",", " ").split()]
        except ValueError:
            c.box = [float("nan")]          # reported by problems() as a malformed box
        c.estimator = "ps" if self.o_ps.isChecked() else "dtfe"
        c.lagrangian = "separate" if self.o_lag_sep.isChecked() else "file"
        c.lagrangian_file = self.o_lag_path.text().strip()
        c.scalar_dataset = self.o_scalar.text().strip()
        c.grid = _int(self.o_grid.currentText())
        c.nsub = self.o_nsub.value()
        c.deposit = self.o_deposit.currentData() or "sampled"
        c.gpu = self.o_gpu.isChecked()
        c.vertex_mass = self.o_vm.isChecked()
        c.volume_weighted = self.o_vw.isChecked()
        c.caustics = self.o_ca.isChecked()
        c.fields = [n for n, _ in rs.PS_FIELDS if self.o_fields[n].isChecked()]
        c.output_dir = self.o_outdir.text().strip()
        c.output_name = self.o_name.text().strip()
        c.partition = self.o_part.value()
        c.max_concurrent = self.o_conc.value()
        c.scratch_dir = self.o_scratch.text().strip()
        c.parallel_triangulation = self.o_pt.isChecked()
        c.precision = self.o_prec.currentData() or "single"
        # Plots
        pl.sets = _checked(self.fig_list)
        pl.method = self.pl_method.currentData() or "auto"
        pl.prefix = self.pl_prefix.text().strip()
        pl.smooth = self.pl_smooth.value()
        opts: dict[str, dict] = {}
        for (set_key, key), (o, w) in self.opt_widgets.items():
            opts.setdefault(set_key, {})[key] = self._get_opt(o, w)
        pl.options = opts

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
        for sim in sims:
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
        if self.explore.outputs:            # Explore lists this root's outputs: follow it (2026-10-05)
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
            self.explore.refresh_outputs()

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
        for w in (self.nsub_spin, self.ps_group, self.slice_group, self.prefix_edit):
            w.setEnabled(ps)
        self._deposit_texts(self.deposit_combo, ps)       # both estimators: standard DTFE's exact cell average
        for cb in self.field_checks.values():
            cb.setEnabled(ps)
        self.fields_note.setText("" if ps else "Standard DTFE computes a fixed set of fields.")
        self.fields_note.setVisible(not ps)
        self.cu_check.setEnabled(ps and s.caustics)
        self.vg_check.setEnabled(ps and bool(s.slice_plane))
        self.nsub_spin.setEnabled(ps and s.deposit == "sampled")
        gpu_ok = rs.gpu_built(s.estimator, s.precision)
        self.gpu_check.setText(P.label("gpu" if ps else "gpu_dtfe") + ("" if gpu_ok else " (this build has no GPU support)"))
        self.gpu_check.setToolTip(P.tooltip("gpu") if ps else P.tooltip("gpu_dtfe", "DTFE"))
        tbb_note = "" if rs.tbb_built(s.precision) else " (this build has no TBB: ignored)"
        self.pt_check.setText(P.label("parallel_triangulation") + tbb_note)
        self.pt_check.setEnabled(ps)
        for combo in (self.prec_combo, self.o_prec, self.p_prec):     # the double pair's state, in words
            i = combo.findData("double")
            built = rs.binary_built("ps", "double") and rs.binary_built("dtfe", "double")
            combo.setItemText(i, rs.PRECISION_TEXT["double"] + ("" if built else " (not built: Setup builds the pair)"))
        # ---- Custom snapshot tab
        c = self.custom
        cps = c.estimator == "ps"
        flat = c.dim == 2                   # the 2D programs: CPU only, sequential insertion
        for w in (self.o_lag_box, self.o_vm, self.o_vw, self.o_ca, self.o_pt):
            w.setEnabled(cps)
        self._deposit_texts(self.o_deposit, cps)
        for cb in self.o_fields.values():
            cb.setEnabled(cps)
        self.o_grid_cells.setText(("²" if flat else "³") + " cells    sub-samples per axis")
        self.o_part.setSuffix(("²" if flat else "³") + " partitions")
        self.o_box.setToolTip(f"Optional: xlo xhi ylo yhi{'' if flat else ' zlo zhi'} in the file's units (all >= 0)")
        self.o_lag_path.setEnabled(cps and c.lagrangian == "separate")
        self.o_nsub.setEnabled(cps and c.deposit == "sampled")
        self.o_scalar.setEnabled(c.input_type == 105)
        self.o_pt.setText(P.label("parallel_triangulation") + (" (3D only)" if flat else "" if rs.tbb_built(c.precision)
                                                               else " (this build has no TBB: ignored)"))
        self.o_pt.setEnabled(cps and not flat)
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
            steps = job.steps() if (tab == PIPELINE or job.sim) and Path(job.data_root).is_dir() else []
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
        checkable = ((tab == GRIDS and self.spec.estimator == "ps" and bool(self.spec.snapshots)
                      and bool(self.spec.sim) and Path(job.data_root).is_dir())
                     or (tab == PIPELINE and bool(self.pipe.sims) and Path(job.data_root).is_dir())
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
            if not d.sim or not Path(d.data_root).is_dir():
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
                                sliced=s.estimator == "ps" and bool(s.slice_plane))
            if t and n:
                est += (f"   ·   ≈ {R.format_duration(t[0])} per snapshot (median of {t[1]} earlier run"
                        f"{'s' if t[1] != 1 else ''}), ≈ {R.format_duration(t[0] * n)} in all")
            return (est + f"   ·   {n} snapshot{'s' if n != 1 else ''}"
                    + ("   ·   grids on scratch disk" if s.scratch_dir else ""))
        if tab == OWN:
            c = self.custom
            gb = rs.grids_gb(c.grid, c.fields if c.estimator == "ps" else rs.DTFE_FIELDS, c.precision)
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

    # ================================================================ running
    def start_run(self):
        self._read_widgets()
        job = self._job()
        if any(level == "error" for level, _ in job.problems()):
            return
        steps = job.steps()
        if not steps or self.proc is not None:
            return
        self._figs_since = time.time()
        self._begin(steps, self.tabs.currentIndex(), job, None)

    def _begin(self, steps: list[rs.Step], tab: int, spec, qjob: rs.QueuedJob | None):
        """Run one job's steps: from the Run button (qjob None) or as the queue's next job."""
        self._steps, self._step_i, self._results = steps, -1, []
        self._job_text = ""
        self._close_tee()                   # a handle left by a step that never finished must not take this job's output
        G.drop_cubes()                      # Explore's cubes are this process's memory: the auto-tuner budgets from what is free
        self.banner.hide()
        if isinstance(spec, rs.CustomSpec):
            self._register_custom_output(spec)
        self._job_tab, self._job_spec, self._current_job = tab, spec, qjob
        self._job_start = time.time()
        self._job_sim = getattr(spec, "sim", "")
        self._buffer, self._summary, self._stopping = "", "", False
        self._snaps_started = self._snaps_ok = 0
        self._dl_files = self._dl_bytes = self._problems = 0
        self._total_snaps = len(spec.snapshots) if tab == GRIDS else 0
        if qjob is None:
            self.log.clear()
        else:
            k = self.queue.index(qjob) + 1
            self.log.appendPlainText(("\n" if self.log.blockCount() > 1 else "")
                                     + f"{'=' * 72}\nQueue job {k}/{len(self.queue)}: {qjob.title}\n{'=' * 72}")
        self.results.setCurrentIndex(self.results.indexOf(self.log))
        self.snap_label.setText("")
        self.eta.setText("")
        self._next_step()

    def _next_step(self):
        self._step_i += 1
        if self._stopping or self._step_i >= len(self._steps):
            self._finish_job()
            return
        st, n = self._steps[self._step_i], len(self._steps)
        if st.after and any(lbl == st.after and code != 0 for lbl, code in self._results):
            # a merge after a failed download: nothing to merge safely (half a tree set would pass)
            self.log.appendPlainText(f"\n!! {st.label}: skipped, '{st.after}' failed; nothing merged (run the job again)")
            self._results.append((st.label, 1))
            self._next_step()
            return
        self.status.setText(f"Step {self._step_i + 1}/{n}: {st.label}" if n > 1 else f"Running {st.label}")
        self.log.appendPlainText(("\n" if self._step_i else "") + "$ " + st.line() + "\n")
        if st.tee:                          # '... 2>&1 | tee <file>': the run log beside the outputs
            try:
                self._tee = open(st.tee, "w")
            except OSError as e:
                self._tee = None
                self.log.appendPlainText(f"!! cannot write the run log {st.tee}: {e}")
        self.bar.setRange(0, 0)             # busy until a partition progress line arrives
        self.eta.setText("")
        penv = QProcessEnvironment.systemEnvironment()
        for k, v in st.env.items():
            penv.insert(k, v)
        self.proc = QProcess(self)
        self.proc.setProcessEnvironment(penv)
        self.proc.setWorkingDirectory(str(rs.REPO_ROOT))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        self.proc.start(st.argv[0], st.argv[1:])
        self.refresh()

    def _on_output(self):
        if self.proc is None:
            return
        raw = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        text = rs.strip_ansi(raw)
        if self._tee is not None:
            try:
                self._tee.write(text)
            except OSError as e:            # the data disk vanished or is full: the run goes on without its copy
                self.log.appendPlainText(f"!! the run log cannot be written any more ({e}); continuing without it")
                self._close_tee()
        self._job_text = (self._job_text + text)[-400_000:]
        self._buffer += text.replace("\r", "\n")
        *lines, self._buffer = self._buffer.split("\n")
        self._consume(lines)

    def _consume(self, lines: list[str]):
        """Progress into the bar and labels; everything else (minus progress fragments) into the log."""
        keep = []
        for line in lines:
            info = rs.parse_progress(line)
            if "planned" in info:                        # the pipeline announcing its snapshots
                self._total_snaps += info["planned"]
                self.snap_label.setText(f"snapshot {self._snaps_started}/{self._total_snaps}")
            if "sim" in info:
                self.status.setText(f"{self._steps[self._step_i].label}: {info['sim']}")
            if "snapshot" in info:
                self._snaps_started += 1
                if self._total_snaps:
                    self.snap_label.setText(f"snapshot {self._snaps_started}/{self._total_snaps}")
                sim = f"{self._job_sim} " if self._job_tab == GRIDS else ""
                self.status.setText(f"Running {sim}snapshot {info['snapshot']:03d}")
                self.bar.setRange(0, 0)
                self.eta.setText("")
            if "total" in info:
                self.bar.setFormat("partitions %v/%m")
                self.bar.setRange(0, info["total"])
                self.bar.setValue(info["done"])
                self.eta.setText(f"{info['elapsed']} elapsed"
                                 + ("" if info["eta"] == "done" else f", ETA ~{info['eta']}"))
            for _path, nbytes in info.get("downloaded", []):
                self._dl_files += 1
                self._dl_bytes += nbytes
                self.snap_label.setText(f"{self._dl_files} file{'s' if self._dl_files != 1 else ''}, "
                                        f"{self._dl_bytes / 1e9:.2f} GB downloaded")
            if "chunk" in info:
                self.bar.setFormat("chunks %v/%m")
                self.bar.setRange(0, info["chunk"][1])
                self.bar.setValue(info["chunk"][0])
            self._problems += info.get("problem", 0)
            if info.get("snapshot_finished"):
                self._snaps_ok += 1
            if "processing complete:" in line:
                self._summary = line.split(":", 1)[1].strip()
            line = rs.PROGRESS.sub("", line)
            if line.strip():
                keep.append(line)
        if keep:
            self.log.appendPlainText("\n".join(keep))

    def _on_error(self, err):
        if err == QProcess.FailedToStart and self.proc is not None:
            st = self._steps[self._step_i]
            self.log.appendPlainText(f"!! could not start {st.argv[0]}: {self.proc.errorString()}")
            self.proc = None
            self._close_tee()               # 'finished' never comes: the run log must not stay open into the next job
            self._results.append((st.label, 127))
            QTimer.singleShot(0, self._next_step)

    def _on_finished(self, code, status):
        if self.proc is None:
            return
        if self._buffer.strip():
            self._consume([self._buffer])
        self._buffer = ""
        st = self._steps[self._step_i]
        code = code if status == QProcess.NormalExit else -1
        self._close_tee()                   # never raises: the job must reach _next_step even when the disk is gone
        self._results.append((st.label, code))
        if code != 0 and not self._stopping:
            self.log.appendPlainText(f"!! {st.label}: exit {code}")
        self.proc = None
        self._next_step()

    def _close_tee(self):
        tee, self._tee = self._tee, None
        if tee is not None:
            try:
                tee.close()
            except OSError as e:
                self.log.appendPlainText(f"!! the run log could not be closed ({e})")

    def _finish_job(self):
        n = len(self._steps)
        ok = sum(1 for _, c in self._results if c == 0)
        if self._stopping:
            self.status.setText("Stopped")
        elif ok == n and self._problems:
            self.status.setText(f"Finished, but {self._problems} line{'s' if self._problems > 1 else ''} "
                                "in the log report a problem")
        elif ok == n:
            self.status.setText("Finished")
        else:
            self.status.setText(f"Finished: {n - ok} of {n} steps failed")
        self.snap_label.setText(self._summary or (f"{ok}/{n} steps OK" if n > 1 else ""))
        self.bar.setRange(0, max(n, 1))
        self.bar.setValue(ok)
        self.bar.setFormat("%v/%m steps OK" if n > 1 else "partitions %v/%m")
        if n == 1 and ok == 1:
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
            self.bar.setFormat("done")
        if self._job_tab == DATA:
            self._fill_sims()               # a newly merged simulation appears everywhere
        self._fill_sim_dependent()          # new outputs, slices and planes
        self._runs_scanned = 0.0            # new run logs: rescan for the estimates and the Runs tab
        if self.results.currentWidget() is self.runs:
            self.runs.reload()
        if self._job_tab in (GRIDS, OWN, PIPELINE) and self.explore.outputs:    # a pipeline writes grids too
            self.explore.refresh_outputs()
        if isinstance(self._job_spec, rs.CustomSpec) and ok == n:
            self._register_custom_output(self._job_spec)     # again: a generated input exists only now
        failed_codes = [c for _, c in self._results if c != 0]
        if failed_codes and not self._stopping:
            self._show_banner(R.diagnose(self._job_text, failed_codes))
        extra = [self.plots.figures_root, self.pipe.figures_root, getattr(self._job_spec, "figures_root", "")]
        count = self.figures.show_run(self._figs_since or self._job_start, [r for r in extra if r])
        fig_tab = self.results.indexOf(self.figures)
        self.results.setTabText(fig_tab, f"Figures ({count} new)" if count else "Figures")
        qjob, self._current_job = self._current_job, None
        if qjob is not None:
            qjob.state = "stopped" if self._stopping else "done" if ok == n else "failed"
            qjob.note = (self._summary or f"{ok}/{n} steps OK") + (
                f", {self._problems} problem line{'s' if self._problems > 1 else ''} in the log" if self._problems else "")
            self._save_queue()
            self._refresh_queue()
        if self._queue_running:
            if self._stopping:
                self._queue_running = False
                self.status.setText("Stopped; the queue is paused")
            elif qjob is not None and qjob.state == "failed" and self.q_stop_on_fail.isChecked():
                self._queue_running = False
                self.status.setText(f"Queue stopped: '{qjob.title}' failed")
                self._notify("Queue stopped", f"'{qjob.title}' failed")
            else:
                QTimer.singleShot(0, self._next_queued)
        else:
            if not self._stopping:
                self._notify("Finished" if ok == n else "Failed", self.status.text()
                             + (f": {self._steps[0].label}" if self._steps else ""))
            if count and ok == n and not self._stopping:
                self.results.setCurrentIndex(fig_tab)
        after, self._after_job = self._after_job, None
        if after is not None and ok == n and not self._stopping:
            after()
        self.refresh()

    def stop_run(self):
        if self.proc is None:
            if self._queue_running:         # waiting for another run: just stop waiting
                self._queue_running = False
                self.status.setText("Queue paused")
                self._refresh_queue()
                self.refresh()
            return
        self._stopping = True
        tree = _descendants(int(self.proc.processId()))
        for pid in reversed(tree):                     # the deepest (PS-DTFE) first
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.proc.terminate()
        # after a grace period, force every process of the run that is still there -- not only
        # the script's shell: a child that outlived its parent would run on unseen
        QTimer.singleShot(5000, lambda: _kill_leftovers(tree))
        self.status.setText("Stopping…")

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
        event.accept()

    # ================================================================ helpers for the new panels
    def status_message(self, text: str):
        self.status.setText(text)

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

    def _register_custom_output(self, spec: rs.CustomSpec):
        """A custom-snapshot job is starting: remember its folder, and keep its settings beside the
        outputs ('<name>.gui.json') -- the Explore tab's query server reads the snapshot from them."""
        root = spec.output_root()
        d = str(root.parent)
        if d not in self.custom_dirs:
            self.custom_dirs.append(d)
            self._save()
        side = spec.settings_sidecar()
        try:
            root.parent.mkdir(parents=True, exist_ok=True)
            Path(str(root) + ".gui.json").write_text(json.dumps(side, indent=1))
        except OSError:
            pass

    def _build_menus(self):
        """The menu bar: Setup, running (Run, Add to Queue, Stop, the memory check), the Explore
        tab's, the queue's, the figure browser's and the Runs browser's actions, and the commands
        behind a run (shown, copied, exported). Each also has its button in the window; the
        _sync_*_actions methods keep the Run, Explore, Queue, Figures and Runs menus in step with
        those buttons."""
        bar = self.menuBar()
        file_menu = bar.addMenu("File")
        self.setup_action = QAction("Setup…", self)
        self.setup_action.setShortcut(QKeySequence("Ctrl+,"))
        self.setup_action.setMenuRole(QAction.MenuRole.NoRole)   # macOS would move 'Setup' into the app menu
        self.setup_action.triggered.connect(lambda: self.run_setup())
        file_menu.addAction(self.setup_action)
        file_menu.addSeparator()
        self.export_action = QAction("Export Commands…", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.triggered.connect(self.export_job)
        file_menu.addAction(self.export_action)
        self.copy_action = QAction("Copy Commands", self)
        self.copy_action.triggered.connect(self.copy_commands)
        bar.addMenu("Edit").addAction(self.copy_action)
        self.show_cmds_action = QAction("Show Commands", self, checkable=True)
        self.show_cmds_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.show_cmds_action.setChecked(self.show_commands)
        self.show_cmds_action.toggled.connect(self._toggle_commands)
        bar.addMenu("View").addAction(self.show_cmds_action)
        run_menu = bar.addMenu("Run")
        for attr, text, keys, slot in (("run_action", "Run", "Ctrl+R", self.start_run),
                                       ("queue_action", "Add to Queue", "Ctrl+Shift+R", self.enqueue),
                                       ("stop_action", "Stop", "Ctrl+.", self.stop_run),
                                       ("mem_action", "Check Memory", "Ctrl+Shift+M", self.check_memory)):
            if attr == "mem_action":
                run_menu.addSeparator()
            action = QAction(text, self, enabled=False)
            action.setShortcut(QKeySequence(keys))
            action.triggered.connect(lambda _=False, slot=slot: slot())
            run_menu.addAction(action)
            setattr(self, attr, action)
        explore_menu = bar.addMenu("Explore")
        for attr, text, keys, what in (("open_output_action", "Open Output…", "Ctrl+O", "open"),
                                       ("find_outputs_action", "Look for Outputs Again", "", "find"),
                                       ("save_image_action", "Save Image…", "Ctrl+S", "save"),
                                       ("zoom_out_action", "Zoom Out", "Ctrl+Shift+O", "zoom"),
                                       ("grid_action", "Figure Grid", "Ctrl+G", "grid"),
                                       ("server_action", "Start Query Server", "", "server")):
            if attr == "server_action":
                explore_menu.addSeparator()
            action = QAction(text, self, enabled=what in ("open", "find"))
            if keys:
                action.setShortcut(QKeySequence(keys))
            action.triggered.connect(lambda _=False, what=what: self._explore_menu(what))
            explore_menu.addAction(action)
            setattr(self, attr, action)
        queue_menu = bar.addMenu("Queue")
        for item in (("q_start_action", "Start Queue", "start"), None,
                     ("q_up_action", "Move Job Up", "up"),
                     ("q_down_action", "Move Job Down", "down"),
                     ("q_remove_action", "Remove Job", "remove"),
                     ("q_clear_action", "Clear Finished Jobs", "clear"), None):
            if item is None:
                queue_menu.addSeparator()
                continue
            attr, text, what = item
            action = QAction(text, self, enabled=False)
            action.triggered.connect(lambda _=False, what=what: self._queue_menu(what))
            queue_menu.addAction(action)
            setattr(self, attr, action)
        # the queue's options (its tick boxes; _build_queue_tab ties each to its box)
        for attr, text in (("q_stop_on_fail_action", "Stop the Queue When a Job Fails"),
                           ("q_wait_action", "Wait While Another DTFE Run Is Going"),
                           ("q_notify_action", "Notify When a Job or the Queue Finishes")):
            action = QAction(text, self, checkable=True)
            queue_menu.addAction(action)
            setattr(self, attr, action)
        figures_menu = bar.addMenu("Figures")
        reveal = "Show in Finder" if sys.platform == "darwin" else "Show Folder"
        for item in (("fig_refresh_action", "Refresh Figures", "refresh"), None,
                     ("fig_open_action", "Open Figure", "open"),
                     ("fig_reveal_action", reveal, "reveal"), None):
            if item is None:
                figures_menu.addSeparator()
                continue
            attr, text, what = item
            action = QAction(text, self, enabled=what == "refresh")
            action.triggered.connect(lambda _=False, what=what: self._figures_menu(what))
            figures_menu.addAction(action)
            setattr(self, attr, action)
        # which figures the browser lists (its drop-down; the Figures tab ties the two together)
        self.fig_mode_group = QActionGroup(self)
        self.fig_mode_actions = []
        for i, text in enumerate(("From the Last Run", "All Figures")):
            action = QAction(text, self.fig_mode_group, checkable=True, checked=i == 0)
            action.triggered.connect(lambda _=False, i=i: self._figures_menu("mode", i))
            figures_menu.addAction(action)
            self.fig_mode_actions.append(action)
        runs_menu = bar.addMenu("Runs")
        reveal = "Show in Finder" if sys.platform == "darwin" else "Show Folder"
        for item in (("runs_refresh_action", "Refresh Runs", "refresh"), None,
                     ("runs_log_action", "Open Log", "log"),
                     ("runs_load_action", "Load Settings", "load"),
                     ("runs_explore_action", "Open in Explore", "explore"),
                     ("runs_reveal_action", reveal, "reveal"), None,
                     ("runs_trash_action", "Move to Trash…", "trash")):
            if item is None:
                runs_menu.addSeparator()
                continue
            attr, text, what = item
            action = QAction(text, self, enabled=what == "refresh")
            action.triggered.connect(lambda _=False, what=what: self._runs_menu(what))
            runs_menu.addAction(action)
            setattr(self, attr, action)

    def _bring_up(self, widget: QWidget):
        """Show one of the results tabs (Queue, Runs), leaving Explore for the last tab with the
        run panel."""
        if self.tabs.currentIndex() == EXPLORE:
            self.tabs.setCurrentIndex(self._panel_tab)
        self.results.setCurrentWidget(widget)

    def _runs_menu(self, what: str):
        """A Runs menu item: its button in the Runs browser. 'Refresh Runs' works from anywhere and
        brings the browser up first; the others act on the run selected there, so they are only
        offered while the browser is on screen."""
        if what == "refresh" and (self.tabs.currentIndex() == EXPLORE or self.results.currentWidget() is not self.runs):
            self._bring_up(self.runs)                               # (showing it reloads it)
            return
        getattr(self.runs, "b_" + what).click()

    def _figures_menu(self, what: str, mode: int = 0):
        """A Figures menu item: its button (or drop-down) in the figure browser. Refreshing and
        choosing which figures to list work from anywhere and bring the browser up; opening and
        revealing act on the figure selected there, so they are only offered while it is on
        screen."""
        if what in ("refresh", "mode"):
            self._bring_up(self.figures)
            if what == "mode":
                self.figures.mode.setCurrentIndex(mode)        # (reloads the list)
            else:
                self.figures.refresh_btn.click()
            return
        {"open": self.figures.open_btn, "reveal": self.figures.reveal_btn}[what].click()

    def _sync_figures_actions(self, tab: int | None = None):
        if not hasattr(self, "figures") or not hasattr(self, "tabs"):
            return
        shown = (self.tabs.currentIndex() if tab is None else tab) != EXPLORE \
            and self.results.currentWidget() is self.figures
        self.fig_open_action.setEnabled(shown and self.figures.open_btn.isEnabled())
        self.fig_reveal_action.setEnabled(shown and self.figures.reveal_btn.isEnabled())
        self.fig_mode_actions[self.figures.mode.currentIndex()].setChecked(True)

    def _queue_menu(self, what: str):
        """A Queue menu item: its button on the Queue tab. Starting the queue (its log comes up) and
        clearing finished jobs (the Queue tab comes up) work from anywhere; moving and removing act
        on the job selected there, so they are only offered while the Queue tab is on screen."""
        if what == "start" and self.tabs.currentIndex() == EXPLORE:
            self.tabs.setCurrentIndex(self._panel_tab)
        elif what == "clear":
            self._bring_up(self.queue_tab)
        getattr(self, "q_" + what).click()

    def _sync_queue_actions(self):
        if not hasattr(self, "queue_tab") or not hasattr(self, "tabs"):
            return
        shown = self.tabs.currentIndex() != EXPLORE and self.results.currentWidget() is self.queue_tab
        self.q_start_action.setText(self.q_start.text().title())      # Start / Resume Queue
        self.q_start_action.setEnabled(self.q_start.isEnabled())
        self.q_clear_action.setEnabled(self.q_clear.isEnabled())
        for what in ("up", "down", "remove"):
            getattr(self, f"q_{what}_action").setEnabled(shown and getattr(self, "q_" + what).isEnabled())

    def _sync_runs_actions(self, tab: int | None = None):
        if not hasattr(self, "results") or not hasattr(self, "tabs"):
            return
        shown = (self.tabs.currentIndex() if tab is None else tab) != EXPLORE \
            and self.results.currentWidget() is self.runs
        for what in ("log", "load", "explore", "reveal", "trash"):
            getattr(self, f"runs_{what}_action").setEnabled(shown and getattr(self.runs, "b_" + what).isEnabled())

    def _explore_menu(self, what: str):
        """An Explore menu item: the same as its button on the Explore tab. Opening or looking for
        outputs works from any tab and brings Explore up first."""
        x = self.explore
        if what in ("open", "find") and self.tabs.currentIndex() != EXPLORE:
            self.tabs.setCurrentIndex(EXPLORE)
        if what == "grid" and self.tabs.currentIndex() != EXPLORE:
            self.tabs.setCurrentIndex(EXPLORE)
        {"open": lambda: x._open_other(), "find": lambda: x.refresh_outputs(),
         "save": lambda: x._save_image(), "zoom": lambda: x.zoom_out(),
         "grid": lambda: x.grid_toggle.setChecked(not x.grid_toggle.isChecked()),
         "server": lambda: x._server_button()}[what]()

    def _sync_explore_actions(self, tab: int | None = None):
        """Save Image needs a map on screen; the server can be started from Explore and stopped
        (or its start cancelled) from anywhere."""
        x = getattr(self, "explore", None)
        if x is None or not hasattr(self, "tabs"):
            return
        on = (self.tabs.currentIndex() if tab is None else tab) == EXPLORE
        self.save_image_action.setEnabled(on and x.can_save())
        self.zoom_out_action.setEnabled(on and x.is_zoomed())
        self.grid_action.setEnabled(x.current is not None)
        self.grid_action.setText("Back to the Map" if x._grid_shown else "Figure Grid")
        phase = x.server_phase()
        self.server_action.setText({"starting": "Cancel Server Start",
                                    "running": "Stop Query Server"}.get(phase, "Start Query Server"))
        self.server_action.setEnabled(x.server_btn.isEnabled() and (on or phase != "stopped"))

    def _sync_run_actions(self, tab: int):
        """The Run menu does what the buttons below the checks do, when they can: Run and Add to
        Queue need the run panel (not Explore), Stop and a running memory check work anywhere."""
        panel = tab != EXPLORE
        self.run_action.setEnabled(panel and self.run_btn.isEnabled())
        self.queue_action.setEnabled(panel and self.queue_btn.isEnabled())
        self.stop_action.setEnabled(self.stop_btn.isEnabled())
        self.mem_action.setText("Cancel Memory Check" if self._mcheck is not None else "Check Memory")
        self.mem_action.setEnabled(self._mcheck is not None
                                   or (tab in (GRIDS, PIPELINE, OWN) and self.mem_btn.isEnabled()))
        self._sync_explore_actions(tab)
        self._sync_figures_actions(tab)
        self._sync_runs_actions(tab)

    def copy_commands(self):
        self._read_widgets()
        steps = self._job().steps()
        if steps:
            QApplication.clipboard().setText(rs.script_text(steps))
            self.status_message(f"copied {len(steps)} command{'s' if len(steps) != 1 else ''}")

    def _toggle_commands(self, on: bool):
        if on == self.show_commands:
            return
        self.show_commands = on
        self.cmd_box.setVisible(on)
        self.show_cmds_action.setChecked(on)
        self._save()
        self.refresh()

    def _results_tab_changed(self, i: int):
        if self.results.widget(i) is self.runs:
            self.runs.reload()
        self._sync_runs_actions()
        self._sync_queue_actions()
        self._sync_figures_actions()

    def _load_run_settings(self, rec: R.RunRecord):
        """The Runs browser's 'Load settings': the run's options into the Grids tab."""
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

    def _notify(self, title: str, text: str):
        if os.environ.get("DTFE_GUI_NO_NOTIFY") == "1" or not self.q_notify.isChecked():
            return
        cmd = R.notify_command(f"DTFE launcher: {title}", text)
        if cmd:
            QProcess.startDetached(cmd[0], cmd[1:])

    # ---------------------------------------------------------------- failure banner
    _ACTION_TEXT = {"check_memory": "Check memory", "scratch": "Set a scratch folder…",
                    "choose_root": "Choose the data folder…", "open_data": "Open the Data tab",
                    "build": "Build the programs…"}

    def _show_banner(self, diagnoses: list[R.Diagnosis]):
        while self.banner_buttons.count():
            w = self.banner_buttons.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        if not diagnoses:
            self.banner.hide()
            return
        self.banner_text.setText("<br>".join(f"<b>{d.title}.</b> {d.advice}" for d in diagnoses))
        for d in diagnoses:
            if d.action in self._ACTION_TEXT:
                b = QPushButton(self._ACTION_TEXT[d.action])
                b.clicked.connect(lambda _=False, a=d.action: self._banner_action(a))
                self.banner_buttons.addWidget(b)
        close = QPushButton("Dismiss")
        close.clicked.connect(self.banner.hide)
        self.banner_buttons.addStretch(1)
        self.banner_buttons.addWidget(close)
        self.banner.show()

    def _banner_action(self, action: str):
        tab = self._job_tab
        if action == "check_memory":
            self.tabs.setCurrentIndex(tab if tab in (GRIDS, PIPELINE, OWN) else GRIDS)
            if self.mem_btn.isEnabled():
                self.check_memory()
        elif action == "scratch":
            t = tab if tab in (GRIDS, PIPELINE, OWN) else GRIDS
            self.tabs.setCurrentIndex(t)
            edit = {GRIDS: self.scratch_edit, PIPELINE: self.p_scratch, OWN: self.o_scratch}[t]
            self._browse_scratch(edit)
        elif action == "choose_root":
            self._browse_root()
        elif action == "open_data":
            self.tabs.setCurrentIndex(DATA)
        elif action == "build":
            self.run_setup(page=1)
        self.banner.hide()

    # ---------------------------------------------------------------- export
    def export_job(self):
        """The current job as a bash script, or a SLURM batch job (e.g. for Hábrók)."""
        self._read_widgets()
        steps = self._job().steps()
        if not steps:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Export the commands")
        f = QFormLayout(dlg)
        kind = QComboBox()
        kind.addItems(["bash script", "SLURM batch job (cluster)"])
        f.addRow("As", kind)
        name = QLineEdit("dtfe")
        wall = QLineEdit("24:00:00")
        cpus = QSpinBox(minimum=1, maximum=256, value=16)
        mem = QLineEdit("64G")
        part = QLineEdit("regular")
        mail = QLineEdit(placeholderText="optional: mail me when it ends")
        rows = [("Job name", name), ("Time limit", wall), ("CPUs", cpus), ("Memory", mem), ("Partition", part),
                ("E-mail", mail)]
        for label, w in rows:
            f.addRow(label, w)
        t = R.time_estimate(self._runs(), sim=self.spec.sim, grid=self.spec.grid, estimator=self.spec.estimator,
                            gpu=False, sliced=bool(self.spec.slice_plane)) if self.tabs.currentIndex() == GRIDS else None
        f.addRow(_note("No GPU on a cluster: GPU options fall back to the CPU. Set REPO and the data paths "
                       "in the script." + (f" Earlier CPU runs: ≈ {R.format_duration(t[0])} per snapshot." if t else "")))

        def sync(i):
            for label, w in rows:
                w.setEnabled(i == 1)
        kind.currentIndexChanged.connect(sync)
        sync(0)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        _grow_fields(dlg)
        dlg.resize(520, dlg.sizeHint().height())
        if dlg.exec() != QDialog.Accepted:
            return
        slurm = None if kind.currentIndex() == 0 else {
            "job_name": name.text().strip(), "time": wall.text().strip(), "cpus": cpus.value(),
            "mem": mem.text().strip(), "partition": part.text().strip(), "mail": mail.text().strip()}
        default = str(Path.home() / (("dtfe_job.slurm" if slurm else "dtfe_job.sh")))
        path, _ = QFileDialog.getSaveFileName(self, "Save the script", default)
        if not path:
            return
        title = rs.job_summary(KIND_OF_TAB.get(self.tabs.currentIndex(), "grids"), self._job())
        Path(path).write_text(rs.export_script(steps, title, slurm))
        os.chmod(path, 0o755)
        self.status_message(f"saved {path}" + ("  (submit with: sbatch " + Path(path).name + ")" if slurm else ""))

    # ---------------------------------------------------------------- setup
    def run_setup(self, page: int = 0) -> bool:
        """The first-run dialog (setup_wizard.py): data folder, programs, API key, demo."""
        import setup_wizard as SW
        dlg = SW.SetupDialog(self, self.root_edit.text().strip())
        if page:
            dlg._go(page)
        ok = dlg.exec() == QDialog.Accepted
        if self.remember:
            self.settings.setValue("setup_done", 1)
        if not ok:
            return False
        root = dlg.data_root()
        if root != self.root_edit.text().strip():
            self.root_edit.setText(root)
            self._root_changed()
        if dlg.run_demo:
            self.start_demo()
        return True

    def start_demo(self):
        """The demo (runspec.demo_spec): generate, run, then open it in Explore."""
        demo = rs.demo_spec()               # (the CPU deposit: faster than the GPU at 32^3 particles)
        self.custom = demo
        self._load_into_widgets()
        self.tabs.setCurrentIndex(OWN)
        self._read_widgets()
        self.refresh()
        if self.proc is not None:
            self.status_message("the demo is set up in the Custom snapshot tab: run it when the current job is done")
            return
        root = str(demo.output_root())
        self._after_job = lambda: self._explore_output(root)
        self._figs_since = time.time()
        self._begin(demo.steps(), OWN, demo, None)

    # ================================================================ memory check
    def check_memory(self):
        """Ask the binary's own auto-tuner (run_ps_dtfe.sh AUTO_TUNE_REPORT=1, or the binary's
        --auto-tune-report for a custom snapshot) what the current Grids, Custom snapshot or Pipeline
        settings need per snapshot. For the pipeline, a snapshot over budget is
        checked again at half the plane size, down to 1024^2 -- which is both the suggestion shown
        and, with 'Adaptive', the plan the pipeline then runs."""
        if self._mcheck is not None:            # the button reads "Cancel check"
            self._cancel_check()
            return
        self._read_widgets()
        tab = self.tabs.currentIndex()
        if tab == GRIDS:
            if self.spec.estimator != "ps" or not self.spec.snapshots:
                return
            key = self.spec.memory_key()
            passes = [(self.spec.sim, None, sorted(self.spec.snapshots))]
        elif tab == OWN:
            if not self._custom_checkable():
                return
            key = self.custom.memory_key()
            passes = [(Path(self.custom.input_file).name, None, [0])]    # the bare report line is snapshot 0
        elif tab == PIPELINE:
            key = self.pipe.memory_key()
            passes = []
            for sim in self.pipe.sims:
                if rs.sim_dir(sim, self.pipe.data_root).is_dir():
                    todo = self.pipe.stale(sim, self.pipe.nu)[0]
                    if todo:
                        passes.append((sim, self.pipe.nu, todo))
            if not passes:
                self.status.setText("Memory check: nothing to compute (every snapshot is up to date)")
                return
        else:
            return
        # the settings as they are NOW: a change during the check must not mix into its later passes
        spec = (rs.RunSpec.from_dict(self.spec.to_dict()) if tab == GRIDS
                else rs.CustomSpec.from_dict(self.custom.to_dict()) if tab == OWN
                else rs.PipelineSpec.from_dict(self.pipe.to_dict()))
        self._mcheck = {"tab": tab, "key": key, "passes": passes, "done": 0, "spec": spec,
                        "first": {sim: list(snaps) for sim, _, snaps in passes}}
        self.mem_results.setdefault(key, {})
        G.drop_cubes()                          # the report measures free memory: not Explore's cached cubes
        self._next_check()
        self.refresh()

    def _next_check(self):
        mc = self._mcheck
        if mc is None:
            return
        if not mc["passes"]:
            self._finish_check()
            return
        sim, nu, snaps = mc["passes"].pop(0)
        step = mc["spec"].check_step() if mc["tab"] in (GRIDS, OWN) else mc["spec"].check_step(sim, snaps, nu)
        mc["current"] = (sim, nu, snaps)
        mc["buf"] = ""
        self.status.setText(f"Checking memory: {sim}" + (f" at {nu}²" if nu else "")
                            + f", {len(snaps)} snapshot{'s' if len(snaps) != 1 else ''}…")
        penv = QProcessEnvironment.systemEnvironment()
        for k, v in step.env.items():
            penv.insert(k, v)
        proc = QProcess(self)
        proc.setProcessEnvironment(penv)
        proc.setWorkingDirectory(str(rs.REPO_ROOT))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: mc.__setitem__("buf", mc["buf"] + bytes(proc.readAllStandardOutput()).decode(errors="replace")))
        proc.finished.connect(lambda code, _status: self._check_pass_done(proc, code))
        proc.errorOccurred.connect(lambda err: self._check_pass_error(proc, err, step.argv[0]))
        mc["proc"] = proc
        proc.start(step.argv[0], step.argv[1:])

    def _check_pass_error(self, proc, err, program: str):
        """The check's process could not start (no 'finished' follows then): before 2026-10-05 _mcheck stayed
        set forever and Run stayed disabled until 'Cancel check'."""
        mc = self._mcheck
        if mc is None or mc.get("proc") is not proc or err != QProcess.FailedToStart:
            return
        self._mcheck = None
        rs.forget_processes()
        self.status.setText(f"Memory check failed: could not start {Path(program).name}")
        self.refresh()

    def _check_pass_done(self, proc, code):
        mc = self._mcheck
        if mc is None or mc.get("proc") is not proc:
            return                                  # cancelled meanwhile
        mc["buf"] += bytes(proc.readAllStandardOutput()).decode(errors="replace")
        sim, nu, snaps = mc["current"]
        reports = rs.parse_reports(mc["buf"])
        results = self.mem_results[mc["key"]]
        over = []
        for n in snaps:
            r = reports.get(n)
            if r is None:
                continue
            results[(sim, n, nu)] = r
            if r.get("over_budget"):
                over.append(n)
        if code != 0 or not any(n in reports for n in snaps):
            mc["failed"] = mc.get("failed", 0) + 1  # a crashed pass, or one that reported nothing, is not "fits"
        mc["done"] += 1
        # pipeline: what does not fit is tried again at half the plane size
        if nu and over and nu // 2 >= rs.ADAPTIVE_MIN_NU:
            mc["passes"].append((sim, nu // 2, over))
        self._next_check()

    def _finish_check(self):
        mc, self._mcheck = self._mcheck, None
        rs.forget_processes()           # the check's own PS-DTFE must not read as "already running"
        results = self.mem_results.get(mc["key"], {})
        if mc["tab"] == PIPELINE:
            plan = {}
            for sim, snaps in mc["first"].items():
                for n in snaps:
                    sizes = sorted({k[2] for k in results if k[0] == sim and k[1] == n}, reverse=True)
                    fits = [nu for nu in sizes if not results[(sim, n, nu)].get("over_budget")]
                    if sizes:
                        plan.setdefault(sim, {})[f"{n:03d}"] = fits[0] if fits else sizes[-1]
            self.pipe.plan = plan
            self.pipe.plan_key = mc["key"]
            self._save()
        checked = [r for k, r in results.items()]
        over = sum(1 for r in checked if r.get("over_budget"))
        failed = mc.get("failed", 0)
        self.status.setText(f"Memory check done: {len(checked)} prediction{'s' if len(checked) != 1 else ''}"
                            + (f", {over} over budget" if over else ", all within budget" if checked else "")
                            + (f"; {failed} pass{'es' if failed != 1 else ''} failed (see the log)" if failed else ""))
        self.refresh()

    def _cancel_check(self):
        mc, self._mcheck = self._mcheck, None
        proc = mc.get("proc") if mc else None
        if proc is not None and proc.state() != QProcess.NotRunning:
            for pid in reversed(_descendants(int(proc.processId()))):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        self.status.setText("Memory check cancelled")
        self.refresh()

    def _custom_checkable(self) -> bool:
        """The Custom tab's memory check needs the phase-space binary and an existing snapshot file."""
        c = self.custom
        return (c.estimator == "ps" and bool(c.input_file) and Path(c.input_file).expanduser().is_file()
                and c.binary().is_file())

    def _memory_lines(self, tab: int) -> list[tuple[str, str]]:
        """What the memory checks found for the CURRENT settings, as check-list lines."""
        if self._mcheck is not None:
            return [("info", "checking memory…")]
        out: list[tuple[str, str]] = []
        if tab == GRIDS and self.spec.estimator == "ps" and self.spec.sim:
            s = self.spec
            res = self.mem_results.get(s.memory_key(), {})
            fits, unchecked = [], []
            for n in sorted(s.snapshots):
                r = res.get((s.sim, n, None))
                if r is None:
                    unchecked.append(n)
                elif r.get("over_budget"):
                    out.append(("warning", f"snapshot {n:03d} needs ~{r['predicted_gb']:.0f} GB, over the "
                                           f"{r['budget_gb']:.0f} GB budget: it will swap and slow the machine "
                                           "(a smaller slice plane, no velocity gradient at the slice, fewer "
                                           "fields or a scratch directory lower it)"))
                else:
                    fits.append(r)
            if fits and not unchecked:
                big = max(fits, key=lambda r: r["predicted_gb"])      # (the budget is measured live, per check)
                out.append(("ok", f"memory: all {len(fits)} snapshot{'s' if len(fits) != 1 else ''} fit (the "
                                  f"largest needs ~{big['predicted_gb']:.0f} of its {big['budget_gb']:.0f} GB budget)"))
            elif unchecked and (s.slice_plane or s.grid >= 512):
                out.append(("info", "memory not checked for " + " ".join(f"{n:03d}" for n in unchecked)
                                    + " yet ('Check memory')"))
        elif tab == OWN and self.custom.estimator == "ps" and self.custom.input_file:
            c = self.custom
            r = self.mem_results.get(c.memory_key(), {}).get((Path(c.input_file).name, 0, None))
            if r is None:
                if c.grid >= 512:
                    out.append(("info", "memory not checked yet ('Check memory')"))
            elif "predicted_gb" not in r:
                out.append(("info", "the memory check gave no prediction for this run"))
            elif r.get("over_budget"):
                out.append(("warning", f"this run needs ~{r['predicted_gb']:.0f} GB, over the {r['budget_gb']:.0f} GB "
                                       "budget: it will swap and slow the machine (a smaller grid, fewer fields "
                                       "or a scratch directory lower it)"))
            else:
                split = (f"{r['partition']}{'²' if c.dim == 2 else '³'} partitions, {r['mc']} at a time"
                         if r.get("partition", 1) > 1 else "one tessellation")
                out.append(("ok", f"memory: fits (~{r['predicted_gb']:.0f} of the {r['budget_gb']:.0f} GB budget; "
                                  f"the auto-tuner plans {split})"))
        elif tab == PIPELINE:
            p = self.pipe
            key = p.memory_key()
            res = self.mem_results.get(key, {})
            planned = p.adaptive and p.plan_key == key
            fits, unchecked = [], []
            for sim in p.sims:
                if not rs.sim_dir(sim, p.data_root).is_dir():
                    continue
                for n in p.stale(sim, p.nu)[0]:
                    nu = (p.planned_nu(sim, n) or p.nu) if planned else p.nu
                    r = res.get((sim, n, nu))
                    if r is None:
                        unchecked.append(f"{n:03d}")
                        continue
                    if not r.get("over_budget"):
                        fits.append(r)
                        continue
                    smaller = sorted((k[2], v) for k, v in res.items()
                                     if k[0] == sim and k[1] == n and k[2] and k[2] < nu and not v.get("over_budget"))
                    hint = (f"{smaller[-1][0]}² fits (~{smaller[-1][1]['predicted_gb']:.0f} GB)" if smaller
                            else "not even 1024² fits" if nu <= rs.ADAPTIVE_MIN_NU else "check smaller sizes")
                    out.append(("warning", f"{sim} {n:03d} at {nu}² needs ~{r['predicted_gb']:.0f} GB, over the "
                                           f"{r['budget_gb']:.0f} GB budget, so it will swap; {hint}"
                                           + ("" if p.adaptive else " (or tick 'Adaptive')")))
            if fits and not unchecked:
                big = max(fits, key=lambda r: r["predicted_gb"])
                out.append(("ok", f"memory: every snapshot to compute fits (the largest needs "
                                  f"~{big['predicted_gb']:.0f} of its {big['budget_gb']:.0f} GB budget)"))
            elif unchecked and not p.plan_only:
                out.append(("info", "memory not checked yet for " + " ".join(unchecked[:12])
                                    + (" …" if len(unchecked) > 12 else "") + " ('Check memory')"))
        return out

    # ================================================================ queue
    def enqueue(self):
        self._read_widgets()
        tab, job = self.tabs.currentIndex(), self._job()
        if any(level == "error" for level, _ in job.problems()):
            return
        qjob = rs.QueuedJob.of(KIND_OF_TAB[tab], job)
        self.queue.append(qjob)
        self._save_queue()
        self._refresh_queue()
        self.q_list.setCurrentRow(len(self.queue) - 1)
        self.results.setCurrentIndex(self.results.indexOf(self.queue_tab))
        if self.proc is None and not self._queue_running:
            self.status.setText(f"Queued: {qjob.title}")

    def start_queue(self):
        if self._queue_running or rs.next_waiting(self.queue) is None or self._mcheck is not None:
            return
        self._queue_running = True
        self._figs_since = time.time()
        if self.proc is None:
            self.log.clear()
        self._next_queued()
        self.refresh()

    def _next_queued(self):
        if not self._queue_running or self.proc is not None:
            return                          # paused, or a job is running (its end calls this again)
        i = rs.next_waiting(self.queue)
        if i is None:
            self._queue_running = False
            counts = Counter(j.state for j in self.queue)
            self.status.setText("Queue finished: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
            self._notify("Queue finished", ", ".join(f"{v} {k}" for k, v in counts.items()))
            self._refresh_queue()
            self.refresh()
            return
        qjob = self.queue[i]
        try:
            spec = qjob.build()
        except (TypeError, ValueError) as e:
            self._skip_queued(qjob, f"unreadable settings: {e}", failed=True)
            return
        if qjob.kind in rs.HEAVY_KINDS and self.q_wait.isChecked():
            busy = rs.running_jobs(max_age=0)
            if busy:
                pid, name = busy[0]
                self.status.setText(f"Queue: waiting for {name} (pid {pid}) to finish before '{qjob.title}'")
                QTimer.singleShot(self._wait_poll_ms, self._next_queued)
                return
        steps = spec.steps()
        errors = [m for level, m in spec.problems() if level == "error"]
        if not steps:
            self._skip_queued(qjob, "nothing left to do", failed=False)
            return
        if errors:
            self._skip_queued(qjob, errors[0], failed=True)
            return
        qjob.state, qjob.note = "running", ""
        self._save_queue()
        self._refresh_queue()
        self._begin(steps, TAB_OF_KIND[qjob.kind], spec, qjob)
        self.refresh()

    def _skip_queued(self, qjob: rs.QueuedJob, why: str, failed: bool):
        """A job that cannot start: 'skipped' (counts as a failure) or 'done' when nothing is left."""
        qjob.state, qjob.note = ("skipped" if failed else "done"), why
        self.log.appendPlainText(f"{'Skipped' if failed else 'Nothing to do for'} '{qjob.title}': {why}")
        self._save_queue()
        self._refresh_queue()
        if failed and self.q_stop_on_fail.isChecked():
            self._queue_running = False
            self.status.setText(f"Queue stopped: '{qjob.title}' could not start ({why})")
            self.refresh()
            return
        QTimer.singleShot(0, self._next_queued)

    def _refresh_queue(self):
        row = self.q_list.currentRow()
        self.q_list.blockSignals(True)
        self.q_list.clear()
        for k, j in enumerate(self.queue, 1):
            it = QListWidgetItem(f"{QUEUE_ICON[j.state]}  {k}. {j.title}" + (f"   ({j.note})" if j.note else ""))
            it.setForeground(QUEUE_COLOR[j.state])
            it.setToolTip(j.title + (f"\n{j.note}" if j.note else ""))
            self.q_list.addItem(it)
        self.q_list.blockSignals(False)
        if self.queue:
            self.q_list.setCurrentRow(min(max(row, 0), len(self.queue) - 1))
        self._show_queued(self.q_list.currentRow())
        waiting = sum(j.state == "waiting" for j in self.queue)
        self.results.setTabText(self.results.indexOf(self.queue_tab),
                                f"Queue ({waiting} waiting)" if waiting else "Queue")
        self._queue_buttons()

    def _queue_buttons(self):
        row = self.q_list.currentRow()
        sel = self.queue[row] if 0 <= row < len(self.queue) else None
        editable = sel is not None and sel is not self._current_job
        self.q_start.setEnabled(not self._queue_running and rs.next_waiting(self.queue) is not None
                                and getattr(self, "_mcheck", None) is None)
        self.q_start.setText("Resume queue" if any(j.state != "waiting" for j in self.queue)
                             and rs.next_waiting(self.queue) is not None else "Start queue")
        self.q_up.setEnabled(editable and row > 0)
        self.q_down.setEnabled(editable and row < len(self.queue) - 1)
        self.q_remove.setEnabled(editable)
        self.q_clear.setEnabled(any(j.state not in ("waiting", "running") for j in self.queue))
        self._sync_queue_actions()

    def _show_queued(self, row: int):
        if not 0 <= row < len(self.queue):
            self.q_detail.setPlainText("Queue jobs from any tab with 'Add to queue'.")
            self._queue_buttons()
            return
        j = self.queue[row]
        text = [f"{j.title}", f"state: {j.state}" + (f" ({j.note})" if j.note else "")]
        try:
            spec = j.build()
            steps = spec.steps()
            text += ["", "Commands if it started now:" if j.state == "waiting" else "Commands (as of now):",
                     rs.script_text(steps) or "(none: nothing left to do)"]
            problems = spec.problems()
            if problems:
                text += ["", "Checks now:"] + [("✖ " if lv == "error" else "⚠ ") + m for lv, m in problems]
        except (TypeError, ValueError) as e:
            text.append(f"unreadable settings: {e}")
        self.q_detail.setPlainText("\n".join(text))
        self._queue_buttons()

    def _move_queued(self, delta: int):
        row = self.q_list.currentRow()
        new = row + delta
        if not (0 <= row < len(self.queue) and 0 <= new < len(self.queue)):
            return
        self.queue[row], self.queue[new] = self.queue[new], self.queue[row]
        self._save_queue()
        self._refresh_queue()
        self.q_list.setCurrentRow(new)

    def _remove_queued(self):
        row = self.q_list.currentRow()
        if 0 <= row < len(self.queue) and self.queue[row] is not self._current_job:
            del self.queue[row]
            self._save_queue()
            self._refresh_queue()

    def _clear_finished(self):
        self.queue = [j for j in self.queue if j.state in ("waiting", "running")]
        self._save_queue()
        self._refresh_queue()

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

    def _save_queue(self, *_):
        if not self.remember:
            return
        self.settings.setValue("queue", json.dumps([j.to_dict() for j in self.queue]))
        self.settings.setValue("queue_options", json.dumps({"stop_on_fail": self.q_stop_on_fail.isChecked(),
                                                            "wait": self.q_wait.isChecked(),
                                                            "notify": self.q_notify.isChecked()}))

    def _restore_json(self, key: str, default):
        raw = self.settings.value(key, "") if self.remember else ""
        try:
            return json.loads(raw) if raw else default
        except (ValueError, TypeError):
            return default

    def _restore_queue(self) -> list[rs.QueuedJob]:
        out = []
        for d in self._restore_json("queue", []):
            try:
                out.append(rs.QueuedJob.from_dict(d))
            except (TypeError, ValueError):
                pass                        # a job from an incompatible older version
        return out

    def _restore(self, key: str, cls):
        raw = self.settings.value(key, "") if self.remember else ""
        try:
            return cls.from_dict(json.loads(raw)) if raw else cls()
        except (ValueError, TypeError):
            return cls()


def _kill_leftovers(pids: list[int]):
    for pid in pids:
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
        f"data root: {root} ({'found' if Path(root).is_dir() else 'NOT found'}), "
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
    _macos_app_name("DTFE Launcher")            # before QApplication builds the menu bar
    app = QApplication(sys.argv)
    app.setApplicationName("DTFE launcher")
    app.setApplicationDisplayName("DTFE Launcher")
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
    working = (bool(w.settings.value("spec")) and Path(w.root_edit.text()).is_dir()
               and (rs.REPO_ROOT / "PS-DTFE").is_file())
    if "--setup" in sys.argv[1:] or (not w.settings.value("setup_done") and not working):
        QTimer.singleShot(300, w.run_setup)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
