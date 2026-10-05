"""The GUI's Runs browser: every output set on disk, read back from its run log (results.py).

One row per run -- simulation, snapshot, output name, grid, options, when, how long, peak memory,
size on disk -- and a status that names every known output-changing fix the run predates (by
the binary's build stamp, else the run date). Actions: open the log, load the run's settings into
the Grids tab (re-run with the same settings), open it in Explore, show it in Finder, move its
outputs to the Trash (never a hard delete).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QFile, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as R  # noqa: E402

LEVEL_COLOR = {"error": QColor("#c62828"), "warning": QColor("#b26a00"), "info": QColor("gray")}
COLUMNS = ["Simulation", "Snap", "Output", "Grid", "Options", "Finished", "Wall", "Peak", "Size", "Status"]


def _gb(x):
    return "—" if x is None else f"{x:.1f} GB"


class RunsBrowser(QWidget):
    load_settings = Signal(object)      # RunRecord -> the Grids tab
    explore = Signal(str)               # output root '<dir>/<prefix>' -> the Explore tab
    message = Signal(str)
    changed = Signal()                  # the buttons' state (the selection): the menu bar follows

    def __init__(self, data_root_fn, extra_dirs_fn):
        super().__init__()
        self._data_root, self._extra_dirs = data_root_fn, extra_dirs_fn
        self.records: list[R.RunRecord] = []
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
        self.b_load.setToolTip("Put this run's settings into the Grids tab, to run it again")
        self.b_load.clicked.connect(lambda: self._sel() and self.load_settings.emit(self._sel()))
        self.b_explore = QPushButton("Explore")
        self.b_explore.clicked.connect(lambda: self._sel() and self.explore.emit(str(self._sel().path.parent / self._sel().prefix)))
        self.b_reveal = QPushButton("Show in Finder" if sys.platform == "darwin" else "Show folder")
        self.b_reveal.clicked.connect(self._reveal)
        self.b_trash = QPushButton("Move to Trash…")
        self.b_trash.clicked.connect(self._trash)
        for b in (self.b_log, self.b_load, self.b_explore, self.b_reveal, self.b_trash):
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
        return len(self.records)

    def _visible(self) -> list[R.RunRecord]:
        words = self.filter.text().lower().split()
        out = []
        for r in self.records:
            stale = R.stale_reasons(r)
            hay = " ".join([r.sim, f"{r.snap:03d}" if r.snap is not None else "", r.prefix, str(r.path),
                            "stale" if stale else "", "failed" if not r.ok else "", r.options_text()]).lower()
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
            cells = [r.sim or "custom", "" if r.snap is None else f"{r.snap:03d}", r.prefix,
                     f"{r.grid}{'²' if r.dim == 2 else '³'}" if r.grid else "—", r.options_text(),
                     r.finished.strftime("%Y-%m-%d %H:%M") if r.finished else "—",
                     R.format_duration(r.wall_seconds), _gb(r.footprint_gb or r.peak_rss_gb),
                     f"{r.size_bytes / 1e9:.1f} GB", status]
            self.table.insertRow(k)
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                it.setData(Qt.UserRole, k)
                if c == len(cells) - 1:
                    it.setForeground(LEVEL_COLOR.get(worst, QColor("#2e7d32")) if r.ok else LEVEL_COLOR["error"])
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
        for b in (self.b_log, self.b_load, self.b_explore, self.b_reveal, self.b_trash):
            b.setEnabled(r is not None)
        if r is None:
            self.detail.setPlainText("Every run with a log on disk: TNG runs under the data root and custom-snapshot "
                                     "runs in their output folders. Select one for its details.")
            return
        self.b_load.setEnabled(bool(r.sim) and r.estimator in ("ps", "dtfe"))
        self.b_explore.setEnabled(any(p.name.endswith((".a_den", ".den")) for p in r.outputs))
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
        files = [p for p in r.outputs if p.is_file()] + [r.path]
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
