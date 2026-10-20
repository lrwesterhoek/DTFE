"""The launcher's own small widgets and widget helpers (moved from app.py, which re-exports CheckDelegate and
FigureBrowser): CheckDelegate, which draws the tick boxes of a checkable list or table; FigureBrowser, the
Figures tab's list of the figures a run wrote, with a preview; and the helpers the window's forms are built
and read with (_mono, _note, _checkable, _grow_fields, _scroll, _dir_row, _exclusive, _checked, _grid_combo,
_state_text, _int)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QImageReader, QIntValidator, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QPushButton, QScrollArea, QAbstractSpinBox, QSizePolicy,
                               QSplitter, QStyle, QStyledItemDelegate, QStyleFactory, QStyleOptionButton,
                               QStyleOptionViewItem, QToolButton, QVBoxLayout, QWidget)

import runspec as rs


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
    appearance) paints the check indicator of the CURRENT row only, so every other row looks as if
    it has none. Fusion's indicator, drawn over the rectangle the view's own style reserves for it
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


def _exclusive(a: QCheckBox, b: QCheckBox):
    """Two options the binary refuses together: ticking one unticks the other."""
    a.toggled.connect(lambda on: on and b.setChecked(False))
    b.toggled.connect(lambda on: on and a.setChecked(False))


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


# ==================================================================== figure browser
class FigureBrowser(QWidget):
    """Figures under the figures root (rs.FIGURES_ROOT): the ones the last run wrote, or all of them, with a preview."""
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
