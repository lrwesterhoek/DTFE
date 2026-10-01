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
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QLineF, QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QSizePolicy, QSlider,
    QSpinBox, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grids as G  # noqa: E402

CMAPS = ["viridis", "magma", "inferno", "cividis", "plasma", "RdBu_r", "coolwarm", "gray", "tab10"]
AXES = ["x", "y", "z"]


def default_cache_dir(near: str) -> Path:
    """Where the query server keeps its tessellations: beside the data they come from (a TNG
    output: the data root; your own file's output: its folder) -- that disk has the room, ~0.15 KB
    per tessellation vertex -- unless it is iCloud Drive or read-only, else the user cache."""
    root = Path(near)
    if root.is_dir() and "Mobile Documents" not in str(root.resolve()) and os.access(root, os.W_OK):
        return root / ".dtfe-tessellation-cache"
    base = Path.home() / ("Library/Caches" if sys.platform == "darwin" else ".cache")
    return base / "DTFE" / "tessellations"


# ==================================================================== workers
class SliceWorker(QObject):
    """Reads and colours one slice off the GUI thread; only the latest request is served."""
    done = Signal(object)       # dict: rgb, plane, vrange, request
    failed = Signal(str)

    @Slot(object)
    def render(self, req: dict):
        try:
            out: G.OutputSet = req["output"]
            plane = out.slice(req["field"], req["axis"], req["index"], req["component"])
            if req.get("smooth", 0) > 0:
                try:
                    from scipy.ndimage import gaussian_filter
                    plane = gaussian_filter(plane, req["smooth"], mode="wrap")
                except ImportError:
                    pass
            # image: horizontal = the first remaining axis, vertical = the second, increasing upward
            img = np.ascontiguousarray(plane.T[::-1])
            rgb, vrange = G.colorize(img, req["cmap"], log=req["log"], clip=(req["clip"], 100 - req["clip"]),
                                     symmetric=req["symmetric"])
            arrows = out.arrow_field(req["field"], req["axis"], req["index"], smooth=req.get("smooth", 0)) \
                if req.get("arrows") else None
            self.done.emit({"rgb": np.ascontiguousarray(rgb), "plane": plane, "vrange": vrange, "request": req,
                            "arrows": arrows})
        except Exception as e:                  # a partial file, a vanished disk
            self.failed.emit(f"{type(e).__name__}: {e}")


class ServerWorker(QObject):
    """A dtfelib.Estimator living on its own thread: start (the tessellation build, possibly long),
    query (milliseconds for a few points), stop. Cancel kills a build in progress."""
    started = Signal(str, object)       # (description, the server's region (3, 2) in Mpc)
    answered = Signal(object, object)   # (points, PointFields)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self):
        super().__init__()
        self.est = None
        self.proc = None

    @Slot(object)
    def start(self, kw: dict):
        self.stop()
        try:
            from dtfelib import Estimator
            kw = dict(kw)
            kw["_on_spawn"] = lambda p: setattr(self, "proc", p)
            self.est = Estimator(**kw)
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
            self.failed.emit(f"query failed: {e}")

    @Slot()
    def stop(self):
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


# ==================================================================== canvas
class ImageCanvas(QLabel):
    """Shows an RGB array scaled to fit (aspect kept); reports the image pixel under the mouse."""
    hovered = Signal(int, int)       # image row, column (-1, -1 when outside)
    clicked = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(240, 240)
        self._img: QImage | None = None
        self._buf = None
        self.marker: tuple[int, int] | None = None
        self.arrows: dict | None = None     # grids.OutputSet.arrow_field(), drawn over the map
        self.setText("Choose an output on the left.")
        self.setStyleSheet("color: gray")

    def set_rgb(self, rgb: np.ndarray | None):
        if rgb is None:
            self._img, self._buf = None, None
            self.setPixmap(QPixmap())
            return
        self._buf = rgb                    # keep the bytes alive for the QImage
        h, w, _ = rgb.shape
        self._img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._redraw()

    def _target(self) -> QRectF | None:
        if self._img is None:
            return None
        w, h = self._img.width(), self._img.height()
        s = min(self.width() / w, self.height() / h)
        return QRectF((self.width() - w * s) / 2, (self.height() - h * s) / 2, w * s, h * s)

    def _redraw(self):
        t = self._target()
        if t is None:
            return
        pm = QPixmap(self.size())
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        # smooth when shrinking a big slice; crisp cells when a small grid is blown up
        p.setRenderHint(QPainter.SmoothPixmapTransform, self._img.width() > t.width())
        p.drawImage(t, self._img)
        if self.arrows is not None:
            self._draw_arrows(p, t)
        if self.marker is not None:
            r, c = self.marker
            s = t.width() / self._img.width()
            x, y = t.left() + (c + 0.5) * s, t.top() + (r + 0.5) * s
            p.setPen(QPen(QColor("white"), 2))
            p.drawEllipse(QPointF(x, y), 7, 7)
            p.setPen(QPen(QColor("black"), 1))
            p.drawEllipse(QPointF(x, y), 9, 9)
        p.end()
        self.setPixmap(pm)

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

    def mouseMoveEvent(self, e):
        self.hovered.emit(*self._pixel(e.position()))

    def leaveEvent(self, e):
        self.hovered.emit(-1, -1)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            r, c = self._pixel(e.position())
            if r >= 0:
                self.clicked.emit(r, c)


class ColorBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(34)
        self.cmap, self.vrange, self.label = "viridis", (0.0, 1.0), ""

    def set(self, cmap: str, vrange, label: str):
        self.cmap, self.vrange, self.label = cmap, vrange, label
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        lut = G.lut(self.cmap)
        g = QLinearGradient(0, 0, self.width(), 0)
        for i in range(0, 256, 16):
            g.setColorAt(i / 255, QColor(*[int(v) for v in lut[i]]))
        g.setColorAt(1.0, QColor(*[int(v) for v in lut[255]]))
        p.fillRect(0, 0, self.width(), 14, g)
        p.setPen(self.palette().windowText().color())
        lo, hi = self.vrange
        p.drawText(0, 30, f"{lo:.3g}")
        p.drawText(QRectF(0, 16, self.width(), 18), Qt.AlignHCenter, self.label)
        p.drawText(QRectF(0, 16, self.width(), 18), Qt.AlignRight, f"{hi:.3g}")
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
    _to_start = Signal(object)
    _to_query = Signal(object)
    _to_stop = Signal()

    def __init__(self, view: ExploreView, data_root_fn, extra_dirs_fn):
        super().__init__()
        self.view = view
        self._data_root = data_root_fn          # () -> current data root
        self._extra_dirs = extra_dirs_fn        # () -> the custom-snapshot output directories
        self.outputs: list[tuple[str, G.OutputSet]] = []
        self.current: G.OutputSet | None = None
        self._plane = None
        self._arrows = None
        self._req = None
        self._pending = None
        self._busy = False
        self._server_output: G.OutputSet | None = None
        self._server_log: Path | None = None

        col = QVBoxLayout(self)
        g = QGroupBox("Output")
        f = QFormLayout(g)
        self.source = QComboBox()
        self.source.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.source.setMinimumContentsLength(20)
        self.source.currentIndexChanged.connect(self._source_changed)
        row = QHBoxLayout()
        row.addWidget(self.source, 1)
        refresh = QToolButton(text="↻")
        refresh.setToolTip("Look for outputs again")
        refresh.clicked.connect(self.refresh_outputs)
        row.addWidget(refresh)
        open_btn = QToolButton(text="…")
        open_btn.setToolTip("Open any output: choose one of its grid files (e.g. <name>.a_den)")
        open_btn.clicked.connect(self._open_other)
        row.addWidget(open_btn)
        f.addRow("Run", row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray")
        f.addRow("", self.info)
        col.addWidget(g)

        g = QGroupBox("Map")
        f = QFormLayout(g)
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
        self.smooth = QDoubleSpinBox(minimum=0.0, maximum=20.0, decimals=1, singleStep=0.5)
        self.smooth.setSuffix(" cells (Gaussian)")
        self.smooth.valueChanged.connect(self.render)
        f.addRow("Smoothing", self.smooth)
        col.addWidget(g)

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
        brow = QHBoxLayout()
        self.server_btn = QPushButton("Start query server")
        self.server_btn.clicked.connect(self._server_button)
        brow.addWidget(self.server_btn)
        self.server_state = QLabel("not running")
        self.server_state.setWordWrap(True)
        brow.addWidget(self.server_state, 1)
        v.addLayout(brow)
        col.addWidget(g)
        col.addStretch(1)

        # workers: their threads start on first use (a window that never opens Explore owns none)
        self._slice_thread = self._server_thread = None
        self._slice_worker = SliceWorker()
        self._slice_worker.done.connect(self._rendered)
        self._slice_worker.failed.connect(self._render_failed)
        self._to_render.connect(self._slice_worker.render)
        self._server = ServerWorker()
        self._server.started.connect(self._server_started)
        self._server.failed.connect(self._server_failed)
        self._server.answered.connect(self._answered)
        self._server.stopped.connect(lambda: self._set_server_state("stopped"))
        self._to_start.connect(self._server.start)
        self._to_query.connect(self._server.query)
        self._to_stop.connect(self._server.stop)
        self._server_running = self._server_starting = False
        self._shut = False                      # shutdown() ran: nothing may start a thread again
        self._build_timer = QTimer(self, interval=1000)
        self._build_timer.timeout.connect(self._poll_build)

        view.canvas.hovered.connect(self._hover)
        view.canvas.clicked.connect(self._click)
        view.save_btn.clicked.connect(self._save_image)
        self._refresh_server_button()

    def _threads(self):
        if self._slice_thread is None:
            self._slice_thread, self._server_thread = QThread(self), QThread(self)
            self._slice_worker.moveToThread(self._slice_thread)
            self._server.moveToThread(self._server_thread)
            self._slice_thread.start()
            self._server_thread.start()

    @staticmethod
    def _note(text):
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet("color: gray")
        return lab

    # ---------------------------------------------------------------- outputs
    def refresh_outputs(self, select: str | None = None):
        keep = select or (str(self.current.root) if self.current else None)
        self.outputs = G.find_outputs(self._data_root(), self._extra_dirs())
        self.source.blockSignals(True)
        self.source.clear()
        for title, o in self.outputs:
            self.source.addItem(title, str(o.root))
        self.source.blockSignals(False)
        idx = max(self.source.findData(keep), 0) if keep else 0
        if self.outputs:
            self.source.setCurrentIndex(idx)
            self._source_changed(idx)
        else:
            self.current = self._plane = self._arrows = None
            self.view.canvas.arrows = None
            self.info.setText("No grids yet: run the demo (Setup), a Grids job or a Custom snapshot job first.")
            self.view.canvas.set_rgb(None)
            self.view.canvas.setText("No outputs with grids found.")
            self._refresh_server_button()
        return len(self.outputs)

    def select_root(self, root: str):
        """Show the output whose files are '<root>.*' (the Runs browser's 'Explore')."""
        i = self.source.findData(str(root))
        if i < 0:
            self.refresh_outputs(select=str(root))
        else:
            self.source.setCurrentIndex(i)

    def _open_other(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open an output: choose one of its grid files",
                                           self._data_root(), "Grid files (*.a_den *.den *.a_vel *.vel);;All files (*)")
        if not f:
            return
        root = Path(f).parent / Path(f).name.split(".")[0]
        o = G.OutputSet(root, box=G.custom_box(G.sidecar(root) or {}), title=f"opened · {root}")
        if not o:
            self.status.emit(f"no readable grids next to {f}")
            return
        self.outputs.append((o.title, o))
        self.source.addItem(o.title, str(o.root))
        self.source.setCurrentIndex(self.source.count() - 1)

    def _source_changed(self, i):
        if not 0 <= i < len(self.outputs):
            return
        o = self.outputs[i][1]
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
        self.info.setText(f"{o.n}³ grid · {box}" + (f" · z = {o.redshift:.2f}" if o.redshift is not None else ""))
        self.info.setToolTip(str(o.root) + ".*")
        for w in (self.index, self.slider):
            w.blockSignals(True)
            w.setRange(0, o.n - 1)
            w.setValue(o.n // 2)
            w.blockSignals(False)
        self._field_changed()
        self._refresh_server_button()

    def _field_changed(self, *_):
        o = self.current
        name = self.field.currentData()
        if o is None or name is None:
            return
        ff = o.files[name]
        self.component.blockSignals(True)
        self.component.clear()
        self.component.addItems(G.COMPONENTS.get(ff.ncomp, ["value"]) if ff.ncomp > 1 else ["value"])
        self.component.setEnabled(ff.ncomp > 1)
        self.component.blockSignals(False)
        vec = name in G.VECTOR_FIELDS and ff.ncomp == 3
        self.vectors.blockSignals(True)
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
        return {"output": o, "field": name, "axis": self.axis.currentIndex(), "index": self.index.value(),
                "component": self.component.currentText(), "cmap": self.cmap.currentText(),
                "log": self.log.isChecked(), "clip": self.clip.value(), "smooth": self.smooth.value(),
                "symmetric": name in G.DIVERGING and not self.log.isChecked(),
                "arrows": self.vectors.isEnabled() and self.vectors.isChecked()}

    def render(self, *_):
        req = self.request()
        if req is None or self._shut:
            return
        o = req["output"]
        pos = o.lo[req["axis"]] + (req["index"] + 0.5) * o.length[req["axis"]] / o.n
        unit = "Mpc" if o.box is not None else "cells"
        self.position.setText(f"{AXES[req['axis']]} = {pos:.2f} {unit}")
        if self._busy:
            self._pending = req             # only the latest request is served
            return
        self._busy = True
        self._req = req
        self._threads()
        self._to_render.emit(req)

    def _rendered(self, res: dict):
        self._busy = False
        if self._pending is not None:
            req, self._pending = self._pending, None
            self.render_request(req)
        req = res["request"]
        self._plane = res["plane"]
        self._req = req
        self._arrows = res.get("arrows")
        self.view.canvas.setStyleSheet("")
        self.view.canvas.marker = None
        self.view.canvas.arrows = self._arrows
        self.view.canvas.set_rgb(res["rgb"])
        label = G.FIELD_LABELS.get(req["field"], req["field"])
        comp = "" if req["component"] in ("value", "") else f" [{req['component']}]"
        self.view.colorbar.set(req["cmap"], res["vrange"], ("log₁₀ " if req["log"] else "") + label + comp)
        o = req["output"]
        self.view.title.setText(f"<b>{label}{comp}</b> — {o.title} — {self.position.text()}")
        self.changed.emit()

    def render_request(self, req):
        self._busy = True
        self._to_render.emit(req)

    def _render_failed(self, msg):
        self._busy = False
        self._pending = self._plane = self._arrows = None
        self.view.canvas.arrows = None
        self.view.canvas.set_rgb(None)
        self.view.canvas.setText(f"Could not read this slice: {msg}")
        self.changed.emit()

    def can_save(self) -> bool:
        return self._plane is not None

    def _cell(self, r: int, c: int):
        """Image pixel -> the 3D cell (i, j, k)."""
        req, plane = self._req, self._plane
        if req is None or plane is None:
            return None
        n0, n1 = plane.shape                  # plane rows = first remaining axis
        row, col = c, n1 - 1 - r              # image = plane.T[::-1]
        if not (0 <= row < n0 and 0 <= col < n1):
            return None
        return req["output"].plane_point(req["axis"], req["index"], row, col), plane[row, col]

    def _hover(self, r, c):
        if r < 0:
            self.view.readout.setText(" ")
            return
        cell = self._cell(r, c)
        if cell is None:
            return
        ijk, value = cell
        o = self._req["output"]
        x = o.cell_center(ijk)
        unit = "Mpc" if o.box is not None else "cells"
        self.view.readout.setText(f"x = {x[0]:.2f}   y = {x[1]:.2f}   z = {x[2]:.2f} {unit}      "
                                  f"cell {tuple(int(v) for v in ijk)}      value = {value:.5g}")

    def _save_image(self):
        if self._plane is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Save the map", str(Path.home() / "explore.png"), "PNG (*.png)")
        if not f:
            return
        req = self._req
        try:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            o = req["output"]
            rest = [a for a in range(3) if a != req["axis"]]
            data = self._plane.T
            if req["log"]:
                data = np.log10(np.where(data > 0, data, np.nan))
            ext = [o.lo[rest[0]], o.lo[rest[0]] + o.length[rest[0]], o.lo[rest[1]], o.lo[rest[1]] + o.length[rest[1]]]
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
        kw = G.server_settings(o)
        if kw is None:
            return None
        cache = Path(self.cache_edit.text().strip() or default_cache_dir(str(o.dir))).expanduser()
        kw.update(per_stream=True, lagrangian_positions=bool(kw.get("phase_space", True)),
                  caustics=bool(kw.get("phase_space", True)), tessellation_cache=str(cache), verbose=1)
        return kw

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
            ok = self.current is not None and G.server_settings(self.current) is not None
            self.server_btn.setEnabled(ok)
            if self.current is not None and not ok:
                self.server_state.setText("no snapshot known for this output (its input file is not beside it)")
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
        self._server_log = Path(tempfile.mkstemp(prefix="dtfe-explore-", suffix=".log")[1])
        kw["log_path"] = str(self._server_log)
        self._server_output = self.current
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
            if o is self.current:
                self._source_changed(self.source.currentIndex())
        self._set_server_state(f"running: {info}. Click the map.")
        self.status.emit(f"query server ready ({info})")

    def _server_failed(self, msg):
        if self._server_starting:
            self._server_starting = False
            self._build_timer.stop()
            self._set_server_state("failed: " + msg)
        else:
            self.view.answer_title.setText("⚠ " + msg)

    def _click(self, r, c):
        cell = self._cell(r, c)
        if cell is None:
            return
        self.view.canvas.marker = (r, c)
        self.view.canvas._redraw()
        ijk, value = cell
        o = self._req["output"]
        x = o.cell_center(ijk)
        if not self._server_running or self._server_output is not o:
            self.view.answer_title.setText(f"<b>cell {tuple(int(v) for v in ijk)}</b>: {value:.5g}. "
                                           "Start the query server (left) to see the streams at this point.")
            self.view.streams.setRowCount(0)
            return
        self.view.answer_title.setText("asking…")
        self._to_query.emit(x.reshape(1, 3))

    def _answered(self, pts, f):
        x = pts[0]
        n = int(f.streams[0])
        caustic = ""
        if f.caustic is not None and int(f.caustic[0]) & 3 == 3:
            caustic = " · on a fold caustic"
        sigma = float(np.sqrt(max(f.dispersion[0][0] + f.dispersion[0][3] + f.dispersion[0][5], 0.0)))
        self.view.answer_title.setText(
            f"<b>({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}) Mpc</b>: {n} stream{'s' if n != 1 else ''} · "
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
            cells = [str(k + 1), f"{f.stream_density[i]:.4g}", f"({v[0]:.0f}, {v[1]:.0f}, {v[2]:.0f})",
                     f"{np.linalg.norm(v):.0f}",
                     "—" if q is None else f"({q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f})",
                     "—" if sval is None else f"{float(np.ravel(sval)[0]):.4g}"]
            t.insertRow(k)
            for c, text in enumerate(cells):
                t.setItem(k, c, QTableWidgetItem(text))

    def shutdown(self):
        """Stop the server and the threads (the window is closing). Idempotent."""
        self._shut = True
        self._server.cancel()
        if self._slice_thread is None:
            self._server.stop()
            return
        self._to_stop.emit()
        for th in (self._slice_thread, self._server_thread):
            th.quit()
            th.wait(5000)
        self._slice_thread = self._server_thread = None
