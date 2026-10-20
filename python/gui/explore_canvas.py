"""The Explore tab's drawing widgets (explore.py re-exports them): ImageCanvas, the map -- an RGB array scaled to
fit, with its axes, scale bar, vector arrows, void outlines and the figure grid's panel labels, reporting the image
pixel under the mouse, clicks and dragged rectangles -- and ColorBar, the map's colour scale with its ticks and the
quantity's name. Qt and numpy only: no workers, no state of the Explore pane."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QImage, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

import grids as G


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
        self.circles: list | None = None    # grids.void_circles(): [(x, y, rx, ry, k)] in image pixels, the void outlines
        self.dashed: list | None = None     # the same for the R_eff outlines, drawn dashed
        # the figure grid: per circle (circles, then dashed) the tile (x0, y0, w, h) in image pixels it is clipped to
        self.circle_clips: list | None = None
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
        if self.circles or self.dashed:
            self._draw_circles(p, t)
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
        names) and a scale bar of a round length in its lower-left corner."""
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

    def _draw_circles(self, p: QPainter, t: QRectF):
        """The void outlines: white over a dark outline (as the arrows), clipped to the map -- or, for the figure
        grid, each to its own tile (circle_clips); the R_eff ones (dashed) dashed."""
        s = t.width() / self._img.width()
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(Qt.NoBrush)
        solid, dashed = list(self.circles or []), list(self.dashed or [])
        clips = self.circle_clips if self.circle_clips and len(self.circle_clips) == len(solid) + len(dashed) else None
        for j, (x, y, rx, ry, _k) in enumerate(solid + dashed):
            if clips is not None:
                cx, cy, cw, ch = clips[j]
                p.setClipRect(QRectF(t.left() + cx * s, t.top() + cy * s, cw * s, ch * s))
            else:
                p.setClipRect(t)
            style = Qt.DashLine if j >= len(solid) else Qt.SolidLine
            for pen in (QPen(QColor(0, 0, 0, 170), 2.6, style), QPen(QColor("white"), 1.2, style)):
                p.setPen(pen)
                p.drawEllipse(QPointF(t.left() + x * s, t.top() + y * s), rx * s, ry * s)
        p.restore()

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
    """The map's colour scale: the gradient; ticks at round values -- whole decades,
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
