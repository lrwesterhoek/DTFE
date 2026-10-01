#!/usr/bin/env python3
"""Draw the GUI launcher's icon: a small cosmic web -- haloes joined by filaments of particles,
each particle linked to its nearest neighbours like a Delaunay tessellation -- on a dark rounded
square with the proportions of a macOS app icon. Deterministic (fixed seed), no data needed.

    python make_icon.py OUT.png                 # one 1024 x 1024 PNG
    python make_icon.py OUT.png --iconset DIR   # ... plus DIR/icon_*.png for `iconutil -c icns DIR`

Needs PySide6 and numpy (the GUI's environment); runs headless.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (QBrush, QColor, QGuiApplication, QImage, QPainter, QPainterPath,  # noqa: E402
                           QPen, QRadialGradient)

# (pixel size, file name) of a macOS .iconset
ICONSET = [(16, "icon_16x16.png"), (32, "icon_16x16@2x.png"), (32, "icon_32x32.png"),
           (64, "icon_32x32@2x.png"), (128, "icon_128x128.png"), (256, "icon_128x128@2x.png"),
           (256, "icon_256x256.png"), (512, "icon_256x256@2x.png"), (512, "icon_512x512.png"),
           (1024, "icon_512x512@2x.png")]


def web_points(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Haloes, and particles along the filaments between neighbouring haloes and around them."""
    haloes = np.array([[0.22, 0.28], [0.68, 0.18], [0.80, 0.62], [0.42, 0.52], [0.18, 0.78],
                       [0.58, 0.86], [0.90, 0.36]]) + rng.normal(0, 0.02, (7, 2))
    d = np.linalg.norm(haloes[:, None] - haloes[None], axis=-1)
    pairs = {tuple(sorted((i, int(j)))) for i in range(len(haloes)) for j in np.argsort(d[i])[1:3]}
    pts = []
    for i, j in pairs:
        t = rng.uniform(0, 1, 70)[:, None]
        normal = np.array([haloes[j, 1] - haloes[i, 1], haloes[i, 0] - haloes[j, 0]])
        normal /= np.linalg.norm(normal)
        bend = 0.05 * np.sin(np.pi * t) * rng.choice([-1, 1])      # filaments are not straight
        pts.append(haloes[i] + t * (haloes[j] - haloes[i]) + bend * normal
                   + rng.normal(0, 0.02, (len(t), 2)))
    for h in haloes:
        pts.append(h + rng.normal(0, 0.04, (34, 2)))
    pts.append(rng.uniform(0.02, 0.98, (25, 2)))                  # the sparse voids
    return haloes, np.clip(np.concatenate(pts), 0.0, 1.0)


def draw(size: int = 1024) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    s = size / 1024.0
    body = QRectF(100 * s, 100 * s, 824 * s, 824 * s)          # Apple's grid: 824 px body in 1024
    outline = QPainterPath()
    outline.addRoundedRect(body, 185 * s, 185 * s)
    bg = QRadialGradient(body.center() + QPointF(-140 * s, -160 * s), 760 * s)
    bg.setColorAt(0.0, QColor("#2a3a78"))
    bg.setColorAt(1.0, QColor("#060817"))
    p.fillPath(outline, QBrush(bg))
    p.setClipPath(outline)

    rng = np.random.default_rng(7)
    haloes, pts = web_points(rng)
    xy = lambda q: QPointF(body.left() + q[0] * body.width(), body.top() + q[1] * body.height())  # noqa: E731

    # each particle to its 3 nearest neighbours: the look of a Delaunay tessellation, fainter
    # for the long edges that cross the voids
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    edges = {tuple(sorted((i, int(j)))) for i in range(len(pts)) for j in np.argsort(d[i])[1:4]}
    for i, j in edges:
        length = d[i, j]
        alpha = int(np.clip(200 * (1 - length / 0.12), 12, 200))
        p.setPen(QPen(QColor(126, 178, 255, alpha), max(1.0, 1.8 * s)))
        p.drawLine(xy(pts[i]), xy(pts[j]))
    p.setPen(Qt.NoPen)
    for q in pts:
        p.setBrush(QColor(200, 222, 255, 230))
        p.drawEllipse(xy(q), 3.2 * s, 3.2 * s)
    for h in haloes:                                               # glowing haloes
        glow = QRadialGradient(xy(h), 52 * s)
        glow.setColorAt(0.0, QColor(255, 226, 150, 255))
        glow.setColorAt(0.25, QColor(255, 196, 110, 170))
        glow.setColorAt(1.0, QColor(255, 170, 80, 0))
        p.setBrush(QBrush(glow))
        p.drawEllipse(xy(h), 52 * s, 52 * s)
    p.setClipping(False)
    p.setPen(QPen(QColor(255, 255, 255, 40), 3 * s))              # a faint rim
    p.setBrush(Qt.NoBrush)
    p.drawPath(outline)
    p.end()
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("png", type=Path)
    ap.add_argument("--iconset", type=Path, default=None)
    args = ap.parse_args()
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841  (QPainter needs one)
    big = draw(1024)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    if not big.save(str(args.png)):
        return 1
    if args.iconset:
        args.iconset.mkdir(parents=True, exist_ok=True)
        for px, name in ICONSET:
            # small sizes are drawn afresh (thin lines stay crisp), large ones scaled from 1024
            im = draw(px) if px <= 64 else big.scaled(px, px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if not im.save(str(args.iconset / name)):
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
