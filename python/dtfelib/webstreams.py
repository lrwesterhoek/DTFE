"""Stream multiplicity and caustic class by cosmic-web environment (survey item 6, 2026-10-06).

A pure accumulation over the grids of one PS-DTFE output: for every class of the T-web or V-web
(0 void, 1 wall, 2 filament, 3 node; any other label is reported as 'other'), the volume and mass
fractions of the box and, WITHIN the class, the distribution of the stream multiplicity in both
weightings, the mean multiplicity, and -- when the run wrote a 'caustic_class' grid (--ps-caustics) --
the fold fraction and the distribution of the collapse multiplicity k (0 uncollapsed, 1 wall, 2
filament, 3 node; 'none' = no collapse bit at all, a cell never deposited into).

'.streams' is a FLOAT (dtfelib.STREAM_TOL): the averaged 'a_streams' grid is a sub-sample MEAN, so a
partly multi-stream cell reads 1.5 and the bins are named by THRESHOLD, not by a stream count:
    empty         s < 1 - tol            fewer than one stream on average: no sub-sample saw one (a cell
                                         the sampled deposit never reached, 0) or only some did (an averaged
                                         2/3). Counted apart, never as single or multi-stream. How many there
                                         are depends on the run: the TNG300 z=0 sampled run of 2026-07-20 had
                                         7.9% (mostly voids), the production TNG100-3-Dark 512^3 grids 12 cells.
    single        |s - 1| <= tol and no hidden-streams bit 1
    hidden multi  |s - 1| <= tol with the bit: multi-stream volume the count does not show (a fold
                                         thinner than the sampling; velocity_single_stream masks it too)
    1 < s <= 3, 3 < s <= 5, s > 5        (each with the tolerance on its lower edge)
Mass weights are the density grid as stored (rho/rho_bar or the physical density: both are the cell
mass up to one constant, which cancels in every fraction). The grids may be memmaps: the accumulation
runs in slabs along axis 0, every grid sliced with the SAME [start:stop] (FieldSet.iter_slabs picks a
thickness per dtype, so it is not used here): 256 MB of float32 per grid and slab, and add()'s temporaries
(the float64 copies of the streams and the density, the int label arrays, the bincount indices) take
about ten times that, so a run of any size stays within a few GB.

    from dtfelib import webstreams as ws
    table = ws.environment_table(fs.load("tweb", mode="memmap"), fs.load("streams", mode="memmap"),
                                 hidden=fs.load("hidden_streams", mode="memmap"),
                                 density=fs.load("density", mode="memmap"))
    print(ws.table_text(table, "TNG100-3-Dark z=0, T-web"))
"""

from __future__ import annotations

import json
import math

import numpy as np

from .io import STREAM_TOL, caustic_collapse_max, caustic_is_fold, single_stream_mask

WEB_NAMES = ("void", "wall", "filament", "node", "other")
STREAM_BINS = ("empty", "single", "hidden multi", "1 < s <= 3", "3 < s <= 5", "s > 5")
COLLAPSE_BINS = ("none", "k=0", "k=1", "k=2", "k=3")
_OTHER = len(WEB_NAMES) - 1


def web_classes(web) -> np.ndarray:
    """int8 labels 0..3 from a (float) web-class grid, rounded; anything else -> 4 ('other')."""
    w = np.rint(np.asarray(web, dtype=np.float32)).astype(np.int16)
    out = np.full(w.shape, _OTHER, dtype=np.int8)
    inside = (w >= 0) & (w <= 3)
    out[inside] = w[inside].astype(np.int8)
    return out


def stream_bins(streams, hidden=None, tol: float = STREAM_TOL) -> np.ndarray:
    """int8 index into STREAM_BINS per cell (see the module docstring for the thresholds)."""
    s = np.asarray(streams, dtype=np.float32)
    out = np.full(s.shape, 3, dtype=np.int8)                 # 1 < s <= 3
    out[s < 1.0 - tol] = 0
    near = np.abs(s - 1.0) <= tol                            # one stream by the count ...
    single = single_stream_mask(s, hidden, tol)              # ... and no hidden bit: THE rule (io.single_stream_mask)
    out[single] = 1
    out[near & ~single] = 2
    out[s > 3.0 + tol] = 4
    out[s > 5.0 + tol] = 5
    return out


def collapse_bins(caustic_class) -> np.ndarray:
    """int8 index into COLLAPSE_BINS: caustic_collapse_max + 1 (-1 'none' -> 0, k -> k + 1)."""
    return (caustic_collapse_max(caustic_class).astype(np.int8) + 1).astype(np.int8)


class Accumulator:
    """Counts and mass sums per (web class, bin); add() slabs, table() the fractions."""

    def __init__(self, with_caustic: bool, tol: float = STREAM_TOL):
        nw, nb, nc = len(WEB_NAMES), len(STREAM_BINS), len(COLLAPSE_BINS)
        self.tol, self.with_caustic = tol, with_caustic
        self.n = np.zeros((nw, nb), dtype=np.int64)         # cells per class and stream bin
        self.m = np.zeros((nw, nb), dtype=np.float64)       # mass (density sum) per class and stream bin
        self.s_sum = np.zeros(nw, dtype=np.float64)         # sum of the multiplicity per class
        self.s_mass = np.zeros(nw, dtype=np.float64)        # mass-weighted sum of the multiplicity
        self.c = np.zeros((nw, nc), dtype=np.int64)         # cells per class and collapse bin
        self.cm = np.zeros((nw, nc), dtype=np.float64)
        self.fold = np.zeros(nw, dtype=np.int64)            # fold cells per class
        self.fold_m = np.zeros(nw, dtype=np.float64)
        self.mass_weighted = False

    def add(self, web, streams, hidden=None, density=None, caustic_class=None):
        w = web_classes(web).ravel()
        b = stream_bins(streams, hidden, self.tol).ravel()
        s = np.asarray(streams, dtype=np.float64).ravel()
        nw, nb = self.n.shape
        self.n += np.bincount(w.astype(np.int64) * nb + b, minlength=nw * nb).reshape(nw, nb)
        self.s_sum += np.bincount(w, weights=s, minlength=nw)
        if density is not None:
            d = np.asarray(density, dtype=np.float64).ravel()
            self.m += np.bincount(w.astype(np.int64) * nb + b, weights=d, minlength=nw * nb).reshape(nw, nb)
            self.s_mass += np.bincount(w, weights=d * s, minlength=nw)
            self.mass_weighted = True
        if self.with_caustic:
            if caustic_class is None:
                raise ValueError("the accumulator was made with_caustic=True: pass the caustic_class slab")
            cb = collapse_bins(caustic_class).ravel()
            nc = self.c.shape[1]
            self.c += np.bincount(w.astype(np.int64) * nc + cb, minlength=nw * nc).reshape(nw, nc)
            fold = caustic_is_fold(caustic_class).ravel()
            self.fold += np.bincount(w, weights=fold, minlength=nw).astype(np.int64)
            if density is not None:
                self.cm += np.bincount(w.astype(np.int64) * nc + cb, weights=d, minlength=nw * nc).reshape(nw, nc)
                self.fold_m += np.bincount(w, weights=d * fold, minlength=nw)

    @staticmethod
    def _frac(num, den):
        return float(num / den) if den > 0 else None

    def table(self) -> dict:
        total_n, total_m = int(self.n.sum()), float(self.m.sum())
        classes = {}
        for i, name in enumerate(WEB_NAMES):
            cells, mass = int(self.n[i].sum()), float(self.m[i].sum())
            if cells == 0 and name == "other":
                continue
            entry = {
                "cells": cells,
                "volume_fraction": self._frac(cells, total_n),
                "mass_fraction": self._frac(mass, total_m) if self.mass_weighted else None,
                "mean_streams": self._frac(self.s_sum[i], cells),
                "mean_streams_mass": self._frac(self.s_mass[i], mass) if self.mass_weighted else None,
                "stream_bins": {b: {"volume": self._frac(self.n[i, j], cells),
                                    "mass": self._frac(self.m[i, j], mass) if self.mass_weighted else None}
                                for j, b in enumerate(STREAM_BINS)},
            }
            if self.with_caustic:
                entry["fold_fraction"] = {"volume": self._frac(self.fold[i], cells),
                                          "mass": self._frac(self.fold_m[i], mass) if self.mass_weighted else None}
                entry["collapse"] = {b: {"volume": self._frac(self.c[i, j], cells),
                                         "mass": self._frac(self.cm[i, j], mass) if self.mass_weighted else None}
                                     for j, b in enumerate(COLLAPSE_BINS)}
            else:
                entry["fold_fraction"] = entry["collapse"] = None
            classes[name] = entry
        return {"classes": classes, "stream_bins": list(STREAM_BINS), "collapse_bins": list(COLLAPSE_BINS),
                "total_cells": total_n, "with_caustic": self.with_caustic, "mass_weighted": self.mass_weighted,
                "tol": self.tol}


def environment_table(web, streams, hidden=None, density=None, caustic_class=None,
                      tol: float = STREAM_TOL, thickness: int | None = None) -> dict:
    """The table of one output (see the module docstring). Every grid (N,N,N), memmaps welcome;
    'thickness' = planes per slab (default: ~256 MB of float32 per grid)."""
    web = np.asarray(web) if not isinstance(web, np.memmap) else web
    n = web.shape[0]
    if thickness is None:
        plane = int(np.prod(web.shape[1:], dtype=np.int64)) * 4
        thickness = max(1, min(n, (256 << 20) // max(plane, 1)))
    acc = Accumulator(with_caustic=caustic_class is not None, tol=tol)
    for start in range(0, n, thickness):
        stop = min(start + thickness, n)
        sl = slice(start, stop)
        acc.add(web[sl], streams[sl], None if hidden is None else hidden[sl],
                None if density is None else density[sl], None if caustic_class is None else caustic_class[sl])
    return acc.table()


def _pct(v) -> str:
    return "   --  " if v is None else f"{100.0 * v:6.2f}%"


def table_text(table: dict, title: str = "") -> str:
    """A fixed-width text rendering: one block per class, the bins in both weightings."""
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    lines.append(f"cells: {table['total_cells']}   tolerance on s: {table['tol']}   "
                 f"mass weights: {'yes' if table['mass_weighted'] else 'no'}   "
                 f"caustic class: {'yes' if table['with_caustic'] else 'no'}")
    head = f"{'class':9s} {'volume':>8s} {'mass':>8s} {'<s>':>7s} {'<s>_m':>7s} | " + " ".join(f"{b:>12s}" for b in STREAM_BINS)
    lines += ["", "per class: its share of the box, the mean multiplicity, then the stream bins WITHIN the class "
                  "(volume-weighted; mass-weighted on the next line)", head, "-" * len(head)]
    for name, e in table["classes"].items():
        sb = e["stream_bins"]
        ms = "   --  " if e["mean_streams"] is None else f"{e['mean_streams']:7.3f}"
        mm = "   --  " if e["mean_streams_mass"] is None else f"{e['mean_streams_mass']:7.3f}"
        lines.append(f"{name:9s} {_pct(e['volume_fraction']):>8s} {_pct(e['mass_fraction']):>8s} {ms:>7s} {mm:>7s} | "
                     + " ".join(f"{_pct(sb[b]['volume']):>12s}" for b in STREAM_BINS))
        lines.append(f"{'  (mass)':9s} {'':>8s} {'':>8s} {'':>7s} {'':>7s} | "
                     + " ".join(f"{_pct(sb[b]['mass']):>12s}" for b in STREAM_BINS))
    if table["with_caustic"]:
        head = f"{'class':9s} {'fold':>8s} {'fold_m':>8s} | " + " ".join(f"{b:>8s}" for b in COLLAPSE_BINS)
        lines += ["", "caustic class within each class: fold (A2) cells, and the collapse multiplicity "
                      "(volume-weighted; mass-weighted on the next line)", head, "-" * len(head)]
        for name, e in table["classes"].items():
            c, f = e["collapse"], e["fold_fraction"]
            lines.append(f"{name:9s} {_pct(f['volume']):>8s} {_pct(f['mass']):>8s} | "
                         + " ".join(f"{_pct(c[b]['volume']):>8s}" for b in COLLAPSE_BINS))
            lines.append(f"{'  (mass)':9s} {'':>8s} {'':>8s} | " + " ".join(f"{_pct(c[b]['mass']):>8s}" for b in COLLAPSE_BINS))
    return "\n".join(lines) + "\n"


def save_table(table: dict, path, title: str = "") -> None:
    """'<path>.json' (the dict, None where undefined) and '<path>.txt' (table_text). The suffix is
    APPENDED: a path named by its redshift ('..._z0.50') must not lose '.50' to with_suffix (z = 0 and
    z = 0.5 landed on one file, found on the first real run 2026-10-05)."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{p}.json").write_text(json.dumps(table, indent=1))
    Path(f"{p}.txt").write_text(table_text(table, title))


def single_stream_fraction(table: dict, weighting: str = "volume") -> dict:
    """{class: fraction of the class that is single-stream (the 'single' bin)}; NaN where undefined."""
    out = {}
    for name, e in table["classes"].items():
        v = e["stream_bins"]["single"][weighting]
        out[name] = math.nan if v is None else v
    return out
