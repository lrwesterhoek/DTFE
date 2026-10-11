#!/usr/bin/env python3
"""The Explore tab's void features of 2026-10-11, offscreen, on a synthetic snapshot (no T7, no binary):

  1. the void inspector: hovering an outline names the void (R_v, R_eff, delta, flags), clicking it fills the
     inspector (every R_v definition the catalogue has, R_eff, central delta, flags, centre), 'Go to its centre'
     moves the slice there, 'R_eff too' adds dashed outlines;
  2. the outlines carried into the saved PNG (matplotlib circles) with a CSV of the voids shown beside it
     (tools/void_outlines.write_csv), and into the figure grid (each panel's own catalogue, its tiles clipped);
  3. the void smoothing: 'production' or a sigma in Mpc, passed to the helper as --sigma-mpc, its own cache key and
     cache namespace;
  4. the memory guard: no question while no DTFE run is going; with one, a question (free memory said), and a
     'no' starts nothing (server, exact zoom, catalogue); a need that fits in the free memory less 4 GB starts
     without a question even then.

The pure parts (grids.void_at, void_circles_mpc, void_info, void_outlines.write_csv) run without Qt; the rest needs
PySide6 and reports SKIP (exit 0) without it. Usage: ~/.venvs/dtfe-gui/bin/python tests/py_gui_explore_test.py
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "gui"))
sys.path.insert(0, str(ROOT / "python" / "tools"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["DTFE_GUI_NO_NOTIFY"] = "1"
TMP = Path(tempfile.mkdtemp(prefix="gui_explore_"))
os.environ["DTFE_FIGURES_ROOT"] = str(TMP / "figures")     # BEFORE runspec is imported: no test figure on the T7

import grids as G  # noqa: E402
import void_outlines as VO  # noqa: E402

PASS, FAIL = [], []
SIM = "SYNEXP-1-Dark"
N, BOX = 64, 64.0


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"   [{detail}]"))


def make_snapdir(data: Path, snap: int = 50, z: float = 0.0, seed: int = 7) -> Path:
    """data/SIM/snapdir_NNN: a combined header (box, redshift) and an N^3 PS-DTFE set ('ps_output': density,
    velocity, streams) of six broad Gaussian dips on 2% noise -- voids at the production smoothing (10 cells)."""
    import h5py
    d = data / SIM / f"snapdir_{snap:03d}"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / f"combined_{snap:03d}.hdf5", "w") as f:
        h = f.create_group("Header").attrs
        h["BoxSize"], h["Redshift"], h["HubbleParam"], h["HFreeUnits"] = BOX * 1000.0, float(z), 0.6774, 1
        h["NumPart_ThisFile"], h["MassTable"] = np.array([0, N ** 3, 0, 0, 0, 0]), np.array([0, 1.0, 0, 0, 0, 0])
        h["NumPart_Total"] = np.array([0, N ** 3, 0, 0, 0, 0])
    rho = 1.0 + 0.02 * np.random.default_rng(seed).normal(size=(N, N, N))
    ii = np.indices((N, N, N))
    for c, w, depth in (((16, 16, 32), 6.0, 0.9), ((46, 20, 30), 7.0, 0.85), ((20, 48, 34), 5.0, 0.9),
                        ((50, 50, 28), 8.0, 0.8), ((8, 36, 54), 6.0, 0.85), ((36, 6, 10), 5.0, 0.8)):
        r2 = sum((((ii[k] - c[k] + N // 2) % N) - N // 2) ** 2 for k in range(3))
        rho -= depth * np.exp(-0.5 * r2 / w ** 2)
    rho.astype(np.float32).tofile(str(d / "ps_output.a_den"))
    np.zeros((N, N, N, 3), np.float32).tofile(str(d / "ps_output.a_vel"))
    np.ones((N, N, N), np.float32).tofile(str(d / "ps_output.a_streams"))
    return d


# ---------------------------------------------------------------------- pure parts (no Qt)
def pure():
    print("pure parts")
    circles = [(10.0, 10.0, 5.0, 5.0, 3), (11.0, 10.0, 2.0, 2.0, 7), (40.0, 40.0, 4.0, 2.0, 9)]
    check("void_at: the smallest outline containing the point (a small void inside a big one stays reachable), "
          "None outside, ellipses (non-square pixels) respected",
          G.void_at(circles, 11.5, 10.0) == 7 and G.void_at(circles, 6.0, 10.0) == 3 and G.void_at(circles, 30, 30) is None
          and G.void_at(circles, 43.5, 40.0) == 9 and G.void_at(circles, 40.0, 43.0) is None and G.void_at(None, 1, 1) is None)
    cen = np.array([[10.0, 20.0, 30.0], [62.5, 5.0, 31.0]])
    rad = np.array([4.0, 3.0])
    lo, hi = (0.0, 0.0), (64.0, 64.0)
    px = G.void_circles(cen, rad, 64.0, 2, 30.5, [0, 1], lo, hi, 128, 128)
    mp = G.void_circles_mpc(cen, rad, 64.0, 2, 30.5, [0, 1], lo, hi)
    same = len(px) == len(mp) and all(abs(u - x / 2.0) < 1e-9 and abs(v - (64.0 - y / 2.0)) < 1e-9 and abs(r - rx / 2.0) < 1e-9
                                      and k == kk for (x, y, rx, _ry, k), (u, v, r, kk) in zip(px, mp))
    check("void_circles_mpc = void_circles in the data frame (Mpc, y up), periodic images included",
          same and any(abs(u - 62.5) < 1e-9 for u, *_ in mp) and any(u < 0 for u, *_ in mp), f"{px} {mp}")
    res = {"sim": "S", "snap": 5, "method": "ps", "prefix": "", "box_mpc": 64.0, "n_grid": 64, "sigma_mpc": 2.0,
           "centres_mpc": [[1.5, 2.5, 3.5], [10.0, 11.0, 12.0]], "r_v_mpc": [4.0, None], "r_eff_mpc": [6.5, None],
           "delta": [-0.9, -0.5], "resolved": [True, False], "distinct": [True, False],
           "r_v_defs": {"crossing": [4.0, None], "delta08": [3.0, None], "ridge": [5.5, None]}}
    long, short = G.void_info(res, 0), G.void_info(res, 0, short=True)
    check("void_info: R_v of every definition the catalogue has, R_eff, central delta, flags, centre",
          all(t in long for t in ("4.00 Mpc", "3.00 Mpc", "5.50 Mpc", "6.50 Mpc", "-0.900", "resolved", "distinct",
                                  "1.50, 2.50, 3.50", "ridge")) and "R_v 4.00 Mpc" in short and "--" not in long,
          long)
    check("void_info: an undefined radius says 'none', the flags say 'not'",
          "R_v none" in G.void_info(res, 1, short=True) and "not resolved" in G.void_info(res, 1), G.void_info(res, 1))
    path = VO.write_csv(res, TMP / "pure.csv", rows=[1, 0])
    lines = Path(path).read_text().splitlines()
    head = lines[1].split(",")
    check("write_csv: a '#' provenance line, the header with every R_v definition, the rows asked for in order, "
          "empty fields where undefined",
          lines[0].startswith("# S snapshot 5 ps") and head == ["void", "x_mpc", "y_mpc", "z_mpc", "r_v_mpc",
                                                               "r_v_delta08_mpc", "r_v_ridge_mpc", "r_eff_mpc", "delta",
                                                               "resolved", "distinct"]
          and lines[2].startswith("1,10,11,12,,,,,-0.5,0,0") and lines[3].startswith("0,1.5,2.5,3.5,4,3,5.5,6.5,-0.9,1,1"),
          "\n".join(lines))
    # a cached catalogue whose radii voids() would re-measure is not 'cached' to the launcher, and the memory gate
    # applies (2026-10-11: 'r_v_cells' present was enough, whatever settings measured it): pipeline.catalog_r_v_ok
    from dtfelib import pipeline
    r3 = {k: np.ones(2) for k in ("r_v_cells", "r_v_delta08_cells", "r_v_ridge_cells")}
    old_set = pipeline.r_v_settings() + np.eye(5)[1]           # another R_V_MAX_SIGMA
    for name, arrays in (("today", dict(r3, r_v_settings=pipeline.r_v_settings())), ("older", dict(r3, r_v_settings=old_set)),
                         ("unrecorded", r3), ("no_rv", {})):
        np.savez(TMP / f"rv_{name}.npz", coords=np.zeros((2, 3)), **arrays)

    class Cached:                                               # a SnapshotProducts whose cache passed the grid stamp
        def __init__(self, path):
            self.path = path

        def _cached(self, _name):
            return self.path
    st = {n: VO.r_v_stale(TMP / f"rv_{n}.npz") for n in ("today", "older", "unrecorded", "no_rv")}
    check("r_v_stale / needs_build: the pipeline's own R_v test; older or unrecorded settings are stale and gated",
          st["today"] is None and all(st[n] for n in ("older", "unrecorded", "no_rv"))
          and "older settings" in st["older"] and not any("--" in str(v) for v in st.values())
          and not VO.needs_build(Cached(TMP / "rv_today.npz")) and VO.needs_build(Cached(TMP / "rv_older.npz"))
          and VO.needs_build(Cached(None)), st)


# ---------------------------------------------------------------------- the GUI
def spin(qa, ms):
    end = time.time() + ms / 1000.0
    while time.time() < end:
        qa.processEvents()
        time.sleep(0.01)


def wait_for(qa, cond, limit):
    end = time.time() + limit
    while time.time() < end:
        qa.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def gui():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (GUI part): PySide6 is not importable ({e})")
        return
    import explore as E
    import runspec as rs
    print("GUI")
    qa = QApplication.instance() or QApplication([])
    data = TMP / "data"
    make_snapdir(data, 50, 0.0, 7)
    make_snapdir(data, 99, 0.0, 8)
    E.VOID_CACHE_DIR = TMP / "void cache"
    view = E.ExploreView()
    x = E.ExploreControls(view, lambda: str(data), lambda: [])
    view.resize(900, 800)
    asked = []
    x.confirm_heavy = lambda title, text: asked.append((title, text)) or False
    real_procs = rs.dtfe_processes
    rs.dtfe_processes = lambda max_age=2.0: {"runs": [], "servers": []}      # no DTFE run going
    try:
        x.refresh_outputs()
        ok = wait_for(qa, lambda: not x._busy and x.current is not None and x._plane is not None, 30)
        check("the synthetic snapshot is shown", ok and x.current.sim == SIM, str(x.current and x.current.root))
        x.voids.setChecked(True)
        ok = wait_for(qa, lambda: x._void_proc is None and bool(x._void_cache), 180)
        res = next(iter(x._void_cache.values()), {})
        check("voids ticked with no run going: the helper runs without a question; its result carries R_eff and "
              "the R_v definitions", ok and "error" not in res and not asked and "r_v_defs" in res
              and "crossing" in res["r_v_defs"] and list(x._void_cache)[0][-1] == 0.0,
              str(res.get("error", ""))[:300])
        c = view.canvas
        # the slice through a void's centre: there the outline is biggest
        sel = [k for k in np.nonzero(res.get("_shown", []))[0] if np.isfinite(res["_r_v"][k])]
        check("the fixture has distinct voids with R_v", len(sel) > 0, f"{len(sel)}")
        if not sel:
            return
        k0 = int(max(sel, key=lambda k: res["_r_v"][k]))
        axis = x.axis.currentIndex()
        i0 = int(res["_centres"][k0][axis] / (BOX / N))
        x.index.setValue((i0 + 5) % N)
        wait_for(qa, lambda: not x._busy and x._req["index"] == (i0 + 5) % N, 30)
        hit = next(((cx, cy, kk) for cx, cy, _rx, _ry, kk in c.circles or [] if kk == k0), None)
        check("an outline of the biggest void on a slice near its centre", hit is not None, f"{c.circles}")
        if hit is None:
            return
        r, col = int(hit[1]), int(hit[0])
        x._hover(r, col)
        check("1. hover over the outline: the readout names the void (R_v, R_eff, delta)",
              f"void {k0}: R_v" in view.readout.text() and "R_eff" in view.readout.text(), view.readout.text())
        x._click(r, col)
        info = x.void_info.text()
        check("1. click: the inspector shows R_v, R_eff, the central delta, the flags and the centre; Go to its "
              "centre enabled", f"void {k0}" in info and "R_eff" in info and "central δ" in info and "distinct" in info
              and "centre (" in info and x.void_goto.isEnabled(), info)
        x._void_goto()
        ok = wait_for(qa, lambda: not x._busy and x._req["index"] == i0, 30)
        check("1. Go to its centre: the slice through the void's centre cell", ok and x.index.value() == i0,
              f"{x.index.value()} vs {i0}")
        x.void_reff.setChecked(True)
        qa.processEvents()
        check("1. R_eff too: dashed outlines over the map, the same voids", bool(c.dashed)
              and c.dashed == x._void_circles("_r_eff") and {k for *_a, k in c.dashed} <= set(sel) | set(
                  np.nonzero(res["_shown"])[0]), f"{c.dashed}")
        c._redraw()                                         # draws both kinds without error
        # ---- 2. the saved PNG and its CSV
        png = TMP / "map.png"
        written = x.save_png(png)
        csvp = TMP / "map_voids.csv"
        rows = x.voids_shown()
        lines = csvp.read_text().splitlines() if csvp.is_file() else []
        ids = [int(ln.split(",")[0]) for ln in lines[2:]]
        check("2. Save image: the PNG and '<name>_voids.csv' with exactly the voids outlined (centre, R_v, R_eff, "
              "delta, flags)", png.is_file() and png.stat().st_size > 10000 and written == [str(png), str(csvp)]
              and ids == rows and k0 in ids and lines[1].startswith("void,x_mpc,y_mpc,z_mpc,r_v_mpc")
              and "r_eff_mpc,delta,resolved,distinct" in lines[1], f"{written} {ids} {rows}")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        orig_close = plt.close
        figs = []
        plt.close = lambda fig=None: figs.append(fig)       # keep the figure to count its circles
        try:
            x.save_png(TMP / "map2.png")
        finally:
            plt.close = orig_close
        fig = figs[0] if figs else None
        n_patch = sum(isinstance(p, Circle) for ax in (fig.axes if fig else []) for p in ax.patches)
        check("2. ... the PNG carries the outlines as on screen (two strokes per circle: R_v and R_eff)",
              fig is not None and n_patch == 2 * (len(c.circles) + len(c.dashed)), f"{n_patch} vs {len(c.circles)}+{len(c.dashed)}")
        if fig is not None:
            orig_close(fig)
        # ---- 2. the figure grid: two snapshots side by side, each with its own catalogue
        x.grid_set_layout(1, 2)
        x.grid_fill("redshifts")
        x.show_grid(True)
        ok = wait_for(qa, lambda: x._void_proc is None and len(x._void_cache) >= 2 and not x._grid_busy
                      and x._grid_pending is None and bool(c.circles), 240)
        keys = [k for k in x._void_cache if "error" not in x._void_cache[k]]
        clips = c.circle_clips or []
        check("2. the figure grid: each panel's own catalogue (found one after the other), its outlines clipped to "
              "its tile", ok and len(keys) == 2 and len(clips) == len(c.circles) + len(c.dashed or [])
              and len({cl[0] for cl in clips}) == 2, f"{len(x._void_cache)} {len(clips)}")
        fig, _n = x.grid_figure()
        n_patch = sum(isinstance(p, Circle) for ax in fig.axes for p in ax.patches)
        check("2. ... and the saved grid figure draws them (matplotlib circles in every panel)",
              n_patch > 0 and all(any(isinstance(p, Circle) for p in ax.patches) for ax in fig.axes[:2]), f"{n_patch}")
        x.show_grid(False)
        wait_for(qa, lambda: not x._busy and not x._grid_shown, 30)
        # ---- 3. the smoothing
        n_before = len(x._void_cache)
        x.void_sigma.setValue(6.0)
        ok = wait_for(qa, lambda: x._void_proc is None and len(x._void_cache) > n_before, 180)
        key = x._void_cache_key(x.current)
        r6 = x._void_cache.get(key, {})
        r0 = x._void_cache.get(x._void_cache_key(x.current, 0.0), {})
        check("3. a sigma in Mpc: its own cache key, the helper finds the voids at that sigma under its own namespace",
              ok and key[-1] == 6.0 and "error" not in r6 and abs(r6.get("sigma_mpc", 0) - 6.0) < 1e-6
              and r6.get("namespace") != r0.get("namespace") and r0.get("sigma_asked_mpc") == 0.0
              and any((TMP / "void cache").glob("*s6_*")), f"{key} {str(r6.get('error', ''))[:200]} "
              f"{r6.get('namespace')} {r0.get('namespace')}")
        st = x.state()
        check("3. the voids tick, R_eff and sigma are remembered", st.get("voids") is True and st.get("void_reff") is True
              and st.get("void_sigma") == 6.0, f"{st}")
        x.void_sigma.setValue(0.0)
        qa.processEvents()
        check("3. back to production: the cached catalogue, no new search", x._void_proc is None
              and x._void_cache_key(x.current) in x._void_cache)
        # ---- 4. the memory guard
        rs.dtfe_processes = lambda max_age=2.0: {"runs": [(4242, "PS-DTFE", 37.0)], "servers": []}
        asked.clear()
        x._voids_request(x._by_root[next(r for r in x._by_root if "snapdir_099" in r)])   # already cached: no question
        check("4. a cached catalogue is never asked about", not asked)
        os.environ["DTFE_FREE_GB"] = "2"                    # tight: the 21 MB build does not fit in 2 - 4 GB
        x.void_sigma.setValue(3.0)
        qa.processEvents()
        k3 = x._void_cache_key(x.current)
        check("4. a DTFE run going: before finding the voids the launcher asks (the run, its memory, what is free), "
              "and a 'no' starts nothing", len(asked) == 1 and "PS-DTFE pid 4242, 37.0 GB" in asked[0][1]
              and "free" in asked[0][1] and x._void_proc is None and "error" in x._void_cache.get(k3, {}),
              f"{asked} {x._void_cache.get(k3)}")
        asked.clear()
        os.environ["DTFE_FREE_GB"] = "30"
        check("4. a run going but the need fits in the free memory less the headroom: no question, started (said)",
              x.heavy_ok("Start the query server", "A query server", 12.0) and not asked)
        os.environ["DTFE_FREE_GB"] = "8"
        ok_heavy = x.heavy_ok("Start the query server", "A query server", 12.0)
        check("4. heavy_ok: asked with the need in GB, the answer obeyed", not ok_heavy and len(asked) == 1
              and "about 12 GB" in asked[0][1] and "--" not in asked[0][1], f"{asked}")
        x.confirm_heavy = lambda title, text: True
        check("4. ... a 'yes' lets it start", x.heavy_ok("Exact zoom", "it", None))
        rs.dtfe_processes = lambda max_age=2.0: {"runs": [], "servers": [(1, "PS-DTFE", 5.0)]}
        x.confirm_heavy = lambda title, text: asked.append(text) or False
        asked.clear()
        check("4. no run going (a server alone is not a run): never asks, never blocks",
              x.heavy_ok("Exact zoom", "it", 50.0) and not asked)
        line = x.memory_status()
        check("4. the memory status line: free GB and the server with its memory", "free" in line
              and "server pid 1, 5.0 GB" in line, line)
        check("launcher texts carry no '--'", all("--" not in t for t in (E.VOID_TIP, E.VOID_REFF_TIP, E.VOID_SIGMA_TIP,
                                                                           x.void_goto.toolTip())))
    finally:
        rs.dtfe_processes = real_procs
        os.environ.pop("DTFE_FREE_GB", None)
        x.shutdown() if hasattr(x, "shutdown") else None


def main():
    try:
        pure()
        gui()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nRESULT: {len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
