#!/usr/bin/env python3
"""Tests for the void-in-cloud criteria (2026-10-11): the velocity criterion (net infall in the 2-3 R_v shell, the
default), the smoothed Delta(<3 R_v) and the raw Delta(<2 R_v), and plot_void_environment.py's --criterion.

  * SYNTHETIC (always run): the shell velocity of a linear (Hubble-like) flow against its analytic volume-weighted
    mean, its sign and wrap rule; v_shell_norm in units of H(z) a R_v; three voids built so that the three criteria
    disagree in a known way (types, vic_<criterion> columns, agreement); the refusals (a criterion without its grid,
    an unknown criterion, a velocity without a redshift); classify_voids' default = the velocity on a stubbed
    snapshot; the figure script end to end on a stubbed pipeline (default untagged, '_density_raw' tagged, the npz
    keys, the figure text names the criterion and its shell, no option syntax in it).
  * DATA-BACKED (the T7; skipped without it): TNG100-3-Dark z = 0 in a fresh interpreter (DTFE_SIM sets the frame):
    the void-in-cloud fraction under every criterion strictly between 0 and 1, and the voids' infall fraction above
    that of random centres with the same radii, whose mean v_shell_norm is ~0 (a periodic flow has no net flux).

Usage:  python3 tests/py_vic_test.py            (exit 0 = pass)
"""

import atexit
import io as _io
import os
import shutil
import subprocess
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "python" / "plot"))
_FIGS_TMP = tempfile.mkdtemp(prefix="vic_figs_")      # before config is imported: never the real figure root
os.environ["DTFE_FIGURES_ROOT"] = _FIGS_TMP
os.environ["DTFE_NO_MIRROR"] = "1"
atexit.register(shutil.rmtree, _FIGS_TMP, True)

PASS, FAIL, SKIP = [], [], []
DATA_BACKED = False


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")
    except FileNotFoundError as e:
        if DATA_BACKED:
            SKIP.append(name)
            print(f"  SKIP  {name}  (data not on disk: {e})")
        else:
            FAIL.append(name)
            print(f"  FAIL  {name}: FileNotFoundError: {e}")
    except Exception as e:
        import traceback
        FAIL.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        print("".join("        " + ln for ln in traceback.format_exc().splitlines(True)[-6:]), end="")


# the volume-weighted mean of x over 2 <= x <= 3: int x^3 dx / int x^2 dx = (3/4)(3^4 - 2^4) / (3^3 - 2^3)
X_MEAN = 0.75 * (81 - 16) / (27 - 8)


def _offsets(n, c):
    """Per cell, the periodic offset (3, n, n, n) of its CENTRE from the centre of cell 'c' (both at i + 0.5)."""
    d = np.indices((n, n, n)).astype(float) - np.asarray(c, float).reshape(3, 1, 1, 1)
    return d - n * np.rint(d / n)


def _flow(n, voids):
    """(n, n, n, 3) velocity: per (centre cell, slope, reach in cells) a linear radial flow v = slope x offset
    inside 'reach' of the centre (zero beyond): v_r = slope r exactly, and trilinear interpolation of a linear field
    is exact wherever the 8 neighbours lie inside the reach."""
    v = np.zeros((n, n, n, 3))
    for c, slope, reach in voids:
        d = _offsets(n, c)
        inside = (d * d).sum(axis=0) < reach ** 2
        v[inside] += slope * np.moveaxis(d, 0, -1)[inside]
    return v


def _density(n, voids):
    """rho/rho_bar = 1 with, per (centre, R, [(r_in, r_out, rho), ...]), shells of given density (in cells, from the
    cell centres), the innermost listed first and overwritten by none of the later ones' ranges."""
    rho = np.ones((n, n, n))
    for c, _r, shells in voids:
        d = _offsets(n, c)
        dist = np.sqrt((d * d).sum(axis=0))
        for r_in, r_out, val in shells:
            rho[(dist >= r_in) & (dist < r_out)] = val
    return rho


# ---------------------------------------------------------------- synthetic: the shell velocity
def t_velocity_shell():
    from dtfelib import classify
    n, r = 64, 5.0
    vel = _flow(n, [((16, 16, 16), -2.0, 3.5 * r), ((48, 48, 48), +1.5, 3.5 * r)])
    centers = (np.array([[16, 16, 16], [48, 48, 48], [16, 48, 16]]) + 0.5) / n
    g = classify.velocity_shell_grid(vel, centers, [r, r, 11.0])       # the third: 3 x 11 > n / 2, dropped
    assert list(g["kept"]) == [True, True, False] and g["dropped"] == 1, g["kept"]
    assert g["bins"][0] == 2.0 and g["bins"][-1] == 3.0, g["bins"]
    want = np.array([-2.0, 1.5]) * r * X_MEAN                          # v_r = slope x r, volume weighted over x
    assert np.allclose(g["v_shell"][:2], want, rtol=2e-3), (g["v_shell"], want)
    assert np.isnan(g["v_shell"][2]) and np.isnan(g["v_r"][2]).all()
    # every shell of the linear flow is exact (trilinear of a linear field): v_r(x) = slope x R
    assert np.allclose(g["v_r"][0], -2.0 * r * g["bins"], rtol=1e-9), g["v_r"][0]


def t_velocity_norm():
    from dtfelib import classify
    from dtfelib.fields import get_cosmology_params
    n, r, cell, z = 64, 5.0, 1.5, 1.0
    c = get_cosmology_params(z)
    # v = -0.3 H(z) a r (km/s, r in Mpc): v_shell / (H a R_v) = -0.3 x X_MEAN whatever the cell size
    slope_cells = -0.3 * c["H_z"] * c["a"] * cell
    vel = _flow(n, [((16, 16, 16), slope_cells, 3.5 * r)])
    rho = np.ones((n, n, n))
    g = classify.measure_grid(rho, (np.array([[16, 16, 16]]) + 0.5) / n, [r * cell], cell, velocity=vel, redshift=z,
                              criterion="velocity")
    # 2e-3: the 11-shell trapezoid of x^3 against the exact integral
    assert abs(g["v_shell_norm"][0] / (-0.3 * X_MEAN) - 1) < 2e-3, g["v_shell_norm"]
    assert abs(g["v_shell_kms"][0] / (slope_cells * r * X_MEAN) - 1) < 2e-3, g["v_shell_kms"]
    # the same flow read at z = 0 with H0 a = 67.74: the km/s are the grid's, only the normalisation moves
    g0 = classify.measure_grid(rho, (np.array([[16, 16, 16]]) + 0.5) / n, [r * cell], cell, velocity=vel, redshift=0.0,
                               criterion="velocity")
    assert abs(g0["v_shell_norm"][0] * 67.74 - g["v_shell_norm"][0] * c["H_z"] * c["a"]) < 1e-9 * c["H_z"], g0
    assert g["type"][0] == "S" and g["shell"] == (2.0, 3.0) and g["criterion"] == "velocity"


def _three_voids(n=64):
    """Three voids, R = 4 cells, built so the criteria disagree in a known way:
      A (16,16,16): a compensating ridge 4-8 (raw Delta(<2R) > 0), an underdense 8-12 shell (smoothed Delta(<3R) < 0),
                    outflow 2-3 R                   -> raw S, smoothed R, velocity R
      B (48,48,48): bare (rho 1 beyond R), infall    -> raw R, smoothed R, velocity S
      C (16,48,48): a broad overdense 4-12 surrounding, infall -> S under all three
    The grids handed in as both the raw and the smoothed density (the test checks the criteria's logic, not the
    smoothing)."""
    r = 4.0
    rho = _density(n, [((16, 16, 16), r, [(0, 4, 0.2), (4, 8, 1.6), (8, 12, 0.5)]),
                       ((48, 48, 48), r, [(0, 4, 0.2)]),
                       ((16, 48, 48), r, [(0, 4, 0.2), (4, 12, 1.6)])])
    vel = _flow(n, [((16, 16, 16), +1.0, 3.5 * r), ((48, 48, 48), -1.0, 3.5 * r), ((16, 48, 48), -1.0, 3.5 * r)])
    coords = np.array([[16, 16, 16], [48, 48, 48], [16, 48, 48], [40, 10, 40]])
    cat = {"coords": coords, "well_resolved": np.array([True, True, True, True]),
           "r_v_cells": np.array([r, r, r, 12.0]),                       # D: 2 x 12 <= 32 < 3 x 12
           "cell_mpc": np.float64(1.0), "delta_values": np.array([-0.8, -0.7, -0.6, -0.5]),
           "bbks_params": np.array([[0.2, 0.05], [0.3, -0.1], [0.1, 0.0], [0.2, 0.0]]),
           "axis_ratios": np.array([[0.8, 0.6], [0.7, 0.5], [0.9, 0.8], [0.8, 0.7]]),
           "semi_axes": np.full((4, 3), 3.0, np.float32)}
    return rho, vel, cat


def t_criteria_catalog():
    from dtfelib import classify
    rho, vel, cat = _three_voids()
    kw = dict(velocity=vel, density_smoothed=rho, redshift=0.0, distinct=False)
    out = {c: classify.classify_catalog(cat, rho, 1.0, criterion=c, **kw) for c in classify.CRITERIA}
    v, s, r = out["velocity"], out["density_smoothed"], out["density_raw"]
    assert list(v["index"]) == [0, 1, 2] and v["dropped"] == 1 and v["n_sample"] == 4, v["index"]
    assert list(r["index"]) == [0, 1, 2, 3] and r["dropped"] == 0, r["index"]       # 2 R_v of D fits, 3 R_v not
    assert list(v["type"]) == ["R", "S", "S"], (v["type"], v["v_shell_kms"])
    assert list(s["type"]) == ["R", "R", "S"], (s["type"], s["delta_smoothed_enclosed"])
    assert list(r["type"][:3]) == ["S", "R", "S"], (r["type"], r["delta_enclosed"])
    assert v["shell"] == (2.0, 3.0) and s["shell"] == (0.0, 3.0) and r["shell"] == (0.0, 2.0)
    assert v["measured"] == classify.CRITERIA and v["criterion"] == "velocity"
    # every criterion's verdict is carried whichever decides; D has no velocity (NaN) and is never void-in-cloud by it
    assert list(r["vic_velocity"]) == [False, True, True, False] and np.isnan(r["v_shell_kms"][3])
    assert list(v["vic_density_raw"]) == [True, False, True] and list(v["vic"]) == list(v["vic_velocity"])
    assert np.allclose(v["v_shell_kms"], np.array([1.0, -1.0, -1.0]) * 4.0 * X_MEAN, rtol=2e-3), v["v_shell_kms"]
    ag = classify.agreement(r)                       # over the voids both criteria measured: D only by the densities
    assert ag[("velocity", "density_raw")][1] == 3 and abs(ag[("velocity", "density_raw")][0] - 1 / 3) < 1e-12, ag
    assert ag[("velocity", "density_smoothed")][1] == 3 and abs(ag[("velocity", "density_smoothed")][0] - 2 / 3) < 1e-12, ag
    assert ag[("density_smoothed", "density_raw")][1] == 3, ag      # D: its 3 R_v smoothed shell is dropped too
    # a NaN velocity cell on B's 2-3 R_v shells: B is dropped and counted under the velocity, not typed void-in-void
    bad = vel.copy()
    bad[48 + 10, 48, 48] = np.nan
    nb = classify.classify_catalog(cat, rho, 1.0, criterion="velocity", **dict(kw, velocity=bad))
    assert list(nb["index"]) == [0, 2] and nb["dropped"] == 2, (nb["index"], nb["dropped"])
    # the raw criterion alone (no other grid): only it is measured, and the old default call is unchanged
    old = classify.classify_catalog(cat, rho, 1.0, distinct=False)
    assert old["criterion"] == "density_raw" and old["measured"] == ("density_raw",)
    assert list(old["type"][:3]) == ["S", "R", "S"] and classify.agreement(old) == {}


def t_refusals():
    from dtfelib import classify
    rho, vel, cat = _three_voids()
    for kw, crit, word in (({}, "velocity", "velocity grid"), ({}, "density_smoothed", "smoothed density"),
                           ({"velocity": vel}, "velocity", "redshift"), ({}, "density", "criterion must be")):
        try:
            classify.classify_catalog(cat, rho, 1.0, criterion=crit, **kw)
        except ValueError as e:
            assert word in str(e), (crit, str(e))
        else:
            raise AssertionError(f"criterion {crit!r} with {sorted(kw)} was not refused")
    assert classify.DEFAULT_CRITERION == "velocity" and classify.CRITERIA == ("velocity", "density_smoothed", "density_raw")


class _StubFS:
    def __init__(self, vel, method="ps", prefix="ps_output.", snapdir=Path(".")):
        self.vel, self.method, self.prefix, self.snapdir = vel, method, prefix, snapdir
        self.loaded = []

    def has(self, name):
        return name == "velocity" and self.vel is not None

    def load(self, name, mode="auto"):
        assert name == "velocity" and self.vel is not None, name
        self.loaded.append((name, mode))
        return self.vel


class _StubProd:
    def __init__(self, rho, vel, cat, z=0.0, snapdir=Path(".")):
        self.fs, self.density_raw, self._cat, self.redshift = _StubFS(vel, snapdir=snapdir), rho, cat, z
        self.delta_smoothed = rho - 1.0

    def voids(self):
        return self._cat

    def release(self):
        pass


def t_classify_voids_default():
    from dtfelib import classify, catalog
    rho, vel, cat = _three_voids()
    saved = (catalog.frame_of, catalog.assert_frame)
    try:
        catalog.frame_of = lambda p: (1.0, 64.0, 64, "header")
        catalog.assert_frame = lambda c, cell: None
        prod = _StubProd(rho, vel, cat)
        e = classify.classify_voids(prod, distinct=False)
        assert e["criterion"] == "velocity" and list(e["type"]) == ["R", "S", "S"], (e["criterion"], e["type"])
        assert prod.fs.loaded == [("velocity", "ram")] and e["redshift"] == 0.0 and e["n_grid"] == 64
        assert list(e["vic_density_smoothed"]) == [False, False, True]     # 1 + delta_smoothed went in
        try:
            classify.classify_voids(_StubProd(rho, None, cat))
        except ValueError as err:
            assert "no velocity grid" in str(err), err
        else:
            raise AssertionError("the velocity criterion ran without a velocity grid")
        d = classify.classify_voids(_StubProd(rho, None, cat), criterion="density_raw", distinct=False)
        assert d["measured"] == ("density_smoothed", "density_raw") and np.isnan(d["v_shell_norm"]).all()
    finally:
        catalog.frame_of, catalog.assert_frame = saved


# ---------------------------------------------------------------- synthetic: the figure script
def t_environment_script():
    import config
    import plot_void_environment as pev
    from dtfelib import classify, catalog, cli
    rho, vel, cat = _three_voids()
    root = Path(tempfile.mkdtemp(prefix="vic_run_"))
    saved = (pev.pipeline, pev.catalog, pev.style.note_source, pev.style.set_suptitle, catalog.frame_of,
             catalog.assert_frame, cli.DATA_ROOT)
    titles = []
    try:
        data = root / "data" / "SYN"
        for s in (50, 99):
            (data / f"snapdir_{s:03d}").mkdir(parents=True)
        real_pl = pev.pipeline
        pev.pipeline = types.SimpleNamespace(
            products=lambda s, *a, **kw: _StubProd(rho, vel, cat, z=0.0 if int(s) == 99 else 1.0,
                                                   snapdir=data / f"snapdir_{int(s):03d}"),
            add_smoothing_mpc_arg=real_pl.add_smoothing_mpc_arg, apply_smoothing_args=real_pl.apply_smoothing_args,
            smoothing_tag=real_pl.smoothing_tag)
        frame = lambda p: (1.0, 64.0, 64, "header")
        catalog.frame_of, catalog.assert_frame = frame, (lambda c, cell: None)
        pev.catalog = types.SimpleNamespace(frame_of=frame, r_v_mpc=catalog.r_v_mpc, R_V_DEFINITIONS=catalog.R_V_DEFINITIONS)
        pev.style.note_source = lambda fs: None
        real_title = saved[3]
        pev.style.set_suptitle = lambda fig, text, *a, **kw: (titles.append(text), real_title(fig, text, *a, **kw))[1]
        tdir = root / "tracking"
        tdir.mkdir()
        for k, c in enumerate(((16, 16, 16), (48, 48, 48))):            # A: R by the velocity, B: S
            np.savez(tdir / f"void{k}_track.npz", snap=np.array([99, 50]), redshift=np.array([0.0, 1.0]),
                     r_v=np.array([4.0, 3.0]), r_eff=np.array([np.nan, np.nan]), e=np.array([0.2, 0.1]),
                     p=np.array([0.0, 0.05]), delta_min=np.array([-0.8, -0.5]),
                     center0_frac=(np.asarray(c) + 0.5) / 64, catalog_index0=np.int64(k))
        args = ["--sim", "SYN", "--data-root", str(root / "data"), "--tracking-dir", str(tdir), "--keep-overlaps"]
        buf = _io.StringIO()
        with redirect_stdout(buf):
            pev.main(args)
        text = buf.getvalue()
        out = Path(config.figures_path("void_environment")) / "SYN"
        names = sorted(q.name for q in out.iterdir())
        # the default criterion is TAGGED (2026-10-11): the untagged names keep density_raw's meaning
        want = ["environment_evolution_overlaps_velocity.png", "environment_stats_overlaps_velocity.npz",
                "environment_tracking_overlaps_velocity.png", "environment_z0.00_overlaps_velocity.png",
                "environment_z1.00_overlaps_velocity.png"]
        assert names == want, (names, text[-600:])
        assert "f_vic = 0.667" in text and "0: R" in text and "1: S" in text, text[-900:]
        assert "agree velocity/density_smoothed 0.667" in text, text[-900:]
        with np.load(out / "environment_stats_overlaps_velocity.npz") as z:
            assert str(z["criterion"]) == "velocity" and list(z["criterion_shell"]) == [2.0, 3.0]
            assert list(z["n"]) == [3, 3] and np.allclose(z["f_vic"], 2 / 3), z["f_vic"]
            assert np.allclose(z["f_vic_density_smoothed"], 1 / 3) and np.allclose(z["f_vic_density_raw"], 2 / 3)
            assert np.allclose(z["agree_velocity__density_raw"], 1 / 3) and list(z["agree_velocity__density_raw_n"]) == [3, 3]
            assert list(z["snap099_vic_velocity"]) == [False, True, True] and z["snap099_v_shell_norm"].shape == (3,)
            assert list(z["tracked_type"]) == ["R", "S"] and z["tracked_measure"][0] > 0 > z["tracked_measure"][1]
        assert titles and all("--" not in t for t in titles), titles
        assert all("v_r(2-3 R_v) < 0" in t for t in titles), titles
        # the raw criterion: the UNTAGGED files (what they held before the velocity became the default), its own
        # fraction, D (3 R_v beyond the wrap rule) now kept
        titles.clear()
        with redirect_stdout(_io.StringIO()):
            pev.main(args + ["--criterion", "density_raw"])
        names = sorted(q.name for q in out.iterdir() if "_velocity" not in q.name)
        assert names == ["environment_evolution_overlaps.png", "environment_stats_overlaps.npz",
                         "environment_tracking_overlaps.png", "environment_z0.00_overlaps.png",
                         "environment_z1.00_overlaps.png"], names
        with np.load(out / "environment_stats_overlaps.npz") as z:
            assert str(z["criterion"]) == "density_raw" and list(z["n"]) == [4, 4]
            assert list(z["tracked_type"]) == ["S", "R"], z["tracked_type"]
        assert all("raw Delta(<2 R_v) > 0" in t and "--" not in t for t in titles), titles
        assert pev.name_tag({99: "ps"}, {99: "ps_output."}, False, 2.0, "crossing", "density_smoothed") == "_density_smoothed"
        assert pev.name_tag({99: "ps"}, {99: "ps_output."}, False, 2.0) == "_velocity"          # the default
        assert pev.name_tag({99: "ps"}, {99: "ps_output."}, False, 2.0, "crossing", "density_raw") == ""
        assert all("--" not in pev.criterion_text(c) for c in classify.CRITERIA)
    finally:
        (pev.pipeline, pev.catalog, pev.style.note_source, pev.style.set_suptitle, catalog.frame_of,
         catalog.assert_frame, _root) = saved
        cli.use_data_root(_root)                    # main()'s --data-root is process-wide
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- data-backed: TNG100-3-Dark
def t_tng100_criteria():
    from dtfelib import cli
    sd = cli.sim_dir("TNG100-3-Dark", os.environ.get("DTFE_DATA_ROOT", str(cli._T7_DATA))) / "snapdir_099"
    if not any(sd.glob("ps_output.*vel")):
        raise FileNotFoundError(f"no PS-DTFE velocity in {sd}")
    code = ("import sys; sys.path.insert(0, 'python'); import numpy as np\n"
            "from dtfelib import pipeline, classify, catalog\n"
            "p = pipeline.products(99, sim='TNG100-3-Dark', method='ps')\n"
            "e = classify.classify_voids(p)\n"
            "cell = catalog.frame_of(p)[0]\n"
            "r = np.repeat(e['r_v_mpc'], 5); c = np.random.default_rng(1).random((r.size, 3))\n"
            "g = classify.measure_grid(p.density_raw, c, r, cell, velocity=p.fs.load('velocity', mode='ram'),\n"
            "                          redshift=p.redshift, criterion='velocity')\n"
            "k = g['kept']\n"
            "print(len(e['vic']), *[float(e['vic_' + c].mean()) for c in classify.CRITERIA],\n"
            "      float(g['vic_velocity'][k].mean()), float(np.nanmean(g['v_shell_norm'][k])))\n")
    env = dict(os.environ, DTFE_SIM="TNG100-3-Dark")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-800:]
    n, fv, fs, fr, f_null, v_null = r.stdout.split()[-6:]
    print(f"    TNG100-3-Dark z = 0: {n} voids, void-in-cloud velocity {float(fv):.3f}, smoothed {float(fs):.3f}, "
          f"raw {float(fr):.3f}; random centres: infall {float(f_null):.3f}, mean v_r/(H a R_v) {float(v_null):+.4f}")
    assert int(n) > 20 and all(0.0 < float(f) < 1.0 for f in (fv, fs, fr)), r.stdout
    assert float(fv) > float(f_null) + 0.15 and abs(float(v_null)) < 0.03, r.stdout


def main():
    global DATA_BACKED
    import matplotlib
    matplotlib.use("Agg")
    print("synthetic")
    check("velocity_shell_grid: linear flow -> slope R x (3/4)(3^4-2^4)/(3^3-2^3), exact per shell, the 3 R_v wrap rule", t_velocity_shell)
    check("v_shell_norm in units of H(z) a R_v (z = 1, cell 1.5 Mpc), the velocity criterion's shell (2, 3)", t_velocity_norm)
    check("the three criteria on three built voids: types, vic_<c> columns, agreement over mutually measured voids", t_criteria_catalog)
    check("refusals: a criterion without its grid, a velocity without a redshift, an unknown criterion", t_refusals)
    check("classify_voids: velocity by default (FieldSet.load into RAM), no velocity grid refused, density_raw without it", t_classify_voids_default)
    check("plot_void_environment.py --criterion end to end (stubbed pipeline): tags, npz keys, figure text, tracking", t_environment_script)
    DATA_BACKED = True
    print("data-backed")
    check("TNG100-3-Dark z = 0: the three fractions, the voids' infall against random centres (mean flux ~0)", t_tng100_criteria)
    print(f"\nRESULT: {len(PASS)} passed, {len(FAIL)} failed" + (f", {len(SKIP)} skipped" if SKIP else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
