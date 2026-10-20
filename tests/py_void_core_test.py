#!/usr/bin/env python3
"""Tests for the void-catalogue core (dtfelib.pipeline / catalog, config):

  * the three R_v definitions (catalog.R_V_DEFINITIONS: crossing, delta08, ridge) from one shell matrix
    (pipeline.void_radii_cells) on an analytic compensated void, the crossing bit for bit void_radius_cells';
  * catalog.r_v_cells / r_v_mpc per definition (NaN without the key), the export's radius columns and provenance;
  * the R_v settings in the cache identity: a catalogue without them, or with other ones, is re-measured once
    (the smoothed field read, the Hessian not), then served silently;
  * DTFE_CACHE_DIR and DTFE_VOID_GATE (config at import; the gate applied to the served dict, never to the cache,
    and tagged through pipeline.smoothing_tag).
  * config.CROSSBOX_SMOOTHING_MPC, the one cross-box smoothing, and the R_v search cap of 24 sigma.
  * the R_v search's half-box margin with the reach recorded; the cache folder setting, 'analyze.py cache
    --move' and the setup's offer to keep the caches out of iCloud Drive.

SYNTHETIC tests always run (CI: no T7); the DATA-BACKED one (TNG100-3-Dark z = 0 on the T7) is skipped when the
snapshot is not on disk.

Usage:  python3 tests/py_void_core_test.py            (exit 0 = pass)
"""

import atexit
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from _harness import REPO, Skip, check as _check, setup, summary

setup("void_core_figs_", paths=("python",), no_mirror=False, mpl_agg=False)
os.environ.setdefault("DTFE_SIM", "TNG50-4-Dark")    # the box these synthetic tests stand for; a data-tier run sets its own
os.environ.pop("DTFE_VOID_GATE", None)          # this process tests the default gate; the subprocesses set their own
_PY = REPO / "python"


def check(name, fn):
    """A crash is a failure, named, without its traceback."""
    _check(name, fn, crash_traceback=False)


# ------------------------------------------------------------------ fixtures
N = 96


def _radius_grid(c):
    idx = np.indices((N, N, N)).astype(np.float64)
    return np.sqrt(sum((((idx[k] + 0.5) - (c[k] + 0.5) + N / 2) % N - N / 2) ** 2 for k in range(3)))


def _profile(r, depth, r0=8.0, s=6.0):
    """A compensated void: delta = -depth (1 - r^2 / r0^2) exp(-r^2 / (2 s^2)): 0 exactly at r0, a ridge beyond."""
    return -depth * (1.0 - (r / r0) ** 2) * np.exp(-0.5 * (r / s) ** 2)


def _expected(depth, level=-0.8):
    """crossing, delta08 and ridge of _profile from a dense 1-D integration (the truth the grid estimate approaches)."""
    r = np.linspace(0.0, 40.0, 400001)
    f = _profile(r, depth)
    enc = np.empty_like(r)
    enc[0] = f[0]
    w = f * r * r
    enc[1:] = 3.0 * np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(r)) / r[1:] ** 3
    up = np.nonzero(enc >= level)[0]
    d08 = r[up[0]] if f[0] < level and up.size else np.nan
    return 8.0, d08, r[np.argmax(f)]


# ------------------------------------------------------------------ tests
def t_radii_definitions():
    from dtfelib import pipeline
    for c in ((40, 50, 30), (1, 94, 0)):                       # in the box and across the periodic edges
        r = _radius_grid(c)
        for depth in (0.95, 0.5):                              # deep: delta08 defined; shallow: the minimum is above -0.8
            field = _profile(r, depth).astype(np.float32)
            got = pipeline.void_radii_cells(field, np.array([c]), 1.0)
            cr, d08, ridge = _expected(depth)
            assert abs(got["r_v_cells"][0] - cr) < 0.05, (c, depth, got["r_v_cells"], cr)
            if np.isnan(d08):
                assert np.isnan(got["r_v_delta08_cells"][0]), (c, depth, got["r_v_delta08_cells"])
            else:
                # 0.2 cells: the trilinear shell means sit ~0.013 above the curved profile near the minimum, where
                # Delta(<r) is flat (the integral itself is exact to 0.003 cells on the true profile's samples)
                assert abs(got["r_v_delta08_cells"][0] - d08) < 0.2, (c, depth, got["r_v_delta08_cells"], d08)
            assert abs(got["r_v_ridge_cells"][0] - ridge) < 0.15, (c, depth, got["r_v_ridge_cells"], ridge)
            # the crossing IS void_radius_cells (one function behind both)
            assert np.array_equal(pipeline.void_radius_cells(field, np.array([c]), 1.0), got["r_v_cells"], equal_nan=True)
    # an overdense 'minimum': all three NaN; a profile still rising at the edge of the search range: no ridge
    c = (40, 50, 30)
    r = _radius_grid(c)
    got = pipeline.void_radii_cells(_profile(r, 0.95).astype(np.float32) + 2.0, np.array([c]), 1.0)
    assert all(np.isnan(v[0]) for v in got.values()), got
    bowl = (0.5 * ((r / 6.0) ** 2 - 1.0)).astype(np.float32)   # rises for ever: crosses 0 at 6, no maximum inside reach
    got = pipeline.void_radii_cells(bowl, np.array([c]), 1.0)
    assert abs(got["r_v_cells"][0] - 6.0) < 0.05 and np.isnan(got["r_v_ridge_cells"][0]), got
    assert all(v.shape == (0,) for v in pipeline.void_radii_cells(bowl, np.zeros((0, 3), int), 1.0).values())
    # the enclosed contrast of a constant profile is that constant at every radius (exact piecewise-linear integral)
    rr = np.array([0.0, 0.2, 0.4, 0.6])
    assert np.allclose(pipeline._enclosed_contrast(rr, np.full((1, 4), -0.7)), -0.7, rtol=0, atol=1e-12)


def t_r_v_accessors():
    from dtfelib import catalog
    assert catalog.R_V_DEFINITIONS == ("crossing", "delta08", "ridge")
    cat = {"coords": np.zeros((3, 3), int), "cell_mpc": np.float64(0.5),
           "r_v_cells": np.array([2.0, np.nan, 4.0]), "r_v_ridge_cells": np.array([5.0, 6.0, np.nan])}
    assert np.allclose(catalog.r_v_mpc(cat), [1.0, np.nan, 2.0], equal_nan=True)            # no definition: crossing
    assert np.allclose(catalog.r_v_mpc(cat, "crossing"), catalog.r_v_mpc(cat), equal_nan=True)
    assert np.allclose(catalog.r_v_mpc(cat, "ridge"), [2.5, 3.0, np.nan], equal_nan=True)
    d08 = catalog.r_v_mpc(cat, "delta08")                       # the key absent: NaN of the catalogue's length
    assert d08.shape == (3,) and np.isnan(d08).all() and d08.dtype == np.float64, d08
    assert np.allclose(catalog.r_v_cells(cat, "ridge"), [5.0, 6.0, np.nan], equal_nan=True)
    try:
        catalog.r_v_mpc(cat, "persistence")
        raise AssertionError("an unknown definition must raise")
    except ValueError as e:
        assert "crossing" in str(e), e
    empty = {"coords": np.zeros((0, 3), int)}
    assert catalog.r_v_mpc(empty, "ridge").shape == (0,)


class _FieldFixture:
    """Two compensated voids on a 48^3 grid (one deep, one shallow), smoothed: a field with real minima."""

    def __init__(self):
        from dtfelib import fields as dtfe, pipeline
        n = self.n = 48
        idx = np.indices((n, n, n)).astype(np.float64)
        delta = np.zeros((n, n, n))
        for c, d in (((12, 12, 12), 0.95), ((34, 30, 26), 0.6)):
            r = np.sqrt(sum((((idx[k] - c[k]) + n / 2) % n - n / 2) ** 2 for k in range(3)))
            delta += _profile(r, d, r0=6.0, s=4.5)
        self.delta_s = dtfe.smooth_field(delta.astype(np.float32), sigma=1.0)
        self.hess = pipeline.hessian_from_field(self.delta_s)


def _products_class(fx, reads, grid_file):
    import config
    from dtfelib import pipeline

    class _P(pipeline.SnapshotProducts):
        @property
        def delta_smoothed(self):
            reads.append("field")
            return fx.delta_s

        @property
        def hessian(self):
            reads.append("hessian")
            return fx.hess

        @property
        def fs(self):
            return types.SimpleNamespace(method="dtfe", grid_n=fx.n, prefix="output.",
                                         meta=types.SimpleNamespace(box_mpc=config.CELL_SIZE * fx.n, redshift=0.0))

        def _grid_path(self):                   # a real file, so the cache carries a grid stamp and can be re-measured
            return grid_file
    return _P


def t_r_v_cache_identity():
    import config
    from dtfelib import pipeline
    fx = _FieldFixture()
    root = Path(tempfile.mkdtemp(prefix="void_core_cache_"))
    old_cache, old_sigma, old_max = config.CACHE_DIR, config.SMOOTHING_SIGMA_CELLS, pipeline.R_V_MAX_SIGMA
    grid = root / "density.bin"
    grid.write_bytes(b"x" * 64)
    reads = []
    P = _products_class(fx, reads, grid)
    saves = []
    old_save = pipeline._save_npz
    try:
        config.CACHE_DIR = root / "cache"
        config.SMOOTHING_SIGMA_CELLS = 1.0
        pipeline._save_npz = lambda path, **a: (saves.append(Path(path).name), old_save(path, **a))[1]

        def load():
            buf, mark, rmark = io.StringIO(), len(saves), len(reads)
            with redirect_stdout(buf):
                cat = P("099", 0.0, sim="SYN").voids()
            return cat, buf.getvalue(), len(saves) - mark, reads[rmark:]
        cat, said, wrote, rd = load()                            # built: radii and settings in the catalogue
        assert wrote == 1 and pipeline.catalog_r_v_ok(cat, n=fx.n), (wrote, said)   # the reach is the 48^3 grid's
        assert np.array_equal(cat["r_v_settings"], pipeline.r_v_record(1.0, fx.n))   # the five settings + the reach
        for k in ("r_v_cells", "r_v_delta08_cells", "r_v_ridge_cells"):
            assert cat[k].shape == (len(cat["coords"]),), k
        assert np.isfinite(cat["r_v_cells"]).any() and np.isfinite(cat["r_v_ridge_cells"]).any()
        _, said, wrote, rd = load()
        assert said == "" and wrote == 0 and rd == [], (said, wrote, rd)  # fresh: read silently, nothing re-measured
        # a legacy catalogue: no settings, no delta08 / ridge -> re-measured ONCE, the Hessian not read
        path = P("099", 0.0, sim="SYN")._cpath("voids")
        with np.load(path) as z:
            old = {k: z[k] for k in z.files if k not in ("r_v_settings", "r_v_delta08_cells", "r_v_ridge_cells")}
        np.savez_compressed(path, **old)
        got, said, wrote, rd = load()
        assert "no record of its R_v settings" in said and said.count("\n") == 1 and wrote == 1, (said, wrote)
        assert rd == ["field"], rd                               # the radii from the smoothed field; no Hessian
        for k in ("r_v_cells", "r_v_delta08_cells", "r_v_ridge_cells", "well_resolved", "coords"):
            assert np.array_equal(got[k], cat[k], equal_nan=got[k].dtype.kind == "f"), k
        _, said, wrote, rd = load()
        assert said == "" and wrote == 0 and rd == [], (said, wrote)
        # another reach: the record is not today's -> re-measured, the settings named
        pipeline.R_V_MAX_SIGMA = 7.0
        got, said, wrote, rd = load()
        assert "not today's" in said and wrote == 1 and rd == ["field"], (said, wrote, rd)
        assert float(got["r_v_settings"][1]) == 7.0
        pipeline.R_V_MAX_SIGMA = old_max
        _, said, wrote, _ = load()
        assert "not today's" in said and wrote == 1, said           # and back
    finally:
        config.CACHE_DIR, config.SMOOTHING_SIGMA_CELLS, pipeline.R_V_MAX_SIGMA = old_cache, old_sigma, old_max
        pipeline._save_npz = old_save
        shutil.rmtree(root, ignore_errors=True)


def t_void_gate_and_export():
    import config
    from dtfelib import catalog, pipeline
    fx = _FieldFixture()
    root = Path(tempfile.mkdtemp(prefix="void_core_gate_"))
    old = (config.CACHE_DIR, config.SMOOTHING_SIGMA_CELLS, config.VOID_GATE, config.ELLIPSOID_CUT_RULES)
    grid = root / "density.bin"
    grid.write_bytes(b"x" * 64)
    P = _products_class(fx, [], grid)
    try:
        config.CACHE_DIR = root / "cache"
        config.SMOOTHING_SIGMA_CELLS = 1.0
        # a max axis tight enough that the gate binds on this fixture (the other cuts as production)
        config.ELLIPSOID_CUT_RULES = dict(old[3], max_axis_sigma=3.0)
        with redirect_stdout(io.StringIO()):
            cat = P("099", 0.0, sim="SYN").voids()
        path = P("099", 0.0, sim="SYN")._cpath("voids")
        before = path.read_bytes()
        assert pipeline.void_gate() == "ellipsoid" and "gate" not in pipeline.smoothing_tag(), pipeline.smoothing_tag()
        config.VOID_GATE = "none"
        with redirect_stdout(io.StringIO()):
            ung = P("099", 0.0, sim="SYN").voids()
        res, res_n = np.asarray(cat["well_resolved"]), np.asarray(ung["well_resolved"])
        assert (res_n | ~res).all() and res_n.sum() > res.sum(), (res, res_n)      # a superset, strictly
        assert str(ung["void_gate"]) == "none" and np.isinf(ung["cuts_mpc"][1])
        assert path.read_bytes() == before, "the gate view was written into the shared cache"
        assert pipeline.smoothing_tag().endswith("gatenone") and "--" not in pipeline.smoothing_tag()
        config.SMOOTHING_SIGMA_CELLS = old[1]                      # the production smoothing: the gate tag alone
        assert pipeline.smoothing_tag() == "gatenone", pipeline.smoothing_tag()
        config.SMOOTHING_SIGMA_CELLS = 1.0
        # the export: the delta08 / ridge radius columns (cells x cell) and the R_v settings + gate in the provenance
        out = catalog.export_voids(P("099", 0.0, sim="SYN"), root / "out")
        for table, prov in (catalog.read_hdf5(out[1]), catalog.read_csv(out[0])):
            for col, key in (("r_v_mpc", "r_v_cells"), ("r_v_delta08_mpc", "r_v_delta08_cells"),
                             ("r_v_ridge_mpc", "r_v_ridge_cells")):
                assert np.allclose(table[col], np.asarray(ung[key]) * config.CELL_SIZE, rtol=1e-8, equal_nan=True), col
            assert str(prov["void_gate"]) == "none" and np.isinf(float(prov["cut_max_axis_mpc"])), prov
            assert int(prov["r_v_algorithm"]) == pipeline.R_V_ALGORITHM and int(prov["r_v_dirs"]) == pipeline.R_V_DIRS
            assert float(prov["r_v_step_sigma"]) == pipeline.R_V_STEP_SIGMA and float(prov["r_v_max_sigma"]) == pipeline.R_V_MAX_SIGMA
            assert float(prov["r_v_delta08"]) == pipeline.R_V_DELTA08
            assert int(np.asarray(table["distinct"]).sum()) >= 1
        config.VOID_GATE = "ellipsoid"
        with redirect_stdout(io.StringIO()):
            back = P("099", 0.0, sim="SYN").voids()
        assert np.array_equal(back["well_resolved"], cat["well_resolved"]) and "void_gate" not in back
        config.VOID_GATE = "everything"
        try:
            pipeline.void_gate()
            raise AssertionError("an unknown gate must raise")
        except ValueError as e:
            assert "DTFE_VOID_GATE" in str(e), e
    finally:
        config.CACHE_DIR, config.SMOOTHING_SIGMA_CELLS, config.VOID_GATE, config.ELLIPSOID_CUT_RULES = old
        shutil.rmtree(root, ignore_errors=True)


_NO_SETTING = tempfile.mkdtemp(prefix="void_core_xdg_")      # an empty config home: no cache setting
atexit.register(shutil.rmtree, _NO_SETTING, True)


def _config_in_child(env_extra, expr):
    env = {k: v for k, v in os.environ.items() if k not in ("DTFE_CACHE_DIR", "DTFE_VOID_GATE")}
    env["XDG_CONFIG_HOME"] = _NO_SETTING        # never this machine's own ~/.config/dtfe/cache_dir: the default tested
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", f"import config; print({expr})"], cwd=str(_PY), env=env,
                          capture_output=True, text=True, timeout=120)


def t_env_overrides():
    """DTFE_CACHE_DIR / DTFE_VOID_GATE are read by config at import: a fresh interpreter each."""
    r = _config_in_child({}, "config.CACHE_DIR")
    assert r.returncode == 0 and Path(r.stdout.strip()) == _PY / "cache", (r.stdout, r.stderr[-300:])
    tmp = tempfile.mkdtemp(prefix="void_core_cdir_")
    try:
        r = _config_in_child({"DTFE_CACHE_DIR": tmp}, "config.CACHE_DIR")
        assert r.returncode == 0 and Path(r.stdout.strip()) == Path(tmp), (r.stdout, r.stderr[-300:])
        r = _config_in_child({"DTFE_CACHE_DIR": "~/void_core_x"}, "config.CACHE_DIR")
        assert r.returncode == 0 and Path(r.stdout.strip()) == Path.home() / "void_core_x", r.stdout
        # the products follow it (cache_dir() reads config.CACHE_DIR at call time)
        r = subprocess.run([sys.executable, "-c", "from dtfelib import pipeline; print(pipeline.cache_dir())"],
                           cwd=str(_PY), env=dict(os.environ, DTFE_CACHE_DIR=tmp), capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and Path(r.stdout.strip()).parent == Path(tmp), (r.stdout, r.stderr[-300:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    r = _config_in_child({}, "config.VOID_GATE")
    assert r.returncode == 0 and r.stdout.strip() == "ellipsoid", r.stdout
    r = _config_in_child({"DTFE_VOID_GATE": " None "}, "config.VOID_GATE")
    assert r.returncode == 0 and r.stdout.strip() == "none", (r.stdout, r.stderr[-300:])
    r = _config_in_child({"DTFE_VOID_GATE": "bogus"}, "config.VOID_GATE")
    assert r.returncode != 0 and "DTFE_VOID_GATE" in r.stderr, (r.returncode, r.stderr[-300:])


def t_real_tng100_z0():
    """TNG100-3-Dark z = 0 (T7): the served catalogue carries the current R_v settings and all three radii; the crossing
    median of the distinct voids is 7.42 Mpc (6.94 under a 15-sigma cap). A fresh
    interpreter (DTFE_SIM decides config's frame); the shared cache is read, or re-measured once if stale."""
    from dtfelib.cli import sim_dir
    if not (sim_dir("TNG100-3-Dark") / "snapdir_099").is_dir():
        raise Skip("TNG100-3-Dark snapdir_099 not on disk")
    code = ("import numpy as np, config; from dtfelib import pipeline, catalog\n"
            "cat = pipeline.products('099', sim='TNG100-3-Dark').voids()\n"
            "assert pipeline.catalog_r_v_ok(cat)\n"
            "res = np.asarray(cat['well_resolved']).astype(bool)\n"
            "d = catalog.distinct_mask(cat, res & np.isfinite(catalog.r_v_mpc(cat)), config.FIELD_RESOLUTION)\n"
            "print(int(d.sum()), *[float(np.nanmedian(catalog.r_v_mpc(cat, k)[d])) for k in catalog.R_V_DEFINITIONS])\n")
    env = dict(os.environ, DTFE_SIM="TNG100-3-Dark", DTFE_NO_MIRROR="1")
    env.pop("DTFE_VOID_GATE", None)
    env.pop("DTFE_SIGMA_MPC", None)             # the production namespace, whatever this shell exports
    env.pop("DTFE_CACHE_DIR", None)
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_PY), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-600:]
    nd, cr, d08, ridge = r.stdout.strip().splitlines()[-1].split()
    # 83 / 7.419 Mpc at the 24-sigma cap (77 / 6.938 at 15 sigma, where 88 minima have no crossing)
    assert int(nd) == 83 and abs(float(cr) - 7.419) < 0.01, r.stdout
    assert float(d08) < float(cr) < float(ridge), r.stdout       # inner sphere < first crossing < compensation ridge


def t_crossbox_constant():
    """config.CROSSBOX_SMOOTHING_MPC = 2.1625 is the ONE cross-box smoothing: plot_void_crossbox's
    default and analyze.py's cache inventory import it (an assignment that names it, not an equal literal), and
    scripts/thesis_runs.sh's literal (a recipe, kept on purpose) equals it."""
    import re
    import config
    assert config.CROSSBOX_SMOOTHING_MPC == 2.1625, config.CROSSBOX_SMOOTHING_MPC
    sys.path.insert(0, str(_PY / "plot"))
    try:
        import plot_void_crossbox
        import analyze
    finally:
        sys.path.remove(str(_PY / "plot"))
    assert plot_void_crossbox.DEFAULT_SIGMA_MPC == config.CROSSBOX_SMOOTHING_MPC, plot_void_crossbox.DEFAULT_SIGMA_MPC
    assert analyze.CACHE_SMOOTHING_MPC == (config.CROSSBOX_SMOOTHING_MPC,), analyze.CACHE_SMOOTHING_MPC
    for f, name in (("plot/plot_void_crossbox.py", "DEFAULT_SIGMA_MPC"), ("analyze.py", "CACHE_SMOOTHING_MPC")):
        src = (_PY / f).read_text()
        assert re.search(rf"^{name} = .*config\.CROSSBOX_SMOOTHING_MPC", src, re.M), f"{f}: {name} not from config"
    sh = (_PY.parent / "scripts" / "thesis_runs.sh").read_text()
    m = re.search(r"^CROSSBOX_SMOOTH_MPC=(\S+)$", sh, re.M)
    assert m and float(m.group(1)) == config.CROSSBOX_SMOOTHING_MPC, m
    assert "--smooth-mpc 2.1625" not in sh, "thesis_runs.sh: a bare literal left beside CROSSBOX_SMOOTH_MPC"


def t_r_v_cap():
    """The R_v search cap: 24 sigma, not 15 (17-25% of the underdense minima have no crossing within
    15 sigma), never above half the production box minus a sigma. A void negative out to 20 sigma (40 cells at sigma
    = 2, in a 128^3 box whose L/2 = 64 cells is not what binds): its crossing is measured, ~40 cells; under a
    15-sigma cap (30 cells) it would be NaN."""
    import config
    from dtfelib import pipeline
    n, sig = 128, 2.0
    assert 20.0 < pipeline.R_V_MAX_SIGMA
    sig0 = config.PRODUCTION_SMOOTHING[0]                                         # not DTFE_SIGMA_MPC's
    assert pipeline.R_V_MAX_SIGMA * sig0 <= 0.5 * config.FIELD_RESOLUTION - sig0
    assert float(pipeline.r_v_settings()[1]) == pipeline.R_V_MAX_SIGMA           # in the cache identity
    c = (60, 70, 3)                                                               # near a periodic face too
    idx = np.indices((n, n, n), dtype=np.float32)
    r2 = sum((((idx[k] + 0.5) - (c[k] + 0.5) + n / 2) % n - n / 2) ** 2 for k in range(3))
    field = (0.3 * (r2 / 40.0 ** 2 - 1.0)).astype(np.float32)                     # 0 exactly at r = 40 cells
    got = pipeline.void_radii_cells(field, np.array([c]), sig)
    assert abs(got["r_v_cells"][0] - 40.0) < 0.1, got["r_v_cells"]


def t_r_v_half_box_margin():
    """The R_v search's half-box clause has a sigma's margin, min(cap sigma, L/2 - sigma), and
    the catalogue records the reach it searched (r_v_settings' sixth element) rather than an algorithm number. A
    catalogue recorded WITHOUT a reach (a legacy cache) is served as it is where the margin changes
    nothing (cap sigma <= L/2 - sigma: the production 10 cells at 512^3) and re-measured, once, where it binds (a large
    sigma). The fixture: 48^3, sigma = 1 cell; cap 24 reaches L/2 = 24 (no margin) but not L/2 - sigma = 23; a
    cap of 10 is short of both."""
    import config
    from dtfelib import pipeline
    # the rule itself, on the real namespaces' numbers
    assert pipeline.r_v_reach_cells(10.0, 512) == 240.0 == pipeline.r_v_reach_cells(10.0, 512, False)
    assert pipeline.r_v_reach_cells(21.4043, 512, False) == 256.0
    assert abs(pipeline.r_v_reach_cells(21.4043, 512) - (256.0 - 21.4043)) < 1e-9
    assert pipeline.r_v_reach_cells(400.0, 512) == 0.0                         # never negative radii
    # the search stops a sigma short of L/2: a void negative out to 30 cells on a 48^3 grid at sigma 1 has no
    # crossing within 23 cells (it would have one within 24 without the margin only if it crossed in the last cell)
    fx = _FieldFixture()
    root = Path(tempfile.mkdtemp(prefix="void_core_margin_"))
    old = (config.CACHE_DIR, config.SMOOTHING_SIGMA_CELLS, pipeline.R_V_MAX_SIGMA)
    grid = root / "density.bin"
    grid.write_bytes(b"x" * 64)
    reads = []
    P = _products_class(fx, reads, grid)
    try:
        config.CACHE_DIR = root / "cache"
        config.SMOOTHING_SIGMA_CELLS = 1.0

        def load():
            buf, rmark = io.StringIO(), len(reads)
            with redirect_stdout(buf):
                cat = P("099", 0.0, sim="SYN").voids()
            return cat, buf.getvalue(), reads[rmark:]

        def legacy():
            """A legacy cached catalogue, as written without the margin: five settings, no reach."""
            path = P("099", 0.0, sim="SYN")._cpath("voids")
            with np.load(path) as z:
                arrays = {k: z[k] for k in z.files}
            assert arrays["r_v_settings"].shape == (6,), arrays["r_v_settings"]
            arrays["r_v_settings"] = arrays["r_v_settings"][:5]
            np.savez_compressed(path, **arrays)
        # a 'production-like' reach: cap 10 sigma = 10 cells, short of L/2 - sigma = 23 -> the margin changes nothing
        pipeline.R_V_MAX_SIGMA = 10.0
        cat, said, rd = load()
        assert float(cat["r_v_settings"][5]) == 10.0, cat["r_v_settings"]
        legacy()
        served, said, rd = load()
        assert said == "" and rd == [], (said, rd)                                # served: no field read, nothing said
        assert served["r_v_settings"].shape == (5,)
        assert pipeline.catalog_r_v_ok(served, 1.0, fx.n) and pipeline.catalog_r_v_ok(served, 10.0, 512)
        # the default cap 24: 24 cells without the margin, 23 with it -> a legacy record is re-measured once
        pipeline.R_V_MAX_SIGMA = old[2]
        cat, said, rd = load()                                                    # the settings changed (cap 10 -> 24)
        assert "not today's" in said and float(cat["r_v_settings"][5]) == 23.0, (said, cat["r_v_settings"])
        legacy()
        assert not pipeline.catalog_r_v_ok({**cat, "r_v_settings": cat["r_v_settings"][:5]}, 1.0, fx.n)
        got, said, rd = load()
        assert "searched out to 24 cells, today's reach is 23" in said and said.count("\n") == 1, said
        assert rd == ["field"] and float(got["r_v_settings"][5]) == 23.0, (rd, got["r_v_settings"])
        _, said, rd = load()
        assert said == "" and rd == [], (said, rd)                                # and then served
    finally:
        config.CACHE_DIR, config.SMOOTHING_SIGMA_CELLS, pipeline.R_V_MAX_SIGMA = old
        shutil.rmtree(root, ignore_errors=True)


def _synthetic_cache(root):
    """A cache of two namespaces and a loose file, each with distinct bytes; returns {relative path: bytes}."""
    files = {"r512_s10_f11_trace_g0.1_v2/099_voids.npz": os.urandom(70_000),
             "r512_s10_f11_trace_g0.1_v2/050_voids.npz": os.urandom(5_000),
             "r512_s21.4291_f23_trace_g0.1_v2/TNG50-3-Dark_099_voids.npz": os.urandom(12_000),
             ".DS_Store": b"x"}
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return files


def t_cache_move():
    """'analyze.py cache --move DIR' on a synthetic cache: a dry run by default (nothing
    moved, no setting); refused while a cache user runs, into a folder holding a namespace of the same name, or when a
    copy does not verify (the old cache untouched, no partial copy left); with --yes every entry moved byte for byte,
    the source removed, the setting recorded -- and a NEW interpreter's config.CACHE_DIR is the moved folder,
    with DTFE_CACHE_DIR still first. cache_users: an analysis python counts, an
    editor on analyze.py does not, nor this process and its ancestors; a DTFE binary does."""
    import config
    import analyze
    root = Path(tempfile.mkdtemp(prefix="void_core_move_"))
    src, dest, xdg = root / "repo" / "cache", root / "local caches" / "DTFE", root / "xdg"
    setting = xdg / "dtfe" / "cache_dir"
    saved = (config.CACHE_DIR, config.CACHE_DIR_SOURCE)
    try:
        files = _synthetic_cache(src)

        def run(**kw):
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = analyze.move_cache(dest, src=src, setting_file=setting, **kw)
            return code, buf.getvalue()

        def untouched():
            return all((src / k).read_bytes() == v for k, v in files.items()) and not setting.exists()
        code, said = run(users=[])
        assert code == 0 and "DRY RUN" in said and untouched() and not dest.exists(), said
        code, said = run(yes=True, users=[(4242, "python3 plot/plot_void_fate.py")])
        assert code == 1 and "in use" in said and "4242" in said and untouched() and not dest.exists(), said
        (dest / "r512_s10_f11_trace_g0.1_v2").mkdir(parents=True)
        code, said = run(yes=True, users=[])
        assert code == 1 and "already holds r512_s10_f11_trace_g0.1_v2" in said and untouched(), said
        (dest / "r512_s10_f11_trace_g0.1_v2").rmdir()
        stub = src / "r512_s10_f11_trace_g0.1_v2" / ".099_voids_b.npz.icloud"   # an evicted iCloud file's placeholder
        stub.write_bytes(b"bplist")
        code, said = run(yes=True, users=[])
        assert code == 1 and "evicted to iCloud" in said and not list(dest.iterdir()), said
        stub.unlink()
        # current macOS evicts in place: the real name, SF_DATALESS in st_flags (no stub at all),
        # and the download it forces lands on the source's disk: its bytes count in the room needed there
        dataless = src / "r512_s10_f11_trace_g0.1_v2" / "099_voids.npz"
        assert dataless.is_file()
        flags, disk_usage = analyze._st_flags, shutil.disk_usage
        total = sum(analyze._dir_size(p)[0] if p.is_dir() else p.stat().st_size for p in src.iterdir())
        analyze._st_flags = lambda q: 0x40000000 if q == dataless else 0
        shutil.disk_usage = lambda p: types.SimpleNamespace(free=int(total * 1.05 + 100e6) + 1)
        try:
            code, said = run(yes=True, users=[])
        finally:
            analyze._st_flags, shutil.disk_usage = flags, disk_usage
        assert code == 1 and "evicted to iCloud (099_voids.npz" in said and not list(dest.iterdir()), said
        assert "downloaded onto the same disk first" in said, said
        assert untouched()
        same = analyze._same_tree
        analyze._same_tree = lambda a, b: False                       # a copy that does not verify
        try:
            code, said = run(yes=True, users=[])
        finally:
            analyze._same_tree = same
        assert code == 1 and "copy failed" in said and untouched() and list(dest.iterdir()) == [], said
        code, said = run(yes=True, users=[])
        assert code == 0 and "Moved" in said, said
        assert all((dest / k).read_bytes() == v for k, v in files.items()), "a moved file differs"
        assert not src.exists() and not list(dest.glob(".*.partial")), list(dest.iterdir())
        assert config.read_cache_setting(setting) == dest.resolve() and config.CACHE_DIR == dest.resolve()
        env = {k: v for k, v in os.environ.items() if k != "DTFE_CACHE_DIR"}
        env["XDG_CONFIG_HOME"] = str(xdg)
        r = subprocess.run([sys.executable, "-c", "import config; print(config.CACHE_DIR); print(config.CACHE_DIR_SOURCE)"],
                           cwd=str(_PY), env=env, capture_output=True, text=True, timeout=120)
        out = r.stdout.strip().splitlines()
        assert r.returncode == 0 and Path(out[0]) == dest.resolve() and str(setting) in out[1], (r.stdout, r.stderr[-300:])
        r = subprocess.run([sys.executable, "-c", "import config; print(config.CACHE_DIR)"], cwd=str(_PY),
                           env=dict(env, DTFE_CACHE_DIR=str(root / "env")), capture_output=True, text=True, timeout=120)
        assert Path(r.stdout.strip()) == root / "env", r.stdout                    # the variable still wins
        # the command line: a dry run by default; --yes only with --prune or --move
        r = subprocess.run([sys.executable, "analyze.py", "cache", "--move", str(root / "elsewhere")], cwd=str(_PY),
                           env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and "DRY RUN" in r.stdout and not (root / "elsewhere").exists(), (r.stdout, r.stderr[-300:])
        r = subprocess.run([sys.executable, "analyze.py", "cache", "--yes"], cwd=str(_PY), env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode != 0 and "--move" in r.stderr, r.stderr[-300:]
        # who counts as a user of the cache
        table = {1: (0, "/sbin/launchd"), 50: (1, "-zsh"), 60: (50, "/opt/py/bin/python3 analyze.py cache --move X"),
                 70: (1, "/usr/bin/python3 /repo/python/plot/plot_void_fate.py --sim TNG100-3-Dark"),
                 71: (1, "vim analyze.py"), 72: (1, "/x/Python /repo/python/gui/app.py"),
                 73: (1, "python3 -m pytest tests/py_analyze_helpers.py"),
                 # the launcher as scripts/gui.sh starts it on macOS (the host bundle), and a job run by its
                 # interpreter (missing either would let --move go ahead with the launcher open)
                 74: (1, "/Users/x/.venvs/dtfe-gui/DTFE Launcher.app/Contents/MacOS/DTFE Launcher "
                         "/Users/x/Mobile Documents/DTFE/python/gui/app.py"),
                 75: (74, "/Users/x/.venvs/dtfe-gui/DTFE Launcher.app/Contents/MacOS/DTFE Launcher "
                          "/repo/python/void_cache.py --sim TNG100-3-Dark"),
                 76: (1, "/usr/bin/python3 tests/py_gui_app2_test.py")}
        got = [pid for pid, _ in analyze.cache_users(table, {80: "PS-DTFE", 50: "DTFE"}, me=60)]
        assert got == [70, 72, 74, 75, 80], got
    finally:
        config.CACHE_DIR, config.CACHE_DIR_SOURCE = saved
        shutil.rmtree(root, ignore_errors=True)


def t_setup_wizard_cache_offer():
    """The setup's first page offers to keep the caches outside iCloud Drive only while
    the repository is synced AND the caches are inside it (config.icloud_cache_offer: ~/Library/Caches/DTFE on macOS,
    ~/.cache/dtfe elsewhere), and its button runs analyze.py's own move (cache_move_command), here on a synthetic
    cache: moved, the setting recorded, this process pointed at the new folder. Needs PySide6 (skipped without)."""
    import config
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        raise Skip("PySide6 not installed (the launcher's venv has it)")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    repo_icloud = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "x" / "DTFE"
    assert config.icloud_cache_offer(repo_icloud, repo_icloud / "python" / "cache") == config.local_cache_dir()
    assert config.icloud_cache_offer(repo_icloud, Path("/tmp/elsewhere")) is None          # already moved
    assert config.icloud_cache_offer(Path("/opt/DTFE"), Path("/opt/DTFE/python/cache")) is None
    expect = ("Library/Caches/DTFE" if sys.platform == "darwin" else ".cache/dtfe")
    assert str(config.local_cache_dir()).endswith(expect), config.local_cache_dir()
    sys.path.insert(0, str(_PY / "gui"))
    import setup_wizard as SW
    app = QApplication.instance() or QApplication([])
    root = Path(tempfile.mkdtemp(prefix="void_core_wizard_"))
    saved_env = {k: os.environ.get(k) for k in ("DTFE_CACHE_DIR", "XDG_CONFIG_HOME")}
    saved = (config.CACHE_DIR, config.CACHE_DIR_SOURCE, config.icloud_cache_offer)
    try:
        src, dest = root / "cache", root / "local" / "DTFE"
        files = _synthetic_cache(src)
        # the move runs in a child process: it finds the cache and the setting through its environment
        os.environ["DTFE_CACHE_DIR"], os.environ["XDG_CONFIG_HOME"] = str(src), str(root / "xdg")
        config.icloud_cache_offer = lambda *a: None
        dlg = SW.SetupDialog(None, str(root))
        assert dlg.cache_box.isHidden(), "offered with nothing to offer"
        dlg.deleteLater()
        config.icloud_cache_offer = lambda *a: dest
        dlg = SW.SetupDialog(None, str(root))
        assert not dlg.cache_box.isHidden() and dlg.cache_dir.text() == str(dest)
        assert "--" not in dlg.cache_btn.text() and "--" not in dlg.cache_info.text()
        cmd = SW.cache_move_command(dest)
        assert cmd[2:] == ["cache", "--move", str(dest), "--yes"] and cmd[1].endswith("analyze.py"), cmd
        # the same function, with the live process check replaced (whatever else runs on this machine meanwhile --
        # another figure script -- would refuse the move, rightly): first a user present, then none
        real_cmd = SW.cache_move_command

        def run_move(users):
            SW.cache_move_command = lambda folder: [
                sys.executable, "-c", f"import sys, analyze; sys.exit(analyze.move_cache({str(folder)!r}, yes=True, "
                                      f"users={users!r}))", str(folder)]
            try:
                dlg._move_cache()
            finally:
                SW.cache_move_command = real_cmd
            assert dlg._cache_proc is not None and dlg._cache_proc.waitForFinished(120000)
            app.processEvents()
            assert dlg._cache_proc is None, "the finished move was not handled"
        run_move([(4242, "python3 analyze.py voids")])
        assert dlg.cache_info.text().startswith("not moved: refused") and "4242" in dlg.cache_info.text(), \
            dlg.cache_info.text()
        assert all((src / k).read_bytes() == v for k, v in files.items()) and dlg.cache_btn.isEnabled()
        run_move([])
        assert all((dest / k).read_bytes() == v for k, v in files.items()) and not src.exists(), dlg.cache_info.text()
        assert dlg.cache_info.text().startswith("✔") and config.CACHE_DIR == dest.resolve(), dlg.cache_info.text()
        assert config.read_cache_setting(root / "xdg" / "dtfe" / "cache_dir") == dest.resolve()
        dlg.deleteLater()
    finally:
        config.CACHE_DIR, config.CACHE_DIR_SOURCE, config.icloud_cache_offer = saved
        for k, v in saved_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        shutil.rmtree(root, ignore_errors=True)


def main():
    print("=" * 60)
    print(" void core: R_v definitions, cache identity, cache dir, void gate")
    print("=" * 60)
    print("synthetic:")
    check("2026-10-11: crossing / delta08 / ridge from one shell matrix on an analytic compensated void (periodic, "
          "shallow, overdense, truncated), the crossing = void_radius_cells", t_radii_definitions)
    check("2026-10-11: catalog.R_V_DEFINITIONS, r_v_cells / r_v_mpc per definition, NaN without the key", t_r_v_accessors)
    check("2026-10-11: R_v settings in the cache identity: built with them; missing or other ones re-measured once from "
          "the smoothed field (no Hessian), then silent", t_r_v_cache_identity)
    check("2026-10-11: DTFE_VOID_GATE=none ungates the served catalogue only (cache bytes unchanged), tag 'gatenone'; "
          "export columns r_v_delta08_mpc / r_v_ridge_mpc and the R_v provenance", t_void_gate_and_export)
    check("2026-10-11: DTFE_CACHE_DIR (default kept, ~ expanded, the products follow) and DTFE_VOID_GATE at config "
          "import (an unknown value refused)", t_env_overrides)
    check("2026-10-11 (G2): config.CROSSBOX_SMOOTHING_MPC = 2.1625 imported by plot_void_crossbox and analyze; "
          "thesis_runs.sh's literal equals it", t_crossbox_constant)
    check("2026-10-11: R_v search cap 24 sigma (a sigma short of L/2 at production): a 20-sigma void is measured, "
          "not NaN", t_r_v_cap)
    check("2026-10-11 (3d): analyze.py cache --move: dry run by default; refused while in use, on a clash or a failed "
          "verify (nothing touched); moved byte for byte, the setting read by a new interpreter", t_cache_move)
    check("2026-10-11 (3d): the setup offers the caches out of iCloud Drive only when the repository and the caches are "
          "in it, and its button runs the same move", t_setup_wizard_cache_offer)
    check("2026-10-11 (3c): the half-box clause has a sigma's margin; the reach is recorded, a legacy record served "
          "where the margin changes nothing and re-measured where it binds", t_r_v_half_box_margin)
    print("data-backed:")
    check("2026-10-11: TNG100-3-Dark z = 0 served with today's R_v settings; distinct median radii in order", t_real_tng100_z0)
    print("=" * 60)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
