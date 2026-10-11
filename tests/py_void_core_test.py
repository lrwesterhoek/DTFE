#!/usr/bin/env python3
"""Tests for the void-catalogue core of 2026-10-11 (dtfelib.pipeline / catalog, config):

  * the three R_v definitions (catalog.R_V_DEFINITIONS: crossing, delta08, ridge) from one shell matrix
    (pipeline.void_radii_cells) on an analytic compensated void, the crossing bit for bit the old one;
  * catalog.r_v_cells / r_v_mpc per definition (NaN without the key), the export's new columns and provenance;
  * the R_v settings in the cache identity: a catalogue without them, or with other ones, is re-measured once
    (the smoothed field read, the Hessian not), then served silently;
  * DTFE_CACHE_DIR and DTFE_VOID_GATE (config at import; the gate applied to the served dict, never to the cache,
    and tagged through pipeline.smoothing_tag).

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

_PY = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(_PY))
# the figure root before config/dtfelib are imported: a temp folder, never the real root (the T7 when it is mounted)
_FIGS_TMP = tempfile.mkdtemp(prefix="void_core_figs_")
os.environ["DTFE_FIGURES_ROOT"] = _FIGS_TMP
os.environ.pop("DTFE_VOID_GATE", None)          # this process tests the default gate; the subprocesses set their own
atexit.register(shutil.rmtree, _FIGS_TMP, True)

PASS, FAIL, SKIP = [], [], []


class Skip(Exception):
    pass


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Skip as e:
        SKIP.append(name)
        print(f"  SKIP  {name}  ({e})")
    except AssertionError as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")
    except Exception as e:                      # a crash is a failure, named
        FAIL.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")


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
    assert np.allclose(catalog.r_v_mpc(cat), [1.0, np.nan, 2.0], equal_nan=True)            # the old call: crossing
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
        assert wrote == 1 and pipeline.catalog_r_v_ok(cat), (wrote, said)
        assert np.array_equal(cat["r_v_settings"], pipeline.r_v_settings())
        for k in ("r_v_cells", "r_v_delta08_cells", "r_v_ridge_cells"):
            assert cat[k].shape == (len(cat["coords"]),), k
        assert np.isfinite(cat["r_v_cells"]).any() and np.isfinite(cat["r_v_ridge_cells"]).any()
        _, said, wrote, rd = load()
        assert said == "" and wrote == 0 and rd == [], (said, wrote, rd)  # fresh: read silently, nothing re-measured
        # a catalogue from before 2026-10-11: no settings, no delta08 / ridge -> re-measured ONCE, the Hessian not read
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
        # the export: the two new radius columns (cells x cell) and the R_v settings + gate in the provenance
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


def _config_in_child(env_extra, expr):
    env = {k: v for k, v in os.environ.items() if k not in ("DTFE_CACHE_DIR", "DTFE_VOID_GATE")}
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
    """TNG100-3-Dark z = 0 (T7): the served catalogue carries today's R_v settings and all three radii; the crossing
    median of the distinct voids is the 2026-10-11 rebuild's 6.94 Mpc. A fresh interpreter (DTFE_SIM decides config's
    frame); the shared cache is read, or re-measured once if stale."""
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
    assert int(nd) == 77 and abs(float(cr) - 6.938) < 0.01, r.stdout
    assert float(d08) < float(cr) < float(ridge), r.stdout       # inner sphere < first crossing < compensation ridge


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
    print("data-backed:")
    check("2026-10-11: TNG100-3-Dark z = 0 served with today's R_v settings; distinct median radii in order", t_real_tng100_z0)
    print("=" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
