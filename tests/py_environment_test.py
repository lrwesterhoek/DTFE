#!/usr/bin/env python3
"""Tests for the void-environment classification and the caustic skeleton's control (2026-10-11).

  * SYNTHETIC (always run): the enclosed contrast against analytic profiles; void-in-cloud / void-in-void on a
    top-hat grid with and without a compensating shell, the wrap rule; classify_catalog's sample; the figure
    script plot_void_environment.py end to end on a stubbed pipeline (its files, the tracking panel by type);
    skeleton.filter_components (size and mass thresholds), the random-sphere null and its z-scores;
    plot_caustic_skeleton's null text and option tags.
  * DATA-BACKED (the T7; skipped with a note without it): TNG100-3-Dark z = 0 classified in a fresh interpreter
    (DTFE_SIM sets the frame), a void-in-cloud fraction strictly between 0 and 1.

Usage:  python3 tests/py_environment_test.py            (exit 0 = pass)
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
_FIGS_TMP = tempfile.mkdtemp(prefix="env_figs_")      # before config is imported: never the real figure root
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


# ---------------------------------------------------------------- synthetic: the classification
def _tophat_grid(n=64, voids=()):
    """rho/rho_bar = 1 with, per (centre cell, R cells, shell density or None), rho 0.2 inside R and the shell
    density between R and 2R; distances from cell centres (sample_grid's convention: cell i at (i + 0.5) / n)."""
    rho = np.ones((n, n, n))
    idx = np.indices((n, n, n)).astype(float)
    for c, r, shell in voids:
        d = idx - np.asarray(c, float).reshape(3, 1, 1, 1)
        d -= n * np.rint(d / n)
        dist = np.sqrt((d * d).sum(axis=0))
        if shell is not None:
            rho[(dist >= r) & (dist < 2 * r)] = shell
        rho[dist < r] = 0.2
    return rho


def t_enclosed_contrast():
    from dtfelib import classify
    x = classify.shell_bins(2.0, 40)
    assert x[-1] == 2.0 and len(x) == 40
    flat = classify.enclosed_contrast(x, np.full(len(x), 0.2))
    assert np.allclose(flat, -0.8, atol=1e-4), flat[:3]          # the fine-grid trapezoid read back by interpolation: ~2e-5
    # a step at x = 1: 0.2 inside, 1.5 outside -> Delta(<2) = (0.2 + 1.5 * 7) / 8 - 1, within the linear-between-shells blur
    step = np.where(x < 1.0, 0.2, 1.5)
    want = (0.2 + 1.5 * 7) / 8 - 1
    got = classify.enclosed_contrast(x, step)[-1]
    assert abs(got - want) < 0.03, (got, want)
    two = classify.enclosed_contrast(x, np.vstack([step, np.full(len(x), 0.2)]))
    assert two.shape == (2, 40) and abs(two[0, -1] - got) < 1e-12 and abs(two[1, -1] + 0.8) < 1e-4


def t_classify_grid():
    from dtfelib import classify
    n, r = 64, 5
    d_cloud = 0.4
    shell = ((1 + d_cloud) * 8 - 0.2) / 7                      # Delta(<2R) = +0.4 exactly in the continuum
    rho = _tophat_grid(n, [((16, 16, 16), r, shell), ((48, 48, 48), r, None)])
    centers = (np.array([[16, 16, 16], [48, 48, 48], [16, 48, 16]]) + 0.5) / n
    g = classify.classify_grid(rho, centers, [r, r, 20.0])        # the third: 2 x 20 > n / 2, dropped
    assert list(g["kept"]) == [True, True, False] and g["dropped"] == 1, g["kept"]
    assert abs(g["delta_enclosed"][0] - d_cloud) < 0.06, g["delta_enclosed"]
    assert abs(g["delta_enclosed"][1] - ((0.2 + 7) / 8 - 1)) < 0.03, g["delta_enclosed"]
    assert list(g["type"]) == ["S", "R", ""] and list(g["vic"]) == [True, False, False], g["type"]
    assert np.isnan(g["delta_enclosed"][2]) and np.isnan(g["profile"][2]).all()
    f, lo, hi = classify.vic_fraction(g["vic"][g["kept"]])
    assert f == 0.5 and 0.0 <= lo <= f <= hi <= 1.0, (f, lo, hi)
    assert all(np.isnan(classify.vic_fraction([])))


def _synthetic_catalog(cell):
    """Three well-resolved voids (two in the top-hat grid, one dropped by the wrap rule) and one unresolved."""
    coords = np.array([[16, 16, 16], [48, 48, 48], [16, 48, 16], [40, 10, 40]])
    return {"coords": coords, "well_resolved": np.array([True, True, True, False]),
            "r_v_cells": np.array([5.0, 5.0, 20.0, 4.0]), "cell_mpc": np.float64(cell),
            "delta_values": np.array([-0.8, -0.7, -0.6, -0.5]),
            "bbks_params": np.array([[0.2, 0.05], [0.3, -0.1], [0.1, 0.0], [0.2, 0.0]]),
            "axis_ratios": np.array([[0.8, 0.6], [0.7, 0.5], [0.9, 0.8], [0.8, 0.7]]),
            "semi_axes": np.full((4, 3), 9.0 * cell, np.float32)}


def t_classify_catalog():
    from dtfelib import classify
    n, cell = 64, 1.5
    rho = _tophat_grid(n, [((16, 16, 16), 5, ((1.4) * 8 - 0.2) / 7), ((48, 48, 48), 5, None)])
    cat = _synthetic_catalog(cell)
    out = classify.classify_catalog(cat, rho, cell)
    assert out["n_sample"] == 3 and out["dropped"] == 1 and list(out["index"]) == [0, 1], out["index"]
    assert list(out["type"]) == ["S", "R"] and np.allclose(out["r_v_mpc"], 7.5), out["r_v_mpc"]
    assert np.allclose(out["e"], [0.2, 0.3]) and np.allclose(out["c_over_a"], [0.6, 0.5]) and np.allclose(out["delta_c"], [-0.8, -0.7])
    assert out["definition"] == "crossing" and out["factor"] == 2.0
    none = classify.classify_catalog(cat, rho, cell, definition="ridge")     # no r_v_ridge_cells key: no void, no crash
    assert none["n_sample"] == 0 and len(none["index"]) == 0


# ---------------------------------------------------------------- synthetic: the figure script
def t_environment_script():
    import config
    import plot_void_environment as pev
    from dtfelib import classify, catalog
    n, cell = 64, 1.5
    root = Path(tempfile.mkdtemp(prefix="env_run_"))
    from dtfelib import cli
    saved = (pev.pipeline, pev.catalog, pev.classify, pev.style.note_source, cli.DATA_ROOT)
    try:
        data = root / "data" / "SYN"
        for s in (50, 99):
            (data / f"snapdir_{s:03d}").mkdir(parents=True)
        rho = _tophat_grid(n, [((16, 16, 16), 5, ((1.4) * 8 - 0.2) / 7), ((48, 48, 48), 5, None)])
        cat = _synthetic_catalog(cell)

        class _FS:
            method, prefix, snapdir = "ps", "ps_output.", data / "snapdir_099"

        class _Prod:
            def __init__(self, snap, *a, **kw):
                self.snap, self.fs, self.density_raw = int(snap), _FS(), rho
                self.redshift = 0.0 if int(snap) == 99 else 1.0
            def voids(self):
                return cat
            def release(self):
                pass
        real_pl = pev.pipeline
        pev.pipeline = types.SimpleNamespace(products=lambda s, *a, **kw: _Prod(s), add_smoothing_mpc_arg=real_pl.add_smoothing_mpc_arg,
                                             apply_smoothing_args=real_pl.apply_smoothing_args, smoothing_tag=real_pl.smoothing_tag)
        frame = lambda p: (cell, cell * n, n, "header")
        pev.catalog = types.SimpleNamespace(frame_of=frame, r_v_mpc=catalog.r_v_mpc, R_V_DEFINITIONS=catalog.R_V_DEFINITIONS)

        def classify_voids(prod, factor=2.0, distinct=True, definition="crossing", criterion="density_raw"):
            out = classify.classify_catalog(prod.voids(), prod.density_raw, cell, factor=factor, distinct=distinct,
                                            definition=definition)
            out.update(method="ps", prefix="ps_output.")
            return out
        pev.classify = types.SimpleNamespace(**{k: getattr(classify, k) for k in dir(classify) if not k.startswith("__")})
        pev.classify.classify_voids = classify_voids
        pev.style.note_source = lambda fs: None
        # a tracking set: void 0 at the void-in-cloud, void 1 at the void-in-void, void 2 without R_v anywhere
        tdir = root / "tracking"
        tdir.mkdir()
        for k, (c, r_v) in enumerate((((16, 16, 16), 7.5), ((48, 48, 48), 7.5), ((30, 30, 30), np.nan))):
            np.savez(tdir / f"void{k}_track.npz", snap=np.array([99, 50]), redshift=np.array([0.0, 1.0]),
                     r_v=np.array([r_v, 6.0]), r_eff=np.array([np.nan, np.nan]), e=np.array([0.2, 0.1]),
                     p=np.array([0.0, 0.05]), delta_min=np.array([-0.8, -0.5]),
                     center0_frac=(np.asarray(c) + 0.5) / n, catalog_index0=np.int64(k))
        np.savez(tdir / "void0_track_dtfe.npz", snap=np.array([99]))          # another set: not read
        buf = _io.StringIO()
        with redirect_stdout(buf):
            pev.main(["--sim", "SYN", "--data-root", str(root / "data"), "--tracking-dir", str(tdir),
                      "--criterion", "density_raw"])
        text = buf.getvalue()
        out = Path(config.figures_path("void_environment")) / "SYN"
        names = sorted(q.name for q in out.iterdir())
        # density_raw: the untagged names (2026-10-11: they keep what they held before the velocity default)
        want = ["environment_evolution.png", "environment_stats.npz", "environment_tracking.png",
                "environment_z0.00.png", "environment_z1.00.png"]
        assert names == want, (names, text[-500:])
        assert "f_vic = 0.500" in text and "tracked void 2: no R_v" in text and "0: S" in text and "1: R" in text, text[-800:]
        with np.load(out / "environment_stats.npz") as z:
            assert list(z["snaps"]) == [50, 99] and list(z["f_vic"]) == [0.5, 0.5] and list(z["n"]) == [2, 2]
            assert list(z["snap099_vic"]) == [True, False] and list(z["snap099_index"]) == [0, 1]
            assert list(z["tracked_voids"]) == [0, 1] and list(z["tracked_type"]) == ["S", "R"]
            assert z["r_v_mpc_S_pct"].shape == (2, 3)
        # a non-default factor tags the set, no tracking: an earlier tracking figure of that tag is not left behind
        (out / "environment_tracking_x1.5.png").write_bytes(b"old")
        with redirect_stdout(_io.StringIO()):
            pev.main(["--sim", "SYN", "--data-root", str(root / "data"), "--tracking-dir", str(root / "none"),
                      "--factor", "1.5", "--criterion", "density_raw"])
        names = sorted(q.name for q in out.iterdir() if "_x1.5" in q.name)
        assert names == ["environment_evolution_x1.5.png", "environment_stats_x1.5.npz",
                         "environment_z0.00_x1.5.png", "environment_z1.00_x1.5.png"], names
        assert pev.name_tag({99: "dtfe"}, {99: "output."}, True, 2.0, "delta08") == "_dtfe_overlaps_rvdelta08_velocity"
        assert pev.name_tag({99: "dtfe"}, {99: "output."}, True, 2.0, "delta08", "density_raw") == "_dtfe_overlaps_rvdelta08"
        assert pev.name_tag({99: "ps"}, {99: "ps_caustic."}, False, 2.0, criterion="density_raw") == "_ps_caustic"
        assert pev.name_tag({99: "ps"}, {99: "ps_caustic."}, False, 2.0) == "_ps_caustic_velocity"
    finally:
        pev.pipeline, pev.catalog, pev.classify, pev.style.note_source, _root = saved
        cli.use_data_root(_root)                    # main()'s --data-root is process-wide
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- synthetic: the caustic skeleton's control
def t_filter_components():
    from dtfelib import skeleton as sk
    n = 16
    m = np.zeros((n, n, n), bool)
    m[4] = True                                     # a slab: one periodic component of 256 cells
    m[12, 3, 3:6] = True                            # a 3-cell flake
    out, info = sk.filter_components(m, min_cells=10)
    assert out[4].all() and not out[12].any() and info == {"n": 2, "kept": 1, "cells_removed": 3, "min_cells": 10,
                                                            "min_mass_fraction": 0.0}, info
    same, info0 = sk.filter_components(m)                        # no threshold: untouched, not labelled
    assert same is m or np.array_equal(same, m)
    assert info0["n"] is None
    den = np.ones((n, n, n))
    den[12, 3, 3:6] = 1000.0                                     # the flake heavy: the mass rule keeps it, drops the slab
    out, info = sk.filter_components(m, density=den, min_mass_fraction=0.3)
    assert out[12, 3, 3:6].all() and not out[4].any() and info["kept"] == 1, info
    try:
        sk.filter_components(m, min_mass_fraction=0.1)
        raise AssertionError("a mass threshold without a density grid was accepted")
    except ValueError:
        pass


def t_random_sphere_null():
    from dtfelib import skeleton as sk
    n = 32
    sheet = np.zeros((n, n, n), bool)
    sheet[0] = sheet[16] = True                     # walls at x = 0 and 16: the farthest cells sit at x = 8 and 24
    masks = {"fold": sheet.copy(), "sheet": sheet}
    r = np.full(6, 3.0)
    centres = np.array([[8, 5, 5], [24, 9, 20], [8, 20, 11], [24, 2, 30], [8, 13, 27], [24, 25, 3]])
    v = sk.sphere_metrics(masks, centres, r, n, (0.8, 1.0, 1.2), cutout_cells=12)
    assert np.allclose(v["dist_r"], 8.0 / 3.0) and not v["bounded"].any(), v["dist_r"]
    assert np.allclose(v["fold"], 0.0), v["fold"]                # shells of 2.4-3.6 cells never reach a wall 8 away
    nul = sk.random_sphere_null(masks, r, n, (0.8, 1.0, 1.2), 12, n_draws=20, seed=3)
    again = sk.random_sphere_null(masks, r, n, (0.8, 1.0, 1.2), 12, n_draws=20, seed=3)
    assert all(np.array_equal(nul[k], again[k], equal_nan=True) for k in sk.NULL_METRICS), "the null is not seeded"
    assert nul["dist_r"].shape == (20,) and np.nanmax(nul["dist_r"]) < 8.0 / 3.0 + 1e-9
    z = sk.null_zscores(v, nul)
    assert z["dist_r"]["z"] > 3 and z["fold"]["z"] < 0 and z["dist_r"]["draws"] == 20, z
    assert abs(z["dist_r"]["void"] - 8.0 / 3.0) < 1e-12
    flat = sk.null_zscores({"fold": [np.nan], "sheet": [], "dist_r": [1.0]},
                           {"fold": np.full(3, np.nan), "sheet": np.ones(3), "dist_r": np.ones(3)})
    assert np.isnan(flat["fold"]["z"]) and np.isnan(flat["sheet"]["z"]) and np.isnan(flat["dist_r"]["z"]), flat  # no spread: NaN
    empty = sk.random_sphere_null(masks, [], n, (1.0,), 4, n_draws=2)
    assert np.isnan(empty["fold"]).all()


def t_caustic_script_null_text():
    import plot_caustic_skeleton as pcs
    from dtfelib import skeleton as sk
    from dtfelib.io import CAUSTIC_COLLAPSE, CAUSTIC_PARITY_POS, CAUSTIC_PARITY_NEG
    n = 8
    cc = np.full((n, n, n), CAUSTIC_COLLAPSE[0] | CAUSTIC_PARITY_POS, np.int32)
    cc[2] |= CAUSTIC_COLLAPSE[1] | CAUSTIC_PARITY_POS | CAUSTIC_PARITY_NEG
    summ = sk.skeleton_summary(cc, cell_mpc=1.0)
    summ["null"] = {"draws": 50, "seed": 0, "box_fold_volume": 0.125, "box_sheet_volume": 0.125, "bounded_fraction_mean": 0.0,
                    "metrics": {"fold": {"void": 0.3, "null_mean": 0.2, "null_std": 0.01, "z": 10.0, "draws": 50},
                                "sheet": {"void": 0.3, "null_mean": 0.2, "null_std": float("nan"), "z": float("nan"), "draws": 50},
                                "dist_r": {"void": None, "null_mean": None, "null_std": None, "z": None, "draws": 0}}}
    summ["sheet_min_density"] = 1.0
    summ["sheet_filter"] = {"n": 3, "kept": 1, "cells_removed": 7, "min_cells": 100, "min_mass_fraction": 0.0}
    txt = pcs.summary_text(sk.json_safe(summ), "t")
    rows = {ln.split()[0] + " " + ln.split()[1]: ln.split()[2:] for ln in txt.splitlines() if len(ln.split()) > 2}
    assert "random-sphere null: 50 draws" in txt and rows["fold shell"] == ["0.3000", "0.2000", "0.0100", "+10.00"], txt
    assert rows["sheet shell"] == ["0.3000", "0.2000", "--", "--"] and rows["wall dist/R"] == ["--", "--", "--", "--"], txt
    assert "1 of 3 kept, 7 cells removed" in txt and "denser than 1 x the mean" in txt, txt
    a = types.SimpleNamespace(connectivity=26, voids=True, shells="0.8,0.9,1.0,1.1,1.2", radius="r_v", keep_overlaps=False,
                              cutout=None, null_draws=pcs.NULL_DRAWS, null_seed=0, min_component=0, min_component_mass=0.0,
                              sheet_min_density=0.0)
    assert pcs.skeleton_tag(a, "ps_caustic.", "0.8,0.9,1.0,1.1,1.2") == "_ps_caustic"
    b = types.SimpleNamespace(**dict(vars(a), null_draws=10, null_seed=2, min_component=500, min_component_mass=1e-4,
                                   sheet_min_density=1.0))
    assert pcs.skeleton_tag(b, "ps_output.", "0.8,0.9,1.0,1.1,1.2") == "_null10_seed2_min500_minm0.0001_dens1"
    c = types.SimpleNamespace(**dict(vars(b), voids=False))                  # the void options count only with --voids
    assert pcs.skeleton_tag(c, "ps_output.", "0.8,0.9,1.0,1.1,1.2") == ""


def t_caustic_snapshots():
    """--all-caustic: the snapdirs with a caustic_class grid under the prefix, in order; one without it, or without a
    header (FieldSet refuses it), is not listed."""
    import plot_caustic_skeleton as pcs
    try:
        import h5py
    except ImportError:
        print("    (h5py is not installed: no snapshot header can be written, check skipped)")
        return
    root = Path(tempfile.mkdtemp(prefix="env_cs_"))
    try:
        n = 8
        for snap, grids in ((33, ("a_den", "causticClass")), (50, ("a_den",)), (99, ("a_den", "causticClass"))):
            d = root / "SYN" / f"snapdir_{snap:03d}"
            d.mkdir(parents=True)
            with h5py.File(d / f"combined_{snap:03d}.hdf5", "w") as f:
                h = f.create_group("Header").attrs
                h["BoxSize"], h["Redshift"], h["HubbleParam"], h["HFreeUnits"] = 100000.0, 0.0, 0.6774, 1
                h["NumPart_ThisFile"], h["MassTable"] = np.array([0, n ** 3, 0, 0, 0, 0]), np.array([0, 1.0, 0, 0, 0, 0])
            for g in grids:
                np.ones(n ** 3, np.float32).tofile(str(d / f"ps_output.{g}"))
        (root / "SYN" / "snapdir_067").mkdir()                        # no header: not listed, no crash
        np.ones(n ** 3, np.float32).tofile(str(root / "SYN" / "snapdir_067" / "ps_output.causticClass"))
        got = pcs.caustic_snapshots("SYN", root, "ps", None)
        assert got == [33, 99], got
        assert pcs.caustic_snapshots("SYN", root, "ps", "ps_caustic") == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- data-backed: TNG100-3-Dark
def t_tng100_vic():
    from dtfelib import cli
    sd = cli.sim_dir("TNG100-3-Dark", os.environ.get("DTFE_DATA_ROOT", str(cli._T7_DATA))) / "snapdir_099"
    if not any(sd.glob("ps_output.*den*")):
        raise FileNotFoundError(f"no PS-DTFE density in {sd}")
    code = ("import sys; sys.path.insert(0, 'python'); from dtfelib import pipeline, classify; "
            "e = classify.classify_voids(pipeline.products(99, sim='TNG100-3-Dark', method='ps')); "
            "print(len(e['vic']), float(e['vic'].mean()))")
    env = dict(os.environ, DTFE_SIM="TNG100-3-Dark")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-800:]
    n, f = r.stdout.split()[-2:]
    print(f"    TNG100-3-Dark z = 0: {n} distinct resolved voids, void-in-cloud fraction {float(f):.3f}")
    assert int(n) > 20 and 0.0 < float(f) < 1.0, r.stdout


def main():
    global DATA_BACKED
    import matplotlib
    matplotlib.use("Agg")
    print("synthetic")
    check("enclosed contrast: flat -0.8, a step's analytic Delta(<2), many profiles at once", t_enclosed_contrast)
    check("classify_grid: compensated top-hat S (Delta +0.4), bare top-hat R (-0.1), the wrap rule drops 2 R_v > L/2", t_classify_grid)
    check("classify_catalog: the resolved R_v sample, rows, shapes and depths carried, a missing definition -> no void", t_classify_catalog)
    check("plot_void_environment.py end to end (stubbed pipeline): files, f_vic, per-void npz, tracking by type, tags", t_environment_script)
    check("skeleton.filter_components: size threshold, mass threshold, none, no density refused", t_filter_components)
    check("skeleton random-sphere null: seeded, the same metrics, z-scores with sign, NaN without spread", t_random_sphere_null)
    check("plot_caustic_skeleton: the null table and filter line in the text, the new option tags", t_caustic_script_null_text)
    check("plot_caustic_skeleton --all-caustic: the snapshots with a caustic_class grid under the prefix", t_caustic_snapshots)
    DATA_BACKED = True
    print("data-backed")
    check("TNG100-3-Dark z = 0: void-in-cloud fraction of the distinct resolved voids", t_tng100_vic)
    print(f"\nRESULT: {len(PASS)} passed, {len(FAIL)} failed" + (f", {len(SKIP)} skipped" if SKIP else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
