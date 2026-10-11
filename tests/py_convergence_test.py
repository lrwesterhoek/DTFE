#!/usr/bin/env python3
"""Tests for the resolution and estimator convergence of the void statistics (2026-10-11): dtfelib.convergence
(the frame context, block averages, catalogues of any grid, void-by-void matching) and plot_void_convergence.py
(its panels, the estimator table, the panels it cannot draw).

Two tiers:
  * SYNTHETIC (always run, CI): analytic compensated voids on a 64^3 grid, hand-made catalogues and results.
  * DATA-BACKED (the T7's TNG100-3-Dark z = 0 grid and its cached catalogues): the 256^3 block-averaged catalogue
    against the native one; skipped with a note when the data is not there.

Usage:  python3 tests/py_convergence_test.py            (exit 0 = pass)
"""

import atexit
import io
import os
import shutil
import subprocess
import sys
import tempfile
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "python" / "plot"))
_FIGS_TMP = tempfile.mkdtemp(prefix="convergence_figs_")
os.environ["DTFE_FIGURES_ROOT"] = _FIGS_TMP
os.environ["DTFE_NO_MIRROR"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
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
    except Exception as e:      # a crash is this test's failure, not the end of the run
        import traceback
        FAIL.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        print("".join("        " + line for line in traceback.format_exc().splitlines(True)[-6:]), end="")


N, BOX, SIGMA = 64, 64.0, 2.0
CENTRES = [(16.3, 16.1, 16.7), (48.2, 16.5, 40.1), (16.6, 48.4, 47.9), (47.5, 47.3, 15.2)]


def _field(seed=1):
    """rho / rho_bar on a 64^3 grid of a 64 Mpc box: four compensated voids (a Gaussian dip of depth -0.8 with a
    ridge at 2.5 s, s = 4-7 cells) on a 2% noise floor."""
    x = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = np.ones((N, N, N)) + 0.02 * np.random.default_rng(seed).standard_normal((N, N, N))
    for i, c in enumerate(CENTRES):
        d = [np.minimum(abs(G - cc), N - abs(G - cc)) for G, cc in zip((X, Y, Z), c)]
        r = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        s = 4.0 + i
        rho += -0.8 * np.exp(-r ** 2 / (2 * s * s)) + 0.4 * np.exp(-(r - 2.5 * s) ** 2 / (2 * 1.5 ** 2))
    return np.clip(rho, 0.01, None).astype(np.float32)


_CACHE = {}


def _cats():
    if not _CACHE:
        from dtfelib import convergence as cv
        rho = _field()
        _CACHE.update(rho=rho, native=cv.build_catalog(rho, BOX, SIGMA),
                      coarse=cv.build_catalog(cv.block_average(rho, 2), BOX, SIGMA))
    return _CACHE


# ---------------------------------------------------------------- dtfelib.convergence
def t_frame_and_blocks():
    """frame() sets the coarse grid's cell, sigma in its cells and the footprint, and restores config on an
    exception; block_average keeps the mean and is the exact block mean; the velocity mean is mass-weighted."""
    import config
    from dtfelib import convergence as cv, pipeline
    before = {k: getattr(config, k) for k in cv.FRAME_KEYS}
    try:
        with cv.frame(256, 110.7, 2.16245):
            assert config.FIELD_RESOLUTION == 256 and abs(config.CELL_SIZE - 110.7 / 256) < 1e-12
            assert abs(config.SMOOTHING_SIGMA_CELLS - 2.16245 / (110.7 / 256)) < 1e-6
            assert config.FOOTPRINT_SIZE == config.footprint_for(config.SMOOTHING_SIGMA_CELLS)
            assert pipeline._param_hash().startswith("r256_"), pipeline._param_hash()
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert {k: getattr(config, k) for k in cv.FRAME_KEYS} == before
    g = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)
    b = cv.block_average(g, 2)
    assert b.shape == (4, 4, 4) and abs(float(b.mean()) - float(g.mean())) < 1e-6
    assert abs(float(b[1, 2, 3]) - float(g[2:4, 4:6, 6:8].mean())) < 1e-6
    rho = np.ones((4, 4, 4), np.float32)
    rho[0, 0, 0] = 3.0                                        # one heavy cell in the first block
    vel = np.zeros((4, 4, 4, 3), np.float32)
    vel[0, 0, 0, 0] = 100.0
    v = cv.block_average_velocity(vel, rho, 2, "mass")
    assert abs(float(v[0, 0, 0, 0]) - 300.0 / 10.0) < 1e-4, v[0, 0, 0, 0]    # sum rho v / sum rho = 300 / (3 + 7)
    try:
        cv.block_average(np.zeros((6, 6, 6)), 4)
        raise AssertionError("a grid not divisible by the factor was accepted")
    except ValueError:
        pass


def t_velocity_weighting():
    """2026-10-11: the coarse velocity follows the run's deposit. 'volume' (the production PS-DTFE convention and
    standard DTFE) is the plain block mean, not sum rho v / sum rho; velocity_weighting reads it from the run log's
    command line (--ps-volume-weighted), 'mass' without it, 'volume' (said) without a log and for standard DTFE."""
    from types import SimpleNamespace
    from dtfelib import convergence as cv
    rng = np.random.default_rng(3)
    rho = (0.1 + rng.random((8, 8, 8)) ** 4 * 20).astype(np.float32)          # far from uniform: the two differ
    vel = rng.normal(0.0, 100.0, (8, 8, 8, 3)).astype(np.float32)
    vol, mass = cv.block_average_velocity(vel, rho, 2), cv.block_average_velocity(vel, rho, 2, "mass")
    assert np.allclose(vol, cv.block_average(vel, 2), atol=1e-4), "the default is not the plain (volume) block mean"
    assert float(np.abs(vol - mass).max()) > 1.0, "the mass-weighted mean equals the volume one on a lumpy rho"
    try:
        cv.block_average_velocity(vel, rho, 2, "momentum")
        raise AssertionError("an unknown weighting was accepted")
    except ValueError:
        pass
    d = Path(tempfile.mkdtemp(prefix="vw_", dir=_FIGS_TMP))
    prod = SimpleNamespace(fs=SimpleNamespace(method="ps", snapdir=d, prefix="ps_mw."))
    w, why = cv.velocity_weighting(prod)
    assert w == "volume" and "assumed" in why, (w, why)
    (d / "ps_mw.runlog").write_text("RUNNING ./PS-DTFE in out --grid 64 --ps-vertex-mass --ps-volume-weighted "
                                    "--periodic\n")
    assert cv.velocity_weighting(prod)[0] == "volume"
    (d / "ps_mw.runlog").write_text("RUNNING ./PS-DTFE in out --grid 64 --ps-vertex-mass --periodic\n"
                                    "PS_VOLUME_WEIGHTED=1 is not the flag; --ps-volume-weightedness neither\n")
    assert cv.velocity_weighting(prod)[0] == "mass", cv.velocity_weighting(prod)
    assert cv.velocity_weighting(SimpleNamespace(fs=SimpleNamespace(method="dtfe", snapdir=d, prefix="output.")))[0] \
        == "volume"


def t_coarse_catalog_matches():
    """The four analytic voids are found on the 64^3 grid and on its 2x block mean at the same sigma in Mpc, with
    the coarse grid's frame recorded; matched one to one (mutual), R_v within 2%, centres within one coarse cell;
    config untouched afterwards."""
    import config
    from dtfelib import convergence as cv, catalog
    before = {k: getattr(config, k) for k in cv.FRAME_KEYS}
    c = _cats()
    assert {k: getattr(config, k) for k in cv.FRAME_KEYS} == before
    a, b = c["native"], c["coarse"]
    assert int(a["n_grid"]) == 64 and int(b["n_grid"]) == 32 and abs(float(b["cell_mpc"]) - 2.0) < 1e-12
    ra, rb = cv.distinct_rows(a, 64), cv.distinct_rows(b, 32)
    assert ra.size == 4 and rb.size == 4, (ra.size, rb.size)
    m = cv.match(a, ra, 64, b, rb, 32, BOX, SIGMA)
    s = cv.match_summary(m, SIGMA)
    assert s["n_matched"] == 4 and s["n_mutual"] == 4 and s["frac_ref_unmatched"] == 0 and s["frac_test_unmatched"] == 0
    assert abs(s["ratio_pct"][1] - 1.0) < 0.02, s["ratio_pct"]
    assert np.all(m["dist_mpc"] <= np.sqrt(3) * 1.0 + 1e-9), m["dist_mpc"]            # half a coarse cell per axis
    want = np.sort(catalog.r_v_mpc(a)[ra])
    assert np.all(np.diff(want) > 0) and want[0] > 5.0, want                            # four voids, growing with s


def t_match_counts():
    """match on hand-made catalogues: a periodic neighbour across the box face is matched, a void beyond the radius
    is not, a test void nobody claims counts as unmatched; the two R_v and deltas carried; mutual flags."""
    from dtfelib import convergence as cv
    ref = {"coords": np.array([[0, 0, 0], [10, 10, 10], [30, 5, 5]]), "r_v_cells": np.array([2.0, 3.0, 4.0]),
           "cell_mpc": np.float64(1.0), "delta_values": np.array([-0.9, -0.8, -0.7])}
    test = {"coords": np.array([[31, 31, 0], [5, 5, 5], [20, 20, 20]]), "r_v_cells": np.array([1.0, 1.5, 9.0]),
            "cell_mpc": np.float64(2.0), "delta_values": np.array([-0.85, -0.75, -0.5])}
    # test on a 32 grid of the same 64 Mpc box: [31, 31, 0] is (63, 63, 1) Mpc, 2.18 Mpc from ref 0's (0.5, 0.5, 0.5)
    # across the corner; [5, 5, 5] is 0.87 Mpc from ref 1; nothing near ref 2, and test 2 is near no reference void
    m = cv.match(ref, np.arange(3), 64, test, np.arange(3), 32, 64.0, 4.0)
    assert list(m["test"]) == [0, 1, -1], (m["test"], m["dist_mpc"])
    assert abs(m["dist_mpc"][0] - np.sqrt(1.5 ** 2 * 2 + 0.5 ** 2)) < 1e-9 and abs(m["dist_mpc"][1] - np.sqrt(0.75)) < 1e-9
    assert list(m["test_matched"]) == [True, True, False] and m["mutual"][:2].all()
    assert abs(m["r_v_test"][1] / m["r_v_ref"][1] - 1.0) < 1e-12               # 1.5 cells x 2 Mpc = 3 cells x 1 Mpc
    assert abs(m["delta_test"][0] + 0.85) < 1e-12 and np.isnan(m["r_v_test"][2])
    s = cv.match_summary(m, 1.0)
    assert s["n_matched"] == 2 and abs(s["frac_ref_unmatched"] - 1 / 3) < 1e-12 and abs(s["frac_test_unmatched"] - 1 / 3) < 1e-12
    m1 = cv.match(ref, np.arange(3), 64, test, np.arange(3), 32, 64.0, 1.0)     # a smaller radius drops the corner pair
    assert list(m1["test"]) == [-1, 1, -1], m1["test"]


def t_stacked_profile():
    """stacked_profile on the analytic field: the stack of the four voids is underdense at the centre and ends near
    the mean, at both resolutions; the coarse centre within 0.1 of the native."""
    from dtfelib import convergence as cv
    c = _cats()
    a, b = c["native"], c["coarse"]
    ra, rb = cv.distinct_rows(a, 64), cv.distinct_rows(b, 32)
    sa = cv.stacked_profile(c["rho"], a, ra, 64, BOX, [0.0, 100.0], m=100, n_boot=20)
    sb = cv.stacked_profile(cv.block_average(c["rho"], 2), b, rb, 32, BOX, [0.0, 100.0], m=100, n_boot=20)
    # min_count 5 > 4 voids: profiles.stack leaves the bin empty; the count still says 4 (or fewer by the wrap rule)
    assert int(sa["count"][0]) == sa["profiled"] and sa["profiled"] >= 2, (sa["count"], sa["profiled"])
    from dtfelib import profiles, catalog
    prof = profiles.profile_grids({"density": c["rho"], "velocity": None, "single": None},
                                  cv.centres_frac(a, ra, 64), catalog.r_v_mpc(a)[ra], m=100)
    assert np.all(prof["density"][:, 0] < 0.5), prof["density"][:, 0]
    assert sb["profiled"] == sa["profiled"], (sa["profiled"], sb["profiled"])


# ---------------------------------------------------------------- plot_void_convergence.py
def _grid_result():
    from dtfelib import convergence as cv, catalog
    c = _cats()
    a, b = c["native"], c["coarse"]
    ra, rb = cv.distinct_rows(a, 64), cv.distinct_rows(b, 32)
    m = cv.match(a, ra, 64, b, rb, 32, BOX, SIGMA)
    st = cv.stacked_profile(c["rho"], a, ra, 64, BOX, [0.0, 100.0], m=60, n_boot=10)
    stb = cv.stacked_profile(cv.block_average(c["rho"], 2), b, rb, 32, BOX, [0.0, 100.0], m=60, n_boot=10)
    lev0 = {"n_grid": 64, "factor": 1, "r_v": catalog.r_v_mpc(a)[ra], "stack": st, "n_minima": len(a["coords"]),
            "n_resolved": int(a["well_resolved"].sum()), "summary": None, "match": None}
    lev1 = {"n_grid": 32, "factor": 2, "r_v": catalog.r_v_mpc(b)[rb], "stack": stb, "n_minima": len(b["coords"]),
            "n_resolved": int(b["well_resolved"].sum()), "summary": cv.match_summary(m, SIGMA), "match": m}
    est = {"box": BOX, "n": 64, "match": m, "summary": cv.match_summary(m, SIGMA), "r_v_ps": lev0["r_v"],
           "r_v_dtfe": lev1["r_v"], "pos_ps": np.asarray(a["coords"], float)[ra], "snap": 99}
    return {"box": BOX, "n": 64, "sigma": SIGMA, "levels": [lev0, lev1]}, est


def t_figure_and_table():
    """draw() with all three rows (the estimator row from a synthetic 'DTFE' match), and with (b) and (c) missing:
    the PNGs, the not-drawn notes in the summary; the estimator table has one line per reference void."""
    import plot_void_convergence as pc
    grid, est = _grid_result()
    out = Path(_FIGS_TMP) / "conv"
    out.mkdir(parents=True, exist_ok=True)
    args = Namespace(sim="SYN", match_sigma=1.0, rv_range=[2.5, 5.0], crossbox_smooth_mpc=2.1625)
    pc.draw(0.0, grid, est, [], args, SIGMA, out / "full.png")
    pc.draw(1.0, None, None, [], args, SIGMA, out / "empty.png")
    assert (out / "full.png").is_file() and (out / "empty.png").is_file()
    lines = pc.summary_lines(0.0, 99, grid, est, [], SIGMA)
    text = "\n".join(lines)
    assert "(b) not drawn" in text and "(c) PS-DTFE 4 vs DTFE 4 distinct: matched 4" in text, text
    assert "(c) not drawn" in "\n".join(pc.summary_lines(1.0, None, None, None, [], SIGMA))
    pc.estimator_table(est, out / "estimator_voids_099.txt")
    rows = [ln for ln in (out / "estimator_voids_099.txt").read_text().splitlines() if not ln.startswith("#")]
    assert len(rows) == 4 and all(len(r.split()) == 12 for r in rows), rows[:2]


def t_crossbox_and_missing():
    """(b) reads plot_void_crossbox's per-box .npz (a hand-written one) and says when there is none; (c) returns None,
    said, for a snapshot without both estimators' grids (an empty data root)."""
    import plot_void_convergence as pc
    d = Path(_FIGS_TMP) / "xb"
    d.mkdir(parents=True, exist_ok=True)
    args = Namespace(crossbox_dir=d, method="ps", crossbox_smooth_mpc=2.1625)
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert pc.load_crossbox(args) == []
    assert "row (b) not drawn" in buf.getvalue(), buf.getvalue()
    rv = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    np.savez_compressed(d / "crossbox_SYN-1_ps_s2.1625Mpc.npz", sim="SYN-1", sigma_mpc=2.1625, box_mpc=100.0,
                        snaps=np.array([50, 99]), redshift=np.array([1.0, 0.0]), r_v_concat=np.r_[rv, rv],
                        r_v_count=np.array([5, 5]), r_v_pct=np.tile([5.6, 7.0, 8.4], (2, 1)),
                        pz_target=np.array([0.0]), pz_bins=np.linspace(0.1, 3, 30),
                        pz_density_mean=np.ones((1, 30)), pz_density_lo=np.ones((1, 30)), pz_density_hi=np.ones((1, 30)),
                        pz_count=np.array([5]))
    cross = pc.load_crossbox(args)
    assert len(cross) == 1 and cross[0]["sim"] == "SYN-1"
    rows = pc.mass_summary(cross, 0.0)
    assert len(rows) == 1 and rows[0][1] == 99 and np.allclose(rows[0][2], rv)
    grid, est = _grid_result()
    pc.draw(0.0, grid, None, cross, Namespace(sim="SYN", match_sigma=1.0, rv_range=[2.5, 5.0],
                                               crossbox_smooth_mpc=2.1625), SIGMA, d / "b.png")
    assert (d / "b.png").is_file()
    root = Path(_FIGS_TMP) / "emptyroot"
    (root / "SYN").mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        r = pc.estimator_rows(99, Namespace(sim="SYN", data_root=root, match_sigma=1.0))
    assert r is None and "row (c) not drawn" in buf.getvalue(), buf.getvalue()


def t_prefix_set():
    """2026-10-11: a --prefix run's outputs carry its grid set (the default names unchanged), and row (c)'s side of
    --method reads that set (said when it is not there), its table header naming both sets."""
    import plot_void_convergence as pc
    assert pc.set_name("ps", None) == "ps" and pc.set_name("ps", "ps_output") == "ps"
    assert pc.set_name("dtfe", "output.") == "dtfe" and pc.set_name("ps", "ps_mw.") == "ps_mw"
    root = Path(_FIGS_TMP) / "emptyroot_prefix"
    (root / "SYN").mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        r = pc.estimator_rows(99, Namespace(sim="SYN", data_root=root, match_sigma=1.0, method="ps", prefix="ps_mw."))
    assert r is None and "sets ps_mw and dtfe" in buf.getvalue(), buf.getvalue()
    grid, est = _grid_result()
    est = dict(est, sets={"ps": "ps_mw", "dtfe": "dtfe"})
    out = Path(_FIGS_TMP) / "conv_prefix"
    out.mkdir(parents=True, exist_ok=True)
    pc.estimator_table(est, out / "estimator_voids_099_ps_mw.txt")
    assert "grid set ps_mw" in (out / "estimator_voids_099_ps_mw.txt").read_text().splitlines()[0]
    text = "\n".join(pc.summary_lines(0.0, 99, dict(grid, velocity_weighting="volume", velocity_weighting_why="x"),
                                      est, [], SIGMA))
    assert "(c) grid sets: PS-DTFE ps_mw, DTFE dtfe" in text and "volume-weighted block means" in text, text


def t_cli():
    """The launcher's contract (K4): the script's help lists its options."""
    pr = subprocess.run([sys.executable, str(REPO / "python" / "plot" / "plot_void_convergence.py"), "--help"],
                        capture_output=True, text=True, timeout=120)
    assert pr.returncode == 0, pr.stderr[-400:]
    for flag in ("--sim", "--z", "--factors", "--match-sigma", "--crossbox-dir", "--no-velocity", "--smooth-mpc"):
        assert flag in pr.stdout, flag


# ---------------------------------------------------------------- data-backed
def t_tng100_grid():
    """TNG100-3-Dark z = 0 (snapshot 99): the 256^3 block-mean catalogue at the production sigma keeps the distinct
    voids (median R_v ratio within 3% of 1, >= 85% of the native voids matched; 2026-10-11: 0.997, 73 of 77)."""
    from dtfelib import cli
    sim = "TNG100-3-Dark"
    if not (cli.sim_dir(sim) / "snapdir_099").is_dir():
        raise Skip(f"no {sim} snapdir_099 under {cli.DATA_ROOT}")
    if os.environ.get("DTFE_SIM") != sim:
        raise Skip("the TNG100 frame needs DTFE_SIM=TNG100-3-Dark (run the data tier with it set)")
    from dtfelib import pipeline, catalog, convergence as cv
    prod = pipeline.products(99, sim=sim, method="ps")
    try:
        if prod._cached("voids") is None:
            raise Skip("no fresh cached catalogue (plot_void_population.py builds it)")
        cell, box, n, _src = catalog.frame_of(prod)
        sigma = pipeline.smoothing_length_mpc()
        if not cv.coarse_cache_path(prod, 2, box, sigma, n).exists():
            raise Skip("no cached 256^3 block-mean catalogue (plot_void_convergence.py builds it)")
        ref = prod.voids()
        test = cv.coarse_catalog(prod, 2, sigma)
    finally:
        prod.release()
    ra, rb = cv.distinct_rows(ref, n), cv.distinct_rows(test, n // 2)
    s = cv.match_summary(cv.match(ref, ra, n, test, rb, n // 2, box, sigma), sigma)
    assert abs(s["ratio_pct"][1] - 1.0) < 0.03 and s["n_matched"] >= 0.85 * s["n_ref"], s
    print(f"        TNG100 z=0: {s['n_matched']}/{s['n_ref']} matched, R_v ratio {s['ratio_pct'][1]:.3f}")


def main():
    print("=" * 60)
    print(" void convergence: grid, particle mass, estimator")
    print("=" * 60)
    print("synthetic:")
    check("convergence: frame() sets and restores config; block means (scalar, mass-weighted velocity)",
          t_frame_and_blocks)
    check("convergence: the coarse velocity follows the run's weighting (volume = plain block mean; from the run log)",
          t_velocity_weighting)
    check("convergence: analytic voids on 64^3 and its 32^3 block mean matched one to one, R_v within 2%",
          t_coarse_catalog_matches)
    check("convergence: match counts, periodic neighbour, radius, unmatched both ways, mutual", t_match_counts)
    check("convergence: stacked profiles at both resolutions", t_stacked_profile)
    check("plot_void_convergence: three rows, missing rows said, estimator table", t_figure_and_table)
    check("plot_void_convergence: crossbox products read (b), none said; no DTFE grids -> (c) not drawn",
          t_crossbox_and_missing)
    check("plot_void_convergence: a --prefix run's names and row (c) carry its grid set", t_prefix_set)
    check("plot_void_convergence: --help lists the options (launcher contract K4)", t_cli)
    print("data-backed:")
    check("TNG100-3-Dark z=0: 256^3 block-mean catalogue against the native one", t_tng100_grid)
    print("-" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
