#!/usr/bin/env python3
"""Tests for the void-population theory and figures: dtfelib.theory (EH98 sigma(R), D(z), the SvdW /
Vdn size function, the BBKS shapes of Gaussian-field minima), plot_void_population's size function figure and its
R_v-definition fallback, plot_void_crossbox's shape row, and analyze.py's void-script registration and void-gate
refusal.

Two tiers:
  * SYNTHETIC (always run, CI): known values and limits of the theory, the figures from hand-made statistics.
  * DATA-BACKED (the cached TNG100-3-Dark z = 0 catalogue under DTFE_DATA_ROOT): the size function's sample and ratio
    on real voids; skipped with a note when DTFE_DATA_ROOT is unset or the data or the cache is not there.

Usage:  python3 tests/py_population_test.py            (exit 0 = pass)
"""

import io
import math
import os
import shutil
import sys
import tempfile
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from _harness import REPO, Skip, check, data_root, setup, summary

_FIGS_TMP = setup("population_figs_")
os.environ.setdefault("DTFE_SIM", "TNG50-4-Dark")    # the box these synthetic tests stand for; a data-tier run sets its own


_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------- theory
def t_sigma8_growth():
    """sigma(8/h Mpc, z=0) = config's sigma_8; D(0) = 1; D(z)(1+z) flat in the matter era; D(1) of Planck 2015."""
    import config
    from dtfelib import theory
    s8 = theory.sigma_tophat(8.0 / config.HUBBLE_H)
    assert abs(s8 / config.COSMOLOGY["sigma8"] - 1.0) < 1e-4, s8
    assert abs(float(theory.growth(0.0)) - 1.0) < 1e-12
    d50, d100 = float(theory.growth(50.0)), float(theory.growth(100.0))
    assert abs(d50 * 51.0 / (d100 * 101.0) - 1.0) < 2e-3, (d50 * 51, d100 * 101)
    assert 0.60 < float(theory.growth(1.0)) < 0.62, theory.growth(1.0)        # Om 0.31 flat LCDM: D(1) ~ 0.61
    # sigma(R) falls with R and scales with D(z)
    s = theory.sigma_tophat(np.array([1.0, 5.0, 20.0]))
    assert np.all(np.diff(s) < 0), s
    assert np.allclose(theory.sigma_tophat(np.array([1.0, 5.0]), z=1.0), s[:2] * float(theory.growth(1.0)), rtol=1e-12)


def t_vdn_limits():
    """The two-barrier multiplicity: equal to its one-barrier limit at small x, integral over ln sigma =
    delta_c / (delta_c + |delta_v|) (gambler's ruin); Vdn's volume fraction is that integral over the sigma range of
    the radii; SvdW = Vdn x 1.71^3; n(>R) is dn/dlnR integrated."""
    from dtfelib import theory as T
    dv, dc = abs(T.DELTA_V_LINEAR), T.DELTA_C
    D = dv / (dc + dv)
    x = np.array([0.1001, 0.15, 0.2])
    sig = x * dv / D
    one = np.sqrt(2.0 / np.pi) * (dv / sig) * np.exp(-0.5 * (dv / sig) ** 2)
    assert np.allclose(T.svdw_multiplicity(sig), one, rtol=1e-6), T.svdw_multiplicity(sig) / one
    s = np.geomspace(1e-2, 1e3, 20000)
    tot = _trapz(T.svdw_multiplicity(s), np.log(s))
    assert abs(tot - dc / (dc + dv)) < 1e-6, (tot, dc / (dc + dv))
    one_tot = _trapz(T.svdw_multiplicity(s, delta_c=np.inf), np.log(s))      # one barrier: every walk, but for the
    want_one = math.erf(dv / (math.sqrt(2.0) * s[0])) - math.erf(dv / (math.sqrt(2.0) * s[-1]))   # sigma range's ends
    assert abs(one_tot - want_one) < 1e-6, (one_tot, want_one)
    r = np.geomspace(0.5, 80.0, 3000)
    vfrac = _trapz(4.0 / 3.0 * np.pi * r ** 3 * T.void_size_function(r, 0.0), np.log(r))
    sl = T.sigma_tophat(r / T.linear_radius_ratio())
    s_rng = np.geomspace(sl.min(), sl.max(), 4000)
    want = _trapz(T.svdw_multiplicity(s_rng), np.log(s_rng))
    assert abs(vfrac / want - 1.0) < 2e-3 and vfrac < dc / (dc + dv), (vfrac, want)
    assert abs(T.linear_radius_ratio() - 1.70998) < 1e-4, T.linear_radius_ratio()
    assert np.allclose(T.void_size_function(r[::300], 0.0, "svdw"),
                       T.void_size_function(r[::300], 0.0) * T.linear_radius_ratio() ** 3, rtol=1e-10)
    r1, r2 = 3.0, 6.0
    g = np.geomspace(r1, r2, 2001)
    diff = T.void_cumulative(r1) - T.void_cumulative(r2)
    assert abs(diff / _trapz(T.void_size_function(g), np.log(g)) - 1.0) < 1e-3, diff
    # fewer large voids at high z (sigma smaller): dn/dlnR(10 Mpc) falls with z
    assert T.void_size_function(10.0, 1.0) < T.void_size_function(10.0, 0.0)


def t_bbks_shapes():
    """BBKS: f(x) -> x^3 - 3x at large x; P(e, p | x) integrates to 1 with the analytic normalisation (the test of
    f, eq A15, against eq 7.6); all minima's shapes do not depend on gamma; higher peaks are rounder (median e
    falls with nu); gamma of the smoothed spectrum is in (0, 1)."""
    from dtfelib import theory as T
    assert abs(float(T.bbks_f(6.0)) / (6.0 ** 3 - 18.0) - 1.0) < 1e-3, T.bbks_f(6.0)
    for x in (1.5, 3.0):
        e, p, pdf = T.bbks_ep_pdf(x)
        tot = pdf.sum() * (e[1] - e[0]) * (p[1] - p[0])
        assert abs(tot - 1.0) < 2e-3, (x, tot)
    a, b = T.minima_shape_percentiles(None, 0.3), T.minima_shape_percentiles(None, 0.8)
    assert np.allclose(a["e"], b["e"]) and np.allclose(a["p"], b["p"])
    assert 0.15 < a["e"][1] < 0.25 and 0.0 < a["p"][1] < 0.05, a        # e ~ 0.19, p small and positive
    meds = [T.minima_shape_percentiles(np.array([nu]), 0.6)["e"][1] for nu in (0.5, 2.0, 4.0)]
    assert meds[0] > meds[1] > meds[2], meds
    g, rstar = T.bbks_gamma(2.1625)
    assert 0.0 < g < 1.0 and rstar > 0, (g, rstar)
    pdf = T.minima_shape_pdf(np.array([1.0, 2.0]), g)
    assert abs(pdf["pdf_e"].sum() * (pdf["e"][1] - pdf["e"][0]) - 1.0) < 1e-9


# ---------------------------------------------------------------- the population script
def _cat(n=40, seed=3, with_delta08=False):
    """A hand-made catalogue: n minima, all resolved and BBKS-valid, R_v crossing in cells, deep ones first."""
    rng = np.random.default_rng(seed)
    coords = rng.integers(0, 512, size=(n, 3))
    e = rng.uniform(0.05, 0.3, n)
    cat = {"coords": coords, "bbks_params": np.c_[e, rng.uniform(-0.5, 0.5, n) * e],
           "well_resolved": np.ones(n, bool), "delta_values": np.sort(rng.uniform(-0.9, -0.2, n)),
           "semi_axes": np.tile([4.0, 3.0, 2.0], (n, 1)).astype(np.float32),
           "r_v_cells": np.full(n, 0.5), "cell_mpc": np.float64(1.0)}
    if with_delta08:
        cat["r_v_delta08_cells"] = np.full(n, 0.4)
    return cat


def t_size_rv_fallback():
    """r_v_definition: 'delta08' falls back to 'crossing' for a catalogue (or a catalog module) without it, and is
    used when there; snapshot_stats' size-function sample and n_rv_all follow."""
    import plot_void_population as pop
    from dtfelib import catalog
    cat = _cat()
    r, d = pop.r_v_definition(cat, "delta08")
    assert d == "crossing" and np.allclose(r, catalog.r_v_mpc(cat)), d
    st = pop.snapshot_stats(cat, 512, keep_overlaps=True, size_rv="delta08")
    assert st["rv_sf_definition"] == "crossing" and st["n_rv_all"] == 40 and st["_r_v_sf"].size == 40, st["n_sf"]
    assert st["_delta_valid"].size == st["n_bbks"]
    if "delta08" in getattr(catalog, "R_V_DEFINITIONS", ()):        # the definition exists: the key is read
        cat8 = _cat(with_delta08=True)
        r8, d8 = pop.r_v_definition(cat8, "delta08")
        assert d8 == "delta08" and np.allclose(r8[np.isfinite(r8)], 0.4), (d8, r8[:3])


def t_ratio_at():
    """ratio_at: the counts within exp(+-0.25) of R0 over V dlnR, divided by the theory averaged over that window."""
    import plot_void_population as pop
    from dtfelib import theory
    r = np.r_[np.full(30, 5.0), np.full(70, 50.0)]
    ratio, err, n = pop.ratio_at(r, 100.0, 5.0, 0.0)
    th = np.mean(theory.void_size_function(5.0 * np.exp(np.linspace(-0.25, 0.25, 41)), 0.0))
    assert n == 30 and abs(ratio - 30 / (1e6 * 0.5) / th) < 1e-12 and abs(err / ratio - 1 / math.sqrt(30)) < 1e-12
    assert np.isnan(pop.ratio_at(r, 100.0, 5.0, np.nan)[0])            # no redshift: no theory


def t_size_function_figure():
    """size_function_figure on hand-made statistics at two snapshots: the PNG, the npz arrays (theory per snapshot,
    the ratio at the median), and the note when the asked R_v definition is missing."""
    import plot_void_population as pop
    out = Path(_FIGS_TMP) / "sf"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    stats = {}
    for s in (50, 99):
        st = pop.snapshot_stats(_cat(200, seed=s), 512, keep_overlaps=True)
        st["_r_v_sf"] = rng.lognormal(math.log(4.0), 0.3, 300)
        stats[s] = st
    args = Namespace(size_rv="delta08", sim="SYN")
    buf = io.StringIO()
    with redirect_stdout(buf):
        sf = pop.size_function_figure(stats, [50, 99], {50: 1.0, 99: 0.0}, {50: 100.0, 99: 100.0}, args, ["ps"],
                                      out / "population_size_function.png")
    assert (out / "population_size_function.png").is_file()
    assert "'delta08' is not in these catalogues" in buf.getvalue(), buf.getvalue()[:300]
    assert sf["size_function_rv"] == "crossing" and sf["size_function_model"] == "vdn"
    for s in (50, 99):
        assert sf[f"size_function_{s}_theory_dn_dlnr"].shape == sf["size_function_theory_r_mpc"].shape
        assert np.isfinite(sf[f"size_function_{s}_ratio_at_median"]) and sf[f"size_function_{s}_ratio_at_median"] > 0
    np.savez_compressed(out / "sf.npz", **sf)                               # every value is npz-able


def t_size_smooth():
    """--size-smooth: smoothing_cells sets the size function's catalogue smoothing (its own cache
    namespace, the production footprint) and restores the process's even on an exception; the names are untagged at
    the default and '_ss<cells>' otherwise; snapshot_stats reports the fraction of the distinct voids whose
    size-function radius exists; the option is in the script's help."""
    import subprocess
    import config
    import plot_void_population as pop
    from dtfelib import pipeline
    saved = (config.SMOOTHING_SIGMA_CELLS, config.FOOTPRINT_SIZE, config.SMOOTHING_SIGMA_MPC)
    ns0 = pipeline._param_hash()
    try:
        with pop.smoothing_cells(5.0):
            assert config.SMOOTHING_SIGMA_CELLS == 5.0 and config.FOOTPRINT_SIZE == int(config.PRODUCTION_SMOOTHING[1])
            assert pipeline._param_hash() != ns0 and "_s5_" in pipeline._param_hash(), pipeline._param_hash()
            raise KeyError("inside")
    except KeyError:
        pass
    assert (config.SMOOTHING_SIGMA_CELLS, config.FOOTPRINT_SIZE, config.SMOOTHING_SIGMA_MPC) == saved
    assert pipeline._param_hash() == ns0
    assert pop.size_smooth_tag(pop.SIZE_SMOOTH_CELLS) == "" and pop.size_smooth_tag(3) == "_ss3"
    assert pop.size_smooth_tag(0) == "_ss0"
    assert pop.size_smooth_active(5.0) == (float(config.SMOOTHING_SIGMA_CELLS) != 5.0)
    assert not pop.size_smooth_active(0) and not pop.size_smooth_active(config.SMOOTHING_SIGMA_CELLS)
    cat = _cat(with_delta08=True)
    cat["r_v_delta08_cells"][::4] = np.nan                    # a quarter without an enclosed -0.8 sphere
    st = pop.snapshot_stats(cat, 512, keep_overlaps=True, size_rv="delta08")
    if st["rv_sf_definition"] == "delta08":
        assert abs(st["frac_sf_defined"] - 0.75) < 1e-12, st["frac_sf_defined"]
    pr = subprocess.run([sys.executable, str(REPO / "python" / "plot" / "plot_void_population.py"), "--help"],
                        capture_output=True, text=True, timeout=120)
    assert pr.returncode == 0 and "--size-smooth" in pr.stdout, pr.stdout[-400:]


# ---------------------------------------------------------------- the cross-box shape row
def _crossbox_result(sim, box, seed):
    """A worker result as measure() writes it, two snapshots, the shape keys included."""
    import plot_void_crossbox as xb
    from dtfelib import profiles, theory
    rng = np.random.default_rng(seed)
    nb = len(profiles.DEFAULT_BINS)
    snaps, zs = np.array([50, 99]), np.array([1.0, 0.0])
    e = [rng.uniform(0.05, 0.35, 120) for _ in snaps]
    p = [rng.uniform(-0.5, 0.5, 120) * x for x in e]
    dmin = [rng.uniform(-0.9, -0.3, 120) for _ in snaps]
    rv = [rng.lognormal(1.2, 0.3, 50) for _ in snaps]
    grf = [xb.grf_shapes(d, z, 2.1625) for d, z in zip(dmin, zs)]
    allm = theory.minima_shape_percentiles(None)
    res = {"sim": sim, "method": "ps", "sigma_mpc": 2.1625, "sigma_cells": 10.0, "footprint": 11,
           "cache_namespace": "x", "box_mpc": box, "cell_mpc": box / 512, "n_grid": 512,
           "refused": np.zeros(0, int), "snaps": snaps, "redshift": zs, "box_per_snap": np.full(2, box),
           "r_v_pct": np.array([np.percentile(x, (16, 50, 84)) for x in rv]), "r_v_concat": np.concatenate(rv),
           "r_v_count": np.array([50, 50]), "n_total": np.array([600, 500]), "n_resolved": np.array([200, 100]),
           "n_deep": np.array([50, 40]), "n_counted": np.array([50, 50]), "n_no_rv": np.array([3, 4]),
           "n_overlapping": np.array([10, 9]), "n_rv_all": np.array([550, 480]), "n_bbks": np.array([120, 120]),
           "e_med": np.array([xb.median_ci(x, 50, 1) for x in e]), "p_med": np.array([xb.median_ci(x, 50, 2) for x in p]),
           "e_pct": np.array([np.percentile(x, (16, 50, 84)) for x in e]),
           "p_pct": np.array([np.percentile(x, (16, 50, 84)) for x in p]),
           "grf_e_pct": np.array([g[0] for g in grf]), "grf_p_pct": np.array([g[1] for g in grf]),
           "grf_all_e_pct": allm["e"], "grf_all_p_pct": allm["p"], "grf_gamma": theory.bbks_gamma(2.1625)[0],
           "e_concat": np.concatenate(e), "p_concat": np.concatenate(p), "delta_valid_concat": np.concatenate(dmin),
           "e_count": np.array([120, 120]), "pz_target": np.array([0.0]), "pz_rv_range_mpc": np.array([5.0, 10.0]),
           "pz_rv_range_sigma": np.array([2.5, 5.0]), "pz_snap": np.array([99]), "pz_redshift": np.array([0.0]),
           "pz_count": np.array([0]), "pz_profiled": np.array([0]), "pz_dropped": np.array([0]),
           "pz_bins": np.asarray(profiles.DEFAULT_BINS, float)}
    for k in ("density", "v_r", "single"):
        for part in ("mean", "lo", "hi"):
            res[f"pz_{k}_{part}"] = np.full((1, nb), np.nan)
    return res


def t_crossbox_shapes():
    """median_ci brackets the median; grf_shapes is NaN without a redshift and rounder for deeper minima; the figure
    from two boxes' results gets its shape row and the table its e, p columns; a legacy result (no shape
    keys) still draws."""
    import plot_void_crossbox as xb
    x = np.random.default_rng(0).normal(0.2, 0.05, 400)
    m, lo, hi = xb.median_ci(x, 200)
    assert lo < m < hi and hi - lo < 0.02, (m, lo, hi)
    assert np.isnan(xb.grf_shapes(np.array([-0.5]), np.nan, 2.0)[0]).all()
    shallow, deep = xb.grf_shapes(np.full(5, -0.2), 0.0, 2.1625), xb.grf_shapes(np.full(5, -3.0), 0.0, 2.1625)
    assert deep[0][1] < shallow[0][1], (deep[0], shallow[0])
    out = Path(_FIGS_TMP) / "xb"
    out.mkdir(parents=True, exist_ok=True)
    results = [_crossbox_result("TNG100-3-Dark", 110.7, 1), _crossbox_result("TNG300-3-Dark", 302.6, 2)]
    args = Namespace(smooth_mpc=2.1625, method="ps")
    path = xb.draw(results, 0.0, args, out)
    assert path.is_file(), path
    lines = xb.summary_lines(results)
    assert any("GRF e" in ln for ln in lines) and any("Gaussian field, all minima" in ln for ln in lines), lines[:3]
    old = [{k: v for k, v in r.items() if k not in ("e_med", "n_rv_all")} for r in results]
    assert xb.draw(old, 0.0, args, out).is_file()
    assert not any("GRF e" in ln for ln in xb.summary_lines(old))


# ---------------------------------------------------------------- analyze.py
def t_analyze_void_scripts():
    """analyze.py registers the void scripts in VOID_SCRIPTS (PLOT_SCRIPTS, the launcher's checklist, unchanged) and
    run_scripts skips a script that is not on disk with a note instead of failing."""
    import analyze
    for name in ("plot_void_population.py", "plot_void_profile_evolution.py", "plot_void_environment.py",
                 "plot_void_fate.py"):
        assert name in analyze.VOID_SCRIPTS, name
    assert not set(analyze.VOID_SCRIPTS) & set(analyze.PLOT_SCRIPTS)
    root = Path(tempfile.mkdtemp(prefix="analyze_voids_"))
    try:
        (root / "plot").mkdir()
        (root / "plot" / "ok.py").write_text("import sys; sys.exit(0)\n")
        results = {}
        buf = io.StringIO()
        with redirect_stdout(buf):
            skipped = analyze.run_scripts(["ok.py", "plot_not_written_yet.py"], root, results)
        assert skipped == ["plot_not_written_yet.py"] and list(results) == ["ok.py"] and results["ok.py"][0] == 0
        assert "not in python/plot (yet): skipped" in buf.getvalue()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def t_analyze_gate_refusal():
    """'analyze.py plot' under DTFE_VOID_GATE=none at the production smoothing is refused for the GATE, by name
    (smoothing_tag() carries 'gatenone'; DTFE_SIGMA_MPC, which is not set, is not blamed)."""
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "DTFE_SIGMA_MPC"}
    env.update(DTFE_SIM="TNG50-3-Dark", DTFE_VOID_GATE="none", DTFE_CACHE_DIR=tempfile.mkdtemp(prefix="analyze_gate_cache_"))
    try:
        pr = subprocess.run([sys.executable, str(REPO / "python" / "analyze.py"), "plot"], env=env,
                            capture_output=True, text=True, timeout=120)
    finally:
        shutil.rmtree(env["DTFE_CACHE_DIR"], ignore_errors=True)
    assert pr.returncode != 0, (pr.returncode, pr.stdout[-400:])
    assert "refused: DTFE_VOID_GATE='none'" in pr.stderr and "DTFE_SIGMA_MPC" not in pr.stderr, pr.stderr[-400:]


# ---------------------------------------------------------------- data-backed
def t_tng100_size_function():
    """TNG100-3-Dark's z = 0 (snapshot 99) catalogue from the production cache: the size function's sample is
    the distinct voids, and measured / Vdn at its median R_v is finite and positive."""
    from dtfelib import cli
    sim = "TNG100-3-Dark"
    if not (cli.sim_dir(sim, data_root()) / "snapdir_099").is_dir():
        raise Skip(f"no {sim} snapdir_099 under {data_root()}")
    if os.environ.get("DTFE_SIM") != sim:
        raise Skip("the TNG100 frame needs DTFE_SIM=TNG100-3-Dark (run the data tier with it set)")
    import plot_void_population as pop
    from dtfelib import pipeline, catalog
    prod = pipeline.products(99, sim=sim, method="ps")
    try:
        if prod._cached("voids") is None:
            raise Skip("no fresh cached catalogue (build it with plot_void_population.py first)")
        cell, box, n, _src = catalog.frame_of(prod)
        cat = prod.voids()
    finally:
        prod.release()
    st = pop.snapshot_stats(cat, n)
    rv = st["_r_v_sf"]
    assert rv.size >= 10, rv.size                  # delta08 at z = 0: 25 distinct voids
    ratio, err, cnt = pop.ratio_at(rv, box, float(np.median(rv)), 0.0)
    assert np.isfinite(ratio) and ratio > 0 and cnt > 0, (ratio, cnt)
    print(f"        TNG100 z=0: {rv.size} distinct voids ({st['rv_sf_definition']}), median R_v "
          f"{np.median(rv):.3g} Mpc, measured/Vdn {ratio:.3f} +- {err:.3f}")


def t_tng100_size_smooth():
    """The --size-smooth catalogue of TNG100-3-Dark at z = 0 and 1 (snapshots 99, 50; 5 cells, from the cache that
    plot_void_population.py builds): the delta08 radius exists for more of the distinct voids than at the production
    10 cells (31% at z = 0), and measured / Vdn at its median is finite."""
    from dtfelib import cli
    sim = "TNG100-3-Dark"
    if not (cli.sim_dir(sim, data_root()) / "snapdir_099").is_dir():
        raise Skip(f"no {sim} snapdir_099 under {data_root()}")
    if os.environ.get("DTFE_SIM") != sim:
        raise Skip("the TNG100 frame needs DTFE_SIM=TNG100-3-Dark (run the data tier with it set)")
    import plot_void_population as pop
    from dtfelib import pipeline, catalog
    for snap, z in ((99, 0.0), (50, 1.0)):
        with pop.smoothing_cells(pop.SIZE_SMOOTH_CELLS):
            prod = pipeline.products(snap, sim=sim, method="ps")
            try:
                if prod._cached("voids") is None:
                    raise Skip("no cached 5-cell catalogue (plot_void_population.py builds it)")
                cell, box, n, _src = catalog.frame_of(prod)
                cat = prod.voids()
            finally:
                prod.release()
            st = pop.snapshot_stats(cat, n)
        rv = st["_r_v_sf"]
        assert st["rv_sf_definition"] == "delta08" and rv.size >= 10, (st["rv_sf_definition"], rv.size)
        if snap == 99:
            assert st["frac_sf_defined"] > 0.312, st["frac_sf_defined"]
        ratio, err, cnt = pop.ratio_at(rv, box, float(np.median(rv)), z)
        assert np.isfinite(ratio) and ratio > 0 and cnt > 0, (ratio, cnt)
        print(f"        TNG100 z={z:g} at {pop.SIZE_SMOOTH_CELLS:g} cells: delta08 defined for "
              f"{st['frac_sf_defined']:.1%} of {st['n_counted']} distinct, {rv.size} in the sample, median "
              f"{np.median(rv):.3g} Mpc, measured/Vdn {ratio:.3f} +- {err:.3f}")


def main():
    print("=" * 60)
    print(" void population: theory, size function, cross-box shapes, analyze.py")
    print("=" * 60)
    print("synthetic:")
    check("theory: sigma_8 reproduced, D(0) = 1, D(z)(1+z) flat in the matter era, sigma(R) x D(z)", t_sigma8_growth)
    check("theory: SvdW/Vdn multiplicity limits (one barrier, delta_c/(delta_c+|delta_v|)), Vdn volume fraction, "
          "SvdW = Vdn x 1.71^3, n(>R) = integral of dn/dlnR", t_vdn_limits)
    check("theory: BBKS f(x) asymptote, P(e,p|x) normalised, all-minima shapes gamma-free, higher minima rounder",
          t_bbks_shapes)
    check("population: R_v definition fallback (delta08 -> crossing), size-function sample, n_rv_all", t_size_rv_fallback)
    check("population: ratio_at = window counts / window-averaged theory", t_ratio_at)
    check("population: size function figure + npz arrays + the missing-definition note", t_size_function_figure)
    check("population: --size-smooth window restores the smoothing, names tagged, fraction defined, in --help",
          t_size_smooth)
    check("crossbox: median_ci, grf_shapes, the shape row and table columns, an old .npz still drawn", t_crossbox_shapes)
    check("analyze.py: VOID_SCRIPTS registered, a missing script skipped with a note", t_analyze_void_scripts)
    check("analyze.py: DTFE_VOID_GATE=none refuses 'plot' for the gate, not the smoothing", t_analyze_gate_refusal)
    print("data-backed:")
    check("TNG100-3-Dark z=0: size-function sample and measured/Vdn at the median R_v", t_tng100_size_function)
    check("TNG100-3-Dark z=0, 1: the --size-smooth catalogue's delta08 fraction and measured/Vdn", t_tng100_size_smooth)
    print("-" * 60)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
