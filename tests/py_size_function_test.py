#!/usr/bin/env python3
"""Tests for the size function's raw-density radius: dtfelib.raw_radius (the enclosed -0.8 radius on
the RAW rho/rho_bar about the catalogue's centres, its cache) and plot_void_population's '--size-rv delta08raw'.

Two tiers:
  * SYNTHETIC (always run, CI): the size function at fixed R (--size-fixed-r: range, ratio, bootstrap, verdict,
    figure, gate refusal) on populations drawn from Vdn; an analytic top-hat void (Delta(<R) = -0.9 inside R0: the
    -0.8 radius is R0 (0.9/0.8)^(1/3)), the periodic wrap, a wisp on the centre, never-below and capped voids, the outward blocks,
    the cache (round trip, keyed on the catalogue's minima, the grid stamp and the settings; only missing rows
    measured), snapshot_stats / size_function_figure with the raw radii, --help.
  * DATA-BACKED (TNG100-3-Dark z = 0's grids and production catalogue under DTFE_DATA_ROOT): the raw radius is defined
    for most distinct voids and is not the smoothing length; skipped with a note when DTFE_DATA_ROOT is unset or the
    data or the catalogue is not there.

Usage:  python3 tests/py_size_function_test.py            (exit 0 = pass)
"""

import io
import shutil
import sys
import tempfile
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import os
from _harness import FAIL, PASS, REPO, SKIP, Skip, check, data_root, setup

_FIGS_TMP = setup("size_function_figs_")
os.environ.setdefault("DTFE_SIM", "TNG50-4-Dark")    # the box these synthetic tests stand for; a data-tier run sets its own


N = 64
R0 = 8.0
R_EXACT = R0 * (0.9 / 0.8) ** (1.0 / 3.0)        # Delta(<R) = -0.9 R0^3 / R^3 outside R0 reaches -0.8 here


def _tophat(centres, r0=R0, inside=0.1, n=N):
    """rho/rho_bar on a periodic n^3 grid: 'inside' within r0 cells of each centre's cell CENTRE (periodic), 1
    elsewhere."""
    g = np.ones((n, n, n), np.float32)
    i = np.arange(n, dtype=np.float64)
    for c in np.asarray(centres, dtype=np.float64).reshape(-1, 3):
        d = [np.minimum(np.abs(i - c[k]), n - np.abs(i - c[k])) for k in range(3)]
        r = np.sqrt(d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2)
        g[r < r0] = inside
    return g


def t_analytic():
    """A top-hat void: the radius within a quarter cell of R0 (0.9/0.8)^(1/3) (the edge is trilinear-blurred over
    one cell), the same at the periodic corner as in the middle, the same for any outward block size; a uniform
    field never goes below -0.8 (NaN, never_below); a reach inside the void gives NaN, capped."""
    from dtfelib import raw_radius as rr
    g = _tophat([[20, 30, 40], [0, 0, 0]])
    out = rr.raw_radii_cells(g, [[20, 30, 40], [0, 0, 0]], reach_cells=28)
    assert np.all(np.abs(out["r_cells"] - R_EXACT) < 0.25), (out["r_cells"], R_EXACT)
    assert abs(out["r_cells"][0] - out["r_cells"][1]) < 1e-9, out["r_cells"]        # the wrap: the corner is no different
    assert not out["never_below"].any() and not out["capped"].any()
    for blk in (1, 5, 1000):
        assert np.allclose(rr.raw_radii_cells(g, [[20, 30, 40]], reach_cells=28, block=blk)["r_cells"],
                           out["r_cells"][0], rtol=0, atol=1e-12), blk
    flat = rr.raw_radii_cells(np.ones((N, N, N), np.float32), [[5, 5, 5]], reach_cells=20)
    assert np.isnan(flat["r_cells"][0]) and flat["never_below"][0] and not flat["capped"][0], flat
    cap = rr.raw_radii_cells(g, [[20, 30, 40]], reach_cells=5.0)
    assert np.isnan(cap["r_cells"][0]) and cap["capped"][0] and not cap["never_below"][0], cap
    assert abs(cap["min_enclosed"][0] + 0.9) < 1e-6, cap["min_enclosed"]


def t_centre_wisp():
    """A wisp ON the centre cell (rho/rho_bar 3: the raw centre above -0.8) does not undefine the void: the
    enclosed contrast dips below -0.8 a cell out and the radius is the void's (the rule: first up-crossing after
    the first point below, not 'the centre must be below' as the smoothed delta08, which would give NaN)."""
    from dtfelib import raw_radius as rr
    g = _tophat([[32, 32, 32]])
    g[32, 32, 32] = 3.0
    out = rr.raw_radii_cells(g, [[32, 32, 32]], reach_cells=28)
    assert np.isfinite(out["r_cells"][0]) and abs(out["r_cells"][0] - R_EXACT) < 0.4, out


def t_not_smoothing():
    """The point of the raw radius: on the SAME void it does not depend on the smoothing that found the centre.
    The smoothed delta08 (pipeline.void_radii_cells on the field smoothed at sigma) of a top-hat void of R0 = 8
    cells moves with sigma; the raw one is computed without sigma at all, so only the centre can move it: a centre
    one cell off changes it by less than a cell."""
    from dtfelib import raw_radius as rr
    from dtfelib import pipeline
    from dtfelib import fields as dtfe
    g = _tophat([[32, 32, 32]])
    d08 = []
    for sig in (1.0, 3.0):
        ds = dtfe.smooth_field(g - 1.0, sigma=sig)
        d08.append(float(pipeline.void_radii_cells(ds, [[32, 32, 32]], sig)["r_v_delta08_cells"][0]))
    raw = rr.raw_radii_cells(g, [[32, 32, 32], [33, 32, 32]], reach_cells=28)["r_cells"]
    assert abs(raw[0] - raw[1]) < 1.0 and abs(raw[0] - R_EXACT) < 0.25, raw
    assert abs(d08[1] - d08[0]) > 0.3, d08                                  # the smoothed one does move


class _FakeProducts:
    """A SnapshotProducts stand-in: a periodic grid, a header frame of config.CELL_SIZE, a cache path, a stamp."""
    def __init__(self, grid, cache_dir, stamp=(1, 2)):
        import config
        self.grid, self.reads = grid, 0
        self.sim, self.snapshot = "SYN", 99
        self.fs = SimpleNamespace(grid_n=grid.shape[0], meta=SimpleNamespace(box_mpc=grid.shape[0] * config.CELL_SIZE),
                                  density=self._density)
        self._dir, self._stamp = Path(cache_dir), np.array(stamp, np.int64)

    def _density(self, units="mean"):
        self.reads += 1
        return self.grid

    def _cpath(self, name):
        return self._dir / f"SYN_ps_099_{name}.npz"

    def _grid_stamp(self):
        return self._stamp


def t_cache():
    """raw_radii: only the asked rows measured (default the well-resolved ones), the cache reused without a grid
    read, extended by the rows asked later, and re-measured when the minima (the catalogue's identity), the grid
    stamp or the settings change; written whole (no temp file left)."""
    import config
    from dtfelib import raw_radius as rr
    d = Path(tempfile.mkdtemp(prefix="rawr_cache_"))
    try:
        g = _tophat([[20, 20, 20], [44, 44, 44]])
        cat = {"coords": np.array([[20, 20, 20], [44, 44, 44], [5, 50, 5]]), "well_resolved": np.array([True, False, True])}
        p = _FakeProducts(g, d)
        buf = io.StringIO()
        with redirect_stdout(buf):
            a = rr.raw_radii(p, cat)
        assert p.reads == 1 and a["measured"].tolist() == [True, False, True], a["measured"]
        assert abs(a["r_cells"][0] - R_EXACT) < 0.25 and np.isnan(a["r_cells"][1]) and a["never_below"][2], a
        assert np.allclose(a["r_mpc"], a["r_cells"] * config.CELL_SIZE, equal_nan=True)
        assert sorted(q.name for q in d.iterdir()) == ["SYN_ps_099_voids_raw08.npz"], list(d.iterdir())
        b = rr.raw_radii(p, cat)
        assert p.reads == 1 and np.array_equal(a["r_cells"], b["r_cells"], equal_nan=True)     # from the cache
        c = rr.raw_radii(p, cat, rows=np.array([1]))                                     # a new row: measured, added
        assert p.reads == 2 and c["measured"].all() and abs(c["r_cells"][1] - R_EXACT) < 0.25, c
        rr.raw_radii(p, cat, rows=np.array([True, True, True]))
        assert p.reads == 2
        cat2 = dict(cat, coords=cat["coords"][[1, 0, 2]])                                # other minima: re-measured
        with redirect_stdout(buf):
            e = rr.raw_radii(p, cat2)
        assert p.reads == 3 and e["measured"].tolist() == [True, False, True] and "re-measuring" in buf.getvalue()
        p2 = _FakeProducts(g, d, stamp=(1, 3))                                           # the grid rewritten
        with redirect_stdout(buf):
            rr.raw_radii(p2, cat2)
        assert p2.reads == 1
        saved = rr.RAW_DIRS
        try:
            rr.RAW_DIRS = 64                                                             # other settings
            with redirect_stdout(buf):
                rr.raw_radii(p2, cat2)
            assert p2.reads == 2
        finally:
            rr.RAW_DIRS = saved
        assert not [q for q in d.iterdir() if "tmp" in q.name]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_held_grid():
    """raw_radii measures on the density grid the products already hold (a fresh voids() build's
    density_raw) instead of reading it again, with the same radii; a products without it still reads once, and the
    read grid is not left in the products (no _ram entry made)."""
    from dtfelib import raw_radius as rr
    d = Path(tempfile.mkdtemp(prefix="rawr_held_"))
    try:
        g = _tophat([[20, 20, 20], [44, 44, 44]])
        cat = {"coords": np.array([[20, 20, 20], [44, 44, 44]]), "well_resolved": np.array([True, True])}
        held = _FakeProducts(g, d / "a")
        held._ram = {"density_raw": g}
        (d / "a").mkdir()
        a = rr.raw_radii(held, cat)
        assert held.reads == 0, f"the held grid was read again ({held.reads} reads)"
        cold = _FakeProducts(g, d / "b")
        cold._ram = {}
        (d / "b").mkdir()
        b = rr.raw_radii(cold, cat)
        assert cold.reads == 1 and cold._ram == {}, (cold.reads, list(cold._ram))
        assert np.array_equal(a["r_cells"], b["r_cells"], equal_nan=True) and abs(a["r_cells"][0] - R_EXACT) < 0.25
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _bowl(centre, r0=R0, n=N):
    """A void whose raw density RISES from its centre (0.05 + 0.15 (r/r0)^2 inside r0, 1.2 outside; periodic about
    the cell centre 'centre'): a real void's profile, so the light smoothing's minimum is its centre (a flat top-hat
    has no minimum to find: every interior cell ties)."""
    i = np.arange(n, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    d = [np.minimum(np.abs(i - c[k]), n - np.abs(i - c[k])) for k in range(3)]
    r = np.sqrt(d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2)
    return np.where(r < r0, 0.05 + 0.15 * (r / r0) ** 2, 1.2).astype(np.float32)


def t_recentre():
    """raw_radius RE-CENTRING: a catalogue centre 5-6 cells off a void's raw centre gives a raw -0.8
    radius well below the void's (the off-centre sphere takes in the wall early); re-centred, the centre walks back to
    within a cell of the raw minimum (also across the periodic edge) and the radius is the one about the true centre.
    A centre already on the minimum does not move."""
    from dtfelib import raw_radius as rr
    true = np.array([62, 20, 30])                          # straddles the x edge (r0 = 8 > 64 - 62)
    g = _bowl(true)
    cat = np.array([[62, 20, 30], [4, 20, 30], [62, 25, 33]])     # on it; 6 cells off ACROSS the edge; sqrt(34) off
    u = rr.raw_radii_cells(g, cat)
    r_true = u["r_cells"][0]
    assert np.isfinite(r_true) and r_true > 7.0, r_true
    assert (u["r_cells"][1:] < 0.85 * r_true).all(), (u["r_cells"], r_true)       # the hypothesis' too-small sphere
    rc = rr.recentred_radii_cells(g, cat, u["r_cells"])
    d = (rc["coords_rc"] - true).astype(float)
    d -= N * np.rint(d / N)
    assert (np.sqrt((d * d).sum(axis=1)) <= 1.0 + 1e-9).all(), rc["coords_rc"]
    assert np.allclose(rc["r_cells"], r_true, atol=0.3), (rc["r_cells"], r_true)
    assert rc["offset_cells"][0] == 0.0 and abs(rc["offset_cells"][1] - 6.0) <= 1.0, rc["offset_cells"]
    assert rc["rule"].tolist() == [1, 1, 1], rc["rule"]
    # no radius and no fallback: left where it is (rule 0); a fallback radius (the crossing R_v) moves it (rule 2)
    rc2 = rr.recentred_radii_cells(g, cat[1:2], np.array([np.nan]))
    assert rc2["rule"][0] == 0 and (rc2["coords_rc"][0] == cat[1]).all(), rc2
    rc3 = rr.recentred_radii_cells(g, cat[1:2], np.array([np.nan]), fallback_cells=np.array([8.0]))
    assert rc3["rule"][0] == 2 and abs(rc3["r_cells"][0] - r_true) < 0.3, rc3


def t_recentre_cache():
    """raw_radii(recentre=True) caches beside the uncentred file under its own name (voids_raw08rc),
    leaves the uncentred radii as they were, is reused without a grid read, re-measures when a re-centring setting
    changes, and returns the re-centred cells, the offsets and R0."""
    import config
    from dtfelib import raw_radius as rr
    d = Path(tempfile.mkdtemp(prefix="rawr_rc_cache_"))
    try:
        g = _bowl([20, 20, 20])
        cat = {"coords": np.array([[25, 20, 20], [20, 20, 20]]), "well_resolved": np.array([True, True]),
               "r_v_cells": np.array([8.0, 8.0])}
        p = _FakeProducts(g, d)
        base = rr.raw_radii(p, cat)
        reads = p.reads
        a = rr.raw_radii(p, cat, recentre=True)
        assert sorted(q.name for q in d.iterdir()) == ["SYN_ps_099_voids_raw08.npz", "SYN_ps_099_voids_raw08rc.npz"]
        assert p.reads == reads + 1, p.reads
        assert np.array_equal(rr.raw_radii(p, cat)["r_cells"], base["r_cells"], equal_nan=True)    # untouched
        assert (a["coords_rc"] == [20, 20, 20]).all() and a["offset_cells"].tolist() == [5.0, 0.0], a
        assert a["r_cells"][0] > base["r_cells"][0] + 1.0 and np.allclose(a["r0_mpc"], base["r_mpc"], equal_nan=True)
        assert np.allclose(a["r_mpc"], a["r_cells"] * config.CELL_SIZE)
        b = rr.raw_radii(p, cat, recentre=True)
        assert p.reads == reads + 1 and np.array_equal(a["r_cells"], b["r_cells"])          # from the cache
        saved = rr.RC_SMOOTH_CELLS
        try:
            rr.RC_SMOOTH_CELLS = 2.0
            buf = io.StringIO()
            with redirect_stdout(buf):
                rr.raw_radii(p, cat, recentre=True)
            assert p.reads == reads + 2 and "re-measuring" in buf.getvalue()
        finally:
            rr.RC_SMOOTH_CELLS = saved
        assert not [q for q in d.iterdir() if "tmp" in q.name]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_same_centre():
    """Two catalogue minima re-centred onto one raw minimum count once in the fixed-R sample (the deeper
    row kept); rows outside the sample are never dropped or added. And the switch is offered."""
    import plot_void_population as pop
    sel = np.array([True, True, True, False, True])
    centres = np.array([[1, 2, 3], [1, 2, 3], [4, 4, 4], [4, 4, 4], [1, 2, 3]])
    depth = np.array([-0.5, -0.9, -0.7, -0.99, -0.6])
    out, ndup = pop._drop_same_centre(sel, centres, depth)
    assert out.tolist() == [False, True, True, False, False] and ndup == 2, (out, ndup)
    import subprocess
    h = subprocess.run([sys.executable, str(REPO / "python" / "plot" / "plot_void_population.py"), "--help"],
                       capture_output=True, text=True, timeout=120).stdout
    assert "--fixed-r-recentre" in h and "--fixed-r-no-recentre" in h and "--fixed-r-distinct" in h, h[-600:]


def t_raw_distinct():
    """--fixed-r-distinct raw: the largest raw sphere first, a sphere OVERLAPPING a kept one dropped
    (periodic), a row outside the sample or without a radius never kept -- where the catalogue rule keeps the
    DEEPEST minimum and drops the large sphere beside it."""
    import plot_void_population as pop
    n = 64
    centres = np.array([[10, 10, 10], [14, 10, 10], [62, 10, 10], [40, 40, 40], [40, 44, 40], [30, 30, 30]])
    r = np.array([3.0, 6.0, 2.0, 5.0, np.nan, 4.0])
    sample = np.array([True, True, True, True, True, False])
    out = pop.raw_sphere_distinct(centres, r, sample, n)
    # row 1 (R 6) kept first; row 0 (4 cells away, R 3) overlaps it; row 2 is 16 cells away across the edge (>= 6 + 2:
    # kept); row 3 kept; row 4 has no radius; row 5 is outside the sample
    assert out.tolist() == [False, True, True, True, False, False], out.tolist()
    # overlap, not centre-inside: a sphere 8 cells from the R 6 one with R 3 (centre outside, spheres overlap) drops
    two = pop.raw_sphere_distinct(np.array([[14, 10, 10], [22, 10, 10]]), np.array([6.0, 3.0]), np.ones(2, bool), n)
    assert two.tolist() == [True, False], two.tolist()
    two = pop.raw_sphere_distinct(np.array([[14, 10, 10], [24, 10, 10]]), np.array([6.0, 3.0]), np.ones(2, bool), n)
    assert two.tolist() == [True, True], two.tolist()
    # the catalogue's depth order on the same two keeps the deeper, SMALLER one (row 0) and drops the large sphere
    from dtfelib import pipeline
    keep = pipeline.distinct_voids(centres[:2], np.array([-0.95, -0.9]), np.array([5.0, 6.0]), n)
    assert keep.tolist() == [True, False], keep.tolist()


def _cat(n=40, seed=3):
    """A hand-made catalogue: n minima, all resolved and BBKS-valid, crossing R_v in cells, deep ones first."""
    rng = np.random.default_rng(seed)
    coords = rng.integers(0, 512, size=(n, 3))
    e = rng.uniform(0.05, 0.3, n)
    return {"coords": coords, "bbks_params": np.c_[e, rng.uniform(-0.5, 0.5, n) * e],
            "well_resolved": np.ones(n, bool), "delta_values": np.sort(rng.uniform(-0.9, -0.2, n)),
            "semi_axes": np.tile([4.0, 3.0, 2.0], (n, 1)).astype(np.float32),
            "r_v_cells": np.full(n, 0.5), "cell_mpc": np.float64(1.0)}


def _raw(n, r=3.0):
    raw = {"r_mpc": np.full(n, r), "never_below": np.zeros(n, bool), "capped": np.zeros(n, bool)}
    raw["r_mpc"][::4] = np.nan
    raw["never_below"][::8] = True
    raw["capped"][4::8] = True
    return raw


def t_snapshot_stats():
    """snapshot_stats with size_rv 'delta08raw': the size function's sample takes the raw radii, the fraction
    defined and the never-below / capped counts are of the distinct voids; without the raw radii it is the
    crossing sample (the figure says so)."""
    import plot_void_population as pop
    cat = _cat()
    st = pop.snapshot_stats(cat, 512, keep_overlaps=True, size_rv="delta08raw", raw=_raw(40))
    assert st["rv_sf_definition"] == "delta08raw" and abs(st["frac_sf_defined"] - 0.75) < 1e-12, st["frac_sf_defined"]
    assert st["n_sf_never_below"] == 5 and st["n_sf_capped"] == 5, (st["n_sf_never_below"], st["n_sf_capped"])
    assert st["_r_v_sf"].size == 30 and np.all(st["_r_v_sf"] == 3.0)
    st0 = pop.snapshot_stats(cat, 512, keep_overlaps=True, size_rv="delta08raw")
    assert st0["rv_sf_definition"] == "crossing" and st0["n_sf_capped"] == 0
    assert "delta08raw" in pop.SIZE_RV_CHOICES and pop.SIZE_FUNCTION_RV in pop.SIZE_RV_CHOICES


def t_figure():
    """size_function_figure on raw-radius statistics: the png, the definition recorded, the never-below / capped
    counts in the arrays, the raw measurement and the figure's caveat said on it."""
    import plot_void_population as pop
    stats = {s: pop.snapshot_stats(_cat(200, seed=s), 512, keep_overlaps=True, size_rv="delta08raw", raw=_raw(200))
             for s in (50, 99)}
    out = Path(_FIGS_TMP) / "sf"
    out.mkdir(parents=True, exist_ok=True)
    texts = []
    import matplotlib.axes
    import matplotlib.figure
    orig, orig_fig = matplotlib.axes.Axes.text, matplotlib.figure.Figure.text

    def spy(self, *a, **k):
        texts.append(str(a[2]) if len(a) > 2 else "")
        return orig(self, *a, **k)

    def spy_fig(self, *a, **k):                           # the caveat is a figure text under the panels
        texts.append(str(a[2]) if len(a) > 2 else "")
        return orig_fig(self, *a, **k)
    matplotlib.axes.Axes.text, matplotlib.figure.Figure.text = spy, spy_fig
    try:
        with redirect_stdout(io.StringIO()):
            sf = pop.size_function_figure(stats, [50, 99], {50: 1.0, 99: 0.0}, {50: 100.0, 99: 100.0},
                                          Namespace(size_rv="delta08raw", sim="SYN"), ["ps"],
                                          out / "population_size_function_delta08raw.png")
    finally:
        matplotlib.axes.Axes.text, matplotlib.figure.Figure.text = orig, orig_fig
    assert (out / "population_size_function_delta08raw.png").is_file()
    assert sf["size_function_rv"] == "delta08raw" and sf["size_function_99_n_capped"] == 25, sf.get("size_function_99_n_capped")
    said = "\n".join(texts)
    assert "raw density" in said and "75%" in said, said[-400:]
    assert pop.SIZE_FUNCTION_CAVEAT["delta08raw"] in said, said[-400:]          # why it is not a test, on the figure


def t_help():
    """--size-rv offers delta08raw (and the help says what it is)."""
    import subprocess
    pr = subprocess.run([sys.executable, str(REPO / "python" / "plot" / "plot_void_population.py"), "--help"],
                        capture_output=True, text=True, timeout=120)
    assert pr.returncode == 0 and "delta08raw" in pr.stdout, pr.stdout[-600:]


def t_tng100_raw():
    """TNG100-3-Dark z = 0 (DTFE_DATA_ROOT), production catalogue: the raw radius is defined for >= 90% of the
    distinct voids (the smoothed delta08: 30%) and its median is 1.5-3 sigma (the smoothed delta08's 0.72 sigma follows the
    smoothing). Reads the 512^3 density once (cached afterwards)."""
    import config
    if config.SIMULATION != "TNG100-3-Dark":
        raise Skip("needs DTFE_SIM=TNG100-3-Dark")
    data_root()                                     # a Skip without one
    from dtfelib import pipeline, catalog, raw_radius as rr
    import plot_void_population as pop
    prod = pipeline.products(99, sim="TNG100-3-Dark")
    try:
        cell, box, n, _ = catalog.frame_of(prod)
        cat = prod.voids()
        raw = rr.raw_radii(prod, cat)
    except (OSError, ValueError) as e:
        raise Skip(f"no TNG100-3-Dark z = 0 grids or catalogue ({type(e).__name__})")
    finally:
        prod.release()
    st = pop.snapshot_stats(cat, n, False, "delta08raw", raw)
    sig = pipeline.smoothing_length_mpc()
    med = float(np.median(st["_r_v_sf"]))
    assert st["frac_sf_defined"] >= 0.9, st["frac_sf_defined"]
    assert 1.5 < med / sig < 3.0, (med, sig)


def _vdn_sample(rng, box, lo, hi, z=0.0, scale=1.0):
    """Radii drawn from the Vdn size function between lo and hi (Poisson count, inverse CDF in ln R) times 'scale' in
    number, uniform centres in the box: a population the model describes exactly."""
    from dtfelib import theory
    g = np.geomspace(lo, hi, 400)
    dn = theory.void_size_function(g, z)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dn[1:] + dn[:-1]) * np.diff(np.log(g)))])
    n = rng.poisson(scale * box ** 3 * cdf[-1])
    r = np.exp(np.interp(rng.uniform(0, cdf[-1], n), cdf, np.log(g)))
    return r, rng.uniform(0, box, size=(n, 3))


def t_fixed_r():
    """--size-fixed-r: the theory's bin average and range integral; the smoothings (default,
    Mpc ones, a duplicate scale dropped); the common range (3 x the largest sigma .. the scarcest smoothing's 5th
    largest radius, empty when too few); a population drawn from Vdn gives ratio ~1 and 'consistent' at every
    smoothing, one at 0.4 of it does not; the sub-cube bootstrap's draws; the figure, the npz arrays and the verdicts;
    the name tags; the refusal under the ellipsoid gate; --help."""
    import config
    from dtfelib import theory
    import plot_void_population as pop
    # theory: the bin average is what a histogram measures, the range integral matches the cumulative's difference
    e = np.geomspace(5.0, 20.0, 4)
    tb = theory.void_dn_dlnr_binned(e, 0.0)
    assert tb.shape == (3,) and np.all(np.diff(tb) < 0), tb
    n_between = theory.void_number_between(6.0, 15.0, 0.0)
    cum = theory.void_cumulative(np.array([6.0, 15.0]), 0.0)
    assert abs(n_between / (cum[0] - cum[1]) - 1.0) < 0.01, (n_between, cum)
    assert theory.void_number_between(15.0, 6.0, 0.0) == 0.0
    # the smoothings
    sm = pop.fixed_r_smoothings()
    assert [(k, v) for k, v, _ in sm] == [("cells", 10.0), ("cells", 5.0), ("cells", 3.0)], sm
    with redirect_stdout(io.StringIO()) as buf:
        sm2 = pop.fixed_r_smoothings([10.0], [10.0 * config.CELL_SIZE * 1.0002])
    assert len(sm2) == 1 and "counted once" in buf.getvalue(), sm2
    assert pop.fixed_r_tag(sm) == "" and pop.fixed_r_tag(pop.fixed_r_smoothings([3.0], [2.1625])) == "_c3_m2.1625"
    assert pop.fixed_r_tag(sm, (6.5, 12.0)) == "_R6.5-12"
    # the common range
    lo, hi = pop.fixed_r_range([np.arange(1.0, 21.0), np.arange(1.0, 31.0)], [1.0, 2.0], 300.0)
    assert lo == 6.0 and hi == 16.0, (lo, hi)                                 # 3 x 2; the first's 5th largest is 16
    assert np.isnan(pop.fixed_r_range([np.arange(1.0, 9.0)], [2.0], 300.0)[0])   # 8 is not above 6 by 5 voids
    assert pop.fixed_r_range([np.arange(1.0, 9.0)], [2.0], 300.0, (2.0, 7.0)) == (2.0, 7.0)
    assert np.isnan(pop.fixed_r_range([np.r_[np.arange(1.0, 6.0), [6.3] * 5]], [2.0], 300.0)[0])   # 6-6.3: too narrow
    # statistics on Vdn-drawn and on depleted populations, three 'smoothings' each
    rng = np.random.default_rng(7)
    box = 400.0
    sig = [2.0, 1.0, 0.6]

    # ONE population seen by every 'smoothing' (as on real data, where they share one density field), each keeping
    # the fraction 'keep' of it: the agreement test is then deterministic
    r_all, pos_all = _vdn_sample(rng, box, 3.0, 40.0, 0.0)

    def entries(keeps):
        out = []
        for i, (s_, keep) in enumerate(zip(sig, keeps)):
            m = rng.uniform(size=r_all.size) < keep
            out.append({"label": f"s{i}", "sigma_mpc": s_, "r": {99: r_all[m]}, "pos": {99: pos_all[m]},
                        "n_resolved": {99: int(m.sum())}, "n_never_below": {99: 0}, "n_capped": {99: 0}})
        return out
    good = pop.fixed_r_statistics(entries((1.0, 1.0, 1.0)), {99: box}, {99: 0.0}, [99])[99]
    bad = pop.fixed_r_statistics(entries((0.4, 0.4, 0.4)), {99: box}, {99: 0.0}, [99])[99]
    split = pop.fixed_r_statistics(entries((1.0, 1.0, 0.4)), {99: box}, {99: 0.0}, [99])[99]
    assert not good["empty"] and good["r_lo"] == 6.0 and good["r_hi"] > good["r_lo"], good["r_hi"]
    for p in good["per"]:
        assert abs(p["ratio"] - 1.0) < 3.0 * p["ratio_err"] + 0.05, (p["ratio"], p["ratio_err"])
        assert p["ratio_boot"] > 0 and p["dn_dlnr_boot"].shape == (pop.FIXED_R_BINS,) and p["n_total"] > 100, p
        assert np.isclose(p["ratio"], p["n_total"] / (box ** 3 * good["theory_total"]))
    assert good["consistent"] and good["converged"], (good["ratio_mean"], good["ratio_scatter"])
    assert not bad["consistent"] and 0.3 < bad["ratio_mean"] < 0.5 and bad["converged"], bad["ratio_mean"]
    assert not split["converged"] and not split["consistent"], (split["chi2"], split["chi2_95"])
    assert "smoothings disagree" in pop.fixed_r_verdict(split) and "NOT consistent" in pop.fixed_r_verdict(split)
    bs = pop.bootstrap_counts(np.array([7.0, 8.0, 9.0]), np.array([[1, 1, 1], [399, 1, 1], [200, 200, 399.9]]), box,
                              np.array([6.0, 8.5, 10.0]), np.array([6.0, 8.5]), nboot=50)
    assert bs["hist"].shape == (50, 2) and bs["cum"].shape == (50, 2) and np.allclose(bs["hist"].sum(1), bs["tot"])
    # the figure: an empty row said, the verdicts and arrays returned
    st_empty = pop.fixed_r_statistics([{"label": "x", "sigma_mpc": 2.0, "r": {33: np.array([1.0, 2.0])},
                                        "pos": {33: np.zeros((2, 3))}, "n_resolved": {33: 2}, "n_never_below": {33: 0},
                                        "n_capped": {33: 0}}], {33: box}, {33: 2.0}, [33])
    assert st_empty[33]["empty"] and "no common complete range" in pop.fixed_r_verdict(st_empty[33])
    few = pop.fixed_r_statistics(entries((1.0, 1.0, 1.0)), {99: box}, {99: 0.0}, [99], r_range=(39.0, 40.0))[99]
    assert few["too_few"] and "too few voids" in pop.fixed_r_verdict(few), pop.fixed_r_verdict(few)
    path = Path(_FIGS_TMP) / "fixedr" / "population_size_function_fixedR.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(io.StringIO()):
        arr = pop.fixed_r_figure({99: good, 33: st_empty[33]}, [99, 33], "SYN", ["ps"], path)
    assert path.is_file() and len(arr["fixed_r_verdicts"]) == 2, arr.get("fixed_r_verdicts")
    assert "CONSISTENT with Vdn" in str(arr["fixed_r_verdicts"][0]) and arr["fixed_r_99_consistent"]
    assert arr["fixed_r_99_2_ratio"] == good["per"][2]["ratio"] and "fixed_r_33_edges_mpc" not in arr
    # gated: refused (nothing converges with the gate on), with the run line
    saved = config.VOID_GATE
    try:
        config.VOID_GATE = "ellipsoid"
        try:
            pop.fixed_r_main(Namespace(sim="SYN"), {})
            raise AssertionError("fixed_r_main ran under the ellipsoid gate")
        except SystemExit as ex:
            assert "DTFE_VOID_GATE=none" in str(ex), ex
    finally:
        config.VOID_GATE = saved
    import subprocess
    pr = subprocess.run([sys.executable, str(REPO / "python" / "plot" / "plot_void_population.py"), "--help"],
                        capture_output=True, text=True, timeout=120)
    assert pr.returncode == 0 and "--size-fixed-r" in pr.stdout and "--fixed-r-smooth-mpc" in pr.stdout, pr.stdout[-600:]


def _fixed_r_entries(r_all, pos_all, keeps, sig, rng, snap=99):
    """One 'smoothing' per keep fraction, each a random subset of ONE population (as on real data, where the
    smoothings share one density field and largely count the same voids)."""
    out = []
    for i, (s_, keep) in enumerate(zip(sig, keeps)):
        m = rng.uniform(size=r_all.size) < keep
        out.append({"label": f"s{i}", "sigma_mpc": s_, "r": {snap: r_all[m]}, "pos": {snap: pos_all[m]},
                    "n_resolved": {snap: int(m.sum())}, "n_never_below": {snap: 0}, "n_capped": {snap: 0}})
    return out


def t_fixed_r_half_open():
    """Every fixed-R count is half-open [lo, hi): r_hi is an exact sample radius
    (the scarcest smoothing's 5th largest), and the closed last bin of np.histogram would count that void in dn/dlnR
    and in the bootstrap's bins while n_total, the bootstrap's 'tot' and Vdn's number leave it out."""
    import plot_void_population as pop
    # the probe: edges[-1] == 9.5 is a sample value
    r = np.array([6.6, 7.0, 7.5, 8.0, 9.0, 9.5, 10.0, 11.2, 12.0, 13.0])
    pos = np.random.default_rng(1).uniform(0, 400.0, size=(r.size, 3))
    edges = np.geomspace(6.49, 9.5, 7)
    bs = pop.bootstrap_counts(r, pos, 400.0, edges, edges[:2], nboot=50)
    assert np.array_equal(bs["hist"].sum(1), bs["tot"]), (bs["hist"].sum(1)[:5], bs["tot"][:5])
    # through fixed_r_statistics: hi is the 5th largest radius of every (identical) smoothing
    rng = np.random.default_rng(7)
    r_all, pos_all = _vdn_sample(rng, 400.0, 3.0, 40.0, 0.0)
    st = pop.fixed_r_statistics(_fixed_r_entries(r_all, pos_all, (1.0, 1.0), (2.0, 1.0), rng), {99: 400.0},
                                {99: 0.0}, [99])[99]
    assert st["r_hi"] in r_all, st["r_hi"]
    for p in st["per"]:
        assert int(p["count_bin"].sum()) == p["n_total"], (p["count_bin"].sum(), p["n_total"])


def t_fixed_r_paired():
    """The smoothings' agreement is tested on the PAIRED sub-cube bootstrap (identical
    resamples for every smoothing), not on independent errors: two catalogues of one population, the second missing
    a random 10% of its voids, differ by ~sqrt(0.1 N) voids in the paired draws but by ~sqrt(2 N) with independent
    errors, so the independent-error chi^2 (kept for reference) says 'agree' while the paired test finds the 10% deficit."""
    import plot_void_population as pop
    rng = np.random.default_rng(11)
    box = 200.0                                    # a few hundred voids: the independent errors (~5%) hide 10%
    r_all, pos_all = _vdn_sample(rng, box, 3.0, 40.0, 0.0)
    st = pop.fixed_r_statistics(_fixed_r_entries(r_all, pos_all, (1.0, 0.9), (2.0, 1.0), rng), {99: box},
                                {99: 0.0}, [99])[99]
    n = [p["n_total"] for p in st["per"]]
    assert 200 < n[0] < 1000 and n[1] < n[0], n
    assert st["chi2"] <= st["chi2_95"], (st["chi2"], st["chi2_95"])      # independent errors: no difference seen
    assert not st["converged"], (st["chi2"], st["chi2_95"])                # the paired test finds the deficit
    assert st["paired_chi2"] > st["chi2_95"], (st["paired_chi2"], st["paired_max_sigma"])
    assert "smoothings disagree" in pop.fixed_r_verdict(st) and "paired" in pop.fixed_r_verdict(st)
    # identical catalogues: zero difference with zero paired spread is agreement (0/0 -> 0), not NaN
    same = pop.fixed_r_statistics(_fixed_r_entries(r_all, pos_all, (1.0, 1.0), (2.0, 1.0), rng), {99: box},
                                  {99: 0.0}, [99])[99]
    assert same["converged"] and same["paired_chi2"] == 0.0, same["paired_chi2"]
    assert "no significant difference" in pop.fixed_r_verdict(same), pop.fixed_r_verdict(same)


def main():
    print("size function raw radius tests")
    for name, fn in (("analytic top-hat void: radius, wrap, blocks, never-below, capped", t_analytic),
                     ("a wisp on the centre does not undefine the void", t_centre_wisp),
                     ("the raw radius does not follow the smoothing (the smoothed delta08 does)", t_not_smoothing),
                     ("raw radii cache: rows, reuse, identity, stamp, settings, atomic", t_cache),
                     ("the products' held density grid used, not read again", t_held_grid),
                     ("re-centring on the raw minimum restores an off-centre void's radius", t_recentre),
                     ("re-centred radii cached apart, reused, re-measured on new settings",
                      t_recentre_cache),
                     ("one re-centred centre counted once; the switch offered", t_same_centre),
                     ("distinct by the raw spheres, largest first, no overlap", t_raw_distinct),
                     ("snapshot_stats with delta08raw", t_snapshot_stats),
                     ("size_function_figure with delta08raw", t_figure),
                     ("--help offers delta08raw", t_help),
                     ("size function at fixed R: range, ratio, bootstrap, verdict, figure, gate", t_fixed_r),
                     ("fixed R: every count half-open [lo, hi), the void at r_hi in none", t_fixed_r_half_open),
                     ("fixed R: smoothings compared on the paired bootstrap, not independent errors",
                      t_fixed_r_paired),
                     ("TNG100 z = 0 (DTFE_DATA_ROOT): raw radius defined and not the smoothing length", t_tng100_raw)):
        check(name, fn)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
