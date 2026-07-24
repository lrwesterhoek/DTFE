#!/usr/bin/env python3
"""Tests for the Python merger-tree / void-tracking stack (dtfelib).

Two tiers:
  * SYNTHETIC (always run): grid sampling conventions, periodic helpers, void-centre
    tracking across the box wrap, catalog matching.
  * DATA-BACKED (run when the simulation data is on disk, else skipped with a note):
    SubLink invariants on the real TNG50-3-Dark trees -- branch monotonicity, the
    contiguous-row arithmetic, merger-event descendant links -- the Subfind
    group-membership invariant behind subhalo_particle_range, and the FieldSet
    single-stream velocity mask on the TNG50-4-Dark raw PS grids.

Usage:  python3 tests/py_dtfelib_test.py            (exit 0 = pass)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

PASS, FAIL, SKIP = [], [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")
    except (FileNotFoundError, KeyError) as e:
        SKIP.append(name)
        print(f"  SKIP  {name}  (data not on disk: {e})")


# ---------------------------------------------------------------- synthetic: sampling
def t_sample_grid():
    from dtfelib.environment import sample_grid
    n = 8
    g = np.zeros((n, n, n)); g[2, 3, 4] = 1.0
    ngp = sample_grid(g, np.array([[2.5/n, 3.5/n, 4.5/n], [2.01/n, 3.99/n, 4.5/n], [0.99, 0.01, 0.5]]), "ngp")
    assert ngp.tolist() == [1.0, 1.0, 0.0], f"NGP containment: {ngp}"
    tri = sample_grid(g, np.array([[2.5/n, 3.5/n, 4.5/n], [3.0/n, 3.5/n, 4.5/n]]), "trilinear")
    assert abs(tri[0] - 1.0) < 1e-12 and abs(tri[1] - 0.5) < 1e-12, f"cell-centre trilinear: {tri}"
    g2 = np.zeros((n, n, n)); g2[0, 0, 0] = 1.0
    tri2 = sample_grid(g2, np.array([[(n-0.25)/n, 0.5/n, 0.5/n]]), "trilinear")
    assert abs(tri2[0] - 0.25) < 1e-12, f"periodic wrap: {tri2}"
    gv = np.zeros((n, n, n, 3)); gv[2, 3, 4] = (1., 2., 3.)
    assert np.allclose(sample_grid(gv, np.array([[2.5/n, 3.5/n, 4.5/n]]), "trilinear")[0], [1, 2, 3])
    rng = np.random.default_rng(1)
    assert np.allclose(sample_grid(np.full((n, n, n), 7.0), rng.random((100, 3)), "trilinear"), 7.0)


# ---------------------------------------------------------------- synthetic: tracking
def t_periodic_helpers():
    from dtfelib.voids import min_image, periodic_median_shift, wrap_frac
    assert abs(min_image(np.array(0.9)) - (-0.1)) < 1e-12
    # displacement crossing the wrap: 0.98 -> 0.02 is +0.04, not -0.96
    d = periodic_median_shift(np.array([[0.98, 0.5, 0.5]]), np.array([[0.02, 0.5, 0.5]]))
    assert abs(d[0] - 0.04) < 1e-12, d
    assert 0 <= wrap_frac(np.array(-0.1)) < 1


def t_track_center_wrap():
    from dtfelib.voids import track_center
    # three tracers orbiting a centre that drifts ACROSS the box edge, snap 99 -> 90
    rng = np.random.default_rng(2)
    offsets = rng.uniform(-0.03, 0.03, size=(3, 3))
    branches = {}
    for i in range(3):
        b = {}
        for j, s in enumerate(range(99, 89, -1)):
            centre = np.array([0.97 + 0.01 * j, 0.5, 0.5])     # crosses 1.0 at j=3
            b[s] = np.mod(centre + offsets[i], 1.0)
        branches[i] = b
    centers = track_center(branches, 99, np.array([0.97, 0.5, 0.5]), min_tracers=3)
    assert len(centers) == 10, f"tracked {len(centers)} snapshots"
    # at snap 90 (j=9) the true centre is 0.97+0.09 = 1.06 -> 0.06 wrapped
    assert abs(centers[90][0] - 0.06) < 1e-9, f"wrap drift: {centers[90]}"


def t_match_catalog_void():
    from dtfelib.voids import match_catalog_void
    cat = {"coords": np.array([[10, 10, 10], [100, 100, 100], [255, 4, 4]])}
    j, d = match_catalog_void(cat, np.array([0.999, 0.018, 0.018]), 256, 0.05)   # wraps to void 2
    assert j == 2, (j, d)
    j, _ = match_catalog_void(cat, np.array([0.5, 0.5, 0.5]), 256, 0.01)         # nothing near
    assert j is None


def t_shape_estimators():
    """Vectorized BBKS/ellipsoid-fit estimators must reproduce the original per-void
    loops bit-for-bit: NaN rows (trace/lambda1 thresholds, sanity gate), ELLIPSOID_CUTS
    gating, the |lambda| argsort tie order, and the exact float32 roundings."""
    from dtfelib import fields, pipeline

    def ref_shapes(eigenvalues):        # the pre-vectorization loop, verbatim
        n = len(eigenvalues)
        axis_ratios = np.zeros((n, 2))
        bbks = np.zeros((n, 2))
        for i, evals in enumerate(eigenvalues):
            l1, l2, l3 = np.sort(evals)
            tr = l1 + l2 + l3
            bbks[i] = [(l3 - l1) / (2 * tr), (l1 - 2 * l2 + l3) / (2 * tr)] \
                if tr > 1e-12 else [np.nan, np.nan]
            if l1 <= 1e-12:
                axis_ratios[i] = [np.nan, np.nan]
                continue
            a, b, c = 1.0 / np.sqrt(l1), 1.0 / np.sqrt(l2), 1.0 / np.sqrt(l3)
            b_a, c_a = b / a, c / a
            axis_ratios[i] = [b_a, c_a] if (0 <= c_a <= b_a <= 1.0) else [np.nan, np.nan]
        return axis_ratios, bbks

    def ref_fits(coords, evals, evecs):  # the pre-vectorization loop around ellipsoid_fit_one
        cell, cuts = pipeline.config.CELL_SIZE, pipeline.config.ELLIPSOID_CUTS
        n = len(coords)
        out = {"well_resolved": np.zeros(n, bool),
               "semi_axes": np.zeros((n, 3), np.float32),
               "orientations": np.zeros((n, 3, 3), np.float32),
               "fit_ellipticity": np.zeros(n, np.float32),
               "fit_prolateness": np.zeros(n, np.float32),
               "positions_mpc": coords.astype(np.float32) * cell}
        for i in range(n):
            fit = pipeline.ellipsoid_fit_one(evals[i], evecs[i], cell)
            if fit is None:
                continue
            axes = fit["semi_axes"]
            if axes.min() < cuts["min_axis_mpc"] or axes.max() > cuts["max_axis_mpc"]:
                continue
            if axes[0] / axes[2] > cuts["max_axis_ratio"]:
                continue
            out["well_resolved"][i] = True
            out["semi_axes"][i] = axes
            out["orientations"][i] = fit["orientation"]
            out["fit_ellipticity"][i] = fit["ell"]
            out["fit_prolateness"][i] = fit["prol"]
        return out

    cell = pipeline.config.CELL_SIZE
    rng = np.random.default_rng(5)
    crafted = np.array([
        [0.0, 0.0, 0.0],            # zero trace -> NaN bbks
        [-1.0, -2.0, -3.0],         # negative trace -> NaN bbks
        [1e-13, 0.5, 1.0],          # lambda1 <= 1e-12: bbks valid, ratios NaN
        [2.0, -2.0, 3.0],           # |lambda| tie -> argsort tie order must match
        [5e-11, 1.0, 2.0],          # |ev| < 1e-10 -> ellipsoid fit degenerate
        [1.0, 1.0, 1.0],            # sphere
        [1e-6, 1e-6, 1e6],          # extreme ratio
    ])
    guaranteed_pass = (2.0 * cell / np.array([[1.0, 2.0, 3.0], [2.0, 2.5, 3.0]])) ** 2
    ev64 = np.vstack([rng.normal(0.0, 1.0, (200, 3)), crafted, guaranteed_pass])
    for ev in (ev64, ev64.astype(np.float32)):
        got = fields.calculate_shape_parameters(list(ev))
        ref_ar, ref_bbks = ref_shapes(list(ev))
        assert np.array_equal(got["axis_ratios"], ref_ar, equal_nan=True), "axis_ratios drifted"
        assert np.array_equal(got["bbks_params"], ref_bbks, equal_nan=True), "bbks_params drifted"

        n = len(ev)
        coords = rng.integers(0, 64, (n, 3))
        evecs = rng.normal(0.0, 1.0, (n, 3, 3)).astype(ev.dtype)
        got_f = pipeline._ellipsoid_fits(coords, ev, evecs)
        ref_f = ref_fits(coords, ev, evecs)
        for k in ref_f:
            assert np.array_equal(got_f[k], ref_f[k]), f"_ellipsoid_fits[{k}] drifted"
        assert got_f["well_resolved"].any() and not got_f["well_resolved"].all()

    empty = fields.calculate_shape_parameters([])
    assert empty["axis_ratios"].shape == (0, 2) and empty["bbks_params"].shape == (0, 2)


# ---------------------------------------------------------------- data-backed: trees
SIM = "TNG50-3-Dark"


def t_main_branch_invariants():
    from dtfelib.trees import TreeSet
    ts = TreeSet(SIM)
    br = ts.main_branch(99, 0, ["SubhaloMass", "SubhaloPos"])
    sn = br["SnapNum"].astype(int)
    assert sn[0] == 99 and int(br["SubfindID"][0]) == 0, "branch must start at the query"
    assert (np.diff(sn) < 0).all(), "SnapNum must be strictly decreasing along the branch"
    assert br["SubhaloMass"].shape[0] == sn.size and br["SubhaloPos"].shape == (sn.size, 3)


def t_descendant_inverse():
    from dtfelib.trees import TreeSet
    ts = TreeSet(SIM)
    br = ts.main_branch(99, 0, [])
    s_early, sid_early = int(br["SnapNum"][-1]), int(br["SubfindID"][-1])
    fwd = ts.descendant_branch(s_early, sid_early, [])
    assert int(fwd["SnapNum"][-1]) == 99, "forward walk must reach the tree root epoch"
    assert int(fwd["SubfindID"][-1]) == 0, "forward walk must land on the original subhalo"


def t_merger_row_arithmetic():
    """Verify on real data that secondary-progenitor rows resolved
    with r-relative arithmetic hold the right IDs and merge into the right descendant."""
    from dtfelib.trees import TreeSet
    ts = TreeSet(SIM)
    chunk, row = ts.find(99, 0)
    sid = ts._col(chunk, "SubhaloID"); fprog = ts._col(chunk, "FirstProgenitorID")
    nprog = ts._col(chunk, "NextProgenitorID"); desc = ts._col(chunk, "DescendantID")
    checked = 0
    r = row
    while checked < 2000:
        p = fprog[r]
        if p == -1:
            break
        pr = r + int(p - sid[r])
        s = nprog[pr]
        while s != -1 and checked < 2000:
            srow = r + int(s - sid[r])
            assert sid[srow] == s, "row arithmetic must resolve the secondary's own ID"
            assert desc[srow] == sid[r], "secondary must merge into the tracked node"
            checked += 1
            s = nprog[srow]
        r = pr
    assert checked > 100, f"only {checked} mergers exercised"


def t_mergers_api():
    from dtfelib.trees import TreeSet
    ts = TreeSet(SIM)
    ev = ts.mergers_along_branch(99, 0, min_ratio=0.1)
    assert ev["snap"].size > 0, "the most massive halo must have >=1:10 mergers"
    assert (ev["ratio"] >= 0.1).all() and np.isfinite(ev["ratio"]).all()
    assert (ev["snap"] <= 99).all() and (ev["sec_mass"] > 0).all()


def t_groupcat_membership():
    from dtfelib import groupcat as gc
    cat = gc.load(SIM, 99, subhalo_fields=("SubhaloGrNr",),
                  group_fields=("GroupFirstSub", "GroupNsubs", "GroupLenType"))
    grnr = cat["SubhaloGrNr"].astype(np.int64)
    first = cat["GroupFirstSub"].astype(np.int64); nsub = cat["GroupNsubs"].astype(np.int64)
    ids = np.arange(grnr.size)
    ok = (ids >= first[grnr]) & (ids < first[grnr] + nsub[grnr])
    assert ok.all(), f"group-membership contiguity violated for {int((~ok).sum())} subhalos"
    # particle ranges: monotone offsets, inside the total DM particle count
    total_dm = int(cat["GroupLenType"][:, gc.DM].sum())
    rng = np.random.default_rng(3)
    for sid in rng.choice(grnr.size, 50, replace=False):
        off, ln = gc.subhalo_particle_range(SIM, 99, int(sid))
        assert 0 <= off and off + ln <= total_dm, f"range [{off},{off+ln}) outside {total_dm}"


def t_environment_box():
    from dtfelib.environment import box_ckpc_h
    box = box_ckpc_h(SIM, 99)
    assert abs(box - 35000.0) < 1.0, f"TNG50 raw box must be 35000 ckpc/h, got {box}"


def t_velocity_single_stream():
    """NaN-mask semantics of FieldSet.velocity_single_stream on real raw PS grids
    (TNG50-4-Dark). These grids come from the SAMPLED deposit, so the raw '.streams'
    multiplicities are integers; the tolerance mask must reproduce the crisp != 1 mask
    exactly there. t_velocity_single_stream_float_mask covers the exact deposit's floats."""
    from dtfelib.cli import DATA_ROOT
    from dtfelib.io import FieldSet
    snapdir = Path(DATA_ROOT) / "TNG50-4-Dark" / "snapdir_099"
    fs = FieldSet(snapdir, method="ps", averaged=False)
    st = fs.load("streams")
    v = fs.load("velocity")
    v1 = fs.velocity_single_stream()
    assert np.array_equal(st, np.round(st)), "fixture assumption: sampled raw streams are integers"
    multi = st != 1
    assert 0 < multi.mean() < 1, f"trivial mask ({multi.mean():.0%} multi-stream)"
    assert np.isnan(v1[multi]).all(), "multi-stream cells must be NaN"
    assert np.array_equal(v1[~multi], v[~multi]), "single-stream velocities must be untouched"
    try:
        FieldSet(snapdir, method="dtfe").velocity_single_stream()
    except ValueError:
        pass
    else:
        raise AssertionError("method='dtfe' must raise ValueError")


def t_velocity_single_stream_float_mask():
    """velocity_single_stream must survive a FLOAT '.streams' grid (--ps-exact-deposit).

    The exact deposit writes the analytic volume-weighted multiplicity, so single-stream
    cells land on 1.0 +/- float32 eps rather than exactly 1. The old 'streams != 1' mask
    NaN'd essentially the whole grid; the tolerance mask must keep every ~1.0 cell.
    """
    import shutil
    import tempfile

    from dtfelib.io import FieldSet
    try:
        import h5py
    except ImportError as e:              # FieldSet needs a combined_*.hdf5 for units
        raise FileNotFoundError(f"h5py unavailable: {e}")   # -> SKIP, not FAIL
    d = Path(tempfile.mkdtemp(prefix="dtfelib_streams_"))
    try:
        n = 8
        rng = np.random.default_rng(11)
        # single-stream cells offset by a whole number of float32 ULPs, so NOT ONE of them
        # is exactly 1.0 (drawing uniform noise instead would round part of the grid back
        # onto 1.0 and let the '!= 1' bug pass). Plus a genuine multi-stream slab at the
        # analytic pancake value 3 and caustic-straddling cells at a fractional
        # multiplicity -- the shapes --ps-exact-deposit actually produces.
        one, ulp = np.float32(1.0), np.spacing(np.float32(1.0))
        st = (one + rng.choice(np.float32([-1.0, 1.0]), (n, n, n))
                  * rng.integers(1, 4, (n, n, n)).astype(np.float32) * ulp).astype(np.float32)
        st[2:4] = 3.0000091
        st[5] = 2.3174     # caustic-straddling cells: genuinely fractional, must be masked
        st.tofile(d / "ps_output.streams")
        assert not np.any(st == 1.0), "fixture must have NO exactly-1.0 cell, else it proves nothing"
        vel = rng.normal(size=(n, n, n, 3)).astype(np.float32)
        vel.tofile(d / "ps_output.vel")
        np.ones((n, n, n), dtype=np.float32).tofile(d / "ps_output.den")
        with h5py.File(d / "combined_000.hdf5", "w") as f:
            h = f.create_group("Header")
            h.attrs["BoxSize"] = 100000.0
            h.attrs["NumPart_ThisFile"] = np.array([0, n**3, 0, 0, 0, 0], dtype=np.int64)
            h.attrs["MassTable"] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            h.attrs["Redshift"] = 0.0
            h.attrs["HubbleParam"] = 0.7
            h.attrs["HFreeUnits"] = 1

        fs = FieldSet(d, method="ps", averaged=False)
        v1 = fs.velocity_single_stream()
        single = np.abs(st.astype(np.float64) - 1.0) < 1e-3      # rows 0,1,4,6,7
        assert single.mean() > 0.5, f"fixture is degenerate ({single.mean():.0%} single-stream)"
        kept = ~np.isnan(v1).any(axis=-1)
        assert np.array_equal(kept, single), (
            f"mask must keep exactly the ~1.0 cells: kept {kept.sum()} of {single.sum()} "
            f"single-stream cells (the '!= 1' bug keeps 0)")
        assert np.array_equal(v1[single], vel[single]), "kept velocities must be untouched"
        assert np.isnan(v1[~single]).all(), "multi-stream cells must be fully NaN"
    finally:
        shutil.rmtree(d)


def t_velocity_scale():
    """FieldSet.load() converts u-units to peculiar km/s: first moments x sqrt(a), the
    dispersion (a velocity VARIANCE) x a, everything else untouched; exact no-op at z=0."""
    import shutil
    import tempfile

    from dtfelib.io import FieldSet
    try:
        import h5py
    except ImportError as e:
        raise FileNotFoundError(f"h5py unavailable: {e}")   # -> SKIP, not FAIL
    rng = np.random.default_rng(23)
    n = 8

    def make(dirpath, redshift):
        np.abs(rng.normal(1, 0.1, (n, n, n))).astype(np.float32).tofile(dirpath / "ps_output.den")
        rng.normal(0, 100, (n, n, n, 3)).astype(np.float32).tofile(dirpath / "ps_output.vel")
        np.abs(rng.normal(0, 50, (n, n, n))).astype(np.float32).tofile(dirpath / "ps_output.velDisp")
        rng.normal(0, 10, (n, n, n)).astype(np.float32).tofile(dirpath / "ps_output.velDiv")
        with h5py.File(dirpath / "combined_000.hdf5", "w") as f:
            h = f.create_group("Header")
            h.attrs["BoxSize"] = 100000.0
            h.attrs["NumPart_ThisFile"] = np.array([0, n**3, 0, 0, 0, 0], dtype=np.int64)
            h.attrs["MassTable"] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            h.attrs["Redshift"] = float(redshift)
            h.attrs["HubbleParam"] = 0.7
            h.attrs["HFreeUnits"] = 1

    d = Path(tempfile.mkdtemp(prefix="dtfelib_vscale_"))
    try:
        make(d, redshift=3.0)                                     # a = 0.25
        fs = FieldSet(d, method="ps", averaged=False)
        assert abs(fs.velocity_scale - 0.5) < 1e-12, fs.velocity_scale
        for name, exp in (("velocity", 0.5), ("divergence", 0.5), ("dispersion", 1.0)):
            raw = fs.load(name, scaled=False)
            sc = fs.load(name)
            assert np.array_equal(sc, raw * np.float32(0.25 ** exp)), f"{name} scaling wrong"
        assert np.array_equal(fs.load("density"), fs.load("density", scaled=False)), \
            "density must NOT be velocity-scaled"

        shutil.rmtree(d); d.mkdir()
        make(d, redshift=0.0)                                     # a = 1: exact no-op
        fs0 = FieldSet(d, method="ps", averaged=False)
        assert np.array_equal(fs0.load("velocity"), fs0.load("velocity", scaled=False)), \
            "z=0 must be bit-identical (factor exactly 1.0)"
    finally:
        shutil.rmtree(d)


def t_outofcore_loading():
    """load(mode=...), load_slice, iter_slabs and field_stats: the auto-tuned out-of-core
    layer must reproduce the eager path EXACTLY (scaling included) and pick ram-vs-memmap
    from the budget (made deterministic here via the DTFE_PY_RAM_GB override)."""
    import os
    import shutil
    import tempfile

    from dtfelib.io import FieldSet
    try:
        import h5py
    except ImportError as e:
        raise FileNotFoundError(f"h5py unavailable: {e}")   # -> SKIP, not FAIL
    rng = np.random.default_rng(31)
    n = 16
    d = Path(tempfile.mkdtemp(prefix="dtfelib_ooc_"))
    saved = {k: os.environ.get(k) for k in ("DTFE_PY_RAM_GB", "DTFE_PY_LOAD_FRAC")}
    try:
        np.abs(rng.normal(1, 0.2, (n, n, n))).astype(np.float32).tofile(d / "ps_output.den")
        rng.normal(0, 100, (n, n, n, 3)).astype(np.float32).tofile(d / "ps_output.vel")
        (np.abs(rng.normal(1, 0.5, (n, n, n))) + 1).astype(np.float32).tofile(d / "ps_output.streams")
        with h5py.File(d / "combined_000.hdf5", "w") as f:
            h = f.create_group("Header")
            h.attrs["BoxSize"] = 100000.0
            h.attrs["NumPart_ThisFile"] = np.array([0, n**3, 0, 0, 0, 0], dtype=np.int64)
            h.attrs["MassTable"] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            h.attrs["Redshift"] = 3.0                        # a = 0.25: scaling is LIVE
            h.attrs["HubbleParam"] = 0.7
            h.attrs["HFreeUnits"] = 1
        fs = FieldSet(d, method="ps", averaged=False)
        ref_vel = fs.load("velocity", mode="ram")            # scaled eager reference
        ref_den = fs.load("density", mode="ram")

        # --- auto picks ram under a huge budget, memmap under a tiny one ---
        os.environ["DTFE_PY_RAM_GB"] = "1000"
        assert not isinstance(fs.load("density"), np.memmap), "huge budget must load eagerly"
        os.environ["DTFE_PY_RAM_GB"] = "0.0000001"
        mm = fs.load("density")                              # density has no u-units factor
        assert isinstance(mm, np.memmap), "tiny budget must return a memmap"
        assert np.array_equal(np.array(mm), ref_den), "memmap content must equal eager load"
        try:
            mm[0, 0, 0] = 9.0
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("the memmap must be read-only")
        # a scale-needing field over budget must refuse with guidance, not silently unscale
        try:
            fs.load("velocity")
        except MemoryError as e:
            assert "load_slice" in str(e), f"error must name the escape hatches: {e}"
        else:
            raise AssertionError("over-budget scaled load must raise MemoryError")
        # ... but scaled=False and mode='ram' both stay available
        assert isinstance(fs.load("velocity", scaled=False), np.memmap)
        assert np.array_equal(fs.load("velocity", mode="ram"), ref_vel)
        # velocity_single_stream forces ram internally: must work under the tiny budget
        v1 = fs.velocity_single_stream()
        assert np.isnan(v1).any() and np.isfinite(v1).any(), "mask must act, not blanket"

        # --- load_slice == eager slice, every axis, scaling included ---
        for axis in (0, 1, 2):
            sl = [slice(None)] * 3
            sl[axis] = n // 2
            assert np.array_equal(fs.load_slice("velocity", axis=axis), ref_vel[tuple(sl)]), \
                f"load_slice axis={axis} != eager slice"

        # --- iter_slabs reassembles the eager load exactly; stats agree ---
        parts = [slab for _, _, slab in fs.iter_slabs("velocity", max_bytes=4 * n * n * 3 * 4)]
        assert len(parts) > 1, "test must actually exercise multiple slabs"
        assert np.array_equal(np.concatenate(parts, axis=0), ref_vel), "slab concat != eager"
        st = fs.field_stats("velocity")
        assert abs(st["mean"] - ref_vel.astype(np.float64).mean()) < 1e-10
        assert st["min"] == float(ref_vel.min()) and st["max"] == float(ref_vel.max())
        assert st["n"] == ref_vel.size
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(d)


def t_caustic_field_loader():
    """FieldSet loads the 0/1 '.caustic' fold-flag grid written by 'PS-DTFE --ps-caustics'
    (synthetic files; the field is float32 like every grid, so no dtype column is needed)."""
    import shutil
    import tempfile

    from dtfelib.io import FieldSet
    try:
        import h5py
    except ImportError as e:              # FieldSet needs a combined_*.hdf5 for units
        raise FileNotFoundError(f"h5py unavailable: {e}")   # -> SKIP, not FAIL
    d = Path(tempfile.mkdtemp(prefix="dtfelib_caustic_"))
    try:
        n = 8
        np.ones((n, n, n), dtype=np.float32).tofile(d / "ps_output.den")
        flag = (np.random.default_rng(7).random((n, n, n)) < 0.2).astype(np.float32)
        flag.tofile(d / "ps_output.caustic")
        with h5py.File(d / "combined_000.hdf5", "w") as f:
            h = f.create_group("Header")
            h.attrs["BoxSize"] = 100000.0
            h.attrs["NumPart_ThisFile"] = np.array([0, n**3, 0, 0, 0, 0], dtype=np.int64)
            h.attrs["MassTable"] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            h.attrs["Redshift"] = 0.0
            h.attrs["HubbleParam"] = 0.7
            h.attrs["HFreeUnits"] = 1
        fs = FieldSet(d, method="ps", averaged=False)
        assert fs.has("caustic"), "has('caustic') must see ps_output.caustic"
        c = fs.load("caustic")
        assert c.shape == (n, n, n) and c.dtype == np.float32, f"{c.shape} {c.dtype}"
        assert np.array_equal(c, flag), "caustic grid roundtrip mismatch"
        assert set(np.unique(c).tolist()) <= {0.0, 1.0}, "flag must be 0/1"
        try:
            FieldSet(d, method="dtfe", averaged=False).load("caustic")
        except ValueError:
            pass
        else:
            raise AssertionError("caustic is ps_only; method='dtfe' must raise ValueError")
    finally:
        shutil.rmtree(d)


def t_alternate_prefix():
    """FieldSet(prefix=...) reads an alternate OUTPUT_PREFIX grid set (e.g. ps_mw.*).

    Contract: the prefix only redirects the on-disk files -- units, scaling and the
    field table are untouched; method 'auto' resolves to 'ps' (only run_ps_dtfe.sh's
    OUTPUT_PREFIX writes alternate prefixes); a prefix with no grids raises, naming it;
    'ps_mw' and 'ps_mw.' are equivalent; the default set stays reachable side by side.
    """
    import shutil
    import tempfile

    from dtfelib.io import FieldSet
    try:
        import h5py
    except ImportError as e:              # FieldSet needs a combined_*.hdf5 for units
        raise FileNotFoundError(f"h5py unavailable: {e}")   # -> SKIP, not FAIL
    d = Path(tempfile.mkdtemp(prefix="dtfelib_prefix_"))
    try:
        n = 8
        rng = np.random.default_rng(23)
        den_out = np.abs(rng.normal(1, 0.1, (n, n, n))).astype(np.float32)
        den_mw = np.abs(rng.normal(1, 0.3, (n, n, n))).astype(np.float32)
        assert not np.array_equal(den_out, den_mw)
        den_out.tofile(d / "ps_output.den")
        den_mw.tofile(d / "ps_mw.den")
        with h5py.File(d / "combined_000.hdf5", "w") as f:
            h = f.create_group("Header")
            h.attrs["BoxSize"] = 100000.0
            h.attrs["NumPart_ThisFile"] = np.array([0, n**3, 0, 0, 0, 0], dtype=np.int64)
            h.attrs["MassTable"] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            h.attrs["Redshift"] = 0.0
            h.attrs["HubbleParam"] = 0.7
            h.attrs["HFreeUnits"] = 1

        fs_mw = FieldSet(d, averaged=False, prefix="ps_mw")
        assert fs_mw.method == "ps", f"prefix must imply method 'ps', got {fs_mw.method!r}"
        assert np.array_equal(fs_mw.load("density"), den_mw), "prefix set must read ps_mw.den"
        assert ", prefix='ps_mw.'" in repr(fs_mw), "repr must show a non-default prefix"
        fs_dot = FieldSet(d, averaged=False, prefix="ps_mw.")
        assert np.array_equal(fs_dot.load("density"), den_mw), "'ps_mw.' must equal 'ps_mw'"
        fs_def = FieldSet(d, averaged=False)   # default set untouched next to the alternate
        assert np.array_equal(fs_def.load("density"), den_out), "default must read ps_output.den"
        assert ", prefix=" not in repr(fs_def), "repr must not show the default prefix"
        try:
            FieldSet(d, averaged=False, prefix="ps_nope")
        except FileNotFoundError as e:
            assert "ps_nope" in str(e), f"error must name the missing prefix: {e}"
        else:
            raise AssertionError("a prefix with no grids on disk must raise FileNotFoundError")
    finally:
        shutil.rmtree(d)


def main():
    print("=" * 60)
    print(" dtfelib merger-tree / void-tracking tests")
    print("=" * 60)
    print("synthetic:")
    check("sample_grid conventions", t_sample_grid)
    check("periodic helpers", t_periodic_helpers)
    check("track_center across box wrap", t_track_center_wrap)
    check("match_catalog_void (periodic)", t_match_catalog_void)
    check("shape estimators: vectorized == loop reference", t_shape_estimators)
    check("FieldSet caustic loader (synthetic)", t_caustic_field_loader)
    check("velocity_single_stream float-streams mask (synthetic)", t_velocity_single_stream_float_mask)
    check("FieldSet velocity_scale u-units conversion (synthetic)", t_velocity_scale)
    check("FieldSet out-of-core loading: auto/memmap/slice/slabs (synthetic)", t_outofcore_loading)
    check("FieldSet alternate OUTPUT_PREFIX set (synthetic)", t_alternate_prefix)
    print(f"data-backed ({SIM}):")
    check("main_branch invariants", t_main_branch_invariants)
    check("descendant_branch inverse walk", t_descendant_inverse)
    check("merger row arithmetic on real trees", t_merger_row_arithmetic)
    check("mergers_along_branch API", t_mergers_api)
    check("groupcat membership + particle ranges", t_groupcat_membership)
    check("environment box frame", t_environment_box)
    check("FieldSet velocity_single_stream (TNG50-4-Dark)", t_velocity_single_stream)
    print("-" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
