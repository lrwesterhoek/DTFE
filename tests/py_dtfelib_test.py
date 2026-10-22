#!/usr/bin/env python3
"""Tests for the Python merger-tree / void-tracking stack (dtfelib).

Two tiers:
  * SYNTHETIC (always run): grid sampling conventions, periodic helpers, void-centre
    tracking across the box wrap, catalog matching.
  * DATA-BACKED (run when the simulation data is on disk, else skipped with a note):
    SubLink invariants on the real TNG50-3-Dark trees -- branch monotonicity, the
    contiguous-row arithmetic, merger-event descendant links -- the Subfind
    group-membership invariant behind subhalo_particle_range, the FieldSet
    single-stream velocity mask on the TNG50-4-Dark raw PS grids, and the caustic
    skeleton on the crossed-waves runs tests/ps_caustic_class_check.sh leaves.

Usage:  python3 tests/py_dtfelib_test.py            (exit 0 = pass)
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from _harness import check as _check, setup, summary

# the figure root is inherited by the scripts run in fresh interpreters
setup("dtfelib_figs_", paths=("python",), mpl_agg=False)
os.environ.setdefault("DTFE_SIM", "TNG50-4-Dark")    # the box these synthetic tests stand for; a data-tier run sets its own
DATA_BACKED = False     # main() sets it before the tier that needs DTFE_DATA_ROOT's data: only THERE is a missing file a skip


def check(name, fn):
    """A missing file or key is a SKIP in the data-backed tier, a FAILURE in the synthetic one."""
    _check(name, fn, missing=(FileNotFoundError, KeyError), data_backed=DATA_BACKED)


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


def t_threaded_filters():
    """smooth_field and find_local_minima run their separable 1-D passes slab-threaded on a big field: the
    SAME bits as scipy's serial gaussian_filter / minimum_filter (float32 and float64, wrap and reflect, sigma below
    and above a cell, the cross-box sigma, a 2-D plane, a NaN), and the threaded path is the one taken."""
    from scipy import ndimage
    from dtfelib import fields
    rng = np.random.default_rng(4)
    cases = [(np.float32, (132, 128, 128)), (np.float64, (128, 136, 130)), (np.float32, (1536, 1400))]
    serial_g, serial_m = fields.gaussian_filter, fields.minimum_filter
    def refused(*a, **k):
        raise AssertionError("the serial scipy filter ran on a field above _THREAD_MIN")
    for dt, shape in cases:
        f = rng.standard_normal(shape).astype(dt)
        f.flat[777] = np.nan
        assert f.size >= getattr(fields, "_THREAD_MIN", 1 << 21)
        want = {(sig, mode): ndimage.gaussian_filter(f, sigma=sig, mode=mode)
                for sig in (0.7, 5.0, 21.4291) for mode in ("wrap", "reflect")}
        want_min = {fp: f == ndimage.minimum_filter(f, size=fp, mode="wrap") for fp in (3, 11)}
        fields.gaussian_filter = fields.minimum_filter = refused
        try:
            for (sig, mode), w in want.items():
                got = fields.smooth_field(f, sig, mode=mode)
                assert got.dtype == w.dtype and np.array_equal(got, w, equal_nan=True), (dt, shape, sig, mode)
            for fp, w in want_min.items():
                mask, coords = fields.find_local_minima(f, footprint_size=fp)
                assert np.array_equal(mask, w) and np.array_equal(coords, np.argwhere(w)), (dt, shape, fp)
        finally:
            fields.gaussian_filter, fields.minimum_filter = serial_g, serial_m
    small = rng.standard_normal((20, 20, 20)).astype(np.float32)          # below _THREAD_MIN: scipy's own call
    assert np.array_equal(fields.smooth_field(small, 2.0), ndimage.gaussian_filter(small, 2.0, mode="wrap"))


def t_threaded_critical_points():
    """pipeline.find_critical_points runs its three box filters (the minima, the maxima, the gradient
    minima) slab-threaded through fields.box_filter: on a synthetic 256^3 field with plateaus (ties between neighbours
    decide the masks) every output is array_equal to the same function on scipy's serial minimum_filter /
    maximum_filter, box_filter's maximum is scipy's to the bit, and no serial filter runs above _THREAD_MIN."""
    from scipy import ndimage
    from dtfelib import fields, pipeline
    import config
    rng = np.random.default_rng(11)
    f = fields.smooth_field(rng.standard_normal((256, 256, 256)).astype(np.float32), 2.0)
    f = (np.round(f * 40) / 40).astype(np.float32)          # plateaus: equal neighbours decide the masks
    g = (np.abs(np.roll(f, 1, 0) - f) + np.abs(np.roll(f, 1, 2) - f)).astype(np.float32)   # a |grad| stand-in
    hess = {"hxx": f, "hyy": g, "hzz": -f, "hxy": g, "hxz": f, "hyz": -g}
    fp = config.FOOTPRINT_SIZE
    assert np.array_equal(fields.box_filter(f, fp, maximum=True), ndimage.maximum_filter(f, size=fp, mode="wrap"))
    real = fields.box_filter
    def serial(field, size, mode="wrap", maximum=False):    # the reference: scipy's serial filters
        return (ndimage.maximum_filter if maximum else ndimage.minimum_filter)(field, size=size, mode=mode)
    fields.box_filter = serial
    try:
        want = pipeline.find_critical_points(f, hess, g)
    finally:
        fields.box_filter = real
    def refused(*a, **k):
        raise AssertionError("a serial scipy box filter ran on a field above _THREAD_MIN")
    saved = {(m, n): getattr(m, n) for m in (fields, pipeline) for n in ("minimum_filter", "maximum_filter")
             if hasattr(m, n)}
    for (m, n) in saved:
        setattr(m, n, refused)
    try:
        got = pipeline.find_critical_points(f, hess, g)
    finally:
        for (m, n), v in saved.items():
            setattr(m, n, v)
    assert sum(len(want[k]["coords"]) for k in ("minima", "maxima", "saddle1", "saddle2")) > 0
    for k in ("minima", "maxima", "saddle1", "saddle2"):
        for part in ("coords", "values", "eigenvalues"):
            assert np.array_equal(got[k][part], want[k][part]), (k, part)
    assert got["field_mean"] == want["field_mean"] and got["field_std"] == want["field_std"]


def t_shape_estimators():
    """Vectorized BBKS/ellipsoid-fit estimators must reproduce the original per-void
    loops bit-for-bit: NaN rows (trace/lambda1 thresholds, sanity gate), the ellipsoid cuts
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
        cell, cuts = pipeline.config.CELL_SIZE, pipeline.ellipsoid_cuts_mpc()
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
    # The reference loop's scalar arithmetic ('1.0 / np.sqrt(float32 scalar)', '2 * tr') stays float32 under numpy 2's
    # promotion rules (NEP 50), as the vectorized code's arrays do; numpy 1.x (CI's Ubuntu 1.26) promotes it to
    # float64, and the BBKS ratios divide by a trace that can nearly cancel, so no tolerance fits. numpy 1.24-1.26 can
    # opt into NEP 50 (np._set_promotion_state('weak'), the transition switch): the reference runs under it, and the
    # comparison stays bit-for-bit on both.
    def nep50(fn, *args):
        get = getattr(np, "_get_promotion_state", None)
        if get is None or get() == "weak":
            return fn(*args)
        old = get()
        np._set_promotion_state("weak")
        try:
            return fn(*args)
        finally:
            np._set_promotion_state(old)
    for ev in (ev64, ev64.astype(np.float32)):
        got = fields.calculate_shape_parameters(list(ev))
        ref_ar, ref_bbks = nep50(ref_shapes, list(ev))
        assert np.array_equal(got["axis_ratios"], ref_ar, equal_nan=True), "axis_ratios drifted"
        assert np.array_equal(got["bbks_params"], ref_bbks, equal_nan=True), "bbks_params drifted"

        n = len(ev)
        coords = rng.integers(0, 64, (n, 3))
        evecs = rng.normal(0.0, 1.0, (n, 3, 3)).astype(ev.dtype)
        got_f = pipeline._ellipsoid_fits(coords, ev, evecs)
        ref_f = nep50(ref_fits, coords, ev, evecs)
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
    from dtfelib.cli import sim_dir
    from dtfelib.io import FieldSet
    snapdir = sim_dir("TNG50-4-Dark") / "snapdir_099"
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
    cells land on 1.0 +/- float32 eps rather than exactly 1. A 'streams != 1' mask would
    NaN essentially the whole grid; the tolerance mask must keep every ~1.0 cell.
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
        # onto 1.0 and let a '!= 1' mask pass). Plus a genuine multi-stream slab at the
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
            f"single-stream cells (a '!= 1' test keeps 0)")
        assert np.array_equal(v1[single], vel[single]), "kept velocities must be untouched"
        assert np.isnan(v1[~single]).all(), "multi-stream cells must be fully NaN"

        # '.hidden_streams': bit 1 = multi-stream volume '.streams' does not show, so a cell with
        # it set is masked although its '.streams' reads 1. Bit 2 (a never-sampled tet's mass,
        # usually of the SAME stream) must NOT mask on its own. Legacy outputs name it
        # '.unresolved' with 0/1 = a mass flag, read as bit 1 (conservative): the
        # legacy name is exercised first, then the current one. Without either file (older
        # outputs, whose '.streams' counted those tets) the streams test alone decides -- above.
        unres = np.zeros((n, n, n), dtype=np.float32)
        unres[0, :4] = 1.0                 # bit 1 inside the single-stream rows (or an old 0/1 flag)
        unres[1, :4] = 2.0                 # bit 2 only: same-stream sub-sample mass, kept
        unres[1, 4:6] = 3.0                # both bits: masked
        unres[2, 0] = 1.0                  # already multi-stream: masked either way
        old = (unres != 0).astype(np.float32)          # a legacy output: 0/1 under the old name
        old.tofile(d / "ps_output.unresolved")
        fs = FieldSet(d, method="ps", averaged=False)
        assert fs.has("hidden_streams"), "the old '.unresolved' file must be found under the new name"
        kept = ~np.isnan(fs.velocity_single_stream()).any(axis=-1)
        assert np.array_equal(kept, single & (old == 0)), "an old 0/1 flag must mask conservatively"
        unres.tofile(d / "ps_output.hidden_streams")    # the current name wins over the old one
        fs = FieldSet(d, method="ps", averaged=False)
        v1 = fs.velocity_single_stream()
        keep = single & ((unres.astype(np.int32) & 1) == 0)
        assert 0 < keep.sum() < single.sum(), "fixture must flag some single-stream cells"
        assert (single & (unres == 2.0)).any(), "fixture must hold bit-2-only single-stream cells"
        kept = ~np.isnan(v1).any(axis=-1)
        assert np.array_equal(kept, keep), (
            f"bit-1 cells must be masked and bit-2-only cells kept: kept {kept.sum()}, expected {keep.sum()}")
        assert np.array_equal(v1[keep], vel[keep]), "kept velocities must be untouched"
    finally:
        shutil.rmtree(d)


def t_sim_dir_layouts():
    """dtfelib.cli.sim_dir / find_sims resolve BOTH data layouts exactly like config.sh's sim_dir:
    per family <root>/<family>/<sim> (e.g. ~/IllustrisTNG/TNG100/TNG100-3-Dark), and flat <root>/<sim> (~/output), under a
    root whose name has a space; an existing flat directory wins, a new simulation goes beside its
    family, a name without a family stays flat."""
    import shutil
    import tempfile

    from dtfelib.cli import find_sims, sim_dir
    root = Path(tempfile.mkdtemp(prefix="dtfelib_layout_")) / "Illustris TNG"
    try:
        (root / "TNG100" / "TNG100-3-Dark" / "snapdir_004").mkdir(parents=True)
        (root / "TNG100" / "TNG100-3-Dark" / "snapdir_004" / "combined_004.hdf5").touch()
        (root / "FLAT-1-Dark" / "snapdir_099").mkdir(parents=True)
        (root / "FLAT-1-Dark" / "snapdir_099" / "combined_099.hdf5").touch()
        assert sim_dir("TNG100-3-Dark", root) == root / "TNG100" / "TNG100-3-Dark"
        assert sim_dir("TNG100-1-Dark", root) == root / "TNG100" / "TNG100-1-Dark", "new sim beside its family"
        assert sim_dir("FLAT-1-Dark", root) == root / "FLAT-1-Dark", "an existing flat dir wins"
        assert sim_dir("NoFamily", root) == root / "NoFamily"
        (root / "TNG100-3-Dark").mkdir()                     # flat AND family: flat wins
        assert sim_dir("TNG100-3-Dark", root) == root / "TNG100-3-Dark"
        found = find_sims(root, "snapdir_*/combined_*.hdf5")
        assert found == ["FLAT-1-Dark", "TNG100-3-Dark"], found
    finally:
        shutil.rmtree(root.parent)


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


def t_limits_store():
    """analyze.py compute's cross-epoch limits survive a run on a subset or on nothing."""
    import json, shutil, tempfile
    import config
    from dtfelib import pipeline
    old, old_fl = config.CACHE_DIR, config.FIELD_LIMITS
    config.CACHE_DIR = Path(tempfile.mkdtemp(prefix="dtfe_limits_"))     # never the real cache
    config.FIELD_LIMITS = {k: v for k, v in old_fl.items() if k != "delta"}   # a pinned 'delta' would bypass the store
    try:
        canon = ["050", "099"]
        lim, cov, miss = pipeline.record_limits({"50": {"delta": 2.0, "potential": 5.0}, 99: {"delta": 3.0, "potential": 1.0}}, canon)
        assert lim == {"delta": 3.0, "potential": 5.0} and cov == canon and miss == [], (lim, cov, miss)
        assert pipeline.load_global_limits() == lim
        lim2, cov2, _ = pipeline.record_limits({"099": {"delta": 0.5, "potential": 0.5}}, canon)
        assert lim2 == {"delta": 2.0, "potential": 5.0}, lim2          # 050's maxima survive a lower 099 rerun
        assert pipeline.load_snapshot_limits()["099"] == {"delta": 0.5, "potential": 0.5}
        lim3, cov3, miss3 = pipeline.record_limits({}, canon + ["004"])
        assert lim3 is None and pipeline.load_global_limits() == lim2 and miss3 == ["004"], (lim3, miss3)
        lim4, _, _ = pipeline.record_limits({"004": {"delta": float("nan"), "potential": 9.0}}, canon + ["004"])
        assert lim4 == {"delta": 2.0, "potential": 9.0}, lim4          # NaN amplitudes are not stored
        sp = config.CACHE_DIR / pipeline._param_hash() / f"snapshot_limits_{pipeline.require_sim()}.json"
        assert sp.is_file(), f"no per-simulation store {sp.name}"          # (a FileNotFoundError would SKIP, not fail)
        assert set(json.loads(sp.read_text())) == {"050", "099", "004"}
        assert pipeline.canonical_snapshot_id(99) == "099" and pipeline.canonical_snapshot_id("abc") == "abc"
        # one store per simulation: a second simulation's run neither reads nor rewrites the first's
        limb, covb, missb = pipeline.record_limits({"099": {"delta": 0.25, "potential": 0.5}}, canon, sim="SimB")
        assert limb == {"delta": 0.25, "potential": 0.5} and covb == ["099"] and missb == ["050"], (limb, covb, missb)
        assert pipeline.load_global_limits() == lim4 and pipeline.load_global_limits("SimB") == limb
        assert pipeline.series_vmax("delta", sim="SimB") == 0.25 and pipeline.series_vmax("delta") == 2.0
        assert (config.CACHE_DIR / pipeline._param_hash() / "global_limits_SimB.json").is_file()
        from dtfelib import figures as style
        arr = np.array([1.0, 2.0, 7.0])                                     # nothing recorded for that simulation: the data's own
        assert pipeline.series_vmax("delta", arr, sim="NoSuchSim") == style.robust_vmax(arr) > 2.0
    finally:
        shutil.rmtree(config.CACHE_DIR, ignore_errors=True)
        config.CACHE_DIR, config.FIELD_LIMITS = old, old_fl


def t_map_limits():
    """A diverging map centred on 0, a log map that skips an empty slice (instead of crashing plot_DTFE)."""
    from dtfelib import figures as style
    skew = np.concatenate([np.full(90, 1.0), np.full(10, -8.0)])
    lo, hi = style.symmetric_limits(skew)
    assert lo == -hi and hi > 0, (lo, hi)
    assert style.symmetric_limits(np.zeros(16)) == (-1.0, 1.0)
    assert style.symmetric_limits(np.array([np.nan, np.inf, 0.0])) == (-1.0, 1.0)
    sparse = np.zeros(10000); sparse[:10] = 1e-3; sparse[3] = -2e-3     # 0.1% non-zero: its percentile is 0
    assert style.symmetric_limits(sparse) == (-2e-3, 2e-3), style.symmetric_limits(sparse)   # the data's own scale, not (-1, 1)
    assert style.symmetric_limits(np.array([-2.0, 2.0, np.nan]), pct=100) == (-2.0, 2.0)
    assert style.log_limits(np.zeros(8)) is None and style.log_limits(np.array([np.nan, -1.0])) is None
    assert style.log_limits(np.array([1e-9, 4.0, np.nan, -3.0])) == (1e-6, 4.0)
    vmin, vmax = style.log_limits(np.array([2.0, 2.0]))
    assert vmin == 2.0 and vmax > vmin


def t_webstreams():
    """Stream bins and caustic class by web class on a synthetic grid with known fractions."""
    from dtfelib import webstreams as ws
    from dtfelib.io import CAUSTIC_PARITY_POS, CAUSTIC_PARITY_NEG, CAUSTIC_COLLAPSE
    n = 8
    web = np.zeros((n, n, n), dtype=np.float32)
    web[2:4] = 1.0; web[4:6] = 2.0; web[6:] = 3.0           # two planes per class along axis 0
    web[0, 0, 0] = 7.0                                      # a label outside 0..3 -> 'other'
    streams = np.ones((n, n, n), dtype=np.float32)
    streams[0, :, :2] = 0.0                                 # empty cells in the void planes (16 cells)
    streams[1, 0, 0] = 1.0004                               # within the tolerance: single
    streams[2, :, 0] = 1.5                                  # a fractional averaged value: 1 < s <= 3 (8 cells)
    streams[4, 0, :] = 3.0                                  # 3.0 is in 1 < s <= 3 (8 cells)
    streams[4, 1, :] = 4.2                                  # 3 < s <= 5 (8 cells)
    streams[6, 0, :] = 9.0                                  # s > 5 (8 cells)
    hidden = np.zeros((n, n, n), dtype=np.float32)
    hidden[1, 1, 1] = 1.0                                   # bit 1 on a single cell: 'hidden multi'
    hidden[1, 1, 2] = 2.0                                   # bit 2 only: still single
    density = np.ones((n, n, n), dtype=np.float32)
    density[:2] = 0.5                                       # voids lighter: mass differs from volume
    density[6:] = 4.0
    density[2, 0, 0] = 3.0                                  # the wall's fold cell (s = 1.5) heavier: mass != volume WITHIN a class
    density[6, 0, 0] = 8.0                                  # the node's k=3 cell (s = 9) too
    cc = np.zeros((n, n, n), dtype=np.int32)
    cc[2, 0, 0] = CAUSTIC_PARITY_POS | CAUSTIC_PARITY_NEG | CAUSTIC_COLLAPSE[1]    # a fold, k = 1
    cc[2, 0, 1] = CAUSTIC_PARITY_POS | CAUSTIC_COLLAPSE[0]                         # one parity: no fold, k = 0
    cc[6, 0, 0] = CAUSTIC_COLLAPSE[3] | CAUSTIC_COLLAPSE[1]                        # k = 3 (the highest bit wins)
    t = ws.environment_table(web, streams, hidden=hidden, density=density, caustic_class=cc, thickness=3)
    c = t["classes"]
    assert list(c) == ["void", "wall", "filament", "node", "other"], list(c)
    assert c["other"]["cells"] == 1 and c["void"]["cells"] == 2 * 64 - 1 and c["wall"]["cells"] == 128, (c["void"]["cells"], c["other"]["cells"])
    assert abs(sum(e["volume_fraction"] for e in c.values()) - 1.0) < 1e-12
    assert abs(sum(e["mass_fraction"] for e in c.values()) - 1.0) < 1e-12
    # void: 127 cells ([0,0,0] is 'other'): 15 empty, 1 hidden multi, 111 single
    v = c["void"]["stream_bins"]
    assert abs(v["empty"]["volume"] - 15 / 127) < 1e-12 and abs(v["single"]["volume"] - 111 / 127) < 1e-12, (v["empty"], v["single"])
    assert abs(v["hidden multi"]["volume"] - 1 / 127) < 1e-12 and v["s > 5"]["volume"] == 0.0
    assert c["other"]["stream_bins"]["empty"]["volume"] == 1.0
    w = c["wall"]["stream_bins"]
    assert abs(w["1 < s <= 3"]["volume"] - 8 / 128) < 1e-12 and abs(w["single"]["volume"] - 120 / 128) < 1e-12
    f = c["filament"]["stream_bins"]
    assert abs(f["1 < s <= 3"]["volume"] - 8 / 128) < 1e-12 and abs(f["3 < s <= 5"]["volume"] - 8 / 128) < 1e-12
    assert abs(c["node"]["stream_bins"]["s > 5"]["volume"] - 8 / 128) < 1e-12
    # mass weights, against boolean indexing of the same arrays (another code path than the bincounts)
    wcls = ws.web_classes(web)
    # the expectations in float64: a float32 sum ratio rounds at ~5e-9, and under numpy 1.x (CI's Ubuntu 1.26)
    # 'python float - float32 scalar' is float64 (numpy 2's NEP 50 keeps it float32)
    d64 = density.astype(np.float64)
    total_m = float(d64.sum())
    for name, lab in (("void", 0), ("wall", 1), ("node", 3)):
        assert abs(c[name]["mass_fraction"] - d64[wcls == lab].sum() / total_m) < 1e-9, name
    assert abs(c["void"]["stream_bins"]["single"]["mass"] - 111 / 127) < 1e-12      # uniform density within the void
    wall, node = wcls == 1, wcls == 3
    assert abs(c["wall"]["stream_bins"]["1 < s <= 3"]["mass"] - d64[wall & (streams == 1.5)].sum() / d64[wall].sum()) < 1e-9
    assert abs(c["node"]["stream_bins"]["s > 5"]["mass"] - d64[node & (streams == 9.0)].sum() / d64[node].sum()) < 1e-9
    assert abs(c["node"]["mean_streams"] - (8 * 9.0 + 120) / 128) < 1e-9
    assert abs(c["node"]["mean_streams_mass"] - (d64[node] * streams[node]).sum() / d64[node].sum()) < 1e-9
    assert c["node"]["mean_streams_mass"] > c["node"]["mean_streams"]                   # the heavy cell has 9 streams
    # caustic class: the wall plane 2 holds one fold (k=1) and one k=0 cell; the node plane 6 one k=3 -- in both weightings
    # the fold is THE geometric one (io.caustic_is_fold with streams and hidden): the mask's fold bits OR
    # streams > 1 + tol OR hidden bit 1 -- the wall's eight s = 1.5 cells (one of them the mask's fold cell, density 3)
    # and the void's hidden-bit cell, which the mask's bits alone (1 wall cell, no void cell) undercount
    from dtfelib.io import caustic_is_fold
    gfold = caustic_is_fold(cc, streams=streams, hidden=hidden)
    void = wcls == 0
    assert abs(c["wall"]["fold_fraction"]["volume"] - 8 / 128) < 1e-12 and abs(c["void"]["fold_fraction"]["volume"] - 1 / 127) < 1e-12
    assert abs(c["wall"]["fold_fraction"]["mass"] - 10.0 / 130.0) < 1e-12 and abs(c["void"]["fold_fraction"]["mass"] - 1 / 127) < 1e-12
    assert abs(c["wall"]["fold_fraction"]["mass"] - d64[wall & gfold].sum() / d64[wall].sum()) < 1e-12
    assert abs(c["void"]["fold_fraction"]["mass"] - d64[void & gfold].sum() / d64[void].sum()) < 1e-12
    cb = c["wall"]["collapse"]
    assert abs(cb["k=1"]["volume"] - 1 / 128) < 1e-12 and abs(cb["k=0"]["volume"] - 1 / 128) < 1e-12 and abs(cb["none"]["volume"] - 126 / 128) < 1e-12
    assert abs(cb["k=1"]["mass"] - 3.0 / d64[wall].sum()) < 1e-12 and abs(cb["k=0"]["mass"] - 1.0 / d64[wall].sum()) < 1e-12
    assert abs(cb["none"]["mass"] - (d64[wall].sum() - 4.0) / d64[wall].sum()) < 1e-12
    assert abs(c["node"]["collapse"]["k=3"]["volume"] - 1 / 128) < 1e-12 and abs(c["node"]["collapse"]["k=3"]["mass"] - 8.0 / d64[node].sum()) < 1e-12
    # the bins' edges and the hidden bit, cell by cell (tol = STREAM_TOL = 1e-3)
    from dtfelib.io import STREAM_TOL
    s_ = np.array([1.0, 1.5, 0.0, 1.0 - 4e-4, np.nextafter(np.float32(1), np.float32(0)), 0.5, 2 / 3, 3.0005, 3.002, 5.0, 5.0005, 5.002, 1.0], np.float32)
    h_ = np.array([3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], np.float32)
    assert list(ws.stream_bins(s_, h_, STREAM_TOL)) == [2, 3, 0, 1, 1, 0, 0, 3, 4, 4, 4, 5, 1], list(ws.stream_bins(s_, h_))
    assert list(ws.stream_bins(s_[:3], None)) == [1, 3, 0]
    assert list(ws.web_classes(np.array([2.6, 2.4, -0.6, 3.6, 0.0], np.float32))) == [3, 2, 4, 4, 0]
    # chunked == unchunked (an odd thickness that does not divide n), and memmaps are fine
    t2 = ws.environment_table(web, streams, hidden=hidden, density=density, caustic_class=cc, thickness=8)
    assert t2 == t, "chunking changed the table"
    # without the caustic grid and without weights: those entries are None, nothing else changes
    t3 = ws.environment_table(web, streams, hidden=hidden)
    assert t3["classes"]["wall"]["fold_fraction"] is None and t3["classes"]["wall"]["collapse"] is None
    assert t3["classes"]["wall"]["mass_fraction"] is None and t3["classes"]["wall"]["stream_bins"]["single"]["mass"] is None
    assert t3["classes"]["wall"]["stream_bins"]["single"]["volume"] == c["wall"]["stream_bins"]["single"]["volume"]
    # without the hidden grid the hidden cell is single
    t4 = ws.environment_table(web, streams, thickness=5)
    assert abs(t4["classes"]["void"]["stream_bins"]["single"]["volume"] - 112 / 127) < 1e-12
    txt = ws.table_text(t, "synthetic")
    assert "void" in txt and "hidden multi" in txt and "k=3" in txt and "caustic class: yes" in txt
    ssf = ws.single_stream_fraction(t)
    assert abs(ssf["void"] - 111 / 127) < 1e-12 and abs(ssf["wall"] - 120 / 128) < 1e-12
    # save_table appends its suffixes: 'x_z0.50' and 'x_z0.00' are two files, not one
    import json, shutil, tempfile
    d = Path(tempfile.mkdtemp(prefix="dtfe_ws_"))
    try:
        ws.save_table(t, d / "x_z0.50", "a"); ws.save_table(t3, d / "x_z0.00", "b")
        assert sorted(q.name for q in d.iterdir()) == ["x_z0.00.json", "x_z0.00.txt", "x_z0.50.json", "x_z0.50.txt"], sorted(q.name for q in d.iterdir())
        assert json.loads((d / "x_z0.50.json").read_text())["classes"]["wall"]["fold_fraction"]["volume"] == c["wall"]["fold_fraction"]["volume"]
        assert (d / "x_z0.00.txt").read_text().startswith("b\n=\n")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_void_catalog_export():
    """The void catalogue as CSV + HDF5 with units and provenance, from a catalogue the
    real pipeline built on a synthetic field: every column against its source in the dict, the eigenvalues in the
    Mpc^-2 they are labelled with, the units read back from both files, every provenance value, a stdlib csv parse,
    the frame guard (box, grid, no header)."""
    import shutil, tempfile, types
    import config
    from dtfelib import pipeline, catalog
    from dtfelib import fields as dtfe
    d = Path(tempfile.mkdtemp(prefix="dtfe_cat_"))
    old_cuts, old_cache, old_deep = config.ELLIPSOID_CUT_RULES, config.CACHE_DIR, config.DEEP_VOID_THRESHOLD
    # cuts no other provenance entry shares (by default min_axis = the gradient threshold = 0.1 and max_axis =
    # max_ratio = 10), a deep threshold off its default (a constant -0.1 would pass at the default): a swapped or
    # hard-coded provenance value cannot pass by coincidence. Set before the build: cuts_mpc / deep_threshold agree
    # (rules: 1.1 cells, 9.5 sigma, L/2, 7.25 -- about 0.11 / 9.5 Mpc)
    config.ELLIPSOID_CUT_RULES = {"min_axis_cells": 1.1, "max_axis_sigma": 9.5, "max_axis_box_frac": 0.5, "max_axis_ratio": 7.25}
    config.DEEP_VOID_THRESHOLD = -0.12
    config.CACHE_DIR = d / "cache"                      # a real SnapshotProducts below: never python/cache
    try:
        n = 32
        rng = np.random.default_rng(7)
        delta = 0.02 * rng.normal(size=(n, n, n)).astype(np.float32)
        idx = np.indices((n, n, n))
        for c, w, dd in (((8, 8, 8), 2.0, 0.9), ((20, 10, 24), 3.5, 0.8), ((6, 24, 14), 5.0, 0.7)):   # three dips
            r2 = sum((((idx[k] - c[k] + n // 2) % n) - n // 2) ** 2 for k in range(3))
            delta -= (dd * np.exp(-0.5 * r2 / w ** 2)).astype(np.float32)
        delta_s = dtfe.smooth_field(delta, sigma=1.0)
        cat = pipeline.build_void_catalog(delta_s, pipeline.hessian_from_field(delta_s))
        n_voids = len(cat["coords"])
        assert n_voids >= 3, n_voids
        cat["well_resolved"] = cat["well_resolved"].copy()
        cat["well_resolved"][0] = False                     # at least one unresolved void whatever the field gives
        assert cat["well_resolved"].any(), "no resolved void in the synthetic field"
        cat["delta_values"] = cat["delta_values"].copy()
        cat["delta_values"][1] = np.float32(-0.05)          # one void above the deep threshold: 'deep' takes both values
        cat["deep"] = cat["delta_values"] < config.DEEP_VOID_THRESHOLD

        class _FS:
            method, prefix = "dtfe", "output."
            def __init__(self, box, grid_n):
                self.grid_n = grid_n
                # a header always (no header = a real FieldSet raising), with its own redshift float: the provenance's
                # is the products' (config's, the one analyze passes, in the cache key and -- via prov -- the file name)
                self.meta = types.SimpleNamespace(box_mpc=box, redshift=0.4997)

        class _Prod:
            sim, snapshot, prefix, redshift = "SYN", "050", None, 0.5
            def __init__(self, box, grid_n=n):
                self.fs = _FS(box, grid_n)
            def voids(self):
                return cat

        _t_void_catalog_export_body(d, cat, n_voids, n, _Prod, catalog, pipeline, config)
    finally:
        config.ELLIPSOID_CUT_RULES, config.CACHE_DIR, config.DEEP_VOID_THRESHOLD = old_cuts, old_cache, old_deep
        shutil.rmtree(d, ignore_errors=True)


def _t_void_catalog_export_body(d, cat, n_voids, n, _Prod, catalog, pipeline, config):
    import csv, re
    from datetime import datetime, timezone
    import h5py
    paths = catalog.export_voids(_Prod(config.CELL_SIZE * n), d)
    assert [q.name for q in paths] == ["voids_SYN_dtfe_050_z0.50.csv", "voids_SYN_dtfe_050_z0.50.h5"], [q.name for q in paths]
    names = [c[0] for c in catalog.COLUMNS]
    unit = {c[0]: c[1] for c in catalog.COLUMNS}
    # the labels the per-Mpc^2 pin below holds the values to: eigenvalues per Mpc^2, lengths in Mpc
    assert unit["lambda1"] == unit["lambda2"] == unit["lambda3"] == "Mpc^-2" and unit["a_mpc"] == unit["x_mpc"] == "Mpc", unit
    # the units and descriptions as the files carry them: every voids/<name> attribute and every '# name [unit]: desc'
    # line of the CSV (the last len(COLUMNS) comment lines; a description may hold ':' itself)
    with h5py.File(paths[1], "r") as f:
        assert sorted(f["voids"].keys()) == sorted(names), sorted(f["voids"].keys())
        got = [(nm, str(f["voids"][nm].attrs["unit"]), str(f["voids"][nm].attrs["description"])) for nm in names]
    assert got == list(catalog.COLUMNS), [g for g, c in zip(got, catalog.COLUMNS) if g != c][:2]
    lines = paths[0].read_text().splitlines()
    comments = [ln for ln in lines if ln.startswith("#")]
    pat = re.compile(r"# (\S+)(?: \[([^\]]*)\])?: (.*)")
    described = [pat.fullmatch(ln) for ln in comments[-len(names):]]
    assert all(described), [ln for ln, m in zip(comments[-len(names):], described) if not m][:2]
    described = [(m.group(1), m.group(2) or "", m.group(3)) for m in described]
    assert described == list(catalog.COLUMNS), [g for g, c in zip(described, catalog.COLUMNS) if g != c][:2]
    # a plain stdlib csv parse BEFORE the module's own reader (which a ragged row or a bad header crashes first):
    # the header is COLUMNS' names, unique, one row per void, every row as wide; no '#' in a data line (pandas'
    # comment='#' cuts a line at ANY '#'), every field a number
    data = [ln for ln in lines if not ln.startswith("#")]
    rows = list(csv.reader(data))
    assert rows[0] == names and len(set(rows[0])) == len(rows[0]), rows[0]
    assert len(rows) - 1 == n_voids and all(len(r) == len(names) for r in rows[1:]), (len(rows), [len(r) for r in rows[1:]])
    assert not any("#" in ln for ln in data), [ln for ln in data if "#" in ln][:1]
    for x in (x for r in rows[1:] for x in r):
        try:
            float(x)                                                 # 'nan' included: pandas' default NaN token
        except ValueError:
            raise AssertionError(f"non-numeric csv field {x!r}")
    try:
        import pandas as pd
        df = pd.read_csv(paths[0], comment="#")
        assert list(df.columns) == names and len(df) == n_voids
    except ImportError:
        print("    (pandas is not installed here: the pandas read_csv(comment='#') check skipped; the stdlib csv parse ran)")
    th, ph = catalog.read_hdf5(paths[1])
    tc, pc = catalog.read_csv(paths[0])
    assert list(ph["columns"]) == names and th["well_resolved"].dtype == bool
    # every column against its SOURCE in the catalogue dict, one map (so a swapped pair, a row taken for a column
    # or a constant flag cannot pass): the fit columns NaN where unresolved, the eigenvalues the cache's cell^-2
    # over cell^2, R_eff the float32 (a b c)^(1/3) of the population script
    res = np.asarray(cat["well_resolved"]).astype(bool)
    nanfit = np.where(res, 1.0, np.nan)
    f64 = lambda a: np.asarray(a, dtype=np.float64)
    ev = f64(cat["eigenvalues"]) / config.CELL_SIZE ** 2
    # the cell CENTRES: the cached positions_mpc are the corners, half a cell off what R_v, the profiles
    # and the outlines centre a void on; float64 throughout (the hdf5 compare is exact)
    pos = (f64(cat["coords"]) + 0.5) * config.CELL_SIZE
    assert np.allclose(pos - 0.5 * config.CELL_SIZE, f64(cat["positions_mpc"]), rtol=1e-6), "positions_mpc are the corners"
    axes, orient = f64(cat["semi_axes"]), f64(cat["orientations"])
    bbks, ratios = f64(cat["bbks_params"]), f64(cat["axis_ratios"])
    src = {"id": np.arange(n_voids), "ix": cat["coords"][:, 0], "iy": cat["coords"][:, 1], "iz": cat["coords"][:, 2],
           "x_mpc": pos[:, 0], "y_mpc": pos[:, 1], "z_mpc": pos[:, 2], "delta_c": f64(cat["delta_values"]),
           "lambda1": ev[:, 0], "lambda2": ev[:, 1], "lambda3": ev[:, 2],
           "e_bbks": bbks[:, 0], "p_bbks": bbks[:, 1], "b_over_a": ratios[:, 0], "c_over_a": ratios[:, 1],
           "well_resolved": res, "deep": np.asarray(cat["deep"]).astype(bool),
           "a_mpc": axes[:, 0] * nanfit, "b_mpc": axes[:, 1] * nanfit, "c_mpc": axes[:, 2] * nanfit,
           "r_eff_mpc": f64(np.where(res, np.prod(cat["semi_axes"], axis=1) ** (1.0 / 3.0), np.nan)),
           "r_v_mpc": f64(cat["r_v_cells"]) * config.CELL_SIZE,
           "r_v_delta08_mpc": f64(cat["r_v_delta08_cells"]) * config.CELL_SIZE,
           "r_v_ridge_mpc": f64(cat["r_v_ridge_cells"]) * config.CELL_SIZE,
           "distinct": catalog.distinct_mask(cat, res & np.isfinite(f64(cat["r_v_cells"])), config.FIELD_RESOLUTION),
           "ellipticity_fit": f64(cat["fit_ellipticity"]) * nanfit, "prolateness_fit": f64(cat["fit_prolateness"]) * nanfit,
           "major_x": orient[:, 0, 0] * nanfit, "major_y": orient[:, 1, 0] * nanfit, "major_z": orient[:, 2, 0] * nanfit,
           "minor_x": orient[:, 0, 2] * nanfit, "minor_y": orient[:, 1, 2] * nanfit, "minor_z": orient[:, 2, 2] * nanfit}
    assert list(src) == names, "the source map covers every column, in COLUMNS order"
    for name in names:
        for tag, t, rtol in (("hdf5", th, 0.0), ("csv", tc, 1e-8)):         # hdf5 exact, csv 9 digits
            if name in catalog._BOOL or name in catalog._INT:
                assert np.array_equal(t[name], np.asarray(src[name]).astype(t[name].dtype)), (tag, name, t[name], src[name])
            else:
                assert np.allclose(t[name], src[name], rtol=rtol, atol=0.0, equal_nan=True), (tag, name, t[name][:3], src[name][:3])
    assert th["deep"].any() and not th["deep"].all(), th["deep"]
    assert np.array_equal(th["deep"], th["delta_c"] < config.DEEP_VOID_THRESHOLD)
    # per Mpc^2 as labelled. The semi-axes are 2 cell / sqrt(|lambda_cell|) = 2 / sqrt(|lambda|) in these units, so
    # axis^2 |lambda| = 4 for each axis with its eigenvalue (a <-> the smallest |lambda|), in BOTH files; a factor cell^2
    # off reads 4 cell^2 -- which only differs from 4 when the cell is not ~1 Mpc
    assert abs(np.log(config.CELL_SIZE)) > 0.2, f"cell {config.CELL_SIZE} Mpc: the unit pin cannot tell Mpc from cells"
    for tag, t in (("hdf5", th), ("csv", tc)):
        lam = np.sort(np.abs(np.stack([t["lambda1"], t["lambda2"], t["lambda3"]], 1)), axis=1)[res]
        ax = np.stack([t["a_mpc"], t["b_mpc"], t["c_mpc"]], 1)[res]
        assert np.allclose(ax ** 2 * lam, 4.0, rtol=1e-5), (tag, ax[:2] ** 2 * lam[:2])
    assert (th["lambda1"] <= th["lambda2"]).all() and (th["lambda2"] <= th["lambda3"]).all()        # eigh order
    # void_table takes the frame from the dict, not from config: a catalogue in another cell (positions and semi-axes
    # x 0.47, its 'cell_mpc' saying so) still gives axis^2 |lambda| = 4; an empty one without a record is 0 rows
    other = dict(cat, cell_mpc=np.float64(0.47 * config.CELL_SIZE), positions_mpc=cat["positions_mpc"] * np.float32(0.47),
                 semi_axes=cat["semi_axes"] * np.float32(0.47))
    to = catalog.void_table(other)
    lam = np.sort(np.abs(np.stack([to["lambda1"], to["lambda2"], to["lambda3"]], 1)), axis=1)[res]
    ax = np.stack([to["a_mpc"], to["b_mpc"], to["c_mpc"]], 1)[res]
    assert np.allclose(ax ** 2 * lam, 4.0, rtol=1e-5), ("void_table in the dict's own cell", ax[:2] ** 2 * lam[:2])
    per_void = ("coords", "eigenvalues", "eigenvectors", "axis_ratios", "bbks_params", "delta_values", "deep", "well_resolved",
                "semi_axes", "orientations", "fit_ellipticity", "fit_prolateness", "positions_mpc", "r_v_cells",
                "r_v_delta08_cells", "r_v_ridge_cells")
    empty = {k: (np.asarray(v)[:0] if k in per_void else v) for k, v in cat.items() if k != "cell_mpc"}
    assert all(len(v) == 0 for v in catalog.void_table(empty).values())
    # the major axis IS the eigenvector of the smallest |lambda| (the longest semi-axis), the minor that of the largest
    vecs, ev_abs = f64(cat["eigenvectors"])[res], np.abs(f64(cat["eigenvalues"]))[res]
    for k, pick in (("major", np.argmin), ("minor", np.argmax)):
        axis = np.stack([th[f"{k}_x"], th[f"{k}_y"], th[f"{k}_z"]], 1)[res]
        want = vecs[np.arange(len(vecs)), :, pick(ev_abs, axis=1)]
        assert np.allclose(np.abs(np.einsum("ij,ij->i", axis, want)), 1.0, atol=1e-5), (k, np.einsum("ij,ij->i", axis, want))
    # every provenance value, in both files; the key SET equal (nothing dropped or added; hdf5 also lists 'columns')
    cuts = pipeline.ellipsoid_cuts_mpc()
    expected = {"catalogue": None, "sim": "SYN", "snapshot": "050", "redshift": 0.5, "method": "dtfe", "prefix": "output",
                "box_mpc": config.CELL_SIZE * n, "box_source": "header", "grid_n": n, "cell_mpc": config.CELL_SIZE,
                "sigma_cells": config.SMOOTHING_SIGMA_CELLS, "sigma_mpc": config.SMOOTHING_SIGMA_CELLS * config.CELL_SIZE,
                "footprint_cells": config.FOOTPRINT_SIZE, "criterion": config.VOID_EIGENVALUE_CRITERION,
                "gradient_threshold": config.GRADIENT_THRESHOLD, "deep_void_threshold": config.DEEP_VOID_THRESHOLD,
                "cut_min_axis_mpc": cuts["min_axis_mpc"], "cut_max_axis_mpc": cuts["max_axis_mpc"],
                "cut_max_axis_ratio": cuts["max_axis_ratio"], "cut_min_axis_cells": config.ELLIPSOID_CUT_RULES["min_axis_cells"],
                "cut_max_axis_sigma": config.ELLIPSOID_CUT_RULES["max_axis_sigma"],
                "cut_max_axis_box_frac": config.ELLIPSOID_CUT_RULES["max_axis_box_frac"],
                "void_gate": "ellipsoid", "r_v_step_sigma": pipeline.R_V_STEP_SIGMA, "r_v_max_sigma": pipeline.R_V_MAX_SIGMA,
                "r_v_dirs": pipeline.R_V_DIRS, "r_v_algorithm": pipeline.R_V_ALGORITHM, "r_v_delta08": pipeline.R_V_DELTA08,
                "param_hash": pipeline._param_hash(), "n_voids": n_voids,
                "created": None}
    assert set(pc) == set(expected), sorted(set(pc) ^ set(expected))
    assert set(ph) == set(expected) | {"columns"}, sorted(set(ph) ^ (set(expected) | {"columns"}))
    for prov in (ph, pc):
        for k, v in expected.items():
            got = prov[k]
            if k == "created":                                      # an aware UTC time stamp of this run
                t = datetime.fromisoformat(str(got))
                assert t.tzinfo is not None and abs((datetime.now(timezone.utc) - t).total_seconds()) < 600, got
            elif k == "catalogue":
                assert "build_void_catalog" in str(got), got
            elif isinstance(v, str):
                assert str(got) == v, (k, got, v)
            else:
                assert abs(float(got) - float(v)) <= 1e-9 * max(1.0, abs(float(v))), (k, got, v)
    # the frame guard: the header's box over the grid must be config's cell. Header rounding (BoxSize in kpc) is no
    # mismatch; a box 1.3 x off refuses naming DTFE_SIM, plus the grid when the grid is not config's; the right box
    # on another grid refuses naming FIELD_RESOLUTION ('DTFE_SIM=' alone cannot be acted on there)
    fr = catalog.frame_of(_Prod(config.BOX_SIZE, config.FIELD_RESOLUTION))
    assert fr[2:] == (config.FIELD_RESOLUTION, "header") and abs(fr[0] - config.CELL_SIZE) <= 1e-12 * config.CELL_SIZE, fr
    assert catalog.frame_of(_Prod(config.CELL_SIZE * n * (1 + 1e-8)))[2:] == (n, "header")
    for box, grid, must, mustnot in (
            (config.CELL_SIZE * n * 1.3, n, ("run with DTFE_SIM=SYN", f"config.FIELD_RESOLUTION = {n}"), ()),
            (config.BOX_SIZE * 1.3, config.FIELD_RESOLUTION, ("run with DTFE_SIM=SYN",), ("FIELD_RESOLUTION",)),
            (config.BOX_SIZE, 2 * config.FIELD_RESOLUTION, ("FIELD_RESOLUTION", f"config.FIELD_RESOLUTION = {2 * config.FIELD_RESOLUTION}"), ())):
        try:
            catalog.frame_of(_Prod(box, grid))
            raise AssertionError(f"box {box:g} / {grid} must refuse")
        except ValueError as e:
            assert all(s in str(e) for s in must) and not any(s in str(e) for s in mustnot), (box, grid, str(e))
    # with DTFE_SIM already the snapshot's simulation, 'run with DTFE_SIM=<that same sim>' is no advice: the hint
    # is picked by cause -- the grid, else the header's box against config's
    for box, grid, must in ((config.BOX_SIZE, 2 * config.FIELD_RESOLUTION, "set config.FIELD_RESOLUTION"),
                            (config.BOX_SIZE * 1.3, config.FIELD_RESOLUTION, "is not config.BOX_SIZE")):
        q = _Prod(box, grid)
        q.sim = config.SIMULATION
        try:
            catalog.frame_of(q)
            raise AssertionError(f"box {box:g} / {grid} under the right sim must refuse")
        except ValueError as e:
            assert must in str(e) and "run with DTFE_SIM=" not in str(e), str(e)
    try:
        catalog.export_voids(_Prod(config.BOX_SIZE, 2 * config.FIELD_RESOLUTION), d / "grid", formats=("csv",))
        raise AssertionError("the export of a grid not config's must refuse")
    except ValueError:
        pass
    assert not (d / "grid").exists()
    # no snapshot header: a REAL SnapshotProducts on a folder with a grid and no combined_*.hdf5 -- FieldSet needs the
    # header for its units, so the frame and the export raise (no fallback to config's box)
    nohdr = _synthetic_snapdir(d / "data", "SYN", 50, 0.5, grids=("den",))
    next(nohdr.glob("combined_*.hdf5")).unlink()
    for what, fn in (("frame_of", catalog.frame_of), ("export_voids", lambda p: catalog.export_voids(p, d / "nohdr"))):
        try:
            fn(pipeline.SnapshotProducts(50, sim="SYN", data_root=d / "data"))
            raise AssertionError(f"{what} without a header must raise")
        except FileNotFoundError as e:
            assert "combined_" in str(e), (what, str(e))
    assert not (d / "nohdr").exists() and not list(Path(config.CACHE_DIR).rglob("*.npz"))
    # with a header the real path has a frame: BoxSize (kpc) / 1000 over the file's grid
    hdr = _synthetic_snapdir(d / "data", "SYN", 51, 0.5, grids=("den",))
    with h5py.File(next(hdr.glob("combined_*.hdf5")), "a") as f:
        f["Header"].attrs["BoxSize"] = config.CELL_SIZE * 8 * 1000.0
    fr = catalog.frame_of(pipeline.SnapshotProducts(51, sim="SYN", data_root=d / "data"))
    assert fr[2:] == (8, "header") and abs(fr[0] - config.CELL_SIZE) <= 1e-9 * config.CELL_SIZE, fr
    # a catalogue built under ANOTHER frame (e.g. TNG100 caches under TNG50's cell): its cell is recovered from
    # positions / indices when it carries no 'cell_mpc', the export refuses it, a fresh build carries the key
    assert abs(pipeline.catalog_cell_mpc(cat) - config.CELL_SIZE) < 1e-12 and pipeline.catalog_frame_ok(cat) and "cell_mpc" in cat
    # the record wins over the recovery (positions still say CELL_SIZE here): the cell the build was told, not a float32 quotient
    assert pipeline.catalog_cell_mpc(dict(cat, cell_mpc=np.float64(0.3))) == 0.3
    stale = {k: (v * 0.47 if k == "positions_mpc" else v) for k, v in cat.items() if k != "cell_mpc"}
    assert abs(pipeline.catalog_cell_mpc(stale) - 0.47 * config.CELL_SIZE) < 1e-9 and not pipeline.catalog_frame_ok(stale)
    assert pipeline.catalog_frame_ok(stale, 0.47 * config.CELL_SIZE) and pipeline.catalog_cell_mpc({"coords": np.zeros((0, 3)), "positions_mpc": np.zeros((0, 3))}) is None

    class _Stale(_Prod):
        def voids(self):
            return stale
    try:
        catalog.export_voids(_Stale(config.CELL_SIZE * n), d, formats=("csv",), stem="stale")
        raise AssertionError("a stale-frame catalogue must refuse")
    except ValueError as e:
        assert "another DTFE_SIM" in str(e), str(e)
    assert not (d / "stale.csv").exists()
    # a second redshift is a second file (the suffix is appended, never with_suffix), unknown formats refuse
    p2 = catalog.export_voids(_Prod(config.CELL_SIZE * n), d, formats=("csv",), stem="voids_z0.00")
    assert p2[0].name == "voids_z0.00.csv" and (d / "voids_SYN_dtfe_050_z0.50.csv").is_file()
    try:
        catalog.export_voids(_Prod(config.CELL_SIZE * n), d, formats=("xlsx",))
        raise AssertionError("an unknown format must refuse")
    except ValueError:
        pass
    # the abundance: n(> R) and dn/dlnR per Mpc^3 with Poisson errors, NaN radii left out
    r = np.array([1.0, 2.0, 2.0, 3.5, np.nan, 8.0])
    nn, err, cnt = catalog.void_abundance(r, 10.0, [0.5, 2.0, 5.0, 10.0])
    assert list(cnt) == [5, 2, 1, 0] and np.allclose(nn, np.array([5, 2, 1, 0]) / 1000.0) and np.allclose(err, np.sqrt([5, 2, 1, 0]) / 1000.0)
    dn, derr, dcnt = catalog.void_dn_dlnr(r, 10.0, [1.0, 2.0, 4.0, 8.0])          # numpy's last bin is closed: 8.0 counts
    assert list(dcnt) == [1, 3, 1] and np.allclose(dn, np.array([1, 3, 1]) / (1000.0 * np.log(2.0))) and np.isclose(derr[1], np.sqrt(3) / (1000.0 * np.log(2.0)))
    nn2, _e, c2 = catalog.void_abundance(th["r_eff_mpc"], 100.0, [0.0])
    assert c2[0] == int(np.isfinite(th["r_eff_mpc"]).sum()) and nn2[0] == c2[0] / 1e6


def t_void_cache_frame():
    """pipeline.SnapshotProducts.voids(): the snapshot header's frame is checked FIRST -- a header that disagrees with
    config.CELL_SIZE is refused with the cache untouched, fresh, stale or absent; a cached catalogue built under
    another frame (its record, or its positions / indices, give another cell), with other ellipsoid cuts or at another
    deep threshold has its fits / flag RE-DERIVED from the cached minima and eigenvalues (no field), the cause named,
    the cache rewritten once; one in this frame with these records is read silently and not rewritten; the export of a
    re-derived 'deep' agrees with its threshold."""
    import io as _io, shutil, tempfile, types
    from contextlib import redirect_stdout
    import config
    from dtfelib import pipeline, catalog
    from dtfelib import fields as dtfe
    n = 16
    rng = np.random.default_rng(2)
    delta = 0.05 * rng.normal(size=(n, n, n)).astype(np.float32)
    idx = np.indices((n, n, n))
    for c, w, dd in (((8, 8, 8), 2.0, 0.8), ((1, 1, 1), 1.5, 0.35)):   # a deep void and a shallow one
        r2 = sum((((idx[k] - c[k] + n // 2) % n) - n // 2) ** 2 for k in range(3))
        delta -= (dd * np.exp(-0.5 * r2 / w ** 2)).astype(np.float32)
    delta_s = dtfe.smooth_field(delta, sigma=1.0)
    hess = pipeline.hessian_from_field(delta_s)

    class _P(pipeline.SnapshotProducts):
        box_factor = 1.0                                             # the header's box relative to config's frame
        delta_smoothed = property(lambda self: delta_s)
        hessian = property(lambda self: hess)
        @property
        def fs(self):
            return types.SimpleNamespace(method="dtfe", grid_n=n,
                                         meta=types.SimpleNamespace(box_mpc=config.CELL_SIZE * n * self.box_factor, redshift=1.0))
    root = Path(tempfile.mkdtemp(prefix="dtfe_vcache_"))
    old, old_save = config.CACHE_DIR, pipeline._save_npz
    saves = []
    p = _P("050", 1.0, sim="SYN")

    def load(writes):
        buf, mark = _io.StringIO(), len(saves)
        with redirect_stdout(buf):
            got = p.voids()
        assert saves[mark:] == [vname] * writes, (f"{writes} cache write(s) expected", saves[mark:], buf.getvalue())
        return got, buf.getvalue()

    def refused(why):
        mark = len(saves)
        try:
            p.voids()
            raise AssertionError(why)
        except ValueError as e:
            assert saves[mark:] == [], ("a refused load wrote the cache", saves[mark:])
            return str(e)
    try:
        config.CACHE_DIR = root / "cache"
        # every cache write, by name: bytes cannot tell a re-save from none (np.savez stamps whole seconds), a spy can.
        # voids() looks _save_npz up in the module at call time
        pipeline._save_npz = lambda path, **a: (saves.append(Path(path).name), old_save(path, **a))[1]
        vname = p._cpath("voids").name
        cat, said = load(1)                                          # built and cached in this frame: one write
        path = p._cpath("voids")
        assert len(cat["coords"]) == 2 and cat["deep"].all(), (cat["coords"], cat["delta_values"])   # the fixture
        assert path.is_file() and "cell_mpc" in cat and abs(float(cat["cell_mpc"]) - config.CELL_SIZE) < 1e-12
        assert np.allclose(cat["cuts_mpc"], [pipeline.ellipsoid_cuts_mpc()[k] for k in ("min_axis_mpc", "max_axis_mpc", "max_axis_ratio")])
        assert float(cat["deep_threshold"]) == config.DEEP_VOID_THRESHOLD
        same, said = load(0)
        assert said == "", said                                      # the build's records: read as it is, no refit or re-save every load
        assert np.array_equal(same["positions_mpc"], cat["positions_mpc"]) and "cell_mpc" in same
        # a legacy cache: another frame, no record of the cell, the cuts or the threshold -> the fits re-derived, not
        # the field
        stale = {k: (v * 0.47 if k in ("positions_mpc", "semi_axes") else v) for k, v in cat.items()
                 if k not in ("cell_mpc", "cuts_mpc", "deep_threshold")}
        stale["well_resolved"] = ~stale["well_resolved"]
        np.savez_compressed(path, **stale)
        calls = []
        p.__class__.delta_smoothed = property(lambda self: calls.append("field") or delta_s)   # must NOT be read
        fresh, said = load(1)
        assert "re-deriving" in said and not calls and abs(float(fresh["cell_mpc"]) - config.CELL_SIZE) < 1e-12, (said, calls)
        assert "no record of its ellipsoid cuts" in said, said
        for k in ("positions_mpc", "semi_axes", "well_resolved", "fit_ellipticity", "orientations", "deep"):
            assert np.array_equal(fresh[k], cat[k]), k
        with np.load(path) as z:
            assert "cell_mpc" in z.files and "cuts_mpc" in z.files and np.array_equal(z["well_resolved"], cat["well_resolved"])   # overwritten
        _, said = load(0)
        assert said == "", said                                      # the refit wrote its records: re-derived ONCE
        # the FRAME alone (the record says another cell, the cuts and threshold are current): re-derived, the cuts not
        # mentioned, the cell named the cache's (from its record) against this frame's
        stale2 = {k: (v * 0.47 if k in ("positions_mpc", "semi_axes") else v) for k, v in cat.items()}
        stale2["cell_mpc"] = np.float64(0.47 * config.CELL_SIZE)
        stale2["well_resolved"] = ~cat["well_resolved"]
        np.savez_compressed(path, **stale2)
        fresh2, said = load(1)
        assert "built with cell" in said and "cuts" not in said and not calls, said
        assert f"built with cell {0.47 * config.CELL_SIZE:.5g} Mpc, not this frame's {config.CELL_SIZE:.5g}" in said, said
        for k in ("positions_mpc", "semi_axes", "well_resolved", "fit_ellipticity", "orientations"):
            assert np.array_equal(fresh2[k], cat[k]), k
        # THE HEADER DECIDES, before the cache is touched: a header 1.3 x config's frame over a STALE cache is refused
        # with the file's bytes unchanged (refused, not re-derived-and-rewritten-then-complained) ...
        np.savez_compressed(path, **stale2)
        before = path.read_bytes()
        p.box_factor = 1.3
        e = refused("a header that disagrees with config.CELL_SIZE must refuse")
        assert "run with DTFE_SIM=SYN" in e and f"config.FIELD_RESOLUTION = {n}" in e, e
        assert path.read_bytes() == before, "the refused load rewrote the cache"
        # ... over a FRESH one (this frame, current records: a cache hit) too -- once rewritten into config's frame a
        # cache is fresh on every later load, and the voids() consumers without a frame check of their own rely on this
        # one ...
        np.savez_compressed(path, **cat)
        before = path.read_bytes()
        refused("a fresh cache must not be served under a header that disagrees with config.CELL_SIZE")
        assert path.read_bytes() == before and not calls, "the refused load touched the cache or read the field"
        # ... and with no cache at all nothing is built: the field is not read, no file appears
        path.unlink()
        refused("a header that disagrees with config.CELL_SIZE must refuse")
        assert not path.exists() and not calls, calls
        p.box_factor = 1.0
        np.savez_compressed(path, **cat)
        # the deep threshold retuned between the two voids: 'deep' follows (delta_values are cached), the record says so,
        # once; the export's column is that flag and its deep_void_threshold the threshold it was taken at (the column
        # and the attribute agree)
        old_deep = config.DEEP_VOID_THRESHOLD
        thr = float(np.mean(cat["delta_values"]))
        config.DEEP_VOID_THRESHOLD = thr
        try:
            redeep, said = load(1)
            assert "deep threshold" in said and not calls, said
            assert np.array_equal(redeep["deep"], np.asarray(cat["delta_values"]) < thr) and redeep["deep"].any() and not redeep["deep"].all(), redeep["deep"]
            assert float(redeep["deep_threshold"]) == thr and np.array_equal(redeep["well_resolved"], cat["well_resolved"])
            _, said = load(0)
            assert said == "", said
            mark = len(saves)
            out = catalog.export_voids(p, root / "out")
            assert saves[mark:] == [], ("the export of a fresh catalogue rewrote the cache", saves[mark:])
            for t, prov in (catalog.read_hdf5(out[1]), catalog.read_csv(out[0])):
                assert float(prov["deep_void_threshold"]) == thr, prov["deep_void_threshold"]
                assert np.array_equal(t["deep"], t["delta_c"] < float(prov["deep_void_threshold"])) and t["deep"].any() and not t["deep"].all(), t["deep"]
        finally:
            config.DEEP_VOID_THRESHOLD = old_deep
        back, said = load(1)                                         # back to the default threshold: the flag returns
        assert "deep threshold" in said and np.array_equal(back["deep"], cat["deep"]) and float(back["deep_threshold"]) == old_deep, said
        # the cuts retuned: the flag follows without a rebuild
        old_cuts = config.ELLIPSOID_CUT_RULES
        config.ELLIPSOID_CUT_RULES = dict(old_cuts, max_axis_sigma=1e-6)
        try:
            tight, said = load(1)
            assert "ellipsoid cuts" in said and not tight["well_resolved"].any() and not calls, said
        finally:
            config.ELLIPSOID_CUT_RULES = old_cuts
        back, said = load(1)
        assert "ellipsoid cuts" in said and np.array_equal(back["well_resolved"], cat["well_resolved"]), said
    finally:
        config.CACHE_DIR, pipeline._save_npz = old, old_save
        shutil.rmtree(root, ignore_errors=True)


def t_export_driver():
    """analyze.py export with pipeline.products stubbed: an id given as '0' is config's
    '000'; an unknown id, a frame refusal, missing data and nothing written each exit 1 on their own while the good
    snapshot is still written; the summary names them; --format maps to the files; a catalogue with no void still
    exports; the default folder is figures/void_catalog/<sim>."""
    import io as _io, shutil, sys as _sys, tempfile, types
    from contextlib import redirect_stdout
    import config
    from dtfelib import pipeline, catalog
    from dtfelib import fields as dtfe
    import analyze
    n = 32
    rng = np.random.default_rng(7)
    delta = 0.02 * rng.normal(size=(n, n, n)).astype(np.float32)
    idx = np.indices((n, n, n))
    for c, w, dd in (((8, 8, 8), 2.0, 0.9), ((20, 10, 24), 3.5, 0.8)):
        r2 = sum((((idx[k] - c[k] + n // 2) % n) - n // 2) ** 2 for k in range(3))
        delta -= (dd * np.exp(-0.5 * r2 / w ** 2)).astype(np.float32)
    delta_s = dtfe.smooth_field(delta, sigma=1.0)
    cat = pipeline.build_void_catalog(delta_s, pipeline.hessian_from_field(delta_s))
    assert len(cat["coords"]) >= 2, len(cat["coords"])
    per_void = ("coords", "eigenvalues", "eigenvectors", "axis_ratios", "bbks_params", "delta_values", "deep", "well_resolved",
                "semi_axes", "orientations", "fit_ellipticity", "fit_prolateness", "positions_mpc", "r_v_cells",
                "r_v_delta08_cells", "r_v_ridge_cells")
    none = {k: (np.asarray(v)[:0] if k in per_void else v) for k, v in cat.items()}     # a snapshot without a void
    k0, k1, k2, k3 = list(config.SNAPSHOT_TO_REDSHIFT)[:4]
    good = config.CELL_SIZE * n
    plan = {k0: dict(box=good, voids=cat), k1: dict(box=good * 1.3, voids=cat),          # k1: the frame guard refuses
            k2: dict(box=good, voids=cat, missing=True), k3: dict(box=good, voids=none)}
    released = []

    class _Stub:
        sim, prefix = "SYN", None
        def __init__(self, snap, z, box, voids, missing=False):
            self.snapshot, self.redshift, self._cat, self._missing = snap, z, voids, missing
            self.fs = types.SimpleNamespace(grid_n=n, method="dtfe", prefix="output.",
                                            meta=types.SimpleNamespace(box_mpc=box, redshift=z))
        @property
        def data_dir(self):
            if self._missing:
                raise FileNotFoundError(f"snapshot directory not found: snapdir_{self.snapshot}")
            return Path("snapdir_" + self.snapshot)
        def voids(self):
            return self._cat
        def release(self):
            released.append(self.snapshot)

    def stem(k):
        return f"voids_SYN_dtfe_{k}_z{config.get_redshift(k):.2f}"

    root = Path(tempfile.mkdtemp(prefix="dtfe_export_"))
    old_products, old_cache, old_figs, old_argv = pipeline.products, config.CACHE_DIR, config.LOCAL_FIGURES_ROOT, list(_sys.argv)
    pipeline.products = lambda snapshot, redshift=None, **kw: _Stub(snapshot, redshift, **plan[snapshot])
    config.CACHE_DIR = root / "cache"                                # export prints cache_dir(), which creates it
    config.LOCAL_FIGURES_ROOT = str(root / "figures")

    def run(snaps, out, formats=("csv", "hdf5"), argv=None, look=None):
        buf = _io.StringIO()
        with redirect_stdout(buf):
            try:
                if argv is None:
                    analyze.export(snaps, out, formats)
                else:
                    _sys.argv = ["analyze.py", "export"] + argv
                    analyze.main()
                code = 0
            except SystemExit as e:
                code = int(e.code or 0)
        look = Path(look or out)                                     # the folder the files should land in
        files = sorted(q.name for q in look.iterdir()) if look.exists() else []
        return code, buf.getvalue(), files
    try:
        both = [stem(k0) + ".csv", stem(k0) + ".h5"]
        said_k0 = f"snapshot {k0} (z={config.get_redshift(k0):.2f}): {len(cat['coords'])} voids -> {both[0]}, {both[1]}"   # the summary line
        code, text, files = run([str(int(k0)), "999", k1, k2], root / "a")
        assert code == 1 and files == both and said_k0 in text, (code, files, text[-600:])
        assert "Unknown snapshot ids: 999" in text and f"Failed snapshots: {k1}, {k2}" in text, text[-600:]
        assert "run with DTFE_SIM=SYN" in text and "data missing or unreadable" in text, text[-800:]
        assert {k0, k1} <= set(released), released                   # every product that loaded data is released
        th, ph = catalog.read_hdf5(root / "a" / both[1])
        assert int(ph["n_voids"]) == len(cat["coords"]) and str(ph["snapshot"]) == k0 and len(th["id"]) == len(cat["coords"])
        # each exit clause on its own (the good snapshot written every time): unknown id, frame refusal, missing data,
        # nothing written (formats=() is the only way to write nothing without another clause firing)
        for i, (snaps, formats, say) in enumerate((([k0, "999"], ("csv", "hdf5"), "Unknown snapshot ids: 999"),
                                                   ([k0, k1], ("csv", "hdf5"), f"Failed snapshots: {k1}"),
                                                   ([k0, k2], ("csv", "hdf5"), f"Failed snapshots: {k2}"),
                                                   ([k0], (), "0 file(s)"))):
            code, text, files = run(snaps, root / f"iso{i}", formats)
            assert code == 1 and files == (both if formats else []) and say in text, (snaps, formats, code, files, text[-400:])
        code, text, files = run([k0], root / "ok")
        assert code == 0 and files == both and said_k0 in text and "Failed" not in text and "Unknown" not in text, (code, files, text[-400:])
        code, text, files = run([k1], root / "x")                    # refused and nothing else: exit 1, no file
        assert code == 1 and files == [], (code, files)
        # a snapshot whose catalogue holds no void is exported (0 rows), not counted failed
        code, text, files = run([k3], root / "none")
        assert code == 0 and files == [stem(k3) + ".csv", stem(k3) + ".h5"] and f"snapshot {k3}" in text and ": 0 voids" in text, (code, files, text[-400:])
        tc, pc = catalog.read_csv(root / "none" / files[0])
        assert len(tc["id"]) == 0 and int(pc["n_voids"]) == 0 and len(catalog.read_hdf5(root / "none" / files[1])[0]["a_mpc"]) == 0
        # --format through the CLI: csv / hdf5 / both
        for fmt, want in (("csv", [both[0]]), ("hdf5", [both[1]]), ("both", both)):
            code, text, files = run(None, root / f"fmt_{fmt}", argv=[k0, "--out", str(root / f"fmt_{fmt}"), "--format", fmt])
            assert code == 0 and files == want, (fmt, code, files, text[-400:])
        # no --out: figures/void_catalog/<the DTFE_SIM simulation>/
        code, text, files = run([k0], None, look=Path(config.LOCAL_FIGURES_ROOT) / "void_catalog" / config.SIMULATION)
        assert code == 0 and files == both, (code, files, text[-400:])
    finally:
        pipeline.products, config.CACHE_DIR, config.LOCAL_FIGURES_ROOT = old_products, old_cache, old_figs
        _sys.argv = old_argv
        shutil.rmtree(root, ignore_errors=True)


def t_void_profiles():
    """Shell profiles of a synthetic void with a known step profile and a linear
    outflow -- exact where every trilinear neighbour sits on one plateau -- the same void across the periodic
    boundary, a bulk velocity cancelling, the single-stream fraction by NEAREST cell, the stack's 16-84% bootstrap
    band on distinct voids and its min_count rule, the wrap rule, the linear-theory overlay at z=0 and z=1, and the
    FieldSet wrapper on a fake snapshot: the catalogue's rows (not the sample's), the cell-centre convention on each
    axis, the hidden-streams bit, the method gate, and a 'deep' sample that follows a retuned DEEP_VOID_THRESHOLD
    (the cuts pinned in cells for that fixture)."""
    import io as _io, shutil, tempfile, types
    from contextlib import redirect_stdout
    import config
    from dtfelib import profiles, catalog, pipeline
    from dtfelib.io import single_stream_mask
    n, R = 64, 8.0
    idx = np.indices((n, n, n)).astype(np.float64)

    def make(c):                                            # a void at the cell centre c (cells)
        d = [((idx[k] + 0.5 - c[k] + n / 2) % n) - n / 2 for k in range(3)]
        r = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        den = np.ones((n, n, n), np.float32)
        den[r < R] = 0.2
        den[(r >= R) & (r < 1.5 * R)] = 1.6                 # a 4-cell ridge: wider than 2 sqrt(3), so a shell in it is exact
        vel = np.zeros((n, n, n, 3), np.float32)
        for k in range(3):
            vel[..., k] = np.where(r < R, 50.0 * d[k] / R, 0.0)   # v = 50 (r / R) n -> v_r = 50 r / R, linear: exact
        streams = np.where(r < 1.2 * R, 1.0, 3.0).astype(np.float32)
        return {"density": den, "velocity": vel, "single": single_stream_mask(streams, None), "r": r}

    g0, g1 = make((32.5, 32.5, 32.5)), make((2.5, 61.5, 5.5))        # the second void straddles three box faces
    bins = (0.5, 1.25, 2.0, 3.0)
    p0 = profiles.profile_grids(g0, [(32.5 / n,) * 3], [R], bins=bins, m=400)
    p1 = profiles.profile_grids(g1, [(2.5 / n, 61.5 / n, 5.5 / n)], [R], bins=bins, m=400)
    for k in ("density", "v_r", "single"):
        assert np.allclose(p0[k], p1[k], rtol=1e-5, atol=1e-5), (k, p0[k], p1[k])      # the wrap changes nothing
    assert p0["dropped"] == 0 and list(p0["index"]) == [0]
    assert abs(p0["density"][0, 0] - 0.2) < 1e-6 and abs(p0["density"][0, 1] - 1.6) < 1e-6 and abs(p0["density"][0, 2] - 1.0) < 1e-6, p0["density"]
    assert abs(p0["v_r"][0, 0] - 25.0) < 1e-4 and abs(p0["v_r"][0, 2]) < 1e-6, p0["v_r"]
    assert p0["single"][0, 0] == 1.0 and p0["single"][0, 2] == 0.0, p0["single"]
    bulk = dict(g0, velocity=g0["velocity"] + np.array([30.0, -20.0, 10.0], np.float32))
    pb = profiles.profile_grids(bulk, [(32.5 / n,) * 3], [R], bins=bins, m=400)
    assert np.abs(pb["v_r"] - p0["v_r"]).max() < 0.2, (pb["v_r"], p0["v_r"])             # a bulk motion is a dipole: cancels
    assert abs(profiles.fibonacci_sphere(400).mean(axis=0)).max() < 5e-3
    # the mask by NEAREST cell: a shell (r = 1.2 R = 9.6 cells) that lands mid-cell on the streams step gives the
    # fraction of its points whose cell (floor) is single-stream -- 0.545 here; trilinear would blend it to 0.516
    pm = profiles.profile_grids(g0, [(32.5 / n,) * 3], [R], bins=(1.2,), m=400)
    pts = np.full(3, 32.5 / n) + 1.2 * R * profiles.fibonacci_sphere(400) / n
    cell = np.floor(np.mod(pts, 1.0) * n).astype(np.int64) % n
    ngp = g0["single"][cell[:, 0], cell[:, 1], cell[:, 2]].mean()
    assert 0.0 < ngp < 1.0 and abs(pm["single"][0, 0] - ngp) < 1e-6, (pm["single"], ngp)
    # the wrap rule: a void whose 3 R_eff shell reaches past half the box is dropped and counted
    pw = profiles.profile_grids(g0, [(32.5 / n,) * 3, (0.5, 0.5, 0.5)], [R, 11.0], bins=bins, m=50)
    assert pw["dropped"] == 1 and list(pw["index"]) == [0] and pw["density"].shape == (1, 4)
    # the stack: two identical voids -> the mean is the profile, a zero-width band; an empty bin stays NaN
    prof = {"bins": np.asarray(bins), "r_eff_mpc": np.array([R * 0.1, R * 0.1]), "density": np.vstack([p0["density"], p1["density"]]),
            "v_r": np.vstack([p0["v_r"], p1["v_r"]]), "single": np.vstack([p0["single"], p1["single"]])}
    st = profiles.stack(prof, edges_mpc=(0.0, 0.5, 1.0, 2.0), n_boot=50, min_count=2)
    assert list(st["count"]) == [0, 2, 0] and np.isnan(st["density"]["mean"][0]).all() and np.isnan(st["density"]["mean"][2]).all()
    assert np.allclose(st["density"]["mean"][1], p0["density"][0], atol=1e-5) and np.allclose(st["density"]["lo"][1], st["density"]["hi"][1], atol=1e-5)
    st1 = profiles.stack(dict(prof, r_eff_mpc=np.array([0.8, 1.5])), edges_mpc=(0.0, 1.0, 2.0), min_count=1)
    assert list(st1["count"]) == [1, 1] and np.isnan(st1["v_r"]["lo"]).all() and np.isfinite(st1["v_r"]["mean"]).all()   # one void: no band
    # the default min_count 5: a bin of 4 voids is left NaN with its count kept, a bin of exactly 5 is stacked
    # (identical rows: without the rule the 4-void bin would get a finite mean AND a finite band)
    pr5 = {"bins": np.asarray(bins), "r_eff_mpc": np.array([0.5] * 4 + [1.5] * 5), "density": np.vstack([p0["density"]] * 9),
           "v_r": np.vstack([p0["v_r"]] * 9), "single": None}
    st5 = profiles.stack(pr5, edges_mpc=(0.0, 1.0, 2.0), n_boot=20)
    assert st5["min_count"] == 5 and list(st5["count"]) == [4, 5], (st5["min_count"], st5["count"])
    for k in ("density", "v_r"):
        assert all(np.isnan(st5[k][q][0]).all() for q in ("mean", "lo", "hi")), k
        assert np.isfinite(st5[k]["mean"][1]).all() and np.isfinite(st5[k]["lo"][1]).all() and np.isfinite(st5[k]["hi"][1]).all(), k
    # the band on DISTINCT voids: the 16th and 84th percentiles of the bootstrap means, redrawn by hand with the same
    # seed (one populated bin, so stack's one rng serves density first, then v_r); strictly around the mean
    sc = np.linspace(0.8, 1.5, 8)[:, None]
    pr8 = {"bins": np.asarray(bins), "r_eff_mpc": np.ones(8), "density": p0["density"] * sc, "v_r": p0["v_r"] * sc, "single": None}
    st8 = profiles.stack(pr8, edges_mpc=(0.0, 2.0), n_boot=400, min_count=2, seed=0)
    rng = np.random.default_rng(0)
    for k in ("density", "v_r"):
        means = np.nanmean(pr8[k][rng.integers(0, 8, size=(400, 8))], axis=1)
        lo, hi = np.nanpercentile(means, 16, axis=0), np.nanpercentile(means, 84, axis=0)
        assert np.allclose(st8[k]["lo"][0], lo, rtol=1e-12, atol=1e-12) and np.allclose(st8[k]["hi"][0], hi, rtol=1e-12, atol=1e-12), (k, st8[k]["lo"][0], lo)
        assert st8[k]["lo"][0, 0] < st8[k]["mean"][0, 0] < st8[k]["hi"][0, 0] and (st8[k]["lo"] <= st8[k]["hi"]).all(), (k, st8[k])
    assert "single" not in st8
    pr4 = {"bins": np.array([0.5]), "r_eff_mpc": np.ones(4), "density": np.array([[0.1], [0.2], [0.3], [0.6]]), "v_r": None, "single": None}
    st4 = profiles.stack(pr4, edges_mpc=(0.0, 2.0), n_boot=200, min_count=2)
    d4 = st4["density"]
    assert np.isclose(d4["mean"][0, 0], 0.3) and np.isclose(d4["lo"][0, 0], 0.2) and np.isclose(d4["hi"][0, 0], 0.4), d4   # 5/95 gives 0.174/0.45
    # linear theory: a flat rho/rho_bar = 0.2 (Delta = -0.8) gives v_r = (0.8 / 3) f H a r > 0 (outflow); at z=1 against
    # numbers taken from config.COSMOLOGY, not from get_cosmology_params: f = Omega_m(1)^0.55 = 0.8732, H(1) = 120.46
    # km/s/Mpc, a = 0.5 -- a dropped a is 2x off, 1/a 4x
    from dtfelib import fields as dtfe
    x = np.linspace(0.1, 3.0, 30)
    v = profiles.linear_v_r(x, np.full(30, 0.2), 5.0, 0.0)
    c = dtfe.get_cosmology_params(0.0)
    assert np.allclose(v, (0.8 / 3.0) * c["f_growth"] * c["H_z"] * c["a"] * x * 5.0, rtol=0.02) and (v > 0).all(), v[:3]
    om, ol, h0 = config.COSMOLOGY["Omega_m"], config.COSMOLOGY["Omega_Lambda"], config.COSMOLOGY["H0"]
    e2 = om * 8.0 + ol
    f1, h1 = (om * 8.0 / e2) ** 0.55, h0 * np.sqrt(e2)
    assert abs(f1 - 0.8732) < 1e-3 and abs(h1 - 120.46) < 1e-2, (f1, h1)
    v1 = profiles.linear_v_r(x, np.full(30, 0.2), 5.0, 1.0)
    assert np.allclose(v1, (0.8 / 3.0) * f1 * h1 * 0.5 * x * 5.0, rtol=1e-4), (v1[:3], ((0.8 / 3.0) * f1 * h1 * 0.5 * x * 5.0)[:3])
    # the FieldSet wrapper. The catalogue's rows are NOT the sample's: row 0 is unresolved, row 2 resolved but dropped
    # by the wrap rule (R_eff 11 cells), so the kept voids are rows 1 and 3, with distinct delta and R_eff
    cs = config.CELL_SIZE
    cat = {"coords": np.array([[10, 10, 10], [32, 32, 32], [40, 20, 50], [2, 61, 5]]),
           "well_resolved": np.array([False, True, True, True]), "deep": np.array([True, True, True, False]),
           "delta_values": np.array([-0.1, -0.8, -0.7, -0.6], np.float32),
           "semi_axes": np.array([[0.0] * 3, [R * cs] * 3, [11.0 * cs] * 3, [6.0 * cs] * 3], np.float32),
           "cell_mpc": cs}                                  # the frame a built catalogue carries
    streams0 = np.where(g0["single"], 1.0, 3.0).astype(np.float32)
    hid = np.zeros((n, n, n), np.float32)
    hid[g0["r"] < 0.75 * R] = 1.0                           # bit 1 (multi-stream the count misses) inside 6 cells of void 1
    hid[(g0["r"] >= 6.5) & (g0["r"] < 9.5)] = 2.0           # bit 2 alone (a quadrature note) must NOT mask

    class _FS:
        grid_n, method, den, hidden = n, "ps", g0["density"], hid
        meta = types.SimpleNamespace(box_mpc=cs * n, redshift=0.0)
        def has(self, name):
            return name in ("density", "velocity", "streams") or (name == "hidden_streams" and self.hidden is not None)
        def load(self, name, mode="auto"):
            return {"velocity": g0["velocity"], "streams": streams0, "hidden_streams": self.hidden}[name]
        def density(self, units="mean"):
            return self.den

    class _Prod:
        sim, snapshot, prefix, redshift = "SYN", "050", None, 0.0
        def __init__(self, fs=None):
            self.fs = fs or _FS()
        def voids(self):
            return cat

    wb = (0.1, 0.5, 1.0, 2.0, 3.0)
    pf = profiles.void_profiles(_Prod(), bins=wb, m=200, radius="r_eff", distinct=False)
    assert list(pf["index"]) == [1, 3] and pf["dropped"] == 1 and pf["method"] == "ps", (pf["index"], pf["dropped"])
    assert np.array_equal(pf["delta_c"], np.array([-0.8, -0.6], np.float32).astype(np.float64)), pf["delta_c"]
    assert np.allclose(pf["r_eff_mpc"], np.array([8.0, 6.0]) * cs, rtol=1e-6, atol=0.0), pf["r_eff_mpc"]
    # row 1 sits on the step void (0.2 inside, 1 beyond the ridge), row 3 far from it (1 everywhere)
    assert abs(pf["density"][0, 0] - 0.2) < 1e-6 and abs(pf["density"][0, 1] - 0.2) < 1e-6 and np.abs(pf["density"][0, 3:] - 1.0).max() < 1e-6, pf["density"]
    assert np.abs(pf["density"][1] - 1.0).max() < 1e-6 and abs(pf["v_r"][0, 0] - 5.0) < 1e-4 and abs(pf["v_r"][0, 1] - 25.0) < 1e-4, (pf["density"], pf["v_r"])
    # the hidden bit: streams read 1 inside 1.2 R, but bit 1 marks r < 6 cells multi-stream -> 0 on the shells at 0.8
    # and 4 cells, 1 on the shell at 8 cells (streams 1, only bit 2 there), 0 beyond 1.2 R (streams 3)
    assert pf["single"] is not None and list(pf["single"][0]) == [0.0, 0.0, 1.0, 0.0, 0.0], pf["single"]
    # the deep sample: rows 1 and 2 are deep, 2 is wrap-dropped
    pd_ = profiles.void_profiles(_Prod(), bins=wb, m=50, sample="deep", radius="r_eff", distinct=False)
    assert list(pd_["index"]) == [1] and pd_["dropped"] == 1, (list(pd_["index"]), pd_["dropped"])
    # the centre convention, axis by axis: a density linear along axis k has the cell CENTRE value on every shell that
    # does not wrap, 1 + 0.01 (c_k + 0.5): 1.325 for row 1 (cell 32), 1.025 / 1.615 / 1.055 on row 3's first shell
    # (cell 2, 61, 5); a corner on any one axis is 0.005 off. Axis 1's Fibonacci column is exactly antisymmetric
    # (1e-6); axes 0 and 2 carry the lattice's small mean (< 6e-5 at 24 cells), hence 2e-4 there
    for k, c3, tol in ((0, 1.025, 2e-4), (1, 1.615, 1e-6), (2, 1.055, 2e-4)):
        fr = _FS()
        fr.den = 1.0 + 0.01 * (idx[k] + 0.5)
        pr = profiles.void_profiles(_Prod(fr), bins=wb, m=200, radius="r_eff", distinct=False)
        assert np.abs(pr["density"][0] - 1.325).max() < tol and abs(pr["density"][1, 0] - c3) < tol, (k, pr["density"])
    # the method gate: a standard-DTFE set has no single-stream fraction, even from a fake that claims a streams grid
    fd = _FS()
    fd.method = "dtfe"
    assert profiles.void_profiles(_Prod(fd), bins=wb, m=50, radius="r_eff", distinct=False)["single"] is None
    try:
        profiles.void_profiles(_Prod(), sample="all")
        raise AssertionError("an unknown sample must refuse")
    except ValueError:
        pass
    # 'deep' follows a RETUNED DEEP_VOID_THRESHOLD through the real SnapshotProducts.voids(): a cache whose deep flag
    # was taken at -0.5 (rows 1, 2, 3 deep) is re-derived at -0.75 (row 1 only), and back
    ev = np.array([[1e-4] * 3, [4.0 / R ** 2] * 3, [4.0 / 11.0 ** 2] * 3, [4.0 / 6.0 ** 2] * 3])   # a = 2 cell / sqrt(lambda)

    class _P(pipeline.SnapshotProducts):
        @property
        def fs(self):
            return _FS()
    old_cache, old_deep, old_cuts = config.CACHE_DIR, config.DEEP_VOID_THRESHOLD, config.ELLIPSOID_CUT_RULES
    config.CACHE_DIR = Path(tempfile.mkdtemp(prefix="dtfe_vprof_"))
    try:
        # the cuts in CELLS (a fresh dict, never the live one mutated): rows 1-3 (6-11 cells) resolved, row 0 (200
        # cells) not, whatever DTFE_SIM or a retune of the Mpc cuts says
        config.ELLIPSOID_CUT_RULES = {"min_axis_cells": 0.5, "max_axis_sigma": 20.0 / config.SMOOTHING_SIGMA_CELLS,
                                      "max_axis_box_frac": 0.5, "max_axis_ratio": 10.0}
        config.DEEP_VOID_THRESHOLD = -0.5
        built = pipeline.refit_catalog({"coords": cat["coords"], "eigenvalues": ev, "eigenvectors": np.tile(np.eye(3), (4, 1, 1)),
                                        "delta_values": cat["delta_values"], "deep": cat["delta_values"] < -0.5})
        assert list(built["well_resolved"]) == [False, True, True, True] and list(built["deep"]) == [False, True, True, True], (built["well_resolved"], built["deep"])
        assert np.allclose(catalog.r_eff_mpc(built)[1:], np.array([8.0, 11.0, 6.0]) * cs, rtol=1e-6)
        p = _P("050", 0.0, sim="SYN")
        np.savez_compressed(p._cpath("voids"), **built)
        buf = _io.StringIO()
        with redirect_stdout(buf):
            assert list(profiles.void_profiles(p, bins=wb, m=50, sample="deep", radius="r_eff", distinct=False)["index"]) == [1, 3]      # the cache as it is
            config.DEEP_VOID_THRESHOLD = -0.75
            got = profiles.void_profiles(p, bins=wb, m=50, sample="deep", radius="r_eff", distinct=False)
            config.DEEP_VOID_THRESHOLD = -0.5
            back = profiles.void_profiles(p, bins=wb, m=50, sample="deep", radius="r_eff", distinct=False)
        assert list(got["index"]) == [1] and list(back["index"]) == [1, 3], (list(got["index"]), list(back["index"]), buf.getvalue())
    finally:
        shutil.rmtree(config.CACHE_DIR, ignore_errors=True)
        config.CACHE_DIR, config.DEEP_VOID_THRESHOLD, config.ELLIPSOID_CUT_RULES = old_cache, old_deep, old_cuts


def _void_pin_config(root, saved, n=64, box=64.0):
    """The void-script tests' config: the frame of the synthetic ladder (config's cell = the headers' box / n, else
    frame_of refuses everything), caches + figures under 'root', and the CURRENT catalogue knobs as
    FRESH objects (never the live dict mutated) -- the fixture's largest semi-axis is 9.3 Mpc against max_axis 10, so
    a retune of the cuts must not move it. The old values go into 'saved' first, for the caller's finally."""
    import config
    pins = {"CACHE_DIR": Path(root) / "cache", "LOCAL_FIGURES_ROOT": str(Path(root) / "figures"),
            "CELL_SIZE": box / n, "BOX_SIZE": box, "FIELD_RESOLUTION": n, "SMOOTHING_SIGMA_CELLS": 1.5,
            "PRODUCTION_SMOOTHING": (1.5, 11),                    # the fixture's sigma IS its production: untagged names
            # 0.1 / 10 Mpc in this 1 Mpc cell with sigma = 1.5 cells: 0.1 cells, 10/1.5 sigma (L/2 = 32 Mpc does not bind)
            "ELLIPSOID_CUT_RULES": {"min_axis_cells": 0.1, "max_axis_sigma": 10.0 / 1.5, "max_axis_box_frac": 0.5,
                                    "max_axis_ratio": 10.0},
            "FOOTPRINT_SIZE": 11, "GRADIENT_THRESHOLD": 0.1, "VOID_EIGENVALUE_CRITERION": "trace",
            "DEEP_VOID_THRESHOLD": -0.1, "PANEL_SNAPSHOTS": ["000", "017", "050", "099"]}
    saved.update({k: getattr(config, k) for k in pins})
    for k, v in pins.items():
        setattr(config, k, v)


def _void_ladder_snapdir(data, snap, z, seed, prefixes, n=64, box=64.0):
    """One snapshot folder of the void-script tests under data/SYN: a combined header (box, redshift z) and an n^3
    density of six Gaussian dips on 2% noise (six well-resolved voids at --smooth 1.5 under the pinned cuts), zero
    velocity, single-stream; the standard-DTFE set ('output') lacks the last dip, so the other estimator's
    catalogue is NOT the PS one. Returns the folder."""
    import h5py
    d = Path(data) / "SYN" / f"snapdir_{snap:03d}"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / f"combined_{snap:03d}.hdf5", "w") as f:
        h = f.create_group("Header").attrs
        h["BoxSize"], h["Redshift"], h["HubbleParam"], h["HFreeUnits"] = box * 1000.0, float(z), 0.6774, 1
        h["NumPart_ThisFile"], h["MassTable"] = np.array([0, n ** 3, 0, 0, 0, 0]), np.array([0, 1.0, 0, 0, 0, 0])
    noise = 1.0 + 0.02 * np.random.default_rng(seed).normal(size=(n, n, n))
    ii = np.indices((n, n, n))
    dips = [np.zeros((n, n, n))]
    for c, w, depth in (((16, 16, 16), 2.0, 0.85), ((44, 20, 40), 2.5, 0.8), ((20, 48, 30), 1.6, 0.9),
                        ((50, 50, 10), 3.0, 0.75), ((8, 36, 54), 2.2, 0.85), ((36, 6, 26), 1.8, 0.8)):
        r2 = sum((((ii[k] - c[k] + n // 2) % n) - n // 2) ** 2 for k in range(3))
        dips.append(dips[-1] + depth * np.exp(-0.5 * r2 / w ** 2))
    for p in prefixes:
        (noise - dips[5 if p == "output" else 6]).astype(np.float32).tofile(str(d / f"{p}.a_den"))
        np.zeros((n, n, n, 3), np.float32).tofile(str(d / f"{p}.a_vel"))
        if p != "output":                                    # streams: PS-DTFE only
            np.ones((n, n, n), np.float32).tofile(str(d / f"{p}.a_streams"))
    return d


def t_void_scripts():
    """plot_void_profiles.py and plot_void_population.py end to end, in process, on two synthetic 64^3 snapshots
    (box 64 Mpc, six Gaussian density dips -> six well-resolved voids at --smooth 1.5; config's frame and catalogue
    knobs pinned by _void_pin_config, caches and figures on temp dirs). Profiles: an .npz + .png
    per redshift with the keys that define R_eff and the stack, named by --prefix when given, an unreadable snapshot
    file counted failed while the NEXT snapshot is still done (exit 1). Population: the abundance figure + npz (the
    other estimator's curves from ITS catalogue), R_eff the one definition (catalog.r_eff_mpc), counts and 'deep'
    at the live threshold, --other-estimator builds catalogues only for the panel snapshots, a header box 1.3x
    config's REFUSED before its cache is touched. (The unreadable-file case of the population script is
    t_void_population_unreadable.)"""
    import io as _io, shutil, sys as _sys, tempfile
    from contextlib import redirect_stdout
    import h5py
    import matplotlib
    import config
    from dtfelib import pipeline, catalog, cli
    from dtfelib import groupcat as gc
    n, box = 64, 64.0
    plot_dir = str(Path(__file__).resolve().parent.parent / "python" / "plot")
    _sys.path.insert(0, plot_dir)
    try:
        import plot_void_profiles as pvp                    # first: its matplotlib.use("Agg")
        import plot_void_population as pop
    finally:
        _sys.path.remove(plot_dir)
    root = Path(tempfile.mkdtemp(prefix="dtfe_voidrun_"))
    saved = {}
    old_argv, old_root, old_voids = list(_sys.argv), cli._data_root_override, pipeline.SnapshotProducts.voids
    old_out, old_fig = pvp.OUTPUT_DIR, pop.FIGURE_ROOT
    old_rc = dict(matplotlib.rcParams.copy())               # style.apply() restyles the process
    data = root / "data"

    def set_box(snap, b):
        with h5py.File(data / "SYN" / f"snapdir_{snap:03d}" / f"combined_{snap:03d}.hdf5", "r+") as f:
            f["Header"].attrs["BoxSize"] = b * 1000.0

    def garbage(snap):                                      # an unreadable snapshot file: h5py raises OSError
        d = data / "SYN" / f"snapdir_{snap:03d}"
        d.mkdir(parents=True, exist_ok=True)
        np.ones(8, np.float32).tofile(str(d / "ps_output.a_den"))
        (d / f"combined_{snap:03d}.hdf5").write_bytes(b"garbage" * 20)
        return d

    def run(mod, args):
        gc.redshift_table.cache_clear()                     # lru-cached on the sim's NAME alone: another run's 'SYN' would leak in
        buf = _io.StringIO()
        _sys.argv = [f"{mod.__name__}.py", "--sim", "SYN", "--data-root", str(data), "--smooth", "1.5"] + args
        with redirect_stdout(buf):
            try:
                mod.main()
                code = 0
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
                if isinstance(e.code, str):
                    print(e.code)                           # a message exit: keep it with the output
        return code, buf.getvalue()

    def npz(path):                                          # read whole and closed: some of these files are rewritten below
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    def cached(snap, method="ps"):
        # the figures' catalogue (--smooth 1.5); the size function keeps its own in another namespace
        hits = [h for h in (root / "cache").rglob(f"SYN_{method}_{snap:03d}_voids.npz") if "_s1.5_" in h.parent.name]
        assert len(hits) == 1, hits
        return hits[0]

    try:
        _void_pin_config(root, saved, n, box)
        pvp.OUTPUT_DIR, pop.FIGURE_ROOT = root / "figures" / "void_profiles", root / "figures" / "void_population"
        _void_ladder_snapdir(data, 50, 0.5, 1, ("ps_output", "output"))
        _void_ladder_snapdir(data, 99, 0.0, 2, ("ps_output", "output", "ps_mw"))
        # ---- plot_void_profiles: two redshifts, two .npz + .png pairs, the keys that say what defines R_eff and the stack
        out = pvp.OUTPUT_DIR / "SYN"
        code, text = run(pvp, ["--snaps", "99", "50", "--edges", "0,10,20", "--n-boot", "37"])
        names = sorted(q.name for q in out.iterdir()) if out.is_dir() else []
        # a non-default --edges tags the names: '_edges0-10-20', the default stack's names stay free
        assert code == 0 and names == [f"void_profiles_ps_resolved_edges0-10-20_z{z}.{e}" for z in ("0.00", "0.50") for e in ("npz", "png")], (code, names, text[-600:])
        assert not (Path(saved["LOCAL_FIGURES_ROOT"]) / "void_profiles" / "SYN").exists()
        f = npz(out / "void_profiles_ps_resolved_edges0-10-20_z0.00.npz")
        need = {"bins", "edges", "count", "r_eff_mpc", "index", "delta_c", "redshift", "sample", "method", "dropped", "points",
                "prefix", "sigma_cells", "cell_mpc", "box_mpc", "n_grid", "n_boot", "min_count", "cuts_mpc"}
        need |= {f"{k}_{q}" for k in ("density", "v_r", "single") for q in ("mean", "lo", "hi")}
        assert need <= set(f), sorted(need - set(f))
        assert str(f["prefix"]) == "" and float(f["sigma_cells"]) == 1.5 and float(f["cell_mpc"]) == 1.0 and float(f["box_mpc"]) == 64.0, f
        assert int(f["n_grid"]) == 64 and int(f["n_boot"]) == 37 and int(f["min_count"]) == 5 and str(f["method"]) == "ps" and str(f["sample"]) == "resolved", f
        assert float(f["redshift"]) == 0.0 and float(npz(out / "void_profiles_ps_resolved_edges0-10-20_z0.50.npz")["redshift"]) == 0.5
        cat99 = npz(cached(99))
        assert np.array_equal(f["r_eff_mpc"], catalog.r_eff_mpc(cat99)[f["index"]]) and np.allclose(f["delta_c"], cat99["delta_values"][f["index"]])
        # the default: the DISTINCT resolved voids with a measured R_v, the shells in r / R_v
        rv99 = catalog.r_v_mpc(cat99)
        dist99 = catalog.distinct_mask(cat99, cat99["well_resolved"].astype(bool) & np.isfinite(rv99), 64)
        want = np.nonzero(dist99 & (3.0 * rv99 <= 32.0))[0]                # and the wrap rule: 3 R_v <= L/2 = 32 cells
        assert str(f["radius"]) == "r_v" and bool(f["distinct"]) and list(f["index"]) == list(want), (list(f["index"]), list(want))
        assert np.allclose(f["r_mpc"], rv99[f["index"]]) and list(f["count"]) == [len(want), 0] and 0 < len(want) <= 6, f["count"]
        assert np.isfinite(f["density_mean"][0]).all() and np.isnan(f["density_mean"][1]).all()
        # --radius r_eff --keep-overlaps: every resolved void, in r / R_eff, its own file
        code, text = run(pvp, ["--snaps", "99", "--edges", "0,10,20", "--radius", "r_eff", "--keep-overlaps"])
        g = npz(out / "void_profiles_ps_resolved_reff_overlaps_edges0-10-20_z0.00.npz")
        assert code == 0 and list(g["count"]) == [6, 0] and str(g["radius"]) == "r_eff" and np.array_equal(g["r_mpc"], g["r_eff_mpc"]), (code, text[-300:])
        # --prefix: the file is named by the set it holds, the primary set's file untouched
        before = (out / "void_profiles_ps_resolved_edges0-10-20_z0.00.npz").read_bytes()
        code, text = run(pvp, ["--snaps", "99", "--prefix", "ps_mw"])
        assert code == 0 and (out / "void_profiles_ps_mw_resolved_z0.00.npz").is_file() and (out / "void_profiles_ps_mw_resolved_z0.00.png").is_file(), (code, sorted(q.name for q in out.iterdir()), text[-400:])
        assert str(npz(out / "void_profiles_ps_mw_resolved_z0.00.npz")["prefix"]) == "ps_mw"
        assert (out / "void_profiles_ps_resolved_edges0-10-20_z0.00.npz").read_bytes() == before
        # --out: the user's folder and nothing into the default one; without --edges the bins are in smoothing lengths
        from dtfelib import profiles as _prof
        own, default_files = root / "own_out", sorted(out.rglob("*"))
        code, text = run(pvp, ["--snaps", "99", "--out", str(own)])
        assert code == 0 and (own / "void_profiles_ps_resolved_z0.00.png").is_file() and sorted(out.rglob("*")) == default_files, (code, text[-400:])
        assert np.allclose(npz(own / "void_profiles_ps_resolved_z0.00.npz")["edges"], _prof.default_edges_mpc(1.5 * 1.0, 64.0))
        # the default bins spelled out by hand are the default (no tag, the same file); other bins never touch it
        dflt = ",".join(f"{e:.4g}" for e in _prof.default_edges_mpc(1.5 * 1.0, 64.0))
        code, text = run(pvp, ["--snaps", "99", "--out", str(own), "--edges", dflt, "--n-boot", "200"])
        assert code == 0 and sorted(q.name for q in own.iterdir()) == ["void_profiles_ps_resolved_z0.00.npz", "void_profiles_ps_resolved_z0.00.png"], (code, sorted(q.name for q in own.iterdir()))
        code, text = run(pvp, ["--snaps", "99", "--out", str(own), "--edges", "0,7.5,20"])
        assert code == 0 and (own / "void_profiles_ps_resolved_edges0-7.5-20_z0.00.npz").is_file(), sorted(q.name for q in own.iterdir())
        assert pvp.edges_tag([0.0, 7.5, 20.0], [0.0, 3.75, 5.25]) == "_edges0-7.5-20" and pvp.edges_tag([0, 3.75, 10.67], [0.0, 3.750001, 32.0 / 3.0]) == ""
        # an unreadable snapshot file FIRST: counted failed, the next snapshot still profiled, exit 1; a missing one too
        d98 = garbage(98)
        shutil.rmtree(out)
        code, text = run(pvp, ["--snaps", "98", "50"])
        names = sorted(q.name for q in out.iterdir()) if out.is_dir() else []
        assert code == 1 and "snapshot 098:" in text and "failed: [98]" in text, (code, text[-600:])
        assert names == ["void_profiles_ps_resolved_z0.50.npz", "void_profiles_ps_resolved_z0.50.png"], names
        code, text = run(pvp, ["--snaps", "67"])
        assert code == 1 and "failed: [67]" in text, (code, text[-300:])
        shutil.rmtree(d98)
        # ---- plot_void_population: the abundance at the panel snapshot, the other estimator's catalogue ONLY there
        calls = []

        def counting(self):
            calls.append((int(self.snapshot), self.fs.method, float(config.SMOOTHING_SIGMA_CELLS)))
            return old_voids(self)
        pipeline.SnapshotProducts.voids = counting
        try:
            code, text = run(pop, ["--panel-snaps", "99", "--other-estimator"])
        finally:
            pipeline.SnapshotProducts.voids = old_voids
        popdir = pop.FIGURE_ROOT / "SYN"
        files = {q.name for q in popdir.iterdir()} if popdir.is_dir() else set()
        assert code == 0 and {"population_evolution.png", "population_distributions.png", "population_abundance.png", "population_stats.npz"} <= files, (code, files, text[-600:])
        # at the figures' smoothing: each snapshot once, the other estimator at the panel only; the size function's own
        # catalogue (plot_void_population --size-smooth) is a second smoothing, of the panel snapshot only
        assert sorted(c[:2] for c in calls if c[2] == 1.5) == [(50, "ps"), (99, "dtfe"), (99, "ps")], calls
        assert all(c[:2] == (99, "ps") for c in calls if c[2] != 1.5), calls
        st = npz(popdir / "population_stats.npz")
        parts = ("n_cum", "n_cum_err", "dn_dlnr", "dn_dlnr_err")
        assert {"snaps", "redshift", "box_mpc", "abundance_radii_mpc", "abundance_edges_mpc"} <= set(st), sorted(st)
        for part in parts:
            assert f"abundance_99_{part}" in st and f"abundance_99_other_{part}" in st and f"abundance_50_{part}" not in st, (part, sorted(st))
        assert list(st["snaps"]) == [50, 99] and np.array_equal(st["box_mpc"], [box, box]) and list(st["n_resolved"]) == [6, 6], (st["snaps"], st["box_mpc"], st["n_resolved"])
        # the abundance from each estimator's OWN cached catalogue and the header's box (config's frame): the PS set
        # has six resolved voids, the standard-DTFE set (one dip fewer) five; counted in R_v, the
        # distinct ones (catalog.distinct_mask)
        cat99 = npz(cached(99))
        r = catalog.r_eff_mpc(cat99)
        rd = catalog.r_eff_mpc(npz(cached(99, "dtfe")))
        assert np.isfinite(r).sum() == 6 and np.isfinite(rd).sum() == 5, (r, rd)

        def counted(c):
            rv_ = catalog.r_v_mpc(c)
            return rv_[catalog.distinct_mask(c, c["well_resolved"].astype(bool) & np.isfinite(rv_), 64)]
        assert counted(cat99).size and np.array_equal(pop.snapshot_stats(cat99, 64)["_r_v"], counted(cat99))
        for tag, rr in (("", counted(cat99)), ("_other", counted(npz(cached(99, "dtfe"))))):
            want = catalog.void_abundance(rr, box, st["abundance_radii_mpc"])[:2] + catalog.void_dn_dlnr(rr, box, st["abundance_edges_mpc"])[:2]
            for part, w in zip(parts, want):
                assert np.array_equal(st[f"abundance_99{tag}_{part}"], w), (tag, part, st[f"abundance_99{tag}_{part}"][:4], w[:4])
        assert np.array_equal(pop.snapshot_stats(cat99)["_r_eff"], r[np.isfinite(r)]), (pop.snapshot_stats(cat99)["_r_eff"], r[np.isfinite(r)])
        # snapshot_stats on a doctored copy: a resolved void that is not BBKS-valid is still resolved (its R_eff
        # kept), and 'deep' is the LIVE threshold's, not the catalogue's frozen flag (here: no void deep)
        e, p = cat99["bbks_params"][:, 0], cat99["bbks_params"][:, 1]
        valid = np.isfinite(e) & np.isfinite(p) & (e >= 0) & (np.abs(p) <= e)
        res = cat99["well_resolved"].astype(bool)
        bb = np.array(cat99["bbks_params"], copy=True)
        bb[np.nonzero(res & valid)[0][0], 0] = -1.0         # e < 0: out of the BBKS sample
        doc = dict(cat99, bbks_params=bb, deep=np.zeros(len(res), bool))
        s1 = pop.snapshot_stats(doc)
        assert s1["n_resolved"] == 6 and s1["n_bbks"] == valid.sum() - 1 and s1["_r_eff"].size == 6, (s1["n_resolved"], s1["n_bbks"], valid.sum())
        dres = np.sort(cat99["delta_values"][res])
        n_deep = int((res & (cat99["delta_values"] < -0.1)).sum())
        assert n_deep > 0 and s1["n_deep"] == n_deep, (s1["n_deep"], n_deep, dres)
        config.DEEP_VOID_THRESHOLD = float(0.5 * (dres[2] + dres[3]))   # retuned between the 3rd and 4th deepest
        try:
            assert pop.snapshot_stats(doc)["n_deep"] == 3, (pop.snapshot_stats(doc)["n_deep"], dres)
        finally:
            config.DEEP_VOID_THRESHOLD = -0.1
        # the names: the run above read PS-DTFE's default grids, so its four files are untagged and its npz
        # says which grids per snapshot; --keep-overlaps and another estimator write their own four, the untagged untouched
        assert list(st["method"]) == ["ps", "ps"] and list(st["prefix"]) == ["ps_output.", "ps_output."], (st["method"], st["prefix"])
        four = ("population_evolution{}.png", "population_distributions{}.png", "population_abundance{}.png", "population_stats{}.npz")
        kept = {f.format(""): (popdir / f.format("")).stat().st_mtime_ns for f in four}
        for args, tag in ((["--keep-overlaps"], "_overlaps"), (["--method", "dtfe"], "_dtfe")):
            code, text = run(pop, args)
            assert code == 0 and all((popdir / f.format(tag)).is_file() for f in four), (args, code, sorted(q.name for q in popdir.iterdir()), text[-400:])
            assert {f.format(""): (popdir / f.format("")).stat().st_mtime_ns for f in four} == kept, (args, "rewrote the untagged files")
        so, sd = npz(popdir / "population_stats_overlaps.npz"), npz(popdir / "population_stats_dtfe.npz")
        assert not bool(so["abundance_distinct"]) and bool(st["abundance_distinct"]) and list(so["method"]) == ["ps", "ps"], so["abundance_distinct"]
        assert list(sd["method"]) == ["dtfe", "dtfe"] and list(sd["prefix"]) == ["output.", "output."], (sd["method"], sd["prefix"])
        # a header box 1.3x config's: REFUSED before its cache is touched -- a cache voids() WOULD re-derive and
        # rewrite (no deep_threshold record) stays byte-identical; the other snapshot carries on (exit 0)
        p50 = cached(50)
        stale = {k: v for k, v in npz(p50).items() if k != "deep_threshold"}
        np.savez_compressed(p50, **stale)
        before = p50.read_bytes()
        set_box(50, 1.3 * box)
        code, text = run(pop, [])
        assert code == 0 and "snap  50: REFUSED" in text and "run with DTFE_SIM=SYN" in text, (code, text[-600:])
        assert list(npz(popdir / "population_stats.npz")["snaps"]) == [99] and p50.read_bytes() == before, "the refused snapshot's cache was rewritten"
        set_box(99, 1.3 * box)                              # every snapshot refused: nothing to analyse, said so, exit 1
        code, text = run(pop, [])
        assert code == 1 and "2 snapshot(s) refused" in text, (code, text[-400:])
    finally:
        for k, v in saved.items():
            setattr(config, k, v)
        _sys.argv = old_argv
        cli.use_data_root(old_root)
        pipeline.SnapshotProducts.voids = old_voids
        pvp.OUTPUT_DIR, pop.FIGURE_ROOT = old_out, old_fig
        dict.update(matplotlib.rcParams, old_rc)
        gc.redshift_table.cache_clear()
        shutil.rmtree(root, ignore_errors=True)


def t_void_crossbox_outlines():
    """plot_void_crossbox.py's per-box worker and figure, tools/void_outlines.py (the launcher's void outlines) and
    pipeline.adopt_snapshot_grid, in process on the synthetic 64^3 ladder (_void_pin_config): the worker's counts and
    R_v are snapshot_stats', its stacks void_profiles' over the common R_v range, a redshift without a snapshot left
    empty; --replot draws a figure per redshift and the table, a missing box said and exit 1; the outline tool's
    resolved centres sit on the dips, 'distinct' is catalog.distinct_mask's, its JSON strict; adopt_snapshot_grid
    takes the grid's frame keeping sigma in Mpc and is a no-op when config is in it."""
    import io as _io, json, os, sys as _sys, shutil, tempfile
    from contextlib import redirect_stdout
    import matplotlib
    import config
    from dtfelib import pipeline, catalog, cli, profiles
    from dtfelib import groupcat as gc
    n, box = 64, 64.0
    here = Path(__file__).resolve().parent.parent / "python"
    dirs = [str(here / "plot"), str(here / "tools")]
    _sys.path[:0] = dirs
    try:
        import plot_void_population as pop
        import plot_void_crossbox as xb
        import void_outlines as vo
    finally:
        for d in dirs:
            _sys.path.remove(d)
    root = Path(tempfile.mkdtemp(prefix="dtfe_crossbox_"))
    saved = {}
    old_root, old_rc = cli._data_root_override, dict(matplotlib.rcParams.copy())
    data, out = root / "data", root / "out"
    try:
        _void_pin_config(root, saved, n, box)
        saved["SMOOTHING_SIGMA_MPC"] = config.SMOOTHING_SIGMA_MPC       # adopt_snapshot_grid below moves it
        config.SMOOTHING_SIGMA_MPC = config.SMOOTHING_SIGMA_CELLS * config.CELL_SIZE
        _void_ladder_snapdir(data, 50, 0.5, 1, ("ps_output",))
        _void_ladder_snapdir(data, 99, 0.0, 2, ("ps_output",))
        gc.redshift_table.cache_clear()
        sig = pipeline.smoothing_length_mpc()
        with redirect_stdout(_io.StringIO()):
            res = xb.measure("SYN", "ps", None, data, zs=(0.0, 0.5, 3.0), rv_range_sigma=(0.0, 100.0), points=60,
                             n_boot=20)
        assert list(res["snaps"]) == [50, 99] and np.allclose(res["redshift"], [0.5, 0.0]), (res["snaps"], res["redshift"])
        prods = {s: pipeline.products(s, sim="SYN", method="ps", data_root=data) for s in (50, 99)}
        for j, s in enumerate((50, 99)):
            st = pop.snapshot_stats(prods[s].voids(), n)
            for k in ("n_total", "n_resolved", "n_counted", "n_no_rv", "n_overlapping"):
                assert res[k][j] == st[k], (s, k, res[k][j], st[k])
            assert np.array_equal(xb.r_v_at(res, s), st["_r_v"]), s
        assert list(res["pz_snap"]) == [99, 50, -1] and res["pz_count"][0] > 0, (res["pz_snap"], res["pz_count"])
        assert list(res["pz_count"][:2]) == list(res["pz_profiled"][:2]), "every profiled void lies in a 0-150 Mpc range"
        assert np.isnan(res["pz_density_mean"][2]).all(), "no snapshot near z = 3: its stack stays empty"
        prof = profiles.void_profiles(prods[99], m=60, sample="resolved", radius="r_v", distinct=True)
        want = profiles.stack(prof, edges_mpc=[0.0, 100.0 * sig], n_boot=20)
        assert np.allclose(res["pz_density_mean"][0], want["density"]["mean"][0], equal_nan=True)
        assert np.allclose(res["pz_v_r_lo"][0], want["v_r"]["lo"][0], equal_nan=True)
        for p_ in prods.values():
            p_.release()
        # ---- the figure from the saved .npz files: two 'boxes', a third one missing
        out.mkdir()
        np.savez_compressed(xb.worker_path(out, "SYN", "ps", sig), **res)
        np.savez_compressed(xb.worker_path(out, "SYN2", "ps", sig), **dict(res, sim="SYN2"))
        buf = _io.StringIO()
        with redirect_stdout(buf):
            try:
                xb.main(["--replot", "--sims", "SYN", "SYN2", "SYN3", "--out", str(out), "--smooth-mpc", repr(sig),
                         "--z", "0", "0.5", "3"])
                code = 0
            except SystemExit as e:
                code = e.code
        names = sorted(q.name for q in out.iterdir())
        tag = xb.sigma_tag(sig)
        assert code == 1 and "SYN3: no" in buf.getvalue(), (code, buf.getvalue()[-400:])
        assert all(f"crossbox_ps_{tag}_z{z}.png" in names for z in ("0.00", "0.50", "3.00")) and f"crossbox_ps_{tag}.txt" in names, names
        table = (out / f"crossbox_ps_{tag}.txt").read_text()
        assert table.count("SYN (ps): box 64 Mpc") == 1 and "SYN2 (ps)" in table, table[:300]
        # ---- the launcher's void outlines: config already in the grid's frame (adopt: nothing changes); the
        # production set (no prefix) reads the catalogue the analysis cached above, so even ~no free memory is
        # enough, while a prefix of its own needs a build, which the memory guard refuses then
        old_free = os.environ.get("DTFE_FREE_GB")
        os.environ["DTFE_FREE_GB"] = "0.001"
        try:
            with redirect_stdout(_io.StringIO()):
                o = vo.outlines("SYN", 99, None, data, "ps")
                try:
                    vo.outlines("SYN", 99, "ps_output", data, "ps")
                    refused = ""
                except MemoryError as e:
                    refused = str(e)
        finally:
            if old_free is None:
                os.environ.pop("DTFE_FREE_GB", None)
            else:
                os.environ["DTFE_FREE_GB"] = old_free
        assert "not cached" in refused and "64^3" in refused, refused
        json.dumps(o, allow_nan=False)                      # strict JSON: NaN radii are null
        cat = pipeline.products(99, sim="SYN", method="ps", data_root=data)
        c99 = cat.voids()
        cat.release()
        assert o["n_grid"] == n and o["box_mpc"] == box and len(o["centres_mpc"]) == len(c99["coords"]), o["n_grid"]
        res_rows = np.nonzero(o["resolved"])[0]
        dips = np.array([(16, 16, 16), (44, 20, 40), (20, 48, 30), (50, 50, 10), (8, 36, 54), (36, 6, 26)]) + 0.5
        cen = np.asarray(o["centres_mpc"])[res_rows]
        dist = np.abs(((cen[:, None, :] - dips[None]) + box / 2) % box - box / 2).max(axis=2).min(axis=1)
        assert res_rows.size == 6 and dist.max() <= 1.5, (res_rows.size, dist)
        rv = catalog.r_v_mpc(c99)
        assert o["distinct"] == catalog.distinct_mask(c99, c99["well_resolved"].astype(bool) & np.isfinite(rv), n).tolist()
        # ---- adopt_snapshot_grid: a 512 frame meets the 64^3 grid: the grid's frame, sigma kept in Mpc
        config.FIELD_RESOLUTION, config.CELL_SIZE = 512, box / 512
        before = config.SMOOTHING_SIGMA_CELLS * config.CELL_SIZE        # 0.1875 Mpc: under one cell of the 64^3 grid
        pr = pipeline.products(99, sim="SYN", method="ps", data_root=data)
        try:
            got = pipeline.adopt_snapshot_grid(pr)
            assert got == (1.0, box, n) and config.FIELD_RESOLUTION == n and config.CELL_SIZE == 1.0, got
            assert np.isclose(config.SMOOTHING_SIGMA_CELLS * config.CELL_SIZE, before) and np.isclose(config.SMOOTHING_SIGMA_MPC, before)
            assert config.FOOTPRINT_SIZE == config.footprint_for(config.SMOOTHING_SIGMA_CELLS)
            sig_cells = config.SMOOTHING_SIGMA_CELLS
            assert pipeline.adopt_snapshot_grid(pr) == got and config.SMOOTHING_SIGMA_CELLS == sig_cells
        finally:
            pr.release()
    finally:
        for k, v in saved.items():
            setattr(config, k, v)
        cli.use_data_root(old_root)
        dict.update(matplotlib.rcParams, old_rc)
        gc.redshift_table.cache_clear()
        shutil.rmtree(root, ignore_errors=True)


def t_void_population_unreadable():
    """plot_void_population.py on a ladder of one good synthetic snapshot and one unreadable snapshot file (a garbage
    combined_098.hdf5 beside a grid): the script's own loop branch -- 'snap 98: unreadable, left out', the rest
    analysed, exit 0. groupcat.redshift_table, called BEFORE that loop, opens every combined_*.hdf5 of the ladder: an
    unreadable one is no anchor (interpolated), not an OSError that aborts the whole run."""
    import io as _io, shutil, sys as _sys, tempfile
    from contextlib import redirect_stdout
    import matplotlib
    import config
    from dtfelib import cli
    from dtfelib import groupcat as gc
    matplotlib.use("Agg")                                   # headless, as the plot scripts' test imports leave it
    plot_dir = str(Path(__file__).resolve().parent.parent / "python" / "plot")
    _sys.path.insert(0, plot_dir)
    try:
        import plot_void_population as pop
    finally:
        _sys.path.remove(plot_dir)
    root = Path(tempfile.mkdtemp(prefix="dtfe_voidbad_"))
    saved, old_argv, old_root, old_fig = {}, list(_sys.argv), cli._data_root_override, pop.FIGURE_ROOT
    old_rc = dict(matplotlib.rcParams.copy())
    data = root / "data"
    try:
        _void_pin_config(root, saved)
        pop.FIGURE_ROOT = root / "figures" / "void_population"
        _void_ladder_snapdir(data, 99, 0.0, 2, ("ps_output",))
        d98 = data / "SYN" / "snapdir_098"
        d98.mkdir(parents=True)
        np.ones(8, np.float32).tofile(str(d98 / "ps_output.a_den"))       # a grid: the snapshot FILE is what fails
        (d98 / "combined_098.hdf5").write_bytes(b"garbage" * 20)
        gc.redshift_table.cache_clear()                     # lru-cached on the sim's NAME alone: a cached 'SYN' table hides it
        _sys.argv = ["plot_void_population.py", "--sim", "SYN", "--data-root", str(data), "--smooth", "1.5"]
        buf, code = _io.StringIO(), 0
        with redirect_stdout(buf):
            try:
                pop.main()
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
                if isinstance(e.code, str):
                    print(e.code)
            except OSError as e:                            # name the frame that let it through
                import traceback
                where = [fr.name for fr in traceback.extract_tb(e.__traceback__) if "h5py" not in fr.filename]
                raise AssertionError(f"plot_void_population aborted on an unreadable combined_098.hdf5 (raised in {where[-1]}): {e}")
        text = buf.getvalue()
        st = pop.FIGURE_ROOT / "SYN" / "population_stats.npz"
        assert code == 0 and "snap  98: unreadable, left out" in text and st.is_file(), (code, text[-600:])
        with np.load(st) as z:
            assert list(z["snaps"]) == [99], list(z["snaps"])
    finally:
        for k, v in saved.items():
            setattr(config, k, v)
        _sys.argv = old_argv
        cli.use_data_root(old_root)
        pop.FIGURE_ROOT = old_fig
        dict.update(matplotlib.rcParams, old_rc)
        gc.redshift_table.cache_clear()
        shutil.rmtree(root, ignore_errors=True)


def t_power_spectrum():
    """The P(k) estimator on a field built with a KNOWN spectrum (a flat one
    comes back exactly, a power law within the bin average), Parseval over every mode, the Hermitian mode
    count -- for an even AND an odd N (the k_z = N/2 plane is self-conjugate for even N only);
    the linear spectrum: sigma_8 reproduced at 8/h Mpc, T(k -> 0) = 1, the growth factor and rate,
    the turnover near k_eq."""
    import config
    from dtfelib import spectra
    box = 100.0
    for n in (33, 64):                  # 33: the last rfft plane is NOT self-conjugate; 64 last, the power law below uses it
        rng = np.random.default_rng(3)
        white = np.fft.rfftn(rng.normal(size=(n, n, n)))                 # Hermitian-consistent random phases
        kx = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
        kz = 2 * np.pi * np.fft.rfftfreq(n, d=box / n)
        kmag = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)

        def field_with(p_of_k):
            amp = np.sqrt(p_of_k(np.where(kmag > 0, kmag, 1.0)) * n ** 6 / box ** 3)
            amp[0, 0, 0] = 0.0
            dk = white / np.abs(white) * amp                               # the magnitude fixed, the phases kept
            return np.fft.irfftn(dk, s=(n, n, n), axes=(0, 1, 2))
        flat = field_with(lambda k: np.full(k.shape, 500.0))
        ps = spectra.power_spectrum(flat, box)
        ok = ps["modes"] > 0
        assert np.allclose(ps["P"][ok], 500.0, rtol=1e-5), (n, ps["P"][ok][:5])   # a flat spectrum: every bin exact
        assert abs(ps["k_f"] - 2 * np.pi / box) < 1e-12 and abs(ps["k_nyquist"] - np.pi * n / box) < 1e-12
        assert ps["k"][ok][0] > 0 and (np.diff(ps["k"][ok]) > 0).all() and ps["modes"][ok][0] == 18, (n, ps["modes"][ok][:3])   # 6 fundamental + 12 sqrt(2) modes
        everything = spectra.power_spectrum(flat, box, kmax=2.0 * ps["k_nyquist"])
        assert everything["modes"].sum() == n ** 3 - 1, (n, everything["modes"].sum())   # every mode but k = 0, once each
        parseval = np.nansum(everything["P"] * everything["modes"]) / box ** 3
        assert abs(parseval - flat.var()) < 1e-6 * flat.var(), (n, parseval, flat.var())   # Parseval
    law = field_with(lambda k: 500.0 * (k / 0.1) ** -1.5)              # n = 64, the loop's last
    pl = spectra.power_spectrum(law, box, nbins=20)
    okl = pl["modes"] > 0
    ratio = pl["P"][okl] / (500.0 * (pl["k"][okl] / 0.1) ** -1.5)
    assert np.abs(ratio - 1.0).max() < 0.15, ratio                                  # the bin average of a power law
    try:
        spectra.power_spectrum(np.zeros((8, 8, 4)), box)
        raise AssertionError("a non-cubic grid must refuse")
    except ValueError:
        pass
    # linear theory
    h = config.HUBBLE_H
    assert abs(spectra.sigma_r(lambda k: spectra.linear_pk(k, 0.0), 8.0 / h) - config.COSMOLOGY["sigma8"]) < 1e-4 * config.COSMOLOGY["sigma8"]
    assert abs(spectra.transfer_eh98(1e-5) - 1.0) < 1e-3 and spectra.transfer_eh98(1.0) < 0.05
    assert abs(spectra.growth_factor(0.0) - 1.0) < 1e-12 and spectra.growth_factor(1.0) < spectra.growth_factor(0.5) < 1.0
    d99, d199 = spectra.growth_factor(99.0) * 100.0, spectra.growth_factor(199.0) * 200.0     # D ~ a early on
    assert abs(d99 / d199 - 1.0) < 0.01, (d99, d199)
    om = config.COSMOLOGY["Omega_m"]
    for z in (0.0, 1.0, 3.0):
        om_z = om * (1 + z) ** 3 / (om * (1 + z) ** 3 + config.COSMOLOGY["Omega_Lambda"])
        assert abs(spectra.growth_rate(z) / om_z ** 0.55 - 1.0) < 0.01, (z, spectra.growth_rate(z), om_z ** 0.55)
    k = np.geomspace(1e-4, 10.0, 4000)
    plin = spectra.linear_pk(k, 0.0)
    k_eq = 0.0746 * om * h * h * (spectra.T_CMB / 2.7) ** -2          # EH98 eq 3: already in 1/Mpc (0.01038 here)
    k_peak = k[np.argmax(plin)]
    assert k_eq <= k_peak <= 1.5 * k_eq, (k_peak, k_eq)               # 0.0110 /Mpc with eq 30 on k in 1/Mpc; 0.0096 with k in h/Mpc fails
    # T(k) itself, against Eisenstein's TFnowiggles evaluated for config.COSMOLOGY (a sound-horizon term taking k in
    # h/Mpc, 1/h too far in k, is 5% off at k = 0.02 /Mpc after the sigma_8 renormalisation)
    assert abs(spectra.transfer_eh98(0.02) / 0.44182 - 1.0) < 1e-3 and abs(spectra.transfer_eh98(0.01) / 0.6847 - 1.0) < 2e-3, (spectra.transfer_eh98(0.02), spectra.transfer_eh98(0.01))
    from scipy.integrate import quad
    r8 = 8.0 / h
    def w2(x):
        return (3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3) ** 2
    s8_quad = np.sqrt(quad(lambda lk: spectra.linear_pk(np.exp(lk), 0.0) * w2(np.exp(lk) * r8) * np.exp(lk) ** 3 / (2 * np.pi ** 2), np.log(1e-5), np.log(1e2), limit=400)[0])
    assert abs(s8_quad - config.COSMOLOGY["sigma8"]) < 2e-3 * config.COSMOLOGY["sigma8"], s8_quad   # an integral of its own, not sigma_r's
    assert np.allclose(spectra.linear_pk(k, 1.0) / plin, spectra.growth_factor(1.0) ** 2)        # D^2 scaling
    assert abs(plin[0] / plin[1] - (k[0] / k[1]) ** config.COSMOLOGY["n_s"]) < 5e-3 * (k[0] / k[1]) ** config.COSMOLOGY["n_s"]


def t_pk_script():
    """plot_pk_compare.py in process on a synthetic snapshot: the file is named by what it holds (both estimators,
    one --method, a --prefix), a lone estimator gets a note, and plot_pk draws each
    estimator's own Nyquist pair when the grids differ in size (one grey pair when they agree)."""
    import io as _io, shutil, sys as _sys, tempfile
    from contextlib import redirect_stdout
    import matplotlib.pyplot as plt
    from dtfelib import spectra
    root = Path(tempfile.mkdtemp(prefix="dtfe_pkrun_"))
    old_argv = list(_sys.argv)
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "plot"))
    try:
        import plot_pk_compare as ppk
        data = root / "data"
        _synthetic_snapdir(data, "SYN", 50, 0.5, n=8, grids=("den",))
        _synthetic_snapdir(data, "SYN", 50, 0.5, n=8, grids=("den",), prefix="output")
        _synthetic_snapdir(data, "SYN", 50, 0.5, n=8, grids=("den",), prefix="ps_mw")
        _synthetic_snapdir(data, "SYN", 99, 0.0, n=8, grids=("den",))                   # PS only

        def run(args):
            out = root / "out"
            buf = _io.StringIO()
            _sys.argv = ["plot_pk_compare.py", "--sim", "SYN", "--data-root", str(data), "--out", str(out)] + args
            with redirect_stdout(buf):
                try:
                    ppk.main(); code = 0
                except SystemExit as e:
                    code = int(e.code or 0)
            return code, buf.getvalue(), sorted(q.name for q in out.iterdir())
        code, text, names = run(["--snaps", "50", "99"])
        assert code == 0 and names == ["pk_dtfe+ps_z0.50.npz", "pk_dtfe+ps_z0.50.png", "pk_ps_z0.00.npz", "pk_ps_z0.00.png"], (code, names, text[-300:])
        assert "only PS-DTFE grids here" in text
        z = np.load(root / "out" / "pk_dtfe+ps_z0.50.npz")
        assert list(z["methods"]) == ["dtfe", "ps"] and str(z["prefix"]) == "" and z["dtfe_k"].shape == z["ps_k"].shape
        code, text, names = run(["--snaps", "50", "--method", "dtfe"])
        assert code == 0 and "pk_dtfe_z0.50.npz" in names and "pk_dtfe+ps_z0.50.npz" in names, names     # the earlier file untouched
        code, text, names = run(["--snaps", "50", "--prefix", "ps_mw"])
        assert code == 0 and "pk_ps_mw_z0.50.png" in names and str(np.load(root / "out" / "pk_ps_mw_z0.50.npz")["prefix"]) == "ps_mw"
        code, text, names = run(["--snaps", "98"])
        assert code == 1 and "failed: [98]" in text, (code, text[-200:])
        code, text, names = run(["--snaps", "99", "--ref-snap", "50"])
        assert code == 0 and "reference snapshot 050" in text, text[-300:]
        # non-default --nbins / --ref-snap are tagged, the default names untouched: untagged, a rerun with other bins
        # or another reference would write over the default figure and npz
        assert "pk_ps_ref50_z0.00.npz" in names and "pk_ps_z0.00.npz" in names, names
        before = (root / "out" / "pk_ps_z0.00.npz").read_bytes()
        code, text, names = run(["--snaps", "99", "--nbins", "0", "--ref-snap", "-1"])
        assert code == 0 and "pk_ps_nb0_noref_z0.00.npz" in names and "pk_ps_nb0_noref_z0.00.png" in names, names
        code, text, names = run(["--snaps", "99", "--nbins", "12"])
        assert code == 0 and "pk_ps_nb12_z0.00.npz" in names and (root / "out" / "pk_ps_z0.00.npz").read_bytes() == before, names
        code, text, names = run(["--snaps", "99", "--nbins", "40", "--ref-snap", "0"])           # the defaults spelled out
        assert code == 0 and not [q for q in names if q.startswith("pk_ps_") and "_z0.00" in q and q not in (
            "pk_ps_z0.00.npz", "pk_ps_z0.00.png", "pk_ps_ref50_z0.00.npz", "pk_ps_ref50_z0.00.png", "pk_ps_nb0_noref_z0.00.npz",
            "pk_ps_nb0_noref_z0.00.png", "pk_ps_nb12_z0.00.npz", "pk_ps_nb12_z0.00.png")], names
        lo, hi = ppk.chi2_interval(np.array([1.0]), np.array([6]))      # M = 6: the truth 0.65-2.19x the measurement
        assert 0.6 < lo[0] < 0.7 and 2.1 < hi[0] < 2.3, (lo, hi)
        # mixed grid sizes: each estimator's own k_Ny and k_Ny/2 lines; the same size: one grey pair
        rng = np.random.default_rng(1)
        p32 = spectra.power_spectrum(rng.normal(size=(32, 32, 32)), 100.0)
        p64 = spectra.power_spectrum(rng.normal(size=(64, 64, 64)), 100.0)
        ppk.plot_pk({"dtfe": p32, "ps": p64}, 0.0, 100.0, "t", root / "mixed.png")
        ppk.plot_pk({"dtfe": p32, "ps": p32}, 0.0, 100.0, "t", root / "same.png")
        assert (root / "mixed.png").is_file() and (root / "same.png").is_file()
        # count the vertical lines plot_pk draws (a probe of the figure before it is closed is not possible: it closes
        # the figure; redo the drawing with a patched close to look at the axes)
        held = {}
        real_close = plt.close
        plt.close = lambda fig=None: held.setdefault("fig", fig)
        try:
            ppk.plot_pk({"dtfe": p32, "ps": p64}, 0.0, 100.0, "t", root / "mixed2.png")
            ax = held["fig"].axes[0]
            xs = sorted(round(float(ln.get_xdata()[0]), 6) for ln in ax.lines if len(set(ln.get_xdata())) == 1)
            assert xs == sorted(round(v, 6) for v in (p32["k_nyquist"], p32["k_nyquist"] / 2, p64["k_nyquist"], p64["k_nyquist"] / 2)), xs
            assert max(ax.lines[0].get_xdata()) >= 1.49 * p64["k_nyquist"]                # the linear curve reaches the larger k_Ny
            held.clear()
            ppk.plot_pk({"dtfe": p32, "ps": p32}, 0.0, 100.0, "t", root / "same2.png")
            ax = held["fig"].axes[0]
            xs = sorted(round(float(ln.get_xdata()[0]), 6) for ln in ax.lines if len(set(ln.get_xdata())) == 1)
            assert xs == sorted(round(v, 6) for v in (p32["k_nyquist"], p32["k_nyquist"] / 2)), xs
            assert held["fig"].axes[1].get_yscale() == "log"
            # the same-realisation panel: an early spectrum grown linearly divides out (exactly 1 here), and the band
            held.clear()
            g = (float(spectra.growth_factor(0.0)) / float(spectra.growth_factor(20.0))) ** 2
            ppk.plot_pk({"ps": p32}, 0.0, 100.0, "t", root / "ref.png", ref=({"ps": dict(p32, P=p32["P"] / g)}, 20.0))
            axes = held["fig"].axes
            assert len(axes) == 3 and np.allclose(axes[2].lines[0].get_ydata(), 1.0), len(axes)
            assert len(axes[1].collections) >= 1                         # the 95% Gaussian band where Delta^2_lin < 1
            held.clear()
            ppk.plot_pk({"ps": p32}, 0.0, 100.0, "t", root / "noref.png", ref=({"ps": p32}, 0.0))   # same z: no panel
            assert len(held["fig"].axes) == 2
            held.clear()
            p16 = spectra.power_spectrum(rng.normal(size=(16, 16, 16)), 100.0, nbins=None)   # other bins: no crash, no panel
            ppk.plot_pk({"ps": spectra.power_spectrum(rng.normal(size=(32, 32, 32)), 100.0, nbins=None)}, 0.0, 100.0, "t",
                        root / "otherbins.png", ref=({"ps": p16}, 20.0))
            assert len(held["fig"].axes) == 2 and not ppk.same_edges(p16["edges"], p32["edges"])
        finally:
            plt.close = real_close
            for f in held.values():
                real_close(f)
    finally:
        _sys.argv = old_argv
        shutil.rmtree(root, ignore_errors=True)


def t_caustic_skeleton():
    """The classes and their fractions on a synthetic caustic_class grid, the
    periodic merge of a wall split by the box face and of two cells across one face, an edge or a corner
    (6 / 18 / 26), the tilted sheet under 26 vs 6, component sizes and mass, the cusp and swallowtail bits
    told apart and 'not computed' told from 0%, the parity-only refusal, json_safe, the per-void shell
    fractions (fold vs sheet vs line, the mean of the 1.0-1.3 R_eff shells against Archimedes' hat-box) and the
    nearest-wall distance: both 'beyond the cut-out' branches, d = half exact, the cut-out wrapping every axis.
    The real crossed-waves run is t_caustic_crossed_waves (data-backed)."""
    import json
    from scipy import ndimage
    from dtfelib import skeleton as sk
    from dtfelib.io import (CAUSTIC_PARITY_POS, CAUSTIC_PARITY_NEG, CAUSTIC_COLLAPSE, CAUSTIC_DEGENERATE, CAUSTIC_CUSP,
                            CAUSTIC_SWALLOWTAIL)
    n = 32
    X, Y, Z = np.indices((n, n, n))
    fold = CAUSTIC_PARITY_POS | CAUSTIC_PARITY_NEG
    cc = np.full((n, n, n), CAUSTIC_COLLAPSE[0], dtype=np.int32)      # everything deposited, uncollapsed
    zplanes = (Z == 0) | (Z == n - 1)                                 # a wall of two planes meeting through the wrap
    tilted = (((X + Y) % n) == 10) & (Z >= 4) & (Z <= 27)              # a sheet tilted 45 degrees, away from the planes
    line = (Y == 5) & (Z == 5)                                        # a filament along x, crossing the tilted sheet once
    cc[zplanes | tilted] |= CAUSTIC_COLLAPSE[1] | fold
    cc[line] |= CAUSTIC_COLLAPSE[2]
    cc[3, 5, 5] |= CAUSTIC_COLLAPSE[3]                                # a node on the line
    cc[20, 20, 20] = 0                                                # never deposited into: no parity bit, no refusal
    cc[1, 1, 1] |= CAUSTIC_DEGENERATE
    m = sk.class_masks(cc)
    assert m["exclusive"]["none"].sum() == 1 and m["exclusive"]["k=3"].sum() == 1 and m["exclusive"]["k=2"].sum() == n - 1
    assert m["exclusive"]["k=1"].sum() == ((zplanes | tilted) & ~line).sum() and m["cumulative"]["sheet"].sum() == (zplanes | tilted | line).sum()
    assert m["indicator"]["umbilic"].sum() == 1 and m["indicator"]["fold"].sum() == (zplanes | tilted).sum()
    den = np.ones((n, n, n), np.float32)
    den[zplanes] = 2.0
    fr = sk.fractions(cc, den)
    assert abs(sum(v["volume"] for v in fr["exclusive"].values()) - 1.0) < 1e-12 and abs(sum(v["mass"] for v in fr["exclusive"].values()) - 1.0) < 1e-12
    assert abs(fr["exclusive"]["none"]["volume"] - 1 / n ** 3) < 1e-15 and fr["cumulative"]["sheet"]["mass"] > fr["cumulative"]["sheet"]["volume"]
    # a parity-only grid (--ps-caustics --ps-linear-deposit --ps-gpu: the fold flag without collapse bits) is refused, alone
    # or as one det>0-parity cell in a full deposit; a parity bit WITH a collapse bit is an ordinary deposit.
    # A lone det<0 bit is legitimate (the geometric overlap, in a cell a folded tetrahedron overlaps but
    # no sample reached): read, as class 'none' and not a fold
    one_pos, one_neg = cc.copy(), cc.copy()
    one_pos[20, 20, 20], one_neg[20, 20, 20] = CAUSTIC_PARITY_POS, CAUSTIC_PARITY_NEG
    mn = sk.class_masks(one_neg)
    assert mn["exclusive"]["none"][20, 20, 20] and not mn["indicator"]["fold"][20, 20, 20] and mn["exclusive"]["none"].sum() == 1
    for bad in (np.full((4, 4, 4), fold, np.int32), np.full((4, 4, 4), CAUSTIC_PARITY_POS, np.int32), one_pos):
        try:
            sk.class_masks(bad)
        except ValueError as e:
            assert "parity-only" in str(e), e
        else:
            raise AssertionError("a parity-only caustic_class grid was read as a stratification")
    assert sk.class_masks(np.full((4, 4, 4), fold | CAUSTIC_COLLAPSE[0], np.int32))["exclusive"]["k=0"].all()
    # fold= replaces the mask's own fold bits (FieldSet.fold_mask(): the exact flag of an older grid), and only
    # them; fractions and skeleton_summary pass it on; a wrong shape is refused
    exact = m["indicator"]["fold"].copy()
    exact[0:3, 0:3, 8] = True                                         # folds the samples missed: k = 0 cells
    mf = sk.class_masks(cc, fold=exact)
    assert np.array_equal(mf["indicator"]["fold"], exact) and all(np.array_equal(mf["exclusive"][k], m["exclusive"][k]) for k in sk.CLASSES)
    assert abs(sk.fractions(cc, fold=exact)["indicator"]["fold"]["volume"] - exact.mean()) < 1e-15
    assert abs(sk.skeleton_summary(cc, fold=exact)["fractions"]["indicator"]["fold"]["volume"] - exact.mean()) < 1e-15
    try:
        sk.class_masks(cc, fold=exact[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("a fold mask of another shape was accepted")
    # the periodic merge: without the wrap the two planes are two components, with it one
    assert ndimage.label(zplanes)[1] == 2 and sk.label_periodic(zplanes, 26)[1] == 1 and sk.label_periodic(zplanes, 6)[1] == 1
    # two cells across ONE box face each: one component under every connectivity (the planes and the tilted loop
    # stay connected when one axis' wrap is dropped); across a box edge 18 and 26 join them, across a
    # corner 26 only
    k = 8

    def pair(a, b):
        mk = np.zeros((k, k, k), bool)
        mk[a] = mk[b] = True
        return mk
    for a, b in (((0, 3, 4), (k - 1, 3, 4)), ((2, 0, 4), (2, k - 1, 4)), ((2, 3, 0), (2, 3, k - 1))):
        assert ndimage.label(pair(a, b))[1] == 2, (a, b)
        for conn in (6, 18, 26):
            assert sk.label_periodic(pair(a, b), conn)[1] == 1, (a, b, conn)
    edge, corner = pair((0, 0, 4), (k - 1, k - 1, 4)), pair((0, 0, 0), (k - 1, k - 1, k - 1))
    assert [sk.label_periodic(edge, c)[1] for c in (6, 18, 26)] == [2, 1, 1], [sk.label_periodic(edge, c)[1] for c in (6, 18, 26)]
    assert [sk.label_periodic(corner, c)[1] for c in (6, 18, 26)] == [2, 2, 1], [sk.label_periodic(corner, c)[1] for c in (6, 18, 26)]
    sheet = m["cumulative"]["sheet"]
    assert sk.label_periodic(sheet, 26)[1] == 2, sk.label_periodic(sheet, 26)[1]       # the planes; the tilted sheet with its line
    # 6-connected: the tilted sheet shreds into 32 columns, the line's face neighbours (y = 4, 5, 6) join three of them
    assert sk.label_periodic(sheet, 6)[1] == 31, sk.label_periodic(sheet, 6)[1]
    comp = sk.components(sheet, den, cell_mpc=0.5, connectivity=26)
    assert comp["n"] == 2 and comp["cells"][0] == zplanes.sum() and comp["cells"][1] == (tilted | line).sum()
    assert abs(comp["volume_mpc3"][0] - zplanes.sum() * 0.125) < 1e-9 and abs(comp["mass_fraction"][0] - 2.0 * zplanes.sum() / den.sum()) < 1e-12
    assert abs(comp["largest_fraction"] - zplanes.sum() / sheet.sum()) < 1e-12 and (comp["labels"] > 0).sum() == sheet.sum()
    lab26, _ = sk.label_periodic(sheet, 26)
    assert len(np.unique(lab26[zplanes])) == 1 and len(np.unique(lab26[tilted])) == 1 and lab26[3, 5, 5] == lab26[5, 5, 5]
    summ = sk.skeleton_summary(cc, den, cell_mpc=0.5)
    json.dumps(summ)
    assert summ["components"]["sheet"]["n"] == 2 and summ["components"]["line"]["n"] == 1 and summ["components"]["node"]["n"] == 1
    assert summ["components"]["sheet"]["top_cells"][0] == int(zplanes.sum()) and summ["fractions"]["indicator"]["cusp"]["volume"] == 0.0
    # no bit 7 or 8 anywhere: cusp/swallowtail were NOT COMPUTED (CPU without --ps-caustic-cusps, or the GPU merge), not
    # 0%;
    # one cusp and two swallowtail cells are told apart, and either bit alone sets the flag
    assert summ["cusp_bits_present"] is False, summ["cusp_bits_present"]
    cc2 = cc.copy()
    cc2[7, 20, 12] |= CAUSTIC_CUSP
    cc2[8, 20, 12] |= CAUSTIC_SWALLOWTAIL
    cc2[9, 20, 12] |= CAUSTIC_SWALLOWTAIL
    m2 = sk.class_masks(cc2)
    assert m2["indicator"]["cusp"].sum() == 1 and m2["indicator"]["cusp"][7, 20, 12] and m2["indicator"]["swallowtail"].sum() == 2
    s2 = sk.skeleton_summary(cc2, den)
    assert s2["cusp_bits_present"] is True and abs(s2["fractions"]["indicator"]["cusp"]["volume"] - 1 / n ** 3) < 1e-15
    assert abs(s2["fractions"]["indicator"]["swallowtail"]["volume"] - 2 / n ** 3) < 1e-15
    for bit in (CAUSTIC_CUSP, CAUSTIC_SWALLOWTAIL):
        cc3 = cc.copy()
        cc3[8, 20, 12] |= bit
        assert sk.skeleton_summary(cc3)["cusp_bits_present"] is True, bit
    # json_safe: NaN / inf -> None (bare NaN is not JSON), also inside a float array; numpy scalars and arrays -> Python
    js = sk.json_safe({"a": np.float32("nan"), "b": [np.inf, 1.5], "c": np.arange(2), "d": np.bool_(True), "e": np.int64(3),
                       "f": (float("nan"), None), "g": np.array([[np.nan, 2.5], [-np.inf, 0.0]], np.float32)})
    assert js == {"a": None, "b": [None, 1.5], "c": [0, 1], "d": True, "e": 3, "f": [None, None], "g": [[None, 2.5], [None, 0.0]]}, js
    assert type(js["e"]) is int and type(js["d"]) is bool and json.dumps(js, allow_nan=False)
    # per void: a void far from every sheet (the tilted sheet's two branches x + y = 10 and 42 are 11.3 cells from
    # cell (13, 13), the z planes 15.5), one whose outer shells cross the z planes (the filament is ~20 cells off: a
    # 'line' mask reads 0 there), and a small one centred ON the filament, 5.5 cells from the planes and 10.6 from the
    # tilted sheet: sheet without fold (a fold<->sheet mix-up reads the same at the other two)
    sh = sk.void_shell_fractions(cc, [(13.5 / n, 13.5 / n, 16.5 / n), (13.5 / n, 13.5 / n, 24.5 / n), (21.5 / n, 5.5 / n, 5.5 / n)],
                                 [8.0, 8.0, 3.0], m=300)
    assert sh["fold"][0] == 0.0 and sh["sheet"][0] == 0.0 and 0.0 < sh["fold"][1] < 0.5 and sh["per_shell"]["fold"].shape == (3, 4), sh["fold"]
    assert sh["sheet"][1] > 0.0 and sh["fold"][2] == 0.0 and sh["sheet"][2] > 0.0, (sh["fold"], sh["sheet"])
    # the per-void value is the MEAN over the 1.0-1.3 R_eff shells (they differ here), each shell Archimedes' hat-box:
    # the planes' cells z = 31, 32 (= 0) are the band dz in [6.5, 8.5] above z = 24.5 -> (min(8.5, r) - 6.5) / 2r
    assert list(sh["shells"]) == [1.0, 1.1, 1.2, 1.3] and (sh["per_shell"]["fold"][1].max() - sh["per_shell"]["fold"][1].min()) > 0.005, sh["per_shell"]["fold"][1]
    for name in ("fold", "sheet"):
        assert np.allclose(sh[name], sh["per_shell"][name].mean(axis=1), rtol=0, atol=1e-12), (name, sh[name], sh["per_shell"][name])
    hat = [(min(8.5, 8.0 * x) - 6.5) / (16.0 * x) for x in (1.0, 1.1, 1.2, 1.3)]
    assert np.abs(sh["per_shell"]["fold"][1] - hat).max() < 0.015, (sh["per_shell"]["fold"][1], hat)
    d, bounded = sk.nearest_wall_distance(sheet, [(13, 13, 16), (13, 13, 31)], cutout_cells=32)
    assert abs(d[0] - np.sqrt(128.0)) < 1e-9 and not bounded[0] and d[1] == 0.0 and not bounded[1], (d, bounded)   # cell (5, 5, 16)
    # the 'd > half' branch: a sheet cell, (5, 5, 16), lies INSIDE the 17^3 cut-out but at sqrt(128) = 11.3 > 8, where
    # a nearer one could sit just outside the cut-out: NaN, the half-width a lower bound
    d8, b8 = sk.nearest_wall_distance(sheet, [(13, 13, 16)], cutout_cells=8)
    assert np.isnan(d8[0]) and b8[0], (d8, b8)
    # the EMPTY cut-out branch: the sheet's z < 10 part has no cell within 4 of (13, 13, 24) (the mask is not empty:
    # (5, 5, 6) of it is found at 0) -> NaN and bounded
    de, be = sk.nearest_wall_distance(sheet & (Z < 10), [(13, 13, 24), (5, 5, 6)], cutout_cells=4)
    assert np.isnan(de[0]) and be[0] and de[1] == 0.0 and not be[1], (de, be)
    # the cut-out WRAPS the box (a clipped one sees nothing across the face): the z = 31 plane is 2 cells below z = 1;
    # at z = 27 it lies ON the cut-out's face, d = half exactly -- exact, since every cell outside the cut-out is further
    # than half (not censored); at z = 26 it is outside. A single cell (0, 0, 0) across the corner from (31, 31, 31) and
    # across the y face only from (1, 30, 1): every axis wraps
    dw, bw = sk.nearest_wall_distance(sheet & (Z == n - 1), [(13, 13, 1), (13, 13, 27), (13, 13, 26)], cutout_cells=4)
    assert list(dw[:2]) == [2.0, 4.0] and np.isnan(dw[2]) and list(bw) == [False, False, True], (dw, bw)
    pt = np.zeros((n, n, n), bool)
    pt[0, 0, 0] = True
    dw, bw = sk.nearest_wall_distance(pt, [(n - 1, n - 1, n - 1), (1, n - 2, 1)], cutout_cells=4)
    assert np.allclose(dw, [np.sqrt(3.0), np.sqrt(6.0)], rtol=0, atol=1e-12) and not bw.any(), (dw, bw)


def t_caustic_fold_exact():
    """The geometric fold flag for any grid: io.caustic_is_fold(mask, streams=,
    hidden=) = streams > 1 + tol | hidden bit 1 | the mask's fold bits, the mask-only form unchanged, the tolerance edge,
    the union with single_stream_mask's complement, the slab path (memmaps, several slabs) equal to one pass, the
    argument checks; FieldSet.fold_mask() on synthetic snapshots: a sample-flagged grid (fold bits only where a sample hit,
    the hidden bit on the thin rest) gives the whole fold, the averaged pair by default, and the fallbacks without a
    streams grid (the class mask, then the 0/1 'caustic' grid) and under method='dtfe'."""
    import shutil, tempfile
    from dtfelib import io as dio
    from dtfelib.io import (CAUSTIC_COLLAPSE, CAUSTIC_PARITY_NEG, CAUSTIC_PARITY_POS, FieldSet, STREAM_TOL,
                            caustic_is_fold, single_stream_mask)
    pair = CAUSTIC_PARITY_POS | CAUSTIC_PARITY_NEG
    # cell by cell: streams, hidden, mask -> fold (tolerance edges in float32, as single_stream_mask computes them)
    st = np.array([1.0, 1.0 + 0.5 * STREAM_TOL, 1.0 + 2 * STREAM_TOL, 3.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.5], np.float32)
    hi = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0], np.float32)
    mk = np.array([CAUSTIC_PARITY_POS, 0, 0, pair, CAUSTIC_PARITY_POS, CAUSTIC_PARITY_POS, pair, CAUSTIC_PARITY_NEG, pair, pair])
    assert list(caustic_is_fold(streams=st, hidden=hi)) == [False, False, True, True, True, False, True, True, False, True]
    assert list(caustic_is_fold(mk, streams=st, hidden=hi)) == [False, False, True, True, True, False, True, True, True, True]
    assert list(caustic_is_fold(mk)) == [False, False, False, True, False, False, True, False, True, True]    # unchanged
    assert list(caustic_is_fold(streams=st)) == [False, False, True, True, False, False, False, False, False, True]
    # every cell that is neither single-stream nor empty is a fold cell, and every fold cell is not single-stream
    rng = np.random.default_rng(3)
    n = 12
    S = rng.choice(np.array([0.0, 1.0, 1.0, 1.0, 1.0004, 1.5, 3.0, 2.0 / 3.0], np.float32), size=(n, n, n))
    H = rng.choice(np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0], np.float32), size=(n, n, n))
    C = rng.choice(np.array([0, CAUSTIC_PARITY_POS | CAUSTIC_COLLAPSE[0], pair | CAUSTIC_COLLAPSE[1]]), size=(n, n, n))
    F = caustic_is_fold(streams=S, hidden=H)
    notsingle = ~single_stream_mask(S, H)
    assert np.array_equal(notsingle, F | (S < 1.0 - STREAM_TOL)) and not (F & single_stream_mask(S, H)).any()
    # the slab path: memmaps, several slabs of an odd size, the same as one pass
    d = Path(tempfile.mkdtemp(prefix="dtfelib_fold_"))
    old_slab = dio._FOLD_SLAB_CELLS
    try:
        for name, arr in (("s", S), ("h", H), ("c", C.astype(np.float32))):
            arr.astype(np.float32).tofile(d / name)
        mm = {k: np.memmap(d / k, dtype=np.float32, mode="r", shape=(n, n, n)) for k in ("s", "h", "c")}
        one = caustic_is_fold(C, streams=S, hidden=H)
        dio._FOLD_SLAB_CELLS = 5 * n * n                              # 3 slabs of 5, 5, 2 planes
        sl = caustic_is_fold(mm["c"], streams=mm["s"], hidden=mm["h"])
        assert sl.dtype == bool and sl.shape == (n, n, n) and np.array_equal(sl, one)
        assert np.array_equal(one, F | caustic_is_fold(C))
        for bad in ((None, None, None), (None, None, H), (C[:-1], S, None), (None, S, H[:, :-1])):
            try:
                caustic_is_fold(bad[0], streams=bad[1], hidden=bad[2])
            except ValueError:
                pass
            else:
                raise AssertionError(f"caustic_is_fold accepted {[None if b is None else np.shape(b) for b in bad]}")
    finally:
        dio._FOLD_SLAB_CELLS = old_slab
        shutil.rmtree(d, ignore_errors=True)
    # FieldSet.fold_mask on a SAMPLE-FLAGGED grid: a wall plane x = 3 that the samples saw (streams 3, fold bits) and a thin
    # one x = 6 they missed (streams 1, hidden bit 1, the mask's det<0 bit absent): the fold is both planes
    root = Path(tempfile.mkdtemp(prefix="dtfelib_foldfs_"))
    try:
        n = 8
        X = np.indices((n, n, n))[0]
        d50 = _synthetic_snapdir(root, "SYN", 50, 0.5, n=n, grids=("den",))
        streams = np.where(X == 3, 3.0, 1.0).astype(np.float32)
        hidden = np.where(X == 6, 1.0, 0.0).astype(np.float32)
        cc = np.full((n, n, n), CAUSTIC_PARITY_POS | CAUSTIC_COLLAPSE[0], np.float32)
        cc[X == 3] = pair | CAUSTIC_COLLAPSE[1]
        old_flag = (X == 3).astype(np.float32)
        np.ones((n, n, n), np.float32).tofile(str(d50 / "ps_output.den"))     # the raw set's grid size (averaged=False)
        for pre in ("ps_output.a_", "ps_output."):                    # the averaged pair and the raw pair
            streams.tofile(str(d50 / f"{pre}streams"))
            hidden.tofile(str(d50 / f"{pre}hidden_streams"))
        cc.tofile(str(d50 / "ps_output.causticClass"))
        old_flag.tofile(str(d50 / "ps_output.caustic"))
        fs = FieldSet(d50, method="ps")
        want = (X == 3) | (X == 6)
        assert np.array_equal(fs.fold_mask(), want) and np.array_equal(FieldSet(d50, method="ps", averaged=False).fold_mask(), want)
        (d50 / "ps_output.a_hidden_streams").unlink()                 # no hidden grid: streams > 1 and the mask's bits only
        assert np.array_equal(fs.fold_mask(), X == 3)
        (d50 / "ps_output.a_streams").unlink()                        # no streams grid: the class mask's parity bits
        assert np.array_equal(fs.fold_mask(), X == 3)
        (d50 / "ps_output.causticClass").unlink()                     # nor a class mask: the 0/1 'caustic' grid
        assert np.array_equal(fs.fold_mask(), X == 3)
        (d50 / "ps_output.caustic").unlink()
        try:
            fs.fold_mask()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("fold_mask() with no streams, caustic_class or caustic grid did not raise")
        np.ones((n, n, n), np.float32).tofile(str(d50 / "output.a_den"))
        try:
            FieldSet(d50, method="dtfe").fold_mask()
        except ValueError:
            pass
        else:
            raise AssertionError("fold_mask() under method='dtfe' did not raise")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def t_caustic_crossed_waves():
    """The crossed-waves runs tests/ps_caustic_class_check.sh leaves under the suites' temp root ($DTFE_TEST_TMP,
    then ${TMPDIR:-/tmp}/dtfe-tests): on pcc_out (CPU, no --ps-caustic-cusps) the nested
    cumulative masks, fold within sheet, more streams with every collapsed axis, a JSON-able summary, no cusp bits;
    on pcc_a4 (--ps-caustic-cusps) the cusp and swallowtail fractions are their own bits' and differ; pcc_nocusp
    reads 'not computed'; the parity-only pcc_out_gpulin is refused. No pcc_out: FileNotFoundError -- a SKIP in the
    data-backed tier, never a PASS that did not run; an absent sub-fixture is noted and skipped alone."""
    import json, os
    from dtfelib import skeleton as sk
    from dtfelib.io import CAUSTIC_CUSP, CAUSTIC_SWALLOWTAIL
    roots = [Path(os.environ["DTFE_TEST_TMP"])] if os.environ.get("DTFE_TEST_TMP") else []
    roots += [Path(os.environ.get("TMPDIR") or "/tmp") / "dtfe-tests"]

    def find(name, suffixes):
        return next((r / name for r in roots if all((r / (name + s)).is_file() for s in suffixes)), None)

    def grid(fx, suffix):
        # float32 by default, float64 under tests/precision.sh (DTFE_TEST_REAL): the cube tells them apart
        raw = Path(str(fx) + suffix).read_bytes()
        for dt in (np.float32, np.float64):
            a = np.frombuffer(raw, dtype=dt) if len(raw) % np.dtype(dt).itemsize == 0 else np.zeros(0)
            g = int(round(a.size ** (1 / 3))) if a.size else 0
            if g and g ** 3 == a.size:
                return a.reshape(g, g, g)
        raise AssertionError(f"{fx}{suffix}: {len(raw)} bytes is no cube of float32 or float64")
    fx = find("pcc_out", (".causticClass", ".den", ".streams", ".hidden_streams", ".caustic"))
    if fx is None:
        raise FileNotFoundError(f"no crossed-waves run pcc_out.{{causticClass,den,streams}} under {', '.join(map(str, roots))} "
                                f"(tests/ps_caustic_class_check.sh writes it)")
    try:                                    # a missing summary key is a defect here, not missing data (KeyError would SKIP)
        rcc = np.rint(grid(fx, ".causticClass")).astype(np.int32)
        rden, rstr, rhid = grid(fx, ".den"), grid(fx, ".streams"), grid(fx, ".hidden_streams")
        rm = sk.class_masks(rcc)
        assert (rm["cumulative"]["node"] <= rm["cumulative"]["line"]).all() and (rm["cumulative"]["line"] <= rm["cumulative"]["sheet"]).all()
        # a fold needs a collapsed axis -- but the collapse bits are per SAMPLE and the fold is geometric:
        # a fold cell with no k >= 1 bit is one whose samples all missed the folded tetrahedra, i.e. a hidden-bit cell
        missed = rm["indicator"]["fold"] & ~rm["cumulative"]["sheet"]
        assert (missed <= ((np.rint(rhid).astype(np.int32) & 1) != 0)).all(), int(missed.sum())
        # the binary's '.caustic' IS dtfelib's fold, from the class mask alone or from the stream grids
        from dtfelib.io import caustic_is_fold
        rfold = grid(fx, ".caustic") == 1
        assert np.array_equal(caustic_is_fold(rcc), rfold) and np.array_equal(caustic_is_fold(streams=rstr, hidden=rhid), rfold)
        thin = find("pcj_crw_s", (".causticClass", ".streams", ".hidden_streams"))
        if thin is not None:                # the young crossed waves of check (J): thin folds, k = 0 fold cells exist
            tcc = np.rint(grid(thin, ".causticClass")).astype(np.int32)
            tm, thid = sk.class_masks(tcc), (np.rint(grid(thin, ".hidden_streams")).astype(np.int32) & 1) != 0
            tmiss = tm["indicator"]["fold"] & ~tm["cumulative"]["sheet"]
            assert tmiss.sum() > 0 and (tmiss <= thid).all(), (int(tmiss.sum()), int((tmiss & ~thid).sum()))
            assert np.array_equal(caustic_is_fold(streams=grid(thin, ".streams"), hidden=grid(thin, ".hidden_streams")),
                                  tm["indicator"]["fold"])
        else:
            print("    (no pcj_crw_s run: the thin-fold part skipped)")
        rfr = sk.fractions(rcc, rden)
        assert abs(sum(v["volume"] for v in rfr["exclusive"].values()) - 1.0) < 1e-12 and rfr["cumulative"]["sheet"]["volume"] > 0
        means = [rstr[rm["exclusive"][c]].mean() for c in ("k=0", "k=1", "k=2") if rm["exclusive"][c].any()]
        assert len(means) >= 2 and all(a < b for a, b in zip(means, means[1:])), means  # more streams with every collapsed axis
        rs = sk.skeleton_summary(rcc, rden, cell_mpc=100.0 / rcc.shape[0])
        assert rs["components"]["sheet"]["n"] >= 1 and json.dumps(rs)
        assert rs["cusp_bits_present"] is False and rfr["indicator"]["cusp"]["volume"] == 0.0   # a CPU run without --ps-caustic-cusps
        a4 = find("pcc_a4", (".causticClass",))
        if a4 is not None:                  # --ps-caustic-cusps: each indicator is ITS bit (a swap keeps both > 0 and different)
            acc = np.rint(grid(a4, ".causticClass")).astype(np.int32)
            afr = sk.fractions(acc)["indicator"]
            cusp, swt = ((acc & CAUSTIC_CUSP) != 0).mean(), ((acc & CAUSTIC_SWALLOWTAIL) != 0).mean()
            assert afr["cusp"]["volume"] > 0 and afr["swallowtail"]["volume"] > 0 and afr["cusp"]["volume"] != afr["swallowtail"]["volume"], afr
            assert abs(afr["cusp"]["volume"] - cusp) < 1e-15 and abs(afr["swallowtail"]["volume"] - swt) < 1e-15, (afr, cusp, swt)
            assert sk.skeleton_summary(acc)["cusp_bits_present"] is True
        else:
            print("    (no pcc_a4 run: the cusp / swallowtail part skipped)")
        nc = find("pcc_nocusp", (".causticClass",))
        if nc is not None:
            assert sk.skeleton_summary(np.rint(grid(nc, ".causticClass")).astype(np.int32))["cusp_bits_present"] is False
        else:
            print("    (no pcc_nocusp run: its 'not computed' check skipped)")
        lin = find("pcc_out_gpulin", (".causticClass",))
        if lin is not None:                 # --ps-caustics --ps-linear-deposit --ps-gpu writes the fold flag alone
            try:
                sk.class_masks(np.rint(grid(lin, ".causticClass")).astype(np.int32))
            except ValueError as e:
                assert "parity-only" in str(e), e
            else:
                raise AssertionError("the parity-only pcc_out_gpulin grid was read as a stratification")
        else:
            print("    (no pcc_out_gpulin run (GPU only): the parity-only refusal on a real grid skipped)")
    except KeyError as e:
        raise AssertionError(f"KeyError {e}")


def t_caustic_skeleton_script():
    """plot_caustic_skeleton.py: summary_text prints '--' and why for the cusp / swallowtail rows of a grid without bit
    7/8 (only those rows), percentages when it has them, mass apart from volume; void_block centres a void on its cell
    centre, its medians are None (no NaN, no RuntimeWarning) with no well-resolved void and with every void beyond the
    cut-out. Then main() in process on synthetic snapshots (its outputs and the caches pointed at
    temp dirs): .json/.txt/.png per redshift, a strict JSON, --connectivity passed on, nothing in the default folder
    under --out; a
    snapshot without a grid, without a snapdir or with a parity-only grid exits 1; a failed --voids (the frame
    refusal) exits 1 with 'voids_error' in the JSON; a delivered one writes the 'voids' block and npz
    without a RuntimeWarning. Names: a non-default option tags the set ('_c6', '_reff', '_shells...',
    '_overlaps', '_cut<N>'), a plain run removes the set's earlier _voids.npz, and the void panel's title says
    'distinct' when the run applied it, not by the radius."""
    import io as _io, json, shutil, sys as _sys, tempfile, types, warnings
    from contextlib import redirect_stdout
    import matplotlib
    import config
    from dtfelib import skeleton as sk
    from dtfelib.io import CAUSTIC_PARITY_POS, CAUSTIC_PARITY_NEG, CAUSTIC_COLLAPSE, CAUSTIC_CUSP, CAUSTIC_DEGENERATE
    root = Path(tempfile.mkdtemp(prefix="dtfe_skelrun_"))
    plot_dir = str(Path(__file__).resolve().parent.parent / "python" / "plot")
    old_cfg = (config.LOCAL_FIGURES_ROOT, config.CACHE_DIR)
    old_argv, old_path = list(_sys.argv), list(_sys.path)
    pcs = saved = None
    try:
        if plot_dir not in _sys.path:
            _sys.path.insert(0, plot_dir)
        import plot_caustic_skeleton as pcs                 # BEFORE config is patched: OUTPUT_DIR is set at import
        saved = (pcs.OUTPUT_DIR, pcs.pipeline, pcs.style)
        config.LOCAL_FIGURES_ROOT, config.CACHE_DIR = str(root / "local"), root / "cache"
        pcs.OUTPUT_DIR = root / "default_out"               # never the repo's figures/ (a run without --out lands here)
        with matplotlib.rc_context():                       # main() calls style.apply(): rcParams back afterwards
            n = 8
            X, Y, Z = np.indices((n, n, n))
            fold = CAUSTIC_PARITY_POS | CAUSTIC_PARITY_NEG
            cc = np.full((n, n, n), CAUSTIC_COLLAPSE[0] | CAUSTIC_PARITY_POS, np.int32)
            cc[X == 2] |= CAUSTIC_COLLAPSE[1] | fold                                  # a wall at x = 2 ...
            cc[(X == 2) & (Y == 4)] |= CAUSTIC_COLLAPSE[2]                            # ... with a filament in it
            # summary_text: no bit 7/8 -> '--' cusp / swallowtail rows and the caveat, not 0.000%, the umbilic
            # row still a number; with a cusp bit, percentages. The wall at density 2 (576 in all) parts mass from volume
            ccu = cc.copy()
            ccu[2, 1, 1] |= CAUSTIC_DEGENERATE                                       # one umbilic cell, in the wall
            den2 = np.where(X == 2, 2.0, 1.0)
            txt = pcs.summary_text(sk.skeleton_summary(ccu, den2, cell_mpc=12.5), "t")
            rows = {ln.split()[0]: ln.split()[1:] for ln in txt.splitlines() if ln.strip()}
            assert rows["cusp"] == ["--", "--"] and rows["swallowtail"] == ["--", "--"] and "no bit 7/8 in this grid" in txt, txt
            assert rows["umbilic"] == ["0.195%", "0.347%"], txt                      # 1 / 512, 2 / 576
            assert rows["fold"] == ["12.500%", "22.222%"] and rows["k=1"] == ["10.938%", "19.444%"], txt   # 64 / 512, 128 / 576; 56, 112
            cc2 = ccu.copy()
            cc2[5, 5, 5] |= CAUSTIC_CUSP
            txt2 = pcs.summary_text(sk.skeleton_summary(cc2, den2, cell_mpc=12.5), "t")
            rows2 = {ln.split()[0]: ln.split()[1:] for ln in txt2.splitlines() if ln.strip()}
            assert rows2["cusp"] == ["0.195%", "0.174%"] and rows2["swallowtail"] == ["0.000%", "0.000%"] and "no bit 7/8" not in txt2, txt2
            assert rows2["umbilic"] == ["0.195%", "0.347%"], txt2
            # void_block: well_resolved picks the rows; no resolved void, every void beyond the cut-out -> None medians,
            # strict JSON, no RuntimeWarning; a normal set -> the medians of the finite values
            cell_c = 12.5
            cat = {"well_resolved": np.array([True, False, True]), "coords": np.array([[4, 4, 4], [6, 6, 6], [2, 4, 4]]),
                   "semi_axes": np.full((3, 3), 2.0 * cell_c, np.float32)}                # R_eff = 2 cells
            far = {"well_resolved": np.array([True, True]), "coords": np.array([[6, 1, 1], [6, 5, 6]]),
                   "semi_axes": np.full((2, 3), 2.0 * cell_c, np.float32)}                # 4 cells from the wall
            none_ = dict(cat, well_resolved=np.zeros(3, bool))
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                v, s = pcs.void_block(cc, cat, cell_c, n, 3)
                assert list(v["index"]) == [0, 2] and len(v["fold"]) == 2 and list(v["dist_cells"]) == [2.0, 0.0], v
                assert s["n"] == 2 and s["beyond_cutout"] == 0 and s["fold_shell_median"] == float(np.median(v["fold"])) > 0, s
                assert abs(s["dist_reff_median"] - 0.5) < 1e-6, s                     # 2 and 0 cells over R_eff = 2 (float32 axes)
                assert np.allclose(v["dist_reff"], [1.0, 0.0], rtol=0, atol=1e-6) and np.allclose(v["r_eff_cells"], 2.0, rtol=0, atol=1e-6), v
                # the void sits on its CELL CENTRE (4.5, not 4): the wall's cell x in [2, 3) is the band dx in [-2.5, -1.5] of
                # shells r = 2 .. 2.6, Archimedes' hat-box (min(2.5, r) - 1.5) / 2r -> 0.166 (from the corner: 0.215); on the
                # wall, dx in [-0.5, 0.5] -> 1 / 2r, 0.219
                for row, band in ((0, (1.5, 2.5)), (1, (-0.5, 0.5))):
                    hat = np.mean([(min(band[1], 2 * x) - max(band[0], -2 * x)) / (4 * x) for x in (1.0, 1.1, 1.2, 1.3)])
                    assert abs(v["fold"][row] - hat) < 0.02 and v["sheet"][row] == v["fold"][row], (row, v["fold"], hat)
                v, s = pcs.void_block(cc, none_, cell_c, n, 3)
                assert s == {"n": 0, "fold_shell_median": None, "dist_reff_median": None, "beyond_cutout": 0} and len(v["index"]) == 0, s
                json.dumps(s, allow_nan=False)
                v, s = pcs.void_block(cc, far, cell_c, n, 2)                            # x 4..8: the wall at 2 is outside
                assert s == {"n": 2, "fold_shell_median": 0.0, "dist_reff_median": None, "beyond_cutout": 2} and v["bounded"].all(), s
                json.dumps(s, allow_nan=False)
                assert pcs._median_or_none([np.nan, 1.0, 3.0, np.inf]) == 2.0 and pcs._median_or_none([]) is None
            # main(), in process: a snapshot with the grid (z = 0.5), one with no caustic_class (99), one parity-only (98),
            # none at all (97)
            data = root / "data"
            d50 = _synthetic_snapdir(data, "SYN", 50, 0.5, n=n, grids=("den",))
            ccr = cc.copy()                                    # two node cells, edge neighbours: 1 node under 26, 2 under 6
            ccr[2, 4, 1] |= CAUSTIC_COLLAPSE[3]
            ccr[2, 5, 2] |= CAUSTIC_COLLAPSE[3]
            ccr.astype(np.float32).tofile(str(d50 / "ps_output.causticClass"))     # written once: no 'a_' (io._NO_AVG_VARIANT)
            _synthetic_snapdir(data, "SYN", 99, 0.0, n=n, grids=("den",))
            d98 = _synthetic_snapdir(data, "SYN", 98, 0.01, n=n, grids=("den",))
            np.full((n, n, n), fold, np.float32).tofile(str(d98 / "ps_output.causticClass"))

            def run(args, fresh=True):
                out = root / "out"
                if fresh:
                    shutil.rmtree(out, ignore_errors=True)
                buf = _io.StringIO()
                _sys.argv = ["plot_caustic_skeleton.py", "--sim", "SYN", "--data-root", str(data), "--out", str(out)] + args
                with redirect_stdout(buf):
                    try:
                        pcs.main(); code = 0
                    except SystemExit as e:
                        code = int(e.code or 0)
                return code, buf.getvalue(), sorted(q.name for q in out.iterdir()) if out.is_dir() else []

            def strict(path):                   # the JSON standard: a bare NaN / Infinity is an error, not a number
                def no_constant(c):
                    raise ValueError(f"bare {c} in {path.name}")
                return json.loads(path.read_text(), parse_constant=no_constant)
            def files(tag=""):                  # a set's three files; a non-default option tags them
                return [f"caustic_skeleton_z0.50{tag}.{e}" for e in ("json", "png", "txt")]
            base = files()
            code, text, names = run(["--snaps", "50"])
            assert code == 0 and names == base, (code, names, text[-600:])
            j = strict(root / "out" / "caustic_skeleton_z0.50.json")
            assert j["cusp_bits_present"] is False and j["snapshot"] == 50 and j["redshift"] == 0.5 and j["prefix"] == "ps_output.", j
            assert j["sim"] == "SYN" and j["connectivity"] == 26 and j["components"]["node"]["n"] == 1, j
            assert abs(sum(e["volume"] for e in j["fractions"]["exclusive"].values()) - 1.0) < 1e-12 and j["components"]["sheet"]["n"] == 1
            assert "voids" not in j and "voids_error" not in j
            t = (root / "out" / "caustic_skeleton_z0.50.txt").read_text()
            assert "no bit 7/8 in this grid" in t and any(ln.split()[:3] == ["cusp", "--", "--"] for ln in t.splitlines()), t
            assert not (root / "default_out").exists()                                         # --out: its folder only
            code, text, names = run(["--snaps", "50", "--connectivity", "6"])                   # reaches the components
            assert names == files("_c6"), names                                                 # its own files: '_c6'
            j = strict(root / "out" / "caustic_skeleton_z0.50_c6.json")
            assert code == 0 and j["connectivity"] == 6 and j["components"]["node"]["n"] == 2 and "connectivity 6" in text, (code, j["components"])
            code, text, names = run(["--snaps", "50", "99", "98", "97"])
            assert code == 1 and names == base and "failed: [99, 98, 97]" in text, (code, names, text[-600:])
            assert "no caustic_class grid" in text and "parity-only" in text, text[-600:]
            # --voids asked for and refused (the header's 100 Mpc / 8 cell is not config.CELL_SIZE: the frame guard)
            assert abs(100.0 / n - config.CELL_SIZE) > 1e-3 * config.CELL_SIZE, config.CELL_SIZE
            code, text, names = run(["--snaps", "50", "--voids", "--radius", "r_eff", "--keep-overlaps"])
            assert code == 1 and names == files("_reff_overlaps") and "failed: [50]" in text and "voids: FAILED" in text, (code, names, text[-600:])
            j = strict(root / "out" / "caustic_skeleton_z0.50_reff_overlaps.json")
            assert "cell" in j.get("voids_error", "") and "voids" not in j, j.get("voids_error")
            assert not list((root / "cache").rglob("*.npz"))                     # no catalogue written (into the temp cache)

            # --voids delivered: a catalogue in config's frame (a stub of pipeline.products for this script only)
            class _FS:
                grid_n = n
                meta = types.SimpleNamespace(box_mpc=config.CELL_SIZE * n, redshift=0.5)

            class _Prod:
                sim, fs = "SYN", _FS()
                def voids(self):                # + the frame, depths and an R_v (row 0) a built catalogue carries: the r_v runs
                    return dict(far, semi_axes=np.full((2, 3), 2.0 * config.CELL_SIZE, np.float32),
                                well_resolved=np.array([True, False]), cell_mpc=config.CELL_SIZE, r_v_cells=np.array([2.0, np.nan]),
                                delta_values=np.array([-0.6, -0.4], np.float32))
                def release(self):
                    pass
            real = saved[1]
            pcs.pipeline = types.SimpleNamespace(products=lambda *a, **kw: _Prod(), add_smoothing_mpc_arg=real.add_smoothing_mpc_arg,
                                                 apply_smoothing_args=real.apply_smoothing_args,
                                                 smoothing_length_mpc=real.smoothing_length_mpc, smoothing_tag=real.smoothing_tag)

            class _TitleSpy:                    # dtfelib.figures as the script sees it, every set_title text recorded
                def __init__(self, real_style):
                    self.real, self.titles = real_style, []
                def __getattr__(self, k):
                    return getattr(self.real, k)
                def set_title(self, ax, text, *a, **k):
                    self.titles.append(text)
                    return self.real.set_title(ax, text, *a, **k)
            pcs.style = spy = _TitleSpy(saved[2])
            with warnings.catch_warnings():             # undefined medians: None, not an all-NaN / empty (nan)median;
                # only numpy's two messages: matplotlib's own autoscaling on the all-NaN column is not under test
                warnings.filterwarnings("error", category=RuntimeWarning, message=r"(All-NaN slice|Mean of empty slice)")
                code, text, names = run(["--snaps", "50", "--voids", "--cutout", "2", "--radius", "r_eff", "--keep-overlaps"])
            tag = "_reff_overlaps_cut2"
            assert code == 0 and names == sorted(files(tag) + [f"caustic_skeleton_z0.50{tag}_voids.npz"]), (code, names, text[-600:])
            j = strict(root / "out" / f"caustic_skeleton_z0.50{tag}.json")
            assert j["voids"] == {"n": 1, "fold_shell_median": 0.0, "dist_reff_median": None, "beyond_cutout": 1, "radius": "r_eff",
                                  "shells": [1.0, 1.1, 1.2, 1.3], "distinct": False, "sigma_mpc": real.smoothing_length_mpc(),
                                  "footprint_cells": config.FOOTPRINT_SIZE} and "voids_error" not in j, j
            with np.load(root / "out" / f"caustic_skeleton_z0.50{tag}_voids.npz") as zv:
                assert sorted(zv.files) == ["bounded", "dist_cells", "dist_reff", "fold", "index", "r_eff_cells", "radius", "sheet",
                                            "shells"], zv.files
                assert list(zv["index"]) == [0] and bool(zv["bounded"][0]) and np.isnan(zv["dist_cells"][0]) and np.isnan(zv["dist_reff"][0])
            assert not (root / "default_out").exists()
            # the names. The default --voids set is untagged; a plain run into the same folder rewrites the three
            # files and removes that set's _voids.npz (an earlier run's, not this one's); other shells are '_shells...'
            code, text, names = run(["--snaps", "50", "--voids"])
            assert code == 0 and names == sorted(base + ["caustic_skeleton_z0.50_voids.npz"]), (code, names, text[-600:])
            assert strict(root / "out" / "caustic_skeleton_z0.50.json")["voids"]["n"] == 1         # row 0 has an R_v of 2 cells
            code, text, names = run(["--snaps", "50"], fresh=False)
            assert code == 0 and names == base, (code, names, text[-400:])
            code, text, names = run(["--snaps", "50", "--voids", "--shells", "0.8,1.0,1.2"])
            assert code == 0 and names == sorted(files("_shells0.8-1-1.2") + ["caustic_skeleton_z0.50_shells0.8-1-1.2_voids.npz"]), names
            # the void panel's title says 'distinct' when the run applied it (summ['voids']['distinct']), whatever the
            # radius (following the radius would call r_v 'distinct' even under --keep-overlaps, a distinct r_eff run
            # not)
            for args, distinct in ((["--radius", "r_eff"], True), (["--keep-overlaps"], False)):
                del spy.titles[:]
                code, text, names = run(["--snaps", "50", "--voids"] + args)
                said = [t for t in spy.titles if "well-resolved voids" in t]
                assert code == 0 and len(said) == 1 and ("distinct" in said[0]) == distinct, (args, said, text[-300:])
    finally:
        if saved is not None:
            pcs.OUTPUT_DIR, pcs.pipeline, pcs.style = saved
        config.LOCAL_FIGURES_ROOT, config.CACHE_DIR = old_cfg
        _sys.argv = old_argv
        _sys.path[:] = old_path
        shutil.rmtree(root, ignore_errors=True)


def _synthetic_snapdir(root, sim, snap, z, n=8, grids=("tweb", "velVweb", "streams", "hidden_streams", "den"), prefix="ps_output",
                       extra=()):
    """A flat-layout snapshot folder with a minimal combined_NNN.hdf5 header (box 100 Mpc, redshift z) and
    constant-ish n^3 float32 grids: the T-web by plane (0..3), the V-web the reverse, streams 1 except a
    multi-stream plane, no hidden bits; 'extra' = (suffix, ncomp) pairs of smooth positive random grids
    (vel, velDiv, velShear, velDisp, velDispTensor, twebEig, velVwebEig); returns the folder."""
    import h5py
    d = Path(root) / sim / f"snapdir_{snap:03d}"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / f"combined_{snap:03d}.hdf5", "w") as f:
        h = f.create_group("Header").attrs
        h["BoxSize"], h["Redshift"], h["HubbleParam"], h["HFreeUnits"] = 100000.0, float(z), 0.6774, 1
        h["NumPart_ThisFile"], h["MassTable"] = np.array([0, n ** 3, 0, 0, 0, 0]), np.array([0, 1.0, 0, 0, 0, 0])
    idx = np.indices((n, n, n))[0]
    data = {"tweb": (idx * 4 // n).astype(np.float32), "velVweb": (3 - idx * 4 // n).astype(np.float32),
            "streams": np.where(idx == n - 1, 3.0, 1.0).astype(np.float32), "hidden_streams": np.zeros((n, n, n), np.float32),
            "den": np.ones((n, n, n), np.float32)}
    for g in grids:
        data[g].tofile(str(d / f"{prefix}.a_{g}"))
    rng = np.random.default_rng(11)
    for suffix, ncomp in extra:
        arr = rng.uniform(0.1, 2.0, size=(n, n, n, ncomp) if ncomp > 1 else (n, n, n)).astype(np.float32)
        if suffix in ("velDiv", "twebEig", "velVwebEig"):
            arr -= 1.0                                      # signed fields
        arr.tofile(str(d / f"{prefix}.a_{suffix}"))
    return d


def t_webstreams_script():
    """plot_web_streams.py end to end (in-process, its outputs pointed at temp dirs):
    two redshifts give two sets of files (z0.50 and z0.00); the tables sum to 1; a truncated V-web grid at one
    snapshot keeps that snapshot's T-web table, is named in the summary and exits 1; a web nobody has exits 1
    only when asked for by name; an unreadable combined file between two good snapshots (h5py's OSError must not
    abort the series) is a failed snapshot, the loop goes on to the next one and the series plot. Names:
    a series over part of the snapdirs on disk is '_snaps<a>-<b>-...' (the whole ladder untagged, a
    failed snapshot still counted as asked), another --prefix '_<prefix>', --raw '_raw', the default files untouched.
    The V-web here is the STORED PS grid ('--ps-gradient stored', whose truncation and absence the test
    stages), so its files carry the '_stored' tag; the default mean-field V-web is tests/py_meanfield_test.py's."""
    import io as _io, shutil, sys as _sys, tempfile
    from contextlib import redirect_stdout
    import json
    import matplotlib as mpl
    import config
    root = Path(tempfile.mkdtemp(prefix="dtfe_wsrun_"))
    old_figs, old_argv = config.LOCAL_FIGURES_ROOT, list(_sys.argv)
    plot_dir = str(Path(__file__).resolve().parent.parent / "python" / "plot")
    _sys.path.insert(0, plot_dir)
    pws = old_pws = None
    try:
        with mpl.rc_context():                      # main() applies the house style: it must not outlive the test
            import plot_web_streams as pws          # BEFORE config is patched: the import-time paths restored below are the real ones
            old_pws = pws.OUTPUT_DIR
            config.LOCAL_FIGURES_ROOT = str(root / "figures")
            pws.OUTPUT_DIR = root / "figures" / "web_streams"   # a run that ignored --out lands here, not in the repo
            data = root / "data"
            _synthetic_snapdir(data, "SYN", 50, 0.5)
            _synthetic_snapdir(data, "SYN", 99, 0.0)
            _synthetic_snapdir(data, "SYN", 98, 0.01)
            (data / "SYN" / "snapdir_098" / "ps_output.a_velVweb").write_bytes(b"\0" * 100)   # truncated
            _synthetic_snapdir(data, "SYN", 97, 0.25)
            (data / "SYN" / "snapdir_097" / "combined_097.hdf5").write_bytes(b"not an hdf5 file" * 16)   # grids fine, header unreadable

            def run(args):
                out = root / "out"
                buf = _io.StringIO()
                _sys.argv = ["plot_web_streams.py", "--sim", "SYN", "--data-root", str(data), "--out", str(out),
                             "--ps-gradient", "stored"] + args
                with redirect_stdout(buf):
                    try:
                        pws.main()
                        code = 0
                    except SystemExit as e:
                        code = int(e.code or 0)
                    except Exception as e:                  # an escaped error is the script's crash: say which
                        code = f"{type(e).__name__}: {e}"
                return code, buf.getvalue(), out
            code, text, out = run(["--snaps", "50", "99"])
            names = sorted(q.name for q in out.iterdir())
            assert code == 0, text[-600:]
            for z in ("0.00", "0.50"):
                for w in ("tweb", "vweb_stored"):
                    assert f"web_streams_{w}_z{z}.json" in names and f"web_streams_{w}_z{z}.png" in names and f"web_streams_{w}_z{z}.txt" in names, names
            assert json.loads((out / "web_streams_vweb_stored_z0.50.json").read_text())["vweb_source"] == "stored"
            # the series over 2 of the 4 snapdirs on disk is named by them: it does not replace the full series
            assert "single_stream_vs_z_tweb_snaps50-99.png" in names and "single_stream_vs_z_vweb_stored_snaps50-99.png" in names \
                and "4 table(s) written" in text and not any(q.startswith("single_stream_vs_z_tweb.") for q in names), (names, text[-300:])
            t = json.loads((out / "web_streams_tweb_z0.50.json").read_text())
            assert abs(sum(e["volume_fraction"] for e in t["classes"].values()) - 1.0) < 1e-12 and t["redshift"] == 0.5 and t["web"] == "T-web"
            assert abs(t["classes"]["node"]["stream_bins"]["1 < s <= 3"]["volume"] - 0.5) < 1e-12      # the multi-stream plane is half of class 3
            assert not (Path(pws.OUTPUT_DIR) / "SYN").exists(), \
                ("--out wrote into the default folder", sorted(str(q.relative_to(root)) for q in root.rglob("*.png") if out not in q.parents))
            shutil.rmtree(out)
            code, text, out = run(["--snaps", "99", "98"])
            names = sorted(q.name for q in out.iterdir())
            assert code == 1 and "web_streams_tweb_z0.01.json" in names and "web_streams_vweb_stored_z0.01.json" not in names, (code, names)
            assert "V-web failed: " in text and "for 098" in text and "3 table(s) written" in text, text[-400:]
            shutil.rmtree(out)
            # the garbage header BETWEEN two good snapshots: 97 failed, 99 and 50 written, the series still drawn
            code, text, out = run(["--snaps", "99", "97", "50"])
            names = sorted(q.name for q in out.iterdir()) if out.is_dir() else []
            assert code == 1, (code, text[-600:])
            for z in ("0.00", "0.50"):
                for w in ("tweb", "vweb_stored"):
                    assert f"web_streams_{w}_z{z}.json" in names, (w, z, names)
            assert not any("z0.25" in q for q in names), names                  # nothing for the unreadable one
            # named by the snapshots ASKED for, sorted (the failed 97 too)
            assert "single_stream_vs_z_tweb_snaps50-97-99.png" in names and "single_stream_vs_z_vweb_stored_snaps50-97-99.png" in names, names
            assert "4 table(s) written" in text and "no table for snapshot(s) [97]" in text, text[-400:]
            l097 = [l for l in text.splitlines() if l.startswith("  snapshot 097: ")]
            assert len(l097) == 1 and "signature" in l097[0], text[-600:]           # h5py's own reason, on 097's line
            shutil.rmtree(out)
            # the whole ladder on disk asked for (97 and 98's V-web failing): the untagged series
            code, text, out = run(["--snaps", "99", "98", "97", "50"])
            names = sorted(q.name for q in out.iterdir())
            assert code == 1 and "single_stream_vs_z_tweb.png" in names and "single_stream_vs_z_vweb_stored.png" in names \
                and not any("_snaps" in q for q in names), (code, names)
            # another grid set at 99 (--prefix ps_caustic): its own names; the default set's z = 0 tables untouched
            _synthetic_snapdir(data, "SYN", 99, 0.0, prefix="ps_caustic")
            kept = {q.name: q.stat().st_mtime_ns for q in out.glob("web_streams_tweb_z0.00.*")}
            assert len(kept) == 3, kept
            code, text, out = run(["--snaps", "99", "--prefix", "ps_caustic", "--web", "tweb"])
            names = sorted(q.name for q in out.iterdir())
            assert code == 0 and all(f"web_streams_tweb_ps_caustic_z0.00.{e}" in names for e in ("json", "png", "txt")), (code, names, text[-300:])
            assert {q.name: q.stat().st_mtime_ns for q in out.glob("web_streams_tweb_z0.00.*")} == kept
            assert json.loads((out / "web_streams_tweb_ps_caustic_z0.00.json").read_text())["prefix"] == "ps_caustic."
            # the raw (unaveraged) grids: '_raw'
            d99 = data / "SYN" / "snapdir_099"
            for g in ("tweb", "streams", "hidden_streams", "den"):
                shutil.copy(d99 / f"ps_output.a_{g}", d99 / f"ps_output.{g}")
            code, text, out = run(["--snaps", "99", "--raw", "--web", "tweb"])
            names = sorted(q.name for q in out.iterdir())
            assert code == 0 and all(f"web_streams_tweb_raw_z0.00.{e}" in names for e in ("json", "png", "txt")), (code, names, text[-300:])
            assert {q.name: q.stat().st_mtime_ns for q in out.glob("web_streams_tweb_z0.00.*")} == kept
            shutil.rmtree(out)
            code, text, out = run(["--snaps", "99", "--web", "vweb"])
            assert code == 0 and sorted(q.name for q in out.iterdir()) == ["web_streams_vweb_stored_z0.00.json", "web_streams_vweb_stored_z0.00.png", "web_streams_vweb_stored_z0.00.txt"]
            (data / "SYN" / "snapdir_099" / "ps_output.a_velVweb").unlink()
            shutil.rmtree(out)
            code, text, out = run(["--snaps", "99", "--web", "vweb"])
            assert code == 1 and "no table for snapshot(s) [99]" in text, (code, text[-300:])
            code, text, out = run(["--snaps", "99"])                                   # under 'both' a missing web is a skip
            assert code == 0 and "V-web no grid for 099" in text, (code, text[-300:])
    finally:
        if old_pws is not None:
            pws.OUTPUT_DIR = old_pws
        config.LOCAL_FIGURES_ROOT = old_figs
        _sys.argv = old_argv
        if plot_dir in _sys.path:
            _sys.path.remove(plot_dir)
        shutil.rmtree(root, ignore_errors=True)


def t_compute_missing_data():
    """analyze.py compute on a series with a missing snapdir and an unreadable snapshot file: both counted
    failed, the loop goes on, nothing recorded, exit 1. Each
    snapshot's OWN line is read (the shared prefix says 'unreadable' for both): 050's names the
    missing folder, 099's carries h5py's 'file signature not found' -- the garbage file WAS opened, not
    skipped as absent and not deferred to warm()'s 'FAILED:'."""
    import io as _io, shutil, tempfile
    from contextlib import redirect_stdout
    import config
    from dtfelib import pipeline
    import analyze
    root = Path(tempfile.mkdtemp(prefix="dtfe_compute_"))
    old_cache, old_sim_dir = config.CACHE_DIR, pipeline.sim_dir
    config.CACHE_DIR = root / "cache"
    pipeline.sim_dir = lambda sim, data_root=None: root / "data" / sim
    try:
        d = root / "data" / config.SIMULATION / "snapdir_099"
        d.mkdir(parents=True)
        np.ones(8, np.float32).tofile(str(d / "output.a_den"))
        (d / "combined_099.hdf5").write_bytes(b"garbage" * 20)                 # unreadable: h5py raises OSError
        buf = _io.StringIO()
        with redirect_stdout(buf):
            try:
                analyze.compute(["050", "099"])
                code = 0
            except SystemExit as e:
                code = int(e.code or 0)
        text = buf.getvalue()
        assert code == 1, text[-500:]
        lines = text.splitlines()

        def own_line(snap):                                                 # the line under 'snapshot NNN (z=...)'
            heads = [i for i, l in enumerate(lines) if l.startswith(f"snapshot {snap} ")]
            assert len(heads) == 1 and heads[0] + 1 < len(lines), (snap, text[-800:])
            return lines[heads[0] + 1]
        l050, l099 = own_line("050"), own_line("099")
        prefix = "    data missing or unreadable, skipped:"
        assert l050.startswith(prefix) and l099.startswith(prefix), (l050, l099)
        assert "snapshot directory not found" in l050, l050                  # no snapdir at all
        assert "signature" in l099 and "no combined_" not in l099 and "FAILED" not in l099, l099   # h5py opened the garbage
        assert "Failed snapshots: 050, 099" in text and "Nothing computed" in text, text[-400:]
        assert not list((root / "cache").rglob("global_limits_*.json"))
    finally:
        config.CACHE_DIR, pipeline.sim_dir = old_cache, old_sim_dir
        shutil.rmtree(root, ignore_errors=True)


def t_slice_maps():
    """Figures' slice-map helpers -- the fixed density range wins, a symmetric
    norm, the stream rule, an empty log slice skipped, no title under the house style, a footnote and ticks,
    a row of panels -- and the three map scripts run in process on a synthetic snapshot, writing their
    53 files. A spy on figures._save sees every
    map the scripts draw (48; the quiver and the histograms save on their own) and pins, under the default
    SHOW_TITLES False, no title anywhere but a colour-bar label naming each panel (lambda_i of the file's web
    in panel i, T-web | V-web | their difference, the house label of each single scalar map -- the class
    map's bar carries tick names instead); the eigenvalue triptychs (both scripts) on norm_signed_log's
    SymLogNorm, the divergence on a 0-centred linear norm, the density maps on DENSITY_MAP_RANGE with the
    floored colour map (density and streams), the other log maps on their own rule; the orientation of all 14 map kinds (the array imshow gets is the mid-plane slice of
    the grid on disk, transposed: rows = the plane's second axis); each kind's figure size. Then an all-zero
    slice: skipped with a note (shear, PS density, PS streams), the comparison's stream panel on the
    single-stream fallback, and plot_DTFE.plot_density's empty density slice skipped too."""
    import io as _io, shutil, sys as _sys, tempfile
    from contextlib import redirect_stdout
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    import config
    from dtfelib import figures as style
    root = Path(tempfile.mkdtemp(prefix="dtfe_maps_"))
    old_cfg = (config.LOCAL_FIGURES_ROOT, config.CACHE_DIR, config.SHOW_TITLES)
    old_argv, old_save = list(_sys.argv), style._save
    plot_dir = str(Path(__file__).resolve().parent.parent / "python" / "plot")
    _sys.path.insert(0, plot_dir)
    patched = []                                                            # (module, attribute, old value)
    try:
        with mpl.rc_context():                  # the scripts' style.apply() (at import, in main) must not outlive the test
            import plot_DTFE, plot_PS_DTFE, plot_cosmic_web                 # BEFORE config is patched: finally restores their REAL paths
            out = root / "out"
            for mod, attr, val in ((plot_DTFE, "OUTPUT_DIR", out / "dtfe"), (plot_PS_DTFE, "FIGURE_ROOT", out / "ps"),
                                   (plot_cosmic_web, "OUTPUT_DIR", out / "web")):
                patched.append((mod, attr, getattr(mod, attr)))
                setattr(mod, attr, val)
            config.LOCAL_FIGURES_ROOT, config.CACHE_DIR = str(root / "figures"), root / "cache"
            config.SHOW_TITLES = False                                      # the house default, stated
            rng = np.random.default_rng(4)
            dens = rng.lognormal(0.0, 1.0, size=(16, 16)).astype(np.float32)
            r = style.slice_map(dens, 100.0, 2, cmap=style.MAP_CMAPS['density'], norm=style.norm_log_range(dens, config.DENSITY_MAP_RANGE),
                                floor=True, label=style.MAP_LABELS['density'], title="a title", path=root / "den.png", footnote="a note")
            assert r["path"].is_file() and (r["vmin"], r["vmax"]) == config.DENSITY_MAP_RANGE, r
            own = style.norm_log_range(dens)                                    # no fixed range: the slice's own
            assert own.vmin == max(float(dens.min()), 1e-6) and own.vmax == float(dens.max())
            assert style.norm_log_range(np.zeros((4, 4))) is None and style.norm_log_range(np.zeros((4, 4)), (0.1, 10.0)).vmin == 0.1
            n_lim = style.norm_log_range(dens, (None, 1e4))
            assert n_lim.vmin == own.vmin and n_lim.vmax == 1e4
            div = rng.normal(size=(16, 16)) - 2.0
            sym = style.norm_symmetric(div)
            assert sym.vmin == -sym.vmax < 0
            assert style.norm_streams(np.zeros((4, 4))) is None
            single = style.norm_streams(np.ones((4, 4), np.float32))
            assert single[0] == 'viridis' and (single[1].vmin, single[1].vmax) == (0.0, 1.0)
            multi = style.norm_streams(np.array([[1.0, 7.0], [1.0, 1.0]], np.float32))
            assert multi[0] == 'inferno' and (multi[1].vmin, multi[1].vmax) == (0.5, 7.0) and style.norm_streams(np.full((2, 2), 1.5))[1].vmax == 2.0
            # no title under the house style (SHOW_TITLES False), one with it on
            fig, ax = plt.subplots(); style.set_title(ax, "t"); assert ax.get_title() == ""; plt.close(fig)
            config.SHOW_TITLES = True
            fig, ax = plt.subplots(); style.set_title(ax, "t"); assert ax.get_title() == "t"; plt.close(fig)
            config.SHOW_TITLES = False
            cmap, norm = style.web_cmap_norm()
            rw = style.slice_row([dict(data=np.zeros((8, 8), int), cmap=cmap, norm=norm, interpolation='nearest', cbar_ticks=[0, 1, 2, 3],
                                       cbar_ticklabels=list(style.WEB_NAMES), title="T"),
                                  dict(data=div, cmap='RdBu_r', norm=style.norm_symmetric(div), title="d")],
                                 100.0, 0, path=root / "row.png", suptitle="row", footnote="under")
            assert rw["path"].is_file() and len(rw["ranges"]) == 2 and rw["ranges"][0] == (-0.5, 3.5)
            # the three scripts, in process, on a synthetic snapshot: their 53 files
            n, z = 8, 0.5
            data = root / "data"
            extra = (("vel", 3), ("velDiv", 1), ("velShear", 5), ("velDisp", 1), ("velDispTensor", 6), ("twebEig", 3), ("velVwebEig", 3))
            snapdir = _synthetic_snapdir(data, "SYN", 50, z, n=n, extra=extra)
            _synthetic_snapdir(data, "SYN", 50, z, n=n, prefix="output", grids=("den",), extra=(("vel", 3), ("velDiv", 1), ("velShear", 5)))
            i, j, k = np.indices((n, n, n))
            den_dtfe = (1.0 + 0.1 * i + 0.01 * j + 0.001 * k).astype(np.float32)   # a different slope per axis; cell 0 = 1: the 'mean' convention
            den_dtfe.tofile(str(snapdir / "output.a_den"))
            den_ps = (1.0 + 0.2 * i + 0.03 * j + 0.004 * k).astype(np.float32)     # other slopes: a script reading the wrong grid shows
            den_ps.tofile(str(snapdir / "ps_output.a_den"))
            seen = []

            def spy(fig, path):                                             # what the map looks like at save time
                panels = []
                for a in fig.axes:
                    if a.images:
                        im = a.images[0]
                        panels.append(dict(data=np.array(np.ma.getdata(im.get_array())), cmap=im.get_cmap(), norm=im.norm,
                                           title=a.get_title(), label=im.colorbar.ax.get_ylabel() if im.colorbar is not None else None))
                sup = getattr(fig, "_suptitle", None)                       # (Figure.get_suptitle is matplotlib >= 3.8: CI and Docker run Ubuntu's 3.6)
                seen.append(dict(path=str(Path(path).relative_to(out)), size=tuple(float(v) for v in fig.get_size_inches()),
                                 suptitle=sup.get_text() if sup is not None else "", panels=panels))
                old_save(fig, path)
            style._save = spy
            for mod, argv in ((plot_DTFE, ["--sim", "SYN", "--data-root", str(data), "--snaps", "50"]),
                              (plot_PS_DTFE, ["--sim", "SYN", "--data-root", str(data), "--snap", "50"]),
                              (plot_cosmic_web, ["--sim", "SYN", "--data-root", str(data), "--snap", "50",
                                                 "--ps-gradient", "stored"])):   # the stored PS V-web pinned below
                _sys.argv = [mod.__name__ + ".py"] + argv
                buf = _io.StringIO()
                with redirect_stdout(buf):
                    try:
                        mod.main(); code = 0
                    except SystemExit as e:
                        code = int(e.code or 0)
                assert code == 0 and "Error" not in buf.getvalue(), (mod.__name__, code, buf.getvalue()[-500:])
            style._save = old_save
            files = sorted(str(q.relative_to(out)) for q in out.rglob("*") if q.is_file())
            assert len(files) == 53, (len(files), files)
            dtfe_maps = [f for f in files if f.startswith("dtfe/")]
            assert len(dtfe_maps) == 12 and all(f.endswith("_z0.50.png") for f in files)
            assert sum(f.startswith("ps/fields/SYN/snap050/") for f in files) == 24 and sum(f.startswith("web/") for f in files) == 17
            assert "web/snapshot_050_z0.50/xy_plane/tweb_vweb_residual_xy_plane_z0.50.png" in files
            assert "ps/fields/SYN/snap050/xy_plane/ps_comparison_xy_plane_z0.50.png" in files

            # ---- what the spy saw: one record per slice_map / slice_row figure
            def kind(rel):                                                  # 'dtfe:density', 'ps:velDisp', 'web:residual', ...
                top, name = rel.split("/")[0], Path(rel).name
                if top == "dtfe":
                    return "dtfe:" + name.split("_")[0]
                if top == "ps":
                    return "ps:" + name.split("_")[1]
                return "web:residual" if name.startswith("tweb_vweb_residual") else "web:" + name.split("_")[1]

            def web_of(rel):                                                # the file's web (the residual row has both)
                return "V" if Path(rel).name.startswith(("vweb_", "ps_vwebEig")) else "T"
            sizes = {"dtfe:density": (8, 7), "dtfe:divergence": (8, 7), "dtfe:shear": (8, 7),
                     "ps:density": (8, 7), "ps:streams": (8, 7), "ps:velDisp": (10, 8.5), "ps:velDispTensor": (10, 8.5),
                     "ps:velMag": (10, 8.5), "ps:twebEig": (21, 6.5), "ps:vwebEig": (21, 6.5), "ps:comparison": (16, 7),
                     "web:classification": (8, 7), "web:eigenvalues": (20, 6), "web:residual": (22, 6)}
            n_panels = {"ps:twebEig": 3, "ps:vwebEig": 3, "web:eigenvalues": 3, "web:residual": 3, "ps:comparison": 2}
            assert len(seen) == 48 and sorted({kind(s["path"]) for s in seen}) == sorted(sizes), (len(seen), sorted({kind(s["path"]) for s in seen}))
            assert len({s["path"] for s in seen}) == 48 and {s["path"] for s in seen} <= set(files)    # one record per written map
            plane_of = {config.SLICE_PLANES[d]['name']: d for d in config.SLICE_PLANES}

            def mid(field, d):                                              # the mid-plane, as imshow wants it (rows vertical)
                return np.take(field, field.shape[d] // 2, axis=d).T

            def grid(prefix, suffix, ncomp=1, exp=None):                    # the grid on disk, in FieldSet's units (a^exp: u-units -> km/s)
                g = np.fromfile(str(snapdir / f"{prefix}.a_{suffix}"), np.float32).reshape((n, n, n) + ((ncomp,) if ncomp > 1 else ()))
                return g * np.float32((1.0 / (1.0 + z)) ** exp) if exp else g
            raw = {"T": grid("ps_output", "twebEig", 3), "V": grid("ps_output", "velVwebEig", 3)}
            div_dtfe = grid("output", "velDiv", 1, 0.5)
            sh = grid("output", "velShear", 5, 0.5)                         # xx xy xz yy yz, zz = -(xx + yy): traceless
            sxx, sxy, sxz, syy, syz = (sh[..., c] for c in range(5))
            szz = -(sxx + syy)
            shear_mag = np.sqrt(sxx ** 2 + syy ** 2 + szz ** 2 + 2 * (sxy ** 2 + sxz ** 2 + syz ** 2))
            vdisp = grid("ps_output", "velDisp", 1, 1.0)                    # a variance: a^1
            t6 = grid("ps_output", "velDispTensor", 6, 1.0)
            txx, txy, txz, tyy, tyz, tzz = (t6[..., c] for c in range(6))
            vdisp_mag = np.sqrt(txx ** 2 + tyy ** 2 + tzz ** 2 + 2 * (txy ** 2 + txz ** 2 + tyz ** 2))
            vmag = np.sqrt((grid("ps_output", "vel", 3, 0.5) ** 2).sum(axis=-1))
            cls = {"T": np.indices((n, n, n))[0] * 4 // n}
            cls["V"] = 3 - cls["T"]
            streams = np.where(np.indices((n, n, n))[0] == n - 1, 3.0, 1.0).astype(np.float32)
            expect = {"dtfe:density": lambda d, w: [mid(den_dtfe, d)],          # per kind: the array of each panel
                      "dtfe:divergence": lambda d, w: [mid(div_dtfe, d)],
                      "dtfe:shear": lambda d, w: [mid(shear_mag, d)],
                      "ps:density": lambda d, w: [mid(den_ps, d)],
                      "ps:streams": lambda d, w: [mid(streams, d)],
                      "ps:velDisp": lambda d, w: [mid(vdisp, d)],
                      "ps:velDispTensor": lambda d, w: [mid(vdisp_mag, d)],
                      "ps:velMag": lambda d, w: [mid(vmag, d)],
                      "ps:comparison": lambda d, w: [mid(den_ps, d), mid(streams, d)],
                      "ps:twebEig": lambda d, w: [mid(raw["T"][..., c], d) for c in range(3)],
                      "ps:vwebEig": lambda d, w: [mid(raw["V"][..., c], d) for c in range(3)],
                      "web:classification": lambda d, w: [mid(cls[w], d)],
                      "web:eigenvalues": lambda d, w: [mid(raw[w][..., c], d) for c in range(3)],
                      "web:residual": lambda d, w: [mid(cls["T"], d), mid(cls["V"], d), mid(cls["T"] - cls["V"], d)]}
            assert set(expect) == set(sizes)                                # every kind's orientation is pinned
            house_label = {"dtfe:density": 'density', "dtfe:divergence": 'divergence', "dtfe:shear": 'shear', "ps:density": 'density',
                           "ps:streams": 'streams', "ps:velDisp": 'velDisp', "ps:velDispTensor": 'velDispTensor', "ps:velMag": 'velMag'}
            floored = {"dtfe:density": (0,), "ps:density": (0,), "ps:streams": (0,), "ps:comparison": (0, 1)}
            density_panels = {"dtfe:density": 0, "ps:density": 0, "ps:comparison": 0}
            eig_kinds = ("ps:twebEig", "ps:vwebEig", "web:eigenvalues")
            for s in seen:
                kd, rel = kind(s["path"]), s["path"]
                d, w = plane_of[Path(rel).parent.name], web_of(rel)
                assert s["size"] == sizes[kd], (rel, s["size"], sizes[kd])          # plot_scalar_log is (10, 8.5)
                assert len(s["panels"]) == n_panels.get(kd, 1), (rel, len(s["panels"]))
                assert s["suptitle"] == "" and all(p["title"] == "" for p in s["panels"]), (rel, s["suptitle"], [p["title"] for p in s["panels"]])
                labels = [p["label"] for p in s["panels"]]
                if len(labels) > 1:                                         # titles are off, the bars name the panels
                    assert all(labels) and len(set(labels)) == len(labels), (rel, labels)
                if kd in house_label:
                    assert labels == [style.MAP_LABELS[house_label[kd]]], (rel, labels)
                if kd == "ps:comparison":
                    assert labels == [style.MAP_LABELS['density'], style.MAP_LABELS['streams']], (rel, labels)
                if kd in eig_kinds:                                         # panel i is lambda_(i+1) of THIS file's web
                    other = {"T": "V", "V": "T"}[w]
                    assert all(rf"\lambda_{ip + 1}" in lab and f"{w}-web" in lab and f"{other}-web" not in lab
                               for ip, lab in enumerate(labels)), (rel, labels)
                if kd == "web:residual":                                    # T-web | V-web | T-web - V-web, in that order
                    lt, lv, lr = labels
                    assert "T-web" in lt and "V-web" not in lt and "V-web" in lv and "T-web" not in lv, (rel, labels)
                    assert "T-web" in lr and "V-web" in lr and lr.index("T-web") < lr.index("V-web"), (rel, labels)
                if kd in eig_kinds:
                    for p in s["panels"]:
                        nm, a = p["norm"], p["data"][np.isfinite(p["data"])]
                        fixed = config.FIELD_LIMITS.get('eigenvalue')
                        vmax = float(fixed) if fixed is not None else float(np.percentile(np.abs(a), config.PERCENTILE_CLIP[1]))
                        ov = config.SYMLOG_LINTHRESH_OVERRIDES.get('eigenvalue')
                        lin = min(ov, vmax / 2) if ov is not None else config.SYMLOG_LINTHRESH_FRAC * vmax
                        assert isinstance(nm, mcolors.SymLogNorm) and nm.vmin == -nm.vmax, (rel, type(nm).__name__, nm.vmin, nm.vmax)
                        assert abs(nm.vmax - vmax) <= 1e-9 * vmax and abs(nm.linthresh - lin) <= 1e-9 * lin, (rel, nm.vmax, vmax, nm.linthresh, lin)
                if kd == "dtfe:divergence":                                 # linear, centred on 0 (white = no divergence) at the |div| percentile
                    nm, a = s["panels"][0]["norm"], np.abs(s["panels"][0]["data"])
                    m = float(np.percentile(a, config.PERCENTILE_CLIP[1]))
                    assert type(nm) is mcolors.Normalize and nm.vmin == -nm.vmax and abs(nm.vmax - m) <= 1e-9 * m, (rel, type(nm).__name__, nm.vmin, nm.vmax, m)
                if kd in ("ps:velDisp", "ps:velDispTensor", "ps:velMag", "dtfe:shear"):    # log: the 1st percentile (shear: its own min) to the max
                    nm, a = s["panels"][0]["norm"], s["panels"][0]["data"]
                    pos = a[a > 0]
                    lo = max(float(pos.min()), 1e-6) if kd == "dtfe:shear" else float(np.percentile(pos, 1))
                    assert isinstance(nm, mcolors.LogNorm) and abs(nm.vmin - lo) <= 1e-9 * lo and nm.vmax == float(a.max()), (rel, nm.vmin, lo, nm.vmax, float(a.max()))
                if kd in density_panels:
                    nm = s["panels"][density_panels[kd]]["norm"]
                    assert isinstance(nm, mcolors.LogNorm) and (nm.vmin, nm.vmax) == tuple(config.DENSITY_MAP_RANGE), (rel, type(nm).__name__, nm.vmin, nm.vmax)
                for ip in floored.get(kd, ()):
                    cm = s["panels"][ip]["cmap"]
                    assert np.allclose(cm.get_bad(), cm(0)) and np.allclose(cm.get_under(), cm(0)), (rel, ip, cm.get_bad(), cm(0))   # the floor
                for ip, want in enumerate(expect[kd](d, w)):                # orientation: the slice on disk, transposed
                    got = s["panels"][ip]["data"]
                    assert got.shape == want.shape and np.allclose(got, want, rtol=1e-6, atol=0), (rel, ip, got[:2], want[:2])
            # an empty slice is skipped with a note, not drawn (nor crashed on); no streams at all: the comparison's
            # stream panel falls back to the single-stream viridis on 0..1
            seen.clear()
            style._save = spy
            zeros, empty = np.zeros((n, n, n), np.float32), root / "out" / "empty"
            buf = _io.StringIO()
            with redirect_stdout(buf):
                plot_DTFE.plot_shear(np.zeros((n, n, n, 5), np.float32), 2, 100.0, z, empty / "shear.png")
                plot_PS_DTFE.plot_density(zeros, 2, 100.0, z, empty / "ps_density.png")
                plot_PS_DTFE.plot_streams(zeros, 2, 100.0, z, empty / "ps_streams.png")
                plot_PS_DTFE.plot_density_comparison(np.ones((n, n, n), np.float32), zeros, 2, 100.0, z, empty / "cmp.png")
            style._save = old_save
            text = buf.getvalue()
            assert "shear: no positive value in this slice, map skipped" in text, text
            assert "No positive density values" in text and "No streams in slice" in text, text
            assert sorted(q.name for q in empty.iterdir()) == ["cmp.png"] and len(seen) == 1, (sorted(q.name for q in empty.iterdir()), len(seen))
            sp = seen[0]["panels"][1]
            assert sp["cmap"].name.startswith(style.MAP_CMAPS['streams_single']) and type(sp["norm"]) is mcolors.Normalize \
                and (sp["norm"].vmin, sp["norm"].vmax) == (0.0, 1.0), (sp["cmap"].name, type(sp["norm"]).__name__, sp["norm"].vmin, sp["norm"].vmax)
            # plot_DTFE.plot_density skips it as plot_PS_DTFE does (under the fixed DENSITY_MAP_RANGE
            # norm_log_range never returns None, so a skip on None would be dead and the slice DRAWN flat)
            buf = _io.StringIO()
            with redirect_stdout(buf):
                plot_DTFE.plot_density(zeros, 2, 100.0, z, empty / "density.png")
            assert "density: no positive value in this slice, map skipped" in buf.getvalue() and not (empty / "density.png").exists(), \
                ("plot_DTFE drew an all-zero density slice", buf.getvalue(), sorted(q.name for q in empty.iterdir()))
    finally:
        style._save = old_save
        for mod, attr, val in patched:
            setattr(mod, attr, val)
        config.LOCAL_FIGURES_ROOT, config.CACHE_DIR, config.SHOW_TITLES = old_cfg
        _sys.argv = old_argv
        if plot_dir in _sys.path:
            _sys.path.remove(plot_dir)
        shutil.rmtree(root, ignore_errors=True)


_POINTEVAL_DRIVER = r'''
import json, runpy, sys
from pathlib import Path
py, script, figroot, argv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.path[:0] = [py, str(Path(script).parent)]
import config
config.LOCAL_FIGURES_ROOT = figroot                    # plot_PS_DTFE reads FIGURE_ROOT at import: patch first
import matplotlib as mpl
from matplotlib.figure import Figure
KEYS = ("font.family", "mathtext.fontset", "font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
        "ytick.labelsize", "axes.linewidth", "savefig.bbox")
seen, _savefig = [], Figure.savefig
def savefig(self, fname, *a, **k):                     # the look at save time: rcParams + what the title was made with
    ax = self.axes[0]
    seen.append({"path": str(fname), "rc": {key: str(mpl.rcParams[key]) for key in KEYS},
                 "title_family": list(ax.title.get_fontfamily()),
                 "math": ax.xaxis.label.get_fontproperties().get_math_fontfamily()})
    return _savefig(self, fname, *a, **k)
Figure.savefig = savefig
sys.argv = [script] + argv
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
print("SPY " + json.dumps({"code": code, "seen": seen}))
'''


def t_pointeval_style():
    """A point-evaluated map has ONE look whichever writer made it -- plot_PS_DTFE.py
    --pointeval-only, plot_PS_DTFE.py's full run (the launcher's 'Field slice maps', which draws them before
    its grid maps) or plot_pointeval.py -- and all three write it to the same path under the figures root.
    Each writer runs in a FRESH interpreter (a subprocess, as the launcher runs them): plot_DTFE and
    plot_cosmic_web apply the house style at IMPORT, so in this process the rcParams depend on which test
    imported what first. A savefig spy records the style rcParams and the title's font at save time; the
    PNGs must decode to the same pixels. The full run's grid maps carry the house serif: the spy sees it."""
    import json, os, shutil, subprocess, sys as _sys, tempfile
    import matplotlib.image as mimg
    root = Path(tempfile.mkdtemp(prefix="dtfe_ptsstyle_"))
    repo = Path(__file__).resolve().parent.parent
    py, plot = repo / "python", repo / "python" / "plot"
    try:
        data = root / "data"
        d = _synthetic_snapdir(data, "SYN", 50, 0.5, grids=("den",))       # density only: the full run draws 3 grid maps, fast
        nu = nv = 16
        rng = np.random.default_rng(29)
        rng.lognormal(0.0, 1.0, size=(1, nv, nu)).astype(np.float64).tofile(str(d / "ps_output.pts_den"))
        np.where(rng.uniform(size=(1, nv, nu)) < 0.2, 3, 1).astype(np.int32).tofile(str(d / "ps_output.pts_streams"))
        (data / "SYN" / "pointeval_plane_16_z.json").write_text(json.dumps(
            {"axis": "z", "u_axis": "x", "v_axis": "y", "u0": 0.0, "u1": 100.0, "v0": 0.0, "v1": 100.0,
             "nu": nu, "nv": nv, "planes": 1, "supersample": 1}))
        common = ["--data-root", str(data)]
        writers = {"pointeval-only": ("plot_PS_DTFE.py", common + ["--sim", "SYN", "--snap", "50", "--pointeval-only"]),
                   "field maps": ("plot_PS_DTFE.py", common + ["--sim", "SYN", "--snap", "50"]),
                   "plot_pointeval": ("plot_pointeval.py", common + ["--sims", "SYN", "--snaps", "50", "--force"])}
        env = dict(os.environ, MPLBACKEND="Agg")
        got = {}
        for i, (name, (script, argv)) in enumerate(writers.items()):
            figroot = root / f"figures_{i}"
            extra = ["--figures-root", str(figroot)] if script == "plot_pointeval.py" else []
            res = subprocess.run([_sys.executable, "-c", _POINTEVAL_DRIVER, str(py), str(plot / script), str(figroot)] + argv + extra,
                                 capture_output=True, text=True, env=env, cwd=str(root), timeout=300)
            spy = [l for l in res.stdout.splitlines() if l.startswith("SPY ")]
            assert res.returncode == 0 and len(spy) == 1, (name, res.returncode, res.stdout[-600:], res.stderr[-1200:])
            rec = json.loads(spy[0][4:])
            assert rec["code"] == 0, (name, rec["code"], res.stdout[-600:])
            got[name] = {str(Path(s["path"]).relative_to(figroot)): s for s in rec["seen"]}
        want = {f"fields/SYN/snap050/xy_plane/ps_{f}_pointeval_xy_plane_z0.50.png" for f in ("density", "streams")}
        for name, recs in got.items():                                      # the same two files, at the same place
            assert {p for p in recs if "_pointeval_" in p} == want, (name, sorted(recs))
        grid = [s for p, s in got["field maps"].items() if "_pointeval_" not in p]
        assert len(grid) == 3 and all(s["rc"]["font.family"] == "['serif']" and s["math"] == "cm" for s in grid), grid   # the spy sees the house style
        for p in sorted(want):
            looks = {name: (recs[p]["rc"], recs[p]["title_family"], recs[p]["math"]) for name, recs in got.items()}
            assert len({json.dumps(v, sort_keys=True) for v in looks.values()}) == 1, (p, looks)
            px = {name: mimg.imread(str(root / f"figures_{i}" / p)) for i, name in enumerate(got)}
            first = px["plot_pointeval"]
            assert all(a.shape == first.shape and np.array_equal(a, first) for a in px.values()), (p, {k: a.shape for k, a in px.items()})
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _run_py(args, env_extra=None, cwd=None, timeout=300):
    """A fresh interpreter (the plot scripts patch config / matplotlib at import): (returncode, stdout + stderr)."""
    import os, subprocess, sys as _sys
    env = dict(os.environ, MPLBACKEND="Agg", **(env_extra or {}))
    res = subprocess.run([_sys.executable] + [str(a) for a in args], capture_output=True, text=True, env=env,
                         cwd=cwd, timeout=timeout)
    return res.returncode, res.stdout + res.stderr


def t_pointeval_supersample_and_estimator():
    """plot_pointeval_panels.py reads a K > 1 plane (planes x nv*K x nu*K points; the density averaged
    over each KxK block as PointPlane does, the stream count the block's largest),
    and plot_pointeval.py renders a standard-DTFE slice (output.pts_*) beside PS-DTFE grids: '--method dtfe', and
    'auto' following the slice on disk (not the grids, which have no ps_output.pts_* here)."""
    import json, shutil, tempfile
    import matplotlib.image as mimg
    root = Path(tempfile.mkdtemp(prefix="dtfe_ptsk_"))
    plot = Path(__file__).resolve().parent.parent / "python" / "plot"
    try:
        data = root / "data"
        d = _synthetic_snapdir(data, "SYN", 50, 0.5, grids=("den",))          # PS-DTFE grids only
        nu, nv, k = 8, 4, 2
        rng = np.random.default_rng(311)
        den = rng.lognormal(0.0, 1.0, size=(1, nv * k, nu * k))
        streams = rng.choice([1, 3, 5], size=(1, nv * k, nu * k)).astype(np.int32)
        den.tofile(str(d / "ps_output.pts_den"))
        streams.tofile(str(d / "ps_output.pts_streams"))
        side = data / "SYN" / "plane_k2.json"
        side.write_text(json.dumps({"axis": "z", "u_axis": "x", "v_axis": "y", "u0": 0.0, "u1": 100.0, "v0": 0.0,
                                    "v1": 50.0, "nu": nu, "nv": nv, "planes": 1, "supersample": k, "center": 50.0,
                                    "thickness": 2.0}))
        _sys_path = list(__import__("sys").path)
        __import__("sys").path.insert(0, str(plot))
        try:
            import plot_pointeval_panels as ppp
            got_den, got_st = ppp.load_planes(d / "ps_output", json.loads(side.read_text()))
        finally:
            __import__("sys").path[:] = _sys_path
        want_den = den.reshape(1, nv, k, nu, k).mean(axis=(2, 4))
        want_st = streams.reshape(1, nv, k, nu, k).max(axis=(2, 4))
        assert got_den.shape == (1, nv, nu) and np.allclose(got_den, want_den, rtol=0, atol=0) \
            and np.array_equal(got_st, want_st), (got_den.shape, got_st.shape)
        out = root / "panels.png"
        rc, log = _run_py([plot / "plot_pointeval_panels.py", "--prefix", d / "ps_output", "--plane", side, "-o", out,
                           "--image", "--panels", "density"])
        assert rc == 0 and out.is_file(), (rc, log[-1500:])
        assert mimg.imread(str(out)).shape[:2] == (nv, nu), mimg.imread(str(out)).shape   # one image pixel per PIXEL

        # a standard-DTFE slice beside the PS-DTFE grids (the K = 1 plane of 16^2)
        for f in ("ps_output.pts_den", "ps_output.pts_streams"):
            (d / f).unlink()
        rng.lognormal(0.0, 1.0, size=(1, 16, 16)).tofile(str(d / "output.pts_den"))
        np.ones((1, 16, 16), np.int32).tofile(str(d / "output.pts_streams"))
        (data / "SYN" / "pointeval_plane_16_z.json").write_text(json.dumps(
            {"axis": "z", "u_axis": "x", "v_axis": "y", "u0": 0.0, "u1": 100.0, "v0": 0.0, "v1": 100.0,
             "nu": 16, "nv": 16, "planes": 1, "supersample": 1}))
        side.unlink()
        rel = "fields/SYN/snap050/xy_plane/{}_density_pointeval_xy_plane_z0.50.png"
        for method in ("auto", "dtfe", "ps"):
            figs = root / f"figs_{method}"
            rc, log = _run_py([plot / "plot_pointeval.py", "--data-root", data, "--sims", "SYN", "--snaps", "50",
                               "--figures-root", figs, "--method", method, "--fields", "density"])
            made = sorted(str(p.relative_to(figs)) for p in figs.rglob("*.png")) if figs.is_dir() else []
            if method == "ps":
                assert made == [] and "no point-evaluated fields" in log, (method, made, log[-800:])
            else:
                assert rc == 0 and made == [rel.format("dtfe")], (method, rc, made, log[-1500:])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def t_halo_tracking_grid_set():
    """plot_halo_tracking.py samples the environment from the grid set asked for (--prefix, --raw), as
    every other script does -- not the default ps_output '_a' grids whatever the launcher chose -- and its
    --data-root reaches the trees, catalogues and grids."""
    import shutil, tempfile
    root = Path(tempfile.mkdtemp(prefix="dtfe_halotrack_"))
    plot = Path(__file__).resolve().parent.parent / "python" / "plot"
    try:
        data = root / "data"
        n = 8
        d = _synthetic_snapdir(data, "SYN", 50, 0.5, n=n, grids=())
        ix, iy, iz = np.indices((n, n, n)).astype(np.float32) + 1.0            # 1..8 along each axis, mean 4.5
        ix.tofile(str(d / "ps_output.a_den"))                                   # each set a ramp along its own axis
        iy.tofile(str(d / "ps_caustic.a_den"))
        iz.tofile(str(d / "ps_caustic.den"))
        drv = r"""
import sys, json
import numpy as np
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from dtfelib import use_data_root
use_data_root(sys.argv[3])
import plot_halo_tracking as pht
box = 100000.0 * 0.6774                                    # the synthetic header: h-free ckpc -> ckpc/h
pos = np.array([[6.5, 1.5, 3.5]]) / 8 * box                # the centre of cell (6, 1, 3)
out = {}
for name, kw in (("default", {}), ("caustic", dict(prefix="ps_caustic")), ("raw", dict(prefix="ps_caustic", averaged=False))):
    t = {0: {"snap": np.array([50]), "pos_ckpc_h": pos.copy()}}
    pht.add_environment(t, "SYN", "auto", **kw)
    out[name] = float(t[0]["env_density"][0])
print("RESULT " + json.dumps(out))
"""
        rc, log = _run_py(["-c", drv, Path(plot).parent, plot, data])
        line = [l for l in log.splitlines() if l.startswith("RESULT ")]
        assert rc == 0 and line, log[-1500:]
        import json
        got = json.loads(line[0][7:])
        want = {"default": 7 / 4.5, "caustic": 2 / 4.5, "raw": 4 / 4.5}
        assert all(abs(got[k] - v) < 1e-6 for k, v in want.items()), got
        other = root / "other root"
        other.mkdir()
        rc, log = _run_py([plot / "plot_halo_tracking.py", "--data-root", data, "--sim", "SYN", "--snap", "50"],
                          env_extra={"DTFE_DATA_ROOT": str(other)})
        assert rc != 0 and f"found under {data / 'SYN'}" in log and str(other) not in log, log[-800:]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def t_evolution_summaries_this_run():
    """plot_velDiv_den.py's and plot_PDF_CDF.py's evolution summaries are built from THIS run's snapshots
    (each summary file also names its simulation and estimator): the per-snapshot summaries of the config ladder sit
    in a folder shared by all simulations, so re-reading them all would let a snapshot this run skipped -- or another
    box's older run -- fill in points, and count a skipped snapshot as processed."""
    import shutil, tempfile
    root = Path(tempfile.mkdtemp(prefix="dtfe_evol_"))
    plot = Path(__file__).resolve().parent.parent / "python" / "plot"
    try:
        data, figs = root / "data", root / "figs"
        rng = np.random.default_rng(5)
        for snap, z in ((99, 0.0), (50, 1.0)):
            d = _synthetic_snapdir(data, "SYN", snap, z, n=8, grids=(), extra=(("velDiv", 1), ("vel", 3)))   # vel: the mean-field divergence
            rng.lognormal(0.0, 0.5, size=(8, 8, 8)).astype(np.float32).tofile(str(d / "ps_output.a_den"))
        stale = figs / "veldiv_density_analysis" / "snapshot_033_z2.00"     # another box's older run, same folder
        stale.mkdir(parents=True)
        (stale / "analysis_summary_z2.00.txt").write_text(
            "Theoretical slope = -99.00 km/s/Mpc\nLinear regime slope = -77.00 km/s/Mpc\n"
            "Difference from theory = -22.2%\nLinear-regime Pearson r = 0.5000\nResidual RMS (linear regime) = 1.00\n")
        rc, log = _run_py([plot / "plot_velDiv_den.py", "--sim", "SYN", "--data-root", data],
                          env_extra={"DTFE_FIGURES_ROOT": str(figs)}, timeout=600)
        evo = figs / "veldiv_density_analysis" / "evolution_data.txt"
        assert rc == 0 and evo.is_file(), (rc, log[-2000:])
        rows = [l.split()[0] for l in evo.read_text().splitlines()[4:] if l.strip()]
        assert rows == ["0.00", "1.00"], (rows, evo.read_text())
        assert "Successfully processed: 2/" in log, log[-600:]
        summ = (figs / "veldiv_density_analysis" / "snapshot_099_z0.00" / "analysis_summary_z0.00.txt").read_text()
        assert "Simulation: SYN, estimator: ps, grids: ps_output*" in summ, summ[:400]
        assert "Divergence: mean field div<v>" in summ, summ[:400]               # the PS default

        # plot_PDF_CDF.py: the same rule (its per-snapshot work stubbed: the critical points need real grids)
        stale2 = figs / "critical_point_analysis" / "snapshot_033_z2.00"
        stale2.mkdir(parents=True)
        (stale2 / "critical_point_analysis_summary_z2.00.txt").write_text(
            "Maxima: 1 (25.0%,\nMinima: 1 (25.0%,\n1saddle: 1 (25.0%,\n2saddle: 1 (25.0%,\nTotal: 4 (1.0e-03 Mpc\n")
        drv = r"""
import sys, json
sys.path[:0] = [sys.argv[1], sys.argv[2]]
sys.argv = ["plot_PDF_CDF.py", "--sim", "SYN"]
import plot_PDF_CDF as m
seen = []
def fake(snapshot, redshift, args):
    if snapshot not in ("050", "099"):
        return None
    return {"redshift": redshift, "total_density": 1e-4, "maxima": 10.0, "minima": 20.0, "1saddle": 30.0, "2saddle": 40.0}
def spy(fig, path, **kw):
    seen.append([list(map(float, l.get_xdata())) for ax in fig.axes for l in ax.lines])
m.process_snapshot, m.save_figure = fake, spy
m.main()
print("RESULT " + json.dumps(seen))
"""
        rc, log = _run_py(["-c", drv, Path(plot).parent, plot], env_extra={"DTFE_FIGURES_ROOT": str(figs), "DTFE_DATA_ROOT": str(data)})
        line = [l for l in log.splitlines() if l.startswith("RESULT ")]
        assert rc == 0 and line, log[-1500:]
        import json
        seen = json.loads(line[0][7:])
        z50 = 1.0 + __import__("config").SNAPSHOT_TO_REDSHIFT["050"]
        assert len(seen) == 2 and all(sorted(x) == sorted([1.0, z50]) for fig in seen for x in fig), seen
        assert "Completed: 2/" in log, log[-600:]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def t_science_fixes():
    """save_figure makes the figure's folder under a figure root with a space and no
    'figures' in its path (any DTFE_FIGURES_ROOT); the ellipsoid cuts are rules turned into Mpc at CALL
    time (10 sigma, L/2, 1 cell); a cache older than its density grid is rebuilt; the profile bins are in
    smoothing lengths, closed at the wrap cap L/6."""
    import os as _os, tempfile as _tf
    from matplotlib.figure import Figure
    import config
    from dtfelib import figures as F, pipeline, profiles as prof
    tmp = Path(_tf.mkdtemp(prefix="dtfe_sci_"))
    old = (config.LOCAL_FIGURES_ROOT, config.SMOOTHING_SIGMA_CELLS, config.CACHE_DIR)
    try:
        # the figure lands where it is saved, its folders made first, and nowhere else
        root = tmp / "DTFE figures"
        config.LOCAL_FIGURES_ROOT = str(root)
        F.save_figure(Figure(figsize=(1, 1)), root / "void_profiles" / "SIM" / "e.png", dpi=20)
        assert [str(q.relative_to(tmp)) for q in tmp.rglob("*.png")] == ["DTFE figures/void_profiles/SIM/e.png"]
        # the cuts at call time: 10 sigma with sigma as set NOW (a --smooth override), L/2 when that binds
        cell, box = float(config.CELL_SIZE), float(config.BOX_SIZE)
        for sig, want in ((10.0, 100.0 * cell), (5.0, 50.0 * cell), (30.0, 0.5 * box)):
            config.SMOOTHING_SIGMA_CELLS = sig
            c = pipeline.ellipsoid_cuts_mpc()
            assert abs(c["max_axis_mpc"] - want) < 1e-9 * want and abs(c["min_axis_mpc"] - cell) < 1e-12 \
                and c["max_axis_ratio"] == 10.0, (sig, c, want)
            assert np.allclose(pipeline._cuts_array(), [cell, want, 10.0])
        config.SMOOTHING_SIGMA_CELLS = old[1]
        # a cache older than the density grid it came from is rebuilt; a newer one is used
        config.CACHE_DIR = tmp / "cache"
        grid = tmp / "ps_output.a_den"
        grid.write_bytes(b"0")

        class _FS:
            method = "ps"

            def _field_path(self, name):
                return grid
        sp = pipeline.SnapshotProducts(50, sim="SYN")
        sp._fs = _FS()
        cp = sp._cpath("voids")
        np.savez_compressed(cp, a=np.arange(3))
        t = cp.stat().st_mtime
        _os.utime(grid, (t - 100, t - 100))
        assert sp._cached("voids") == cp
        _os.utime(grid, (t + 100, t + 100))
        sp._stamp = None                                          # the stamp is taken once per products object
        import io as _io
        from contextlib import redirect_stdout
        buf = _io.StringIO()
        with redirect_stdout(buf):
            assert sp._cached("voids") is None
        assert "older than ps_output.a_den" in buf.getvalue(), buf.getvalue()
        assert sp._cached("critical") is None                     # no cache at all
        # a cache WITH the grid's stamp: equality decides, not the clock (a copy with a new mtime, a future grid)
        sp = pipeline.SnapshotProducts(50, sim="SYN")
        sp._fs = _FS()
        sp._save(cp, {"a": np.arange(3)})
        assert np.array_equal(np.load(cp)["grid_stamp"], [grid.stat().st_size, grid.stat().st_mtime_ns])
        _os.utime(cp, (t - 1000, t - 1000))                      # the cache now OLDER than the grid: the stamp still matches
        assert sp._cached("voids") == cp
        grid.write_bytes(b"01")                                   # another grid (size and time): the stamp no longer does
        _os.utime(cp, (t + 1000, t + 1000))
        sp2 = pipeline.SnapshotProducts(50, sim="SYN")
        sp2._fs = _FS()
        buf = _io.StringIO()
        with redirect_stdout(buf):
            assert sp2._cached("voids") is None
        assert "another version of the grid" in buf.getvalue(), buf.getvalue()

        class _NoPaths:                                           # a cache-only double: no grid to judge by
            method = "ps"
        sp3 = pipeline.SnapshotProducts(50, sim="SYN")
        sp3._fs = _NoPaths()
        assert sp3._cached("voids") == cp
        # sigma = 0 (no smoothing) counts as one cell: no zero-size cut, bins or radii
        config.SMOOTHING_SIGMA_CELLS = 0.0
        assert abs(pipeline.smoothing_length_mpc() - cell) < 1e-12 and abs(pipeline.ellipsoid_cuts_mpc()["max_axis_mpc"] - 10 * cell) < 1e-9
        config.SMOOTHING_SIGMA_CELLS = old[1]
        try:
            prof.default_edges_mpc(0.0, 60.0)
            raise AssertionError("a zero smoothing length gave bins")
        except ValueError:
            pass
        # the wall search in growing cut-outs equals one search at the full half-width
        from dtfelib import skeleton as _sk
        m = np.zeros((48, 48, 48), dtype=bool)
        m[0, 0, 0] = True                                         # one wall cell; near, far (doubling) and unreachable voids
        cen = np.array([[3, 0, 0], [10, 5, 0], [15, 0, 0], [47, 47, 47], [24, 24, 24]])
        d1, b1 = _sk.nearest_wall_distance(m, cen, 20, start_cells=2)
        d2, b2 = _sk.nearest_wall_distance(m, cen, 20, start_cells=20)
        assert np.array_equal(b1, b2) and np.allclose(d1, d2, equal_nan=True), (d1, d2, b1, b2)
        assert np.allclose(d1[:4], [3.0, np.sqrt(125.0), 15.0, np.sqrt(3.0)]) and list(b1) == [False] * 4 + [True], (d1, b1)
        # profile bins in sigma, closed at the wrap cap
        e = prof.default_edges_mpc(2.1625, 110.717449, "r_eff")
        assert np.allclose(e, (0.0, 10.8125, 12.975, 15.1375, 110.717449 / 6)), e
        e = prof.default_edges_mpc(2.1625, 110.717449)                # R_v (the default): 0, 2.5, 3.5, 5 sigma
        assert np.allclose(e, (0.0, 5.40625, 7.56875, 10.8125, 110.717449 / 6)), e
        assert prof.default_edges_mpc(5.0, 60.0) == (0.0, 10.0), prof.default_edges_mpc(5.0, 60.0)
    finally:
        config.LOCAL_FIGURES_ROOT, config.SMOOTHING_SIGMA_CELLS, config.CACHE_DIR = old
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


def t_void_radius_and_physical_smoothing():
    """The measured void radius R_v (where the spherically averaged smoothed delta around the minimum first
    rises to 0) on an analytic field, an overdense minimum and the periodic wrap; distinct voids deepest first (and
    across the box edge); the smoothing in Mpc -- set_smoothing and DTFE_SIGMA_MPC -- with its footprint, and the
    anchor: 10 cells given in Mpc is exactly the current sigma, footprint and cache namespace."""
    import os as _os, subprocess as _sp, sys as _sys
    import config
    from dtfelib import pipeline
    n = 64
    idx = np.indices((n, n, n)).astype(np.float64)

    def bowl(c, r0):                                             # delta = ((r / r0)^2 - 1) / 2: 0 exactly at r = r0
        r2 = sum((((idx[k] + 0.5) - (c[k] + 0.5) + n / 2) % n - n / 2) ** 2 for k in range(3))
        return 0.5 * (r2 / r0 ** 2 - 1.0)
    rv = pipeline.void_radius_cells(bowl((20, 30, 40), 6.0), np.array([[20, 30, 40]]), 1.0)
    assert abs(rv[0] - 6.0) < 0.05, rv                            # linear interpolation over 0.2-cell steps of a parabola
    rv = pipeline.void_radius_cells(bowl((1, 62, 0), 6.0), np.array([[1, 62, 0]]), 1.0)
    assert abs(rv[0] - 6.0) < 0.05, rv                            # the same across the periodic edges
    assert np.isnan(pipeline.void_radius_cells(bowl((20, 30, 40), 6.0) + 1.0, np.array([[20, 30, 40]]), 1.0)[0])
    assert pipeline.void_radius_cells(bowl((20, 30, 40), 6.0), np.zeros((0, 3), int), 1.0).shape == (0,)
    # distinct, deepest first: B lies inside A's R_v, D inside A's across the edge; C and E stand alone
    coords = np.array([[2, 10, 10], [5, 10, 10], [40, 10, 10], [62, 10, 10], [40, 40, 40]])   # D: 4 cells from A via the edge
    delta = np.array([-0.9, -0.5, -0.6, -0.4, -0.3])
    r_v = np.array([5.0, 4.0, 3.0, 2.0, np.nan])
    keep = pipeline.distinct_voids(coords, delta, r_v, n)
    assert list(keep) == [True, False, True, False, False], keep  # E: no R_v, not distinct
    assert list(pipeline.distinct_voids(coords[[1, 2]], delta[[1, 2]], r_v[[1, 2]], n)) == [True, True]   # per sample
    # each void claims min(R_v, R_eff): A's R_v of 20 is capped at its R_eff of 5 (B at 8 cells survives); a void
    # without R_v takes part with its R_eff
    c2 = np.array([[10, 10, 10], [18, 10, 10], [10, 40, 10], [10, 43, 10]])
    d2, rv2, re2 = np.array([-0.9, -0.5, -0.8, -0.3]), np.array([20.0, 3.0, np.nan, 2.0]), np.array([5.0, 4.0, 6.0, 3.0])
    assert list(pipeline.distinct_voids(c2, d2, rv2, n)) == [True, False, False, True]            # R_v alone: A swallows B
    assert list(pipeline.distinct_voids(c2, d2, rv2, n, re2)) == [True, True, True, False]        # capped; C claims R_eff
    # the smoothing in Mpc
    old = (config.SMOOTHING_SIGMA_CELLS, config.FOOTPRINT_SIZE, config.SMOOTHING_SIGMA_MPC)
    try:
        hash10 = pipeline._param_hash()
        assert pipeline.set_smoothing(mpc=10.0 * config.CELL_SIZE) == 10.0 and config.FOOTPRINT_SIZE == 11
        assert pipeline._param_hash() == hash10, (pipeline._param_hash(), hash10)            # the anchor: the current caches
        s = pipeline.set_smoothing(mpc=2.1625)
        assert abs(s - 2.1625 / config.CELL_SIZE) < 1e-8 and config.FOOTPRINT_SIZE == config.footprint_for(s) \
            and config.FOOTPRINT_SIZE % 2 == 1 and abs(config.SMOOTHING_SIGMA_MPC - 2.1625) < 1e-8, (s, config.FOOTPRINT_SIZE)
        fp = config.FOOTPRINT_SIZE
        assert fp != 11 and pipeline.set_smoothing(cells=5.0) == 5.0 and config.FOOTPRINT_SIZE == 11   # cells: production fp
        assert [config.footprint_for(x) for x in (10.0, 3.655, 21.43, 0.5)] == [11, 5, 23, 3]
        try:
            pipeline.set_smoothing(cells=5.0, mpc=1.0)
            raise AssertionError("both units accepted")
        except ValueError:
            pass
        pipeline.set_smoothing(mpc=2.1625)
        assert pipeline.smoothing_tag() == f"s2.1625Mpc_f{config.FOOTPRINT_SIZE}", pipeline.smoothing_tag()   # 6 digits:
        pipeline.set_smoothing(mpc=2.16)                                                      # 2.16 is its own tag
        assert pipeline.smoothing_tag() == f"s2.16Mpc_f{config.FOOTPRINT_SIZE}", pipeline.smoothing_tag()
        assert pipeline.set_smoothing(cells=10.0) == 10.0 and config.FOOTPRINT_SIZE == 11 and pipeline.smoothing_tag() == ""
        pipeline.set_smoothing(mpc=10.0 * config.CELL_SIZE)
        assert pipeline.smoothing_tag() == ""                     # the production smoothing: the default names
    finally:
        config.SMOOTHING_SIGMA_CELLS, config.FOOTPRINT_SIZE, config.SMOOTHING_SIGMA_MPC = old
    # DTFE_SIGMA_MPC at import (the module-level copies and analyze.py follow), in a fresh interpreter per box
    py = Path(__file__).resolve().parents[1] / "python"
    probe = "import config; print(config.SMOOTHING_SIGMA_CELLS, config.FOOTPRINT_SIZE, round(config.SMOOTHING_SIGMA_MPC, 6))"
    for sim, sig, want in (("TNG300-3-Dark", "2.1625", (2.1625 / (302.62769412459404 / 512), 5)),
                           ("TNG50-4-Dark", str(10 * 51.668142899320934 / 512), (10.0, 11))):
        env = dict(_os.environ, DTFE_SIM=sim, DTFE_SIGMA_MPC=sig)
        out = _sp.run([_sys.executable, "-c", probe], cwd=py, env=env, capture_output=True, text=True).stdout.split()
        assert len(out) == 3 and abs(float(out[0]) - want[0]) < 1e-6 and int(out[1]) == want[1], (sim, out)
    # the core figures (saved under their production names) are safe from it: analyze.py compute/plot/all refuse
    env = dict(_os.environ, DTFE_SIM="TNG100-3-Dark", DTFE_SIGMA_MPC="5.0")
    r = _sp.run([_sys.executable, "analyze.py", "plot"], cwd=py, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "refused" in (r.stderr + r.stdout), (r.returncode, r.stderr[-300:])


def t_distinct_claims():
    """catalog.distinct_mask takes its claims from EVERY well-resolved void, each the sphere of
    min(R_v, R_eff) (its R_eff when it has no R_v), then narrows to the caller's sample. The callers pass the voids WITH
    an R_v, so claims from the sample alone would let a deeper void without one claim nothing, and the minima inside
    it count, stack and draw as voids of their own. An unresolved void claims nothing and is never distinct.
    snapshot_stats' counts still add up (no R_v + overlapping + counted = resolved); void_profiles' 'deep' sample
    is the de-duplication within the deep voids, its n_overlapping counting the void a no-R_v one blocks (synthetic
    dicts, a 64^3 frame of config's cell)."""
    import sys as _sys, types
    import config
    from dtfelib import catalog, pipeline, profiles
    plot_dir = str(Path(__file__).resolve().parent.parent / "python" / "plot")
    _sys.path.insert(0, plot_dir)
    try:
        import plot_void_population as pop
    finally:
        _sys.path.remove(plot_dir)
    n, cs = 64, config.CELL_SIZE

    def cat_of(rows):                   # rows of (cell, delta, R_v cells, R_eff cells, resolved, deep)
        reff = np.array([r[3] for r in rows], np.float64)
        return {"coords": np.array([r[0] for r in rows]), "delta_values": np.array([r[1] for r in rows], np.float32),
                "r_v_cells": np.array([r[2] for r in rows], np.float64),
                "semi_axes": np.repeat((reff * cs)[:, None], 3, axis=1).astype(np.float32),   # R_eff = (a b c)^(1/3)
                "well_resolved": np.array([r[4] for r in rows]), "deep": np.array([r[5] for r in rows]),
                "bbks_params": np.tile([0.2, 0.1], (len(rows), 1)), "cell_mpc": cs,
                "positions_mpc": np.array([r[0] for r in rows], np.float64) * cs}
    # A, deep without R_v (R_eff 5 cells), holds B 2 cells away (R_v 3): the R_v sample's own claims would keep B ([F,
    # T])
    two = cat_of([((10, 10, 10), -0.9, np.nan, 5.0, True, True), ((12, 10, 10), -0.5, 3.0, 4.0, True, True)])
    res, rv = two["well_resolved"], catalog.r_v_mpc(two)
    assert catalog.distinct_mask(two, res & np.isfinite(rv), n).tolist() == [False, False], catalog.distinct_mask(two, res & np.isfinite(rv), n)
    assert catalog.distinct_mask(two, res, n).tolist() == [True, False], catalog.distinct_mask(two, res, n)
    # + C apart, D an UNRESOLVED deeper void 1 cell from E: D claims nothing (E stays distinct) and is never distinct itself
    five = cat_of([((10, 10, 10), -0.9, np.nan, 5.0, True, True), ((12, 10, 10), -0.5, 3.0, 4.0, True, True),
                   ((40, 40, 40), -0.6, 4.0, 4.0, True, True), ((30, 10, 10), -0.95, 6.0, 5.0, False, True),
                   ((31, 10, 10), -0.4, 2.0, 3.0, True, False)])
    assert catalog.distinct_mask(five, np.ones(5, bool), n).tolist() == [True, False, True, False, True]
    # snapshot_stats: A no R_v, B overlapping (inside A's R_eff), C and E counted -- 1 + 1 + 2 = the 4 resolved
    st = pop.snapshot_stats(five, n)
    assert (st["n_resolved"], st["n_no_rv"], st["n_overlapping"], st["n_counted"]) == (4, 1, 1, 2), st
    assert st["n_no_rv"] + st["n_overlapping"] + st["n_counted"] == st["n_resolved"]
    assert np.allclose(st["_r_v"], [4.0 * cs, 2.0 * cs]) and not np.isclose(st["_r_v"], 3.0 * cs).any(), st["_r_v"]   # not B's
    # void_profiles: 'deep' (A, C, B -- a depth prefix) == the de-duplication within the deep voids; B blocked by A
    # counts as overlapping; 'resolved' also loses E (2 cells from C, inside its 4)
    prof_cat = cat_of([((10, 10, 10), -0.9, np.nan, 5.0, True, True), ((12, 10, 10), -0.7, 3.0, 4.0, True, True),
                       ((40, 40, 40), -0.8, 4.0, 4.0, True, True), ((42, 40, 40), -0.3, 3.0, 3.0, True, False),
                       ((20, 50, 20), -0.2, 3.0, 3.0, True, False)])

    class _FS:
        grid_n, method = n, "dtfe"
        meta = types.SimpleNamespace(box_mpc=cs * n, redshift=0.0)
        def has(self, name):
            return False
        def density(self, units="mean"):
            return np.ones((n, n, n), np.float32)

    class _Prod:
        sim, fs = config.SIMULATION, _FS()
        def voids(self):
            return prof_cat
    deep = prof_cat["well_resolved"] & prof_cat["deep"]
    rows = np.nonzero(deep)[0]
    within = pipeline.distinct_voids(prof_cat["coords"][rows], prof_cat["delta_values"][rows], prof_cat["r_v_cells"][rows], n,
                                     catalog.r_eff_mpc(prof_cat)[rows] / cs)
    want = [int(r) for r in rows[within & np.isfinite(prof_cat["r_v_cells"][rows])]]
    pd_ = profiles.void_profiles(_Prod(), bins=(0.5, 1.0), m=20, sample="deep")
    assert list(pd_["index"]) == want == [2] and pd_["n_no_radius"] == 1 and pd_["n_overlapping"] == 1, (list(pd_["index"]), want, pd_["n_overlapping"])
    pr = profiles.void_profiles(_Prod(), bins=(0.5, 1.0), m=20, sample="resolved")
    assert list(pr["index"]) == [2, 4] and pr["n_overlapping"] == 2 and pr["dropped"] == 0, (list(pr["index"]), pr["n_overlapping"])


def t_cache_atomic():
    """A cache is written whole or not at all (pipeline._save_npz: a sibling temp file, then os.replace). A
    normal save leaves one file; a save interrupted midway (KeyboardInterrupt from an array) the previous file
    byte-identical and no temp sibling. A cut, empty or non-zip cache (a killed non-atomic write) is 'unreadable' to
    _cached and rebuilt by voids() with ONE write, not a raise on every later load; tools/void_outlines.py turns
    a damaged cache member into its clean 'no void catalogue' exit."""
    import io as _io, os as _os, shutil, sys as _sys, tempfile, types, zipfile
    from contextlib import redirect_stdout
    import config
    from dtfelib import pipeline
    from dtfelib import fields as dtfe
    tmp = Path(tempfile.mkdtemp(prefix="dtfe_atomic_"))
    old_cache, old_save = config.CACHE_DIR, pipeline._save_npz
    vo = old_outlines = None
    old_sim = _os.environ.get("DTFE_SIM")
    try:
        config.CACHE_DIR = tmp / "cache"
        d = tmp / "npz"
        d.mkdir()
        p = d / "x.npz"
        pipeline._save_npz(p, a=np.arange(5))
        assert sorted(q.name for q in d.iterdir()) == ["x.npz"], sorted(q.name for q in d.iterdir())
        with np.load(p) as z:
            assert list(z["a"]) == [0, 1, 2, 3, 4]
        before = p.read_bytes()

        class _Interrupt:                                        # numpy 2 passes dtype/copy, 1.26 dtype only
            def __array__(self, *a, **k):
                raise KeyboardInterrupt
        try:
            pipeline._save_npz(p, a=np.arange(7), b=_Interrupt())  # 'a' is in the zip when 'b' interrupts it
            raise AssertionError("the interrupted save returned")
        except KeyboardInterrupt:
            pass
        assert p.read_bytes() == before and sorted(q.name for q in d.iterdir()) == ["x.npz"], sorted(q.name for q in d.iterdir())
        # _cached: a damaged cache with the grid there is unreadable, said, None (not BadZipFile / EOFError / ValueError)
        grid = tmp / "ps_output.a_den"
        grid.write_bytes(b"0")

        class _FS:
            method = "ps"
            def _field_path(self, name):
                return grid
        sp = pipeline.SnapshotProducts(50, sim="SYN")
        sp._fs = _FS()
        cp = sp._cpath("voids")
        sp._save(cp, {"a": np.arange(2000)})
        assert sp._cached("voids") == cp
        whole = cp.read_bytes()
        for what, data in (("cut", whole[: len(whole) // 2]), ("empty", b""), ("not a zip", b"not a zip file " * 8)):
            cp.write_bytes(data)
            buf = _io.StringIO()
            with redirect_stdout(buf):
                got = sp._cached("voids")
            assert got is None and f"cached {cp.name} is unreadable" in buf.getvalue(), (what, got, buf.getvalue())
        # voids() over a cut catalogue cache, the grid present: rebuilt from the field and written once, then read
        n = 16
        rng = np.random.default_rng(2)
        delta = 0.05 * rng.normal(size=(n, n, n)).astype(np.float32)
        idx = np.indices((n, n, n))
        r2 = sum((((idx[k] - 8 + n // 2) % n) - n // 2) ** 2 for k in range(3))
        delta -= (0.8 * np.exp(-0.5 * r2 / 4.0)).astype(np.float32)
        delta_s = dtfe.smooth_field(delta, sigma=1.0)
        hess = pipeline.hessian_from_field(delta_s)

        class _P(pipeline.SnapshotProducts):
            delta_smoothed = property(lambda self: delta_s)
            hessian = property(lambda self: hess)
            @property
            def fs(self):
                return types.SimpleNamespace(method="dtfe", grid_n=n, _field_path=lambda name: grid,
                                             meta=types.SimpleNamespace(box_mpc=config.CELL_SIZE * n, redshift=1.0))
        pv = _P("050", 1.0, sim="SYN")
        vp = pv._cpath("voids")
        vp.write_bytes(whole[: len(whole) // 2])
        saves = []
        pipeline._save_npz = lambda path, **a: (saves.append(Path(path).name), old_save(path, **a))[1]
        buf = _io.StringIO()
        with redirect_stdout(buf):
            cat = pv.voids()
        assert saves == [vp.name] and "is unreadable" in buf.getvalue() and len(cat["coords"]) >= 1, (saves, buf.getvalue())
        with np.load(vp) as z:
            assert "coords" in z.files and "grid_stamp" in z.files, z.files
        buf = _io.StringIO()
        with redirect_stdout(buf):
            again = pv.voids()
        assert saves == [vp.name] and buf.getvalue() == "" and np.array_equal(again["coords"], cat["coords"]), (saves, buf.getvalue())
        assert not list(Path(config.CACHE_DIR).rglob("*.tmp.npz"))
        # tools/void_outlines.py: a damaged member (BadZipFile, which _cached cannot see) is the clean exit, not a traceback
        tools = str(Path(__file__).resolve().parent.parent / "python" / "tools")
        _sys.path.insert(0, tools)
        try:
            import void_outlines as vo
        finally:
            _sys.path.remove(tools)
        old_outlines = vo.outlines

        def damaged(*a, **k):
            raise zipfile.BadZipFile("Bad CRC-32 for file 'coords.npy'")
        vo.outlines = damaged
        _os.environ["DTFE_SIM"] = "SYN"
        try:
            vo.main(["--sim", "SYN", "--snap", "99"])
            raise AssertionError("void_outlines.main returned on a damaged cache")
        except SystemExit as e:
            assert str(e.code).startswith("no void catalogue for SYN snapshot 99: Bad CRC-32"), e.code
    finally:
        config.CACHE_DIR, pipeline._save_npz = old_cache, old_save
        if vo is not None and old_outlines is not None:
            vo.outlines = old_outlines
        if old_sim is None:
            _os.environ.pop("DTFE_SIM", None)
        else:
            _os.environ["DTFE_SIM"] = old_sim
        shutil.rmtree(tmp, ignore_errors=True)


def t_void_tracking_script():
    """plot_void_tracking.py: catalog_at lets the frame guard's ValueError through (not read
    as 'no field grids'), gives Nones for a snapshot without grids and releases the products every time; main() turns a
    refused reference snapshot into a 'REFUSED' exit before any tree is read; the set's name tag (_set_tag); _remove_stale
    removes only THIS set's void<K> files the run did not write, and nothing when it tracked no void."""
    import io as _io, shutil, sys as _sys, tempfile, types
    from contextlib import redirect_stdout
    from dtfelib import pipeline
    plot_dir = str(Path(__file__).resolve().parent.parent / "python" / "plot")
    _sys.path.insert(0, plot_dir)
    try:
        import plot_void_tracking as pvt
    finally:
        _sys.path.remove(plot_dir)
    ns = types.SimpleNamespace
    tmp = Path(tempfile.mkdtemp(prefix="dtfe_vtrack_"))
    old_products, old_argv = pipeline.products, list(_sys.argv)
    released = []

    class _Prod:
        fs = ns(grid_n=8, method="dtfe", prefix="output.")
        def __init__(self, exc):
            self.exc = exc
        def voids(self):
            if self.exc is not None:
                raise self.exc
            return {"coords": np.zeros((0, 3), int)}
        def release(self):
            released.append(1)
    try:
        args = ns(sim="SYN", method="auto", prefix=None, data_root=None, snap=99)
        refusal = ValueError("the snapshot's cell is 0.4 Mpc: run with DTFE_SIM=SYN")
        pipeline.products = lambda *a, **k: _Prod(refusal)
        try:
            pvt.catalog_at(99, args)
            raise AssertionError("catalog_at swallowed the frame refusal")
        except ValueError as e:
            assert "DTFE_SIM=SYN" in str(e), str(e)
        pipeline.products = lambda *a, **k: _Prod(FileNotFoundError("snapshot directory not found"))
        assert pvt.catalog_at(99, args) == (None, None, None)
        pipeline.products = lambda *a, **k: _Prod(None)
        cat, n, grids = pvt.catalog_at(99, args)
        assert n == 8 and grids == ("dtfe", "output.") and len(released) == 3, (n, grids, released)
        # main(): the reference snapshot refused -> SystemExit naming it, before TreeSet (no trees on disk here)
        pipeline.products = lambda *a, **k: _Prod(refusal)
        _sys.argv = ["plot_void_tracking.py", "--sim", "SYN", "--snap", "99", "--data-root", str(tmp)]
        buf = _io.StringIO()
        with redirect_stdout(buf):
            try:
                pvt.main()
                code = 0
            except SystemExit as e:
                code = e.code
        assert isinstance(code, str) and code.startswith("snapshot 99 of SYN: REFUSED: ") and "DTFE_SIM=SYN" in code, code
        # the set's tag: '' for PS-DTFE's default grids from snapshot 99, else the grids and the reference snapshot
        assert pvt._set_tag(ns(snap=99), ("ps", "ps_output.")) == ""
        assert pvt._set_tag(ns(snap=67), ("dtfe", "output.")) == "_dtfe_snap67"
        assert pvt._set_tag(ns(snap=99), ("ps", "ps_caustic.")) == "_ps_caustic"
        # _remove_stale: void0..5 of three sets ('' in R_v, its R_eff figures, a dtfe set); voids 0, 1, 3 written now
        out = tmp / "vt"
        out.mkdir()
        made = []
        for k in range(6):
            made += [f"void{k}_evolution.png", f"void{k}_track.npz", f"void{k}_evolution_reff.png",
                     f"void{k}_evolution_dtfe.png", f"void{k}_track_dtfe.npz"]
        made.append("voids_summary.png")
        for f in made:
            (out / f).write_bytes(b"x")
        with redirect_stdout(_io.StringIO()):
            pvt._remove_stale(out, ns(tag="", radius="r_v"), {0, 1, 3})
        gone = sorted(set(made) - {q.name for q in out.iterdir()})
        assert gone == sorted(f"void{k}_{s}" for k in (2, 4, 5) for s in ("evolution.png", "track.npz")), gone
        buf = _io.StringIO()
        with redirect_stdout(buf):
            pvt._remove_stale(out, ns(tag="", radius="r_v"), set())
        assert "no void tracked" in buf.getvalue() and len(list(out.iterdir())) == len(made) - 6, buf.getvalue()
    finally:
        pipeline.products = old_products
        _sys.argv = old_argv
        shutil.rmtree(tmp, ignore_errors=True)


def t_figures_path_and_analyze_plot():
    """config.figures_path names a folder under the figure root read at CALL time (ANALYSIS_DIRS' name, else the key);
    figures.save_figure saves there and nowhere else. analyze.py 'plot' hands its scripts the caller's environment
    unchanged (os.environ untouched; run_step stubbed: no plot script runs) and its closing line names the figure root,
    without '--'."""
    import io as _io, shutil, sys as _sys, tempfile
    from contextlib import redirect_stdout
    from matplotlib.figure import Figure
    import config
    from dtfelib import figures as F
    import analyze
    tmp = Path(tempfile.mkdtemp(prefix="dtfe_figpaths_"))
    old = (config.LOCAL_FIGURES_ROOT, config.SIMULATION)
    old_argv, old_step, old_env = list(_sys.argv), analyze.run_step, dict(os.environ)
    try:
        root = tmp / "figs"
        config.LOCAL_FIGURES_ROOT = str(root)
        assert config.figures_path("void_analysis") == f"{root}/void_analysis", config.figures_path("void_analysis")
        assert config.figures_path("smoothing_comparison") == f"{root}/smoothing_comparison"
        F.save_figure(Figure(figsize=(1, 1)), root / "void_analysis" / "a.png", dpi=10)
        assert [str(q.relative_to(tmp)) for q in tmp.rglob("*.png")] == ["figs/void_analysis/a.png"]
        steps = []
        analyze.run_step = lambda cmd, label, env=None: (steps.append((label, env)), (0, 0.0))[1]
        config.SIMULATION = "TNG100-3-Dark"
        _sys.argv = ["analyze.py", "plot", "--only", "plot_eigenvalues.py"]
        buf = _io.StringIO()
        with redirect_stdout(buf):
            try:
                analyze.main()
                code = 0
            except SystemExit as e:
                code = e.code
        lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("All done")]
        assert code in (0, None) and len(lines) == 1 and "--" not in lines[0], (code, buf.getvalue()[-600:])
        assert f"Figures are under {root}; key numbers in {root}/void_analysis/void_shape_table.txt" in lines[0], lines
        assert steps == [("plot_eigenvalues.py", None)] and dict(os.environ) == old_env, steps
    finally:
        config.LOCAL_FIGURES_ROOT, config.SIMULATION = old
        _sys.argv, analyze.run_step = old_argv, old_step
        shutil.rmtree(tmp, ignore_errors=True)


def t_analyze_ladder_on_disk():
    """analyze.py check / compute / export on the snapshots ON DISK (each -3-Dark box has 9 of the 16
    config snapshots). check names the estimator it
    finds and accepts PS-DTFE's grids; an absent snapdir is 'not on disk' and does not fail; an incomplete set exits 1,
    nothing on disk at all exits 1; compute and export without ids take only the snapdirs present and say which they
    left out (an explicit absent id still fails: t_compute_missing_data). A synthetic sim under a temp data root."""
    import io as _io, shutil, sys as _sys, tempfile
    from contextlib import redirect_stdout
    import config
    from dtfelib import cli, pipeline
    import analyze
    root = Path(tempfile.mkdtemp(prefix="dtfe_ladder_"))
    old = (config.SIMULATION, cli._data_root_override, config.CACHE_DIR, config.LOCAL_FIGURES_ROOT, pipeline.products,
           list(_sys.argv))
    sim = "SYNLADDER"

    def call(fn):
        buf = _io.StringIO()
        with redirect_stdout(buf):
            try:
                fn()
                code = 0
            except SystemExit as e:
                code = e.code
        return code, buf.getvalue()

    def check():
        _sys.argv = ["analyze.py", "check"]
        code, text = call(analyze.main)
        return code, text, {ln.split()[0]: ln for ln in text.splitlines() if ln.strip().startswith("snapdir_")}
    try:
        config.SIMULATION = sim
        cli.use_data_root(root / "data")
        config.CACHE_DIR, config.LOCAL_FIGURES_ROOT = root / "cache", str(root / "figures")
        code, text, lines = check()                              # no snapdir of the ladder at all
        assert code == 1 and "no snapshot of the config ladder" in text and len(lines) == 16, (code, text[-400:])
        full = (("vel", 3), ("velDiv", 1), ("velShear", 6), ("velGrad", 9))     # with 'den': the five the scripts read
        for snap in ("050", "099"):                              # the header's redshift is config's (else a mismatch)
            _synthetic_snapdir(root / "data", sim, int(snap), config.get_redshift(snap), grids=("den",), extra=full)
        code, text, lines = check()
        assert code == 0 and all("OK (all PS-DTFE fields)" in lines[f"snapdir_{s}"] for s in ("050", "099")), (code, text[-800:])
        assert sum("not on disk" in ln for ln in lines.values()) == 14 and "not on disk" in lines["snapdir_067"], lines
        (root / "data" / sim / "snapdir_050" / "ps_output.a_velGrad").unlink()
        code, text, lines = check()
        assert code == 1 and "INCOMPLETE (PS-DTFE): missing ps_output.a_velGrad" in lines["snapdir_050"] and "Re-run DTFE for: 050" in text, (code, text[-600:])
        # compute / export without ids: only 050 and 099 are asked for, the 14 others named (products stubbed: no data read)
        asked = []

        class _NoData:
            def __init__(self, snap):
                asked.append(snap)
                self.snapshot = snap
            @property
            def data_dir(self):
                raise FileNotFoundError(f"stub: no grids read for {self.snapshot}")
            def release(self):
                pass
        pipeline.products = lambda snap, z=None, **kw: _NoData(snap)
        absent = ", ".join(s for s in config.SNAPSHOT_TO_REDSHIFT if s not in ("050", "099"))
        code, text = call(lambda: analyze.compute([]))
        assert code == 1 and asked == ["050", "099"] and f"Phase 1 compute: 2 snapshot(s); not on disk, left out: {absent}" in text, (code, asked, text[-600:])
        del asked[:]
        code, text = call(lambda: analyze.export([], root / "export"))
        assert code == 1 and asked == ["050", "099"] and f"Not on disk, left out: {absent}" in text, (code, asked, text[-600:])
    finally:
        config.SIMULATION, old_root, config.CACHE_DIR, config.LOCAL_FIGURES_ROOT, pipeline.products, _sys.argv = old
        cli.use_data_root(old_root)
        shutil.rmtree(root, ignore_errors=True)


def t_pointeval_name_tags():
    """pointeval.render_fields: a non-default rendering -- project='slab', smooth > 0, fixed_range -- gets
    its own name (<base>_slab / _smooth<s> / _fixed): under the plane render's name it would overwrite it,
    or with force=False be skipped as 'up to date' against it. The default name keeps its freshness test."""
    import json, os as _os, shutil, tempfile
    from dtfelib import pointeval
    from dtfelib.io import FieldSet
    root = Path(tempfile.mkdtemp(prefix="dtfe_ptstags_"))
    try:
        data = root / "data"
        d = _synthetic_snapdir(data, "SYN", 50, 0.5, grids=("den",))
        nu = nv = 8
        np.random.default_rng(31).lognormal(0.0, 1.0, size=(2, nv, nu)).astype(np.float64).tofile(str(d / "ps_output.pts_den"))
        t = (d / "ps_output.pts_den").stat().st_mtime
        _os.utime(d / "ps_output.pts_den", (t - 100, t - 100))   # the source clearly older than any render (coarse clocks)
        (data / "SYN" / "pointeval_plane_8_z.json").write_text(json.dumps(
            {"axis": "z", "u_axis": "x", "v_axis": "y", "u0": 0.0, "u1": 100.0, "v0": 0.0, "v1": 100.0,
             "nu": nu, "nv": nv, "planes": 2, "supersample": 1}))
        fs = FieldSet(d, method="ps")
        out, base = root / "out", "ps_density_pointeval_xy_plane_z0.50"
        w, fresh = pointeval.render_fields(fs, out, fields=["density"], force=False, quiet=True)
        assert [q.name for q in w] == [f"{base}.png"] and fresh == 0, (w, fresh)
        plane = out / "xy_plane" / f"{base}.png"
        stamp = plane.stat().st_mtime_ns
        w, fresh = pointeval.render_fields(fs, out, fields=["density"], force=False, quiet=True)
        assert w == [] and fresh == 1, (w, fresh)                # the default: up to date against its own file
        for kw, tag in ((dict(project="slab"), "_slab"), (dict(smooth=2.0), "_smooth2"), (dict(fixed_range=True), "_fixed")):
            w, fresh = pointeval.render_fields(fs, out, fields=["density"], force=False, quiet=True, **kw)
            assert [q.name for q in w] == [f"{base}{tag}.png"] and fresh == 0, (kw, w, fresh)
        assert plane.stat().st_mtime_ns == stamp, "a non-default rendering rewrote the plane render"
        assert sorted(q.name for q in (out / "xy_plane").iterdir()) == sorted(f"{base}{tg}.png" for tg in ("", "_slab", "_smooth2", "_fixed"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def t_void_scripts_convergence():
    """analyze.py voids runs plot_void_convergence.py after the fate census, by name, with
    the '--sim <SIM>' every void script gets (run_scripts' extra); not in the core PLOT_SCRIPTS."""
    import analyze
    assert analyze.VOID_SCRIPTS[-2:] == ["plot_void_fate.py", "plot_void_convergence.py"], analyze.VOID_SCRIPTS
    assert "plot_void_convergence.py" not in analyze.PLOT_SCRIPTS
    seen = []
    old = analyze.run_step
    root = Path(tempfile.mkdtemp(prefix="dtfe_voidreg_"))
    try:
        (root / "plot").mkdir()
        (root / "plot" / "plot_void_convergence.py").write_text("pass\n")
        analyze.run_step = lambda cmd, label, env=None: seen.append((label, cmd[2:])) or (0, 0.0)
        skipped = analyze.run_scripts(analyze.VOID_SCRIPTS, root, {}, extra=("--sim", "SYN"))
        assert seen == [("plot_void_convergence.py", ["--sim", "SYN"])] and "plot_void_fate.py" in skipped, (seen, skipped)
    finally:
        analyze.run_step = old
        shutil.rmtree(root, ignore_errors=True)


def t_cache_prune():
    """analyze.py cache --prune on a synthetic cache root: a '_v1' namespace and a smoothing
    no current setting produces (2.16 Mpc on TNG50-3-Dark: s21.4043_f23) are listed with their sizes, the production
    namespace, a --smooth in cells (s5_f11) and the 2.1625 Mpc one (s21.4291_f23) are kept; inside a kept namespace an
    old fate link (no link_version = 1) and a dead process's temp file are listed, a current link is not; the dry run
    deletes nothing, --yes deletes exactly the list, --keep-mpc 2.16 keeps that namespace; a folder that is not a
    namespace is left alone. The command line refuses --yes without 'cache --prune'."""
    import io as _io, subprocess, sys as _sys
    from contextlib import redirect_stdout
    import config
    import analyze
    from dtfelib import lagrangian
    root = Path(tempfile.mkdtemp(prefix="dtfe_cacheprune_"))
    keys = ("SMOOTHING_SIGMA_CELLS", "FOOTPRINT_SIZE", "FIELD_RESOLUTION", "VOID_EIGENVALUE_CRITERION",
            "GRADIENT_THRESHOLD", "SMOOTHING_SIGMA_MPC")
    old = {k: getattr(config, k) for k in keys}
    old_env = os.environ.pop("DTFE_SIGMA_MPC", None)
    try:
        config.SMOOTHING_SIGMA_CELLS, config.FOOTPRINT_SIZE, config.FIELD_RESOLUTION = 10.0, 11, 512
        config.VOID_EIGENVALUE_CRITERION, config.GRADIENT_THRESHOLD = "trace", 0.1

        def ns(name, *files):
            d = root / name
            d.mkdir()
            for f in files:
                np.savez(d / f, x=np.zeros(64))
            return d
        prod = ns("r512_s10_f11_trace_g0.1_v2", "TNG50-3-Dark_ps_099_voids.npz", "TNG50-3-Dark_ps_fate_067_099.npz")
        np.savez(prod / "TNG50-3-Dark_ps_fate_050_067.npz", link_version=lagrangian.LINK_VERSION)
        dead = subprocess.Popen([_sys.executable, "-c", "pass"])
        dead.wait()                                   # a pid that is gone
        (prod / f"TNG50-3-Dark_ps_050_voids.npz.{dead.pid}.tmp.npz").write_bytes(b"cut")
        (prod / f"TNG50-3-Dark_ps_040_voids.npz.{os.getpid()}.tmp.npz").write_bytes(b"live")    # a live writer's: kept
        old_v1 = ns("r512_s10_f11_trace_g0.1_v1", "099_voids.npz", "050_voids.npz")
        s216 = ns("r512_s21.4043_f23_trace_g0.1_v2", "TNG50-3-Dark_ps_099_voids.npz")
        keep = [prod, ns("r512_s21.4291_f23_trace_g0.1_v2", "TNG50-3-Dark_ps_099_voids.npz"),
                ns("r512_s5_f11_trace_g0.1_v2", "TNG100-3-Dark_ps_099_voids.npz"), ns("notes")]
        inv = analyze.cache_inventory(root)
        listed = sorted(str(e["path"].relative_to(root)) for e in inv if e["prune"])
        want = sorted(["r512_s10_f11_trace_g0.1_v1", "r512_s21.4043_f23_trace_g0.1_v2",
                       "r512_s10_f11_trace_g0.1_v2/TNG50-3-Dark_ps_fate_067_099.npz",
                       f"r512_s10_f11_trace_g0.1_v2/TNG50-3-Dark_ps_050_voids.npz.{dead.pid}.tmp.npz"])
        assert listed == want, listed
        why = {e["path"].name: e["reason"] for e in inv}
        assert "2.1625" in why["r512_s21.4291_f23_trace_g0.1_v2"] and why["r512_s10_f11_trace_g0.1_v2"] == "production smoothing"
        assert "cells" in why["r512_s5_f11_trace_g0.1_v2"] and "v1" in why["r512_s10_f11_trace_g0.1_v1"], why
        before = sorted(str(q.relative_to(root)) for q in root.rglob("*"))
        buf = _io.StringIO()
        with redirect_stdout(buf):
            total = analyze.cache_command(prune=True, root=root)
        text = buf.getvalue()
        assert sorted(str(q.relative_to(root)) for q in root.rglob("*")) == before, "the dry run deleted something"
        assert "DRY RUN" in text and "would delete" in text and total == sum(e["bytes"] for e in inv if e["prune"]) > 0, text
        assert "r512_s21.4291" not in text and "notes" not in text, text
        # --keep-mpc 2.16: the 2.16 Mpc namespace counts as current
        assert not any(e["prune"] and e["path"] == s216 for e in analyze.cache_inventory(root, keep_mpc=(2.16,)))
        with redirect_stdout(_io.StringIO()):
            analyze.cache_command(prune=True, yes=True, root=root)
        assert not old_v1.exists() and not s216.exists() and all(d.is_dir() for d in keep)
        assert sorted(q.name for q in prod.iterdir()) == sorted(["TNG50-3-Dark_ps_099_voids.npz", "TNG50-3-Dark_ps_fate_050_067.npz",
                                                                 f"TNG50-3-Dark_ps_040_voids.npz.{os.getpid()}.tmp.npz"])
        r = subprocess.run([_sys.executable, str(Path(analyze.__file__)), "cache", "--yes"], capture_output=True, text=True)
        assert r.returncode == 2 and "--yes goes with" in r.stderr, (r.returncode, r.stderr[-300:])
    finally:
        for k, v in old.items():
            setattr(config, k, v)
        if old_env is not None:
            os.environ["DTFE_SIGMA_MPC"] = old_env
        shutil.rmtree(root, ignore_errors=True)


def main():
    print("=" * 60)
    print(" dtfelib merger-tree / void-tracking tests")
    print("=" * 60)
    print("synthetic:")
    check("figures_path under the figure root at call time, save_figure there only; analyze.py plot: the caller's environment, a closing line naming the figure root (synthetic)", t_figures_path_and_analyze_plot)
    check("the measured void radius R_v (analytic, periodic, overdense), distinct voids deepest first, the smoothing in Mpc (set_smoothing, DTFE_SIGMA_MPC, footprint) and its 10-cell anchor (synthetic)", t_void_radius_and_physical_smoothing)
    check("distinct_mask's claims from every resolved void (a no-R_v void blocks, an unresolved one never), snapshot_stats' counts add up, void_profiles 'deep' = the de-duplication within it (synthetic)", t_distinct_claims)
    check("save_figure under a root without 'figures' in its path; the cuts are 10 sigma / L/2 / 1 cell at CALL time; a cache older than its grid is rebuilt; profile bins in sigma (synthetic)", t_science_fixes)
    check("_save_npz whole or not at all (an interrupted save leaves the old file, no temp); a cut / empty / non-zip cache is rebuilt with one write; void_outlines' clean exit on a damaged member (synthetic)", t_cache_atomic)
    check("cross-epoch limits store: a subset run keeps the series' maxima (synthetic)", t_limits_store)
    check("map ranges: symmetric about 0, log range of an empty slice (synthetic)", t_map_limits)
    check("webstreams: stream bins + caustic class by web class, chunked == whole (synthetic)", t_webstreams)
    check("void catalogue export: every column against its source, units read back, CSV parse, provenance key by key, the frame guard's hint by cause, no header = FileNotFoundError (synthetic)", t_void_catalog_export)
    check("void cache frame: a header not config's refused before the cache is touched; frame-only, cuts-only and record-less caches re-derived without the field; 'deep' follows its threshold (synthetic)", t_void_cache_frame)
    check("analyze.py export: each exit clause alone, the summary, --format, a void-less catalogue, the default folder (synthetic)", t_export_driver)
    check("void profiles: exact step + outflow, periodic wrap, bulk cancels, the catalogue rows, the hidden bit, the cell-centre convention, the 16/84 band, min_count, linear v_r at z = 0 and 1 (synthetic)", t_void_profiles)
    check("plot_void_profiles.py + plot_void_population.py end to end: npz keys + the --prefix tag, an unreadable snapshot counted failed and the next still done, abundance from each estimator's own catalogue, the one R_eff, 'deep' at the live threshold, --other-estimator at the panel only, the population's _overlaps / _dtfe names, the profiles' _edges tag, a 1.3x header REFUSED before its cache (synthetic)", t_void_scripts)
    check("plot_void_population.py: an unreadable snapshot file is left out and the rest analysed (synthetic)", t_void_population_unreadable)
    check("plot_void_tracking.py: catalog_at lets a frame refusal through, main() exits REFUSED, the set's name tag, only this set's stale void<K> files removed (synthetic)", t_void_tracking_script)
    check("plot_void_crossbox.py worker + --replot figure, tools/void_outlines.py, pipeline.adopt_snapshot_grid (synthetic)", t_void_crossbox_outlines)
    check("power spectrum: a known spectrum recovered for an even and an odd N, Parseval, mode count; linear P(k): sigma_8, D(z), f, k_eq, T(k) pinned (synthetic)", t_power_spectrum)
    check("plot_pk_compare.py end to end: names by estimator/prefix, the _nb<N> / _ref<S> / _noref tags of non-default options, a lone estimator's note, the Nyquist pairs (synthetic)", t_pk_script)
    check("analyze.py voids registers plot_void_convergence.py after the fate census, run with --sim (synthetic)", t_void_scripts_convergence)
    check("analyze.py cache --prune lists the '_v1' and unreachable-smoothing namespaces, old fate links and dead temp files (dry run by default), --yes deletes exactly those, --keep-mpc (synthetic)", t_cache_prune)
    check("caustic skeleton: classes, periodic merge across a face/edge/corner, 6/18/26-connectivity, components, cusp vs swallowtail bits and 'not computed', the parity-only refusal, json_safe, per-void shells + the wall distance (synthetic)", t_caustic_skeleton)
    check("the geometric fold flag: caustic_is_fold with streams/hidden, its slabs and checks, FieldSet.fold_mask on a sample-flagged grid and its fallbacks (synthetic)", t_caustic_fold_exact)
    check("plot_caustic_skeleton.py end to end: files per redshift, strict JSON, '--' cusp rows, --connectivity, nothing in the default folder under --out, no grid / parity-only / failed --voids exit 1, void_block with no or censored voids, the option name tags, the stale _voids.npz removed, the 'distinct' title (synthetic)", t_caustic_skeleton_script)
    check("plot_web_streams.py end to end: files per redshift, a truncated web, an unreadable snapshot file mid-series, the exit rule, the _snaps / _<prefix> / _raw name tags (synthetic)", t_webstreams_script)
    check("analyze.py compute: a missing snapdir and an unreadable snapshot file are counted failed, each with its own reason (synthetic)", t_compute_missing_data)
    check("analyze.py check / compute / export on the snapdirs on disk: PS-DTFE grids complete, absent = 'not on disk' (no failure), incomplete or nothing at all exit 1, the default ids (synthetic)", t_analyze_ladder_on_disk)
    check("slice maps: the house-style helpers' ranges and rules; the three map scripts write the same 53 files with the house norms, floors, panel labels, orientation and sizes; empty slices skipped (synthetic)", t_slice_maps)
    check("point-eval maps: one look and one path from plot_PS_DTFE (--pointeval-only and full) and plot_pointeval.py, fresh interpreters (synthetic)", t_pointeval_style)
    check("point-eval maps: slab / smooth / fixed-range renders get their own names, not 'up to date' against the plane render (synthetic)", t_pointeval_name_tags)
    check("plot_pointeval_panels.py reads a K > 1 plane (KxK blocks: density averaged, streams the largest); plot_pointeval.py renders a standard-DTFE slice beside PS-DTFE grids (--method dtfe, and auto following the slice) (synthetic)", t_pointeval_supersample_and_estimator)
    check("plot_halo_tracking.py samples the environment from --prefix / --raw's grids, and --data-root reaches the trees (synthetic)", t_halo_tracking_grid_set)
    check("plot_velDiv_den.py / plot_PDF_CDF.py evolution summaries from this run's snapshots only, each summary naming its simulation and estimator (synthetic)", t_evolution_summaries_this_run)
    check("sample_grid conventions", t_sample_grid)
    check("periodic helpers", t_periodic_helpers)
    check("track_center across box wrap", t_track_center_wrap)
    check("match_catalog_void (periodic)", t_match_catalog_void)
    check("shape estimators: vectorized == loop reference", t_shape_estimators)
    check("smooth_field / find_local_minima slab-threaded == scipy's serial filters bit for bit (synthetic)",
          t_threaded_filters)
    check("find_critical_points' minimum/maximum box filters slab-threaded == scipy's serial filters bit "
          "for bit at 256^3 (synthetic)", t_threaded_critical_points)
    check("FieldSet caustic loader (synthetic)", t_caustic_field_loader)
    check("velocity_single_stream float-streams + hidden-streams mask (synthetic)", t_velocity_single_stream_float_mask)
    check("FieldSet velocity_scale u-units conversion (synthetic)", t_velocity_scale)
    check("sim_dir / find_sims: per-family and flat data layouts (synthetic)", t_sim_dir_layouts)
    check("FieldSet out-of-core loading: auto/memmap/slice/slabs (synthetic)", t_outofcore_loading)
    check("FieldSet alternate OUTPUT_PREFIX set (synthetic)", t_alternate_prefix)
    global DATA_BACKED
    DATA_BACKED = True
    print(f"data-backed ({SIM}):")
    check("main_branch invariants", t_main_branch_invariants)
    check("descendant_branch inverse walk", t_descendant_inverse)
    check("merger row arithmetic on real trees", t_merger_row_arithmetic)
    check("mergers_along_branch API", t_mergers_api)
    check("groupcat membership + particle ranges", t_groupcat_membership)
    check("environment box frame", t_environment_box)
    check("FieldSet velocity_single_stream (TNG50-4-Dark)", t_velocity_single_stream)
    check("caustic skeleton on the crossed-waves runs (tests/ps_caustic_class_check.sh's output): nested masks, fold in sheet, streams rise with k; pcc_a4 cusp bits, pcc_nocusp 'not computed', pcc_out_gpulin refused", t_caustic_crossed_waves)
    print("-" * 60)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
