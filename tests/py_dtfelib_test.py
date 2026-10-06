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

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
# the figure root before config/dtfelib are imported (inherited by the scripts run in fresh interpreters):
# a temp folder, never the real root (the T7 when it is mounted)
_FIGS_TMP = tempfile.mkdtemp(prefix="dtfelib_figs_")
os.environ["DTFE_FIGURES_ROOT"] = _FIGS_TMP
atexit.register(shutil.rmtree, _FIGS_TMP, True)

PASS, FAIL, SKIP = [], [], []
DATA_BACKED = False     # main() sets it before the tier that needs the T7: only THERE is a missing file a skip


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")
    except (FileNotFoundError, KeyError) as e:
        if DATA_BACKED:
            SKIP.append(name)
            print(f"  SKIP  {name}  (data not on disk: {e})")
        else:                   # a synthetic test builds its own inputs: a missing key or file is a FAILURE
            FAIL.append(name)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    except Exception as e:      # any other crash is this test's failure, not the end of the run (review 2026-10-05)
        import traceback
        FAIL.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        print("".join("        " + line for line in traceback.format_exc().splitlines(True)[-6:]), end="")


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

        # '.hidden_streams': bit 1 = multi-stream volume '.streams' does not show, so a cell with
        # it set is masked although its '.streams' reads 1. Bit 2 (a never-sampled tet's mass,
        # usually of the SAME stream) must NOT mask on its own. The 2026-09-28/29 outputs wrote
        # it as '.unresolved' with 0/1 = the old mass flag, read as bit 1 (conservative): the
        # legacy name is exercised first, then the current one. Without either file (older
        # outputs, whose '.streams' counted those tets) the streams test alone decides -- above.
        unres = np.zeros((n, n, n), dtype=np.float32)
        unres[0, :4] = 1.0                 # bit 1 inside the single-stream rows (or an old 0/1 flag)
        unres[1, :4] = 2.0                 # bit 2 only: same-stream sub-sample mass, kept
        unres[1, 4:6] = 3.0                # both bits: masked
        unres[2, 0] = 1.0                  # already multi-stream: masked either way
        old = (unres != 0).astype(np.float32)          # a 2026-09-28/29 output: 0/1 under the old name
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
    per family <root>/<family>/<sim> (the Samsung T7), and flat <root>/<sim> (~/output), under a
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
    """analyze.py compute's cross-epoch limits survive a run on a subset or on nothing (2026-10-06)."""
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
        sp = config.CACHE_DIR / pipeline._param_hash() / f"snapshot_limits_{pipeline.DEFAULT_SIM}.json"
        assert sp.is_file(), f"no per-simulation store {sp.name}"          # (a FileNotFoundError would SKIP, not fail)
        assert set(json.loads(sp.read_text())) == {"050", "099", "004"}
        assert pipeline.canonical_snapshot_id(99) == "099" and pipeline.canonical_snapshot_id("abc") == "abc"
        # one store per simulation (review 2026-10-05): a second simulation's run neither reads nor rewrites the first's
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
    """A diverging map centred on 0, a log map that skips an empty slice (plot_DTFE's crash, 2026-10-05)."""
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
    """Stream bins and caustic class by web class on a synthetic grid with known fractions (2026-10-06)."""
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
    total_m = float(density.sum())
    for name, lab in (("void", 0), ("wall", 1), ("node", 3)):
        assert abs(c[name]["mass_fraction"] - density[wcls == lab].sum() / total_m) < 1e-9, name
    assert abs(c["void"]["stream_bins"]["single"]["mass"] - 111 / 127) < 1e-12      # uniform density within the void
    wall, node = wcls == 1, wcls == 3
    assert abs(c["wall"]["stream_bins"]["1 < s <= 3"]["mass"] - density[wall & (streams == 1.5)].sum() / density[wall].sum()) < 1e-9
    assert abs(c["node"]["stream_bins"]["s > 5"]["mass"] - density[node & (streams == 9.0)].sum() / density[node].sum()) < 1e-9
    assert abs(c["node"]["mean_streams"] - (8 * 9.0 + 120) / 128) < 1e-9
    assert abs(c["node"]["mean_streams_mass"] - (density[node] * streams[node]).sum() / density[node].sum()) < 1e-9
    assert c["node"]["mean_streams_mass"] > c["node"]["mean_streams"]                   # the heavy cell has 9 streams
    # caustic class: the wall plane 2 holds one fold (k=1) and one k=0 cell; the node plane 6 one k=3 -- in both weightings
    assert abs(c["wall"]["fold_fraction"]["volume"] - 1 / 128) < 1e-12 and c["void"]["fold_fraction"]["volume"] == 0.0
    assert abs(c["wall"]["fold_fraction"]["mass"] - 3.0 / density[wall].sum()) < 1e-12 and c["void"]["fold_fraction"]["mass"] == 0.0
    cb = c["wall"]["collapse"]
    assert abs(cb["k=1"]["volume"] - 1 / 128) < 1e-12 and abs(cb["k=0"]["volume"] - 1 / 128) < 1e-12 and abs(cb["none"]["volume"] - 126 / 128) < 1e-12
    assert abs(cb["k=1"]["mass"] - 3.0 / density[wall].sum()) < 1e-12 and abs(cb["k=0"]["mass"] - 1.0 / density[wall].sum()) < 1e-12
    assert abs(cb["none"]["mass"] - (density[wall].sum() - 4.0) / density[wall].sum()) < 1e-12
    assert abs(c["node"]["collapse"]["k=3"]["volume"] - 1 / 128) < 1e-12 and abs(c["node"]["collapse"]["k=3"]["mass"] - 8.0 / density[node].sum()) < 1e-12
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
    # save_table appends its suffixes: 'x_z0.50' and 'x_z0.00' are two files, not one (the first real run, 2026-10-05)
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
    """Survey item 7 (2026-10-05): the void catalogue as CSV + HDF5 with units and provenance, from a catalogue the
    real pipeline built on a synthetic field: every column against its source in the dict, the eigenvalues in the
    Mpc^-2 they are labelled with, the units read back from both files, every provenance value, a stdlib csv parse,
    the frame guard (box, grid, no header) (review 2026-10-05 [0] [10] [17] [18] [23] [34] [36])."""
    import shutil, tempfile, types
    import config
    from dtfelib import pipeline, catalog
    from dtfelib import fields as dtfe
    d = Path(tempfile.mkdtemp(prefix="dtfe_cat_"))
    old_cuts, old_cache, old_deep = config.ELLIPSOID_CUTS, config.CACHE_DIR, config.DEEP_VOID_THRESHOLD
    # cuts no other provenance entry shares (by default min_axis = the gradient threshold = 0.1 and max_axis =
    # max_ratio = 10), a deep threshold off its default (a constant -0.1 would pass at the default): a swapped or
    # hard-coded provenance value cannot pass by coincidence. Set before the build: cuts_mpc / deep_threshold agree
    config.ELLIPSOID_CUTS = {"min_axis_mpc": 0.11, "max_axis_mpc": 9.5, "max_axis_ratio": 7.25}
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
        config.ELLIPSOID_CUTS, config.CACHE_DIR, config.DEEP_VOID_THRESHOLD = old_cuts, old_cache, old_deep
        shutil.rmtree(d, ignore_errors=True)


def _t_void_catalog_export_body(d, cat, n_voids, n, _Prod, catalog, pipeline, config):
    import csv, re
    from datetime import datetime, timezone
    import h5py
    paths = catalog.export_voids(_Prod(config.CELL_SIZE * n), d)
    assert [q.name for q in paths] == ["voids_SYN_dtfe_050_z0.50.csv", "voids_SYN_dtfe_050_z0.50.h5"], [q.name for q in paths]
    names = [c[0] for c in catalog.COLUMNS]
    unit = {c[0]: c[1] for c in catalog.COLUMNS}
    # the labels the [0] pin below holds the values to: eigenvalues per Mpc^2, lengths in Mpc
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
    pos, axes, orient = f64(cat["positions_mpc"]), f64(cat["semi_axes"]), f64(cat["orientations"])
    bbks, ratios = f64(cat["bbks_params"]), f64(cat["axis_ratios"])
    src = {"id": np.arange(n_voids), "ix": cat["coords"][:, 0], "iy": cat["coords"][:, 1], "iz": cat["coords"][:, 2],
           "x_mpc": pos[:, 0], "y_mpc": pos[:, 1], "z_mpc": pos[:, 2], "delta_c": f64(cat["delta_values"]),
           "lambda1": ev[:, 0], "lambda2": ev[:, 1], "lambda3": ev[:, 2],
           "e_bbks": bbks[:, 0], "p_bbks": bbks[:, 1], "b_over_a": ratios[:, 0], "c_over_a": ratios[:, 1],
           "well_resolved": res, "deep": np.asarray(cat["deep"]).astype(bool),
           "a_mpc": axes[:, 0] * nanfit, "b_mpc": axes[:, 1] * nanfit, "c_mpc": axes[:, 2] * nanfit,
           "r_eff_mpc": f64(np.where(res, np.prod(cat["semi_axes"], axis=1) ** (1.0 / 3.0), np.nan)),
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
    # [0]: per Mpc^2 as labelled. The semi-axes are 2 cell / sqrt(|lambda_cell|) = 2 / sqrt(|lambda|) in these units, so
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
                "semi_axes", "orientations", "fit_ellipticity", "fit_prolateness", "positions_mpc")
    empty = {k: (np.asarray(v)[:0] if k in per_void else v) for k, v in cat.items() if k != "cell_mpc"}
    assert all(len(v) == 0 for v in catalog.void_table(empty).values())
    # the major axis IS the eigenvector of the smallest |lambda| (the longest semi-axis), the minor that of the largest
    vecs, ev_abs = f64(cat["eigenvectors"])[res], np.abs(f64(cat["eigenvalues"]))[res]
    for k, pick in (("major", np.argmin), ("minor", np.argmax)):
        axis = np.stack([th[f"{k}_x"], th[f"{k}_y"], th[f"{k}_z"]], 1)[res]
        want = vecs[np.arange(len(vecs)), :, pick(ev_abs, axis=1)]
        assert np.allclose(np.abs(np.einsum("ij,ij->i", axis, want)), 1.0, atol=1e-5), (k, np.einsum("ij,ij->i", axis, want))
    # every provenance value, in both files; the key SET equal (nothing dropped or added; hdf5 also lists 'columns')
    cuts = config.ELLIPSOID_CUTS
    expected = {"catalogue": None, "sim": "SYN", "snapshot": "050", "redshift": 0.5, "method": "dtfe", "prefix": "output",
                "box_mpc": config.CELL_SIZE * n, "box_source": "header", "grid_n": n, "cell_mpc": config.CELL_SIZE,
                "sigma_cells": config.SMOOTHING_SIGMA_CELLS, "sigma_mpc": config.SMOOTHING_SIGMA_CELLS * config.CELL_SIZE,
                "footprint_cells": config.FOOTPRINT_SIZE, "criterion": config.VOID_EIGENVALUE_CRITERION,
                "gradient_threshold": config.GRADIENT_THRESHOLD, "deep_void_threshold": config.DEEP_VOID_THRESHOLD,
                "cut_min_axis_mpc": cuts["min_axis_mpc"], "cut_max_axis_mpc": cuts["max_axis_mpc"],
                "cut_max_axis_ratio": cuts["max_axis_ratio"], "param_hash": pipeline._param_hash(), "n_voids": n_voids,
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
    # on another grid refuses naming FIELD_RESOLUTION (review [23]: 'DTFE_SIM=' alone cannot be acted on there)
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
    # with DTFE_SIM already the snapshot's simulation, 'run with DTFE_SIM=<that same sim>' is no advice (review
    # [15]/[23]): the hint is picked by cause -- the grid, else the header's box against config's
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
    # header for its units, so the frame and the export raise (review [17]: no fallback to config's box)
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
    # a catalogue built under ANOTHER frame (July's TNG100 caches under TNG50's cell): its cell is recovered from
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
    # the abundance (survey item 10): n(> R) and dn/dlnR per Mpc^3 with Poisson errors, NaN radii left out
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
    re-derived 'deep' agrees with its threshold (2026-10-05; review [1] [2] [3] [12])."""
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
        assert np.allclose(cat["cuts_mpc"], [config.ELLIPSOID_CUTS[k] for k in ("min_axis_mpc", "max_axis_mpc", "max_axis_ratio")])
        assert float(cat["deep_threshold"]) == config.DEEP_VOID_THRESHOLD
        same, said = load(0)
        assert said == "", said                                      # the build's records: read as it is, no refit or re-save every load
        assert np.array_equal(same["positions_mpc"], cat["positions_mpc"]) and "cell_mpc" in same
        # a July-style cache: another frame, no record of the cell, the cuts or the threshold -> the fits re-derived, not the field
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
        # the FRAME alone (the record says another cell, the cuts and threshold are today's): re-derived, the cuts not
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
        # ... over a FRESH one (this frame, today's records: a cache hit) too -- once rewritten into config's frame a cache
        # is fresh on every later load, and the voids() consumers without a frame check of their own rely on this one ...
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
        # once; the export's column is that flag and its deep_void_threshold the threshold it was taken at (review [2]:
        # the column used to be the build's, the attribute today's)
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
        back, said = load(1)                                         # back to the thesis threshold: the flag returns
        assert "deep threshold" in said and np.array_equal(back["deep"], cat["deep"]) and float(back["deep_threshold"]) == old_deep, said
        # the cuts retuned: the flag follows without a rebuild
        old_cuts = config.ELLIPSOID_CUTS
        config.ELLIPSOID_CUTS = dict(old_cuts, max_axis_mpc=1e-3)
        try:
            tight, said = load(1)
            assert "ellipsoid cuts" in said and not tight["well_resolved"].any() and not calls, said
        finally:
            config.ELLIPSOID_CUTS = old_cuts
        back, said = load(1)
        assert "ellipsoid cuts" in said and np.array_equal(back["well_resolved"], cat["well_resolved"]), said
    finally:
        config.CACHE_DIR, pipeline._save_npz = old, old_save
        shutil.rmtree(root, ignore_errors=True)


def t_export_driver():
    """analyze.py export (review 2026-10-05 [16] [35]) with pipeline.products stubbed: an id given as '0' is config's
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
                "semi_axes", "orientations", "fit_ellipticity", "fit_prolateness", "positions_mpc")
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
    """Survey item 8 (2026-10-05): shell profiles of a synthetic void with a known step profile and a linear
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
    pf = profiles.void_profiles(_Prod(), bins=wb, m=200)
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
    pd_ = profiles.void_profiles(_Prod(), bins=wb, m=50, sample="deep")
    assert list(pd_["index"]) == [1] and pd_["dropped"] == 1, (list(pd_["index"]), pd_["dropped"])
    # the centre convention, axis by axis: a density linear along axis k has the cell CENTRE value on every shell that
    # does not wrap, 1 + 0.01 (c_k + 0.5): 1.325 for row 1 (cell 32), 1.025 / 1.615 / 1.055 on row 3's first shell
    # (cell 2, 61, 5); a corner on any one axis is 0.005 off. Axis 1's Fibonacci column is exactly antisymmetric
    # (1e-6); axes 0 and 2 carry the lattice's small mean (< 6e-5 at 24 cells), hence 2e-4 there
    for k, c3, tol in ((0, 1.025, 2e-4), (1, 1.615, 1e-6), (2, 1.055, 2e-4)):
        fr = _FS()
        fr.den = 1.0 + 0.01 * (idx[k] + 0.5)
        pr = profiles.void_profiles(_Prod(fr), bins=wb, m=200)
        assert np.abs(pr["density"][0] - 1.325).max() < tol and abs(pr["density"][1, 0] - c3) < tol, (k, pr["density"])
    # the method gate: a standard-DTFE set has no single-stream fraction, even from a fake that claims a streams grid
    fd = _FS()
    fd.method = "dtfe"
    assert profiles.void_profiles(_Prod(fd), bins=wb, m=50)["single"] is None
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
    old_cache, old_deep, old_cuts = config.CACHE_DIR, config.DEEP_VOID_THRESHOLD, config.ELLIPSOID_CUTS
    config.CACHE_DIR = Path(tempfile.mkdtemp(prefix="dtfe_vprof_"))
    try:
        # the cuts in CELLS (a fresh dict, never the live one mutated): rows 1-3 (6-11 cells) resolved, row 0 (200
        # cells) not, whatever DTFE_SIM or a retune of the Mpc cuts says
        config.ELLIPSOID_CUTS = {"min_axis_mpc": 0.5 * cs, "max_axis_mpc": 20.0 * cs, "max_axis_ratio": 10.0}
        config.DEEP_VOID_THRESHOLD = -0.5
        built = pipeline.refit_catalog({"coords": cat["coords"], "eigenvalues": ev, "eigenvectors": np.tile(np.eye(3), (4, 1, 1)),
                                        "delta_values": cat["delta_values"], "deep": cat["delta_values"] < -0.5})
        assert list(built["well_resolved"]) == [False, True, True, True] and list(built["deep"]) == [False, True, True, True], (built["well_resolved"], built["deep"])
        assert np.allclose(catalog.r_eff_mpc(built)[1:], np.array([8.0, 11.0, 6.0]) * cs, rtol=1e-6)
        p = _P("050", 0.0, sim="SYN")
        np.savez_compressed(p._cpath("voids"), **built)
        buf = _io.StringIO()
        with redirect_stdout(buf):
            assert list(profiles.void_profiles(p, bins=wb, m=50, sample="deep")["index"]) == [1, 3]      # the cache as it is
            config.DEEP_VOID_THRESHOLD = -0.75
            got = profiles.void_profiles(p, bins=wb, m=50, sample="deep")
            config.DEEP_VOID_THRESHOLD = -0.5
            back = profiles.void_profiles(p, bins=wb, m=50, sample="deep")
        assert list(got["index"]) == [1] and list(back["index"]) == [1, 3], (list(got["index"]), list(back["index"]), buf.getvalue())
    finally:
        shutil.rmtree(config.CACHE_DIR, ignore_errors=True)
        config.CACHE_DIR, config.DEEP_VOID_THRESHOLD, config.ELLIPSOID_CUTS = old_cache, old_deep, old_cuts


def _void_pin_config(root, saved, n=64, box=64.0):
    """The void-script tests' config: the frame of the synthetic ladder (config's cell = the headers' box / n, else
    frame_of refuses everything), caches + figures + the thesis mirror under 'root', and TODAY's catalogue knobs as
    FRESH objects (never the live dict mutated) -- the fixture's largest semi-axis is 9.3 Mpc against max_axis 10, so
    a retune of the cuts must not move it. The old values go into 'saved' first, for the caller's finally."""
    import config
    pins = {"CACHE_DIR": Path(root) / "cache", "THESIS_FIGURES_DIR": Path(root) / "mirror",
            "LOCAL_FIGURES_ROOT": str(Path(root) / "figures"),
            "CELL_SIZE": box / n, "BOX_SIZE": box, "FIELD_RESOLUTION": n, "SMOOTHING_SIGMA_CELLS": 1.5,
            "ELLIPSOID_CUTS": {"min_axis_mpc": 0.1, "max_axis_mpc": 10.0, "max_axis_ratio": 10.0},
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
    knobs pinned by _void_pin_config, caches, figures and the thesis mirror on temp dirs). Profiles: an .npz + .png
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
    old_argv, old_root, old_voids = list(_sys.argv), cli.DATA_ROOT, pipeline.SnapshotProducts.voids
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
        hits = list((root / "cache").rglob(f"SYN_{method}_{snap:03d}_voids.npz"))
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
        assert code == 0 and names == [f"void_profiles_ps_resolved_z{z}.{e}" for z in ("0.00", "0.50") for e in ("npz", "png")], (code, names, text[-600:])
        assert sorted(q.name for q in (root / "mirror" / "void_profiles" / "SYN").iterdir()) == [q for q in names if q.endswith(".png")]
        assert not (Path(saved["LOCAL_FIGURES_ROOT"]) / "void_profiles" / "SYN").exists()
        f = npz(out / "void_profiles_ps_resolved_z0.00.npz")
        need = {"bins", "edges", "count", "r_eff_mpc", "index", "delta_c", "redshift", "sample", "method", "dropped", "points",
                "prefix", "sigma_cells", "cell_mpc", "box_mpc", "n_grid", "n_boot", "min_count", "cuts_mpc"}
        need |= {f"{k}_{q}" for k in ("density", "v_r", "single") for q in ("mean", "lo", "hi")}
        assert need <= set(f), sorted(need - set(f))
        assert str(f["prefix"]) == "" and float(f["sigma_cells"]) == 1.5 and float(f["cell_mpc"]) == 1.0 and float(f["box_mpc"]) == 64.0, f
        assert int(f["n_grid"]) == 64 and int(f["n_boot"]) == 37 and int(f["min_count"]) == 5 and str(f["method"]) == "ps" and str(f["sample"]) == "resolved", f
        assert list(f["count"]) == [6, 0] and np.isfinite(f["density_mean"][0]).all() and np.isnan(f["density_mean"][1]).all(), f["count"]
        assert float(f["redshift"]) == 0.0 and float(npz(out / "void_profiles_ps_resolved_z0.50.npz")["redshift"]) == 0.5
        cat99 = npz(cached(99))
        assert np.array_equal(f["r_eff_mpc"], catalog.r_eff_mpc(cat99)[f["index"]]) and np.allclose(f["delta_c"], cat99["delta_values"][f["index"]])
        # --prefix: the file is named by the set it holds, the primary set's file untouched
        before = (out / "void_profiles_ps_resolved_z0.00.npz").read_bytes()
        code, text = run(pvp, ["--snaps", "99", "--prefix", "ps_mw"])
        assert code == 0 and (out / "void_profiles_ps_mw_resolved_z0.00.npz").is_file() and (out / "void_profiles_ps_mw_resolved_z0.00.png").is_file(), (code, sorted(q.name for q in out.iterdir()), text[-400:])
        assert str(npz(out / "void_profiles_ps_mw_resolved_z0.00.npz")["prefix"]) == "ps_mw"
        assert (out / "void_profiles_ps_resolved_z0.00.npz").read_bytes() == before
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
            calls.append((int(self.snapshot), self.fs.method))
            return old_voids(self)
        pipeline.SnapshotProducts.voids = counting
        try:
            code, text = run(pop, ["--panel-snaps", "99", "--other-estimator"])
        finally:
            pipeline.SnapshotProducts.voids = old_voids
        popdir = pop.FIGURE_ROOT / "SYN"
        files = {q.name for q in popdir.iterdir()} if popdir.is_dir() else set()
        assert code == 0 and {"population_evolution.png", "population_distributions.png", "population_abundance.png", "population_stats.npz"} <= files, (code, files, text[-600:])
        assert sorted(calls) == [(50, "ps"), (99, "dtfe"), (99, "ps")], calls
        st = npz(popdir / "population_stats.npz")
        parts = ("n_cum", "n_cum_err", "dn_dlnr", "dn_dlnr_err")
        assert {"snaps", "redshift", "box_mpc", "abundance_radii_mpc", "abundance_edges_mpc"} <= set(st), sorted(st)
        for part in parts:
            assert f"abundance_99_{part}" in st and f"abundance_99_other_{part}" in st and f"abundance_50_{part}" not in st, (part, sorted(st))
        assert list(st["snaps"]) == [50, 99] and np.array_equal(st["box_mpc"], [box, box]) and list(st["n_resolved"]) == [6, 6], (st["snaps"], st["box_mpc"], st["n_resolved"])
        # the abundance from each estimator's OWN cached catalogue (R_eff the one definition) and the header's box
        # (config's frame): the PS set has six voids, the standard-DTFE set (one dip fewer) five
        cat99 = npz(cached(99))
        r = catalog.r_eff_mpc(cat99)
        rd = catalog.r_eff_mpc(npz(cached(99, "dtfe")))
        assert np.isfinite(r).sum() == 6 and np.isfinite(rd).sum() == 5, (r, rd)
        for tag, rr in (("", r), ("_other", rd)):
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
        cli.DATA_ROOT = old_root
        pipeline.SnapshotProducts.voids = old_voids
        pvp.OUTPUT_DIR, pop.FIGURE_ROOT = old_out, old_fig
        dict.update(matplotlib.rcParams, old_rc)
        gc.redshift_table.cache_clear()
        shutil.rmtree(root, ignore_errors=True)


def t_void_population_unreadable():
    """plot_void_population.py on a ladder of one good synthetic snapshot and one unreadable snapshot file (a garbage
    combined_098.hdf5 beside a grid): the script's own loop branch -- 'snap 98: unreadable, left out', the rest
    analysed, exit 0. Found by the test review 2026-10-05: groupcat.redshift_table, called BEFORE that loop, opened
    every combined_*.hdf5 of the ladder unguarded, so the OSError aborted the whole run (now: no anchor, interpolated)."""
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
    saved, old_argv, old_root, old_fig = {}, list(_sys.argv), cli.DATA_ROOT, pop.FIGURE_ROOT
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
            except OSError as e:                            # name the frame that let it through (today: redshift_table)
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
        cli.DATA_ROOT = old_root
        pop.FIGURE_ROOT = old_fig
        dict.update(matplotlib.rcParams, old_rc)
        gc.redshift_table.cache_clear()
        shutil.rmtree(root, ignore_errors=True)


def t_power_spectrum():
    """Survey item 25 (2026-10-05): the P(k) estimator on a field built with a KNOWN spectrum (a flat one
    comes back exactly, a power law within the bin average), Parseval over every mode, the Hermitian mode
    count -- for an even AND an odd N (review 2 #27: the k_z = N/2 plane is self-conjugate for even N only);
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
    # T(k) itself, against Eisenstein's TFnowiggles evaluated for config.COSMOLOGY (review 2026-10-05: the sound-horizon
    # term took k in h/Mpc, 1/h too far in k; after the sigma_8 renormalisation 5% at k = 0.02 /Mpc)
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
    one --method, a --prefix), nothing is mirrored under --out, a lone estimator gets a note, and plot_pk draws each
    estimator's own Nyquist pair when the grids differ in size (one grey pair when they agree)."""
    import io as _io, shutil, sys as _sys, tempfile
    from contextlib import redirect_stdout
    import matplotlib.pyplot as plt
    import config
    from dtfelib import spectra
    root = Path(tempfile.mkdtemp(prefix="dtfe_pkrun_"))
    old_thesis, old_argv = config.THESIS_FIGURES_DIR, list(_sys.argv)
    config.THESIS_FIGURES_DIR = root / "mirror"
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
        assert "only PS-DTFE grids here" in text and not (root / "mirror").exists()
        z = np.load(root / "out" / "pk_dtfe+ps_z0.50.npz")
        assert list(z["methods"]) == ["dtfe", "ps"] and str(z["prefix"]) == "" and z["dtfe_k"].shape == z["ps_k"].shape
        code, text, names = run(["--snaps", "50", "--method", "dtfe"])
        assert code == 0 and "pk_dtfe_z0.50.npz" in names and "pk_dtfe+ps_z0.50.npz" in names, names     # the earlier file untouched
        code, text, names = run(["--snaps", "50", "--prefix", "ps_mw"])
        assert code == 0 and "pk_ps_mw_z0.50.png" in names and str(np.load(root / "out" / "pk_ps_mw_z0.50.npz")["prefix"]) == "ps_mw"
        code, text, names = run(["--snaps", "98"])
        assert code == 1 and "failed: [98]" in text, (code, text[-200:])
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
        finally:
            plt.close = real_close
            for f in held.values():
                real_close(f)
    finally:
        config.THESIS_FIGURES_DIR = old_thesis
        _sys.argv = old_argv
        shutil.rmtree(root, ignore_errors=True)


def t_caustic_skeleton():
    """Survey item 9 (2026-10-05): the classes and their fractions on a synthetic caustic_class grid, the
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
    # or as one such cell -- either parity -- in a full deposit; a parity bit WITH a collapse bit is an ordinary deposit (review [24])
    one_pos, one_neg = cc.copy(), cc.copy()
    one_pos[20, 20, 20], one_neg[20, 20, 20] = CAUSTIC_PARITY_POS, CAUSTIC_PARITY_NEG
    for bad in (np.full((4, 4, 4), fold, np.int32), one_pos, one_neg):
        try:
            sk.class_masks(bad)
        except ValueError as e:
            assert "parity-only" in str(e), e
        else:
            raise AssertionError("a parity-only caustic_class grid was read as a stratification")
    assert sk.class_masks(np.full((4, 4, 4), fold | CAUSTIC_COLLAPSE[0], np.int32))["exclusive"]["k=0"].all()
    # the periodic merge: without the wrap the two planes are two components, with it one
    assert ndimage.label(zplanes)[1] == 2 and sk.label_periodic(zplanes, 26)[1] == 1 and sk.label_periodic(zplanes, 6)[1] == 1
    # two cells across ONE box face each: one component under every connectivity (the planes and the tilted loop
    # stay connected when one axis' wrap is dropped -- review [7]); across a box edge 18 and 26 join them, across a
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
    # no bit 7 or 8 anywhere: cusp/swallowtail were NOT COMPUTED (CPU without --ps-caustic-cusps, or the GPU merge), not 0%
    # (review [6]); one cusp and two swallowtail cells are told apart, and either bit alone sets the flag
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
    # tilted sheet: sheet without fold (a fold<->sheet mix-up reads the same at the other two -- review [7])
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


def t_caustic_crossed_waves():
    """The crossed-waves runs tests/ps_caustic_class_check.sh leaves under the suites' temp root ($DTFE_TEST_TMP,
    then ${TMPDIR:-/tmp}/dtfe-tests, then the old tests/tmp): on pcc_out (CPU, no --ps-caustic-cusps) the nested
    cumulative masks, fold within sheet, more streams with every collapsed axis, a JSON-able summary, no cusp bits;
    on pcc_a4 (--ps-caustic-cusps) the cusp and swallowtail fractions are their own bits' and differ; pcc_nocusp
    reads 'not computed'; the parity-only pcc_out_gpulin is refused. No pcc_out: FileNotFoundError -- a SKIP in the
    data-backed tier, never a PASS that did not run (review [40]); an absent sub-fixture is noted and skipped alone."""
    import json, os
    from dtfelib import skeleton as sk
    from dtfelib.io import CAUSTIC_CUSP, CAUSTIC_SWALLOWTAIL
    roots = [Path(os.environ["DTFE_TEST_TMP"])] if os.environ.get("DTFE_TEST_TMP") else []
    roots += [Path(os.environ.get("TMPDIR") or "/tmp") / "dtfe-tests", Path(__file__).resolve().parent / "tmp"]

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
    fx = find("pcc_out", (".causticClass", ".den", ".streams"))
    if fx is None:
        raise FileNotFoundError(f"no crossed-waves run pcc_out.{{causticClass,den,streams}} under {', '.join(map(str, roots))} "
                                f"(tests/ps_caustic_class_check.sh writes it)")
    try:                                    # a missing summary key is a defect here, not missing data (KeyError would SKIP)
        rcc = np.rint(grid(fx, ".causticClass")).astype(np.int32)
        rden, rstr = grid(fx, ".den"), grid(fx, ".streams")
        rm = sk.class_masks(rcc)
        assert (rm["cumulative"]["node"] <= rm["cumulative"]["line"]).all() and (rm["cumulative"]["line"] <= rm["cumulative"]["sheet"]).all()
        assert (rm["indicator"]["fold"] <= rm["cumulative"]["sheet"]).all()               # a fold needs a collapsed axis
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
    cut-out. Then main() in process on synthetic snapshots (its outputs, the thesis mirror and the caches pointed at
    temp dirs): .json/.txt/.png per redshift, a strict JSON, --connectivity passed on, no mirror under --out; a
    snapshot without a grid, without a snapdir or with a parity-only grid exits 1; a failed --voids (the frame
    refusal) exits 1 with 'voids_error' in the JSON (review [25]); a delivered one writes the 'voids' block and npz
    without a RuntimeWarning."""
    import io as _io, json, shutil, sys as _sys, tempfile, types, warnings
    from contextlib import redirect_stdout
    import matplotlib
    import config
    from dtfelib import skeleton as sk
    from dtfelib.io import CAUSTIC_PARITY_POS, CAUSTIC_PARITY_NEG, CAUSTIC_COLLAPSE, CAUSTIC_CUSP, CAUSTIC_DEGENERATE
    root = Path(tempfile.mkdtemp(prefix="dtfe_skelrun_"))
    plot_dir = str(Path(__file__).resolve().parent.parent / "python" / "plot")
    old_cfg = (config.THESIS_FIGURES_DIR, config.LOCAL_FIGURES_ROOT, config.CACHE_DIR)
    old_argv, old_path = list(_sys.argv), list(_sys.path)
    pcs = saved = None
    try:
        if plot_dir not in _sys.path:
            _sys.path.insert(0, plot_dir)
        import plot_caustic_skeleton as pcs                 # BEFORE config is patched: OUTPUT_DIR is set at import
        saved = (pcs.OUTPUT_DIR, pcs.MIRROR, pcs.pipeline)
        config.THESIS_FIGURES_DIR, config.LOCAL_FIGURES_ROOT, config.CACHE_DIR = root / "mirror", str(root / "local"), root / "cache"
        pcs.OUTPUT_DIR = root / "default_out"               # never the repo's figures/ (a run without --out lands here)
        with matplotlib.rc_context():                       # main() calls style.apply(): rcParams back afterwards
            n = 8
            X, Y, Z = np.indices((n, n, n))
            fold = CAUSTIC_PARITY_POS | CAUSTIC_PARITY_NEG
            cc = np.full((n, n, n), CAUSTIC_COLLAPSE[0] | CAUSTIC_PARITY_POS, np.int32)
            cc[X == 2] |= CAUSTIC_COLLAPSE[1] | fold                                  # a wall at x = 2 ...
            cc[(X == 2) & (Y == 4)] |= CAUSTIC_COLLAPSE[2]                            # ... with a filament in it
            # summary_text: no bit 7/8 -> '--' cusp / swallowtail rows and the caveat, not 0.000% (review [6]), the umbilic
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
            # strict JSON, no RuntimeWarning (review [26]); a normal set -> the medians of the finite values
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

            def run(args):
                out = root / "out"
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
            base = ["caustic_skeleton_z0.50.json", "caustic_skeleton_z0.50.png", "caustic_skeleton_z0.50.txt"]
            code, text, names = run(["--snaps", "50"])
            assert code == 0 and names == base, (code, names, text[-600:])
            j = strict(root / "out" / "caustic_skeleton_z0.50.json")
            assert j["cusp_bits_present"] is False and j["snapshot"] == 50 and j["redshift"] == 0.5 and j["prefix"] == "ps_output.", j
            assert j["sim"] == "SYN" and j["connectivity"] == 26 and j["components"]["node"]["n"] == 1, j
            assert abs(sum(e["volume"] for e in j["fractions"]["exclusive"].values()) - 1.0) < 1e-12 and j["components"]["sheet"]["n"] == 1
            assert "voids" not in j and "voids_error" not in j
            t = (root / "out" / "caustic_skeleton_z0.50.txt").read_text()
            assert "no bit 7/8 in this grid" in t and any(ln.split()[:3] == ["cusp", "--", "--"] for ln in t.splitlines()), t
            assert not (root / "mirror").exists() and not (root / "default_out").exists()      # --out: no thesis mirror
            code, text, names = run(["--snaps", "50", "--connectivity", "6"])                   # reaches the components
            j = strict(root / "out" / "caustic_skeleton_z0.50.json")
            assert code == 0 and j["connectivity"] == 6 and j["components"]["node"]["n"] == 2 and "connectivity 6" in text, (code, j["components"])
            code, text, names = run(["--snaps", "50", "99", "98", "97"])
            assert code == 1 and names == base and "failed: [99, 98, 97]" in text, (code, names, text[-600:])
            assert "no caustic_class grid" in text and "parity-only" in text, text[-600:]
            # --voids asked for and refused (the header's 100 Mpc / 8 cell is not config.CELL_SIZE: the frame guard)
            assert abs(100.0 / n - config.CELL_SIZE) > 1e-3 * config.CELL_SIZE, config.CELL_SIZE
            code, text, names = run(["--snaps", "50", "--voids"])
            assert code == 1 and names == base and "failed: [50]" in text and "voids: FAILED" in text, (code, names, text[-600:])
            j = strict(root / "out" / "caustic_skeleton_z0.50.json")
            assert "cell" in j.get("voids_error", "") and "voids" not in j, j.get("voids_error")
            assert not list((root / "cache").rglob("*.npz"))                     # no catalogue written (into the temp cache)

            # --voids delivered: a catalogue in config's frame (a stub of pipeline.products for this script only)
            class _FS:
                grid_n = n
                meta = types.SimpleNamespace(box_mpc=config.CELL_SIZE * n, redshift=0.5)

            class _Prod:
                sim, fs = "SYN", _FS()
                def voids(self):
                    return dict(far, semi_axes=np.full((2, 3), 2.0 * config.CELL_SIZE, np.float32),
                                well_resolved=np.array([True, False]))
                def release(self):
                    pass
            pcs.pipeline = types.SimpleNamespace(products=lambda *a, **kw: _Prod())
            with warnings.catch_warnings():             # undefined medians: None, not an all-NaN / empty (nan)median (review [26]);
                # only numpy's two messages: matplotlib's own autoscaling on the all-NaN column is not under test
                warnings.filterwarnings("error", category=RuntimeWarning, message=r"(All-NaN slice|Mean of empty slice)")
                code, text, names = run(["--snaps", "50", "--voids", "--cutout", "2"])
            assert code == 0 and names == sorted(base + ["caustic_skeleton_z0.50_voids.npz"]), (code, names, text[-600:])
            j = strict(root / "out" / "caustic_skeleton_z0.50.json")
            assert j["voids"] == {"n": 1, "fold_shell_median": 0.0, "dist_reff_median": None, "beyond_cutout": 1} and "voids_error" not in j, j
            with np.load(root / "out" / "caustic_skeleton_z0.50_voids.npz") as zv:
                assert sorted(zv.files) == ["bounded", "dist_cells", "dist_reff", "fold", "index", "r_eff_cells", "sheet"], zv.files
                assert list(zv["index"]) == [0] and bool(zv["bounded"][0]) and np.isnan(zv["dist_cells"][0]) and np.isnan(zv["dist_reff"][0])
            assert not (root / "mirror").exists() and not (root / "default_out").exists()
    finally:
        if saved is not None:
            pcs.OUTPUT_DIR, pcs.MIRROR, pcs.pipeline = saved
        config.THESIS_FIGURES_DIR, config.LOCAL_FIGURES_ROOT, config.CACHE_DIR = old_cfg
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
    """plot_web_streams.py end to end (in-process, its outputs and the thesis mirror pointed at temp dirs):
    two redshifts give two sets of files (z0.50 and z0.00); the tables sum to 1; a truncated V-web grid at one
    snapshot keeps that snapshot's T-web table, is named in the summary and exits 1; a web nobody has exits 1
    only when asked for by name; an unreadable combined file between two good snapshots (review 2 #31: h5py's
    OSError aborted the series) is a failed snapshot, the loop goes on to the next one and the series plot."""
    import io as _io, shutil, sys as _sys, tempfile
    from contextlib import redirect_stdout
    import json
    import matplotlib as mpl
    import config
    root = Path(tempfile.mkdtemp(prefix="dtfe_wsrun_"))
    old_thesis, old_figs, old_argv = config.THESIS_FIGURES_DIR, config.LOCAL_FIGURES_ROOT, list(_sys.argv)
    plot_dir = str(Path(__file__).resolve().parent.parent / "python" / "plot")
    _sys.path.insert(0, plot_dir)
    pws = old_pws = None
    try:
        with mpl.rc_context():                      # main() applies the house style: it must not outlive the test
            import plot_web_streams as pws          # BEFORE config is patched: the import-time paths restored below are the real ones
            old_pws = (pws.OUTPUT_DIR, pws.MIRROR)  # (main() sets the MIRROR global on every run)
            config.THESIS_FIGURES_DIR, config.LOCAL_FIGURES_ROOT = root / "mirror", str(root / "figures")
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
                _sys.argv = ["plot_web_streams.py", "--sim", "SYN", "--data-root", str(data), "--out", str(out)] + args
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
                for w in ("tweb", "vweb"):
                    assert f"web_streams_{w}_z{z}.json" in names and f"web_streams_{w}_z{z}.png" in names and f"web_streams_{w}_z{z}.txt" in names, names
            assert "single_stream_vs_z_tweb.png" in names and "4 table(s) written" in text, text[-300:]
            t = json.loads((out / "web_streams_tweb_z0.50.json").read_text())
            assert abs(sum(e["volume_fraction"] for e in t["classes"].values()) - 1.0) < 1e-12 and t["redshift"] == 0.5 and t["web"] == "T-web"
            assert abs(t["classes"]["node"]["stream_bins"]["1 < s <= 3"]["volume"] - 0.5) < 1e-12      # the multi-stream plane is half of class 3
            assert not (root / "mirror").exists() and not (Path(pws.OUTPUT_DIR) / "SYN").exists(), \
                ("--out wrote into the thesis mirror or the default folder", sorted(str(q.relative_to(root)) for q in root.rglob("*.png") if out not in q.parents))
            shutil.rmtree(out)
            code, text, out = run(["--snaps", "99", "98"])
            names = sorted(q.name for q in out.iterdir())
            assert code == 1 and "web_streams_tweb_z0.01.json" in names and "web_streams_vweb_z0.01.json" not in names, (code, names)
            assert "V-web failed: " in text and "for 098" in text and "3 table(s) written" in text, text[-400:]
            shutil.rmtree(out)
            # the garbage header BETWEEN two good snapshots: 97 failed, 99 and 50 written, the series still drawn
            code, text, out = run(["--snaps", "99", "97", "50"])
            names = sorted(q.name for q in out.iterdir()) if out.is_dir() else []
            assert code == 1, (code, text[-600:])
            for z in ("0.00", "0.50"):
                for w in ("tweb", "vweb"):
                    assert f"web_streams_{w}_z{z}.json" in names, (w, z, names)
            assert not any("z0.25" in q for q in names), names                  # nothing for the unreadable one
            assert "single_stream_vs_z_tweb.png" in names and "single_stream_vs_z_vweb.png" in names, names
            assert "4 table(s) written" in text and "no table for snapshot(s) [97]" in text, text[-400:]
            l097 = [l for l in text.splitlines() if l.startswith("  snapshot 097: ")]
            assert len(l097) == 1 and "signature" in l097[0], text[-600:]           # h5py's own reason, on 097's line
            shutil.rmtree(out)
            code, text, out = run(["--snaps", "99", "--web", "vweb"])
            assert code == 0 and sorted(q.name for q in out.iterdir()) == ["web_streams_vweb_z0.00.json", "web_streams_vweb_z0.00.png", "web_streams_vweb_z0.00.txt"]
            (data / "SYN" / "snapdir_099" / "ps_output.a_velVweb").unlink()
            shutil.rmtree(out)
            code, text, out = run(["--snaps", "99", "--web", "vweb"])
            assert code == 1 and "no table for snapshot(s) [99]" in text, (code, text[-300:])
            code, text, out = run(["--snaps", "99"])                                   # under 'both' a missing web is a skip
            assert code == 0 and "V-web no grid for 099" in text, (code, text[-300:])
    finally:
        if old_pws is not None:
            pws.OUTPUT_DIR, pws.MIRROR = old_pws
        config.THESIS_FIGURES_DIR, config.LOCAL_FIGURES_ROOT = old_thesis, old_figs
        _sys.argv = old_argv
        if plot_dir in _sys.path:
            _sys.path.remove(plot_dir)
        shutil.rmtree(root, ignore_errors=True)


def t_compute_missing_data():
    """analyze.py compute on a series with a missing snapdir and an unreadable snapshot file: both counted
    failed, the loop goes on, nothing recorded, exit 1 (the old exists() test sat after the raise). Each
    snapshot's OWN line is read (review 2 #39: the shared prefix says 'unreadable' for both): 050's names the
    missing folder, 099's carries h5py's 'file signature not found' -- the garbage file WAS opened, not
    skipped as absent and not deferred to warm()'s 'FAILED:'."""
    import io as _io, shutil, sys as _sys, tempfile
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
    """Survey item 11 (2026-10-05): figures' slice-map helpers -- the fixed density range wins, a symmetric
    norm, the stream rule, an empty log slice skipped, no title under the house style, a footnote and ticks,
    a row of panels -- and the three map scripts run in process on a synthetic snapshot, writing the SAME
    53 files as before the refactor (no thesis mirror). Review 2 #8/#9/#30: a spy on figures._save sees every
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
    old_cfg = (config.THESIS_FIGURES_DIR, config.LOCAL_FIGURES_ROOT, config.CACHE_DIR, config.SHOW_TITLES)
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
            config.THESIS_FIGURES_DIR, config.LOCAL_FIGURES_ROOT, config.CACHE_DIR = root / "mirror", str(root / "figures"), root / "cache"
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
            assert not (root / "mirror").exists()                               # the map scripts' maps are not mirrored
            # the three scripts, in process, on a synthetic snapshot: the same files as before the refactor
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

            def spy(fig, path, mirror):                                     # what the map looks like at save time
                panels = []
                for a in fig.axes:
                    if a.images:
                        im = a.images[0]
                        panels.append(dict(data=np.array(np.ma.getdata(im.get_array())), cmap=im.get_cmap(), norm=im.norm,
                                           title=a.get_title(), label=im.colorbar.ax.get_ylabel() if im.colorbar is not None else None))
                sup = getattr(fig, "_suptitle", None)                       # (Figure.get_suptitle is matplotlib >= 3.8: CI and Docker run Ubuntu's 3.6)
                seen.append(dict(path=str(Path(path).relative_to(out)), size=tuple(float(v) for v in fig.get_size_inches()),
                                 suptitle=sup.get_text() if sup is not None else "", panels=panels))
                old_save(fig, path, mirror)
            style._save = spy
            for mod, argv in ((plot_DTFE, ["--sim", "SYN", "--data-root", str(data), "--snaps", "50"]),
                              (plot_PS_DTFE, ["--sim", "SYN", "--data-root", str(data), "--snap", "50"]),
                              (plot_cosmic_web, ["--sim", "SYN", "--data-root", str(data), "--snap", "50"])):
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
            assert not (root / "mirror").exists()

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
                assert s["size"] == sizes[kd], (rel, s["size"], sizes[kd])          # #30: plot_scalar_log is (10, 8.5) again
                assert len(s["panels"]) == n_panels.get(kd, 1), (rel, len(s["panels"]))
                assert s["suptitle"] == "" and all(p["title"] == "" for p in s["panels"]), (rel, s["suptitle"], [p["title"] for p in s["panels"]])
                labels = [p["label"] for p in s["panels"]]
                if len(labels) > 1:                                         # #8: titles are off, the bars name the panels
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
            # plot_DTFE.plot_density skips it as plot_PS_DTFE does (test review 2026-10-06: under the fixed
            # DENSITY_MAP_RANGE norm_log_range never returns None, so its old skip was dead and the slice was DRAWN flat)
            buf = _io.StringIO()
            with redirect_stdout(buf):
                plot_DTFE.plot_density(zeros, 2, 100.0, z, empty / "density.png")
            assert "density: no positive value in this slice, map skipped" in buf.getvalue() and not (empty / "density.png").exists(), \
                ("plot_DTFE drew an all-zero density slice", buf.getvalue(), sorted(q.name for q in empty.iterdir()))
    finally:
        style._save = old_save
        for mod, attr, val in patched:
            setattr(mod, attr, val)
        config.THESIS_FIGURES_DIR, config.LOCAL_FIGURES_ROOT, config.CACHE_DIR, config.SHOW_TITLES = old_cfg
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
config.THESIS_FIGURES_DIR = str(Path(figroot) / "mirror")
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
    """Review 2 #29: a point-evaluated map has ONE look whichever writer made it -- plot_PS_DTFE.py
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


def t_thesis_mirror_outside_repo():
    """config.THESIS_FIGURES_DIR lies OUTSIDE the repo. From 2026-07-10 to 2026-10-06 its fallback was
    DTFE/Figures (config.py had moved into python/), which on macOS's case-insensitive disk is the repo's own
    figures/ folder: every figure save_plot_to_multiple_paths mirrored landed in the repo, not the thesis."""
    import config
    repo = str(Path(__file__).resolve().parents[1]).lower()
    mirror = Path(str(config.THESIS_FIGURES_DIR)).resolve()
    low = str(mirror).lower()
    assert not (low == repo or low.startswith(repo + "/")), f"the thesis mirror {mirror} is inside the repo"
    if config._THESIS_DIR is None:
        assert mirror == Path(__file__).resolve().parents[2] / "Figures", mirror


def main():
    print("=" * 60)
    print(" dtfelib merger-tree / void-tracking tests")
    print("=" * 60)
    print("synthetic:")
    check("the thesis figure mirror lies outside the repo (case-insensitively: DTFE/Figures IS DTFE/figures on macOS)", t_thesis_mirror_outside_repo)
    check("cross-epoch limits store: a subset run keeps the series' maxima (synthetic)", t_limits_store)
    check("map ranges: symmetric about 0, log range of an empty slice (synthetic)", t_map_limits)
    check("webstreams: stream bins + caustic class by web class, chunked == whole (synthetic)", t_webstreams)
    check("void catalogue export: every column against its source, units read back, CSV parse, provenance key by key, the frame guard's hint by cause, no header = FileNotFoundError (synthetic)", t_void_catalog_export)
    check("void cache frame: a header not config's refused before the cache is touched; frame-only, cuts-only and July caches re-derived without the field; 'deep' follows its threshold (synthetic)", t_void_cache_frame)
    check("analyze.py export: each exit clause alone, the summary, --format, a void-less catalogue, the default folder (synthetic)", t_export_driver)
    check("void profiles: exact step + outflow, periodic wrap, bulk cancels, the catalogue rows, the hidden bit, the cell-centre convention, the 16/84 band, min_count, linear v_r at z = 0 and 1 (synthetic)", t_void_profiles)
    check("plot_void_profiles.py + plot_void_population.py end to end: npz keys + the --prefix tag, an unreadable snapshot counted failed and the next still done, abundance from each estimator's own catalogue, the one R_eff, 'deep' at the live threshold, --other-estimator at the panel only, a 1.3x header REFUSED before its cache (synthetic)", t_void_scripts)
    check("plot_void_population.py: an unreadable snapshot file is left out and the rest analysed (synthetic)", t_void_population_unreadable)
    check("power spectrum: a known spectrum recovered for an even and an odd N, Parseval, mode count; linear P(k): sigma_8, D(z), f, k_eq, T(k) pinned (synthetic)", t_power_spectrum)
    check("plot_pk_compare.py end to end: names by estimator/prefix, no mirror under --out, a lone estimator's note, the Nyquist pairs (synthetic)", t_pk_script)
    check("caustic skeleton: classes, periodic merge across a face/edge/corner, 6/18/26-connectivity, components, cusp vs swallowtail bits and 'not computed', the parity-only refusal, json_safe, per-void shells + the wall distance (synthetic)", t_caustic_skeleton)
    check("plot_caustic_skeleton.py end to end: files per redshift, strict JSON, '--' cusp rows, --connectivity, no mirror under --out, no grid / parity-only / failed --voids exit 1, void_block with no or censored voids (synthetic)", t_caustic_skeleton_script)
    check("plot_web_streams.py end to end: files per redshift, a truncated web, an unreadable snapshot file mid-series, the exit rule (synthetic)", t_webstreams_script)
    check("analyze.py compute: a missing snapdir and an unreadable snapshot file are counted failed, each with its own reason (synthetic)", t_compute_missing_data)
    check("slice maps: the house-style helpers' ranges and rules; the three map scripts write the same 53 files with the house norms, floors, panel labels, orientation and sizes; empty slices skipped (synthetic)", t_slice_maps)
    check("point-eval maps: one look and one path from plot_PS_DTFE (--pointeval-only and full) and plot_pointeval.py, fresh interpreters (synthetic)", t_pointeval_style)
    check("sample_grid conventions", t_sample_grid)
    check("periodic helpers", t_periodic_helpers)
    check("track_center across box wrap", t_track_center_wrap)
    check("match_catalog_void (periodic)", t_match_catalog_void)
    check("shape estimators: vectorized == loop reference", t_shape_estimators)
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
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
