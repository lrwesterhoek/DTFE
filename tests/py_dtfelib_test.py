#!/usr/bin/env python3
"""Tests for the Python merger-tree / void-tracking stack (dtfelib).

Two tiers:
  * SYNTHETIC (always run): grid sampling conventions, periodic helpers, void-centre
    tracking across the box wrap, catalog matching.
  * DATA-BACKED (run when the simulation data is on disk, else skipped with a note):
    SubLink invariants on the real TNG50-3-Dark trees -- branch monotonicity, the
    contiguous-row arithmetic, merger-event descendant links -- and the Subfind
    group-membership invariant behind subhalo_particle_range.

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


def main():
    print("=" * 60)
    print(" dtfelib merger-tree / void-tracking tests")
    print("=" * 60)
    print("synthetic:")
    check("sample_grid conventions", t_sample_grid)
    check("periodic helpers", t_periodic_helpers)
    check("track_center across box wrap", t_track_center_wrap)
    check("match_catalog_void (periodic)", t_match_catalog_void)
    print(f"data-backed ({SIM}):")
    check("main_branch invariants", t_main_branch_invariants)
    check("descendant_branch inverse walk", t_descendant_inverse)
    check("merger row arithmetic on real trees", t_merger_row_arithmetic)
    check("mergers_along_branch API", t_mergers_api)
    check("groupcat membership + particle ranges", t_groupcat_membership)
    check("environment box frame", t_environment_box)
    print("-" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
