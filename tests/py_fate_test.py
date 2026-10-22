#!/usr/bin/env python3
"""Tests for the void FATE stack: dtfelib.lagrangian (members, links, classes), plot_void_fate.py
(census, table, figure, the link cache) and plot_void_tracking.py (the tracker on synthetic merger trees, and its
--link particles mode).

Two tiers:
  * SYNTHETIC (always run, CI): particles and catalogues built here, a SubLink-like tree object.
  * DATA-BACKED (DTFE_DATA_ROOT, else skipped): one chunk of the real TNG50-3-Dark combined snapshot.

Usage:  python3 tests/py_fate_test.py            (exit 0 = pass)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import os
from _harness import REPO as ROOT, Skip, check, data_root, setup, summary

_TMP = setup("fate_test_", figures_sub="figs")
os.environ.setdefault("DTFE_SIM", "TNG50-4-Dark")    # the box these synthetic tests stand for; a data-tier run sets its own


# ---------------------------------------------------------------- a synthetic two-snapshot universe
def _ball(rng, centre, r, n):
    """n points uniformly inside a ball (fractional units, wrapped)."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    v *= r * rng.random(n)[:, None] ** (1 / 3)
    return np.mod(np.asarray(centre) + v, 1.0)


def _universe(seed=0):
    """Five progenitor voids (A) and four descendant voids (B), with the particles of each A void placed by
    construction: A0 -> B0 whole (SURVIVE, and A0 sits on the box corner: periodic), A1 and A2 -> B1 (MERGE),
    A3 -> 40% B2 + 40% B3 (SPLIT), A4 -> outside every sphere (VANISH). Plus 'field' particles in no sphere."""
    rng = np.random.default_rng(seed)
    ca = np.array([[0.02, 0.02, 0.98], [0.3, 0.3, 0.3], [0.38, 0.3, 0.3], [0.7, 0.2, 0.5], [0.5, 0.8, 0.5]])
    ra = np.full(5, 0.04)
    cb = np.array([[0.01, 0.03, 0.99], [0.34, 0.3, 0.3], [0.7, 0.12, 0.5], [0.7, 0.28, 0.5]])
    rb = np.array([0.05, 0.08, 0.035, 0.035])
    n = 400
    ids, pa, pb = [], [], []
    nxt = 1000
    for i in range(5):
        ids.append(np.arange(nxt, nxt + n, dtype=np.uint64)); nxt += n
        pa.append(_ball(rng, ca[i], 0.97 * ra[i], n))
    pb.append(_ball(rng, cb[0], 0.9 * rb[0], n))
    pb.append(_ball(rng, cb[1], 0.9 * rb[1], n))
    pb.append(_ball(rng, cb[1], 0.9 * rb[1], n))
    pb.append(np.concatenate([_ball(rng, cb[2], 0.9 * rb[2], 160), _ball(rng, cb[3], 0.9 * rb[3], 160),
                              _ball(rng, [0.7, 0.6, 0.9], 0.02, 80)]))
    pb.append(_ball(rng, [0.1, 0.6, 0.6], 0.02, n))
    field_ids = np.arange(nxt, nxt + 3000, dtype=np.uint64)
    fa = np.column_stack([rng.uniform(0.85, 0.95, 3000), rng.uniform(0.5, 0.6, 3000), rng.uniform(0.2, 0.3, 3000)])
    ids = np.concatenate(ids + [field_ids])
    pa = np.concatenate(pa + [fa])
    pb = np.concatenate(pb + [fa])
    # the later snapshot stores its particles in another order (as Gadget does): only the IDs link them
    perm = rng.permutation(ids.size)
    return SimpleNamespace(ids=ids, pa=pa, pb=pb, ids_b=ids[perm], pb_perm=pb[perm], ca=ca, ra=ra, cb=cb, rb=rb)


def t_link_classes():
    from dtfelib import lagrangian as L
    u = _universe()
    a = L.void_members([(u.ids[:2500], u.pa[:2500]), (u.ids[2500:], u.pa[2500:])], u.ca, u.ra)   # two chunks
    b = L.void_members([(u.ids_b, u.pb_perm)], u.cb, u.rb)
    assert a.n_members.tolist() == [400] * 5, a.n_members           # the corner void found across the wrap
    assert b.n_members.tolist() == [400, 800, 160, 160], b.n_members
    assert np.all(np.diff(a.ids.astype(np.int64)) >= 0)
    lk = L.link(a, b)
    fate = L.classify(lk)
    want = ["survive", "merge", "merge", "split", "vanish"]
    assert [L.CLASSES[c] for c in fate["cls"]] == want, [L.CLASSES[c] for c in fate["cls"]]
    assert fate["desc"].tolist()[:3] == [0, 1, 1] and fate["desc"][4] == -1, fate["desc"]
    assert np.isclose(fate["f_max"][3], 0.4) and np.isclose(fate["f_second"][3], 0.4), fate
    assert fate["n_split"].tolist() == [1, 1, 1, 2, 0], fate["n_split"]
    # the blocked join gives the same table as one block
    lk1 = L.link(a, b, block=7)
    assert np.array_equal(lk1.count, lk.count) and np.array_equal(lk1.desc, lk.desc)
    # the backward link: B1's main progenitor is A1 or A2 (400 each), B0's A0 with all of its members
    prog, share = L.main_progenitor(lk)
    assert prog[0] == 0 and np.isclose(share[0], 1.0) and prog[1] in (1, 2) and np.isclose(share[1], 0.5), (prog, share)
    # thresholds are module constants a caller can move: at 45% the 40/40 split is a vanish
    f2 = L.classify(lk, split_frac=0.45)
    assert L.CLASSES[f2["cls"][3]] == "vanish"
    # chain: B -> C where B0 survives and B1 vanishes: only A0 reaches C
    c_fate = {"cls": np.array([0, 3, 3, 3], np.int8), "desc": np.array([0, -1, -1, -1])}
    assert L.chain_survival([fate, c_fate]).tolist() == [True, False, False, False, False]
    # enclosed density: void members thinner than the mean
    assert np.all(np.isfinite(a.enclosed_delta()))


def t_exclusive_members():
    """--exclusive: overlapping spheres share no particle; each goes to the nearest centre in R_v units, per chunk."""
    from dtfelib import lagrangian as L
    rng = np.random.default_rng(5)
    n = 60000
    frac = rng.random((n, 3))
    ids = rng.permutation(n).astype(np.uint64) + 11
    c = np.array([[0.5, 0.5, 0.5], [0.6, 0.5, 0.5], [0.02, 0.5, 0.5], [0.97, 0.5, 0.5]])   # two pairs overlap
    r = np.array([0.1, 0.06, 0.06, 0.05])
    chunks = [(ids[:25000], frac[:25000]), (ids[25000:], frac[25000:])]
    m = L.void_members(chunks, c, r)
    e = L.void_members(chunks, c, r, exclusive=True)
    assert m.ids.size > np.unique(m.ids).size, "the spheres must overlap for this test"
    assert e.ids.size == np.unique(e.ids).size == np.unique(m.ids).size, (e.ids.size, np.unique(m.ids).size)
    assert np.all(np.diff(e.ids.astype(np.int64)) > 0)
    # the brute-force nearest-in-R_v sphere of every member
    pos = dict(zip(ids.tolist(), frac))
    for pid, v in zip(e.ids[::7].tolist(), e.void[::7].tolist()):
        d = pos[pid] - c
        d -= np.rint(d)
        q = np.linalg.norm(d, axis=1) / r
        assert q[v] <= 1 and v == int(np.argmin(q)), (pid, v, q)
    assert e.n_members.sum() == e.ids.size and np.all(e.n_members <= m.n_members)
    # a POOL: two unfollowed spheres after the followed ones, one overlapping a followed sphere. The
    # followed voids' share of the pool is exactly their own partition (a particle in a followed sphere stays there),
    # and without 'exclusive' exactly their own (overlapping) members
    pc = np.vstack([c, [[0.55, 0.5, 0.5], [0.3, 0.3, 0.3]]])
    pr = np.concatenate([r, [0.08, 0.05]])
    ep = L.void_members(chunks, pc, pr, exclusive=True, n_sample=4)
    es = ep.sample()
    assert ep.n_sample == 4 and es.n_members.size == 4 and np.array_equal(es.ids, e.ids) and np.array_equal(es.void, e.void)
    assert ep.ids.size == np.unique(ep.ids).size and ep.n_members[4] > 0 and ep.n_members[5] > 0
    mp = L.void_members(chunks, pc, pr, n_sample=4).sample()
    assert np.array_equal(mp.ids, m.ids) and np.array_equal(mp.void, m.void) and np.array_equal(mp.n_members, m.n_members)


def _lexsort_exclusive_ref(chunks, centres, radii, n_s):
    """The exclusive pairs as a lexsort makes them (every (id, q) pair of a chunk lexsorted, the
    first per id kept): ([(ids, void, chunk index) per chunk], particles whose nearest q is tied between spheres)
    -- the reference of the running best."""
    from scipy.spatial import cKDTree
    out, ties = [], 0
    for ids, frac in chunks:
        tree = cKDTree(np.mod(frac, 1.0), boxsize=1.0 + 1e-12, balanced_tree=False, compact_nodes=False)
        ci, cv, cq, cx = [], [], [], []
        for v in range(radii.size):
            idx = np.asarray(tree.query_ball_point(centres[v], radii[v]), dtype=np.int64)
            if not idx.size:
                continue
            d = frac[idx] - centres[v]
            d -= np.rint(d)
            q = np.sqrt(np.einsum("ij,ij->i", d, d)) / radii[v]
            ci.append(ids[idx]), cv.append(np.full(idx.size, v, np.int32)), cx.append(idx)
            cq.append((q + (2.0 if v >= n_s else 0.0)).astype(np.float32))
        ci, cv, cq, cx = map(np.concatenate, (ci, cv, cq, cx))
        o = np.lexsort((cq, ci))
        first = np.ones(o.size, dtype=bool)
        first[1:] = ci[o][1:] != ci[o][:-1]
        out.append((ci[o][first], cv[o][first], cx[o][first]))
        tie = np.zeros(o.size, dtype=bool)
        tie[1:] = ~first[1:] & first[:-1] & (cq[o][1:] == cq[o][:-1])
        ties += int(tie.sum())
    return out, ties


def t_exclusive_running_best():
    """The exclusive partition is a running per-particle best, not a lexsort of every pair -- the same
    (id, void, index) as the lexsort, ties (equal q: the lower sphere) included, with and without the pool rule."""
    from dtfelib import lagrangian as L
    rng = np.random.default_rng(11)
    n = 40000
    frac = np.round(rng.random((n, 3)) * 32) / 32          # a lattice: many particles at EQUAL q from two centres
    ids = (rng.permutation(5 * n)[:n] + 3).astype(np.uint64)
    c = np.round(rng.random((30, 3)) * 32) / 32
    r = rng.choice([0.09, 0.125, 0.2], 30)                  # repeated radii: exact ties of q
    chunks = [(ids[:17000], frac[:17000]), (ids[17000:], frac[17000:])]
    want, ties = _lexsort_exclusive_ref(chunks, c, r, 18)
    assert ties > 20, f"the lattice must make tied q's ({ties})"
    lexsort = np.lexsort
    def no_lexsort(*a, **k):
        raise AssertionError("the exclusive members lexsorted the chunk's pairs")
    np.lexsort = no_lexsort
    try:
        got = list(L._chunk_pairs(chunks, c, r, True, 18, with_index=True))
        plain = list(L._chunk_pairs(chunks, c, r, True, 18))
    finally:
        np.lexsort = lexsort
    assert len(got) == len(plain) == len(want) == 2
    for (wi, wv, wx), g, p in zip(want, got, plain):
        assert len(g[0]) == 1 and np.array_equal(g[0][0], wi) and np.array_equal(g[1][0], wv), "pairs differ"
        assert np.array_equal(g[3][0], wx) and np.array_equal(p[0][0], wi) and np.array_equal(p[1][0], wv)
        assert g[1][0].dtype == np.int32 and g[3][0].dtype == np.int64


def t_combined_reader():
    import h5py
    from dtfelib import lagrangian as L
    u = _universe(1)
    root = Path(_TMP) / "data"
    snapdir = root / "SYN-1" / "snapdir_050"
    snapdir.mkdir(parents=True)
    box = 35000.0 / 0.6774
    with h5py.File(snapdir / "combined_050.hdf5", "w") as f:
        f.create_group("Header").attrs["BoxSize"] = box
        f["PartType1/Coordinates"] = (u.pa * box).astype(np.float32)
        f["PartType1/ParticleIDs"] = u.ids
    p = L.combined_path("SYN-1", 50, root)
    chunks = list(L.iter_particles(p, chunk=777))
    assert len(chunks) == int(np.ceil(u.ids.size / 777))
    ids = np.concatenate([c[0] for c in chunks]); frac = np.concatenate([c[1] for c in chunks])
    assert ids.dtype == np.uint64 and np.array_equal(ids, u.ids)
    assert frac.min() >= 0 and frac.max() < 1 and np.allclose(frac, u.pa, atol=1e-6)
    m_file = L.void_members(p, u.ca, u.ra)
    m_mem = L.void_members([(u.ids, u.pa)], u.ca, u.ra)
    assert m_file.n_members.tolist() == m_mem.n_members.tolist() and m_file.n_particles == u.ids.size
    try:
        L.combined_path("SYN-1", 99, root)
        raise AssertionError("a missing combined snapshot must raise FileNotFoundError")
    except FileNotFoundError:
        pass


class _Chunks:
    """An in-memory 'combined snapshot': one (ids, frac) chunk, each pass over it counted as a read."""

    def __init__(self, owner, particles):
        self.owner, self.particles = owner, particles

    def __iter__(self):
        self.owner.reads += 1
        return iter([self.particles])


class FakeVoidSnapshot:
    """What lagrangian.VoidSnapshot gives the scripts, from synthetic particles and spheres; its member methods ARE
    VoidSnapshot's (the pool read once, the followed voids' share of it, the drops). 'extra_centres' / 'extra_radii':
    unfollowed minima of the descendant pool, after the followed ones as void_pool orders them."""

    def __init__(self, snap, particles, centres, radii, n=64, box=64.0, delta=None, extra_centres=None,
                 extra_radii=None):
        self.sim, self.snap, self.method, self.prefix = "SYN-1", snap, "ps", "ps_output."
        self.n, self.box, self.cell, self.exclusive = n, box, box / n, False
        self.centres, self.radii = np.asarray(centres, float), np.asarray(radii, float)
        ec = np.zeros((0, 3)) if extra_centres is None else np.asarray(extra_centres, float).reshape(-1, 3)
        er = np.zeros(0) if extra_radii is None else np.asarray(extra_radii, float)
        k, k_x = len(radii), er.size
        self.rows = np.arange(k) * 2 + 1                      # catalogue rows are not sample indices
        cat_n = 2 * (k + k_x) + 2
        self.pool_rows = np.concatenate([self.rows, np.arange(k, k + k_x) * 2 + 1])
        self.pool_centres, self.pool_radii = np.vstack([self.centres, ec]), np.concatenate([self.radii, er])
        coords = np.zeros((cat_n, 3), int)
        coords[self.pool_rows] = np.floor(self.pool_centres * n).astype(int)
        dv = np.zeros(cat_n)
        dv[self.rows] = -0.9 + 0.1 * np.arange(k) if delta is None else delta
        sa = np.ones((cat_n, 3))
        ori = np.tile(np.eye(3), (cat_n, 1, 1))
        rvc = np.zeros(cat_n)
        rvc[self.pool_rows] = self.pool_radii * n
        self.cat = {"coords": coords, "delta_values": dv, "semi_axes": sa, "orientations": ori,
                    "well_resolved": np.ones(cat_n, bool), "bbks_params": np.full((cat_n, 2), 0.1),
                    "eigenvalues": np.tile([0.5, 0.6, 0.7], (cat_n, 1)), "eigenvectors": ori,
                    "r_v_cells": rvc, "cell_mpc": self.cell}
        self._particles = particles
        self.path = _Chunks(self, particles)                  # void_members reads an iterable of (ids, frac) chunks
        self.n_particles = particles[0].size
        self.reads = 0
        self._members = None
        self._prog = None

    def members(self):
        from dtfelib import lagrangian as L
        return L.VoidSnapshot.members(self)

    def prog_members(self):
        from dtfelib import lagrangian as L
        return L.VoidSnapshot.prog_members(self)

    def stream_links(self, progs, keep_prog=True):
        from dtfelib import lagrangian as L
        return L.VoidSnapshot.stream_links(self, progs, keep_prog)

    def drop_members(self, keep_prog=False):
        from dtfelib import lagrangian as L
        return L.VoidSnapshot.drop_members(self, keep_prog=True) if keep_prog else L.VoidSnapshot.drop_members(self)


def _fake_ladder():
    u = _universe(2)
    a = FakeVoidSnapshot(33, (u.ids, u.pa), u.ca, u.ra)
    b = FakeVoidSnapshot(67, (u.ids_b, u.pb_perm), u.cb, u.rb)
    c = FakeVoidSnapshot(99, (u.ids_b, u.pb_perm), u.cb[:2], u.rb[:2])     # B0, B1 live on; B2, B3 gone
    return a, b, c


def t_fate_census_script():
    import config
    from dtfelib import lagrangian as L
    import plot_void_fate as F
    saved = config.CACHE_DIR
    config.CACHE_DIR = Path(_TMP) / "cache"
    try:
        a, b, c = _fake_ladder()
        args = SimpleNamespace(sim="SYN-1", ref_snap=33)
        intervals, direct = F.census([a, b, c], args, "")
        assert [F.class_counts(f) for *_x, f in intervals] == [
            {"survive": 1, "merge": 2, "split": 1, "vanish": 1, "unresolved": 0},
            {"survive": 2, "merge": 0, "split": 0, "vanish": 2, "unresolved": 0}]
        assert direct is not None and F.class_counts(direct[3])["merge"] == 2
        cached = sorted(p.name for p in L.link_cache_path(a, b).parent.glob("SYN-1_ps_fate_*.npz"))
        assert cached == ["SYN-1_ps_fate_033_067.npz", "SYN-1_ps_fate_033_099.npz", "SYN-1_ps_fate_067_099.npz"], cached
        reads = a.reads
        F.census([a, b, c], args, "")                         # all three links from the cache: no particle read
        assert a.reads == reads, "a cached link must not read the particles again"
        c2 = FakeVoidSnapshot(99, c._particles, c.centres, c.radii * 1.1)
        F.census([a, b, c2], args, "")                        # other voids at 99: those links are rebuilt
        assert c2.reads == 1
        chain = F.chained_survival(intervals)
        assert chain["snaps"].tolist() == [33, 67] and np.allclose(chain["frac"], [0.6, 0.5]), chain
        ztab = {33: 2.0, 67: 0.5, 99: 0.0}
        out = Path(_TMP) / "fate_out"
        out.mkdir()
        F.plot_census(intervals, direct, chain, ztab, SimpleNamespace(sim="SYN-1"), out / "fate_census.png")
        F.write_table(intervals, ztab, out / "fate_table.csv")
        assert (out / "fate_census.png").stat().st_size > 10000
        lines = [ln for ln in (out / "fate_table.csv").read_text().splitlines() if not ln.startswith("#")]
        assert lines[0].startswith("snap_a,z_a,snap_b") and len(lines) == 1 + 5 + 4, len(lines)
        row = dict(zip(lines[0].split(","), lines[2].split(",")))      # A1: merges into B1 (catalogue row 3)
        assert row["fate"] == "merge" and row["desc_row"] == "3" and row["row"] == "3", row
        assert float(row["enclosed_delta"]) > 0       # the synthetic voids are dense balls: only the sign convention
    finally:
        config.CACHE_DIR = saved


def _pool_universe(seed=7):
    """Four followed progenitors (A) and a later snapshot (B) with two FOLLOWED voids S0, S1 and two UNFOLLOWED
    minima of its pool, U0 and U1 (U1 a sub-minimum overlapping S1): P0 -> inside S0 (survive); P1 -> inside U0
    only (a census whose descendants are the followed voids alone would call it a vanish; it
    lives on, 'unresolved'); P2 -> into no sphere at all (vanish: no void receives the thresholds); P3 -> wholly in
    U1 and most of it in S1 (the followed void is preferred: survive)."""
    rng = np.random.default_rng(seed)
    ca = np.array([[0.2, 0.2, 0.2], [0.4, 0.6, 0.2], [0.6, 0.4, 0.8], [0.8, 0.8, 0.8]])
    ra = np.full(4, 0.04)
    cs, rs = np.array([[0.3, 0.3, 0.3], [0.7, 0.2, 0.5]]), np.array([0.06, 0.05])
    cu, ru = np.array([[0.7, 0.7, 0.7], [0.74, 0.2, 0.5]]), np.array([0.05, 0.05])
    n = 500
    ids = np.arange(100, 100 + 4 * n, dtype=np.uint64)
    pa = np.concatenate([_ball(rng, ca[i], 0.97 * ra[i], n) for i in range(4)])
    pb = np.concatenate([_ball(rng, cs[0], 0.04, n), _ball(rng, cu[0], 0.04, n), _ball(rng, [0.1, 0.8, 0.5], 0.02, n),
                         _ball(rng, [0.745, 0.2, 0.5], 0.015, n)])
    return SimpleNamespace(ids=ids, pa=pa, pb=pb, ca=ca, ra=ra, cs=cs, rs=rs, cu=cu, ru=ru)


def t_pool_unresolved():
    """The descendants are every minimum of the next catalogue, not the followed sample alone."""
    import config
    from dtfelib import lagrangian as L
    import plot_void_fate as F
    u = _pool_universe()
    a = FakeVoidSnapshot(50, (u.ids, u.pa), u.ca, u.ra)
    b = FakeVoidSnapshot(67, (u.ids, u.pb), u.cs, u.rs, extra_centres=u.cu, extra_radii=u.ru)
    saved = config.CACHE_DIR
    config.CACHE_DIR = Path(_TMP) / "cache_pool"
    try:
        intervals, direct = F.census([a, b], SimpleNamespace(sim="SYN-1", ref_snap=33), "")
        assert direct is None and a.reads == 1 and b.reads == 1
        fate = intervals[0][3]
        got = [L.CLASSES[c] for c in fate["cls"]]
        # P1's matter sits in an unfollowed minimum: NOT a vanish
        assert got[1] != "vanish", f"matter in an unfollowed minimum counted as vanished: {got}"
        assert got == ["survive", "unresolved", "vanish", "survive"], got
        assert fate["desc"].tolist()[:2] == [0, 2] and fate["desc"][2] == -1, fate["desc"]
        assert fate["in_sample"].tolist() == [True, False, False, True], fate["in_sample"]
        # P3: the overlapping unfollowed U1 holds all of it, the followed S1 most: S1 is the main descendant
        assert fate["desc"][3] == 1 and np.isclose(fate["f_max"][3], 1.0) and 0.5 <= fate["f_sample"][3] < 1.0, fate
        assert np.isclose(fate["f_desc"][3], fate["f_sample"][3]) and fate["n_main"].tolist() == [1, 1, 0, 1]
        # the contrast: the same matter against the followed voids alone vanishes
        old = L.classify(L.link(a.prog_members(), L.void_members([(u.ids, u.pb)], u.cs, u.rs)))
        assert L.CLASSES[old["cls"][1]] == "vanish" and L.CLASSES[old["cls"][2]] == "vanish"
        # the chain follows FOLLOWED main descendants only: P1's ends at U0 (pool index 2, no fate of its own)
        nxt = {"cls": np.zeros(2, np.int8), "desc": np.array([0, 1]), "in_sample": np.array([True, True])}
        assert L.chain_survival([fate, nxt]).tolist() == [True, False, False, True]
        assert np.allclose(F.chained_survival(intervals)["frac"], [0.5])
        # the table: desc_row is a catalogue row of the POOL, with the followed flag
        out = Path(_TMP) / "fate_pool"
        out.mkdir()
        F.write_table(intervals, {50: 1.0, 67: 0.5}, out / "fate_table.csv")
        F.plot_census(intervals, None, F.chained_survival(intervals), {50: 1.0, 67: 0.5}, SimpleNamespace(sim="SYN-1"),
                      out / "fate_census.png")
        lines = [ln for ln in (out / "fate_table.csv").read_text().splitlines() if not ln.startswith("#")]
        rows = [dict(zip(lines[0].split(","), ln.split(","))) for ln in lines[1:]]
        assert [r["fate"] for r in rows] == got and [r["desc_in_sample"] for r in rows] == ["1", "0", "-1", "1"]
        assert rows[1]["desc_row"] == str(int(b.pool_rows[2])) and int(rows[1]["desc_row"]) not in b.rows.tolist()
        # the link cache: the current version is served; a version-1 cache (descendants = the followed voids) is not
        p = L.link_cache_path(a, b)
        with np.load(p) as z:
            assert int(z["link_version"]) == L.LINK_VERSION == 2 and int(z["n_desc_sample"]) == 2
            arrays = {k: z[k] for k in z.files}
        reads = b.reads
        F.census([a, b], SimpleNamespace(sim="SYN-1", ref_snap=33), "")
        assert b.reads == reads, "a version-2 cache must be served"
        v1 = {k: v for k, v in arrays.items() if k not in ("link_version", "n_desc_sample")}
        np.savez(p, **v1)
        msgs = []
        lk = L.cached_link(a, b, log=msgs.append)
        assert b.reads == reads + 1 and any("version 1" in m for m in msgs), msgs
        assert [L.CLASSES[c] for c in L.classify(lk)["cls"]] == got
        with np.load(p) as z:
            assert int(z["link_version"]) == 2
    finally:
        config.CACHE_DIR = saved


def t_void_pool():
    """void_pool: the followed rows first with their R_v, then every other row with a usable radius:
    min(R_v, R_eff), R_eff without R_v, R_v without R_eff (not well resolved); a row with neither is left out."""
    from dtfelib import lagrangian as L
    n = 64
    coords = np.array([[1, 1, 1], [10, 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40], [50, 50, 50]])
    rv = np.array([5.0, 6.0, np.nan, 7.0, np.nan, 3.0])
    resolved = np.array([True, True, True, False, False, True])
    semi = np.array([[2.0] * 3, [4.0] * 3, [3.0] * 3, [1.0] * 3, [1.0] * 3, [1.0] * 3])  # R_eff in Mpc, cell = 0.5
    cat = {"coords": coords, "r_v_cells": rv, "well_resolved": resolved, "semi_axes": semi, "cell_mpc": 0.5,
           "delta_values": -0.5 * np.ones(6)}
    rows, r = L.void_pool(cat, n, np.array([5, 0]))
    assert rows.tolist() == [5, 0, 1, 2, 3], rows
    assert np.allclose(r, [3.0, 5.0, 6.0, 6.0, 7.0]), r      # followed: R_v; row 1 min(6, 8); 2: R_eff 6; 3: R_v
    c, rad = L.void_spheres(cat, n, rows, r)
    assert np.allclose(rad, r / n) and np.allclose(c[0], (50.5) / n)


def t_membership_cap():
    """The membership spheres are capped at FATE_MAX_SIGMA (15) smoothing lengths -- the R_v search reaches
    24, the census keeps the spheres its numbers come from: a followed void and a pool minimum whose
    radius exceeds the cap get the capped sphere (an explicit sigma, or config's), one below it keeps its radius, the
    particles between the cap and R_v are no members, the catalogue's R_v is untouched, and a link cache made with an
    uncapped sphere is relinked while one the cap does not change is served."""
    import config
    from dtfelib import lagrangian as L
    n, sig = 128, 2.0
    cap = 30.0                                                       # 15 sigma of 2 cells
    assert L.FATE_MAX_SIGMA == 15.0 and L.membership_cap_cells(sig) == cap
    assert L.membership_cap_cells(0.0) == L.FATE_MAX_SIGMA           # sigma < 1 cell counts as one (the R_v search's)
    coords = np.array([[10, 10, 10], [64, 64, 64], [100, 30, 60], [30, 100, 90], [5, 70, 110]])
    # rows 0, 1 followed (R_v 40 > cap, 12 < cap); 2: R_v 50, R_eff 50; 3: R_eff 12 only; 4: R_v 35, R_eff 40
    rv = np.array([40.0, 12.0, 50.0, np.nan, 35.0])
    semi = np.array([[1.0] * 3, [1.0] * 3, [25.0] * 3, [6.0] * 3, [20.0] * 3])     # Mpc, cell 0.5: R_eff in cells x 2
    cat = {"coords": coords, "r_v_cells": rv.copy(), "well_resolved": np.ones(5, bool), "semi_axes": semi,
           "cell_mpc": 0.5, "delta_values": -0.5 * np.ones(5)}
    rows, r = L.void_pool(cat, n, np.array([0, 1]), sigma_cells=sig)
    assert rows.tolist() == [0, 1, 2, 3, 4], rows
    assert np.allclose(r, [cap, 12.0, cap, 12.0, cap]), f"the pool's radii are not capped at {cap} cells: {r}"
    c, rad = L.void_spheres(cat, n, rows[:2], r[:2])
    assert np.allclose(rad, [cap / n, 12.0 / n])
    saved = config.SMOOTHING_SIGMA_CELLS
    config.SMOOTHING_SIGMA_CELLS = sig                               # the default: the catalogue's served smoothing
    try:
        _rows, r_def = L.void_pool(cat, n, np.array([0, 1]))
        _c, rad_def = L.void_spheres(cat, n, [0, 1])                 # R_v spheres without radii: capped too
    finally:
        config.SMOOTHING_SIGMA_CELLS = saved
    assert np.array_equal(r_def, r) and np.allclose(rad_def, rad), (r_def, rad_def)
    assert np.array_equal(cat["r_v_cells"], rv, equal_nan=True), "the measured R_v must not change"
    # members: within the capped sphere only (20 cells in, 35 cells out, although R_v is 40)
    ids = np.array([5, 9], dtype=np.uint64)
    frac = np.mod(c[0] + np.array([[20.0, 0, 0], [35.0, 0, 0]]) / n, 1.0)
    m = L.void_members([(ids, frac)], c[:1], rad[:1])
    assert m.n_members.tolist() == [1] and m.ids.tolist() == [5], (m.n_members, m.ids)
    # the link caches compare the spheres: a link made with B1's uncapped sphere is relinked once it is capped, a
    # sphere the cap leaves alone keeps its cache
    saved_cache = config.CACHE_DIR
    config.CACHE_DIR = Path(_TMP) / "cache_cap"
    try:
        a, b, _c3 = _fake_ladder()
        L.cached_link(a, b, log=lambda m: None)
        assert b.reads == 1
        same = FakeVoidSnapshot(67, b._particles, b.centres, np.minimum(b.radii, 0.2))      # the cap binds nowhere
        msgs = []
        L.cached_link(a, same, log=msgs.append)
        assert same.reads == 0 and not msgs, msgs
        capped = FakeVoidSnapshot(67, b._particles, b.centres, np.minimum(b.radii, 0.06))  # B1: 0.08 -> 0.06
        L.cached_link(a, capped, log=msgs.append)
        assert capped.reads == 1 and any("other voids" in q for q in msgs), msgs
    finally:
        config.CACHE_DIR = saved_cache


def _link_arrays_equal(x, y):
    return all(np.array_equal(getattr(x, k), getattr(y, k)) and getattr(x, k).dtype == getattr(y, k).dtype
               for k in ("prog", "desc", "count", "n_prog_members", "n_desc_members")) and x.n_desc_sample == y.n_desc_sample


def t_stream_identical():
    """link_stream (the census's descendant side, never holding the pool's member pairs) gives the SAME
    links as link(a, void_members(pool)) -- prog, desc, count, members, dtypes -- for two progenitor sets at once,
    across chunks and the wrap, with and without 'exclusive'; the followed members it keeps are exactly
    void_members(pool).sample(); and its peak allocation is far below holding the pool (tracemalloc)."""
    import tracemalloc
    from dtfelib import lagrangian as L
    rng = np.random.default_rng(11)
    n = 120_000
    ids = (rng.permutation(n) * 3 + 7).astype(np.uint64)
    fa, fb = rng.random((n, 3)), rng.random((n, 3))
    # progenitors: 30 overlapping spheres (wrap included); the pool: 12 followed + 40 large unfollowed (each particle
    # in ~4 pool spheres: the pairs outnumber the particles, as on TNG100)
    ca, ra = rng.random((30, 3)), rng.uniform(0.05, 0.12, 30)
    cp = np.vstack([rng.random((12, 3)), rng.random((40, 3))])
    rp = np.concatenate([rng.uniform(0.05, 0.1, 12), rng.uniform(0.2, 0.3, 40)])
    chunks_a = [(ids[i:i + 17_000], fa[i:i + 17_000]) for i in range(0, n, 17_000)]
    chunks_b = [(ids[i:i + 23_000], fb[i:i + 23_000]) for i in range(0, n, 23_000)]
    for excl in (False, True):
        a1 = L.void_members(chunks_a, ca, ra, exclusive=excl)
        a2 = L.void_members(chunks_a, ca[:9], ra[:9] * 1.3, exclusive=excl)
        pool = L.void_members(chunks_b, cp, rp, exclusive=excl, n_sample=12)
        assert pool.ids.size > 2 * n or excl, pool.ids.size
        want = [L.link(a1, pool), L.link(a2, pool, block=5000)]
        got, sample = L.link_stream([a1, a2], chunks_b, cp, rp, exclusive=excl, n_sample=12, block=9_000)
        assert len(got) == 2 and all(_link_arrays_equal(g, w) for g, w in zip(got, want)), f"exclusive={excl}"
        saved, L.DENSE_MAX = L.DENSE_MAX, 0                 # the sparse accumulation (a huge k_a x k_b table)
        try:
            sparse, _s = L.link_stream([a1, a2], chunks_b, cp, rp, exclusive=excl, n_sample=12, keep_sample=False)
        finally:
            L.DENSE_MAX = saved
        assert all(_link_arrays_equal(g, w) for g, w in zip(sparse, want)), f"sparse, exclusive={excl}"
        assert sum(int(lk.count.size) for lk in got) > 100
        ref = pool.sample()
        assert np.array_equal(sample.ids, ref.ids) and np.array_equal(sample.void, ref.void), f"exclusive={excl}"
        assert np.array_equal(sample.n_members, ref.n_members) and sample.n_particles == ref.n_particles == n
        assert np.array_equal(sample.radii, ref.radii) and sample.n_sample == 12
        assert [L.CLASSES[c] for c in L.classify(got[0])["cls"]] == [L.CLASSES[c] for c in L.classify(want[0])["cls"]]
        got2, none = L.link_stream([a1], chunks_b, cp, rp, exclusive=excl, n_sample=12, keep_sample=False)
        assert none is None and _link_arrays_equal(got2[0], want[0])
        # chunks in worker threads change nothing, bit for bit (one worker: the serial path)
        for w in (1, 3, 8):
            g, smp = L.link_stream([a1, a2], chunks_b, cp, rp, exclusive=excl, n_sample=12, block=9_000, workers=w)
            assert all(_link_arrays_equal(x, y) for x, y in zip(g, want)), f"workers {w}, exclusive={excl}"
            assert np.array_equal(smp.ids, ref.ids) and np.array_equal(smp.void, ref.void) and smp.n_particles == n
        par = list(L._chunk_pairs(chunks_b, cp, rp, excl, 12, workers=3))
        ser = list(L._chunk_pairs(chunks_b, cp, rp, excl, 12, workers=1))
        assert len(par) == len(ser) == len(chunks_b) and all(
            len(x[0]) == len(y[0]) and all(np.array_equal(p, q) for p, q in zip(x[0] + x[1], y[0] + y[1]))
            for x, y in zip(par, ser))
    # the packed (id << 32 | void) in-place sort, and its fallback when an id needs more than 32 bits mid-way
    # (separate arrays + stable argsort): both the order of the stable argsort of the arrival order
    big = [(ids[i:i + 17_000] + (np.uint64(1) << np.uint64(40) if i >= 51_000 else np.uint64(0)), fb[i:i + 17_000])
           for i in range(0, n, 17_000)]
    for chunks in (chunks_b, big):
        got_i, got_v = [], []
        for c_i, c_v, _n in L._chunk_pairs(chunks, cp, rp, False, 12):
            got_i += c_i
            got_v += c_v
        ref_i, ref_v = np.concatenate(got_i), np.concatenate(got_v)
        o = np.argsort(ref_i, kind="stable")
        m = L.void_members(chunks, cp, rp, n_sample=12)
        assert np.array_equal(m.ids, ref_i[o]) and np.array_equal(m.void, ref_v[o]) and m.ids.dtype == np.uint64
    # memory: the pool's pairs (12 B each, plus their sort) are what holding the pool costs; the stream holds a chunk's
    # (15 chunks here; a real snapshot has 12-30 of 8e6 particles)
    a1 = L.void_members(chunks_a, ca, ra)
    small = [(ids[i:i + 8_000], fb[i:i + 8_000]) for i in range(0, n, 8_000)]
    tracemalloc.start()
    L.link(a1, L.void_members(small, cp, rp, n_sample=12))
    _c, peak_old = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    L.link_stream([a1], small, cp, rp, n_sample=12, keep_sample=False, workers=1)   # one chunk at a time
    _c, peak_new = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak_new < 0.5 * peak_old, (peak_new, peak_old)


def t_link_cache_r_v_key():
    """The link caches are keyed on both catalogues' R_v settings (their r_v_settings record): a
    changed R_v search rebuilds the link even when the spheres it recorded are the same; a cache without the record
    (made before it) is rebuilt once; an unchanged one is served."""
    import config
    from dtfelib import lagrangian as L
    import plot_void_fate as F
    saved = config.CACHE_DIR
    config.CACHE_DIR = Path(_TMP) / "cache_rv"
    try:
        a, b, _c = _fake_ladder()
        rec = np.array([0.2, 15.0, 128, 2, -0.8])
        a.r_v_settings, b.r_v_settings = rec, rec.copy()
        F.census([a, b], SimpleNamespace(sim="SYN-1", ref_snap=-1), "")
        p = L.link_cache_path(a, b)
        with np.load(p) as z:
            assert np.array_equal(z["prog_r_v_settings"], rec) and np.array_equal(z["desc_r_v_settings"], rec)
            arrays = {k: z[k] for k in z.files}
        msgs = []
        reads = b.reads
        L.cached_link(a, b, log=msgs.append)
        assert b.reads == reads and not msgs, ("an unchanged cache must be served", msgs)
        b.r_v_settings = np.array([0.2, 25.0, 128, 2, -0.8])          # the R_v search reaches further: same spheres
        lk = L.cached_link(a, b, log=msgs.append)
        assert b.reads == reads + 1, "a link made with other R_v settings must be rebuilt"
        assert any("other R_v settings" in m for m in msgs), msgs
        with np.load(p) as z:
            assert np.array_equal(z["desc_r_v_settings"], b.r_v_settings)
        assert [L.CLASSES[c] for c in L.classify(lk)["cls"]] == ["survive", "merge", "merge", "split", "vanish"]
        np.savez(p, **{k: v for k, v in arrays.items() if "r_v_settings" not in k})   # a cache from before the key
        b.r_v_settings = rec
        msgs.clear()
        L.cached_link(a, b, log=msgs.append)
        assert b.reads == reads + 2 and any("recorded their catalogues' R_v settings" in m for m in msgs), msgs
        # a catalogue's own record is the key; without one, the current settings (pipeline.r_v_settings)
        from dtfelib import pipeline
        assert np.array_equal(L.r_v_record({"r_v_settings": rec}), rec)
        assert np.array_equal(L.r_v_record({}), pipeline.r_v_settings())
    finally:
        config.CACHE_DIR = saved


# ---------------------------------------------------------------- the tracker on synthetic merger trees
class FakeTrees:
    """SubLink main branches of tracer subhalos drifting by a constant comoving velocity (positions ckpc/h);
    branches reach back to snap 'earliest' only (as real trees thin out at high z)."""

    def __init__(self, pos0, drift_per_snap, snaps, earliest, box):
        self.pos0, self.drift, self.snaps, self.earliest, self.box = pos0, drift_per_snap, snaps, earliest, box

    def main_branch(self, snap0, sid, fields):
        if sid >= len(self.pos0):
            raise KeyError(sid)
        s = np.array([q for q in sorted(self.snaps, reverse=True) if self.earliest <= q <= snap0])
        pos = np.mod(self.pos0[sid][None, :] - self.drift[None, :] * (snap0 - s)[:, None], self.box)
        return {"SnapNum": s, "SubhaloPos": pos}


def _tracking_catalog(centre_frac, n, delta, decoy_frac):
    """A two-void catalogue: the tracked void at centre_frac, a decoy elsewhere."""
    coords = np.floor(np.array([decoy_frac, centre_frac]) * n).astype(int)
    ori = np.tile(np.eye(3), (2, 1, 1))
    return {"coords": coords, "delta_values": np.array([-0.2, delta]), "well_resolved": np.array([True, True]),
            "deep": np.array([True, True]), "bbks_params": np.array([[0.1, 0.0], [0.2, 0.05]]),
            "semi_axes": np.array([[3.0, 2.5, 2.0], [4.0, 3.0, 2.0]]), "orientations": ori,
            "eigenvalues": np.array([[0.01, 0.02, 0.03], [0.004, 0.007, 0.016]]), "eigenvectors": ori,
            "r_v_cells": np.array([10.0, 14.0]), "cell_mpc": 51.668142899320934 / n}


def t_tracker_trees():
    import config
    from dtfelib import voids as V
    import plot_void_tracking as T
    n, box = 64, 35000.0
    snaps = [99, 67, 50, 33, 25, 17]
    drift = np.array([60.0, -30.0, 0.0])                       # ckpc/h per snapshot number
    c0 = np.array([0.5, 0.5, 0.5])
    rng = np.random.default_rng(3)
    pos0 = np.mod(c0 + rng.normal(scale=0.01, size=(40, 3)), 1.0) * box
    pos0 = np.vstack([pos0, [[0.05 * box, 0.05 * box, 0.05 * box]]])       # one far subhalo: never a tracer
    ts = FakeTrees(pos0, drift, snaps, earliest=33, box=box)
    pool_frac = V.wrap_frac(pos0 / box)
    ids = np.arange(len(pos0))
    # unit tests of the two tree-side helpers
    sel = V.select_tracers(pool_frac, c0, 0.05, max_tracers=10)
    assert sel.size == 10 and 40 not in sel
    d = np.linalg.norm(V.min_image(pool_frac[sel] - c0), axis=1)
    assert np.all(np.diff(d) >= 0), "select_tracers: nearest first"
    br = V.tracer_branches(ts, 99, [0, 1, 999], box)
    assert sorted(br) == [0, 1] and sorted(br[0]) == [33, 50, 67, 99]
    assert np.allclose(br[0][67], np.mod(pos0[0] / box - drift / box * 32, 1.0))
    # catalogues: the void moves with the tracers' drift (frozen below snap 33), its depth labels the snapshot
    catalogs = {}
    for s in snaps:
        cen = np.mod(c0 - drift / box * (99 - max(s, 33)), 1.0)
        catalogs[s] = (_tracking_catalog(cen, n, -0.5 - 0.001 * s, [0.1, 0.9, 0.2]), n)
    cat0 = catalogs[99][0]
    args = SimpleNamespace(snap=99, sel_factor=1.5, max_tracers=200, min_tracers=3, match_factor=0.75,
                           method="ps", sim="SYN-1", radius="r_v", link="trees")
    out = T.track_voids_trees(cat0, n, np.array([1]), catalogs, ts, pool_frac, ids, box, {s: 0.1 * (99 - s) for s in snaps}, args)
    assert len(out) == 1
    k, t, r_eff0, extra = out[0]
    assert t["snap"].tolist() == snaps
    assert np.allclose(t["delta_min"], [-0.5 - 0.001 * s for s in snaps]), t["delta_min"]   # the right void each time
    assert t["tree_identity"].tolist() == [True, True, True, True, False, False]
    assert t["n_tracers"][0] == 40 and t["n_tracers"][-1] == 0
    assert np.all(np.isfinite(t["r_v"])) and np.allclose(t["r_v"], 14.0 * 51.668142899320934 / n)
    assert np.all(t["match_dist_mpc"] < 0.75 * r_eff0)
    assert extra["catalog_index0"] == 1 and np.isclose(r_eff0, (4.0 * 3.0 * 2.0) ** (1 / 3))
    # too few tracers: skipped, not a crash
    args.min_tracers = 100
    assert T.track_voids_trees(cat0, n, np.array([1]), catalogs, ts, pool_frac, ids, box, {}, args) == []
    # the figures: one void, then a summary of many with bands, on log(1+z) axes; names by tag
    outdir = Path(_TMP) / "track"
    outdir.mkdir()
    args.tag = T._set_tag(args, ("ps", "ps_output."))
    assert args.tag == ""
    T.plot_void(t, 0, args, outdir, r_eff0)
    many = [(i, {**t, "r_v": t["r_v"] * (1 + 0.05 * i)}) for i in range(8)]
    T.plot_summary(many, args, outdir)
    assert (outdir / "void0_evolution.png").is_file() and (outdir / "voids_summary.png").is_file()
    s_, x, pct = T.summary_bands(many, "r_v")
    assert np.allclose(x, 1 + np.array([0.1 * (99 - s) for s in snaps])) and np.all(pct[0] <= pct[2])
    assert config.SHOW_TITLES in (True, False)


def t_tracker_particles():
    import plot_void_tracking as T
    a, b, c = _fake_ladder()
    from dtfelib import lagrangian as L
    links = [L.link(a.prog_members(), b.members()), L.link(b.prog_members(), c.members())]
    ztab = {33: 2.0, 67: 0.5, 99: 0.0}
    out = T.track_voids_particles([a, b, c], 5, ztab, links=links)
    assert len(out) == 2                                    # only two distinct voids at the reference snapshot
    k, t, r_eff0, extra = out[0]                            # the deepest at 99: B0 (row 1), from A0 (row 1)
    assert t["snap"].tolist() == [99, 67, 33] and t["catalog_row"].tolist() == [1, 1, 1], t["catalog_row"]
    assert np.isnan(t["link_frac"][0]) and np.allclose(t["link_frac"][1:], 1.0), t["link_frac"]
    k, t, _r, _e = out[1]                                   # B1 (row 3): half its matter from A1, half from A2
    assert t["catalog_row"][:2].tolist() == [3, 3] and t["catalog_row"][2] in (3, 5) and np.isclose(t["link_frac"][2], 0.5)
    args = SimpleNamespace(snap=99, method="ps", sim="SYN-1", radius="r_v", link="particles")
    args.tag = T._set_tag(args, ("ps", "ps_output."))
    assert args.tag == "_particles"
    outdir = Path(_TMP) / "track_p"
    outdir.mkdir()
    T.plot_void(t, 1, args, outdir, r_eff0)
    assert (outdir / "void1_evolution_particles.png").is_file()
    # --exclusive: partitioned members, its own '_exclusive' set; refused with the trees
    args.exclusive = True
    assert T._set_tag(args, ("ps", "ps_output.")) == "_particles_exclusive"
    assert T.parse_args(["--link", "particles", "--exclusive"]).exclusive
    import contextlib, io as _io
    with contextlib.redirect_stderr(_io.StringIO()):
        try:
            T.parse_args(["--exclusive"])
            raise AssertionError("--exclusive with the merger-tree link must be refused")
        except SystemExit as e:
            assert e.code == 2, e.code


def t_tracker_link_tag():
    """plot_void_tracking --link particles reads the census's link files: lagrangian.link_tag is the one
    rule ('' for PS-DTFE's grids, '_dtfe', '_<prefix>'), plot_void_fate.name_tag is it, and the tracker passes it to
    cached_link (without it a --prefix run would read and rewrite the default grids' links)."""
    import config
    import plot_void_fate as F
    import plot_void_tracking as T
    from dtfelib import lagrangian as L
    assert [L.link_tag("ps", "ps_output."), L.link_tag("dtfe", "output."), L.link_tag("ps", "ps_mw.")] == ["", "_dtfe", "_ps_mw"]
    assert F.name_tag("ps", "ps_mw.") == "_ps_mw" and F.name_tag("ps", "ps_output.") == ""
    saved = config.CACHE_DIR
    config.CACHE_DIR = Path(_TMP) / "cache_tag"
    try:
        a, b, c = _fake_ladder()
        for v in (a, b, c):
            v.prefix = "ps_mw."
        T.track_voids_particles([a, b, c], 2, {33: 2.0, 67: 0.5, 99: 0.0})
        names = sorted(p.name for p in (config.CACHE_DIR).rglob("*_fate_*.npz"))
        assert names == ["SYN-1_ps_mw_ps_fate_033_067.npz", "SYN-1_ps_mw_ps_fate_067_099.npz"], names
        # the census of the same set finds them: no particle read
        reads = [v.reads for v in (a, b, c)]
        F.census([a, b, c], SimpleNamespace(sim="SYN-1", ref_snap=-1), F.name_tag("ps", "ps_mw."))
        assert [v.reads for v in (a, b, c)] == reads, ([v.reads for v in (a, b, c)], reads)
    finally:
        config.CACHE_DIR = saved


def t_help_and_flags():
    """The fate flags are in each script's own help; the launcher text rule: the help has them, nothing else."""
    import subprocess
    for script, flags in (("plot_void_fate.py", ("--ref-snap", "--snaps", "--out", "--exclusive")),
                          ("plot_void_tracking.py", ("--link", "--exclusive"))):
        r = subprocess.run([sys.executable, str(ROOT / "python" / "plot" / script), "--help"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-500:]
        for fl in flags:
            assert fl in r.stdout, f"{script}: {fl} not in its help"


# ---------------------------------------------------------------- data-backed
def t_real_combined_chunk():
    from dtfelib import lagrangian as L
    try:
        p = L.combined_path("TNG50-3-Dark", 99, data_root())
    except FileNotFoundError as e:
        raise Skip(f"no TNG50-3-Dark combined snapshot: {e}")
    ids, frac = next(L.iter_particles(p, chunk=200_000))
    assert ids.dtype == np.uint64 and ids.size == 200_000 and np.unique(ids).size == ids.size
    # the file is in Subfind order (the first chunk at z = 0 is one halo, < 1 Mpc across): only the frame is checked
    assert frac.min() >= 0 and frac.max() < 1


def main():
    print("synthetic:")
    check("lagrangian: members across chunks and the periodic wrap, link by ID, survive/merge/split/vanish, "
          "main progenitor, chain, thresholds", t_link_classes)
    check("lagrangian: exclusive members (a partition: nearest centre in R_v units, across chunks and the wrap)",
          t_exclusive_members)
    check("exclusive members by a running per-particle best == the lexsort of every pair (ties: the lower "
          "sphere; the pool rule; the chunk index), no lexsort", t_exclusive_running_best)
    check("lagrangian: the combined-snapshot reader (chunks, fractions, members from file == in memory)", t_combined_reader)
    check("plot_void_fate: census classes, the link cache (reuse, rebuild on other voids), chained survival, "
          "figure, table", t_fate_census_script)
    check("lagrangian.void_pool: followed first with R_v, then min(R_v, R_eff) / R_eff / R_v, none left out",
          t_void_pool)
    check("membership spheres capped at FATE_MAX_SIGMA smoothing lengths (followed and pool; config's sigma "
          "by default; R_v untouched; no members beyond the cap; uncapped link caches relinked)", t_membership_cap)
    check("descendant pool: matter in an unfollowed minimum is 'unresolved', not 'vanish'; matter in no "
          "void vanishes; the followed void preferred; chains end at unfollowed voids; table; version-1 cache "
          "rebuilt", t_pool_unresolved)
    check("the streamed link (pool never held) == link(a, void_members(pool)) for two progenitor sets, "
          "exclusive or not, in worker threads or not; kept followed members == .sample(); "
          "far lower peak allocation", t_stream_identical)
    check("link caches keyed on both catalogues' R_v settings (changed -> relinked, unrecorded -> "
          "relinked once, unchanged -> served)", t_link_cache_r_v_key)
    check("plot_void_tracking: tracker on synthetic merger trees (select_tracers, tracer_branches, frozen identity, "
          "the right void per snapshot, skip), figures + bands on log(1+z)", t_tracker_trees)
    check("plot_void_tracking --link particles: the main-progenitor walk, link share, the _particles tag, --exclusive "
          "(its tag, refused with the trees)", t_tracker_particles)
    check("plot_void_tracking --link particles shares the census's link tag (a --prefix set's own link "
          "files, the census then reads them without a particle read)", t_tracker_link_tag)
    check("the fate flags in each script's help", t_help_and_flags)
    print("data-backed:")
    check("lagrangian.iter_particles on the real TNG50-3-Dark combined snapshot (one chunk)", t_real_combined_chunk)
    print("-" * 60)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
