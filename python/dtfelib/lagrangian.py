"""Lagrangian (particle-ID) linking of voids between snapshots: the FATE of every void (2026-10-11).

The merger-tree tracers of voids.py follow a void through the few subhalos near it, which a DM-only void barely
holds at z > 3. Here the void is followed through its own MATTER instead: the particles inside it at one snapshot
are found again, by ParticleID, at the next, and the voids that hold them there are its descendants.

    from dtfelib import lagrangian as L
    a = L.void_members(L.combined_path(sim, 50), *L.void_spheres(cat50, 512, rows50))   # MemberSet (followed)
    pool, r_cells = L.void_pool(cat67, 512, rows67)                                       # followed first, then the rest
    b = L.void_members(L.combined_path(sim, 67), *L.void_spheres(cat67, 512, pool, r_cells), n_sample=rows67.size)
    lk = L.link(a, b)                     # sparse contingency: shared particle counts (prog, desc, count)
    fate = L.classify(lk)                 # per progenitor: class, main descendant, shared fractions

DEFINITIONS (the wave's shared contract C6; each threshold is a module constant below):
  * the voids FOLLOWED (the progenitors, and the only voids a chain continues through) are the DISTINCT
    well-resolved voids with a measured radius R_v (catalog.distinct_mask, the profile / abundance sample):
    neighbouring minima of one void are not linked as voids of their own;
  * the DESCENDANTS are drawn from the whole POOL of the next catalogue (2026-10-11, void_pool): the followed voids
    with their R_v spheres, then EVERY other minimum with a usable radius -- min(R_v, R_eff), R_eff where it has no
    R_v and R_v where it has no R_eff (not well resolved), the claim rule of catalog.distinct_mask. At late times the
    followed sample is a small part of the minima (TNG100-3-Dark z = 0: 77 of 495; 416 have a usable radius), and with
    the sample as the only descendants a void whose matter sits in an unfollowed minimum was counted as VANISHED,
    mixing real dissolution with dropping out of the sample (the census's title question);
  * a void's MEMBERS are the particles within its radius of its centre (the centre of its minimum's grid cell,
    periodic); overlapping spheres may share a particle, so a progenitor's shared fractions can sum above 1;
  * SURVIVE: >= SURVIVE_FRAC (50%) of its members end in one descendant void, which no other progenitor also gets
    >= 50% of its members into; MERGE: two or more progenitors pass >= 50% to the same descendant; SPLIT: no
    descendant gets 50%, but two or more get >= SPLIT_FRAC (25%) each; UNRESOLVED: it would survive or merge, but its
    main descendant is not followed (a minimum that is not distinct, not well resolved or without an R_v); VANISH:
    NO void of the pool receives these thresholds (collapsed, crushed between walls, or spread thinner than 25% over
    every minimum). The main descendant is the followed void with the largest share when one holds >= 50% (an
    unfollowed minimum overlapping it -- often a sub-minimum of the same void -- may hold more: the matter is still in
    the followed void), else the pool void with the largest share. CLASSES keeps the first four codes of before.
  CAVEAT (measured on TNG50-3-Dark, 2026-10-11): at z >= 2 the R_v spheres of the distinct voids overlap heavily (the
  member pairs are 1.2-1.8x the particles), so two overlapping progenitors that share their matter both pass 50% to
  one descendant: MERGE is over-counted there against a partition of the matter. EXCLUSIVE members
  (void_members(..., exclusive=True); plot_void_fate.py --exclusive) give each particle to the sphere whose centre is
  nearest in units of its R_v: a partition, not C6's literal definition, so it is an option. Of the pool, a particle
  inside any followed sphere goes to a followed void (so the progenitors' members are the same as without the pool);
  only the matter outside every followed sphere is shared out among the other minima.

MEMORY: a combined snapshot holds the whole box (TNG50-3-Dark: 1.57e8 particles, 3.1 GB of IDs and positions); it is
read in CHUNK-particle slices (h5py, float32 positions), each sorted into a periodic cKDTree only for the time its
spheres are queried. What remains is the member pairs (12 bytes each, plus their sort): at high z nearly every
particle is a member, up to 2.9e8 pairs on TNG50-3-Dark, ~10 GB per snapshot while it is sorted; the join of two
snapshots runs in blocks (link's 'block'). A census of TNG50-3-Dark peaked at 29 GB with a one-block join. The
descendant pool (2026-10-11) adds the unfollowed minima's spheres -- the ungated ones are large (median R_v ~70 cells
of 512 on TNG100-3-Dark) -- so a snapshot's pool holds a few times its followed voids' pairs; the census keeps only
the progenitors' share of a snapshot once it has served as the descendant side (VoidSnapshot.drop_members).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from .cli import sim_dir

SURVIVE_FRAC = 0.5      # shared fraction that makes a descendant THE descendant (survive / merge)
SPLIT_FRAC = 0.25       # shared fraction each of >= 2 descendants needs for a split
# 'unresolved' appended (2026-10-11): the codes 0-3 of the links' first census keep their meaning
CLASSES = ("survive", "merge", "split", "vanish", "unresolved")
CLASS_CODE = {c: i for i, c in enumerate(CLASSES)}
# the cached links' format: 2 = descendants from the whole pool (2026-10-11); a version-1 link (descendants = the
# followed sample only) counted the matter in unfollowed minima as vanished, so it is never served
LINK_VERSION = 2
CHUNK = int(os.environ.get("DTFE_FATE_CHUNK", 8_000_000))   # particles per read slice (8e6: ~0.6 GB with its tree)


def combined_path(sim: str, snap: int, root=None) -> Path:
    """The merged snapshot (python/tools/merge_HDF5.py) of 'sim' at 'snap'; FileNotFoundError when absent."""
    p = sim_dir(sim, root) / f"snapdir_{int(snap):03d}" / f"combined_{int(snap):03d}.hdf5"
    if not p.is_file():
        raise FileNotFoundError(f"no combined snapshot {p}")
    return p


def iter_particles(path, chunk: int = CHUNK):
    """(ParticleIDs uint64, positions in FRACTIONAL box units [0,1) float64 (m,3)) per chunk of PartType1. The box is
    the header's BoxSize, in the coordinates' own units (h-free ckpc for a merged file), so the fractions are the
    void catalogue's frame whatever the unit convention."""
    with h5py.File(path, "r") as f:
        box = float(f["Header"].attrs["BoxSize"])
        pos, ids = f["PartType1/Coordinates"], f["PartType1/ParticleIDs"]
        for lo in range(0, ids.shape[0], chunk):
            hi = min(lo + chunk, ids.shape[0])
            frac = np.asarray(pos[lo:hi], dtype=np.float64) / box
            yield np.asarray(ids[lo:hi], dtype=np.uint64), np.mod(frac, 1.0)


def void_sample(cat, n_grid: int) -> np.ndarray:
    """Catalogue rows of the voids that are linked: distinct, well resolved, with a measured R_v (sorted)."""
    from . import catalog                         # pipeline-heavy: imported only when a real catalogue is used
    rv = catalog.r_v_mpc(cat)
    resolved = np.asarray(cat["well_resolved"]).astype(bool) & np.isfinite(rv)
    return np.nonzero(catalog.distinct_mask(cat, resolved, int(n_grid)))[0]


def void_pool(cat, n_grid: int, sample_rows) -> tuple[np.ndarray, np.ndarray]:
    """(catalogue rows, radii in CELLS) of the DESCENDANT pool (2026-10-11): the followed sample first, in its order and
    with its R_v (so pool index == sample index for the first len(sample_rows), and a followed void's sphere is the
    same as progenitor and as descendant), then every other row with a usable radius, ascending: min(R_v, R_eff), R_eff
    where it has no R_v (catalog.distinct_mask's claim rule) and R_v where it has no R_eff -- a minimum the ellipsoid
    gate dropped (its curvature length, not its R_v, is large: ~78% of the z = 0 minima) is a void the matter can be
    in. A row with neither radius cannot hold members and is left out."""
    from . import catalog, pipeline
    sample_rows = np.asarray(sample_rows, dtype=np.int64)
    rv = catalog.r_v_cells(cat)
    cell = pipeline.catalog_cell_mpc(cat)
    r_eff = catalog.r_eff_mpc(cat) / cell if cell else np.full(rv.size, np.nan)
    claim = np.fmin(rv, r_eff)                      # fmin: the finite one where only one exists
    usable = np.isfinite(claim) & (claim > 0)
    usable[sample_rows] = False
    rest = np.nonzero(usable)[0]
    return (np.concatenate([sample_rows, rest]),
            np.concatenate([np.asarray(cat["r_v_cells"], dtype=np.float64)[sample_rows], claim[rest]]))


def void_spheres(cat, n_grid: int, rows, radii_cells=None) -> tuple[np.ndarray, np.ndarray]:
    """(centres (k,3), radii (k,)) of catalogue rows in fractional box units: the minimum's cell CENTRE and R_v (or
    'radii_cells', in cells: void_pool's radii)."""
    rows = np.asarray(rows, dtype=np.int64)
    centres = (np.asarray(cat["coords"], dtype=np.float64)[rows] + 0.5) / float(n_grid)
    radii = np.asarray(cat["r_v_cells"], dtype=np.float64)[rows] if radii_cells is None else \
        np.asarray(radii_cells, dtype=np.float64)
    return np.mod(centres, 1.0), radii / float(n_grid)


@dataclass
class MemberSet:
    """The member particles of k voids: (ids, void) pairs sorted by id (a particle in two spheres appears twice),
    the member count per void and the particles read (for the mean density inside each sphere). 'n_sample'
    (2026-10-11): the first n_sample voids are the FOLLOWED ones of a pool (void_pool), None = all of them."""
    ids: np.ndarray          # uint64, sorted
    void: np.ndarray         # int32, the void (0..k-1, in the order of the spheres) of each pair
    n_members: np.ndarray    # int64 (k,)
    n_particles: int
    radii: np.ndarray        # fractional R_v (k,)
    n_sample: int | None = None

    def sample(self) -> "MemberSet":
        """The members of the followed voids alone (the progenitor side of a link): the pairs of voids < n_sample,
        still sorted by id -- the same set void_members gives for the followed spheres alone."""
        k = self.n_members.size if self.n_sample is None else int(self.n_sample)
        if k >= self.n_members.size:
            return self
        m = self.void < k
        return MemberSet(self.ids[m], self.void[m], self.n_members[:k].copy(), self.n_particles, self.radii[:k], k)

    def enclosed_delta(self) -> np.ndarray:
        """Mean density contrast of the matter inside each sphere: members / (N x 4/3 pi R^3) - 1 (a void: < 0)."""
        vol = 4.0 / 3.0 * np.pi * self.radii ** 3
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.n_members / (float(self.n_particles) * vol) - 1.0


def void_members(particles, centres, radii, exclusive: bool = False, n_sample: int | None = None) -> MemberSet:
    """Members (C6: within R_v of the centre, periodic) of every sphere, from an iterable of (ids, frac) chunks --
    iter_particles(path), or a single in-memory (ids, frac) pair in a list. exclusive (2026-10-11): a particle
    inside several spheres belongs only to the one whose centre is nearest in units of its R_v -- a partition of
    the matter, so overlapping progenitors cannot both pass 50% of the same particles (EXCLUSIVE below). 'n_sample'
    (2026-10-11): the first n_sample spheres are the followed voids of a pool; under 'exclusive' a particle inside any
    of them goes to the nearest of THOSE, so MemberSet.sample() is exactly the followed spheres' own partition."""
    if isinstance(particles, (str, Path)):
        particles = iter_particles(particles)
    centres = np.mod(np.asarray(centres, dtype=np.float64).reshape(-1, 3), 1.0)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    k = radii.size
    n_s = k if n_sample is None else int(n_sample)
    got_ids, got_void, n_part = [], [], 0
    for ids, frac in particles:
        n_part += ids.size
        if not k or not ids.size:
            continue
        # unbalanced, non-compact: 3-4x faster to build, and each tree serves only k ball queries
        tree = cKDTree(np.mod(frac, 1.0), boxsize=1.0 + 1e-12, balanced_tree=False, compact_nodes=False)
        c_ids, c_void, c_q = [], [], []
        for v in range(k):
            if not radii[v] > 0:
                continue
            idx = np.asarray(tree.query_ball_point(centres[v], radii[v]), dtype=np.int64)
            if idx.size:
                c_ids.append(ids[idx])
                c_void.append(np.full(idx.size, v, dtype=np.int32))
                if exclusive:            # distance in units of this void's R_v
                    d = frac[idx] - centres[v]
                    d -= np.rint(d)
                    q = np.sqrt(np.einsum("ij,ij->i", d, d)) / radii[v]
                    # an unfollowed sphere ranks behind every followed one that holds the particle (q <= 1 there)
                    c_q.append((q + (2.0 if v >= n_s else 0.0)).astype(np.float32))
        del tree
        if exclusive and c_ids:
            # a particle lives in ONE chunk, so the nearest sphere is chosen here, chunk by chunk
            ci, cv, cq = np.concatenate(c_ids), np.concatenate(c_void), np.concatenate(c_q)
            o = np.lexsort((cq, ci))
            ci, cv = ci[o], cv[o]
            first = np.ones(ci.size, dtype=bool)
            first[1:] = ci[1:] != ci[:-1]
            c_ids, c_void = [ci[first]], [cv[first]]
        got_ids += c_ids
        got_void += c_void
    ids = np.concatenate(got_ids) if got_ids else np.zeros(0, np.uint64)
    void = np.concatenate(got_void) if got_void else np.zeros(0, np.int32)
    del got_ids, got_void                         # the per-void pieces: a second copy of every pair (3 GB at z = 5)
    order = np.argsort(ids, kind="stable")
    ids, void = ids[order], void[order]
    return MemberSet(ids, void, np.bincount(void, minlength=k).astype(np.int64), n_part, radii,
                     None if n_sample is None else n_s)


@dataclass
class Link:
    """Shared member counts between the k_a progenitor and k_b descendant voids of one interval (sparse: only
    the pairs that share a particle). The descendants are a pool (2026-10-11): 0..n_desc_sample-1 are the followed
    voids of the later snapshot (its sample indices), the rest unfollowed minima; -1 / None = all followed."""
    prog: np.ndarray         # int32, progenitor index (0..k_a-1)
    desc: np.ndarray         # int32, descendant index (0..k_b-1)
    count: np.ndarray        # int64, particles shared
    n_prog_members: np.ndarray
    n_desc_members: np.ndarray
    n_desc_sample: int | None = None

    def n_followed(self) -> int:
        k_b = int(self.n_desc_members.size)
        return k_b if self.n_desc_sample is None or self.n_desc_sample < 0 else min(int(self.n_desc_sample), k_b)


def link(a: MemberSet, b: MemberSet, block: int = 20_000_000) -> Link:
    """The contingency table of two member sets by shared ParticleID (a many-to-many join on the sorted ids): 'a'
    the progenitors (MemberSet.sample() of a pool), 'b' the descendant pool (its n_sample = the followed ones)."""
    k_a, k_b = a.n_members.size, b.n_members.size
    acc = np.zeros(0, np.int64), np.zeros(0, np.int64)
    # in blocks of progenitor pairs: at z = 20 nearly every particle is a member, and one join over all of them
    # held ~10 GB of index arrays at once
    for lo_i in range(0, a.ids.size, block):
        ids, void = a.ids[lo_i:lo_i + block], a.void[lo_i:lo_i + block]
        lo = np.searchsorted(b.ids, ids, side="left")
        mult = np.searchsorted(b.ids, ids, side="right") - lo
        tot = int(mult.sum())
        if tot == 0:
            continue
        starts = np.repeat(lo, mult) + (np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(mult) - mult, mult))
        key = np.repeat(void.astype(np.int64), mult) * k_b + b.void[starts]
        del starts
        uk, cnt = np.unique(key, return_counts=True)
        acc = (np.concatenate([acc[0], uk]), np.concatenate([acc[1], cnt]))
    uk, inv = np.unique(acc[0], return_inverse=True)
    cnt = np.bincount(inv.reshape(-1), weights=acc[1], minlength=uk.size).astype(np.int64)
    return Link((uk // k_b).astype(np.int32), (uk % k_b).astype(np.int32), cnt,
                a.n_members.copy(), b.n_members.copy(), b.n_sample)


def _top_two(prog, desc, f, k_a):
    """Per progenitor: the descendant with the largest fraction (-1 if none), that fraction and the second largest."""
    order = np.lexsort((-f, prog))                # per progenitor, largest fraction first (ties: lower index first)
    p_s, d_s, f_s = prog[order], desc[order], f[order]
    first = np.ones(p_s.size, dtype=bool)
    first[1:] = p_s[1:] != p_s[:-1]
    second = np.zeros(p_s.size, dtype=bool)
    second[1:] = first[:-1] & ~first[1:]
    best = np.full(k_a, -1, dtype=np.int64)
    f_max, f_second = np.zeros(k_a), np.zeros(k_a)
    best[p_s[first]] = d_s[first]
    f_max[p_s[first]] = f_s[first]
    f_second[p_s[second]] = f_s[second]
    return best, f_max, f_second


def classify(lk: Link, survive_frac: float = SURVIVE_FRAC, split_frac: float = SPLIT_FRAC) -> dict:
    """Per progenitor (C6, over the whole descendant pool): 'cls' (code into CLASSES), 'desc' (main descendant pool
    index, -1 if none: the followed void with the largest share when that share is >= survive_frac, else the pool
    void with the largest share, also for a split), 'in_sample' (that descendant is followed: its pool index is its
    sample index at the later snapshot), 'f_desc' (its share), 'f_max' and 'f_second' (the two largest shared
    fractions over the pool), 'f_sample' (the largest in a followed void), 'n_split' (pool voids holding
    >= split_frac) and 'n_main' (progenitors passing >= survive_frac to the same main descendant: >= 2 is a merge,
    also into an unfollowed one). A progenitor without members vanishes."""
    k_a = lk.n_prog_members.size
    k_s = lk.n_followed()
    f = np.zeros(lk.count.size)
    nm = lk.n_prog_members[lk.prog].astype(np.float64)
    np.divide(lk.count, nm, out=f, where=nm > 0)
    best, f_max, f_second = _top_two(lk.prog, lk.desc, f, k_a)
    s = lk.desc < k_s
    best_s, f_sample, _f2 = _top_two(lk.prog[s], lk.desc[s], f[s], k_a)
    # a followed void holding half the matter is THE descendant even when an overlapping unfollowed minimum (a
    # sub-minimum of the same void, as a rule) holds more: the matter is in the followed void
    pick_s = f_sample >= survive_frac
    desc = np.where(pick_s, best_s, best)
    f_desc = np.where(pick_s, f_sample, f_max)
    n_split = np.bincount(lk.prog[f >= split_frac], minlength=k_a)
    main = f_desc >= survive_frac
    # a descendant reached with >= survive_frac by two or more progenitors: they MERGE
    shared = np.bincount(desc[main], minlength=max(int(lk.n_desc_members.size), 1))
    n_main = np.where(main, shared[np.where(main, desc, 0)], 0)
    desc_out = np.where(f_desc > 0, desc, -1)
    in_sample = (desc_out >= 0) & (desc_out < k_s)
    cls = np.full(k_a, CLASS_CODE["vanish"], dtype=np.int8)
    cls[main] = CLASS_CODE["survive"]
    cls[main & (n_main >= 2)] = CLASS_CODE["merge"]
    cls[~main & (n_split >= 2)] = CLASS_CODE["split"]
    cls[main & ~in_sample] = CLASS_CODE["unresolved"]
    return {"cls": cls, "desc": desc_out, "in_sample": in_sample, "f_desc": f_desc, "f_max": f_max,
            "f_second": f_second, "f_sample": f_sample, "n_split": n_split, "n_main": n_main}


def main_progenitor(lk: Link) -> tuple[np.ndarray, np.ndarray]:
    """Per DESCENDANT (pool index; a followed void's is its sample index): the progenitor (a followed void of the
    earlier snapshot) that gave it the most particles (-1 if none) and that share of the descendant's members --
    the backward link a tracker walks from z = 0 to the past, through followed voids only."""
    k_b = lk.n_desc_members.size
    prog = np.full(k_b, -1, dtype=np.int64)
    frac = np.zeros(k_b)
    if lk.count.size:
        order = np.lexsort((-lk.count, lk.desc))
        d_s = lk.desc[order]
        first = np.ones(d_s.size, dtype=bool)
        first[1:] = d_s[1:] != d_s[:-1]
        prog[d_s[first]] = lk.prog[order][first]
        nm = lk.n_desc_members[d_s[first]].astype(np.float64)
        frac[d_s[first]] = np.where(nm > 0, lk.count[order][first] / np.maximum(nm, 1), 0.0)
    return prog, frac


def chain_survival(fates: list[dict]) -> np.ndarray:
    """Whether each void of the FIRST interval's progenitors still has a main descendant (survive or merge) after
    every interval of 'fates' (consecutive, each 'desc' indexing the next one's progenitors). The chain follows only
    main descendants that are themselves FOLLOWED (2026-10-11): a void whose matter went into an unfollowed minimum
    ('unresolved') has no fate of its own at the next step, so its chain ends there (not alive) -- the chained
    survival is a lower limit on the matter's survival as a void, and the direct link (plot_void_fate --ref-snap)
    the check of how much it loses."""
    alive = np.ones(fates[0]["cls"].size, dtype=bool)
    cur = np.arange(fates[0]["cls"].size)
    for fate in fates:
        ok = np.isin(fate["cls"][cur], (CLASS_CODE["survive"], CLASS_CODE["merge"]))
        if "in_sample" in fate:
            ok &= np.asarray(fate["in_sample"])[cur]
        alive &= ok
        cur = np.where(alive, fate["desc"][cur], 0)
    return alive


# ------------------------------------------------------------------ the cached link of one interval
def link_to_arrays(lk: Link) -> dict:
    return {"prog": lk.prog, "desc": lk.desc, "count": lk.count, "n_prog_members": lk.n_prog_members,
            "n_desc_members": lk.n_desc_members, "n_desc_sample": lk.n_followed(), "link_version": LINK_VERSION}


def link_from_arrays(z) -> Link:
    lk = Link(*(np.asarray(z[k]) for k in ("prog", "desc", "count", "n_prog_members", "n_desc_members")))
    lk.n_desc_sample = int(z["n_desc_sample"]) if "n_desc_sample" in z else None
    return lk


# ------------------------------------------------------------------ one snapshot's linked voids, and cached links
class VoidSnapshot:
    """One snapshot's linked voids: the pipeline catalogue, the FOLLOWED rows (void_sample) and their spheres
    (fractional), the descendant POOL (void_pool: pool_rows, pool_centres, pool_radii; the followed first), the
    frame and the particle count; the members of the pool are read from the combined snapshot on first use, in ONE
    pass (the followed ones are its first n_sample spheres: prog_members). ValueError (the frame guard: DTFE_SIM is
    not 'sim') and FileNotFoundError (no grids, no combined file) are the caller's to report."""

    def __init__(self, sim, snap, method=None, prefix=None, data_root=None, exclusive=False):
        from . import catalog, pipeline
        self.sim, self.snap, self.exclusive = sim, int(snap), bool(exclusive)
        self.path = combined_path(sim, snap, data_root)
        prod = pipeline.products(int(snap), sim=sim, method=method, prefix=prefix, data_root=data_root)
        try:
            self.cell, self.box, self.n, _src = catalog.frame_of(prod)
            self.cat = prod.voids()
            self.fs, self.method, self.prefix = prod.fs, prod.fs.method, prod.fs.prefix
        finally:
            prod.release()
        with h5py.File(self.path, "r") as f:
            self.n_particles = int(f["PartType1/ParticleIDs"].shape[0])
        self.rows = void_sample(self.cat, self.n)
        self.centres, self.radii = void_spheres(self.cat, self.n, self.rows)
        self.pool_rows, pool_r = void_pool(self.cat, self.n, self.rows)
        self.pool_centres, self.pool_radii = void_spheres(self.cat, self.n, self.pool_rows, pool_r)
        self._members = None
        self._prog = None

    def members(self) -> MemberSet:
        """The members of the whole pool (the descendant side of a link)."""
        if self._members is None:
            t0 = time.time()
            self._members = void_members(self.path, self.pool_centres, self.pool_radii, self.exclusive,
                                         n_sample=self.rows.size)
            print(f"    snap {self.snap:3d}: {self.rows.size} followed voids of {self.pool_rows.size} in the pool, "
                  f"{self._members.ids.size:,} member particles read in {time.time() - t0:.0f} s", flush=True)
        return self._members

    def prog_members(self) -> MemberSet:
        """The members of the followed voids (the progenitor side of a link): the pool's share when it is read, else
        the followed spheres alone (the same set, void_members' n_sample rule) -- the first snapshot of a census and a
        snapshot whose descendant-side link was cached never need their whole pool."""
        if self._prog is None:
            if self._members is not None:
                self._prog = self._members.sample()
            else:
                t0 = time.time()
                self._prog = void_members(self.path, self.centres, self.radii, self.exclusive)
                print(f"    snap {self.snap:3d}: {self.rows.size} followed voids, {self._prog.ids.size:,} member "
                      f"particles read in {time.time() - t0:.0f} s", flush=True)
        return self._prog

    def drop_members(self, keep_prog: bool = False):
        """Free the pool's members; keep_prog keeps the followed voids' (when read) for this snapshot's own link to
        the next, so at most one snapshot's whole pool is held besides the progenitors'."""
        if keep_prog and self._members is not None and self._prog is None:
            self._prog = self._members.sample()
        self._members = None
        if not keep_prog:
            self._prog = None


def link_tag(method, prefix) -> str:
    """The grid set's tag in link-cache and census names: '' for PS-DTFE's default grids, else '_<method>' (another
    estimator's own grids) or '_<prefix>'. One rule for plot_void_fate and plot_void_tracking --link particles
    (2026-10-11): the tracker passed no tag, so a --prefix run read and rewrote the default set's link files (the
    sphere check relinked them, each run undoing the other's)."""
    default_prefix = {"ps": "ps_output.", "dtfe": "output."}
    g = method if prefix == default_prefix.get(method) else str(prefix).rstrip(".")
    return "" if g == "ps" else f"_{g}"


def link_cache_path(a: VoidSnapshot, b: VoidSnapshot, tag: str = "") -> Path:
    """The pipeline-cache file of the link a -> b (its parameter namespace: smoothing, footprint, criterion);
    '_excl' for exclusive members (a partition: other counts)."""
    from . import pipeline
    excl = "_excl" if getattr(a, "exclusive", False) else ""
    return pipeline.cache_dir() / f"{a.sim}{tag}_{a.method}_fate{excl}_{a.snap:03d}_{b.snap:03d}.npz"


def cached_link(a: VoidSnapshot, b: VoidSnapshot, tag: str = "", log=print) -> Link:
    """The shared-particle link a -> b (a's followed voids -> b's whole pool): from the cache when it is of today's
    LINK_VERSION and both snapshots' spheres are the ones it was made with, else from the particles (then cached,
    atomically as every pipeline cache). The classes are not cached: they are taken from the counts at call time
    (classify), so new thresholds need no particle read. A version-1 cache (2026-10-11 morning: the descendants were
    the followed sample alone) is rebuilt, whatever its spheres."""
    from . import pipeline
    p = link_cache_path(a, b, tag)
    spheres = {"prog_centres": a.centres, "prog_radii": a.radii,
               "desc_centres": b.pool_centres, "desc_radii": b.pool_radii}
    if p.exists():
        try:
            with np.load(p) as z:
                version = int(z["link_version"]) if "link_version" in z else 1
                if version != LINK_VERSION:
                    log(f"    cached {p.name}: link version {version} (descendants = the followed voids only), "
                        f"today's is {LINK_VERSION}: relinking")
                elif int(z["n_desc_sample"]) == b.rows.size and all(
                        z[k].shape == v.shape and np.allclose(z[k], v, rtol=0, atol=1e-12) for k, v in spheres.items()):
                    return link_from_arrays(z)
                else:
                    log(f"    cached {p.name}: made for other voids than today's catalogues: relinking")
        except (OSError, ValueError, KeyError) as e:
            log(f"    cached {p.name} unreadable ({e}): relinking")
    lk = link(a.prog_members(), b.members())
    pipeline._save_npz(p, **link_to_arrays(lk), **spheres)
    return lk
