"""A/B a sampled-deposit grid set against an exact-deposit (--ps-exact-deposit) one.

Answers the practical question "is the exact deposit worth its ~2.7x wall time?" with
numbers instead of impressions, focused on the regimes that matter for void work:

  * mass conservation      (both should be ~1; the exact deposit is exact by construction)
  * global agreement       median/percentile relative differences, cell by cell
  * VOID interiors         rho/rho_bar < 0.2 -- where sampling noise is worst in relative
                           terms, because a cell holds few sub-sample hits
  * the density PDF        low-density tail, where a void catalogue's thresholds live
  * per-field summaries    velocity divergence / dispersion when both sets have them

Both sets must come from the SAME snapshot and grid; pass the two on-disk prefixes:

    python3 tools/compare_deposits.py --sim TNG300-3-Dark --snap 99 \
        --a ps_sampled --b ps_output

'--a' is the reference (sampled), '--b' the candidate (exact).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # python/ on sys.path


EMPTY = 1e-12   # a cell at or below this is EMPTY: the tessellation is space-filling, so
                # a zero-density cell is a deposit artifact, never physics


def rel_diff(a, b, mask=None):
    """Relative difference |b-a|/|a| over cells where BOTH sets are non-empty.

    Cells where either set is empty are excluded and reported separately: dividing by a
    spurious zero manufactures 1e30-scale 'differences' that swamp the real signal (and
    that is exactly the artifact this comparison exists to measure).
    """
    if mask is not None:
        a, b = a[mask], b[mask]
    both = (np.abs(a) > EMPTY) & (np.abs(b) > EMPTY)
    return np.abs(b[both] - a[both]) / np.abs(a[both])


def summarize(name, a, b, mask=None, unit=""):
    r = rel_diff(a, b, mask)
    if r.size == 0:
        print(f"  {name:22s} (no cells in mask)")
        return
    p50, p90, p99 = np.percentile(r, [50, 90, 99])
    print(f"  {name:22s} median {p50:8.2e}   p90 {p90:8.2e}   p99 {p99:8.2e}   "
          f"max {r.max():8.2e}{unit}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", required=True)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--a", default="ps_sampled", help="reference prefix (default ps_sampled)")
    p.add_argument("--b", default="ps_output", help="candidate prefix (default ps_output)")
    p.add_argument("--void-cut", type=float, default=0.2,
                   help="rho/rho_bar below this counts as void interior (default 0.2)")
    args = p.parse_args()

    from dtfelib.io import FieldSet
    from dtfelib.cli import DATA_ROOT
    snapdir = DATA_ROOT / args.sim / f"snapdir_{args.snap:03d}"
    fa = FieldSet(snapdir, prefix=args.a)
    fb = FieldSet(snapdir, prefix=args.b)
    print(f"A (reference): {fa!r}")
    print(f"B (candidate): {fb!r}\n")

    da = fa.density(units="mean").astype(np.float64)
    db = fb.density(units="mean").astype(np.float64)
    if da.shape != db.shape:
        sys.exit(f"grid mismatch: {da.shape} vs {db.shape}")

    print("MASS CONSERVATION (grid mean of rho/rho_bar; 1.0 = exact)")
    print(f"  A {da.mean():.6f}      B {db.mean():.6f}      "
          f"difference {abs(db.mean() - da.mean()):.2e}\n")

    # THE headline for a sampled-vs-exact comparison: the Lagrangian tessellation is
    # space-filling, so every Eulerian cell is overlapped by some tetrahedron and no cell
    # can be physically empty. Cells left at zero are cells the sub-sampling missed.
    za, zb = da <= EMPTY, db <= EMPTY
    print("EMPTY CELLS (rho <= 1e-12) -- a space-filling tessellation admits none")
    print(f"  A {za.sum():>12,} ({100 * za.mean():6.3f}%)      "
          f"B {zb.sum():>12,} ({100 * zb.mean():6.3f}%)")
    only_a, only_b = za & ~zb, zb & ~za
    if only_a.any():
        print(f"  empty in A, filled in B: {only_a.sum():,} cells; there B has median "
              f"rho/rho_bar = {np.median(db[only_a]):.4g}")
    if only_b.any():
        print(f"  empty in B, filled in A: {only_b.sum():,} cells; there A has median "
              f"rho/rho_bar = {np.median(da[only_b]):.4g}")
    print()

    print("DENSITY, relative difference |B-A|/A (cells non-empty in BOTH sets)")
    summarize("all cells", da, db)
    void = da < args.void_cut
    summarize(f"voids (rho<{args.void_cut})", da, db, void)
    summarize("mean-density shell", da, db, (da > 0.8) & (da < 1.25))
    summarize("overdense (rho>10)", da, db, da > 10)
    print(f"  void cells: {void.sum():,} of {da.size:,} ({100 * void.mean():.2f}%)\n")

    print("DENSITY PDF, low-density tail (fraction of cells below each threshold)")
    print(f"  {'threshold':>12s}  {'A':>12s}  {'B':>12s}  {'B/A':>8s}")
    for t in (0.05, 0.1, 0.2, 0.3, 0.5):
        fa_, fb_ = (da < t).mean(), (db < t).mean()
        ratio = fb_ / fa_ if fa_ > 0 else float("nan")
        print(f"  {t:12.2f}  {fa_:12.6f}  {fb_:12.6f}  {ratio:8.4f}")
    print()

    print("SMOOTHNESS (sampling noise shows up as cell-to-cell scatter)")
    for tag, d in (("A", da), ("B", db)):
        lap = np.abs(np.diff(d, axis=0)).mean() + np.abs(np.diff(d, axis=1)).mean() \
            + np.abs(np.diff(d, axis=2)).mean()
        vm = d[void] if void.any() else d
        print(f"  {tag}: mean |neighbour difference| {lap / 3:.5f}   "
              f"void-interior scatter (std/mean) {vm.std() / max(vm.mean(), 1e-30):.5f}")
    print()

    for name, label in (("divergence", "velocity divergence"),
                        ("dispersion", "velocity dispersion")):
        if fa.has(name) and fb.has(name):
            va = fa.load(name).astype(np.float64)
            vb = fb.load(name).astype(np.float64)
            print(f"{label.upper()}, relative difference")
            summarize("all cells", va, vb)
            summarize(f"voids (rho<{args.void_cut})", va, vb, void)
            print()

    print("Interpretation:\n"
          "  * EMPTY CELLS is usually the decisive line. Any cell the sampled deposit leaves at\n"
          "    zero is a sampling artifact (no sub-sample point landed inside a tetrahedron),\n"
          "    and those cells sit in the deepest low-density regions -- exactly where a void\n"
          "    catalogue thresholds. The exact deposit integrates the tet-cell overlap\n"
          "    analytically and cannot produce them.\n"
          "  * Median relative differences well below ~1e-2 everywhere would mean the sampled\n"
          "    deposit is converged at this grid/nSub and the exact deposit buys little.\n"
          "  * A low-density PDF tail that shrinks from A to B is the same artifact seen in the\n"
          "    statistic that void work actually uses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
