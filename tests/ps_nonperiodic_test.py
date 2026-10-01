#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TMP = os.path.join(HERE, "tmp")

N = 48
BOX = 100.0
AMP = 0.5
MARGIN = 0.2
SEED = 42
JITTER = 0.02
GRIDS = [48, 64]
MASS_LO, MASS_HI = 0.85, 1.15

# Part 2, a --box REGION cut from the cloud: a coarse multi-stream clump (crossed waves, tets spanning
# many cells), deposited on [40,60]^3 at 0.25 Mpc (holds the whole cloud: nothing crosses its faces)
# and on the region [46,54]^3 at the same cells. Every region cell must equal the same cell of the
# large grid -- face cells included. Before 2026-09-30 a tet cut by the region face gave its WHOLE
# mass to its inside part: face cells ~3x too heavy, the region's interior +36%.
REGION_N = 24
FULL_BOX, FULL_GRID = (40.0, 60.0), 80
REGION_BOX, REGION_GRID = (46.0, 54.0), 32
REGION_OFF = 24          # (46 - 40) / 0.25
CPU_TOL, GPU_TOL = 1e-5, 1e-4   # max |region - full| / peak (CPU: equal but for summation order)

# Part 3, --partition: the same clump split 2x2x2 must equal the single triangulation (the partial
# sums differ only in summation order). The convex hull of a finite cloud is a GLOBAL object -- the
# flat slivers on a nearly planar face join particles a whole face apart -- so each partition used
# to build different hull tetrahedra: 0.04% of cells off by up to 1% of peak, mass 0.99997. The
# Lagrangian domain is now the cloud's alpha shape (--ps-alpha-shape, 3 mean spacings), which is
# local; '--ps-alpha-shape 0' restores the hull, and must still show the old difference.
SPLIT_TOL = 1e-5


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--binary", default=None, help="PS-DTFE binary to test (default: the repo's)")
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    binary = args.binary or os.path.join(ROOT, "PS-DTFE")
    snap = os.path.join(TMP, "ps_np.hdf5")
    out = os.path.join(TMP, "ps_np_out")

    print("=" * 60)
    print(" PS-DTFE non-periodic mass-conservation test (Tier 2.1)")
    print(f"   N={N}^3  box={BOX} Mpc  margin={MARGIN}  AMP={AMP} (single-stream clump)")
    print("=" * 60)

    if not args.no_build and not args.binary:
        # rebuild in the current GPU mode (see o_ps/.build_mode; a plain make strips GPU support)
        try:
            mode = open(os.path.join(ROOT, "o_ps", ".build_mode")).read().strip()
        except OSError:
            mode = ""
        run(["make", "PS-DTFE"] + ([mode] if mode else []))
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        sys.exit(f"FAIL: '{binary}' not built")

    run([sys.executable, os.path.join(HERE, "generate_ps_test_data.py"),
         "--out", snap, "--n", str(N), "--box", str(BOX), "--amplitude-factor", str(AMP),
         "--margin-frac", str(MARGIN), "--jitter-frac", str(JITTER), "--seed", str(SEED)])

    fails = []
    for grid in GRIDS:
        run([binary, snap, out, "--grid", str(grid), "--field", "density",
             "--input", "105", "--MpcUnit", "1", "--verbose", "0"])
        d = np.fromfile(out + ".den", dtype=np.float32).astype(np.float64).reshape(grid, grid, grid)
        # '.den' is rho/rho_bar with rho_bar = N*m / V_lagBox for a non-periodic cloud
        # (Lagrangian bounding-box normalization); recovered/expected mass is then
        # sum(d)*cellVol / V_lagBox. The bbox spans (N-1) lattice spacings (+O(jitter)).
        spacing = (1.0 - 2.0 * MARGIN) * BOX / N
        v_lagbox = ((1.0 - 2.0 * MARGIN) * BOX - spacing) ** 3
        mass = float(d.sum() * (BOX / grid) ** 3 / v_lagbox)
        finite = bool(np.isfinite(d).all())
        nonneg = bool(d.min() >= 0.0)
        lo, hi = int(0.30 * grid), int(0.70 * grid)
        cov = float((d[lo:hi, lo:hi, lo:hi] > 0).mean())
        print(f"  grid={grid:>3}: mass ratio={mass:.4f}  interior coverage={100*cov:.1f}%  "
              f"finite={finite}  min>=0={nonneg}")
        if not (MASS_LO <= mass <= MASS_HI):
            fails.append(f"grid {grid}: mass ratio {mass:.3f} outside [{MASS_LO},{MASS_HI}]")
        if not finite:
            fails.append(f"grid {grid}: non-finite density")
        if not nonneg:
            fails.append(f"grid {grid}: negative density")
        if cov < 0.80:
            fails.append(f"grid {grid}: cloud interior coverage {100*cov:.1f}% < 80%")
        try:
            os.remove(out + ".den"); os.remove(out + ".streams")
        except OSError:
            pass

    fails += region_faces(binary)
    fails += partition_split(binary)

    print("-" * 60)
    if fails:
        print("RESULT: FAIL")
        for f in fails:
            print("  - " + f)
        return 1
    print("RESULT: PASS  (non-periodic finite cloud conserves mass and covers its interior; a --box region "
          "equals the same cells of a larger grid, faces included; a --partition split equals one triangulation)")
    return 0


def region_faces(binary):
    """Part 2: a --box region cut from the cloud keeps only the mass inside it."""
    print(f"  region [{REGION_BOX[0]:g},{REGION_BOX[1]:g}]^3 of a {REGION_N}^3 crossed-wave clump vs the same "
          f"cells of [{FULL_BOX[0]:g},{FULL_BOX[1]:g}]^3")
    snap = os.path.join(TMP, "ps_np_clump.hdf5")
    run([sys.executable, os.path.join(HERE, "generate_ps_test_data.py"),
         "--out", snap, "--n", str(REGION_N), "--box", str(BOX), "--margin-frac", str(MARGIN),
         "--crossed-waves", "--amplitude-factor", "1.8", "--jitter-frac", str(JITTER), "--seed", str(SEED)])
    h = subprocess.run([binary, "--full_help"], capture_output=True, text=True)
    helptext = h.stdout + h.stderr
    try:
        gpu = "METAL=1" in open(os.path.join(ROOT, "o_ps", ".build_mode")).read()
    except OSError:
        gpu = False
    variants = [("sampled", []), ("linear", ["--ps-linear-deposit"])]
    if "--ps-exact-deposit" in helptext:
        variants.append(("exact", ["--ps-exact-deposit"]))
    if gpu:
        variants += [(name + " GPU", extra + ["--ps-gpu"]) for name, extra in list(variants)]
    # one triangulation: a partitioned non-periodic run differs from a single one at the Lagrangian
    # partition seams (a separate, known issue), and this part tests the region faces only
    common = ["--input", "105", "--MpcUnit", "1", "--verbose", "0", "--avg-subsamples", "3",
              "--partition", "1", "1", "1",
              "--field", "density", "velocity", "density_a", "velocity_a"]
    n, o = REGION_GRID, REGION_OFF
    face = np.zeros((n, n, n), bool)
    face[[0, -1], :, :] = True
    face[:, [0, -1], :] = True
    face[:, :, [0, -1]] = True
    fails = []
    for name, extra in variants:
        tol = GPU_TOL if "GPU" in name else CPU_TOL
        full, reg = os.path.join(TMP, "ps_np_full"), os.path.join(TMP, "ps_np_region")
        run([binary, snap, full, "--grid", str(FULL_GRID), "--box"] + [str(x) for x in FULL_BOX * 3] + common + extra)
        run([binary, snap, reg, "--grid", str(n), "--box"] + [str(x) for x in REGION_BOX * 3] + common + extra)
        worst = []
        for ext, comps in (("den", 1), ("a_den", 1), ("vel", 3), ("a_vel", 3), ("streams", 1), ("a_streams", 1)):
            f = np.fromfile(f"{full}.{ext}", dtype=np.float32).astype(np.float64)
            f = f.reshape(FULL_GRID, FULL_GRID, FULL_GRID, comps)[o:o + n, o:o + n, o:o + n]
            r = np.fromfile(f"{reg}.{ext}", dtype=np.float32).astype(np.float64).reshape(n, n, n, comps)
            ok = np.isfinite(f) & np.isfinite(r)   # velocity is NaN in empty cells, in both
            if not np.array_equal(np.isfinite(f), np.isfinite(r)):
                fails.append(f"{name}: '.{ext}' empty cells differ between the region and the full grid")
                continue
            d = float(np.abs(r - f)[ok].max() / max(float(np.abs(f[ok]).max()), 1e-30))
            worst.append((d, ext))
            if d > tol:
                fails.append(f"{name}: '.{ext}' region differs from the full grid by {d:.1e} of peak (tol {tol:g})")
            if ext in ("den", "a_den"):
                ratio = float(r[face].sum() / f[face].sum())
                if abs(ratio - 1.0) > tol:
                    fails.append(f"{name}: '.{ext}' face-layer mass region/full = {ratio:.4f} (tol {tol:g})")
        d, ext = max(worst) if worst else (float("nan"), "-")
        print(f"    {name:12s} worst |region - full| / peak = {d:.1e} ('.{ext}')")
        for base in (full, reg):
            for fn in os.listdir(TMP):
                if fn.startswith(os.path.basename(base) + "."):
                    os.remove(os.path.join(TMP, fn))
    os.remove(snap)
    return fails



def partition_split(binary):
    """Part 3: a --partition 2 2 2 split of a non-periodic cloud equals the single triangulation."""
    print(f"  --partition 2 2 2 vs one triangulation, {REGION_N}^3 crossed-wave clump on [{FULL_BOX[0]:g},{FULL_BOX[1]:g}]^3")
    snap = os.path.join(TMP, "ps_np_split.hdf5")
    run([sys.executable, os.path.join(HERE, "generate_ps_test_data.py"),
         "--out", snap, "--n", str(REGION_N), "--box", str(BOX), "--margin-frac", str(MARGIN),
         "--crossed-waves", "--amplitude-factor", "1.8", "--jitter-frac", str(JITTER), "--seed", str(SEED)])
    h = subprocess.run([binary, "--full_help"], capture_output=True, text=True)
    if "--ps-alpha-shape" not in h.stdout + h.stderr:
        os.remove(snap)
        return [f"'{binary}' has no --ps-alpha-shape (built before 2026-10-01?): non-periodic splits differ at the hull"]
    common = ["--input", "105", "--MpcUnit", "1", "--verbose", "0", "--avg-subsamples", "3",
              "--grid", str(FULL_GRID), "--box"] + [str(x) for x in FULL_BOX * 3] + \
             ["--field", "density", "velocity", "density_a", "velocity_a"]
    base = os.path.join(TMP, "ps_np_split")
    fails = []
    worst_hull = 0.0
    for name, extra in (("alpha shape", []), ("alpha shape, --ps-vertex-mass", ["--ps-vertex-mass"]),
                        ("whole hull (--ps-alpha-shape 0)", ["--ps-alpha-shape", "0"])):
        run([binary, snap, base + "1"] + common + extra + ["--partition", "1", "1", "1"])
        run([binary, snap, base + "8"] + common + extra + ["--partition", "2", "2", "2"])
        worst = (0.0, "-")
        for ext in ("den", "a_den", "vel", "a_vel", "streams", "a_streams"):
            a = np.fromfile(f"{base}8.{ext}", dtype=np.float32).astype(np.float64)
            b = np.fromfile(f"{base}1.{ext}", dtype=np.float32).astype(np.float64)
            ok = np.isfinite(a) & np.isfinite(b)
            d = float(np.abs(a - b)[ok].max() / max(float(np.abs(b[ok]).max()), 1e-30))
            worst = max(worst, (d, ext))
            if ext in ("streams", "a_streams") and "hull" not in name and not np.array_equal(a[ok], b[ok]):
                # the sample counts are exact integers averaged over the cell: summation order only
                if d > 1e-6:
                    fails.append(f"{name}: '.{ext}' split differs from one triangulation by {d:.1e} of peak")
        print(f"    {name:32s} worst |split - single| / peak = {worst[0]:.1e} ('.{worst[1]}')")
        if "hull" in name:
            worst_hull = worst[0]
        elif worst[0] > SPLIT_TOL:
            fails.append(f"{name}: split differs from one triangulation by {worst[0]:.1e} of peak ('.{worst[1]}', tol {SPLIT_TOL:g})")
        for fn in os.listdir(TMP):
            if fn.startswith("ps_np_split1.") or fn.startswith("ps_np_split8."):
                os.remove(os.path.join(TMP, fn))
    if worst_hull < 10 * SPLIT_TOL:
        fails.append(f"the whole-hull control split agrees to {worst_hull:.1e}: the check no longer sees the hull "
                     "slivers it guards against (test data changed?)")
    os.remove(snap)
    return fails


if __name__ == "__main__":
    sys.exit(main())
