#!/usr/bin/env python3
"""PS-DTFE convergence test: single-stream density profile vs the analytic Zel'dovich solution.

Test-design notes:
  * Uses the volume-AVERAGED field (density_a, --avg-subsamples 3): since the deposit became
    mass-conserving, the plain (nSub=1) field carries per-tetrahedron quantization noise that
    GROWS as tets shrink toward the cell size -- the '_a' fields are the science-quality
    estimator (see run_ps_dtfe.sh) and the ones whose convergence is meaningful.
  * The particle series stays at N <= GRID/2: at N ~ GRID the synthetic lattice is
    commensurate with the sampling grid and aliasing noise (slab-correlated moire from the
    tet/sub-sample lattices) dominates the window error, which says nothing about the
    estimator's truncation error.
  * The error is expected to fall and then PLATEAU at the estimator noise floor set by the
    jitter and the fixed grid/nSub, so the criteria compare finest vs coarsest rather than
    demanding strict monotonicity.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TMP = os.path.join(HERE, "tmp")

GRID = 96
BOX = 100.0
AMP = 1.8
SEED = 42
JITTER = 0.02
NSUB = 3
N_SERIES = [16, 24, 32, 48]
FINEST_TOL = 0.008


from ps_test_helpers import make_cmd


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def analytic_single_stream_profile(x, amp):
    q = np.linspace(0.0, 1.0, 400000)
    xb = q + amp / (2 * np.pi) * np.sin(2 * np.pi * q)
    jac = 1.0 + amp * np.cos(2 * np.pi * q)
    ss = jac > 0.0
    xs = np.mod(xb[ss], 1.0)
    rho = 1.0 / jac[ss]
    order = np.argsort(xs)
    return np.interp(x, xs[order], rho[order])


def single_stream_error(den_path, grid, amp):
    d = np.fromfile(den_path, dtype=np.float32).astype(np.float64).reshape(grid, grid, grid)
    prof = d.mean(axis=(1, 2)) / d.mean()
    x = (np.arange(grid) + 0.5) / grid
    ana = analytic_single_stream_profile(x, amp)
    sel = ((x > 0.05) & (x < 0.35)) | ((x > 0.65) & (x < 0.95))
    return float(np.mean(np.abs(prof[sel] - ana[sel])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    binary = os.path.join(ROOT, "PS-DTFE")

    print("=" * 60)
    print(" PS-DTFE convergence test (Tier 4.4)")
    print(f"   grid={GRID}^3  box={BOX} Mpc  AMP={AMP}  N series={N_SERIES}")
    print("=" * 60)

    if not args.no_build:
        run(make_cmd())
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        sys.exit(f"FAIL: '{binary}' not built")

    errors = []
    for n in N_SERIES:
        snap = os.path.join(TMP, f"ps_conv_{n}.hdf5")
        out = os.path.join(TMP, f"ps_conv_{n}_out")
        run([sys.executable, os.path.join(HERE, "generate_ps_test_data.py"),
             "--out", snap, "--n", str(n), "--box", str(BOX), "--amplitude-factor", str(AMP),
             "--jitter-frac", str(JITTER), "--seed", str(SEED)])
        run([binary, snap, out, "--grid", str(GRID), "--periodic",
             "--avg-subsamples", str(NSUB), "--field", "density_a",
             "--input", "105", "--MpcUnit", "1", "--verbose", "0"])
        err = single_stream_error(out + ".a_den", GRID, AMP)
        errors.append(err)
        print(f"  N={n:>3}^3 ({n**3:>8} particles):  single-stream density_a L1 error = {err:.4f}")

    fails = []
    if errors[-1] > 0.6 * errors[0]:
        fails.append(f"insufficient convergence: finest {errors[-1]:.4f} not < 0.6 * coarsest {errors[0]:.4f}")
    if errors[-1] > FINEST_TOL:
        fails.append(f"finest-N error {errors[-1]:.4f} exceeds {FINEST_TOL} (estimator noise floor)")

    print("-" * 60)
    if fails:
        print("RESULT: FAIL")
        for f in fails:
            print("  - " + f)
        return 1
    print(f"RESULT: PASS  (single-stream density converges: {errors[0]:.4f} -> {errors[-1]:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
