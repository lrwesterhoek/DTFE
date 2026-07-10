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


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    binary = os.path.join(ROOT, "PS-DTFE")
    snap = os.path.join(TMP, "ps_np.hdf5")
    out = os.path.join(TMP, "ps_np_out")

    print("=" * 60)
    print(" PS-DTFE non-periodic mass-conservation test (Tier 2.1)")
    print(f"   N={N}^3  box={BOX} Mpc  margin={MARGIN}  AMP={AMP} (single-stream clump)")
    print("=" * 60)

    if not args.no_build:
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

    print("-" * 60)
    if fails:
        print("RESULT: FAIL")
        for f in fails:
            print("  - " + f)
        return 1
    print("RESULT: PASS  (non-periodic finite cloud conserves mass and covers its interior)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
