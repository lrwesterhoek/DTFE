#!/usr/bin/env python3
"""PS-DTFE 3D stream-topology test: three crossed Zel'dovich waves vs separable analytics.

Test-design notes (why these parameters and criteria):
  * N=48 with grid=96 keeps the tessellation coarser than the sampling grid. At N=grid the
    synthetic lattice's sample points coincide EXACTLY with undisplaced tessellation
    faces/edges and the inclusive point-in-tet test counts every incident tet -- a
    measurement artifact of commensurate lattices, impossible on real (glass-IC, fully-3D)
    data. Jitter also helps break these coincidences.
  * The MEAN stream count is compared after CLIPPING at the analytic maximum (27): the
    caustic tips stack an unbounded number of near-degenerate flattened tets in cells that
    are measure-zero analytically, so the raw mean diverges with resolution by construction.
    Clipping makes the statistic volume-dominated, which is what the analytics describe.
"""

import argparse
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TMP = os.path.join(HERE, "tmp")

N = 48
GRID = 96
BOX = 100.0
AMP = 1.8
SEED = 42
JITTER = 0.02


from ps_test_helpers import make_cmd, load


def run(cmd):
    print("   $ " + " ".join(cmd[:3]) + " ...")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    binary = os.path.join(ROOT, "PS-DTFE")
    snap = os.path.join(TMP, "ps_3d.hdf5")
    out = os.path.join(TMP, "ps_3d_out")

    print("=" * 60)
    print(" PS-DTFE 3D test — three crossed Zel'dovich waves (Tier 4.1)")
    print(f"   N={N}^3  grid={GRID}^3  box={BOX} Mpc  AMP={AMP}")
    print("=" * 60)

    if not args.no_build:
        run(make_cmd())
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        sys.exit(f"FAIL: '{binary}' not built")

    run([sys.executable, os.path.join(HERE, "generate_ps_test_data.py"),
         "--out", snap, "--n", str(N), "--box", str(BOX), "--amplitude-factor", str(AMP),
         "--crossed-waves", "--jitter-frac", str(JITTER), "--seed", str(SEED)])
    run([binary, snap, out, "--grid", str(GRID), "--periodic", "--partition", "2", "2", "2",
         "--field", "density", "--input", "105", "--MpcUnit", "1", "--verbose", "0"])

    den = load(out + ".den", GRID)
    streams = np.rint(load(out + ".streams", GRID)).astype(int)

    a = math.acos(-1.0 / AMP)
    f = (a + AMP * math.sin(a) - math.pi) / math.pi
    multi_a = 1.0 - (1.0 - f) ** 3
    mean_a = (1.0 + 2.0 * f) ** 3

    covered = streams > 0
    multi_m = float((streams[covered] > 1).mean())
    mean_m = float(np.minimum(streams, 27)[covered].mean())   # clipped: see module docstring
    max_m = int(streams.max())
    canonical = np.isin(streams[covered], [1, 3, 9, 27]).mean()
    # '.den' is mean-normalized (rho/rho_bar with rho_bar = N^3 m / V), so the volume-weighted
    # grid mean IS the recovered/expected mass ratio.
    mass_ratio = float(den.mean())

    print(f"\n  per-axis 3-stream fraction f = {f:.4f}")
    print(f"  multi-stream fraction : measured {multi_m:.4f}  analytic {multi_a:.4f}")
    print(f"  mean stream count     : measured {mean_m:.4f} (clipped at 27)  analytic {mean_a:.4f}")
    print(f"  max stream count      : measured {max_m}  (analytic triple caustic = 27; discrete"
          f" caustic-tip stacks exceed it)")
    print(f"  counts in {{1,3,9,27}}  : {100*canonical:.1f}% of covered cells")
    print(f"  mass ratio            : {mass_ratio:.4f}   coverage {100*covered.mean():.1f}%")

    fails = []
    if abs(multi_m - multi_a) > 0.05:
        fails.append(f"multi-stream fraction {multi_m:.3f} off analytic {multi_a:.3f} (>0.05)")
    if abs(mean_m - mean_a) > 0.30:
        fails.append(f"clipped mean stream count {mean_m:.3f} off analytic {mean_a:.3f} (>0.30)")
    if max_m < 27:
        fails.append(f"max stream count {max_m} < 27 (triple caustic not resolved)")
    if canonical < 0.90:
        fails.append(f"only {100*canonical:.1f}% of cells have a separable {{1,3,9,27}} count (<90%)")
    if not (0.85 <= mass_ratio <= 1.15):
        fails.append(f"mass ratio {mass_ratio:.3f} outside [0.85, 1.15]")
    if covered.mean() < 0.999:
        fails.append(f"grid coverage {100*covered.mean():.1f}% < 99.9% (periodic partition tiling?)")

    print("-" * 60)
    if fails:
        print("RESULT: FAIL")
        for x in fails:
            print("  - " + x)
        return 1
    print("RESULT: PASS  (3D crossed-wave stream structure matches the separable analytics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
