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
GRID = 96
BOX = 100.0
AMP = 1.8
SEED = 42
JITTER = 0.02

VEL_MEDIAN_TOL = 1.0e-3
DEN_PROFILE_TOL = 0.15


from ps_test_helpers import load


def run(cmd):
    print("   $ " + " ".join(cmd[:3]) + " ...")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    dtfe = os.path.join(ROOT, "DTFE")
    psdtfe = os.path.join(ROOT, "PS-DTFE")
    snap = os.path.join(TMP, "ps_xcheck.hdf5")
    std_root = os.path.join(TMP, "xcheck_std")
    ps_root = os.path.join(TMP, "xcheck_ps")

    print("=" * 60)
    print(" PS-DTFE vs standard-DTFE single-stream cross-check (Tier 4.3)")
    print(f"   N={N}^3  grid={GRID}^3  box={BOX} Mpc  AMP={AMP}")
    print("=" * 60)

    if not args.no_build:
        # respect the current build mode of each build dir: a plain 'make' after a GPU build
        # (METAL=1/CUDA=1/HIP=1) flips the mode stamp and silently strips the GPU support;
        # .build_mode records the make argument of the current mode
        def make_cmd(target, obj_dir):
            try:
                mode = open(os.path.join(ROOT, obj_dir, ".build_mode")).read().strip()
            except OSError:
                mode = "METAL=1" if os.path.exists(os.path.join(ROOT, obj_dir, ".metal_mode_on")) else ""
            return ["make", target] + ([mode] if mode else [])
        run(make_cmd("DTFE", "o"))
        run(make_cmd("PS-DTFE", "o_ps"))
    for b in (dtfe, psdtfe):
        if not (os.path.isfile(b) and os.access(b, os.X_OK)):
            sys.exit(f"FAIL: '{b}' not built")

    run([sys.executable, os.path.join(HERE, "generate_ps_test_data.py"),
         "--out", snap, "--n", str(N), "--box", str(BOX),
         "--amplitude-factor", str(AMP), "--jitter-frac", str(JITTER), "--seed", str(SEED)])

    common = ["--grid", str(GRID), "--periodic", "--field", "density", "velocity",
              "--input", "105", "--MpcUnit", "1", "--verbose", "0"]
    run([dtfe, snap, std_root] + common)
    run([psdtfe, snap, ps_root] + common)

    sd = load(std_root + ".den", GRID)
    pd = load(ps_root + ".den", GRID)
    sv = load(std_root + ".vel", GRID, 3)
    pv = load(ps_root + ".vel", GRID, 3)
    st = load(ps_root + ".streams", GRID)

    print(f"\n  density normalization: standard/PS mean ratio = {sd.mean()/pd.mean():.2f} "
          f"(std mean={sd.mean():.3f} ~1; PS mean={pd.mean():.4f}=N*m/L^3)")

    fails = []

    ss = (st == 1) & (sd > 0) & (pd > 0)
    relv = np.abs(sv[..., 0][ss] - pv[..., 0][ss]) / (np.abs(sv[..., 0][ss]) + 1e-30)
    vmed = float(np.median(relv))
    print(f"  velocity (single-stream, {ss.sum()} cells): median relΔ={vmed:.2e}  "
          f"90th={np.percentile(relv,90):.2e}   [tol {VEL_MEDIAN_TOL:.0e}]")
    if vmed > VEL_MEDIAN_TOL:
        fails.append(f"single-stream velocity median relΔ {vmed:.2e} exceeds {VEL_MEDIAN_TOL:.0e}")

    sdn, pdn = sd / sd.mean(), pd / pd.mean()
    sp, pp = sdn.mean(axis=(1, 2)), pdn.mean(axis=(1, 2))
    sm = st.mean(axis=(1, 2)) < 1.01
    reld = np.abs(sp[sm] - pp[sm]) / (0.5 * (sp[sm] + pp[sm]))
    dmed = float(np.median(reld))
    print(f"  density profile (single-stream, {sm.sum()} slices): median relΔ={dmed:.2e}  "
          f"max={reld.max():.2e}   [tol {DEN_PROFILE_TOL:.2f}]")
    if dmed > DEN_PROFILE_TOL:
        fails.append(f"single-stream density profile median relΔ {dmed:.2e} exceeds {DEN_PROFILE_TOL:.2f}")

    print("-" * 60)
    if fails:
        print("RESULT: FAIL")
        for f in fails:
            print("  - " + f)
        return 1
    print("RESULT: PASS  (single-stream: velocity matches standard DTFE tightly, "
          "density to estimator accuracy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
