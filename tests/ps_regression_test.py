#!/usr/bin/env python3
"""
PS-DTFE regression test against the analytic 1-D Zel'dovich pancake, with metrics
tracked against tests/reference/ps_regression_baseline.json to catch physics
regressions. Caustics for x(q)=q+A sin(k q), J=1+AMP cos(k q): a=arccos(-1/AMP),
x_hi/L=(a+AMP sin a)/2pi, f_multi=(a+AMP sin a-pi)/pi (AMP=1.8 -> ~0.1636).
Requires numpy, h5py, the PS-DTFE build toolchain.
"""

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF_DIR = os.path.join(HERE, "reference")
BASELINE = os.path.join(REF_DIR, "ps_regression_baseline.json")
TMP = os.path.join(HERE, "tmp")

N = 48
GRID = 96
BOX = 100.0
AMP = 1.8
SEED = 42
JITTER = 0.02


def analytic_pancake(amp):
    a = math.acos(-1.0 / amp)
    x_hi = (a + amp * math.sin(a)) / (2.0 * math.pi)
    x_lo = (2.0 * math.pi - a - amp * math.sin(a)) / (2.0 * math.pi)
    return x_hi - x_lo, x_lo, x_hi


from ps_test_helpers import load

load_field = load                            # scalar grid


def load_vec(path, grid, ncomp=3):           # vector grid (velocity: 3 components)
    return load(path, grid, ncomp)


def check_velocity(vel):
    vx, vy, vz = vel[..., 0], vel[..., 1], vel[..., 2]
    sx = np.abs(vx).max() + 1e-30
    failures = []
    print("  velocity : max|vx|=%.4g  max|vy|/max|vx|=%.2e  max|vz|/max|vx|=%.2e  "
          "<vx>/max|vx|=%.2e" % (np.abs(vx).max(), np.abs(vy).max()/sx,
                                 np.abs(vz).max()/sx, vx.mean()/sx))
    if not np.isfinite(vel).all():
        failures.append("velocity field has non-finite values")
    if np.abs(vx).max() == 0.0:
        failures.append("v_x is identically zero (velocity not interpolated)")
    if np.abs(vy).max() > 1e-3 * sx or np.abs(vz).max() > 1e-3 * sx:
        failures.append("transverse velocity non-zero (pancake has v_y=v_z=0)")
    if abs(vx.mean()) > 5e-2 * sx:
        failures.append("<v_x> not ~0 (expected antisymmetric pancake velocity)")
    return failures


def analytic_dispersion_profile(grid, box, amp):
    k = 2.0 * math.pi / box
    A = amp / k
    qg = np.linspace(0.0, box, 200001)
    fg = qg + A * np.sin(k * qg)
    dfg = 1.0 + amp * np.cos(k * qg)
    brk = np.where(np.diff(np.sign(dfg)) != 0)[0]
    bnds = np.concatenate(([0], brk + 1, [len(qg) - 1]))
    branches = [(bnds[i], bnds[i + 1]) for i in range(len(bnds) - 1)]
    dx = box / grid
    xc = (np.arange(grid) + 0.5) * dx
    sig = np.zeros(grid); vbar = np.zeros(grid); nstream = np.zeros(grid, dtype=int)
    for i, X in enumerate(xc):
        qs = []
        for a, b in branches:
            fb, qb = fg[a:b + 1], qg[a:b + 1]
            if X < min(fb[0], fb[-1]) or X > max(fb[0], fb[-1]):
                continue
            qs.append(np.interp(X, fb, qb) if fb[-1] > fb[0]
                      else np.interp(X, fb[::-1], qb[::-1]))
        v = np.array([100.0 * A * math.sin(k * q) for q in qs])
        rho = np.array([1.0 / abs(1.0 + amp * math.cos(k * q)) for q in qs])
        w = rho / rho.sum()
        vb = float((w * v).sum())
        sig[i] = float((w * (v - vb) ** 2).sum()); vbar[i] = vb; nstream[i] = len(qs)
    return sig, vbar, nstream


def check_dispersion(disp, vel, streams, grid=GRID, box=BOX, amp=AMP):
    sigA, vbarA, nA = analytic_dispersion_profile(grid, box, amp)
    sig_code = disp.mean(axis=(1, 2))
    vx_code = vel[..., 0].mean(axis=(1, 2))
    m = (nA == 1) & (np.abs(vbarA) > 0.2 * np.abs(vbarA).max())
    norm = float(np.sum(vx_code[m] * vbarA[m]) / np.sum(vbarA[m] ** 2)) if m.any() else 1.0
    ic = grid // 2
    sig_analytic = norm ** 2 * sigA[ic]
    ratio = sig_code[ic] / (sig_analytic + 1e-30)
    void = streams == 1
    slab = streams > 1
    void_med = float(np.median(disp[void])) if void.any() else 0.0
    slab_med = float(np.median(disp[slab])) if slab.any() else 0.0
    print("  dispersion: slab-centre sigma^2=%.3e  analytic=%.3e  ratio=%.3f   "
          "void median=%.2e  slab median=%.2e"
          % (sig_code[ic], sig_analytic, ratio, void_med, slab_med))
    failures = []
    if not np.isfinite(disp).all():
        failures.append("dispersion has non-finite values")
    if disp.min() < 0.0:
        failures.append("dispersion trace has negative values (min=%.2e)" % disp.min())
    if void_med > 1e-3 * slab_med:
        failures.append("single-stream dispersion not ~0 (void median %.2e vs slab %.2e)"
                        % (void_med, slab_med))
    if not (0.9 <= ratio <= 1.1):
        failures.append("slab-centre dispersion %.3e off analytic %.3e (ratio %.3f, >10%%)"
                        % (sig_code[ic], sig_analytic, ratio))
    return failures, round(ratio, 3)


def run(cmd, **kw):
    print("   $ " + " ".join(cmd if len(" ".join(cmd)) < 200 else cmd[:3] + ["..."]))
    subprocess.run(cmd, check=True, **kw)


def analyze(den, streams, grid=GRID, box=BOX, amp=AMP, n=N,
            update_baseline=False, baseline_path=BASELINE,
            extra_metrics=None, extra_tol=None, pts_streams=None):
    f_analytic, xlo_a, xhi_a = analytic_pancake(amp)
    expected_mean = n ** 3 * 1.0 / box ** 3
    cell_vol = (box / grid) ** 3

    # Stream statistics use the EXACT containment counts from the point evaluation
    # (--sample-points at the cell centres) when available: the deposit's .streams at
    # nSub=1 additionally counts the centroid FALLBACK deposits of sub-cell tetrahedra
    # (what makes it exactly mass-conserving), inflating ~1/4 of the counts by
    # construction at N=grid/2 -- a deposit property, not what the analytics describe.
    counts = pts_streams if pts_streams is not None else streams

    rho_1d = den.mean(axis=(1, 2))
    multi_col = (counts > 1).mean(axis=(1, 2))
    multi_x = multi_col > 0.5

    f_measured = float((counts > 1).mean())
    max_streams = int(counts.max())
    # '.den' is mean-normalized (rho/rho_bar), so its volume-weighted mean IS the mass ratio
    mass_ratio = float(den.mean())
    coverage = float((den > 0).mean())

    idx = np.where(multi_x)[0]
    if idx.size == 0:
        slab_lo = slab_hi = float("nan")
    else:
        slab_lo = idx.min() / grid
        slab_hi = (idx.max() + 1) / grid

    ix_lo, ix_hi = int(round(xlo_a * grid)), int(round(xhi_a * grid))
    ix_mid = int(round(0.5 * grid))
    ix_void = int(round(0.25 * grid))
    streams_mid = int(np.median(counts[ix_mid]))
    streams_void = int(np.median(counts[ix_void]))

    def _peak(c):
        return rho_1d[max(0, c - 1):min(grid, c + 2)].max()
    caustic_overdensity = min(_peak(ix_lo), _peak(ix_hi)) / rho_1d.mean()

    print("\n  analytic : f_multi=%.4f  caustics x/L=[%.4f, %.4f]"
          % (f_analytic, xlo_a, xhi_a))
    print("  measured : f_multi=%.4f  slab x/L=[%.4f, %.4f]  max_streams=%d"
          % (f_measured, slab_lo, slab_hi, max_streams))
    print("  density  : caustic overdensity=%.2fx mean   grid_mass/expected=%.3f   non-zero cells=%.1f%%"
          % (caustic_overdensity, mass_ratio, 100.0 * coverage))
    print("  streams  : slab-centre median=%d   void median=%d   "
          "(diagnostic; depressed by incomplete cell coverage)" % (streams_mid, streams_void))

    cell = 1.0 / grid
    failures = []
    if abs(f_measured - f_analytic) > 0.03:
        failures.append("multi-stream fraction %.4f off from analytic %.4f (>0.03)"
                        % (f_measured, f_analytic))
    if idx.size == 0 or not (abs(slab_lo - xlo_a) <= 3.0 * cell
                             and abs(slab_hi - xhi_a) <= 3.0 * cell):
        failures.append("3-stream slab [%.4f,%.4f] does not match analytic caustics "
                        "[%.4f,%.4f] within 3 cells" % (slab_lo, slab_hi, xlo_a, xhi_a))
    if caustic_overdensity < 2.0:
        failures.append("density not caustic-enhanced at x_lo/x_hi (%.2fx mean, <2x)"
                        % caustic_overdensity)
    if not (0.90 <= mass_ratio <= 1.15):
        failures.append("mass not conserved: grid mass / expected = %.3f (outside [0.90, 1.15])"
                        % mass_ratio)
    # Stream-count PARITY (odd counts away from the caustics) uses the same exact
    # containment counts as the stream statistics above (see the pts_streams note there).
    parity_counts = pts_streams if pts_streams is not None else streams
    covered_cells = int((parity_counts > 0).sum())
    even_cells = int(np.isin(np.rint(parity_counts).astype(int), [2, 4, 6, 8]).sum())
    even_frac = even_cells / max(covered_cells, 1)
    if pts_streams is not None and even_frac > 0.01:
        failures.append("stream-count parity broken: %.2f%% of covered points have an even "
                        "count (>1%%); should be odd away from the caustics" % (100 * even_frac))

    metrics = {"mass_ratio": round(mass_ratio, 4),
               "multistream_fraction": round(f_measured, 4),
               "max_streams": max_streams,
               "coverage": round(coverage, 4),
               "slab_stream_median": streams_mid,
               "void_stream_median": streams_void}
    tol = {"mass_ratio": 0.05, "multistream_fraction": 0.02, "max_streams": 4,
           "coverage": 0.05, "slab_stream_median": 1, "void_stream_median": 1}
    if extra_metrics:
        metrics.update(extra_metrics)
        tol.update(extra_tol or {})

    if update_baseline or not os.path.isfile(baseline_path):
        with open(baseline_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print("\n  baseline %s: %s" %
              ("updated" if update_baseline else "created", metrics))
    else:
        with open(baseline_path) as fh:
            base = json.load(fh)
        for key, val in metrics.items():
            if key in base and abs(val - base[key]) > tol[key]:
                failures.append("metric '%s'=%.4g drifted from baseline %.4g (tol %.4g)"
                                % (key, val, base[key], tol[key]))
        print("  baseline : %s" % base)

    print("-" * 60)
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print("  - " + f)
        return 1, metrics, failures
    print("RESULT: PASS  (PS-DTFE matches the analytic pancake within tolerance)")
    return 0, metrics, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-build", action="store_true", help="reuse existing PS-DTFE")
    ap.add_argument("--update-baseline", action="store_true",
                    help="(re)write the tracked-metric baseline and exit success")
    args = ap.parse_args()

    binary = os.path.join(ROOT, "PS-DTFE")
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(REF_DIR, exist_ok=True)
    snap = os.path.join(TMP, "ps_regression.hdf5")
    out_root = os.path.join(TMP, "ps_regression_out")

    print("=" * 60)
    print(" PS-DTFE regression test (analytic Zel'dovich pancake)")
    print(f"   N={N}^3  grid={GRID}^3  box={BOX} Mpc  AMP={AMP}")
    print("=" * 60)

    if not args.no_build:
        # rebuild in the current GPU mode (o_ps/.build_mode records METAL=1/CUDA=1/HIP=1;
        # a plain 'make PS-DTFE' would wipe o_ps and silently strip the GPU support)
        try:
            mode = open(os.path.join(ROOT, "o_ps", ".build_mode")).read().strip()
        except OSError:
            mode = ""
        run(["make", "PS-DTFE"] + ([mode] if mode else []), cwd=ROOT)
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        sys.exit(f"FAIL: '{binary}' not found; build it first (make PS-DTFE).")

    run([sys.executable, os.path.join(HERE, "generate_ps_test_data.py"),
         "--out", snap, "--n", str(N), "--box", str(BOX),
         "--amplitude-factor", str(AMP), "--jitter-frac", str(JITTER),
         "--seed", str(SEED)])

    for ext in (".den", ".streams", ".vel", ".a_den", ".velDisp", ".velDispTensor",
                ".pts_den", ".pts_vel", ".pts_velDisp", ".pts_streams"):
        try:
            os.remove(out_root + ext)
        except OSError:
            pass

    # cell centres as --sample-points input: the point evaluation returns the EXACT
    # per-centre stream containment counts used by the parity check in analyze()
    pts_file = os.path.join(TMP, "ps_regression_centers.bin")
    c = (np.arange(GRID) + 0.5) * BOX / GRID
    gx, gy, gz = np.meshgrid(c, c, c, indexing="ij")
    np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64).tofile(pts_file)

    run([binary, snap, out_root, "--grid", str(GRID), "--periodic",
         "--field", "density", "velocity", "density_a", "dispersion",
         "--sample-points", pts_file,
         "--input", "105", "--MpcUnit", "1", "--verbose", "1"])

    den = load_field(out_root + ".den", GRID)
    streams = load_field(out_root + ".streams", GRID)
    vel = load_vec(out_root + ".vel", GRID)
    disp = load_field(out_root + ".velDisp", GRID)
    pts_streams = np.fromfile(out_root + ".pts_streams", dtype=np.int32).reshape(GRID, GRID, GRID)

    slab_mask = streams > 1
    extra_metrics = {
        "max_vx": round(float(np.abs(vel[..., 0]).max()), 1),
        "slab_dispersion": round(float(np.median(disp[slab_mask])) if slab_mask.any() else 0.0, 2),
    }
    extra_tol = {"max_vx": 0.05 * abs(extra_metrics["max_vx"]) + 1.0,
                 "slab_dispersion": 0.10 * abs(extra_metrics["slab_dispersion"]) + 1.0}

    code, _metrics, failures = analyze(den, streams, update_baseline=args.update_baseline,
                                       extra_metrics=extra_metrics, extra_tol=extra_tol,
                                       pts_streams=pts_streams)

    vel_failures = check_velocity(vel)
    if vel_failures:
        print("VELOCITY CHECK FAIL:")
        for f in vel_failures:
            print("  - " + f)
        code = 1
    else:
        print("  velocity : OK (transverse ~0, <v_x> ~0, finite)")

    disp_failures, _disp_ratio = check_dispersion(disp, vel, streams)
    if disp_failures:
        print("DISPERSION CHECK FAIL:")
        for f in disp_failures:
            print("  - " + f)
        code = 1
    else:
        print("  dispersion: OK (~0 single-stream, matches analytic 3-stream slab)")

    a_den = load_field(out_root + ".a_den", GRID)
    # '.a_den' is mean-normalized (rho/rho_bar): the grid mean itself is the mass ratio
    a_ratio = a_den.mean()
    a_cov = float((a_den > 0).mean())
    print("  density_a: mean/expected=%.3f  non-zero cells=%.1f%%  finite=%s"
          % (a_ratio, 100 * a_cov, bool(np.isfinite(a_den).all())))
    if not np.isfinite(a_den).all():
        print("AVERAGED DENSITY FAIL: non-finite values"); code = 1
    elif a_den.min() < 0:
        print("AVERAGED DENSITY FAIL: negative density"); code = 1
    elif not (0.9 <= a_ratio <= 1.1):
        print("AVERAGED DENSITY FAIL: mass not conserved (mean/expected=%.3f)" % a_ratio); code = 1
    elif a_cov < 0.99:
        print("AVERAGED DENSITY FAIL: coverage %.1f%% < 99%%" % (100 * a_cov)); code = 1
    else:
        print("  density_a: OK (finite, full coverage, mass-conserving)")
    return code


if __name__ == "__main__":
    sys.exit(main())
