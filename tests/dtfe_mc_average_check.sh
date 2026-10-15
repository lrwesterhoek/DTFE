#!/usr/bin/env bash
# The standard-DTFE Monte-Carlo cell average ('_a', method 1, the default without --exact-average) is UNBIASED
# and independent of the partition / thread / padding split (2026-10-13, audit finding 1). Until today every
# tetrahedron mapped the same short Sobol prefix from CGAL's vertex(0); the prefix does not average to the
# centroid and CGAL's vertex slots follow density, so every production '_a' grid lost 0.17-0.29% of its mass,
# and the realization changed by ~10-23% rms per cell with the thread count or the partitioning. Now the
# vertices are ordered by a hash of the cell's vertex set and every sample is centred (averaged_interpolation_1.cc:
# the comment above 'mcLegacy'). Asserted on demo/z0_64.gadget (262144 particles, z=0, tracked in the repo) at
# 64^3, --periodic --padding 12, density_a velocity_a:
#   A1  box mean of '.a_den' within 1e-4 of 1 (measured +2.0e-5; double build +1.4e-5), and with '-s 20' (the
#       minimum count 4, the worst Sobol prefix; measured -5.1e-6). The old realization (DTFE_MC_LEGACY=1, kept
#       for A/B) must FAIL the same bound (measured -3.35e-3), so the check is known to see the bias
#   A2  the per-cell noise against --exact-average, rms(a - exact) / <exact>, is no larger than the old
#       realization's and below 0.07 (measured 0.0625 against 0.0826, both builds)
#   A3  split invariance: --partition 2 2 2 --max-concurrent 1, OMP_NUM_THREADS=1 and --padding 20 against
#       the default run: the cells agree to float summation order (measured 6e-6 relative), except where a
#       sample lies EXACTLY on a cell face, which the region frames of two splits may round to different sides
#       (the demo's float-kpc positions and the dyadic Sobol samples put one there in the double build: two
#       cells 3e-3 apart); at most 1e-4 of the cells may differ by more than 1e-4, and the box sums agree to
#       1e-6. The old realization differed in all cells, by 10% rms between one and ten threads
#   A4  the momentum: the box sum of '.a_vel' agrees with --exact-average's to 1e-4 of sum|v| (measured 1.2e-5)
#   A5  GPU builds only (a GPU stamp in o/ or o_d/; SKIP otherwise, so CI's CPU-only build runs A1-A4): --gpu
#       keeps the box mean within 1e-4 of 1 and matches the CPU grid like tests/dtfe_metal_check.sh (means
#       to 1e-5 of the peak, at most 1e-3 of the cells more than 1e-3 apart: a sample on a cell face)
# --exact-average itself is untouched (its grids are byte-identical to before; tests/run_tests.py's stored
# regression covers it). About 20 s on the Mac (7 runs of 2-3 s).
# Usage: tests/dtfe_mc_average_check.sh [--no-build]   (--no-build is accepted and ignored: nothing is built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${DTFE_BIN}"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
[ -x "${DTFE_BIN}" ] || { echo "FAIL: ${DTFE_BIN} is not built"; exit 1; }
SNAP="demo/z0_64.gadget"
[ -f "$SNAP" ] || { echo "FAIL: $SNAP is missing (it is tracked in the repository)"; exit 1; }
TMP="${DTFE_TEST_TMP}/dtfe_mc_average_check"
mkdir -p "$TMP"
OBJ=o; precision_double && OBJ=o_d
GPU=0
for m in metal cudaemu cuda hip; do [ -f "$OBJ/.gpu_mode_$m" ] && GPU=1; done

C=(--grid 64 --periodic --field density_a velocity_a)
# run <name> [env assignments ...] -- [extra args ...]   (--padding 12 unless the extra args give one)
run() {
    local name="$1"; shift
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    local pad=(--padding 12)
    case " $* " in *" --padding "*) pad=() ;; esac
    env ${envs[@]+"${envs[@]}"} "${DTFE_BIN}" "$SNAP" "$TMP/$name" "${C[@]}" ${pad[@]+"${pad[@]}"} "$@" > "$TMP/$name.log" 2>&1 \
        || { echo "FAIL: run $name (see $TMP/$name.log)"; exit 1; }
}
echo ">> demo/z0_64.gadget at 64^3: the default average, the old one (DTFE_MC_LEGACY=1), --exact-average"
run n
run l DTFE_MC_LEGACY=1 --
run e -- --exact-average
echo ">> -s 20; --partition 2 2 2; one thread; --padding 20"
run s -- -s 20
run p -- --partition 2 2 2 --max-concurrent 1
run t OMP_NUM_THREADS=1 --
run d -- --padding 20
if [ "$GPU" = "1" ]; then
    echo ">> GPU build: --gpu"
    run g -- --gpu
    grep -q " GPU (" "$TMP/g.log" || { echo "FAIL: the --gpu run did not attempt the GPU (see $TMP/g.log)"; exit 1; }
    grep -q "using the CPU interpolation" "$TMP/g.log" && { echo "FAIL: the --gpu run fell back to the CPU (see $TMP/g.log)"; exit 1; }
else
    echo ">> A5 SKIP: $OBJ carries no GPU stamp (a METAL=1 / CUDA=1 / HIP=1 / CUDAEMU=1 build runs it)"
fi

"$PY" - "$TMP" "$GPU" <<'PYEOF'
import numpy as np, os, sys
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
tmp, gpu = sys.argv[1], sys.argv[2] == "1"
fails = 0
def check(tag, ok, msg):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {tag} {msg}")
def rd(name, ext="a_den"):
    return np.fromfile(os.path.join(tmp, f"{name}.{ext}"), dtype=REAL).astype(np.float64)
n, l, e, s = rd("n"), rd("l"), rd("e"), rd("s")
if not (n.size == l.size == e.size == s.size == 64 ** 3):
    print(f"FAIL: grid sizes {n.size}, {l.size}, {e.size}, {s.size} (not 64^3)"); sys.exit(1)
MEAN_TOL = 1e-4
dn, dl, ds = n.mean() - 1, l.mean() - 1, s.mean() - 1
check("A1", abs(dn) < MEAN_TOL and abs(ds) < MEAN_TOL and abs(dl) > MEAN_TOL,
      f"box mean - 1: {dn:+.2e} (default), {ds:+.2e} (-s 20), both < {MEAN_TOL:.0e}; the old realization "
      f"{dl:+.2e} fails it, as it must (exact {e.mean() - 1:+.1e})")
rn = float(np.sqrt(((n - e) ** 2).mean()) / e.mean())
rl = float(np.sqrt(((l - e) ** 2).mean()) / e.mean())
check("A2", rn <= rl and rn < 0.07, f"rms(a - exact)/<exact>: {rn:.4f} (old realization {rl:.4f}; < 0.07)")
worst = []
for name, what in (("p", "--partition 2 2 2"), ("t", "OMP_NUM_THREADS=1"), ("d", "--padding 20")):
    x = rd(name)
    relc = np.abs(x - n) / np.maximum(np.abs(n), 1e-30)
    nbad = int((relc > 1e-4).sum())
    dsum = abs(float(x.sum() - n.sum())) / float(n.sum())
    worst.append(f"{what} {relc.max():.1e}")
    check("A3", nbad <= 1e-4 * n.size and dsum < 1e-6,
          f"{what} against the default run: {nbad} cell(s) differ by more than 1e-4 (<= {int(1e-4 * n.size)}; "
          f"the median cell {np.median(relc):.0e}, the largest {relc.max():.1e}), box sums {dsum:.0e} apart (< 1e-6)")
nv, ev = rd("n", "a_vel").reshape(-1, 3), rd("e", "a_vel").reshape(-1, 3)
mom = float(np.abs(nv.sum(0) - ev.sum(0)).max() / np.abs(ev).sum(0).max())
check("A4", mom < 1e-4, f"box sum of '.a_vel' against --exact-average: {mom:.1e} of sum|v| (< 1e-4)")
if gpu:
    g = rd("g")
    dg = g.mean() - 1
    scale = float(np.abs(n).max())
    meanrel = abs(float(g.mean() - n.mean())) / scale
    frac = float((np.abs(g - n) > 1e-3 * scale).sum()) / n.size
    check("A5", abs(dg) < MEAN_TOL and meanrel < 1e-5 and frac < 1e-3,
          f"--gpu: box mean - 1 {dg:+.2e}; against the CPU: mean {meanrel:.1e} of the peak (< 1e-5), "
          f"cells > 1e-3 of the peak apart {frac:.1e} (< 1e-3)")
print("RESULT:", "PASS  (the Monte-Carlo '_a' average conserves the mass and does not depend on the split)"
      if not fails else f"FAIL ({fails} check(s))")
sys.exit(1 if fails else 0)
PYEOF
