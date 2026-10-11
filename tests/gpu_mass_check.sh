#!/usr/bin/env bash
# The Metal GPU deposit's MASS grid (2026-10-08): an exact 64-bit fixed-point sum (two atomic words per cell,
# metal/ps_deposit.metal depositMass) in place of the float atomic add, which lost small shares downward and
# depended on the atomic order (TNG50-3-Dark z=0: box mean 0.99941). Asserted on synthetic snapshots:
#   M1  two identical --ps-gpu runs write byte-identical '.a_den', on one domain and with --partition 2 2 2
#       --max-concurrent 1 (the float grid differed run to run; the moment grids stay float, so '.a_vel' may
#       still differ between runs and is not compared). One partition at a time: concurrent partitions are
#       merged in completion order (DTFE.cpp, unless the auto-tuner's speed split orders them), and those float
#       sums differ in the last bit from run to run on the CPU as well (--max-concurrent 4: 259 of 32^3 cells
#       at 1e-16 in the double build), which is not the deposit's doing
#   M2  PS_METAL_FLOAT_MASS=1 (the old float grid, kept for A/B) still runs and agrees with the default to 1e-5
#       of the peak
#   M3  heavy accumulation (a 128^3 pancake on a 2^3 grid, nSub 1: 1.47e7 tetrahedra, ~1.8e6 deposits per
#       cell): the GPU box mean equals the CPU's to 3e-6 (measured 8.7e-7 against the single-precision CPU,
#       1.5e-8 against the double one; the float grid gave 7.8e-6 on the same case, so the old grid fails here)
# GPU builds only, Metal only (the CUDA/HIP port still accumulates the mass in float): skipped (exit 0) otherwise,
# like tests/gpu_scalar_check.sh, and a local GPU suite (not in tests/ci_suite.sh). M3 takes ~40 s.
# Usage: tests/gpu_mass_check.sh [--no-build]   (--no-build is accepted and ignored)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
OBJ=o_ps; precision_double && OBJ=o_ps_d
if [ ! -d "$OBJ" ] || [ -f "$OBJ/.gpu_mode_off" ]; then echo "SKIP: $OBJ carries no GPU stamp (build with METAL=1)"; exit 0; fi
if [ ! -f "$OBJ/.gpu_mode_metal" ]; then echo "SKIP: $OBJ is not a Metal build (the CUDA/HIP port still sums the mass in float)"; exit 0; fi
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/gpu_mass_check"
mkdir -p "$TMP"
SNAP="$TMP/waves.hdf5"; PAN="$TMP/pancake128.hdf5"
[ -f "$SNAP" ] || "$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 48 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
[ -f "$PAN" ] || "$PY" tests/generate_ps_test_data.py --out "$PAN" --n 128 >/dev/null || { echo "FAIL data gen"; exit 1; }
# run <out> <command...>: a non-zero exit or a CPU fallback (any of the deposit's fallback warnings, and the
# aggregate count of the partition passes, 2026-10-08) fails the suite
run() { local out="$1"; shift; "$@" > "$out.log" 2>&1 || { echo "FAIL: $(basename "$out") (see $out.log)"; exit 1; }
        grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead" "$out.log" \
            && { echo "FAIL: $(basename "$out") fell back to the CPU (see $out.log)"; exit 1; }; return 0; }
C=(--grid 32 --periodic --input 105 --MpcUnit 1 --verbose 2 --field density_a velocity_a --avg-subsamples 2 --ps-gpu)
P=(--partition 2 2 2 --max-concurrent 1)
echo ">> M1/M2: --ps-gpu twice (one domain, 2 x 2 x 2 partitions), and once with PS_METAL_FLOAT_MASS=1"
run "$TMP/s1" "${PS_BIN}" "$SNAP" "$TMP/s1" "${C[@]}"
grep -q " GPU deposit (" "$TMP/s1.log" || { echo "FAIL: s1 did not run the GPU deposit (see $TMP/s1.log)"; exit 1; }
run "$TMP/s2" "${PS_BIN}" "$SNAP" "$TMP/s2" "${C[@]}"
run "$TMP/p1" "${PS_BIN}" "$SNAP" "$TMP/p1" "${C[@]}" "${P[@]}"
run "$TMP/p2" "${PS_BIN}" "$SNAP" "$TMP/p2" "${C[@]}" "${P[@]}"
run "$TMP/f1" env PS_METAL_FLOAT_MASS=1 "${PS_BIN}" "$SNAP" "$TMP/f1" "${C[@]}"
echo ">> M3: heavy accumulation, 128^3 pancake on a 2^3 grid, CPU and GPU"
H=(--grid 2 --periodic --MpcUnit 1 --verbose 2 --field density_a --avg-subsamples 1 --partition 1 1 1)
run "$TMP/h_cpu" "${PS_BIN}" "$PAN" "$TMP/h_cpu" "${H[@]}"
run "$TMP/h_gpu" "${PS_BIN}" "$PAN" "$TMP/h_gpu" "${H[@]}" --ps-gpu
grep -q " GPU deposit (" "$TMP/h_gpu.log" || { echo "FAIL: h_gpu did not run the GPU deposit (see $TMP/h_gpu.log)"; exit 1; }
"$PY" - "$TMP" <<'PYEOF'
import numpy as np, sys
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
from pathlib import Path
tmp = Path(sys.argv[1]); fails = 0
def check(tag, ok, msg):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {tag} {msg}")
def rd(name):
    return np.fromfile(tmp / f"{name}.a_den", dtype=REAL).astype(np.float64)
for a, b, what in (("s1", "s2", "one domain"), ("p1", "p2", "--partition 2 2 2 --max-concurrent 1")):
    x, y = (tmp / f"{a}.a_den").read_bytes(), (tmp / f"{b}.a_den").read_bytes()
    check("M1", len(x) > 0 and x == y, f"two --ps-gpu runs ({what}): '.a_den' byte-identical ({len(x)} bytes)")
s, f = rd("s1"), rd("f1")
peak = float(np.abs(s - f).max() / np.abs(s).max())
check("M2", s.size == f.size and peak < 1e-5, f"PS_METAL_FLOAT_MASS=1 against the fixed-point grid: peak-rel {peak:.1e} (< 1e-5)")
c, g = rd("h_cpu"), rd("h_gpu")
rel = abs(float(g.mean()) / float(c.mean()) - 1)
check("M3", c.size == 8 and rel < 3e-6, f"heavy accumulation: GPU box mean {g.mean():.9f}, CPU {c.mean():.9f}, |ratio - 1| {rel:.1e} (< 3e-6)")
print("RESULT:", "PASS  (the GPU mass grid is reproducible run to run, the float A/B grid agrees, and a heavy "
      "accumulation keeps the CPU's box mean)" if not fails else f"FAIL ({fails} check(s))")
sys.exit(1 if fails else 0)
PYEOF
