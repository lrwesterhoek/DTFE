#!/usr/bin/env bash
# The Metal GPU deposit's MASS grid (2026-10-08): an exact 64-bit fixed-point sum (two atomic words per cell,
# metal/ps_deposit.metal depositMass) in place of the float atomic add, which lost small shares downward and
# depended on the atomic order (TNG50-3-Dark z=0: box mean 0.99941). Asserted on synthetic snapshots:
#   M1  two identical --ps-gpu runs write byte-identical '.a_den', on one domain and with --partition 2 2 2
#       --max-concurrent 1 (the float grid differed run to run), and since 2026-10-14 byte-identical '.a_vel'
#       too: the moment grids are exact fixed-point sums as well (tests/ps_fixed_moments_check.sh covers every
#       grid; PS_METAL_FLOAT_MOMENTS=1 brings back their float sums, and this check then skips '.a_vel').
#       One partition at a time: concurrent partitions were
#       merged in completion order (DTFE.cpp, unless the auto-tuner's speed split ordered them; index order is the
#       default since 2026-10-14, '--completion-order-merge' the old way), and those float
#       sums differed in the last bit from run to run on the CPU as well (--max-concurrent 4: 259 of 32^3 cells
#       at 1e-16 in the double build), which is not the deposit's doing; the pin is kept so the check does not
#       depend on the merge
#   M2  PS_METAL_FLOAT_MASS=1 (the old float grid, kept for A/B) still runs and agrees with the default to 1e-5
#       of the peak
#   M3  heavy accumulation (a 128^3 pancake on a 2^3 grid, nSub 1: 1.47e7 tetrahedra, ~1.8e6 deposits per
#       cell): the GPU box mean equals the CPU's to 3e-6 (measured 8.7e-7 against the single-precision CPU,
#       1.5e-8 against the double one; the float grid gave 7.8e-6 on the same case, so the old grid fails here)
#   M4  the run log splits 'field interpolation' into its phases (2026-10-11: extraction walk, cost sort, GPU
#       deposit, deferred tets on the CPU, normalization), with the deferred-tet count equal to the GPU deposit's
#       own count, and a partitioned run prints the run's totals over all 8 passes (the per-partition deferred
#       line is suppressed there, so these totals are the only place the deferred work shows)
#   M5  many GPU-deferred tetrahedra (48^3 crossed waves, A k = 3, on 96^3 at nSub 3: ~12000 of them) are
#       deposited on several CPU threads in slab mode (2026-10-11; they were one thread, which outlasted the GPU
#       deposit on such data), and the grids are byte-identical to the one-thread loop (DTFE_DEPOSIT_SLABS=0)
# GPU builds only: Metal, and the CUDA/HIP port since it has the fixed-point grid too (2026-10-11; CUDAEMU=1 runs it
# on the CPU emulation): skipped (exit 0) otherwise,
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
gpu_stamp=""                                              # any one GPU backend stamp will do
for m in metal cudaemu cuda hip; do [ -f "$OBJ/.gpu_mode_$m" ] && gpu_stamp="$m"; done
if [ -z "$gpu_stamp" ]; then echo "SKIP: $OBJ is neither a Metal nor a CUDA/HIP (or CUDAEMU) build"; exit 0; fi
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/gpu_mass_check"
mkdir -p "$TMP"
SNAP="$TMP/waves.hdf5"; PAN="$TMP/pancake128.hdf5"; DEF="$TMP/waves_deferred.hdf5"
[ -f "$SNAP" ] || "$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 48 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
[ -f "$PAN" ] || "$PY" tests/generate_ps_test_data.py --out "$PAN" --n 128 >/dev/null || { echo "FAIL data gen"; exit 1; }
[ -f "$DEF" ] || "$PY" tests/generate_ps_test_data.py --out "$DEF" --n 48 --crossed-waves --amplitude-factor 3.0 \
    --seed 9 >/dev/null || { echo "FAIL data gen"; exit 1; }
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
echo ">> M5: the GPU-deferred tetrahedra on 4 CPU threads (slab mode) and on one"
D=(--grid 96 --periodic --input 105 --MpcUnit 1 --verbose 2 --field density_a velocity_a --avg-subsamples 3 --ps-gpu
   --partition 1 1 1)
run "$TMP/d1" env DTFE_DEPOSIT_THREADS=4 "${PS_BIN}" "$DEF" "$TMP/d1" "${D[@]}"
run "$TMP/d2" env DTFE_DEPOSIT_THREADS=4 DTFE_DEPOSIT_SLABS=0 "${PS_BIN}" "$DEF" "$TMP/d2" "${D[@]}"
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
floatMoments = any(_os.environ.get(k, "0") not in ("", "0") for k in ("PS_METAL_FLOAT_MOMENTS", "PS_GPU_FLOAT_MOMENTS"))
for a, b, what in (("s1", "s2", "one domain"), ("p1", "p2", "--partition 2 2 2 --max-concurrent 1")):
    for ext in ("a_den",) + (() if floatMoments else ("a_vel",)):
        x, y = (tmp / f"{a}.{ext}").read_bytes(), (tmp / f"{b}.{ext}").read_bytes()
        check("M1", len(x) > 0 and x == y, f"two --ps-gpu runs ({what}): '.{ext}' byte-identical ({len(x)} bytes)")
s, f = rd("s1"), rd("f1")
peak = float(np.abs(s - f).max() / np.abs(s).max())
check("M2", s.size == f.size and peak < 1e-5, f"PS_METAL_FLOAT_MASS=1 against the fixed-point grid: peak-rel {peak:.1e} (< 1e-5)")
import re
s1log, p1log = (tmp / "s1.log").read_text(), (tmp / "p1.log").read_text()
ph = re.search(r"field interpolation phases: .*?extraction walk [0-9.]+ s, cost sort [0-9.]+ s, GPU deposit [0-9.]+ s"
               r".*?deferred tets on the CPU [0-9.]+ s \((\d+) tets\)", s1log)
gd = re.search(r"GPU deposit \(.*?\), \d+ tetrahedra \((\d+) with a sample on a face", s1log)
check("M4", ph is not None and gd is not None and ph.group(1) == gd.group(1)
      and re.search(r"run total over 8 pass\(es\): .*GPU deposit [0-9.]+ s", p1log) is not None,
      f"phase timers in the run log (one domain: {ph.group(1) + ' deferred tets' if ph else 'MISSING'}, "
      f"GPU deposit says {gd.group(1) if gd else '?'}; 2 x 2 x 2: run totals over 8 passes)")
d1log, d2log = (tmp / "d1.log").read_text(), (tmp / "d2.log").read_text()
m1 = re.search(r"deposited the (\d+) GPU-deferred tetrahedra on the CPU in [0-9.e+-]+ s \((\d+) threads?\)", d1log)
m2 = re.search(r"deposited the (\d+) GPU-deferred tetrahedra on the CPU in [0-9.e+-]+ s \((\d+) threads?\)", d2log)
same = all((tmp / f"d1.{e}").read_bytes() == (tmp / f"d2.{e}").read_bytes() for e in ("a_den", "a_streams", "a_hidden_streams", "a_vel"))
check("M5", m1 is not None and m2 is not None and int(m1.group(1)) >= 2048 and m1.group(1) == m2.group(1)
      and int(m1.group(2)) > 1 and int(m2.group(2)) == 1 and same,
      f"GPU-deferred tets on the CPU: {m1.group(1) if m1 else '?'} tets on {m1.group(2) if m1 else '?'} threads vs "
      f"{m2.group(2) if m2 else '?'} (DTFE_DEPOSIT_SLABS=0); '.a_den', '.a_streams', '.a_hidden_streams', '.a_vel' "
      f"{'byte-identical' if same else 'DIFFER'}")
c, g = rd("h_cpu"), rd("h_gpu")
rel = abs(float(g.mean()) / float(c.mean()) - 1)
check("M3", c.size == 8 and rel < 3e-6, f"heavy accumulation: GPU box mean {g.mean():.9f}, CPU {c.mean():.9f}, |ratio - 1| {rel:.1e} (< 3e-6)")
print("RESULT:", "PASS  (the GPU mass grid is reproducible run to run, the float A/B grid agrees, and a heavy "
      "accumulation keeps the CPU's box mean)" if not fails else f"FAIL ({fails} check(s))")
sys.exit(1 if fails else 0)
PYEOF
