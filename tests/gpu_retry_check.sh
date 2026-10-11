#!/usr/bin/env bash
# Chunk-level checkpoints inside the Metal GPU deposit and its cause-aware retry policy (2026-10-11,
# src/CGAL_triangulation/ps_metal_host.cc). A failed command buffer no longer redoes the whole partition
# deposit: the pass restores its last snapshot (its start, or a checkpoint every DTFE_GPU_CHECKPOINT_CHUNKS
# buffers) and continues from there. The test hook DTFE_GPU_FAULT_AT=<k>[:<kind>] lets the k-th command
# buffer of the process run to completion -- its atomics ARE in the grids -- and then reports it as failed,
# so only a correct restore keeps the grids exact. Asserted on a synthetic snapshot (pinned chunk sizes,
# PS_METAL_CHUNK=2000, and a checkpoint every 2 buffers, so the fault lands after a checkpoint):
#   R1  one-thread-per-tet deposit, an 'innocent victim' fault at buffer 3: the run stays on the GPU, logs
#       the resume from the checkpoint at the same pace, its '.a_den' (the exact fixed-point mass grid) is
#       byte-identical to a run without the fault, '.a_streams' identical, '.a_vel' (float atomics, whose order
#       differs run to run anyway: 4e-7) agrees to 1e-6 of the peak
#   R2  --ps-exact-deposit (the items passes + leftovers), an 'oom' fault: same assertions, and the pace
#       escalates (shorter buffers) as a resource error should; its '.a_streams' is a float sum here: 1e-6
#   R3  --partition 2 2 2 --max-concurrent 1, a 'timeout' (watchdog) fault in a later partition: same
#   R4  a 'persist' fault (every buffer from the k-th on fails): the policy gives the partition to the CPU
#       after 3 failures of the same chunk, and the run still finishes
#   R5  the failed buffers count into the run's retry counter (the progress sidecar's gpu_retries reads the
#       same counter, src/run_progress.h; the sidecar itself is removed when a run succeeds, so the count is
#       read from the deposit's summary line)
# Metal builds only (the CUDA/HIP port keeps the from-scratch retry): skipped (exit 0) otherwise, a local GPU
# suite like tests/gpu_mass_check.sh (not in tests/ci_suite.sh). ~30 s.
# Usage: tests/gpu_retry_check.sh [--no-build]   (--no-build is accepted and ignored)
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
if [ ! -f "$OBJ/.gpu_mode_metal" ]; then echo "SKIP: $OBJ is not a Metal build (the CUDA/HIP port keeps the from-scratch retry)"; exit 0; fi
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/gpu_retry_check"
mkdir -p "$TMP"
SNAP="$TMP/waves.hdf5"
[ -f "$SNAP" ] || "$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 48 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
# run <out> <command...>: a non-zero exit fails the suite; so does a CPU fallback unless ALLOW_CPU=1
run() { local out="$1"; shift; "$@" > "$out.log" 2>&1 || { echo "FAIL: $(basename "$out") (see $out.log)"; exit 1; }
        [ "${ALLOW_CPU:-0}" = "1" ] && return 0
        grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead" "$out.log" \
            && { echo "FAIL: $(basename "$out") fell back to the CPU (see $out.log)"; exit 1; }; return 0; }
# pinned chunking: the controller rescales by elapsed time, so only the opening chunk and the checkpoint
# interval are fixed; the fault at buffer 3 then lands after the checkpoint at buffer 2
export PS_METAL_CHUNK=2000 DTFE_GPU_CHECKPOINT_CHUNKS=2
unset DTFE_GPU_FAULT_AT
C=(--grid 32 --periodic --input 105 --MpcUnit 1 --verbose 2 --field density_a velocity_a --avg-subsamples 2 --ps-gpu)
P=(--partition 2 2 2 --max-concurrent 1)
echo ">> R1: one-thread-per-tet deposit, without and with an innocent-victim fault at buffer 3"
run "$TMP/t0" "${PS_BIN}" "$SNAP" "$TMP/t0" "${C[@]}"
grep -q " GPU deposit (" "$TMP/t0.log" || { echo "FAIL: t0 did not run the GPU deposit (see $TMP/t0.log)"; exit 1; }
run "$TMP/t1" env DTFE_GPU_FAULT_AT=3 "${PS_BIN}" "$SNAP" "$TMP/t1" "${C[@]}"
echo ">> R2: --ps-exact-deposit, without and with an out-of-memory fault at buffer 3"
run "$TMP/e0" "${PS_BIN}" "$SNAP" "$TMP/e0" "${C[@]}" --ps-exact-deposit
run "$TMP/e1" env DTFE_GPU_FAULT_AT=3:oom "${PS_BIN}" "$SNAP" "$TMP/e1" "${C[@]}" --ps-exact-deposit
echo ">> R3: 2 x 2 x 2 partitions, without and with a watchdog fault at buffer 9"
run "$TMP/p0" "${PS_BIN}" "$SNAP" "$TMP/p0" "${C[@]}" "${P[@]}"
run "$TMP/p1" env DTFE_GPU_FAULT_AT=9:timeout "${PS_BIN}" "$SNAP" "$TMP/p1" "${C[@]}" "${P[@]}"
echo ">> R4: a persistent fault from buffer 3 on: the partition goes to the CPU"
ALLOW_CPU=1 run "$TMP/x1" env DTFE_GPU_FAULT_AT=3:persist "${PS_BIN}" "$SNAP" "$TMP/x1" "${C[@]}"
"$PY" - "$TMP" <<'PYEOF'
import numpy as np, re, sys
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
from pathlib import Path
tmp = Path(sys.argv[1]); fails = 0
def check(tag, ok, msg):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {tag} {msg}")
def log(name):
    return (tmp / f"{name}.log").read_text(errors="replace")
def rd(name, ext):
    return np.fromfile(tmp / f"{name}.{ext}", dtype=REAL).astype(np.float64)
def failed_buffers(text):
    m = re.findall(r"after (\d+) failed buffer\(s\)", text)
    return sum(int(x) for x in m)
for tag, ref, run, what, cause, pace in (
        ("R1", "t0", "t1", "per-tet deposit", "innocent victim", "250 ms buffers"),
        ("R2", "e0", "e1", "--ps-exact-deposit", "resource", "62 ms buffers"),
        ("R3", "p0", "p1", "--partition 2 2 2", "watchdog", None)):
    L = log(run)
    resumed = re.search(r"\[" + cause + r"\] -- resuming from the (checkpoint|pass start) at thread", L)
    check(tag, resumed is not None and "recovered on the GPU" in L,
          f"{what}: the fault is classified as '{cause}' and the pass resumes on the GPU"
          + (f" (from the {resumed.group(1)})" if resumed else ""))
    if pace:
        check(tag, f"target {pace}" in L, f"{what}: the retry runs at {pace} ({'same pace' if cause == 'innocent victim' else 'escalated'})")
    x, y = (tmp / f"{ref}.a_den").read_bytes(), (tmp / f"{run}.a_den").read_bytes()
    check(tag, len(x) > 0 and x == y, f"{what}: '.a_den' byte-identical to the run without the fault ({len(x)} bytes)")
    # the stream count is an integer atomic, except under --ps-exact-deposit (a float sum of volume fractions,
    # as order-dependent as the moments: two fault-free runs differ by 3e-7 of the peak)
    for ext, exact in (("a_streams", tag != "R2"), ("a_vel", False)):
        a, b = rd(ref, ext), rd(run, ext)
        if exact:
            check(tag, np.array_equal(a, b), f"{what}: '.{ext}' identical")
            continue
        peak = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))
        check(tag, a.size == b.size and peak < 1e-6, f"{what}: '.{ext}' peak-rel {peak:.1e} (< 1e-6)")
check("R1", "resuming from the checkpoint at thread" in log("t1"),
      "the retry starts from a periodic checkpoint, not from the pass start")
L = log("x1")
check("R4", "3 failures of the same chunk" in L and "gave up" in L
      and re.search(r"using the CPU deposit|deposit unavailable|ran on the CPU instead", L) is not None,
      "persistent fault: 3 failures of the same chunk, then the CPU deposit")
d0, d1 = rd("t0", "a_den"), rd("x1", "a_den")
rel = float(np.abs(d0 - d1).max() / np.abs(d0).max())
check("R4", rel < 1e-4, f"persistent fault: the CPU's '.a_den' agrees with the GPU's to {rel:.1e} of the peak (< 1e-4)")
n1, n3, nx = failed_buffers(log("t1")), failed_buffers(log("p1")), failed_buffers(log("x1"))
check("R5", (n1, n3, nx) == (1, 1, 3), f"failed buffers counted: {n1}, {n3}, {nx} (expected 1, 1, 3)")
print("RESULT:", "PASS  (a failed GPU buffer resumes from its checkpoint with exact grids, by cause, and "
      "only a persistent failure reaches the CPU)" if not fails else f"FAIL ({fails} check(s))")
sys.exit(1 if fails else 0)
PYEOF
