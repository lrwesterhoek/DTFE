#!/usr/bin/env bash
# Chunk-level checkpoints inside the GPU deposit and its cause-aware retry policy (2026-10-11,
# src/CGAL_triangulation/ps_metal_host.cc; ported to the CUDA/HIP host ps_gpu_cuda.cu the same day, where a
# 'buffer' below is one chunk kernel and the snapshot a device-to-host copy). A failed command buffer no longer redoes the whole partition
# deposit: the pass restores its last snapshot (its start, or a checkpoint every DTFE_GPU_CHECKPOINT_CHUNKS
# buffers) and continues from there. The test hook DTFE_GPU_FAULT_AT=<k>[:<kind>] lets the k-th command
# buffer of the process run to completion -- its atomics ARE in the grids -- and then reports it as failed,
# so only a correct restore keeps the grids exact. Asserted on a synthetic snapshot (pinned chunk sizes,
# PS_METAL_CHUNK=2000, and a checkpoint every 2 buffers, so the fault lands after a checkpoint):
#   R1  one-thread-per-tet deposit, an 'innocent victim' fault at buffer 3: the run stays on the GPU, logs
#       the resume from the checkpoint at the same pace, its '.a_den' (the exact fixed-point mass grid) is
#       byte-identical to a run without the fault, '.a_streams' identical, and so is '.a_vel' since 2026-10-14:
#       the moment grids are exact fixed-point sums too (it was a float atomic sum, 4e-7 apart run to run, held
#       to 1e-6 of the peak -- a tolerance that flaked under GPU load). PS_METAL_FLOAT_MOMENTS=1 (or
#       PS_GPU_FLOAT_MOMENTS=1) in the environment brings back the float sums and that 1e-6 tolerance
#   R2  --ps-exact-deposit (the items passes + leftovers), an 'oom' fault: same assertions, and the pace
#       escalates (shorter buffers) as a resource error should; its '.a_streams' (the exact multiplicity) is a
#       fixed-point sum since 2026-10-14 as well: identical
#   R3  --partition 2 2 2 --max-concurrent 1, a 'timeout' (watchdog) fault in a later partition: same
#   R4  a 'persist' fault (every buffer from the k-th on fails): the policy gives the partition to the CPU
#       after 3 failures of the same chunk, and the run still finishes
#   R5  the failed buffers count into the run's retry counter (the progress sidecar's gpu_retries reads the
#       same counter, src/run_progress.h; the sidecar itself is removed when a run succeeds, so the count is
#       read from the deposit's summary line)
#   R6  the same counts by cause (G1, 2026-10-11): each run's log ends its computation with 'GPU deposit
#       retries: N (innocent a, timeout b, oom c, other d)' -- R1 innocent 1, R2 oom 1, R3 timeout 1, R4
#       innocent 3 -- and a fault-free run prints no such line (the progress sidecar's gpu_retries_by_cause
#       reads the same counters)
#   R7  the escalated pace recovers (2026-10-11): the same watchdog fault as R3 with DTFE_GPU_PACE_RECOVER_SEC=0
#       (the clean GPU work after which a notch is undone; 90 s by default) steps the pace back to the default
#       250 ms buffers / 8 ms gaps at the next clean buffer, and the grids stay exact; R3 itself (the default
#       90 s, a ~1 s run) keeps the escalation for the rest of its deposits. The pace is process-wide, so the
#       later partitions start from the eased pace -- inferred from the log line, not observed with a second
#       injected fault. The CUDA/HIP host has the same rule since 2026-10-11 (ps_gpu_cuda.cu), from its own
#       pace: 250 ms kernels with no gap, escalated to 62 ms / 8 ms, eased back to 250 ms / 0 ms
# GPU builds only: Metal, and the CUDA/HIP port (CUDAEMU=1 runs it on the CPU emulation, tests/cudaemu_check.sh):
# skipped (exit 0) otherwise, a local GPU suite like tests/gpu_mass_check.sh (not in tests/ci_suite.sh). ~20 s
# on Metal (M1 Max), ~18 s under the emulation.
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
gpu_stamp=""                                              # any one GPU backend stamp will do
for m in metal cudaemu cuda hip; do [ -f "$OBJ/.gpu_mode_$m" ] && gpu_stamp="$m"; done
if [ -z "$gpu_stamp" ]; then echo "SKIP: $OBJ is neither a Metal nor a CUDA/HIP (or CUDAEMU) build"; exit 0; fi
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
echo ">> R7: the same watchdog fault, then the pace recovers after a clean stretch (set to 0 s)"
run "$TMP/p2" env DTFE_GPU_FAULT_AT=9:timeout DTFE_GPU_PACE_RECOVER_SEC=0 "${PS_BIN}" "$SNAP" "$TMP/p2" "${C[@]}" "${P[@]}"
echo ">> R4: a persistent fault from buffer 3 on: the partition goes to the CPU"
ALLOW_CPU=1 run "$TMP/x1" env DTFE_GPU_FAULT_AT=3:persist "${PS_BIN}" "$SNAP" "$TMP/x1" "${C[@]}"
GPU_STAMP="$gpu_stamp" "$PY" - "$TMP" <<'PYEOF'
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
def failed_buffers(text):   # Metal: buffers, CUDA/HIP: kernels
    m = re.findall(r"after (\d+) failed (?:buffer|kernel)\(s\)", text)
    return sum(int(x) for x in m)
for tag, ref, run, what, cause, pace in (
        ("R1", "t0", "t1", "per-tet deposit", "innocent victim", "250 ms"),
        ("R2", "e0", "e1", "--ps-exact-deposit", "resource", "62 ms"),
        ("R3", "p0", "p1", "--partition 2 2 2", "watchdog", None)):
    L = log(run)
    resumed = re.search(r"\[" + cause + r"\] -- resuming from the (checkpoint|pass start) at thread", L)
    check(tag, resumed is not None and "recovered on the GPU" in L,
          f"{what}: the fault is classified as '{cause}' and the pass resumes on the GPU"
          + (f" (from the {resumed.group(1)})" if resumed else ""))
    if pace:
        check(tag, re.search(f"target {pace} (buffers|kernels)", L) is not None,
              f"{what}: the retry runs at {pace} ({'same pace' if cause == 'innocent victim' else 'escalated'})")
    x, y = (tmp / f"{ref}.a_den").read_bytes(), (tmp / f"{run}.a_den").read_bytes()
    check(tag, len(x) > 0 and x == y, f"{what}: '.a_den' byte-identical to the run without the fault ({len(x)} bytes)")
    # the stream count is an integer atomic, and since 2026-10-14 the velocity moment and the exact deposit's
    # multiplicity are exact fixed-point sums: all identical (with the float sums of before, under
    # PS_METAL_FLOAT_MOMENTS=1, only to 1e-6 of the peak -- they differed run to run by ~3e-7)
    floatMoments = any(_os.environ.get(k, "0") not in ("", "0") for k in ("PS_METAL_FLOAT_MOMENTS", "PS_GPU_FLOAT_MOMENTS"))
    for ext, exact in (("a_streams", not (floatMoments and tag == "R2")), ("a_vel", not floatMoments)):
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
def by_cause(name):   # the last 'GPU deposit retries:' line of the run, as (N, innocent, timeout, oom, other)
    m = re.findall(r"GPU deposit retries: (\d+) \(innocent (\d+), timeout (\d+), oom (\d+), other (\d+)\)", log(name))
    return tuple(int(x) for x in m[-1]) if m else None
for name, want in (("t0", None), ("t1", (1, 1, 0, 0, 0)), ("e1", (1, 0, 0, 1, 0)),
                   ("p1", (1, 0, 1, 0, 0)), ("x1", (3, 3, 0, 0, 0))):
    got = by_cause(name)
    check("R6", got == want, f"{name}: retries by cause {got} (expected {want})")
# each host from its own default pace: Metal 250 ms buffers / 8 ms gaps, CUDA/HIP 250 ms kernels / no gap
metal = _os.environ.get("GPU_STAMP") == "metal"
esc_gap, def_gap = ("24", "8") if metal else ("8", "0")
L3, L7 = log("p1"), log("p2")
eased = re.findall(r"pace eased to (\d+) ms (?:buffers|kernels), (\d+) ms gaps", L7)
check("R7", re.search(r"\[watchdog\].*target 62 ms (?:buffers|kernels), " + esc_gap + r" ms gaps", L7) is not None
      and eased[-1:] == [("250", def_gap)] and "(the default)" in L7,
      f"a watchdog fault escalates to 62 ms / {esc_gap} ms and a clean stretch eases it back to {eased[-1:]} "
      f"(expected 250 ms / {def_gap} ms)")
check("R7", "pace eased" not in L3,
      "with the default 90 s recovery interval the escalation holds through the rest of a short run (R3)")
x, y = (tmp / "p0.a_den").read_bytes(), (tmp / "p2.a_den").read_bytes()
check("R7", x == y, f"'.a_den' byte-identical to the run without the fault after the pace recovered")
print("RESULT:", "PASS  (a failed GPU buffer resumes from its checkpoint with exact grids, by cause, and "
      "only a persistent failure reaches the CPU)" if not fails else f"FAIL ({fails} check(s))")
sys.exit(1 if fails else 0)
PYEOF
