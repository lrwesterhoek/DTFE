#!/usr/bin/env bash
# The standard-DTFE GPU deposit ('--gpu', ./DTFE): the PS deposit's chunk checkpoints and
# cause-aware retry (src/CGAL_triangulation/gpu_metal_retry.h / gpu_cuda_retry.cuh, shared with the PS hosts;
# the pass loop is runBatch in dtfe_metal_host.cc / dtfe_gpu_cuda.cu), and its exact sums -- the '_a' grids as
# 64-bit fixed-point sums, metal/dtfe_deposit.metal's header -- which make two runs byte-identical (with float
# atomics two fault-free runs differ in 26-49% of the cells, up to 1e-6). tests/gpu_retry_check.sh is the PS twin of
# this suite and its fault hook is the same: DTFE_GPU_FAULT_AT=<k>[:innocent|timeout|oom|persist] lets the k-th command
# buffer of the process (a CUDA/HIP kernel) run to completion -- its atomics ARE in the grids -- and then reports it as
# failed, so only a correct restore keeps the grids exact. Asserted on a synthetic snapshot with pinned opening chunks
# (DTFE_METAL_CHUNK / DTFE_GPU_CHUNK 4096) and a checkpoint every 2 buffers:
#   E1  exact sums: two fault-free runs with different opening chunks (4096 and 65536 work items) write
#       byte-identical '.a_den', '.a_vel', '.a_velGrad' -- the item kernel over 9 sub-domains whose batches race
#       for the device, the per-tetrahedron kernel (DTFE_GPU_ITEMS=0), and --exact-average
#   E2  DTFE_METAL_FLOAT_GRIDS=1 (DTFE_GPU_FLOAT_GRIDS=1; the float atomics, kept for A/B) runs the float sums
#       and agrees with the default to 1e-5 of the peak; the default agrees with the CPU's box mean to 1e-5 of
#       the peak (tests/dtfe_metal_check.sh is the full CPU parity suite)
#   R1  the item kernel in one batch (--max-concurrent 1), an 'innocent victim' fault at buffer 3: the batch
#       resumes from the checkpoint at the same pace (250 ms) and its three grids are byte-identical to E1's
#       fault-free one-batch run
#   R2  --exact-average, an 'oom' fault: classified 'resource', the pace escalates (62 ms), grids byte-identical
#   R3  9 sub-domains under --partition 2 2 2, a 'timeout' (watchdog) fault at buffer 9: byte-identical
#   R4  a 'persist' fault (every buffer from the 3rd on fails) under --exact-average: the sub-domain goes to the
#       CPU after 3 failures of the same chunk, the run says how many sub-domains ran on the CPU, and the CPU's
#       '.a_den' agrees with the GPU's to 1e-4 of the peak
#   R5  the failed buffers are counted: 1, 1, 3 (the deposit's summary lines)
#   R6  the run's closing 'GPU deposit retries: N (innocent a, timeout b, oom c, other d)' -- the line the
#       launcher's Runs tab reads; the progress sidecar's gpu_retries_by_cause has the same counters -- R1 innocent
#       1, R2 oom 1, R3 timeout 1, R4 innocent 3, and none for a fault-free run
#   R7  the escalated pace eases back after a clean stretch (DTFE_GPU_PACE_RECOVER_SEC=0) to the default 250 ms
#       buffers / 8 ms gaps (CUDA/HIP: kernels / no gap), grids byte-identical; R3's default 90 s holds it
#   S1  the opt-in shared full grid (DTFE_GPU_SHARED=1; float sums, no periodic checkpoints): an innocent fault
#       re-zeroes the batch's owner box and resumes from its start; the grids agree with a fault-free shared run
#       to 1e-5 of the peak
#   R8  the image-plane shape of the recipe's dtfe27planes stage: '--sample-points' with
#       '--exact-average' under --partition 2 2 2 (one triangulation per partition, DTFE_parallel's single-
#       triangulation branch) and a 'persist' fault, so every partition's grid falls back to the CPU: the fallback
#       runs the THREADED exact average (DTFE_DEPOSIT_THREADS=4: the log says so once per partition), and its
#       '.a_*' grids and '.pts_*' answers are byte-identical to a one-thread CPU run
#   N1  the non-finite guard of the exact sums (ps_deposit_params.h psFixedShareOk): DTFE_TEST_NAN_SHARE=50
#       makes about one tetrahedron in 50 hand the kernel NaN vertex densities; the run reports once how many density
#       shares it left out, and its '.a_den' has no NaN or infinity
# GPU builds only: Metal, or the CUDA/HIP port (CUDAEMU=1 runs it on the CPU emulation, tests/cudaemu_check.sh);
# skipped (exit 0) otherwise, a local GPU suite like tests/gpu_retry_check.sh (not in tests/ci_suite.sh). ~30 s on
# Metal (M1 Max), a few minutes under the emulation. DTFE_TEST_PRECISION=double runs ./DTFE-double.
# Usage: tests/dtfe_gpu_retry_check.sh [--no-build]   (--no-build is accepted and ignored)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${DTFE_BIN}"
[ -x "${DTFE_BIN}" ] || { echo "FAIL: ${DTFE_BIN} is not built"; exit 1; }
OBJ=o; precision_double && OBJ=o_d
require_gpu_backend "$OBJ"                                # sets gpu_stamp (tests/lib.sh)
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/dtfe_gpu_retry_check"
mkdir -p "$TMP"
SNAP="$TMP/waves.hdf5"
[ -f "$SNAP" ] || "$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 48 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
# run <out> <command...>: a non-zero exit fails the suite; so does a CPU fallback unless ALLOW_CPU=1
run() { local out="$1"; shift; "$@" > "$out.log" 2>&1 || { echo "FAIL: $(basename "$out") (see $out.log)"; exit 1; }
        grep -q " GPU (" "$out.log" || { echo "FAIL: $(basename "$out") did not attempt the GPU (see $out.log)"; exit 1; }
        [ "${ALLOW_CPU:-0}" = "1" ] && return 0
        grep -q "using the CPU interpolation\|ran on the CPU instead" "$out.log" \
            && { echo "FAIL: $(basename "$out") fell back to the CPU (see $out.log)"; exit 1; }; return 0; }
# pinned opening chunks (the controller rescales by elapsed time from there; every chunk starts at a multiple of
# 256 items, which the exact sums need) and a checkpoint every 2 buffers: the fault at buffer 3 then lands after
# the checkpoint at buffer 2
export DTFE_METAL_CHUNK=4096 DTFE_GPU_CHUNK=4096 DTFE_GPU_CHECKPOINT_CHUNKS=2
unset DTFE_GPU_FAULT_AT DTFE_GPU_SHARED DTFE_METAL_FLOAT_GRIDS DTFE_GPU_FLOAT_GRIDS DTFE_GPU_ITEMS
C=(--grid 32 --periodic --input 105 --MpcUnit 1 --verbose 2 --field density_a velocity_a gradient_a --partition 1 1 1)
ONE=(--max-concurrent 1)                 # one sub-domain: one batch, so 'buffer k' is a buffer of that batch
echo ">> E1: two fault-free runs with different opening chunks (item kernel over 9 sub-domains, per-tet kernel, exact)"
run "$TMP/a1" env DTFE_METAL_TIMING=1 DTFE_GPU_TIMING=1 "${DTFE_BIN}" "$SNAP" "$TMP/a1" "${C[@]}" --gpu
run "$TMP/a2" env DTFE_METAL_CHUNK=65536 DTFE_GPU_CHUNK=65536 "${DTFE_BIN}" "$SNAP" "$TMP/a2" "${C[@]}" --gpu
run "$TMP/k1" env DTFE_GPU_ITEMS=0 "${DTFE_BIN}" "$SNAP" "$TMP/k1" "${C[@]}" --gpu
run "$TMP/k2" env DTFE_GPU_ITEMS=0 DTFE_METAL_CHUNK=25088 DTFE_GPU_CHUNK=25088 "${DTFE_BIN}" "$SNAP" "$TMP/k2" "${C[@]}" --gpu
run "$TMP/x1" "${DTFE_BIN}" "$SNAP" "$TMP/x1" "${C[@]}" --gpu --exact-average
run "$TMP/x2" env DTFE_METAL_CHUNK=1024 DTFE_GPU_CHUNK=1024 "${DTFE_BIN}" "$SNAP" "$TMP/x2" "${C[@]}" --gpu --exact-average
echo ">> E2: the float sums (A/B) and the CPU"
run "$TMP/f1" env DTFE_METAL_FLOAT_GRIDS=1 DTFE_GPU_FLOAT_GRIDS=1 DTFE_METAL_TIMING=1 DTFE_GPU_TIMING=1 \
    "${DTFE_BIN}" "$SNAP" "$TMP/f1" "${C[@]}" --gpu
"${DTFE_BIN}" "$SNAP" "$TMP/c1" "${C[@]}" > "$TMP/c1.log" 2>&1 || { echo "FAIL: the CPU run (see $TMP/c1.log)"; exit 1; }
echo ">> R1: one batch, without and with an innocent-victim fault at buffer 3"
run "$TMP/t0" "${DTFE_BIN}" "$SNAP" "$TMP/t0" "${C[@]}" "${ONE[@]}" --gpu
run "$TMP/t1" env DTFE_GPU_FAULT_AT=3 "${DTFE_BIN}" "$SNAP" "$TMP/t1" "${C[@]}" "${ONE[@]}" --gpu
echo ">> R2: --exact-average, without and with an out-of-memory fault at buffer 3"
run "$TMP/e0" "${DTFE_BIN}" "$SNAP" "$TMP/e0" "${C[@]}" "${ONE[@]}" --gpu --exact-average
run "$TMP/e1" env DTFE_GPU_FAULT_AT=3:oom "${DTFE_BIN}" "$SNAP" "$TMP/e1" "${C[@]}" "${ONE[@]}" --gpu --exact-average
echo ">> R3: --partition 2 2 2 over 9 threads, without and with a watchdog fault at buffer 9"
P=(--grid 32 --periodic --input 105 --MpcUnit 1 --verbose 2 --field density_a velocity_a gradient_a --partition 2 2 2)
run "$TMP/p0" "${DTFE_BIN}" "$SNAP" "$TMP/p0" "${P[@]}" --gpu
run "$TMP/p1" env DTFE_GPU_FAULT_AT=9:timeout "${DTFE_BIN}" "$SNAP" "$TMP/p1" "${P[@]}" --gpu
echo ">> R7: the same watchdog fault, then the pace recovers after a clean stretch (set to 0 s)"
run "$TMP/p2" env DTFE_GPU_FAULT_AT=9:timeout DTFE_GPU_PACE_RECOVER_SEC=0 "${DTFE_BIN}" "$SNAP" "$TMP/p2" "${P[@]}" --gpu
echo ">> R4: --exact-average, a persistent fault from buffer 3 on: the sub-domain goes to the CPU"
ALLOW_CPU=1 run "$TMP/y1" env DTFE_GPU_FAULT_AT=3:persist "${DTFE_BIN}" "$SNAP" "$TMP/y1" "${C[@]}" "${ONE[@]}" --gpu --exact-average
echo ">> S1: the shared full grid (DTFE_GPU_SHARED=1), without and with an innocent fault at buffer 5"
run "$TMP/s0" env DTFE_GPU_SHARED=1 "${DTFE_BIN}" "$SNAP" "$TMP/s0" "${C[@]}" --gpu
run "$TMP/s1" env DTFE_GPU_SHARED=1 DTFE_GPU_FAULT_AT=5 "${DTFE_BIN}" "$SNAP" "$TMP/s1" "${C[@]}" --gpu
echo ">> R8: a --sample-points plane with --exact-average, a persistent fault: every partition on the threaded CPU average"
"$PY" - "$TMP/plane.bin" <<'PYEOF' || { echo "FAIL: R8 points"; exit 1; }
import sys, numpy as np
c = (np.arange(48) + 0.5) * 100. / 48           # a 48^2 plane across the 100 Mpc box, off the grid planes
X, Y = np.meshgrid(c, c, indexing="ij")
np.stack([X.ravel(), Y.ravel(), np.full(X.size, 50.3)], 1).astype(np.float64).tofile(sys.argv[1])
PYEOF
PL=(--grid 32 --periodic --input 105 --MpcUnit 1 --verbose 2 --padding 15 --field density_a velocity_a gradient_a
    --exact-average --partition 2 2 2 --max-concurrent 1 --sample-points "$TMP/plane.bin" --pts-vel-grad)
ALLOW_CPU=1 run "$TMP/v1" env DTFE_GPU_FAULT_AT=3:persist DTFE_DEPOSIT_THREADS=4 "${DTFE_BIN}" "$SNAP" "$TMP/v1" "${PL[@]}" --gpu
env DTFE_DEPOSIT_THREADS=1 "${DTFE_BIN}" "$SNAP" "$TMP/v0" "${PL[@]}" > "$TMP/v0.log" 2>&1 \
    || { echo "FAIL: the one-thread CPU plane run (see $TMP/v0.log)"; exit 1; }
echo ">> N1: the non-finite guard (DTFE_TEST_NAN_SHARE=50: NaN vertex densities in about one tetrahedron in 50)"
run "$TMP/n1" env DTFE_TEST_NAN_SHARE=50 "${DTFE_BIN}" "$SNAP" "$TMP/n1" "${C[@]}" "${ONE[@]}" --gpu
GPU_STAMP="$gpu_stamp" "$PY" - "$TMP" <<'PYEOF'
import numpy as np, re, sys
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
from pathlib import Path
tmp = Path(sys.argv[1]); fails = 0
GRIDS = ("a_den", "a_vel", "a_velGrad")
def check(tag, ok, msg):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {tag} {msg}")
def log(name):
    return (tmp / f"{name}.log").read_text(errors="replace")
def rd(name, ext):
    return np.fromfile(tmp / f"{name}.{ext}", dtype=REAL).astype(np.float64)
def same(a, b):        # the three grids byte for byte: (all identical, 'a_den=... a_vel=...' detail)
    out = []
    for ext in GRIDS:
        x, y = (tmp / f"{a}.{ext}").read_bytes(), (tmp / f"{b}.{ext}").read_bytes()
        out.append((ext, len(x) > 0 and x == y))
    return all(ok for _, ok in out), " ".join(f"{e}={'same' if ok else 'DIFFERENT'}" for e, ok in out)
def peak(a, b, ext):   # max |a - b| / max |a|
    x, y = rd(a, ext), rd(b, ext)
    return float(np.abs(x - y).max() / max(np.abs(x).max(), 1e-300)) if x.size == y.size else float("inf")
def failed_buffers(text):   # Metal: buffers, CUDA/HIP: kernels
    return sum(int(x) for x in re.findall(r"after (\d+) failed (?:buffer|kernel)\(s\)", text))
def by_cause(name):   # the last 'GPU deposit retries:' line of the run, as (N, innocent, timeout, oom, other)
    m = re.findall(r"GPU deposit retries: (\d+) \(innocent (\d+), timeout (\d+), oom (\d+), other (\d+)\)", log(name))
    return tuple(int(x) for x in m[-1]) if m else None

for a, b, what in (("a1", "a2", "item kernel, 9 sub-domains, opening chunk 4096 vs 65536"),
                   ("k1", "k2", "per-tet kernel (DTFE_GPU_ITEMS=0), opening chunk 4096 vs 25088"),
                   ("x1", "x2", "--exact-average, opening chunk 4096 vs 1024")):
    ok, detail = same(a, b)
    check("E1", ok, f"{what}: byte-identical grids ({detail})")
L = log("a1")
check("E1", "fixed-point sums" in L and "float sums" not in L,
      "the default runs the exact sums (every batch's timing line says 'fixed-point sums')")
L = log("f1")
check("E2", "float sums" in L and "fixed-point sums" not in L,
      "DTFE_METAL_FLOAT_GRIDS=1 / DTFE_GPU_FLOAT_GRIDS=1 runs the float atomics (the timing lines say 'float sums')")
for ext in GRIDS:
    p = peak("a1", "f1", ext)
    check("E2", p < 1e-5, f"float sums against the exact ones: '.{ext}' peak-rel {p:.1e} (< 1e-5)")
for ext in ("a_den", "a_vel"):
    c, g = rd("c1", ext), rd("a1", ext)
    m = abs(float(c.mean()) - float(g.mean())) / max(float(np.abs(c).max()), 1e-300)
    check("E2", m < 1e-5, f"GPU against the CPU: '.{ext}' box means {m:.1e} of the peak apart (< 1e-5)")

for tag, ref, run, what, cause, pace in (
        ("R1", "t0", "t1", "item kernel, one batch", "innocent victim", "250 ms"),
        ("R2", "e0", "e1", "--exact-average", "resource", "62 ms"),
        ("R3", "p0", "p1", "--partition 2 2 2, 9 threads", "watchdog", None)):
    L = log(run)
    resumed = re.search(r"\[" + cause + r"\] -- resuming from the (checkpoint|pass start) at thread", L)
    check(tag, resumed is not None and "recovered on the GPU" in L,
          f"{what}: the fault is classified as '{cause}' and the batch resumes on the GPU"
          + (f" (from the {resumed.group(1)})" if resumed else ""))
    if pace:
        check(tag, re.search(f"target {pace} (buffers|kernels)", L) is not None,
              f"{what}: the retry runs at {pace} ({'same pace' if cause == 'innocent victim' else 'escalated'})")
    ok, detail = same(ref, run)
    check(tag, ok, f"{what}: grids byte-identical to the run without the fault ({detail})")
check("R1", "resuming from the checkpoint at thread" in log("t1"),
      "the retry starts from a periodic checkpoint, not from the batch's start")
L = log("y1")
check("R4", "3 failures of the same chunk" in L and "gave up" in L and "using the CPU interpolation" in L
      and "1 sub-domain deposit(s) asked for the GPU and ran on the CPU instead" in L,
      "persistent fault: 3 failures of the same chunk, the CPU interpolation, and the run's count of such sub-domains")
p = peak("e0", "y1", "a_den")
check("R4", p < 1e-4, f"persistent fault: the CPU's '.a_den' agrees with the GPU's to {p:.1e} of the peak (< 1e-4)")
n1, n2, ny = failed_buffers(log("t1")), failed_buffers(log("e1")), failed_buffers(log("y1"))
check("R5", (n1, n2, ny) == (1, 1, 3), f"failed buffers counted: {n1}, {n2}, {ny} (expected 1, 1, 3)")
for name, want in (("t0", None), ("a1", None), ("t1", (1, 1, 0, 0, 0)), ("e1", (1, 0, 0, 1, 0)),
                   ("p1", (1, 0, 1, 0, 0)), ("y1", (3, 3, 0, 0, 0))):
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
ok, detail = same("p0", "p2")
check("R7", ok, f"grids byte-identical to the run without the fault after the pace recovered ({detail})")
L = log("s1")
check("S1", re.search(r"\[innocent victim\] -- resuming from the pass start at thread 0 of", L) is not None
      and "recovered on the GPU" in L,
      "shared grid: the fault re-zeroes the batch's owner box and resumes from its start")
worst = max(peak("s0", "s1", ext) for ext in GRIDS)
check("S1", worst < 1e-5, f"shared grid: the grids agree with a fault-free shared run to {worst:.1e} of the peak (< 1e-5)")
L = log("v1")
nfall, nthr = L.count("using the CPU interpolation"), len(re.findall(r"exact average on 4 CPU threads", L))
check("R8", nfall == 8 and nthr == 8 and "exact average on" not in log("v0"),
      f"planes + exact average, persistent fault: {nfall} of 8 partitions fell back, {nthr} of them on the threaded "
      "CPU exact average (4 threads); the one-thread run says nothing of the kind")
outs = sorted(p.name[3:] for p in tmp.glob("v0.*") if p.name[3:] not in ("log", "runlog") and not p.name.endswith((".json", ".txt")))
diff = [e for e in outs if not (tmp / f"v1.{e}").is_file() or (tmp / f"v0.{e}").read_bytes() != (tmp / f"v1.{e}").read_bytes()]
check("R8", len(outs) >= 6 and any(e.startswith("pts_") for e in outs) and not diff,
      f"the fallback's grids and point answers byte-identical to a one-thread CPU run ({len(outs)} outputs"
      + (f"; DIFFER: {diff})" if diff else ")"))
m = re.findall(r"(\d+) share\(s\) of the exact fixed-point sums were not finite.*?density grid (\d+)", log("n1"))
check("N1", len(m) == 1 and int(m[0][1]) > 0,
      "NaN vertex densities: the run reports its dropped density shares once"
      + (f" ({m[0][1]})" if m else " -- NO such warning"))
den = rd("n1", "a_den")
check("N1", den.size > 0 and bool(np.isfinite(den).all()), f"NaN vertex densities: '.a_den' finite everywhere ({den.size} cells)")
check("N1", not re.search(r"share\(s\) of the exact fixed-point sums", log("a1")), "a run without the hook reports no dropped share")
print("RESULT:", "PASS  (two GPU runs write the same grids, and a failed GPU buffer resumes from its checkpoint with "
      "exact grids, by cause; only a persistent failure reaches the CPU)" if not fails else f"FAIL ({fails} check(s))")
sys.exit(1 if fails else 0)
PYEOF
