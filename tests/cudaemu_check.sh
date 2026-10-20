#!/usr/bin/env bash
# The CUDA/HIP GPU backend through its CPU emulation: the parity suites the Metal backend
# passes, run against binaries built with 'make DTFE PS-DTFE CUDAEMU=1' (src/CGAL_triangulation/
# gpu_cuda_emu.h; metal/README.md "Validation without hardware"), at warp width 32 (CUDA) and 64 (the HIP
# wavefront, GPU_EMU_WARP=64). No NVIDIA or AMD GPU is needed, so this runs on a plain Linux CI runner and
# keeps the port's kernels and host logic honest between hardware runs (it does NOT check the memory
# model, scheduling or numerics of a real device).
#
# This script BUILDS NOTHING: it refuses unless o/ and o_ps/ carry the CUDAEMU stamp. (dtfe_metal_check.sh
# runs 'make DTFE <mode>' itself; with the objects up to date that make is a no-op.)
#
# Per warp width it runs, each with its own DTFE_TEST_TMP so the logs can be searched afterwards:
#   dtfe_metal_check, gpu_scalar_check, ps_linear_deposit, ps_vertex_mass, ps_volume_weighted,
#   ps_halo_release, ps_hidden_streams, ps_caustic_class, ps_nonperiodic_test.py, point_exact_serve,
#   ps_window_check, ps_smoke_test, gpu_mass_check, gpu_retry_check (the chunk checkpoints + cause-aware
#   retry of the Metal host, the per-cause retry counts, and the pace recovery
#   after a clean stretch, R7), ps_mass_exact_check (the host hands its fixed-point mass words to
#   the CPU, which adds the deferred and released tetrahedra into them -- 56% of the tets deferred there),
#   dtfe_gpu_retry_check (the standard-DTFE host dtfe_gpu_cuda.cu with the same retry policy and
#   its exact fixed-point '_a' sums -- two runs byte-identical, a faulted run byte-identical to a clean one),
#   ps_fixed_moments_check (every moment grid of the port is an exact fixed-point sum too --
#   addMoment in ps_gpu_cuda.cu -- two runs with other chunk sizes byte-identical in every grid)
# and its own check of the port's fixed-point mass grid (depositMass in ps_gpu_cuda.cu, as in the Metal kernel):
#   E1  two --ps-gpu runs write byte-identical '.a_den' (one domain, and --partition 2 2 2
#       --max-concurrent 1); float atomics would differ run to run
#   E2  PS_GPU_FLOAT_MASS=1 (the float grid, kept for A/B) agrees with the default to 1e-5 of the peak
#   E3  heavy accumulation (a 128^3 pancake on a 2^3 grid, nSub 1: 1.47e7 tetrahedra, ~1.8e6 deposits per
#       cell): the GPU box mean equals the CPU's to 3e-6 (the float grid gives 7.8e-6 here on Metal;
#       a 64^3 pancake does not tell the two apart: float 1.2e-6, fixed 1.2e-7 under the emulation)
# A suite that exits non-zero fails the run. A suite whose kept logs never name the emulated device
# ("CPU emulation of the CUDA backend, <W>-lane warps") is marked PASS* -- informational for most (several
# suites run at --verbose 1 or discard their logs; their own CPU-vs-GPU assertions and fallback greps still
# ran), a failure for those in CUDAEMU_GPU_REQUIRED, whose logs always carry the device line.
# The gates of gpu_mass_check.sh, gpu_retry_check.sh and ps_nonperiodic_test.py accept the emulated build,
# so each runs its GPU deposits here.
#
# The time-aware auto-tuner splits small runs into 2^3 partitions when it predicts a speed gain, and its
# prediction depends on the core count: on a 4- or 8-core Linux machine the CPU REFERENCE runs of
# gpu_scalar_check would be split while the --ps-gpu runs are not, and a split sampled deposit differs from one
# domain at 4e-3 of the peak -- a red suite that has nothing to do with the port. DTFE_AUTO_T_VERTEX (the
# tuner's per-vertex triangulation cost) is therefore set negligible here, so no speed split ever wins and
# every machine runs the same single domain the Mac does (export your own value to override).
#
# Disk: the suites write ~2.4 GB per width; a suite that passes has its grids deleted (its stdout.txt is
# kept), a failing one keeps everything for the post-mortem. CUDAEMU_KEEP=1 keeps all.
#
# Usage: tests/cudaemu_check.sh            (both widths)
#        CUDAEMU_WARPS=32 tests/cudaemu_check.sh
#        CUDAEMU_SUITES="gpu_scalar_check ps_window_check" tests/cudaemu_check.sh   (a subset; 'mass' = E1-E3)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1

for d in o o_ps; do
    if [ ! -f "$d/.gpu_mode_cudaemu" ]; then
        echo "FAIL: $d carries no CUDAEMU stamp -- build first: make DTFE PS-DTFE CUDAEMU=1"
        exit 1
    fi
done
for b in ./DTFE ./PS-DTFE; do [ -x "$b" ] || { echo "FAIL: $b is not built (make DTFE PS-DTFE CUDAEMU=1)"; exit 1; }; done
export GPU_BUILD=CUDAEMU=1      # dtfe_metal_check.sh: the mode its no-op make uses
export DTFE_TEST_PRECISION=single
export DTFE_AUTO_T_VERTEX="${DTFE_AUTO_T_VERTEX:-1e-15}"
PY="${PYTHON:-python3}"
export PYTHON="$PY"
BASE="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/cudaemu_check"
mkdir -p "$BASE"
WARPS="${CUDAEMU_WARPS:-32 64}"
ALL_SUITES="mass dtfe_metal_check gpu_scalar_check ps_linear_deposit_check ps_vertex_mass_check
ps_volume_weighted_check ps_halo_release_check ps_hidden_streams_check ps_caustic_class_check
ps_nonperiodic_test point_exact_serve_check ps_window_check ps_smoke_test gpu_mass_check gpu_retry_check
ps_mass_exact_check dtfe_gpu_retry_check ps_fixed_moments_check"
SUITES="${CUDAEMU_SUITES:-$ALL_SUITES}"
REQUIRED="${CUDAEMU_GPU_REQUIRED:-mass dtfe_metal_check ps_window_check gpu_retry_check ps_mass_exact_check dtfe_gpu_retry_check ps_fixed_moments_check}"
# suites that run the CPU path only by design (the smoke test) are never reported NO-GPU
CPU_ONLY="ps_smoke_test"

# mass_check <dir> <warp>: E1-E3 above
mass_check() {
    local T="$1" W="$2"
    local SNAP="$T/waves.hdf5" PAN="$T/pancake128.hdf5"
    "$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 48 --box 100 --amplitude-factor 1.8 \
        --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; return 1; }
    "$PY" tests/generate_ps_test_data.py --out "$PAN" --n 128 >/dev/null || { echo "FAIL data gen"; return 1; }
    run() { local out="$1"; shift; "$@" > "$out.log" 2>&1 || { echo "FAIL: $(basename "$out") (see $out.log)"; return 1; }
            if grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead" "$out.log"; then
                echo "FAIL: $(basename "$out") fell back to the CPU (see $out.log)"; return 1; fi; return 0; }
    local C=(--grid 32 --periodic --input 105 --MpcUnit 1 --verbose 2 --field density_a velocity_a --avg-subsamples 2 --ps-gpu)
    local P=(--partition 2 2 2 --max-concurrent 1)
    local H=(--grid 2 --periodic --MpcUnit 1 --verbose 2 --field density_a --avg-subsamples 1 --partition 1 1 1)
    run "$T/s1" ./PS-DTFE "$SNAP" "$T/s1" "${C[@]}" || return 1
    run "$T/s2" ./PS-DTFE "$SNAP" "$T/s2" "${C[@]}" || return 1
    run "$T/p1" ./PS-DTFE "$SNAP" "$T/p1" "${C[@]}" "${P[@]}" || return 1
    run "$T/p2" ./PS-DTFE "$SNAP" "$T/p2" "${C[@]}" "${P[@]}" || return 1
    run "$T/f1" env PS_GPU_FLOAT_MASS=1 ./PS-DTFE "$SNAP" "$T/f1" "${C[@]}" || return 1
    run "$T/h_cpu" ./PS-DTFE "$PAN" "$T/h_cpu" "${H[@]}" || return 1
    run "$T/h_gpu" ./PS-DTFE "$PAN" "$T/h_gpu" "${H[@]}" --ps-gpu || return 1
    "$PY" - "$T" "$W" <<'PYEOF'
import sys
from pathlib import Path
import numpy as np
tmp, warp = Path(sys.argv[1]), sys.argv[2]; fails = 0
def check(tag, ok, msg):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {tag} {msg}")
def rd(name):
    return np.fromfile(tmp / f"{name}.a_den", dtype=np.float32).astype(np.float64)
for a, b, what in (("s1", "s2", "one domain"), ("p1", "p2", "--partition 2 2 2 --max-concurrent 1")):
    x, y = (tmp / f"{a}.a_den").read_bytes(), (tmp / f"{b}.a_den").read_bytes()
    check("E1", len(x) > 0 and x == y, f"warp {warp}: two --ps-gpu runs ({what}): '.a_den' byte-identical ({len(x)} bytes)")
s, f = rd("s1"), rd("f1")
peak = float(np.abs(s - f).max() / np.abs(s).max())
check("E2", s.size == f.size and peak < 1e-5, f"warp {warp}: PS_GPU_FLOAT_MASS=1 against the fixed-point grid: peak-rel {peak:.1e} (< 1e-5)")
c, g = rd("h_cpu"), rd("h_gpu")
rel = abs(float(g.mean()) / float(c.mean()) - 1)
check("E3", c.size == 8 and rel < 3e-6, f"warp {warp}: heavy accumulation: GPU box mean {g.mean():.9f}, CPU {c.mean():.9f}, |ratio - 1| {rel:.1e} (< 3e-6)")
sys.exit(1 if fails else 0)
PYEOF
}

run_suite() {   # run_suite <name> <tmpdir>: the suite's own command line, output to <tmpdir>/stdout.txt
    local s="$1" T="$2"
    case "$s" in
        mass)                mass_check "$T" "$GPU_EMU_WARP" ;;
        ps_nonperiodic_test) "$PY" tests/ps_nonperiodic_test.py --no-build ;;
        dtfe_metal_check)    bash tests/dtfe_metal_check.sh ;;
        *)                   bash "tests/$s.sh" --no-build ;;
    esac
}

declare -a SUMMARY
nfail=0
for W in $WARPS; do
    export GPU_EMU_WARP="$W"
    for s in $SUITES; do
        T="$BASE/w$W/$s"
        rm -rf "$T"; mkdir -p "$T"
        echo ">> [warp $W] $s"
        t0=$(date +%s)
        DTFE_TEST_TMP="$T" run_suite "$s" "$T" > "$T/stdout.txt" 2>&1
        rc=$?
        dt=$(( $(date +%s) - t0 ))
        # the lines a reader wants: the suite's PASS/FAIL/SKIP/RESULT lines and its parity figures
        grep -E "PASS|FAIL|SKIP|RESULT|OK|parity|peak|rel" "$T/stdout.txt" | grep -v "^>> " | tail -40 | sed "s/^/   /"
        ngpu=$(grep -rlI "CPU emulation of the CUDA backend, ${W}-lane warps" "$T" 2>/dev/null | wc -l | tr -d ' ')
        status=PASS
        if [ "$rc" -ne 0 ]; then status=FAIL
        elif grep -q "^SKIP" "$T/stdout.txt"; then status=SKIP
        elif [ "$s" = ps_nonperiodic_test ] && ! grep -q " GPU " "$T/stdout.txt"; then
            status="PASS (CPU only)"       # its GPU variants did not run (a build-mode stamp without a GPU backend)
        elif [ "$ngpu" -eq 0 ] && ! [[ " $CPU_ONLY " == *" $s "* ]]; then
            status="PASS*"
            [[ " $REQUIRED " == *" $s "* ]] && status="FAIL (no emulated GPU run)"
        fi
        [[ "$status" == FAIL* ]] && nfail=$((nfail + 1))
        if [[ "$status" != FAIL* ]] && [ "${CUDAEMU_KEEP:-0}" != 1 ]; then
            find "$T" -mindepth 1 ! -name stdout.txt -exec rm -rf {} + 2>/dev/null
        fi
        SUMMARY+=("$(printf '  warp %-3s %-26s %-28s %4ss  %2s device logs   w%s/%s/stdout.txt' \
            "$W" "$s" "$status" "$dt" "$ngpu" "$W" "$s")")
    done
done
echo
echo "== cudaemu_check summary (outputs under $BASE)"
printf '%s\n' "${SUMMARY[@]}"
echo "  PASS* = passed, but no kept log names the emulated device (the suite logs at --verbose 1 or discards"
echo "          its logs); its own CPU-vs-GPU assertions ran. SKIP / CPU only: see the header of this script."
if [ "$nfail" -eq 0 ]; then echo "RESULT: PASS (CUDA/HIP port, CPU emulation, warps: $WARPS)"; exit 0; fi
echo "RESULT: FAIL ($nfail suite run(s))"
exit 1
