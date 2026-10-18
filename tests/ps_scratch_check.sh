#!/usr/bin/env bash
# --scratch-dir (out-of-core mmap accumulators, src/scratch_alloc.cc) correctness check.
#
# The scratch route replaces the global operator new/delete: allocations >= a threshold are
# served from mmap'ed, immediately-unlinked files so the full-grid accumulators live on disk.
# DTFE_SCRATCH_MIN_GB is set TINY here so even this test's small grids exercise the route.
# Verified:
#  A) BIT-IDENTICAL OUTPUTS vs a normal in-RAM run, on the partitioned (--max-concurrent 1,
#     the deterministic merge order -- mc>1 float merges are order-nondeterministic even
#     WITHOUT scratch) and the unpartitioned deposit paths. It is the same memory semantics,
#     so anything but equality is a bug.
#  B) GPU parity path (GPU builds): scratch + --ps-gpu also bit-identical to plain --ps-gpu (exact
#     fixed-point GPU sums since 2026-10-14; a float tolerance until 2026-10-15).
#  C) NO LEFTOVER FILES: scratch files are unlinked at creation, so the directory must be
#     empty after every run -- even though GBs flowed through it.
#  D) GUARDS: an iCloud path and a nonexistent directory are both rejected up front.
#  S) the staged ordered merge's SPILL (2026-10-15) into the scratch folder, with the sub-grids themselves in
#     scratch mappings: bit-identical to A's --max-concurrent 1 run, and C covers its files too.
#
# Portable (2026-10-11): runs on Linux too (tests/ci_suite.sh core); B is GPU-only and SKIPs on a CPU build.
#
# Usage: tests/ps_scratch_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

N="${N:-64}"; BOX="${BOX:-100.0}"
BIN="./PS-DTFE"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"; mkdir -p "${TMP}"
SNAP="${TMP}/scr_input_pancake.hdf5"
# a LOCAL temp folder on either system (was /private/tmp: macOS only; 2026-10-11 Linux port). TMPDIR on a Mac is
# /var/folders/..., local and never under iCloud, which the binary would refuse (section D)
SC="$(mktemp -d "${TMPDIR:-/tmp}/dtfe-scratch-check.XXXXXX")"
trap 'rm -rf "${SC}"' EXIT

echo "============================================================"
echo " --scratch-dir check   N=${N}^3   scratch=${SC}"
echo "============================================================"

if [ "${1:-}" != "--no-build" ]; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }
# GPU deposit compiled in: the build mode names a backend (o_ps/.gpu_mode_off alone also called the CUDA
# emulator and a never-stamped tree "GPU")
GPU_BUILT=0
grep -Eq "METAL=1|CUDA=1|HIP=1|CUDAEMU=1" o_ps/.build_mode 2>/dev/null && GPU_BUILT=1

echo ">> generating test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null

FAILS=0
check() { if [ "$2" -eq 1 ]; then echo "   OK   $1"; else echo "   FAIL $1"; FAILS=$((FAILS+1)); fi; }

BASEARGS=(--grid 128 --field density_a velocity_a dispersion_a --input 105 --MpcUnit 1
          --verbose 2 --periodic --avg-subsamples 1 --ps-volume-weighted)

compare_outputs() {  # $1 = ref root, $2 = scratch root, $3 = label
    local allsame=1
    for f in "$1".*; do
        local e="${f#$1}"
        [ "${e}" = ".log" ] && continue
        cmp -s "${f}" "$2${e}" || { allsame=0; echo "        differs: ${e}"; }
    done
    check "$3 outputs bit-identical to the in-RAM run" "${allsame}"
}

# ---------- (A) partitioned (deterministic mc=1) and unpartitioned bit-identity ----------
echo ">> A: bit-identity, partitioned (mc=1) + unpartitioned"
"${BIN}" "${SNAP}" "${TMP}/scr_p_ref" "${BASEARGS[@]}" --partition 2 2 2 --max-concurrent 1 \
    > "${TMP}/scr_p_ref.log" 2>&1
DTFE_SCRATCH_MIN_GB=0.0001 "${BIN}" "${SNAP}" "${TMP}/scr_p_mm" "${BASEARGS[@]}" \
    --partition 2 2 2 --max-concurrent 1 --scratch-dir "${SC}" > "${TMP}/scr_p_mm.log" 2>&1 \
    && check "A scratch partitioned run exits 0" 1 || check "A scratch partitioned run exits 0" 0
compare_outputs "${TMP}/scr_p_ref" "${TMP}/scr_p_mm" "A partitioned"

"${BIN}" "${SNAP}" "${TMP}/scr_u_ref" "${BASEARGS[@]}" > "${TMP}/scr_u_ref.log" 2>&1
DTFE_SCRATCH_MIN_GB=0.0001 "${BIN}" "${SNAP}" "${TMP}/scr_u_mm" "${BASEARGS[@]}" \
    --scratch-dir "${SC}" > "${TMP}/scr_u_mm.log" 2>&1
compare_outputs "${TMP}/scr_u_ref" "${TMP}/scr_u_mm" "A unpartitioned"
if grep -q "SCRATCH:" "${TMP}/scr_p_mm.log"; then check "A arming banner printed" 1; else check "A arming banner printed" 0; fi

# ---------- (B) GPU parity path ----------
# Since 2026-10-14 every grid of a GPU deposit is an exact 64-bit fixed-point sum (the mass since 2026-10-08,
# the moments since 2026-10-14; ps_deposit_params.h PSFxGrid), and --max-concurrent 1 merges the partitions in
# index order, so the scratch run equals the plain --ps-gpu run byte for byte, as the CPU runs of A do. (Until
# 2026-10-15 this compared to a float tolerance: the float atomics of before landed in a different order per run.)
if [ "${GPU_BUILT}" -eq 1 ]; then
    echo ">> B: GPU deposit path (exact fixed-point sums: bit-identity)"
    "${BIN}" "${SNAP}" "${TMP}/scr_g_ref" "${BASEARGS[@]}" --ps-gpu --partition 2 2 2 \
        --max-concurrent 1 > "${TMP}/scr_g_ref.log" 2>&1
    DTFE_SCRATCH_MIN_GB=0.0001 "${BIN}" "${SNAP}" "${TMP}/scr_g_mm" "${BASEARGS[@]}" --ps-gpu \
        --partition 2 2 2 --max-concurrent 1 --scratch-dir "${SC}" > "${TMP}/scr_g_mm.log" 2>&1
    # a CPU fallback in one run of the pair (GPU contention) breaks the identity for another reason: say so
    if grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead" "${TMP}/scr_g_ref.log" "${TMP}/scr_g_mm.log"; then
        echo "        (a GPU run fell back to the CPU: see ${TMP}/scr_g_ref.log / scr_g_mm.log)"
        check "B GPU runs stayed on the GPU" 0
    else
        compare_outputs "${TMP}/scr_g_ref" "${TMP}/scr_g_mm" "B GPU scratch run:"
    fi
else
    echo "   SKIP  B (this build has no GPU backend)"
fi

# ---------- (S) the staged ordered merge's spill into the scratch folder (2026-10-15) ----------
# four partitions in flight, partition 0 held back so the others park, every parked one SPILLED (DTFE_MERGE_SPILL=1)
# to --scratch-dir while the tiny threshold puts their sub-grids themselves in scratch mappings: freed mappings and
# sub-grids read back into new ones, through the replaced operator new/delete. Byte-identical to A's mc=1 run, and
# section C below then checks the folder is empty (the spill files are unlinked at creation too).
echo ">> S: the ordered merge's spill to the scratch folder (4 in flight, mmap'ed sub-grids)"
DTFE_MERGE_SPILL=1 DTFE_TEST_DELAY_PARTITION=0:2 DTFE_SCRATCH_MIN_GB=0.0001 OMP_NUM_THREADS=4 "${BIN}" "${SNAP}" \
    "${TMP}/scr_s_mm" "${BASEARGS[@]}" --partition 2 2 2 --max-concurrent 4 --scratch-dir "${SC}" \
    > "${TMP}/scr_s_mm.log" 2>&1 && check "S spilling scratch run exits 0" 1 || check "S spilling scratch run exits 0" 0
NSP="$(sed -n 's/.*ordered merge: .* \([0-9]*\) spilled to disk.*/\1/p' "${TMP}/scr_s_mm.log")"
check "S ${NSP:-no} partition(s) spilled to the scratch folder" $([ "${NSP:-0}" -gt 0 ] && echo 1 || echo 0)
compare_outputs "${TMP}/scr_p_ref" "${TMP}/scr_s_mm" "S spilled"

# ---------- (C) nothing left behind ----------
NLEFT=$(ls "${SC}" 2>/dev/null | wc -l | tr -d ' ')
check "C scratch dir empty after runs (unlink-at-creation)" $([ "${NLEFT}" -eq 0 ] && echo 1 || echo 0)

# ---------- (D) guards ----------
echo ">> D: path guards"
ICLOUD_DIR="${SC}/Mobile Documents/fake"
mkdir -p "${ICLOUD_DIR}"
set +e
"${BIN}" "${SNAP}" "${TMP}/scr_x" "${BASEARGS[@]}" --scratch-dir "${ICLOUD_DIR}" > "${TMP}/scr_x.log" 2>&1
RC_ICLOUD=$?
"${BIN}" "${SNAP}" "${TMP}/scr_y" "${BASEARGS[@]}" --scratch-dir /nonexistent/dtfe/path > "${TMP}/scr_y.log" 2>&1
RC_NODIR=$?
set -e
check "D iCloud path rejected (nonzero exit)" $([ "${RC_ICLOUD}" -ne 0 ] && echo 1 || echo 0)
if grep -q "iCloud" "${TMP}/scr_x.log"; then check "D iCloud rejection names the reason" 1; else check "D iCloud rejection names the reason" 0; fi
check "D nonexistent dir rejected (nonzero exit)" $([ "${RC_NODIR}" -ne 0 ] && echo 1 || echo 0)

echo "------------------------------------------------------------"
if [ "${FAILS}" -gt 0 ]; then
    echo "RESULT: FAIL (${FAILS} check(s))"
    exit 1
fi
echo "RESULT: PASS  (--scratch-dir is bit-identical to in-RAM runs, leaves no files behind,"
echo "               and rejects unsafe directories)"
echo "============================================================"
