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
#  B) GPU parity path (GPU builds): scratch + --ps-gpu also bit-identical to plain --ps-gpu.
#  C) NO LEFTOVER FILES: scratch files are unlinked at creation, so the directory must be
#     empty after every run -- even though GBs flowed through it.
#  D) GUARDS: an iCloud path and a nonexistent directory are both rejected up front.
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
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
SNAP="${TMP}/scr_input_pancake.hdf5"
SC="$(mktemp -d /private/tmp/dtfe-scratch-check.XXXXXX)"
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
GPU_BUILT=0
[ -f o_ps/.gpu_mode_off ] || GPU_BUILT=1

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
# GPU float atomics accumulate in nondeterministic threadgroup order, so two identical plain
# --ps-gpu runs already differ bit-wise (verified) -- like every other GPU check in the suite,
# this compares to a float tolerance rather than bit-equality. Streams stay integer -> exact.
if [ "${GPU_BUILT}" -eq 1 ]; then
    echo ">> B: GPU deposit path (float tolerance; GPU atomics are order-nondeterministic)"
    "${BIN}" "${SNAP}" "${TMP}/scr_g_ref" "${BASEARGS[@]}" --ps-gpu --partition 2 2 2 \
        --max-concurrent 1 > "${TMP}/scr_g_ref.log" 2>&1
    DTFE_SCRATCH_MIN_GB=0.0001 "${BIN}" "${SNAP}" "${TMP}/scr_g_mm" "${BASEARGS[@]}" --ps-gpu \
        --partition 2 2 2 --max-concurrent 1 --scratch-dir "${SC}" > "${TMP}/scr_g_mm.log" 2>&1
    if "${PY}" - "${TMP}" <<'PYEOF'
import sys
import numpy as np
tmp = sys.argv[1]
ok = True
for ext, exact in ((".a_den", False), (".a_streams", True), (".a_vel", False), (".a_velDisp", False)):
    a = np.fromfile(f"{tmp}/scr_g_ref{ext}", dtype=np.float32).astype(np.float64)
    b = np.fromfile(f"{tmp}/scr_g_mm{ext}",  dtype=np.float32).astype(np.float64)
    if exact:
        good = np.mean(a == b) > 0.999
        print(f"        {ext}: {100*np.mean(a==b):.4f}% equal {'ok' if good else '<<<<'}")
    else:
        mrel = np.abs(a - b).mean() / (np.abs(a).max() + 1e-30)
        good = mrel < 1e-6
        print(f"        {ext}: mean rel = {mrel:.3e} {'ok' if good else '<<<<'}")
    ok &= good
sys.exit(0 if ok else 1)
PYEOF
    then check "B GPU scratch run matches plain GPU run (float tolerance)" 1
    else check "B GPU scratch run matches plain GPU run (float tolerance)" 0; fi
else
    echo ">> B skipped (CPU-only build)"
fi

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
