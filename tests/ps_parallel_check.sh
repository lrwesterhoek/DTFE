#!/usr/bin/env bash
# PS-DTFE parallel-vs-serial correctness check: runs serial and all-cores, asserts
# outputs agree (a race shows up as a mismatch). .streams must match exactly (integer
# counts are order-independent), float fields only within tolerance.
#
# Usage:
#   tests/ps_parallel_check.sh                 # build, run x2, compare
#   tests/ps_parallel_check.sh --no-build
#   N=64 GRID=128 tests/ps_parallel_check.sh   # bigger problem
# Requires the build toolchain plus python3 with numpy and h5py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# python with a working h5py (plain python3 resolves to a broken x86_64 h5py on this machine)
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

N="${N:-32}"; GRID="${GRID:-64}"; BOX="${BOX:-100.0}"; AMP="${AMP:-1.8}"
PARTITION="${PARTITION:-2 2 2}"   # Lagrangian partition grid; parallelism is over these
BIN="./PS-DTFE"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
# Input name must NOT share a prefix with the output roots (ps_serial/ps_par),
# or the per-run 'rm -f <root>.*' would delete it.
SNAP="${TMP}/ps_pcheck_input.hdf5"
NPROC="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

echo "============================================================"
echo " PS-DTFE parallel-vs-serial check   N=${N}^3  grid=${GRID}^3  cores=${NPROC}  partition=[${PARTITION}]"
echo "============================================================"

if [ "${1:-}" != "--no-build" ]; then
    echo ">> building PS-DTFE ..."
    # respect the current build mode (don't silently strip GPU support from a GPU binary;
    # o_ps/.build_mode records the make argument, e.g. METAL=1)
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

echo ">> generating test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" \
    --out "${SNAP}" --n "${N}" --box "${BOX}" --amplitude-factor "${AMP}" >/dev/null

run() {  # $1 = thread count, $2 = output root
    rm -f "$2".*
    local log="$2.log"
    local t0=$SECONDS
    set +e
    OMP_NUM_THREADS="$1" "${BIN}" "${SNAP}" "$2" \
        --grid "${GRID}" --periodic --partition ${PARTITION} --field density velocity density_a \
        --input 105 --MpcUnit 1 --verbose 1 > "$log" 2>&1
    local rc=$?
    set -e
    echo "   [threads=$1] wall time: $(( SECONDS - t0 )) s"
    if [ "$rc" -ne 0 ]; then
        echo "   ERROR: PS-DTFE (threads=$1) exited with code $rc -- last 25 lines of $log:"
        tail -n 25 "$log" | sed 's/^/      | /'
        exit 1
    fi
}

echo ">> run 1: serial  (OMP_NUM_THREADS=1, --partition ${PARTITION})"
run 1 "${TMP}/ps_serial"
echo ">> run 2: parallel (OMP_NUM_THREADS=${NPROC}, --partition ${PARTITION})"
run "${NPROC}" "${TMP}/ps_par"

echo ">> files produced:"
ls -la "${TMP}"/ps_serial.* "${TMP}"/ps_par.* 2>&1 | sed 's/^/   /' || true

echo ">> comparing serial vs parallel outputs ..."
"${PY}" - "${TMP}/ps_serial" "${TMP}/ps_par" "${GRID}" <<'PY'
import os, sys
import numpy as np

a, b, grid = sys.argv[1], sys.argv[2], int(sys.argv[3])
ncell = grid ** 3
fails = []

def load(path, ncomp):
    raw = np.fromfile(path, dtype=np.uint8); n = ncell * ncomp
    if raw.size == n * 4: return raw.view(np.float32).astype(np.float64)
    if raw.size == n * 8: return raw.view(np.float64)
    sys.exit(f"FAIL: {path} unexpected size {raw.size}")

def compare(ext, ncomp, exact, tol=1e-3):
    fa, fb = a + ext, b + ext
    if not (os.path.exists(fa) and os.path.exists(fb)):
        print(f"   {ext:10s} MISSING (serial={os.path.exists(fa)}, parallel={os.path.exists(fb)})")
        fails.append(f"{ext} not produced by both runs"); return
    x, y = load(fa, ncomp), load(fb, ncomp)
    dmax = float(np.abs(x - y).max())
    rel = dmax / (float(np.abs(x).max()) + 1e-30)
    print(f"   {ext:10s} max|Δ|={dmax:.3e}  max relΔ={rel:.3e}"
          + ("  [must be exact]" if exact else f"  [tol {tol:.0e}]"))
    if exact and dmax != 0.0:
        fails.append(f"{ext} not identical ({dmax:.3e}) -- possible race")
    if (not exact) and rel > tol:
        fails.append(f"{ext} relΔ {rel:.2e} exceeds {tol:.0e} -- possible race")

compare(".streams",  1, True)       # integer counts -> order-independent -> exact
# .a_streams is volume-averaged (fractional): partitions sum in a run-dependent order,
# so it agrees only to ~1 float32 ULP. Benign FP non-associativity, not a race (a race
# would drop/double whole streams, rel ~1e-2 >> tol).
compare(".a_streams", 1, False)
compare(".den",      1, False)
compare(".a_den",    1, False)
compare(".vel",      3, False)

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (parallel == serial: integer stream counts identical, "
      "averaged/float fields agree within rounding)")
PY

echo "============================================================"
echo " parallel check PASSED"
echo "============================================================"
