#!/usr/bin/env bash
# PS-DTFE build + run smoke test: build, generate a Zel'dovich pancake with shell crossing,
# run PS-DTFE, and check the output is physically sane (finite non-negative density, mass
# roughly conserved, multi-stream regions). Proves the build runs; accuracy is a separate test.
#
# Usage:
#   tests/ps_smoke_test.sh                 # build, run, check
#   tests/ps_smoke_test.sh --no-build      # skip 'make', reuse existing PS-DTFE
#   N=48 GRID=96 BOX=100 tests/ps_smoke_test.sh   # override problem size
# Requires the build toolchain plus python3 with numpy and h5py. Exits non-zero on first failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# python with a working h5py (plain python3 resolves to a broken x86_64 h5py on this machine)
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Parameters (override via environment).
N="${N:-32}"                 # particles per side (total N^3)
GRID="${GRID:-64}"           # output grid per side (GRID^3)
BOX="${BOX:-100.0}"          # box size in Mpc
AMP="${AMP:-1.8}"            # Zel'dovich amplitude factor (>1 => shell crossing)
BIN="${BIN:-./PS-DTFE}"
TMP_DIR="${TMP_DIR:-${SCRIPT_DIR}/tmp}"
INPUT_H5="${TMP_DIR}/ps_zeldovich.hdf5"
OUT_ROOT="${TMP_DIR}/ps_out"

DO_BUILD=1
[ "${1:-}" = "--no-build" ] && DO_BUILD=0

echo "============================================================"
echo " PS-DTFE smoke test"
echo "   root      : ${ROOT_DIR}"
echo "   particles : ${N}^3   grid: ${GRID}^3   box: ${BOX} Mpc"
echo "============================================================"

if [ "${DO_BUILD}" -eq 1 ]; then
    echo ">> [1/4] Building PS-DTFE ..."
    # macOS SDK handling (MACOS_ISYSROOT) lives in the Makefile; needed because an
    # OS/CLT update can leave Homebrew clang pointing at a non-existent SDK.
    # Respect the current build mode: a plain 'make PS-DTFE' would silently strip the
    # GPU support from a METAL=1/CUDA=1/HIP=1 binary (the Makefile wipes o_ps on mode
    # changes and records the mode's make argument in o_ps/.build_mode).
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"}
else
    echo ">> [1/4] Skipping build (--no-build)"
fi
[ -x "${BIN}" ] || { echo "FAIL: '${BIN}' not found or not executable."; exit 1; }

echo ">> [2/4] Generating synthetic Zel'dovich pancake ..."
mkdir -p "${TMP_DIR}"
GEN_OUT="$("${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" \
    --out "${INPUT_H5}" --n "${N}" --box "${BOX}" --amplitude-factor "${AMP}")"
printf '%s\n' "${GEN_OUT}"
EXPECTED_MEAN="$(printf '%s\n' "${GEN_OUT}" | awk -F': ' '/expected mean density/ {print $2}' | awk '{print $1}')"

# Run PS-DTFE (density + stream count; coordinates already in Mpc).
echo ">> [3/4] Running PS-DTFE ..."
rm -f "${OUT_ROOT}.den" "${OUT_ROOT}.streams"
"${BIN}" "${INPUT_H5}" "${OUT_ROOT}" \
    --grid "${GRID}" --periodic --field density --input 105 --MpcUnit 1 --verbose 2

echo ">> [4/4] Checking outputs ..."
"${PY}" - "${OUT_ROOT}.den" "${OUT_ROOT}.streams" "${GRID}" "${EXPECTED_MEAN}" <<'PY'
import sys
import numpy as np

den_file, streams_file, grid, expected_mean = sys.argv[1:5]
grid = int(grid)
expected_mean = float(expected_mean)
ncells = grid ** 3
failures = []


def load(path):
    """Load a raw DTFE binary field, inferring float32/float64 from file size."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == ncells * 4:
        return raw.view(np.float32).astype(np.float64)
    if raw.size == ncells * 8:
        return raw.view(np.float64)
    sys.exit(f"FAIL: {path} has {raw.size} bytes, "
             f"expected {ncells*4} (f32) or {ncells*8} (f64) for {grid}^3.")


# Density. The '.den' output is mean-normalized (rho/rho_bar, with rho_bar = N*m/L^3 the
# physical mean printed by the generator), so the mass-conserving deposit must give a grid
# mean of ~1 regardless of the input's physical units.
den = load(den_file)
n_finite = np.isfinite(den).sum()
mean = den.mean()
nonzero_frac = (den > 0).mean()

print(f"  density : cells={den.size}  finite={n_finite}/{den.size}")
print(f"            min={den.min():.4g}  max={den.max():.4g}  mean={mean:.4g}")
print(f"            non-zero cells={nonzero_frac*100:.1f}%  "
      f"(output is rho/rho_bar; physical rho_bar = {expected_mean:.6g})")

if n_finite != den.size:
    failures.append("density contains non-finite (NaN/Inf) values")
if den.min() < -1e-3 * max(abs(mean), 1e-30):
    failures.append(f"density has significantly negative values (min={den.min():.4g})")
if nonzero_frac < 0.10:
    failures.append(f"density field is mostly empty ({nonzero_frac*100:.1f}% non-zero)")
if not (0.95 <= mean <= 1.05):
    failures.append(f"normalized mean density is not ~1 -- the deposit lost/gained mass (mean={mean:.4g})")

# Stream count.
streams = load(streams_file)
max_streams = streams.max()
multi_frac = (streams > 1).mean()
near_int = np.allclose(streams, np.round(streams), atol=1e-4)

print(f"  streams : max={max_streams:.0f}  "
      f"multi-stream cells={multi_frac*100:.1f}%  integer-valued={near_int}")

if streams.min() < -1e-6:
    failures.append("stream count has negative values")
if not near_int:
    failures.append("stream count is not integer-valued")
if max_streams < 2:
    failures.append("no multi-stream regions detected (expected >=3 for a pancake)")

# Verdict.
print("------------------------------------------------------------")
if failures:
    print("RESULT: FAIL")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (build runs; density and stream-count outputs are sane)")
PY

echo "============================================================"
echo " PS-DTFE smoke test PASSED"
echo "   outputs: ${OUT_ROOT}.den , ${OUT_ROOT}.streams"
echo "============================================================"
