#!/usr/bin/env bash
# PS-DTFE partition/padding independence check (2026-10-12).
#
# A partitioned run must deposit the same tetrahedra as one single triangulation, whatever the padding:
# ownership tiles them, and a tetrahedron's keep/drop decision must be a function of the tetrahedron.
# It was not: psFilterCell's degeneracy cut scaled |det| by the edges at CGAL's vertex 0, and which
# vertex is 0 follows the insertion history -- different for every --partition / --padding /
# --ps-lag-padding variant. Folded tetrahedra on a caustic sit right at that cut, so ~20 of them were kept
# by one run and dropped by another: '.a_den' moved by up to 0.8 relative in ~40 cells (2,000 with
# --ps-vertex-mass, whose degrees then changed too), hidden-stream flags flipped, and the vertex-mass
# total was off by ~5e-6. The OLD binaries fail every check below; the fix makes the differences pure
# rounding (double: <= 4e-15 relative).
#
# 32^3 crossed waves (two fold planes per axis), --grid 64, --ps-vertex-mass (the run_ps_dtfe.sh default),
# CPU deposit; runs: single, --partition 2 2 2 at the default padding, at --padding 50, and at
# --ps-lag-padding 8. Checks against the single run:
#   P1 '.a_hidden_streams' identical (integer flags)
#   P2 '.a_den' per cell within rounding (float 1e-4, double 1e-10, relative, cells above 1e-6 of the peak)
#   P3 '.a_streams' within rounding (absolute 1e-3: one moved sub-sample is 1/64)
#   P4 vertex-mass conservation in every run: sum('.a_den') = grid cells (mean density 1) to 1e-6
#
# Usage:
#   tests/ps_partition_padding_check.sh [--no-build]
#   DTFE_TEST_PRECISION=double tests/ps_partition_padding_check.sh --no-build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"

N="${N:-32}"; GRID="${GRID:-64}"
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP}/ps_padcheck"
mkdir -p "${TMP}"
SNAP="${TMP}/input_cw.hdf5"

echo "============================================================"
echo " PS-DTFE partition/padding independence   N=${N}^3  grid=${GRID}^3  (${DTFE_TEST_PRECISION})"
echo "============================================================"

if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box 100 \
    --amplitude-factor 1.8 --crossed-waves >/dev/null

run() {  # $1 = output name, rest = extra options
    local name="$1"; shift
    local out="${TMP}/run_${name}"
    rm -f "${out}".a_den "${out}".a_streams "${out}".a_hidden_streams
    if ! "${BIN}" "${SNAP}" "${out}" --grid "${GRID}" --periodic --field density_a --ps-vertex-mass \
            --input 105 --MpcUnit 1 --verbose 1 "$@" > "${out}.log" 2>&1; then
        echo "FAIL: run ${name} exited non-zero -- last lines of ${out}.log:"
        tail -n 15 "${out}.log" | sed 's/^/   | /'
        exit 1
    fi
    echo "   ran ${name}"
}
run single
run p2       --partition 2 2 2
run p2pad50  --partition 2 2 2 --padding 50
run p2lag8   --partition 2 2 2 --ps-lag-padding 8

"${PY}" - "${TMP}" "${GRID}" "${DTFE_TEST_REAL}" <<'PY'
import sys
import numpy as np
tmp, grid, real = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ncell = grid ** 3
double = real == "float64"
den_tol = 1e-10 if double else 1e-4
fails = []

def load(name, ext):
    raw = np.fromfile(f"{tmp}/run_{name}{ext}", dtype=np.uint8)
    per = raw.size // ncell
    if ext == ".a_hidden_streams":
        return raw.view({1: np.uint8, 4: np.int32, 8: np.float64}[per]).astype(np.int64)
    return raw.view({4: np.float32, 8: np.float64}[per]).astype(np.float64)

ref = {e: load("single", e) for e in (".a_den", ".a_streams", ".a_hidden_streams")}
for name in ("single", "p2", "p2pad50", "p2lag8"):
    den = load(name, ".a_den")
    drift = abs(den.sum() - ncell) / ncell
    print(f"   P4 {name:8s} sum(.a_den) - cells = {den.sum() - ncell:+.3e}  (relative {drift:.1e})")
    if drift > 1e-6:
        fails.append(f"P4 {name}: vertex-mass total off by {drift:.1e} of the mass")
    if name == "single":
        continue
    hs = load(name, ".a_hidden_streams")
    nflip = int((hs != ref[".a_hidden_streams"]).sum())
    a, b = ref[".a_den"], den
    big = np.abs(a) > 1e-6 * np.abs(a).max()
    rel = np.abs(a - b)[big] / np.abs(a)[big]
    nbad = int((rel > den_tol).sum())
    ds = float(np.abs(load(name, ".a_streams") - ref[".a_streams"]).max())
    print(f"   {name:8s} vs single: hidden-stream flips {nflip}, .a_den max rel {rel.max():.2e} "
          f"({nbad} cells > {den_tol:.0e}), .a_streams max |d| {ds:.2e}")
    if nflip:
        fails.append(f"P1 {name}: {nflip} hidden-stream flags differ from the single run")
    if nbad:
        fails.append(f"P2 {name}: {nbad} '.a_den' cells differ by more than {den_tol:.0e} (max {rel.max():.2e})")
    if ds > 1e-3:
        fails.append(f"P3 {name}: '.a_streams' differs by {ds:.2e}")

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (every partition/padding variant deposits the single run's tetrahedra)")
PY

echo "============================================================"
echo " partition/padding check PASSED"
echo "============================================================"
