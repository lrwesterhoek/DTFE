#!/usr/bin/env bash
# PS-DTFE slab-mode CPU deposit: bit-identity with the single-threaded deposit.
#
# When the per-thread accumulator copies would not fit (a fine grid: one copy of the density+velocity
# accumulators is 24 GB at 1024^3), the CPU deposit runs in SLAB mode: each thread owns a slab of
# local x planes, takes whole slabs as it comes free, and walks the slab's tetrahedra in the loop's
# own order (ps_interpolation.cc). Every cell then receives its contributions in the order the
# single-threaded loop gives them, so the outputs must be BYTE-IDENTICAL to DTFE_DEPOSIT_THREADS=1,
# whatever the thread or slab count. On test-sized grids the copies always fit, so the suite forces
# the mode (DTFE_DEPOSIT_SLABS=1) and compares every output file with cmp, for every deposit variant:
# sampled (nSub 1 and 3), linear, exact, vertex mass, volume weighted, caustics, halo release, a
# non-periodic box, a --ps-window, --partition 2 2 2 (periodic sub-grids that wrap), and a few slabs
# that each hold many tetrahedra (DTFE_DEPOSIT_NSLABS=3) next to many thin ones. The 2D build too.
#
# Usage: tests/ps_slab_deposit_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-20}"; GRID="${GRID:-40}"; BOX="${BOX:-100.0}"
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
SNAP="${TMP}/psl_crossed.hdf5"
SNAP_NP="${TMP}/psl_clump.hdf5"
SNAP_2D="${TMP}/psl_2d.hdf5"

echo "============================================================"
echo " PS-DTFE slab-mode deposit check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building PS-DTFE ..."
    make PS-DTFE $(cat o_ps/.build_mode 2>/dev/null || true) >/dev/null
fi

echo ">> generating test snapshots ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box "${BOX}" --crossed-waves >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_NP}" --n "${N}" --box "${BOX}" \
    --margin-frac 0.2 --jitter-frac 0.02 >/dev/null

fails=0
check() { if [ "$1" = "1" ]; then echo "   PASS  $2"; else echo "   FAIL  $2"; fails=$((fails+1)); fi; }

# case <name> <binary> <snapshot> <extra args...>: serial vs forced slab mode, every output file cmp'ed
case_() {
    local name="$1" bin="$2" snap="$3"; shift 3
    local a="${TMP}/psl_${name}_serial" b="${TMP}/psl_${name}_slab"
    rm -f "${a}".* "${b}".* "${a}"_* "${b}"_*
    local common=( --grid "${GRID}" --input 105 --MpcUnit 1 --verbose "${VERBOSE:-2}" "$@" )
    set +e
    DTFE_DEPOSIT_THREADS=1 "${bin}" "${snap}" "${a}" "${common[@]}" > "${a}.log" 2>&1; local ra=$?
    env DTFE_DEPOSIT_SLABS=1 ${NSLABS:+DTFE_DEPOSIT_NSLABS=${NSLABS}} "${bin}" "${snap}" "${b}" "${common[@]}" > "${b}.log" 2>&1; local rb=$?
    set -e
    if [ "$ra" -ne 0 ] || [ "$rb" -ne 0 ]; then
        check 0 "${name}: both runs exit 0 (serial $ra, slab $rb; ${a}.log / ${b}.log)"; return
    fi
    if ! grep -q "SLAB mode" "${b}.log"; then
        check 0 "${name}: the forced run used slab mode (no 'SLAB mode' line in ${b}.log)"; return
    fi
    local files=0 differ=""
    for f in "${a}".*; do
        local ext="${f#"${a}"}"
        case "${ext}" in .log) continue ;; esac
        files=$((files+1))
        cmp -s "${f}" "${b}${ext}" || differ="${differ} ${ext}"
    done
    local slabs; slabs="$(grep -o 'over [0-9]* slabs' "${b}.log" | head -1)"
    check "$([ "${files}" -gt 0 ] && [ -z "${differ}" ] && echo 1 || echo 0)" \
          "${name}: ${files} output files byte-identical to the single-threaded deposit (${slabs})${differ:+ -- DIFFER:${differ}}"
}

FIELDS=( --field density velocity dispersion divergence shear )
echo ">> serial vs slab runs ..."
case_ sampled_nsub1   "${BIN}" "${SNAP}"    --periodic "${FIELDS[@]}" --avg-subsamples 1
case_ sampled_nsub3   "${BIN}" "${SNAP}"    --periodic "${FIELDS[@]}" --avg-subsamples 3
case_ linear          "${BIN}" "${SNAP}"    --periodic "${FIELDS[@]}" --ps-linear-deposit
case_ exact           "${BIN}" "${SNAP}"    --periodic "${FIELDS[@]}" --ps-exact-deposit
case_ vertex_mass     "${BIN}" "${SNAP}"    --periodic "${FIELDS[@]}" --ps-vertex-mass
case_ volume_weighted "${BIN}" "${SNAP}"    --periodic "${FIELDS[@]}" --ps-volume-weighted
case_ caustics        "${BIN}" "${SNAP}"    --periodic --field density --ps-caustics
case_ halo_release    "${BIN}" "${SNAP}"    --periodic "${FIELDS[@]}" --ps-halo-release 2
case_ nonperiodic     "${BIN}" "${SNAP_NP}" --field density velocity
case_ window          "${BIN}" "${SNAP}"    --periodic --field density velocity \
                                            --ps-window 10 47.5 20 80 0 100
# (a partition reports its slab mode at verbose 3 only)
VERBOSE=3 case_ partition "${BIN}" "${SNAP}" --periodic "${FIELDS[@]}" --partition 2 2 2 --max-concurrent 1
NSLABS=3 case_ three_slabs "${BIN}" "${SNAP}" --periodic "${FIELDS[@]}" --ps-exact-deposit

# the 2D build (single precision only: the 2D set has no double pair in the battery)
BIN2D="${PS_BIN}-2d"
if ! precision_double && [ -x "${BIN2D}" ]; then
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_2D}" --n 48 --box "${BOX}" --dim 2 \
        --crossed-waves >/dev/null
    GRID=96 case_ two_d "${BIN2D}" "${SNAP_2D}" --periodic --field density velocity --ps-exact-deposit
else
    if precision_double; then echo "   SKIP  2D (double mode: the 2D set has no double pair in the battery)"
    else echo "   SKIP  2D (no ${BIN2D}: make DTFE PS-DTFE DIM=2)"; fi
fi

echo "------------------------------------------------------------"
if [ "${fails}" -gt 0 ]; then
    echo "RESULT: FAIL  (${fails} case(s))"
    exit 1
fi
echo "RESULT: PASS  (slab-mode CPU deposit is byte-identical to the single-threaded deposit for every variant)"
