#!/usr/bin/env bash
# PS-DTFE strong-scaling benchmark: times 1/2/4/8 threads (capped at core AND partition
# count, since parallelism is coarse-grained over Lagrangian partitions) and prints a
# wall/speedup/efficiency table. Also a regression guard: FAILS if the top-core speedup
# drops below MIN_EFFICIENCY x cores, catching a serialized partition loop.
#
# Usage:
#   tests/ps_scaling_benchmark.sh                 # build, benchmark, assert
#   tests/ps_scaling_benchmark.sh --no-build
#   N=64 GRID=128 tests/ps_scaling_benchmark.sh   # bigger problem
#   MIN_EFFICIENCY=0.3 tests/ps_scaling_benchmark.sh   # looser guard (loaded box)
# Requires the build toolchain plus python3 with numpy and h5py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# python with a working h5py (plain python3 resolves to a broken x86_64 h5py on this machine)
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14

N="${N:-48}"; GRID="${GRID:-96}"; BOX="${BOX:-100.0}"; AMP="${AMP:-1.8}"
PARTITION="${PARTITION:-2 2 2}"   # Lagrangian partition grid; parallelism is over these
FIELDS="${FIELDS:-density velocity density_a}"
MIN_EFFICIENCY="${MIN_EFFICIENCY:-0.4}"   # require speedup >= MIN_EFFICIENCY * cores at top core count
BIN="./PS-DTFE"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
SNAP="${TMP}/ps_scaling_input.hdf5"

NPROC="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
NPART=1; for p in ${PARTITION}; do NPART=$(( NPART * p )); done   # product of PARTITION grid
# top core count = min(8, cores, partitions)
MAXC=8; [ "${NPROC}" -lt "${MAXC}" ] && MAXC="${NPROC}"; [ "${NPART}" -lt "${MAXC}" ] && MAXC="${NPART}"

CORES=()
for c in 1 2 4 8; do [ "$c" -le "${MAXC}" ] && CORES+=("$c"); done

echo "============================================================"
echo " PS-DTFE strong-scaling   N=${N}^3  grid=${GRID}^3  partition=[${PARTITION}] (${NPART} parts)  cores<=${MAXC}/${NPROC}"
echo "============================================================"

if [ "${1:-}" != "--no-build" ]; then
    echo ">> building PS-DTFE ..."
    # rebuild in the current GPU mode (a plain make would strip GPU support from the binary)
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

echo ">> generating test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" \
    --out "${SNAP}" --n "${N}" --box "${BOX}" --amplitude-factor "${AMP}" >/dev/null

now() { "${PY}" -c 'import time;print("%.4f"%time.time())'; }

declare -a TIMES
T1=""
for T in "${CORES[@]}"; do
    out="${TMP}/ps_scaling_${T}"
    rm -f "${out}".*
    t0="$(now)"
    set +e
    OMP_NUM_THREADS="${T}" "${BIN}" "${SNAP}" "${out}" \
        --grid "${GRID}" --periodic --partition ${PARTITION} --field ${FIELDS} \
        --input 105 --MpcUnit 1 --verbose 0 > "${out}.log" 2>&1
    rc=$?
    set -e
    t1="$(now)"
    if [ "${rc}" -ne 0 ]; then
        echo "   ERROR: PS-DTFE (threads=${T}) exited ${rc} -- last 25 lines of ${out}.log:"
        tail -n 25 "${out}.log" | sed 's/^/      | /'
        exit 1
    fi
    wall="$("${PY}" -c "print('%.3f'%(${t1}-${t0}))")"
    TIMES+=("${wall}")
    [ -z "${T1}" ] && T1="${wall}"
    rm -f "${out}".* "${out}.log"
done

echo
printf "   %-8s %-12s %-10s %-10s\n" "cores" "wall(s)" "speedup" "efficiency"
echo   "   -------------------------------------------------"
TOP_SPEEDUP=""
for i in "${!CORES[@]}"; do
    T="${CORES[$i]}"; W="${TIMES[$i]}"
    sp="$("${PY}" -c "print('%.2f'%(${T1}/${W}))")"
    ef="$(python3 -c "print('%.0f%%'%(100.0*${T1}/${W}/${T}))")"
    printf "   %-8s %-12s %-10s %-10s\n" "${T}" "${W}" "${sp}x" "${ef}"
    TOP_SPEEDUP="${sp}"
done
echo   "   -------------------------------------------------"

# Tracked regression guard: speedup at the top core count must clear the floor.
FLOOR="$(python3 -c "print('%.2f'%(${MIN_EFFICIENCY}*${MAXC}))")"
PASS="$(python3 -c "print(1 if ${TOP_SPEEDUP} >= ${FLOOR} else 0)")"
echo
if [ "${PASS}" -ne 1 ]; then
    echo "RESULT: FAIL  (${TOP_SPEEDUP}x at ${MAXC} cores < required ${FLOOR}x"
    echo "              = ${MIN_EFFICIENCY} efficiency x ${MAXC} cores -- scaling regressed?)"
    exit 1
fi
echo "RESULT: PASS  (${TOP_SPEEDUP}x speedup at ${MAXC} cores >= ${FLOOR}x floor)"
echo "============================================================"
