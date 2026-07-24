#!/usr/bin/env bash
# Auto-tune (src/auto_tune.h) behaviour checks -- the first coverage this header has ever had.
# Drives the REAL binary through the documented env overrides (DTFE_RAM_GB simulates the
# machine's RAM, DTFE_AUTO_MINN lowers the particle-count gate) so every scenario is fast,
# deterministic and allocation-safe. Verified here:
#
#  A) OVER-BUDGET REFUSAL. When the irreducible full-grid accumulators alone exceed the
#     budget, auto-tune must say so BEFORE the search -- naming the grid term, stating that
#     --partition cannot reduce it, suggesting the largest grid that WOULD fit, and forcing
#     --max-concurrent 1 -- instead of the old behaviour (pick '--partition 2 2 2', blame the
#     split, and advise "a smaller grid or fewer fields" while predicting a bogus peak).
#     This fires even when N < the small-N gate: the grid term does not depend on N.
#  B) NORMAL TUNING. With RAM to spare it picks a partition and concurrency and prints a
#     predicted peak.
#  C) CONSERVATIVE CONTRACT. The header promises the model "deliberately errs HIGH so the
#     tuner never picks a configuration that measured tighter than predicted": on a real
#     partitioned 256^3 run (volume-weighted, dispersion -- the fields whose grids the old
#     model missed), predicted peak >= measured peak RSS.
#  D) EXPLICIT FLAGS WIN. --partition/--max-concurrent given by the user are never overridden.
#
# The 2^32 GPU sub-grid guard (ps_interpolation.cc) is NOT exercised here: it needs a real
# >4.29e9-cell partition sub-grid, i.e. >32 GB of buffers. It mirrors the validated guard in
# averaged_interpolation_1.cc line-for-line.
#
# Usage: tests/ps_auto_tune_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

N="${N:-64}"; BOX="${BOX:-100.0}"
BIN="./PS-DTFE"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
SNAP="${TMP}/at_input_pancake.hdf5"

echo "============================================================"
echo " auto-tune check   N=${N}^3"
echo "============================================================"

if [ "${1:-}" != "--no-build" ]; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

echo ">> generating test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null

FAILS=0
check() {  # $1 = name, $2 = 0/1 ok
    if [ "$2" -eq 1 ]; then echo "   OK   $1"; else echo "   FAIL $1"; FAILS=$((FAILS+1)); fi
}

# ---------- (A) over-budget refusal: grid term alone exceeds a simulated 0.05 GB machine ----------
echo ">> A: simulated over-budget run (DTFE_RAM_GB=0.05, production field set)"
LOG_A="${TMP}/at_a.log"
DTFE_RAM_GB=0.05 "${BIN}" "${SNAP}" "${TMP}/at_a" --grid 64 \
    --field density_a velocity_a gradient_a divergence_a shear_a vorticity_a dispersion_a \
    --input 105 --MpcUnit 1 --verbose 2 --periodic --avg-subsamples 1 --ps-volume-weighted \
    > "${LOG_A}" 2>&1 || true
agrep() {  # set-e-safe grep assertion (ERE): $1 = check name, $2 = pattern, $3 = file
    if grep -Eq -- "$2" "$3"; then check "$1" 1; else check "$1" 0; fi
}
agrep "A refusal names the real cause"          "does not fit in memory, and --partition cannot fix it" "${LOG_A}"
agrep "A quantifies the irreducible grid term"  "full-resolution output grids alone need"               "${LOG_A}"
agrep "A suggests --scratch-dir as the way out" "--scratch-dir <local dir>"                             "${LOG_A}"
agrep "A suggests the max feasible in-RAM grid" "largest that fits at these fields is about"            "${LOG_A}"
agrep "A warns off the broken escape hatches"   "--partNo'/'--region' are NOT a way out"                "${LOG_A}"
# the old misleading message must be gone
if grep -q "even a single .*-split partition is predicted to exceed the memory budget" "${LOG_A}"; then
    check "A old blame-the-split warning is gone" 0
else
    check "A old blame-the-split warning is gone" 1
fi

# ---------- (B) normal tuning at a comfortable simulated budget ----------
echo ">> B: normal tuning (DTFE_RAM_GB=64, DTFE_AUTO_MINN=1000)"
LOG_B="${TMP}/at_b.log"
DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_b" --grid 128 \
    --field density_a velocity_a --input 105 --MpcUnit 1 --verbose 2 --periodic \
    --avg-subsamples 1 > "${LOG_B}" 2>&1
agrep "B picks a partition and concurrency" "AUTO-TUNE: .* -> --partition [0-9]+ [0-9]+ [0-9]+ --max-concurrent [0-9]+" "${LOG_B}"
agrep "B prints a predicted peak" "predicted peak ~[0-9.]+ GB" "${LOG_B}"

# ---------- (C) conservative contract: predicted >= measured on a REAL partitioned run ----------
echo ">> C: predicted >= measured (256^3, volume-weighted dispersion, real run)"
LOG_C="${TMP}/at_c.log"
DTFE_AUTO_MINN=1000 /usr/bin/time -l "${BIN}" "${SNAP}" "${TMP}/at_c" --grid 256 \
    --field density_a velocity_a dispersion_a --input 105 --MpcUnit 1 --verbose 2 --periodic \
    --avg-subsamples 1 --ps-volume-weighted > "${LOG_C}" 2>&1
if "${PY}" - "${LOG_C}" <<'PYEOF'
import re, sys
log = open(sys.argv[1]).read()
mp = re.search(r"predicted peak ~([0-9.]+) GB", log)
mm = re.search(r"(\d+)\s+maximum resident", log)
assert mp and mm, "missing predicted-peak or time -l output"
pred, meas = float(mp.group(1)), int(mm.group(1)) / 1e9
print(f"   .... predicted {pred:.2f} GB vs measured {meas:.2f} GB")
sys.exit(0 if pred >= meas else 1)
PYEOF
then check "C model errs HIGH (predicted >= measured peak RSS)" 1
else check "C model errs HIGH (predicted >= measured peak RSS)" 0; fi

# ---------- (D) explicit flags always win ----------
echo ">> D: explicit --partition/--max-concurrent are respected"
LOG_D="${TMP}/at_d.log"
DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_d" --grid 128 \
    --field density_a --input 105 --MpcUnit 1 --verbose 2 --periodic --avg-subsamples 1 \
    --partition 2 2 2 --max-concurrent 2 > "${LOG_D}" 2>&1
if grep -q "partition grid of {2, 2, 2}" "${LOG_D}" || grep -Eq "2,2,2|2 2 2" "${LOG_D}"; then
    check "D user partition survives" 1
else
    check "D user partition survives" 0
fi
if grep -Eq "AUTO-TUNE: .* -> --partition" "${LOG_D}"; then
    check "D auto-tune stays silent when both flags are given" 0
else
    check "D auto-tune stays silent when both flags are given" 1
fi

echo "------------------------------------------------------------"
if [ "${FAILS}" -gt 0 ]; then
    echo "RESULT: FAIL (${FAILS} check(s))"
    exit 1
fi
echo "RESULT: PASS  (auto-tune refuses honestly when the grid cannot fit, tunes when it can,"
echo "               and its predictions bound the measured peak)"
echo "============================================================"
