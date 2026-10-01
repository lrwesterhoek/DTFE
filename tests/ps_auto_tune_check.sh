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
#  D) EXPLICIT FLAGS WIN. --partition/--max-concurrent given by the user are never overridden --
#     including '--partition 1 1 1', which asks for a single triangulation.
#  E) THE MEMORY BUDGET. 60% of RAM by default (80% let a 64 GB Mac swap 19 GB with a desktop
#     open next to the run); DTFE_MEM_FRACTION and DTFE_MEM_BUDGET_GB override it; a simulated
#     machine (DTFE_RAM_GB) skips the live "what do other programs leave free" term, and the live
#     run states its budget.
#  G) the stream estimate of a non-periodic cloud larger than its output region stays physical.
#  H) the TIME-AWARE split of a small set (2026-10-01): a run whose time is the sequential
#     triangulation (a slice of 0.26M particles) is split 2^3 for speed; a deposit-bound run
#     (a 256^3 grid) and a GPU run (its deposit serialises on the GPU) keep one domain, and so
#     does a non-periodic cloud that makes no periodic copies (the model charged it 1.7x: fixed);
#     the single-domain report predicts a sane peak (it said 52 GB for a 0.26M run: the GUI's
#     memory check believed it); and two auto-split runs give bit-identical grids (the split
#     merges its partitions in index order).
#  F) --auto-tune-report (the GUI's memory check): ONE machine-readable line, exit 0, nothing
#     written; a hand-set --partition/--max-concurrent is reported as given (the prediction uses
#     the concurrency the run will use); DTFE_AUTO_PTS_N sizes a plane that does not exist yet.
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
agrep "A quantifies the irreducible grid term"  "full-resolution output grids need"                     "${LOG_A}"
agrep "A suggests --scratch-dir as the way out" "--scratch-dir <local dir>"                             "${LOG_A}"
agrep "A suggests the max feasible in-RAM grid" "largest that fits at these fields is about"            "${LOG_A}"
agrep "A warns off the broken escape hatches"   "--partNo'/'--region' are NOT a way out"                "${LOG_A}"
# the old misleading message must be gone
if grep -q "even a single .*-split partition is predicted to exceed the memory budget" "${LOG_A}"; then
    check "A old blame-the-split warning is gone" 0
else
    check "A old blame-the-split warning is gone" 1
fi
# Over budget with a LARGE particle set (DTFE_AUTO_MINN=1000 makes this one count as large): the
# fallback must still SPLIT -- one partition at a time, the smallest per-partition triangulation --
# rather than build one unpartitioned triangulation on top of an already over-budget run (for
# 94M particles that was ~100 GB; seen on TNG100-3 with an 8192^2 --sample-points slice).
LOG_A2="${TMP}/at_a2.log"
DTFE_RAM_GB=0.05 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_a2" --grid 64 \
    --field density_a velocity_a gradient_a divergence_a shear_a vorticity_a dispersion_a \
    --input 105 --MpcUnit 1 --verbose 2 --periodic --avg-subsamples 1 --ps-volume-weighted \
    > "${LOG_A2}" 2>&1 || true
agrep "A over budget + large N: a real split at --max-concurrent 1" \
      "OVER BUDGET: --partition [2-6] [2-6] [2-6] --max-concurrent 1" "${LOG_A2}"

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
# '--partition 1 1 1' asks for ONE triangulation. It leaves partitionOn false (nothing is split),
# which the auto-tuner used to read as "not given" -- and then split the data anyway. Same
# settings as B, where the tuner does pick a split.
LOG_D1="${TMP}/at_d1.log"
DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_d1" --grid 128 \
    --field density_a --input 105 --MpcUnit 1 --verbose 2 --periodic --avg-subsamples 1 \
    --partition 1 1 1 > "${LOG_D1}" 2>&1
if grep -Eq "AUTO-TUNE: .* -> --partition ([2-9]|[1-9][0-9])" "${LOG_D1}"; then
    check "D '--partition 1 1 1' keeps one triangulation (no auto split)" 0
else
    check "D '--partition 1 1 1' keeps one triangulation (no auto split)" 1
fi

# ---------- (E) the memory budget ----------
echo ">> E: memory budget rules"
LOG_E="${TMP}/at_e.log"
run_e() {
    env "$@" DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_e" --grid 128 --field density_a velocity_a \
        --input 105 --MpcUnit 1 --verbose 2 --periodic --avg-subsamples 1 > "${LOG_E}" 2>&1
}
run_e DTFE_RAM_GB=64
agrep "E default budget = 60% of RAM (simulated 64 GB: no live term)" \
      "memory budget: 38\.4 GB = 60% of 64 GB RAM$" "${LOG_E}"
run_e DTFE_RAM_GB=64 DTFE_MEM_FRACTION=0.5
agrep "E DTFE_MEM_FRACTION sets the share" "memory budget: 32 GB = 50% of 64 GB RAM" "${LOG_E}"
run_e DTFE_RAM_GB=64 DTFE_MEM_BUDGET_GB=20
agrep "E DTFE_MEM_BUDGET_GB sets the budget outright" "memory budget: 20 GB set by DTFE_MEM_BUDGET_GB" "${LOG_E}"
run_e PATH="${PATH}"
agrep "E a live run states its budget (60% of RAM, or less when other programs use memory)" \
      "memory budget: [0-9.]+ GB( = 60% of|: other programs leave)" "${LOG_E}"

# ---------- (F) --auto-tune-report ----------
echo ">> F: --auto-tune-report"
LOG_F="${TMP}/at_f.log"
run_f() {
    env "$@" DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_f" --grid 64 \
        --field density_a velocity_a --input 105 --MpcUnit 1 --verbose 2 --periodic --avg-subsamples 1 \
        --auto-tune-report ${F_ARGS[@]+"${F_ARGS[@]}"} > "${LOG_F}" 2>&1
}
F_ARGS=()
rm -f "${TMP}"/at_f.*
run_f PATH="${PATH}"; rc=$?
check "F exits 0" "$([ "${rc}" = "0" ] && echo 1 || echo 0)"
check "F prints exactly one report line" "$([ "$(grep -c '^AUTO-TUNE-REPORT ' "${LOG_F}")" = "1" ] && echo 1 || echo 0)"
agrep "F the line carries the decision, prediction and budget" \
      "^AUTO-TUNE-REPORT partition=[0-9]+ mc=[0-9]+ predicted_gb=[0-9.]+ budget_gb=38\.40 .*over_budget=[01]$" "${LOG_F}"
check "F nothing is written next to the output root" \
      "$([ -z "$(ls "${TMP}"/at_f.* 2>/dev/null | grep -v 'at_f.log$')" ] && echo 1 || echo 0)"
F_ARGS=(--partition 2 2 2 --max-concurrent 3)
run_f PATH="${PATH}"
agrep "F a hand-set split and concurrency are reported as given" "^AUTO-TUNE-REPORT partition=2 mc=3 " "${LOG_F}"
F_ARGS=()
run_f PATH="${PATH}"
p0=$(sed -n 's/.* pts_gb=\([0-9.]*\) .*/\1/p' "${LOG_F}")
run_f DTFE_AUTO_PTS_N=4000000 DTFE_AUTO_PTS_VELGRAD=1
p1=$(sed -n 's/.* pts_gb=\([0-9.]*\) .*/\1/p' "${LOG_F}")
check "F DTFE_AUTO_PTS_N sizes a plane that is not on disk (pts ${p0} -> ${p1} GB)" \
      "$(awk -v a="${p0:-x}" -v b="${p1:-0}" 'BEGIN{print (a == 0 && b > 0.5) ? 1 : 0}')"

# ---------- (G) the stream estimate of a NON-periodic cloud with a smaller output region ----------
# The estimate used to tile the OUTPUT region (--box) and wrap initial positions into it: a cloud
# larger than the region folded far-apart particles into one patch, whose Delaunay tetrahedra then
# spanned the cloud in Eulerian space -- 8.1 million "streams" per point on a TNG100-3 Lagrangian
# sub-cube, so the tuner ran every point evaluation serially on the smallest split (222 s instead
# of seconds). It must tile the cloud's own Lagrangian bounding box.
echo ">> G: non-periodic stream estimate with a region smaller than the cloud"
SNAP_G="${TMP}/at_input_clump.hdf5"
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_G}" --n 48 --box "${BOX}" \
    --amplitude-factor 1.0 --margin-frac 0.2 >/dev/null
env DTFE_RAM_GB=64 DTFE_AUTO_PTS_N=1000000 "${BIN}" "${SNAP_G}" "${TMP}/at_g" --grid 16 --field density \
    --input 105 --MpcUnit 1 --verbose 2 --box 40 60 40 60 40 60 --auto-tune-report > "${TMP}/at_g.log" 2>&1
sg=$(sed -n 's/.* streams_est=\([0-9.]*\) .*/\1/p' "${TMP}/at_g.log")
# (below 1 is fine: for a non-periodic cloud the Eulerian/Lagrangian volume ratio is its mean
# compression -- this clump's is ~0.5 -- and the budget clamps the estimate at 1 stream per point;
# the old code gave 75593 here)
check "G the estimate stays physical: 0 < ${sg:-none} < 20 streams/point" \
      "$(awk -v s="${sg:-0}" 'BEGIN{print (s > 0 && s < 20) ? 1 : 0}')"

# ---------- (H) the time-aware split of a small set ----------
echo ">> H: time-aware split for small sets (speed, not memory)"
GPU_H=0; grep -q "METAL=1" o_ps/.build_mode 2>/dev/null && GPU_H=1
COMMON_H=(--input 105 --MpcUnit 1 --verbose 2 --auto-tune-report)
# H1: a slice of the 64^3 box (4.2M points, 16^3 grid) is triangulation-bound -> 2^3 partitions
env DTFE_RAM_GB=64 DTFE_AUTO_PTS_N=4194304 "${BIN}" "${SNAP}" "${TMP}/at_h1" --grid 16 --field density --periodic \
    "${COMMON_H[@]}" > "${TMP}/at_h1.log" 2>&1
agrep "H1 a triangulation-bound small run is split for speed" "^AUTO-TUNE-REPORT partition=[23] mc=[2-9]|^AUTO-TUNE-REPORT partition=[23] mc=[1-9][0-9]" "${TMP}/at_h1.log"
# H2: a 256^3 sampled grid is deposit-bound -> one domain, and its predicted peak is sane
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_h2" --grid 256 --field density_a velocity_a --periodic \
    "${COMMON_H[@]}" > "${TMP}/at_h2.log" 2>&1
agrep "H2 a deposit-bound small run keeps one domain" "^AUTO-TUNE-REPORT partition=1 " "${TMP}/at_h2.log"
ph=$(sed -n 's/.* predicted_gb=\([0-9.]*\) .*/\1/p' "${TMP}/at_h2.log")
check "H2 the single-domain report predicts a sane peak: ${ph:-none} GB < 12 (it used to say 52)" \
      "$(awk -v p="${ph:-99}" 'BEGIN{print (p > 0 && p < 12) ? 1 : 0}')"
# H3 (GPU builds): the GPU deposit serialises on one queue -> a split cannot speed it up
if [ "${GPU_H}" = 1 ]; then
    env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_h3" --grid 256 --field density_a velocity_a --periodic --ps-gpu \
        "${COMMON_H[@]}" > "${TMP}/at_h3.log" 2>&1
    agrep "H3 a GPU grid keeps one domain" "^AUTO-TUNE-REPORT partition=1 " "${TMP}/at_h3.log"
fi
# H4: the non-periodic clump (48^3) makes no periodic copies, so its single domain is cheap
# enough to keep (charged the periodic 1.7x it would be split)
env DTFE_RAM_GB=64 DTFE_AUTO_PTS_N=4194304 "${BIN}" "${SNAP_G}" "${TMP}/at_h4" --grid 16 --field density \
    "${COMMON_H[@]}" > "${TMP}/at_h4.log" 2>&1
agrep "H4 a non-periodic small cloud is costed without periodic copies (one domain)" "^AUTO-TUNE-REPORT partition=1 " "${TMP}/at_h4.log"
# H5: two auto-split runs (a 16^3 grid of the 64^3 box: triangulation-bound -> 2^3) give bit-identical grids
for r in a b; do
    "${BIN}" "${SNAP}" "${TMP}/at_h5$r" --grid 16 --field density --input 105 --MpcUnit 1 --periodic --verbose 0 > "${TMP}/at_h5$r.log" 2>&1
done
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_h5r" --grid 16 --field density --periodic "${COMMON_H[@]}" > "${TMP}/at_h5r.log" 2>&1
agrep "H5 the 16^3 grid run is a speed split" "^AUTO-TUNE-REPORT partition=[23] " "${TMP}/at_h5r.log"
if cmp -s "${TMP}/at_h5a.den" "${TMP}/at_h5b.den" && cmp -s "${TMP}/at_h5a.streams" "${TMP}/at_h5b.streams"; then
    check "H5 two auto-split runs are bit-identical (.den, .streams: ordered merge)" 1
else
    check "H5 two auto-split runs are bit-identical (.den, .streams: ordered merge)" 0
fi
rm -f "${TMP}"/at_h*.den "${TMP}"/at_h*.streams "${TMP}"/at_h*.hidden_streams

echo "------------------------------------------------------------"
if [ "${FAILS}" -gt 0 ]; then
    echo "RESULT: FAIL (${FAILS} check(s))"
    exit 1
fi
echo "RESULT: PASS  (auto-tune refuses honestly when the grid cannot fit, tunes when it can,"
echo "               its predictions bound the measured peak, splits a small set only when that"
echo "               is faster, and it keeps to a 60%-of-RAM budget that also leaves room for"
echo "               what other programs use)"
echo "============================================================"
