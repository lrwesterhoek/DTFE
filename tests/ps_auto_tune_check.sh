#!/usr/bin/env bash
# Auto-tune (src/auto_tune.h) behaviour checks.
# Drives the REAL binary through the documented env overrides (DTFE_RAM_GB simulates the
# machine's RAM, DTFE_AUTO_MINN lowers the particle-count gate) so every scenario is fast,
# deterministic and allocation-safe. Verified here:
#
#  A) OVER-BUDGET REFUSAL. When the irreducible full-grid accumulators alone exceed the
#     budget, auto-tune must say so BEFORE the search -- naming the grid term, stating that
#     --partition cannot reduce it, suggesting the largest grid that WOULD fit, and forcing
#     --max-concurrent 1 -- not pick '--partition 2 2 2', blame the split, and advise "a smaller
#     grid or fewer fields" while predicting a bogus peak.
#     This fires even when N < the small-N gate: the grid term does not depend on N.
#  B) NORMAL TUNING. With RAM to spare it picks a partition and concurrency and prints a
#     predicted peak.
#  C) CONSERVATIVE CONTRACT. The header promises the model "deliberately errs HIGH so the
#     tuner never picks a configuration that measured tighter than predicted": on a real
#     partitioned 256^3 run (volume-weighted, dispersion -- fields whose extra grids the model
#     must count), predicted peak >= measured peak RSS.
#  D) EXPLICIT FLAGS WIN. --partition/--max-concurrent given by the user are never overridden --
#     including '--partition 1 1 1', which asks for a single triangulation.
#  E) THE MEMORY BUDGET. 60% of RAM by default (80% lets a 64 GB Mac swap 19 GB with a desktop
#     open next to the run); DTFE_MEM_FRACTION and DTFE_MEM_BUDGET_GB override it; a simulated
#     machine (DTFE_RAM_GB) skips the live "what do other programs leave free" term, and the live
#     run states its budget.
#  G) the stream estimate of a non-periodic cloud larger than its output region stays physical.
#  H) the TIME-AWARE split of a small set: a run whose time is the sequential
#     triangulation (a slice of 0.26M particles) is split 2^3 for speed; a deposit-bound run
#     (a 256^3 grid) and a GPU run (its deposit serialises on the GPU) keep one domain, and so
#     does a non-periodic cloud that makes no periodic copies (the model must not charge it 1.7x);
#     the single-domain report predicts a sane peak (the GUI's memory check believes it); and two
#     auto-split runs give bit-identical grids (the split merges its partitions in index order).
#  I) Very fine grids: GPU sub-grids kept under 2^32 cells by the split; a small set on a huge grid
#     splits for memory instead of staying in one domain over budget.
#  F) --auto-tune-report (the GUI's memory check): ONE machine-readable line, exit 0, nothing
#     written; a hand-set --partition/--max-concurrent is reported as given (the prediction uses
#     the concurrency the run will use); DTFE_AUTO_PTS_N sizes a plane that does not exist yet.
#  J) the machine's free memory over a plan: an explicit budget capped by it, the partition memory gate.
#  M) the GPU deposit's host snapshot (ps_metal_host.cc GpuSnapshot) is in the prediction, ONCE (the deposits serialise).
#  N) the '--sample-points' stream estimate is made on the points' own slab: on a 1-D pancake
#     (80^3, A k = 1.8; the 3-stream band covers ~16% of the box, box mean 1.33) a plane outside the band
#     estimates ~1 stream/point and one through it ~3 (both measured exactly in '.pts_streams'), each within
#     10%; the box-wide estimate (DTFE_AUTO_PTS_BOXMEAN=1) gives ~1.3 for both. On the TNG50-3-Dark
#     z=0 z2048 plane the box mean 17.98 budgets 34.95 streams/point (19.3 GB) while its points average 5.38.
#  O) '--parallel-triangulation' (TBB builds) is costed: its insert copies (+203 B per vertex in flight,
#     measured) raise the predicted peak. SKIP without TBB.
#  P) the tessellation model pads a partition by the binary's DEFAULT Lagrangian padding ('--ps-lag-padding
#     8' spacings, not the Eulerian '--padding' fraction floored at 0.3 of a partition): the default predicts what
#     an explicit '--ps-lag-padding 8' does, and '--ps-lag-padding 0' (the Eulerian rule) predicts something else.
#  K) partitions in flight never outnumber the threads: neither a hand-set split with the
#     concurrency left to the tuner nor '--max-concurrent 0' ("all threads") starts every
#     partition at once (64 threads for '--partition 4 4 4').
#  R) STANDARD DTFE '--region' / '--regionMpc': the tuner plans the region's particles, not the whole
#     snapshot's (planning the snapshot splits the launcher's exact zoom of a 20^3-cell TNG50 region 3x3x3 with
#     10 threads: ~21 min instead of ~1). Uses the standard binary (built here too); SKIP when it is not built.
#  S) ONE DOMAIN: '--partition 1 1 1' deposits into the run's own grids, so the prediction counts them
#     once -- the 768^3 crossed-waves report sits between the measured peak and 15% above it --
#     and a real one-domain run's peak stays under its prediction.
#
# The 2^32 GPU sub-grid guard (ps_interpolation.cc) is NOT exercised here: it needs a real
# >4.29e9-cell partition sub-grid, i.e. >32 GB of buffers. It mirrors the validated guard in
# averaged_interpolation_1.cc line-for-line.
#
# Portable: runs on Linux too (tests/ci_suite.sh core) -- the peak RSS from getrusage, no macOS-only
# command; the GPU-only parts (H3, I2/I3, L, M) SKIP on a CPU build.
#
# Usage: tests/ps_auto_tune_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-64}"; BOX="${BOX:-100.0}"
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"; mkdir -p "${TMP}"
SNAP="${TMP}/at_input_pancake.hdf5"

echo "============================================================"
echo " auto-tune check   N=${N}^3"
echo "============================================================"

if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
    echo ">> building DTFE (section R) ..."
    make DTFE $(cat o/.build_mode 2>/dev/null || true) >/dev/null
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
# the misleading blame-the-split message must not appear
if grep -q "even a single .*-split partition is predicted to exceed the memory budget" "${LOG_A}"; then
    check "A no blame-the-split warning" 0
else
    check "A no blame-the-split warning" 1
fi
# Over budget with a LARGE particle set (DTFE_AUTO_MINN=1000 makes this one count as large): the
# fallback must still SPLIT -- one partition at a time, the smallest per-partition triangulation --
# rather than build one unpartitioned triangulation on top of an already over-budget run (for
# 94M particles that is ~100 GB, e.g. TNG100-3 with an 8192^2 --sample-points slice).
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
# The peak RSS comes from getrusage(RUSAGE_CHILDREN) in a python parent, not '/usr/bin/time -l' (macOS
# only; GNU time spells it -v and is not even installed on a bare Ubuntu): ru_maxrss is in BYTES on macOS
# and in KILOBYTES on Linux.
peak_rss_run() {  # $1 = log; the rest = the command. Appends 'PEAK_RSS_BYTES <n>' to the log; exits as the command
    "${PY}" - "$@" <<'PYEOF'
import resource, subprocess, sys
with open(sys.argv[1], "w") as log:
    rc = subprocess.call(sys.argv[2:], stdout=log, stderr=subprocess.STDOUT)
    kb_or_b = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    log.write(f"\nPEAK_RSS_BYTES {kb_or_b if sys.platform == 'darwin' else kb_or_b * 1024}\n")
sys.exit(rc)
PYEOF
}
peak_rss_run "${LOG_C}" env DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_c" --grid 256 \
    --field density_a velocity_a dispersion_a --input 105 --MpcUnit 1 --verbose 2 --periodic \
    --avg-subsamples 1 --ps-volume-weighted
if "${PY}" - "${LOG_C}" <<'PYEOF'
import re, sys
log = open(sys.argv[1]).read()
mp = re.search(r"predicted peak ~([0-9.]+) GB", log)
mm = re.search(r"PEAK_RSS_BYTES (\d+)", log)
assert mp and mm, "missing predicted-peak or peak-RSS output"
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
# which the auto-tuner must not read as "not given" (and then split the data anyway). Same
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
run_f PATH="${PATH}" && rc=0 || rc=$?       # (a bare 'run_f; rc=$?' aborts under set -e before the check)
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
# The estimate tiles the cloud's own Lagrangian bounding box. Tiling the OUTPUT region (--box) and
# wrapping initial positions into it folds far-apart particles of a cloud larger than the region into
# one patch, whose Delaunay tetrahedra then span the cloud in Eulerian space -- 8.1 million "streams"
# per point on a TNG100-3 Lagrangian sub-cube, so the tuner would run every point evaluation serially
# on the smallest split (222 s instead of seconds).
echo ">> G: non-periodic stream estimate with a region smaller than the cloud"
SNAP_G="${TMP}/at_input_clump.hdf5"
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_G}" --n 48 --box "${BOX}" \
    --amplitude-factor 1.0 --margin-frac 0.2 >/dev/null
env DTFE_RAM_GB=64 DTFE_AUTO_PTS_N=1000000 "${BIN}" "${SNAP_G}" "${TMP}/at_g" --grid 16 --field density \
    --input 105 --MpcUnit 1 --verbose 2 --box 40 60 40 60 40 60 --auto-tune-report > "${TMP}/at_g.log" 2>&1
sg=$(sed -n 's/.* streams_est=\([0-9.]*\) .*/\1/p' "${TMP}/at_g.log")
# (below 1 is fine: for a non-periodic cloud the Eulerian/Lagrangian volume ratio is its mean
# compression -- this clump's is ~0.5 -- and the budget clamps the estimate at 1 stream per point)
check "G the estimate stays physical: 0 < ${sg:-none} < 20 streams/point" \
      "$(awk -v s="${sg:-0}" 'BEGIN{print (s > 0 && s < 20) ? 1 : 0}')"

# ---------- (H) the time-aware split of a small set ----------
echo ">> H: time-aware split for small sets (speed, not memory)"
GPU_H=0; grep -Eq "METAL=1|CUDA=1|HIP=1" o_ps/.build_mode 2>/dev/null && GPU_H=1   # (-E: GNU and BSD grep alike)
COMMON_H=(--input 105 --MpcUnit 1 --verbose 2 --auto-tune-report)
# H1: a slice of the 64^3 box (4.2M points, 16^3 grid) is triangulation-bound -> 2^3 partitions. The split pays
# only with enough threads (on 4 it rightly stays one domain), so the thread
# count is pinned to a 10-core machine's: the report path only plans, it runs nothing
env OMP_NUM_THREADS=10 DTFE_RAM_GB=64 DTFE_AUTO_PTS_N=4194304 "${BIN}" "${SNAP}" "${TMP}/at_h1" --grid 16 --field density --periodic \
    "${COMMON_H[@]}" > "${TMP}/at_h1.log" 2>&1
agrep "H1 a triangulation-bound small run is split for speed" "^AUTO-TUNE-REPORT partition=[23] mc=[2-9]|^AUTO-TUNE-REPORT partition=[23] mc=[1-9][0-9]" "${TMP}/at_h1.log"
# H2: a 256^3 sampled grid is deposit-bound -> one domain, and its predicted peak is sane
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_h2" --grid 256 --field density_a velocity_a --periodic \
    "${COMMON_H[@]}" > "${TMP}/at_h2.log" 2>&1
agrep "H2 a deposit-bound small run keeps one domain" "^AUTO-TUNE-REPORT partition=1 " "${TMP}/at_h2.log"
ph=$(sed -n 's/.* predicted_gb=\([0-9.]*\) .*/\1/p' "${TMP}/at_h2.log")
check "H2 the single-domain report predicts a sane peak: ${ph:-none} GB < 12" \
      "$(awk -v p="${ph:-99}" 'BEGIN{print (p > 0 && p < 12) ? 1 : 0}')"
# H3 (GPU builds): the GPU deposit serialises on one queue -> a split cannot speed it up
if [ "${GPU_H}" = 1 ]; then
    env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_h3" --grid 256 --field density_a velocity_a --periodic --ps-gpu \
        "${COMMON_H[@]}" > "${TMP}/at_h3.log" 2>&1
    agrep "H3 a GPU grid keeps one domain" "^AUTO-TUNE-REPORT partition=1 " "${TMP}/at_h3.log"
else
    echo "   SKIP  H3 (this build has no GPU backend)"
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

# ---------- (I) very fine grids: the GPU's 2^32-cell sub-grid cap, and a small set's memory ----------
# The GPU kernels number a partition's sub-grid cells with 32 bits: above 2^32 the deposit falls back to
# the CPU. The tuner splits to keep every GPU sub-grid under the cap (a 1700^3 grid: 2^3) and warns
# about a user split that does not. And a small particle set on such a grid does not stay in one
# domain when that domain's deposit buffers do not fit (83 GB unsplit): it splits for memory.
echo ">> I: very fine grids (1700^3, grids on scratch)"
ISCR="${TMPDIR:-/tmp}/dtfe-at-scratch"; mkdir -p "${ISCR}"
COMMON_I=(--grid 1700 --periodic --field density_a --input 105 --MpcUnit 1 --verbose 2 --scratch-dir "${ISCR}" --auto-tune-report)
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_i_cpu" "${COMMON_I[@]}" > "${TMP}/at_i_cpu.log" 2>&1
pi=$(sed -n 's/^AUTO-TUNE-REPORT partition=\([0-9]*\) .*/\1/p' "${TMP}/at_i_cpu.log" | head -1)
ob=$(sed -n 's/^AUTO-TUNE-REPORT .*over_budget=\([01]\).*/\1/p' "${TMP}/at_i_cpu.log" | head -1)
check "I1 CPU: a small set on a 1700^3 grid splits for memory (partition ${pi:-none}, over_budget ${ob:-none})" \
      "$([ -n "${pi}" ] && [ "${pi}" -ge 2 ] && [ "${ob}" = "0" ] && echo 1 || echo 0)"
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_i_gpu" "${COMMON_I[@]}" --ps-gpu > "${TMP}/at_i_gpu.log" 2>&1
if grep -q "GPU (--ps-gpu)" "${TMP}/at_i_gpu.log"; then
    pg=$(sed -n 's/^AUTO-TUNE-REPORT partition=\([0-9]*\) .*/\1/p' "${TMP}/at_i_gpu.log" | head -1)
    check "I2 GPU: the tuner splits a 1700^3 grid so each sub-grid stays under 2^32 cells (partition ${pg:-none})" \
          "$([ -n "${pg}" ] && [ "${pg}" -ge 2 ] && grep -q "under the 2^32 cells the GPU kernels number" "${TMP}/at_i_gpu.log" && echo 1 || echo 0)"
    env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_i_user" "${COMMON_I[@]}" --ps-gpu --partition 1 1 1 > "${TMP}/at_i_user.log" 2>&1
    agrep "I3 GPU: a user's unsplit 1700^3 grid is told which split keeps the GPU" "above the 2\^32 the GPU kernels number" "${TMP}/at_i_user.log"
else
    echo "   SKIP  I2/I3 (this build has no GPU backend)"
fi
rm -rf "${ISCR}"

# ---------- (J) the machine's free memory over a plan ----------
# An explicit budget must not skip the free-memory rule: DTFE_MEM_BUDGET_GB=52 on a 64 GB Mac gets a TNG50-3-Dark
# z=0 run SIGKILLed by macOS once other programs take memory. DTFE_FREE_GB simulates what the
# other programs leave free (the OS query's test hook).
echo ">> J: an explicit budget capped by the free memory; the partition memory gate"
run_e DTFE_FREE_GB=10 DTFE_MEM_BUDGET_GB=40
agrep "J1 DTFE_MEM_BUDGET_GB above the free memory is capped (10 GB free - 4 GB headroom = 6 GB), saying so" \
      "memory budget: 6 GB: DTFE_MEM_BUDGET_GB asks 40 GB, but the other programs leave 10 GB free, minus 4 GB headroom" "${LOG_E}"
run_e DTFE_FREE_GB=10 DTFE_MEM_BUDGET_GB=5
agrep "J2 a budget the free memory holds stays as set" "memory budget: 5 GB set by DTFE_MEM_BUDGET_GB" "${LOG_E}"
run_e DTFE_FREE_GB=44.2 DTFE_MEM_BUDGET_GB=40.5
agrep "J2b ... and so does one the free memory would trim by under 0.5 GB (44.2 - 4 vs 40.5): no 'but' line" \
      "memory budget: 40.5 GB set by DTFE_MEM_BUDGET_GB" "${LOG_E}"
run_e DTFE_RAM_GB=64 DTFE_FREE_GB=10 DTFE_MEM_BUDGET_GB=40
agrep "J3 a simulated machine (DTFE_RAM_GB) skips the live cap: tests stay deterministic" \
      "memory budget: 40 GB set by DTFE_MEM_BUDGET_GB" "${LOG_E}"
# the gate: a hand-set split with the concurrency left to the tuner (its report sizes one partition), the
# same run with ~no free memory: every partition beyond the first waits, the run completes, the grids agree
run_j() {
    env "$@" DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_j" --grid 64 --field density_a \
        --input 105 --MpcUnit 1 --verbose 2 --periodic --avg-subsamples 1 --partition 2 2 2 > "${TMP}/at_j.log" 2>&1
}
run_j DTFE_FREE_GB=1000                      # ample by decree: the live reading would make J5b depend on the machine
for f in a_den a_streams; do cp "${TMP}/at_j.${f}" "${TMP}/at_j_free.${f}" 2>/dev/null || true; done
cp "${TMP}/at_j.log" "${TMP}/at_j_free.log"
rm -f "${TMP}"/at_j.a_*                      # a gated run that writes nothing must not compare the free one
run_j DTFE_FREE_GB=0.001 && rc=0 || rc=$?    # (a bare 'run_j; rc=$?' aborts under set -e: J4 could never fail)
mcj=$(sed -n 's/.*building up to \([0-9]*\) partition triangulation.*/\1/p' "${TMP}/at_j.log" | head -1)
if [ "${mcj:-1}" -ge 2 ]; then
    check "J4 low free memory: the run still completes (exit ${rc})" "$([ "${rc}" = "0" ] && echo 1 || echo 0)"
    agrep "J5 ... partitions wait for one in flight to finish, and it says so" "PS-DTFE memory gate: partition [0-9]+ waits" "${TMP}/at_j.log"
    if grep -q "PS-DTFE memory gate" "${TMP}/at_j_free.log"; then
        check "J5b with the memory free the gate holds nothing" 0
    else
        check "J5b with the memory free the gate holds nothing" 1
    fi
    if "${PY}" - "${TMP}" <<'PYEOF'
import os, sys, numpy as np
t = sys.argv[1]
real = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))   # the grids are float64 in double mode
worst = 0.0
for f in ("a_den", "a_streams"):
    a = np.fromfile(f"{t}/at_j_free.{f}", real)
    b = np.fromfile(f"{t}/at_j.{f}", real)
    assert a.size == b.size and a.size > 0, f
    worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-30))))
print(f"   .... gated vs free: max rel {worst:.1e} (the merge's float sums, in index order)")
sys.exit(0 if worst < 1e-5 else 1)
PYEOF
    then check "J6 ... and its grids equal the ungated run's to float rounding" 1
    else check "J6 ... and its grids equal the ungated run's to float rounding" 0; fi
else
    echo "   SKIP  J4-J6 (the tuner ran this split one partition at a time: nothing to gate)"
fi

# ---------- (K) partitions in flight never outnumber the threads ----------
# OMP_NUM_THREADS=2 makes the thread count the machine's own; '--partition 4 4 4' has 64 partitions. A small set
# with the split hand-set and the concurrency left to the tuner (K1), and '--max-concurrent 0' = all threads (K2),
# would each start all 64 at once.
echo ">> K: partitions in flight are capped at the threads"
run_k() {
    env "$@" OMP_NUM_THREADS=2 DTFE_RAM_GB=64 DTFE_FREE_GB=1000 "${BIN}" "${SNAP}" "${TMP}/at_k" --grid 16 --field density \
        --input 105 --MpcUnit 1 --verbose 2 --periodic --partition 4 4 4 ${K_ARGS[@]+"${K_ARGS[@]}"} > "${TMP}/at_k.log" 2>&1
}
for k in K1 K2; do
    if [ "${k}" = K1 ]; then K_ARGS=(); what="a hand-set split, the concurrency left to the tuner"
    else K_ARGS=(--max-concurrent 0); what="'--max-concurrent 0'"; fi
    run_k PATH="${PATH}" && rc=0 || rc=$?
    mck=$(sed -n 's/.*building up to \([0-9]*\) partition triangulation.*/\1/p' "${TMP}/at_k.log" | head -1)
    check "${k} ${what}: at most 2 of the 64 partitions at once on 2 threads (${mck:-none}, exit ${rc})" \
          "$([ "${rc}" = "0" ] && [ -n "${mck}" ] && [ "${mck}" -le 2 ] && echo 1 || echo 0)"
done
K_ARGS=()
rm -f "${TMP}"/at_k.*

# ---------- (L) the look-ahead and the budget agree ----------
# At --max-concurrent 1 with the GPU deposit the run prepares the next partition during the deposit. Its bytes
# belong in the prediction the split is chosen by: added after it, a plan that fits reports over_budget=1 and
# the launcher's memory check halves a plane that fits. A budget between 'one partition' and 'one partition + the
# look-ahead' must switch the look-ahead off and report a prediction within the budget.
echo ">> L: the look-ahead is counted against the budget"
run_l() {  # $1 = budget in GB, $2 = log
    env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 DTFE_AUTO_MINN=1000 DTFE_MEM_BUDGET_GB="$1" "${BIN}" "${SNAP}" "${TMP}/at_l" \
        --grid 64 --field density_a --input 105 --MpcUnit 1 --verbose 2 --periodic --ps-gpu \
        --partition 2 2 2 --max-concurrent 1 --auto-tune-report > "$2" 2>&1 || true
}
run_l 50 "${TMP}/at_l1.log"
if grep -q "GPU (--ps-gpu)" "${TMP}/at_l1.log"; then
    p1=$(sed -n 's/^AUTO-TUNE-REPORT .*predicted_gb=\([0-9.]*\) .*/\1/p' "${TMP}/at_l1.log" | head -1)
    # just under 'one partition + the look-ahead' (~0.1 GB at 2^3 of 64^3 particles), above one partition
    bl=$(awk -v p="${p1:-0}" 'BEGIN{printf "%.4f", p - 0.03}')
    run_l "${bl}" "${TMP}/at_l2.log"
    pl=$(sed -n 's/^AUTO-TUNE-REPORT .*predicted_gb=\([0-9.]*\) .*/\1/p' "${TMP}/at_l2.log" | head -1)
    ol=$(sed -n 's/^AUTO-TUNE-REPORT .*over_budget=\([01]\).*/\1/p' "${TMP}/at_l2.log" | head -1)
    check "L a budget ${bl} GB under one partition + look-ahead (${p1:-none} GB): look-ahead off, predicted ${pl:-none} GB, over_budget ${ol:-none}" \
          "$([ "${ol}" = "0" ] && grep -q "look-ahead off for this run" "${TMP}/at_l2.log" \
             && ! grep -q "look-ahead off" "${TMP}/at_l1.log" \
             && awk -v p="${pl:-99}" -v b="${bl}" 'BEGIN{exit !(p <= b)}' && echo 1 || echo 0)"
else
    echo "   SKIP  L (this build has no GPU backend)"
fi
rm -f "${TMP}"/at_l*

# ---------- (M) the GPU deposit's host snapshot is in the prediction, once ----------
# The Metal deposit keeps a host copy of its accumulator buffers (ps_metal_host.cc GpuSnapshot: the grids, the
# 8 B fixed-point mass, the per-tet masses) to resume a failed command buffer: ~4.7 GB
# per partition at 1024^3 split 2^3. The deposits run one at a time, so the copy is counted ONCE whatever the
# concurrency: predicted(c) = fixed + c x per-partition + snapshot, i.e. snapshot = p(2) - fixed - 2 (p(3) - p(2)).
echo ">> M: the GPU deposit's snapshot is counted once"
run_m() {  # $1 = --max-concurrent, $2 = log
    env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 "${BIN}" "${SNAP}" "${TMP}/at_m" --grid 256 \
        --field density_a velocity_a --ps-volume-weighted --input 105 --MpcUnit 1 --verbose 2 --periodic --ps-gpu \
        --partition 2 2 2 --max-concurrent "$1" --auto-tune-report > "$2" 2>&1 || true
}
run_m 2 "${TMP}/at_m2.log"
if grep -q "GPU (--ps-gpu)" "${TMP}/at_m2.log"; then
    run_m 3 "${TMP}/at_m3.log"
    gb() { sed -n "s/^AUTO-TUNE-REPORT .*$1=\([0-9.]*\) .*/\1/p" "$2" | head -1; }
    p2=$(gb predicted_gb "${TMP}/at_m2.log"); p3=$(gb predicted_gb "${TMP}/at_m3.log"); f2=$(gb fixed_gb "${TMP}/at_m2.log")
    # expected: (24 B of selected grids + 4 B of the 64-bit mass) x the (256/2 + 40)^3 sub-grid cells = 0.133 GB
    # (+ the per-tet masses; measured 0.114 GB) -- with the velocity moment and its weight (16 B) as
    # 64-bit words, 16 B more: 0.209 GB (measured 0.190 GB). PS_METAL_FLOAT_MOMENTS=1 keeps the floats.
    sm=$(awk -v a="${p2:-0}" -v b="${p3:-0}" -v f="${f2:-0}" 'BEGIN{printf "%.3f", a - f - 2*(b - a)}')
    if [ "${PS_METAL_FLOAT_MOMENTS:-0}" != "0" ] || [ "${PS_GPU_FLOAT_MOMENTS:-0}" != "0" ]; then lo=0.10; hi=0.17; ex=0.13
    else lo=0.16; hi=0.25; ex=0.21; fi
    check "M the GPU snapshot is counted once: ${sm} GB beyond 2 partitions (expected ~${ex} GB; predicted ${p2:-none} / ${p3:-none} GB at 2 / 3 in flight)" \
          "$(awk -v s="${sm}" -v lo="${lo}" -v hi="${hi}" 'BEGIN{exit !(s >= lo && s <= hi)}' && echo 1 || echo 0)"
else
    echo "   SKIP  M (this build has no GPU backend)"
fi
rm -f "${TMP}"/at_m*

# ---------- (N) the stream estimate on the sample points' slab ----------
echo ">> N: the sample-point stream estimate on the plane's own slab"
SNAP_N="${TMP}/at_input_pancake80.hdf5"
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_N}" --n 80 --box "${BOX}" \
    --amplitude-factor 1.8 --jitter-frac 0.02 --seed 7 >/dev/null
"${PY}" - "${TMP}" "${BOX}" <<'PYEOF'
import numpy as np, sys
tmp, box = sys.argv[1], float(sys.argv[2])
u = (np.arange(64) + 0.5) * box / 64
Y, Z = np.meshgrid(u, u, indexing="ij")
for name, x in (("far", 0.1 * box), ("fold", 0.5 * box)):    # outside / through the 3-stream band
    np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1).astype(np.float64).tofile(f"{tmp}/at_n_{name}.bin")
PYEOF
est_n() {  # $1 = plane, $2 = DTFE_AUTO_PTS_BOXMEAN
    env DTFE_RAM_GB=64 DTFE_AUTO_PTS_BOXMEAN="$2" "${BIN}" "${SNAP_N}" "${TMP}/at_n" --grid 32 --periodic --input 105 \
        --MpcUnit 1 --field density_a --sample-points "${TMP}/at_n_$1.bin" --auto-tune-report 2>/dev/null \
        | sed -n 's/^AUTO-TUNE-REPORT .* streams_est=\([0-9.]*\) .*/\1/p'
}
nf=$(est_n far 0); nd=$(est_n fold 0); nb=$(est_n far 1)
check "N a plane outside the fold estimates ${nf:-none} streams/point (true 1, within 10%)" \
      "$(awk -v s="${nf:-0}" 'BEGIN{print (s > 0.9 && s < 1.1) ? 1 : 0}')"
check "N a plane through the fold estimates ${nd:-none} streams/point (true 3, within 10%)" \
      "$(awk -v s="${nd:-0}" 'BEGIN{print (s > 2.7 && s < 3.3) ? 1 : 0}')"
check "N the box-wide estimate (DTFE_AUTO_PTS_BOXMEAN=1) is the box mean, ${nb:-none} (true 1.33)" \
      "$(awk -v s="${nb:-0}" 'BEGIN{print (s > 1.2 && s < 1.45) ? 1 : 0}')"
rm -f "${TMP}"/at_n*

# ---------- (O) the parallel insert's transient ----------
echo ">> O: --parallel-triangulation is in the predicted peak"
if grep -q -- "--parallel-triangulation" <<< "$("${BIN}" --help 2>&1)"; then
    pred_o() {
        env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_o" --grid 32 --periodic --input 105 --MpcUnit 1 --field density \
            --partition 2 2 2 --max-concurrent 2 --auto-tune-report "$@" 2>/dev/null \
            | sed -n 's/^AUTO-TUNE-REPORT .*predicted_gb=\([0-9.]*\) .*/\1/p'
    }
    o0=$(pred_o); o1=$(pred_o --parallel-triangulation)
    check "O the parallel insert raises the predicted peak (${o0:-none} -> ${o1:-none} GB)" \
          "$(awk -v a="${o0:-0}" -v b="${o1:-0}" 'BEGIN{print (a > 0 && b > a) ? 1 : 0}')"
else
    echo "   SKIP  O (no TBB in this build)"
fi

# ---------- (P) the default Lagrangian padding is what the tuner models ----------
# A hand-set 4^3 split of the 64^3 set (report path, runs nothing): k = 8 spacings pad each side by 0.5 of a
# partition, the Eulerian rule by max(0.3, padding fraction x 4) ~ 0.31, so the two predictions differ.
echo ">> P: the tuner models the default '--ps-lag-padding 8'"
pred_p() {
    env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_p" --grid 32 --periodic --input 105 --MpcUnit 1 --field density \
        --partition 4 4 4 --max-concurrent 4 --auto-tune-report "$@" 2>/dev/null \
        | sed -n 's/^AUTO-TUNE-REPORT .*predicted_gb=\([0-9.]*\) .*/\1/p'
}
pd=$(pred_p); p8=$(pred_p --ps-lag-padding 8); p0=$(pred_p --ps-lag-padding 0)
check "P the default predicts like '--ps-lag-padding 8' (${pd:-none} = ${p8:-none} GB), not like 0 (${p0:-none} GB)" \
      "$(awk -v d="${pd:-x}" -v e="${p8:-y}" -v z="${p0:-x}" 'BEGIN{print (d != "x" && d == e && d != z) ? 1 : 0}')"

# ---------- (R) standard DTFE --region: the tuner plans the region, not the snapshot ----------
# A '--region'/'--regionMpc' run triangulates only the region plus its padding, so the standard binary's tuner
# plans the region's particle count and box; planning the whole snapshot's turns the launcher's exact zoom of a
# 20^3-cell TNG50 region into '--partition 3 3 3, 10 thread(s)' -- 27 x 10 sub-triangulations, each padded again
# around its tiny sub-box. On a simulated 0.2 GB machine (the standard tuner has no small-N gate): the whole 64^3 box
# needs a split (the control, so the machine is tight enough to show the difference); a 10 x 10 x 2 Mpc region of
# 264 particles keeps one domain with every thread (the tuner stays silent); a region that IS the box plans exactly
# like the box; and a real region run left to the tuner is byte-identical to one with '--partition 1 1 1' (a tuner
# planning the snapshot cuts that one to ONE thread, with a bogus 'no split fits' warning, and splits the other).
echo ">> R: standard DTFE --region plans the region's particles (${DTFE_BIN})"
if [ -x "${DTFE_BIN}" ]; then
    std_r() {  # $1 = log, rest = extra args (report mode: nothing is written)
        local log="$1"; shift
        env DTFE_RAM_GB=0.2 "${DTFE_BIN}" "${SNAP}" "${TMP}/at_r" --grid 16 16 1 --periodic --input 105 --MpcUnit 1 \
            --field density_a --auto-tune-report "$@" > "${log}" 2>&1 || true
    }
    std_r "${TMP}/at_r_box.log"
    std_r "${TMP}/at_r_reg.log" --regionMpc 40 50 40 50 49 51
    std_r "${TMP}/at_r_all.log" --region 0 1 0 1 0 1
    agrep "R control: the whole box needs a split on this simulated machine" \
          "AUTO-TUNE: .* -> --partition [2-6] [2-6] [2-6]" "${TMP}/at_r_box.log"
    r_reg="$(grep -o 'AUTO-TUNE: .*' "${TMP}/at_r_reg.log" || true)"
    check "R a 10x10x2 Mpc region keeps one domain and every thread (no AUTO-TUNE line; got: '${r_reg}')" \
          "$([ -z "${r_reg}" ] && grep -q 'AUTO-TUNE-REPORT' "${TMP}/at_r_reg.log" && echo 1 || echo 0)"
    r_box="$(grep -o 'AUTO-TUNE: .*' "${TMP}/at_r_box.log" || true)"
    r_all="$(grep -o 'AUTO-TUNE: .*' "${TMP}/at_r_all.log" || true)"
    check "R a region that is the whole box plans exactly like the box" \
          "$([ -n "${r_box}" ] && [ "${r_box}" = "${r_all}" ] && echo 1 || echo 0)"
    run_r() {  # $1 = output root, rest = extra args: a real (CPU) region run on the same simulated machine
        local out="$1"; shift
        env DTFE_RAM_GB=0.2 "${DTFE_BIN}" "${SNAP}" "${out}" --grid 32 32 4 --periodic --input 105 --MpcUnit 1 \
            --field density_a velocity_a --regionMpc 30 60 30 60 45 50 "$@" > "${out}.log" 2>&1 || true
    }
    run_r "${TMP}/at_r_auto"
    run_r "${TMP}/at_r_p1" --partition 1 1 1
    check "R a real region run left to the tuner == the same run with '--partition 1 1 1', byte for byte" \
          "$([ -s "${TMP}/at_r_auto.a_den" ] && cmp -s "${TMP}/at_r_auto.a_den" "${TMP}/at_r_p1.a_den" \
             && cmp -s "${TMP}/at_r_auto.a_vel" "${TMP}/at_r_p1.a_vel" && ! grep -q 'AUTO-TUNE:' "${TMP}/at_r_p1.log" \
             && echo 1 || echo 0)"
    rm -f "${TMP}"/at_r*
else
    echo "   SKIP  R (${DTFE_BIN} is not built)"
fi

# ---------- (S) one domain: its grids are counted once ----------
# '--partition 1 1 1' (and one domain picked by the tuner) deposits straight into the run's own grids, which 'fixed'
# books, so the partition term must not add them again (nor every fixed-point grid's conversion at once). The 64^3
# crossed waves on a 768^3 grid measure 22.9 GB peak footprint (24.87 GB in the double build, whose peak
# is the export). S1 pins that report (the run itself takes 23 GB): at or above the measurement, within 15% of it; the
# threads and the machine are pinned (the copy budget and the threads' sample lists follow both). S2 runs one domain
# for real and its peak must stay under the prediction -- the model's contract; no upper bound there: the model follows
# macOS's allocator, which keeps freed blocks under 4 GiB, and glibc returns them (the Linux peak is lower).
echo ">> S: one domain (the 768^3 report; a real 256^3 run)"
SNAP_W="${TMP}/at_input_waves.hdf5"
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_W}" --n 64 --crossed-waves >/dev/null
env OMP_NUM_THREADS=10 DTFE_RAM_GB=64 "${BIN}" "${SNAP_W}" "${TMP}/at_s1" --grid 768 --periodic --input 105 --MpcUnit 1 \
    --avg-subsamples 1 --field density_a velocity_a --partition 1 1 1 --auto-tune-report > "${TMP}/at_s1.log" 2>&1
ps1=$(sed -n 's/^AUTO-TUNE-REPORT .*predicted_gb=\([0-9.]*\) .*/\1/p' "${TMP}/at_s1.log" | head -1)
if precision_double; then ms1=24.87; else ms1=22.9; fi
check "S1 one domain at 768^3 predicts ${ps1:-none} GB: at least the measured ${ms1} GB, at most 15% above it" \
      "$(awk -v p="${ps1:-0}" -v m="${ms1}" 'BEGIN{print (p >= m && p <= 1.15*m) ? 1 : 0}')"
"${BIN}" "${SNAP_W}" "${TMP}/at_s2" --grid 256 --periodic --input 105 --MpcUnit 1 --verbose 1 \
    --avg-subsamples 1 --field density_a velocity_a --partition 1 1 1 > "${TMP}/at_s2.log" 2>&1 || true
ms2=$("${PY}" - "${TMP}/at_s2.log" <<'PYEOF'
import re, sys
m = re.search(r"memory model\s*:\s*([\d.]+) GB predicted \(auto-tune\) vs ([\d.]+) GB measured", open(sys.argv[1]).read())
print(f"{m.group(1)} {m.group(2)}" if m else "none none")
PYEOF
)
check "S2 a real one-domain 256^3 run stays under its prediction (predicted / measured GB: ${ms2})" \
      "$(echo "${ms2}" | awk '$1 != "none" && $1 >= $2 {ok=1} END{print ok ? 1 : 0}')"
rm -f "${SNAP_W}" "${TMP}"/at_s1* "${TMP}"/at_s2.*[a-z]

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
