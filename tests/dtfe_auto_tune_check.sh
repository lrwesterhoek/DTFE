#!/usr/bin/env bash
# The STANDARD-DTFE auto-tuner (src/auto_tune.h, 2026-10-13): what PS-DTFE's tuner had and the standard one lacked,
# and the memory model of the production '--exact-average' runs.
#
# The 2026-10-13 audit measured the exact-average standard runs 25-30% above the sampled ones in peak footprint and
# found no term for it in the tuner. The code allocates nothing extra for the exact average; what made the model
# wrong is the padded vertex count: the tuner took every thread sub-box to hold the box-average particle count, while
# the load-balanced split gives a dense partition a small volume and a fixed-length padding shell (TNG50-3-Dark z=0:
# 35.4M padded particles in one partition's 4 threads, 7.8M modeled; the exact run, whose threads all wait on the
# GPU with their triangulations alive, peaked at 44.5 GB against 39.9 predicted). The tuner now replays the run's
# own split on a count grid of the particles. Verified here, on synthetic crossed waves (clustered knots) and the
# repository's binaries:
#
#  A) '--auto-tune-report' on the standard binary: ONE valid 'AUTO-TUNE-REPORT partition=.. mc=..' line (it printed
#     'AUTO-TUNE-REPORT unsupported'), exit 0, nothing written; a hand-set split and thread count are reported as
#     given (mc = the threads of a partition; the partitions run one after another).
#  B) the counted padded vertices (DTFE_AUTO_TUNE_DEBUG=1) against the run's own 'Thread i had N particles' lines:
#     the same busiest partition, its sum and every thread's count within 10%, at 3^3 x 8 and 4^3 x 4 threads.
#  C) the reason for B: on this clustered set the busiest partition holds well over the box-average count the old
#     model used (N / P^3 x the padding factor) -- at least 1.5x.
#  D) a user's --max-concurrent is the thread count the tuner plans with when it chooses the split (the standard
#     tuner started at every core whatever the user gave).
#  E) an auto-tuned run ends its SUMMARY with the 'memory model' line (predicted vs measured, as PS runs do), and the
#     measured peak stays within 1.35x of the prediction (the allocator keeps freed triangulation memory between
#     partitions: up to ~30% above the plan on sub-GB runs, while at TNG scale the prediction is 19-90% HIGH, see
#     auto_tune.h): CPU sampled average, and the GPU exact average when the build has a GPU.
#  F) the tuner only chooses: an auto-tuned run is byte-identical to the same split given by hand.
#  G) on a simulated small machine the tuner picks a split its counted model says fits (over_budget=0) and names the
#     busiest partition's vertex count in its AUTO-TUNE line.
#  H) the 2D build (DTFE-2d, when built): a valid report from the counted model, counts within 15% of the run's.
#  I) (2026-10-15) the terms the plan lacked, each a field of the report line: I1 a '--sample-points' plane's records
#     (pts_gb: 326 B per point with --pts-vel-grad, one record per point, in fixed_gb too; DTFE_AUTO_PTS_N sizes a plane
#     that is not on disk); on GPU builds I2 the GPU deposit's host snapshot of its grids (snapshot_gb: 8-B words, half
#     under DTFE_METAL_FLOAT_GRIDS=1, none on the CPU), I3 the partition loop's look-ahead (lookahead_gb, in
#     predicted_gb; none with --no-overlap or on one domain) and I4 a budget that holds the plan but not its
#     look-ahead: the look-ahead is switched off for the run (said, lookahead_gb=0, over_budget=0).
#
# Usage: tests/dtfe_auto_tune_check.sh [--no-build]     (DTFE_TEST_TMP, DTFE_TEST_PRECISION=double as the others)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

BIN="${DTFE_BIN}"
precision_require "${BIN}"
# The thread count is pinned to a 10-core machine's (as ps_auto_tune_check.sh H1 does): the tuner plans with every
# core, so C, D, F and G would plan other splits on a 4-core runner; the real runs (B, E, F) then use the same threads
# the plan assumed, oversubscribed on a smaller machine, which changes nothing they check.
export OMP_NUM_THREADS=10
TMP="${DTFE_TEST_TMP}/dtfe_auto_tune"; mkdir -p "${TMP}"
N="${N:-64}"; BOX="${BOX:-100.0}"
SNAP="${TMP}/dat_crossed${N}.hdf5"

echo "============================================================"
echo " standard-DTFE auto-tune check   N=${N}^3 crossed waves"
echo "============================================================"

if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building DTFE ..."
    make DTFE $(cat o/.build_mode 2>/dev/null || true) >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }
GPU=0; "${BIN}" --help 2>&1 | grep -q -- "--gpu" && grep -Eq "METAL=1|CUDA=1|HIP=1" "o$(precision_double && echo _d)/.build_mode" 2>/dev/null && GPU=1

echo ">> generating the clustered test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box "${BOX}" --crossed-waves \
    --amplitude-factor 3.0 >/dev/null

FAILS=0
check() {  # $1 = name, $2 = 0/1 ok
    if [ "$2" -eq 1 ]; then echo "   OK   $1"; else echo "   FAIL $1"; FAILS=$((FAILS+1)); fi
}
agrep() {  # set-e-safe grep assertion (ERE): $1 = check name, $2 = pattern, $3 = file
    if grep -Eq -- "$2" "$3"; then check "$1" 1; else check "$1" 0; fi
}
COMMON=(--periodic --input 105 --MpcUnit 1 --padding 10)
REPORT_RE='^AUTO-TUNE-REPORT partition=[0-9]+ mc=[0-9]+ predicted_gb=[0-9.]+ budget_gb=[0-9.]+ fixed_gb=[0-9.]+ .*over_budget=[01]$'

# ---------- (A) the report ----------
echo ">> A: --auto-tune-report on the standard binary"
rm -f "${TMP}"/da_a.*
rc=0; env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/da_a" --grid 64 "${COMMON[@]}" --field density_a velocity_a \
    --auto-tune-report > "${TMP}/da_a_log.txt" 2>&1 || rc=$?
check "A exits 0 (${rc})" "$([ "${rc}" = "0" ] && echo 1 || echo 0)"
check "A prints exactly one valid report line (not 'unsupported')" \
      "$([ "$(grep -cE "${REPORT_RE}" "${TMP}/da_a_log.txt")" = "1" ] && ! grep -q 'unsupported' "${TMP}/da_a_log.txt" && echo 1 || echo 0)"
check "A nothing is written next to the output root" "$([ -z "$(ls "${TMP}" | grep '^da_a\.')" ] && echo 1 || echo 0)"
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/da_a" --grid 64 "${COMMON[@]}" --field density_a \
    --partition 2 2 2 --max-concurrent 3 --auto-tune-report > "${TMP}/da_a2_log.txt" 2>&1 || true
agrep "A a hand-set split and thread count are reported as given" "^AUTO-TUNE-REPORT partition=2 mc=3 " "${TMP}/da_a2_log.txt"

# ---------- (B) + (C) the counted padded vertices against the run's own ----------
echo ">> B/C: the counted padded vertices against the run's 'Thread i had' lines"
compare_counts() {  # $1 = tag, $2 = split, $3 = threads; prints the checks
    local tag="$1" split="$2" mc="$3"
    env DTFE_AUTO_TUNE_DEBUG=1 DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/da_${tag}" --grid 48 "${COMMON[@]}" \
        --field density_a --partition ${split} --max-concurrent "${mc}" --auto-tune-report > "${TMP}/da_${tag}_rep.txt" 2>&1 || true
    "${BIN}" "${SNAP}" "${TMP}/da_${tag}" --grid 48 "${COMMON[@]}" --field density_a --partition ${split} \
        --max-concurrent "${mc}" --verbose 2 > "${TMP}/da_${tag}_run.txt" 2>&1 || true
    "${PY}" - "${TMP}/da_${tag}_rep.txt" "${TMP}/da_${tag}_run.txt" "${N}" "${split%% *}" "${mc}" "${BOX}" <<'PYEOF'
import re, sys
rep, run = open(sys.argv[1], errors="replace").read(), open(sys.argv[2], errors="replace").read()
n, P, T = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
m = re.search(r"AUTO-TUNE-DEBUG standard: counted from the particles .*busiest partition (\d+) of (\d+): (\d+) padded "
              r"vertices.*per thread:([ 0-9]+)", rep)
parts = run.split("<<< Interpolating the fields for partition")[1:]
real = [[int(x) for x in re.findall(r"Thread \d+ had (\d+) particles", p)] for p in parts]
ok_b = ok_c = False
if m and real and all(real):
    pm, nparts, vm = int(m.group(1)), int(m.group(2)), float(m.group(3))
    tm = [float(x) for x in m.group(4).split()]
    ir = max(range(len(real)), key=lambda k: sum(real[k]))
    vr, tr = sum(real[ir]), real[ir]
    worst = max(abs(a - b) / b for a, b in zip(tm, tr)) if len(tm) == len(tr) else 9.
    print(f"   .... busiest partition {pm} of {nparts} (the run's: {ir + 1} of {len(real)}): {vm:.0f} counted vs {vr} "
          f"padded vertices; per thread worst {100 * worst:.1f}%")
    ok_b = nparts == len(real) and abs(vm - vr) / vr < 0.10 and worst < 0.10 and sum(real[pm - 1]) > 0.9 * vr
    # the old model's count: every sub-box at the box average, (N / P^3) x prod(1 + 2 pad P T^(1/3)), pad in box units
    pad = 10. / n
    uniform = n**3 / P**3 * (1. + 2. * pad * P * T ** (1. / 3.)) ** 3
    print(f"   .... the uniform count of the old model: {uniform:.0f} ({vr / uniform:.2f}x less than the run's)")
    ok_c = vr > 1.5 * uniform
print(f"OKB {int(ok_b)} OKC {int(ok_c)}")
PYEOF
}
for cfg in "b3|3 3 3|8" "b4|4 4 4|4"; do
    tag="${cfg%%|*}"; rest="${cfg#*|}"; split="${rest%%|*}"; mc="${rest##*|}"
    out="$(compare_counts "${tag}" "${split}" "${mc}")"
    printf '%s\n' "${out}" | grep '^   \.\.\.\.' || true
    check "B ${split// /x} x ${mc}: the busiest partition and its padded vertices (sum and per thread) within 10% of the run's" \
          "$(printf '%s\n' "${out}" | grep -q '^OKB 1' && echo 1 || echo 0)"
    check "C ... and the run's busiest partition holds >= 1.5x the box-average count the old model used" \
          "$(printf '%s\n' "${out}" | grep -q 'OKC 1' && echo 1 || echo 0)"
done
rm -f "${TMP}"/da_b*.a_den "${TMP}"/da_b*badGridPoints*

# ---------- (D) a user's --max-concurrent ----------
echo ">> D: a user's --max-concurrent is planned with"
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/da_d" --grid 48 "${COMMON[@]}" --field density_a --max-concurrent 2 \
    --auto-tune-report > "${TMP}/da_d.txt" 2>&1 || true
agrep "D '--max-concurrent 2' with the split left to the tuner: reported mc=2" "^AUTO-TUNE-REPORT partition=[0-9]+ mc=2 " "${TMP}/da_d.txt"

# ---------- (E) the SUMMARY's memory model line, predicted vs measured ----------
echo ">> E: predicted vs measured peak on real runs (128^3, the production fields)"
FIELDS=(density_a velocity_a gradient_a divergence_a shear_a vorticity_a tweb_a vweb_a)
run_e() {  # $1 = tag, rest = extra args
    local tag="$1"; shift
    env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 "${BIN}" "${SNAP}" "${TMP}/da_${tag}" --grid 128 "${COMMON[@]}" --field "${FIELDS[@]}" \
        --lambda_th 0.3 --partition 2 2 2 "$@" > "${TMP}/da_${tag}.txt" 2>&1 || true
    "${PY}" - "${TMP}/da_${tag}.txt" <<'PYEOF'
import re, sys
m = re.search(r"memory model\s*: ([0-9.]+) GB predicted \(auto-tune\) vs ([0-9.]+) GB measured", open(sys.argv[1], errors="replace").read())
if not m:
    print("NOLINE"); sys.exit(0)
p, q = float(m.group(1)), float(m.group(2))
print(f"   .... predicted {p:.2f} GB, measured {q:.2f} GB ({q / p:.2f}x)")
print("OKE 1" if q <= 1.35 * p else "OKE 0")
PYEOF
    rm -f "${TMP}/da_${tag}".a_* "${TMP}/da_${tag}"*badGridPoints*
}
out="$(run_e e_cpu)"; printf '%s\n' "${out}" | grep '^   \.\.\.\.' || true
check "E the SUMMARY of an auto-tuned standard run carries the 'memory model' line" "$(printf '%s\n' "${out}" | grep -q NOLINE && echo 0 || echo 1)"
check "E CPU, sampled average: measured <= 1.35x predicted" "$(printf '%s\n' "${out}" | grep -q 'OKE 1' && echo 1 || echo 0)"
if [ "${GPU}" = 1 ]; then
    out="$(run_e e_gpu --gpu --exact-average)"; printf '%s\n' "${out}" | grep '^   \.\.\.\.' || true
    check "E GPU, exact average: measured <= 1.35x predicted" "$(printf '%s\n' "${out}" | grep -q 'OKE 1' && echo 1 || echo 0)"
else
    echo "   SKIP  E GPU (this build has no GPU backend)"
fi

# ---------- (F) the tuner only chooses ----------
echo ">> F: an auto-tuned run equals the same split given by hand"
env DTFE_RAM_GB=1.2 DTFE_FREE_GB=1000 "${BIN}" "${SNAP}" "${TMP}/da_f_auto" --grid 48 "${COMMON[@]}" --field density_a velocity_a \
    > "${TMP}/da_f_auto.txt" 2>&1 || true
fsplit="$(sed -n 's/.*AUTO-TUNE: .* -> --partition \([0-9 ]*\), \([0-9]*\) thread(s).*/\1|\2/p' "${TMP}/da_f_auto.txt" | head -1)"
if [ -n "${fsplit}" ]; then
    "${BIN}" "${SNAP}" "${TMP}/da_f_hand" --grid 48 "${COMMON[@]}" --field density_a velocity_a --partition ${fsplit%%|*} \
        --max-concurrent "${fsplit##*|}" > "${TMP}/da_f_hand.txt" 2>&1 || true
    check "F the tuner's split (${fsplit%%|*}, ${fsplit##*|} threads) given by hand: byte-identical .a_den and .a_vel" \
          "$([ -s "${TMP}/da_f_auto.a_den" ] && cmp -s "${TMP}/da_f_auto.a_den" "${TMP}/da_f_hand.a_den" \
             && cmp -s "${TMP}/da_f_auto.a_vel" "${TMP}/da_f_hand.a_vel" && echo 1 || echo 0)"
else
    check "F the tuner chose a split on a simulated 1.2 GB machine (no AUTO-TUNE line)" 0
fi
rm -f "${TMP}"/da_f_*.a_* "${TMP}"/da_f_*badGridPoints*

# ---------- (G) a tight machine ----------
echo ">> G: a split that fits a simulated small machine"
env DTFE_RAM_GB=1.2 "${BIN}" "${SNAP}" "${TMP}/da_g" --grid 96 "${COMMON[@]}" --field density_a velocity_a gradient_a \
    --auto-tune-report > "${TMP}/da_g.txt" 2>&1 || true
agrep "G the plan fits its budget (over_budget=0) with a split" "^AUTO-TUNE-REPORT partition=[2-6] mc=[0-9]+ .*over_budget=0$" "${TMP}/da_g.txt"
agrep "G ... and the AUTO-TUNE line names the busiest partition's padded vertices" \
      "AUTO-TUNE: .*the busiest partition triangulates ~[0-9.]+ M padded vertices at once" "${TMP}/da_g.txt"

# ---------- (H) 2D ----------
echo ">> H: the 2D build"
BIN2D="./DTFE-2d"; precision_double && BIN2D="./DTFE-2d-double"
if [ -x "${BIN2D}" ]; then
    SNAP2D="${TMP}/dat_crossed2d.hdf5"
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP2D}" --n 512 --box "${BOX}" --dim 2 --crossed-waves \
        --amplitude-factor 3.0 >/dev/null
    env DTFE_AUTO_TUNE_DEBUG=1 DTFE_RAM_GB=64 "${BIN2D}" "${SNAP2D}" "${TMP}/da_h" --grid 128 "${COMMON[@]}" --field density_a \
        --partition 3 3 --max-concurrent 4 --auto-tune-report > "${TMP}/da_h_rep.txt" 2>&1 || true
    "${BIN2D}" "${SNAP2D}" "${TMP}/da_h" --grid 128 "${COMMON[@]}" --field density_a --partition 3 3 --max-concurrent 4 \
        --verbose 2 > "${TMP}/da_h_run.txt" 2>&1 || true
    agrep "H a valid 2D report" "${REPORT_RE}" "${TMP}/da_h_rep.txt"
    check "H the counted busiest partition within 15% of the 2D run's" "$("${PY}" - "${TMP}/da_h_rep.txt" "${TMP}/da_h_run.txt" <<'PYEOF'
import re, sys
rep, run = open(sys.argv[1], errors="replace").read(), open(sys.argv[2], errors="replace").read()
m = re.search(r"busiest partition \d+ of \d+: (\d+) padded vertices", rep)
real = [sum(int(x) for x in re.findall(r"Thread \d+ had (\d+) particles", p))
        for p in run.split("<<< Interpolating the fields for partition")[1:]]
print(1 if m and real and abs(float(m.group(1)) - max(real)) < 0.15 * max(real) else 0)
PYEOF
)"
    rm -f "${TMP}"/da_h.a_* "${TMP}"/da_h*badGridPoints*
else
    echo "   SKIP  H (${BIN2D} is not built)"
fi

# ---------- (I) the plane records, the GPU snapshot and the look-ahead in the report ----------
echo ">> I: the plane records, the GPU snapshot and the look-ahead in the report line"
field() { sed -n "s/^AUTO-TUNE-REPORT .* $1=\([0-9.]*\) .*/\1/p; s/^AUTO-TUNE-REPORT .* $1=\([0-9.]*\)$/\1/p" "$2" | head -1; }
"${PY}" - "${TMP}/da_i_plane.bin" "${BOX}" <<'PYEOF'
import sys, numpy as np
c = (np.arange(16) + 0.5) * float(sys.argv[2]) / 16
X, Y = np.meshgrid(c, c, indexing="ij")
np.stack([X.ravel(), Y.ravel(), np.full(X.size, 0.5 * float(sys.argv[2]))], 1).astype(np.float64).tofile(sys.argv[1])
PYEOF
IR=("${SNAP}" "${TMP}/da_i" --grid 192 "${COMMON[@]}" --field density_a velocity_a gradient_a --partition 2 2 2
    --auto-tune-report)
env DTFE_RAM_GB=64 "${BIN}" "${IR[@]}" --max-concurrent 1 > "${TMP}/da_i0.txt" 2>&1 || true
env DTFE_RAM_GB=64 DTFE_AUTO_PTS_N=4194304 DTFE_AUTO_PTS_VELGRAD=1 "${BIN}" "${IR[@]}" --max-concurrent 1 \
    --sample-points "${TMP}/da_i_plane.bin" > "${TMP}/da_i1.txt" 2>&1 || true
p1="$(field pts_gb "${TMP}/da_i1.txt")"; f0="$(field fixed_gb "${TMP}/da_i0.txt")"; f1="$(field fixed_gb "${TMP}/da_i1.txt")"
ok="$(awk -v p="${p1:-0}" -v a="${f0:-0}" -v b="${f1:-0}" 'BEGIN{print (p > 1.362 && p < 1.372 && b - a > p - 0.02 && b - a < p + 0.02) ? 1 : 0}')"
grep -q 'streams_budget=1.000 ' "${TMP}/da_i1.txt" || ok=0
check "I1 a 2048^2 plane with --pts-vel-grad: pts_gb ${p1:-?} (4194304 x 326 B = 1.367), in fixed_gb (${f0:-?} -> ${f1:-?}), one record per point" "${ok}"
s0="$(field snapshot_gb "${TMP}/da_i0.txt")"; l0="$(field lookahead_gb "${TMP}/da_i0.txt")"
check "I2/I3 a CPU run counts no GPU snapshot and no look-ahead (${s0:-?}, ${l0:-?})" \
      "$([ "${s0}" = "0.000" ] && [ "${l0}" = "0.000" ] && echo 1 || echo 0)"
if [ "${GPU}" = 1 ]; then
    env DTFE_RAM_GB=64 "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu > "${TMP}/da_i2.txt" 2>&1 || true
    env DTFE_RAM_GB=64 DTFE_METAL_FLOAT_GRIDS=1 DTFE_GPU_FLOAT_GRIDS=1 "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu \
        > "${TMP}/da_i3.txt" 2>&1 || true
    env DTFE_RAM_GB=64 "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu --no-overlap > "${TMP}/da_i4.txt" 2>&1 || true
    s2="$(field snapshot_gb "${TMP}/da_i2.txt")"; s3="$(field snapshot_gb "${TMP}/da_i3.txt")"
    check "I2 --gpu: the snapshot of the largest thread sub-grid (${s2:-?} GB), half under the float grids (${s3:-?} GB)" \
          "$(awk -v a="${s2:-0}" -v b="${s3:-0}" 'BEGIN{print (a > 0.02 && b > 0. && a / b > 1.95 && a / b < 2.05) ? 1 : 0}')"
    l2="$(field lookahead_gb "${TMP}/da_i2.txt")"; l4="$(field lookahead_gb "${TMP}/da_i4.txt")"
    q2="$(field predicted_gb "${TMP}/da_i2.txt")"; q4="$(field predicted_gb "${TMP}/da_i4.txt")"
    check "I3 --gpu over 8 partitions: the look-ahead (${l2:-?} GB) is in predicted_gb (${q4:-?} -> ${q2:-?}); none with --no-overlap (${l4:-?})" \
          "$(awk -v l="${l2:-0}" -v m="${l4:-1}" -v a="${q4:-0}" -v b="${q2:-0}" 'BEGIN{d = b - a - l; print (l > 0.01 && m == 0 && d > -0.011 && d < 0.011) ? 1 : 0}')"
    # a budget between the plan without the look-ahead and the plan with it (the budget is 60% of DTFE_RAM_GB)
    ram="$(awk -v a="${q4:-0}" -v l="${l2:-0}" 'BEGIN{printf "%.4f", (a + l / 2.) / 0.6}')"
    env DTFE_RAM_GB="${ram}" "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu > "${TMP}/da_i5.txt" 2>&1 || true
    check "I4 a budget that holds the plan (${q4:-?} GB) but not its look-ahead: switched off, said, over_budget=0" \
          "$(grep -q 'look-ahead off for this run' "${TMP}/da_i5.txt" && grep -Eq 'lookahead_gb=0.000 over_budget=0$' "${TMP}/da_i5.txt" \
             && echo 1 || echo 0)"
else
    echo "   SKIP  I2-I4 (this build has no GPU backend)"
fi
rm -f "${TMP}"/da_i.* "${TMP}"/da_i*badGridPoints*

echo "------------------------------------------------------------"
if [ "${FAILS}" -gt 0 ]; then
    echo "RESULT: FAIL (${FAILS} check(s))"
    exit 1
fi
echo "RESULT: PASS  (the standard auto-tuner reports its plan, counts the padded vertices of the run's own split,"
echo "               plans with the user's thread count, and its prediction holds the measured peak)"
echo "============================================================"
