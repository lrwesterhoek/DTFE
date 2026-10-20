#!/usr/bin/env bash
# '--fixed-point-merge' (opt-in; src/ps_fixed_merge.h): every partition is added AS IT FINISHES into exact
# integer full grids (int64 words + int8 carries in units fixed before the loop), on small synthetic snapshots:
#   X1  the CPU deposit, two field sets (the moments with a scalar; and volume-weighted + exact deposit + caustics, which
#       fill mass_weight, disp_weight, disp_velocity, tet_touch and the OR-merged caustic mask): --max-concurrent 1, 2
#       and 4, and 4 with partition 0 held back 2 s (DTFE_TEST_DELAY_PARTITION=0:2: another completion order) -- every
#       output byte-identical among them; the run log says the merge mode, prints its 'fixed-point merge:' line (the
#       bytes, each accumulator's unit and headroom) and no 'ordered merge:' line
#   X2  against the default (ordered) merge of the same settings: a table of the largest difference per output grid,
#       relative to the grid's peak; every one below 2e-6 (expected ~1e-7: each partition value is rounded to its unit,
#       the float sums round at every add)
#   X3  '--scratch-dir' with a tiny threshold (DTFE_SCRATCH_MIN_GB): the integer grids are scratch-backed, the outputs
#       byte-identical to X1's
#   X4  overflow, FATAL and order-free: DTFE_TEST_FXMERGE_SHIFT=24 (units 2^24 finer: densities above 8 do not fit
#       2^63 units) stops the run at --max-concurrent 1 and at 4 with partition 0 held back, naming the accumulator,
#       with no output written; DTFE_TEST_FXMERGE_SUM_LIMIT=40 (the sum bound's limit lowered) stops it with the sum
#       message
#   X5  refusals: '--completion-order-merge' beside it, '--ps-window', the double build (when built)
#   X6  resume: a run killed after 10 merges (DTFE_TEST_EXIT_AFTER_PARTITIONS=10, 4 in flight, partition 0 held back)
#       holds the integer words in its checkpoint; '--resume' at --max-concurrent 2 writes X1's outputs byte for byte;
#       a default '--resume' of that checkpoint is refused by name, and so is a '--fixed-point-merge --resume' of a
#       default run's checkpoint, and a resume with other units (DTFE_TEST_FXMERGE_SHIFT=1)
#   X7  GPU builds only (SKIP otherwise): the --ps-gpu deposit at --max-concurrent 1, 2 and 4 with partition 0 held
#       back -- byte-identical (none falling back to the CPU); against the GPU default merge, the X2 table
# Usage: tests/ps_fixed_merge_check.sh [--no-build]   (--no-build is accepted and ignored: the binaries are used as built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair (which refuses the flag: X5 only)
precision_require "${PS_BIN}"
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
BIN="$(cd "$(dirname "${PS_BIN}")" && pwd)/$(basename "${PS_BIN}")"
TMP="${DTFE_TEST_TMP}/ps_fixed_merge_check"
rm -rf "$TMP"; mkdir -p "$TMP"
SNAP="$TMP/waves.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 40 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
# a scalar per particle (read with --scalar-dataset Potential), so the scalar moments are merged too
"$PY" - "$SNAP" <<'PYEOF' || { echo "FAIL scalar dataset"; exit 1; }
import h5py, numpy as np, sys
with h5py.File(sys.argv[1], "a") as f:
    x = f["PartType1/Coordinates"][:]
    f["PartType1/Potential"] = (1e3 * np.sin(x[:, 0] / 10.) + x[:, 1]).astype(np.float32)
PYEOF
PASS=0; FAILS=0
check() { if [ "$2" = 0 ]; then PASS=$((PASS+1)); echo "  PASS $1"; else FAILS=$((FAILS+1)); echo "  FAIL $1"; fi; }
# same <a> <b>: tests/lib.sh
# table <a> <b> <limit>: the largest |a - b| of every output grid relative to its peak; fails above the limit
table() {
    "$PY" - "$1" "$2" "$3" <<'PYEOF'
import glob, numpy as np, os, sys
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))
a, b, lim = sys.argv[1], sys.argv[2], float(sys.argv[3])
worst, n = 0., 0
print(f"    {'grid':<16} {'max|d|/peak':>12} {'cells equal':>12}")
for f in sorted(glob.glob(a + ".*")):
    e = f[len(a):]
    if e.endswith(".log") or ".runlog" in e or ".part." in e: continue
    x = np.fromfile(f, dtype=REAL).astype(np.float64); y = np.fromfile(b + e, dtype=REAL).astype(np.float64)
    if x.size != y.size or not x.size: print(f"    {e}: sizes differ"); sys.exit(1)
    peak = max(np.abs(x).max(), 1e-300)
    r = float(np.abs(x - y).max() / peak)
    worst = max(worst, r); n += 1
    print(f"    {e:<16} {r:12.2e} {np.mean(x == y):12.4f}")
print(f"    worst {worst:.2e} over {n} grids (limit {lim:g})")
sys.exit(0 if n and worst < lim else 1)
PYEOF
}
C=(--grid 48 --periodic --input 105 --MpcUnit 1 --verbose 2 --partition 3 3 3)
F1=(--field density velocity gradient dispersion scalar scalarGradient density_a velocity_a --scalar-dataset Potential)
F2=(--field density velocity dispersion --ps-volume-weighted --ps-exact-deposit --ps-caustics)
FX=--fixed-point-merge

if precision_double; then
    echo ">> X5 (double build): --fixed-point-merge is refused"
    "$BIN" "$SNAP" "$TMP/dbl" "${C[@]}" "${F1[@]}" $FX > "$TMP/dbl.log" 2>&1; rc=$?
    [ "$rc" != 0 ] && grep -q "not available in the double build" "$TMP/dbl.log" && [ ! -e "$TMP/dbl.den" ]
    check "X5 the double build refuses --fixed-point-merge (exit $rc)" $?
    echo "RESULT: $PASS passed, $FAILS failed"
    [ "$FAILS" = 0 ] || exit 1
    exit 0
fi

echo ">> X1: the CPU deposit at --max-concurrent 1, 2, 4 and 4 with partition 0 held back"
"$BIN" --help > "$TMP/help.txt" 2>&1   # (to a file: 'grep -q' on the pipe can SIGPIPE the writer under pipefail)
grep -q -- "--fixed-point-merge" "$TMP/help.txt"; check "X1 '--fixed-point-merge' is in the help" $?
for s in 1 2; do
    eval "F=(\"\${F$s[@]}\")"
    OMP_NUM_THREADS=1 "$BIN" "$SNAP" "$TMP/s${s}m1" "${C[@]}" "${F[@]}" $FX --max-concurrent 1 --ps-no-overlap > "$TMP/s${s}m1.log" 2>&1 \
        || { echo "FAIL: s${s}m1 (see $TMP/s${s}m1.log)"; exit 1; }
    OMP_NUM_THREADS=2 "$BIN" "$SNAP" "$TMP/s${s}m2" "${C[@]}" "${F[@]}" $FX --max-concurrent 2 > "$TMP/s${s}m2.log" 2>&1 \
        || { echo "FAIL: s${s}m2 (see $TMP/s${s}m2.log)"; exit 1; }
    OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/s${s}m4" "${C[@]}" "${F[@]}" $FX --max-concurrent 4 > "$TMP/s${s}m4.log" 2>&1 \
        || { echo "FAIL: s${s}m4 (see $TMP/s${s}m4.log)"; exit 1; }
    OMP_NUM_THREADS=4 DTFE_TEST_DELAY_PARTITION=0:2 "$BIN" "$SNAP" "$TMP/s${s}d4" "${C[@]}" "${F[@]}" $FX --max-concurrent 4 \
        > "$TMP/s${s}d4.log" 2>&1 || { echo "FAIL: s${s}d4 (see $TMP/s${s}d4.log)"; exit 1; }
    same "$TMP/s${s}m1" "$TMP/s${s}m2" && same "$TMP/s${s}m1" "$TMP/s${s}m4" && same "$TMP/s${s}m1" "$TMP/s${s}d4"
    check "X1 set $s: --max-concurrent 1, 2, 4 and 4 in another completion order -- every output byte-identical" $?
done
grep -q "partition merge order  : as they finish, into exact integer grids" "$TMP/s1d4.log" \
    && grep -q "fixed-point merge: .* GB of integer grids" "$TMP/s1d4.log" && grep -q "u.scalar: unit 2^-" "$TMP/s1d4.log" \
    && grep -q "u.tet_touch: unit 2^-" "$TMP/s2d4.log" && grep -q "u.disp_velocity: unit 2^-" "$TMP/s2d4.log" \
    && ! grep -q "ordered merge:" "$TMP/s1d4.log" && ! grep -q "partitions merge in index order" "$TMP/s1d4.log"
check "X1 the log names the mode and gives the 'fixed-point merge:' line (units, headroom), no ordered-merge line" $?
grep -a "fixed-point merge:" "$TMP/s1d4.log" | tr ';' '\n' | sed 's/^ */    /' | head -20

echo ">> X2: against the default (ordered) merge"
for s in 1 2; do
    eval "F=(\"\${F$s[@]}\")"
    OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/s${s}def" "${C[@]}" "${F[@]}" --max-concurrent 4 > "$TMP/s${s}def.log" 2>&1 \
        || { echo "FAIL: s${s}def (see $TMP/s${s}def.log)"; exit 1; }
    table "$TMP/s${s}def" "$TMP/s${s}m4" 2e-6
    check "X2 set $s: --fixed-point-merge differs from the ordered merge by < 2e-6 of each grid's peak" $?
done

echo ">> X3: --scratch-dir backs the integer grids"
SC="$TMP/scratch"; mkdir -p "$SC"
OMP_NUM_THREADS=4 DTFE_SCRATCH_MIN_GB=0.0001 "$BIN" "$SNAP" "$TMP/sc" "${C[@]}" "${F1[@]}" $FX --max-concurrent 4 \
    --scratch-dir "$SC" > "$TMP/sc.log" 2>&1 || { echo "FAIL: sc (see $TMP/sc.log)"; exit 1; }
grep -q "SCRATCH:" "$TMP/sc.log" && same "$TMP/s1m1" "$TMP/sc" && [ -z "$(ls -A "$SC")" ]
check "X3 scratch-backed integer grids (a tiny threshold): byte-identical, the scratch folder left empty" $?

echo ">> X4: overflow stops the run, whatever the order"
for r in "o1:1:" "o4:4:0:2"; do
    IFS=: read -r name mc dp ds <<< "$r"
    env OMP_NUM_THREADS=4 DTFE_TEST_FXMERGE_SHIFT=24 ${dp:+DTFE_TEST_DELAY_PARTITION=$dp:$ds} "$BIN" "$SNAP" "$TMP/$name" "${C[@]}" \
        "${F1[@]}" $FX --max-concurrent "$mc" > "$TMP/$name.log" 2>&1; rc=$?
    [ "$rc" != 0 ] && grep -q -- "--fixed-point-merge: partition [0-9]* holds a value of [ua]\.[a-z_]*" "$TMP/$name.log" \
        && grep -q "2^63 units" "$TMP/$name.log" && [ ! -e "$TMP/$name.den" ]
    check "X4 DTFE_TEST_FXMERGE_SHIFT=24 at --max-concurrent $mc${dp:+ (partition 0 held back)}: stopped (exit $rc), naming the accumulator, no output" $?
done
OMP_NUM_THREADS=4 DTFE_TEST_FXMERGE_SUM_LIMIT=40 "$BIN" "$SNAP" "$TMP/os" "${C[@]}" "${F1[@]}" $FX --max-concurrent 4 \
    > "$TMP/os.log" 2>&1; rc=$?
[ "$rc" != 0 ] && grep -q "the merged partitions' largest values of .* sum to 2^" "$TMP/os.log" && [ ! -e "$TMP/os.den" ]
check "X4 DTFE_TEST_FXMERGE_SUM_LIMIT=40: the sum bound stops the run (exit $rc)" $?

echo ">> X5: refusals"
"$BIN" "$SNAP" "$TMP/bad1" "${C[@]}" "${F1[@]}" $FX --completion-order-merge > "$TMP/bad1.log" 2>&1; rc1=$?
"$BIN" "$SNAP" "$TMP/bad2" "${C[@]}" "${F1[@]}" $FX --ps-window 0 50 0 50 0 50 > "$TMP/bad2.log" 2>&1; rc2=$?
[ "$rc1" != 0 ] && grep -q "contradict each other" "$TMP/bad1.log" && [ "$rc2" != 0 ] \
    && grep -q "cannot be combined with '--ps-window'" "$TMP/bad2.log" && [ ! -e "$TMP/bad1.den" ] && [ ! -e "$TMP/bad2.den" ]
check "X5 '--completion-order-merge' (exit $rc1) and '--ps-window' (exit $rc2) beside it are refused" $?
if [ -x ./PS-DTFE-double ]; then
    ./PS-DTFE-double "$SNAP" "$TMP/dbl" "${C[@]}" --field density $FX > "$TMP/dbl.log" 2>&1; rc=$?
    [ "$rc" != 0 ] && grep -q "not available in the double build" "$TMP/dbl.log" && [ ! -e "$TMP/dbl.den" ]
    check "X5 the double build refuses it (exit $rc)" $?
else
    echo "  SKIP X5 double: ./PS-DTFE-double is not built"
fi

echo ">> X6: resume"
OMP_NUM_THREADS=4 DTFE_TEST_DELAY_PARTITION=0:2 DTFE_TEST_EXIT_AFTER_PARTITIONS=10 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" "${F1[@]}" \
    $FX --max-concurrent 4 > "$TMP/k.log" 2>&1; rc1=$?
CK="$TMP/k.runlog.checkpoint/checkpoint"
nm="$(grep -a -m1 '^merged_count' "$CK" 2>/dev/null | awk '{print $2}')"
[ "$rc1" = 137 ] && [ "${nm:-0}" = 10 ] && grep -aq "^field x.fxw.u.density 8 " "$CK" && grep -aq "^field x.fxc.u.density 1 " "$CK" \
    && grep -aq "^field x.fxmerge.state " "$CK" && grep -aq "^merge=fixed-point" "$CK" && ! grep -aq "^field u.density " "$CK"
check "X6 killed after 10 merges (exit $rc1): the checkpoint holds ${nm:-?} partitions as integer words and carries" $?
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" "${F1[@]}" --max-concurrent 4 --resume > "$TMP/k_other.log" 2>&1; rc2=$?
OMP_NUM_THREADS=4 DTFE_TEST_FXMERGE_SHIFT=1 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" "${F1[@]}" $FX --max-concurrent 4 --resume \
    > "$TMP/k_shift.log" 2>&1; rc3=$?
[ "$rc2" != 0 ] && grep -q "was written by a run with '--fixed-point-merge'" "$TMP/k_other.log" \
    && [ "$rc3" != 0 ] && grep -q "the checkpoint's fixed-point unit of u.density" "$TMP/k_shift.log" && [ -s "$CK" ]
check "X6 a default '--resume' of it (exit $rc2) and one with other units (exit $rc3) are refused; the checkpoint is kept" $?
OMP_NUM_THREADS=2 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" "${F1[@]}" $FX --max-concurrent 2 --resume > "$TMP/k2.log" 2>&1; rc4=$?
[ "$rc4" = 0 ] && same "$TMP/s1m1" "$TMP/k" && [ ! -e "$TMP/k.runlog.checkpoint" ]
check "X6 '--fixed-point-merge --resume' at --max-concurrent 2 (exit $rc4): byte-identical to the uninterrupted runs" $?
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=5 "$BIN" "$SNAP" "$TMP/kd" "${C[@]}" "${F1[@]}" --max-concurrent 4 \
    > "$TMP/kd.log" 2>&1; rc5=$?
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/kd" "${C[@]}" "${F1[@]}" $FX --max-concurrent 4 --resume > "$TMP/kd2.log" 2>&1; rc6=$?
[ "$rc5" = 137 ] && [ "$rc6" != 0 ] && grep -q "was written by a run with the default merge" "$TMP/kd2.log"
check "X6 a default run's checkpoint resumed with '--fixed-point-merge' is refused (exit $rc6)" $?

echo ">> X7: the GPU deposit"
OBJ=o_ps
if [ ! -d "$OBJ" ] || [ -f "$OBJ/.gpu_mode_off" ] || [ ! -f "$OBJ/.gpu_mode_metal" ]; then
    echo "  SKIP X7: $OBJ is not a Metal GPU build"
else
    G=("${C[@]}" --field density velocity gradient dispersion density_a --ps-volume-weighted --ps-gpu)
    OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/g1" "${G[@]}" $FX --max-concurrent 1 > "$TMP/g1.log" 2>&1 || { echo "FAIL: g1 (see $TMP/g1.log)"; exit 1; }
    OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/g2" "${G[@]}" $FX --max-concurrent 2 > "$TMP/g2.log" 2>&1 || { echo "FAIL: g2 (see $TMP/g2.log)"; exit 1; }
    OMP_NUM_THREADS=4 DTFE_TEST_DELAY_PARTITION=0:2 "$BIN" "$SNAP" "$TMP/g4" "${G[@]}" $FX --max-concurrent 4 > "$TMP/g4.log" 2>&1 \
        || { echo "FAIL: g4 (see $TMP/g4.log)"; exit 1; }
    ! grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead\|gave up" "$TMP/g1.log" "$TMP/g2.log" "$TMP/g4.log"
    check "X7 no GPU pass fell back to the CPU" $?
    same "$TMP/g1" "$TMP/g2" && same "$TMP/g1" "$TMP/g4"
    check "X7 --ps-gpu at --max-concurrent 1, 2 and 4 (partition 0 held back): every output byte-identical" $?
    OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/gdef" "${G[@]}" --max-concurrent 4 > "$TMP/gdef.log" 2>&1 || { echo "FAIL: gdef (see $TMP/gdef.log)"; exit 1; }
    table "$TMP/gdef" "$TMP/g4" 2e-6
    check "X7 against the GPU run's default merge: < 2e-6 of each grid's peak" $?
fi

echo "RESULT: $PASS passed, $FAILS failed"
[ "$FAILS" = 0 ] || exit 1
exit 0
