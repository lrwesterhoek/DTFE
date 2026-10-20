#!/usr/bin/env bash
# The partition loop's run-time features, on small synthetic snapshots:
#   R1  '--ordered-merge': two runs at --max-concurrent 4 merge in index order and write byte-identical grids
#   R2  the partition look-ahead (forced on the CPU deposit with DTFE_PS_OVERLAP=1): every output byte-identical
#       to '--ps-no-overlap', and the log says it ran
#   R3  GPU builds only: the look-ahead of a '--ps-gpu' run at --max-concurrent 1 (on by default) against
#       '--ps-no-overlap': every grid byte-identical (.den, .a_den, .streams, ... and the velocity, an exact
#       fixed-point sum like the rest); SKIP on a CPU build
#   R4  crash-safe outputs: a finished run leaves no '<file>.part.<pid>' and no progress sidecar; a stale
#       '.part.<pid>' of a dead pid is removed at the start, one of a live pid is kept, and one whose final file
#       is missing is kept with its mv (it may be the only copy); a run whose output
#       folder is missing writes the whole set to $TMPDIR/dtfe-rescue/<folder>_<name>, byte-identical to a
#       normal run, says so and exits 3
#   R5  the progress sidecar '<output>.runlog.progress' while a run goes: the contract's keys, phase
#       'partitions', done counting up to total; also predicted_gb (-1 here: the split and the
#       concurrency are given, so the tuner predicts nothing) and peak_gb (> 0)
#   R6  the SUMMARY's 'memory model' line: a run whose concurrency the auto-tuner picks prints its
#       predicted peak against the measured one (macOS: the peak footprint), both positive; R5's run, with
#       nothing left to the tuner, prints none
#   R7  the STAGED ordered merge, the default. Partition 0 is held 2 s before its merge
#       (DTFE_TEST_DELAY_PARTITION=0:2), so the others finish first: with NO merge flag the run says it merges in
#       index order, its 'ordered merge:' line counts staged partitions (> 0, none waiting), and every output is
#       byte-identical to R1's '--ordered-merge' run; DTFE_MERGE_STAGE_GB=0 (no staging: the plain ordered merge) and
#       a machine with no free memory (DTFE_FREE_GB=0.001: no room to stage) wait instead and are byte-identical too;
#       '--completion-order-merge' says so, stages nothing and equals R1 to float rounding; it and '--ordered-merge'
#       together are refused; a default run killed after 10 merges (DTFE_TEST_EXIT_AFTER_PARTITIONS=10) while
#       partitions are staged has checkpointed exactly partitions 0-9 (the merge order is the index order) and
#       resumes to R1's grids byte for byte; with the concurrency left to the auto-tuner (the memory gate guards
#       every partition start) a partition is always staged, and the grids equal DTFE_MERGE_STAGE_GB=0's byte for byte.
#       THE SPILL: with no free memory a parked partition is spilled to disk (--scratch-dir, which is
#       empty afterwards: the files are unlinked) instead of waiting -- the 'ordered merge:' line counts the spills,
#       none waited -- and DTFE_MERGE_SPILL=1 spills every parked one (the system temp folder, with DTFE_SPILL_MIN_GB=2:
#       a CI runner has less than its 16 GB margin free; and under the memory gate): every output byte-identical to a
#       '--max-concurrent 1' run and to R1; a spilling run killed after 10 merges checkpointed partitions 0-9 and
#       resumes byte-identical; DTFE_MERGE_SPILL=0 brings back the wait
# Usage: tests/ps_runtime_check.sh [--no-build]   (--no-build is accepted and ignored: the binaries are used as built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}"
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
BIN="$(cd "$(dirname "${PS_BIN}")" && pwd)/$(basename "${PS_BIN}")"
TMP="${DTFE_TEST_TMP}/ps_runtime_check"
rm -rf "$TMP"; mkdir -p "$TMP"
SNAP="$TMP/waves.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 40 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
PASS=0; FAILS=0
check() { if [ "$2" = 0 ]; then PASS=$((PASS+1)); echo "  PASS $1"; else FAILS=$((FAILS+1)); echo "  FAIL $1"; fi; }
# same <a> <b>: tests/lib.sh
C=(--grid 48 --periodic --input 105 --MpcUnit 1 --verbose 2 --field density velocity density_a)

echo ">> R1: --ordered-merge, --partition 3 3 3 --max-concurrent 4, twice"
"$BIN" --help 2>&1 | grep -q -- "--ordered-merge"; check "R1 '--ordered-merge' is in the help" $?
for r in o1 o2; do OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/$r" "${C[@]}" --partition 3 3 3 --max-concurrent 4 --ordered-merge \
    > "$TMP/$r.log" 2>&1 || { echo "FAIL: $r (see $TMP/$r.log)"; exit 1; }; done
grep -q "partition merge order  : partition index" "$TMP/o1.log"; check "R1 the run says it merges in index order" $?
same "$TMP/o1" "$TMP/o2"; check "R1 two --ordered-merge runs at --max-concurrent 4: byte-identical grids" $?

echo ">> R7: the staged ordered merge (the default since 2026-10-14), partition 0 held back 2 s"
"$BIN" --help 2>&1 | grep -q -- "--completion-order-merge"; check "R7 '--completion-order-merge' is in the help" $?
R7=(env OMP_NUM_THREADS=4 DTFE_TEST_DELAY_PARTITION=0:2 "$BIN" "$SNAP")
P3=(--partition 3 3 3 --max-concurrent 4)
"${R7[@]}" "$TMP/st" "${C[@]}" "${P3[@]}" > "$TMP/st.log" 2>&1 || { echo "FAIL: st (see $TMP/st.log)"; exit 1; }
nst="$(sed -n 's/.*ordered merge: \([0-9]*\) partition(s) finished before their turn.*/\1/p' "$TMP/st.log")"
nwt="$(sed -n 's/.* \([0-9]*\) waited for their turn.*/\1/p' "$TMP/st.log")"
grep -q "partition merge order  : partition index" "$TMP/st.log" && grep -q "PS-DTFE: partitions merge in index order" "$TMP/st.log" \
    && [ "${nst:-0}" -gt 0 ] && [ "${nwt:-1}" = 0 ]
check "R7 no merge flag: index order, ${nst:-?} partition(s) staged, ${nwt:-?} waited" $?
same "$TMP/o1" "$TMP/st"; check "R7 ... every output byte-identical to the --ordered-merge run (R1)" $?
DTFE_MERGE_STAGE_GB=0 "${R7[@]}" "$TMP/st0" "${C[@]}" "${P3[@]}" > "$TMP/st0.log" 2>&1 || { echo "FAIL: st0 (see $TMP/st0.log)"; exit 1; }
# no free memory and no spill (with the spill on, such a partition goes to disk; below)
DTFE_MERGE_SPILL=0 DTFE_FREE_GB=0.001 "${R7[@]}" "$TMP/stf" "${C[@]}" "${P3[@]}" > "$TMP/stf.log" 2>&1 || { echo "FAIL: stf (see $TMP/stf.log)"; exit 1; }
grep -q "ordered merge: 0 partition(s) finished before their turn" "$TMP/st0.log" && ! grep -q " 0 waited for their turn" "$TMP/st0.log" \
    && grep -q "ordered merge: 0 partition(s) finished before their turn" "$TMP/stf.log" && grep -q "(no room to stage)" "$TMP/stf.log" \
    && same "$TMP/o1" "$TMP/st0" && same "$TMP/o1" "$TMP/stf"
check "R7 DTFE_MERGE_STAGE_GB=0 and no free memory (DTFE_MERGE_SPILL=0): nothing staged, partitions wait their turn, byte-identical" $?
# THE SPILL: the reference is the same split at --max-concurrent 1 (no parking at all), on ONE thread:
# four partitions on four threads deposit each on one, and the double build's CPU sums are equal between thread
# counts only to rounding (README: run-to-run identical grids)
OMP_NUM_THREADS=1 "$BIN" "$SNAP" "$TMP/m1" "${C[@]}" --partition 3 3 3 --max-concurrent 1 --ps-no-overlap > "$TMP/m1.log" 2>&1 \
    || { echo "FAIL: m1 (see $TMP/m1.log)"; exit 1; }
nspill() { sed -n 's/.*ordered merge: .* \([0-9]*\) spilled to disk.*/\1/p' "$1"; }
SPD="$TMP/spill"; mkdir -p "$SPD"
DTFE_FREE_GB=0.001 "${R7[@]}" "$TMP/sp" "${C[@]}" "${P3[@]}" --scratch-dir "$SPD" > "$TMP/sp.log" 2>&1 || { echo "FAIL: sp (see $TMP/sp.log)"; exit 1; }
nsp="$(nspill "$TMP/sp.log")"
grep -q "when the memory cannot hold it, spilled to disk in '$SPD'" "$TMP/sp.log" && [ "${nsp:-0}" -gt 0 ] \
    && grep -q " 0 waited for their turn" "$TMP/sp.log" && [ -z "$(ls -A "$SPD")" ] && same "$TMP/m1" "$TMP/sp" && same "$TMP/o1" "$TMP/sp"
check "R7 no free memory: ${nsp:-?} partition(s) spilled to --scratch-dir (left empty), none waited; byte-identical to --max-concurrent 1 and R1" $?
DTFE_MERGE_SPILL=1 DTFE_SPILL_MIN_GB=2 "${R7[@]}" "$TMP/sf" "${C[@]}" "${P3[@]}" > "$TMP/sf.log" 2>&1 || { echo "FAIL: sf (see $TMP/sf.log)"; exit 1; }
nsf="$(nspill "$TMP/sf.log")"
nst2="$(sed -n 's/.*ordered merge: \([0-9]*\) partition(s) finished before their turn.*/\1/p' "$TMP/sf.log")"
grep -q "every one with sub-grids is spilled to disk" "$TMP/sf.log" && [ "${nsf:-0}" -gt 0 ] && [ "$nsf" = "$nst2" ] \
    && same "$TMP/m1" "$TMP/sf"
check "R7 DTFE_MERGE_SPILL=1 (the system temp folder): all ${nsf:-?} parked partition(s) spilled, byte-identical to --max-concurrent 1" $?
DTFE_MERGE_SPILL=1 DTFE_SPILL_MIN_GB=2 DTFE_TEST_EXIT_AFTER_PARTITIONS=10 "${R7[@]}" "$TMP/sk3" "${C[@]}" "${P3[@]}" > "$TMP/sk3.log" 2>&1; rc1=$?
bits="$(grep -a -m1 '^merged ' "$TMP/sk3.runlog.checkpoint/checkpoint" 2>/dev/null | awk '{print $2}')"
DTFE_MERGE_SPILL=1 DTFE_SPILL_MIN_GB=2 OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/sk3" "${C[@]}" "${P3[@]}" --resume > "$TMP/sk4.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ "$bits" = "111111111100000000000000000" ] && [ "$rc2" = 0 ] && same "$TMP/m1" "$TMP/sk3"
check "R7 spilling run killed after 10 merges (exit $rc1): partitions 0-9 checkpointed (${bits:-none}), resumed byte-identical (exit $rc2)" $?
"${R7[@]}" "$TMP/co" "${C[@]}" "${P3[@]}" --completion-order-merge > "$TMP/co.log" 2>&1 || { echo "FAIL: co (see $TMP/co.log)"; exit 1; }
grep -q "partition merge order  : completion order" "$TMP/co.log" && ! grep -q "ordered merge:" "$TMP/co.log" \
    && "$PY" - "$TMP/o1" "$TMP/co" <<'PYEOF'
import numpy as np, os, sys
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))
for e in ("den", "a_den", "vel"):
    a, b = (np.fromfile(f"{p}.{e}", dtype=REAL).astype(np.float64) for p in sys.argv[1:3])
    if a.size != b.size or not a.size or np.abs(a - b).max() > 1e-5 * max(np.abs(a).max(), 1e-300): sys.exit(1)
PYEOF
check "R7 --completion-order-merge: completion order, nothing staged, the grids equal R1's to float rounding" $?
"$BIN" "$SNAP" "$TMP/bad" "${C[@]}" "${P3[@]}" --ordered-merge --completion-order-merge > "$TMP/bad.log" 2>&1; rc=$?
[ "$rc" != 0 ] && grep -q "contradict each other" "$TMP/bad.log" && [ ! -e "$TMP/bad.den" ]
check "R7 '--ordered-merge --completion-order-merge' refused (exit $rc)" $?
DTFE_TEST_EXIT_AFTER_PARTITIONS=10 "${R7[@]}" "$TMP/sk" "${C[@]}" "${P3[@]}" > "$TMP/sk.log" 2>&1; rc1=$?
bits="$(grep -a -m1 '^merged ' "$TMP/sk.runlog.checkpoint/checkpoint" 2>/dev/null | awk '{print $2}')"
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/sk" "${C[@]}" "${P3[@]}" --resume > "$TMP/sk2.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ "$bits" = "111111111100000000000000000" ] && [ "$rc2" = 0 ] && same "$TMP/o1" "$TMP/sk"
check "R7 killed after 10 merges with partitions staged (exit $rc1): partitions 0-9 checkpointed (${bits:-none}), resumed byte-identical (exit $rc2)" $?
# the gate-guarded case: '--partition 3 3 3' with the concurrency left to the tuner (psPartitionBytes > 0)
GA=(env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 OMP_NUM_THREADS=4 DTFE_TEST_DELAY_PARTITION=0:2 "$BIN" "$SNAP")
"${GA[@]}" "$TMP/sg" "${C[@]}" --partition 3 3 3 > "$TMP/sg.log" 2>&1 || { echo "FAIL: sg (see $TMP/sg.log)"; exit 1; }
DTFE_MERGE_STAGE_GB=0 "${GA[@]}" "$TMP/sg0" "${C[@]}" --partition 3 3 3 > "$TMP/sg0.log" 2>&1 || { echo "FAIL: sg0 (see $TMP/sg0.log)"; exit 1; }
nsg="$(sed -n 's/.*ordered merge: \([0-9]*\) partition(s) finished before their turn.*/\1/p' "$TMP/sg.log")"
grep -q "the memory gate admits the next partition" "$TMP/sg.log" && [ "${nsg:-0}" -gt 0 ] && same "$TMP/sg0" "$TMP/sg"
check "R7 auto-tuned concurrency (the memory gate guards the starts): ${nsg:-?} staged, byte-identical to DTFE_MERGE_STAGE_GB=0" $?
DTFE_MERGE_SPILL=1 DTFE_SPILL_MIN_GB=2 "${GA[@]}" "$TMP/sgs" "${C[@]}" --partition 3 3 3 > "$TMP/sgs.log" 2>&1 || { echo "FAIL: sgs (see $TMP/sgs.log)"; exit 1; }
nsgs="$(nspill "$TMP/sgs.log")"
grep -q "the memory gate admits the next partition" "$TMP/sgs.log" && [ "${nsgs:-0}" -gt 0 ] && same "$TMP/sg0" "$TMP/sgs"
check "R7 ... and spilling under the memory gate (DTFE_MERGE_SPILL=1): ${nsgs:-?} spilled, byte-identical" $?
# the budget room: the free memory plentiful (DTFE_FREE_GB=1000), but no room in the budget beside the
# plan's predicted peak -- every parked partition must spill (on the free-memory rule alone TNG100 z=0 parks 74 GB)
DTFE_TEST_PARK_ROOM_GB=0 DTFE_SPILL_MIN_GB=2 "${GA[@]}" "$TMP/sgr" "${C[@]}" --partition 3 3 3 > "$TMP/sgr.log" 2>&1 || { echo "FAIL: sgr (see $TMP/sgr.log)"; exit 1; }
nsgr="$(nspill "$TMP/sgr.log")"
nstr="$(sed -n 's/.*ordered merge: \([0-9]*\) partition(s) finished before their turn.*/\1/p' "$TMP/sgr.log")"
grep -q "the memory budget leaves beside the plan" "$TMP/sgr.log" && [ "${nsgr:-0}" -gt 0 ] && [ "$nsgr" = "$nstr" ] \
    && same "$TMP/sg0" "$TMP/sgr"
check "R7 ... and no room in the memory budget (DTFE_TEST_PARK_ROOM_GB=0): all ${nsgr:-?} parked partition(s) spilled, byte-identical" $?

echo ">> R2: the look-ahead on the CPU deposit (DTFE_PS_OVERLAP=1) against --ps-no-overlap"
"$BIN" "$SNAP" "$TMP/n1" "${C[@]}" --partition 3 3 3 --max-concurrent 1 --ps-no-overlap > "$TMP/n1.log" 2>&1 \
    || { echo "FAIL: n1 (see $TMP/n1.log)"; exit 1; }
DTFE_PS_OVERLAP=1 "$BIN" "$SNAP" "$TMP/l1" "${C[@]}" --partition 3 3 3 --max-concurrent 1 > "$TMP/l1.log" 2>&1 \
    || { echo "FAIL: l1 (see $TMP/l1.log)"; exit 1; }
grep -q "look-ahead on" "$TMP/l1.log" && ! grep -q "look-ahead on" "$TMP/n1.log"; check "R2 the look-ahead ran (and not with --ps-no-overlap)" $?
same "$TMP/n1" "$TMP/l1"; check "R2 look-ahead vs --ps-no-overlap: every output byte-identical" $?
# the production shape: point evaluation and the tessellation cache (built and saved by the look-ahead thread,
# then a second run whose look-ahead only loads)
"$PY" -c "import numpy as np; np.random.default_rng(3).uniform(0, 100, (2000, 3)).astype(np.float64).tofile('$TMP/pts.bin')"
mkdir -p "$TMP/cache_n" "$TMP/cache_l"
S2=(--sample-points "$TMP/pts.bin" --pts-vel-grad --partition 3 3 3 --max-concurrent 1)
"$BIN" "$SNAP" "$TMP/n2" "${C[@]}" "${S2[@]}" --tessellation-cache "$TMP/cache_n" --ps-no-overlap > "$TMP/n2.log" 2>&1 \
    || { echo "FAIL: n2 (see $TMP/n2.log)"; exit 1; }
for r in l2 l3; do DTFE_PS_OVERLAP=1 "$BIN" "$SNAP" "$TMP/$r" "${C[@]}" "${S2[@]}" --tessellation-cache "$TMP/cache_l" \
    > "$TMP/$r.log" 2>&1 || { echo "FAIL: $r (see $TMP/$r.log)"; exit 1; }; done
[ -s "$TMP/l2.pts_den" ] && grep -q "look-ahead on" "$TMP/l3.log" && grep -q "cached" "$TMP/l3.log"
check "R2 with --sample-points and --tessellation-cache: the look-ahead ran, the second run from the cache" $?
same "$TMP/n2" "$TMP/l2" && same "$TMP/n2" "$TMP/l3"
check "R2 ... grids and '.pts_*' byte-identical (look-ahead building, look-ahead loading, no look-ahead)" $?

echo ">> R3: the look-ahead of a --ps-gpu run"
OBJ=o_ps; precision_double && OBJ=o_ps_d
if [ ! -d "$OBJ" ] || [ -f "$OBJ/.gpu_mode_off" ] || [ ! -f "$OBJ/.gpu_mode_metal" ]; then
    echo "  SKIP R3: $OBJ is not a Metal GPU build"
else
    G=("${C[@]}" --partition 3 3 3 --max-concurrent 1 --ps-gpu)
    "$BIN" "$SNAP" "$TMP/gn" "${G[@]}" --ps-no-overlap > "$TMP/gn.log" 2>&1 || { echo "FAIL: gn (see $TMP/gn.log)"; exit 1; }
    "$BIN" "$SNAP" "$TMP/gl" "${G[@]}" > "$TMP/gl.log" 2>&1 || { echo "FAIL: gl (see $TMP/gl.log)"; exit 1; }
    grep -q "look-ahead on: .* on the GPU" "$TMP/gl.log" \
        && ! grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead" "$TMP/gl.log" "$TMP/gn.log"
    check "R3 a --ps-gpu run at --max-concurrent 1 looks ahead by default (and neither run fell back to the CPU)" $?
    ok=0
    # the velocity too: every PS grid of a GPU deposit is an exact fixed-point sum
    for e in den a_den streams a_streams hidden_streams a_hidden_streams vel; do
        cmp -s "$TMP/gn.$e" "$TMP/gl.$e" || { echo "    differs: .$e"; ok=1; }
    done
    check "R3 GPU look-ahead vs --ps-no-overlap: every grid (the velocity included) byte-identical" $ok
fi

echo ">> R4: crash-safe outputs"
OUT="$TMP/safe"; mkdir -p "$OUT"
sleep 0 & DEAD=$!; wait "$DEAD"   # a pid that no longer runs
echo stale > "$OUT/r.den.part.$DEAD"
echo old > "$OUT/r.den"            # the stale part's final file exists: the part is a leftover, removed
echo only > "$OUT/r.only.part.$DEAD"   # no final file: maybe the only copy of an output, kept (2026-10-11)
echo alive > "$OUT/r.den.part.$$"
"$BIN" "$SNAP" "$OUT/r" "${C[@]}" --partition 2 2 2 --max-concurrent 1 > "$OUT/r.log" 2>&1; rc=$?
[ "$rc" = 0 ] && [ -s "$OUT/r.den" ] && [ -s "$OUT/r.a_den" ]; check "R4 a normal run exits 0 with its outputs in place" $?
[ ! -e "$OUT/r.den.part.$DEAD" ] && grep -q "stale partial output" "$OUT/r.log"; check "R4 the dead pid's '.part' file was removed, and said so" $?
[ -e "$OUT/r.den.part.$$" ]; check "R4 a live pid's '.part' file was kept" $?
[ -e "$OUT/r.only.part.$DEAD" ] && grep -q "Kept the partial output '$OUT/r.only.part.$DEAD'" "$OUT/r.log" \
    && grep -q "mv '$OUT/r.only.part.$DEAD' '$OUT/r.only'" "$OUT/r.log"
check "R4 a dead pid's '.part' file whose final file is missing was kept, with the mv that puts it in place" $?
rm -f "$OUT/r.den.part.$$" "$OUT/r.only.part.$DEAD"
n=$(find "$OUT" -name 'r.*.part.*' | wc -l | tr -d ' ')
[ "$n" = 0 ] && [ ! -e "$OUT/r.runlog.progress" ]; check "R4 no '.part' file and no progress sidecar left after the run" $?
# a MISSING output folder (an unmounted disk is a missing path; a read-only folder would not stop root in CI)
RO="$TMP/ro/gone"; mkdir -p "$TMP/ro"; RTMP="$TMP/rescuetmp"; mkdir -p "$RTMP"
TMPDIR="$RTMP" DTFE_OUTPUT_WAIT=0 "$BIN" "$SNAP" "$RO/r" "${C[@]}" --partition 2 2 2 --max-concurrent 1 > "$TMP/ro.log" 2>&1; rc=$?
RESCUE="$RTMP/dtfe-rescue/gone_r"
[ "$rc" = 3 ] && grep -q "OUTPUT RESCUE" "$TMP/ro.log" && grep -q "mv '$RESCUE'" "$TMP/ro.log"
check "R4 a missing output folder: exit 3 and the rescue folder named with the mv command" $?
ok=0; nf=0
for f in "$OUT"/r.*; do
    case "$f" in *.log|*.runlog*|*.part.*) continue ;; esac
    nf=$((nf+1)); cmp -s "$f" "$RESCUE/$(basename "$f")" || { echo "    rescue differs/missing: $(basename "$f")"; ok=1; }
done
[ "$nf" -gt 0 ] || ok=1
check "R4 the rescued set is complete and byte-identical to the normal run ($nf files)" $ok

echo ">> R5: the progress sidecar while a run goes"
P5="$TMP/prog"
"$BIN" "$SNAP" "$P5" --grid 64 --periodic --input 105 --MpcUnit 1 --verbose 1 --field density gradient \
    --partition 4 4 4 --max-concurrent 1 > "$P5.log" 2>&1 &
BPID=$!
"$PY" - "$P5.runlog.progress" "$BPID" > "$P5.snaps" <<'PYEOF'
import json, os, sys, time
path, pid = sys.argv[1], int(sys.argv[2])
seen = []
while True:
    try:
        with open(path) as f: seen.append(f.read())
    except OSError:
        pass
    try: os.kill(pid, 0)
    except OSError: break
    time.sleep(0.02)
for s in seen:
    if s.strip(): print(s.strip())
PYEOF
wait "$BPID"; rc=$?
[ "$rc" = 0 ]; check "R5 the run finished" $?
"$PY" - "$P5.snaps" "$BPID" "$P5.runlog.progress" <<'PYEOF'
import json, os, sys
snaps = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
keys = {"pid", "phase", "done", "total", "elapsed_s", "eta_s", "rss_gb", "gpu_retries", "predicted_gb", "peak_gb",
        "updated"}
parts = [s for s in snaps if s.get("phase") == "partitions"]
ok = (bool(snaps) and all(keys <= set(s) for s in snaps) and all(s["pid"] == int(sys.argv[2]) for s in snaps)
      and len(parts) >= 2 and all(p["total"] == 64 for p in parts)
      and [p["done"] for p in parts] == sorted(p["done"] for p in parts) and max(p["done"] for p in parts) >= 32
      and all(p["rss_gb"] > 0 and p["peak_gb"] > 0 and p["predicted_gb"] == -1 for p in parts)
      and not os.path.exists(sys.argv[3]))
print(f"    {len(snaps)} snapshots read, phases {sorted({s.get('phase') for s in snaps})}, "
      f"done up to {max([p['done'] for p in parts] or [0])}/64")
sys.exit(0 if ok else 1)
PYEOF
check "R5 the sidecar carries the contract's keys, counts the partitions up and is gone at the end" $?

echo ">> R6: the SUMMARY's predicted-vs-measured memory line"
"$BIN" "$SNAP" "$TMP/mm" --grid 32 --periodic --input 105 --MpcUnit 1 --verbose 1 --field density \
    --partition 2 2 2 > "$TMP/mm.log" 2>&1 || { echo "FAIL: mm (see $TMP/mm.log)"; exit 1; }
"$PY" - "$TMP/mm.log" "$P5.log" <<'PYEOF'
import re, sys
m = re.search(r"memory model\s*:\s*([\d.]+) GB predicted \(auto-tune\) vs ([\d.]+) GB measured", open(sys.argv[1]).read())
print(f"    {m.group(0) if m else 'no memory model line'}")
sys.exit(0 if m and float(m.group(1)) > 0 and float(m.group(2)) > 0 and "memory model" not in open(sys.argv[2]).read() else 1)
PYEOF
check "R6 an auto-tuned run prints its predicted and measured peak (and a run without a plan prints none)" $?

echo "RESULT: $PASS passed, $FAILS failed"
[ "$FAILS" = 0 ] || exit 1
exit 0
