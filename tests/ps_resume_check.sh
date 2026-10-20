#!/usr/bin/env bash
# Resume of a killed partitioned PS-DTFE run (src/partition_checkpoint.h), on a small synthetic
# snapshot split into 64 partitions:
#   C1  an uninterrupted '--ordered-merge' run (the reference) checkpoints, says what it cost, and removes its
#       '<output>.runlog.checkpoint' once the outputs are in place
#   C2  a run killed after 20 partitions (DTFE_TEST_EXIT_AFTER_PARTITIONS=20: a SIGKILL right after a checkpoint)
#       exits 137, writes no output, and leaves the checkpoint with its manifest of >= 20 merged partitions
#   C3  '--resume' with a changed setting ('--grid') refuses, names the differing line, keeps the checkpoint
#   C4  '--resume' (with '--ps-no-overlap' and '--tessellation-cache', which the settings key leaves out) continues: every
#       output byte-identical to the reference, the checkpoint removed; a dead writer's temp file is cleaned up.
#       The same --max-concurrent and thread count as the killed run: the deposit inside a partition splits its
#       sums by the threads a partition gets, so another count differs at float rounding (C6 covers that case)
#   C5  '--resume' with nothing to resume refuses
#   C6  a run killed while merging in COMPLETION order ('--completion-order-merge') resumes too: mass conserved
#   C7  '--scratch-dir': the checkpoint goes to the scratch folder under the name scripts/config.sh
#       ps_checkpoint_dir computes (the basename + a hash of the whole output path)
#   C9  too little room for a checkpoint (DTFE_TEST_CHECKPOINT_FREE_BYTES): the run warns, writes none, and finishes
#       with the same outputs
#   C10 the production path: NO --partition (run_ps_dtfe.sh's default), the auto-tuner splits (DTFE_RAM_GB=0.05:
#       a memory-driven 6^3 split); killed, then '--resume' without --partition on a "bigger machine"
#       (DTFE_RAM_GB=64, where the tuner alone picks 2 2 2) adopts the checkpoint's split and finishes; the grids equal an
#       uninterrupted run's (byte-identical when both ran at the same --max-concurrent)
#   C11 '--interlace' (two passes, the second on a half-cell-shifted grid) writes no checkpoint, and '--resume' with it
#       is refused: one checkpoint and key for both passes would mix the shifted and unshifted grids
#   C12 a run killed before any checkpoint could be written leaves no (empty) checkpoint folder for the driver to resume
#   C13 a checkpoint of another binary ('binary=' key line: the revision AND link time) is refused, naming PS_RESUME=0
#   C14 a commit whose renames failed (exit 3, the outputs left as '.part.<pid>') keeps the checkpoint
#   C8  GPU builds only: a killed '--ps-gpu' run resumes to the exactly summed grids of an uninterrupted one
#       byte for byte in EVERY grid, the velocity and dispersion included (exact fixed-point sums); SKIP otherwise
# Usage: tests/ps_resume_check.sh [--no-build]   (--no-build is accepted and ignored: the binaries are used as built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}"
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
BIN="$(cd "$(dirname "${PS_BIN}")" && pwd)/$(basename "${PS_BIN}")"
TMP="${DTFE_TEST_TMP}/ps_resume_check"
rm -rf "$TMP"; mkdir -p "$TMP"
TMP="$(cd "$TMP" && pwd -P)"
SNAP="$TMP/waves.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 40 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
PASS=0; FAILS=0
check() { if [ "$2" = 0 ]; then PASS=$((PASS+1)); echo "  PASS $1"; else FAILS=$((FAILS+1)); echo "  FAIL $1"; fi; }
# same <a> <b>, merged_count <checkpoint folder>: tests/lib.sh
C=(--grid 48 --periodic --input 105 --MpcUnit 1 --field density velocity density_a velocity_a dispersion_a
   --partition 4 4 4)

echo ">> C1: the uninterrupted --ordered-merge reference (64 partitions, --max-concurrent 4)"
"$BIN" --help 2>&1 | grep -q -- "--resume"; check "C1 '--resume' is in the help" $?
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/ref" "${C[@]}" --verbose 2 --max-concurrent 4 --ordered-merge > "$TMP/ref.log" 2>&1 \
    || { echo "FAIL: ref (see $TMP/ref.log)"; exit 1; }
grep -q "checkpoints of the merged grids go to '$TMP/ref.runlog.checkpoint'" "$TMP/ref.log" \
    && grep -q "checkpoints: [0-9]* written" "$TMP/ref.log" && [ ! -e "$TMP/ref.runlog.checkpoint" ]
check "C1 the run checkpointed and removed '<output>.runlog.checkpoint' after writing its outputs" $?
grep -a "checkpoints: " "$TMP/ref.log" | sed 's/^/   /'

echo ">> C2: killed after 20 partitions"
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=20 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" --verbose 2 --max-concurrent 4 --ordered-merge \
    > "$TMP/k.log" 2>&1; rc=$?
CK="$TMP/k.runlog.checkpoint"
n="$(merged_count "$CK" 2>/dev/null || echo 0)"
[ "$rc" = 137 ] && [ ! -e "$TMP/k.den" ] && [ -s "$CK/checkpoint" ] && [ "${n:-0}" -ge 20 ] && [ "${n:-0}" -lt 64 ]
check "C2 exit 137 (SIGKILL), no outputs, a checkpoint of ${n:-?} of 64 merged partitions" $?

echo ">> C3: --resume with another --grid refuses"
"$BIN" "$SNAP" "$TMP/k" "${C[@]/48/40}" --verbose 2 --max-concurrent 4 --resume > "$TMP/k3.log" 2>&1; rc=$?
[ "$rc" != 0 ] && grep -q "'--resume' refused" "$TMP/k3.log" && grep -q "grid=40" "$TMP/k3.log" \
    && [ -s "$CK/checkpoint" ] && [ ! -e "$TMP/k.den" ]
check "C3 refused, naming the differing line, and the checkpoint is kept" $?

echo ">> C4: --resume (with --ps-no-overlap --tessellation-cache)"
sleep 0 & DEAD=$!; wait "$DEAD"   # a pid that no longer runs
echo torn > "$CK/checkpoint.tmp.$DEAD"
mkdir -p "$TMP/tc"
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" --max-concurrent 4 --resume --verbose 2 --ps-no-overlap \
    --tessellation-cache "$TMP/tc" > "$TMP/k4.log" 2>&1; rc=$?
[ "$rc" = 0 ] && grep -q -- "--resume: $n of 64 partitions are already in the checkpoint" "$TMP/k4.log" \
    && grep -q "partitions $((64 - n))/$((64 - n))" "$TMP/k4.log"
check "C4 the resume computed only the $((64 - n)) partitions the checkpoint lacked" $?
same "$TMP/ref" "$TMP/k"; check "C4 every output byte-identical to the uninterrupted --ordered-merge run" $?
[ ! -e "$CK" ]; check "C4 the checkpoint (and the dead writer's temp file) removed on success" $?

echo ">> C5: --resume with nothing to resume"
"$BIN" "$SNAP" "$TMP/none" "${C[@]}" --verbose 2 --max-concurrent 4 --resume > "$TMP/none.log" 2>&1; rc=$?
[ "$rc" != 0 ] && grep -q "nothing to resume" "$TMP/none.log" && [ ! -e "$TMP/none.den" ]
check "C5 refused: nothing to resume" $?

echo ">> C6: killed while merging in completion order, then resumed"
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=25 "$BIN" "$SNAP" "$TMP/u" "${C[@]}" --verbose 2 --max-concurrent 4 \
    --completion-order-merge > "$TMP/u.log" 2>&1; rc1=$?
# (the resume keeps the flag, as run_ps_dtfe.sh ORDERED_MERGE=0 would: --resume merges in index order over it)
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/u" "${C[@]}" --verbose 2 --max-concurrent 4 --completion-order-merge --resume \
    > "$TMP/u2.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ "$rc2" = 0 ] && [ ! -e "$TMP/u.runlog.checkpoint" ]
check "C6 killed (137), resumed (0), checkpoint removed" $?
"$PY" - "$TMP/ref" "$TMP/u" <<'PYEOF'
import numpy as np, os, sys
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))
worst = 0.
for e in ("den", "a_den", "a_vel"):
    a, b = (np.fromfile(f"{p}.{e}", dtype=REAL).astype(np.float64) for p in sys.argv[1:3])
    if a.size != b.size or not a.size: sys.exit(1)
    d = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))
    worst = max(worst, d)
    print(f"    .{e}: total {b.sum():.6e} vs {a.sum():.6e}, peak-relative difference {d:.1e}")
sys.exit(0 if worst < 1e-5 else 1)
PYEOF
check "C6 the grids equal the reference to float rounding (no partition lost or added twice)" $?

echo ">> C7: --scratch-dir holds the checkpoint"
SCR="$TMP/scratch"; mkdir -p "$SCR"
SCK="$(source scripts/config.sh && ps_checkpoint_dir "$TMP/s" "$SCR")"
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=10 "$BIN" "$SNAP" "$TMP/s" "${C[@]}" --verbose 2 --max-concurrent 4 --ordered-merge \
    --scratch-dir "$SCR" > "$TMP/s.log" 2>&1; rc1=$?
[ "$rc1" = 137 ] && [ -s "$SCK/checkpoint" ] && [ ! -e "$TMP/s.runlog.checkpoint" ]
check "C7 the checkpoint is in the scratch folder, where config.sh ps_checkpoint_dir says ($(basename "$SCK"))" $?
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/s" "${C[@]}" --verbose 2 --max-concurrent 4 --scratch-dir "$SCR" --resume > "$TMP/s2.log" 2>&1; rc2=$?
[ "$rc2" = 0 ] && [ ! -e "$SCK" ] && same "$TMP/ref" "$TMP/s"
check "C7 resumed from the scratch folder: byte-identical, checkpoint removed" $?

echo ">> C9: no room for a checkpoint"
OMP_NUM_THREADS=4 DTFE_TEST_CHECKPOINT_FREE_BYTES=1000000 "$BIN" "$SNAP" "$TMP/room" "${C[@]}" --verbose 2 --max-concurrent 4 \
    --ordered-merge > "$TMP/room.log" 2>&1; rc=$?
[ "$rc" = 0 ] && grep -q "checkpoint could not be written (only 0 GB free" "$TMP/room.log" \
    && ! grep -q "checkpoints: [0-9]* written" "$TMP/room.log" && [ ! -e "$TMP/room.runlog.checkpoint" ] \
    && same "$TMP/ref" "$TMP/room"
check "C9 warned, wrote no checkpoint, and the outputs are the reference's" $?

echo ">> C10: auto-tuned split (no --partition), killed, resumed"
A=(--grid 48 --periodic --input 105 --MpcUnit 1 --verbose 2 --field density velocity density_a velocity_a)
AT=(env OMP_NUM_THREADS=4 DTFE_RAM_GB=0.05 DTFE_AUTO_MINN=1000)
"${AT[@]}" "$BIN" "$SNAP" "$TMP/aref" "${A[@]}" > "$TMP/aref.log" 2>&1 || { echo "FAIL: aref (see $TMP/aref.log)"; exit 1; }
"${AT[@]}" DTFE_TEST_EXIT_AFTER_PARTITIONS=50 "$BIN" "$SNAP" "$TMP/a" "${A[@]}" > "$TMP/a1.log" 2>&1; rc1=$?
splitof() { grep -a -m1 'Lagrangian-space partitioning with' "$1" | sed 's/.*partitions \[//; s/\].*//'; }
split="$(splitof "$TMP/aref.log")"
# the resume sees a big machine (DTFE_RAM_GB=64): left alone the auto-tuner would pick 2 2 2 there and the key
# would refuse; adopting the checkpoint's split, it only tunes --max-concurrent
env OMP_NUM_THREADS=4 DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "$BIN" "$SNAP" "$TMP/a" "${A[@]}" --resume > "$TMP/a2.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ "$rc2" = 0 ] && [ -n "$split" ] && grep -q "AUTO-TUNE: .* -> --partition" "$TMP/aref.log" \
    && grep -q -- "--resume: the partition split \[.*\] is the checkpoint's" "$TMP/a2.log" \
    && [ "$(splitof "$TMP/a2.log")" = "$split" ] \
    && grep -q -- "--resume: 50 of " "$TMP/a2.log" && [ ! -e "$TMP/a.runlog.checkpoint" ]
check "C10 the auto-tuner's split [${split}] was adopted from the checkpoint, the resume finished and removed it" $?
mcof() { grep -a -m1 'partition triangulation(s) concurrently' "$1" | sed 's/.*building up to *//; s/ .*//'; }
# the memory-driven split merges in index order at --max-concurrent 1 (the killed run too): byte-identical then
if [ "$(mcof "$TMP/aref.log")" = 1 ] && [ "$(mcof "$TMP/a1.log")" = 1 ] && [ "$(mcof "$TMP/a2.log")" = 1 ]; then
    same "$TMP/aref" "$TMP/a"; check "C10 every output byte-identical to the uninterrupted auto-tuned run (same --max-concurrent)" $?
else
    "$PY" - "$TMP/aref" "$TMP/a" <<'PYEOF'
import numpy as np, os, sys
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))
worst = 0.
for e in ("den", "a_den", "a_vel"):
    a, b = (np.fromfile(f"{p}.{e}", dtype=REAL).astype(np.float64) for p in sys.argv[1:3])
    if a.size != b.size or not a.size: sys.exit(1)
    worst = max(worst, float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)))
print(f"    peak-relative difference {worst:.1e}")
sys.exit(0 if worst < 1e-5 else 1)
PYEOF
    check "C10 the grids equal the uninterrupted auto-tuned run's to float rounding (another --max-concurrent)" $?
fi

echo ">> C11: '--interlace' is not checkpointed, and '--resume' refuses it"
# both interlace passes (the second on a half-cell-shifted grid) would share ONE checkpoint and key, so a resume
# would mix the shifted and unshifted grids: a killed interlaced run leaves nothing to resume from.
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=5 "$BIN" "$SNAP" "$TMP/il" "${C[@]}" --verbose 2 --max-concurrent 4 \
    --interlace > "$TMP/il.log" 2>&1; rc1=$?
"$BIN" "$SNAP" "$TMP/il" "${C[@]}" --verbose 2 --max-concurrent 4 --interlace --resume > "$TMP/il2.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ ! -e "$TMP/il.runlog.checkpoint" ] && grep -q "no checkpoints: an '--interlace' run" "$TMP/il.log" \
    && [ "$rc2" != 0 ] && grep -q "'--resume' cannot be combined with '--interlace'" "$TMP/il2.log"
check "C11 a killed --interlace run leaves no checkpoint (exit $rc1), and --resume --interlace is refused" $?
# the second pass runs on a copy of the options with 'interlace' cleared: it must not checkpoint either
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/il" "${C[@]}" --verbose 2 --max-concurrent 4 --interlace > "$TMP/il3.log" 2>&1; rc=$?
[ "$rc" = 0 ] && ! grep -q "checkpoints: [0-9]* written" "$TMP/il3.log" \
    && [ "$(grep -c "no checkpoints: an '--interlace' run" "$TMP/il3.log")" = 2 ] && [ ! -e "$TMP/il.runlog.checkpoint" ]
check "C11 a whole --interlace run (both passes) writes no checkpoint (exit $rc)" $?

echo ">> C12: a run killed without ever writing a checkpoint leaves no checkpoint folder"
# a folder made before the room check would outlive the kill empty, and run_ps_dtfe.sh would resume from it
OMP_NUM_THREADS=4 DTFE_TEST_CHECKPOINT_FREE_BYTES=1000000 DTFE_TEST_EXIT_AFTER_PARTITIONS=10 "$BIN" "$SNAP" "$TMP/nr" \
    "${C[@]}" --verbose 2 --max-concurrent 4 > "$TMP/nr.log" 2>&1; rc=$?
[ "$rc" = 137 ] && grep -q "checkpoint could not be written (only 0 GB free" "$TMP/nr.log" && [ ! -e "$TMP/nr.runlog.checkpoint" ]
check "C12 killed (exit $rc) with no room for a checkpoint: no '<output>.runlog.checkpoint' folder left" $?

echo ">> C13: a checkpoint of another binary (a rebuild) is refused"
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=5 "$BIN" "$SNAP" "$TMP/rb" "${C[@]}" --verbose 2 --max-concurrent 4 \
    > "$TMP/rb.log" 2>&1; rc1=$?
# the killed run's checkpoint as an older build would have left it: its 'binary=' key line rewritten
"$PY" - "$TMP/rb.runlog.checkpoint/checkpoint" <<'PYEOF'
import re, sys
p = sys.argv[1]
b = open(p, "rb").read()
nb, n = re.subn(rb"\nbinary=[^\n]*\n", b"\nbinary=an-older-build (built 2000-01-01T00:00:00Z)\n", b, count=1)
open(p, "wb").write(nb)
sys.exit(0 if n == 1 else 1)
PYEOF
rcp=$?
"$BIN" "$SNAP" "$TMP/rb" "${C[@]}" --verbose 2 --max-concurrent 4 --resume > "$TMP/rb2.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ "$rcp" = 0 ] && [ "$rc2" != 0 ] && grep -q "'--resume' refused" "$TMP/rb2.log" \
    && grep -q "checkpoint: binary=an-older-build" "$TMP/rb2.log" && grep -q "PS_RESUME=0" "$TMP/rb2.log" \
    && [ -s "$TMP/rb.runlog.checkpoint/checkpoint" ]
check "C13 refused, naming the 'binary=' line and PS_RESUME=0; the checkpoint is kept" $?
rm -rf "$TMP/rb.runlog.checkpoint"

echo ">> C14: outputs that could not be put in place keep the checkpoint"
# a commit whose renames fail exits 3 with the outputs as '.part.<pid>' files; the checkpoint must not be deleted then
mkdir -p "$TMP/cm.den"; : > "$TMP/cm.den/blocker"   # a folder where the '.den' output goes: its rename fails
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/cm" "${C[@]}" --verbose 2 --max-concurrent 4 > "$TMP/cm.log" 2>&1; rc=$?
[ "$rc" = 3 ] && grep -q "could not be put in place" "$TMP/cm.log" && [ -s "$TMP/cm.runlog.checkpoint/checkpoint" ]
check "C14 exit $rc (renames failed): the checkpoint stays for a resume" $?
rm -rf "$TMP/cm.den" "$TMP/cm.runlog.checkpoint" "$TMP"/cm.*.part.*

echo ">> C8: a killed --ps-gpu run"
OBJ=o_ps; precision_double && OBJ=o_ps_d
if [ ! -d "$OBJ" ] || [ -f "$OBJ/.gpu_mode_off" ] || [ ! -f "$OBJ/.gpu_mode_metal" ]; then
    echo "  SKIP C8: $OBJ is not a Metal GPU build"
else
    G=("${C[@]}" --verbose 2 --max-concurrent 1 --ordered-merge --ps-gpu)
    "$BIN" "$SNAP" "$TMP/gref" "${G[@]}" > "$TMP/gref.log" 2>&1 || { echo "FAIL: gref (see $TMP/gref.log)"; exit 1; }
    DTFE_TEST_EXIT_AFTER_PARTITIONS=12 "$BIN" "$SNAP" "$TMP/g" "${G[@]}" > "$TMP/g1.log" 2>&1; rc1=$?
    "$BIN" "$SNAP" "$TMP/g" "${G[@]}" --resume > "$TMP/g2.log" 2>&1; rc2=$?
    [ "$rc1" = 137 ] && [ "$rc2" = 0 ] && grep -q -- "--resume: 12 of 64" "$TMP/g2.log" \
        && ! grep -q "ran on the CPU instead" "$TMP/gref.log" "$TMP/g1.log" "$TMP/g2.log"
    check "C8 killed after 12 GPU partitions, resumed the other 52 (no CPU fallback)" $?
    ok=0
    # every grid, the velocity included: every PS grid of a GPU deposit is an exact fixed-point sum
    # (the moments too, ps_deposit_params.h PSFxGrid) and the partitions merge in index order, so the resumed run
    # is byte-identical in all of them
    for e in den a_den streams a_streams hidden_streams a_hidden_streams vel a_vel a_velDisp; do
        cmp -s "$TMP/gref.$e" "$TMP/g.$e" || { echo "    differs: .$e"; ok=1; }
    done
    check "C8 every grid (velocity and dispersion included: exact fixed-point sums) byte-identical to the uninterrupted GPU run" $ok
fi

echo "RESULT: $PASS passed, $FAILS failed"
[ "$FAILS" = 0 ] || exit 1
exit 0
