#!/usr/bin/env bash
# Resume of a killed partitioned STANDARD-DTFE run (2026-10-13): the standard partition loop checkpoints through
# PS-DTFE's code (src/partition_checkpoint.h: beginLoop / afterMerge / reportWrites; tests/ps_resume_check.sh is
# the PS half). demo/z0_64.gadget (262144 particles, z=0, tracked in the repo) on a 40^3 grid, --periodic
# --padding 5, split 3 3 3 (27 partitions), 2 threads each (OMP_NUM_THREADS=2 --max-concurrent 2), the fields
# density velocity density_a velocity_a gradient_a tweb_a vweb_a: the T-web and the V-web are made AFTER the loop
# from the full density / gradient grids, so they test that the checkpoint holds what the post-processing needs.
#   D1  '--resume' is in both helps; an uninterrupted run (the reference) says where its checkpoints go, writes
#       some, and removes '<output>.runlog.checkpoint' once its outputs are in place
#   D2  a run killed after 10 partitions (DTFE_TEST_EXIT_AFTER_PARTITIONS=10: a SIGKILL right after a checkpoint)
#       exits 137, writes no output, and leaves a checkpoint listing exactly 10 of 27 (the loop is serial)
#   D3  '--resume' with a changed setting ('--grid') refuses, names the differing line and DTFE_RESUME=0 (the
#       launcher's results.py RESUME_REFUSED keys on "'--resume' refused"), keeps the checkpoint
#   D4  '--resume' continues and is killed again after 3 more partitions: its progress sidecar counts the restored
#       ones as done (done 12 of 27 -- the 3rd computed one died in its checkpoint write, before its progress line);
#       a second '--resume' (with '--tessellation-cache', which the key leaves out) computes the last 14 only, every
#       output -- the T-web and V-web included -- byte-identical to the reference, the checkpoint removed, a dead
#       writer's temp file cleaned up
#   D5  '--resume' with nothing to resume refuses; on a run of one triangulation ('--partition 1') it refuses too
#   D6  '--scratch-dir': the checkpoint goes to the scratch folder under the name scripts/config.sh
#       ps_checkpoint_dir computes, and the resume from there is byte-identical
#   D7  '--interlace' (two passes) writes no checkpoint; '--resume --interlace' is refused. DTFE_CHECKPOINT=0
#       writes none either
#   D8  a checkpoint of another binary ('binary=' key line) is refused, naming the line and DTFE_RESUME=0; a PS-DTFE
#       checkpoint is not read as a standard one
#   D9  the production path: NO --partition, the auto-tuner splits (DTFE_RAM_GB=0.08: 5 5 5 at 1-2 threads since the
#       standard memory plan of 2026-10-13); killed, then '--resume' without --partition on a "bigger machine" (DTFE_RAM_GB=64, where the tuner
#       alone would not split) adopts the checkpoint's split and finishes, byte-identical, at the killed run's thread
#       count (passed as --max-concurrent: the count is left out of the key on purpose, as PS-DTFE's --max-concurrent,
#       but another count re-splits each partition, which with this short padding moves the grids beyond rounding)
#   D10 outputs that could not be put in place (exit 3) keep the checkpoint
#   D11 scripts/run_dtfe.sh (copied beside a stand-in binary): a checkpoint file beside the outputs, or under
#       SCRATCH_DIR, makes the run pass --resume and say so; DTFE_RESUME=0, and a folder without a 'checkpoint'
#       file, do not
#   D12 GPU builds only (a GPU stamp in o/ or o_d/; SKIP otherwise): a killed '--gpu --exact-average' run (the
#       production standard) resumes; '.den' (CPU) byte-identical, and the '_a' grids too (the GPU's exact fixed-point
#       sums, 2026-10-13; within 1e-5 of their peak when DTFE_METAL_FLOAT_GRIDS / DTFE_GPU_FLOAT_GRIDS=1)
#   D13 (2026-10-14) a '--sample-points --pts-vel-grad' plane (64^2 points across the box, the shape of the
#       dtfe27planes runs) with the same split: the uninterrupted run checkpoints its point answers too and removes the
#       checkpoint; killed after 10 partitions it leaves exactly one 'points.<i>' file per merged partition; a resume
#       with another point file under the same name refuses, naming the 'sample-points' key line; '--resume' reads the
#       10 partitions' answers back, computes the other 17, and every output -- the '.pts_*' files and the grids -- is
#       byte-identical to the uninterrupted run (until 2026-10-14 such a run wrote no checkpoint and '--resume'
#       refused it: 'its point accumulators are not checkpointed')
# About a minute on the Mac (about 30 runs of 2-4 s).
# Usage: tests/dtfe_resume_check.sh [--no-build]   (--no-build is accepted and ignored: the binaries are used as built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${DTFE_BIN}"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
[ -x "${DTFE_BIN}" ] || { echo "FAIL: ${DTFE_BIN} is not built"; exit 1; }
BIN="$(cd "$(dirname "${DTFE_BIN}")" && pwd)/$(basename "${DTFE_BIN}")"
SNAP="$ROOT/demo/z0_64.gadget"
[ -f "$SNAP" ] || { echo "FAIL: $SNAP is missing (it is tracked in the repository)"; exit 1; }
TMP="${DTFE_TEST_TMP}/dtfe_resume_check"
rm -rf "$TMP"; mkdir -p "$TMP"
TMP="$(cd "$TMP" && pwd -P)"
OBJ=o; precision_double && OBJ=o_d
PASS=0; FAILS=0
check() { if [ "$2" = 0 ]; then PASS=$((PASS+1)); echo "  PASS $1"; else FAILS=$((FAILS+1)); echo "  FAIL $1"; fi; }
# same <a> <b>: every output of run a has a byte-identical twin in run b (and there is at least one); the logs, the
# run's bookkeeping and the padding test's per-thread '.badGridPoints<n>' (written in place, last partition wins)
# are not outputs
same() {
    local n=0 f
    for f in "$1".*; do
        case "$f" in *.log|*.runlog*|*.part.*|*.badGridPoints*) continue ;; esac
        cmp -s "$f" "$2${f#"$1"}" || { echo "    differs: ${f#"$1"}"; return 1; }
        n=$((n+1))
    done
    [ "$n" -gt 0 ]
}
merged_count() { grep -a -m1 '^merged_count' "$1/checkpoint" | awk '{print $2}'; }
C=(--grid 40 --periodic --padding 5 --field density velocity density_a velocity_a gradient_a tweb_a vweb_a
   --lambda_th 0.3 --verbose 2 --partition 3 3 3)
RUN=(env OMP_NUM_THREADS=2 "$BIN" "$SNAP")
MC=(--max-concurrent 2)

echo ">> D1: the uninterrupted reference (27 partitions)"
# (into files first: under pipefail a 'grep -q' that stops reading early fails the pipe with the binary's SIGPIPE)
"$BIN" --help > "$TMP/help.txt" 2>&1; "$BIN" --full_help > "$TMP/full_help.txt" 2>&1
grep -q -- "--resume" "$TMP/help.txt" && grep -q "standard DTFE, partitioned runs" "$TMP/full_help.txt"
check "D1 '--resume' is in the help and in the full help" $?
"${RUN[@]}" "$TMP/ref" "${C[@]}" "${MC[@]}" > "$TMP/ref.log" 2>&1 || { echo "FAIL: ref (see $TMP/ref.log)"; exit 1; }
grep -q "DTFE: checkpoints of the merged grids go to '$TMP/ref.runlog.checkpoint'" "$TMP/ref.log" \
    && grep -q "checkpoints: [0-9]* written" "$TMP/ref.log" && [ ! -e "$TMP/ref.runlog.checkpoint" ] \
    && [ -s "$TMP/ref.a_tweb" ] && [ -s "$TMP/ref.a_velVweb" ]
check "D1 the run checkpointed and removed '<output>.runlog.checkpoint' after writing its outputs" $?
grep -a "checkpoints: " "$TMP/ref.log" | sed 's/^/   /'

echo ">> D2: killed after 10 partitions"
DTFE_TEST_EXIT_AFTER_PARTITIONS=10 "${RUN[@]}" "$TMP/k" "${C[@]}" "${MC[@]}" > "$TMP/k1.log" 2>&1; rc=$?
CK="$TMP/k.runlog.checkpoint"
n="$(merged_count "$CK" 2>/dev/null || echo 0)"
[ "$rc" = 137 ] && [ ! -e "$TMP/k.den" ] && [ ! -e "$TMP/k.a_den" ] && [ -s "$CK/checkpoint" ] && [ "${n:-0}" = 10 ] \
    && [ "$(head -1 "$CK/checkpoint")" = "DTFE-CHECKPOINT 1" ]
check "D2 exit $rc (137 = SIGKILL), no outputs, a standard checkpoint of ${n:-?} of 27 partitions (10 expected)" $?

echo ">> D3: --resume with another --grid refuses"
"${RUN[@]}" "$TMP/k" "${C[@]/40/32}" "${MC[@]}" --resume > "$TMP/k3.log" 2>&1; rc=$?
[ "$rc" != 0 ] && grep -q "'--resume' refused" "$TMP/k3.log" && grep -q "grid=32" "$TMP/k3.log" \
    && grep -q "run_dtfe.sh: DTFE_RESUME=0" "$TMP/k3.log" && [ -s "$CK/checkpoint" ] && [ ! -e "$TMP/k.den" ]
check "D3 refused, naming the differing line and DTFE_RESUME=0; the checkpoint is kept" $?

echo ">> D4: --resume, killed again after 3 more, then resumed to the end"
DTFE_TEST_EXIT_AFTER_PARTITIONS=3 "${RUN[@]}" "$TMP/k" "${C[@]}" "${MC[@]}" --resume > "$TMP/k4a.log" 2>&1; rc=$?
n2="$(merged_count "$CK" 2>/dev/null || echo 0)"
side="$(cat "$TMP/k.runlog.progress" 2>/dev/null)"
[ "$rc" = 137 ] && grep -q -- "DTFE: --resume: 10 of 27 partitions are already in the checkpoint" "$TMP/k4a.log" \
    && grep -q "partitions 11/27" "$TMP/k4a.log" && [ "${n2:-0}" = 13 ]
check "D4 the first resume started at partition 11 of 27 and was killed with 13 in the checkpoint (exit $rc)" $?
"$PY" - "$side" <<'PYEOF'
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception:
    sys.exit(1)
print(f"    sidecar: phase {d.get('phase')}, done {d.get('done')} of {d.get('total')}")
sys.exit(0 if (d.get("phase"), d.get("done"), d.get("total")) == ("partitions", 12, 27) else 1)
PYEOF
check "D4 the progress sidecar of the resumed run counts the restored partitions as done (12 of 27)" $?
sleep 0 & DEAD=$!; wait "$DEAD"   # a pid that no longer runs
echo torn > "$CK/checkpoint.tmp.$DEAD"
mkdir -p "$TMP/tc"
"${RUN[@]}" "$TMP/k" "${C[@]}" "${MC[@]}" --resume --tessellation-cache "$TMP/tc" > "$TMP/k4b.log" 2>&1; rc=$?
[ "$rc" = 0 ] && grep -q -- "--resume: 13 of 27 partitions are already in the checkpoint" "$TMP/k4b.log" \
    && [ "$(grep -c '<<< Interpolating the fields for partition' "$TMP/k4b.log")" = 14 ] \
    && grep -q "partitions 27/27" "$TMP/k4b.log"
check "D4 the second resume computed only the 14 partitions the checkpoint lacked (exit $rc)" $?
same "$TMP/ref" "$TMP/k"; check "D4 every output (T-web and V-web included) byte-identical to the uninterrupted run" $?
[ ! -e "$CK" ]; check "D4 the checkpoint (and the dead writer's temp file) removed on success" $?

echo ">> D5: --resume with nothing to resume, and on one triangulation"
"${RUN[@]}" "$TMP/none" "${C[@]}" "${MC[@]}" --resume > "$TMP/none.log" 2>&1; rc1=$?
ONE=("${C[@]:0:$((${#C[@]} - 3))}" 1 1 1)   # the same options, '--partition 1 1 1': one triangulation
"${RUN[@]}" "$TMP/one" "${ONE[@]}" --resume > "$TMP/one.log" 2>&1; rc2=$?
[ "$rc1" != 0 ] && grep -q "nothing to resume" "$TMP/none.log" && [ ! -e "$TMP/none.den" ] \
    && [ "$rc2" != 0 ] && grep -q "'--resume' continues a partitioned run" "$TMP/one.log" && [ ! -e "$TMP/one.den" ]
check "D5 refused: nothing to resume; not a partitioned run" $?

echo ">> D6: --scratch-dir holds the checkpoint"
SCR="$TMP/scratch"; mkdir -p "$SCR"
SCK="$(source scripts/config.sh && ps_checkpoint_dir "$TMP/s" "$SCR")"
DTFE_TEST_EXIT_AFTER_PARTITIONS=7 "${RUN[@]}" "$TMP/s" "${C[@]}" "${MC[@]}" --scratch-dir "$SCR" > "$TMP/s1.log" 2>&1; rc1=$?
[ "$rc1" = 137 ] && [ -s "$SCK/checkpoint" ] && [ ! -e "$TMP/s.runlog.checkpoint" ]
check "D6 the checkpoint is in the scratch folder, where config.sh ps_checkpoint_dir says ($(basename "$SCK"))" $?
"${RUN[@]}" "$TMP/s" "${C[@]}" "${MC[@]}" --scratch-dir "$SCR" --resume > "$TMP/s2.log" 2>&1; rc2=$?
[ "$rc2" = 0 ] && [ ! -e "$SCK" ] && same "$TMP/ref" "$TMP/s"
check "D6 resumed from the scratch folder: byte-identical, checkpoint removed" $?

echo ">> D7: '--interlace' and DTFE_CHECKPOINT=0 write no checkpoint"
DTFE_TEST_EXIT_AFTER_PARTITIONS=5 "${RUN[@]}" "$TMP/il" "${C[@]}" "${MC[@]}" --interlace > "$TMP/il.log" 2>&1; rc1=$?
"${RUN[@]}" "$TMP/il" "${C[@]}" "${MC[@]}" --interlace --resume > "$TMP/il2.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ ! -e "$TMP/il.runlog.checkpoint" ] && grep -q "no checkpoints: an '--interlace' run" "$TMP/il.log" \
    && [ "$rc2" != 0 ] && grep -q "'--resume' cannot be combined with '--interlace'" "$TMP/il2.log"
check "D7 a killed --interlace run leaves no checkpoint (exit $rc1), and --resume --interlace is refused" $?
DTFE_CHECKPOINT=0 DTFE_TEST_EXIT_AFTER_PARTITIONS=5 "${RUN[@]}" "$TMP/off" "${C[@]}" "${MC[@]}" > "$TMP/off.log" 2>&1; rc=$?
[ "$rc" = 137 ] && [ ! -e "$TMP/off.runlog.checkpoint" ] && ! grep -q "checkpoints of the merged grids go to" "$TMP/off.log"
check "D7 DTFE_CHECKPOINT=0: no checkpoint written (killed, exit $rc)" $?

echo ">> D8: a checkpoint of another binary, or of PS-DTFE, is refused"
DTFE_TEST_EXIT_AFTER_PARTITIONS=5 "${RUN[@]}" "$TMP/rb" "${C[@]}" "${MC[@]}" > "$TMP/rb.log" 2>&1; rc1=$?
"$PY" - "$TMP/rb.runlog.checkpoint/checkpoint" <<'PYEOF'
import re, sys
p = sys.argv[1]
b = open(p, "rb").read()
nb, n = re.subn(rb"\nbinary=[^\n]*\n", b"\nbinary=an-older-build (built 2000-01-01T00:00:00Z)\n", b, count=1)
open(p, "wb").write(nb)
sys.exit(0 if n == 1 else 1)
PYEOF
rcp=$?
"${RUN[@]}" "$TMP/rb" "${C[@]}" "${MC[@]}" --resume > "$TMP/rb2.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ "$rcp" = 0 ] && [ "$rc2" != 0 ] && grep -q "'--resume' refused" "$TMP/rb2.log" \
    && grep -q "checkpoint: binary=an-older-build" "$TMP/rb2.log" && grep -q "DTFE_RESUME=0" "$TMP/rb2.log" \
    && [ -s "$TMP/rb.runlog.checkpoint/checkpoint" ]
check "D8 refused, naming the 'binary=' line and DTFE_RESUME=0; the checkpoint is kept" $?
# the same file as PS-DTFE writes it: its first line says so, and the standard binary will not read it
"$PY" - "$TMP/rb.runlog.checkpoint/checkpoint" <<'PYEOF'
import sys
p = sys.argv[1]
b = open(p, "rb").read()
open(p, "wb").write(b.replace(b"DTFE-CHECKPOINT 1\n", b"DTFE-PS-CHECKPOINT 1\n", 1))
PYEOF
"${RUN[@]}" "$TMP/rb" "${C[@]}" "${MC[@]}" --resume > "$TMP/rb3.log" 2>&1; rc3=$?
[ "$rc3" != 0 ] && grep -q "is not a DTFE checkpoint" "$TMP/rb3.log"
check "D8 a PS-DTFE checkpoint is not a standard one ('is not a DTFE checkpoint')" $?
rm -rf "$TMP/rb.runlog.checkpoint"

echo ">> D9: auto-tuned split (no --partition), killed, resumed"
A=(--grid 40 --periodic --padding 5 --field density velocity density_a velocity_a --verbose 2)
AT=(env OMP_NUM_THREADS=2 DTFE_RAM_GB=0.08 DTFE_AUTO_MINN=1000)
"${AT[@]}" "$BIN" "$SNAP" "$TMP/aref" "${A[@]}" > "$TMP/aref.log" 2>&1 || { echo "FAIL: aref (see $TMP/aref.log)"; exit 1; }
"${AT[@]}" DTFE_TEST_EXIT_AFTER_PARTITIONS=5 "$BIN" "$SNAP" "$TMP/a" "${A[@]}" > "$TMP/a1.log" 2>&1; rc1=$?
splitof() { grep -a -m1 'partitions defined via the grid' "$1" | sed 's/.*the grid \[//; s/\].*//'; }
threadsof() { grep -a -m1 'AUTO-TUNE: .* thread(s)' "$1" | sed 's/.*, \([0-9]*\) thread(s).*/\1/'; }
split="$(splitof "$TMP/aref.log")"
t1="$(threadsof "$TMP/a1.log")"
# The resume runs at the killed run's thread count (2026-10-13, integration of the parity batch). '--max-concurrent' is
# left out of the key on purpose (a run killed for memory may resume with fewer threads), but the thread count is the
# sub-box split inside each partition, and this test's padding of 5 spacings is short at z = 0: the resume re-tuned on
# the "bigger machine" took 2 threads where the double build's killed run had 1 (the standard memory plan of the same
# batch), and its grids differed by 2.8e-2 of the peak, not by float rounding. So the thread count is passed, as a user
# resuming would keep it; the split is still the checkpoint's (no --partition), on a machine where the tuner would not split.
env OMP_NUM_THREADS=2 DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "$BIN" "$SNAP" "$TMP/a" "${A[@]}" --resume ${t1:+--max-concurrent "$t1"} \
    > "$TMP/a2.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ "$rc2" = 0 ] && [ -n "$split" ] && [ "$split" != "1,1,1" ] && [ -n "$t1" ] \
    && grep -q "AUTO-TUNE: .* -> --partition" "$TMP/aref.log" \
    && grep -q -- "--resume: the partition split \[.*\] is the checkpoint's" "$TMP/a2.log" \
    && [ "$(splitof "$TMP/a2.log")" = "$split" ] && grep -q -- "--resume: 5 of " "$TMP/a2.log" \
    && [ ! -e "$TMP/a.runlog.checkpoint" ]
check "D9 the auto-tuner's split [${split}] was adopted from the checkpoint, the resume finished and removed it" $?
same "$TMP/aref" "$TMP/a"; check "D9 every output byte-identical to the uninterrupted auto-tuned run (resumed at its ${t1:-?} thread(s))" $?

echo ">> D10: outputs that could not be put in place keep the checkpoint"
mkdir -p "$TMP/cm.den"; : > "$TMP/cm.den/blocker"   # a folder where the '.den' output goes: its rename fails
"${RUN[@]}" "$TMP/cm" "${C[@]}" "${MC[@]}" > "$TMP/cm.log" 2>&1; rc=$?
[ "$rc" = 3 ] && grep -q "could not be put in place" "$TMP/cm.log" && [ -s "$TMP/cm.runlog.checkpoint/checkpoint" ]
check "D10 exit $rc (renames failed): the checkpoint stays for a resume" $?
rm -rf "$TMP/cm.den" "$TMP/cm.runlog.checkpoint" "$TMP"/cm.*.part.*

echo ">> D11: scripts/run_dtfe.sh passes --resume when a checkpoint exists"
# a copy of the driver beside a stand-in binary that only says how it was called (the driver finds its binary one
# level above its own folder), on a fake data root: no real run, deterministic
DREPO="$TMP/drv_repo"; DDATA="$TMP/drv_data"
mkdir -p "$DREPO/scripts" "$DDATA/SIM/snapdir_099"
cp scripts/run_dtfe.sh scripts/config.sh "$DREPO/scripts/"
printf '#!/usr/bin/env bash\necho "RUNNING: $0 $*"\necho "total wall time : 1 s"\nexit 0\n' > "$DREPO/DTFE"
chmod +x "$DREPO/DTFE"
: > "$DDATA/SIM/snapdir_099/combined_099.hdf5"
drive() {   # <log> [VAR=value ...]: the copied driver on SIM 99; echoes its exit status
    local log="$1"; shift
    local rc=0
    env DTFE_DATA_ROOT="$DDATA" DTFE_SPACE_CHECK=0 "$@" "$DREPO/scripts/run_dtfe.sh" -s SIM -g 16 99 > "$log" 2>&1 || rc=$?
    echo "$rc"
}
DCK="$DDATA/SIM/snapdir_099/output.runlog.checkpoint"
rc=$(drive "$TMP/d_plain.log")
[ "$rc" = 0 ] && grep -aq '^RUNNING: ' "$TMP/d_plain.log" && ! grep -aqE '^RUNNING: .*--resume' "$TMP/d_plain.log"
check "D11 no checkpoint: no --resume (exit $rc)" $?
mkdir -p "$DCK"; : > "$DCK/checkpoint"
rc=$(drive "$TMP/d_resume.log")
[ "$rc" = 0 ] && grep -aqE '^RUNNING: .*--resume' "$TMP/d_resume.log" \
    && grep -aq "Resuming: a killed run's checkpoint is in .*output\.runlog\.checkpoint" "$TMP/d_resume.log"
check "D11 a checkpoint beside the outputs: --resume is passed and the driver says so (exit $rc)" $?
rc=$(drive "$TMP/d_noresume.log" DTFE_RESUME=0)
[ "$rc" = 0 ] && ! grep -aqE '^RUNNING: .*--resume' "$TMP/d_noresume.log" \
    && grep -aq 'DTFE_RESUME=0, so this run starts over' "$TMP/d_noresume.log"
check "D11 DTFE_RESUME=0: no --resume, and the driver says the run starts over (exit $rc)" $?
rm -f "$DCK/checkpoint"; : > "$DCK/checkpoint.tmp.99999"
rc=$(drive "$TMP/d_emptyck.log")
! grep -aqE '(^RUNNING: .*--resume|Resuming)' "$TMP/d_emptyck.log"
check "D11 a checkpoint folder without a 'checkpoint' file: no --resume (exit $rc)" $?
rm -rf "$DCK"
DSCR="$TMP/drv_scratch"; mkdir -p "$DSCR"
DSCK="$(source scripts/config.sh && ps_checkpoint_dir "$DDATA/SIM/snapdir_099/output" "$DSCR")"
mkdir -p "$DSCK"; : > "$DSCK/checkpoint"
rc=$(drive "$TMP/d_scr.log" SCRATCH_DIR="$DSCR")
[ "$rc" = 0 ] && grep -aqE '^RUNNING: .*--resume' "$TMP/d_scr.log"
check "D11 SCRATCH_DIR: the checkpoint in the scratch folder makes the run pass --resume (exit $rc)" $?

echo ">> D12: a killed --gpu --exact-average run"
GPU=0
for m in metal cudaemu cuda hip; do [ -f "$OBJ/.gpu_mode_$m" ] && GPU=1; done
if [ "$GPU" != 1 ]; then
    echo "  SKIP D12: $OBJ carries no GPU stamp (a METAL=1 / CUDA=1 / HIP=1 / CUDAEMU=1 build runs it)"
else
    G=("${C[@]}" "${MC[@]}" --gpu --exact-average)
    "${RUN[@]}" "$TMP/gref" "${G[@]}" > "$TMP/gref.log" 2>&1 || { echo "FAIL: gref (see $TMP/gref.log)"; exit 1; }
    DTFE_TEST_EXIT_AFTER_PARTITIONS=12 "${RUN[@]}" "$TMP/g" "${G[@]}" > "$TMP/g1.log" 2>&1; rc1=$?
    "${RUN[@]}" "$TMP/g" "${G[@]}" --resume > "$TMP/g2.log" 2>&1; rc2=$?
    [ "$rc1" = 137 ] && [ "$rc2" = 0 ] && grep -q -- "--resume: 12 of 27" "$TMP/g2.log" && grep -q " GPU (" "$TMP/g2.log" \
        && ! grep -q "using the CPU interpolation" "$TMP/gref.log" "$TMP/g1.log" "$TMP/g2.log" && [ ! -e "$TMP/g.runlog.checkpoint" ]
    check "D12 killed after 12 GPU partitions, resumed the other 15 on the GPU (no CPU fallback)" $?
    cmp -s "$TMP/gref.den" "$TMP/g.den" && cmp -s "$TMP/gref.vel" "$TMP/g.vel"
    check "D12 the unaveraged (CPU) grids byte-identical" $?
    "$PY" - "$TMP/gref" "$TMP/g" <<'PYEOF'
import numpy as np, os, sys
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))
worst = 0.
for e in ("a_den", "a_vel", "a_velGrad"):
    a, b = (np.fromfile(f"{p}.{e}", dtype=REAL).astype(np.float64) for p in sys.argv[1:3])
    if a.size != b.size or not a.size: sys.exit(1)
    d = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))
    worst = max(worst, d)
    print(f"    .{e}: peak-relative difference {d:.1e}")
sys.exit(0 if worst < 1e-5 else 1)
PYEOF
    check "D12 the '_a' grids within 1e-5 of their peak" $?
    # the exact GPU sums (2026-10-13): a resumed GPU run is byte-identical as a CPU one is, unless the float atomics
    # are asked back (the integrator's check of the GPU and resume patches together)
    if [ "${DTFE_METAL_FLOAT_GRIDS:-0}" = 1 ] || [ "${DTFE_GPU_FLOAT_GRIDS:-0}" = 1 ]; then
        echo "  SKIP D12 byte identity: the float atomics were asked for"
    else
        cmp -s "$TMP/gref.a_den" "$TMP/g.a_den" && cmp -s "$TMP/gref.a_vel" "$TMP/g.a_vel" \
            && cmp -s "$TMP/gref.a_velGrad" "$TMP/g.a_velGrad"
        check "D12 the '_a' grids byte-identical (exact GPU sums)" $?
    fi
fi

echo ">> D13: a killed --sample-points plane resumes (its point answers checkpointed per partition)"
"$PY" - "$TMP" <<'PYEOF' || { echo "FAIL: D13 points"; exit 1; }
import sys, numpy as np
tmp = sys.argv[1]
c = (np.arange(64) + 0.5) * 100. / 64          # the demo box is 100 Mpc/h (100000 kpc/h in the Gadget header)
X, Y = np.meshgrid(c, c, indexing="ij")
np.stack([X.ravel(), Y.ravel(), np.full(X.size, 50.3)], 1).astype(np.float64).tofile(f"{tmp}/plane.bin")
np.stack([X.ravel(), Y.ravel(), np.full(X.size, 31.7)], 1).astype(np.float64).tofile(f"{tmp}/plane_other.bin")
PYEOF
P=("${C[@]}" "${MC[@]}" --sample-points "$TMP/plane.bin" --pts-vel-grad)
points_list() { ls "$1" 2>/dev/null | sed -n 's/^points\.\([0-9]*\)$/\1/p' | sort -n | tr '\n' ' '; }
merged_list() { grep -a -m1 '^merged ' "$1/checkpoint" | awk '{print $2}' | fold -w1 | grep -n 1 | cut -d: -f1 | awk '{print $1-1}' | tr '\n' ' '; }
"${RUN[@]}" "$TMP/pref" "${P[@]}" > "$TMP/pref.log" 2>&1 || { echo "FAIL: pref (see $TMP/pref.log)"; exit 1; }
grep -q "DTFE: checkpoints of the merged grids go to '$TMP/pref.runlog.checkpoint'" "$TMP/pref.log" \
    && grep -q "with the '--sample-points' stream records of every merged partition" "$TMP/pref.log" \
    && ! grep -q "no checkpoints" "$TMP/pref.log" && [ ! -e "$TMP/pref.runlog.checkpoint" ] \
    && [ -s "$TMP/pref.pts_den" ] && [ -s "$TMP/pref.pts_velGrad" ]
check "D13 a standard --sample-points run checkpoints (grids + point answers) and removes the checkpoint after its outputs" $?
DTFE_TEST_EXIT_AFTER_PARTITIONS=10 "${RUN[@]}" "$TMP/p" "${P[@]}" > "$TMP/p1.log" 2>&1; rc=$?
PK="$TMP/p.runlog.checkpoint"
n="$(merged_count "$PK" 2>/dev/null || echo 0)"
[ "$rc" = 137 ] && [ ! -e "$TMP/p.pts_den" ] && [ "${n:-0}" = 10 ] && [ "$(points_list "$PK")" = "$(merged_list "$PK")" ] \
    && [ -n "$(points_list "$PK")" ]
check "D13 killed after 10 partitions (exit $rc): a manifest of ${n:-?} of 27 and one points.<i> file for each ($(points_list "$PK"))" $?
cp -p "$TMP/plane.bin" "$TMP/plane.keep"; cp "$TMP/plane_other.bin" "$TMP/plane.bin"
"${RUN[@]}" "$TMP/p" "${P[@]}" --resume > "$TMP/p2.log" 2>&1; rc=$?
cp -p "$TMP/plane.keep" "$TMP/plane.bin"
[ "$rc" != 0 ] && grep -q "'--resume' refused" "$TMP/p2.log" && grep -q "sample-points" "$TMP/p2.log" \
    && [ -s "$PK/checkpoint" ] && [ ! -e "$TMP/p.pts_den" ]
check "D13 --resume with another point file under the same name refuses, naming the sample-points key line" $?
"${RUN[@]}" "$TMP/p" "${P[@]}" --resume > "$TMP/p3.log" 2>&1; rc=$?
[ "$rc" = 0 ] && grep -q -- "DTFE: --resume: 10 of 27 partitions are already in the checkpoint" "$TMP/p3.log" \
    && grep -q "stream records were read back" "$TMP/p3.log" \
    && [ "$(grep -c '<<< Interpolating the fields for partition' "$TMP/p3.log")" = 17 ] && [ ! -e "$PK" ]
check "D13 the resume read the 10 partitions' answers back and computed the other 17 (exit $rc); checkpoint removed" $?
same "$TMP/pref" "$TMP/p"; check "D13 every output (.pts_* and grids) byte-identical to the uninterrupted plane run" $?

echo "RESULT: $PASS passed, $FAILS failed"
[ "$FAILS" = 0 ] || exit 1
exit 0
