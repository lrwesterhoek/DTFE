#!/usr/bin/env bash
# The partition loop's run-time features (2026-10-11), on small synthetic snapshots:
#   R1  '--ordered-merge': two runs at --max-concurrent 4 merge in index order and write byte-identical grids
#   R2  the partition look-ahead (forced on the CPU deposit with DTFE_PS_OVERLAP=1): every output byte-identical
#       to '--ps-no-overlap', and the log says it ran
#   R3  GPU builds only: the look-ahead of a '--ps-gpu' run at --max-concurrent 1 (on by default) against
#       '--ps-no-overlap': the exactly summed grids (.den, .a_den, .streams, ...) byte-identical, the float-atomic
#       velocity within 1e-5 of its peak (two GPU runs differ there anyway); SKIP on a CPU build
#   R4  crash-safe outputs: a finished run leaves no '<file>.part.<pid>' and no progress sidecar; a stale
#       '.part.<pid>' of a dead pid is removed at the start, one of a live pid is kept, and one whose final file
#       is missing is kept with its mv (it may be the only copy, 2026-10-11); a run whose output
#       folder is missing writes the whole set to $TMPDIR/dtfe-rescue/<folder>_<name>, byte-identical to a
#       normal run, says so and exits 3
#   R5  the progress sidecar '<output>.runlog.progress' while a run goes: the contract's keys, phase
#       'partitions', done counting up to total
# Usage: tests/ps_runtime_check.sh [--no-build]   (--no-build is accepted and ignored: the binaries are used as built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
BIN="$(cd "$(dirname "${PS_BIN}")" && pwd)/$(basename "${PS_BIN}")"
TMP="${DTFE_TEST_TMP}/ps_runtime_check"
rm -rf "$TMP"; mkdir -p "$TMP"
SNAP="$TMP/waves.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 40 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
PASS=0; FAILS=0
check() { if [ "$2" = 0 ]; then PASS=$((PASS+1)); echo "  PASS $1"; else FAILS=$((FAILS+1)); echo "  FAIL $1"; fi; }
# same <a> <b>: every output of run a has a byte-identical twin in run b (and there is at least one)
same() {
    local n=0 f
    for f in "$1".*; do
        case "$f" in *.log|*.runlog*|*.part.*) continue ;; esac
        cmp -s "$f" "$2${f#"$1"}" || { echo "    differs: ${f#"$1"}"; return 1; }
        n=$((n+1))
    done
    [ "$n" -gt 0 ]
}
C=(--grid 48 --periodic --input 105 --MpcUnit 1 --verbose 2 --field density velocity density_a)

echo ">> R1: --ordered-merge, --partition 3 3 3 --max-concurrent 4, twice"
"$BIN" --help 2>&1 | grep -q -- "--ordered-merge"; check "R1 '--ordered-merge' is in the help" $?
for r in o1 o2; do OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/$r" "${C[@]}" --partition 3 3 3 --max-concurrent 4 --ordered-merge \
    > "$TMP/$r.log" 2>&1 || { echo "FAIL: $r (see $TMP/$r.log)"; exit 1; }; done
grep -q "partition merge order  : partition index" "$TMP/o1.log"; check "R1 the run says it merges in index order" $?
same "$TMP/o1" "$TMP/o2"; check "R1 two --ordered-merge runs at --max-concurrent 4: byte-identical grids" $?

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
    for e in den a_den streams a_streams hidden_streams a_hidden_streams; do
        cmp -s "$TMP/gn.$e" "$TMP/gl.$e" || { echo "    differs: .$e"; ok=1; }
    done
    check "R3 GPU look-ahead vs --ps-no-overlap: the exactly summed grids byte-identical" $ok
    "$PY" - "$TMP/gn.vel" "$TMP/gl.vel" <<'PYEOF'
import numpy as np, os, sys
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))
a, b = (np.fromfile(p, dtype=REAL).astype(np.float64) for p in sys.argv[1:3])
peak = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)) if a.size == b.size and a.size else 1.
print(f"    .vel peak-relative difference {peak:.1e}")
sys.exit(0 if peak < 1e-5 else 1)
PYEOF
    check "R3 GPU look-ahead vs --ps-no-overlap: velocity within 1e-5 of its peak (float atomics)" $?
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
keys = {"pid", "phase", "done", "total", "elapsed_s", "eta_s", "rss_gb", "gpu_retries", "updated"}
parts = [s for s in snaps if s.get("phase") == "partitions"]
ok = (bool(snaps) and all(keys <= set(s) for s in snaps) and all(s["pid"] == int(sys.argv[2]) for s in snaps)
      and len(parts) >= 2 and all(p["total"] == 64 for p in parts)
      and [p["done"] for p in parts] == sorted(p["done"] for p in parts) and max(p["done"] for p in parts) >= 32
      and all(p["rss_gb"] > 0 for p in parts) and not os.path.exists(sys.argv[3]))
print(f"    {len(snaps)} snapshots read, phases {sorted({s.get('phase') for s in snaps})}, "
      f"done up to {max([p['done'] for p in parts] or [0])}/64")
sys.exit(0 if ok else 1)
PYEOF
check "R5 the sidecar carries the contract's keys, counts the partitions up and is gone at the end" $?

echo "RESULT: $PASS passed, $FAILS failed"
[ "$FAILS" = 0 ] || exit 1
exit 0
