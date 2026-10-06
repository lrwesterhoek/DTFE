#!/usr/bin/env bash
# The composite server's partition split (auto_tune.h autoTuneServe), chosen when '--serve' runs out of
# memory for one tessellation and no '--partition' is given:
#   A  nothing cached: the split that holds ~8 partitions in memory (finer than the coarsest that fits)
#   B  a complete split of THIS snapshot in the cache: reused, nothing rebuilt
#   C  a split cached for ANOTHER snapshot: ignored (the scan compares the snapshot's key lines)
#   D  a cached split too coarse for the memory budget: not reused, and said so
# A small set and a small DTFE_MEM_BUDGET_GB force the composite path; every cache is a fresh folder.
# Usage: tests/serve_split_check.sh [--no-build]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
source "${SCRIPT_DIR}/precision.sh"
BIN="${PS_BIN}"
precision_require "${BIN}"
if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    make PS-DTFE $(cat o_ps/.build_mode 2>/dev/null || true) >/dev/null
fi
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/serve_split"; rm -rf "$TMP"; mkdir -p "$TMP"
echo "============================================================"
echo " composite server split check"
echo "============================================================"
"$PY" tests/generate_ps_test_data.py --out "$TMP/a.hdf5" --n 24 --box 100 >/dev/null
"$PY" tests/generate_ps_test_data.py --out "$TMP/b.hdf5" --n 24 --box 100 --seed 7 >/dev/null
COMMON=( --periodic --MpcUnit 1 )
seed() {   # seed <cache> <snapshot> <n>: a grid run with --partition n n n writes that split's partitions
    "$BIN" "$2" "$TMP/g" --grid 16 "${COMMON[@]}" --field density --partition "$3" "$3" "$3" \
        --tessellation-cache "$1" > "$TMP/seed.log" 2>&1
}
serve() {  # serve <cache>: the auto split of a.hdf5's server; prints "<n> <reason> | <partitions built>"
    DTFE_MEM_BUDGET_GB=0.019 "$BIN" "$TMP/a.hdf5" "$TMP/srv" "${COMMON[@]}" --serve --tessellation-cache "$1" \
        < /dev/null > "$TMP/srv.log" 2>&1
    local line; line="$(grep -a -F 'AUTO-TUNE (serve)' "$TMP/srv.log")"
    local n; n="$(sed -n 's/.*composite of \([0-9]*\)^3 partitions.*/\1/p' <<<"$line")"
    local why; why="$(sed -n 's/.*partitions (\([^;]*\);.*/\1/p' <<<"$line")"
    echo "$n|$why|$(grep -a -c 'built,' "$TMP/srv.log")"
}
fails=0
check() { if [ "$1" = "1" ]; then echo "   PASS  $2"; else echo "   FAIL  $2"; fails=$((fails+1)); fi; }

mkdir -p "$TMP/c_empty"
IFS='|' read -r nA whyA builtA <<<"$(serve "$TMP/c_empty")"
check "$([ "${nA:-0}" -ge 3 ] && [[ "$whyA" == nothing\ cached* ]] && echo 1)" \
      "A nothing cached: a split of ${nA}^3 (${whyA}), ${builtA} partitions built"

mkdir -p "$TMP/c_same"; seed "$TMP/c_same" "$TMP/a.hdf5" 3
IFS='|' read -r nB whyB builtB <<<"$(serve "$TMP/c_same")"
check "$([ "$nB" = 3 ] && [ "$builtB" = 0 ] && [[ "$whyB" == *fully\ in\ the\ tessellation\ cache* ]] && echo 1)" \
      "B this snapshot's 3^3 set cached: reused (${nB}^3, ${builtB} partitions built)"

mkdir -p "$TMP/c_other"; seed "$TMP/c_other" "$TMP/b.hdf5" 3
IFS='|' read -r nC whyC builtC <<<"$(serve "$TMP/c_other")"
check "$([ "$nC" = "$nA" ] && [[ "$whyC" == nothing\ cached* ]] && echo 1)" \
      "C another snapshot's 3^3 set: ignored (${nC}^3, ${whyC})"

mkdir -p "$TMP/c_coarse"; seed "$TMP/c_coarse" "$TMP/a.hdf5" 2
IFS='|' read -r nD whyD builtD <<<"$(serve "$TMP/c_coarse")"
check "$([ "$nD" = "$nA" ] && [[ "$whyD" == *too\ coarse* ]] && echo 1)" \
      "D this snapshot's 2^3 set, too coarse for the budget: not reused (${nD}^3, ${whyD})"

echo "------------------------------------------------------------"
if [ "$fails" -gt 0 ]; then echo "RESULT: FAIL ($fails)"; exit 1; fi
echo "RESULT: PASS  (the composite server reuses a cached split of the same snapshot, else holds ~8 partitions)"
