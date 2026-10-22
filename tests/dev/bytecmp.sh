#!/usr/bin/env bash
# Byte-for-byte comparison of the outputs of two sets of binaries: a reference set against this checkout's own
# (./DTFE, ./PS-DTFE, their -double and -2d builds). The gate for a change that must not move any output (a
# refactor, a cleanup): 10 runs on two synthetic crossed-wave snapshots -- standard DTFE and PS-DTFE, single, double
# and 2D, CPU and GPU, one domain and '--partition 2 2 2', every derived and web field -- and every output file
# compared with cmp. Logs are kept but not compared (they carry times and memory).
#
# The reference set is any folder holding the six binaries, e.g. a copy taken before the change:
#     mkdir -p /tmp/ref && cp DTFE DTFE-double DTFE-2d PS-DTFE PS-DTFE-double PS-DTFE-2d /tmp/ref/
# (build all three pairs first: make DTFE PS-DTFE METAL=1; ... METAL=1 DOUBLE=1; ... DIM=2). The GPU runs need a
# METAL=1 (or CUDA/HIP) build on both sides; a run that fell back to the CPU is listed at the end.
# Usage: tests/dev/bytecmp.sh <reference binaries dir> <output dir>      (exit 0 = every file identical)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ $# -eq 2 ] || { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 2; }
REF="$(cd "$1" && pwd)" || { echo "no reference folder $1"; exit 2; }
OUT="$2"; mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)"
cd "$ROOT" || exit 2
source tests/lib.sh   # PY: a python with a working h5py
for b in DTFE DTFE-double DTFE-2d PS-DTFE PS-DTFE-double PS-DTFE-2d; do
    [ -x "$REF/$b" ] || { echo "missing reference binary $REF/$b"; exit 2; }
    [ -x "$ROOT/$b" ] || { echo "missing binary $ROOT/$b (build all three pairs)"; exit 2; }
done
"$PY" tests/generate_ps_test_data.py --out "$OUT/w48.hdf5" --n 48 --box 100 --amplitude-factor 1.8 --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null
"$PY" tests/generate_ps_test_data.py --out "$OUT/w2d.hdf5" --n 64 --box 100 --amplitude-factor 1.8 --jitter-frac 0.02 --seed 7 --crossed-waves --dim 2 >/dev/null
C=(--periodic --input 105 --MpcUnit 1 --verbose 1)
F3=(--field density_a velocity_a gradient_a divergence_a shear_a vorticity_a tweb_a vweb_a)
F2=(--field density_a velocity_a gradient_a divergence_a vorticity_a)
runs=(
  "dt:DTFE:w48:--grid 64 ${F3[*]}"
  "dtp:DTFE:w48:--grid 64 --partition 2 2 2 ${F3[*]}"
  "dtg:DTFE:w48:--grid 64 --gpu ${F3[*]}"
  "dtd:DTFE-double:w48:--grid 64 ${F3[*]}"
  "ps:PS-DTFE:w48:--grid 64 ${F3[*]} --ps-caustics"
  "psp:PS-DTFE:w48:--grid 64 --partition 2 2 2 ${F3[*]}"
  "psg:PS-DTFE:w48:--grid 64 --ps-gpu ${F3[*]} --ps-caustics"
  "psd:PS-DTFE-double:w48:--grid 64 ${F3[*]}"
  "dt2:DTFE-2d:w2d:--grid 128 ${F2[*]}"
  "ps2:PS-DTFE-2d:w2d:--grid 128 ${F2[*]}"
)
n=0; d=0
for r in "${runs[@]}"; do
  IFS=: read -r tag bin snap args <<< "$r"
  for side in ref new; do
    B="$ROOT/$bin"; [ $side = ref ] && B="$REF/$bin"
    mkdir -p "$OUT/$side"
    # shellcheck disable=SC2086   # $args: the run's options, word-split on purpose
    "$B" "$OUT/$snap.hdf5" "$OUT/$side/$tag" "${C[@]}" $args > "$OUT/$side/$tag.log" 2>&1 || echo "  RUN FAILED: $side $tag (rc $?)"
  done
  for f in "$OUT/ref/$tag".*; do
    case "$f" in *.log) continue;; esac
    g="$OUT/new/${f#"$OUT/ref/"}"; n=$((n+1))
    cmp -s "$f" "$g" || { d=$((d+1)); echo "  DIFFERS: ${f#"$OUT/ref/"}"; }
  done
  [ "$(ls "$OUT/ref/$tag".* | wc -l)" = "$(ls "$OUT/new/$tag".* | wc -l)" ] || { d=$((d+1)); echo "  FILE SETS DIFFER: $tag"; }
done
grep -l -E 'using the CPU deposit|deposit unavailable|gave up|falling back' "$OUT"/new/*g.log "$OUT"/ref/*g.log 2>/dev/null | sed 's/^/  GPU FALLBACK in /'
echo "BYTECMP: $n output files compared, $d differ"
[ "$d" = 0 ]
