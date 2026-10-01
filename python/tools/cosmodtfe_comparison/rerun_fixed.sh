#!/usr/bin/env bash
# re-measure the four stages the fixes touch (point values, slice, CPU deposit, exact GPU deposit)
C="$1"; R="$2"; cd "$C"
note() { echo "=== $(date +%H:%M:%S) $*   load: $(uptime | sed 's/.*averages: //')"; }
SKIP="ps_grid_gpu dtfe_slice dtfe_grid"
for n in 64 128 192; do
  note "ours $n^3"
  bash run_ours.sh "$R" data/zel_$n.h5 out/o$n data/slice2048.bin data/centres256.bin 256 $SKIP
done
note "ours TNG sub-cube"
OURS_COMMON="--input 105 --MpcUnit 1 --verbose 1 --box 40 71 40 71 40 71" PS_EXTRA="--ps-vertex-mass" \
  bash run_ours.sh "$R" data/tng_sub.h5 out/otng data/tng_slice2048.bin data/tng_centres256.bin 256 $SKIP
note DONE
