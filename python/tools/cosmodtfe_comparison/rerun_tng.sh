#!/usr/bin/env bash
# re-measure our TNG phase-space stages after the region-face and alpha-shape fixes (non-periodic region)
C="$1"; R="$2"; cd "$C"
note() { echo "=== $(date +%H:%M:%S) $*   load: $(uptime | sed 's/.*averages: //')"; }
note "ours TNG sub-cube (PS stages)"
OURS_COMMON="--input 105 --MpcUnit 1 --verbose 1 --box 40 71 40 71 40 71" PS_EXTRA="--ps-vertex-mass" \
  bash run_ours.sh "$R" data/tng_sub.h5 out/otng data/tng_slice2048.bin data/tng_centres256.bin 256 dtfe_slice dtfe_grid
note DONE
