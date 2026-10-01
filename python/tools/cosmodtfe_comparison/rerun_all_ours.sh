#!/usr/bin/env bash
cd "/private/tmp/claude-501/-Users-luukw-Library-Mobile-Documents-com-apple-CloudDocs-Documents-Rijksuniversiteit-Groningen-Bachelor-Physics-Year-3-Bachelor-Research-Project-PH-Fate-of-Cosmic-Voids-DTFE/3df24db9-82b6-41b9-859a-21379a680265/scratchpad/cmp"
note() { echo "=== $(date +%H:%M:%S) $*"; }
A=(data/zel_64.h5 --grid 16 --field density --input 105 --MpcUnit 1 --periodic --verbose 2 --sample-points data/slice2048.bin --ps-stream-density geometric)
note "variance: TBB triangulation time, 3 runs main + 2 runs scratch build"
for i in 1 2 3; do "/Users/luukw/Library/Mobile Documents/com~apple~CloudDocs/Documents/Rijksuniversiteit Groningen/Bachelor/Physics/Year 3/Bachelor Research Project PH/Fate of Cosmic Voids/DTFE/PS-DTFE" "${A[0]}" out/var_main "${A[@]:1}" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "Time:.*triangulation" | sed -E 's/.*Time: ([0-9.]+).*/main triangulation \1 s/'; done
for i in 1 2; do "/private/tmp/claude-501/-Users-luukw-Library-Mobile-Documents-com-apple-CloudDocs-Documents-Rijksuniversiteit-Groningen-Bachelor-Physics-Year-3-Bachelor-Research-Project-PH-Fate-of-Cosmic-Voids-DTFE/3df24db9-82b6-41b9-859a-21379a680265/scratchpad/tbbbuild/PS-DTFE" "${A[0]}" out/var_scr "${A[@]:1}" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "Time:.*triangulation" | sed -E 's/.*Time: ([0-9.]+).*/scratch triangulation \1 s/'; done
rm -f out/var_main.* out/var_scr.*
for n in 64 128 192; do
  note "ours $n^3 (all stages)"
  bash run_ours.sh "/Users/luukw/Library/Mobile Documents/com~apple~CloudDocs/Documents/Rijksuniversiteit Groningen/Bachelor/Physics/Year 3/Bachelor Research Project PH/Fate of Cosmic Voids/DTFE" data/zel_$n.h5 out/o$n data/slice2048.bin data/centres256.bin 256
done
note "ours TNG sub-cube (all stages)"
OURS_COMMON="--input 105 --MpcUnit 1 --verbose 1 --box 40 71 40 71 40 71" PS_EXTRA="--ps-vertex-mass" \
  bash run_ours.sh "/Users/luukw/Library/Mobile Documents/com~apple~CloudDocs/Documents/Rijksuniversiteit Groningen/Bachelor/Physics/Year 3/Bachelor Research Project PH/Fate of Cosmic Voids/DTFE" data/tng_sub.h5 out/otng data/tng_slice2048.bin data/tng_centres256.bin 256
note DONE
