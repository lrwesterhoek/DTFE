#!/usr/bin/env bash
# the whole comparison, sequentially, on an otherwise idle machine
C="$1"; R="$2"; J="$3"
cd "$C"
JL=(julia -t 10 --project="$J" run_cosmo.jl)
note() { echo "=== $(date +%H:%M:%S) $*   load: $(uptime | sed 's/.*averages: //')"; }
for spec in "192 22" "128 21" "64 18"; do
  set -- $spec; n=$1; d=$2
  rm -f out/c${n}_dtfe.timing.jsonl out/c${n}_ps.timing.jsonl out/o${n}.timing.jsonl
  note "CosmoDTFE dtfe $n^3 depth $d"; "${JL[@]}" dtfe data/zel_$n.h5 out/c${n}_dtfe 2048 10.0 256 $d 0.05
  note "CosmoDTFE ps $n^3 depth $d";   "${JL[@]}" ps   data/zel_$n.h5 out/c${n}_ps   2048 10.0 256 $d 0.05
  note "ours $n^3";                    bash run_ours.sh "$R" data/zel_$n.h5 out/o$n data/slice2048.bin data/centres256.bin 256
done
NBAR=$(python3 -c "print(94196375 / 110.71744906997343**3)")
rm -f out/ctng_dtfe.timing.jsonl out/ctng_ps.timing.jsonl out/otng.timing.jsonl
note "CosmoDTFE dtfe TNG sub-cube";  "${JL[@]}" dtfe data/tng_sub.h5 out/ctng_dtfe 2048 55.5 256 22 0.0 40 71 0 $NBAR
note "CosmoDTFE ps TNG sub-cube";    "${JL[@]}" ps   data/tng_sub.h5 out/ctng_ps   2048 55.5 256 22 0.0 40 71 0 $NBAR
note "ours TNG sub-cube"
OURS_COMMON="--input 105 --MpcUnit 1 --verbose 1 --box 40 71 40 71 40 71" PS_EXTRA="--ps-vertex-mass" \
  bash run_ours.sh "$R" data/tng_sub.h5 out/otng data/tng_slice2048.bin data/tng_centres256.bin 256
note "DONE"
