#!/usr/bin/env bash
# Our side of the comparison: PS-DTFE and standard DTFE on one snapshot, every run under
# /usr/bin/time -l (wall time + peak memory footprint), one JSON line per run in <outroot>.timing.jsonl.
#   run_ours.sh <repo> <snap.h5> <outroot> <slice.bin> <centres.bin> <grid> [skip...]
set -u
REPO="$1"; SNAP="$2"; OUT="$3"; SLICE="$4"; CENTRES="$5"; G="$6"; shift 6
SKIP=" $* "
PS="$REPO/PS-DTFE"; DT="$REPO/DTFE"
# OURS_COMMON overrides the shared options (e.g. a non-periodic --box region); PS_EXTRA is added to
# every PS-DTFE run (e.g. --ps-vertex-mass for TNG initial conditions)
read -r -a COMMON <<< "${OURS_COMMON:---input 105 --MpcUnit 1 --periodic --verbose 1}"
read -r -a PSX <<< "${PS_EXTRA:-}"
: >> "$OUT.timing.jsonl"

timed() {  # name, binary, args...
    local name="$1"; shift
    case "$SKIP" in *" $name "*) return;; esac
    local log="$OUT.$name.log"
    /usr/bin/time -l "$@" > "$log" 2>&1
    local rc=$?
    local wall=$(awk '/ real /{print $1}' "$log" | tail -1)
    local fp=$(awk '/peak memory footprint/{print $1}' "$log" | tail -1)
    local rss=$(awk '/maximum resident set size/{print $1}' "$log" | tail -1)
    printf '{"stage":"%s","rc":%s,"seconds":%s,"footprint_gb":%.3f,"maxrss_gb":%.3f}\n' \
        "$name" "$rc" "${wall:-0}" "$(echo "${fp:-0} / 1073741824" | bc -l)" "$(echo "${rss:-0} / 1073741824" | bc -l)" >> "$OUT.timing.jsonl"
    printf '  %-18s rc=%s  %8.2f s  footprint %6.2f GB\n' "$name" "$rc" "${wall:-0}" "$(echo "${fp:-0} / 1073741824" | bc -l)"
}

echo "ours  $(basename "$SNAP")  grid $G"
# PS-DTFE point evaluation on the slice (the '-g 16' grid is the documented points-only setting)
timed ps_slice      "$PS" "$SNAP" "$OUT.ps_slice" --grid 16 --field density "${COMMON[@]}" ${PSX[@]+"${PSX[@]}"} \
                    --sample-points "$SLICE" --ps-stream-density geometric
# PS-DTFE mass-conserving cell averages: the default sampled deposit (nSub=3), CPU and GPU, and the exact one
timed ps_grid_cpu   "$PS" "$SNAP" "$OUT.ps_grid_cpu" --grid "$G" --field density_a velocity_a "${COMMON[@]}" ${PSX[@]+"${PSX[@]}"}
timed ps_grid_gpu   "$PS" "$SNAP" "$OUT.ps_grid_gpu" --grid "$G" --field density_a velocity_a "${COMMON[@]}" ${PSX[@]+"${PSX[@]}"} --ps-gpu
timed ps_grid_exact "$PS" "$SNAP" "$OUT.ps_grid_exact" --grid "$G" --field density velocity "${COMMON[@]}" ${PSX[@]+"${PSX[@]}"} \
                    --ps-exact-deposit --ps-gpu
# PS-DTFE point values at the cell centres: exactly what CosmoDTFE's grid evaluation computes
timed ps_centres    "$PS" "$SNAP" "$OUT.ps_centres" --grid 16 --field density "${COMMON[@]}" ${PSX[@]+"${PSX[@]}"} \
                    --sample-points "$CENTRES" --ps-stream-density geometric
# standard (Eulerian) DTFE: the slice, and the grid at cell centres
timed dtfe_slice    "$DT" "$SNAP" "$OUT.dtfe_slice" --grid 16 --field density "${COMMON[@]}" --sample-points "$SLICE"
timed dtfe_grid     "$DT" "$SNAP" "$OUT.dtfe_grid" --grid "$G" --field density "${COMMON[@]}"
