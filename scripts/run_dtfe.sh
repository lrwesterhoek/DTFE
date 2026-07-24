#!/usr/bin/env bash
#
# Batch-run standard DTFE over a set of TNG snapshots (redshifts).
#
# Companion to download_snapshots.sh, which fetches the snapshot files first.
# Shared defaults (DATA_ROOT, SIMULATION, SNAPSHOTS, GRID_SIZE, PADDING) live in config.sh.
#
# Usage:
#   ./run_dtfe.sh [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-m] [snapshot ...]
#     -m   run the '_a' interpolation on the Apple GPU (same as DTFE_METAL=1; needs 'make DTFE METAL=1')
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # the binaries and Makefile live one level up
source "${SCRIPT_DIR}/config.sh"

# ---- Script-specific configuration ----------------------------------------
PARTITION="${PARTITION:-}"       # EMPTY (default) = the binary AUTO-TUNES the split from the particle
                                 # count, grid, fields and available RAM/cores (see the AUTO-TUNE line
                                 # it prints). Set to override, e.g. PARTITION="2 2 2".
MAX_CONCURRENT="${MAX_CONCURRENT:-}"  # cap on concurrent triangulations to bound peak RAM (0 = all cores).
                                 # EMPTY (default) = auto-tuned together with the partition split.
DTFE_METAL="${DTFE_METAL:-0}"  # 1 = run the '_a' interpolation on the Apple GPU (--gpu; needs 'make DTFE METAL=1')

DATA_DIR=""                # default: $DATA_ROOT/$SIMULATION (config.sh); override with -d
OUTPUT_SUBDIR="output"

FIELDS="density_a velocity_a gradient_a divergence_a shear_a vorticity_a"

usage() { echo "Usage: $0 [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-m] [snapshot ...]"; }

# ---- Parse arguments ------------------------------------------------------
while getopts "d:s:g:mh" opt; do
    case "$opt" in
        d) DATA_DIR="$OPTARG" ;;
        s) SIMULATION="$OPTARG" ;;
        g) GRID_SIZE="$OPTARG" ;;
        m) DTFE_METAL=1 ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# Any remaining positional arguments override the default snapshot list.
if [ "$#" -gt 0 ]; then
    SNAPSHOTS=("$@")
fi

[ -z "${DATA_DIR}" ] && DATA_DIR="${DATA_ROOT}/${SIMULATION}"

cd "$REPO_ROOT" || exit 1

echo "Starting DTFE processing..."
echo "Data directory: ${DATA_DIR}"
echo "Grid size: ${GRID_SIZE}"
echo ""

for i in "${SNAPSHOTS[@]}"; do
    n_str=$(printf "%03d" "$i")

    input_dir="${DATA_DIR}/${INPUT_SUBDIR}_${n_str}"
    input_file="${input_dir}/combined_${n_str}.hdf5"
    # output prefix: files land as <snapdir>/output.a_den etc. (the 'dtfe' convention in dtfelib)
    output_root="${input_dir}/output"

    echo "Processing snapshot ${n_str}..."

    if [ ! -f "${input_file}" ]; then
        echo "  Warning: Input file not found: ${input_file}"
        echo "  Skipping snapshot ${n_str}"
        continue
    fi

    # GPU interpolation toggle (-m / DTFE_METAL=1); ignored with a warning on non-METAL builds.
    metal_args=()
    [ "${DTFE_METAL}" = "1" ] && metal_args=(--gpu)

    # pass --partition/--max-concurrent only when set; otherwise the binary auto-tunes them
    part_args=()
    [ -n "${PARTITION}" ] && part_args+=(--partition ${PARTITION})
    [ -n "${MAX_CONCURRENT}" ] && part_args+=(--max-concurrent "${MAX_CONCURRENT}")

    echo "  Running DTFE on ${input_file}..."
    ./DTFE "${input_file}" "${output_root}" \
        --grid ${GRID_SIZE} \
        --padding ${PADDING} \
        --periodic \
        ${part_args[@]+"${part_args[@]}"} \
        --field ${FIELDS} \
        ${metal_args[@]+"${metal_args[@]}"}

    if [ $? -eq 0 ]; then
        echo "  Snapshot ${n_str} processed successfully"
    else
        echo "  Error processing snapshot ${n_str}"
    fi
    echo ""
done

echo "DTFE processing complete!"
