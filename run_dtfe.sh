#!/usr/bin/env bash
#
# Batch-run standard DTFE over a set of TNG snapshots (redshifts).
#
# Companion to download_snapshots.sh, which fetches the snapshot files first.
#
# Usage:
#   ./run_dtfe.sh [-d DATA_DIR] [-g GRID_SIZE] [snapshot ...]
#
set -uo pipefail

# ---- Configuration --------------------------------------------------------
# Snapshot numbers (each corresponds to a redshift) to process.
SNAPSHOTS=(1 2 3 4 5 8 13 17 21 25 33 40 50 67 72 78 84 91)

GRID_SIZE=512
PADDING=20
PARTITION="2 2 2"
MAX_CONCURRENT=0           # cap on concurrent triangulations to bound peak RAM (0 = all
                           # cores). Fits 64 GB at 512/2 2 2; set 2-4 with less RAM.

DATA_DIR="/Users/luukw/output/TNG50-4-Dark"
INPUT_SUBDIR="snapdir"
OUTPUT_SUBDIR="output"

FIELDS="density_a velocity_a gradient_a divergence_a shear_a vorticity_a"

# ---- Parse arguments ------------------------------------------------------
while getopts "d:g:h" opt; do
    case "$opt" in
        d) DATA_DIR="$OPTARG" ;;
        g) GRID_SIZE="$OPTARG" ;;
        h) echo "Usage: $0 [-d DATA_DIR] [-g GRID_SIZE] [snapshot ...]"; exit 0 ;;
        *) echo "Usage: $0 [-d DATA_DIR] [-g GRID_SIZE] [snapshot ...]"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# Any remaining positional arguments override the default snapshot list.
if [ "$#" -gt 0 ]; then
    SNAPSHOTS=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "Starting DTFE processing..."
echo "Data directory: ${DATA_DIR}"
echo "Grid size: ${GRID_SIZE}"
echo ""

for i in "${SNAPSHOTS[@]}"; do
    # Format snapshot number with leading zeros
    n_str=$(printf "%03d" "$i")

    # Define paths
    input_dir="${DATA_DIR}/${INPUT_SUBDIR}_${n_str}"
    input_file="${input_dir}/combined_${n_str}.hdf5"
    output_dir="${input_dir}/snapdir_${n_str}"

    echo "Processing snapshot ${n_str}..."

    # Create output directory if it doesn't exist
    mkdir -p "${output_dir}"

    # Check if input file exists
    if [ ! -f "${input_file}" ]; then
        echo "  Warning: Input file not found: ${input_file}"
        echo "  Skipping snapshot ${n_str}"
        continue
    fi

    # Run DTFE
    echo "  Running DTFE on ${input_file}..."
    ./DTFE "${input_file}" "${output_dir}" \
        --grid ${GRID_SIZE} \
        --padding ${PADDING} \
        --periodic \
        --partition ${PARTITION} \
        --max-concurrent ${MAX_CONCURRENT} \
        --field ${FIELDS}

    if [ $? -eq 0 ]; then
        echo "  Snapshot ${n_str} processed successfully"
    else
        echo "  Error processing snapshot ${n_str}"
    fi
    echo ""
done

echo "DTFE processing complete!"
