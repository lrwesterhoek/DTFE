#!/usr/bin/env bash
#
# Download TNG snapshot files for a set of snapshots (redshifts).
#
# Companion to run_dtfe.sh: this fetches the raw snapshot HDF5 files from the
# TNG public data release; run_dtfe.sh then processes them with DTFE.
#
# Usage:
#   ./download_snapshots.sh -k API_KEY [-d DATA_DIR] [-s SIMULATION] [snapshot ...]
#
# Examples:
#   ./download_snapshots.sh -k 0123456789abcdef
#   ./download_snapshots.sh -k 0123456789abcdef -d /data/TNG50-4-Dark 1 2 3
#
set -uo pipefail

# ---- Defaults (override with flags / positional args) ---------------------
API_KEY=""
DATA_DIR="/Users/luukw/output/TNG50-4-Dark"
SIMULATION="TNG50-4-Dark"
INPUT_SUBDIR="snapdir"

# Snapshot numbers (each corresponds to a redshift) to download.
SNAPSHOTS=(1 2 3 4 5 8 13 17 21 25 33 40 50 67 72 78 84 91)

usage() {
    echo "Usage: $0 -k API_KEY [-d DATA_DIR] [-s SIMULATION] [snapshot ...]"
}

# ---- Parse arguments ------------------------------------------------------
while getopts "k:d:s:h" opt; do
    case "$opt" in
        k) API_KEY="$OPTARG" ;;
        d) DATA_DIR="$OPTARG" ;;
        s) SIMULATION="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# Any remaining positional arguments override the default snapshot list.
if [ "$#" -gt 0 ]; then
    SNAPSHOTS=("$@")
fi

if [ -z "${API_KEY}" ]; then
    echo "Error: an API key is required (-k API_KEY)." >&2
    usage
    exit 1
fi

echo "Downloading TNG snapshots..."
echo "Simulation:     ${SIMULATION}"
echo "Data directory: ${DATA_DIR}"
echo "Snapshots:      ${SNAPSHOTS[*]}"
echo ""

for i in "${SNAPSHOTS[@]}"; do
    # Format snapshot number with leading zeros (e.g. 1 -> 001)
    n_str=$(printf "%03d" "$i")
    input_dir="${DATA_DIR}/${INPUT_SUBDIR}_${n_str}"

    echo "Downloading snapshot ${n_str}..."

    # Create the destination directory if it doesn't exist
    mkdir -p "${input_dir}"

    if wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 \
            --content-disposition \
            --header="API-Key: ${API_KEY}" \
            -P "${input_dir}" \
            "http://www.tng-project.org/api/${SIMULATION}/files/snapshot-${i}/?format=api"; then
        echo "  Snapshot ${n_str} downloaded successfully"
    else
        echo "  Error downloading snapshot ${n_str}"
    fi
    echo ""
done

echo "Download complete!"
