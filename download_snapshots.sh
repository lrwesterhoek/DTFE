#!/usr/bin/env bash
#
# Download TNG data products for a set of snapshots (redshifts).
#
# Two modes:
#   snapshots (default) : raw particle snapshot chunks -> <DATA_DIR>/snapdir_NNN/
#   group catalogs (-c) : FoF/Subfind fof_subhalo_tab chunks -> <DATA_DIR>/groups_NNN/
#                         (needed by the merger-tree tracker for subhalo selection and
#                          particle-based shape/orientation; trees themselves are separate)
#
# Usage:
#   ./download_snapshots.sh [-k API_KEY] [-c] [-d DATA_DIR] [-s SIMULATION] [snapshot ...]
#
# Examples:
#   ./download_snapshots.sh -c -s TNG50-3-Dark            # groupcats for the default ladder
#   ./download_snapshots.sh -c -s TNG300-3-Dark 99 50     # groupcats, snapshots 99+50 only
#   ./download_snapshots.sh -k 0123456789abcdef 1 2 3     # raw snapshots
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ---- Defaults (override with flags / positional args) ---------------------
# API key resolution: -k flag > TNG_API_KEY env > ~/.tng_api_key file. The key file lives
# OUTSIDE the repository so it can never be committed; create it once with:
#   echo "<your-key>" > ~/.tng_api_key && chmod 600 ~/.tng_api_key
API_KEY="${TNG_API_KEY:-}"
if [ -z "${API_KEY}" ] && [ -r "${HOME}/.tng_api_key" ]; then
    API_KEY="$(head -n1 "${HOME}/.tng_api_key" | tr -d '[:space:]')"
fi
DATA_DIR=""                       # default: $DATA_ROOT/<SIMULATION> (config.sh)
MODE="snapshot"                   # 'snapshot' or 'groupcat' (-c)

# Snapshot numbers (each corresponds to a redshift) to download; overrides the
# short default list from config.sh with the full redshift ladder.
SNAPSHOTS=(1 2 3 4 5 8 13 17 21 25 33 40 50 67 72 78 84 91 99)

usage() {
    echo "Usage: $0 [-k API_KEY] [-c|-t] [-d DATA_DIR] [-s SIMULATION] [snapshot ...]"
    echo "  -c  download FoF/Subfind group catalogs instead of raw snapshots"
    echo "  -t  download the SubLink merger trees (whole simulation; snapshot list ignored)"
    echo "  API key: -k flag, TNG_API_KEY env, or ~/.tng_api_key file (recommended)."
}

# ---- Parse arguments ------------------------------------------------------
while getopts "k:d:s:cth" opt; do
    case "$opt" in
        k) API_KEY="$OPTARG" ;;
        d) DATA_DIR="$OPTARG" ;;
        s) SIMULATION="$OPTARG" ;;
        c) MODE="groupcat" ;;
        t) MODE="tree" ;;
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
    echo "Error: an API key is required (-k, TNG_API_KEY, or ~/.tng_api_key)." >&2
    usage
    exit 1
fi
[ -z "${DATA_DIR}" ] && DATA_DIR="${DATA_ROOT}/${SIMULATION}"

echo "Downloading TNG ${MODE} files..."
echo "Simulation:     ${SIMULATION}"
echo "Data directory: ${DATA_DIR}"
[ "${MODE}" != "tree" ] && echo "Snapshots:      ${SNAPSHOTS[*]}"
echo ""

# SubLink merger trees are whole-simulation files (tree_extended.X.hdf5), not per-snapshot.
if [ "${MODE}" = "tree" ]; then
    dest_dir="${DATA_DIR}/Merger Trees"       # where dtfelib.trees.TreeSet looks
    echo "Downloading sublink trees -> ${dest_dir} ..."
    mkdir -p "${dest_dir}"
    if wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 \
            --content-disposition \
            --header="API-Key: ${API_KEY}" \
            -P "${dest_dir}" \
            "http://www.tng-project.org/api/${SIMULATION}/files/sublink/?format=api"; then
        echo "  sublink trees downloaded successfully"
    else
        echo "  Error downloading sublink trees"
    fi
    echo ""
    echo "Download complete!"
    exit 0
fi

for i in "${SNAPSHOTS[@]}"; do
    # Format snapshot number with leading zeros (e.g. 1 -> 001)
    n_str=$(printf "%03d" "$i")
    if [ "${MODE}" = "groupcat" ]; then
        dest_dir="${DATA_DIR}/groups_${n_str}"      # fof_subhalo_tab_NNN.X.hdf5 chunks
        endpoint="groupcat-${i}"
    else
        dest_dir="${DATA_DIR}/${INPUT_SUBDIR}_${n_str}"
        endpoint="snapshot-${i}"
    fi

    echo "Downloading ${endpoint} -> ${dest_dir} ..."
    mkdir -p "${dest_dir}"

    if wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 \
            --content-disposition \
            --header="API-Key: ${API_KEY}" \
            -P "${dest_dir}" \
            "http://www.tng-project.org/api/${SIMULATION}/files/${endpoint}/?format=api"; then
        echo "  ${endpoint} downloaded successfully"
    else
        echo "  Error downloading ${endpoint}"
    fi
    echo ""
done

echo "Download complete!"
