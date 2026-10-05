#!/usr/bin/env bash
#
# Download TNG data products for a set of snapshots (redshifts).
#
# Modes:
#   snapshots (default) : raw particle snapshot chunks -> <DATA_DIR>/snapdir_NNN/
#   group catalogs (-c) : FoF/Subfind fof_subhalo_tab chunks -> <DATA_DIR>/groups_NNN/
#                         (needed by the merger-tree tracker for subhalo selection and
#                          particle-based shape/orientation; trees themselves are separate)
#   merger trees   (-t) : SubLink tree_extended chunks (whole simulation)
#   initial conds  (-i) : ics.hdf5 -> <DATA_DIR>/ (single file, whole simulation). The TNG
#                         API only serves ICs under the BARYONIC run's name, so for a
#                         '-Dark' simulation the '-Dark' suffix is stripped for the API call
#                         (same ICs; the Dark run is the DM-only evolution of them). Feed
#                         the file to python/tools/convert_ic_units.py to produce the
#                         h-free combined_ics.hdf5 that run_ps_dtfe.sh expects.
#
# Usage:
#   ./download_snapshots.sh [-k API_KEY] [-c|-t|-i] [-d DATA_DIR] [-s SIMULATION] [snapshot ...]
#
# Examples:
#   ./download_snapshots.sh -c -s TNG50-3-Dark            # groupcats for the default ladder
#   ./download_snapshots.sh -c -s TNG300-3-Dark 99 50     # groupcats, snapshots 99+50 only
#   ./download_snapshots.sh -k 0123456789abcdef 1 2 3     # raw snapshots
#   ./download_snapshots.sh -i -s TNG100-3-Dark           # ICs (served as TNG100-3)
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
DATA_DIR=""                       # default: sim_dir <SIMULATION> (config.sh)
MODE="snapshot"                   # 'snapshot' or 'groupcat' (-c)

# Snapshot numbers (each corresponds to a redshift) to download; overrides the
# short default list from config.sh with the full redshift ladder.
SNAPSHOTS=(1 2 3 4 5 8 13 17 21 25 33 40 50 67 72 78 84 91 99)

# The key travels in a header: https only (it went over plain http until 2026-10-06). Every failed
# fetch is counted and the exit status says so -- a wget that fails left the script exiting 0 and the
# launcher reading "Download complete!".
errors=0
finish() {
    echo ""
    if [ "${errors}" -gt 0 ]; then
        echo "Download finished with ${errors} error(s): see 'Error downloading' above. Run again to fetch what is missing."
        exit 1
    fi
    echo "Download complete!"
    exit 0
}

# A chunk left half-written by a stopped download is skipped by wget -nc for ever, and a merge of it
# would be a snapshot with missing particles (merge_HDF5.py refuses that now): before fetching, drop
# the files of this product that HDF5 calls truncated or signature-less, or that are too small to be
# a chunk at all (< 4 KB; the SAME two rules as merge_HDF5.py's _is_incomplete -- keep them in step)
# -- only those (a lock or a permission error keeps the file) and only when nothing wrote them in the
# last minute (another download may be running). A chunk that is kept although it does not open, or
# that could not be removed, is an ERROR of this run: wget -nc would skip it and the merge refuse it.
# Needs a python with h5py (PY, as the merge step; else the check is skipped with a note).
prune_incomplete() {        # <dir> <glob>  -> 0, or 1 when a chunk that does not open is still there
    local dir="$1" pat="$2" py="${PY:-python3}"
    [ -d "${dir}" ] || return 0
    if ! "${py}" -c "import h5py" >/dev/null 2>&1; then
        echo "  note: ${py} has no h5py; existing chunks are not checked for completeness"
        return 0
    fi
    HDF5_USE_FILE_LOCKING=FALSE "${py}" - "${dir}" "${pat}" <<'PYEOF'
import sys, time
from pathlib import Path
import h5py
d, pat = Path(sys.argv[1]), sys.argv[2]
kept = False
for p in sorted(d.glob(pat)):
    try:
        with h5py.File(p, "r"):
            pass
        continue
    except OSError as e:
        why = str(e).splitlines()[0]
    try:
        small = p.stat().st_size < 4096
        incomplete = small or any(k in why for k in ("truncated file", "file signature not found", "bad byte number"))
        if not incomplete:
            print(f"  note: {p.name} could not be opened ({why[:90]}); kept -- delete it (or fix its permissions) by hand")
            kept = True
            continue
        if time.time() - p.stat().st_mtime < 60:
            print(f"  note: {p.name} is incomplete but was written less than a minute ago (another download?); kept")
            kept = True
            continue
        p.unlink()
    except OSError as e2:
        print(f"  note: could not remove {p.name} ({e2}); kept")
        kept = True
        continue
    print(f"  Removed incomplete chunk {p.name} (not a readable HDF5 file): it will be downloaded again")
sys.exit(1 if kept else 0)
PYEOF
}

# After prune_incomplete: a chunk that does not open is still there -> this run cannot complete the product
kept_error() {              # <what>
    echo "  Error downloading $1: an unreadable chunk is still in place (see the note above); wget -nc skips it. Run again in a minute, or remove it."
    errors=$((errors + 1))
}

# A recursive fetch that exits 0 but left no chunk (the listing linked nothing https, or nothing at all)
fetched_none() {            # <dir> <glob>
    local f
    for f in "$1"/$2; do [ -e "$f" ] && return 1; done
    return 0
}

usage() {
    echo "Usage: $0 [-k API_KEY] [-c|-t|-i] [-d DATA_DIR] [-s SIMULATION] [snapshot ...]"
    echo "  -c  download FoF/Subfind group catalogs instead of raw snapshots"
    echo "  -t  download the SubLink merger trees (whole simulation; snapshot list ignored)"
    echo "  -i  download the initial conditions ics.hdf5 (whole simulation; snapshot list"
    echo "      ignored; '-Dark' is stripped for the API call -- ICs are served under the"
    echo "      baryonic run's name only)"
    echo "  API key: -k flag, TNG_API_KEY env, or ~/.tng_api_key file (recommended)."
}

# ---- Parse arguments ------------------------------------------------------
while getopts "k:d:s:ctih" opt; do
    case "$opt" in
        k) API_KEY="$OPTARG" ;;
        d) DATA_DIR="$OPTARG" ;;
        s) SIMULATION="$OPTARG" ;;
        c) MODE="groupcat" ;;
        t) MODE="tree" ;;
        i) MODE="ics" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# Any remaining positional arguments override the default snapshot list.
if [ "$#" -gt 0 ]; then
    SNAPSHOTS=("$@")
fi
# Snapshot numbers as decimal: bash's printf reads a leading zero as OCTAL, so '010' became 008 and '099'
# an error (then 000) -- the folders are named snapdir_0NN, so typing them that way is natural (2026-10-05).
for _k in "${!SNAPSHOTS[@]}"; do
    _s="${SNAPSHOTS[$_k]}"
    if ! [[ "${_s}" =~ ^[0-9]+$ ]]; then
        echo "Error: snapshot '${_s}' is not a number." >&2
        exit 2
    fi
    SNAPSHOTS[$_k]=$((10#${_s}))
done

if [ -z "${API_KEY}" ]; then
    echo "Error: an API key is required (-k, TNG_API_KEY, or ~/.tng_api_key)." >&2
    usage
    exit 1
fi
[ -z "${DATA_DIR}" ] && DATA_DIR="$(sim_dir "${SIMULATION}")"

echo "Downloading TNG ${MODE} files..."
echo "Simulation:     ${SIMULATION}"
echo "Data directory: ${DATA_DIR}"
[ "${MODE}" != "tree" ] && [ "${MODE}" != "ics" ] && echo "Snapshots:      ${SNAPSHOTS[*]}"
echo ""

# Initial conditions: one whole-simulation ics.hdf5, direct file endpoint (no chunk listing).
# The API serves ICs only under the baryonic run's name, so strip a '-Dark' suffix.
if [ "${MODE}" = "ics" ]; then
    # already converted? the raw ics.hdf5 is usually deleted after convert_ic_units.py
    if [ -f "${DATA_DIR}/combined_ics.hdf5" ]; then
        echo "Skipping ICs: ${DATA_DIR}/combined_ics.hdf5 already exists (delete it to re-download)."
        echo ""
        echo "Download complete!"
        exit 0
    fi
    ICS_SIM="${SIMULATION%-Dark}"
    [ "${ICS_SIM}" != "${SIMULATION}" ] && echo "ICs are served under '${ICS_SIM}' (shared with the -Dark run)."
    echo "Downloading ics.hdf5 -> ${DATA_DIR} ..."
    mkdir -p "${DATA_DIR}"
    prune_incomplete "${DATA_DIR}" "ics.hdf5" || kept_error "ics.hdf5"
    if wget -nc -nv --https-only --content-disposition \
            --header="API-Key: ${API_KEY}" \
            -P "${DATA_DIR}" \
            "https://www.tng-project.org/api/${ICS_SIM}/files/ics.hdf5"; then
        echo "  ics.hdf5 downloaded successfully"
        echo "  next: python3 python/tools/convert_ic_units.py '${DATA_DIR}/ics.hdf5' '${DATA_DIR}/combined_ics.hdf5'"
    else
        echo "  Error downloading ics.hdf5"
        errors=$((errors + 1))
    fi
    finish
fi

# SubLink merger trees are whole-simulation files (tree_extended.X.hdf5), not per-snapshot.
if [ "${MODE}" = "tree" ]; then
    dest_dir="${DATA_DIR}/Merger Trees"       # where dtfelib.trees.TreeSet looks
    # already merged? merge_HDF5.py --trees replaces the chunks
    if [ -f "${dest_dir}/combined_tree_extended.hdf5" ]; then
        echo "Skipping trees: ${dest_dir}/combined_tree_extended.hdf5 already exists (delete it to re-download)."
        echo ""
        echo "Download complete!"
        exit 0
    fi
    echo "Downloading sublink trees -> ${dest_dir} ..."
    mkdir -p "${dest_dir}"
    prune_incomplete "${dest_dir}" "tree_extended.*.hdf5" || kept_error "sublink trees"
    if wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 --https-only \
            --content-disposition \
            --header="API-Key: ${API_KEY}" \
            -P "${dest_dir}" \
            "https://www.tng-project.org/api/${SIMULATION}/files/sublink/?format=api"; then
        if fetched_none "${dest_dir}" "tree_extended.*.hdf5"; then
            echo "  Error downloading sublink trees: no chunk was fetched (the listing offered nothing over https?)"
            errors=$((errors + 1))
        else
            echo "  sublink trees downloaded successfully"
        fi
    else
        echo "  Error downloading sublink trees"
        errors=$((errors + 1))
    fi
    finish
fi

for i in "${SNAPSHOTS[@]}"; do
    # Format snapshot number with leading zeros (e.g. 1 -> 001)
    n_str=$(printf "%03d" "$i")
    if [ "${MODE}" = "groupcat" ]; then
        dest_dir="${DATA_DIR}/groups_${n_str}"      # fof_subhalo_tab_NNN.X.hdf5 chunks
        endpoint="groupcat-${i}"
        # already merged? merge_HDF5.py --groupcats replaces the chunks
        if [ -f "${dest_dir}/combined_fof_subhalo_tab_${n_str}.hdf5" ]; then
            echo "Skipping ${endpoint}: ${dest_dir}/combined_fof_subhalo_tab_${n_str}.hdf5 already exists (delete it to re-download)."
            echo ""
            continue
        fi
    else
        dest_dir="${DATA_DIR}/${INPUT_SUBDIR}_${n_str}"
        endpoint="snapshot-${i}"
        # already merged? merge_HDF5.py's raw chunks are usually deleted after the merge,
        # so wget -nc has nothing to skip on and would re-download the whole snapshot
        if [ -f "${dest_dir}/combined_${n_str}.hdf5" ]; then
            echo "Skipping ${endpoint}: ${dest_dir}/combined_${n_str}.hdf5 already exists (delete it to re-download)."
            echo ""
            continue
        fi
    fi

    echo "Downloading ${endpoint} -> ${dest_dir} ..."
    mkdir -p "${dest_dir}"
    if [ "${MODE}" = "groupcat" ]; then
        chunk_glob="fof_subhalo_tab_${n_str}.*.hdf5"
    else
        chunk_glob="snap_${n_str}.*.hdf5"
    fi
    prune_incomplete "${dest_dir}" "${chunk_glob}" || kept_error "${endpoint}"

    if wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 --https-only \
            --content-disposition \
            --header="API-Key: ${API_KEY}" \
            -P "${dest_dir}" \
            "https://www.tng-project.org/api/${SIMULATION}/files/${endpoint}/?format=api"; then
        if fetched_none "${dest_dir}" "${chunk_glob}"; then
            echo "  Error downloading ${endpoint}: no chunk was fetched (the listing offered nothing over https?)"
            errors=$((errors + 1))
        else
            echo "  ${endpoint} downloaded successfully"
        fi
    else
        echo "  Error downloading ${endpoint}"
        errors=$((errors + 1))
    fi
    echo ""
done

finish
