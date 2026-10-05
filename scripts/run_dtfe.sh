#!/usr/bin/env bash
#
# Batch-run standard DTFE over a set of TNG snapshots (redshifts).
#
# Companion to download_snapshots.sh, which fetches the snapshot files first.
# Shared defaults (DATA_ROOT, SIMULATION, SNAPSHOTS, GRID_SIZE, PADDING) live in config.sh.
#
# Usage:
#   ./run_dtfe.sh [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-m] [-e] [snapshot ...]
#     -m   run the '_a' interpolation on the Apple GPU (same as DTFE_METAL=1; needs 'make DTFE METAL=1')
#     -e   every '_a' cell the EXACT average of the linear interpolant over the cell (--exact-average,
#          same as DTFE_EXACT_AVERAGE=1) instead of the Monte-Carlo sample mean; ~6.6x faster on the GPU
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
DTFE_EXACT_AVERAGE="${DTFE_EXACT_AVERAGE:-0}"  # 1 = exact cell averages (--exact-average; noise-free, slower on the CPU)
DTFE_PRECISION="${DTFE_PRECISION:-single}"  # double = the double-precision binary ./DTFE-double ('make DTFE DOUBLE=1'):
                                 # everything in double from the input read onward, float64 outputs, ~2x the memory
SCRATCH_DIR="${SCRATCH_DIR:-}"   # out-of-core mode (--scratch-dir): the full-resolution grids as mmap'ed files in this
                                 # LOCAL directory instead of RAM (bit-identical results; iCloud paths are refused) --
                                 # the same knob as run_ps_dtfe.sh; the launcher's scratch folder arrives here
LAMBDA_TH="${LAMBDA_TH:-0.3}"    # eigenvalue threshold of the T-web / V-web classes (--lambda_th), the SAME default as
                                 # run_ps_dtfe.sh (the binary's own default is 0.0: a run without this flag classifies
                                 # differently from a scripted one)

DATA_DIR=""                # default: sim_dir $SIMULATION (config.sh); override with -d
OUTPUT_SUBDIR="output"

FIELDS="density_a velocity_a gradient_a divergence_a shear_a vorticity_a"

usage() { echo "Usage: $0 [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-m] [-e] [snapshot ...]"; }

# ---- Parse arguments ------------------------------------------------------
while getopts "d:s:g:meh" opt; do
    case "$opt" in
        d) DATA_DIR="$OPTARG" ;;
        s) SIMULATION="$OPTARG" ;;
        g) GRID_SIZE="$OPTARG" ;;
        m) DTFE_METAL=1 ;;
        e) DTFE_EXACT_AVERAGE=1 ;;
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

[ -z "${DATA_DIR}" ] && DATA_DIR="$(sim_dir "${SIMULATION}")"

cd "$REPO_ROOT" || exit 1

case "${DTFE_PRECISION}" in
    single) DTFE_BIN="./DTFE" ;;
    double) DTFE_BIN="./DTFE-double" ;;
    *) echo "Error: DTFE_PRECISION must be 'single' or 'double' (got '${DTFE_PRECISION}')." >&2; exit 1 ;;
esac
if [ ! -x "${DTFE_BIN}" ]; then
    if [ "${DTFE_PRECISION}" = "double" ]; then
        echo "Error: ${DTFE_BIN} not found. Build the double-precision pair with 'make DTFE PS-DTFE DOUBLE=1' (add METAL=1 on Apple Silicon), or let the launcher's Setup do it." >&2
    else
        echo "Error: ${DTFE_BIN} not found or not executable. Build it with 'make DTFE'." >&2
    fi
    exit 1
fi

# Peak-memory wrapper, resolved once and OPTIONAL (as run_ps_dtfe.sh): macOS spells the flag -l, GNU -v,
# a bare container has no /usr/bin/time. Without one the run is identical, just with no footprint line.
TIME_WRAP=()
if [ -x /usr/bin/time ]; then
    if /usr/bin/time -l true >/dev/null 2>&1; then
        TIME_WRAP=(/usr/bin/time -l)          # BSD/macOS: "<bytes>  maximum resident set size"
    elif /usr/bin/time -v true >/dev/null 2>&1; then
        TIME_WRAP=(/usr/bin/time -v)          # GNU: "Maximum resident set size (kbytes): <n>"
    fi
fi

# Keep the Mac awake for the whole batch; caffeinate follows this PID and lifts on exit.
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -m -w $$ &
fi

echo "Starting DTFE processing..."
echo "Data directory: ${DATA_DIR}"
echo "Grid size: ${GRID_SIZE}"
[ "${DTFE_EXACT_AVERAGE}" = "1" ] && echo "Cell averages: exact (--exact-average)"
[ -n "${SCRATCH_DIR}" ] && echo "Scratch directory: ${SCRATCH_DIR} (grids out of core)"
echo ""

n_ok=0; n_skipped=0; n_failed=0
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
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # GPU interpolation toggle (-m / DTFE_METAL=1); ignored with a warning on non-METAL builds.
    metal_args=()
    [ "${DTFE_METAL}" = "1" ] && metal_args=(--gpu)
    exact_args=()
    [ "${DTFE_EXACT_AVERAGE}" = "1" ] && exact_args=(--exact-average)

    # pass --partition/--max-concurrent only when set; otherwise the binary auto-tunes them
    part_args=()
    [ -n "${PARTITION}" ] && part_args+=(--partition ${PARTITION})
    [ -n "${MAX_CONCURRENT}" ] && part_args+=(--max-concurrent "${MAX_CONCURRENT}")
    scratch_args=()
    [ -n "${SCRATCH_DIR}" ] && scratch_args=(--scratch-dir "${SCRATCH_DIR}")

    # the tee pipe hides the terminal from the binary's isatty() check: force colours through it when
    # this script itself runs on a terminal (the runlog is de-ANSI'd below), as run_ps_dtfe.sh does
    [ -t 1 ] && export CLICOLOR_FORCE=1
    echo "  Running DTFE on ${input_file}..."
    # the run log beside the outputs (<snapdir>/output.runlog), as run_ps_dtfe.sh keeps one: the
    # GUI's Runs browser reads the settings, build stamp, wall time and peak memory back from it
    run_log="${output_root}.runlog"
    SECONDS=0
    ${TIME_WRAP[@]+"${TIME_WRAP[@]}"} "${DTFE_BIN}" "${input_file}" "${output_root}" \
        --grid ${GRID_SIZE} \
        --padding ${PADDING} \
        --periodic \
        ${part_args[@]+"${part_args[@]}"} \
        --field ${FIELDS} \
        --lambda_th ${LAMBDA_TH} \
        ${scratch_args[@]+"${scratch_args[@]}"} \
        ${metal_args[@]+"${metal_args[@]}"} \
        ${exact_args[@]+"${exact_args[@]}"} 2>&1 | tee "${run_log}"
    rc=${PIPESTATUS[0]}                 # the binary's status, taken BEFORE anything else runs a pipe

    # strip ANSI colour codes from the saved log (a temp file: BSD and GNU sed disagree on -i)
    if sed -E $'s/\033\\[[0-9;]*m//g' "${run_log}" > "${run_log}.tmp" 2>/dev/null; then
        mv "${run_log}.tmp" "${run_log}"
    else
        rm -f "${run_log}.tmp"
    fi
    # peak memory in whichever dialect TIME_WRAP produced (absent without one); macOS also reports the
    # peak FOOTPRINT, what the run really needed including compressed or swapped pages
    peak_bytes=$(awk 'tolower($0) ~ /maximum resident set size/ {
                          if (tolower($0) ~ /kbytes/) { gsub(/[^0-9]/, "", $NF); print $NF * 1024 }
                          else                        { print $1 }
                          exit }' "${run_log}")
    if [ -n "${peak_bytes}" ]; then
        peak_gb=$(awk -v b="${peak_bytes}" 'BEGIN{printf "%.1f", b/1073741824}')
        fp_bytes=$(awk 'tolower($0) ~ /peak memory footprint/ { print $1; exit }' "${run_log}")
        if [ -n "${fp_bytes}" ]; then
            echo "  Peak memory: $(awk -v b="${fp_bytes}" 'BEGIN{printf "%.1f", b/1073741824}') GB footprint, ${peak_gb} GB resident"
        else
            echo "  Peak memory: ${peak_gb} GB resident"
        fi
    fi
    echo "  Wall time: $((SECONDS/60)) min $((SECONDS%60)) s"

    if [ "${rc}" -eq 0 ]; then
        echo "  Snapshot ${n_str} processed successfully"
        n_ok=$((n_ok + 1))
    else
        echo "  Error processing snapshot ${n_str}"
        n_failed=$((n_failed + 1))
    fi
    echo ""
done

# Honest exit status, as run_ps_dtfe.sh: a failed snapshot, or nothing processed, exits 1 (before 2026-10-05
# this script always exited 0, so the launcher read a failed standard-DTFE job as done). A partial skip stays 0.
echo "DTFE processing complete: ${n_ok} processed, ${n_skipped} skipped, ${n_failed} failed."
if [ "${n_failed}" -gt 0 ]; then
    echo "ERROR: ${n_failed} snapshot(s) failed -- see the messages above and the .runlog files." >&2
    exit 1
fi
if [ "${n_ok}" -eq 0 ]; then
    echo "ERROR: no snapshot was processed (${n_skipped} skipped for missing inputs) -- check DATA_ROOT/SIMULATION (looked under '${DATA_DIR}')." >&2
    exit 1
fi
