# Shared configuration for run_dtfe.sh, run_ps_dtfe.sh and download_snapshots.sh.
# Sourced, not executed. Every value can be overridden per run:
#   - environment: DTFE_DATA_ROOT=/scratch/tng DTFE_SIM=TNG50-3-Dark ./run_ps_dtfe.sh
#                  GRID_SIZE=256 ./run_ps_dtfe.sh          (or DTFE_GRID_SIZE / DTFE_PADDING)
#   - script flags: ./run_ps_dtfe.sh -d /path/to/sim -g 256 99
# The Python side (python/dtfelib/cli.py) honours the same DTFE_DATA_ROOT / DTFE_SIM variables.

# Root of the simulation data. Two layouts are recognised (see sim_dir below):
#   per family  $DATA_ROOT/<family>/<sim>/snapdir_NNN/   the Samsung T7 (default):
#               Illustris TNG/TNG100/TNG100-3-Dark, Illustris TNG/TNG300/TNG300-3-Dark, ...
#   flat        $DATA_ROOT/<sim>/snapdir_NNN/            e.g. DTFE_DATA_ROOT=~/output (the old copy)
# where <family> is the simulation name up to its first '-' (TNG100-3-Dark -> TNG100). With the
# T7 unplugged the run scripts stop at "no snapshots under ..." rather than writing elsewhere.
DATA_ROOT="${DTFE_DATA_ROOT:-/Volumes/Samsung T7/Illustris TNG}"

# The directory of simulation $1 under DATA_ROOT, in whichever layout holds it: an existing
# <sim> or <family>/<sim> directory wins; a simulation not on disk yet goes into its family
# folder when that exists (so downloads land beside their siblings), else flat.
# Python: dtfelib.cli.sim_dir -- keep the two in step.
sim_dir() {
    local sim="$1" fam="${1%%-*}"
    if [ -d "${DATA_ROOT}/${sim}" ]; then
        echo "${DATA_ROOT}/${sim}"
    elif [ "${fam}" != "${sim}" ] && [ -d "${DATA_ROOT}/${fam}" ]; then
        echo "${DATA_ROOT}/${fam}/${sim}"
    else
        echo "${DATA_ROOT}/${sim}"
    fi
}

# Simulation to process; snapshots live in $(sim_dir $SIMULATION)/snapdir_NNN/.
SIMULATION="${DTFE_SIM:-TNG50-3-Dark}"

# Snapshot numbers (each corresponds to a redshift) the run scripts process by default;
# positional arguments to the scripts override this list.
SNAPSHOTS=(99)

# Interpolation grid cells per axis and padding (mean interparticle spacings).
# Environment-overridable like DATA_ROOT above. The run scripts source this file BEFORE parsing
# their flags, so precedence is:  -g flag  >  environment  >  the default here.
GRID_SIZE="${GRID_SIZE:-${DTFE_GRID_SIZE:-1024}}"
PADDING="${PADDING:-${DTFE_PADDING:-50}}"

# Subdirectory prefix of the snapshot chunks inside the simulation directory.
INPUT_SUBDIR="snapdir"

# ---- run guard + preflight (2026-10-11), shared by run_dtfe.sh and run_ps_dtfe.sh ------------------
# Functions only: nothing here runs or exits when the file is sourced (run_ps_pipeline.sh and
# download_snapshots.sh source it too and print their own messages).

# data_root_problem <data dir>: prints why the data cannot be there and returns 1 when the DATA ROOT
# itself is missing (the T7 unplugged, or remounted by macOS as '/Volumes/Samsung T7 1'); returns 0
# otherwise. A missing SIMULATION under an existing root is not this function's business: the drivers
# keep skipping its snapshots and exit 1 with 'nothing processed'.
data_root_problem() {
    local dir="$1" root="${DATA_ROOT}"
    [ -d "${dir}" ] && return 0
    case "${dir}" in "${root}"|"${root}"/*) ;; *) root="$(dirname "${dir}")" ;; esac   # -d given: its parent
    [ -d "${root}" ] && return 0
    echo "Error: the data folder '${root}' does not exist: is the data disk mounted?" >&2
    local v found=0
    for v in /Volumes/Samsung\ T7*; do
        [ -d "${v}" ] || continue
        [ "${found}" = 0 ] && echo "       Volumes that look like it:" >&2
        echo "         ${v}   (DTFE_DATA_ROOT='${v}/Illustris TNG' to use it)" >&2
        found=1
    done
    echo "       Set DTFE_DATA_ROOT (or -d DATA_DIR) to where the snapshots are." >&2
    return 1
}

# expected_output_bytes <grid> <bytes per value> <fields...>: an APPROXIMATE size of the grids a run
# writes -- components per field from a TNG100 512^3 run's files (density 1, velocity 3, gradient 9,
# divergence 1, shear 5, vorticity 3, dispersion 1+6, tweb/vweb 1+3 eigenvalues), plus the stream count
# and the hidden-streams flag every PS run writes, plus 10% for the point files and the caustic grids.
expected_output_bytes() {
    local g="$1" b="$2" f n=2; shift 2
    for f in "$@"; do
        case "${f%_a}" in
            density|divergence) n=$((n + 1)) ;;
            velocity|vorticity) n=$((n + 3)) ;;
            gradient) n=$((n + 9)) ;;
            shear) n=$((n + 5)) ;;
            dispersion) n=$((n + 7)) ;;
            tweb|vweb) n=$((n + 4)) ;;
            *) n=$((n + 1)) ;;
        esac
    done
    echo $(( g * g * g * b * n * 11 / 10 ))
}

# output_space_problem <output dir> <bytes needed>: 1 (with the reason) when the folder the grids go to
# is not writable or has less free space than the grids need. The grids are written beside the old ones
# and renamed at the end (<file>.part.<pid>), so the old set does not count as free. DTFE_SPACE_CHECK=0
# skips the space half (the estimate is approximate).
output_space_problem() {
    local dir="$1" need="$2" avail_kb
    if [ ! -w "${dir}" ]; then
        echo "  Error: the output folder '${dir}' is not writable" >&2
        return 1
    fi
    [ "${DTFE_SPACE_CHECK:-1}" = "0" ] && return 0
    avail_kb="$(df -Pk "${dir}" 2>/dev/null | awk 'NR==2 {print $4}')"
    [[ "${avail_kb}" =~ ^[0-9]+$ ]] || return 0           # no df answer: do not guess
    if [ $((avail_kb * 1024)) -lt "${need}" ]; then
        echo "  Error: '${dir}' has $(awk -v k="${avail_kb}" 'BEGIN{printf "%.1f", k/1048576}') GB free; this run's grids need about $(awk -v b="${need}" 'BEGIN{printf "%.1f", b/1073741824}') GB (DTFE_SPACE_CHECK=0 skips this check)" >&2
        return 1
    fi
    return 0
}

# The single-instance lock (shared contract C1): '<prefix>.runlog.lock' is a DIRECTORY made with mkdir
# (atomic on every filesystem; macOS has no flock(1)) holding 'pid', the driver's pid, for as long as a
# driver runs that snapshot into that prefix. Two drivers on the same snapshot used to run side by side
# and both got SIGKILLed for memory (the 2026-10-03 double launch). A lock whose pid is gone, or now
# belongs to a process that is not one of the drivers (a reused pid), is stale: removed with a message --
# unless a live process still works on that snapshot (2026-10-11): a driver SIGKILLed on its own leaves its
# binary (and tee) running, and removing that lock started a second 40-50 GB binary beside the orphan.
# run_lock_take <lock dir>: 0 = taken; 1 = held by a live driver or process (the refusal printed).
# run_lock_live_process <output root>: prints 'pid args' of the first live process naming the root on its
# command line (the binary's output argument, or tee's '<root>.runlog.partial'), as the launcher's
# _live_roots does (python/gui/results.py); status 1 when there is none or ps cannot say
run_lock_live_process() {
    local root="$1" line pid args
    [ -n "${root}" ] || return 1
    while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"      # ps right-aligns the pid
        pid="${line%% *}"; args="${line#* }"
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        [ "${pid}" = "$$" ] && continue
        case " ${args} " in
            *" ${root} "*|*" ${root}."*)
                echo "${pid} ${args}"; return 0 ;;
        esac
    done < <(ps -A -ww -o pid=,args= 2>/dev/null)
    return 1
}
run_lock_take() {
    local lock="$1" pid cmd tries live root
    root="${lock%.runlog.lock}"
    for tries in 1 2; do
        if mkdir "${lock}" 2>/dev/null; then
            echo "$$" > "${lock}/pid"
            return 0
        fi
        [ -d "${lock}" ] || { echo "  Error: cannot create the lock '${lock}' (is the output folder writable?)" >&2; return 1; }
        pid="$(cat "${lock}/pid" 2>/dev/null || true)"
        if [ -z "${pid}" ]; then                     # mkdir done, pid not yet written: give it a moment
            sleep 1; pid="$(cat "${lock}/pid" 2>/dev/null || true)"
        fi
        cmd=""
        [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null && cmd="$(ps -o args= -p "${pid}" 2>/dev/null || true)"
        case "${cmd}" in
            *run_dtfe.sh*|*run_ps_dtfe.sh*)
                echo "  Error: another driver (pid ${pid}) is running this snapshot into the same output:" >&2
                echo "         ${lock}" >&2
                echo "         Two runs of one snapshot do not fit in memory together; wait for it, or stop it." >&2
                return 1 ;;
        esac
        # the driver is gone; its binary may not be (the lock's root is '<root>.runlog.lock')
        if [ "${root}" != "${lock}" ] && live="$(run_lock_live_process "${root}")"; then
            echo "  Error: the driver that locked this snapshot (pid ${pid:-unknown}) is gone, but a process still" >&2
            echo "         works on it (pid ${live%% *}): ${live#* }" >&2
            echo "         Two runs of one snapshot do not fit in memory together; wait for it, or stop it" >&2
            echo "         (kill ${live%% *}). The lock stays: ${lock}" >&2
            return 1
        fi
        echo "  Note: removing a stale lock (pid ${pid:-unknown} is no longer running): ${lock}"
        rm -rf "${lock}"
    done
    echo "  Error: could not take the lock '${lock}'" >&2
    return 1
}
# run_lock_release <lock dir>: removes it when it is this driver's own (never another driver's)
run_lock_release() {
    [ -n "$1" ] && [ -d "$1" ] || return 0
    [ "$(cat "$1/pid" 2>/dev/null)" = "$$" ] && rm -rf "$1"
    return 0
}

# ps_checkpoint_dir <output root> [<scratch dir>]: where PS-DTFE keeps the checkpoint of a partitioned run
# (2026-10-11, src/partition_checkpoint.h directoryFor -- keep the two in step): '<output>.runlog.checkpoint'
# beside the outputs, or with a scratch dir '<scratch>/<basename>.<FNV-1a 64 of the absolute output path, 16
# hex>.runlog.checkpoint'. run_ps_dtfe.sh passes '--resume' when its 'checkpoint' file exists. The absolute path is the one the
# binary sees: a relative output is taken from the physical current directory (getcwd, so 'pwd -P').
ps_checkpoint_dir() {
    local out="$1" scratch="${2:-}"
    if [ -z "${scratch}" ]; then
        printf '%s.runlog.checkpoint\n' "${out}"
        return 0
    fi
    case "${out}" in /*) ;; *) out="$(pwd -P)/${out}" ;; esac
    local LC_ALL=C h=-3750763034362895579 i c   # the FNV-1a offset basis 14695981039346656037, as a signed 64-bit
    for (( i = 0; i < ${#out}; i++ )); do
        printf -v c '%d' "'${out:i:1}"
        h=$(( (h ^ (c & 255)) * 1099511628211 ))   # bash wraps at 64 bits, as the C++ uint64_t does
    done
    while [ "${scratch}" != "/" ] && [ "${scratch%/}" != "${scratch}" ]; do scratch="${scratch%/}"; done
    printf '%s/%s.%016x.runlog.checkpoint\n' "${scratch}" "${out##*/}" "${h}"
}
