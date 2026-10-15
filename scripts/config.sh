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
# their flags, so precedence is:  -g flag  >  environment  >  the default here. PADDING's 50 is the PS-DTFE
# scripts' default; run_dtfe.sh has its own, 15 (complete at z = 0 on TNG50/100/300, 2026-10-12; see there).
# GRID_SIZE 512 since 2026-10-13: the production grids of both estimators (every thesis grid on the T7, and
# run_ps_pipeline.sh's and the launcher's default) are 512^3; the 1024 here made a bare driver run a grid 8x the
# production one (~14 GB per field set, a scratch dir needed) that nothing downstream reads.
GRID_SIZE="${GRID_SIZE:-${DTFE_GRID_SIZE:-512}}"
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

# gpu_build_default <repo root> <binary>: 1 when the Makefile built <binary> (./DTFE, ./DTFE-double, ./PS-DTFE,
# ./PS-DTFE-double) with a GPU backend, else 0 -- the drivers' default for DTFE_METAL / PS_METAL ('auto', 2026-10-13:
# every production grid ran on the GPU, but a driver run without -m ran on the CPU). The answer is the stamp the
# Makefile's mode check leaves in the pair's object folder (o, o_d, o_ps, o_ps_d): '.gpu_mode_metal', '_cuda' or
# '_hip'. The binary itself cannot be asked: '--help' lists --gpu / --ps-gpu in every build and '--version' names no
# backend. NOT the launcher's rule (python/gui/runspec.py gpu_built: any stamp but '.gpu_mode_off'), on purpose: the
# CI's 'cudaemu' build emulates the CUDA kernels on the CPU, far slower than the CPU path, so it is no GPU here (-m
# still switches it on). No object folder (binaries copied without their build) = 0. A stale stamp (a binary put
# back by install_lib.sh after a failed rebuild in another mode) costs nothing: '--gpu' on a CPU build falls back to
# the CPU with a warning, and the CPU on a GPU build is slower, not different.
gpu_build_default() {
    local root="$1" obj m
    case "${2##*/}" in
        DTFE) obj=o ;;
        DTFE-double) obj=o_d ;;
        PS-DTFE) obj=o_ps ;;
        PS-DTFE-double) obj=o_ps_d ;;
        *) echo 0; return 0 ;;
    esac
    for m in metal cuda hip; do
        [ -e "${root}/${obj}/.gpu_mode_${m}" ] && { echo 1; return 0; }
    done
    echo 0
}

# ---- a rerun's left-over files (2026-10-13, audit finding 7) -------------------------------------------------
# The binary replaces its outputs as a set (output_staging.h: '<file>.part.<pid>' renamed into place only once all
# are written), but it never removed what an EARLIER run with the same prefix wrote and this one does not: a rerun
# without the image plane, the caustics or the T-web left the old '.pts_*', '.caustic*' or '.a_tweb*' beside the new
# grids, a mixed set that read as one run (the README promised a folder never holds a mix). Now the drivers, after a
# run that EXITED 0, move every '<prefix>.*' file the run did not write into '<prefix>.superseded/<stamp>/' beside
# them -- moved, never deleted -- with a copy of the old run's log, so the moved files keep their provenance.
# 'Did not write': the file existed before the run with the same inode (the binary renames every output into place,
# so a written file is a new inode) AND is not newer than the run's start (the one output written in place, a
# thread's '.badGridPoints<n>', keeps its inode but not its time). A failed, stopped or killed run moves nothing:
# the old set stays complete, as the run-log contract keeps its log. Neither a dry run nor an auto-tune report
# writes anything, so neither moves anything. Not moved: the run's own bookkeeping ('<prefix>.runlog' and every
# '<prefix>.runlog.*': the failed / partial log, the lock, the progress sidecar, a PS checkpoint), the launcher's
# settings sidecar '<prefix>.gui.json' and the binary's own '<file>.part.<pid>' (output_staging.h handles those).
# The launcher's Runs browser does not see the folder (it reads '<snapdir>/<prefix>.runlog*' and only the FILES
# '<prefix>.*' as a run's outputs).
#
# supersede_snapshot <output root>: what a run may later supersede, recorded BEFORE it starts -- one line
# '<inode> <name>' per candidate file (stdout); the caller keeps the text and passes it back afterwards.
supersede_snapshot() {
    local root="$1" f n ino
    for f in "${root}".*; do
        [ -f "${f}" ] && [ ! -L "${f}" ] || continue        # files only: '.superseded/', '.runlog.lock/' are folders
        n="${f##*/}"
        case "${n#"${root##*/}"}" in
            .runlog|.runlog.*|.gui.json|*.part.[0-9]*) continue ;;
        esac
        ino="$(ls -id -- "${f}" 2>/dev/null | awk '{ print $1 }')"
        [[ "${ino}" =~ ^[0-9]+$ ]] || continue
        printf '%s %s\n' "${ino}" "${n}"
    done
    return 0
}
# supersede_extras <output root> <recorded list> <run start marker file> <saved old runlog or ''>: after a run that
# exited 0, moves each recorded file the run did not write (same inode, not newer than the marker) into
# '<root>.superseded/<YYYY-MM-DD_HHMMSS>/' (the marker's time; '-<pid>' added when that folder exists), with the
# saved old log as '<prefix>.runlog' there. Prints one line per move and a summary; a failed move is said and left
# (the run itself succeeded). Never deletes, never overwrites ('mv -n', checked). Returns 0.
supersede_extras() {
    local root="$1" list="$2" marker="$3" oldlog="$4" dir pfx line ino n f now dest moved=0 failed=0
    [ -n "${list}" ] || return 0
    dir="${root%/*}"; pfx="${root##*/}"
    while IFS= read -r line; do
        [ -n "${line}" ] || continue
        ino="${line%% *}"; n="${line#* }"; f="${dir}/${n}"
        [ -f "${f}" ] || continue                                  # gone (the binary removed it): nothing to do
        now="$(ls -id -- "${f}" 2>/dev/null | awk '{ print $1 }')"
        [ "${now}" = "${ino}" ] || continue                        # a new inode: this run wrote it
        [ -n "${marker}" ] && [ "${f}" -nt "${marker}" ] && continue   # rewritten in place by this run
        if [ -z "${dest:-}" ]; then
            dest="${root}.superseded/$(date -r "${marker}" '+%Y-%m-%d_%H%M%S' 2>/dev/null || date '+%Y-%m-%d_%H%M%S')"
            [ -e "${dest}" ] && dest="${dest}-$$"
            if ! mkdir -p "${dest}"; then
                echo "  Warning: cannot create '${dest}': the earlier run's files this run did not write stay beside the new grids" >&2
                return 0
            fi
        fi
        if [ -e "${dest}/${n}" ] || ! mv -n "${f}" "${dest}/${n}" || [ -e "${f}" ]; then
            echo "  Warning: could not move '${n}' into ${dest} (left in place)" >&2
            failed=$((failed + 1)); continue
        fi
        echo "  Superseded: ${n} (an earlier run's; this run did not write it)"
        moved=$((moved + 1))
    done <<< "${list}"
    if [ "${moved}" -gt 0 ] && [ -n "${oldlog}" ] && [ -f "${oldlog}" ] && [ ! -e "${dest}/${pfx}.runlog" ]; then
        cp -p "${oldlog}" "${dest}/${pfx}.runlog" 2>/dev/null || true
    fi
    if [ "${moved}" -gt 0 ]; then
        echo "  ${moved} file(s) of an earlier run with this prefix moved to ${dest} (nothing deleted; its run log copied there)"
    fi
    [ -n "${dest:-}" ] && [ "${moved}" -eq 0 ] && rmdir "${dest}" "${root}.superseded" 2>/dev/null
    return 0
}

# ps_checkpoint_dir <output root> [<scratch dir>]: where PS-DTFE -- and since 2026-10-13 standard DTFE, by the same
# rule -- keeps the checkpoint of a partitioned run
# (2026-10-11, src/partition_checkpoint.h directoryFor -- keep the two in step): '<output>.runlog.checkpoint'
# beside the outputs, or with a scratch dir '<scratch>/<basename>.<FNV-1a 64 of the absolute output path, 16
# hex>.runlog.checkpoint'. run_ps_dtfe.sh and run_dtfe.sh pass '--resume' when its 'checkpoint' file exists. The absolute path is the one the
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
