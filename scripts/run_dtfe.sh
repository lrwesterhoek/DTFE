#!/usr/bin/env bash
#
# Batch-run standard DTFE over a set of TNG snapshots (redshifts).
#
# Companion to download_snapshots.sh, which fetches the snapshot files first.
# Shared defaults (DATA_ROOT, SIMULATION, SNAPSHOTS, GRID_SIZE) live in config.sh. PADDING (or DTFE_PADDING), the
# padding in mean particle spacings, defaults to 15 here, not config.sh's 50 (the PS-DTFE scripts' value).
#
# Usage:
#   ./run_dtfe.sh [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-m] [-e] [snapshot ...]
#     -m   run the '_a' interpolation on the GPU (same as DTFE_METAL=1). The default (DTFE_METAL unset, 'auto')
#          is the GPU when the binary was built with one (the Makefile's o/.gpu_mode_metal|cuda|hip stamp;
#          o_d for DTFE_PRECISION=double), else the CPU; DTFE_METAL=0 runs the CPU on a GPU build.
#     -e   every '_a' cell the EXACT average of the linear interpolant over the cell (--exact-average). The
#          DEFAULT since 2026-10-13 (DTFE_EXACT_AVERAGE=1, the production standard: the Monte-Carlo cell mean
#          lost 0.17-0.29% of the mass in every 2026-10-12 grid, the exact one ~3e-8); -e is kept for old
#          command lines. DTFE_EXACT_AVERAGE=0 gives the Monte-Carlo sample mean (1.1-1.7x faster at 512^3 on
#          the GPU, 2026-10-13 audit; the GPU is what makes the exact average affordable: ~6.6x the CPU's speed)
#   Production defaults (2026-10-13): GRID_SIZE 512 (config.sh), padding 15, the GPU when built, the exact cell
#   average, FIELDS = density, velocity, its gradient / divergence / shear / vorticity, the T-web and the V-web,
#   LAMBDA_TH 0.3 -- what scripts/thesis_runs.sh's dtfe27 stage runs, minus its explicit split.
#   OUTPUT_PREFIX (default 'output'): the files are <snapdir>/<prefix>.*; another prefix keeps a variant beside
#   the default set (dtfelib reads 'output.' for standard DTFE). A 'ps_*' prefix is refused: that is PS-DTFE's
#   naming (an OUTPUT_PREFIX exported for run_ps_dtfe.sh would otherwise overwrite the phase-space grids).
#   A rerun into a prefix that holds an earlier run's files: once it exits 0, the files it did not write (an image
#   plane, caustics or a field the new run lacks) are MOVED into <prefix>.superseded/<date_time>/ with the old
#   run log (config.sh supersede_extras; never deleted), so the folder holds one run's set.
#   -h / --help prints this header and runs nothing; --version the repository revision.
#   DRY_RUN=1 ./run_dtfe.sh ...   prints every snapshot's expanded command and paths, runs nothing (exit 0
#          when at least one snapshot is planned).
#   AUTO_TUNE_REPORT=1 ./run_dtfe.sh ...   only asks the binary's auto-tuner what each snapshot would need (one
#          'AUTO-TUNE-REPORT snap=NNN partition=.. mc=.. predicted_gb=..' line each; mc = the threads of a partition,
#          the partitions run one after the other); nothing is computed or written (2026-10-13, as run_ps_dtfe.sh).
#   Run logs, beside the outputs: output.runlog belongs to the grids on disk, output.runlog.failed to the
#   last failed or stopped attempt, output.runlog.partial to a run still going (or killed outright).
#   While a snapshot runs, <prefix>.runlog.lock (a folder holding the driver's pid) refuses a second driver on
#   it; a lock left by a killed driver is removed with a note. Before anything runs: the data disk must be
#   there (exit 2 otherwise) and the output folder writable with room for the grids (DTFE_SPACE_CHECK=0 skips
#   the space half).
#   DTFE_RESUME (default 1): a partitioned run killed part-way resumes, as run_ps_dtfe.sh's PS_RESUME. When the
#   snapshot's checkpoint (<prefix>.runlog.checkpoint, or in SCRATCH_DIR) exists, the next run passes --resume and
#   computes only the partitions it lacks; DTFE_RESUME=0 starts over (the new run replaces the checkpoint). A run with
#   SAMPLE_POINTS too since 2026-10-14 (the binary saves each finished partition's point answers beside the grids).
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # the binaries and Makefile live one level up
# Padding (mean interparticle spacings) of each partition and of each thread's sub-box: standard DTFE's own default
# is 15, not config.sh's 50 (which stays the PS-DTFE scripts' value). 15 is complete -- no '*.badGridPoints*' file --
# on the z = 0 snapshots of TNG50/100/300-3-Dark, the hardest case (the emptiest voids give the largest Delaunay
# circumspheres at the partition edges), and 22-29% faster than 20 there; 10 is not complete at z = 0, and 50 costs
# 33x the vertices (2026-10-12). A PADDING or DTFE_PADDING from the caller still wins: it is read here, before
# config.sh fills PADDING with its own default.
_caller_padding="${PADDING:-${DTFE_PADDING:-}}"
source "${SCRIPT_DIR}/config.sh"
PADDING="${_caller_padding:-15}"

# ---- Script-specific configuration ----------------------------------------
PARTITION="${PARTITION:-}"       # EMPTY (default) = the binary AUTO-TUNES the split from the particle
                                 # count, grid, fields and available RAM/cores (see the AUTO-TUNE line
                                 # it prints). Set to override, e.g. PARTITION="2 2 2".
MAX_CONCURRENT="${MAX_CONCURRENT:-}"  # cap on concurrent triangulations to bound peak RAM (0 = all cores).
                                 # EMPTY (default) = auto-tuned together with the partition split.
DTFE_METAL="${DTFE_METAL:-auto}"  # 1 = the '_a' interpolation on the GPU (--gpu), 0 = the CPU; 'auto' (default,
                                 # 2026-10-13; it was 0) = 1 when the binary is a GPU build (config.sh
                                 # gpu_build_default), resolved below once DTFE_PRECISION has chosen the binary
DTFE_EXACT_AVERAGE="${DTFE_EXACT_AVERAGE:-1}"  # 1 (default since 2026-10-13, the production standard) = exact cell
                                 # averages (--exact-average: no sampling noise, mass exact to ~3e-8); 0 = the
                                 # Monte-Carlo sample mean, which lost 0.17-0.29% of the mass in the 2026-10-12 grids
DRY_RUN="${DRY_RUN:-0}"          # 1 = print the commands and paths, run nothing (survey rank 19, 2026-10-05)
AUTO_TUNE_REPORT="${AUTO_TUNE_REPORT:-0}"  # 1 = only ASK the binary's auto-tuner what this run would need
                                 # (--auto-tune-report, standard DTFE since 2026-10-13): per snapshot one line
                                 #   AUTO-TUNE-REPORT snap=NNN partition=.. mc=.. predicted_gb=.. budget_gb=.. over_budget=0|1 ...
                                 # after reading the input; nothing is computed or written (the snapshot's .runlog,
                                 # lock and space check are left alone). The launcher's memory check uses it.
DTFE_PRECISION="${DTFE_PRECISION:-single}"  # double = the double-precision binary ./DTFE-double ('make DTFE DOUBLE=1'):
                                 # everything in double from the input read onward, float64 outputs, ~2x the memory
SCRATCH_DIR="${SCRATCH_DIR:-}"   # out-of-core mode (--scratch-dir): the full-resolution grids as mmap'ed files in this
                                 # LOCAL directory instead of RAM (bit-identical results; iCloud paths are refused) --
                                 # the same knob as run_ps_dtfe.sh; the launcher's scratch folder arrives here
SAMPLE_POINTS="${SAMPLE_POINTS:-}"  # a --sample-points file: the standard DTFE interpolant at those points too
                                 # ('.pts_*'). With PARTITION (or an auto-tuned split) as well since 2026-10-13: each
                                 # partition answers the points inside the tetrahedra it owns, equal to one triangulation
                                 # of the whole box to float rounding, so a full-box image plane fits at production
                                 # scale; such a partition is ONE triangulation (no thread sub-boxes), its grids equal a
                                 # run without points to float rounding (the dummy-point padding test is off). A killed
                                 # one resumes (DTFE_RESUME) since 2026-10-14, its finished partitions' points kept
PTS_VEL_GRAD="${PTS_VEL_GRAD:-0}"   # 1 = with SAMPLE_POINTS, also the velocity gradient at each point
TESS_CACHE="${TESS_CACHE:-}"     # a LOCAL folder: reuse the Delaunay tessellation of an identical earlier run
                                 # (--tessellation-cache; created if missing; a partitioned standard run does not
                                 # cache, the binary says so) -- the same knob as run_ps_dtfe.sh
LAMBDA_TH="${LAMBDA_TH:-0.3}"    # eigenvalue threshold of the T-web / V-web classes (--lambda_th), the SAME default as
                                 # run_ps_dtfe.sh (the binary's own default is 0.0: a run without this flag classifies
                                 # differently from a scripted one)
DTFE_RESUME="${DTFE_RESUME:-1}"  # 1 (default) = resume a killed partitioned run, as run_ps_dtfe.sh's PS_RESUME (2026-10-13):
                                 # when the snapshot's checkpoint file exists (<prefix>.runlog.checkpoint/checkpoint
                                 # beside the outputs, or under SCRATCH_DIR; config.sh ps_checkpoint_dir, the same rule
                                 # for both binaries), --resume is passed and only the partitions it lacks are computed.
                                 # The binary refuses, naming the differing settings, when the settings changed; then
                                 # DTFE_RESUME=0 starts over (the new run replaces the checkpoint)

DATA_DIR=""                # default: sim_dir $SIMULATION (config.sh); override with -d
OUTPUT_PREFIX="${OUTPUT_PREFIX:-output}"   # -> <snapdir>/<prefix>.* (2026-10-13; it was always 'output'). Another
                                 # prefix keeps a variant beside the default set, e.g. OUTPUT_PREFIX=output_mc with
                                 # DTFE_EXACT_AVERAGE=0; dtfelib reads 'output.' for standard DTFE. 'ps_*' is refused
                                 # below: PS-DTFE's naming, so an export meant for run_ps_dtfe.sh cannot overwrite
                                 # the phase-space grids with standard ones

# the grids to write; env-overridable as in run_ps_dtfe.sh, e.g. FIELDS="density_a velocity_a" for a lighter run
# (the launcher passes its ticks here; no dispersion_a: that is a phase-space field). The default is the production
# set since 2026-10-13: the T-web and V-web joined it (the 2026-10-12 grids of all 27 snapshots carry them, and the
# void and web figures read them; a run without them left the old '.a_tweb*' of an earlier run beside new grids)
FIELDS="${FIELDS:-density_a velocity_a gradient_a divergence_a shear_a vorticity_a tweb_a vweb_a}"
case " ${FIELDS} " in               # a FIELDS exported for run_ps_dtfe.sh may carry the phase-space dispersion
    *" dispersion"*) echo "Note: the velocity dispersion is a PS-DTFE field; standard DTFE leaves it out"
                     FIELDS=$(printf '%s\n' ${FIELDS} | grep -v '^dispersion' | tr '\n' ' ') ;;
esac
if [ -z "${FIELDS// }" ]; then
    echo "Error: no field standard DTFE computes in FIELDS (the velocity dispersion is a PS-DTFE field)" >&2
    exit 2
fi

usage() { echo "Usage: $0 [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-m] [-e] [snapshot ...]"; }

case "${1:-}" in                                 # -h / --help print the header and run nothing; --version the revision
    -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//'; usage; exit 0 ;;
    --version) git -C "${REPO_ROOT}" describe --always --dirty 2>/dev/null || echo unknown; exit 0 ;;
esac
# OUTPUT_PREFIX (above): one word, and not PS-DTFE's 'ps_*' naming (after --help, which a stray export must not block)
case "${OUTPUT_PREFIX}" in
    ""|*/*|*" "*|.*)
        echo "Error: OUTPUT_PREFIX must be one word without '/' or spaces (got '${OUTPUT_PREFIX}')." >&2; exit 2 ;;
    ps_*)
        echo "Error: OUTPUT_PREFIX='${OUTPUT_PREFIX}' is PS-DTFE's naming (an OUTPUT_PREFIX exported for run_ps_dtfe.sh?):" >&2
        echo "       standard DTFE would write its grids over the phase-space ones. Use another prefix (default 'output')." >&2
        exit 2 ;;
esac

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
data_root_problem "${DATA_DIR}" || exit 2     # preflight (2026-10-11): the data disk unmounted, said up front

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
# DTFE_METAL=auto (the default): the GPU when this binary was built with one (2026-10-13; config.sh)
gpu_auto=""
if [ "${DTFE_METAL}" = "auto" ]; then
    DTFE_METAL="$(gpu_build_default "${REPO_ROOT}" "${DTFE_BIN}")"
    gpu_auto=" (DTFE_METAL unset: ${DTFE_BIN#./} is $([ "${DTFE_METAL}" = "1" ] && echo "a GPU" || echo "a CPU-only") build; DTFE_METAL=0 or 1 chooses)"
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
if [ "${DRY_RUN}" != "1" ] && command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -m -w $$ &
fi

# The run-log contract (2026-10-08), as run_ps_dtfe.sh: while the binary runs its log is
# 'output.runlog.partial'; exit 0 renames it onto 'output.runlog' (the log of the grids now on disk) and
# removes a stale 'output.runlog.failed'; a failure or a stop renames it to 'output.runlog.failed' and
# leaves 'output.runlog' alone. Teeing straight into 'output.runlog' let a re-run that failed early, or
# was stopped or killed, replace the log of the grids still on disk (the Runs browser then showed them as
# a failed run with the new settings). A '.runlog.partial' left behind is a run killed outright (SIGKILL),
# or one still going.
live_log=""; live_rc=""            # the .partial being written, and its binary's exit status once known
live_lock=""                       # this driver's 'output.runlog.lock' (config.sh run_lock_take), while held
sup_marker=""; sup_oldlog=""       # the run's start marker and a copy of the old run log (config.sh supersede_extras)
sup_clean() { rm -f ${sup_marker:+"${sup_marker}"} ${sup_oldlog:+"${sup_oldlog}"}; sup_marker=""; sup_oldlog=""; }
finish_runlog() {                  # <the .partial> <the binary's exit status>
    local part="$1" final="${1%.partial}"
    [ -f "${part}" ] || return 0
    # strip ANSI colour codes from the kept log (a temp file: BSD and GNU sed disagree on -i)
    if sed -E $'s/\033\\[[0-9;]*m//g' "${part}" > "${part}.tmp" 2>/dev/null; then
        mv "${part}.tmp" "${part}"
    else
        rm -f "${part}.tmp"
    fi
    if [ "$2" = "0" ]; then
        mv -f "${part}" "${final}" && rm -f "${final}.failed"
    else
        mv -f "${part}" "${final}.failed"
    fi
}
# a stop (the launcher's Stop: SIGTERM to the binary, then to this script; ^C: SIGINT) runs the trap once
# the binary's pipeline has ended; the signal is then re-raised, so a caller sees a stop, not a failure
on_stop() {                        # <signal name>
    trap - "$1"
    if [ -n "${live_log}" ] && [ -f "${live_log}" ]; then
        finish_runlog "${live_log}" "${live_rc:-1}"
        [ "${live_rc:-1}" = "0" ] || echo "  Stopped: snapshot ${n_str} did not finish; its log is ${live_log%.partial}.failed"
    fi
    run_lock_release "${live_lock}"
    sup_clean
    kill -s "$1" $$
}
trap 'on_stop INT' INT; trap 'on_stop TERM' TERM; trap 'on_stop HUP' HUP
trap 'run_lock_release "${live_lock}"; sup_clean' EXIT   # any other way out (an error, set -u) frees the lock too

echo "Starting DTFE processing..."
echo "Data directory: ${DATA_DIR}"
echo "Grid size: ${GRID_SIZE}   Fields: ${FIELDS}   Output prefix: ${OUTPUT_PREFIX}"
if [ "${DTFE_EXACT_AVERAGE}" = "1" ]; then
    echo "Cell averages: exact (--exact-average)"
else
    echo "Cell averages: Monte-Carlo sample means (DTFE_EXACT_AVERAGE=${DTFE_EXACT_AVERAGE})"
fi
echo "'_a' interpolation: $([ "${DTFE_METAL}" = "1" ] && echo "GPU (--gpu)" || echo "CPU")${gpu_auto}"
[ -n "${SCRATCH_DIR}" ] && echo "Scratch directory: ${SCRATCH_DIR} (grids out of core)"
echo ""

n_ok=0; n_skipped=0; n_failed=0; n_planned=0
for i in "${SNAPSHOTS[@]}"; do
    n_str=$(printf "%03d" "$i")

    input_dir="${DATA_DIR}/${INPUT_SUBDIR}_${n_str}"
    input_file="${input_dir}/combined_${n_str}.hdf5"
    # output prefix: files land as <snapdir>/output.a_den etc. (the 'dtfe' convention in dtfelib; OUTPUT_PREFIX)
    output_root="${input_dir}/${OUTPUT_PREFIX}"

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
    sp_args=()
    [ -n "${SAMPLE_POINTS}" ] && sp_args=(--sample-points "${SAMPLE_POINTS}")
    [ -n "${SAMPLE_POINTS}" ] && [ "${PTS_VEL_GRAD}" = "1" ] && sp_args+=(--pts-vel-grad)
    tess_args=()
    if [ -n "${TESS_CACHE}" ]; then
        mkdir -p "${TESS_CACHE}"
        tess_args=(--tessellation-cache "${TESS_CACHE}")
    fi
    # a killed partitioned run left its checkpoint (the binary removes it once the outputs are in place): resume it,
    # as run_ps_dtfe.sh does (2026-10-13). The FILE, not the folder: a folder with no 'checkpoint' in it (a kill
    # during the first write) holds nothing to resume, and the binary refuses '--resume' there. Not for a report
    # (AUTO_TUNE_REPORT=1 computes nothing, so there is nothing to resume), as run_ps_dtfe.sh
    resume_args=()
    ckpt_dir="$(ps_checkpoint_dir "${output_root}" "${SCRATCH_DIR}")"
    if [ "${AUTO_TUNE_REPORT}" != "1" ] && [ -f "${ckpt_dir}/checkpoint" ]; then
        if [ "${DTFE_RESUME}" = "0" ]; then
            echo "  Note: a killed run's checkpoint is in ${ckpt_dir}; DTFE_RESUME=0, so this run starts over and replaces it."
        else
            echo "  Resuming: a killed run's checkpoint is in ${ckpt_dir}; passing --resume, only the missing partitions are computed (DTFE_RESUME=0 starts over)."
            resume_args=(--resume)
        fi
    fi

    # the tee pipe hides the terminal from the binary's isatty() check: force colours through it when
    # this script itself runs on a terminal (the runlog is de-ANSI'd below), as run_ps_dtfe.sh does
    [ -t 1 ] && export CLICOLOR_FORCE=1
    echo "  Running DTFE on ${input_file}..."
    # the run log beside the outputs (<snapdir>/<prefix>.runlog), as run_ps_dtfe.sh keeps one: the
    # GUI's Runs browser reads the settings, build stamp, wall time and peak memory back from it
    run_log="${output_root}.runlog"
    # a report writes nothing next to the data: not even over the snapshot's real run log
    [ "${AUTO_TUNE_REPORT}" = "1" ] && [ "${DRY_RUN}" != "1" ] && run_log="$(mktemp -t dtfe-auto-tune-report.XXXXXX)"
    report_args=()
    [ "${AUTO_TUNE_REPORT}" = "1" ] && report_args=(--auto-tune-report)
    SECONDS=0
    cmd=("${DTFE_BIN}" "${input_file}" "${output_root}"
        --grid ${GRID_SIZE}
        --padding ${PADDING}
        --periodic
        ${part_args[@]+"${part_args[@]}"}
        --field ${FIELDS}
        --lambda_th ${LAMBDA_TH}
        ${scratch_args[@]+"${scratch_args[@]}"}
        ${tess_args[@]+"${tess_args[@]}"}
        ${sp_args[@]+"${sp_args[@]}"}
        ${metal_args[@]+"${metal_args[@]}"}
        ${exact_args[@]+"${exact_args[@]}"}
        ${report_args[@]+"${report_args[@]}"}
        ${resume_args[@]+"${resume_args[@]}"})
    if [ "${DRY_RUN}" = "1" ]; then                 # DRY_RUN=1: the expanded command and the paths, nothing run
        echo "  DRY RUN -- would run:"
        printf '    %q' "${cmd[@]}"; echo
        echo "  input:   ${input_file} (exists)"
        echo "  outputs: ${output_root}.* and the run log ${run_log}"
        n_planned=$((n_planned + 1))
        echo ""
        continue
    fi
    # A report (AUTO_TUNE_REPORT=1) computes and writes nothing: no space check, no lock (the launcher's memory check
    # runs one beside a live run), no run-log contract -- the binary's one line, from a temp log (as run_ps_dtfe.sh)
    if [ "${AUTO_TUNE_REPORT}" = "1" ]; then
        "${cmd[@]}" 2>&1 | tee "${run_log}"
        rc=${PIPESTATUS[0]}
        report=$(grep -a '^AUTO-TUNE-REPORT' "${run_log}" | head -1)
        rm -f "${run_log}"
        if [ "${rc}" -eq 0 ] && [ -n "${report}" ]; then
            echo "  AUTO-TUNE-REPORT snap=${n_str} ${report#AUTO-TUNE-REPORT }"
            n_ok=$((n_ok + 1))
        else
            echo "  Error: no auto-tune report for snapshot ${n_str} (exit ${rc})"
            n_failed=$((n_failed + 1))
        fi
        echo ""
        continue
    fi
    # room for the grids, then the snapshot's single-instance lock (2026-10-11, config.sh; as run_ps_dtfe.sh)
    bpv=4; [ "${DTFE_PRECISION}" = "double" ] && bpv=8
    if ! output_space_problem "${input_dir}" "$(expected_output_bytes "${GRID_SIZE%% *}" "${bpv}" ${FIELDS})"; then
        echo "  Error processing snapshot ${n_str}: no room for its grids (nothing was run)"
        n_failed=$((n_failed + 1)); echo ""; continue
    fi
    if ! run_lock_take "${run_log}.lock"; then
        echo "  Error processing snapshot ${n_str}: refused, another run holds it (nothing was run)"
        n_failed=$((n_failed + 1)); echo ""; continue
    fi
    live_lock="${run_log}.lock"
    # what an earlier run with this prefix left, recorded before this one starts: what it does not rewrite is
    # moved aside once it has exited 0 (config.sh supersede_extras, 2026-10-13)
    sup_clean
    sup_marker="$(mktemp "${TMPDIR:-/tmp}/dtfe-run-start.XXXXXX" 2>/dev/null)" || sup_marker=""
    sup_list="$(supersede_snapshot "${output_root}")"
    if [ -n "${sup_list}" ] && [ -f "${run_log}" ]; then
        sup_oldlog="$(mktemp "${TMPDIR:-/tmp}/dtfe-old-runlog.XXXXXX" 2>/dev/null)" && cp -p "${run_log}" "${sup_oldlog}" \
            || sup_oldlog=""
    fi
    # the binary's output goes to '<prefix>.runlog.partial' until it exits (the run-log contract above)
    live_log="${run_log}.partial"; live_rc=""
    ${TIME_WRAP[@]+"${TIME_WRAP[@]}"} "${cmd[@]}" 2>&1 | tee "${live_log}"
    rc=${PIPESTATUS[0]}                 # the binary's status, taken BEFORE anything else runs a pipe
    live_rc="${rc}"

    # de-ANSI'd, the log becomes 'output.runlog' on success and 'output.runlog.failed' otherwise; the
    # peak memory below is read from whichever it became
    finish_runlog "${live_log}" "${rc}"
    live_log=""
    [ "${rc}" -eq 0 ] && supersede_extras "${output_root}" "${sup_list}" "${sup_marker}" "${sup_oldlog}"
    sup_clean
    run_lock_release "${live_lock}"; live_lock=""
    [ "${rc}" -eq 0 ] || run_log="${run_log}.failed"
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
        echo "  Error processing snapshot ${n_str} (its log: ${run_log})"
        # exit 3: the run finished but its output folder could not be written, so the binary put the grids in its
        # rescue folder; the log's OUTPUT RESCUE line names it and the mv that puts them in place (2026-10-11)
        [ "${rc}" -eq 3 ] && echo "  The grids were computed and saved to the rescue folder: see the OUTPUT RESCUE line in ${run_log}"
        n_failed=$((n_failed + 1))
    fi
    echo ""
done

# Honest exit status, as run_ps_dtfe.sh: a failed snapshot, or nothing processed, exits 1 (before 2026-10-05
# this script always exited 0, so the launcher read a failed standard-DTFE job as done). A partial skip stays 0.
if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY RUN: ${n_planned} snapshot(s) planned, ${n_skipped} skipped for missing inputs; nothing was run."
    [ "${n_planned}" -gt 0 ] && exit 0
    echo "ERROR: nothing to run (looked under '${DATA_DIR}')." >&2
    exit 1
fi
echo "DTFE processing complete: ${n_ok} processed, ${n_skipped} skipped, ${n_failed} failed."
if [ "${n_failed}" -gt 0 ]; then
    echo "ERROR: ${n_failed} snapshot(s) failed; see the messages above and their .runlog.failed files." >&2
    exit 1
fi
if [ "${n_ok}" -eq 0 ]; then
    echo "ERROR: no snapshot was processed (${n_skipped} skipped for missing inputs) -- check DATA_ROOT/SIMULATION (looked under '${DATA_DIR}')." >&2
    exit 1
fi
