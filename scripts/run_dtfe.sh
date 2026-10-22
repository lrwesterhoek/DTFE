#!/usr/bin/env bash
#
# Batch-run standard DTFE over a set of TNG snapshots (redshifts).
#
# Companion to download_snapshots.sh, which fetches the snapshot files first.
# DTFE_DATA_ROOT (the folder that holds the simulations; -d names a simulation's folder instead) and the simulation
# (-s or DTFE_SIM) are required; the
# shared defaults (SNAPSHOTS, GRID_SIZE) live in config.sh. PADDING (or DTFE_PADDING), the
# padding in mean particle spacings, defaults to 15 here, not config.sh's 50 (the PS-DTFE scripts' value).
#
# Usage:
#   ./run_dtfe.sh [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-m] [-e] [snapshot ...]
#     -m   run the '_a' interpolation on the GPU (same as DTFE_METAL=1). The default (DTFE_METAL unset, 'auto')
#          is the GPU when the binary was built with one (the Makefile's o/.gpu_mode_metal|cuda|hip stamp;
#          o_d for DTFE_PRECISION=double), else the CPU; DTFE_METAL=0 runs the CPU on a GPU build.
#     -e   every '_a' cell the EXACT average of the linear interpolant over the cell (--exact-average). The
#          DEFAULT (DTFE_EXACT_AVERAGE=1, the production standard: no sampling noise, the mass exact to ~3e-8);
#          -e overrides an exported DTFE_EXACT_AVERAGE=0. DTFE_EXACT_AVERAGE=0 gives the Monte-Carlo sample mean (1.1-1.7x
#          faster at 512^3 on the GPU; the GPU is what makes the exact average affordable: ~6.6x the CPU's speed)
#   Production defaults: GRID_SIZE 512 (config.sh), padding 15, the GPU when built, the exact cell
#   average, FIELDS = density, velocity, its gradient / divergence / shear / vorticity, the T-web and the V-web,
#   LAMBDA_TH 0.3.
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
#          the partitions run one after the other); nothing is computed or written (as run_ps_dtfe.sh).
#   Run logs, beside the outputs: output.runlog belongs to the grids on disk, output.runlog.failed to the
#   last failed or stopped attempt, output.runlog.partial to a run still going (or killed outright).
#   While a snapshot runs, <prefix>.runlog.lock (a folder holding the driver's pid) refuses a second driver on
#   it; a lock left by a killed driver is removed with a note. Before anything runs: the data disk must be
#   there (exit 2 otherwise) and the output folder writable with room for the grids (DTFE_SPACE_CHECK=0 skips
#   the space half).
#   DTFE_RESUME (default 1): a partitioned run killed part-way resumes, as run_ps_dtfe.sh's PS_RESUME. When the
#   snapshot's checkpoint (<prefix>.runlog.checkpoint, or in SCRATCH_DIR) exists, the next run passes --resume and
#   computes only the partitions it lacks; DTFE_RESUME=0 starts over (the new run replaces the checkpoint). A run with
#   SAMPLE_POINTS too (the binary saves each finished partition's point answers beside the grids).
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # the binaries and Makefile live one level up
# Padding (mean interparticle spacings) of each partition and of each thread's sub-box: standard DTFE's own default
# is 15, not config.sh's 50 (which stays the PS-DTFE scripts' value). 15 is complete -- no '*.badGridPoints*' file --
# on the z = 0 snapshots of TNG50/100/300-3-Dark, the hardest case (the emptiest voids give the largest Delaunay
# circumspheres at the partition edges), and 22-29% faster than 20 there; 10 is not complete at z = 0, and 50 costs
# 33x the vertices. A PADDING or DTFE_PADDING from the caller still wins: it is read here, before
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
                                 # the drivers' rule) = 1 when the binary is a GPU build (config.sh
                                 # gpu_build_default), resolved below once DTFE_PRECISION has chosen the binary
DTFE_EXACT_AVERAGE="${DTFE_EXACT_AVERAGE:-1}"  # 1 (default, the production standard) = exact cell
                                 # averages (--exact-average: no sampling noise, mass exact to ~3e-8); 0 = the
                                 # Monte-Carlo sample mean
DRY_RUN="${DRY_RUN:-0}"          # 1 = print the commands and paths, run nothing
AUTO_TUNE_REPORT="${AUTO_TUNE_REPORT:-0}"  # 1 = only ASK the binary's auto-tuner what this run would need
                                 # (--auto-tune-report): per snapshot one line
                                 #   AUTO-TUNE-REPORT snap=NNN partition=.. mc=.. predicted_gb=.. budget_gb=.. over_budget=0|1 ...
                                 # after reading the input; nothing is computed or written (the snapshot's .runlog,
                                 # lock and space check are left alone). The launcher's memory check uses it.
DTFE_PRECISION="${DTFE_PRECISION:-single}"  # double = the double-precision binary ./DTFE-double ('make DTFE DOUBLE=1'):
                                 # everything in double from the input read onward, float64 outputs, ~2x the memory
SCRATCH_DIR="${SCRATCH_DIR:-}"   # out-of-core mode (--scratch-dir): the full-resolution grids as mmap'ed files in this
                                 # LOCAL directory instead of RAM (bit-identical results; iCloud paths are refused) --
                                 # the same knob as run_ps_dtfe.sh; the launcher's scratch folder arrives here
SAMPLE_POINTS="${SAMPLE_POINTS:-}"  # a --sample-points file: the standard DTFE interpolant at those points too
                                 # ('.pts_*'). With PARTITION (or an auto-tuned split) as well: each
                                 # partition answers the points inside the tetrahedra it owns, equal to one triangulation
                                 # of the whole box to float rounding, so a full-box image plane fits at production
                                 # scale; such a partition is ONE triangulation (no thread sub-boxes), its grids equal a
                                 # run without points to float rounding (the dummy-point padding test is off). A killed
                                 # one resumes (DTFE_RESUME), its finished partitions' points kept
PTS_VEL_GRAD="${PTS_VEL_GRAD:-0}"   # 1 = with SAMPLE_POINTS, also the velocity gradient at each point
TESS_CACHE="${TESS_CACHE:-}"     # a LOCAL folder: reuse the Delaunay tessellation of an identical earlier run
                                 # (--tessellation-cache; created if missing; a partitioned run caches one file per
                                 # thread sub-box of each partition, one per partition with SAMPLE_POINTS) -- the
                                 # same knob as run_ps_dtfe.sh
LAMBDA_TH="${LAMBDA_TH:-0.3}"    # eigenvalue threshold of the T-web / V-web classes (--lambda_th), the SAME default as
                                 # run_ps_dtfe.sh (the binary's own default is 0.0: a run without this flag classifies
                                 # differently from a scripted one)
DTFE_RESUME="${DTFE_RESUME:-1}"  # 1 (default) = resume a killed partitioned run, as run_ps_dtfe.sh's PS_RESUME:
                                 # when the snapshot's checkpoint file exists (<prefix>.runlog.checkpoint/checkpoint
                                 # beside the outputs, or under SCRATCH_DIR; config.sh ps_checkpoint_dir, the same rule
                                 # for both binaries), --resume is passed and only the partitions it lacks are computed.
                                 # The binary refuses, naming the differing settings, when the settings changed; then
                                 # DTFE_RESUME=0 starts over (the new run replaces the checkpoint)

DATA_DIR=""                # default: sim_dir $SIMULATION (config.sh); override with -d
OUTPUT_PREFIX="${OUTPUT_PREFIX:-output}"   # -> <snapdir>/<prefix>.* (default 'output'). Another
                                 # prefix keeps a variant beside the default set, e.g. OUTPUT_PREFIX=output_mc with
                                 # DTFE_EXACT_AVERAGE=0; dtfelib reads 'output.' for standard DTFE. 'ps_*' is refused
                                 # below: PS-DTFE's naming, so an export meant for run_ps_dtfe.sh cannot overwrite
                                 # the phase-space grids with standard ones

# the grids to write; env-overridable as in run_ps_dtfe.sh, e.g. FIELDS="density_a velocity_a" for a lighter run
# (the launcher passes its ticks here; no dispersion_a: that is a phase-space field). The default is the production
# set, the T-web and V-web included (the grids of all 27 snapshots carry them, and the void and web figures read
# them)
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

driver_help_version "${BASH_SOURCE[0]}" "${1:-}"   # -h / --help print the header and run nothing; --version the revision
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
[ -n "${DATA_DIR}" ] || require_data_root   # -d names the simulation's folder itself
require_simulation

snapshot_args "$@"              # positional arguments override the default snapshot list (config.sh)

[ -z "${DATA_DIR}" ] && DATA_DIR="$(sim_dir "${SIMULATION}")"
data_root_problem "${DATA_DIR}" || exit 2     # preflight: the data disk unmounted, said up front

cd "$REPO_ROOT" || exit 1

driver_binary DTFE_BIN DTFE "${DTFE_PRECISION}"      # ./DTFE or ./DTFE-double (config.sh)
gpu_auto_resolve DTFE_METAL "${DTFE_BIN}"             # DTFE_METAL=auto: the GPU when this binary was built with one
time_wrap_resolve                                     # TIME_WRAP: the optional peak-memory wrapper
keep_awake "${DRY_RUN}"
# the run-log contract (config.sh): 'output.runlog.partial' while the binary runs, then 'output.runlog' on exit 0
# or 'output.runlog.failed' on a failure or a stop; a stop also releases the lock and re-raises the signal
runlog_contract_start

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
    # n_str, input_dir, input_file and output_root (config.sh): files land as <snapdir>/output.a_den etc. (the
    # 'dtfe' convention in dtfelib; OUTPUT_PREFIX)
    if ! snapshot_start "$i"; then
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
    # a killed partitioned run left its checkpoint: resume it (config.sh), as run_ps_dtfe.sh does
    resume_args_for "${output_root}" DTFE_RESUME "${DTFE_RESUME}"

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
        dry_run_print "${input_file}" "${output_root}" "${run_log}" "${cmd[@]}"
        n_planned=$((n_planned + 1))
        echo ""
        continue
    fi
    # A report (AUTO_TUNE_REPORT=1) computes and writes nothing: no space check, no lock (the launcher's memory check
    # runs one beside a live run), no run-log contract -- the binary's one line, from a temp log (as run_ps_dtfe.sh)
    if [ "${AUTO_TUNE_REPORT}" = "1" ]; then
        "${cmd[@]}" 2>&1 | tee "${run_log}"
        rc=${PIPESTATUS[0]}
        if auto_tune_report_line "${run_log}" "${rc}"; then n_ok=$((n_ok + 1)); else n_failed=$((n_failed + 1)); fi
        echo ""
        continue
    fi
    # room for the grids, then the snapshot's single-instance lock (config.sh; as run_ps_dtfe.sh)
    if ! snapshot_preflight "${input_dir}" "${run_log}" "${DTFE_PRECISION}"; then
        n_failed=$((n_failed + 1)); echo ""; continue
    fi
    # what an earlier run with this prefix left, recorded before this one starts: what it does not rewrite is
    # moved aside once it has exited 0 (config.sh sup_record, run_finish)
    sup_record "${output_root}" "${run_log}"
    # the binary's output goes to '<prefix>.runlog.partial' until it exits (the run-log contract above)
    live_log="${run_log}.partial"; live_rc=""
    ${TIME_WRAP[@]+"${TIME_WRAP[@]}"} "${cmd[@]}" 2>&1 | tee "${live_log}"
    rc=${PIPESTATUS[0]}                 # the binary's status, taken BEFORE anything else runs a pipe
    live_rc="${rc}"

    # the log becomes 'output.runlog' or 'output.runlog.failed', the earlier run's leftovers move aside on exit 0
    # (config.sh run_finish); the peak memory below is read from whichever log it became
    run_finish "${rc}" "${output_root}"
    [ "${rc}" -eq 0 ] || run_log="${run_log}.failed"
    peak_txt="$(peak_memory_text "${run_log}")"
    [ -n "${peak_txt}" ] && echo "  Peak memory: ${peak_txt}"
    if snapshot_outcome "${rc}" "${run_log}"; then n_ok=$((n_ok + 1)); else n_failed=$((n_failed + 1)); fi
    echo ""
done

# Honest exit status, as run_ps_dtfe.sh: a failed snapshot, or nothing processed, exits 1 (so the launcher does
# not read a failed standard-DTFE job as done). A partial skip stays 0.
[ "${DRY_RUN}" = "1" ] && dry_run_exit "${n_planned}" "${n_skipped}" "${DATA_DIR}"
echo "DTFE processing complete: ${n_ok} processed, ${n_skipped} skipped, ${n_failed} failed."
if [ "${n_failed}" -gt 0 ]; then
    echo "ERROR: ${n_failed} snapshot(s) failed; see the messages above and their .runlog.failed files." >&2
    exit 1
fi
if [ "${n_ok}" -eq 0 ]; then
    echo "ERROR: no snapshot was processed (${n_skipped} skipped for missing inputs) -- check DATA_ROOT/SIMULATION (looked under '${DATA_DIR}')." >&2
    exit 1
fi
