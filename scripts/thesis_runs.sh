#!/usr/bin/env bash
# The runs that produced the thesis grids, image planes and run logs on the data disk, as a recipe in the
# repository (2026-10-11; they lived only in ~/DTFE-rerun/run_all.sh and run_tng50z0_tng100caustics.sh, with the
# repository and data paths hard-coded). Paths come from this script's location and scripts/config.sh
# (DTFE_DATA_ROOT, default the Samsung T7), so the recipe survives a move of either.
#
# Usage:
#   scripts/thesis_runs.sh STAGE [STAGE ...]     run the stages in the order given
#   DRY_RUN=1 scripts/thesis_runs.sh all         print every command (the drivers print theirs), run nothing
#   scripts/thesis_runs.sh --help                this header, nothing run
# Stages:
#   production  the 2026-10-03 rerun: TNG100-3-Dark, TNG50-3-Dark, TNG300-3-Dark, every snapshot recomputed
#               (FORCE=1) by run_ps_pipeline.sh with the T-web and V-web, a 52 GB memory budget, the full-grid
#               accumulators in SCRATCH_DIR, 512^3 grids and image planes of NU = 4096 / 2048 / 4096 pixels
#               (TNG50 at 2048: its z=0 streams need ~92 GB at 4096), then each simulation's point-evaluated
#               figures (plot_pointeval.py)
#   tng50z0     the 2026-10-07 redo of TNG50-3-Dark snapshot 099 (the production run was killed for memory at
#               a 52 GB budget): the same settings at a 40 GB budget, then its z=0 figures (fields, void
#               profiles at the grid and at 2.1625 Mpc, the population)
#   caustics    TNG100-3-Dark snapshot 099 with the caustic classes (PS_CAUSTICS=1), density + T-web only, under
#               its OWN prefix ps_caustic (the production ps_output.* grids stay untouched), budget 44 GB, then
#               the caustic skeleton and the web-streams figures
#   all         production tng50z0 caustics
# Run ONE stage at a time and nothing else heavy beside it: each run peaks at 40-50 GB, and two together got
# both SIGKILLed (2026-10-03). The drivers' <prefix>.runlog.lock refuses a second run of the same snapshot.
# A full 'all' takes about a day on the 64 GB / 10-core M-series machine (TNG50 z=0 alone ~8 h).
#
# Knobs (environment): SCRATCH_DIR (default ~/DTFE-scratch; LOCAL, not iCloud), PY (the python for the
# figures), DTFE_DATA_ROOT (config.sh). Everything else is the recipe and fixed here on purpose.
# The memory budget: DTFE_MEM_BUDGET_GB is an upper cap; the auto-tuner also caps the budget by the memory free
# when a run starts, so the partition plan (and the wall time) depends on what else is open on the machine.
# Each run log records the plan it used ('memory budget: ... GB' and the AUTO-TUNE line).
# Exit status: 0 when every stage's runs succeeded, 1 otherwise (the stages after a failure still run, as the
# original scripts did), 2 for a usage error.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//'; }
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage; echo "Error: name the stage(s) to run (production, tng50z0, caustics or all)." >&2; exit 2 ;;
esac
stages=()
for s in "$@"; do
    case "${s}" in
        production|tng50z0|caustics) stages+=("${s}") ;;
        all) stages+=(production tng50z0 caustics) ;;
        *) echo "Error: unknown stage '${s}' (production, tng50z0, caustics, all)." >&2; exit 2 ;;
    esac
done

DRY_RUN="${DRY_RUN:-0}"
export DRY_RUN                      # the drivers print their commands and run nothing
PY="${PY:-$(command -v /opt/homebrew/bin/python3 || command -v python3)}"
export SCRATCH_DIR="${SCRATCH_DIR:-${HOME}/DTFE-scratch}"
export DTFE_PRECISION=single         # the thesis grids are single precision, whatever the caller's shell says (2026-10-11)
export MPLBACKEND=Agg
# The thesis field set: the pipeline's default list plus the T-web and V-web classifications.
THESIS_FIELDS="density_a velocity_a gradient_a divergence_a shear_a vorticity_a dispersion_a tweb_a vweb_a"

data_root_problem "${DATA_ROOT}" || exit 2
if [ "${DRY_RUN}" != "1" ]; then
    mkdir -p "${SCRATCH_DIR}" || { echo "Error: cannot create SCRATCH_DIR '${SCRATCH_DIR}'" >&2; exit 2; }
fi
cd "${REPO_ROOT}" || exit 2

stamp() { echo "=== $(date '+%Y-%m-%d %H:%M:%S') $*"; }
# fig <command...>: a figure script; printed only under DRY_RUN (they have no dry run of their own)
fig() { echo "+ $*"; [ "${DRY_RUN}" = "1" ] || "$@"; }
overall=0

stage_production() {
    local job sim nu rc
    for job in "TNG100-3-Dark 4096" "TNG50-3-Dark 2048" "TNG300-3-Dark 4096"; do
        sim="${job% *}"; nu="${job#* }"
        stamp "${sim}: grids + ${nu}^2 image plane"
        rc=0
        DTFE_MEM_BUDGET_GB=52 FORCE=1 FIELDS="${THESIS_FIELDS}" SIMS="${sim}" NU="${nu}" GRID_SIZE=512 \
            "${SCRIPT_DIR}/run_ps_pipeline.sh" || rc=$?
        stamp "${sim}: exit ${rc}"
        [ "${rc}" = "0" ] || overall=1
        stamp "${sim}: figures"
        export DTFE_SIM="${sim}"
        fig "${PY}" python/plot/plot_pointeval.py --sims "${sim}" \
            --plane "$(sim_dir "${sim}")/hires_plane_z${nu}.json" || overall=1
        unset DTFE_SIM
    done
}

stage_tng50z0() {
    local sim=TNG50-3-Dark rc=0
    stamp "${sim} 099: grids + 2048^2 image plane (budget 40 GB)"
    DTFE_MEM_BUDGET_GB=40 FORCE=1 \
        SAMPLE_POINTS="$(sim_dir "${sim}")/hires_plane_z2048.bin" PS_VOLUME_WEIGHTED=1 PTS_VEL_GRAD=1 \
        FIELDS="${THESIS_FIELDS}" \
        "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g 512 -m 99 || rc=$?
    stamp "${sim} 099: exit ${rc}"
    if [ "${rc}" != "0" ]; then overall=1; return; fi
    stamp "${sim} figures (z=0)"
    export DTFE_SIM="${sim}"
    fig "${PY}" python/plot/plot_pointeval.py --sims "${sim}" --snaps 99 \
        --plane "$(sim_dir "${sim}")/hires_plane_z2048.json" || overall=1
    fig "${PY}" python/plot/plot_void_profiles.py --sim "${sim}" --method ps --snaps 99 || overall=1
    fig "${PY}" python/plot/plot_void_profiles.py --sim "${sim}" --method ps --snaps 99 --smooth-mpc 2.1625 || overall=1
    fig "${PY}" python/plot/plot_void_population.py --sim "${sim}" --method ps --smooth-mpc 2.1625 || overall=1
    unset DTFE_SIM
}

stage_caustics() {
    local sim=TNG100-3-Dark rc=0
    stamp "${sim} 099: caustics (prefix ps_caustic, budget 44 GB)"
    DTFE_MEM_BUDGET_GB=44 PS_CAUSTICS=1 OUTPUT_PREFIX=ps_caustic FIELDS="density_a tweb_a" \
        "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g 512 -m 99 || rc=$?
    stamp "${sim} 099 caustics: exit ${rc}"
    if [ "${rc}" != "0" ]; then overall=1; return; fi
    stamp "${sim} caustic skeleton (z=0)"
    export DTFE_SIM="${sim}"
    fig "${PY}" python/plot/plot_caustic_skeleton.py --sim "${sim}" --prefix ps_caustic --snaps 99 --voids || overall=1
    fig "${PY}" python/plot/plot_web_streams.py --sim "${sim}" --prefix ps_caustic --snaps 99 --web tweb || overall=1
    unset DTFE_SIM
}

for s in "${stages[@]}"; do
    "stage_${s}"
done
stamp "ALL DONE ($([ "${overall}" = "0" ] && echo "every run succeeded" || echo "WITH FAILURES: see the exit lines above"))"
exit "${overall}"
