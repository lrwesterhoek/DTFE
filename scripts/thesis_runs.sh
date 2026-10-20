#!/usr/bin/env bash
# The runs that produced the thesis grids, image planes and run logs on the data disk, as a recipe in the
# repository. Paths come from this script's location and scripts/config.sh
# (DTFE_DATA_ROOT, default the Samsung T7), so the recipe survives a move of either.
# The grids on the data disk are those of ps27 (the PS-DTFE run of all 27 snapshots, after run_psrerun.sh) and
# dtfe27 (standard DTFE on all 27 with the exact cell average, the production standard). 'production' and
# 'tng50z0' are the historical runs those replaced: refused unless HISTORICAL=1.
#
# Usage:
#   scripts/thesis_runs.sh STAGE [STAGE ...]     run the stages in the order given
#   DRY_RUN=1 scripts/thesis_runs.sh all         print every command (the drivers print theirs), run nothing
#   scripts/thesis_runs.sh --help                this header, nothing run
# Stages (the production grids):
#   ps27        PS-DTFE on all 27 snapshots (z = 20 ... 0: 000 004 017 025 033 040 050 067 099 of TNG100-3-Dark,
#               TNG300-3-Dark, TNG50-3-Dark, in that order): 512^3, nine fields with the T-web / V-web, nSub 3,
#               Eulerian padding 50, vertex masses, volume-weighted moments, GPU, scratch accumulators, a 52 GB
#               budget, auto split, plus the caustic classes (--ps-caustics) and the 4096^2 image plane of every box
#               with velocity gradients (hires_plane_z4096; made from snapdir_000 when missing); the binary's own
#               Lagrangian padding; ORDERED_MERGE=1, the drivers' and the binary's default (the staged merge in
#               partition index order: bit-reproducible sums, no thread idling behind a slow partition). Each
#               snapshot's old ps_output.* files are first MOVED (never deleted) to <data
#               root>/_backup/ps_output_<date>/<family>/<sim>/snapdir_NNN/ (same volume: a rename; an existing backup
#               there is not moved again, nothing is overwritten); a failed run with no new ps_output.a_den gets them
#               moved back. A snapshot whose ps_output.runlog shows this run (the caustics, the 4096^2 plane and the
#               ordered merge) is skipped: FORCE=1 reruns it. So a set merged in completion order is rerun by the next
#               ps27, and a ps27 cut short continues where it stopped without FORCE (which would redo the snapshots
#               already rerun)
#   dtfe27      standard DTFE (run_dtfe.sh) on the same 27 snapshots, TNG50-3-Dark (4x4x4 partitions, 4 at a
#               time), TNG300-3-Dark (5x5x5, 10) and TNG100-3-Dark (3x3x3, 10) in that order: 512^3, GPU, density,
#               velocity, its gradient / divergence / shear / vorticity, the T-web and V-web at lambda_th 0.3, no
#               scratch dir, the exact cell average (--exact-average) and padding 15, the production standard. The
#               old output.* files are moved to <data root>/_backup/output_<date>/<family>/<sim>/snapdir_NNN/ first, as
#               in ps27. A snapshot whose output.runlog shows the exact average at padding 15 is skipped (FORCE=1 reruns
#               it); sampled grids are not, so they are replaced
#   dtfe27planes  standard-DTFE 4096^2 IMAGE PLANES of the same 27 snapshots: the plane files ps27 uses
#               (<sim>/hires_plane_z4096.bin, made from snapdir_000 when missing, as ps27 makes them), evaluated at
#               the points (--sample-points) with the velocity gradient (--pts-vel-grad), at dtfe27's production
#               settings (512^3, padding 15, its splits, GPU, the exact cell average), into the DEDICATED prefix
#               output_plane: the output.* grids of dtfe27 are never read, moved or replaced. The binary needs one
#               grid field, so the smallest is asked for: density_a (one float per cell; at the exact average it is
#               dtfe27's output.a_den to rounding, a free cross-check). Under --sample-points every partition is
#               ONE serial Delaunay insertion (no thread sub-boxes) and the partitions run one after another, so
#               the cores idle during each triangulation and the stage is slower per snapshot than dtfe27; the
#               point evaluation itself runs on every core, the GPU deposits only the density_a grid. Old
#               output_plane.* files are moved to <data root>/_backup/output_plane_<date>/... first, as in dtfe27;
#               a snapshot whose output_plane.runlog shows the plane with velocity gradients at padding 15 and the
#               exact average, beside a non-empty output_plane.pts_den, is skipped (FORCE=1 reruns it), so a stage
#               cut short continues where it stopped (a killed partitioned run resumes from its checkpoint)
#   caustics    TNG100-3-Dark snapshot 099 with the caustic classes (PS_CAUSTICS=1), density + T-web only, under
#               its OWN prefix ps_caustic (the production ps_output.* grids stay untouched), budget 44 GB, the
#               mass-weighted moments of the original caustics run (PS_VOLUME_WEIGHTED=0: the driver's default is 1;
#               density and T-web do not depend on it), then the caustic skeleton and the web-streams
#               figures. ps_caustic.causticClass and ps_caustic.runlog there: the grids are skipped (FORCE=1)
#   voids       every void figure of each box (TNG50-3-Dark, TNG100-3-Dark, TNG300-3-Dark; SIMS= limits them) from
#               the grids on disk, with the thesis's void definitions (R_v = first crossing, the production
#               smoothing per box, PS-DTFE grids), script by script (analyze.py refuses its core plots ungated):
#                 1 per box: population twice (DTFE_VOID_GATE=ellipsoid for the shapes e, p; =none for the sizes and
#                   the abundance, files tagged gatenone), the size function at fixed R (10 / 5 / 3 cells; TNG300
#                   at 2.1625 Mpc and 3 / 2 cells; TNG100 and TNG300 also at 2.1625 Mpc alone), profiles at z=0,
#                   profile evolution and the environment (velocity void-in-cloud criterion; reads stage 4's
#                   tracking set when a rerun finds it), all ungated; the caustic skeleton where the box has
#                   ps_caustic grids (TNG100 z=0 so far)
#                 2 once: the cross-box comparison of the boxes (two or more) at sigma = 2.1625 Mpc, gated and ungated
#                 3 per box: the convergence check, ungated (its R_v rows; row (b) reads stage 2's ungated per-box
#                   files)
#                 4 per box, the expensive part last: the fate census --exclusive (ungated; every snapshot's
#                   particles read once: gated TNG100 16 min / 12.5 GB, TNG50 26 min / 15 GB, TNG300 ~45 min /
#                   20-25 GB; the ungated sample is larger, not yet measured), then tracking --link particles
#                   --exclusive under the same gate (it reuses the census's link caches only for the same voids),
#                   then the environment again, with the tracking panel those files now give it
#               Each figure is timed; a failed one is reported at the end and the stage goes on (a few hours)
#   all         ps27 dtfe27 dtfe27planes caustics caustics2 voids (each skips what is made already; more than a day
#               from nothing)
# The historical runs (HISTORICAL=1 only; the driver defaults they predate are pinned back to what they ran with:
# the caustic classes off):
#   production  the rerun ps27 replaced: TNG100-3-Dark, TNG50-3-Dark, TNG300-3-Dark, every snapshot recomputed
#               (FORCE=1) by run_ps_pipeline.sh with the T-web and V-web, a 52 GB memory budget, the full-grid
#               accumulators in SCRATCH_DIR, 512^3 grids and image planes of NU = 4096 / 2048 / 4096 pixels
#               (TNG50 at 2048: its z=0 streams need ~92 GB at 4096), then each simulation's point-evaluated
#               figures (plot_pointeval.py)
#   tng50z0     the redo of TNG50-3-Dark snapshot 099 (the production run was killed for memory at
#               a 52 GB budget): the same settings at a 40 GB budget, then its z=0 figures (fields, void
#               profiles at the grid and at 2.1625 Mpc, the population)
# The overnight runs ('overnight' runs the four in this order):
#   dtfe        standard DTFE (run_dtfe.sh) on TNG100-3-Dark z = 0, 1, 5 (snapshots 099 050 017): dtfe27's settings
#               and rules (exact cell average, padding 15, 3x3x3 ten at a time, the old files to _backup first)
#               for those three alone. Row (c) of the convergence figure and its DTFE-vs-PS-DTFE void table need
#               both estimators' grids of one snapshot. A snapshot dtfe27 made is skipped
#   caustics2   TNG100-3-Dark z = 1 and 2 (snapshots 050 033) with the caustic classes under the ps_caustic prefix,
#               the settings of the z = 0 caustics run pinned (a later default must not split the
#               series) except the Lagrangian padding, the binary's default (the padding changes the
#               grids only at rounding); a snapshot with ps_caustic.causticClass AND ps_caustic.runlog is skipped
#               (FORCE=1 reruns)
#   refresh     the figure repairs on the T7 and in the thesis Figures/ tree: the TNG100 z=0 web-streams table of
#               the default grids (the ps_caustic run had overwritten it), the TNG100 void catalogue export (CSV +
#               HDF5, analyze.py export), the TNG100 convergence figure at z = 0, 1, 5 (ungated; row (c) from the
#               dtfe stage), the caustic skeleton of every ps_caustic snapshot (--all-caustic), then the untagged
#               caustic_skeleton_z0.00.* files (made with --prefix ps_caustic by a build whose names did not
#               carry the prefix) MOVED -- never deleted -- into an 'old_untagged' folder beside them, once the
#               tagged caustic_skeleton_z0.00_ps_caustic.png exists there (THESIS_FIGURES: the thesis tree)
#   prune       python/analyze.py cache --prune --yes --keep-cells 3 2 (the listing first, then the delete; 3 and 2
#               cells are the fixed-R size function's smoothings), last
#   overnight   dtfe caustics2 refresh prune
# Expected (64 GB M1 Max): ps27 8.5 h for the 27 (TNG100 5-17 min, TNG300 16-19 min, TNG50 12-40 min per snapshot
# and 2.5 h at z = 0; 34.5-44.9 GB resident); dtfe27 7.6 h (TNG50 15-29 min, TNG300 18-21 min, TNG100 9-12 min per
# snapshot; the exact average costs 1.14-1.65x the sampled wall time at 512^3); dtfe27planes ~9 h, 7-11 h (an
# ESTIMATE from the code paths, not measured on TNG: under --sample-points each partition is ONE serial
# Delaunay insertion, ~0.45 M points/s -- twice the per-thread rate of dtfe27's 9-10 concurrent sub-boxes, 0.19-0.27
# M/s in its run logs, as the same 2.1M-particle synthetic set gave single against 10 threads -- over the 165-176 M
# (TNG100), 465-477 M (TNG300) and 287-370 M (TNG50) padded partition points of a snapshot, plus the density_a GPU
# deposit, the point walk and the I/O: TNG100 ~10-14, TNG300 ~23-34, TNG50 ~16-26 min per snapshot; one triangulation
# in memory at a time, ~15-25 GB); dtfe ~10 min per snapshot;
# caustics2 ~10 min, ~41 GB per snapshot (TNG100 z = 1 with every field, a 4096^2 plane and the caustics, the
# default Lagrangian padding: 8 min 57 s, 41.0 GB resident; the Eulerian padding takes 24 min 39 s, 42.1 GB at
# z = 0); refresh 20-40 min of python (convergence at three z, the skeleton with its void null, the export of 13
# snapshots from cached catalogues), ~15 GB; prune seconds. 'overnight': about 1-1.5 h (dtfe skips the grids dtfe27
# made).
# Run ONE stage at a time and nothing else heavy beside it: each run peaks at 40-50 GB, and two together get
# both SIGKILLed. The drivers' <prefix>.runlog.lock refuses a second run of the same snapshot;
# this recipe holds SCRATCH_DIR/thesis_runs.lock while it runs (a second copy is refused, exit 2), and the ps27,
# dtfe27, dtfe27planes, dtfe and caustics2 stages wait up to HEAVY_WAIT_SEC (default 30 min) for another DTFE / PS-DTFE
# run (not an Explore server) to end, then refuse to start (a listed failure).
# A full 'all' from nothing takes more than a day on the 64 GB / 10-core M-series machine.
# A stage killed part-way leaves that one snapshot's old set in _backup (said when it was moved) and its new set
# unfinished; the next run of the stage finds the backup there (not moved again), and run_ps_dtfe.sh resumes a
# partitioned PS run from its checkpoint (run_dtfe.sh a partitioned standard run too: the dtfe27
# stage pins its split and threads, so the rerun continues the killed run's partitions; a dtfe27planes run too,
# the binary checkpointing a standard '--sample-points' run's point records with its grids).
#
# Knobs (environment): SCRATCH_DIR (default ~/DTFE-scratch; LOCAL, not iCloud), PY (the python for the
# figures), DTFE_DATA_ROOT (config.sh), SIMS (the boxes of the voids, ps27, dtfe27 and dtfe27planes stages,
# space-separated; DTFE_FIGURES_ROOT and DTFE_NO_MIRROR reach the figure scripts as usual), FORCE=1 (ps27 / dtfe27 /
# dtfe27planes / dtfe / caustics / caustics2: rerun finished snapshots), HISTORICAL=1 (allow the historical production
# / tng50z0 stages), THESIS_FIGURES (the thesis Figures/ tree of the refresh stage's move; default python/config.py's
# rule), HEAVY_WAIT_SEC (how long ps27 / dtfe27 / dtfe27planes / dtfe / caustics2 wait for another run to end; default
# 1800).
# Everything else is the recipe and fixed here on purpose.
# The memory budget: DTFE_MEM_BUDGET_GB is an upper cap; the auto-tuner also caps the budget by the memory free
# when a run starts, so the partition plan (and the wall time) depends on what else is open on the machine.
# Each run log records the plan it used ('memory budget: ... GB' and the AUTO-TUNE line).
# Exit status: 0 when every stage's runs succeeded, 1 otherwise (the stages after a failure still run), 2 for a
# usage error.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//'; }
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage; echo "Error: name the stage(s) to run (ps27, dtfe27, dtfe27planes, caustics, voids, all, dtfe, caustics2, refresh, prune, overnight; HISTORICAL=1: production, tng50z0)." >&2; exit 2 ;;
esac
HISTORICAL="${HISTORICAL:-0}"       # 1 = allow the historical stages (production, tng50z0)
stages=()
for s in "$@"; do
    case "${s}" in
        production|tng50z0)
            # these made the grids that ps27 replaced (no caustic classes, TNG50's plane at 2048^2); a
            # run of either would put the superseded set back over the production one
            if [ "${HISTORICAL}" != "1" ]; then
                echo "Error: '${s}' is a historical stage (the 2026-10-03 / 10-07 PS-DTFE grids, replaced on 2026-10-12): the" >&2
                echo "       production grids come from 'ps27' (PS-DTFE) and 'dtfe27' (standard DTFE). HISTORICAL=1 runs it anyway." >&2
                exit 2
            fi
            stages+=("${s}") ;;
        ps27|dtfe27|dtfe27planes|caustics|voids|dtfe|caustics2|refresh|prune) stages+=("${s}") ;;
        all) stages+=(ps27 dtfe27 dtfe27planes caustics caustics2 voids) ;;
        overnight) stages+=(dtfe caustics2 refresh prune) ;;
        *) echo "Error: unknown stage '${s}' (ps27, dtfe27, dtfe27planes, caustics, voids, all, dtfe, caustics2, refresh, prune, overnight; HISTORICAL=1: production, tng50z0)." >&2; exit 2 ;;
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
# The cross-box smoothing in Mpc of the void figures: python/config.py's CROSSBOX_SMOOTHING_MPC. A
# literal on purpose: this is the recipe of the runs that were made, and reading config would import it (DTFE_SIM,
# the box lookup) just to print a number; keep the two equal (tests/py_void_core_test.py checks it).
CROSSBOX_SMOOTH_MPC=2.1625

data_root_problem "${DATA_ROOT}" || exit 2
if [ "${DRY_RUN}" != "1" ]; then
    mkdir -p "${SCRATCH_DIR}" || { echo "Error: cannot create SCRATCH_DIR '${SCRATCH_DIR}'" >&2; exit 2; }
fi
cd "${REPO_ROOT}" || exit 2

# One recipe at a time: two heavy runs together get both SIGKILLed, and an overnight
# chain started twice (a second terminal, a re-run of the command) would do exactly that. A folder holding this
# script's pid; one whose pid is gone, or is no longer a thesis_runs.sh, is stale and removed with a note. Not
# under DRY_RUN: a dry run creates nothing.
RECIPE_LOCK=""
recipe_lock_release() {
    [ -n "${RECIPE_LOCK}" ] && [ "$(cat "${RECIPE_LOCK}/pid" 2>/dev/null)" = "$$" ] && rm -rf "${RECIPE_LOCK}"
    return 0
}
if [ "${DRY_RUN}" != "1" ]; then
    lock="${SCRATCH_DIR}/thesis_runs.lock"
    for _try in 1 2; do
        if mkdir "${lock}" 2>/dev/null; then
            echo "$$" > "${lock}/pid"; RECIPE_LOCK="${lock}"; break
        fi
        pid="$(cat "${lock}/pid" 2>/dev/null || true)"
        [ -n "${pid}" ] || { sleep 1; pid="$(cat "${lock}/pid" 2>/dev/null || true)"; }
        if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null \
            && ps -o args= -p "${pid}" 2>/dev/null | grep -q 'thesis_runs\.sh'; then
            echo "Error: another thesis_runs.sh (pid ${pid}) is running: ${lock}" >&2
            echo "       Two runs of the recipe do not fit in memory together; wait for it, or stop it." >&2
            exit 2
        fi
        echo "Note: removing a stale recipe lock (pid ${pid:-unknown} is no longer running): ${lock}"
        rm -rf "${lock}"
    done
    [ -n "${RECIPE_LOCK}" ] || { echo "Error: could not take the recipe lock '${lock}'" >&2; exit 2; }
    trap 'recipe_lock_release' EXIT
    trap 'recipe_lock_release; trap - INT; kill -s INT $$' INT
    trap 'recipe_lock_release; trap - TERM; kill -s TERM $$' TERM
fi

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
        # PS_CAUSTICS=0: run_ps_dtfe.sh's default is 1; the historical run had no caustic classes
        DTFE_MEM_BUDGET_GB=52 FORCE=1 FIELDS="${THESIS_FIELDS}" SIMS="${sim}" NU="${nu}" GRID_SIZE=512 PS_CAUSTICS=0 \
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
        SAMPLE_POINTS="$(sim_dir "${sim}")/hires_plane_z2048.bin" PS_VOLUME_WEIGHTED=1 PTS_VEL_GRAD=1 PS_CAUSTICS=0 \
        FIELDS="${THESIS_FIELDS}" \
        "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g 512 -m 99 || rc=$?
    stamp "${sim} 099: exit ${rc}"
    if [ "${rc}" != "0" ]; then overall=1; return; fi
    stamp "${sim} figures (z=0)"
    export DTFE_SIM="${sim}"
    fig "${PY}" python/plot/plot_pointeval.py --sims "${sim}" --snaps 99 \
        --plane "$(sim_dir "${sim}")/hires_plane_z2048.json" || overall=1
    fig "${PY}" python/plot/plot_void_profiles.py --sim "${sim}" --method ps --snaps 99 || overall=1
    fig "${PY}" python/plot/plot_void_profiles.py --sim "${sim}" --method ps --snaps 99 --smooth-mpc "${CROSSBOX_SMOOTH_MPC}" || overall=1
    fig "${PY}" python/plot/plot_void_population.py --sim "${sim}" --method ps --smooth-mpc "${CROSSBOX_SMOOTH_MPC}" || overall=1
    unset DTFE_SIM
}

stage_caustics() {
    local sim=TNG100-3-Dark rc=0 d
    d="$(sim_dir "${sim}")/snapdir_099"
    # made already -> skipped (FORCE=1 reruns), as caustics2 does; 'all' runs this stage again
    if [ "${FORCE:-0}" != "1" ] && [ -s "${d}/ps_caustic.causticClass" ] && [ -f "${d}/ps_caustic.runlog" ]; then
        stamp "caustics: ${sim} 099: ps_caustic.causticClass and ps_caustic.runlog are there: grids skipped (FORCE=1 reruns them)"
    else
        stamp "${sim} 099: caustics (prefix ps_caustic, budget 44 GB)"
        # PS_VOLUME_WEIGHTED=0: the original caustics run's mass-weighted moments (run_ps_dtfe.sh's default is 1;
        # the density and T-web grids do not depend on it, its RUNNING line does)
        DTFE_MEM_BUDGET_GB=44 PS_CAUSTICS=1 PS_VOLUME_WEIGHTED=0 OUTPUT_PREFIX=ps_caustic FIELDS="density_a tweb_a" \
            "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g 512 -m 99 || rc=$?
        stamp "${sim} 099 caustics: exit ${rc}"
        if [ "${rc}" != "0" ]; then overall=1; return; fi
    fi
    stamp "${sim} caustic skeleton (z=0)"
    export DTFE_SIM="${sim}"
    fig "${PY}" python/plot/plot_caustic_skeleton.py --sim "${sim}" --prefix ps_caustic --snaps 99 --voids || overall=1
    fig "${PY}" python/plot/plot_web_streams.py --sim "${sim}" --prefix ps_caustic --snaps 99 --web tweb || overall=1
    unset DTFE_SIM
}

# The voids stage. Every figure goes through vfig: printed with its environment (env VAR=... so a
# DRY_RUN line is the whole command), timed, and a failure is collected in void_failed while the stage goes on --
# one figure's bug must not cost the hours of the ones after it.
VOID_SIMS_DEFAULT="TNG50-3-Dark TNG100-3-Dark TNG300-3-Dark"
# The boxes whose size function at fixed R is also drawn at the cross-box sigma alone: the two the
# comparison at one scale in Mpc was measured in (TNG100 0.93 +- 0.15, TNG300 0.84 +- 0.04 of Vdn at z=0)
FIXED_R_XBOX_SIMS="TNG100-3-Dark TNG300-3-Dark"
void_failed=()
# vfig LABEL COMMAND...
vfig() {
    local label="$1" t0=${SECONDS} rc=0
    shift
    stamp "voids: ${label}"
    fig "$@" || rc=$?
    [ "${DRY_RUN}" = "1" ] || stamp "voids: ${label}: exit ${rc} ($((SECONDS - t0)) s)"
    if [ "${rc}" != "0" ]; then void_failed+=("${label} (exit ${rc})"); overall=1; fi
}

stage_voids() {
    local sims=() sim snaps d fixed_r
    read -r -a sims <<< "${SIMS:-${VOID_SIMS_DEFAULT}}"
    # A caller's DTFE_SIGMA_MPC would move every per-box figure off the production smoothing (they tag it, but the
    # recipe is the production one); the gate is set per figure below, so a leaked DTFE_VOID_GATE changes nothing.
    unset DTFE_SIGMA_MPC
    local ok=()
    for sim in "${sims[@]}"; do
        if [ -d "$(sim_dir "${sim}")" ]; then ok+=("${sim}")
        else stamp "voids: ${sim}: no folder $(sim_dir "${sim}"): skipped"; void_failed+=("${sim}: no data"); overall=1; fi
    done
    [ "${#ok[@]}" -gt 0 ] || return

    # 1. the cheap per-box figures. The shapes keep the ellipsoid gate (the e, p of fitted ellipsoids); the R_v
    #    statistics drop it (DTFE_VOID_GATE=none: it removed ~78% of the z=0 minima for a large curvature length,
    #    not an unresolved R_v), and the pipeline tags those files and folders gatenone.
    for sim in "${ok[@]}"; do
        vfig "${sim} population (shapes, gated)" env DTFE_SIM="${sim}" DTFE_VOID_GATE=ellipsoid \
            "${PY}" python/plot/plot_void_population.py --sim "${sim}" --method ps
        vfig "${sim} population (sizes, ungated)" env DTFE_SIM="${sim}" DTFE_VOID_GATE=none \
            "${PY}" python/plot/plot_void_population.py --sim "${sim}" --method ps
        # the size function at FIXED radius: the raw -0.8 radius of the distinct voids from catalogues
        # at several smoothings, against Vdn only over the radii where every smoothing is complete. Ungated by
        # definition (the script refuses gated: the ellipsoid gate cuts the large voids differently at each
        # smoothing). The default 10 / 5 / 3 cells leave TNG300 no common range (3 sigma of its 5.91 Mpc = 17.7 Mpc,
        # 10 voids above it at z=0), so TNG300 uses 2.1625 Mpc and 3 and 2 cells (6.5-15.7 Mpc at z=0). Seconds
        # from the cached catalogues, minutes the first time: cheap, so before the fate census.
        case "${sim}" in
            TNG300-3-Dark) fixed_r="--fixed-r-smooth 3 2 --fixed-r-smooth-mpc ${CROSSBOX_SMOOTH_MPC}" ;;
            *) fixed_r="" ;;
        esac
        # shellcheck disable=SC2086  # the smoothing options split on purpose (none for the default smoothings)
        vfig "${sim} size function at fixed R" env DTFE_SIM="${sim}" DTFE_VOID_GATE=none \
            "${PY}" python/plot/plot_void_population.py --sim "${sim}" --method ps --size-fixed-r ${fixed_r}
        # and at the cross-box sigma alone, the same scale in Mpc in each box (its own png tag _m2.1625)
        case " ${FIXED_R_XBOX_SIMS} " in
            *" ${sim} "*)
                vfig "${sim} size function at fixed R (cross-box sigma)" env DTFE_SIM="${sim}" DTFE_VOID_GATE=none \
                    "${PY}" python/plot/plot_void_population.py --sim "${sim}" --method ps --size-fixed-r \
                    --fixed-r-smooth-mpc "${CROSSBOX_SMOOTH_MPC}" ;;
        esac
        vfig "${sim} profiles z=0" env DTFE_SIM="${sim}" DTFE_VOID_GATE=none \
            "${PY}" python/plot/plot_void_profiles.py --sim "${sim}" --method ps --snaps 99
        vfig "${sim} profile evolution" env DTFE_SIM="${sim}" DTFE_VOID_GATE=none \
            "${PY}" python/plot/plot_void_profile_evolution.py --sim "${sim}" --method ps --rv-definition crossing
        # the tracking set of stage 4 (its files' tag): drawn when a rerun finds it, else skipped with a note; drawn
        # again after the tracking below, so a first run's figure has its tracking panel too
        vfig "${sim} environment" env DTFE_SIM="${sim}" DTFE_VOID_GATE=none \
            "${PY}" python/plot/plot_void_environment.py --sim "${sim}" --method ps --criterion velocity \
            --r-v-definition crossing --tracking-tag _particles_exclusive
        # the caustic skeleton only where a caustics run left its class grid; the
        # shapes' gate, as the caustics stage draws it
        snaps=""
        for d in "$(sim_dir "${sim}")"/snapdir_*; do
            [ -e "${d}/ps_caustic.causticClass" ] && snaps="${snaps} $((10#${d##*_}))"
        done
        if [ -n "${snaps}" ]; then
            # shellcheck disable=SC2086  # the snapshot numbers split on purpose
            vfig "${sim} caustic skeleton" env DTFE_SIM="${sim}" DTFE_VOID_GATE=ellipsoid \
                "${PY}" python/plot/plot_caustic_skeleton.py --sim "${sim}" --prefix ps_caustic --snaps ${snaps} --voids
        else
            stamp "voids: ${sim} caustic skeleton: no ps_caustic grids, skipped"
        fi
    done

    # 2. the boxes side by side at one sigma in Mpc, both gates (each per-box worker inherits the gate and tags
    #    its .npz); no DTFE_SIM: the script gives each worker its box's. One box compares nothing: skipped (a
    #    SIMS= run of one box keeps the cross-box files of the full set)
    if [ "${#ok[@]}" -ge 2 ]; then
        vfig "cross-box (gated)" env -u DTFE_SIM DTFE_VOID_GATE=ellipsoid "${PY}" python/plot/plot_void_crossbox.py \
            --sims "${ok[@]}" --method ps --smooth-mpc "${CROSSBOX_SMOOTH_MPC}"
        vfig "cross-box (ungated)" env -u DTFE_SIM DTFE_VOID_GATE=none "${PY}" python/plot/plot_void_crossbox.py \
            --sims "${ok[@]}" --method ps --smooth-mpc "${CROSSBOX_SMOOTH_MPC}"
    else
        stamp "voids: cross-box: one box (${ok[*]}), nothing to compare: skipped"
    fi

    # 3. convergence: a test of the method under the thesis's own definitions. Its rows are R_v statistics (matched
    #    R_v ratios, dn/dlnR_v, median R_v, stacks in r/R_v), so ungated like the others; its mass row reads the
    #    ungated cross-box files of step 2 (the '_gatenone' ones: plot_void_convergence globs the gate's own tag).
    for sim in "${ok[@]}"; do
        vfig "${sim} convergence" env DTFE_SIM="${sim}" DTFE_VOID_GATE=none \
            "${PY}" python/plot/plot_void_convergence.py --sim "${sim}" --method ps \
            --crossbox-smooth-mpc "${CROSSBOX_SMOOTH_MPC}"
    done

    # 4. the expensive part last: the fate census reads every snapshot's particles. Tracking follows the
    #    census's exclusive links, cached per followed-void sample: the same gate, or it reads the particles again.
    for sim in "${ok[@]}"; do
        vfig "${sim} fate census" env DTFE_SIM="${sim}" DTFE_VOID_GATE=none \
            "${PY}" python/plot/plot_void_fate.py --sim "${sim}" --method ps --exclusive
        vfig "${sim} tracking" env DTFE_SIM="${sim}" DTFE_VOID_GATE=none \
            "${PY}" python/plot/plot_void_tracking.py --sim "${sim}" --method ps --link particles --exclusive
        # the environment once more, now that its tracking panel's files exist (step 1 draws it before the
        # tracking has run, so a first run's figure would lack that panel)
        vfig "${sim} environment (with the tracking)" env DTFE_SIM="${sim}" DTFE_VOID_GATE=none \
            "${PY}" python/plot/plot_void_environment.py --sim "${sim}" --method ps --criterion velocity \
            --r-v-definition crossing --tracking-tag _particles_exclusive
    done
    if [ "${#void_failed[@]}" -gt 0 ]; then
        stamp "voids: ${#void_failed[@]} FAILED: $(printf '%s; ' "${void_failed[@]}")"
    else
        stamp "voids: $([ "${DRY_RUN}" = "1" ] && echo "DRY_RUN: every command printed, none run" || echo "every figure made")"
    fi
}

# ---- the overnight stages -----------------------------------------------------------------------------------
# Each step goes through one of two runners, printed with its environment (env VAR=..., so a log line is the whole
# command), timed, a failure collected in step_failed while the stage goes on:
#   drive STAGE LABEL CMD...  a run driver (run_dtfe.sh / run_ps_dtfe.sh): ALWAYS started, as it honours DRY_RUN
#                             itself (prints its expanded binary command and runs nothing)
#   step  STAGE LABEL CMD...  a figure / analysis command: printed only under DRY_RUN (fig)
step_failed=()
OVERNIGHT_SIM=TNG100-3-Dark
drive_rc=0                          # the last drive's exit status (ps27 / dtfe27 move the old files back on a failure)
drive() {
    local stg="$1" label="$2" t0=${SECONDS} rc=0
    shift 2
    stamp "${stg}: ${label}"
    echo "+ $*"
    "$@" || rc=$?
    drive_rc="${rc}"
    [ "${DRY_RUN}" = "1" ] || stamp "${stg}: ${label}: exit ${rc} ($((SECONDS - t0)) s)"
    if [ "${rc}" != "0" ]; then step_failed+=("${stg}: ${label} (exit ${rc})"); overall=1; fi
}
step() {
    local stg="$1" label="$2" t0=${SECONDS} rc=0
    shift 2
    stamp "${stg}: ${label}"
    fig "$@" || rc=$?
    [ "${DRY_RUN}" = "1" ] || stamp "${stg}: ${label}: exit ${rc} ($((SECONDS - t0)) s)"
    if [ "${rc}" != "0" ]; then step_failed+=("${stg}: ${label} (exit ${rc})"); overall=1; fi
}
# heavy_running: the DTFE / PS-DTFE runs going now (pid and command), one per line; an Explore server (--serve)
# holds memory too but does not grow, so it is named, not refused. Matched on the EXECUTABLE (ps comm, the path
# as started), not the first word of the command line: '/Applications/DTFE Launcher.app/...' split at its space
# would read as '/Applications/DTFE', i.e. a DTFE binary, and hold a stage for 30 min while the launcher is open.
heavy_running() {
    local pid
    ps -A -o pid=,comm= 2>/dev/null | awk '{ p = $0; sub(/^ *[0-9]+ /, "", p); sub(/.*\//, "", p)
        if (p ~ /^(PS-)?DTFE(-double)?(-2d)?$/) print $1 }' | while read -r pid; do
        ps -ww -o pid=,args= -p "${pid}" 2>/dev/null | grep -v -e '--serve'
    done
}
# heavy_guard STAGE: 0 when nothing heavy runs; else waits for it, up to HEAVY_WAIT_SEC (default 1800 s, polled
# each minute: a test or a short run still finishing at bedtime must not cost the night), then says what runs and
# records the stage as failed (not started). Not under DRY_RUN, which starts nothing.
heavy_guard() {
    local busy line waited=0 limit="${HEAVY_WAIT_SEC:-1800}"
    [ "${DRY_RUN}" = "1" ] && return 0
    busy="$(heavy_running)"
    while [ -n "${busy}" ] && [ "${waited}" -lt "${limit}" ]; do
        [ "${waited}" = "0" ] && stamp "$1: another DTFE / PS-DTFE run is going: waiting for it (up to ${limit} s)"
        sleep 60; waited=$((waited + 60))
        busy="$(heavy_running)"
    done
    [ -z "${busy}" ] && return 0
    stamp "$1: REFUSED, another run is going (two at once got both SIGKILLed, 2026-10-03):"
    while IFS= read -r line; do echo "    ${line}"; done <<< "${busy}"
    step_failed+=("$1: not started, another DTFE / PS-DTFE run was going"); overall=1
    return 1
}
expect() { stamp "$1: expected $2"; }

# ---- the production grids: ps27, dtfe27, dtfe27planes ----------------------------------------------------------------
# The two drivers that made the grids on the data disk, as stages (under ~/DTFE-rerun/, with the paths hard-coded):
# run_psrerun.sh (list_all27.txt) and, for standard DTFE, run_dtfe512.sh, here with the exact cell average and
# padding 15. Every knob the drivers read is pinned per run, so a caller's shell (or a later driver default) cannot
# change the recipe. Not carried over: run_dtfe512.sh's one-off copy of TNG300 099's stale output.runlog
# (the backup move below keeps every old file), and run_psrerun.sh's stop-on-fail switch (a failed snapshot is
# listed and the stage goes on, as every stage here does).
THESIS_SNAPS="0 4 17 25 33 40 50 67 99"     # the nine snapshots of every box, z = 20 ... 0
THESIS_DTFE_FIELDS="density_a velocity_a gradient_a divergence_a shear_a vorticity_a tweb_a vweb_a"
BACKUP_DATE="$(date '+%Y-%m-%d')"           # the backup folders' date: the day the stage starts
# backup_dir PREFIX SIM NNN: where a snapshot's old <PREFIX>.* files go before a rerun (run_psrerun.sh's layout:
# <data root>/_backup/<prefix>_<date>/<family>/<sim>/snapdir_NNN; same volume, so a move is a rename)
backup_dir() { echo "${DATA_ROOT}/_backup/$1_${BACKUP_DATE}/${2%%-*}/$2/snapdir_$3"; }
# in_sims SIM: the box is among SIMS (all boxes when SIMS is unset)
in_sims() { [ -z "${SIMS:-}" ] && return 0; case " ${SIMS} " in *" $1 "*) return 0 ;; esac; return 1; }
# running_has RUNLOG TOKEN...: the log's RUNNING: line (the binary's command) carries every token (' --x ' form)
running_has() {
    local line t
    line="$(grep -a -m1 '^RUNNING:' "$1" 2>/dev/null) "
    shift
    for t in "$@"; do case "${line}" in *"${t}"*) ;; *) return 1 ;; esac; done
    return 0
}
# backup_move STAGE PREFIX SNAPDIR BACKUP: the snapshot's <PREFIX>.* files (files only: not a lock or superseded
# folder) MOVED into BACKUP with mv -n -- unless BACKUP holds a <PREFIX>.a_den already (an earlier attempt moved
# them: not moved again, nothing overwritten). Under DRY_RUN the move is printed, nothing moves. 1 = could not.
backup_move() {
    local stg="$1" pfx="$2" d="$3" b="$4" f n=0 left=0
    if [ -e "${b}/${pfx}.a_den" ]; then
        stamp "${stg}: backup already there (${b}): the old ${pfx}.* files are not moved again"
        return 0
    fi
    for f in "${d}/${pfx}".*; do [ -f "${f}" ] && n=$((n + 1)); done
    if [ "${n}" -eq 0 ]; then stamp "${stg}: no old ${pfx}.* files in ${d}: nothing to back up"; return 0; fi
    echo "+ mv -n ${d}/${pfx}.* ${b}/ (${n} files)"
    [ "${DRY_RUN}" = "1" ] && return 0
    mkdir -p "${b}" || { stamp "${stg}: cannot create the backup folder ${b}: the run is not started"; return 1; }
    for f in "${d}/${pfx}".*; do
        [ -f "${f}" ] || continue
        mv -n "${f}" "${b}/" 2>/dev/null
        [ -e "${f}" ] && left=$((left + 1))
    done
    if [ "${left}" -gt 0 ]; then
        stamp "${stg}: ${left} of ${n} old ${pfx}.* files could not be moved to ${b} (a file of that name is there?): the run is not started"
        return 1
    fi
    stamp "${stg}: moved ${n} old ${pfx}.* files to ${b}"
}
# after_run STAGE PREFIX SNAPDIR BACKUP: a failed run that left no new <PREFIX>.a_den gets the old files moved back
# (those whose name is free); a good one is summarised as run_psrerun.sh did (box mean of the density - 1, NaN cells,
# wall time and peak memory from the run log; the mean only when PY has numpy)
after_run() {
    local stg="$1" pfx="$2" d="$3" b="$4" f n=0 m
    [ "${DRY_RUN}" = "1" ] && return 0
    if [ "${drive_rc}" != "0" ]; then
        if [ ! -s "${d}/${pfx}.a_den" ] && [ -d "${b}" ]; then
            for f in "${b}/${pfx}".*; do
                [ -f "${f}" ] || continue
                [ -e "${d}/${f##*/}" ] || { mv -n "${f}" "${d}/" && n=$((n + 1)); }
            done
            stamp "${stg}: FAILED (exit ${drive_rc}): ${n} old ${pfx}.* files moved back from ${b}"
        fi
        return 0
    fi
    m="$("${PY}" -c 'import sys
import numpy as np
a = np.memmap(sys.argv[1], dtype=np.float32, mode="r"); s = 0.0; bad = 0
for i in range(0, a.size, 1 << 26):
    c = np.asarray(a[i:i + (1 << 26)], dtype=np.float64); ok = np.isfinite(c); bad += int((~ok).sum()); s += c[ok].sum()
print("mean-1 %+.2e nan %d" % (s / a.size - 1, bad))' "${d}/${pfx}.a_den" 2>/dev/null | tail -1)" || m=""
    stamp "${stg}: OK ${m:-(no density check: no numpy)}  $(grep -a 'total wall time' "${d}/${pfx}.runlog" 2>/dev/null | head -1 | sed 's/^ *//')  $(grep -a 'peak memory (RSS)' "${d}/${pfx}.runlog" 2>/dev/null | head -1 | sed 's/^ *//')"
}

# dtfe_exact_snapshot STAGE SIM SNAP PARTITION MAX_CONCURRENT: one standard-DTFE snapshot with the production
# settings (dtfe27; the dtfe stage runs three of them)
dtfe_exact_snapshot() {
    local stg="$1" sim="$2" snap="$3" part="$4" mc="$5" n d b
    n="$(printf '%03d' "${snap}")"; d="$(sim_dir "${sim}")/snapdir_${n}"; b="$(backup_dir output "${sim}" "${n}")"
    if [ ! -f "${d}/combined_${n}.hdf5" ]; then
        stamp "${stg}: ${sim} ${n}: no ${d}/combined_${n}.hdf5: skipped"
        step_failed+=("${stg}: ${sim} ${n}: input missing"); overall=1; return
    fi
    # finished = made with THESE settings: sampled grids, or those at padding 20, are not
    if [ "${FORCE:-0}" != "1" ] && [ -s "${d}/output.a_den" ] && [ -f "${d}/output.runlog" ] \
        && running_has "${d}/output.runlog" " --padding 15 " " --exact-average "; then
        stamp "${stg}: ${sim} ${n}: output.* made with the exact average at padding 15 already: skipped (FORCE=1 reruns it)"
        return
    fi
    backup_move "${stg}" output "${d}" "${b}" || { step_failed+=("${stg}: ${sim} ${n}: the old files could not be backed up"); overall=1; return; }
    # pinned knob by knob (run_dtfe.sh reads them all from the environment): no scratch dir (run_dtfe512.sh ran
    # without; the recipe exports SCRATCH_DIR for the PS stages), no memory budget (the split is explicit), the
    # default prefix, and no auto-tune report (run_dtfe.sh reads AUTO_TUNE_REPORT: a leaked
    # AUTO_TUNE_REPORT=1 would only report)
    drive "${stg}" "${sim} ${n} standard DTFE 512^3, exact cell averages, ${part// /x} x ${mc}" env DTFE_MEM_BUDGET_GB= \
        SCRATCH_DIR= PADDING=15 FIELDS="${THESIS_DTFE_FIELDS}" LAMBDA_TH=0.3 SAMPLE_POINTS= PTS_VEL_GRAD=0 \
        PARTITION="${part}" MAX_CONCURRENT="${mc}" TESS_CACHE= DTFE_EXACT_AVERAGE=1 DTFE_METAL=1 OUTPUT_PREFIX=output \
        DTFE_PRECISION=single AUTO_TUNE_REPORT=0 "${SCRIPT_DIR}/run_dtfe.sh" -s "${sim}" -g 512 -m "${snap}"
    after_run "${stg}" output "${d}" "${b}"
}

# dtfe27's boxes in their order, each with its split and threads per partition ('box|split|threads'); dtfe27planes
# runs the same splits, so its partition boxes are dtfe27's
DTFE27_JOBS=("TNG50-3-Dark|4 4 4|4" "TNG300-3-Dark|5 5 5|10" "TNG100-3-Dark|3 3 3|10")

stage_dtfe27() {
    local job sim part mc snap
    expect dtfe27 "~7.6 h for the 27 (TNG50 15-29 min, TNG300 18-21 min, TNG100 9-12 min per snapshot; footprint <= 38 GB)"
    heavy_guard dtfe27 || return
    for job in "${DTFE27_JOBS[@]}"; do
        IFS='|' read -r sim part mc <<< "${job}"
        in_sims "${sim}" || continue
        for snap in ${THESIS_SNAPS}; do
            dtfe_exact_snapshot dtfe27 "${sim}" "${snap}" "${part}" "${mc}"
        done
    done
}

# dtfe_plane_snapshot STAGE SIM SNAP PARTITION MAX_CONCURRENT: one snapshot's standard-DTFE 4096^2 image
# plane, the point-evaluation twin of ps27's (the same plane file, --pts-vel-grad), at dtfe27's production settings,
# under its OWN prefix output_plane. What differs from dtfe_exact_snapshot, and why:
#   - OUTPUT_PREFIX=output_plane: dtfe27's output.* grids are never touched. The backup move, the move back after a
#     failure and the supersede move of run_dtfe.sh all go by '<prefix>.*', and 'output.*' does not match
#     'output_plane.*' (nor the other way round), so neither set can move the other's files.
#   - FIELDS=density_a: the binary refuses a run without a grid field (user_options.cc: "No valid field interpolation
#     quantities were selected"), so the smallest one is asked for -- one float per cell, deposited on the GPU at the
#     exact cell average, i.e. dtfe27's output.a_den to rounding (a free cross-check of the two runs). The point
#     evaluation does not need it: its densities and velocities come from the tessellation's vertices.
#   - SAMPLE_POINTS + PTS_VEL_GRAD=1: the box's plane (ensure_plane, shared with ps27) and the velocity gradient.
#   - MAX_CONCURRENT is passed as dtfe27's for the record but changes nothing here: under --sample-points each
#     partition is ONE triangulation (DTFE_parallel's single-triangulation branch: thread sub-boxes would answer a
#     point twice), inserted serially; the point walk uses every core (pointEvalThreads), the GPU only the grid.
#   - Skipped when output_plane.pts_den is there and output_plane.runlog shows this recipe (the plane, the gradients,
#     padding 15, the exact average). A run killed part-way resumes from its checkpoint through run_dtfe.sh's
#     DTFE_RESUME (the binary checkpoints a standard '--sample-points' run).
dtfe_plane_snapshot() {
    local stg="$1" sim="$2" snap="$3" part="$4" mc="$5" n d b plane
    n="$(printf '%03d' "${snap}")"; d="$(sim_dir "${sim}")/snapdir_${n}"; b="$(backup_dir output_plane "${sim}" "${n}")"
    plane="$(sim_dir "${sim}")/hires_plane_z4096.bin"
    if [ ! -f "${d}/combined_${n}.hdf5" ]; then
        stamp "${stg}: ${sim} ${n}: no ${d}/combined_${n}.hdf5: skipped"
        step_failed+=("${stg}: ${sim} ${n}: input missing"); overall=1; return
    fi
    if [ "${FORCE:-0}" != "1" ] && [ -s "${d}/output_plane.pts_den" ] && [ -f "${d}/output_plane.runlog" ] \
        && running_has "${d}/output_plane.runlog" " --padding 15 " "/hires_plane_z4096.bin " " --pts-vel-grad " " --exact-average "; then
        stamp "${stg}: ${sim} ${n}: output_plane.* (the 4096^2 plane with velocity gradients, padding 15) made already: skipped (FORCE=1 reruns it)"
        return
    fi
    backup_move "${stg}" output_plane "${d}" "${b}" || { step_failed+=("${stg}: ${sim} ${n}: the old files could not be backed up"); overall=1; return; }
    # pinned knob by knob, as dtfe_exact_snapshot (run_dtfe.sh reads them all from the environment)
    drive "${stg}" "${sim} ${n} standard DTFE 4096^2 image plane + velocity gradients (prefix output_plane), ${part// /x}" \
        env DTFE_MEM_BUDGET_GB= SCRATCH_DIR= PADDING=15 FIELDS="density_a" LAMBDA_TH=0.3 SAMPLE_POINTS="${plane}" \
        PTS_VEL_GRAD=1 PARTITION="${part}" MAX_CONCURRENT="${mc}" TESS_CACHE= DTFE_EXACT_AVERAGE=1 DTFE_METAL=1 \
        OUTPUT_PREFIX=output_plane DTFE_PRECISION=single AUTO_TUNE_REPORT=0 "${SCRIPT_DIR}/run_dtfe.sh" -s "${sim}" -g 512 \
        -m "${snap}"
    after_run "${stg}" output_plane "${d}" "${b}"
}

stage_dtfe27planes() {
    local job sim part mc snap
    expect dtfe27planes "~9 h for the 27 (an estimate, 7-11 h: TNG100 ~10-14, TNG300 ~23-34, TNG50 ~16-26 min per snapshot; every partition one serial triangulation; ~15-25 GB); a stage cut short continues where it stopped (finished snapshots skipped, a killed run resumed)"
    heavy_guard dtfe27planes || return
    for job in "${DTFE27_JOBS[@]}"; do
        IFS='|' read -r sim part mc <<< "${job}"
        in_sims "${sim}" || continue
        ensure_plane dtfe27planes "${sim}" || continue
        for snap in ${THESIS_SNAPS}; do
            dtfe_plane_snapshot dtfe27planes "${sim}" "${snap}" "${part}" "${mc}"
        done
    done
}

# ps27_snapshot SIM SNAP: one PS-DTFE snapshot as run_psrerun.sh runs it (every line of its env)
ps27_snapshot() {
    local sim="$1" snap="$2" n d b plane
    n="$(printf '%03d' "${snap}")"; d="$(sim_dir "${sim}")/snapdir_${n}"; b="$(backup_dir ps_output "${sim}" "${n}")"
    plane="$(sim_dir "${sim}")/hires_plane_z4096.bin"
    if [ ! -f "${d}/combined_${n}.hdf5" ]; then
        stamp "ps27: ${sim} ${n}: no ${d}/combined_${n}.hdf5: skipped"
        step_failed+=("ps27: ${sim} ${n}: input missing"); overall=1; return
    fi
    # finished = made by this recipe: the caustic classes, the 4096^2 plane and the ordered merge
    # ('--ordered-merge' in the RUNNING line, as run_ps_dtfe.sh passes ORDERED_MERGE=1); a set without them (or
    # merged in completion order) is not, so the next ps27 replaces it and an interrupted one goes on
    if [ "${FORCE:-0}" != "1" ] && [ -s "${d}/ps_output.a_den" ] && [ -f "${d}/ps_output.runlog" ] \
        && running_has "${d}/ps_output.runlog" " --ps-caustics " "/hires_plane_z4096.bin " " --ordered-merge "; then
        stamp "ps27: ${sim} ${n}: ps_output.* made with the caustics, the 4096^2 plane and the ordered merge already: skipped (FORCE=1 reruns it)"
        return
    fi
    backup_move ps27 ps_output "${d}" "${b}" || { step_failed+=("ps27: ${sim} ${n}: the old files could not be backed up"); overall=1; return; }
    # the Lagrangian positions as the driver finds them (the IC file beside the snapshots): a caller's
    # LAGRANGIAN_INPUT is dropped (-u: the driver tells an empty one from an unset one)
    drive ps27 "${sim} ${n} PS-DTFE 512^3 + caustics + 4096^2 plane (budget 52 GB)" env -u LAGRANGIAN_INPUT \
        DTFE_MEM_BUDGET_GB=52 SCRATCH_DIR="${SCRATCH_DIR}" FIELDS="${THESIS_FIELDS}" PADDING=50 AVG_SUBSAMPLES=3 \
        LAMBDA_TH=0.3 PS_VERTEX_MASS=1 PS_VOLUME_WEIGHTED=1 PS_EXACT=0 PS_LINEAR_DEPOSIT=0 PS_PARALLEL_TRI=0 \
        PS_LAG_PADDING= PS_CAUSTICS=1 PS_CAUSTIC_CUSPS=0 SAMPLE_POINTS="${plane}" PTS_VEL_GRAD=1 PTS_CONVENTIONS=grid PARTITION= \
        MAX_CONCURRENT= TESS_CACHE= OUTPUT_PREFIX=ps_output ORDERED_MERGE=1 DTFE_PRECISION=single PS_METAL=1 THREADS= \
        AUTO_TUNE_REPORT=0 "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g 512 -m "${snap}"
    after_run ps27 ps_output "${d}" "${b}"
}

# ensure_plane STAGE SIM: the box's 4096^2 image plane <sim>/hires_plane_z4096.{bin,json}, made as
# run_ps_pipeline.sh makes one, from the first snapshot, when it is not there (the T7's were made
# from snapdir_000). 1 = the box has no snapshot or no plane could be made (said, and the stage's failure). Shared
# by stage_ps27 and dtfe27planes, so both evaluate the very same plane files.
ensure_plane() {
    local stg="$1" sim="$2" simdir plane first snap
    simdir="$(sim_dir "${sim}")"; plane="${simdir}/hires_plane_z4096"
    if [ ! -f "${plane}.bin" ] || [ ! -f "${plane}.json" ]; then
        first=""
        for snap in ${THESIS_SNAPS}; do
            snap="$(printf '%03d' "${snap}")"
            [ -f "${simdir}/snapdir_${snap}/combined_${snap}.hdf5" ] && { first="${simdir}/snapdir_${snap}/combined_${snap}.hdf5"; break; }
        done
        if [ -z "${first}" ]; then
            stamp "${stg}: ${sim}: no snapshot under ${simdir}: skipped"; step_failed+=("${stg}: ${sim}: no data"); overall=1
            return 1
        fi
        step "${stg}" "${sim} image plane 4096^2" "${PY}" python/tools/make_image_plane.py --combined "${first}" \
            --axis z --nu 4096 --planes 1 --thickness 2 --supersample 1 -o "${plane}"
        if [ "${DRY_RUN}" != "1" ] && { [ ! -f "${plane}.bin" ] || [ ! -f "${plane}.json" ]; }; then
            stamp "${stg}: ${sim}: no image plane, its snapshots are not run"; overall=1
            return 1
        fi
    fi
    return 0
}

stage_ps27() {
    local sim snap
    expect ps27 "8.5 h for the 27 (2026-10-12/13): per snapshot TNG100 5-17 min, TNG300 16-19 min, TNG50 12-40 min and 2.5 h at z = 0; 34.5-44.9 GB resident"
    heavy_guard ps27 || return
    for sim in TNG100-3-Dark TNG300-3-Dark TNG50-3-Dark; do
        in_sims "${sim}" || continue
        ensure_plane ps27 "${sim}" || continue
        for snap in ${THESIS_SNAPS}; do
            ps27_snapshot "${sim}" "${snap}"
        done
    done
}

stage_dtfe() {
    local sim="${OVERNIGHT_SIM}" snap
    expect dtfe "per snapshot ~10 min (the exact average at padding 15; sampled 5-7 min on 2026-10-12), ~33 GB footprint (3 snapshots)"
    heavy_guard dtfe || return
    for snap in 99 50 17; do                # z = 0, 1, 5 (config.SNAPSHOT_TO_REDSHIFT)
        # dtfe27's settings for TNG100-3-Dark (OVERNIGHT_SIM), these three snapshots alone, so a FORCE=1 rerun
        # cannot replace the production run's grids with another set
        dtfe_exact_snapshot dtfe "${sim}" "${snap}" "3 3 3" 10
    done
}

stage_caustics2() {
    local sim="${OVERNIGHT_SIM}" snap n d
    expect caustics2 "per snapshot: ~10 min, ~41 GB (TNG100 z = 1 with every field and the caustics, default Lagrangian padding: 8 min 57 s, 41.0 GB resident, 2026-10-12; 2 snapshots)"
    heavy_guard caustics2 || return
    for snap in 50 33; do                   # z = 1, 2
        n="$(printf '%03d' "${snap}")"; d="$(sim_dir "${sim}")/snapdir_${n}"
        if [ "${FORCE:-0}" != "1" ] && [ -s "${d}/ps_caustic.causticClass" ] && [ -f "${d}/ps_caustic.runlog" ]; then
            stamp "caustics2: ${sim} ${n}: ps_caustic.causticClass and ps_caustic.runlog are there: skipped (FORCE=1 reruns it)"
            continue
        fi
        # The z = 0 run's settings (its RUNNING line: --grid 512 --padding 50 --avg-subsamples 3
        # --field density_a tweb_a --lambda_th 0.3 --ps-gpu --ps-vertex-mass --ps-caustics, IC file, scratch dir)
        # pinned knob by knob: the series z = 0, 1, 2 must come from one recipe, and a default changed later or a
        # knob left in the caller's shell would split it -- the sampling, the vertex masses, the volume weighting
        # and the deposit change the grids themselves, the parallel triangulation makes a rerun differ at rounding.
        # The one knob NOT pinned to the z = 0 run is the Lagrangian padding: PS_LAG_PADDING= (empty, whatever the
        # caller's shell says) = the binary's default 8, not the z = 0 run's Eulerian rule (0). With the
        # padding-independent degeneracy cut (ps_cell_filter.h) the padding moves PS results only at float rounding,
        # and 8 runs ~6x faster in ~3.5x less memory.
        # Also the Lagrangian input as the driver finds it (-u: an empty LAGRANGIAN_INPUT forces the
        # snapshot path) and no auto-tune report (a leaked AUTO_TUNE_REPORT=1 would only report)
        drive caustics2 "${sim} ${n} caustics (prefix ps_caustic, budget 44 GB)" env -u LAGRANGIAN_INPUT DTFE_MEM_BUDGET_GB=44 \
            PS_CAUSTICS=1 OUTPUT_PREFIX=ps_caustic FIELDS="density_a tweb_a" PADDING=50 AVG_SUBSAMPLES=3 LAMBDA_TH=0.3 \
            PS_VERTEX_MASS=1 PS_VOLUME_WEIGHTED=0 PS_EXACT=0 PS_LINEAR_DEPOSIT=0 PS_PARALLEL_TRI=0 PS_LAG_PADDING= \
            PS_CAUSTIC_CUSPS=0 SAMPLE_POINTS= PARTITION= MAX_CONCURRENT= TESS_CACHE= AUTO_TUNE_REPORT=0 \
            "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g 512 -m "${snap}"
    done
}

# Where the figures are, by python's rules (dtfelib/cli.py FIGURES_ROOT: DTFE_FIGURES_ROOT, else the T7's
# 'Illustris TNG/figures' while it is mounted -- NOT derived from DTFE_DATA_ROOT --, else python/figures;
# config.py THESIS_FIGURES_DIR: the folder holding Thesis_Final.tex above the repo + /Figures, else <repo>/../Figures)
figures_root() {
    if [ -n "${DTFE_FIGURES_ROOT:-}" ]; then echo "${DTFE_FIGURES_ROOT}"
    elif [ -d "/Volumes/Samsung T7/Illustris TNG" ]; then echo "/Volumes/Samsung T7/Illustris TNG/figures"
    else echo "${REPO_ROOT}/python/figures"; fi
}
thesis_figures() {
    local d="${REPO_ROOT}"
    if [ -n "${THESIS_FIGURES:-}" ]; then echo "${THESIS_FIGURES}"; return; fi
    while [ "${d}" != "/" ] && [ -n "${d}" ]; do
        [ -e "${d}/Thesis_Final.tex" ] && { echo "${d}/Figures"; return; }
        d="$(dirname "${d}")"
    done
    echo "$(dirname "${REPO_ROOT}")/Figures"
}
# The untagged caustic skeleton files of the first ps_caustic run (made before the names carried the prefix: a
# --prefix ps_caustic run writes caustic_skeleton_z0.00_ps_caustic.*). Exactly these names: a glob
# 'caustic_skeleton_z0.00_*' would
# take the tagged set too.
OLD_SKELETON_FILES="caustic_skeleton_z0.00.json caustic_skeleton_z0.00.txt caustic_skeleton_z0.00.png caustic_skeleton_z0.00_voids.npz"
# move_old_skeletons DIR EVIDENCE_DIR: MOVES (never deletes) DIR's untagged files into DIR/old_untagged when the
# tagged caustic_skeleton_z0.00_ps_caustic.png is in DIR and the untagged JSON (in EVIDENCE_DIR or its old_untagged:
# the thesis tree mirrors only the png) says they came from the ps_caustic grids. An existing destination is not
# overwritten (said, and a failure). Under DRY_RUN the moves are printed, nothing moves. Returns 1 on a refusal.
move_old_skeletons() {
    local dir="$1" ev="$2" f have=() rc=0 j
    for f in ${OLD_SKELETON_FILES}; do [ -e "${dir}/${f}" ] && have+=("${f}"); done
    if [ "${#have[@]}" -eq 0 ]; then
        stamp "refresh: ${dir}: no untagged caustic_skeleton_z0.00 files (moved already, or never made): nothing to move"
        return 0
    fi
    if [ ! -e "${dir}/caustic_skeleton_z0.00_ps_caustic.png" ]; then
        stamp "refresh: ${dir}: the tagged caustic_skeleton_z0.00_ps_caustic.png is not there: the untagged files stay"
        return 1
    fi
    j=""
    for f in "${ev}/caustic_skeleton_z0.00.json" "${ev}/old_untagged/caustic_skeleton_z0.00.json"; do
        [ -f "${f}" ] && { j="${f}"; break; }
    done
    if [ -z "${j}" ] || ! grep -q '"prefix": "ps_caustic\.' "${j}"; then
        stamp "refresh: ${dir}: cannot tell that the untagged files came from the ps_caustic grids (${j:-no JSON}): they stay"
        return 1
    fi
    [ "${DRY_RUN}" = "1" ] || mkdir -p "${dir}/old_untagged" || return 1
    for f in "${have[@]}"; do
        if [ -e "${dir}/old_untagged/${f}" ]; then
            stamp "refresh: ${dir}/old_untagged/${f} exists already: ${f} NOT moved (nothing overwritten)"
            rc=1; continue
        fi
        echo "+ mv ${dir}/${f} ${dir}/old_untagged/${f}"
        [ "${DRY_RUN}" = "1" ] || mv "${dir}/${f}" "${dir}/old_untagged/${f}" || rc=1
    done
    return "${rc}"
}

stage_refresh() {
    local sim="${OVERNIGHT_SIM}" figs thesis
    expect refresh "20-40 min of python, ~15 GB (the convergence at three z and the skeleton's void null dominate)"
    unset DTFE_SIGMA_MPC                    # the production smoothing, as the voids stage
    # 1. the T-web streams table of the DEFAULT grids at z = 0 (the first ps_caustic run had replaced it; the
    #    names carry the prefix, so the ps_caustic one is tagged and this one is the default's)
    step refresh "${sim} web streams z=0 (T-web, ps_output)" env DTFE_SIM="${sim}" \
        "${PY}" python/plot/plot_web_streams.py --sim "${sim}" --snaps 99 --web tweb
    # 2. the void catalogue export (figures/void_catalog/<sim>/: the files there were pre-R_v corner-position
    #    exports); the production gate, whatever the caller's shell says (it only sets the well_resolved column)
    step refresh "${sim} void catalogue export (CSV + HDF5)" env DTFE_SIM="${sim}" DTFE_VOID_GATE=ellipsoid \
        "${PY}" python/analyze.py export
    # 3. convergence, ungated as in the voids stage, at z = 0, 1 AND 5: row (c) and its void-by-void table
    #    (estimator_voids_<snap>.txt) exist only where the dtfe stage left output.* grids
    step refresh "${sim} convergence z = 0 1 5 (ungated)" env DTFE_SIM="${sim}" DTFE_VOID_GATE=none \
        "${PY}" python/plot/plot_void_convergence.py --sim "${sim}" --method ps \
        --crossbox-smooth-mpc "${CROSSBOX_SMOOTH_MPC}" --z 0 1 5
    # 4. the caustic skeleton of every snapshot with ps_caustic grids (z = 0, and 1, 2 from caustics2), the shapes'
    #    gate as the voids stage draws it
    step refresh "${sim} caustic skeleton, every ps_caustic snapshot" env DTFE_SIM="${sim}" DTFE_VOID_GATE=ellipsoid \
        "${PY}" python/plot/plot_caustic_skeleton.py --sim "${sim}" --prefix ps_caustic --all-caustic --voids
    # 5. the old untagged z = 0 skeleton files: moved aside, never deleted, on the figure root and
    #    in the thesis tree (DTFE_NO_MIRROR=1: the thesis tree is left alone, as by the figure scripts)
    figs="$(figures_root)/caustic_skeleton/${sim}"
    stamp "refresh: old untagged caustic skeleton files -> old_untagged/ (${figs})"
    move_old_skeletons "${figs}" "${figs}" || { step_failed+=("refresh: move of the untagged skeleton files in ${figs}"); overall=1; }
    if [ -n "${DTFE_NO_MIRROR:-}" ]; then
        stamp "refresh: DTFE_NO_MIRROR set: the thesis Figures/ tree is left alone"
    else
        thesis="$(thesis_figures)/caustic_skeleton/${sim}"
        stamp "refresh: old untagged caustic skeleton files -> old_untagged/ (${thesis})"
        move_old_skeletons "${thesis}" "${figs}" || { step_failed+=("refresh: move of the untagged skeleton files in ${thesis}"); overall=1; }
    fi
}

stage_prune() {
    expect prune "seconds"
    # --keep-cells 3 2: the voids stage's size function at fixed R builds catalogues at 3 cells (and TNG300 at 3 and
    # 2), which cache_inventory does not count as current (config.SMOOTHING_COMPARISON_SIGMAS is 5 10 20): without it
    # the prune takes them and the next voids run rebuilds them. The
    # listing first, so the log shows what the delete took.
    step prune "cache: what --prune would delete (listing)" "${PY}" python/analyze.py cache --prune --keep-cells 3 2
    step prune "cache: --prune --yes (deletes the caches no current setting reaches)" \
        "${PY}" python/analyze.py cache --prune --yes --keep-cells 3 2
}

for s in "${stages[@]}"; do
    t_stage=${SECONDS}
    "stage_${s}"
    [ "${DRY_RUN}" = "1" ] || stamp "stage ${s} done ($(( (SECONDS - t_stage) / 60 )) min)"
done
stamp "ALL DONE ($([ "${overall}" = "0" ] && echo "every run succeeded" || echo "WITH FAILURES: see the exit lines above"))"
if [ "${#void_failed[@]}" -gt 0 ]; then         # the voids stage's failed figures, again where a long log ends
    printf 'failed void figure: %s\n' "${void_failed[@]}"
fi
if [ "${#step_failed[@]}" -gt 0 ]; then         # the overnight stages' failed steps, likewise
    printf 'failed step: %s\n' "${step_failed[@]}"
fi
exit "${overall}"
