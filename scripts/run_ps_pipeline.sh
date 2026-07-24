#!/usr/bin/env bash
# COMPUTE ONLY: for every snapshot of every simulation, regenerate the analysis grids
# (current physics: --ps-vertex-mass + per-field --ps-volume-weighted) AND point-evaluate a
# full-box high-resolution image plane of the continuous phase-space field. Both share the
# snapshot's single triangulation (run_ps_dtfe.sh SAMPLE_POINTS), so the image planes cost
# only the point evaluation on top of the grid regen.
#
# This script does no plotting. Render the figures afterwards with:
#     python3 python/plot/plot_pointeval.py            # all sims/snapshots, one file per field
# (the .pts_* files stay on disk, so figures can be restyled any time without recomputing).
#
#   ./run_ps_pipeline.sh                            # everything: all sims, all snapshots on disk
#   SIMS="TNG100-3-Dark" ./run_ps_pipeline.sh       # one simulation
#   DRY_RUN=1 ./run_ps_pipeline.sh                  # print the plan, run nothing
#
# RESUMABLE: a snapshot is skipped when its <prefix>.pts_den exists and is newer than both
# its combined_*.hdf5 and the plane file (and, with PTS_VEL_GRAD=1, its .pts_velGrad exists).
# Interrupted batches continue where they stopped. FORCE=1 recomputes everything.
#
# Env knobs (all optional):
#   SIMS          simulations to process (default: every dir under $DATA_ROOT with snapshots)
#   NU            image-plane pixels across the box (default 8192; ~13 kpc/px at TNG100-3)
#   AXIS, CENTER  plane orientation/position (default z, box centre). Changing CENTER with an
#                 existing plane file regenerates the plane -- which makes every snapshot's
#                 image plane stale, so the batch recomputes (as it must: geometry changed).
#   GRID_SIZE     analysis grid passed to run_ps_dtfe.sh (default 512)
#   FIELDS        forwarded to run_ps_dtfe.sh (default: its full production list)
#   FORCE         1 = ignore all freshness checks (default 0)
#   PTS_VEL_GRAD  1 (default) = also evaluate the velocity gradient at every sample point
#                 (.pts_velGrad), which is what makes the velDiv/velShear/velVort maps
#                 possible. A snapshot lacking it counts as STALE, so a rerun recomputes it.
# Snapshots are auto-discovered (snapdir_*/combined_NNN.hdf5). The plane file is generated
# once per simulation (box size is constant across its snapshots) and reused.
#
# DISK: the per-point files are KEPT (~9 GB per snapshot at NU=8192) -- they are the input
# to every figure. Delete <snapdir>/<prefix>.pts_vel* by hand once the figures are final.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # python/ and the binaries live one level up
USER_GRID="${GRID_SIZE:-}"          # capture BEFORE config.sh, which sets its own GRID_SIZE
source "${SCRIPT_DIR}/config.sh"

PY="${PY:-/opt/homebrew/bin/python3}"
NU="${NU:-8192}"
AXIS="${AXIS:-z}"
CENTER="${CENTER:-}"
GRID_SIZE="${USER_GRID:-512}"       # 512^3 analysis grids by default (figures don't depend on it)
PTS_VEL_GRAD="${PTS_VEL_GRAD:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PREFIX="${OUTPUT_PREFIX:-ps_output}"

if [ -z "${SIMS:-}" ]; then
    SIMS=""
    for d in "${DATA_ROOT}"/*/; do
        compgen -G "${d}snapdir_*/combined_[0-9][0-9][0-9].hdf5" > /dev/null && SIMS="${SIMS} $(basename "$d")"
    done
fi
echo "Simulations:${SIMS}"
echo "Slice: axis ${AXIS}, ${NU} px, centre ${CENTER:-box/2}   Grid: ${GRID_SIZE}^3   vel-grad: ${PTS_VEL_GRAD}   force: ${FORCE}"

run() { echo "+ $*"; [ "${DRY_RUN}" = "1" ] || "$@"; }

# fresh <target> <prereq...>: target exists and is strictly newer than every prerequisite
# (a missing prerequisite counts as stale -- we cannot verify against what is not there).
fresh() {
    local t="$1" p; shift
    [ -e "$t" ] || return 1
    for p in "$@"; do { [ -e "$p" ] && [ "$t" -nt "$p" ]; } || return 1; done
    return 0
}

overall_rc=0
for sim in ${SIMS}; do
    simdir="${DATA_ROOT}/${sim}"

    # ---- discover snapshots (numbers, sorted) --------------------------------------------
    snaps=()
    for f in "${simdir}"/snapdir_*/combined_[0-9][0-9][0-9].hdf5; do
        [ -e "$f" ] || continue
        n=$(basename "$f"); n=${n#combined_}; n=${n%.hdf5}
        snaps+=("$((10#$n))")
    done
    if [ ${#snaps[@]} -eq 0 ]; then echo "-- ${sim}: no combined snapshots, skipping"; continue; fi
    echo ""
    echo "== ${sim}: snapshots ${snaps[*]}"

    # ---- one plane file per simulation ---------------------------------------------------
    plane="${simdir}/hires_plane_${AXIS}${NU}"
    plane_new=0        # set when the plane is (re)generated: invalidates every slice/figure
    if [ -f "${plane}.json" ] && [ -n "${CENTER}" ]; then
        # a CENTER differing from the stored geometry must regenerate the plane (and thereby
        # invalidate every slice computed against the old one)
        same=$("${PY}" -c "import json,sys; c=json.load(open(sys.argv[1]))['center']; \
print(1 if abs(c-float(sys.argv[2]))<1e-9 else 0)" "${plane}.json" "${CENTER}") || same=0
        if [ "${same}" != "1" ]; then
            echo "-- ${sim}: CENTER=${CENTER} differs from the existing plane; regenerating"
            run rm -f "${plane}.bin" "${plane}.json"
            plane_new=1
        fi
    fi
    if [ "${plane_new}" = "1" ] || [ ! -f "${plane}.bin" ] || [ ! -f "${plane}.json" ]; then
        first=$(printf "%03d" "${snaps[0]}")
        center_args=(); [ -n "${CENTER}" ] && center_args=(--center "${CENTER}")
        run "${PY}" "${REPO_ROOT}/python/tools/make_image_plane.py" \
            --combined "${simdir}/snapdir_${first}/combined_${first}.hdf5" \
            --axis "${AXIS}" --nu "${NU}" --planes 1 ${center_args[@]+"${center_args[@]}"} \
            -o "${plane}" || { echo "!! ${sim}: plane generation failed"; overall_rc=1; continue; }
        plane_new=1
    else
        echo "-- reusing ${plane}.bin"
    fi

    # ---- grids + point evaluation, one triangulation per snapshot ------------------------
    # resume: only snapshots whose slice is missing or older than its inputs are computed
    compute=()
    for n in "${snaps[@]}"; do
        sd="${simdir}/snapdir_$(printf "%03d" "$n")"
        cf="${sd}/combined_$(printf "%03d" "$n").hdf5"
        # with PTS_VEL_GRAD on, a snapshot lacking .pts_velGrad is stale even if its
        # .pts_den is current (it predates the flag)
        vg_ok=1
        if [ "${PTS_VEL_GRAD}" = "1" ] && [ ! -f "${sd}/${PREFIX}.pts_velGrad" ]; then vg_ok=0; fi
        if [ "${FORCE}" != "1" ] && [ "${plane_new}" != "1" ] && [ "${vg_ok}" = "1" ] \
            && fresh "${sd}/${PREFIX}.pts_den" "${cf}" "${plane}.bin"; then
            echo "-- snap ${n}: ${PREFIX}.pts_den up to date, skipping compute"
        else
            compute+=("$n")
        fi
    done
    if [ ${#compute[@]} -gt 0 ]; then
        env_args=(SAMPLE_POINTS="${plane}.bin" PS_VOLUME_WEIGHTED=1 PTS_VEL_GRAD="${PTS_VEL_GRAD}")
        [ -n "${FIELDS:-}" ] && env_args+=(FIELDS="${FIELDS}")
        run env "${env_args[@]}" \
            "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g "${GRID_SIZE}" -m "${compute[@]}" \
            || { echo "!! ${sim}: run_ps_dtfe.sh reported failure (continuing to plots)"; overall_rc=1; }
    else
        echo "-- ${sim}: all snapshots up to date"
    fi

done

echo ""
[ ${overall_rc} -eq 0 ] && echo "ALL DONE (compute). Render figures with: python3 python/plot/plot_pointeval.py" \
    || echo "DONE WITH FAILURES (see !! lines above)"
exit ${overall_rc}
