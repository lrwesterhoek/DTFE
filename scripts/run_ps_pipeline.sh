#!/usr/bin/env bash
# COMPUTE ONLY: for every snapshot of every simulation, regenerate the analysis grids
# (current physics: --ps-vertex-mass + per-field --ps-volume-weighted, both defaults) AND point-evaluate a
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
#   ./run_ps_pipeline.sh --help                     # this header, nothing run (--version: the revision)
#
# RESUMABLE: a snapshot is skipped when its <prefix>.pts_den exists and is newer than both
# its combined_*.hdf5 and the plane file (and, with PTS_VEL_GRAD=1, its .pts_velGrad exists).
# Interrupted batches continue where they stopped. FORCE=1 recomputes everything.
#
# Env knobs (all optional):
#   SIMS          simulations to process (default: every simulation under $DATA_ROOT with
#                 snapshots, in either layout -- flat <sim>/ or per-family <family>/<sim>/)
#   NU            image-plane pixels across the box (default 8192; ~13 kpc/px at TNG100-3)
#   AXIS, CENTER  plane orientation/position (default z, box centre). Changing CENTER with an
#                 existing plane file regenerates the plane -- which makes every snapshot's
#                 image plane stale, so the batch recomputes (as it must: geometry changed).
#   GRID_SIZE     analysis grid passed to run_ps_dtfe.sh (default 512)
#   FIELDS        forwarded to run_ps_dtfe.sh (default: its full production list)
#   SNAPS         only these snapshot numbers, e.g. "50 67 99" (default: every snapshot on disk);
#                 the GUI's adaptive plane size runs the pipeline once per plane size with it
#   FORCE         1 = ignore all freshness checks (default 0)
#   PTS_VEL_GRAD  1 (default) = also evaluate the velocity gradient at every sample point
#                 (.pts_velGrad), which is what makes the velDiv/velShear/velVort maps
#                 possible. A snapshot lacking it counts as STALE, so a rerun recomputes it.
#   PLANES, THICKNESS, SUPERSAMPLE, WINDOW
#                 the plane's geometry, forwarded to make_image_plane.py: PLANES sampling planes
#                 (default 1 = one crisp cross-section; a handful ghosts, use >= 16 with
#                 plot_pointeval.py --project slab) across a slab of THICKNESS Mpc (default 2), KxK sub-samples per pixel
#                 averaged into pixel-area means (default 1), and "u0 u1 v0 v1" in Mpc to image
#                 only that part of the plane (default: the whole box).
#   PS_GPU        1 (default) = the deposit on the GPU (run_ps_dtfe.sh -m); 0 = CPU
#   PS_VOLUME_WEIGHTED  1 (default) = volume-weighted velocity moments, the production convention
# The other run options reach run_ps_dtfe.sh through the environment exactly as from a terminal:
# PS_EXACT, AVG_SUBSAMPLES, PS_VERTEX_MASS, PS_CAUSTICS, PS_CAUSTIC_CUSPS, PS_PARALLEL_TRI,
# PARTITION, MAX_CONCURRENT, SCRATCH_DIR, TESS_CACHE, DTFE_PRECISION, OUTPUT_PREFIX, LAMBDA_TH,
# THREADS (see its header for each). The GUI's Pipeline tab sets all of them.
# Snapshots are auto-discovered (snapdir_*/combined_NNN.hdf5). The plane file is generated
# once per simulation (box size is constant across its snapshots) and reused -- as long as its
# sidecar has the requested geometry (make_image_plane.py --same-as, exit 3 = differs); otherwise it
# is regenerated, which makes every snapshot of that simulation stale (the geometry changed, so it
# must). A comparison that FAILS (no python, a bad WINDOW, a missing module) is not 'differs': the
# plane is kept, the simulation skipped with a '!!' line, and the script exits 1.
#
# DISK: the per-point files are KEPT (~9 GB per snapshot at NU=8192) -- they are the input
# to every figure. Delete <snapdir>/<prefix>.pts_vel* by hand once the figures are final.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # python/ and the binaries live one level up
# -h / --help print this header and run nothing (a '--help' used to START THE WHOLE BATCH: the script took no
# arguments at all); --version the revision; anything else is an error -- the knobs are environment variables
case "${1:-}" in
    -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//'; exit 0 ;;
    --version) git -C "${REPO_ROOT}" describe --always --dirty 2>/dev/null || echo unknown; exit 0 ;;
    "") ;;
    *) echo "Error: run_ps_pipeline.sh takes no arguments (its knobs are environment variables; --help lists them)" >&2; exit 2 ;;
esac
USER_GRID="${GRID_SIZE:-}"          # capture BEFORE config.sh, which sets its own GRID_SIZE
source "${SCRIPT_DIR}/config.sh"

PY="${PY:-$(command -v /opt/homebrew/bin/python3 || command -v python3)}"   # the launcher passes its own PY
NU="${NU:-8192}"
AXIS="${AXIS:-z}"
CENTER="${CENTER:-}"
GRID_SIZE="${USER_GRID:-512}"       # 512^3 analysis grids by default (figures don't depend on it)
PTS_VEL_GRAD="${PTS_VEL_GRAD:-1}"
PLANES="${PLANES:-1}"
THICKNESS="${THICKNESS:-2}"
SUPERSAMPLE="${SUPERSAMPLE:-1}"
WINDOW="${WINDOW:-}"
PS_GPU="${PS_GPU:-1}"
PS_VOLUME_WEIGHTED="${PS_VOLUME_WEIGHTED:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PREFIX="${OUTPUT_PREFIX:-ps_output}"

# the plane's geometry, as make_image_plane.py takes it (shared by the generation and the --same-as check)
plane_args=(--axis "${AXIS}" --nu "${NU}" --planes "${PLANES}" --thickness "${THICKNESS}" --supersample "${SUPERSAMPLE}")
[ -n "${CENTER}" ] && plane_args+=(--center "${CENTER}")
if [ -n "${WINDOW}" ]; then
    read -r w_u0 w_u1 w_v0 w_v1 w_more <<< "${WINDOW}"
    if [ -z "${w_v1}" ] || [ -n "${w_more:-}" ]; then
        echo "!! WINDOW must be four numbers 'u0 u1 v0 v1' (Mpc), got '${WINDOW}'" >&2; exit 2
    fi
    plane_args+=(--u0 "${w_u0}" --u1 "${w_u1}" --v0 "${w_v0}" --v1 "${w_v1}")
fi
gpu_args=(); [ "${PS_GPU}" = "1" ] && gpu_args=(-m)

if [ -z "${SIMS:-}" ]; then
    SIMS=""
    for d in "${DATA_ROOT}"/*/ "${DATA_ROOT}"/*/*/; do
        compgen -G "${d}snapdir_*/combined_[0-9][0-9][0-9].hdf5" > /dev/null || continue
        case " ${SIMS} " in *" $(basename "$d") "*) ;; *) SIMS="${SIMS} $(basename "$d")" ;; esac
    done
fi
echo "Simulations:${SIMS}"
echo "Slice: axis ${AXIS}, ${NU} px, centre ${CENTER:-box/2}, planes ${PLANES}$([ "${PLANES}" != "1" ] && echo " over ${THICKNESS} Mpc"), ${SUPERSAMPLE}x${SUPERSAMPLE} sub-samples${WINDOW:+, window ${WINDOW}}   Grid: ${GRID_SIZE}^3   GPU: ${PS_GPU}   vel-grad: ${PTS_VEL_GRAD}   force: ${FORCE}"

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
    simdir="$(sim_dir "${sim}")"

    # ---- discover snapshots (numbers, sorted) --------------------------------------------
    snaps=()
    for f in "${simdir}"/snapdir_*/combined_[0-9][0-9][0-9].hdf5; do
        [ -e "$f" ] || continue
        n=$(basename "$f"); n=${n#combined_}; n=${n%.hdf5}
        n=$((10#$n))
        if [ -n "${SNAPS:-}" ]; then
            case " ${SNAPS} " in *" ${n} "*|*" $(printf "%03d" "${n}") "*) ;; *) continue ;; esac
        fi
        snaps+=("${n}")
    done
    if [ ${#snaps[@]} -eq 0 ]; then echo "-- ${sim}: no combined snapshots, skipping"; continue; fi
    echo ""
    echo "== ${sim}: snapshots ${snaps[*]}"

    # ---- one plane file per simulation ---------------------------------------------------
    plane="${simdir}/hires_plane_${AXIS}${NU}"
    plane_new=0        # set when the plane is (re)generated: invalidates every slice/figure
    if [ -f "${plane}.json" ]; then
        # a stored geometry differing from the requested one (CENTER when given, planes, thickness,
        # sub-samples, window) must regenerate the plane -- and thereby invalidate every slice
        # computed against the old one. Exit 3 = differs (or an unreadable sidecar); any other failure
        # of the tool (no python, a bad WINDOW, a missing module) keeps the plane and skips the sim.
        same_rc=0
        same_err="$("${PY}" "${REPO_ROOT}/python/tools/make_image_plane.py" --same-as "${plane}.json" \
                    "${plane_args[@]}" 2>&1 >/dev/null)" || same_rc=$?
        if [ "${same_rc}" = "3" ]; then
            echo "-- ${sim}: the existing plane's geometry (centre, planes, thickness, sub-samples or window) differs; regenerating"
            run rm -f "${plane}.bin" "${plane}.json"
            plane_new=1
        elif [ "${same_rc}" != "0" ]; then
            echo "!! ${sim}: could not compare the plane's geometry (make_image_plane.py --same-as exited ${same_rc}); the plane is kept, the simulation skipped"
            [ -n "${same_err}" ] && echo "${same_err}" | sed 's/^/   /'
            overall_rc=1; continue
        fi
    fi
    if [ "${plane_new}" = "1" ] || [ ! -f "${plane}.bin" ] || [ ! -f "${plane}.json" ]; then
        first=$(printf "%03d" "${snaps[0]}")
        run "${PY}" "${REPO_ROOT}/python/tools/make_image_plane.py" \
            --combined "${simdir}/snapdir_${first}/combined_${first}.hdf5" \
            "${plane_args[@]}" \
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
        # PS_METAL follows PS_GPU too: -m only switches the GPU ON, an exported PS_METAL=1 would otherwise leak in
        env_args=(SAMPLE_POINTS="${plane}.bin" PS_VOLUME_WEIGHTED="${PS_VOLUME_WEIGHTED}" PTS_VEL_GRAD="${PTS_VEL_GRAD}"
                  PS_METAL="${PS_GPU}")
        [ -n "${FIELDS:-}" ] && env_args+=(FIELDS="${FIELDS}")
        run env "${env_args[@]}" \
            "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g "${GRID_SIZE}" ${gpu_args[@]+"${gpu_args[@]}"} "${compute[@]}" \
            || { echo "!! ${sim}: run_ps_dtfe.sh reported failure (continuing with the next simulation)"; overall_rc=1; }
    else
        echo "-- ${sim}: all snapshots up to date"
    fi

done

echo ""
[ ${overall_rc} -eq 0 ] && echo "ALL DONE (compute). Render figures with: python3 python/plot/plot_pointeval.py" \
    || echo "DONE WITH FAILURES (see !! lines above)"
exit ${overall_rc}
