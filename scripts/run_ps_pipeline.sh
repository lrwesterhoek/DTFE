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
# its combined_*.hdf5 and the plane file (and, with PTS_VEL_GRAD=1, its .pts_velGrad exists),
# AND the settings in its <prefix>.runlog match this run's (grid, sub-samples, padding, lambda_th,
# precision, the deposit options, and every requested field among its fields; not the GPU/CPU
# choice, the split or the build).
# A snapshot without a run log is judged by the file times alone (said per snapshot).
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

# run_settings <command line>: the options of a run_ps_dtfe.sh command that change its grids, one per line,
# sorted -- from the RUNNING: line a run log starts with, or from a DRY_RUN=1 'would run' line (its %q
# quoting only touches the paths, which are not compared). The binary's name carries the precision.
run_settings() {
    printf '%s\n' "$1" | awk '
        BEGIN { split("--grid --padding --avg-subsamples --field --lambda_th", v); for (i in v) val[v[i]] = 1
                split("--periodic --ps-vertex-mass --ps-volume-weighted --ps-exact-deposit --ps-linear-deposit " \
                      "--ps-caustics --ps-caustic-cusps --pts-vel-grad", b); for (i in b) flag[b[i]] = 1 }
        { i = 1; if ($1 == "RUNNING:") i = 2
          bin = $i; sub(/.*\//, "", bin); print "binary " bin
          cur = ""; line = ""
          for (i = i + 1; i <= NF; i++) {
              if ($i ~ /^--/) {
                  if (line != "") print line
                  line = ""; cur = ""
                  if ($i in val) { cur = $i; line = $i } else if ($i in flag) print $i
              } else if (cur != "") line = line " " $i
          }
          if (line != "") print line }' | sort
}

# same_settings <have> <want> (run_settings outputs): equal, except that the grids on disk may carry MORE
# fields than asked for (the thesis set adds tweb_a/vweb_a to the pipeline's default list: a rerun with the
# default must not recompute grids it already has)
same_settings() {
    [ "$(printf '%s\n' "$1" | grep -v '^--field ')" = "$(printf '%s\n' "$2" | grep -v '^--field ')" ] || return 1
    local have=" $(printf '%s\n' "$1" | sed -n 's/^--field //p') " f
    for f in $(printf '%s\n' "$2" | sed -n 's/^--field //p'); do
        case "${have}" in *" ${f} "*) ;; *) return 1 ;; esac
    done
    return 0
}

# fresh <target> <prereq...>: target exists and is strictly newer than every prerequisite
# (a missing prerequisite counts as stale -- we cannot verify against what is not there).
fresh() {
    local t="$1" p; shift
    [ -e "$t" ] || return 1
    for p in "$@"; do { [ -e "$p" ] && [ "$t" -nt "$p" ]; } || return 1; done
    return 0
}

overall_rc=0
n_found=0          # snapshots found (after SNAPS) over all simulations; 0 at the end is an error
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
    if [ ${#snaps[@]} -eq 0 ]; then echo "-- ${sim}: no combined snapshots${SNAPS:+ numbered ${SNAPS}}, skipping"; continue; fi
    n_found=$((n_found + ${#snaps[@]}))
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
    # PS_METAL follows PS_GPU too: -m only switches the GPU ON, an exported PS_METAL=1 would otherwise leak in
    env_args=(SAMPLE_POINTS="${plane}.bin" PS_VOLUME_WEIGHTED="${PS_VOLUME_WEIGHTED}" PTS_VEL_GRAD="${PTS_VEL_GRAD}"
              PS_METAL="${PS_GPU}")
    [ -n "${FIELDS:-}" ] && env_args+=(FIELDS="${FIELDS}")
    # The settings this batch would run each snapshot with, asked from run_ps_dtfe.sh itself (DRY_RUN=1: reads,
    # runs and writes nothing), so the comparison below cannot drift from what it runs (2026-10-11: the check
    # compared file times only, so a changed grid or field list printed 'up to date' for every snapshot).
    planned=""
    if [ "${FORCE}" != "1" ] && [ "${plane_new}" != "1" ]; then
        planned="$(env "${env_args[@]}" DRY_RUN=1 "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g "${GRID_SIZE}" \
                       ${gpu_args[@]+"${gpu_args[@]}"} "${snaps[@]}" 2>/dev/null \
                   | awk '/^Processing snapshot / { s = $3; sub(/\.+$/, "", s) }
                          /DRY RUN -- would run:/ { want = 1; next }
                          want { sub(/^ +/, ""); print s "\t" $0; want = 0 }')"
    fi
    # resume: only snapshots whose slice is missing, older than its inputs, or made with other settings are computed
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
            rl="${sd}/${PREFIX}.runlog"
            want="$(printf '%s\n' "${planned}" | awk -F '\t' -v s="$(printf "%03d" "$n")" '$1 == s { print $2; exit }')"
            if [ ! -f "${rl}" ]; then
                echo "-- snap ${n}: ${PREFIX}.pts_den up to date by file times (no ${PREFIX}.runlog to compare the settings with), skipping compute"
            elif [ -z "${want}" ]; then
                echo "-- snap ${n}: ${PREFIX}.pts_den up to date by file times (this run's settings could not be worked out: run_ps_dtfe.sh's dry run failed), skipping compute"
            else
                have_s="$(run_settings "$(grep -a -m1 '^RUNNING:' "${rl}")")"
                want_s="$(run_settings "${want}")"
                if same_settings "${have_s}" "${want_s}"; then
                    echo "-- snap ${n}: ${PREFIX}.pts_den up to date (same settings), skipping compute"
                else
                    echo "-- snap ${n}: settings differ from its ${PREFIX}.runlog, recomputing"
                    comm -23 <(printf '%s\n' "${have_s}") <(printf '%s\n' "${want_s}") | sed 's/^/     was: /'
                    comm -13 <(printf '%s\n' "${have_s}") <(printf '%s\n' "${want_s}") | sed 's/^/     now: /'
                    compute+=("$n")
                fi
            fi
        else
            compute+=("$n")
        fi
    done
    if [ ${#compute[@]} -gt 0 ]; then
        run env "${env_args[@]}" \
            "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g "${GRID_SIZE}" ${gpu_args[@]+"${gpu_args[@]}"} "${compute[@]}" \
            || { echo "!! ${sim}: run_ps_dtfe.sh reported failure (continuing with the next simulation)"; overall_rc=1; }
    else
        echo "-- ${sim}: all snapshots up to date"
    fi

done

echo ""
# Finding NOTHING is a failure, as in run_ps_dtfe.sh (2026-10-08): with the data drive unmounted, a
# mistyped SIMS or a SNAPS matching no snapshot, nothing ran and this still printed ALL DONE with exit 0,
# which the launcher's Pipeline tab showed as a success. Every snapshot found is either computed, up to
# date, or in a simulation that failed with a '!!' line (overall_rc 1), so a partial skip stays exit 0.
if [ "${n_found}" -eq 0 ]; then
    echo "!! nothing to compute: no combined snapshot${SNAPS:+ numbered ${SNAPS}} of $([ -n "${SIMS// }" ] && echo ${SIMS} || echo "any simulation") under '${DATA_ROOT}'"
    if [ ! -d "${DATA_ROOT}" ]; then
        echo "   That folder does not exist: is the data drive mounted?"
    else
        echo "   Check SIMS and SNAPS, and the data root (DTFE_DATA_ROOT)."
    fi
    overall_rc=1
fi
[ ${overall_rc} -eq 0 ] && echo "ALL DONE (compute). Render figures with: python3 python/plot/plot_pointeval.py" \
    || echo "DONE WITH FAILURES (see !! lines above)"
exit ${overall_rc}
