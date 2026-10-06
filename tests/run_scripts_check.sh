#!/usr/bin/env bash
# Contract check for the batch driver scripts/run_ps_dtfe.sh (and the shared scripts/config.sh, and
# run_dtfe.sh's exact-average switch).
#
# Every other suite tests the BINARY; this one tests the shell around it, which is where two silent
# failures lived unnoticed (a discarded environment variable, and snapshots skipped for a file they
# did not need). Runs against a FAKE DATA_ROOT built here, so it never touches real data.
#
#  A) GRID_SIZE from the ENVIRONMENT reaches the binary, and '-g' still beats it:
#     precedence must be  -g flag > environment > config.sh default.
#  B) A snapshot carrying its own 'InitialCoordinates' is PROCESSED without a combined_ics.hdf5 --
#     PS-DTFE reads them when '--lagrangianInput' is omitted, so skipping it drops usable data.
#  C) A combined_ics.hdf5, when present, is still passed as '--lagrangianInput' (B must not have
#     moved everyone onto the snapshot path), and an empty LAGRANGIAN_INPUT= forces it off.
#  D) A snapshot with NEITHER source is skipped with a message naming both, and the run exits 1.
#  E) Exit status is honest: processing nothing exits 1, while a PARTIAL skip stays 0 -- a
#     partly-downloaded snapshot ladder is normal.
#  F) No macOS-only command on the critical path ('/usr/bin/time -l', 'sed -i ""'), which used to
#     kill every snapshot on Linux. Asserted structurally, so it holds on either platform.
#  G) The PER-FAMILY data layout of the Samsung T7 (<root>/<family>/<sim>, e.g. Illustris TNG/
#     TNG100/TNG100-3-Dark), under a root whose name has a space like the T7's: the driver finds
#     and processes the simulation there, config.sh's sim_dir puts a new simulation beside its
#     family and lets a flat <root>/<sim> win, and run_ps_pipeline.sh discovers nested sims.
#  H) scripts/run_dtfe.sh -e / DTFE_EXACT_AVERAGE=1 passes --exact-average to the standard binary.
#  I) download_snapshots.sh + merge_HDF5.py never turn a half-finished download into a merged snapshot.
#  J) --help runs nothing (run_ps_pipeline.sh --help used to start the whole batch), --version prints the
#     revision, an argument to the pipeline is an error, DRY_RUN=1 prints the drivers' commands and runs nothing.
#  K) the binaries' own --help lists the phase-space / GPU / point-evaluation / server options the scripts and the
#     launcher use (it listed none of them), and the help texts carry none of the survey's misspellings.
#  L) scripts/install_lib.sh keeps a user's binaries through a failed rebuild: backup, restore, the failure hint.
#
# Usage: tests/run_scripts_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the drivers run the double pair
precision_double && export DTFE_PRECISION=double     # the drivers' own switch (run_ps_dtfe.sh / run_dtfe.sh)

# python with a working h5py (plain python3 resolves to a broken x86_64 h5py on the dev machine)
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
export PY                               # run_ps_pipeline.sh runs make_image_plane.py with the same python

N="${N:-16}"
GRID="${GRID:-24}"
BOX="${BOX:-100.0}"
TMP="${TMP_DIR:-${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}}/run_scripts"
export DTFE_FIGURES_ROOT="${TMP}/figures"   # the pipeline's plots: never the real figure root (the T7 when mounted)
DRIVER="scripts/run_ps_dtfe.sh"

echo "============================================================"
echo " run_ps_dtfe.sh contract check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
precision_require "${PS_BIN}"
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} not built"; exit 1; }

fails=0
check() {
    if [ "$1" = "0" ]; then printf '   PASS  %s\n' "$2"; else printf '   FAIL  %s\n' "$2"; fails=$((fails+1)); fi
}
# Does <file> ($1) match extended-regex <pattern> ($2)? Echo 0 for yes, 1 for no, so the result
# feeds check() directly. '-e' is mandatory here: nearly every pattern below starts with '--'
# (the flags we are asserting about), which grep would otherwise read as its own option.
has()   { grep -aqE -e "$2" "$1" && echo 0 || echo 1; }
lacks() { grep -aqE -e "$2" "$1" && echo 1 || echo 0; }

rm -rf "${TMP}"
mkdir -p "${TMP}/HASIC/snapdir_099" "${TMP}/NOIC/snapdir_099" "${TMP}/PARTIAL/snapdir_099"

echo ">> building the fake DATA_ROOT ..."
# HASIC: snapshot carries InitialCoordinates, no combined_ics.hdf5 beside it
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/HASIC/snapdir_099/combined_099.hdf5" \
        --n "${N}" --box "${BOX}" >/dev/null
# NOIC: same snapshot with InitialCoordinates removed, and still no combined_ics.hdf5
"${PY}" - "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${TMP}/NOIC/snapdir_099/combined_099.hdf5" <<'PYEOF'
import shutil, sys
import h5py
shutil.copy(sys.argv[1], sys.argv[2])
with h5py.File(sys.argv[2], "a") as f:
    del f["PartType1"]["InitialCoordinates"]
PYEOF
# PARTIAL: one good snapshot; snapshot 098 is requested below but never created
cp "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${TMP}/PARTIAL/snapdir_099/combined_099.hdf5"

export DTFE_DATA_ROOT="${TMP}"
# keep every run tiny and quiet: one field, no averaging sub-samples worth mentioning
COMMON_ENV=(FIELDS=density_a AVG_SUBSAMPLES=1)

# Runs the driver with the given env assignments, capturing stdout+stderr and the exit code.
# Echoes the exit code; the log path is "$2".
drive() {
    local log="$1"; shift
    local rc=0
    env "${COMMON_ENV[@]}" "$@" "${DRIVER}" 99 >"${log}" 2>&1 || rc=$?
    echo "${rc}"
}

echo ""
echo "(A) GRID_SIZE from the environment reaches the binary, and -g still overrides it."
rc=$(drive "${TMP}/a_env.log" GRID_SIZE=24 DTFE_SIM=HASIC)
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "env-only run succeeds (exit ${rc})"
check "$(has "${TMP}/a_env.log" '--grid 24( |$)')" "GRID_SIZE=24 reached the binary as '--grid 24'"
check "$(has "${TMP}/a_env.log" 'Grid size: 24')" "GRID_SIZE=24 is what the script reports"
if precision_double; then
    check "$(has "${TMP}/a_env.log" 'PS-DTFE-double ')" "DTFE_PRECISION=double: the driver ran ./PS-DTFE-double"
fi
# -g must win over the environment
env "${COMMON_ENV[@]}" GRID_SIZE=24 DTFE_SIM=HASIC "${DRIVER}" -g 32 99 >"${TMP}/a_flag.log" 2>&1 || true
check "$(has   "${TMP}/a_flag.log" '--grid 32( |$)')" "-g 32 beats GRID_SIZE=24 (flag > environment)"
check "$(lacks "${TMP}/a_flag.log" '--grid 24( |$)')" \
      "the overridden GRID_SIZE=24 appears nowhere in the command"
# and with neither, config.sh's own default stands
default_grid=$(bash -c 'source scripts/config.sh; echo "${GRID_SIZE}"')
check "$([ "${default_grid}" = "1024" ] && echo 0 || echo 1)" \
      "config.sh default is untouched when nothing overrides it (got ${default_grid})"

echo ""
echo "(B) a snapshot with its own 'InitialCoordinates' runs WITHOUT a combined_ics.hdf5."
check "$(has "${TMP}/a_env.log" "Lagrangian positions: the snapshot's own 'InitialCoordinates'")" \
      "the script says it is using the snapshot's own Lagrangian positions"
check "$(has "${TMP}/a_env.log" "read from 'InitialCoordinates' dataset in main input file")" \
      "the BINARY confirms it read them from the snapshot"
check "$(lacks "${TMP}/a_env.log" '--lagrangianInput')" "no --lagrangianInput was passed"
check "$([ -f "${TMP}/HASIC/snapdir_099/ps_output.a_den" ] && echo 0 || echo 1)" \
      "the density grid was actually written"

echo ""
echo "(C) a combined_ics.hdf5 beside the snapshot is still used, and can be forced off."
cp "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${TMP}/HASIC/combined_ics.hdf5"
rc=$(drive "${TMP}/c_ics.log" GRID_SIZE=24 DTFE_SIM=HASIC)
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "run with an IC file succeeds (exit ${rc})"
check "$(has "${TMP}/c_ics.log" '--lagrangianInput .*combined_ics\.hdf5')" \
      "--lagrangianInput points at the IC file"
# explicitly empty LAGRANGIAN_INPUT must fall back to the snapshot even though the file exists
rc=$(drive "${TMP}/c_forced.log" GRID_SIZE=24 DTFE_SIM=HASIC LAGRANGIAN_INPUT=)
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "LAGRANGIAN_INPUT= run succeeds (exit ${rc})"
check "$(has "${TMP}/c_forced.log" "read from 'InitialCoordinates' dataset in main input file")" \
      "LAGRANGIAN_INPUT= forces the snapshot path despite the IC file"
rm -f "${TMP}/HASIC/combined_ics.hdf5"

echo ""
echo "(D) a snapshot with NEITHER Lagrangian source is skipped, naming both, and exits 1."
rc=$(drive "${TMP}/d_none.log" GRID_SIZE=24 DTFE_SIM=NOIC)
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "exit status is 1 (got ${rc})"
check "$(has "${TMP}/d_none.log" 'has NO Lagrangian positions')" "the warning names the real problem"
check "$(has "${TMP}/d_none.log" "no 'InitialCoordinates' dataset either")" \
      "it says the snapshot itself was checked too"
check "$(has "${TMP}/d_none.log" 'Skipping snapshot 099')" "the snapshot is skipped rather than failed"

echo ""
echo "(E) exit status is honest about doing nothing, and forgiving about partial ladders."
rc=$(drive "${TMP}/e_nothing.log" GRID_SIZE=24 DTFE_SIM=DOES_NOT_EXIST)
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a wrong DTFE_SIM exits 1, not 0 (got ${rc})"
check "$(has "${TMP}/e_nothing.log" '0 processed')" "the summary reports 0 processed"
# 98 is absent, 99 is fine: a partly-downloaded ladder must still succeed
rc=0
env "${COMMON_ENV[@]}" GRID_SIZE=24 DTFE_SIM=PARTIAL "${DRIVER}" 98 99 >"${TMP}/e_partial.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "a partial ladder still exits 0 (got ${rc})"
check "$(has "${TMP}/e_partial.log" '1 processed, 1 skipped')" "the summary counts both outcomes"

echo ""
echo "(F) no macOS-only command sits on the critical path (so Linux/CI can run this driver)."
# The test is about the INVOCATION, not the mention: a guarded probe of '/usr/bin/time -l' is
# exactly how the portable wrapper decides which dialect exists, so grepping for the string alone
# would fail on the very code that fixes the problem. What must not exist is time(1) sitting
# directly in front of the binary.
check "$(lacks "${DRIVER}" '/usr/bin/time[^|]*\./PS-DTFE')" \
      "time(1) does not prefix the binary unconditionally (GNU spells -l as -v; containers have neither)"
check "$(has   "${DRIVER}" 'TIME_WRAP\[@\].*"\${cmd\[@\]}"')" \
      "the binary is invoked through the resolved-once TIME_WRAP instead (PS_BIN: the pair chosen by DTFE_PRECISION)"
check "$(lacks "${DRIVER}" "sed .*-i +''")" \
      "no BSD-only \"sed -i ''\" (GNU sed reads the '' as the script, not as the suffix)"
# and the run really did produce its log without a time(1) present
check "$([ -f "${TMP}/HASIC/snapdir_099/ps_output.runlog" ] && echo 0 || echo 1)" \
      "the .runlog is written and de-ANSI'd whether or not time(1) exists"

echo ""
echo "(G) the per-family layout (<root>/<family>/<sim>) under a root with a space in its name."
FAMROOT="${TMP}/fam root"
mkdir -p "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099"
cp "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/"
rc=0
# GRID_SIZE=24 like every other section: without it the config default 1024^3 (~14 GB of grids)
# ran here -- slow on a 64 GB Mac, and killed for memory in a 16 GB Linux container
env "${COMMON_ENV[@]}" GRID_SIZE=24 DTFE_DATA_ROOT="${FAMROOT}" DTFE_SIM=FAM-1-Dark "${DRIVER}" 99 \
    >"${TMP}/g_family.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "a run on the per-family layout succeeds (exit ${rc})"
check "$([ -f "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.a_den" ] && echo 0 || echo 1)" \
      "its grids land in <root>/FAM/FAM-1-Dark/snapdir_099"
# sim_dir: an unknown simulation of an existing family goes beside its siblings; a flat
# <root>/<sim> directory, when present, wins over the family folder
newsim="$(DTFE_DATA_ROOT="${FAMROOT}" bash -c 'source scripts/config.sh; sim_dir FAM-2-Dark')"
check "$([ "${newsim}" = "${FAMROOT}/FAM/FAM-2-Dark" ] && echo 0 || echo 1)" \
      "sim_dir places a new FAM-2-Dark in its family folder (got '${newsim}')"
mkdir -p "${FAMROOT}/FAM-3-Dark"
flat="$(DTFE_DATA_ROOT="${FAMROOT}" bash -c 'source scripts/config.sh; sim_dir FAM-3-Dark')"
check "$([ "${flat}" = "${FAMROOT}/FAM-3-Dark" ] && echo 0 || echo 1)" \
      "an existing flat <root>/FAM-3-Dark wins (got '${flat}')"
DRY_RUN=1 DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline.log" 2>&1 || true
check "$(has "${TMP}/g_pipeline.log" '^Simulations:.* FAM-1-Dark')" \
      "run_ps_pipeline.sh discovers the nested FAM-1-Dark"
# the pipeline's knobs (2026-10-04): the GPU flag follows PS_GPU, the plane's geometry reaches
# make_image_plane.py, and a plane whose sidecar has another geometry is regenerated
check "$(has "${TMP}/g_pipeline.log" 'run_ps_dtfe.sh -s FAM-1-Dark -g [0-9]+ -m 99')" \
      "the pipeline's default run_ps_dtfe.sh call carries -m (PS_GPU=1)"
check "$(has "${TMP}/g_pipeline.log" 'make_image_plane.py .*--planes 1 --thickness 2 --supersample 1 -o ')" \
      "... and makes a single plane without sub-sampling"
DRY_RUN=1 PS_GPU=0 PS_VOLUME_WEIGHTED=0 PLANES=4 THICKNESS=3 SUPERSAMPLE=2 WINDOW="0 50 0 25" CENTER=40 \
    DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline2.log" 2>&1 || true
check "$(has "${TMP}/g_pipeline2.log" 'PS_VOLUME_WEIGHTED=0 .*run_ps_dtfe.sh -s FAM-1-Dark -g [0-9]+ 99')" \
      "PS_GPU=0 drops -m and PS_VOLUME_WEIGHTED=0 is passed on"
check "$(has "${TMP}/g_pipeline2.log" 'make_image_plane.py .*--axis z --nu [0-9]+ --planes 4 --thickness 3 --supersample 2 --center 40 --u0 0 --u1 50 --v0 0 --v1 25 -o ')" \
      "PLANES, THICKNESS, SUPERSAMPLE, CENTER and WINDOW reach make_image_plane.py"
# a real plane (tiny), then the same geometry is reused and another one regenerates it
"${PY}" python/tools/make_image_plane.py --combined "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/combined_099.hdf5" \
    --mpc-unit 1 --axis z --nu 8 --planes 1 -o "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8" >/dev/null 2>&1 || true
if [ -f "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8.json" ]; then
    DRY_RUN=1 NU=8 DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline3.log" 2>&1 || true
    check "$(has "${TMP}/g_pipeline3.log" '^-- reusing .*hires_plane_z8.bin')" \
          "an existing plane of the requested geometry is reused"
    DRY_RUN=1 NU=8 PLANES=16 DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline4.log" 2>&1 || true
    check "$(has "${TMP}/g_pipeline4.log" 'geometry .* differs; regenerating')" \
          "another plane count regenerates the plane (and so every snapshot)"
    check "$([ -f "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8.bin" ] && echo 0 || echo 1)" \
          "... but a DRY_RUN removes nothing"
    # a comparison that FAILS (here: no python) is not 'differs': the plane stays, the sim is skipped with !!
    rc5=0; PY=/nonexistent/python3 NU=8 DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline5.log" 2>&1 || rc5=$?
    check "$(has "${TMP}/g_pipeline5.log" '^!! FAM-1-Dark: could not compare the plane')" \
          "a failing geometry check is reported, not read as 'differs'"
    check "$([ "${rc5}" != "0" ] && [ -f "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8.json" ] && echo 0 || echo 1)" \
          "... the plane is kept and the pipeline exits non-zero (${rc5})"
else
    echo "   SKIP  plane reuse/regeneration (make_image_plane.py could not write a plane with ${PY})"
fi

echo ""
echo "(H) run_dtfe.sh -e (or DTFE_EXACT_AVERAGE=1) runs the standard binary with --exact-average; a plain run does not."
if [ -x "${DTFE_BIN}" ]; then
    rc=0; scripts/run_dtfe.sh -s HASIC -g 16 -e 99 >"${TMP}/h_exact.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh -e succeeds (exit ${rc})"
    check "$(has "${TMP}/h_exact.log" 'RUNNING: .*--exact-average')" "the binary was called with --exact-average"
    if precision_double; then
        check "$(has "${TMP}/h_exact.log" 'RUNNING: .*DTFE-double ')" "DTFE_PRECISION=double: run_dtfe.sh ran ./DTFE-double"
    fi
    check "$(has "${TMP}/h_exact.log" 'EXACT cell-tetrahedron integration')" "... and reports the exact cell average"
    # 2026-10-06 (survey rank 5): the wrapper run_ps_dtfe.sh has -- wall time, peak memory, a de-ANSI'd runlog
    check "$(has "${TMP}/h_exact.log" '^  Wall time: [0-9]+ min [0-9]+ s$')" "run_dtfe.sh reports the wall time per snapshot"
    if [ -x /usr/bin/time ]; then
        check "$(has "${TMP}/h_exact.log" '^  Peak memory: .*GB resident')" "... and the peak memory (through /usr/bin/time)"
    fi
    CLICOLOR_FORCE=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_color.log" 2>&1 || true
    check "$(lacks "${TMP}/HASIC/snapdir_099/output.runlog" $'\033\\[')" "the runlog is stripped of colour codes when colours are forced"
    check "$(has "${TMP}/HASIC/snapdir_099/output.runlog" 'total wall time')" "... and still carries the binary's own wall-time line for the Runs browser"
    # a binary that fails (a truncated input) must still be a failed snapshot: the status is taken before the strip
    mkdir -p "${TMP}/BADIN/snapdir_099"; head -c 1000 "${TMP}/HASIC/snapdir_099/combined_099.hdf5" > "${TMP}/BADIN/snapdir_099/combined_099.hdf5"
    rc_bad=0; scripts/run_dtfe.sh -s BADIN -g 16 99 >"${TMP}/h_badin.log" 2>&1 || rc_bad=$?
    check "$([ "${rc_bad}" = "1" ] && echo 0 || echo 1)" "a snapshot the binary cannot read is a FAILED snapshot, exit 1 (${rc_bad})"
    check "$(has "${TMP}/h_badin.log" 'processing complete: 0 processed, 0 skipped, 1 failed')" "... counted as failed, not processed (the exit status is read before the log is post-processed)"
    check "$([ -f "${TMP}/HASIC/snapdir_099/output.a_den" ] && echo 0 || echo 1)" "its density grid was written"
    DTFE_EXACT_AVERAGE=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_env.log" 2>&1 || true
    # 2026-10-05: run_dtfe.sh reports failures in its exit status (it always exited 0 before), passes the
    # launcher's scratch folder and the scripts' lambda_th default, and the binaries refuse --redshiftSpace
    rc_none=0; scripts/run_dtfe.sh -s HASIC -g 16 123 >"${TMP}/h_none.log" 2>&1 || rc_none=$?
    check "$([ "${rc_none}" != "0" ] && echo 0 || echo 1)" "run_dtfe.sh exits non-zero when no snapshot was processed (${rc_none})"
    check "$(has "${TMP}/h_none.log" 'processing complete: 0 processed, 1 skipped, 0 failed')" "... and says so in its summary line"
    SCR_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dtfe-scratch-check.XXXXXX")"   # local: the binary refuses iCloud paths (and Linux has no /private/tmp)
    rc_scr=0; SCRATCH_DIR="${SCR_DIR}" scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_scratch.log" 2>&1 || rc_scr=$?
    check "$([ "${rc_scr}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh with SCRATCH_DIR succeeds (exit ${rc_scr})"
    check "$(has "${TMP}/h_scratch.log" "RUNNING: .*--scratch-dir ${SCR_DIR}")" "... and hands the folder to the binary (--scratch-dir)"
    check "$(has "${TMP}/h_scratch.log" 'RUNNING: .*--lambda_th 0.3')" "run_dtfe.sh passes the scripts' lambda_th default (0.3, the binary's own is 0)"
    rm -rf "${SCR_DIR}"
    # FIELDS reaches the standard binary (the launcher's ticks: the T-web), a phase-space dispersion is left out;
    # TESS_CACHE hands a (created) local folder to --tessellation-cache (2026-10-06)
    TC_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dtfe-tess-check.XXXXXX")/cache"
    rc_f=0; FIELDS="density_a tweb_a dispersion_a" TESS_CACHE="${TC_DIR}" scripts/run_dtfe.sh -s HASIC -g 16 99 \
        >"${TMP}/h_fields.log" 2>&1 || rc_f=$?
    check "$([ "${rc_f}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh with FIELDS and TESS_CACHE succeeds (exit ${rc_f})"
    check "$(has "${TMP}/h_fields.log" 'RUNNING: .*--field density_a tweb_a +--')" "... FIELDS is the field list (the T-web for standard DTFE)"
    check "$(has "${TMP}/h_fields.log" 'standard DTFE leaves it out')" "... the phase-space dispersion is left out, and it says so"
    check "$(has "${TMP}/h_fields.log" "RUNNING: .*--tessellation-cache ${TC_DIR}")" "... TESS_CACHE becomes --tessellation-cache"
    check "$([ -d "${TC_DIR}" ] && echo 0 || echo 1)" "... and the cache folder is created"
    check "$([ -f "${TMP}/HASIC/snapdir_099/output.a_tweb" ] || ls "${TMP}/HASIC/snapdir_099/"output.*tweb* >/dev/null 2>&1; echo $?)" \
        "... and the T-web grid is written"
    rm -rf "$(dirname "${TC_DIR}")"
    # SAMPLE_POINTS: the standard DTFE interpolant at a points file too (the launcher's hi-res slice, 2026-10-06)
    printf '1 1 1\n2 3 4\n5 5 5\n' > "${TMP}/h_pts.txt"
    rc_sp=0; SAMPLE_POINTS="${TMP}/h_pts.txt" PTS_VEL_GRAD=1 scripts/run_dtfe.sh -s HASIC -g 16 99 \
        >"${TMP}/h_pts.log" 2>&1 || rc_sp=$?
    check "$([ "${rc_sp}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh with SAMPLE_POINTS succeeds (exit ${rc_sp})"
    check "$(has "${TMP}/h_pts.log" "RUNNING: .*--sample-points ${TMP}/h_pts.txt +--pts-vel-grad")" \
        "... and hands the points to the standard binary (with the velocity gradient)"
    check "$([ -s "${TMP}/HASIC/snapdir_099/output.pts_den" ] && echo 0 || echo 1)" "... which writes the point values (.pts_den)"
    # PS_LINEAR_DEPOSIT reaches the phase-space binary (a dry run: the command only)
    DRY_RUN=1 PS_LINEAR_DEPOSIT=1 PS_VOLUME_WEIGHTED=0 scripts/run_ps_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_lin.log" 2>&1 || true
    check "$(has "${TMP}/h_lin.log" '--ps-linear-deposit')" "run_ps_dtfe.sh: PS_LINEAR_DEPOSIT=1 passes --ps-linear-deposit"
    # zero-padded snapshot numbers are decimal (bash's printf read '099' as octal: an error, then snapshot 000)
    rc_oct=0; scripts/run_dtfe.sh -s HASIC -g 16 099 >"${TMP}/h_oct.log" 2>&1 || rc_oct=$?
    check "$([ "${rc_oct}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh accepts a zero-padded snapshot number (099, exit ${rc_oct})"
    check "$(has "${TMP}/h_oct.log" 'Snapshot 099 processed successfully')" "... and runs snapshot 99, not 0 (octal)"
    rc_bad=0; scripts/run_dtfe.sh -s HASIC -g 16 9x >"${TMP}/h_bad.log" 2>&1 || rc_bad=$?
    check "$([ "${rc_bad}" = "2" ] && echo 0 || echo 1)" "a non-numeric snapshot argument is refused with exit 2 (${rc_bad})"
    rc_oct=0; scripts/run_ps_dtfe.sh -s HASIC -g 16 -n 1 099 >"${TMP}/h_oct_ps.log" 2>&1 || rc_oct=$?
    check "$(has "${TMP}/h_oct_ps.log" 'Processing snapshot 099')" "run_ps_dtfe.sh reads 099 as snapshot 99 too (exit ${rc_oct})"
    rc_rs=0; "${DTFE_BIN}" "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${TMP}/rs_out" --grid 16 --redshiftSpace 0 0 1 --input 105 >"${TMP}/h_rs.log" 2>&1 || rc_rs=$?
    check "$([ "${rc_rs}" != "0" ] && echo 0 || echo 1)" "--redshiftSpace is refused (it was a silent no-op), exit ${rc_rs}"
    check "$(has "${TMP}/h_rs.log" 'not implemented in this version')" "... with a message saying the transform is not implemented"
    check "$(has "${TMP}/h_env.log" 'RUNNING: .*--exact-average')" "DTFE_EXACT_AVERAGE=1 does the same"
    scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_plain.log" 2>&1 || true
    check "$(lacks "${TMP}/h_plain.log" '--exact-average')" "a plain run passes no --exact-average"
else
    echo "   SKIP  (${DTFE_BIN} not built)"
fi

echo ""
echo "(I) download_snapshots.sh + merge_HDF5.py: a half-finished download never becomes a merged snapshot (2026-10-06)."
# TNG-like raw chunks: the header of each promises NumFilesPerSnapshot = 4
DL="${TMP}/DL/SIMX"
mkdir -p "${DL}/snapdir_099"
"${PY}" - "${DL}/snapdir_099" <<'PYEOF'
import sys
from pathlib import Path
import h5py, numpy as np
d, n, chunks, per = Path(sys.argv[1]), 99, 4, 512
rng = np.random.default_rng(99)
for c in range(chunks):
    with h5py.File(d / f"snap_{n:03d}.{c}.hdf5", "w") as f:
        hd = f.create_group("Header").attrs
        hd["BoxSize"], hd["HubbleParam"], hd["Redshift"], hd["Time"] = 75000.0, 0.6774, 0.0, 1.0
        hd["MassTable"] = np.array([0, 0.5, 0, 0, 0, 0], dtype=float)
        hd["NumPart_ThisFile"] = np.array([0, per, 0, 0, 0, 0], dtype=np.int64)
        hd["NumPart_Total"] = np.array([0, per * chunks, 0, 0, 0, 0], dtype=np.uint32)
        hd["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        hd["NumFilesPerSnapshot"] = chunks
        p = f.create_group("PartType1")
        p["Coordinates"] = rng.uniform(0, 75000.0, (per, 3))
        p["ParticleIDs"] = np.arange(c * per, (c + 1) * per, dtype=np.uint64)
        p["Velocities"] = rng.normal(0, 100, (per, 3)).astype(np.float32)
# group catalogue of snapshot 099: 4 chunks (the header says NumFiles = 4), the last with no groups at all
g = d.parent / "groups_099"; g.mkdir(exist_ok=True)
ng = [5, 3, 4, 0]
for c in range(4):
    with h5py.File(g / f"fof_subhalo_tab_099.{c}.hdf5", "w") as f:
        hd = f.create_group("Header").attrs
        hd["BoxSize"], hd["HubbleParam"], hd["Redshift"] = 75000.0, 0.6774, 0.0
        hd["NumFiles"] = np.int32(4)
        hd["Ngroups_Total"], hd["Nsubgroups_Total"] = np.int32(sum(ng)), np.int32(sum(ng))
        hd["Ngroups_ThisFile"], hd["Nsubgroups_ThisFile"] = np.int32(ng[c]), np.int32(ng[c])
        f.create_group("IDs")
        if ng[c]:
            grp = f.create_group("Group"); grp["GroupMass"] = np.full(ng[c], 1.5, dtype=np.float32)
            grp["GroupPos"] = rng.uniform(0, 75000.0, (ng[c], 3)).astype(np.float32)
            sub = f.create_group("Subhalo"); sub["SubhaloMass"] = np.full(ng[c], 0.5, dtype=np.float32)
PYEOF
cp "${DL}/snapdir_099/snap_099.0.hdf5" "${TMP}/snap_099.0.keep"
merge_dl() { "${PY}" python/tools/merge_HDF5.py -d "${DL}" "$@"; }     # the path has spaces: no word splitting
echo "stale" > "${DL}/snapdir_099/combined_099.hdf5.partial"            # a merge stopped half-way last time
rc=0; merge_dl 99 >"${TMP}/i_merge_ok.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "merging 4 complete chunks exits 0 (${rc})"
check "$([ -f "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "... the combined file is written under its final name; a stale .partial does not block it and is gone"
check "$(has "${TMP}/i_merge_ok.log" 'Particles merged: \[2048\]$')" "... and the totals verify against the header"
"${PY}" - "${DL}/snapdir_099/combined_099.hdf5" >"${TMP}/i_content.log" 2>&1 <<'PYEOF'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r") as f:
    a = f["Header"].attrs
    ok = (list(a["NumPart_Total"]) == [0, 2048, 0, 0, 0, 0] and not any(a["NumPart_Total_HighWord"])
          and np.array_equal(f["PartType1/ParticleIDs"][...], np.arange(2048))
          and abs(float(a["BoxSize"]) - 75000.0 / 0.6774) < 1e-6 and "HFreeUnits" in a)
print("CONTENT OK" if ok else f"CONTENT BAD {dict(a)}")
PYEOF
check "$(has "${TMP}/i_content.log" 'CONTENT OK')" "... its header holds int64 totals, a zero high word, the h-free box; the IDs are 0..2047 in chunk order"
rm -f "${DL}/snapdir_099/combined_099.hdf5"
mv "${DL}/snapdir_099/snap_099.2.hdf5" "${TMP}/snap_099.2.keep"
echo "stale" > "${DL}/snapdir_099/combined_099.hdf5.partial"            # left by a stopped merge: a refusal removes it too
rc=0; merge_dl 99 >"${TMP}/i_merge_missing.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a chunk missing from disk (3 of the header's 4): exit 1 (${rc})"
check "$([ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "... no combined file at all (the glob used to merge the 3 and call it done)"
check "$(has "${TMP}/i_merge_missing.log" "Problem: Can't find 1 subfile\(s\) \(the header promises 4\)")" "... the header's count is the yardstick"
check "$(has "${TMP}/i_merge_missing.log" 'Missing: .*snap_099\.2\.hdf5')" "... and the missing chunk is named"
check "$(has "${TMP}/i_merge_missing.log" 'Done, with problems: no combined file for snapshot 099')" "... the summary line says what has none"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL}/snapdir_099/snap_099.2.hdf5"      # a download stopped half-way
echo "stale" > "${DL}/snapdir_099/combined_099.hdf5.partial"
rc=0; merge_dl 99 >"${TMP}/i_merge_trunc.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "a truncated chunk: exit 1 and no combined file (it was merged with an 'Error ... continue' before) (${rc})"
check "$(has "${TMP}/i_merge_trunc.log" "Error: couldn't read snap_099\.2\.hdf5 \(an incomplete download\)")" "... the chunk is named as incomplete"
check "$(has "${TMP}/i_merge_trunc.log" 'Run the download again: it removes an incomplete chunk')" "... with the way out"
head -c 30 "${TMP}/snap_099.2.keep" > "${DL}/snapdir_099/snap_099.2.hdf5"         # stopped inside the signature
rc=0; merge_dl 99 >"${TMP}/i_merge_tiny.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a 30-byte chunk (HDF5 says 'bad byte number' or no signature): exit 1 (${rc})"
check "$(has "${TMP}/i_merge_tiny.log" "Error: couldn't read snap_099\.2\.hdf5 \(an incomplete download\)")" "... classed as incomplete by its size"
rm -f "${DL}/snapdir_099/snap_099.2.hdf5"
cp "${TMP}/snap_099.2.keep" "${DL}/snapdir_099/snap_099.2.hdf5"
rc=0; merge_dl 99 >"${TMP}/i_merge_again.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ -f "${DL}/snapdir_099/combined_099.hdf5" ] && echo 0 || echo 1)" "the chunk restored: the merge succeeds again (${rc})"
# the totals gate: -n 3 merges three of the four chunks -> MISMATCH, no file, --delete-chunks deletes nothing
rm -f "${DL}/snapdir_099/combined_099.hdf5"
rc=0; merge_dl -n 3 --delete-chunks 99 >"${TMP}/i_merge_n3.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "-n 3 against a header of 4: MISMATCH -> exit 1, no combined file, no .partial (${rc})"
check "$(has "${TMP}/i_merge_n3.log" 'MISMATCH vs header total \[2048\]')" "... the MISMATCH line names the header's total"
check "$(has "${TMP}/i_merge_n3.log" 'WARNING: -n 3 given, but the header says 4')" "... and -n disagreeing with the header is said"
check "$([ "$(ls "${DL}"/snapdir_099/snap_099.*.hdf5 | wc -l | tr -d ' ')" = "4" ] && echo 0 || echo 1)" "... --delete-chunks deleted nothing after the refusal"
# the high word of the totals counts (the -1 runs exceed 2^32)
"${PY}" - "${DL}/snapdir_099/snap_099.0.hdf5" <<'PYEOF'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r+") as f:
    f["Header"].attrs["NumPart_Total_HighWord"] = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint32)
PYEOF
rc=0; merge_dl 99 >"${TMP}/i_merge_hw.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && echo 0 || echo 1)" "a high word in the header's total is part of the expected count: MISMATCH (${rc})"
check "$(has "${TMP}/i_merge_hw.log" 'MISMATCH vs header total \[4294969344\]')" "... 2^32 + 2048 named as the total"
cp "${TMP}/snap_099.0.keep" "${DL}/snapdir_099/snap_099.0.hdf5"
# a merge with --delete-chunks, then the same command again: done, not a problem
DLC="${TMP}/DLC/SIMX"; mkdir -p "${DLC}"; cp -R "${DL}/snapdir_099" "${DLC}/snapdir_099"
rc=0; "${PY}" python/tools/merge_HDF5.py -d "${DLC}" --delete-chunks 99 >"${TMP}/i_merge_del.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ -f "${DLC}/snapdir_099/combined_099.hdf5" ] && [ -z "$(ls "${DLC}"/snapdir_099/snap_099.*.hdf5 2>/dev/null)" ] && echo 0 || echo 1)" \
      "--delete-chunks after a verified merge: exit 0, the combined file, no chunk left (${rc})"
rc=0; "${PY}" python/tools/merge_HDF5.py -d "${DLC}" --delete-chunks 99 >"${TMP}/i_merge_del2.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "the same merge again (no chunks, the combined file there): exit 0, nothing to do (${rc})"
check "$(has "${TMP}/i_merge_del2.log" 'already exists and no chunks are left')" "... and says so"
check "$(lacks "${TMP}/i_merge_del2.log" 'Done, with problems')" "... with no problem reported"
# the disk failing while the .partial is written: no file, the error named (ulimit -f caps the file size)
rm -f "${DL}/snapdir_099/combined_099.hdf5"
rc=0; ( ulimit -f 40; PYTHONUNBUFFERED=1 merge_dl 99 ) >"${TMP}/i_merge_disk.log" 2>&1 || rc=$?
check "$([ "${rc}" != "0" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "a write failure (file size limit): non-zero exit, no combined file, no .partial (${rc})"
check "$(has "${TMP}/i_merge_disk.log" 'could not write .*combined_099\.hdf5\.partial')" "... reported as the disk's, not a chunk's"
# group catalogues: the header's NumFiles, an empty chunk without Group/Subhalo groups, a missing chunk
rc=0; merge_dl --groupcats 99 >"${TMP}/i_gc_ok.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ -f "${DL}/groups_099/combined_fof_subhalo_tab_099.hdf5" ] && [ ! -e "${DL}/groups_099/combined_fof_subhalo_tab_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "a group catalogue of 4 chunks, one with no groups at all, merges: exit 0, no .partial (${rc})"
check "$(has "${TMP}/i_gc_ok.log" 'Groups: 12/12  Subhalos: 12/12  \(verified\)')" "... its totals verify"
rm -f "${DL}/groups_099/combined_fof_subhalo_tab_099.hdf5"
mv "${DL}/groups_099/fof_subhalo_tab_099.1.hdf5" "${TMP}/gc1.keep"
rc=0; merge_dl --groupcats 99 >"${TMP}/i_gc_missing.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ ! -e "${DL}/groups_099/combined_fof_subhalo_tab_099.hdf5" ] && echo 0 || echo 1)" "a group-catalogue chunk missing: exit 1, no combined file (${rc})"
check "$(has "${TMP}/i_gc_missing.log" "Problem: Can't find 1 subfile\(s\) \(the header promises 4\)")" "... the header's NumFiles is the yardstick"
check "$(has "${TMP}/i_gc_missing.log" 'Done, with problems: no combined file for group catalogue 099')" "... named in the summary"
mv "${TMP}/gc1.keep" "${DL}/groups_099/fof_subhalo_tab_099.1.hdf5"
# the download script: a fake wget (no network) that fails for snapshot-98; a truncated chunk aged past
# the one-minute guard is removed before wget runs, a fresh one is kept
DL2="${TMP}/DL2/SIMX"
mkdir -p "${DL2}/snapdir_099" "${TMP}/fakebin"
cp "${TMP}/snap_099.2.keep" "${DL2}/snapdir_099/snap_099.0.hdf5"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL2}/snapdir_099/snap_099.1.hdf5"
touch -t 202001010000 "${DL2}/snapdir_099/snap_099.1.hdf5"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL2}/snapdir_099/snap_099.2.hdf5"          # written just now
cat > "${TMP}/fakebin/wget" <<'WEOF'
#!/usr/bin/env bash
# no network: records every argument, fails for snapshot-98, else leaves the product's first chunk
# (a 'fake' byte file) in -P <dir> unless a file of that name exists (as -nc would)
url=""; dest=""; all="$*"
while [ $# -gt 0 ]; do case "$1" in http*) url="$1" ;; -P) dest="$2"; shift ;; esac; shift; done
echo "ARGS: $all" >> "${FAKE_WGET_LOG}"; echo "$url" >> "${FAKE_WGET_LOG}"
case "$url" in *snapshot-98*) echo "fake wget: 404 for $url" >&2; exit 8 ;; esac
name=""
case "$url" in
    */snapshot-*) n=$(echo "$url" | sed -n 's|.*/snapshot-\([0-9]*\)/.*|\1|p'); name=$(printf "snap_%03d.0.hdf5" "$n") ;;
    */groupcat-*) n=$(echo "$url" | sed -n 's|.*/groupcat-\([0-9]*\)/.*|\1|p'); name=$(printf "fof_subhalo_tab_%03d.0.hdf5" "$n") ;;
    */sublink/*) name="tree_extended.0.hdf5" ;;
    */ics.hdf5) name="ics.hdf5" ;;
esac
[ -z "${FAKE_WGET_EMPTY:-}" ] && [ -n "$name" ] && [ -n "$dest" ] && [ ! -e "$dest/$name" ] && echo fake > "$dest/$name"
exit 0
WEOF
chmod +x "${TMP}/fakebin/wget"
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls.log" TNG_API_KEY=dummykey \
    scripts/download_snapshots.sh -d "${DL2}" -s SIMX 98 99 >"${TMP}/i_download.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "download_snapshots.sh exits 1 when a fetch failed (it exited 0 before) (${rc})"
check "$(has "${TMP}/i_download.log" 'Error downloading snapshot-98')" "... the failed endpoint is reported"
check "$(has "${TMP}/i_download.log" 'Download finished with 2 error')" "... and the last line counts it and the kept unreadable chunk (no 'Download complete!')"
check "$(has "${TMP}/i_download.log" 'Error downloading snapshot-99: an unreadable chunk is still in place')" "... a kept unreadable chunk is an error of this run (wget -nc would skip it for ever)"
check "$(has "${TMP}/i_urls.log" '^ARGS: .*--https-only')" "the recursive fetch is https-only (redirects and links with the key header too)"
check "$(has "${TMP}/i_urls.log" '^ARGS: .*--header=API-Key: dummykey')" "... and the key travels as a header"
check "$(lacks "${TMP}/i_download.log" 'Download complete!')" "... really no 'Download complete!'"
check "$([ "$(grep -c '^https://www.tng-project.org/api/SIMX/files/snapshot-9[89]/' "${TMP}/i_urls.log")" = "2" ] && echo 0 || echo 1)" \
      "both fetches went to https (the key is a header; it travelled over plain http before)"
check "$(lacks "${TMP}/i_download.log" 'dummykey')" "the key is not in the output"
check "$([ ! -e "${DL2}/snapdir_099/snap_099.1.hdf5" ] && echo 0 || echo 1)" "an old truncated chunk is removed before wget runs (wget -nc would skip it for ever)"
check "$(has "${TMP}/i_download.log" 'Removed incomplete chunk snap_099\.1\.hdf5')" "... and the log says which"
check "$([ -f "${DL2}/snapdir_099/snap_099.0.hdf5" ] && echo 0 || echo 1)" "a complete chunk stays"
check "$([ -f "${DL2}/snapdir_099/snap_099.2.hdf5" ] && echo 0 || echo 1)" "a truncated chunk written less than a minute ago stays (another download may be writing it)"
check "$(has "${TMP}/i_download.log" 'snap_099\.2\.hdf5 is incomplete but was written less than a minute ago')" "... and is noted"
touch -t 202001010000 "${DL2}/snapdir_099/snap_099.2.hdf5"                                  # a minute later...
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls2.log" TNG_API_KEY=dummykey \
    scripts/download_snapshots.sh -d "${DL2}" -s SIMX 99 >"${TMP}/i_download_ok.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "a run with no failure exits 0 (${rc})"
check "$(has "${TMP}/i_download_ok.log" 'Removed incomplete chunk snap_099\.2\.hdf5')" "... the chunk that aged past the guard is removed now"
check "$(has "${TMP}/i_download_ok.log" 'Download complete!')" "... and ends with 'Download complete!'"
# the other three products, with an aged truncated file each: pruned, fetched over https with the key as a header
DL3="${TMP}/DL3/SIMX"; mkdir -p "${DL3}/Merger Trees" "${DL3}/groups_099"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL3}/ics.hdf5";                      touch -t 202001010000 "${DL3}/ics.hdf5"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL3}/Merger Trees/tree_extended.0.hdf5"; touch -t 202001010000 "${DL3}/Merger Trees/tree_extended.0.hdf5"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL3}/groups_099/fof_subhalo_tab_099.0.hdf5"; touch -t 202001010000 "${DL3}/groups_099/fof_subhalo_tab_099.0.hdf5"
rc_i=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls_i.log" TNG_API_KEY=dummykey scripts/download_snapshots.sh -i -d "${DL3}" -s SIMX-Dark >"${TMP}/i_dl_ics.log" 2>&1 || rc_i=$?
rc_t=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls_t.log" TNG_API_KEY=dummykey scripts/download_snapshots.sh -t -d "${DL3}" -s SIMX >"${TMP}/i_dl_trees.log" 2>&1 || rc_t=$?
rc_c=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls_c.log" TNG_API_KEY=dummykey scripts/download_snapshots.sh -c -d "${DL3}" -s SIMX 99 >"${TMP}/i_dl_gc.log" 2>&1 || rc_c=$?
check "$([ "${rc_i}" = "0" ] && [ "${rc_t}" = "0" ] && [ "${rc_c}" = "0" ] && echo 0 || echo 1)" "-i, -t and -c runs with a working fetch exit 0 (${rc_i} ${rc_t} ${rc_c})"
check "$(has "${TMP}/i_urls_i.log" '^https://www.tng-project.org/api/SIMX/files/ics.hdf5$')" "-i: https, and '-Dark' stripped for the API name"
check "$(has "${TMP}/i_urls_t.log" '^https://www.tng-project.org/api/SIMX/files/sublink/')" "-t: https"
check "$(has "${TMP}/i_urls_c.log" '^https://www.tng-project.org/api/SIMX/files/groupcat-99/')" "-c: https"
check "$([ "$(cat "${TMP}"/i_urls_i.log "${TMP}"/i_urls_t.log "${TMP}"/i_urls_c.log | grep -c 'header=API-Key: dummykey')" = "3" ] && echo 0 || echo 1)" "the key is a header in all three"
check "$(lacks "${TMP}/i_urls_i.log" '^https.*dummykey')" "... never in a URL"
check "$(has "${TMP}/i_dl_ics.log" 'Removed incomplete chunk ics\.hdf5')" "-i prunes an aged truncated ics.hdf5 first"
check "$(has "${TMP}/i_dl_trees.log" 'Removed incomplete chunk tree_extended\.0\.hdf5')" "-t prunes an aged truncated tree chunk first"
check "$(has "${TMP}/i_dl_gc.log" 'Removed incomplete chunk fof_subhalo_tab_099\.0\.hdf5')" "-c prunes an aged truncated catalogue chunk first"
# a recursive fetch that exits 0 but fetched nothing is an error (the listing offered nothing https)
DL4="${TMP}/DL4/SIMX"; mkdir -p "${DL4}"
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls4.log" FAKE_WGET_EMPTY=1 TNG_API_KEY=dummykey \
    scripts/download_snapshots.sh -d "${DL4}" -s SIMX 99 >"${TMP}/i_dl_empty.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "wget exiting 0 with no chunk fetched: exit 1 (${rc})"
check "$(has "${TMP}/i_dl_empty.log" 'Error downloading snapshot-99: no chunk was fetched')" "... and named"
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls3.log" TNG_API_KEY=dummykey PY=/nonexistent/python3 \
    scripts/download_snapshots.sh -d "${DL2}" -s SIMX 99 >"${TMP}/i_download_nopy.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "without a python with h5py the check is skipped, the download still runs (${rc})"
check "$(has "${TMP}/i_download_nopy.log" 'has no h5py; existing chunks are not checked')" "... and says so"

echo ""
echo "(J) --help runs nothing, --version prints the revision, DRY_RUN=1 prints the commands and runs nothing (2026-10-05, survey rank 19)."
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/run_ps_pipeline.sh; do
    rc=0; "${scr}" --help >"${TMP}/j_help_${scr##*/}.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "${scr##*/} --help exits 0 (${rc})"
    check "$(lacks "${TMP}/j_help_${scr##*/}.log" 'Processing snapshot|RUNNING:|Slice: axis')" "... and starts nothing"
done
check "$(has "${TMP}/j_help_run_ps_pipeline.sh.log" 'COMPUTE ONLY')" "run_ps_pipeline.sh --help prints its header (it used to start the whole batch)"
check "$(has "${TMP}/j_help_run_dtfe.sh.log" '^Usage: ')" "run_dtfe.sh --help ends with the usage line"
rc=0; scripts/run_ps_pipeline.sh bogus >"${TMP}/j_bogus.log" 2>&1 || rc=$?
check "$([ "${rc}" = "2" ] && echo 0 || echo 1)" "run_ps_pipeline.sh with an argument refuses with exit 2 (${rc})"
check "$(has "${TMP}/j_bogus.log" 'takes no arguments')" "... and says why"
rc=0; scripts/run_ps_dtfe.sh --version >"${TMP}/j_version.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ -s "${TMP}/j_version.log" ] && echo 0 || echo 1)" "run_ps_dtfe.sh --version exits 0 with a revision"
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" -s HASIC -g 16 99 98 >"${TMP}/j_dry.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "DRY_RUN=1 run_ps_dtfe.sh exits 0 with a planned snapshot (${rc})"
check "$(has "${TMP}/j_dry.log" 'DRY RUN -- would run:')" "... prints the expanded command"
check "$(has "${TMP}/j_dry.log" 'PS-DTFE.*combined_099\.hdf5.*--grid +16')" "... with the binary, the input and the grid"
check "$(has "${TMP}/j_dry.log" 'DRY RUN: 1 snapshot\(s\) planned, 1 skipped')" "... one planned, the missing 098 skipped, nothing run"
check "$(lacks "${TMP}/j_dry.log" 'RUNNING:')" "... the binary never ran"
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" -s NOSUCH -g 16 99 >"${TMP}/j_dry2.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "DRY_RUN=1 with nothing to run exits 1 (${rc})"
if [ -x "${DTFE_BIN}" ]; then
    touch "${TMP}/j_marker"; sleep 1
    rc=0; DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/j_dry3.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "DRY_RUN=1 run_dtfe.sh exits 0 (${rc})"
    check "$(has "${TMP}/j_dry3.log" 'DTFE.*combined_099\.hdf5.*--grid +16')" "... prints the standard binary's command"
    check "$([ ! "${TMP}/HASIC/snapdir_099/output.runlog" -nt "${TMP}/j_marker" ] && echo 0 || echo 1)" "... and leaves the run log untouched"
fi

echo ""
echo "(K) the binaries' --help lists the phase-space, GPU, point-evaluation and server options; no known misspelling (2026-10-05, survey rank 21)."
"${PS_BIN}" --help >"${TMP}/k_help_ps.log" 2>&1 || true
"${DTFE_BIN}" --help >"${TMP}/k_help_dtfe.log" 2>&1 || true
for o in ps-gpu ps-exact-deposit ps-window ps-caustics ps-vertex-mass ps-volume-weighted sample-points serve serve-progress tessellation-cache auto-tune-report; do
    check "$(has "${TMP}/k_help_ps.log" "^ *--${o}( |$)")" "PS-DTFE --help lists --${o}"
done
for o in gpu exact-average sample-points serve auto-tune-report; do
    check "$(has "${TMP}/k_help_dtfe.log" "^ *--${o}( |$)")" "DTFE --help lists --${o}"
done
check "$(lacks "${TMP}/k_help_dtfe.log" '^ *--ps-gpu')" "... and not PS-DTFE's own --ps-gpu"
"${PS_BIN}" --full_help >"${TMP}/k_full_ps.log" 2>&1 || true
check "$(lacks "${TMP}/k_full_ps.log" 'encompasing|documenation|vortivity|chuncks|coordiante|tesselation|specifed|suplied|strech|mannually|DMPC_UNIT')" \
      "the full help carries none of the misspellings the survey listed"
check "$(has "${TMP}/k_full_ps.log" "'MPC_UNIT'")" "... and names the Makefile's MPC_UNIT, not DMPC_UNIT"

echo ""
echo "(L) scripts/install_lib.sh: the binaries kept through a failed rebuild (2026-10-05, survey rank 24)."
LREPO="${TMP}/fake_repo"; rm -rf "${LREPO}"; mkdir -p "${LREPO}"
( cd "${LREPO}" && printf 'old DTFE\n' > DTFE && printf 'old PS-DTFE\n' > PS-DTFE && printf 'old 2d\n' > DTFE-2d && chmod +x DTFE PS-DTFE DTFE-2d
  source "${ROOT_DIR}/scripts/install_lib.sh"
  n="$(backup_binaries "${LREPO}/.backup")"
  echo "backed up ${n}" > "${TMP}/l_backup.log"
  rm -f DTFE PS-DTFE DTFE-2d; printf 'half-linked\n' > PS-DTFE                 # 'make clean' then a failing build
  m="$(restore_binaries "${LREPO}/.backup")"
  echo "restored ${m}" >> "${TMP}/l_backup.log"
  printf 'line1\nline2\nld: symbol not found\n' > build.log
  build_failed_hint build.log 2 > "${TMP}/l_hint.log" )
check "$(has "${TMP}/l_backup.log" '^backed up 3$')" "backup_binaries counts the three binaries present (of the six names)"
check "$(has "${TMP}/l_backup.log" '^restored 3$')" "restore_binaries puts the three back"
check "$([ "$(cat "${LREPO}/PS-DTFE")" = "old PS-DTFE" ] && [ "$(cat "${LREPO}/DTFE")" = "old DTFE" ] && [ -x "${LREPO}/DTFE-2d" ] && echo 0 || echo 1)" \
      "... byte for byte, the half-linked one replaced, the mode kept"
check "$(has "${TMP}/l_hint.log" 'ld: symbol not found')" "build_failed_hint prints the log's tail"
check "$(lacks "${TMP}/l_hint.log" '^line1$')" "... only the last N lines"
check "$(has "${TMP}/l_hint.log" 'make deps-check')" "... and the usual causes"

echo ""
echo "------------------------------------------------------------"
if [ "${fails}" -eq 0 ]; then
    rm -rf "${TMP}"
    echo "RESULT: PASS  (the batch driver honours its environment, needs no IC file when the"
    echo "               snapshot carries one, reports honest exit codes, is portable, and"
    echo "               handles the per-family data layout; run_dtfe.sh's exact cell average;"
    echo "               downloads merge only when complete; --help and DRY_RUN run nothing)"
    exit 0
fi
echo "RESULT: FAIL  (${fails} check(s); logs kept under ${TMP})"
exit 1
