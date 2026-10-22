#!/usr/bin/env bash
# Contract check for the batch driver scripts/run_ps_dtfe.sh (and the shared scripts/config.sh, and
# run_dtfe.sh's exact-average switch).
#
# Every other suite tests the BINARY; this one tests the shell around it, where failures stay silent
# (a discarded environment variable, or snapshots skipped for a file they do not need). Runs against a
# FAKE DATA_ROOT built here, so it never touches real data.
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
#  F) No macOS-only command on the critical path ('/usr/bin/time -l', 'sed -i ""'), which would
#     kill every snapshot on Linux. Asserted structurally, so it holds on either platform.
#  G) The PER-FAMILY data layout (<root>/<family>/<sim>, e.g. <root>/TNG100/TNG100-3-Dark), under a
#     root whose name has a space: the driver finds and processes the simulation there, config.sh's sim_dir puts a new simulation beside its
#     family and lets a flat <root>/<sim> win, and run_ps_pipeline.sh discovers nested sims -- and exits 1
#     when it finds nothing to compute (no data root, a mistyped SIMS, a SNAPS matching nothing).
#  H) scripts/run_dtfe.sh passes --exact-average to the standard binary by default (-e and
#     DTFE_EXACT_AVERAGE=1 too, DTFE_EXACT_AVERAGE=0 not); its padding default is 15 (a caller's PADDING /
#     DTFE_PADDING wins), run_ps_dtfe.sh keeps config.sh's 50. run_dtfe.sh AUTO_TUNE_REPORT=1 (as
#     run_ps_dtfe.sh): the standard binary's own report line per snapshot ('snap=099'), exit 0, nothing written.
#  I) download_snapshots.sh + merge_HDF5.py never turn a half-finished download into a merged snapshot.
#  J) --help runs nothing (run_ps_pipeline.sh --help must not start the whole batch), --version prints the
#     revision, an argument to the pipeline is an error, DRY_RUN=1 prints the drivers' commands and runs nothing.
#     There is no default data root or simulation: every driver without DTFE_DATA_ROOT (or without -s / DTFE_SIM /
#     SIMS) refuses with exit 2 and says so, -d or not, while --help still works and sourcing config.sh exits nothing.
#  K) the binaries' own --help lists the phase-space / GPU / point-evaluation / server options the scripts and the
#     launcher use, and the help texts carry none of the known misspellings.
#  L) scripts/install_lib.sh keeps a user's binaries through a failed rebuild: backup, restore (the 2D double pair
#     too), the failure hint; install.sh's detection of the extra sets it rebuilds (sourced alone, never run).
#  M) the run-log contract of run_dtfe.sh / run_ps_dtfe.sh, against stand-in binaries: a run's log is
#     <prefix>.runlog.partial while it goes, <prefix>.runlog after exit 0, <prefix>.runlog.failed after a
#     failure or a stop; a failed, stopped or killed re-run never touches the .runlog of the grids on disk.
#  N) the Makefile stamps BUILD_MODE, ARCH_FLAGS and EXTRA_FLAGS in the object dirs (a dry run, builds nothing).
#  O) the drivers' single-instance lock (<prefix>.runlog.lock: a second driver refused, stale locks removed, a stop
#     releases it) and their preflight (data disk mounted, output folder writable with room for the grids).
#  P) run_ps_pipeline.sh's freshness test compares the run settings with the snapshot's run log, not only file times
#     (the plane's conventions too, PTS_CONVENTIONS, which it hands to run_ps_dtfe.sh).
#  R) run_ps_dtfe.sh ORDERED_MERGE=1 (the default) passes --ordered-merge, ORDERED_MERGE=0
#     --completion-order-merge, FIXED_POINT_MERGE=1 --fixed-point-merge instead of either;
#     PTS_CONVENTIONS=legacy with SAMPLE_POINTS passes '--pts-conventions legacy', grid (the default) and a run without
#     points pass nothing, another value is refused; a killed run's checkpoint (<prefix>.runlog.checkpoint,
#     or config.sh ps_checkpoint_dir's name under SCRATCH_DIR) makes the next run pass --resume and say so; PS_RESUME=0,
#     an auto-tune report and a checkpoint folder with no 'checkpoint' file in it do not.
#  S) the production defaults and the rerun hygiene, against the stand-in binaries:
#     the GPU when the Makefile's stamp says the binary has one (metal/cuda/hip; not off or cudaemu; DTFE_METAL /
#     PS_METAL=0 wins), the drivers' production flags by default (run_ps_dtfe.sh: volume weighting, caustics, the nine
#     fields; run_dtfe.sh: the exact average and the web fields; PS_LINEAR_DEPOSIT=1 drops the volume weighting),
#     run_dtfe.sh's OUTPUT_PREFIX (a 'ps_*' one refused), and a rerun's left-over files: after exit 0 the files the run
#     did not write are MOVED to <prefix>.superseded/<stamp>/ with the old run log (the rewritten ones, the run log
#     family, a lock and a checkpoint stay); a failed run, a dry run and an auto-tune report move nothing.
#     run_ps_pipeline.sh's PS_GPU unset follows the PS binary's stamp the same way.
#
# Usage: tests/run_scripts_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the drivers run the double pair
precision_double && export DTFE_PRECISION=double     # the drivers' own switch (run_ps_dtfe.sh / run_dtfe.sh)

export PY                               # (tests/lib.sh) run_ps_pipeline.sh runs make_image_plane.py with the same python

N="${N:-16}"
GRID="${GRID:-24}"
BOX="${BOX:-100.0}"
TMP="${TMP_DIR:-${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}}/run_scripts"
export DTFE_FIGURES_ROOT="${TMP}/figures"   # the pipeline's plots: never the caller's figure root
DRIVER="scripts/run_ps_dtfe.sh"

echo "============================================================"
echo " run_ps_dtfe.sh contract check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
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
check "$([ "${default_grid}" = "512" ] && echo 0 || echo 1)" \
      "config.sh's default is the production 512 when nothing overrides it (got ${default_grid})"

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
# would run here -- slow on a 64 GB Mac, and killed for memory in a 16 GB Linux container
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
rc=0; DRY_RUN=1 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline.log" 2>&1 || rc=$?
check "$(has "${TMP}/g_pipeline.log" '^Simulations:.* FAM-1-Dark')" \
      "run_ps_pipeline.sh discovers the nested FAM-1-Dark"
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... and a dry run with a snapshot to plan exits 0 (${rc})"
# the pipeline's knobs: the GPU flag follows PS_GPU, the plane's geometry reaches
# make_image_plane.py, and a plane whose sidecar has another geometry is regenerated. PS_GPU unset: the
# GPU exactly when this repository's PS binary is a GPU build (config.sh gpu_build_default; (S) varies the stamp)
g_bin="$([ "${DTFE_PRECISION:-single}" = double ] && echo PS-DTFE-double || echo PS-DTFE)"
g_gpu="$(bash -c 'source scripts/config.sh >/dev/null 2>&1; gpu_build_default "$1" "$2"' _ "${ROOT_DIR}" "${g_bin}")"
if [ "${g_gpu}" = "1" ]; then
    check "$(has "${TMP}/g_pipeline.log" 'run_ps_dtfe.sh -s FAM-1-Dark -g [0-9]+ -m 99')" \
          "the pipeline's default run_ps_dtfe.sh call carries -m (${g_bin} is a GPU build here)"
else
    check "$(has "${TMP}/g_pipeline.log" 'run_ps_dtfe.sh -s FAM-1-Dark -g [0-9]+ 99')" \
          "the pipeline's default run_ps_dtfe.sh call has no -m (${g_bin} is a CPU-only build here)"
fi
check "$(has "${TMP}/g_pipeline.log" 'make_image_plane.py .*--planes 1 --thickness 2 --supersample 1 -o ')" \
      "... and makes a single plane without sub-sampling"
DRY_RUN=1 PS_GPU=0 PS_VOLUME_WEIGHTED=0 PLANES=4 THICKNESS=3 SUPERSAMPLE=2 WINDOW="0 50 0 25" CENTER=40 \
    SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline2.log" 2>&1 || true
check "$(has "${TMP}/g_pipeline2.log" 'PS_VOLUME_WEIGHTED=0 .*run_ps_dtfe.sh -s FAM-1-Dark -g [0-9]+ 99')" \
      "PS_GPU=0 drops -m and PS_VOLUME_WEIGHTED=0 is passed on"
check "$(has "${TMP}/g_pipeline2.log" 'make_image_plane.py .*--axis z --nu [0-9]+ --planes 4 --thickness 3 --supersample 2 --center 40 --u0 0 --u1 50 --v0 0 --v1 25 -o ')" \
      "PLANES, THICKNESS, SUPERSAMPLE, CENTER and WINDOW reach make_image_plane.py"
# finding nothing to compute is a failure: with the data drive unmounted, a mistyped SIMS or a SNAPS
# matching no snapshot, an ALL DONE with exit 0 would show in the launcher as a success
mkdir -p "${TMP}/empty root"
rc=0; env -u SIMS DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_nosims.log" 2>&1 || rc=$?
check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/g_pipe_nosims.log" 'no simulation given: set SIMS')" = 0 ] && echo 0 || echo 1)" \
      "run_ps_pipeline.sh without SIMS refuses with exit 2: there is no default simulation (SIMS=all discovers) (${rc})"
rc=0; SIMS=all DTFE_DATA_ROOT="${TMP}/empty root" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_none.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "run_ps_pipeline.sh with no simulation under the data root exits 1 (${rc})"
check "$(has "${TMP}/g_pipe_none.log" '^!! nothing to compute')" "... says there is nothing to compute"
check "$(lacks "${TMP}/g_pipe_none.log" 'ALL DONE')" "... and not ALL DONE"
rc=0; SIMS=all DTFE_DATA_ROOT="${TMP}/no such root" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_unmounted.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a data root that does not exist (the drive unplugged) exits 1 (${rc})"
check "$(has "${TMP}/g_pipe_unmounted.log" 'is the data drive mounted')" "... and asks whether the drive is mounted"
rc=0; DRY_RUN=1 SIMS=FAM-1-Drak DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_typo.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a mistyped SIMS exits 1 (${rc})"
check "$(has "${TMP}/g_pipe_typo.log" '^!! nothing to compute: no combined snapshot of FAM-1-Drak ')" "... naming the simulation"
rc=0; DRY_RUN=1 SNAPS="50 67" SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_snaps.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a SNAPS matching no snapshot exits 1 (${rc})"
check "$(has "${TMP}/g_pipe_snaps.log" 'FAM-1-Dark: no combined snapshots numbered 50 67, skipping')" "... naming the numbers it looked for"
rc=0; DRY_RUN=1 SIMS="FAM-1-Drak FAM-1-Dark" DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_half.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "one simulation of two without snapshots is a partial skip: exit 0 (${rc})"
# a real plane (tiny), then the same geometry is reused and another one regenerates it
"${PY}" python/tools/make_image_plane.py --combined "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/combined_099.hdf5" \
    --mpc-unit 1 --axis z --nu 8 --planes 1 -o "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8" >/dev/null 2>&1 || true
if [ -f "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8.json" ]; then
    DRY_RUN=1 NU=8 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline3.log" 2>&1 || true
    check "$(has "${TMP}/g_pipeline3.log" '^-- reusing .*hires_plane_z8.bin')" \
          "an existing plane of the requested geometry is reused"
    DRY_RUN=1 NU=8 PLANES=16 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline4.log" 2>&1 || true
    check "$(has "${TMP}/g_pipeline4.log" 'geometry .* differs; regenerating')" \
          "another plane count regenerates the plane (and so every snapshot)"
    check "$([ -f "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8.bin" ] && echo 0 || echo 1)" \
          "... but a DRY_RUN removes nothing"
    # a comparison that FAILS (here: no python) is not 'differs': the plane stays, the sim is skipped with !!
    rc5=0; PY=/nonexistent/python3 NU=8 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline5.log" 2>&1 || rc5=$?
    check "$(has "${TMP}/g_pipeline5.log" '^!! FAM-1-Dark: could not compare the plane')" \
          "a failing geometry check is reported, not read as 'differs'"
    check "$([ "${rc5}" != "0" ] && [ -f "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8.json" ] && echo 0 || echo 1)" \
          "... the plane is kept and the pipeline exits non-zero (${rc5})"
    # every snapshot up to date is a success, not 'nothing to compute'; a slice of the 8^2 plane holds
    # 64 points (the pipeline compares the count)
    head -c 512 /dev/zero > "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_den"
    touch -t 203001010000 "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_den" \
                          "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_velGrad"
    # the settings of the (A)-style run that made these grids: the pipeline compares them too (P)
    rc6=0; DRY_RUN=1 NU=8 GRID_SIZE=24 FIELDS=density_a AVG_SUBSAMPLES=1 PS_VOLUME_WEIGHTED=1 PTS_VEL_GRAD=0 \
        SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline6.log" 2>&1 || rc6=$?
    check "$([ "${rc6}" = "0" ] && echo 0 || echo 1)" "every snapshot up to date still exits 0 (${rc6})"
    check "$(has "${TMP}/g_pipeline6.log" 'FAM-1-Dark: all snapshots up to date')" "... with nothing to recompute"
    rm -f "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_den" "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_velGrad"
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
    # what the wrapper run_ps_dtfe.sh has -- wall time, peak memory, a de-ANSI'd runlog
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
    check "$([ -s "${TMP}/BADIN/snapdir_099/output.runlog.failed" ] && [ ! -e "${TMP}/BADIN/snapdir_099/output.runlog" ] \
             && [ ! -e "${TMP}/BADIN/snapdir_099/output.runlog.partial" ] && echo 0 || echo 1)" \
          "... its log is output.runlog.failed: no output.runlog claims grids it never wrote"
    check "$([ -f "${TMP}/HASIC/snapdir_099/output.a_den" ] && echo 0 || echo 1)" "its density grid was written"
    DTFE_EXACT_AVERAGE=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_env.log" 2>&1 || true
    # run_dtfe.sh reports failures in its exit status, passes the
    # launcher's scratch folder and the scripts' lambda_th default, and the binaries refuse --redshiftSpace
    rc_none=0; scripts/run_dtfe.sh -s HASIC -g 16 123 >"${TMP}/h_none.log" 2>&1 || rc_none=$?
    check "$([ "${rc_none}" != "0" ] && echo 0 || echo 1)" "run_dtfe.sh exits non-zero when no snapshot was processed (${rc_none})"
    check "$(has "${TMP}/h_none.log" 'processing complete: 0 processed, 1 skipped, 0 failed')" "... and says so in its summary line"
    SCR_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dtfe-scratch-check.XXXXXX")"   # local: the binary refuses iCloud paths (and Linux has no /private/tmp)
    rc_scr=0; SCRATCH_DIR="${SCR_DIR}" scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_scratch.log" 2>&1 || rc_scr=$?
    check "$([ "${rc_scr}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh with SCRATCH_DIR succeeds (exit ${rc_scr})"
    check "$(has "${TMP}/h_scratch.log" "RUNNING: .*--scratch-dir ${SCR_DIR}")" "... and hands the folder to the binary (--scratch-dir)"
    check "$(has "${TMP}/h_scratch.log" 'RUNNING: .*--lambda_th 0.3')" "run_dtfe.sh passes the scripts' lambda_th default (0.3, the binary's own is 0)"
    # standard DTFE pads with 15 spacings by default (complete at z = 0 on TNG50/100/300), not config.sh's
    # 50, which stays the PS-DTFE scripts' value; a caller's PADDING or DTFE_PADDING still wins, PADDING first
    pad_of() { env -u PADDING -u DTFE_PADDING "$@" DRY_RUN=1 "${PAD_DRIVER}" -s HASIC -g 16 99 2>&1 \
                   | grep -oE -e '--padding +[0-9]+' | head -1 | grep -oE '[0-9]+$'; }
    PAD_DRIVER=scripts/run_dtfe.sh
    pad_def="$(pad_of)"; pad_p="$(pad_of PADDING=12)"; pad_d="$(pad_of DTFE_PADDING=13)"; pad_pd="$(pad_of PADDING=11 DTFE_PADDING=13)"
    check "$([ "${pad_def}" = "15" ] && echo 0 || echo 1)" "run_dtfe.sh pads with 15 spacings by default (${pad_def:-none})"
    check "$([ "${pad_p}" = "12" ] && [ "${pad_d}" = "13" ] && [ "${pad_pd}" = "11" ] && echo 0 || echo 1)" \
          "... a caller's PADDING or DTFE_PADDING wins, PADDING first (${pad_p:-none} ${pad_d:-none} ${pad_pd:-none})"
    PAD_DRIVER=scripts/run_ps_dtfe.sh; pad_ps="$(pad_of)"
    check "$([ "${pad_ps}" = "50" ] && echo 0 || echo 1)" "... while run_ps_dtfe.sh keeps config.sh's 50 (${pad_ps:-none})"
    rm -rf "${SCR_DIR}"
    # FIELDS reaches the standard binary (the launcher's ticks: the T-web), a phase-space dispersion is left out;
    # TESS_CACHE hands a (created) local folder to --tessellation-cache
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
    # SAMPLE_POINTS: the standard DTFE interpolant at a points file too (the launcher's hi-res slice)
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
    # zero-padded snapshot numbers are decimal (bash's printf reads '099' as octal: an error, then snapshot 000)
    rc_oct=0; scripts/run_dtfe.sh -s HASIC -g 16 099 >"${TMP}/h_oct.log" 2>&1 || rc_oct=$?
    check "$([ "${rc_oct}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh accepts a zero-padded snapshot number (099, exit ${rc_oct})"
    check "$(has "${TMP}/h_oct.log" 'Snapshot 099 processed successfully')" "... and runs snapshot 99, not 0 (octal)"
    rc_bad=0; scripts/run_dtfe.sh -s HASIC -g 16 9x >"${TMP}/h_bad.log" 2>&1 || rc_bad=$?
    check "$([ "${rc_bad}" = "2" ] && echo 0 || echo 1)" "a non-numeric snapshot argument is refused with exit 2 (${rc_bad})"
    rc_oct=0; scripts/run_ps_dtfe.sh -s HASIC -g 16 -n 1 099 >"${TMP}/h_oct_ps.log" 2>&1 || rc_oct=$?
    check "$(has "${TMP}/h_oct_ps.log" 'Processing snapshot 099')" "run_ps_dtfe.sh reads 099 as snapshot 99 too (exit ${rc_oct})"
    rc_rs=0; "${DTFE_BIN}" "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${TMP}/rs_out" --grid 16 --redshiftSpace 0 0 1 --input 105 >"${TMP}/h_rs.log" 2>&1 || rc_rs=$?
    check "$([ "${rc_rs}" != "0" ] && echo 0 || echo 1)" "--redshiftSpace is refused (not a silent no-op), exit ${rc_rs}"
    check "$(has "${TMP}/h_rs.log" 'not implemented in this version')" "... with a message saying the transform is not implemented"
    check "$(has "${TMP}/h_env.log" 'RUNNING: .*--exact-average')" "DTFE_EXACT_AVERAGE=1 does the same"
    # the exact average is the default (the production standard); DTFE_EXACT_AVERAGE=0 is the sampled one
    rc_pl=0; env -u DTFE_EXACT_AVERAGE scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_plain.log" 2>&1 || rc_pl=$?
    check "$([ "${rc_pl}" = "0" ] && [ "$(has "${TMP}/h_plain.log" 'RUNNING: .*--exact-average')" = 0 ] \
             && [ "$(has "${TMP}/h_plain.log" 'RUNNING: .*--field density_a velocity_a gradient_a divergence_a shear_a vorticity_a tweb_a vweb_a ')" = 0 ] \
             && echo 0 || echo 1)" "a plain run passes --exact-average and the web fields (the production defaults; exit ${rc_pl})"
    DTFE_EXACT_AVERAGE=0 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_mc.log" 2>&1 || true
    check "$(lacks "${TMP}/h_mc.log" '--exact-average')" "DTFE_EXACT_AVERAGE=0 passes no --exact-average (the Monte-Carlo cell mean)"
    # the standard binary's auto-tuner reports (not 'AUTO-TUNE-REPORT unsupported'), and the driver
    # asks it the way run_ps_dtfe.sh does: a report per snapshot, nothing computed, nothing written beside the data
    rm -rf "${TMP}/HASIC/snapdir_099/output."*        # (the grids, run logs and the superseded folder of the runs above)
    rc_rep=0; AUTO_TUNE_REPORT=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_report.log" 2>&1 || rc_rep=$?
    check "$([ "${rc_rep}" = "0" ] && [ "$(grep -acE '^  AUTO-TUNE-REPORT snap=099 partition=[0-9]+ mc=[0-9]+ predicted_gb=[0-9.]+ budget_gb=[0-9.]+ .*over_budget=[01]$' "${TMP}/h_report.log")" = 1 ] \
             && echo 0 || echo 1)" "run_dtfe.sh AUTO_TUNE_REPORT=1: one valid standard report line for snapshot 099 (exit ${rc_rep})"
    check "$([ -z "$(ls "${TMP}/HASIC/snapdir_099/" | grep '^output\.')" ] && [ "$(lacks "${TMP}/h_report.log" 'unsupported')" = 0 ] \
             && echo 0 || echo 1)" "... and nothing is written beside the data: no grids, no run log, no lock"
else
    echo "   SKIP  (${DTFE_BIN} not built)"
fi

echo ""
echo "(I) download_snapshots.sh + merge_HDF5.py: a half-finished download never becomes a merged snapshot."
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
      "... no combined file at all (not the 3 merged and called done)"
check "$(has "${TMP}/i_merge_missing.log" "Problem: Can't find 1 subfile\(s\) \(the header promises 4\)")" "... the header's count is the yardstick"
check "$(has "${TMP}/i_merge_missing.log" 'Missing: .*snap_099\.2\.hdf5')" "... and the missing chunk is named"
check "$(has "${TMP}/i_merge_missing.log" 'Done, with problems: no combined file for snapshot 099')" "... the summary line says what has none"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL}/snapdir_099/snap_099.2.hdf5"      # a download stopped half-way
echo "stale" > "${DL}/snapdir_099/combined_099.hdf5.partial"
rc=0; merge_dl 99 >"${TMP}/i_merge_trunc.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "a truncated chunk: exit 1 and no combined file (not merged past an 'Error ... continue') (${rc})"
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
touch -t 209901010000 "${DL2}/snapdir_099/snap_099.2.hdf5"   # ... and stays "just now" however loaded the machine (the guard keeps a chunk younger than 60 s; a slow run would age a real 'now' past that)
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
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "download_snapshots.sh exits 1 when a fetch failed, not 0 (${rc})"
check "$(has "${TMP}/i_download.log" 'Error downloading snapshot-98')" "... the failed endpoint is reported"
check "$(has "${TMP}/i_download.log" 'Download finished with 2 error')" "... and the last line counts it and the kept unreadable chunk (no 'Download complete!')"
check "$(has "${TMP}/i_download.log" 'Error downloading snapshot-99: an unreadable chunk is still in place')" "... a kept unreadable chunk is an error of this run (wget -nc would skip it for ever)"
check "$(has "${TMP}/i_urls.log" '^ARGS: .*--https-only')" "the recursive fetch is https-only (redirects and links with the key header too)"
check "$(has "${TMP}/i_urls.log" '^ARGS: .*--header=API-Key: dummykey')" "... and the key travels as a header"
check "$(lacks "${TMP}/i_download.log" 'Download complete!')" "... really no 'Download complete!'"
check "$([ "$(grep -c '^https://www.tng-project.org/api/SIMX/files/snapshot-9[89]/' "${TMP}/i_urls.log")" = "2" ] && echo 0 || echo 1)" \
      "both fetches went to https (the key is a header, never sent over plain http)"
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
echo "(J) --help runs nothing, --version prints the revision, DRY_RUN=1 prints the commands and runs nothing."
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/run_ps_pipeline.sh; do
    rc=0; "${scr}" --help >"${TMP}/j_help_${scr##*/}.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "${scr##*/} --help exits 0 (${rc})"
    check "$(lacks "${TMP}/j_help_${scr##*/}.log" 'Processing snapshot|RUNNING:|Slice: axis')" "... and starts nothing"
done
check "$(has "${TMP}/j_help_run_ps_pipeline.sh.log" 'COMPUTE ONLY')" "run_ps_pipeline.sh --help prints its header (and does not start the whole batch)"
check "$(has "${TMP}/j_help_run_dtfe.sh.log" '^Usage: ')" "run_dtfe.sh --help ends with the usage line"
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/download_snapshots.sh; do
    rc=0; env -u DTFE_SIM "${scr}" 99 >"${TMP}/j_nosim_${scr##*/}.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/j_nosim_${scr##*/}.log" 'no simulation given: pass -s SIMULATION')" = 0 ] && echo 0 || echo 1)" \
          "${scr##*/} without -s or DTFE_SIM refuses with exit 2: there is no default simulation (${rc})"
done
# no default data root either (config.sh require_data_root): every driver refuses before it reads or writes anything
NOROOT='no data root: set DTFE_DATA_ROOT to the folder that holds the simulations \(e\.g\. DTFE_DATA_ROOT=~/IllustrisTNG\)\.'
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/download_snapshots.sh; do
    rc=0; env -u DTFE_DATA_ROOT TNG_API_KEY=dummykey "${scr}" -s HASIC 99 >"${TMP}/j_noroot_${scr##*/}.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/j_noroot_${scr##*/}.log" "^Error: ${NOROOT}$")" = 0 ] \
             && [ "$(lacks "${TMP}/j_noroot_${scr##*/}.log" 'Processing snapshot|RUNNING:|Downloading|Data directory')" = 0 ] && echo 0 || echo 1)" \
          "${scr##*/} without DTFE_DATA_ROOT refuses with exit 2 and says so, before anything runs: there is no default data root (${rc})"
done
# -d names the simulation's folder, so it stands in for the data root: without a simulation the next refusal is the
# missing simulation, not the data root (nothing runs or downloads)
rc_d=""
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/download_snapshots.sh; do
    rc=0; env -u DTFE_DATA_ROOT -u DTFE_SIM TNG_API_KEY=dummykey "${scr}" -d "${TMP}/HASIC" 99 >"${TMP}/j_noroot_d.log" 2>&1 || rc=$?
    [ "$(has "${TMP}/j_noroot_d.log" 'no simulation given')" = 0 ] && [ "$(lacks "${TMP}/j_noroot_d.log" "${NOROOT}")" = 0 ] || rc="${rc}-msg"
    rc_d="${rc_d} ${rc}"
done
check "$([ "${rc_d}" = " 2 2 2" ] && echo 0 || echo 1)" "... -d DATA_DIR stands in for it: the refusal is then the missing simulation (${rc_d# })"
rc=0; DTFE_DATA_ROOT= scripts/run_ps_dtfe.sh -s HASIC 99 >"${TMP}/j_noroot_empty.log" 2>&1 || rc=$?
check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/j_noroot_empty.log" "${NOROOT}")" = 0 ] && echo 0 || echo 1)" \
      "... an empty DTFE_DATA_ROOT= is no data root either (${rc})"
rc=0; env -u DTFE_DATA_ROOT DRY_RUN=1 SIMS=all scripts/run_ps_pipeline.sh >"${TMP}/j_noroot_pipe.log" 2>&1 || rc=$?
check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/j_noroot_pipe.log" "^Error: ${NOROOT}$")" = 0 ] \
         && [ "$(lacks "${TMP}/j_noroot_pipe.log" '^Simulations:')" = 0 ] && echo 0 || echo 1)" \
      "run_ps_pipeline.sh without DTFE_DATA_ROOT refuses with exit 2 before SIMS=all looks for simulations (${rc})"
rc_h=""
for c in "scripts/run_dtfe.sh --help" "scripts/run_ps_dtfe.sh --help" "scripts/run_ps_pipeline.sh --help" "scripts/download_snapshots.sh -h"; do
    rc=0; env -u DTFE_DATA_ROOT ${c} >"${TMP}/j_noroot_help.log" 2>&1 || rc=$?
    rc_h="${rc_h} ${rc}"
done
check "$([ "${rc_h}" = " 0 0 0 0" ] && echo 0 || echo 1)" "... while --help (-h) of all four still works without it (${rc_h# })"
src="$(env -u DTFE_DATA_ROOT bash -c 'source scripts/config.sh; echo "sourced root=[${DATA_ROOT}]"' 2>&1)"
check "$([ "${src}" = "sourced root=[]" ] && echo 0 || echo 1)" "... and sourcing config.sh without it exits nothing and leaves DATA_ROOT empty (${src})"
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
echo "(K) the binaries' --help lists the phase-space, GPU, point-evaluation and server options; no known misspelling."
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
      "the full help carries none of the known misspellings"
check "$(has "${TMP}/k_full_ps.log" "'MPC_UNIT'")" "... and names the Makefile's MPC_UNIT, not DMPC_UNIT"

echo ""
echo "(L) scripts/install_lib.sh: the binaries kept through a failed rebuild."
LREPO="${TMP}/fake_repo"; rm -rf "${LREPO}"; mkdir -p "${LREPO}"
# the 2D double pair too: install.sh rebuilds it, so a failed build must be able to bring it back
( cd "${LREPO}" && printf 'old DTFE\n' > DTFE && printf 'old PS-DTFE\n' > PS-DTFE && printf 'old 2d\n' > DTFE-2d \
      && printf 'old 2d double\n' > DTFE-2d-double && printf 'old PS 2d double\n' > PS-DTFE-2d-double \
      && chmod +x DTFE PS-DTFE DTFE-2d DTFE-2d-double PS-DTFE-2d-double
  source "${ROOT_DIR}/scripts/install_lib.sh"
  n="$(backup_binaries "${LREPO}/.backup")"
  echo "backed up ${n}" > "${TMP}/l_backup.log"
  rm -f DTFE PS-DTFE DTFE-2d DTFE-2d-double PS-DTFE-2d-double; printf 'half-linked\n' > PS-DTFE   # 'make clean' then a failing build
  m="$(restore_binaries "${LREPO}/.backup")"
  echo "restored ${m}" >> "${TMP}/l_backup.log"
  printf 'line1\nline2\nld: symbol not found\n' > build.log
  build_failed_hint build.log 2 > "${TMP}/l_hint.log" )
check "$(has "${TMP}/l_backup.log" '^backed up 5$')" "backup_binaries counts the five binaries present (of the eight names)"
check "$(has "${TMP}/l_backup.log" '^restored 5$')" "restore_binaries puts the five back"
check "$([ "$(cat "${LREPO}/PS-DTFE")" = "old PS-DTFE" ] && [ "$(cat "${LREPO}/DTFE")" = "old DTFE" ] && [ -x "${LREPO}/DTFE-2d" ] && echo 0 || echo 1)" \
      "... byte for byte, the half-linked one replaced, the mode kept"
check "$([ "$(cat "${LREPO}/DTFE-2d-double")" = "old 2d double" ] && [ "$(cat "${LREPO}/PS-DTFE-2d-double")" = "old PS 2d double" ] \
         && [ -x "${LREPO}/PS-DTFE-2d-double" ] && echo 0 || echo 1)" "... the 2D double pair among them"
check "$(has "${TMP}/l_hint.log" 'ld: symbol not found')" "build_failed_hint prints the log's tail"
check "$(lacks "${TMP}/l_hint.log" '^line1$')" "... only the last N lines"
check "$(has "${TMP}/l_hint.log" 'make deps-check')" "... and the usual causes"
# install.sh rebuilds the double, 2D and 2D double sets it finds before 'make clean' (else a re-run would throw
# them away). install.sh itself is never run here: its detection block, cut out between 'EXTRA_SETS=()'
# and the 'also to be rebuilt' line, is sourced in fake folders with info() stubbed out
sed -n '/^EXTRA_SETS=()/,/also to be rebuilt/p' scripts/install.sh > "${TMP}/l_extra_sets.sh"
check "$(grep -q '^EXTRA_SETS=()' "${TMP}/l_extra_sets.sh" && grep -q 'DIM=2 DOUBLE=1' "${TMP}/l_extra_sets.sh" \
         && grep -q 'also to be rebuilt' "${TMP}/l_extra_sets.sh" && echo 0 || echo 1)" \
      "install.sh's extra-set block is found (EXTRA_SETS=() ... also to be rebuilt): a moved marker fails here, not silently"
extra_sets() {   # <PREC_ARG> <GPU_ARG> <binaries present...>: the make arguments of each set, one [..] per line
    local d="${TMP}/l_sets"; rm -rf "${d}"; mkdir -p "${d}"
    ( cd "${d}" && PREC_ARG="$1" && GPU_ARG="$2" && shift 2 && for b in DTFE PS-DTFE "$@"; do : > "${b}"; done
      info() { :; }
      source "${TMP}/l_extra_sets.sh" || true                   # its last line is false when there is no extra set
      for s in ${EXTRA_SETS[@]+"${EXTRA_SETS[@]}"}; do printf '[%s]' "${s}"; done )
}
got="$(extra_sets "" "METAL=1")"
check "$([ -z "${got}" ] && echo 0 || echo 1)" "... the single pair alone: no extra set (${got:-none})"
got="$(extra_sets "" "METAL=1" DTFE-2d DTFE-double DTFE-2d-double)"
check "$([ "${got}" = "[METAL=1 DOUBLE=1][DIM=2][DIM=2 DOUBLE=1]" ] && echo 0 || echo 1)" \
      "... DTFE-double, DTFE-2d and DTFE-2d-double present: the double set with the GPU backend, then DIM=2 and DIM=2 DOUBLE=1 (${got})"
got="$(extra_sets "" "" PS-DTFE-2d-double)"
check "$([ "${got}" = "[DIM=2 DOUBLE=1]" ] && echo 0 || echo 1)" "... either binary of a pair is enough: PS-DTFE-2d-double alone (${got})"
got="$(extra_sets "DOUBLE=1" "")"
check "$([ "${got}" = "[ DOUBLE=1]" ] && echo 0 || echo 1)" "... --double with no double pair yet builds it, CPU-only here (${got})"

echo ""
echo "(M) the run-log contract: a failed, stopped or killed re-run keeps the .runlog of the grids on disk."
# A driver that tees straight into <prefix>.runlog lets a re-run that fails early, or is stopped or killed,
# replace the log of the grids still on disk (the Runs browser then shows them as a failed run with the
# new settings). Copies of the drivers run here beside STAND-IN binaries (the drivers find theirs one level
# above their own folder), so a success, a failure, a stop and a SIGKILL are all deterministic and need no build.
MREPO="${TMP}/runlog_repo"; MDATA="${TMP}/runlog_data"; MPID="${TMP}/m_binary.pid"
mkdir -p "${MREPO}/scripts" "${MDATA}/SIM/snapdir_099"
cp scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/config.sh "${MREPO}/scripts/"
cat > "${MREPO}/DTFE" <<'FEOF'
#!/usr/bin/env bash
# stand-in binary. FAKE_MODE: ok (exit 0), fail (exit 1), hang (wait, in place of the binary, until killed),
# report (an auto-tune report line), write (as the real binary: '<out>.a_den' and '<out>.a_vel' written as
# '.part.<pid>' and renamed into place, a second later '<out>.badGridPoints0' rewritten IN PLACE when it exists)
echo "RUNNING: $0 $*"
printf 'fake attempt %s \033[1mcoloured\033[0m\n' "${FAKE_TAG:-}"
case "${FAKE_MODE:-ok}" in
    ok)     echo "total wall time : 1 s"; exit 0 ;;
    write)  sleep 1
            [ -f "$2.badGridPoints0" ] && echo new > "$2.badGridPoints0"
            for s in a_den a_vel; do echo new > "$2.$s.part.$$" && mv -f "$2.$s.part.$$" "$2.$s"; done
            echo "total wall time : 1 s"; exit 0 ;;
    fail)   echo "~~~ ERROR ~~~ a fake failure"; exit 1 ;;
    hang)   echo $$ > "${FAKE_PID}"; exec sleep 60 ;;
    report) echo "AUTO-TUNE-REPORT partition=1x1x1 mc=1 predicted_gb=0.1 budget_gb=1 over_budget=0"; exit 0 ;;
esac
FEOF
chmod +x "${MREPO}/DTFE"
for b in DTFE-double PS-DTFE PS-DTFE-double; do cp "${MREPO}/DTFE" "${MREPO}/${b}"; done
: > "${MDATA}/SIM/snapdir_099/combined_099.hdf5"
: > "${MDATA}/SIM/combined_ics.hdf5"          # run_ps_dtfe.sh: Lagrangian positions present, no h5py probe
mdrive() {   # <driver> <log> [VAR=value ...]: the copied driver on SIM 99; echoes its exit status
    local scr="$1" log="$2"; shift 2
    local rc=0
    env DTFE_DATA_ROOT="${MDATA}" "$@" "${MREPO}/scripts/${scr}" -s SIM -g 16 99 >"${log}" 2>&1 || rc=$?
    echo "${rc}"
}
mhang() {    # <driver> <log> <tag>: starts a run whose binary hangs, waits until it is in place; sets mpid
    rm -f "${MPID}"
    env DTFE_DATA_ROOT="${MDATA}" FAKE_MODE=hang FAKE_TAG="$3" FAKE_PID="${MPID}" \
        "${MREPO}/scripts/$1" -s SIM -g 16 99 >"$2" 2>&1 &
    mpid=$!
    for _ in $(seq 100); do [ -s "${MPID}" ] && break; sleep 0.1; done
}
for pair in "run_dtfe.sh output" "run_ps_dtfe.sh ps_output"; do
    scr="${pair% *}"; pfx="${pair#* }"
    L="${MDATA}/SIM/snapdir_099/${pfx}.runlog"
    rm -f "${L}" "${L}.failed" "${L}.partial"
    rc=$(mdrive "${scr}" "${TMP}/m_${pfx}_ok.log" FAKE_MODE=ok FAKE_TAG=first)
    check "$([ "${rc}" = "0" ] && [ -f "${L}" ] && [ ! -e "${L}.partial" ] && [ ! -e "${L}.failed" ] && echo 0 || echo 1)" \
          "${scr}: a good run's log is ${pfx}.runlog, and nothing else is left (exit ${rc})"
    check "$(lacks "${L}" $'\033\\[')" "... de-ANSI'd"
    cp "${L}" "${TMP}/m_${pfx}_first.runlog"
    # a failure (an early error, a bad option, a missing input): the grids on disk keep their log
    rc=$(mdrive "${scr}" "${TMP}/m_${pfx}_fail.log" FAKE_MODE=fail FAKE_TAG=second)
    check "$([ "${rc}" = "1" ] && cmp -s "${L}" "${TMP}/m_${pfx}_first.runlog" && echo 0 || echo 1)" \
          "... a failed re-run exits 1 and leaves the .runlog of the grids on disk byte for byte (${rc})"
    check "$([ ! -e "${L}.partial" ] && grep -aq 'fake attempt second' "${L}.failed" && ! grep -aq $'\033\\[' "${L}.failed" && echo 0 || echo 1)" \
          "... its own log is kept, de-ANSI'd, as ${pfx}.runlog.failed (for the failure diagnosis); no .partial left"
    check "$(has "${TMP}/m_${pfx}_fail.log" "Error processing snapshot 099 \(its log: .*${pfx}\.runlog\.failed\)")" "... and the driver names it"
    # a stop as the launcher's Stop does it: SIGTERM to the driver (the trap waits for the binary), then the binary
    mhang "${scr}" "${TMP}/m_${pfx}_stop.log" third
    check "$([ -f "${L}.partial" ] && cmp -s "${L}" "${TMP}/m_${pfx}_first.runlog" && echo 0 || echo 1)" \
          "... while a re-run goes its log is ${pfx}.runlog.partial; the .runlog stays as it was"
    kill -TERM "${mpid}" 2>/dev/null || true
    sleep 0.3
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    rc=0; wait "${mpid}" 2>/dev/null || rc=$?
    check "$([ "${rc}" = "143" ] && echo 0 || echo 1)" "... a stopped run ends by its signal, 143, so a calling script stops too (${rc})"
    check "$([ ! -e "${L}.partial" ] && cmp -s "${L}" "${TMP}/m_${pfx}_first.runlog" && grep -aq 'fake attempt third' "${L}.failed" && echo 0 || echo 1)" \
          "... the stopped attempt's log is the .runlog.failed now; the .runlog is untouched, no .partial left"
    check "$(has "${TMP}/m_${pfx}_stop.log" 'Stopped: snapshot 099 did not finish')" "... and the driver says so"
    # a run killed outright (SIGKILL, a restart): its .partial is left, the .runlog still untouched
    mhang "${scr}" "${TMP}/m_${pfx}_kill.log" fourth
    kill -KILL "${mpid}" 2>/dev/null || true
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true       # the orphaned binary
    wait "${mpid}" 2>/dev/null || true
    sleep 1                                                             # its orphaned tee writes its last lines
    check "$(grep -aq 'fake attempt fourth' "${L}.partial" && cmp -s "${L}" "${TMP}/m_${pfx}_first.runlog" && echo 0 || echo 1)" \
          "... a run killed outright leaves its ${pfx}.runlog.partial, and the .runlog still untouched"
    # the next success: its log is the .runlog, the stale .failed and .partial are gone
    rc=$(mdrive "${scr}" "${TMP}/m_${pfx}_ok2.log" FAKE_MODE=ok FAKE_TAG=fifth)
    check "$([ "${rc}" = "0" ] && grep -aq 'fake attempt fifth' "${L}" && [ ! -e "${L}.failed" ] && [ ! -e "${L}.partial" ] && echo 0 || echo 1)" \
          "... the next good run: its log is the .runlog; the stale .runlog.failed and .runlog.partial are gone (${rc})"
done
# an auto-tune report writes nothing next to the data: not the .runlog, nor a .partial or .failed (run_dtfe.sh has
# the knob too)
for pair in "run_dtfe.sh output" "run_ps_dtfe.sh ps_output"; do
    scr="${pair% *}"; pfx="${pair#* }"
    cp "${MDATA}/SIM/snapdir_099/${pfx}.runlog" "${TMP}/m_before_report.runlog"
    rc=$(mdrive "${scr}" "${TMP}/m_${pfx}_report.log" FAKE_MODE=report AUTO_TUNE_REPORT=1)
    check "$([ "${rc}" = "0" ] && grep -aqE '^  AUTO-TUNE-REPORT snap=099 ' "${TMP}/m_${pfx}_report.log" \
             && [ "$(has "${TMP}/m_${pfx}_report.log" '^RUNNING: .*--auto-tune-report')" = 0 ] && echo 0 || echo 1)" \
          "${scr} AUTO_TUNE_REPORT=1 passes --auto-tune-report and reports (${rc})"
    check "$(cmp -s "${MDATA}/SIM/snapdir_099/${pfx}.runlog" "${TMP}/m_before_report.runlog" \
             && [ ! -e "${MDATA}/SIM/snapdir_099/${pfx}.runlog.partial" ] \
             && [ ! -e "${MDATA}/SIM/snapdir_099/${pfx}.runlog.failed" ] && echo 0 || echo 1)" \
          "... and leaves the snapshot's run logs alone"
done

echo ""
echo "(O) the single-instance lock and the preflight of run_dtfe.sh / run_ps_dtfe.sh."
# Contract C1: '<prefix>.runlog.lock' is a directory holding the driver's pid while it runs that snapshot. A
# second driver on the same snapshot is refused (two TNG runs together get both SIGKILLed for memory);
# a lock whose driver is gone is stale and removed with a message; a stop releases it.
for pair in "run_dtfe.sh output" "run_ps_dtfe.sh ps_output"; do
    scr="${pair% *}"; pfx="${pair#* }"
    L="${MDATA}/SIM/snapdir_099/${pfx}.runlog"; K="${L}.lock"
    rm -rf "${K}"
    mhang "${scr}" "${TMP}/o_${pfx}_hold.log" sixth
    check "$([ -d "${K}" ] && [ "$(cat "${K}/pid" 2>/dev/null)" = "${mpid}" ] && echo 0 || echo 1)" \
          "${scr}: while it runs, ${pfx}.runlog.lock holds the driver's pid (${mpid})"
    cp "${L}" "${TMP}/o_${pfx}_before.runlog"
    rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_second.log" FAKE_MODE=ok FAKE_TAG=intruder)
    check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "... a second driver on the same snapshot is refused: exit 1 (${rc})"
    check "$(has "${TMP}/o_${pfx}_second.log" "another driver \(pid ${mpid}\) is running this snapshot")" "... naming the pid that holds it"
    check "$(lacks "${TMP}/o_${pfx}_second.log" '^RUNNING:')" "... and its binary never started"
    check "$([ -f "${L}.partial" ] && ! grep -aq 'intruder' "${L}.partial" && cmp -s "${L}" "${TMP}/o_${pfx}_before.runlog" \
             && [ "$(cat "${K}/pid" 2>/dev/null)" = "${mpid}" ] && echo 0 || echo 1)" \
          "... the running attempt's .partial, the grids' .runlog and the lock are untouched"
    kill -TERM "${mpid}" 2>/dev/null || true
    sleep 0.3
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    rc=0; wait "${mpid}" 2>/dev/null || rc=$?
    check "$([ "${rc}" = "143" ] && [ ! -e "${K}" ] && echo 0 || echo 1)" "... a stop still exits 143 and releases the lock (${rc})"
    # the driver alone SIGKILLed: its binary and tee keep running on the snapshot, so its lock is not
    # stale (removing it would start a second binary beside the orphan)
    mhang "${scr}" "${TMP}/o_${pfx}_orphan.log" orphan
    kill -KILL "${mpid}" 2>/dev/null || true
    wait "${mpid}" 2>/dev/null || true
    rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_beside.log" FAKE_MODE=ok FAKE_TAG=beside)
    check "$([ "${rc}" = "1" ] && [ -d "${K}" ] && grep -aq 'is gone, but a process still' "${TMP}/o_${pfx}_beside.log" \
             && ! grep -aq '^RUNNING:' "${TMP}/o_${pfx}_beside.log" && echo 0 || echo 1)" \
          "... a driver SIGKILLed while its run goes on: the next one is refused, names the process, keeps the lock (${rc})"
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    unlocked_root_wait() {  # until no process names '<prefix>.runlog.partial' (tee ends once the binary has)
        local a
        for _ in $(seq 50); do a="$(ps -A -ww -o args= 2>/dev/null)"; case "${a}" in *"${L}.partial"*) sleep 0.1 ;; *) return 0 ;; esac; done
    }
    unlocked_root_wait; rm -rf "${K}"
    # killed outright: the lock stays behind with a dead pid; the next run removes it and goes ahead
    mhang "${scr}" "${TMP}/o_${pfx}_kill.log" seventh
    kill -KILL "${mpid}" 2>/dev/null || true
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    wait "${mpid}" 2>/dev/null || true
    unlocked_root_wait
    check "$([ -d "${K}" ] && echo 0 || echo 1)" "... a SIGKILLed driver leaves its lock"
    rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_after.log" FAKE_MODE=ok FAKE_TAG=eighth)
    check "$([ "${rc}" = "0" ] && [ ! -e "${K}" ] && grep -aq 'fake attempt eighth' "${L}" && echo 0 || echo 1)" \
          "... the next run removes the stale lock, succeeds and leaves no lock (${rc})"
    check "$(has "${TMP}/o_${pfx}_after.log" 'removing a stale lock \(pid [0-9]+ is no longer running\)')" "... and says so"
    # a reused pid: alive, but not a driver -- stale as well
    sleep 30 & other=$!
    mkdir -p "${K}"; echo "${other}" > "${K}/pid"
    rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_reused.log" FAKE_MODE=ok)
    kill "${other}" 2>/dev/null || true; wait "${other}" 2>/dev/null || true
    check "$([ "${rc}" = "0" ] && [ ! -e "${K}" ] && echo 0 || echo 1)" "... a lock whose pid now belongs to another program is stale too (${rc})"
    # preflight: the grids must fit on the output volume (a 20000^3 grid needs ~1e14 bytes); DTFE_SPACE_CHECK=0 skips
    rc=0; env DTFE_DATA_ROOT="${MDATA}" FAKE_MODE=ok "${MREPO}/scripts/${scr}" -s SIM -g 20000 99 >"${TMP}/o_${pfx}_space.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "... grids larger than the free space: refused, exit 1 (${rc})"
    check "$(has "${TMP}/o_${pfx}_space.log" "GB free; this run's grids need about")" "... saying how much is free and needed"
    check "$(lacks "${TMP}/o_${pfx}_space.log" '^RUNNING:')" "... before the binary started"
    rc=0; env DTFE_DATA_ROOT="${MDATA}" FAKE_MODE=ok DTFE_SPACE_CHECK=0 "${MREPO}/scripts/${scr}" -s SIM -g 20000 99 >"${TMP}/o_${pfx}_nospace.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... DTFE_SPACE_CHECK=0 skips the space check (${rc})"
    # the data disk unplugged: one clear message up front, exit 2 (not a skip of every snapshot under a wrong path)
    rc=0; env DTFE_DATA_ROOT="${TMP}/no such disk/Illustris TNG" "${MREPO}/scripts/${scr}" -s SIM 99 >"${TMP}/o_${pfx}_root.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "2" ] && echo 0 || echo 1)" "... a data root that does not exist: exit 2 (${rc})"
    check "$(has "${TMP}/o_${pfx}_root.log" 'does not exist: is the data disk mounted')" "... asking whether the disk is mounted"
    check "$(lacks "${TMP}/o_${pfx}_root.log" 'Processing snapshot')" "... before any snapshot"
    # an output folder that is not writable (root writes anywhere: skipped under root, as in the Docker image)
    if [ "$(id -u)" != "0" ]; then
        chmod a-w "${MDATA}/SIM/snapdir_099"
        rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_ro.log" FAKE_MODE=ok)
        chmod u+w "${MDATA}/SIM/snapdir_099"
        check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "... an output folder that is not writable: refused, exit 1 (${rc})"
        check "$(has "${TMP}/o_${pfx}_ro.log" 'is not writable')" "... saying so"
    fi
done
# a report and a dry run take no lock: the launcher's memory check runs a report beside a live run
for pair in "run_ps_dtfe.sh ps_output" "run_dtfe.sh output"; do
    scr="${pair% *}"; pfx="${pair#* }"
    mhang "${scr}" "${TMP}/o_hold_report_${pfx}.log" ninth
    rc=$(mdrive "${scr}" "${TMP}/o_report_${pfx}.log" FAKE_MODE=report AUTO_TUNE_REPORT=1)
    rc2=$(mdrive "${scr}" "${TMP}/o_dry_${pfx}.log" DRY_RUN=1)
    check "$([ "${rc}" = "0" ] && [ "${rc2}" = "0" ] && [ "$(cat "${MDATA}/SIM/snapdir_099/${pfx}.runlog.lock/pid" 2>/dev/null)" = "${mpid}" ] && echo 0 || echo 1)" \
          "${scr}: an auto-tune report and a dry run go ahead beside a live run, and leave its lock alone (${rc}, ${rc2})"
    kill -TERM "${mpid}" 2>/dev/null || true; sleep 0.3
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    wait "${mpid}" 2>/dev/null || true
done

echo ""
echo "(R) run_ps_dtfe.sh: ORDERED_MERGE (1 by default), and --resume when a killed run left its checkpoint."
# Contracts K1/K2: the binary checkpoints a partitioned run in '<output>.runlog.checkpoint' (under --scratch-dir:
# '<scratch>/<basename>.<hash of the output path>.runlog.checkpoint') and removes it on success, so a folder there
# means a killed run. Against the stand-in binaries of (M), which print their arguments.
RSD="${MDATA}/SIM/snapdir_099"; RCK="${RSD}/ps_output.runlog.checkpoint"
rm -rf "${RCK}" "${RSD}/ps_output.runlog.lock"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_plain.log" FAKE_MODE=ok)
check "$([ "${rc}" = "0" ] && grep -aq '^RUNNING: ' "${TMP}/r_plain.log" && echo 0 || echo 1)" "a plain run goes (${rc})"
# the ordered (staged) merge is the default; the driver says it either way
check "$([ "$(lacks "${TMP}/r_plain.log" '^RUNNING: .*--(completion-order-merge|resume)')" = 0 ] \
         && [ "$(has "${TMP}/r_plain.log" '^RUNNING: .*--ordered-merge')" = 0 ] && echo 0 || echo 1)" \
      "... with --ordered-merge (the default) and no --resume"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_ordered.log" FAKE_MODE=ok ORDERED_MERGE=1)
check "$(has "${TMP}/r_ordered.log" '^RUNNING: .*--ordered-merge')" "ORDERED_MERGE=1 passes --ordered-merge (${rc})"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_compl.log" FAKE_MODE=ok ORDERED_MERGE=0)
check "$([ "$(has "${TMP}/r_compl.log" '^RUNNING: .*--completion-order-merge')" = 0 ] \
         && [ "$(lacks "${TMP}/r_compl.log" '^RUNNING: .*--ordered-merge')" = 0 ] && echo 0 || echo 1)" \
      "ORDERED_MERGE=0 passes --completion-order-merge, the merge as partitions finish (${rc})"
# FIXED_POINT_MERGE=1 passes the opt-in order-free merge in place of the merge-order flag
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_fixed.log" FAKE_MODE=ok FIXED_POINT_MERGE=1)
check "$([ "$(has "${TMP}/r_fixed.log" '^RUNNING: .*--fixed-point-merge')" = 0 ] \
         && [ "$(lacks "${TMP}/r_fixed.log" '^RUNNING: .*--(ordered-merge|completion-order-merge)')" = 0 ] && echo 0 || echo 1)" \
      "FIXED_POINT_MERGE=1 passes --fixed-point-merge and no merge-order flag (${rc})"
mkdir -p "${RCK}"; : > "${RCK}/checkpoint"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_resume.log" FAKE_MODE=ok)
check "$([ "${rc}" = "0" ] && grep -aqE -e '^RUNNING: .*--resume' "${TMP}/r_resume.log" && echo 0 || echo 1)" "a checkpoint folder beside the outputs: --resume is passed (${rc})"
check "$(has "${TMP}/r_resume.log" "Resuming: a killed run's checkpoint is in .*ps_output\.runlog\.checkpoint")" "... and the driver says so, naming the folder"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_noresume.log" FAKE_MODE=ok PS_RESUME=0)
check "$(lacks "${TMP}/r_noresume.log" '^RUNNING: .*--resume')" "PS_RESUME=0: no --resume (${rc})"
check "$(has "${TMP}/r_noresume.log" 'PS_RESUME=0, so this run starts over')" "... and the driver says the run starts over"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_report.log" FAKE_MODE=report AUTO_TUNE_REPORT=1)
check "$(lacks "${TMP}/r_report.log" '(--resume|Resuming)')" "an auto-tune report never resumes (${rc})"
# a folder with no 'checkpoint' file (a run killed during its first write, or one that never had room) holds
# nothing to resume: the binary refuses '--resume' there, so passing it would make every rerun fail
rm -f "${RCK}/checkpoint"; : > "${RCK}/checkpoint.tmp.99999"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_emptyck.log" FAKE_MODE=ok)
check "$(lacks "${TMP}/r_emptyck.log" '(^RUNNING: .*--resume|Resuming)')" "a checkpoint folder without a 'checkpoint' file: no --resume (${rc})"
rm -rf "${RCK}"
RSCR="${TMP}/r scratch"; mkdir -p "${RSCR}"
SCK="$(source scripts/config.sh && ps_checkpoint_dir "${RSD}/ps_output" "${RSCR}")"
check "$([[ "$(basename "${SCK}")" =~ ^ps_output\.[0-9a-f]{16}\.runlog\.checkpoint$ ]] && echo 0 || echo 1)" \
      "config.sh ps_checkpoint_dir under a scratch dir: <basename>.<16 hex>.runlog.checkpoint ($(basename "${SCK}"))"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_scr0.log" FAKE_MODE=ok SCRATCH_DIR="${RSCR}")
check "$(lacks "${TMP}/r_scr0.log" '^RUNNING: .*--resume')" "SCRATCH_DIR set, no checkpoint there: no --resume (${rc})"
mkdir -p "${SCK}"; : > "${SCK}/checkpoint"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_scr1.log" FAKE_MODE=ok SCRATCH_DIR="${RSCR}")
check "$(has "${TMP}/r_scr1.log" '^RUNNING: .*--resume')" "... the checkpoint in the scratch folder: --resume is passed (${rc})"
rm -rf "${SCK}"
# the plane's conventions: legacy only on request and only with points; the binary's default otherwise
: > "${TMP}/r_points.txt"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_pc_def.log" FAKE_MODE=ok SAMPLE_POINTS="${TMP}/r_points.txt")
rc2=$(mdrive run_ps_dtfe.sh "${TMP}/r_pc_leg.log" FAKE_MODE=ok SAMPLE_POINTS="${TMP}/r_points.txt" PTS_CONVENTIONS=legacy)
check "$([ "$(lacks "${TMP}/r_pc_def.log" '^RUNNING: .*--pts-conventions')" = 0 ] \
         && [ "$(has "${TMP}/r_pc_leg.log" '^RUNNING: .*--sample-points .*r_points\.txt.* --pts-conventions legacy')" = 0 ] && echo 0 || echo 1)" \
      "PTS_CONVENTIONS: the grids' (the default) passes nothing, legacy passes --pts-conventions legacy with the points (${rc} ${rc2})"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_pc_nopts.log" FAKE_MODE=ok PTS_CONVENTIONS=legacy)
check "$(lacks "${TMP}/r_pc_nopts.log" '^RUNNING: .*--pts-conventions')" "... and nothing without points (${rc})"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_pc_bad.log" FAKE_MODE=ok PTS_CONVENTIONS=old)
check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/r_pc_bad.log" "PTS_CONVENTIONS must be 'grid' or 'legacy'")" = 0 ] \
         && [ "$(lacks "${TMP}/r_pc_bad.log" '^RUNNING: ')" = 0 ] && echo 0 || echo 1)" \
      "... another value is refused before anything runs (exit ${rc})"

echo ""
echo "(P) run_ps_pipeline.sh's 'up to date' compares the run settings, not only the file times."
# File times alone would call a changed grid or field list 'up to date, skipping compute'. The pipeline asks
# run_ps_dtfe.sh (DRY_RUN=1) for the command it would run and compares its options with the RUNNING: line of the
# snapshot's <prefix>.runlog. Against the stand-in binaries of (M), on a real snapshot so the plane can be made.
PDATA="${TMP}/pipe_data"; rm -rf "${PDATA}"; mkdir -p "${PDATA}/SIM/snapdir_099"
cp "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${PDATA}/SIM/snapdir_099/"
: > "${PDATA}/SIM/combined_ics.hdf5"
cp scripts/run_ps_pipeline.sh "${MREPO}/scripts/"
ln -sfn "${ROOT_DIR}/python" "${MREPO}/python"
pdrive() {   # <log> [VAR=value ...]: the copied pipeline on SIM; echoes its exit status
    local log="$1"; shift
    local rc=0
    env DTFE_DATA_ROOT="${PDATA}" SIMS=SIM NU=8 GRID_SIZE=16 FIELDS="density_a velocity_a" PS_GPU=0 FAKE_MODE=ok \
        "$@" "${MREPO}/scripts/run_ps_pipeline.sh" >"${log}" 2>&1 || rc=$?
    echo "${rc}"
}
SD="${PDATA}/SIM/snapdir_099"
rc=$(pdrive "${TMP}/p_first.log")
head -c 512 /dev/zero > "${SD}/ps_output.pts_den"           # a slice of the 8^2 plane: 64 float64, as a real run leaves
touch -t 203001010000 "${SD}/ps_output.pts_den" "${SD}/ps_output.pts_velGrad"   # what a real run leaves (newer than its inputs)
check "$([ "${rc}" = "0" ] && grep -aq '^RUNNING: ./PS-DTFE[^ ]* .*--grid 16 ' "${SD}/ps_output.runlog" && echo 0 || echo 1)" \
      "a first pipeline run computes the snapshot, its run log carries the command (${rc})"
rc=$(pdrive "${TMP}/p_same.log")
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... a rerun with the same settings exits 0 (${rc})"
check "$(has "${TMP}/p_same.log" 'snap 99: ps_output.pts_den up to date \(same settings\), skipping compute')" "... and skips the snapshot: same settings"
check "$(lacks "${TMP}/p_same.log" 'run_ps_dtfe.sh -s SIM')" "... running nothing"
# a slice newer than the plane but of ANOTHER plane size (a 16^2 one against the 8^2 plane: TNG100's
# 4096^2 slices beside an older hires_plane_z8192) is not 'up to date': skipping the compute would fail every render
head -c 2048 /dev/zero > "${SD}/ps_output.pts_den"; touch -t 203001010000 "${SD}/ps_output.pts_den"
rc=$(pdrive "${TMP}/p_size.log" DRY_RUN=1)
check "$(has "${TMP}/p_size.log" 'snap 99: ps_output.pts_den holds 256 points, the plane 64: a slice of another plane, recomputing')" \
      "a slice of another plane size: recomputed, though newer than the plane and with the same settings (${rc})"
check "$(has "${TMP}/p_size.log" 'run_ps_dtfe.sh -s SIM -g 16 99')" "... run_ps_dtfe.sh is called for it"
head -c 512 /dev/zero > "${SD}/ps_output.pts_den"; touch -t 203001010000 "${SD}/ps_output.pts_den"
rc=$(pdrive "${TMP}/p_grid.log" GRID_SIZE=24 DRY_RUN=1)
check "$(has "${TMP}/p_grid.log" 'snap 99: settings differ from its ps_output.runlog, recomputing')" "a different grid: the snapshot is recomputed (${rc})"
check "$(grep -aq 'was: --grid 16$' "${TMP}/p_grid.log" && grep -aq 'now: --grid 24$' "${TMP}/p_grid.log" && echo 0 || echo 1)" "... the difference is shown"
check "$(has "${TMP}/p_grid.log" 'run_ps_dtfe.sh -s SIM -g 24 99')" "... and run_ps_dtfe.sh is called for it"
rc=$(pdrive "${TMP}/p_vw.log" PS_VOLUME_WEIGHTED=0 DRY_RUN=1)
check "$(has "${TMP}/p_vw.log" 'was: --ps-volume-weighted')" "a deposit option switched off (PS_VOLUME_WEIGHTED=0) counts too"
rc=$(pdrive "${TMP}/p_sub.log" FIELDS=density_a DRY_RUN=1)
check "$(has "${TMP}/p_sub.log" 'up to date \(same settings\)')" "fewer fields than the grids on disk carry: still up to date"
rc=$(pdrive "${TMP}/p_more.log" FIELDS="density_a velocity_a tweb_a" DRY_RUN=1)
check "$(has "${TMP}/p_more.log" 'settings differ')" "... a field the grids lack: recomputed"
rc=$(pdrive "${TMP}/p_gpu.log" PS_GPU=1 DRY_RUN=1)
check "$(has "${TMP}/p_gpu.log" 'up to date \(same settings\)')" "the GPU/CPU choice alone does not make grids stale"
# the plane's conventions are a setting of the slice; the pipeline hands them to run_ps_dtfe.sh
rc=$(pdrive "${TMP}/p_pc.log" PTS_CONVENTIONS=legacy DRY_RUN=1)
check "$([ "$(has "${TMP}/p_pc.log" 'snap 99: settings differ from its ps_output.runlog, recomputing')" = 0 ] \
         && grep -aq 'now: --pts-conventions legacy$' "${TMP}/p_pc.log" \
         && [ "$(has "${TMP}/p_pc.log" '^\+ env .*PTS_CONVENTIONS=legacy .*run_ps_dtfe\.sh -s SIM')" = 0 ] && echo 0 || echo 1)" \
      "PTS_CONVENTIONS=legacy: a slice in the grids' conventions is recomputed, the difference shown, the knob passed on (${rc})"
rc=$(pdrive "${TMP}/p_pcg.log" PTS_CONVENTIONS=grid DRY_RUN=1)
check "$(has "${TMP}/p_pcg.log" 'up to date \(same settings\)')" "... PTS_CONVENTIONS=grid (the default, said or not): up to date (${rc})"
mv "${SD}/ps_output.runlog" "${SD}/ps_output.runlog.keep"
rc=$(pdrive "${TMP}/p_nolog.log" GRID_SIZE=24 DRY_RUN=1)
check "$(has "${TMP}/p_nolog.log" 'up to date by file times \(no ps_output.runlog to compare the settings with\)')" \
      "no run log: judged by the file times, and said so"
mv "${SD}/ps_output.runlog.keep" "${SD}/ps_output.runlog"

echo ""
echo "(S) production defaults, OUTPUT_PREFIX and a rerun's left-over files."
# Against the stand-in binaries of (M) in a data folder of its own. The drivers' GPU default follows the Makefile's
# stamp in the pair's object folder (MREPO/o, o_ps, o_d), which the stand-in repository lacks until made here.
SDATA="${TMP}/sup_data"; SSD="${SDATA}/SIM/snapdir_099"; rm -rf "${SDATA}"; mkdir -p "${SSD}"
: > "${SSD}/combined_099.hdf5"; : > "${SDATA}/SIM/combined_ics.hdf5"
sdrive() {   # <driver> <log> [VAR=value ...]: the copied driver on SDATA's SIM 99; echoes its exit status
    local scr="$1" log="$2"; shift 2
    local rc=0
    env -u DTFE_METAL -u PS_METAL -u DTFE_EXACT_AVERAGE -u PS_VOLUME_WEIGHTED -u PS_CAUSTICS -u FIELDS -u OUTPUT_PREFIX \
        -u PS_LINEAR_DEPOSIT -u DTFE_PRECISION DTFE_DATA_ROOT="${SDATA}" "$@" "${MREPO}/scripts/${scr}" -s SIM -g 16 99 >"${log}" 2>&1 || rc=$?
    echo "${rc}"
}
sline() { grep -aE -e '^ +\./(PS-)?DTFE(-double)? ' "$1" | head -1; }      # the DRY RUN's planned command
gpu_of() {   # <driver> <stamp or none> [VAR=value ...]: 'gpu' / 'cpu' as the planned command says
    local scr="$1" st="$2"; shift 2
    rm -rf "${MREPO}/o" "${MREPO}/o_ps" "${MREPO}/o_d" "${MREPO}/o_ps_d"
    if [ "${st}" != "none" ]; then mkdir -p "${MREPO}/${st%%/*}"; : > "${MREPO}/${st}"; fi
    sdrive "${scr}" "${TMP}/s_gpu.log" DRY_RUN=1 "$@" >/dev/null
    sline "${TMP}/s_gpu.log" | grep -qE -e ' --(ps-)?gpu( |$)' && echo gpu || echo cpu
}
g=""
for c in "run_dtfe.sh o/.gpu_mode_metal" "run_dtfe.sh o/.gpu_mode_cuda" "run_dtfe.sh o/.gpu_mode_off" \
         "run_dtfe.sh o/.gpu_mode_cudaemu" "run_dtfe.sh none" "run_ps_dtfe.sh o_ps/.gpu_mode_metal" \
         "run_ps_dtfe.sh o_ps/.gpu_mode_hip" "run_ps_dtfe.sh o_ps/.gpu_mode_off" "run_ps_dtfe.sh o/.gpu_mode_metal"; do
    g="${g} $(gpu_of ${c})"
done
check "$([ "${g}" = " gpu gpu cpu cpu cpu gpu gpu cpu cpu" ] && echo 0 || echo 1)" \
      "the GPU by default exactly when the pair's object folder says metal / cuda / hip (not off, the CPU-emulated cudaemu, none, or the other pair's) (${g# })"
g="$(gpu_of run_dtfe.sh o/.gpu_mode_metal DTFE_METAL=0) $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal PS_METAL=0)"
g="${g} $(gpu_of run_dtfe.sh o/.gpu_mode_off DTFE_METAL=1) $(gpu_of run_dtfe.sh o/.gpu_mode_metal DTFE_PRECISION=double)"
g="${g} $(gpu_of run_dtfe.sh o_d/.gpu_mode_metal DTFE_PRECISION=double)"
check "$([ "${g}" = "cpu cpu gpu cpu gpu" ] && echo 0 || echo 1)" \
      "... DTFE_METAL / PS_METAL = 0 or 1 wins over the stamp; the double pair reads o_d (${g})"
check "$(has "${TMP}/s_gpu.log" "^'_a' interpolation: GPU \(--gpu\) \(DTFE_METAL unset: DTFE-double is a GPU build")" "... and the driver says which, and why"
rm -rf "${MREPO}/o" "${MREPO}/o_ps" "${MREPO}/o_d" "${MREPO}/o_ps_d"
rc=$(sdrive run_ps_dtfe.sh "${TMP}/s_psdef.log" DRY_RUN=1)
check "$([ "${rc}" = "0" ] && sline "${TMP}/s_psdef.log" | grep -qE -e '--field +density_a +velocity_a +gradient_a +divergence_a +shear_a +vorticity_a +dispersion_a +tweb_a +vweb_a +--lambda_th +0\.3 .*--ps-volume-weighted .*--ps-caustics' \
         && echo 0 || echo 1)" "run_ps_dtfe.sh's defaults are production's: the nine fields, volume-weighted moments, the caustic classes (${rc})"
rc=$(sdrive run_ps_dtfe.sh "${TMP}/s_pslin.log" DRY_RUN=1 PS_LINEAR_DEPOSIT=1)
rc2=$(sdrive run_ps_dtfe.sh "${TMP}/s_pslin2.log" DRY_RUN=1 PS_LINEAR_DEPOSIT=1 PS_VOLUME_WEIGHTED=1)
check "$(sline "${TMP}/s_pslin.log" | grep -q -e '--ps-linear-deposit' && ! sline "${TMP}/s_pslin.log" | grep -q -e '--ps-volume-weighted' \
         && grep -aq 'velocity moments stay mass-weighted' "${TMP}/s_pslin.log" && sline "${TMP}/s_pslin2.log" | grep -q -e '--ps-volume-weighted' \
         && echo 0 || echo 1)" "... PS_LINEAR_DEPOSIT=1 drops the default volume weighting (said); an explicit PS_VOLUME_WEIGHTED=1 stays (${rc}, ${rc2})"
rc=$(sdrive run_dtfe.sh "${TMP}/s_dtdef.log" DRY_RUN=1)
check "$([ "${rc}" = "0" ] && sline "${TMP}/s_dtdef.log" | grep -qE -e 'snapdir_099/output +--grid +16 +--padding +15 +--periodic +--field +density_a +velocity_a +gradient_a +divergence_a +shear_a +vorticity_a +tweb_a +vweb_a +--lambda_th +0\.3 +--exact-average$' \
         && echo 0 || echo 1)" "run_dtfe.sh's defaults: padding 15, the eight fields with the T-web / V-web, the exact cell average (${rc})"
# OUTPUT_PREFIX for run_dtfe.sh: the files and the run log under it; a PS-DTFE-style prefix refused before anything runs
rc=$(sdrive run_dtfe.sh "${TMP}/s_pfx.log" OUTPUT_PREFIX=output_mc DTFE_EXACT_AVERAGE=0 FAKE_MODE=ok)
check "$([ "${rc}" = "0" ] && [ -f "${SSD}/output_mc.runlog" ] && [ ! -e "${SSD}/output.runlog" ] \
         && grep -aq '^RUNNING: .*snapdir_099/output_mc --grid' "${SSD}/output_mc.runlog" && echo 0 || echo 1)" \
      "run_dtfe.sh OUTPUT_PREFIX=output_mc: the binary's output and the run log are output_mc.* (${rc})"
rc=$(sdrive run_dtfe.sh "${TMP}/s_pfxps.log" OUTPUT_PREFIX=ps_output FAKE_MODE=ok)
check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/s_pfxps.log" "is PS-DTFE's naming")" = 0 ] && [ "$(lacks "${TMP}/s_pfxps.log" '^RUNNING:')" = 0 ] \
         && echo 0 || echo 1)" "... a ps_* prefix (an export meant for run_ps_dtfe.sh) is refused, exit 2, nothing run (${rc})"
rm -f "${SSD}/output_mc."*
# a rerun's left-over files: seeded as an earlier, fuller run left them
s_seed() {
    rm -rf "${SSD}/output."* "${SSD}/output.superseded"
    for f in a_den a_vel pts_den a_tweb badGridPoints0; do echo old > "${SSD}/output.${f}"; done
    echo '{}' > "${SSD}/output.gui.json"; mkdir -p "${SSD}/output.runlog.checkpoint"
    printf 'RUNNING: ./DTFE old run\nold log\n' > "${SSD}/output.runlog"; echo failed > "${SSD}/output.runlog.failed"
    touch -t 202601010000 "${SSD}/output."*
}
s_seed
rc=$(sdrive run_dtfe.sh "${TMP}/s_sup.log" FAKE_MODE=write)
SUP="$(ls -d "${SSD}/output.superseded/"*/ 2>/dev/null | head -1)"; SUP="${SUP%/}"
check "$([ "${rc}" = "0" ] && [ -n "${SUP}" ] && [ "$(ls "${SUP}" | tr '\n' ' ')" = "output.a_tweb output.pts_den output.runlog " ] \
         && [ "$(cat "${SUP}/output.pts_den")" = old ] && grep -aq '^old log$' "${SUP}/output.runlog" \
         && [[ "$(basename "${SUP}")" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]{6}$ ]] && echo 0 || echo 1)" \
      "a rerun that exits 0: the files it did not write (.pts_den, .a_tweb) MOVED to output.superseded/<date_time>/ with the old run log (${rc}; $(ls "${SUP}" 2>/dev/null | tr '\n' ' '))"
check "$([ "$(cat "${SSD}/output.a_den")" = new ] && [ "$(cat "${SSD}/output.a_vel")" = new ] && [ "$(cat "${SSD}/output.badGridPoints0")" = new ] \
         && grep -aq 'fake attempt' "${SSD}/output.runlog" && [ -f "${SSD}/output.gui.json" ] && [ -d "${SSD}/output.runlog.checkpoint" ] \
         && [ ! -e "${SSD}/output.pts_den" ] && [ ! -e "${SSD}/output.a_tweb" ] && [ ! -e "${SSD}/output.runlog.failed" ] && echo 0 || echo 1)" \
      "... the files it wrote stay (renamed into place, or rewritten in place like .badGridPoints0), the run log is the new one; the settings sidecar and a checkpoint are not touched"
check "$(has "${TMP}/s_sup.log" '2 file\(s\) of an earlier run with this prefix moved to .*output\.superseded/.* \(nothing deleted')" "... and the driver says so"
s_seed
rc=$(sdrive run_dtfe.sh "${TMP}/s_supfail.log" FAKE_MODE=fail)
check "$([ "${rc}" = "1" ] && [ ! -e "${SSD}/output.superseded" ] && [ "$(cat "${SSD}/output.pts_den")" = old ] && grep -aq '^old log$' "${SSD}/output.runlog" \
         && echo 0 || echo 1)" "a FAILED rerun moves nothing: the earlier run's set stays whole with its run log (${rc})"
rc=$(sdrive run_dtfe.sh "${TMP}/s_supdry.log" DRY_RUN=1)
rc2=$(sdrive run_ps_dtfe.sh "${TMP}/s_suprep.log" FAKE_MODE=report AUTO_TUNE_REPORT=1 OUTPUT_PREFIX=output)
rc3=$(sdrive run_dtfe.sh "${TMP}/s_suprep_std.log" FAKE_MODE=report AUTO_TUNE_REPORT=1)
check "$([ ! -e "${SSD}/output.superseded" ] && [ -f "${SSD}/output.pts_den" ] && [ -f "${SSD}/output.a_tweb" ] && echo 0 || echo 1)" \
      "... nor does a dry run or an auto-tune report (${rc}, ${rc2}, run_dtfe.sh ${rc3})"
rm -rf "${SSD}/output."* "${SSD}/output.superseded"
rc=$(sdrive run_dtfe.sh "${TMP}/s_supnone.log" FAKE_MODE=write)
check "$([ "${rc}" = "0" ] && [ ! -e "${SSD}/output.superseded" ] && [ "$(lacks "${TMP}/s_supnone.log" 'Superseded')" = 0 ] && echo 0 || echo 1)" \
      "... and a first run into an empty prefix makes no superseded folder (${rc})"
# run_ps_pipeline.sh's GPU default: PS_GPU unset follows the PS binary's stamp as run_ps_dtfe.sh does, so a CPU-only
# build is never asked for the GPU (the copy (P) made, on SDATA's SIM: a dry run plans the plane and the snapshot)
pgpu_of() {   # <stamp or none> [VAR=value ...]: 'gpu' / 'cpu' as the pipeline's planned run_ps_dtfe.sh call says
    local st="$1"; shift
    rm -rf "${MREPO}/o" "${MREPO}/o_ps" "${MREPO}/o_d" "${MREPO}/o_ps_d"
    if [ "${st}" != "none" ]; then mkdir -p "${MREPO}/${st%%/*}"; : > "${MREPO}/${st}"; fi
    env -u PS_GPU -u PS_METAL -u DTFE_PRECISION -u SIMS -u SNAPS DTFE_DATA_ROOT="${SDATA}" SIMS=SIM NU=8 DRY_RUN=1 "$@" \
        "${MREPO}/scripts/run_ps_pipeline.sh" >"${TMP}/s_pgpu.log" 2>&1 || true
    local line
    line="$(grep -aE -e 'run_ps_dtfe\.sh -s SIM ' "${TMP}/s_pgpu.log" | tail -1)"
    if [ -z "${line}" ]; then echo none
    elif [[ "${line}" == *" -m 99"* && "${line}" == *"PS_METAL=1 "* ]]; then echo gpu
    elif [[ "${line}" != *" -m "* && "${line}" == *"PS_METAL=0 "* ]]; then echo cpu
    else echo mixed; fi
}
g=""
for c in "o_ps/.gpu_mode_metal" "o_ps/.gpu_mode_cuda" "o_ps/.gpu_mode_off" "o_ps/.gpu_mode_cudaemu" "none" \
         "o/.gpu_mode_metal"; do
    g="${g} $(pgpu_of ${c})"
done
check "$([ "${g}" = " gpu gpu cpu cpu cpu cpu" ] && echo 0 || echo 1)" \
      "run_ps_pipeline.sh, PS_GPU unset: the GPU (-m, PS_METAL=1) exactly when the PS binary's object folder says metal / cuda (not off, cudaemu, none, or the standard pair's) (${g# })"
g="$(pgpu_of o_ps/.gpu_mode_metal PS_GPU=0) $(pgpu_of o_ps/.gpu_mode_off PS_GPU=1)"
g="${g} $(pgpu_of o_ps/.gpu_mode_metal DTFE_PRECISION=double) $(pgpu_of o_ps_d/.gpu_mode_metal DTFE_PRECISION=double)"
check "$([ "${g}" = "cpu gpu cpu gpu" ] && echo 0 || echo 1)" "... PS_GPU = 0 or 1 wins over the stamp; the double pair reads o_ps_d (${g})"
check "$(has "${TMP}/s_pgpu.log" 'GPU: 1 \(PS_GPU unset: PS-DTFE-double is a GPU build\)')" "... and the header says which, and why"
rm -rf "${MREPO}/o" "${MREPO}/o_ps" "${MREPO}/o_d" "${MREPO}/o_ps_d"
rm -rf "${SDATA}"

echo ""
echo "(N) the Makefile stamps BUILD_MODE, ARCH_FLAGS and EXTRA_FLAGS in the object dirs."
# Without the stamp, 'make DTFE BUILD_MODE=debug' after a release build would find every object up to date and link
# the -O3 objects with the sanitizers (an ARCH_FLAGS / EXTRA_FLAGS change would keep the old objects the same way).
# The stamp's name carries the mode and a checksum of the three, so a switch wipes the objects. A dry run of the
# mode check alone (make -n -p) prints the stamp and builds nothing; the PS-DTFE target would start its recursive
# make, which make runs even under -n.
stamp() { make -n -p ps_gpu_mode_check "$@" 2>/dev/null | sed -n 's/^FLAGS_STAMP := //p' | head -1 || true; }
s_rel="$(stamp)"; s_dbg="$(stamp BUILD_MODE=debug)"
s_ext="$(stamp EXTRA_FLAGS=-DDTFE_STAMP_PROBE)"; s_arch="$(stamp ARCH_FLAGS=-mtune=stamp-probe)"
s_quote="$(stamp "EXTRA_FLAGS=-DQ='1'")"
check "$([[ "${s_rel}" =~ ^\.flags_release_[0-9]+$ ]] && [[ "${s_dbg}" =~ ^\.flags_debug_[0-9]+$ ]] \
         && [ "${s_rel#.flags_release_}" != "${s_dbg#.flags_debug_}" ] && echo 0 || echo 1)" \
      "BUILD_MODE=debug and the default release print different stamps (${s_rel} / ${s_dbg})"
check "$([[ "${s_ext}" =~ ^\.flags_release_[0-9]+$ ]] && [[ "${s_arch}" =~ ^\.flags_release_[0-9]+$ ]] \
         && [ "${s_ext}" != "${s_rel}" ] && [ "${s_arch}" != "${s_rel}" ] && echo 0 || echo 1)" \
      "... EXTRA_FLAGS and ARCH_FLAGS change the checksum too (${s_ext} / ${s_arch})"
check "$([[ "${s_quote}" =~ ^\.flags_release_[0-9]+$ ]] && echo 0 || echo 1)" "... and a quote in EXTRA_FLAGS still gives a stamp (${s_quote:-none})"
OBJ_PS=o_ps; precision_double && OBJ_PS=o_ps_d
if [ -d "${OBJ_PS}" ]; then
    stamps="$(ls -A "${OBJ_PS}" | grep -E '^\.flags_' || true)"
    check "$([ "$(printf '%s\n' "${stamps}" | grep -cE '^\.flags_[a-z]+_[0-9]+$')" = "1" ] && [ "$(printf '%s\n' "${stamps}" | grep -c .)" = "1" ] \
             && echo 0 || echo 1)" "the build of ${PS_BIN} left exactly one such stamp in ${OBJ_PS} (${stamps:-none})"
fi

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
