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
#     family and lets a flat <root>/<sim> win, and run_ps_pipeline.sh discovers nested sims -- and exits 1
#     when it finds nothing to compute (no data root, a mistyped SIMS, a SNAPS matching nothing).
#  H) scripts/run_dtfe.sh passes --exact-average to the standard binary by default (2026-10-13; -e and
#     DTFE_EXACT_AVERAGE=1 too, DTFE_EXACT_AVERAGE=0 not); its padding default is 15 (a caller's PADDING /
#     DTFE_PADDING wins), run_ps_dtfe.sh keeps config.sh's 50. run_dtfe.sh AUTO_TUNE_REPORT=1 (2026-10-13, as
#     run_ps_dtfe.sh): the standard binary's own report line per snapshot ('snap=099'), exit 0, nothing written.
#  I) download_snapshots.sh + merge_HDF5.py never turn a half-finished download into a merged snapshot.
#  J) --help runs nothing (run_ps_pipeline.sh --help used to start the whole batch), --version prints the
#     revision, an argument to the pipeline is an error, DRY_RUN=1 prints the drivers' commands and runs nothing.
#  K) the binaries' own --help lists the phase-space / GPU / point-evaluation / server options the scripts and the
#     launcher use (it listed none of them), and the help texts carry none of the survey's misspellings.
#  L) scripts/install_lib.sh keeps a user's binaries through a failed rebuild: backup, restore (the 2D double pair
#     too), the failure hint; install.sh's detection of the extra sets it rebuilds (sourced alone, never run).
#  M) the run-log contract of run_dtfe.sh / run_ps_dtfe.sh, against stand-in binaries: a run's log is
#     <prefix>.runlog.partial while it goes, <prefix>.runlog after exit 0, <prefix>.runlog.failed after a
#     failure or a stop; a failed, stopped or killed re-run never touches the .runlog of the grids on disk.
#  N) the Makefile stamps BUILD_MODE, ARCH_FLAGS and EXTRA_FLAGS in the object dirs (a dry run, builds nothing).
#  O) the drivers' single-instance lock (<prefix>.runlog.lock: a second driver refused, stale locks removed, a stop
#     releases it) and their preflight (data disk mounted, output folder writable with room for the grids).
#  P) run_ps_pipeline.sh's freshness test compares the run settings with the snapshot's run log, not only file times.
#  Q) scripts/thesis_runs.sh (the thesis run recipe): DRY_RUN=1 prints the exact settings of each stage, runs nothing;
#     the voids stage hands each figure its box, gate and flags (the size function at fixed R and the convergence
#     check ungated, per box), in order, draws each box's environment again after its tracking, and goes on after a
#     failed one (stand-in PY). The overnight stages (2026-10-12): DRY_RUN=1 of 'overnight' plans dtfe (TNG100
#     099 050 017) < caustics2 (050 033, the z = 0 caustic run's flags whatever the caller's shell says, the
#     Lagrangian padding the binary's default) < refresh < prune and runs nothing; finished snapshots are skipped (FORCE=1 replans them); refresh MOVES the untagged
#     caustic_skeleton_z0.00.* into old_untagged (never deletes, never overwrites, only beside the tagged png); a
#     second recipe is refused by the lock, a stale lock removed; the heavy stages refuse while a PS-DTFE runs.
#     2026-10-13: 'production' and 'tng50z0' are refused without HISTORICAL=1 (and pin the caustics off); ps27 and dtfe27
#     plan the 27 snapshots of the 2026-10-12 drivers in their order with their exact flags (the standard-DTFE ones with
#     the exact average at padding 15), the caller's knobs ignored; a snapshot is skipped only when its run log shows
#     those settings; a real run (stand-in binaries and ps) MOVES the old files to _backup/<prefix>_<date>/... first and
#     back again when the run fails. 2026-10-14: dtfe27planes plans the standard-DTFE 4096^2 planes of the same 27
#     snapshots (dtfe27's splits and order, the box's hires_plane_z4096.bin, --pts-vel-grad, density_a only, padding 15,
#     GPU, the exact average) under the prefix output_plane, the caller's knobs ignored; 'all' runs it after dtfe27; a
#     snapshot is skipped only when output_plane.runlog shows that plane run; a real run moves old output_plane.* files
#     to _backup/output_plane_<date>/... and never touches the output.* grids beside them.
#  R) run_ps_dtfe.sh ORDERED_MERGE=1 (the default since 2026-10-14) passes --ordered-merge, ORDERED_MERGE=0
#     --completion-order-merge; a killed run's checkpoint (<prefix>.runlog.checkpoint, or
#     config.sh ps_checkpoint_dir's name under SCRATCH_DIR) makes the next run pass --resume and say so; PS_RESUME=0,
#     an auto-tune report and a checkpoint folder with no 'checkpoint' file in it do not.
#  S) the production defaults and the rerun hygiene of 2026-10-13 (audit finding 7), against the stand-in binaries:
#     the GPU when the Makefile's stamp says the binary has one (metal/cuda/hip; not off or cudaemu; DTFE_METAL /
#     PS_METAL=0 wins), the drivers' production flags by default (run_ps_dtfe.sh: volume weighting, caustics, the nine
#     fields; run_dtfe.sh: the exact average and the web fields; PS_LINEAR_DEPOSIT=1 drops the volume weighting),
#     run_dtfe.sh's OUTPUT_PREFIX (a 'ps_*' one refused), and a rerun's left-over files: after exit 0 the files the run
#     did not write are MOVED to <prefix>.superseded/<stamp>/ with the old run log (the rewritten ones, the run log
#     family, a lock and a checkpoint stay); a failed run, a dry run and an auto-tune report move nothing.
#     run_ps_pipeline.sh's PS_GPU unset follows the PS binary's stamp the same way (it was 1 on every build).
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
check "$([ "${default_grid}" = "512" ] && echo 0 || echo 1)" \
      "config.sh's default is the production 512 when nothing overrides it (got ${default_grid}; 1024 until 2026-10-13)"

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
rc=0; DRY_RUN=1 DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline.log" 2>&1 || rc=$?
check "$(has "${TMP}/g_pipeline.log" '^Simulations:.* FAM-1-Dark')" \
      "run_ps_pipeline.sh discovers the nested FAM-1-Dark"
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... and a dry run with a snapshot to plan exits 0 (${rc})"
# the pipeline's knobs (2026-10-04): the GPU flag follows PS_GPU, the plane's geometry reaches
# make_image_plane.py, and a plane whose sidecar has another geometry is regenerated. PS_GPU unset (2026-10-13): the
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
    DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline2.log" 2>&1 || true
check "$(has "${TMP}/g_pipeline2.log" 'PS_VOLUME_WEIGHTED=0 .*run_ps_dtfe.sh -s FAM-1-Dark -g [0-9]+ 99')" \
      "PS_GPU=0 drops -m and PS_VOLUME_WEIGHTED=0 is passed on"
check "$(has "${TMP}/g_pipeline2.log" 'make_image_plane.py .*--axis z --nu [0-9]+ --planes 4 --thickness 3 --supersample 2 --center 40 --u0 0 --u1 50 --v0 0 --v1 25 -o ')" \
      "PLANES, THICKNESS, SUPERSAMPLE, CENTER and WINDOW reach make_image_plane.py"
# finding nothing to compute is a failure (2026-10-08): with the data drive unmounted, a mistyped SIMS or
# a SNAPS matching no snapshot the pipeline printed ALL DONE and exited 0, which the launcher showed as a success
mkdir -p "${TMP}/empty root"
rc=0; DTFE_DATA_ROOT="${TMP}/empty root" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_none.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "run_ps_pipeline.sh with no simulation under the data root exits 1 (${rc})"
check "$(has "${TMP}/g_pipe_none.log" '^!! nothing to compute')" "... says there is nothing to compute"
check "$(lacks "${TMP}/g_pipe_none.log" 'ALL DONE')" "... and not ALL DONE"
rc=0; DTFE_DATA_ROOT="${TMP}/no such root" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_unmounted.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a data root that does not exist (the drive unplugged) exits 1 (${rc})"
check "$(has "${TMP}/g_pipe_unmounted.log" 'is the data drive mounted')" "... and asks whether the drive is mounted"
rc=0; DRY_RUN=1 SIMS=FAM-1-Drak DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_typo.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a mistyped SIMS exits 1 (${rc})"
check "$(has "${TMP}/g_pipe_typo.log" '^!! nothing to compute: no combined snapshot of FAM-1-Drak ')" "... naming the simulation"
rc=0; DRY_RUN=1 SNAPS="50 67" DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_snaps.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a SNAPS matching no snapshot exits 1 (${rc})"
check "$(has "${TMP}/g_pipe_snaps.log" 'FAM-1-Dark: no combined snapshots numbered 50 67, skipping')" "... naming the numbers it looked for"
rc=0; DRY_RUN=1 SIMS="FAM-1-Drak FAM-1-Dark" DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_half.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "one simulation of two without snapshots is a partial skip: exit 0 (${rc})"
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
    # every snapshot up to date is a success, not 'nothing to compute' (2026-10-08); a slice of the 8^2 plane holds
    # 64 points (the pipeline compares the count since 2026-10-11)
    head -c 512 /dev/zero > "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_den"
    touch -t 203001010000 "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_den" \
                          "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_velGrad"
    # the settings of the (A)-style run that made these grids: since 2026-10-11 the pipeline compares them too (P)
    rc6=0; DRY_RUN=1 NU=8 GRID_SIZE=24 FIELDS=density_a AVG_SUBSAMPLES=1 PS_VOLUME_WEIGHTED=1 PTS_VEL_GRAD=0 \
        DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline6.log" 2>&1 || rc6=$?
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
    check "$([ -s "${TMP}/BADIN/snapdir_099/output.runlog.failed" ] && [ ! -e "${TMP}/BADIN/snapdir_099/output.runlog" ] \
             && [ ! -e "${TMP}/BADIN/snapdir_099/output.runlog.partial" ] && echo 0 || echo 1)" \
          "... its log is output.runlog.failed: no output.runlog claims grids it never wrote (2026-10-08)"
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
    # 2026-10-12: standard DTFE pads with 15 spacings by default (complete at z = 0 on TNG50/100/300), not config.sh's
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
    # 2026-10-13: the exact average is the default (the production standard); DTFE_EXACT_AVERAGE=0 is the sampled one
    rc_pl=0; env -u DTFE_EXACT_AVERAGE scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_plain.log" 2>&1 || rc_pl=$?
    check "$([ "${rc_pl}" = "0" ] && [ "$(has "${TMP}/h_plain.log" 'RUNNING: .*--exact-average')" = 0 ] \
             && [ "$(has "${TMP}/h_plain.log" 'RUNNING: .*--field density_a velocity_a gradient_a divergence_a shear_a vorticity_a tweb_a vweb_a ')" = 0 ] \
             && echo 0 || echo 1)" "a plain run passes --exact-average and the web fields (the production defaults since 2026-10-13; exit ${rc_pl})"
    DTFE_EXACT_AVERAGE=0 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_mc.log" 2>&1 || true
    check "$(lacks "${TMP}/h_mc.log" '--exact-average')" "DTFE_EXACT_AVERAGE=0 passes no --exact-average (the Monte-Carlo cell mean)"
    # 2026-10-13: the standard binary's auto-tuner reports (it printed 'AUTO-TUNE-REPORT unsupported'), and the driver
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
# the 2D double pair too (2026-10-08): install.sh rebuilds it, so a failed build must be able to bring it back
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
         && [ -x "${LREPO}/PS-DTFE-2d-double" ] && echo 0 || echo 1)" "... the 2D double pair among them (2026-10-08)"
check "$(has "${TMP}/l_hint.log" 'ld: symbol not found')" "build_failed_hint prints the log's tail"
check "$(lacks "${TMP}/l_hint.log" '^line1$')" "... only the last N lines"
check "$(has "${TMP}/l_hint.log" 'make deps-check')" "... and the usual causes"
# install.sh rebuilds the double, 2D and 2D double sets it finds before 'make clean' (2026-10-08; a re-run threw
# them away before). install.sh itself is never run here: its detection block, cut out between 'EXTRA_SETS=()'
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
echo "(M) the run-log contract: a failed, stopped or killed re-run keeps the .runlog of the grids on disk (2026-10-08)."
# Both drivers tee'd straight into <prefix>.runlog, so a re-run that failed early, or was stopped or killed,
# replaced the log of the grids still on disk (the Runs browser then showed them as a failed run with the
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
# the knob since 2026-10-13)
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
echo "(O) the single-instance lock and the preflight of run_dtfe.sh / run_ps_dtfe.sh (2026-10-11)."
# Contract C1: '<prefix>.runlog.lock' is a directory holding the driver's pid while it runs that snapshot. A
# second driver on the same snapshot is refused (two TNG runs together got both SIGKILLed for memory,
# 2026-10-03); a lock whose driver is gone is stale and removed with a message; a stop releases it.
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
    # 2026-10-11: the driver alone SIGKILLed: its binary and tee keep running on the snapshot, so its lock is not
    # stale (removing it started a second binary beside the orphan, the 2026-10-03 double launch)
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
    # the data disk unplugged: one clear message up front, exit 2 (it used to skip every snapshot under a wrong path)
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
echo "(R) run_ps_dtfe.sh: ORDERED_MERGE (1 by default since 2026-10-14), and --resume when a killed run left its checkpoint (2026-10-11)."
# Contracts K1/K2: the binary checkpoints a partitioned run in '<output>.runlog.checkpoint' (under --scratch-dir:
# '<scratch>/<basename>.<hash of the output path>.runlog.checkpoint') and removes it on success, so a folder there
# means a killed run. Against the stand-in binaries of (M), which print their arguments.
RSD="${MDATA}/SIM/snapdir_099"; RCK="${RSD}/ps_output.runlog.checkpoint"
rm -rf "${RCK}" "${RSD}/ps_output.runlog.lock"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_plain.log" FAKE_MODE=ok)
check "$([ "${rc}" = "0" ] && grep -aq '^RUNNING: ' "${TMP}/r_plain.log" && echo 0 || echo 1)" "a plain run goes (${rc})"
# 2026-10-14: the ordered (staged) merge is the default; the driver says it either way
check "$([ "$(lacks "${TMP}/r_plain.log" '^RUNNING: .*--(completion-order-merge|resume)')" = 0 ] \
         && [ "$(has "${TMP}/r_plain.log" '^RUNNING: .*--ordered-merge')" = 0 ] && echo 0 || echo 1)" \
      "... with --ordered-merge (the default since 2026-10-14) and no --resume"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_ordered.log" FAKE_MODE=ok ORDERED_MERGE=1)
check "$(has "${TMP}/r_ordered.log" '^RUNNING: .*--ordered-merge')" "ORDERED_MERGE=1 passes --ordered-merge (${rc})"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_compl.log" FAKE_MODE=ok ORDERED_MERGE=0)
check "$([ "$(has "${TMP}/r_compl.log" '^RUNNING: .*--completion-order-merge')" = 0 ] \
         && [ "$(lacks "${TMP}/r_compl.log" '^RUNNING: .*--ordered-merge')" = 0 ] && echo 0 || echo 1)" \
      "ORDERED_MERGE=0 passes --completion-order-merge, the merge as partitions finish (${rc})"
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
# nothing to resume: the binary refuses '--resume' there, and the folder made every rerun fail (2026-10-11)
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

echo ""
echo "(P) run_ps_pipeline.sh's 'up to date' compares the run settings, not only the file times (2026-10-11)."
# A changed grid or field list printed 'up to date, skipping compute' for every snapshot. The pipeline now asks
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
# 2026-10-11: a slice newer than the plane but of ANOTHER plane size (a 16^2 one against the 8^2 plane: TNG100's
# 4096^2 slices beside an older hires_plane_z8192) was 'up to date', so the compute was skipped and every render failed
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
mv "${SD}/ps_output.runlog" "${SD}/ps_output.runlog.keep"
rc=$(pdrive "${TMP}/p_nolog.log" GRID_SIZE=24 DRY_RUN=1)
check "$(has "${TMP}/p_nolog.log" 'up to date by file times \(no ps_output.runlog to compare the settings with\)')" \
      "no run log: judged by the file times, and said so"
mv "${SD}/ps_output.runlog.keep" "${SD}/ps_output.runlog"

echo ""
echo "(S) production defaults, OUTPUT_PREFIX and a rerun's left-over files (2026-10-13, audit finding 7)."
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
# run_ps_pipeline.sh's GPU default (2026-10-13, integration item 7): PS_GPU was 1 on every build, so a CPU-only build
# was asked for the GPU on every snapshot; unset, it now follows the PS binary's stamp as run_ps_dtfe.sh does (the
# copy (P) made, on SDATA's SIM: a dry run plans the plane and the snapshot)
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
echo "(Q) scripts/thesis_runs.sh: the thesis run recipe, repo-relative, DRY_RUN prints it and runs nothing (2026-10-11)."
TROOT="${TMP}/thesis root/Illustris TNG"; rm -rf "${TMP}/thesis root"
for s in TNG50/TNG50-3-Dark TNG100/TNG100-3-Dark TNG300/TNG300-3-Dark; do
    mkdir -p "${TROOT}/${s}/snapdir_099"; : > "${TROOT}/${s}/snapdir_099/combined_099.hdf5"; : > "${TROOT}/${s}/combined_ics.hdf5"
done
cp scripts/thesis_runs.sh "${MREPO}/scripts/"
rc=0; env DTFE_DATA_ROOT="${TROOT}" DRY_RUN=1 HISTORICAL=1 SCRATCH_DIR="${TMP}/thesis scratch" "${MREPO}/scripts/thesis_runs.sh" tng50z0 caustics \
    >"${TMP}/q_dry.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ ! -e "${TMP}/thesis scratch" ] && echo 0 || echo 1)" "DRY_RUN=1 HISTORICAL=1 tng50z0 caustics exits 0 and creates nothing (${rc})"
check "$(lacks "${TMP}/q_dry.log" '^RUNNING:')" "... no binary ran"
check "$(has "${TMP}/q_dry.log" 'PS-DTFE .*TNG50-3-Dark/snapdir_099/ps_output +--grid +512 .*--scratch-dir .*--field +density_a +velocity_a +gradient_a +divergence_a +shear_a +vorticity_a +dispersion_a +tweb_a +vweb_a .*--ps-gpu .*--ps-volume-weighted .*--sample-points .*TNG50-3-Dark/hires_plane_z2048\.bin +--pts-vel-grad')" \
      "... TNG50 099: 512^3, scratch, the nine thesis fields, GPU, volume-weighted, the 2048^2 plane with velocity gradients"
check "$(has "${TMP}/q_dry.log" 'PS-DTFE .*TNG100-3-Dark/snapdir_099/ps_caustic +--grid +512 .*--field +density_a +tweb_a +--lambda_th.*--ps-caustics')" \
      "... TNG100 099 caustics: its own ps_caustic prefix, density + T-web, --ps-caustics"
# 2026-10-13: the drivers' defaults changed (caustics and volume weighting on); the historical stages keep their own
q_tng50="$(grep -aE -e 'PS-DTFE .*TNG50-3-Dark/snapdir_099/ps_output ' "${TMP}/q_dry.log" || true)"
q_caus="$(grep -aE -e 'PS-DTFE .*/ps_caustic ' "${TMP}/q_dry.log" || true)"
check "$([ -n "${q_tng50}" ] && [ -n "${q_caus}" ] && [[ "${q_tng50}" != *--ps-caustics* ]] && [[ "${q_caus}" != *--ps-volume-weighted* ]] \
         && echo 0 || echo 1)" \
      "... tng50z0 without the caustic classes and caustics with the mass-weighted moments, as they ran (the drivers' new defaults pinned off)"
check "$(grep -aq 'budget 40 GB' "${TMP}/q_dry.log" && grep -aq 'budget 44 GB' "${TMP}/q_dry.log" && echo 0 || echo 1)" "... the budgets are named"
check "$(has "${TMP}/q_dry.log" '^\+ .*plot_void_population\.py --sim TNG50-3-Dark --method ps --smooth-mpc 2\.1625$')" "... the figures are printed, not run"
check "$(has "${TMP}/q_dry.log" '^\+ .*plot_caustic_skeleton\.py --sim TNG100-3-Dark --prefix ps_caustic --snaps 99 --voids$')" "... the caustic skeleton too"
# a GPU build (the stamps of both PS binaries): the pipeline's GPU default follows the build since 2026-10-13
mkdir -p "${MREPO}/o_ps" "${MREPO}/o_ps_d"; : > "${MREPO}/o_ps/.gpu_mode_metal"; : > "${MREPO}/o_ps_d/.gpu_mode_metal"
rc=0; env -u PS_GPU DTFE_DATA_ROOT="${TROOT}" DRY_RUN=1 HISTORICAL=1 "${MREPO}/scripts/thesis_runs.sh" production >"${TMP}/q_prod.log" 2>&1 || rc=$?
rm -rf "${MREPO}/o_ps" "${MREPO}/o_ps_d"
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "DRY_RUN=1 HISTORICAL=1 production exits 0 (${rc})"
# (its pipeline's dry run prints the run_ps_dtfe.sh call, not the binary's line: the recipe itself is read)
check "$(sed -n '/^stage_production()/,/^}/p' scripts/thesis_runs.sh | grep -q 'PS_CAUSTICS=0 .*run_ps_pipeline\.sh\|PS_CAUSTICS=0 \\$' && echo 0 || echo 1)" \
      "... with the caustic classes pinned off, as on 2026-10-03"
check "$(has "${TMP}/q_prod.log" 'FIELDS=density_a velocity_a gradient_a divergence_a shear_a vorticity_a dispersion_a tweb_a vweb_a .*run_ps_dtfe.sh -s TNG100-3-Dark -g 512 -m 99')" \
      "... the pipeline runs TNG100 at 512^3 with the nine fields on the GPU"
check "$(has "${TMP}/q_prod.log" 'make_image_plane.py .*--nu 2048 .*TNG50-3-Dark/hires_plane_z2048')" "... TNG50's plane is 2048^2"
check "$(has "${TMP}/q_prod.log" 'make_image_plane.py .*--nu 4096 .*TNG300-3-Dark/hires_plane_z4096')" "... TNG300's 4096^2"
check "$(has "${TMP}/q_prod.log" 'force: 1')" "... FORCE=1: everything recomputed"
# 2026-10-13: the historical stages (their grids were replaced by ps27's) need HISTORICAL=1; 'all' no longer has them
rc=0; env -u HISTORICAL DTFE_DATA_ROOT="${TROOT}" DRY_RUN=1 "${MREPO}/scripts/thesis_runs.sh" production >"${TMP}/q_hist1.log" 2>&1 || rc=$?
rc2=0; env -u HISTORICAL DTFE_DATA_ROOT="${TROOT}" DRY_RUN=1 "${MREPO}/scripts/thesis_runs.sh" caustics tng50z0 >"${TMP}/q_hist2.log" 2>&1 || rc2=$?
check "$([ "${rc}" = "2" ] && [ "${rc2}" = "2" ] && [ "$(has "${TMP}/q_hist1.log" "'production' is a historical stage.*HISTORICAL=1 runs it anyway|HISTORICAL=1 runs it anyway")" = 0 ] \
         && [ "$(lacks "${TMP}/q_hist2.log" 'DRY RUN|^\+ ')" = 0 ] && echo 0 || echo 1)" \
      "production / tng50z0 without HISTORICAL=1: refused, exit 2, nothing planned (not even the stages beside them) (${rc}, ${rc2})"
rc=0; "${MREPO}/scripts/thesis_runs.sh" >"${TMP}/q_none.log" 2>&1 || rc=$?
rc2=0; "${MREPO}/scripts/thesis_runs.sh" everything >"${TMP}/q_bad.log" 2>&1 || rc2=$?
check "$([ "${rc}" = "2" ] && [ "${rc2}" = "2" ] && echo 0 || echo 1)" "no stage, or an unknown one: exit 2, nothing run (${rc}, ${rc2})"
rc=0; env DTFE_DATA_ROOT="${TMP}/no such disk" DRY_RUN=1 "${MREPO}/scripts/thesis_runs.sh" all >"${TMP}/q_root.log" 2>&1 || rc=$?
check "$([ "${rc}" = "2" ] && echo 0 || echo 1)" "the data disk missing: exit 2 up front (${rc})"
# The voids stage (2026-10-11): every void figure of each box, script by script, with the agreed definitions. PY is a
# stand-in that logs the environment and arguments it was started with (and fails the fate census), so DRY_RUN is
# seen to start nothing and a real run to hand each figure its gate and to go on after a failure. The caller's
# DTFE_VOID_GATE=none must not reach the gated figures.
: > "${TROOT}/TNG100/TNG100-3-Dark/snapdir_099/ps_caustic.causticClass"
cat > "${TMP}/q_py.sh" <<'PYEOF'
#!/usr/bin/env bash
echo "PYSTUB gate=${DTFE_VOID_GATE:-unset} sim=${DTFE_SIM:-unset} $*" >> "${Q_PY_LOG}"
case "$*" in *plot_void_fate.py*) exit 1 ;; esac
exit 0
PYEOF
chmod +x "${TMP}/q_py.sh"
qv() { env DTFE_DATA_ROOT="${TROOT}" PY="${TMP}/q_py.sh" DTFE_VOID_GATE=none DTFE_FIGURES_ROOT="${TMP}/q_figs" \
           SCRATCH_DIR="${TMP}/q_scratch" "$@"; }
rc=0; qv DRY_RUN=1 Q_PY_LOG="${TMP}/q_py_dry.log" "${MREPO}/scripts/thesis_runs.sh" voids >"${TMP}/q_voids.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ ! -e "${TMP}/q_py_dry.log" ] && [ ! -e "${TMP}/q_figs" ] && [ ! -e "${TMP}/q_scratch" ] \
         && echo 0 || echo 1)" "DRY_RUN=1 voids exits 0 and starts no figure script, creates nothing (${rc})"
# qall LABEL REGEX: REGEX (with @BOX@ for the box) is a printed command line for each of the three boxes
qall() {
    local sim r=0
    for sim in TNG50-3-Dark TNG100-3-Dark TNG300-3-Dark; do
        grep -aqE -e "^\+ ${2//@BOX@/${sim}}\$" "${TMP}/q_voids.log" || { r=1; echo "    missing for ${sim}: ${2//@BOX@/${sim}}"; }
    done
    check "${r}" "$1"
}
QPY='.*q_py\.sh python/plot'
qall "... population twice per box: the shapes gated (despite the caller's none), the sizes ungated" \
     "env DTFE_SIM=@BOX@ DTFE_VOID_GATE=ellipsoid ${QPY}/plot_void_population\.py --sim @BOX@ --method ps"
qall "... ... (the ungated one)" "env DTFE_SIM=@BOX@ DTFE_VOID_GATE=none ${QPY}/plot_void_population\.py --sim @BOX@ --method ps"
qall "... profiles at z=0 and their evolution ungated, R_v the first crossing" \
     "env DTFE_SIM=@BOX@ DTFE_VOID_GATE=none ${QPY}/plot_void_profiles\.py --sim @BOX@ --method ps --snaps 99"
qall "... ... (the evolution)" \
     "env DTFE_SIM=@BOX@ DTFE_VOID_GATE=none ${QPY}/plot_void_profile_evolution\.py --sim @BOX@ --method ps --rv-definition crossing"
qall "... environment ungated, the velocity criterion, the exclusive particle-tracking set" \
     "env DTFE_SIM=@BOX@ DTFE_VOID_GATE=none ${QPY}/plot_void_environment\.py --sim @BOX@ --method ps --criterion velocity --r-v-definition crossing --tracking-tag _particles_exclusive"
# convergence ungated (2026-10-11 review: its rows are R_v statistics, and it reads the ungated cross-box files now)
qall "... convergence ungated, at the cross-box sigma" \
     "env DTFE_SIM=@BOX@ DTFE_VOID_GATE=none ${QPY}/plot_void_convergence\.py --sim @BOX@ --method ps --crossbox-smooth-mpc 2\.1625"
check "$(lacks "${TMP}/q_voids.log" 'DTFE_VOID_GATE=ellipsoid .*plot_void_convergence')" "... ... and never gated"
qall "... fate census --exclusive, ungated" "env DTFE_SIM=@BOX@ DTFE_VOID_GATE=none ${QPY}/plot_void_fate\.py --sim @BOX@ --method ps --exclusive"
qall "... tracking by exclusive particle links, the census's gate" \
     "env DTFE_SIM=@BOX@ DTFE_VOID_GATE=none ${QPY}/plot_void_tracking\.py --sim @BOX@ --method ps --link particles --exclusive"
# the size function at fixed R (2026-10-11): ungated, the default 10 / 5 / 3 cells for TNG50 and TNG100, TNG300 at
# 2.1625 Mpc and 3 / 2 cells (no common range at 10 / 5 / 3 cells there), and the cross-box sigma alone for TNG100 and
# TNG300 only (TNG50 not measured there); 35 figure commands in all for the three boxes (27 before the fixed-R ones,
# 32 before the environment's second call after the tracking, 2026-10-11)
QFR="${QPY}/plot_void_population\.py --sim"
check "$([ "$(has "${TMP}/q_voids.log" "^\+ env DTFE_SIM=TNG50-3-Dark DTFE_VOID_GATE=none ${QFR} TNG50-3-Dark --method ps --size-fixed-r\$")" = 0 ] \
         && [ "$(has "${TMP}/q_voids.log" "^\+ env DTFE_SIM=TNG100-3-Dark DTFE_VOID_GATE=none ${QFR} TNG100-3-Dark --method ps --size-fixed-r\$")" = 0 ] \
         && [ "$(has "${TMP}/q_voids.log" "^\+ env DTFE_SIM=TNG300-3-Dark DTFE_VOID_GATE=none ${QFR} TNG300-3-Dark --method ps --size-fixed-r --fixed-r-smooth 3 2 --fixed-r-smooth-mpc 2\.1625\$")" = 0 ] \
         && echo 0 || echo 1)" "... the size function at fixed R per box, ungated: 10 / 5 / 3 cells, TNG300 at 2.1625 Mpc + 3 / 2 cells"
check "$([ "$(grep -acE "^\+ env DTFE_SIM=(TNG100|TNG300)-3-Dark DTFE_VOID_GATE=none ${QFR} (TNG100|TNG300)-3-Dark --method ps --size-fixed-r --fixed-r-smooth-mpc 2\.1625\$" "${TMP}/q_voids.log")" = "2" ] \
         && [ "$(grep -ac 'size-fixed-r' "${TMP}/q_voids.log")" = "5" ] \
         && [ "$(lacks "${TMP}/q_voids.log" 'DTFE_VOID_GATE=ellipsoid .*--size-fixed-r')" = 0 ] \
         && [ "$(grep -ac '^+ ' "${TMP}/q_voids.log")" = "35" ] && echo 0 || echo 1)" \
      "... and at the cross-box sigma alone for TNG100 and TNG300 only, never gated; 35 figure commands in all"
check "$([ "$(grep -acE "^\+ env -u DTFE_SIM DTFE_VOID_GATE=(ellipsoid|none) ${QPY}/plot_void_crossbox\.py --sims TNG50-3-Dark TNG100-3-Dark TNG300-3-Dark --method ps --smooth-mpc 2\.1625$" "${TMP}/q_voids.log")" = "2" ] \
         && [ "$(has "${TMP}/q_voids.log" 'DTFE_VOID_GATE=none .*plot_void_crossbox')" = 0 ] && echo 0 || echo 1)" \
      "... the cross-box figure once per gate, the three boxes at 2.1625 Mpc"
check "$([ "$(grep -ac 'plot_caustic_skeleton' "${TMP}/q_voids.log")" = "1" ] \
         && [ "$(has "${TMP}/q_voids.log" '^\+ env DTFE_SIM=TNG100-3-Dark DTFE_VOID_GATE=ellipsoid .*plot_caustic_skeleton\.py --sim TNG100-3-Dark --prefix ps_caustic --snaps 99 --voids$')" = 0 ] \
         && [ "$(has "${TMP}/q_voids.log" 'TNG300-3-Dark caustic skeleton: no ps_caustic grids, skipped')" = 0 ] && echo 0 || echo 1)" \
      "... the caustic skeleton only where ps_caustic grids exist (TNG100 099)"
# the order: the cheap per-box figures (the size function at fixed R among them), the cross-box pair, convergence
# (reads the cross-box files), fate + tracking last, each box's environment again after its tracking
qline() { grep -anE "^\+ .*$1" "${TMP}/q_voids.log" | cut -d: -f1 | sed -n "$2"; }
check "$([ "$(qline 'plot_void_environment' '3p')" -lt "$(qline 'plot_void_crossbox' '1p')" ] \
         && [ "$(qline 'size-fixed-r' '$p')" -lt "$(qline 'plot_void_crossbox' '1p')" ] \
         && [ "$(qline 'plot_void_crossbox' '$p')" -lt "$(qline 'plot_void_convergence' '1p')" ] \
         && [ "$(qline 'plot_void_convergence' '$p')" -lt "$(qline 'plot_void_fate' '1p')" ] \
         && [ "$(qline 'plot_void_fate' '$p')" -lt "$(qline 'plot_void_tracking' '$p')" ] && echo 0 || echo 1)" \
      "... in the order cheap figures (fixed-R size function too) < cross-box < convergence < fate census + tracking"
# 2026-10-11: the environment reads the tracking's files for its tracking panel, but was drawn only before the
# tracking had run (a first run's figure lacked the panel): each box's is drawn again right after its tracking
qenv() {
    local sim r=0 t e
    for sim in TNG50-3-Dark TNG100-3-Dark TNG300-3-Dark; do       # numbered among the printed commands only
        t="$(grep -a '^+ ' "${TMP}/q_voids.log" | grep -nE "plot_void_tracking\.py --sim ${sim} " | cut -d: -f1 | tail -1)"
        e="$(grep -a '^+ ' "${TMP}/q_voids.log" | grep -nE "plot_void_environment\.py --sim ${sim} " | cut -d: -f1 | tail -1)"
        [ -n "${t}" ] && [ -n "${e}" ] && [ "${e}" -eq $((t + 1)) ] \
            && [ "$(grep -acE "^\+ .*plot_void_environment\.py --sim ${sim} " "${TMP}/q_voids.log")" = "2" ] || r=1
    done
    echo "${r}"
}
check "$(qenv)" "... each box's environment drawn twice, the second time right after its tracking (the tracking panel's files)"
rc=0; qv DRY_RUN=1 SIMS=TNG100-3-Dark Q_PY_LOG="${TMP}/q_py_dry.log" "${MREPO}/scripts/thesis_runs.sh" voids >"${TMP}/q_voids1.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ "$(grep -ac '^+ ' "${TMP}/q_voids1.log")" = "12" ] && [ "$(lacks "${TMP}/q_voids1.log" 'TNG50-3-Dark|TNG300-3-Dark')" = 0 ] \
         && [ "$(has "${TMP}/q_voids1.log" 'cross-box: one box \(TNG100-3-Dark\), nothing to compare: skipped')" = 0 ] && echo 0 || echo 1)" \
      "SIMS=TNG100-3-Dark: only that box's twelve figures (both fixed-R ones, the environment twice), no cross-box (${rc})"
rc=0; qv Q_PY_LOG="${TMP}/q_py_bad.log" "${MREPO}/scripts/thesis_runs.sh" voids everything >"${TMP}/q_voidsbad.log" 2>&1 || rc=$?
check "$([ "${rc}" = "2" ] && [ ! -e "${TMP}/q_py_bad.log" ] && echo 0 || echo 1)" "voids beside an unknown stage: exit 2, no figure started (${rc})"
# a real run against the stand-in (two boxes): each figure gets its gate, the failed fate census is reported at the
# end with exit 1, and the tracking after it still runs
rc=0; qv SIMS="TNG50-3-Dark TNG100-3-Dark" Q_PY_LOG="${TMP}/q_py_run.log" "${MREPO}/scripts/thesis_runs.sh" voids >"${TMP}/q_voidsrun.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ "$(grep -ac '^PYSTUB' "${TMP}/q_py_run.log")" = "24" ] \
         && [ "$(has "${TMP}/q_py_run.log" '^PYSTUB gate=ellipsoid sim=TNG50-3-Dark python/plot/plot_void_population\.py')" = 0 ] \
         && [ "$(has "${TMP}/q_py_run.log" '^PYSTUB gate=none sim=unset python/plot/plot_void_crossbox\.py --sims TNG50-3-Dark TNG100-3-Dark ')" = 0 ] \
         && [ "$(has "${TMP}/q_py_run.log" '^PYSTUB gate=none sim=TNG100-3-Dark python/plot/plot_void_tracking\.py')" = 0 ] \
         && [ "$(has "${TMP}/q_py_run.log" '^PYSTUB gate=none sim=TNG50-3-Dark python/plot/plot_void_convergence\.py')" = 0 ] \
         && [ "$(has "${TMP}/q_py_run.log" '^PYSTUB gate=none sim=TNG50-3-Dark python/plot/plot_void_population\.py --sim TNG50-3-Dark --method ps --size-fixed-r$')" = 0 ] \
         && [ "$(has "${TMP}/q_py_run.log" '^PYSTUB gate=none sim=TNG100-3-Dark python/plot/plot_void_population\.py --sim TNG100-3-Dark --method ps --size-fixed-r --fixed-r-smooth-mpc 2\.1625$')" = 0 ] \
         && echo 0 || echo 1)" "a run: every figure started with its gate and box, the ones after a failure too (${rc})"
check "$(has "${TMP}/q_voidsrun.log" 'voids: TNG100-3-Dark tracking: exit 0 \([0-9]+ s\)')" "... each figure timed"
check "$(grep -aqx 'failed void figure: TNG50-3-Dark fate census (exit 1)' "${TMP}/q_voidsrun.log" \
         && grep -aqx 'failed void figure: TNG100-3-Dark fate census (exit 1)' "${TMP}/q_voidsrun.log" \
         && [ "$(grep -ac '^failed void figure' "${TMP}/q_voidsrun.log")" = "2" ] && echo 0 || echo 1)" \
      "... the failed figures listed at the end, exit 1"

# The overnight stages (2026-10-12): dtfe (standard DTFE on TNG100 z = 0, 1, 5), caustics2 (ps_caustic at z = 1, 2 with
# the z = 0 run's settings pinned, but the binary's default Lagrangian padding: since the padding fix it changes the
# grids only at rounding, and it is ~6x faster), refresh (figure repairs; the old untagged skeleton files MOVED, never deleted),
# prune. The caller's shell carries knobs that would change the recipe; none may reach the planned commands.
for s in 050 033 017; do
    mkdir -p "${TROOT}/TNG100/TNG100-3-Dark/snapdir_${s}"; : > "${TROOT}/TNG100/TNG100-3-Dark/snapdir_${s}/combined_${s}.hdf5"
done
QFIG="${TMP}/q_figs/caustic_skeleton/TNG100-3-Dark"; QTH="${TMP}/q_thesis/caustic_skeleton/TNG100-3-Dark"
# seed: the untagged set of 2026-10-08 (made with --prefix ps_caustic) beside the tagged one, on the figure root and in
# the thesis tree (which mirrors the png only)
q_seed() {
    rm -rf "${TMP}/q_figs" "${TMP}/q_thesis"; mkdir -p "${QFIG}" "${QTH}"
    printf '{\n "prefix": "ps_caustic.",\n "snapshot": 99\n}\n' > "${QFIG}/caustic_skeleton_z0.00.json"
    for f in caustic_skeleton_z0.00.txt caustic_skeleton_z0.00.png caustic_skeleton_z0.00_voids.npz \
             caustic_skeleton_z0.00_ps_caustic.json caustic_skeleton_z0.00_ps_caustic.png \
             caustic_skeleton_z0.00_ps_caustic.txt caustic_skeleton_z0.00_ps_caustic_voids.npz; do
        echo "${f}" > "${QFIG}/${f}"
    done
    echo old > "${QTH}/caustic_skeleton_z0.00.png"; echo new > "${QTH}/caustic_skeleton_z0.00_ps_caustic.png"
}
q_seed
# the caller's knobs (an env prefix inside qv's: qv runs env, which cannot call a shell function)
QO=(env -u DTFE_NO_MIRROR THESIS_FIGURES="${TMP}/q_thesis" PS_LAG_PADDING=8 PS_PARALLEL_TRI=1 PS_VOLUME_WEIGHTED=1
    PS_EXACT=1 AVG_SUBSAMPLES=1 PADDING=40 FIELDS="density_a vweb_a" SAMPLE_POINTS=/no/such/plane.bin
    DTFE_EXACT_AVERAGE=0 PARTITION="2 2 2" LAMBDA_TH=0.7 OUTPUT_PREFIX=leaky LAGRANGIAN_INPUT= DTFE_METAL=0 PS_METAL=0
    PS_CAUSTICS=0 MAX_CONCURRENT=3 TESS_CACHE=/no/such/cache AUTO_TUNE_REPORT=1)
rm -rf "${TMP}/q_scratch"           # the voids run above made it; a dry run must not
rc=0; qv "${QO[@]}" DRY_RUN=1 Q_PY_LOG="${TMP}/q_py_on.log" "${MREPO}/scripts/thesis_runs.sh" overnight >"${TMP}/q_on.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ ! -e "${TMP}/q_py_on.log" ] && [ ! -e "${TMP}/q_scratch" ] && [ ! -e "${QFIG}/old_untagged" ] \
         && [ ! -e "${QTH}/old_untagged" ] && [ -e "${QFIG}/caustic_skeleton_z0.00.png" ] && [ -e "${QTH}/caustic_skeleton_z0.00.png" ] \
         && [ "$(lacks "${TMP}/q_on.log" '^RUNNING:')" = 0 ] && echo 0 || echo 1)" \
      "DRY_RUN=1 overnight exits 0: no binary, no figure script, no scratch / lock, nothing moved (${rc})"
# the binaries' planned lines (the drivers' DRY RUN): one per snapshot, in the recipe's order
qbin() { grep -aE -e "^ +\./$1 " "${TMP}/q_on.log"; }
check "$([ "$(qbin DTFE | grep -cE -e 'TNG100-3-Dark/snapdir_(099|050|017)/output +--grid +512 +--padding +15 +--periodic +--partition +3 +3 +3 +--max-concurrent +10 +--field +density_a +velocity_a +gradient_a +divergence_a +shear_a +vorticity_a +tweb_a +vweb_a +--lambda_th +0\.3 +--gpu +--exact-average$')" = "3" ] \
         && [ "$(qbin DTFE | grep -oE 'snapdir_0[0-9]{2}/output ' | tr -d '\n')" = 'snapdir_099/output snapdir_050/output snapdir_017/output ' ] \
         && [ "$(lacks "${TMP}/q_on.log" 'leaky|--partition +2 +2 +2|sample-points|--padding +40|--lambda_th +0\.7|--max-concurrent +3|tessellation-cache|auto-tune-report')" = 0 ] \
         && [ "$(qbin DTFE | grep -c -e '--scratch-dir')" = "0" ] && echo 0 || echo 1)" \
      "... dtfe: standard DTFE on TNG100 099, 050, 017 at 512^3 with dtfe27's settings (padding 15, 3x3x3 ten at a time, every field with the T-web / V-web at lambda_th 0.3, GPU, the EXACT cell average, no scratch dir as on 2026-10-12); the caller's knobs ignored"
check "$([ "$(qbin PS-DTFE | grep -cE -e 'TNG100-3-Dark/snapdir_0(50|33)/ps_caustic +--grid +512 +--padding +50 +--periodic +--scratch-dir .* +--avg-subsamples +3 +--input +105 +--MpcUnit +1000 +--field +density_a +tweb_a +--lambda_th +0\.3 +--ps-gpu +--ps-vertex-mass +--ps-caustics +--lagrangianInput .*TNG100-3-Dark/combined_ics\.hdf5$')" = "2" ] \
         && [ "$(qbin PS-DTFE | grep -oE 'snapdir_0[0-9]{2}/ps_caustic ' | tr -d '\n')" = 'snapdir_050/ps_caustic snapdir_033/ps_caustic ' ] \
         && [ "$(lacks "${TMP}/q_on.log" 'ps-lag-padding|parallel-triangulation|ps-volume-weighted|ps-exact-deposit')" = 0 ] && echo 0 || echo 1)" \
      "... caustics2: TNG100 050 then 033 under ps_caustic with the z = 0 run's exact flags (parallel triangulation, volume weighting, exact deposit pinned off) and the binary's default lag padding (no flag: the caller's PS_LAG_PADDING=8 does not reach it)"
QPYR='.*q_py\.sh python'
check "$([ "$(has "${TMP}/q_on.log" "^\+ env DTFE_SIM=TNG100-3-Dark ${QPYR}/plot/plot_web_streams\.py --sim TNG100-3-Dark --snaps 99 --web tweb\$")" = 0 ] \
         && [ "$(has "${TMP}/q_on.log" "^\+ env DTFE_SIM=TNG100-3-Dark DTFE_VOID_GATE=ellipsoid ${QPYR}/analyze\.py export\$")" = 0 ] \
         && [ "$(has "${TMP}/q_on.log" "^\+ env DTFE_SIM=TNG100-3-Dark DTFE_VOID_GATE=none ${QPYR}/plot/plot_void_convergence\.py --sim TNG100-3-Dark --method ps --crossbox-smooth-mpc 2\.1625 --z 0 1 5\$")" = 0 ] \
         && [ "$(has "${TMP}/q_on.log" "^\+ env DTFE_SIM=TNG100-3-Dark DTFE_VOID_GATE=ellipsoid ${QPYR}/plot/plot_caustic_skeleton\.py --sim TNG100-3-Dark --prefix ps_caustic --all-caustic --voids\$")" = 0 ] \
         && echo 0 || echo 1)" \
      "... refresh: the default grids' web streams at z=0, the export (production gate), convergence at z = 0 1 5 ungated, the skeleton of every ps_caustic snapshot"
check "$([ "$(grep -acE '^\+ mv .*caustic_skeleton_z0\.00(\.json|\.txt|\.png|_voids\.npz) .*/old_untagged/' "${TMP}/q_on.log")" = "5" ] \
         && [ "$(lacks "${TMP}/q_on.log" '^\+ mv .*_ps_caustic|^\+ rm |rm -')" = 0 ] && echo 0 || echo 1)" \
      "... the move printed for the four untagged files and the thesis png, never a tagged file, never a removal"
check "$([ "$(has "${TMP}/q_on.log" "^\+ ${QPYR}/analyze\.py cache --prune --keep-cells 3 2\$")" = 0 ] \
         && [ "$(has "${TMP}/q_on.log" "^\+ ${QPYR}/analyze\.py cache --prune --yes --keep-cells 3 2\$")" = 0 ] && echo 0 || echo 1)" \
      "... prune: the listing, then --yes, keeping the fixed-R smoothings (3 and 2 cells)"
# the order: every dtfe run < every caustics2 run < the refresh figures < the move < prune last
qon() { grep -anE -e "$1" "${TMP}/q_on.log" | cut -d: -f1 | sed -n "$2"; }
check "$([ "$(qon '^ +\./DTFE ' '$p')" -lt "$(qon '^ +\./PS-DTFE ' '1p')" ] \
         && [ "$(qon '^ +\./PS-DTFE ' '$p')" -lt "$(qon 'plot_web_streams' '1p')" ] \
         && [ "$(qon 'plot_web_streams' '1p')" -lt "$(qon 'analyze\.py export' '1p')" ] \
         && [ "$(qon 'analyze\.py export' '1p')" -lt "$(qon 'plot_void_convergence' '1p')" ] \
         && [ "$(qon 'plot_void_convergence' '1p')" -lt "$(qon 'plot_caustic_skeleton' '1p')" ] \
         && [ "$(qon 'plot_caustic_skeleton' '1p')" -lt "$(qon '^\+ mv ' '1p')" ] \
         && [ "$(qon '^\+ mv ' '$p')" -lt "$(qon 'cache --prune' '1p')" ] \
         && [ "$(grep -acE '^=== .* (dtfe|caustics2|refresh|prune): expected ' "${TMP}/q_on.log")" = "4" ] && echo 0 || echo 1)" \
      "... in the order dtfe < caustics2 < refresh (web streams, export, convergence, skeleton, move) < prune, each stage's expected cost printed"
rc=0; qv "${QO[@]}" DRY_RUN=1 Q_PY_LOG="${TMP}/q_py_on.log" "${MREPO}/scripts/thesis_runs.sh" dtfe prune overnightx >"${TMP}/q_onbad.log" 2>&1 || rc=$?
check "$([ "${rc}" = "2" ] && [ "$(lacks "${TMP}/q_onbad.log" 'DTFE processing')" = 0 ] && echo 0 || echo 1)" "an unknown stage beside the new ones: exit 2, nothing planned (${rc})"
# finished snapshots are skipped (a morning restart), FORCE=1 reruns them; a run log alone is not a finished run, nor
# (2026-10-13) grids made with other settings: the sampled 2026-10-12 set (padding 20, no --exact-average) is rerun
T1="${TROOT}/TNG100/TNG100-3-Dark"
: > "${T1}/snapdir_099/output.runlog"
echo x > "${T1}/snapdir_050/output.a_den"
echo 'RUNNING: ./DTFE in out --grid 512 --padding 15 --periodic --field density_a --lambda_th 0.3 --gpu --exact-average ' > "${T1}/snapdir_050/output.runlog"
echo x > "${T1}/snapdir_017/output.a_den"
echo 'RUNNING: ./DTFE in out --grid 512 --padding 20 --periodic --field density_a --lambda_th 0.3 --gpu ' > "${T1}/snapdir_017/output.runlog"
echo x > "${TROOT}/TNG100/TNG100-3-Dark/snapdir_033/ps_caustic.causticClass"; : > "${TROOT}/TNG100/TNG100-3-Dark/snapdir_033/ps_caustic.runlog"
rc=0; qv DRY_RUN=1 "${MREPO}/scripts/thesis_runs.sh" dtfe caustics2 >"${TMP}/q_skip.log" 2>&1 || rc=$?
rc2=0; qv DRY_RUN=1 FORCE=1 "${MREPO}/scripts/thesis_runs.sh" dtfe caustics2 >"${TMP}/q_force.log" 2>&1 || rc2=$?
check "$([ "${rc}" = "0" ] && [ "$(grep -acE '^ +\./DTFE ' "${TMP}/q_skip.log")" = "2" ] && [ "$(grep -acE '^ +\./PS-DTFE ' "${TMP}/q_skip.log")" = "1" ] \
         && [ "$(has "${TMP}/q_skip.log" 'dtfe: TNG100-3-Dark 050: output\.\* made with the exact average at padding 15 already: skipped')" = 0 ] \
         && [ "$(has "${TMP}/q_skip.log" 'snapdir_017/output ')" = 0 ] \
         && [ "$(has "${TMP}/q_skip.log" '^\+ mv -n .*snapdir_017/output\.\* .*_backup/output_[0-9-]+/TNG100/TNG100-3-Dark/snapdir_017/ \(2 files\)$')" = 0 ] \
         && [ "$(has "${TMP}/q_skip.log" 'caustics2: TNG100-3-Dark 033: .*skipped')" = 0 ] \
         && [ "$(has "${TMP}/q_skip.log" 'snapdir_099/output ')" = 0 ] \
         && [ "${rc2}" = "0" ] && [ "$(grep -acE '^ +\./DTFE ' "${TMP}/q_force.log")" = "3" ] \
         && [ "$(grep -acE '^ +\./PS-DTFE ' "${TMP}/q_force.log")" = "2" ] && echo 0 || echo 1)" \
      "finished snapshots skipped (grids + a run log with dtfe27's settings; a run log alone, or the sampled padding-20 set, is not one: its backup move printed), FORCE=1 plans them again (${rc}, ${rc2})"
rm -f "${T1}/snapdir_099/output.runlog" "${T1}/snapdir_050/output."* "${T1}/snapdir_017/output."* "${T1}/snapdir_033/ps_caustic."*
# refresh for real against the stand-in PY: each step gets its gate (the caller says none), the untagged files are
# MOVED into old_untagged (figure root + thesis tree), the tagged ones stay, no file is lost, the lock is released
qcount() { find "${TMP}/q_figs" "${TMP}/q_thesis" -type f | wc -l | tr -d ' '; }
n_before="$(qcount)"
rc=0; qv "${QO[@]}" Q_PY_LOG="${TMP}/q_py_refresh.log" "${MREPO}/scripts/thesis_runs.sh" refresh >"${TMP}/q_refresh.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ "$(qcount)" = "${n_before}" ] \
         && [ -f "${QFIG}/old_untagged/caustic_skeleton_z0.00.json" ] && [ -f "${QFIG}/old_untagged/caustic_skeleton_z0.00.txt" ] \
         && [ -f "${QFIG}/old_untagged/caustic_skeleton_z0.00.png" ] && [ -f "${QFIG}/old_untagged/caustic_skeleton_z0.00_voids.npz" ] \
         && [ "$(cat "${QTH}/old_untagged/caustic_skeleton_z0.00.png")" = "old" ] \
         && [ ! -e "${QFIG}/caustic_skeleton_z0.00.png" ] && [ ! -e "${QTH}/caustic_skeleton_z0.00.png" ] \
         && [ "$(ls "${QFIG}" | grep -c '_ps_caustic')" = "4" ] && [ "$(cat "${QTH}/caustic_skeleton_z0.00_ps_caustic.png")" = "new" ] \
         && [ ! -e "${TMP}/q_scratch/thesis_runs.lock" ] && echo 0 || echo 1)" \
      "refresh: the untagged z=0 skeleton files MOVED to old_untagged (figure root and thesis tree), the tagged ones untouched, no file lost, lock released (${rc})"
check "$([ "$(grep -ac '^PYSTUB' "${TMP}/q_py_refresh.log")" = "4" ] \
         && [ "$(has "${TMP}/q_py_refresh.log" '^PYSTUB gate=ellipsoid sim=TNG100-3-Dark python/analyze\.py export$')" = 0 ] \
         && [ "$(has "${TMP}/q_py_refresh.log" '^PYSTUB gate=none sim=TNG100-3-Dark python/plot/plot_void_convergence\.py .*--z 0 1 5$')" = 0 ] \
         && [ "$(has "${TMP}/q_py_refresh.log" '^PYSTUB gate=ellipsoid sim=TNG100-3-Dark python/plot/plot_caustic_skeleton\.py .*--all-caustic --voids$')" = 0 ] \
         && [ "$(has "${TMP}/q_refresh.log" 'refresh: .*convergence .*: exit 0 \([0-9]+ s\)')" = 0 ] && echo 0 || echo 1)" \
      "... its four python steps started with their gate and box, each timed"
rc=0; qv "${QO[@]}" Q_PY_LOG="${TMP}/q_py_refresh2.log" "${MREPO}/scripts/thesis_runs.sh" refresh >"${TMP}/q_refresh2.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ "$(qcount)" = "${n_before}" ] && [ "$(has "${TMP}/q_refresh2.log" 'nothing to move')" = 0 ] && echo 0 || echo 1)" \
      "... a second refresh: nothing left to move, still exit 0 (${rc})"
# the guards of the move: no tagged png beside them -> they stay (exit 1); an occupied destination is never
# overwritten; DTFE_NO_MIRROR leaves the thesis tree alone
q_seed; rm -f "${QFIG}/caustic_skeleton_z0.00_ps_caustic.png"
rc=0; qv "${QO[@]}" Q_PY_LOG="${TMP}/q_py_g.log" "${MREPO}/scripts/thesis_runs.sh" refresh >"${TMP}/q_guard1.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ -e "${QFIG}/caustic_skeleton_z0.00.png" ] && [ ! -e "${QFIG}/old_untagged" ] \
         && [ "$(has "${TMP}/q_guard1.log" '^failed step: refresh: move of the untagged skeleton files in .*q_figs')" = 0 ] && echo 0 || echo 1)" \
      "... without the tagged png beside them the untagged files stay, and it is a listed failure, exit 1 (${rc})"
q_seed; mkdir -p "${QFIG}/old_untagged"; echo keep > "${QFIG}/old_untagged/caustic_skeleton_z0.00.txt"
rc=0; qv "${QO[@]}" DTFE_NO_MIRROR=1 Q_PY_LOG="${TMP}/q_py_g.log" "${MREPO}/scripts/thesis_runs.sh" refresh >"${TMP}/q_guard2.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ "$(cat "${QFIG}/old_untagged/caustic_skeleton_z0.00.txt")" = "keep" ] \
         && [ -e "${QFIG}/caustic_skeleton_z0.00.txt" ] && [ -e "${QFIG}/old_untagged/caustic_skeleton_z0.00.png" ] \
         && [ -e "${QTH}/caustic_skeleton_z0.00.png" ] && [ ! -e "${QTH}/old_untagged" ] \
         && [ "$(has "${TMP}/q_guard2.log" 'NOT moved \(nothing overwritten\)')" = 0 ] && echo 0 || echo 1)" \
      "... an existing old_untagged file is never overwritten (that one stays, exit 1); DTFE_NO_MIRROR=1 leaves the thesis tree alone (${rc})"
# one recipe at a time: a live thesis_runs.sh holding the lock refuses a second (exit 2, nothing run); a stale lock
# (its pid gone) is removed with a note
mkdir -p "${TMP}/q_fake" "${TMP}/q_scratch"
printf '#!/usr/bin/env bash\nsleep 30\n' > "${TMP}/q_fake/thesis_runs.sh"; chmod +x "${TMP}/q_fake/thesis_runs.sh"
"${TMP}/q_fake/thesis_runs.sh" & qpid=$!
mkdir -p "${TMP}/q_scratch/thesis_runs.lock"; echo "${qpid}" > "${TMP}/q_scratch/thesis_runs.lock/pid"
rc=0; qv Q_PY_LOG="${TMP}/q_py_lock.log" "${MREPO}/scripts/thesis_runs.sh" prune >"${TMP}/q_lock.log" 2>&1 || rc=$?
kill "${qpid}" 2>/dev/null || true; wait "${qpid}" 2>/dev/null || true
check "$([ "${rc}" = "2" ] && [ ! -e "${TMP}/q_py_lock.log" ] && [ "$(has "${TMP}/q_lock.log" "another thesis_runs\.sh \(pid ${qpid}\) is running")" = 0 ] \
         && [ "$(cat "${TMP}/q_scratch/thesis_runs.lock/pid")" = "${qpid}" ] && echo 0 || echo 1)" \
      "a second recipe while one runs: refused, exit 2, nothing started, the other's lock kept (${rc})"
rc=0; qv Q_PY_LOG="${TMP}/q_py_lock.log" "${MREPO}/scripts/thesis_runs.sh" prune >"${TMP}/q_lock2.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ "$(has "${TMP}/q_lock2.log" 'removing a stale recipe lock')" = 0 ] && [ ! -e "${TMP}/q_scratch/thesis_runs.lock" ] \
         && [ "$(grep -ac '^PYSTUB' "${TMP}/q_py_lock.log")" = "2" ] \
         && [ "$(sed -n 1p "${TMP}/q_py_lock.log" | grep -cE 'python/analyze\.py cache --prune --keep-cells 3 2$')" = "1" ] \
         && [ "$(sed -n 2p "${TMP}/q_py_lock.log" | grep -cE 'python/analyze\.py cache --prune --yes --keep-cells 3 2$')" = "1" ] \
         && echo 0 || echo 1)" "... a stale lock (its pid gone) removed with a note; prune lists, then deletes (${rc})"
# a DTFE / PS-DTFE run already going: the heavy stages refuse to start (exit 1, listed) once HEAVY_WAIT_SEC (0 here)
# is over, nothing else is refused
mkdir -p "${TMP}/q_bin"; ln -sf /bin/sleep "${TMP}/q_bin/PS-DTFE"     # a link: macOS kills a COPIED system binary
"${TMP}/q_bin/PS-DTFE" 30 & qpid=$!
rc=0; qv HEAVY_WAIT_SEC=0 Q_PY_LOG="${TMP}/q_py_busy.log" "${MREPO}/scripts/thesis_runs.sh" dtfe caustics2 ps27 dtfe27 >"${TMP}/q_busy.log" 2>&1 || rc=$?
kill "${qpid}" 2>/dev/null || true; wait "${qpid}" 2>/dev/null || true
check "$([ "${rc}" = "1" ] && [ "$(lacks "${TMP}/q_busy.log" 'DTFE processing|\+ mv ')" = 0 ] \
         && [ "$(has "${TMP}/q_busy.log" "^ +${qpid} .*q_bin/PS-DTFE 30")" = 0 ] \
         && [ "$(grep -ac '^failed step: .*not started, another DTFE / PS-DTFE run was going' "${TMP}/q_busy.log")" = "4" ] && echo 0 || echo 1)" \
      "dtfe / caustics2 / ps27 / dtfe27 while another PS-DTFE runs: refused (named), listed, nothing moved, exit 1 (${rc})"
# 2026-10-13: the guard matches the EXECUTABLE, so a program whose path has a space after 'DTFE' (the launcher app,
# '/Applications/DTFE Launcher.app/Contents/MacOS/DTFE Launcher') is not a DTFE run -- the first word of its command
# line, '/Applications/DTFE', held dtfe27 for 30 min while the launcher was open; a real binary still is
mkdir -p "${TMP}/q_app/DTFE Launcher.app/Contents/MacOS"; ln -sf /bin/sleep "${TMP}/q_app/DTFE Launcher.app/Contents/MacOS/DTFE Launcher"
"${TMP}/q_app/DTFE Launcher.app/Contents/MacOS/DTFE Launcher" 30 & qapp=$!
"${TMP}/q_bin/PS-DTFE" 30 & qpid=$!
sleep 1
qheavy="$(eval "$(sed -n '/^heavy_running() {/,/^}/p' "${MREPO}/scripts/thesis_runs.sh")"; heavy_running)"
kill "${qapp}" "${qpid}" 2>/dev/null || true; wait "${qapp}" "${qpid}" 2>/dev/null || true
check "$( { printf '%s\n' "${qheavy}" | grep -qE "^ *${qpid} .*q_bin/PS-DTFE 30"; } && ! printf '%s\n' "${qheavy}" | grep -qE "^ *${qapp} " \
         && echo 0 || echo 1)" "... the guard goes by the executable: a 'DTFE Launcher.app' (a path with a space) is not a DTFE run, a PS-DTFE is"

# ps27 and dtfe27 (2026-10-13): the two drivers of 2026-10-12 that made the grids on the data disk, as stages. Every box
# gets the nine snapshots; the caller's knobs (QO) must not reach a planned command.
for b in TNG50/TNG50-3-Dark TNG100/TNG100-3-Dark TNG300/TNG300-3-Dark; do
    for s in 000 004 017 025 033 040 050 067 099; do
        mkdir -p "${TROOT}/${b}/snapdir_${s}"; [ -e "${TROOT}/${b}/snapdir_${s}/combined_${s}.hdf5" ] || : > "${TROOT}/${b}/snapdir_${s}/combined_${s}.hdf5"
    done
done
rm -rf "${TMP}/q_scratch"
rc=0; qv "${QO[@]}" DRY_RUN=1 Q_PY_LOG="${TMP}/q_py_27.log" "${MREPO}/scripts/thesis_runs.sh" ps27 dtfe27 >"${TMP}/q_27.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ ! -e "${TMP}/q_py_27.log" ] && [ ! -e "${TMP}/q_scratch" ] && [ ! -e "${TROOT}/../_backup" ] && [ ! -e "${TROOT}/_backup" ] \
         && [ "$(lacks "${TMP}/q_27.log" '^RUNNING:')" = 0 ] && echo 0 || echo 1)" "DRY_RUN=1 ps27 dtfe27 exits 0: no binary, no python, nothing made or moved (${rc})"
q27() { grep -aE -e "^ +\./$1 " "${TMP}/q_27.log"; }
# the PS line is the 2026-10-12 RUNNING line token for token (~/DTFE-rerun/psrerun-2026-10-12/*.log), with --ordered-merge
# since 2026-10-14 (ORDERED_MERGE=1: the staged merge in index order, the new default)
QPS='ps_output +--grid +512 +--padding +50 +--periodic +--scratch-dir +.*q_scratch +--ordered-merge +--avg-subsamples +3 +--input +105 +--MpcUnit +1000 +--field +density_a +velocity_a +gradient_a +divergence_a +shear_a +vorticity_a +dispersion_a +tweb_a +vweb_a +--lambda_th +0\.3 +--ps-gpu +--ps-vertex-mass +--ps-volume-weighted +--sample-points +.*/hires_plane_z4096\.bin +--pts-vel-grad +--ps-caustics +--lagrangianInput +.*/combined_ics\.hdf5$'
check "$([ "$(q27 PS-DTFE | grep -cE -e "${QPS}")" = "27" ] \
         && [ "$(q27 PS-DTFE | grep -oE 'TNG[0-9]+-3-Dark/snapdir_[0-9]{3}/ps_output' | tr '\n' ' ')" = "$(for b in TNG100 TNG300 TNG50; do for s in 000 004 017 025 033 040 050 067 099; do printf '%s-3-Dark/snapdir_%s/ps_output ' "${b}" "${s}"; done; done)" ] \
         && echo 0 || echo 1)" "ps27: 27 PS-DTFE runs (TNG100, TNG300, TNG50; z = 20 ... 0), each the 2026-10-12 command token for token: the 2026-10-03 one + --ps-caustics, the box's 4096^2 plane (+ --ordered-merge since 2026-10-14)"
check "$([ "$(grep -acE '^\+ .*make_image_plane\.py --combined .*-3-Dark/snapdir_000/combined_000\.hdf5 --axis z --nu 4096 --planes 1 --thickness 2 --supersample 1 -o .*-3-Dark/hires_plane_z4096$' "${TMP}/q_27.log")" = "3" ] \
         && echo 0 || echo 1)" "... a box without its 4096^2 plane gets one made first, from snapdir_000, as run_ps_pipeline.sh makes it"
QDT='output +--grid +512 +--padding +15 +--periodic +--partition +%s +--max-concurrent +%s +--field +density_a +velocity_a +gradient_a +divergence_a +shear_a +vorticity_a +tweb_a +vweb_a +--lambda_th +0\.3 +--gpu +--exact-average$'
check "$([ "$(q27 DTFE | grep -cE -e "TNG50-3-Dark/snapdir_[0-9]{3}/$(printf "${QDT}" '4 +4 +4' 4)")" = "9" ] \
         && [ "$(q27 DTFE | grep -cE -e "TNG300-3-Dark/snapdir_[0-9]{3}/$(printf "${QDT}" '5 +5 +5' 10)")" = "9" ] \
         && [ "$(q27 DTFE | grep -cE -e "TNG100-3-Dark/snapdir_[0-9]{3}/$(printf "${QDT}" '3 +3 +3' 10)")" = "9" ] \
         && [ "$(q27 DTFE | grep -oE 'TNG[0-9]+-3-Dark/snapdir_[0-9]{3}/output' | tr '\n' ' ')" = "$(for b in TNG50 TNG300 TNG100; do for s in 000 004 017 025 033 040 050 067 099; do printf '%s-3-Dark/snapdir_%s/output ' "${b}" "${s}"; done; done)" ] \
         && echo 0 || echo 1)" "dtfe27: 27 standard-DTFE runs (TNG50 4x4x4 x4, TNG300 5x5x5 x10, TNG100 3x3x3 x10), the 2026-10-12 command with padding 15 and --exact-average, no scratch dir"
check "$([ "$(lacks "${TMP}/q_27.log" 'leaky|--partition +2 +2 +2|/no/such|--padding +40|--lambda_th +0\.7|--max-concurrent +3 |ps-lag-padding|parallel-triangulation|ps-exact-deposit|auto-tune-report')" = 0 ] \
         && [ "$(q27 'PS-DTFE|DTFE' | grep -c -e '--avg-subsamples +1 ')" = "0" ] && echo 0 || echo 1)" "... the caller's knobs (prefix, split, padding, fields, plane, threshold, deposit, report) reach neither"
# the skip rule: ps27 skips a snapshot only when its run log shows the caustics AND the 4096^2 plane; SIMS limits the boxes
T3="${TROOT}/TNG300/TNG300-3-Dark/snapdir_099"
echo x > "${T3}/ps_output.a_den"
echo 'RUNNING: ./PS-DTFE in out --grid 512 --ordered-merge --field density_a --ps-gpu --sample-points /x/hires_plane_z4096.bin --pts-vel-grad --ps-caustics --lagrangianInput /x/ics ' > "${T3}/ps_output.runlog"
# 2026-10-14: the 2026-10-12 set (caustics and plane, merged in completion order: no --ordered-merge) is rerun
T6="${TROOT}/TNG300/TNG300-3-Dark/snapdir_050"
echo x > "${T6}/ps_output.a_den"
echo 'RUNNING: ./PS-DTFE in out --grid 512 --field density_a --ps-gpu --sample-points /x/hires_plane_z4096.bin --pts-vel-grad --ps-caustics --lagrangianInput /x/ics ' > "${T6}/ps_output.runlog"
T5="${TROOT}/TNG300/TNG300-3-Dark/snapdir_067"
echo x > "${T5}/ps_output.a_den"
echo 'RUNNING: ./PS-DTFE in out --grid 512 --field density_a --ps-gpu --sample-points /x/hires_plane_z4096.bin --pts-vel-grad --lagrangianInput /x/ics ' > "${T5}/ps_output.runlog"
rc=0; qv DRY_RUN=1 SIMS=TNG300-3-Dark "${MREPO}/scripts/thesis_runs.sh" ps27 >"${TMP}/q_27skip.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ "$(grep -acE '^ +\./PS-DTFE ' "${TMP}/q_27skip.log")" = "8" ] && [ "$(lacks "${TMP}/q_27skip.log" 'TNG50-3-Dark|TNG100-3-Dark')" = 0 ] \
         && [ "$(has "${TMP}/q_27skip.log" 'ps27: TNG300-3-Dark 099: ps_output\.\* made with the caustics, the 4096\^2 plane and the ordered merge already: skipped')" = 0 ] \
         && [ "$(has "${TMP}/q_27skip.log" '^\+ mv -n .*snapdir_067/ps_output\.\* .*_backup/ps_output_[0-9-]+/TNG300/TNG300-3-Dark/snapdir_067/ \(2 files\)$')" = 0 ] \
         && [ "$(has "${TMP}/q_27skip.log" '^\+ mv -n .*snapdir_050/ps_output\.\* .*_backup/ps_output_[0-9-]+/TNG300/TNG300-3-Dark/snapdir_050/ \(2 files\)$')" = 0 ] \
         && echo 0 || echo 1)" "ps27: a snapshot made by this recipe is skipped; one without the caustics, and (2026-10-14) the 2026-10-12 set merged in completion order, are rerun (their backup moves printed); SIMS=TNG300-3-Dark: that box alone (${rc})"
rm -f "${T3}/ps_output."* "${T5}/ps_output."* "${T6}/ps_output."*
# a real dtfe27 run of one box against the stand-in binaries, and a stand-in ps (no other run on the machine, whatever
# this machine runs: the heavy guard reads ps): the old files of a snapshot are MOVED to _backup/output_<date>/... first,
# never deleted; a failed run gets them moved back
mkdir -p "${TMP}/q_nops"; printf '#!/bin/sh\nexit 0\n' > "${TMP}/q_nops/ps"; chmod +x "${TMP}/q_nops/ps"
for f in a_den pts_den; do echo "old ${f}" > "${T1}/snapdir_099/output.${f}"; done
echo 'RUNNING: ./DTFE in out --grid 512 --padding 20 --gpu ' > "${T1}/snapdir_099/output.runlog"
rc=0; qv PATH="${TMP}/q_nops:${PATH}" SIMS=TNG100-3-Dark FAKE_MODE=ok DTFE_SPACE_CHECK=0 Q_PY_LOG="${TMP}/q_py_27run.log" \
    "${MREPO}/scripts/thesis_runs.sh" dtfe27 >"${TMP}/q_27run.log" 2>&1 || rc=$?
QB="$(ls -d "${TROOT}"/_backup/output_*/TNG100/TNG100-3-Dark/snapdir_099 2>/dev/null | head -1)"
check "$([ "${rc}" = "0" ] && [ -n "${QB}" ] && [ "$(cat "${QB}/output.a_den")" = "old a_den" ] && [ "$(cat "${QB}/output.pts_den")" = "old pts_den" ] \
         && grep -aq -e '--padding 20' "${QB}/output.runlog" && [ ! -e "${T1}/snapdir_099/output.a_den" ] && [ ! -e "${T1}/snapdir_099/output.pts_den" ] \
         && grep -aq '^RUNNING: .*--padding 15 .*--exact-average' "${T1}/snapdir_099/output.runlog" \
         && [ "$(grep -ac '^RUNNING:' "${TMP}/q_27run.log")" = "9" ] && [ "$(has "${TMP}/q_27run.log" 'dtfe27: moved 3 old output\.\* files to ')" = 0 ] \
         && echo 0 || echo 1)" "a dtfe27 run (one box, stand-ins): the 9 snapshots run; the old output.* (grids, plane, run log) MOVED to _backup/output_<date>/TNG100/TNG100-3-Dark/snapdir_099 first (${rc})"
echo "old 050" > "${T1}/snapdir_050/output.a_den"; echo 'RUNNING: ./DTFE old 050 ' > "${T1}/snapdir_050/output.runlog"
rc=0; qv PATH="${TMP}/q_nops:${PATH}" SIMS=TNG100-3-Dark FAKE_MODE=fail DTFE_SPACE_CHECK=0 Q_PY_LOG="${TMP}/q_py_27run.log" \
    "${MREPO}/scripts/thesis_runs.sh" dtfe27 >"${TMP}/q_27fail.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ "$(cat "${T1}/snapdir_050/output.a_den")" = "old 050" ] && grep -aq 'old 050' "${T1}/snapdir_050/output.runlog" \
         && [ -f "${T1}/snapdir_050/output.runlog.failed" ] && [ -z "$(ls "${QB%/*}/snapdir_050" 2>/dev/null)" ] \
         && [ "$(has "${TMP}/q_27fail.log" 'dtfe27: FAILED \(exit 1\): 2 old output\.\* files moved back from ')" = 0 ] \
         && [ "$(grep -ac '^failed step: dtfe27: ' "${TMP}/q_27fail.log")" = "9" ] && echo 0 || echo 1)" \
      "... a FAILED run: the old files moved back (the snapshot as before, the attempt's log as .runlog.failed), listed, exit 1 (${rc})"
# dtfe27planes (2026-10-14): the standard-DTFE 4096^2 planes of the same 27 snapshots, at dtfe27's splits and settings,
# under the prefix output_plane (density_a is the one grid field the binary needs); the caller's knobs (QO) ignored
rm -rf "${TMP}/q_scratch"
rc=0; qv "${QO[@]}" DRY_RUN=1 Q_PY_LOG="${TMP}/q_py_pl.log" "${MREPO}/scripts/thesis_runs.sh" dtfe27planes >"${TMP}/q_pl.log" 2>&1 || rc=$?
qpl() { grep -aE -e "^ +\./DTFE " "${TMP}/q_pl.log"; }
QPL='output_plane +--grid +512 +--padding +15 +--periodic +--partition +%s +--max-concurrent +%s +--field +density_a +--lambda_th +0\.3 +--sample-points +.*%s/hires_plane_z4096\.bin +--pts-vel-grad +--gpu +--exact-average$'
check "$([ "${rc}" = "0" ] && [ ! -e "${TMP}/q_py_pl.log" ] && [ ! -e "${TMP}/q_scratch" ] && [ ! -e "${TROOT}/_backup/output_plane_$(date '+%Y-%m-%d')" ] \
         && [ "$(lacks "${TMP}/q_pl.log" '^RUNNING:')" = 0 ] \
         && [ "$(qpl | grep -cE -e "TNG50-3-Dark/snapdir_[0-9]{3}/$(printf "${QPL}" '4 +4 +4' 4 TNG50-3-Dark)")" = "9" ] \
         && [ "$(qpl | grep -cE -e "TNG300-3-Dark/snapdir_[0-9]{3}/$(printf "${QPL}" '5 +5 +5' 10 TNG300-3-Dark)")" = "9" ] \
         && [ "$(qpl | grep -cE -e "TNG100-3-Dark/snapdir_[0-9]{3}/$(printf "${QPL}" '3 +3 +3' 10 TNG100-3-Dark)")" = "9" ] \
         && [ "$(qpl | grep -oE 'TNG[0-9]+-3-Dark/snapdir_[0-9]{3}/output_plane' | tr '\n' ' ')" = "$(for b in TNG50 TNG300 TNG100; do for s in 000 004 017 025 033 040 050 067 099; do printf '%s-3-Dark/snapdir_%s/output_plane ' "${b}" "${s}"; done; done)" ] \
         && [ "$(lacks "${TMP}/q_pl.log" 'leaky|/no/such|--partition +2 +2 +2|--padding +40|--lambda_th +0\.7|--max-concurrent +3 |vweb|tessellation-cache|auto-tune-report|--scratch-dir')" = 0 ] \
         && [ "$(grep -acE '^\+ .*make_image_plane\.py --combined .*-3-Dark/snapdir_000/combined_000\.hdf5 --axis z --nu 4096 --planes 1 --thickness 2 --supersample 1 -o .*-3-Dark/hires_plane_z4096$' "${TMP}/q_pl.log")" = "3" ] \
         && echo 0 || echo 1)" \
      "dtfe27planes (DRY_RUN): 27 standard-DTFE plane runs in dtfe27's order and splits, each the box's 4096^2 plane with --pts-vel-grad, density_a only, padding 15, GPU, exact average, prefix output_plane; a missing plane made first as ps27 makes it; the caller's knobs ignored (${rc})"
rc=0; qv DRY_RUN=1 Q_PY_LOG="${TMP}/q_py_pl.log" "${MREPO}/scripts/thesis_runs.sh" all >"${TMP}/q_all.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ "$(grep -aoE '(ps27|dtfe27|dtfe27planes|caustics2): expected' "${TMP}/q_all.log" | tr '\n' ' ')" = "ps27: expected dtfe27: expected dtfe27planes: expected caustics2: expected " ] \
         && [ "$(grep -acE '^ +\./DTFE .*/output_plane ' "${TMP}/q_all.log")" = "27" ] && echo 0 || echo 1)" \
      "'all' runs dtfe27planes right after dtfe27 (${rc})"
# the skip rule: output_plane.pts_den AND a run log with the plane, the gradients, padding 15 and the exact average
echo x > "${T1}/snapdir_099/output_plane.pts_den"
echo 'RUNNING: ./DTFE in out --grid 512 --padding 15 --periodic --partition 3 3 3 --field density_a --sample-points /x/hires_plane_z4096.bin --pts-vel-grad --gpu --exact-average ' > "${T1}/snapdir_099/output_plane.runlog"
echo x > "${T1}/snapdir_067/output_plane.pts_den"
echo 'RUNNING: ./DTFE in out --grid 512 --padding 15 --periodic --field density_a --sample-points /x/hires_plane_z4096.bin --gpu --exact-average ' > "${T1}/snapdir_067/output_plane.runlog"
rc=0; qv DRY_RUN=1 SIMS=TNG100-3-Dark "${MREPO}/scripts/thesis_runs.sh" dtfe27planes >"${TMP}/q_plskip.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ "$(grep -acE '^ +\./DTFE ' "${TMP}/q_plskip.log")" = "8" ] && [ "$(lacks "${TMP}/q_plskip.log" 'TNG50-3-Dark|TNG300-3-Dark')" = 0 ] \
         && [ "$(has "${TMP}/q_plskip.log" 'dtfe27planes: TNG100-3-Dark 099: output_plane\.\* .* made already: skipped')" = 0 ] \
         && [ "$(has "${TMP}/q_plskip.log" '^\+ mv -n .*snapdir_067/output_plane\.\* .*_backup/output_plane_[0-9-]+/TNG100/TNG100-3-Dark/snapdir_067/ \(2 files\)$')" = 0 ] \
         && echo 0 || echo 1)" "dtfe27planes: a finished plane is skipped, one without --pts-vel-grad is rerun (its backup move printed); SIMS=TNG100-3-Dark: that box alone (${rc})"
rm -f "${T1}/snapdir_099/output_plane."* "${T1}/snapdir_067/output_plane."*
# a real run of one box (stand-ins): dtfe27's output.* grids beside it stay where they are, untouched; an old
# output_plane.* set is MOVED to _backup/output_plane_<date>/... first
: > "${T1}/hires_plane_z4096.bin"; : > "${T1}/hires_plane_z4096.json"
echo "dtfe27 grid" > "${T1}/snapdir_040/output.a_den"; echo 'RUNNING: ./DTFE dtfe27 --padding 15 --exact-average ' > "${T1}/snapdir_040/output.runlog"
echo "old plane" > "${T1}/snapdir_040/output_plane.pts_den"; echo 'RUNNING: ./DTFE old plane ' > "${T1}/snapdir_040/output_plane.runlog"
rc=0; qv PATH="${TMP}/q_nops:${PATH}" SIMS=TNG100-3-Dark FAKE_MODE=ok DTFE_SPACE_CHECK=0 Q_PY_LOG="${TMP}/q_py_plrun.log" \
    "${MREPO}/scripts/thesis_runs.sh" dtfe27planes >"${TMP}/q_plrun.log" 2>&1 || rc=$?
QPB="$(ls -d "${TROOT}"/_backup/output_plane_*/TNG100/TNG100-3-Dark/snapdir_040 2>/dev/null | head -1)"
check "$([ "${rc}" = "0" ] && [ -n "${QPB}" ] && [ "$(cat "${QPB}/output_plane.pts_den")" = "old plane" ] && grep -aq 'old plane' "${QPB}/output_plane.runlog" \
         && [ "$(cat "${T1}/snapdir_040/output.a_den")" = "dtfe27 grid" ] && grep -aq 'dtfe27 --padding 15' "${T1}/snapdir_040/output.runlog" \
         && [ ! -e "${QPB}/output.a_den" ] && [ -z "$(ls "${T1}/snapdir_040" | grep -E '^output\.superseded')" ] \
         && grep -aq '^RUNNING: .*/output_plane .*--sample-points .*/hires_plane_z4096\.bin --pts-vel-grad --gpu --exact-average' "${T1}/snapdir_040/output_plane.runlog" \
         && [ "$(grep -ac '^RUNNING: .*/output_plane ' "${TMP}/q_plrun.log")" = "9" ] \
         && echo 0 || echo 1)" "a dtfe27planes run (one box, stand-ins): 9 plane runs; the old output_plane.* MOVED to _backup/output_plane_<date>/..., dtfe27's output.* beside them untouched (${rc})"
rm -f "${T1}/hires_plane_z4096.bin" "${T1}/hires_plane_z4096.json" "${T1}/snapdir_040/output."* "${T1}/snapdir_040/output_plane."*
rm -rf "${TROOT}/_backup"

echo ""
echo "(N) the Makefile stamps BUILD_MODE, ARCH_FLAGS and EXTRA_FLAGS in the object dirs (2026-10-08)."
# 'make DTFE BUILD_MODE=debug' after a release build found every object up to date and linked the -O3 objects
# with the sanitizers (an ARCH_FLAGS / EXTRA_FLAGS change kept the old objects the same way). The stamp's name
# carries the mode and a checksum of the three, so a switch wipes the objects. A dry run of the mode check alone
# (make -n -p) prints the stamp and builds nothing; the PS-DTFE target would start its recursive make, which
# make runs even under -n.
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
