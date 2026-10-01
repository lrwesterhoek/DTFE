#!/usr/bin/env bash
# Contract check for the batch driver scripts/run_ps_dtfe.sh (and the shared scripts/config.sh).
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
#
# Usage: tests/run_scripts_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# python with a working h5py (plain python3 resolves to a broken x86_64 h5py on the dev machine)
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14

N="${N:-16}"
GRID="${GRID:-24}"
BOX="${BOX:-100.0}"
TMP="${TMP_DIR:-${SCRIPT_DIR}/tmp}/run_scripts"
DRIVER="scripts/run_ps_dtfe.sh"

echo "============================================================"
echo " run_ps_dtfe.sh contract check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

if [ "${1:-}" != "--no-build" ]; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x ./PS-DTFE ] || { echo "FAIL: ./PS-DTFE not built"; exit 1; }

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
check "$(has   "${DRIVER}" 'TIME_WRAP\[@\].*\./PS-DTFE')" \
      "the binary is invoked through the resolved-once TIME_WRAP instead"
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

echo ""
echo "------------------------------------------------------------"
if [ "${fails}" -eq 0 ]; then
    rm -rf "${TMP}"
    echo "RESULT: PASS  (the batch driver honours its environment, needs no IC file when the"
    echo "               snapshot carries one, reports honest exit codes, is portable, and"
    echo "               handles the per-family data layout)"
    exit 0
fi
echo "RESULT: FAIL  (${fails} check(s); logs kept under ${TMP})"
exit 1
