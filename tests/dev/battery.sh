#!/usr/bin/env bash
# The local battery (macOS, Apple Silicon): CI's stages plus what CI cannot run -- the Metal GPU suites, single and
# double precision -- and the Python suites under CI's numpy 1.26 / matplotlib 3.6 (Ubuntu 24.04's versions), with
# no simulation data (DTFE_DATA_ROOT points at an empty folder, as on CI). 31 stages, about 1.5-2 h on 10 cores.
# It builds the three binary pairs first (METAL=1, METAL=1 DOUBLE=1, DIM=2) and runs on this checkout; do not edit
# the tree while it runs (the summary says if it changed).
#
# Usage: tests/dev/battery.sh <scratch dir> [--ci-python PATH] [--gui-python PATH] [--only NAME,NAME...] [--list]
#   <scratch dir>   logs (<scratch>/battery/<stage>.log and SUMMARY.txt), the suites' temporary files and the
#                   empty data root; any folder (a /tmp folder is fine: nothing in it is needed afterwards)
#   --ci-python     python with CI's package versions for the 13 '*_np126' stages (default $DTFE_CI_PYTHON, else
#                   ~/.venvs/dtfe-ci/bin/python); tests/dev/make_ci_venv.sh makes one
#   --gui-python    python with PySide6 for the 'ci_gui' stage (default $DTFE_GUI_PYTHON, else
#                   ~/.venvs/dtfe-gui/bin/python)
#   --only          run only these stages (names as --list prints them)
#   --list          print the stage names and exit
# Exit 0 when every stage that ran passed and the tree did not change during the run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 2; }

SCRATCH=""; ONLY=""; LIST=0
CI_PY="${DTFE_CI_PYTHON:-$HOME/.venvs/dtfe-ci/bin/python}"
GUI_PY="${DTFE_GUI_PYTHON:-$HOME/.venvs/dtfe-gui/bin/python}"
while [ $# -gt 0 ]; do
    case "$1" in
        --ci-python)  [ $# -ge 2 ] || usage; CI_PY="$2"; shift 2 ;;
        --gui-python) [ $# -ge 2 ] || usage; GUI_PY="$2"; shift 2 ;;
        --only)       [ $# -ge 2 ] || usage; ONLY=",$2,"; shift 2 ;;
        --list)       LIST=1; shift ;;
        -h|--help)    usage ;;
        -*)           echo "unknown option $1"; usage ;;
        *)            [ -z "$SCRATCH" ] || usage; SCRATCH="$1"; shift ;;
    esac
done

PY_SUITES="dtfelib:py_dtfelib_test void_core:py_void_core_test profiles:py_profiles_test population:py_population_test
           environment:py_environment_test fate:py_fate_test convergence:py_convergence_test sizefn:py_size_function_test
           vic:py_vic_test meanfield:py_meanfield_test gui_explore:py_gui_explore_test gui_app2:py_gui_app2_test
           runspec:py_gui_runspec_test"
selected() { [ -z "$ONLY" ] || [[ "$ONLY" == *",$1,"* ]]; }
if [ "$LIST" = 1 ]; then
    for s in build build_d build_2d ci_core ci_double ci_2d ci_gui dtfe_metal gpu_scalar gpu_mass gpu_retry dtfe_gpu_retry \
             auto_tune scratch padding_d gpu_mass_d auto_tune_d gpu_retry_d; do echo "$s"; done
    for t in $PY_SUITES; do echo "${t%%:*}_np126"; done
    exit 0
fi
[ -n "$SCRATCH" ] || usage
mkdir -p "$SCRATCH" && SCRATCH="$(cd "$SCRATCH" && pwd)" || exit 2

# the pythons the selected stages need, checked before an hour of running
need_ci=0; for t in $PY_SUITES; do selected "${t%%:*}_np126" && need_ci=1; done
if [ "$need_ci" = 1 ] && ! "$CI_PY" -c "import numpy, scipy, h5py, matplotlib, PySide6" 2>/dev/null; then
    echo "no CI-like python at $CI_PY (numpy, scipy, h5py, matplotlib, PySide6): make one with"
    echo "    tests/dev/make_ci_venv.sh            (or pass --ci-python PATH, or --only without the *_np126 stages)"
    exit 2
fi
if selected ci_gui && ! "$GUI_PY" -c "import PySide6" 2>/dev/null; then
    echo "no python with PySide6 at $GUI_PY for the ci_gui stage: pass --gui-python PATH (or leave ci_gui out with --only)"
    exit 2
fi

L="$SCRATCH/battery"; mkdir -p "$L"
export DTFE_TEST_TMP="$SCRATCH/battery-tmp"; mkdir -p "$DTFE_TEST_TMP"
export MPLBACKEND=Agg
ER="$SCRATCH/emptyroot"; mkdir -p "$ER"
cd "$ROOT" || exit 2
# the tree's state: the diff against HEAD and the untracked (not ignored) files' contents
fingerprint() { { git diff HEAD; git ls-files -o --exclude-standard -z | xargs -0 shasum 2>/dev/null; } | shasum | cut -c1-16; }
tree0="$(fingerprint)"
summary="$L/SUMMARY.txt"
{
    echo "battery of $ROOT"
    echo "started $(date '+%Y-%m-%d %H:%M'), HEAD $(git rev-parse --short HEAD), $(git status --porcelain | wc -l | tr -d ' ') changed/untracked paths"
    [ "$need_ci" = 1 ] && echo "CI-like python: $CI_PY ($("$CI_PY" -c 'import sys, numpy, scipy, matplotlib; print(f"py{sys.version_info[0]}.{sys.version_info[1]} numpy {numpy.__version__} scipy {scipy.__version__} matplotlib {matplotlib.__version__}")'))"
} | tee "$summary"
ran=0; passed=0
run() { local name="$1"; shift
    selected "$name" || return 0
    local t0=$SECONDS
    "$@" > "$L/$name.log" 2>&1; local rc=$?
    ran=$((ran + 1)); [ "$rc" = 0 ] && passed=$((passed + 1))
    echo "$(printf '%-28s' "$name") rc=$rc  $((SECONDS - t0)) s" | tee -a "$summary"; }

J="-j$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
run build       make "$J" DTFE PS-DTFE METAL=1
run build_d     make "$J" DTFE PS-DTFE METAL=1 DOUBLE=1
run build_2d    make "$J" DTFE PS-DTFE DIM=2
run ci_core     tests/ci_suite.sh core
run ci_double   tests/ci_suite.sh double
run ci_2d       tests/ci_suite.sh 2d
run ci_gui      env DTFE_GUI_PYTHON="$GUI_PY" tests/ci_suite.sh gui
run dtfe_metal  tests/dtfe_metal_check.sh --no-build
run gpu_scalar  tests/gpu_scalar_check.sh --no-build
run gpu_mass    tests/gpu_mass_check.sh --no-build
run gpu_retry   tests/gpu_retry_check.sh --no-build
run dtfe_gpu_retry tests/dtfe_gpu_retry_check.sh --no-build
run auto_tune   tests/ps_auto_tune_check.sh --no-build
run scratch     tests/ps_scratch_check.sh --no-build
run padding_d   env DTFE_TEST_PRECISION=double tests/ps_partition_padding_check.sh --no-build
run gpu_mass_d  env DTFE_TEST_PRECISION=double tests/gpu_mass_check.sh --no-build
run auto_tune_d env DTFE_TEST_PRECISION=double tests/ps_auto_tune_check.sh --no-build
run gpu_retry_d env DTFE_TEST_PRECISION=double tests/gpu_retry_check.sh --no-build
for t in $PY_SUITES; do
    run "${t%%:*}_np126" env DTFE_DATA_ROOT="$ER" "$CI_PY" "tests/${t#*:}.py"
done

tree_ok=1; [ "$(fingerprint)" = "$tree0" ] || tree_ok=0
{
    [ "$tree_ok" = 1 ] || echo "WARNING: the tree changed during the battery: the stages may have run on different code"
    echo "BATTERY DONE $(date '+%H:%M'): $passed of $ran stages passed"
} | tee -a "$summary"
[ "$passed" = "$ran" ] && [ "$tree_ok" = 1 ]
