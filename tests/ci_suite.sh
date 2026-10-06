#!/usr/bin/env bash
# The Linux CI suite, ONE list shared by .github/workflows/ci.yml and the Dockerfile so the two cannot
# drift. CPU-only: the GPU suites (dtfe_metal_check, gpu_scalar_check) need a GPU build and stay local,
# as do ps_auto_tune_check and ps_scratch_check (macOS-isms: /usr/bin/time -l, /private/tmp).
#
#   tests/ci_suite.sh build     the single pair, the double pair (DOUBLE=1) and the 2D pair (DIM=2)
#   tests/ci_suite.sh core      every CPU suite against ./DTFE and ./PS-DTFE
#   tests/ci_suite.sh double    ps_double_check, then the program-running suites against the double
#                               pair (DTFE_TEST_PRECISION=double: float64 grids, tests/precision.sh)
#   tests/ci_suite.sh 2d        tests/ps_2d_check.sh (the 2D pair against analytic 2D flows)
#   tests/ci_suite.sh gui       the launcher's job specifications; the end-to-end window test too when
#                               DTFE_GUI_PYTHON names a Python with PySide6 (QT_QPA_PLATFORM=offscreen)
#   tests/ci_suite.sh all       build core double 2d
# Several stages may be given at once; the first failing suite stops the run (set -e).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
PY="${PYTHON:-python3}"
ARCH="${ARCH_FLAGS:--mtune=generic}"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
step() { echo; echo "==== $*"; }

stage_build() {
    step "build: the single pair"
    make deps-check
    make DTFE PS-DTFE ARCH_FLAGS="$ARCH" -j"$JOBS"
    step "build: the double pair (DOUBLE=1)"
    make DTFE PS-DTFE DOUBLE=1 ARCH_FLAGS="$ARCH" -j"$JOBS"
    step "build: the 2D pair (DIM=2)"
    make DTFE PS-DTFE DIM=2 ARCH_FLAGS="$ARCH" -j"$JOBS"
    ./DTFE --version; ./PS-DTFE --version; ./DTFE-double --version; ./PS-DTFE-double --version
    ./DTFE-2d --version; ./PS-DTFE-2d --version
}

# the suites that run the programs (also the double pass); the python ones take --no-build
run_program_suites() {
    local ref; ref="$(mktemp -d)"
    step "standard DTFE regression";            DTFE_TEST_REFERENCE_DIR="$ref" "$PY" tests/run_tests.py
    step "PS-DTFE smoke test";                  tests/ps_smoke_test.sh --no-build
    step "PS-DTFE regression (analytic pancake)"; "$PY" tests/ps_regression_test.py --no-build
    step "3D crossed waves";                    "$PY" tests/ps_3d_test.py --no-build
    step "single-stream convergence";           "$PY" tests/ps_convergence_test.py --no-build
    step "PS vs standard DTFE (single stream)"; "$PY" tests/ps_standard_cross_check.py --no-build
    step "linear/exact deposit";                tests/ps_linear_deposit_check.sh --no-build
    step "halo release";                        tests/ps_halo_release_check.sh --no-build
    step "vertex mass";                         tests/ps_vertex_mass_check.sh --no-build
    step "volume weighted";                     tests/ps_volume_weighted_check.sh --no-build
    step "parallel == serial";                  tests/ps_parallel_check.sh --no-build
    step "slab-mode deposit == serial";         tests/ps_slab_deposit_check.sh --no-build
    step "caustic stratification";              tests/ps_caustic_class_check.sh --no-build
    step "tessellation cache";                  tests/ps_tessellation_cache_check.sh --no-build
    step "PS point evaluation";                 tests/ps_point_eval_check.sh --no-build
    step "standard point evaluation";           tests/dtfe_point_eval_check.sh --no-build
    step "exact point location + --serve";      tests/point_exact_serve_check.sh --no-build
    step "exact hidden-streams flag";           tests/ps_hidden_streams_check.sh --no-build
    step "window == full run";                  tests/ps_window_check.sh --no-build
    step "non-periodic suite";                  "$PY" tests/ps_nonperiodic_test.py --no-build
    step "batch drivers";                       tests/run_scripts_check.sh --no-build
    step "T-web (tidal tensor)";                tests/tweb_check.sh --no-build
    step "interlacing (unit, 2D + 3D)";         tests/interlacing_check.sh
    step "composite server split";              tests/serve_split_check.sh --no-build
    rm -rf "$ref"
}

# The oldest supported Python (pyproject: >= 3.10) must PARSE every file: a PEP 701 f-string (3.12) is a
# SyntaxError there, and the suites run under a newer interpreter that cannot see it (plot_caustic_skeleton.py,
# review 2026-10-05). Needs a REAL 3.10/3.11 -- ast.parse(feature_version=) under 3.12+ does not catch it.
old_python_parse() {
    local p
    for p in python3.10 python3.11; do
        if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(sys.version_info >= (3, 12))' 2>/dev/null; then
            "$p" - python tests <<'EOF'
import sys, pathlib
bad = n = 0
for root in sys.argv[1:]:
    for f in sorted(pathlib.Path(root).rglob("*.py")):
        if {"tmp", "__pycache__", "cache", "figures"} & set(f.parts):
            continue
        n += 1
        try:
            compile(f.read_bytes(), str(f), "exec", dont_inherit=True)
        except SyntaxError as e:
            print(f"  {f}:{e.lineno}: {e.msg}")
            bad += 1
print(f"  {n} files parsed under Python {sys.version.split()[0]}: " + (f"{bad} FAILED" if bad else "all OK"))
sys.exit(1 if bad else 0)
EOF
            return
        fi
    done
    echo "  no python3.10 / python3.11 on PATH: the oldest-Python parse check skipped"
}

stage_core() {
    unset DTFE_TEST_PRECISION
    step "every .py parses under the oldest supported Python"; old_python_parse
    run_program_suites
    # AFTER the program suites: ps_caustic_class_check.sh leaves the crossed-waves run (pcc_out.*) under the
    # temp root, which the caustic-skeleton test's real-grid half reads (else it is a SKIP, not a pass)
    step "dtfelib python suite (synthetic tier; the data-backed tier skips without the T7)"; "$PY" tests/py_dtfelib_test.py
}

# after stage_build the pairs EXIST: a suite that would skip for a missing binary or feature is a failure here
# (DTFE_TEST_REQUIRED=1 turns every skip_or_fail into exit 1; survey rank 20, 2026-10-05)
stage_double() {
    step "the double pair against the single pair"; DTFE_TEST_REQUIRED=1 tests/ps_double_check.sh --no-build
    export DTFE_TEST_PRECISION=double DTFE_TEST_REQUIRED=1
    run_program_suites
    unset DTFE_TEST_PRECISION DTFE_TEST_REQUIRED
}

stage_2d() {
    step "2D phase space (PS-DTFE-2d, DTFE-2d)"; DTFE_TEST_REQUIRED=1 tests/ps_2d_check.sh --no-build
}

stage_gui() {
    step "launcher job specifications (no Qt)"; "$PY" tests/py_gui_runspec_test.py
    if [ -n "${DTFE_GUI_PYTHON:-}" ]; then
        step "launcher end-to-end (offscreen)"
        QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}" DTFE_GUI_TEST_REQUIRED=1 "$DTFE_GUI_PYTHON" tests/py_gui_app_test.py
    fi
}

[ "$#" -gt 0 ] || { sed -n '2,17p' "$0"; exit 2; }
for s in "$@"; do
    case "$s" in
        build|core|double|2d|gui) "stage_$s" ;;
        all) stage_build; stage_core; stage_double; stage_2d ;;
        *) echo "unknown stage '$s' (build core double 2d gui all)"; exit 2 ;;
    esac
done
echo; echo "==== ci_suite: $* PASSED"
