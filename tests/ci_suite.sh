#!/usr/bin/env bash
# The Linux CI suite, ONE list shared by .github/workflows/ci.yml and the Dockerfile so the two cannot
# drift. CPU-only: the GPU suites (dtfe_metal_check, gpu_scalar_check, gpu_mass_check, gpu_retry_check, and the
# standard binary's dtfe_gpu_retry_check since 2026-10-13) need a GPU
# build and stay local (the CUDA/HIP port runs them through its CPU emulation in the ci.yml job cudaemu:
# tests/cudaemu_check.sh). ps_auto_tune_check and ps_scratch_check run here since 2026-10-11 (ported: the peak RSS from
# getrusage instead of '/usr/bin/time -l', a temp folder instead of /private/tmp; their GPU parts SKIP).
#
#   tests/ci_suite.sh build     the single pair, the double pair (DOUBLE=1) and the 2D pair (DIM=2)
#   tests/ci_suite.sh core      every CPU suite against ./DTFE and ./PS-DTFE
#   tests/ci_suite.sh double    ps_double_check, then the program-running suites against the double
#                               pair (DTFE_TEST_PRECISION=double: float64 grids, tests/precision.sh)
#   tests/ci_suite.sh 2d        tests/ps_2d_check.sh (the 2D pair against analytic 2D flows)
#   tests/ci_suite.sh gui       the launcher's job specifications; the end-to-end window test too when
#                               DTFE_GUI_PYTHON names a Python with PySide6 (QT_QPA_PLATFORM=offscreen)
#   tests/ci_suite.sh oldpy     every .py parses under the oldest supported Python: OLD_PYTHON (CI: setup-python
#                               3.10), else python3.10 / python3.11 on PATH; none found is a FAILURE here
#   tests/ci_suite.sh variants  what users build beyond the pairs: PS-DTFE + DTFE with TBB=0 LIBDEFLATE=0 (the
#                               build without the optional libraries), 'make library' (libDTFE) and the
#                               demo/examples program linked against it; CI runs it as its own job
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
# The standard-DTFE regression compares with the STORED reference (tests/reference/regression_periodic.txt, a
# periodic set without Delaunay ties, to 3e-4 relative: run_tests.py says why) -- until 2026-10-11 it wrote a fresh
# one here and compared nothing. The double pass compares the same reference (the double build sits 2.7e-5 from it).
run_program_suites() {
    step "standard DTFE regression (stored reference)"; env -u DTFE_TEST_REFERENCE_DIR "$PY" tests/run_tests.py
    step "standard DTFE Monte-Carlo average (unbiased, split-invariant)"; tests/dtfe_mc_average_check.sh --no-build
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
    step "partition / padding independence";     tests/ps_partition_padding_check.sh --no-build
    step "periodic seam (exact periodic copies)"; tests/ps_periodic_seam_check.sh --no-build
    step "slab-mode deposit == serial";         tests/ps_slab_deposit_check.sh --no-build
    step "exact mass grid (CPU fixed point)";   tests/ps_mass_exact_check.sh --no-build
    step "exact moment grids (fixed point; thread-count independent)"; tests/ps_fixed_moments_check.sh --no-build
    step "caustic stratification";              tests/ps_caustic_class_check.sh --no-build
    step "tessellation cache";                  tests/ps_tessellation_cache_check.sh --no-build
    step "PS point evaluation";                 tests/ps_point_eval_check.sh --no-build
    step "PS points in the grids' conventions"; tests/ps_point_conventions_check.sh --no-build
    step "standard point evaluation";           tests/dtfe_point_eval_check.sh --no-build
    step "exact point location + --serve";      tests/point_exact_serve_check.sh --no-build
    step "exact hidden-streams flag";           tests/ps_hidden_streams_check.sh --no-build
    step "window == full run";                  tests/ps_window_check.sh --no-build
    step "non-periodic suite";                  "$PY" tests/ps_nonperiodic_test.py --no-build
    step "batch drivers";                       tests/run_scripts_check.sh --no-build
    step "T-web (tidal tensor)";                tests/tweb_check.sh --no-build
    # 2026-10-13: dtfelib's mean-field velocity derivatives (a PS grid given standard DTFE's grad<v> and V-web), checked
    # against the standard binary's own a_velDiv / a_velGrad / V-web and a PS run; single and double pair alike
    step "mean-field velocity derivatives + V-web"; MPLBACKEND=Agg "$PY" tests/py_meanfield_test.py
    step "interlacing (unit, 2D + 3D)";         tests/interlacing_check.sh
    step "composite server split";              tests/serve_split_check.sh --no-build
    step "ordered merge, look-ahead, crash-safe outputs, progress"; tests/ps_runtime_check.sh --no-build
    step "resume of a killed partitioned run";  tests/ps_resume_check.sh --no-build
    step "--fixed-point-merge (order-free integer merge)"; tests/ps_fixed_merge_check.sh --no-build
    step "resume of a killed partitioned standard run"; tests/dtfe_resume_check.sh --no-build
    step "resume with --sample-points";         tests/ps_resume_points_check.sh --no-build
}

# The oldest supported Python (pyproject: >= 3.10) must PARSE every file: a PEP 701 f-string (3.12) is a
# SyntaxError there, and the suites run under a newer interpreter that cannot see it (plot_caustic_skeleton.py,
# review 2026-10-05). Needs a REAL 3.10/3.11 -- ast.parse(feature_version=) under 3.12+ does not catch it.
old_python_parse() {
    local p
    for p in ${OLD_PYTHON:-} python3.10 python3.11; do
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
    [ "${1:-}" != "required" ]          # the oldpy stage: a skip is a failure (2026-10-11)
}

stage_core() {
    unset DTFE_TEST_PRECISION
    step "every .py parses under the oldest supported Python"; old_python_parse
    run_program_suites
    # against the single pair (the double stage does not repeat them): the memory model's predictions against a measured peak,
    # and the out-of-core accumulators bit-identical to in-RAM runs (2026-10-11, ported to Linux)
    step "auto-tune (memory model, budget, splits)"; tests/ps_auto_tune_check.sh --no-build
    # 2026-10-14: the standard binary's tuner too (its report, the counted padded vertices of the run's own split, the
    # user's thread count, the SUMMARY's memory model line). CPU-only here: its GPU half (E with --gpu --exact-average)
    # skips on a build whose o/.build_mode names no GPU, H runs on the DTFE-2d the build stage makes; it pins
    # OMP_NUM_THREADS=10 on purpose (the plans of a 10-core machine), so a 4-core runner only oversubscribes. The one
    # timing-sensitive check is E's measured peak <= 1.35x the prediction (getrusage max RSS on Linux)
    step "standard auto-tune (report, counted split, memory model)"; tests/dtfe_auto_tune_check.sh --no-build
    step "--scratch-dir (out-of-core accumulators)"; tests/ps_scratch_check.sh --no-build
    # AFTER the program suites: ps_caustic_class_check.sh leaves the crossed-waves run (pcc_out.*) under the
    # temp root, which the caustic-skeleton test's real-grid half reads (else it is a SKIP, not a pass)
    step "dtfelib python suite (synthetic tier; the data-backed tier skips without the T7)"; "$PY" tests/py_dtfelib_test.py
    # the suites' own contract (2026-10-08): skips that DTFE_TEST_REQUIRED=1 turns into failures, which the double
    # stage relies on, and run_tests.py's regression SKIP in a fresh reference folder, which it does not
    step "skip / fail contract of run_tests.py and ps_double_check.sh"; tests/run_tests_check.sh --no-build
    # the 2026-10-11 void analyses, one suite each (synthetic; the T7 parts skip)
    step "void catalogue: R_v definitions, cache identity, gate"; MPLBACKEND=Agg "$PY" tests/py_void_core_test.py
    step "void profiles across z + HSW fits";   MPLBACKEND=Agg "$PY" tests/py_profiles_test.py
    step "size function theory, cross-box shapes"; MPLBACKEND=Agg "$PY" tests/py_population_test.py
    step "void environment + caustic control";  MPLBACKEND=Agg "$PY" tests/py_environment_test.py
    step "void fate: particle links + census";  MPLBACKEND=Agg "$PY" tests/py_fate_test.py
    step "void convergence: grid, mass, estimator"; MPLBACKEND=Agg "$PY" tests/py_convergence_test.py
    step "size function on the raw density";    MPLBACKEND=Agg "$PY" tests/py_size_function_test.py
    step "void-in-cloud criteria";               MPLBACKEND=Agg "$PY" tests/py_vic_test.py
    # 2026-10-11: what the README tells a user to type exists in the code, and the drivers' knobs are documented
    step "docs drift (README / COMMANDS.md against the code)"; "$PY" tests/docs_drift_check.py
    # the python analysis stack end to end on a synthetic two-snapshot set (pins its own temp folders; its mirror
    # check needs the thesis mirror ON, so a DTFE_NO_MIRROR from the caller's shell is dropped)
    step "python selftest (synthetic pipeline + the core plot scripts)"; env -u DTFE_NO_MIRROR MPLBACKEND=Agg "$PY" python/selftest.py
}

stage_oldpy() {
    step "every .py parses under the oldest supported Python (required)"; old_python_parse required
}

# Compile-only checks of the configurations the main job does not build (2026-10-11). Run in a fresh checkout:
# the TBB toggle wipes o_ps/ by its stamp, but nothing else here should reuse the pairs' objects.
stage_variants() {
    step "build: PS-DTFE + DTFE without TBB and libdeflate (TBB=0 LIBDEFLATE=0)"
    make DTFE PS-DTFE TBB=0 LIBDEFLATE=0 ARCH_FLAGS="$ARCH" -j"$JOBS"
    ./DTFE --version; ./PS-DTFE --version
    step "build: the shared library (make library)"
    make library ARCH_FLAGS="$ARCH" -j"$JOBS"
    ls -la libDTFE.* || { echo "no libDTFE built"; exit 1; }
    step "build: demo/examples against libDTFE (the library's documented use)"
    make -C demo/examples ARCH_FLAGS="$ARCH"     # a command-line variable reaches its "make library" too (same stamp)
    ls -la demo/examples/DTFE_particle_density
}

# after stage_build the pairs EXIST: a suite that would skip for a missing binary or feature is a failure here
# (DTFE_TEST_REQUIRED=1 turns every skip_or_fail into exit 1; survey rank 20, 2026-10-05; run_tests.py's regression
# compares with the stored reference here too since 2026-10-11)
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
    # the 2026-10-11 launcher additions: their Qt tier runs with PySide6, their pure tier everywhere
    local gp="${DTFE_GUI_PYTHON:-$PY}"
    step "Explore: void inspector, outlines, smoothing, memory guard"; QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}" "$gp" tests/py_gui_explore_test.py
    step "launcher: memory strip, Running now, void catalogues, run files"; QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}" "$gp" tests/py_gui_app2_test.py
}

[ "$#" -gt 0 ] || { sed -n '2,18p' "$0"; exit 2; }
for s in "$@"; do
    case "$s" in
        build|core|double|2d|gui|oldpy|variants) "stage_$s" ;;
        all) stage_build; stage_core; stage_double; stage_2d ;;
        *) echo "unknown stage '$s' (build core double 2d gui oldpy variants all)"; exit 2 ;;
    esac
done
echo; echo "==== ci_suite: $* PASSED"
