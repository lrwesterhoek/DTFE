#!/usr/bin/env bash
# CPU vs GPU (--gpu) parity check for the standard-DTFE method-1 '_a' interpolation.
#
# Runs ./DTFE on a synthetic Zel'dovich snapshot -- on the CPU and with --gpu -- for both
# the single-box and the partitioned path, and compares the raw output grids. The GPU
# deposit is expected to match to float rounding only (atomic summation order, float vs
# double sample placement), so the tolerances mirror tests/ps_parallel_check.sh: means must
# agree to ~1e-5 relative, per-cell differences to max|field| must stay below REL_TOL
# outside a tiny borderline-cell fraction. The single box also runs the GPU twice (two runs
# may differ by atomic summation order only, REPEAT_TOL -- a watchdog retry that let stray
# deposits through would show as whole samples, ~1e-2) and once with the per-tetrahedron
# kernel (DTFE_GPU_ITEMS=0), which places the same samples with the same float arithmetic and
# so must agree with the work-item kernel to the same REPEAT_TOL.
#
# Requires a Metal build; fails loudly if the binary fell back to the CPU (then the two
# runs would be identical, which defeats the test). The backend is taken from the
# current build mode (o/.build_mode), the GPU_BUILD env var, or defaults to METAL=1.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair (built by make ... DOUBLE=1)

PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14

TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/metal_check"
mkdir -p "$TMP"

# this test REQUIRES a GPU build: reuse the current mode if it is a GPU one, else default
GPU_BUILD="${GPU_BUILD:-$(cat o/.build_mode 2>/dev/null || true)}"
[ -z "${GPU_BUILD}" ] && GPU_BUILD="METAL=1"
echo ">> building DTFE ${GPU_BUILD} (test requires a GPU build)"
# shellcheck disable=SC2086  # the build mode may be two words (METAL=1 DOUBLE=1): word-split it
if precision_double; then
    precision_require "${DTFE_BIN}"
    [ -f o_d/.gpu_mode_off ] && { echo "SKIP: the double pair was built without a GPU backend"; exit 0; }
else
    make DTFE ${GPU_BUILD} -j4 >/dev/null || { echo "FAIL build"; exit 1; }
fi

SNAP="$TMP/xcheck.hdf5"
[ -f "$SNAP" ] || "$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 48 --box 100 \
    --amplitude-factor 1.8 --jitter-frac 0.02 --seed 42 >/dev/null || { echo "FAIL data gen"; exit 1; }

FIELDS="density_a velocity_a gradient_a"
COMMON=(--grid 96 --periodic --input 105 --MpcUnit 1 --field $FIELDS)

run() { # run <label> <outprefix> [extra args...]
    local label="$1" out="$2"; shift 2
    "${DTFE_BIN}" "$SNAP" "$out" "${COMMON[@]}" "$@" > "$out.log" 2>&1 \
        || { echo "FAIL run $label (see $out.log)"; exit 1; }
}

# The '<Backend> GPU (' dispatch line prints BEFORE the deposit call, so its presence alone
# does not prove the GPU ran; the fallback warning ('...; using the CPU interpolation') is
# printed on every failure path, so its absence is the actual success criterion.
assert_gpu() { # assert_gpu <label> <log>
    grep -q " GPU (" "$2" || { echo "FAIL: $1 run did not attempt the GPU"; exit 1; }
    grep -q "using the CPU interpolation" "$2" && { echo "FAIL: $1 run fell back to the CPU at runtime"; exit 1; }
    return 0
}

echo ">> single box"
run cpu       "$TMP/s_cpu"
run gpu       "$TMP/s_gpu" --gpu
assert_gpu "--gpu" "$TMP/s_gpu.log"
run gpu-again "$TMP/s_gpu2" --gpu            # run-to-run: only atomic summation order may differ
assert_gpu "--gpu (second run)" "$TMP/s_gpu2.log"
echo ">> single box, per-tetrahedron kernel (DTFE_GPU_ITEMS=0: the A/B reference kernel)"
DTFE_GPU_ITEMS=0 run gpu-pertet "$TMP/s_gpu0" --gpu
assert_gpu "--gpu per-tet kernel" "$TMP/s_gpu0.log"
# A coarse grid makes most tetrahedra single-cell: the fast path, whose cell index the host
# shifts into the shared full grid (a 512^3 run once put those in the wrong cells while this
# test's 96^3 grid, with hardly any single-cell tetrahedron, stayed green). The shared grid
# (DTFE_GPU_SHARED=1, off by default: the per-sub-domain grids measured faster) is compared
# with the default per-call path and the CPU.
echo ">> coarse 32^3 grid: the single-cell fast path through the shared grid"
COARSE=(--grid 32 --periodic --input 105 --MpcUnit 1 --field $FIELDS)
"${DTFE_BIN}" "$SNAP" "$TMP/c_cpu" "${COARSE[@]}" > "$TMP/c_cpu.log" 2>&1 || { echo "FAIL run coarse cpu"; exit 1; }
DTFE_GPU_SHARED=1 DTFE_METAL_TIMING=1 "${DTFE_BIN}" "$SNAP" "$TMP/c_gpu" "${COARSE[@]}" --gpu > "$TMP/c_gpu.log" 2>&1 || { echo "FAIL run coarse gpu"; exit 1; }
assert_gpu "coarse --gpu shared" "$TMP/c_gpu.log"
"${DTFE_BIN}" "$SNAP" "$TMP/c_gpu1" "${COARSE[@]}" --gpu > "$TMP/c_gpu1.log" 2>&1 || { echo "FAIL run coarse gpu per-call"; exit 1; }
assert_gpu "coarse --gpu per-call" "$TMP/c_gpu1.log"
if grep -q "processors will be used\|processors available" "$TMP/c_gpu.log" && ! grep -q "shared grid:" "$TMP/c_gpu.log"; then
    echo "FAIL: the multi-threaded coarse --gpu run did not use the shared grid"; exit 1
fi

echo ">> partitioned (2 2 2)"
run cpu-part  "$TMP/p_cpu" --partition 2 2 2 --max-concurrent 2
run gpu-part  "$TMP/p_gpu" --partition 2 2 2 --max-concurrent 2 --gpu
assert_gpu "partitioned --gpu" "$TMP/p_gpu.log"
# --exact-average: the analytic volume average (r3d) on the GPU (depositExactAverage) against the
# CPU's double-precision r3d -- no sampling at all, so this pair should sit well inside the sampled
# tolerances (only the float clipping and the atomic order differ)
echo ">> exact volume average (--exact-average)"
run cpu-exact "$TMP/e_cpu" --exact-average
run gpu-exact "$TMP/e_gpu" --exact-average --gpu
assert_gpu "--exact-average --gpu" "$TMP/e_gpu.log"
# a FINE region (the launcher's exact zoom: 512^2 cells of 1.95 kpc): the GPU clips each piece in a
# frame near the piece -- in the tetrahedron's vertex-0 frame a kpc piece Mpc away lost its volume
# to float cancellation. Same tolerance as the box above, the cells are 260x smaller.
FINE=(--grid 512 512 1 --periodic --input 105 --MpcUnit 1 --field density_a velocity_a --exact-average --regionMpc 40 41 40 41 50 50.002)
"${DTFE_BIN}" "$SNAP" "$TMP/f_cpu" "${FINE[@]}" > "$TMP/f_cpu.log" 2>&1 || { echo "FAIL run fine region cpu"; exit 1; }
"${DTFE_BIN}" "$SNAP" "$TMP/f_gpu" "${FINE[@]}" --gpu > "$TMP/f_gpu.log" 2>&1 || { echo "FAIL run fine region gpu"; exit 1; }
assert_gpu "--exact-average --gpu on a fine region" "$TMP/f_gpu.log"

"$PY" - "$TMP" <<'EOF'
import numpy as np, sys
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
tmp = sys.argv[1]
REL_TOL   = 5e-2    # per-cell ceiling (borderline tets flip a sample across a cell wall)
FRAC_TOL  = 1e-3    # at most this fraction of cells may exceed 1e-3 relative
MEAN_TOL  = 1e-5    # field means must agree this closely (relative to max|field|)
REPEAT_TOL = 1e-4   # two GPU runs: atomic summation order only (a stray deposit would be ~1e-2)
ok = True
for pre in ("s", "p", "e"):
    for suf in ("a_den", "a_vel", "a_velGrad"):
        c = np.fromfile(f"{tmp}/{pre}_cpu.{suf}", dtype=REAL)
        g = np.fromfile(f"{tmp}/{pre}_gpu.{suf}", dtype=REAL)
        if c.shape != g.shape:
            print(f"FAIL {pre}.{suf}: shape {c.shape} vs {g.shape}"); ok = False; continue
        scale = float(np.max(np.abs(c))) or 1.0
        d = np.abs(c - g)
        meandiff = abs(float(c.mean()) - float(g.mean())) / scale
        frac = float((d > 1e-3 * scale).sum()) / c.size
        peak = float(d.max()) / scale
        line = f"{pre}.{suf:10s} mean-rel={meandiff:.2e} peak-rel={peak:.2e} frac>1e-3={frac:.2e}"
        if meandiff > MEAN_TOL or peak > REL_TOL or frac > FRAC_TOL:
            print("FAIL " + line); ok = False
        else:
            print("PASS " + line)
        if pre == "s":   # the item kernel repeated, and the per-tet kernel, against the first GPU run
            # both kernels place the same samples with the same float arithmetic (measured 2e-6 apart),
            # so like the repeat only the atomic summation order may differ
            for other, what in (("s_gpu2", "repeat"), ("s_gpu0", "per-tet kernel")):
                h = np.fromfile(f"{tmp}/{other}.{suf}", dtype=REAL)
                peak2 = float(np.abs(g - h).max()) / scale
                line = f"gpu vs {what:14s} {suf:10s} peak-rel={peak2:.2e}"
                if peak2 > REPEAT_TOL:
                    print("FAIL " + line); ok = False
                else:
                    print("PASS " + line)
for suf in ("a_den", "a_vel"):                  # the fine region (kpc cells): the clip frame must not cancel
    c = np.fromfile(f"{tmp}/f_cpu.{suf}", dtype=REAL).astype(float)
    g = np.fromfile(f"{tmp}/f_gpu.{suf}", dtype=REAL).astype(float)
    scale = float(np.max(np.abs(c))) or 1.0
    peak = float(np.abs(c - g).max()) / scale
    line = f"fine-region.{suf:10s} peak-rel={peak:.2e} (1.95 kpc cells; < 1e-4)"
    if peak > 1e-4:
        print("FAIL " + line); ok = False
    else:
        print("PASS " + line)
for suf in ("a_den", "a_vel", "a_velGrad"):     # the coarse grid: fast-path cells through the shared grid
    c = np.fromfile(f"{tmp}/c_cpu.{suf}", dtype=REAL)
    g = np.fromfile(f"{tmp}/c_gpu.{suf}", dtype=REAL)
    h = np.fromfile(f"{tmp}/c_gpu1.{suf}", dtype=REAL)
    scale = float(np.max(np.abs(c))) or 1.0
    d = np.abs(c - g)
    line = (f"c.{suf:10s} mean-rel={abs(float(c.mean()) - float(g.mean())) / scale:.2e} peak-rel={float(d.max()) / scale:.2e} "
            f"frac>1e-3={float((d > 1e-3 * scale).sum()) / c.size:.2e} shared-vs-percall={float(np.abs(g - h).max()) / scale:.2e}")
    if (abs(float(c.mean()) - float(g.mean())) / scale > MEAN_TOL or float(d.max()) / scale > REL_TOL
            or float((d > 1e-3 * scale).sum()) / c.size > FRAC_TOL or float(np.abs(g - h).max()) / scale > REPEAT_TOL):
        print("FAIL " + line); ok = False
    else:
        print("PASS " + line)
sys.exit(0 if ok else 1)
EOF
rc=$?
[ $rc -eq 0 ] && echo "dtfe_metal_check: ALL PASS" || echo "dtfe_metal_check: FAILURES"
exit $rc
