#!/usr/bin/env bash
# CPU vs GPU (--gpu) parity check for the standard-DTFE method-1 '_a' interpolation.
#
# Runs ./DTFE twice on a synthetic Zel'dovich snapshot -- once on the CPU, once with
# --gpu -- for both the single-box and the partitioned path, and compares the raw
# output grids. The GPU deposit is expected to match to float rounding only (atomic
# summation order, float vs double sample placement), so the tolerances mirror
# tests/ps_parallel_check.sh: means must agree to ~1e-5 relative, per-cell differences
# to max|field| must stay below REL_TOL outside a tiny borderline-cell fraction.
#
# Requires a Metal build; fails loudly if the binary fell back to the CPU (then the two
# runs would be identical, which defeats the test). The backend is taken from the
# current build mode (o/.build_mode), the GPU_BUILD env var, or defaults to METAL=1.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1

PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14

TMP="$ROOT/tests/tmp/metal_check"
mkdir -p "$TMP"

# this test REQUIRES a GPU build: reuse the current mode if it is a GPU one, else default
GPU_BUILD="${GPU_BUILD:-$(cat o/.build_mode 2>/dev/null || true)}"
[ -z "${GPU_BUILD}" ] && GPU_BUILD="METAL=1"
echo ">> building DTFE ${GPU_BUILD} (test requires a GPU build)"
make DTFE "${GPU_BUILD}" -j4 >/dev/null || { echo "FAIL build"; exit 1; }

SNAP="$TMP/xcheck.hdf5"
[ -f "$SNAP" ] || "$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 48 --box 100 \
    --amplitude-factor 1.8 --jitter-frac 0.02 --seed 42 >/dev/null || { echo "FAIL data gen"; exit 1; }

FIELDS="density_a velocity_a gradient_a"
COMMON=(--grid 96 --periodic --input 105 --MpcUnit 1 --field $FIELDS)

run() { # run <label> <outprefix> [extra args...]
    local label="$1" out="$2"; shift 2
    ./DTFE "$SNAP" "$out" "${COMMON[@]}" "$@" > "$out.log" 2>&1 \
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

echo ">> partitioned (2 2 2)"
run cpu-part  "$TMP/p_cpu" --partition 2 2 2 --max-concurrent 2
run gpu-part  "$TMP/p_gpu" --partition 2 2 2 --max-concurrent 2 --gpu
assert_gpu "partitioned --gpu" "$TMP/p_gpu.log"

"$PY" - "$TMP" <<'EOF'
import numpy as np, sys
tmp = sys.argv[1]
REL_TOL   = 5e-2    # per-cell ceiling (borderline tets flip a sample across a cell wall)
FRAC_TOL  = 1e-3    # at most this fraction of cells may exceed 1e-3 relative
MEAN_TOL  = 1e-5    # field means must agree this closely (relative to max|field|)
ok = True
for pre in ("s", "p"):
    for suf in ("a_den", "a_vel", "a_velGrad"):
        c = np.fromfile(f"{tmp}/{pre}_cpu.{suf}", dtype=np.float32)
        g = np.fromfile(f"{tmp}/{pre}_gpu.{suf}", dtype=np.float32)
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
sys.exit(0 if ok else 1)
EOF
rc=$?
[ $rc -eq 0 ] && echo "dtfe_metal_check: ALL PASS" || echo "dtfe_metal_check: FAILURES"
exit $rc
