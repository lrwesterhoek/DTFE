#!/bin/bash
# GPU scalar-field parity check, both binaries: the 'scalar_a' and 'scalarGradient_a' grids of a
# per-particle dataset ('--scalar-dataset', a smooth function of position written into the test
# snapshot) must agree between the CPU and the GPU deposit the way the velocity and its gradient do.
# Builds with ONE scalar component (the default, NO_SCALARS=1): the GPU declines other builds.
#   DTFE    sampled '_a' interpolation: the item kernel (default), the per-tet kernel (DTFE_GPU_ITEMS=0)
#   PS-DTFE the sampled deposit (nSub 2) and the exact deposit (--ps-exact-deposit)
# Tolerance = the parity class of the velocity fields on the same box: values (scalar, density,
# velocity) mean 1e-3 and per-cell 5e-2 beyond a 1e-3 fraction; GRADIENTS at caustics are a
# float-sensitive quantity (the inverse edge matrix of a tetrahedron near a fold amplifies the
# float rounding of its edges by |rows|^3/|det|, up to ~1e6 at the degeneracy cut), so the two
# gradients are held to the class the phase-space VELOCITY gradient has shown since the GPU deposit
# exists: the mean of |g| within 5e-2 (3-4% on this folded box, far less on standard DTFE), per-cell
# 5e-2 beyond a 2e-3 fraction. The scalar gradient comes out tighter than the velocity gradient.
# Skipped (exit 0) when the binaries carry no GPU stamp. --no-build is accepted and ignored.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${DTFE_BIN}" "${PS_BIN}"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
for b in DTFE PS-DTFE; do [ -x "./$b" ] || { echo "FAIL: ./$b is not built"; exit 1; }; done
OBJS="o o_ps"; precision_double && OBJS="o_d o_ps_d"
for d in ${OBJS}; do
    if [ ! -d "$d" ] || [ -f "$d/.gpu_mode_off" ]; then echo "SKIP: $d carries no GPU stamp (build with METAL=1 / CUDA=1 / HIP=1)"; exit 0; fi
done
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/gpu_scalar_check"
mkdir -p "$TMP"
SNAP="$TMP/waves.hdf5"
[ -f "$SNAP" ] || "$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 32 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 5 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
"$PY" - "$SNAP" <<'PYEOF' || { echo "FAIL scalar dataset"; exit 1; }
import h5py, numpy as np, sys
with h5py.File(sys.argv[1], "a") as f:
    p = f["PartType1"]
    if "Phi" not in p:
        x = p["Coordinates"][:]
        phi = np.sin(2*np.pi*x[:, 0]/100.0) * np.cos(2*np.pi*x[:, 1]/100.0) + 0.3*x[:, 2]/100.0
        p["Phi"] = phi.astype(np.float32)
PYEOF
G=32
FIELDS=(--field density_a velocity_a gradient_a scalar_a scalarGradient_a --scalar-dataset Phi)
COMMON=(--grid $G --periodic --input 105 --MpcUnit 1 --verbose 1 "${FIELDS[@]}")
run() { local out="$1"; shift; "$@" > "$out.log" 2>&1 || { echo "FAIL: $* (see $out.log)"; exit 1; }
        grep -q "using the CPU interpolation\|using the CPU deposit\|deposit unavailable" "$out.log" && { echo "FAIL: $out fell back to the CPU"; exit 1; }; return 0; }
echo ">> DTFE: CPU, GPU item kernel, GPU per-tet kernel"
"${DTFE_BIN}" "$SNAP" "$TMP/d_cpu" "${COMMON[@]}" > "$TMP/d_cpu.log" 2>&1 || { echo "FAIL: DTFE cpu"; exit 1; }
run "$TMP/d_gpu"  "${DTFE_BIN}" "$SNAP" "$TMP/d_gpu" "${COMMON[@]}" --gpu
run "$TMP/d_tet"  env DTFE_GPU_ITEMS=0 "${DTFE_BIN}" "$SNAP" "$TMP/d_tet" "${COMMON[@]}" --gpu
echo ">> PS-DTFE: CPU and GPU, sampled (nSub 2) and exact"
"${PS_BIN}" "$SNAP" "$TMP/p_cpu" "${COMMON[@]}" --avg-subsamples 2 > "$TMP/p_cpu.log" 2>&1 || { echo "FAIL: PS-DTFE cpu"; exit 1; }
run "$TMP/p_gpu" "${PS_BIN}" "$SNAP" "$TMP/p_gpu" "${COMMON[@]}" --avg-subsamples 2 --ps-gpu
"${PS_BIN}" "$SNAP" "$TMP/e_cpu" "${COMMON[@]}" --ps-exact-deposit > "$TMP/e_cpu.log" 2>&1 || { echo "FAIL: PS-DTFE exact cpu"; exit 1; }
run "$TMP/e_gpu" "${PS_BIN}" "$SNAP" "$TMP/e_gpu" "${COMMON[@]}" --ps-exact-deposit --ps-gpu
"$PY" - "$TMP" <<'PYEOF'
import numpy as np, sys
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
from pathlib import Path
tmp = Path(sys.argv[1]); fails = 0
GRAD = ("a_scalarGrad", "a_velGrad")
def check(ref, got, name):
    global fails
    a = np.fromfile(tmp / f"{ref}.{name}", dtype=REAL); b = np.fromfile(tmp / f"{got}.{name}", dtype=REAL)
    sc = max(float(np.abs(a).max()), 1e-30); d = np.abs(a - b)
    grad = name in GRAD
    tolMean, tolPeak, tolFrac = (5e-2, 5e-2, 2e-3) if grad else (1e-3, 5e-2, 1e-3)
    # gradients have a near-zero mean: compare the mean magnitudes instead
    m = abs(np.abs(a).mean() - np.abs(b).mean()) / max(np.abs(a).mean(), 1e-30) if grad \
        else abs(a.mean() - b.mean()) / max(abs(a.mean()), 1e-3 * sc)
    p = d.max() / sc; fr = float(np.mean(d > 1e-3 * sc))
    ok = m < tolMean and (p < tolPeak or fr < tolFrac)
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {got:6s} {name:13s} mean{'|abs|' if grad else ''}-rel={m:.2e} peak-rel={p:.2e} frac>1e-3={fr:.2e}")
for got in ("d_gpu", "d_tet"):
    for name in ("a_scalar", "a_scalarGrad", "a_den", "a_vel", "a_velGrad"):
        check("d_cpu", got, name)
for ref, got in (("p_cpu", "p_gpu"), ("e_cpu", "e_gpu")):
    for name in ("a_scalar", "a_scalarGrad", "a_den", "a_vel", "a_velGrad"):
        check(ref, got, name)
# the scalar must not be a copy of something else: it has to vary and track its definition's range
s = np.fromfile(tmp / "d_gpu.a_scalar", dtype=REAL)
ok = -1.3 < s.min() < -0.5 and 0.5 < s.max() < 1.6
fails += not ok
print(f"  {'PASS' if ok else 'FAIL'} the GPU scalar grid spans the dataset's range ({s.min():.2f} .. {s.max():.2f})")
print("gpu_scalar_check:", "ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
PYEOF
