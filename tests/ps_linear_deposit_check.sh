#!/usr/bin/env bash
# PS-DTFE --ps-linear-deposit correctness check.
#
# The linear deposit weights each tetrahedron's interior sub-samples by the DTFE-interpolated
# linear density profile, RENORMALIZED per tetrahedron so the deposited total still equals
# the tetrahedron's mass exactly. Verified here:
#  A) UNIFORM box (amp=0): the vertex densities are constant, so the linear weights are
#     constant and the linear deposit must reproduce the uniform deposit to float rounding.
#  B) PANCAKE (multi-stream): mass conservation is unchanged (grid mean of rho/rho_bar = 1
#     to float rounding, at nSub=1 AND nSub=3); the uniform and linear grids differ smoothly
#     (identical .streams, high correlation, bounded max ratio -- no new caustic spikes).
#  C) GPU parity (GPU builds only): the GPU linear deposit matches the CPU linear deposit
#     within the usual float32 atomic tolerances.
#
# Usage: tests/ps_linear_deposit_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"
BIN="./PS-DTFE"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
SNAP_UNI="${TMP}/pld_input_uniform.hdf5"
SNAP_PAN="${TMP}/pld_input_pancake.hdf5"

echo "============================================================"
echo " PS-DTFE --ps-linear-deposit check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

if [ "${1:-}" != "--no-build" ]; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }
GPU_BUILT=0
[ -f o_ps/.gpu_mode_off ] || GPU_BUILT=1

echo ">> generating test snapshots ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_UNI}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 0 --jitter-frac 0.3 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_PAN}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null

run() {  # $1 = output root, rest = extra args
    local out="$1"; shift
    rm -f "${out}".*
    local log="${out}.log"
    set +e
    "${BIN}" "$@" "${out}" --grid "${GRID}" --field density velocity density_a \
        --input 105 --MpcUnit 1 --verbose 1 > "$log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "   ERROR: PS-DTFE exited with code $rc -- last 25 lines of $log:"
        tail -n 25 "$log" | sed 's/^/      | /'
        exit 1
    fi
}

echo ">> run 1: uniform box, uniform deposit"
run "${TMP}/pld_uni_u" "${SNAP_UNI}" --periodic
echo ">> run 2: uniform box, --ps-linear-deposit"
run "${TMP}/pld_uni_l" "${SNAP_UNI}" --periodic --ps-linear-deposit
echo ">> run 3: pancake, uniform deposit"
run "${TMP}/pld_pan_u" "${SNAP_PAN}" --periodic
echo ">> run 4: pancake, --ps-linear-deposit"
run "${TMP}/pld_pan_l" "${SNAP_PAN}" --periodic --ps-linear-deposit
if [ "${GPU_BUILT}" -eq 1 ]; then
    echo ">> run 5: pancake, --ps-linear-deposit --ps-gpu (GPU parity)"
    run "${TMP}/pld_pan_lg" "${SNAP_PAN}" --periodic --ps-linear-deposit --ps-gpu
else
    echo ">> run 5 skipped (CPU-only build: no GPU parity to check)"
fi

echo ">> checking the numbers ..."
"${PY}" - "${TMP}" "${GRID}" "${GPU_BUILT}" <<'PY'
import sys
import numpy as np

tmp, grid, gpu = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ncell = grid ** 3
fails = []

def check(name, ok, detail):
    print(f"   {'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)

def load(root, ext, ncomp=1):
    d = np.fromfile(root + ext, dtype=np.float32).astype(np.float64)
    assert d.size == ncell * ncomp, (root + ext, d.size)
    return d

# ---------- (A) uniform box: linear == uniform to float rounding ----------
for ext in (".den", ".a_den", ".vel"):
    nc = 3 if ext == ".vel" else 1
    u = load(f"{tmp}/pld_uni_u", ext, nc)
    l = load(f"{tmp}/pld_uni_l", ext, nc)
    rel = np.abs(u - l).max() / (np.abs(u).max() + 1e-30)
    check(f"A uniform{ext} linear==uniform", rel < 1e-5, f"max rel = {rel:.3e}")

# ---------- (B) pancake: mass conservation + smooth A/B differences ----------
for ext, tag in ((".den", "nSub=1"), (".a_den", "nSub=3")):
    dl = load(f"{tmp}/pld_pan_l", ext)
    check(f"B mass conserved [{tag}]", abs(dl.mean() - 1.0) < 1e-4,
          f"mean rho/rho_bar = {dl.mean():.6f}")
du = load(f"{tmp}/pld_pan_u", ".den")
dl = load(f"{tmp}/pld_pan_l", ".den")
su = np.fromfile(f"{tmp}/pld_pan_u.streams", dtype=np.float32)
sl = np.fromfile(f"{tmp}/pld_pan_l.streams", dtype=np.float32)
check("B streams identical", np.array_equal(su, sl),
      "the weighting must not change which samples deposit")
ratio = dl.max() / du.max()
check("B no new caustic spikes", 0.2 < ratio < 5.0,
      f"max(linear)/max(uniform) = {ratio:.3f}")
m = (du > 0) & (dl > 0)
corr = np.corrcoef(np.log(du[m]), np.log(dl[m]))[0, 1]
check("B grids differ smoothly", corr > 0.98, f"log-density correlation = {corr:.5f}")
big = np.mean(np.abs(np.log(np.maximum(dl[m], 1e-12) / np.maximum(du[m], 1e-12))) > np.log(2.0))
check("B few cells change >2x", big < 0.02, f"{big*100:.3f}% of covered cells")

# ---------- (C) GPU parity for the linear deposit ----------
if gpu:
    dg = load(f"{tmp}/pld_pan_lg", ".den")
    sg = np.fromfile(f"{tmp}/pld_pan_lg.streams", dtype=np.float32)
    eq = np.mean(sl == sg)
    scale = np.abs(dl).max()
    mrel = np.abs(dl - dg).mean() / scale
    check("C GPU streams match CPU", eq > 0.999, f"{eq*100:.4f}% equal")
    check("C GPU density matches CPU", mrel < 1e-4, f"mean rel = {mrel:.3e}")
else:
    print("   SKIP C GPU parity (CPU-only build)")

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (--ps-linear-deposit conserves mass exactly and only reshapes the"
      " sub-tetrahedron distribution)")
PY

echo "============================================================"
echo " linear-deposit check PASSED"
echo "============================================================"
