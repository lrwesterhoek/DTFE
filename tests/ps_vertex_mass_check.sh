#!/usr/bin/env bash
# PS-DTFE --ps-vertex-mass correctness check.
#
# The flag replaces the default per-tetrahedron mass rho_bar*V_lag with the chart-independent
# sum over the tet's vertices of particleMass/degree(vertex). It exists to remove the
# IC-Lagrangian bias: when the "Lagrangian" input is a PERTURBED IC snapshot (TNG's z=127
# coordinates, which already carry delta_ic = D(z_ic)/D(z) * delta), V_lag varies with the
# perturbation and every density mode comes out filtered by 1 - D(z_ic)/D(z). It is ON BY
# DEFAULT in run_ps_dtfe.sh, so this is the production path. Verified here:
#  A) MASS CONSERVATION with the flag: grid mean rho/rho_bar = 1 on a uniform box AND on the
#     multi-stream pancake (the clean-tessellation cases; the known z=0 TNG deficit comes from
#     dropped degenerate tets, which these snapshots do not have).
#  B) DEGREE-PASS IDEMPOTENCY: the unaveraged and averaged passes share ONE triangulation and
#     each re-runs the vertex-degree count. If the second pass accumulated instead of reset,
#     every mass would double/halve -- so .den AND .a_den must BOTH conserve mass in the same
#     run, and (pancake, nSub=1) .den must equal .a_den bit-for-bit as usual.
#  C) THE BIAS ITSELF: a synthetic perturbed-IC pancake with
#     InitialCoordinates = q + alpha*(x - q), alpha=0.1645 (min-image displacement) must show
#     a density-contrast transfer slope of ~(1-alpha) vs the true-IC run WITHOUT the flag --
#     that is the artefact -- and ~1 WITH it. Measured on a SINGLE-STREAM pancake
#     (amplitude 0.8, no shell crossing) where the linear statement is exact.
#  D) CPU vs GPU parity with the flag (GPU builds only).
#
# Usage: tests/ps_vertex_mass_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"; ALPHA="${ALPHA:-0.1645}"
BIN="./PS-DTFE"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
SNAP_UNI="${TMP}/pvm_input_uniform.hdf5"
SNAP_PAN="${TMP}/pvm_input_pancake.hdf5"
SNAP_LIN="${TMP}/pvm_input_linear.hdf5"       # single-stream pancake (true ICs)
SNAP_PIC="${TMP}/pvm_input_perturbed.hdf5"    # same, ICs displaced by alpha*(x-q)

echo "============================================================"
echo " PS-DTFE --ps-vertex-mass check   N=${N}^3  grid=${GRID}^3  alpha=${ALPHA}"
echo "============================================================"

if [ "${1:-}" != "--no-build" ]; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }
GPU_BUILT=0
[ -f o_ps/.gpu_mode_off ] || GPU_BUILT=1
# capture FIRST -- piping the binary straight into grep -q would SIGPIPE it under pipefail
FULL_HELP="$("${BIN}" --full_help 2>/dev/null || true)"
if ! grep -q -- '--ps-vertex-mass' <<<"${FULL_HELP}"; then
    echo "SKIP: this PS-DTFE has no --ps-vertex-mass support."
    exit 0
fi

echo ">> generating test snapshots ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_UNI}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 0 --jitter-frac 0.3 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_PAN}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_LIN}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 0.8 >/dev/null
SNAP_CRW="${TMP}/pvm_input_crossed.hdf5"
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_CRW}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 --crossed-waves >/dev/null

echo ">> deriving the perturbed-IC snapshot (InitialCoordinates = q + alpha*(x-q)) ..."
"${PY}" - "${SNAP_LIN}" "${SNAP_PIC}" "${ALPHA}" "${BOX}" <<'PY'
import shutil
import sys

import h5py
import numpy as np

src, dst, alpha, box = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
shutil.copyfile(src, dst)
with h5py.File(dst, "r+") as f:
    q = f["PartType1/InitialCoordinates"][...].astype(np.float64)
    x = f["PartType1/Coordinates"][...].astype(np.float64)
    # min-image displacement: x-q wraps at the periodic faces
    d = x - q
    d -= box * np.round(d / box)
    f["PartType1/InitialCoordinates"][...] = ((q + alpha * d) % box).astype(np.float32)
print(f"   perturbed ICs written (max |alpha*d| = {np.abs(alpha*d).max():.3f} Mpc)")
PY

run() {  # $1 = output root, $2 = input snapshot, rest = extra flags
    local out="$1" snap="$2"; shift 2
    rm -f "${out}".*
    local log="${out}.log"
    set +e
    "${BIN}" "${snap}" "${out}" --grid "${GRID}" --field density velocity density_a \
        --input 105 --MpcUnit 1 --verbose 1 --avg-subsamples 1 --periodic "$@" > "$log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "   ERROR: PS-DTFE exited with code $rc -- last 25 lines of $log:"
        tail -n 25 "$log" | sed 's/^/      | /'
        exit 1
    fi
}

echo ">> run 1: uniform box, --ps-vertex-mass"
run "${TMP}/pvm_uni_vm" "${SNAP_UNI}" --ps-vertex-mass
echo ">> run 2: pancake, --ps-vertex-mass"
run "${TMP}/pvm_pan_vm" "${SNAP_PAN}" --ps-vertex-mass
echo ">> run 3: single-stream pancake, true ICs, default masses"
run "${TMP}/pvm_lin_true" "${SNAP_LIN}"
echo ">> run 4: single-stream pancake, perturbed ICs, default masses (the bias)"
run "${TMP}/pvm_lin_pic" "${SNAP_PIC}"
echo ">> run 5: single-stream pancake, perturbed ICs, --ps-vertex-mass (the fix)"
run "${TMP}/pvm_lin_picvm" "${SNAP_PIC}" --ps-vertex-mass
if [ "${GPU_BUILT}" -eq 1 ]; then
    echo ">> run 6: pancake, --ps-vertex-mass --ps-gpu (GPU parity)"
    run "${TMP}/pvm_pan_vmg" "${SNAP_PAN}" --ps-vertex-mass --ps-gpu
else
    echo ">> run 6 skipped (CPU-only build: no GPU parity to check)"
fi
echo ">> run 7: crossed waves (DROPS degenerate cells), --ps-vertex-mass"
run "${TMP}/pvm_crw_vm" "${SNAP_CRW}" --ps-vertex-mass

echo ">> checking the numbers ..."
"${PY}" - "${TMP}" "${GRID}" "${GPU_BUILT}" "${ALPHA}" <<'PY'
import sys

import numpy as np

tmp, grid, gpu, alpha = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
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

# ---------- (A) mass conservation with the flag ----------
for root, tag in (("pvm_uni_vm", "uniform"), ("pvm_pan_vm", "pancake")):
    d = load(f"{tmp}/{root}", ".den")
    check(f"A mass conserved [{tag}]", abs(d.mean() - 1.0) < 1e-4,
          f"mean rho/rho_bar = {d.mean():.6f}")

# ---------- (B) degree-pass idempotency across the shared u/a triangulation ----------
du = load(f"{tmp}/pvm_pan_vm", ".den")
da = load(f"{tmp}/pvm_pan_vm", ".a_den")
check("B a-pass mass conserved too", abs(da.mean() - 1.0) < 1e-4,
      f"mean = {da.mean():.6f} (a non-reset degree pass would double every degree -> halve it)")
# at nSub=1 the sampled u and a deposits are the same arithmetic on the same triangulation
check("B .den == .a_den at nSub=1", np.array_equal(du, da),
      "both passes recount degrees on the SHARED dt; any leak between them breaks this")

# ---------- (C) the IC-Lagrangian bias and its removal ----------
# transfer slope of the perturbed-IC contrast against the true-IC contrast, over covered cells
dt = load(f"{tmp}/pvm_lin_true", ".den") - 1.0
dp = load(f"{tmp}/pvm_lin_pic", ".den") - 1.0
dv = load(f"{tmp}/pvm_lin_picvm", ".den") - 1.0
m = np.abs(dt) > 0.02                      # regress where there is actual contrast
slope_bias = float(np.dot(dp[m], dt[m]) / np.dot(dt[m], dt[m]))
slope_fix  = float(np.dot(dv[m], dt[m]) / np.dot(dt[m], dt[m]))
check("C default masses show the 1-alpha suppression",
      abs(slope_bias - (1.0 - alpha)) < 0.05,
      f"transfer slope = {slope_bias:.4f} (expected ~{1.0-alpha:.4f}) over {int(m.sum())} cells")
check("C --ps-vertex-mass removes it",
      abs(slope_fix - 1.0) < 0.05,
      f"transfer slope = {slope_fix:.4f} (expected ~1)")
check("C the fix beats the bias", abs(slope_fix - 1.0) < abs(slope_bias - 1.0),
      f"|{slope_fix:.4f}-1| < |{slope_bias:.4f}-1|")
# and the perturbed-IC runs conserve mass either way (the bias reshapes contrast, not mass)
check("C perturbed-IC mass conserved (default)", abs((dp + 1).mean() - 1.0) < 1e-4,
      f"mean = {(dp+1).mean():.6f}")
check("C perturbed-IC mass conserved (vertex-mass)", abs((dv + 1).mean() - 1.0) < 1e-4,
      f"mean = {(dv+1).mean():.6f}")

# ---------- (E) mass survives dropped tets: the degree counts only DEPOSITING tets ----------
# Crossed waves at this amplitude drop a handful of degenerate (non-invertible) Eulerian
# cells. Because deg(v) counts only deposit-surviving tets, the dropped tets' vertex shares
# renormalize onto the kept tets and the total stays sum(m_v) -- before this, every dropped
# tet took its shares with it (deficit ~ the dropped fraction; 0.74% at TNG z=0, 5.8e-4 here).
dc = load(f"{tmp}/pvm_crw_vm", ".den")
check("E crossed-waves mass survives dropped tets", abs(dc.mean() - 1.0) < 5e-5,
      f"mean = {dc.mean():.8f} (pre-fix binaries: ~0.99942)")

# ---------- (D) GPU parity with the flag ----------
if gpu:
    dc = load(f"{tmp}/pvm_pan_vm", ".den")
    dg = load(f"{tmp}/pvm_pan_vmg", ".den")
    sc = np.fromfile(f"{tmp}/pvm_pan_vm.streams", dtype=np.float32)
    sg = np.fromfile(f"{tmp}/pvm_pan_vmg.streams", dtype=np.float32)
    vc = load(f"{tmp}/pvm_pan_vm", ".vel", 3)
    vg = load(f"{tmp}/pvm_pan_vmg", ".vel", 3)
    eq = float((sc == sg).mean())
    mrel_d = float(np.abs(dc - dg).mean() / (np.abs(dc).max() + 1e-30))
    mrel_v = float(np.abs(vc - vg).mean() / (np.abs(vc).max() + 1e-30))
    check("D GPU streams match CPU", eq > 0.999, f"{eq*100:.4f}% equal")
    check("D GPU density matches CPU", mrel_d < 1e-4, f"mean rel = {mrel_d:.3e}")
    check("D GPU velocity matches CPU", mrel_v < 1e-4, f"mean rel = {mrel_v:.3e}")
else:
    print("   SKIP D GPU parity (CPU-only build)")

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (--ps-vertex-mass conserves mass, keeps the shared-triangulation degree"
      " pass idempotent, and removes the perturbed-IC contrast suppression)")
PY

echo "============================================================"
echo " vertex-mass check PASSED"
echo "============================================================"
