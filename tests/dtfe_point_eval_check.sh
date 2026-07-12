#!/usr/bin/env bash
# STANDARD-DTFE --sample-points check (the sibling of ps_point_eval_check.sh).
#
# The standard binary shares the PS point-evaluation machinery (ps_point_eval.cc) but
# evaluates the EULERIAN tessellation: exactly one containing tetrahedron per point, so
# '.pts_streams' is a 0/1 coverage flag and '.pts_velDisp' is identically zero. Verified:
#  A) uniform jittered lattice, cell-centre points: full coverage, stream count exactly 1,
#     density ~1 (DTFE estimator level), velocity and dispersion exactly 0.
#  B) pancake, standard vs PS-DTFE at the same cell centres: single-stream point values
#     agree at the tolerances of ps_standard_cross_check.py (velocity median rel <= 1e-3,
#     density median rel <= 0.15 -- the two tessellations differ, so agreement is at
#     estimator level, not exact).
#  C) text vs binary point-file readers give byte-identical outputs.
#  D) --pts-den-grad: adding the flag leaves the other outputs byte-identical; on a
#     jitter-free pancake the transverse gradient vanishes and grad_x matches a central
#     finite difference of '.pts_den' at x +- eps probe points.
#  E) clear rejections: --per-stream / --per-stream-ids (PS-only) and --partition (the
#     standard point evaluation runs on a single triangulation).
#
# Usage: tests/dtfe_point_eval_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"
BIN="./DTFE"; PSBIN="./PS-DTFE"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
SNAP_UNI="${TMP}/dpe_input_uniform.hdf5"
SNAP_PAN="${TMP}/dpe_input_pancake.hdf5"
SNAP_PAN0="${TMP}/dpe_input_pancake0.hdf5"
PTS_BIN="${TMP}/dpe_centres.bin"
PTS_TXT="${TMP}/dpe_centres.txt"
PTS_FD="${TMP}/dpe_fd_probes.bin"
SEL_FD="${TMP}/dpe_fd_selection.txt"

echo "============================================================"
echo " standard-DTFE --sample-points check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

if [ "${1:-}" != "--no-build" ]; then
    echo ">> building DTFE ..."
    BUILD_MODE="$(cat o/.build_mode 2>/dev/null || true)"
    make DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

# capture FIRST -- piping the binary straight into grep -q would SIGPIPE it under pipefail
FULL_HELP="$("${BIN}" --full_help 2>/dev/null || true)"
if ! grep -q -- '--sample-points' <<<"${FULL_HELP}"; then
    echo "SKIP: this DTFE has no --sample-points support."
    exit 0
fi
HAVE_PS=0
[ -x "${PSBIN}" ] && HAVE_PS=1

echo ">> generating test snapshots + cell-centre points ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_UNI}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 0 --jitter-frac 0.3 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_PAN}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_PAN0}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 --jitter-frac 0 >/dev/null
"${PY}" - "${GRID}" "${BOX}" "${PTS_BIN}" "${PTS_TXT}" <<'PY'
import sys
import numpy as np
grid, box, fbin, ftxt = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
c = (np.arange(grid) + 0.5) * box / grid
x, y, z = np.meshgrid(c, c, c, indexing="ij")
pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
pts.astype(np.float64).tofile(fbin)
np.savetxt(ftxt, pts, fmt="%.17g %.17g %.17g")
PY

run() {  # $1 = binary, $2 = output root, $3 = input snapshot, rest = extra flags
    local bin="$1" out="$2" snap="$3"; shift 3
    rm -f "${out}".*
    local log="${out}.log"
    set +e
    "${bin}" "${snap}" "${out}" --grid "${GRID}" --field density velocity \
        --input 105 --MpcUnit 1 --verbose 1 "$@" > "$log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "   ERROR: $(basename "${bin}") exited with code $rc -- last 25 lines of $log:"
        tail -n 25 "$log" | sed 's/^/      | /'
        exit 1
    fi
}

echo ">> run 1: uniform lattice, standard binary, cell centres (binary points)"
run "${BIN}" "${TMP}/dpe_uni" "${SNAP_UNI}" --periodic --sample-points "${PTS_BIN}"
echo ">> run 2: uniform lattice, standard binary, cell centres (TEXT points)"
run "${BIN}" "${TMP}/dpe_uni_txt" "${SNAP_UNI}" --periodic --sample-points "${PTS_TXT}"
echo ">> run 3: pancake, standard binary, cell centres"
run "${BIN}" "${TMP}/dpe_pan" "${SNAP_PAN}" --periodic --sample-points "${PTS_BIN}"
if [ "${HAVE_PS}" -eq 1 ]; then
    echo ">> run 4: pancake, PS-DTFE, same cell centres (cross-estimator reference)"
    run "${PSBIN}" "${TMP}/dpe_pan_ps" "${SNAP_PAN}" --periodic --sample-points "${PTS_BIN}"
else
    echo ">> run 4 skipped (no PS-DTFE binary: cross-estimator section will be skipped)"
fi
echo ">> run 5: jitter-free pancake, standard binary, --pts-den-grad"
run "${BIN}" "${TMP}/dpe_grad" "${SNAP_PAN0}" --periodic --sample-points "${PTS_BIN}" --pts-den-grad
echo ">> run 6: jitter-free pancake, standard binary (no gradient flag)"
run "${BIN}" "${TMP}/dpe_grad_off" "${SNAP_PAN0}" --periodic --sample-points "${PTS_BIN}"

echo ">> (C) text vs binary point files give byte-identical outputs"
for ext in .pts_den .pts_vel .pts_velDisp .pts_streams; do
    cmp -s "${TMP}/dpe_uni${ext}" "${TMP}/dpe_uni_txt${ext}" \
        || { echo "FAIL: ${ext} differs between text and binary point input"; exit 1; }
done
echo "   OK"
echo ">> (D) --pts-den-grad leaves the existing outputs byte-identical"
for ext in .pts_den .pts_vel .pts_velDisp .pts_streams; do
    cmp -s "${TMP}/dpe_grad_off${ext}" "${TMP}/dpe_grad${ext}" \
        || { echo "FAIL: ${ext} changed when --pts-den-grad was added"; exit 1; }
done
echo "   OK"

# finite-difference probes: single-stream points with meaningful |grad_x|, x +- eps
"${PY}" - "${TMP}" "${GRID}" "${BOX}" "${PTS_FD}" "${SEL_FD}" <<'PY'
import sys
import numpy as np
tmp, grid, box, fdbin, fdsel = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
st = np.fromfile(f"{tmp}/dpe_grad.pts_streams", dtype=np.int32)
grad = np.fromfile(f"{tmp}/dpe_grad.pts_denGrad").reshape(-1, 3)
c = (np.arange(grid) + 0.5) * box / grid
x, y, z = np.meshgrid(c, c, c, indexing="ij")
pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
gx = np.abs(grad[:, 0])
cand = np.where((st == 1) & (gx > np.median(gx[st == 1])))[0][::997][:40]
eps = 0.02 * box / grid
probes = np.empty((2 * cand.size, 3))
probes[0::2] = pts[cand]; probes[0::2, 0] -= eps
probes[1::2] = pts[cand]; probes[1::2, 0] += eps
probes.astype(np.float64).tofile(fdbin)
np.savetxt(fdsel, np.column_stack([cand, np.full(cand.size, eps)]), fmt="%.17g")
PY
echo ">> run 7: jitter-free pancake, standard binary, FD probe points"
run "${BIN}" "${TMP}/dpe_fd" "${SNAP_PAN0}" --periodic --sample-points "${PTS_FD}"

echo ">> (E) PS-only / unsupported combinations are rejected"
set +e
"${BIN}" "${SNAP_PAN}" "${TMP}/dpe_rej" --grid "${GRID}" --field density --input 105 \
    --MpcUnit 1 --sample-points "${PTS_BIN}" --per-stream > "${TMP}/dpe_rej1.log" 2>&1
RC_PS=$?
"${BIN}" "${SNAP_PAN}" "${TMP}/dpe_rej" --grid "${GRID}" --field density --input 105 \
    --MpcUnit 1 --sample-points "${PTS_BIN}" --partition 2 2 2 > "${TMP}/dpe_rej2.log" 2>&1
RC_PART=$?
set -e
[ "${RC_PS}" -ne 0 ]   && grep -q "PS-DTFE only" "${TMP}/dpe_rej1.log" \
    || { echo "FAIL: --per-stream was not rejected by the standard binary"; exit 1; }
[ "${RC_PART}" -ne 0 ] && grep -q "single triangulation" "${TMP}/dpe_rej2.log" \
    || { echo "FAIL: --partition + --sample-points was not rejected"; exit 1; }
echo "   OK"

echo ">> checking the numbers ..."
"${PY}" - "${TMP}" "${GRID}" "${BOX}" "${HAVE_PS}" "${SEL_FD}" <<'PY'
import sys
import numpy as np

tmp, grid, box, have_ps, fdsel = (sys.argv[1], int(sys.argv[2]), float(sys.argv[3]),
                                  int(sys.argv[4]), sys.argv[5])
ncell = grid ** 3
fails = []

def check(name, ok, detail):
    print(f"   {'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)

def load(root, ext, dtype=np.float64, ncomp=1):
    d = np.fromfile(root + ext, dtype=dtype)
    assert d.size == ncell * ncomp, (root + ext, d.size)
    return d.reshape(-1, ncomp) if ncomp > 1 else d

# ---------- (A) uniform lattice ----------
d = load(f"{tmp}/dpe_uni", ".pts_den")
v = load(f"{tmp}/dpe_uni", ".pts_vel", ncomp=3)
s = load(f"{tmp}/dpe_uni", ".pts_streams", dtype=np.int32)
disp = load(f"{tmp}/dpe_uni", ".pts_velDisp", ncomp=6)
check("A1 coverage is 0/1 and complete", set(np.unique(s).tolist()) == {1},
      f"stream values {sorted(set(np.unique(s).tolist()))}, coverage {(s == 1).mean():.4f}")
check("A2 uniform density ~1", abs(float(d.mean()) - 1.0) < 0.02,
      f"mean = {d.mean():.4f} (DTFE estimator level; per-point scatter {d.std():.3f})")
check("A3 velocities exactly 0", float(np.abs(v).max()) == 0.0,
      f"max |v| = {np.abs(v).max():.3e} (uniform snapshot has zero velocities)")
check("A4 dispersion exactly 0", float(np.abs(disp).max()) == 0.0,
      "single containing tet per point -> no dispersion by definition")

# ---------- (B) pancake: standard vs PS-DTFE in single-stream regions ----------
if have_ps:
    sd = load(f"{tmp}/dpe_pan", ".pts_den")
    sv = load(f"{tmp}/dpe_pan", ".pts_vel", ncomp=3)
    ss = load(f"{tmp}/dpe_pan", ".pts_streams", dtype=np.int32)
    pd_ = load(f"{tmp}/dpe_pan_ps", ".pts_den")
    pv = load(f"{tmp}/dpe_pan_ps", ".pts_vel", ncomp=3)
    ps = load(f"{tmp}/dpe_pan_ps", ".pts_streams", dtype=np.int32)
    m = (ps == 1) & (ss == 1)
    check("B1 single-stream overlap sane", m.mean() > 0.5,
          f"{m.sum()}/{ncell} points single-stream in both")
    scale = float(np.abs(pv[m]).max()) + 1e-30
    relv = np.abs(sv[m, 0] - pv[m, 0]) / (np.abs(pv[m, 0]) + 1e-3 * scale)
    check("B2 velocity matches PS (median)", float(np.median(relv)) < 1e-3,
          f"median rel = {np.median(relv):.2e} (ps_standard_cross_check tolerance)")
    reld = np.abs(sd[m] - pd_[m]) / pd_[m]
    check("B3 density matches PS (median, estimator level)", float(np.median(reld)) < 0.15,
          f"median rel = {np.median(reld):.3f}")
else:
    print("   SKIP B cross-estimator (no PS-DTFE binary)")

# ---------- (D) gradient: transverse ~0, grad_x == central FD ----------
grad = load(f"{tmp}/dpe_grad", ".pts_denGrad", ncomp=3)
st = load(f"{tmp}/dpe_grad", ".pts_streams", dtype=np.int32)
m1 = st == 1
gxs = float(np.abs(grad[m1, 0]).max())
tr = float(max(np.abs(grad[m1, 1]).max(), np.abs(grad[m1, 2]).max()))
check("D1 transverse gradient vanishes", tr < 3e-6 * max(gxs, 1.0),
      f"max |grad_yz| = {tr:.3e} vs max |grad_x| = {gxs:.3e}")
sel = np.loadtxt(fdsel).reshape(-1, 2)
cand = sel[:, 0].astype(int); eps = float(sel[0, 1])
fd_den = np.fromfile(f"{tmp}/dpe_fd.pts_den")
fd = (fd_den[1::2] - fd_den[0::2]) / (2 * eps)
gx = grad[cand, 0]
rel = np.abs(fd - gx) / (np.abs(gx) + 1e-12)
check("D2 grad_x matches central FD of .pts_den",
      float(np.median(rel)) < 0.02 and float(np.percentile(rel, 90)) < 0.05,
      f"median rel = {np.median(rel):.2e}, p90 = {np.percentile(rel, 90):.2e} ({cand.size} probes)")

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (standard-DTFE point evaluation: 0/1 coverage, estimator-level"
      " agreement with PS-DTFE, exact gradients)")
PY

echo "============================================================"
echo " standard-DTFE point-evaluation check PASSED"
echo "============================================================"
