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
#  E) clear rejections: --per-stream / --per-stream-ids (PS-only) and --partNo (one partition
#     would answer only its own share of the points). '--partition' is accepted:
#  F) --partition: each partition triangulates its padded sub-box and answers the points inside
#     the tetrahedra whose Eulerian centroid lies in its sub-box (the composite server's ownership),
#     so every point is answered exactly once, by the tetrahedron the single triangulation uses.
#     Periodic crossed waves at cell centres, random points, PARTICLE POSITIONS (vertex ties, the
#     symbolic tie-break), float32 grid-line coordinates (the partition cuts) and points 1e-5 from
#     the box faces (the periodic seam): split 2x2x2, 3x3x3 and 3x2x1 (a partition spanning a whole
#     periodic axis) give the single run's '.pts_streams' exactly and every value to 1e-12 of the
#     field's peak (measured ~1e-15: only the vertex order and the vertex densities' summation
#     order differ). A non-periodic cloud: the same 1.5 spacings or more inside the particles'
#     bounding box -- closer, a partition can tessellate the convex hull differently (flat hull
#     tetrahedra have huge circumspheres), as the composite server's help says. A too small
#     '--padding' is reported (points owned twice / periodic points covered by none). A GPU build
#     (--gpu) writes the CPU's points byte for byte (point evaluation is CPU-only). 2D (DTFE-2d,
#     single pass) the same. demo/z0_64.gadget (z = 0, voids): 128^2 planes through the box
#     middle and ON the periodic seam x = 0 at the production padding 15 equal the single run.
#
# Usage: tests/dtfe_point_eval_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"
BIN="${DTFE_BIN}"; PSBIN="${PS_BIN}"
precision_require "${BIN}" "${PSBIN}"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"; mkdir -p "${TMP}"
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

build_target "${1:-}" DTFE o
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

# capture FIRST -- piping the binary straight into grep -q would SIGPIPE it under pipefail
# a help that fails is a broken binary, not a missing feature; a missing feature fails under DTFE_TEST_REQUIRED=1
# (skip_or_fail, tests/precision.sh)
FULL_HELP="$("${BIN}" --full_help 2>/dev/null)" || { echo "FAIL: ${BIN} --full_help exited non-zero"; exit 1; }
if ! grep -q -- '--sample-points' <<<"${FULL_HELP}"; then
    skip_or_fail "this DTFE has no --sample-points support."
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
    die_on_failed_run "$rc" "$log" "$(basename "${bin}")"
}

runp() {  # like run, at --verbose 2 (the partition and GPU lines) and with the caller's own --field
    local bin="$1" out="$2" snap="$3"; shift 3
    rm -f "${out}".*
    local log="${out}.log"
    set +e
    "${bin}" "${snap}" "${out}" --grid "${GRID}" --input 105 --MpcUnit 1 --verbose 2 "$@" > "$log" 2>&1
    local rc=$?
    set -e
    die_on_failed_run "$rc" "$log" "$(basename "${bin}")"
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
    --MpcUnit 1 --sample-points "${PTS_BIN}" --partition 2 2 2 --partNo 1 > "${TMP}/dpe_rej2.log" 2>&1
RC_PART=$?
set -e
[ "${RC_PS}" -ne 0 ]   && grep -q "PS-DTFE only" "${TMP}/dpe_rej1.log" \
    || { echo "FAIL: --per-stream was not rejected by the standard binary"; exit 1; }
[ "${RC_PART}" -ne 0 ] && grep -q "cannot be combined with '--partNo'" "${TMP}/dpe_rej2.log" \
    || { echo "FAIL: --partNo + --sample-points was not rejected"; exit 1; }
echo "   OK"

# ---------------------------------------------------------------------------------------------
# (F) --partition: partitioned point evaluation == the single triangulation
# ---------------------------------------------------------------------------------------------
SNAP_X="${TMP}/dpe_input_crossed.hdf5"
SNAP_NP="${TMP}/dpe_input_cloud.hdf5"
PTS_X="${TMP}/dpe_part_pts.bin"
PTS_NP="${TMP}/dpe_part_pts_np.bin"
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_X}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 --crossed-waves >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_NP}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 0.05 --crossed-waves --margin-frac 0.15 --jitter-frac 0.3 >/dev/null
# the points: cell centres, random, particle positions (vertex ties), float32 grid-line coordinates
# (optimalPartitionSplit's cuts lo + dx*k, formed in float), points 1e-5 from the x faces
"${PY}" - "${SNAP_X}" "${PTS_X}" "${SNAP_NP}" "${PTS_NP}" "${BOX}" "${GRID}" <<'PY'
import sys
import h5py
import numpy as np
box, grid = float(sys.argv[5]), int(sys.argv[6])
for snap, out in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    rng = np.random.default_rng(7)
    with h5py.File(snap, "r") as f:
        x = f["PartType1/Coordinates"][...].astype(np.float64)
    c = (np.arange(16) + 0.5) * box / 16
    gx, gy, gz = np.meshgrid(c, c, c, indexing="ij")
    pts = [np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1), rng.uniform(0, box, (2000, 3)),
           x[rng.choice(len(x), 2000, replace=False)]]
    dx = np.float32(np.float32(box) / np.float32(grid))
    cut = (dx * rng.integers(0, grid + 1, (2000, 3)).astype(np.float32)).astype(np.float64)
    r = rng.uniform(0, box, (2000, 3))
    for a in range(3):
        m = rng.random(2000) < 0.5
        r[m, a] = cut[m, a]
    pts.append(r)
    e = rng.uniform(0, box, (400, 3))
    e[:200, 0] = rng.uniform(0, 1e-5, 200)
    e[200:, 0] = box - rng.uniform(0, 1e-5, 200)
    pts.append(e)
    np.concatenate(pts).astype(np.float64).tofile(out)
PY
PART_FLAGS=(--sample-points "${PTS_X}" --pts-den-grad --pts-vel-grad --periodic --field density velocity)
echo ">> run 8: crossed waves, single triangulation + --partition 2 2 2 / 3 3 3 / 3 2 1"
runp "${BIN}" "${TMP}/dpe_x_p1"   "${SNAP_X}" "${PART_FLAGS[@]}"
runp "${BIN}" "${TMP}/dpe_x_p2"   "${SNAP_X}" "${PART_FLAGS[@]}" --partition 2 2 2
runp "${BIN}" "${TMP}/dpe_x_p3"   "${SNAP_X}" "${PART_FLAGS[@]}" --partition 3 3 3
runp "${BIN}" "${TMP}/dpe_x_p321" "${SNAP_X}" "${PART_FLAGS[@]}" --partition 3 2 1
grep -q "point evaluation in 27 partitions" "${TMP}/dpe_x_p3.log" \
    || { echo "FAIL: the partitioned run did not say it evaluates the points per partition"; exit 1; }
for r in dpe_x_p1 dpe_x_p2 dpe_x_p3 dpe_x_p321; do
    if grep -q "WARNING" "${TMP}/${r}.log"; then echo "FAIL: ${r} warned (see ${TMP}/${r}.log)"; exit 1; fi
done
echo ">> run 9: non-periodic cloud, single triangulation + --partition 2 2 2 / 3 3 3"
NP_FLAGS=(--sample-points "${PTS_NP}" --pts-den-grad --pts-vel-grad --field density velocity)
runp "${BIN}" "${TMP}/dpe_np_p1" "${SNAP_NP}" "${NP_FLAGS[@]}"
runp "${BIN}" "${TMP}/dpe_np_p2" "${SNAP_NP}" "${NP_FLAGS[@]}" --partition 2 2 2
runp "${BIN}" "${TMP}/dpe_np_p3" "${SNAP_NP}" "${NP_FLAGS[@]}" --partition 3 3 3
echo ">> run 10: --padding 0.3 (far too small) is reported"
runp "${BIN}" "${TMP}/dpe_x_low" "${SNAP_X}" "${PART_FLAGS[@]}" --partition 3 3 3 --padding 0.3
grep -q "covered by none, although the box is periodic" "${TMP}/dpe_x_low.log" \
    && grep -q "the partitions' '--padding' is too small" "${TMP}/dpe_x_low.log" \
    || { echo "FAIL: a too small partition padding was not reported (see ${TMP}/dpe_x_low.log)"; exit 1; }
echo "   OK"
OBJ=o; precision_double && OBJ=o_d
GPU=0
[ -n "$(gpu_backend_of "${OBJ}")" ] && GPU=1
if [ "${GPU}" = "1" ]; then
    echo ">> run 11: --gpu (the '_a' grid per partition on the GPU): the points are the CPU's, byte for byte"
    GPU_FLAGS=(--sample-points "${PTS_X}" --pts-den-grad --pts-vel-grad --periodic --field density_a --partition 2 2 2)
    runp "${BIN}" "${TMP}/dpe_x_p2c" "${SNAP_X}" "${GPU_FLAGS[@]}"
    runp "${BIN}" "${TMP}/dpe_x_p2g" "${SNAP_X}" "${GPU_FLAGS[@]}" --gpu
    grep -q " GPU (" "${TMP}/dpe_x_p2g.log" \
        || { echo "FAIL: the --gpu run did not attempt the GPU (see ${TMP}/dpe_x_p2g.log)"; exit 1; }
    for ext in .pts_den .pts_vel .pts_velDisp .pts_streams .pts_denGrad .pts_velGrad; do
        cmp -s "${TMP}/dpe_x_p2c${ext}" "${TMP}/dpe_x_p2g${ext}" \
            || { echo "FAIL: ${ext} differs between the CPU and the --gpu partitioned run"; exit 1; }
    done
    echo "   OK"
else
    echo ">> run 11 skipped (${OBJ} carries no GPU stamp)"
fi
HAVE_2D=0
if ! precision_double && [ -x ./DTFE-2d ]; then
    HAVE_2D=1
    echo ">> run 12: 2D (DTFE-2d), single triangulation + --partition 3 3"
    SNAP_2D="${TMP}/dpe_input_2d.hdf5"; PTS_2D="${TMP}/dpe_pts_2d.bin"
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_2D}" --n 40 --box "${BOX}" \
        --amplitude-factor 1.8 --crossed-waves --dim 2 >/dev/null
    "${PY}" - "${SNAP_2D}" "${PTS_2D}" "${BOX}" <<'PY'
import sys
import h5py
import numpy as np
box = float(sys.argv[3])
rng = np.random.default_rng(3)
with h5py.File(sys.argv[1], "r") as f:
    x = f["PartType1/Coordinates"][...].astype(np.float64)
c = (np.arange(48) + 0.5) * box / 48
gx, gy = np.meshgrid(c, c, indexing="ij")
np.concatenate([np.stack([gx.ravel(), gy.ravel()], 1), rng.uniform(0, box, (1500, 2)),
                x[rng.choice(len(x), 800, replace=False)]]).astype(np.float64).tofile(sys.argv[2])
PY
    runp ./DTFE-2d "${TMP}/dpe_2d_p1" "${SNAP_2D}" --periodic --field density --sample-points "${PTS_2D}" --pts-den-grad --pts-vel-grad
    runp ./DTFE-2d "${TMP}/dpe_2d_p3" "${SNAP_2D}" --periodic --field density --sample-points "${PTS_2D}" --pts-den-grad --pts-vel-grad \
        --partition 3 3
else
    echo ">> run 12 skipped (no ./DTFE-2d, or the double pass)"
fi
HAVE_Z0=0
if [ -f demo/z0_64.gadget ]; then
    HAVE_Z0=1
    echo ">> run 13: demo/z0_64.gadget (z = 0), 128^2 planes at z = 50 and ON the seam x = 0, padding 15"
    PTS_Z0="${TMP}/dpe_z0_planes.bin"
    "${PY}" - "${PTS_Z0}" <<'PY'
import sys
import numpy as np
c = (np.arange(128) + 0.5) * 100.0 / 128
a, b = np.meshgrid(c, c, indexing="ij")
mid = np.stack([a.ravel(), b.ravel(), np.full(a.size, 50.0)], 1)
seam = np.stack([np.zeros(a.size), a.ravel(), b.ravel()], 1)
np.concatenate([mid, seam]).astype(np.float64).tofile(sys.argv[1])
PY
    for p in 1 2; do
        rm -f "${TMP}/dpe_z0_p${p}".*
        part=(); [ "${p}" = "2" ] && part=(--partition 2 2 2)
        "${BIN}" demo/z0_64.gadget "${TMP}/dpe_z0_p${p}" --grid 16 --periodic --field density --padding 15 \
            --sample-points "${PTS_Z0}" --pts-vel-grad --verbose 1 ${part[@]+"${part[@]}"} > "${TMP}/dpe_z0_p${p}.log" 2>&1 \
            || { echo "FAIL: the demo/z0_64 run p${p} (see ${TMP}/dpe_z0_p${p}.log)"; exit 1; }
    done
else
    echo ">> run 13 skipped (demo/z0_64.gadget is missing)"
fi

echo ">> checking the numbers ..."
"${PY}" - "${TMP}" "${GRID}" "${BOX}" "${HAVE_PS}" "${SEL_FD}" "${HAVE_2D}" "${HAVE_Z0}" "${SNAP_NP}" <<'PY'
import sys
import numpy as np

tmp, grid, box, have_ps, fdsel = (sys.argv[1], int(sys.argv[2]), float(sys.argv[3]),
                                  int(sys.argv[4]), sys.argv[5])
have_2d, have_z0, snap_np = int(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
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

# ---------- (F) --partition == the single triangulation ----------
def outputs(root, dim=3):
    o = {".pts_streams": np.fromfile(root + ".pts_streams", dtype=np.int32).reshape(-1, 1)}
    for ext, nc in ((".pts_den", 1), (".pts_vel", dim), (".pts_denGrad", dim), (".pts_velGrad", dim * dim)):
        try:
            o[ext] = np.fromfile(root + ext).reshape(-1, nc)
        except FileNotFoundError:
            pass
    return o

def differing(ref, tst, tol=1e-12):
    """points whose stream count differs or one of whose values moves by more than tol of that field's
    peak, and the share of points whose values are all bit-identical"""
    bad = np.zeros(len(ref[".pts_streams"]), bool)
    same = np.ones_like(bad)
    for ext, a in ref.items():
        b = tst[ext]
        if ext == ".pts_streams":
            bad |= (a != b)[:, 0]
            continue
        bad |= np.abs(a - b).max(1) > tol * max(float(np.abs(a).max()), 1e-300)
        same &= np.all(a == b, axis=1)
    return bad, same.mean()

ref = outputs(f"{tmp}/dpe_x_p1")
check("F0 the periodic single run covers every point", bool(np.all(ref[".pts_streams"] == 1)),
      f"coverage {float(np.mean(ref['.pts_streams'] == 1)):.4f} of {len(ref['.pts_streams'])} points")
SPLITS = {"p2": "2 2 2", "p3": "3 3 3", "p321": "3 2 1"}
for name in ("p2", "p3", "p321"):
    bad, same = differing(ref, outputs(f"{tmp}/dpe_x_{name}"))
    check(f"F1 --partition {SPLITS[name]} == the single triangulation (periodic)", not bad.any(),
          f"{int(bad.sum())} points differ (stream count, or a value by > 1e-12 of its peak); "
          f"{100 * same:.1f}% bit-identical")

# non-periodic: identical 1.5 spacings or more inside the cloud's bounding box (the hull may differ)
import h5py
with h5py.File(snap_np, "r") as f:
    xs = f["PartType1/Coordinates"][...].astype(np.float64)
pnp = np.fromfile(f"{tmp}/dpe_part_pts_np.bin").reshape(-1, 3)
spacing = float(np.cbrt(np.prod(xs.max(0) - xs.min(0)) / len(xs)))
inner = np.min(np.minimum(pnp - xs.min(0), xs.max(0) - pnp), axis=1) / spacing
refn = outputs(f"{tmp}/dpe_np_p1")
for name in ("p2", "p3"):
    bad, same = differing(refn, outputs(f"{tmp}/dpe_np_{name}"))
    far = bad & (inner > 1.5)
    check(f"F2 non-periodic --partition {SPLITS[name]} == the single triangulation away from the hull",
          not far.any() and bad.mean() < 0.005,
          f"{int(bad.sum())} of {len(bad)} points differ, all within {inner[bad].max() if bad.any() else 0.:.2f} "
          f"spacings of the cloud's faces ({int(far.sum())} farther than 1.5)")

if have_2d:
    bad, same = differing(outputs(f"{tmp}/dpe_2d_p1", 2), outputs(f"{tmp}/dpe_2d_p3", 2))
    check("F3 2D --partition 3 3 == the single triangulation", not bad.any(),
          f"{int(bad.sum())} points differ; {100 * same:.1f}% bit-identical")
if have_z0:
    z1, z2 = outputs(f"{tmp}/dpe_z0_p1"), outputs(f"{tmp}/dpe_z0_p2")
    bad, same = differing(z1, z2)
    check("F4 demo z = 0 planes (middle + seam), --partition 2 2 2 == the single run at padding 15",
          not bad.any() and bool(np.all(z1[".pts_streams"] == 1)),
          f"{int(bad.sum())} of {len(bad)} points differ; coverage {float(np.mean(z1['.pts_streams'] == 1)):.4f}; "
          f"{100 * same:.1f}% bit-identical")

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (standard-DTFE point evaluation: 0/1 coverage, estimator-level"
      " agreement with PS-DTFE, exact gradients, partitioned == single triangulation)")
PY

echo "============================================================"
echo " standard-DTFE point-evaluation check PASSED"
echo "============================================================"
