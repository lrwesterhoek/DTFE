#!/usr/bin/env bash
# PS-DTFE arbitrary-point evaluation (--sample-points) correctness check.
#
# What is verified, and why these are the right bounds:
#  A) UNIFORM box (amp=0: Eulerian == Lagrangian bit-for-bit). The 'geometric' per-stream
#     density is rho_bar * |det Lag| / |det Eul| = rho_bar EXACTLY, so every point evaluation
#     must equal rho/rho_bar = 1 to double rounding -- and therefore matches the (exactly
#     mass-conserving) nSub=1 grid deposit's mean at float level. The 'dtfe' variant must
#     match to float32 vertex-density rounding. Velocities are identically zero.
#  B) PANCAKE (amp=1.8, multi-stream): cross-checks against the nSub=1 grid deposit at the
#     48^3 cell centers.
#       - stream sets: the deposit's .streams at nSub=1 counts the tetrahedra containing the
#         cell centre (plus centroid-fallback deposits of sub-cell tetrahedra, which do not
#         occur in single-stream cells here) -> integer equality in single-stream cells.
#       - VELOCITY: at nSub=1 the deposit evaluates each stream's velocity AT the cell centre,
#         so single-stream cells must match the point evaluation to float32 rounding. This is
#         the float-level deposit cross-check; the deposit's DENSITY cannot match pointwise
#         because it deposits mass shares m/N (quantized at tetrahedron scale), so density is
#         checked in aggregate (mean ratio) and via the exact uniform case (A).
#       - single-stream points have exactly zero dispersion.
#  C) PARTITION-SPLIT CONSISTENCY: --partition 2 2 2 vs serial must agree to float rounding
#     (stream records are collected per tetrahedron and reduced in a deterministically sorted
#     order, so only float-ULP wrap differences of periodic-image copies remain).
#  D) Ragged --per-stream layout invariants + estimator consistency (sum of per-stream
#     densities == total; sorted descending; offsets consistent; 'dtfe' vs 'geometric' agree
#     to ~5% in smooth single-stream regions).
#  E) Text and binary sample-point inputs give byte-identical outputs.
#  F) --per-stream-ids (when the binary supports it): adding the flag leaves every existing
#     output byte-identical; '.pts_stream_ids' has sum(streams) records of 4 ascending
#     uint64 snapshot ParticleIDs and is byte-identical between serial and --partition 2 2 2
#     (the id quadruple is sorted, so it is orientation- and partition-invariant).
#  G) --pts-den-grad (when the binary supports it): adding the flag leaves every existing
#     output byte-identical; on a jitter-free pancake (density varies only along x) the
#     transverse gradients vanish, grad_x matches a central finite difference of '.pts_den'
#     at x +- eps probe points away from the caustics, the per-point gradient equals the sum
#     of its per-stream gradients, and serial vs --partition 2 2 2 agree to float rounding.
#
# Usage:
#   tests/ps_point_eval_check.sh              # build, run, check
#   tests/ps_point_eval_check.sh --no-build
# Requires the build toolchain plus python3 with numpy and h5py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# python with a working h5py (plain python3 resolves to a broken x86_64 h5py on this machine)
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"
BIN="./PS-DTFE"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
# input/point file names must NOT share a prefix with the output roots (rm -f "<root>".*)
SNAP_UNI="${TMP}/ppe_input_uniform.hdf5"
SNAP_PAN="${TMP}/ppe_input_pancake.hdf5"
SNAP_PAN0="${TMP}/ppe_input_pancake0.hdf5"
PTS_BIN="${TMP}/ppe_input_centers.bin"
PTS_TXT="${TMP}/ppe_input_centers.txt"
PTS_FD="${TMP}/ppe_input_fdprobes.bin"
SEL_FD="${TMP}/ppe_input_fdsel.txt"

echo "============================================================"
echo " PS-DTFE point-evaluation check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

if [ "${1:-}" != "--no-build" ]; then
    echo ">> building PS-DTFE ..."
    # respect the current build mode (o_ps/.build_mode: METAL=1 / CUDA=1 / HIP=1 / empty)
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

echo ">> generating test snapshots + sample points (all ${GRID}^3 cell centres) ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_UNI}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 0 --jitter-frac 0.3 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_PAN}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null
"${PY}" - "${PTS_BIN}" "${PTS_TXT}" "${GRID}" "${BOX}" <<'PY'
import sys
import numpy as np
binf, txtf, grid, box = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
c = (np.arange(grid) + 0.5) * box / grid
x, y, z = np.meshgrid(c, c, c, indexing="ij")
pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float64)
pts.tofile(binf)                                   # raw float64 N x 3, no header
with open(txtf, "w") as f:                         # same points as text, full precision
    for p in pts:
        f.write("%.17g %.17g %.17g\n" % (p[0], p[1], p[2]))
PY

run() {  # $1 = output root, rest = extra args
    local out="$1"; shift
    rm -f "${out}".*
    local log="${out}.log"
    set +e
    "${BIN}" "$@" "${out}" --grid "${GRID}" --field density velocity dispersion \
        --input 105 --MpcUnit 1 --verbose 1 > "$log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "   ERROR: PS-DTFE exited with code $rc -- last 25 lines of $log:"
        tail -n 25 "$log" | sed 's/^/      | /'
        exit 1
    fi
}

echo ">> run 1: uniform box, 'geometric', --per-stream"
run "${TMP}/ppe_uni_geo" "${SNAP_UNI}" --periodic \
    --sample-points "${PTS_BIN}" --per-stream --ps-stream-density geometric
echo ">> run 2: uniform box, 'dtfe' (default)"
run "${TMP}/ppe_uni_dtfe" "${SNAP_UNI}" --periodic --sample-points "${PTS_BIN}"
echo ">> run 3: pancake, serial, 'geometric', --per-stream"
run "${TMP}/ppe_pan_ser" "${SNAP_PAN}" --periodic \
    --sample-points "${PTS_BIN}" --per-stream --ps-stream-density geometric
echo ">> run 4: pancake, --partition 2 2 2, 'geometric', --per-stream"
run "${TMP}/ppe_pan_par" "${SNAP_PAN}" --periodic --partition 2 2 2 \
    --sample-points "${PTS_BIN}" --per-stream --ps-stream-density geometric
echo ">> run 5: pancake, serial, 'dtfe', --per-stream"
run "${TMP}/ppe_pan_dtfe" "${SNAP_PAN}" --periodic --sample-points "${PTS_BIN}" --per-stream
echo ">> run 6: pancake, serial, 'geometric', TEXT sample-point input"
run "${TMP}/ppe_pan_txt" "${SNAP_PAN}" --periodic \
    --sample-points "${PTS_TXT}" --per-stream --ps-stream-density geometric

echo ">> (E) text vs binary sample-point input: outputs must be byte-identical"
for ext in .pts_den .pts_vel .pts_velDisp .pts_streams .pts_stream_offsets .pts_stream_records; do
    cmp -s "${TMP}/ppe_pan_ser${ext}" "${TMP}/ppe_pan_txt${ext}" \
        || { echo "FAIL: ${ext} differs between text and binary point input"; exit 1; }
done
echo "   OK"

# --per-stream-ids section, gated so the script still runs against binaries predating the flag
# (capture first: 'grep -q' would SIGPIPE the binary and trip pipefail)
HAVE_IDS=0
FULL_HELP="$("${BIN}" --full_help 2>/dev/null || true)"
if grep -q -- '--per-stream-ids' <<<"${FULL_HELP}"; then
    HAVE_IDS=1
    echo ">> run 7: pancake, serial, 'geometric', --per-stream-ids (implies --per-stream)"
    run "${TMP}/ppe_pan_ids" "${SNAP_PAN}" --periodic \
        --sample-points "${PTS_BIN}" --per-stream-ids --ps-stream-density geometric
    echo ">> run 8: pancake, --partition 2 2 2, 'geometric', --per-stream-ids"
    run "${TMP}/ppe_pan_ids_par" "${SNAP_PAN}" --periodic --partition 2 2 2 \
        --sample-points "${PTS_BIN}" --per-stream-ids --ps-stream-density geometric
    echo ">> (F) --per-stream-ids leaves the existing outputs byte-identical"
    for ext in .pts_den .pts_vel .pts_velDisp .pts_streams .pts_stream_offsets .pts_stream_records; do
        cmp -s "${TMP}/ppe_pan_ser${ext}" "${TMP}/ppe_pan_ids${ext}" \
            || { echo "FAIL: ${ext} changed when --per-stream-ids was added"; exit 1; }
    done
    echo "   OK"
else
    echo ">> (F) skipped: this PS-DTFE binary has no --per-stream-ids"
fi

# --pts-den-grad section, gated like (F)
HAVE_GRAD=0
if grep -q -- '--pts-den-grad' <<<"${FULL_HELP}"; then
    HAVE_GRAD=1
    # jitter-free pancake: the density varies only along x, so the transverse gradient
    # components must vanish (the Lagrangian lattice is y/z-translation-invariant)
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_PAN0}" --n "${N}" --box "${BOX}" \
        --amplitude-factor 1.8 --jitter-frac 0 >/dev/null
    echo ">> run 9: jitter-free pancake, serial, 'dtfe', --per-stream --pts-den-grad"
    run "${TMP}/ppe_grad_ser" "${SNAP_PAN0}" --periodic \
        --sample-points "${PTS_BIN}" --per-stream --pts-den-grad
    echo ">> run 10: jitter-free pancake, --partition 2 2 2, 'dtfe', --per-stream --pts-den-grad"
    run "${TMP}/ppe_grad_par" "${SNAP_PAN0}" --periodic --partition 2 2 2 \
        --sample-points "${PTS_BIN}" --per-stream --pts-den-grad
    echo ">> run 11: jitter-free pancake, serial, 'dtfe', --per-stream (no gradient flag)"
    run "${TMP}/ppe_grad_off" "${SNAP_PAN0}" --periodic \
        --sample-points "${PTS_BIN}" --per-stream
    echo ">> (G) --pts-den-grad leaves the existing outputs byte-identical"
    for ext in .pts_den .pts_vel .pts_velDisp .pts_streams .pts_stream_offsets .pts_stream_records; do
        cmp -s "${TMP}/ppe_grad_off${ext}" "${TMP}/ppe_grad_ser${ext}" \
            || { echo "FAIL: ${ext} changed when --pts-den-grad was added"; exit 1; }
    done
    echo "   OK"
    # finite-difference probes: single-stream cell centres away from the caustics, x +- eps
    "${PY}" - "${TMP}" "${GRID}" "${BOX}" "${PTS_FD}" "${SEL_FD}" <<'PY'
import sys
import numpy as np
tmp, grid, box, fdbin, fdsel = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
ncell = grid ** 3
st = np.fromfile(f"{tmp}/ppe_grad_ser.pts_streams", dtype=np.int32)
grad = np.fromfile(f"{tmp}/ppe_grad_ser.pts_denGrad").reshape(-1, 3)
c = (np.arange(grid) + 0.5) * box / grid
x, y, z = np.meshgrid(c, c, c, indexing="ij")
pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
gx = np.abs(grad[:, 0])
# single-stream, meaningful slope (above-median |grad_x|), keep a deterministic subset
cand = np.where((st == 1) & (gx > np.median(gx[st == 1])))[0][::997][:40]
eps = 0.02 * box / grid
probes = np.empty((2 * cand.size, 3))
probes[0::2] = pts[cand]; probes[0::2, 0] -= eps
probes[1::2] = pts[cand]; probes[1::2, 0] += eps
probes.astype(np.float64).tofile(fdbin)
np.savetxt(fdsel, np.column_stack([cand, np.full(cand.size, eps)]), fmt="%.17g")
PY
    echo ">> run 12: jitter-free pancake, serial, FD probe points"
    run "${TMP}/ppe_grad_fd" "${SNAP_PAN0}" --periodic --sample-points "${PTS_FD}"
else
    echo ">> (G) skipped: this PS-DTFE binary has no --pts-den-grad"
fi

echo ">> checking the numbers ..."
"${PY}" - "${TMP}" "${GRID}" "${HAVE_IDS}" "${HAVE_GRAD}" <<'PY'
import sys
import numpy as np

tmp, grid = sys.argv[1], int(sys.argv[2])
have_ids = len(sys.argv) > 3 and sys.argv[3] == "1"
have_grad = len(sys.argv) > 4 and sys.argv[4] == "1"
ncell = grid ** 3
fails = []

def check(name, ok, detail):
    print(f"   {'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)

def load_pts(root):
    den = np.fromfile(root + ".pts_den")
    vel = np.fromfile(root + ".pts_vel").reshape(-1, 3)
    dsp = np.fromfile(root + ".pts_velDisp").reshape(-1, 6)
    st  = np.fromfile(root + ".pts_streams", dtype=np.int32)
    assert den.size == ncell and st.size == ncell
    return den, vel, dsp, st

def load_ragged(root):
    off = np.fromfile(root + ".pts_stream_offsets", dtype=np.uint64).astype(np.int64)
    rec = np.fromfile(root + ".pts_stream_records").reshape(-1, 4)
    return off, rec

# ---------- (A) uniform box: exact analytic values ----------
den, vel, dsp, st = load_pts(f"{tmp}/ppe_uni_geo")
off, rec = load_ragged(f"{tmp}/ppe_uni_geo")
check("A1 uniform coverage", st.min() >= 1, f"min streams {st.min()}")
check("A2 uniform mostly single-stream", np.mean(st == 1) > 0.99,
      f"{np.mean(st == 1) * 100:.3f}% single-stream (face-grazing tolerance cases excepted)")
# every stream's geometric density is rho_bar exactly => den == stream count to double rounding
check("A3 uniform geometric density exact", np.abs(den - st).max() < 1e-9,
      f"max|rho/rho_bar - streams| = {np.abs(den - st).max():.3e}")
check("A4 uniform per-stream densities exact", np.abs(rec[:, 0] - 1).max() < 1e-9,
      f"max|d_s - 1| = {np.abs(rec[:, 0] - 1).max():.3e}")
check("A5 uniform velocities zero", np.abs(vel).max() == 0 and np.abs(dsp).max() == 0,
      f"max|v| = {np.abs(vel).max():.3e}, max|sigma| = {np.abs(dsp).max():.3e}")
# the nSub=1 grid deposit conserves mass exactly -> its mean is 1; point evals are 1 pointwise
# (the deposit grid is stored as float32, so the mean carries ~1e-5 accumulation rounding)
gden = np.fromfile(f"{tmp}/ppe_uni_geo.den", dtype=np.float32).astype(np.float64)
check("A6 matches deposit mean (float32 level)", abs(gden.mean() - den.mean()) < 1e-4,
      f"|mean(deposit) - mean(points)| = {abs(gden.mean() - den.mean()):.3e}")
# 'dtfe' variant: vertex densities are rho_bar to float32 rounding
den_b, _, _, st_b = load_pts(f"{tmp}/ppe_uni_dtfe")
check("A7 uniform dtfe density (float32 level)", np.abs(den_b - st_b).max() < 1e-5,
      f"max|rho/rho_bar - streams| = {np.abs(den_b - st_b).max():.3e}")

# ---------- (B) pancake vs the nSub=1 grid deposit ----------
den, vel, dsp, st = load_pts(f"{tmp}/ppe_pan_ser")
gden = np.fromfile(f"{tmp}/ppe_pan_ser.den", dtype=np.float32).astype(np.float64)
gst  = np.fromfile(f"{tmp}/ppe_pan_ser.streams", dtype=np.float32)
gvel = np.fromfile(f"{tmp}/ppe_pan_ser.vel", dtype=np.float32).reshape(-1, 3).astype(np.float64)
check("B0 multi-stream present", st.max() >= 3, f"max streams {st.max()}")
m1 = gst == 1  # single-stream cells per the deposit
agree = np.mean(st[m1] == 1)
check("B1 single-stream stream sets match deposit", agree > 0.999,
      f"{agree * 100:.4f}% of {m1.sum()} cells")
both1 = m1 & (st == 1)
vscale = np.abs(gvel).max()
dv = np.abs(gvel[both1] - vel[both1]).max() / vscale
check("B2 single-stream velocity matches deposit (float level)", dv < 1e-5,
      f"max rel diff = {dv:.3e}")
check("B3 single-stream dispersion is zero", np.abs(dsp[st == 1]).max() == 0,
      f"max = {np.abs(dsp[st == 1]).max():.3e}")
ratio = den[both1].mean() / gden[both1].mean()
check("B4 single-stream mean density vs deposit", 0.9 < ratio < 1.12,
      f"mean(points)/mean(deposit) = {ratio:.4f} (deposit is quantized at tet scale pointwise)")

# ---------- (C) partition-split consistency ----------
den2, vel2, dsp2, st2 = load_pts(f"{tmp}/ppe_pan_par")
eq = np.mean(st == st2)
check("C1 partition streams identical", eq > 0.999, f"{eq * 100:.4f}% equal")
same = st == st2
dd = np.abs(den[same] - den2[same]).max() / (np.abs(den).max() + 1e-30)
dv = np.abs(vel[same] - vel2[same]).max() / (np.abs(vel).max() + 1e-30)
ds = np.abs(dsp[same] - dsp2[same]).max() / (np.abs(dsp).max() + 1e-30)
check("C2 partition density (float level)", dd < 1e-5, f"max rel = {dd:.3e}")
check("C3 partition velocity (float level)", dv < 1e-5, f"max rel = {dv:.3e}")
check("C4 partition dispersion (float level)", ds < 1e-5, f"max rel = {ds:.3e}")
offa, reca = load_ragged(f"{tmp}/ppe_pan_ser")
offb, recb = load_ragged(f"{tmp}/ppe_pan_par")
if reca.shape == recb.shape:
    dr = np.abs(reca[:, 0] - recb[:, 0]).max() / (np.abs(reca[:, 0]).max() + 1e-30)
    dw = np.abs(reca[:, 1:] - recb[:, 1:]).max() / (np.abs(reca[:, 1:]).max() + 1e-30)
    check("C5 partition per-stream records (float level)", dr < 1e-5 and dw < 1e-5,
          f"den rel = {dr:.3e}, vel rel = {dw:.3e}")
else:
    check("C5 partition per-stream records (float level)", False,
          f"record counts differ: {reca.shape[0]} vs {recb.shape[0]}")

# ---------- (D) ragged layout + estimator consistency ----------
for tag, root in [("geometric", f"{tmp}/ppe_pan_ser"), ("dtfe", f"{tmp}/ppe_pan_dtfe")]:
    den_r, _, _, st_r = load_pts(root)
    off, rec = load_ragged(root)
    ok_layout = (off.size == ncell + 1 and off[0] == 0 and off[-1] == rec.shape[0]
                 and np.all(np.diff(off) == st_r))
    sums = np.zeros(ncell)
    nz = np.diff(off) > 0
    sums[nz] = np.add.reduceat(rec[:, 0], off[:-1][nz])
    ok_sum = np.abs(sums - den_r).max() < 1e-9
    ok_sorted = True
    for i in np.where(st_r > 1)[0]:
        if np.any(np.diff(rec[off[i]:off[i + 1], 0]) > 0):
            ok_sorted = False
            break
    check(f"D1 [{tag}] ragged layout consistent", ok_layout, "offsets/counts/records")
    check(f"D2 [{tag}] per-stream sums == total", ok_sum,
          f"max diff = {np.abs(sums - den_r).max():.3e}")
    check(f"D3 [{tag}] streams sorted by density desc", ok_sorted, "")
den_bb, _, _, st_bb = load_pts(f"{tmp}/ppe_pan_dtfe")
check("D4 variant runs see identical streams", np.all(st == st_bb), "")
mask = st == 1
r = den_bb[mask] / np.maximum(den[mask], 1e-12)
q5, q95 = np.percentile(r, [5, 95])
check("D5 dtfe vs geometric agree in smooth regions", 0.8 < q5 and q95 < 1.25,
      f"single-stream b/a ratio 5-95% = [{q5:.3f}, {q95:.3f}], median {np.median(r):.3f}")

# ---------- (F) --per-stream-ids: stream identities ----------
if have_ids:
    import h5py
    ids_s = np.fromfile(f"{tmp}/ppe_pan_ids.pts_stream_ids", dtype=np.uint64).reshape(-1, 4)
    ids_p = np.fromfile(f"{tmp}/ppe_pan_ids_par.pts_stream_ids", dtype=np.uint64).reshape(-1, 4)
    st_i = np.fromfile(f"{tmp}/ppe_pan_ids.pts_streams", dtype=np.int32)
    check("F1 ids record count == sum(streams)", ids_s.shape[0] == st_i.sum(),
          f"{ids_s.shape[0]} records vs {st_i.sum()} streams")
    check("F2 id quadruples ascending (4 distinct vertices)",
          bool((np.diff(ids_s.astype(np.int64), axis=1) > 0).all()), "")
    if ids_s.shape == ids_p.shape:
        check("F3 serial vs partition ids identical", bool(np.array_equal(ids_s, ids_p)), "")
    else:
        check("F3 serial vs partition ids identical", False,
              f"record counts differ: {ids_s.shape[0]} vs {ids_p.shape[0]}")
    with h5py.File(f"{tmp}/ppe_input_pancake.hdf5", "r") as f:
        snap_ids = f["PartType1/ParticleIDs"][:]
    check("F4 every id is a snapshot ParticleID",
          bool(np.isin(ids_s.ravel(), snap_ids).all()), f"{ids_s.size} ids checked")

# ---------- (G) --pts-den-grad: density gradients ----------
if have_grad:
    grad = np.fromfile(f"{tmp}/ppe_grad_ser.pts_denGrad").reshape(-1, 3)
    _, _, _, st_g = load_pts(f"{tmp}/ppe_grad_ser")
    off_g, rec_g = load_ragged(f"{tmp}/ppe_grad_ser")
    sgrad = np.fromfile(f"{tmp}/ppe_grad_ser.pts_stream_dengrad").reshape(-1, 3)
    check("G1 gradient layout", grad.shape[0] == ncell and sgrad.shape[0] == rec_g.shape[0],
          f"{grad.shape[0]} points, {sgrad.shape[0]} stream records")
    # jitter-free pancake: density is a function of x only -> transverse components vanish up
    # to float32 vertex-density rounding (measured ~1e-6 of the grad_x scale; 3x headroom)
    m1 = st_g == 1
    gxs = np.abs(grad[m1, 0]).max()
    tr = np.abs(grad[m1, 1:]).max()
    check("G2 transverse gradient vanishes (1D pancake)", tr < 3e-6 * gxs,
          f"max|grad_yz| = {tr:.3e} vs 3e-6 * max|grad_x| = {3e-6 * gxs:.3e}")
    # per-point gradient == sum of its per-stream gradients (same reduction, same order)
    gsum = np.zeros((ncell, 3))
    nzg = np.diff(off_g) > 0
    for d in range(3):
        gsum[nzg, d] = np.add.reduceat(sgrad[:, d], off_g[:-1][nzg])
    check("G3 per-stream gradients sum to total", np.abs(gsum - grad).max() < 1e-9 * max(gxs, 1.0),
          f"max diff = {np.abs(gsum - grad).max():.3e}")
    # partition consistency (float level, as C2-C4)
    grad_p = np.fromfile(f"{tmp}/ppe_grad_par.pts_denGrad").reshape(-1, 3)
    _, _, _, st_gp = load_pts(f"{tmp}/ppe_grad_par")
    same_g = st_g == st_gp
    dgp = np.abs(grad[same_g] - grad_p[same_g]).max() / (np.abs(grad).max() + 1e-30)
    check("G4 partition gradient (float level)", dgp < 1e-5, f"max rel = {dgp:.3e}")
    # central finite difference of .pts_den at x +- eps vs grad_x, away from caustics
    sel = np.loadtxt(f"{tmp}/ppe_input_fdsel.txt").reshape(-1, 2)
    idx, eps = sel[:, 0].astype(int), sel[0, 1]
    fden = np.fromfile(f"{tmp}/ppe_grad_fd.pts_den")
    fst = np.fromfile(f"{tmp}/ppe_grad_fd.pts_streams", dtype=np.int32)
    fd = (fden[1::2] - fden[0::2]) / (2 * eps)
    ok = (fst[0::2] == 1) & (fst[1::2] == 1)      # both probes single-stream
    rel = np.abs(fd[ok] - grad[idx[ok], 0]) / (np.abs(grad[idx[ok], 0]) + 1e-12)
    check("G5 grad_x matches central FD of .pts_den", np.median(rel) < 0.02 and np.percentile(rel, 90) < 0.05,
          f"{ok.sum()}/{idx.size} probes, median rel = {np.median(rel):.2e}, p90 = {np.percentile(rel, 90):.2e}")

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (point evaluation matches the deposit where the math is exact, "
      "is partition-split consistent, and the per-stream layout is self-consistent)")
PY

echo "============================================================"
echo " point-evaluation check PASSED"
echo "============================================================"
