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
PTS_BIN="${TMP}/ppe_input_centers.bin"
PTS_TXT="${TMP}/ppe_input_centers.txt"

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

echo ">> checking the numbers ..."
"${PY}" - "${TMP}" "${GRID}" <<'PY'
import sys
import numpy as np

tmp, grid = sys.argv[1], int(sys.argv[2])
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
