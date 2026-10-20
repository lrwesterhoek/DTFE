#!/usr/bin/env bash
# PS-DTFE point evaluation in the grids' conventions ('--pts-conventions grid', the default).
#
# The points follow the run's two grid options, --ps-vertex-mass and --ps-volume-weighted (ps_point_eval.cc, 'The
# grid deposit's conventions at the points'). With rho_bar*V_lag tetrahedron masses and density-weighted stream means
# instead ('--pts-conventions legacy' keeps those estimators), evolved (z = 127) initial positions as the Lagrangian
# input put a plane's density contrast at ~0.8 of the grid's at z = 20. Verified here, on synthetic boxes:
#  A) IDENTITY: without --ps-vertex-mass / --ps-volume-weighted 'grid' and 'legacy' give byte-identical '.pts_*'
#     files, and with both options the GRIDS of the two runs are byte-identical (the conventions are the points').
#  B) THE IC BIAS: single-stream crossed waves whose InitialCoordinates are displaced towards the final positions
#     by alpha = 0.1645 (TNG's D(z=127)/D(z=20)), --ps-vertex-mass --ps-volume-weighted --ps-exact-deposit, a slab
#     of four sub-planes through one cell layer: the plane's cell blocks against the grid's cells (and against the
#     analytic cell averages) have slope ~1 with 'grid', ~1-alpha with 'legacy' -- for 'dtfe' and 'geometric';
#     the perturbed-IC plane equals the true-IC plane to ~1%. The exact deposit isolates the conventions: the
#     sampled nSub=3 deposit of this one-particle-per-cell box is itself noisy at the cell scale (printed).
#  C) MASS AND NOISE on a uniform near-lattice box (jitter 0.01, TNG-like initial conditions): random points
#     average to rho_bar, and the point noise stays below the vertex-share grid's own (the per-tetrahedron
#     m_tet/V_eul -- the grid's density -- puts a normal mass into near-flat slivers no point hits: a mean of 0.89).
#  D) VOLUME WEIGHTING on a pancake and crossed waves (--per-stream): '.pts_vel' is the plain mean of the stream
#     velocities under 'grid' and their density-weighted mean under 'legacy'; '.pts_velDisp', '.pts_den' and
#     '.pts_streams' do not depend on --ps-volume-weighted; single-stream points agree between the two.
#  E) PARTITIONS: --partition 2 2 2 equals the single run to float rounding with --ps-vertex-mass (the vertex
#     ratios need the partition's 2-ring, inside its padding).
#  F) --serve (dtfelib.Estimator with the two options): bit-identical to the --sample-points run.
# Usage: tests/ps_point_conventions_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-32}"; GRID="${GRID:-32}"; BOX="${BOX:-100.0}"; ALPHA="${ALPHA:-0.1645}"; AMP="${AMP:-0.4}"
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP}/ps_point_conventions"; rm -rf "${TMP}"; mkdir -p "${TMP}"

echo "============================================================"
echo " PS-DTFE point conventions check   N=${N}^3  grid=${GRID}^3  alpha=${ALPHA}"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }
FULL_HELP="$("${BIN}" --full_help 2>/dev/null)" || { echo "FAIL: ${BIN} --full_help exited non-zero"; exit 1; }
grep -q -- '--pts-conventions' <<<"${FULL_HELP}" || skip_or_fail "this PS-DTFE has no --pts-conventions."

echo ">> test data ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/in_lin.hdf5" --n "${N}" --box "${BOX}" \
    --amplitude-factor "${AMP}" --crossed-waves --jitter-frac 0.05 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/in_uni.hdf5" --n "${N}" --box "${BOX}" \
    --amplitude-factor 0 --jitter-frac 0.01 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/in_crw.hdf5" --n 24 --box "${BOX}" \
    --amplitude-factor 1.8 --crossed-waves --jitter-frac 0.02 --seed 7 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/in_pan.hdf5" --n 24 --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null
"${PY}" - "${TMP}" "${ALPHA}" "${BOX}" "${GRID}" <<'PY'
import shutil, sys
import h5py
import numpy as np
tmp, alpha, box, g = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
# the perturbed initial positions: q + alpha (x - q), the min-image displacement (tests/ps_vertex_mass_check.sh)
shutil.copyfile(f"{tmp}/in_lin.hdf5", f"{tmp}/in_pic.hdf5")
with h5py.File(f"{tmp}/in_pic.hdf5", "r+") as f:
    q = f["PartType1/InitialCoordinates"][...].astype(np.float64)
    x = f["PartType1/Coordinates"][...].astype(np.float64)
    d = x - q
    d -= box * np.round(d / box)
    f["PartType1/InitialCoordinates"][...] = ((q + alpha * d) % box).astype(np.float32)
# the slab: four sub-planes of (4g)^2 points filling cell layer g/2
n, k0 = 4 * g, g // 2
c = (np.arange(n) + 0.5) * box / n
zs = (k0 + (np.arange(4) + 0.5) / 4) * box / g
X, Y, Z = np.meshgrid(c, c, zs, indexing="ij")
np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float64).tofile(f"{tmp}/in_slab.bin")
# a regular 2g^3 set (noise) and 2e5 uniform random points (mean)
c2 = (np.arange(2 * g) + 0.5) * box / (2 * g)
X, Y, Z = np.meshgrid(c2, c2, c2, indexing="ij")
np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float64).tofile(f"{tmp}/in_full.bin")
np.random.default_rng(11).uniform(0., box, (200000, 3)).astype(np.float64).tofile(f"{tmp}/in_rand.bin")
# a multi-stream plane through the crossed waves (not on the wave's symmetry planes)
c3 = (np.arange(96) + 0.5) * box / 96
X, Y = np.meshgrid(c3, c3, indexing="ij")
np.stack([X.ravel(), Y.ravel(), np.full(X.size, 0.503 * box)], 1).astype(np.float64).tofile(f"{tmp}/in_plane.bin")
PY

run() {  # $1 = output root, $2 = snapshot, rest = flags
    local out="$1" snap="$2"; shift 2
    rm -f "${out}".*
    if ! OMP_NUM_THREADS=2 "${BIN}" "${snap}" "${out}" --grid "${GRID}" --periodic --input 105 --MpcUnit 1 \
            --verbose 1 "$@" > "${out}.log" 2>&1; then
        echo "   ERROR: PS-DTFE failed -- last 20 lines of ${out}.log:"; tail -n 20 "${out}.log" | sed 's/^/      | /'
        exit 1
    fi
}
VMW=(--ps-vertex-mass --ps-volume-weighted)
PE=(--per-stream --pts-vel-grad --pts-den-grad)

echo ">> A: identity runs (multi-stream crossed waves, --partition 2 2 2)"
AF=(--field density_a velocity_a dispersion_a --sample-points "${TMP}/in_plane.bin" "${PE[@]}"
    --partition 2 2 2 --max-concurrent 2 --ordered-merge)
run "${TMP}/a_off_grid"   "${TMP}/in_crw.hdf5" "${AF[@]}"
run "${TMP}/a_off_legacy" "${TMP}/in_crw.hdf5" "${AF[@]}" --pts-conventions legacy
run "${TMP}/a_on_grid"    "${TMP}/in_crw.hdf5" "${AF[@]}" "${VMW[@]}"
run "${TMP}/a_on_legacy"  "${TMP}/in_crw.hdf5" "${AF[@]}" "${VMW[@]}" --pts-conventions legacy
echo ">> B: perturbed-IC slab (exact deposit), both estimators and conventions; the sampled deposit for comparison"
for est in dtfe geometric; do
    for conv in grid legacy; do
        run "${TMP}/b_${est}_${conv}" "${TMP}/in_pic.hdf5" --field density_a "${VMW[@]}" --ps-exact-deposit \
            --sample-points "${TMP}/in_slab.bin" --ps-stream-density "${est}" --pts-conventions "${conv}"
    done
    run "${TMP}/b_${est}_true" "${TMP}/in_lin.hdf5" --field density_a --ps-exact-deposit \
        --sample-points "${TMP}/in_slab.bin" --ps-stream-density "${est}"
done
run "${TMP}/b_sampled_grid"   "${TMP}/in_pic.hdf5" --field density_a "${VMW[@]}" --sample-points "${TMP}/in_slab.bin"
run "${TMP}/b_sampled_legacy" "${TMP}/in_pic.hdf5" --field density_a "${VMW[@]}" --sample-points "${TMP}/in_slab.bin" \
    --pts-conventions legacy
echo ">> C: uniform near-lattice box"
for est in dtfe geometric; do
    run "${TMP}/c_${est}_full" "${TMP}/in_uni.hdf5" --field density_a "${VMW[@]}" --ps-exact-deposit \
        --sample-points "${TMP}/in_full.bin" --ps-stream-density "${est}"
    run "${TMP}/c_${est}_rand" "${TMP}/in_uni.hdf5" --field density_a "${VMW[@]}" \
        --sample-points "${TMP}/in_rand.bin" --ps-stream-density "${est}"
done
echo ">> D: volume weighting (a pancake: single- and three-stream points, --per-stream)"
DF=(--field density_a --sample-points "${TMP}/in_plane.bin" "${PE[@]}")
run "${TMP}/d_vw_grid"   "${TMP}/in_pan.hdf5" "${DF[@]}" --ps-volume-weighted
run "${TMP}/d_vw_legacy" "${TMP}/in_pan.hdf5" "${DF[@]}" --ps-volume-weighted --pts-conventions legacy
run "${TMP}/d_both_grid" "${TMP}/in_crw.hdf5" "${DF[@]}" "${VMW[@]}"
echo ">> E: --partition 2 2 2 vs one triangulation (--ps-vertex-mass)"
run "${TMP}/e_single" "${TMP}/in_crw.hdf5" "${DF[@]}" "${VMW[@]}"
run "${TMP}/e_part"   "${TMP}/in_crw.hdf5" "${DF[@]}" "${VMW[@]}" --partition 2 2 2 --max-concurrent 2

echo ">> checks"
PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
"${PY}" - "${TMP}" "${GRID}" "${ALPHA}" "${AMP}" "${BOX}" "${BIN}" <<'PY'
import os, sys
import numpy as np
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
tmp, g, alpha, amp, box, binary = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
fails = []

def check(tag, ok, msg):
    print(f"   {'OK  ' if ok else 'FAIL'} {tag} {msg}")
    if not ok:
        fails.append(tag)

def pts(root, ext, dtype=np.float64):
    return np.fromfile(f"{tmp}/{root}.pts_{ext}", dtype=dtype)

def outputs(root):
    return sorted(f[len(root) + 1:] for f in os.listdir(tmp)
                  if f.startswith(root + ".") and not f.endswith((".log", ".runlog")) and ".runlog" not in f)

def same_files(r1, r2, select):
    names = [e for e in outputs(r1) if select(e)]
    diff = [e for e in names if open(f"{tmp}/{r1}.{e}", "rb").read() != open(f"{tmp}/{r2}.{e}", "rb").read()]
    return names, diff

# ---- A
names, diff = same_files("a_off_grid", "a_off_legacy", lambda e: True)
check("A1", len(names) > 10 and not diff, f"no grid option: 'grid' and 'legacy' byte-identical in all {len(names)} outputs "
      f"(differ: {diff})")
names, diff = same_files("a_on_grid", "a_on_legacy", lambda e: not e.startswith("pts_"))
check("A2", len(names) >= 5 and not diff, f"--ps-vertex-mass --ps-volume-weighted: the {len(names)} grids byte-identical "
      f"between the two point conventions (differ: {diff})")
_, diff = same_files("a_on_grid", "a_on_legacy", lambda e: e.startswith("pts_"))
check("A3", "pts_den" in diff and "pts_vel" in diff, f"... and the points differ where they should ({', '.join(diff)})")

# ---- B
def slope(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / np.dot(b, b))
def slab_blocks(root):
    p = pts(root, "den").reshape(4 * g, 4 * g, 4)
    return p.reshape(g, 4, g, 4, 4).mean(axis=(1, 3, 4)) - 1.0
def grid_layer(root):
    return np.fromfile(f"{tmp}/{root}.a_den", dtype=REAL).astype(np.float64).reshape(g, g, g)[:, :, g // 2] - 1.0
# analytic cell averages of the separable crossed waves (x = q + A sin kq per axis, single stream): the fraction
# of Lagrangian length mapped into each cell, per axis (the jitter of the lattice aside)
k = 2. * np.pi / box
A = amp / k
def q_of(x):
    lo, hi = x - 2. * A - 1., x + 2. * A + 1.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mid + A * np.sin(k * mid) < x else (lo, mid)
    return 0.5 * (lo + hi)
qe = np.array([q_of(x) for x in np.arange(g + 1) * box / g])
f1 = np.diff(qe) / (box / g)
ana = np.einsum("i,j->ij", f1, f1) * f1[g // 2] - 1.0
for est in ("dtfe", "geometric"):
    grid = grid_layer(f"b_{est}_grid")
    sg, sl = slope(slab_blocks(f"b_{est}_grid"), grid), slope(slab_blocks(f"b_{est}_legacy"), grid)
    ag, al = slope(slab_blocks(f"b_{est}_grid"), ana), slope(slab_blocks(f"b_{est}_legacy"), ana)
    st = slope(slab_blocks(f"b_{est}_grid"), slab_blocks(f"b_{est}_true"))
    print(f"   {est}: plane-vs-grid slope grid {sg:.4f} legacy {sl:.4f}; vs analytic grid {ag:.4f} legacy {al:.4f}; "
          f"grid plane vs the true-IC plane {st:.4f}; exact grid vs analytic {slope(grid, ana):.4f}")
    check(f"B1 {est}", abs(sg - 1.) < 0.05 and abs(ag - 1.) < 0.05,
          f"'grid': the plane's contrast is the grid's (slope {sg:.3f}) and the true one ({ag:.3f})")
    # the waves are not linear (contrast up to ~3), which deepens the suppression a little beyond 1-alpha
    check(f"B2 {est}", abs(sl - (1. - alpha)) < 0.08 and abs(al - (1. - alpha)) < 0.08 and sg - sl > 0.12,
          f"'legacy': the evolved-IC suppression, ~1-alpha = {1. - alpha:.3f} (slopes {sl:.3f}, {al:.3f})")
    check(f"B3 {est}", abs(st - 1.) < 0.02, f"'grid': the perturbed-IC plane equals the true-IC plane (slope {st:.4f})")
gs = grid_layer("b_sampled_grid")
print(f"   (the production nSub=3 sampled deposit of this box: plane-vs-grid slope 'grid' "
      f"{slope(slab_blocks('b_sampled_grid'), gs):.3f}, 'legacy' {slope(slab_blocks('b_sampled_legacy'), gs):.3f}; "
      f"that grid against the analytic cells {slope(gs, ana):.3f}, residual rms {np.std(gs - ana):.3f} -- "
      f"the exact deposit's {np.std(grid_layer('b_dtfe_grid') - ana):.3f})")

# ---- C
for est in ("dtfe", "geometric"):
    full = pts(f"c_{est}_full", "den")
    rnd = pts(f"c_{est}_rand", "den")
    cells = np.fromfile(f"{tmp}/c_{est}_full.a_den", dtype=REAL).astype(np.float64)
    se = rnd.std() / np.sqrt(rnd.size)
    check(f"C1 {est}", abs(rnd.mean() - 1.) < max(5. * se, 1e-3),
          f"2e5 random points average {rnd.mean():.5f} (rho_bar = 1; standard error {se:.1e})")
    check(f"C2 {est}", full.std() < 0.08 and full.std() < 1.2 * cells.std(),
          f"point noise std {full.std():.4f} on the near-lattice (the vertex-share grid's own: {cells.std():.4f} per cell)")

# ---- D
def ragged(root):
    off = pts(root, "stream_offsets", np.uint64).astype(np.int64)
    rec = pts(root, "stream_records").reshape(-1, 4)
    return off, rec
for root, weighted in (("d_vw_grid", False), ("d_vw_legacy", True), ("d_both_grid", False)):
    off, rec = ragged(root)
    vel = pts(root, "vel").reshape(-1, 3)
    n = np.diff(off)
    idx = np.repeat(np.arange(n.size), n)
    w = rec[:, 0] if weighted else np.ones(len(rec))
    num = np.zeros((n.size, 3))
    np.add.at(num, idx, rec[:, 1:] * w[:, None])
    den = np.zeros(n.size)
    np.add.at(den, idx, w)
    ok = den > 0
    expect = num[ok] / den[ok][:, None]
    err = float(np.abs(vel[ok] - expect).max() / np.abs(expect).max())
    check(f"D1 {root}", err < 1e-12, f"'.pts_vel' is the {'density-weighted' if weighted else 'plain'} mean of the "
          f"stream velocities: max relative difference {err:.1e} over {int(ok.sum())} points "
          f"({int((n > 1).sum())} multi-stream)")
same = lambda e: open(f"{tmp}/d_vw_grid.pts_{e}", "rb").read() == open(f"{tmp}/d_vw_legacy.pts_{e}", "rb").read()
check("D2", all(same(e) for e in ("den", "streams", "velDisp", "stream_records", "denGrad")),
      "--ps-volume-weighted alone: '.pts_den', '.pts_streams', '.pts_velDisp' (mass-weighted, as in the grids), the "
      "records and '.pts_denGrad' byte-identical between 'grid' and 'legacy'")
s = pts("d_vw_grid", "streams", np.int32)
vg, vl = pts("d_vw_grid", "velGrad").reshape(-1, 9), pts("d_vw_legacy", "velGrad").reshape(-1, 9)
v1, v2 = pts("d_vw_grid", "vel").reshape(-1, 3), pts("d_vw_legacy", "vel").reshape(-1, 3)
one, multi = s == 1, s > 1
e1 = float(max(np.abs(vg[one] - vl[one]).max() / np.abs(vl).max(), np.abs(v1[one] - v2[one]).max() / np.abs(v2).max()))
dm = float(np.abs(v1[multi] - v2[multi]).max() / np.abs(v2).max()) if multi.any() else 0.
check("D3", e1 < 1e-12 and dm > 1e-3, f"single-stream points agree ({e1:.1e}); multi-stream ones differ "
      f"(max {dm:.3f} of the peak velocity, {int(multi.sum())} points)")

# ---- E
worst = 0.
for e, dt in (("den", np.float64), ("vel", np.float64), ("velDisp", np.float64), ("velGrad", np.float64)):
    a, b = pts("e_single", e, dt), pts("e_part", e, dt)
    worst = max(worst, float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)))
same_s = np.array_equal(pts("e_single", "streams", np.int32), pts("e_part", "streams", np.int32))
tol = 1e-12 if REAL == np.float64 else 2e-5
check("E1", same_s and worst < tol, f"--partition 2 2 2 vs one triangulation: streams identical, values within "
      f"{worst:.1e} of their peak (float rounding of the periodic copies; at most {tol:.0e})")

# ---- F
from dtfelib import Estimator
P = np.fromfile(f"{tmp}/in_plane.bin").reshape(-1, 3)
prec = "double" if REAL == np.float64 else "single"
with Estimator(f"{tmp}/in_crw.hdf5", mpc_unit=1, per_stream=True, density_gradient=True, velocity_gradient=True,
               binary=os.path.abspath(binary), precision=prec, options=["--ps-vertex-mass", "--ps-volume-weighted"]) as est:
    f = est(P)
    ok = (np.array_equal(f.density, pts("d_both_grid", "den"))
          and np.array_equal(f.velocity.ravel(), pts("d_both_grid", "vel"))
          and np.array_equal(f.dispersion.ravel(), pts("d_both_grid", "velDisp"))
          and np.array_equal(f.velocity_gradient.ravel(), pts("d_both_grid", "velGrad"))
          and np.array_equal(f.density_gradient.ravel(), pts("d_both_grid", "denGrad"))
          and np.array_equal(np.column_stack([f.stream_density, f.stream_velocity]).ravel(),
                             pts("d_both_grid", "stream_records")))
    g2 = est(P[::397])          # a small request: the server's cell index, the same vertex ratios
    ok2 = np.array_equal(g2.density, f.density[::397]) and np.array_equal(g2.velocity, f.velocity[::397])
check("F1", ok and ok2, "--serve with the two options: bit-identical to --sample-points (full walk and a small request)")

print("-" * 60)
if fails:
    print("RESULT: FAIL (" + ", ".join(fails) + ")")
    sys.exit(1)
print("RESULT: PASS")
PY
echo "============================================================"
echo " point conventions check PASSED"
echo "============================================================"
