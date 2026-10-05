#!/usr/bin/env bash
# Exact point location + the '--serve' / dtfelib.Estimator contract, both binaries.
#
# What is verified, and why these are the right bounds:
#  X) EXACT, TIE-CONSISTENT STREAM COUNTING (PS-DTFE --sample-points). On crossed waves every
#     point off a caustic has an ODD number of streams (a folded sheet covers it 1, 3, 9 or 27
#     times), so any even count is a double-count or a gap -- the bound is exactly zero:
#       X1  grid cell centres: no even count. (The old +-1e-6 barycentric tolerance counted a
#           point within 1e-6 of a shared face twice: ~50 even counts per 2.1M points.)
#       X2  AT PARTICLE POSITIONS -- every query sits ON a vertex shared by ~25 tetrahedra:
#           no even count, mean streams < 12. (The tolerance counted the whole vertex star:
#           mean ~27, 69% even.)
#       (The single-tessellation runs below pass --partition 1 1 1: since 2026-10-01 the auto-tuner
#       splits a small set for SPEED, and a split's sums differ from one tessellation's at float
#       rounding; serve == sample-points holds for the SAME partitioning.)
#       X3  --partition 2 2 2 gives byte-identical stream counts at both point sets (every
#           tie is broken the same way in every partition: periodic copies are mapped to one
#           canonical coordinate).
#       X4  1 thread vs the default pool: byte-identical outputs.
#  D) THE GRID DEPOSIT uses the same exact test for its sub-samples, and at nSub=1 its samples
#     are the cell centres, so its '.streams' IS the exact count there:
#       D1  '.streams' == the X1 point counts in every cell. (With the old +-1e-6 tolerance
#           ~67 of 2.1M cells read one stream too many -- a sample near a shared face counted
#           by both tetrahedra.)
#       D2  --partition 2 2 2 gives a byte-identical '.streams'. (The tolerance tripped
#           differently on each run's periodic copies: 2 of 2.1M cells differed.)
#       D3  (GPU builds) the --ps-gpu deposit reaches the SAME decisions: '.streams' and
#           '.hidden_streams' byte-identical to the CPU's. The kernel classifies in float with a
#           rigorous error bound and defers the undecidable tets to the CPU's exact test.
#       D4  (GPU builds) the same with --partition 2 2 2 -- whose periodic sub-grids WRAP the
#           seam, which the GPU's sampled path once skipped (most of each partition's cells).
#  S) STANDARD DTFE --sample-points: coverage is exactly 1 at every cell centre AND every
#     particle position (tiny/flat tetrahedra are no longer dropped, and a query at a vertex
#     lands in exactly one tetrahedron).
#  V) --serve through dtfelib.Estimator, both binaries: answers are BIT-IDENTICAL to a
#     --sample-points run on the same points (same code, same reduction) -- for a request
#     answered through the server's cell index (few points) and for one that walks the whole
#     tessellation (many points); every optional output (--per-stream, --per-stream-ids,
#     --pts-den-grad, --pts-vel-grad) round-trips unchanged; a request with a NaN is
#     rejected and the session keeps serving; grid() returns the tensor-product shape.
#  C) COMPOSITE SERVER (--serve --partition, CosmoDTFE's CompositeEstimator, out of core):
#       C1  PS, all partitions in memory: bit-identical to --sample-points --partition 2 2 2
#           (every output incl. per-stream records, ids, scalars, Lagrangian positions).
#       C2  PS with --tessellation-cache and ONE resident partition (every request reloads
#           partitions from disk): bit-identical again; a second session starts from the
#           cache + occupancy files without building (its log says 'cached'), same answers.
#       C3  standard DTFE composite (Eulerian partitions, centroid ownership): coverage exactly
#           1 at cell centres AND particle positions, density within 1e-5 of the single server
#           (partitions round their periodic copies differently), both with 2x2x2 and 3x2x1.
#       C4  --serve + --tessellation-cache under a memory budget too small for ONE tessellation
#           (DTFE_MEM_BUDGET_GB): the auto-tuner makes it a composite by itself, and its answers
#           equal --sample-points with the split it chose.
#       C5  the automatic resident set counts BYTES (DTFE_SERVE_BUDGET_GB): with room for all 8
#           tessellations nothing reloads, and a cell index is built only when it fits too (else
#           the request walks); with room for 3 a request needing 8 evicts and reloads; answers
#           bit-identical throughout.
#  L) PER-STREAM LAGRANGIAN POSITIONS AND SCALARS (--pts-lagrangian, --pts-scalar):
#       L1  every stream satisfies x - q(x) = v/100 (Zel'dovich: v = 100 s, and x, q, v are
#           interpolated with the same barycentric weights) to float rounding.
#       L2  a periodic per-particle scalar f(q) (--scalar-dataset) comes back within the linear-
#           interpolation bound; the point value is the density-weighted stream mean.
#  A) Estimator.from_arrays: numpy arrays in, bit-identical to the snapshot-file server (PS);
#     a non-periodic standard cloud with negative coordinates reproduces a linear field.
#  T) the tessellation cache keys on --periodic: a non-periodic run never loads the cache a
#     periodic run wrote (it did until 2026-09-30).
#
# Usage:
#   tests/point_exact_serve_check.sh              # build, run, check
#   tests/point_exact_serve_check.sh --no-build
# Requires python3 with numpy and h5py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
PY="${PYTHON:-python3}"

N="${N:-32}"; GRID="${GRID:-64}"; BOX=100.0
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
# tessellation caches must live on a LOCAL, non-synced volume (the repo may sit in iCloud Drive)
TESS_BASE="$(mktemp -d "${TMPDIR:-/tmp}/dtfe-pxs-tess.XXXXXX")"
trap 'rm -rf "${TESS_BASE}"' EXIT
SNAP="${TMP}/pxs_input_crossed.hdf5"
CENTRES="${TMP}/pxs_input_centres.bin"
PARTS="${TMP}/pxs_input_particles.bin"

echo "============================================================"
echo " exact point location + --serve check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building DTFE + PS-DTFE ..."
    make DTFE    $(cat o/.build_mode 2>/dev/null || true) >/dev/null
    make PS-DTFE $(cat o_ps/.build_mode 2>/dev/null || true) >/dev/null
fi
precision_require "${DTFE_BIN}" "${PS_BIN}"
for b in "${DTFE_BIN}" "${PS_BIN}"; do [ -x "$b" ] || { echo "FAIL: $b not built"; exit 1; }; done

echo ">> generating crossed waves (streams 1/3/9/27) + cell-centre and particle point sets ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.5 --crossed-waves >/dev/null
"${PY}" - "${SNAP}" "${CENTRES}" "${PARTS}" "${GRID}" "${BOX}" <<'PY'
import sys
import h5py
import numpy as np
snap, centres, parts, grid, box = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
c = (np.arange(grid) + 0.5) * box / grid
x, y, z = np.meshgrid(c, c, c, indexing="ij")
np.stack([x.ravel(), y.ravel(), z.ravel()], 1).astype(np.float64).tofile(centres)
with h5py.File(snap, "r+") as f:    # the particles' own float32 positions, exactly
    np.asarray(f["PartType1/Coordinates"][:], dtype=np.float64).tofile(parts)
    q = f["PartType1/InitialCoordinates"][:].astype(np.float64)
    # a smooth PERIODIC per-particle scalar for --pts-scalar (L2)
    f["PartType1"].create_dataset("Potential", data=np.sin(2*np.pi*q[:, 0]/box) + 0.5*np.cos(2*np.pi*q[:, 1]/box))
PY

run() {  # $1 = binary, $2 = output root, rest = extra args
    local bin="$1" out="$2"; shift 2
    rm -f "${out}".*
    local log="${out}.log"
    set +e
    "${bin}" "${SNAP}" "${out}" --grid "${RUN_GRID:-16}" --periodic --field density --input 105 --MpcUnit 1 \
        --verbose 1 "$@" > "$log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "   ERROR: ${bin} exited with code $rc -- last 25 lines of $log:"
        tail -n 25 "$log" | sed 's/^/      | /'
        exit 1
    fi
}

echo ">> PS-DTFE runs: serial, --partition 2 2 2, 1 thread; standard DTFE runs"
run "${PS_BIN}" "${TMP}/pxs_ps_c"    --sample-points "${CENTRES}"
run "${PS_BIN}" "${TMP}/pxs_ps_p"    --sample-points "${PARTS}"
run "${PS_BIN}" "${TMP}/pxs_ps_c_P2" --sample-points "${CENTRES}" --partition 2 2 2
run "${PS_BIN}" "${TMP}/pxs_ps_p_P2" --sample-points "${PARTS}"   --partition 2 2 2
DTFE_PTS_THREADS=1 run "${PS_BIN}" "${TMP}/pxs_ps_c_T1" --sample-points "${CENTRES}"
run "${DTFE_BIN}"    "${TMP}/pxs_st_c"    --sample-points "${CENTRES}"
run "${DTFE_BIN}"    "${TMP}/pxs_st_p"    --sample-points "${PARTS}"
# the grid deposit on the cell-centre grid itself (nSub=1: one sample per cell, at its centre)
RUN_GRID="${GRID}" run "${PS_BIN}" "${TMP}/pxs_dep"    --partition 1 1 1
RUN_GRID="${GRID}" run "${PS_BIN}" "${TMP}/pxs_dep_P2" --partition 2 2 2
GPU_BUILT=0
OBJ_PS=o_ps; precision_double && OBJ_PS=o_ps_d
[ -f ${OBJ_PS}/.gpu_mode_off ] || GPU_BUILT=1
if [ "${GPU_BUILT}" -eq 1 ]; then
    RUN_GRID="${GRID}" run "${PS_BIN}" "${TMP}/pxs_dep_gpu"    --partition 1 1 1 --ps-gpu
    RUN_GRID="${GRID}" run "${PS_BIN}" "${TMP}/pxs_dep_gpu_P2" --partition 2 2 2 --ps-gpu
fi
# reference outputs for the --serve round trip: every optional point output switched on
run "${PS_BIN}" "${TMP}/pxs_ps_full" --sample-points "${PARTS}" --per-stream --per-stream-ids --partition 1 1 1 \
    --pts-den-grad --pts-vel-grad
# references for the composite server (C) and the per-stream Lagrangian/scalar outputs (L)
run "${PS_BIN}" "${TMP}/pxs_ps_comp_ref" --sample-points "${PARTS}" --per-stream --per-stream-ids \
    --pts-lagrangian --pts-scalar --scalar-dataset Potential --partition 2 2 2
# (T) cache identity: a periodic run writes the cache, a non-periodic one must not load it
TESS_T="${TESS_BASE}/T"; mkdir -p "${TESS_T}"
run "${DTFE_BIN}" "${TMP}/pxs_T_per" --tessellation-cache "${TESS_T}"
set +e
"${DTFE_BIN}" "${SNAP}" "${TMP}/pxs_T_np" --grid 16 --field density --input 105 --MpcUnit 1 \
    --tessellation-cache "${TESS_T}" --verbose 1 > "${TMP}/pxs_T_np.log" 2>&1
set -e

echo ">> checks"
PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
"${PY}" - "${TMP}" "${SNAP}" "${CENTRES}" "${PARTS}" "${GPU_BUILT}" "${BOX}" "${N}" "${TESS_BASE}" <<'PY'
import sys
import h5py
import numpy as np
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh

tmp, snap, centres, parts = sys.argv[1:5]
gpu_built = sys.argv[5] == "1"
fails = []

def check(tag, ok, msg):
    print(f"   {'OK  ' if ok else 'FAIL'} {tag} {msg}")
    if not ok:
        fails.append(tag)

def pts(root, ext, dtype=np.float64):
    return np.fromfile(f"{tmp}/{root}.pts_{ext}", dtype=dtype)

s_c = pts("pxs_ps_c", "streams", np.int32)
s_p = pts("pxs_ps_p", "streams", np.int32)
check("X1", (s_c % 2 == 0).sum() == 0,
      f"cell centres: {(s_c % 2 == 0).sum()} even stream counts of {s_c.size} (must be 0); "
      f"counts {dict(zip(*[a.tolist() for a in np.unique(s_c, return_counts=True)]))}")
check("X2", (s_p % 2 == 0).sum() == 0 and s_p.mean() < 12.,
      f"particle positions: {(s_p % 2 == 0).sum()} even of {s_p.size}, mean streams "
      f"{s_p.mean():.2f} (must be 0 and < 12; the vertex star is ~25)")
same_c = np.array_equal(s_c, pts("pxs_ps_c_P2", "streams", np.int32))
same_p = np.array_equal(s_p, pts("pxs_ps_p_P2", "streams", np.int32))
check("X3", same_c and same_p, f"--partition 2 2 2 stream counts byte-identical: "
      f"centres {same_c}, particles {same_p}")
same_t = all(np.array_equal(pts("pxs_ps_c", e), pts("pxs_ps_c_T1", e))
             for e in ("den", "vel", "velDisp"))
check("X4", same_t, "1 thread vs the default pool: den/vel/velDisp byte-identical")

g1 = np.fromfile(f"{tmp}/pxs_dep.streams", dtype=REAL)
g2 = np.fromfile(f"{tmp}/pxs_dep_P2.streams", dtype=REAL)
nd = int((np.rint(g1).astype(np.int32) != s_c).sum())
check("D1", g1.size == s_c.size and nd == 0 and np.array_equal(g1, np.rint(g1)),
      f"grid deposit '.streams' (nSub=1) vs the exact counts at the cell centres: {nd} of "
      f"{g1.size} cells differ (must be 0)")
check("D2", np.array_equal(g1, g2), f"--partition 2 2 2 grid '.streams' byte-identical: "
      f"{int((g1 != g2).sum())} cells differ")
if gpu_built:
    u1 = np.fromfile(f"{tmp}/pxs_dep.hidden_streams", dtype=REAL)
    for tag, root in (("D3", "pxs_dep_gpu"), ("D4", "pxs_dep_gpu_P2")):
        gs = np.fromfile(f"{tmp}/{root}.streams", dtype=REAL)
        gu = np.fromfile(f"{tmp}/{root}.hidden_streams", dtype=REAL)
        nd = int((gs != g1).sum()) + int((gu != u1).sum())
        if _os.environ.get("DTFE_TEST_PRECISION") == "double":
            # the double build hands the float GPU kernels float copies of its tetrahedra, so a sample
            # within float rounding of a face may land differently than in the CPU's double geometry
            # (ps_double_check D5); a mapping bug (D4's guard: 91% of cells) is still far outside this
            check(tag, nd <= 1e-4 * g1.size,
                  f"--ps-gpu{' --partition 2 2 2' if tag == 'D4' else ''} vs CPU (double build): '.streams' "
                  f"{int((gs != g1).sum())} and '.hidden_streams' {int((gu != u1).sum())} cells differ "
                  f"(at most {1e-4 * g1.size:.0f}: float kernel geometry)")
        else:
            check(tag, nd == 0,
                  f"--ps-gpu{' --partition 2 2 2' if tag == 'D4' else ''} vs CPU: '.streams' "
                  f"{int((gs != g1).sum())} and '.hidden_streams' {int((gu != u1).sum())} cells differ (must be 0)")
else:
    print("   SKIP D3/D4 (CPU-only build)")

st_c = pts("pxs_st_c", "streams", np.int32)
st_p = pts("pxs_st_p", "streams", np.int32)
check("S1", st_c.min() == 1 and st_c.max() == 1,
      f"standard DTFE coverage at cell centres: min {st_c.min()} max {st_c.max()} (must be exactly 1)")
check("S2", st_p.min() == 1 and st_p.max() == 1,
      f"standard DTFE coverage at particle positions: min {st_p.min()} max {st_p.max()} (must be exactly 1)")

from dtfelib import Estimator as _Estimator
import os as _os2
PREC = _os2.environ.get("DTFE_TEST_PRECISION", "single")      # tests/precision.sh
PSBIN = "./PS-DTFE-double" if PREC == "double" else "./PS-DTFE"
class Estimator(_Estimator):          # every server of this suite (from_arrays too) runs the precision under test
    def __init__(self, *a, **kw):
        kw.setdefault("precision", PREC)
        super().__init__(*a, **kw)

P = np.fromfile(parts).reshape(-1, 3)
C = np.fromfile(centres).reshape(-1, 3)
with Estimator(snap, mpc_unit=1, per_stream=True, per_stream_ids=True, density_gradient=True,
               velocity_gradient=True) as est:
    f = est(P)                                   # walks the whole tessellation
    ok = (np.array_equal(f.density, pts("pxs_ps_full", "den"))
          and np.array_equal(f.velocity.ravel(), pts("pxs_ps_full", "vel"))
          and np.array_equal(f.dispersion.ravel(), pts("pxs_ps_full", "velDisp"))
          and np.array_equal(f.streams, pts("pxs_ps_full", "streams", np.int32))
          and np.array_equal(f.density_gradient.ravel(), pts("pxs_ps_full", "denGrad"))
          and np.array_equal(f.velocity_gradient.ravel(), pts("pxs_ps_full", "velGrad"))
          and np.array_equal(f.offsets, pts("pxs_ps_full", "stream_offsets", np.uint64))
          and np.array_equal(np.column_stack([f.stream_density, f.stream_velocity]).ravel(),
                             pts("pxs_ps_full", "stream_records"))
          and np.array_equal(f.stream_ids.ravel(), pts("pxs_ps_full", "stream_ids", np.uint64))
          and np.array_equal(f.stream_density_gradient.ravel(), pts("pxs_ps_full", "stream_dengrad")))
    check("V1", ok, "PS --serve, all optional outputs: bit-identical to --sample-points")

    rng = np.random.default_rng(3)
    sel = rng.choice(len(P), 50, replace=False)  # few points -> the server's cell index
    g = est(P[sel])
    lo, hi = f.offsets[sel], f.offsets[sel + 1]
    ok = (np.array_equal(g.density, f.density[sel]) and np.array_equal(g.streams, f.streams[sel])
          and np.array_equal(g.velocity, f.velocity[sel])
          and all(np.array_equal(g.stream_ids[g.stream_slice(i)], f.stream_ids[lo[i]:hi[i]])
                  for i in range(len(sel))))
    check("V2", ok, "PS --serve, 50-point request (cell index) == the same points of a full walk")

    try:
        est([[np.nan, 1.0, 2.0]])
        rejected = False
    except ValueError:
        rejected = True
    alive = est(P[:3]).density
    check("V3", rejected and np.array_equal(alive, f.density[:3]),
          "a NaN request is rejected and the session keeps answering")

    xs = np.linspace(5., 95., 4)
    q = est.grid(xs, xs, [50.])
    check("V4", q.density.shape == (4, 4, 1) and q.velocity.shape == (4, 4, 1, 3)
          and q.streams.shape == (4, 4, 1), f"grid() shapes {q.density.shape} {q.velocity.shape}")

with Estimator(snap, mpc_unit=1, phase_space=False) as est:
    a = est(C)
    b = est(C[::997])
    ok = (np.array_equal(a.density, pts("pxs_st_c", "den"))
          and np.array_equal(a.streams, st_c)
          and np.array_equal(b.density, a.density[::997]))
    check("V5", ok, "standard --serve: full walk and cell-index requests bit-identical to --sample-points")

# ---------------- C: the composite server
kw = dict(mpc_unit=1, per_stream=True, per_stream_ids=True, lagrangian_positions=True,
          scalar_dataset="Potential")
def same_as_ref(f, root):
    return (np.array_equal(f.density, pts(root, "den")) and np.array_equal(f.streams, pts(root, "streams", np.int32))
            and np.array_equal(f.velocity.ravel(), pts(root, "vel"))
            and np.array_equal(f.offsets, pts(root, "stream_offsets", np.uint64))
            and np.array_equal(np.column_stack([f.stream_density, f.stream_velocity]).ravel(), pts(root, "stream_records"))
            and np.array_equal(f.stream_ids.ravel(), pts(root, "stream_ids", np.uint64))
            and np.array_equal(f.scalar, pts(root, "scalar")) and np.array_equal(f.stream_scalar, pts(root, "stream_scalar"))
            and np.array_equal(f.stream_lagrangian.ravel(), pts(root, "stream_lagpos")))
with Estimator(snap, partition=2, **kw) as est:
    comp = est(P)
    check("C1", est.n_partitions == 8 and same_as_ref(comp, "pxs_ps_comp_ref"),
          f"PS composite ({est.n_partitions} partitions in memory) == --sample-points --partition 2 2 2, every output")
import os, shutil, glob
tess = f"{sys.argv[8]}/C"
shutil.rmtree(tess, ignore_errors=True); os.makedirs(tess)
with Estimator(snap, partition=2, tessellation_cache=tess, resident=1, **kw) as est:
    a = est(P); b = est(P[:40])
    ok1 = est.resident == 1 and same_as_ref(a, "pxs_ps_comp_ref") and np.array_equal(b.density, a.density[:40])
occ = sorted(glob.glob(f"{tess}/*.occ"))
stamps = {f: os.stat(f).st_mtime_ns for f in glob.glob(f"{tess}/*")}
with Estimator(snap, partition=2, tessellation_cache=tess, resident=1, **kw) as est:
    c = est(P)
untouched = stamps == {f: os.stat(f).st_mtime_ns for f in glob.glob(f"{tess}/*")}
check("C2", ok1 and same_as_ref(c, "pxs_ps_comp_ref") and len(occ) == 8 and untouched,
      f"out-of-core composite (1 resident, reloads from the cache) bit-identical; a second session "
      f"starts from the {len(occ)} cache + occupancy files without rewriting any ({untouched})")
with h5py.File(snap) as fh:
    Xp = fh["PartType1/Coordinates"][:].astype(np.float64)
Q = np.vstack([C[::7], Xp[::3]])
with Estimator(snap, mpc_unit=1, phase_space=False) as est:
    single = est(Q)
ok, msg = True, []
for part in (2, (3, 2, 1)):
    with Estimator(snap, mpc_unit=1, phase_space=False, partition=part) as est:
        f = est(Q)
    rel = float(np.max(np.abs(f.density - single.density) / single.density))
    ok &= bool(f.streams.min() == 1 and f.streams.max() == 1 and rel < 1e-5)
    msg.append(f"{part}: coverage {f.streams.min()}..{f.streams.max()}, max rel {rel:.1e}")
check("C3", ok, "standard composite, " + "; ".join(msg))
import subprocess
os.environ["DTFE_MEM_BUDGET_GB"] = "0.02"
try:
    os.makedirs(f"{sys.argv[8]}/A")
    with Estimator(snap, mpc_unit=1, tessellation_cache=f"{sys.argv[8]}/A") as est:
        auto = est(P)
        nparts = est.n_partitions
finally:
    del os.environ["DTFE_MEM_BUDGET_GB"]
side = round(nparts ** (1 / 3))
ref = subprocess.run([PSBIN, snap, f"{tmp}/pxs_auto_ref", "--grid", "16", "--periodic", "--field", "density",
                      "--input", "105", "--MpcUnit", "1", "--sample-points", parts, "--partition",
                      str(side), str(side), str(side)], capture_output=True, text=True)
same = ref.returncode == 0 and np.array_equal(auto.density, pts("pxs_auto_ref", "den")) \
    and np.array_equal(auto.streams, pts("pxs_auto_ref", "streams", np.int32))
check("C4", nparts > 1 and side ** 3 == nparts and same,
      f"a budget too small for one tessellation: the server chose {nparts} partitions by itself; answers "
      f"== --sample-points --partition {side} {side} {side}: {same}")

import re
def budget_session(budget_gb, requests, name):
    """a composite session from the C2 cache under DTFE_SERVE_BUDGET_GB; its answers, its reported
    resident count and its progress log"""
    log = f"{sys.argv[8]}/{name}.log"
    os.environ["DTFE_SERVE_BUDGET_GB"] = repr(float(budget_gb))
    try:
        with Estimator(snap, partition=2, tessellation_cache=tess, progress=True, log_path=log, **kw) as est:
            outs = [est(q) for q in requests]
            nres = est.resident
    finally:
        del os.environ["DTFE_SERVE_BUDGET_GB"]
    return outs, nres, open(log, errors="replace").read()
def loaded_per_request(text):
    return [int(v) for v in re.findall(r"\[request done\] \d+ points, \d+ of \d+ partitions, (\d+) loaded", text)]
_, _, text = budget_session(1000., [P[:5]], "c5_probe")
nv = sorted(int(v) for v in re.findall(r"\[partitions \d+/\d+\] partition \d+: [^,]+, (\d+) vertices", text))
TB, IB = 666e-9, 192e-9                       # GB per vertex: tessellation, index estimate (ps_point_eval.cc)
small = [P[i:i + 5] for i in range(0, 50, 5)]  # few points: the cell-index path once a partition earned it
ok5, msg5 = len(nv) == 8, [f"{len(nv)} partitions"]
if ok5:
    # (a) everything fits: nothing reloads, and the indexes get built once earned
    (ra, _, *_), nres_a, ta = budget_session(1000., [P, P] + small, "c5_all")
    la = loaded_per_request(ta)
    built_a = ta.count("building its cell index")
    # (b) the 8 tessellations fit but no index on top: all stay resident, every request walks
    gb_b = TB * sum(nv) + 0.5 * IB * nv[0]
    (rb, _, *_), nres_b, tb = budget_session(gb_b, [P, P] + small, "c5_noindex")
    lb = loaded_per_request(tb)
    # (c) room for the 3 largest only: requests needing all 8 evict and reload, answers unchanged
    gb_c = TB * sum(nv[-3:]) * 1.0001
    (rc, rc2), nres_c, tc = budget_session(gb_c, [P, P], "c5_three")
    # the handshake reports how many of the LARGEST partition fit
    fits = lambda gb: max(1, min(8, int(gb * 1e9 // (666. * nv[-1]))))
    lc = loaded_per_request(tc)
    ok5 = (nres_a == 8 and la[1] == 0 and built_a >= 1
           and nres_b == fits(gb_b) and lb[1] == 0 and tb.count("building its cell index") == 0
           and tb.count("would not fit in memory, walking") >= 1
           and nres_c == fits(gb_c) and 1 <= lc[1] <= 5
           and same_as_ref(ra, "pxs_ps_comp_ref") and same_as_ref(rb, "pxs_ps_comp_ref")
           and same_as_ref(rc, "pxs_ps_comp_ref") and same_as_ref(rc2, "pxs_ps_comp_ref"))
    msg5 = [f"ample: {nres_a} resident, 2nd request loaded {la[1]}, {built_a} index(es) built",
            f"tessellations only: all stay ({nres_b} of the largest fit), 2nd loaded {lb[1]}, indexes skipped "
            f"{tb.count('would not fit in memory, walking')}x",
            f"3 largest: {nres_c} of the largest fit, 2nd loaded {lc[1]}", "answers bit-identical"]
check("C5", ok5, "automatic resident set by bytes: " + "; ".join(msg5))

# ---------------- L: per-stream Lagrangian positions and scalars
L = float(sys.argv[6])
counts = np.diff(comp.offsets.astype(np.int64))
X = np.repeat(P, counts, axis=0)
s = X - comp.stream_lagrangian
s -= L * np.round(s / L)
err = float(np.abs(s - comp.stream_velocity / 100.).max())
check("L1", err < 5e-5 and comp.stream_lagrangian.min() >= 0 and comp.stream_lagrangian.max() < L,
      f"x - q(x) = v/100 on every stream: max error {err:.1e} Mpc (float rounding); q inside the box")
qq = comp.stream_lagrangian
e2 = float(np.abs(comp.stream_scalar - (np.sin(2*np.pi*qq[:, 0]/L) + 0.5*np.cos(2*np.pi*qq[:, 1]/L))).max())
h = L / float(sys.argv[7])
bound = 1.5 * (h*np.sqrt(3))**2 / 8 * (2*np.pi/L)**2
seg = np.repeat(np.arange(len(P)), counts)
w = comp.stream_density
mean = np.bincount(seg, w*comp.stream_scalar, len(P)) / np.bincount(seg, w, len(P))
check("L2", e2 < bound and np.allclose(comp.scalar, mean, rtol=1e-12, atol=1e-12),
      f"per-stream scalar vs f(q): max error {e2:.2g} (linear-interpolation bound {bound:.2g}); point "
      f"value = density-weighted stream mean")

# ---------------- A: Estimator.from_arrays
with h5py.File(snap) as fh:
    g = fh["PartType1"]
    ax, av, aq, ai, apot = (g[k][:] for k in ("Coordinates", "Velocities", "InitialCoordinates", "ParticleIDs", "Potential"))
with Estimator(snap, mpc_unit=1, peculiar=False, per_stream=True, per_stream_ids=True,
               lagrangian_positions=True, scalar_dataset="Potential") as est:
    r1 = est(P[:500])
with Estimator.from_arrays(ax, velocities=av, lagrangian=aq, ids=ai, values=apot, box=L,
                           per_stream=True, per_stream_ids=True, lagrangian_positions=True) as est:
    r2 = est(P[:500])
    tmpdir = est._owned_dir
same = all(np.array_equal(getattr(r1, k), getattr(r2, k)) for k in
           ("density", "velocity", "streams", "scalar", "stream_scalar", "stream_lagrangian", "stream_ids", "offsets"))
rng = np.random.default_rng(5)
cloud = rng.uniform(-3, 7, (3000, 3))
with Estimator.from_arrays(cloud, masses=rng.uniform(0.5, 2, 3000), values=cloud[:, 0]) as est:
    lin = est(np.array([[2., 2., 2.], [0., 1., 5.]]))
check("A1", same and not tmpdir.exists() and lin.streams.min() == 1
      and float(np.abs(lin.scalar - [2., 0.]).max()) < 1e-5,
      "from_arrays: bit-identical to the snapshot server (PS); temp file removed; a negative-"
      "coordinate standard cloud reproduces a linear field")

# ---------------- T: the cache keys on --periodic
np_log = open(f"{tmp}/pxs_T_np.log").read()
d_per = np.fromfile(f"{tmp}/pxs_T_per.den", dtype=REAL)
d_np = np.fromfile(f"{tmp}/pxs_T_np.den", dtype=REAL)
check("T1", "cache hit" not in np_log and d_np.size == d_per.size and not np.array_equal(d_np, d_per),
      "a non-periodic run does not load the tessellation a periodic run cached (its densities "
      "differ, as they must; the old cache key had no 'periodic' and served the periodic ones)")

print("------------------------------------------------------------")
if fails:
    print(f"RESULT: FAIL ({', '.join(fails)})")
    sys.exit(1)
print("RESULT: PASS  (exact tie-consistent point location in both binaries; --serve answers "
      "bit-identical to --sample-points; composite servers, per-stream q(x)/scalars, from_arrays)")
PY

echo "============================================================"
echo " exact point location + --serve check PASSED"
echo "============================================================"
