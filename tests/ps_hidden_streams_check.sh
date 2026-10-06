#!/usr/bin/env bash
# PS-DTFE '.hidden_streams' bit 1 is EXACT: a cell it leaves clear, with '.streams' == 1, is
# single-stream at every point -- and it no longer drops single-stream cells just because a
# tetrahedron of their own stream is too small to hold a sub-sample.
#
# The flag (quantities.h) is geometric: a folded (negatively oriented) tetrahedron overlaps the
# cell with positive volume. In a periodic box the map has degree 1, so a point is multi-stream
# exactly where a folded tetrahedron covers it. Checked against two references that share no
# code with the flag:
#
#  X) DENSE EXACT POINT EVALUATION: 4^3 points per cell (--sample-points, the exact SoS
#     inside test). Every point of a cell with streams==1 and no bit 1 must be single-stream
#     (0 violations, nSub=1 and nSub=3 grids). The cells the old conservative mass flag
#     dropped but bit 1 keeps are counted and must exist (else the test is vacuous).
#  Y) THE EXACT DEPOSIT (--ps-exact-deposit): its '.streams' is the analytic cell-mean stream
#     count (1/V_cell) sum V(tet ^ cell), which is 1 iff the cell is single-stream almost
#     everywhere. Unflagged single-stream cells must read 1 (to float rounding); EVERY cell
#     whose multi-stream volume the samples missed (streams==1 but exact > 1) must be flagged;
#     flagged cells must hold real multi-stream volume (exact deposit or a dense point).
#  Z) SAME FLAG EVERYWHERE: serial vs --partition 2 2 2 and CPU vs GPU (a Metal build) give
#     byte-identical '.hidden_streams' and '.streams' on both grids -- except the partitioned
#     '.a_streams', a float sum of count/nSub^3 per partition (a few ulps, pre-existing).
#
# Two geometries just past shell crossing, where the walls are thinner than the sampling (the
# case the flag exists for): the Zel'dovich pancake (plane folds, amplitude 1.05) and
# --crossed-waves (3D folds, amplitude 1.3). 27 particles per axis on a 48^3 grid, so the
# particle lattice and the sample lattices do not line up. On the crossed waves the samples miss
# the walls of ~12k cells; the old mass flag caught only ~40% of them.
#
# Usage:
#   tests/ps_hidden_streams_check.sh              # build, run, check
#   tests/ps_hidden_streams_check.sh --no-build
# Requires the build toolchain plus python3 with numpy and h5py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-27}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"; K="${K:-4}"
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"; mkdir -p "${TMP}"
# input/point file names must NOT share a prefix with the output roots (rm -f "<root>".*)
SNAP_PAN="${TMP}/pux_input_pancake.hdf5"
SNAP_CRW="${TMP}/pux_input_crossed.hdf5"
PTS="${TMP}/pux_input_dense.bin"

echo "============================================================"
echo " PS-DTFE exact '.hidden_streams' check   N=${N}^3  grid=${GRID}^3  ${K}^3 points/cell"
echo "============================================================"

if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }
GPU=0
grep -q "METAL=1\|CUDA=1\|HIP=1" o_ps/.build_mode 2>/dev/null && GPU=1

echo ">> generating the pancake, the crossed waves and ${K}^3 points per cell ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_PAN}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.05 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_CRW}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.3 --crossed-waves >/dev/null
"${PY}" - "${PTS}" "${GRID}" "${BOX}" "${K}" <<'PY'
import sys
import numpy as np
out, grid, box, k = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
# cell-major order: the K^3 points of cell (i,j,k) are consecutive, off the nSub=1/3 sample lattices
off = (np.arange(k) + 0.5) / k
c = np.arange(grid)
ci, cj, ck, oi, oj, ok = np.meshgrid(c, c, c, off, off, off, indexing="ij")
h = box / grid
pts = np.stack([(ci + oi) * h, (cj + oj) * h, (ck + ok) * h], axis=-1).reshape(-1, 3)
pts.astype(np.float64).tofile(out)
PY

run() {  # $1 = output root, $2 = snapshot, rest = options (files first: --partition takes 3 values)
    local out="$1" snap="$2"; shift 2
    rm -f "${out}".*
    local log="${out}.log"
    set +e
    "${BIN}" "${snap}" "${out}" --grid "${GRID}" --input 105 --MpcUnit 1 --verbose 1 --periodic "$@" > "$log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "   ERROR: PS-DTFE exited with code $rc -- last 25 lines of $log:"
        tail -n 25 "$log" | sed 's/^/      | /'
        exit 1
    fi
}

for geo in pan crw; do
    snap="${SNAP_PAN}"; [ "$geo" = crw ] && snap="${SNAP_CRW}"
    echo ">> ${geo}: sampled deposit, nSub 1 and 3 (CPU, serial)"
    run "${TMP}/pux_${geo}" "${snap}" --field density density_a --avg-subsamples 3
    echo ">> ${geo}: the same, --partition 2 2 2"
    run "${TMP}/pux_${geo}_par" "${snap}" --field density density_a --avg-subsamples 3 --partition 2 2 2
    if [ "${GPU}" = 1 ]; then
        echo ">> ${geo}: the same on the GPU"
        run "${TMP}/pux_${geo}_gpu" "${snap}" --field density density_a --avg-subsamples 3 --ps-gpu
    fi
    echo ">> ${geo}: exact deposit"
    run "${TMP}/pux_${geo}_ex" "${snap}" --field density --ps-exact-deposit
    echo ">> ${geo}: dense exact point evaluation"
    run "${TMP}/pux_${geo}_pts" "${snap}" --field density --sample-points "${PTS}"
done

echo ">> checking ..."
"${PY}" - "${TMP}" "${GRID}" "${K}" "${GPU}" <<'PY'
import sys
import numpy as np
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
tmp, grid, k, gpu = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
ncell = grid ** 3
fails = []

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        fails.append(name)

def load(root, ext):
    a = np.fromfile(root + ext, dtype=REAL)
    assert a.size == ncell, (root + ext, a.size)
    return a

TOL = 1e-3     # dtfelib.STREAM_TOL / PS_STREAM_TOL
for geo, title in (("pan", "pancake"), ("crw", "crossed waves")):
    print(f"{title}:")
    base = f"{tmp}/pux_{geo}"
    pts = np.fromfile(f"{base}_pts.pts_streams", dtype=np.int32).reshape(ncell, k ** 3)
    cell_multi = (pts > 1).any(axis=1)                       # some dense point is multi-stream
    exact = load(f"{base}_ex", ".streams").astype(np.float64)  # (1/V) sum V(tet ^ cell)
    for tag, sext, uext in (("nSub=1", ".streams", ".hidden_streams"), ("nSub=3", ".a_streams", ".a_hidden_streams")):
        s = load(base, sext)
        u = np.rint(load(base, uext)).astype(np.int32)
        single = np.abs(s - 1.0) <= TOL
        keep = single & ((u & 1) == 0)                       # the new single-stream mask
        old_keep = single & (u == 0)                         # the old conservative one (bit 2 = old flag)
        bad = int((keep & cell_multi).sum())
        check(f"X [{tag}] every dense point of a kept cell is single-stream", bad == 0,
              f"{bad} of {int(keep.sum())} kept cells hold a multi-stream point")
        rec = int((keep & ~old_keep).sum())
        check(f"X [{tag}] single-stream cells the old mass flag dropped are kept now", rec > 0,
              f"{rec} cells recovered: the mask keeps {int(keep.sum())} of {int(single.sum())} "
              f"streams==1 cells, the old flag {int(old_keep.sum())}")
        dev = float(np.abs(exact[keep] - 1.0).max()) if keep.any() else 0.0
        check(f"Y [{tag}] the exact deposit reads 1 in every kept cell", dev < 2e-5,
              f"max |exact streams - 1| = {dev:.2e}")
        flagged = single & ((u & 1) != 0)
        missed = single & (exact > 1.0 + 2e-5)               # multi-stream volume the samples missed
        nm, nmf = int(missed.sum()), int((missed & flagged).sum())
        check(f"Y [{tag}] every wall the samples missed is flagged", nmf == nm and nm > 0,
              f"{nmf} of {nm} (the old mass flag: {int((missed & ((u & 2) != 0)).sum())})")
        confirmed = flagged & ((exact > 1.0 + 2e-5) | cell_multi)
        nf, nc = int(flagged.sum()), int(confirmed.sum())
        lowest = float((exact[flagged] - 1.0).min()) if nf else 0.0
        check(f"Y [{tag}] flagged cells hold real multi-stream volume", nc >= 0.95 * nf and lowest > -2e-5,
              f"{nc} of {nf} confirmed by the exact deposit or a dense point; min exact streams - 1 = {lowest:.2e}")
    for ext in (".streams", ".hidden_streams", ".a_streams", ".a_hidden_streams"):
        a, b = load(base, ext), load(f"{base}_par", ext)
        if ext == ".a_streams":                              # float sum of count/nSub^3 per partition
            d = float(np.abs(a.astype(np.float64) - b).max())
            check(f"Z serial == --partition 2 2 2 on {ext} (to float rounding)", d < 1e-5, f"max |diff| = {d:.1e}")
        else:
            check(f"Z serial == --partition 2 2 2 on {ext}", np.array_equal(a, b))
        if gpu and _os.environ.get("DTFE_TEST_PRECISION") == "double" and ext == ".a_streams":
            # double build: the GPU gets float copies of the double tetrahedra (ps_double_check D5), so a
            # sample on a face may count differently; the integer grids and both flag grids stay exact
            g = load(f"{base}_gpu", ext)
            fr = float(np.mean(a != g)); d = float(np.abs(a.astype(np.float64) - g).max())
            check(f"Z CPU ~ GPU on {ext} (double build: float kernel geometry)", fr <= 1e-4,
                  f"{fr:.1e} of cells differ, max |diff| = {d:.1e}")
        elif gpu:
            check(f"Z CPU == GPU on {ext}", np.array_equal(a, load(f"{base}_gpu", ext)))

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("   -", f)
    sys.exit(1)
print("RESULT: PASS  ('.hidden_streams' bit 1 is exact against dense points and the exact deposit)")
PY
