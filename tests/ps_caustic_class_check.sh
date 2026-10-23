#!/usr/bin/env bash
# PS-DTFE caustic-stratification check ('.causticClass', written by --ps-caustics on the CPU).
#
# The stratification comes from the deformation tensor J = Ax * Lag^-1, which is CONSTANT inside a
# linearly mapped tetrahedron and therefore exact: the number of negative real eigenvalues of J is
# the number of principal axes along which the sheet has collapsed and inverted (k = 0 uncollapsed,
# 1 wall, 2 filament, 3 node). Asserted here on the crossed-wave snapshot, where three independent
# 1-D collapses make the multiplicity structure known by construction:
#  A) '.causticClass' is written and uses only the documented bits (0..6).
#  B) The parity bits agree EXACTLY with '.caustic': cells with both bits set are precisely the
#     cells the fold flag marks. (This is what proves the class bits leave the fold bits alone.)
#  C) k = 0 in every single-stream cell -- no axis can have collapsed where there is one stream.
#  D) The mean stream count increases strictly with k: uncollapsed < wall < filament < node.
#  E) The GPU deposit reproduces BOTH '.caustic' and '.causticClass' byte-for-byte; only the
#     --ps-linear-deposit combination falls back to the parity-only fold flag, and says so.
#  J) The fold flag is the GEOMETRIC overlap of a folded tetrahedron, not a per-sample
#     hit: on thin young folds it equals dtfelib's (streams > 1 + tol) | (hidden_streams & 1) cell for
#     cell, is the same for nSub 1, nSub 3 and the exact deposit, and is split- and GPU-invariant.
#
# Usage: tests/ps_caustic_class_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-32}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"; mkdir -p "${TMP}"
SNAP="${TMP}/pcc_input.hdf5"
OUT="${TMP}/pcc_out"

echo "============================================================"
echo " PS-DTFE caustic stratification check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

rm -f "${OUT}".*
echo ">> generating the crossed-wave snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box "${BOX}" \
        --crossed-waves >/dev/null

echo ">> CPU deposit with --ps-caustics ..."
"${BIN}" "${SNAP}" "${OUT}" --grid "${GRID}" --periodic --field density --MpcUnit 1 \
         --max-concurrent 1 --ps-caustics >/dev/null

[ -f "${OUT}.causticClass" ] || { echo "   FAIL  no .causticClass written"; exit 1; }

"${PY}" - "${OUT}" <<'EOF'
import sys
import numpy as np
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh

root = sys.argv[1]
cls = np.fromfile(root + ".causticClass", dtype=REAL)
fold = np.fromfile(root + ".caustic", dtype=REAL)
strm = np.fromfile(root + ".streams", dtype=REAL)
m = cls.astype(np.int64)

fails = 0
def check(ok, msg):
    global fails
    print(f"   {'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        fails += 1

# exact integers survived the float round-trip, and only documented bits are used
check(np.all(cls == np.floor(cls)), "mask values are exact integers")
check(int(m.max()) <= 0x7F, f"only bits 0..6 used (max mask {int(m.max())})")

# (B) parity bits reproduce the fold flag exactly
both = ((m & 3) == 3)
check(int(both.sum()) == int((fold == 1).sum()) and np.array_equal(both, fold == 1),
      f"parity bits match .caustic exactly ({int(both.sum())} fold cells)")

# (C) no collapsed axis where there is a single stream
COLLAPSE = (1 << 2, 1 << 3, 1 << 4, 1 << 5)
kmax = np.full(m.shape, -1, dtype=np.int64)
for k, bit in enumerate(COLLAPSE):
    kmax = np.where((m & bit) != 0, k, kmax)
single = np.round(strm) == 1
check(single.sum() > 0 and bool((kmax[single] == 0).all()),
      f"k = 0 in all {int(single.sum())} single-stream cells")

# (D) mean stream count strictly increases with the collapse multiplicity
means = [float(strm[kmax == k].mean()) for k in range(4) if (kmax == k).any()]
check(len(means) >= 3 and all(a < b for a, b in zip(means, means[1:])),
      "mean streams increase strictly with k: " + " < ".join(f"{x:.1f}" for x in means))

# (D4) the umbilic indicator must live ON the caustic: two eigenvalues vanishing together makes the
# map singular there by construction, so a flagged cell on no fold cannot be an umbilic. Without the
# at-zero half of the test 15% of cells are flagged, two thirds of them off-fold.
umb = ((m >> 6) & 1) != 0
fold_cells = (m & 3) == 3
check(int((umb & ~fold_cells).sum()) == 0,
      f"every umbilic-indicator cell lies on a fold ({int(umb.sum())} flagged, "
      f"{int((umb & ~fold_cells).sum())} off-fold)")
check(int(umb.sum()) < 0.02 * int(fold_cells.sum()),
      f"umbilic indicator stays rare: {int(umb.sum())} cells = "
      f"{100*umb.sum()/max(int(fold_cells.sum()),1):.2f}% of fold cells, bound 2%")

print("")
print(f"   cells per k: " + ", ".join(f"k={k}:{int((kmax==k).sum())}" for k in range(4)))
print(f"   umbilic-indicator cells (bit6): {int(umb.sum())}")
sys.exit(1 if fails else 0)
EOF
py_rc=$?

echo ""
echo "(F) per-point caustic mask ('.pts_caustic', --ps-caustics + --sample-points)"
POINTS="${TMP}/pcc_points.bin"
"${PY}" - "$POINTS" "$BOX" <<'EOF'
import numpy as np, sys
rng = np.random.default_rng(11)
L = float(sys.argv[2])
rng.uniform(0.02*L, 0.98*L, size=(60000, 3)).astype(np.float64).tofile(sys.argv[1])
EOF
"${BIN}" "${SNAP}" "${OUT}_pts" --grid 16 --periodic --field density --MpcUnit 1 \
         --max-concurrent 1 --ps-caustics --sample-points "${POINTS}" >/dev/null
if [ ! -f "${OUT}_pts.pts_caustic" ]; then
    echo "   FAIL  no .pts_caustic written"
    py_rc=1
else
"${PY}" - "${OUT}_pts" <<'EOF' || py_rc=1
import sys
import numpy as np

root = sys.argv[1]
m = np.fromfile(root + ".pts_caustic", dtype=np.int32).astype(np.int64)
s = np.fromfile(root + ".pts_streams", dtype=np.int32)

fails = 0
def check(ok, msg):
    global fails
    print(f"   {'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        fails += 1

COLLAPSE = (1 << 2, 1 << 3, 1 << 4, 1 << 5)
k = np.full(m.shape, -1, dtype=np.int64)
for i, bit in enumerate(COLLAPSE):
    k = np.where((m & bit) != 0, i, k)

check(int(m.max()) <= 0x7F, f"only bits 0..6 used (max mask {int(m.max())})")

# One-way implications, and the identity they combine to. With exact point location a
# multi-stream point always lies in an inverted tetrahedron (the map has degree 1, so
# streams = 1 + 2 x folds crossed), whose negative eigenvalue gives k >= 1; so k = 0 <=>
# streams = 1 exactly. (A +-1e-6 tolerance, which lets a point on a shared face be claimed by
# both tetrahedra, makes it only a near-identity.)
covered = s > 0
check(bool((k[s == 1] == 0).all()), f"every single-stream point has k = 0 ({int((s==1).sum())} points)")
check(bool((s[(m & 3) == 3] > 1).all()), "every point flagged on a fold is multi-stream")
nbad = int((((k == 0) != (s == 1)) & covered).sum())
check(nbad == 0, f"k=0 and streams=1 agree on every covered point ({nbad} of {int(covered.sum())} disagree)")

# Crossed waves: three independent 1-D collapses, so k collapsed axes <=> 3^k streams.
ok = True
for i in range(4):
    sel = covered & (k == i)
    if sel.sum() < 50:
        continue
    mean = float(s[sel].mean())
    good = abs(mean - 3.0**i) / (3.0**i) < 0.05
    print(f"      k={i}: n={int(sel.sum()):6d}  mean streams={mean:8.3f}  expected {3**i}")
    ok = ok and good
check(ok, "mean stream count matches the analytic 3^k for every k")

sys.exit(1 if fails else 0)
EOF
fi

echo ""
echo "(G) A3 cusp indicator ('--ps-caustic-cusps', bit 7). The 1-D Zel'dovich wave is the analytic"
echo "    control: its caustic is two PARALLEL PLANES, so at A*k=1.8 there are no cusps anywhere,"
echo "    while at A*k=1.0 the fold is exactly tangent (lambda_c = 1+cos(kq) is stationary at kq=pi)."
for A in 1.8 1.0; do
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/pcc_w${A}.hdf5" --n 32 \
            --box "${BOX}" --amplitude-factor "${A}" >/dev/null
    "${BIN}" "${TMP}/pcc_w${A}.hdf5" "${TMP}/pcc_c${A}" --grid 48 --periodic --field density \
             --MpcUnit 1 --max-concurrent 1 --ps-caustics --ps-caustic-cusps >/dev/null
done
# and one default run: the opt-in flag must be the ONLY way bit 7 appears
"${BIN}" "${TMP}/pcc_w1.8.hdf5" "${TMP}/pcc_nocusp" --grid 48 --periodic --field density \
         --MpcUnit 1 --max-concurrent 1 --ps-caustics >/dev/null
# --ps-gpu with the cusp indicator: the deposit runs on the CPU (the stencil needs the triangulation's neighbours),
# so bits 7 and 8 are the CPU run's, byte for byte -- a GPU deposit left them unset in nearly every cell
GPUG=0
[ -d o_ps ] && [ ! -f o_ps/.gpu_mode_off ] && grep -q "METAL=1\|CUDA=1\|HIP=1" o_ps/.build_mode 2>/dev/null && GPUG=1
if [ "${GPUG}" = 1 ]; then
    "${BIN}" "${TMP}/pcc_w1.0.hdf5" "${TMP}/pcc_c1.0_gpu" --grid 48 --periodic --field density \
             --MpcUnit 1 --max-concurrent 1 --ps-caustics --ps-caustic-cusps --ps-gpu > "${TMP}/pcc_c1.0_gpu.log" 2>&1
fi
"${PY}" - "${TMP}" "${GPUG}" <<'PYEOF' || py_rc=1
import sys
import numpy as np
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh

tmp = sys.argv[1]
def mask(name):
    return np.fromfile(f"{tmp}/{name}.causticClass", dtype=REAL).astype(np.int64)

fails = 0
def check(ok, msg):
    global fails
    print(f"   {'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        fails += 1

d = mask("pcc_nocusp")
check(int((((d >> 7) & 1) != 0).sum()) == 0, "without --ps-caustic-cusps bit 7 is never set")

g = mask("pcc_c1.8")
fold = ((g & 3) == 3)
cusp = ((g >> 7) & 1) != 0
rate = float(cusp.sum()) / max(int(fold.sum()), 1)
# Finite differences on an irregular tessellation are never perfectly clean, but the
# vertex-incident stencil keeps the floor on this provably cusp-free case at ~0.01% of the
# fold cells (~7% with only the four face neighbours). The bound is set well above the
# measured value and far below the narrow stencil's, so it catches a regression back to it.
check(rate < 0.01, f"A*k=1.8 (no cusps exist): false-positive rate {rate*100:.3f}% of fold cells, bound 1%")

t = mask("pcc_c1.0")
tang = int((((t >> 7) & 1) != 0).sum())
check(tang > 0, f"A*k=1.0 (fold exactly tangent): {tang} cells flagged on the tangency plane")

if len(sys.argv) > 2 and sys.argv[2] == "1":
    import filecmp
    log = open(f"{tmp}/pcc_c1.0_gpu.log").read()
    gt = mask("pcc_c1.0_gpu")
    check(filecmp.cmp(f"{tmp}/pcc_c1.0.causticClass", f"{tmp}/pcc_c1.0_gpu.causticClass", shallow=False)
          and "needs the CPU deposit" in log,
          f"--ps-gpu with --ps-caustic-cusps deposits on the CPU: '.causticClass' byte-identical to the CPU run "
          f"({int((((gt >> 7) & 1) != 0).sum())} cusp cells)")
else:
    print("   SKIP  --ps-gpu with --ps-caustic-cusps (CPU-only build)")

sys.exit(1 if fails else 0)
PYEOF

echo ""
echo "(H) A4 swallowtail indicator (bit 8). Three 1-D waves whose stratum is known exactly:"
echo "    A*k=1.8 is a generic A2 fold (no A3, no A4); A*k=1.0 is an exact A3 (lambda_c is"
echo "    stationary but D2 = k^2 != 0, so it is NOT a swallowtail); and the three-harmonic wave"
echo "    --swallowtail is built so lambda_c = (8/7)(cos kq + 1/2)^3 has a TRIPLE root, i.e."
echo "    D1 = D2 = 0 with D3 != 0 -- an exact A4 at q = L/3 and 2L/3."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/pcc_a4.hdf5" --n 32 \
        --box "${BOX}" --swallowtail >/dev/null
"${BIN}" "${TMP}/pcc_a4.hdf5" "${TMP}/pcc_a4" --grid 48 --periodic --field density \
         --MpcUnit 1 --max-concurrent 1 --ps-caustics --ps-caustic-cusps >/dev/null
"${PY}" - "${TMP}" <<'PYEOF' || py_rc=1
import sys
import numpy as np
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh

tmp = sys.argv[1]
G = 48
def mask(name):
    return np.fromfile(f"{tmp}/{name}.causticClass", dtype=REAL).astype(np.int64)

fails = 0
def check(ok, msg):
    global fails
    print(f"   {'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        fails += 1

def counts(name):
    d = mask(name)
    return int((((d >> 7) & 1) != 0).sum()), int((((d >> 8) & 1) != 0).sum())

# bit 8 is opt-in exactly like bit 7
nc3, nc4 = counts("pcc_nocusp")
check(nc4 == 0, "without --ps-caustic-cusps bit 8 is never set")

# a generic fold must stay completely dark -- this is the property worth protecting
_, a2 = counts("pcc_c1.8")
check(a2 == 0, f"A*k=1.8 (generic A2 fold): {a2} cells flagged A4, want 0")

# an exact A3 is the hard control: lambda_c IS tangent there, only D2 tells it apart from an A4
c3_a3, c4_a3 = counts("pcc_c1.0")
floor = c4_a3 / max(c3_a3, 1)
# Bounds sit BETWEEN the deduplicated nearest-64 stencil (0.51% floor / 74.5% signal / ~146x) and
# a duplicate-filled gather (1.38% / 17.2% / 12x), so a regression to that gather FAILS.
check(floor < 0.01,
      f"A*k=1.0 (exact A3, not A4): false-positive floor {floor*100:.2f}% of cusp cells, bound 1%")

# and the real thing must light up well clear of that floor
c3_a4, c4_a4 = counts("pcc_a4")
rate = c4_a4 / max(c3_a4, 1)
check(rate > 0.40,
      f"three-harmonic (exact A4): {c4_a4} cells = {rate*100:.2f}% of cusp cells, bound 40%")
# With the deduplicated stencil the floor measures exactly zero, so the ratio is unbounded; state
# the separation as "the floor is far below the signal" rather than dividing by it.
check(floor * 20.0 < rate,
      f"A4 signal ({rate*100:.2f}%) clears the A3 floor ({floor*100:.2f}%) by more than 20x")

# The flagged cells must sit where the analytic A4 planes land in EULERIAN space: the wave displaces
# q = L/3 and 2L/3 to x ~= 49 and 51. A flag scattered over the volume would be noise, whatever the
# counts say.
sw = (((mask("pcc_a4") >> 8) & 1) != 0).reshape(G, G, G)
slabs = np.where(sw.any(axis=(1, 2)))[0]
check(len(slabs) > 0 and slabs.min() >= 21 and slabs.max() <= 26,
      f"A4 cells confined to the analytic planes: x-slabs {slabs.tolist()} within [21,26]")

sys.exit(1 if fails else 0)
PYEOF

echo ""
echo "(I) D4 umbilic indicator (bit 6) on a wave with NO caustic anywhere. A*k=0.15 never shell-"
echo "    crosses, so there is no singularity of any kind to indicate. The transverse eigenvalues"
echo "    are nevertheless exactly equal everywhere (nothing displaces y or z), which is precisely"
echo "    the trap: 'two eigenvalues coincide' is true in the whole box and, as the criterion, flags 97.6%"
echo "    of it. An umbilic needs the coincident pair to sit AT ZERO, not merely to be equal."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/pcc_weak.hdf5" --n 32 \
        --box "${BOX}" --amplitude-factor 0.15 >/dev/null
"${BIN}" "${TMP}/pcc_weak.hdf5" "${TMP}/pcc_weak" --grid 48 --periodic --field density \
         --MpcUnit 1 --max-concurrent 1 --ps-caustics >/dev/null
"${PY}" - "${TMP}" <<'PYEOF' || py_rc=1
import sys
import numpy as np
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh

tmp = sys.argv[1]
m = np.fromfile(f"{tmp}/pcc_weak.causticClass", dtype=REAL).astype(np.int64)
fails = 0
def check(ok, msg):
    global fails
    print(f"   {'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        fails += 1

touched = int((m != 0).sum())
check(touched > 0, f"the weak wave actually deposited ({touched} cells)")
check(int(((m & 3) == 3).sum()) == 0, "no fold cells (the wave does not shell-cross)")
umb = int((((m >> 6) & 1) != 0).sum())
check(umb == 0, f"no umbilic indicator anywhere: {umb} cells flagged of {touched}, want 0")

sys.exit(1 if fails else 0)
PYEOF

echo ""
echo "(J) The fold flag is GEOMETRIC: '.caustic' = a folded tetrahedron"
echo "    overlaps the cell, not 'a folded tetrahedron deposited a sample here'. Two geometries just past"
echo "    shell crossing, whose young walls are thinner than the sampling: the pancake (amplitude 1.05) and"
echo "    the crossed waves (1.3), 27 particles per axis on 48^3 so no lattice lines up. A flag per deposited"
echo "    sample marks only 4364 of 4608 and 15844 of 25408 fold cells there. Asserted, cell for cell:"
echo "    J1 '.caustic' == dtfelib's fold, (streams > 1 + tol) | (hidden_streams & 1), from either pair;"
echo "    J2 folds the samples missed ARE flagged (non-vacuous), and are exactly the hidden-bit cells;"
echo "    J3 the flag does not depend on the deposit (nSub 1 == nSub 3 == the exact deposit) and is exactly the"
echo "       cells the exact deposit gives more than one stream (r3d volumes: shares no code with the flag);"
echo "    J4 bit 0 (per sample) is present in every fold cell, so '(bits & 3) == 3' loses none;"
echo "    J5 '--partition 2 2 2' and the GPU deposit give byte-identical '.caustic'."
GPUJ=0
[ -d o_ps ] && [ ! -f o_ps/.gpu_mode_off ] && grep -q "METAL=1\|CUDA=1\|HIP=1" o_ps/.build_mode 2>/dev/null && GPUJ=1
for geo in pan crw; do
    if [ "${geo}" = pan ]; then gen=(--amplitude-factor 1.05); else gen=(--amplitude-factor 1.3 --crossed-waves); fi
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/pcj_input_${geo}.hdf5" --n 27 --box "${BOX}" \
            "${gen[@]}" >/dev/null
    J=( "${TMP}/pcj_input_${geo}.hdf5" )
    JO=( --grid 48 --periodic --MpcUnit 1 --ps-caustics )
    rm -f "${TMP}/pcj_${geo}"_*.*
    "${BIN}" "${J[@]}" "${TMP}/pcj_${geo}_s"   "${JO[@]}" --max-concurrent 1 --field density density_a --avg-subsamples 3 >/dev/null
    "${BIN}" "${J[@]}" "${TMP}/pcj_${geo}_a3"  "${JO[@]}" --max-concurrent 1 --field density_a --avg-subsamples 3 >/dev/null
    "${BIN}" "${J[@]}" "${TMP}/pcj_${geo}_ex"  "${JO[@]}" --max-concurrent 1 --field density --ps-exact-deposit >/dev/null
    "${BIN}" "${J[@]}" "${TMP}/pcj_${geo}_par" "${JO[@]}" --field density density_a --avg-subsamples 3 --partition 2 2 2 >/dev/null
    if [ "${GPUJ}" = 1 ]; then
        "${BIN}" "${J[@]}" "${TMP}/pcj_${geo}_gpu" "${JO[@]}" --max-concurrent 1 --field density density_a \
                 --avg-subsamples 3 --ps-gpu >/dev/null
    fi
done
"${PY}" - "${TMP}" "${ROOT}/python" "${GPUJ}" <<'PYEOF' || py_rc=1
import filecmp
import sys
import numpy as np
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh

tmp, pydir, gpu = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
sys.path.insert(0, pydir)
from dtfelib.io import STREAM_TOL, caustic_is_fold, single_stream_mask

fails = 0
def check(ok, msg):
    global fails
    print(f"   {'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        fails += 1

def L(root, ext):
    return np.fromfile(f"{tmp}/{root}{ext}", dtype=REAL)

for geo, title in (("pan", "pancake"), ("crw", "crossed waves")):
    print(f"   {title}:")
    s = f"pcj_{geo}_s"
    fold = L(s, ".caustic") == 1
    cc = np.rint(L(s, ".causticClass")).astype(np.int64)
    st, hid = L(s, ".streams"), L(s, ".hidden_streams")
    rec_u = caustic_is_fold(streams=st, hidden=hid)
    rec_a = caustic_is_fold(streams=L(s, ".a_streams"), hidden=L(s, ".a_hidden_streams"))
    n_u, n_a = int((fold ^ rec_u).sum()), int((fold ^ rec_a).sum())
    check(n_u == 0 and n_a == 0 and fold.sum() > 0,
          f"J1 '.caustic' == (streams > 1 + tol) | (hidden & 1): {int(fold.sum())} fold cells, "
          f"{n_u} differ (raw pair), {n_a} differ (averaged pair)")
    check(np.array_equal(caustic_is_fold(cc, streams=st, hidden=hid), fold) and np.array_equal(caustic_is_fold(cc), fold),
          "J1 caustic_is_fold(mask) and caustic_is_fold(mask, streams, hidden) both reproduce it")
    near = np.abs(st - 1.0) <= STREAM_TOL
    thin = fold & near                                       # a fold cell whose one sample saw one stream
    hbit = (np.rint(hid).astype(np.int32) & 1) != 0
    check(int(thin.sum()) > 0 and np.array_equal(thin, hbit & near),
          f"J2 {int(thin.sum())} fold cells whose samples missed the fold are flagged, exactly the hidden-bit cells")
    check(np.array_equal(~single_stream_mask(st, hid) & (st >= 1.0 - STREAM_TOL), fold),
          "J2 fold == not single-stream (outside empty cells): the flag and the single-stream mask partition the grid")
    a3, ex = L(f"pcj_{geo}_a3", ".caustic") == 1, L(f"pcj_{geo}_ex", ".caustic") == 1
    check(np.array_equal(a3, fold) and np.array_equal(ex, fold),
          f"J3 nSub 1 == nSub 3 == exact deposit ({int((a3 ^ fold).sum())} / {int((ex ^ fold).sum())} cells differ)")
    # J3 against a reference sharing no code with the flag: the exact deposit's '.streams' is the analytic cell-mean
    # multiplicity (r3d clipping, not the separating-axis test), > 1 exactly where folded volume sits in the cell
    exm = L(f"pcj_{geo}_ex", ".streams").astype(np.float64) - 1.0
    nbad = int(((exm > 2e-5) ^ fold).sum())
    check(nbad == 0 and float(exm[fold].min()) > 0.01,
          f"J3 fold == exact-deposit multiplicity > 1 + 2e-5 ({nbad} cells differ; least excess on a fold cell "
          f"{float(exm[fold].min()):.3f} streams)")
    neg_only = int((((cc & 3) == 2)).sum())
    check(neg_only == 0 and bool((cc[fold] & 1).all()), f"J4 bit 0 set in every fold cell ({neg_only} det<0-only cells)")
    check(filecmp.cmp(f"{tmp}/{s}.caustic", f"{tmp}/pcj_{geo}_par.caustic", shallow=False),
          "J5 '--partition 2 2 2' '.caustic' byte-identical")
    if gpu:
        check(filecmp.cmp(f"{tmp}/{s}.caustic", f"{tmp}/pcj_{geo}_gpu.caustic", shallow=False),
              "J5 GPU '.caustic' byte-identical (the host marks the GPU's folded tetrahedra)")
    else:
        print("   SKIP  J5 GPU (CPU-only build)")
sys.exit(1 if fails else 0)
PYEOF

echo ""
echo "(E) GPU behaviour: BOTH the fold flag and the full stratification must match the CPU exactly"
echo "    (the double build, whose GPU works on float copies, to 1e-4 of the cells)."
echo "    The per-tet mask reaches the kernel through the otherwise-unused 'dens' buffer, so only the"
echo "    --ps-linear-deposit combination (which needs that buffer) falls back to parity."
# GPU grid == CPU grid ($1 = extension): byte for byte in the single build. The double build hands the GPU
# float copies of its double tetrahedra (ps_double_check D5), so a sample on a face may count differently there:
# up to 1e-4 of the cells may differ (a cell whose GPU sample counts 2 streams instead of 1 differs in its
# fold class too)
gpu_matches() {
    cmp -s "${OUT}.$1" "${OUT}_gpu.$1" && return 0
    precision_double || return 1
    "${PY}" - "${OUT}.$1" "${OUT}_gpu.$1" <<'EOF2'
import sys
import numpy as np
a, b = (np.fromfile(f, dtype=np.float64) for f in sys.argv[1:3])
fr = float(np.mean(a != b)) if a.size == b.size and a.size else 1.0
print(f"         (double build: {fr:.1e} of the cells differ -- the GPU's float copies of the tetrahedra; <= 1e-4 passes)")
sys.exit(0 if fr <= 1e-4 else 1)
EOF2
}
if [ -d o_ps ] && [ ! -f o_ps/.gpu_mode_off ]; then   # any GPU backend stamp (metal / cuda / hip)
    msg="$("${BIN}" "${SNAP}" "${OUT}_gpu" --grid "${GRID}" --periodic --field density --MpcUnit 1 \
            --max-concurrent 1 --ps-caustics --ps-gpu 2>&1 || true)"
    if gpu_matches caustic; then
        echo "   PASS  GPU '.caustic' matches the CPU (byte for byte; the double build to 1e-4 of the cells)"
    else
        echo "   FAIL  GPU '.caustic' differs from the CPU"
        py_rc=1
    fi
    if gpu_matches causticClass; then
        echo "   PASS  GPU '.causticClass' matches the CPU (mask rides in the dens buffer; byte for byte, the double build to 1e-4 of the cells)"
    else
        echo "   FAIL  GPU '.causticClass' differs from the CPU"
        py_rc=1
    fi
    # --ps-linear-deposit needs 'dens' for real densities, so the mask cannot ride along: that
    # combination must fall back to the parity-only fold flag AND say so.
    lin="$("${BIN}" "${SNAP}" "${OUT}_gpulin" --grid "${GRID}" --periodic --field density --MpcUnit 1 \
            --max-concurrent 1 --ps-caustics --ps-linear-deposit --ps-gpu 2>&1 || true)"
    if grep -q "ps-linear-deposit" <<< "${lin}"; then
        echo "   PASS  --ps-caustics --ps-linear-deposit --ps-gpu warns that the stratification is CPU-only"
    else
        echo "   FAIL  the --ps-linear-deposit GPU combination did not warn"
        py_rc=1
    fi
else
    echo "   SKIP  (CPU-only build)"
fi

echo ""
echo "------------------------------------------------------------"
if [ "${py_rc}" -eq 0 ]; then
    echo "ps_caustic_class_check: ALL PASS"
    exit 0
fi
echo "ps_caustic_class_check: FAILURES"
exit 1
