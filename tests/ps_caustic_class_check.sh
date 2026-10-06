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
#     cells the fold flag marks. (This is what proves the added bits did not disturb the old ones.)
#  C) k = 0 in every single-stream cell -- no axis can have collapsed where there is one stream.
#  D) The mean stream count increases strictly with k: uncollapsed < wall < filament < node.
#  E) The GPU deposit reproduces BOTH '.caustic' and '.causticClass' byte-for-byte; only the
#     --ps-linear-deposit combination falls back to the parity-only fold flag, and says so.
#
# Usage: tests/ps_caustic_class_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
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

if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
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
# at-zero half of the test this flagged 15% of cells, two thirds of them off-fold.
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
# streams = 1 exactly. (It was a near-identity while a +-1e-6 tolerance let a point on a
# shared face be claimed by both tetrahedra.)
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
"${PY}" - "${TMP}" <<'PYEOF' || py_rc=1
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
# vertex-incident stencil brought the floor on this provably cusp-free case down to ~0.01% of the
# fold cells (it was ~7% with only the four face neighbours). The bound is set well above the
# measured value and far below the old one, so it catches a regression back to a narrow stencil.
check(rate < 0.01, f"A*k=1.8 (no cusps exist): false-positive rate {rate*100:.3f}% of fold cells, bound 1%")

t = mask("pcc_c1.0")
tang = int((((t >> 7) & 1) != 0).sum())
check(tang > 0, f"A*k=1.0 (fold exactly tangent): {tang} cells flagged on the tangency plane")

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
# the old duplicate-filled gather (1.38% / 17.2% / 12x), so a regression to that gather FAILS.
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
echo "    the trap: 'two eigenvalues coincide' is true in the whole box and used to flag 97.6% of"
echo "    it. An umbilic needs the coincident pair to sit AT ZERO, not merely to be equal."
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
echo "(E) GPU behaviour: BOTH the fold flag and the full stratification must match the CPU exactly."
echo "    The per-tet mask reaches the kernel through the otherwise-unused 'dens' buffer, so only the"
echo "    --ps-linear-deposit combination (which needs that buffer) falls back to parity."
if [ -d o_ps ] && [ ! -f o_ps/.gpu_mode_off ]; then   # any GPU backend stamp (metal / cuda / hip)
    msg="$("${BIN}" "${SNAP}" "${OUT}_gpu" --grid "${GRID}" --periodic --field density --MpcUnit 1 \
            --max-concurrent 1 --ps-caustics --ps-gpu 2>&1 || true)"
    if cmp -s "${OUT}.caustic" "${OUT}_gpu.caustic"; then
        echo "   PASS  GPU '.caustic' is byte-identical to the CPU"
    else
        echo "   FAIL  GPU '.caustic' differs from the CPU"
        py_rc=1
    fi
    if cmp -s "${OUT}.causticClass" "${OUT}_gpu.causticClass"; then
        echo "   PASS  GPU '.causticClass' is byte-identical to the CPU (mask rides in the dens buffer)"
    else
        echo "   FAIL  GPU '.causticClass' differs from the CPU"
        py_rc=1
    fi
    # --ps-linear-deposit needs 'dens' for real densities, so the mask cannot ride along: that
    # combination must fall back to the parity-only fold flag AND say so.
    lin="$("${BIN}" "${SNAP}" "${OUT}_gpulin" --grid "${GRID}" --periodic --field density --MpcUnit 1 \
            --max-concurrent 1 --ps-caustics --ps-linear-deposit --ps-gpu 2>&1 || true)"
    if echo "${lin}" | grep -q "ps-linear-deposit"; then
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
