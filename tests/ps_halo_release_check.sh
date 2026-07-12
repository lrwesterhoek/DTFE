#!/usr/bin/env bash
# PS-DTFE --ps-halo-release correctness check.
#
# The halo-interior release deposits any tetrahedron whose geometric stream density
# rho_geo = rho_bar*V_lag/V_eul exceeds the threshold D monolithically at its Eulerian
# centroid cell (the mass-conserving sub-sample-spacing fallback) instead of rasterizing
# it over its bbox. Verified here on the Zel'dovich pancake AND a --crossed-waves snapshot:
#  A) the flag actually releases tetrahedra (D=2 is deliberately low so released tets are
#     large enough to have covered cell centres -- outputs must CHANGE in the multi-stream
#     region) while the grid means of .den and .a_den stay 1 to float accumulation (mass
#     conservation is exact per tetrahedron, so the mean is UNCHANGED vs the base run).
#  B) cells that are single-stream in the base run keep BIT-IDENTICAL .den/.vel/.streams:
#     released tets live in multi-stream regions by construction.
#  C) partition/thread invariance with the flag on: two '--partition 2 2 2' runs at 1 and
#     all threads give bit-exact .streams and float-rounding FP fields (the protocol the
#     deposit guarantees; serial-vs-partitioned keeps the KNOWN pre-existing +-1-stream
#     centroid-cell edge at periodic faces, so it is not asserted bit-exact here).
#  D) GPU parity (GPU builds): the GPU deposit classifies kept/released identically
#     (equal released counts) and matches the CPU deposit within the usual tolerances.
#
# Usage: tests/ps_halo_release_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

N="${N:-32}"; GRID="${GRID:-64}"; BOX="${BOX:-100.0}"; D="${D:-2}"
BIN="./PS-DTFE"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
SNAP_PAN="${TMP}/phr_input_pancake.hdf5"
SNAP_CRW="${TMP}/phr_input_crossed.hdf5"
NPROC="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

echo "============================================================"
echo " PS-DTFE --ps-halo-release check   N=${N}^3  grid=${GRID}^3  D=${D}"
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
if ! grep -q -- '--ps-halo-release' <<<"${FULL_HELP}"; then
    echo "SKIP: this PS-DTFE has no --ps-halo-release support."
    exit 0
fi

echo ">> generating test snapshots ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_PAN}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_CRW}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 --crossed-waves >/dev/null

run() {  # $1 = threads, $2 = output root, $3 = input snapshot, rest = extra flags
    # out comes right after the input: multitoken options (--partition X Y Z) must not be
    # able to swallow the trailing positional
    local threads="$1" out="$2" snap="$3"; shift 3
    rm -f "${out}".*
    local log="${out}.log"
    set +e
    OMP_NUM_THREADS="${threads}" "${BIN}" "${snap}" "${out}" --grid "${GRID}" \
        --field density velocity density_a --input 105 --MpcUnit 1 --verbose 2 "$@" > "$log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "   ERROR: PS-DTFE exited with code $rc -- last 25 lines of $log:"
        tail -n 25 "$log" | sed 's/^/      | /'
        exit 1
    fi
}

echo ">> run 1: pancake, base deposit"
run 1 "${TMP}/phr_pan_base" "${SNAP_PAN}" --periodic
echo ">> run 2: pancake, --ps-halo-release ${D}"
run 1 "${TMP}/phr_pan_rel" "${SNAP_PAN}" --periodic --ps-halo-release "${D}"
echo ">> run 3: crossed-waves, base deposit"
run 1 "${TMP}/phr_crw_base" "${SNAP_CRW}" --periodic
echo ">> run 4: crossed-waves, --ps-halo-release ${D}"
run 1 "${TMP}/phr_crw_rel" "${SNAP_CRW}" --periodic --ps-halo-release "${D}"
echo ">> run 5: pancake, released, --partition 2 2 2, 1 thread"
run 1 "${TMP}/phr_pan_p1" "${SNAP_PAN}" --periodic --ps-halo-release "${D}" --partition 2 2 2
echo ">> run 6: pancake, released, --partition 2 2 2, ${NPROC} threads"
run "${NPROC}" "${TMP}/phr_pan_pN" "${SNAP_PAN}" --periodic --ps-halo-release "${D}" --partition 2 2 2
if [ "${GPU_BUILT}" -eq 1 ]; then
    echo ">> run 7: pancake, released, --ps-gpu (GPU parity)"
    run 1 "${TMP}/phr_pan_gpu" "${SNAP_PAN}" --periodic --ps-halo-release "${D}" --ps-gpu
else
    echo ">> run 7 skipped (CPU-only build: no GPU parity to check)"
fi

released_count() {  # sum of the per-pass 'released N halo-interior' lines in a log
    grep -o 'released [0-9]* halo-interior' "$1" 2>/dev/null | awk '{s+=$2} END {print s+0}'
}
REL_PAN="$(released_count "${TMP}/phr_pan_rel.log")"
REL_CRW="$(released_count "${TMP}/phr_crw_rel.log")"
REL_GPU=0
[ "${GPU_BUILT}" -eq 1 ] && REL_GPU="$(released_count "${TMP}/phr_pan_gpu.log")"

echo ">> checking the numbers ..."
"${PY}" - "${TMP}" "${GRID}" "${GPU_BUILT}" "${REL_PAN}" "${REL_CRW}" "${REL_GPU}" <<'PY'
import sys
import numpy as np

tmp, grid, gpu, rel_pan, rel_crw, rel_gpu = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                                             int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
ncell = grid ** 3
fails = []

def check(name, ok, detail):
    print(f"   {'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)

def load(root, ext, ncomp=1):
    d = np.fromfile(root + ext, dtype=np.float32)
    assert d.size == ncell * ncomp, (root + ext, d.size)
    return d.reshape((grid, grid, grid) if ncomp == 1 else (grid, grid, grid, ncomp))

for tag, rel in (("pan", rel_pan), ("crw", rel_crw)):
    base, flag = f"{tmp}/phr_{tag}_base", f"{tmp}/phr_{tag}_rel"
    # ---------- (A) releases happen; mass conservation exact ----------
    check(f"A [{tag}] tetrahedra released", rel > 0, f"{rel} released (both passes)")
    db, df = load(base, ".den"), load(flag, ".den")
    ab, af = load(base, ".a_den"), load(flag, ".a_den")
    check(f"A [{tag}] .den mean unchanged", abs(float(df.mean()) - float(db.mean())) < 1e-6,
          f"base {db.mean():.8f} vs released {df.mean():.8f}")
    check(f"A [{tag}] .a_den mean unchanged", abs(float(af.mean()) - float(ab.mean())) < 1e-6,
          f"base {ab.mean():.8f} vs released {af.mean():.8f}")
    # 1e-3 absolute: the crossed-waves BASE deposit already sits ~5e-4 below 1 (mass of
    # dropped degenerate cells; flag-independent). The exact-conservation statement for the
    # RELEASE is the mean-unchanged check above (1e-6 vs base).
    check(f"A [{tag}] mass conserved", abs(float(df.mean()) - 1.0) < 1e-3,
          f"mean rho/rho_bar = {df.mean():.6f}")
    changed = int((af != ab).sum())
    check(f"A [{tag}] deposit actually changed", changed > 0,
          f"{changed} .a_den cells differ (release must not be a no-op at D used here)")
    # ---------- (B) single-stream cells are bit-identical ----------
    sb, sf = load(base, ".streams"), load(flag, ".streams")
    vb, vf = load(base, ".vel", 3), load(flag, ".vel", 3)
    m = sb == 1
    check(f"B [{tag}] single-stream .den identical", np.array_equal(db[m], df[m]),
          f"{int(m.sum())} cells with streams==1")
    check(f"B [{tag}] single-stream .vel identical", np.array_equal(vb[m], vf[m]),
          "released tets must live elsewhere")
    check(f"B [{tag}] single-stream .streams identical", np.array_equal(sb[m], sf[m]),
          "no released centroid may land in a single-stream cell")

# ---------- (C) partition/thread invariance with the flag ----------
s1, sN = load(f"{tmp}/phr_pan_p1", ".streams"), load(f"{tmp}/phr_pan_pN", ".streams")
check("C partitioned streams thread-invariant", np.array_equal(s1, sN),
      "possible race in the released-deposit path")
for ext, nc in ((".den", 1), (".a_den", 1), (".vel", 3)):
    f1, fN = load(f"{tmp}/phr_pan_p1", ext, nc), load(f"{tmp}/phr_pan_pN", ext, nc)
    scale = float(np.abs(f1).max()) + 1e-30
    dmax = float(np.abs(f1 - fN).max()) / scale
    check(f"C partitioned{ext} thread-invariant", dmax < 1e-3, f"max rel = {dmax:.3e}")

# ---------- (D) GPU parity ----------
if gpu:
    check("D GPU released count == CPU", rel_gpu == rel_pan,
          f"gpu {rel_gpu} vs cpu {rel_pan} (classification must be identical)")
    dg = load(f"{tmp}/phr_pan_gpu", ".den")
    sg = load(f"{tmp}/phr_pan_gpu", ".streams")
    df = load(f"{tmp}/phr_pan_rel", ".den")
    sf = load(f"{tmp}/phr_pan_rel", ".streams")
    eq = float((sf == sg).mean())
    mrel = float(np.abs(df - dg).mean() / (np.abs(df).max() + 1e-30))
    check("D GPU streams match CPU", eq > 0.999, f"{eq*100:.4f}% equal")
    check("D GPU density matches CPU", mrel < 1e-4, f"mean rel = {mrel:.3e}")
else:
    print("   SKIP D GPU parity (CPU-only build)")

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (--ps-halo-release conserves mass exactly, leaves single-stream cells"
      " bit-identical, and stays partition/thread invariant)")
PY

echo "============================================================"
echo " halo-release check PASSED"
echo "============================================================"
