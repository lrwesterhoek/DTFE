#!/usr/bin/env bash
# PS-DTFE --ps-volume-weighted correctness check.
#
# The flag switches the VELOCITY-family moments (velocity, gradient and its derived
# divergence/shear/vorticity) from mass-weighted means to Eulerian-VOLUME-weighted means --
# the standard-DTFE '_a' convention that linear theory's -aHf*delta refers to. Its help text
# makes four exactly-testable per-field claims, asserted here on the multi-stream pancake
# (in single-stream cells the two weightings coincide trivially; the multi-stream slab is
# where they must differ):
#  A) .den and .streams are BIT-IDENTICAL to a default run: density and the stream count
#     always keep the mass deposit.
#  B) .velDisp is BIT-IDENTICAL too: sigma_ij is an f-weighted CBE moment BY DEFINITION, so
#     the dispersion deliberately stays mass-weighted (it carries its own disp_weight /
#     disp_velocity mean+normalizer under the flag -- this equality is what proves that
#     plumbing reproduces the mass-weighted arithmetic exactly).
#  C) .vel / .velDiv / .velShear DO differ, and they differ ONLY in multi-stream cells.
#  D) '--ps-volume-weighted --ps-linear-deposit' is REJECTED (opposite sample weightings).
#  E) CPU vs GPU parity with the flag (GPU builds only).
#
# Usage: tests/ps_volume_weighted_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"
BIN="./PS-DTFE"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
SNAP_PAN="${TMP}/pvw_input_pancake.hdf5"

echo "============================================================"
echo " PS-DTFE --ps-volume-weighted check   N=${N}^3  grid=${GRID}^3"
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
if ! grep -q -- '--ps-volume-weighted' <<<"${FULL_HELP}"; then
    echo "SKIP: this PS-DTFE has no --ps-volume-weighted support."
    exit 0
fi

echo ">> generating test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_PAN}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null

run() {  # $1 = output root, rest = extra flags
    local out="$1"; shift
    rm -f "${out}".*
    local log="${out}.log"
    set +e
    "${BIN}" "${SNAP_PAN}" "${out}" --grid "${GRID}" \
        --field density velocity dispersion divergence shear --input 105 --MpcUnit 1 \
        --verbose 1 --avg-subsamples 1 --periodic "$@" > "$log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "   ERROR: PS-DTFE exited with code $rc -- last 25 lines of $log:"
        tail -n 25 "$log" | sed 's/^/      | /'
        exit 1
    fi
}

echo ">> run 1: default (mass-weighted) deposit"
run "${TMP}/pvw_mw"
echo ">> run 2: --ps-volume-weighted"
run "${TMP}/pvw_vw" --ps-volume-weighted
if [ "${GPU_BUILT}" -eq 1 ]; then
    echo ">> run 3: --ps-volume-weighted --ps-gpu (GPU parity)"
    run "${TMP}/pvw_vwg" --ps-volume-weighted --ps-gpu
else
    echo ">> run 3 skipped (CPU-only build: no GPU parity to check)"
fi

echo ">> D: the linear-deposit combination must be rejected"
set +e
"${BIN}" "${SNAP_PAN}" "${TMP}/pvw_rej" --grid "${GRID}" --field density --input 105 \
    --MpcUnit 1 --periodic --ps-volume-weighted --ps-linear-deposit > "${TMP}/pvw_rej.log" 2>&1
RC_REJ=$?
set -e

echo ">> checking the numbers ..."
"${PY}" - "${TMP}" "${GRID}" "${GPU_BUILT}" "${RC_REJ}" <<'PY'
import sys

import numpy as np

tmp, grid, gpu, rc_rej = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ncell = grid ** 3
fails = []

def check(name, ok, detail):
    print(f"   {'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)

def load(root, ext, ncomp=1):
    d = np.fromfile(root + ext, dtype=np.float32)
    assert d.size == ncell * ncomp, (root + ext, d.size)
    return d

streams = load(f"{tmp}/pvw_mw", ".streams")
multi = streams > 1.001                      # float contract: never compare == 1 exactly
check("pancake actually multi-streams", 0.05 < float(multi.mean()) < 0.5,
      f"{100*multi.mean():.1f}% multi-stream cells")

# ---------- (A)+(B) the bit-identical per-field contract ----------
for ext, ncomp, why in ((".den", 1, "density always keeps the mass deposit"),
                        (".streams", 1, "stream count is deposit-independent"),
                        (".velDisp", 1, "dispersion stays MASS-weighted by definition")):
    a = load(f"{tmp}/pvw_mw", ext, ncomp)
    b = load(f"{tmp}/pvw_vw", ext, ncomp)
    check(f"A/B {ext} bit-identical", np.array_equal(a, b), why)

# ---------- (C) the velocity family must differ, and only where streams overlap ----------
multi3 = np.repeat(multi, 3)
for ext, ncomp in ((".vel", 3), (".velDiv", 1), (".velShear", 5)):
    a = load(f"{tmp}/pvw_mw", ext, ncomp).astype(np.float64)
    b = load(f"{tmp}/pvw_vw", ext, ncomp).astype(np.float64)
    ndiff = int((a != b).sum())
    check(f"C {ext} differs", ndiff > 0, f"{ndiff} components differ")
    if ext == ".vel":
        # single-stream cells hold one sample, so the weighting is mathematically irrelevant --
        # but the stored value is (v*w)/w and the float ROUNDING of that round-trip depends on
        # w, so the two runs differ by ~1 ulp there. Assert rounding-level agreement, and that
        # the multi-stream difference is the genuinely physical (orders larger) one.
        single_mask = ~np.repeat(multi, ncomp)
        scale = float(np.abs(a).max()) + 1e-30
        rel_single = float(np.abs(a - b)[single_mask].max()) / scale
        rel_multi  = float(np.abs(a - b)[~single_mask].max()) / scale
        check("C .vel single-stream cells agree to float rounding", rel_single < 1e-5,
              f"max rel = {rel_single:.3e} ((v*w)/w ulp noise only)")
        check("C .vel multi-stream difference is the physical one", rel_multi > 100 * max(rel_single, 1e-12),
              f"multi {rel_multi:.3e} vs single {rel_single:.3e}")

# ---------- (D) rejection of the linear-deposit combination ----------
check("D --ps-linear-deposit combination rejected", rc_rej != 0, f"exit code {rc_rej}")

# ---------- (E) GPU parity with the flag ----------
if gpu:
    for ext, ncomp, tol in ((".den", 1, 1e-4), (".vel", 3, 1e-4),
                            (".velDisp", 1, 1e-4), (".velShear", 5, 1e-3)):
        c = load(f"{tmp}/pvw_vw", ext, ncomp).astype(np.float64)
        g = load(f"{tmp}/pvw_vwg", ext, ncomp).astype(np.float64)
        mrel = float(np.abs(c - g).mean() / (np.abs(c).max() + 1e-30))
        check(f"E GPU{ext} matches CPU", mrel < tol, f"mean rel = {mrel:.3e}")
    sc = load(f"{tmp}/pvw_vw", ".streams")
    sg = load(f"{tmp}/pvw_vwg", ".streams")
    eq = float((sc == sg).mean())
    check("E GPU streams match CPU", eq > 0.999, f"{eq*100:.4f}% equal")
else:
    print("   SKIP E GPU parity (CPU-only build)")

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (--ps-volume-weighted changes ONLY the velocity family, keeps"
      " density/streams/dispersion bit-identical, and rejects the linear-deposit combination)")
PY

echo "============================================================"
echo " volume-weighted check PASSED"
echo "============================================================"
