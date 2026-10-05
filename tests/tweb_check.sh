#!/usr/bin/env bash
# T-web (tidal-tensor classification, '--field tweb') correctness check.
#
# The T-web solves Poisson's equation for the density contrast on the grid by FFT and classifies each
# cell by the eigenvalues of the tidal tensor T_ij = d_i d_j phi. Two exact properties are tested:
#  A) the TRACE: T_xx + T_yy + T_zz = laplacian(phi) = delta for ANY density field, so the three
#     eigenvalues of every cell sum to its delta (the k = 0 mode is zero on both sides). Both binaries.
#  B) a field that depends on x ALONE has T_xx = delta and every other component zero, so each cell's
#     eigenvalues are {delta, 0, 0}. The phase-space density of a 1D pancake on an UNJITTERED lattice is
#     such a field: every tetrahedron in a slab between two lattice planes has the same density, and
#     the slab's faces map to planes x = const. The EXACT deposit (--ps-exact-deposit) integrates that
#     field over the cells, so each x-plane of the grid is uniform too (the sampled deposit spreads a
#     tetrahedron's mass over the samples it happens to hold, which is not); the check verifies it.
#  C) the classification is the number of eigenvalues above --lambda_th (cells within rounding of the
#     threshold are left out).
# Nothing else tested the T-web before 2026-10-02 (it was certified once against a python reference).
# The 2D builds compute it too since 2026-10-03 (two eigenvalues, {delta, 0} for the 1D field): the same
# three checks run on a 2D pancake with PS-DTFE-2d and DTFE-2d when they are built (single precision), and
#  D) on 2D crossed waves (a large xy cross term) the eigenvalues equal an independent numpy FFT solve of
#     the same density grid (full axis: Nyquist negative; half axis: positive).
#
# Usage: tests/tweb_check.sh [--no-build]
#   GRID=1300 STD=0 STRICT=0 PS_ARGS="--avg-subsamples 1" EXTRA="--scratch-dir /path" TWEB_TMP=/path \
#       tests/tweb_check.sh --no-build                 -- the >2^31-cell run: the exact deposit would take
#   an hour there, and the sampled one is not exactly 1D, so STRICT=0 reports the premise and (B) as
#   information; (A) and (C) stay exact. The checks stream the grids plane by plane, at any size.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"; TH="${TH:-0.3}"
N2="${N2:-48}"; GRID2="${GRID2:-96}"     # the 2D pancake: particles and cells per side
STD="${STD:-1}"                          # 0: skip the standard-DTFE trace check
STRICT="${STRICT:-1}"                    # 0: the 1D premise and (B) are reported, not required
PS_ARGS="${PS_ARGS:---ps-exact-deposit}" # the PS-DTFE run's deposit
EXTRA="${EXTRA:-}"                       # extra flags for both runs (e.g. --scratch-dir <dir>)
precision_require "${PS_BIN}" "${DTFE_BIN}"
TMP="${TWEB_TMP:-${SCRIPT_DIR}/tmp}"; mkdir -p "${TMP}"
SNAP="${TMP}/tweb_pancake.hdf5"
SNAP2="${TMP}/tweb_pancake_2d.hdf5"

echo "============================================================"
echo " T-web check   N=${N}^3  grid=${GRID}^3  lambda_th=${TH}"
echo "============================================================"

if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building PS-DTFE and DTFE ..."
    make PS-DTFE $(cat o_ps/.build_mode 2>/dev/null || true) >/dev/null
    make DTFE $(cat o/.build_mode 2>/dev/null || true) >/dev/null
fi

# single-stream (A*k = 0.8, max delta = 4): at a caustic a slab can be thinner than float rounding, and
# a single-precision build then keeps some of its flat tetrahedra and drops others -- a plane is no
# longer uniform there, which is the deposit's float limit, not the T-web's
echo ">> generating an unjittered 1D pancake (single-stream) ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 0.8 --jitter-frac 0 >/dev/null

run() {  # $1 = binary, $2 = output root, rest = this binary's own flags (RUN_SNAP / RUN_GRID: the 2D run)
    local bin="$1" out="$2"; shift 2
    rm -f "${out}".* "${out}"_*
    set +e
    # shellcheck disable=SC2086
    "${bin}" "${RUN_SNAP:-${SNAP}}" "${out}" --grid "${RUN_GRID:-${GRID}}" --periodic --input 105 --MpcUnit 1 \
        --field density tweb --lambda_th "${TH}" --verbose 1 "$@" ${EXTRA} > "${out}.log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "   ERROR: ${bin} exited with code $rc -- last 20 lines of ${out}.log:"
        tail -n 20 "${out}.log" | sed 's/^/      | /'
        exit 1
    fi
}

echo ">> PS-DTFE ..."
# shellcheck disable=SC2086
run "${PS_BIN}" "${TMP}/tweb_ps" ${PS_ARGS}
if [ "${STD}" = "1" ]; then
    echo ">> standard DTFE ..."
    run "${DTFE_BIN}" "${TMP}/tweb_std"
fi

# the 2D builds (single precision only: the 2D set has no double pair in the battery)
DO2D=0
if ! precision_double && [ -x "${PS_BIN}-2d" ] && [ -x "${DTFE_BIN}-2d" ]; then
    DO2D=1
    echo ">> generating an unjittered 2D pancake and running PS-DTFE-2d / DTFE-2d ..."
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP2}" --n "${N2}" --box "${BOX}" \
        --amplitude-factor 0.8 --jitter-frac 0 --dim 2 >/dev/null
    # shellcheck disable=SC2086
    RUN_SNAP="${SNAP2}" RUN_GRID="${GRID2}" run "${PS_BIN}-2d" "${TMP}/tweb2_ps" ${PS_ARGS}
    if [ "${STD}" = "1" ]; then
        RUN_SNAP="${SNAP2}" RUN_GRID="${GRID2}" run "${DTFE_BIN}-2d" "${TMP}/tweb2_std"
    fi
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/tweb_crossed_2d.hdf5" --n "${N2}" --box "${BOX}" \
        --amplitude-factor 0.8 --crossed-waves --dim 2 >/dev/null
    RUN_SNAP="${TMP}/tweb_crossed_2d.hdf5" RUN_GRID=64 run "${PS_BIN}-2d" "${TMP}/tweb2_cw" --ps-exact-deposit
else
    if precision_double; then echo "   SKIP  2D (double mode: the 2D set has no double pair in the battery)"
    else echo "   SKIP  2D (no ${PS_BIN}-2d / ${DTFE_BIN}-2d: make DTFE PS-DTFE DIM=2)"; fi
fi

echo ">> checking the numbers ..."
"${PY}" - "${TMP}" "${GRID}" "${TH}" "${STD}" "${STRICT}" "${DO2D}" "${GRID2}" <<'PYEOF'
import os, sys
import numpy as np
tmp, g3, th, std, strict = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1", sys.argv[5] == "1"
do2d, g2 = sys.argv[6] == "1", int(sys.argv[7])
real = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))
fails = []

def check(ok, msg):
    print(f"   {'OK  ' if ok else 'FAIL'} {msg}")
    if not ok:
        fails.append(msg)

def grids(pre, g, dim):
    den = np.memmap(pre + ".den", dtype=real, mode="r")
    eig = np.memmap(pre + ".twebEig", dtype=real, mode="r")
    tw = np.memmap(pre + ".tweb", dtype=real, mode="r")
    if den.size != g**dim or eig.size != dim * g**dim or tw.size != g**dim:
        sys.exit(f"FAIL: {pre}: sizes {den.size}/{eig.size}/{tw.size} for a {g}^{dim} grid")
    plane = g ** (dim - 1)
    mean = 0.
    for x in range(g):
        mean += float(den[x * plane:(x + 1) * plane].astype(np.float64).sum())
    return den, eig, tw, mean / g**dim

def scan(pre, oneD, g, dim):
    """per x-plane (a row in 2D): the trace error, the {delta,0,..} error, the plane's spread, class mismatches"""
    den, eig, tw, mean = grids(pre, g, dim)
    plane = g ** (dim - 1)
    dmax = tr = sep = spread = 0.
    wrong = tested = 0
    ss_err = ss_noise = 0.              # sums of squares: the deviation from {delta,0,..}, delta's in-plane noise
    for x in range(g):
        d = den[x * plane:(x + 1) * plane].astype(np.float64) / mean - 1.
        e = eig[dim * x * plane:dim * (x + 1) * plane].astype(np.float64).reshape(-1, dim)
        c = tw[x * plane:(x + 1) * plane].astype(np.float64)
        dmax = max(dmax, float(np.abs(d).max()))
        tr = max(tr, float(np.abs(e.sum(axis=1) - d).max()))
        if oneD:
            spread = max(spread, float(d.max() - d.min()))
            want = np.sort(np.stack([d] + [np.zeros_like(d)] * (dim - 1), axis=1), axis=1)
            err = np.sort(e, axis=1) - want
            sep = max(sep, float(np.abs(err).max()))
            ss_err += float((err * err).sum())
            ss_noise += float(((d - d.mean()) ** 2).sum())
        clear = np.all(np.abs(e - th) > 1e-4 * (1. + abs(th)), axis=1)
        wrong += int(np.count_nonzero((c != (e > th).sum(axis=1)) & clear))
        tested += int(np.count_nonzero(clear))
    return dmax, tr, sep, spread, wrong, tested, np.sqrt(ss_err / g**dim), np.sqrt(ss_noise / g**dim)

def verify(g, dim, ps, stdpre):
  tag = "" if dim == 3 else "2D "
  zeros = ", 0" * (dim - 1)
  dmax, tr, sep, spread, wrong, tested, rms_err, rms_noise = scan(ps, True, g, dim)
  tol = 2e-5 * max(1., dmax)
  check(dmax > 2., f"{tag}the pancake has structure: max|delta| = {dmax:.3g}")
  if strict:
    check(spread <= tol, f"{tag}premise: every x-{'plane' if dim == 3 else 'row'} of the PS density is uniform (spread {spread:.3g} <= {tol:.2g})")
  check(tr <= tol, f"{tag}A PS-DTFE: eigenvalues sum to delta in every cell (max error {tr:.3g} <= {tol:.2g})")
  if strict:
    check(sep <= tol, f"{tag}B PS-DTFE: eigenvalues are {{delta{zeros}}} (max error {sep:.3g} <= {tol:.2g})")
  else:
    # The off-axis parts of the tidal tensor are Riesz transforms of delta's in-plane noise (|k_i k_j / k^2|
    # <= 1, so by Parseval their rms cannot exceed the noise's): a correct transform keeps the rms
    # deviation from {delta, 0, 0} at the noise's level, a broken one makes it of the order of delta itself.
    check(rms_err <= 2. * rms_noise + 1e-5,
          f"{tag}B' PS-DTFE: rms deviation from {{delta{zeros}}} {rms_err:.3g} <= 2 x the rms in-plane noise of delta "
          f"{rms_noise:.3g} (the sampled deposit is not exactly 1D; max deviation {sep:.3g}, max plane spread {spread:.3g})")
  check(wrong == 0 and tested > 0.9 * g**dim,
        f"{tag}C PS-DTFE: class = number of eigenvalues > lambda_th ({wrong} wrong of {tested} decidable cells)")
  if std:
    dmax, tr, _, _, wrong, tested, _, _ = scan(stdpre, False, g, dim)
    tol = 2e-5 * max(1., dmax)
    check(tr <= tol, f"{tag}A standard DTFE: eigenvalues sum to delta in every cell (max error {tr:.3g} <= {tol:.2g})")
    check(wrong == 0 and tested > 0.9 * g**dim,
          f"{tag}C standard DTFE: class = number of eigenvalues > lambda_th ({wrong} wrong of {tested} decidable cells)")

verify(g3, 3, f"{tmp}/tweb_ps", f"{tmp}/tweb_std")
if do2d:
  verify(g2, 2, f"{tmp}/tweb2_ps", f"{tmp}/tweb2_std")
  g = 64                                              # D: crossed waves against numpy
  den = np.fromfile(f"{tmp}/tweb2_cw.den", real).astype(np.float64).reshape(g, g)
  eig = np.fromfile(f"{tmp}/tweb2_cw.twebEig", real).astype(np.float64).reshape(g, g, 2)
  dk = np.fft.rfft2(den / den.mean() - 1.)
  kx, ky = 2 * np.pi * np.fft.fftfreq(g)[:, None], 2 * np.pi * np.fft.rfftfreq(g)[None, :]
  k2 = kx ** 2 + ky ** 2
  k2[0, 0] = 1.
  t = {}
  for name, f in (("xx", kx * kx), ("xy", kx * ky), ("yy", ky * ky)):
    tk = f / k2 * dk
    tk[0, 0] = 0.
    t[name] = np.fft.irfft2(tk, s=(g, g))
  mid, disc = (t["xx"] + t["yy"]) / 2., np.hypot((t["xx"] - t["yy"]) / 2., t["xy"])
  err = float(np.abs(eig - np.stack([mid + disc, mid - disc], -1)).max())
  big = float(np.abs(t["xy"]).max())
  check(big > 0.5 and err <= 2e-5 * max(1., float(np.abs(den / den.mean() - 1.).max())),
        f"2D D PS-DTFE: crossed waves (max |T_xy| {big:.3g}): eigenvalues equal a numpy FFT solve (max error {err:.3g})")
print("------------------------------------------------------------")
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (T-web: the tidal tensor's trace is delta in every cell, a 1D field gives exactly "
      "{delta, 0, 0}, and the class counts the eigenvalues above lambda_th" + ("; 2D too" if do2d else "") + ")")
PYEOF
