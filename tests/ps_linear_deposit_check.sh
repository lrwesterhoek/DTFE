#!/usr/bin/env bash
# PS-DTFE --ps-linear-deposit correctness check.
#
# The linear deposit weights each tetrahedron's interior sub-samples by the DTFE-interpolated
# linear density profile, RENORMALIZED per tetrahedron so the deposited total still equals
# the tetrahedron's mass exactly. Verified here:
#  A) UNIFORM box (amp=0): the vertex densities are constant, so the linear weights are
#     constant and the linear deposit must reproduce the uniform deposit to float rounding.
#  B) PANCAKE (multi-stream): mass conservation is unchanged (grid mean of rho/rho_bar = 1
#     to float rounding, at nSub=1 AND nSub=3); the uniform and linear grids differ smoothly
#     (identical .streams, high correlation, bounded max ratio -- no new caustic spikes).
#  C) GPU parity (GPU builds only): the GPU linear deposit matches the CPU linear deposit
#     within the usual float32 atomic tolerances.
#  D) [gated on --ps-exact-deposit support] the r3d exact conservative deposit: mass
#     conservation at (or below) the sampled deposit's float floor, '.den' == '.a_den'
#     bit-exact (the exact deposit is nSub-independent), the raw integer '.tetTouch'
#     count bit-exact between same-split partitioned runs at 1 and N threads, '.streams'
#     (the volume-weighted multiplicity -- a float, hence checked to a tolerance, not
#     bit-exactly) hitting the pancake's ANALYTIC values of 1 and 3, the sampled '.a_den'
#     at nSub = 1/3/5 approaching the exact grid MONOTONICALLY (L1), and the
#     '--ps-linear-deposit' composition (exact order-1 linear-density shares) conserving
#     mass and staying strongly correlated with the exact uniform deposit.
#
# Usage: tests/ps_linear_deposit_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"
BIN="./PS-DTFE"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
SNAP_UNI="${TMP}/pld_input_uniform.hdf5"
SNAP_PAN="${TMP}/pld_input_pancake.hdf5"

echo "============================================================"
echo " PS-DTFE --ps-linear-deposit check   N=${N}^3  grid=${GRID}^3"
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
HAVE_EXACT=0
grep -q -- '--ps-exact-deposit' <<<"${FULL_HELP}" && HAVE_EXACT=1

echo ">> generating test snapshots ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_UNI}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 0 --jitter-frac 0.3 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_PAN}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null

run() {  # $1 = output root, $2 = input snapshot, rest = extra flags
    # out right after the input: multitoken options (--partition X Y Z) must not be able
    # to swallow the trailing positional
    local out="$1" snap="$2"; shift 2
    rm -f "${out}".*
    local log="${out}.log"
    set +e
    "${BIN}" "${snap}" "${out}" --grid "${GRID}" --field density velocity density_a \
        --input 105 --MpcUnit 1 --verbose 1 "$@" > "$log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "   ERROR: PS-DTFE exited with code $rc -- last 25 lines of $log:"
        tail -n 25 "$log" | sed 's/^/      | /'
        exit 1
    fi
}

echo ">> run 1: uniform box, uniform deposit"
run "${TMP}/pld_uni_u" "${SNAP_UNI}" --periodic
echo ">> run 2: uniform box, --ps-linear-deposit"
run "${TMP}/pld_uni_l" "${SNAP_UNI}" --periodic --ps-linear-deposit
echo ">> run 3: pancake, uniform deposit"
run "${TMP}/pld_pan_u" "${SNAP_PAN}" --periodic
echo ">> run 4: pancake, --ps-linear-deposit"
run "${TMP}/pld_pan_l" "${SNAP_PAN}" --periodic --ps-linear-deposit
if [ "${GPU_BUILT}" -eq 1 ]; then
    echo ">> run 5: pancake, --ps-linear-deposit --ps-gpu (GPU parity)"
    run "${TMP}/pld_pan_lg" "${SNAP_PAN}" --periodic --ps-linear-deposit --ps-gpu
else
    echo ">> run 5 skipped (CPU-only build: no GPU parity to check)"
fi

if [ "${HAVE_EXACT}" -eq 1 ]; then
    echo ">> run 6: pancake, --ps-exact-deposit"
    run "${TMP}/pld_pan_ex" "${SNAP_PAN}" --periodic --ps-exact-deposit
    echo ">> run 7: pancake, --ps-exact-deposit, --partition 2 2 2, 1 thread"
    OMP_NUM_THREADS=1 run "${TMP}/pld_pan_ex_p1" "${SNAP_PAN}" --periodic --ps-exact-deposit --partition 2 2 2
    echo ">> run 8: pancake, --ps-exact-deposit, --partition 2 2 2, all threads"
    run "${TMP}/pld_pan_ex_pN" "${SNAP_PAN}" --periodic --ps-exact-deposit --partition 2 2 2
    echo ">> run 9: pancake, --ps-exact-deposit --ps-linear-deposit (composition)"
    run "${TMP}/pld_pan_ex_l" "${SNAP_PAN}" --periodic --ps-exact-deposit --ps-linear-deposit
    echo ">> runs 10-11: pancake, sampled '.a_den' at nSub 1/5 (3 is run 3) for the convergence ladder"
    run "${TMP}/pld_pan_ns1" "${SNAP_PAN}" --periodic --avg-subsamples 1
    run "${TMP}/pld_pan_ns5" "${SNAP_PAN}" --periodic --avg-subsamples 5
    if [ "${GPU_BUILT}" -eq 1 ]; then
        echo ">> runs 12-13: exact deposit + dispersion, CPU vs --ps-gpu (float32 r3d port parity)"
        # dedicated pair (with the dispersion field run() does not request): the GPU exact
        # deposit is a hand-written float32 r3d port and had NO parity coverage in the suite
        for tag in exd exg; do
            GPUFLAG=""; [ "${tag}" = "exg" ] && GPUFLAG="--ps-gpu"
            rm -f "${TMP}/pld_pan_${tag}".*
            set +e
            "${BIN}" "${SNAP_PAN}" "${TMP}/pld_pan_${tag}" --grid "${GRID}" \
                --field density velocity dispersion --input 105 --MpcUnit 1 --verbose 1 \
                --periodic --ps-exact-deposit ${GPUFLAG} > "${TMP}/pld_pan_${tag}.log" 2>&1
            rc=$?
            set -e
            if [ "${rc}" -ne 0 ]; then
                echo "   ERROR: PS-DTFE exited with code ${rc} -- last 25 lines:"
                tail -n 25 "${TMP}/pld_pan_${tag}.log" | sed 's/^/      | /'
                exit 1
            fi
        done
    else
        echo ">> runs 12-13 skipped (CPU-only build: no exact-deposit GPU parity to check)"
    fi
else
    echo ">> runs 6-13 skipped (binary lacks --ps-exact-deposit)"
fi

echo ">> checking the numbers ..."
"${PY}" - "${TMP}" "${GRID}" "${GPU_BUILT}" "${HAVE_EXACT}" <<'PY'
import sys
import numpy as np

tmp, grid, gpu, have_exact = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ncell = grid ** 3
fails = []

def check(name, ok, detail):
    print(f"   {'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)

def load(root, ext, ncomp=1):
    d = np.fromfile(root + ext, dtype=np.float32).astype(np.float64)
    assert d.size == ncell * ncomp, (root + ext, d.size)
    return d

# ---------- (A) uniform box: linear == uniform to float rounding ----------
for ext in (".den", ".a_den", ".vel"):
    nc = 3 if ext == ".vel" else 1
    u = load(f"{tmp}/pld_uni_u", ext, nc)
    l = load(f"{tmp}/pld_uni_l", ext, nc)
    rel = np.abs(u - l).max() / (np.abs(u).max() + 1e-30)
    check(f"A uniform{ext} linear==uniform", rel < 1e-5, f"max rel = {rel:.3e}")

# ---------- (B) pancake: mass conservation + smooth A/B differences ----------
for ext, tag in ((".den", "nSub=1"), (".a_den", "nSub=3")):
    dl = load(f"{tmp}/pld_pan_l", ext)
    check(f"B mass conserved [{tag}]", abs(dl.mean() - 1.0) < 1e-4,
          f"mean rho/rho_bar = {dl.mean():.6f}")
du = load(f"{tmp}/pld_pan_u", ".den")
dl = load(f"{tmp}/pld_pan_l", ".den")
su = np.fromfile(f"{tmp}/pld_pan_u.streams", dtype=np.float32)
sl = np.fromfile(f"{tmp}/pld_pan_l.streams", dtype=np.float32)
check("B streams identical", np.array_equal(su, sl),
      "the weighting must not change which samples deposit")
ratio = dl.max() / du.max()
check("B no new caustic spikes", 0.2 < ratio < 5.0,
      f"max(linear)/max(uniform) = {ratio:.3f}")
m = (du > 0) & (dl > 0)
corr = np.corrcoef(np.log(du[m]), np.log(dl[m]))[0, 1]
check("B grids differ smoothly", corr > 0.98, f"log-density correlation = {corr:.5f}")
big = np.mean(np.abs(np.log(np.maximum(dl[m], 1e-12) / np.maximum(du[m], 1e-12))) > np.log(2.0))
check("B few cells change >2x", big < 0.02, f"{big*100:.3f}% of covered cells")

# ---------- (C) GPU parity for the linear deposit ----------
if gpu:
    dg = load(f"{tmp}/pld_pan_lg", ".den")
    sg = np.fromfile(f"{tmp}/pld_pan_lg.streams", dtype=np.float32)
    eq = np.mean(sl == sg)
    scale = np.abs(dl).max()
    mrel = np.abs(dl - dg).mean() / scale
    check("C GPU streams match CPU", eq > 0.999, f"{eq*100:.4f}% equal")
    check("C GPU density matches CPU", mrel < 1e-4, f"mean rel = {mrel:.3e}")
else:
    print("   SKIP C GPU parity (CPU-only build)")

# ---------- (D) exact conservative deposit (--ps-exact-deposit, r3d) ----------
if have_exact:
    ex = load(f"{tmp}/pld_pan_ex", ".den")
    exa = load(f"{tmp}/pld_pan_ex", ".a_den")
    du_mean_err = abs(du.mean() - 1.0)          # sampled uniform-deposit floor (run 3)
    check("D mass conserved (exact <= sampled floor)",
          abs(ex.mean() - 1.0) <= du_mean_err + 1e-7,
          f"|mean-1| = {abs(ex.mean()-1.0):.3e} (sampled: {du_mean_err:.3e})")
    check("D .den == .a_den (nSub-independent)",
          np.array_equal(ex, exa), "the exact deposit is the nSub->infinity limit")
    # '.tetTouch': the RAW INTEGER count of tetrahedra with a nonzero cell intersection.
    # This is the field that carries the old integer-'.streams' contract -- quantities.cc
    # sums it linearly across partitions, so it stays bit-exact under any thread order.
    tx = np.fromfile(f"{tmp}/pld_pan_ex.tetTouch", dtype=np.float32).astype(np.float64)
    # 'integer-valued' alone is vacuous -- any int < 2^24 casts exactly, so an all-zero or
    # scrambled grid would pass it. Pin the count to >= 1 as well: the periodic tessellation
    # is space-filling, so EVERY cell is touched by at least one tet (measured min = 2).
    check("D exact tetTouch integer-valued and >= 1",
          bool(np.allclose(tx, np.round(tx), atol=1e-6)) and tx.min() >= 1,
          f"max frac dev = {np.abs(tx - np.round(tx)).max():.2e}, "
          f"touch range = [{tx.min():.0f}, {tx.max():.0f}]")
    t1 = np.fromfile(f"{tmp}/pld_pan_ex_p1.tetTouch", dtype=np.float32)
    tN = np.fromfile(f"{tmp}/pld_pan_ex_pN.tetTouch", dtype=np.float32)
    check("D partitioned tetTouch thread-invariant (bit-exact)", np.array_equal(t1, tN),
          "same split, 1 vs all threads (integer counts merge order-independently)")

    # '.streams' is now the VOLUME-WEIGHTED multiplicity (1/V_cell) sum V_int -- a float, so
    # neither integer-valued nor bit-exact (float shares sum order-dependently). What the new
    # definition buys is that it is the ANALYTIC nSub->infinity limit, so the two properties
    # below are exact statements about the Zel'dovich pancake rather than sampling artefacts.
    sx = np.fromfile(f"{tmp}/pld_pan_ex.streams", dtype=np.float32).astype(np.float64)
    # The pancake folds into exactly 3 streams behind the caustic. The sampled deposit
    # overshoots this badly (a grazing sub-sample counts a full +1; run 3's 'su' reports
    # ~14), while the exact deposit weights each tet by its intersected volume and lands on 3.
    check("D exact streams max == analytic 3", abs(sx.max() - 3.0) < 0.01,
          f"max = {sx.max():.6f} (analytic 3.00; the sampled deposit gives {su.max():.2f})")
    # Outside the caustic the tessellation covers each cell exactly once, so those cells
    # must read exactly 1.0 -- and every cell is covered at least once (space-filling).
    near1 = np.abs(sx - 1.0) < 1e-3
    check("D exact streams single-stream cells == 1.0",
          sx.min() > 1.0 - 1e-3 and near1.mean() > 0.5,
          f"min = {sx.min():.6f}, {near1.mean()*100:.2f}% of cells within 1e-3 of 1.0")
    # ties the two fields together: each touching tet contributes exactly 1 to '.tetTouch' and
    # its volume share V_int/V_cell <= 1 to '.streams', so streams <= tetTouch pointwise. This
    # is what would catch a zeroed/mis-normalized accumulator in EITHER field. (The bound needs
    # the bbox rasterization -- --ps-halo-release's centroid fallback deposits V_tet/V_cell,
    # which may exceed 1 for a tet bigger than a cell. No run here uses it.)
    viol = int(np.sum(sx > tx + 1e-6))
    check("D exact streams <= tetTouch (pointwise)", viol == 0,
          f"{viol} cells violate; min slack = {(tx - sx).min():.6f}")
    # partition split: same tolerance form as the density check just below
    s1 = np.fromfile(f"{tmp}/pld_pan_ex_p1.streams", dtype=np.float32)
    sN = np.fromfile(f"{tmp}/pld_pan_ex_pN.streams", dtype=np.float32)
    smax = np.abs(s1.astype(np.float64) - sN).max() / (np.abs(s1).max() + 1e-30)
    check("D partitioned streams thread-invariant", smax < 1e-5, f"max rel = {smax:.3e}")
    d1 = load(f"{tmp}/pld_pan_ex_p1", ".den")
    dN = load(f"{tmp}/pld_pan_ex_pN", ".den")
    dmax = np.abs(d1 - dN).max() / (np.abs(d1).max() + 1e-30)
    check("D partitioned density thread-invariant", dmax < 1e-5, f"max rel = {dmax:.3e}")
    # sampled '.a_den' must approach the exact grid MONOTONICALLY with nSub
    l1 = {}   # nSub=3 is the default of run 3 (pld_pan_u)
    for ns, root in ((1, "pld_pan_ns1"), (3, "pld_pan_u"), (5, "pld_pan_ns5")):
        l1[ns] = np.abs(load(f"{tmp}/{root}", ".a_den") - exa).mean()
    check("D sampled a_den converges to exact (L1, nSub 1>3>5)",
          l1[1] > l1[3] > l1[5],
          f"L1 = {l1[1]:.4e} (nSub=1) > {l1[3]:.4e} (nSub=3) > {l1[5]:.4e} (nSub=5)")
    exl = load(f"{tmp}/pld_pan_ex_l", ".den")
    check("D exact+linear composition conserves mass",
          abs(exl.mean() - 1.0) <= du_mean_err + 1e-6,
          f"|mean-1| = {abs(exl.mean()-1.0):.3e}")
    mm = (ex > 0) & (exl > 0)
    corr_ex = np.corrcoef(np.log(ex[mm]), np.log(exl[mm]))[0, 1]
    check("D exact+linear differs smoothly from exact", corr_ex > 0.98,
          f"log-density correlation = {corr_ex:.5f}")
    # float32 r3d port parity: the GPU exact deposit (exact_init_tet/exact_clip_plane/
    # exact_reduce2 in metal/ps_deposit.metal + the CUDA mirror) vs the CPU double r3d
    if gpu:
        for ext, nc, tol in ((".den", 1, 1e-4), (".vel", 3, 1e-4), (".velDisp", 1, 1e-4)):
            c = load(f"{tmp}/pld_pan_exd", ext, nc)
            g = load(f"{tmp}/pld_pan_exg", ext, nc)
            mrel = np.abs(c - g).mean() / (np.abs(c).max() + 1e-30)
            check(f"D exact GPU{ext} matches CPU", mrel < tol, f"mean rel = {mrel:.3e}")
        sc = np.fromfile(f"{tmp}/pld_pan_exd.streams", dtype=np.float32).astype(np.float64)
        sg = np.fromfile(f"{tmp}/pld_pan_exg.streams", dtype=np.float32).astype(np.float64)
        srel = np.abs(sc - sg).max() / (np.abs(sc).max() + 1e-30)
        check("D exact GPU streams match CPU", srel < 1e-3,
              f"max rel = {srel:.3e} (float volume-weighted multiplicity)")
    else:
        print("   SKIP D exact GPU parity (CPU-only build)")
else:
    print("   SKIP D exact deposit (binary lacks --ps-exact-deposit)")

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (--ps-linear-deposit conserves mass exactly and only reshapes the"
      " sub-tetrahedron distribution)")
PY

echo "============================================================"
echo " linear-deposit check PASSED"
echo "============================================================"
