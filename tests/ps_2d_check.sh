#!/usr/bin/env bash
# The 2D phase-space build (PS-DTFE-2d, 'make PS-DTFE DIM=2') against analytic 2D Zel'dovich flows, and
# the 2D standard build (DTFE-2d). Synthetic 2D snapshots: two-column HDF5 datasets
# (tests/generate_ps_test_data.py --dim 2).
#   P  the pancake (a 1-D wave in 2D, A*k = 1.8): the stream count in every column equals the
#      analytic number of Zel'dovich streams, for the sampled AND the exact deposit; mass conserved
#      (to 1e-3 under the default Lagrangian-area masses, which drop the few triangles squeezed flat
#      on the caustic, as in 3D; exactly under --ps-vertex-mass); the exact multiplicity averages to
#      the analytic stream count
#   C  single-stream density converges to the analytic profile (L1 error falls by > 40%, < 0.008)
#   V  single-stream velocity = the analytic 100 A sin(k q(x)) of the stream
#   X  crossed waves: point stream counts in {1, 3, 9}, and their fractions are the products of the
#      1-D fractions (the flow is separable)
#   E  the sampled deposit converges to the exact one: away from the caustics the density difference
#      halves from nSub 6 to 12 (first order in the sample spacing); velocity and dispersion agree
#   H  '.hidden_streams' bit 1 (a folded triangle overlaps the cell) marks no cell of the analytic
#      single-stream region away from the caustics, and every analytic multi-stream column
#   F  --ps-caustics: '.caustic' (both orientations overlap the cell) is exactly the parity bits of
#      '.causticClass', set throughout the analytic multi-stream region and nowhere in the single-
#      stream one; the collapse multiplicity is k = 1 inside the pancake and k = 0 outside
#   S  --partition 2 2 == one triangulation (streams identical; density identical to float rounding
#      but in the few caustic cells where a copy's float rounding flips a flat triangle's drop);
#      --ps-vertex-mass, --ps-volume-weighted, --ps-halo-release, --ps-linear-deposit conserve mass;
#      a non-periodic clump conserves mass and covers its interior; the tessellation cache hits and
#      reproduces the run bit for bit
#   D  the standard 2D build runs on the same file: its mean density is the particle density
#   PE point evaluation (--sample-points, 2D since 2026-10-02): at the cell centres the point stream
#      counts equal the same run's nSub = 1 grid counts EXACTLY; single-stream point velocities are
#      the analytic ones; a --partition 2 2 run gives the same points; a text points file the same
#      outputs as the binary one; the standard DTFE-2d evaluates points too; and the interactive
#      server (--serve, 2D since 2026-10-03): dtfelib's Estimator(dim=2) answers byte-identical to
#      --sample-points, a composite (partition 2 2) server equals the same split's point run, and
#      Estimator.from_arrays builds a 2D estimator from (N, 2) arrays
#   W  --ps-window (2D since 2026-10-02): the window run's cells are the full run's, bit for bit with the
#      sampled deposit, to float rounding with the exact one; the stream counts identical
#   CU --ps-caustic-cusps (2D since 2026-10-02) on a pancake modulated along y, psi_x = A sin(kq_x)
#      (1 + eps cos(kq_y)), whose fold lens closes in two analytic cusps: both are flagged, every flag
#      lies on the ridge k q_x = pi where the null-direction derivative of lambda_c vanishes (the
#      indicator marks that ridge within its fold band, as in 3D), the unmodulated pancake -- no cusps
#      -- gets none, and no swallowtail bit is ever set in 2D
#   EA the standard DTFE-2d's --exact-average (2D since 2026-10-02, triangles clipped by ps_clip2d.h):
#      density 1 to float rounding on a perfect lattice (the sampled average is off by percents), mass
#      conserved on a pancake, and the sampled average agrees with it to its own noise
# Usage: tests/ps_2d_check.sh [--no-build]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
if [ "${1:-}" != "--no-build" ]; then
    echo ">> building PS-DTFE-2d and DTFE-2d (make ... DIM=2) ..."
    make PS-DTFE DTFE DIM=2 -j4 >/dev/null || { echo "FAIL: the 2D build"; exit 1; }
fi
# (this suite does not source precision.sh: the 2D pair is its own) a missing pair is a skip, or a failure when
# the stage requires it (DTFE_TEST_REQUIRED=1, ci_suite.sh's 2d stage after building it)
skip_or_fail() { if [ "${DTFE_TEST_REQUIRED:-0}" = "1" ]; then echo "FAIL (required): $*" >&2; exit 1; fi; echo "SKIP: $*"; exit 0; }
for b in ./PS-DTFE-2d ./DTFE-2d; do [ -x "$b" ] || skip_or_fail "$b is not built (make PS-DTFE DTFE DIM=2)"; done
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/ps_2d"
mkdir -p "$TMP"
CACHE="${TMPDIR:-/tmp}/dtfe-2d-cache"; rm -rf "$CACHE"; mkdir -p "$CACHE"
GEN() { "$PY" tests/generate_ps_test_data.py --dim 2 "$@" >/dev/null || { echo "FAIL data gen $*"; exit 1; }; }
RUN() { local out="$1"; shift; "$@" > "$out.log" 2>&1 || { echo "FAIL: $* (see $out.log)"; exit 1; }; }
G=128
COMMON=(--grid $G --periodic --MpcUnit 1 --verbose 2)

echo ">> pancake: sampled (nSub 3), exact, exact + caustics, sampled nSub 6, partitions, flags"
GEN --out "$TMP/pan.hdf5" --n 96 --box 100 --amplitude-factor 1.8 --jitter-frac 0.02 --seed 3
RUN "$TMP/pan_s"   ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pan_s"   "${COMMON[@]}" --field density density_a velocity_a --avg-subsamples 3
RUN "$TMP/pan_s6"  ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pan_s6"  "${COMMON[@]}" --field density_a velocity_a dispersion_a --avg-subsamples 6
RUN "$TMP/pan_s12" ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pan_s12" "${COMMON[@]}" --field density_a --avg-subsamples 12
RUN "$TMP/pan_x"   ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pan_x"   "${COMMON[@]}" --field density_a velocity_a dispersion_a --ps-exact-deposit --ps-caustics
RUN "$TMP/pan_p1"  ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pan_p1"  "${COMMON[@]}" --field density_a --avg-subsamples 2 --partition 1 1 --max-concurrent 1
RUN "$TMP/pan_p2"  ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pan_p2"  "${COMMON[@]}" --field density_a --avg-subsamples 2 --partition 2 2 --max-concurrent 1
RUN "$TMP/pan_vm"  ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pan_vm"  "${COMMON[@]}" --field density_a velocity_a --ps-vertex-mass --ps-volume-weighted
RUN "$TMP/pan_hr"  ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pan_hr"  "${COMMON[@]}" --field density_a --ps-halo-release 2
RUN "$TMP/pan_lin" ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pan_lin" "${COMMON[@]}" --field density_a --ps-linear-deposit
RUN "$TMP/pan_c1"  ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pan_c1"  "${COMMON[@]}" --field density_a --tessellation-cache "$CACHE"
RUN "$TMP/pan_c2"  ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pan_c2"  "${COMMON[@]}" --field density_a --tessellation-cache "$CACHE"
echo ">> convergence series (single-stream density, nSub 3, grid 96)"
for n in 16 24 32 48; do
    GEN --out "$TMP/conv_$n.hdf5" --n $n --box 100 --amplitude-factor 1.8 --jitter-frac 0.02 --seed 42
    RUN "$TMP/conv_$n" ./PS-DTFE-2d "$TMP/conv_$n.hdf5" "$TMP/conv_$n" --grid 96 --periodic --MpcUnit 1 --field density_a --avg-subsamples 3
done
echo ">> crossed waves, a non-periodic clump, the standard 2D build"
GEN --out "$TMP/crw.hdf5" --n 96 --box 100 --amplitude-factor 1.5 --crossed-waves --jitter-frac 0.02 --seed 5
RUN "$TMP/crw" ./PS-DTFE-2d "$TMP/crw.hdf5" "$TMP/crw" "${COMMON[@]}" --field density density_a --avg-subsamples 2
GEN --out "$TMP/clump.hdf5" --n 64 --box 100 --amplitude-factor 0.5 --margin-frac 0.2 --jitter-frac 0.02 --seed 9
RUN "$TMP/clump" ./PS-DTFE-2d "$TMP/clump.hdf5" "$TMP/clump" --grid $G --MpcUnit 1 --verbose 2 --field density --box 0 100 0 100
RUN "$TMP/std" ./DTFE-2d "$TMP/pan.hdf5" "$TMP/std" --grid 64 --periodic --MpcUnit 1 --verbose 2 --field density_a
echo ">> point evaluation at the cell centres (binary and text points, a partition split), --serve refused"
"$PY" - "$TMP" "$G" <<'PTSEOF'
import numpy as np, sys
tmp, g = sys.argv[1], int(sys.argv[2]); L = 100.0
c = (np.arange(g) + 0.5) * L / g
X, Y = np.meshgrid(c, c, indexing="ij")
P = np.stack([X.ravel(), Y.ravel()], axis=1)
P.astype(np.float64).tofile(f"{tmp}/centres.bin")
np.savetxt(f"{tmp}/centres.txt", P[:997], fmt="%.17g")
P[:997].astype(np.float64).tofile(f"{tmp}/centres997.bin")
PTSEOF
RUN "$TMP/pts"    ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pts"    "${COMMON[@]}" --field density velocity --avg-subsamples 1 --sample-points "$TMP/centres.bin"
RUN "$TMP/pts_p2" ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pts_p2" "${COMMON[@]}" --field density --partition 2 2 --max-concurrent 1 --sample-points "$TMP/centres.bin"
RUN "$TMP/pts_tx" ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pts_tx" "${COMMON[@]}" --field density --sample-points "$TMP/centres.txt"
RUN "$TMP/pts_bn" ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/pts_bn" "${COMMON[@]}" --field density --sample-points "$TMP/centres997.bin"
RUN "$TMP/pts_std" ./DTFE-2d "$TMP/pan.hdf5" "$TMP/pts_std" --grid 64 --periodic --MpcUnit 1 --verbose 2 --field density --sample-points "$TMP/centres.bin"
echo ">> --ps-window (sampled and exact) and --ps-caustic-cusps (a modulated pancake with two cusps)"
WIN=(--ps-window 20 47.5 30 90)
RUN "$TMP/w_full"  ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/w_full"  "${COMMON[@]}" --field density velocity --avg-subsamples 2
RUN "$TMP/w_win"   ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/w_win"   "${COMMON[@]}" --field density velocity --avg-subsamples 2 "${WIN[@]}"
RUN "$TMP/w_xfull" ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/w_xfull" "${COMMON[@]}" --field density --ps-exact-deposit
RUN "$TMP/w_xwin"  ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/w_xwin"  "${COMMON[@]}" --field density --ps-exact-deposit "${WIN[@]}"
# a partitioned window over a tessellation cache: the first run writes the partitions and their occupancy
# maps, the second skips the partitions whose streams never reach the window (2D maps since 2026-10-03)
WCACHE="$TMP/w_cache"; rm -rf "$WCACHE"; mkdir -p "$WCACHE"
WP=( --field density --avg-subsamples 2 --partition 2 2 --max-concurrent 1 )
RUN "$TMP/w_pfull" ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/w_pfull" "${COMMON[@]}" "${WP[@]}"
SMALL=(--ps-window 5 20 5 20)
RUN "$TMP/w_pc1" ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/w_pc1" "${COMMON[@]}" "${WP[@]}" "${SMALL[@]}" --tessellation-cache "$WCACHE"
RUN "$TMP/w_pc2" ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/w_pc2" "${COMMON[@]}" "${WP[@]}" "${SMALL[@]}" --tessellation-cache "$WCACHE"
GEN --out "$TMP/mod.hdf5" --n 128 --box 100 --amplitude-factor 1.2 --modulation 0.3 --jitter-frac 0.02 --seed 11
RUN "$TMP/cu_mod"  ./PS-DTFE-2d "$TMP/mod.hdf5" "$TMP/cu_mod" --grid 256 --periodic --MpcUnit 1 --verbose 2 --field density --ps-caustics --ps-caustic-cusps
RUN "$TMP/cu_flat" ./PS-DTFE-2d "$TMP/pan.hdf5" "$TMP/cu_flat" "${COMMON[@]}" --field density --ps-caustics --ps-caustic-cusps
echo ">> the standard 2D build's --exact-average (a perfect lattice, a single-stream pancake)"
GEN --out "$TMP/lat.hdf5" --n 64 --box 100 --amplitude-factor 0 --jitter-frac 0
GEN --out "$TMP/span.hdf5" --n 64 --box 100 --amplitude-factor 0.8 --jitter-frac 0.05 --seed 21
RUN "$TMP/ea_lat"  ./DTFE-2d "$TMP/lat.hdf5"  "$TMP/ea_lat"  --grid 48 --periodic --MpcUnit 1 --verbose 2 --field density_a --exact-average
RUN "$TMP/ea_lats" ./DTFE-2d "$TMP/lat.hdf5"  "$TMP/ea_lats" --grid 48 --periodic --MpcUnit 1 --verbose 2 --field density_a --samples 400
RUN "$TMP/ea_pan"  ./DTFE-2d "$TMP/span.hdf5" "$TMP/ea_pan"  --grid 48 --periodic --MpcUnit 1 --verbose 2 --field density_a velocity_a --exact-average
RUN "$TMP/ea_pans" ./DTFE-2d "$TMP/span.hdf5" "$TMP/ea_pans" --grid 48 --periodic --MpcUnit 1 --verbose 2 --field density_a velocity_a --samples 400
"$PY" - "$TMP" "$G" "$ROOT" <<'PYEOF'
import numpy as np, sys
from pathlib import Path
tmp = Path(sys.argv[1]); G = int(sys.argv[2]); L = 100.0
sys.path.insert(0, str(Path(sys.argv[3]) / "python"))
fails = []
def check(tag, ok, msg):
    print(f"   {'OK  ' if ok else 'FAIL'} {tag} {msg}")
    if not ok: fails.append(tag)
def grid(name, ext, n=G, ncomp=1):
    a = np.fromfile(tmp / f"{name}.{ext}", np.float32).astype(np.float64)
    return a.reshape((n, n) + ((ncomp,) if ncomp > 1 else ()))

# analytic 1-D Zel'dovich wave x = q + A sin(kq): streams and density per Eulerian x
def zeldovich(x, Ak):
    k = 2*np.pi/L; A = Ak/k
    q = np.linspace(0, L, 400_001)[:-1]
    f = lambda xc: (q + A*np.sin(k*q) - xc + L/2) % L - L/2
    n, rho, vel = [], [], []
    for xc in x:
        fx = f(xc); i = np.where((np.sign(fx[:-1]) != np.sign(fx[1:])) & (np.abs(fx[:-1]) < 1.0))[0]
        J = 1 + Ak*np.cos(k*q[i]); n.append(len(i)); rho.append(np.sum(1/np.abs(J)))
        vel.append(100*A*np.sin(k*q[i][0]) if len(i) == 1 else np.nan)
    return np.array(n), np.array(rho), np.array(vel)
x = (np.arange(G) + 0.5) * L / G
n_ana, rho_ana, v_ana = zeldovich(x, 1.8)
caustic = np.zeros(G, bool)          # columns at or next to an analytic caustic (the stream count jumps)
jump = np.where(np.diff(n_ana) != 0)[0]
for j in jump: caustic[max(0, j-1):j+3] = True

# P: streams and mass
for name, label in (("pan_s", "sampled"), ("pan_x", "exact")):
    st = grid(name, "a_streams"); den = grid(name, "a_den")
    col = np.rint(st.mean(axis=1))
    ok = np.all(col[~caustic] == n_ana[~caustic])
    check(f"P {label}", ok and abs(den.mean() - 1) < 1e-3,
          f"stream count per column = analytic away from the {caustic.sum()} caustic columns "
          f"({np.mean(col[~caustic] == n_ana[~caustic]):.1%}); mean density {den.mean():.7f}")
stx = grid("pan_x", "a_streams")
check("P exact multiplicity", abs(stx.mean() - n_ana.mean()) < 0.02,
      f"the exact deposit's area-weighted stream count averages {stx.mean():.4f} (analytic {n_ana.mean():.4f})")

# C: convergence of the single-stream density
def ana_prof(xr, amp=1.8):
    q = np.linspace(0, 1, 400000); xb = q + amp/(2*np.pi)*np.sin(2*np.pi*q); jac = 1 + amp*np.cos(2*np.pi*q)
    ss = jac > 0; xs = np.mod(xb[ss], 1); r = 1/jac[ss]; o = np.argsort(xs); return np.interp(xr, xs[o], r[o])
xr = (np.arange(96) + 0.5) / 96; sel = ((xr > 0.05) & (xr < 0.35)) | ((xr > 0.65) & (xr < 0.95))
errs = []
for n in (16, 24, 32, 48):
    d = grid(f"conv_{n}", "a_den", 96); prof = d.mean(axis=1) / d.mean()
    errs.append(float(np.mean(np.abs(prof[sel] - ana_prof(xr)[sel]))))
check("C convergence", errs[-1] < 0.6*errs[0] and errs[-1] < 0.008,
      "single-stream density L1 error " + " -> ".join(f"{e:.4f}" for e in errs))

# V: velocity in the single-stream region
v = grid("pan_s", "a_vel", ncomp=2)[..., 0].mean(axis=1)
ss = (n_ana == 1) & ~caustic
rel = np.abs(v[ss] - v_ana[ss]) / (100*1.8/(2*np.pi/L))
check("V single-stream velocity", float(np.median(rel)) < 0.01 and float(rel.max()) < 0.05,
      f"|v - v_analytic| / v_max: median {np.median(rel):.1e}, max {rel.max():.1e}")
vy = grid("pan_s", "a_vel", ncomp=2)[..., 1]
check("V transverse velocity", float(np.abs(vy).max()) < 1e-3 * 100*1.8/(2*np.pi/L), f"max |v_y| {np.abs(vy).max():.2e} (the wave is along x)")

# X: crossed waves, separable stream counts
sx = np.rint(grid("crw", "streams"))
vals = set(np.unique(sx).astype(int))
n1, _, _ = zeldovich(x, 1.5)
f3 = float(np.mean(n1 == 3)); exp9, exp3 = f3*f3, 2*f3*(1-f3)
got9, got3 = float(np.mean(sx == 9)), float(np.mean(sx == 3))
check("X crossed waves", vals <= {1, 3, 9} and 9 in vals and abs(got9 - exp9) < 0.02 and abs(got3 - exp3) < 0.03,
      f"point stream counts {sorted(vals)}; 9-stream fraction {got9:.3f} (separable {exp9:.3f}), "
      f"3-stream {got3:.3f} ({exp3:.3f})")
dc = grid("crw", "a_den"); check("X mass", abs(dc.mean() - 1) < 1e-5, f"mean density {dc.mean():.7f}")

# E: the sampled deposit converges to the exact one (first order in the sample spacing); next to a
# caustic the triangles are thinner than any sample spacing, so those columns are left out
far = ~caustic
ex = grid("pan_x", "a_den")
d6 = np.abs(ex[far] - grid("pan_s6", "a_den")[far]).sum() / np.abs(ex[far]).sum()
d12 = np.abs(ex[far] - grid("pan_s12", "a_den")[far]).sum() / np.abs(ex[far]).sum()
check("E sampled -> exact density", d12 < 0.7 * d6 and d12 < 0.012,
      f"relative L1 |exact - sampled| away from the caustics: nSub 6 {d6:.2e} -> nSub 12 {d12:.2e}")
for ext, nc, tol in (("a_vel", 2, 0.02), ("a_velDisp", 1, 0.08)):
    a = grid("pan_x", ext, ncomp=nc); b = grid("pan_s6", ext, ncomp=nc)
    num = np.abs(a - b).sum(); den = np.abs(b).sum() + 1e-30
    check(f"E exact ~ sampled {ext}", num/den < tol, f"relative L1 difference {num/den:.2e} (< {tol})")

# H: the exact folded-overlap flag
hid = grid("pan_x", "a_hidden_streams").astype(int)
flag = (hid & 1) != 0
far1 = (n_ana == 1) & ~caustic
check("H no false flag", not flag[far1].any(), f"{int(flag[far1].sum())} single-stream cells away from the caustics flagged")
st_s = grid("pan_x", "a_streams")
multi_cols = np.where(n_ana >= 3)[0]
covered = all(((flag[c] | (st_s[c] > 1.5)).all()) for c in multi_cols if not caustic[c])
check("H multi-stream marked", covered, "every multi-stream column is either counted (> 1 stream) or flagged")

# F: caustics
fold = grid("pan_x", "caustic") > 0
cls = grid("pan_x", "causticClass").astype(int)
check("F parity bits == .caustic", np.array_equal((cls & 3) == 3, fold), f"{int(fold.sum())} fold cells")
inside3 = (n_ana == 3) & ~caustic; single = (n_ana == 1) & ~caustic
check("F fold region", fold[inside3].all() and not fold[single].any(),
      "set in every cell of the analytic multi-stream region, in no single-stream cell (caustic columns aside)")
k1 = (cls & (1 << 3)) != 0; k0 = (cls & (1 << 2)) != 0
inside = (n_ana == 3) & ~caustic; outside = (n_ana == 1) & ~caustic
check("F collapse multiplicity", k1[inside].all() and not k1[outside].any() and k0[outside].all(),
      "k = 1 (one axis collapsed) everywhere inside the pancake, nowhere outside; k = 0 outside")

# S: splits, mass-convention flags, the non-periodic clump, the cache
a, b = grid("pan_p1", "a_den"), grid("pan_p2", "a_den")
diff = np.abs(a - b) / np.abs(a).max() > 1e-5
check("S partition 2 2 == 1 1", np.array_equal(grid("pan_p1", "a_streams"), grid("pan_p2", "a_streams"))
      and diff.mean() < 1e-3 and caustic[np.where(diff)[0]].all(),
      f"streams identical; density differs beyond float rounding in {int(diff.sum())} cells, all on caustic columns")
for name, what, tol in (("pan_vm", "vertex mass + volume weighted", 1e-6), ("pan_hr", "halo release D=2", 1e-3),
                        ("pan_lin", "linear deposit", 1e-3)):
    d = grid(name, "a_den"); check(f"S {what}", abs(d.mean() - 1) < tol, f"mean density {d.mean():.7f} (to {tol:g})")
cl = grid("clump", "den")
spacing = 0.6 * L / 64; area = (0.6 * L - spacing) ** 2
mass = float(cl.sum() * (L / G) ** 2 / area)
lo, hi = int(0.3 * G), int(0.7 * G)
check("S non-periodic clump", abs(mass - 1) < 0.02 and (cl[lo:hi, lo:hi] > 0).all(),
      f"recovered mass {mass:.4f}, interior covered {(cl[lo:hi, lo:hi] > 0).mean():.1%}")
log2 = (tmp / "pan_c2.log").read_text()
same = all((tmp / f"pan_c1.{e}").read_bytes() == (tmp / f"pan_c2.{e}").read_bytes() for e in ("a_den", "a_streams"))
check("S tessellation cache", "Reused the cached tessellation" in log2 and same, "the second run hits and its grids are identical")

# PE: point evaluation
pst = np.fromfile(tmp / "pts.pts_streams", np.int32)
gst = np.rint(np.fromfile(tmp / "pts.streams", np.float32)).astype(np.int32)
check("PE1 point streams == grid streams", pst.size == G*G and np.array_equal(pst, gst) and pst.max() == 3,
      f"{pst.size} cell centres, {int((pst != gst).sum())} differ, max {pst.max()}")
pv = np.fromfile(tmp / "pts.pts_vel", np.float64).reshape(G, G, 2)
pst2 = pst.reshape(G, G)
single = (n_ana == 1) & ~caustic
rel = np.abs(pv[single, :, 0] - v_ana[single][:, None]) / np.abs(v_ana[single][:, None]).clip(1.)
ok1 = (pst2[single] == 1).all()
check("PE2 point velocity", ok1 and float(np.median(rel)) < 0.01 and float(np.abs(pv[..., 1]).max()) < 1e-3 * 100*1.8/(2*np.pi/L),
      f"median rel {float(np.median(rel)):.2e}, max |v_y| {float(np.abs(pv[..., 1]).max()):.2e}")
d1 = np.fromfile(tmp / "pts.pts_den", np.float64); d2 = np.fromfile(tmp / "pts_p2.pts_den", np.float64)
s2 = np.fromfile(tmp / "pts_p2.pts_streams", np.int32)
check("PE3 partition 2 2", np.array_equal(pst, s2) and float(np.abs(d1 - d2).max() / d1.max()) < 1e-6,
      f"streams identical {np.array_equal(pst, s2)}, max rel density difference {float(np.abs(d1 - d2).max() / d1.max()):.1e}")
same = all((tmp / f"pts_tx.{e}").read_bytes() == (tmp / f"pts_bn.{e}").read_bytes() for e in ("pts_den", "pts_streams", "pts_vel"))
check("PE4 text points == binary points", same, "997 points from a text file and from a binary file")
# PE5/PE7/PE8: the 2D server through dtfelib's Estimator
from dtfelib import Estimator
root = Path(sys.argv[3])
centres = np.fromfile(tmp / "centres.bin", np.float64).reshape(-1, 2)
common = dict(dim=2, binary=root / "PS-DTFE-2d", periodic=True, mpc_unit=1.0, peculiar=False)
with Estimator(tmp / "pan.hdf5", **common) as est:
    f = est(centres)
    same = (np.array_equal(f.density, np.fromfile(tmp / "pts.pts_den", np.float64))
            and np.array_equal(f.streams, pst) and np.array_equal(f.velocity.reshape(-1), np.fromfile(tmp / "pts.pts_vel", np.float64))
            and est.dim == 2 and f.velocity.shape == (G*G, 2) and f.dispersion.shape == (G*G, 3))
    g = est.grid((np.arange(4) + 0.5) * L / 4, (np.arange(3) + 0.5) * L / 3)
    grid_ok = g.density.shape == (4, 3) and g.velocity.shape == (4, 3, 2)
check("PE5 Estimator(dim=2) == --sample-points", same and grid_ok,
      f"{G*G} points byte-identical (density, streams, velocity); grid(xs, ys) shapes {g.density.shape}, {g.velocity.shape}")
import tempfile
with tempfile.TemporaryDirectory(dir=tmp) as cache:
    with Estimator(tmp / "pan.hdf5", partition=2, tessellation_cache=cache, **common) as est:
        f2 = est(centres)
        comp_ok = (est.n_partitions == 4 and np.array_equal(f2.streams, s2)
                   and np.array_equal(f2.density, np.fromfile(tmp / "pts_p2.pts_den", np.float64)))
check("PE7 composite 2 x 2 server == the same split's points", comp_ok,
      f"{est.n_partitions} partitions; streams and density byte-identical to --sample-points --partition 2 2")
import h5py
with h5py.File(tmp / "pan.hdf5", "r") as h:
    x = h["PartType1/Coordinates"][:].astype(np.float64); q = h["PartType1/InitialCoordinates"][:].astype(np.float64)
    v = h["PartType1/Velocities"][:].astype(np.float64)
with Estimator.from_arrays(x, velocities=v, lagrangian=q, box=L, binary=root / "PS-DTFE-2d") as est:
    fa = est(centres)
    rel = float(np.abs(fa.density - f.density).max() / f.density.max())
check("PE8 Estimator.from_arrays with (N, 2) arrays", est.dim == 2 and np.array_equal(fa.streams, f.streams) and rel < 1e-6,
      f"a 2D estimator from arrays: streams identical to the file-based server, density to {rel:.1e}")
sd2 = np.fromfile(tmp / "pts_std.pts_den", np.float64)
check("PE6 standard DTFE-2d points", sd2.size == G*G and np.isfinite(sd2).all() and sd2.min() > 0 and abs(sd2.mean() - 1) < 0.15,
      f"{sd2.size} points, mean {sd2.mean():.3f}")

# W: the window = the full run's cells (dx = 100/128: x cells [25, 61), y cells [38, 116))
xs, ys = slice(25, 61), slice(38, 116)
def win(name, ext):
    return np.fromfile(tmp / f"{name}.{ext}", np.float32).astype(np.float64).reshape(36, 78)
same_den = np.array_equal(grid("w_full", "den")[xs, ys], win("w_win", "den"))
same_str = np.array_equal(grid("w_full", "streams")[xs, ys], win("w_win", "streams"))
# '.hidden_streams' is what the window port changed (markFlipped walks the window's hull only)
same_hid = np.array_equal(grid("w_full", "hidden_streams")[xs, ys], win("w_win", "hidden_streams"))
fv = grid("w_full", "vel", ncomp=2)[xs, ys]
wv = np.fromfile(tmp / "w_win.vel", np.float32).astype(np.float64).reshape(36, 78, 2)
check("W sampled window == full run", same_den and same_str and same_hid and np.array_equal(fv, wv),
      f"density {same_den}, streams {same_str}, hidden streams {same_hid}, velocity {np.array_equal(fv, wv)} (36 x 78 cells)")
fx, wx = grid("w_xfull", "den")[xs, ys], win("w_xwin", "den")
rel = float(np.abs(fx - wx).max() / fx.max())
hid_x = np.array_equal(grid("w_xfull", "hidden_streams")[xs, ys], win("w_xwin", "hidden_streams"))
check("W exact window == full run", rel < 1e-5 and hid_x and np.array_equal(grid("w_xfull", "streams")[xs, ys], win("w_xwin", "streams")),
      f"max relative density difference {rel:.1e}, streams identical, hidden streams identical {hid_x}")

# W3: the cached, partitioned window skips partitions (occupancy maps) and still equals the full run;
# dx = 100/128: x and y cells [6, 26)
log2 = (tmp / "w_pc2.log").read_text(errors="replace")
xs3 = slice(6, 26)
w2 = np.fromfile(tmp / "w_pc2.den", np.float32).astype(np.float64)
full3 = grid("w_pfull", "den")[xs3, xs3]
same3 = w2.size == full3.size and np.array_equal(full3.reshape(-1), w2)
check("W3 partitioned window over the cache", same3 and "never reach the window" in log2,
      f"density == the full run's window cells: {same3}; partitions skipped by their occupancy maps: "
      f"{'yes' if 'never reach the window' in log2 else 'no'}")

# CU: cusps of the modulated pancake (A k = 1.2, eps = 0.3): at k q_x = pi (Eulerian x = 50, where
# psi_x = 0) and cos(k q_y) = (1/(A k) - 1)/eps, i.e. y = 34.4 and 65.6
cm = np.fromfile(tmp / "cu_mod.causticClass", np.float32).astype(np.int64).reshape(256, 256)
cf = np.fromfile(tmp / "cu_flat.causticClass", np.float32).astype(np.int64).reshape(G, G)
cusp = (cm & 128) > 0
ix, iy = np.where(cusp); cx = (ix + 0.5) * L / 256; cy = (iy + 0.5) * L / 256
yc = L * np.arccos((1 / 1.2 - 1) / 0.3) / (2 * np.pi)
near = [float(np.hypot(cx - 50, cy - y0).min()) if cx.size else 99. for y0 in (yc, L - yc)]
check("CU1 no cusp on a plain pancake", int(((cf & 128) > 0).sum()) == 0 and int(((cf & 3) == 3).sum()) > 0,
      f"{int(((cf & 128) > 0).sum())} cusp cells among {int(((cf & 3) == 3).sum())} fold cells (theory: none)")
check("CU2 both cusps flagged", max(near) < 1.0, f"nearest flagged cell {near[0]:.2f} / {near[1]:.2f} Mpc from the cusps at y = {yc:.1f}, {L - yc:.1f}")
check("CU3 flags on the ridge", cx.size > 0 and float(np.abs(cx - 50).max()) < 1.0,
      f"{cx.size} flagged cells, all within {float(np.abs(cx - 50).max()) if cx.size else 0:.2f} Mpc of x = 50 (k q_x = pi)")
check("CU4 no swallowtail bit in 2D", not ((cm & 256) > 0).any() and not ((cf & 256) > 0).any(), "bit 8 never set")

# EA: the standard build's exact average
def a48(name, ext="a_den"):
    return np.fromfile(tmp / f"{name}.{ext}", np.float32).astype(np.float64)
lx, ls = a48("ea_lat"), a48("ea_lats")
check("EA1 exact average on a lattice", float(np.abs(lx - 1).max()) < 1e-5 and float(np.abs(ls - 1).max()) > 10 * float(np.abs(lx - 1).max()),
      f"max |rho - 1| {float(np.abs(lx - 1).max()):.1e} (400 samples per cell: {float(np.abs(ls - 1).max()):.1e})")
px, ps_ = a48("ea_pan"), a48("ea_pans")
check("EA2 exact average conserves mass", abs(px.mean() - 1) < 1e-6 and px.size == 48*48, f"mean {px.mean():.9f}")
l1 = float(np.abs(px - ps_).mean() / px.mean())
check("EA3 sampled ~ exact", l1 < 0.02, f"relative L1 |exact - sampled(400)| {l1:.1e}")

# D: the standard 2D build
sd = np.fromfile(tmp / "std.a_den", np.float32)
check("D standard DTFE-2d", sd.size == 64*64 and np.isfinite(sd).all() and sd.min() > 0, f"{sd.size} cells, mean {sd.mean():.4g}, min {sd.min():.3g}")

print("RESULT:", "PASS  (2D phase space: analytic stream counts, mass, convergence, velocity, separable crossed "
      "waves, exact == sampled, hidden and fold flags, splits and options, point evaluation, window, cusps; "
      "the standard 2D build and its exact average)" if not fails
      else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
PYEOF
