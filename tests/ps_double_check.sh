#!/usr/bin/env bash
# The double-precision pair (DTFE-double / PS-DTFE-double, 'make DTFE PS-DTFE DOUBLE=1 [METAL=1]')
# against the single-precision pair, on the same crossed-wave snapshot:
#   D1  the double binaries write float64 grids (file size = 8 bytes per value) and the float ones float32
#   D2  phase-space sampled cell averages, CPU: double vs single agree the way two float paths do (means to
#       1e-5, per cell to 5e-2 of the peak outside a 1e-3 fraction: a few borderline samples change cell, and a
#       handful of co-spherical tetrahedra resolve differently in double); stream counts and caustic flags identical
#   D3  phase-space point evaluation (slice): double vs single densities in the same class, streams identical
#   D4  standard-DTFE cell averages, CPU: double vs single in the same class
#   D5  GPU deposits of the DOUBLE binaries (--ps-gpu / --gpu): the GPU kernels are single precision and the
#       double build hands them float copies of its tetrahedra, so each must match ITS OWN CPU run the way the
#       single build's GPU matches its CPU (means to 1e-5, per-cell to 5e-2 outside a 1e-3 fraction) and must
#       conserve mass; skipped when the double pair was built without METAL=1
#   D6  the Python side reads float64 grids: python/gui/grids.py infers the dtype from the file size
# Both pairs must be built (./DTFE ./PS-DTFE ./DTFE-double ./PS-DTFE-double); --no-build is accepted and ignored.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
for b in DTFE PS-DTFE DTFE-double PS-DTFE-double; do
    [ -x "./$b" ] || { echo "SKIP: ./$b is not built (make DTFE PS-DTFE DOUBLE=1 builds the double pair)"; exit 0; }
done
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/double_check"
mkdir -p "$TMP"
SNAP="$TMP/waves.hdf5"
[ -f "$SNAP" ] || "$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 48 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
"$PY" - "$TMP" <<'PYEOF'
import numpy as np, sys
from pathlib import Path
tmp = Path(sys.argv[1]); n = 64
pts = np.stack(np.meshgrid(np.linspace(0.5, 99.5, n), np.linspace(0.5, 99.5, n), [50.0], indexing="ij"), -1).reshape(-1, 3)
pts.astype(np.float64).tofile(tmp / "slice.bin")
PYEOF
G=48
COMMON=(--grid $G --periodic --input 105 --MpcUnit 1 --verbose 1)
run() { local out="$1"; shift; "$@" > "$out.log" 2>&1 || { echo "FAIL run $(basename "$out") (see $out.log)"; exit 1; }; }
# the banners: DTFE prints "Metal GPU (...)" and PS-DTFE "Metal GPU deposit (...)", both at verbose 2;
# every fallback path prints a warning naming the CPU
gpu_ok() { grep -q " GPU (\| GPU deposit (" "$1" && ! grep -q "using the CPU deposit\|using the CPU interpolation\|deposit unavailable" "$1"; }
echo ">> phase-space grids and slices, both pairs, CPU"
run "$TMP/ps_f"  ./PS-DTFE        "$SNAP" "$TMP/ps_f"  "${COMMON[@]}" --field density_a velocity_a --avg-subsamples 2 --ps-caustics
run "$TMP/ps_d"  ./PS-DTFE-double "$SNAP" "$TMP/ps_d"  "${COMMON[@]}" --field density_a velocity_a --avg-subsamples 2 --ps-caustics
run "$TMP/pts_f" ./PS-DTFE        "$SNAP" "$TMP/pts_f" --grid 16 --periodic --input 105 --MpcUnit 1 --verbose 1 --field density --sample-points "$TMP/slice.bin"
run "$TMP/pts_d" ./PS-DTFE-double "$SNAP" "$TMP/pts_d" --grid 16 --periodic --input 105 --MpcUnit 1 --verbose 1 --field density --sample-points "$TMP/slice.bin"
echo ">> standard DTFE cell averages, both pairs, CPU"
run "$TMP/dt_f"  ./DTFE        "$SNAP" "$TMP/dt_f" "${COMMON[@]}" --field density_a velocity_a
run "$TMP/dt_d"  ./DTFE-double "$SNAP" "$TMP/dt_d" "${COMMON[@]}" --field density_a velocity_a
DOUBLE_GPU=0
if [ -d o_ps_d ] && [ ! -f o_ps_d/.gpu_mode_off ] && [ -d o_d ] && [ ! -f o_d/.gpu_mode_off ]; then   # any GPU backend stamp
    DOUBLE_GPU=1
    echo ">> GPU deposits of the double pair"
    run "$TMP/ps_dg" ./PS-DTFE-double "$SNAP" "$TMP/ps_dg" --grid $G --periodic --input 105 --MpcUnit 1 --verbose 2 --field density_a velocity_a --avg-subsamples 2 --ps-caustics --ps-gpu
    gpu_ok "$TMP/ps_dg.log" || { echo "FAIL: PS-DTFE-double --ps-gpu did not run on the GPU"; exit 1; }
    run "$TMP/dt_dg" ./DTFE-double "$SNAP" "$TMP/dt_dg" --grid $G --periodic --input 105 --MpcUnit 1 --verbose 2 --field density_a velocity_a --gpu
    gpu_ok "$TMP/dt_dg.log" || { echo "FAIL: DTFE-double --gpu did not run on the GPU"; exit 1; }
else
    echo ">> the double pair has no GPU (built without METAL=1 / CUDA=1 / HIP=1): D5 skipped"
fi
"$PY" - "$TMP" "$G" "$DOUBLE_GPU" "$ROOT" <<'PYEOF'
import sys, numpy as np
from pathlib import Path
tmp, G, dgpu, root = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1", Path(sys.argv[4])
ok = True
def check(name, cond, detail=""):
    global ok
    print(("   OK   " if cond else "   FAIL ") + name + ("" if cond or not detail else f"  [{detail}]"))
    ok = ok and cond
def rd(path, ncomp=1, want=None):
    """A raw grid, dtype from the file size (G^3 * ncomp values); also returns the bytes per value."""
    size = Path(path).stat().st_size
    per = size // (G ** 3 * ncomp)
    assert per in (4, 8) and size == G ** 3 * ncomp * per, f"{path}: {size} bytes for {G}^3 x {ncomp}"
    a = np.fromfile(path, dtype=np.float32 if per == 4 else np.float64).astype(np.float64)
    return (a.reshape(-1, ncomp) if ncomp > 1 else a), per
def cmp(a, b):
    sc = float(np.max(np.abs(a))) or 1.0
    d = np.abs(a - b)
    return abs(float(a.mean() - b.mean())) / sc, float(d.max()) / sc, float((d > 1e-3 * sc).mean())
def close(m, p, f):      # the parity class of tests/dtfe_metal_check.sh (sample flips, not a different estimator)
    return m < 1e-5 and p < 5e-2 and f < 1e-3
# D1 dtype
for name, per_want in (("ps_f.a_den", 4), ("ps_d.a_den", 8), ("dt_f.a_den", 4), ("dt_d.a_den", 8)):
    _, per = rd(tmp / name)
    check(f"D1 {name} is written with {per_want} bytes per value", per == per_want, str(per))
# D2 phase-space grids: double vs single
fd, _ = rd(tmp / "ps_f.a_den"); dd, _ = rd(tmp / "ps_d.a_den")
m, p, f = cmp(fd, dd)
check("D2 PS density, double vs single: mean 1e-5, per cell 5e-2 of peak outside a 1e-3 fraction", close(m, p, f), f"mean-rel {m:.1e} peak-rel {p:.1e} frac {f:.1e}")
fs, _ = rd(tmp / "ps_f.a_streams"); ds, _ = rd(tmp / "ps_d.a_streams")
check("D2 PS stream counts identical", np.array_equal(fs, ds), f"{(fs != ds).sum()} cells differ")
fv, _ = rd(tmp / "ps_f.a_vel", 3); dv, _ = rd(tmp / "ps_d.a_vel", 3)
m, p, f = cmp(fv, dv)
check("D2 PS velocity, double vs single in the parity class", close(m, p, f), f"mean-rel {m:.1e} peak-rel {p:.1e} frac {f:.1e}")
fc, _ = rd(tmp / "ps_f.caustic"); dc, _ = rd(tmp / "ps_d.caustic")
check("D2 PS caustic flags identical", np.array_equal(fc, dc), f"{(fc != dc).sum()} cells differ")
# D3 point evaluation (float64 in both builds)
pf = np.fromfile(tmp / "pts_f.pts_den"); pd = np.fromfile(tmp / "pts_d.pts_den")
sf = np.fromfile(tmp / "pts_f.pts_streams", dtype=np.int32); sd = np.fromfile(tmp / "pts_d.pts_streams", dtype=np.int32)
m, p, f = cmp(pf, pd)
check("D3 point densities, double vs single in the parity class", pf.shape == pd.shape and close(m, p, f), f"mean-rel {m:.1e} peak-rel {p:.1e} frac {f:.1e}")
check("D3 point stream counts identical", np.array_equal(sf, sd))
# D4 standard DTFE
fd, _ = rd(tmp / "dt_f.a_den"); dd, _ = rd(tmp / "dt_d.a_den")
m, p, f = cmp(fd, dd)
check("D4 standard-DTFE density, double vs single in the parity class", close(m, p, f), f"mean-rel {m:.1e} peak-rel {p:.1e} frac {f:.1e}")
# D5 the double pair's GPU against its own CPU (the dtfe_metal_check tolerances)
if dgpu:
    for pre, suf, comps in (("ps_d", "a_den", 1), ("ps_d", "a_vel", 3), ("dt_d", "a_den", 1), ("dt_d", "a_vel", 3)):
        c, _ = rd(tmp / f"{pre}.{suf}", comps); g, perg = rd(tmp / f"{pre}g.{suf}", comps)
        m, p, f = cmp(c, g)
        check(f"D5 {pre} {suf}: GPU (float deposit) vs CPU (double): mean 1e-5, peak 5e-2, frac>1e-3 below 1e-3; float64 file",
              m < 1e-5 and p < 5e-2 and f < 1e-3 and perg == 8, f"mean-rel {m:.1e} peak-rel {p:.1e} frac {f:.1e} bytes {perg}")
    c, _ = rd(tmp / "ps_d.a_streams"); g, _ = rd(tmp / "ps_dg.a_streams")
    check("D5 PS-DTFE-double GPU stream counts identical to its CPU", np.array_equal(c, g), f"{(c != g).sum()} differ")
    c, _ = rd(tmp / "ps_d.a_den"); g, _ = rd(tmp / "ps_dg.a_den")
    check("D5 PS-DTFE-double GPU conserves the mass (sum to 1e-6)", abs(c.sum() - g.sum()) / c.sum() < 1e-6, f"{abs(c.sum()-g.sum())/c.sum():.1e}")
# D6 the GUI reader
sys.path.insert(0, str(root / "python" / "gui"))
import grids as GR
o = GR.OutputSet(tmp / "ps_d")
plane = o.slice("density", 2, G // 2) if o else None
check("D6 python/gui/grids.py opens the float64 grids and reads a plane", bool(o) and o.n == G and plane is not None
      and plane.shape == (G, G) and np.isfinite(plane).all() and abs(float(plane.mean()) - float(rd(tmp / "ps_d.a_den")[0].reshape(G, G, G)[:, :, G // 2].mean())) < 1e-9,
      f"n={getattr(o, 'n', None)} dtype={getattr(o, 'dtype', None)}")
print("RESULT:", "PASS  (the double pair matches the single pair to float rounding, writes float64, and its GPU deposit matches its CPU like the single build's)" if ok else "FAIL")
sys.exit(0 if ok else 1)
PYEOF
