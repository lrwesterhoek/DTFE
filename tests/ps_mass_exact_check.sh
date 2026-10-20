#!/usr/bin/env bash
# The PS-DTFE MASS grid is exact on the CPU too. The GPU kernels sum it as 64-bit fixed point, and the CPU's
# shares -- the GPU-DEFERRED tetrahedra (a sample within float rounding of a face), the --ps-halo-release centroid
# deposits, and a whole CPU deposit (no GPU, or the GPU gave up) -- go into the same integers: the CPU adds
# rint(share * 2^k) (ps_interpolation.cc 'massFx', the GPU's own words after a GPU deposit) and converts once.
# A float add would round a share against the cell's running sum (below half an ulp it vanishes, always
# downward; TNG50-3-Dark z=0.5: a box mean 1.63e-5 low, all of it the CPU's deferred deposit).
# Asserted on two synthetic snapshots:
#   X1  deferral-heavy: a 32^3 lattice WITHOUT jitter, displaced along x only (A k = 0.5), on a 2^3 grid at nSub 16
#       (--partition 1 1 1, and 2 2 2): every sample lies exactly on a lattice plane y or z = const, i.e. on
#       tetrahedron faces, which the GPU kernel cannot classify in float -- it defers 56% of the tetrahedra (110592
#       of 196608) to the CPU. Box mean of '.a_den' = 1 to 1e-7: GPU, GPU with --ps-halo-release 1.2 (40% of the
#       mass released to host-side centroid deposits) and CPU. Float adds give +1.9e-4 (GPU), +1.4e-4 (GPU,
#       released) and -1.4e-5 (CPU) here. The 2^3-partition runs to 2.5e-7 in single precision: the
#       merge of 8 float partition grids rounds by an ulp (1.19e-7 at 1), deterministically so in index order.
#   X2  heavy accumulation, CPU: a 64^3 pancake on a 4^3 grid (nSub 4, --ps-vertex-mass, so the tets carry exactly
#       the particles' mass): 1.8e6 tetrahedra into 64 cells. Box mean = 1 to 1e-7 (the float sums: +2.3e-5 on one
#       thread, -1.6e-6 on ten).
#   X3  thread independence: X2's '.a_den' is byte-identical for 1 thread, 4 threads in the copy mode
#       (DTFE_DEPOSIT_SLABS=0: one shared grid, atomic adds -- float copies summed per thread would differ) and 4 in
#       slab mode; X1's GPU deposit likewise with its deferred tets on 1 and 4 threads -- the logs must show the
#       deferred tets deposited on 1 thread and on more than 1 in slab mode (that path never keeps float copies, so the bytes
#       alone would also match on a binary that silently ran the 4-thread run on one). (DTFE_TEST_PRECISION=double:
#       the double build keeps its double sums, which agree to 1e-12 instead of bit for bit.)
#   X4  PS_MASS_AUDIT=1, when the binary carries the audit (ps_mass_audit.h): every pass of every run above has
#       'gained/shares-1' = 0 (the CPU's shares all reached the grid; the double build's sums of 1.8e6 shares
#       round: < 1e-9 there) and 'grid/tets-1' below 1e-7 (one float rounding per cell, <= 6e-8). A binary
#       without the audit skips this check (the others still fail it).
#   X5  PS_METAL_FLOAT_MASS=1 (single build) selects the float grid, on the CPU too, for A/B comparisons:
#       X1's CPU run then writes a different '.a_den' (its mean is reported).
#   X6  the audit only observes: every run above sets PS_MASS_AUDIT=1, so X1's 4-thread GPU run and X2's copy-mode
#       run are repeated with PS_MASS_AUDIT=0 (the production path) and must write the same '.a_den'.
#   X7  the copy mode shares its counts and flags: the int stream count, the two hidden-stream flags and the
#       caustic mask are written by every thread into the real arrays (relaxed atomics), as the fixed-point words are,
#       so the single build keeps NO private copy. 32^3 crossed waves (multi-stream) on 64^3 at nSub 2, --ps-caustics,
#       PS_MASS_AUDIT=0, copy mode (DTFE_DEPOSIT_SLABS=0) on 1, 4 and 10 threads: the logs must say 'CPU deposit on 4 /
#       10 threads (copy mode' (single build: '0 MB of private accumulators'), and EVERY output file is byte-identical
#       to the 1-thread run's. Double build: the counts, flags and masks (.a_streams .a_hidden_streams .caustic
#       .causticClass) byte-identical, the double sums (.a_den .a_vel, still copies summed in thread order) to 1e-12.
# The GPU halves (X1 GPU, X3 GPU) need a GPU build (Metal, or CUDA/HIP incl. the CUDAEMU=1 emulation); they print
# SKIP otherwise and the CPU halves still run, so the suite runs in tests/ci_suite.sh (core and double). 12 s
# without the GPU halves, 23 s with them (M-series Mac, 10 cores; X7 adds ~8 s: 31 s).
# Usage: tests/ps_mass_exact_check.sh [--no-build]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}"
if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building PS-DTFE ..."
    make PS-DTFE $(cat o_ps/.build_mode 2>/dev/null || true) >/dev/null || { echo "FAIL: build"; exit 1; }
fi
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
OBJ=o_ps; precision_double && OBJ=o_ps_d
GPU=0                                                     # any one GPU backend stamp will do (gpu_mass_check.sh)
if [ -d "$OBJ" ] && [ ! -f "$OBJ/.gpu_mode_off" ]; then
    [ -n "$(gpu_backend_of "$OBJ")" ] && GPU=1
fi
TMP="${DTFE_TEST_TMP}/ps_mass_exact_check"
mkdir -p "$TMP"
LAT="$TMP/lattice32.hdf5"; PAN="$TMP/pancake64.hdf5"; CW="$TMP/crossed32.hdf5"
"$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$LAT" --n 32 --amplitude-factor 0.5 --jitter-frac 0 >/dev/null \
    || { echo "FAIL data gen"; exit 1; }
"$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$CW" --n 32 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
"$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$PAN" --n 64 >/dev/null || { echo "FAIL data gen"; exit 1; }

echo "============================================================"
echo " PS-DTFE exact mass grid (${PS_BIN}; GPU halves: $([ $GPU = 1 ] && echo yes || echo 'no GPU build: SKIP'))"
echo "============================================================"
t0=$SECONDS
# run <tag> <snapshot> [env assignments...] -- <PS-DTFE args...>: a non-zero exit fails the suite; a --ps-gpu run
# must have run the GPU deposit (no CPU fallback)
run() {
    local tag="$1" snap="$2"; shift 2
    local envs=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    rm -f "$TMP/$tag".*
    env PS_MASS_AUDIT=1 ${envs[@]+"${envs[@]}"} "${PS_BIN}" "$snap" "$TMP/$tag" --periodic --input 105 --MpcUnit 1 --verbose 2 \
        --field density_a velocity_a "$@" > "$TMP/$tag.log" 2>&1 || { echo "FAIL: $tag exited non-zero (see $TMP/$tag.log)"; exit 1; }
    case " $* " in
        *" --ps-gpu "*)
            grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead" "$TMP/$tag.log" \
                && { echo "FAIL: $tag fell back to the CPU (see $TMP/$tag.log)"; exit 1; }
            grep -q " GPU deposit (\|run total over .*GPU deposit" "$TMP/$tag.log" \
                || { echo "FAIL: $tag did not run the GPU deposit (see $TMP/$tag.log)"; exit 1; } ;;
    esac
    return 0
}
L=(--grid 2 --avg-subsamples 16)
P=(--grid 4 --avg-subsamples 4 --ps-vertex-mass --partition 1 1 1)
echo ">> X1 lattice (deferral-heavy), CPU"
run lat_cpu_p1 "$LAT" -- "${L[@]}" --partition 1 1 1
run lat_cpu_p2 "$LAT" -- "${L[@]}" --partition 2 2 2
if [ "$GPU" = 1 ]; then
    echo ">> X1 lattice, GPU (and its deferred tets on 1 and 4 CPU threads: X3)"
    run lat_gpu_p1 "$LAT" DTFE_DEPOSIT_THREADS=4 -- "${L[@]}" --partition 1 1 1 --ps-gpu
    run lat_gpu_p1_t1 "$LAT" DTFE_DEPOSIT_THREADS=1 -- "${L[@]}" --partition 1 1 1 --ps-gpu
    run lat_gpu_p2 "$LAT" -- "${L[@]}" --partition 2 2 2 --ps-gpu
    run lat_gpu_rel "$LAT" -- "${L[@]}" --partition 1 1 1 --ps-gpu --ps-halo-release 1.2
    run lat_gpu_p1_na "$LAT" DTFE_DEPOSIT_THREADS=4 PS_MASS_AUDIT=0 -- "${L[@]}" --partition 1 1 1 --ps-gpu
fi
echo ">> X2/X3 pancake (heavy accumulation), CPU on 1 thread, 4 (copy mode) and 4 (slab mode)"
run pan_t1 "$PAN" DTFE_DEPOSIT_THREADS=1 -- "${P[@]}"
run pan_t4c "$PAN" DTFE_DEPOSIT_THREADS=4 DTFE_DEPOSIT_SLABS=0 -- "${P[@]}"
run pan_t4s "$PAN" DTFE_DEPOSIT_THREADS=4 DTFE_DEPOSIT_SLABS=1 -- "${P[@]}"
run pan_t4c_na "$PAN" DTFE_DEPOSIT_THREADS=4 DTFE_DEPOSIT_SLABS=0 PS_MASS_AUDIT=0 -- "${P[@]}"
echo ">> X7 crossed waves with caustics, copy mode on 1, 4 and 10 threads (shared counts and flags)"
for t in 1 4 10; do
    run "cw_t$t" "$CW" DTFE_DEPOSIT_THREADS=$t DTFE_DEPOSIT_SLABS=0 PS_MASS_AUDIT=0 -- \
        --grid 64 --avg-subsamples 2 --ps-caustics --partition 1 1 1
done
if ! precision_double; then
    echo ">> X5 PS_METAL_FLOAT_MASS=1 (the float grid, for A/B)"
    run lat_cpu_float "$LAT" PS_METAL_FLOAT_MASS=1 -- "${L[@]}" --partition 1 1 1
fi

"$PY" - "$TMP" "$GPU" "${DTFE_TEST_PRECISION}" <<'PYEOF'
import re, sys
import numpy as np
from pathlib import Path
tmp, gpu, double = Path(sys.argv[1]), sys.argv[2] == "1", sys.argv[3] == "double"
REAL = np.float64 if double else np.float32
fails = 0
def check(tag, ok, msg):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {tag} {msg}")
def den(tag):
    return np.fromfile(tmp / f"{tag}.a_den", dtype=REAL).astype(np.float64)
def mean_ok(tag, what, tol=1e-7):
    d = den(tag)
    dev = float(d.mean() - 1.0)
    check("X1" if tag.startswith("lat") else "X2", d.size > 0 and abs(dev) < tol,
          f"{what}: box mean - 1 = {dev:+.2e} (|.| < {tol:.1e}; {d.size} cells)")
# A partitioned run adds its 8 partitions' float grids into each cell (DTFE.cpp's merge), which rounds once per add:
# 2^-23 = 1.19e-7 is ONE float ulp at 1, so 1e-7 cannot hold there. The merge is in partition index order by
# default and this lattice lands on +1.19e-7 every time (the completion order lands anywhere in +6e-8 ... +1.2e-7
# from run to run). Two ulps for the partitioned single-precision runs; the deposit itself stays
# exact (the one-domain checks, and X4's audit of the grids against the tetrahedra).
TOL_SPLIT = 1e-7 if double else 2.5e-7
def same(a, b, what, tag="X3"):
    x, y = (tmp / f"{a}.a_den").read_bytes(), (tmp / f"{b}.a_den").read_bytes()
    if double:   # double sums in thread order: equal to rounding, not bit for bit
        u, v = den(a), den(b)
        rel = float(np.max(np.abs(u - v)) / np.max(np.abs(u))) if u.size == v.size and u.size else 1.0
        check(tag, rel < 1e-12, f"{what}: '.a_den' agrees to {rel:.1e} of the peak (< 1e-12, double build)")
    else:
        check(tag, len(x) > 0 and x == y, f"{what}: '.a_den' byte-identical ({len(x)} bytes)")

mean_ok("lat_cpu_p1", "lattice, CPU, one domain")
mean_ok("lat_cpu_p2", "lattice, CPU, 2^3 partitions", TOL_SPLIT)
if gpu:
    log = (tmp / "lat_gpu_p1.log").read_text()
    m = re.search(r"GPU deposit \(.*?\), (\d+) tetrahedra \((\d+) with a sample on a face", log)
    frac = int(m.group(2)) / max(1, int(m.group(1))) if m else 0.
    check("X1", frac > 0.3, f"the lattice is deferral-heavy on the GPU: {m.group(2) if m else '?'} of "
          f"{m.group(1) if m else '?'} tetrahedra deferred to the CPU ({frac:.0%}, > 30%)")
    mean_ok("lat_gpu_p1", "lattice, GPU, one domain")
    mean_ok("lat_gpu_p2", "lattice, GPU, 2^3 partitions", TOL_SPLIT)
    rel = re.search(r"released (\d+) halo-interior", (tmp / "lat_gpu_rel.log").read_text())
    check("X1", rel is not None and int(rel.group(1)) > 0, f"--ps-halo-release 1.2 released {rel.group(1) if rel else 'no'} tetrahedra")
    mean_ok("lat_gpu_rel", "lattice, GPU, --ps-halo-release 1.2")
    same("lat_gpu_p1_t1", "lat_gpu_p1", "lattice, GPU, the deferred tets on 1 vs 4 CPU threads")
    def deferred_threads(tag):
        m = re.search(r"GPU-deferred tetrahedra on the CPU in [0-9.eE+-]+ s \((\d+) threads?\)", (tmp / f"{tag}.log").read_text())
        return int(m.group(1)) if m else None
    t1, t4 = deferred_threads("lat_gpu_p1_t1"), deferred_threads("lat_gpu_p1")
    s4 = re.search(r"GPU-deferred tetrahedra on the CPU in SLAB mode: (\d+) threads", (tmp / "lat_gpu_p1.log").read_text())
    slab4 = s4 is not None and t4 is not None and int(s4.group(1)) == t4
    check("X3", t1 == 1 and t4 is not None and t4 > 1 and slab4, f"the deferred tets ran on {t1} thread(s) and on {t4} "
          f"({'slab mode' if slab4 else 'NOT slab mode'}); want 1, and more than 1 in slab mode")
    same("lat_gpu_p1_na", "lat_gpu_p1", "lattice, GPU, PS_MASS_AUDIT=0 vs 1", "X6")
else:
    print("  SKIP X1/X3 GPU halves: no GPU build")
mean_ok("pan_t1", "pancake, CPU, 1 thread")
slab = re.search(r"CPU deposit in SLAB mode: (\d+) threads", (tmp / "pan_t4s.log").read_text())
check("X3", slab is not None and int(slab.group(1)) > 1, f"the slab run used slab mode ({slab.group(1) if slab else 'NOT'} threads)")
same("pan_t1", "pan_t4c", "pancake, CPU, 1 thread vs 4 (copy mode)")
same("pan_t1", "pan_t4s", "pancake, CPU, 1 thread vs 4 (slab mode)")
same("pan_t4c_na", "pan_t4c", "pancake, CPU, 4 threads (copy mode), PS_MASS_AUDIT=0 vs 1", "X6")

# X7: the copy mode on 1/4/10 threads, its counts and flags shared -- the thread count from the log, every file cmp'ed
SHARED = {".a_streams", ".a_hidden_streams", ".caustic", ".causticClass"}   # integer counts, flags, masks
ref = sorted(p for p in tmp.glob("cw_t1.*") if p.suffix != ".log")
exts = {p.name[len("cw_t1"):] for p in ref}
check("X7", SHARED <= exts, f"the 1-thread run wrote {' '.join(sorted(exts))} (needs {' '.join(sorted(SHARED))})")
for t in (4, 10):
    log = (tmp / f"cw_t{t}.log").read_text()
    m = re.search(r"CPU deposit on (\d+) threads? \(copy mode: ([0-9.eE+-]+) MB of private accumulators", log)
    nt, mb = (int(m.group(1)), float(m.group(2))) if m else (None, None)
    check("X7", nt == t and (double or mb == 0.), f"{t}-thread run: the copy mode on {nt} threads with {mb} MB of "
          f"private accumulators per extra thread (want {t}{'' if double else ', 0 MB: nothing copied'})")
    same_b, close, bad = [], [], []
    for p in ref:
        ext = p.name[len("cw_t1"):]
        q = tmp / f"cw_t{t}{ext}"
        x, y = p.read_bytes(), (q.read_bytes() if q.exists() else b"")
        if x == y and x:
            same_b.append(ext)
        elif double and ext not in SHARED and len(x) == len(y) and x:
            u, v = np.frombuffer(x, REAL).astype(np.float64), np.frombuffer(y, REAL).astype(np.float64)
            rel = float(np.max(np.abs(u - v)) / max(np.max(np.abs(u)), 1e-300))
            (close if rel < 1e-12 else bad).append(f"{ext} ({rel:.1e})")
        else:
            bad.append(ext)
    check("X7", not bad and same_b, f"{t} vs 1 thread: byte-identical {' '.join(same_b) or 'nothing'}"
          + (f"; to 1e-12 {' '.join(close)}" if close else "") + (f"; DIFFER {' '.join(bad)}" if bad else ""))

# X4: the audit lines, when the binary has them
num = r"([-+0-9.eE]+|nan|inf)"
rows = []
for f in sorted(tmp.glob("*.log")):
    if f.stem == "lat_cpu_float": continue
    rows += [(f.stem, l) for l in f.read_text().splitlines() if l.startswith("PS mass audit: partition")]
if rows:
    gs = [(t, float(re.search(r"gained/shares-1 " + num, l).group(1))) for t, l in rows]
    gt = [(t, float(re.search(r"grid/tets-1 " + num, l).group(1))) for t, l in rows]
    wg = max(gs, key=lambda p: abs(p[1])); wt = max(gt, key=lambda p: abs(p[1]))
    check("X4", (abs(wg[1]) < 1e-9) if double else all(v == 0. for _, v in gs),
          f"PS_MASS_AUDIT: gained/shares-1 over {len(rows)} passes: worst {wg[1]:.1e} ({wg[0]}; "
          f"{'< 1e-9' if double else 'must be exactly 0'})")
    check("X4", abs(wt[1]) < 1e-7, f"PS_MASS_AUDIT: grid/tets-1 worst {wt[1]:+.1e} ({wt[0]}; |.| < 1e-7)")
else:
    print("  SKIP X4: this binary prints no PS_MASS_AUDIT lines")
if not double:
    a, b = (tmp / "lat_cpu_float.a_den").read_bytes(), (tmp / "lat_cpu_p1.a_den").read_bytes()
    check("X5", len(a) == len(b) and a != b, f"PS_METAL_FLOAT_MASS=1 writes the float grid (box mean - 1 = "
          f"{den('lat_cpu_float').mean() - 1.0:+.2e}, the fixed-point grid {den('lat_cpu_p1').mean() - 1.0:+.2e})")
print("RESULT:", "PASS  (the mass grid is exact and thread-independent on the CPU, and after a GPU deposit)"
      if not fails else f"FAIL ({fails} check(s))")
sys.exit(1 if fails else 0)
PYEOF
rc=$?
echo ">> ps_mass_exact_check: $((SECONDS - t0)) s"
exit $rc
