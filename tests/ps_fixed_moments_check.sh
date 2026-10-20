#!/usr/bin/env bash
# Every PS-DTFE grid is an exact, order-independent sum. The MASS grid is a 64-bit fixed-point sum (GPU and
# CPU), and so are the moment grids -- velocity, the dispersion's second moment, its own
# mean and weight under --ps-volume-weighted, the velocity gradient, the moment weight, the scalar and its gradient,
# the exact deposit's multiplicity, the released tets' multiplicity: float sums would depend on the order the GPU
# atomics land in (two runs differ in the last bits of ~30-50% of the cells) and, on the CPU, on the thread count
# (thread-private copies summed in thread order). Each is a set of int64 words in a power-of-two unit
# from a bound of everything the pass deposits (ps_deposit_params.h PSFxGrid), converted to float once. Asserted on
# 32^3 crossed waves (folds everywhere: deferred tets, multi-stream cells, large gradients):
#   F1  two --ps-gpu runs with different opening chunk sizes (PS_METAL_CHUNK / PS_GPU_CHUNK 2000 and 25000: other
#       atomics share a command buffer, other orders) write byte-identical grids -- every '.a_*' output -- for the
#       sampled deposit, --ps-volume-weighted, --ps-exact-deposit (its items' per-tet weight totals are exact too)
#       and --ps-halo-release (the host's released tets). GPU builds only.
#   F2  the CPU deposit on 1 thread, 6 threads in the copy mode (DTFE_DEPOSIT_SLABS=0: one set of words, atomic
#       adds) and 6 in the slab mode writes byte-identical grids, for the same four variants. Single build only:
#       the double build keeps its double CPU sums (agreeing to ~1e-15, not bit for bit), as for the mass.
#   F3  the float sums (PS_METAL_FLOAT_MOMENTS=1, the A/B switch; PS_GPU_FLOAT_MOMENTS=1 is the CUDA/HIP
#       spelling, either works on either backend) agree with the exact ones to float rounding: peak-relative
#       difference below 1e-5 per grid, GPU and CPU (a sum of the same float shares, in another order)
#   F4  --partition 2 2 2 --max-concurrent 1 (partitions merge in index order): two GPU runs byte-identical, and
#       1 vs 6 CPU threads byte-identical (single build) -- the whole run, not only one partition's sub-grid
#   F5  the non-finite guard (ps_deposit_params.h psFixedShareOk): DTFE_TEST_NAN_SHARE=50 gives about one
#       tetrahedron in 50 a NaN mass, on the CPU and (GPU builds) in the kernel; each run says once how many shares
#       it left out, by grid (the mass grid among them), its '.a_den' has no NaN or infinity, and a run without
#       the hook says nothing of the kind. Single build only (the double build's CPU sums are doubles, no words).
# 2D (PS-DTFE-2d, when built): F2's thread independence for the sampled deposit.
# ~1-2 min (M-series Mac, 10 cores).
# Usage: tests/ps_fixed_moments_check.sh [--no-build]   (--no-build is accepted and ignored)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}"
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
OBJ=o_ps; precision_double && OBJ=o_ps_d
GPU=0                                                     # any one GPU backend stamp will do (gpu_mass_check.sh)
if [ -d "$OBJ" ] && [ ! -f "$OBJ/.gpu_mode_off" ]; then
    [ -n "$(gpu_backend_of "$OBJ")" ] && GPU=1
fi
TMP="${DTFE_TEST_TMP}/ps_fixed_moments_check"
mkdir -p "$TMP"
SNAP="$TMP/waves32.hdf5"
[ -f "$SNAP" ] || "$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$SNAP" --n 32 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
# run <out> <command...>: a non-zero exit fails the suite; a GPU run (GPUOK=1) must not fall back to the CPU
run() { local out="$1"; shift; "$@" > "$out.log" 2>&1 || { echo "FAIL: $(basename "$out") (see $out.log)"; exit 1; }
        if [ "${GPUOK:-0}" = "1" ] && grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead" "$out.log"; then
            echo "FAIL: $(basename "$out") fell back to the CPU (see $out.log)"; exit 1; fi; return 0; }
C=(--grid 32 --periodic --input 105 --MpcUnit 1 --verbose 2 --field density_a velocity_a dispersion_a gradient_a
   --avg-subsamples 2)
VARIANTS="samp vw ex hr"
opts() { case "$1" in samp) ;; vw) echo --ps-volume-weighted ;; ex) echo --ps-exact-deposit ;;
         hr) echo --ps-halo-release 50 ;; esac; }

echo "============================================================"
echo " PS-DTFE exact moment grids (${PS_BIN}; GPU: $([ $GPU = 1 ] && echo yes || echo 'no GPU build: F1 SKIP'))"
echo "============================================================"
t0=$SECONDS
unset PS_METAL_FLOAT_MOMENTS PS_GPU_FLOAT_MOMENTS PS_METAL_CHUNK PS_GPU_CHUNK
for v in $VARIANTS; do
    # shellcheck disable=SC2046
    O=($(opts "$v"))
    echo ">> $v: ${O[*]+${O[*]}} ($v)"
    if [ "$GPU" = 1 ]; then
        GPUOK=1 run "$TMP/g1_$v" env PS_METAL_CHUNK=2000 PS_GPU_CHUNK=2000 "${PS_BIN}" "$SNAP" "$TMP/g1_$v" "${C[@]}" ${O[@]+"${O[@]}"} --ps-gpu
        GPUOK=1 run "$TMP/g2_$v" env PS_METAL_CHUNK=25000 PS_GPU_CHUNK=25000 "${PS_BIN}" "$SNAP" "$TMP/g2_$v" "${C[@]}" ${O[@]+"${O[@]}"} --ps-gpu
        GPUOK=1 run "$TMP/gf_$v" env PS_METAL_FLOAT_MOMENTS=1 "${PS_BIN}" "$SNAP" "$TMP/gf_$v" "${C[@]}" ${O[@]+"${O[@]}"} --ps-gpu
    fi
    run "$TMP/c1_$v" env DTFE_DEPOSIT_THREADS=1 "${PS_BIN}" "$SNAP" "$TMP/c1_$v" "${C[@]}" ${O[@]+"${O[@]}"}
    run "$TMP/cf_$v" env DTFE_DEPOSIT_THREADS=1 PS_METAL_FLOAT_MOMENTS=1 "${PS_BIN}" "$SNAP" "$TMP/cf_$v" "${C[@]}" ${O[@]+"${O[@]}"}
    if ! precision_double; then
        run "$TMP/c6_$v" env DTFE_DEPOSIT_THREADS=6 DTFE_DEPOSIT_SLABS=0 "${PS_BIN}" "$SNAP" "$TMP/c6_$v" "${C[@]}" ${O[@]+"${O[@]}"}
        run "$TMP/s6_$v" env DTFE_DEPOSIT_THREADS=6 DTFE_DEPOSIT_SLABS=1 "${PS_BIN}" "$SNAP" "$TMP/s6_$v" "${C[@]}" ${O[@]+"${O[@]}"}
    fi
done
P=(--partition 2 2 2 --max-concurrent 1)
echo ">> F4: --partition 2 2 2 --max-concurrent 1, --ps-volume-weighted"
if [ "$GPU" = 1 ]; then
    GPUOK=1 run "$TMP/pg1" env PS_METAL_CHUNK=2000 PS_GPU_CHUNK=2000 "${PS_BIN}" "$SNAP" "$TMP/pg1" "${C[@]}" "${P[@]}" --ps-volume-weighted --ps-gpu
    GPUOK=1 run "$TMP/pg2" "${PS_BIN}" "$SNAP" "$TMP/pg2" "${C[@]}" "${P[@]}" --ps-volume-weighted --ps-gpu
fi
if ! precision_double; then
    run "$TMP/pc1" env DTFE_DEPOSIT_THREADS=1 "${PS_BIN}" "$SNAP" "$TMP/pc1" "${C[@]}" "${P[@]}" --ps-volume-weighted
    run "$TMP/pc6" env DTFE_DEPOSIT_THREADS=6 DTFE_DEPOSIT_SLABS=0 "${PS_BIN}" "$SNAP" "$TMP/pc6" "${C[@]}" "${P[@]}" --ps-volume-weighted
fi
N5=0
if ! precision_double; then
    N5=1
    echo ">> F5: the non-finite guard (DTFE_TEST_NAN_SHARE=50: about one tetrahedron in 50 with a NaN mass)"
    run "$TMP/n_cpu" env DTFE_TEST_NAN_SHARE=50 "${PS_BIN}" "$SNAP" "$TMP/n_cpu" "${C[@]}"
    if [ "$GPU" = 1 ]; then
        GPUOK=1 run "$TMP/n_gpu" env DTFE_TEST_NAN_SHARE=50 "${PS_BIN}" "$SNAP" "$TMP/n_gpu" "${C[@]}" --ps-gpu
    fi
fi
D2=0
if ! precision_double && [ -x ./PS-DTFE-2d ]; then
    D2=1
    SNAP2="$TMP/pancake2d.hdf5"
    [ -f "$SNAP2" ] || "$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$SNAP2" --n 64 --dim 2 >/dev/null 2>&1 \
        || { echo "  (no 2D data generator option: the 2D check is skipped)"; D2=0; }
    if [ "$D2" = 1 ]; then
        echo ">> 2D: 1 vs 6 CPU threads"
        C2=(--grid 32 --periodic --input 105 --MpcUnit 1 --verbose 2 --field density_a velocity_a dispersion_a gradient_a --avg-subsamples 2)
        run "$TMP/d2_1" env DTFE_DEPOSIT_THREADS=1 ./PS-DTFE-2d "$SNAP2" "$TMP/d2_1" "${C2[@]}"
        run "$TMP/d2_6" env DTFE_DEPOSIT_THREADS=6 DTFE_DEPOSIT_SLABS=0 ./PS-DTFE-2d "$SNAP2" "$TMP/d2_6" "${C2[@]}"
    fi
fi

GPU="$GPU" D2="$D2" N5="$N5" VARIANTS="$VARIANTS" "$PY" - "$TMP" <<'PYEOF'
import numpy as np, os, re, sys
from pathlib import Path
tmp = Path(sys.argv[1]); fails = 0
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
double = os.environ.get("DTFE_TEST_PRECISION") == "double"
gpu = os.environ.get("GPU") == "1"
def check(tag, ok, msg):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {tag} {msg}")
def grids(name):
    return sorted(p.name[len(name) + 1:] for p in tmp.glob(name + ".*")
                  if not p.name.endswith((".log", ".runlog", ".json", ".txt")))
def same(a, b, what, tag):
    ea, eb = grids(a), grids(b)
    if not ea or ea != eb:
        check(tag, False, f"{what}: outputs {ea} vs {eb}")
        return
    diff = [e for e in ea if (tmp / f"{a}.{e}").read_bytes() != (tmp / f"{b}.{e}").read_bytes()]
    check(tag, not diff, f"{what}: {len(ea)} grids byte-identical" + (f" -- DIFFER: {diff}" if diff else ""))
def close(a, b, what, tag, tol=1e-5):
    worst, wname = 0., ""
    for e in grids(a):
        if not e.startswith("a_") or e in ("a_streams", "a_hidden_streams"): continue
        x = np.fromfile(tmp / f"{a}.{e}", dtype=REAL).astype(np.float64)
        y = np.fromfile(tmp / f"{b}.{e}", dtype=REAL).astype(np.float64)
        r = float(np.abs(x - y).max() / max(np.abs(x).max(), 1e-300))
        if r > worst: worst, wname = r, e
    check(tag, worst < tol, f"{what}: worst grid {wname or '-'} peak-rel {worst:.1e} (< {tol:g})")
for v in os.environ["VARIANTS"].split():
    if gpu:
        same(f"g1_{v}", f"g2_{v}", f"{v}: two GPU runs (chunks 2000 / 25000)", "F1")
        close(f"g1_{v}", f"gf_{v}", f"{v}: GPU exact vs the float sums (PS_METAL_FLOAT_MOMENTS=1)", "F3")
    if not double:
        same(f"c1_{v}", f"c6_{v}", f"{v}: CPU 1 thread vs 6 (copy mode)", "F2")
        same(f"c1_{v}", f"s6_{v}", f"{v}: CPU 1 thread vs 6 (slab mode)", "F2")
    close(f"c1_{v}", f"cf_{v}", f"{v}: CPU exact vs the float sums (PS_METAL_FLOAT_MOMENTS=1)", "F3")
if gpu:
    same("pg1", "pg2", "--partition 2 2 2, volume-weighted: two GPU runs", "F4")
if not double:
    same("pc1", "pc6", "--partition 2 2 2, volume-weighted: CPU 1 thread vs 6", "F4")
if os.environ.get("D2") == "1":
    same("d2_1", "d2_6", "2D: CPU 1 thread vs 6 (copy mode)", "F2")
DROP = re.compile(r"(\d+) share\(s\) of the exact fixed-point sums were not finite.*?\((.*?)\)\.")
if os.environ.get("N5") == "1":
    for run in ["n_cpu"] + (["n_gpu"] if gpu else []):
        log = (tmp / f"{run}.log").read_text(errors="replace")
        m = DROP.findall(log)
        mass = re.search(r"mass \(density\) grid (\d+)", m[-1][1]) if m else None
        check("F5", len(m) == 1 and int(m[0][0]) > 0 and mass is not None and int(mass.group(1)) > 0,
              f"{run}: the run reports its dropped shares once, by grid: "
              + (f"{m[0][0]} ({m[0][1]})" if m else "NO such warning"))
        den = np.fromfile(tmp / f"{run}.a_den", dtype=REAL)
        check("F5", den.size > 0 and bool(np.isfinite(den).all()), f"{run}: '.a_den' finite everywhere ({den.size} cells)")
    clean = [r for r in (["c1_samp"] + (["g1_samp"] if gpu else [])) if DROP.search((tmp / f"{r}.log").read_text(errors="replace"))]
    check("F5", not clean, "runs without the hook report no dropped share" + (f" -- they do: {clean}" if clean else ""))
print("RESULT:", "PASS  (every grid of a pass is an exact sum: run- and thread-count independent)"
      if not fails else f"FAIL ({fails} check(s))")
sys.exit(1 if fails else 0)
PYEOF
rc=$?
echo "(${SECONDS}s since start; $((SECONDS - t0))s of runs and checks)"
exit $rc
