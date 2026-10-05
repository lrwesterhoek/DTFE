#!/bin/bash
# --ps-window: a window run EQUALS the full run's window cells. Periodic crossed waves, full run at
# grid 64 vs window runs of a sub-box -- one straddling the periodic seam -- for the sampled deposit
# (nSub 2), the exact deposit, --ps-vertex-mass --ps-volume-weighted --ps-caustics, --ps-halo-release, and a
# --partition 2 2 2 split (held to the partitioned full run: partitions differ from the single
# triangulation by summation order). Then the launcher's zoom shape: an anisotropic 256x256x16 grid
# windowed to one coarse slice, exact deposit, with the wall times of both runs.
# Contracts: the SAMPLED deposit counts a straddling tet's samples over its whole bounding box, so
# the CPU window is identical bytes (every field, masks and counts included). The EXACT deposit
# clips only the window's cells and normalizes each tet by its whole weight, so its moments agree
# to float rounding (the counts, masks and multiplicity are still identical bytes); the GPU takes
# the usual parity class. With '--tessellation-cache', a window run skips the partitions whose occupancy
# maps (written by a composite server or by an earlier batch run) never reach the window: identical
# bytes to the uncached run, the skip count reported. --no-build is accepted and ignored.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
precision_require "$PS_BIN"
[ -x "$PS_BIN" ] || { echo "FAIL: $PS_BIN is not built"; exit 1; }
OBJ_PS=o_ps; precision_double && OBJ_PS=o_ps_d     # the build whose GPU mode counts
GPU=0; [ -d $OBJ_PS ] && [ ! -f $OBJ_PS/.gpu_mode_off ] && ls $OBJ_PS/.gpu_mode_* >/dev/null 2>&1 && GPU=1
TMP="$ROOT/tests/tmp/window_check"
mkdir -p "$TMP"
SNAP="$TMP/waves.hdf5"
[ -f "$SNAP" ] || "$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 32 --box 100 --seed 9 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
G=64
BASE=(--periodic --MpcUnit 1 --field density velocity dispersion gradient --ps-caustics --max-concurrent 1)
# window A: cells [8,24) x [40,52) x [30,31) (dx = 100/64 = 1.5625 Mpc); window B straddles the seam along x:
# cells [56,64)+[0,8), i.e. x in [87.5, 112.5) -- given as a box beyond the period, which the option wraps
WA=(--ps-window 12.5 37.5 62.5 81.25 46.875 48.4375)
WB=(--ps-window 87.5 112.5 62.5 81.25 46.875 48.4375)
run() { local out="$1"; shift; local t0=$SECONDS; "$@" > "$out.log" 2>&1 || { echo "FAIL: $* (see $out.log)"; exit 1; }; echo "   $(basename "$out"): $((SECONDS - t0)) s"; }
for mode in sampled exact; do
    EXTRA=(--avg-subsamples 2); [ $mode = exact ] && EXTRA=(--ps-exact-deposit)
    echo ">> $mode deposit: full run, window, seam window, --partition 2 2 2 (full and window), vertex mass + volume weighted"
    run "$TMP/${mode}_full"     "$PS_BIN" "$SNAP" "$TMP/${mode}_full"     --grid $G "${BASE[@]}" "${EXTRA[@]}"
    run "$TMP/${mode}_win"      "$PS_BIN" "$SNAP" "$TMP/${mode}_win"      --grid $G "${BASE[@]}" "${EXTRA[@]}" "${WA[@]}"
    run "$TMP/${mode}_seam"     "$PS_BIN" "$SNAP" "$TMP/${mode}_seam"     --grid $G "${BASE[@]}" "${EXTRA[@]}" "${WB[@]}"
    run "$TMP/${mode}_fullpart" "$PS_BIN" "$SNAP" "$TMP/${mode}_fullpart" --grid $G "${BASE[@]}" "${EXTRA[@]}" --partition 2 2 2
    run "$TMP/${mode}_part"     "$PS_BIN" "$SNAP" "$TMP/${mode}_part"     --grid $G "${BASE[@]}" "${EXTRA[@]}" --partition 2 2 2 "${WA[@]}"
    run "$TMP/${mode}_fullvm"   "$PS_BIN" "$SNAP" "$TMP/${mode}_fullvm"   --grid $G "${BASE[@]}" "${EXTRA[@]}" --ps-vertex-mass --ps-volume-weighted
    run "$TMP/${mode}_winvm"    "$PS_BIN" "$SNAP" "$TMP/${mode}_winvm"    --grid $G "${BASE[@]}" "${EXTRA[@]}" --ps-vertex-mass --ps-volume-weighted "${WA[@]}"
    # --ps-halo-release with a window: a released tetrahedron deposits whole at its centroid cell,
    # which the window keeps or drops like any cell (D=2 releases plenty on these waves)
    run "$TMP/${mode}_fullhr"   "$PS_BIN" "$SNAP" "$TMP/${mode}_fullhr"   --grid $G "${BASE[@]}" "${EXTRA[@]}" --ps-halo-release 2
    run "$TMP/${mode}_winhr"    "$PS_BIN" "$SNAP" "$TMP/${mode}_winhr"    --grid $G "${BASE[@]}" "${EXTRA[@]}" --ps-halo-release 2 "${WA[@]}"
    nrel=$(sed $'s/\x1b\[[0-9;]*m//g' "$TMP/${mode}_winhr.log" | grep -o 'released [0-9]* halo-interior' | awk '{s+=$2} END {print s+0}')
    [ "$nrel" -gt 0 ] && echo "  PASS $mode: the halo-release window run released $nrel tetrahedra" \
        || { echo "  FAIL $mode: the halo-release window run released none (the pair is not exercised)"; exit 1; }
    if [ $GPU = 1 ]; then
        run "$TMP/${mode}_wingpu" "$PS_BIN" "$SNAP" "$TMP/${mode}_wingpu" --grid $G "${BASE[@]}" "${EXTRA[@]}" "${WA[@]}" --ps-gpu
        grep -q "using the CPU deposit\|deposit unavailable" "$TMP/${mode}_wingpu.log" && { echo "FAIL: the GPU window run fell back to the CPU"; exit 1; }
        run "$TMP/${mode}_seamgpu" "$PS_BIN" "$SNAP" "$TMP/${mode}_seamgpu" --grid $G "${BASE[@]}" "${EXTRA[@]}" "${WB[@]}" --ps-gpu
        run "$TMP/${mode}_winhrgpu" "$PS_BIN" "$SNAP" "$TMP/${mode}_winhrgpu" --grid $G "${BASE[@]}" "${EXTRA[@]}" --ps-halo-release 2 "${WA[@]}" --ps-gpu
        # --ps-vertex-mass on the GPU computes the vertex degrees lazily for the window's tetrahedra:
        # the self-check compares every one with the full pass
        PS_DEBUG_DEGREES=1 "$PS_BIN" "$SNAP" "$TMP/${mode}_degchk" --grid $G "${BASE[@]}" "${EXTRA[@]}" "${WA[@]}" \
            --ps-gpu --ps-vertex-mass --ps-volume-weighted > "$TMP/${mode}_degchk.log" 2>&1 \
            || { echo "FAIL: the lazy-degree self-check failed (see $TMP/${mode}_degchk.log)"; exit 1; }
        nchk=$(grep -c "differ" "$TMP/${mode}_degchk.log"); nbad=$(grep "differ" "$TMP/${mode}_degchk.log" | grep -vc ", 0 differ")
        [ "$nchk" -ge 1 ] && [ "$nbad" = 0 ] && echo "  PASS $mode: the GPU window's lazy vertex degrees equal the full pass ($nchk checks)" \
            || { echo "  FAIL $mode: lazy vertex degrees ($nchk checks, $nbad differ)"; exit 1; }
    fi
done
echo ">> the launcher's zoom shape: grid 256x256x16, exact deposit, window = cells [64,192)x[64,192)x[8,9) (one coarse slice)"
run "$TMP/zoom_full" "$PS_BIN" "$SNAP" "$TMP/zoom_full" --grid 256 256 16 "${BASE[@]}" --ps-exact-deposit
run "$TMP/zoom_win"  "$PS_BIN" "$SNAP" "$TMP/zoom_win"  --grid 256 256 16 "${BASE[@]}" --ps-exact-deposit --ps-window 25 75 25 75 50 56.25
if [ $GPU = 1 ]; then
    run "$TMP/zoom_wingpu" "$PS_BIN" "$SNAP" "$TMP/zoom_wingpu" --grid 256 256 16 "${BASE[@]}" --ps-exact-deposit --ps-window 25 75 25 75 50 56.25 --ps-gpu
fi
echo ">> the tessellation cache + occupancy maps: a window run skips the partitions that never reach the window"
# the cache must live on a local, non-synced volume (the option refuses iCloud Drive, where the repo is)
CACHE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/dtfe-window-check.XXXXXX")" || { echo "FAIL: no temp dir"; exit 1; }
trap 'rm -rf "$CACHE_ROOT"' EXIT
CACHE="$CACHE_ROOT/serve"; mkdir -p "$CACHE"
# a composite server builds and caches its 2x2x2 partitions (tessellation + occupancy map each), then
# exits when stdin closes -- exactly what the launcher's query server leaves on disk
"$PS_BIN" "$SNAP" "$TMP/serve" --serve --periodic --MpcUnit 1 --partition 2 2 2 --tessellation-cache "$CACHE" \
    --per-stream --verbose 1 < /dev/null > "$TMP/serve.log" 2>&1 || { echo "FAIL: the composite server did not build its cache (see $TMP/serve.log)"; exit 1; }
nt=$(ls "$CACHE"/*.tess 2>/dev/null | wc -l | tr -d ' '); no=$(ls "$CACHE"/*.tess.occ 2>/dev/null | wc -l | tr -d ' ')
[ "$nt" = 8 ] && [ "$no" = 8 ] && echo "  PASS the server cached 8 tessellations + 8 occupancy maps" || { echo "  FAIL the server cached $nt tessellations, $no maps"; exit 1; }
ng=$(ls "$CACHE"/globals_*.txt 2>/dev/null | wc -l | tr -d ' ')
[ "$ng" = 1 ] && echo "  PASS ... and the snapshot's globals record (particle count, mean density, initial-position box)" \
    || { echo "  FAIL the server left $ng globals records"; exit 1; }
PART=(--partition 2 2 2)
skipped() { sed -n 's/.*--ps-window: \([0-9]*\) of \([0-9]*\) partitions never reach.*/\1/p' "$1" | head -1; }
for mode in sampled exact; do
    EXTRA=(--avg-subsamples 2); [ $mode = exact ] && EXTRA=(--ps-exact-deposit)
    for w in A B; do
        WIN=("${WA[@]}"); [ $w = B ] && WIN=("${WB[@]}")
        run "$TMP/${mode}_${w}_nocache" "$PS_BIN" "$SNAP" "$TMP/${mode}_${w}_nocache" --grid $G "${BASE[@]}" "${EXTRA[@]}" "${PART[@]}" "${WIN[@]}"
        run "$TMP/${mode}_${w}_cached"  "$PS_BIN" "$SNAP" "$TMP/${mode}_${w}_cached"  --grid $G "${BASE[@]}" "${EXTRA[@]}" "${PART[@]}" "${WIN[@]}" --tessellation-cache "$CACHE"
        grep -q "the snapshot is not read" "$TMP/${mode}_${w}_cached.log" && echo "  PASS $mode window $w: every needed partition cached, the snapshot was not read" \
            || { echo "  FAIL $mode window $w: the run read the snapshot although its record and partitions are cached"; exit 1; }
        k=$(skipped "$TMP/${mode}_${w}_cached.log")
        [ -n "$k" ] && [ "$k" -gt 0 ] && echo "  PASS $mode window $w: $k of 8 partitions skipped by the server's maps" \
            || { echo "  FAIL $mode window $w: no partition skipped ('$k')"; exit 1; }
        grep -q "Reused the cached tessellation" "$TMP/${mode}_${w}_cached.log" && echo "  PASS $mode window $w: the processed partitions were loaded from the cache" \
            || { echo "  FAIL $mode window $w: no cache hit"; exit 1; }
        for f in den vel velDisp velDispTensor velGrad streams hidden_streams caustic causticClass; do
            cmp -s "$TMP/${mode}_${w}_nocache.$f" "$TMP/${mode}_${w}_cached.$f" || { echo "  FAIL $mode window $w: .$f differs between the cached (skipping) run and the uncached one"; exit 1; }
        done
        echo "  PASS $mode window $w: every grid identical bytes to the uncached run"
    done
done
# maps written by a BATCH run (no server): the first window run builds, caches and maps; the second skips
CACHE2="$CACHE_ROOT/batch"; mkdir -p "$CACHE2"
run "$TMP/batch1" "$PS_BIN" "$SNAP" "$TMP/batch1" --grid $G "${BASE[@]}" --avg-subsamples 2 "${PART[@]}" "${WA[@]}" --tessellation-cache "$CACHE2"
grep -q "8 without a map yet" "$TMP/batch1.log" && echo "  PASS the first batch run saw no maps" || { echo "  FAIL the first batch run's map count (see $TMP/batch1.log)"; exit 1; }
no=$(ls "$CACHE2"/*.tess.occ 2>/dev/null | wc -l | tr -d ' ')
[ "$no" = 8 ] && echo "  PASS the batch run left 8 occupancy maps beside its 8 tessellations" || { echo "  FAIL the batch run left $no maps"; exit 1; }
run "$TMP/batch2" "$PS_BIN" "$SNAP" "$TMP/batch2" --grid $G "${BASE[@]}" --avg-subsamples 2 "${PART[@]}" "${WA[@]}" --tessellation-cache "$CACHE2"
k1=$(skipped "$TMP/sampled_A_cached.log"); k2=$(skipped "$TMP/batch2.log")
[ "$k2" = "$k1" ] && echo "  PASS the second batch run skips the same $k2 partitions as with the server's maps" || { echo "  FAIL skips: batch maps $k2, server maps $k1"; exit 1; }
for f in den vel velDisp streams caustic; do
    cmp -s "$TMP/sampled_A_nocache.$f" "$TMP/batch2.$f" || { echo "  FAIL .$f differs after skipping by the batch maps"; exit 1; }
done
echo "  PASS ... with identical grids"
# the launcher's own flags: vertex masses (per partition, from its padded tessellation) and volume weights --
# skipping a partition must not touch them either
VM=(--ps-exact-deposit --ps-vertex-mass --ps-volume-weighted "${PART[@]}" "${WA[@]}")
run "$TMP/vm_nocache" "$PS_BIN" "$SNAP" "$TMP/vm_nocache" --grid $G "${BASE[@]}" "${VM[@]}"
run "$TMP/vm_cached"  "$PS_BIN" "$SNAP" "$TMP/vm_cached"  --grid $G "${BASE[@]}" "${VM[@]}" --tessellation-cache "$CACHE"
for f in den vel velDisp streams hidden_streams caustic; do
    cmp -s "$TMP/vm_nocache.$f" "$TMP/vm_cached.$f" || { echo "  FAIL .$f differs with vertex masses + volume weights when partitions are skipped"; exit 1; }
done
echo "  PASS with --ps-vertex-mass --ps-volume-weighted: $(skipped "$TMP/vm_cached.log") skipped, identical grids"
# the globals record without the tessellations: the run starts without the particles, finds the needed
# partitions missing, reads them after all and rebuilds those partitions -- the same bytes
LATE="$CACHE_ROOT/late"; mkdir -p "$LATE"; cp "$CACHE"/globals_*.txt "$CACHE"/*.occ "$LATE"/
run "$TMP/late" "$PS_BIN" "$SNAP" "$TMP/late" --grid $G "${BASE[@]}" --avg-subsamples 2 "${PART[@]}" "${WA[@]}" --tessellation-cache "$LATE"
grep -q "reading the particles after all" "$TMP/late.log" && echo "  PASS needed partitions missing from the cache: the particles are read after all" \
    || { echo "  FAIL the late read did not happen (see $TMP/late.log)"; exit 1; }
for f in den vel velDisp streams hidden_streams caustic causticClass; do
    cmp -s "$TMP/sampled_A_nocache.$f" "$TMP/late.$f" || { echo "  FAIL .$f differs after the late read"; exit 1; }
done
echo "  PASS ... with identical grids"
# no record: the ordinary read (the cache without its globals record)
NOREC="$CACHE_ROOT/norec"; mkdir -p "$NOREC"; cp "$CACHE"/*.tess "$CACHE"/*.occ "$NOREC"/
run "$TMP/norec" "$PS_BIN" "$SNAP" "$TMP/norec" --grid $G "${BASE[@]}" --avg-subsamples 2 "${PART[@]}" "${WA[@]}" --tessellation-cache "$NOREC"
if grep -q "Particles deferred" "$TMP/norec.log"; then echo "  FAIL deferred the particles without a record"; exit 1; fi
grep -q "Reading GADGET" "$TMP/norec.log" && cmp -s "$TMP/sampled_A_nocache.den" "$TMP/norec.den" \
    && echo "  PASS without a globals record the snapshot is read as before, the same bytes" \
    || { echo "  FAIL the run without a record (see $TMP/norec.log)"; exit 1; }
# a cached partition is LOADED, so its particles are never selected: a cache file whose header still
# matches but whose body is damaged must stop the run with the reason -- never yield an empty partition
BAD="$CACHE_ROOT/damaged"; cp -R "$CACHE" "$BAD"
for f in "$BAD"/*.tess; do head -c 4096 "$f" > "$f.cut" && mv "$f.cut" "$f"; done
if "$PS_BIN" "$SNAP" "$TMP/bad" --grid $G "${BASE[@]}" --avg-subsamples 2 "${PART[@]}" "${WA[@]}" --tessellation-cache "$BAD" > "$TMP/bad.log" 2>&1; then
    echo "  FAIL a damaged cached tessellation was accepted (see $TMP/bad.log)"; exit 1
fi
grep -q "could not be read back" "$TMP/bad.log" && echo "  PASS a damaged cached tessellation stops the run with the reason" \
    || { echo "  FAIL the damaged-cache run failed without naming the cause (see $TMP/bad.log)"; exit 1; }
# a window across the whole box in the plane and a third of it in depth reaches every partition
run "$TMP/wide" "$PS_BIN" "$SNAP" "$TMP/wide" --grid $G "${BASE[@]}" --avg-subsamples 2 "${PART[@]}" --tessellation-cache "$CACHE" \
    --ps-window 0 100 0 100 33.0 67.0
[ "$(skipped "$TMP/wide.log")" = 0 ] && echo "  PASS a window across the box skips no partition" || { echo "  FAIL the wide window skipped $(skipped "$TMP/wide.log")"; exit 1; }
echo ">> the auto-tuner sizes a window run by its window: a 65536x65536x64 virtual grid, one 128^2 window"
run "$TMP/huge" "$PS_BIN" "$SNAP" "$TMP/huge" --grid 65536 65536 64 "${BASE[@]}" --ps-exact-deposit "${PART[@]}" \
    --ps-window 50 50.1953125 50 50.1953125 50 51.5625
if grep -q "does not fit in memory" "$TMP/huge.log"; then echo "  FAIL the auto-tuner judged the virtual grid, not the window"; exit 1; fi
[ "$(wc -c < "$TMP/huge.den" | tr -d ' ')" = $((128*128*REAL_BYTES)) ] && echo "  PASS ran within budget and wrote the 128^2 window" \
    || { echo "  FAIL the huge-grid window's density has the wrong size"; exit 1; }

if [ $GPU = 1 ]; then
    echo ">> the GPU exact deposit on a fine virtual grid (32768^2 x 64: 3 kpc cells) agrees with the CPU"
    # the GPU clips each piece in a frame near the piece (exact_clip.metal.inc); in the tetrahedron's
    # vertex-0 frame a 3 kpc piece Mpc away lost its volume to float cancellation (20% per cell here)
    FW=(--grid 32768 32768 64 "${BASE[@]}" --ps-exact-deposit "${PART[@]}" --ps-window 40 40.390625 40 40.390625 50 51.5625)
    run "$TMP/fine_cpu" "$PS_BIN" "$SNAP" "$TMP/fine_cpu" "${FW[@]}"
    run "$TMP/fine_gpu" "$PS_BIN" "$SNAP" "$TMP/fine_gpu" "${FW[@]}" --ps-gpu
    "$PY" - "$TMP" <<'PYEOF2' || exit 1
import numpy as np, sys
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
t = sys.argv[1]; bad = 0
for f in ("den", "vel", "velDisp"):
    a = np.fromfile(f"{t}/fine_cpu.{f}", REAL).astype(float); b = np.fromfile(f"{t}/fine_gpu.{f}", REAL).astype(float)
    p = float(np.abs(a - b).max() / np.abs(a).max())
    print(f"  {'PASS' if p < 1e-4 else 'FAIL'} fine-grid GPU {f:8s} peak-rel={p:.2e} (< 1e-4)"); bad += p >= 1e-4
sys.exit(1 if bad else 0)
PYEOF2
fi

echo ">> --ps-window with --ps-linear-deposit is refused"
if "$PS_BIN" "$SNAP" "$TMP/lin" --grid $G "${BASE[@]}" --ps-exact-deposit --ps-linear-deposit "${WA[@]}" > "$TMP/lin.log" 2>&1; then
    echo "  FAIL: --ps-window + --ps-linear-deposit ran"; exit 1
else
    grep -q "ps-linear-deposit" "$TMP/lin.log" && echo "  PASS refused with the reason" || { echo "  FAIL: refused without naming --ps-linear-deposit"; exit 1; }
fi
"$PY" - "$TMP" "$G" "$GPU" <<'PYEOF'
import numpy as np, sys
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
from pathlib import Path
tmp = Path(sys.argv[1]); G = int(sys.argv[2]); gpu = sys.argv[3] == "1"
fails = 0
FIELDS = (("den", 1), ("vel", 3), ("velDisp", 1), ("velDispTensor", 6), ("velGrad", 9), ("streams", 1),
          ("hidden_streams", 1), ("caustic", 1), ("causticClass", 1))
EXACT_ONLY = ("streams", "hidden_streams", "caustic", "causticClass", "tetTouch")   # identical bytes on the CPU; the GPU's
# exact multiplicity ('.streams', a float volume share) takes the parity class and its '.tetTouch' is not compared
def cube(prefix, name, ncomp, shape):
    p = tmp / f"{prefix}.{name}"
    n = int(np.prod(shape)) * ncomp
    if not p.is_file() or p.stat().st_size != n * REAL.itemsize:
        return None
    return np.fromfile(p, dtype=REAL).reshape(tuple(shape) + ((ncomp,) if ncomp > 1 else ()))
def window_of(full, lo, dims):
    """the full grid's cells of the window at lo (wrapping each axis)"""
    idx = [np.arange(lo[d], lo[d] + dims[d]) % full.shape[d] for d in range(3)]
    return full[np.ix_(*idx)]
def compare(tag, fullp, winp, lo, dims, shape, cls):
    """cls: 'bytes' (identical), 'tight' (exact CPU: moments to float rounding), 'gpu' (the parity class)"""
    global fails
    fields = FIELDS + ((("tetTouch", 1),) if "exact" in tag or "zoom" in tag else ())
    for name, nc in fields:
        full = cube(fullp, name, nc, shape)
        b = cube(winp, name, nc, dims)
        if full is None or b is None:
            print(f"  FAIL {tag} {name}: missing or wrongly sized ({'full' if full is None else 'window'})"); fails += 1; continue
        a = window_of(full, lo, dims)
        if cls == "gpu" and name == "tetTouch":
            continue            # the raw tet-touch count: a float clip can find or lose a sliver piece
        if cls == "bytes" or (name in EXACT_ONLY and not (cls == "gpu" and name == "streams")):
            ok = np.array_equal(a, b)
            print(f"  {'PASS' if ok else 'FAIL'} {tag} {name:15s} identical bytes" + ("" if ok else f" (max |diff| {np.abs(a-b).max():.3e})"))
        else:
            sc = max(float(np.abs(a).max()), 1e-30); d = np.abs(a - b)
            p = d.max() / sc; fr = float(np.mean(d > 1e-3 * sc))
            if cls == "tight":
                tol = 1e-3 if name in ("velDisp", "velDispTensor") else 1e-5   # the dispersion cancels E[v^2]-E[v]^2
                ok = p < tol
                print(f"  {'PASS' if ok else 'FAIL'} {tag} {name:15s} peak-rel={p:.2e} (< {tol:g})")
            else:
                ok = p < 5e-2 or fr < (2e-3 if name == "velGrad" else 1e-3)
                print(f"  {'PASS' if ok else 'FAIL'} {tag} {name:15s} peak-rel={p:.2e} frac>1e-3={fr:.2e}")
        fails += not ok
LO, DIMS, SHAPE = (8, 40, 30), (16, 12, 1), (G, G, G)
SEAM = (56, 40, 30)
for mode, cls in (("sampled", "bytes"), ("exact", "tight")):
    compare(f"{mode} window == full",                        f"{mode}_full",     f"{mode}_win",   LO,   DIMS, SHAPE, cls)
    compare(f"{mode} seam window == full (wrapped cells)",   f"{mode}_full",     f"{mode}_seam",  SEAM, DIMS, SHAPE, cls)
    compare(f"{mode} window, partition 2 2 2 == full, partition 2 2 2", f"{mode}_fullpart", f"{mode}_part", LO, DIMS, SHAPE, cls)
    compare(f"{mode} window (vertex mass, volume weighted) == full", f"{mode}_fullvm", f"{mode}_winvm", LO, DIMS, SHAPE, cls)
    compare(f"{mode} window (halo release D=2) == full",   f"{mode}_fullhr",   f"{mode}_winhr", LO, DIMS, SHAPE, cls)
    if gpu:
        compare(f"{mode} GPU window ~ CPU full",             f"{mode}_full",     f"{mode}_wingpu",  LO,   DIMS, SHAPE, "gpu")
        compare(f"{mode} GPU seam window ~ CPU full",        f"{mode}_full",     f"{mode}_seamgpu", SEAM, DIMS, SHAPE, "gpu")
        compare(f"{mode} GPU halo-release window ~ CPU full", f"{mode}_fullhr",  f"{mode}_winhrgpu", LO, DIMS, SHAPE, "gpu")
ZLO, ZDIMS, ZSHAPE = (64, 64, 8), (128, 128, 1), (256, 256, 16)
compare("zoom-shaped exact window == full", "zoom_full", "zoom_win", ZLO, ZDIMS, ZSHAPE, "tight")
if gpu:
    compare("zoom-shaped exact GPU window ~ CPU full", "zoom_full", "zoom_wingpu", ZLO, ZDIMS, ZSHAPE, "gpu")
print("ps_window_check:", "ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
PYEOF
