#!/usr/bin/env bash
# Interlacing ('--interlace', src/interlacing.cc) against an exact answer: tests/interlacing_unit.cpp
# compiled against the source in 3D and 2D, single and double precision: the function is checked on grids
# holding exact samples of a band-limited field (see the .cpp). Then end to end (2026-10-11), when the binaries
# are built: the second pass must grid the box moved by half a cell. A periodic run's pass 2 equals a plain
# run on the particles moved by -dx/2 (what main.cpp now runs), so '--interlace' must equal interlace(plain,
# plain of the moved particles) -- for TSC, the DTFE estimator and PS-DTFE; and a non-periodic run without a
# region is refused. Until 2026-10-11 a run without '--region' gridded
# the SAME cells twice (the setup reset the shifted region to the box): interlace(plain, plain), a
# low-passed copy moved half a cell.
# Usage: tests/interlacing_check.sh [--no-build]   (the flag is accepted for the battery; the end-to-end part uses the built binaries)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
CXX="${CXX:-$(command -v clang++ || command -v g++)}"
FLAGS=( -std=c++17 -O2 -Isrc )
if pkg-config --exists fftw3 fftw3f 2>/dev/null; then
    read -r -a PC <<<"$(pkg-config --cflags --libs fftw3 fftw3f)"; FLAGS+=( "${PC[@]}" )
else
    for p in /opt/homebrew /usr/local; do [ -d "$p/include" ] && FLAGS+=( -I"$p/include" -L"$p/lib" ); done
    FLAGS+=( -lfftw3 -lfftw3f )
fi
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"; mkdir -p "$TMP"
echo "============================================================"
echo " interlacing check (exact samples of a band-limited field)"
echo "============================================================"
fails=0
for dim in 3 2; do
    for prec in single double; do
        defs=( -DNO_DIM=$dim ); [ "$prec" = double ] && defs+=( -DDOUBLE )
        exe="$TMP/interlacing_unit_${dim}d_$prec"
        if ! "$CXX" "${defs[@]}" tests/interlacing_unit.cpp src/interlacing.cc "${FLAGS[@]}" -o "$exe" 2> "$exe.log"; then
            echo "   FAIL  ${dim}D $prec: does not compile ($exe.log)"; fails=$((fails+1)); continue
        fi
        printf '   %-7s' "$prec"; "$exe" || fails=$((fails+1))
    done
done

PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
if [ ! -x ./DTFE ] || ! "$PY" -c 'import numpy' 2>/dev/null; then
    echo "   SKIP  end to end: ./DTFE or numpy missing"
else
    E2E="$TMP/interlace_e2e"; rm -rf "$E2E"; mkdir -p "$E2E"
    BINS="./DTFE"; [ -x ./PS-DTFE ] && "$PY" -c 'import h5py' 2>/dev/null && BINS="./DTFE ./PS-DTFE"
    for bin in $BINS; do
        "$PY" - "$bin" "$E2E" <<'PYEOF'
import os, subprocess, sys
import numpy as np
binary, tmp = sys.argv[1], sys.argv[2]
L, n = 100.0, 16
dx = L / n
common = ["--grid", str(n), "--periodic", "--MpcUnit", "1"]

def run(name, infile, args):
    out = os.path.join(tmp, name)
    r = subprocess.run([binary, infile, out] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   FAIL  {name}: exit {r.returncode}: {(r.stdout + r.stderr).strip().splitlines()[-1:]}")
        sys.exit(1)
    ext = ".den" if os.path.exists(out + ".den") else ".a_den"
    return np.fromfile(out + ext, dtype=np.float32).reshape(n, n, n).astype(np.float64)

def interlace(a, b):
    # src/interlacing.cc: F = (F_a + F_b exp(-i pi sum_d m_d/n_d)) / 2, m signed on the full axes
    fa, fb = np.fft.rfftn(a), np.fft.rfftn(b)
    m = np.fft.fftfreq(n) * n
    mz = np.fft.rfftfreq(n) * n
    s = m[:, None, None] / n + m[None, :, None] / n + mz[None, None, :] / n
    return np.fft.irfftn(0.5 * (fa + fb * np.exp(-1j * np.pi * s)), s=a.shape, axes=(0, 1, 2))

cases = []
if binary.endswith("PS-DTFE"):
    import h5py
    snap = os.path.join(tmp, "pan.hdf5")
    subprocess.run([sys.executable, "tests/generate_ps_test_data.py", "--out", snap, "--n", "24", "--box", str(L),
                    "--amplitude-factor", "0.6", "--seed", "5"], check=True, capture_output=True)
    moved = os.path.join(tmp, "pan_moved.hdf5")
    with h5py.File(snap, "r") as f, h5py.File(moved, "w") as g:
        for k in f:
            f.copy(k, g)
        for k in f.attrs:
            g.attrs[k] = f.attrs[k]
        x = g["PartType1"]["Coordinates"][...]        # the Eulerian map moved; the Lagrangian lattice kept
        g["PartType1"]["Coordinates"][...] = np.mod(x - dx / 2, L)
    cases.append(("ps", snap, moved, ["--field", "density_a", "--partition", "1", "1", "1"], 1e-4))
else:
    rng = np.random.default_rng(11)
    pos = rng.uniform(0, L, (6000, 3))
    for tag, p in (("p", pos), ("moved", np.mod(pos - dx / 2, L))):
        with open(os.path.join(tmp, f"pts_{tag}.txt"), "w") as f:
            f.write(f"{len(p)}\n0 {L} 0 {L} 0 {L}\n")
            np.savetxt(f, np.c_[p, np.ones(len(p))], fmt="%.7f")
    pin, pmoved = os.path.join(tmp, "pts_p.txt"), os.path.join(tmp, "pts_moved.txt")
    cases.append(("tsc", pin, pmoved, ["-i", "111", "-f", "density", "--TSC"], 1e-5))
    # the DTFE field at the cell centres (density_a's Monte Carlo points are not translation-invariant)
    cases.append(("dtfe", pin, pmoved, ["-i", "111", "-f", "density", "--partition", "1", "1", "1"], 1e-4))

bad = 0
for tag, fin, fmoved, args, tol in cases:
    a = run(f"{tag}_plain", fin, common + args)
    a2 = run(f"{tag}_moved", fmoved, common + args)
    b = run(f"{tag}_interlaced", fin, common + args + ["--interlace"])
    want, smoothed = interlace(a, a2), interlace(a, a)
    err = np.abs(b - want).max() / np.abs(want).max()
    gap = np.abs(want - smoothed).max() / np.abs(want).max()
    ok = err < tol and gap > 1e-2
    bad += not ok
    print(f"   {'PASS' if ok else 'FAIL'}  {os.path.basename(binary)} {tag}: --interlace vs interlace(plain, "
          f"plain of the particles moved -dx/2) {err:.1e} (tol {tol:.0e}); a same-cells pass 2 is {gap:.1e} off")
sys.exit(1 if bad else 0)
PYEOF
        [ $? -eq 0 ] || fails=$((fails+1))
    done
    # without '--periodic' the moved region leaves the box: refused before the first pass
    printf '8\n0 1 0 1 0 1\n' > "$E2E/np.txt"
    for x in 0.1 0.6; do for y in 0.1 0.6; do for z in 0.1 0.6; do echo "$x $y $z 1" >> "$E2E/np.txt"; done; done; done
    if ./DTFE "$E2E/np.txt" "$E2E/np" -i 111 --grid 4 --MpcUnit 1 -f density --TSC --interlace > "$E2E/np.log" 2>&1; then
        echo "   FAIL  a non-periodic --interlace run without a region is not refused"; fails=$((fails+1))
    elif grep -q "needs a '--region' or '--regionMpc'" "$E2E/np.log"; then
        echo "   PASS  a non-periodic --interlace run without a region is refused before the first pass"
    else
        echo "   FAIL  non-periodic --interlace: not the expected refusal ($E2E/np.log)"; fails=$((fails+1))
    fi
fi
echo "------------------------------------------------------------"
if [ "$fails" -gt 0 ]; then echo "RESULT: FAIL ($fails)"; exit 1; fi
echo "RESULT: PASS  (interlacing undoes the half-cell shift for every Fourier mode, 2D and 3D, single and double; end to end, pass 2 grids the box moved half a cell)"
