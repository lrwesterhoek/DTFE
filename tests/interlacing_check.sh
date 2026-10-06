#!/usr/bin/env bash
# Interlacing ('--interlace', src/interlacing.cc) against an exact answer: tests/interlacing_unit.cpp
# compiled against the source in 3D and 2D, single and double precision. No binary is involved: the
# function is checked on grids holding exact samples of a band-limited field (see the .cpp).
# Usage: tests/interlacing_check.sh [--no-build]   (the flag is accepted for the battery; nothing to build)
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
echo "------------------------------------------------------------"
if [ "$fails" -gt 0 ]; then echo "RESULT: FAIL ($fails)"; exit 1; fi
echo "RESULT: PASS  (interlacing undoes the half-cell shift for every Fourier mode, 2D and 3D, single and double)"
