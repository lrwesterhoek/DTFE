#!/usr/bin/env bash
# Build the PS-DTFE Metal deposit validation harness.
# Offline-compiles the kernels to ps_deposit.metallib (runtime compile is the fallback).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The harness loads metal/ps_deposit.metallib whenever it exists, so a library left over from
# an older kernel is silently tested instead of the current source. Remove it first; without
# the offline compiler (Xcode's optional Metal toolchain) the harness compiles from source.
rm -f metal/ps_deposit.air metal/ps_deposit.metallib
if xcrun -sdk macosx metal -v >/dev/null 2>&1; then
    echo ">> compiling Metal kernels (offline)"
    xcrun -sdk macosx metal    -c metal/ps_deposit.metal -o metal/ps_deposit.air
    xcrun -sdk macosx metallib metal/ps_deposit.air      -o metal/ps_deposit.metallib
else
    echo ">> no offline Metal compiler (xcodebuild -downloadComponent MetalToolchain); the harness compiles the kernel at run time"
fi

echo ">> building host binary"
clang++ -std=c++17 -O2 -I third_party/metal-cpp metal/validate_deposit.cpp \
    -framework Metal -framework Foundation -o metal/validate_deposit
echo "built metal/validate_deposit"
