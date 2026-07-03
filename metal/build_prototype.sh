#!/usr/bin/env bash
# Build the PS-DTFE Metal deposit prototype + validation harness.
# Offline-compiles the kernels to ps_deposit.metallib (runtime compile is the fallback).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo ">> compiling Metal kernels (offline)"
xcrun -sdk macosx metal    -c metal/ps_deposit.metal -o metal/ps_deposit.air
xcrun -sdk macosx metallib metal/ps_deposit.air      -o metal/ps_deposit.metallib

echo ">> building host binaries"
clang++ -std=c++17 -O2 -I third_party/metal-cpp metal/deposit_prototype.cpp \
    -framework Metal -framework Foundation -o metal/deposit_prototype
clang++ -std=c++17 -O2 -I third_party/metal-cpp metal/validate_deposit.cpp \
    -framework Metal -framework Foundation -o metal/validate_deposit
echo "built metal/deposit_prototype, metal/validate_deposit, metal/ps_deposit.metallib"
