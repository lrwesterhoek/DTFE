#!/usr/bin/env bash
# One-command reinstall + full recompile of DTFE and PS-DTFE, platform-aware.
#
#   ./install.sh              install any missing dependencies, wipe the build, and rebuild
#                             BOTH binaries with the best backend for this machine:
#                               macOS (Apple Silicon)      -> METAL=1  (GPU deposit)
#                               anything else              -> CPU-only
#   ./install.sh --cpu        force a CPU-only build (skip the Metal backend detection)
#   ./install.sh --no-deps    skip the dependency-installation step (just clean + rebuild)
#   ./install.sh --docker     build the containerized LINUX image instead (docker build .).
#                             NOTE: a container is a Linux VM even on a Mac -- it can NEVER
#                             build or run the Metal backend. For native macOS binaries run
#                             this script WITHOUT --docker.
#   ./install.sh --jobs N     parallel build jobs (default: all cores)
#
# Idempotent: safe to re-run any time; the Makefile's build-mode stamps guarantee objects
# from different GPU modes are never mixed (a mode switch wipes the object directory).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # run from the repo root (this script lives in scripts/)

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${BLUE}[install]${NC} $1"; }
good()  { echo -e "${GREEN}[install]${NC} $1"; }
warn()  { echo -e "${YELLOW}[install]${NC} $1"; }
fail()  { echo -e "${RED}[install]${NC} $1"; exit 1; }

DO_DEPS=1
FORCE_CPU=0
USE_DOCKER=0
JOBS="$( (sysctl -n hw.ncpu || nproc || echo 4) 2>/dev/null | head -1 )"
while [ $# -gt 0 ]; do
    case "$1" in
        --no-deps) DO_DEPS=0 ;;
        --cpu)     FORCE_CPU=1 ;;
        --docker)  USE_DOCKER=1 ;;
        --jobs)    shift; JOBS="${1:?--jobs needs a number}" ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) fail "unknown option '$1' (see ./install.sh --help)" ;;
    esac
    shift
done

# ---------------------------------------------------------------- docker path (opt-in)
if [ "$USE_DOCKER" -eq 1 ]; then
    command -v docker >/dev/null 2>&1 || fail "docker CLI not found"
    docker info >/dev/null 2>&1 \
        || fail "no running docker daemon (install/start Docker Desktop, OrbStack or colima first)"
    warn "docker builds a LINUX (CPU-only) image; it cannot contain the Metal backend."
    warn "For native macOS binaries with Metal, run ./install.sh without --docker."
    exec docker build -t dtfe .
fi

# ---------------------------------------------------------------- 1/4 dependencies
UNAME_S="$(uname -s)"
if [ "$DO_DEPS" -eq 1 ]; then
    info "step 1/4: installing missing dependencies (skip with --no-deps)"
    bash scripts/install_dependencies.sh
else
    info "step 1/4: skipped (--no-deps)"
fi
make deps-check

# ---------------------------------------------------------------- 2/4 pick the backend
GPU_ARG=""
BACKEND="CPU-only"
if [ "$FORCE_CPU" -eq 0 ]; then
    if [ "$UNAME_S" = "Darwin" ]; then
        if [ "$(uname -m)" != "arm64" ]; then
            warn "Intel Mac detected: the Metal deposit is only validated on Apple Silicon; building CPU-only."
        else
            # the Metal host needs Apple's metal-cpp headers, vendored (git-ignored) in third_party/
            if [ ! -f third_party/metal-cpp/Metal/Metal.hpp ]; then
                info "third_party/metal-cpp missing -- fetching Apple's metal-cpp headers ..."
                MCPP_URL="https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
                if command -v curl >/dev/null 2>&1 \
                   && mkdir -p third_party \
                   && curl -fsSL --max-time 120 -o third_party/metal-cpp.zip "$MCPP_URL" \
                   && unzip -q -o third_party/metal-cpp.zip -d third_party \
                   && [ -f third_party/metal-cpp/Metal/Metal.hpp ]; then
                    rm -f third_party/metal-cpp.zip
                    good "metal-cpp headers installed to third_party/metal-cpp"
                else
                    rm -f third_party/metal-cpp.zip
                    warn "could not fetch metal-cpp automatically; building CPU-only."
                    warn "To enable Metal later: download metal-cpp from https://developer.apple.com/metal/cpp/"
                    warn "unzip it to third_party/metal-cpp, then re-run ./install.sh"
                fi
            fi
            if [ -f third_party/metal-cpp/Metal/Metal.hpp ]; then
                GPU_ARG="METAL=1"
                BACKEND="Metal GPU (--gpu / --ps-gpu at runtime, automatic CPU fallback)"
            fi
        fi
    fi
fi
info "step 2/4: build backend -> ${BACKEND}"

# ---------------------------------------------------------------- 3/4 clean rebuild
info "step 3/4: make clean + rebuild (jobs: ${JOBS})"
make clean >/dev/null
# shellcheck disable=SC2086  # GPU_ARG is deliberately word-split ("" or METAL=1 etc.)
make DTFE    $GPU_ARG -j"$JOBS"
make PS-DTFE $GPU_ARG -j"$JOBS"

# ---------------------------------------------------------------- 4/4 report + verify hints
good "step 4/4: done -- ./DTFE and ./PS-DTFE rebuilt (${BACKEND})"
if [ -n "$GPU_ARG" ]; then
    info "build mode stamped in o/.build_mode and o_ps/.build_mode ($GPU_ARG);"
    info "incremental rebuilds must keep it: make PS-DTFE \$(cat o_ps/.build_mode)"
fi
info "verify with the fast checks:"
info "    tests/ps_smoke_test.sh --no-build"
info "    tests/ps_point_eval_check.sh --no-build"
[ "$UNAME_S" = "Darwin" ] && [ -n "$GPU_ARG" ] && info "    tests/dtfe_metal_check.sh   (CPU vs GPU parity)"
exit 0
