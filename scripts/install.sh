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
#   ./install.sh --no-python  skip step 4 (the editable 'pip install -e python' of dtfelib)
#   ./install.sh --no-gui     skip step 5 (the GUI's PySide6 environment; e.g. on a cluster)
#   ./install.sh --no-app     set up the GUI, but no "DTFE Launcher" app / Desktop shortcut
#   ./install.sh --double     build the double-precision binaries (DOUBLE=1; CPU-only)
#
# The GUI (scripts/gui.sh) gets its own virtual environment, ~/.venvs/dtfe-gui (DTFE_GUI_VENV
# overrides), made with --system-site-packages so it sees this Python's numpy/h5py, holding
# PySide6 (~1 GB). Already set up -> left alone. It never fails the install: the binaries come first.
# Then scripts/install_gui_app.sh makes it launchable like any application: "DTFE Launcher.app" in
# /Applications + a Desktop shortcut on macOS (give it Full Disk Access once: the repository and
# the data are on protected locations), a .desktop entry on Linux.
#
# After it, Python can drive the binaries directly:
#     from dtfelib import Estimator
#     est = Estimator("snapshot.hdf5"); est(points).density
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
DO_PYTHON=1
DO_GUI=1
DO_APP=1
PREC_ARG=""
JOBS="$( (sysctl -n hw.ncpu || nproc || echo 4) 2>/dev/null | head -1 )"
while [ $# -gt 0 ]; do
    case "$1" in
        --no-deps) DO_DEPS=0 ;;
        --cpu)     FORCE_CPU=1 ;;
        --docker)  USE_DOCKER=1 ;;
        --no-python) DO_PYTHON=0 ;;
        --no-gui)  DO_GUI=0 ;;
        --no-app)  DO_APP=0 ;;
        --double)  PREC_ARG="DOUBLE=1"; FORCE_CPU=1 ;;   # the GPU kernels are single precision
        --jobs)    shift; JOBS="${1:?--jobs needs a number}" ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

# ---------------------------------------------------------------- 1/6 dependencies
UNAME_S="$(uname -s)"
if [ "$DO_DEPS" -eq 1 ]; then
    info "step 1/6: installing missing dependencies (skip with --no-deps)"
    bash scripts/install_dependencies.sh
else
    info "step 1/6: skipped (--no-deps)"
fi
make deps-check
# optional: the tbb library lets PS-DTFE triangulate in parallel (Makefile TBB=auto detects it)
TBB_INC="$([ "$UNAME_S" = "Darwin" ] && echo "$(brew --prefix tbb 2>/dev/null)/include" || echo /usr/include)"
if [ -f "$TBB_INC/tbb/task_arena.h" ] || [ -f "$TBB_INC/oneapi/tbb/task_arena.h" ]; then
    info "tbb found: PS-DTFE will build its triangulation in parallel (TBB=0 disables)"
else
    warn "tbb not found: PS-DTFE triangulates on one core (optional: brew install tbb / apt-get install libtbb-dev)"
fi

# ---------------------------------------------------------------- 2/6 pick the backend
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
[ -n "$PREC_ARG" ] && BACKEND="${BACKEND}, double precision"
info "step 2/6: build backend -> ${BACKEND}"

# ---------------------------------------------------------------- 3/6 clean rebuild
info "step 3/6: make clean + rebuild (jobs: ${JOBS})"
make clean >/dev/null
# shellcheck disable=SC2086  # GPU_ARG/PREC_ARG are deliberately word-split ("" or METAL=1 etc.)
make DTFE    $GPU_ARG $PREC_ARG -j"$JOBS"
make PS-DTFE $GPU_ARG $PREC_ARG -j"$JOBS"

# ---------------------------------------------------------------- 4/6 python package
if [ "$DO_PYTHON" -eq 1 ]; then
    PY="${PYTHON:-python3}"
    info "step 4/6: pip install -e python  (dtfelib, editable; skip with --no-python)"
    if ! command -v "$PY" >/dev/null 2>&1; then
        warn "no '$PY' found; skipped. Install dtfelib later with: python3 -m pip install -e python"
    elif "$PY" -m pip install -e python; then
        good "dtfelib installed: 'from dtfelib import Estimator' works from any directory"
    else
        # e.g. PEP 668 'externally-managed-environment' on a Homebrew/system interpreter
        warn "pip could not install dtfelib into '$PY' (see above). The binaries are built;"
        warn "install dtfelib inside a virtual environment instead:"
        warn "    python3 -m venv .venv && . .venv/bin/activate && pip install -e python"
    fi
else
    info "step 4/6: skipped (--no-python)"
fi

# ---------------------------------------------------------------- 5/6 the GUI
GUI_STATE="not installed (--no-gui)"
if [ "$DO_GUI" -eq 1 ]; then
    PY="${PYTHON:-python3}"
    GUI_VENV="${DTFE_GUI_VENV:-$HOME/.venvs/dtfe-gui}"
    GUI_PY="$GUI_VENV/bin/python"
    info "step 5/6: the GUI -> PySide6 in ${GUI_VENV}  (skip with --no-gui)"
    GUI_STATE="NOT installed (see the warnings of step 5)"
    if [ -x "$GUI_PY" ] && "$GUI_PY" -c "import PySide6" >/dev/null 2>&1; then
        good "PySide6 $("$GUI_PY" -c 'import PySide6; print(PySide6.__version__)') already there; left as is"
        GUI_STATE="ready"
    elif ! command -v "$PY" >/dev/null 2>&1; then
        warn "no '$PY' found; GUI skipped"
    elif ! "$PY" -m venv --system-site-packages "$GUI_VENV"; then
        warn "could not create $GUI_VENV (Debian/Ubuntu: sudo apt-get install python3-venv); GUI skipped"
    elif ! "$GUI_PY" -m pip install --quiet --upgrade pip || ! "$GUI_PY" -m pip install --quiet "PySide6>=6.5"; then
        warn "pip could not install PySide6 into $GUI_VENV (network?); retry: $GUI_PY -m pip install PySide6"
    else
        good "PySide6 $("$GUI_PY" -c 'import PySide6; print(PySide6.__version__)') installed"
        GUI_STATE="ready"
    fi
    if [ "$GUI_STATE" = "ready" ]; then
        # the GUI imports dtfelib (numpy, h5py) from this Python, and loads Qt's libraries: check
        # both now, headless, instead of letting the first launch fail
        if ! "$GUI_PY" -c "import numpy, h5py" >/dev/null 2>&1; then
            warn "the GUI needs numpy and h5py in '$PY' (its venv sees that Python's packages):"
            warn "    $PY -m pip install numpy h5py     (or: brew install numpy / apt-get install python3-h5py)"
            GUI_STATE="installed, but numpy/h5py are missing"
        elif ! QT_QPA_PLATFORM=offscreen "$GUI_PY" -c "import sys; sys.path.insert(0, 'python/gui'); import app" \
                >/dev/null 2>&1; then
            warn "PySide6 is installed but Qt's libraries do not load. On Debian/Ubuntu:"
            warn "    sudo apt-get install libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3 libxcb-cursor0"
            GUI_STATE="installed, but Qt does not load"
        fi
    fi
    [ "$GUI_STATE" = "ready" ] && good "GUI ready: scripts/gui.sh"
    if [ "$GUI_STATE" = "ready" ] && [ "$DO_APP" -eq 1 ]; then
        if DTFE_GUI_PYTHON="$GUI_PY" bash scripts/install_gui_app.sh; then
            GUI_STATE="ready (DTFE Launcher: Applications + Desktop)"
        else
            warn "could not make the DTFE Launcher app (see above); scripts/gui.sh still works"
        fi
    fi
else
    info "step 5/6: skipped (--no-gui)"
fi

# ---------------------------------------------------------------- 6/6 report + verify hints
good "step 6/6: done -- ./DTFE and ./PS-DTFE rebuilt (${BACKEND}); GUI: ${GUI_STATE}"
if [ -n "$GPU_ARG$PREC_ARG" ]; then
    info "build mode stamped in o/.build_mode and o_ps/.build_mode ($GPU_ARG$PREC_ARG);"
    info "incremental rebuilds must keep it: make PS-DTFE \$(cat o_ps/.build_mode)"
fi
info "verify with the fast checks:"
info "    tests/ps_smoke_test.sh --no-build"
info "    tests/ps_point_eval_check.sh --no-build"
[ "$UNAME_S" = "Darwin" ] && [ -n "$GPU_ARG" ] && info "    tests/dtfe_metal_check.sh   (CPU vs GPU parity)"
case "$GUI_STATE" in ready*) info "    ${GUI_PY} tests/py_gui_app_test.py   (the GUI, end to end)" ;; esac
exit 0
