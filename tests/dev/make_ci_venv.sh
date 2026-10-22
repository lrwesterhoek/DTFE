#!/usr/bin/env bash
# Makes the CI-like python of tests/dev/battery.sh's '*_np126' stages: python 3.11 with the package versions of CI's
# Ubuntu 24.04 (numpy 1.26.4, scipy 1.11.4, matplotlib 3.6.3; CI installs them with apt), plus h5py and PySide6.
# The Mac's own python has numpy 2 and a newer matplotlib, which hide what fails on CI (both broke CI once).
# Keep it outside /tmp: macOS clears /tmp on a restart and purges files there unused for ~3 days.
# Usage: tests/dev/make_ci_venv.sh [venv dir]      (default ~/.venvs/dtfe-ci; PYTHON311 picks the interpreter)
set -euo pipefail
DIR="${1:-$HOME/.venvs/dtfe-ci}"
PY311="${PYTHON311:-$(command -v python3.11 || true)}"
if [ -z "$PY311" ] && command -v uv >/dev/null 2>&1; then PY311="$(uv python find 3.11 2>/dev/null || true)"; fi
[ -n "$PY311" ] || { echo "no python3.11 found: install one (python.org, or 'uv python install 3.11') or set PYTHON311"; exit 1; }
"$PY311" -m venv "$DIR"
"$DIR/bin/pip" install --quiet --upgrade pip
"$DIR/bin/pip" install --quiet numpy==1.26.4 scipy==1.11.4 matplotlib==3.6.3 h5py PySide6
"$DIR/bin/python" -c 'import sys, numpy, scipy, matplotlib, h5py, PySide6; print(f"{sys.executable}: numpy {numpy.__version__}, scipy {scipy.__version__}, matplotlib {matplotlib.__version__}, h5py {h5py.__version__}, PySide6 {PySide6.__version__}")'
