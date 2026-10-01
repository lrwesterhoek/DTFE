#!/usr/bin/env bash
# Graphical launcher for run_ps_dtfe.sh / run_dtfe.sh (python/gui/app.py, PySide6 prototype).
#
#   scripts/gui.sh
#
# Uses DTFE_GUI_PYTHON if set, else the venv scripts/install.sh sets up (~/.venvs/dtfe-gui, or
# DTFE_GUI_VENV), else python3. Set up once with scripts/install.sh, or by hand:
#   python3 -m venv ~/.venvs/dtfe-gui --system-site-packages   # keeps numpy/h5py from the system
#   ~/.venvs/dtfe-gui/bin/pip install PySide6
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${DTFE_GUI_PYTHON:-}"
if [ -z "${PY}" ]; then
    VENV="${DTFE_GUI_VENV:-${HOME}/.venvs/dtfe-gui}"
    if [ -x "${VENV}/bin/python" ]; then PY="${VENV}/bin/python"; else PY=python3; fi
fi
# macOS: run inside the GUI host bundle scripts/install_gui_app.sh puts next to the venv, so the
# Dock, Cmd-Tab and the menu bar say "DTFE Launcher" rather than "Python" (the plain interpreter
# is the fallback whenever the host is missing or cannot import PySide6)
HOST="${VENV:-}/DTFE Launcher.app/Contents/MacOS/DTFE Launcher"
if [ "$(uname -s)" = "Darwin" ] && [ -n "${VENV:-}" ] && [ "${PY}" = "${VENV}/bin/python" ] \
   && [ -x "${HOST}" ] && "${HOST}" -c "import PySide6" 2>/dev/null; then
    exec "${HOST}" "${REPO_ROOT}/python/gui/app.py" "$@"
fi
if ! "${PY}" -c "import PySide6" 2>/dev/null; then
    echo "PySide6 is not available to ${PY}. Set it up once with scripts/install.sh --no-deps" >&2
    echo "(which also rebuilds), or by hand:" >&2
    echo "  python3 -m venv ~/.venvs/dtfe-gui --system-site-packages" >&2
    echo "  ~/.venvs/dtfe-gui/bin/pip install PySide6" >&2
    exit 1
fi
exec "${PY}" "${REPO_ROOT}/python/gui/app.py" "$@"
