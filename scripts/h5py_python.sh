# Sourced by scripts/config.sh and tests/lib.sh. h5py_python prints the python the scripts and test suites run for
# numpy + h5py: $PYTHON when set (taken as given), else the first of the python3 on PATH, Homebrew's and /usr/local's
# that imports both (a python3 first on PATH may lack h5py, or carry one built for another architecture), else
# python3, whose import error then names what is missing.
h5py_python() {
    if [ -n "${PYTHON:-}" ]; then echo "${PYTHON}"; return 0; fi
    local p
    for p in python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
        if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import numpy, h5py' >/dev/null 2>&1; then
            command -v "$p"
            return 0
        fi
    done
    echo python3
}
