"""Shared helpers for the PS-DTFE test scripts (import with tests/ as the script dir)."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# DTFE_TEST_PRECISION=double runs a suite against the double-precision pair (tests/precision.sh says
# the same for the shell suites): its GRID outputs are float64, its .pts_* outputs unchanged.
PRECISION = os.environ.get("DTFE_TEST_PRECISION", "single")
DOUBLE = PRECISION == "double"
REAL = np.dtype(np.float64 if DOUBLE else np.float32)    # the dtype of every grid output


def binary(name="PS-DTFE"):
    """The binary a suite runs: './PS-DTFE' or './PS-DTFE-double' (likewise DTFE)."""
    return os.path.join(ROOT, name + ("-double" if DOUBLE else ""))


def require(path):
    """Exit (FAIL) when a binary is not built -- in double mode with a SKIP line instead, like the shell
    suites: the double pair is optional ('make DTFE PS-DTFE DOUBLE=1')."""
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return
    if DOUBLE:
        print(f"SKIP: {path} is not built (make DTFE PS-DTFE DOUBLE=1 builds the double pair)")
        sys.exit(0)
    sys.exit(f"FAIL: '{path}' not built")


def read_grid(path, count=None):
    """A grid output as float64, read in the build's Real type."""
    d = np.fromfile(path, dtype=REAL)
    if count is not None and d.size != count:
        sys.exit(f"FAIL: {path} holds {d.size} values of {REAL}, expected {count}")
    return d.astype(np.float64)


def make_cmd():
    """Rebuild respecting the current build mode (never downgrade a GPU binary). In double mode: None
    (the suites do not rebuild the double pair)."""
    if DOUBLE:
        return None
    try:
        mode = open(os.path.join(ROOT, "o_ps", ".build_mode")).read().strip()
    except OSError:
        mode = "METAL=1" if os.path.isfile(os.path.join(ROOT, "o_ps", ".metal_mode_on")) else ""
    return ["make", "PS-DTFE"] + ([mode] if mode else [])


def load(path, grid, ncomp=1):
    """Load a raw binary DTFE grid; float32/float64 is inferred from the file size."""
    raw = np.fromfile(path, dtype=np.uint8)
    n = grid ** 3 * ncomp
    if raw.size == n * 4:
        d = raw.view(np.float32).astype(np.float64)
    elif raw.size == n * 8:
        d = raw.view(np.float64)
    else:
        sys.exit(f"FAIL: {path} has {raw.size} bytes, expected {n*4} or {n*8}")
    return d.reshape((grid, grid, grid) + ((ncomp,) if ncomp > 1 else ()))
