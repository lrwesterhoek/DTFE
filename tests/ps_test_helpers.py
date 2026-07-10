"""Shared helpers for the PS-DTFE test scripts (import with tests/ as the script dir)."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def make_cmd():
    """Rebuild respecting the current build mode (never downgrade a GPU binary)."""
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
