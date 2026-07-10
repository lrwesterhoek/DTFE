"""Shared command-line interface for analysis/plot scripts.

Every script gains the same flags so the whole pipeline is method- and snapshot-agnostic:

    python3 plot/plot_PS_DTFE.py                        # defaults: TNG50-4-Dark, snap 99, auto
    python3 plot/plot_PS_DTFE.py --method dtfe
    python3 plot/plot_PS_DTFE.py --snap 50 --smooth 1.5
    python3 plot/plot_PS_DTFE.py --sim TNG50-3-Dark --raw

Scripts call make_fieldset() and get a ready FieldSet plus the parsed args for their own
extra options (pass extra=... to register script-specific flags on the same parser).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .io import FieldSet

# Same environment variables as the shell side (config.sh): DTFE_DATA_ROOT / DTFE_SIM.
DATA_ROOT = Path(os.environ.get("DTFE_DATA_ROOT", str(Path.home() / "output")))
DEFAULT_SIM = os.environ.get("DTFE_SIM", "TNG50-4-Dark")
DEFAULT_SNAP = 99


def make_parser(description: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT,
                   help=f"simulation output root (default {DATA_ROOT})")
    p.add_argument("--sim", default=DEFAULT_SIM,
                   help=f"simulation name (default {DEFAULT_SIM})")
    p.add_argument("--snap", type=int, default=DEFAULT_SNAP,
                   help=f"snapshot number (default {DEFAULT_SNAP})")
    p.add_argument("--method", choices=("auto", "ps", "dtfe"), default="auto",
                   help="field estimator to load: PS-DTFE, standard DTFE, or auto-detect")
    p.add_argument("--raw", action="store_true",
                   help="use the unaveraged fields instead of the volume-averaged '_a' ones")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="Gaussian smoothing applied at plot/analysis time, in grid cells (default 0)")
    return p


def snapdir(args) -> Path:
    return args.data_root / args.sim / f"snapdir_{args.snap:03d}"


def make_fieldset(description: str = "", extra=None, argv=None):
    """Parse standard flags (+ optional script-specific ones) and open the FieldSet.

    extra: callable(parser) that registers additional arguments.
    Returns (fieldset, args).
    """
    parser = make_parser(description)
    if extra is not None:
        extra(parser)
    args = parser.parse_args(argv)
    fs = FieldSet(snapdir(args), method=args.method, averaged=not args.raw)
    return fs, args
