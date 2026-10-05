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

# Same environment variables as the shell side (config.sh): DTFE_DATA_ROOT / DTFE_SIM. The default
# root is the Samsung T7, which groups simulations by family (Illustris TNG/TNG100/TNG100-3-Dark);
# sim_dir() resolves that layout and the flat one (e.g. DTFE_DATA_ROOT=~/output).
DATA_ROOT = Path(os.environ.get("DTFE_DATA_ROOT", "/Volumes/Samsung T7/Illustris TNG"))
DEFAULT_SIM = os.environ.get("DTFE_SIM", "TNG50-4-Dark")
DEFAULT_SNAP = 99


def sim_dir(sim: str, root=None) -> Path:
    """Directory of simulation `sim` under `root` (default DATA_ROOT), in either layout: flat
    <root>/<sim>, or per family <root>/<family>/<sim> with <family> the name up to its first '-'
    (TNG100-3-Dark -> TNG100). An existing directory wins; a simulation not on disk yet goes into
    its family folder when that exists, else flat. Mirrors sim_dir in scripts/config.sh."""
    root = DATA_ROOT if root is None else Path(root)
    flat = root / sim
    family = root / sim.split("-", 1)[0]
    if flat.is_dir():
        return flat
    if family != flat and family.is_dir():
        return family / sim
    return flat


def use_data_root(root) -> Path:
    """Make `root` this process's default data root: what a script's --data-root means for every dtfelib
    helper that is not handed a root explicitly (groupcat, trees, environment, pipeline.products without
    data_root=...). sim_dir() reads the default at call time, so the switch reaches them all. Returns it."""
    global DATA_ROOT
    DATA_ROOT = Path(root)
    return DATA_ROOT


def find_sims(root=None, pattern: str = "snapdir_*") -> list[str]:
    """Names of the simulations under `root` (default DATA_ROOT), in either layout, whose
    directory contains something matching `pattern` (e.g. 'snapdir_*/combined_*.hdf5')."""
    root = DATA_ROOT if root is None else Path(root)
    if not root.is_dir():
        return []
    names = set()
    for d in list(root.iterdir()) + [c for f in root.iterdir() if f.is_dir() for c in f.iterdir()]:
        if d.is_dir() and any(d.glob(pattern)):
            names.add(d.name)
    return sorted(names)


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
    p.add_argument("--prefix", default=None,
                   help="alternate on-disk grid prefix (run_ps_dtfe.sh OUTPUT_PREFIX, e.g. "
                        "'ps_mw'); implies --method ps unless one is given explicitly")
    p.add_argument("--raw", action="store_true",
                   help="use the unaveraged fields instead of the volume-averaged '_a' ones")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="Gaussian smoothing applied at plot/analysis time, in grid cells (default 0)")
    return p


def snapdir(args) -> Path:
    return sim_dir(args.sim, args.data_root) / f"snapdir_{args.snap:03d}"


def make_fieldset(description: str = "", extra=None, argv=None):
    """Parse standard flags (+ optional script-specific ones) and open the FieldSet.

    extra: callable(parser) that registers additional arguments.
    Returns (fieldset, args).
    """
    parser = make_parser(description)
    if extra is not None:
        extra(parser)
    args = parser.parse_args(argv)
    fs = FieldSet(snapdir(args), method=args.method, averaged=not args.raw,
                  prefix=args.prefix)
    return fs, args
