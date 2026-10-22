"""Shared command-line interface for analysis/plot scripts.

Every script gains the same flags so the whole pipeline is method- and snapshot-agnostic:

    python3 plot/plot_PS_DTFE.py --sim TNG50-3-Dark                # snap 99, method auto
    python3 plot/plot_PS_DTFE.py --sim TNG50-3-Dark --method dtfe
    DTFE_SIM=TNG100-3-Dark python3 plot/plot_PS_DTFE.py --snap 50 --smooth 1.5

There is no default simulation: every run names it, with --sim or DTFE_SIM (require_sim).

Scripts call make_fieldset() and get a ready FieldSet plus the parsed args for their own
extra options (pass extra=... to register script-specific flags on the same parser).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .io import FieldSet

# Same environment variables as the shell side (config.sh): DTFE_DATA_ROOT / DTFE_SIM. Neither has a default: the data
# root is read where data is needed (require_data_root), never at import, so every module imports without it.
# sim_dir() resolves both layouts under it: flat <root>/<sim>, or per family <root>/TNG100/TNG100-3-Dark.
NO_DATA_ROOT = "no data root: set DTFE_DATA_ROOT to the folder that holds the simulations"


class NoSimulation(ValueError):
    """No simulation was named: there is no default, a run gives --sim NAME or DTFE_SIM=NAME."""


class NoDataRoot(ValueError):
    """No data root was named: there is no default, a run sets DTFE_DATA_ROOT (or gives a script's --data-root)."""


_data_root_override = None          # use_data_root(): a script's --data-root for every helper of this process


def require_data_root(root=None) -> Path:
    """The folder that holds the simulations: 'root' when given, else the one use_data_root() set for this process,
    else $DTFE_DATA_ROOT (~ expanded). There is no default; without any of them this raises NoDataRoot."""
    if root:
        return Path(root).expanduser()
    if _data_root_override is not None:
        return _data_root_override
    env = os.environ.get("DTFE_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    raise NoDataRoot(NO_DATA_ROOT)


def require_sim(sim: str | None = None) -> str:
    """The simulation to work on: 'sim' when given, else $DTFE_SIM. There is no default; without either this raises
    NoSimulation, naming both ways to give one and the simulations found under the data root (when one is set)."""
    sim = sim or os.environ.get("DTFE_SIM")
    if sim:
        return sim
    try:
        root = require_data_root()
        found = find_sims(root) if root.is_dir() else []
    except (NoDataRoot, OSError):
        root, found = None, []
    raise NoSimulation("no simulation given: pass --sim NAME or set DTFE_SIM=NAME"
                       + (f" (under {root}: {', '.join(found)})" if found else ""))


# Where every figure goes (config.LOCAL_FIGURES_ROOT, the launcher's figure browser): DTFE_FIGURES_ROOT, else
# python/figures (git-ignored). Not derived from the data root: the launcher hands its data root only to the scripts
# it runs, so the launcher and its scripts would look in different places. The tests set DTFE_FIGURES_ROOT to a
# temp dir.
FIGURES_ROOT = Path(os.environ.get("DTFE_FIGURES_ROOT") or Path(__file__).resolve().parents[1] / "figures")
DEFAULT_SNAP = 99


def sim_dir(sim: str, root=None) -> Path:
    """Directory of simulation `sim` under `root` (default require_data_root()), in either layout: flat
    <root>/<sim>, or per family <root>/<family>/<sim> with <family> the name up to its first '-'
    (TNG100-3-Dark -> TNG100). An existing directory wins; a simulation not on disk yet goes into
    its family folder when that exists, else flat. Mirrors sim_dir in scripts/config.sh."""
    root = require_data_root(root)
    flat = root / sim
    family = root / sim.split("-", 1)[0]
    if flat.is_dir():
        return flat
    if family != flat and family.is_dir():
        return family / sim
    return flat


def use_data_root(root) -> Path | None:
    """Make `root` this process's data root: what a script's --data-root means for every dtfelib helper that is
    not handed a root explicitly (groupcat, trees, environment, pipeline.products without data_root=...).
    require_data_root() reads it at call time, so the switch reaches them all. None (a script run without
    --data-root) clears it: DTFE_DATA_ROOT applies again. Returns the root now in force, None when none is."""
    global _data_root_override
    _data_root_override = Path(root).expanduser() if root else None
    try:
        return require_data_root()
    except NoDataRoot:
        return None


def find_sims(root=None, pattern: str = "snapdir_*") -> list[str]:
    """Names of the simulations under `root` (default require_data_root()), in either layout, whose
    directory contains something matching `pattern` (e.g. 'snapdir_*/combined_*.hdf5')."""
    root = require_data_root(root)
    if not root.is_dir():
        return []
    names = set()
    for d in list(root.iterdir()) + [c for f in root.iterdir() if f.is_dir() for c in f.iterdir()]:
        if d.is_dir() and any(d.glob(pattern)):
            names.add(d.name)
    return sorted(names)


def make_parser(description: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-root", type=Path, default=None,
                   help="the folder that holds the simulations (default: DTFE_DATA_ROOT; one of the two is needed)")
    env_sim = os.environ.get("DTFE_SIM")
    p.add_argument("--sim", default=env_sim, required=not env_sim,
                   help="simulation name (required unless DTFE_SIM is set; there is no default)")
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


# The PS-DTFE velocity-gradient family a figure script uses: the scripts that set PS beside standard DTFE
# or linear theory -- plot_velDiv_den, plot_cosmic_web, plot_web_streams, plot_shear_triaxial, plot_DTFE,
# plot_halo_tracking -- read PS's divergence / shear / vorticity / gradient / V-web as the MEAN FIELD by default,
# the quantity standard DTFE stores (FieldSet.load(name, mean_field=True); dtfelib/fields.py), and '--ps-gradient
# stored' gives back the binary's PS grids, the stream average <grad v>. Standard-DTFE grids are read as stored
# either way (they are the mean field).
PS_GRADIENT_CHOICES = ("mean-field", "stored")


def add_ps_gradient_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register '--ps-gradient {mean-field,stored}' (default mean-field) on a script's parser; returns it."""
    parser.add_argument("--ps-gradient", choices=PS_GRADIENT_CHOICES, default="mean-field",
                        help="PS-DTFE grids only: which velocity-gradient family to use for PS -- 'mean-field' "
                             "(default): the derivatives of the PS velocity grid, grad<v>, the quantity standard "
                             "DTFE stores, computed by FFT; 'stored': the binary's PS grids, the stream average "
                             "<grad v> of each stream's own gradient. Standard-DTFE grids are always read as stored.")
    return parser


def mean_field_wanted(args) -> bool:
    """True unless the script was run with '--ps-gradient stored' (a parser without the flag: True)."""
    return getattr(args, "ps_gradient", "mean-field") != "stored"


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
    try:
        args.data_root = require_data_root(args.data_root)
    except NoDataRoot as e:
        parser.error(str(e))
    fs = FieldSet(snapdir(args), method=args.method, averaged=not args.raw,
                  prefix=args.prefix)
    return fs, args
