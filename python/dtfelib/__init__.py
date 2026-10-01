"""dtfelib: unified access to DTFE / PS-DTFE outputs for the analysis pipeline.

Contract: the C++ binaries (DTFE, PS-DTFE) are the only producers of raw field grids;
everything downstream (smoothing, statistics, plots) is Python and goes through this
package so scripts are agnostic to the estimator, snapshot, simulation, and units.
"""

from .io import FieldSet, FIELDS, STREAM_TOL, SnapshotMeta, PointPlane
try:
    from . import pointeval  # noqa: F401  (figure rendering for --sample-points)
except ModuleNotFoundError as _e:   # the loaders and the Estimator need only numpy (+h5py);
    if not (_e.name or "").startswith("matplotlib"):   # only the figures need matplotlib
        raise
from .estimator import Estimator, PointFields, find_binary
from .cli import make_parser, make_fieldset, snapdir, sim_dir, find_sims, DATA_ROOT, DEFAULT_SIM, DEFAULT_SNAP

__all__ = [
    "FieldSet", "FIELDS", "STREAM_TOL", "SnapshotMeta", "PointPlane",
    "Estimator", "PointFields", "find_binary",
    "make_parser", "make_fieldset", "snapdir",
    "DATA_ROOT", "DEFAULT_SIM", "DEFAULT_SNAP", "sim_dir", "find_sims",
]
