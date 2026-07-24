"""dtfelib: unified access to DTFE / PS-DTFE outputs for the analysis pipeline.

Contract: the C++ binaries (DTFE, PS-DTFE) are the only producers of raw field grids;
everything downstream (smoothing, statistics, plots) is Python and goes through this
package so scripts are agnostic to the estimator, snapshot, simulation, and units.
"""

from .io import FieldSet, FIELDS, STREAM_TOL, SnapshotMeta, PointPlane
from . import pointeval  # noqa: F401  (figure rendering for --sample-points)
from .cli import make_parser, make_fieldset, snapdir, DATA_ROOT, DEFAULT_SIM, DEFAULT_SNAP

__all__ = [
    "FieldSet", "FIELDS", "STREAM_TOL", "SnapshotMeta", "PointPlane",
    "make_parser", "make_fieldset", "snapdir",
    "DATA_ROOT", "DEFAULT_SIM", "DEFAULT_SNAP",
]
