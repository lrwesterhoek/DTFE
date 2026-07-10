
import os
from pathlib import Path

# The snapshot->redshift ladder and the Planck cosmology below are IDENTICAL for all IllustrisTNG
# runs (same output times, same parameters), so they are simulation-independent constants here.
SNAPSHOT_TO_REDSHIFT = {
    '000': 20.05,
    '004': 10.00,
    '008': 8.01,
    '013': 6.01,
    '017': 5.00,
    '021': 4.01,
    '025': 3.01,
    '033': 2.00,
    '040': 1.50,
    '050': 1.00,
    '067': 0.50,
    '072': 0.40,
    '078': 0.30,
    '084': 0.20,
    '091': 0.10,
    '099': 0.00,
}

PANEL_SNAPSHOTS = ['000', '017', '050', '099']

# ---- Active simulation & box geometry ---------------------------------------------------
# One simulation is analysed at a time: DTFE_SIM env var, else dtfelib's default. Every
# script that opens fields via dtfelib.cli can also override per run with --sim.
from dtfelib.cli import DATA_ROOT, DEFAULT_SIM
SIMULATION = os.environ.get("DTFE_SIM", DEFAULT_SIM)

# Comoving box sizes in h-free Mpc (raw BoxSize[ckpc/h] / h / 1000). Known simulations are
# pinned here (immune to stale on-disk headers from older unit conventions); a simulation
# NOT in this table gets its box read from a combined_*.hdf5 header on disk. When adding a
# new simulation, either add an entry here or make sure its merged snapshots exist first.
SIMULATION_BOX_MPC = {
    "TNG50-3-Dark":  51.668142899320934,
    "TNG50-4-Dark":  51.668142899320934,
    "TNG300-3-Dark": 302.62769412459404,
}


def _box_size_mpc(sim):
    if sim in SIMULATION_BOX_MPC:
        return SIMULATION_BOX_MPC[sim]
    root = DATA_ROOT / sim
    for combined in sorted(root.glob("snapdir_*/combined_*.hdf5"), reverse=True):
        import h5py
        with h5py.File(combined, "r") as f:
            return float(f["Header"].attrs["BoxSize"]) / 1000.0   # h-free ckpc -> Mpc
    raise KeyError(
        f"unknown simulation {sim!r}: not in config.SIMULATION_BOX_MPC and no "
        f"combined_*.hdf5 under {root} to read the box size from. Add a table entry "
        f"(box in h-free Mpc) or merge the snapshots first (python/tools/merge_HDF5.py).")


BOX_SIZE = _box_size_mpc(SIMULATION)
FIELD_RESOLUTION = 512     # grid of the C++ field outputs; FieldSet.grid_n reads the true value per file
CELL_SIZE = BOX_SIZE / FIELD_RESOLUTION

HUBBLE_H = 0.6774          # identical for every IllustrisTNG run (Planck 2015)
COSMOLOGY = {
    'H0': 67.74,
    'Omega_m': 0.3089,
    'Omega_Lambda': 0.6911,
    'Omega_b': 0.0486,
    'sigma8': 0.8159,
    'n_s': 0.9667,
}


# Gaussian smoothing (grid cells) for the derived-product pipeline (delta, Hessian, void
# catalogs, critical points). 10.0 is the production value every thesis correlation/void
# catalog was built with (cache namespaces r512_s10_*); at 0 the catalogs degenerate into
# single-cell noise dips.
SMOOTHING_SIGMA_CELLS = 10.0
SMOOTHING_SIGMA_MPC = SMOOTHING_SIGMA_CELLS * CELL_SIZE

RAW_MAPS_SIGMA = 0.0
VIS_SMOOTHING_SIGMA = RAW_MAPS_SIGMA

SMOOTHING_COMPARISON_SIGMAS = [5.0, 10.0, 20.0]

FOOTPRINT_SIZE = 11

GRADIENT_THRESHOLD = 0.1

DEEP_VOID_THRESHOLD = -0.1
ELLIPSOID_CUTS = {
    'min_axis_mpc': 0.1,
    'max_axis_mpc': 10.0,
    'max_axis_ratio': 10.0,
}

CORRELATION_MAX_SEP = None
CORRELATION_N_BINS = 50
CORRELATION_YLIM = 0.75

CORRELATION_N_EVAL = 60
CORRELATION_MIN_PAIRS = 20
CORRELATION_N_PERMUTATIONS = 1000
CORRELATION_BAND_PERCENTILES = (2.5, 97.5)
CORRELATION_M_YLIM = 0.2
CORRELATION_SEED = 42

# Draw the shaded central-95% (2.5-97.5 percentile) Monte-Carlo null
# envelopes on the correlation / alignment / profile plots
# (plot_marked_correlation_BBKS, plot_tidal_correlation, plot_phi_delta,
# plot_conclusions_synthesis). Off by default. The null statistics are still
# computed and reported in the console/summary output regardless of this flag.
SHOW_NULL_BANDS = False

EXTREMA_MATCH_DISTANCE_CELLS = 11.0
EXTREMA_NULL_DRAWS = 1000

PHI_PROFILE_RMAX_MPC = 20.0
PHI_PROFILE_NBINS = 30

SHEAR_MAPS_SIGMA = 1.0


def correlation_max_sep():
    if CORRELATION_MAX_SEP is not None:
        return float(CORRELATION_MAX_SEP)
    return float(3 ** 0.5 / 2 * BOX_SIZE)

VOID_EIGENVALUE_CRITERION = 'trace'


_REPO_ROOT = Path(__file__).resolve().parent


def _find_thesis_dir():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'Thesis_Final.tex').exists():
            return parent
    return None


_THESIS_DIR = _find_thesis_dir()
THESIS_FIGURES_DIR = (_THESIS_DIR / 'Figures') if _THESIS_DIR else (_REPO_ROOT.parent / 'Figures')

LOCAL_FIGURES_ROOT = str(_REPO_ROOT / 'figures')

ANALYSIS_DIRS = {
    'dtfe': 'dtfe_analysis',
    'veldiv': 'veldiv_density_analysis',
    'critical_points': 'critical_point_analysis',
    'eigenvalues': 'eigenvalue_analysis',
    'shear': 'shear_analysis',
    'void_analysis': 'void_analysis',
    'void_shapes': 'void_shape_analysis',
    'ellipses': 'void_analysis_plots',
    'correlation': 'correlation_analysis',
    'alignment': 'alignment_analysis',
    'tidal': 'tidal_correlation',
    'extrema': 'extrema_correlation',
}


def figures_path(analysis_key):
    subdir = ANALYSIS_DIRS.get(analysis_key, analysis_key)
    return f"{LOCAL_FIGURES_ROOT}/{subdir}"


# Data location: single source of truth is dtfelib (DATA_ROOT / SIMULATION, both env-
# overridable; every plot script can also override per run with --sim / --data-root).
# Note: TNG50-3-Dark's on-disk DTFE outputs predate the 2026-06-17 unit overhaul and are
# stale for absolute units (mean-relative quantities remain usable).

CACHE_DIR = _REPO_ROOT / 'cache'


PERCENTILE_CLIP = (0.5, 99.5)

SYMLOG_LINTHRESH_FRAC = 0.02
SYMLOG_LINTHRESH_OVERRIDES = {'delta': 0.1}

FIELD_KINDS = {
    'density': 'positive',
    'shear': 'positive',
    'delta': 'signed_centered',
    'divergence': 'signed_linear',
    'residual': 'signed_linear',
    'potential': 'signed',
    'eigenvalue': 'signed',
    'velocity': 'positive',
    'triaxiality': 'bounded',
    'probability': 'bounded',
}

FIELD_LIMITS = {
    'density': None,
    'shear': None,
    'delta': None,
    'divergence': None,
    'residual': None,
    'potential': None,
    'eigenvalue': None,
    'velocity': None,
    'triaxiality': (0.0, 1.0),
    'probability': (0.0, 1.0),
}

AXIS_UNITS = "Mpc"
VELOCITY_UNITS = "km/s"
DPI = 300

SHOW_TITLES = False

SLICE_PLANES = {
    0: {'name': 'yz_plane', 'axis_labels': ('y', 'z'), 'axes': (1, 2)},
    1: {'name': 'xz_plane', 'axis_labels': ('x', 'z'), 'axes': (0, 2)},
    2: {'name': 'xy_plane', 'axis_labels': ('x', 'y'), 'axes': (0, 1)},
}

def snapshot_items():
    return sorted(SNAPSHOT_TO_REDSHIFT.items(), key=lambda kv: -kv[1])


def get_redshift(snapshot_id):
    return SNAPSHOT_TO_REDSHIFT.get(snapshot_id)


def cosmic_time_gyr(z, n_steps=20000):
    import math
    Om, Ol = COSMOLOGY['Omega_m'], COSMOLOGY['Omega_Lambda']
    H0_inv_gyr = 977.79222 / COSMOLOGY['H0']
    lo, hi = math.log(1.0 + z), math.log(3001.0)
    dx = (hi - lo) / n_steps
    s = 0.0
    for i in range(n_steps + 1):
        zp = math.exp(lo + i * dx) - 1.0
        f = 1.0 / math.sqrt(Om * (1.0 + zp) ** 3 + Ol)
        w = 1 if i in (0, n_steps) else (4 if i % 2 else 2)
        s += w * f
    return s * dx / 3.0 * H0_inv_gyr


def verify_snapshot_redshifts(base_dir=None, tolerance=0.02):
    import glob
    base = Path(base_dir) if base_dir else DATA_ROOT / SIMULATION
    mismatches = []
    try:
        import h5py
    except ImportError:
        print("verify_snapshot_redshifts: h5py not available, skipping")
        return mismatches
    for snap, z_cfg in SNAPSHOT_TO_REDSHIFT.items():
        pattern = str(base / f"snapdir_{snap}" / "*.hdf5")
        files = glob.glob(pattern)
        if not files:
            continue
        with h5py.File(files[0], 'r') as f:
            z_hdr = float(f['Header'].attrs.get('Redshift', float('nan')))
        if abs(z_hdr - z_cfg) > tolerance:
            mismatches.append((snap, z_cfg, z_hdr))
    return mismatches
