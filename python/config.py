
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
from dtfelib.cli import DATA_ROOT, DEFAULT_SIM, FIGURES_ROOT, sim_dir
SIMULATION = os.environ.get("DTFE_SIM", DEFAULT_SIM)

# Comoving box sizes in h-free Mpc (raw BoxSize[ckpc/h] / h / 1000). Known simulations are
# pinned here (immune to stale on-disk headers from older unit conventions); a simulation
# NOT in this table gets its box read from a combined_*.hdf5 header on disk. When adding a
# new simulation, either add an entry here or make sure its merged snapshots exist first.
SIMULATION_BOX_MPC = {
    "TNG50-3-Dark":  51.668142899320934,
    "TNG50-4-Dark":  51.668142899320934,
    "TNG100-3-Dark": 110.71744906997343,    # 75 Mpc/h / 0.6774 (added 2026-10-07: CI has no snapshot to read it from)
    "TNG300-3-Dark": 302.62769412459404,
}


def _box_size_mpc(sim):
    if sim in SIMULATION_BOX_MPC:
        return SIMULATION_BOX_MPC[sim]
    root = sim_dir(sim)
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

RAW_MAPS_SIGMA = 0.0
VIS_SMOOTHING_SIGMA = RAW_MAPS_SIGMA

SMOOTHING_COMPARISON_SIGMAS = [5.0, 10.0, 20.0]

FOOTPRINT_SIZE = 11
# what every default figure and catalogue is made with: a run at another smoothing tags its outputs
# (dtfelib.pipeline.smoothing_tag) instead of overwriting these
PRODUCTION_SMOOTHING = (SMOOTHING_SIGMA_CELLS, FOOTPRINT_SIZE)


def footprint_for(sigma_cells):
    """The minima finder's footprint for a smoothing length in cells: the odd number of cells nearest 1.1 sigma
    (11 at the production 10), at least 3 -- so a smoothing fixed in Mpc finds minima in the same PHYSICAL window
    in every box (a fixed 11 cells is 3 sigma at TNG300's 3.66-cell 2.16 Mpc and suppresses minima TNG100 keeps)."""
    return max(3, 2 * int(round((1.1 * float(sigma_cells) - 1.0) / 2.0)) + 1)


# A smoothing length FIXED IN Mpc (2026-10-07): DTFE_SIGMA_MPC=2.16 smooths every box at the same physical scale --
# 10 cells is 1.0 / 2.2 / 5.9 Mpc in TNG50 / 100 / 300, so their catalogues were different voids -- and the footprint
# follows it (footprint_for). Here, at import, so every module-level copy follows; the void scripts' --smooth-mpc
# does the same at run time (dtfelib.pipeline.set_smoothing). FOR THE VOID SCRIPTS AND 'analyze.py export' (which
# tag their outputs): analyze.py compute/plot/all refuse to run under it, and no figure made under it is copied
# into the thesis Figures/ tree. The cells are rounded to 1e-9, so 10 x CELL_SIZE gives exactly 10.0 cells,
# footprint 11 and today's caches.
if os.environ.get("DTFE_SIGMA_MPC"):
    SMOOTHING_SIGMA_CELLS = round(float(os.environ["DTFE_SIGMA_MPC"]) / CELL_SIZE, 9)
    FOOTPRINT_SIZE = footprint_for(SMOOTHING_SIGMA_CELLS)
SMOOTHING_SIGMA_MPC = SMOOTHING_SIGMA_CELLS * CELL_SIZE

GRADIENT_THRESHOLD = 0.1

DEEP_VOID_THRESHOLD = -0.1
# The ellipsoid fit's resolution gate (a void's well_resolved flag) as RULES in the field's own scales; in Mpc
# only at call time, through dtfelib.pipeline.ellipsoid_cuts_mpc(), so a script's --smooth reaches it. Until
# 2026-10-07 the gate was a fixed 10 Mpc -- TNG50's 10 sigma -- which kept 2 of 495 TNG100 voids and 0 of 650
# TNG300 voids at z = 0: the semi-axes are curvature lengths of the smoothed field and grow with the smoothing.
# 10 sigma reproduces the TNG50 sample (+72/-0 of 8433 flags over the thesis snapshots; no thesis number used
# the gate: its tables average ALL voids) and keeps 103 (TNG100) and 409 (TNG300) voids at z = 0. The box
# clause: a semi-axis beyond L/2 overlaps its own periodic image (inert at sigma = 10 cells, binding above 25.6).
# The minimum axis (was 0.1 Mpc, one TNG50 cell) and the axis ratio never decide a void at sigma = 10 cells.
ELLIPSOID_CUT_RULES = {
    'min_axis_cells': 1.0,
    'max_axis_sigma': 10.0,
    'max_axis_box_frac': 0.5,
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
# The fallback is the 'Figures' folder NEXT TO the repo. _REPO_ROOT is DTFE/python (config.py moved there
# 2026-07-10), so that folder is two levels up: one level up is DTFE/Figures, which on macOS's case-insensitive
# disk IS the repo's own figures/ folder -- every mirrored figure from the move to 2026-10-06 landed there.
THESIS_FIGURES_DIR = (_THESIS_DIR / 'Figures') if _THESIS_DIR else (_REPO_ROOT.parent.parent / 'Figures')

LOCAL_FIGURES_ROOT = str(FIGURES_ROOT)     # dtfelib.cli: DTFE_FIGURES_ROOT, else the T7, else python/figures

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

# the density slice maps' colour range in rho/rho_bar, the same for both estimators so DTFE and PS-DTFE
# panels compare side by side (None for either end: the slice's own range); plot_DTFE.py and
# plot_PS_DTFE.py used to carry this pair each (survey item 11, 2026-10-05)
DENSITY_MAP_RANGE = (1e-1, 1e4)

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
    base = Path(base_dir) if base_dir else sim_dir(SIMULATION)
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
