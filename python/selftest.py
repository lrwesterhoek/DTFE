"""Smoke test of the python analysis stack on a synthetic dataset.

Builds a fake two-snapshot DTFE output set in a temp dir, then runs the compute phase and
each core plot script against it. Every target runs in its OWN subprocess (fresh
interpreter), so matplotlib/numpy memory is returned to the OS after each step --
peak RSS stays at a single script's footprint instead of accumulating across all of
them. The child mode (--run-one) applies the config overrides and runs one target.

    python3 python/selftest.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE / 'plot'))

N = 64
SNAPS = ['000', '099']
REDSHIFTS = {'000': 20.05, '099': 0.00}
SIM = 'TNG50-3-Dark'
BOX_CKPC = 51668.142899320934   # matches config.SIMULATION_BOX_MPC[SIM] (h-free ckpc)

# The compute phase first (produces the caches the plots read); then the core figure
# scripts. The tracking/synthesis scripts (plot_PS_DTFE, plot_cosmic_web,
# plot_halo_tracking, plot_void_*, plot_conclusions_synthesis) need PS grids and/or
# merger trees and are exercised by tests/py_dtfelib_test.py instead.
# 'module:arg' runs the module's main() with that single CLI argument.
TARGETS = [
    'analyze:compute',
    'plot_DTFE', 'plot_velDiv_den', 'plot_eigenvalues', 'plot_shear_triaxial',
    'plot_PDF_CDF', 'plot_contour_filter', 'plot_shape_filter',
    'plot_ellipses_contour_3D', 'plot_marked_correlation_BBKS',
    'plot_smoothing_comparison', 'plot_phi_delta', 'plot_tidal_correlation',
]


# ---------------------------------------------------------------- synthetic data

def make_field(rng, n, amplitude):
    f = rng.standard_normal((n, n, n)).astype(np.float32)
    f = gaussian_filter(f, sigma=3, mode='wrap')
    f -= f.mean()
    f /= max(f.std(), 1e-12)
    return (amplitude * f).astype(np.float32)


def write_fake_snapshot(snapdir, snap, rng, amplitude):
    snapdir.mkdir(parents=True, exist_ok=True)
    delta = make_field(rng, N, amplitude)
    density = (1.0 + np.clip(delta, -0.9, None)).astype(np.float32)
    density.tofile(snapdir / 'output.a_den')
    for name, ncomp, amp in [('output.a_vel', 3, 200.0),
                             ('output.a_velDiv', 1, 150.0),
                             ('output.a_velShear', 5, 80.0),
                             ('output.a_velGrad', 9, 80.0)]:
        arr = np.stack([make_field(rng, N, amp) for _ in range(ncomp)], axis=-1)
        if ncomp == 1:
            arr = arr[..., 0]
        arr.astype(np.float32).tofile(snapdir / name)

    # minimal combined_*.hdf5: FieldSet reads its units/metadata from this header
    import h5py
    with h5py.File(snapdir / f'combined_{snap}.hdf5', 'w') as f:
        h = f.create_group('Header')
        h.attrs['BoxSize'] = BOX_CKPC
        h.attrs['Redshift'] = REDSHIFTS[snap]
        h.attrs['HubbleParam'] = 0.6774
        h.attrs['NumPart_ThisFile'] = np.array([0, N**3, 0, 0, 0, 0], dtype=np.int64)
        h.attrs['MassTable'] = np.array([0.0, 1e-3, 0.0, 0.0, 0.0, 0.0])   # authored h-free
        h.attrs['HFreeUnits'] = 0.6774   # marker: FieldSet must not h-correct this MassTable


# ---------------------------------------------------------------- child: one target

def run_one(target, tmp):
    """Apply the selftest config overrides, then import and run a single target."""
    import matplotlib
    matplotlib.use('Agg')

    # The data-root/simulation sandbox comes from the DTFE_DATA_ROOT / DTFE_SIM env vars
    # set by the parent (dtfelib.cli and config read them at import, so pipeline paths,
    # BASE_DATA_DIR and the --sim defaults all resolve inside tmp). Patch here only what
    # has no environment hook.
    import config
    assert config.SIMULATION == SIM, "selftest child must run with DTFE_SIM set"
    config.FIELD_RESOLUTION = N
    config.CELL_SIZE = config.BOX_SIZE / N
    config.SMOOTHING_SIGMA_CELLS = 3.0
    config.SMOOTHING_SIGMA_MPC = 3.0 * config.CELL_SIZE
    config.FOOTPRINT_SIZE = 5
    config.SNAPSHOT_TO_REDSHIFT = dict(REDSHIFTS)
    config.PANEL_SNAPSHOTS = SNAPS
    config.CACHE_DIR = tmp / 'cache'
    config.THESIS_FIGURES_DIR = tmp / 'Figures'
    config.LOCAL_FIGURES_ROOT = str(tmp / 'python' / 'figures')

    # config is patched BEFORE the target module is imported, so module-level constants
    # (OUTPUT_DIR, smoothing, the snapshot ladder, ...) capture the sandbox values naturally.
    import importlib
    name, _, sub = target.partition(':')
    mod = importlib.import_module(name)
    sys.argv = [name] + ([sub] if sub else [])
    mod.main()


# ---------------------------------------------------------------- parent: orchestrate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-one', metavar='TARGET', help=argparse.SUPPRESS)
    ap.add_argument('--tmp', metavar='DIR', help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.run_one:
        run_one(args.run_one, Path(args.tmp))
        return

    tmp = Path(tempfile.mkdtemp(prefix='voids_selftest_'))
    print(f"Selftest workspace: {tmp}")
    try:
        rng = np.random.default_rng(42)
        for snap, amp in zip(SNAPS, (0.05, 0.8)):
            write_fake_snapshot(tmp / 'output' / SIM / f'snapdir_{snap}', snap, rng, amp)

        # sandbox: dtfelib.cli / config resolve the data root and simulation from these
        child_env = {**os.environ,
                     'DTFE_DATA_ROOT': str(tmp / 'output'),
                     'DTFE_SIM': SIM}

        # per-target wall-clock cap: a hung or runaway script is killed and reported
        # instead of stalling the whole selftest (SELFTEST_TIMEOUT overrides, seconds)
        timeout = float(os.environ.get('SELFTEST_TIMEOUT', 600))

        failures = []
        for target in TARGETS:
            print(f"\n--- {target} ---", flush=True)
            try:
                r = subprocess.run([sys.executable, __file__,
                                    '--run-one', target, '--tmp', str(tmp)],
                                   env=child_env, timeout=timeout)
            except subprocess.TimeoutExpired:
                failures.append((target, f'timeout after {timeout:.0f}s (killed)'))
                continue
            if r.returncode == 0:
                print(f"{target}: OK")
            else:
                failures.append((target, f'exit {r.returncode}'))

        n_pngs = len(list((tmp / 'Figures').rglob('*.png')))
        print(f"\nMirrored figures produced: {n_pngs}")
        if n_pngs == 0:
            failures.append(('mirror', 'no figures in thesis mirror'))

        print('\n' + '=' * 60)
        if failures:
            print(f"SELFTEST FAILED ({len(failures)}):")
            for name, why in failures:
                print(f"  - {name}: {why}")
            sys.exit(1)
        print(f"SELFTEST PASSED: pipeline + all {len(TARGETS) - 1} plot scripts ran clean.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
