"""One driver for the python analysis stack.

    python3 analyze.py check                 # data availability + config consistency
    python3 analyze.py compute [snap ...]    # phase 1: shared products -> cache (once)
    python3 analyze.py plot [--only ...]     # phase 2: all figures from the cache
    python3 analyze.py all                   # compute (own subprocess) followed by plot

Phase 1 does the expensive 512^3 work once per snapshot into python/cache/<param-hash>/;
phase 2 reads the cache. Changing sigma/footprint/criterion in config.py changes the hash
and triggers a recompute. Under 'all' the compute phase runs in its own subprocess, so its
working set is returned to the OS before the plot scripts start.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# Core figure set only: the tracking/synthesis scripts (plot_PS_DTFE, plot_cosmic_web,
# plot_halo_tracking, plot_void_tracking, plot_void_population, plot_conclusions_synthesis)
# need PS-DTFE grids and/or merger-tree data and are run separately.
PLOT_SCRIPTS = [
    'plot_DTFE.py',
    'plot_velDiv_den.py',
    'plot_eigenvalues.py',
    'plot_shear_triaxial.py',
    'plot_PDF_CDF.py',
    'plot_contour_filter.py',
    'plot_shape_filter.py',
    'plot_ellipses_contour_3D.py',
    'plot_marked_correlation_BBKS.py',
    'plot_smoothing_comparison.py',
    'plot_phi_delta.py',
    'plot_tidal_correlation.py'
]

FIELD_FILES = ['output.a_den', 'output.a_vel', 'output.a_velDiv',
               'output.a_velShear', 'output.a_velGrad']


# ---------------------------------------------------------------- check

def check_data():
    from dtfelib.cli import DATA_ROOT
    base = DATA_ROOT / config.SIMULATION
    missing = []
    for snap, z in config.snapshot_items():
        d = base / f"snapdir_{snap}"
        if not d.exists():
            missing.append(snap)
            print(f"  snapdir_{snap}  (z={z:5.2f})  MISSING (whole directory)")
            continue
        absent = [ff for ff in FIELD_FILES if not (d / ff).exists()]
        if absent:
            missing.append(snap)
            print(f"  snapdir_{snap}  (z={z:5.2f})  INCOMPLETE: missing {', '.join(absent)}")
        else:
            print(f"  snapdir_{snap}  (z={z:5.2f})  OK (all fields)")
    mism = config.verify_snapshot_redshifts()
    for snap, z_cfg, z_hdr in mism:
        print(f"  !! redshift mismatch snap {snap}: config {z_cfg} vs header {z_hdr}")
    if missing:
        print(f"\n  -> Re-run DTFE for: {', '.join(missing)}")
    return missing, mism


# ---------------------------------------------------------------- compute (phase 1)

def _suggest_limits(per_snapshot):
    import numpy as np
    limits = {}
    for field, values in per_snapshot.items():
        vals = [v for v in values if v is not None and np.isfinite(v)]
        if vals:
            limits[field] = float(np.max(vals))
    return limits


def compute(snaps):
    from dtfelib import pipeline
    from dtfelib import figures as style

    snaps = snaps or list(config.SNAPSHOT_TO_REDSHIFT)
    print(f"Phase 1 compute: {len(snaps)} snapshot(s)")
    print(f"  sigma = {config.SMOOTHING_SIGMA_CELLS} cells "
          f"({config.SMOOTHING_SIGMA_MPC:.2f} Mpc), footprint = "
          f"{config.FOOTPRINT_SIZE}, criterion = {config.VOID_EIGENVALUE_CRITERION}")
    print(f"  cache: {pipeline.cache_dir()}\n")

    amp = {'delta': [], 'eigenvalue': [], 'potential': []}
    t_total = time.time()
    failed = []

    for snap in snaps:
        z = config.get_redshift(snap)
        if z is None:
            print(f"  !! unknown snapshot {snap}, skipped")
            continue
        print(f"snapshot {snap} (z={z:.2f})")
        t0 = time.time()
        p = pipeline.products(snap, z)
        if not p.data_dir.exists():
            print("    data directory missing, skipped")
            failed.append(snap)
            continue
        try:
            p.warm()
            ds = p.delta_slices()
            es = p.eigen_slices()
            ps = p.phi_slices()
            amp['delta'].append(max(style.robust_vmax(ds[d]) for d in ds))
            amp['eigenvalue'].append(max(
                max(style.robust_vmax(es[d][k]) for k in
                    ('lambda1', 'lambda2', 'lambda3')) for d in es))
            amp['potential'].append(max(style.robust_vmax(ps[d]) for d in ps))
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append(snap)
        finally:
            p.release()
        print(f"    done in {(time.time() - t0) / 60:.1f} min")

    limits = _suggest_limits(amp)
    pipeline.save_global_limits(limits)
    print(f"\nGlobal cross-epoch limits written: {limits}")
    print(f"Total: {(time.time() - t_total) / 60:.1f} min")
    if failed:
        print(f"Failed snapshots: {', '.join(failed)}")
        sys.exit(1)


# ---------------------------------------------------------------- plot (phase 2)

def run_step(cmd, label):
    print(f"\n{'=' * 70}\nRunning {label}\n{'=' * 70}")
    t0 = time.time()
    proc = subprocess.run(cmd)
    dt = time.time() - t0
    print(f"-> {label}: exit {proc.returncode} ({dt / 60:.1f} min)")
    return proc.returncode, dt


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', nargs='?', default='all',
                        choices=['check', 'compute', 'plot', 'all'])
    parser.add_argument('snaps', nargs='*',
                        help="snapshots for 'compute' (default: all in config)")
    parser.add_argument('--only', nargs='*', default=None,
                        help='run only the named plot scripts')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    print(f"Canonical dataset: {len(config.SNAPSHOT_TO_REDSHIFT)} snapshots "
          f"(z = {max(config.SNAPSHOT_TO_REDSHIFT.values())} .. "
          f"{min(config.SNAPSHOT_TO_REDSHIFT.values())})")
    print(f"Void criterion: {config.VOID_EIGENVALUE_CRITERION}, "
          f"footprint: {config.FOOTPRINT_SIZE}, "
          f"smoothing: {config.SMOOTHING_SIGMA_CELLS} cells "
          f"({config.SMOOTHING_SIGMA_MPC:.2f} Mpc)")
    print(f"Cache: {config.CACHE_DIR}")
    print(f"Thesis figure mirror: {config.THESIS_FIGURES_DIR}\n")

    if args.mode == 'check':
        print("Data check:")
        missing, mism = check_data()
        sys.exit(1 if (missing or mism) else 0)

    if args.mode == 'compute':
        compute(args.snaps)
        return

    results = {}

    if args.mode == 'all':
        # compute in its own subprocess so its working set is released before plotting
        code, dt = run_step([sys.executable, __file__, 'compute'] + args.snaps,
                            'analyze.py compute')
        results['compute'] = (code, dt)
        if code != 0:
            print("\ncompute failed; aborting before the plot phase.")
            sys.exit(1)

    if args.mode in ('plot', 'all'):
        scripts = args.only if args.only else PLOT_SCRIPTS
        for script in scripts:
            results[script] = run_step(
                [sys.executable, str(script_dir / 'plot' / script)], script)

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    failed = []
    for label, (code, dt) in results.items():
        status = 'OK    ' if code == 0 else 'FAILED'
        if code != 0:
            failed.append(label)
        print(f"  {status} {label:38s} {dt / 60:6.1f} min")
    if failed:
        print(f"\n{len(failed)} step(s) failed: {', '.join(failed)}")
        sys.exit(1)
    print("\nAll done. Figures are mirrored into the thesis Figures/ tree; "
          "key numbers in python/figures/void_analysis/void_shape_table.txt "
          "and shear_analysis/shear_stats.txt.")


if __name__ == '__main__':
    main()
