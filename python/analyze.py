"""One driver for the python analysis stack.

    python3 analyze.py check                 # data availability + config consistency
    python3 analyze.py compute [snap ...]    # phase 1: shared products -> cache (once)
    python3 analyze.py plot [--only ...]     # phase 2: all figures from the cache
    python3 analyze.py all                   # compute (own subprocess) followed by plot
    python3 analyze.py export [snap ...]     # the void catalogues as CSV + HDF5 (figures/void_catalog/<sim>/)

Phase 1 does the expensive 512^3 work once per snapshot into python/cache/<param-hash>/;
phase 2 reads the cache. Changing sigma/footprint/criterion in config.py changes the hash
and triggers a recompute. Under 'all' the compute phase runs in its own subprocess, so its
working set is returned to the OS before the plot scripts start. Without snapshot ids,
compute and export take config's ladder where the snapdir exists.

The core figures' names carry neither the simulation nor the estimator, so the thesis
Figures/ tree takes them only from config.THESIS_SIMULATION's standard-DTFE grids
(config.THESIS_METHOD); --no-mirror copies nothing there.
"""

import argparse
import subprocess
import os
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

# the grids the core scripts read, after the estimator's prefix (ESTIMATOR_PREFIX)
FIELD_FILES = ['a_den', 'a_vel', 'a_velDiv', 'a_velShear', 'a_velGrad']
ESTIMATOR_PREFIX = {'dtfe': 'output.', 'ps': 'ps_output.'}
ESTIMATOR_NAME = {'dtfe': 'standard DTFE', 'ps': 'PS-DTFE'}


def grid_estimator(d):
    """The estimator the scripts' FieldSet(method='auto') loads from snapdir d: PS-DTFE when any ps_output.* field
    grid is there, else standard DTFE; None for neither (io.FieldSet._detect_method's rule: run logs do not count)."""
    from dtfelib import io
    known = {s for s, _, _ in io.FIELDS.values()} | set(io.LEGACY_SUFFIX.values())
    for m in ('ps', 'dtfe'):
        pre = ESTIMATOR_PREFIX[m]
        if any(p.name[len(pre):].removeprefix('a_') in known for p in Path(d).glob(pre + '*')):
            return m
    return None


def snapshots_on_disk():
    """The default snapshot list: config's ladder where the simulation has a snapdir, and the ids it has not. Each
    -3-Dark box has 9 of the 16, so the whole ladder made compute (and 'all') fail on the 7 absent ones (2026-10-08)."""
    from dtfelib import pipeline
    base = pipeline.sim_dir(config.SIMULATION)
    present = [s for s in config.SNAPSHOT_TO_REDSHIFT if (base / f"snapdir_{s}").is_dir()]
    return present, [s for s in config.SNAPSHOT_TO_REDSHIFT if s not in present]


# ---------------------------------------------------------------- check

def check_data():
    """Per config snapshot: absent (not on disk: compute and plot leave it out), or the estimator the scripts load
    there and whether its grids are complete. Until 2026-10-08 only standard DTFE's output.* counted, so a box with
    PS-DTFE grids alone (every box on the T7) read INCOMPLETE everywhere while compute and plot used those grids.
    'missing' = the incomplete snapdirs (exit 1), and no snapdir at all."""
    from dtfelib.cli import sim_dir
    base = sim_dir(config.SIMULATION)
    missing, absent_dirs, used = [], [], set()
    for snap, z in config.snapshot_items():
        d = base / f"snapdir_{snap}"
        if not d.is_dir():
            absent_dirs.append(snap)
            print(f"  snapdir_{snap}  (z={z:5.2f})  not on disk")
            continue
        m = grid_estimator(d)
        if m is None:
            missing.append(snap)
            print(f"  snapdir_{snap}  (z={z:5.2f})  INCOMPLETE: no field grids (ps_output.* or output.*)")
            continue
        used.add(m)
        absent = [ESTIMATOR_PREFIX[m] + ff for ff in FIELD_FILES if not (d / (ESTIMATOR_PREFIX[m] + ff)).exists()]
        if absent:
            missing.append(snap)
            print(f"  snapdir_{snap}  (z={z:5.2f})  INCOMPLETE ({ESTIMATOR_NAME[m]}): missing {', '.join(absent)}")
        else:
            print(f"  snapdir_{snap}  (z={z:5.2f})  OK (all {ESTIMATOR_NAME[m]} fields)")
    mism = config.verify_snapshot_redshifts()
    for snap, z_cfg, z_hdr in mism:
        print(f"  !! redshift mismatch snap {snap}: config {z_cfg} vs header {z_hdr}")
    if len(absent_dirs) == len(config.SNAPSHOT_TO_REDSHIFT):
        missing = absent_dirs                       # nothing on disk at all: every snapshot is missing, as before
        print(f"\n  !! no snapshot of the config ladder under {base}")
    elif absent_dirs:
        print(f"\n  not on disk (compute and plot leave them out): {', '.join(absent_dirs)}")
    if missing:
        print(f"\n  -> Re-run DTFE for: {', '.join(missing)}")
    if used and (used != {config.THESIS_METHOD} or config.SIMULATION != config.THESIS_SIMULATION):
        print(f"\n  The figures will be {config.SIMULATION} {' and '.join(ESTIMATOR_NAME[m] for m in sorted(used))}; "
              f"the thesis Figures/ tree takes {config.THESIS_SIMULATION} {ESTIMATOR_NAME[config.THESIS_METHOD]} "
              f"figures only.")
    return missing, mism


# ---------------------------------------------------------------- compute (phase 1)

def compute(snaps):
    from dtfelib import pipeline
    from dtfelib import figures as style

    # ids as config spells them ('99' -> '099'); an id config does not know is reported, not skipped quietly.
    # No ids: the ladder's snapshots ON DISK, the absent ones said (an id given and absent still fails)
    absent = []
    if snaps:
        snaps = [pipeline.canonical_snapshot_id(s) for s in snaps]
    else:
        snaps, absent = snapshots_on_disk()
    print(f"Phase 1 compute: {len(snaps)} snapshot(s)"
          + (f"; not on disk, left out: {', '.join(absent)}" if absent else ""))
    print(f"  sigma = {config.SMOOTHING_SIGMA_CELLS} cells "
          f"({config.SMOOTHING_SIGMA_MPC:.2f} Mpc), footprint = "
          f"{config.FOOTPRINT_SIZE}, criterion = {config.VOID_EIGENVALUE_CRITERION}")
    print(f"  cache: {pipeline.cache_dir()}\n")

    per_snap = {}                       # this run's amplitudes; the series' limits come from the store
    t_total = time.time()
    failed, unknown = [], []

    for snap in snaps:
        z = config.get_redshift(snap)
        if z is None:
            print(f"  !! unknown snapshot {snap} (not in config.SNAPSHOT_TO_REDSHIFT), skipped")
            unknown.append(snap)
            continue
        print(f"snapshot {snap} (z={z:.2f})")
        t0 = time.time()
        p = pipeline.products(snap, z)
        try:
            p.data_dir                  # builds the FieldSet: raises for a missing snapdir, no grids, no snapshot file,
        except Exception as e:          # an unreadable one (h5py: OSError) -- the old 'exists()' test sat after the raise
            print(f"    data missing or unreadable, skipped: {e}")
            failed.append(snap)
            continue
        try:
            p.warm()
            ds = p.delta_slices()
            es = p.eigen_slices()
            ps = p.phi_slices()
            per_snap[snap] = {
                'delta': max(style.robust_vmax(ds[d]) for d in ds),
                'eigenvalue': max(max(style.robust_vmax(es[d][k]) for k in ('lambda1', 'lambda2', 'lambda3'))
                                  for d in es),
                'potential': max(style.robust_vmax(ps[d]) for d in ps)}
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append(snap)
        finally:
            p.release()
        print(f"    done in {(time.time() - t0) / 60:.1f} min")

    # the global limits = the maximum over EVERY snapshot computed so far (the per-snapshot store), not
    # over this run's: a subset run used to overwrite the series' limits with its own (2026-10-06)
    limits, covered, missing = pipeline.record_limits(per_snap, sim=config.SIMULATION)
    if limits is not None:
        print(f"\nGlobal cross-epoch limits written for {config.SIMULATION}: {limits}")
        print(f"  from {len(covered)} of {len(covered) + len(missing)} canonical snapshots"
              + (f"; not computed yet: {', '.join(missing)} (run compute for them)" if missing else ""))
    else:
        print("\nNothing computed: the global limits are left as they are.")
    print(f"Total: {(time.time() - t_total) / 60:.1f} min")
    if unknown:
        print(f"Unknown snapshot ids: {', '.join(unknown)} (config.SNAPSHOT_TO_REDSHIFT knows "
              f"{', '.join(config.SNAPSHOT_TO_REDSHIFT)})")
    if failed:
        print(f"Failed snapshots: {', '.join(failed)}")
    if failed or unknown or not per_snap:
        sys.exit(1)


# ---------------------------------------------------------------- export (void catalogues)

def export(snaps, out_dir=None, formats=("csv", "hdf5")):
    """One CSV and one HDF5 per snapshot from the cached void catalogue (computed when not cached yet),
    with units and provenance (dtfelib.catalog), under figures/void_catalog/<sim>/ (or --out). Exit 1 on
    an unknown id, a failed snapshot or nothing exported (the honest-exit rule)."""
    from dtfelib import pipeline, catalog
    absent = []
    if snaps:
        snaps = [pipeline.canonical_snapshot_id(s) for s in snaps]
    else:
        snaps, absent = snapshots_on_disk()         # as compute: the ladder's snapshots on disk (2026-10-08)
    if absent:
        print(f"Not on disk, left out: {', '.join(absent)}")
    out = Path(out_dir).expanduser() if out_dir else \
        Path(config.LOCAL_FIGURES_ROOT) / "void_catalog" / config.SIMULATION / pipeline.smoothing_tag()   # DTFE_SIGMA_MPC: own folder
    print(f"Export: void catalogues of {len(snaps)} snapshot(s) of {config.SIMULATION} -> {out}")
    print(f"  sigma = {config.SMOOTHING_SIGMA_CELLS} cells, footprint = {config.FOOTPRINT_SIZE}, "
          f"criterion = {config.VOID_EIGENVALUE_CRITERION}; cache: {pipeline.cache_dir()}")
    failed, unknown, written = [], [], []
    t_total = time.time()
    for snap in snaps:
        z = config.get_redshift(snap)
        if z is None:
            print(f"  !! unknown snapshot {snap} (not in config.SNAPSHOT_TO_REDSHIFT), skipped")
            unknown.append(snap)
            continue
        p = pipeline.products(snap, z)
        t0 = time.time()
        try:
            p.data_dir                      # builds the FieldSet: raises for a missing snapdir, no grids, an unreadable file
        except Exception as e:
            print(f"  snapshot {snap}: data missing or unreadable, skipped: {e}")
            failed.append(snap)
            continue
        try:
            paths = catalog.export_voids(p, out, formats)
            n = len(p.voids()["coords"])
        except Exception as e:              # the frame guard (ValueError), a corrupt cache, memory
            print(f"  snapshot {snap}: FAILED: {e}")
            failed.append(snap)
            continue
        finally:
            p.release()
        written += paths
        print(f"  snapshot {snap} (z={z:.2f}): {n} voids -> {', '.join(q.name for q in paths)} ({time.time() - t0:.1f} s)")
    print(f"\n{len(written)} file(s) under {out}; total {(time.time() - t_total) / 60:.1f} min")
    if unknown:
        print(f"Unknown snapshot ids: {', '.join(unknown)}")
    if failed:
        print(f"Failed snapshots: {', '.join(failed)}")
    if failed or unknown or not written:
        sys.exit(1)


# ---------------------------------------------------------------- plot (phase 2)

def run_step(cmd, label, env=None):
    print(f"\n{'=' * 70}\nRunning {label}\n{'=' * 70}")
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    dt = time.time() - t0
    print(f"-> {label}: exit {proc.returncode} ({dt / 60:.1f} min)")
    return proc.returncode, dt


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', nargs='?', default='all',
                        choices=['check', 'compute', 'plot', 'all', 'export'])
    parser.add_argument('snaps', nargs='*',
                        help="snapshots for 'compute' and 'export' (default: config's ladder where the snapdir exists)")
    parser.add_argument('--only', nargs='*', default=None,
                        help='run only the named plot scripts')
    parser.add_argument('--out', default=None, help="'export': the output folder (default figures/void_catalog/<sim>)")
    parser.add_argument('--format', default='both', choices=['csv', 'hdf5', 'both'], help="'export': which files")
    parser.add_argument('--no-mirror', action='store_true',
                        help="copy nothing into the thesis Figures/ tree (it takes only "
                             f"{config.THESIS_SIMULATION} {ESTIMATOR_NAME[config.THESIS_METHOD]} figures anyway)")
    args = parser.parse_args()
    # --no-mirror reaches the plot subprocesses through their environment (dtfelib.figures reads DTFE_NO_MIRROR)
    child_env = dict(os.environ, DTFE_NO_MIRROR='1') if args.no_mirror else None

    script_dir = Path(__file__).resolve().parent

    print(f"Canonical dataset: {len(config.SNAPSHOT_TO_REDSHIFT)} snapshots "
          f"(z = {max(config.SNAPSHOT_TO_REDSHIFT.values())} .. "
          f"{min(config.SNAPSHOT_TO_REDSHIFT.values())})")
    print(f"Void criterion: {config.VOID_EIGENVALUE_CRITERION}, "
          f"footprint: {config.FOOTPRINT_SIZE}, "
          f"smoothing: {config.SMOOTHING_SIGMA_CELLS} cells "
          f"({config.SMOOTHING_SIGMA_MPC:.2f} Mpc)")
    print(f"Cache: {config.CACHE_DIR}")
    # the core figures' names carry neither the simulation nor the estimator: the thesis tree takes them only from
    # the thesis simulation's standard-DTFE grids (dtfelib.figures; the scripts say it when a grid set is refused)
    if args.no_mirror or os.environ.get('DTFE_NO_MIRROR'):
        mirror_off = "asked for"
    elif config.SIMULATION != config.THESIS_SIMULATION:
        mirror_off = f"{config.SIMULATION} is not the thesis simulation {config.THESIS_SIMULATION}"
    else:
        mirror_off = None
    print("Thesis figure mirror: " + (f"off ({mirror_off})" if mirror_off else
          f"{config.THESIS_FIGURES_DIR} (figures made from {config.THESIS_SIMULATION} "
          f"{ESTIMATOR_NAME[config.THESIS_METHOD]} grids only)") + "\n")

    if args.mode == 'check':
        print("Data check:")
        missing, mism = check_data()
        sys.exit(1 if (missing or mism) else 0)

    if args.mode == 'export':
        export(args.snaps, args.out, ('csv', 'hdf5') if args.format == 'both' else (args.format,))
        return
    from dtfelib import pipeline as _pl
    if _pl.smoothing_tag():
        # the thesis scripts save under the PRODUCTION names (and into the thesis Figures/ tree): a left-over
        # DTFE_SIGMA_MPC would overwrite them with another smoothing's figures and void_shape_table.txt
        sys.exit(f"refused: the smoothing is {config.SMOOTHING_SIGMA_CELLS:g} cells / footprint {config.FOOTPRINT_SIZE} "
                 f"(DTFE_SIGMA_MPC={os.environ.get('DTFE_SIGMA_MPC', '')!r}), not the production "
                 f"{config.PRODUCTION_SMOOTHING}: '{args.mode}' writes the thesis figures under their production names. "
                 f"Unset DTFE_SIGMA_MPC; it is for the void scripts and 'export' (both tag their outputs).")
    if args.mode == 'compute':
        compute(args.snaps)
        return

    results = {}

    if args.mode == 'all':
        # compute in its own subprocess so its working set is released before plotting
        code, dt = run_step([sys.executable, __file__, 'compute'] + args.snaps,
                            'analyze.py compute', env=child_env)
        results['compute'] = (code, dt)
        if code != 0:
            print("\ncompute failed; aborting before the plot phase.")
            sys.exit(1)

    if args.mode in ('plot', 'all'):
        scripts = args.only if args.only else PLOT_SCRIPTS
        for script in scripts:
            results[script] = run_step(
                [sys.executable, str(script_dir / 'plot' / script)], script, env=child_env)

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
    print("\nAll done. " + (f"Nothing was copied into the thesis Figures/ tree ({mirror_off}); " if mirror_off else
                            "Figures made from the thesis grids are mirrored into the thesis Figures/ tree; ")
          + f"key numbers in {config.LOCAL_FIGURES_ROOT}/void_analysis/void_shape_table.txt "
          "and shear_analysis/shear_stats.txt.")


if __name__ == '__main__':
    main()
