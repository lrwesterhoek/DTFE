"""One driver for the python analysis stack.

    python3 analyze.py check                 # data availability + config consistency
    python3 analyze.py compute [snap ...]    # phase 1: shared products -> cache (once)
    python3 analyze.py plot [--only ...]     # phase 2: all figures from the cache
    python3 analyze.py all                   # compute (own subprocess) followed by plot
    python3 analyze.py export [snap ...]     # the void catalogues as CSV + HDF5 (figures/void_catalog/<sim>/)
    python3 analyze.py voids [--only ...]    # the void-population scripts (VOID_SCRIPTS) for config's simulation
    python3 analyze.py cache [--prune [--yes]]   # the cache namespaces, sizes, and which no current setting reaches

Phase 1 does the expensive 512^3 work once per snapshot into python/cache/<param-hash>/;
phase 2 reads the cache. Changing sigma/footprint/criterion in config.py changes the hash
and triggers a recompute. Under 'all' the compute phase runs in its own subprocess, so its
working set is returned to the OS before the plot scripts start. Without snapshot ids,
compute and export take config's ladder where the snapdir exists.

The core figures' names carry neither the simulation nor the estimator, so the thesis
Figures/ tree takes them only from config.THESIS_SIMULATION's standard-DTFE grids
(config.THESIS_METHOD); --no-mirror copies nothing there.

'voids' runs VOID_SCRIPTS with --sim config.SIMULATION (DTFE_SIM): each writes per-simulation folders and tags a
non-default smoothing itself, so DTFE_SIGMA_MPC is allowed there. A script named there (or in --only) that is not in
python/plot yet is skipped with a note, not failed (2026-10-11: the profile-evolution, environment and fate figures
were written alongside).

'cache' (2026-10-11) lists python/cache's namespaces with their sizes and whether today's code can still reach
them (cache_inventory: an older '_v<N>' format, or a smoothing no current setting produces); '--prune' is a DRY RUN
listing what would be deleted and its size, and only '--prune --yes' deletes. '--keep-mpc' / '--keep-cells' count
more smoothings as current.
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

# The void-population scripts, run by 'voids' (2026-10-11), in this order: the catalogue statistics first (the
# population script also draws the size function), then the profiles, their evolution, the environment (S/R
# types), the fate census and the convergence check of the void statistics across resolution and estimator
# (plot_void_convergence.py, contract K4, 2026-10-11 afternoon; default arguments '--sim <SIM>' like the others).
# Registered by name; a missing one is skipped with a note (run_scripts).
VOID_SCRIPTS = [
    'plot_void_population.py',
    'plot_void_profiles.py',
    'plot_void_profile_evolution.py',
    'plot_void_environment.py',
    'plot_void_fate.py',
    'plot_void_convergence.py',
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


# ---------------------------------------------------------------- cache (list / prune the namespaces)

# The physical smoothings (Mpc) today's code asks for by default: the cross-box comparison's sigma
# (plot_void_crossbox.DEFAULT_SIGMA_MPC, runspec.CROSSBOX_SMOOTH_MPC, scripts/thesis_runs.sh). A namespace made with
# another Mpc value (2.16 on 2026-10-11 morning: r512_s21.4043_f23 next to 2.1625's r512_s21.4291_f23) is reached only
# by typing that value again; 'cache --keep-mpc' and DTFE_SIGMA_MPC add values.
CACHE_SMOOTHING_MPC = (2.1625,)
NAMESPACE_RE = r"^r(?P<res>\d+)_s(?P<sig>[^_]+)_f(?P<fp>\d+)_(?P<crit>.+)_g(?P<g>[^_]+)_v(?P<ver>\d+)$"


def _dir_size(path):
    """(bytes, files) under 'path' (files only, links not followed)."""
    total, n = 0, 0
    for p in Path(path).rglob('*'):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
            n += 1
    return total, n


def _human(nbytes):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if nbytes < 1024 or unit == 'GB':
            return f"{nbytes:.0f} {unit}" if unit == 'B' else f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, OverflowError):
        return True                                 # another user's process, or not a pid: not ours to call dead
    return True


def cache_inventory(root=None, keep_mpc=(), keep_cells=()):
    """Every entry of the pipeline cache (config.CACHE_DIR) with a verdict (2026-10-11, contract K5): a list of
    {'path', 'kind' ('namespace' / 'file'), 'bytes', 'files', 'newest', 'prune' (bool), 'reason'}.

    A NAMESPACE (r<N>_s<sigma>_f<footprint>_<criterion>_g<threshold>_v<version>, pipeline._param_hash) is UNREACHABLE,
    and pruned, when
      * its version is not today's ('_v1': the format before the June unit overhaul; nothing reads it), or
      * no smoothing today's code asks for produces its sigma/footprint: the production one (config.PRODUCTION_SMOOTHING),
        a --smooth in cells with the production footprint for config.SMOOTHING_COMPARISON_SIGMAS (+ keep_cells), or a
        smoothing in Mpc -- CACHE_SMOOTHING_MPC, DTFE_SIGMA_MPC and keep_mpc -- for a simulation whose files it holds
        (round(mpc / cell, 9) cells, config.footprint_for), the way config / pipeline.set_smoothing spell it.
    Another resolution, criterion or gradient threshold is a config change away: kept. Inside a kept namespace a FILE
    is pruned when it is an interrupted save's temp file ('<name>.npz.<pid>.tmp.npz', pipeline._save_npz) whose
    process is gone, or a fate link (lagrangian) of an older LINK_VERSION, which cached_link never serves."""
    import re
    root = Path(config.CACHE_DIR if root is None else root)
    out = []
    if not root.is_dir():
        return out
    from dtfelib import pipeline
    m_now = re.match(NAMESPACE_RE, pipeline._param_hash())
    sig0, fp0 = config.PRODUCTION_SMOOTHING
    cells_ok = {f"{float(c):g}" for c in (sig0, *config.SMOOTHING_COMPARISON_SIGMAS, *keep_cells)}
    mpcs = list(CACHE_SMOOTHING_MPC) + [float(m) for m in keep_mpc]
    if os.environ.get('DTFE_SIGMA_MPC'):
        mpcs.append(float(os.environ['DTFE_SIGMA_MPC']))
    try:
        from dtfelib.lagrangian import LINK_VERSION
    except ImportError:                             # h5py / scipy missing: the link check is left out, not guessed
        LINK_VERSION = None
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = re.match(NAMESPACE_RE, d.name)
        nbytes, nfiles = _dir_size(d)
        files = [p for p in d.iterdir() if p.is_file()]
        newest = max((p.stat().st_mtime for p in files), default=d.stat().st_mtime)
        entry = {'path': d, 'kind': 'namespace', 'bytes': nbytes, 'files': nfiles, 'newest': newest}
        if m is None:
            out.append(dict(entry, prune=False, reason="not a cache namespace name: left alone"))
            continue
        sims = sorted({p.name.split('_')[0] for p in files if p.suffix == '.npz' and not p.name[:1].isdigit()})
        if m['ver'] != m_now['ver']:
            out.append(dict(entry, prune=True, reason=f"cache format v{m['ver']}, today's is v{m_now['ver']}: nothing "
                                                       "reads it"))
            continue
        if (m['res'], m['crit'], m['g']) != (m_now['res'], m_now['crit'], m_now['g']):
            out.append(dict(entry, prune=False, reason="another resolution / criterion / gradient threshold (config): "
                                                        "kept"))
            continue
        fp = int(m['fp'])
        why = None
        if m['sig'] in cells_ok and fp == int(fp0):
            why = "production smoothing" if m['sig'] == f"{float(sig0):g}" else f"--smooth {m['sig']} (cells)"
        else:
            for sim in sims:
                box = config.SIMULATION_BOX_MPC.get(sim)
                if box is None:
                    continue
                for mpc in mpcs:
                    cells = round(mpc / (box / config.FIELD_RESOLUTION), 9)
                    if f"{cells:g}" == m['sig'] and config.footprint_for(cells) == fp:
                        why = f"--smooth-mpc {mpc:g} on {sim}"
        if why is None:
            held = ', '.join(sims) if sims else 'no simulation files'
            out.append(dict(entry, prune=True, reason=f"sigma {m['sig']} cells / footprint {fp}: no current smoothing "
                                                       f"(cells {sorted(cells_ok, key=float)}, Mpc {mpcs}) gives it for "
                                                       f"{held}"))
            continue
        out.append(dict(entry, prune=False, reason=why))
        for p in sorted(files):
            st = p.stat()
            fe = {'path': p, 'kind': 'file', 'bytes': st.st_size, 'files': 1, 'newest': st.st_mtime}
            t = re.match(r"^.+\.npz\.(\d+)\.tmp\.npz$", p.name)
            if t and not _pid_alive(t.group(1)):
                out.append(dict(fe, prune=True, reason=f"an interrupted save's temp file (pid {t.group(1)} is gone)"))
            elif LINK_VERSION is not None and re.match(r"^.+_fate(_excl)?_\d{3}_\d{3}\.npz$", p.name):
                try:
                    import numpy as np
                    with np.load(p) as z:
                        v = int(z['link_version']) if 'link_version' in z.files else 1
                except Exception as e:              # a cut file: cached_link relinks it too
                    out.append(dict(fe, prune=True, reason=f"an unreadable fate link ({type(e).__name__})"))
                    continue
                if v != LINK_VERSION:
                    out.append(dict(fe, prune=True, reason=f"fate link version {v}, today's is {LINK_VERSION}: never "
                                                           "served (relinked on the next census)"))
    return out


def cache_command(prune=False, yes=False, keep_mpc=(), keep_cells=(), root=None):
    """'analyze.py cache': every namespace with its size and verdict; --prune lists only what would go (a DRY RUN);
    --prune --yes deletes it. Returns the bytes listed for (or freed by) the prune."""
    import shutil
    root = Path(config.CACHE_DIR if root is None else root)
    inv = cache_inventory(root, keep_mpc, keep_cells)
    print(f"Pipeline cache {root}" + ("" if root.is_dir() else " (absent)"))
    gone = [e for e in inv if e['prune']]
    # the files of one namespace with one reason are one line (27 old fate links are not 27 lines)
    shown, groups = [], {}
    for e in (gone if prune else inv):
        if e['kind'] == 'namespace':
            shown.append(e)
        elif e['prune']:
            key = (e['path'].parent, e['reason'])
            if key not in groups:
                groups[key] = dict(e, bytes=0, files=0, newest=0.0, names=[])
                shown.append(groups[key])
            g = groups[key]
            g['bytes'] += e['bytes']
            g['files'] += 1
            g['newest'] = max(g['newest'], e['newest'])
            g['names'].append(e['path'].name)
    for e in shown:
        if e['kind'] == 'namespace':
            name = e['path'].name
        else:
            ex = e['names'][0] + (f" (+{len(e['names']) - 1} more)" if len(e['names']) > 1 else "")
            name = f"  {e['path'].parent.name}/{ex}"
        mark = ("DELETE " if yes else "would delete ") if (prune and e['prune']) else ("prune  " if e['prune'] else "keep   ")
        print(f"  {mark}{name:44s} {_human(e['bytes']):>9s} {e['files']:5d} file(s), newest "
              f"{time.strftime('%Y-%m-%d', time.localtime(e['newest']))}: {e['reason']}")
    total = sum(e['bytes'] for e in gone)
    if not gone:
        print("  nothing to prune")
        return 0
    if not prune:
        print(f"\n{len(gone)} unreachable entr{'y' if len(gone) == 1 else 'ies'}, {_human(total)}: "
              "'analyze.py cache --prune' lists them, '--prune --yes' deletes them.")
        return total
    if not yes:
        print(f"\nDRY RUN: {len(gone)} entr{'y' if len(gone) == 1 else 'ies'}, {_human(total)} would be deleted; "
              "add --yes to delete them.")
        return total
    freed = 0
    for e in gone:
        p = e['path']
        if root.resolve() not in p.resolve().parents:   # never outside the cache, whatever a link points at
            print(f"  !! {p} is not inside {root}: left alone")
            continue
        try:
            shutil.rmtree(p) if e['kind'] == 'namespace' else p.unlink()
            freed += e['bytes']
        except OSError as err:
            print(f"  !! {p}: {err}")
    print(f"\nDeleted {_human(freed)} of {_human(total)}.")
    return freed


# ---------------------------------------------------------------- plot (phase 2)

def run_step(cmd, label, env=None):
    print(f"\n{'=' * 70}\nRunning {label}\n{'=' * 70}")
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    dt = time.time() - t0
    print(f"-> {label}: exit {proc.returncode} ({dt / 60:.1f} min)")
    return proc.returncode, dt


def run_scripts(scripts, script_dir, results, env=None, extra=()):
    """Run each plot script (python/plot/<name> + extra) into 'results' {label: (exit, seconds)}; a script not on
    disk is noted and left out of results (a registered figure whose script is not written yet: not a failure).
    Returns the skipped names."""
    skipped = []
    for script in scripts:
        path = script_dir / 'plot' / script
        if not path.is_file():
            print(f"\n{script}: not in python/plot (yet): skipped")
            skipped.append(script)
            continue
        results[script] = run_step([sys.executable, str(path), *extra], script, env=env)
    return skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', nargs='?', default='all',
                        choices=['check', 'compute', 'plot', 'all', 'export', 'voids', 'cache'])
    parser.add_argument('snaps', nargs='*',
                        help="snapshots for 'compute' and 'export' (default: config's ladder where the snapdir exists)")
    parser.add_argument('--only', nargs='*', default=None,
                        help="run only the named plot scripts (in 'voids' mode: these instead of VOID_SCRIPTS)")
    parser.add_argument('--out', default=None, help="'export': the output folder (default figures/void_catalog/<sim>)")
    parser.add_argument('--format', default='both', choices=['csv', 'hdf5', 'both'], help="'export': which files")
    parser.add_argument('--no-mirror', action='store_true',
                        help="copy nothing into the thesis Figures/ tree (it takes only "
                             f"{config.THESIS_SIMULATION} {ESTIMATOR_NAME[config.THESIS_METHOD]} figures anyway)")
    parser.add_argument('--prune', action='store_true',
                        help="'cache': list only what is unreachable, as a DRY RUN (nothing is deleted without --yes)")
    parser.add_argument('--yes', action='store_true', help="'cache --prune': really delete what it lists")
    parser.add_argument('--keep-mpc', type=float, nargs='*', default=[], metavar='MPC',
                        help=f"'cache': more smoothings in Mpc to count as current (besides {list(CACHE_SMOOTHING_MPC)} "
                             "and DTFE_SIGMA_MPC)")
    parser.add_argument('--keep-cells', type=float, nargs='*', default=[], metavar='CELLS',
                        help="'cache': more smoothings in cells (footprint 11) to count as current")
    args = parser.parse_args()
    if args.yes and not (args.mode == 'cache' and args.prune):
        parser.error("--yes goes with 'cache --prune' only")
    if args.mode != 'cache' and (args.prune or args.keep_mpc or args.keep_cells):
        parser.error("--prune / --keep-mpc / --keep-cells go with 'cache' only")
    if args.mode == 'cache':
        # before the banner and the gate/smoothing refusals: it reads no grid and writes no figure
        cache_command(args.prune, args.yes, args.keep_mpc, args.keep_cells)
        return
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
    results, skipped = {}, []
    if args.mode == 'voids':
        # before the smoothing refusal: these scripts tag a non-default smoothing's files and folders themselves
        skipped = run_scripts(args.only if args.only else VOID_SCRIPTS, script_dir, results, env=child_env,
                              extra=('--sim', str(config.SIMULATION)))
        summarize(results, skipped, "Void figures are in per-simulation folders under "
                                    f"{config.LOCAL_FIGURES_ROOT} (void_population/, void_profiles/, ...).")
        return
    from dtfelib import pipeline as _pl
    try:
        gate = _pl.void_gate()
    except ValueError as e:
        sys.exit(f"refused: {e}")
    if gate != 'ellipsoid' and args.mode in ('plot', 'all'):
        # checked apart from the smoothing (2026-10-11): smoothing_tag() carries 'gatenone' too, and the smoothing
        # refusal below blamed DTFE_SIGMA_MPC at the production smoothing. 'compute' stays allowed: its caches (the
        # void catalogue keeps the production gate, gate_view is never saved; slices, limits) do not depend on it
        sys.exit(f"refused: DTFE_VOID_GATE={os.environ.get('DTFE_VOID_GATE', '')!r} (void gate '{gate}', not the "
                 f"production 'ellipsoid'): '{args.mode}' writes the thesis figures under their production names. "
                 f"Unset DTFE_VOID_GATE; it is for the void scripts (they tag their outputs 'gate{gate}').")
    sig0, fp0 = config.PRODUCTION_SMOOTHING
    if (float(config.SMOOTHING_SIGMA_CELLS), int(config.FOOTPRINT_SIZE)) != (float(sig0), int(fp0)):
        # the thesis scripts save under the PRODUCTION names (and into the thesis Figures/ tree): a left-over
        # DTFE_SIGMA_MPC would overwrite them with another smoothing's figures and void_shape_table.txt (the same
        # test as smoothing_tag()'s smoothing part)
        sys.exit(f"refused: the smoothing is {config.SMOOTHING_SIGMA_CELLS:g} cells / footprint {config.FOOTPRINT_SIZE} "
                 f"(DTFE_SIGMA_MPC={os.environ.get('DTFE_SIGMA_MPC', '')!r}), not the production "
                 f"{config.PRODUCTION_SMOOTHING}: '{args.mode}' writes the thesis figures under their production names. "
                 f"Unset DTFE_SIGMA_MPC; it is for the void scripts and 'export' (both tag their outputs).")
    if args.mode == 'compute':
        compute(args.snaps)
        return

    if args.mode == 'all':
        # compute in its own subprocess so its working set is released before plotting
        code, dt = run_step([sys.executable, __file__, 'compute'] + args.snaps,
                            'analyze.py compute', env=child_env)
        results['compute'] = (code, dt)
        if code != 0:
            print("\ncompute failed; aborting before the plot phase.")
            sys.exit(1)

    if args.mode in ('plot', 'all'):
        skipped = run_scripts(args.only if args.only else PLOT_SCRIPTS, script_dir, results, env=child_env)

    summarize(results, skipped,
              (f"Nothing was copied into the thesis Figures/ tree ({mirror_off}); " if mirror_off else
               "Figures made from the thesis grids are mirrored into the thesis Figures/ tree; ")
              + f"key numbers in {config.LOCAL_FIGURES_ROOT}/void_analysis/void_shape_table.txt "
              "and shear_analysis/shear_stats.txt.")


def summarize(results, skipped, done_note):
    """The summary table; exit 1 when a step failed. A skipped (missing) script is listed, not failed."""
    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    failed = []
    for label, (code, dt) in results.items():
        status = 'OK    ' if code == 0 else 'FAILED'
        if code != 0:
            failed.append(label)
        print(f"  {status} {label:38s} {dt / 60:6.1f} min")
    for label in skipped:
        print(f"  SKIP   {label:38s} (not in python/plot)")
    if failed:
        print(f"\n{len(failed)} step(s) failed: {', '.join(failed)}")
        sys.exit(1)
    print("\nAll done. " + done_note)


if __name__ == '__main__':
    main()
