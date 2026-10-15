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
more smoothings as current. 'cache --move DIR' (2026-10-11 evening) moves the whole cache to DIR -- every entry
copied and verified before any source is removed -- and records DIR as the cache folder (config.cache_setting_file,
read after DTFE_CACHE_DIR), e.g. out of iCloud Drive; a DRY RUN without '--yes', refused while a figure script, an
analysis, the launcher or a DTFE binary may be using the cache (move_cache, cache_users).
"""

import argparse
import re
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

# The physical smoothings (Mpc) today's code asks for by default: the cross-box comparison's sigma,
# config.CROSSBOX_SMOOTHING_MPC (2026-10-11, G2: the one number plot_void_crossbox, the launcher and this read). A
# namespace made with another Mpc value (2.16 on 2026-10-11 morning: r512_s21.4043_f23 next to 2.1625's
# r512_s21.4291_f23) is reached only by typing that value again; 'cache --keep-mpc' and DTFE_SIGMA_MPC add values.
CACHE_SMOOTHING_MPC = (config.CROSSBOX_SMOOTHING_MPC,)
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


FIXED_R_CELLS = (10.0, 5.0, 3.0, 2.0)     # see cache_inventory


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
    # the fixed-R size function's catalogues (plot_void_population.FIXED_R_SMOOTH_CELLS 10 5 3, and TNG300's 3 2 in
    # thesis_runs.sh voids) and its --size-smooth 5 are live too (2026-10-12)
    cells_ok = {f"{float(c):g}" for c in (sig0, *config.SMOOTHING_COMPARISON_SIGMAS, *FIXED_R_CELLS, *keep_cells)}
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


# ---------------------------------------------------------------- cache --move (2026-10-11 evening)

# what reads or writes the cache: the figure / analysis scripts, the launcher (its Explore tab and Runs status read
# catalogues in-process, its jobs run the scripts) and the binaries a Pipeline job runs before its analysis. A python
# whose command line names one of these is a user; the binaries are found by name, as the launcher's own check does
# (runspec.dtfe_processes' pgrep pattern). The interpreter may also be the macOS host bundle scripts/gui.sh starts the
# launcher in ('<venv>/DTFE Launcher.app/Contents/MacOS/DTFE Launcher <repo>/python/gui/app.py'; its sys.executable,
# runspec.PYTHON's last fallback for the jobs), and the launcher is matched by its script path whatever runs it: the
# old pattern wanted a 'python...' word before it and so missed the usual way the launcher is open (2026-10-11 review).
_CACHE_USER_SCRIPTS = re.compile(r"(?i)(?:python[^\s/]*|DTFE Launcher)\s.*?(?:\banalyze\.py|\bselftest\.py|"
                                 r"\bplot_\w+\.py|\bvoid_outlines\.py|\bvoid_cache\.py)(?:\s|$)"
                                 r"|(?:^|[\s/])gui/app\.py(?:\s|$)")
_DTFE_BINARY = r"^(PS-)?DTFE(-.*)?$"


def _ps_table():
    """{pid: (ppid, args)} of every process (one ps call; -A and -o spelled the same on macOS and Linux). {} without ps."""
    try:
        text = subprocess.run(["ps", "-A", "-ww", "-o", "pid=,ppid=,args="], capture_output=True, text=True,
                              timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    out = {}
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            out[int(parts[0])] = (int(parts[1]), parts[2] if len(parts) > 2 else "")
    return out


def cache_users(table=None, dtfe_names=None, me=None):
    """[(pid, what)] of the processes that may be using the pipeline cache now: a python running an analysis or
    figure script or the launcher, and the DTFE / PS-DTFE binaries (a Pipeline job runs one, then its analysis).
    This process and its ANCESTORS are left out: 'analyze.py cache --move' run by the launcher's setup is the
    launcher's own child, and the launcher then points itself at the new folder. 'table' ({pid: (ppid, args)})
    and 'dtfe_names' ({pid: name}) are for tests; by default ps and pgrep are asked."""
    table = _ps_table() if table is None else table
    if dtfe_names is None:
        try:
            lines = subprocess.run(["pgrep", "-l", _DTFE_BINARY], capture_output=True, text=True,
                                   timeout=5).stdout.splitlines()
        except (OSError, subprocess.SubprocessError):
            lines = []
        dtfe_names = {int(a): b for a, b in (ln.split(None, 1) for ln in lines if " " in ln) if a.isdigit()}
    me = os.getpid() if me is None else me
    mine, pid = set(), me
    while pid and pid not in mine:                  # me, my parent, ... up to launchd / init
        mine.add(pid)
        pid = table.get(pid, (0, ""))[0]
    users = []
    for pid, (_ppid, args) in sorted(table.items()):
        if pid not in mine and _CACHE_USER_SCRIPTS.search(args):
            users.append((pid, args if len(args) < 160 else args[:157] + "..."))
    seen = {pid for pid, _ in users}
    users += [(pid, name) for pid, name in sorted(dtfe_names.items()) if pid not in mine and pid not in seen]
    return users


_SF_DATALESS = 0x40000000                           # macOS <sys/stat.h>: the file's bytes are in the cloud only


def _st_flags(q):
    """st_flags of 'q' itself (0 where the platform has none or q is gone); a hook for the tests, which cannot set
    SF_DATALESS (only the file provider can)."""
    try:
        return getattr(q.lstat(), 'st_flags', 0)
    except OSError:
        return 0


def _evicted(q):
    """Whether 'q' is evicted to iCloud: a legacy '.<name>.icloud' stub, or (current macOS) dataless in place."""
    return q.name.endswith('.icloud') or bool(_st_flags(q) & _SF_DATALESS)


def _lstat_size(q):
    try:
        return q.lstat().st_size if not q.is_dir() else 0
    except OSError:
        return 0


def _same_device(a, b):
    try:
        return Path(a).stat().st_dev == Path(b).stat().st_dev
    except OSError:
        return True                                 # unknown: count the download, the safe side


def _same_tree(a, b):
    """Whether folder (or file) 'b' holds exactly what 'a' does: the same relative file names, sizes and bytes."""
    import filecmp
    a, b = Path(a), Path(b)
    if a.is_file():
        return b.is_file() and a.stat().st_size == b.stat().st_size and filecmp.cmp(a, b, shallow=False)
    fa = {p.relative_to(a): p for p in a.rglob('*') if p.is_file()}
    fb = {p.relative_to(b): p for p in b.rglob('*') if p.is_file()}
    if set(fa) != set(fb):
        return False
    return all(fa[k].stat().st_size == fb[k].stat().st_size and filecmp.cmp(fa[k], fb[k], shallow=False) for k in fa)


def move_cache(dest, yes=False, src=None, users=None, setting_file=None):
    """'analyze.py cache --move DIR' (and the launcher's setup): move the pipeline cache (config.CACHE_DIR) to DIR
    and record DIR as the cache folder (config.write_cache_setting), so every later script and the launcher use it
    without DTFE_CACHE_DIR. A DRY RUN unless yes: it lists what would move, how much, the room there, and whatever
    would stop it. With yes, refused (exit 1, nothing touched) while a run, a figure script or the launcher may be
    using the cache (cache_users; 'users' replaces the check in tests), when DIR is inside the cache or holds a
    namespace of the same name, or without room. Then EVERY entry is copied into a hidden '.<name>.<pid>.partial'
    beside its final name and verified byte for byte before anything is renamed: a failure leaves the old cache as
    it was and removes the copies. Only then each copy is renamed to its name (atomic: same folder), the setting
    written, and each source removed -- copy, verify, then remove. Returns the exit code."""
    import shutil
    src = Path(config.CACHE_DIR if src is None else src).expanduser()
    dest = Path(os.path.expanduser(str(dest))).resolve()
    setting_file = Path(config.cache_setting_file() if setting_file is None else setting_file)
    print(f"Pipeline cache {src}" + ("" if src.is_dir() else " (absent)") + f" -> {dest}")
    if os.environ.get('DTFE_CACHE_DIR'):
        print("  note: DTFE_CACHE_DIR is set in this shell; it overrides the recorded setting wherever it is set")
    src_r = src.resolve()
    if dest == src_r:
        print("  the cache is already there" + ("; recorded as the setting" if yes else ""))
        if yes:
            config.write_cache_setting(dest, setting_file)
        return 0
    if src_r in dest.parents or dest in src_r.parents:
        print(f"  !! refused: {dest} and {src_r} contain one another")
        return 1
    items = sorted(src.iterdir()) if src.is_dir() else []
    clash = [p.name for p in items if (dest / p.name).exists()]
    sizes = {p: (_dir_size(p) if p.is_dir() else (p.stat().st_size, 1)) for p in items}
    total = sum(b for b, _ in sizes.values())
    for p in items:
        b, n = sizes[p]
        print(f"  {'move ' if yes else 'would move '}{p.name:44s} {_human(b):>9s} {n:5d} file(s)")
    if not items:
        print("  nothing cached yet: only the setting is recorded")
    probe = dest
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        free = None
    why = []
    # an EVICTED iCloud file is a '.<name>.icloud' stub beside where it was: copying moves the stub, and removing it
    # from iCloud Drive deletes the file in the cloud -- the catalogue would be gone (2026-10-11). Current macOS no
    # longer writes stubs: it evicts in place under the real name with SF_DATALESS in st_flags (_evicted), and copying
    # one downloads it first, onto the source's disk -- so its bytes count twice in the room needed when the
    # destination shares that disk (2026-10-11 review: the check never fired on this Mac, and the space check reserved
    # 1.05x where the copy took up to 2x).
    evicted = [q for p in items for q in ([p] if p.is_file() else [p, *p.rglob('*')]) if _evicted(q)]
    evicted_bytes = sum(_lstat_size(q) for q in evicted if not q.name.endswith('.icloud'))
    if evicted:
        why.append(f"{len(evicted)} file(s) are evicted to iCloud ({evicted[0].name}, ...): download them first "
                   "(Finder: Download Now on the cache folder), then move")
    need = total * 1.05 + 100e6
    if evicted_bytes and free is not None and _same_device(probe, src):
        need += evicted_bytes
    partial = sorted(p.name for p in dest.glob('.*.partial')) if dest.is_dir() else []
    if partial:
        print(f"  note: {', '.join(partial)} in {dest} is an interrupted move's copy: remove it")
    if clash:
        why.append(f"{dest} already holds {', '.join(clash)} (merge by hand, or pick another folder)")
    if free is not None and free < need:
        why.append(f"{_human(free)} free at {probe}, {_human(total)} to copy"
                   + (f" (+{_human(evicted_bytes)} evicted to iCloud, downloaded onto the same disk first)"
                      if need > total * 1.05 + 100e6 else ""))
    users = cache_users() if users is None else users
    if users:
        why.append("the cache may be in use by: " + "; ".join(f"pid {pid} {what}" for pid, what in users)
                   + " (let them finish, close the launcher, then move)")
    print(f"  {len(items)} entr{'y' if len(items) == 1 else 'ies'}, {_human(total)}; the setting goes to {setting_file}")
    if why:
        print("  !! " + ("refused: " if yes else "it would be refused now: ") + "; ".join(why))
    if not yes:
        print("\nDRY RUN: nothing moved; add --yes to move the cache and record the setting.")
        return 0
    if why:
        return 1
    dest.mkdir(parents=True, exist_ok=True)
    partial = {p: dest / f".{p.name}.{os.getpid()}.partial" for p in items}
    try:
        for p in items:                             # 1: copy and verify everything; the old cache untouched
            (shutil.copytree if p.is_dir() else shutil.copy2)(p, partial[p])
            if not _same_tree(p, partial[p]):
                raise OSError(f"the copy of {p.name} differs from it")
    except OSError as e:
        print(f"  !! the copy failed ({e}): removed it, the cache stays in {src}")
        for q in partial.values():
            shutil.rmtree(q, ignore_errors=True) if q.is_dir() else q.unlink(missing_ok=True)
        return 1
    for p in items:                                 # 2: into place (one folder: atomic renames)
        os.replace(partial[p], dest / p.name)
    config.write_cache_setting(dest, setting_file)
    config.CACHE_DIR, config.CACHE_DIR_SOURCE = dest, f"the setting in {setting_file}"
    for p in items:                                 # 3: the sources, each verified copy now in place
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        except OSError as e:
            print(f"  !! {p}: {e} (its copy is in {dest}; remove it by hand)")
    try:
        src.rmdir()                                 # the folder itself when nothing else is left in it
    except OSError:
        pass
    print(f"\nMoved {_human(total)} to {dest}; recorded in {setting_file}.")
    return 0


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
    parser.add_argument('--yes', action='store_true',
                        help="'cache --prune': really delete what it lists; 'cache --move': really move")
    parser.add_argument('--move', default=None, metavar='DIR',
                        help="'cache': move the cache to DIR (copied, verified, then removed) and record DIR as the "
                             "cache folder for every later run; a DRY RUN without --yes")
    parser.add_argument('--keep-mpc', type=float, nargs='*', default=[], metavar='MPC',
                        help=f"'cache': more smoothings in Mpc to count as current (besides {list(CACHE_SMOOTHING_MPC)} "
                             "and DTFE_SIGMA_MPC)")
    parser.add_argument('--keep-cells', type=float, nargs='*', default=[], metavar='CELLS',
                        help="'cache': more smoothings in cells (footprint 11) to count as current")
    args = parser.parse_args()
    if args.yes and not (args.mode == 'cache' and (args.prune or args.move)):
        parser.error("--yes goes with 'cache --prune' or 'cache --move' only")
    if args.mode != 'cache' and (args.prune or args.keep_mpc or args.keep_cells or args.move):
        parser.error("--prune / --move / --keep-mpc / --keep-cells go with 'cache' only")
    if args.move and (args.prune or args.keep_mpc or args.keep_cells):
        parser.error("--move goes alone (prune before or after it)")
    if args.mode == 'cache' and args.move:
        sys.exit(move_cache(args.move, args.yes))
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
    print(f"Cache: {config.CACHE_DIR} ({config.CACHE_DIR_SOURCE})")
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
