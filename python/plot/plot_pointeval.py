"""Render POINT-EVALUATED field maps ('--sample-points') for many snapshots at once.

The plotting half of the pipeline: run_ps_pipeline.sh computes the grids and the '.pts_*'
image planes, this script turns the latter into figures. Unlike the grid slice maps of
plot_PS_DTFE.py -- cell-averaged, one cell thick -- these evaluate the tessellation AT each
pixel centre, giving a zero-thickness cross-section at a resolution limited only by the
tessellation (see dtfelib/pointeval.py for the estimator differences). Styling is identical
to the grid maps, and files sit beside them under the same naming convention:

    <figures>/fields/<sim>/snapNNN/<plane>/ps_<field>_pointeval_<plane>_z<z>.png

Fields come from whatever the run wrote: density, streams, speed, dispTrace, dispMag and --
for runs with PTS_VEL_GRAD=1 -- velDiv, velShear, velVort (plus denGradMag with
--pts-den-grad). Snapshots and simulations are auto-discovered; nothing is recomputed here.

    python3 plot/plot_pointeval.py                          # every sim, every snapshot
    python3 plot/plot_pointeval.py --sims TNG300-3-Dark      # one simulation
    python3 plot/plot_pointeval.py --snaps 0,4,17,33         # selected snapshots
    python3 plot/plot_pointeval.py --fields density,velDiv --force
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import config
from dtfelib import pointeval
from dtfelib.cli import DATA_ROOT
from dtfelib.io import FieldSet


def discover_sims(data_root: Path):
    return sorted(d.name for d in data_root.iterdir()
                  if d.is_dir() and any(d.glob("snapdir_*/combined_*.hdf5")))


def discover_snaps(simdir: Path):
    out = []
    for f in sorted(simdir.glob("snapdir_*/combined_*.hdf5")):
        try:
            out.append(int(f.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(set(out))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT,
                   help=f"simulation output root (default {DATA_ROOT})")
    p.add_argument("--sims", default=None,
                   help="comma-separated simulations (default: all with snapshots on disk)")
    p.add_argument("--snaps", default=None,
                   help="comma-separated snapshot numbers (default: all present)")
    p.add_argument("--fields", default=None,
                   help=f"comma-separated subset of {','.join(pointeval.STYLES)} (default: all available)")
    p.add_argument("--prefix", default=None,
                   help="alternate on-disk grid prefix (run_ps_dtfe.sh OUTPUT_PREFIX, e.g. 'ps_mw')")
    p.add_argument("--plane", type=Path, default=None,
                   help="pin the plane sidecar (.json) instead of auto-discovering it -- "
                        "REQUIRED when a simulation has several planes of equal pixel count "
                        "(the xy/xz/yz views), since the .pts_* files carry no axis to match on")
    p.add_argument("--figures-root", type=Path, default=Path(config.LOCAL_FIGURES_ROOT),
                   help="figure tree root (default: config.LOCAL_FIGURES_ROOT)")
    p.add_argument("--force", action="store_true",
                   help="re-render even when the PNG is newer than its .pts_* source")
    p.add_argument("--project", choices=("plane", "slab"), default="plane",
                   help="'plane' (default) = the central sampling plane, a zero-thickness "
                        "cross-section; 'slab' = average every plane in the file, the "
                        "finite-thickness analogue of a grid cell")
    p.add_argument("--smooth", type=float, default=0.0, metavar="SIGMA",
                   help="Gaussian smoothing in output pixels applied to EVERY field's primary "
                        "output (default 0); also suppresses the auto smoothed-derivative "
                        "companions below")
    p.add_argument("--smooth-derivatives", type=float, default=pointeval.DERIVATIVE_SMOOTH,
                   metavar="SIGMA",
                   help="the gradient fields (velDiv/velShear/velVort/denGradMag) are "
                        "piecewise-constant per tetrahedron and look faceted; each also gets a "
                        f"Gaussian-smoothed companion '<field>..._smooth<sigma>.png' at this "
                        f"sigma in pixels (default {pointeval.DERIVATIVE_SMOOTH:g}; 0 disables)")
    p.add_argument("--fixed-range", action="store_true",
                   help="use the grid maps' fixed 1e-1..1e4 density range for side-by-side "
                        "comparison with them. Default is a percentile stretch: the fixed "
                        "range clips ~26%% of a z=0 point-eval image to the darkest colour")
    args = p.parse_args()

    bad = set((args.fields or "").split(",")) - set(pointeval.STYLES) - {""}
    if bad:
        sys.exit(f"unknown field(s) {sorted(bad)}; choose from {', '.join(pointeval.STYLES)}")
    fields = [f for f in (args.fields or "").split(",") if f] or None
    sims = [s for s in (args.sims or "").split(",") if s] or discover_sims(args.data_root)
    want = {int(n) for n in (args.snaps or "").replace(" ", "").split(",") if n} or None

    total, failed, skipped = 0, 0, 0
    for sim in sims:
        simdir = args.data_root / sim
        if not simdir.is_dir():
            print(f"!! {sim}: not found under {args.data_root}")
            failed += 1
            continue
        snaps = [n for n in discover_snaps(simdir) if want is None or n in want]
        if not snaps:
            print(f"-- {sim}: no matching snapshots")
            continue
        print(f"\n== {sim}: snapshots {' '.join(map(str, snaps))}")
        for n in snaps:
            snapdir = simdir / f"snapdir_{n:03d}"
            try:
                fs = FieldSet(snapdir, prefix=args.prefix)
            except (FileNotFoundError, ValueError) as e:
                print(f"-- snap {n:03d}: {e}")
                skipped += 1
                continue
            out_dir = args.figures_root / "fields" / sim / f"snap{n:03d}"
            print(f"-- snap {n:03d} (z = {fs.meta.redshift:.2f}) -> {out_dir}")
            try:
                written, up_to_date = pointeval.render_fields(fs, out_dir, fields=fields,
                                                          force=args.force,
                                                          fixed_range=args.fixed_range,
                                                          project=args.project,
                                                          smooth=args.smooth,
                                                          plane=args.plane,
                                                          smooth_derivatives=args.smooth_derivatives)
            except Exception as e:                      # one bad snapshot must not kill the batch
                print(f"!! snap {n:03d}: {type(e).__name__}: {e}")
                failed += 1
                continue
            if up_to_date < 0:
                print(f"   (no point-evaluated fields on disk for snap {n:03d} -- "
                      f"run scripts/run_ps_pipeline.sh first)")
                skipped += 1
            elif not written:
                print(f"   ({up_to_date} figure(s) already up to date; --force to re-render)")
            total += len(written)

    print(f"\n{total} figure(s) written"
          + (f", {skipped} snapshot(s) skipped" if skipped else "")
          + (f", {failed} failure(s)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
