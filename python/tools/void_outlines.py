"""The voids of one snapshot's grids as JSON, for the launcher's Explore map ('voids': the outlines over the slice).

Runs the analysis pipeline's own catalogue (pipeline.SnapshotProducts.voids(): the smoothing, minima, Hessian,
ellipsoid cuts and R_v of the void scripts; cached, so only the first call per snapshot costs a minute) in config's
frame ADOPTED from the grid itself (pipeline.adopt_snapshot_grid: its header's box over its n, the smoothing kept
in Mpc), so any full-box grid set of a simulation works, at any resolution. Run it with DTFE_SIM set to the
simulation (config reads the box for it) and DTFE_DATA_ROOT to its data root:

    DTFE_SIM=TNG100-3-Dark python3 tools/void_outlines.py --sim TNG100-3-Dark --snap 99 --method ps

With the production grids (no --prefix) the catalogue is the analysis scripts' own cache file. A BUILD costs
about 80 bytes per cell (10 GB at 512^3, 80 GB at 1024^3): when the catalogue is not cached and that would not fit
in what the other programs leave free (less 4 GB), the tool refuses instead of starting it (DTFE_FREE_GB overrides
the free-memory reading, as for the C++ programs).

Prints ONE line of JSON on stdout (the pipeline's own messages go to stderr): the centres (cell centres in Mpc,
the map's frame), R_v and R_eff in Mpc (null where there is none), 'resolved' (the ellipsoid cuts), 'distinct'
(catalog.distinct_mask: the resolved voids with an R_v whose centre lies inside the sphere of min(R_v, R_eff) of no
deeper resolved void, one without R_v claiming its R_eff; the sample of the profiles and the abundance), the
central delta, and the frame and smoothing they were found at. Exit 1 with the reason on stderr when there is no
catalogue (no grids, no snapshot header).
"""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # python/ on sys.path


BUILD_BYTES_PER_CELL = 80          # a catalogue build: smoothing FFTs, the Hessian cubes, delta (10 GB at 512^3)
HEADROOM_BYTES = 4e9


def available_bytes():
    """What the other programs leave free now (bytes), or None when it cannot be read: DTFE_FREE_GB, else psutil,
    else macOS vm_stat (free + inactive + speculative + purgeable pages), else Linux MemAvailable."""
    if os.environ.get("DTFE_FREE_GB"):
        return float(os.environ["DTFE_FREE_GB"]) * 1e9
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        import re
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            page = int(re.search(r"page size of (\d+) bytes", out.stdout).group(1))
            pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"^Pages ([\w ]+):\s+(\d+)\.", out.stdout, re.M)}
            return float(page * sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable")))
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def needs_build(prod) -> bool:
    """voids() would BUILD the catalogue (none cached for this grid, or one without R_v), not read it."""
    import numpy as np
    p = prod._cached("voids")
    if p is None:
        return True
    with np.load(p) as z:
        return "r_v_cells" not in z.files


def _finite_or_none(x):
    return [float(v) if v == v else None for v in x]      # NaN -> null: JSON has no NaN


def outlines(sim, snap, prefix=None, data_root=None, method="auto") -> dict:
    import numpy as np
    from dtfelib import pipeline, catalog
    prod = pipeline.products(int(snap), sim=sim, method=method, prefix=prefix, data_root=data_root)
    try:
        cell, box, n = pipeline.adopt_snapshot_grid(prod)
        if needs_build(prod):
            need, free = BUILD_BYTES_PER_CELL * float(n) ** 3, available_bytes()
            if free is not None and need > free - HEADROOM_BYTES:
                raise MemoryError(f"its catalogue is not cached, and building it ({n}^3) needs about {need / 1e9:.0f} GB "
                                  f"with {free / 1e9:.0f} GB free: build it with the analysis scripts when the machine is "
                                  f"free (analyze.py compute), then tick again")
        cat = prod.voids()
        method = str(prod.fs.method)
    finally:
        prod.release()
    resolved = np.asarray(cat["well_resolved"]).astype(bool)
    r_v, r_eff = catalog.r_v_mpc(cat), catalog.r_eff_mpc(cat)
    distinct = catalog.distinct_mask(cat, resolved & np.isfinite(r_v), n)
    centres = (np.asarray(cat["coords"], dtype=np.float64) + 0.5) * cell
    return {"sim": sim, "snap": int(snap), "prefix": prefix or "", "method": method, "box_mpc": box, "cell_mpc": cell,
            "n_grid": n, "sigma_mpc": pipeline.smoothing_length_mpc(), "footprint": int(pipeline.config.FOOTPRINT_SIZE),
            "namespace": pipeline._param_hash(), "centres_mpc": centres.round(6).tolist(),
            "r_v_mpc": _finite_or_none(r_v), "r_eff_mpc": _finite_or_none(r_eff),
            "resolved": resolved.tolist(), "distinct": distinct.tolist(),
            "delta": _finite_or_none(np.asarray(cat["delta_values"], dtype=np.float64))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sim", required=True)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--method", choices=("auto", "ps", "dtfe"), default="auto", help="the estimator of the grids")
    p.add_argument("--prefix", default=None, help="the grid set's output prefix (default the method's)")
    p.add_argument("--data-root", type=Path, default=None, help="the data root (default DTFE_DATA_ROOT's)")
    p.add_argument("--cache-dir", type=Path, default=None, help="the catalogue cache (default config.CACHE_DIR)")
    a = p.parse_args(argv)
    if os.environ.get("DTFE_SIM") != a.sim:
        raise SystemExit(f"run with DTFE_SIM={a.sim}: config reads the box for the simulation it names")
    import config
    if a.cache_dir is not None:
        config.CACHE_DIR = a.cache_dir
    import zipfile
    try:
        with contextlib.redirect_stdout(sys.stderr):          # the pipeline's progress lines: not on the JSON line
            res = outlines(a.sim, a.snap, a.prefix, a.data_root, a.method)
    # a damaged cache member (pipeline._cached reads only the stamp) is a reason, not a traceback (2026-10-08)
    except (OSError, ValueError, KeyError, MemoryError, EOFError, zipfile.BadZipFile) as e:
        raise SystemExit(f"no void catalogue for {a.sim} snapshot {a.snap}: {e}")
    sys.stdout.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
