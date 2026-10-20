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

--sigma-mpc S finds the voids at a smoothing of S Mpc instead of the production one (the same physical scale in
every box: the cross-box comparison), cached under that smoothing's own namespace; 'r_v_defs' carries every R_v
definition the catalogue has (catalog.R_V_DEFINITIONS once it exists: 'crossing', 'delta08', 'ridge'). --csv PATH
also writes the table of the voids (write_csv: centre, the R_v's, R_eff, delta, the flags), as the launcher's
Explore map does for the voids it shows.
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


_R_V_KEYS = ("r_v_settings", "r_v_cells", "r_v_delta08_cells", "r_v_ridge_cells")   # what catalog_r_v_ok reads


def r_v_stale(path) -> str | None:
    """Why voids() would re-measure the radii of the cached catalogue at 'path' (the pipeline's own test,
    pipeline.catalog_r_v_ok: the current R_v settings and all three radii), None when it reads them as they are. Only
    the four keys are loaded (an npz loads per key), so the Runs tab's status stays cheap; testing for
    'r_v_cells' alone would call a cache with older R_v settings, or none recorded, 'cached'."""
    import numpy as np
    from dtfelib import pipeline
    with np.load(path) as z:
        cat = {k: z[k] for k in _R_V_KEYS if k in z.files}
    if pipeline.catalog_r_v_ok(cat):
        return None
    if "r_v_cells" not in cat:
        return "built before the measured radius R_v"
    if "r_v_settings" not in cat:
        return "R_v measured before its settings were recorded (re-measured on next read)"
    return "R_v measured with older settings (re-measured on next read)"


def needs_build(prod) -> bool:
    """voids() would BUILD the catalogue (none cached for this grid) or re-measure its radii from the smoothed field
    (r_v_stale), not just read it: either reads the density grid and smooths it, so the memory gate applies."""
    p = prod._cached("voids")
    return p is None or r_v_stale(p) is not None


def _finite_or_none(x):
    return [float(v) if v == v else None for v in x]      # NaN -> null: JSON has no NaN


def r_v_definitions(cat) -> dict:
    """{definition: R_v in Mpc} for every R_v definition the catalogue HAS ('crossing' always, the first up-crossing of
    the mean; 'delta08' and 'ridge' only once catalog.R_V_DEFINITIONS exists and the cache holds them, contract C3):
    a definition that is NaN everywhere (the key absent from an older cache) is left out, not shown as 'none'."""
    import numpy as np
    from dtfelib import catalog
    defs = getattr(catalog, "R_V_DEFINITIONS", ("crossing",))
    out = {}
    for d in defs:
        try:
            r = catalog.r_v_mpc(cat) if d == "crossing" else catalog.r_v_mpc(cat, definition=d)
        except (TypeError, KeyError, ValueError):
            continue
        r = np.asarray(r, dtype=np.float64)
        if d == "crossing" or np.isfinite(r).any():
            out[d] = r
    return out


CSV_HEAD = ("void", "x_mpc", "y_mpc", "z_mpc")


def write_csv(res: dict, path, rows=None) -> str:
    """The voids of an outlines() result as a CSV table (one row per void; 'rows' = the catalogue rows to write, all
    by default): the centre (cell centre, Mpc), R_v (the 'crossing' definition) and every other R_v definition the
    result has, R_eff, the central delta, resolved and distinct (0/1). Empty fields where a value is undefined. A
    '#' line first says the snapshot, the frame and the smoothing, so a quoted number can be traced back."""
    import csv
    path = Path(path)
    n = len(res["centres_mpc"])
    rows = range(n) if rows is None else [int(k) for k in rows]
    defs = {d: v for d, v in (res.get("r_v_defs") or {}).items() if d != "crossing"}
    head = list(CSV_HEAD) + ["r_v_mpc"] + [f"r_v_{d}_mpc" for d in defs] + ["r_eff_mpc", "delta", "resolved", "distinct"]

    def num(v):
        return "" if v is None or v != v else f"{float(v):.6g}"
    with open(path, "w", newline="") as f:
        f.write(f"# {res.get('sim', '')} snapshot {res.get('snap', '')} {res.get('method', '')}"
                f"{(' prefix ' + res['prefix']) if res.get('prefix') else ''}: box {num(res.get('box_mpc'))} Mpc, "
                f"{res.get('n_grid', '')}^3 cells, sigma {num(res.get('sigma_mpc'))} Mpc; centres = cell centres; "
                f"R_v = first up-crossing of the mean, R_eff = (a b c)^(1/3) of the resolved ellipsoid\n")
        w = csv.writer(f)
        w.writerow(head)
        for k in rows:
            c = res["centres_mpc"][k]
            w.writerow([k] + [num(v) for v in c] + [num(res["r_v_mpc"][k])] + [num(v[k]) for v in defs.values()]
                       + [num(res["r_eff_mpc"][k]), num(res["delta"][k]), int(bool(res["resolved"][k])),
                          int(bool(res["distinct"][k]))])
    return str(path)


def outlines(sim, snap, prefix=None, data_root=None, method="auto", sigma_mpc=None) -> dict:
    """The voids of one snapshot's grids (the module docstring); sigma_mpc > 0 = that smoothing in Mpc (set BEFORE
    the frame is adopted, which keeps it in Mpc), None or 0 = the production one."""
    import numpy as np
    from dtfelib import pipeline, catalog
    if sigma_mpc:
        pipeline.set_smoothing(mpc=float(sigma_mpc))
    prod = pipeline.products(int(snap), sim=sim, method=method, prefix=prefix, data_root=data_root)
    try:
        cell, box, n = pipeline.adopt_snapshot_grid(prod)
        if needs_build(prod):
            need, free = BUILD_BYTES_PER_CELL * float(n) ** 3, available_bytes()
            if free is not None and need > free - HEADROOM_BYTES:
                raise MemoryError(f"its catalogue is not cached (or its R_v is re-measured), and building it ({n}^3) "
                                  f"needs about {need / 1e9:.0f} GB "
                                  f"with {free / 1e9:.0f} GB free: build it with the analysis scripts when the machine is "
                                  f"free (analyze.py compute), then tick again")
        cat = prod.voids()
        method = str(prod.fs.method)
    finally:
        prod.release()
    resolved = np.asarray(cat["well_resolved"]).astype(bool)
    r_v, r_eff = catalog.r_v_mpc(cat), catalog.r_eff_mpc(cat)
    defs = r_v_definitions(cat)
    distinct = catalog.distinct_mask(cat, resolved & np.isfinite(r_v), n)
    centres = (np.asarray(cat["coords"], dtype=np.float64) + 0.5) * cell
    return {"sim": sim, "snap": int(snap), "prefix": prefix or "", "method": method, "box_mpc": box, "cell_mpc": cell,
            "n_grid": n, "sigma_mpc": pipeline.smoothing_length_mpc(), "footprint": int(pipeline.config.FOOTPRINT_SIZE),
            "namespace": pipeline._param_hash(), "centres_mpc": centres.round(6).tolist(),
            "r_v_mpc": _finite_or_none(r_v), "r_eff_mpc": _finite_or_none(r_eff),
            "resolved": resolved.tolist(), "distinct": distinct.tolist(),
            "delta": _finite_or_none(np.asarray(cat["delta_values"], dtype=np.float64)),
            "sigma_asked_mpc": float(sigma_mpc) if sigma_mpc else 0.0,
            "r_v_defs": {d: _finite_or_none(v) for d, v in defs.items()}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sim", required=True)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--method", choices=("auto", "ps", "dtfe"), default="auto", help="the estimator of the grids")
    p.add_argument("--prefix", default=None, help="the grid set's output prefix (default the method's)")
    p.add_argument("--data-root", type=Path, default=None, help="the data root (default DTFE_DATA_ROOT's)")
    p.add_argument("--cache-dir", type=Path, default=None, help="the catalogue cache (default config.CACHE_DIR)")
    p.add_argument("--sigma-mpc", type=float, default=0.0,
                   help="the smoothing in Mpc (default 0: the production smoothing, the thesis's catalogue)")
    p.add_argument("--csv", type=Path, default=None, help="also write the table of all voids (write_csv) here")
    a = p.parse_args(argv)
    if os.environ.get("DTFE_SIM") != a.sim:
        raise SystemExit(f"run with DTFE_SIM={a.sim}: config reads the box for the simulation it names")
    import config
    if a.cache_dir is not None:
        config.CACHE_DIR = a.cache_dir
    import zipfile
    try:
        with contextlib.redirect_stdout(sys.stderr):          # the pipeline's progress lines: not on the JSON line
            res = outlines(a.sim, a.snap, a.prefix, a.data_root, a.method, a.sigma_mpc)
    # a damaged cache member (pipeline._cached reads only the stamp) is a reason, not a traceback
    except (OSError, ValueError, KeyError, MemoryError, EOFError, zipfile.BadZipFile) as e:
        raise SystemExit(f"no void catalogue for {a.sim} snapshot {a.snap}: {e}")
    if a.csv is not None:
        write_csv(res, a.csv)
    sys.stdout.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
