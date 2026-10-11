"""The void catalogue of a run's grids: is it cached, and building it -- for the launcher's Runs tab and its
'Build void catalogue' queue job (2026-10-11).

The catalogue is the analysis pipeline's own (pipeline.SnapshotProducts.voids(), cached under python/cache or
DTFE_CACHE_DIR), in the frame ADOPTED from the grid (pipeline.adopt_snapshot_grid), exactly as the Explore tab's
void outlines (tools/void_outlines.py) read it: for the production grids that is the analysis scripts' cache
file. Run with DTFE_SIM set to the simulation (config reads the box for it) and DTFE_DATA_ROOT to its data root:

    DTFE_SIM=TNG100-3-Dark python3 python/gui/void_cache.py status < runs.json
    DTFE_SIM=TNG100-3-Dark python3 python/gui/void_cache.py build --sim TNG100-3-Dark --snap 99 --method ps

status  reads a JSON list of {sim, snap, method, prefix} on stdin and prints ONE JSON line: a list of
        {'state', 'why'} in the same order, state one of
          'cached'  a catalogue built from the density grid on disk now (pipeline._cached's grid stamp test)
          'stale'   a catalogue is there but from another version of the grid (rewritten since): the next
                    reader rebuilds it; or its radii R_v are not today's (pipeline.catalog_r_v_ok: older R_v
                    settings, none recorded, or no R_v at all): the next reader re-measures them
          'absent'  none built yet
          'error'   no grid or header to judge by (why says what)
        Nothing is built and nothing is written (the cache folder is not even created).
build   builds (or confirms) one snapshot's catalogue, refusing when it would not fit in the memory the other
        programs leave free (void_outlines' rule), and prints how many voids it holds. Exit 1 with the reason.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                # python/: config, dtfelib
sys.path.insert(0, str(HERE.parent / "tools"))      # void_outlines

DEFAULT_PREFIX = {"ps": "ps_output", "dtfe": "output"}   # the method's own grid prefix (analyze.ESTIMATOR_PREFIX)


def products_prefix(method: str, prefix: str | None):
    """None for the method's own prefix (the products' cache namespace without a tag), else the prefix."""
    return None if not prefix or prefix == DEFAULT_PREFIX.get(method) else prefix


def status_one(sim: str, snap: int, method: str, prefix: str | None, data_root=None) -> dict:
    from dtfelib import pipeline
    prod = pipeline.products(int(snap), sim=sim, method=method, prefix=products_prefix(method, prefix),
                             data_root=data_root)
    try:
        pipeline.adopt_snapshot_grid(prod)
        ns = Path(pipeline.config.CACHE_DIR) / pipeline._param_hash()
        if not ns.is_dir():                          # pipeline.cache_dir() would create it: a status only looks
            return {"state": "absent", "why": f"no catalogue cache under {ns.name}"}
        p = prod._cpath("voids")
        if not p.exists():
            return {"state": "absent", "why": f"no {p.name}"}
        with contextlib.redirect_stdout(sys.stderr):  # _cached says why a cache is stale: not on the JSON line
            ok = prod._cached("voids")
        if ok is None:
            return {"state": "stale", "why": "built from another version of the grid (rewritten since)"}
        import void_outlines as VO
        why = VO.r_v_stale(ok)                       # the pipeline's own R_v test (catalog_r_v_ok), 2026-10-11
        if why:
            return {"state": "stale", "why": why}
        return {"state": "cached", "why": p.name}
    finally:
        prod.release()


def status(items, data_root=None) -> list[dict]:
    out = []
    for it in items:
        try:
            out.append(status_one(it["sim"], int(it["snap"]), it.get("method") or "ps", it.get("prefix"), data_root))
        except Exception as e:                       # no grid, no header, an unreadable cache: a reason per row
            out.append({"state": "error", "why": f"{type(e).__name__}: {e}"})
    return out


def build(sim: str, snap: int, method: str, prefix: str | None, data_root=None) -> str:
    import void_outlines as VO
    res = VO.outlines(sim, snap, products_prefix(method, prefix), data_root, method)
    n = len(res["centres_mpc"])
    resolved = sum(bool(v) for v in res["resolved"])
    distinct = sum(bool(v) for v in res["distinct"])
    return (f"void catalogue of {sim} snapshot {int(snap):03d} ({res['method']}, {res['n_grid']}^3, sigma "
            f"{res['sigma_mpc']:.4g} Mpc, cache {res['namespace']}): {n} minima, {resolved} resolved, {distinct} distinct")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=("status", "build"))
    p.add_argument("--sim", default=None)
    p.add_argument("--snap", type=int, default=None)
    p.add_argument("--method", choices=("ps", "dtfe"), default="ps")
    p.add_argument("--prefix", default=None, help="the grid set's output prefix (default the method's)")
    p.add_argument("--data-root", type=Path, default=None, help="the data root (default DTFE_DATA_ROOT's)")
    a = p.parse_args(argv)
    if a.mode == "status":
        items = json.loads(sys.stdin.read() or "[]")
        sims = {it.get("sim") for it in items}
        if sims - {os.environ.get("DTFE_SIM")}:
            raise SystemExit(f"run with DTFE_SIM set to the one simulation of the list (got {sorted(map(str, sims))})")
        sys.stdout.write(json.dumps(status(items, a.data_root)) + "\n")
        return 0
    if a.sim is None or a.snap is None:
        raise SystemExit("build needs --sim and --snap")
    if os.environ.get("DTFE_SIM") != a.sim:
        raise SystemExit(f"run with DTFE_SIM={a.sim}: config reads the box for the simulation it names")
    import zipfile
    try:
        print(build(a.sim, a.snap, a.method, a.prefix, a.data_root), flush=True)
    except (OSError, ValueError, KeyError, MemoryError, EOFError, zipfile.BadZipFile) as e:
        print(f"no void catalogue for {a.sim} snapshot {a.snap}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
