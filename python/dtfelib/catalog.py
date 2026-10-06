"""Void catalogue export (survey item 7, 2026-10-05): one row per void of
pipeline.SnapshotProducts.voids(), as CSV and HDF5, with units and the parameters it was built with.

    from dtfelib import catalog, pipeline
    p = pipeline.products("099", 0.0)
    paths = catalog.export_voids(p, "figures/void_catalog/TNG50-4-Dark")   # [.csv, .h5]

    import pandas as pd;  df = pd.read_csv(paths[0], comment="#")           # the CSV: '#' lines = provenance
    import h5py;          f = h5py.File(paths[1]);  f.attrs["sigma_cells"];  f["voids/r_eff_mpc"][:]

The columns (COLUMNS below, same order in both files) come straight from the catalogue dict: the grid
cell of the smoothed-delta minimum and its position in Mpc, the central delta, the three Hessian
eigenvalues (ASCENDING, as numpy.linalg.eigh gives them), the BBKS shape (e, p) and the axis ratios
from those eigenvalues, the ellipsoid fit (semi-axes a >= b >= c -- sorted by |lambda|, which is NOT the
eigenvalue order -- the effective radius (a b c)^(1/3), the fit's ellipticity 1 - c/a and prolateness,
the major and minor axis directions) and the two sample flags (well_resolved: passes
config.ELLIPSOID_CUTS; deep: central delta below config.DEEP_VOID_THRESHOLD). Fit columns are NaN for a
void that is not well resolved, as plot_void_population treats them. R_eff is r_eff_mpc(), the one
definition in the code base (plot_void_population.snapshot_stats uses it).

THE FRAME: the catalogue's positions and semi-axes are computed with config.CELL_SIZE, which comes
from the DTFE_SIM simulation's box (config.py), not from the snapshot a script's --sim names. So the
snapshot header's cell (box / grid) must equal config's, else frame_of (pipeline.snapshot_frame, which
SnapshotProducts.voids() calls first) refuses before any cache is read or written. A cache built under
another frame or other ellipsoid cuts is re-derived on load (pipeline.refit_catalog: milliseconds, no
field). The export needs the snapshot header: without a combined file FieldSet raises FileNotFoundError.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
from . import pipeline

# (name, unit, description) -- the order of the columns in both files
COLUMNS = (
    ("id", "", "row index in the pipeline catalogue (the npz under python/cache)"),
    ("ix", "cell", "grid cell of the minimum of the smoothed delta, axis 0"),
    ("iy", "cell", "grid cell, axis 1"),
    ("iz", "cell", "grid cell, axis 2"),
    ("x_mpc", "Mpc", "position of the cell, axis 0 (cell index x cell size; comoving, h-free)"),
    ("y_mpc", "Mpc", "position, axis 1"),
    ("z_mpc", "Mpc", "position, axis 2"),
    ("delta_c", "", "central density contrast of the SMOOTHED field at the minimum"),
    ("lambda1", "Mpc^-2", "Hessian eigenvalue of the smoothed delta, smallest (eigh order: ascending); the cache holds cell^-2, divided by cell_mpc^2 here"),
    ("lambda2", "Mpc^-2", "Hessian eigenvalue, middle"),
    ("lambda3", "Mpc^-2", "Hessian eigenvalue, largest"),
    ("e_bbks", "", "BBKS ellipticity (lambda3 - lambda1) / (2 tr); NaN when tr <= 0"),
    ("p_bbks", "", "BBKS prolateness (lambda1 - 2 lambda2 + lambda3) / (2 tr); NaN when tr <= 0"),
    ("b_over_a", "", "axis ratio b/a from 1/sqrt(lambda); NaN when lambda1 <= 0 or the ratios are not ordered"),
    ("c_over_a", "", "axis ratio c/a from 1/sqrt(lambda)"),
    ("well_resolved", "0/1", "ellipsoid fit within config.ELLIPSOID_CUTS (the fit columns are NaN otherwise)"),
    ("deep", "0/1", "central delta below config.DEEP_VOID_THRESHOLD"),
    ("a_mpc", "Mpc", "ellipsoid semi-axis 2 cell / sqrt(|lambda|), largest (a >= b >= c: sorted by |lambda|)"),
    ("b_mpc", "Mpc", "ellipsoid semi-axis, middle"),
    ("c_mpc", "Mpc", "ellipsoid semi-axis, smallest"),
    ("r_eff_mpc", "Mpc", "effective radius (a b c)^(1/3)"),
    ("ellipticity_fit", "", "1 - c/a of the ellipsoid fit"),
    ("prolateness_fit", "", "(a^2 - b^2) / (a^2 - c^2) of the ellipsoid fit"),
    ("major_x", "", "unit vector of the major axis (a), component 0"),
    ("major_y", "", "major axis, component 1"),
    ("major_z", "", "major axis, component 2"),
    ("minor_x", "", "unit vector of the minor axis (c), component 0"),
    ("minor_y", "", "minor axis, component 1"),
    ("minor_z", "", "minor axis, component 2"),
)
_BOOL = ("well_resolved", "deep")
_INT = ("id", "ix", "iy", "iz")


def r_eff_mpc(cat) -> np.ndarray:
    """The effective radius (a b c)^(1/3) of every void, NaN where the ellipsoid is not well resolved:
    THE definition of R_eff in this code base (plot_void_population.snapshot_stats)."""
    resolved = np.asarray(cat["well_resolved"]).astype(bool)
    axes = np.asarray(cat["semi_axes"])                     # the stored dtype (float32): the script's exact values
    return np.where(resolved, np.prod(axes, axis=1) ** (1.0 / 3.0), np.nan)


def void_abundance(r_eff_mpc, box_mpc: float, radii_mpc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cumulative void abundance n(> R) per Mpc^3 (comoving) at every radius of 'radii_mpc', with its
    Poisson error sqrt(N) / V and the counts N: (n, err, N). NaN radii (unresolved voids) are left out.
    Survey item 10 (2026-10-05)."""
    r = np.asarray(r_eff_mpc, dtype=np.float64)
    r = r[np.isfinite(r)]
    radii = np.asarray(radii_mpc, dtype=np.float64)
    counts = np.array([(r > R).sum() for R in radii], dtype=np.int64)
    vol = float(box_mpc) ** 3
    return counts / vol, np.sqrt(counts) / vol, counts


def void_dn_dlnr(r_eff_mpc, box_mpc: float, edges_mpc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The differential abundance dn / dlnR per Mpc^3 in the bins 'edges_mpc' (counts / (V dlnR)), its
    Poisson error and the counts per bin: (dn, err, N), one value per bin (len(edges) - 1)."""
    r = np.asarray(r_eff_mpc, dtype=np.float64)
    r = r[np.isfinite(r)]
    edges = np.asarray(edges_mpc, dtype=np.float64)
    counts, _ = np.histogram(r, bins=edges)
    dln = np.log(edges[1:] / edges[:-1])
    vol = float(box_mpc) ** 3
    return counts / (vol * dln), np.sqrt(counts) / (vol * dln), counts.astype(np.int64)


def void_table(cat) -> dict:
    """The catalogue dict (pipeline.build_void_catalog's keys) as named columns, one 1-D array per
    COLUMNS entry, in that order. Pure: no file, no config lookups (the frame is already in the dict)."""
    coords = np.asarray(cat["coords"])
    n = len(coords)
    resolved = np.asarray(cat["well_resolved"]).astype(bool)
    nanfit = np.where(resolved, 1.0, np.nan)                   # NaN in every fit column of an unresolved void
    pos = np.asarray(cat["positions_mpc"], dtype=np.float64)
    cell = pipeline.catalog_cell_mpc(cat) or 1.0                # the Hessian is in cell^-2 (k in radians per cell)
    ev = np.asarray(cat["eigenvalues"], dtype=np.float64).reshape(n, 3) / cell ** 2
    bbks = np.asarray(cat["bbks_params"], dtype=np.float64).reshape(n, 2)
    ratios = np.asarray(cat["axis_ratios"], dtype=np.float64).reshape(n, 2)
    axes = np.asarray(cat["semi_axes"], dtype=np.float64).reshape(n, 3)
    orient = np.asarray(cat["orientations"], dtype=np.float64).reshape(n, 3, 3)
    cols = {
        "id": np.arange(n, dtype=np.int64),
        "ix": coords[:, 0].astype(np.int64), "iy": coords[:, 1].astype(np.int64), "iz": coords[:, 2].astype(np.int64),
        "x_mpc": pos[:, 0], "y_mpc": pos[:, 1], "z_mpc": pos[:, 2],
        "delta_c": np.asarray(cat["delta_values"], dtype=np.float64),
        "lambda1": ev[:, 0], "lambda2": ev[:, 1], "lambda3": ev[:, 2],
        "e_bbks": bbks[:, 0], "p_bbks": bbks[:, 1],
        "b_over_a": ratios[:, 0], "c_over_a": ratios[:, 1],
        "well_resolved": resolved,
        "deep": np.asarray(cat["deep"]).astype(bool),
        "a_mpc": axes[:, 0] * nanfit, "b_mpc": axes[:, 1] * nanfit, "c_mpc": axes[:, 2] * nanfit,
        "r_eff_mpc": r_eff_mpc(cat).astype(np.float64),
        "ellipticity_fit": np.asarray(cat["fit_ellipticity"], dtype=np.float64) * nanfit,
        "prolateness_fit": np.asarray(cat["fit_prolateness"], dtype=np.float64) * nanfit,
        "major_x": orient[:, 0, 0] * nanfit, "major_y": orient[:, 1, 0] * nanfit, "major_z": orient[:, 2, 0] * nanfit,
        "minor_x": orient[:, 0, 2] * nanfit, "minor_y": orient[:, 1, 2] * nanfit, "minor_z": orient[:, 2, 2] * nanfit,
    }
    assert list(cols) == [c[0] for c in COLUMNS]
    return cols


def frame_of(products) -> tuple[float, float, int, str]:
    """(cell_mpc, box_mpc, grid_n, 'header') of a snapshot's catalogue: pipeline.snapshot_frame's rule (the
    header's box over the grid must be config.CELL_SIZE, else ValueError; no header = FileNotFoundError)
    -- one rule for the export, the profiles, the population and voids() itself."""
    return pipeline.snapshot_frame(products)


def assert_frame(cat, cell_mpc: float) -> None:
    """Refuse a catalogue dict whose own frame (pipeline.catalog_cell_mpc: its 'cell_mpc', else recovered
    from positions / cell indices) is not 'cell_mpc': built under another DTFE_SIM. SnapshotProducts.voids
    rebuilds such a cache itself; this guards a dict handed in from elsewhere."""
    if not pipeline.catalog_frame_ok(cat, cell_mpc):
        raise ValueError(f"the catalogue's frame is {pipeline.catalog_cell_mpc(cat):.5g} Mpc per cell, the snapshot's "
                         f"{cell_mpc:.5g}: built under another DTFE_SIM; load it through SnapshotProducts.voids(), "
                         f"which re-derives its fits in this frame")


def provenance(products, n_voids: int, frame) -> dict:
    """What the catalogue was built from and with: the HDF5 root attributes, the CSV's '#' lines."""
    cell, box, n, source = frame
    cuts = config.ELLIPSOID_CUTS
    z = products.redshift
    return {
        "catalogue": "DTFE void catalogue: minima of the smoothed density contrast (dtfelib.pipeline.build_void_catalog)",
        "sim": str(products.sim),
        "snapshot": str(products.snapshot),
        "redshift": float("nan") if z is None else float(z),
        "method": str(products.fs.method),
        "prefix": str(products.prefix or (products.fs.prefix.rstrip(".") if hasattr(products.fs, "prefix") else "")),
        "box_mpc": float(box),
        "box_source": source,                         # always 'header': the box comes from the snapshot file
        "grid_n": int(n),
        "cell_mpc": float(cell),
        "sigma_cells": float(config.SMOOTHING_SIGMA_CELLS),
        "sigma_mpc": float(config.SMOOTHING_SIGMA_CELLS * cell),
        "footprint_cells": int(config.FOOTPRINT_SIZE),
        "criterion": str(config.VOID_EIGENVALUE_CRITERION),
        "gradient_threshold": float(config.GRADIENT_THRESHOLD),
        "deep_void_threshold": float(config.DEEP_VOID_THRESHOLD),
        "cut_min_axis_mpc": float(cuts["min_axis_mpc"]),
        "cut_max_axis_mpc": float(cuts["max_axis_mpc"]),
        "cut_max_axis_ratio": float(cuts["max_axis_ratio"]),
        "param_hash": pipeline._param_hash(),         # the cache directory the catalogue lives in
        "n_voids": int(n_voids),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _fmt(name, v) -> str:
    if name in _BOOL:
        return "1" if v else "0"
    if name in _INT:
        return str(int(v))
    return "nan" if not np.isfinite(v) else f"{float(v):.9g}"


def write_csv(table: dict, prov: dict, path) -> Path:
    """'# key: value' provenance lines, '# name [unit]: description' per column, the header row, the
    rows (bools as 0/1, NaN as 'nan', floats to 9 significant digits). pandas: read_csv(path, comment='#')."""
    path = Path(path)
    names = [c[0] for c in COLUMNS]
    lines = [f"# {k}: {v}" for k, v in prov.items()]
    lines += [f"# {name}" + (f" [{unit}]" if unit else "") + f": {desc}" for name, unit, desc in COLUMNS]
    lines.append(",".join(names))
    n = len(table["id"])
    cols = [table[k] for k in names]
    for i in range(n):
        lines.append(",".join(_fmt(k, c[i]) for k, c in zip(names, cols)))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_hdf5(table: dict, prov: dict, path) -> Path:
    """Root attributes = the provenance; group 'voids' with one dataset per column (bools as int8 0/1),
    each with 'unit' and 'description' attributes; 'columns' = the column names in order."""
    import h5py
    path = Path(path)
    with h5py.File(path, "w") as f:
        for k, v in prov.items():
            f.attrs[k] = v
        f.attrs["columns"] = [c[0] for c in COLUMNS]
        g = f.create_group("voids")
        for name, unit, desc in COLUMNS:
            data = table[name]
            if name in _BOOL:
                data = np.asarray(data, dtype=np.int8)
            d = g.create_dataset(name, data=data)
            d.attrs["unit"] = unit
            d.attrs["description"] = desc
    return path


def read_hdf5(path) -> tuple[dict, dict]:
    """(table, provenance) back from write_hdf5's file (bools as bools again)."""
    import h5py
    with h5py.File(path, "r") as f:
        prov = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        table = {}
        for name, _u, _d in COLUMNS:
            arr = f["voids"][name][()]
            table[name] = arr.astype(bool) if name in _BOOL else arr
    return table, prov


def read_csv(path) -> tuple[dict, dict]:
    """(table, provenance) back from write_csv's file: the '# key: value' lines (values as text), the
    columns as float64 (ints and 0/1 bools cast back by name)."""
    prov, header, rows = {}, None, []
    colnames = {c[0] for c in COLUMNS}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            body = line[1:].strip()
            key = body.split(":", 1)[0].strip()
            if ":" in body and key not in colnames and not key.endswith("]"):   # not a column's description line
                prov[key] = body.split(":", 1)[1].strip()
            continue
        if header is None:
            header = line.split(",")
        elif line.strip():
            rows.append(line.split(","))
    names = header or [c[0] for c in COLUMNS]
    raw = np.array(rows, dtype=object)
    table = {}
    for j, name in enumerate(names):
        col = raw[:, j] if len(rows) else np.zeros(0, dtype=object)
        if name in _BOOL:
            table[name] = np.array([c == "1" for c in col], dtype=bool)
        elif name in _INT:
            table[name] = np.array([int(c) for c in col], dtype=np.int64)
        else:
            table[name] = np.array([float(c) for c in col], dtype=np.float64)
    return table, prov


def export_voids(products, out_dir, formats=("csv", "hdf5"), stem: str | None = None) -> list[Path]:
    """Write the void catalogue of 'products' (pipeline.SnapshotProducts) under out_dir as
    voids_<sim>_<method>_<snapshot>_z<z>.csv and .h5 (the suffix APPENDED to the stem: with_suffix would
    eat the redshift's decimals). Returns the paths written. Raises ValueError on a frame mismatch
    (frame_of), FileNotFoundError when the snapshot has no grids."""
    cat = products.voids()
    frame = frame_of(products)
    assert_frame(cat, frame[0])
    table = void_table(cat)
    prov = provenance(products, len(table["id"]), frame)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if stem is None:
        z = prov["redshift"]
        ztxt = f"{z:.2f}" if np.isfinite(z) else "unknown"
        stem = f"voids_{prov['sim']}_{prov['method']}_{prov['snapshot']}_z{ztxt}"
    paths = []
    for fmt in formats:
        if fmt == "csv":
            paths.append(write_csv(table, prov, out / f"{stem}.csv"))
        elif fmt in ("hdf5", "h5"):
            paths.append(write_hdf5(table, prov, out / f"{stem}.h5"))
        else:
            raise ValueError(f"unknown export format {fmt!r} (csv, hdf5)")
    return paths
