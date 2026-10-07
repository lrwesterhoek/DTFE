"""The caustic skeleton (survey item 9, 2026-10-05): walls, filaments and nodes from the 'caustic_class'
bitmask of a '--ps-caustics' run, as periodic connected components, with their volume and mass fractions,
and -- per void of the catalogue -- how much caustic sits on its edge and how far the nearest wall is.

    from dtfelib import skeleton
    cc = np.rint(fs.load("caustic_class")).astype(np.int32)
    summary = skeleton.skeleton_summary(cc, density=fs.density(units="mean"), cell_mpc=fs.cell_mpc)

THE CLASSES. io.caustic_collapse_max gives every cell its highest collapse multiplicity k present (-1:
no tetrahedron ever deposited into it, 0: single-stream / uncollapsed only, 1: a wall (one axis
collapsed and inverted), 2: a filament, 3: a node). The FRACTIONS are over these EXCLUSIVE classes
('none', 'k=0', 'k=1', 'k=2', 'k=3' = webstreams.COLLAPSE_BINS, so the two modules' tables agree and the
fractions sum to 1), plus the fold (io.caustic_is_fold: both map parities in the cell), the A3 cusp and
A4 swallowtail indicators (bits 7, 8, only under --ps-caustic-cusps) and the umbilic bit 6, each as a
fraction alone (never labelled: they are proximity indicators). The COMPONENTS are built from the
CUMULATIVE masks -- 'sheet' = k >= 1 (every collapsed cell: the walls with the filaments and nodes
running through them), 'line' = k >= 2, 'node' = k = 3 -- because a wall is the connected sheet, not the
sheet with its filaments cut out. Mass weights are the density grid (rho/rho_bar or physical: a
constant cancels in every fraction).

CONNECTIVITY AND THE PERIODIC BOX. label_periodic labels a mask with scipy.ndimage.label under a chosen
connectivity (default 26 -- face, edge and corner neighbours: a sheet tilted to the grid falls apart into
stripes under the 6-connected default) on the mask padded by one wrapped cell, then unions every halo
cell's label with the label of the interior cell it copies (aligned pairs across each face: they ARE the
same cell) through scipy.sparse.csgraph.connected_components -- no Python loop over labels, there can be
millions at 512^3 -- and relabels the interior consecutively, largest component first.

PER VOID (void_shell_fractions, nearest_wall_distance): the fold and sheet fractions of the shell 1.0-1.3
R_eff (profiles.shell_means by nearest cell, the void's cell centre as its centre), and the distance from
the void's centre to the nearest sheet cell, from the Euclidean distance transform of a periodic cut-out
of 'cutout_cells' half-width around the centre (np.take with wrap); no sheet in the cut-out -> NaN with
the cut-out's half-width as the lower bound.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .io import (CAUSTIC_DEGENERATE, CAUSTIC_PARITY_NEG, CAUSTIC_PARITY_POS, caustic_collapse_max, caustic_is_cusp,
                 caustic_is_fold, caustic_is_swallowtail)

CLASSES = ("none", "k=0", "k=1", "k=2", "k=3")                 # exclusive, = webstreams.COLLAPSE_BINS
INDICATORS = ("fold", "cusp", "swallowtail", "umbilic")        # fractions only
CUMULATIVE = ("sheet", "line", "node")                         # k >= 1, k >= 2, k = 3: the components


def class_masks(caustic_class) -> dict:
    """{'exclusive': {name: bool mask} over CLASSES, 'indicator': over INDICATORS, 'cumulative': over
    CUMULATIVE} of a caustic_class grid (any integer-valued array)."""
    cc = np.rint(np.asarray(caustic_class)).astype(np.int64)
    kmax = caustic_collapse_max(cc)
    # every deposited tetrahedron sets a parity bit TOGETHER with one collapse bit; a cell with a parity bit and no
    # collapse bit comes from a parity-only grid (--ps-caustics with --ps-linear-deposit --ps-gpu writes the fold
    # flag alone), which would read as 99.96% 'none' (review 2026-10-05)
    if np.any(((cc & (CAUSTIC_PARITY_POS | CAUSTIC_PARITY_NEG)) != 0) & (kmax == -1)):
        raise ValueError("parity-only caustic_class grid (fold bits without collapse bits: a --ps-caustics "
                         "--ps-linear-deposit --ps-gpu run); the stratification needs the CPU or the GPU's full deposit")
    return {
        "exclusive": {"none": kmax == -1, "k=0": kmax == 0, "k=1": kmax == 1, "k=2": kmax == 2, "k=3": kmax == 3},
        "indicator": {"fold": caustic_is_fold(cc), "cusp": caustic_is_cusp(cc), "swallowtail": caustic_is_swallowtail(cc),
                      "umbilic": (cc & CAUSTIC_DEGENERATE) != 0},
        "cumulative": {"sheet": kmax >= 1, "line": kmax >= 2, "node": kmax == 3},
    }


def _fraction(mask, density=None) -> dict:
    n = int(mask.size)
    vol = float(mask.sum()) / n if n else float("nan")
    if density is None:
        return {"volume": vol, "mass": None}
    d = np.asarray(density, dtype=np.float64)
    tot = float(d.sum())
    return {"volume": vol, "mass": float(d[mask].sum()) / tot if tot > 0 else None}


def fractions(caustic_class, density=None) -> dict:
    """Volume (and mass, with a density grid) fractions: {'exclusive': {class: {'volume', 'mass'}},
    'indicator': {...}, 'cumulative': {...}}; the exclusive volumes sum to 1."""
    m = class_masks(caustic_class)
    return {group: {name: _fraction(mask, density) for name, mask in masks.items()} for group, masks in m.items()}


def label_periodic(mask, connectivity: int = 26) -> tuple[np.ndarray, int]:
    """(labels, n): the connected components of the boolean mask in a PERIODIC box, labelled 1..n in
    order of size (largest first), 0 outside the mask; 'connectivity' 6, 18 or 26."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("label_periodic wants a 3-D mask")
    rank = {6: 1, 18: 2, 26: 3}.get(int(connectivity))
    if rank is None:
        raise ValueError("connectivity must be 6, 18 or 26")
    structure = ndimage.generate_binary_structure(3, rank)
    padded = np.pad(mask, 1, mode="wrap")
    lab, n = ndimage.label(padded, structure=structure)
    if n == 0:
        return np.zeros(mask.shape, dtype=np.int32), 0
    a_list, b_list = [], []
    for axis in range(3):
        size = mask.shape[axis]
        for halo, src in ((0, size), (size + 1, 1)):          # the padded halo cell and the interior cell it copies
            la = np.take(lab, halo, axis=axis).ravel()
            lb = np.take(lab, src, axis=axis).ravel()
            both = (la > 0) & (lb > 0) & (la != lb)
            a_list.append(la[both])
            b_list.append(lb[both])
    a = np.concatenate(a_list)
    b = np.concatenate(b_list)
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n + 1, n + 1))
    _ncomp, comp = connected_components(graph, directed=False)
    inner = lab[1:-1, 1:-1, 1:-1]
    merged = comp[inner]                                      # component id per cell (0 shares a component only with itself)
    merged = np.where(inner > 0, merged + 1, 0)
    present, inverse = np.unique(merged, return_inverse=True)
    counts = np.bincount(inverse.ravel())
    order = [i for i in np.argsort(-counts) if present[i] != 0]
    newid = np.zeros(len(present), dtype=np.int32)
    for rank_, i in enumerate(order, start=1):
        newid[i] = rank_
    return newid[inverse].reshape(mask.shape).astype(np.int32), len(order)


def components(mask, density=None, cell_mpc: float = 1.0, connectivity: int = 26, top: int = 10) -> dict:
    """The periodic components of a mask: {'n', 'labels', 'cells' (per component, largest first),
    'volume_mpc3', 'mass_fraction' (of the whole box's mass; None without a density), 'largest_fraction'
    (the largest component's share of the mask's cells), 'top': the first 'top' sizes}."""
    labels, n = label_periodic(mask, connectivity)
    cells = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.int64)
    out = {"n": int(n), "labels": labels, "cells": cells, "volume_mpc3": cells * float(cell_mpc) ** 3,
           "largest_fraction": float(cells[0] / cells.sum()) if n else None, "top": cells[:top].tolist(),
           "mass_fraction": None}
    if density is not None and n:
        d = np.asarray(density, dtype=np.float64)
        tot = float(d.sum())
        mass = np.bincount(labels.ravel(), weights=d.ravel(), minlength=n + 1)[1:]
        out["mass_fraction"] = mass / tot if tot > 0 else None
    return out


def skeleton_summary(caustic_class, density=None, cell_mpc: float = 1.0, connectivity: int = 26, top: int = 10) -> dict:
    """fractions() plus, per CUMULATIVE class, its components' count, largest share and top sizes
    (cells and Mpc^3), as a JSON-able dict (no label grids)."""
    m = class_masks(caustic_class)
    out = {"fractions": fractions(caustic_class, density), "cell_mpc": float(cell_mpc),
           "connectivity": int(connectivity), "components": {},
           # bits 7 and 8 exist only under --ps-caustic-cusps on the CPU deposit: 0% there means 'not computed'
           "cusp_bits_present": bool(m["indicator"]["cusp"].any() or m["indicator"]["swallowtail"].any())}
    for name in CUMULATIVE:
        c = components(m["cumulative"][name], density, cell_mpc, connectivity, top)
        out["components"][name] = {"n": c["n"], "largest_fraction": c["largest_fraction"], "top_cells": c["top"],
                                   "top_volume_mpc3": (np.asarray(c["top"]) * float(cell_mpc) ** 3).tolist(),
                                   "top_mass_fraction": None if c["mass_fraction"] is None else c["mass_fraction"][:top].tolist()}
    return out


def void_shell_fractions(caustic_class, centers_frac, r_eff_cells, shells=(1.0, 1.1, 1.2, 1.3), m: int = 200) -> dict:
    """Per void: the fold and sheet fractions of its shells x R_eff (nearest cell, profiles.shell_means),
    averaged over the shells: {'fold': (n,), 'sheet': (n,), 'shells': x, 'per_shell': {name: (n, len(x))}}."""
    from .profiles import fibonacci_sphere, shell_means
    masks = class_masks(caustic_class)
    dirs = fibonacci_sphere(m)
    centers = np.asarray(centers_frac, dtype=np.float64).reshape(-1, 3)
    r_eff = np.asarray(r_eff_cells, dtype=np.float64).reshape(-1)
    out = {"shells": np.asarray(shells, dtype=np.float64), "per_shell": {}}
    for name, mask in (("fold", masks["indicator"]["fold"]), ("sheet", masks["cumulative"]["sheet"])):
        per = np.empty((len(r_eff), len(shells)))
        for j, x in enumerate(shells):
            per[:, j] = shell_means(mask.astype(np.float32), centers, x * r_eff, dirs, mode="ngp")
        out["per_shell"][name] = per
        out[name] = per.mean(axis=1)
    return out


def _wall_distance_in(mask, ci, half):
    """The distance (cells) from integer centre 'ci' to the nearest True cell of 'mask' within a periodic cut-out
    of half-width 'half', or None when none lies within 'half' (one there could sit outside the cut-out)."""
    n = mask.shape[0]
    offs = np.arange(-half, half + 1)
    idx = [(ci[k] + offs) % n for k in range(3)]
    cut = mask[np.ix_(idx[0], idx[1], idx[2])]
    if not cut.any():
        return None
    if cut[half, half, half]:
        return 0.0
    d = float(ndimage.distance_transform_edt(~cut)[half, half, half])
    return d if d <= half else None                          # beyond 'half': the nearest may sit outside the cut-out


def nearest_wall_distance(sheet_mask, centers_cells, cutout_cells: int = 32, start_cells: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """(distance, bounded): per centre (cells, (n, 3)), the Euclidean distance in cells to the nearest
    True cell of 'sheet_mask', searched in periodic cut-outs of half-width 'start_cells', doubled for the
    centres not yet resolved, up to 'cutout_cells'; with no sheet cell within that the distance is NaN and
    'bounded' True (the half-width is a lower bound). A wall found at d <= the half-width is THE nearest (every
    closer cell lies inside the cube), so the result is the one of a single search at 'cutout_cells' -- at a
    fraction of the cost: a 100-cell cut-out for every void was 31x slower than 32 (2026-10-07)."""
    mask = np.asarray(sheet_mask, dtype=bool)
    centers = np.asarray(centers_cells, dtype=np.float64).reshape(-1, 3)
    full = int(cutout_cells)
    dist = np.full(len(centers), np.nan)
    bounded = np.ones(len(centers), dtype=bool)
    todo = np.arange(len(centers))
    half = max(1, min(int(start_cells), full))
    while todo.size:
        left = []
        for i in todo:
            d = _wall_distance_in(mask, np.rint(centers[i]).astype(np.int64), half)
            if d is None:
                left.append(i)
            else:
                dist[i], bounded[i] = d, False
        if half >= full:
            break
        todo, half = np.asarray(left, dtype=np.int64), min(2 * half, full)
    return dist, bounded


def json_safe(obj):
    """The summary with every NaN or infinity turned into None (json.dumps writes bare NaN, which no JSON
    reader accepts); numpy scalars and arrays become Python numbers and lists."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
