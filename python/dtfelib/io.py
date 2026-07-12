"""Unified loader for DTFE / PS-DTFE raw field grids.

One class, FieldSet, hides everything that differs between the two estimators so every
analysis/plot script works with either method unchanged:

  * file naming        ps_output.a_den  vs  output.a_den   (method='ps' | 'dtfe' | 'auto')
  * DENSITY UNITS      PS-DTFE writes PHYSICAL density (1e10 Msun / Mpc^3, box mean ~2.66 for
                       TNG50-x-Dark); standard DTFE writes density NORMALIZED TO THE BOX MEAN
                       (rho/rho_bar, mean ~1, from factor=(NO_DIM+1)/averageDensity in
                       triangulation.cpp). FieldSet.density(units=...) converts either way using
                       rho_bar derived from the snapshot header, so scripts never see the
                       difference. This asymmetry is what made old cross-method figures differ
                       by ~1e8; files produced before the 2026-06-17 h-free unit overhaul of
                       combined_*.hdf5 are stale on top of that and should be regenerated.
  * metadata           grid N inferred from file size; box size [Mpc], redshift, h, particle
                       mass/count read from the snapshot's combined_*.hdf5 header (kills the
                       hardcoded BOX_SIZE / FIELD_RESOLUTION / REDSHIFT constants).
  * availability       PS-only fields (streams, dispersion, ...) raise a clear error under
                       method='dtfe'; use .has(name) to branch.

Usage:
    fs = FieldSet("/Users/luukw/output/TNG50-4-Dark/snapdir_099")          # auto-detect method
    rho   = fs.density(units="mean")            # rho/rho_bar for BOTH methods
    v     = fs.load("velocity")                 # (N,N,N,3) km/s
    v1    = fs.velocity_single_stream()         # velocity, NaN where stream count != 1 (PS only)
    if fs.has("streams"): s = fs.load("streams")
    print(fs)                                   # method, grid, box, z, rho_bar
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# canonical field name -> (file suffix, components, ps_only)
# The 'a_' averaged prefix is inserted automatically (averaged=True default).
FIELDS = {
    "density":           ("den",           1, False),
    "velocity":          ("vel",           3, False),
    "streams":           ("streams",       1, True),
    "caustic":           ("caustic",       1, True),   # 0/1 fold-caustic cell flag (--ps-caustics)
    "dispersion":        ("velDisp",       1, True),
    "dispersion_tensor": ("velDispTensor", 6, True),   # xx,xy,xz,yy,yz,zz
    "divergence":        ("velDiv",        1, False),
    "shear":             ("velShear",      5, False),
    "vorticity":         ("velVort",       3, False),
    "gradient":          ("velGrad",       9, False),
    "tweb":              ("tweb",          1, False),
    "tweb_eigenvalues":  ("twebEig",       3, False),
    "vweb":              ("velVweb",       1, False),
    "vweb_eigenvalues":  ("velVwebEig",    3, False),
}

_PREFIX = {"ps": "ps_output.", "dtfe": "output."}


@dataclass
class SnapshotMeta:
    box_mpc: float        # comoving box size, h-free Mpc
    redshift: float
    hubble_h: float
    n_particles: int
    particle_mass: float  # 1e10 Msun, h-free
    rho_bar: float        # 1e10 Msun / Mpc^3, h-free (n * m / V)


class FieldSet:
    def __init__(self, snapdir, method: str = "auto", averaged: bool = True):
        self.snapdir = Path(snapdir)
        if not self.snapdir.is_dir():
            raise FileNotFoundError(f"snapshot directory not found: {self.snapdir}")
        self.averaged = averaged
        self.method = self._detect_method() if method == "auto" else method
        if self.method not in _PREFIX:
            raise ValueError(f"method must be 'ps', 'dtfe' or 'auto', got {method!r}")
        self.meta = self._read_meta()
        self._grid_n: int | None = None

    # ------------------------------------------------------------------ discovery
    def _detect_method(self) -> str:
        # only real field grids count -- run logs and other same-prefix artifacts (e.g. the
        # ps_output.runlog an aborted run leaves behind) must not trigger detection, or a
        # snapshot with no usable PS grids would be classified 'ps' and then fail to load
        suffixes = {s for s, _, _ in FIELDS.values()}
        def has_fields(prefix: str) -> bool:
            return any(p.name[len(prefix):].removeprefix("a_") in suffixes
                       for p in self.snapdir.glob(prefix + "*"))
        ps = has_fields("ps_output.")
        dt = has_fields("output.")
        if ps and dt:
            # both estimators on disk: default deterministically to PS-DTFE (the primary
            # estimator of this project) rather than erroring -- but say so, loudly.
            print(f"[dtfelib] {self.snapdir.name}: both ps_output.* and output.* present; "
                  "using method='ps' (pass --method dtfe to switch)")
            return "ps"
        if ps: return "ps"
        if dt: return "dtfe"
        raise FileNotFoundError(f"no DTFE output fields (ps_output.*/output.*) in {self.snapdir}")

    def _field_path(self, name: str) -> Path:
        if name not in FIELDS:
            raise KeyError(f"unknown field {name!r}; known: {', '.join(FIELDS)}")
        suffix, _, ps_only = FIELDS[name]
        if ps_only and self.method != "ps":
            raise ValueError(f"field {name!r} is only produced by PS-DTFE (method='ps')")
        avg = "a_" if self.averaged else ""
        return self.snapdir / f"{_PREFIX[self.method]}{avg}{suffix}"

    def has(self, name: str) -> bool:
        try:
            return self._field_path(name).exists()
        except (KeyError, ValueError):
            return False

    @property
    def grid_n(self) -> int:
        if self._grid_n is None:
            for probe in ("density", "velocity"):
                p = self._field_path(probe)
                if p.exists():
                    ncomp = FIELDS[probe][1]
                    n = round((p.stat().st_size / 4 / ncomp) ** (1 / 3))
                    if n ** 3 * ncomp * 4 != p.stat().st_size:
                        raise ValueError(f"{p}: size is not an N^3 float32 grid")
                    self._grid_n = n
                    break
            else:
                raise FileNotFoundError(f"no density/velocity file to infer grid size in {self.snapdir}")
        return self._grid_n

    # ------------------------------------------------------------------ metadata
    def _read_meta(self) -> SnapshotMeta:
        import h5py
        snaps = sorted(self.snapdir.glob("combined_*.hdf5"))
        if not snaps:
            raise FileNotFoundError(f"no combined_*.hdf5 snapshot in {self.snapdir} (needed for units)")
        with h5py.File(snaps[0], "r") as f:
            h = f["Header"].attrs
            box_mpc = float(h["BoxSize"]) / 1000.0          # h-free ckpc -> Mpc (see tools/merge_HDF5.py)
            npart = int(h["NumPart_ThisFile"][1])
            mass = float(h["MassTable"][1])                  # 1e10 Msun, h-free
            if mass <= 0:                                    # per-particle masses instead of table
                mass = float(f["PartType1/Masses"][0])
            if "HFreeUnits" not in h:
                # combined files from before the units unification copied the raw TNG
                # MassTable (1e10 Msun/h) verbatim; normalize so meta is h-free either way
                mass /= float(h["HubbleParam"])
            return SnapshotMeta(
                box_mpc=box_mpc,
                redshift=float(h["Redshift"]),
                hubble_h=float(h["HubbleParam"]),
                n_particles=npart,
                particle_mass=mass,
                rho_bar=npart * mass / box_mpc ** 3,
            )

    # ------------------------------------------------------------------ loading
    def load(self, name: str) -> np.ndarray:
        """Raw field as stored on disk. Shape (N,N,N) or (N,N,N,ncomp)."""
        path = self._field_path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} (field {name!r}, method {self.method!r})")
        ncomp = FIELDS[name][1]
        data = np.fromfile(path, dtype=np.float32)
        n = self.grid_n
        if data.size != n ** 3 * ncomp:
            raise ValueError(f"{path}: {data.size} floats != {n}^3 x {ncomp}")
        return data.reshape((n, n, n) if ncomp == 1 else (n, n, n, ncomp))

    def density(self, units: str = "mean") -> np.ndarray:
        """Density with METHOD-INDEPENDENT units.

        units='mean'     -> rho / rho_bar   (dimensionless; 1 = cosmic mean)
        units='physical' -> 1e10 Msun / Mpc^3 (comoving, h-free)

        The on-disk convention is AUTO-DETECTED from the grid's own mean (mass conservation
        pins it to either 1 or rho_bar): current binaries write rho/rho_bar for BOTH methods
        (PS-DTFE switched 2026-07); PS files from before that store the physical mass density
        and still load correctly. A grid whose mean matches neither convention (e.g. standard
        DTFE outputs from before the 2026-06-17 unit overhaul) triggers a loud warning --
        regenerate those with the current binary before using absolute values.
        """
        raw = self.load("density")
        conv = self._density_convention(raw)
        if units == "mean":
            return raw if conv == "mean" else raw / self.meta.rho_bar
        if units == "physical":
            return raw * self.meta.rho_bar if conv == "mean" else raw
        raise ValueError("units must be 'mean' or 'physical'")

    def _density_convention(self, raw: np.ndarray) -> str:
        """'mean' (grid mean ~1) or 'physical' (grid mean ~rho_bar), detected by sampling."""
        m = float(np.mean(raw.flat[::997], dtype=np.float64))
        if 0.2 < m < 5.0:
            return "mean"
        ratio = m / self.meta.rho_bar
        if 0.2 < ratio < 5.0:
            return "physical"
        # neither convention: stale pre-overhaul grid; pick the closer one in log space but say so
        guess = "mean" if abs(np.log10(max(m, 1e-30))) < abs(np.log10(max(ratio, 1e-30))) else "physical"
        print(f"[dtfelib] WARNING: {self.snapdir.name}: density grid mean {m:.3e} matches neither "
              f"rho/rho_bar (~1) nor the physical mean ({self.meta.rho_bar:.3e}); this looks like a "
              f"stale pre-2026-06-17 output -- absolute values are unreliable, regenerate with the "
              f"current binary. Proceeding as units='{guess}'.")
        return guess

    def velocity_single_stream(self) -> np.ndarray:
        """Velocity (N,N,N,3) km/s with NaN wherever the stream count != 1.

        PS-DTFE only (raises ValueError under method='dtfe'). In single-stream regions the
        multi-stream velocity reduces to the standard DTFE flow, so this isolates the cold,
        single-valued flow (void interiors) from multi-stream walls and caustics. The mask
        uses this FieldSet's streams grid: with averaged=True that is the fractional
        'a_streams' sub-sample mean (any partially multi-stream cell is masked too); pass
        averaged=False for the raw integer counts (the crisp streams != 1 mask).
        """
        if self.method != "ps":
            raise ValueError("velocity_single_stream() needs PS-DTFE outputs (method='ps'); "
                             "the stream-count field is only produced by PS-DTFE")
        vel = self.load("velocity")
        vel[self.load("streams") != 1] = np.nan
        return vel

    # ------------------------------------------------------------------ helpers
    def slice(self, field: np.ndarray, axis: int = 2, index: int | None = None) -> np.ndarray:
        """Mid-plane (or given-index) slice; works for scalar and multi-component fields."""
        index = field.shape[axis] // 2 if index is None else index
        sl = [slice(None)] * 3
        sl[axis] = index
        return field[tuple(sl)]

    @property
    def cell_mpc(self) -> float:
        return self.meta.box_mpc / self.grid_n

    def __repr__(self):
        m = self.meta
        return (f"FieldSet({self.snapdir.name}, method={self.method!r}, "
                f"grid={self.grid_n}^3, box={m.box_mpc:.2f} Mpc, z={m.redshift:.2f}, "
                f"rho_bar={m.rho_bar:.3f}e10 Msun/Mpc^3)")
