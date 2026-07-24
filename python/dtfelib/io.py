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
  * memory             load() AUTO-TUNES against the machine: fields within the load budget
                       (DTFE_PY_LOAD_FRAC x available RAM; DTFE_PY_RAM_GB overrides) read
                       eagerly as before, bigger ones return a read-only memmap. Per-plane
                       access via load_slice(), streaming via iter_slabs()/field_stats() --
                       a 1024^3 velGrad (38.6 GB) plots and summarizes at MB-level RSS.

Usage:
    fs = FieldSet("$DATA_ROOT/TNG50-4-Dark/snapdir_099")                   # auto-detect method
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
    "tet_touch":         ("tetTouch",      1, True),   # --ps-exact-deposit: raw count of tetrahedra touching the cell (geometric multiplicity, runs to hundreds -- NOT a stream count; use 'streams' for that)
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

# Fields the binary writes ONCE, with no 'a_' (averaged) counterpart: the deposit produces a
# single grid for them regardless of the u/a pass (see writeOutputData in src/input_output.cc).
# Without this, averaged=True (the default) would look for a nonexistent 'ps_output.a_caustic'.
_NO_AVG_VARIANT = {"caustic", "tet_touch"}

# Velocity-derived fields are stored in Gadget u-units (u = v_pec/sqrt(a)): the binaries
# difference the snapshot velocities verbatim. FieldSet.load() converts them to peculiar
# km/s by scaling with a**exp -- exp = 0.5 for first moments (velocity and its spatial
# derivatives), 1.0 for second moments (the dispersion is a velocity VARIANCE). This
# centralizes what plot_shear_triaxial.py and plot_velDiv_den.py used to do ad hoc while
# every other consumer silently read raw u-units (1/sqrt(a) inflated at high z; 4.6x at
# z=20). At z=0 the factor is exactly 1.0, so outputs are bit-identical there.
# NOT listed: vweb/vweb_eigenvalues -- the BINARY applies sqrt(a) itself since 2026-07-18
# (scaling here would double-correct); tweb is density-based; caustic/streams/tetTouch are
# dimensionless.
_VELOCITY_SCALE_EXP = {
    "velocity": 0.5, "divergence": 0.5, "shear": 0.5, "vorticity": 0.5, "gradient": 0.5,
    "dispersion": 1.0, "dispersion_tensor": 1.0,
}

# Tolerance for comparing 'streams' against an integer multiplicity. '.streams' is a float32
# under every deposit -- the sampled one carries a sub-sample mean, the exact one the analytic
# volume-weighted multiplicity -- so an exact '== 1' test is never correct (it masked the whole
# grid under --ps-exact-deposit, where single-stream cells land on 1.0 +/- float32 eps).
STREAM_TOL = 1e-3


# ---------------------------------------------------------------- machine awareness
# The C++ side auto-tunes against the machine (src/auto_tune.h); these are the Python
# loader's equivalents. FieldSet.load(mode="auto") keeps today's eager-ndarray behavior
# whenever the field fits the budget -- memmap only kicks in where an eager load would
# have swapped (1024^3 velGrad is 38.6 GB), so existing workflows are unchanged.
#
# Env overrides (mirroring the binary's DTFE_RAM_GB style; also what makes tests deterministic):
#   DTFE_PY_RAM_GB     available-RAM override in GB
#   DTFE_PY_LOAD_FRAC  fraction of available RAM one eager load may use (default 0.5)

def _available_ram_bytes() -> int:
    """Available (not total) RAM in bytes, best effort across platforms."""
    import os
    env = os.environ.get("DTFE_PY_RAM_GB")
    if env:
        try:
            v = float(env)
            if v > 0:
                return int(v * 1e9)
        except ValueError:
            pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:  # macOS: total physical RAM (no portable 'available'; the 0.5 budget absorbs the gap)
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        if out.returncode == 0:
            return int(out.stdout.strip())
    except (OSError, ValueError):
        pass
    try:  # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(8e9)  # conservative fallback


def _load_budget_bytes() -> int:
    """Bytes one eager load() may allocate before mode='auto' switches to memmap."""
    import os
    frac = 0.5
    env = os.environ.get("DTFE_PY_LOAD_FRAC")
    if env:
        try:
            v = float(env)
            if 0 < v <= 1:
                frac = v
        except ValueError:
            pass
    return int(frac * _available_ram_bytes())


@dataclass
class SnapshotMeta:
    box_mpc: float        # comoving box size, h-free Mpc
    redshift: float
    hubble_h: float
    n_particles: int
    particle_mass: float  # 1e10 Msun, h-free
    rho_bar: float        # 1e10 Msun / Mpc^3, h-free (n * m / V)


class FieldSet:
    def __init__(self, snapdir, method: str = "auto", averaged: bool = True,
                 prefix: str | None = None):
        self.snapdir = Path(snapdir)
        if not self.snapdir.is_dir():
            raise FileNotFoundError(f"snapshot directory not found: {self.snapdir}")
        self.averaged = averaged
        if prefix is not None:
            # alternate on-disk prefix (run_ps_dtfe.sh's OUTPUT_PREFIX override), e.g.
            # prefix='ps_mw' reads ps_mw.a_den etc. Only the PS run script can write
            # alternate prefixes, so method 'auto' resolves to 'ps'; pass method='dtfe'
            # explicitly for a hand-produced standard-DTFE set.
            prefix = prefix.rstrip(".") + "."
            self.method = "ps" if method == "auto" else method
        else:
            self.method = self._detect_method() if method == "auto" else method
        if self.method not in _PREFIX:
            raise ValueError(f"method must be 'ps', 'dtfe' or 'auto', got {method!r}")
        self.prefix = _PREFIX[self.method] if prefix is None else prefix
        if prefix is not None and not self._has_fields(prefix):
            raise FileNotFoundError(
                f"no field grids with prefix {prefix!r} in {self.snapdir} "
                f"(written with run_ps_dtfe.sh OUTPUT_PREFIX={prefix.rstrip('.')}?)")
        self.meta = self._read_meta()
        self._grid_n: int | None = None

    # ------------------------------------------------------------------ discovery
    def _has_fields(self, prefix: str) -> bool:
        # only real field grids count -- run logs and other same-prefix artifacts (e.g. the
        # ps_output.runlog an aborted run leaves behind) must not trigger detection, or a
        # snapshot with no usable PS grids would be classified 'ps' and then fail to load
        suffixes = {s for s, _, _ in FIELDS.values()}
        return any(p.name[len(prefix):].removeprefix("a_") in suffixes
                   for p in self.snapdir.glob(prefix + "*"))

    def _detect_method(self) -> str:
        ps = self._has_fields("ps_output.")
        dt = self._has_fields("output.")
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
        avg = "a_" if (self.averaged and name not in _NO_AVG_VARIANT) else ""
        return self.snapdir / f"{self.prefix}{avg}{suffix}"

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
    @property
    def velocity_scale(self) -> float:
        """sqrt(a): the u-units -> peculiar km/s factor for FIRST velocity moments.

        Second moments (the dispersion) scale with its square, a. load() applies these
        automatically per _VELOCITY_SCALE_EXP; exactly 1.0 at z=0.
        """
        return float(np.sqrt(1.0 / (1.0 + self.meta.redshift)))

    def _scale_factor32(self, name: str) -> np.float32 | None:
        """The in-place u-units -> peculiar-km/s factor for 'name', or None when it is 1
        (z=0, or a field that carries no velocity units)."""
        exp = _VELOCITY_SCALE_EXP.get(name)
        if exp is None:
            return None
        factor = np.float32((1.0 / (1.0 + self.meta.redshift)) ** exp)
        return None if factor == np.float32(1.0) else factor

    def field_nbytes(self, name: str) -> int:
        """On-disk size of a field's grid in bytes, without reading it."""
        return self._field_path(name).stat().st_size

    def _open_memmap(self, name: str) -> np.memmap:
        """Read-only memmap of the raw on-disk grid, reshaped and size-checked."""
        path = self._field_path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} (field {name!r}, method {self.method!r})")
        ncomp = FIELDS[name][1]
        n = self.grid_n
        if path.stat().st_size != n ** 3 * ncomp * 4:
            raise ValueError(f"{path}: size is not {n}^3 x {ncomp} float32")
        shape = (n, n, n) if ncomp == 1 else (n, n, n, ncomp)
        return np.memmap(path, dtype=np.float32, mode="r", shape=shape)

    def load(self, name: str, scaled: bool = True, mode: str = "auto") -> np.ndarray:
        """Field as stored on disk, shape (N,N,N) or (N,N,N,ncomp) -- with velocity-derived
        fields converted from Gadget u-units to peculiar km/s (see _VELOCITY_SCALE_EXP;
        a no-op at z=0). Pass scaled=False for the raw on-disk numbers.

        mode='auto' (default) AUTO-TUNES the read against the machine, mirroring the
        binary's auto_tune: a field within the load budget (DTFE_PY_LOAD_FRAC x available
        RAM, DTFE_PY_RAM_GB overrides detection) is read eagerly exactly as before --
        scaling happens IN PLACE on the freshly read buffer, so there is no second
        full-grid copy -- while a larger one comes back as a READ-ONLY np.memmap that the
        kernel pages against the file (velGrad at 1024^3 is 38.6 GB; slicing and reductions
        work, writes raise). mode='ram' / mode='memmap' force either path.

        A memmap cannot carry the u-units factor without materializing the cube, so an
        over-budget field that NEEDS scaling (velocity-derived, z>0) raises MemoryError
        naming the escape hatches: load_slice() for plot planes, iter_slabs() for
        streaming statistics, scaled=False for raw u-units, or mode='ram' to accept the
        memory hit. At z=0 (every factor exactly 1) memmap serves ANY field."""
        if mode not in ("auto", "ram", "memmap"):
            raise ValueError(f"mode must be 'auto', 'ram' or 'memmap', got {mode!r}")
        factor = self._scale_factor32(name) if scaled else None
        if mode == "auto":
            mode = "ram" if self.field_nbytes(name) <= _load_budget_bytes() else "memmap"
        if mode == "memmap":
            if factor is not None:
                raise MemoryError(
                    f"field {name!r} ({self.field_nbytes(name)/1e9:.1f} GB) exceeds the load "
                    f"budget ({_load_budget_bytes()/1e9:.1f} GB) but needs the u-units factor "
                    f"{float(factor):.4g} (z={self.meta.redshift:.2f}), which a read-only "
                    f"memmap cannot carry. Use load_slice() for plot planes, iter_slabs() for "
                    f"streaming access, scaled=False for raw u-units, or mode='ram' to force "
                    f"an eager load.")
            return self._open_memmap(name)
        path = self._field_path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} (field {name!r}, method {self.method!r})")
        ncomp = FIELDS[name][1]
        data = np.fromfile(path, dtype=np.float32)
        n = self.grid_n
        if data.size != n ** 3 * ncomp:
            raise ValueError(f"{path}: {data.size} floats != {n}^3 x {ncomp}")
        if factor is not None:
            data *= factor                    # in place: fromfile owns the buffer
        return data.reshape((n, n, n) if ncomp == 1 else (n, n, n, ncomp))

    def load_slice(self, name: str, axis: int = 2, index: int | None = None,
                   scaled: bool = True) -> np.ndarray:
        """One plane of a field -- shape (N,N) or (N,N,ncomp) -- WITHOUT loading the cube.

        Reads via a read-only memmap and copies out only the requested plane, so peak
        memory is one plane (4 MB at 1024^3) instead of the full grid (4.3-38.6 GB).
        index=None takes the mid-plane, matching FieldSet.slice(). Same u-units ->
        peculiar km/s scaling as load().

        I/O note: the grids are row-major, so axis=0 planes are contiguous on disk (a
        single small read); axis 1/2 planes are strided and stream through the file's
        pages -- still bounded memory, just more read traffic. Either way nothing close
        to the cube is ever resident.
        """
        if not 0 <= axis <= 2:
            raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
        n = self.grid_n
        index = n // 2 if index is None else index
        if not 0 <= index < n:
            raise IndexError(f"slice index {index} outside [0, {n})")
        mm = self._open_memmap(name)
        sl = [slice(None)] * 3
        sl[axis] = index
        plane = np.array(mm[tuple(sl)])       # copy out the plane; the memmap is then dropped
        del mm
        if scaled:
            factor = self._scale_factor32(name)
            if factor is not None:
                plane *= factor
        return plane

    def iter_slabs(self, name: str, scaled: bool = True, max_bytes: int | None = None):
        """Stream a field as (start, stop, slab) tuples of axis-0 slabs, each a SCALED,
        writable ndarray -- bounded memory at any grid size, for statistics/histograms
        over grids that must not be loaded whole.

        Axis-0 slabs are contiguous on disk, so this is pure sequential I/O. Slab
        thickness auto-tunes so one slab stays within max_bytes (default: 1/8 of the
        load budget, at least one plane). Concatenating every slab reproduces
        load(name, mode='ram') exactly, scaling included."""
        mm = self._open_memmap(name)
        n = self.grid_n
        row_bytes = int(np.prod(mm.shape[1:], dtype=np.int64)) * 4
        if max_bytes is None:
            max_bytes = max(_load_budget_bytes() // 8, row_bytes)
        thick = max(1, min(n, int(max_bytes // row_bytes)))
        factor = self._scale_factor32(name) if scaled else None
        for start in range(0, n, thick):
            stop = min(start + thick, n)
            slab = np.array(mm[start:stop])   # copy: writable, detached from the mapping
            if factor is not None:
                slab *= factor
            yield start, stop, slab

    def field_stats(self, name: str, scaled: bool = True) -> dict:
        """Streaming min/max/mean/std of the full (scaled) field at slab-bounded memory:
        the run-summary numbers for any grid size. float64 accumulation."""
        cnt = 0
        s = s2 = 0.0
        mn, mx = np.inf, -np.inf
        for _, _, slab in self.iter_slabs(name, scaled=scaled):
            d = slab.astype(np.float64, copy=False)
            cnt += d.size
            s += float(d.sum())
            s2 += float(np.square(d).sum())
            mn = min(mn, float(d.min()))
            mx = max(mx, float(d.max()))
        mean = s / cnt
        var = max(s2 / cnt - mean * mean, 0.0)
        return {"n": cnt, "min": mn, "max": mx, "mean": mean, "std": var ** 0.5}

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
        """Velocity (N,N,N,3) km/s with NaN wherever the stream multiplicity differs from 1.

        PS-DTFE only (raises ValueError under method='dtfe'). In single-stream regions the
        multi-stream velocity reduces to the standard DTFE flow, so this isolates the cold,
        single-valued flow (void interiors) from multi-stream walls and caustics.

        The mask is a TOLERANCE (|streams - 1| > 1e-3), never an exact float equality: only
        the sampled deposit's raw grid holds integers. Under '--ps-exact-deposit' '.streams'
        is the analytic volume-weighted multiplicity (1/V_cell) sum V_int -- a float32 that
        sits at 1.0 +/- eps in single-stream cells, so an '!= 1' test masked the whole grid.
        The plateau is sharp either way (the pancake check finds every single-stream cell
        inside 1e-5 of 1.0), so 1e-3 admits float noise without admitting real structure.

        Which grid is masked follows this FieldSet: averaged=True uses the fractional
        'a_streams' sub-sample mean (any partially multi-stream cell is masked too);
        averaged=False uses the raw '.streams' grid.
        """
        if self.method != "ps":
            raise ValueError("velocity_single_stream() needs PS-DTFE outputs (method='ps'); "
                             "the stream-count field is only produced by PS-DTFE")
        # mode='ram': the NaN masking below WRITES into the array, so it must never be the
        # read-only memmap that mode='auto' returns for over-budget grids. The streams grid
        # is only compared against, so auto (possibly memmap) is fine there.
        vel = self.load("velocity", mode="ram")
        vel[np.abs(self.load("streams") - 1.0) > STREAM_TOL] = np.nan
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
        pfx = "" if self.prefix == _PREFIX[self.method] else f", prefix={self.prefix!r}"
        return (f"FieldSet({self.snapdir.name}, method={self.method!r}{pfx}, "
                f"grid={self.grid_n}^3, box={m.box_mpc:.2f} Mpc, z={m.redshift:.2f}, "
                f"rho_bar={m.rho_bar:.3f}e10 Msun/Mpc^3)")


class PointPlane:
    """Point-evaluated image plane: the '<prefix>.pts_*' outputs of a '--sample-points' run
    on a plane generated by tools/make_image_plane.py, plus that tool's JSON sidecar.

    Grid-free continuous-field maps: each field is returned as the CENTRAL sampled plane,
    a (nv, nu[, c]) array at the plane file's pixel resolution. Units mirror FieldSet.load:
    velocities (and the velocity gradient) are scaled by sqrt(a) to peculiar km/s [/Mpc],
    the dispersion (a velocity VARIANCE) by a; density is rho/rho_bar; exact no-op at z=0.

        pp = PointPlane(snapdir / "ps_output", simdir / "hires_plane_z8192.json")
        rho = pp.field("density")          # (nv, nu)
        div = pp.field("velDiv")           # from .pts_velGrad (--pts-vel-grad runs)

    Derived names (velDiv/velShear/velVort) need '.pts_velGrad'; available() lists what the
    on-disk files support. Files are float64 as written by the binary; no memmap is needed
    (a single 8192^2 plane is ~0.5 GB).
    """

    #: field -> (file extension, components, u-unit scale exponent as in _VELOCITY_SCALE_EXP)
    _PTS_FIELDS = {
        "density":    (".pts_den",     1, None),
        "streams":    (".pts_streams", 1, None),
        "velocity":   (".pts_vel",     3, 0.5),
        "dispersion": (".pts_velDisp", 6, 1.0),
        "velGrad":    (".pts_velGrad", 9, 0.5),
        "denGrad":    (".pts_denGrad", 3, None),
    }
    _DERIVED = ("speed", "dispTrace", "dispMag", "velDiv", "velShear", "velVort", "denGradMag")

    def __init__(self, prefix, plane_json, redshift: float | None = None,
                 project: str = "plane"):
        import json
        self.prefix = Path(prefix)
        self.side = json.loads(Path(plane_json).read_text())
        self.nu, self.nv = int(self.side["nu"]), int(self.side["nv"])
        self.planes = int(self.side["planes"])
        # KxK sub-samples per output pixel (make_image_plane --supersample): averaging the
        # block turns the POINT sample into an estimate of the pixel-AREA average, the
        # quantity the deposit grid reports. 1 = plain point sampling (the default).
        self.supersample = int(self.side.get("supersample", 1))
        # 'plane' = the central sampling plane (a zero-thickness cross-section);
        # 'slab' = mean over all planes, i.e. the slab-VOLUME average when combined with
        # enough planes (see make_image_plane --planes; few planes ghost).
        if project not in ("plane", "slab"):
            raise ValueError("project must be 'plane' or 'slab'")
        self.project = project
        # The plane file is generated ONCE per simulation and reused for every snapshot, so
        # its sidecar redshift belongs to whichever snapshot defined the geometry -- using it
        # would apply the wrong sqrt(a) to every other snapshot's velocities. Authoritative
        # order: explicit argument > this snapshot's own header > sidecar (last resort).
        if redshift is not None:
            self.redshift = float(redshift)
        else:
            self.redshift = self._redshift_from_snapdir()
            if self.redshift is None:
                self.redshift = float(self.side.get("redshift", 0.0))
                print(f"[dtfelib] PointPlane: no combined_*.hdf5 next to {self.prefix.name}; "
                      f"falling back to the plane sidecar's z={self.redshift:.3f} for the "
                      f"velocity scaling -- pass redshift= if that is not this snapshot's z")
        n_disk = Path(f"{self.prefix}.pts_den").stat().st_size // 8
        k = self.supersample
        if n_disk != self.planes * self.nv * k * self.nu * k:
            raise ValueError(
                f"{self.prefix}.pts_den holds {n_disk} points but the sidecar describes "
                f"{self.planes}x{self.nv * k}x{self.nu * k}; wrong plane file for this run?")

    def _redshift_from_snapdir(self) -> float | None:
        """This snapshot's redshift from its own combined_*.hdf5 (the prefix's directory)."""
        try:
            import h5py
        except ImportError:
            return None
        snaps = sorted(self.prefix.parent.glob("combined_*.hdf5"))
        if not snaps:
            return None
        try:
            with h5py.File(snaps[0], "r") as f:
                return float(f["Header"].attrs["Redshift"])
        except (OSError, KeyError):
            return None

    def _scale(self, exp: float | None) -> float:
        return 1.0 if exp is None else float((1.0 / (1.0 + self.redshift)) ** exp)

    def _load(self, name: str) -> np.ndarray:
        ext, comps, exp = self._PTS_FIELDS[name]
        dtype = np.int32 if name == "streams" else np.float64
        k = self.supersample
        raw = np.fromfile(f"{self.prefix}{ext}", dtype=dtype)
        shape = (self.planes, self.nv * k, self.nu * k) + ((comps,) if comps > 1 else ())
        arr = raw.reshape(shape)
        arr = arr.mean(axis=0) if self.project == "slab" else arr[self.planes // 2]
        if k > 1:                                   # average each KxK sub-sample block
            tail = (comps,) if comps > 1 else ()
            arr = arr.reshape((self.nv, k, self.nu, k) + tail).mean(axis=(1, 3))
        s = self._scale(exp)
        return arr if s == 1.0 else arr * s

    def available(self):
        base = [n for n, (ext, _, _) in self._PTS_FIELDS.items()
                if Path(f"{self.prefix}{ext}").is_file()]
        out = list(base)
        if "velocity" in base:
            out.append("speed")
        if "dispersion" in base:
            out += ["dispTrace", "dispMag"]
        if "velGrad" in base:
            out += ["velDiv", "velShear", "velVort"]
        if "denGrad" in base:
            out.append("denGradMag")
        return out

    def field(self, name: str) -> np.ndarray:
        if name in self._PTS_FIELDS:
            return self._load(name)
        if name == "speed":
            v = self._load("velocity")
            return np.sqrt((v ** 2).sum(axis=-1))
        if name == "dispTrace":
            d = self._load("dispersion")                     # xx xy xz yy yz zz
            tr = d[..., 0] + d[..., 3] + d[..., 5]
            return np.sqrt(np.maximum(tr, 0.0))              # 3D velocity dispersion sigma
        if name == "dispMag":
            d = self._load("dispersion")
            xx, xy, xz, yy, yz, zz = (d[..., i] for i in range(6))
            return np.sqrt(xx**2 + yy**2 + zz**2 + 2*(xy**2 + xz**2 + yz**2))
        if name in ("velDiv", "velShear", "velVort"):
            # [d*3+j] = dv_j/dx_d (the binary's layout) -> g[d, j]
            g = self._load("velGrad").reshape(self.nv, self.nu, 3, 3)
            if name == "velDiv":
                return g[..., 0, 0] + g[..., 1, 1] + g[..., 2, 2]
            if name == "velVort":
                wx = g[..., 1, 2] - g[..., 2, 1]             # eps_kij dv_j/dx_i
                wy = g[..., 2, 0] - g[..., 0, 2]
                wz = g[..., 0, 1] - g[..., 1, 0]
                return np.sqrt(wx**2 + wy**2 + wz**2)
            sym = 0.5 * (g + np.swapaxes(g, -1, -2))
            div3 = (g[..., 0, 0] + g[..., 1, 1] + g[..., 2, 2]) / 3.0
            for i in range(3):
                sym[..., i, i] -= div3
            return np.sqrt((sym ** 2).sum(axis=(-1, -2)))
        if name == "denGradMag":
            return np.sqrt((self._load("denGrad") ** 2).sum(axis=-1))
        raise KeyError(f"unknown point field {name!r}; known: "
                       f"{', '.join(list(self._PTS_FIELDS) + list(self._DERIVED))}")

    @property
    def extent(self):
        """[u0, u1, v0, v1] in box units (h-free Mpc) for imshow."""
        return [self.side["u0"], self.side["u1"], self.side["v0"], self.side["v1"]]

    def __repr__(self):
        extra = f", {self.supersample}x{self.supersample} sub-samples" if self.supersample > 1 else ""
        return (f"PointPlane({self.prefix.name}, {self.nu}x{self.nv}x{self.planes}{extra}, "
                f"axis={self.side.get('axis')}, project={self.project}, "
                f"z={self.redshift:.2f}, fields: {', '.join(self.available())})")
