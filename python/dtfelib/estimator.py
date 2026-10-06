"""Interactive DTFE / PS-DTFE: build the tessellation ONCE, then evaluate it at any points.

The batch workflow writes '.pts_*' files for one fixed '--sample-points' set per run. An
Estimator instead keeps a DTFE / PS-DTFE process alive in '--serve' mode, holding the
tessellation and its vertex densities in memory, and answers every call from it -- no
rebuild, no files:

    from dtfelib import Estimator

    with Estimator("snap_099.hdf5") as est:             # PS-DTFE; tessellation built here
        f = est(points)                                 # (N, 3) box coordinates [Mpc]
        f.density, f.velocity, f.streams, f.dispersion  # per-point numpy arrays
        rho = est.density(halo_centres)                 # convenience accessors
        cube = est.grid(xs, ys, zs)                     # fields on the tensor-product grid

Every answer is exactly what '--sample-points' would write for the same points (the server
runs the same code), including the exact, tie-consistent stream counting: a point on a
shared face, edge or vertex -- e.g. a particle position -- is counted once per stream.

Units mirror FieldSet.load / PointPlane: density is rho/rho_bar; velocities (and velocity
gradients) are converted to peculiar km/s with sqrt(a), the dispersion (a velocity variance)
with a, where a is the snapshot's header 'Time' (no-op at z=0; pass peculiar=False for the
raw Gadget units). Point coordinates are in the box coordinate system the binary uses, i.e.
Mpc after the '--MpcUnit' conversion (the binary's default is 1000, for ckpc/h snapshots).

Session cost: the build (the same as one batch run's triangulation + vertex densities) is
paid once; each call then costs roughly the cheap per-tetrahedron occupancy test over the
tessellation plus the work proportional to the points -- milliseconds to tens of ms for small
requests, and the multi-threaded batch speed for large ones. '--tessellation-cache' makes
even the build instant on a snapshot seen before (tessellation_cache=...).
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Estimator", "PointFields", "find_binary"]

_HELLO = b"DTFESRV2"
_REPLY = b"DTFEPTS1"
_ERROR = b"DTFEERR1"

# handshake flag bits (ps_point_eval.cc, '--serve' protocol version 2)
(_F_PS, _F_PER_STREAM, _F_IDS, _F_DEN_GRAD, _F_VEL_GRAD, _F_CAUSTIC, _F_GEOMETRIC, _F_PERIODIC,
 _F_SCALAR, _F_LAGRANGIAN, _F_COMPOSITE, _F_DIM2) = (1 << i for i in range(12))   # _F_DIM2: a 2D build


def find_binary(phase_space: bool = True, precision: str = "single", dim: int = 3) -> Path:
    """Locates the DTFE / PS-DTFE executable (precision 'double': DTFE-double / PS-DTFE-double, built by
    'make DTFE PS-DTFE DOUBLE=1'; dim 2: DTFE-2d / PS-DTFE-2d, 'make DTFE PS-DTFE DIM=2'):
    $DTFE_BIN_DIR, then the repository root this package lives in (python/dtfelib -> ../..), then $PATH."""
    if precision not in ("single", "double"):
        raise ValueError(f"precision must be 'single' or 'double', not {precision!r}")
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, not {dim!r}")
    name = (("PS-DTFE" if phase_space else "DTFE") + ("-2d" if dim == 2 else "")
            + ("-double" if precision == "double" else ""))
    candidates = []
    if os.environ.get("DTFE_BIN_DIR"):
        candidates.append(Path(os.environ["DTFE_BIN_DIR"]) / name)
    candidates.append(Path(__file__).resolve().parents[2] / name)
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError(
        f"cannot find the '{name}' executable. Build it ('scripts/install.sh', or "
        f"'make {name}'), or point DTFE_BIN_DIR at the directory holding it. Looked in: "
        + ", ".join(str(c) for c in candidates) + ", and $PATH.")


@dataclass
class PointFields:
    """Per-point fields of one Estimator call (the '.pts_*' outputs as arrays). The shapes are for a 3D
    server; a 2D one (dim=2) has 2 velocity components, 3 dispersion components (xx xy yy), 2x2
    velocity gradients, 3 particle IDs per stream and 2-component positions.

    density     (N,)      rho/rho_bar, summed over streams
    velocity    (N, 3)    density-weighted mean velocity over streams
    dispersion  (N, 6)    velocity dispersion tensor, xx xy xz yy yz zz (0 for 1 stream)
    streams     (N,)      number of streams (tetrahedra) containing the point; the standard
                          binary reports 0/1 coverage
    caustic     (N,)      caustic mask (caustics=True), bits as in '.causticClass'
    density_gradient  (N, 3)       d(rho/rho_bar)/dx_i (density_gradient=True)
    velocity_gradient (N, 3, 3)    [.., d, j] = dv_j/dx_d, density-weighted stream mean
                                   (velocity_gradient=True)
    scalar      (N,)      the particles' scalar value (scalars=True / scalar_dataset=...), the
                          density-weighted mean over the point's streams; (N, S) for S > 1
    per-stream decomposition (per_stream=True; ragged, point i owns rows
    offsets[i]:offsets[i+1], densest stream first):
    offsets (N+1,), stream_density (R,), stream_velocity (R, 3),
    stream_ids (R, 4) sorted Lagrangian-vertex ParticleIDs (per_stream_ids=True),
    stream_density_gradient (R, 3) (per_stream=True and density_gradient=True),
    stream_scalar (R,) each stream's own scalar value (per_stream=True and scalars=True),
    stream_lagrangian (R, 3) each stream's Lagrangian coordinate q(x) -- where the matter it
                          brings to the point started (lagrangian_positions=True)

    grid() reshapes the per-point arrays to the grid's shape; the ragged per-stream arrays
    keep their flat layout (index them through offsets.reshape(-1) as usual).
    """
    density: np.ndarray
    velocity: np.ndarray
    dispersion: np.ndarray
    streams: np.ndarray
    caustic: np.ndarray | None = None
    density_gradient: np.ndarray | None = None
    velocity_gradient: np.ndarray | None = None
    offsets: np.ndarray | None = None
    stream_density: np.ndarray | None = None
    stream_velocity: np.ndarray | None = None
    stream_ids: np.ndarray | None = None
    stream_density_gradient: np.ndarray | None = None
    scalar: np.ndarray | None = None
    stream_scalar: np.ndarray | None = None
    stream_lagrangian: np.ndarray | None = None
    shape: tuple = field(default=())

    def __len__(self) -> int:
        return int(np.prod(self.shape)) if self.shape else len(self.density)

    def stream_slice(self, i: int) -> slice:
        """Rows of the per-stream arrays that belong to (flat) point i."""
        if self.offsets is None:
            raise ValueError("per-stream records were not requested (Estimator(per_stream=True))")
        off = self.offsets.reshape(-1)
        return slice(int(off[i]), int(off[i + 1]))


class Estimator:
    """A live DTFE / PS-DTFE tessellation answering point queries (see the module docstring).

    snapshot           Gadget snapshot the binary reads (HDF5 by default, input_type=105).
    phase_space        True: PS-DTFE (multi-stream fields; needs Lagrangian positions, from the
                       snapshot's 'InitialCoordinates' or lagrangian_input). False: standard DTFE.
    periodic           periodic box (default True).
    mpc_unit           '--MpcUnit' (length of 1 Mpc in input units); None keeps the binary's
                       default (1000, for ckpc/h snapshots).
    per_stream, per_stream_ids, density_gradient, velocity_gradient, caustics
                       the matching '--sample-points' outputs (--per-stream, --per-stream-ids,
                       --pts-den-grad, --pts-vel-grad, --ps-caustics).
    stream_density     'dtfe' (default) or 'geometric' ('--ps-stream-density').
    scalars            interpolate the particles' scalar value too ('--pts-scalar'); needs a
                       scalar source, normally scalar_dataset.
    scalar_dataset     per-particle HDF5 dataset read as that scalar ('--scalar-dataset', e.g.
                       'Potential'); implies scalars=True.
    lagrangian_positions
                       PS-DTFE: also return each stream's Lagrangian coordinate q(x)
                       ('--pts-lagrangian'; implies per_stream).
    tessellation_cache directory for '--tessellation-cache' (a LOCAL, non-synced path): the
                       first session writes the tessellation, later ones load it instantly.
    partition          int or (nx, ny, nz): serve a COMPOSITE of partition tessellations
                       ('--partition') -- for boxes too large for one tessellation. With
                       tessellation_cache the partitions live on disk and at most 'resident'
                       ('--serve-resident'; default: what the memory budget allows) are held in
                       memory; answers are bit-identical to an unpartitioned server.
    max_concurrent     partitions triangulated at once while the composite starts
                       ('--max-concurrent').
    peculiar           convert velocities to peculiar km/s with the snapshot's scale factor
                       (FieldSet convention); False returns the binary's raw units.
    options            extra command-line arguments, passed through verbatim.
    binary             explicit path to the executable (default: find_binary()).
    dim                2 for a 2D snapshot and build (PS-DTFE-2d / DTFE-2d): points are (N, 2), every
                       vector has 2 components; the server's handshake says which it is.
    precision          'single' (default) or 'double': which build find_binary() picks when no binary
                       is given -- the double pair interpolates in float64 (its answers are float64 either way)
    verbose            the binary's '--verbose' level; its log goes to this process's stderr
                       when > 0 and is otherwise kept (and shown if the server fails).
    progress           the server writes plain progress lines to its log ('--serve-progress'):
                       '[partitions k/N] ...' while it builds, '[request k/P] partition p: in memory |
                       loading from the cache' and '[request done] ...' for every request -- whatever
                       'verbose' says. A launcher follows them for its progress bar and timings.
    log_path           write the server's log to this file instead, at the 'verbose' level (a
                       GUI follows the tessellation build there: '[partitions k/N]' lines).
    """

    def __init__(self, snapshot, *, phase_space: bool = True, periodic: bool = True,
                 mpc_unit: float | None = None, input_type: int = 105,
                 lagrangian_input=None, per_stream: bool = False, per_stream_ids: bool = False,
                 density_gradient: bool = False, velocity_gradient: bool = False,
                 caustics: bool = False, stream_density: str = "dtfe",
                 scalars: bool = False, scalar_dataset: str | None = None,
                 lagrangian_positions: bool = False,
                 tessellation_cache=None, partition=None, resident: int | None = None,
                 max_concurrent: int | None = None, peculiar: bool = True, options=(),
                 binary=None, precision: str = "single", dim: int = 3, verbose: int = 0, log_path=None,
                 progress: bool = False, _owned_dir=None, _shift=None, _on_spawn=None):
        self._owned_dir = Path(_owned_dir) if _owned_dir else None   # from_arrays' temporary snapshot
        # from_arrays: the server's coordinates = the caller's + _shift (it keeps the box >= 0,
        # because the binary's option parser reads a negative '--box' value as a flag)
        self._shift = None if _shift is None else np.asarray(_shift, dtype=np.float64).reshape(-1)
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, not {dim!r}")
        self.dim = int(dim)             # confirmed (or corrected) by the server's handshake
        self._closed = True             # until the server is up (close() is then a no-op)
        self.snapshot = Path(snapshot)
        if not self.snapshot.exists():
            raise FileNotFoundError(f"snapshot not found: {self.snapshot}")
        self.phase_space = bool(phase_space)
        self.binary = Path(binary) if binary else find_binary(self.phase_space, precision, self.dim)
        if stream_density not in ("dtfe", "geometric"):
            raise ValueError("stream_density must be 'dtfe' or 'geometric'")
        if lagrangian_positions and not self.phase_space:
            raise ValueError("lagrangian_positions needs the phase-space estimator (phase_space=True)")

        self._workdir = Path(tempfile.mkdtemp(prefix="dtfe-serve-"))
        cmd = [str(self.binary), str(self.snapshot), str(self._workdir / "serve"),
               "--serve", "--input", str(int(input_type)), "--verbose", str(int(verbose))]
        if periodic:
            cmd.append("--periodic")
        if mpc_unit is not None:
            cmd += ["--MpcUnit", repr(float(mpc_unit))]
        if lagrangian_input is not None:
            cmd += ["--lagrangianInput", str(lagrangian_input)]
        if per_stream:
            cmd.append("--per-stream")
        if per_stream_ids:
            cmd.append("--per-stream-ids")
        if density_gradient:
            cmd.append("--pts-den-grad")
        if velocity_gradient:
            cmd.append("--pts-vel-grad")
        if caustics:
            cmd.append("--ps-caustics")
        if self.phase_space:
            cmd += ["--ps-stream-density", stream_density]
        if scalar_dataset:
            cmd += ["--scalar-dataset", str(scalar_dataset)]
            scalars = True
        if scalars:
            cmd.append("--pts-scalar")
        if lagrangian_positions:
            cmd.append("--pts-lagrangian")
        if tessellation_cache is not None:
            cmd += ["--tessellation-cache", str(tessellation_cache)]
        if partition is not None:
            parts = [int(partition)] * self.dim if np.ndim(partition) == 0 else [int(v) for v in partition]
            if len(parts) != self.dim or min(parts) < 1:
                shutil.rmtree(self._workdir, ignore_errors=True)      # nothing started: leave no work folder
                raise ValueError(f"partition must be a positive int or {self.dim} positive ints")
            cmd += ["--partition"] + [str(v) for v in parts]
        if resident is not None:
            cmd += ["--serve-resident", str(int(resident))]
        if max_concurrent is not None:
            cmd += ["--max-concurrent", str(int(max_concurrent))]
        if progress:
            cmd.append("--serve-progress")
        cmd += [str(o) for o in options]
        self.command = cmd

        self._log = None
        try:
            if log_path is not None:
                self._log_path = Path(log_path)
                self._log = open(self._log_path, "wb")
            else:
                self._log_path = self._workdir / "server.log"
                self._log = None if verbose > 0 else open(self._log_path, "wb")
            self._lock = threading.Lock()
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=self._log)
        except BaseException:                   # the binary missing, a log that cannot be opened: close() would
            if self._log is not None:           # return early (nothing started), so clean up here
                self._log.close()
            shutil.rmtree(self._workdir, ignore_errors=True)
            raise
        self._closed = False
        if _on_spawn is not None:       # lets a caller cancel a long tessellation build
            _on_spawn(self._proc)
        try:
            self._handshake()
        except BaseException:
            self.close()
            raise

        self.scale_factor = self._read_scale_factor() if peculiar else 1.0

    # ------------------------------------------------------------------ protocol plumbing
    def _server_failure(self, what: str) -> RuntimeError:
        code = self._proc.poll()
        tail = ""
        if self._log is not None:
            self._log.flush()
            try:
                lines = self._log_path.read_text(errors="replace").splitlines()
                tail = "\n".join(lines[-25:])
            except OSError:
                pass
        msg = f"DTFE server {what} (exit code {code}).\n  command: {' '.join(self.command)}"
        if tail:
            msg += "\n  last log lines:\n" + "\n".join("    " + l for l in tail.splitlines())
        return RuntimeError(msg)

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self._proc.stdout.readinto(view[got:])
            if not k:
                raise self._server_failure("closed its output unexpectedly")
            got += k
        return bytes(buf)

    def _read_array(self, count: int, dtype) -> np.ndarray:
        dtype = np.dtype(dtype)
        out = np.empty(count, dtype=dtype)
        if count:
            view = memoryview(out).cast("B")
            got, n = 0, out.nbytes
            while got < n:
                k = self._proc.stdout.readinto(view[got:])
                if not k:
                    raise self._server_failure("closed its output mid-reply")
                got += k
        return out

    def _handshake(self):
        magic = self._proc.stdout.read(8)
        if magic != _HELLO:
            if magic == b"DTFESRV1":
                raise self._server_failure("speaks the old serve protocol (version 1); rebuild the "
                                           "binary ('make PS-DTFE' / 'make DTFE')")
            raise self._server_failure("failed while building the tessellation"
                                       if not magic else f"sent an unexpected greeting {magic!r}")
        flags, n_scalar, nvert = struct.unpack("<IIQ", self._read_exact(16))
        dim = 2 if flags & _F_DIM2 else 3      # 3D servers never set the bit
        region = struct.unpack(f"<{2 * dim}d", self._read_exact(16 * dim))
        n_parts, n_resident = struct.unpack("<II", self._read_exact(8))
        self.flags = flags
        self.n_scalar = int(n_scalar)
        self.n_vertices = int(nvert)
        self.n_partitions = int(n_parts)
        self.resident = int(n_resident)
        if dim != self.dim:
            raise RuntimeError(f"{self.binary.name} is a {dim}D build: pass dim={dim} to the Estimator")
        self.region = np.asarray(region, dtype=np.float64).reshape(dim, 2)
        if self._shift is not None:
            self.region = self.region - self._shift[:, None]
        if bool(flags & _F_PS) != self.phase_space:
            raise RuntimeError(f"{self.binary.name} is not a "
                               f"{'PS-DTFE' if self.phase_space else 'standard DTFE'} binary")

    def _read_scale_factor(self) -> float:
        try:
            import h5py
            with h5py.File(self.snapshot, "r") as f:
                return float(f["Header"].attrs["Time"])
        except Exception:      # not HDF5, no h5py, no header: raw units are all we can offer
            return 1.0

    # ------------------------------------------------------------------ queries
    def __call__(self, points) -> PointFields:
        """Evaluates every field at 'points' ((N, dim) or (dim,), box coordinates; dim = 3 or 2)."""
        pts = np.ascontiguousarray(points, dtype=np.float64)
        single = pts.ndim == 1
        d = self.dim
        pts = pts.reshape(-1, d)
        if self._shift is not None:
            pts = np.ascontiguousarray(pts + self._shift)
        n = len(pts)
        if n == 0:
            z = np.zeros(0)
            return PointFields(z, z.reshape(0, d), z.reshape(0, d * (d + 1) // 2), np.zeros(0, np.int32),
                               shape=(0,))
        if self._closed:
            raise RuntimeError("the Estimator is closed")
        with self._lock:
            try:
                self._proc.stdin.write(struct.pack("<Q", n))
                self._proc.stdin.write(memoryview(pts).cast("B"))
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                raise self._server_failure("is not accepting requests") from None
            fields = self._read_reply(n)
        if self.scale_factor != 1.0:
            s = np.sqrt(self.scale_factor)
            fields.velocity *= s
            fields.dispersion *= self.scale_factor
            if fields.velocity_gradient is not None:
                fields.velocity_gradient *= s
            if fields.stream_velocity is not None:
                fields.stream_velocity *= s
        if self._shift is not None and fields.stream_lagrangian is not None:
            fields.stream_lagrangian -= self._shift
        if single:
            fields.shape = ()
        return fields

    def _read_reply(self, n: int) -> PointFields:
        magic = self._read_exact(8)
        if magic == _ERROR:
            (length,) = struct.unpack("<Q", self._read_exact(8))
            raise ValueError("DTFE server rejected the request: "
                             + self._read_exact(length).decode(errors="replace"))
        if magic != _REPLY:
            raise self._server_failure(f"sent a corrupt reply header {magic!r}")
        n_back, R = struct.unpack("<QQ", self._read_exact(16))
        if n_back != n:
            raise self._server_failure(f"answered {n_back} points for a request of {n}")

        f = self.flags
        d, nd = self.dim, self.dim * (self.dim + 1) // 2     # vector and symmetric-tensor components
        den = self._read_array(n, "<f8")
        vel = self._read_array(d * n, "<f8").reshape(n, d)
        disp = self._read_array(nd * n, "<f8").reshape(n, nd)
        streams = self._read_array(n, "<i4")
        out = PointFields(den, vel, disp, streams, shape=(n,))
        if f & _F_CAUSTIC:
            out.caustic = self._read_array(n, "<i4")
        if f & _F_DEN_GRAD:
            out.density_gradient = self._read_array(d * n, "<f8").reshape(n, d)
        if f & _F_VEL_GRAD:
            out.velocity_gradient = self._read_array(d * d * n, "<f8").reshape(n, d, d)
        if f & _F_PER_STREAM:
            out.offsets = self._read_array(n + 1, "<u8")
            rec = self._read_array((1 + d) * R, "<f8").reshape(R, 1 + d)
            out.stream_density = rec[:, 0].copy()
            out.stream_velocity = rec[:, 1:].copy()
            if f & _F_IDS:
                out.stream_ids = self._read_array((d + 1) * R, "<u8").reshape(R, d + 1)
            if f & _F_DEN_GRAD:
                out.stream_density_gradient = self._read_array(d * R, "<f8").reshape(R, d)
        if f & _F_SCALAR:
            S = self.n_scalar
            sc = self._read_array(S * n, "<f8")
            out.scalar = sc if S == 1 else sc.reshape(n, S)
            if f & _F_PER_STREAM:
                rs = self._read_array(S * R, "<f8")
                out.stream_scalar = rs if S == 1 else rs.reshape(R, S)
        if f & _F_LAGRANGIAN:
            out.stream_lagrangian = self._read_array(d * R, "<f8").reshape(R, d)
        return out

    def density(self, points) -> np.ndarray:
        """rho/rho_bar at the points (summed over streams)."""
        return self(points).density

    def velocity(self, points) -> np.ndarray:
        """Density-weighted mean velocity at the points, (N, dim)."""
        return self(points).velocity

    def streams(self, points) -> np.ndarray:
        """Number of streams at the points (standard DTFE: 0/1 coverage)."""
        return self(points).streams

    def grid(self, xs, ys, zs=None) -> PointFields:
        """Fields on the tensor-product grid xs x ys x zs (1D coordinate arrays; a 2D server takes xs
        and ys only), returned with per-point arrays shaped (nx, ny[, nz][, k]) -- the analogue of
        CosmoDTFE's estimator((xs, ys, zs)). Ragged per-stream arrays stay flat (C order over the grid)."""
        axes = (xs, ys) if self.dim == 2 else (xs, ys, zs)
        if self.dim == 2 and zs is not None:
            raise ValueError("a 2D server takes grid(xs, ys)")
        if self.dim == 3 and zs is None:
            raise ValueError("a 3D server takes grid(xs, ys, zs)")
        axes = [np.asarray(a, dtype=np.float64).reshape(-1) for a in axes]
        mesh = np.meshgrid(*axes, indexing="ij")
        f = self(np.stack([m.ravel() for m in mesh], axis=1))
        shape = tuple(len(a) for a in axes)
        d = self.dim
        f.density = f.density.reshape(shape)
        f.velocity = f.velocity.reshape(shape + (d,))
        f.dispersion = f.dispersion.reshape(shape + (d * (d + 1) // 2,))
        f.streams = f.streams.reshape(shape)
        if f.caustic is not None:
            f.caustic = f.caustic.reshape(shape)
        if f.density_gradient is not None:
            f.density_gradient = f.density_gradient.reshape(shape + (d,))
        if f.velocity_gradient is not None:
            f.velocity_gradient = f.velocity_gradient.reshape(shape + (d, d))
        if f.scalar is not None:
            f.scalar = f.scalar.reshape(shape + f.scalar.shape[1:])
        f.shape = shape
        return f

    # ------------------------------------------------------------------ lifecycle
    def close(self):
        """Ends the server (idempotent). Called automatically by 'with' and on garbage collection."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        proc = self._proc
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write(struct.pack("<Q", 0))
                    proc.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        finally:
            for stream in (proc.stdout,):
                try:
                    stream.close()
                except OSError:
                    pass
            if self._log is not None:
                self._log.close()
            shutil.rmtree(self._workdir, ignore_errors=True)
            if self._owned_dir is not None:
                shutil.rmtree(self._owned_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"{self.n_vertices} vertices"
        if not self._closed and getattr(self, "n_partitions", 1) > 1:
            state += f", {self.n_partitions} partitions, {self.resident} resident"
        kind = "PS-DTFE" if self.phase_space else "DTFE"
        return f"<Estimator {kind} {self.snapshot.name} ({state})>"

    # ------------------------------------------------------------------ in-memory input
    @classmethod
    def from_arrays(cls, positions, *, velocities=None, masses=None, lagrangian=None, values=None,
                    ids=None, box=None, periodic: bool | None = None, phase_space: bool | None = None,
                    tmpdir=None, **kwargs) -> "Estimator":
        """An Estimator over particles held in numpy arrays -- CosmoDTFE's
        DensityEstimator(points, weights) / PhaseSpaceEstimator(source, warped, values) -- instead
        of a snapshot file. The arrays are written to a temporary Gadget-HDF5 file that the server
        reads (deleted again by close()), so every option of the constructor applies.

        positions   (N, 3) Eulerian positions, in Mpc (any consistent length unit: the box and
                    the query points use the same one; MpcUnit is 1). (N, 2) builds a 2D Estimator
                    (the DIM=2 binaries), with every array below 2-component too
        velocities  (N, 3), returned as given (no scale-factor conversion); default zeros
        masses      scalar or (N,); default 1 per particle. Densities come out as rho/rho_bar,
                    so only relative masses matter
        lagrangian  (N, 3) Lagrangian (initial) positions -> the phase-space estimator
        values      (N,) any per-particle quantity, interpolated per stream like the velocity
                    (the Estimator's 'scalar' / 'stream_scalar' outputs)
        ids         (N,) uint64 particle IDs (default 1..N; the per_stream_ids output)
        box         L (the periodic cube [0, L]^3), or ((xlo, ylo, zlo), (xhi, yhi, zhi)); default:
                    the particles' bounding box, non-periodic
        periodic    default: True when a box is given
        phase_space default: True when lagrangian is given
        tmpdir      where the temporary file goes (default: the system temp dir; must be local)
        **kwargs    passed to Estimator (per_stream, velocity_gradient, partition, ...)
        """
        try:
            import h5py
        except ImportError as e:        # pragma: no cover - h5py is a dtfelib dependency
            raise ImportError("Estimator.from_arrays needs h5py to write the temporary snapshot") from e
        x = np.asarray(positions, dtype=np.float64)
        dim = x.shape[-1] if x.ndim == 2 else 3
        if dim not in (2, 3):
            raise ValueError(f"positions must be (N, 3) or (N, 2), not {x.shape}")
        x = np.ascontiguousarray(x).reshape(-1, dim)
        n = len(x)
        if n < 5:
            raise ValueError("a tessellation needs at least 5 particles")
        if not np.all(np.isfinite(x)):
            raise ValueError("positions contain NaN or inf")
        if phase_space is None:
            phase_space = lagrangian is not None
        if phase_space and lagrangian is None:
            raise ValueError("the phase-space estimator needs the Lagrangian positions (lagrangian=...)")

        def per_particle(a, k, name):
            a = np.asarray(a, dtype=np.float64)
            a = a.reshape(-1, k) if k > 1 else a.reshape(-1)
            if len(a) != n:
                raise ValueError(f"{name} holds {len(a)} entries for {n} particles")
            return a

        v = np.zeros((n, dim)) if velocities is None else per_particle(velocities, dim, "velocities")
        q = None if lagrangian is None else per_particle(lagrangian, dim, "lagrangian")
        if box is None:
            pts = x if q is None else np.vstack([x, q])
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            pad = 1e-3 * np.maximum(hi - lo, 1e-12)
            lo, hi = lo - pad, hi + pad
            if periodic is None:
                periodic = False
        elif np.ndim(box) == 0:
            lo, hi = np.zeros(dim), np.full(dim, float(box))
        else:
            lo, hi = (np.asarray(b, dtype=np.float64).reshape(dim) for b in box)
        if periodic is None:
            periodic = True
        if np.any(hi <= lo):
            raise ValueError("the box must have a positive extent along every axis")
        # the binary reads '--box -3 ...' as an option: move a box with negative corners to 0
        shift = np.where(lo < 0, -lo, 0.0)
        if np.any(shift != 0):
            x, lo, hi = x + shift, lo + shift, hi + shift
            q = None if q is None else q + shift
        else:
            shift = None

        work = Path(tempfile.mkdtemp(prefix="dtfe-arrays-", dir=tmpdir))
        snap = work / "arrays.hdf5"
        mass_table = np.zeros(6)
        with h5py.File(snap, "w") as f:
            head = f.create_group("Header")
            npart = np.zeros(6, np.int32)
            npart[1] = n
            head.attrs["NumPart_ThisFile"] = npart
            head.attrs["NumPart_Total"] = npart.astype(np.uint32)
            head.attrs["NumFilesPerSnapshot"] = np.int32(1)
            head.attrs["BoxSize"] = float(np.max(hi - lo))
            head.attrs["Time"] = 1.0              # velocities are returned exactly as given
            head.attrs["Redshift"] = 0.0
            g = f.create_group("PartType1")
            g.create_dataset("Coordinates", data=x)
            g.create_dataset("Velocities", data=v)
            g.create_dataset("ParticleIDs", data=(np.arange(1, n + 1, dtype=np.uint64) if ids is None
                                                  else np.asarray(ids, dtype=np.uint64).reshape(-1)))
            if q is not None:
                g.create_dataset("InitialCoordinates", data=q)
            if masses is None or np.ndim(masses) == 0:
                mass_table[1] = 1.0 if masses is None else float(masses)
            else:
                m = per_particle(masses, 1, "masses")
                if np.all(m == m[0]):
                    mass_table[1] = float(m[0])
                else:
                    g.create_dataset("Masses", data=m)
            if mass_table[1] < 0:
                raise ValueError("masses must be positive")
            head.attrs["MassTable"] = mass_table
            if values is not None:
                g.create_dataset("Values", data=per_particle(values, 1, "values"))
                kwargs.setdefault("scalar_dataset", "Values")

        options = list(kwargs.pop("options", ()))
        options += ["--box"] + [repr(float(c)) for pair in zip(lo, hi) for c in pair]
        kwargs.setdefault("peculiar", False)
        try:
            return cls(snap, phase_space=phase_space, periodic=periodic, mpc_unit=1.0,
                       options=options, dim=dim, _owned_dir=work, _shift=shift, **kwargs)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
