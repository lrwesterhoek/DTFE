"""The Custom tab's job, moved out of runspec.py: CustomSpec (the binary itself on any snapshot file: Gadget
HDF5/binary or text, units, box, periodicity and initial positions spelled out), the snapshot-file probes it uses
(dimension, and the HDF5 header's box size, redshift, particle count and datasets) and the first-run demo job.
runspec re-exports all of it and is the module to import; this one reads runspec's own names through it at call
time (_rs.NAME)."""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path

import runspec as _rs
from dtfelib.estimator import STANDARD_PADDING
from runspec import CUSTOM_OUT_DEFAULT, LAMBDA_TH_DEFAULT   # the dataclass defaults


@dataclass
class CustomSpec:
    """The binary on the user's own snapshot file: one call, every assumption the TNG scripts make
    (layout, kpc units, the IC file) spelled out as options. Writes '<output_dir>/<name>.*' and a
    run log beside it; the GUI keeps these settings next to the outputs ('<name>.gui.json') so
    the Explore tab can query the same snapshot later. dim = 2 runs the 2D set (DTFE-2d, PS-DTFE-2d:
    a snapshot whose coordinates have two columns), CPU only."""
    input_file: str = ""
    input_type: int = 105
    dim: int = 3                    # 2: the 2D programs (make DIM=2) on a 2D snapshot (snapshot_dim)
    mpc_unit: float = 1.0           # 1 Mpc in the file's length unit (1 = Mpc, 1000 = kpc)
    periodic: bool = True
    box: list[float] = field(default_factory=list)   # [] = the header's box; else xlo xhi ylo yhi [zlo zhi]
    estimator: str = "ps"
    lagrangian: str = "file"        # "file": InitialCoordinates in the snapshot; "separate": lagrangian_file
    lagrangian_file: str = ""
    grid: int = 128
    nsub: int = 3
    deposit: str = ""               # as RunSpec's: "" = the estimator's production one (standard DTFE: the exact average)
    fields: list[str] = field(default_factory=lambda: ["density_a", "velocity_a", "dispersion_a"])
    gpu: bool = False
    vertex_mass: bool = False       # needed when the initial positions are PERTURBED (e.g. TNG's z=127)
    volume_weighted: bool = True
    caustics: bool = False
    caustic_cusps: bool = False
    parallel_triangulation: bool = False   # --parallel-triangulation (opt-in, see RunSpec)
    precision: str = "single"       # "double" = the -double binary (float64 end to end, ~2x memory), see RunSpec
    alpha_shape: float = 3.0        # --ps-alpha-shape, non-periodic clouds only: emitted when not the default 3
    scalar_dataset: str = ""
    output_dir: str = str(CUSTOM_OUT_DEFAULT)
    output_name: str = ""           # "" = the input file's name
    partition: int = 0
    max_concurrent: int = 0
    scratch_dir: str = ""
    lambda_th: float = LAMBDA_TH_DEFAULT   # --lambda_th with a web field (the binary's own default is 0.0)
    mem_budget_gb: float = 0.0      # DTFE_MEM_BUDGET_GB in the binary's environment, see RunSpec
    tess_cache: str = ""            # --tessellation-cache <folder>; "" = none
    linear_deposit: bool = False    # --ps-linear-deposit (PS only; not with the volume-weighted velocities)
    ordered_merge: bool = True      # --ordered-merge (PS only; off: --completion-order-merge), see RunSpec
    sample_points: str = ""         # --sample-points <file>: the fields at these points too ('.pts_*'); "" = none
    pts_vel_grad: bool = True       # ... with the velocity gradient at each point (--pts-vel-grad)
    pts_conventions: str = "grid"   # ... PS-DTFE: their estimators, '--pts-conventions legacy' when not the grids'
                                    # (the binary's default), see RunSpec
    window: list[float] = field(default_factory=list)   # --ps-window x0 x1 y0 y1 [z0 z1] in Mpc: only these cells
    halo_release: float = 0.0       # --ps-halo-release D: tetrahedra denser than D x rho_bar deposited at their centroid
    stream_density: str = "dtfe"    # --ps-stream-density for the sample points: 'dtfe' (interpolated) | 'geometric'
    per_stream: bool = False        # --per-stream: the sample points' values stream by stream (ragged '.pts_*')
    per_stream_ids: bool = False    # --per-stream-ids: ... with each stream's tetrahedron particle IDs
    extra_args: str = ""            # more program options as on a command line, appended after the tab's own
    demo_n: int = 0                 # > 0: first generate the synthetic demo snapshot (crossed waves, n^3) ...
    demo_file: str = ""             # ... but only at THIS path (demo_spec sets it): the setting is persisted, and
                                    # without the path any later missing input would quietly become a demo

    def __post_init__(self):
        if not self.deposit:            # a standard-DTFE run starts at the exact cell average
            self.deposit = _rs.production_deposit(self.estimator)

    def name(self) -> str:
        return self.output_name.strip() or (Path(self.input_file).stem if self.input_file else "output")

    def output_root(self) -> Path:
        return Path(self.output_dir).expanduser() / self.name()

    @property
    def sim(self) -> str:           # the run panel's "choose a simulation" test
        return self.input_file

    @property
    def data_root(self) -> str:
        return str(Path(self.output_dir).expanduser())

    @property
    def snapshots(self) -> list[int]:
        return [0]

    def binary(self) -> Path:
        return _rs.REPO_ROOT / _rs.binary_name(self.estimator, self.precision, self.dim)

    def command(self) -> list[str]:
        return self._own_command() + _extra_options(self.extra_args)

    def _own_command(self) -> list[str]:
        """The command the tab's own controls make (command() appends the extra options)."""
        ps = self.estimator == "ps"
        three = self.dim != 2           # a 2D run: no GPU, no parallel insertion (both 3D only)
        out = self.output_root()
        a = [str(self.binary()), str(Path(self.input_file).expanduser()), str(out),
             "--grid", str(self.grid), "--input", str(self.input_type), "--MpcUnit", f"{self.mpc_unit:g}"]
        if self.periodic:
            a.append("--periodic")
        if len(self.box) == 2 * self.dim:
            a += ["--box"] + [f"{v:g}" for v in self.box]
        fields = list(self.fields) if ps else _rs.dtfe_fields(self.fields)
        if self.scalar_dataset and "scalar_a" not in fields:
            fields.append("scalar_a")
        a += ["--field", *fields]
        if any(f in _rs.WEB_FIELDS for f in fields):      # the scripts' threshold, not the binary's 0.0
            a += ["--lambda_th", f"{self.lambda_th:g}"]
        if self.scalar_dataset:
            a += ["--scalar-dataset", self.scalar_dataset]
        if self.tess_cache:
            a += ["--tessellation-cache", str(Path(self.tess_cache).expanduser())]
        if ps:
            if self.deposit == "exact":
                a.append("--ps-exact-deposit")
            else:
                a += ["--avg-subsamples", str(self.nsub)]
            if self.lagrangian == "separate" and self.lagrangian_file:
                a += ["--lagrangianInput", str(Path(self.lagrangian_file).expanduser())]
            for on, flag in ((self.gpu and three, "--ps-gpu"), (self.vertex_mass, "--ps-vertex-mass"),
                             (self.volume_weighted, "--ps-volume-weighted"), (self.caustics, "--ps-caustics"),
                             (self.caustic_cusps, "--ps-caustic-cusps"),
                             (self.parallel_triangulation and three, "--parallel-triangulation")):
                if on:
                    a.append(flag)
            if not self.periodic and self.alpha_shape != 3.0:
                a += ["--ps-alpha-shape", f"{self.alpha_shape:g}"]
            if self.linear_deposit:
                a.append("--ps-linear-deposit")
            # the merge order said either way (index order is the binary's default; the flag keeps it in
            # the run log, which 'Load settings' reads back)
            a.append("--ordered-merge" if self.ordered_merge else "--completion-order-merge")
            if self.halo_release > 0:
                a += ["--ps-halo-release", _rs._num(self.halo_release)]
            if len(self.window) == 2 * self.dim:
                a += ["--ps-window"] + [_rs._num(v) for v in self.window]
        else:
            if self.gpu and three:
                a.append("--gpu")
            if self.deposit == "exact":
                a.append("--exact-average")
            # the padding: the binary's own 5 mean spacings leave wrong values within a few spacings of
            # a box face (or of the --box region's faces); 15, run_dtfe.sh's and the standard server's value
            # (dtfelib.estimator.STANDARD_PADDING). PS-DTFE pads in Lagrangian space by its own rules: nothing added
            a += ["--padding", f"{STANDARD_PADDING:g}"]
        if self.partition > 0:
            a += ["--partition"] + [str(self.partition)] * self.dim
        if self.max_concurrent > 0:
            a += ["--max-concurrent", str(self.max_concurrent)]
        if self.scratch_dir:
            a += ["--scratch-dir", str(Path(self.scratch_dir).expanduser())]
        if self.sample_points:
            a += ["--sample-points", str(Path(self.sample_points).expanduser())]
            if self.pts_vel_grad:
                a.append("--pts-vel-grad")
            if ps and self.stream_density != "dtfe":
                a += ["--ps-stream-density", self.stream_density]
            if ps and self.per_stream:
                a.append("--per-stream")
            if ps and self.per_stream_ids:
                a.append("--per-stream-ids")
            if ps and self.pts_conventions == "legacy":     # 'grid' is the binary's default: no flag for it
                a += ["--pts-conventions", "legacy"]
        return a

    def memory_key(self) -> str:
        return _rs._memory_key(self, ("output_dir", "output_name", "demo_n", "demo_file", "lambda_th", "tess_cache",
                                      "pts_conventions"))

    def demo_pending(self) -> bool:
        """The demo snapshot is to be generated first: the demo's own input path, not on disk yet."""
        f = Path(self.input_file).expanduser() if self.input_file else None
        return (self.demo_n > 0 and bool(self.demo_file) and f is not None
                and f == Path(self.demo_file).expanduser() and not f.is_file())

    def check_step(self) -> _rs.Step:
        """The same binary call in report mode: its auto-tuner's split and memory prediction for this
        file, nothing computed or written (the report line parses as snapshot 0)."""
        return _rs.Step(f"memory check {Path(self.input_file).name or '?'}", self.command() + ["--auto-tune-report"],
                        self.run_env())

    def run_env(self) -> dict[str, str]:
        """The binary's environment: the memory budget (the cache is an argument here, not TESS_CACHE)."""
        return {k: v for k, v in _rs._run_option_env(self).items() if k != "TESS_CACHE"}

    def steps(self) -> list[_rs.Step]:
        out = []
        if self.demo_pending():
            demo = Path(self.input_file).expanduser()
            out.append(_rs.Step("create the demo folder", ["mkdir", "-p", str(demo.parent)], {}))
            out.append(_rs.Step(f"generate the demo snapshot ({self.demo_n}{'²' if self.dim == 2 else '³'} particles, "
                                "crossed waves)",
                                [_rs.PYTHON, str(_rs.DEMO_GENERATOR), "--out", str(demo), "--n", str(self.demo_n),
                                 "--box", "100", "--amplitude-factor", "1.5", "--crossed-waves"]
                                + (["--dim", "2"] if self.dim == 2 else []), {}))
        root = self.output_root()
        if not root.parent.is_dir() and str(root.parent) != str(Path(self.input_file).expanduser().parent):
            out.append(_rs.Step("create the output folder", ["mkdir", "-p", str(root.parent)], {}))
        out.append(_rs.Step(f"{self.binary().name} {Path(self.input_file).name or '?'}", self.command(), self.run_env(),
                            tee=str(root) + ".runlog"))
        return out

    def settings_sidecar(self) -> dict:
        """What the Explore tab needs to query this snapshot again (written as '<name>.gui.json').
        'box' stays in the file's units (what --box takes); 'box_mpc' is the box in Mpc, the unit of
        the grids and of the server's coordinates -- from --box, else the Gadget header's BoxSize."""
        d = self.to_dict()
        f = Path(self.input_file).expanduser()
        d["input_file"] = str(f)
        if self.lagrangian_file:
            d["lagrangian_file"] = str(Path(self.lagrangian_file).expanduser())
        if self.lagrangian != "separate":
            d["lagrangian_file"] = ""
        d["box_mpc"] = []
        if len(self.box) == 2 * self.dim and self.mpc_unit > 0:
            d["box_mpc"] = [v / self.mpc_unit for v in self.box]
        elif f.is_file() and self.input_type == 105 and self.mpc_unit > 0:
            b = hdf5_boxsize(f)
            if b:
                d["box_mpc"] = [0.0, b / self.mpc_unit] * self.dim
        # the header's redshift: Explore scales the u-unit velocities by sqrt(a) with it (without it a z > 0
        # snapshot's velocities show 1/sqrt(a) too large); None = no scaling
        d["redshift"] = hdf5_redshift(f) if f.is_file() and self.input_type == 105 else None
        return d

    def problems(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        ps = self.estimator == "ps"
        f = Path(self.input_file).expanduser() if self.input_file else None
        if f is None:
            return [("error", "no snapshot file chosen")]
        demo_pending = self.demo_pending()
        if not f.is_file() and not demo_pending:
            out.append(("error", f"snapshot file not found: {f}"))
        if self.input_type not in _rs.INPUT_TYPES:
            out.append(("error", f"unknown input type {self.input_type}"))
        d = self.dim
        if d not in _rs.DIMS:
            return out + [("error", f"unknown dimension {d} (2 or 3)")]
        found = snapshot_dim(f, self.input_type) if f.is_file() else None
        if found is not None and found != d:
            out.append(("error", f"this snapshot is {found}D (its coordinates have {found} columns): "
                                 f"set the dimensions to {found}D"))
        if ps and self.input_type != 105:
            out.append(("error", "the phase-space estimator needs a Gadget HDF5 snapshot (particle IDs and "
                                 "initial positions); use standard DTFE for this format"))
        if self.mpc_unit <= 0:
            out.append(("error", "the length unit must be positive (1 Mpc in the file's units)"))
        if self.alpha_shape < 0:
            out.append(("error", "the alpha shape takes a number of mean particle spacings >= 0 (0 keeps the whole hull)"))
        out += _rs._precision_problems(self.estimator, self.precision, d)
        if d == 2:
            if self.gpu:
                out.append(("info", "the 2D programs run on the CPU (the GPU kernels are 3D): the GPU setting is ignored"))
            if ps and self.parallel_triangulation:
                out.append(("info", "the parallel triangulation is 3D only: a 2D run inserts sequentially"))
        elif ps and self.parallel_triangulation and not _rs.tbb_built(self.precision):
            out.append(("warning", "parallel triangulation asked for, but the binary was built without TBB: ignored"))
        elif ps and self.parallel_triangulation:
            out.append(("warning", "parallel triangulation: faster on small sets, but two identical runs then differ "
                                   "at float rounding (the stream counts do not)"))
        n = hdf5_particle_count(f) if (f.is_file() and self.input_type == 105) else None
        out += _rs.gpu_advice(self.gpu, self.deposit, self.fields if ps else (_rs.dtfe_fields(self.fields) or _rs.DTFE_FIELDS), n,
                              self.estimator,
                              grid=self.grid, precision=self.precision, dim=d)
        if self.box:
            if len(self.box) != 2 * d or any(self.box[2 * i + 1] <= self.box[2 * i] for i in range(d)):
                names = "xlo xhi ylo yhi" + (" zlo zhi" if d == 3 else "")
                out.append(("error", f"the box needs {2 * d} numbers, {names}, each hi > lo"))
            elif min(self.box) < 0:
                out.append(("error", "box corners must be >= 0 (the binary's option parser reads a negative "
                                     "number as an option): shift the coordinates, or leave the box to the header"))
        if ps and f.is_file() and self.input_type == 105:
            has_ic = hdf5_has(f, "InitialCoordinates")
            if self.lagrangian == "file" and has_ic is False:
                out.append(("error", "this snapshot stores no 'InitialCoordinates': choose the initial-"
                                     "conditions file, or use standard DTFE"))
            if self.lagrangian == "separate":
                lf = Path(self.lagrangian_file).expanduser() if self.lagrangian_file else None
                if lf is None or not lf.is_file():
                    out.append(("error", "choose the initial-conditions file (the particles' Lagrangian positions)"))
            if self.scalar_dataset and hdf5_has(f, self.scalar_dataset) is False:
                out.append(("error", f"no dataset '{self.scalar_dataset}' in the snapshot's particle groups"))
        if self.scalar_dataset and self.input_type != 105:
            out.append(("error", "a per-particle scalar dataset needs Gadget HDF5 input"))
        if not 8 <= self.grid <= (65536 if d == 2 else 4096):
            out.append(("error", f"grid {self.grid}{'²' if d == 2 else '³'} is outside 8..{65536 if d == 2 else 4096}"))
        if ps and not self.fields:
            out.append(("error", "no fields selected"))
        if not ps and not _rs.dtfe_fields(self.fields):
            out.append(("error", "no field standard DTFE computes is ticked (the velocity dispersion is PS-DTFE's)"))
        name = self.name()
        if re.search(r"[\s/]", name):
            out.append(("error", "output name must be a single word (no spaces or '/')"))
        od = Path(self.output_dir).expanduser()
        if not self.output_dir.strip():
            out.append(("error", "choose an output folder"))
        elif "Mobile Documents" in str(od):
            out.append(("warning", "the output folder is in iCloud Drive: grids are large, and syncing them "
                                   "is slow, so a local folder is better"))
        if self.gpu and d == 3 and not _rs.gpu_built(self.estimator, self.precision):
            out.append(("warning", f"GPU requested, but the binary was built without GPU support ({_rs.GPU_BUILD_HINT}): "
                                   "it will use the CPU"))
        if not self.binary().is_file() and self.precision == "single":
            how = f"'make {self.binary().name}'" if d == 3 else f"'{_rs.build_hint('single', 2)}'"
            out.append(("error", f"{self.binary().name} is not built (Setup, or {how})"))
        out += _rs._scratch_problems(self.scratch_dir)
        out += _rs._run_option_problems(self)
        if self.sample_points:
            if not Path(self.sample_points).expanduser().is_file():
                out.append(("error", f"sample-points file not found: {self.sample_points}"))
            # (standard DTFE with an explicit Partition is allowed: the binary answers the points per partition,
            # each from the tetrahedra it owns)
        if self.window:
            if not ps:
                out.append(("info", "the window is a PS-DTFE option: this standard-DTFE run ignores it"))
            elif (len(self.window) != 2 * d or not all(math.isfinite(v) for v in self.window)
                  or any(self.window[2 * i + 1] <= self.window[2 * i] for i in range(d))):
                out.append(("error", f"the window needs {2 * d} numbers in Mpc, each upper > lower"))
            elif any(f in _rs.WEB_FIELDS for f in self.fields):
                out.append(("error", "the T-web / V-web classes need the whole periodic grid: not with a window"))
            elif self.linear_deposit:
                out.append(("error", "the window cannot be combined with the linear deposit (the binary refuses it)"))
        if ps:
            out += _rs._pts_conventions_problems(self, bool(self.sample_points))
        if (self.per_stream or self.per_stream_ids or self.stream_density != "dtfe") and not self.sample_points:
            out.append(("warning", "the per-stream outputs and the stream density apply to sample points: choose a "
                                   "points file (or they do nothing)"))
        if not ps and (self.per_stream or self.per_stream_ids or self.halo_release > 0):
            out.append(("info", "per-stream outputs and the halo release are PS-DTFE options: this standard-DTFE run "
                                "leaves them out"))
        if ps and self.linear_deposit and self.volume_weighted:
            out.append(("error", "the linear deposit weights by density inside each tetrahedron, the volume-weighted "
                                 "velocities by volume: untick one (the binary refuses both)"))
        if (self.output_root().parent / (self.name() + ".a_den")).exists():
            out.append(("info", f"'{name}' already has outputs in this folder: they will be overwritten"))
        if self.extra_args.strip():
            try:
                extra = shlex.split(self.extra_args)
            except ValueError as e:
                out.append(("error", f"the extra options do not parse ({e}): close the quotes"))
            else:
                twice = sorted({t for t in extra if t.startswith("--")}
                               & {t for t in self._own_command() if t.startswith("--")})
                if twice:
                    out.append(("error", f"the extra options repeat {', '.join(twice)}, which the tab sets already "
                                         "(the program takes each option once): change it on the tab"))
        out += _rs._busy_problems()
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CustomSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _extra_options(text: str) -> list[str]:
    """The extra-options line split as a shell splits it; [] when its quotes do not close (problems() says so)."""
    try:
        return shlex.split(text)
    except ValueError:
        return []


def snapshot_dim(path, input_type: int = 105) -> int | None:
    """2 or 3: the dimension of a snapshot file, None when the file cannot tell. Gadget HDF5: the
    columns of its particles' Coordinates (a 2D snapshot holds (N, 2) datasets, as
    tests/generate_ps_test_data.py --dim 2 writes); the text formats: the box on the second line
    holds 2 x dim numbers (io/text_io.cc); Gadget binary carries no dimension (None)."""
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    if input_type == 105:
        try:
            import h5py
            with h5py.File(p, "r") as h:
                for name in sorted(h.keys()):
                    c = h[name].get("Coordinates") if name.startswith("PartType") else None
                    if c is not None and len(c.shape) == 2 and c.shape[1] in (2, 3):
                        return int(c.shape[1])
        except Exception:
            return None
        return None
    if input_type in (111, 112):
        try:
            with open(p, errors="replace") as fh:
                fh.readline()
                n = len(fh.readline().replace(",", " ").split())
        except OSError:
            return None
        return n // 2 if n in (4, 6) else None
    return None


def hdf5_boxsize(path) -> float | None:
    """The Gadget header's BoxSize of an HDF5 snapshot (file units), or None."""
    try:
        import h5py
        with h5py.File(path, "r") as h:
            b = float(h["Header"].attrs["BoxSize"])
        return b if b > 0 else None
    except Exception:
        return None


def hdf5_redshift(path) -> float | None:
    """The Gadget header's Redshift of an HDF5 snapshot, or None (no file, no header, no such attribute)."""
    try:
        import h5py
        with h5py.File(path, "r") as h:
            hd = h["Header"].attrs
            return float(hd["Redshift"]) if "Redshift" in hd else None
    except Exception:
        return None


def hdf5_particle_count(path) -> int | None:
    """The particles of an HDF5 snapshot (the header's NumPart_Total, all types), or None."""
    try:
        import h5py
        with h5py.File(path, "r") as h:
            n = int(sum(int(x) for x in h["Header"].attrs["NumPart_Total"]))
        return n if n > 0 else None
    except Exception:
        return None


def hdf5_has(path, dataset: str) -> bool | None:
    """Whether any /PartTypeN group of an HDF5 snapshot holds 'dataset' (None: cannot tell)."""
    try:
        import h5py
        with h5py.File(path, "r") as h:
            groups = [k for k in h.keys() if k.startswith("PartType")]
            return any(dataset in h[g] for g in groups) if groups else False
    except Exception:
        return None


def demo_spec(output_dir=None, gpu: bool | None = None) -> CustomSpec:
    """The first-run demo: a synthetic 32^3 crossed-wave snapshot (streams 1/3/9/27) and a quick
    PS-DTFE run on it -- about half a minute, no download. gpu None: where the deposit pays for
    32^3 particles (the CPU, see recommended_gpu)."""
    od = Path(output_dir).expanduser() if output_dir else CUSTOM_OUT_DEFAULT / "demo"
    fields = ["density_a", "velocity_a", "dispersion_a", "divergence_a"]
    if gpu is None:
        gpu = _rs.recommended_gpu("sampled", fields, 32 ** 3)
    return CustomSpec(input_file=str(od / "crossed_waves_32.hdf5"), input_type=105, mpc_unit=1.0,
                      periodic=True, estimator="ps", lagrangian="file", grid=64, nsub=2,
                      fields=fields, gpu=gpu, vertex_mass=False, volume_weighted=True, caustics=True,
                      output_dir=str(od), output_name="demo", demo_n=32, demo_file=str(od / "crossed_waves_32.hdf5"))
