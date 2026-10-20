"""The Pipeline tab's job, moved out of runspec.py: PipelineSpec (scripts/run_ps_pipeline.sh -- the grids and a
full-box image plane for every stale snapshot of the chosen simulations, then optionally plot_pointeval.py per
simulation) and the plane-window parser only it uses. runspec re-exports both and is the module to import; this one
reads runspec's own names through it at call time (_rs.NAME), so whatever a test patches there reaches it too."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import runspec as _rs
from dtfelib.cli import DATA_ROOT, FIGURES_ROOT, sim_dir
from dtfelib.io import plane_geometry_matches
from runspec import LAMBDA_TH_DEFAULT, POINTEVAL_DERIVATIVE_SMOOTH, PS_DEFAULT_FIELDS   # the dataclass defaults


def _window_values(text: str) -> list[float] | None:
    """[u0, u1, v0, v1] (Mpc) from a 'u0 u1 v0 v1' window; None when blank; ValueError when malformed."""
    parts = text.replace(",", " ").split()
    if not parts:
        return None
    if len(parts) != 4:
        raise ValueError("the window needs four numbers: u0 u1 v0 v1 (Mpc; blank = the whole box)")
    vals = [float(v) for v in parts]
    if min(vals) < 0 or not (vals[1] > vals[0] and vals[3] > vals[2]):
        raise ValueError("the window needs u1 > u0 and v1 > v0, all >= 0 (Mpc)")
    return vals


@dataclass
class PipelineSpec:
    """scripts/run_ps_pipeline.sh: grids + a full-box image plane for every STALE snapshot of the
    chosen simulations (resumable; snapshots are discovered, not chosen), then optionally the
    point-evaluated figures. The run options travel to run_ps_dtfe.sh through the environment, as
    the Grids job's do (the script's own defaults are the production ones: GPU, vertex masses,
    volume-weighted moments); the plane's geometry reaches tools/make_image_plane.py the same way."""
    data_root: str = str(DATA_ROOT)
    sims: list[str] = field(default_factory=list)
    nu: int = 8192                  # image-plane pixels across the box
    axis: str = "z"
    center: str = ""                # Mpc along the axis; "" = box centre
    planes: int = 1                 # sampling planes across a slab (PLANES; 1 = one crisp cross-section)
    thickness: float = 2.0          # the slab's depth in Mpc, used when planes > 1 (THICKNESS)
    supersample: int = 1            # K: KxK sample points per pixel, averaged = pixel-area means (SUPERSAMPLE)
    window: str = ""                # "u0 u1 v0 v1" in Mpc: only this part of the plane (WINDOW); "" = the whole box
    vel_grad: bool = True
    grid: int = 512
    fields: list[str] = field(default_factory=lambda: list(PS_DEFAULT_FIELDS))
    deposit: str = "sampled"        # "sampled" (nsub^3 samples per cell) | "exact" (r3d clipping): PS_EXACT
    nsub: int = 3                   # AVG_SUBSAMPLES
    gpu: bool | None = None         # PS_GPU: the script's -m; None = the build's (gpu_build_default)
    vertex_mass: bool = True        # PS_VERTEX_MASS (needed for TNG IC inputs)
    volume_weighted: bool = True    # PS_VOLUME_WEIGHTED (the production convention)
    caustics: bool = True           # PS_CAUSTICS (on: run_ps_dtfe.sh's default, the production runs')
    caustic_cusps: bool = False     # PS_CAUSTIC_CUSPS
    parallel_triangulation: bool = False   # PS_PARALLEL_TRI
    lambda_th: float = LAMBDA_TH_DEFAULT   # LAMBDA_TH
    mem_budget_gb: float = 0.0      # DTFE_MEM_BUDGET_GB, see RunSpec
    tess_cache: str = ""            # TESS_CACHE (run_ps_pipeline.sh forwards it to run_ps_dtfe.sh)
    linear_deposit: bool = False    # PS_LINEAR_DEPOSIT, see RunSpec
    ordered_merge: bool = True      # ORDERED_MERGE (run_ps_pipeline.sh's run_ps_dtfe.sh inherits it), see RunSpec
    start_over: bool = False        # PS_RESUME=0 (inherited the same way), see RunSpec
    pts_conventions: str = "grid"   # PTS_CONVENTIONS: the image plane's estimators, see RunSpec
    output_prefix: str = "ps_output"
    precision: str = "single"       # "double" = the PS-DTFE-double binary (DTFE_PRECISION), see RunSpec
    partition: int = 0
    max_concurrent: int = 0
    scratch_dir: str = ""
    force: bool = False
    plan_only: bool = False         # DRY_RUN=1: print the plan, compute nothing
    render: bool = True             # then plot_pointeval.py for each simulation, with the render_* options
    render_fields: str = ""         # comma-separated POINTEVAL_FIELDS; "" = every field the run wrote
    render_project: str = "plane"   # "plane" = the central plane | "slab" = the mean over every plane (planes > 1)
    render_smooth: float = 0.0      # Gaussian smoothing of every map, in output pixels (0 = none)
    render_smooth_derivatives: float = POINTEVAL_DERIVATIVE_SMOOTH   # the gradient maps' smoothed companions (0 = none)
    render_fixed_range: bool = False   # the grid maps' fixed density range instead of a percentile stretch
    render_force: bool = False      # re-render figures newer than their data
    figures_root: str = str(FIGURES_ROOT)
    # adaptive plane size: per snapshot the largest of nu, nu/2, ... (>= ADAPTIVE_MIN_NU) that the
    # memory check found to fit -- {sim: {"NNN": nu}}, filled by the GUI's "Check memory"
    adaptive: bool = False
    plan: dict = field(default_factory=dict)
    plan_key: str = ""              # memory_key() the plan was checked for (settings changed -> stale)

    def __post_init__(self):
        if self.gpu is None:            # a fresh pipeline's GPU tick is the PS build's (a saved one is kept)
            self.gpu = _rs.gpu_build_default("ps", self.precision)

    def plane_json(self, sim: str, nu: int | None = None) -> Path:
        return sim_dir(sim, self.data_root) / f"hires_plane_{self.axis}{nu or self.nu}.json"

    def adaptive_sizes(self) -> list[int]:
        sizes, nu = [], self.nu
        while nu >= _rs.ADAPTIVE_MIN_NU:
            sizes.append(nu)
            nu //= 2
        return sizes or [self.nu]

    def planned_nu(self, sim: str, snap: int) -> int | None:
        v = self.plan.get(sim, {}).get(f"{snap:03d}")
        return int(v) if v else None

    # ------------------------------------------------------------------ the plane's geometry
    def window_values(self) -> list[float] | None:
        return _window_values(self.window)

    def plane_points(self, nu: int | None = None) -> int:
        """Sample points of the plane file the pipeline makes: planes x nu x nv x K^2, nv following the
        window's aspect as make_image_plane.py sizes it (square pixels). A malformed window reads as the
        whole box here; problems() reports it."""
        nu = nu or self.nu
        try:
            w = self.window_values()
        except ValueError:
            w = None
        nv = nu if w is None else max(2, round(nu * (w[3] - w[2]) / (w[1] - w[0])))
        return self.planes * nu * nv * self.supersample ** 2

    def plane_matches(self, sim: str, nu: int | None = None) -> bool | None:
        """Whether the plane file on disk has this job's geometry. False also for a missing or unreadable
        sidecar (make_image_plane.py --same-as says 'differs' then, and the script regenerates the plane,
        which makes every snapshot of that simulation stale); None when this job's own window or centre
        is malformed (problems() reports that)."""
        import json
        try:
            side = json.loads(self.plane_json(sim, nu).read_text())
        except (OSError, ValueError):
            return False
        try:
            w = self.window_values()
            center = float(self.center) if self.center.strip() else None
        except ValueError:
            return None
        return plane_geometry_matches(side, center=center, planes=self.planes, thickness=self.thickness,
                                      supersample=self.supersample, window=w)

    # ------------------------------------------------------------------ memory check
    def memory_key(self) -> str:
        return _rs._memory_key(self, ("sims", "nu", "force", "plan_only", "adaptive", "plan", "plan_key",
                                      "figures_root", "center", "thickness", "lambda_th", "start_over", "pts_conventions")
                               + _rs.PIPELINE_RENDER_KEYS)

    def check_step(self, sim: str, snaps: list[int], nu: int) -> _rs.Step:
        """run_ps_dtfe.sh in report mode with every setting run_ps_pipeline.sh would pass it (the run
        options through the environment, the GPU flag, this plane) -- a plane not generated yet, or one
        the script will regenerate, is sized by its point count."""
        argv = [str(_rs.PS_SCRIPT), "-s", sim, "-g", str(self.grid)] + (["-m"] if self.gpu else []) \
            + [str(n) for n in sorted(snaps)]
        env = {k: v for k, v in self.command()[1].items() if k not in _rs.PIPELINE_ONLY_KEYS}
        env["AUTO_TUNE_REPORT"] = "1"
        plane_bin = self.plane_json(sim, nu).with_suffix(".bin")
        if plane_bin.is_file() and self.plane_json(sim, nu).is_file() and self.plane_matches(sim, nu) is not False:
            env["SAMPLE_POINTS"] = str(plane_bin)
        else:
            env["DTFE_AUTO_PTS_N"] = str(self.plane_points(nu))
            env["DTFE_AUTO_PTS_VELGRAD"] = "1" if self.vel_grad else "0"
        return _rs.Step(f"memory check {sim} {nu}^2", argv, env)

    def stale(self, sim: str, nu: int | None = None) -> tuple[list[int], int]:
        """(snapshots the pipeline would compute, snapshots on disk) -- the script's own test:
        <prefix>.pts_den newer than the combined file and the plane and holding the plane's point count
        (+ .pts_velGrad present), and a plane file of this job's geometry (a differing one is regenerated:
        everything is stale then). The count: else a 4096^2 slice newer than an older 8192^2 plane
        would read as current, so the script would skip the compute and every render fail on the size mismatch."""
        import json
        sp = sim_dir(sim, self.data_root)
        plane_bin = self.plane_json(sim, nu).with_suffix(".bin")
        plane_ok = (plane_bin.is_file() and self.plane_json(sim, nu).is_file()
                    and self.plane_matches(sim, nu) is not False)
        want_pts = None
        if plane_ok:                        # planes x nv x nu x K^2 from the sidecar, as run_ps_pipeline.sh counts
            try:
                s = json.loads(self.plane_json(sim, nu).read_text())
                want_pts = int(s.get("planes", 1)) * int(s["nv"]) * int(s["nu"]) * int(s.get("supersample", 1)) ** 2
            except (OSError, ValueError, KeyError, TypeError):
                want_pts = _rs.plane_points(plane_bin)
        todo, total = [], 0
        for f in sorted(sp.glob("snapdir_*/combined_[0-9][0-9][0-9].hdf5")):
            n, total = int(f.stem.split("_")[1]), total + 1
            den = f.parent / f"{self.output_prefix}.pts_den"
            fresh = (not self.force and plane_ok
                     and den.is_file() and den.stat().st_mtime > max(f.stat().st_mtime, plane_bin.stat().st_mtime)
                     and den.stat().st_size // 8 == want_pts
                     and (not self.vel_grad or (f.parent / f"{self.output_prefix}.pts_velGrad").is_file()))
            if not fresh:
                todo.append(n)
        return todo, total

    # ------------------------------------------------------------------ checks
    def problems(self) -> list[tuple[str, str]]:
        out = _rs._root_problems(self.data_root) + _rs._run_option_problems(self)
        if not _rs._root_problems(self.data_root):
            if not self.sims:
                out.append(("error", "no simulations selected"))
            for sim in self.sims:
                if not sim_dir(sim, self.data_root).is_dir():
                    out.append(("error", f"{sim} not found under {self.data_root}"))
        if not 16 <= self.nu <= 65536:
            out.append(("error", f"image plane of {self.nu} px is outside 16..65536"))
        if self.axis not in ("x", "y", "z"):
            out.append(("error", f"axis must be x, y or z (got {self.axis!r})"))
        if self.center.strip():
            try:
                float(self.center)
            except ValueError:
                out.append(("error", f"plane centre must be a number in Mpc (got {self.center!r})"))
            else:
                out.append(("warning", "a centre differing from an existing plane regenerates it, "
                                       "which makes EVERY snapshot of that simulation stale"))
        if not 1 <= self.planes <= 256:
            out.append(("error", f"planes across the slab must be 1..256 (got {self.planes})"))
        elif self.planes > 1 and self.thickness <= 0:
            out.append(("error", "a slab of several planes needs a positive thickness (Mpc)"))
        elif 1 < self.planes < _rs.PLANE_GHOSTING_MIN:
            out.append(("warning", f"{self.planes} planes across the slab: a handful of planes ghosts every inclined "
                                   f"structure (one displaced copy per plane); use 1 for a crisp cross-section, "
                                   f">= {_rs.PLANE_GHOSTING_MIN} for a smooth projection"))
        if not 1 <= self.supersample <= 8:
            out.append(("error", f"sub-samples per pixel must be 1..8 (got {self.supersample})"))
        try:
            self.window_values()
        except ValueError as e:
            out.append(("error", str(e)))
        out += _rs._precision_problems("ps", self.precision)
        if self.gpu and not _rs.gpu_built("ps", self.precision):
            out.append(("warning", f"GPU requested, but the binary was built without GPU support ({_rs.GPU_BUILD_HINT}): "
                                   "the deposit will run on the CPU"))
        out += _rs._ps_option_problems(self, max((_rs.dm_particles(s) or 0 for s in self.sims), default=0) or None)
        out += _rs._pts_conventions_problems(self, True)
        if self.output_prefix.strip() == "" or re.search(r"[\s/]", self.output_prefix):
            out.append(("error", "output prefix must be a single word (no spaces or '/')"))
        if self.render:
            bad = [f for f in self.render_fields.replace(" ", "").split(",") if f and f not in _rs.POINTEVAL_FIELDS]
            if bad:
                out.append(("error", f"unknown point-evaluated field(s) {', '.join(bad)}; "
                                     f"choose from {', '.join(_rs.POINTEVAL_FIELDS)}"))
            if self.render_smooth < 0 or self.render_smooth_derivatives < 0:
                out.append(("error", "smoothing widths must be >= 0 pixels"))
        out += _rs._scratch_problems(self.scratch_dir)
        pts = self.plane_points()
        pts_gb = pts * 545e-9
        gb = _rs.grids_gb(self.grid, self.fields, self.precision) + pts_gb
        if not self.scratch_dir and gb > 40:
            shape = f"{self.nu}^2 image plane" + (f" ({pts / 1e6:.0f}M points)" if pts != self.nu ** 2 else "")
            out.append(("warning", f"~{_rs.grids_gb(self.grid, self.fields, self.precision):.0f} GB of grids + "
                                   f"~{pts_gb:.0f} GB for the {shape} will not fit next to the triangulation on "
                                   "64 GB: set a scratch directory (bit-identical results)"))
        if self.force:
            out.append(("warning", "FORCE recomputes every snapshot, including up-to-date ones"))
        if self.adaptive:
            unplanned = [sim for sim in self.sims if sim_dir(sim, self.data_root).is_dir() and not self.plan.get(sim)]
            if unplanned:
                out.append(("error", "adaptive plane size: run 'Check memory' first ("
                                     + ", ".join(unplanned) + " not planned yet)"))
            elif self.plan_key != self.memory_key():
                out.append(("error", "adaptive plane size: the settings changed since the memory check; "
                                     "run 'Check memory' again"))
        if not self.plan_only:
            out += _rs._busy_problems()
        return out

    # ------------------------------------------------------------------ commands
    def command(self) -> tuple[list[str], dict[str, str]]:
        env = {"DTFE_DATA_ROOT": str(self.data_root), "SIMS": " ".join(self.sims),
               "NU": str(self.nu), "AXIS": self.axis, "GRID_SIZE": str(self.grid),
               "PTS_VEL_GRAD": "1" if self.vel_grad else "0",
               "PLANES": str(self.planes), "SUPERSAMPLE": str(self.supersample),
               # the run options, always spelled out (as RunSpec does): a stray PS_* exported in the
               # login shell must not leak into a run
               "PS_EXACT": "1" if self.deposit == "exact" else "0",
               "AVG_SUBSAMPLES": str(self.nsub),
               "PS_GPU": "1" if self.gpu else "0",
               "PS_METAL": "1" if self.gpu else "0",      # -m only switches the GPU ON; pin the env knob too
               "PS_VERTEX_MASS": "1" if self.vertex_mass else "0",
               "PS_VOLUME_WEIGHTED": "1" if self.volume_weighted else "0",
               "PS_CAUSTICS": "1" if self.caustics else "0",
               "PS_CAUSTIC_CUSPS": "1" if self.caustic_cusps else "0",
               "PS_PARALLEL_TRI": "1" if self.parallel_triangulation else "0",
               "PS_LINEAR_DEPOSIT": "1" if self.linear_deposit else "0",
               "ORDERED_MERGE": "1" if self.ordered_merge else "0",
               "PS_RESUME": "0" if self.start_over else "1",
               "PTS_CONVENTIONS": self.pts_conventions}
        if self.planes > 1:
            env["THICKNESS"] = _rs._num(self.thickness)
        try:
            w = self.window_values()
        except ValueError:
            w = None
        if w:
            env["WINDOW"] = " ".join(_rs._num(v) for v in w)
        if self.center.strip():
            env["CENTER"] = self.center.strip()
        if self.lambda_th != LAMBDA_TH_DEFAULT:
            env["LAMBDA_TH"] = f"{self.lambda_th:g}"
        env.update(_rs._run_option_env(self))
        if self.fields != PS_DEFAULT_FIELDS:
            env["FIELDS"] = " ".join(self.fields)
        if self.output_prefix != "ps_output":
            env["OUTPUT_PREFIX"] = self.output_prefix
        if self.precision == "double":
            env["DTFE_PRECISION"] = "double"
        if self.partition > 0:
            env["PARTITION"] = f"{self.partition} {self.partition} {self.partition}"
        if self.max_concurrent > 0:
            env["MAX_CONCURRENT"] = str(self.max_concurrent)
        if self.scratch_dir:
            env["SCRATCH_DIR"] = str(Path(self.scratch_dir).expanduser())
        if self.force:
            env["FORCE"] = "1"
        if self.plan_only:
            env["DRY_RUN"] = "1"
        if _rs.PYTHON != _rs.PIPELINE_PY_DEFAULT:
            env["PY"] = _rs.PYTHON
        return [str(_rs.PIPELINE_SCRIPT)], env

    def _render_step(self, sim: str, nu: int, snaps: list[int] | None) -> _rs.Step:
        a = [_rs.PYTHON, str(_rs.PLOT_DIR / "plot_pointeval.py"), "--sims", sim, "--plane", str(self.plane_json(sim, nu))]
        if snaps:
            a += ["--snaps", ",".join(str(n) for n in sorted(snaps))]
        if self.output_prefix != "ps_output":
            a += ["--prefix", self.output_prefix]
        fields = ",".join(f for f in self.render_fields.replace(" ", "").split(",") if f)
        if fields:
            a += ["--fields", fields]
        if self.render_project == "slab" and self.planes > 1:     # one plane: the plane itself
            a += ["--project", "slab"]
        if self.render_smooth > 0:
            a += ["--smooth", f"{self.render_smooth:g}"]
        if self.render_smooth_derivatives != POINTEVAL_DERIVATIVE_SMOOTH:
            a += ["--smooth-derivatives", f"{self.render_smooth_derivatives:g}"]
        if self.render_fixed_range:
            a.append("--fixed-range")
        if self.render_force:
            a.append("--force")
        if Path(self.figures_root) != FIGURES_ROOT:
            a += ["--figures-root", str(self.figures_root)]
        return _rs.Step(f"plot_pointeval.py {sim}" + (f" {nu}^2" if snaps else ""), a, _rs._plot_env(self.data_root, sim))

    def adaptive_groups(self, sim: str) -> dict[int, list[int]]:
        """{nu: [snapshots]} from the memory-check plan (adaptive runs only)."""
        groups: dict[int, list[int]] = {}
        for snap, nu in sorted(self.plan.get(sim, {}).items()):
            groups.setdefault(int(nu), []).append(int(snap))
        return dict(sorted(groups.items(), reverse=True))

    def steps(self) -> list[_rs.Step]:
        if self.adaptive and self.plan_key == self.memory_key() and any(self.plan.get(sim) for sim in self.sims):
            # one pipeline call per (simulation, plane size), each on its own snapshots (SNAPS),
            # each rendering its own plane -- the adaptive plan from the memory check
            steps = []
            for sim in self.sims:
                for nu, snaps in self.adaptive_groups(sim).items():
                    argv, env = self.command()
                    env.update({"SIMS": sim, "NU": str(nu), "SNAPS": " ".join(str(n) for n in snaps)})
                    compute = _rs.Step(f"run_ps_pipeline.sh {sim} {nu}^2 ({len(snaps)} snapshots)", argv, env)
                    steps.append(compute)
                    if self.render and not self.plan_only:
                        steps.append(self._render_after(self._render_step(sim, nu, snaps), compute))
            return steps
        argv, env = self.command()
        compute = _rs.Step("run_ps_pipeline.sh " + " ".join(self.sims), argv, env)
        steps = [compute]
        if self.render and not self.plan_only:
            for sim in self.sims:
                steps.append(self._render_after(self._render_step(sim, self.nu, None), compute))
        return steps

    @staticmethod
    def _render_after(render: _rs.Step, compute: _rs.Step) -> _rs.Step:
        """A render step runs only when its compute step exited 0: run_ps_pipeline.sh exits 1 when it
        computed nothing (no simulation or snapshot found), and the render after it would show a green job regardless."""
        render.after = compute.label
        return render

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineSpec":
        return cls(**{k: v for k, v in _rs._current_figures_root(d).items() if k in cls.__dataclass_fields__})
