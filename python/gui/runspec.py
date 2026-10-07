"""Job specifications behind the DTFE / PS-DTFE GUI (app.py).

Three jobs, one per GUI tab, each holding what the terminal workflow sets for it plus the checks
for combinations the scripts reject, and the EXACT commands it stands for (``steps()``):

  RunSpec       one scripts/run_ps_dtfe.sh or run_dtfe.sh call: grids (+ a hi-res slice)
  PipelineSpec  scripts/run_ps_pipeline.sh (grids + image plane for every stale snapshot of
                the chosen simulations), optionally followed by plot_pointeval.py per simulation
  PlotSpec      the figure scripts in python/plot (and analyze.py), one step per script call,
                described declaratively by FIGURE_SETS
  DataSpec      scripts/download_snapshots.sh (snapshots, group catalogues, trees, ICs) and
                python/tools/merge_HDF5.py (the h-free combined files the other jobs read)
  CustomSpec    the binary itself on ANY snapshot file (Gadget HDF5/binary, text): the "your
                file" tab, for data that is not IllustrisTNG -- units, box, periodicity and the
                initial positions spelled out instead of assumed

QueuedJob freezes any of them for the GUI's job queue (kind + settings; commands at start).

The GUI never runs anything the terminal could not: it shows those commands and executes them.

Pure Python (no Qt), so it is testable headless: tests/py_gui_runspec_test.py.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from dtfelib.cli import DATA_ROOT, FIGURES_ROOT, find_sims, sim_dir  # noqa: E402  (after the path fix-up)
from dtfelib.io import plane_geometry_matches  # noqa: E402

PS_SCRIPT = REPO_ROOT / "scripts" / "run_ps_dtfe.sh"
DTFE_SCRIPT = REPO_ROOT / "scripts" / "run_dtfe.sh"
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_ps_pipeline.sh"
PLOT_DIR = REPO_ROOT / "python" / "plot"
ANALYZE = REPO_ROOT / "python" / "analyze.py"
# FIGURES_ROOT (dtfelib.cli, = config.LOCAL_FIGURES_ROOT): DTFE_FIGURES_ROOT, else the T7, else python/figures
LEGACY_FIGURES_ROOT = REPO_ROOT / "python" / "figures"     # the default until 2026-10-06, saved verbatim in specs


def _current_figures_root(d: dict) -> dict:
    """A saved spec's figures_root with the OLD default (python/figures, stored as a plain path because the
    field saved the default it showed) read as today's default: settings and queued jobs from before the
    move to the T7 would otherwise keep writing into the repo."""
    root = d.get("figures_root")
    if root and Path(root) == LEGACY_FIGURES_ROOT:
        d = dict(d, figures_root=str(FIGURES_ROOT))
    return d
# the interpreter for the figure scripts (numpy/scipy/matplotlib/h5py): $PY as in the
# pipeline script, else the python3 on PATH -- NOT the GUI's own venv interpreter
PYTHON = os.environ.get("PY") or shutil.which("python3") or sys.executable
PIPELINE_PY_DEFAULT = "/opt/homebrew/bin/python3"         # run_ps_pipeline.sh's own PY default

# (name passed to --field, label). run_ps_dtfe.sh takes any subset via FIELDS; its default is
# the first seven. run_dtfe.sh has a FIXED list (DTFE_FIELDS) -- the GUI shows it read-only.
PS_FIELDS = [
    ("density_a", "density"),
    ("velocity_a", "velocity"),
    ("gradient_a", "velocity gradient"),
    ("divergence_a", "divergence"),
    ("shear_a", "shear"),
    ("vorticity_a", "vorticity"),
    ("dispersion_a", "velocity dispersion"),
    ("tweb_a", "T-web"),
    ("vweb_a", "V-web"),
]
PS_DEFAULT_FIELDS = [n for n, _ in PS_FIELDS[:7]]
WEB_FIELDS = ("tweb_a", "vweb_a")        # the cosmic-web classes: the only consumers of lambda_th
LAMBDA_TH_DEFAULT = 0.3                  # run_ps_dtfe.sh / run_dtfe.sh LAMBDA_TH (the binary alone: 0.0, so a Custom
                                         # run classified differently from a scripted one before 2026-10-05)
DTFE_FIELDS = ["density_a", "velocity_a", "gradient_a", "divergence_a", "shear_a", "vorticity_a"]
DTFE_ALLOWED_FIELDS = DTFE_FIELDS + list(WEB_FIELDS)   # what the standard binary computes (no dispersion: PS only)


def dtfe_fields(fields) -> list[str]:
    """The ticked fields a standard-DTFE run computes, in the list's order (the velocity dispersion is a
    phase-space field: left out)."""
    return [f for f in DTFE_ALLOWED_FIELDS if f in fields]

# full-resolution accumulator bytes per grid cell, roughly (auto_tune.h has the exact model)
_FIELD_BYTES = {"density_a": 4, "velocity_a": 12, "gradient_a": 36, "divergence_a": 36,
                "shear_a": 36, "vorticity_a": 36, "dispersion_a": 24, "tweb_a": 72, "vweb_a": 52}


@dataclass
class RunSpec:
    data_root: str = str(DATA_ROOT)
    sim: str = ""
    snapshots: list[int] = field(default_factory=list)
    estimator: str = "ps"            # "ps" (run_ps_dtfe.sh) | "dtfe" (run_dtfe.sh)
    deposit: str = "sampled"         # PS: "sampled" (nSub^3 samples/cell) | "exact" (r3d clipping);
                                     # standard DTFE: "sampled" (Monte-Carlo cell means) | "exact" (--exact-average)
    grid: int = 512
    nsub: int = 3
    fields: list[str] = field(default_factory=lambda: list(PS_DEFAULT_FIELDS))
    gpu: bool = True
    vertex_mass: bool = True         # --ps-vertex-mass (script default on; needed for TNG IC inputs)
    volume_weighted: bool = True     # --ps-volume-weighted (the production convention)
    caustics: bool = False
    caustic_cusps: bool = False
    parallel_triangulation: bool = False   # --parallel-triangulation (opt-in: faster small runs, repeats differ at rounding)
    precision: str = "single"        # "single" (./PS-DTFE, ./DTFE) | "double" (the -double pair: float64 end to end, ~2x memory)
    slice_plane: str = ""            # a --sample-points file (hi-res slice); "" = none
    slice_vel_grad: bool = True
    partition: int = 0               # 0 = auto-tuned
    max_concurrent: int = 0          # 0 = auto-tuned (NOT the binary's "0 = all cores")
    scratch_dir: str = ""
    output_prefix: str = "ps_output"
    lambda_th: float = LAMBDA_TH_DEFAULT   # T-web / V-web eigenvalue threshold (LAMBDA_TH; the scripts' default)
    mem_budget_gb: float = 0.0       # DTFE_MEM_BUDGET_GB: the auto-tuner's memory budget; 0 = its own (free RAM)
    tess_cache: str = ""             # TESS_CACHE: reuse the tessellation across runs (a local folder); "" = none
    linear_deposit: bool = False     # PS_LINEAR_DEPOSIT: --ps-linear-deposit (density-weighted inside each tet)

    # ------------------------------------------------------------------ data
    def sim_path(self) -> Path:
        return sim_dir(self.sim, self.data_root)

    # ------------------------------------------------------------------ checks
    def problems(self) -> list[tuple[str, str]]:
        """(level, message) pairs, level 'error' (blocks Run) or 'warning'."""
        out: list[tuple[str, str]] = []
        root = Path(self.data_root)
        if not root.is_dir():
            out.append(("error", f"data root not found: {root} (is the T7 plugged in?)"))
        elif not self.sim:
            out.append(("error", "no simulation selected"))
        elif not self.sim_path().is_dir():
            out.append(("error", f"{self.sim} not found under {root}"))
        if not self.snapshots:
            out.append(("error", "no snapshots selected"))
        if self.grid < 8 or self.grid > 4096:
            out.append(("error", f"grid {self.grid}^3 is outside 8..4096"))
        out += _scratch_problems(self.scratch_dir)
        out += _run_option_problems(self)
        out += _precision_problems(self.estimator, self.precision)
        if self.estimator == "dtfe":
            if not dtfe_fields(self.fields):
                out.append(("error", "no field standard DTFE computes is ticked (the velocity dispersion is PS-DTFE's)"))
            elif "dispersion_a" in self.fields:
                out.append(("info", "the velocity dispersion is a phase-space field: standard DTFE leaves it out"))
        if self.gpu and not gpu_built(self.estimator, self.precision):
            out.append(("warning", f"GPU requested, but the binary was built without GPU support ({GPU_BUILD_HINT}): "
                                   "the deposit will run on the CPU"))

        if self.estimator == "ps":
            out += _ps_option_problems(self, dm_particles(self.sim))
            if self.slice_plane and not Path(self.slice_plane).is_file():
                out.append(("error", f"slice file not found: {self.slice_plane}"))
            if self.output_prefix.strip() == "" or re.search(r"[\s/]", self.output_prefix):
                out.append(("error", "output prefix must be a single word (no spaces or '/')"))
        else:
            if self.slice_plane and not Path(self.slice_plane).is_file():
                out.append(("error", f"slice file not found: {self.slice_plane}"))
            if self.slice_plane and self.partition > 0:
                out.append(("error", "a hi-res slice with standard DTFE evaluates ONE triangulation: set Partition to "
                                     "auto (the binary refuses an explicit split)"))
            out += gpu_advice(self.gpu, self.deposit, dtfe_fields(self.fields) or DTFE_FIELDS, dm_particles(self.sim),
                              "dtfe", grid=self.grid, precision=self.precision)

        out += _busy_problems()

        # memory: the full-resolution grids are irreducible (partitions share them)
        gb = self.grid_gb()
        slice_pts = plane_points(self.slice_plane) if self.slice_plane else 0
        if not self.scratch_dir and gb + slice_pts * 545e-9 > 40:
            out.append(("warning", f"~{gb:.0f} GB of full-resolution grids"
                                   + (f" + ~{slice_pts * 545e-9:.0f} GB for {slice_pts/1e6:.0f}M slice points"
                                      if slice_pts else "")
                                   + " will not fit next to the triangulation on 64 GB: set a scratch "
                                     "directory (bit-identical results, grids on disk)"))
        return out

    def grid_gb(self) -> float:
        return grids_gb(self.grid, self.fields if self.estimator == "ps" else dtfe_fields(self.fields), self.precision)

    def runnable(self) -> bool:
        return not any(level == "error" for level, _ in self.problems())

    # ------------------------------------------------------------------ command
    def command(self) -> tuple[list[str], dict[str, str]]:
        """argv of the run script and the environment variables it adds."""
        env = {"DTFE_DATA_ROOT": str(self.data_root)}
        if self.precision == "double":
            env["DTFE_PRECISION"] = "double"
        if self.partition > 0:
            env["PARTITION"] = f"{self.partition} {self.partition} {self.partition}"
        if self.max_concurrent > 0:
            env["MAX_CONCURRENT"] = str(self.max_concurrent)
        if self.scratch_dir:                # both scripts (run_dtfe.sh got its hook 2026-10-05: before, a
            env["SCRATCH_DIR"] = str(Path(self.scratch_dir).expanduser())   # standard run silently kept the grids in RAM)
        if self.lambda_th != LAMBDA_TH_DEFAULT:
            env["LAMBDA_TH"] = f"{self.lambda_th:g}"
        env.update(_run_option_env(self))
        snaps = [str(n) for n in sorted(self.snapshots)]
        if self.estimator == "dtfe":
            argv = [str(DTFE_SCRIPT), "-s", self.sim, "-g", str(self.grid)]
            if dtfe_fields(self.fields) != DTFE_FIELDS:     # the ticks (T-web / V-web, a lighter set); else the script's
                env["FIELDS"] = " ".join(dtfe_fields(self.fields))
            if self.slice_plane:                            # the hi-res slice: the standard interpolant at the points
                env["SAMPLE_POINTS"] = str(self.slice_plane)
                env["PTS_VEL_GRAD"] = "1" if self.slice_vel_grad else "0"
            env["DTFE_METAL"] = "1" if self.gpu else "0"     # -m only switches the GPU ON; pin the env knob too
            if self.gpu:
                argv.append("-m")
            if self.deposit == "exact":
                argv.append("-e")
            return argv + snaps, env
        argv = [str(PS_SCRIPT), "-s", self.sim, "-g", str(self.grid), "-n", str(self.nsub)]
        if self.gpu:
            argv.append("-m")
        if self.deposit == "exact":
            argv.append("-e")
        env.update({
            "FIELDS": " ".join(self.fields),
            "PS_METAL": "1" if self.gpu else "0",
            "PS_VERTEX_MASS": "1" if self.vertex_mass else "0",
            "PS_VOLUME_WEIGHTED": "1" if self.volume_weighted else "0",
            "PS_CAUSTICS": "1" if self.caustics else "0",
            "PS_CAUSTIC_CUSPS": "1" if self.caustic_cusps else "0",
            "PS_PARALLEL_TRI": "1" if self.parallel_triangulation else "0",
            "PS_LINEAR_DEPOSIT": "1" if self.linear_deposit else "0",
            "OUTPUT_PREFIX": self.output_prefix,
        })
        if self.slice_plane:
            env["SAMPLE_POINTS"] = str(self.slice_plane)
            env["PTS_VEL_GRAD"] = "1" if self.slice_vel_grad else "0"
        return argv + snaps, env

    def shell_line(self) -> str:
        """The equivalent terminal command (repo-relative script, quoted as needed)."""
        return shell(*self.command())

    def memory_key(self) -> str:
        return _memory_key(self, ("snapshots", "lambda_th", "tess_cache"))

    def check_step(self) -> "Step":
        """The same run_ps_dtfe.sh call in report mode: the binary's own memory prediction per
        snapshot, nothing computed or written. PS-DTFE only (the report covers its tuner)."""
        argv, env = self.command()
        env["AUTO_TUNE_REPORT"] = "1"
        return Step(f"memory check {self.sim}", argv, env)

    def steps(self) -> list["Step"]:
        argv, env = self.command()
        return [Step(f"{Path(argv[0]).name} {' '.join(argv[len(argv) - len(self.snapshots):])}", argv, env)]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunSpec":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ---------------------------------------------------------------------- steps and quoting
@dataclass
class Step:
    """One command of a job: what the GUI runs, and shows, for it. tee: also copy the output to
    this file (the run log), exactly as '2>&1 | tee <file>' would."""
    label: str
    argv: list[str]
    env: dict[str, str]
    tee: str = ""
    after: str = ""         # the label of a step that must have exited 0 first (a merge after its download)

    def line(self) -> str:
        return shell(self.argv, self.env) + (f" 2>&1 | tee {_q(self.tee)}" if self.tee else "")


def _q(s: str) -> str:
    return s if re.fullmatch(r"[A-Za-z0-9_./:=+,-]+", s) else "'" + s.replace("'", "'\\''") + "'"


def _display_path(a: str) -> str:
    """Repo files repo-relative, the PATH python3 as 'python3'; everything else verbatim."""
    if a == PYTHON and shutil.which("python3") == a:
        return "python3"
    try:
        if not Path(a).is_absolute():
            return a
        rel = str(Path(a).relative_to(REPO_ROOT))
        return rel if "/" in rel else "./" + rel      # a repo-root binary must not be looked up on PATH
    except ValueError:
        return a


def shell(argv: list[str], env: dict[str, str]) -> str:
    """The terminal command for argv + added environment, runnable from the repository root."""
    return " ".join([f"{k}={_q(v)}" for k, v in env.items()]
                    + [_q(_display_path(argv[0]))] + [_q(_display_path(a)) for a in argv[1:]])


def script_text(steps: list[Step]) -> str:
    return "\n".join(st.line() for st in steps)


def export_script(steps: list[Step], title: str, slurm: dict | None = None) -> str:
    """The job as a runnable bash script (run from the repository root, as the GUI does), or as a
    SLURM batch job for a cluster (e.g. Hábrók): slurm = {job_name, time, cpus, mem, partition,
    mail} (empty values are left out). Like the GUI, the script goes on after a failed step."""
    lines = ["#!/usr/bin/env bash"]
    if slurm is not None:
        opts = [("job-name", slurm.get("job_name") or "dtfe"), ("time", slurm.get("time")),
                ("cpus-per-task", slurm.get("cpus")), ("mem", slurm.get("mem")),
                ("partition", slurm.get("partition")), ("output", "%x-%j.log")]
        if slurm.get("mail"):
            opts += [("mail-type", "END,FAIL"), ("mail-user", slurm["mail"])]
        lines += [f"#SBATCH --{k}={v}" for k, v in opts if v not in (None, "", 0)]
    lines += [f"# {title}",
              f"# exported by the DTFE launcher on {time.strftime('%Y-%m-%d %H:%M')}: {len(steps)} "
              f"command{'s' if len(steps) != 1 else ''}, run in order from the repository root",
              "# (on another machine: fix REPO and the data paths first)",
              f"REPO={_q(str(REPO_ROOT))}",
              'cd "$REPO" || exit 1']
    if slurm is not None:
        lines.append('export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"   # the binaries use OpenMP')
    lines += ["set -o pipefail", "failed=0"]
    for st in steps:
        lines += [f"# {st.label}", st.line() + " || failed=$((failed + 1))"]
    lines += ['[ "$failed" -eq 0 ] || echo "$failed step(s) failed" >&2', 'exit "$failed"', ""]
    return "\n".join(lines)


def _scratch_problems(scratch_dir: str) -> list[tuple[str, str]]:
    if not scratch_dir:
        return []
    sd = Path(scratch_dir).expanduser()
    if not sd.is_dir():
        return [("error", f"scratch directory does not exist: {sd}")]
    if "Mobile Documents" in str(sd.resolve()):
        return [("error", "the scratch directory is on iCloud Drive; the binary refuses it "
                          "(use a local path, e.g. /private/tmp/dtfe-scratch)")]
    return []


def machine_ram_gb() -> float:
    """This machine's physical memory in GB (0 when the OS will not say)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def _run_option_problems(spec) -> list[tuple[str, str]]:
    """The shared run options: the tessellation cache folder (local, existing: the binary refuses iCloud and
    does not create it -- the scripts do, the Custom run does not) and the memory budget."""
    out = []
    tc = getattr(spec, "tess_cache", "")
    if tc:
        d = Path(tc).expanduser()
        if "Mobile Documents" in str(d.resolve()):
            out.append(("error", "the tessellation cache is on iCloud Drive; the binary refuses it (use a local "
                                 "folder, e.g. ~/Library/Caches/DTFE/tessellations)"))
        elif not d.is_dir() and isinstance(spec, CustomSpec):
            out.append(("error", f"tessellation cache folder does not exist: {d}"))
    budget = getattr(spec, "mem_budget_gb", 0.0)
    ram = machine_ram_gb()
    if budget and ram and budget > ram:
        out.append(("warning", f"memory budget {budget:g} GB is more than this machine's {ram:.0f} GB: the run "
                               "may swap"))
    return out


def _run_option_env(spec) -> dict[str, str]:
    """DTFE_MEM_BUDGET_GB (the auto-tuner's budget; read by the binary itself, through any script) and
    TESS_CACHE (the run scripts' --tessellation-cache hook), only when set."""
    env = {}
    if getattr(spec, "mem_budget_gb", 0.0) > 0:
        env["DTFE_MEM_BUDGET_GB"] = f"{spec.mem_budget_gb:g}"
    if getattr(spec, "tess_cache", ""):
        env["TESS_CACHE"] = str(Path(spec.tess_cache).expanduser())
    return env


def _root_problems(data_root: str) -> list[tuple[str, str]]:
    root = Path(data_root)
    return [] if root.is_dir() else [("error", f"data root not found: {root} (is the T7 plugged in?)")]


def _busy_problems() -> list[tuple[str, str]]:
    """Another run on the machine is a warning (two production runs do not fit in memory together); a resident
    query server is told with its memory (a 24 GB server beside a large run made the Mac swap once), nothing
    more -- the auto-tuner's budget already starts from the free memory."""
    procs = dtfe_processes()
    out = []
    if procs["runs"]:
        out.append(("warning", "already running: " + ", ".join(f"{name} (pid {pid}{_gb_text(gb)})"
                                                                for pid, name, gb in procs["runs"])
                               + "; two production runs do not fit in memory together"))
    if procs["servers"]:
        out.append(("info", "query server in memory: " + ", ".join(f"{name} (pid {pid}{_gb_text(gb)})"
                                                                     for pid, name, gb in procs["servers"])
                            + "; a large run has to fit beside it (stop the server first if it does not)"))
    return out


# ---------------------------------------------------------------------- memory check
# run_ps_dtfe.sh AUTO_TUNE_REPORT=1 prints, per snapshot, what the binary's own auto-tuner would
# decide and how much memory it predicts -- the same model and stream estimate a real run uses.
# The binary itself (a custom-snapshot run) prints the line without 'snap=': it is keyed 0.
_REPORT = re.compile(r"AUTO-TUNE-REPORT (?:snap=(\d+) )?(\w+=.*)$")


def parse_reports(text: str) -> dict[int, dict]:
    """{snapshot: {partition, mc, predicted_gb, budget_gb, ..., over_budget}} from report output
    (the script's lines carry 'snap=NNN'; the bare binary's line is snapshot 0)."""
    out = {}
    for line in text.splitlines():
        m = _REPORT.search(line)
        if not m:
            continue
        rec = {}
        for kv in m.group(2).split():
            k, _, v = kv.partition("=")
            try:
                rec[k] = int(v) if k in ("partition", "mc", "over_budget") else float(v)
            except ValueError:
                pass
        out[int(m.group(1) or 0)] = rec
    return out


def _memory_key(spec, ignore: tuple) -> str:
    """The settings that change a run's memory, as a string (check results are kept per key)."""
    import json
    return json.dumps({k: v for k, v in spec.to_dict().items() if k not in ignore}, sort_keys=True)


ADAPTIVE_MIN_NU = 1024          # the adaptive plane size never goes below this


# ---------------------------------------------------------------------- pipeline
POINTEVAL_DERIVATIVE_SMOOTH = 10.0   # == dtfelib.pointeval.DERIVATIVE_SMOOTH (not imported: it pulls in matplotlib)
POINTEVAL_FIELDS = ("density", "streams", "speed", "dispTrace", "dispMag", "velDiv", "velShear", "velVort",
                    "denGradMag")    # == the keys of dtfelib.pointeval.STYLES (plot_pointeval.py --fields)
PLANE_GHOSTING_MIN = 16              # make_image_plane.py: fewer planes across a slab ghost every inclined structure
# the options that only change the figures: they do not enter the memory key
PIPELINE_RENDER_KEYS = ("render", "render_fields", "render_project", "render_smooth", "render_smooth_derivatives",
                        "render_fixed_range", "render_force")
# what run_ps_pipeline.sh reads itself; everything else in command()'s environment is for run_ps_dtfe.sh
PIPELINE_ONLY_KEYS = ("SIMS", "SNAPS", "NU", "AXIS", "CENTER", "PLANES", "THICKNESS", "SUPERSAMPLE", "WINDOW",
                      "GRID_SIZE", "PS_GPU", "FORCE", "DRY_RUN", "PY")


def _ps_option_problems(spec, n_particles) -> list[tuple[str, str]]:
    """The phase-space run options' checks, shared by the Grids and Pipeline jobs (duck-typed on the
    option names both carry: fields, deposit, nsub, gpu, caustics, caustic_cusps, parallel_triangulation,
    vertex_mass, precision)."""
    out: list[tuple[str, str]] = []
    if not spec.fields:
        out.append(("error", "no fields selected"))
    if spec.caustic_cusps and not spec.caustics:
        out.append(("error", "caustic cusps need 'caustics' switched on"))
    if spec.caustic_cusps and spec.gpu:
        out.append(("warning", "caustic cusps are a CPU-deposit pass; the GPU run leaves them to the CPU"))
    if spec.deposit == "exact" and spec.nsub != 3:
        out.append(("warning", "the exact deposit ignores the sub-sample count"))
    if getattr(spec, "linear_deposit", False):
        if spec.volume_weighted:
            out.append(("error", "the linear deposit weights by density inside each tetrahedron, the volume-weighted "
                                 "velocities by volume: untick one (the binary refuses both)"))
        if spec.caustics and spec.gpu:
            out.append(("warning", "the linear deposit on the GPU keeps only the fold parity of the caustic classes "
                                   "(no collapse counts): run it on the CPU for the full classification"))
    out += gpu_advice(spec.gpu, spec.deposit, spec.fields, n_particles, "ps", precision=spec.precision)
    if spec.parallel_triangulation and not tbb_built(spec.precision):
        out.append(("warning", "parallel triangulation asked for, but the binary was built without TBB: ignored"))
    elif spec.parallel_triangulation:
        out.append(("warning", "parallel triangulation: faster on small sets, but two identical runs then "
                               "differ at float rounding (the stream counts do not)"))
    if not spec.vertex_mass:
        out.append(("warning", "without vertex masses, TNG runs (z=127 IC Lagrangian input) "
                               "suppress density contrast by 1 - D(127)/D(z)"))
    return out


def _num(v: float) -> str:
    """A float for a script argument that reads back as the same float ('%g' kept six digits, so the
    launcher compared other numbers than the sidecar held): whole numbers plain, others as repr."""
    v = float(v)
    return str(int(v)) if v == int(v) else repr(v)


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
    gpu: bool = True                # PS_GPU: the script's -m
    vertex_mass: bool = True        # PS_VERTEX_MASS (needed for TNG IC inputs)
    volume_weighted: bool = True    # PS_VOLUME_WEIGHTED (the production convention)
    caustics: bool = False          # PS_CAUSTICS
    caustic_cusps: bool = False     # PS_CAUSTIC_CUSPS
    parallel_triangulation: bool = False   # PS_PARALLEL_TRI
    lambda_th: float = LAMBDA_TH_DEFAULT   # LAMBDA_TH
    mem_budget_gb: float = 0.0      # DTFE_MEM_BUDGET_GB, see RunSpec
    tess_cache: str = ""            # TESS_CACHE (run_ps_pipeline.sh forwards it to run_ps_dtfe.sh)
    linear_deposit: bool = False    # PS_LINEAR_DEPOSIT, see RunSpec
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

    def plane_json(self, sim: str, nu: int | None = None) -> Path:
        return sim_dir(sim, self.data_root) / f"hires_plane_{self.axis}{nu or self.nu}.json"

    def adaptive_sizes(self) -> list[int]:
        sizes, nu = [], self.nu
        while nu >= ADAPTIVE_MIN_NU:
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
        return _memory_key(self, ("sims", "nu", "force", "plan_only", "adaptive", "plan", "plan_key",
                                  "figures_root", "center", "thickness", "lambda_th") + PIPELINE_RENDER_KEYS)

    def check_step(self, sim: str, snaps: list[int], nu: int) -> Step:
        """run_ps_dtfe.sh in report mode with every setting run_ps_pipeline.sh would pass it (the run
        options through the environment, the GPU flag, this plane) -- a plane not generated yet, or one
        the script will regenerate, is sized by its point count."""
        argv = [str(PS_SCRIPT), "-s", sim, "-g", str(self.grid)] + (["-m"] if self.gpu else []) \
            + [str(n) for n in sorted(snaps)]
        env = {k: v for k, v in self.command()[1].items() if k not in PIPELINE_ONLY_KEYS}
        env["AUTO_TUNE_REPORT"] = "1"
        plane_bin = self.plane_json(sim, nu).with_suffix(".bin")
        if plane_bin.is_file() and self.plane_json(sim, nu).is_file() and self.plane_matches(sim, nu) is not False:
            env["SAMPLE_POINTS"] = str(plane_bin)
        else:
            env["DTFE_AUTO_PTS_N"] = str(self.plane_points(nu))
            env["DTFE_AUTO_PTS_VELGRAD"] = "1" if self.vel_grad else "0"
        return Step(f"memory check {sim} {nu}^2", argv, env)

    def stale(self, sim: str, nu: int | None = None) -> tuple[list[int], int]:
        """(snapshots the pipeline would compute, snapshots on disk) -- the script's own test:
        <prefix>.pts_den newer than the combined file and the plane (+ .pts_velGrad present), and a
        plane file of this job's geometry (a differing one is regenerated: everything is stale then)."""
        sp = sim_dir(sim, self.data_root)
        plane_bin = self.plane_json(sim, nu).with_suffix(".bin")
        plane_ok = (plane_bin.is_file() and self.plane_json(sim, nu).is_file()
                    and self.plane_matches(sim, nu) is not False)
        todo, total = [], 0
        for f in sorted(sp.glob("snapdir_*/combined_[0-9][0-9][0-9].hdf5")):
            n, total = int(f.stem.split("_")[1]), total + 1
            den = f.parent / f"{self.output_prefix}.pts_den"
            fresh = (not self.force and plane_ok
                     and den.is_file() and den.stat().st_mtime > max(f.stat().st_mtime, plane_bin.stat().st_mtime)
                     and (not self.vel_grad or (f.parent / f"{self.output_prefix}.pts_velGrad").is_file()))
            if not fresh:
                todo.append(n)
        return todo, total

    # ------------------------------------------------------------------ checks
    def problems(self) -> list[tuple[str, str]]:
        out = _root_problems(self.data_root) + _run_option_problems(self)
        if not _root_problems(self.data_root):
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
        elif 1 < self.planes < PLANE_GHOSTING_MIN:
            out.append(("warning", f"{self.planes} planes across the slab: a handful of planes ghosts every inclined "
                                   f"structure (one displaced copy per plane); use 1 for a crisp cross-section, "
                                   f">= {PLANE_GHOSTING_MIN} for a smooth projection"))
        if not 1 <= self.supersample <= 8:
            out.append(("error", f"sub-samples per pixel must be 1..8 (got {self.supersample})"))
        try:
            self.window_values()
        except ValueError as e:
            out.append(("error", str(e)))
        out += _precision_problems("ps", self.precision)
        if self.gpu and not gpu_built("ps", self.precision):
            out.append(("warning", f"GPU requested, but the binary was built without GPU support ({GPU_BUILD_HINT}): "
                                   "the deposit will run on the CPU"))
        out += _ps_option_problems(self, max((dm_particles(s) or 0 for s in self.sims), default=0) or None)
        if self.output_prefix.strip() == "" or re.search(r"[\s/]", self.output_prefix):
            out.append(("error", "output prefix must be a single word (no spaces or '/')"))
        if self.render:
            bad = [f for f in self.render_fields.replace(" ", "").split(",") if f and f not in POINTEVAL_FIELDS]
            if bad:
                out.append(("error", f"unknown point-evaluated field(s) {', '.join(bad)}; "
                                     f"choose from {', '.join(POINTEVAL_FIELDS)}"))
            if self.render_smooth < 0 or self.render_smooth_derivatives < 0:
                out.append(("error", "smoothing widths must be >= 0 pixels"))
        out += _scratch_problems(self.scratch_dir)
        pts = self.plane_points()
        pts_gb = pts * 545e-9
        gb = grids_gb(self.grid, self.fields, self.precision) + pts_gb
        if not self.scratch_dir and gb > 40:
            shape = f"{self.nu}^2 image plane" + (f" ({pts / 1e6:.0f}M points)" if pts != self.nu ** 2 else "")
            out.append(("warning", f"~{grids_gb(self.grid, self.fields, self.precision):.0f} GB of grids + "
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
            out += _busy_problems()
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
               "PS_LINEAR_DEPOSIT": "1" if self.linear_deposit else "0"}
        if self.planes > 1:
            env["THICKNESS"] = _num(self.thickness)
        try:
            w = self.window_values()
        except ValueError:
            w = None
        if w:
            env["WINDOW"] = " ".join(_num(v) for v in w)
        if self.center.strip():
            env["CENTER"] = self.center.strip()
        if self.lambda_th != LAMBDA_TH_DEFAULT:
            env["LAMBDA_TH"] = f"{self.lambda_th:g}"
        env.update(_run_option_env(self))
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
        if PYTHON != PIPELINE_PY_DEFAULT:
            env["PY"] = PYTHON
        return [str(PIPELINE_SCRIPT)], env

    def _render_step(self, sim: str, nu: int, snaps: list[int] | None) -> Step:
        a = [PYTHON, str(PLOT_DIR / "plot_pointeval.py"), "--sims", sim, "--plane", str(self.plane_json(sim, nu))]
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
        return Step(f"plot_pointeval.py {sim}" + (f" {nu}^2" if snaps else ""), a, _plot_env(self.data_root, sim))

    def adaptive_groups(self, sim: str) -> dict[int, list[int]]:
        """{nu: [snapshots]} from the memory-check plan (adaptive runs only)."""
        groups: dict[int, list[int]] = {}
        for snap, nu in sorted(self.plan.get(sim, {}).items()):
            groups.setdefault(int(nu), []).append(int(snap))
        return dict(sorted(groups.items(), reverse=True))

    def steps(self) -> list[Step]:
        if self.adaptive and self.plan_key == self.memory_key() and any(self.plan.get(sim) for sim in self.sims):
            # one pipeline call per (simulation, plane size), each on its own snapshots (SNAPS),
            # each rendering its own plane -- the adaptive plan from the memory check
            steps = []
            for sim in self.sims:
                for nu, snaps in self.adaptive_groups(sim).items():
                    argv, env = self.command()
                    env.update({"SIMS": sim, "NU": str(nu), "SNAPS": " ".join(str(n) for n in snaps)})
                    steps.append(Step(f"run_ps_pipeline.sh {sim} {nu}^2 ({len(snaps)} snapshots)", argv, env))
                    if self.render and not self.plan_only:
                        steps.append(self._render_step(sim, nu, snaps))
            return steps
        argv, env = self.command()
        steps = [Step("run_ps_pipeline.sh " + " ".join(self.sims), argv, env)]
        if self.render and not self.plan_only:
            for sim in self.sims:
                steps.append(self._render_step(sim, self.nu, None))
        return steps

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineSpec":
        return cls(**{k: v for k, v in _current_figures_root(d).items() if k in cls.__dataclass_fields__})


def grids_gb(grid: int, fields: list[str], precision: str = "single", dim: int = 3) -> float:
    b = 8 + sum(_FIELD_BYTES.get(f, 12) for f in fields)   # + stream counts/weights
    if precision == "double":
        b *= 2                                              # Real = double: every grid twice the bytes
    return grid ** dim * b / 1e9


PRECISIONS = ("single", "double")
PRECISION_TEXT = {"single": "single precision (the default)",
                  "double": "double precision: positions and fields in float64 from the input read onward, "
                            "about twice the memory"}


DIMS = (3, 2)
DIM_TEXT = {3: "3D", 2: "2D: a snapshot in a plane (two-column coordinates; the 2D programs, CPU only)"}


def binary_name(estimator: str, precision: str = "single", dim: int = 3) -> str:
    """The binary a job runs: PS-DTFE / DTFE, then '-2d' for the 2D set ('make ... DIM=2') and
    '-double' for the double-precision pair -- the Makefile's names (PS-DTFE-2d-double)."""
    return (("PS-DTFE" if estimator == "ps" else "DTFE") + ("-2d" if dim == 2 else "")
            + ("-double" if precision == "double" else ""))


def binary_built(estimator: str, precision: str = "single", dim: int = 3) -> bool:
    return (REPO_ROOT / binary_name(estimator, precision, dim)).is_file()


def build_hint(precision: str = "single", dim: int = 3) -> str:
    """The make call that builds a pair (the targets are DTFE and PS-DTFE whatever the set)."""
    args = ["make", "DTFE", "PS-DTFE"] + (["DIM=2"] if dim == 2 else []) + (["DOUBLE=1"] if precision == "double" else [])
    return " ".join(args)


def _precision_problems(estimator: str, precision: str, dim: int = 3) -> list[tuple[str, str]]:
    if precision not in PRECISIONS:
        return [("error", f"unknown precision '{precision}' (single or double)")]
    if precision == "double" and not binary_built(estimator, "double", dim):
        gpu = "" if dim == 2 else ", with METAL=1 on Apple Silicon, CUDA=1 or HIP=1 on Linux"
        return [("error", f"the double-precision program {binary_name(estimator, 'double', dim)} is not built: "
                          + ("Setup builds the pair (or " if dim == 3 else "build it with ")
                          + f"'{build_hint('double', dim)}'{gpu}" + (")" if dim == 3 else ""))]
    return []


def _plot_env(data_root: str, sim: str) -> dict[str, str]:
    # DTFE_SIM keeps python/config.py (box size, figure names) on the plotted simulation;
    # MPLBACKEND=Agg because several scripts end in plt.show(), which would block a GUI run
    return {"DTFE_DATA_ROOT": str(data_root), "DTFE_SIM": sim, "MPLBACKEND": "Agg"}


# ---------------------------------------------------------------------- figures
@dataclass(frozen=True)
class Opt:
    """One option of a figure set. kind: bool | int | float | choice | text | plane | multi (a checklist of
    'choices', the value a comma-separated string of the ticked ones). limits: a float's (min, max, decimals)
    when the default box (0-1000, 2 decimals) cannot hold its values."""
    key: str
    label: str
    kind: str
    default: object
    choices: tuple = ()
    help: str = ""
    limits: tuple = ()


# analyze.py's core plot scripts (its PLOT_SCRIPTS: kept equal by the runspec test) + the synthesis figure,
# which analyze.py runs only when named in --only
THESIS_SCRIPTS = ("plot_DTFE.py", "plot_velDiv_den.py", "plot_eigenvalues.py", "plot_shear_triaxial.py",
                  "plot_PDF_CDF.py", "plot_contour_filter.py", "plot_shape_filter.py", "plot_ellipses_contour_3D.py",
                  "plot_marked_correlation_BBKS.py", "plot_smoothing_comparison.py", "plot_phi_delta.py",
                  "plot_tidal_correlation.py", "plot_conclusions_synthesis.py")
POINTEVAL_DERIV_SMOOTH = 10.0   # dtfelib.pointeval.DERIVATIVE_SMOOTH (plot_pointeval's own default): the runspec test
DENSITY_LIMITS = (0.0, 1e7, 4)  # rho/rho_bar: TNG100-3-Dark z=0 reaches 2.3e5 (99.99th pct 913); voids go below 0.01


@dataclass(frozen=True)
class FigureSet:
    key: str
    title: str
    script: str             # path under python/
    per: str                # "snapshot" (one call each) | "series" (one call, all) | "sim"
    about: str
    output: str             # where under FIGURES_ROOT the figures land
    common: bool = True     # takes dtfelib's --method/--prefix/--smooth
    opts: tuple = ()


_REF = Opt("ref", "Reference snapshot", "int", 99, help="The snapshot the tracking starts from")
_RAW = Opt("raw", "Unaveraged fields", "bool", False,
           help="The raw fields instead of the volume-averaged ones")
_PLANE = Opt("plane", "Image plane", "plane", "",
             help="the plane sidecar (.json) the .pts_* files were evaluated on")
# the void scripts' smoothing in Mpc (2026-10-07): the same physical scale in every box, the minima window following
_SMOOTH_MPC = Opt("smooth_mpc", "Void smoothing in Mpc (0 = the default 10 cells)", "float", 0.0,
                  help="e.g. 2.1625, TNG100's 10 cells: TNG50, TNG100 and TNG300 then find the same voids; "
                       "the outputs get their own names", limits=(0.0, 100.0, 4))
_RADIUS = Opt("radius", "Void radius", "choice", "r_v", ("r_v", "r_eff"),
              help="r_v: the measured radius, where the smoothed density around the void rises back to the mean; "
                   "r_eff: the ellipsoid fit's curvature length, about 1.75 r_v")
_OVERLAPS = Opt("keep_overlaps", "Keep overlapping voids", "bool", False,
                help="also the voids whose centre lies inside the region of a deeper void")

FIGURE_SETS: tuple[FigureSet, ...] = (
    FigureSet("fields", "Field slice maps", "plot/plot_PS_DTFE.py", "snapshot",
              "Slices through every field grid of a snapshot: density, streams, velocity, "
              "dispersion and web eigenvalues, plus the point-evaluated maps when a hi-res "
              "slice exists.", "fields/<sim>/snapNNN",
              opts=(Opt("pointeval_only", "Only the point-evaluated maps", "bool",
                        False, help="Skips every grid load"), _RAW)),
    FigureSet("web", "Cosmic-web classification", "plot/plot_cosmic_web.py", "snapshot",
              "T-web and V-web classes and eigenvalue maps. Needs the T-web / V-web fields.",
              "cosmic_web", opts=(_RAW,)),
    FigureSet("series", "Standard-DTFE field series", "plot/plot_DTFE.py", "series",
              "Density, velocity, divergence and shear slices across the selected snapshots "
              "(standard DTFE unless the estimator says otherwise).", "dtfe"),
    FigureSet("pointeval", "Point-evaluated hi-res maps", "plot/plot_pointeval.py", "series",
              "Every point-evaluated field of the hi-res slices (density, streams, speed, "
              "dispersion and, with the velocity gradient, divergence, shear and vorticity) as a "
              "zero-thickness cross-section. Skips figures newer than their data.",
              "fields/<sim>/snapNNN/<plane>", common=False,
              opts=(_PLANE,
                    Opt("fields", "Fields", "text", "", help="comma-separated subset; blank = all available"),
                    Opt("project", "Projection", "choice", "plane", ("plane", "slab"),
                        help="'slab' averages every plane in the file"),
                    Opt("smooth_px", "Smoothing (output pixels)", "float", 0.0),
                    Opt("smooth_deriv", "Gradient maps also smoothed at (output pixels)", "float", POINTEVAL_DERIV_SMOOTH,
                        help="the smoothed variant of the divergence, shear and vorticity maps (0 = none)"),
                    Opt("fixed_range", "Fixed density range of the grid maps", "bool", False),
                    Opt("force", "Re-render up-to-date figures", "bool", False))),
    FigureSet("panels", "Publication panels", "plot/plot_pointeval_panels.py", "snapshot",
              "Projected density, the inverted 'sheet' view and the pointwise stream count of a "
              "hi-res slice, as one publication-resolution figure per snapshot.",
              "pointeval_panels/<sim>", common=False,
              opts=(_PLANE,
                    Opt("panels", "Panels", "text", "density,sheet,streams"),
                    Opt("project", "Projection", "choice", "plane", ("plane", "mean")),
                    Opt("vmin", "Density scale minimum (0 = auto)", "float", 0.0,
                        help="rho/rho_bar; auto = the 0.1th percentile, which adapts to each snapshot's contrast",
                        limits=DENSITY_LIMITS),
                    Opt("vmax", "Density scale maximum (0 = auto)", "float", 0.0, help="rho/rho_bar; auto = the 99.99th percentile",
                        limits=DENSITY_LIMITS),
                    Opt("dpi", "Output dpi (0 = native: one image pixel per sample)", "int", 0),
                    Opt("image", "Pixel-exact PNGs, one per panel (no axes; for zooming and posters)", "bool", False))),
    FigureSet("voidpop", "Void population", "plot/plot_void_population.py", "sim",
              "Counts, sizes, shapes and depths of all voids at every snapshot with grids.",
              "void_population/<sim>",
              opts=(Opt("panel_selected", "Distribution panels at the selected snapshots", "bool", False,
                        help="default: config.PANEL_SNAPSHOTS"), _OVERLAPS, _SMOOTH_MPC)),
    FigureSet("voidtrack", "Void tracking", "plot/plot_void_tracking.py", "sim",
              "Size, BBKS shape and orientation of the deepest voids followed back in time "
              "through the merger trees.", "void_tracking/<sim>",
              opts=(_REF, Opt("n_voids", "Voids", "int", 5))),
    FigureSet("halotrack", "Halo tracking", "plot/plot_halo_tracking.py", "sim",
              "Mass, shape, orientation, mergers and field environment of subhalos along their "
              "main progenitor branch. Needs the merger trees (shapes also the group catalogues).",
              "halo_tracking/<sim>",
              opts=(_REF, Opt("top", "Most massive centrals", "int", 5),
                    Opt("subhalos", "Subhalo IDs", "text", "",
                        help="SubfindIDs at the reference snapshot, space-separated (replaces the top N)"),
                    Opt("environment", "Field environment", "bool", True),
                    Opt("min_ratio", "Merger ratio to mark", "float", 0.1))),
    FigureSet("voidprof", "Void profiles", "plot/plot_void_profiles.py", "series",
              "Stacked density, radial velocity and single-stream fraction around the voids, in units of each "
              "void's radius, per size bin, at the selected snapshots.", "void_profiles/<sim>",
              opts=(_RADIUS, Opt("sample", "Voids", "choice", "resolved", ("resolved", "deep"),
                                 help="deep: only the voids whose centre is below the deep-void threshold"),
                    _OVERLAPS, _SMOOTH_MPC)),
    FigureSet("webstreams", "Streams by web environment", "plot/plot_web_streams.py", "series",
              "Volume and mass fractions of every T-web and V-web class by stream multiplicity, per snapshot "
              "and across redshift.", "web_streams/<sim>",
              opts=(Opt("web", "Web", "choice", "both", ("both", "tweb", "vweb")),)),
    FigureSet("pk", "Power spectrum", "plot/plot_pk_compare.py", "series",
              "P(k) of the density grids against linear theory, with the cosmic-variance band, and divided by "
              "the same box's early spectrum grown linearly (its random fluctuations cancel there).",
              "power_spectrum/<sim>",
              opts=(Opt("nbins", "k bins (0 = one per fundamental mode)", "int", 40),
                    Opt("ref", "Same-box panel (from the first snapshot)", "bool", True))),
    FigureSet("skeleton", "Caustic skeleton", "plot/plot_caustic_skeleton.py", "series",
              "Walls, filaments and nodes from the collapse classes of a run with caustics on, and per void the "
              "fold fraction around its wall and the distance to the nearest wall.", "caustic_skeleton/<sim>",
              opts=(Opt("voids", "Per-void statistics", "bool", True), _RADIUS, _OVERLAPS, _SMOOTH_MPC)),
    FigureSet("thesis", "Thesis analysis set", "analyze.py", "series",
              "The core analysis figures (eigenvalues, void shapes, correlations, ...) through "
              "analyze.py: 'compute' fills python/cache from the standard-DTFE grids with the "
              "smoothing of python/config.py, 'plot' draws from that cache, 'export' writes the void "
              "catalogues (CSV + HDF5, under void_catalog/<sim>).", "<analysis>/",
              common=False,
              opts=(Opt("mode", "Mode", "choice", "all", ("all", "compute", "plot", "check", "export")),
                    Opt("only", "Only these scripts", "multi", "", THESIS_SCRIPTS,
                        help="tick some to run only those; none ticked = the full core set (the synthesis figure "
                             "is run only when ticked)"))),
)
FIGURE_SET = {fs.key: fs for fs in FIGURE_SETS}


@dataclass
class PlotSpec:
    """The figure scripts: one Step per script call (per snapshot for the per-snapshot sets)."""
    data_root: str = str(DATA_ROOT)
    sim: str = ""
    snapshots: list[int] = field(default_factory=list)
    sets: list[str] = field(default_factory=lambda: ["fields"])
    method: str = "auto"            # auto | ps | dtfe  (dtfelib --method)
    prefix: str = ""                # "" = the scripts' default grid prefix
    smooth: float = 0.0             # Gaussian smoothing in grid cells (dtfelib --smooth)
    options: dict = field(default_factory=dict)      # {set key: {option key: value}}
    figures_root: str = str(FIGURES_ROOT)            # only plot_pointeval can redirect its output

    def opt(self, set_key: str, key: str):
        default = next(o.default for o in FIGURE_SET[set_key].opts if o.key == key)
        return self.options.get(set_key, {}).get(key, default)

    def sim_path(self) -> Path:
        return sim_dir(self.sim, self.data_root)

    def _common(self) -> list[str]:
        a = []
        if self.method != "auto":
            a += ["--method", self.method]
        if self.prefix:
            a += ["--prefix", self.prefix]
        if self.smooth > 0:
            a += ["--smooth", f"{self.smooth:g}"]
        return a

    def _void_args(self, key) -> list[str]:
        """The void options a set has: the radius, overlapping voids, the smoothing in Mpc."""
        names = {o.key for o in FIGURE_SET[key].opts}
        a = []
        if "radius" in names and self.opt(key, "radius") != "r_v":
            a += ["--radius", str(self.opt(key, "radius"))]
        if "keep_overlaps" in names and self.opt(key, "keep_overlaps"):
            a.append("--keep-overlaps")
        if "smooth_mpc" in names and float(self.opt(key, "smooth_mpc")) > 0:
            a += ["--smooth-mpc", f"{float(self.opt(key, 'smooth_mpc')):g}"]
        return a

    def steps(self) -> list[Step]:
        out: list[Step] = []
        env = _plot_env(self.data_root, self.sim)
        snaps = sorted(self.snapshots)
        for key in [k for k, _ in ((fs.key, fs) for fs in FIGURE_SETS) if k in self.sets]:
            fs = FIGURE_SET[key]
            script = str(REPO_ROOT / "python" / fs.script)
            base = [PYTHON, script]
            name = Path(fs.script).name
            if key in ("fields", "web"):
                extra = (["--pointeval-only"] if key == "fields" and self.opt(key, "pointeval_only") else []) \
                    + (["--raw"] if self.opt(key, "raw") else [])
                for n in snaps:
                    out.append(Step(f"{name} snap {n:03d}",
                                    base + ["--sim", self.sim, "--snap", str(n)] + self._common() + extra, env))
            elif key == "series":
                out.append(Step(f"{name} {len(snaps)} snapshots",
                                base + ["--sim", self.sim, "--snaps", *map(str, snaps)] + self._common(), env))
            elif key == "pointeval":
                a = base + ["--sims", self.sim, "--snaps", ",".join(map(str, snaps))]
                if self.opt(key, "plane"):
                    a += ["--plane", str(self.opt(key, "plane"))]
                if self.prefix:
                    a += ["--prefix", self.prefix]
                if str(self.opt(key, "fields")).strip():
                    a += ["--fields", str(self.opt(key, "fields")).replace(" ", "")]
                if self.opt(key, "project") != "plane":
                    a += ["--project", str(self.opt(key, "project"))]
                if float(self.opt(key, "smooth_px")) > 0:
                    a += ["--smooth", f"{float(self.opt(key, 'smooth_px')):g}"]
                if float(self.opt(key, "smooth_deriv")) != POINTEVAL_DERIV_SMOOTH:
                    a += ["--smooth-derivatives", f"{float(self.opt(key, 'smooth_deriv')):g}"]
                if self.opt(key, "fixed_range"):
                    a.append("--fixed-range")
                if self.opt(key, "force"):
                    a.append("--force")
                if Path(self.figures_root) != FIGURES_ROOT:
                    a += ["--figures-root", str(self.figures_root)]
                out.append(Step(f"{name} {len(snaps)} snapshots", a, env))
            elif key == "panels":
                plane = str(self.opt(key, "plane"))
                for n in snaps:
                    sd = self.sim_path() / f"snapdir_{n:03d}"
                    png = Path(self.figures_root) / "pointeval_panels" / self.sim / \
                        f"snap{n:03d}_{Path(plane).stem or 'plane'}.png"
                    a = base + ["--prefix", str(sd / (self.prefix or "ps_output")), "--plane", plane,
                                "-o", str(png)]
                    if self.opt(key, "panels") != "density,sheet,streams":
                        a += ["--panels", str(self.opt(key, "panels")).replace(" ", "")]
                    if self.opt(key, "project") != "plane":
                        a += ["--project", str(self.opt(key, "project"))]
                    for opt_key, flag in (("vmin", "--vmin"), ("vmax", "--vmax")):
                        if float(self.opt(key, opt_key)) > 0:
                            a += [flag, f"{float(self.opt(key, opt_key)):g}"]
                    if int(self.opt(key, "dpi")) > 0:
                        a += ["--dpi", str(int(self.opt(key, "dpi")))]
                    if self.opt(key, "image"):
                        a.append("--image")
                    out.append(Step(f"{name} snap {n:03d}", a, env))
            elif key == "voidpop":
                a = base + ["--sim", self.sim] + self._common()
                if self.opt(key, "panel_selected") and snaps:
                    a += ["--panel-snaps", *map(str, snaps)]
                out.append(Step(name, a + self._void_args(key), env))
            elif key in ("voidprof", "webstreams", "pk", "skeleton"):
                a = base + ["--sim", self.sim, "--snaps", *map(str, snaps)] + self._common()
                if key == "voidprof" and self.opt(key, "sample") != "resolved":
                    a += ["--sample", str(self.opt(key, "sample"))]
                if key == "webstreams" and self.opt(key, "web") != "both":
                    a += ["--web", str(self.opt(key, "web"))]
                if key == "pk":
                    if int(self.opt(key, "nbins")) != 40:
                        a += ["--nbins", str(int(self.opt(key, "nbins")))]
                    if not self.opt(key, "ref"):
                        a += ["--ref-snap", "-1"]
                if key == "skeleton" and self.opt(key, "voids"):
                    a.append("--voids")
                out.append(Step(f"{name} {len(snaps)} snapshots", a + self._void_args(key), env))
            elif key == "voidtrack":
                out.append(Step(name, base + ["--sim", self.sim, "--snap", str(self.opt(key, "ref"))]
                                + self._common() + ["--n-voids", str(self.opt(key, "n_voids"))], env))
            elif key == "halotrack":
                a = base + ["--sim", self.sim, "--snap", str(self.opt(key, "ref"))] + self._common()
                ids = str(self.opt(key, "subhalos")).replace(",", " ").split()
                a += (["--subhalo", *ids] if ids else ["--top", str(self.opt(key, "top"))])
                if not self.opt(key, "environment"):
                    a.append("--no-environment")
                if float(self.opt(key, "min_ratio")) != 0.1:
                    a += ["--min-ratio", f"{float(self.opt(key, 'min_ratio')):g}"]
                out.append(Step(name, a, env))
            elif key == "thesis":
                mode = str(self.opt(key, "mode"))
                a = base + [mode]
                if mode in ("all", "compute", "export"):
                    a += [f"{n:03d}" for n in snaps]
                only = [s.strip() for s in str(self.opt(key, "only")).split(",") if s.strip()]
                if only and mode in ("all", "plot"):
                    a += ["--only", *only]
                out.append(Step(f"analyze.py {mode}", a, env))
        return out

    def problems(self) -> list[tuple[str, str]]:
        out = _root_problems(self.data_root)
        if not out:
            if not self.sim:
                out.append(("error", "no simulation selected"))
            elif not self.sim_path().is_dir():
                out.append(("error", f"{self.sim} not found under {self.data_root}"))
        if not self.sets:
            out.append(("error", "no figure sets selected"))
        if out:
            return out
        snaps = sorted(self.snapshots)
        need_snaps = [k for k in self.sets
                      if FIGURE_SET[k].per in ("snapshot", "series")
                      and not (k == "thesis" and self.opt(k, "mode") in ("plot", "check"))]
        if need_snaps and not snaps:
            out.append(("error", "no snapshots selected (needed by: "
                                 + ", ".join(FIGURE_SET[k].title for k in need_snaps) + ")"))
        sp = self.sim_path()
        grid_sets = [k for k in ("fields", "web", "series") if k in self.sets
                     and not (k == "fields" and self.opt(k, "pointeval_only"))]
        if grid_sets:
            missing = [n for n in snaps if not has_grids(sp / f"snapdir_{n:03d}", self.method, self.prefix)]
            if missing:
                out.append(("warning", f"no {'' if self.method == 'auto' else self.method + ' '}grids for "
                                       f"snapshot{'s' if len(missing) > 1 else ''} "
                                       + " ".join(f"{n:03d}" for n in missing)
                                       + ": those figures will fail"))
        if "series" in self.sets and self.method == "auto" and not any(
                has_grids(sp / f"snapdir_{n:03d}", "dtfe", "") for n in snaps):
            out.append(("warning", "the DTFE series plots standard-DTFE grids by default; "
                                   "set the estimator to PS-DTFE to plot those instead"))
        prefix = self.prefix or "ps_output"
        for k in ("pointeval", "panels"):
            if k not in self.sets:
                continue
            missing = [n for n in snaps if not (sp / f"snapdir_{n:03d}" / f"{prefix}.pts_den").is_file()]
            if missing:
                out.append(("warning", f"{FIGURE_SET[k].title}: no point-evaluated slice ({prefix}.pts_den) for "
                                       + " ".join(f"{n:03d}" for n in missing)
                                       + "; run the pipeline or a hi-res slice first"))
            plane = str(self.opt(k, "plane"))
            if plane and not Path(plane).is_file():
                out.append(("error", f"{FIGURE_SET[k].title}: plane file not found: {plane}"))
            elif not plane and k == "panels":
                out.append(("error", "Publication panels: choose the image plane"))
            elif not plane:
                sizes = [plane_points(p.with_suffix(".bin")) for p in plane_sidecars(sp)]
                if len(sizes) != len(set(sizes)):
                    out.append(("warning", "Point-evaluated maps: several planes have the same pixel count, "
                                           "so 'auto' may pick the wrong one: choose the plane"))
        if "halotrack" in self.sets or "voidtrack" in self.sets:
            if not (sp / "Merger Trees").is_dir() and not any(sp.glob("trees*")):
                out.append(("warning", f"no merger trees under {sp}; the tracking figures need them "
                                       "(Data tab: merger trees)"))
        if "thesis" in self.sets and self.opt("thesis", "mode") in ("all", "compute", "export"):
            missing = [n for n in snaps if not has_grids(sp / f"snapdir_{n:03d}", "dtfe", "")]
            if missing:
                out.append(("warning", "Thesis analysis set: no standard-DTFE grids for "
                                       + " ".join(f"{n:03d}" for n in missing)))
        if self.smooth < 0:
            out.append(("error", "smoothing must be >= 0"))
        mpc = [FIGURE_SET[k].title for k in self.sets if any(o.key == "smooth_mpc" for o in FIGURE_SET[k].opts)
               and float(self.opt(k, "smooth_mpc")) > 0]
        if mpc and self.smooth > 0:
            out.append(("error", "the smoothing is given in cells and in Mpc (" + ", ".join(mpc) + "): use one"))
        if "skeleton" in self.sets:
            cc = (self.prefix or "ps_output") + ".causticClass"
            missing = [n for n in snaps if not (sp / f"snapdir_{n:03d}" / cc).is_file()]
            if missing:
                out.append(("warning", f"Caustic skeleton: no {cc} for " + " ".join(f"{n:03d}" for n in missing)
                                       + " (it needs a run with caustics on)"))
        if self.prefix and re.search(r"[\s/]", self.prefix):
            out.append(("error", "grid prefix must be a single word (no spaces or '/')"))
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlotSpec":
        known = {k: v for k, v in _current_figures_root(d).items() if k in cls.__dataclass_fields__}
        known["sets"] = [k for k in known.get("sets", []) if k in FIGURE_SET]
        return cls(**known)


# ---------------------------------------------------------------------- download and merge
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "download_snapshots.sh"
MERGE_TOOL = REPO_ROOT / "python" / "tools" / "merge_HDF5.py"
API_KEY_FILE = Path.home() / ".tng_api_key"     # download_snapshots.sh reads it (or TNG_API_KEY)

# DM particles per side of every TNG run (the -Dark twin has the same DM count). A raw -Dark
# snapshot is ~61 bytes per particle (measured on the T7: TNG50-3/100-3/300-3, 4 chunks each),
# the merged combined_NNN.hdf5 32 (float32 x3 + uint64 + float32 x3).
TNG_DM_PER_SIDE = {"TNG50-1": 2160, "TNG50-2": 1080, "TNG50-3": 540, "TNG50-4": 270,
                   "TNG100-1": 1820, "TNG100-2": 910, "TNG100-3": 455,
                   "TNG300-1": 2500, "TNG300-2": 1250, "TNG300-3": 625}
TNG_SIMULATIONS = [f"{run}{suffix}" for run in TNG_DM_PER_SIDE for suffix in ("-Dark", "")]
RAW_BYTES_PER_PARTICLE, MERGED_BYTES_PER_PARTICLE = 61, 32


def _snapshot_ladder() -> dict[int, float]:
    """python/config.py's SNAPSHOT_TO_REDSHIFT, read as data (importing config has side effects)."""
    import ast
    try:
        tree = ast.parse((REPO_ROOT / "python" / "config.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SNAPSHOT_TO_REDSHIFT"
                                                   for t in node.targets):
                return {int(k): float(v) for k, v in ast.literal_eval(node.value).items()}
    except (OSError, SyntaxError, ValueError):
        pass
    return {}


SNAPSHOT_LADDER = _snapshot_ladder()


def dm_particles(sim: str) -> int | None:
    side = TNG_DM_PER_SIDE.get(sim.removesuffix("-Dark"))
    return side ** 3 if side else None


def api_key_available() -> bool:
    return bool(os.environ.get("TNG_API_KEY", "").strip()) or (API_KEY_FILE.is_file()
                                                              and os.access(API_KEY_FILE, os.R_OK))


def snapshot_state(sim_path, n: int) -> dict:
    """What is on disk for snapshot n: raw chunks and merged files, particles and group catalogue."""
    sd, gd = Path(sim_path) / f"snapdir_{n:03d}", Path(sim_path) / f"groups_{n:03d}"
    return {"chunks": len(list(sd.glob(f"snap_{n:03d}.*.hdf5"))),
            "merged": (sd / f"combined_{n:03d}.hdf5").is_file(),
            "gc_chunks": len(list(gd.glob(f"fof_subhalo_tab_{n:03d}.*.hdf5"))),
            "gc_merged": (gd / f"combined_fof_subhalo_tab_{n:03d}.hdf5").is_file()}


def sim_state(sim_path) -> dict:
    """Whole-simulation products: merger trees and initial conditions."""
    sp, td = Path(sim_path), Path(sim_path) / "Merger Trees"
    return {"tree_chunks": len(list(td.glob("tree_extended.*.hdf5"))),
            "tree_merged": (td / "combined_tree_extended.hdf5").is_file(),
            "ics_raw": (sp / "ics.hdf5").is_file() or (sp / "snap_ics.hdf5").is_file(),
            "ics_converted": (sp / "combined_ics.hdf5").is_file()}


def data_snapshots(sim_path) -> list[int]:
    """The redshift ladder plus every snapshot or group catalogue already on disk."""
    found = set(SNAPSHOT_LADDER)
    for d in list(Path(sim_path).glob("snapdir_*")) + list(Path(sim_path).glob("groups_*")):
        try:
            found.add(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(found)


@dataclass
class DataSpec:
    """Getting a simulation onto disk: download_snapshots.sh (particle snapshots, group
    catalogues, merger trees, initial conditions) and merge_HDF5.py (h-free combined files).
    Downloads run first, then merges, one step each. The API key is never on a command line."""
    data_root: str = str(DATA_ROOT)
    sim: str = ""
    snapshots: list[int] = field(default_factory=list)
    download_snapshots: bool = True
    download_groupcats: bool = False
    download_trees: bool = False
    download_ics: bool = False
    merge_snapshots: bool = True
    merge_groupcats: bool = False
    merge_trees: bool = False
    convert_ics: bool = False
    delete_chunks: bool = False

    def sim_path(self) -> Path:
        return sim_dir(self.sim, self.data_root)

    def _downloads(self) -> bool:
        return self.download_snapshots or self.download_groupcats or self.download_trees or self.download_ics

    def _todo(self) -> dict:
        """What is still missing on disk -- merged items are left out of every step: the download
        script would skip them anyway, and re-merging would rewrite combined_NNN.hdf5, whose new
        mtime makes every grid and slice built from it look stale to the pipeline."""
        sp = self.sim_path()
        states = {n: snapshot_state(sp, n) for n in sorted(set(self.snapshots))}
        whole = sim_state(sp)
        return {"snaps": [n for n, st in states.items() if not st["merged"]],
                "gcs": [n for n, st in states.items() if not st["gc_merged"]],
                "trees": not whole["tree_merged"], "ics": not whole["ics_converted"],
                "states": states, "whole": whole}

    def steps(self) -> list[Step]:
        todo = self._todo()
        snaps, gcs = [str(n) for n in todo["snaps"]], [str(n) for n in todo["gcs"]]
        # PY: the download script checks the chunks already on disk with h5py (an incomplete one is fetched
        # again) -- the same python the merge step runs with
        env = {"DTFE_DATA_ROOT": str(self.data_root), "PY": PYTHON}
        dl = [str(DOWNLOAD_SCRIPT)]
        out = []
        if self.download_snapshots and snaps:
            out.append(Step(f"download snapshots {' '.join(snaps)}", dl + ["-s", self.sim, *snaps], env))
        if self.download_groupcats and gcs:
            out.append(Step(f"download group catalogues {' '.join(gcs)}", dl + ["-c", "-s", self.sim, *gcs], env))
        if self.download_trees and todo["trees"]:
            out.append(Step("download merger trees", dl + ["-t", "-s", self.sim], env))
        if self.download_ics and todo["ics"]:
            out.append(Step("download initial conditions", dl + ["-i", "-s", self.sim], env))
        merge = [PYTHON, str(MERGE_TOOL), "-d", str(self.sim_path())]
        tail = ["--delete-chunks"] if self.delete_chunks else []
        # a merge runs only when its download step exited 0: half a download merged is a product with
        # missing particles (merge_HDF5.py refuses what it can detect; a missing TREE chunk it cannot)
        have = {st.label.split(" ")[1]: st.label for st in out}        # 'snapshots', 'group', 'merger', 'initial'
        if self.merge_snapshots and snaps:
            out.append(Step(f"merge snapshots {' '.join(snaps)}", merge + snaps + tail, {}, after=have.get("snapshots", "")))
        if self.merge_groupcats and gcs:
            out.append(Step(f"merge group catalogues {' '.join(gcs)}", merge + ["--groupcats", *gcs] + tail, {},
                            after=have.get("group", "")))
        if self.merge_trees and todo["trees"]:
            out.append(Step("merge merger trees", merge + ["--trees"] + tail, {}, after=have.get("merger", "")))
        if self.convert_ics and todo["ics"]:
            out.append(Step("convert initial conditions", merge + ["--ics"] + tail, {}, after=have.get("initial", "")))
        return out

    def stale_partials(self) -> list:
        """combined_*.hdf5.partial files a stopped merge left for the products about to be merged: the
        merge removes them first, so their bytes are free for it (and worth a word)."""
        todo = self._todo()
        sp = self.sim_path()
        cands = []
        if self.merge_snapshots:
            cands += [sp / f"snapdir_{n:03d}" / f"combined_{n:03d}.hdf5.partial" for n in todo["snaps"]]
        if self.merge_groupcats:
            cands += [sp / f"groups_{n:03d}" / f"combined_fof_subhalo_tab_{n:03d}.hdf5.partial" for n in todo["gcs"]]
        if self.merge_trees and todo["trees"]:
            cands.append(sp / "Merger Trees" / "combined_tree_extended.hdf5.partial")
        if self.convert_ics and todo["ics"]:
            cands.append(sp / "combined_ics.hdf5.partial")
        return [c for c in cands if c.is_file()]

    def download_bytes(self) -> tuple[int, int]:
        """(estimated raw download, estimated merged output) in bytes for the chosen snapshots,
        skipping what the script would skip (already merged) and chunks already on disk."""
        n_part = dm_particles(self.sim) or 0
        raw = merged = 0
        for n in set(self.snapshots):
            st = snapshot_state(self.sim_path(), n)
            if self.download_snapshots and not st["merged"] and not st["chunks"]:
                raw += n_part * RAW_BYTES_PER_PARTICLE
            if self.merge_snapshots and not st["merged"]:
                merged += n_part * MERGED_BYTES_PER_PARTICLE
        return raw, merged

    def problems(self) -> list[tuple[str, str]]:
        out = _root_problems(self.data_root)
        if out:
            return out
        if not self.sim:
            return [("error", "no simulation named")]
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", self.sim):
            return [("error", f"not a simulation name: {self.sim!r}")]
        per_snap = (self.download_snapshots or self.download_groupcats or self.merge_snapshots
                    or self.merge_groupcats)
        whole = self.download_trees or self.download_ics or self.merge_trees or self.convert_ics
        if not (per_snap or whole):
            return out + [("error", "nothing selected to download or merge")]
        if per_snap and not self.snapshots and not whole:
            return out + [("error", "no snapshots selected")]
        todo = self._todo()
        if not self.steps():
            return out + [("error", "nothing left to do: everything selected is already merged")]
        if self.sim not in TNG_SIMULATIONS:
            out.append(("warning", f"{self.sim} is not a TNG run this GUI knows; the TNG API may not serve it"))
        if self._downloads():
            if not api_key_available():
                out.append(("error", f"no TNG API key: put it in {API_KEY_FILE} (chmod 600) or set TNG_API_KEY"))
            if shutil.which("wget") is None:
                out.append(("error", "wget is not installed (brew install wget)"))
            if not self.sim.endswith("-Dark") and self.download_snapshots:
                out.append(("warning", "full-physics snapshots also hold gas and stars: several times larger "
                                       "downloads, and only the dark matter is merged"))
        skipped = []
        if self.download_snapshots or self.merge_snapshots:
            skipped += [f"snapshot {n:03d}" for n, st in todo["states"].items() if st["merged"]]
        if self.download_groupcats or self.merge_groupcats:
            skipped += [f"groups {n:03d}" for n, st in todo["states"].items() if st["gc_merged"]]
        if (self.download_trees or self.merge_trees) and not todo["trees"]:
            skipped.append("merger trees")
        if (self.download_ics or self.convert_ics) and not todo["ics"]:
            skipped.append("initial conditions")
        if skipped:
            out.append(("warning", "already merged, left out: " + ", ".join(skipped)
                                   + " (re-merging would make their grids look stale)"))
        if self.merge_snapshots and not self.download_snapshots:
            empty = [n for n in todo["snaps"] if not todo["states"][n]["chunks"]]
            if empty:
                out.append(("warning", "no chunks to merge for snapshot " + " ".join(f"{n:03d}" for n in empty)))
        if self.merge_groupcats and not self.download_groupcats:
            empty = [n for n in todo["gcs"] if not todo["states"][n]["gc_chunks"]]
            if empty:
                out.append(("warning", "no group-catalogue chunks to merge for " + " ".join(f"{n:03d}" for n in empty)))
        if self.delete_chunks:
            out.append(("warning", "deleting the raw chunks after the merge removes "
                                   "the only copy of the fields the merge drops (potential, Subfind densities)"))
        raw, merged = self.download_bytes()
        stale = self.stale_partials()
        if stale:
            gb = sum(p.stat().st_size for p in stale) / 1e9
            out.append(("warning", f"a stopped merge left {gb:.1f} GB in " + ", ".join(p.name for p in stale)
                                   + "; the merge removes it first"))
        try:
            free = shutil.disk_usage(self.data_root).free
            free += sum(p.stat().st_size for p in stale)
        except OSError:
            free = None
        if free is not None and raw + merged > free:
            out.append(("error", f"~{(raw + merged) / 1e9:.0f} GB needed, {free / 1e9:.0f} GB free on the data disk"))
        elif free is not None and raw + merged > 0.8 * free:
            out.append(("warning", f"~{(raw + merged) / 1e9:.0f} GB of {free / 1e9:.0f} GB free on the data disk"))
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DataSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------- your own snapshot file
CUSTOM_OUT_DEFAULT = Path.home() / "DTFE-output"
DEMO_GENERATOR = REPO_ROOT / "tests" / "generate_ps_test_data.py"
# --input types the binary reads (input_output.cc); only Gadget HDF5 carries the particle IDs and
# initial positions a phase-space run needs
INPUT_TYPES = {105: "Gadget HDF5 (.hdf5)", 101: "Gadget binary (type 1/2)",
               111: "Text: x y z mass per line", 112: "Text: x y z per line"}


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
    deposit: str = "sampled"
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
    sample_points: str = ""         # --sample-points <file>: the fields at these points too ('.pts_*'); "" = none
    pts_vel_grad: bool = True       # ... with the velocity gradient at each point (--pts-vel-grad)
    window: list[float] = field(default_factory=list)   # --ps-window x0 x1 y0 y1 [z0 z1] in Mpc: only these cells
    halo_release: float = 0.0       # --ps-halo-release D: tetrahedra denser than D x rho_bar deposited at their centroid
    stream_density: str = "dtfe"    # --ps-stream-density for the sample points: 'dtfe' (interpolated) | 'geometric'
    per_stream: bool = False        # --per-stream: the sample points' values stream by stream (ragged '.pts_*')
    per_stream_ids: bool = False    # --per-stream-ids: ... with each stream's tetrahedron particle IDs
    demo_n: int = 0                 # > 0: first generate the synthetic demo snapshot (crossed waves, n^3) ...
    demo_file: str = ""             # ... but only at THIS path (demo_spec sets it): the setting is persisted, and
                                    # before 2026-10-05 any later missing input quietly became a demo

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
        return REPO_ROOT / binary_name(self.estimator, self.precision, self.dim)

    def command(self) -> list[str]:
        ps = self.estimator == "ps"
        three = self.dim != 2           # a 2D run: no GPU, no parallel insertion (both 3D only)
        out = self.output_root()
        a = [str(self.binary()), str(Path(self.input_file).expanduser()), str(out),
             "--grid", str(self.grid), "--input", str(self.input_type), "--MpcUnit", f"{self.mpc_unit:g}"]
        if self.periodic:
            a.append("--periodic")
        if len(self.box) == 2 * self.dim:
            a += ["--box"] + [f"{v:g}" for v in self.box]
        fields = list(self.fields) if ps else dtfe_fields(self.fields)
        if self.scalar_dataset and "scalar_a" not in fields:
            fields.append("scalar_a")
        a += ["--field", *fields]
        if any(f in WEB_FIELDS for f in fields):      # the scripts' threshold, not the binary's 0.0
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
            if self.halo_release > 0:
                a += ["--ps-halo-release", _num(self.halo_release)]
            if len(self.window) == 2 * self.dim:
                a += ["--ps-window"] + [_num(v) for v in self.window]
        else:
            if self.gpu and three:
                a.append("--gpu")
            if self.deposit == "exact":
                a.append("--exact-average")
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
        return a

    def memory_key(self) -> str:
        return _memory_key(self, ("output_dir", "output_name", "demo_n", "demo_file", "lambda_th", "tess_cache"))

    def demo_pending(self) -> bool:
        """The demo snapshot is to be generated first: the demo's own input path, not on disk yet."""
        f = Path(self.input_file).expanduser() if self.input_file else None
        return (self.demo_n > 0 and bool(self.demo_file) and f is not None
                and f == Path(self.demo_file).expanduser() and not f.is_file())

    def check_step(self) -> Step:
        """The same binary call in report mode: its auto-tuner's split and memory prediction for this
        file, nothing computed or written (the report line parses as snapshot 0)."""
        return Step(f"memory check {Path(self.input_file).name or '?'}", self.command() + ["--auto-tune-report"],
                    self.run_env())

    def run_env(self) -> dict[str, str]:
        """The binary's environment: the memory budget (the cache is an argument here, not TESS_CACHE)."""
        return {k: v for k, v in _run_option_env(self).items() if k != "TESS_CACHE"}

    def steps(self) -> list[Step]:
        out = []
        if self.demo_pending():
            demo = Path(self.input_file).expanduser()
            out.append(Step("create the demo folder", ["mkdir", "-p", str(demo.parent)], {}))
            out.append(Step(f"generate the demo snapshot ({self.demo_n}{'²' if self.dim == 2 else '³'} particles, "
                            "crossed waves)",
                            [PYTHON, str(DEMO_GENERATOR), "--out", str(demo), "--n", str(self.demo_n),
                             "--box", "100", "--amplitude-factor", "1.5", "--crossed-waves"]
                            + (["--dim", "2"] if self.dim == 2 else []), {}))
        root = self.output_root()
        if not root.parent.is_dir() and str(root.parent) != str(Path(self.input_file).expanduser().parent):
            out.append(Step("create the output folder", ["mkdir", "-p", str(root.parent)], {}))
        out.append(Step(f"{self.binary().name} {Path(self.input_file).name or '?'}", self.command(), self.run_env(),
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
        if self.input_type not in INPUT_TYPES:
            out.append(("error", f"unknown input type {self.input_type}"))
        d = self.dim
        if d not in DIMS:
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
        out += _precision_problems(self.estimator, self.precision, d)
        if d == 2:
            if self.gpu:
                out.append(("info", "the 2D programs run on the CPU (the GPU kernels are 3D): the GPU setting is ignored"))
            if ps and self.parallel_triangulation:
                out.append(("info", "the parallel triangulation is 3D only: a 2D run inserts sequentially"))
        elif ps and self.parallel_triangulation and not tbb_built(self.precision):
            out.append(("warning", "parallel triangulation asked for, but the binary was built without TBB: ignored"))
        elif ps and self.parallel_triangulation:
            out.append(("warning", "parallel triangulation: faster on small sets, but two identical runs then differ "
                                   "at float rounding (the stream counts do not)"))
        n = hdf5_particle_count(f) if (f.is_file() and self.input_type == 105) else None
        out += gpu_advice(self.gpu, self.deposit, self.fields if ps else (dtfe_fields(self.fields) or DTFE_FIELDS), n,
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
        if not ps and not dtfe_fields(self.fields):
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
        if self.gpu and d == 3 and not gpu_built(self.estimator, self.precision):
            out.append(("warning", f"GPU requested, but the binary was built without GPU support ({GPU_BUILD_HINT}): "
                                   "it will use the CPU"))
        if not self.binary().is_file() and self.precision == "single":
            how = f"'make {self.binary().name}'" if d == 3 else f"'{build_hint('single', 2)}'"
            out.append(("error", f"{self.binary().name} is not built (Setup, or {how})"))
        out += _scratch_problems(self.scratch_dir)
        out += _run_option_problems(self)
        if self.sample_points:
            if not Path(self.sample_points).expanduser().is_file():
                out.append(("error", f"sample-points file not found: {self.sample_points}"))
            if not ps and self.partition > 0:
                out.append(("error", "sample points with standard DTFE evaluate ONE triangulation: set Partition to "
                                     "auto (the binary refuses an explicit split)"))
        if self.window:
            if not ps:
                out.append(("info", "the window is a PS-DTFE option: this standard-DTFE run ignores it"))
            elif (len(self.window) != 2 * d or not all(math.isfinite(v) for v in self.window)
                  or any(self.window[2 * i + 1] <= self.window[2 * i] for i in range(d))):
                out.append(("error", f"the window needs {2 * d} numbers in Mpc, each upper > lower"))
            elif any(f in WEB_FIELDS for f in self.fields):
                out.append(("error", "the T-web / V-web classes need the whole periodic grid: not with a window"))
            elif self.linear_deposit:
                out.append(("error", "the window cannot be combined with the linear deposit (the binary refuses it)"))
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
        out += _busy_problems()
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CustomSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


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
        gpu = recommended_gpu("sampled", fields, 32 ** 3)
    return CustomSpec(input_file=str(od / "crossed_waves_32.hdf5"), input_type=105, mpc_unit=1.0,
                      periodic=True, estimator="ps", lagrangian="file", grid=64, nsub=2,
                      fields=fields, gpu=gpu, vertex_mass=False, volume_weighted=True, caustics=True,
                      output_dir=str(od), output_name="demo", demo_n=32, demo_file=str(od / "crossed_waves_32.hdf5"))


# ---------------------------------------------------------------------- job queue
JOB_KINDS = {"data": DataSpec, "grids": RunSpec, "pipeline": PipelineSpec, "plots": PlotSpec,
             "custom": CustomSpec}
QUEUE_STATES = ("waiting", "running", "done", "failed", "stopped", "skipped")
# jobs that hold a triangulation and full grids: never started next to another DTFE run
HEAVY_KINDS = ("grids", "pipeline", "custom")


def _snaps_text(snaps) -> str:
    snaps = sorted(set(snaps))
    return " ".join(f"{n:03d}" for n in snaps[:8]) + (f" (+{len(snaps) - 8})" if len(snaps) > 8 else "")


def job_summary(kind: str, spec) -> str:
    """One line naming what a job does, for the queue list."""
    if kind == "custom":
        est = "PS-DTFE" if spec.estimator == "ps" else "DTFE"
        cells = "²" if getattr(spec, "dim", 3) == 2 else "³"
        return f"Custom snapshot · {Path(spec.input_file).name} · {est} {spec.grid}{cells} -> {spec.output_root()}"
    if kind == "grids":
        est = "PS-DTFE" if spec.estimator == "ps" else "DTFE"
        extra = ", exact" if spec.deposit == "exact" else ""
        return f"Grids · {spec.sim} · {_snaps_text(spec.snapshots)} · {est} {spec.grid}³{extra}"
    if kind == "pipeline":
        return (f"Pipeline · {', '.join(spec.sims)} · plane {spec.nu}² · grid {spec.grid}³"
                + (", exact" if spec.deposit == "exact" else "")
                + (" · plan only" if spec.plan_only else " · + figures" if spec.render else ""))
    if kind == "plots":
        names = [FIGURE_SET[k].title for k in spec.sets if k in FIGURE_SET]
        return f"Plots · {spec.sim} · {_snaps_text(spec.snapshots)} · {', '.join(names)}"
    parts = [w for w, on in (("download", spec._downloads()),
                             ("merge", spec.merge_snapshots or spec.merge_groupcats or spec.merge_trees
                              or spec.convert_ics)) if on]
    return f"Data · {spec.sim} · {_snaps_text(spec.snapshots)} · {' + '.join(parts) or 'nothing'}"


@dataclass
class QueuedJob:
    """A job frozen when it was queued: its settings (not its commands -- those are worked out
    when it starts, so a Data job fetches only what is still missing then, and a job queued after
    a download sees the new files). state: waiting | running | done | failed | stopped | skipped."""
    kind: str
    spec: dict
    title: str = ""
    state: str = "waiting"
    note: str = ""

    @classmethod
    def of(cls, kind: str, spec) -> "QueuedJob":
        return cls(kind, spec.to_dict(), job_summary(kind, spec))

    def build(self):
        return JOB_KINDS[self.kind].from_dict(self.spec)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QueuedJob":
        job = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if job.kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {job.kind!r}")
        if job.state == "running":          # the window closed during it: it was stopped
            job.state, job.note = "stopped", "interrupted when the window closed"
        return job


def next_waiting(queue: list[QueuedJob]) -> int | None:
    return next((i for i, j in enumerate(queue) if j.state == "waiting"), None)


# ---------------------------------------------------------------------- discovery helpers
def simulations(root) -> list[str]:
    return find_sims(root, "snapdir_*/combined_*.hdf5")


def snapshots(sim_path: Path, prefix: str = "ps_output", estimator: str = "ps", done=None) -> list[dict]:
    """Snapshots with a merged file: number, redshift (from the header) and whether outputs exist
    (done(snapdir) -> bool overrides the grid test, e.g. for point-evaluated slices)."""
    out = []
    for f in sorted(Path(sim_path).glob("snapdir_*/combined_[0-9][0-9][0-9].hdf5")):
        n = int(f.stem.split("_")[1])
        out.append({"n": n, "z": _redshift(f),
                    "done": done(f.parent) if done else has_outputs(f.parent, prefix, estimator)})
    return sorted(out, key=lambda s: s["n"])


def has_outputs(snapdir: Path, prefix: str, estimator: str) -> bool:
    if estimator == "ps":
        return (snapdir / f"{prefix}.a_den").exists() or (snapdir / f"{prefix}.den").exists()
    return any(snapdir.glob("output*.a_den")) or any((snapdir / "output").glob("*.a_den"))


def has_grids(snapdir: Path, method: str, prefix: str) -> bool:
    """Would a figure script's FieldSet(method=..., prefix=...) find grids here?"""
    ps = has_outputs(snapdir, prefix or "ps_output", "ps")
    if method == "ps" or prefix:
        return ps
    dtfe = has_outputs(snapdir, "", "dtfe")
    return dtfe if method == "dtfe" else (ps or dtfe)


def plane_sidecars(sim_path) -> list[Path]:
    """Image-plane sidecars (.json) in dtfelib.pointeval's discovery order."""
    sp = Path(sim_path)
    return sorted(sp.glob("pointeval_plane_*.json")) + sorted(sp.glob("hires_plane_*.json"))


IMAGE_SUFFIXES = (".png", ".pdf", ".jpg", ".jpeg", ".svg")


def figures(roots, since: float | None = None, limit: int = 2000) -> list[Path]:
    """Figure files under roots, newest first; since = only those modified at/after this time."""
    found: dict[Path, float] = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith(IMAGE_SUFFIXES):
                    path = Path(dirpath) / f
                    try:
                        mt = path.stat().st_mtime
                    except OSError:
                        continue
                    if since is None or mt >= since:
                        found[path.resolve()] = mt
    return sorted(found, key=lambda p: found[p], reverse=True)[:limit]


_Z_CACHE: dict[str, float | None] = {}


def _redshift(path: Path) -> float | None:
    key = f"{path}:{path.stat().st_mtime_ns}"
    if key not in _Z_CACHE:
        try:
            import h5py
            with h5py.File(path, "r") as h:
                _Z_CACHE[key] = float(h["Header"].attrs["Redshift"])
        except Exception:
            _Z_CACHE[key] = None
    return _Z_CACHE[key]


def planes(sim_path: Path) -> list[Path]:
    """--sample-points files next to the snapshots (make_image_plane.py output)."""
    return sorted(p for p in Path(sim_path).glob("*plane*.bin") if p.is_file())


def plane_points(path) -> int:
    """Points in a raw float64 x3 --sample-points file (0 if unreadable)."""
    try:
        return Path(path).stat().st_size // 24
    except OSError:
        return 0


def _objdir(estimator: str, precision: str = "single", dim: int = 3) -> Path:
    """The Makefile's object directory of a pair: o / o_ps, o_d / o_ps_d for the double pair, and
    '_2d' before the '_d' for the 2D set (o_2d, o_ps_2d_d)."""
    return REPO_ROOT / (("o_ps" if estimator == "ps" else "o") + ("_2d" if dim == 2 else "")
                        + ("_d" if precision == "double" else ""))


def gpu_built(estimator: str, precision: str = "single", dim: int = 3) -> bool:
    if dim == 2:                    # the GPU kernels deposit tetrahedra into 3D cells: a 2D build is CPU-only
        return False
    objdir = _objdir(estimator, precision)
    return objdir.is_dir() and not (objdir / ".gpu_mode_off").exists()


# the make argument that gives this machine a GPU build, for messages
GPU_BUILD_HINT = "METAL=1 on a Mac, CUDA=1 or HIP=1 on Linux"
GPU_BACKENDS = {"metal": "Metal", "cuda": "CUDA", "hip": "HIP"}


def gpu_backend(estimator: str, precision: str = "single", dim: int = 3) -> str:
    """The GPU backend a pair was built with ('Metal', 'CUDA', 'HIP'), from the Makefile's
    .gpu_mode_<mode> stamp; '' for a CPU-only build (every 2D one) or an unbuilt pair."""
    if dim == 2:
        return ""
    objdir = _objdir(estimator, precision)
    for stamp, name in GPU_BACKENDS.items():
        if (objdir / f".gpu_mode_{stamp}").exists():
            return name
    return ""


def gpu_compiler() -> str:
    """The GPU compiler this machine offers for a build: 'METAL=1' on a Mac (the Metal host needs
    only the system SDK), 'CUDA=1' when nvcc is on the PATH or under /usr/local/cuda, 'HIP=1' when
    hipcc is (or under /opt/rocm); '' when there is none -- the build is then CPU-only."""
    if sys.platform == "darwin":
        return "METAL=1"
    if shutil.which("nvcc") or Path("/usr/local/cuda/bin/nvcc").is_file():
        return "CUDA=1"
    if shutil.which("hipcc") or Path("/opt/rocm/bin/hipcc").is_file():
        return "HIP=1"
    return ""


def tbb_built(precision: str = "single", dim: int = 3) -> bool:
    """PS-DTFE was built with the TBB parallel insertion (o_ps/.tbb_1, the Makefile's stamp). The
    parallel insertion is 3D only (CGAL's parallel Delaunay): a 2D build inserts sequentially."""
    return dim == 3 and (_objdir("ps", precision) / ".tbb_1").exists()


# Where the GPU deposit pays, measured on this machine (2026-10-01, 256^3 sampled grids): 0.26M
# particles CPU 18 s vs GPU 24 s, 2.1M 51 vs 20 s, 7.1M 115 vs 40 s; the exact deposit: 0.26M CPU
# 57 s vs GPU 29 s (the threaded CPU deposit made it feasible; the GPU's lead grows with size);
# a run that writes no grid (a slice alone) deposits nothing.
# Standard DTFE (2026-10-01 evening, after the work-item kernel with the threadgroup cell table):
# 7.1M particles at 512^3 CPU 39.1 s vs GPU 17.3 s (with the velocity gradient 48.2 vs 32.0 s), 2.1M at
# 512^3 24.6 vs 10.6 s, 0.26M at 256^3 3.8 vs 2.0 s; 7.1M at 256^3 ties (14.2 vs 14.4 s: reading,
# triangulation and writing are the floor there). The per-tetrahedron kernel of the morning was 10x
# SLOWER at 512^3 (396.7 s). So the GUI starts standard-DTFE jobs on the GPU when it is built and
# says so when a fine-grid run has it unticked. Its exact cell average (--exact-average, 2026-10-02):
# 0.26M particles at 256^3 CPU 26.9 s vs GPU 4.1 s (interpolation step).
GPU_PAYS_FROM = 1_000_000
DTFE_GPU_PAYS_FROM_GRID = 512


def gpu_advice(gpu: bool, deposit: str, fields, n_particles, estimator: str,
               grid: int = 0, precision: str = "single", dim: int = 3) -> list[tuple[str, str]]:
    """A warning when the GPU setting is the slower one for this run; [] when it is right or unknown
    (and for a 2D run, which has no GPU)."""
    if dim == 2:
        return []
    if estimator == "dtfe":
        if deposit == "exact" and not gpu and gpu_built("dtfe", precision):
            return [("warning", "the exact cell average is ~6.6x faster on the GPU (4.1 s against 26.9 s at 256³ "
                                "with 0.26M particles); tick the GPU")]
        if not gpu and grid >= DTFE_GPU_PAYS_FROM_GRID and gpu_built("dtfe", precision):
            return [("warning", f"standard DTFE on a {grid}³ grid: the GPU interpolation is 1.5-2.3x faster than "
                                "the CPU (17 s against 39 s at 512³ with 7.1M particles); tick the GPU")]
        return []
    if estimator != "ps" or not fields:
        return []
    if deposit == "exact":
        if gpu or not gpu_built("ps", precision):
            return []
        return [("warning", "the exact deposit is 2x faster on the GPU at 0.26M particles (29 s against 57 s on the "
                            "CPU) and the gap grows with the particle count; tick the GPU")]
    if n_particles is None:
        return []
    if gpu and n_particles < GPU_PAYS_FROM:
        return [("warning", f"{n_particles/1e6:.2g}M particles: below ~1M the threaded CPU deposit is faster than "
                            "the GPU (measured 18 s against 24 s at 0.26M); untick the GPU")]
    if not gpu and n_particles >= 2 * GPU_PAYS_FROM and gpu_built("ps", precision):
        return [("warning", f"{n_particles/1e6:.2g}M particles: the GPU deposit is 2-3x faster from ~2M particles "
                            "(measured 20 s against 51 s at 2.1M); tick the GPU")]
    return []


def recommended_gpu(deposit: str, fields, n_particles, estimator: str = "ps", precision: str = "single",
                    dim: int = 3) -> bool:
    """The GPU setting a fresh job should start with (the GPU built; see gpu_advice). Standard DTFE:
    the GPU (1.5-2.3x faster on fine grids, a tie on coarse ones). A double build hands the GPU float
    copies of its tetrahedra: the deposit's sums are single precision, everything else double.
    A 2D job: never (the 2D programs are CPU-only)."""
    if not gpu_built(estimator, precision, dim):
        return False
    if estimator != "ps":
        return True
    if deposit == "exact":
        return True
    return bool(fields) and (n_particles is None or n_particles >= GPU_PAYS_FROM)


_JOBS_CACHE: tuple[float, dict] = (-1e9, {"runs": [], "servers": []})
_SERVE = re.compile(r"(^|\s)--serve(\s|$)")     # the '--serve' token itself, not --serve-resident or a path
_REPORT_RUN = re.compile(r"(^|\s)--auto-tune-report(\s|$)")   # a memory check: the tuner's prediction, no run


def dtfe_processes(max_age: float = 2.0) -> dict:
    """The DTFE / PS-DTFE binaries alive on this machine (from any terminal), classified once per max_age
    seconds (pgrep costs ~30 ms and the checks run on every click):
      'runs'     [(pid, name, gb)]  jobs that compute and occupy the machine; gb = their resident memory
                                    (None when ps could not say)
      'servers'  [(pid, name, gb)]  query servers ('--serve': resident, idle between requests, but their
                                    tessellations are real memory a large run has to live beside)
    Not listed at all: report-mode runs ('--auto-tune-report', a memory check from any tab or terminal) and
    this process's own direct children (the Explore tab's exact zoom). Counted as runs, the servers made every
    tab warn while Explore's server was up and a queue with 'wait' ticked waited forever; the jobs the
    launcher runs through a script are grandchildren (bash -> binary) and do count."""
    global _JOBS_CACHE
    now = time.monotonic()
    if now - _JOBS_CACHE[0] > max_age:
        try:
            # every set: PS-DTFE, DTFE-double, PS-DTFE-2d-double (Linux truncates the name to 15 characters)
            lines = subprocess.run(["pgrep", "-l", "^(PS-)?DTFE(-.*)?$"], capture_output=True, text=True,
                                   timeout=5).stdout.splitlines()
        except (OSError, subprocess.TimeoutExpired):
            lines = []
        names = {int(pid): name for pid, name in (ln.split(None, 1) for ln in lines if " " in ln)}
        _JOBS_CACHE = (now, _classify(names))
    return _JOBS_CACHE[1]


def forget_processes():
    """Drop the cached classification (a check or job of our own just ended: its process must not linger)."""
    global _JOBS_CACHE
    _JOBS_CACHE = (-1e9, {"runs": [], "servers": []})


def running_jobs(max_age: float = 2.0) -> list[tuple[int, str]]:
    """(pid, name) of the DTFE / PS-DTFE binaries COMPUTING on this machine (dtfe_processes()['runs'])."""
    return [(pid, name) for pid, name, _gb in dtfe_processes(max_age)["runs"]]


def running_servers(max_age: float = 2.0) -> list[tuple[int, str, float | None]]:
    """(pid, name, resident GB) of the query servers alive on this machine (dtfe_processes()['servers'])."""
    return list(dtfe_processes(max_age)["servers"])


def _classify(names: dict[int, str]) -> dict:
    """One ps call lists pid, parent, resident set and the full command line (-ww: never truncated; the
    repository path has spaces, so the line is matched, not split); macOS and procps spell it the same. A
    pid gone meanwhile is simply absent; without ps every named binary counts as a run, as before."""
    out = {"runs": [], "servers": []}
    if not names:
        return out
    try:
        text = subprocess.run(["ps", "-ww", "-o", "pid=,ppid=,rss=,args=", "-p", ",".join(map(str, names))],
                              capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        out["runs"] = [(pid, name, None) for pid, name in names.items()]
        return out
    me = os.getpid()
    for line in text.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3 or not all(v.isdigit() for v in parts[:3]):
            continue
        pid, ppid, rss_kb = int(parts[0]), int(parts[1]), int(parts[2])
        args = parts[3] if len(parts) > 3 else ""
        if pid not in names:
            continue
        kind = _kind(ppid, args, me)
        if kind in ("run", "server"):
            out["runs" if kind == "run" else "servers"].append((pid, names[pid], rss_kb / 1e6))
    return out


def _kind(ppid: int, args: str, me: int) -> str:
    """What a DTFE binary with this parent and command line is: 'run' (occupies the machine), 'server'
    (a query server: resident memory, idle CPU), 'report' (a memory check: the tuner's prediction only),
    'own' (a direct child of this launcher: its exact zoom). Decided 2026-10-05: report-mode runs are
    never counted (the memory check from any tab or terminal is not a run), servers are listed apart with
    their memory instead of being counted or hidden."""
    if _SERVE.search(args):              # a server is a server, the launcher's own included (its memory counts)
        return "server"
    if _REPORT_RUN.search(args):
        return "report"
    if ppid == me:
        return "own"
    return "run"


def _gb_text(gb) -> str:
    return "" if gb is None else f", {gb:.1f} GB"


# ---------------------------------------------------------------------- progress parsing
_PARTS = re.compile(r"\[partitions (\d+)/(\d+) \| (\d+)%\]\s+elapsed ([^,]+),\s*"
                    r"(?:ETA ~?(\d+h(?: \d+m)?(?: \d+s)?|\d+m(?: \d+s)?|\d+s)|done)")
# wget -nv: one line per file fetched ('... URL:<url> [<bytes>/<bytes>] -> "<path>" [1]')
_WGET = re.compile(r'URL:\S+ \[(\d+)(?:/\d+)?\] -> "([^"]+\.hdf5)"')
_CHUNK = re.compile(r"^\s*\[(\d+)/(\d+)\] ")                  # merge_HDF5.py: '  [k/N] ...'
# lines that mean something went wrong although the tool exits 0 (download/merge report, not fail)
PROBLEM = re.compile(r"MISMATCH|Problem: Can't find|Error downloading|Error: |FAILED|NOT deleting")
PROGRESS = re.compile(r"\s*\[partitions \d+/\d+ \| \d+%\][^\n\[]*")   # the fragment to hide in logs
_SNAP = re.compile(r"Processing snapshot (\d+)")
# run_ps_pipeline.sh echoes the run_ps_dtfe.sh call with the snapshots it will compute
_PLAN = re.compile(r"^\+ .*run_ps_dtfe\.sh -s (\S+) -g \d+(?: -m)?((?: \d+)+)\s*$")   # -m only with PS_GPU=1
_SIM = re.compile(r"^== (\S+): snapshots")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(s: str) -> str:
    return _ANSI.sub("", s)


def parse_progress(text: str) -> dict:
    """The latest partition progress and snapshot seen in a chunk of run-script output."""
    info: dict = {}
    for m in _SNAP.finditer(text):
        info["snapshot"] = int(m.group(1))
    for m in _PARTS.finditer(text):
        info.update(done=int(m.group(1)), total=int(m.group(2)), percent=int(m.group(3)),
                    elapsed=m.group(4).strip(), eta=(m.group(5) or "done").strip())
    if "processed successfully" in text:
        info["snapshot_finished"] = True
    for m in _WGET.finditer(text):
        info["downloaded"] = info.get("downloaded", []) + [(m.group(2), int(m.group(1)))]
    for line in text.splitlines():
        m = _CHUNK.match(line)
        if m:
            info["chunk"] = (int(m.group(1)), int(m.group(2)))
        if PROBLEM.search(line):
            info["problem"] = info.get("problem", 0) + 1
        m = _PLAN.match(line)
        if m:
            info["planned"] = info.get("planned", 0) + len(m.group(2).split())
        m = _SIM.match(line)
        if m:
            info["sim"] = m.group(1)
    return info
