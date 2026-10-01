"""What earlier runs left on disk, read back from their run logs -- for the GUI's Runs browser,
its time estimates and its failure advice. Pure Python (no Qt): tests/py_gui_runspec_test.py.

  scan_runs(root, extra_dirs)   every '<prefix>.runlog' under the data root (per-family layout)
                                and under the custom-snapshot output directories -> RunRecord
  stale_reasons(record)         which KNOWN_FIXES (output-changing fixes of the binary) the run
                                predates, judged by the binary's build stamp ('Build:' line, since
                                2026-09-30) or, without one, by the date the run finished
  time_estimate(records, ...)   the median wall time of comparable earlier runs
  diagnose(log, exit_codes)     plain-language causes and fixes for a failed job
  notify_command(title, text)   the OS command that shows a desktop notification

A run log is the binary's own output, tee'd next to its outputs by run_ps_dtfe.sh, run_dtfe.sh
and the GUI's custom-snapshot jobs, followed (macOS) by '/usr/bin/time -l'.
"""

from __future__ import annotations

import datetime as _dt
import re
import shutil
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------- run records
_BUILD = re.compile(r"^Build: (\S+) (\S+) \(built ([0-9T:\-]+Z?)\)", re.M)
_RUNNING = re.compile(r"^RUNNING: (.*)$", re.M)
_WALL = re.compile(r"total wall time\s*:\s*((?:\d+h\s*)?(?:\d+m\s*)?(?:[\d.]+s)?)")
_RSS = re.compile(r"peak memory \(RSS\)\s*:\s*([\d.]+)\s*GB")
_FOOTPRINT = re.compile(r"^\s*(\d+)\s+peak memory footprint", re.M)
_GRID = re.compile(r"--grid (\d+)")
_PARTITION = re.compile(r"--partition (\d+) (\d+) (\d+)")
_CONC = re.compile(r"--max-concurrent (\d+)")
_NSUB = re.compile(r"--avg-subsamples (\d+)")
_FIELDS = re.compile(r"--field ((?:(?!--)\S+\s*)+)")


def parse_duration(text: str) -> float | None:
    """'1h 2m 3s' / '58m 11s' / '12.3s' -> seconds."""
    total, found = 0.0, False
    for num, unit in re.findall(r"([\d.]+)\s*([hms])", text or ""):
        total += float(num) * {"h": 3600, "m": 60, "s": 1}[unit]
        found = True
    return total if found else None


def format_duration(sec: float | None) -> str:
    if sec is None:
        return "—"
    sec = int(round(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s" if m else f"{s}s"


@dataclass
class RunRecord:
    path: Path                      # the .runlog
    sim: str = ""                   # "" for a custom-snapshot run
    snap: int | None = None
    prefix: str = ""
    binary: str = ""                # PS-DTFE | DTFE ("" when unknown)
    command: str = ""               # the RUNNING line (paths unquoted, as the binary printed it)
    build_rev: str = ""
    build_time: _dt.datetime | None = None
    finished: _dt.datetime | None = None   # the log's modification time = when the run ended
    wall_seconds: float | None = None
    peak_rss_gb: float | None = None
    footprint_gb: float | None = None
    ok: bool = False                # reached the binary's summary without an ERROR block
    outputs: list[Path] = field(default_factory=list)
    redshift: float | None = None

    # -- the settings the RUNNING line shows
    def has(self, flag: str) -> bool:
        return re.search(rf"(^|\s){re.escape(flag)}(\s|$)", self.command) is not None

    @property
    def estimator(self) -> str:
        return "ps" if self.binary == "PS-DTFE" else "dtfe" if self.binary == "DTFE" else ""

    @property
    def grid(self) -> int | None:
        m = _GRID.search(self.command)
        return int(m.group(1)) if m else None

    @property
    def partitions(self) -> int:
        m = _PARTITION.search(self.command)
        return int(m.group(1)) * int(m.group(2)) * int(m.group(3)) if m else 1

    @property
    def partition(self) -> int:
        m = _PARTITION.search(self.command)
        return int(m.group(1)) if m else 0

    @property
    def max_concurrent(self) -> int:
        m = _CONC.search(self.command)
        return int(m.group(1)) if m else 0

    @property
    def nsub(self) -> int | None:
        m = _NSUB.search(self.command)
        return int(m.group(1)) if m else None

    @property
    def gpu(self) -> bool:
        return self.has("--ps-gpu") or self.has("--gpu")

    @property
    def sliced(self) -> bool:
        return self.has("--sample-points")

    @property
    def fields(self) -> list[str]:
        m = _FIELDS.search(self.command)
        return m.group(1).split() if m else []

    @property
    def when(self) -> _dt.datetime | None:
        """The binary's build time if the log names it, else when the run finished."""
        return self.build_time or self.finished

    @property
    def size_bytes(self) -> int:
        total = 0
        for p in self.outputs:
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    def options_text(self) -> str:
        bits = []
        if self.estimator == "ps":
            if self.has("--ps-exact-deposit"):
                bits.append("exact")
            elif self.nsub:
                bits.append(f"nSub {self.nsub}")
            if self.has("--ps-vertex-mass"):
                bits.append("vertex mass")
            if self.has("--ps-volume-weighted"):
                bits.append("volume-weighted")
            if self.has("--ps-caustics"):
                bits.append("caustics")
        if self.gpu:
            bits.append("GPU")
        if self.partitions > 1:
            bits.append(f"{self.partition}³ partitions")
        if self.sliced:
            bits.append("slice")
        return ", ".join(bits)

    def settings(self) -> dict:
        """The Grids-tab (RunSpec) settings this run used, as far as its command shows them."""
        d = {"estimator": self.estimator or "ps", "gpu": self.gpu}
        if self.grid:
            d["grid"] = self.grid
        if self.estimator == "ps":
            d.update(deposit="exact" if self.has("--ps-exact-deposit") else "sampled",
                     vertex_mass=self.has("--ps-vertex-mass"),
                     volume_weighted=self.has("--ps-volume-weighted"),
                     caustics=self.has("--ps-caustics"), caustic_cusps=self.has("--ps-caustic-cusps"),
                     parallel_triangulation=self.has("--parallel-triangulation"),
                     output_prefix=self.prefix or "ps_output")
            m = re.search(r"(?:^|\s)--ps-alpha-shape\s+([0-9.]+)", self.command)
            if m:
                d["alpha_shape"] = float(m.group(1))
            if self.nsub:
                d["nsub"] = self.nsub
            if self.fields:
                d["fields"] = self.fields
        return d


def parse_runlog(path: Path, sim: str = "", snap: int | None = None) -> RunRecord:
    path = Path(path)
    rec = RunRecord(path=path, sim=sim, snap=snap, prefix=path.name[: -len(".runlog")])
    try:
        text = path.read_text(errors="replace")
        rec.finished = _dt.datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return rec
    m = _BUILD.search(text)
    if m:
        rec.binary, rec.build_rev = m.group(1), m.group(2)
        try:
            rec.build_time = _dt.datetime.fromisoformat(m.group(3).replace("Z", "+00:00")) \
                .astimezone().replace(tzinfo=None)
        except ValueError:
            pass
    m = _RUNNING.search(text)
    if m:
        rec.command = m.group(1).strip()
        exe = rec.command.split()[0] if rec.command else ""
        rec.binary = rec.binary or ("PS-DTFE" if exe.endswith("PS-DTFE") else "DTFE" if exe.endswith("DTFE") else "")
    m = _WALL.search(text)
    rec.wall_seconds = parse_duration(m.group(1)) if m else None
    m = _RSS.search(text)
    rec.peak_rss_gb = float(m.group(1)) if m else None
    feet = _FOOTPRINT.findall(text)
    rec.footprint_gb = int(feet[-1]) / 1e9 if feet else None
    rec.ok = bool(m or rec.wall_seconds is not None) and "~~~ ERROR ~~~" not in text
    rec.outputs = sorted(p for p in path.parent.glob(rec.prefix + ".*")
                         if p.is_file() and p.suffix not in (".runlog", ".json"))
    return rec


_Z: dict[str, float | None] = {}


def _redshift_of(snapdir: Path) -> float | None:
    """The snapshot's redshift from its combined file's header (cached per file and mtime)."""
    files = sorted(Path(snapdir).glob("combined_[0-9][0-9][0-9].hdf5"))
    if not files:
        return None
    f = files[0]
    try:
        key = f"{f}:{f.stat().st_mtime_ns}"
    except OSError:
        return None
    if key not in _Z:
        try:
            import h5py
            with h5py.File(f, "r") as h:
                _Z[key] = float(h["Header"].attrs["Redshift"])
        except Exception:
            _Z[key] = None
    return _Z[key]


def _snap_of(snapdir: Path) -> int | None:
    try:
        return int(snapdir.name.split("_")[1])
    except (IndexError, ValueError):
        return None


def scan_runs(root, extra_dirs=()) -> list[RunRecord]:
    """Every run log under the data root (<family>/<sim>/snapdir_NNN or flat <sim>/snapdir_NNN)
    and in the extra (custom-snapshot output) directories, newest first."""
    out: list[RunRecord] = []
    root = Path(root)
    seen: set[Path] = set()
    if root.is_dir():
        for pattern in ("*/snapdir_*/*.runlog", "*/*/snapdir_*/*.runlog"):
            for log in root.glob(pattern):
                if log in seen:
                    continue
                seen.add(log)
                rec = parse_runlog(log, log.parent.parent.name, _snap_of(log.parent))
                rec.redshift = _redshift_of(log.parent)
                out.append(rec)
    for d in extra_dirs:
        d = Path(d)
        if d.is_dir():
            for log in d.glob("*.runlog"):
                if log not in seen:
                    seen.add(log)
                    out.append(parse_runlog(log))
    return sorted(out, key=lambda r: r.finished or _dt.datetime.min, reverse=True)


# ---------------------------------------------------------------------- known output-changing fixes
@dataclass(frozen=True)
class Fix:
    key: str
    fixed: str                      # ISO local time: outputs made (built) before this are affected
    level: str                      # error (wrong) | warning (biased) | info
    message: str
    applies: object                 # RunRecord -> bool: the run is of the affected kind


def _any_output(rec: RunRecord, *suffixes) -> bool:
    names = {p.name[len(rec.prefix) + 1:] for p in rec.outputs}
    return any(s in names for s in suffixes)


KNOWN_FIXES: tuple[Fix, ...] = (
    Fix("gpu-partition-wrap", "2026-09-29T03:00", "error",
        "periodic runs split into partitions on the GPU skipped the wrapped part of every "
        "partition's sub-grid (most cells went to centroid fallbacks; fixed 2026-09-29): regenerate",
        lambda r: r.estimator == "ps" and r.gpu and r.partitions > 1),
    Fix("vweb-sqrt-a", "2026-07-18T00:00", "warning",
        "V-web eigenvalues carried an extra 1/sqrt(a) at z > 0 (fixed 2026-07-18)",
        lambda r: _any_output(r, "velVweb", "a_velVweb") and (r.redshift or 1.0) > 0.01),
    Fix("caustic-d4", "2026-07-28T00:00", "info",
        "caustic-class bit 6 (the D4 indicator) was near-vacuous (fixed 2026-07-28)",
        lambda r: _any_output(r, "causticClass")),
    Fix("cache-periodic", "2026-09-30T12:00", "info",
        "a standard-DTFE run with a tessellation cache could load the tessellation of the other "
        "periodicity (fixed 2026-09-30)",
        lambda r: r.estimator == "dtfe" and r.has("--tessellation-cache")),
    # the old flag is the file '.unresolved' (renamed '.hidden_streams' when it became exact)
    Fix("hidden-streams", "2026-09-30T15:30", "warning",      # (local time, like the build stamps)
        "made before the exact hidden-streams flag (2026-09-30): single-stream masks drop cells "
        "that only hold a small tetrahedron of their own stream, and keep cells crossed by walls "
        "thinner than the sampling. Regenerate before quoting single-stream statistics",
        lambda r: r.estimator == "ps" and _any_output(r, "unresolved", "a_unresolved")),
    Fix("region-faces", "2026-09-30T20:57", "warning",
        "a non-periodic region cut from a larger cloud: tetrahedra crossing the region's face gave "
        "their whole mass to the part inside, so face cells read up to 3x too dense and tetrahedra spanning "
        "many cells biased the interior too (fixed 2026-09-30): regenerate",
        lambda r: r.estimator == "ps" and not r.has("--periodic") and r.has("--box")),
    Fix("alpha-shape", "2026-09-30T21:15", "warning",
        "a non-periodic run: the flat slivers on the cloud's convex hull (particles a whole face apart) "
        "put spurious mass and streams across the grid, and differed between partitions, so split and "
        "auto-tuned runs disagreed with one triangulation at the partition seams. The Lagrangian domain "
        "is now the cloud's alpha shape (fixed 2026-09-30): regenerate",
        lambda r: r.estimator == "ps" and not r.has("--periodic")),
)


def stale_reasons(rec: RunRecord) -> list[tuple[str, str]]:
    """(level, message) for every known fix this run predates, plus problems its outputs show."""
    out: list[tuple[str, str]] = []
    when = rec.when
    for fx in KNOWN_FIXES:
        if when is None or not fx.applies(rec):
            continue
        if when < _dt.datetime.fromisoformat(fx.fixed):
            how = "" if rec.build_time else " (judged by the run date: this log has no build stamp)"
            out.append((fx.level, fx.message + how))
    # file-based: '.streams' without '.hidden_streams' (or its old name '.unresolved') predates the
    # 2026-09-28 stream-count fix
    if rec.estimator == "ps" and _any_output(rec, "streams", "a_streams") \
            and not _any_output(rec, "hidden_streams", "a_hidden_streams", "unresolved", "a_unresolved"):
        out.append(("warning", "the stream counts still include centroid-fallback tetrahedra (+1 each; "
                               "fixed 2026-09-28): stream fractions are inflated, so regenerate "
                               "before quoting them"))
    if rec.estimator == "ps" and rec.sim.startswith("TNG") and rec.command and not rec.has("--ps-vertex-mass"):
        out.append(("warning", "made without tetrahedron masses from the particles: with TNG's z=127 "
                               "initial positions the density contrast is suppressed by 1 - D(127)/D(z)"))
    return out


# ---------------------------------------------------------------------- time estimates
def time_estimate(records, *, sim: str, grid: int, estimator: str, gpu: bool,
                  sliced: bool) -> tuple[float, int] | None:
    """(median seconds per snapshot, number of runs) of successful earlier runs with the same
    simulation, grid, estimator, GPU use and slice/no slice; None without any."""
    times = [r.wall_seconds for r in records
             if r.ok and r.wall_seconds and r.sim == sim and r.grid == grid
             and r.estimator == estimator and r.gpu == gpu and r.sliced == sliced]
    return (statistics.median(times), len(times)) if times else None


# ---------------------------------------------------------------------- failure advice
@dataclass
class Diagnosis:
    title: str
    advice: str
    action: str = ""                # a GUI action key ("check_memory", "scratch", "choose_root",
                                    # "open_data", "build", "") -- the button the banner offers


_OOM = re.compile(r"std::bad_alloc|Cannot allocate memory|out of memory|Killed: 9|MemoryError", re.I)


def diagnose(log: str, exit_codes=()) -> list[Diagnosis]:
    """Likely causes of a failed job from the tail of its log and its exit codes."""
    tail = "\n".join(log.splitlines()[-400:])
    out: list[Diagnosis] = []
    codes = list(exit_codes)
    if _OOM.search(tail) or any(c in (137, -9, 9) for c in codes):
        out.append(Diagnosis("Out of memory",
                             "The run needed more memory than the machine had. 'Check memory' shows "
                             "what each snapshot needs; a scratch directory moves the grids to disk "
                             "(bit-identical results), and a smaller slice plane or fewer concurrent "
                             "partitions lower the rest.", "check_memory"))
    if re.search(r"no snapshots under|data root not found|No such file or directory: '?/Volumes/", tail):
        out.append(Diagnosis("Data not found",
                             "The data folder is missing. Is the external disk plugged in? Choose "
                             "the data root again.", "choose_root"))
    if re.search(r"No 'InitialCoordinates'|neither .*InitialCoordinates|lagrangian.*(not found|cannot open)",
                 tail, re.I) and (codes and any(codes)):
        out.append(Diagnosis("Initial positions missing",
                             "PS-DTFE needs each particle's initial (Lagrangian) position: download "
                             "and convert the initial conditions (Data tab), or use a snapshot that "
                             "stores 'InitialCoordinates'.", "open_data"))
    if "unrecognised option" in tail or "unrecognized option" in tail:
        out.append(Diagnosis("The binary is older than the scripts",
                             "An option was not recognised: rebuild the binaries (make DTFE PS-DTFE).",
                             "build"))
    if "points into iCloud" in tail or "iCloud Drive" in tail:
        out.append(Diagnosis("iCloud path refused",
                             "Scratch and cache directories must be on a local disk, not iCloud Drive "
                             "(e.g. /private/tmp/dtfe-scratch).", "scratch"))
    if re.search(r"Operation not permitted|Permission denied", tail):
        out.append(Diagnosis("No permission",
                             "macOS refused access to a file. Apps started from Finder need Full Disk "
                             "Access for iCloud Drive and external disks (System Settings -> Privacy & "
                             "Security -> Full Disk Access).", ""))
    if re.search(r"OVER BUDGET|over the memory budget|exceed the memory budget", tail) and not out:
        out.append(Diagnosis("Over the memory budget",
                             "The auto-tuner warned that this run does not fit in memory.", "check_memory"))
    if not out and any(codes):
        out.append(Diagnosis("The run failed",
                             "See the last lines of the log above for the program's own message.", ""))
    return out


# ---------------------------------------------------------------------- notifications
def notify_command(title: str, text: str) -> list[str] | None:
    """The command that shows a desktop notification, or None where there is none."""
    if sys.platform == "darwin" and shutil.which("osascript"):
        esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')   # noqa: E731
        return ["osascript", "-e", f'display notification "{esc(text)}" with title "{esc(title)}"']
    if shutil.which("notify-send"):
        return ["notify-send", title, text]
    return None
