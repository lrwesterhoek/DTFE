"""What earlier runs left on disk, read back from their run logs -- for the GUI's Runs browser,
its time estimates and its failure advice. Pure Python (no Qt): tests/py_gui_runspec_test.py.

  scan_runs(root, extra_dirs)   every '<prefix>.runlog' under the data root (per-family layout)
                                and under the custom-snapshot output directories -> RunRecord, with
                                a later failed, killed or running attempt ('.runlog.failed/.partial')
  stale_reasons(record)         which KNOWN_FIXES (output-changing fixes of the binary) the run
                                predates, judged by the binary's build stamp ('Build:' line, since
                                2026-09-30) or, without one, by the date the run finished; and the
                                attempt after it (attempt_note)
  time_estimate(records, ...)   the median wall time of comparable earlier runs
  diagnose(log, exit_codes)     plain-language causes and fixes for a failed job
  void_status(records, root)    whether each run's void catalogue is cached, stale or absent (void_cache.py)
  read_progress / is_run_file   the run-directory files of a run going (contract C1, 2026-10-11): the binary's
                                '<prefix>.runlog.progress', the scripts' '.runlog.lock' and '<grid>.part.<pid>'
                                are never outputs nor logs; a '.partial' with a live sidecar shows its progress
  notify_command(title, text)   the OS command that shows a desktop notification

A run log is the binary's own output, tee'd next to its outputs by run_ps_dtfe.sh, run_dtfe.sh
and the GUI's custom-snapshot jobs, followed (macOS) by '/usr/bin/time -l'. The run-log contract
(2026-10-08): a run writes '<prefix>.runlog.partial'; exit 0 renames it onto '<prefix>.runlog' (the
log of the grids on disk), a failure or stop onto '<prefix>.runlog.failed' (the last failed attempt,
the old '.runlog' untouched); a '.partial' left behind is a run still going, or one killed outright.
"""

from __future__ import annotations

import datetime as _dt
import glob as _glob
import json
import math
import re
import shlex
import shutil
import statistics
import subprocess
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
_PARTITION = re.compile(r"--partition (\d+) (\d+)(?: (\d+))?")    # a 2D run's split has two numbers
_CONC = re.compile(r"--max-concurrent (\d+)")
_NSUB = re.compile(r"--avg-subsamples (\d+)")
_FIELDS = re.compile(r"--field ((?:(?!--)\S+\s*)+)")
# the GPU deposit restarted after a watchdog kill: DTFE says 'the deposit', PS-DTFE 'the partition deposit'
_GPU_RETRY = re.compile(r"retrying the (?:partition )?deposit from scratch")
# the run-log contract's three names: the log of record, the last failed attempt, a run going or killed
LOG_SUFFIXES = (".runlog", ".runlog.failed", ".runlog.partial")
# the run-directory files of a run going (contract C1, 2026-10-11): the binary's JSON progress sidecar, the run
# scripts' lock DIRECTORY (holding the driver's 'pid'), and a grid written as '<file>.part.<pid>' until renamed
# at the end. None of them is an output or a run log.
PROGRESS_SUFFIX = ".runlog.progress"
LOCK_SUFFIX = ".runlog.lock"
# the checkpoint DIRECTORY of a killed partitioned PS-DTFE run (contract K2, 2026-10-11): its merged accumulators
# and the manifest of finished partitions; the next run of the same job resumes from it ('--resume', which
# run_ps_dtfe.sh passes by itself when it is there) and removes it on success
CHECKPOINT_SUFFIX = ".runlog.checkpoint"
# the binary's errors when '--resume' cannot continue the checkpoint (DTFE.cpp, 2026-10-11): other settings, nothing
# to resume (no manifest), not partitioned, a '--sample-points' run. Only the error phrasings: its info lines
# ("'--resume' continues it", "... continues a killed run from it") are in the logs of runs that CAN be resumed.
RESUME_REFUSED = re.compile(r"'--resume'(?::|\s+refused|\s+cannot|\s+continues a partitioned run)")
_PART = re.compile(r"\.part\.\d+$")


def is_run_file(name: str) -> bool:
    """A file of a run going (C1): its progress sidecar (and the sidecar's '.tmp'), its lock, a grid half written
    as '<file>.part.<pid>', a killed run's checkpoint directory and what is in it (K2). Never an output, never a
    run log."""
    return (PROGRESS_SUFFIX in name or LOCK_SUFFIX in name or CHECKPOINT_SUFFIX in name or _PART.search(name) is not None
            or name.endswith(".part.tmp"))


def pid_alive(pid) -> bool | None:
    """Whether process 'pid' exists (None: cannot say). EPERM means it exists under another user."""
    import os
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


_DTFE_NAME = re.compile(r"^(PS-)?DTFE(-.*)?$")    # every binary set (runspec.dtfe_processes' pgrep pattern)


def dtfe_pid_alive(pid) -> bool | None:
    """Whether process 'pid' is alive AND a DTFE / PS-DTFE binary (None: cannot say). A bare pid test is not
    enough for a leftover sidecar (2026-10-11): a run killed outright with its driver (a reboot, a group SIGKILL)
    leaves its sidecar, and once the launcher's own polling has wrapped the pids (~a day) its pid belongs to some
    other process, which read as the run still going. 'ps -o comm=' is argv[0] on macOS (a path, maybe with
    spaces) and the executable's 15-character name on Linux: its basename is matched. Without ps the bare pid
    test stands, as before."""
    alive = pid_alive(pid)
    if not alive:
        return alive
    try:
        pr = subprocess.run(["ps", "-p", str(int(pid)), "-o", "comm="], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return alive
    comm = pr.stdout.strip()
    if pr.returncode != 0 or not comm:
        return False                                # gone between the two tests
    return _DTFE_NAME.match(comm.rstrip("/").rsplit("/", 1)[-1]) is not None


def read_progress(path) -> dict | None:
    """The binary's progress sidecar '<prefix>.runlog.progress' (C1) as a dict, with 'state' added: 'running'
    (its pid is a live DTFE binary, dtfe_pid_alive), 'died' (the pid is gone or now another program's: a run
    killed outright leaves the sidecar behind; a run that finishes removes it) or 'unknown'. None without a
    readable sidecar (the binary writes it atomically, so a half-written one is never seen; a missing key reads
    as absent)."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    alive = dtfe_pid_alive(d.get("pid"))
    d["state"] = "running" if alive else "died" if alive is False else "unknown"
    return d


def progress_text(d: dict | None) -> str:
    """'phase 3/27 (11%), ETA 2h 05m, 37.2 GB' -- one line for a progress sidecar; 'died' for a dead pid."""
    if not d:
        return ""
    bits = []
    phase = str(d.get("phase") or "").strip()
    done, total = d.get("done"), d.get("total")
    try:
        frac = f"{int(done)}/{int(total)}" + (f" ({100.0 * float(done) / float(total):.0f}%)" if float(total) > 0 else "")
    except (TypeError, ValueError):
        frac = ""
    head = " ".join(b for b in (phase, frac) if b)
    if head:
        bits.append(head)
    if d.get("state") == "died":
        bits.append("died (its process is gone)")
        return ", ".join(bits)
    for key, label in (("eta_s", "ETA"), ("elapsed_s", "elapsed")):
        try:
            v = float(d.get(key))
        except (TypeError, ValueError):
            continue
        if v >= 0 and math.isfinite(v):
            bits.append(f"{label} {format_duration(v)}")
            break
    try:
        bits.append(f"{float(d.get('rss_gb')):.1f} GB")
    except (TypeError, ValueError):
        pass
    try:
        if int(d.get("gpu_retries") or 0):
            bits.append(f"{int(d['gpu_retries'])} GPU retries")
    except (TypeError, ValueError):
        pass
    return ", ".join(bits)


def log_prefix(name: str) -> str:
    """'ps_output.runlog.failed' -> 'ps_output' (any of the three log names; a name without one stays)."""
    for sfx in LOG_SUFFIXES:
        if name.endswith(sfx):
            return name[: -len(sfx)]
    return name


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
    binary: str = ""                # PS-DTFE | DTFE (-2d, -double; "" when unknown)
    command: str = ""               # the RUNNING line (paths unquoted, as the binary printed it)
    build_rev: str = ""
    build_time: _dt.datetime | None = None
    finished: _dt.datetime | None = None   # the log's modification time = when the run ended
    gpu_retried: bool = False       # the GPU deposit was restarted after a watchdog kill (the log says so)
    wall_seconds: float | None = None
    peak_rss_gb: float | None = None
    footprint_gb: float | None = None
    ok: bool = False                # reached the binary's summary without an ERROR block
    outputs: list[Path] = field(default_factory=list)
    redshift: float | None = None
    # a later run under this name that left no log of record (scan_runs, the run-log contract; 2026-10-08):
    # 'failed' ('<prefix>.runlog.failed'), 'running' or 'killed' ('<prefix>.runlog.partial' with or without a
    # live process naming it; 'partial' when ps cannot say), 'resumable' (failed or killed, with its checkpoint
    # directory '<prefix>.runlog.checkpoint' left: the same job continues from it, K2), '' when none is newer
    # than this log. Without a '<prefix>.runlog' the record IS that attempt (path == attempt_log, never ok).
    attempt: str = ""
    attempt_log: Path | None = None
    attempt_time: _dt.datetime | None = None
    attempt_cause: str = ""         # diagnose()'s titles for the attempt's log ("Out of memory", ...)
    attempt_resume_refused: bool = False   # the attempt failed because the binary refused to resume its checkpoint
                                           # (RESUME_REFUSED): never 'resumable', the same job fails again (2026-10-11)
    attempt_rewrote: int = 0        # outputs modified after this log was: the attempt rewrote them
    progress: dict | None = None    # the binary's progress sidecar of a run going (C1, read_progress; 2026-10-11)

    # -- the settings the RUNNING line shows
    def has(self, flag: str) -> bool:
        return re.search(rf"(^|\s){re.escape(flag)}(\s|$)", self.command) is not None

    @property
    def estimator(self) -> str:
        return "ps" if self.binary.startswith("PS-DTFE") else "dtfe" if self.binary.startswith("DTFE") else ""

    @property
    def precision(self) -> str:
        """'double' for the DTFE-double / PS-DTFE-double pair (their Build line names it), else 'single'."""
        return "double" if self.binary.endswith("-double") else "single"

    @property
    def dim(self) -> int:
        """2 for the 2D programs (DTFE-2d, PS-DTFE-2d[-double]), else 3."""
        return 2 if "-2d" in self.binary else 3

    @property
    def grid(self) -> int | None:
        m = _GRID.search(self.command)
        return int(m.group(1)) if m else None

    @property
    def partitions(self) -> int:
        m = _PARTITION.search(self.command)
        return math.prod(int(g) for g in m.groups() if g) if m else 1

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

    def checkpoint_dir(self) -> Path:
        """Where a killed partitioned run of this output keeps its checkpoint (K2; under '--scratch-dir' it lives
        there instead, which a scan of the output folder cannot see)."""
        return Path(str(self.path.parent / self.prefix) + CHECKPOINT_SUFFIX)

    def log_files(self) -> list[Path]:
        """Every log of this output set on disk (the run-log contract's three names, 2026-10-08): what Move to
        Trash must take with the outputs, or the attempt's log alone shows up as a failed run afterwards."""
        base = str(self.path.parent / self.prefix)
        return [p for p in (Path(base + sfx) for sfx in LOG_SUFFIXES) if p.is_file()]

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
            if self.has("--ps-linear-deposit"):       # the two deposit variants of one snapshot must differ here
                bits.append("linear deposit")
            if self.has("--ps-vertex-mass"):
                bits.append("vertex mass")
            if self.has("--ps-volume-weighted"):
                bits.append("volume-weighted")
            if self.has("--ps-caustics"):
                bits.append("caustics")
            if self.has("--ordered-merge"):
                bits.append("ordered merge")
        if self.gpu:
            bits.append("GPU")
        if self.precision == "double":
            bits.append("double")
        if self.partitions > 1:
            bits.append(f"{self.partition}{'²' if self.dim == 2 else '³'} partitions")
        if self.dim == 2:
            bits.append("2D")
        if self.sliced:
            bits.append("slice")
        return ", ".join(bits)

    def settings(self) -> dict:
        """The Grids-tab (RunSpec) settings this run used, as far as its command shows them."""
        d = {"estimator": self.estimator or "ps", "gpu": self.gpu, "precision": self.precision}
        if self.grid:
            d["grid"] = self.grid
        if self.estimator == "ps":
            d.update(deposit="exact" if self.has("--ps-exact-deposit") else "sampled",
                     # always set: 'Load settings' must also turn the box OFF for a standard-deposit run (2026-10-08)
                     linear_deposit=self.has("--ps-linear-deposit"),
                     vertex_mass=self.has("--ps-vertex-mass"),
                     volume_weighted=self.has("--ps-volume-weighted"),
                     caustics=self.has("--ps-caustics"), caustic_cusps=self.has("--ps-caustic-cusps"),
                     parallel_triangulation=self.has("--parallel-triangulation"),
                     ordered_merge=self.has("--ordered-merge"),       # K1: off too, as linear_deposit (2026-10-11)
                     output_prefix=self.prefix or "ps_output")
            m = re.search(r"(?:^|\s)--ps-alpha-shape\s+([0-9.]+)", self.command)
            if m:
                d["alpha_shape"] = float(m.group(1))
            if self.nsub:
                d["nsub"] = self.nsub
            if self.fields:
                d["fields"] = self.fields
        return d


def custom_settings(rec: RunRecord) -> dict | None:
    """The settings a custom-snapshot run was made with: its '<prefix>.gui.json' sidecar (CustomSpec keys),
    None for a TNG run or a custom run without one."""
    if rec.sim:
        return None
    try:
        d = json.loads(Path(str(rec.path.parent / rec.prefix) + ".gui.json").read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("input_file") else None


INPUT_SUFFIXES = (".hdf5", ".h5", ".hdf")      # snapshots: the binaries read them, never write them


def run_inputs(rec: RunRecord) -> set[Path]:
    """The files a run READ that sit beside its outputs: every existing file named on its command line and
    the settings sidecar's input_file. A custom run named after its input (the default) in the input's own
    folder has 'X.hdf5' beside 'X.a_den': the input must never be one of the run's outputs (the Runs
    browser's Move to Trash takes the outputs)."""
    found = set()
    try:
        words = shlex.split(rec.command)
    except ValueError:
        words = rec.command.split()
    # the log prints the command UNQUOTED: a path with a space (anything under iCloud's 'Mobile Documents') splits
    # above, so the file options' values are also read up to the next ' --', and the sidecar names the rest
    for m in re.finditer(r"--(?:sample-points|lagrangianInput|scalar-file)\s+(.+?)(?=\s--|$)", rec.command):
        words.append(m.group(1).strip())
    side = Path(str(rec.path.parent / rec.prefix) + ".gui.json")
    try:
        sd = json.loads(side.read_text())
        for key in ("input_file", "sample_points", "lagrangian_file"):
            if sd.get(key):
                words.append(str(sd[key]))
    except (OSError, ValueError, AttributeError):
        pass
    for w in words:
        f = Path(w).expanduser()
        f = f if f.is_absolute() else rec.path.parent / f
        try:
            if f.is_file():
                found.add(f.resolve())
        except OSError:
            pass
    return found


# the text-derived fields of every run log read, by path, with the (size, mtime_ns) they were read at
# (2026-10-11): the launcher rescans every 30 s on its GUI thread, and a TNG50 z=0 log is large and on the
# external disk; an unchanged log is not read again
_LOG_FIELDS = ("binary", "build_rev", "build_time", "command", "gpu_retried", "wall_seconds", "peak_rss_gb",
               "footprint_gb", "ok")
_LOG_CACHE: dict[str, tuple[int, int, dict]] = {}


def _parse_text(rec: RunRecord, text: str):
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
        if not rec.binary:
            for name in ("PS-DTFE-2d-double", "DTFE-2d-double", "PS-DTFE-2d", "DTFE-2d",
                         "PS-DTFE-double", "DTFE-double", "PS-DTFE", "DTFE"):
                if exe.endswith(name):
                    rec.binary = name
                    break
    rec.gpu_retried = _GPU_RETRY.search(text) is not None
    m = _WALL.search(text)
    rec.wall_seconds = parse_duration(m.group(1)) if m else None
    m = _RSS.search(text)
    rec.peak_rss_gb = float(m.group(1)) if m else None
    feet = _FOOTPRINT.findall(text)
    rec.footprint_gb = int(feet[-1]) / 2**30 if feet else None     # GiB: the unit of the RSS line and of the run scripts
    # a '.failed' or '.partial' log is an attempt that did not exit 0, whatever it printed (2026-10-08)
    rec.ok = bool(m or rec.wall_seconds is not None) and "~~~ ERROR ~~~" not in text and rec.path.name.endswith(".runlog")


def parse_runlog(path: Path, sim: str = "", snap: int | None = None) -> RunRecord:
    path = Path(path)
    rec = RunRecord(path=path, sim=sim, snap=snap, prefix=log_prefix(path.name))
    try:
        st = path.stat()
        rec.finished = _dt.datetime.fromtimestamp(st.st_mtime)
        hit = _LOG_CACHE.get(str(path))
        if hit is not None and hit[:2] == (st.st_size, st.st_mtime_ns):
            for k, v in hit[2].items():
                setattr(rec, k, v)
        else:
            _parse_text(rec, path.read_text(errors="replace"))
            _LOG_CACHE[str(path)] = (st.st_size, st.st_mtime_ns, {k: getattr(rec, k) for k in _LOG_FIELDS})
    except OSError:
        return rec
    inputs = run_inputs(rec)
    log = rec.prefix + ".runlog"    # by name: '.runlog.failed', '.runlog.partial' (and a script's '.runlog.tmp') too
    rec.outputs = sorted(p for p in path.parent.glob(_glob.escape(rec.prefix) + ".*")       # escaped: '[' in a name
                         if p.is_file() and p.suffix not in (".runlog", ".json") and not p.name.startswith(log)
                         and not is_run_file(p.name)         # C1: the sidecar, the lock, '<grid>.part.<pid>'
                         and p.suffix.lower() not in INPUT_SUFFIXES and p.resolve() not in inputs)
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


def _mtime(p: Path) -> float | None:
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def _live_roots(roots) -> set[str] | None:
    """Of these output roots ('<dir>/<prefix>'), the ones a live process names on its command line: the binary
    (its output argument) or a run script's 'tee <prefix>.runlog.partial'. One ps call; None when ps cannot
    say. (A run started with a relative path names its root differently: it reads as killed.)"""
    try:
        text = subprocess.run(["ps", "-A", "-ww", "-o", "args="], capture_output=True, text=True,
                              timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {r for r in roots if re.search(re.escape(r) + r"(?=[\s.]|$)", text, re.M)}


def _tail(p: Path, nbytes: int = 65536) -> str:
    try:
        with open(p, "rb") as fh:
            fh.seek(max(0, fh.seek(0, 2) - nbytes))
            return fh.read().decode(errors="replace")
    except OSError:
        return ""


def _read_run(logs: dict[str, Path], sim: str = "", snap: int | None = None) -> RunRecord | None:
    """The record of one output set from its logs ({suffix: path}, the run-log contract): '<prefix>.runlog'
    when there is one, the attempt after it ('.runlog.failed' or '.runlog.partial', the newer one) noted on
    it; without one the attempt is the record (2026-10-08: a stopped or failed re-run no longer truncates
    the log of the old grids, so the Runs tab keeps showing them as they are). None: the logs went meanwhile."""
    record = logs.get(".runlog")
    t0 = _mtime(record) if record is not None else None
    attempts = [(t, sfx, p) for sfx, p in logs.items() if sfx != ".runlog"
                for t in [_mtime(p)] if t is not None and (t0 is None or t > t0)]
    if record is None and not attempts:
        return None
    rec = parse_runlog(record if record is not None else max(attempts)[2], sim, snap)
    if attempts:
        t, sfx, p = max(attempts)
        rec.attempt = "failed" if sfx == ".runlog.failed" else "partial"
        rec.attempt_log, rec.attempt_time = p, _dt.datetime.fromtimestamp(t)
        tail = _tail(p)
        rec.attempt_cause = ", ".join(d.title for d in diagnose(tail))
        rec.attempt_resume_refused = RESUME_REFUSED.search(tail) is not None
        if t0 is not None:          # a run's outputs are all written before its log ends: newer ones are the attempt's
            rec.attempt_rewrote = sum(1 for q in rec.outputs if (_mtime(q) or 0.0) > t0 + 5.0)   # 5 s: FAT's 2 s mtimes
        if sfx == ".runlog.partial":
            rec.progress = read_progress(Path(str(p)[: -len(".partial")] + ".progress"))
    return rec


def scan_runs(root, extra_dirs=()) -> list[RunRecord]:
    """Every run under the data root (<family>/<sim>/snapdir_NNN or flat <sim>/snapdir_NNN) and in the
    extra (custom-snapshot output) directories, one per output prefix with any of the three log names
    (_read_run), newest first. A '.runlog.partial' is 'running' when a live process names its root, else
    'killed' (a run killed outright: no exit code reached its end)."""
    out: list[RunRecord] = []
    root = Path(root)
    groups: dict[Path, tuple[dict, str, int | None]] = {}       # '<dir>/<prefix>' -> ({suffix: log}, sim, snap)

    def add(log: Path, sim: str = "", snap: int | None = None):
        sfx = next((s for s in LOG_SUFFIXES if log.name.endswith(s)), None)
        if sfx is None or not log.is_file():                    # '*.runlog*' also finds a script's '.runlog.tmp'
            return
        logs, _, _ = groups.setdefault(log.parent / log_prefix(log.name), ({}, sim, snap))   # found under the data
        logs[sfx] = log                                                                   # root first: its sim stays

    if root.is_dir():
        for pattern in ("*/snapdir_*/*.runlog*", "*/*/snapdir_*/*.runlog*"):
            for log in root.glob(pattern):
                add(log, log.parent.parent.name, _snap_of(log.parent))
    for d in extra_dirs:
        d = Path(d)
        if d.is_dir():
            for log in d.glob("*.runlog*"):
                add(log)
    for logs, sim, snap in groups.values():
        rec = _read_run(logs, sim, snap)
        if rec is None:
            continue
        if sim:
            rec.redshift = _redshift_of(rec.path.parent)
        out.append(rec)
    partial = [r for r in out if r.attempt == "partial"]
    live = _live_roots([str(r.path.parent / r.prefix) for r in partial]) if partial else set()
    for r in partial:
        state = (r.progress or {}).get("state")
        if state in ("running", "died"):            # the sidecar's pid decides, whatever path the run was given
            r.attempt = "running" if state == "running" else "killed"
        elif live is not None:
            r.attempt = "running" if str(r.path.parent / r.prefix) in live else "killed"
    # K2: no process continues it, but the next run can -- when it is a Grids or Pipeline job (run_ps_dtfe.sh passes
    # '--resume' by itself; it writes under the data root's snapdir_*, so the record has a sim; a Custom run never
    # resumes and starts over), and not when the attempt failed BECAUSE the resume was refused (other settings, no
    # manifest): the same job then fails the same way every time (2026-10-11), and only 'Start over' runs it
    for r in out:
        if (r.attempt in ("failed", "killed") and r.checkpoint_dir().is_dir() and r.sim and r.estimator == "ps"
                and not r.attempt_resume_refused):
            r.attempt = "resumable"
    return sorted(out, key=lambda r: r.finished or _dt.datetime.min, reverse=True)


# ---------------------------------------------------------------------- void catalogues
VOID_CACHE_TOOL = Path(__file__).resolve().parent / "void_cache.py"
VOID_STATES = ("cached", "stale", "absent", "error")


def void_candidate(rec: RunRecord) -> bool:
    """A run whose grids the void pipeline reads: a finished TNG 3D full-box run with a density grid."""
    return (bool(rec.sim) and rec.snap is not None and rec.ok and rec.dim == 3 and rec.estimator in ("ps", "dtfe")
            and any(p.name.endswith((".a_den", ".den")) for p in rec.outputs))


def void_status(records, data_root, python: str | None = None, timeout: float = 120.0) -> dict[str, tuple[str, str]]:
    """{str(record.path): (state, why)} of the void catalogue of every void_candidate run (void_cache.py status:
    'cached', 'stale' = built from an older grid or its R_v not today's, 'absent', 'error'), one subprocess per simulation
    (config reads the box for DTFE_SIM). Reads only; seconds for a whole data root, so call it off the GUI thread."""
    import os
    by_sim: dict[str, list[RunRecord]] = {}
    for r in records:
        if void_candidate(r):
            by_sim.setdefault(r.sim, []).append(r)
    out: dict[str, tuple[str, str]] = {}
    for sim, recs in by_sim.items():
        items = [{"sim": sim, "snap": r.snap, "method": r.estimator, "prefix": r.prefix} for r in recs]
        env = dict(os.environ, DTFE_SIM=sim, DTFE_DATA_ROOT=str(data_root), MPLBACKEND="Agg")
        try:
            pr = subprocess.run([python or sys.executable, str(VOID_CACHE_TOOL), "status"], input=json.dumps(items),
                                capture_output=True, text=True, timeout=timeout, env=env)
            got = json.loads(pr.stdout.strip().splitlines()[-1]) if pr.returncode == 0 and pr.stdout.strip() else None
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            got = None
        for k, r in enumerate(recs):
            d = got[k] if isinstance(got, list) and k < len(got) and isinstance(got[k], dict) else \
                {"state": "error", "why": "the status helper failed"}
            out[str(r.path)] = (str(d.get("state", "error")), str(d.get("why", "")))
    return out


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
    Fix("dtfe-gpu-retry", "2026-10-01T15:43", "warning",
        "a standard-DTFE GPU run whose deposit was restarted after the GPU watchdog killed a pass: the "
        "killed pass's atomics could still land after the grids were re-zeroed, a whole extra sample in "
        "sparse cells (up to 2% of the velocity peak; fixed 2026-10-01, and the new kernel never "
        "restarts): regenerate",
        lambda r: r.estimator == "dtfe" and r.gpu and r.gpu_retried),
    Fix("alpha-shape", "2026-09-30T21:15", "warning",
        "a non-periodic run: the flat slivers on the cloud's convex hull (particles a whole face apart) "
        "put spurious mass and streams across the grid, and differed between partitions, so split and "
        "auto-tuned runs disagreed with one triangulation at the partition seams. The Lagrangian domain "
        "is now the cloud's alpha shape (fixed 2026-09-30): regenerate",
        lambda r: r.estimator == "ps" and not r.has("--periodic")),
    Fix("interlace-phase", "2026-10-02T23:30", "error",
        "an interlaced grid: the half-cell phase correction used the raw FFT index, so every Fourier mode "
        "negative along exactly one axis was shifted the wrong way (a band-limited field came back 31% "
        "off; fixed 2026-10-02): regenerate",
        lambda r: r.has("--interlace")),
)


# the launcher's start-over control, as the notes name it (no '--': launcher text; 2026-10-11)
START_OVER_HINT = "'Start over' on the Grids or Pipeline tab (PS_RESUME=0)"


def attempt_note(rec: RunRecord) -> tuple[str, str] | None:
    """(level, message) about the attempt after this run's log (RunRecord.attempt), or None: a 'warning' when
    it rewrote outputs (or the record is a killed or failed attempt itself), else 'info'."""
    if not rec.attempt or rec.attempt_log is None:
        return None
    name = rec.attempt_log.name
    cause = f"; likely cause: {rec.attempt_cause}" if rec.attempt_cause else ""
    when = f"{rec.attempt_time:%Y-%m-%d %H:%M}" if rec.attempt_time else "later"
    prog = progress_text(rec.progress) if rec.attempt == "running" else ""
    if prog:
        prog = f" ({prog})"
    if rec.attempt == "failed" and rec.attempt_resume_refused:     # 2026-10-11: rerunning as is fails again
        return ("warning", f"the last attempt ({name}, {when}) failed because its checkpoint could not be resumed (written "
                           f"with other settings, or incomplete): {START_OVER_HINT} runs the job from the beginning"
                + ("" if rec.path == rec.attempt_log else
                   f"; it rewrote {rec.attempt_rewrote} of the outputs" if rec.attempt_rewrote else
                   "; the outputs are still this log's"))
    if rec.path == rec.attempt_log:                     # no run under this name ever finished
        return {"failed": ("warning", f"the last attempt under this name failed ({name}{cause}); none has finished"),
                "running": ("info", f"still running{prog} (its log so far: {name})"),
                "killed": ("warning", f"killed outright: its log {name} stops mid-run, so the outputs may be "
                                      f"incomplete{cause}"),
                "resumable": ("warning", f"resumable: stopped before the end ({name}{cause}) with its checkpoint "
                                         f"kept; running the same job again continues from the finished partitions "
                                         f"({START_OVER_HINT} runs it from the beginning instead)"),
                }.get(rec.attempt, ("warning", f"killed outright, or still running: its log {name} has no end yet"))
    n = rec.attempt_rewrote
    if rec.attempt == "running":
        return ("info", f"a new run under this name is going{prog} ({name}): "
                + (f"it has rewritten {n} of the outputs so far" if n else "the outputs are this run's until it ends"))
    what = {"failed": f"failed on {when} ({name}{cause})",
            "killed": f"was killed outright on {when} ({name} stops mid-run{cause})",
            "resumable": f"stopped on {when} ({name}{cause}) with its checkpoint kept: running the same job "
                         f"again resumes it ({START_OVER_HINT} starts over)"}.get(
        rec.attempt, f"was killed outright or is still going ({name}, last written {when})")
    return (("warning", f"a later run under this name {what}; it rewrote {n} of the outputs, which no longer "
                        "match this log: regenerate") if n else
            ("info", f"a later run under this name {what}; the outputs are still this run's"))


def stale_reasons(rec: RunRecord) -> list[tuple[str, str]]:
    """(level, message) for every known fix this run predates, plus problems its outputs show and the attempt
    after its log (attempt_note: a failed, killed or running re-run under its name)."""
    out: list[tuple[str, str]] = []
    note = attempt_note(rec)
    if note is not None:
        out.append(note)
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
                  sliced: bool, precision: str = "single", dim: int = 3) -> tuple[float, int] | None:
    """(median seconds per snapshot, number of runs) of successful earlier runs with the same
    simulation, grid, estimator, GPU use, slice/no slice, precision and dimension; None without any
    (a double-precision run takes about twice as long: it must not predict a single one)."""
    times = [r.wall_seconds for r in records
             if r.ok and r.wall_seconds and r.sim == sim and r.grid == grid
             and r.estimator == estimator and r.gpu == gpu and r.sliced == sliced
             and r.precision == precision and r.dim == dim]
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
    if RESUME_REFUSED.search(tail):                 # 2026-10-11: the same job would fail the same way
        out.append(Diagnosis("Resume refused",
                             "A killed run's checkpoint is there, but it cannot be continued (written with other "
                             "settings, or incomplete). Tick 'Start over' on the Grids or Pipeline tab "
                             "(PS_RESUME=0) to run the job from the beginning; it replaces the checkpoint.",
                             "start_over"))
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
