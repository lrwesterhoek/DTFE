"""First-run setup: the four things a new user needs before the launcher is any use.

  1  Data folder      where simulations and grids live (defaults to the configured root when it
                      exists -- an external disk that is unplugged is said so -- else ~/DTFE-data)
  2  Programs         are DTFE / PS-DTFE built, with which GPU backend (Metal, CUDA, HIP) and (PS-DTFE) the TBB parallel
                      triangulation, are the optional double-precision pair and the 2D programs there,
                      and does the Python that runs the figure scripts have its packages? 'Build' runs
                      make (the commands are shown; ticks add the double pair and the 2D programs)
  3  TNG API key      optional, only for downloading IllustrisTNG data: saved to ~/.tng_api_key with
                      owner-only permissions, never shown again or put on a command line
  4  Try it           a synthetic demo (32^3 crossed waves, streams 1/3/9/27): generated and run in
                      about half a minute, then opened in the Explore tab -- no download

Shown on the first launch and from the window's Setup button. Nothing here runs a computation
itself: 'Build' runs make, and the demo is queued as an ordinary custom-snapshot job.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QStackedWidget, QToolButton, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runspec as rs  # noqa: E402

TNG_REGISTER = "https://www.tng-project.org/users/register/"


def binary_status(name: str) -> tuple[bool, str]:
    """(built, one-line description) of ./DTFE, ./PS-DTFE, their -double pair or the -2d programs."""
    exe = rs.REPO_ROOT / name
    precision = "double" if name.endswith("-double") else "single"
    dim = 2 if "-2d" in name else 3
    if not exe.is_file():
        why = " (optional: the 2D programs, for snapshots in a plane)" if dim == 2 else \
            " (optional: the double-precision pair)" if precision == "double" else ""
        return False, f"{name}: not built" + why
    try:
        ver = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        ver = ""
    est = "ps" if name.startswith("PS-DTFE") else "dtfe"
    backend = rs.gpu_backend(est, precision, dim)
    gpu = f"with GPU ({backend})" if backend else "CPU only"
    tbb = ""
    if est == "ps" and dim == 3:    # the parallel Delaunay insertion needs the tbb library at build time (3D only)
        tbb = " · parallel triangulation (TBB)" if rs.tbb_built(precision) else " · no TBB (sequential triangulation only)"
    return True, f"{ver or name} · {gpu}{tbb}"


def python_status() -> tuple[bool, str]:
    try:
        r = subprocess.run([rs.PYTHON, "-c", "import numpy, h5py, matplotlib, scipy"], capture_output=True,
                           text=True, timeout=60)
        ok = r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return ok, (f"{rs.PYTHON}: numpy, h5py, matplotlib, scipy " + ("found" if ok else
                "MISSING (pip install numpy h5py matplotlib scipy)"))


def build_command(double: bool = False, dim: int = 3) -> list[str]:
    """make for both binaries -- with the GPU backend this machine can build (Metal on a Mac, CUDA or
    HIP on Linux when nvcc / hipcc is found), as scripts/install.sh does; double=True builds the
    double-precision pair (DTFE-double, PS-DTFE-double) instead, same backend; dim=2 the 2D programs
    (DTFE-2d, PS-DTFE-2d: CPU only, the GPU kernels are 3D)."""
    cmd = ["make", "DTFE", "PS-DTFE", f"-j{os.cpu_count() or 4}"]
    gpu = rs.gpu_compiler() if dim == 3 else ""
    if gpu:
        cmd.append(gpu)
    if double:
        cmd.append("DOUBLE=1")
    if dim == 2:
        cmd.append("DIM=2")
    return cmd


BINARIES = ("PS-DTFE", "DTFE")
DOUBLE_BINARIES = ("PS-DTFE-double", "DTFE-double")
TWO_D_BINARIES = ("PS-DTFE-2d", "DTFE-2d")


def save_api_key(key: str, path: Path = rs.API_KEY_FILE) -> None:
    """Writes the key with owner-only permissions (created 0600, never world-readable)."""
    key = key.strip()
    if not key:
        raise ValueError("empty key")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key + "\n")
    os.chmod(path, 0o600)


class SetupDialog(QDialog):
    def __init__(self, parent=None, data_root: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Set up the DTFE launcher")
        self.resize(720, 520)
        self.run_demo = False
        self._build: QProcess | None = None
        self._build_queue: list[list[str]] = []

        v = QVBoxLayout(self)
        self.step = QLabel("")
        self.step.setStyleSheet("color: gray")
        v.addWidget(self.step)
        self.pages = QStackedWidget()
        v.addWidget(self.pages, 1)
        nav = QHBoxLayout()
        self.skip = QPushButton("Skip setup")
        self.skip.clicked.connect(self.reject)
        self.back = QPushButton("Back")
        self.back.clicked.connect(lambda: self._go(-1))
        self.next = QPushButton("Next")
        self.next.setDefault(True)
        self.next.clicked.connect(lambda: self._go(+1))
        nav.addWidget(self.skip)
        nav.addStretch(1)
        nav.addWidget(self.back)
        nav.addWidget(self.next)
        v.addLayout(nav)

        # ---- 1 data folder
        p = QWidget()
        pv = QVBoxLayout(p)
        pv.addWidget(self._title("Where should simulations and results live?"))
        pv.addWidget(self._text(
            "The launcher keeps every simulation in its own folder under one data folder "
            "(e.g. <data>/TNG100/TNG100-3-Dark/snapdir_099/…), with the grids next to their snapshots. "
            "Grids are large (a 512³ run with all fields is ~25 GB): pick a disk with room; an external "
            "disk is fine. Your own snapshot files can live anywhere."))
        row = QHBoxLayout()
        self.root = QLineEdit(data_root if data_root and Path(data_root).is_dir() else str(Path.home() / "DTFE-data"))
        self.root.textChanged.connect(self._root_note)
        row.addWidget(self.root, 1)
        b = QToolButton(text="…")
        b.clicked.connect(self._browse)
        row.addWidget(b)
        pv.addLayout(row)
        self.root_info = self._text("")
        pv.addWidget(self.root_info)
        if data_root and not Path(data_root).is_dir():
            pv.addWidget(self._text(f"(The configured folder {data_root} is not there right now. An external "
                                    "disk that is unplugged? Choose it again once it is connected.)"))
        pv.addStretch(1)
        self.pages.addWidget(p)

        # ---- 2 programs
        p = QWidget()
        pv = QVBoxLayout(p)
        pv.addWidget(self._title("The programs"))
        self.prog_info = QLabel("")
        self.prog_info.setWordWrap(True)
        self.prog_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        pv.addWidget(self.prog_info)
        row = QHBoxLayout()
        self.build_btn = QPushButton("Build the programs")
        self.build_btn.clicked.connect(self._start_build)
        row.addWidget(self.build_btn)
        self.build_note = self._text("")
        row.addWidget(self.build_note, 1)
        pv.addLayout(row)
        self.build_double = QCheckBox("Also build the double-precision pair (every position and field in float64; "
                                      "about twice the memory per run; picked per run with the precision setting)")
        self.build_double.toggled.connect(self._build_note)
        pv.addWidget(self.build_double)
        self.build_2d = QCheckBox("Also build the 2D programs (for snapshots in a plane, whose coordinates have two "
                                  "columns; CPU only; picked per run with the dimensions setting)")
        self.build_2d.toggled.connect(self._build_note)
        pv.addWidget(self.build_2d)
        self._build_note()
        self.build_log = QPlainTextEdit(readOnly=True)
        self.build_log.setMaximumBlockCount(4000)
        pv.addWidget(self.build_log, 1)
        self.pages.addWidget(p)

        # ---- 3 API key
        p = QWidget()
        pv = QVBoxLayout(p)
        pv.addWidget(self._title("IllustrisTNG downloads (optional)"))
        pv.addWidget(self._text(
            f"Downloading TNG data needs a free API key ({TNG_REGISTER}; it is shown on your account page). "
            f"It is saved to {rs.API_KEY_FILE}, readable only by you, and never shown or put on a command "
            "line. Skip this if you use your own snapshots."))
        self.key_state = self._text("")
        pv.addWidget(self.key_state)
        row = QHBoxLayout()
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("paste the API key")
        row.addWidget(self.key, 1)
        self.key_btn = QPushButton("Save key")
        self.key_btn.clicked.connect(self._save_key)
        row.addWidget(self.key_btn)
        pv.addLayout(row)
        pv.addStretch(1)
        self.pages.addWidget(p)

        # ---- 4 demo
        p = QWidget()
        pv = QVBoxLayout(p)
        pv.addWidget(self._title("Try it"))
        pv.addWidget(self._text(
            "The demo makes a small synthetic simulation (32³ particles on crossed Zel'dovich waves, "
            "folded into 1, 3, 9 or 27 streams), runs PS-DTFE on it (about half a minute) and opens the "
            "result in the Explore tab, where a click on the map lists the streams at that point. "
            f"It writes to {rs.CUSTOM_OUT_DEFAULT / 'demo'}."))
        self.demo = QCheckBox("Run the demo when I finish")
        self.demo.setChecked(True)
        pv.addWidget(self.demo)
        pv.addStretch(1)
        self.pages.addWidget(p)

        self._root_note()
        self._go(0)

    @staticmethod
    def _title(text):
        lab = QLabel(f"<h3>{text}</h3>")
        return lab

    @staticmethod
    def _text(text):
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return lab

    # ---------------------------------------------------------------- pages
    def _go(self, delta: int):
        i = self.pages.currentIndex() + delta
        if i >= self.pages.count():
            self._finish()
            return
        i = max(0, i)
        self.pages.setCurrentIndex(i)
        self.step.setText(f"Step {i + 1} of {self.pages.count()}")
        self.back.setEnabled(i > 0)
        self.next.setText("Finish" if i == self.pages.count() - 1 else "Next")
        if i == 1:
            self._programs()
        if i == 2:
            self.key_state.setText("A key is already saved." if rs.api_key_available() else "No key saved yet.")

    def _root_note(self):
        root = Path(self.root.text().strip()).expanduser()
        if root.is_dir():
            sims = rs.simulations(root)
            try:
                free = shutil.disk_usage(root).free / 1e9
            except OSError:
                free = 0
            self.root_info.setText(f"found: {len(sims)} simulation{'s' if len(sims) != 1 else ''}"
                                   + (f" ({', '.join(sims[:5])}{'…' if len(sims) > 5 else ''})" if sims else "")
                                   + f" · {free:.0f} GB free")
        else:
            self.root_info.setText("does not exist yet: it will be created")
        if "Mobile Documents" in str(root):
            self.root_info.setText(self.root_info.text() + " · ⚠ this is iCloud Drive: grids of tens of GB "
                                   "would be synced, so a local or external disk is better")

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Data folder", self.root.text())
        if d:
            self.root.setText(d)

    def _build_note(self, *_):
        cmds = [" ".join(c) for c in self._build_commands()]
        self.build_note.setText("runs: " + "; then ".join(cmds) + "   (a few minutes each; needs CGAL, Boost, "
                                "FFTW, HDF5, GSL; scripts/install.sh installs them)")

    def _programs(self):
        lines = []
        for name in BINARIES:
            ok, text = binary_status(name)
            lines.append(("✔ " if ok else "✖ ") + text)
        for name in DOUBLE_BINARIES + TWO_D_BINARIES:
            ok, text = binary_status(name)
            lines.append(("✔ " if ok else "○ ") + text)
        ok, text = python_status()
        lines.append(("✔ " if ok else "⚠ ") + text)
        lines.append(("✔ wget found" if shutil.which("wget") else "⚠ wget not found (only needed for TNG downloads: "
                                                                 "brew install wget)"))
        self.prog_info.setText("\n".join(lines))
        both = all(binary_status(n)[0] for n in BINARIES)
        self.build_btn.setText("Rebuild the programs" if both else "Build the programs")
        if all(binary_status(n)[0] for n in DOUBLE_BINARIES) and not self.build_double.isChecked():
            self.build_double.setChecked(True)      # a pair that exists is kept up to date
        if all(binary_status(n)[0] for n in TWO_D_BINARIES) and not self.build_2d.isChecked():
            self.build_2d.setChecked(True)

    def _build_commands(self) -> list[list[str]]:
        return ([build_command()] + ([build_command(double=True)] if self.build_double.isChecked() else [])
                + ([build_command(dim=2)] if self.build_2d.isChecked() else []))

    def _start_build(self):
        if self._build is not None:
            return
        self._build_queue = self._build_commands()
        self._run_next_build()

    def _run_next_build(self):
        if not self._build_queue:
            self._build = None
            self.build_btn.setEnabled(True)
            self._programs()
            return
        cmd = self._build_queue.pop(0)
        self.build_log.appendPlainText("$ " + " ".join(cmd))
        self._build = QProcess(self)
        self._build.setWorkingDirectory(str(rs.REPO_ROOT))
        self._build.setProcessChannelMode(QProcess.MergedChannels)
        self._build.readyReadStandardOutput.connect(
            lambda: self.build_log.appendPlainText(
                bytes(self._build.readAllStandardOutput()).decode(errors="replace").rstrip()))
        self._build.finished.connect(self._build_done)
        self._build.errorOccurred.connect(self._build_error)
        self.build_btn.setEnabled(False)
        self._build.start(cmd[0], cmd[1:])

    def _build_error(self, err):
        """'finished' never comes for a program that cannot start (no make on a fresh machine): say so and
        free the button (before 2026-10-06 it stayed disabled for good)."""
        if err != QProcess.FailedToStart or self._build is None:
            return
        self.build_log.appendPlainText(f"cannot start {self._build.program()}: {self._build.errorString()} "
                                       "(install the Xcode command-line tools: xcode-select --install)")
        self._build_queue = []
        self._build = None
        self.build_btn.setEnabled(True)
        self._programs()

    def _build_done(self, code, _status):
        self.build_log.appendPlainText(f"make finished with exit code {code}")
        if code != 0:
            self._build_queue = []
        self._run_next_build()

    def _save_key(self):
        try:
            save_api_key(self.key.text())
        except (OSError, ValueError) as e:
            self.key_state.setText(f"not saved: {e}")
            return
        self.key.clear()
        self.key_state.setText(f"Saved to {rs.API_KEY_FILE} (readable only by you).")

    def data_root(self) -> str:
        return str(Path(self.root.text().strip()).expanduser())

    def _finish(self):
        root = Path(self.data_root())
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.pages.setCurrentIndex(0)
            self.root_info.setText(f"cannot create it: {e}")
            return
        self.run_demo = self.demo.isChecked()
        self.accept()

    def reject(self):
        if self._build is not None:
            self._build.kill()
            self._build.waitForFinished(3000)
        super().reject()
