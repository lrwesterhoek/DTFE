"""First-run setup: the four things a new user needs before the launcher is any use.

  1  Data folder      where simulations and grids live (defaults to the configured root when it
                      exists -- an external disk that is unplugged is said so -- else ~/DTFE-data)
  2  Programs         are DTFE / PS-DTFE built, with GPU support, and does the Python that runs the
                      figure scripts have its packages? 'Build' runs make (the command is shown)
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
    """(built, one-line description) of ./DTFE or ./PS-DTFE."""
    exe = rs.REPO_ROOT / name
    if not exe.is_file():
        return False, f"{name}: not built"
    try:
        ver = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        ver = ""
    est = "ps" if name == "PS-DTFE" else "dtfe"
    gpu = "with GPU (Metal)" if rs.gpu_built(est) else "CPU only"
    return True, f"{ver or name} · {gpu}"


def python_status() -> tuple[bool, str]:
    try:
        r = subprocess.run([rs.PYTHON, "-c", "import numpy, h5py, matplotlib, scipy"], capture_output=True,
                           text=True, timeout=60)
        ok = r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return ok, (f"{rs.PYTHON}: numpy, h5py, matplotlib, scipy " + ("found" if ok else
                "MISSING (pip install numpy h5py matplotlib scipy)"))


def build_command() -> list[str]:
    """make for both binaries -- with the GPU on a Mac (Metal), as scripts/install.sh does."""
    cmd = ["make", "DTFE", "PS-DTFE", f"-j{os.cpu_count() or 4}"]
    if sys.platform == "darwin":
        cmd.append("METAL=1")
    return cmd


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
        row.addWidget(self._text("runs: " + " ".join(build_command()) + "   (a few minutes; needs CGAL, Boost, "
                                 "FFTW, HDF5, GSL; scripts/install.sh installs them)"), 1)
        pv.addLayout(row)
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

    def _programs(self):
        lines = []
        for name in ("PS-DTFE", "DTFE"):
            ok, text = binary_status(name)
            lines.append(("✔ " if ok else "✖ ") + text)
        ok, text = python_status()
        lines.append(("✔ " if ok else "⚠ ") + text)
        lines.append(("✔ wget found" if shutil.which("wget") else "⚠ wget not found (only needed for TNG downloads: "
                                                                 "brew install wget)"))
        self.prog_info.setText("\n".join(lines))
        both = all(binary_status(n)[0] for n in ("PS-DTFE", "DTFE"))
        self.build_btn.setText("Rebuild the programs" if both else "Build the programs")

    def _start_build(self):
        if self._build is not None:
            return
        cmd = build_command()
        self.build_log.appendPlainText("$ " + " ".join(cmd))
        self._build = QProcess(self)
        self._build.setWorkingDirectory(str(rs.REPO_ROOT))
        self._build.setProcessChannelMode(QProcess.MergedChannels)
        self._build.readyReadStandardOutput.connect(
            lambda: self.build_log.appendPlainText(
                bytes(self._build.readAllStandardOutput()).decode(errors="replace").rstrip()))
        self._build.finished.connect(self._build_done)
        self.build_btn.setEnabled(False)
        self._build.start(cmd[0], cmd[1:])

    def _build_done(self, code, _status):
        self.build_log.appendPlainText(f"make finished with exit code {code}")
        self._build = None
        self.build_btn.setEnabled(True)
        self._programs()

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
