"""MainWindow's run panel, runs and queue (RunMixin, moved from app.py): the panel beside the tabs (commands,
checks, Run / Add to queue / Stop, progress and the clock, the failure banner, the Log, Queue, Figures, Runs
and Running now tabs, find in the log), running a job's steps with live progress and a run log beside the
outputs, stopping it, and the job queue (adding, ordering, waiting for other DTFE runs, saving it). The names
app.py defines are read through _app, imported last (see app_tabs.py)."""

from __future__ import annotations

import json
import os
from collections import Counter
import signal
import subprocess
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QRegularExpression, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QTextCursor, QTextDocument
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPlainTextEdit, QProgressBar, QPushButton, QTabWidget, QToolButton,
                               QVBoxLayout, QWidget)

import runspec as rs
import results as R
import grids as G
from app_widgets import _mono, _note, FigureBrowser


class RunMixin:
    # ================================================================ run panel
    def _build_run_panel(self) -> QWidget:
        v = QVBoxLayout()
        # the exact commands a run executes: hidden unless asked for (View > Show Commands)
        self.cmd_box = QWidget()
        cb = QVBoxLayout(self.cmd_box)
        cb.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        self.cmd_title = QLabel("")
        self.cmd_title.setStyleSheet("font-weight: bold")
        head.addWidget(self.cmd_title, 1)
        copy = QPushButton("Copy")
        copy.setToolTip("Copy the commands, to run them in a terminal")
        copy.clicked.connect(self.copy_commands)
        head.addWidget(copy)
        self.export_btn = QPushButton("Export…")
        self.export_btn.setToolTip("Save the commands as a bash script, or as a SLURM job for a cluster")
        self.export_btn.clicked.connect(self.export_job)
        head.addWidget(self.export_btn)
        self.cmd_hide = QPushButton("Hide")
        self.cmd_hide.clicked.connect(lambda: self._toggle_commands(False))
        head.addWidget(self.cmd_hide)
        cb.addLayout(head)
        self.cmd_view = QPlainTextEdit(readOnly=True)
        self.cmd_view.setFont(_mono())
        self.cmd_view.setMaximumHeight(130)
        cb.addWidget(self.cmd_view)
        self.cmd_box.setVisible(self.show_commands)
        v.addWidget(self.cmd_box)

        v.addWidget(QLabel("<b>Checks</b>"))
        self.checks = QListWidget()
        self.checks.setMaximumHeight(110)
        self.checks.setWordWrap(True)
        v.addWidget(self.checks)
        self.estimate = _note()
        v.addWidget(self.estimate)

        run = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self.start_run)
        self.queue_btn = QPushButton("Add to queue")
        self.queue_btn.clicked.connect(self.enqueue)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_run)
        self.status = QLabel("Idle")
        run.addWidget(self.run_btn)
        run.addWidget(self.queue_btn)
        self.mem_btn = QPushButton("Check memory")
        self.mem_btn.setToolTip("Ask the binary's auto-tuner what this run needs per snapshot (and how it "
                                "would split the work), against this machine's memory budget: it reads each "
                                "snapshot and computes nothing")
        self.mem_btn.clicked.connect(self.check_memory)
        run.addWidget(self.mem_btn)
        run.addWidget(self.stop_btn)
        run.addWidget(self.status, 1)
        v.addLayout(run)
        prog = QHBoxLayout()
        self.snap_label = QLabel("")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.bar.setFormat("")
        self.eta = QLabel("")
        self.clock = QLabel("")             # the job's wall clock: 'mm:ss elapsed', then 'took 12m 40s'
        self.clock.setStyleSheet("color: gray")
        self._clock_timer = QTimer(self, interval=1000)
        self._clock_timer.timeout.connect(self._tick_clock)
        prog.addWidget(self.snap_label)
        prog.addWidget(self.bar, 1)
        prog.addWidget(self.eta)
        prog.addWidget(self.clock)
        v.addLayout(prog)

        # after a failed job: what probably went wrong, and a button that fixes it where one can
        self.banner = QFrame()
        self.banner.setFrameShape(QFrame.StyledPanel)
        self.banner.setStyleSheet("QFrame { background: rgba(198, 40, 40, 0.08); }")
        bl = QVBoxLayout(self.banner)
        self.banner_text = QLabel("")
        self.banner_text.setWordWrap(True)
        self.banner_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bl.addWidget(self.banner_text)
        self.banner_buttons = QHBoxLayout()
        bl.addLayout(self.banner_buttons)
        self.banner.hide()
        v.addWidget(self.banner)

        self.results = QTabWidget()
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setFont(_mono())
        self.log.setMaximumBlockCount(20000)
        self.results.addTab(self.log, "Log")
        self.queue_tab = self._build_queue_tab()
        self.results.addTab(self.queue_tab, "Queue")
        self.figures = FigureBrowser()
        self.figures.changed.connect(lambda: self._sync_figures_actions())
        self.figures.mode.currentIndexChanged.connect(lambda _i: self._sync_figures_actions())
        self.results.addTab(self.figures, "Figures")
        self.results.addTab(self.runs, "Runs")
        self.results.addTab(self.running_now, "Running now")
        self.results.setTabToolTip(self.results.indexOf(self.running_now),
                                   "Every DTFE / PS-DTFE run on this machine, wherever it was started (a terminal, "
                                   "nohup, this launcher): progress, ETA, memory")
        self.results.setTabToolTip(self.results.indexOf(self.runs),
                                   "Every run on disk: settings, time, memory, size, and outputs made before a fix")
        self.results.currentChanged.connect(self._results_tab_changed)
        self.results.setCornerWidget(self._build_log_find(), Qt.TopRightCorner)
        v.addWidget(self.results, 1)
        w = QWidget()
        w.setLayout(v)
        return w

    def _focus_log_find(self):
        if self.right.currentIndex() != 0:          # Explore shows its own pane: back to the run panel
            self.tabs.setCurrentIndex(self._panel_tab)
        self.results.setCurrentIndex(self.results.indexOf(self.log))
        self.log_find.setFocus()
        self.log_find.selectAll()

    def _build_log_find(self) -> QWidget:
        """Find in the log (survey item 16): a field in the result tabs' corner, shown with the Log tab;
        Enter / the arrows step through the matches (wrapping), '!' to the next error, warning or FAILED."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 4, 0)
        h.setSpacing(2)
        self.log_find = QLineEdit(placeholderText="Find in the log")
        self.log_find.setClearButtonEnabled(True)
        self.log_find.setMaximumWidth(220)
        self.log_find.returnPressed.connect(lambda: self.find_in_log(self.log_find.text()))
        self.log_find.textChanged.connect(lambda _t: self.log_find_count.setText(""))
        h.addWidget(self.log_find)
        for text, tip, fn in (("▲", "Previous match", lambda: self.find_in_log(self.log_find.text(), back=True)),
                              ("▼", "Next match (Enter)", lambda: self.find_in_log(self.log_find.text())),
                              ("!", "Next error, warning or failure", lambda: self.find_in_log(_app.PROBLEM_LINES, regex=True))):
            b = QToolButton(text=text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            h.addWidget(b)
        self.log_find_count = QLabel("")
        self.log_find_count.setStyleSheet("color: gray")
        h.addWidget(self.log_find_count)
        return w

    def find_in_log(self, text: str, back: bool = False, regex: bool = False) -> bool:
        """Select the next (previous) match after (before) the cursor, wrapping at the end; the count of
        matches beside the field. False when there is none."""
        if not text:
            return False
        self.results.setCurrentIndex(self.results.indexOf(self.log))
        pattern = QRegularExpression(text if regex else QRegularExpression.escape(text),
                                     QRegularExpression.MultilineOption
                                     | QRegularExpression.CaseInsensitiveOption)
        if not pattern.match(self.log.toPlainText()).hasMatch():
            self.log_find_count.setText("no match")     # before the cursor moves: the log keeps following the run
            return False
        flags = QTextDocument.FindBackward if back else QTextDocument.FindFlag(0)
        found = self.log.find(pattern, flags)
        if not found:                       # wrap: from the other end
            c = self.log.textCursor()
            c.movePosition(QTextCursor.End if back else QTextCursor.Start)
            self.log.setTextCursor(c)
            found = self.log.find(pattern, flags)
        doc = self.log.toPlainText()
        it = pattern.globalMatch(doc)
        starts = []
        while it.hasNext():
            starts.append(it.next().capturedStart())
        if found:
            k = sum(1 for s_ in starts if s_ < self.log.textCursor().selectionStart()) + 1
            self.log_find_count.setText(f"{k} of {len(starts)}")
            self.log.centerCursor()
        else:
            self.log_find_count.setText("no match")
        return found

    def copy_log(self):
        QApplication.clipboard().setText(self.log.toPlainText())
        self.status_message("the log is on the clipboard")

    def open_run_log(self):
        """The newest run's full log file (the Log tab keeps the last 20000 lines)."""
        if self._last_log_path and Path(self._last_log_path).is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_log_path))

    def _sync_log_actions(self):
        if hasattr(self, "open_log_action"):
            self.open_log_action.setEnabled(bool(self._last_log_path) and Path(self._last_log_path).is_file())

    def _build_queue_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.q_list = QListWidget()
        self.q_list.currentRowChanged.connect(self._show_queued)
        v.addWidget(self.q_list, 2)
        btns = QHBoxLayout()
        self.q_start = QPushButton("Start queue")
        self.q_start.clicked.connect(self.start_queue)
        self.q_up, self.q_down = QPushButton("Up"), QPushButton("Down")
        self.q_up.clicked.connect(lambda: self._move_queued(-1))
        self.q_down.clicked.connect(lambda: self._move_queued(+1))
        self.q_remove, self.q_clear = QPushButton("Remove"), QPushButton("Clear finished")
        self.q_remove.clicked.connect(self._remove_queued)
        self.q_clear.clicked.connect(self._clear_finished)
        for b in (self.q_start, self.q_up, self.q_down, self.q_remove, self.q_clear):
            btns.addWidget(b)
        btns.addStretch(1)
        v.addLayout(btns)
        opts = self._restore_json("queue_options", {})
        self.q_stop_on_fail = QCheckBox("Stop the queue when a job fails (later jobs usually need its output)")
        self.q_stop_on_fail.setChecked(opts.get("stop_on_fail", True))
        self.q_wait = QCheckBox("Before a grids, pipeline or custom-snapshot job, wait while another DTFE / PS-DTFE "
                                "run is going")
        self.q_wait.setChecked(opts.get("wait", True))
        self.q_notify = QCheckBox("Notify me when a job or the queue finishes (a desktop notification)")
        self.q_notify.setChecked(opts.get("notify", True))
        for cb in (self.q_stop_on_fail, self.q_wait, self.q_notify):
            cb.toggled.connect(self._save_queue)
            v.addWidget(cb)
        for cb, action in ((self.q_stop_on_fail, self.q_stop_on_fail_action), (self.q_wait, self.q_wait_action),
                           (self.q_notify, self.q_notify_action)):      # the Queue menu's copies
            action.setChecked(cb.isChecked())
            action.toggled.connect(cb.setChecked)
            cb.toggled.connect(action.setChecked)
        self.q_detail = QPlainTextEdit(readOnly=True)
        self.q_detail.setFont(_mono())
        v.addWidget(self.q_detail, 1)
        return w

    # ================================================================ running
    def start_run(self):
        self._read_widgets()
        job = self._job()
        if any(level == "error" for level, _ in job.problems()):
            return
        steps = job.steps()
        if not steps or self.proc is not None:
            return
        self._figs_since = time.time()
        self._begin(steps, self.tabs.currentIndex(), job, None)

    def _begin(self, steps: list[rs.Step], tab: int, spec, qjob: rs.QueuedJob | None):
        """Run one job's steps: from the Run button (qjob None) or as the queue's next job."""
        self._steps, self._step_i, self._results = steps, -1, []
        self._job_text = ""
        self._close_tee()                   # a handle left by a step that never finished must not take this job's output
        self._tee_path = ""                 # ... nor its log be renamed at this job's first step end: it stays .partial
        G.drop_cubes()                      # Explore's cubes are this process's memory: the auto-tuner budgets from what is free
        G.cap_cubes(True)                   # ... and they stay small while the job runs (lifted in _finish_job)
        if tab in (_app.GRIDS, _app.OWN, _app.PIPELINE):    # a void-catalogue search costs ~80 B per cell: a heavy job gets the memory
            self.explore.stop_void_search()  # (the next tick or output change asks again)
        self.banner.hide()
        if isinstance(spec, rs.CustomSpec):
            spec = rs.CustomSpec.from_dict(spec.to_dict())    # a COPY: the Custom tab keeps editing the live one while
        self._job_tab, self._job_spec, self._current_job = tab, spec, qjob      # the job runs, and the finish must describe THIS run
        self._job_start = time.time()
        self._job_t0 = self._step_t0 = time.monotonic()
        self._snap_t0s = []
        self._queue_est = None
        self._per_snap_prior = self._job_estimate(spec, tab, per_snapshot=True)
        self._clock_timer.start()
        self._job_sim = getattr(spec, "sim", "")
        self._buffer, self._summary, self._stopping = "", "", False
        self._snaps_started = self._snaps_ok = 0
        self._dl_files = self._dl_bytes = self._problems = 0
        self._total_snaps = len(spec.snapshots) if tab == _app.GRIDS else 0
        self._total_known = tab == _app.GRIDS
        if tab == _app.PIPELINE and isinstance(spec, rs.PipelineSpec):     # every simulation's snapshots, for the ETA: the
            try:                                                       # script announces each one when it starts
                self._total_snaps = sum(len(spec.stale(sim)[0]) for sim in spec.sims
                                        if rs.sim_dir(sim, spec.data_root).is_dir())
                self._total_known = self._total_snaps > 0
            except (OSError, ValueError):
                self._total_snaps = 0
        if qjob is None:
            self.log.clear()
        else:
            k = self.queue.index(qjob) + 1
            self.log.appendPlainText(("\n" if self.log.blockCount() > 1 else "")
                                     + f"{'=' * 72}\nQueue job {k}/{len(self.queue)}: {qjob.title}\n{'=' * 72}")
        if isinstance(spec, rs.CustomSpec):
            self._register_custom_output(spec)                 # AFTER the log is cleared: its "!! cannot write" line must survive
        self.results.setCurrentIndex(self.results.indexOf(self.log))
        self.snap_label.setText("")
        self.eta.setText("")
        self._tick_clock()
        self._next_step()

    def _tick_clock(self):
        """Every second while a job runs: its elapsed time, the step's when there are several, and what is
        left: of the job (its snapshots) and of the queue (the waiting jobs that have an estimate)."""
        if self._job_t0 is None:
            return
        now = time.monotonic()
        text = f"{_app.clock_text(now - self._job_t0)} elapsed"
        if len(self._steps) > 1 and self._step_t0 is not None and 0 <= self._step_i < len(self._steps):
            text += f" (step {self._step_i + 1}: {_app.clock_text(now - self._step_t0)})"
        left = self._job_left(now)
        if left is not None:
            text += f" · ~{_app.took_text(left)} left (done ~{time.strftime('%H:%M', time.localtime(time.time() + left))})"
        if self._queue_running:
            if self._queue_est is None or now - self._queue_est[0] > 60:     # it reads the run logs: not every tick
                self._queue_est = (now, *self._queue_left())
            _, q, unknown = self._queue_est
            if q or unknown:
                total = (left or 0.0) + q
                text += f" · queue ~{_app.took_text(total)}" + (f" + {unknown} job{'s' if unknown > 1 else ''} without an "
                                                          "estimate" if unknown else "")
        self.clock.setText(text)

    def _job_left(self, now: float) -> float | None:
        """Seconds left of this job: the snapshots still to go at the pace of the ones done (earlier runs' pace
        until one is), minus the time the current one has already run. None when the job has no snapshot count."""
        total, started = self._total_snaps, len(self._snap_t0s)
        if not total or not started:
            return None
        per = (self._snap_t0s[-1] - self._snap_t0s[0]) / (started - 1) if started >= 2 else self._per_snap_prior
        if per is None:
            return None
        return per * max(0, total - started) + max(0.0, per - (now - self._snap_t0s[-1]))

    def _job_estimate(self, spec, kind_or_tab, per_snapshot: bool = False) -> float | None:
        """Seconds a Grids job will take (per snapshot, or all of it) from earlier runs of the same settings;
        None for other jobs or without earlier runs."""
        if not isinstance(spec, rs.RunSpec) or kind_or_tab not in (_app.GRIDS, "grids"):
            return None
        t = R.time_estimate(self._runs(), sim=spec.sim, grid=spec.grid, estimator=spec.estimator,
                            gpu=spec.gpu and rs.gpu_built(spec.estimator),
                            sliced=bool(spec.slice_plane), precision=spec.precision)
        if t is None:
            return None
        return t[0] if per_snapshot else t[0] * max(1, len(spec.snapshots))

    def _queue_left(self) -> tuple[float, int]:
        """(seconds of the waiting jobs that have an estimate, how many have none)."""
        total, unknown = 0.0, 0
        for job in self.queue:
            if job.state != "waiting":
                continue
            try:
                est = self._job_estimate(job.build(), job.kind)
            except (TypeError, ValueError):
                est = None
            if est is None:
                unknown += 1
            else:
                total += est
        return total, unknown

    def _next_step(self):
        self._step_i += 1
        if self._stopping or self._step_i >= len(self._steps):
            self._finish_job()
            return
        st, n = self._steps[self._step_i], len(self._steps)
        if st.after and any(lbl == st.after and code != 0 for lbl, code in self._results):
            # a merge after a failed download (half a tree set would pass), a render after a pipeline that computed
            # nothing: the step depends on the one before it
            self.log.appendPlainText(f"\n!! {st.label}: skipped, '{st.after}' failed (run the job again)")
            self._results.append((st.label, 1))
            self._next_step()
            return
        self.status.setText(f"Step {self._step_i + 1}/{n}: {st.label}" if n > 1 else f"Running {st.label}")
        self._step_t0 = time.monotonic()
        self.log.appendPlainText(("\n" if self._step_i else "") + "$ " + st.line() + "\n")
        if st.tee:                          # '... 2>&1 | tee <file>': the run log beside the outputs
            # written as '<file>.partial' and put in place when the step ends (_close_tee): else a stopped or
            # failed re-run under the same name would truncate the log of the old grids, which stay on disk, and the
            # Runs tab would show them as failed (the run-log contract, as the run scripts keep it)
            part = str(st.tee) + ".partial"
            try:
                self._tee = open(part, "w")
                self._tee_path = str(st.tee)
                self._last_log_path = part
                self._sync_log_actions()
            except OSError as e:
                self._tee = None
                self.log.appendPlainText(f"!! cannot write the run log {part}: {e}")
        self.bar.setRange(0, 0)             # busy until a partition progress line arrives
        self.eta.setText("")
        penv = QProcessEnvironment.systemEnvironment()
        # a pipe makes Python block-buffer print(): a plot, analysis or merge step would show nothing (no log, no
        # problem lines, no merge chunk bar) until it exits; harmless for the binaries
        penv.insert("PYTHONUNBUFFERED", "1")
        for k, v in st.env.items():
            penv.insert(k, v)
        self.proc = QProcess(self)
        self.proc.setProcessEnvironment(penv)
        self.proc.setWorkingDirectory(str(rs.REPO_ROOT))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        proc = self.proc
        self.proc.started.connect(lambda: self._keep_awake(st.argv, int(proc.processId())))
        self.proc.start(st.argv[0], st.argv[1:])
        self.refresh()

    def _keep_awake(self, argv, pid: int):
        """caffeinate for a step no run script keeps awake (rs.keep_awake_argv: Custom runs, analysis, downloads).
        It follows the step's pid (-w), so it ends with the step, a Stop or a crash included."""
        cmd = rs.keep_awake_argv(argv, pid)
        if cmd is None:
            return
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError:
            pass

    def _on_output(self):
        if self.proc is None:
            return
        raw = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        text = rs.strip_ansi(raw)
        if self._tee is not None:
            try:
                self._tee.write(text)
            except OSError as e:            # the data disk vanished or is full: the run goes on without its copy
                self.log.appendPlainText(f"!! the run log cannot be written any more ({e}); continuing without it")
                self._close_tee()
        self._job_text = (self._job_text + text)[-400_000:]
        self._buffer += text.replace("\r", "\n")
        *lines, self._buffer = self._buffer.split("\n")
        self._consume(lines)

    def _consume(self, lines: list[str]):
        """Progress into the bar and labels; everything else (minus progress fragments) into the log."""
        keep = []
        for line in lines:
            info = rs.parse_progress(line)
            if "planned" in info and not self._total_known:     # the pipeline announcing its snapshots
                self._total_snaps += info["planned"]
                self.snap_label.setText(f"snapshot {self._snaps_started}/{self._total_snaps}")
            if "sim" in info:
                self.status.setText(f"{self._steps[self._step_i].label}: {info['sim']}")
            if "snapshot" in info:
                self._snaps_started += 1
                self._snap_t0s.append(time.monotonic())
                if self._total_snaps:
                    self.snap_label.setText(f"snapshot {self._snaps_started}/{self._total_snaps}")
                sim = f"{self._job_sim} " if self._job_tab == _app.GRIDS else ""
                self.status.setText(f"Running {sim}snapshot {info['snapshot']:03d}")
                self.bar.setRange(0, 0)
                self.eta.setText("")
            if "total" in info:
                self.bar.setFormat("partitions %v/%m")
                self.bar.setRange(0, info["total"])
                self.bar.setValue(info["done"])
                self.eta.setText(f"{info['elapsed']} elapsed"
                                 + ("" if info["eta"] == "done" else f", ETA ~{info['eta']}"))
            for _path, nbytes in info.get("downloaded", []):
                self._dl_files += 1
                self._dl_bytes += nbytes
                self.snap_label.setText(f"{self._dl_files} file{'s' if self._dl_files != 1 else ''}, "
                                        f"{self._dl_bytes / 1e9:.2f} GB downloaded")
            if "chunk" in info:
                self.bar.setFormat("chunks %v/%m")
                self.bar.setRange(0, info["chunk"][1])
                self.bar.setValue(info["chunk"][0])
            self._problems += info.get("problem", 0)
            if info.get("snapshot_finished"):
                self._snaps_ok += 1
            if "processing complete:" in line:
                self._summary = line.split(":", 1)[1].strip()
            line = rs.PROGRESS.sub("", line)
            if line.strip():
                keep.append(line)
        if keep:
            self.log.appendPlainText("\n".join(keep))

    def _on_error(self, err):
        if err == QProcess.FailedToStart and self.proc is not None:
            st = self._steps[self._step_i]
            self.log.appendPlainText(f"!! could not start {st.argv[0]}: {self.proc.errorString()}")
            self.proc = None
            self._close_tee(127)            # 'finished' never comes: the run log must not stay open into the next job
            self._results.append((st.label, 127))
            QTimer.singleShot(0, self._next_step)

    def _on_finished(self, code, status):
        if self.proc is None:
            return
        if self._buffer.strip():
            self._consume([self._buffer])
        self._buffer = ""
        st = self._steps[self._step_i]
        code = code if status == QProcess.NormalExit else -1
        self._close_tee(code)               # never raises: the job must reach _next_step even when the disk is gone
        self._results.append((st.label, code))
        if code != 0 and not self._stopping:
            self.log.appendPlainText(f"!! {st.label}: exit {code}")
        self.proc = None
        self._next_step()

    def _close_tee(self, code: int | None = None):
        """Close the run-log copy. With the step's exit code it is also put in place (the run-log contract):
        '<log>.partial' becomes '<log>' on exit 0 (and a stale '<log>.failed' goes), else
        '<log>.failed', leaving the log of the old grids alone. Without one (a write error mid-run) only the
        handle closes; the step's end still renames it. Never raises."""
        tee, self._tee = self._tee, None
        if tee is not None:
            try:
                tee.close()
            except OSError as e:
                self.log.appendPlainText(f"!! the run log could not be closed ({e})")
        if code is None or not self._tee_path:
            return
        final, self._tee_path = self._tee_path, ""
        dest = final if code == 0 else final + ".failed"
        try:
            os.replace(final + ".partial", dest)
            if code == 0:
                Path(final + ".failed").unlink(missing_ok=True)     # the grids on disk are this run's now
            self._last_log_path = dest
        except OSError as e:
            self.log.appendPlainText(f"!! the run log could not be renamed to {Path(dest).name} ({e})")
        self._sync_log_actions()

    def _finish_job(self):
        n = len(self._steps)
        ok = sum(1 for _, c in self._results if c == 0)
        self._clock_timer.stop()
        G.cap_cubes(False)
        took = _app.took_text(time.monotonic() - self._job_t0) if self._job_t0 is not None else ""
        self._job_t0 = self._step_t0 = None
        self.clock.setText(f"took {took}" if took else "")
        if self._stopping:
            self.status.setText("Stopped")
        elif ok == n and self._problems:
            self.status.setText(f"Finished, but {self._problems} line{'s' if self._problems > 1 else ''} "
                                "in the log report a problem")
        elif ok == n:
            self.status.setText("Finished")
        else:
            self.status.setText(f"Finished: {n - ok} of {n} steps failed")
        self.snap_label.setText(self._summary or (f"{ok}/{n} steps OK" if n > 1 else ""))
        self.bar.setRange(0, max(n, 1))
        self.bar.setValue(ok)
        self.bar.setFormat("%v/%m steps OK" if n > 1 else "partitions %v/%m")
        if n == 1 and ok == 1:
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
            self.bar.setFormat("done")
        if self._job_tab == _app.DATA:
            self._fill_sims()               # a newly merged simulation appears everywhere
        self._fill_sim_dependent()          # new outputs, slices and planes
        self._runs_scanned = 0.0            # new run logs: rescan for the estimates and the Runs tab
        if self.results.currentWidget() is self.runs:
            self.runs.reload()
        if isinstance(self._job_spec, rs.CustomSpec) and ok == n:
            self._register_custom_output(self._job_spec, final=True)     # again: a generated input exists only now --
        if self._job_tab in (_app.GRIDS, _app.OWN, _app.PIPELINE) and self.explore.outputs:    # BEFORE the refresh reads the sidecars
            self.explore.refresh_outputs()
        failed_codes = [c for _, c in self._results if c != 0]
        if failed_codes and not self._stopping:
            self._show_banner(R.diagnose(self._job_text, failed_codes))
        extra = [self.plots.figures_root, self.pipe.figures_root, getattr(self._job_spec, "figures_root", "")]
        count = self.figures.show_run(self._figs_since or self._job_start, [r for r in extra if r])
        fig_tab = self.results.indexOf(self.figures)
        self.results.setTabText(fig_tab, f"Figures ({count} new)" if count else "Figures")
        qjob, self._current_job = self._current_job, None
        if qjob is not None:
            qjob.state = "stopped" if self._stopping else "done" if ok == n else "failed"
            qjob.note = (self._summary or f"{ok}/{n} steps OK") + (
                f", {self._problems} problem line{'s' if self._problems > 1 else ''} in the log" if self._problems else "") + (
                f", took {took}" if took else "")
            self._save_queue()
            self._refresh_queue()
        if self._queue_running:
            if self._stopping:
                self._queue_running = False
                self.status.setText("Stopped; the queue is paused")
            elif qjob is not None and qjob.state == "failed" and self.q_stop_on_fail.isChecked():
                self._queue_running = False
                self.status.setText(f"Queue stopped: '{qjob.title}' failed")
                self._notify("Queue stopped", f"'{qjob.title}' failed after {took}")
            else:
                QTimer.singleShot(0, self._next_queued)
        else:
            if not self._stopping:
                self._notify("Finished" if ok == n else "Failed", self.status.text()
                             + (f": {self._steps[0].label}" if self._steps else "") + (f" (took {took})" if took else ""))
            if count and ok == n and not self._stopping:
                self.results.setCurrentIndex(fig_tab)
        after, self._after_job = self._after_job, None
        if after is not None and ok == n and not self._stopping:
            after()
        self.refresh()

    def stop_run(self):
        if self.proc is None:
            if self._queue_running:         # waiting for another run: just stop waiting
                self._queue_running = False
                self.status.setText("Queue paused")
                self._refresh_queue()
                self.refresh()
            return
        self._stopping = True
        tree = _app._descendants(int(self.proc.processId()))
        for pid in reversed(tree):                     # the deepest (PS-DTFE) first
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.proc.terminate()
        # after a grace period, force every process of the run that is still there -- not only
        # the script's shell: a child that outlived its parent would run on unseen
        started = _app._start_times(tree)                   # a pid freed and reused in the grace period is not ours
        QTimer.singleShot(5000, lambda: _app._kill_leftovers(tree, started))
        self.status.setText("Stopping…")

    def queue_void_build(self, rec: R.RunRecord):
        """The Runs tab's 'Build void catalogue': a Plots job of the 'voidcat' set for this run's snapshot, grids
        and prefix, added to the queue."""
        if not R.void_candidate(rec):
            return
        spec = rs.PlotSpec(data_root=self.root_edit.text().strip(), sim=rec.sim, snapshots=[int(rec.snap)],
                           sets=["voidcat"], method=rec.estimator,
                           prefix="" if rec.prefix in ("ps_output", "output") else rec.prefix,
                           figures_root=self.plots.figures_root)
        qjob = rs.QueuedJob.of("plots", spec)
        self.queue.append(qjob)
        self._save_queue()
        self._refresh_queue()
        self.q_list.setCurrentRow(len(self.queue) - 1)
        self.results.setCurrentIndex(self.results.indexOf(self.queue_tab))
        self.status.setText(f"Queued: {qjob.title}" + ("" if self._queue_running else " (Start queue runs it)"))

    def _notify(self, title: str, text: str):
        if os.environ.get("DTFE_GUI_NO_NOTIFY") == "1" or not self.q_notify.isChecked():
            return
        cmd = R.notify_command(f"DTFE launcher: {title}", text)
        if cmd:
            QProcess.startDetached(cmd[0], cmd[1:])

    # ---------------------------------------------------------------- failure banner
    _ACTION_TEXT = {"check_memory": "Check memory", "scratch": "Set a scratch folder…",
                    "choose_root": "Choose the data folder…", "open_data": "Open the Data tab",
                    "build": "Build the programs…", "start_over": "Tick Start over"}

    def _show_banner(self, diagnoses: list[R.Diagnosis]):
        while self.banner_buttons.count():
            w = self.banner_buttons.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        if not diagnoses:
            self.banner.hide()
            return
        self.banner_text.setText("<br>".join(f"<b>{d.title}.</b> {d.advice}" for d in diagnoses))
        for d in diagnoses:
            if d.action in self._ACTION_TEXT:
                b = QPushButton(self._ACTION_TEXT[d.action])
                b.clicked.connect(lambda _=False, a=d.action: self._banner_action(a))
                self.banner_buttons.addWidget(b)
        close = QPushButton("Dismiss")
        close.clicked.connect(self.banner.hide)
        self.banner_buttons.addStretch(1)
        self.banner_buttons.addWidget(close)
        self.banner.show()

    def _banner_action(self, action: str):
        tab = self._job_tab
        if action == "check_memory":
            self.tabs.setCurrentIndex(tab if tab in (_app.GRIDS, _app.PIPELINE, _app.OWN) else _app.GRIDS)
            if self.mem_btn.isEnabled():
                self.check_memory()
        elif action == "scratch":
            t = tab if tab in (_app.GRIDS, _app.PIPELINE, _app.OWN) else _app.GRIDS
            self.tabs.setCurrentIndex(t)
            edit = {_app.GRIDS: self.scratch_edit, _app.PIPELINE: self.p_scratch, _app.OWN: self.o_scratch}[t]
            self._browse_scratch(edit)
        elif action == "choose_root":
            self._browse_root()
        elif action == "open_data":
            self.tabs.setCurrentIndex(_app.DATA)
        elif action == "build":
            self.run_setup(page=1)
        elif action == "start_over":                # a refused resume: the tab's start-over tick
            t = tab if tab in (_app.GRIDS, _app.PIPELINE) else _app.GRIDS
            self.tabs.setCurrentIndex(t)
            (self.so_check if t == _app.GRIDS else self.p_so).setChecked(True)
        self.banner.hide()

    # ================================================================ queue
    def enqueue(self):
        self._read_widgets()
        tab, job = self.tabs.currentIndex(), self._job()
        if any(level == "error" for level, _ in job.problems()):
            return
        qjob = rs.QueuedJob.of(_app.KIND_OF_TAB[tab], job)
        self.queue.append(qjob)
        self._save_queue()
        self._refresh_queue()
        self.q_list.setCurrentRow(len(self.queue) - 1)
        self.results.setCurrentIndex(self.results.indexOf(self.queue_tab))
        if self.proc is None and not self._queue_running:
            self.status.setText(f"Queued: {qjob.title}")

    def start_queue(self):
        if self._queue_running or rs.next_waiting(self.queue) is None or self._mcheck is not None:
            return
        self._queue_running = True
        self._figs_since = time.time()
        if self.proc is None:
            self.log.clear()
        self._next_queued()
        self.refresh()

    def _next_queued(self):
        if not self._queue_running or self.proc is not None:
            return                          # paused, or a job is running (its end calls this again)
        i = rs.next_waiting(self.queue)
        if i is None:
            self._queue_running = False
            counts = Counter(j.state for j in self.queue)
            self.status.setText("Queue finished: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
            self._notify("Queue finished", ", ".join(f"{v} {k}" for k, v in counts.items()))
            self._refresh_queue()
            self.refresh()
            return
        qjob = self.queue[i]
        if self._mcheck is not None:        # a memory check started while the queue waited: its snapshot read and
            # free-memory reading must not run beside a job (Run and Start queue refuse too). Poll again: the
            # check's end does not call this
            self.status.setText(f"Queue: waiting for the memory check to finish before '{qjob.title}'")
            QTimer.singleShot(self._wait_poll_ms, self._next_queued)
            return
        try:
            spec = qjob.build()
        except (TypeError, ValueError) as e:
            self._skip_queued(qjob, f"unreadable settings: {e}", failed=True)
            return
        if qjob.kind in rs.HEAVY_KINDS and self.q_wait.isChecked():
            busy = rs.running_jobs(max_age=0)
            if busy:
                pid, name = busy[0]
                # its ETA from the progress sidecar the last machine poll read, when it has one
                prog = next((e.get("progress") for e in self.machine.get("running") or [] if e.get("pid") == pid), None)
                eta = (prog or {}).get("eta_s")
                eta = f", ETA {R.format_duration(float(eta))}" if isinstance(eta, (int, float)) and eta >= 0 else ""
                self.status.setText(f"Queue: waiting for {name} (pid {pid}{eta}) to finish before '{qjob.title}'")
                QTimer.singleShot(self._wait_poll_ms, self._next_queued)
                return
        steps = spec.steps()
        errors = [m for level, m in spec.problems() if level == "error"]
        if not steps:
            self._skip_queued(qjob, "nothing left to do", failed=False)
            return
        if errors:
            self._skip_queued(qjob, errors[0], failed=True)
            return
        qjob.state, qjob.note = "running", ""
        self._save_queue()
        self._refresh_queue()
        self._begin(steps, _app.TAB_OF_KIND[qjob.kind], spec, qjob)
        self.refresh()

    def _skip_queued(self, qjob: rs.QueuedJob, why: str, failed: bool):
        """A job that cannot start: 'skipped' (counts as a failure) or 'done' when nothing is left."""
        qjob.state, qjob.note = ("skipped" if failed else "done"), why
        self.log.appendPlainText(f"{'Skipped' if failed else 'Nothing to do for'} '{qjob.title}': {why}")
        self._save_queue()
        self._refresh_queue()
        if failed and self.q_stop_on_fail.isChecked():
            self._queue_running = False
            self.status.setText(f"Queue stopped: '{qjob.title}' could not start ({why})")
            self.refresh()
            return
        QTimer.singleShot(0, self._next_queued)

    def _refresh_queue(self):
        row = self.q_list.currentRow()
        self.q_list.blockSignals(True)
        self.q_list.clear()
        for k, j in enumerate(self.queue, 1):
            it = QListWidgetItem(f"{_app.QUEUE_ICON[j.state]}  {k}. {j.title}" + (f"   ({j.note})" if j.note else ""))
            it.setForeground(_app.QUEUE_COLOR[j.state])
            it.setToolTip(j.title + (f"\n{j.note}" if j.note else ""))
            self.q_list.addItem(it)
        self.q_list.blockSignals(False)
        if self.queue:
            self.q_list.setCurrentRow(min(max(row, 0), len(self.queue) - 1))
        self._show_queued(self.q_list.currentRow())
        waiting = sum(j.state == "waiting" for j in self.queue)
        self.results.setTabText(self.results.indexOf(self.queue_tab),
                                f"Queue ({waiting} waiting)" if waiting else "Queue")
        self._queue_buttons()

    def _queue_buttons(self):
        row = self.q_list.currentRow()
        sel = self.queue[row] if 0 <= row < len(self.queue) else None
        editable = sel is not None and sel is not self._current_job
        self.q_start.setEnabled(not self._queue_running and rs.next_waiting(self.queue) is not None
                                and getattr(self, "_mcheck", None) is None)
        self.q_start.setText("Resume queue" if any(j.state != "waiting" for j in self.queue)
                             and rs.next_waiting(self.queue) is not None else "Start queue")
        self.q_up.setEnabled(editable and row > 0)
        self.q_down.setEnabled(editable and row < len(self.queue) - 1)
        self.q_remove.setEnabled(editable)
        self.q_clear.setEnabled(any(j.state not in ("waiting", "running") for j in self.queue))
        self._sync_queue_actions()

    def _show_queued(self, row: int):
        if not 0 <= row < len(self.queue):
            self.q_detail.setPlainText("Queue jobs from any tab with 'Add to queue'.")
            self._queue_buttons()
            return
        j = self.queue[row]
        text = [f"{j.title}", f"state: {j.state}" + (f" ({j.note})" if j.note else "")]
        try:
            spec = j.build()
            steps = spec.steps()
            text += ["", "Commands if it started now:" if j.state == "waiting" else "Commands (as of now):",
                     rs.script_text(steps) or "(none: nothing left to do)"]
            problems = spec.problems()
            if problems:
                text += ["", "Checks now:"] + [("✖ " if lv == "error" else "⚠ ") + m for lv, m in problems]
        except (TypeError, ValueError) as e:
            text.append(f"unreadable settings: {e}")
        self.q_detail.setPlainText("\n".join(text))
        self._queue_buttons()

    def _move_queued(self, delta: int):
        row = self.q_list.currentRow()
        new = row + delta
        if not (0 <= row < len(self.queue) and 0 <= new < len(self.queue)):
            return
        self.queue[row], self.queue[new] = self.queue[new], self.queue[row]
        self._save_queue()
        self._refresh_queue()
        self.q_list.setCurrentRow(new)

    def _remove_queued(self):
        row = self.q_list.currentRow()
        if 0 <= row < len(self.queue) and self.queue[row] is not self._current_job:
            del self.queue[row]
            self._save_queue()
            self._refresh_queue()

    def _clear_finished(self):
        self.queue = [j for j in self.queue if j.state in ("waiting", "running")]
        self._save_queue()
        self._refresh_queue()

    def _save_queue(self, *_):
        if not self.remember:
            return
        self.settings.setValue("queue", json.dumps([j.to_dict() for j in self.queue]))
        self.settings.setValue("queue_options", json.dumps({"stop_on_fail": self.q_stop_on_fail.isChecked(),
                                                            "wait": self.q_wait.isChecked(),
                                                            "notify": self.q_notify.isChecked()}))

    def _restore_queue(self) -> list[rs.QueuedJob]:
        out = []
        for d in self._restore_json("queue", []):
            try:
                out.append(rs.QueuedJob.from_dict(d))
            except (TypeError, ValueError):
                pass                        # a job from an incompatible older version
        return out


import app as _app  # noqa: E402
