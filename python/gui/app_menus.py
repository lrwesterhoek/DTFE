"""MainWindow's menu bar (MenusMixin, moved from app.py): the File, Edit, View, Run, Explore, Queue, Figures and
Runs menus, the _sync_*_actions methods that keep them in step with the window's buttons, and what they
offer beyond those buttons: copying, showing and exporting a job's commands (a bash script or a SLURM job),
the Setup dialog and the demo. The names app.py defines are read through _app, imported last (see
app_tabs.py)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QLineEdit,
                               QSpinBox, QWidget)

import runspec as rs
import results as R
from app_widgets import _note, _grow_fields


class MenusMixin:
    def _build_menus(self):
        """The menu bar: Setup, running (Run, Add to Queue, Stop, the memory check), the Explore
        tab's, the queue's, the figure browser's and the Runs browser's actions, and the commands
        behind a run (shown, copied, exported). Each also has its button in the window; the
        _sync_*_actions methods keep the Run, Explore, Queue, Figures and Runs menus in step with
        those buttons."""
        bar = self.menuBar()
        file_menu = bar.addMenu("File")
        self.setup_action = QAction("Setup…", self)
        self.setup_action.setShortcut(QKeySequence("Ctrl+,"))
        self.setup_action.setMenuRole(QAction.MenuRole.NoRole)   # macOS would move 'Setup' into the app menu
        self.setup_action.triggered.connect(lambda: self.run_setup())
        file_menu.addAction(self.setup_action)
        file_menu.addSeparator()
        self.export_action = QAction("Export Commands…", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.triggered.connect(self.export_job)
        file_menu.addAction(self.export_action)
        self.copy_action = QAction("Copy Commands", self)
        self.copy_action.triggered.connect(self.copy_commands)
        edit_menu = bar.addMenu("Edit")
        edit_menu.addAction(self.copy_action)
        edit_menu.addSeparator()
        self.find_log_action = QAction("Find in Log", self)
        self.find_log_action.setShortcut(QKeySequence.Find)
        self.find_log_action.triggered.connect(self._focus_log_find)
        edit_menu.addAction(self.find_log_action)
        self.copy_log_action = QAction("Copy Log", self)
        self.copy_log_action.triggered.connect(self.copy_log)
        edit_menu.addAction(self.copy_log_action)
        self.open_log_action = QAction("Open Run Log", self)
        self.open_log_action.setToolTip("The newest run's full log file, beside its outputs")
        self.open_log_action.triggered.connect(self.open_run_log)
        self.open_log_action.setEnabled(False)
        edit_menu.addAction(self.open_log_action)
        self.show_cmds_action = QAction("Show Commands", self, checkable=True)
        self.show_cmds_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.show_cmds_action.setChecked(self.show_commands)
        self.show_cmds_action.toggled.connect(self._toggle_commands)
        bar.addMenu("View").addAction(self.show_cmds_action)
        run_menu = bar.addMenu("Run")
        for attr, text, keys, slot in (("run_action", "Run", "Ctrl+R", self.start_run),
                                       ("queue_action", "Add to Queue", "Ctrl+Shift+R", self.enqueue),
                                       ("stop_action", "Stop", "Ctrl+.", self.stop_run),
                                       ("mem_action", "Check Memory", "Ctrl+Shift+M", self.check_memory)):
            if attr == "mem_action":
                run_menu.addSeparator()
            action = QAction(text, self, enabled=False)
            action.setShortcut(QKeySequence(keys))
            action.triggered.connect(lambda _=False, slot=slot: slot())
            run_menu.addAction(action)
            setattr(self, attr, action)
        explore_menu = bar.addMenu("Explore")
        for attr, text, keys, what in (("open_output_action", "Open Output…", "Ctrl+O", "open"),
                                       ("find_outputs_action", "Look for Outputs Again", "", "find"),
                                       ("save_image_action", "Save Image…", "Ctrl+S", "save"),
                                       ("save_options_action", "Save Options…", "", "save_options"),
                                       ("zoom_out_action", "Zoom Out", "Ctrl+Shift+O", "zoom"),
                                       ("grid_action", "Figure Grid", "Ctrl+G", "grid"),
                                       ("server_action", "Start Query Server", "", "server")):
            if attr == "server_action":
                explore_menu.addSeparator()
            action = QAction(text, self, enabled=what in ("open", "find", "save_options"))
            if keys:
                action.setShortcut(QKeySequence(keys))
            action.triggered.connect(lambda _=False, what=what: self._explore_menu(what))
            explore_menu.addAction(action)
            setattr(self, attr, action)
        queue_menu = bar.addMenu("Queue")
        for item in (("q_start_action", "Start Queue", "start"), None,
                     ("q_up_action", "Move Job Up", "up"),
                     ("q_down_action", "Move Job Down", "down"),
                     ("q_remove_action", "Remove Job", "remove"),
                     ("q_clear_action", "Clear Finished Jobs", "clear"), None):
            if item is None:
                queue_menu.addSeparator()
                continue
            attr, text, what = item
            action = QAction(text, self, enabled=False)
            action.triggered.connect(lambda _=False, what=what: self._queue_menu(what))
            queue_menu.addAction(action)
            setattr(self, attr, action)
        # the queue's options (its tick boxes; _build_queue_tab ties each to its box)
        for attr, text in (("q_stop_on_fail_action", "Stop the Queue When a Job Fails"),
                           ("q_wait_action", "Wait While Another DTFE Run Is Going"),
                           ("q_notify_action", "Notify When a Job or the Queue Finishes")):
            action = QAction(text, self, checkable=True)
            queue_menu.addAction(action)
            setattr(self, attr, action)
        figures_menu = bar.addMenu("Figures")
        reveal = "Show in Finder" if sys.platform == "darwin" else "Show Folder"
        for item in (("fig_refresh_action", "Refresh Figures", "refresh"), None,
                     ("fig_open_action", "Open Figure", "open"),
                     ("fig_reveal_action", reveal, "reveal"), None):
            if item is None:
                figures_menu.addSeparator()
                continue
            attr, text, what = item
            action = QAction(text, self, enabled=what == "refresh")
            action.triggered.connect(lambda _=False, what=what: self._figures_menu(what))
            figures_menu.addAction(action)
            setattr(self, attr, action)
        # which figures the browser lists (its drop-down; the Figures tab ties the two together)
        self.fig_mode_group = QActionGroup(self)
        self.fig_mode_actions = []
        for i, text in enumerate(("From the Last Run", "All Figures")):
            action = QAction(text, self.fig_mode_group, checkable=True, checked=i == 0)
            action.triggered.connect(lambda _=False, i=i: self._figures_menu("mode", i))
            figures_menu.addAction(action)
            self.fig_mode_actions.append(action)
        runs_menu = bar.addMenu("Runs")
        reveal = "Show in Finder" if sys.platform == "darwin" else "Show Folder"
        for item in (("runs_refresh_action", "Refresh Runs", "refresh"), None,
                     ("runs_log_action", "Open Log", "log"),
                     ("runs_load_action", "Load Settings", "load"),
                     ("runs_explore_action", "Open in Explore", "explore"),
                     ("runs_reveal_action", reveal, "reveal"), None,
                     ("runs_trash_action", "Move to Trash…", "trash")):
            if item is None:
                runs_menu.addSeparator()
                continue
            attr, text, what = item
            action = QAction(text, self, enabled=what == "refresh")
            action.triggered.connect(lambda _=False, what=what: self._runs_menu(what))
            runs_menu.addAction(action)
            setattr(self, attr, action)

    def _bring_up(self, widget: QWidget):
        """Show one of the results tabs (Queue, Runs), leaving Explore for the last tab with the
        run panel."""
        if self.tabs.currentIndex() == _app.EXPLORE:
            self.tabs.setCurrentIndex(self._panel_tab)
        self.results.setCurrentWidget(widget)

    def _runs_menu(self, what: str):
        """A Runs menu item: its button in the Runs browser. 'Refresh Runs' works from anywhere and
        brings the browser up first; the others act on the run selected there, so they are only
        offered while the browser is on screen."""
        if what == "refresh" and (self.tabs.currentIndex() == _app.EXPLORE or self.results.currentWidget() is not self.runs):
            self._bring_up(self.runs)                               # (showing it reloads it)
            return
        getattr(self.runs, "b_" + what).click()

    def _figures_menu(self, what: str, mode: int = 0):
        """A Figures menu item: its button (or drop-down) in the figure browser. Refreshing and
        choosing which figures to list work from anywhere and bring the browser up; opening and
        revealing act on the figure selected there, so they are only offered while it is on
        screen."""
        if what in ("refresh", "mode"):
            self._bring_up(self.figures)
            if what == "mode":
                self.figures.mode.setCurrentIndex(mode)        # (reloads the list)
            else:
                self.figures.refresh_btn.click()
            return
        {"open": self.figures.open_btn, "reveal": self.figures.reveal_btn}[what].click()

    def _sync_figures_actions(self, tab: int | None = None):
        if not hasattr(self, "figures") or not hasattr(self, "tabs"):
            return
        shown = (self.tabs.currentIndex() if tab is None else tab) != _app.EXPLORE \
            and self.results.currentWidget() is self.figures
        self.fig_open_action.setEnabled(shown and self.figures.open_btn.isEnabled())
        self.fig_reveal_action.setEnabled(shown and self.figures.reveal_btn.isEnabled())
        self.fig_mode_actions[self.figures.mode.currentIndex()].setChecked(True)

    def _queue_menu(self, what: str):
        """A Queue menu item: its button on the Queue tab. Starting the queue (its log comes up) and
        clearing finished jobs (the Queue tab comes up) work from anywhere; moving and removing act
        on the job selected there, so they are only offered while the Queue tab is on screen."""
        if what == "start" and self.tabs.currentIndex() == _app.EXPLORE:
            self.tabs.setCurrentIndex(self._panel_tab)
        elif what == "clear":
            self._bring_up(self.queue_tab)
        getattr(self, "q_" + what).click()

    def _sync_queue_actions(self):
        if not hasattr(self, "queue_tab") or not hasattr(self, "tabs"):
            return
        shown = self.tabs.currentIndex() != _app.EXPLORE and self.results.currentWidget() is self.queue_tab
        self.q_start_action.setText(self.q_start.text().title())      # Start / Resume Queue
        self.q_start_action.setEnabled(self.q_start.isEnabled())
        self.q_clear_action.setEnabled(self.q_clear.isEnabled())
        for what in ("up", "down", "remove"):
            getattr(self, f"q_{what}_action").setEnabled(shown and getattr(self, "q_" + what).isEnabled())

    def _runs_changed(self):
        """The Runs browser's selection or table changed: its menu follows. After a rescan (Refresh, Move to
        Trash, a job's end) Explore drops outputs whose grids left the disk: else a trashed run would stay
        listed there and its next slice read fail."""
        self._sync_runs_actions()
        if self.runs.records is not getattr(self, "_runs_seen", None):
            self._runs_seen = self.runs.records
            self.explore.refresh_if_vanished()

    def _sync_runs_actions(self, tab: int | None = None):
        if not hasattr(self, "results") or not hasattr(self, "tabs"):
            return
        shown = (self.tabs.currentIndex() if tab is None else tab) != _app.EXPLORE \
            and self.results.currentWidget() is self.runs
        for what in ("log", "load", "explore", "reveal", "trash"):
            getattr(self, f"runs_{what}_action").setEnabled(shown and getattr(self.runs, "b_" + what).isEnabled())

    def _explore_menu(self, what: str):
        """An Explore menu item: the same as its button on the Explore tab. Opening or looking for
        outputs works from any tab and brings Explore up first."""
        x = self.explore
        if what in ("open", "find") and self.tabs.currentIndex() != _app.EXPLORE:
            self.tabs.setCurrentIndex(_app.EXPLORE)
        if what in ("grid", "save_options") and self.tabs.currentIndex() != _app.EXPLORE:
            self.tabs.setCurrentIndex(_app.EXPLORE)
        {"open": lambda: x._open_other(), "find": lambda: x.refresh_outputs(),
         "save": lambda: x._save_image(), "save_options": lambda: x.open_save_options(),
         "zoom": lambda: x.zoom_out(),
         "grid": lambda: x.grid_toggle.setChecked(not x.grid_toggle.isChecked()),
         "server": lambda: x._server_button()}[what]()

    def _sync_explore_actions(self, tab: int | None = None):
        """Save Image needs a map on screen; the server can be started from Explore and stopped
        (or its start cancelled) from anywhere."""
        x = getattr(self, "explore", None)
        if x is None or not hasattr(self, "tabs"):
            return
        on = (self.tabs.currentIndex() if tab is None else tab) == _app.EXPLORE
        self.save_image_action.setEnabled(on and x.can_save())
        self.zoom_out_action.setEnabled(on and x.is_zoomed())
        self.grid_action.setEnabled(x.current is not None)
        self.grid_action.setText("Back to the Map" if x._grid_shown else "Figure Grid")
        phase = x.server_phase()
        self.server_action.setText({"starting": "Cancel Server Start",
                                    "running": "Stop Query Server"}.get(phase, "Start Query Server"))
        self.server_action.setEnabled(x.server_btn.isEnabled() and (on or phase != "stopped"))

    def _sync_run_actions(self, tab: int):
        """The Run menu does what the buttons below the checks do, when they can: Run and Add to
        Queue need the run panel (not Explore), Stop and a running memory check work anywhere."""
        panel = tab != _app.EXPLORE
        self.run_action.setEnabled(panel and self.run_btn.isEnabled())
        self.queue_action.setEnabled(panel and self.queue_btn.isEnabled())
        self.stop_action.setEnabled(self.stop_btn.isEnabled())
        self.mem_action.setText("Cancel Memory Check" if self._mcheck is not None else "Check Memory")
        self.mem_action.setEnabled(self._mcheck is not None
                                   or (tab in (_app.GRIDS, _app.PIPELINE, _app.OWN) and self.mem_btn.isEnabled()))
        self._sync_explore_actions(tab)
        self._sync_figures_actions(tab)
        self._sync_runs_actions(tab)

    def copy_commands(self):
        self._read_widgets()
        steps = self._job().steps()
        if steps:
            QApplication.clipboard().setText(rs.script_text(steps))
            self.status_message(f"copied {len(steps)} command{'s' if len(steps) != 1 else ''}")

    def _toggle_commands(self, on: bool):
        if on == self.show_commands:
            return
        self.show_commands = on
        self.cmd_box.setVisible(on)
        self.show_cmds_action.setChecked(on)
        self._save()
        self.refresh()

    def _results_tab_changed(self, i: int):
        if self.results.widget(i) is self.runs:
            self.runs.reload()
        self._sync_runs_actions()
        self._sync_queue_actions()
        self._sync_figures_actions()

    # ---------------------------------------------------------------- export
    def export_job(self):
        """The current job as a bash script, or a SLURM batch job (e.g. for Hábrók)."""
        self._read_widgets()
        steps = self._job().steps()
        if not steps:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Export the commands")
        f = QFormLayout(dlg)
        kind = QComboBox()
        kind.addItems(["bash script", "SLURM batch job (cluster)"])
        f.addRow("As", kind)
        name = QLineEdit("dtfe")
        wall = QLineEdit("24:00:00")
        cpus = QSpinBox(minimum=1, maximum=256, value=16)
        mem = QLineEdit("64G")
        part = QLineEdit("regular")
        mail = QLineEdit(placeholderText="optional: mail me when it ends")
        rows = [("Job name", name), ("Time limit", wall), ("CPUs", cpus), ("Memory", mem), ("Partition", part),
                ("E-mail", mail)]
        for label, w in rows:
            f.addRow(label, w)
        t = R.time_estimate(self._runs(), sim=self.spec.sim, grid=self.spec.grid, estimator=self.spec.estimator,
                            gpu=False, sliced=bool(self.spec.slice_plane), precision=self.spec.precision) \
            if self.tabs.currentIndex() == _app.GRIDS else None
        f.addRow(_note("No GPU on a cluster: GPU options fall back to the CPU. Set REPO and the data paths "
                       "in the script." + (f" Earlier CPU runs: ≈ {R.format_duration(t[0])} per snapshot." if t else "")))

        def sync(i):
            for label, w in rows:
                w.setEnabled(i == 1)
        kind.currentIndexChanged.connect(sync)
        sync(0)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        _grow_fields(dlg)
        dlg.resize(520, dlg.sizeHint().height())
        if dlg.exec() != QDialog.Accepted:
            return
        slurm = None if kind.currentIndex() == 0 else {
            "job_name": name.text().strip(), "time": wall.text().strip(), "cpus": cpus.value(),
            "mem": mem.text().strip(), "partition": part.text().strip(), "mail": mail.text().strip()}
        default = str(Path.home() / (("dtfe_job.slurm" if slurm else "dtfe_job.sh")))
        path, _ = QFileDialog.getSaveFileName(self, "Save the script", default)
        if not path:
            return
        title = rs.job_summary(_app.KIND_OF_TAB.get(self.tabs.currentIndex(), "grids"), self._job())
        Path(path).write_text(rs.export_script(steps, title, slurm))
        os.chmod(path, 0o755)
        self.status_message(f"saved {path}" + ("  (submit with: sbatch " + Path(path).name + ")" if slurm else ""))

    # ---------------------------------------------------------------- setup
    def run_setup(self, page: int = 0) -> bool:
        """The first-run dialog (setup_wizard.py): data folder, programs, API key, demo."""
        import setup_wizard as SW
        dlg = SW.SetupDialog(self, self.root_edit.text().strip())
        if page:
            dlg._go(page)
        ok = dlg.exec() == QDialog.Accepted
        if self.remember:
            self.settings.setValue("setup_done", 1)
        if not ok:
            return False
        root = dlg.data_root()
        if root != self.root_edit.text().strip():
            self.root_edit.setText(root)
            self._root_changed()
        if dlg.run_demo:
            self.start_demo()
        return True

    def start_demo(self):
        """The demo (runspec.demo_spec): generate, run, then open it in Explore."""
        demo = rs.demo_spec()               # (the CPU deposit: faster than the GPU at 32^3 particles)
        self.custom = demo
        self._load_into_widgets()
        self.tabs.setCurrentIndex(_app.OWN)
        self._read_widgets()
        self.refresh()
        if self.proc is not None:
            self.status_message("the demo is set up in the Custom snapshot tab: run it when the current job is done")
            return
        root = str(demo.output_root())
        self._after_job = lambda: self._explore_output(root)
        self._figs_since = time.time()
        self._begin(demo.steps(), _app.OWN, demo, None)


import app as _app  # noqa: E402
