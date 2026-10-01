#!/usr/bin/env python3
"""End-to-end check of the PySide6 GUI (python/gui/app.py), offscreen, on a synthetic per-family
data root: discovery and the command preview; REAL runs through the Run button of all three
tabs -- Grids (a two-snapshot PS-DTFE run: progress, log, exit summary, 'done' marks), Plots
(plot_PS_DTFE.py per snapshot, the figures appearing in the browser), Pipeline (plan only,
then grids + image plane + plot_pointeval.py) and Data (download through a FAKE wget that copies
local TNG-like chunks -- no network -- then merge_HDF5.py) -- and Stop killing the process tree.

The slice-map script can only write into python/figures; its output for the synthetic
simulation (python/figures/fields/GUITEST-1-Dark) is removed at the end.

Needs PySide6 and the built ./PS-DTFE (make PS-DTFE); without either it reports SKIP and exits
0 -- unless DTFE_GUI_TEST_REQUIRED=1 (as in CI), where a skip is a failure, so a broken PySide6
install can never turn the suite silently green.

Usage: ~/.venvs/dtfe-gui/bin/python tests/py_gui_app_test.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "gui"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["DTFE_GUI_NO_NOTIFY"] = "1"          # no desktop notifications from a test run
REQUIRED = os.environ.get("DTFE_GUI_TEST_REQUIRED") == "1"

try:
    from PySide6.QtCore import QEventLoop, QSettings, QTimer
    from PySide6.QtWidgets import QApplication
except ImportError as e:
    print(f"SKIP: PySide6 is not importable ({e}) -- python3 -m venv ~/.venvs/dtfe-gui "
          "--system-site-packages; ~/.venvs/dtfe-gui/bin/pip install PySide6")
    sys.exit(1 if REQUIRED else 0)

import app as A  # noqa: E402
import runspec as rs  # noqa: E402

PASS, FAIL = [], []
SIM = "GUITEST-1-Dark"
DL_SIM = "GUIDL-1-Dark"
SECRET = "gui-test-secret-key-0123"

# Stands in for wget: copies the fixture chunks of the requested snapshot into -P <dir> and
# prints wget -nv's one line per file. Never touches the network.
FAKE_WGET = r"""#!/usr/bin/env bash
dest=""; url=""
while [ $# -gt 0 ]; do
    case "$1" in -P) dest="$2"; shift ;; http*) url="$1" ;; esac
    shift
done
ep=$(echo "$url" | sed -n 's|.*/files/\([^/]*\)/.*|\1|p')
[ -d "@FIXTURES@/$ep" ] || { echo "fake wget: no fixture for $url" >&2; exit 8; }
for f in "@FIXTURES@/$ep"/*.hdf5; do
    cp "$f" "$dest/"
    b=$(wc -c < "$f" | tr -d ' ')
    echo "2026-09-29 12:00:00 URL:$url$(basename "$f") [$b/$b] -> \"$dest/$(basename "$f")\" [1]"
done
"""


def make_raw_chunks(folder: Path, n: int, chunks: int, per_chunk: int = 2048, h: float = 0.6774):
    """TNG-like raw snapshot chunks (h-units, float64 coordinates) for merge_HDF5.py."""
    import h5py
    import numpy as np
    folder.mkdir(parents=True)
    rng = np.random.default_rng(n)
    for c in range(chunks):
        with h5py.File(folder / f"snap_{n:03d}.{c}.hdf5", "w") as f:
            hd = f.create_group("Header").attrs
            hd["BoxSize"], hd["HubbleParam"] = 75000.0, h
            hd["Redshift"] = {50: 1.0, 99: 0.0}[n]
            hd["Time"] = 1.0 / (1.0 + hd["Redshift"])
            hd["MassTable"] = np.array([0, 0.5, 0, 0, 0, 0], dtype=float)
            hd["NumPart_ThisFile"] = np.array([0, per_chunk, 0, 0, 0, 0], dtype=np.int64)
            hd["NumPart_Total"] = np.array([0, per_chunk * chunks, 0, 0, 0, 0], dtype=np.int64)
            hd["NumFilesPerSnapshot"] = chunks
            p = f.create_group("PartType1")
            p["Coordinates"] = rng.uniform(0, 75000.0, (per_chunk, 3))
            p["ParticleIDs"] = np.arange(c * per_chunk, (c + 1) * per_chunk, dtype=np.uint64)
            p["Velocities"] = rng.normal(0, 100, (per_chunk, 3)).astype(np.float32)


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"   {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))


def wait_idle(w, limit):
    loop, t0 = QEventLoop(), time.time()
    poll = QTimer(interval=50)
    poll.timeout.connect(lambda: (w.proc is None or time.time() - t0 > limit) and loop.quit())
    poll.start()
    loop.exec()
    return time.time() - t0


def wait_queue(w, limit):
    """Until the queue has stopped (finished, paused or stopped at a failure) and nothing runs."""
    loop, t0 = QEventLoop(), time.time()
    poll = QTimer(interval=100)
    poll.timeout.connect(lambda: ((not w._queue_running and w.proc is None) or time.time() - t0 > limit)
                         and loop.quit())
    poll.start()
    loop.exec()
    return time.time() - t0


def wait_check(w, limit):
    """Until the memory check has finished."""
    loop, t0 = QEventLoop(), time.time()
    poll = QTimer(interval=100)
    poll.timeout.connect(lambda: (w._mcheck is None or time.time() - t0 > limit) and loop.quit())
    poll.start()
    loop.exec()
    return time.time() - t0


def main():
    if not (ROOT / "PS-DTFE").exists():
        print("SKIP: ./PS-DTFE is not built (make PS-DTFE)")
        return 1 if REQUIRED else 0
    qa = QApplication([])
    tmp = Path(tempfile.mkdtemp(prefix="gui_app_"))
    data = tmp / "fam root"                                   # a space, like 'Samsung T7'
    repo_figs = rs.FIGURES_ROOT / "fields" / SIM
    try:
        sim = data / "GUITEST" / SIM
        for n in (50, 99):
            (sim / f"snapdir_{n:03d}").mkdir(parents=True)
        subprocess.run([sys.executable, str(ROOT / "tests" / "generate_ps_test_data.py"),
                        "--out", str(sim / "snapdir_099" / "combined_099.hdf5"), "--n", "24",
                        "--box", "100.0"], check=True, capture_output=True)
        shutil.copy(sim / "snapdir_099" / "combined_099.hdf5", sim / "snapdir_050" / "combined_050.hdf5")

        spec = rs.RunSpec(data_root=str(data), sim=SIM, snapshots=[50, 99], grid=32, gpu=False,
                          partition=2, max_concurrent=2, output_prefix="ps_gui")
        pipe = rs.PipelineSpec(sims=[SIM], nu=64, grid=32, figures_root=str(tmp / "figs"))
        w = A.MainWindow(spec, remember=False, pipeline=pipe, plots=rs.PlotSpec(sets=[]))
        w.show()
        qa.processEvents()

        print("window:")
        check("simulation discovered in the per-family root",
              [w.sim_combo.itemText(i) for i in range(w.sim_combo.count())] == [SIM])
        check("both snapshots listed and ticked, none done yet",
              [(w.snap_list.item(i).data(0x0100), w.snap_list.item(i).checkState().value)
               for i in range(w.snap_list.count())] == [(50, 2), (99, 2)]
              and "done" not in w.snap_list.item(0).text())
        check("the command preview is the RunSpec's shell line", w.cmd_view.toPlainText() == spec.shell_line())
        check("Run is enabled for a valid spec", w.run_btn.isEnabled())
        w.dtfe_radio.setChecked(True)
        qa.processEvents()
        check("DTFE mode disables the PS-only options and switches script",
              not w.ps_group.isEnabled() and not w.slice_group.isEnabled()
              and "scripts/run_dtfe.sh" in w.cmd_view.toPlainText())
        w.ps_radio.setChecked(True)
        w.snap_list.item(0).setCheckState(w.snap_list.item(0).checkState().Unchecked)
        w.snap_list.item(1).setCheckState(w.snap_list.item(1).checkState().Unchecked)
        qa.processEvents()
        check("no snapshots ticked blocks Run, with a neutral hint (not a red error)", not w.run_btn.isEnabled()
              and any(w.checks.item(i).text().startswith("→") and "tick the snapshots" in w.checks.item(i).text()
                      for i in range(w.checks.count()))
              and not any(w.checks.item(i).text().startswith("✖") for i in range(w.checks.count())))
        w._select_snaps("all")
        qa.processEvents()
        check("'All' re-ticks them and re-enables Run", w.spec.snapshots == [50, 99] and w.run_btn.isEnabled())

        print("run:")
        seen = set()
        tick = QTimer(interval=20)
        tick.timeout.connect(lambda: w.proc is not None and seen.add((w.bar.value(), w.bar.maximum(),
                                                                       w.snap_label.text())))
        tick.start()
        # ... and every bar change as it happens: on a tiny run the last partition and the end of
        # the run can fall between two 20 ms samples (missed 8/8 once on Ubuntu)
        w.bar.valueChanged.connect(lambda v: w.proc is not None
                                   and seen.add((v, w.bar.maximum(), w.snap_label.text())))
        w.run_btn.click()
        dt = wait_idle(w, 300)
        tick.stop()
        log = w.log.toPlainText()
        check(f"a real two-snapshot run finishes ({dt:.0f} s)", w.status.text() == "Finished", w.status.text())
        check("the exit summary is shown", w.snap_label.text() == "2 processed, 0 skipped, 0 failed.",
              w.snap_label.text())
        check("the partition bar reached 8/8", (8, 8) in {(v, m) for v, m, _ in seen}, str(sorted(seen)[-3:]))
        check("the snapshot counter stepped 1/2 -> 2/2",
              {"snapshot 1/2", "snapshot 2/2"} <= {s for _, _, s in seen})
        check("progress fragments kept out of the log, results kept in",
              "[partitions" not in log and log.count("processed successfully") == 2)
        check("both snapshots now marked done",
              all("done" in w.snap_list.item(i).text() for i in range(w.snap_list.count())))
        check("outputs written with the chosen prefix",
              all((sim / f"snapdir_{n:03d}" / "ps_gui.a_den").is_file() for n in (50, 99)))

        print("plots:")
        w.tabs.setCurrentIndex(A.PLOTS)
        qa.processEvents()
        marks = lambda: ["grids" in w.snap_list.item(i).text() for i in range(w.snap_list.count())]  # noqa: E731
        check("the default prefix finds no grids (the run wrote 'ps_gui')", marks() == [False, False])
        check("no figure set ticked blocks Run", not w.run_btn.isEnabled())
        w.fig_list.item(0).setCheckState(w.fig_list.item(0).checkState().Checked)     # Field slice maps
        w.pl_method.setCurrentIndex(w.pl_method.findData("ps"))
        w.pl_prefix.setText("ps_gui")
        w.pl_prefix.editingFinished.emit()
        qa.processEvents()
        check("... the 'ps_gui' prefix marks both snapshots as having grids", marks() == [True, True])
        steps = w.plots.steps()
        check("one plot_PS_DTFE.py call per snapshot, with estimator and prefix",
              [st.argv[2:] for st in steps] == [["--sim", SIM, "--snap", str(n), "--method", "ps", "--prefix",
                                                 "ps_gui"] for n in (50, 99)]
              and all(st.env["MPLBACKEND"] == "Agg" and st.env["DTFE_SIM"] == SIM for st in steps),
              str([st.argv[2:] for st in steps]))
        check("the preview lists both commands", w.cmd_view.toPlainText() == rs.script_text(steps))
        w.run_btn.click()
        dt = wait_idle(w, 300)
        n_figs = w.figures.list.count()
        check(f"both plot steps succeed ({dt:.0f} s)", w.status.text() == "Finished"
              and w.snap_label.text() == "2/2 steps OK", f"{w.status.text()} | {w.snap_label.text()}")
        check("the figures they wrote are listed and the browser is shown",
              n_figs > 0 and w.results.currentWidget() is w.figures
              and w.results.tabText(w.results.indexOf(w.figures)) == f"Figures ({n_figs} new)"
              and all(SIM in w.figures.list.item(i).text() for i in range(n_figs)), f"{n_figs} figures")
        check("the first figure is previewed", w.figures.preview.pixmap() is not None
              and not w.figures.preview.pixmap().isNull())
        check("the Figures menu can open or reveal the selected figure, like the buttons",
              w.fig_open_action.isEnabled() and w.fig_reveal_action.isEnabled()
              and w.fig_mode_actions[0].isChecked())
        w.results.setCurrentIndex(w.results.indexOf(w.log))
        qa.processEvents()
        check("... but not while the browser is off screen", not w.fig_open_action.isEnabled()
              and not w.fig_reveal_action.isEnabled() and w.fig_refresh_action.isEnabled())
        w.fig_mode_actions[1].trigger()               # Figures > All Figures
        qa.processEvents()
        check("Figures > All Figures brings the browser up, listing every figure",
              w.results.currentWidget() is w.figures and w.figures.mode.currentIndex() == 1
              and w.figures.list.count() >= n_figs and w.fig_open_action.isEnabled())
        w.figures.mode.setCurrentIndex(0)
        qa.processEvents()
        check("choosing in the drop-down ticks the menu's choice", w.fig_mode_actions[0].isChecked()
              and not w.fig_mode_actions[1].isChecked() and w.figures.list.count() == n_figs)
        w.results.setCurrentIndex(w.results.indexOf(w.log))
        w.fig_refresh_action.trigger()                # Figures > Refresh Figures
        qa.processEvents()
        check("Figures > Refresh Figures brings the browser up too",
              w.results.currentWidget() is w.figures and w.figures.list.count() == n_figs)

        print("pipeline:")
        w.tabs.setCurrentIndex(A.PIPELINE)
        qa.processEvents()
        check("the pipeline tab disables the snapshot list and plans both snapshots",
              not w.snap_list.isEnabled() and f"{SIM}: 2 of 2 snapshots to compute" in w.estimate.text(),
              w.estimate.text())
        w.p_plan.setChecked(True)
        qa.processEvents()
        check("plan only: one DRY_RUN step, no render", len(w.pipe.steps()) == 1
              and w.pipe.steps()[0].env.get("DRY_RUN") == "1" and not w.p_render.isEnabled())
        w.run_btn.click()
        wait_idle(w, 60)
        check("the plan announces both snapshots and computes nothing",
              w.status.text() == "Finished" and "run_ps_dtfe.sh" in w.log.toPlainText()
              and not (sim / "hires_plane_z64.bin").exists(), w.status.text())
        w.p_plan.setChecked(False)
        qa.processEvents()
        check("the real pipeline is two steps: pipeline, then plot_pointeval.py on its own plane",
              [st.label for st in w.pipe.steps()] == [f"run_ps_pipeline.sh {SIM}", f"plot_pointeval.py {SIM}"]
              and str(sim / "hires_plane_z64.json") in w.pipe.steps()[1].argv)
        seen.clear()
        tick.start()
        w.run_btn.click()
        dt = wait_idle(w, 600)
        tick.stop()
        check(f"pipeline + render succeed ({dt:.0f} s)", w.status.text() == "Finished", w.status.text())
        check("the pipeline's snapshot plan drove the counter (2 planned)",
              any(s.endswith("/2") for _, _, s in seen), str({s for _, _, s in seen}))
        check("slices written, and nothing left to compute",
              all((sim / f"snapdir_{n:03d}" / "ps_output.pts_den").is_file() for n in (50, 99))
              and f"{SIM}: 0 of 2 snapshots to compute" in w.estimate.text(), w.estimate.text())
        pe = rs.figures([tmp / "figs"])
        check("the point-evaluated figures are in the browser",
              len(pe) > 0 and w.figures.list.count() == len(pe), f"{len(pe)} on disk, {w.figures.list.count()} listed")

        print("data (download + merge, with a fake wget -- no network):")
        fixtures, fakebin = tmp / "fixtures", tmp / "bin"
        for n in (50, 99):
            make_raw_chunks(fixtures / f"snapshot-{n}", n, chunks=2)
        fakebin.mkdir()
        (fakebin / "wget").write_text(FAKE_WGET.replace("@FIXTURES@", str(fixtures)))
        (fakebin / "wget").chmod(0o755)
        os.environ["PATH"] = f"{fakebin}:{os.environ['PATH']}"
        os.environ["TNG_API_KEY"] = SECRET
        (data / "GUIDL").mkdir()                       # an (empty) family folder for the new run
        w.tabs.setCurrentIndex(A.DATA)
        w.d_sim.setCurrentText(DL_SIM)
        w.d_sim.lineEdit().editingFinished.emit()
        qa.processEvents()
        rows = {w.d_table.item(r, 0).data(0x0100): r for r in range(w.d_table.rowCount())}
        check("the Data tab lists the redshift ladder for a simulation not on disk yet",
              {0, 50, 99} <= set(rows) and "(new folder)" in w.d_target.text()
              and w.d_target.text().startswith(f"GUIDL/{DL_SIM}")
              and w.d_target.toolTip().startswith(str(data / "GUIDL" / DL_SIM)), w.d_target.text())
        for n in (50, 99):
            w.d_table.item(rows[n], 0).setCheckState(w.d_table.item(rows[n], 0).checkState().Checked)
        qa.processEvents()
        check("download + merge of the ticked snapshots, as two steps",
              [st.label for st in w.data.steps()] == ["download snapshots 50 99", "merge snapshots 50 99"]
              and w.run_btn.isEnabled(), str([st.label for st in w.data.steps()]))
        check("the API key is on no command line", SECRET not in w.cmd_view.toPlainText()
              and all(SECRET not in " ".join(st.argv) + str(st.env) for st in w.data.steps()))
        seen.clear()
        tick.start()
        w.run_btn.click()
        dt = wait_idle(w, 120)
        tick.stop()
        sd = data / "GUIDL" / DL_SIM
        check(f"download + merge succeed ({dt:.0f} s)", w.status.text() == "Finished", w.status.text())
        check("the downloaded files were counted", any("4 files" in s for _, _, s in seen),
              str({s for _, _, s in seen}))
        check("the merge showed its chunk progress", (2, 2) in {(v, m) for v, m, _ in seen})
        check("both combined files verify against the header totals",
              all((sd / f"snapdir_{n:03d}" / f"combined_{n:03d}.hdf5").is_file() for n in (50, 99))
              and w.log.toPlainText().count("Particles merged: [4096]") == 2
              and "MISMATCH" not in w.log.toPlainText(), w.log.toPlainText()[-600:])
        check("the key never reached the log", SECRET not in w.log.toPlainText())
        rows = {w.d_table.item(r, 0).data(0x0100): r for r in range(w.d_table.rowCount())}
        check("the table now shows them merged, chunks kept",
              all(w.d_table.item(rows[n], 2).text() == "merged (+2 chunks)" for n in (50, 99)))
        check("the new simulation appears in the shared simulation list",
              DL_SIM in [w.sim_combo.itemText(i) for i in range(w.sim_combo.count())])
        check("running it again has nothing left to do (merged items are never re-merged)",
              not w.run_btn.isEnabled() and w.data.steps() == [])
        os.environ.pop("TNG_API_KEY")

        print("memory check:")
        (sim / "hires_plane_z64.bin").unlink(missing_ok=True)
        import numpy as np
        np.random.default_rng(3).uniform(0, 100, (64 * 64, 3)).tofile(sim / "slice_plane_z64.bin")
        w._fill_sim_dependent()                     # the window lists a simulation's planes when it loads it
        w.tabs.setCurrentIndex(A.GRIDS)
        w.sim_combo.setCurrentText(SIM)
        qa.processEvents()
        w._select_snaps("all")
        w.prefix_edit.setText("ps_mem")
        w.prefix_edit.editingFinished.emit()
        w.grid_combo.setCurrentText("32")
        w.plane_combo.setCurrentIndex(w.plane_combo.findData(str(sim / "slice_plane_z64.bin")))
        qa.processEvents()
        check("the slice is selected", w.spec.slice_plane.endswith("slice_plane_z64.bin"), w.spec.slice_plane)
        lines = lambda: [w.checks.item(i).text() for i in range(w.checks.count())]  # noqa: E731
        check("an unchecked slice run says memory is not checked yet",
              any("memory not checked" in t for t in lines()), str(lines()))
        check("'Check memory' is offered for a PS grids run, in the window and in the Run menu",
              w.mem_btn.isEnabled() and w.mem_action.isEnabled() and w.mem_action.text() == "Check Memory")
        w.mem_btn.click()
        wait_check(w, 120)
        check("the check reports every snapshot fits", any(t.startswith("✔") and "memory: all 2 snapshots fit" in t
                                                            for t in lines()), str(lines()))
        res = w.mem_results[w.spec.memory_key()]
        check("... from the binary's own report (partition, concurrency, prediction, budget)",
              {(SIM, 50, None), (SIM, 99, None)} <= set(res) and all("predicted_gb" in r and "budget_gb" in r
                                                                      for r in res.values()), str(res))
        os.environ["DTFE_MEM_BUDGET_GB"] = "0.0001"             # a machine far too small
        w.mem_action.trigger()                                    # this time from the menu
        check("Run > Check Memory starts the same check (the menu item then cancels it)",
              w._mcheck is not None and w.mem_action.text() == "Cancel Memory Check"
              and w.mem_btn.text() == "Cancel check")
        wait_check(w, 120)
        check("over the budget: a warning per snapshot, naming what lowers it",
              sum("over the" in t and "budget" in t and "smaller slice plane" in t for t in lines()) == 2, str(lines()))
        del os.environ["DTFE_MEM_BUDGET_GB"]
        w.grid_combo.setCurrentText("16")
        qa.processEvents()
        check("changed settings are not covered by the old check", any("memory not checked" in t for t in lines()))

        rs.ADAPTIVE_MIN_NU, min_nu = 64, rs.ADAPTIVE_MIN_NU      # small planes for a small box
        w.tabs.setCurrentIndex(A.PIPELINE)
        w.p_nu.setCurrentText("256")
        w.p_grid.setCurrentText("32")
        w.p_prefix.setText("ps_mem")
        w.p_prefix.editingFinished.emit()
        w.p_render.setChecked(False)
        qa.processEvents()
        predicted = {}
        for nu in (256, 128):
            predicted[nu] = w.pipe.check_step(SIM, [99], nu)
        # the report for both sizes at a generous budget, then a budget BETWEEN them
        outs = {}
        for nu, st in predicted.items():
            env = dict(os.environ, **st.env)
            r = subprocess.run(st.argv, env=env, capture_output=True, text=True, cwd=str(ROOT))
            outs[nu] = rs.parse_reports(r.stdout).get(99, {}).get("predicted_gb")
        check("a larger plane needs more memory", outs[256] and outs[128] and outs[256] > outs[128], str(outs))
        os.environ["DTFE_MEM_BUDGET_GB"] = f"{(outs[256] + outs[128]) / 2:.4f}"
        w.p_adaptive.setChecked(True)
        qa.processEvents()
        check("adaptive without a check is refused", not w.run_btn.isEnabled()
              and any("run 'Check memory' first" in t for t in lines()), str(lines()))
        w.mem_btn.click()
        wait_check(w, 300)
        check("the check halves the plane where it does not fit, and plans the fitting size",
              w.pipe.plan.get(SIM) == {"050": 128, "099": 128} and w.run_btn.isEnabled(), str(w.pipe.plan))
        check("the pipeline then runs at that size, on those snapshots",
              [(st.env.get("NU"), st.env.get("SNAPS")) for st in w.pipe.steps()] == [("128", "50 99")],
              str([(st.env.get("NU"), st.env.get("SNAPS")) for st in w.pipe.steps()]))
        w.p_adaptive.setChecked(False)
        qa.processEvents()
        check("without Adaptive the over-budget size is a warning that names the size that fits",
              any("at 256² needs" in t and "128² fits" in t for t in lines()), str(lines()))
        del os.environ["DTFE_MEM_BUDGET_GB"]
        rs.ADAPTIVE_MIN_NU = min_nu

        print("queue:")
        w.q_wait.setChecked(False)           # (a real DTFE run may be going on this machine)
        w.q_stop_on_fail.setChecked(True)
        w.tabs.setCurrentIndex(A.GRIDS)
        w.sim_combo.setCurrentText(SIM)
        qa.processEvents()
        w.prefix_edit.setText("ps_q1")
        w.prefix_edit.editingFinished.emit()
        w.grid_combo.setCurrentText("16")
        w._select_snaps("none")
        w.snap_list.item(0).setCheckState(w.snap_list.item(0).checkState().Checked)          # 050 only
        qa.processEvents()
        w.queue_btn.click()                                                                    # job 1: grids
        check("'Add to queue' queues the Grids job and shows the queue",
              len(w.queue) == 1 and w.queue[0].kind == "grids" and w.queue[0].state == "waiting"
              and w.results.currentWidget() is w.queue_tab and "050" in w.queue[0].title, str(w.queue))
        w.tabs.setCurrentIndex(A.PLOTS)
        qa.processEvents()
        w.pl_prefix.setText("ps_q1")                  # grids that job 1 has not written yet
        w.pl_prefix.editingFinished.emit()           # (snapshots shared with the Grids tab: 050)
        qa.processEvents()
        check("a plot of grids a queued job will write can be queued (only a warning now)",
              w.queue_btn.isEnabled())
        w.queue_btn.click()                                                                    # job 2: plots
        fields_item, thesis_item = w.fig_list.item(0), w.fig_list.item(len(rs.FIGURE_SETS) - 1)
        fields_item.setCheckState(fields_item.checkState().Unchecked)
        thesis_item.setCheckState(thesis_item.checkState().Checked)
        w.opt_widgets[("thesis", "mode")][1].setCurrentText("check")   # exits 1: no DTFE grids here
        qa.processEvents()
        w.queue_btn.click()                                                                    # job 3: fails
        thesis_item.setCheckState(thesis_item.checkState().Unchecked)
        fields_item.setCheckState(fields_item.checkState().Checked)
        w.opt_widgets[("fields", "pointeval_only")][1].setChecked(True)
        qa.processEvents()
        check("Run > Add to Queue is offered like the button", w.queue_action.isEnabled())
        w.queue_action.trigger()                                                               # job 4, from the menu
        check("four jobs queued, all waiting", [j.state for j in w.queue] == ["waiting"] * 4
              and w.results.tabText(w.results.indexOf(w.queue_tab)) == "Queue (4 waiting)")
        titles = [j.title for j in w.queue]
        w.q_list.setCurrentRow(3)
        qa.processEvents()
        check("the Queue menu offers what the buttons offer for the selected (last) job",
              all(getattr(w, f"q_{k}_action").isEnabled() == getattr(w, f"q_{k}").isEnabled()
                  for k in ("start", "up", "down", "remove", "clear"))
              and w.q_up_action.isEnabled() and not w.q_down_action.isEnabled()
              and w.q_start_action.text() == "Start Queue")
        w.q_up_action.trigger()                       # Queue > Move Job Up
        moved = [j.title for j in w.queue]
        w.q_down_action.trigger()                     # ... and back down
        check("Move Job Up / Down reorder the queue", moved == titles[:2] + [titles[3], titles[2]]
              and [j.title for j in w.queue] == titles, str(moved))
        check("the queue's options are ticked in the menu like on the tab",
              w.q_stop_on_fail_action.isChecked() and not w.q_wait_action.isChecked())
        w.q_wait_action.setChecked(True)
        check("... and ticking one in the menu ticks the tab's box (and back)", w.q_wait.isChecked())
        w.q_wait.setChecked(False)
        check("...", not w.q_wait_action.isChecked())
        w.results.setCurrentIndex(w.results.indexOf(w.log))
        qa.processEvents()
        check("with the Queue tab not on screen: no moving or removing, but the queue can start",
              not w.q_up_action.isEnabled() and not w.q_remove_action.isEnabled() and w.q_start_action.isEnabled())
        w.q_start_action.trigger()                    # Queue > Start Queue
        check("Queue > Start Queue starts it (its log comes up, as with the button)",
              w._queue_running and w.results.currentWidget() is w.log and not w.q_start_action.isEnabled())
        dt = wait_queue(w, 600)
        states = [j.state for j in w.queue]
        check(f"the queue runs in order and stops at the failed job ({dt:.0f} s)",
              states == ["done", "done", "failed", "waiting"] and "failed" in w.status.text(),
              f"{states} | {w.status.text()}")
        check("job 2 plotted the grids job 1 wrote",
              (sim / "snapdir_050" / "ps_q1.a_den").is_file()
              and len(rs.figures([repo_figs / "snap050"], since=w._figs_since)) > 0)
        log = w.log.toPlainText()
        check("one log for the whole queue, with a header per job",
              all(f"Queue job {k}/4" in log for k in (1, 2, 3)) and "Queue job 4/4" not in log)
        check("the failed job's note says how far it got", w.queue[2].note == "0/1 steps OK", w.queue[2].note)
        w.q_stop_on_fail.setChecked(False)
        qa.processEvents()
        check("the button offers to resume (and so does the menu)", w.q_start.text() == "Resume queue"
              and w.q_start.isEnabled() and w.q_start_action.text() == "Resume Queue" and w.q_start_action.isEnabled())
        w.q_start.click()
        wait_queue(w, 300)
        check("resumed: the last job ran too", [j.state for j in w.queue] == ["done", "done", "failed", "done"]
              and w.status.text().startswith("Queue finished"), w.status.text())

        orig_jobs = rs.running_jobs
        rs.running_jobs = lambda max_age=2.0: [(424242, "PS-DTFE")]     # pretend another run is going
        try:
            check("Queue > Clear Finished Jobs is offered for a finished queue", w.q_clear_action.isEnabled())
            w.q_clear_action.trigger()
            check("'Clear finished' empties a finished queue", w.queue == [] and not w.q_clear_action.isEnabled())
            w.q_wait.setChecked(True)
            w._wait_poll_ms = 200
            w.tabs.setCurrentIndex(A.GRIDS)
            w.prefix_edit.setText("ps_q2")
            w.prefix_edit.editingFinished.emit()
            qa.processEvents()
            w.queue_btn.click()
            w.q_start.click()
            loop = QEventLoop()
            QTimer.singleShot(1000, loop.quit)
            loop.exec()
            check("a grids job waits while another PS-DTFE runs",
                  w.proc is None and w.queue[0].state == "waiting" and "waiting for PS-DTFE (pid 424242)"
                  in w.status.text() and w.stop_btn.isEnabled(), w.status.text())
            w.stop_btn.click()
            check("Stop while waiting pauses the queue", not w._queue_running and w.status.text() == "Queue paused")
        finally:
            rs.running_jobs = orig_jobs
        w.q_wait.setChecked(False)
        w.q_start.click()
        wait_queue(w, 300)
        check("resumed once free, the job runs", w.queue[0].state == "done"
              and (sim / "snapdir_050" / "ps_q2.a_den").is_file(), w.queue[0].note)

        ini = QSettings(str(tmp / "gui.ini"), QSettings.IniFormat)
        w2 = A.MainWindow(remember=True, settings=ini)
        w2.queue = [rs.QueuedJob.of("grids", w.spec), rs.QueuedJob.of("plots", w.plots)]
        w2.queue[1].state = "running"
        w2.q_wait.setChecked(False)
        w2._save_queue()
        w3 = A.MainWindow(remember=True, settings=QSettings(str(tmp / "gui.ini"), QSettings.IniFormat))
        check("the queue and its options survive a restart (a job left running comes back 'stopped')",
              [(j.kind, j.state) for j in w3.queue] == [("grids", "waiting"), ("plots", "stopped")]
              and w3.queue[0].spec == w.spec.to_dict() and not w3.q_wait.isChecked(),
              str([(j.kind, j.state) for j in w3.queue]))
        w2.close()
        w3.close()

        new_user_features(qa, w, tmp, data, sim)

        print("stop:")
        w.tabs.setCurrentIndex(A.GRIDS)
        qa.processEvents()
        w.deposit_combo.setCurrentIndex(w.deposit_combo.findData("exact"))  # slow: exact CPU deposit
        w.grid_combo.setCurrentText("256")
        w.prefix_edit.setText("ps_gui_stop")
        w.prefix_edit.editingFinished.emit()
        qa.processEvents()
        check("Run > Run is offered like the button, Stop is not (nothing running)",
              w.run_action.isEnabled() and not w.stop_action.isEnabled())
        w.run_action.trigger()                        # from the menu this time
        check("... and starts the run: now Stop is offered and Run is not",
              w.proc is not None and w.stop_action.isEnabled() and not w.run_action.isEnabled())
        pid = int(w.proc.processId())
        loop = QEventLoop()
        QTimer.singleShot(3000, loop.quit)
        loop.exec()
        tree = A._descendants(pid)
        names = subprocess.run(["ps", "-o", "comm=", "-p", ",".join(map(str, tree))],
                               capture_output=True, text=True).stdout.split()
        check("the run's process tree reaches the PS-DTFE binary",
              any(n.endswith("PS-DTFE") for n in names), str(names))
        w.stop_action.trigger()                       # Run > Stop
        wait_idle(w, 30)
        # a process takes a moment to die (tearing down GBs of memory in a VM: it was still 'R' at
        # the instant the shell exited, on Ubuntu), so allow the GUI's 5 s grace + its SIGKILL;
        # a ZOMBIE (state Z) has exited and only waits for its parent to collect it -- where PID 1
        # never does (a docker RUN shell), a killed orphan stays listed that way; it is not running
        deadline = time.time() + 10
        while True:
            rows = subprocess.run(["ps", "-o", "pid=,stat=,comm=", "-p", ",".join(map(str, tree))],
                                  capture_output=True, text=True).stdout.splitlines()
            alive = [r.strip() for r in rows if r.split() and not r.split()[1].startswith("Z")]
            if not alive or time.time() > deadline:
                break
            loop = QEventLoop()
            QTimer.singleShot(250, loop.quit)       # keep the GUI's timers (its SIGKILL) running
            loop.exec()
        check("Stop ends every process of the run", not alive and w.status.text() == "Stopped",
              f"alive={alive} all={[r.strip() for r in rows]} status={w.status.text()}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(repo_figs, ignore_errors=True)       # only the synthetic simulation's figures
    print("-" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


def spin(ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait_for(cond, limit):
    t0 = time.time()
    while not cond() and time.time() - t0 < limit:
        spin(100)
    return cond()


def new_user_features(qa, w, tmp, data, sim):
    import presets as P
    import setup_wizard as SW
    lines = lambda: [w.checks.item(i).text() for i in range(w.checks.count())]  # noqa: E731

    print("layout:")
    views = (w.snap_list, w.d_table, w.p_sims, w.fig_list)
    check("every checkable list and table draws its own tick boxes (macOS draws only the current row's)",
          all(isinstance(v.itemDelegate(), A.CheckDelegate) for v in views))
    for i in range(A.EXPLORE):          # paint every tab: the delegate's drawing path runs
        w.tabs.setCurrentIndex(i)
        qa.processEvents()
        check(f"tab {i} paints", not w.grab().isNull())
    forms = w.findChildren(A.QFormLayout)
    check("every form lets its fields take the available width (paths are not squeezed)",
          forms and all(f.fieldGrowthPolicy() == A.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
                        for f in forms), f"{len(forms)} forms")
    # the window's own words never contain '--' (flags live in the commands, which are exempt)
    def visible_texts():
        from PySide6.QtWidgets import (QAbstractButton, QComboBox, QGroupBox, QLabel, QLineEdit,
                                       QListWidget, QPlainTextEdit, QTableWidget, QTabWidget, QWidget)
        exempt = {w.cmd_view, w.log, w.q_detail}
        out = []
        for wd in w.findChildren(QWidget):
            if wd in exempt:
                continue
            out.append(("tooltip", wd.toolTip()))
            if isinstance(wd, (QLabel, QAbstractButton)):
                out.append(("text", wd.text()))
            if isinstance(wd, QGroupBox):
                out.append(("title", wd.title()))
            if isinstance(wd, QLineEdit):
                out.append(("placeholder", wd.placeholderText()))
            if isinstance(wd, QComboBox):
                out += [("item", wd.itemText(i)) for i in range(wd.count())]
                out += [("item tip", str(wd.itemData(i, 3) or "")) for i in range(wd.count())]
            if isinstance(wd, QListWidget):
                out += [("list", wd.item(i).text() + " " + wd.item(i).toolTip()) for i in range(wd.count())]
            if isinstance(wd, QTableWidget):
                out += [("cell", wd.item(r, c).text()) for r in range(wd.rowCount()) for c in range(wd.columnCount())
                        if wd.item(r, c) is not None]
            if isinstance(wd, QTabWidget):
                out += [("tab", wd.tabText(i) + " " + wd.tabToolTip(i)) for i in range(wd.count())]
        return [(k, t) for k, t in out if "--" in t]
    bad = []
    for i in range(A.EXPLORE + 1):
        w.tabs.setCurrentIndex(i)
        qa.processEvents()
        bad += visible_texts()
    w.dtfe_radio.setChecked(True)
    qa.processEvents()
    bad += visible_texts()
    w.ps_radio.setChecked(True)
    qa.processEvents()
    check("no '--' in any label, title, tooltip, placeholder, list, table or check message",
          not bad, str(bad[:5]))

    w.tabs.setCurrentIndex(A.DATA)
    qa.processEvents()
    f = w._data_form
    check("Data tab: the shared simulation and snapshot rows are hidden (one picker, the tab's own)",
          not f.isRowVisible(w.snap_list) and not f.isRowVisible(w.sim_combo) and f.isRowVisible(0))
    w.tabs.setCurrentIndex(A.GRIDS)
    qa.processEvents()
    check("Grids tab: shown again", f.isRowVisible(w.snap_list) and f.isRowVisible(w.sim_combo))
    check("the commands are hidden by default (title row too), but kept up to date",
          w.cmd_box.isHidden() and not w.show_cmds_action.isChecked()
          and w.cmd_view.toPlainText().startswith("DTFE_DATA_ROOT"))
    check("the menus can still export and copy them",
          w.export_action.isEnabled() and w.copy_action.isEnabled())
    file_menu = w.menuBar().actions()[0].menu()
    opened = []
    w.run_setup = lambda page=0: opened.append(page) or False     # (not the real, modal dialog)
    w.setup_action.trigger()
    del w.run_setup
    check("File > Setup… opens the setup dialog, kept in the File menu (not moved to the app menu)",
          opened == [0] and file_menu.actions()[0] is w.setup_action and w.setup_action.text() == "Setup…"
          and w.setup_action.menuRole() == A.QAction.MenuRole.NoRole)
    w.show_cmds_action.trigger()
    qa.processEvents()
    check("View > Show Commands shows them", not w.cmd_box.isHidden() and w.show_commands
          and w.cmd_title.text().startswith("Command"))
    w.cmd_hide.click()
    qa.processEvents()
    check("'Hide' hides them again and unticks the menu item",
          w.cmd_box.isHidden() and not w.show_cmds_action.isChecked() and not w.show_commands)
    check("the options' tooltips are the binary's own --full_help text, in plain prose",
          w.vm_check.toolTip().startswith("PS-DTFE only") and len(w.vm_check.toolTip()) > 200
          and "--" not in w.vm_check.toolTip())

    print("presets:")
    combo = w._preset_grids[0]
    combo.setCurrentIndex(combo.findData("Quick look"))
    w._apply_preset("grids")
    qa.processEvents()
    check("'Quick look' sets 128^3 and density + velocity in the form and the command",
          w.spec.grid == 128 and w.spec.fields == ["density_a", "velocity_a"] and w.grid_combo.currentText() == "128"
          and "-g 128" in w.cmd_view.toPlainText() and w._preset_grids[2].text() != "")
    w.user_presets["mine"] = P.capture(w.spec)
    w._fill_presets("grids")
    check("a saved preset is offered next to the built-in ones", combo.findData("mine") > len(P.BUILTIN_PRESETS) - 1)
    combo.setCurrentIndex(combo.findData("mine"))
    w._delete_preset("grids")
    check("... and can be deleted", combo.findData("mine") < 0 and "mine" not in w.user_presets)

    print("custom snapshot:")
    demo = rs.demo_spec(tmp / "demo", gpu=False)
    w.custom = demo
    w._load_into_widgets()
    w.tabs.setCurrentIndex(A.OWN)
    qa.processEvents()
    check("the demo in the Custom snapshot tab: make its folder, generate, run -- all shown and runnable",
          w.run_btn.isEnabled() and "generate_ps_test_data.py" in w.cmd_view.toPlainText()
          and w.cmd_title.text().startswith("Commands (3"), w.cmd_view.toPlainText())
    w.run_btn.click()
    dt = wait_idle(w, 300)
    root = demo.output_root()
    runlog = Path(str(root) + ".runlog")
    side = Path(str(root) + ".gui.json")
    import json
    check(f"the demo runs ({dt:.0f} s): grids, a run log copied from the output, the settings sidecar",
          w.status.text() == "Finished" and Path(str(root) + ".a_den").is_file()
          and "total wall time" in runlog.read_text() and json.loads(side.read_text())["box_mpc"] == [0, 100] * 3,
          w.status.text())

    print("runs:")
    w.results.setCurrentIndex(w.results.indexOf(w.log))
    w.tabs.setCurrentIndex(A.EXPLORE)
    qa.processEvents()
    check("the Runs menu's run actions are off while the browser is not on screen",
          w.runs_refresh_action.isEnabled() and not any(
              a.isEnabled() for a in (w.runs_log_action, w.runs_load_action, w.runs_explore_action,
                                      w.runs_reveal_action, w.runs_trash_action)))
    w.runs_refresh_action.trigger()                   # Runs > Refresh Runs, from Explore
    qa.processEvents()
    check("Runs > Refresh Runs brings the browser up (leaving Explore for the last run-panel tab)",
          w.tabs.currentIndex() == A.OWN and w.results.currentWidget() is w.runs, str(w.tabs.currentIndex()))
    rows = [(r.sim, r.prefix) for r in w.runs._rows]
    check("the Runs browser lists the TNG-layout runs and the custom-snapshot run",
          ("", "demo") in rows and (SIM, "ps_gui") in rows, str(rows))
    rec = next(r for r in w.runs._rows if r.prefix == "ps_gui")
    check("... with their build stamp, grid and wall time",
          rec.build_rev != "" and rec.grid == 32 and rec.wall_seconds is not None and rec.ok, str(rec))
    check("nothing selected: the menu's run actions are off, like the buttons",
          not w.runs_load_action.isEnabled() and not w.runs.b_load.isEnabled())
    w.runs.table.selectRow(w.runs._rows.index(rec))
    qa.processEvents()
    check("a run selected: the menu offers what its buttons offer",
          all(getattr(w, f"runs_{k}_action").isEnabled() == getattr(w.runs, f"b_{k}").isEnabled()
              for k in ("log", "load", "explore", "reveal", "trash"))
          and w.runs_load_action.isEnabled() and w.runs_trash_action.isEnabled())
    w.runs_load_action.trigger()                      # Runs > Load Settings
    qa.processEvents()
    check("'Load settings' puts the run's settings into the Grids tab",
          w.tabs.currentIndex() == A.GRIDS and w.spec.grid == 32 and w.spec.snapshots == [rec.snap]
          and w.spec.output_prefix == "ps_gui" and w.grid_combo.currentText() == "32")
    w._runs_scanned = 0.0
    w.refresh()
    check("the estimate line has the time of the earlier identical runs",
          "earlier run" in w.estimate.text() and "per snapshot" in w.estimate.text(), w.estimate.text())

    print("explore:")
    w._explore_output(str(root))
    qa.processEvents()
    ok = wait_for(lambda: w.explore._plane is not None and not w.explore._busy, 30)
    x = w.explore
    check("Explore opens the demo output and draws a slice",
          ok and w.tabs.currentIndex() == A.EXPLORE and x.current is not None and str(x.current.root) == str(root)
          and w.right.currentWidget() is w.explore_view and w.explore_view.canvas._img is not None)
    check("on Explore (no run panel) the Run menu offers no Run, Add to Queue or memory check",
          not w.run_action.isEnabled() and not w.queue_action.isEnabled() and not w.mem_action.isEnabled())
    check("... while the Explore menu offers Save Image… and Start Query Server, like the buttons",
          w.save_image_action.isEnabled() and w.server_action.isEnabled()
          and w.server_action.text() == "Start Query Server" and x.server_btn.isEnabled())
    from PySide6.QtTest import QTest
    w.activateWindow()
    x.field.setFocus()
    qa.processEvents()
    i0 = x.index.value()
    QTest.keyClick(x.field, A.Qt.Key_Right)
    qa.processEvents()
    check("→ shows the next slice (focus on a drop-down list)", x.index.value() == i0 + 1
          and x.slider.value() == i0 + 1, f"{i0} -> {x.index.value()}")
    QTest.keyClick(x.field, A.Qt.Key_Left, A.Qt.ShiftModifier)
    qa.processEvents()
    check("shift+← goes back 10 slices", x.index.value() == i0 - 9, str(x.index.value()))
    x.cache_edit.setFocus()
    QTest.keyClick(x.cache_edit, A.Qt.Key_Right)
    qa.processEvents()
    check("in a text field the arrows stay the cursor's", x.index.value() == i0 - 9)
    x.index.setValue(i0)
    x.field.setFocus()
    qa.processEvents()
    f0 = x.field.currentIndex()
    QTest.keyClick(x.field, A.Qt.Key_Down)
    qa.processEvents()
    check("↓ shows the next field", x.field.count() > 1 and x.field.currentIndex() == f0 + 1,
          f"{f0} -> {x.field.currentIndex()} of {x.field.count()}")
    QTest.keyClick(x.field, A.Qt.Key_Up)
    qa.processEvents()
    check("↑ the previous one", x.field.currentIndex() == f0)
    QTest.keyClick(x.field, A.Qt.Key_Up)
    qa.processEvents()
    if f0 == 0:
        check("... and stops at the top of the list", x.field.currentIndex() == 0)
    x.field.setCurrentIndex(f0)
    x.index.setFocus()
    qa.processEvents()
    QTest.keyClick(x.index, A.Qt.Key_Up)
    qa.processEvents()
    check("in the slice-number field ↑ still steps its value (not the field)",
          x.index.value() == i0 + 1 and x.field.currentIndex() == f0, f"{x.index.value()} {x.field.currentIndex()}")
    x.index.setValue(i0)
    x.field.setFocus()
    w.tabs.setCurrentIndex(A.PLOTS)
    qa.processEvents()
    w.pl_prefix.setFocus()
    w.tabs.setFocus()
    QTest.keyClick(w.tabs, A.Qt.Key_Right)
    qa.processEvents()
    check("off the Explore tab the arrows leave the slices alone", x.index.value() == i0)
    w.tabs.setCurrentIndex(A.EXPLORE)
    qa.processEvents()
    x.field.setCurrentIndex(x.field.findData("velocity"))
    x.component.setCurrentText("z")
    x.axis.setCurrentIndex(0)
    ok = wait_for(lambda: not x._busy and x._req and x._req["field"] == "velocity" and x._req["axis"] == 0, 30)
    check("a vector field component on another axis", ok and x._plane is not None and x._plane.shape == (64, 64))
    arrows = w.explore_view.canvas.arrows
    check("velocity opens as a vector field: its in-plane arrows (y, z for an x slice) over the map",
          x.vectors.isEnabled() and x.vectors.isChecked() and arrows is not None
          and arrows["axes"] == (1, 2) and len(arrows["x"]) == 32 * 32 and arrows["vmax"] > 0)
    x.vectors.setChecked(False)
    ok = wait_for(lambda: not x._busy and x._req is not None and not x._req["arrows"], 30)
    check("... and without 'vectors' the plain map", ok and w.explore_view.canvas.arrows is None)
    x.vectors.setChecked(True)
    x.field.setCurrentIndex(x.field.findData("density"))
    x.axis.setCurrentIndex(2)
    wait_for(lambda: not x._busy and x._req["field"] == "density" and x._req["axis"] == 2, 30)
    check("a scalar field offers no vectors", not x.vectors.isEnabled() and w.explore_view.canvas.arrows is None)
    x.cache_edit.setText(str(tmp / "tess"))
    x._cache_user_set = True
    w.server_action.trigger()                         # Explore > Start Query Server
    check("... which starts it (the menu item then cancels the start)",
          x._server_starting and w.server_action.text() == "Cancel Server Start")
    wait_for(lambda: not x._server_starting, 120)
    check("the query server starts on the output's own snapshot", x._server_running, x.server_state.text())
    check("... and the menu item now stops it", w.server_action.text() == "Stop Query Server"
          and w.server_action.isEnabled())
    img = w.explore_view.canvas._img
    x._click(img.height() // 2, img.width() // 3)
    wait_for(lambda: w.explore_view.streams.rowCount() > 0 or "⚠" in w.explore_view.answer_title.text(), 30)
    title = w.explore_view.answer_title.text()
    check("a click lists the streams at that point, each with its Lagrangian origin",
          "stream" in title and "Mpc" in title and w.explore_view.streams.rowCount() >= 1
          and w.explore_view.streams.item(0, 4).text().startswith("("), title)
    w.tabs.setCurrentIndex(A.GRIDS)
    qa.processEvents()
    check("off the Explore tab: no Save Image, but the running server can still be stopped",
          not w.save_image_action.isEnabled() and w.server_action.isEnabled())
    w.server_action.trigger()                         # Explore > Stop Query Server, from the Grids tab
    qa.processEvents()
    check("... which stops it", not x._server_running and w.server_action.text() == "Start Query Server"
          and not w.server_action.isEnabled())
    opened = []
    x._open_other = lambda: opened.append(w.tabs.currentIndex())    # (not the real, modal file dialog)
    w.open_output_action.trigger()
    del x._open_other
    check("Explore > Open Output… works from any tab: Explore comes up first",
          opened == [A.EXPLORE] and w.tabs.currentIndex() == A.EXPLORE and w.save_image_action.isEnabled())
    x.shutdown()

    print("failure advice:")
    bad = tmp / "not_a_snapshot.hdf5"
    bad.write_text("this is not HDF5")
    w.custom = rs.CustomSpec(input_file=str(bad), estimator="ps", grid=16, output_dir=str(tmp / "bad"),
                             output_name="bad")      # (PS-DTFE: the one binary every test setup builds)
    w._load_into_widgets()
    w.tabs.setCurrentIndex(A.OWN)
    qa.processEvents()
    w.run_btn.click()
    wait_idle(w, 120)
    check("a failed job shows what probably went wrong, and can be dismissed",
          not w.banner.isHidden() and "failed" in w.banner_text.text().lower(), w.banner_text.text())
    w.banner.hide()

    print("setup:")
    dlg = SW.SetupDialog(w, str(data))
    check("setup: four pages, the configured data folder proposed", dlg.pages.count() == 4
          and dlg.data_root() == str(data))
    dlg._go(1)
    check("setup: the programs page reports the built binaries with their build stamp",
          "PS-DTFE" in dlg.prog_info.text() and "built" in dlg.prog_info.text(), dlg.prog_info.text())
    keyfile = tmp / "api_key"
    SW.save_api_key("  test-key-123  ", keyfile)
    check("setup: the API key is saved owner-only (0600), trimmed",
          keyfile.read_text() == "test-key-123\n" and (keyfile.stat().st_mode & 0o777) == 0o600)
    new_root = tmp / "new data root"
    dlg.root.setText(str(new_root))
    for _ in range(3):
        dlg._go(1)
    check("setup: Finish creates the data folder and remembers the demo choice",
          dlg.result() == 1 and new_root.is_dir() and dlg.run_demo)


if __name__ == "__main__":
    sys.exit(main())
