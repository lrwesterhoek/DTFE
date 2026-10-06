#!/usr/bin/env python3
"""End-to-end check of the PySide6 GUI (python/gui/app.py), offscreen, on a synthetic per-family
data root: discovery and the command preview; REAL runs through the Run button of all three
tabs -- Grids (a two-snapshot PS-DTFE run: progress, log, exit summary, 'done' marks), Plots
(plot_PS_DTFE.py per snapshot, the figures appearing in the browser), Pipeline (plan only,
then grids + image plane + plot_pointeval.py) and Data (download through a FAKE wget that copies
local TNG-like chunks -- no network -- then merge_HDF5.py) -- and Stop killing the process tree.

Every figure goes into a temp folder (DTFE_FIGURES_ROOT, set before the launcher is imported and inherited
by the scripts it runs) -- never the real figures root, which is the T7 when it is mounted.

Needs PySide6 and the built ./PS-DTFE (make PS-DTFE); without either it reports SKIP and exits
0 -- unless DTFE_GUI_TEST_REQUIRED=1 (as in CI), where a skip is a failure, so a broken PySide6
install can never turn the suite silently green.

Usage: ~/.venvs/dtfe-gui/bin/python tests/py_gui_app_test.py
"""

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "gui"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["DTFE_GUI_NO_NOTIFY"] = "1"          # no desktop notifications from a test run
FIGS_TMP = tempfile.mkdtemp(prefix="gui_app_figs_")
os.environ["DTFE_FIGURES_ROOT"] = FIGS_TMP      # BEFORE runspec is imported: no test figure on the T7
atexit.register(shutil.rmtree, FIGS_TMP, True)
REQUIRED = os.environ.get("DTFE_GUI_TEST_REQUIRED") == "1"

try:
    from PySide6.QtCore import QEventLoop, QSettings, QTimer
    from PySide6.QtWidgets import QApplication
except ImportError as e:
    print(f"SKIP: PySide6 is not importable ({e}) -- python3 -m venv ~/.venvs/dtfe-gui "
          "--system-site-packages; ~/.venvs/dtfe-gui/bin/pip install PySide6")
    sys.exit(1 if REQUIRED else 0)

import app as A  # noqa: E402
import explore as E  # noqa: E402
import grids as G  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
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
            hd["Redshift"] = {50: 1.0, 67: 0.5, 99: 0.0}[n]
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
        check("DTFE mode disables the PS-only options (the hi-res slice works with both estimators) and switches script",
              not w.ps_group.isEnabled() and w.slice_group.isEnabled()
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

        print("pipeline options:")
        check("the run options show the scripts' defaults; cusps wait for caustics, the slab and projection for planes",
              w.p_deposit.currentData() == "sampled" and w.p_nsub.value() == 3 and w.p_nsub.isEnabled()
              and w.p_gpu.isChecked() and w.p_vm.isChecked() and w.p_vw.isChecked() and not w.p_ca.isChecked()
              and not w.p_cu.isEnabled() and not w.p_pt.isHidden() and w.p_planes.value() == 1
              and not w.p_thick.isEnabled() and w.p_super.value() == 1 and not w.p_rproject.isEnabled()
              and w.p_render_box.isEnabled() and w.p_rsmoothd.value() == rs.POINTEVAL_DERIVATIVE_SMOOTH)
        w.p_ca.setChecked(True)
        w.p_cu.setChecked(True)
        w.p_gpu.setChecked(False)
        w.p_nsub.setValue(2)
        w.p_planes.setValue(4)
        qa.processEvents()
        texts = lambda: [w.checks.item(i).text() for i in range(w.checks.count())]  # noqa: E731
        check("a handful of planes warns of ghosting; cusps on a GPU-less run pass the checks",
              any("ghosts" in t for t in texts()) and not any("cusps" in t for t in texts()), str(texts()))
        w.p_planes.setValue(16)
        w.p_thick.setValue(3.0)
        w.p_super.setValue(2)
        w.p_rfields.setText("density, streams")
        w.p_rfields.editingFinished.emit()
        w.p_rproject.setCurrentIndex(w.p_rproject.findData("slab"))
        w.p_rfixed.setChecked(True)
        qa.processEvents()
        env = w.pipe.command()[1]
        st = w.pipe.steps()
        check("the options reach the pipeline's environment, the render step and the estimate",
              w.p_cu.isEnabled() and w.p_thick.isEnabled() and w.p_rproject.isEnabled()
              and [env[k] for k in ("PS_CAUSTICS", "PS_CAUSTIC_CUSPS", "PS_GPU", "AVG_SUBSAMPLES", "PLANES", "THICKNESS",
                                    "SUPERSAMPLE")] == ["1", "1", "0", "2", "16", "3", "2"]
              and st[1].argv[st[1].argv.index("--fields") + 1] == "density,streams"
              and st[1].argv[st[1].argv.index("--project") + 1] == "slab" and "--fixed-range" in st[1].argv
              and "× 16 planes" in w.estimate.text() and f"{16 * 64 * 64 * 4 / 1e6:.1f}M points" in w.estimate.text(),
              f"{env} | {st[1].argv} | {w.estimate.text()}")
        mstep = w.pipe.check_step(SIM, [99], 64)
        check("the memory check of these settings carries them, sized by the planned points, without -m",
              "-m" not in mstep.argv and mstep.env["PS_CAUSTICS"] == "1" and mstep.env["AVG_SUBSAMPLES"] == "2"
              and mstep.env.get("DTFE_AUTO_PTS_N") == str(16 * 64 * 64 * 4), f"{mstep.argv} {mstep.env}")
        check("at 16 planes the ghosting warning is gone", not any("ghosts" in t for t in texts()), str(texts()))
        w.p_ca.setChecked(False)
        qa.processEvents()
        check("unticking caustics greys the (still ticked) cusps box without blocking the run",
              not w.p_cu.isEnabled() and w.p_cu.isChecked() and not w.pipe.caustic_cusps
              and not any("cusps" in t for t in texts()), str(texts()))
        w.p_ca.setChecked(True)
        qa.processEvents()
        check("... and ticking it again brings the cusps back", w.pipe.caustic_cusps and w.p_cu.isEnabled())
        w.p_deposit.setCurrentIndex(w.p_deposit.findData("exact"))
        qa.processEvents()
        check("the exact deposit ticks the GPU (as on the Grids tab) and greys the sub-samples",
              w.pipe.deposit == "exact" and (w.p_gpu.isChecked() or not rs.gpu_built("ps"))
              and not w.p_nsub.isEnabled() and w.pipe.command()[1]["PS_EXACT"] == "1")
        for cb in (w.p_ca, w.p_cu, w.p_rfixed):
            cb.setChecked(False)
        w.p_deposit.setCurrentIndex(w.p_deposit.findData("sampled"))
        w.p_gpu.setChecked(True)
        w.p_nsub.setValue(3)
        w.p_planes.setValue(1)
        w.p_super.setValue(1)
        w.p_rfields.setText("")
        w.p_rfields.editingFinished.emit()
        w.p_rproject.setCurrentIndex(0)
        qa.processEvents()
        check("back at the defaults: the same two steps, GPU on",
              [st.label for st in w.pipe.steps()] == [f"run_ps_pipeline.sh {SIM}", f"plot_pointeval.py {SIM}"]
              and w.pipe.command()[1]["PS_GPU"] == "1" and w.pipe.planes == 1 and w.pipe.render_fields == "")
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
        # a download that stopped early: the fixture serves 2 chunks of a snapshot whose header promises 3
        make_raw_chunks(fixtures / "snapshot-67", 67, chunks=3)
        (fixtures / "snapshot-67" / "snap_067.2.hdf5").unlink()
        w.d_table.item(rows[67], 0).setCheckState(w.d_table.item(rows[67], 0).checkState().Checked)
        qa.processEvents()
        steps67 = w.data.steps()
        check("snapshot 67 ticked: download + merge, the merge chained to its download", [st.label for st in steps67]
              == ["download snapshots 67", "merge snapshots 67"] and steps67[1].after == "download snapshots 67",
              str([(st.label, st.after) for st in steps67]))
        (sd / "snapdir_067").mkdir(parents=True, exist_ok=True)
        (sd / "snapdir_067" / "combined_067.hdf5.partial").write_text("stale")       # a merge stopped half-way before
        w.run_btn.click()
        wait_idle(w, 120)
        st67 = rs.snapshot_state(sd, 67)
        check("2 of the header's 3 chunks: the merge step FAILS (it merged what was there and called it done)",
              w.status.text() == "Finished: 1 of 2 steps failed", w.status.text())
        check("... no combined file, the chunks kept, the stale .partial gone and no new one",
              st67 == {"chunks": 2, "merged": False, "gc_chunks": 0, "gc_merged": False}
              and not list((sd / "snapdir_067").glob("*.partial")), str(st67))
        check("... the log names the missing chunk and the header's count",
              "snap_067.2.hdf5" in w.log.toPlainText() and "the header promises 3" in w.log.toPlainText(),
              w.log.toPlainText()[-500:])
        rows = {w.d_table.item(r, 0).data(0x0100): r for r in range(w.d_table.rowCount())}
        check("the table does not show it merged", "merged" not in w.d_table.item(rows[67], 2).text(),
              w.d_table.item(rows[67], 2).text())
        w.d_table.item(rows[67], 0).setCheckState(w.d_table.item(rows[67], 0).checkState().Unchecked)
        qa.processEvents()
        # a download that fails outright (no fixture: the fake wget exits 8): the merge is skipped, not run on nothing
        w.d_table.item(rows[33], 0).setCheckState(w.d_table.item(rows[33], 0).checkState().Checked)
        qa.processEvents()
        w.run_btn.click()
        wait_idle(w, 120)
        log33 = w.log.toPlainText()
        check("a failed download: its merge step is skipped and counted failed (half a download is never merged)",
              w.status.text() == "Finished: 2 of 2 steps failed" and "merge snapshots 33: skipped, 'download snapshots 33' failed" in log33
              and "Error downloading snapshot-33" in log33 and not rs.snapshot_state(sd, 33)["merged"],
              f"{w.status.text()} | {log33[-400:]}")
        w.d_table.item(rows[33], 0).setCheckState(w.d_table.item(rows[33], 0).checkState().Unchecked)
        qa.processEvents()
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
        w.p_nu.setCurrentText("1024")
        w.p_grid.setCurrentText("32")
        w.p_prefix.setText("ps_mem")
        w.p_prefix.editingFinished.emit()
        w.p_render.setChecked(False)
        qa.processEvents()
        # Planes large enough that their points, not the tessellation, set the difference (1024²: ~0.2
        # GB of point records, 512²: ~0.05). The budget must sit between the two sizes' SMALLEST
        # possible footprints: since 2026-10-02 the tuner splits a small particle set whose one domain
        # does not fit, so a budget between the unsplit predictions is met by a split at the larger
        # plane, and nothing would be halved. A budget just above the fixed part (the full grid and
        # the points, which no split reduces) makes the binary report that smallest split.
        steps = {nu: w.pipe.check_step(SIM, [99], nu) for nu in (1024, 512)}

        def report(nu, budget=None):
            env = dict(os.environ, **steps[nu].env)
            if budget is not None:
                env["DTFE_MEM_BUDGET_GB"] = budget
            r = subprocess.run(steps[nu].argv, env=env, capture_output=True, text=True, cwd=str(ROOT))
            return rs.parse_reports(r.stdout).get(99, {})

        fixed = max(report(nu).get("fixed_gb", 0.) for nu in steps)
        outs = {nu: report(nu, f"{fixed + 0.01:.4f}").get("predicted_gb") for nu in steps}
        check("a larger plane needs more memory, even split as finely as the tuner goes",
              outs[1024] and outs[512] and outs[1024] - outs[512] > 0.05, str(outs))
        os.environ["DTFE_MEM_BUDGET_GB"] = f"{(outs[1024] + outs[512]) / 2:.4f}"
        w.p_adaptive.setChecked(True)
        qa.processEvents()
        check("adaptive without a check is refused", not w.run_btn.isEnabled()
              and any("run 'Check memory' first" in t for t in lines()), str(lines()))
        w.mem_btn.click()
        wait_check(w, 300)
        check("the check halves the plane where it does not fit, and plans the fitting size",
              w.pipe.plan.get(SIM) == {"050": 512, "099": 512} and w.run_btn.isEnabled(), str(w.pipe.plan))
        check("the pipeline then runs at that size, on those snapshots",
              [(st.env.get("NU"), st.env.get("SNAPS")) for st in w.pipe.steps()] == [("512", "50 99")],
              str([(st.env.get("NU"), st.env.get("SNAPS")) for st in w.pipe.steps()]))
        w.p_adaptive.setChecked(False)
        qa.processEvents()
        check("without Adaptive the over-budget size is a warning that names the size that fits",
              any("at 1024² needs" in t and "512² fits" in t for t in lines()), str(lines()))
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
        vmin_w, vmax_w = w.opt_widgets[("panels", "vmin")][1], w.opt_widgets[("panels", "vmax")][1]
        vmax_w.setValue(2.3e5)
        vmin_w.setValue(0.003)
        check("the panels' density limits hold real densities (TNG100 z=0 peaks at 2.3e5; voids go below 0.01; the "
              "default float box stopped at 1000 and 0.01)", vmax_w.value() == 2.3e5 and abs(vmin_w.value() - 0.003) < 1e-9,
              f"{vmin_w.value()} {vmax_w.value()}")
        vmax_w.setValue(0.0)
        vmin_w.setValue(0.0)
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
        check("the failed job's note says how far it got, and how long it took",
              w.queue[2].note.startswith("0/1 steps OK, took ") and w.queue[2].note.endswith(" s"), w.queue[2].note)
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


def region_mean(x, r0, c0, r1, c1):
    """The slice's mean over the rectangle the test drags (image rows/cols -> plane cells, as the
    explore tab maps them), from the plane on screen now -- call it BEFORE the drag."""
    plane = x._plane
    n0, n1 = plane.shape
    rows = sorted((c0, c1))
    cols = sorted((n1 - 1 - r1, n1 - 1 - r0))
    return float(np.nanmean(plane[rows[0]:rows[1] + 1, cols[0]:cols[1] + 1]))


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
    check("... and leaves the GPU to the run's size (an unknown TNG-layout count: the GPU when built)",
          w.spec.gpu == rs.gpu_built("ps") and w.gpu_check.isChecked() == w.spec.gpu)
    w.pt_check.setChecked(True)
    qa.processEvents()
    tbb_note_ok = ("TBB" not in w.pt_check.text()) if rs.tbb_built() else ("TBB" in w.pt_check.text())
    check("the parallel-triangulation box sets the script's knob in the command, and names a build without TBB",
          w.spec.parallel_triangulation and "PS_PARALLEL_TRI=1" in w.cmd_view.toPlainText()
          and w.pt_check.isEnabled() and tbb_note_ok, w.pt_check.text())
    w.pt_check.setChecked(False)
    qa.processEvents()
    check("... and unticked it is off", "PS_PARALLEL_TRI=0" in w.cmd_view.toPlainText())
    w.prec_combo.setCurrentIndex(w.prec_combo.findData("double"))
    qa.processEvents()
    both_double = rs.binary_built("ps", "double") and rs.binary_built("dtfe", "double")
    lines_now = [w.checks.item(i).text() for i in range(w.checks.count())]
    check("precision: 'double' puts DTFE_PRECISION=double in the command, doubles the estimate, and blocks "
          "Run only when the pair is not built",
          w.spec.precision == "double" and "DTFE_PRECISION=double" in w.cmd_view.toPlainText()
          and "(float64)" in w.estimate.text()
          and (any("double-precision" in t for t in lines_now) != both_double), str(lines_now))
    w.prec_combo.setCurrentIndex(w.prec_combo.findData("single"))
    qa.processEvents()
    check("... back to single: the knob is gone", w.spec.precision == "single"
          and "DTFE_PRECISION" not in w.cmd_view.toPlainText())
    w.deposit_combo.setCurrentIndex(w.deposit_combo.findData("exact"))
    qa.processEvents()
    check("picking the exact deposit ticks the GPU (when built): the only practical way to run it",
          w.gpu_check.isChecked() == rs.gpu_built("ps") and w.spec.deposit == "exact")
    w.deposit_combo.setCurrentIndex(w.deposit_combo.findData("sampled"))
    w.gpu_check.setChecked(False)
    qa.processEvents()
    # standard DTFE: the same choice is its exact cell average (run_dtfe.sh -e), in its own words
    w.dtfe_radio.setChecked(True)
    qa.processEvents()
    w.gpu_check.setChecked(False)
    w.deposit_combo.setCurrentIndex(w.deposit_combo.findData("exact"))
    qa.processEvents()
    check("standard DTFE: the deposit choice stays available as its exact cell average, ticks the GPU (when "
          "built) and puts -e on the run_dtfe.sh line",
          w.deposit_combo.isEnabled() and w.deposit_combo.itemText(w.deposit_combo.findData("exact")) == P.label("exact_dtfe")
          and w.spec.estimator == "dtfe" and w.spec.deposit == "exact" and w.gpu_check.isChecked() == rs.gpu_built("dtfe")
          and any("run_dtfe.sh" in ln and " -e " in ln + " " for ln in w.cmd_view.toPlainText().splitlines())
          and not w.nsub_spin.isEnabled(), w.cmd_view.toPlainText())
    w.deposit_combo.setCurrentIndex(w.deposit_combo.findData("sampled"))
    w.ps_radio.setChecked(True)
    qa.processEvents()
    check("... and back on PS-DTFE the choice reads as the phase-space deposit again",
          w.deposit_combo.itemText(w.deposit_combo.findData("exact")) == P.label("exact") and w.spec.estimator == "ps")
    w.gpu_check.setChecked(False)
    qa.processEvents()
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
    keep_custom = rs.CustomSpec.from_dict(w.custom.to_dict())
    w.custom = rs.CustomSpec()
    w._load_into_widgets()
    w._load_run_settings(A.R.parse_runlog(runlog))
    check("Runs > Load settings of a custom run fills the Custom tab from its settings file",
          w.tabs.currentIndex() == A.OWN and Path(w.custom.input_file) == Path(demo.input_file)
          and w.custom.grid == demo.grid and w.custom.output_name == demo.output_name
          and Path(w.o_file.text()) == Path(demo.input_file), w.custom.input_file)
    w.custom = keep_custom
    w._load_into_widgets()
    check("a periodic box: no alpha-shape row", not w.o_snap_form.isRowVisible(w.o_alpha)
          and "ps-alpha-shape" not in w.cmd_view.toPlainText())
    w.o_periodic.setChecked(False)
    qa.processEvents()
    check("a non-periodic cloud: the alpha shape appears, at the default 3 (no flag emitted)",
          w.o_snap_form.isRowVisible(w.o_alpha) and w.o_alpha.value() == 3.0
          and "ps-alpha-shape" not in w.cmd_view.toPlainText())
    w.o_alpha.setValue(2.5)
    qa.processEvents()
    check("... another value reaches the command", w.custom.alpha_shape == 2.5
          and "--ps-alpha-shape 2.5" in w.cmd_view.toPlainText(), w.cmd_view.toPlainText())
    w.o_alpha.setValue(3.0)
    w.o_periodic.setChecked(True)
    qa.processEvents()
    w.o_gpu.setChecked(True)
    w._own_last_file = ""
    w.o_file.setText(str(demo.input_file))
    w._own_file_changed()
    qa.processEvents()
    check("choosing a file sets the GPU from its particle count (32³: the CPU deposit is faster)",
          not w.o_gpu.isChecked() and not w.custom.gpu)
    combo = w._preset_custom[0]
    combo.setCurrentIndex(combo.findData("Quick look"))
    w._apply_preset("custom")
    qa.processEvents()
    check("a preset on the custom target keeps the GPU to the file's size", w.custom.grid == 128 and not w.custom.gpu)
    combo.setCurrentIndex(combo.findData("Exact deposit"))
    w._apply_preset("custom")
    qa.processEvents()
    check("... unless it names the GPU (exact deposit)", w.custom.deposit == "exact" and w.custom.gpu
          and w.o_gpu.isChecked())
    w.custom = rs.demo_spec(tmp / "demo", gpu=False)
    w._load_into_widgets()
    w._changed()
    qa.processEvents()
    check("the custom tab offers 'Check memory' for an existing PS snapshot file",
          w.mem_btn.isVisible() and w.mem_btn.isEnabled() and w.mem_action.isEnabled())
    w.mem_btn.click()
    qa.processEvents()
    wait_check(w, 120)
    lines = [w.checks.item(i).text() for i in range(w.checks.count())]
    check("... and the binary's report says it fits, with the auto-tuner's split",
          any("memory: fits" in t and "auto-tuner plans" in t for t in lines), str(lines))

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
    sims = [x.sim_combo.itemText(i) for i in range(x.sim_combo.count())]
    check("the Output menus: the simulation list (natural order, then the custom snapshots), the demo under "
          "'Custom snapshots' by its run name, no third menu for a single output",
          sims == [SIM, "Custom snapshots"] and x.sim_combo.currentData() == E.GROUP_CUSTOM
          and x._out_form.labelForField(x.snap_combo).text() == "Run"
          and x.snap_combo.currentText().startswith(Path(root).name) and x.snap_combo.currentData() == str(root)
          and not x._out_form.isRowVisible(x.prefix_combo), f"{sims} {x.snap_combo.currentText()}")
    x.sim_combo.setCurrentIndex(x.sim_combo.findData(SIM))
    ok = wait_for(lambda: not x._busy and x.current is not None and x.current.sim == SIM, 30)
    zs = [x.snap_combo.itemText(i) for i in range(x.snap_combo.count())]
    names99 = [x.prefix_combo.itemText(i) for i in range(x.prefix_combo.count())]
    check("choosing the simulation lists its snapshots by redshift, earliest first, and shows the latest (the "
          "production prefix ps_output when that snapshot has it, else its only one)",
          ok and x._out_form.labelForField(x.snap_combo).text() == "Redshift"
          and zs == ["z = 0.00   ·   snapshot 050", "z = 0.00   ·   snapshot 099"]
          and x.snap_combo.currentData() == 99 and x.current.snap == 99
          and x.current.prefix == ("ps_output" if "ps_output" in names99 else "ps_gui")
          and x._out_form.isRowVisible(x.prefix_combo) == (len(names99) > 1)
          and "z = 0.00" in x.info.text(), f"{zs} {names99} {x.current and x.current.root}")
    x.snap_combo.setCurrentIndex(0)
    ok = wait_for(lambda: not x._busy and x.current is not None and x.current.snap == 50, 30)
    check("choosing a redshift shows that snapshot's output", ok and x.current.dir.name == "snapdir_050"
          and w.explore_view.canvas._img is not None, str(x.current and x.current.root))
    w._explore_output(str(root))
    ok = wait_for(lambda: not x._busy and x.current is not None and str(x.current.root) == str(root), 30)
    check("Runs > Explore (select_root) sets all the menus back to the demo", ok
          and x.sim_combo.currentData() == E.GROUP_CUSTOM and x.snap_combo.currentData() == str(root)
          and x.prefix_combo.currentData() == str(root))
    texts = sims + zs + [x._out_form.labelForField(x.snap_combo).text(), x.sim_combo.toolTip(), x.snap_combo.toolTip(),
                         x.prefix_combo.toolTip()]
    check("no '--' in the Output menus' texts", not [t for t in texts if "--" in t])
    # a second output of the same snapshot: the third menu appears; the chosen prefix survives a refresh
    # and stepping through the snapshots; one output again hides the menu
    x.sim_combo.setCurrentIndex(x.sim_combo.findData(SIM))
    wait_for(lambda: not x._busy and x.current is not None and x.current.sim == SIM and x.current.snap == 99, 30)
    s99 = sim / "snapdir_099"
    for sfx in ("a_den", "a_streams", "a_hidden_streams"):
        if (s99 / f"ps_gui.{sfx}").is_file():
            shutil.copy(s99 / f"ps_gui.{sfx}", s99 / f"ps_dup.{sfx}")
    i_before, pref_before = x.index.value(), x.current.prefix
    x.refresh_outputs()
    ok = wait_for(lambda: not x._busy and x.current is not None and x.current.snap == 99, 30)
    prefixes = [x.prefix_combo.itemText(i) for i in range(x.prefix_combo.count())]
    check("a snapshot with several outputs shows the third menu; a refresh keeps the output and the slice shown",
          ok and x._out_form.isRowVisible(x.prefix_combo) and "ps_dup" in prefixes and len(prefixes) >= 2
          and prefixes == sorted(prefixes) and x.current.prefix == pref_before and x.index.value() == i_before,
          f"{prefixes} {x.current and x.current.prefix} (before: {pref_before})")
    x.prefix_combo.setCurrentIndex(x.prefix_combo.findData(str(s99 / "ps_dup")))
    wait_for(lambda: not x._busy and x.current is not None and x.current.prefix == "ps_dup", 30)
    x.snap_combo.setCurrentIndex(0)                     # snapshot 050 (no ps_dup there)
    wait_for(lambda: not x._busy and x.current is not None and x.current.snap == 50, 30)
    menu_ok_050 = x._out_form.isRowVisible(x.prefix_combo) == (x.prefix_combo.count() > 1) and x.current.prefix != "ps_dup"
    x.snap_combo.setCurrentIndex(1)                     # back to 099
    ok = wait_for(lambda: not x._busy and x.current is not None and x.current.snap == 99, 30)
    check("the chosen output prefix is kept when stepping through the snapshots (the menu shows exactly where "
          "there are several)", ok and menu_ok_050 and x.current.prefix == "ps_dup"
          and x._out_form.isRowVisible(x.prefix_combo), str(x.current and x.current.root))
    for p_ in s99.glob("ps_dup.*"):
        p_.unlink()
    x.refresh_outputs()
    ok = wait_for(lambda: not x._busy and x.current is not None and x.current.snap == 99 and x.current.prefix != "ps_dup", 30)
    check("... its files gone, a refresh falls back to a remaining output (the menu stays only for several)",
          ok and x._out_form.isRowVisible(x.prefix_combo) == (x.prefix_combo.count() > 1)
          and "ps_dup" not in [x.prefix_combo.itemText(i) for i in range(x.prefix_combo.count())],
          str(x.current and x.current.root))
    n_before = len(x.outputs)
    ok = x.open_file(str(root) + ".a_den")
    check("opening a grid file of a listed output selects it in place: no 'Opened files' entry, nothing listed twice",
          ok and len(x.outputs) == n_before and x.sim_combo.findData(E.GROUP_OPENED) < 0
          and str(x.current.root) == str(root) and x.sim_combo.currentData() == E.GROUP_CUSTOM)
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
    i_before = x.index.value()
    x.refresh_outputs()                                # new OutputSet objects for the same files
    ok = wait_for(lambda: not x._busy and x.current is not None and str(x.current.root) == str(root), 30)
    w.explore_view.streams.setRowCount(0)
    w.explore_view.answer_title.setText("")
    x._click(img.height() // 2, img.width() // 3)
    wait_for(lambda: "asking" not in w.explore_view.answer_title.text()
             and (w.explore_view.streams.rowCount() > 0 or "⚠" in w.explore_view.answer_title.text()
                  or "Start the query server" in w.explore_view.answer_title.text()), 30)
    check("looking for outputs again keeps the slice and the running server's answers",
          ok and x.index.value() == i_before and x._server_running and w.explore_view.streams.rowCount() >= 1
          and "Start the query server" not in w.explore_view.answer_title.text(), w.explore_view.answer_title.text())
    print("map axes, colour bar, difference map:")
    from PySide6.QtGui import QFontMetrics
    cv, cb, o = w.explore_view.canvas, w.explore_view.colorbar, x.current
    keep_field = x.field.currentData()
    rest_ = [a for a in range(3) if a != x.axis.currentIndex()]
    ax_, t_ = cv.axes, cv._target()
    check("the map has axes in its output's frame (Mpc with a box, else cells), the image inside their margins",
          ax_ is not None and abs(ax_["x"][0] - o.lo[rest_[0]]) < 1e-9 and abs(ax_["y"][1] - (o.lo + o.length)[rest_[1]]) < 1e-9
          and ax_["unit"] == ("Mpc" if o.box is not None else "cells") and t_.left() > 20 and t_.bottom() < cv.height() - 20,
          f"{ax_} {t_}")
    fm = QFontMetrics(cb.font())
    spans = sorted((l_, l_ + fm.horizontalAdvance(tx)) for _, l_, tx in cb.tick_labels(400))
    check("the colour bar: round ticks plus its two ends, no two labels overlapping, the quantity on a row of its own",
          len(spans) >= 3 and all(b[0] >= a[1] + 6 for a, b in zip(spans, spans[1:])) and cb.height() >= 14 + 2 * fm.height(),
          str(cb.tick_labels(400)))
    cb.set("magma", (-1.2, 3.7), "log₁₀ density", True)
    texts = [tx for _, _, tx in cb.tick_labels(600)]
    check("... a log map's ticks are whole decades written 10^k, the bar then drops 'log₁₀ ' (Save image keeps it)",
          "10²" in texts and "1" in texts and cb.shown_label() == "density" and cb.label == "log₁₀ density", str(texts))
    twin = o.dir / "cmp2"                              # the same cells, its density doubled: A - B = -A exactly
    made = []
    for f_ in o.dir.glob(o.prefix + ".*"):
        dst_ = o.dir / ("cmp2" + f_.name[len(o.prefix):])
        if f_.name.endswith(".a_den"):
            (np.fromfile(f_, dtype=o.dtype) * 2).astype(o.dtype).tofile(dst_)
        elif f_.suffix not in (".runlog",):
            shutil.copy(f_, dst_)
        else:
            continue
        made.append(dst_)
    try:
        x.refresh_outputs()
        wait_for(lambda: not x._busy and x.current is not None and str(x.current.root) == str(root), 30)
        x.field.setCurrentIndex(x.field.findData("density"))
        wait_for(lambda: not x._busy, 30)
        listed = [x.compare.itemData(i) for i in range(1, x.compare.count())]
        check("'Compare with' lists the other runs on the same cells, and only those",
              str(twin) in listed and all(E.ExploreControls.comparable(x.current, x._by_root[r_], "density") for r_ in listed),
              str(listed))
        x.compare.setCurrentIndex(x.compare.findData(str(twin)))
        ok = wait_for(lambda: not x._busy and x._planes is not None, 30)
        a_, b_ = x._planes if x._planes is not None else (None, None)
        lo_, hi_ = cb.vrange
        check("A − B: the cell-by-cell difference on a 0-centred RdBu_r scale; the colour controls are off; the "
              "bar, the title and the hover say A − B / A and B",
              ok and np.allclose(b_, 2 * a_, rtol=1e-6) and np.allclose(x._plane, a_ - b_) and x._req["cmap"] == "RdBu_r"
              and not x._req["log"] and abs(lo_ + hi_) <= 1e-9 * max(1.0, abs(hi_)) and not x.cmap.isEnabled()
              and not x.log.isEnabled() and cb.label.startswith("A − B") and "B = " in w.explore_view.title.text(),
              f"{cb.label} {cb.vrange} {w.explore_view.title.text()}")
        x._hover(cv._img.height() // 2, cv._img.width() // 2)
        check("... the hover reads A and B beside the difference", "(A = " in w.explore_view.readout.text(),
              w.explore_view.readout.text())
        x.compare_mode.setCurrentIndex(x.compare_mode.findData("ratio"))
        ok = wait_for(lambda: not x._busy and x._req.get("compare_mode") == "ratio", 30)
        pos = x._planes[0] > 0
        img_nan = ~np.isfinite(x._plane.T[::-1])
        check("log₁₀ A/B: −log₁₀ 2 wherever A > 0, NaN elsewhere -- drawn grey, not as the scale's end",
              ok and np.allclose(x._plane[pos], -np.log10(2.0), atol=1e-5) and np.isnan(x._plane[~pos]).all()
              and (not img_nan.any() or (cv._buf[img_nan] == E.NAN_GREY).all()), f"{int((~pos).sum())} empty cells")
        signed = next((f for f in ("velocity", "divergence", "vorticity") if f in x.current.files), None)
        if signed:
            x.field.setCurrentIndex(x.field.findData(signed))
            wait_for(lambda: not x._busy, 30)
            check(f"a signed field ({signed}) offers A − B only", not x.compare_mode.model().item(1).isEnabled()
                  and x.compare_mode.currentData() == "diff")
        x.compare.setCurrentIndex(0)
        ok = wait_for(lambda: not x._busy and x._planes is None, 30)
        check("compare off: the plain map and its colour controls back", ok and x.cmap.isEnabled() and x.log.isEnabled()
              and "A − B" not in cb.label)
        x.field.setCurrentIndex(x.field.findData("density"))
        wait_for(lambda: not x._busy, 30)
        x.vmin_edit.setText("0.5")
        x.vmax_edit.setText("4")
        x.fixed.setChecked(True)
        ok = wait_for(lambda: not x._busy and x._req.get("vrange") is not None, 30)
        want = (np.log10(0.5), np.log10(4.0)) if x.log.isChecked() else (0.5, 4.0)
        check("a fixed colour range: the map and its bar on exactly that range (log10 on a log map), the clip off",
              ok and np.allclose(cb.vrange, want) and not x.clip.isEnabled(), f"{cb.vrange} {want}")
        st = x.state()
        x.fixed.setChecked(False)
        wait_for(lambda: not x._busy, 30)
        back = x.restore_state(st)
        wait_for(lambda: not x._busy, 30)
        check("Explore's state round-trips (the output, field, slice and the fixed range: what a restart restores)",
              back and x.fixed.isChecked() and x.vmin_edit.text() == "0.5" and x.field.currentData() == "density"
              and x.index.value() == st["index"], str(st))
        npy, csv = tmp / "slice_values.npy", tmp / "slice_values.csv"
        x.save_values(npy)
        x.save_values(csv)
        arr, lines = np.load(npy), csv.read_text().splitlines()
        check("Save image… also writes the values: .npy as drawn (rows from the bottom), .csv x, y, value per cell",
              arr.shape == x._plane.T.shape and np.allclose(arr, x._plane.T) and lines[0].endswith(",value")
              and len(lines) == 1 + x._plane.size, f"{arr.shape} {lines[0]} {len(lines)}")
        x.fixed.setChecked(False)
        wait_for(lambda: not x._busy, 30)
    finally:
        for f_ in made:
            f_.unlink(missing_ok=True)
        x.refresh_outputs()
        wait_for(lambda: not x._busy and x.current is not None and str(x.current.root) == str(root), 30)
        x.field.setCurrentIndex(x.field.findData(keep_field))
        wait_for(lambda: not x._busy, 30)
    # a point-evaluated hi-res slice ('.pts_den' + its plane sidecar) in place of the grid slice
    import json as _json
    pts_den, side_ = Path(str(root) + ".pts_den"), Path(str(root)).parent / "pointeval_plane_16_z.json"
    np.arange(16 * 16, dtype=np.float64).tofile(pts_den)            # the value at (v, u) is 16 v + u
    side_.write_text(_json.dumps({"nu": 16, "nv": 16, "planes": 1, "supersample": 1, "axis": "z", "u_axis": "x",
                                  "v_axis": "y", "u0": 0, "u1": 100, "v0": 0, "v1": 100, "center": 50.0, "box": 100.0}))
    keep_axis = x.axis.currentIndex()
    try:
        x.field.setCurrentIndex(x.field.findData("density"))
        wait_for(lambda: not x._busy, 30)
        x.axis.setCurrentIndex(0)               # the menu at x: a z-plane labelled from the menu would read y, z
        wait_for(lambda: not x._busy, 30)
        x._pts_fill()
        j = x.pts_combo.findData(str(side_))
        x.pts_combo.setCurrentIndex(max(j, 0))
        ok = wait_for(lambda: not x._busy and bool(x._req.get("pts")) and x._pts_side is not None, 30)
        check("a hi-res slice: listed when its sidecar matches, shown in place of the grid slice (plane[u, v], "
              "its own Mpc axes; the slice axis and position greyed)",
              ok and j > 0 and x._plane.shape == (16, 16) and np.allclose(x._plane, np.arange(256.0).reshape(16, 16).T)
              and "hi-res slice" in w.explore_view.title.text() and w.explore_view.canvas.axes["x"][:2] == (0.0, 100.0)
              and not x.axis.isEnabled(), w.explore_view.title.text())
        x._hover(0, 0)
        check("... the hover names the hi-res point at the plane's position", "hi-res point" in w.explore_view.readout.text()
              and "z = 50.00" in w.explore_view.readout.text(), w.explore_view.readout.text())
        running, x._server_running = x._server_running, False     # the server runs here: _click reads only this
        try:
            x._click(0, 0)
        finally:
            x._server_running = running
        check("... a click with no server names the hi-res point, not a grid cell",
              "hi-res point" in w.explore_view.answer_title.text(), w.explore_view.answer_title.text())
        import types as _types
        from matplotlib.axes import Axes as _Axes
        labels, png = [], tmp / "pts_map.png"
        orig_x, orig_y, orig_dlg = _Axes.set_xlabel, _Axes.set_ylabel, E.QFileDialog.getSaveFileName
        _Axes.set_xlabel = lambda self_, s, *a, **k: (labels.append(s), orig_x(self_, s, *a, **k))[1]
        _Axes.set_ylabel = lambda self_, s, *a, **k: (labels.append(s), orig_y(self_, s, *a, **k))[1]
        E.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(png), ""))
        try:
            x._save_image()
        finally:
            _Axes.set_xlabel, _Axes.set_ylabel, E.QFileDialog.getSaveFileName = orig_x, orig_y, orig_dlg
        check("... Save image labels the plane's own axes (x, y of a z-plane), not the greyed menu's (y, z)",
              png.is_file() and [s.split()[0] for s in labels[:2]] == ["x", "y"], str(labels))
        late = {"output": x.current, "axis": 2, "index": 0, "a0": (20.0, 30.0), "a1": (40.0, 50.0), "res": 8,
                "depth": 1, "memo": True, "seconds": 0.0}
        x._zoomed(late, _types.SimpleNamespace(density=np.ones((8, 8, 1)), streams=None, caustic=None, velocity=None,
                                               dispersion=None, sigma=None, velocity_gradient=None, offsets=None))
        check("... a zoom that lands while the hi-res slice shows is kept in memory, not drawn in the slice's frame",
              not x.is_zoomed() and w.explore_view.canvas.axes["x"][:2] == (0.0, 100.0), str(w.explore_view.canvas.axes))
        x.pts_combo.setCurrentIndex(0)
        ok = wait_for(lambda: not x._busy and not x._req.get("pts"), 30)
        check("... and off again: the grid slice", ok and x._plane.shape != (16, 16) or x.current.n == 16)
    finally:
        pts_den.unlink(missing_ok=True)
        side_.unlink(missing_ok=True)
        x._pts_fill()
        x.axis.setCurrentIndex(keep_axis)
        x.field.setCurrentIndex(x.field.findData(keep_field))
        wait_for(lambda: not x._busy, 30)
    # the scalar field in a zoom (an exact zoom of a run with a scalar dataset), the server's resident partitions,
    # and the server button after a restore (refresh_outputs judged it before the snapshots were back)
    import types as _types
    saved_zoom = (x._zoom, x._zoom_fields)
    x._zoom = {"axis": 2, "output": x.current}
    x._zoom_fields = _types.SimpleNamespace(density=np.ones((4, 4, 1)), scalar=np.full((4, 4, 1), 7.0))
    sp = x._zoom_plane("scalar", "")
    x._zoom, x._zoom_fields = saved_zoom
    sroot = tmp / "scalar_zoom"
    np.ones(8, dtype=np.float32).tofile(str(sroot) + ".a_den")
    np.full(8, 3.0, dtype=np.float32).tofile(str(sroot) + ".a_scalar")
    ez = E.load_exact_planes(sroot, (2, 2, 2))
    check("a zoom shows the scalar field (the exact zoom's '.a_scalar' is read; the server's answer has none)",
          sp is not None and np.all(sp == 7.0) and getattr(ez, "scalar", None) is not None and np.all(ez.scalar == 3.0))
    print("figure grid:")
    cur_root, cur_field = str(x.current.root), x.field.currentData()
    x.grid_set_layout(1, 2)
    x.grid_fill("fields")
    rows = [x.grid_row(i) for i in range(2)]
    check("'this snapshot across fields' fills a 1 × 2 grid with the map's output, starting at its field",
          all(r is not None and str(r[0].root) == cur_root for r in rows) and rows[0][1] == cur_field
          and rows[1][1] != cur_field, str([(str(r[0].root), r[1], r[2]) if r else None for r in rows]))
    x.grid_toggle.setChecked(True)
    ok = wait_for(lambda: x._grid_shown and x._grid_index is not None and not x._grid_busy, 30)
    check("the grid's tiles come from the worker and replace the map (two fields: no colour bar below; the title "
          "names it; the menu offers the way back)", ok and w.explore_view.canvas._img is not None
          and not w.explore_view.colorbar.isVisible() and "Figure grid" in w.explore_view.title.text()
          and w.grid_action.text() == "Back to the Map" and w.explore_view.canvas.labels, x.grid_state.text())
    gpng, gpdf = tmp / "grid.png", tmp / "grid.pdf"
    x.save_grid(str(gpng))
    x.save_grid(str(gpdf))
    check("Save figure writes the grid as PNG and as PDF", gpng.is_file() and gpng.stat().st_size > 20_000
          and gpdf.is_file() and gpdf.stat().st_size > 2_000, f"{gpng.stat().st_size if gpng.is_file() else 0}")
    x._dragged(1, 1, 5, 5)
    check("a drag on the picture does nothing (the grid has no cells under the mouse)", not x.is_zoomed() and x._grid_shown)
    # the worker's tiles: resampled before colouring, the same pixels as colouring the whole plane first
    import grids as Gm
    req0 = x.grid_request()
    items = E.grid_items(req0)
    n0 = items[0]["img"].shape[0]
    rng = np.random.default_rng(3)
    items[0]["img"] = np.where(rng.random((2 * n0, 2 * n0)) < 0.1, 0.0, rng.random((2 * n0, 2 * n0)) * 5)
    items[0]["img"][::7, ::5] = np.nan                            # a log panel with empty cells and NaNs,
    items[0]["log"] = True                                         # twice the size of the other
    items[1]["img"] = -np.abs(rng.random((n0, n0)))                # no positive value at all (floor 1.0)
    items[1]["log"] = True
    T = n0 // 2
    rgb, _labels, _bar = E.grid_tiles(req0, items, max_tile=T)
    same = True
    for k, it in enumerate(items):                                 # the old order, inline: colour, then resample
        old, _ = Gm.colorize(it["img"], it["cmap"], log=it["log"], clip=(req0["clip"], 100 - req0["clip"]),
                             symmetric=it["symmetric"], vrange=(it["lo"], it["hi"]))
        ri = (np.arange(T) * old.shape[0]) // T
        ci = (np.arange(T) * old.shape[1]) // T
        gap = 0 if req0["merged"] else 6
        tile = rgb[0:T, k * (T + gap):k * (T + gap) + T]
        same = same and np.array_equal(tile, old[ri][:, ci])
    check("grid_tiles resamples before colouring: pixels identical to colouring the full plane first (log panels "
          "with NaN, empty cells, and one with no positive value; mixed sizes)", same and rgb.shape[0] == T)
    # a failed compose with a newer request waiting: the newer one is sent, the grid stays
    x._grid_pending = x.grid_request()
    x._grid_failed("simulated")
    stayed = x._grid_shown and x._grid_busy and x._grid_pending is None
    ok = wait_for(lambda: x._grid_shown and not x._grid_busy, 30)
    check("a failed compose with a newer request pending sends that request and keeps the grid",
          stayed and ok and x._grid_shown and not x.grid_state.text().startswith("⚠"), x.grid_state.text())
    x._grid_failed("simulated")
    ok = wait_for(lambda: not x._grid_shown and x._plane is not None and not x._busy, 30)
    check("a failed compose with nothing pending leaves the grid for the map and says why",
          ok and not x.grid_toggle.isChecked() and "simulated" in x.grid_state.text(), x.grid_state.text())
    x.grid_toggle.setChecked(True)
    ok = wait_for(lambda: x._grid_shown and x._grid_index is not None and not x._grid_busy, 30)
    check("the grid comes back on request", ok)
    x.field.setCurrentIndex((x.field.currentIndex() + 1) % x.field.count())
    ok = wait_for(lambda: x._plane is not None and not x._busy and not x._grid_shown, 30)
    check("changing the map brings the map back: the grid is off, its toggle too, the colour bar shows",
          ok and not x.grid_toggle.isChecked() and w.explore_view.colorbar.isVisible()
          and w.explore_view.streams.isVisible() and w.grid_action.text() == "Figure Grid")
    x.field.setCurrentIndex(x.field.findData(cur_field))
    wait_for(lambda: x._plane is not None and not x._busy, 30)
    # across redshifts: on the TNG-layout simulation (two snapshots with the same prefix), then back to the demo
    x.sim_combo.setCurrentIndex(x.sim_combo.findData(SIM))
    wait_for(lambda: not x._busy and x.current is not None and x.current.sim == SIM, 30)
    x.grid_set_layout(2, 1)
    x.grid_fill("redshifts")
    rows = [x.grid_row(i) for i in range(2)]
    fld = x.field.currentData()
    check("'this field across redshifts' lists this simulation's snapshots oldest first, the same field and prefix",
          [r[0].snap for r in rows if r] == [50, 99] and {r[1] for r in rows if r} == {fld}
          and len({r[0].prefix for r in rows if r}) == 1, str([(str(r[0].root), r[1]) if r else None for r in rows]))
    x.grid_toggle.setChecked(True)
    ok = wait_for(lambda: x._grid_shown and x._grid_index is not None and not x._grid_busy, 30)
    check("a 2 × 1 grid of two redshifts shows, one quantity: the colour bar below with the shared range",
          ok and w.explore_view.canvas._img is not None and w.explore_view.colorbar.isVisible(), x.grid_state.text())
    check("two snapshots of one box: 'merge the axes' is offered, the shared range stays a free choice",
          x.grid_merge.isEnabled() and x.grid_shared.isEnabled())
    x.grid_merge.setChecked(True)
    ok = wait_for(lambda: x._grid_shown and not x._grid_busy, 30)
    n_axes = len(x.grid_figure()[0].axes)
    check("merged: the saved figure has ONE colour bar (2 panels + 1 bar axes), the shared range implied (ticked, "
          "greyed)", ok and n_axes == 3 and x.grid_shared.isChecked() and not x.grid_shared.isEnabled(), str(n_axes))
    mpng = tmp / "grid_merged.png"
    x.save_grid(str(mpng))
    check("the merged figure saves", mpng.is_file() and mpng.stat().st_size > 20_000)
    i0 = x.index.value()
    x.step_slice(1)                                   # the → key
    ok = wait_for(lambda: x._grid_shown and x._grid_index == i0 + 1 and not x._grid_busy, 10)
    check("→ with the grid shown steps the slice for every panel: new tiles at the new position, the grid stays, "
          "the title says where", ok and x.index.value() == i0 + 1
          and x.position.text() in w.explore_view.title.text() and not x.grid_toggle.text().startswith("Show"),
          f"{ok} {x.index.value()} {x._grid_index} {w.explore_view.title.text()}")
    img1 = w.explore_view.canvas._img
    x.cmap.setCurrentIndex((x.cmap.currentIndex() + 1) % x.cmap.count())
    ok = wait_for(lambda: x._grid_shown and w.explore_view.canvas._img is not img1 and not x._grid_busy, 10)
    check("a colour change keeps the grid and recolours the tiles", ok and x._grid_shown)
    x.step_slice(-1)
    wait_for(lambda: x._grid_index == i0 and not x._grid_busy, 10)
    # the composer's columns: the output split into simulation, redshift and type (DTFE / PS-DTFE / a prefix)
    hdr = [x.grid_table.horizontalHeaderItem(c).text() for c in range(x.grid_table.columnCount())]
    r0 = [x.grid_table.cellWidget(0, c) for c in range(5)]
    check("the composer's rows: Simulation, Redshift, Type, Field, Component -- the output in three menus keyed as "
          "the Output menus are", hdr == list(E.GRID_COLS) and r0[0].currentData() == SIM and r0[1].currentData() == 50
          and r0[2].currentText() == G.type_label(x._by_root[r0[2].currentData()])
          and r0[2].currentData(0x0101) == x.grid_row(0)[0].prefix and r0[2].toolTip().endswith(".*"),
          f"{hdr} {[w_.currentText() for w_ in r0]} {r0[2].toolTip()!r}")
    prev_root = str(x.current.root)
    s50 = sim / "snapdir_050"
    std = s50 / "output"                                  # a standard-DTFE output beside the phase-space ones
    np.full(32 ** 3, 2.0, dtype=np.float32).tofile(str(std) + ".a_den")
    np.full(32 ** 3, 2.0, dtype=np.float32).tofile(str(s99 / "output") + ".a_den")     # ... at both snapshots
    alt_files = []                                        # ps_alt: a phase-space run at both snapshots, never the preferred prefix
    for sd in (s50, s99):
        for f_ in sd.glob("ps_gui.*"):
            if f_.suffix not in (".runlog", ".json"):
                shutil.copy(f_, sd / f_.name.replace("ps_gui.", "ps_alt."))
                alt_files.append(sd / f_.name.replace("ps_gui.", "ps_alt."))
    if not (s50 / "ps_output.a_den").exists():            # the production prefix at 050 for certain (type_order's 2nd key);
        for f_ in s50.glob("ps_gui.*"):                   # never over what the pipeline left there
            dst_ = s50 / f_.name.replace("ps_gui.", "ps_output.")
            if f_.suffix not in (".runlog", ".json") and not dst_.exists():
                shutil.copy(f_, dst_)
                alt_files.append(dst_)
    x.refresh_outputs()
    x.select_root(str(s50 / "ps_q2"))                     # a prefix that sorts AFTER ps_gui: the fill must keep the map's own run
    ok = wait_for(lambda: x._plane is not None and not x._busy and x.current is not None and x.current.prefix == "ps_q2", 30)
    x.grid_set_layout(1, 2)
    x.grid_fill("types")
    rows = [x.grid_row(i) for i in range(2)]
    typ0 = x.grid_table.cellWidget(0, E._C_TYPE)
    labels = [typ0.itemText(k) for k in range(typ0.count())]
    # the whole menu in type_order (review 2026-10-05 [38]): DTFE, the production run, the preferred prefix (ps_q2),
    # then the rest by name -- built from what is listed at 050, so a run an earlier section left there still fits
    at50 = sorted({o_.prefix for _t, o_ in x.outputs if o_.sim == SIM and o_.snap == 50})
    want_types = ["DTFE", "PS-DTFE", "PS-DTFE · ps_q2"] + [f"PS-DTFE · {p}" for p in at50
                                                          if p not in ("output", "ps_output", "ps_q2")]
    check("'this snapshot across outputs': the map's own run (ps_q2, not the alphabetically first ps_*) and the standard "
          "DTFE of that snapshot side by side, DTFE first; the Type menu lists every output of that snapshot, in order: "
          "DTFE, the production PS-DTFE, the preferred ps_q2, then the rest by name",
          ok and [r[0].prefix for r in rows if r] == ["output", "ps_q2"] and labels == want_types and typ0.isEnabled(),
          f"{[r[0].prefix for r in rows if r]} {labels} want {want_types} (preferred {x._pref_prefix})")
    # type_order's 2nd key: from the standard DTFE the other estimator's pick is its DEFAULT prefix (ps_output), not
    # the first by name (ps_alt); the preferred prefix is now 'output' (picked by name), so only that key can say so
    x.select_root(str(std))
    ok_std = wait_for(lambda: x._plane is not None and not x._busy and x.current is not None and x.current.prefix == "output", 30)
    x.grid_set_layout(1, 2)
    x.grid_fill("types")
    std_rows = [x.grid_row(i) for i in range(2)]
    check("... from the standard DTFE: the production PS-DTFE run (ps_output) beside it, not the first by name (ps_alt)",
          ok_std and [r[0].prefix for r in std_rows if r] == ["output", "ps_output"],
          f"{[r[0].prefix for r in std_rows if r]} (preferred {x._pref_prefix})")
    # ... and the production run stays ABOVE a preferred prefix that sorts before it by name (ps_alt < ps_output): ps_q2
    # sorts after it, so one key ranking both alike would pass the menu check above
    x.select_root(str(s50 / "ps_alt"))
    ok_alt = wait_for(lambda: x._plane is not None and not x._busy and x.current is not None and x.current.prefix == "ps_alt", 30)
    x.grid_set_layout(1, 2)
    x.grid_fill("types")
    alt_rows = [x.grid_row(i) for i in range(2)]
    typ0_alt = x.grid_table.cellWidget(0, E._C_TYPE)
    alt_labels = [typ0_alt.itemText(k) for k in range(typ0_alt.count())]
    want_alt = ["DTFE", "PS-DTFE", "PS-DTFE · ps_alt"] + [f"PS-DTFE · {p}" for p in at50
                                                         if p not in ("output", "ps_output", "ps_alt")]
    check("... from ps_alt (a preferred prefix that sorts BEFORE ps_output): the Type menu still reads DTFE, the "
          "production PS-DTFE, then ps_alt, then the rest by name",
          ok_alt and [r[0].prefix for r in alt_rows if r] == ["output", "ps_alt"] and alt_labels == want_alt,
          f"{[r[0].prefix for r in alt_rows if r]} {alt_labels} want {want_alt} (preferred {x._pref_prefix})")
    x.select_root(str(s50 / "ps_q2"))                     # back to the ps_q2 map and its fill: the steps below use both
    ok = wait_for(lambda: x._plane is not None and not x._busy and x.current is not None and x.current.prefix == "ps_q2", 30)
    x.grid_set_layout(1, 2)
    x.grid_fill("types")
    rows = [x.grid_row(i) for i in range(2)]
    check("... the same box: 'merge the axes' is offered", x.grid_merge.isEnabled())
    x.grid_toggle.setChecked(True)
    ok = wait_for(lambda: x._grid_shown and x._grid_index is not None and not x._grid_busy, 30)
    tile_labels = list(w.explore_view.canvas.labels or [])
    check("the tiles are labelled by estimator, not by the raw prefix ('DTFE' and 'PS-DTFE · ps_q2')",
          ok and any(t[2].endswith("· DTFE") for t in tile_labels) and any(t[2].endswith("· PS-DTFE · ps_q2") for t in tile_labels),
          str(tile_labels))
    sent = []
    x._to_render_grid.connect(lambda req: sent.append(req))
    x.grid_set_row(1, x._by_root[str(s50 / "ps_alt")])   # row 1 = ps_alt (the preferred prefix is ps_q2): one recompose
    ok1 = wait_for(lambda: not x._grid_busy and len(sent) >= 1, 30)
    snap1 = x.grid_table.cellWidget(1, E._C_SNAP)
    snap1.setCurrentIndex(snap1.findData(99))             # row 1 (ps_alt) to z = 0: its type sticks
    ok2 = wait_for(lambda: not x._grid_busy and len(sent) >= 2, 30)
    r1 = x.grid_row(1)
    check("a row's redshift changed: its output type sticks (ps_alt, although snapshot 99 has ps_output and the preferred "
          "ps_q2 is absent there), the field too, and each change recomposes once", ok and ok1 and ok2 and r1 is not None
          and r1[0].snap == 99 and r1[0].prefix == "ps_alt" and r1[1] == rows[1][1] and len(sent) == 2,
          f"{r1 and (r1[0].snap, r1[0].prefix, r1[1])} {len(sent)}")
    snap0 = x.grid_table.cellWidget(0, E._C_SNAP)
    snap0.setCurrentIndex(snap0.findData(99))             # row 0 (DTFE) to z = 0: stays DTFE
    wait_for(lambda: not x._grid_busy and len(sent) >= 3, 30)
    r0_ = x.grid_row(0)
    check("a DTFE row stepped to a snapshot that has a DTFE run stays DTFE (the preferred prefix would say ps_q2)",
          r0_ is not None and r0_[0].snap == 99 and r0_[0].prefix == "output", f"{r0_ and (r0_[0].snap, r0_[0].prefix)}")
    Path(str(s99 / "output") + ".a_den").unlink()         # snapshot 99 loses its DTFE run
    x.refresh_outputs()                                   # (leaves the grid; row 0 at 099 falls back to a phase-space run)
    wait_for(lambda: not x._busy, 30)
    fell = x.grid_row(0)
    snap0 = x.grid_table.cellWidget(0, E._C_SNAP)
    snap0.setCurrentIndex(snap0.findData(50))             # back to z = 1: the row's own choice, DTFE, comes back
    back = x.grid_row(0)
    check("a row's chosen type survives a snapshot without it: at 099 it falls back to a phase-space run, stepped back "
          "to 050 it is DTFE again", fell is not None and fell[0].snap == 99 and G.estimator_label(fell[0]) == "PS-DTFE"
          and back is not None and back[0].snap == 50 and back[0].prefix == "output",
          f"{fell and (fell[0].snap, fell[0].prefix)} {back and (back[0].snap, back[0].prefix)}")
    x.grid_toggle.setChecked(True)
    wait_for(lambda: x._grid_shown and not x._grid_busy, 30)
    sim1 = x.grid_table.cellWidget(1, E._C_SIM)
    sim1.setCurrentIndex(sim1.findData(E.GRID_NONE))
    wait_for(lambda: not x._grid_busy, 30)
    check("a row's simulation set to '—' leaves that panel blank (its redshift and type menus empty)",
          x.grid_row(1) is None and x.grid_table.cellWidget(1, E._C_SNAP).count() == 0
          and x.grid_table.cellWidget(1, E._C_TYPE).count() == 0 and x._grid_shown)
    sim0 = x.grid_table.cellWidget(0, E._C_SIM)
    sim0.setCurrentIndex(sim0.findData(E.GRID_NONE))      # the last panel blanked while the grid is shown
    wait_for(lambda: not x._grid_busy and not x._busy, 30)
    check("the last panel blanked while the grid is shown: the map comes back with the warning, the rows stay blank "
          "(no silent refill from the map)", not x._grid_shown and x.grid_state.text().startswith("⚠ no panel")
          and x.grid_row(0) is None and x.grid_row(1) is None, f"{x._grid_shown} {x.grid_state.text()!r}")
    # a file opened by hand goes straight into a panel; two opened folders of one name are told apart
    opened = []
    for sub_ in ("opened_a", "opened_b"):
        d_ = tmp / sub_ / "elsewhere"
        d_.mkdir(parents=True, exist_ok=True)
        np.full(32 ** 3, 1.5, dtype=np.float32).tofile(str(d_ / "foo.a_den"))
        opened.append(d_ / "foo")
    ok = x.open_file(str(opened[0]) + ".a_den")
    wait_for(lambda: not x._busy and x.current is not None and str(x.current.root) == str(opened[0]), 30)
    x.grid_set_row(0, x._by_root[str(opened[0])])
    got = x.grid_row(0)
    check("a file opened by hand can be placed in a panel at once (its 'Opened files' group is in the row's menu)",
          ok and got is not None and str(got[0].root) == str(opened[0])
          and x.grid_table.cellWidget(0, E._C_SIM).currentData() == E.GROUP_OPENED, f"{ok} {got}")
    x.open_file(str(opened[1]) + ".a_den")
    wait_for(lambda: not x._busy and x.current is not None and str(x.current.root) == str(opened[1]), 30)
    menu = x.grid_table.cellWidget(0, E._C_SNAP)
    texts = [menu.itemText(k) for k in range(menu.count())]
    check("two opened folders named 'elsewhere' read 'opened_a/elsewhere' and 'opened_b/elsewhere' in the Redshift menu, "
          "and the row kept its output", texts == ["opened_a/elsewhere", "opened_b/elsewhere"]
          and x.grid_row(0) is not None and str(x.grid_row(0)[0].root) == str(opened[0]), str(texts))
    # review 2026-10-05 [38] fix 3: a refresh re-lists row 0's opened file under the custom snapshots (its folder joined
    # them): the row follows its OUTPUT into that group -- kept on 'Opened files', which no longer holds it, the row
    # swapped to opened_b
    w.custom_dirs.append(str(opened[0].parent))
    try:
        x.refresh_outputs()
        wait_for(lambda: not x._busy, 30)
        moved, grp0 = x.grid_row(0), x.grid_table.cellWidget(0, E._C_SIM).currentData()
    finally:
        w.custom_dirs.remove(str(opened[0].parent))
    x.refresh_outputs()                                   # opened_a left every list: opened again, opened_b shown as before
    x.open_file(str(opened[0]) + ".a_den")
    x.select_root(str(opened[1]))
    wait_for(lambda: not x._busy and x.current is not None and str(x.current.root) == str(opened[1]), 30)
    check("a row's opened file whose folder then joins the custom snapshots: after the refresh the row keeps that output, "
          "its Simulation now 'Custom snapshots'", moved is not None and str(moved[0].root) == str(opened[0])
          and grp0 == E.GROUP_CUSTOM, f"{moved and moved[0].root} {grp0}")
    # the snapshot key: a simulation output with no number -> its folder; a custom run -> the snapshot FILE it read
    kd = tmp / "keys"; kd.mkdir(exist_ok=True)
    for nm, side_ in (("a", {"input_file": "~/snaps/one.hdf5"}), ("b", {"input_file": "~/snaps/one.hdf5"}),
                      ("c", {"input_file": "~/snaps/two.hdf5"}), ("d", None)):
        np.full(8, 1.0, dtype=np.float32).tofile(str(kd / f"{nm}.a_den"))
        if side_:
            (kd / f"{nm}.gui.json").write_text(json.dumps(side_))
    ko = {nm: G.OutputSet(kd / nm, box=None) for nm in "abcd"}
    weird = G.OutputSet(kd / "d", box=None, sim="X", snap=None)
    keys = {nm: E.ExploreControls._grid_snap_key(o) for nm, o in ko.items()}
    check("the composer's snapshot key: custom runs of ONE snapshot file share it whatever their names, another file "
          "differs, no sidecar -> the folder, an unnumbered snapdir -> its folder (not '?')",
          keys["a"] == keys["b"] == str(Path("~/snaps/one.hdf5").expanduser().resolve()) and keys["c"] != keys["a"]
          and keys["d"] == str(kd) and E.ExploreControls._grid_snap_key(weird) == str(kd), str(keys))
    # the sidecar at a job's start never overwrites one that describes grids already on disk; the finish does
    sspec = rs.CustomSpec(input_file=str(kd / "one.hdf5"), estimator="dtfe", output_dir=str(kd), output_name="side")
    (kd / "side.gui.json").write_text(json.dumps({"estimator": "ps", "note": "the old PS run"}))
    w._register_custom_output(sspec)
    kept_old = json.loads((kd / "side.gui.json").read_text()).get("note") == "the old PS run"
    w._register_custom_output(sspec, final=True)
    final_ = json.loads((kd / "side.gui.json").read_text())
    (kd / "side.gui.json").unlink()
    w._register_custom_output(sspec)
    fresh = json.loads((kd / "side.gui.json").read_text())
    check("a re-run under an old name keeps the old sidecar at its start (a stopped run leaves the old grids), rewrites it "
          "when it finishes, and a fresh name gets one at the start", kept_old and final_.get("estimator") == "dtfe"
          and "note" not in final_ and fresh.get("estimator") == "dtfe", f"{kept_old} {final_} {fresh}")
    cli_spec = rs.CustomSpec(input_file=str(kd / "one.hdf5"), estimator="ps", output_dir=str(kd), output_name="cli")
    np.ones(8, np.float32).tofile(str(kd / "cli.a_den"))                 # grids a CLI run left, no sidecar
    w._register_custom_output(cli_spec)
    no_side = not (kd / "cli.gui.json").exists()
    w._register_custom_output(cli_spec, final=True)
    check("grids without a sidecar are not described at a job's START (a CLI run's); the finish writes one, with no .tmp left",
          no_side and (kd / "cli.gui.json").is_file() and not (kd / "cli.gui.json.tmp").exists())
    bad_spec = rs.CustomSpec(input_file=str(kd / "one.hdf5"), estimator="ps", output_dir=str(kd), output_name="bad")
    (kd / "bad.gui.json.tmp").mkdir()                                     # the temp file's name taken by a folder: the write fails
    w.log.clear()
    w._register_custom_output(bad_spec)
    check("a sidecar that cannot be written is logged ('!! cannot write the settings sidecar'), not swallowed",
          "!! cannot write the settings sidecar bad.gui.json" in w.log.toPlainText(), w.log.toPlainText()[-200:])
    # ... and through a Run-button job's real start (review 2026-10-05 [32]): _begin clears the log FIRST, then writes
    # the sidecar, so the line survives. _next_step held: the finish would write (and log) the sidecar again
    idle = w.proc is None
    w.log.appendPlainText("an earlier job's log")
    w._next_step = lambda: None
    try:
        w._begin([], A.OWN, bad_spec, None)
    finally:
        del w._next_step
    log_ = w.log.toPlainText()
    check("... a job's start (_begin from the Run button) clears the old log, THEN logs the '!! cannot write' line: it survives",
          idle and "an earlier job's log" not in log_ and log_.count("!! cannot write the settings sidecar bad.gui.json") == 1,
          log_[-300:])
    if str(kd) in w.custom_dirs:
        w.custom_dirs.remove(str(kd))
    shutil.rmtree(kd)
    # REVIEW 2026-10-05: custom runs of one NAME in two folders (the Custom tab's default name is the input file's
    # stem) -- the row's choice sticks by its root and estimator, duplicate labels get their folder, deep paths tell
    in_dir = tmp / "cust_in"; in_dir.mkdir(exist_ok=True)
    cds = {k: tmp / f"cust_{k}" for k in ("dtfe", "ps", "ps2", "z")}
    for k, d_ in cds.items():
        d_.mkdir(exist_ok=True)
    for k, est in (("dtfe", "dtfe"), ("ps", "ps"), ("ps2", "ps")):
        for nm in ("X", "Y"):
            if k == "ps2" and nm == "Y":
                continue
            np.full(32 ** 3, 1.0 + 0.1 * (nm == "Y"), dtype=np.float32).tofile(str(cds[k] / f"{nm}.a_den"))
            (cds[k] / f"{nm}.gui.json").write_text(json.dumps({"estimator": est, "input_file": str(in_dir / f"{nm}.hdf5"), "box_mpc": [0, 100] * 3}))
    for nm, fam in (("A", "TNG50-3"), ("B", "TNG100-3")):
        np.full(32 ** 3, 1.0, dtype=np.float32).tofile(str(cds["z"] / f"{nm}.a_den"))
        (cds["z"] / f"{nm}.gui.json").write_text(json.dumps({"estimator": "ps", "input_file": str(in_dir / fam / "snapdir_099" / "combined_099.hdf5"), "box_mpc": [0, 100] * 3}))
    for k in ("dupA", "dupB"):                            # two more PS runs X of X.hdf5, in two folders both named 'out':
        cds[k] = tmp / k / "out"                          # the Type label's third stage (the full folder)
        cds[k].mkdir(parents=True, exist_ok=True)
        np.full(32 ** 3, 1.0, dtype=np.float32).tofile(str(cds[k] / "X.a_den"))
        (cds[k] / "X.gui.json").write_text(json.dumps({"estimator": "ps", "input_file": str(in_dir / "X.hdf5"), "box_mpc": [0, 100] * 3}))
    w.custom_dirs += [str(d_) for d_ in cds.values()]
    x.refresh_outputs()
    x.select_root(str(cds["ps"] / "X"))
    wait_for(lambda: not x._busy and x.current is not None and str(x.current.root) == str(cds["ps"] / "X"), 30)
    x.grid_set_layout(1, 2)
    x.grid_fill("types")
    rows = [x.grid_row(i) for i in range(2)]
    typ1 = x.grid_table.cellWidget(1, E._C_TYPE)
    labels = [typ1.itemText(k) for k in range(typ1.count())]
    check("'across outputs' on a custom PS run named X: the DTFE run X of the same snapshot file (another folder) beside it; "
          "two PS runs of that name are told apart by their folder in the Type menu",
          [str(r[0].root) for r in rows if r] == [str(cds["dtfe"] / "X"), str(cds["ps"] / "X")]
          and "PS-DTFE · X (cust_ps)" in labels and "PS-DTFE · X (cust_ps2)" in labels and "DTFE · X" in labels, f"{rows} {labels}")
    by_label = {typ1.itemText(k): typ1.itemData(k) for k in range(typ1.count())}
    want_dup = {f"PS-DTFE · X ({cds[k]})": str(cds[k] / "X") for k in ("dupA", "dupB")}
    check("... two more of them in two folders both named 'out' read with their FULL folder, each entry its own run "
          "(review 2026-10-05 [32]); the told-apart ones keep the short folder",
          all(by_label.get(lab) == root_ for lab, root_ in want_dup.items()) and len(labels) == len(set(labels))
          and by_label.get("PS-DTFE · X (cust_ps)") == str(cds["ps"] / "X"), str(by_label))
    snap1 = x.grid_table.cellWidget(1, E._C_SNAP)
    key_y = E.ExploreControls._grid_snap_key(x._by_root[str(cds["ps"] / "Y")])
    key_x = E.ExploreControls._grid_snap_key(x._by_root[str(cds["ps"] / "X")])
    snap1.setCurrentIndex(snap1.findData(key_y))
    away = x.grid_row(1)
    snap1.setCurrentIndex(snap1.findData(key_x))
    back = x.grid_row(1)
    check("a PS row of name X stepped to Y (PS there) and back comes back as PS X, not the DTFE run of the same name",
          away is not None and str(away[0].root) == str(cds["ps"] / "Y") and back is not None and str(back[0].root) == str(cds["ps"] / "X"),
          f"{away and away[0].root} {back and back[0].root}")
    typ1.setCurrentIndex(typ1.findData(str(cds["ps2"] / "X")))           # the user's own pick: the second PS run
    snap1.setCurrentIndex(snap1.findData(key_y))
    snap1.setCurrentIndex(snap1.findData(key_x))
    check("the user's pick of the second PS run X survives a step away and back (its root is remembered)",
          x.grid_row(1) is not None and str(x.grid_row(1)[0].root) == str(cds["ps2"] / "X"), str(x.grid_row(1) and x.grid_row(1)[0].root))
    texts = [snap1.itemText(k) for k in range(snap1.count())]
    check("two custom runs of 'snapdir_099/combined_099.hdf5' from two simulations read with their simulation folder; X.hdf5 "
          "and Y.hdf5 stay short", "TNG50-3/snapdir_099/combined_099.hdf5" in texts and "TNG100-3/snapdir_099/combined_099.hdf5" in texts
          and "X.hdf5" in texts and "Y.hdf5" in texts, str(texts))
    x.grid_fill("redshifts")
    check("'across redshifts' from a custom PS run keeps to ITS estimator (X and Y of the PS folder, not the DTFE run of X)",
          sorted(str(r[0].root) for r in [x.grid_row(i) for i in range(2)] if r) == sorted([str(cds["ps"] / "X"), str(cds["ps"] / "Y")]),
          str([r and str(r[0].root) for r in [x.grid_row(i) for i in range(2)]]))
    # the newest NUMBERED snapshot is a simulation row's default, not an unnumbered folder; numbers come first
    backup = sim / "snapdir_backup"; backup.mkdir(exist_ok=True)
    shutil.copy(s50 / "ps_gui.a_den", backup / "ps_gui.a_den")
    x.refresh_outputs()
    sim0 = x.grid_table.cellWidget(0, E._C_SIM)
    sim0.setCurrentIndex(sim0.findData(E.GRID_NONE))
    sim0.setCurrentIndex(sim0.findData(SIM))
    snap0 = x.grid_table.cellWidget(0, E._C_SNAP)
    texts = [snap0.itemText(k) for k in range(snap0.count())]
    check("a blank row set to the simulation: the Redshift menu lists the numbers first, then '(snapdir_backup)', and "
          "defaults to the newest number", texts[-1].endswith("(snapdir_backup)") and snap0.currentData() == 99
          and [snap0.itemData(k) for k in range(snap0.count())][:2] == [50, 99], f"{texts} {snap0.currentData()}")
    shutil.rmtree(backup)
    x.refresh_outputs()
    # two estimators at two redshifts: the tiles and the saved figure name the estimator; every cell menu has a tooltip
    x.grid_set_row(0, x._by_root[str(std)])                                   # DTFE at 050
    x.grid_set_row(1, x._by_root[str(s99 / "ps_gui")])                        # PS-DTFE at 099
    x.grid_merge.setChecked(False)                                            # titles per panel (merged: inside labels)
    x.grid_toggle.setChecked(True)
    ok = wait_for(lambda: x._grid_shown and x._grid_index is not None and not x._grid_busy, 30)
    tl = [t[2] for t in (w.explore_view.canvas.labels or [])]
    titles = [a.get_title() for a in x.grid_figure()[0].axes if a.get_title()]
    check("a DTFE panel at z = 1 beside a PS-DTFE one at z = 0: both the tiles and the saved figure's titles name the estimator",
          ok and any("· DTFE" in t for t in tl) and any("PS-DTFE" in t for t in tl)
          and any("DTFE" in t and "PS-DTFE" not in t for t in titles) and any("PS-DTFE" in t for t in titles), f"{tl} {titles}")
    fld0 = x.grid_table.cellWidget(0, E._C_FLD)
    check("every cell menu carries its text as a tooltip (the narrow columns clip it)",
          fld0.toolTip() == fld0.currentText() and x.grid_table.cellWidget(0, E._C_SIM).toolTip() == SIM, f"{fld0.toolTip()!r}")
    sent.clear()                                                              # (the signal is connected above: one slot)
    x.grid_set_row(1, x._by_root[str(s50 / "ps_gui")], field="density", component="value")
    wait_for(lambda: not x._grid_busy, 30)
    spin(200)
    check("grid_set_row with a field and a component while the grid is shown recomposes exactly once",
          len(sent) == 1 and x.grid_row(1) is not None and x.grid_row(1)[1] == "density", str(len(sent)))
    x.grid_toggle.setChecked(False)
    wait_for(lambda: not x._grid_shown and not x._busy, 30)
    for i in range(2):
        simw = x.grid_table.cellWidget(i, E._C_SIM)
        simw.setCurrentIndex(simw.findData(E.GRID_NONE))
    x.grid_toggle.setChecked(True)
    ok = wait_for(lambda: x._grid_shown and not x._grid_busy, 30)
    check("the grid turned on with every panel blank is filled from the map", ok and x.grid_row(0) is not None
          and str(x.grid_row(0)[0].root) == str(x.current.root) and x.grid_row(1) is not None, str(x.grid_row(0)))
    x.grid_toggle.setChecked(False)
    wait_for(lambda: not x._grid_shown and not x._busy, 30)
    # a job's finish: the sidecar is written BEFORE Explore is refreshed, and from a COPY of the spec taken at the start
    old_custom, old_tab = w.custom, w.tabs.currentIndex()  # restored below: the later sections run THE demo again
    demo3 = rs.demo_spec(tmp / "demo3", gpu=False)
    demo3.estimator = "ps"
    root3 = demo3.output_root()
    root3.parent.mkdir(parents=True, exist_ok=True)
    Path(str(root3) + ".gui.json").write_text(json.dumps({"estimator": "dtfe", "note": "an old DTFE run"}))
    w.custom = demo3
    w._load_into_widgets()
    w.tabs.setCurrentIndex(A.OWN)
    qa.processEvents()
    w.run_btn.click()
    spin(300)
    w.o_dtfe.setChecked(True)                              # edited WHILE the job runs: must not reach this job's sidecar
    w.o_name.setText("demo3_edited")
    w.o_name.editingFinished.emit()
    dt = wait_idle(w, 300)
    side3 = json.loads(Path(str(root3) + ".gui.json").read_text())
    o3 = x._by_root.get(str(root3))
    check(f"a PS re-run under an old DTFE name ({dt:.0f} s): the finished sidecar says PS, the edits made during the run "
          "(estimator, name) reached neither it nor a stray file, and Explore's listing (refreshed after the write) calls it PS-DTFE",
          w.status.text() == "Finished" and side3.get("estimator") == "ps" and "note" not in side3
          and not Path(str(root3.parent / "demo3_edited") + ".gui.json").exists()
          and o3 is not None and G.estimator_label(o3) == "PS-DTFE", f"{w.status.text()} {side3} {o3 and G.estimator_label(o3)}")
    w.custom = old_custom
    w._load_into_widgets()
    w.tabs.setCurrentIndex(old_tab)                        # the zoom and server menu actions live on the Explore tab
    qa.processEvents()
    if str(root3.parent) in w.custom_dirs:
        w.custom_dirs.remove(str(root3.parent))
    shutil.rmtree(root3.parent, ignore_errors=True)
    for d_ in cds.values():
        if str(d_) in w.custom_dirs:
            w.custom_dirs.remove(str(d_))
        shutil.rmtree(d_)
    for k in ("dupA", "dupB"):
        shutil.rmtree(tmp / k)
    shutil.rmtree(in_dir)
    for o_ in opened:
        shutil.rmtree(o_.parent.parent)
    Path(str(std) + ".a_den").unlink()
    for f_ in alt_files:
        f_.unlink()
    x.refresh_outputs()
    x.select_root(prev_root)
    wait_for(lambda: x._plane is not None and not x._busy and x.current is not None and str(x.current.root) == prev_root, 30)
    x.grid_set_layout(2, 1)
    x.grid_fill("redshifts")
    x.grid_merge.setChecked(True)
    x.grid_toggle.setChecked(True)
    wait_for(lambda: x._grid_shown and not x._grid_busy, 30)
    x.grid_set_row(1, x._by_root[cur_root])          # the demo run: a box of another size
    ok = wait_for(lambda: not x._grid_busy, 30)
    n_axes = len(x.grid_figure()[0].axes)
    check("a panel from a box of another size greys 'merge the axes' out; the figure falls back to a bar per panel",
          ok and not x.grid_merge.isEnabled() and n_axes == 4 and x.grid_shared.isEnabled(),
          f"{x.grid_merge.isEnabled()} {n_axes}")
    x.grid_merge.setChecked(False)
    x.grid_toggle.setChecked(False)
    x.select_root(cur_root)
    ok = wait_for(lambda: x._plane is not None and not x._busy and x.current is not None
                  and str(x.current.root) == cur_root and not x._grid_shown, 30)
    check("back on the demo output with its map", ok and x.field.currentData() == cur_field, f"{x.current and x.current.root} {x.field.currentData()}")
    plane0 = x._plane
    x._pending = dict(x._req)
    x._render_failed("simulated")
    kept = x._busy and x._plane is plane0 and x._pending is None
    ok = wait_for(lambda: not x._busy and x._plane is not None, 30)
    check("a failed slice with a newer request pending sends that request and keeps the picture meanwhile",
          kept and ok and w.explore_view.canvas._img is not None)
    img = w.explore_view.canvas._img
    print("zoom:")
    x.zoom_res.setValue(96)
    check("no zoom yet: Zoom Out is off in the menu and on the tab", not w.zoom_out_action.isEnabled()
          and not x.zoom_out_btn.isEnabled() and not x.is_zoomed())
    x._dragged(img.height() // 4, img.width() // 4, img.height() // 2, img.width() // 2)
    ok = wait_for(lambda: x.is_zoomed() and not x._zoom_pending, 60)
    z = x._zoom
    check("dragging a rectangle evaluates the region through the server at the zoom resolution",
          ok and x._plane.shape == (96, 96) and w.explore_view.canvas._img.width() == 96
          and z["depth"] == 1 and z["a0"][0] > 0 and z["a0"][1] < 100 and z["a1"][0] > 0 and z["a1"][1] < 100
          and "zoom 96²" in w.explore_view.title.text() and w.zoom_out_action.isEnabled() and x.zoom_out_btn.isEnabled(),
          f"{ok} {getattr(x._plane, 'shape', None)} {z}")
    ax_ = w.explore_view.canvas.axes
    check("... and the axes follow the zoomed region", ax_ is not None and abs(ax_["x"][0] - z["a0"][0]) < 1e-9
          and abs(ax_["x"][1] - z["a0"][1]) < 1e-9 and abs(ax_["y"][0] - z["a1"][0]) < 1e-9, str(ax_))
    f0 = x._zoom_fields
    x.field.setCurrentIndex(x.field.findData("velocity"))
    qa.processEvents()
    check("another field inside the zoom comes from the same evaluation (no new query)",
          x._zoom_fields is f0 and x.is_zoomed() and x._plane.shape == (96, 96) and "velocity" in w.explore_view.title.text())
    x.log.setChecked(not x.log.isChecked())
    qa.processEvents()
    check("... and so does a colour change", x._zoom_fields is f0 and x.is_zoomed())
    x.field.setCurrentIndex(x.field.findData("density"))
    qa.processEvents()
    pt = x._point(10, 20)
    check("hovering inside the zoom reports a position inside the zoomed region",
          pt is not None and z["a0"][0] <= pt[0][0] <= z["a0"][1] and z["a1"][0] <= pt[0][1] <= z["a1"][1], str(pt))
    got = []
    w.explore_view.canvas.dragged.connect(lambda *a: got.append(a))
    canvas = w.explore_view.canvas
    t = canvas._target()
    p0 = QPointF(t.left() + t.width() * 0.3, t.top() + t.height() * 0.3).toPoint()
    p1 = QPointF(t.left() + t.width() * 0.6, t.top() + t.height() * 0.6).toPoint()
    QTest.mousePress(canvas, A.Qt.LeftButton, pos=p0)
    QTest.mouseMove(canvas, QPointF(t.left() + t.width() * 0.45, t.top() + t.height() * 0.45).toPoint())
    QTest.mouseMove(canvas, p1)
    QTest.mouseRelease(canvas, A.Qt.LeftButton, pos=p1)
    qa.processEvents()
    ok = wait_for(lambda: x.is_zoomed() and x._zoom["depth"] == 2 and not x._zoom_pending, 60)
    check("a real mouse drag on the map zooms again, one level deeper, inside the first zoom",
          got and ok and x._zoom["a0"][0] >= z["a0"][0] and x._zoom["a0"][1] <= z["a0"][1], f"{got} {x._zoom and x._zoom['depth']}")
    QTest.mousePress(canvas, A.Qt.LeftButton, pos=p0)
    QTest.mouseRelease(canvas, A.Qt.LeftButton, pos=p0)
    qa.processEvents()
    check("a press and release without a drag is still a click (a query, not a zoom)",
          x._zoom["depth"] == 2 and w.explore_view.canvas.marker is not None)
    check("each zoom reports how long it took: the timer beside the progress bar and the state line, the bar gone",
          x.zoom_time.text().endswith(" s") and " s; drag again" in x.zoom_state.text() and not x.zoom_progress.isVisible(),
          f"{x.zoom_time.text()} | {x.zoom_state.text()}")
    f2 = x._zoom_fields
    x.zoom_back_btn.click()
    qa.processEvents()
    check("Back returns to the previous zoom level from memory, without a new evaluation",
          x.is_zoomed() and x._zoom["depth"] == 1 and x._zoom_fields is f0 and x._plane.shape == (96, 96)
          and not x._zoom_pending, str(x._zoom and x._zoom["depth"]))
    x._dragged(*got[0])                               # the same rectangle as the real drag above
    qa.processEvents()
    check("dragging the same rectangle again brings that zoom back from memory at once",
          x.is_zoomed() and x._zoom["depth"] == 2 and x._zoom_fields is f2 and not x._zoom_pending
          and "from memory" in x.zoom_state.text() and x.zoom_time.text() == "from memory", x.zoom_state.text())
    x.zoom_back_btn.click()
    qa.processEvents()
    w.zoom_out_action.trigger()                       # Explore > Zoom Out
    ok = wait_for(lambda: not x.is_zoomed() and not x._busy and x._plane is not None and x._plane.shape == (64, 64), 30)
    check("Zoom Out returns to the slice from disk", ok and not w.zoom_out_action.isEnabled() and x.zoom_state.text() == "")
    check("... and Back is off again (no zoom history on the slice)", not x.zoom_back_btn.isEnabled())

    # a COMPOSITE query server (a large snapshot's): its zoom requests report their progress over the
    # partitions they visit (--serve-progress), which the bar follows and the state line sums up
    print("zoom progress on a composite query server:")
    w.server_action.trigger()                         # stop the single-tessellation server
    ok = wait_for(lambda: x.server_phase() == "stopped", 30)
    # with the server stopped and no partition set: the residency and the button after a restore
    x.resident.setValue(3)
    kw = x.server_kwargs()
    check("the 'Partitions in memory' value reaches the server with no partition set (the binary splits a big "
          "snapshot itself)", ok and kw is not None and kw.get("resident") == 3 and "partition" not in kw, str(kw))
    x.resident.setValue(0)
    x.server_state.setText(E.NO_SNAPSHOT_TEXT)
    x.restore_state({"snapshots": {}, "resident": 0})
    check("a restore re-judges the server button (its stale 'no snapshot known' goes once the snapshot is known)",
          x.server_btn.isEnabled() and x.server_state.text() != E.NO_SNAPSHOT_TEXT, x.server_state.text())
    x.server_partition = 2
    x.cache_edit.setText(str(tmp / "tess_comp"))
    w.server_action.trigger()
    wait_for(lambda: not x._server_starting, 300)
    check("the demo snapshot's server starts as a 2x2x2 composite", x._server_running and x._server.n_partitions == 8,
          x.server_state.text())
    maxima = []
    x.zoom_progress.valueChanged.connect(lambda _v: maxima.append(x.zoom_progress.maximum()))
    x._dragged(img.height() // 4, img.width() // 4, img.height() // 2, img.width() // 2)
    ok = wait_for(lambda: x.is_zoomed() and not x._zoom_pending, 120)
    sm = x._zoom.get("summary") if ok else None
    check("the bar runs over the partitions the request visits, and the state line says how many and how many "
          "were loaded from disk",
          ok and sm is not None and sm["total"] == 8 and 1 <= sm["partitions"] <= 8
          and max(maxima or [0]) == sm["partitions"] and f"{sm['partitions']} of 8 partitions" in x.zoom_state.text(),
          f"{x.zoom_state.text()} | {maxima[-3:]} | {sm}")
    # the cache keys hold canonical paths (canonical_path.h): the launcher's lookup must match a symlinked
    # name of the snapshot as well, or its estimate would call cached partitions missing
    import grids as Gm
    kwc = x.server_kwargs() or {}
    snapp = Path(str(kwc.get("snapshot", "")))
    link = tmp / "snaplink"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(snapp.parent)
    sets_abs = Gm.cached_partition_sets(tmp / "tess_comp", snapp, kwc.get("lagrangian_input"), True)
    sets_sym = Gm.cached_partition_sets(tmp / "tess_comp", link / snapp.name, kwc.get("lagrangian_input"), True)
    check("the launcher finds a snapshot's cached partitions through a symlinked path too",
          len(sets_abs.get((2, 2, 2), [])) == 8 and len(sets_sym.get((2, 2, 2), [])) == 8,
          f"{ {k: len(v) for k, v in sets_abs.items()} } vs { {k: len(v) for k, v in sets_sym.items()} }")
    w.zoom_out_action.trigger()
    wait_for(lambda: not x.is_zoomed() and not x._busy, 30)
    w.server_action.trigger()                         # back to the single-tessellation server for what follows
    wait_for(lambda: x.server_phase() == "stopped", 30)
    x.server_partition = None
    x.cache_edit.setText(str(tmp / "tess"))
    w.server_action.trigger()
    wait_for(lambda: not x._server_starting, 120)
    check("... and the single-tessellation server is back", x._server_running and x._server.n_partitions == 1,
          x.server_state.text())
    # a partition building its cell index (once, ~20 s on a large snapshot) says so on the state line
    saved = (x._server_log, x._zoom_log_pos, x._zoom_pending, x._exact_running)
    fake = tmp / "fake_server.log"
    fake.write_text("[request 1/3] partition 4: in memory\n[request 1/3] partition 4 done in 0.50 s (load 0.00 s, index 0.00 s)\n"
                    "[request 2/3] partition 6: building its cell index\n")
    x._server_log, x._zoom_log_pos, x._zoom_pending, x._exact_running = fake, 0, True, False
    x._read_request_progress()
    shown = x.zoom_state.text()
    x._server_log, x._zoom_log_pos, x._zoom_pending, x._exact_running = saved
    x.zoom_state.setText("")
    check("a server line about building a partition's cell index shows as 'indexing partition k of n'",
          "indexing partition 2 of 3" in shown, shown)

    print("exact zoom and the standard-DTFE server:")
    # B2 (2026-10-05): the planes an exact zoom wrote are in u-units; with the output's redshift the loader
    # applies the slice map's (1/(1+z))^exp (velocity 0.5, dispersion 1.0), the density stays
    ez = Path(tempfile.mkdtemp(prefix="gui_exact_")) / "zoom"
    try:
        np.full(4 * 4 * 1, 2.0, dtype=np.float32).tofile(str(ez) + ".a_den")
        np.full(4 * 4 * 1 * 3, 3.0, dtype=np.float32).tofile(str(ez) + ".a_vel")
        np.full(4 * 4 * 1, 5.0, dtype=np.float32).tofile(str(ez) + ".a_velDisp")
        raw = E.load_exact_planes(ez, (4, 4, 1))
        at_z1 = E.load_exact_planes(ez, (4, 4, 1), redshift=1.0)
        # the planes stay in the files' float32 (2026-10-06: no float64 copy of every component), so the factor
        # is applied at float32 rounding
        check("an exact zoom's velocity and dispersion planes get the slice map's redshift factor, the density not",
              float(raw.velocity.max()) == 3.0 and abs(float(at_z1.velocity.max()) - 3.0 * 0.5 ** 0.5) < 3e-7
              and raw.sigma is not None and abs(float(at_z1.sigma.max()) - 5.0 * 0.5) < 3e-7
              and float(at_z1.density.max()) == 2.0, f"{raw.velocity.max()} {at_z1.velocity.max()} {at_z1.sigma.max() if at_z1.sigma is not None else None}")
    finally:
        shutil.rmtree(ez.parent)
    x.zoom_exact.setChecked(True)
    geo = x._geometry()
    cmd, _r, grid_, edges = x.exact_zoom_command(x.current, geo[0], geo[1], geo[2], (geo[3][0], geo[4][0]), (geo[3][1], geo[4][1]), 64)
    gi = cmd.index("--grid") if cmd and "--grid" in cmd else -1
    check("the exact zoom's command for a PERIODIC phase-space run: PS-DTFE on the whole periodic box, deposited "
          "into a window of a virtual grid (the run's cells along the slice axis), the exact deposit, no region cut",
          cmd is not None and cmd[0].endswith("PS-DTFE") and "--periodic" in cmd and "--ps-window" in cmd
          and "--ps-exact-deposit" in cmd and "--box" not in cmd and "--regionMpc" not in cmd and gi >= 0
          and cmd[gi + 1 + geo[0]] == str(x.current.n) and all(int(cmd[gi + 1 + a]) >= 64 for a in geo[2])
          and grid_[geo[0]] == 1 and edges is not None, str(cmd))
    check("... which reuses the query server's tessellation cache folder and asks for the progress lines",
          "--tessellation-cache" in cmd and cmd[cmd.index("--tessellation-cache") + 1] == str(tmp / "tess")
          and "--verbose" in cmd and cmd[cmd.index("--verbose") + 1] == "2", str(cmd))
    x.zoom_res.setValue(64)
    # an 8-of-32-cell square: the zoom's virtual grid (64 cells across it) then tiles the box exactly, so the
    # window is the same region as the slice's cells; the on-disk run is the SAMPLED deposit, whose region
    # mass differs from the exact one by what the straddling streams put across the edge (a few percent here)
    sq = (img.height() // 4, img.width() // 4, img.height() // 2 - 1, img.width() // 2 - 1)
    ref = region_mean(x, *sq)
    x._dragged(*sq)
    ok = wait_for(lambda: x.is_zoomed() and not x._zoom_pending and bool(x._zoom.get("exact")), 300)
    plane = x._plane
    m64 = None if plane is None else float(np.nanmean(plane))
    check("an exact zoom of the PERIODIC phase-space demo runs and shows the region's cells: their mean density "
          "within 5% of the slice's mean over the same region on disk",
          ok and plane is not None and plane.shape == (64, 64) and float(np.nanmin(plane)) >= 0.0
          and m64 is not None and abs(m64 - ref) <= 0.05 * ref and "exact zoom 64²" in w.explore_view.title.text(),
          f"{x.zoom_state.text()} | zoom mean {m64} vs slice region mean {ref}")
    x.field.setCurrentIndex(x.field.findData("streams"))
    wait_for(lambda: not x._busy and x._req["field"] == "streams", 30)
    check("... and its stream count (the demo's folds reach 3 streams)",
          x.is_zoomed() and x._plane is not None and float(np.nanmax(x._plane)) >= 3.0, str(None if x._plane is None else float(np.nanmax(x._plane))))
    x.field.setCurrentIndex(x.field.findData("density"))
    wait_for(lambda: not x._busy and x._req["field"] == "density", 30)
    w.zoom_out_action.trigger()
    wait_for(lambda: not x.is_zoomed() and not x._busy, 30)
    # the contract: the exact deposit's mass inside a cell-aligned region does not depend on the grid, so the
    # same square at twice the resolution (a virtual grid of 512) has the same mean to float rounding
    x.zoom_res.setValue(128)
    x._dragged(*sq)
    ok = wait_for(lambda: x.is_zoomed() and not x._zoom_pending and bool(x._zoom.get("exact")), 300)
    m128 = None if x._plane is None else float(np.nanmean(x._plane))
    check("the same square zoomed at 128² (a finer virtual grid) has the same mean density to 1e-4: the exact "
          "deposit's mass inside a cell-aligned region is grid-independent",
          ok and x._plane is not None and x._plane.shape == (128, 128) and m64 is not None and m128 is not None
          and abs(m128 - m64) <= 1e-4 * m64, f"{x.zoom_state.text()} | {m64} vs {m128}")
    x.zoom_res.setValue(64)
    w.zoom_out_action.trigger()
    wait_for(lambda: not x.is_zoomed() and not x._busy, 30)

    # a PARTITIONED tessellation cache, as the query server of a large snapshot leaves it (built here
    # by the binary's server, which exits when its stdin closes): an exact zoom then reuses that split,
    # loads the partitions that reach the region and skips the others
    print("exact zoom over a partitioned tessellation cache:")
    pcache = tmp / "tess_parts"
    pcache.mkdir()
    scmd = [cmd[0], cmd[1], str(tmp / "pserve"), "--serve", "--input", cmd[cmd.index("--input") + 1]]
    for flag in ("--MpcUnit", "--lagrangianInput"):
        if flag in cmd:
            scmd += [flag, cmd[cmd.index(flag) + 1]]
    scmd += ["--periodic", "--partition", "2", "2", "2", "--tessellation-cache", str(pcache), "--per-stream"]
    r = subprocess.run(scmd, stdin=subprocess.DEVNULL, capture_output=True)   # stdout = the binary protocol
    check("a composite server with a 2x2x2 split caches its 8 partitions and their occupancy maps",
          r.returncode == 0 and len(list(pcache.glob("*.tess"))) == 8 and len(list(pcache.glob("*.tess.occ"))) == 8,
          r.stderr.decode(errors="replace")[-300:])
    # a slice a quarter of the way up the box: the demo's waves move nothing along z, so the 4 upper
    # partitions cannot reach it and must be skipped. First the zoom without the partitioned cache, as
    # the reference the cached one must reproduce
    x.index.setValue(x.current.n // 4)
    wait_for(lambda: not x._busy and x._req is not None and x._req["index"] == x.current.n // 4, 30)
    x._dragged(*sq)
    ok = wait_for(lambda: x.is_zoomed() and not x._zoom_pending and bool(x._zoom.get("exact")) and not x._exact_running, 300)
    m_ref = None if (not ok or x._plane is None) else float(np.nanmean(x._plane))
    w.zoom_out_action.trigger()
    wait_for(lambda: not x.is_zoomed() and not x._busy, 30)
    x.cache_edit.setText(str(pcache))
    geo = x._geometry()
    plan = x.exact_zoom_plan(x.current, geo[0], geo[1], geo[2], (geo[3][0], geo[4][0]), (geo[3][1], geo[4][1]), 64)
    shutil.rmtree(Path(plan["root"]).parent, ignore_errors=True)
    est = x.exact_zoom_estimate(plan)
    check("the zoom finds that split in the cache and passes it, with the cache folder",
          plan["partition"] == (2, 2, 2) and "--partition" in plan["cmd"]
          and plan["cmd"][plan["cmd"].index("--tessellation-cache") + 1] == str(pcache)
          and len(plan["sets"].get((2, 2, 2), [])) == 8, str(plan["cmd"]))
    check("its estimate reads the occupancy maps: the 4 upper partitions never reach this slice, the others "
          "are loaded, none is built",
          est is not None and est["n_total"] == 8 and est["n_build"] == 0 and est["n_skip"] == 4
          and est["n_load"] == 4, str(est))
    check("... and finds the snapshot's globals recorded beside them, so it counts no time for reading the snapshot",
          est is not None and est.get("read_skipped") is True, str(est))
    if est is not None:
        print(f"   info: the estimate loads {est['n_load']} and skips {est['n_skip']} of {est['n_total']} partitions, "
              f"about {est['seconds']:.1f} s")
    seen = []
    x._exact_worker.progress.connect(seen.append)
    x._zoom_memo.clear()                             # the reference zoom above is remembered: run it again
    x._dragged(*sq)
    ok = wait_for(lambda: x.is_zoomed() and not x._zoom_pending and bool(x._zoom.get("exact")) and not x._exact_running, 300)
    mc = None if x._plane is None else float(np.nanmean(x._plane))
    check("the zoom over the cache shows its progress, skips the 4 partitions the maps rule out (the binary says "
          "so), and gives the uncached zoom's density (to float rounding)",
          ok and mc is not None and m_ref is not None and abs(mc - m_ref) <= 1e-5 * m_ref
          and any("partition " in t for t in seen) and any("4 of 8 partitions skipped" in t for t in seen),
          f"{x.zoom_state.text()} | {mc} vs {m_ref} | {seen[-4:]}")
    check("... and the binary agrees: every partition it needs is cached, so it never read the snapshot",
          any("the snapshot is not read" in t for t in seen), str(seen))
    x._exact_worker.progress.disconnect(seen.append)
    w.zoom_out_action.trigger()
    wait_for(lambda: not x.is_zoomed() and not x._busy, 30)

    # a long exact zoom asks first, with the estimate; declining starts nothing
    asked = []
    old_limit = E.EXACT_CONFIRM_SECONDS
    E.EXACT_CONFIRM_SECONDS = -1.0                   # every estimate is "long" now
    x.confirm_exact = lambda text: (asked.append(text), False)[1]
    x._zoom_memo.clear()
    x._dragged(*sq)
    qa.processEvents()
    check("an exact zoom estimated to take longer asks first, saying what it will do, and declining starts nothing",
          len(asked) == 1 and "Estimated time" in asked[0] and "partitions reach the region" in asked[0]
          and "--" not in asked[0] and not x._exact_running and x.zoom_state.text() == "exact zoom not started",
          f"{asked[:1]} | {x.zoom_state.text()}")
    E.EXACT_CONFIRM_SECONDS = old_limit
    x.confirm_exact = None

    # progress and Cancel, on a stand-in run that only reads and then waits
    fake_root = tmp / "fake_zoom" / "zoom"
    fake_root.parent.mkdir()
    fake = dict(plan, cmd=[sys.executable, "-c", "import time; print('Reading the snapshot', flush=True); time.sleep(120)"],
                root=str(fake_root), ps=False, cache_dir=None)
    x.exact_zoom_plan = lambda *a: dict(fake)
    x._zoom_memo.clear()
    x._dragged(*sq)
    ok = wait_for(lambda: "reading the snapshot" in x.zoom_state.text(), 30)
    check("a running exact zoom reports what it is doing and offers Cancel", ok and x._exact_running
          and x.zoom_cancel_btn.isEnabled(), x.zoom_state.text())
    x._dragged(*sq)
    qa.processEvents()
    check("... and a second drag meanwhile is refused with the reason (nothing is queued)",
          "still running" in x.zoom_state.text() and x._exact_running, x.zoom_state.text())
    x.zoom_cancel_btn.click()
    ok = wait_for(lambda: not x._exact_running, 30)
    check("Cancel stops it: the state says so, the button is off, no zoom, its temporary folder is gone",
          ok and x.zoom_state.text() == "exact zoom cancelled" and not x.zoom_cancel_btn.isEnabled()
          and not x.is_zoomed() and not fake_root.parent.exists(), x.zoom_state.text())
    del x.exact_zoom_plan                            # the method again
    x.cache_edit.setText(str(tmp / "tess"))
    x.zoom_exact.setChecked(False)
    # a standard-DTFE run of the demo snapshot: its server (no streams) and its exact zoom (a region cut
    # from the periodic box, the exact volume average)
    dtfe_spec = rs.CustomSpec(input_file=str(demo.input_file), estimator="dtfe", grid=32,
                              output_dir=str(tmp / "demo_dtfe"), mpc_unit=demo.mpc_unit, periodic=True, gpu=False)
    w.custom = dtfe_spec
    w._load_into_widgets()
    w.tabs.setCurrentIndex(A.OWN)
    qa.processEvents()
    w.run_btn.click()
    wait_idle(w, 300)
    droot = dtfe_spec.output_root()
    check("a standard-DTFE run of the demo snapshot writes its grids", Path(str(droot) + ".a_den").is_file(), w.status.text())
    w._explore_output(str(droot))
    ok = wait_for(lambda: w.explore._plane is not None and not w.explore._busy and x.current is not None
                  and str(x.current.root) == str(droot), 30)
    check("Explore opens it", ok)
    x.cache_edit.setText(str(tmp / "tess_dtfe"))
    x._cache_user_set = True
    if x.server_phase() != "stopped":                  # the phase-space demo's server is still up: the menu item toggles
        w.server_action.trigger()
        wait_for(lambda: x.server_phase() == "stopped", 30)
    w.server_action.trigger()
    check("Explore > Start Query Server on a standard-DTFE output starts a build", x._server_starting, x.server_state.text())
    wait_for(lambda: not x._server_starting, 120)
    check("the query server starts on a standard-DTFE output (no per-stream request to the standard binary)",
          x._server_running, x.server_state.text())
    img = w.explore_view.canvas._img
    x.zoom_res.setValue(64)
    x._dragged(img.height() // 4, img.width() // 4, img.height() // 2, img.width() // 2)
    ok = wait_for(lambda: x.is_zoomed() and not x._zoom_pending, 60)
    check("a point-evaluated zoom on the standard-DTFE server", ok and x._plane is not None and x._plane.shape == (64, 64)
          and "zoom 64²" in w.explore_view.title.text(), x.zoom_state.text())
    x.field.setCurrentIndex(x.field.findData("velocity"))
    wait_for(lambda: not x._busy and x._req["field"] == "velocity", 30)
    check("... with the velocity from the same evaluation", x.is_zoomed() and x._plane.shape == (64, 64))
    x.field.setCurrentIndex(x.field.findData("density"))
    wait_for(lambda: not x._busy and x._req["field"] == "density", 30)
    x.zoom_exact.setChecked(True)
    geo = x._geometry()
    cmd, root_, grid_, _e = x.exact_zoom_command(x.current, geo[0], geo[1], geo[2], (geo[3][0], geo[4][0]), (geo[3][1], geo[4][1]), 48)
    check("the exact zoom's command: the standard binary on the region, one slice cell thick, the exact average",
          cmd is not None and cmd[0].endswith("DTFE") and "--regionMpc" in cmd and "--exact-average" in cmd
          and cmd[cmd.index("--grid") + 1:cmd.index("--grid") + 4].count("1") == 1
          and "--" not in w.explore_view.title.text(), str(cmd))
    x.zoom_res.setValue(64)                           # (64 is the spin box's minimum)
    ref = region_mean(x, img.height() // 4, img.width() // 4, img.height() // 2, img.width() // 2)
    x._dragged(img.height() // 4, img.width() // 4, img.height() // 2, img.width() // 2)
    ok = wait_for(lambda: x.is_zoomed() and not x._zoom_pending and bool(x._zoom.get("exact")), 180)
    plane = x._plane
    got = None if plane is None else float(np.nanmean(plane))
    check("an exact zoom runs the binary on the region and shows its cells: their mean density within 5% of the "
          "slice's mean over the same region on disk",
          ok and plane is not None and plane.shape == (64, 64) and float(np.nanmin(plane)) >= 0.0
          and got is not None and abs(got - ref) <= 0.05 * ref and "exact zoom 64²" in w.explore_view.title.text(),
          f"{x.zoom_state.text()} | zoom mean {got} vs slice region mean {ref}")
    x.field.setCurrentIndex(x.field.findData("gradient"))
    wait_for(lambda: not x._busy and x._req["field"] == "gradient", 30)
    check("... and the velocity gradient of the same run", x.is_zoomed() and x._zoom.get("exact") and x._plane.shape == (64, 64))
    w.zoom_out_action.trigger()
    wait_for(lambda: not x.is_zoomed() and not x._busy, 30)
    x.field.setCurrentIndex(x.field.findData("density"))
    # the phase-space exact zoom of a NON-periodic run of the demo snapshot (the periodic box read as a
    # finite cloud): the same windowed run, without --periodic
    np_spec = rs.CustomSpec(input_file=str(demo.input_file), estimator="ps", grid=32, deposit="exact",
                            output_dir=str(tmp / "demo_np"), mpc_unit=demo.mpc_unit, periodic=False, gpu=False,
                            fields=["density_a", "velocity_a"], caustics=False, volume_weighted=True)
    w.custom = np_spec
    w._load_into_widgets()
    w.tabs.setCurrentIndex(A.OWN)
    qa.processEvents()
    w.run_btn.click()
    wait_idle(w, 300)
    nroot = np_spec.output_root()
    check("a non-periodic phase-space run of the demo snapshot writes its grids", Path(str(nroot) + ".a_den").is_file(),
          w.status.text())
    w._explore_output(str(nroot))
    ok = wait_for(lambda: w.explore._plane is not None and not w.explore._busy and x.current is not None
                  and str(x.current.root) == str(nroot), 30)
    geo = x._geometry()
    cmd, _r, _g, _e = x.exact_zoom_command(x.current, geo[0], geo[1], geo[2], (geo[3][0], geo[4][0]), (geo[3][1], geo[4][1]), 64)
    check("the exact zoom's command for a non-periodic phase-space run: PS-DTFE on the whole cloud, a window, the exact deposit",
          ok and cmd is not None and cmd[0].endswith("PS-DTFE") and "--ps-window" in cmd and "--ps-exact-deposit" in cmd
          and "--periodic" not in cmd and "--box" not in cmd and "--ps-volume-weighted" in cmd, str(cmd))
    img = w.explore_view.canvas._img
    x.zoom_res.setValue(64)
    sq = (img.height() // 4, img.width() // 4, img.height() // 2 - 1, img.width() // 2 - 1)   # 8 of 32 cells
    ref = region_mean(x, *sq)
    x._dragged(*sq)
    ok = wait_for(lambda: x.is_zoomed() and not x._zoom_pending and bool(x._zoom.get("exact")), 180)
    plane = x._plane
    got = None if plane is None else float(np.nanmean(plane))
    check("an exact zoom of the non-periodic phase-space run (the exact deposit on disk too): the region's cells, "
          "their mean density equal to the slice's mean over the same cells to 1e-4 (the window IS the full run's "
          "region, on a finer grid)",
          ok and plane is not None and plane.shape == (64, 64) and float(np.nanmin(plane)) >= 0.0
          and got is not None and abs(got - ref) <= 1e-4 * ref,
          f"{x.zoom_state.text()} | zoom mean {got} vs slice region mean {ref}")
    x.field.setCurrentIndex(x.field.findData("streams"))
    wait_for(lambda: not x._busy and x._req["field"] == "streams", 30)
    check("... and its stream count", x.is_zoomed() and x._plane is not None and float(np.nanmax(x._plane)) >= 1.0)
    x.zoom_exact.setChecked(False)
    w.zoom_out_action.trigger()
    wait_for(lambda: not x.is_zoomed() and not x._busy, 30)
    x.field.setCurrentIndex(x.field.findData("density"))
    w._explore_output(str(root))                       # back to the phase-space demo for what follows
    wait_for(lambda: not x._busy and x.current is not None and str(x.current.root) == str(root), 30)
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
    two_d_gui(qa, w, tmp)                             # (Explore still alive: it shuts down next)
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
    check("setup: ... and whether PS-DTFE has the TBB parallel triangulation",
          ("parallel triangulation (TBB)" in dlg.prog_info.text()) == rs.tbb_built()
          and dlg.prog_info.text().count("TBB") >= 1, dlg.prog_info.text())
    check("setup: the double-precision pair is listed (built or as the optional pair), with a tick to build it",
          "PS-DTFE-double" in dlg.prog_info.text() and "DTFE-double" in dlg.prog_info.text()
          and dlg.build_double.isChecked() == (rs.binary_built("ps", "double") and rs.binary_built("dtfe", "double"))
          and ("DOUBLE=1" in dlg.build_note.text()) == dlg.build_double.isChecked(), dlg.build_note.text())
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

    print("the job clock, the log, the window, drops:")
    check("a job's wall clock: 'took ...' beside the bar once it ends", w.clock.text().startswith("took "), w.clock.text())
    check("clock and duration texts", A.clock_text(7) == "0:07" and A.clock_text(760) == "12:40"
          and A.clock_text(7509) == "2:05:09" and A.took_text(7) == "7 s" and A.took_text(760) == "12m 40s"
          and A.took_text(7500) == "2h 05m")
    w.log.setPlainText("alpha\nbeta one\ngamma\nBETA two\n!! step: exit 1\n")
    found = []
    for back in (False, False, False, True):
        ok = w.find_in_log("beta", back=back)
        found.append((ok, w.log.textCursor().selectedText(), w.log_find_count.text()))
    check("find in the log: the next match (any case), the count, wrapping at the end, and backwards",
          found == [(True, "beta", "1 of 2"), (True, "BETA", "2 of 2"), (True, "beta", "1 of 2"), (True, "BETA", "2 of 2")],
          str(found))
    c_ = w.log.textCursor()
    c_.movePosition(c_.MoveOperation.Start)
    w.log.setTextCursor(c_)
    ok = w.find_in_log(A.PROBLEM_LINES, regex=True)
    check("... '!' goes to the next problem line", ok and w.log.textCursor().selectedText() == "!! ",
          w.log.textCursor().selectedText())
    pos_before = w.log.textCursor().position()
    check("... no match says so, and leaves the log where it was (it keeps following a run)",
          not w.find_in_log("zeta") and w.log_find_count.text() == "no match" and w.log.textCursor().position() == pos_before)
    w.copy_log()
    check("Copy Log puts the whole log on the clipboard", QApplication.clipboard().text() == w.log.toPlainText())
    check("Open Run Log is offered exactly when the newest run's log file exists",
          w.open_log_action.isEnabled() == (bool(w._last_log_path) and Path(w._last_log_path).is_file()),
          w._last_log_path)
    geo_ini = str(tmp / "gui_geometry.ini")
    wg = A.MainWindow(remember=True, settings=QSettings(geo_ini, QSettings.IniFormat))
    wg.show()
    wg.resize(wg.minimumSizeHint().width() + 123, 777)
    qa.processEvents()
    wg.close()
    saved = QSettings(geo_ini, QSettings.IniFormat)
    wh = A.MainWindow(remember=True, settings=QSettings(geo_ini, QSettings.IniFormat))
    wh.show()
    qa.processEvents()
    wf = A.MainWindow(remember=False, settings=QSettings(geo_ini, QSettings.IniFormat))
    # restoreGeometry fits a window onto the current screen -- offscreen that is 800 x 800, smaller than the
    # launcher's minimum width, so the exact size is only seen on a real screen; here: saved, and used
    check("the window's geometry and pane split are saved on close and used at the next start (not by a window "
          "that does not remember: 1400 x 900)",
          saved.value("geometry") is not None and saved.value("splitter") is not None
          and wh.size().toTuple() != (1400, 900) and wf.size().toTuple() == (1400, 900),
          f"{wh.size().toTuple()} {wf.size().toTuple()}")
    wh.close()
    # drops on a window of their own: a dropped folder changes the data root, which the main window's later
    # sections must keep
    did = wf.open_dropped(str(root) + ".a_den")
    check("a grid file dropped on the window opens in Explore", did == "explore" and wf.tabs.currentIndex() == A.EXPLORE
          and str(wf.explore.current.root) == str(root), did)
    snap_file = Path(demo.input_file)
    did = wf.open_dropped(str(snap_file))
    check("... a snapshot file (.hdf5) becomes the Custom tab's input", did == "custom" and wf.tabs.currentIndex() == A.OWN
          and wf.o_file.text() == str(snap_file), did)
    did = wf.open_dropped(str(tmp))
    check("... a folder becomes the data root", did == "root" and wf.root_edit.text() == str(tmp), did)
    (tmp / "notes.txt").write_text("x")
    did = wf.open_dropped(str(tmp / "notes.txt"))
    check("... anything else is refused with a reason", did == "" and "notes.txt" in wf.status.text(), wf.status.text())
    wf.close()                                     # closeEvent shuts its Explore threads down
    wo = A.MainWindow(rs.RunSpec(data_root=str(data), sim=SIM, snapshots=[99]), remember=False,
                      settings=QSettings(str(tmp / "gui_opts.ini"), QSettings.IniFormat))
    wo.dtfe_radio.setChecked(True)
    qa.processEvents()
    lam = wo.fields_box.lambda_spin
    check("standard DTFE keeps the fields it computes tickable (the T-web too); only the dispersion is greyed",
          [n for n, cb in wo.field_checks.items() if not cb.isEnabled()] == ["dispersion_a"])
    for web in ("tweb_a", "vweb_a"):
        wo.field_checks[web].setChecked(False)
    qa.processEvents()
    off = lam.isEnabled()
    wo.field_checks["tweb_a"].setChecked(True)
    qa.processEvents()
    lam.setValue(0.5)
    wo.budget_spin.setValue(40)
    wo.tess_edit.setText(str(tmp / "tess"))
    wo.tess_edit.editingFinished.emit()
    wo._read_widgets()
    env_ = wo.spec.command()[1]
    check("the T-web threshold is live only with a web class ticked; it, the memory budget, the tessellation cache "
          "and the T-web reach the standard-DTFE run",
          not off and lam.isEnabled() and env_.get("LAMBDA_TH") == "0.5" and env_.get("DTFE_MEM_BUDGET_GB") == "40"
          and env_.get("TESS_CACHE") == str(tmp / "tess") and "tweb_a" in env_.get("FIELDS", "").split(), str(env_))
    wo.custom = rs.CustomSpec(input_file=str(demo.input_file), mem_budget_gb=12, tess_cache=str(tmp / "tess"))
    wo._load_into_widgets()
    last = wo.custom.steps()[-1]
    check("the Custom tab shows them, and its run passes --tessellation-cache and the budget in the binary's environment",
          wo.o_budget.value() == 12 and wo.o_tess.text() == str(tmp / "tess") and "--tessellation-cache" in last.argv
          and last.env.get("DTFE_MEM_BUDGET_GB") == "12", f"{last.argv} {last.env}")
    import time as _time
    now_ = _time.monotonic()
    wo._total_snaps, wo._snap_t0s, wo._per_snap_prior = 4, [now_ - 300, now_ - 200], None
    left2 = wo._job_left(now_)
    wo._snap_t0s, wo._per_snap_prior = [now_ - 20], 50.0
    left1 = wo._job_left(now_)
    wo._total_snaps, wo._snap_t0s = 0, []
    check("the job ETA: snapshots left at the pace of the done ones (earlier runs' pace before the first ends), "
          "minus what the current one has run", abs(left2 - 200) < 1e-6 and abs(left1 - (3 * 50 + 30)) < 1e-6,
          f"{left2} {left1}")
    wo.close()


def two_d_gui(qa, w, tmp):
    """The 2D programs through the launcher: a 2D file is recognised, runs on the -2d binary, and
    Explore shows it as one plane, queries it, zooms into it (point and exact)."""
    import json
    import setup_wizard as SW

    print("2D:")
    if not (rs.binary_built("ps", dim=2) and rs.binary_built("dtfe", dim=2)):
        print("   SKIP  the 2D programs are not built (make DTFE PS-DTFE DIM=2)")
        return
    x = w.explore
    if x.server_phase() != "stopped":
        w.server_action.trigger()
        wait_for(lambda: x.server_phase() == "stopped", 30)
    snap = tmp / "demo2d" / "plane.hdf5"
    snap.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(rs.DEMO_GENERATOR), "--out", str(snap), "--n", "48", "--box", "100",
                    "--amplitude-factor", "1.5", "--crossed-waves", "--dim", "2"], check=True, capture_output=True)
    w.custom = rs.CustomSpec(output_dir=str(tmp / "out2d"), grid=64, estimator="ps", gpu=True,
                             fields=["density_a", "velocity_a", "dispersion_a", "tweb_a"])
    w._load_into_widgets()
    w.tabs.setCurrentIndex(A.OWN)
    qa.processEvents()
    w._own_last_file = ""
    w.o_file.setText(str(snap))
    w._own_file_changed()
    qa.processEvents()
    check("a 2D file sets the dimensions to 2D; the GPU and parallel insertion boxes are off, the T-web's on; ² labels",
          w.o_dim.currentData() == 2 and w.custom.dim == 2 and not w.o_gpu.isEnabled() and not w.o_pt.isEnabled()
          and w.o_fields["tweb_a"].isEnabled() and w.o_grid_cells.text().startswith("²")
          and w.o_part.suffix().startswith("²") and "zlo" not in w.o_box.toolTip())
    text = w.cmd_view.toPlainText()
    check("... and the command runs PS-DTFE-2d with the T-web, without the GPU",
          "PS-DTFE-2d" in text and "--ps-gpu" not in text and "tweb_a" in text and w.run_btn.isEnabled(), text)
    w.run_btn.click()
    dt = wait_idle(w, 300)
    root = w.custom.output_root()
    side = json.loads(Path(str(root) + ".gui.json").read_text())
    check(f"the 2D run ({dt:.0f} s) writes 64² grids and a sidecar that says 2D",
          w.status.text() == "Finished" and Path(str(root) + ".a_den").stat().st_size == 64 * 64 * 4
          and side["dim"] == 2 and side["box_mpc"] == [0, 100] * 2, w.status.text())
    w._explore_output(str(root))
    ok = wait_for(lambda: x._plane is not None and not x._busy and x.current is not None
                  and str(x.current.root) == str(root), 30)
    o = x.current
    check("Explore shows the 2D output as its one plane: no axis or slice to pick",
          ok and o.dim == 2 and x._plane.shape == (64, 64) and "64² grid (2D)" in x.info.text()
          and not x.axis.isEnabled() and not x.index.isEnabled() and not x.slider.isEnabled()
          and "2D" in x.position.text(), x.info.text())
    x.field.setCurrentIndex(x.field.findData("tweb_eigenvalues"))
    wait_for(lambda: not x._busy and x._req["field"] == "tweb_eigenvalues", 30)
    ecomps = [x.component.itemText(i) for i in range(x.component.count())]
    check("the 2D T-web: its classes and two eigenvalues are maps",
          "tweb" in o.fields() and o.files["tweb_eigenvalues"].ncomp == 2 and ecomps == ["1", "2"]
          and x._plane is not None and x._plane.shape == (64, 64), f"{o.fields()} {ecomps}")
    x.field.setCurrentIndex(x.field.findData("velocity"))
    wait_for(lambda: not x._busy and x._req["field"] == "velocity", 30)
    comps = [x.component.itemText(i) for i in range(x.component.count())]
    check("2D velocity: norm, x, y, and arrows", comps == ["norm", "x", "y"] and x.vectors.isEnabled()
          and w.explore_view.canvas.arrows is not None, str(comps))
    x.field.setCurrentIndex(x.field.findData("density"))
    wait_for(lambda: not x._busy and x._req["field"] == "density", 30)
    x.cache_edit.setText(str(tmp / "tess2d"))
    x._cache_user_set = True
    w.server_action.trigger()
    wait_for(lambda: not x._server_starting, 120)
    check("the query server starts on the 2D snapshot (the -2d binary)", x._server_running, x.server_state.text())
    img = w.explore_view.canvas._img
    w.explore_view.streams.setRowCount(0)             # (the rows of the 3D query above)
    x._click(img.height() // 2, img.width() // 3)
    wait_for(lambda: "asking" not in w.explore_view.answer_title.text()
             and (w.explore_view.streams.rowCount() > 0 or "⚠" in w.explore_view.answer_title.text()), 30)
    title = w.explore_view.answer_title.text()
    q = w.explore_view.streams.item(0, 4).text() if w.explore_view.streams.rowCount() else ""
    check("a click lists the streams at a 2D point, positions and origins with two coordinates",
          "stream" in title and title.split("Mpc")[0].count(",") == 1 and q.startswith("(") and q.count(",") == 1,
          f"{title} | {q}")
    x.zoom_exact.setChecked(False)
    x.zoom_res.setValue(64)
    x._dragged(img.height() // 4, img.width() // 4, img.height() // 2, img.width() // 2)
    ok = wait_for(lambda: x.is_zoomed() and not x._zoom_pending, 60)
    check("a point zoom of the 2D plane", ok and x._plane.shape == (64, 64) and "zoom 64²" in w.explore_view.title.text(),
          x.zoom_state.text())
    f0 = x._zoom_fields
    shown = {}
    for name in ("velocity", "dispersion", "streams"):
        x.field.setCurrentIndex(x.field.findData(name))
        qa.processEvents()
        shown[name] = x._plane is not None and x._plane.shape == (64, 64) and bool(np.isfinite(x._plane).all())
    check("... every field of the zoom from the one evaluation (2-component velocity, 3-component dispersion)",
          all(shown.values()) and x._zoom_fields is f0, str(shown))
    x.field.setCurrentIndex(x.field.findData("density"))
    qa.processEvents()
    x.zoom_out()
    wait_for(lambda: not x.is_zoomed() and not x._busy, 30)
    x.zoom_exact.setChecked(True)
    geo = x._geometry()
    cmd, _r, grid_, _e = x.exact_zoom_command(o, geo[0], geo[1], geo[2], (geo[3][0], geo[4][0]),
                                             (geo[3][1], geo[4][1]), 64)
    wi = cmd.index("--ps-window") if cmd and "--ps-window" in cmd else -1
    gi = cmd.index("--grid") if cmd and "--grid" in cmd else -1
    check("the 2D exact zoom's command: PS-DTFE-2d, a 4-number window of a 2-number virtual grid, no GPU",
          cmd is not None and Path(cmd[0]).name == "PS-DTFE-2d" and wi > 0 and cmd[wi + 5] == "--field"
          and gi > 0 and cmd[gi + 3].startswith("--") and "--ps-gpu" not in cmd and grid_ == [64, 64], str(cmd))
    sq = (img.height() // 4, img.width() // 4, img.height() // 2 - 1, img.width() // 2 - 1)
    ref = region_mean(x, *sq)
    x._dragged(*sq)
    ok = wait_for(lambda: x.is_zoomed() and not x._zoom_pending and bool(x._zoom.get("exact")) and not x._exact_running, 300)
    m = None if x._plane is None else float(np.nanmean(x._plane))
    check("an exact zoom of the 2D plane: its cells' mean density within 5% of the run's over the same region",
          ok and x._plane.shape == (64, 64) and m is not None and abs(m - ref) <= 0.05 * ref
          and "exact zoom 64²" in w.explore_view.title.text(), f"{x.zoom_state.text()} | {m} vs {ref}")
    x.zoom_out()
    wait_for(lambda: not x.is_zoomed() and not x._busy, 30)
    w.server_action.trigger()
    wait_for(lambda: x.server_phase() == "stopped", 30)

    d = rs.CustomSpec(input_file=str(snap), dim=2, estimator="dtfe", grid=48, output_dir=str(tmp / "out2d_dtfe"))
    w.custom = d
    w._load_into_widgets()
    w.tabs.setCurrentIndex(A.OWN)
    qa.processEvents()
    w.run_btn.click()
    wait_idle(w, 300)
    droot = d.output_root()
    w._explore_output(str(droot))
    ok = wait_for(lambda: x._plane is not None and not x._busy and x.current is not None
                  and str(x.current.root) == str(droot), 30)
    check("a standard DTFE-2d run opens in Explore as a 48² plane", ok and x.current.dim == 2
          and x._plane.shape == (48, 48), w.status.text())
    geo = x._geometry()
    cmd, _r, grid_, _e = x.exact_zoom_command(x.current, geo[0], geo[1], geo[2], (geo[3][0], geo[4][0]),
                                             (geo[3][1], geo[4][1]), 64)
    ri = cmd.index("--regionMpc") if cmd and "--regionMpc" in cmd else -1
    check("... its exact zoom: DTFE-2d on a 4-number region, a 64 x 64 grid, the exact average",
          cmd is not None and Path(cmd[0]).name == "DTFE-2d" and ri > 0 and cmd[ri + 5] == "--field"
          and cmd[cmd.index("--grid") + 1:cmd.index("--grid") + 3] == ["64", "64"] and "--exact-average" in cmd
          and "--gpu" not in cmd, str(cmd))
    x._dragged(img.height() // 4, img.width() // 4, img.height() // 2, img.width() // 2)
    ok = wait_for(lambda: x.is_zoomed() and not x._zoom_pending and bool(x._zoom.get("exact")) and not x._exact_running, 120)
    check("... which runs and shows 64² positive cells", ok and x._plane.shape == (64, 64)
          and float(np.nanmin(x._plane)) >= 0.0, x.zoom_state.text())
    x.zoom_out()
    wait_for(lambda: not x.is_zoomed() and not x._busy, 30)
    x.zoom_exact.setChecked(False)

    dlg = SW.SetupDialog(w, str(tmp))
    dlg._go(1)
    check("setup: the 2D programs are listed, ticked to be rebuilt when they exist (make ... DIM=2)",
          "PS-DTFE-2d" in dlg.prog_info.text() and dlg.build_2d.isChecked() and "DIM=2" in dlg.build_note.text()
          and "METAL" not in [c for c in SW.build_command(dim=2) if c.startswith("METAL")], dlg.build_note.text())
    dlg.reject()


if __name__ == "__main__":
    sys.exit(main())
