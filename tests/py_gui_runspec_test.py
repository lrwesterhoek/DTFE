#!/usr/bin/env python3
"""Headless checks for the GUI's job specifications (python/gui/runspec.py): the commands
RunSpec, PipelineSpec, PlotSpec and CustomSpec stand for, the combinations they refuse, script
export, progress parsing and data discovery; and the GUI's other Qt-free modules: run logs, stale
outputs, time estimates and failure advice (results.py), presets and the binary's own help text
(presets.py), and the Explore tab's grid reader (grids.py). No Qt needed.

Usage: python3 tests/py_gui_runspec_test.py
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "gui"))

import runspec as rs  # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"   {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))


def levels(spec):
    return {lvl for lvl, _ in spec.problems()}, " | ".join(m for _, m in spec.problems())


def main():
    tmp = Path(tempfile.mkdtemp(prefix="gui_runspec_")) / "Illustris TNG"
    try:
        # a fake per-family data root with one simulation, two snapshots, one plane file
        sim = tmp / "TNG100" / "TNG100-3-Dark"
        for n, z in ((4, 10.0), (99, 0.0)):
            d = sim / f"snapdir_{n:03d}"
            d.mkdir(parents=True)
            import h5py
            with h5py.File(d / f"combined_{n:03d}.hdf5", "w") as f:
                f.create_group("Header").attrs["Redshift"] = z
        (sim / "snapdir_004" / "ps_output.a_den").touch()
        (sim / "hires_plane_z64.bin").write_bytes(b"\0" * 24 * 64 * 64)

        print("discovery:")
        check("simulations() finds the per-family sim", rs.simulations(tmp) == ["TNG100-3-Dark"])
        snaps = rs.snapshots(sim)
        check("snapshots() lists numbers and redshifts",
              [(s["n"], s["z"]) for s in snaps] == [(4, 10.0), (99, 0.0)], str(snaps))
        check("snapshots() marks existing outputs", [s["done"] for s in snaps] == [True, False])
        check("planes() + plane_points()",
              [p.name for p in rs.planes(sim)] == ["hires_plane_z64.bin"]
              and rs.plane_points(sim / "hires_plane_z64.bin") == 64 * 64)

        print("command:")
        spec = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99, 4], grid=256,
                          nsub=2, gpu=True, deposit="exact", caustics=True,
                          slice_plane=str(sim / "hires_plane_z64.bin"), partition=4, max_concurrent=3,
                          output_prefix="ps_test")
        argv, env = spec.command()
        check("PS argv: script, sim, grid, nSub, GPU, exact, sorted snapshots",
              argv == [str(rs.PS_SCRIPT), "-s", "TNG100-3-Dark", "-g", "256", "-n", "2", "-m", "-e", "4", "99"],
              str(argv))
        check("PS env carries the physics, slice and resources",
              env["DTFE_DATA_ROOT"] == str(tmp) and env["PS_CAUSTICS"] == "1"
              and env["PS_VOLUME_WEIGHTED"] == "1" and env["PS_VERTEX_MASS"] == "1"
              and env["PARTITION"] == "4 4 4" and env["MAX_CONCURRENT"] == "3"
              and env["SAMPLE_POINTS"].endswith("hires_plane_z64.bin") and env["PTS_VEL_GRAD"] == "1"
              and env["OUTPUT_PREFIX"] == "ps_test" and "SCRATCH_DIR" not in env, str(env))
        line = spec.shell_line()
        check("shell_line is a runnable, quoted terminal command",
              line.startswith("DTFE_DATA_ROOT='") and "scripts/run_ps_dtfe.sh -s TNG100-3-Dark" in line
              and line.endswith(" 4 99"), line)
        auto = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4])
        check("auto partition/concurrency pass nothing (the binary auto-tunes)",
              "PARTITION" not in auto.command()[1] and "MAX_CONCURRENT" not in auto.command()[1])
        dt = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], estimator="dtfe", gpu=False)
        check("DTFE argv uses run_dtfe.sh with no PS options",
              dt.command()[0] == [str(rs.DTFE_SCRIPT), "-s", "TNG100-3-Dark", "-g", "512", "4"]
              and "FIELDS" not in dt.command()[1])
        dte = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], estimator="dtfe", gpu=False,
                         deposit="exact")
        check("standard DTFE with the exact choice: run_dtfe.sh -e (the exact cell average), named in the queue",
              dte.command()[0] == [str(rs.DTFE_SCRIPT), "-s", "TNG100-3-Dark", "-g", "512", "-e", "4"]
              and rs.job_summary("grids", dte).endswith(", exact"), str(dte.command()[0]))
        check("to_dict/from_dict round trip", rs.RunSpec.from_dict(spec.to_dict()) == spec)

        print("checks:")
        lv, msg = levels(rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], grid=128))
        check("a sane small run has no errors", "error" not in lv, msg)
        lv, msg = levels(rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[]))
        check("no snapshots is an error", "error" in lv and "no snapshots" in msg, msg)
        lv, msg = levels(rs.RunSpec(data_root=str(tmp / "missing"), sim="TNG100-3-Dark", snapshots=[4]))
        check("a missing data root is an error naming the T7", "error" in lv and "T7" in msg, msg)
        lv, msg = levels(rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4],
                                    caustic_cusps=True, caustics=False))
        check("cusps without caustics is an error", "error" in lv and "cusps" in msg, msg)
        icloud = tmp.parent / "Mobile Documents" / "scratch"      # iCloud Drive's path component
        icloud.mkdir(parents=True)                               # (not the repo's own location: CI)
        lv, msg = levels(rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4],
                                    scratch_dir=str(icloud)))
        check("an iCloud scratch directory is an error", "error" in lv and "iCloud" in msg, msg)
        lv, msg = levels(rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], grid=1024))
        check("1024^3 without scratch warns about memory", "warning" in lv and "scratch" in msg, msg)
        ok = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], grid=1024,
                        scratch_dir=str(tmp.parent))
        check("... and a scratch directory clears the warning",
              not any("scratch" in m for _, m in ok.problems()), " | ".join(m for _, m in ok.problems()))
        lv, msg = levels(rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], output_prefix="a b"))
        check("an output prefix with a space is an error", "error" in lv, msg)

        print("progress:")
        p = rs.parse_progress("Processing snapshot 033...\n\t Done:  1% 2%  [partitions 45/125 | 36%]  "
                              "elapsed 8m 40s, ETA ~15m 25s\r")
        check("partition progress and snapshot parsed",
              p.get("snapshot") == 33 and p.get("done") == 45 and p.get("total") == 125
              and p.get("percent") == 36 and p.get("elapsed") == "8m 40s", str(p))
        check("ANSI colour codes stripped", rs.strip_ansi("\x1b[1mPS-DTFE:\x1b[0m ok") == "PS-DTFE: ok")
        check("snapshot completion detected",
              rs.parse_progress("  Snapshot 033 processed successfully\n").get("snapshot_finished") is True)

        print("pipeline:")
        pipe = rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=64, scratch_dir=str(tmp.parent))
        argv, env = pipe.command()
        check("pipeline: the script, SIMS/NU/AXIS/GRID_SIZE always, defaults otherwise omitted",
              argv == [str(rs.PIPELINE_SCRIPT)] and env["SIMS"] == "TNG100-3-Dark" and env["NU"] == "64"
              and env["AXIS"] == "z" and env["GRID_SIZE"] == "512" and env["PTS_VEL_GRAD"] == "1"
              and "FIELDS" not in env and "OUTPUT_PREFIX" not in env and "FORCE" not in env
              and "DRY_RUN" not in env and env["SCRATCH_DIR"] == str(tmp.parent), str(env))
        st = pipe.steps()
        check("... then plot_pointeval.py pinned to the pipeline's own plane",
              len(st) == 2 and st[1].argv[1].endswith("plot_pointeval.py")
              and st[1].argv[st[1].argv.index("--plane") + 1] == str(sim / "hires_plane_z64.json")
              and st[1].env["MPLBACKEND"] == "Agg", str(st[1].argv if len(st) > 1 else st))
        pipe.plan_only = True
        check("plan only: DRY_RUN=1 and no render step",
              len(pipe.steps()) == 1 and pipe.command()[1]["DRY_RUN"] == "1")
        pipe.plan_only = False
        check("stale(): no slices yet -> every snapshot", pipe.stale("TNG100-3-Dark") == ([4, 99], 2))
        (sim / "hires_plane_z64.bin").touch()
        # a sidecar as make_image_plane.py writes it (an unreadable one counts as 'differs' since 2026-10-04)
        (sim / "hires_plane_z64.json").write_text('{"box": 100.0, "center": 50.0, "planes": 1, "thickness": 2.0, '
                                                   '"nu": 64, "nv": 64, "u0": 0.0, "u1": 100.0, "v0": 0.0, "v1": 100.0}')
        for n in (4, 99):
            for ext in ("pts_den", "pts_velGrad"):
                (sim / f"snapdir_{n:03d}" / f"ps_output.{ext}").touch()
        # explicit times: "newer" is STRICT (the script's -nt), and Linux stamps files from a
        # coarse clock, so files made milliseconds apart can share one mtime (failed on Ubuntu)
        now = time.time()
        for f in [sim / "hires_plane_z64.bin", sim / "hires_plane_z64.json"] \
                + [sim / f"snapdir_{n:03d}" / f"combined_{n:03d}.hdf5" for n in (4, 99)]:
            os.utime(f, (now - 100, now - 100))
        for n in (4, 99):
            for ext in ("pts_den", "pts_velGrad"):
                os.utime(sim / f"snapdir_{n:03d}" / f"ps_output.{ext}", (now, now))
        (sim / "snapdir_099" / "ps_output.pts_velGrad").unlink()
        check("stale(): fresh slices skipped, a missing .pts_velGrad is stale (the script's rule)",
              pipe.stale("TNG100-3-Dark") == ([99], 2), str(pipe.stale("TNG100-3-Dark")))
        pipe.force = True
        check("stale(): FORCE recomputes everything", pipe.stale("TNG100-3-Dark") == ([4, 99], 2))
        pipe.force = False
        lv, msg = levels(rs.PipelineSpec(data_root=str(tmp), sims=[]))
        check("pipeline: no simulations is an error", "error" in lv and "no simulations" in msg, msg)
        lv, msg = levels(rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], center="half"))
        check("pipeline: a non-numeric centre is an error", "error" in lv and "centre" in msg, msg)
        lv, msg = levels(rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=8192))
        check("pipeline: an 8192^2 plane without scratch warns (it did not fit on 64 GB)",
              "warning" in lv and "scratch" in msg, msg)
        pipeline_options(tmp, sim)

        print("plots:")
        (sim / "snapdir_099" / "ps_output.a_den").touch()
        (sim / "snapdir_004" / "ps_output.a_den").unlink()       # 004: no grids now
        (sim / "pointeval_plane_64_x.json").touch()
        (sim / "pointeval_plane_64_x.bin").write_bytes(b"\0" * 24 * 64 * 64)
        every = rs.PlotSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99, 4],
                            sets=[f.key for f in rs.FIGURE_SETS], method="ps", smooth=1.5,
                            options={"pointeval": {"fields": "density, velDiv", "force": True},
                                     "panels": {"plane": str(sim / "hires_plane_z64.json")},
                                     "halotrack": {"subhalos": "0, 17"},
                                     "thesis": {"only": "plot_DTFE.py"}})
        by = {}
        for stp in every.steps():
            by.setdefault(Path(stp.argv[1]).name, []).append(stp.argv[2:])
        check("per-snapshot sets: one call per snapshot, sorted, with the common flags",
              by["plot_PS_DTFE.py"] == [["--sim", "TNG100-3-Dark", "--snap", n, "--method", "ps", "--smooth", "1.5"]
                                        for n in ("4", "99")] and len(by["plot_cosmic_web.py"]) == 2,
              str(by.get("plot_PS_DTFE.py")))
        check("the series is one call with every snapshot",
              by["plot_DTFE.py"] == [["--sim", "TNG100-3-Dark", "--snaps", "4", "99", "--method", "ps",
                                      "--smooth", "1.5"]], str(by["plot_DTFE.py"]))
        check("point-eval maps: comma lists, fields de-spaced, --force, NO grid smoothing",
              by["plot_pointeval.py"] == [["--sims", "TNG100-3-Dark", "--snaps", "4,99", "--fields",
                                           "density,velDiv", "--force"]], str(by["plot_pointeval.py"]))
        pan = by["plot_pointeval_panels.py"]
        check("panels: per snapshot, its own grid prefix, the plane, an -o under python/figures",
              len(pan) == 2 and pan[0][:2] == ["--prefix", str(sim / "snapdir_004" / "ps_output")]
              and pan[0][pan[0].index("-o") + 1].startswith(str(rs.FIGURES_ROOT / "pointeval_panels")), str(pan))
        check("halo tracking: explicit IDs replace --top",
              by["plot_halo_tracking.py"][0][-3:] == ["--subhalo", "0", "17"], str(by["plot_halo_tracking.py"]))
        check("thesis set: analyze.py all, zero-padded snapshots, --only",
              by["analyze.py"] == [["all", "004", "099", "--only", "plot_DTFE.py"]], str(by["analyze.py"]))
        check("every plot step sets DTFE_SIM and MPLBACKEND=Agg (several scripts call plt.show())",
              all(stp.env["DTFE_SIM"] == "TNG100-3-Dark" and stp.env["MPLBACKEND"] == "Agg"
                  for stp in every.steps()))
        lv, msg = levels(every)
        check("plots: snapshot 004 has no grids -> warned", "004: those figures will fail" in msg, msg)
        check("plots: two 64^2 planes -> 'auto' is ambiguous", "same pixel count" in msg, msg)
        lv, msg = levels(rs.PlotSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99], sets=["panels"]))
        check("panels without a plane is an error", "error" in lv and "choose the image plane" in msg, msg)
        lv, msg = levels(rs.PlotSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[], sets=["fields"]))
        check("per-snapshot figures without snapshots is an error", "error" in lv and "no snapshots" in msg, msg)
        lv, msg = levels(rs.PlotSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[], sets=["voidpop"]))
        check("... but a per-simulation set needs none", "error" not in lv, msg)
        check("from_dict drops figure sets that no longer exist",
              rs.PlotSpec.from_dict({"sets": ["fields", "gone"]}).sets == ["fields"])
        t0 = time.time()
        time.sleep(0.05)
        (tmp / "figs").mkdir()
        (tmp / "figs" / "new.png").touch()
        check("figures(since=...) lists only what a run wrote",
              [p.name for p in rs.figures([tmp / "figs", ROOT / "python" / "gui"], since=t0)] == ["new.png"])
        check("pipeline plan line parsed (snapshots to compute, simulation)",
              rs.parse_progress("+ env A=1 /x/scripts/run_ps_dtfe.sh -s TNG100-3-Dark -g 512 -m 50 67 99\n"
                                "== TNG100-3-Dark: snapshots 0 4") == {"planned": 3, "sim": "TNG100-3-Dark"})

        print("download + merge:")
        check("the redshift ladder comes from config.py (16 snapshots, z(99) = 0)",
              len(rs.SNAPSHOT_LADDER) == 16 and rs.SNAPSHOT_LADDER.get(99) == 0.0 and rs.SNAPSHOT_LADDER.get(0) == 20.05)
        dsim = tmp / "TNG50" / "TNG50-4-Dark"                  # family folder exists, the run does not
        (tmp / "TNG50").mkdir()
        key_file, saved_key = rs.API_KEY_FILE, os.environ.pop("TNG_API_KEY", None)
        rs.API_KEY_FILE = tmp.parent / "no_key_here"
        try:
            d = rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[99, 50], download_groupcats=True,
                            download_trees=True, download_ics=True, merge_groupcats=True, merge_trees=True,
                            convert_ics=True, delete_chunks=True)
            got = [(st.label, st.argv[1:] if st.argv[0] != rs.PYTHON else st.argv[2:]) for st in d.steps()]
            check("downloads (snapshots, groups, trees, ICs) first, then merges, into the family folder",
                  [lbl for lbl, _ in got] == ["download snapshots 50 99", "download group catalogues 50 99",
                                              "download merger trees", "download initial conditions",
                                              "merge snapshots 50 99", "merge group catalogues 50 99",
                                              "merge merger trees", "convert initial conditions"]
                  and got[0][1] == ["-s", "TNG50-4-Dark", "50", "99"] and got[1][1][:1] == ["-c"]
                  and got[4][1] == ["-d", str(dsim), "50", "99", "--delete-chunks"]
                  and all(a[-1] == "--delete-chunks" for _, a in got[4:]), str(got))
            check("no -k on any command line (the key stays in its file / env)",
                  all("-k" not in st.argv for st in d.steps()))
            lv, msg = levels(d)
            check("no API key is an error naming the key file", "error" in lv and "no_key_here" in msg, msg)
            check("--delete-chunks is warned about", "only copy" in msg, msg)
            os.environ["TNG_API_KEY"] = "x"
            lv, msg = levels(d)
            check("... TNG_API_KEY clears it", "API key" not in msg, msg)
            # 64 x 615 GB: more than any disk here
            lv, msg = levels(rs.DataSpec(data_root=str(tmp), sim="TNG50-1-Dark", snapshots=list(range(64))))
            check("a download larger than the free space is an error", "error" in lv and "GB free" in msg, msg)
            lv, msg = levels(rs.DataSpec(data_root=str(tmp), sim="TNG50-4", snapshots=[99]))
            check("a full-physics run warns about gas and stars", "gas and stars" in msg, msg)
            lv, msg = levels(rs.DataSpec(data_root=str(tmp), sim="Millennium", snapshots=[99]))
            check("an unknown run warns", "not a TNG run" in msg, msg)
            lv, msg = levels(rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[],
                                         download_snapshots=False, merge_snapshots=False))
            check("nothing ticked is an error", "error" in lv and "nothing selected" in msg, msg)
            # on disk: 050 merged (+ chunks), 099 two chunks, group catalogue 099 merged, trees merged
            for n, files in ((50, ["snap_050.0.hdf5", "combined_050.hdf5"]), (99, ["snap_099.0.hdf5", "snap_099.1.hdf5"])):
                (dsim / f"snapdir_{n:03d}").mkdir(parents=True)
                for f in files:
                    (dsim / f"snapdir_{n:03d}" / f).touch()
            (dsim / "groups_099").mkdir()
            (dsim / "groups_099" / "combined_fof_subhalo_tab_099.hdf5").touch()
            (dsim / "Merger Trees").mkdir()
            (dsim / "Merger Trees" / "combined_tree_extended.hdf5").touch()
            check("snapshot_state / sim_state read the disk",
                  rs.snapshot_state(dsim, 50) == {"chunks": 1, "merged": True, "gc_chunks": 0, "gc_merged": False}
                  and rs.snapshot_state(dsim, 99) == {"chunks": 2, "merged": False, "gc_chunks": 0, "gc_merged": True}
                  and rs.sim_state(dsim) == {"tree_chunks": 0, "tree_merged": True, "ics_raw": False,
                                             "ics_converted": False})
            labels = [st.label for st in d.steps()]
            check("merged items are left out of every step (re-merging would stale their grids)",
                  labels == ["download snapshots 99", "download group catalogues 50", "download initial conditions",
                             "merge snapshots 99", "merge group catalogues 50", "convert initial conditions"],
                  str(labels))
            lv, msg = levels(d)
            check("... and named in a warning", "left out: snapshot 050, groups 099, merger trees" in msg, msg)
            n4 = 270 ** 3
            check("size estimate skips merged snapshots and chunks already on disk",
                  rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[50, 99, 67]).download_bytes()
                  == (n4 * 61, 2 * n4 * 32))
            check("the ladder plus what is on disk", 99 in rs.data_snapshots(dsim) and 8 in rs.data_snapshots(dsim))
            done = rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[50], download_snapshots=True)
            lv, msg = levels(done)
            check("everything merged: nothing left to do (an error, so Run stays off)",
                  done.steps() == [] and "nothing left to do" in msg, msg)
        finally:
            rs.API_KEY_FILE = key_file
            os.environ.pop("TNG_API_KEY", None)
            if saved_key is not None:
                os.environ["TNG_API_KEY"] = saved_key
        p = rs.parse_progress('2026-09-29 12:00:00 URL:https://x/files/snapshot-99/snap_099.0.hdf5 [1234/1234] '
                              '-> "/a b/snap_099.0.hdf5" [1]\n  [2/4] Reading snap_099.1.hdf5...\n'
                              '  Error downloading snapshot-5\n  Particles merged: [8]  (MISMATCH vs header total [9])')
        check("wget files, merge chunks and problem lines parsed",
              p.get("downloaded") == [("/a b/snap_099.0.hdf5", 1234)] and p.get("chunk") == (2, 4)
              and p.get("problem") == 2, str(p))
        os.environ["TNG_API_KEY"] = "k"
        try:
            dl_steps = [st for st in rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[67]).steps()
                        if "download" in st.label]
        finally:
            os.environ.pop("TNG_API_KEY", None)
        check("the download step carries PY (the script checks the chunks on disk with the merge's python)",
              dl_steps and all(st.env.get("PY") == rs.PYTHON for st in dl_steps), str([st.env for st in dl_steps]))
        os.environ["TNG_API_KEY"] = "k"
        try:
            both = rs.DataSpec(data_root=str(tmp), sim="TNG50-2-Dark", snapshots=[67], download_groupcats=True,
                               merge_groupcats=True, download_trees=True, merge_trees=True).steps()     # nothing on disk
            only = rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[67], download_snapshots=False).steps()
        finally:
            os.environ.pop("TNG_API_KEY", None)
        after = {st.label: st.after for st in both}
        check("every merge step is chained to its own download step (skipped by the launcher when that failed)",
              after.get("merge snapshots 67") == "download snapshots 67"
              and after.get("merge group catalogues 67") == "download group catalogues 67"
              and after.get("merge merger trees") == "download merger trees", str(after))
        check("a merge without a download step is chained to nothing", [st.after for st in only] == [""], str(only))
        # a stopped merge's leftover: named, and its bytes counted as free for the merge that removes it
        sp = rs.sim_dir("TNG50-4-Dark", str(tmp))
        stale = sp / "snapdir_067" / "combined_067.hdf5.partial"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"x" * 4096)
        try:
            d = rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[67], download_snapshots=False)
            lv, msg = levels(d)
            check("a stale .partial of a snapshot about to be merged is listed and named in a warning",
                  [p.name for p in d.stale_partials()] == ["combined_067.hdf5.partial"] and "a stopped merge left" in msg
                  and "combined_067.hdf5.partial" in msg, msg)
        finally:
            stale.unlink()

        print("memory check:")
        rep = rs.parse_reports("  AUTO-TUNE-REPORT snap=099 partition=5 mc=2 predicted_gb=33.75 budget_gb=34.88 "
                               "fixed_gb=20.75 pts_gb=15.47 streams_est=3.633 streams_budget=6.266 over_budget=0\n"
                               "Processing snapshot 004...\n  AUTO-TUNE-REPORT snap=004 partition=6 mc=1 "
                               "predicted_gb=95.0 budget_gb=41.2 over_budget=1")
        check("report lines parsed per snapshot (ints for split/concurrency/verdict, floats for GB)",
              rep == {99: {"partition": 5, "mc": 2, "predicted_gb": 33.75, "budget_gb": 34.88, "fixed_gb": 20.75,
                           "pts_gb": 15.47, "streams_est": 3.633, "streams_budget": 6.266, "over_budget": 0},
                      4: {"partition": 6, "mc": 1, "predicted_gb": 95.0, "budget_gb": 41.2, "over_budget": 1}},
              str(rep))
        grids = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99, 4], scratch_dir=str(tmp.parent))
        st = grids.check_step()
        check("a grids check is the SAME run_ps_dtfe.sh call plus AUTO_TUNE_REPORT=1",
              st.argv == grids.command()[0] and st.env == dict(grids.command()[1], AUTO_TUNE_REPORT="1"))
        check("the memory key ignores the snapshot choice but not the settings",
              rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4]).memory_key()
              == rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99]).memory_key()
              != rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], grid=256).memory_key())
        pm = rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=64, grid=128, scratch_dir=str(tmp.parent))
        have = pm.check_step("TNG100-3-Dark", [99, 4], 64)                 # hires_plane_z64.bin exists
        need = pm.check_step("TNG100-3-Dark", [99], 32)                    # hires_plane_z32 does not
        check("a pipeline check passes the pipeline's settings, the plane file when it exists ...",
              have.argv[1:] == ["-s", "TNG100-3-Dark", "-g", "128", "-m", "4", "99"]
              and have.env["SAMPLE_POINTS"].endswith("hires_plane_z64.bin") and have.env["AUTO_TUNE_REPORT"] == "1"
              and have.env["PS_VOLUME_WEIGHTED"] == "1" and have.env["SCRATCH_DIR"] == str(tmp.parent), str(have.env))
        check("... and only its point count when it does not exist yet",
              need.env.get("DTFE_AUTO_PTS_N") == str(32 * 32) and "SAMPLE_POINTS" not in need.env
              and need.env.get("DTFE_AUTO_PTS_VELGRAD") == "1", str(need.env))
        check("adaptive sizes halve down to the minimum",
              rs.PipelineSpec(nu=8192).adaptive_sizes() == [8192, 4096, 2048, 1024])
        pm.adaptive = True
        lv, msg = levels(pm)
        check("adaptive without a check is an error", "error" in lv and "Check memory" in msg, msg)
        pm.plan, pm.plan_key = {"TNG100-3-Dark": {"004": 64, "099": 32}}, pm.memory_key()
        got = [(st.label, st.env.get("NU"), st.env.get("SNAPS")) for st in pm.steps()]
        check("a checked plan runs the pipeline once per plane size, on its own snapshots, rendering its plane",
              [(g[1], g[2]) for g in got if g[1]] == [("64", "4"), ("32", "99")]
              and [st.argv[st.argv.index("--plane") + 1].endswith(f"hires_plane_z{nu}.json")
                   for st, nu in zip(pm.steps()[1::2], (64, 32))] == [True, True], str(got))
        pm.grid = 256
        lv, msg = levels(pm)
        check("a plan checked for other settings is refused", "error" in lv and "changed since" in msg, msg)

        print("queue:")
        jobs = [rs.QueuedJob.of("grids", rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99, 4])),
                rs.QueuedJob.of("pipeline", rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=64)),
                rs.QueuedJob.of("plots", rs.PlotSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99],
                                                     sets=["fields", "voidpop"])),
                rs.QueuedJob.of("data", rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[99]))]
        check("each job kind gets a one-line summary",
              [j.title for j in jobs] == ["Grids · TNG100-3-Dark · 004 099 · PS-DTFE 512³",
                                          "Pipeline · TNG100-3-Dark · plane 64² · grid 512³ · + figures",
                                          "Plots · TNG100-3-Dark · 099 · Field slice maps, Void population",
                                          "Data · TNG50-4-Dark · 099 · download + merge"],
              str([j.title for j in jobs]))
        check("a queued job keeps its settings, not its commands (rebuilt exactly)",
              all(rs.QueuedJob.from_dict(j.to_dict()).build().to_dict() == j.spec for j in jobs)
              and all("steps" not in j.to_dict() for j in jobs))
        jobs[0].state, jobs[1].state = "done", "running"
        back = [rs.QueuedJob.from_dict(j.to_dict()) for j in jobs]
        check("restored: a job left running is 'stopped', the first waiting one is next",
              [j.state for j in back] == ["done", "stopped", "waiting", "waiting"] and rs.next_waiting(back) == 2)
        try:
            rs.QueuedJob.from_dict({"kind": "nonsense", "spec": {}})
            check("an unknown job kind is refused", False)
        except ValueError:
            check("an unknown job kind is refused", True)
        check("grids, pipeline and custom-snapshot jobs wait for other DTFE runs",
              set(rs.HEAVY_KINDS) == {"grids", "pipeline", "custom"})
        new_features(tmp, sim)
        two_d(tmp)
        bug_fixes_2026_10_05(tmp, sim)
        cube_cache_2026_10_06()
        type_labels_2026_10_06()
        launcher_fixes()
    finally:
        shutil.rmtree(tmp.parent)
    print("-" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


def _write_log(path: Path, text: str, mtime: float | None = None):
    path.write_text(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def new_features(tmp: Path, sim: Path):
    import datetime as dt
    import json

    import numpy as np
    import grids as G
    import presets as P
    import results as R

    print("custom snapshot (CustomSpec):")
    work = tmp.parent / "own"
    work.mkdir()
    snap = work / "cloud.hdf5"
    import h5py
    with h5py.File(snap, "w") as f:
        f.create_group("Header").attrs["BoxSize"] = 50000.0
        g = f.create_group("PartType1")
        g.create_dataset("Coordinates", data=np.zeros((8, 3)))
        g.create_dataset("Potential", data=np.zeros(8))
    c = rs.CustomSpec(input_file=str(snap), mpc_unit=1000.0, periodic=True, estimator="ps", grid=64,
                      fields=["density_a"], output_dir=str(work / "out"), output_name="run1", scalar_dataset="Potential")
    steps = c.steps()
    a = steps[-1].argv
    check("custom: the binary itself, units, periodic, fields, scalar, output root",
          Path(a[0]).name == "PS-DTFE" and a[1] == str(snap) and a[2] == str(work / "out" / "run1")
          and a[a.index("--MpcUnit") + 1] == "1000" and "--periodic" in a
          and a[a.index("--field") + 1:a.index("--field") + 3] == ["density_a", "scalar_a"]
          and a[a.index("--scalar-dataset") + 1] == "Potential", str(a))
    check("custom: a missing output folder gets its mkdir step first",
          steps[0].argv[:2] == ["mkdir", "-p"] and len(steps) == 2)
    check("custom: the run step tees its run log beside the outputs",
          steps[-1].tee == str(work / "out" / "run1.runlog")
          and steps[-1].line().endswith(f"2>&1 | tee {rs._q(steps[-1].tee)}"))
    check("custom: the repo binary is shown as ./PS-DTFE (never a PATH lookup)",
          " ./PS-DTFE " in " " + steps[-1].line() or steps[-1].line().startswith("./PS-DTFE"), steps[-1].line())
    lv, msg = levels(c)
    check("custom: a snapshot without InitialCoordinates is refused for PS-DTFE",
          "error" in lv and "InitialCoordinates" in msg, msg)
    c2 = rs.CustomSpec.from_dict({**c.to_dict(), "estimator": "dtfe", "scalar_dataset": ""})
    lv, msg = levels(c2)
    check("custom: ... but fine for standard DTFE (with the fixed field set)",
          "error" not in lv and rs.DTFE_FIELDS[0] in c2.command(), msg)
    for bad, word in (({"input_type": 111}, "Gadget HDF5"), ({"box": [-1, 1, 0, 1, 0, 1]}, ">= 0"),
                      ({"box": [0, 1, 0]}, "6 numbers"), ({"input_file": str(work / 'nope.hdf5')}, "not found"),
                      ({"lagrangian": "separate", "lagrangian_file": ""}, "initial-conditions")):
        lv, msg = levels(rs.CustomSpec.from_dict({**c.to_dict(), **bad}))
        check(f"custom: refused -- {word}", "error" in lv and word in msg, msg)
    side = rs.CustomSpec.from_dict({**c.to_dict(), "box": []}).settings_sidecar()
    check("custom: the settings sidecar gives the box in Mpc (header BoxSize / MpcUnit)",
          side["box_mpc"] == [0.0, 50.0] * 3 and side["box"] == [], str(side["box_mpc"]))
    side = rs.CustomSpec.from_dict({**c.to_dict(), "box": [0, 20000, 0, 20000, 0, 40000]}).settings_sidecar()
    check("custom: ... and from --box (file units) when given", side["box_mpc"] == [0, 20, 0, 20, 0, 40])
    d = rs.demo_spec(work / "demo")
    ds = d.steps()
    check("demo: generate the snapshot, then run PS-DTFE on it (no download)",
          len(ds) == 3 and ds[1].argv[1].endswith("generate_ps_test_data.py") and "--crossed-waves" in ds[1].argv
          and Path(ds[2].argv[0]).name == "PS-DTFE" and "error" not in levels(d)[0], levels(d)[1])
    check("custom: to_dict/from_dict round trip", rs.CustomSpec.from_dict(c.to_dict()) == c)

    # ---- every binary flag the GUI can emit exists in the built binary's own --full_help, and
    # every script knob it sets is one the script reads: a renamed option can never ship silently
    import re as _re
    import presets as P
    ps_help, dt_help = P.flag_help("PS-DTFE"), P.flag_help("DTFE")
    if ps_help and dt_help:
        emitted = set()
        for spec in (rs.CustomSpec(input_file=str(snap), estimator="ps", deposit="exact", gpu=True, vertex_mass=True,
                                   volume_weighted=True, caustics=True, caustic_cusps=True, scalar_dataset="Masses",
                                   partition=2, max_concurrent=2, scratch_dir=str(work), lagrangian="separate",
                                   lagrangian_file=str(snap), box=[0, 1, 0, 1, 0, 1], parallel_triangulation=True,
                                   alpha_shape=2.5, periodic=False),
                     rs.CustomSpec(input_file=str(snap), estimator="dtfe", gpu=True)):
            emitted |= {a for a in spec.command() if a.startswith("--")}
        known = set(ps_help) | set(dt_help)
        unknown = sorted(emitted - known)
        check("every binary flag a custom-snapshot command can emit exists in the binaries' help",
              not unknown, str(unknown))
        script = (ROOT / "scripts" / "run_ps_dtfe.sh").read_text()
        knobs = set(_re.findall(r'^([A-Z_]+)="\$\{\1:-', script, _re.M))
        used = set(rs.RunSpec(sim="TNG50-4", snapshots=[99], slice_plane=str(snap), scratch_dir=str(work),
                              partition=2, max_concurrent=2, parallel_triangulation=True).command()[1]) - {"DTFE_DATA_ROOT"}
        check("every environment knob the Grids job sets is one run_ps_dtfe.sh reads",
              used <= knobs, str(sorted(used - knobs)))
    else:
        check("flags-exist check skipped: a binary is not built", True)

    # ---- GPU advice: where the deposit pays (measured 2026-10-01), one warning per condition
    ps_gpu, dt_gpu = rs.gpu_built("ps"), rs.gpu_built("dtfe")
    adv = rs.gpu_advice
    check("gpu advice: a small PS run with the GPU ticked is told the CPU is faster; a big one without it, the GPU",
          len(adv(True, "sampled", ["density_a"], 260_000, "ps")) == 1
          and len(adv(False, "sampled", ["density_a"], 2_100_000, "ps")) == (1 if ps_gpu else 0)
          and adv(False, "sampled", ["density_a"], 260_000, "ps") == []
          and adv(True, "sampled", ["density_a"], 7_100_000, "ps") == [])
    check("gpu advice: the exact deposit without the GPU warns whatever the particle count (none known too)",
          len(adv(False, "exact", ["density_a"], None, "ps")) == (1 if ps_gpu else 0)
          and adv(True, "exact", ["density_a"], None, "ps") == []
          and adv(True, "sampled", ["density_a"], None, "ps") == [])
    check("gpu advice: standard DTFE with the GPU unticked on a fine grid is told the GPU is 1.5-2.3x faster; not at 256³",
          adv(True, "sampled", rs.DTFE_FIELDS, 7_100_000, "dtfe", grid=512) == []
          and adv(False, "sampled", rs.DTFE_FIELDS, 7_100_000, "dtfe", grid=256) == []
          and len(adv(False, "sampled", rs.DTFE_FIELDS, 7_100_000, "dtfe", grid=1024)) == (1 if dt_gpu else 0))
    rec = rs.recommended_gpu
    check("recommended GPU: exact always (when built), sampled from ~1M particles, standard DTFE when built",
          rec("exact", ["density_a"], 1000) == ps_gpu and rec("sampled", ["density_a"], 32 ** 3) is False
          and rec("sampled", ["density_a"], 19_000_000) == ps_gpu and rec("sampled", ["density_a"], None) == ps_gpu
          and rec("sampled", ["density_a"], 1e9, "dtfe") == dt_gpu)
    ex = rs.RunSpec(sim="TNG50-4", snapshots=[99], deposit="exact", gpu=False, data_root=str(tmp))
    msgs = [m for level, m in ex.problems() if level == "warning" and "exact" in m]
    check("a Grids exact run without the GPU gets ONE warning (the duplicate is gone)", len(msgs) == (1 if ps_gpu else 0), str(msgs))
    dtg = rs.RunSpec(sim="TNG50-4", snapshots=[99], estimator="dtfe", gpu=True, grid=512, data_root=str(tmp))
    cm = rs.CustomSpec(input_file=str(snap), estimator="dtfe", gpu=True, grid=512)
    check("a standard-DTFE 512³ run with the GPU gets no GPU warning (Grids and custom)",
          not any("GPU" in m and level == "warning" and "METAL" not in m for level, m in dtg.problems() + cm.problems()),
          str(dtg.problems()))
    dtg.gpu = cm.gpu = False
    check("... and with the GPU unticked both are told it is faster (when built)",
          (any("faster" in m for _, m in dtg.problems()) and any("faster" in m for _, m in cm.problems())) == dt_gpu,
          str(cm.problems()))
    dex = rs.RunSpec(sim="TNG50-4", snapshots=[99], estimator="dtfe", deposit="exact", gpu=False, grid=256,
                     data_root=str(tmp))
    cex = rs.CustomSpec(input_file=str(snap), estimator="dtfe", deposit="exact", gpu=False, grid=256)
    check("standard DTFE exact: the custom command passes --exact-average (not the PS flag), and without the GPU "
          "both specs say the GPU is ~6.6x faster (when built)",
          "--exact-average" in cex.command() and "--ps-exact-deposit" not in cex.command()
          and "--exact-average" not in rs.CustomSpec(input_file=str(snap), estimator="dtfe").command()
          and (any("6.6x" in m for _, m in dex.problems()) and any("6.6x" in m for _, m in cex.problems())) == dt_gpu,
          str(cex.problems()))
    # ---- precision: the double pair is a setting, picked per job; the knob and binary names
    both_double = rs.binary_built("ps", "double") and rs.binary_built("dtfe", "double")
    rp = rs.RunSpec(sim="TNG50-4", snapshots=[99], precision="double", data_root=str(tmp))
    check("precision: a double Grids job sets DTFE_PRECISION=double for the script, a single one sets nothing",
          rp.command()[1].get("DTFE_PRECISION") == "double"
          and "DTFE_PRECISION" not in rs.RunSpec(sim="TNG50-4", snapshots=[99]).command()[1])
    check("precision: binary names and the memory estimate (twice the bytes)",
          rs.binary_name("ps", "double") == "PS-DTFE-double" and rs.binary_name("dtfe") == "DTFE"
          and abs(rs.grids_gb(256, ["density_a"], "double") / rs.grids_gb(256, ["density_a"]) - 2) < 1e-9
          and rs.RunSpec(grid=256, precision="double").grid_gb() == 2 * rs.RunSpec(grid=256).grid_gb())
    cd = rs.CustomSpec(input_file=str(snap), estimator="dtfe", precision="double")
    check("precision: a double custom job runs the -double binary; missing pair = a clear error naming Setup",
          cd.binary().name == "DTFE-double"
          and (any("double-precision" in m and level == "error" for level, m in cd.problems()) != both_double))
    pp = rs.PipelineSpec(sims=["TNG50-4"], precision="double")
    check("precision: the pipeline passes the knob through", pp.command()[1].get("DTFE_PRECISION") == "double")
    check("precision: an unknown value is an error", any("precision" in m for level, m in
          rs.RunSpec(sim="TNG50-4", snapshots=[99], precision="half", data_root=str(tmp)).problems() if level == "error"))
    check("precision: the GPU and TBB stamps are read from the pair's own object directory",
          rs._objdir("ps", "double").name == "o_ps_d" and rs._objdir("dtfe").name == "o"
          and isinstance(rs.gpu_built("ps", "double"), bool) and isinstance(rs.tbb_built("double"), bool))

    # ---- the Custom tab's memory check: a report-mode call keyed on what changes memory
    c1 = rs.CustomSpec(input_file=str(snap), grid=256, output_dir=str(work), output_name="a")
    c2 = rs.CustomSpec(input_file=str(snap), grid=256, output_dir=str(work / "elsewhere"), output_name="b")
    c3 = rs.CustomSpec(input_file=str(snap), grid=512, output_dir=str(work), output_name="a")
    check("custom memory key: the output folder and name do not change it, the grid does",
          c1.memory_key() == c2.memory_key() and c1.memory_key() != c3.memory_key())
    st = c1.check_step()
    check("custom check step: the run's own command in report mode, named after the file",
          st.argv[:-1] == c1.command() and st.argv[-1] == "--auto-tune-report" and "cloud.hdf5" in st.label, st.label)
    q = rs.QueuedJob.of("custom", c)
    check("custom: a queued custom-snapshot job rebuilds and has a summary",
          q.build() == c and q.title.startswith("Custom snapshot · cloud.hdf5 · PS-DTFE 64³"), q.title)

    print("export:")
    txt = rs.export_script(steps, "a test job")
    check("bash export: shebang, cd to the repo, every step, failure count as exit status",
          txt.startswith("#!/usr/bin/env bash") and 'cd "$REPO"' in txt and steps[-1].line() in txt
          and 'exit "$failed"' in txt and "#SBATCH" not in txt)
    sl = rs.export_script(steps, "a test job", {"job_name": "j1", "time": "02:00:00", "cpus": 8, "mem": "32G",
                                                "partition": "regular", "mail": ""})
    check("SLURM export: the #SBATCH header, OMP threads from SLURM, no empty options",
          "#SBATCH --job-name=j1" in sl and "#SBATCH --time=02:00:00" in sl and "#SBATCH --cpus-per-task=8" in sl
          and "#SBATCH --mem=32G" in sl and "#SBATCH --partition=regular" in sl and "mail-user" not in sl
          and "OMP_NUM_THREADS" in sl)
    ex = work / "job.sh"
    ex.write_text(rs.export_script([rs.Step("echo", ["echo", "hello"], {}), rs.Step("fail", ["false"], {})], "t"))
    import subprocess
    r = subprocess.run(["bash", str(ex)], capture_output=True, text=True)
    check("an exported script runs, goes on after a failed step and exits with the failure count",
          r.returncode == 1 and "hello" in r.stdout, f"{r.returncode} {r.stdout} {r.stderr}")

    print("run logs (results.py):")
    snapdir = sim / "snapdir_099"
    for suffix in ("a_den", "a_streams", "a_vel"):
        (snapdir / f"ps_output.{suffix}").write_bytes(b"\0" * 4 * 8)
    log = snapdir / "ps_output.runlog"
    head = ("Build: PS-DTFE 308f08a-dirty (built 2026-09-01T10:00:00Z)\n"
            "RUNNING: ./PS-DTFE /x/y z/combined_099.hdf5 /x/ps_output --grid 512 --periodic --partition 5 5 5 "
            "--max-concurrent 3 --avg-subsamples 3 --input 105 --MpcUnit 1000 --field density_a velocity_a "
            "--ps-gpu --ps-volume-weighted\n"
            "   peak memory (RSS) : 47.24 GB\n   total wall time   : 1h 2m 3s\n"
            "         97186996544  peak memory footprint\n")
    _write_log(log, head)
    recs = R.scan_runs(tmp)
    r0 = recs[0] if recs else None
    check("scan_runs finds the per-family run log", len(recs) == 1 and r0.sim == "TNG100-3-Dark" and r0.snap == 99)
    check("parsed: build stamp, grid, partitions, GPU, wall time, RSS, footprint, fields, outputs",
          r0.binary == "PS-DTFE" and r0.build_rev == "308f08a-dirty" and r0.build_time is not None
          and r0.grid == 512 and r0.partitions == 125 and r0.partition == 5 and r0.max_concurrent == 3
          and r0.gpu and r0.wall_seconds == 3723 and r0.peak_rss_gb == 47.24 and abs(r0.footprint_gb - 97.19) < 0.01
          and r0.fields == ["density_a", "velocity_a"] and r0.ok
          and {p.name for p in r0.outputs} >= {"ps_output.a_den", "ps_output.a_streams", "ps_output.a_vel"}
          and not any(p.suffix == ".runlog" for p in r0.outputs)
          and r0.redshift == 0.0, f"{r0}")
    reasons = R.stale_reasons(r0)
    check("stale: a GPU + partitions build before the 2026-09-29 fix is WRONG",
          any(lv == "error" and "partition" in m for lv, m in reasons), str(reasons))
    check("stale: '.streams' without '.hidden_streams' predates the stream-count fix",
          any(lv == "warning" and "centroid-fallback" in m for lv, m in reasons))
    check("stale: a TNG PS run without --ps-vertex-mass is flagged", any("tetrahedron masses" in m for _, m in reasons))
    (snapdir / "ps_output.a_unresolved").write_bytes(b"\0" * 32)       # the old name of '.hidden_streams'
    _write_log(log, head.replace("2026-09-01", "2026-09-30").replace("--ps-gpu", "--ps-gpu --ps-vertex-mass"))
    reasons = R.stale_reasons(R.scan_runs(tmp)[0])            # built 2026-09-30 10:00, before the exact flag
    check("stale: an old '.unresolved' (before the exact hidden-streams flag) is flagged, and nothing else",
          len(reasons) == 1 and reasons[0][0] == "warning" and "hidden-streams flag" in reasons[0][1], str(reasons))
    (snapdir / "ps_output.a_unresolved").unlink()                    # a current build writes the new name
    (snapdir / "ps_output.a_hidden_streams").write_bytes(b"\0" * 32)
    _write_log(log, head.replace("2026-09-01T10:00", "2026-09-30T18:00").replace("--ps-gpu", "--ps-gpu --ps-vertex-mass"))
    r1 = R.scan_runs(tmp)[0]
    check("stale: the same run built after every fix is clean", R.stale_reasons(r1) == [], str(R.stale_reasons(r1)))
    np_head = head.replace("2026-09-01T10:00", "2026-09-30T18:00").replace("--periodic ", "--box 20 40 20 40 20 40 ") \
                  .replace("--ps-gpu", "--ps-gpu --ps-vertex-mass")
    _write_log(log, np_head)
    reasons = R.stale_reasons(R.scan_runs(tmp)[0])
    check("stale: a non-periodic --box run built before the region-face and alpha-shape fixes is flagged by both",
          any("crossing the region" in m for _, m in reasons)
          and any("alpha shape" in m for _, m in reasons) and len(reasons) == 2, str(reasons))
    _write_log(log, np_head.replace("2026-09-30T18:00", "2026-09-30T23:00"))
    r3 = R.scan_runs(tmp)[0]
    check("stale: the same non-periodic run built after both fixes is clean", R.stale_reasons(r3) == [], str(R.stale_reasons(r3)))
    dd_head = ("Build: PS-DTFE-double 3e0c1a6-dirty (built 2026-10-01T18:00:00Z)\n"
               "RUNNING: ./PS-DTFE-double /x/y z/combined_099.hdf5 /x/ps_output --grid 256 --periodic --input 105 "
               "--MpcUnit 1000 --field density_a --ps-gpu --ps-vertex-mass\n   peak memory (RSS) : 9.0 GB\n   total wall time   : 1m 2s\n")
    _write_log(log, dd_head)
    r5 = R.scan_runs(tmp)[0]
    check("run logs: the double pair is recognised (estimator, precision, settings, options text)",
          r5.estimator == "ps" and r5.precision == "double" and r5.settings()["precision"] == "double"
          and "double" in r5.options_text() and R.stale_reasons(r5) == [], str(R.stale_reasons(r5)))
    dt_head = ("Build: DTFE 3e0c1a6-dirty (built 2026-10-01T09:00:00Z)\n"
               "RUNNING: ./DTFE /x/y z/combined_099.hdf5 /x/ps_output --grid 512 --periodic --input 105 --MpcUnit 1000 "
               "--field density_a velocity_a --gpu\n"
               "DTFE Metal: Metal command buffer failed: Impacting Interactivity -- retrying the deposit from scratch "
               "(attempt 2/3, target 62 ms buffers, 24 ms gaps)\n"
               "   peak memory (RSS) : 7.24 GB\n   total wall time   : 6m 37s\n")
    _write_log(log, dt_head)
    r4 = R.scan_runs(tmp)[0]
    reasons = R.stale_reasons(r4)
    check("stale: a standard-DTFE GPU run whose deposit was restarted by the GPU watchdog is flagged (built before the fix)",
          r4.estimator == "dtfe" and r4.gpu and r4.gpu_retried and len(reasons) == 1
          and "watchdog" in reasons[0][1] and reasons[0][0] == "warning", str(reasons))
    _write_log(log, dt_head.replace("retrying the deposit from scratch", "nothing of the kind"))
    check("stale: ... the same run without a restart is clean", R.stale_reasons(R.scan_runs(tmp)[0]) == [])
    _write_log(log, dt_head.replace("2026-10-01T09:00", "2026-10-01T23:00"))
    check("stale: ... and a restart with the fixed build is clean (its retry drains the queue first)",
          R.stale_reasons(R.scan_runs(tmp)[0]) == [], str(R.stale_reasons(R.scan_runs(tmp)[0])))
    _write_log(log, head.split("\n", 1)[1], mtime=dt.datetime(2026, 8, 1).timestamp())
    r2 = R.scan_runs(tmp)[0]
    reasons = R.stale_reasons(r2)
    check("stale: without a build stamp the run date decides, and says so",
          r2.build_time is None and any("run date" in m for _, m in reasons), str(reasons))
    check("settings(): the Grids-tab settings of the run",
          r1.settings()["grid"] == 512 and r1.settings()["vertex_mass"] and r1.settings()["fields"] == ["density_a", "velocity_a"])
    est = R.time_estimate([r0, r1, r2], sim="TNG100-3-Dark", grid=512, estimator="ps", gpu=True, sliced=False)
    check("time estimate: the median of comparable earlier runs", est == (3723, 3), str(est))
    check("time estimate: none for another grid",
          R.time_estimate([r0], sim="TNG100-3-Dark", grid=256, estimator="ps", gpu=True, sliced=False) is None)
    _write_log(log, "RUNNING: ./PS-DTFE a b --grid 64\n~~~ ERROR ~~~ something\n")
    check("a log with an ERROR block is not ok", not R.scan_runs(tmp)[0].ok)
    for text, codes, key in (("... std::bad_alloc\n", [1], "Out of memory"), ("x", [137], "Out of memory"),
                             ("unrecognised option '--foo'", [1], "older"), ("points into iCloud Drive", [1], "iCloud"),
                             ("no snapshots under /Volumes/T7", [1], "Data not found"), ("boom", [2], "failed")):
        found = [d.title for d in R.diagnose(text, codes)]
        check(f"diagnose: {key}", any(key in t for t in found), str(found))
    check("diagnose: a clean log with exit 0 has no advice", R.diagnose("all fine", [0]) == [])
    n = R.notify_command('a "quoted" title', "text")
    check("notify_command: a runnable command (or None where there is none)", n is None or isinstance(n, list))

    print("presets and help (presets.py):")
    helptext = ("Main options:\n"
                "  -g [ --grid ] arg      specify grid size\n"
                "                         along each direction.\n"
                "  --ps-vertex-mass       PS-DTFE only: assign each\n"
                "                         tetrahedron its mass.\n"
                "  --a-really-long-option-name-that-pushes arg\n"
                "                         the text goes on the next line\n"
                "\nField choices:\n")
    h = P.parse_help(helptext)
    check("parse_help: short+long, wrapped continuation, long names with the text below",
          h.get("--grid") == "specify grid size along each direction."
          and h.get("--ps-vertex-mass") == "PS-DTFE only: assign each tetrahedron its mass."
          and h.get("--a-really-long-option-name-that-pushes") == "the text goes on the next line", str(h))
    spec = rs.RunSpec(grid=512, fields=list(rs.PS_DEFAULT_FIELDS))
    planes = [sim / "hires_plane_z64.bin"]
    changed = P.apply_preset(spec, P.BUILTIN_PRESETS["Quick look"][1], planes)
    check("preset 'Quick look': 128^3, density+velocity, no slice",
          spec.grid == 128 and spec.fields == ["density_a", "velocity_a"] and spec.slice_plane == ""
          and "grid" in changed)
    P.apply_preset(spec, P.BUILTIN_PRESETS["Publication"][1], planes)
    check("preset 'Publication': the largest image plane and caustics",
          spec.slice_plane == str(planes[0]) and spec.caustics and spec.grid == 512)
    cap = P.capture(spec)
    check("capture(): only preset-able settings", "grid" in cap and "sim" not in cap and "snapshots" not in cap)
    cc = rs.CustomSpec()
    P.apply_preset(cc, P.BUILTIN_PRESETS["Publication"][1], planes)
    check("a preset applies to a custom-snapshot job too (fields it lacks are skipped)", cc.grid == 512 and cc.caustics)
    real = P.flag_help("PS-DTFE")
    if real:
        check("the binary's own --full_help: every production flag has a description",
              all(real.get(f) for f in ("--ps-vertex-mass", "--ps-volume-weighted", "--ps-caustics", "--grid",
                                        "--serve", "--pts-lagrangian", "--pts-scalar")), str({k: bool(v) for k, v in real.items()}))
    else:
        print("   SKIP  flag_help (PS-DTFE not built)")

    texts = ([fx.message for fx in R.KNOWN_FIXES]
             + [d.title + " " + d.advice for d in R.diagnose(
                 "std::bad_alloc unrecognised option points into iCloud Operation not permitted "
                 "no snapshots under /Volumes/x No 'InitialCoordinates' OVER BUDGET", [1, 137])]
             + [t for t, _ in P.BUILTIN_PRESETS.values()] + [lab for lab, _ in P.OPTION_TEXT.values()]
             + [P.tooltip(k) for k in P.OPTION_TEXT]
             + [fs.title + " " + fs.about + " " + " ".join(o.label + " " + o.help for o in fs.opts)
                for fs in rs.FIGURE_SETS])
    check("no '--' in the stale-output rules, failure advice, presets, option tooltips or figure sets",
          not [t for t in texts if "--" in t], str([t for t in texts if "--" in t][:3]))
    check("plain_help: option names lose their dashes, ' -- ' and 'a--b' become real dashes",
          P.plain_help("use '--partition' -- or --serve; the div--density slope")
          == "use 'partition' — or serve; the div–density slope")

    print("Explore data (grids.py):")
    n = 8
    rng = np.random.default_rng(0)
    den = rng.random((n, n, n)).astype(np.float32) + 0.5
    vel = rng.random((n, n, n, 3)).astype(np.float32)
    (snapdir / "ps_output.a_den").write_bytes(den.tobytes())
    (snapdir / "ps_output.a_vel").write_bytes(vel.tobytes())
    for suffix in ("a_streams", "a_hidden_streams"):
        (snapdir / f"ps_output.{suffix}").write_bytes(b"\0" * 4 * n ** 3)
    with h5py.File(snapdir / "combined_099.hdf5", "a") as f:
        f["Header"].attrs["BoxSize"] = 80000.0
    outs = G.find_outputs(tmp)
    o = outs[0][1] if outs else None
    check("find_outputs: the TNG-layout run, its box (h-free kpc -> Mpc) and redshift",
          o is not None and o.n == n and o.length[0] == 80.0 and o.redshift == 0.0
          and o.fields()[:2] == ["density", "streams"], str(outs))
    check("find_outputs: a TNG-layout output knows its simulation and snapshot number; natural simulation order",
          o is not None and o.sim == "TNG100-3-Dark" and o.snap == 99
          and sorted(["TNG300-3-Dark", "TNG50-3-Dark", "TNG100-3-Dark"], key=G.sim_sort_key)
          == ["TNG50-3-Dark", "TNG100-3-Dark", "TNG300-3-Dark"], f"{o and (o.sim, o.snap)}")
    check("slice: plane = cube[..., k] for the z normal", np.allclose(o.slice("density", 2, 3), den[:, :, 3]))
    check("slice: x normal, y normal", np.allclose(o.slice("density", 0, 1), den[1])
          and np.allclose(o.slice("density", 1, 5), den[:, 5, :]))
    check("slice: |v| and one component",
          np.allclose(o.slice("velocity", 2, 0, "norm"), np.linalg.norm(vel[:, :, 0], axis=-1), rtol=1e-6)
          and np.allclose(o.slice("velocity", 2, 0, "y"), vel[:, :, 0, 1]))
    check("value(): one cell", abs(o.value("density", (2, 3, 4)) - float(den[2, 3, 4])) < 1e-6)
    check("cell_center / plane_point", np.allclose(o.cell_center((0, 0, 7)), [5, 5, 75])
          and list(o.plane_point(2, 7, 1, 2)) == [1, 2, 7])
    ar = o.arrow_field("velocity", 2, 0, per_side=4)
    bu = vel[:, :, 0, 0].reshape(4, 2, 4, 2).mean(axis=(1, 3))        # 8 cells, 4 arrows a side
    check("arrow_field: in-plane components block-averaged, longest arrow = 1",
          ar is not None and ar["spacing"] == 2.0 and len(ar["x"]) == 16
          and np.allclose(ar["u"], bu, rtol=1e-6) and abs(np.hypot(ar["dx"], ar["dy"]).max() - 1) < 1e-9
          and o.arrow_field("density", 2, 0) is None, str(ar and ar["spacing"]))
    vd = work / "vec"                                   # a pure +y flow, to pin the orientation
    vd.mkdir(exist_ok=True)
    (vd / "flow.a_den").write_bytes(den.tobytes())
    flow = np.zeros_like(vel)
    flow[..., 1] = 2.0
    (vd / "flow.a_vel").write_bytes(flow.tobytes())
    ov = G.OutputSet(vd / "flow")
    up, right = ov.arrow_field("velocity", 2, 3, per_side=8), ov.arrow_field("velocity", 0, 3, per_side=8)
    check("arrow_field: +y points UP in a z slice (the image is plane.T[::-1]) and RIGHT in an x slice",
          np.allclose(up["dx"], 0) and np.allclose(up["dy"], -1) and np.allclose(right["dx"], 1)
          and np.allclose(right["dy"], 0) and right["axes"] == (1, 2))
    check("arrow_field: arrows sit on the block centres, image y measured from the top",
          np.allclose(sorted(set(up["x"])), np.arange(8) + 0.5) and np.allclose(sorted(set(up["y"])), np.arange(8) + 0.5))
    rgb, vr = G.colorize(np.array([[1.0, 10.0], [100.0, 1000.0]]), "viridis", log=True, clip=(0, 100))
    check("colorize: log10 range, RGB bytes", rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8 and vr == (0.0, 3.0))
    _, vr = G.colorize(np.array([[-1.0, 3.0]]), "RdBu_r", clip=(0, 100), symmetric=True)
    check("colorize: a symmetric range for diverging maps", vr == (-3.0, 3.0))
    kw = G.server_settings(o)
    check("server_settings (TNG layout): the snapdir's combined file, periodic, PS",
          kw and kw["snapshot"].endswith("combined_099.hdf5") and kw["phase_space"] and kw["periodic"], str(kw))
    od = work / "out"
    od.mkdir(exist_ok=True)
    (od / "run1.a_den").write_bytes(den.tobytes())
    (od / "run1.gui.json").write_text(json.dumps(c.settings_sidecar()))
    outs = G.find_outputs(tmp, [od])
    mine = [x for t, x in outs if t.startswith("custom ·")]
    check("find_outputs: a custom-snapshot run from its folder, box from its sidecar; no simulation or snapshot",
          len(mine) == 1 and mine[0].length[0] == 50.0 and mine[0].sim == "" and mine[0].snap is None, str(outs))
    kw = G.server_settings(mine[0])
    check("server_settings (custom snapshot): the job's own input, units and estimator",
          kw and kw["snapshot"] == str(snap) and kw["mpc_unit"] == 1000.0 and kw["phase_space"], str(kw))
    for name, npart in (("big.hdf5", 100_000_000), ("small.hdf5", 32_768)):
        with h5py.File(work / name, "w") as f:
            f.create_group("Header").attrs["NumPart_Total"] = np.array([0, npart, 0, 0, 0, 0], dtype=np.uint32)
    big = G.server_cost(work / "big.hdf5", ram_bytes=64 * 2 ** 30)
    small = G.server_cost(work / "small.hdf5", ram_bytes=64 * 2 ** 30)
    check("server_cost: a TNG100-sized box on 64 GB is split, with its cache size; a small one is not",
          big["split"] and 40 < big["cache_gb"] < 80 and big["single_gb"] > big["budget_gb"]
          and not small["split"], f"{big} {small}")
    check("server_cost: None for a file it cannot read", G.server_cost(work / "nope.hdf5") is None)
    (od / "broken.a_den").write_bytes(b"\0" * 7)
    check("an unreadable grid is skipped, not an error", not G.OutputSet(od / "broken"))



def two_d(tmp: Path):
    """The 2D programs (make DIM=2) in the launcher: names, the custom-snapshot job, the grid reader."""
    import json
    import subprocess

    import numpy as np
    import grids as G
    import results as R

    print("2D (the -2d programs):")
    check("binary and object-directory names: -2d before -double, _2d before _d",
          rs.binary_name("ps", "double", 2) == "PS-DTFE-2d-double" and rs.binary_name("dtfe", dim=2) == "DTFE-2d"
          and rs._objdir("ps", "double", 2).name == "o_ps_2d_d" and rs._objdir("dtfe", dim=2).name == "o_2d"
          and rs.build_hint("single", 2) == "make DTFE PS-DTFE DIM=2")
    check("a 2D run has no GPU and no parallel insertion (the kernels and CGAL's parallel build are 3D)",
          not rs.gpu_built("ps", dim=2) and not rs.gpu_built("dtfe", dim=2) and not rs.tbb_built(dim=2)
          and rs.gpu_backend("ps", dim=2) == "" and not rs.recommended_gpu("exact", ["density_a"], 10**8, dim=2)
          and rs.gpu_advice(False, "exact", ["density_a"], 10**8, "ps", dim=2) == [])

    work = tmp.parent / "two_d"
    work.mkdir()
    import h5py
    for name, cols in (("plane.hdf5", 2), ("cube.hdf5", 3)):
        with h5py.File(work / name, "w") as f:
            f.create_group("Header").attrs["BoxSize"] = 100.0
            f.create_group("PartType1").create_dataset("Coordinates", data=np.zeros((16, cols)))
    (work / "plane.txt").write_text("2\n0 100 0 100\n1 2 1\n3 4 1\n")
    (work / "cube.txt").write_text("1\n0 1 0 1 0 1\n0.5 0.5 0.5 1\n")
    check("snapshot_dim: HDF5 coordinate columns, the text box line, unknown for Gadget binary",
          rs.snapshot_dim(work / "plane.hdf5") == 2 and rs.snapshot_dim(work / "cube.hdf5") == 3
          and rs.snapshot_dim(work / "plane.txt", 111) == 2 and rs.snapshot_dim(work / "cube.txt", 112) == 3
          and rs.snapshot_dim(work / "plane.hdf5", 101) is None and rs.snapshot_dim(work / "nope.hdf5") is None)

    c = rs.CustomSpec(input_file=str(work / "plane.hdf5"), dim=2, estimator="ps", grid=64, gpu=True,
                      parallel_triangulation=True, fields=["density_a", "velocity_a", "tweb_a"],
                      box=[0, 100, 0, 100], partition=2, output_dir=str(work / "out"), output_name="p")
    a = c.command()
    check("custom 2D: the -2d binary, a 4-number box, a 2-number split; no GPU, no parallel insertion; the T-web kept",
          Path(a[0]).name == "PS-DTFE-2d" and a[a.index("--box") + 1:a.index("--box") + 6] == ["0", "100", "0", "100", "--field"]
          and a[a.index("--partition") + 1:] == ["2", "2"] and "--ps-gpu" not in a
          and "--parallel-triangulation" not in a and "tweb_a" in a and "velocity_a" in a, str(a))
    lv = {m: lvl for lvl, m in c.problems()}
    check("custom 2D: says the GPU and the parallel insertion do not apply; nothing about the T-web (2D has it)",
          any("CPU" in m for m in lv) and any("parallel" in m for m in lv)
          and not any("T-web" in m for m in lv), str(lv))
    d = rs.CustomSpec(input_file=str(work / "plane.hdf5"), dim=2, estimator="dtfe", gpu=True,
                      output_dir=str(work / "out"))
    check("custom 2D standard DTFE: DTFE-2d, no GPU flag", Path(d.command()[0]).name == "DTFE-2d" and "--gpu" not in d.command())
    bad = rs.CustomSpec(input_file=str(work / "plane.hdf5"), dim=2, box=[0, 1, 0, 1, 0, 1], output_dir=str(work / "out"))
    wrong = rs.CustomSpec(input_file=str(work / "plane.hdf5"), dim=3, output_dir=str(work / "out"))
    wrong3 = rs.CustomSpec(input_file=str(work / "cube.hdf5"), dim=2, output_dir=str(work / "out"))
    check("custom 2D: a 6-number box is refused (it needs 4)",
          any(lvl == "error" and "4 numbers" in m for lvl, m in bad.problems()), levels(bad)[1])
    check("custom: the dimension must match the file (a 2D file run as 3D, and the reverse)",
          any(lvl == "error" and "is 2D" in m for lvl, m in wrong.problems())
          and any(lvl == "error" and "is 3D" in m for lvl, m in wrong3.problems()))
    side = c.settings_sidecar()
    check("custom 2D: the sidecar records the dimension and a 4-number box in Mpc",
          side["dim"] == 2 and side["box_mpc"] == [0.0, 100.0, 0.0, 100.0], str(side))
    demo = rs.CustomSpec(input_file=str(work / "demo" / "d.hdf5"), dim=2, demo_n=32, output_dir=str(work / "out"),
                         demo_file=str(work / "demo" / "d.hdf5"))      # generated only at the demo's own path (B5)
    gen = [st.argv for st in demo.steps() if "generate" in st.label]
    check("custom 2D demo: the generator writes a 2D snapshot", gen and gen[0][-2:] == ["--dim", "2"], str(gen))
    texts = [rs.DIM_TEXT[2]] + [m for _, m in c.problems() + bad.problems() + wrong.problems()]
    check("no '--' in the 2D texts", not [t for t in texts if "--" in t], str([t for t in texts if "--" in t]))

    # the grid reader on a synthetic 2D output (12^2 cells; no sidecar: the files tell)
    n = 12
    rng = np.random.default_rng(3)
    od = work / "grids"
    od.mkdir()
    den = (rng.random((n, n)) + 0.5).astype(np.float32)
    vel = rng.random((n, n, 2)).astype(np.float32)
    grad = rng.random((n, n, 4)).astype(np.float32)
    vort = rng.random((n, n)).astype(np.float32)
    disp = rng.random((n, n, 3)).astype(np.float32)
    for sfx, arr in (("a_den", den), ("a_vel", vel), ("a_velGrad", grad), ("a_velVort", vort),
                     ("a_velDispTensor", disp), ("a_streams", np.ones((n, n), np.float32))):
        (od / f"r2.{sfx}").write_bytes(arr.tobytes())
    o = G.OutputSet(od / "r2")
    check("OutputSet: a 2D output is told from its files (a 2-component velocity), every field kept",
          o.dim == 2 and o.n == n and {k: f.ncomp for k, f in o.files.items()}
          == {"density": 1, "velocity": 2, "gradient": 4, "vorticity": 1, "dispersion_tensor": 3, "streams": 1},
          f"{o.dim} {o.n} {[(k, f.ncomp) for k, f in o.files.items()]}")
    check("OutputSet 2D: the slice is the whole grid whatever the axis and index",
          np.allclose(o.slice("density", 0, 5), den) and np.allclose(o.slice("density"), den))
    check("OutputSet 2D: component menus and values (velocity x/y, gradient trace = xx + yy, tensor xy)",
          G.components("velocity", 2, 2) == ["norm", "x", "y"]
          and G.components("gradient", 4, 2) == ["trace (divergence)", "xx", "xy", "yx", "yy"]
          and np.allclose(o.slice("velocity", component="y"), vel[..., 1])
          and np.allclose(o.slice("gradient", component="trace (divergence)"), grad[..., 0] + grad[..., 3])
          and np.allclose(o.slice("gradient", component="yx"), grad[..., 2])
          and np.allclose(o.slice("dispersion_tensor", component="xy"), disp[..., 1]))
    check("OutputSet 2D: the vorticity map is the curl's z (the grid stores -w_z / 2)",
          G.components("vorticity", 1, 2) == ["value"] and np.allclose(o.slice("vorticity"), -2.0 * vort))
    ar = o.arrow_field("velocity", per_side=6)
    check("OutputSet 2D: velocity arrows from the two components; value and cell of (i, j)",
          ar is not None and ar["u"].shape == (6, 6) and ar["axes"] == (0, 1)
          and abs(o.value("density", (3, 4)) - float(den[3, 4])) < 1e-6 and list(o.plane_point(2, 0, 3, 4)) == [3, 4])
    o.box = ((0.0, 0.0), (60.0, 30.0))
    check("OutputSet 2D: a two-corner box; lo / length keep a third, zero-thickness entry",
          list(o.lo) == [0.0, 0.0, 0.0] and list(o.length) == [60.0, 30.0, 0.0])
    # scalar-only outputs: 32^2 float32 is also 8^3 float64, 256^2 float32 also 32^3 float64; 64^2 float32
    # is also 16^3 float32. A phase-space run's stream counts tell the precision; else 3D, as before
    (od / "s32.a_den").write_bytes(np.ones((32, 32), np.float32).tobytes())
    (od / "s32.a_streams").write_bytes(np.ones((32, 32), np.float32).tobytes())
    (od / "c32.a_den").write_bytes(np.ones((32, 32, 32), np.float64).tobytes())
    (od / "c32.a_streams").write_bytes(np.ones((32, 32, 32), np.float64).tobytes())
    (od / "d32.a_den").write_bytes(np.ones((32, 32, 32), np.float64).tobytes())
    (od / "s64.a_den").write_bytes(np.ones((64, 64), np.float32).tobytes())
    (od / "s64.gui.json").write_text(json.dumps({"dim": 2}))
    s32, c32, d32 = G.OutputSet(od / "s32"), G.OutputSet(od / "c32"), G.OutputSet(od / "d32")
    check("OutputSet: same bytes in two precisions -- the stream counts pick the reading (a 32^2 float32 plane, "
          "a 32^3 float64 cube); without them 3D, as before; a sidecar settles any tie",
          (s32.dim, s32.n) == (2, 32) and (c32.dim, c32.n, c32.dtype) == (3, 32, np.float64)
          and (d32.dim, d32.n) == (3, 32) and (G.OutputSet(od / "s64").dim, G.OutputSet(od / "s64").n) == (2, 64),
          f"{(s32.dim, s32.n)} {(c32.dim, c32.n, c32.dtype)} {(d32.dim, d32.n)}")
    # one precision, same bytes (512^2 == 64^3): neighbouring cells decide -- the reading whose neighbour
    # stride holds clearly the smaller differences; a field without structure stays 3D
    def smooth(shape, ell):
        k2 = sum(a * a for a in np.meshgrid(*[np.fft.fftfreq(m) for m in shape], indexing="ij"))
        f = np.fft.ifftn(np.fft.fftn(rng.standard_normal(shape)) * np.exp(-0.5 * k2 * (2 * np.pi * ell) ** 2)).real
        return (1 + 0.6 * f / f.std()).clip(0.01).astype(np.float32)
    smooth((512, 512), 4).tofile(od / "p512.a_den")
    smooth((64, 64, 64), 2).tofile(od / "c64.a_den")
    np.ones((512, 512), np.float32).tofile(od / "f512.a_den")
    p512, c64, f512 = G.OutputSet(od / "p512"), G.OutputSet(od / "c64"), G.OutputSet(od / "f512")
    check("OutputSet: a smooth 512^2 plane and a smooth 64^3 cube of the same bytes are told apart by their "
          "neighbours; a featureless one stays 3D", (p512.dim, p512.n) == (2, 512) and (c64.dim, c64.n) == (3, 64)
          and f512.dim == 3, f"{(p512.dim, p512.n)} {(c64.dim, c64.n)} {f512.dim}")
    check("custom_box / server_settings: a 2D sidecar gives two-entry corners, the 2D server and a 4-number box",
          G.custom_box({"dim": 2, "box_mpc": [0, 50, 0, 40]}) == ((0, 0), (50, 40)))
    (od / "c2.a_den").write_bytes(den.tobytes())
    (od / "c2.a_vel").write_bytes(vel.tobytes())
    (od / "c2.gui.json").write_text(json.dumps(dict(side, input_file=str(work / "plane.hdf5"))))
    kw = G.server_settings(G.OutputSet(od / "c2"))
    check("server_settings 2D: dim 2 and the run's 4-number box", kw and kw.get("dim") == 2
          and kw["options"][:1] == ["--box"] and len(kw["options"]) == 5, str(kw))
    # a 3D vorticity grid holds [W_xy, W_xz, W_yz] = [-w_z, w_y, -w_x] / 2
    od3 = work / "v3"
    od3.mkdir()
    w3 = rng.random((4, 4, 4, 3)).astype(np.float32)
    (od3 / "v.a_den").write_bytes(np.ones((4, 4, 4), np.float32).tobytes())
    (od3 / "v.a_velVort").write_bytes(w3.tobytes())
    o3 = G.OutputSet(od3 / "v")
    check("OutputSet 3D: the vorticity map is the curl (x, y, z) = 2 (-W_yz, W_xz, -W_xy)",
          np.allclose(o3.slice("vorticity", 2, 1, "x"), -2 * w3[:, :, 1, 2])
          and np.allclose(o3.slice("vorticity", 2, 1, "y"), 2 * w3[:, :, 1, 1])
          and np.allclose(o3.slice("vorticity", 2, 1, "z"), -2 * w3[:, :, 1, 0]))

    # the binaries themselves: a rigid rotation (curl = 2 Omega along z) through DTFE-2d and DTFE
    rot_bins = [(dim, rs.REPO_ROOT / ("DTFE-2d" if dim == 2 else "DTFE")) for dim in (2, 3)]
    if all(b.is_file() for _, b in rot_bins):
        sys.path.insert(0, str(ROOT / "tests"))
        from generate_ps_test_data import write_snapshot
        got = {}
        for dim, b in rot_bins:
            m = 24 if dim == 2 else 12
            g = (np.arange(m) + 0.5) * 100.0 / m
            pos = np.stack([x.ravel() for x in np.meshgrid(*([g] * dim), indexing="ij")], 1)
            pos = pos + np.random.default_rng(1).uniform(-0.3, 0.3, pos.shape)
            v = np.zeros_like(pos)
            v[:, 0], v[:, 1] = -2.0 * (pos[:, 1] - 50), 2.0 * (pos[:, 0] - 50)
            if dim == 3:
                v[:, 2] = 0.3 * (pos[:, 0] - 50)                   # dv_z/dx: curl_y = -0.3
            write_snapshot(str(work / f"rot{dim}.hdf5"), pos, pos, v, 100.0, 1.0)
            subprocess.run([str(b), str(work / f"rot{dim}.hdf5"), str(work / f"rot{dim}"), "--grid", "8",
                            "--input", "105", "--MpcUnit", "1", "--box", *(["0", "100"] * dim),
                            "--field", "vorticity", "gradient", "--verbose", "1"], check=True, capture_output=True)
            ro = G.OutputSet(work / f"rot{dim}", dim=dim)
            got[dim] = ([ro.slice("vorticity")[4, 4]] if dim == 2 else
                        [ro.slice("vorticity", 2, 4, k)[4, 4] for k in "xyz"]) + \
                       [ro.slice("gradient", 2, 4, "xy")[4, 4]]
        check("the binaries' grids on a rigid rotation: the map shows the curl (2D: 4; 3D: 0, -0.3, 4) and "
              "gradient xy = dv_x/dy = -2", np.allclose(got[2], [4.0, -2.0], atol=1e-3)
              and np.allclose(got[3], [0.0, -0.3, 4.0, -2.0], atol=1e-3), str(got))
        # real outputs at the 64^2 == 16^3 tie, no sidecar: density, stream counts, all scalar
        dims = {}
        for dim, b, n_, grid in ((2, "PS-DTFE-2d", 48, 64), (3, "PS-DTFE", 16, 16)):
            snap_ = work / f"tie{dim}.hdf5"
            subprocess.run([sys.executable, str(ROOT / "tests" / "generate_ps_test_data.py"), "--out", str(snap_),
                            "--n", str(n_), "--box", "100", "--amplitude-factor", "1.2", "--crossed-waves",
                            "--dim", str(dim)], check=True, capture_output=True)
            subprocess.run([str(rs.REPO_ROOT / b), str(snap_), str(work / f"tie{dim}"), "--grid", str(grid),
                            "--periodic", "--input", "105", "--MpcUnit", "1", "--field", "density", "--verbose", "1"],
                           check=True, capture_output=True)
            t = G.OutputSet(work / f"tie{dim}")
            dims[dim] = (t.dim, t.n)
        check("real scalar-only outputs at the 64^2 == 16^3 tie (no sidecar): PS-DTFE-2d reads as 2D, PS-DTFE as 3D",
              dims == {2: (2, 64), 3: (3, 16)}, str(dims))
    else:
        print("   SKIP  the rigid-rotation check (DTFE-2d or DTFE not built)")

    log = work / "r2.runlog"
    _write_log(log, "Build: PS-DTFE-2d 3e0c1a65fd-dirty (built 2026-10-02T21:34:55Z)\n"
               "RUNNING: /x/PS-DTFE-2d in.hdf5 r2 --grid 64 --partition 2 2 --field density_a\n")
    rec = R.parse_runlog(log)
    check("run records: the 2D binary, its 2-number split, '²' in the run list's words",
          rec.dim == 2 and rec.partitions == 4 and rec.partition == 2 and "2² partitions" in rec.options_text()
          and "2D" in rec.options_text(), f"{rec.binary} {rec.partitions} {rec.options_text()}")
    _write_log(log, "Build: DTFE 3e0c1a65fd-dirty (built 2026-10-01T10:00:00Z)\n"
               "RUNNING: /x/DTFE in.hdf5 r2 --grid 64 --periodic --CIC --interlace\n")
    old = R.stale_reasons(R.parse_runlog(log))
    _write_log(log, "Build: DTFE 3e0c1a65fd-dirty (built 2026-10-03T08:00:00Z)\n"
               "RUNNING: /x/DTFE in.hdf5 r2 --grid 64 --periodic --CIC --interlace\n")
    new = R.stale_reasons(R.parse_runlog(log))
    check("stale outputs: an interlaced grid made before the phase fix is flagged, a newer one is not",
          any(lvl == "error" and "interlaced" in m for lvl, m in old) and not any("interlaced" in m for _, m in new),
          f"{old} | {new}")


def pipeline_options(tmp: Path, sim: Path):
    """2026-10-04: the Pipeline tab's run options (deposit, sub-samples, GPU, vertex mass, volume weighting,
    caustics, cusps, parallel triangulation), the plane's geometry (planes, thickness, sub-samples, window)
    and the render step's options reach the scripts, and the memory check follows every one of them."""
    import json
    import subprocess
    print("pipeline options:")
    base = dict(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=64, grid=128, scratch_dir=str(tmp.parent))
    run_keys = ("PS_EXACT", "AVG_SUBSAMPLES", "PS_GPU", "PS_VERTEX_MASS", "PS_VOLUME_WEIGHTED", "PS_CAUSTICS",
                "PS_CAUSTIC_CUSPS", "PS_PARALLEL_TRI", "PLANES", "SUPERSAMPLE")
    env = rs.PipelineSpec(**base).command()[1]
    check("defaults: every run option is spelled out for run_ps_dtfe.sh (a stray PS_* in the login shell cannot leak in)",
          [env[k] for k in run_keys] == ["0", "3", "1", "1", "1", "0", "0", "0", "1", "1"]
          and "THICKNESS" not in env and "WINDOW" not in env, str(env))
    q = rs.PipelineSpec(**base, deposit="exact", nsub=1, gpu=False, vertex_mass=False, volume_weighted=False,
                        caustics=True, caustic_cusps=True, parallel_triangulation=True, planes=16, thickness=3.5,
                        supersample=2, window="10 60 0 25", precision="double")
    env = q.command()[1]
    check("every option switched: the environment follows",
          [env[k] for k in run_keys + ("THICKNESS", "WINDOW", "DTFE_PRECISION")]
          == ["1", "1", "0", "0", "0", "1", "1", "1", "16", "2", "3.5", "10 60 0 25", "double"], str(env))
    st = q.check_step("TNG100-3-Dark", [99], 32)
    check("the memory check carries the run options and the precision, and no -m without the GPU",
          st.argv[1:] == ["-s", "TNG100-3-Dark", "-g", "128", "99"] and st.env["DTFE_PRECISION"] == "double"
          and st.env["PS_EXACT"] == "1" and st.env["PS_CAUSTICS"] == "1" and st.env["AUTO_TUNE_REPORT"] == "1"
          and st.env["PTS_VEL_GRAD"] == "1" and not any(k in st.env for k in rs.PIPELINE_ONLY_KEYS),
          f"{st.argv} {st.env}")
    # 16 planes x 32 x nv x 2^2, nv = round(32 * 25 / 50) = 16 (square pixels over the window's aspect)
    check("a plane not made yet is sized by planes x nu x nv x K^2, nv following the window's aspect",
          st.env.get("DTFE_AUTO_PTS_N") == str(16 * 32 * 16 * 4) and q.plane_points(32) == 16 * 32 * 16 * 4
          and rs.PipelineSpec(**base).plane_points() == 64 * 64, str(st.env.get("DTFE_AUTO_PTS_N")))
    keys = {name: rs.PipelineSpec(**base, **kw).memory_key() for name, kw in
            (("plain", {}), ("caustics", dict(caustics=True)), ("super", dict(supersample=2)),
             ("render", dict(render_smooth=3.0, render_fields="density", render_fixed_range=True, thickness=9.0)))}
    check("the memory key follows the compute options, not the render ones (nor the thickness)",
          len({keys["plain"], keys["caustics"], keys["super"]}) == 3 and keys["render"] == keys["plain"])

    side = sim / "hires_plane_z64.json"         # an old plane's sidecar, as make_image_plane.py wrote them
    side.write_text(json.dumps({"box": 110.7, "center": 55.35, "planes": 1, "thickness": 2.0, "nu": 64, "nv": 64,
                                "u0": 0.0, "u1": 110.7, "v0": 0.0, "v1": 110.7}))
    p = rs.PipelineSpec(**base)
    check("plane_matches: the default geometry matches an old single-plane sidecar (no 'supersample' key = 1)",
          p.plane_matches("TNG100-3-Dark") is True)
    differ = {k: rs.PipelineSpec(**base, **kw).plane_matches("TNG100-3-Dark")
              for k, kw in (("planes", dict(planes=4)), ("supersample", dict(supersample=2)),
                            ("window", dict(window="0 50 0 50")), ("center", dict(center="30")))}
    check("plane_matches: another plane count, sub-sampling, window or centre differs",
          all(v is False for v in differ.values()), str(differ))
    check("plane_matches: the sidecar's own centre and a window spelling the whole box both match",
          rs.PipelineSpec(**base, center="55.35", window="0 110.7 0 110.7").plane_matches("TNG100-3-Dark") is True)
    check("stale(): a plane of another geometry makes every snapshot stale, a matching one keeps the fresh ones",
          p.stale("TNG100-3-Dark") == ([99], 2)
          and rs.PipelineSpec(**base, planes=4).stale("TNG100-3-Dark") == ([4, 99], 2),
          f"{p.stale('TNG100-3-Dark')} {rs.PipelineSpec(**base, planes=4).stale('TNG100-3-Dark')}")
    check("check_step: a plane the script will regenerate is sized by its point count, not read from disk",
          "SAMPLE_POINTS" in p.check_step("TNG100-3-Dark", [99], 64).env
          and rs.PipelineSpec(**base, planes=4).check_step("TNG100-3-Dark", [99], 64).env.get("DTFE_AUTO_PTS_N")
          == str(4 * 64 * 64))
    tool = str(rs.REPO_ROOT / "python" / "tools" / "make_image_plane.py")
    before = {f.name: (f.stat().st_mtime_ns, f.stat().st_size) for f in sim.iterdir()}
    same = subprocess.run([sys.executable, tool, "--same-as", str(side), "--nu", "64"], capture_output=True, text=True)
    diff = subprocess.run([sys.executable, tool, "--same-as", str(side), "--nu", "64", "--planes", "4"],
                          capture_output=True, text=True)
    bad = subprocess.run([sys.executable, tool, "--same-as", str(side), "--nu", "64", "--u0", "50", "--u1", "10"],
                         capture_output=True, text=True)
    gone = subprocess.run([sys.executable, tool, "--same-as", str(sim / "no_such_plane.json"), "--nu", "64"],
                          capture_output=True, text=True)
    check("make_image_plane.py --same-as (the script's check): exit 0 and 'same' for this geometry, 3 and 'differs' "
          "for another or an unreadable sidecar, 1 for invalid input (not read as 'differs'); nothing written",
          same.returncode == 0 and same.stdout.strip() == "same" and diff.returncode == 3
          and diff.stdout.strip() == "differs" and gone.returncode == 3 and bad.returncode == 1
          and {f.name: (f.stat().st_mtime_ns, f.stat().st_size) for f in sim.iterdir()} == before,
          f"{same.returncode} {same.stdout!r} {same.stderr[-200:]} | {diff.returncode} {diff.stdout!r} | "
          f"{bad.returncode} {bad.stderr[-120:]!r} | {gone.returncode}")
    # the launcher's reading of a broken or missing sidecar agrees with the script's (regenerate = all stale)
    good = side.read_text()
    side.write_text("{ not json")
    broken = (rs.PipelineSpec(**base).plane_matches("TNG100-3-Dark"), rs.PipelineSpec(**base).stale("TNG100-3-Dark"))
    side.unlink()
    missing = (rs.PipelineSpec(**base).plane_matches("TNG100-3-Dark"),
               "DTFE_AUTO_PTS_N" in rs.PipelineSpec(**base).check_step("TNG100-3-Dark", [99], 64).env)
    side.write_text(good)
    check("an unreadable or missing sidecar reads as 'differs': every snapshot stale, the check sized by point count",
          broken == (False, ([4, 99], 2)) and missing == (False, True), f"{broken} {missing}")
    check("a malformed window of the job itself is not the plane's fault (None; problems() reports it)",
          rs.PipelineSpec(**base, window="1 2").plane_matches("TNG100-3-Dark") is None)
    # the numbers the script receives read back exactly (':g' kept six digits)
    fine = rs.PipelineSpec(**base, window="12.3456789 50 0 50", planes=2, thickness=0.123456789)
    env = fine.command()[1]
    u0, u1, v0, v1 = (float(v) for v in env["WINDOW"].split())      # as make_image_plane.py stores what it got
    side2 = dict(json.loads(good), u0=u0, u1=u1, v0=v0, v1=v1, planes=2, thickness=float(env["THICKNESS"]))
    side.write_text(json.dumps(side2))
    check("WINDOW/THICKNESS reach the script at full precision, so the sidecar they produce matches the job",
          env["WINDOW"] == "12.3456789 50 0 50" and env["THICKNESS"] == "0.123456789"
          and fine.plane_matches("TNG100-3-Dark") is True, f"{env['WINDOW']} {env['THICKNESS']}")
    side.write_text(good)
    check("the GPU knob is pinned in the environment too (PS_METAL / DTFE_METAL), not only by -m",
          rs.PipelineSpec(**base).command()[1]["PS_METAL"] == "1"
          and rs.PipelineSpec(**base, gpu=False).command()[1]["PS_METAL"] == "0"
          and rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99], gpu=False).command()[1]["PS_METAL"] == "0"
          and rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99], estimator="dtfe",
                         gpu=False).command()[1]["DTFE_METAL"] == "0")

    r = rs.PipelineSpec(**base, render_fields="density, velDiv", render_smooth=2.0, render_smooth_derivatives=0.0,
                        render_fixed_range=True, render_force=True, planes=16, render_project="slab")
    a = r.steps()[1].argv
    check("the render step passes the figure options to plot_pointeval.py",
          a[a.index("--fields") + 1] == "density,velDiv" and a[a.index("--project") + 1] == "slab"
          and a[a.index("--smooth") + 1] == "2" and a[a.index("--smooth-derivatives") + 1] == "0"
          and "--fixed-range" in a and "--force" in a, str(a))
    a0 = rs.PipelineSpec(**base, render_project="slab").steps()[1].argv
    check("... the defaults add nothing, and 'slab' of a single plane is the plane itself",
          not any(f in a0 for f in ("--fields", "--project", "--smooth", "--smooth-derivatives", "--fixed-range",
                                    "--force")), str(a0))
    lv, msg = levels(rs.PipelineSpec(**base, caustic_cusps=True))
    check("cusps without caustics is an error (the Grids tab's rule, shared)", "error" in lv and "cusps" in msg, msg)
    lv, msg = levels(rs.PipelineSpec(**base, planes=4))
    check("a handful of planes warns about ghosting", "warning" in lv and "ghosts" in msg, msg)
    lv, msg = levels(rs.PipelineSpec(**base, window="0 50 0"))
    check("a malformed window is an error", "error" in lv and "window" in msg, msg)
    lv, msg = levels(rs.PipelineSpec(**base, render_fields="density,bogus"))
    check("an unknown point-evaluated field is an error", "error" in lv and "bogus" in msg, msg)
    check("the queue summary names an exact deposit",
          ", exact" in rs.job_summary("pipeline", rs.PipelineSpec(**base, deposit="exact"))
          and ", exact" not in rs.job_summary("pipeline", rs.PipelineSpec(**base)))
    check("the pipeline's plan line is parsed with and without -m (PS_GPU)",
          rs.parse_progress("+ env SAMPLE_POINTS=/p.bin PS_VOLUME_WEIGHTED=1 PTS_VEL_GRAD=1 /r/scripts/run_ps_dtfe.sh "
                            "-s S -g 512 -m 50 99\n").get("planned") == 2
          and rs.parse_progress("+ env SAMPLE_POINTS=/p.bin PS_VOLUME_WEIGHTED=0 PTS_VEL_GRAD=1 /r/scripts/run_ps_dtfe.sh "
                                "-s S -g 512 50 99\n").get("planned") == 2)
    from dtfelib import pointeval
    check("runspec's point-eval constants mirror dtfelib.pointeval (not imported there: matplotlib)",
          rs.POINTEVAL_FIELDS == tuple(pointeval.STYLES) and rs.POINTEVAL_DERIVATIVE_SMOOTH == pointeval.DERIVATIVE_SMOOTH)
    old = rs.PipelineSpec.from_dict({"sims": ["TNG50-4"], "nu": 4096, "render": False})
    check("settings saved before these options load with the scripts' defaults",
          old.deposit == "sampled" and old.gpu and old.planes == 1 and old.render_project == "plane" and not old.render)


def bug_fixes_2026_10_05(tmp: Path, sim: Path):
    """The bugs of the 2026-10-05 audit: the Explore cube cache follows the file, a standard-DTFE run gets its
    scratch folder, the demo is generated only at its own path, one lambda_th default, every products() caller
    passes --data-root."""
    import tempfile as tf
    import numpy as np
    import grids as G
    print("bug fixes 2026-10-05:")
    # B1: a grid written again is read again (the cache was keyed on the path alone)
    od = Path(tf.mkdtemp(prefix="gui_cube_"))
    try:
        n = 8
        np.arange(n ** 3, dtype=np.float32).tofile(od / "run.a_den")
        o1 = G.OutputSet(od / "run")
        before = float(o1.slice("density", 2, 3)[1, 2])
        np.full(n ** 3, 7.0, dtype=np.float32).tofile(od / "run.a_den")       # the re-run: same size
        st = (od / "run.a_den").stat()
        os.utime(od / "run.a_den", ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))   # a coarse clock cannot hide it
        after = float(G.OutputSet(od / "run").slice("density", 2, 3)[1, 2])
        keys = [k for k in G._CUBES if k[0] == (od / "run.a_den").resolve()]
        check("Explore's cube cache follows the file: a re-run's values replace the old ones (one entry per file)",
              before != 7.0 and after == 7.0 and len(keys) == 1, f"{before} {after} {len(keys)}")
    finally:
        shutil.rmtree(od)
    # B4: the standard-DTFE Grids run carries its scratch folder (the command returned before setting it)
    sd = tmp.parent
    env = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99], estimator="dtfe", scratch_dir=str(sd)).command()[1]
    check("a standard-DTFE Grids run passes SCRATCH_DIR (run_dtfe.sh has the hook now)", env.get("SCRATCH_DIR") == str(sd), str(env))
    # B5: the demo is generated only at the demo's own path
    demo = rs.demo_spec(output_dir=str(tmp / "demo"))
    other = rs.CustomSpec.from_dict(dict(demo.to_dict(), input_file=str(tmp / "elsewhere" / "missing.hdf5")))
    check("the demo generates its snapshot at its own path only; another missing path is 'file not found'",
          demo.demo_pending() and any("generate the demo" in st.label for st in demo.steps())
          and not other.demo_pending() and not any("generate" in st.label for st in other.steps())
          and any("not found" in m for lvl, m in other.problems() if lvl == "error"),
          str([st.label for st in other.steps()]))
    legacy = rs.CustomSpec.from_dict({"input_file": str(tmp / "gone.hdf5"), "demo_n": 32})   # saved before demo_file
    check("a saved demo_n without demo_file (older settings) generates nothing", not legacy.demo_pending())
    # B13: one lambda_th default across the tabs
    cs = rs.CustomSpec(input_file=str(sim / "snapdir_099" / "combined_099.hdf5"), fields=["density_a", "tweb_a"])
    cs0 = rs.CustomSpec(input_file=str(sim / "snapdir_099" / "combined_099.hdf5"), fields=["density_a"])
    a, a0 = cs.command(), cs0.command()
    check("a Custom run with a web field passes the scripts' lambda_th (0.3), one without passes none",
          a[a.index("--lambda_th") + 1] == "0.3" and "--lambda_th" not in a0, str(a))
    check("the Grids/Pipeline specs pass LAMBDA_TH only when it differs from the scripts' default",
          "LAMBDA_TH" not in rs.RunSpec(sim="TNG100-3-Dark", snapshots=[99], data_root=str(tmp)).command()[1]
          and rs.RunSpec(sim="TNG100-3-Dark", snapshots=[99], data_root=str(tmp), lambda_th=0.2).command()[1]["LAMBDA_TH"] == "0.2"
          and rs.PipelineSpec(sims=["TNG100-3-Dark"], data_root=str(tmp), lambda_th=0.4).command()[1]["LAMBDA_TH"] == "0.4"
          and rs.RunSpec(sim="TNG100-3-Dark", snapshots=[99], data_root=str(tmp), lambda_th=0.2).memory_key()
          == rs.RunSpec(sim="TNG100-3-Dark", snapshots=[99], data_root=str(tmp)).memory_key()
          and rs.CustomSpec(input_file="x.hdf5", lambda_th=0.2).memory_key() == rs.CustomSpec(input_file="x.hdf5").memory_key())
    # B14: no plot script asks pipeline.products() without the root it was given
    import re
    bad = sorted(f.name for f in (ROOT / "python" / "plot").glob("*.py")
                 if re.search(r"pipeline\.products\((?![^)]*data_root)", f.read_text().replace("\n", " ")))
    check("every pipeline.products() call in python/plot passes data_root", bad == [], ", ".join(bad))
    from dtfelib import cli as dcli
    keep = dcli.DATA_ROOT
    try:
        dcli.use_data_root(tmp)
        check("use_data_root(): the helpers without a root argument (trees, group catalogues, the ladder) follow --data-root",
              dcli.sim_dir("TNG100-3-Dark") == tmp / "TNG100" / "TNG100-3-Dark" and dcli.DATA_ROOT == tmp)
    finally:
        dcli.use_data_root(keep)


def cube_cache_2026_10_06():
    """The fast-grid review (2026-10-06): the cube cache is thread-safe and droppable, a big cube is read
    whole only along its strided axis, a short file does not leave the lock held."""
    import tempfile as tf
    import threading
    import numpy as np
    import grids as G
    print("cube cache 2026-10-06:")
    od = Path(tf.mkdtemp(prefix="gui_cubes_"))
    eager = G.CUBE_EAGER_BYTES
    try:
        n = 8
        ref = np.arange(n ** 3, dtype=np.float32).reshape(n, n, n)
        ref.tofile(od / "big.a_den")
        key = (od / "big.a_den").resolve()
        G.drop_cubes()
        G.CUBE_EAGER_BYTES = 1000                                   # the 2 KB cube counts as 'big'
        o = G.OutputSet(od / "big")
        p0 = o.slice("density", 0, 3)
        lazy = not any(k[0] == key for k in G._CUBES)
        p2 = o.slice("density", 2, 5)
        cached = any(k[0] == key for k in G._CUBES)
        p0b = o.slice("density", 0, 3)
        still = any(k[0] == key for k in G._CUBES)
        check("a cube above CUBE_EAGER_BYTES: an x plane comes from the memmap (not read whole), the strided z plane "
              "caches it, and the x plane then comes from the cache", lazy and cached and still
              and np.array_equal(p0, ref[3]) and np.array_equal(p2, ref[:, :, 5]) and np.array_equal(p0b, ref[3]),
              f"{lazy} {cached} {still}")
        np.arange(32 * 32, dtype=np.float32).tofile(od / "flat.a_den")
        G.OutputSet(od / "flat", dim=2).slice("density")
        check("a 2D grid is always read whole and cached, whatever its size against the threshold",
              any(k[0] == (od / "flat.a_den").resolve() for k in G._CUBES))
        G.CUBE_EAGER_BYTES = eager
        dropped = G.drop_cubes()
        check("drop_cubes() returns the bytes held and empties the cache", dropped == 32 * 32 * 4 + n ** 3 * 4
              and not G._CUBES, f"{dropped} {len(G._CUBES)}")
        # two threads reading the same cold cube: one entry, both the right array, no exception
        got, errs = [], []
        go = threading.Barrier(2)

        def reader():
            try:
                go.wait(5)
                got.append(G.OutputSet(od / "big").slice("density", 2, 1))
            except Exception as e:            # noqa: BLE001
                errs.append(repr(e))
        ts = [threading.Thread(target=reader) for _ in range(2)]
        [t.start() for t in ts]
        [t.join(10) for t in ts]
        entries = [k for k in G._CUBES if k[0] == key]
        check("two threads slicing the same cold cube at once: one cache entry, both get the plane",
              not errs and len(got) == 2 and len(entries) == 1 and all(np.array_equal(g, ref[:, :, 1]) for g in got),
              f"{errs} {len(got)} {len(entries)}")
        # a file shorter than its grid (a run rewriting it): the error comes out, the lock does not stay held
        o = G.OutputSet(od / "big")
        G.drop_cubes()
        with open(od / "big.a_den", "r+b") as fh:
            fh.truncate(n ** 3 * 4 - 16)
        raised = False
        try:
            o.slice("density", 2, 1)
        except ValueError:
            raised = True
        ok = G._CUBES_LOCK.acquire(timeout=2)
        if ok:
            G._CUBES_LOCK.release()
        again = G.OutputSet(od / "flat", dim=2).slice("density")
        check("a short file raises ValueError from slice() and leaves the lock free (the next slice works)",
              raised and ok and again.shape == (32, 32), f"{raised} {ok}")
    finally:
        G.CUBE_EAGER_BYTES = eager
        G.drop_cubes()
        shutil.rmtree(od)


def type_labels_2026_10_06():
    """The figure grid's Type column: which estimator made an output -- the sidecar, else a streams grid
    (only the phase-space binary writes one), else the prefix; the prefix shown only when not the default."""
    import json
    import tempfile as tf
    import numpy as np
    import grids as G
    print("type labels 2026-10-06:")
    od = Path(tf.mkdtemp(prefix="gui_types_"))
    try:
        n = 4
        def out(prefix, streams=False, estimator=None):
            np.ones(n ** 3, dtype=np.float32).tofile(od / f"{prefix}.a_den")
            if streams:
                np.ones(n ** 3, dtype=np.float32).tofile(od / f"{prefix}.a_streams")
            if estimator:
                (od / f"{prefix}.gui.json").write_text(json.dumps({"estimator": estimator}))
            return G.OutputSet(od / prefix)
        got = {
            "ps_output": G.type_label(out("ps_output")),
            "output": G.type_label(out("output")),
            "ps_x density only": G.type_label(out("ps_x")),
            "foo + streams": G.type_label(out("foo", streams=True)),
            "sidecar dtfe wins": G.type_label(out("ps_y", streams=True, estimator="dtfe")),
            "sidecar ps wins": G.type_label(out("bar", estimator="ps")),
        }
        want = {"ps_output": "PS-DTFE", "output": "DTFE", "ps_x density only": "PS-DTFE · ps_x",
                "foo + streams": "PS-DTFE · foo", "sidecar dtfe wins": "DTFE · ps_y", "sidecar ps wins": "PS-DTFE · bar"}
        check("type labels: the default prefixes bare, others with the prefix; a ps prefix without streams is still "
              "PS-DTFE; a streams grid makes any prefix PS-DTFE; the sidecar beats both", got == want, str(got))
    finally:
        shutil.rmtree(od)


def launcher_fixes():
    """2026-10-04: running_jobs() classifies by command line and parent, and no plot script builds
    the flat <root>/<sim> path by hand (the T7 keeps simulations per family)."""
    import subprocess
    print("running jobs:")
    me = os.getpid()
    # pgrep names five DTFE binaries; ps shows what they are: the launcher's query server (--serve, a direct
    # child), its exact zoom (a direct child, --ps-window), a run from another terminal (bash -> binary: a
    # grandchild), a run whose OUTPUT PATH contains '--serve' (the token rule), a memory check from a terminal
    # (--auto-tune-report, never counted) and someone else's query server (listed apart, with its memory)
    pgrep_out = "100 PS-DTFE\n200 PS-DTFE\n300 DTFE-double\n400 PS-DTFE\n500 PS-DTFE\n600 PS-DTFE\n"
    ps_out = (f"  100 {me} 24000000 /Users/x/Mobile Documents/DTFE/PS-DTFE /Volumes/T7/snap.hdf5 /tmp/s/serve --serve "
              f"--input 105 --verbose 0 --serve-resident 4 --serve-progress\n"
              f"  200 {me} 1500000 /Users/x/Mobile Documents/DTFE/PS-DTFE /Volumes/T7/snap.hdf5 /tmp/z/zoom --ps-window "
              f"1 2 3 4 5 6 --field density_a\n"
              f"  300 4242 18200000 /Users/x/Mobile Documents/DTFE/DTFE-double /Volumes/T7/snap.hdf5 out --grid 256 256 256\n"
              f"  400 4343 900000 ./PS-DTFE snap.hdf5 /tmp/--serve-test/out --periodic\n"
              f"  500 4444 300000 ./PS-DTFE snap.hdf5 /tmp/r/out --grid 512 --auto-tune-report\n"
              f"  600 4545 9300000 ./PS-DTFE other.hdf5 /tmp/s2/serve --serve --input 105\n")
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[0] == "pgrep":
            return subprocess.CompletedProcess(argv, 0, stdout=pgrep_out, stderr="")
        if argv[0] == "ps":
            assert argv[-1] == "100,200,300,400,500,600" and "-ww" in argv and "rss=" in argv[3], argv
            return subprocess.CompletedProcess(argv, 0, stdout=ps_out, stderr="")
        raise AssertionError(argv)

    orig_run, orig_cache = rs.subprocess.run, rs._JOBS_CACHE
    try:
        rs.subprocess.run = fake_run
        rs._JOBS_CACHE = (-1e9, {"runs": [], "servers": []})
        rs.dtfe_processes(max_age=0)             # one classification; the two views below read the cache
        jobs = rs.running_jobs()
        check("runs: a run from another terminal and one whose path merely contains '--serve'; not the launcher's own "
              "server or exact zoom, not a memory check (--auto-tune-report), not another server",
              jobs == [(300, "DTFE-double"), (400, "PS-DTFE")], str(jobs))
        servers = rs.running_servers()
        check("servers: every query server, with its resident memory in GB (ps rss)",
              [(p, n, round(g, 1)) for p, n, g in servers] == [(100, "PS-DTFE", 24.0), (600, "PS-DTFE", 9.3)], str(servers))
        check("one pgrep, then one ps over exactly the named pids", [c[0] for c in calls] == ["pgrep", "ps"], str(calls))
        msgs = rs._busy_problems()
        check("the busy warning names the runs with their memory; the servers are an info line with theirs",
              [lvl for lvl, _ in msgs] == ["warning", "info"] and "DTFE-double (pid 300, 18.2 GB)" in msgs[0][1]
              and "PS-DTFE (pid 400, 0.9 GB)" in msgs[0][1] and "PS-DTFE (pid 100, 24.0 GB)" in msgs[1][1]
              and "(pid 600, 9.3 GB)" in msgs[1][1], str(msgs))
        rs._JOBS_CACHE = (-1e9, {"runs": [], "servers": []})
        rs.subprocess.run = lambda argv, **kw: (fake_run(argv, **kw) if argv[0] == "pgrep"
                                               else (_ for _ in ()).throw(OSError("no ps")))
        check("without ps every named binary counts as a run (no memory known), as before",
              rs.running_jobs(max_age=0) == [(100, "PS-DTFE"), (200, "PS-DTFE"), (300, "DTFE-double"), (400, "PS-DTFE"),
                                             (500, "PS-DTFE"), (600, "PS-DTFE")]
              and rs.running_servers(max_age=0) == [] and "(pid 100)" in rs._busy_problems()[0][1])
        rs._JOBS_CACHE = (-1e9, {"runs": [], "servers": []})
        rs.subprocess.run = lambda argv, **kw: subprocess.CompletedProcess(argv, 1, stdout="", stderr="")
        check("nothing running: no ps call, empty lists, no message",
              rs.running_jobs(max_age=0) == [] and rs.running_servers(max_age=0) == [] and not rs._busy_problems())
    finally:
        rs.subprocess.run, rs._JOBS_CACHE = orig_run, orig_cache
    check("the kinds: server by the --serve token, report by --auto-tune-report, own by the parent, else a run",
          rs._kind(1, "x --serve y", 99) == "server" and rs._kind(1, "x --auto-tune-report", 99) == "report"
          and rs._kind(99, "x --grid 8", 99) == "own" and rs._kind(1, "x --grid 8", 99) == "run"
          and rs._kind(99, "x --serve", 99) == "server")
    check("the token rule: '--serve' alone, not --serve-resident/-progress or a path",
          bool(rs._SERVE.search("a --serve b")) and bool(rs._SERVE.search("x --serve"))
          and not rs._SERVE.search("a --serve-resident 4 b") and not rs._SERVE.search("/tmp/--serve/out"))

    print("plot scripts:")
    flat = sorted(f.name for f in (ROOT / "python" / "plot").glob("*.py")
                  if "args.data_root / args.sim" in f.read_text())
    check("no plot script builds <root>/<sim> by hand (sim_dir resolves the T7's per-family layout)",
          flat == [], ", ".join(flat))
    from dtfelib.cli import sim_dir
    root = Path(tempfile.mkdtemp(prefix="gui_simdir_"))
    try:
        (root / "TNG100" / "TNG100-3-Dark").mkdir(parents=True)
        check("sim_dir: a per-family simulation resolves under its family folder",
              sim_dir("TNG100-3-Dark", root) == root / "TNG100" / "TNG100-3-Dark")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    sys.exit(main())
