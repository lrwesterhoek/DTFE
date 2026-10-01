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
        (sim / "hires_plane_z64.json").touch()
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
    check("find_outputs: a custom-snapshot run from its folder, box from its sidecar",
          len(mine) == 1 and mine[0].length[0] == 50.0, str(outs))
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



if __name__ == "__main__":
    sys.exit(main())
