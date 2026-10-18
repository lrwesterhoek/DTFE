#!/usr/bin/env python3
"""Headless checks for the GUI's job specifications (python/gui/runspec.py): the commands
RunSpec, PipelineSpec, PlotSpec and CustomSpec stand for, the combinations they refuse, script
export, progress parsing and data discovery; and the GUI's other Qt-free modules: run logs, stale
outputs, time estimates and failure advice (results.py), presets and the binary's own help text
(presets.py), and the Explore tab's grid reader (grids.py). No Qt needed.

Usage: python3 tests/py_gui_runspec_test.py
"""

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "gui"))
FIGS_TMP = tempfile.mkdtemp(prefix="runspec_figs_")
os.environ["DTFE_FIGURES_ROOT"] = FIGS_TMP      # BEFORE runspec is imported: never the real root (the T7)
atexit.register(shutil.rmtree, FIGS_TMP, True)

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
        dt = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], estimator="dtfe", gpu=False,
                        deposit="sampled")
        check("DTFE argv uses run_dtfe.sh with no PS options",
              dt.command()[0] == [str(rs.DTFE_SCRIPT), "-s", "TNG100-3-Dark", "-g", "512", "4"]
              and "FIELDS" not in dt.command()[1])
        dte = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], estimator="dtfe", gpu=False,
                         deposit="exact")
        check("standard DTFE with the exact choice: run_dtfe.sh -e (the exact cell average), named in the queue",
              dte.command()[0] == [str(rs.DTFE_SCRIPT), "-s", "TNG100-3-Dark", "-g", "512", "-e", "4"]
              and rs.job_summary("grids", dte).endswith(", exact"), str(dte.command()[0]))
        # 2026-10-13: run_dtfe.sh's default is the exact average, and -e only switches it on: the job pins the knob
        check("... the cell average is pinned both ways (DTFE_EXACT_AVERAGE=0 for 'sampled', 1 for 'exact')",
              dt.command()[1].get("DTFE_EXACT_AVERAGE") == "0" and dte.command()[1].get("DTFE_EXACT_AVERAGE") == "1",
              f"{dt.command()[1]} {dte.command()[1]}")
        # 2026-10-13 (integration item 8): a standard-DTFE job STARTS at the exact cell average, run_dtfe.sh's and the
        # production setting; a PS-DTFE job at the sampled deposit; a choice saved in a settings dict is kept
        dflt = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], estimator="dtfe", gpu=False)
        check("a standard-DTFE job starts at the exact cell average (-e, DTFE_EXACT_AVERAGE=1), a PS-DTFE one at the "
              "sampled deposit, Grids and Custom alike; a saved choice is kept, a settings dict without one gets the "
              "estimator's own",
              dflt.deposit == "exact" and "-e" in dflt.command()[0] and dflt.command()[1]["DTFE_EXACT_AVERAGE"] == "1"
              and rs.RunSpec().deposit == "sampled" and rs.CustomSpec().deposit == "sampled"
              and rs.CustomSpec(estimator="dtfe").deposit == "exact"
              and rs.production_deposit("dtfe") == "exact" and rs.production_deposit("ps") == "sampled"
              and rs.RunSpec.from_dict({"estimator": "dtfe", "deposit": "sampled"}).deposit == "sampled"
              and rs.RunSpec.from_dict({"estimator": "dtfe"}).deposit == "exact"
              and rs.CustomSpec.from_dict(rs.CustomSpec(estimator="dtfe", deposit="sampled").to_dict()).deposit
              == "sampled", f"{dflt.deposit} {dflt.command()}")
        def _script_default(script, var):    # the driver's own default, read from its 'VAR="${VAR:-...}"' line
            out = subprocess.run(["bash", "-c", f'eval "$(grep -E "^{var}=" "$1" | head -1)"; printf %s "${{{var}}}"', "_",
                           str(script)], capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "")})
            return out.stdout
        check("the launcher's defaults are the drivers' (2026-10-13): the field lists, the caustics, the volume weighting",
              _script_default(rs.PS_SCRIPT, "FIELDS").split() == rs.PS_DEFAULT_FIELDS
              and _script_default(rs.DTFE_SCRIPT, "FIELDS").split() == rs.DTFE_FIELDS
              and _script_default(rs.PS_SCRIPT, "PS_CAUSTICS") == "1" and rs.RunSpec().caustics and rs.PipelineSpec().caustics
              and _script_default(rs.PS_SCRIPT, "PS_VOLUME_WEIGHTED") == "1" and rs.RunSpec().volume_weighted
              and rs.PipelineSpec().volume_weighted and rs.RunSpec().grid == rs.PipelineSpec().grid == 512,
              f"{_script_default(rs.PS_SCRIPT, 'FIELDS')!r} {_script_default(rs.DTFE_SCRIPT, 'FIELDS')!r}")
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
            # a slice OF this plane: 64 x 64 float64 (the count is compared since 2026-10-11)
            (sim / f"snapdir_{n:03d}" / "ps_output.pts_den").write_bytes(b"\0" * 8 * 64 * 64)
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
        # 2026-10-11: a slice newer than the plane but of ANOTHER plane size (TNG100's 4096^2 slices beside an
        # older hires_plane_z8192) read as current: the script skipped the compute and every render failed
        den4 = sim / "snapdir_004" / "ps_output.pts_den"
        den4.write_bytes(b"\0" * 8 * 32 * 32)
        os.utime(den4, (now, now))
        check("stale(): a slice of another plane size is stale, though newer than the plane (its point count)",
              pipe.stale("TNG100-3-Dark") == ([4, 99], 2), str(pipe.stale("TNG100-3-Dark")))
        den4.write_bytes(b"\0" * 8 * 64 * 64)
        os.utime(den4, (now, now))
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
        check("point-eval maps: comma lists, fields de-spaced, --force, NO grid smoothing, the estimator (2026-10-11)",
              by["plot_pointeval.py"] == [["--sims", "TNG100-3-Dark", "--snaps", "4,99", "--method", "ps", "--fields",
                                           "density,velDiv", "--force"]], str(by["plot_pointeval.py"]))
        pan = by["plot_pointeval_panels.py"]
        check("panels: per snapshot, its own grid prefix, the plane, an -o under python/figures",
              len(pan) == 2 and pan[0][:2] == ["--prefix", str(sim / "snapdir_004" / "ps_output")]
              and pan[0][pan[0].index("-o") + 1].startswith(str(rs.FIGURES_ROOT / "pointeval_panels")), str(pan))
        check("halo tracking: explicit IDs replace --top",
              by["plot_halo_tracking.py"][0][-3:] == ["--subhalo", "0", "17"], str(by["plot_halo_tracking.py"]))
        check("thesis set: analyze.py all, zero-padded snapshots, --only",
              by["analyze.py"] == [["all", "004", "099", "--only", "plot_DTFE.py"]], str(by["analyze.py"]))
        spec_e = rs.PlotSpec.from_dict(dict(spec.to_dict(), sets=["thesis"], options={"thesis": {"mode": "export", "only": "plot_DTFE.py"}}))
        argv_e = [st.argv for st in spec_e.steps()]
        check("thesis set, mode 'export': analyze.py export with the snapshots and no --only (2026-10-05)",
              [a[a.index("export"):] for a in argv_e if "export" in a] == [["export", "004", "099"]], str(argv_e))
        thesis_grids_warning()
        tick_helpers()
        run_outputs_not_inputs()
        run_options(tmp)
        audit_claims()
        cube_cap()
        point_window_linear(tmp)
        plane_reads()
        plot_options(tmp)
        plot_void_sets(tmp)
        plot_void_script_options(tmp)
        fixed_r_size_function(tmp)
        crossbox_smoothing_constant()
        gpu_retry_causes(tmp)
        explore_extras(tmp)
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
        # 2026-10-13: the standard binary's tuner reports too, through run_dtfe.sh's own AUTO_TUNE_REPORT
        sgrids = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99, 4], estimator="dtfe")
        sst = sgrids.check_step()
        check("... and a standard-DTFE grids check is the same run_dtfe.sh call plus AUTO_TUNE_REPORT=1",
              sst.argv == sgrids.command()[0] and sst.argv[0].endswith("run_dtfe.sh")
              and sst.env == dict(sgrids.command()[1], AUTO_TUNE_REPORT="1"), str(sst.argv))
        check("the plan as the launcher says it: PS-DTFE runs mc partitions at a time, standard DTFE its partitions "
              "one after another with mc threads each",
              rs.report_split_text({"partition": 4, "mc": 5}, "ps") == "4³ partitions, 5 at a time"
              and rs.report_split_text({"partition": 1, "mc": 10}, "ps") == "one tessellation"
              and rs.report_split_text({"partition": 4, "mc": 4}, "dtfe") == "4³ partitions one after another, 4 threads each"
              and rs.report_split_text({"partition": 1, "mc": 1}, "dtfe", 2) == "one domain, 1 thread"
              and rs.report_split_text({"partition": 3, "mc": 9}, "dtfe", 2) == "3² partitions one after another, 9 threads each")
        check("the memory key ignores the snapshot choice but not the settings",
              rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4]).memory_key()
              == rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99]).memory_key()
              != rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], grid=256).memory_key())
        pm = rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=64, grid=128, scratch_dir=str(tmp.parent))
        have = pm.check_step("TNG100-3-Dark", [99, 4], 64)                 # hires_plane_z64.bin exists
        need = pm.check_step("TNG100-3-Dark", [99], 32)                    # hires_plane_z32 does not
        check("a pipeline check passes the pipeline's settings, the plane file when it exists ...",
              have.argv[1:] == ["-s", "TNG100-3-Dark", "-g", "128"] + (["-m"] if rs.gpu_build_default("ps") else [])
              + ["4", "99"]
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
        # 2026-10-08: run_ps_pipeline.sh exits 1 when it computed nothing, and the render after it must not run then
        check("... each render waits for its own pipeline call (after = that call's label; the calls wait for nothing)",
              [st.after for st in pm.steps()[1::2]] == [st.label for st in pm.steps()[0::2]]
              and all(st.after == "" for st in pm.steps()[0::2]), str([(st.label, st.after) for st in pm.steps()]))
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
        figures_root_rule()
        void_outline_geometry()
        run_log_contract()
        launcher_2026_10_11()
        cpp_fix_flags_2026_10_11()
        pts_conventions_flag_2026_10_13()
        pts_conventions_choice_2026_10_15()
        estimator_per_snapshot(tmp)
    finally:
        shutil.rmtree(tmp.parent)
    print("-" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


def estimator_per_snapshot(tmp: Path):
    """2026-10-11: the void catalogue build resolves estimator 'auto' per snapshot (it turned 'auto' into 'ps' while
    its check accepted a snapdir with standard-DTFE grids only, so the job failed there), and the point-evaluated
    sets read the estimator's own slice (they assumed ps_output, so a standard-DTFE slice beside PS-DTFE grids could
    not be rendered)."""
    import json
    print("estimator per snapshot:")
    root = tmp.parent / "est_root"
    sp = root / "BOX-1-Dark"
    for n in (4, 99):
        (sp / f"snapdir_{n:03d}").mkdir(parents=True, exist_ok=True)
    (sp / "snapdir_004" / "output.a_den").touch()                       # standard DTFE only
    (sp / "snapdir_099" / "ps_output.a_den").touch()                    # both
    (sp / "snapdir_099" / "output.a_den").touch()

    def methods(**kw):
        spec = rs.PlotSpec(data_root=str(root), sim="BOX-1-Dark", snapshots=[99, 4], sets=["voidcat"], **kw)
        return [st.argv[st.argv.index("--method") + 1] for st in spec.steps()]
    auto = rs.PlotSpec(data_root=str(root), sim="BOX-1-Dark", snapshots=[4, 99], sets=["voidcat"])
    lv, msg = levels(auto)
    check("void catalogue build, estimator auto: standard DTFE where only output.* grids are, PS-DTFE where "
          "ps_output.* are; the check agrees (no missing grids)",
          methods() == ["dtfe", "ps"] and "no " not in msg and "error" not in lv, f"{methods()} {msg}")
    check("... a chosen estimator, or a grid prefix (PS-DTFE), wins",
          methods(method="dtfe") == ["dtfe", "dtfe"] and methods(prefix="ps_caustic") == ["ps", "ps"]
          and methods(method="ps") == ["ps", "ps"], f"{methods(method='dtfe')} {methods(prefix='ps_caustic')}")

    side = sp / "hires_plane_z16.json"
    side.write_text(json.dumps({"box": 100.0, "nu": 16, "nv": 16, "planes": 1}))
    sd = sp / "snapdir_099"
    (sd / "output.pts_den").write_bytes(b"\0" * 8 * 16 * 16)            # a standard-DTFE slice beside PS-DTFE grids

    def pts(**kw):
        spec = rs.PlotSpec(data_root=str(root), sim="BOX-1-Dark", snapshots=[99], sets=["pointeval", "panels"],
                           options={"panels": {"plane": str(side)}}, **kw)
        st = {Path(x.argv[1]).name: x.argv for x in spec.steps()}
        pe, pa = st["plot_pointeval.py"], st["plot_pointeval_panels.py"]
        return (pe[pe.index("--method") + 1] if "--method" in pe else "auto", pa[pa.index("--prefix") + 1],
                any("no point-evaluated slice" in m for _, m in spec.problems()))
    check("point-evaluated sets, estimator DTFE: plot_pointeval gets --method dtfe, the panels read output.pts_*, "
          "and the check finds the slice",
          pts(method="dtfe") == ("dtfe", str(sd / "output"), False), str(pts(method="dtfe")))
    check("... auto: the slice that is there (output.pts_* here, though PS-DTFE grids are beside it); PS-DTFE: "
          "ps_output.pts_*, and its absence is warned about",
          pts() == ("auto", str(sd / "output"), False) and pts(method="ps") == ("ps", str(sd / "ps_output"), True),
          f"{pts()} {pts(method='ps')}")
    (sd / "ps_output.pts_den").write_bytes(b"\0" * 8 * 16 * 16)
    check("... auto with both slices: PS-DTFE first, as before",
          pts() == ("auto", str(sd / "ps_output"), False), str(pts()))
    # 2026-10-15: standard DTFE's grids are 'output.*' exactly; 'output*.a_den' matched the plane run's one-field
    # grid 'output_plane.a_den' (thesis_runs.sh dtfe27planes) and any other 'output_<variant>' beside them
    pl = sp / "snapdir_050"
    pl.mkdir()
    (pl / "output_plane.a_den").touch()
    (pl / "output_mc.a_den").touch()
    before = (rs.has_outputs(pl, "", "dtfe"), rs.has_grids(pl, "dtfe", ""), rs.has_grids(pl, "auto", ""))
    (pl / "output.a_den").touch()
    check("standard DTFE's grids are 'output.*' only: a snapdir with 'output_plane.a_den' and 'output_mc.a_den' has "
          "none (Grids marks, figure checks), one with 'output.a_den' has them",
          before == (False, False, False) and rs.has_outputs(pl, "", "dtfe") and rs.has_grids(pl, "dtfe", ""),
          str(before))


def gpu_branches(tmp: Path, snap: Path):
    """The GPU advice and recommendation for a GPU build AND a CPU-only one, whatever this host has: rs._objdir
    (where gpu_built, gpu_backend and tbb_built read the Makefile's stamps) points at a fake object directory with
    or without '.gpu_mode_off'."""
    real = rs._objdir
    fake = Path(tempfile.mkdtemp(prefix="gui_objdir_"))
    try:
        rs._objdir = lambda estimator, precision="single", dim=3: fake / (estimator + precision + str(dim))
        for built in (True, False):
            for d in (rs._objdir(e, p) for e in ("ps", "dtfe") for p in ("single", "double")):
                d.mkdir(exist_ok=True)
                (d / ".gpu_mode_off").unlink(missing_ok=True)
                (d / ".gpu_mode_metal").unlink(missing_ok=True)
                (d / (".gpu_mode_metal" if built else ".gpu_mode_off")).touch()
            print(f"  (faked build: {'GPU (Metal)' if built else 'CPU only'})")
            check(f"the faked stamp is what the build checks read ({built})",
                  rs.gpu_built("ps") is built and rs.gpu_built("dtfe") is built
                  and rs.gpu_backend("ps") == ("Metal" if built else ""))
            # 2026-10-15: a fresh job's GPU tick is the build's (it was on for a CPU-only build too); a saved one stays
            check(f"a fresh job's GPU tick is the build's ({built}): the Grids job (both estimators, double too), the "
                  "pipeline; a saved tick is kept either way",
                  rs.RunSpec().gpu is built and rs.RunSpec(estimator="dtfe").gpu is built
                  and rs.RunSpec(precision="double").gpu is built and rs.PipelineSpec().gpu is built
                  and rs.RunSpec.from_dict({"gpu": not built}).gpu is (not built)
                  and rs.PipelineSpec.from_dict({"gpu": not built}).gpu is (not built)
                  and rs.RunSpec(gpu=True).gpu is True,
                  f"{rs.RunSpec().gpu} {rs.PipelineSpec().gpu}")
            _gpu_advice_checks(tmp, snap)
        for d in (rs._objdir(e, p) for e in ("ps", "dtfe") for p in ("single", "double")):
            (d / ".gpu_mode_off").unlink(missing_ok=True)
            (d / ".gpu_mode_cudaemu").touch()
        check("the CI's emulated CUDA build (its kernels on the CPU): built for gpu_built, but a fresh job starts "
              "without the GPU, as scripts/config.sh gpu_build_default says",
              rs.gpu_built("ps") and not rs.gpu_build_default("ps") and rs.RunSpec().gpu is False
              and rs.PipelineSpec().gpu is False)
    finally:
        rs._objdir = real
        shutil.rmtree(fake, ignore_errors=True)


def _gpu_advice_checks(tmp: Path, snap: Path):
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
          and "--exact-average" not in rs.CustomSpec(input_file=str(snap), estimator="dtfe", deposit="sampled").command()
          and (any("6.6x" in m for _, m in dex.problems()) and any("6.6x" in m for _, m in cex.problems())) == dt_gpu,
          str(cex.problems()))
    # 2026-10-13 (integration items 6 and 8): the Custom tab's standard run starts at the exact cell average and pads
    # with 15 mean spacings (dtfelib.estimator.STANDARD_PADDING, run_dtfe.sh's and the standard server's value) instead
    # of the binary's 5, which leaves wrong values within a few spacings of a box face; PS-DTFE keeps its own padding
    cd0, cd2 = (rs.CustomSpec(input_file=str(snap), estimator="dtfe").command(),
                rs.CustomSpec(input_file=str(snap), estimator="dtfe", dim=2, deposit="sampled").command())
    cps = rs.CustomSpec(input_file=str(snap), estimator="ps").command()
    check("the Custom tab's standard run: --exact-average by default and '--padding 15' (3D and 2D, either average); "
          "a PS-DTFE run passes no --padding",
          "--exact-average" in cd0 and rs.STANDARD_PADDING == 15.0
          and all(c.count("--padding") == 1 and c[c.index("--padding") + 1] == "15" for c in (cd0, cd2))
          and "--padding" not in cps and "--exact-average" not in cd2, f"{cd0} {cd2} {cps}")


def tick_helpers():
    """The Explore map's axes and colour bar (survey items 12, 13): round ticks, their labels, the scale bar."""
    import numpy as np
    import grids as G
    t = G.nice_ticks(0, 51.67)
    check("axis ticks: round values inside the range, about five (0 .. 51.7 Mpc: every 10)", t == [0, 10, 20, 30, 40, 50], str(t))
    t = G.nice_ticks(-6.46, 6.46)
    check("... a 2.5 step keeps its decimal ('2' for 2.5 would be wrong)",
          [G.tick_text(v, t) for v in t] == ["-5.0", "-2.5", "0.0", "2.5", "5.0"], str(t))
    check("... a degenerate range has none", G.nice_ticks(1, 1) == [] and G.nice_ticks(0, float("nan")) == [])
    check("... an axis in cells keeps to whole steps (a 16-cell grid: 0, 5, 10, 15, not 2.5 apart; 4 cells: every one)",
          G.nice_ticks(0, 16, integer=True) == [0, 5, 10, 15] and G.nice_ticks(0, 4, integer=True) == [0, 1, 2, 3, 4]
          and G.nice_ticks(0, 16) == [0, 2.5, 5, 7.5, 10, 12.5, 15], str(G.nice_ticks(0, 16)))
    t = G.nice_ticks(120, 120.004)
    check("... a deep zoom far from the origin keeps the labels apart (not five '120')",
          [G.tick_text(v, t) for v in t] == ["120.000", "120.001", "120.002", "120.003", "120.004"], str(t))
    z = np.zeros((64, 64))
    z[3, 4], z[10, 10] = 0.1, -0.05                 # two runs that agree in all but two cells
    check("a 0-centred range of a difference that is 0 almost everywhere: its largest value, not (0, 1) "
          "(which painted the whole map the scale's end)",
          G.value_range(z, symmetric=True) == (-0.1, 0.1) and G.value_range(np.zeros((4, 4)), symmetric=True) == (-1.0, 1.0),
          str(G.value_range(z, symmetric=True)))
    check("the scale bar: a round length near a fifth of the width",
          [G.nice_length(s) for s in (110.7, 51.6, 6.5)] == [20.0, 10.0, 1.0])
    dec = G.colorbar_ticks(-1.2, 3.7, log=True)
    check("a log colour bar over several decades: 10^k at the whole decades",
          [tx for _, tx in dec] == ["10⁻¹", "1", "10", "10²", "10³"] and [v for v, _ in dec] == [-1, 0, 1, 2, 3], str(dec))
    few = G.colorbar_ticks(np.log10(0.0249), np.log10(7.58), log=True)
    check("... under three decades: 1, 2, 5 x 10^k as numbers",
          [tx for _, tx in few] == ["0.05", "0.1", "0.2", "0.5", "1", "2", "5"], str(few))


def thesis_grids_warning():
    """Review 2026-10-05 [37]: the Thesis analysis set warns when a snapshot has no standard-DTFE grids in the modes
    that READ them -- all, compute and export (analyze.export builds its FieldSet from the grids, like compute) --
    and not in plot / check (they read python/cache). Phase-space grids do not count: compute reads the standard
    DTFE. Its own data root: an empty snapdir_099, then PS grids only, then a standard-DTFE run. The estimator and
    grid prefix the user set for the OTHER sets do not count either: the thesis step passes analyze.py neither."""
    root = Path(tempfile.mkdtemp(prefix="gui_thesis_"))
    try:
        sd = root / "TNG100" / "TNG100-3-Dark" / "snapdir_099"
        sd.mkdir(parents=True)
        (sd.parent / "snapdir_050").mkdir()              # stays empty: the two-snapshot spec names it every time

        def warned(snaps=(99,), **kw):
            got, errs, msgs = set(), [], set()
            for mode in ("all", "compute", "export", "plot", "check"):
                spec = rs.PlotSpec(data_root=str(root), sim="TNG100-3-Dark", snapshots=list(snaps), sets=["thesis"],
                                   options={"thesis": {"mode": mode}}, **kw)
                probs = spec.problems()
                errs += [m for lvl, m in probs if lvl == "error"]      # an error returns early: the warning never runs
                hit = [m for lvl, m in probs if lvl == "warning" and "no standard-DTFE grids for" in m]
                if [m for m in hit if "no standard-DTFE grids for 099" in m or len(snaps) > 1]:
                    got.add(mode)
                msgs.update(hit)
            return got, errs, msgs

        other = dict(method="ps", prefix="ps_output")    # the PS-DTFE choice for the other sets, on a PS run that exists
        empty, e0, _ = warned()
        two_empty, e3, m3 = warned((50, 99))
        (sd / "ps_output.a_den").touch()
        ps_only, e1, _ = warned()
        ps_other, e4, _ = warned(**other)
        (sd / "output.a_den").touch()
        std, e2, _ = warned()
        std_other, e5, _ = warned(**other)
        two_std, e6, m6 = warned((50, 99))
        check("thesis set: no standard-DTFE grids for 099 is warned in modes all, compute AND export, not in plot / check "
              "(they read the cache)", empty == {"all", "compute", "export"} and not e0, f"{sorted(empty)} {e0}")
        check("... phase-space grids alone do not count (compute and export read the standard DTFE)",
              ps_only == {"all", "compute", "export"} and not e1, f"{sorted(ps_only)} {e1}")
        check("... and a standard-DTFE run silences it in every mode", std == set() and not e2, f"{sorted(std)} {e2}")
        # analyze.py gets no --method / --prefix: a PS estimator and prefix set for the other sets (that run exists
        # here) must not silence it, nor make the standard-DTFE run count as missing
        check("... the estimator and grid prefix set for the other sets change nothing (analyze.py takes neither): "
              "PS-DTFE / ps_output still warned, the standard-DTFE run still silences it",
              ps_other == {"all", "compute", "export"} and std_other == set() and not (e4 + e5),
              f"{sorted(ps_other)} {sorted(std_other)} {e4 + e5}")
        # every snapshot without them is named, and only those: 050 stays empty, 099 gets its run in between
        check("... two snapshots: the warning names every one without them ('050 099'), then only 050 once 099 has its run",
              two_empty == two_std == {"all", "compute", "export"} and not (e3 + e6)
              and len(m3) == 1 and next(iter(m3)).endswith("grids for 050 099")
              and len(m6) == 1 and next(iter(m6)).endswith("grids for 050"),
              f"{sorted(two_empty)} {m3} {sorted(two_std)} {m6} {e3 + e6}")
    finally:
        shutil.rmtree(root)


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
                                   alpha_shape=2.5, periodic=False, ordered_merge=True),
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

    # ---- GPU advice: where the deposit pays (measured 2026-10-01), one warning per condition; asserted for BOTH
    # builds by faking the Makefile's GPU stamp (2026-10-11: on CPU-only CI the GPU branches were never asserted)
    gpu_branches(tmp, snap)
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
    # 2026-10-13: the standard binary answers the same report-mode call (it printed 'AUTO-TUNE-REPORT unsupported'),
    # here on a real (synthetic) snapshot: the fixture above has no velocities
    import subprocess as sp              # (a plain 'subprocess' is a local of this function, imported further down)
    std_in = work / "std_in.hdf5"
    sp.run([sys.executable, str(ROOT / "tests" / "generate_ps_test_data.py"), "--out", str(std_in), "--n", "16"],
           capture_output=True, check=False)
    cs = rs.CustomSpec(input_file=str(std_in), estimator="dtfe", grid=32, output_dir=str(work), output_name="std")
    sst = cs.check_step()
    check("custom check step, standard DTFE: the standard binary's own command in report mode",
          sst.argv[:-1] == cs.command() and sst.argv[-1] == "--auto-tune-report" and Path(sst.argv[0]).name == "DTFE")
    if Path(sst.argv[0]).is_file() and std_in.is_file():
        r = sp.run(sst.argv, env=dict(os.environ, **sst.env, DTFE_RAM_GB="64"), capture_output=True, text=True,
                   cwd=str(ROOT))
        rec = rs.parse_reports(r.stdout).get(0, {})
        check("... which reports a plan the launcher parses (partition, threads, prediction, budget)",
              r.returncode == 0 and rec.get("partition", 0) >= 1 and rec.get("mc", 0) >= 1
              and rec.get("predicted_gb", 0) > 0 and rec.get("budget_gb") == 38.4
              and not list(work.glob("std.*")), r.stdout[-400:])
    q = rs.QueuedJob.of("custom", c)
    check("custom: a queued custom-snapshot job rebuilds and has a summary",
          q.build() == c and q.title.startswith("Custom snapshot · cloud.hdf5 · PS-DTFE 64³"), q.title)

    print("export:")
    txt = rs.export_script(steps, "a test job")
    check("bash export: shebang, cd to the repo, every step, failure count as exit status",
          txt.startswith("#!/usr/bin/env bash") and 'cd "$REPO"' in txt
          and all(rs.shell(st.argv, st.env) in txt for st in steps)
          and 'exit "$failed"' in txt and "#SBATCH" not in txt)
    tee = steps[-1].tee
    check("bash export: a step with a run log tees into '<log>.partial', never straight into '<log>' (the run-log "
          "contract, 2026-10-11)", bool(tee) and f"| tee {rs._q(tee + '.partial')}" in txt
          and f"| tee {rs._q(tee)}\n" not in txt and f"| tee {rs._q(tee)} " not in txt, txt[-600:])
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
    # the run-log contract in an exported script, run for real: a good step's log lands on '<log>' (and a stale
    # '<log>.failed' goes), a failing step's on '<log>.failed' with the old '<log>' untouched; no '.partial' left
    lg = work / "export logs"                      # a space: the quoting is exercised too
    lg.mkdir()
    (lg / "ok.runlog.failed").write_text("an older failed attempt\n")
    (lg / "bad.runlog").write_text("the log of the grids on disk\n")
    ex.write_text(rs.export_script([rs.Step("ok", ["echo", "hello"], {}, tee=str(lg / "ok.runlog")),
                                    rs.Step("bad", ["sh", "-c", "echo oops; exit 3"], {}, tee=str(lg / "bad.runlog"))],
                                   "t"))
    r = subprocess.run(["bash", str(ex)], capture_output=True, text=True)
    check("exported run logs: exit 0 -> '<log>' (stale .failed removed), a failure -> '<log>.failed' (the old log "
          "kept), no '.partial' left, the failure counted",
          r.returncode == 1 and (lg / "ok.runlog").read_text() == "hello\n" and not (lg / "ok.runlog.failed").exists()
          and (lg / "bad.runlog.failed").read_text() == "oops\n"
          and (lg / "bad.runlog").read_text() == "the log of the grids on disk\n"
          and not list(lg.glob("*.partial")), f"{r.returncode} {sorted(p.name for p in lg.iterdir())} {r.stderr}")

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
          and r0.gpu and r0.wall_seconds == 3723 and r0.peak_rss_gb == 47.24 and abs(r0.footprint_gb - 90.51) < 0.01
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
    # 2026-10-13: the thesis presets give a standard-DTFE job its production cell average (the exact one; they set the
    # Monte-Carlo one before), a PS-DTFE job the sampled deposit; 'Quick look' samples both
    sd, sp, ql, cd = (rs.RunSpec(estimator="dtfe", deposit="sampled"), rs.RunSpec(deposit="exact"),
                      rs.RunSpec(estimator="dtfe"), rs.CustomSpec(estimator="dtfe", deposit="sampled"))
    ch = P.apply_preset(sd, P.BUILTIN_PRESETS["Standard"][1], planes)
    P.apply_preset(sp, P.BUILTIN_PRESETS["Standard"][1], planes)
    P.apply_preset(ql, P.BUILTIN_PRESETS["Quick look"][1], planes)
    P.apply_preset(cd, P.BUILTIN_PRESETS["Publication"][1], planes)
    check("preset 'Standard'/'Publication' on standard DTFE: the exact cell average; on PS-DTFE the sampled deposit; "
          "'Quick look' samples both; a preset naming the estimator follows it",
          sd.deposit == "exact" and "deposit" in ch and sp.deposit == "sampled" and ql.deposit == "sampled"
          and cd.deposit == "exact" and "deposit_dtfe" not in sd.to_dict()
          and P.apply_preset(rs.RunSpec(estimator="dtfe"), {"estimator": "ps", "deposit": "sampled",
                                                            "deposit_dtfe": "exact"}) == ["estimator", "deposit"],
          f"{sd.deposit} {sp.deposit} {ql.deposit} {cd.deposit} {ch}")
    check("the standard-DTFE averaging labels name the exact one the default",
          "default" in P.label("exact_dtfe") and "default" not in P.label("sampled_dtfe"))
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
    # 2026-10-13 (integration item 9): the 'caustic' map of a periodic PS run is the EXACT fold flag,
    # dtfelib.io.caustic_is_fold(class mask, streams=, hidden=), not the stored '.caustic' -- which the grids made before
    # 2026-10-13 set only where a folded tetrahedron deposited a sample (21.0% of TNG100-3-Dark z=1 against 33.6%). A
    # non-periodic run (its sidecar) and an output without a streams grid show the stored grid; 'caustic_class' stays raw
    from dtfelib.io import caustic_is_fold
    fd = work / "fold"
    fd.mkdir(exist_ok=True)
    st = np.ones((n, n, n), np.float32)
    st[1, 2, 3] = 3.0                                   # multi-stream cells: a fold
    st[4, :, 5] = 3.0
    st[0, 0, 0] = 0.0                                   # an empty cell
    hid = np.zeros((n, n, n), np.float32)
    hid[2, 2, 3] = 1.0                                  # bit 1: a fold thinner than the sampling
    hid[3, 3, 3] = 2.0                                  # bit 2 alone: mass of an unsampled tetrahedron, no fold
    ccm = np.full((n, n, n), 1 + 4, np.float32)          # a positive tetrahedron with k = 0 everywhere
    ccm[6, 6, 3] = 1 + 2 + 8                             # both parity bits: a fold by the class mask
    raw = np.zeros((n, n, n), np.float32)
    raw[1, 2, 3] = 1.0                                  # the old per-sample flag: one of the fold cells only
    for sfx, arr in (("a_den", den), ("a_streams", st), ("a_hidden_streams", hid), ("caustic", raw),
                     ("causticClass", ccm)):
        (fd / f"fold.{sfx}").write_bytes(arr.tobytes())
    want = caustic_is_fold(ccm, streams=st, hidden=hid)
    of = G.OutputSet(fd / "fold")
    planes_ok = all(np.array_equal(of.slice("caustic", ax, k), np.take(want, k, axis=ax).astype(np.float64))
                    for ax in range(3) for k in (0, 2, 3, 5))
    check("Explore's caustic map of a periodic PS run: the exact fold flag on every axis (streams > 1, hidden bit 1, "
          "the class mask's fold bits; not hidden bit 2 alone or an empty cell), also in value(); the class map stays raw",
          of.fold_exact() and planes_ok and want[1, 2, 3] and want[4, 0, 5] and want[2, 2, 3] and want[6, 6, 3]
          and not want[3, 3, 3] and not want[0, 0, 0] and int(want.sum()) == 1 + n + 1 + 1
          and of.value("caustic", (2, 2, 3)) == 1.0 and of.value("caustic", (3, 3, 3)) == 0.0
          and np.array_equal(of.slice("caustic_class", 2, 3), ccm[:, :, 3].astype(np.float64)),
          f"{of.fold_exact()} {planes_ok} {int(want.sum())}")
    (fd / "fold.gui.json").write_text(json.dumps({"estimator": "ps", "periodic": False, "dim": 3}))
    onp = G.OutputSet(fd / "fold")
    nd = work / "fold_nostreams"
    nd.mkdir(exist_ok=True)
    for sfx, arr in (("a_den", den), ("caustic", raw)):
        (nd / f"fold.{sfx}").write_bytes(arr.tobytes())
    ons = G.OutputSet(nd / "fold")
    check("... a non-periodic run (the union over-flags there) and an output without a streams grid show the stored "
          "'.caustic'", not onp.fold_exact() and np.array_equal(onp.slice("caustic", 2, 3), raw[:, :, 3])
          and not ons.fold_exact() and np.array_equal(ons.slice("caustic", 2, 3), raw[:, :, 3]))
    (fd / "fold.gui.json").unlink()
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
    # built after the 2026-10-11 interlace-pass2 / cic-faces fixes too (cpp_fix_flags_2026_10_11 has the window)
    _write_log(log, "Build: DTFE 3e0c1a65fd-dirty (built 2026-10-12T08:00:00Z)\n"
               "RUNNING: /x/DTFE in.hdf5 r2 --grid 64 --periodic --CIC --interlace\n")
    new = R.stale_reasons(R.parse_runlog(log))
    check("stale outputs: an interlaced grid made before the phase fix is flagged, a newer one is not",
          any(lvl == "error" and "interlaced" in m for lvl, m in old) and not any("interlaced" in m for _, m in new),
          f"{old} | {new}")


def run_outputs_not_inputs():
    """A custom run named after its input (the default) in the input's own folder: 'X.hdf5' beside 'X.a_den'.
    The input is never one of the run's outputs -- the Runs browser's Move to Trash takes the outputs."""
    import json
    import results as R
    d = Path(tempfile.mkdtemp(prefix="gui_runs_"))
    try:
        for name in ("X.hdf5", "X.a_den", "X.den", "X.txt", "Y.txt", "r[1].a_den", "r1.a_den"):
            (d / name).write_bytes(b"\0" * 8)
        (d / "X.gui.json").write_text(json.dumps({"input_file": str(d / "X.hdf5")}))
        (d / "X.runlog").write_text(f"RUNNING: /x/PS-DTFE {d / 'X.hdf5'} X --grid 16\ntotal wall time : 1s\n")
        (d / "Y.runlog").write_text(f"RUNNING: /x/DTFE {d / 'Y.txt'} Y --grid 16\ntotal wall time : 1s\n")
        (d / "r[1].runlog").write_text("RUNNING: /x/DTFE in.hdf5 r[1] --grid 16\ntotal wall time : 1s\n")
        names = lambda log: [p_.name for p_ in R.parse_runlog(d / log).outputs]          # noqa: E731
        check("a run's outputs never include its input snapshot, nor a file its command line reads (Move to Trash)",
              names("X.runlog") == ["X.a_den", "X.den", "X.txt"] and names("Y.runlog") == [],
              f"{names('X.runlog')} {names('Y.runlog')}")
        check("... and a prefix with '[' matches only its own files", names("r[1].runlog") == ["r[1].a_den"],
              str(names("r[1].runlog")))
        sp = d / "My Runs"                    # the log prints paths unquoted: a space must not hide an input
        sp.mkdir()
        for name in ("halo.hdf5", "halo.pts", "halo.a_den"):
            (sp / name).write_bytes(b"\0" * 8)
        (sp / "halo.runlog").write_text(f"RUNNING: /x/PS-DTFE {sp / 'halo.hdf5'} {sp / 'halo'} --sample-points "
                                        f"{sp / 'halo.pts'} --grid 16\ntotal wall time : 1s\n")
        got = [p_.name for p_ in R.parse_runlog(sp / "halo.runlog").outputs]
        check("... nor an input in a folder whose path has a space (the sample-points file of 'My Runs/halo')",
              got == ["halo.a_den"], str(got))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def point_window_linear(tmp: Path):
    """Sample points for Custom and standard-DTFE runs, the PS window, the linear deposit (audit items 14 + the
    binary options, 2026-10-06)."""
    pts = tmp / "pts.txt"
    pts.write_text("1 2 3\n4 5 6\n")
    snap = tmp / "TNG100" / "TNG100-3-Dark" / "snapdir_099" / "combined_099.hdf5"
    c = rs.CustomSpec(input_file=str(snap), sample_points=str(pts), window=[0, 10, 0, 10, 0, 1], linear_deposit=True,
                      volume_weighted=False)
    a = c.command()
    check("a Custom run: --sample-points (+ the velocity gradient), the PS window, the linear deposit",
          a[a.index("--sample-points") + 1] == str(pts) and "--pts-vel-grad" in a and "--ps-linear-deposit" in a
          and a[a.index("--ps-window") + 1:a.index("--ps-window") + 7] == ["0", "10", "0", "10", "0", "1"], str(a))
    errs = lambda spec: [m for lv, m in spec.problems() if lv == "error"]            # noqa: E731
    dt = rs.CustomSpec(input_file=str(snap), estimator="dtfe", sample_points=str(pts), partition=2, window=[0, 1, 0, 1, 0, 1])
    webw = rs.CustomSpec(input_file=str(snap), fields=["density_a", "tweb_a"], window=[0, 1, 0, 1, 0, 1])
    both = rs.CustomSpec(input_file=str(snap), linear_deposit=True, volume_weighted=True)
    bad = rs.CustomSpec(input_file=str(snap), window=[float("nan")])
    notes = [m for lv, m in dt.problems() if lv == "info"]
    check("... refused where the binary would refuse: a window with a web class, linear + volume weighting, an "
          "unreadable window; standard DTFE points with an explicit split are fine since 2026-10-13 (the binary answers "
          "them per partition); a window left from PS-DTFE is ignored by a standard-DTFE run (a note, not a block: its "
          "field is greyed there)",
          not any("triangulation" in m for m in errs(dt)) and not any("window" in m for m in errs(dt))
          and any("ignores it" in m for m in notes)
          and any("whole periodic grid" in m for m in errs(webw)) and any("linear deposit" in m for m in errs(both))
          and any("window needs 6 numbers" in m for m in errs(bad)), f"{errs(dt)} {errs(webw)} {errs(both)} {errs(bad)}")
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99])
    env = rs.RunSpec(estimator="dtfe", slice_plane=str(pts), **base).command()[1]
    lin = rs.RunSpec(linear_deposit=True, volume_weighted=False, **base).command()[1]
    check("the Grids job: a hi-res slice for standard DTFE too (SAMPLE_POINTS), PS_LINEAR_DEPOSIT spelled out",
          env.get("SAMPLE_POINTS") == str(pts) and lin.get("PS_LINEAR_DEPOSIT") == "1"
          and rs.RunSpec(**base).command()[1].get("PS_LINEAR_DEPOSIT") == "0", f"{env} {lin}")
    nodt = [m for lv, m in rs.CustomSpec(input_file=str(snap), estimator="dtfe", fields=["dispersion_a"]).problems()
            if lv == "error"]
    check("a Custom standard-DTFE run with only the dispersion ticked is refused (it would pass a bare --field)",
          any("no field standard DTFE computes" in m for m in nodt), str(nodt))
    sp = rs.RunSpec(estimator="dtfe", slice_plane=str(pts), partition=2, **base)
    check("... a standard-DTFE slice with an explicit partition runs since 2026-10-13 (each partition answers its "
          "share of the points): no error, SAMPLE_POINTS and PARTITION both passed",
          not any(lv == "error" and "triangulation" in m for lv, m in sp.problems())
          and sp.command()[1].get("SAMPLE_POINTS") == str(pts) and sp.command()[1].get("PARTITION") == "2 2 2",
          f"{sp.problems()} {sp.command()[1]}")


def plot_options(tmp: Path):
    """The Plots tab: the thesis scripts as a checklist (= analyze.py's PLOT_SCRIPTS + the synthesis figure),
    the point-eval gradient smoothing, the publication panels' range, dpi and pixel-exact images (2026-10-06)."""
    import ast
    import re
    repo = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo / "python" / "analyze.py").read_text())
    core = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") == "PLOT_SCRIPTS" for t in n.targets))
    deriv = float(re.search(r"^DERIVATIVE_SMOOTH = ([0-9.]+)", (repo / "python" / "dtfelib" / "pointeval.py").read_text(),
                            re.M).group(1))
    check("the thesis checklist is analyze.py's PLOT_SCRIPTS + the synthesis figure; the point-eval default is "
          "dtfelib's", list(rs.THESIS_SCRIPTS) == core + ["plot_conclusions_synthesis.py"]
          and rs.POINTEVAL_DERIV_SMOOTH == deriv, f"{rs.THESIS_SCRIPTS} vs {core}; {deriv}")
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99])
    plane = str(tmp / "TNG100" / "TNG100-3-Dark" / "hires_plane_z64.json")
    spec = rs.PlotSpec(sets=["thesis", "pointeval", "panels"], **base,
                       options={"thesis": {"mode": "plot", "only": "plot_DTFE.py,plot_conclusions_synthesis.py"},
                                "pointeval": {"smooth_deriv": 4.0},
                                "panels": {"plane": plane, "vmin": 0.05, "vmax": 300.0, "dpi": 150, "image": True}})
    argv = {st.label.split()[0]: st.argv for st in spec.steps()}
    a_th = next(st.argv for st in spec.steps() if "analyze.py" in st.label)
    a_pe = next(st.argv for st in spec.steps() if "plot_pointeval.py" in " ".join(st.argv))
    a_pa = next(st.argv for st in spec.steps() if "plot_pointeval_panels.py" in " ".join(st.argv))
    check("... the ticked thesis scripts reach --only (the synthesis figure too); the gradient smoothing reaches "
          "--smooth-derivatives; the panels get --vmin/--vmax/--dpi/--image",
          a_th[a_th.index("--only") + 1:] == ["plot_DTFE.py", "plot_conclusions_synthesis.py"]
          and a_pe[a_pe.index("--smooth-derivatives") + 1] == "4" and a_pa[a_pa.index("--vmin") + 1] == "0.05"
          and a_pa[a_pa.index("--vmax") + 1] == "300" and a_pa[a_pa.index("--dpi") + 1] == "150" and "--image" in a_pa,
          f"{a_th} | {a_pe} | {a_pa}")
    plain = rs.PlotSpec(sets=["pointeval", "panels"], **base, options={"panels": {"plane": plane}})
    flat = " ".join(" ".join(st.argv) for st in plain.steps())
    check("... and the defaults add none of those flags", not any(f in flat for f in ("--smooth-derivatives", "--vmin",
                                                                                     "--dpi", "--image")), flat)


def plot_void_sets(tmp: Path):
    """The Plots tab's void and phase-space sets (2026-10-07): void profiles, streams by web environment, P(k) and the
    caustic skeleton, with the void radius, overlapping voids and the smoothing in Mpc; every flag one the script's own
    parser knows; the thesis set stays last (the app test picks it as the list's last item)."""
    import subprocess as _sp
    repo = Path(__file__).resolve().parents[1]
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[50, 99])
    sets = ["voidpop", "voidprof", "webstreams", "pk", "skeleton"]
    plain = {st.argv[1].split("/")[-1]: st.argv for st in rs.PlotSpec(sets=sets, **base).steps()}
    flat = " ".join(" ".join(a) for a in plain.values())
    check("the void sets: one call each over the selected snapshots, the defaults adding no option flag (the skeleton "
          "its per-void statistics); the thesis set still last",
          len(plain) == 5 and all(a[a.index("--snaps") + 1:a.index("--snaps") + 3] == ["50", "99"]
                                  for k, a in plain.items() if k != "plot_void_population.py")
          and not any(f in flat for f in ("--radius", "--keep-overlaps", "--smooth-mpc", "--web", "--nbins", "--ref-snap",
                                          "--sample"))
          and "--voids" in plain["plot_caustic_skeleton.py"] and rs.FIGURE_SETS[-1].key == "thesis", flat)
    opts = {"voidpop": {"keep_overlaps": True, "smooth_mpc": 2.1625},
            "voidprof": {"radius": "r_eff", "sample": "deep", "keep_overlaps": True, "smooth_mpc": 2.1625},
            "webstreams": {"web": "tweb"}, "pk": {"nbins": 0, "ref": False},
            "skeleton": {"voids": False, "radius": "r_eff"}}
    got = {st.argv[1].split("/")[-1]: st.argv for st in rs.PlotSpec(sets=sets, options=opts, **base).steps()}
    def has(name, *flags):
        a = got[name]
        return all(f in a and (i + 1 == len(flags) or True) for i, f in enumerate(flags))
    pv, pp = got["plot_void_profiles.py"], got["plot_void_population.py"]
    check("... the options reach their flags (radius, deep sample, overlaps, smoothing in Mpc, web, k bins, no same-box "
          "panel, no per-void block)",
          pv[pv.index("--radius") + 1] == "r_eff" and pv[pv.index("--sample") + 1] == "deep" and "--keep-overlaps" in pv
          and pv[pv.index("--smooth-mpc") + 1] == "2.1625" and "--keep-overlaps" in pp and "--smooth-mpc" in pp
          and has("plot_web_streams.py", "--web", "tweb") and got["plot_pk_compare.py"][-2:] in (["--ref-snap", "-1"],)
          and "--nbins" in got["plot_pk_compare.py"] and "--voids" not in got["plot_caustic_skeleton.py"]
          and "--radius" in got["plot_caustic_skeleton.py"], str(got))
    unknown = []
    for name, a in got.items():
        helptext = _sp.run([rs.PYTHON, str(repo / "python" / "plot" / name), "--help"], capture_output=True, text=True,
                           cwd=repo / "python").stdout
        unknown += [f"{name} {f}" for f in a if f.startswith("--") and f not in helptext]
    check("... and every flag is one the script's own parser knows (--help)", not unknown, str(unknown))
    both = rs.PlotSpec(sets=["voidprof"], smooth=5.0, options={"voidprof": {"smooth_mpc": 2.0}}, **base)
    sk = rs.PlotSpec(sets=["skeleton"], **base)
    check("... the smoothing in cells AND in Mpc is an error; the skeleton without caustic grids a warning",
          any(lvl == "error" and "cells and in Mpc" in m for lvl, m in both.problems())
          and any(lvl == "warning" and "causticClass" in m for lvl, m in sk.problems()),
          f"{both.problems()} | {sk.problems()}")


# (set key, option key) -> (script, the script's constant that is its default, or a literal) (2026-10-11)
_VOID_SCRIPT_DEFAULTS = {
    ("voidpop", "size_smooth"): ("plot_void_population", "SIZE_SMOOTH_CELLS"),
    ("voidpop", "size_rv"): ("plot_void_population", "SIZE_FUNCTION_RV"),
    ("voidpop", "fixed_r_smooth"): ("plot_void_population", "FIXED_R_SMOOTH_CELLS"),
    ("voidfate", "ref_snap"): ("plot_void_fate", "REF_SNAP"),
    ("voidconv", "rv_range"): ("plot_void_convergence", "DEFAULT_RV_RANGE"),
    ("skeleton", "null_draws"): ("plot_caustic_skeleton", "NULL_DRAWS"),
}


def plot_void_script_options(tmp: Path):
    """The void scripts' own options on the Plots tab (2026-10-11): population size function, tracking link,
    fate links, convergence factors, environment factor and R_v definition, profile-evolution R_v definition,
    skeleton null and wall cuts. Defaults equal the scripts' own and add no flag; every set value reaches its
    flag, each one the script's parser knows; a malformed list is an error, not a crash in the script."""
    import subprocess as _sp
    repo = Path(__file__).resolve().parents[1]
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[50, 99])
    sets = ["voidpop", "voidtrack", "voidfate", "voidconv", "voidenv", "voidevol", "skeleton"]
    new = {"voidpop": ("size_smooth", "size_rv"), "voidtrack": ("link", "exclusive"),
           "voidfate": ("exclusive", "ref_snap"), "voidconv": ("factors", "match_sigma", "rv_range"),
           "voidenv": ("factor", "r_v_definition"), "voidevol": ("rv_definition",),
           "skeleton": ("null_draws", "min_component", "min_component_mass", "sheet_min_density", "all_caustic")}
    have = {k: {o.key for o in rs.FIGURE_SET[k].opts} for k in sets}
    missing = [f"{k}.{o}" for k, os_ in new.items() for o in os_ if o not in have[k]]
    check("the void scripts' options are Plots-tab options, each with a tooltip and no '--' in its text",
          not missing and all(o.help and "--" not in o.help + o.label for k in sets for o in rs.FIGURE_SET[k].opts
                              if o.key in new[k]), str(missing))
    if missing:
        return
    flags = ("--size-smooth", "--size-rv", "--link", "--exclusive", "--ref-snap", "--factors", "--match-sigma",
             "--rv-range", "--factor", "--r-v-definition", "--rv-definition", "--null-draws", "--min-component",
             "--min-component-mass", "--sheet-min-density", "--all-caustic")
    plain = " ".join(" ".join(st.argv) for st in rs.PlotSpec(sets=sets, **base).steps())
    check("... the defaults add none of their flags", not any(f" {f} " in plain + " " for f in flags), plain)
    # the defaults against the scripts' own: dtfelib's constants imported, the plot scripts' read in a subprocess
    # (they import matplotlib)
    sys.path.insert(0, str(repo / "python"))
    from dtfelib import convergence as cv, classify
    code = ("import sys, json; sys.path[:0] = ['plot', '.']; import importlib; out = {}\n"
            + "".join(f"out['{s}.{c}'] = getattr(importlib.import_module('{s}'), '{c}')\n"
                      for s, c in _VOID_SCRIPT_DEFAULTS.values())
            + "print(json.dumps(out))")
    import json
    r = _sp.run([rs.PYTHON, "-c", code], capture_output=True, text=True, cwd=repo / "python",
                env=dict(os.environ, MPLBACKEND="Agg"))
    try:
        consts = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        consts = {}
    d = {(k, o.key): o.default for k in sets for o in rs.FIGURE_SET[k].opts}
    want = {key: consts.get(f"{s}.{c}") for key, (s, c) in _VOID_SCRIPT_DEFAULTS.items()}
    want[("voidconv", "rv_range")] = " ".join(f"{v:g}" for v in want[("voidconv", "rv_range")] or ())
    want[("voidpop", "fixed_r_smooth")] = " ".join(f"{v:g}" for v in want[("voidpop", "fixed_r_smooth")] or ())
    want.update({("voidconv", "factors"): " ".join(map(str, cv.DEFAULT_FACTORS)),
                 ("voidconv", "match_sigma"): cv.DEFAULT_MATCH_SIGMA, ("voidenv", "factor"): classify.ENV_FACTOR,
                 ("voidtrack", "link"): "trees", ("voidenv", "r_v_definition"): "crossing",
                 ("voidevol", "rv_definition"): "crossing"})
    off = {key: (d.get(key), v) for key, v in want.items() if d.get(key) != v}
    check("... and they equal the scripts' own defaults", consts and not off, f"{off} {r.stderr[-300:]}")
    opts = {"voidpop": {"size_smooth": 0.0, "size_rv": "ridge"},
            "voidtrack": {"link": "particles", "exclusive": True},
            "voidfate": {"exclusive": True, "ref_snap": 50},
            "voidconv": {"factors": "2, 4 8", "match_sigma": 1.5, "rv_range": "2 6"},
            "voidenv": {"factor": 3.0, "r_v_definition": "delta08"}, "voidevol": {"rv_definition": "ridge"},
            "skeleton": {"null_draws": 0, "min_component": 100, "min_component_mass": 0.001,
                         "sheet_min_density": 2.5, "all_caustic": True}}
    got = {st.argv[1].split("/")[-1]: st.argv for st in rs.PlotSpec(sets=sets, options=opts, **base).steps()}

    def after(name, flag, n=1):
        a = got.get(name, [])
        return a[a.index(flag) + 1:a.index(flag) + 1 + n] if flag in a else None
    sk = got.get("plot_caustic_skeleton.py", [])
    check("... the set values reach their flags (size smoothing 0 is a value, not the default)",
          after("plot_void_population.py", "--size-smooth") == ["0"]
          and after("plot_void_population.py", "--size-rv") == ["ridge"]
          and after("plot_void_tracking.py", "--link") == ["particles"] and "--exclusive" in got["plot_void_tracking.py"]
          and after("plot_void_fate.py", "--ref-snap") == ["50"] and "--exclusive" in got["plot_void_fate.py"]
          and after("plot_void_convergence.py", "--factors", 3) == ["2", "4", "8"]
          and after("plot_void_convergence.py", "--match-sigma") == ["1.5"]
          and after("plot_void_convergence.py", "--rv-range", 2) == ["2", "6"]
          and after("plot_void_environment.py", "--factor") == ["3"]
          and after("plot_void_environment.py", "--r-v-definition") == ["delta08"]
          and after("plot_void_profile_evolution.py", "--rv-definition") == ["ridge"]
          and after("plot_caustic_skeleton.py", "--null-draws") == ["0"]
          and after("plot_caustic_skeleton.py", "--min-component") == ["100"]
          and after("plot_caustic_skeleton.py", "--min-component-mass") == ["0.001"]
          and after("plot_caustic_skeleton.py", "--sheet-min-density") == ["2.5"]
          and "--all-caustic" in sk and "--snaps" not in sk, str(got))
    unknown = []
    for name, a in got.items():
        helptext = _sp.run([rs.PYTHON, str(repo / "python" / "plot" / name), "--help"], capture_output=True, text=True,
                           cwd=repo / "python", env=dict(os.environ, MPLBACKEND="Agg")).stdout
        unknown += [f"{name} {f}" for f in a if f.startswith("--") and f not in helptext]
    check("... every flag one the script's own parser knows (--help)", not unknown, str(unknown))
    bad = rs.PlotSpec(sets=["voidconv"], options={"voidconv": {"factors": "2 x", "rv_range": "3"}}, **base)
    allc = rs.PlotSpec(sets=["skeleton"], options={"skeleton": {"all_caustic": True}},
                       **dict(base, snapshots=[]))
    trees = rs.PlotSpec(sets=["voidtrack"], options={"voidtrack": {"exclusive": True}}, **base)
    msgs = " | ".join(m for lv, m in bad.problems() + trees.problems() if lv == "error")
    check("... a malformed factor list or R_v range, one void per particle with the tree link are errors; every "
          "caustic snapshot needs none selected",
          "factors" in msgs and "R_v range" in msgs and "particle link" in msgs
          and not any(lv == "error" and "no snapshots" in m for lv, m in allc.problems()),
          f"{msgs} || {allc.problems()}")


def fixed_r_size_function(tmp: Path):
    """The Void population set's 'size function at fixed R' tick (2026-10-11): a step of its own with the script's
    fixed-R flag and its smoothings, run with DTFE_VOID_GATE=none in ITS environment (the script refuses it gated)
    and shown in the exported command; the population step and the job's other steps keep the caller's gate."""
    import subprocess as _sp
    repo = Path(__file__).resolve().parents[1]
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[50, 99])
    opts_ = {o.key: o for o in rs.FIGURE_SET["voidpop"].opts}
    new = ("size_fixed_r", "fixed_r_smooth", "fixed_r_smooth_mpc")
    check("fixed R: the tick and its smoothings are Void population options on one row, each with a tooltip, no '--'",
          all(k in opts_ for k in new) and len({opts_[k].row for k in new if k in opts_}) == 1
          and all(opts_[k].row and opts_[k].help and "--" not in opts_[k].help + opts_[k].label for k in new
                  if k in opts_), str(sorted(opts_)))
    if not all(k in opts_ for k in new):
        return
    plain = rs.PlotSpec(sets=["voidpop"], method="ps", **base).steps()
    check("... off by default: one population step, no fixed-R flag, no gate in its environment",
          len(plain) == 1 and not any("fixed-r" in a for a in plain[0].argv) and "DTFE_VOID_GATE" not in plain[0].env,
          str([st.argv for st in plain]))
    on = rs.PlotSpec(sets=["voidpop", "voidprof"], method="ps", prefix="ps_x", smooth=0.0,
                     options={"voidpop": {"size_fixed_r": True, "smooth_mpc": 2.1625}}, **base)
    st = on.steps()
    fr = [s for s in st if "--size-fixed-r" in s.argv]
    others = [s for s in st if "--size-fixed-r" not in s.argv]
    a = fr[0].argv if fr else []
    check("... ticked: a step of its own after the population one, the estimator and prefix, the script's default "
          "smoothings (no flag), no population smoothing or snapshots",
          len(fr) == 1 and len(st) == 3 and st[1] is fr[0] and "plot_void_population.py" in a[1]
          and a[2:] == ["--sim", "TNG100-3-Dark", "--method", "ps", "--prefix", "ps_x", "--size-fixed-r"]
          and "--smooth-mpc" in st[0].argv, str(a))
    script = rs.export_script(st, "fixed R")
    check("... run ungated: DTFE_VOID_GATE=none in that step's environment and in its exported command; the population "
          "and the other sets' steps keep no gate of the launcher's",
          bool(fr) and fr[0].env.get("DTFE_VOID_GATE") == "none" and "DTFE_VOID_GATE=none" in fr[0].line()
          and script.count("DTFE_VOID_GATE=none") == 1 and fr[0].env.get("DTFE_SIM") == "TNG100-3-Dark"
          and all("DTFE_VOID_GATE" not in s.env for s in others), script)

    def fr_args(cells, mpc):
        s = rs.PlotSpec(sets=["voidpop"], options={"voidpop": {"size_fixed_r": True, "fixed_r_smooth": cells,
                                                               "fixed_r_smooth_mpc": mpc}}, **base)
        argv = s.steps()[-1].argv
        return argv[argv.index("--size-fixed-r") + 1:], [m for lv, m in s.problems() if lv == "error"]
    tng300, mpc_only, cells_only, dflt = (fr_args("3, 2", "2.1625"), fr_args("", "2.1625"), fr_args("5 3", ""),
                                          fr_args("10 5 3", ""))
    check("... the smoothings reach their flags: cells beside Mpc spelled out (the script takes exactly those given), "
          "Mpc alone, other cells alone; the default adds none",
          tng300 == (["--fixed-r-smooth", "3", "2", "--fixed-r-smooth-mpc", "2.1625"], [])
          and mpc_only == (["--fixed-r-smooth-mpc", "2.1625"], []) and cells_only == (["--fixed-r-smooth", "5", "3"], [])
          and dflt == ([], []), f"{tng300} {mpc_only} {cells_only} {dflt}")
    bad, empty, neg = fr_args("10 x", ""), fr_args("", ""), fr_args("0 5", "")
    check("... a word that is no number, no smoothing at all, or one of 0 is an error before the job runs",
          any("positive numbers" in m for m in bad[1]) and any("needs a smoothing" in m for m in empty[1])
          and any("positive numbers" in m for m in neg[1]), f"{bad[1]} {empty[1]} {neg[1]}")
    helptext = _sp.run([rs.PYTHON, str(repo / "python" / "plot" / "plot_void_population.py"), "--help"],
                       capture_output=True, text=True, cwd=repo / "python", env=dict(os.environ, MPLBACKEND="Agg")).stdout
    flags = {f for f in a + tng300[0] if f.startswith("--")}
    check("... every flag one the script's own parser knows (--help)", flags and all(f in helptext for f in flags),
          str(sorted(f for f in flags if f not in helptext)))


def crossbox_smoothing_constant():
    """G2 (2026-10-11): the cross-box smoothing is python/config.py's CROSSBOX_SMOOTHING_MPC (2.1625 when an older
    config lacks it): a changed config value is the launcher's default and its unchanged-default test."""
    code = ("import sys; sys.path[:0] = ['python', 'python/gui']; import config; "
            "config.CROSSBOX_SMOOTHING_MPC = 3.25; import runspec as rs; "
            "print(rs.CROSSBOX_SMOOTH_MPC, rs.FIGURE_SET['crossbox'].opts[1].default, "
            "'--smooth-mpc' in rs.PlotSpec(sim='X', sets=['crossbox']).steps()[0].argv)")
    r = subprocess.run([rs.PYTHON, "-c", code], capture_output=True, text=True, cwd=ROOT,
                       env=dict(os.environ, DTFE_FIGURES_ROOT=FIGS_TMP))
    check("the cross-box smoothing comes from config.CROSSBOX_SMOOTHING_MPC (default and its no-flag test)",
          r.stdout.split() == ["3.25", "3.25", "False"], r.stdout + r.stderr[-400:])
    code0 = ("import sys; sys.path[:0] = ['python', 'python/gui']; import config\n"
             "if hasattr(config, 'CROSSBOX_SMOOTHING_MPC'): del config.CROSSBOX_SMOOTHING_MPC\n"
             "import runspec as rs; print(rs.CROSSBOX_SMOOTH_MPC)")
    r = subprocess.run([rs.PYTHON, "-c", code0], capture_output=True, text=True, cwd=ROOT,
                       env=dict(os.environ, DTFE_FIGURES_ROOT=FIGS_TMP))
    check("... and 2.1625 when the config has no such constant", r.stdout.strip() == "2.1625", r.stdout + r.stderr[-400:])


def gpu_retry_causes(tmp: Path):
    """G1 (2026-10-11): the GPU deposit retries by cause, from the run log's closing line ('GPU deposit retries: N
    (innocent a, timeout b, oom c, other d)') and from the progress sidecar's 'gpu_retries_by_cause'."""
    import results as R
    d = Path(tempfile.mkdtemp(prefix="gui_retries_"))
    try:
        log = d / "ps_output.runlog"
        log.write_text("RUNNING: /x/PS-DTFE in.hdf5 out --grid 64 --ps-gpu\n  total wall time : 1m 2s\n"
                       "  peak memory (RSS) : 3.5 GB\nGPU deposit retries: 3 (innocent 0, timeout 2, oom 1, other 0)\n")
        rec = R.parse_runlog(log)
        rec2 = R.parse_runlog(log)     # the second read comes from the log cache
        txt = R.retries_text(rec.gpu_retries, rec.gpu_retries_by_cause) if hasattr(R, "retries_text") else ""
        check("a run log's retry line: the total and the causes (the cached read too), the nonzero causes named",
              getattr(rec, "gpu_retries", None) == 3 and not rec.gpu_retried    # no from-scratch restart message
              and getattr(rec, "gpu_retries_by_cause", None) == {"innocent": 0, "timeout": 2, "oom": 1, "other": 0}
              and getattr(rec2, "gpu_retries_by_cause", None) == rec.gpu_retries_by_cause
              and txt == "3 GPU retries (timeout 2, oom 1)", f"{getattr(rec, 'gpu_retries', None)} "
                                                              f"{getattr(rec, 'gpu_retries_by_cause', None)} {txt!r}")
        plain = d / "plain.runlog"
        plain.write_text("RUNNING: /x/PS-DTFE in.hdf5 plain\n  total wall time : 1s\n")
        p = R.parse_runlog(plain)
        side = {"phase": "deposit", "gpu_retries": 2, "gpu_retries_by_cause": {"innocent": 1, "timeout": 0,
                                                                                 "oom": 0, "other": 1}}
        check("... none without the line; a sidecar's causes reach progress_text (its total alone as before)",
              getattr(p, "gpu_retries", None) == 0 and not getattr(p, "gpu_retries_by_cause", None)
              and R.progress_text(side) == "deposit, 2 GPU retries (innocent 1, other 1)"
              and R.progress_text({"phase": "deposit", "gpu_retries": 1}) == "deposit, 1 GPU retries",
              R.progress_text(side))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def explore_extras(tmp: Path):
    """The hi-res slices an output offers, a snapshot chosen by hand for the server, the scalar field, the Custom
    tab's research options (2026-10-06)."""
    import json
    import numpy as np
    import grids as G
    sim = Path(tempfile.mkdtemp(prefix="gui_extras_")) / "SIM"
    sd = sim / "snapdir_050"
    sd.mkdir(parents=True)
    try:
        np.zeros(8 ** 3, dtype=np.float32).tofile(sd / "ps_output.a_den")
        np.ones(8 ** 3, dtype=np.float32).tofile(sd / "ps_output.a_scalar")
        np.zeros(32 * 32, dtype=np.float64).tofile(sd / "ps_output.pts_den")       # one 32x32 plane
        geo = {"nu": 32, "nv": 32, "planes": 1, "supersample": 1, "axis": "z", "u_axis": "x", "v_axis": "y",
               "u0": 0, "u1": 10, "v0": 0, "v1": 10, "center": 5.0, "box": 10.0}
        (sim / "pointeval_plane_32_z.json").write_text(json.dumps(geo))
        (sim / "pointeval_plane_64_z.json").write_text(json.dumps(dict(geo, nu=64, nv=64)))   # another size: not this one
        o = G.OutputSet(sd / "ps_output", box=((0, 0, 0), (10, 10, 10)))
        planes = G.find_pts_planes(o)
        check("an output's hi-res slices: the plane sidecars whose geometry matches its '.pts_den' (not another size)",
              [Path(pth).name for _, pth in planes] == ["pointeval_plane_32_z.json"] and "z-plane" in planes[0][0],
              str(planes))
        check("the per-particle scalar is an Explore field ('.a_scalar')", "scalar" in o.fields(), str(o.fields()))
        snap = sim / "elsewhere.hdf5"
        snap.write_bytes(b"\0")
        kw0, kw1 = G.server_settings(o), G.server_settings(o, str(snap))
        check("a snapshot chosen by hand serves an output whose own is not found (and only an existing file)",
              kw0 is None and kw1 is not None and kw1["snapshot"] == str(snap)
              and G.server_settings(o, str(sim / "nope.hdf5")) is None, f"{kw0} {kw1}")
    finally:
        shutil.rmtree(sim.parent, ignore_errors=True)
    pts = tmp / "pts.txt"
    pts.write_text("1 2 3\n")
    inp = tmp / "TNG100" / "TNG100-3-Dark" / "snapdir_099" / "combined_099.hdf5"
    c = rs.CustomSpec(input_file=str(inp), sample_points=str(pts), stream_density="geometric", per_stream=True,
                      per_stream_ids=True, halo_release=500)
    a = c.command()
    check("the Custom tab's research options: --ps-halo-release, and for the sample points --ps-stream-density, "
          "--per-stream, --per-stream-ids", a[a.index("--ps-halo-release") + 1] == "500"
          and a[a.index("--ps-stream-density") + 1] == "geometric" and "--per-stream" in a and "--per-stream-ids" in a,
          str(a))
    nopts = [m for lv, m in rs.CustomSpec(input_file=str(inp), per_stream=True).problems() if lv == "warning"]
    dt = rs.CustomSpec(input_file=str(inp), estimator="dtfe", sample_points=str(pts), per_stream=True, halo_release=500)
    check("... per-stream without sample points warns; a standard-DTFE run leaves the PS-only ones out",
          any("apply to sample points" in m for m in nopts) and "--per-stream" not in dt.command()
          and "--ps-halo-release" not in dt.command(), f"{nopts} {dt.command()}")


def plane_reads():
    """Explore reads a plane of an uncached grid with plain reads, never a memmap (a mapped page of an unplugged
    drive kills the launcher with SIGBUS; a read raises OSError, which the slice worker reports)."""
    import numpy as np
    import grids as G
    d = Path(tempfile.mkdtemp(prefix="gui_planes_"))
    try:
        n, rng, ok = 20, np.random.default_rng(2), True
        for nc, sfx in ((1, "a_den"), (3, "a_vel")):
            cube = rng.standard_normal((n, n, n, nc) if nc > 1 else (n, n, n)).astype(np.float32)
            cube.tofile(d / f"o.{sfx}")
            for axis in range(3):
                for idx in (0, 9, n - 1):
                    sl = [slice(None)] * 3
                    sl[axis] = idx
                    ok &= np.array_equal(G._read_plane(d / f"o.{sfx}", np.float32, n, nc, axis, idx), cube[tuple(sl)])
        gone = False
        try:
            G._read_plane(d / "missing.a_den", np.float32, n, 1, 0, 0)
        except OSError:
            gone = True
        check("a plane read from the file equals the cube's plane on every axis (1 and 3 components); a vanished "
              "file is an OSError, not a crash", ok and gone)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def cube_cap():
    """While a job runs, Explore's cube cache keeps at most RUN_CUBE_BYTES (the binary budgets from free memory)."""
    import numpy as np
    import grids as G
    d = Path(tempfile.mkdtemp(prefix="gui_cubecap_"))
    old = G.RUN_CUBE_BYTES
    try:
        small, big = d / "s.a_den", d / "b.a_den"
        np.zeros(16 ** 3, dtype=np.float32).tofile(small)        # 16 kB
        np.zeros(32 ** 3, dtype=np.float32).tofile(big)          # 128 kB
        G.drop_cubes()
        G.RUN_CUBE_BYTES = 100_000
        G.cap_cubes(True)
        capped = (G._cached_cube(small, np.float32, (16, 16, 16)) is not None,
                  G._cached_cube(big, np.float32, (32, 32, 32)) is None)
        G.cap_cubes(False)
        free = G._cached_cube(big, np.float32, (32, 32, 32)) is not None
        G._cached_cube(small, np.float32, (16, 16, 16))
        G.cap_cubes(True)                                        # a job starts: the cache trimmed to the cap
        held = G.drop_cubes()
        check("Explore's cube cache: capped while a job runs (a small cube cached, a big one memory-mapped), "
              "the cap trims what is held, lifted after", capped == (True, True) and free and held <= 100_000,
              f"{capped} {free} {held}")
    finally:
        G.cap_cubes(False)
        G.RUN_CUBE_BYTES = old
        G.drop_cubes()
        shutil.rmtree(d, ignore_errors=True)


def audit_claims():
    """Two of the 2026-10-05 audit's confirmed claims: an output name with a dot is listed; time estimates
    compare runs of one precision."""
    import numpy as np
    import results as R
    import grids as G
    d = Path(tempfile.mkdtemp(prefix="gui_claims_"))
    try:
        for name in ("snap_z0.5", "plain", "run.v1"):
            np.zeros(8 ** 3, dtype=np.float32).tofile(d / f"{name}.a_den")
        roots = sorted(o.prefix for _, o in G.find_outputs(d / "no-root", [str(d)]))
        check("an output name with a dot ('snap_z0.5', 'run.v1') is an output of its own, not cut at the first dot",
              roots == ["plain", "run.v1", "snap_z0.5"], str(roots))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    def rec(sec, binary):
        r = R.RunRecord(path=Path(f"/x/{binary}.runlog"), sim="S", prefix="ps_output")
        r.ok, r.wall_seconds, r.binary = True, sec, binary
        r.command = f"/x/{binary} in.hdf5 out --grid 64"
        return r
    recs = [rec(100, "PS-DTFE"), rec(1000, "PS-DTFE-double"), rec(1, "PS-DTFE-2d")]
    kw = dict(sim="S", grid=64, estimator="ps", gpu=False, sliced=False)
    got = (R.time_estimate(recs, **kw), R.time_estimate(recs, precision="double", **kw))
    check("time estimates: a double-precision run (2x slower) never predicts a single one, nor a 2D run a 3D one",
          got == ((100, 1), (1000, 1)), str(got))
    # a run under a second ('total wall time : 0s') still has a time (2026-10-12: it was dropped as if it had none)
    fast = R.time_estimate([rec(0.0, "PS-DTFE"), rec(0.0, "PS-DTFE")], **kw)
    check("... and runs under a second ('0s') still give an estimate", fast == (0.0, 2), str(fast))


def run_options(tmp: Path):
    """The memory budget, the tessellation cache and the standard-DTFE field ticks (audit items 3, 9, 15)."""
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99])
    env0 = rs.RunSpec(estimator="dtfe", **base).command()[1]
    web = rs.RunSpec(estimator="dtfe", fields=["density_a", "tweb_a", "dispersion_a"], mem_budget_gb=40,
                     tess_cache="/private/tmp/dtfe-tess", **base)
    env1 = web.command()[1]
    check("standard DTFE: the script's own field list by default; the ticks (the T-web) when they differ, without the "
          "phase-space dispersion", "FIELDS" not in env0 and env1.get("FIELDS") == "density_a tweb_a", str(env1))
    check("... the memory budget and the tessellation cache reach the run scripts (DTFE_MEM_BUDGET_GB, TESS_CACHE), "
          "only when set", env1.get("DTFE_MEM_BUDGET_GB") == "40" and env1.get("TESS_CACHE") == "/private/tmp/dtfe-tess"
          and "DTFE_MEM_BUDGET_GB" not in env0 and "TESS_CACHE" not in env0, str(env1))
    pe = rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], mem_budget_gb=24, tess_cache="/private/tmp/t").command()[1]
    check("... the pipeline too", pe.get("DTFE_MEM_BUDGET_GB") == "24" and pe.get("TESS_CACHE") == "/private/tmp/t", str(pe))
    icloud = Path.home() / "Library" / "Mobile Documents" / "dtfe-tess"
    probs = rs._run_option_problems(rs.RunSpec(tess_cache=str(icloud), mem_budget_gb=1e6, **base))
    check("an iCloud tessellation cache is an error (the binary refuses it); a budget above the machine's memory a warning",
          any(lv == "error" and "iCloud" in m for lv, m in probs) and any(lv == "warning" and "swap" in m for lv, m in probs),
          str(probs))
    nodtfe = rs.RunSpec(estimator="dtfe", fields=["dispersion_a"], **base).problems()
    check("... standard DTFE with only the dispersion ticked has nothing to compute: an error",
          any(lv == "error" and "no field standard DTFE computes" in m for lv, m in nodtfe), str(nodtfe))
    a = rs.RunSpec(estimator="ps", **base)
    check("the memory check is kept per budget, not per cache folder",
          a.memory_key() != rs.RunSpec(estimator="ps", mem_budget_gb=8, **base).memory_key()
          and a.memory_key() == rs.RunSpec(estimator="ps", tess_cache="/x", **base).memory_key())


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
    # 2026-10-13: the caustic classes are on by default (run_ps_dtfe.sh's default and the production runs'); the GPU
    # since 2026-10-15 when the PS pair was built with one (rs.gpu_build_default, scripts/config.sh's rule)
    gpu = "1" if rs.gpu_build_default("ps") else "0"
    check("defaults: every run option is spelled out for run_ps_dtfe.sh (a stray PS_* in the login shell cannot leak in)",
          [env[k] for k in run_keys] == ["0", "3", gpu, "1", "1", "1", "0", "0", "1", "1"]
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
            (("plain", {}), ("caustics", dict(caustics=False)), ("super", dict(supersample=2)),
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
          rs.PipelineSpec(**base, gpu=True).command()[1]["PS_METAL"] == "1"
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
    lv, msg = levels(rs.PipelineSpec(**base, caustic_cusps=True, caustics=False))
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
          old.deposit == "sampled" and old.gpu is rs.gpu_build_default("ps") and old.planes == 1
          and old.render_project == "plane" and not old.render)


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
    # child), its exact zoom (a direct child writing into a 'dtfe-zoom-' folder), a run from another terminal
    # (bash -> binary: a grandchild), a run whose OUTPUT PATH contains '--serve' (the token rule), a memory check
    # from a terminal (--auto-tune-report, never counted), someone else's query server (listed apart, with its
    # memory) and a Custom-tab run (2026-10-11: a direct child too, with a --ps-window of its own, but a run)
    pgrep_out = "100 PS-DTFE\n200 PS-DTFE\n300 DTFE-double\n400 PS-DTFE\n500 PS-DTFE\n600 PS-DTFE\n700 PS-DTFE\n"
    ps_out = (f"  100 {me} 24000000 /Users/x/Mobile Documents/DTFE/PS-DTFE /Volumes/T7/snap.hdf5 /tmp/s/serve --serve "
              f"--input 105 --verbose 0 --serve-resident 4 --serve-progress\n"
              f"  200 {me} 1500000 /Users/x/Mobile Documents/DTFE/PS-DTFE /Volumes/T7/snap.hdf5 "
              f"/var/folders/ab/T/{rs.ZOOM_DIR_PREFIX}k3_x9q/zoom --ps-window "
              f"1 2 3 4 5 6 --field density_a\n"
              f"  300 4242 18200000 /Users/x/Mobile Documents/DTFE/DTFE-double /Volumes/T7/snap.hdf5 out --grid 256 256 256\n"
              f"  400 4343 900000 ./PS-DTFE snap.hdf5 /tmp/--serve-test/out --periodic\n"
              f"  500 4444 300000 ./PS-DTFE snap.hdf5 /tmp/r/out --grid 512 --auto-tune-report\n"
              f"  600 4545 9300000 ./PS-DTFE other.hdf5 /tmp/s2/serve --serve --input 105\n"
              f"  700 {me} 5100000 /Users/x/Mobile Documents/DTFE/PS-DTFE /Users/x/cloud.hdf5 /Users/x/out/zoom "
              f"--grid 64 --ps-window 0 1 0 1 0 1\n")
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[0] == "pgrep":
            return subprocess.CompletedProcess(argv, 0, stdout=pgrep_out, stderr="")
        if argv[0] == "ps":
            assert argv[-1] == "100,200,300,400,500,600,700" and "-ww" in argv and "rss=" in argv[3], argv
            return subprocess.CompletedProcess(argv, 0, stdout=ps_out, stderr="")
        raise AssertionError(argv)

    orig_run, orig_cache = rs.subprocess.run, rs._JOBS_CACHE
    try:
        rs.subprocess.run = fake_run
        rs._JOBS_CACHE = (-1e9, {"runs": [], "servers": []})
        rs.dtfe_processes(max_age=0)             # one classification; the two views below read the cache
        jobs = rs.running_jobs()
        check("runs: a run from another terminal, one whose path merely contains '--serve' and the launcher's own "
              "Custom-tab run (a direct child, even with --ps-window and an output named 'zoom'); not the launcher's "
              "own server or exact zoom, not a memory check (--auto-tune-report), not another server",
              jobs == [(300, "DTFE-double"), (400, "PS-DTFE"), (700, "PS-DTFE")], str(jobs))
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
                                             (500, "PS-DTFE"), (600, "PS-DTFE"), (700, "PS-DTFE")]
              and rs.running_servers(max_age=0) == [] and "(pid 100)" in rs._busy_problems()[0][1])
        rs._JOBS_CACHE = (-1e9, {"runs": [], "servers": []})
        rs.subprocess.run = lambda argv, **kw: subprocess.CompletedProcess(argv, 1, stdout="", stderr="")
        check("nothing running: no ps call, empty lists, no message",
              rs.running_jobs(max_age=0) == [] and rs.running_servers(max_age=0) == [] and not rs._busy_problems())
    finally:
        rs.subprocess.run, rs._JOBS_CACHE = orig_run, orig_cache
    zoom = f"x /tmp/{rs.ZOOM_DIR_PREFIX}ab12/zoom --grid 8"
    check("the kinds: server by the --serve token, report by --auto-tune-report, own = a child writing into an "
          "exact-zoom folder, else a run (a Custom-tab child included; 2026-10-11)",
          rs._kind(1, "x --serve y", 99) == "server" and rs._kind(1, "x --auto-tune-report", 99) == "report"
          and rs._kind(99, zoom, 99) == "own" and rs._kind(99, "x --grid 8", 99) == "run"
          and rs._kind(1, zoom, 99) == "run" and rs._kind(1, "x --grid 8", 99) == "run"
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


def figures_root_rule():
    """Every figure on the T7 (2026-10-06): DTFE_FIGURES_ROOT, else the T7 while mounted, else python/figures --
    one rule for the launcher and its scripts; saved specs holding the old default read as today's."""
    check("this suite's figures go to its temp folder (DTFE_FIGURES_ROOT, set before runspec is imported)",
          rs.FIGURES_ROOT == Path(FIGS_TMP), str(rs.FIGURES_ROOT))
    env = {k: v for k, v in os.environ.items() if k != "DTFE_FIGURES_ROOT"}
    out = subprocess.run([sys.executable, "-c", "import config, dtfelib.cli as c; print(c.FIGURES_ROOT); "
                          "print(config.LOCAL_FIGURES_ROOT)"], cwd=ROOT / "python", env=env,
                         capture_output=True, text=True).stdout.split("\n")
    t7 = Path("/Volumes/Samsung T7/Illustris TNG")
    want = str(t7 / "figures") if t7.is_dir() else str(ROOT / "python" / "figures")
    check("without DTFE_FIGURES_ROOT: the T7's 'Illustris TNG/figures' while it is mounted (the volume root is root's), "
          "else python/figures -- the same for dtfelib and config", out[:2] == [want, want], str(out[:2]))
    legacy = str(rs.LEGACY_FIGURES_ROOT)
    pipe, plot = rs.PipelineSpec.from_dict({"figures_root": legacy}), rs.PlotSpec.from_dict({"figures_root": legacy})
    own = rs.PlotSpec.from_dict({"figures_root": "/somewhere/else"})
    check("settings and queued jobs saved with the old default (python/figures) load as today's root; a folder of "
          "the user's own stays", pipe.figures_root == plot.figures_root == str(rs.FIGURES_ROOT)
          and own.figures_root == "/somewhere/else", f"{pipe.figures_root} {plot.figures_root} {own.figures_root}")


def void_outline_geometry():
    """The Explore map's void outlines (grids.void_circles): where a sphere cuts the slice, in image pixels (x along
    the first remaining axis, y DOWN the second), its periodic images across the box edges, a zoom window with
    non-square pixels, the wrap along the slice normal, and what is left out."""
    import numpy as np
    import grids as G
    print("void outlines:")
    L = 100.0
    c = np.array([[10.0, 20.0, 53.0],             # 3 Mpc off the slice: a circle of radius 4
                  [98.0, 50.0, 50.0],             # in the slice, straddling x = L: two circles
                  [30.0, 30.0, 97.0],             # 95 Mpc away along z, i.e. 5 across the wrap from z = 2
                  [60.0, 60.0, 50.0]])            # no radius (NaN): left out
    r = np.array([5.0, 6.0, 6.0, np.nan])
    got = G.void_circles(c, r, L, 2, 50.0, [0, 1], (0.0, 0.0), (L, L), 200, 200)
    want = [(20.0, 160.0, 8.0, 8.0, 0), (-4.0, 100.0, 12.0, 12.0, 1), (196.0, 100.0, 12.0, 12.0, 1)]
    check("a sphere 3 Mpc off the slice leaves a circle of sqrt(R^2 - d^2), at (x, L - y) in image pixels; one "
          "straddling the box edge is drawn on both sides; a void without R_v is not",
          np.allclose(sorted(got), sorted(want)), str(got))
    near = G.void_circles(c, r, L, 2, 2.0, [0, 1], (0.0, 0.0), (L, L), 200, 200)
    check("the distance to the slice wraps too (z = 97 is 5 Mpc from z = 2)",
          [k for *_, k in near] == [2] and np.isclose(near[0][2], 2 * np.sqrt(11.0)), str(near))
    check("... unless the box is not periodic",
          G.void_circles(c, r, L, 2, 2.0, [0, 1], (0.0, 0.0), (L, L), 200, 200, periodic=False) == [])
    zoom = G.void_circles(c, r, L, 2, 50.0, [0, 1], (90.0, 40.0), (100.0, 60.0), 100, 100)
    check("a zoom window: only what reaches it, in its own pixels (10 per Mpc across, 5 up: an ellipse)",
          np.allclose(zoom, [(80.0, 50.0, 60.0, 30.0, 1)]), str(zoom))
    xs = G.void_circles(c, r, L, 0, 10.0, [1, 2], (0.0, 0.0), (L, L), 100, 100)
    check("an x slice: the remaining axes y, z", np.allclose(xs, [(20.0, 47.0, 5.0, 5.0, 0)]), str(xs))


def run_log_contract():
    """The run-log contract (2026-10-08): a run writes '<prefix>.runlog.partial'; exit 0 renames it onto
    '<prefix>.runlog' (the log of the grids on disk), a failure or stop onto '<prefix>.runlog.failed' (the old
    '.runlog' untouched). The Runs browser reads the three names as ONE output set: the log of record, and a
    newer failed, running or killed attempt noted on it (results.py). Plus the deposit and GPU-retry fixes of
    the same round, and the Pipeline job's render chained to its pipeline call."""
    import results as R
    print("run-log contract 2026-10-08:")
    base = Path(tempfile.mkdtemp(prefix="gui_runlogs_"))
    child = None
    try:
        check("log_prefix: the three log names give the output prefix; any other name stays as it is",
              [R.log_prefix(n) for n in ("a.runlog", "a.runlog.failed", "a.runlog.partial", "a.a_den", "a.runlog.tmp")]
              == ["a", "a", "a", "a.a_den", "a.runlog.tmp"])
        d = base / "own"
        d.mkdir()
        t0 = time.time() - 3600.0        # every mtime set by hand: 'newer than the log' and the 5 s rule must not ride on the clock
        old = ("Build: PS-DTFE 1111111-dirty (built 2026-10-05T10:00:00Z)\n"
               "RUNNING: /x/PS-DTFE in.hdf5 run --grid 512 --periodic --field density_a --ps-vertex-mass\n"
               "   peak memory (RSS) : 2.0 GB\n   total wall time   : 1m 2s\n")
        new = ("Build: PS-DTFE 2222222-dirty (built 2026-10-07T10:00:00Z)\n"         # the attempt: other settings
               "RUNNING: /x/PS-DTFE in.hdf5 run --grid 256 --periodic --field density_a --ps-vertex-mass\n"
               "terminate called after throwing an instance of 'std::bad_alloc'\n")
        for sfx in ("a_den", "a_vel"):
            (d / f"run.{sfx}").write_bytes(b"\0" * 32)
            os.utime(d / f"run.{sfx}", (t0 - 10, t0 - 10))
        _write_log(d / "run.runlog", old, t0)
        for sfx in (".runlog.failed", ".runlog.partial", ".runlog.tmp"):
            _write_log(d / ("run" + sfx), old, t0 + 30)      # a summary and a wall time in each
        names = [p.name for p in R.parse_runlog(d / "run.runlog").outputs]
        check("a run's outputs leave out every log name: '.runlog.failed', '.runlog.partial' and a script's '.runlog.tmp' "
              "(their suffixes are '.failed', '.partial', '.tmp')", names == ["run.a_den", "run.a_vel"], str(names))
        att = [R.parse_runlog(d / ("run" + sfx)) for sfx in (".runlog.failed", ".runlog.partial")]
        check("... a '.failed' or '.partial' log is never ok, even with a summary and a wall time; its prefix is the output's",
              R.parse_runlog(d / "run.runlog").ok and all(not a.ok and a.wall_seconds == 62 and a.prefix == "run"
                                                          and [p.name for p in a.outputs] == names for a in att),
              str([(a.ok, a.wall_seconds, a.prefix) for a in att]))
        (d / "run.runlog.partial").unlink()
        _write_log(d / "run.runlog.failed", new, t0 + 60)      # a NEWER failed re-run under the same name
        none = base / "no data root"
        recs = R.scan_runs(none, [d])
        r = recs[0] if len(recs) == 1 else None
        reasons = R.stale_reasons(r) if r else []
        check("a '.runlog' and a newer '.runlog.failed' (and a '.runlog.tmp'): ONE record, the log of record with its "
              "own build and settings, ok; the attempt noted on it ('failed', its likely cause, nothing rewritten)",
              r is not None and r.path == d / "run.runlog" and r.ok and r.build_rev == "1111111-dirty" and r.grid == 512
              and r.attempt == "failed" and r.attempt_log == d / "run.runlog.failed" and r.attempt_cause == "Out of memory"
              and r.attempt_rewrote == 0, str(recs))
        check("... the Status column's note: one 'info' line naming the .failed file and the cause (the outputs are this "
              "run's still)", len(reasons) == 1 and reasons[0][0] == "info" and "run.runlog.failed" in reasons[0][1]
              and "Out of memory" in reasons[0][1] and "still this run's" in reasons[0][1], str(reasons))
        os.utime(d / "run.runlog.failed", (t0 - 60, t0 - 60))
        r = R.scan_runs(none, [d])[0]
        check("... a .failed OLDER than the .runlog is no attempt after it: nothing noted", r.attempt == ""
              and r.attempt_log is None and R.stale_reasons(r) == [], f"{r.attempt!r} {R.stale_reasons(r)}")
        os.utime(d / "run.runlog.failed", (t0 + 60, t0 + 60))
        os.utime(d / "run.a_den", (t0 + 10, t0 + 10))         # written by the attempt, after the log of record ended
        r = R.scan_runs(none, [d])[0]
        note = R.attempt_note(r)
        check("an output more than 5 s newer than the .runlog: the attempt rewrote it, a 'warning' to regenerate",
              r.attempt_rewrote == 1 and note is not None and note[0] == "warning" and "regenerate" in note[1]
              and "rewrote 1 of the outputs" in note[1], f"{r.attempt_rewrote} {note}")
        os.utime(d / "run.a_den", (t0 - 10, t0 - 10))
        (d / "run.runlog").unlink()
        recs = R.scan_runs(none, [d])
        r = recs[0] if len(recs) == 1 else None
        note = R.attempt_note(r) if r else None
        check("only a '.runlog.failed' (a first run that failed): it is the record, never ok, its 'warning' says none "
              "has finished", r is not None and r.path == d / "run.runlog.failed" and r.prefix == "run" and not r.ok
              and r.attempt == "failed" and note is not None and note[0] == "warning" and "none has finished" in note[1]
              and note in R.stale_reasons(r), f"{recs} {note}")
        # only '.runlog.partial': killed outright, or still going while a process names its root (one ps call)
        (d / "run.runlog.failed").unlink()
        (d / "run.runlog.tmp").unlink()
        _write_log(d / "run.runlog.partial", new.split("terminate")[0], t0 + 30)
        _write_log(d / "run2.runlog.partial", new.split("terminate")[0].replace(" run ", " run2 "), t0 + 30)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(d / "run2")])
        got = {rec.prefix: rec for rec in R.scan_runs(none, [d])}
        if shutil.which("ps"):
            check("only a '.runlog.partial': no live process names '<dir>/run' (a sibling '<dir>/run2' does not count) -> "
                  "'killed'; the one a live process names -> 'running'; neither ok, each the record",
                  set(got) == {"run", "run2"} and got["run"].attempt == "killed" and got["run2"].attempt == "running"
                  and not got["run"].ok and not got["run2"].ok and got["run"].path == d / "run.runlog.partial"
                  and R.attempt_note(got["run2"])[0] == "info",
                  str({k: (v.attempt, v.ok, v.path.name) for k, v in got.items()}))
        else:
            check("only a '.runlog.partial', without ps: 'partial' (killed or still going), never ok",
                  set(got) == {"run", "run2"} and all(v.attempt == "partial" and not v.ok for v in got.values()),
                  str({k: v.attempt for k, v in got.items()}))
        # a data-root snapdir that is also a custom output folder: one record, with its simulation
        sd = base / "data" / "FAM" / "FAM-1-Dark" / "snapdir_099"
        sd.mkdir(parents=True)
        (sd / "ps_output.a_den").write_bytes(b"\0" * 32)
        _write_log(sd / "ps_output.runlog", old.replace(" run ", f" {sd / 'ps_output'} "), t0)
        _write_log(sd / "ps_output.runlog.partial", new, t0 + 30)
        _write_log(sd / "ps_output.runlog.failed", new, t0 + 60)
        recs = R.scan_runs(base / "data", [sd])
        r = recs[0] if len(recs) == 1 else None
        check("a data-root snapdir also given as an extra folder: one record, the data root's simulation kept, the newer "
              "attempt (the .failed) noted; log_files() lists every log of the set (what Move to Trash takes)",
              r is not None and r.sim == "FAM-1-Dark" and r.snap == 99 and r.path == sd / "ps_output.runlog"
              and r.attempt == "failed" and [p.name for p in r.log_files()]
              == ["ps_output.runlog", "ps_output.runlog.failed", "ps_output.runlog.partial"],
              str([(x.sim, x.path, x.attempt, [p.name for p in x.log_files()]) for x in recs]))
        # the linear deposit in 'Load settings' and the Options column; the PS-DTFE wording of the GPU retry
        _write_log(d / "lin.runlog", "RUNNING: /x/PS-DTFE in.hdf5 lin --grid 64 --ps-linear-deposit\n")
        _write_log(d / "std.runlog", "RUNNING: /x/PS-DTFE in.hdf5 std --grid 64\n")
        lin, std = R.parse_runlog(d / "lin.runlog"), R.parse_runlog(d / "std.runlog")
        check("settings(): a --ps-linear-deposit run turns the box on, a standard-deposit run turns it OFF (the key is "
              "there); the Options column tells the two apart",
              lin.settings().get("linear_deposit") is True and std.settings().get("linear_deposit") is False
              and "linear deposit" in lin.options_text() and "linear deposit" not in std.options_text(),
              f"{lin.settings()} {std.settings()} {lin.options_text()!r}")
        # 2026-10-13: a standard-DTFE run's cell average reaches 'Load settings' both ways (it was left as it was)
        _write_log(d / "dx.runlog", "RUNNING: /x/DTFE in.hdf5 dx --grid 64 --periodic --exact-average --padding 15\n")
        _write_log(d / "dm.runlog", "RUNNING: /x/DTFE in.hdf5 dm --grid 64 --periodic --padding 15\n")
        dx, dm = R.parse_runlog(d / "dx.runlog"), R.parse_runlog(d / "dm.runlog")
        check("settings(): a standard run with --exact-average loads the exact choice, one without the Monte-Carlo one; "
              "the Options column says 'exact average'",
              dx.settings().get("deposit") == "exact" and dm.settings().get("deposit") == "sampled"
              and dx.settings().get("estimator") == "dtfe" and "exact average" in dx.options_text()
              and "exact average" not in dm.options_text() and "exact average" not in lin.options_text(),
              f"{dx.settings()} {dm.settings()} {dx.options_text()!r}")
        _write_log(d / "gpu.runlog", "Build: PS-DTFE 3333333 (built 2026-10-01T09:00:00Z)\n"
                   "RUNNING: /x/PS-DTFE in.hdf5 gpu --grid 64 --periodic --partition 2 2 2 --ps-gpu --ps-vertex-mass\n"
                   "PS-DTFE Metal: Metal command buffer failed: Impacting Interactivity; retrying the partition deposit "
                   "from scratch (attempt 2/3)\n   peak memory (RSS) : 1.0 GB\n   total wall time   : 12s\n")
        g = R.parse_runlog(d / "gpu.runlog")
        check("PS-DTFE's 'retrying the partition deposit from scratch' counts as a GPU retry; built before the DTFE "
              "retry fix, a PS run is still not flagged (that fix is the standard DTFE's)",
              g.gpu_retried and g.estimator == "ps" and g.gpu and R.stale_reasons(g) == [], str(R.stale_reasons(g)))
        steps = rs.PipelineSpec(data_root=str(base / "data"), sims=["FAM-1-Dark", "FAM-2-Dark"], nu=64).steps()
        check("a Pipeline job's render steps wait for its pipeline call: a pipeline that computed nothing exits 1 "
              "(run_ps_pipeline.sh, 2026-10-08) and the job then skips them",
              [(st.label, st.after) for st in steps] == [("run_ps_pipeline.sh FAM-1-Dark FAM-2-Dark", ""),
                                                         ("plot_pointeval.py FAM-1-Dark", steps[0].label),
                                                         ("plot_pointeval.py FAM-2-Dark", steps[0].label)],
              str([(st.label, st.after) for st in steps]))
    finally:
        if child is not None:
            child.kill()
            child.wait()
        shutil.rmtree(base, ignore_errors=True)


def launcher_2026_10_11():
    """The launcher's part of the 2026-10-11 afternoon wave: the ordered merge (K1) from every tab into its run
    and back through 'Load settings', and a killed run's checkpoint directory (K2) in the Runs browser."""
    import results as R
    print("launcher 2026-10-11 (ordered merge, checkpoints):")
    base = Path(tempfile.mkdtemp(prefix="gui_wave2_"))
    try:
        snap = base / "cloud.hdf5"
        on = rs.RunSpec(sim="TNG50-4", snapshots=[99], ordered_merge=True).command()[1]
        off = rs.RunSpec(sim="TNG50-4", snapshots=[99], ordered_merge=False).command()[1]
        pon = rs.PipelineSpec(sims=["TNG50-4"], ordered_merge=True)
        # 2026-10-14: ON by default (the staged merge in index order: run_ps_dtfe.sh's and the binary's default); off
        # is the old merge as partitions finish (ORDERED_MERGE=0 / --completion-order-merge)
        check("ordered merge: Grids and Pipeline jobs set ORDERED_MERGE for run_ps_dtfe.sh (spelled out when off, so a "
              "stray export cannot leak in), the Pipeline's memory check too; the Custom tab passes the flag (PS only); "
              "on by default since 2026-10-14, off = --completion-order-merge",
              on.get("ORDERED_MERGE") == "1" and off.get("ORDERED_MERGE") == "0"
              and rs.RunSpec(sim="TNG50-4", snapshots=[99]).command()[1].get("ORDERED_MERGE") == "1"
              and pon.command()[1].get("ORDERED_MERGE") == "1"
              and pon.check_step("TNG50-4", [99], 1024).env.get("ORDERED_MERGE") == "1"
              and rs.PipelineSpec(sims=["TNG50-4"]).command()[1].get("ORDERED_MERGE") == "1"
              and rs.PipelineSpec(sims=["TNG50-4"], ordered_merge=False).command()[1].get("ORDERED_MERGE") == "0"
              and "--ordered-merge" in rs.CustomSpec(input_file=str(snap), ordered_merge=True).command()
              and "--ordered-merge" in rs.CustomSpec(input_file=str(snap)).command()
              and "--completion-order-merge" in rs.CustomSpec(input_file=str(snap), ordered_merge=False).command()
              and "--ordered-merge" not in rs.CustomSpec(input_file=str(snap), ordered_merge=False).command()
              and not {"--ordered-merge", "--completion-order-merge"} & set(
                  rs.CustomSpec(input_file=str(snap), estimator="dtfe", ordered_merge=True).command()), str(on))
        d = base / "runs"
        d.mkdir()
        _write_log(d / "om.runlog", "RUNNING: /x/PS-DTFE in.hdf5 om --grid 64 --partition 2 2 2 --ordered-merge\n"
                   "   total wall time   : 12s\n")
        _write_log(d / "plain.runlog", "RUNNING: /x/PS-DTFE in.hdf5 plain --grid 64 --partition 2 2 2\n"
                   "   total wall time   : 12s\n")
        _write_log(d / "co.runlog", "RUNNING: /x/PS-DTFE in.hdf5 co --grid 64 --partition 2 2 2 --completion-order-merge\n"
                   "   total wall time   : 12s\n")
        om, plain = R.parse_runlog(d / "om.runlog"), R.parse_runlog(d / "plain.runlog")
        co = R.parse_runlog(d / "co.runlog")
        # 2026-10-14: a run without either flag loads today's default (on); only '--completion-order-merge' turns it off
        check("ordered merge: settings() carries it (ON for a run without a flag, today's default; OFF for a "
              "--completion-order-merge run), the Options column says which, and the settings load into a RunSpec",
              om.settings().get("ordered_merge") is True and plain.settings().get("ordered_merge") is True
              and co.settings().get("ordered_merge") is False
              and "ordered merge" in om.options_text() and "ordered merge" not in plain.options_text()
              and "completion-order merge" in co.options_text()
              and all(hasattr(rs.RunSpec(), k) for k in om.settings()), f"{om.settings()} {om.options_text()!r}")
        import presets as P
        check("ordered merge: the tick's label has no '--', and names the option it stands for",
              "--" not in P.label("ordered_merge") and P.OPTION_TEXT["ordered_merge"][1] == "--ordered-merge")

        # K2: a killed partitioned run leaves '<prefix>.runlog.checkpoint/' (a directory); it is a run file, never an
        # output or a log, and with no live process the run is 'resumable' -- when run_ps_dtfe.sh ran it (a Grids or
        # Pipeline job: under the data root's snapdir_*, 2026-10-11), and its attempt did not fail on a refused resume
        t0 = time.time() - 600.0
        sd = base / "root" / "TNGK-1-Dark" / "snapdir_099"           # where run_ps_dtfe.sh writes
        sd.mkdir(parents=True)
        for sfx in ("a_den", "a_vel"):
            (sd / f"ck.{sfx}").write_bytes(b"\0" * 32)
        _write_log(sd / "ck.runlog.partial", "RUNNING: /x/PS-DTFE in.hdf5 ck --grid 64 --partition 2 2 2\n"
                   "PS-DTFE: checkpoints of the merged grids go to 'ck.runlog.checkpoint' (at most ~5% of the run; "
                   "removed when the outputs are written; '--resume' continues a killed run from it).\n"
                   "partition 3/8 done\n", t0)
        ckd = sd / "ck.runlog.checkpoint"
        ckd.mkdir()
        (ckd / "manifest.json").write_text("{}")
        (sd / "ck.runlog.checkpoint.tmp").write_text("")      # a checkpoint being written: a run file too
        _write_log(sd / "ck2.runlog", "RUNNING: /x/PS-DTFE in.hdf5 ck2 --grid 64\n   total wall time   : 1s\n", t0 - 60)
        _write_log(sd / "ck2.runlog.failed", "RUNNING: /x/PS-DTFE in.hdf5 ck2 --grid 64 --partition 2 2 2\n"
                   "~~~ ERROR ~~~\n", t0)
        (sd / "ck2.runlog.checkpoint").mkdir()
        # the resume refused (the settings changed since the kill): the same job fails the same way every time
        _write_log(sd / "rf.runlog", "RUNNING: /x/PS-DTFE in.hdf5 rf --grid 64\n   total wall time   : 1s\n", t0 - 60)
        _write_log(sd / "rf.runlog.failed", "RUNNING: /x/PS-DTFE in.hdf5 rf --grid 128 --partition 2 2 2 --resume\n"
                   "~~~ ERROR ~~~ '--resume' refused: the checkpoint 'rf.runlog.checkpoint' was written by a run with "
                   "other settings.\n", t0)
        (sd / "rf.runlog.checkpoint").mkdir()
        _write_log(sd / "rf0.runlog.failed", "RUNNING: /x/PS-DTFE in.hdf5 rf0 --grid 64 --partition 2 2 2 --resume\n"
                   "~~~ ERROR ~~~ '--resume': nothing to resume -- no manifest. Run without '--resume' to start the "
                   "run.\n", t0)
        (sd / "rf0.runlog.checkpoint").mkdir()
        # a Custom run (its own output folder): it never resumes, a checkpoint or not
        _write_log(d / "cu.runlog.failed", "RUNNING: /x/PS-DTFE in.hdf5 cu --grid 64 --partition 2 2 2\n"
                   "~~~ ERROR ~~~\n", t0)
        (d / "cu.runlog.checkpoint").mkdir()
        _write_log(d / "gone.runlog.partial", "RUNNING: /x/PS-DTFE in.hdf5 gone --grid 64\n", t0)
        # 2026-10-13: a killed partitioned STANDARD run (run_dtfe.sh resumes it too, DTFE_RESUME) is 'resumable' alike
        _write_log(sd / "output.runlog.partial", "RUNNING: /x/DTFE in.hdf5 output --grid 64 --partition 2 2 2\n"
                   "DTFE: checkpoints of the merged grids go to 'output.runlog.checkpoint' (at most ~5% of the run; "
                   "removed when the outputs are written; '--resume' continues a killed run from it).\n", t0)
        (sd / "output.runlog.checkpoint").mkdir()
        check("a checkpoint directory and its files are run files (never an output, never a log)",
              R.is_run_file("ck.runlog.checkpoint") and R.is_run_file("ck.runlog.checkpoint.tmp")
              and not R.is_run_file("ck.runlog") and not R.is_run_file("ck.a_den"))
        got = {r.prefix: r for r in R.scan_runs(base / "root", [d])}
        ck, ck2, gone, sk = got.get("ck"), got.get("ck2"), got.get("gone"), got.get("output")
        rf, rf0, cu = got.get("rf"), got.get("rf0"), got.get("cu")
        if shutil.which("ps"):
            check("a killed run with its checkpoint is 'resumable' (a failed attempt after a finished log too); one "
                  "killed without a checkpoint stays 'killed'; the checkpoint is not among the outputs",
                  ck is not None and ck.attempt == "resumable" and not ck.ok
                  and [p.name for p in ck.outputs] == ["ck.a_den", "ck.a_vel"]
                  and ck2 is not None and ck2.ok and ck2.attempt == "resumable"
                  and gone is not None and gone.attempt == "killed",
                  str({k: (v.attempt, [p.name for p in v.outputs]) for k, v in got.items()}))
            check("... a killed standard-DTFE run with its checkpoint too (run_dtfe.sh resumes since 2026-10-13)",
                  sk is not None and sk.estimator == "dtfe" and sk.attempt == "resumable",
                  str(None if sk is None else (sk.estimator, sk.attempt)))
            n1, n2 = R.attempt_note(ck), R.attempt_note(ck2)
            check("... and its note says the same job resumes it, and names the start-over tick",
                  n1 is not None and "resum" in n1[1] and n2 is not None and "resum" in n2[1]
                  and all("Start over" in m and "PS_RESUME=0" in m for _, m in (n1, n2))
                  and not [m for _, m in (n1, n2) if "--" in m], f"{n1} {n2}")
        else:
            check("without ps a '.runlog.partial' stays 'partial' (its process cannot be ruled out), checkpoint or not",
                  ck is not None and ck.attempt == "partial")
        # 2026-10-11: a refused resume is never 'resumable' (rerunning fails again), nor is a Custom run's checkpoint
        nr, nr0 = (R.attempt_note(r) if r is not None else None for r in (rf, rf0))
        check("a resume the binary refused (other settings; nothing to resume) stays 'failed', its note and cause name "
              "'Start over' (PS_RESUME=0, no '--'); a Custom run's checkpoint is never 'resumable'",
              rf is not None and rf.attempt == "failed" and getattr(rf, "attempt_resume_refused", False) and rf.ok
              and rf0 is not None and rf0.attempt == "failed" and getattr(rf0, "attempt_resume_refused", False)
              and "Resume refused" in rf.attempt_cause
              and all(n is not None and n[0] == "warning" and "Start over" in n[1] and "PS_RESUME=0" in n[1]
                      and "--" not in n[1] for n in (nr, nr0))
              and cu is not None and cu.attempt == "failed" and not getattr(cu, "attempt_resume_refused", False),
              str({k: (v.attempt, getattr(v, "attempt_resume_refused", None), v.attempt_cause) for k, v in got.items()}) + f" {nr} {nr0}")
        dg = R.diagnose("~~~ ERROR ~~~ '--resume' refused: the checkpoint 'x' was written by a run with other settings.")
        check("... diagnose() offers the start-over tick for it; the binary's info lines about '--resume' are no refusal",
              [d_.action for d_ in dg] == ["start_over"] and "--" not in dg[0].advice
              and hasattr(R, "RESUME_REFUSED") and not R.RESUME_REFUSED.search("this run starts over and replaces it ('--resume' continues it).")
              and not R.RESUME_REFUSED.search("removed when the outputs are written; '--resume' continues a killed "
                                              "run from it)"), str(dg))
        # the start-over tick: PS_RESUME=0 for run_ps_dtfe.sh, spelled out as 1 otherwise (a stray export cannot leak in)
        check("start over: Grids and Pipeline jobs set PS_RESUME=0 (standard Grids jobs DTFE_RESUME=0) when ticked, 1 "
              "otherwise; the memory check ignores it",
              rs.RunSpec(sim="TNG50-4", snapshots=[99], start_over=True).command()[1].get("PS_RESUME") == "0"
              and off.get("PS_RESUME") == "1"
              and rs.PipelineSpec(sims=["TNG50-4"], start_over=True).command()[1].get("PS_RESUME") == "0"
              and rs.PipelineSpec(sims=["TNG50-4"]).command()[1].get("PS_RESUME") == "1"
              and rs.RunSpec(estimator="dtfe", sim="TNG50-4", snapshots=[99], start_over=True).command()[1].get("DTFE_RESUME") == "0"
              and rs.RunSpec(estimator="dtfe", sim="TNG50-4", snapshots=[99]).command()[1].get("DTFE_RESUME") == "1"
              and rs.RunSpec(start_over=True).memory_key() == rs.RunSpec().memory_key()
              and rs.PipelineSpec(start_over=True).memory_key() == rs.PipelineSpec().memory_key()
              and "start_over" not in om.settings(), str(off))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def cpp_fix_flags_2026_10_11():
    """results.KNOWN_FIXES for the cpp-bug fixes of 2026-10-11: an interlaced grid without a region (pass 2 gridded
    the same cells), PCS (no padding, half-cell shift), CIC (extrapolated face and seam cells), and a fractional
    region or padding on a box whose axes differ (the [i%2] axis indexing). Grids built between the interlace-phase
    fix (2026-10-02) and 2026-10-11T23:00 are flagged at their level, the same commands built after it are not."""
    import results as R
    print("stale outputs: the cpp-bug fixes of 2026-10-11:")
    work = Path(tempfile.mkdtemp(prefix="gui_cppfix_"))
    try:
        log = work / "r.runlog"

        def reasons(built, args):
            _write_log(log, f"Build: DTFE 3e0c1a65fd-dirty (built {built})\nRUNNING: /x/DTFE in.hdf5 r --grid 64 {args}\n")
            return R.stale_reasons(R.parse_runlog(log))

        before, after = "2026-10-05T08:00:00Z", "2026-10-12T08:00:00Z"
        has = lambda got, lvl, word: any(lv == lvl and word in m for lv, m in got)   # noqa: E731
        cases = (   # (arguments, the (level, word) the old build must show, words no message may hold)
            ("--periodic --TSC --interlace", [("error", "second pass")], ["PCS", "CIC"]),
            ("--periodic --TSC --interlace --regionMpc 0 50 0 50 0 50", [], ["second pass"]),
            ("--periodic --TSC --interlace --region 0 0.5 0 0.5 0 0.5", [], ["second pass"]),
            ("--periodic --PCS", [("error", "PCS")], ["CIC", "second pass"]),
            ("--periodic --CIC", [("warning", "CIC")], ["PCS"]),
            ("--box 0 100 0 100 0 200 --region 0 1 0 1 0.5 1", [("error", "fractional region")], ["padding"]),
            ("--box 10 110 10 110 20 120 --region 0 1 0 1 0.5 1", [("error", "fractional region")], []),
            ("--box 0 100 0 100 0 100 --region 0 1 0 1 0.5 1 --padding 0.05", [], ["fractional"]),
            ("--box 0 100 0 100 0 200 --padding 0.05", [("warning", "fractional padding")], ["fractional region"]),
            ("--box 10 110 10 110 20 120 --padding 0.05", [], ["fractional"]),
            ("--box 0 100 0 100 0 200 --regionMpc 0 100 0 100 100 200 --paddingMpc 5 5 5 5 10 10", [], ["fractional"]),
        )
        bad = []
        for args, want, never in cases:
            old, new = reasons(before, args), reasons(after, args)
            gone = [w for _, w in want] + never          # after the fix: none of the messages
            if not all(has(old, lvl, w) for lvl, w in want) or any(w in m for _, m in old for w in never) \
                    or any(w in m for _, m in new for w in gone):
                bad.append(f"{args}: old {old} | new {new}")
        check("KNOWN_FIXES: grids built before 2026-10-11T23:00 are flagged for interlace without a region (error), "
              "PCS (error), CIC (warning), a fractional region (error) or padding (warning) on a box whose axes "
              "differ; the same commands built after it, and the unaffected ones, are not", not bad, " || ".join(bad))
        check("the new KNOWN_FIXES messages are launcher text (no '--')",
              not [f.key for f in R.KNOWN_FIXES if "--" in f.message], "")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def pts_conventions_flag_2026_10_13():
    """results.KNOWN_FIXES 'pts-grid-conventions' (the parity batch of 2026-10-13): a PS image plane built before the
    points followed the grids' --ps-vertex-mass / --ps-volume-weighted is flagged (warning); the same command built
    after it is not, nor a plane without the two grid options, one asked for '--pts-conventions legacy', a PS run
    without a plane, or a standard plane."""
    import results as R
    print("stale outputs: PS image planes in the grids' conventions (2026-10-13):")
    work = Path(tempfile.mkdtemp(prefix="gui_ptsconv_"))
    try:
        log = work / "r.runlog"

        def flagged(exe, built, args):
            _write_log(log, f"Build: {exe} 3e0c1a65fd-dirty (built {built})\nRUNNING: /x/{exe} in.hdf5 r --grid 64 {args}\n")
            return any(lv == "warning" and "image plane made before" in m for lv, m in R.stale_reasons(R.parse_runlog(log)))

        before, after = "2026-10-13T12:56:00Z", "2026-10-14T08:00:00Z"
        plane = "--periodic --sample-points p.bin --pts-vel-grad"
        cases = (   # (binary, arguments, flagged when built before, after)
            ("PS-DTFE", f"{plane} --ps-vertex-mass --ps-volume-weighted", True, False),
            ("PS-DTFE", f"{plane} --ps-volume-weighted", True, False),
            ("PS-DTFE", f"{plane} --ps-vertex-mass --ps-volume-weighted --pts-conventions legacy", False, False),
            ("PS-DTFE", plane, False, False),
            ("PS-DTFE", "--periodic --ps-vertex-mass --ps-volume-weighted", False, False),
            ("DTFE", plane, False, False),
        )
        bad = [f"{exe} {args}" for exe, args, b, a in cases
               if flagged(exe, before, args) != b or flagged(exe, after, args) != a]
        check("KNOWN_FIXES pts-grid-conventions: a PS plane with the grids' vertex masses or volume weighting built "
              "before 2026-10-13T20:00 is flagged, the same built after it, a legacy or option-less plane, a PS run "
              "without a plane and a standard plane are not", not bad, " || ".join(bad))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def pts_conventions_choice_2026_10_15():
    """The plane conventions' choice (PS-DTFE '--pts-conventions grid|legacy', 2026-10-15) where the launcher makes a
    phase-space image plane: a Grids job with a slice and a Pipeline job set PTS_CONVENTIONS for run_ps_dtfe.sh /
    run_ps_pipeline.sh, spelled out both ways; a Custom run passes '--pts-conventions legacy' to the binary only for
    legacy (grid is its default, the run log as before); standard DTFE and a run without points get nothing. Legacy
    is said (info, no '--'), an unknown value refused; the memory check ignores it; 'Load settings' of a run reads it
    back from the run log, the Runs tab names it, and a saved job without it opens at the grids' conventions."""
    import results as R
    print("the plane conventions' choice (2026-10-15):")
    work = Path(tempfile.mkdtemp(prefix="gui_ptschoice_"))
    try:
        plane = work / "hires_plane_z64.bin"
        plane.write_bytes(b"\0" * 24 * 64)
        snap = work / "snap.hdf5"
        snap.write_bytes(b"")
        g = lambda **kw: rs.RunSpec(sim="TNG50-4", snapshots=[99], **kw).command()[1]        # noqa: E731
        c = lambda **kw: rs.CustomSpec(input_file=str(snap), **kw).command()                # noqa: E731
        check("Grids: PTS_CONVENTIONS with a PS slice (grid by default, legacy when chosen); none without a slice or "
              "for standard DTFE; Pipeline: always (its plane), both ways",
              g(slice_plane=str(plane)).get("PTS_CONVENTIONS") == "grid"
              and g(slice_plane=str(plane), pts_conventions="legacy").get("PTS_CONVENTIONS") == "legacy"
              and "PTS_CONVENTIONS" not in g() and "PTS_CONVENTIONS" not in g(estimator="dtfe", slice_plane=str(plane))
              and rs.PipelineSpec(sims=["TNG50-4"]).command()[1].get("PTS_CONVENTIONS") == "grid"
              and rs.PipelineSpec(sims=["TNG50-4"], pts_conventions="legacy").command()[1].get("PTS_CONVENTIONS") == "legacy"
              and rs.PipelineSpec(sims=["TNG50-4"], pts_conventions="legacy").check_step("TNG50-4", [99], 1024).env.get(
                  "PTS_CONVENTIONS") == "legacy")
        leg = c(sample_points=str(plane), pts_conventions="legacy")
        check("Custom: '--pts-conventions legacy' with points for legacy only (grid = the binary's default, nothing "
              "added); not without points nor for standard DTFE",
              "--pts-conventions" in leg and leg[leg.index("--pts-conventions") + 1] == "legacy"
              and "--pts-conventions" not in c(sample_points=str(plane))
              and "--pts-conventions" not in c(pts_conventions="legacy")
              and "--pts-conventions" not in c(sample_points=str(plane), pts_conventions="legacy", estimator="dtfe"))
        msgs = lambda spec: [(lv, m) for lv, m in spec.problems() if "conventions" in m]     # noqa: E731
        rl = msgs(rs.RunSpec(sim="TNG50-4", snapshots=[99], slice_plane=str(plane), pts_conventions="legacy"))
        pl = msgs(rs.PipelineSpec(sims=["TNG50-4"], pts_conventions="legacy"))
        cl = msgs(rs.CustomSpec(input_file=str(snap), sample_points=str(plane), pts_conventions="legacy"))
        bad = msgs(rs.CustomSpec(input_file=str(snap), sample_points=str(plane), pts_conventions="old"))
        plain = msgs(rs.RunSpec(sim="TNG50-4", snapshots=[99], slice_plane=str(plane)))
        check("legacy is said on every tab (info: the estimators before 13 October 2026), an unknown value is refused, "
              "the grids' conventions say nothing; no '--' in the messages",
              [lv for lv, _ in rl + pl + cl] == ["info"] * 3 and all("13 October 2026" in m for _, m in rl + pl + cl)
              and [lv for lv, _ in bad] == ["error"] and not plain and not any("--" in m for _, m in rl + pl + cl + bad),
              str(rl + pl + cl + bad))
        same = (rs.RunSpec(slice_plane=str(plane)).memory_key() == rs.RunSpec(slice_plane=str(plane),
                                                                             pts_conventions="legacy").memory_key()
                and rs.PipelineSpec().memory_key() == rs.PipelineSpec(pts_conventions="legacy").memory_key()
                and rs.CustomSpec().memory_key() == rs.CustomSpec(pts_conventions="legacy").memory_key())
        old = {k: v for k, v in rs.RunSpec().to_dict().items() if k != "pts_conventions"}
        check("the memory check ignores it; a job saved before the choice opens at the grids' conventions",
              same and rs.RunSpec.from_dict(old).pts_conventions == "grid"
              and rs.PipelineSpec.from_dict({}).pts_conventions == "grid")
        log = work / "ps_output.runlog"
        recs = {}
        for name, extra in (("legacy", " --pts-conventions legacy"), ("grid", ""), ("noplane", None)):
            args = "--grid 64 --ps-vertex-mass --ps-volume-weighted" + (
                "" if extra is None else f" --sample-points {plane} --pts-vel-grad{extra}")
            _write_log(log, f"RUNNING: /x/PS-DTFE in.hdf5 {work}/ps_output {args}\n")
            recs[name] = R.parse_runlog(log)
        st = {k: r.settings().get("pts_conventions") for k, r in recs.items()}
        check("Load settings: a run's plane conventions from its run log (legacy only when it says so; none without a "
              "plane), and the Runs tab names a legacy plane",
              st == {"legacy": "legacy", "grid": "grid", "noplane": None}
              and rs.RunSpec.from_dict(dict(rs.RunSpec().to_dict(), **recs["legacy"].settings())).pts_conventions == "legacy"
              and "legacy conventions" in recs["legacy"].options_text() and "legacy" not in recs["grid"].options_text()
              and "--" not in recs["legacy"].options_text(), f"{st} {recs['legacy'].options_text()}")
        import presets as P
        check("the menu's label and texts (the binary's own help behind the tooltip): no '--'",
              P.label("pts_conventions") and P.OPTION_TEXT["pts_conventions"][1] == "--pts-conventions"
              and set(rs.PTS_CONVENTIONS_TEXT) == set(rs.PTS_CONVENTIONS) == {"grid", "legacy"}
              and not any("--" in t for t in list(rs.PTS_CONVENTIONS_TEXT.values()) + [P.label("pts_conventions")]))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
