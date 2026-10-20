"""MainWindow's memory check (CheckMixin, moved from app.py): asking the binary's own auto-tuner what the
Grids, Custom snapshot or Pipeline settings need per snapshot (one process per pass; a pipeline snapshot
over budget is tried again at half the plane size), cancelling it, and the check-list lines its results
give. The names app.py defines are read through _app, imported last (see app_tabs.py)."""

from __future__ import annotations

import os
import signal
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment

import runspec as rs
import grids as G


class CheckMixin:
    # ================================================================ memory check
    def check_memory(self):
        """Ask the binary's own auto-tuner (run_ps_dtfe.sh AUTO_TUNE_REPORT=1, or the binary's
        --auto-tune-report for a custom snapshot) what the current Grids, Custom snapshot or Pipeline
        settings need per snapshot. For the pipeline, a snapshot over budget is
        checked again at half the plane size, down to 1024^2 -- which is both the suggestion shown
        and, with 'Adaptive', the plan the pipeline then runs."""
        if self._mcheck is not None:            # the button reads "Cancel check"
            self._cancel_check()
            return
        self._read_widgets()
        tab = self.tabs.currentIndex()
        if tab == _app.GRIDS:
            if not self.spec.snapshots:         # either estimator: run_ps_dtfe.sh or run_dtfe.sh AUTO_TUNE_REPORT=1
                return
            key = self.spec.memory_key()
            passes = [(self.spec.sim, None, sorted(self.spec.snapshots))]
        elif tab == _app.OWN:
            if not self._custom_checkable():
                return
            key = self.custom.memory_key()
            passes = [(Path(self.custom.input_file).name, None, [0])]    # the bare report line is snapshot 0
        elif tab == _app.PIPELINE:
            key = self.pipe.memory_key()
            passes = []
            for sim in self.pipe.sims:
                if rs.sim_dir(sim, self.pipe.data_root).is_dir():
                    todo = self.pipe.stale(sim, self.pipe.nu)[0]
                    if todo:
                        passes.append((sim, self.pipe.nu, todo))
            if not passes:
                self.status.setText("Memory check: nothing to compute (every snapshot is up to date)")
                return
        else:
            return
        # the settings as they are NOW: a change during the check must not mix into its later passes
        spec = (rs.RunSpec.from_dict(self.spec.to_dict()) if tab == _app.GRIDS
                else rs.CustomSpec.from_dict(self.custom.to_dict()) if tab == _app.OWN
                else rs.PipelineSpec.from_dict(self.pipe.to_dict()))
        self._mcheck = {"tab": tab, "key": key, "passes": passes, "done": 0, "spec": spec,
                        "first": {sim: list(snaps) for sim, _, snaps in passes}}
        self.mem_results.setdefault(key, {})
        G.drop_cubes()                          # the report measures free memory: not Explore's cached cubes
        self._next_check()
        self.refresh()

    def _next_check(self):
        mc = self._mcheck
        if mc is None:
            return
        if not mc["passes"]:
            self._finish_check()
            return
        sim, nu, snaps = mc["passes"].pop(0)
        step = mc["spec"].check_step() if mc["tab"] in (_app.GRIDS, _app.OWN) else mc["spec"].check_step(sim, snaps, nu)
        mc["current"] = (sim, nu, snaps)
        mc["buf"] = ""
        self.status.setText(f"Checking memory: {sim}" + (f" at {nu}²" if nu else "")
                            + f", {len(snaps)} snapshot{'s' if len(snaps) != 1 else ''}…")
        penv = QProcessEnvironment.systemEnvironment()
        for k, v in step.env.items():
            penv.insert(k, v)
        proc = QProcess(self)
        proc.setProcessEnvironment(penv)
        proc.setWorkingDirectory(str(rs.REPO_ROOT))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: mc.__setitem__("buf", mc["buf"] + bytes(proc.readAllStandardOutput()).decode(errors="replace")))
        proc.finished.connect(lambda code, _status: self._check_pass_done(proc, code))
        proc.errorOccurred.connect(lambda err: self._check_pass_error(proc, err, step.argv[0]))
        mc["proc"] = proc
        proc.start(step.argv[0], step.argv[1:])

    def _check_pass_error(self, proc, err, program: str):
        """The check's process could not start (no 'finished' follows then): else _mcheck would stay
        set forever and Run disabled until 'Cancel check'."""
        mc = self._mcheck
        if mc is None or mc.get("proc") is not proc or err != QProcess.FailedToStart:
            return
        self._mcheck = None
        rs.forget_processes()
        self.status.setText(f"Memory check failed: could not start {Path(program).name}")
        self.refresh()

    def _check_pass_done(self, proc, code):
        mc = self._mcheck
        if mc is None or mc.get("proc") is not proc:
            return                                  # cancelled meanwhile
        mc["buf"] += bytes(proc.readAllStandardOutput()).decode(errors="replace")
        sim, nu, snaps = mc["current"]
        reports = rs.parse_reports(mc["buf"])
        results = self.mem_results[mc["key"]]
        over = []
        for n in snaps:
            r = reports.get(n)
            if r is None:
                continue
            results[(sim, n, nu)] = r
            if r.get("over_budget"):
                over.append(n)
        if code != 0 or not any(n in reports for n in snaps):
            mc["failed"] = mc.get("failed", 0) + 1  # a crashed pass, or one that reported nothing, is not "fits"
            # its output into the log, which the status line points to (in the buffer alone, 'see the log' would
            # show nothing of a missing snapshot or a crash); the last 200 lines are enough
            tail = "\n".join(rs.strip_ansi(mc["buf"]).splitlines()[-200:])
            self.log.appendPlainText(f"== memory check of {sim}" + (f" at {nu}²" if nu else "")
                                     + f" failed (exit {code}):\n" + (tail or "(no output)"))
        mc["done"] += 1
        # pipeline: what does not fit is tried again at half the plane size
        if nu and over and nu // 2 >= rs.ADAPTIVE_MIN_NU:
            mc["passes"].append((sim, nu // 2, over))
        self._next_check()

    def _finish_check(self):
        mc, self._mcheck = self._mcheck, None
        rs.forget_processes()           # the check's own PS-DTFE must not read as "already running"
        results = self.mem_results.get(mc["key"], {})
        if mc["tab"] == _app.PIPELINE:
            plan = {}
            for sim, snaps in mc["first"].items():
                for n in snaps:
                    sizes = sorted({k[2] for k in results if k[0] == sim and k[1] == n}, reverse=True)
                    fits = [nu for nu in sizes if not results[(sim, n, nu)].get("over_budget")]
                    if sizes:
                        plan.setdefault(sim, {})[f"{n:03d}"] = fits[0] if fits else sizes[-1]
            self.pipe.plan = plan
            self.pipe.plan_key = mc["key"]
            self._save()
        checked = [r for k, r in results.items()]
        over = sum(1 for r in checked if r.get("over_budget"))
        failed = mc.get("failed", 0)
        self.status.setText(f"Memory check done: {len(checked)} prediction{'s' if len(checked) != 1 else ''}"
                            + (f", {over} over budget" if over else ", all within budget" if checked else "")
                            + (f"; {failed} pass{'es' if failed != 1 else ''} failed (see the log)" if failed else ""))
        self.refresh()

    def _cancel_check(self):
        mc, self._mcheck = self._mcheck, None
        proc = mc.get("proc") if mc else None
        if proc is not None and proc.state() != QProcess.NotRunning:
            for pid in reversed(_app._descendants(int(proc.processId()))):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        self.status.setText("Memory check cancelled")
        self.refresh()

    def _custom_checkable(self) -> bool:
        """The Custom tab's memory check needs the estimator's binary and an existing snapshot file (the standard
        binary reports too)."""
        c = self.custom
        return (bool(c.input_file) and Path(c.input_file).expanduser().is_file()
                and c.binary().is_file())

    def _memory_lines(self, tab: int) -> list[tuple[str, str]]:
        """What the memory checks found for the CURRENT settings, as check-list lines."""
        if self._mcheck is not None:
            return [("info", "checking memory…")]
        out: list[tuple[str, str]] = []
        if tab == _app.GRIDS and self.spec.sim:
            s = self.spec
            res = self.mem_results.get(s.memory_key(), {})
            fits, unchecked = [], []
            for n in sorted(s.snapshots):
                r = res.get((s.sim, n, None))
                if r is None:
                    unchecked.append(n)
                elif r.get("over_budget"):
                    out.append(("warning", f"snapshot {n:03d} needs ~{r['predicted_gb']:.0f} GB, over the "
                                           f"{r['budget_gb']:.0f} GB budget: it will swap and slow the machine "
                                           + ("(a smaller slice plane, no velocity gradient at the slice, fewer "
                                              "fields or a scratch directory lower it)" if s.estimator == "ps" else
                                              "(fewer fields, a smaller padding or a scratch directory lower it)")))
                else:
                    fits.append(r)
            if fits and not unchecked:
                big = max(fits, key=lambda r: r["predicted_gb"])      # (the budget is measured live, per check)
                out.append(("ok", f"memory: all {len(fits)} snapshot{'s' if len(fits) != 1 else ''} fit (the "
                                  f"largest needs ~{big['predicted_gb']:.0f} of its {big['budget_gb']:.0f} GB budget)"))
            elif unchecked and (s.slice_plane or s.grid >= 512):
                out.append(("info", "memory not checked for " + " ".join(f"{n:03d}" for n in unchecked)
                                    + " yet ('Check memory')"))
        elif tab == _app.OWN and self.custom.input_file:
            c = self.custom
            r = self.mem_results.get(c.memory_key(), {}).get((Path(c.input_file).name, 0, None))
            if r is None:
                if c.grid >= 512:
                    out.append(("info", "memory not checked yet ('Check memory')"))
            elif "predicted_gb" not in r:
                out.append(("info", "the memory check gave no prediction for this run"))
            elif r.get("over_budget"):
                out.append(("warning", f"this run needs ~{r['predicted_gb']:.0f} GB, over the {r['budget_gb']:.0f} GB "
                                       "budget: it will swap and slow the machine (a smaller grid, fewer fields "
                                       "or a scratch directory lower it)"))
            else:
                split = rs.report_split_text(r, c.estimator, c.dim)
                out.append(("ok", f"memory: fits (~{r['predicted_gb']:.0f} of the {r['budget_gb']:.0f} GB budget; "
                                  f"the auto-tuner plans {split})"))
        elif tab == _app.PIPELINE:
            p = self.pipe
            key = p.memory_key()
            res = self.mem_results.get(key, {})
            planned = p.adaptive and p.plan_key == key
            fits, unchecked = [], []
            for sim in p.sims:
                if not rs.sim_dir(sim, p.data_root).is_dir():
                    continue
                for n in p.stale(sim, p.nu)[0]:
                    nu = (p.planned_nu(sim, n) or p.nu) if planned else p.nu
                    r = res.get((sim, n, nu))
                    if r is None:
                        unchecked.append(f"{n:03d}")
                        continue
                    if not r.get("over_budget"):
                        fits.append(r)
                        continue
                    smaller = sorted((k[2], v) for k, v in res.items()
                                     if k[0] == sim and k[1] == n and k[2] and k[2] < nu and not v.get("over_budget"))
                    hint = (f"{smaller[-1][0]}² fits (~{smaller[-1][1]['predicted_gb']:.0f} GB)" if smaller
                            else "not even 1024² fits" if nu <= rs.ADAPTIVE_MIN_NU else "check smaller sizes")
                    out.append(("warning", f"{sim} {n:03d} at {nu}² needs ~{r['predicted_gb']:.0f} GB, over the "
                                           f"{r['budget_gb']:.0f} GB budget, so it will swap; {hint}"
                                           + ("" if p.adaptive else " (or tick 'Adaptive')")))
            if fits and not unchecked:
                big = max(fits, key=lambda r: r["predicted_gb"])
                out.append(("ok", f"memory: every snapshot to compute fits (the largest needs "
                                  f"~{big['predicted_gb']:.0f} of its {big['budget_gb']:.0f} GB budget)"))
            elif unchecked and not p.plan_only:
                out.append(("info", "memory not checked yet for " + " ".join(unchecked[:12])
                                    + (" …" if len(unchecked) > 12 else "") + " ('Check memory')"))
        return out


import app as _app  # noqa: E402
