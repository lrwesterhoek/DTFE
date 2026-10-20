"""The Plots tab's job, moved out of runspec.py: PlotSpec, the figure scripts in python/plot (and analyze.py), one
step per script call, built from the declarative figure sets that stay in runspec (FIGURE_SETS, Opt, the void
scripts' default options). runspec re-exports it and is the module to import; this one reads runspec's own names
through it at call time (_rs.NAME)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import runspec as _rs
from dtfelib.cli import DATA_ROOT, FIGURES_ROOT, sim_dir


@dataclass
class PlotSpec:
    """The figure scripts: one Step per script call (per snapshot for the per-snapshot sets)."""
    data_root: str = str(DATA_ROOT)
    sim: str = ""
    snapshots: list[int] = field(default_factory=list)
    sets: list[str] = field(default_factory=lambda: ["fields"])
    method: str = "auto"            # auto | ps | dtfe  (dtfelib --method)
    prefix: str = ""                # "" = the scripts' default grid prefix
    smooth: float = 0.0             # Gaussian smoothing in grid cells (dtfelib --smooth)
    options: dict = field(default_factory=dict)      # {set key: {option key: value}}
    figures_root: str = str(FIGURES_ROOT)            # only plot_pointeval can redirect its output

    def opt(self, set_key: str, key: str):
        default = next(o.default for o in _rs.FIGURE_SET[set_key].opts if o.key == key)
        return self.options.get(set_key, {}).get(key, default)

    def sim_path(self) -> Path:
        return sim_dir(self.sim, self.data_root)

    def _common(self) -> list[str]:
        a = []
        if self.method != "auto":
            a += ["--method", self.method]
        if self.prefix:
            a += ["--prefix", self.prefix]
        if self.smooth > 0:
            a += ["--smooth", f"{self.smooth:g}"]
        return a

    def _snap_estimator(self, n: int) -> str:
        """'ps' or 'dtfe' for snapshot n: the estimator chosen, else what a script's FieldSet(method='auto') loads
        there (a grid prefix means PS-DTFE; PS-DTFE grids when present, else standard DTFE): turning 'auto' into 'ps'
        would fail the void catalogue build on every snapshot of a box with standard-DTFE grids only, which its
        check accepts."""
        if self.method in ("ps", "dtfe"):
            return self.method
        sd = self.sim_path() / f"snapdir_{n:03d}"
        if self.prefix or _rs.has_outputs(sd, "ps_output", "ps"):
            return "ps"
        return "dtfe" if _rs.has_outputs(sd, "", "dtfe") else "ps"

    def _pts_prefix(self, n: int) -> str:
        """The '.pts_*' prefix (without the dot) the point-evaluated sets read for snapshot n: the grid prefix when
        one is set, else the estimator's own (output for standard DTFE); with 'auto' the slice that is there, PS-DTFE
        first, as plot_pointeval.py picks it: assuming ps_output would leave a standard-DTFE hi-res slice beside
        PS-DTFE grids unrenderable from the launcher."""
        if self.prefix:
            return self.prefix
        if self.method in ("ps", "dtfe"):
            return "ps_output" if self.method == "ps" else "output"
        sd = self.sim_path() / f"snapdir_{n:03d}"
        for pre in ("ps_output", "output"):
            if (sd / f"{pre}.pts_den").is_file():
                return pre
        return "ps_output"

    def _void_args(self, key) -> list[str]:
        """The void options a set has: the radius, overlapping voids, the smoothing in Mpc."""
        names = {o.key for o in _rs.FIGURE_SET[key].opts}
        a = []
        if "radius" in names and self.opt(key, "radius") != "r_v":
            a += ["--radius", str(self.opt(key, "radius"))]
        if "keep_overlaps" in names and self.opt(key, "keep_overlaps"):
            a.append("--keep-overlaps")
        if "smooth_mpc" in names and float(self.opt(key, "smooth_mpc")) > 0:
            a += ["--smooth-mpc", f"{float(self.opt(key, 'smooth_mpc')):g}"]
        return a

    @staticmethod
    def _numbers(text, kind=float) -> list | None:
        """'2, 4 8' -> [2, 4, 8]; None when a word is no number of that kind."""
        try:
            return [kind(w) for w in str(text).replace(",", " ").split()]
        except ValueError:
            return None

    def _script_args(self, key) -> list[str]:
        """The void scripts' own options: a flag only for a value other than the script's default."""
        names = {o.key for o in _rs.FIGURE_SET[key].opts}
        a = []

        def num(opt_key, flag, default):
            v = float(self.opt(key, opt_key))
            if opt_key in names and v != default:
                a.extend([flag, f"{v:g}"])
        if key == "voidpop":
            if self.opt(key, "size_rv") != _rs.SIZE_FUNCTION_RV:
                a += ["--size-rv", str(self.opt(key, "size_rv"))]
            num("size_smooth", "--size-smooth", _rs.SIZE_SMOOTH_CELLS)
        elif key in ("voidtrack", "voidfate"):
            if key == "voidtrack" and self.opt(key, "link") != "trees":
                a += ["--link", str(self.opt(key, "link"))]
            if key == "voidfate" and int(self.opt(key, "ref_snap")) != _rs.FATE_REF_SNAP:
                a += ["--ref-snap", str(int(self.opt(key, "ref_snap")))]
            if self.opt(key, "exclusive"):
                a.append("--exclusive")
        elif key == "voidconv":
            f = self._numbers(self.opt(key, "factors"), int)
            if f and f != self._numbers(_rs.CONV_FACTORS, int):
                a += ["--factors", *map(str, f)]
            num("match_sigma", "--match-sigma", _rs.CONV_MATCH_SIGMA)
            r = self._numbers(self.opt(key, "rv_range"))
            if r and r != self._numbers(_rs.CONV_RV_RANGE):
                a += ["--rv-range", *(f"{v:g}" for v in r)]
        elif key == "voidenv":
            if self.opt(key, "criterion") != _rs.ENV_CRITERIA[0]:
                a += ["--criterion", str(self.opt(key, "criterion"))]
            num("factor", "--factor", _rs.ENV_FACTOR)
            if self.opt(key, "r_v_definition") != "crossing":
                a += ["--r-v-definition", str(self.opt(key, "r_v_definition"))]
        elif key == "voidevol":
            if self.opt(key, "rv_definition") != "crossing":
                a += ["--rv-definition", str(self.opt(key, "rv_definition"))]
        elif key == "skeleton":
            if int(self.opt(key, "null_draws")) != _rs.SKELETON_NULL_DRAWS:
                a += ["--null-draws", str(int(self.opt(key, "null_draws")))]
            if int(self.opt(key, "min_component")) > 0:
                a += ["--min-component", str(int(self.opt(key, "min_component")))]
            num("min_component_mass", "--min-component-mass", 0.0)
            num("sheet_min_density", "--sheet-min-density", 0.0)
            if self.opt(key, "all_caustic"):
                a.append("--all-caustic")
        return a

    def _fixed_r_smoothings(self) -> tuple[list | None, list | None]:
        """The size function at fixed R's smoothings as typed: (cells, Mpc), None for a word that is no number."""
        return (self._numbers(self.opt("voidpop", "fixed_r_smooth")),
                self._numbers(self.opt("voidpop", "fixed_r_smooth_mpc")))

    def _fixed_r_step(self, base: list[str], env: dict[str, str]) -> _rs.Step:
        """The Void population set's size function at fixed R: its own call, since the script's
        --size-fixed-r makes ONLY that figure (the population figures stay in their step, under the caller's gate),
        and UNGATED in its own environment: the script refuses it under the ellipsoid gate. A copy of env, which
        every step of the job shares. The estimator and grid prefix only: the smoothings are its own (the common
        --smooth and the population's --smooth-mpc would only mislead), and so are its snapshots (z = 0, 0.5, 1, 2)."""
        cells, mpc = self._fixed_r_smoothings()
        a = base + ["--sim", self.sim]
        if self.method != "auto":
            a += ["--method", self.method]
        if self.prefix:
            a += ["--prefix", self.prefix]
        a.append("--size-fixed-r")
        # the script takes exactly the smoothings given (either list alone means only those): the cells are spelled
        # out beside Mpc ones, left to the script's own default otherwise
        if cells and (mpc or cells != self._numbers(_rs.FIXED_R_SMOOTH_CELLS)):
            a += ["--fixed-r-smooth", *(f"{v:g}" for v in cells)]
        if mpc:
            a += ["--fixed-r-smooth-mpc", *(f"{v:g}" for v in mpc)]
        return _rs.Step(f"{Path(base[1]).name} size function at fixed R", a, dict(env, DTFE_VOID_GATE="none"))

    def steps(self) -> list[_rs.Step]:
        out: list[_rs.Step] = []
        env = _rs._plot_env(self.data_root, self.sim)
        snaps = sorted(self.snapshots)
        for key in [k for k, _ in ((fs.key, fs) for fs in _rs.FIGURE_SETS) if k in self.sets]:
            fs = _rs.FIGURE_SET[key]
            script = str(_rs.REPO_ROOT / "python" / fs.script)
            base = [_rs.PYTHON, script]
            name = Path(fs.script).name
            if key in ("fields", "web"):
                extra = (["--pointeval-only"] if key == "fields" and self.opt(key, "pointeval_only") else []) \
                    + (["--raw"] if self.opt(key, "raw") else [])
                for n in snaps:
                    out.append(_rs.Step(f"{name} snap {n:03d}",
                                        base + ["--sim", self.sim, "--snap", str(n)] + self._common() + extra, env))
            elif key == "series":
                out.append(_rs.Step(f"{name} {len(snaps)} snapshots",
                                    base + ["--sim", self.sim, "--snaps", *map(str, snaps)] + self._common(), env))
            elif key == "pointeval":
                a = base + ["--sims", self.sim, "--snaps", ",".join(map(str, snaps))]
                if self.opt(key, "plane"):
                    a += ["--plane", str(self.opt(key, "plane"))]
                if self.method != "auto":           # a standard-DTFE slice beside PS-DTFE grids
                    a += ["--method", self.method]
                if self.prefix:
                    a += ["--prefix", self.prefix]
                if str(self.opt(key, "fields")).strip():
                    a += ["--fields", str(self.opt(key, "fields")).replace(" ", "")]
                if self.opt(key, "project") != "plane":
                    a += ["--project", str(self.opt(key, "project"))]
                if float(self.opt(key, "smooth_px")) > 0:
                    a += ["--smooth", f"{float(self.opt(key, 'smooth_px')):g}"]
                if float(self.opt(key, "smooth_deriv")) != _rs.POINTEVAL_DERIV_SMOOTH:
                    a += ["--smooth-derivatives", f"{float(self.opt(key, 'smooth_deriv')):g}"]
                if self.opt(key, "fixed_range"):
                    a.append("--fixed-range")
                if self.opt(key, "force"):
                    a.append("--force")
                if Path(self.figures_root) != FIGURES_ROOT:
                    a += ["--figures-root", str(self.figures_root)]
                out.append(_rs.Step(f"{name} {len(snaps)} snapshots", a, env))
            elif key == "panels":
                plane = str(self.opt(key, "plane"))
                for n in snaps:
                    sd = self.sim_path() / f"snapdir_{n:03d}"
                    png = Path(self.figures_root) / "pointeval_panels" / self.sim / \
                        f"snap{n:03d}_{Path(plane).stem or 'plane'}.png"
                    a = base + ["--prefix", str(sd / self._pts_prefix(n)), "--plane", plane,
                                "-o", str(png)]
                    if self.opt(key, "panels") != "density,sheet,streams":
                        a += ["--panels", str(self.opt(key, "panels")).replace(" ", "")]
                    if self.opt(key, "project") != "plane":
                        a += ["--project", str(self.opt(key, "project"))]
                    for opt_key, flag in (("vmin", "--vmin"), ("vmax", "--vmax")):
                        if float(self.opt(key, opt_key)) > 0:
                            a += [flag, f"{float(self.opt(key, opt_key)):g}"]
                    if int(self.opt(key, "dpi")) > 0:
                        a += ["--dpi", str(int(self.opt(key, "dpi")))]
                    if self.opt(key, "image"):
                        a.append("--image")
                    out.append(_rs.Step(f"{name} snap {n:03d}", a, env))
            elif key == "voidpop":
                a = base + ["--sim", self.sim] + self._common()
                if self.opt(key, "panel_selected") and snaps:
                    a += ["--panel-snaps", *map(str, snaps)]
                out.append(_rs.Step(name, a + self._void_args(key) + self._script_args(key), env))
                if self.opt(key, "size_fixed_r"):
                    out.append(self._fixed_r_step(base, env))
            elif key in ("voidprof", "webstreams", "pk", "skeleton"):
                # every caustic snapshot: the script picks them itself and ignores a snapshot list
                every = key == "skeleton" and self.opt(key, "all_caustic")
                a = base + ["--sim", self.sim] + ([] if every else ["--snaps", *map(str, snaps)]) + self._common()
                if key == "voidprof" and self.opt(key, "sample") != "resolved":
                    a += ["--sample", str(self.opt(key, "sample"))]
                if key == "webstreams" and self.opt(key, "web") != "both":
                    a += ["--web", str(self.opt(key, "web"))]
                if key == "pk":
                    if int(self.opt(key, "nbins")) != 40:
                        a += ["--nbins", str(int(self.opt(key, "nbins")))]
                    if not self.opt(key, "ref"):
                        a += ["--ref-snap", "-1"]
                if key == "skeleton" and self.opt(key, "voids"):
                    a.append("--voids")
                out.append(_rs.Step(name + (" every caustic snapshot" if every else f" {len(snaps)} snapshots"),
                                    a + self._void_args(key) + self._script_args(key), env))
            elif key == "voidtrack":
                out.append(_rs.Step(name, base + ["--sim", self.sim, "--snap", str(self.opt(key, "ref"))]
                                    + self._common() + ["--n-voids", str(self.opt(key, "n_voids"))]
                                    + self._void_args(key) + self._script_args(key), env))
            elif key == "crossbox":
                a = base + ["--sims", *str(self.opt(key, "sims")).replace(",", " ").split(),
                            "--data-root", str(self.data_root)]
                if self.method != "auto":
                    a += ["--method", self.method]
                if self.prefix:
                    a += ["--prefix", self.prefix]
                if float(self.opt(key, "smooth_mpc")) != _rs.CROSSBOX_SMOOTH_MPC:
                    a += ["--smooth-mpc", f"{float(self.opt(key, 'smooth_mpc')):g}"]
                if str(self.opt(key, "z")).split() != ["0", "0.5", "1"]:
                    a += ["--z", *str(self.opt(key, "z")).replace(",", " ").split()]
                if int(self.opt(key, "n_boot")) > 0:
                    a += ["--n-boot", str(int(self.opt(key, "n_boot")))]
                if self.opt(key, "replot"):
                    a.append("--replot")
                if Path(self.figures_root) != FIGURES_ROOT:
                    a += ["--out", str(Path(self.figures_root) / "void_crossbox")]
                out.append(_rs.Step(name, a, env))      # its per-box workers get their own DTFE_SIM
            elif key in ("voidevol", "voidenv", "voidfate", "voidconv"):
                out.append(_rs.Step(name, base + ["--sim", self.sim] + self._common() + self._script_args(key), env))
            elif key == "voidcat":
                for n in snaps:
                    a = base + ["build", "--sim", self.sim, "--snap", str(n), "--method", self._snap_estimator(n)]
                    if self.prefix:
                        a += ["--prefix", self.prefix]
                    out.append(_rs.Step(f"void catalogue snap {n:03d}", a, env))
            elif key == "halotrack":
                a = base + ["--sim", self.sim, "--snap", str(self.opt(key, "ref"))] + self._common()
                ids = str(self.opt(key, "subhalos")).replace(",", " ").split()
                a += (["--subhalo", *ids] if ids else ["--top", str(self.opt(key, "top"))])
                if not self.opt(key, "environment"):
                    a.append("--no-environment")
                if float(self.opt(key, "min_ratio")) != 0.1:
                    a += ["--min-ratio", f"{float(self.opt(key, 'min_ratio')):g}"]
                out.append(_rs.Step(name, a, env))
            elif key == "thesis":
                mode = str(self.opt(key, "mode"))
                a = base + [mode]
                if mode in ("all", "compute", "export"):
                    a += [f"{n:03d}" for n in snaps]
                only = [s.strip() for s in str(self.opt(key, "only")).split(",") if s.strip()]
                if only and mode in ("all", "plot"):
                    a += ["--only", *only]
                out.append(_rs.Step(f"analyze.py {mode}", a, env))
        return out

    def problems(self) -> list[tuple[str, str]]:
        out = _rs._root_problems(self.data_root)
        if not out:
            if not self.sim:
                out.append(("error", "no simulation selected"))
            elif not self.sim_path().is_dir():
                out.append(("error", f"{self.sim} not found under {self.data_root}"))
        if not self.sets:
            out.append(("error", "no figure sets selected"))
        if out:
            return out
        snaps = sorted(self.snapshots)
        need_snaps = [k for k in self.sets
                      if _rs.FIGURE_SET[k].per in ("snapshot", "series")
                      and not (k == "thesis" and self.opt(k, "mode") in ("plot", "check"))
                      and not (k == "skeleton" and self.opt(k, "all_caustic"))]
        if need_snaps and not snaps:
            out.append(("error", "no snapshots selected (needed by: "
                                 + ", ".join(_rs.FIGURE_SET[k].title for k in need_snaps) + ")"))
        sp = self.sim_path()
        absent = [Path(_rs.FIGURE_SET[k].script).name for k in self.sets
                  if not (_rs.REPO_ROOT / "python" / _rs.FIGURE_SET[k].script).is_file()]
        if absent:
            out.append(("error", "not in this checkout: " + ", ".join(absent)))
        grid_sets = [k for k in ("fields", "web", "series", "voidcat") if k in self.sets
                     and not (k == "fields" and self.opt(k, "pointeval_only"))]
        if grid_sets:
            missing = [n for n in snaps if not _rs.has_grids(sp / f"snapdir_{n:03d}", self.method, self.prefix)]
            if missing:
                out.append(("warning", f"no {'' if self.method == 'auto' else self.method + ' '}grids for "
                                       f"snapshot{'s' if len(missing) > 1 else ''} "
                                       + " ".join(f"{n:03d}" for n in missing)
                                       + ": those figures will fail"))
        if "series" in self.sets and self.method == "auto" and not any(
                _rs.has_grids(sp / f"snapdir_{n:03d}", "dtfe", "") for n in snaps):
            out.append(("warning", "the DTFE series plots standard-DTFE grids by default; "
                                   "set the estimator to PS-DTFE to plot those instead"))
        for k in ("pointeval", "panels"):
            if k not in self.sets:
                continue
            missing = [n for n in snaps if not (sp / f"snapdir_{n:03d}" / f"{self._pts_prefix(n)}.pts_den").is_file()]
            if missing:
                pre = sorted({self._pts_prefix(n) for n in missing})
                out.append(("warning", f"{_rs.FIGURE_SET[k].title}: no point-evaluated slice "
                                       f"({' or '.join(p + '.pts_den' for p in pre)}) for "
                                       + " ".join(f"{n:03d}" for n in missing)
                                       + "; run the pipeline or a hi-res slice first"))
            plane = str(self.opt(k, "plane"))
            if plane and not Path(plane).is_file():
                out.append(("error", f"{_rs.FIGURE_SET[k].title}: plane file not found: {plane}"))
            elif not plane and k == "panels":
                out.append(("error", "Publication panels: choose the image plane"))
            elif not plane:
                sizes = [_rs.plane_points(p.with_suffix(".bin")) for p in _rs.plane_sidecars(sp)]
                if len(sizes) != len(set(sizes)):
                    out.append(("warning", "Point-evaluated maps: several planes have the same pixel count, "
                                           "so 'auto' may pick the wrong one: choose the plane"))
        if "halotrack" in self.sets or "voidtrack" in self.sets:
            if not (sp / "Merger Trees").is_dir() and not any(sp.glob("trees*")):
                out.append(("warning", f"no merger trees under {sp}; the tracking figures need them "
                                       "(Data tab: merger trees)"))
        if "thesis" in self.sets and self.opt("thesis", "mode") in ("all", "compute", "export"):
            missing = [n for n in snaps if not _rs.has_grids(sp / f"snapdir_{n:03d}", "dtfe", "")]
            if missing:
                out.append(("warning", "Thesis analysis set: no standard-DTFE grids for "
                                       + " ".join(f"{n:03d}" for n in missing)))
        if self.smooth < 0:
            out.append(("error", "smoothing must be >= 0"))
        mpc = [_rs.FIGURE_SET[k].title for k in self.sets if any(o.key == "smooth_mpc" for o in _rs.FIGURE_SET[k].opts)
               and float(self.opt(k, "smooth_mpc")) > 0]
        if mpc and self.smooth > 0:
            out.append(("error", "the smoothing is given in cells and in Mpc (" + ", ".join(mpc) + "): use one"))
        if "voidtrack" in self.sets and self.opt("voidtrack", "exclusive") and self.opt("voidtrack", "link") != "particles":
            # the script's own parser refuses it (merger-tree tracers have no member particles)
            out.append(("error", "Void tracking: 'each particle in one void' needs the particle link"))
        if "voidconv" in self.sets:
            # free-text lists: refused here, not as an argparse error in the job's log
            f = self._numbers(self.opt("voidconv", "factors"), int)
            if not f or min(f) < 2:
                out.append(("error", "Void convergence: the grid factors must be whole numbers of 2 or more"))
            r = self._numbers(self.opt("voidconv", "rv_range"))
            if not r or len(r) != 2 or not 0 <= r[0] < r[1]:
                out.append(("error", "Void convergence: the stack R_v range must be two numbers, low then high"))
        if "voidpop" in self.sets and self.opt("voidpop", "size_fixed_r"):
            # the fixed-R smoothings are free text: refused here, not as the script's error in the log
            cells, mpc = self._fixed_r_smoothings()
            if cells is None or mpc is None or any(v <= 0 for v in cells + mpc):
                out.append(("error", "Void population: the fixed-R smoothings must be positive numbers"))
            elif not cells and not mpc:
                out.append(("error", "Void population: the size function at fixed R needs a smoothing in cells or Mpc"))
        if "skeleton" in self.sets and not self.opt("skeleton", "all_caustic"):
            cc = (self.prefix or "ps_output") + ".causticClass"
            missing = [n for n in snaps if not (sp / f"snapdir_{n:03d}" / cc).is_file()]
            if missing:
                out.append(("warning", f"Caustic skeleton: no {cc} for " + " ".join(f"{n:03d}" for n in missing)
                                       + " (it needs a run with caustics on)"))
        if self.prefix and re.search(r"[\s/]", self.prefix):
            out.append(("error", "grid prefix must be a single word (no spaces or '/')"))
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlotSpec":
        known = {k: v for k, v in _rs._current_figures_root(d).items() if k in cls.__dataclass_fields__}
        known["sets"] = [k for k in known.get("sets", []) if k in _rs.FIGURE_SET]
        return cls(**known)
