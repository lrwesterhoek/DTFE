#!/usr/bin/env python3
"""Tests for the void-profile evolution and the universal-profile fits (2026-10-11): dtfelib/profiles.py's
interior_single_fraction, stack_scatter, hsw_delta / fit_hsw / fit_stack, r_v_by_definition, and
python/plot/plot_void_profile_evolution.py end to end.

Two tiers:
  * SYNTHETIC (always run, CI): exact cell counts on hand-built masks (periodic), a known HSW profile recovered
    within its errors, the normalised linear-theory identity, the R_v-definition fallback, and the script on two
    synthetic 64^3 snapshots (per-snapshot caches, strict JSON, --replot reproduces the tables without a grid).
  * DATA-BACKED (the T7's TNG100-3-Dark z = 0; skipped when it is not on disk): the script on the real snapshot.

Usage:  python3 tests/py_profiles_test.py            (exit 0 = pass)
"""

import atexit
import io as _io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
# the figure root before config/dtfelib are imported: a temp folder, never the real root (the T7 when mounted)
_FIGS_TMP = tempfile.mkdtemp(prefix="profiles_figs_")
os.environ["DTFE_FIGURES_ROOT"] = _FIGS_TMP
os.environ["DTFE_NO_MIRROR"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
atexit.register(shutil.rmtree, _FIGS_TMP, True)

PASS, FAIL, SKIP = [], [], []


class Skip(Exception):
    pass


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Skip as e:
        SKIP.append(name)
        print(f"  SKIP  {name}  ({e})")
    except AssertionError as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")
    except Exception as e:      # a crash is this test's failure, not the end of the run
        import traceback
        FAIL.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        print("".join("        " + line for line in traceback.format_exc().splitlines(True)[-6:]), end="")


# ---------------------------------------------------------------- synthetic: the pieces
def t_interior_fraction():
    """interior_single_fraction against a brute-force count over the whole periodic box, a void across the wrap."""
    from dtfelib import profiles
    n = 24
    rng = np.random.default_rng(3)
    mask = rng.random((n, n, n)) < 0.7
    coords = np.array([[1, 22, 5], [12, 12, 12]])
    radii = np.array([6.3, 4.0])
    got = profiles.interior_single_fraction(mask, coords, radii, x=(0.5, 1.0))
    ii = np.indices((n, n, n))
    for i, (c, r) in enumerate(zip(coords, radii)):
        d2 = sum(((((ii[k] - c[k]) + n // 2) % n) - n // 2) ** 2 for k in range(3))
        for j, xv in enumerate((0.5, 1.0)):
            want = mask[d2 <= (xv * r) ** 2 * (1 + 1e-12)].mean()
            assert abs(got[i, j] - want) < 1e-12, (i, xv, got[i, j], want)
    # a void whose sphere is all single stream gives exactly 1, a radius of NaN gives NaN, a cell on the sphere counts
    ones = np.ones((n, n, n), bool)
    ones[0, 0, 0] = False
    f = profiles.interior_single_fraction(ones, np.array([[12, 12, 12], [0, 0, 3], [5, 5, 5]]), np.array([5.0, 3.0, np.nan]))
    assert f[0, 1] == 1.0 and np.isnan(f[2]).all(), f
    assert abs(f[1, 1] - (1 - 1 / 123.0)) < 1e-12, f[1, 1]       # 123 cell centres within 3 cells; (0,0,0) on the sphere


def t_single_converted_once():
    """2026-10-11: profile_grids converts the single-stream mask to float32 ONCE, not per radial bin (a 512 MB copy
    of a 512^3 mask per bin), and its 'single' profile is the per-bin conversion's, bit for bit."""
    from dtfelib import profiles
    n = 32
    rng = np.random.default_rng(8)
    grids = {"density": rng.random((n, n, n)).astype(np.float32) + 0.1, "velocity": None,
             "single": rng.random((n, n, n)) < 0.6}
    centers, r_eff = rng.random((7, 3)), rng.uniform(1.5, 5.0, 7)
    seen, real = [], profiles.shell_means
    def spy(grid, *a, **k):
        if k.get("mode") == "ngp":
            seen.append(grid)
        return real(grid, *a, **k)
    profiles.shell_means = spy
    try:
        got = profiles.profile_grids(grids, centers, r_eff)
    finally:
        profiles.shell_means = real
    nb = len(profiles.DEFAULT_BINS)
    assert len(seen) == nb and all(g is seen[0] for g in seen) and seen[0].dtype == np.float32, \
        f"{len({id(g) for g in seen})} single-stream copies for {nb} bins"
    dirs = profiles.fibonacci_sphere(got["m"])
    c, r = centers[got["index"]], r_eff[got["index"]]
    want = np.stack([real(grids["single"].astype(np.float32), c, x * r, dirs, mode="ngp")
                     for x in profiles.DEFAULT_BINS], axis=1)
    assert np.array_equal(got["single"], want, equal_nan=True)


def t_hsw_fit():
    """A known HSW profile with noise recovered within 4 sigma; too few shells is ok=False, never an exception."""
    from dtfelib import profiles
    x = np.asarray(profiles.DEFAULT_BINS)
    true = {"delta_c": -0.82, "r_s": 0.93, "alpha": 2.4, "beta": 8.5}
    rng = np.random.default_rng(7)
    e = np.full(x.size, 0.01)
    y = 1.0 + profiles.hsw_delta(x, *true.values()) + rng.normal(0, 0.01, x.size)
    f = profiles.fit_hsw(x, y, e)
    assert f["ok"] and f["constrained"] and not f["at_bound"], f
    for k, v in true.items():
        assert abs(f["params"][k] - v) < 4 * f["errors"][k] + 1e-3, (k, f["params"][k], f["errors"][k], v)
    assert 0.3 < f["chi2_dof"] < 2.5, f["chi2_dof"]
    bad = profiles.fit_hsw(x[:4], y[:4], e[:4])
    assert not bad["ok"] and np.isnan(bad["params"]["alpha"]), bad
    # stack_scatter + fit_stack: two bins, the second below min_count
    n_v = 40
    dens = np.tile(y, (n_v + 3, 1)) + rng.normal(0, 0.05, (n_v + 3, x.size))
    prof = {"bins": x, "density": dens, "r_mpc": np.r_[np.full(n_v, 5.0), np.full(3, 15.0)]}
    mean, err = profiles.stack_scatter(prof, (0, 10, 20))
    want_err = dens[:n_v].std(axis=0, ddof=1) / np.sqrt(n_v)
    assert np.allclose(err[0], want_err) and np.isnan(mean[1]).all(), (err[0][:3], want_err[:3])
    fits = profiles.fit_stack(prof, (0, 10, 20))
    assert fits[0]["ok"] and fits[0]["count"] == n_v and fits[0]["mean_r_mpc"] == 5.0, fits[0]
    assert not fits[1]["ok"] and fits[1]["count"] == 3 and fits[1]["edges"] == [10.0, 20.0], fits[1]


def t_linear_normalised():
    """The panel-(b) units: linear_v_r(R = 1) / (a H) = -(1/3) f x Delta(<x), whatever z (a flat profile here)."""
    from dtfelib import profiles
    from dtfelib import fields as dtfe
    x = np.asarray(profiles.DEFAULT_BINS)
    for z in (0.0, 1.0, 5.0):
        c = dtfe.get_cosmology_params(z)
        v = profiles.linear_v_r(x, np.full(x.size, 0.3), 1.0, z) / (c["a"] * c["H_z"])
        assert np.allclose(v, -(1 / 3) * c["f_growth"] * x * (-0.7), rtol=1e-4), (z, v[:3], (-(1 / 3) * c["f_growth"] * x * (-0.7))[:3])


def t_definition_fallback():
    """r_v_by_definition: 'crossing' is r_v_mpc; a definition the catalogue lacks falls back to it, said once."""
    from dtfelib import profiles, catalog
    cat = {"coords": np.zeros((3, 3), int), "r_v_cells": np.array([2.0, np.nan, 4.0]), "cell_mpc": 0.5}
    r, used = profiles.r_v_by_definition(cat, "crossing")
    assert used == "crossing" and np.allclose(r, [1.0, np.nan, 2.0], equal_nan=True), r
    said = []
    r2, used2 = profiles.r_v_by_definition(cat, "ridge", say=said.append)
    assert used2 == "crossing" and np.allclose(r2, r, equal_nan=True) and len(said) == 1 and "--" not in said[0], said
    if "ridge" in getattr(catalog, "R_V_DEFINITIONS", ()):          # contract C3 landed: the key is read
        cat["r_v_ridge_cells"] = np.array([3.0, 3.0, np.nan])
        r3, used3 = profiles.r_v_by_definition(cat, "ridge", say=said.append)
        assert used3 == "ridge" and np.allclose(r3, [1.5, 1.5, np.nan], equal_nan=True), r3
        cat["r_v_delta08_cells"] = np.full(3, np.nan)                # undefined everywhere (high z): no fallback
        r4, used4 = profiles.r_v_by_definition(cat, "delta08", say=said.append)
        assert used4 == "delta08" and np.isnan(r4).all() and len(said) == 1, (used4, r4, said)


# ---------------------------------------------------------------- synthetic: the script end to end
def _pin_config(root, saved, n=64, box=64.0):
    """tests/py_dtfelib_test.py's _void_pin_config: the synthetic ladder's frame and today's catalogue knobs."""
    import config
    pins = {"CACHE_DIR": Path(root) / "cache", "THESIS_FIGURES_DIR": Path(root) / "mirror",
            "LOCAL_FIGURES_ROOT": str(Path(root) / "figures"),
            "CELL_SIZE": box / n, "BOX_SIZE": box, "FIELD_RESOLUTION": n, "SMOOTHING_SIGMA_CELLS": 1.5,
            "PRODUCTION_SMOOTHING": (1.5, 11),
            "ELLIPSOID_CUT_RULES": {"min_axis_cells": 0.1, "max_axis_sigma": 10.0 / 1.5, "max_axis_box_frac": 0.5,
                                    "max_axis_ratio": 10.0},
            "FOOTPRINT_SIZE": 11, "GRADIENT_THRESHOLD": 0.1, "VOID_EIGENVALUE_CRITERION": "trace",
            "DEEP_VOID_THRESHOLD": -0.1}
    saved.update({k: getattr(config, k) for k in pins})
    for k, v in pins.items():
        setattr(config, k, v)


DIPS = (((16, 16, 16), 2.0, 0.85), ((44, 20, 40), 2.5, 0.8), ((20, 48, 30), 1.6, 0.9),
        ((50, 50, 10), 3.0, 0.75), ((8, 36, 54), 2.2, 0.85), ((36, 6, 26), 1.8, 0.8))


def _snapdir(data, snap, z, seed, n=64, box=64.0, multi_r=3.0):
    """One PS-DTFE snapshot folder under data/SYN: header, six Gaussian dips on 2% noise, zero velocity, and a
    streams grid of 3 inside a sphere of 'multi_r' cells around the first dip (multi-stream void centre), plus a
    hidden-streams bit-1 shell around the second dip (counted as multi-stream too)."""
    import h5py
    d = Path(data) / "SYN" / f"snapdir_{snap:03d}"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / f"combined_{snap:03d}.hdf5", "w") as f:
        h = f.create_group("Header").attrs
        h["BoxSize"], h["Redshift"], h["HubbleParam"], h["HFreeUnits"] = box * 1000.0, float(z), 0.6774, 1
        h["NumPart_ThisFile"], h["MassTable"] = np.array([0, n ** 3, 0, 0, 0, 0]), np.array([0, 1.0, 0, 0, 0, 0])
    ii = np.indices((n, n, n))
    den = 1.0 + 0.02 * np.random.default_rng(seed).normal(size=(n, n, n))
    r = []
    for c, w, depth in DIPS:
        r2 = sum((((ii[k] - c[k] + n // 2) % n) - n // 2) ** 2 for k in range(3))
        den -= depth * np.exp(-0.5 * r2 / w ** 2)
        r.append(np.sqrt(r2))
    den.astype(np.float32).tofile(str(d / "ps_output.a_den"))
    np.zeros((n, n, n, 3), np.float32).tofile(str(d / "ps_output.a_vel"))
    np.where(r[0] <= multi_r, 3.0, 1.0).astype(np.float32).tofile(str(d / "ps_output.a_streams"))
    np.where((r[1] > 1.5) & (r[1] <= 2.5), 1.0, 0.0).astype(np.float32).tofile(str(d / "ps_output.a_hidden_streams"))
    return d


def t_script_synthetic():
    import config
    from dtfelib import cli, profiles
    plot_dir = str(REPO / "python" / "plot")
    sys.path.insert(0, plot_dir)
    try:
        import plot_void_profile_evolution as pve
    finally:
        sys.path.remove(plot_dir)
    root = Path(tempfile.mkdtemp(prefix="profiles_run_"))
    saved, old_root = {}, cli.DATA_ROOT
    data, out = root / "data", root / "out"

    def run(args):
        buf = _io.StringIO()
        with redirect_stdout(buf):
            try:
                pve.main(["--sim", "SYN", "--data-root", str(data), "--out", str(out)] + args)
                code = 0
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
                if isinstance(e.code, str):
                    print(e.code)
        return code, buf.getvalue()

    try:
        _pin_config(root, saved)
        code, text = run(["--replot"])                      # nothing cached yet: a clear exit 1
        assert code == 1 and "no per-snapshot profiles" in text, (code, text[-300:])
        _snapdir(data, 50, 1.0, 1)
        _snapdir(data, 99, 0.0, 2)
        common = ["--rv-range", "0,20", "--edges", "0,4,30", "--min-count", "2", "--n-boot", "30", "--points", "100"]
        code, text = run(common)
        assert code == 0, (code, text[-800:])
        names = sorted(q.name for q in out.iterdir())
        assert names == ["per_snapshot", "profile_evolution.npz", "profile_evolution.png", "profile_fits.json",
                         "profile_fits.png", "profile_fits.txt"], names
        assert sorted(q.name for q in (out / "per_snapshot").iterdir()) == ["profiles_050.npz", "profiles_099.npz"]
        with np.load(out / "per_snapshot" / "profiles_099.npz") as f:
            c = {k: f[k] for k in f.files}
        assert c["density"].shape[0] == len(c["r_mpc"]) >= 4 and c["single_interior"].shape == (len(c["r_mpc"]), 2), \
            {k: np.shape(v) for k, v in c.items()}
        # the box fraction: everything but the multi-stream sphere and the hidden-bit shell
        n = 64
        ii = np.indices((n, n, n))
        rr = [np.sqrt(sum((((ii[k] - cc[k] + n // 2) % n) - n // 2) ** 2 for k in range(3))) for cc, _w, _d in DIPS[:2]]
        want_box = 1.0 - ((rr[0] <= 3.0) | ((rr[1] > 1.5) & (rr[1] <= 2.5))).mean()
        assert abs(float(c["single_box"]) - want_box) < 1e-9, (float(c["single_box"]), want_box)
        # the first dip's void (its catalogue cell at the dip centre) is multi-stream inside: f(<0.5 R_v) < 1
        cat_coords = None
        from dtfelib import pipeline
        p = pipeline.products(99, sim="SYN", data_root=data)
        try:
            cat_coords = np.asarray(p.voids()["coords"])[c["index"]]
        finally:
            p.release()
        first = np.nonzero(np.all(cat_coords == np.array([16, 16, 16]), axis=1))[0]
        assert first.size == 1, cat_coords
        mask = np.ones((n, n, n), bool)
        mask[(rr[0] <= 3.0) | ((rr[1] > 1.5) & (rr[1] <= 2.5))] = False
        want = profiles.interior_single_fraction(mask, cat_coords, c["r_mpc"] / config.CELL_SIZE)
        assert np.allclose(c["single_interior"], want), (c["single_interior"][:3], want[:3])
        assert c["single_interior"][first[0], 0] < 1.0, c["single_interior"][first[0]]
        doc = json.loads((out / "profile_fits.json").read_text())        # strict JSON (allow_nan=False wrote it)
        assert doc["sim"] == "SYN" and doc["snaps"] == [50, 99] and len(doc["fits"]) == 4, (doc["snaps"], len(doc["fits"]))
        assert all(set(r["params"]) == set(profiles.HSW_PARAMS) for r in doc["fits"])
        assert [o["snap"] for o in doc["overlay"]] == [50, 99] and "f_ss_lt_0.5Rv" in doc["overlay"][1], doc["overlay"][1]
        txt = (out / "profile_fits.txt").read_text()
        assert "f_ss(<0.5Rv)" in txt and txt.count("\n") > 8, txt[:400]
        assert "R_v 0-20 sigma" in txt and "$" not in txt, txt[:600]
        # plot_void_profiles.py: the HSW fit per R_v bin in its npz (2026-10-11); none in r / R_eff
        try:
            sys.path.insert(0, plot_dir)
            import plot_void_profiles as pvp
        finally:
            sys.path.remove(plot_dir)
        argv0 = list(sys.argv)
        try:
            for extra in ([], ["--radius", "r_eff"]):
                sys.argv = ["plot_void_profiles.py", "--sim", "SYN", "--data-root", str(data), "--out", str(root / "pvp"),
                            "--snaps", "99", "--edges", "0,4,30", "--n-boot", "20", "--points", "100"] + extra
                with redirect_stdout(_io.StringIO()):
                    try:
                        pvp.main()
                    except SystemExit as e:
                        assert not e.code, e.code
        finally:
            sys.argv = argv0
        with np.load(root / "pvp" / "void_profiles_ps_resolved_edges0-4-30_z0.00.npz") as f:    # --edges: tagged (2026-10-11)
            assert f["hsw_params"].shape == (2, 4) and f["hsw_errors"].shape == (2, 4), f["hsw_params"].shape
            assert len(f["hsw_ok"]) == 2 and len(f["hsw_constrained"]) == 2 and len(f["hsw_chi2_dof"]) == 2
            assert list(f["hsw_param_names"]) == list(profiles.HSW_PARAMS), f["hsw_param_names"]
            # a bin is fitted exactly when it holds min_count voids (here the six voids sit in one of the two bins)
            assert [bool(o) for o in f["hsw_ok"]] == [int(c) >= int(f["min_count"]) for c in f["count"]] \
                and f["hsw_ok"].any(), (f["hsw_ok"], f["count"])
        with np.load(root / "pvp" / "void_profiles_ps_resolved_reff_edges0-4-30_z0.00.npz") as f:
            assert "hsw_params" not in f.files, f.files
        # --replot: no grid read (the snapdirs gone), the same tables
        shutil.rmtree(data)
        before = (out / "profile_fits.json").read_text()
        (out / "profile_fits.json").unlink()
        code, text = run(common + ["--replot"])
        assert code == 0 and (out / "profile_fits.json").read_text() == before, (code, text[-500:])
        # an option tag: --sample deep looks for profiles_<snap>_deep.npz only
        code, text = run(common + ["--replot", "--sample", "deep"])
        assert code == 1 and "tag '_deep'" in text, (code, text[-300:])
    finally:
        for k, v in saved.items():
            setattr(config, k, v)
        cli.DATA_ROOT = old_root
        shutil.rmtree(root, ignore_errors=True)


def t_help_text():
    """The launcher-facing help has no '--' outside flag names (the house rule for shown text)."""
    out = subprocess.run([sys.executable, str(REPO / "python" / "plot" / "plot_void_profile_evolution.py"), "--help"],
                         capture_output=True, text=True, timeout=120,
                         env={**os.environ, "DTFE_FIGURES_ROOT": _FIGS_TMP})
    assert out.returncode == 0, out.stderr[-400:]
    assert " -- " not in out.stdout and "--rv-range" in out.stdout and "--replot" in out.stdout, out.stdout[-400:]


# ---------------------------------------------------------------- data-backed
def t_tng100_z0():
    """The script on the T7's TNG100-3-Dark z = 0 (into a temp folder): voids profiled, fits converged, the interior
    single-stream fractions between 0 and 1 with r < 0.5 R_v at least as single-stream as r < R_v."""
    from dtfelib import cli
    sim = "TNG100-3-Dark"
    snapdir = cli.sim_dir(sim) / "snapdir_099"
    if not (snapdir / "ps_output.a_streams").is_file():
        raise Skip(f"{snapdir} not on disk")
    tmp = Path(tempfile.mkdtemp(prefix="profiles_tng_"))
    try:
        env = {**os.environ, "DTFE_SIM": sim, "DTFE_FIGURES_ROOT": str(tmp), "DTFE_NO_MIRROR": "1", "MPLBACKEND": "Agg"}
        out = subprocess.run([sys.executable, str(REPO / "python" / "plot" / "plot_void_profile_evolution.py"),
                              "--sim", sim, "--snaps", "99", "--out", str(tmp / "o"), "--n-boot", "50"],
                             capture_output=True, text=True, timeout=1200, env=env)
        assert out.returncode == 0, (out.stdout[-600:], out.stderr[-600:])
        doc = json.loads((tmp / "o" / "profile_fits.json").read_text())
        assert sum(r["ok"] for r in doc["fits"]) >= 2, doc["fits"]
        o = doc["overlay"][0]
        a, b = o["f_ss_lt_0.5Rv"]["mean"], o["f_ss_lt_1Rv"]["mean"]
        assert o["count"] >= 10 and 0 < b <= a <= 1 and 0 < o["single_box"] < 1, o
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("=" * 60)
    print(" void-profile evolution + HSW fits tests")
    print("=" * 60)
    print("synthetic:")
    check("interior_single_fraction: exact periodic count at 0.5 and 1 R, all-single = 1, NaN radius, a cell on the sphere", t_interior_fraction)
    check("2026-10-11: profile_grids converts the single-stream mask once, not per bin; same profile bit for bit", t_single_converted_once)
    check("fit_hsw recovers a known HSW 2014 profile within 4 sigma; too few shells ok=False; stack_scatter = std/sqrt(n); fit_stack's min_count", t_hsw_fit)
    check("panel (b) units: linear_v_r(R = 1) / (a H) = -(1/3) f x Delta(<x) at z = 0, 1, 5", t_linear_normalised)
    check("r_v_by_definition: crossing, the fallback said once, the C3 key read when catalog has it", t_definition_fallback)
    check("plot_void_profile_evolution.py on two synthetic snapshots: files, per-snapshot caches, single-stream box and interior fractions exact, strict JSON, --replot without grids gives the same tables, the _deep tag", t_script_synthetic)
    check("plot_void_profile_evolution.py --help: no ' -- ' in the shown text", t_help_text)
    print("data-backed:")
    check("TNG100-3-Dark z = 0 on the T7: fits, interior single-stream fractions ordered", t_tng100_z0)
    print("-" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if SKIP:
        print(f"({len(SKIP)} skipped)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
