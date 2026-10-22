#!/usr/bin/env python3
"""Tests for the MEAN-FIELD velocity derivatives: dtfelib.fields.mean_field_derivatives, the
FieldSet API around it (load(name, mean_field=True), mean_field(), has(..., mean_field=True), lambda_th) and the
figure scripts' '--ps-gradient' default.

Why: standard DTFE stores the derivatives of its single-valued velocity field, grad<v>; PS-DTFE stores the stream
average of each stream's own gradient <grad v> (dtfelib/io.py, the note above FIELDS). The mean-field functions give
a PS output the standard quantity, from its velocity grid, in the binary's conventions.

  * SYNTHETIC (always run): a band-limited analytic flow with both a potential and a rotational part -- the spectral
    gradient equals the analytic one (float64 to 1e-10, float32 to 3e-6 of the largest value), the divergence, the
    5 shear and 3 vorticity components equal a transliteration of web_fields.cc's velocityShear / velocityVorticity loops
    (which pins the component ORDER and the axis convention c = a*3 + b = dv_a/dx_b), the V-web eigenvalues equal
    numpy's of -sqrt(a)/(2*100h) (g + g^T) and the labels count them above lambda_th; the vectorized Cardano
    against numpy (random, isotropic, the pancake's {d, 0, 0}); the spectra plan (one alive for the divergence, two
    for the V-web, three for the shear) and memmap input; FieldSet: PS mean_field == the analytic grid times sqrt(a),
    standard DTFE's mean_field=True byte-identical to its stored grid, has() needs the velocity grid for PS, the
    run log's --lambda_th, the refusals (a non-gradient name, the memory gate); the figure scripts' default
    (plot_web_streams' table source and names, plot_cosmic_web and plot_shear_triaxial's report). The
    one-plane entry point the launcher's Explore uses, mean_field_plane: np.take of the cube's result to float
    rounding for all six names, every axis, planes at the box faces and on a slab boundary, random (not band-limited)
    float32 and float64 grids of even and odd size, read through a minimal slab reader (no array, no memmap).
  * BINARY (the acceptance test; skipped when the binaries are not built -- a FAILURE under DTFE_TEST_REQUIRED=1):
    a Zel'dovich field plus a rotational flow, 32^3 particles, through the standard binary with --exact-average at
    32^3 and 64^3: the mean field of its OWN a_vel reproduces its a_velDiv (correlation >= 0.99), every gradient,
    shear and vorticity component (>= 0.97), its V-web labels (>= 95% of the cells), and our V-web normalisation +
    Cardano applied to its stored gradient give its stored eigenvalues and labels exactly. Through PS-DTFE
    (--ps-volume-weighted, shell-crossed): the stored divergence has a positive box mean, the mean-field one none,
    and the mean-field V-web's filament fraction lies closer to standard DTFE's than the stored PS V-web's.
    DTFE_TEST_PRECISION=double runs the -double pair (float64 grids, float64 results).

Usage:  python3 tests/py_meanfield_test.py        (exit 0 = pass; DTFE_TEST_TMP = where the binary runs write)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from _harness import REPO, Skip, check as _check, setup, summary

setup("meanfield_figs_", paths=("python",), mpl_agg=False)


def check(name, fn):
    """Skip marks a binary-backed check whose binary is not built: a FAILURE under DTFE_TEST_REQUIRED=1."""
    _check(name, fn, required=True)


# ---------------------------------------------------------------- the analytic flow
# v_a(x) = sum_m A_ma sin(k_m . x + phi_m): band-limited (|m_i| < N/2), so its spectral gradient is exact.
# Mode 1 is longitudinal (A || k: divergence), mode 2 transverse (A . k = 0: vorticity), mode 3 a mix along a
# diagonal (cross terms of every pair of axes).
MODES = (((1, 0, 0), (3.0, 0.0, 0.0), 0.3),
         ((0, 2, 1), (1.0, 0.5, -1.0), 1.1),
         ((1, -1, 2), (-0.7, 2.0, 1.4), 2.0))


def analytic_flow(n, box, dtype=np.float64, scale=1.0):
    """(velocity (n,n,n,3), gradient (n,n,n,3,3) with g[..., a, b] = dv_a/dx_b), at the cell centres; 'scale'
    multiplies the amplitudes (400: gradients of ~100 km/s/Mpc, V-web eigenvalues of order 1)."""
    x = (np.arange(n) + 0.5) * box / n
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1)
    v = np.zeros((n, n, n, 3))
    g = np.zeros((n, n, n, 3, 3))
    for m, amp, phi in MODES:
        k = 2 * np.pi * np.asarray(m, float) / box
        ph = X @ k + phi
        for a in range(3):
            v[..., a] += scale * amp[a] * np.sin(ph)
            for b in range(3):
                g[..., a, b] += scale * amp[a] * k[b] * np.cos(ph)
    return v.astype(dtype), g


def cpp_shear(g):
    """web_fields.cc velocityShear, line for line (velGrad[i*3+j] = g[..., i, j])."""
    vg = g.reshape(g.shape[:-2] + (9,))
    temp = []
    for i in range(2):
        for j in range(i, 3):
            temp.append((vg[..., i * 3 + j] + vg[..., j * 3 + i]) / 2.0)
    trace = vg[..., 2 * 3 + 2].copy()
    for d in range(2):
        trace = trace + temp[d * 3 - d * (d - 1) // 2]
    trace = trace / 3
    for d in range(2):
        temp[d * 3 - d * (d - 1) // 2] = temp[d * 3 - d * (d - 1) // 2] - trace
    return np.stack(temp, axis=-1)


def cpp_vorticity(g):
    """web_fields.cc velocityVorticity, line for line."""
    vg = g.reshape(g.shape[:-2] + (9,))
    return np.stack([(vg[..., i * 3 + j] - vg[..., j * 3 + i]) / 2.0 for i in range(3) for j in range(i + 1, 3)], axis=-1)


def t_analytic():
    from dtfelib import fields
    n, box = 24, 50.0
    for dtype, tol in ((np.float64, 1e-10), (np.float32, 3e-6)):
        v, g = analytic_flow(n, box, dtype)
        r = fields.mean_field_derivatives(v, box, fields.MEAN_FIELD_NAMES[:4])
        scale = np.abs(g).max()
        assert all(a.dtype == dtype for a in r.values()), {k: a.dtype for k, a in r.items()}
        want = {"gradient": g.reshape(n, n, n, 9), "divergence": np.trace(g, axis1=-2, axis2=-1),
                "shear": cpp_shear(g), "vorticity": cpp_vorticity(g)}
        for name, w in want.items():
            assert r[name].shape == w.shape, (name, r[name].shape, w.shape)
            err = np.abs(r[name] - w).max() / scale
            assert err < tol, (np.dtype(dtype).name, name, err)
        # the layout: dv_y/dx of mode 1 (v_x only, varying along x) is 0, dv_x/dx is not
        assert np.abs(r["gradient"][..., 0]).max() > 0.1 * scale


def t_vweb_normalisation():
    from dtfelib import fields
    n, box, a, h, lam = 24, 50.0, 0.5, 0.7, 0.3
    v, g = analytic_flow(n, box, scale=400.0)
    r = fields.mean_field_derivatives(v, box, ("vweb", "vweb_eigenvalues"), scale_factor=a, hubble_h=h, lambda_th=lam)
    sigma = -np.sqrt(a) / (2 * 100 * h) * (g + np.swapaxes(g, -1, -2))
    ev = np.linalg.eigvalsh(sigma)[..., ::-1]
    err = np.abs(r["vweb_eigenvalues"] - ev).max() / np.abs(ev).max()
    assert err < 1e-10, err
    assert np.array_equal(r["vweb"], (r["vweb_eigenvalues"] > lam).sum(-1).astype(np.float32))
    assert r["vweb"].dtype == np.float32 and set(np.unique(r["vweb"])) <= {0.0, 1.0, 2.0, 3.0}
    assert len(np.unique(r["vweb"])) >= 3, np.unique(r["vweb"])            # the flow has several classes
    # the binary's fallbacks: no scale factor / no h -> 1
    r1 = fields.mean_field_derivatives(v, box, "vweb_eigenvalues", scale_factor=None, hubble_h=None)
    ev1 = np.linalg.eigvalsh(-0.5 * (g + np.swapaxes(g, -1, -2)))[..., ::-1]
    assert np.abs(r1["vweb_eigenvalues"] - ev1).max() / np.abs(ev1).max() < 1e-10


def t_cardano():
    from dtfelib import fields
    rng = np.random.default_rng(3)
    m = rng.normal(size=(5000, 3, 3))
    m = m + np.swapaxes(m, -1, -2)
    e = fields.symmetric_eigenvalues3(m[:, 0, 0], m[:, 0, 1], m[:, 0, 2], m[:, 1, 1], m[:, 1, 2], m[:, 2, 2])
    assert np.abs(e - np.linalg.eigvalsh(m)[:, ::-1]).max() < 1e-10
    iso = fields.symmetric_eigenvalues3(2.0, 0.0, 0.0, 2.0, 0.0, 2.0)
    assert np.array_equal(iso, [2.0, 2.0, 2.0]), iso
    pan = fields.symmetric_eigenvalues3(0.0, 0.0, 0.0, 3.096, 0.0, 0.0)  # the pancake's {d, 0, 0}, out of order
    assert np.abs(pan - [3.096, 0.0, 0.0]).max() < 1e-12, pan
    assert np.array_equal(fields.classify_web(np.array([[0.5, 0.3, -1.0], [0.31, 0.31, 0.31]]), 0.3), [1.0, 3.0])


def t_plan_and_memmap():
    from dtfelib import fields
    spectra = {nm: fields._mf_max_spectra(fields._mf_jobs((nm,))) for nm in fields.MEAN_FIELD_NAMES}
    assert spectra == {"gradient": 1, "divergence": 1, "shear": 3, "vorticity": 2, "vweb": 2, "vweb_eigenvalues": 2}, spectra
    assert fields.mean_field_peak_bytes(512, ("divergence",)) < fields.mean_field_peak_bytes(512, ("vweb",)) \
        < 9 * 512 ** 3 * 8, "the V-web must stay below the naive 9 float64 components"
    n, box = 16, 30.0
    v, _ = analytic_flow(n, box, np.float32)
    tmp = Path(tempfile.mkdtemp(prefix="mf_mm_"))
    try:
        v.tofile(str(tmp / "v"))
        mm = np.memmap(tmp / "v", dtype=np.float32, mode="r", shape=v.shape)
        a = fields.mean_field_derivatives(mm, box, ("divergence", "vweb"), workers=1)
        b = fields.mean_field_derivatives(v, box, ("divergence", "vweb"))
        assert np.array_equal(a["divergence"], b["divergence"]) and np.array_equal(a["vweb"], b["vweb"])
        del mm
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for bad in (np.zeros((4, 4, 4)), np.zeros((4, 4, 5, 3))):
        try:
            fields.mean_field_derivatives(bad, 1.0)
            raise AssertionError(f"shape {bad.shape} accepted")
        except ValueError:
            pass


class _Slabs:
    """The minimal velocity input mean_field_gradient_plane takes (the launcher's file reader): .shape, .dtype and
    [s:e] along axis 0 -- every slab asked for is recorded."""

    def __init__(self, a):
        self._a, self.shape, self.dtype, self.asked = a, a.shape, a.dtype, []

    def __getitem__(self, sl):
        self.asked.append((sl.start, sl.stop))
        return np.array(self._a[sl])


def t_plane():
    """mean_field_plane: the plane 'index' normal to 'axis' of mean_field_derivatives' result, from one
    pass in slabs -- the normal derivative as the periodic kernel irfft(i k) summed along the line, the in-plane ones
    as 1-D FFTs of the plane."""
    from dtfelib import fields
    rng = np.random.default_rng(11)
    names, lam = fields.MEAN_FIELD_NAMES, 0.3
    for n, dtype, tol in ((24, np.float32, 2e-6), (15, np.float64, 1e-11)):
        v = (300.0 * rng.normal(size=(n, n, n, 3))).astype(dtype)            # every mode up to Nyquist (n even)
        cube = fields.mean_field_derivatives(v, 40.0, names, scale_factor=0.5, hubble_h=0.7, lambda_th=lam)
        slab = 4 * n * n * 3 * np.dtype(dtype).itemsize                       # 4 layers a slab: several slabs
        for axis in range(3):
            for index in (0, 4, 7, n - 1):                                    # the faces, a slab's first and last layer
                src = _Slabs(v)
                got = fields.mean_field_plane(src, 40.0, names, axis, index, scale_factor=0.5, hubble_h=0.7,
                                              lambda_th=lam, slab_bytes=slab)
                assert src.asked[0] == (0, 4) and src.asked[-1][1] == n and len(src.asked) == -(-n // 4), src.asked
                for nm in names:
                    want = np.take(cube[nm], index, axis=axis)
                    assert got[nm].dtype == want.dtype and got[nm].shape == want.shape, (nm, got[nm].dtype, want.shape)
                    if nm == "vweb":                  # the labels: equal wherever no eigenvalue sits at the threshold
                        ev = np.take(cube["vweb_eigenvalues"], index, axis=axis)
                        clear = (np.abs(ev - lam) > 1e-4 * np.abs(ev).max()).all(axis=-1)
                        assert clear.mean() > 0.9 and np.array_equal(got[nm][clear], want[clear]), (n, axis, index)
                        continue
                    err = float(np.abs(got[nm].astype(np.float64) - want).max() / np.abs(want).max())
                    assert err < tol, (n, np.dtype(dtype).name, axis, index, nm, err)
    g = fields.mean_field_gradient_plane(v, 40.0, 1, 3)
    assert g.shape == (15, 15, 3, 3) and g.dtype == np.float64
    assert np.array_equal(fields.mean_field_from_gradient(g, "divergence", real=np.float64)["divergence"],
                          np.trace(g, axis1=-2, axis2=-1)), "the divergence is the trace"
    for bad in ((np.zeros((4, 4, 4)), 0, 0), (np.zeros((4, 4, 4, 3)), 3, 0), (np.zeros((4, 4, 4, 3)), 0, 4)):
        try:
            fields.mean_field_gradient_plane(bad[0], 1.0, bad[1], bad[2])
            raise AssertionError(f"accepted {bad[0].shape} axis {bad[1]} index {bad[2]}")
        except ValueError:
            pass


def _snapdir(root, z=1.0, h=0.7, box=50.0, n=16):
    """A snapshot folder with a combined_*.hdf5 header (box in h-free ckpc, as the merged files) for FieldSet."""
    import h5py
    d = Path(root) / "snapdir_050"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / "combined_050.hdf5", "w") as f:
        a = f.create_group("Header").attrs
        a["BoxSize"], a["Redshift"], a["HubbleParam"], a["HFreeUnits"] = box * 1000.0, float(z), h, 1
        a["NumPart_ThisFile"], a["MassTable"] = np.array([0, n ** 3, 0, 0, 0, 0]), np.array([0, 1.0, 0, 0, 0, 0])
    return d


def t_fieldset_api():
    from dtfelib import FieldSet, fields
    from dtfelib.io import DEFAULT_LAMBDA_TH, MEAN_FIELD_NAMES
    assert MEAN_FIELD_NAMES == fields.MEAN_FIELD_NAMES
    n, box, z, h = 16, 50.0, 1.0, 0.7
    root = Path(tempfile.mkdtemp(prefix="mf_fs_"))
    try:
        d = _snapdir(root, z, h, box, n)
        v, g = analytic_flow(n, box, np.float32, scale=400.0)
        junk = np.random.default_rng(2).normal(size=(n, n, n)).astype(np.float32)
        for prefix in ("ps_output", "output"):
            v.tofile(str(d / f"{prefix}.a_vel"))
            junk.tofile(str(d / f"{prefix}.a_velDiv"))
        ps, dt = FieldSet(d, method="ps"), FieldSet(d, method="dtfe")
        sa = np.float32(np.sqrt(1.0 / (1.0 + z)))
        div = np.trace(g, axis1=-2, axis2=-1)
        mf = ps.load("divergence", mean_field=True)
        assert mf.dtype == np.float32 and np.abs(mf - div * sa).max() < 1e-5 * np.abs(div).max(), "PS mean field != analytic x sqrt(a)"
        assert np.array_equal(ps.load("divergence"), junk * sa), "the stored PS grid must stay what it was"
        assert np.array_equal(ps.load("divergence", scaled=False, mean_field=True),
                              fields.mean_field_derivatives(v, box, "divergence")["divergence"])
        assert np.array_equal(dt.load("divergence", mean_field=True), dt.load("divergence")), "standard DTFE: the stored grid"
        assert np.abs(dt.mean_field("divergence") - div * sa).max() < 1e-5 * np.abs(div).max()
        assert ps.has("vweb", mean_field=True) and not ps.has("vweb") and not dt.has("vweb", mean_field=True), "has()"
        # lambda_th: the run log's, else the run scripts' default; the V-web is classified with it
        assert ps.lambda_th == DEFAULT_LAMBDA_TH == 0.3, ps.lambda_th
        (d / "ps_output.runlog").write_text("RUNNING: ./PS-DTFE in out --grid 16 --periodic --lambda_th 0.05 --field vweb_a\n")
        assert ps.lambda_th == 0.05, ps.lambda_th
        both = ps.mean_field(("vweb", "vweb_eigenvalues"))
        assert np.array_equal(both["vweb"], fields.classify_web(both["vweb_eigenvalues"], 0.05))
        assert not np.array_equal(both["vweb"], ps.mean_field("vweb", lambda_th=0.5)), "lambda_th ignored"
        (d / "output.runlog").write_text("... Computing the T-web from the tidal tensor of the raw density grid (lambda_th = 0.2) ...\n")
        assert dt.lambda_th == 0.2, dt.lambda_th                          # the binary's own report, without a RUNNING line
        e = fields.mean_field_derivatives(v, box, "vweb_eigenvalues", scale_factor=1.0 / (1.0 + z), hubble_h=h)
        assert np.array_equal(both["vweb_eigenvalues"], e["vweb_eigenvalues"]), "the V-web is never sqrt(a)-scaled twice"
        # refusals
        for call in (lambda: ps.load("density", mean_field=True), lambda: ps.mean_field("tweb")):
            try:
                call()
                raise AssertionError("a non-gradient name accepted")
            except ValueError:
                pass
        old = os.environ.get("DTFE_PY_RAM_GB")
        os.environ["DTFE_PY_RAM_GB"] = "0.000001"
        try:
            ps.mean_field("divergence")
            raise AssertionError("the memory gate let a 1 kB budget through")
        except MemoryError:
            pass
        finally:
            if old is None:
                os.environ.pop("DTFE_PY_RAM_GB", None)
            else:
                os.environ["DTFE_PY_RAM_GB"] = old
        (d / "ps_output.a_vel").unlink()
        assert not ps.has("divergence", mean_field=True)
        try:
            ps.load("divergence", mean_field=True)
            raise AssertionError("a PS mean field without a velocity grid")
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _run_py(args, env_extra=None, timeout=600):
    env = dict(os.environ, MPLBACKEND="Agg", **(env_extra or {}))
    res = subprocess.run([sys.executable] + [str(a) for a in args], capture_output=True, text=True, env=env, timeout=timeout)
    return res.returncode, res.stdout + res.stderr


def t_scripts_default():
    """The figure scripts read PS's mean field by default and the stored grids under '--ps-gradient stored'."""
    plot = REPO / "python" / "plot"
    root = Path(tempfile.mkdtemp(prefix="mf_scripts_"))
    try:
        data, figs = root / "data", root / "figs"
        n, box = 16, 50.0
        d = _snapdir(data / "SYN", 0.0, 0.7, box, n)
        v, _ = analytic_flow(n, box, np.float32, scale=400.0)
        v.tofile(str(d / "ps_output.a_vel"))
        idx = np.indices((n, n, n))[0]
        np.ones((n, n, n), np.float32).tofile(str(d / "ps_output.a_den"))
        np.where(idx == 0, 3.0, 1.0).astype(np.float32).tofile(str(d / "ps_output.a_streams"))
        np.zeros((n, n, n), np.float32).tofile(str(d / "ps_output.a_hidden_streams"))
        stored = (idx * 4 // n).astype(np.float32)                         # a stored PS V-web unlike the mean field
        stored.tofile(str(d / "ps_output.a_velVweb"))
        stored.tofile(str(d / "ps_output.a_tweb"))
        out = root / "ws"
        args = [plot / "plot_web_streams.py", "--sim", "SYN", "--data-root", data, "--snap", "50", "--web", "vweb", "--out", out]
        rc, log = _run_py(args, {"DTFE_FIGURES_ROOT": str(figs)})
        assert rc == 0, log[-1500:]
        t = json.loads((out / "web_streams_vweb_z0.00.json").read_text())
        assert t["vweb_source"] == "mean-field", t.get("vweb_source")
        from dtfelib import FieldSet
        lab = FieldSet(d, method="ps").load("vweb", mean_field=True)
        frac = {nm: float(np.mean(lab == k)) for k, nm in enumerate(("void", "wall", "filament", "node"))}
        assert all(abs(t["classes"][nm]["volume_fraction"] - f) < 1e-12 for nm, f in frac.items() if f > 0), (t["classes"], frac)
        rc, log = _run_py(args + ["--ps-gradient", "stored"], {"DTFE_FIGURES_ROOT": str(figs)})
        assert rc == 0, log[-1500:]
        names = sorted(q.name for q in out.iterdir())
        assert "web_streams_vweb_stored_z0.00.json" in names and "web_streams_vweb_z0.00.json" in names, names
        t2 = json.loads((out / "web_streams_vweb_stored_z0.00.json").read_text())
        assert t2["vweb_source"] == "stored" and abs(t2["classes"]["void"]["volume_fraction"] - 0.25) < 1e-12, t2["classes"]["void"]
        # plot_cosmic_web: the mean-field V-web by default, the stored one asked for
        rc, log = _run_py([plot / "plot_cosmic_web.py", "--sim", "SYN", "--data-root", data, "--snap", "50"],
                          {"DTFE_FIGURES_ROOT": str(figs)})
        assert rc == 0 and "V-web classification + eigenvalues: mean field" in log, log[-1500:]
        rc, log = _run_py([plot / "plot_cosmic_web.py", "--sim", "SYN", "--data-root", data, "--snap", "50",
                           "--ps-gradient", "stored"], {"DTFE_FIGURES_ROOT": str(figs)})
        assert rc == 0 and "Loaded V-web classification: vweb" in log and "mean field" not in log, log[-1500:]
        # plot_shear_triaxial: the summary's data source
        rc, log = _run_py([plot / "plot_shear_triaxial.py", "--sim", "SYN", "--data-root", data, "--snap", "50", "--method", "ps"],
                          {"DTFE_FIGURES_ROOT": str(figs)})
        summ = list(figs.rglob("shear_analysis_summary_z*.txt"))
        assert rc == 0 and summ and "mean field" in summ[0].read_text(), (rc, summ, log[-1500:])
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- the binaries (acceptance)
def _binaries():
    double = os.environ.get("DTFE_TEST_PRECISION", "single") == "double"
    suffix = "-double" if double else ""
    return REPO / f"DTFE{suffix}", REPO / f"PS-DTFE{suffix}", (np.float64 if double else np.float32)


def _zeldovich_snapshot(path, n=32, box=50.0, z=1.0, amp=0.6, rot=60.0, seed=7):
    """A Gadget-HDF5 snapshot (input 105, h-free kpc as the merged TNG files): a Gaussian Zel'dovich displacement
    (rms 'amp' lattice spacings, linear-theory velocities) plus a ROTATIONAL flow rot*(sin ky, sin kz, sin kx) at the
    Eulerian position (u-units), so that every component of the gradient, the vorticity included, is real."""
    import h5py
    rng = np.random.default_rng(seed)
    a = 1.0 / (1.0 + z)
    kf = 2 * np.pi / box
    k1 = np.fft.fftfreq(n, 1.0 / n) * kf
    kz = np.fft.rfftfreq(n, 1.0 / n) * kf
    KX, KY, KZ = np.meshgrid(k1, k1, kz, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K2[0, 0, 0] = 1.0
    phi = (rng.normal(size=K2.shape) + 1j * rng.normal(size=K2.shape)) * K2 ** -1.25 * np.exp(-K2 / (0.35 * n / 2 * kf) ** 2)
    phi[0, 0, 0] = 0.0
    disp = np.stack([np.fft.irfftn(1j * K * phi, s=(n, n, n), axes=(0, 1, 2)) for K in (KX, KY, KZ)], axis=-1)
    spacing = box / n
    disp *= amp * spacing / disp.std()
    c1 = (np.arange(n) + 0.5) * spacing
    Q = np.stack(np.meshgrid(c1, c1, c1, indexing="ij"), axis=-1).reshape(-1, 3)
    D = disp.reshape(-1, 3)
    X = np.mod(Q + D, box)
    H = 70.0 * np.sqrt(0.3 / a ** 3 + 0.7)
    f = (0.3 / a ** 3 / (0.3 / a ** 3 + 0.7)) ** 0.55
    u = a * H * f * D / np.sqrt(a)
    kr = 2 * kf
    u += rot * np.stack([np.sin(kr * X[:, 1]), np.sin(kr * X[:, 2]), np.sin(kr * X[:, 0])], axis=-1)
    with h5py.File(path, "w") as fh:
        hd = fh.create_group("Header").attrs
        npart = np.zeros(6, np.int32)
        npart[1] = n ** 3
        mt = np.zeros(6)
        mt[1] = 1.0
        hd["NumPart_ThisFile"], hd["NumPart_Total"], hd["MassTable"] = npart, npart.astype(np.uint32), mt
        hd["NumFilesPerSnapshot"], hd["BoxSize"], hd["Time"], hd["Redshift"] = np.int32(1), box * 1000.0, a, z
        hd["Omega0"], hd["OmegaLambda"], hd["HubbleParam"], hd["HFreeUnits"] = 0.3, 0.7, 0.7, np.int32(1)
        p = fh.create_group("PartType1")
        p.create_dataset("Coordinates", data=(X * 1000).astype(np.float32))
        p.create_dataset("Velocities", data=u.astype(np.float32))
        p.create_dataset("ParticleIDs", data=np.arange(1, n ** 3 + 1, dtype=np.uint64))
        p.create_dataset("InitialCoordinates", data=(Q * 1000).astype(np.float32))


def _run_bin(cmd, log):
    with open(log, "w") as fh:
        r = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, timeout=1800)
    assert r.returncode == 0, f"{Path(cmd[0]).name} exited {r.returncode}: " + Path(log).read_text()[-1500:]


def _corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def _tmp_root():
    root = Path(os.environ.get("DTFE_TEST_TMP") or Path(os.environ.get("TMPDIR", "/tmp")) / "dtfe-tests") / "meanfield"
    root.mkdir(parents=True, exist_ok=True)
    return root


def t_binary_standard():
    from dtfelib import FieldSet, fields
    dtfe_bin, _, real = _binaries()
    if not os.access(dtfe_bin, os.X_OK):
        raise Skip(f"{dtfe_bin.name} is not built")
    root = _tmp_root()
    for grid in (32, 64):
        d = root / f"std{grid}" / "snapdir_050"
        shutil.rmtree(d.parent, ignore_errors=True)
        d.mkdir(parents=True)
        _zeldovich_snapshot(d / "combined_050.hdf5")
        _run_bin([dtfe_bin, d / "combined_050.hdf5", d / "output", "--grid", grid, "--periodic", "--input", 105,
                  "--padding", 15, "--exact-average", "--field", "density_a", "velocity_a", "gradient_a", "divergence_a",
                  "shear_a", "vorticity_a", "vweb_a", "--lambda_th", 0.3, "--verbose", 1], d / "output.runlog")
        fs = FieldSet(d, method="dtfe")
        assert fs.lambda_th == 0.3 and fs._dtype_of("velocity") == real, (fs.lambda_th, fs._dtype_of("velocity"))
        mf = fs.mean_field(fields.MEAN_FIELD_NAMES)
        assert all(a.dtype == real for nm, a in mf.items() if nm != "vweb")
        st = {nm: fs.load(nm) for nm in fields.MEAN_FIELD_NAMES}
        c_div = _corr(mf["divergence"], st["divergence"])
        rel = float(np.std(mf["divergence"] - st["divergence"]) / np.std(st["divergence"]))
        c_comp = {nm: min(_corr(mf[nm][..., c], st[nm][..., c]) for c in range(st[nm].shape[-1]))
                  for nm in ("gradient", "shear", "vorticity")}
        agree = float(np.mean(mf["vweb"] == st["vweb"]))
        print(f"    standard DTFE {grid}^3: divergence corr {c_div:.4f} (rms difference {100 * rel:.1f}% of its rms), "
              f"lowest component corr gradient {c_comp['gradient']:.4f} shear {c_comp['shear']:.4f} "
              f"vorticity {c_comp['vorticity']:.4f}, V-web labels {100 * agree:.2f}% equal")
        assert c_div >= 0.99 and rel <= 0.15, (grid, c_div, rel)
        assert all(c >= 0.97 for c in c_comp.values()), (grid, c_comp)
        assert agree >= 0.95, (grid, agree)
        assert abs(float(mf["divergence"].mean())) < 1e-4 * float(mf["divergence"].std())
        # the V-web convention itself, exactly: our normalisation + Cardano on the binary's OWN gradient
        g = fs.load("gradient", scaled=False).astype(np.float64).reshape(grid, grid, grid, 3, 3)    # raw u-units/Mpc
        s = -np.sqrt(0.5) / (2 * 70.0) * (g + np.swapaxes(g, -1, -2))       # a = 0.5, h = 0.7
        e = fields.symmetric_eigenvalues3(s[..., 0, 0], s[..., 0, 1], s[..., 0, 2], s[..., 1, 1], s[..., 1, 2], s[..., 2, 2])
        err = float(np.abs(e - st["vweb_eigenvalues"]).max() / np.abs(st["vweb_eigenvalues"]).max())
        same = float(np.mean(fields.classify_web(e.astype(real), 0.3) == st["vweb"]))
        assert err < (1e-5 if real == np.float32 else 1e-10) and same > 0.9999, (grid, err, same)


def t_binary_ps():
    from dtfelib import FieldSet
    dtfe_bin, ps_bin, real = _binaries()
    for b in (dtfe_bin, ps_bin):
        if not os.access(b, os.X_OK):
            raise Skip(f"{b.name} is not built")
    d = _tmp_root() / "ps64" / "snapdir_050"
    shutil.rmtree(d.parent, ignore_errors=True)
    d.mkdir(parents=True)
    _zeldovich_snapshot(d / "combined_050.hdf5", amp=1.2, rot=0.0)           # shell-crossed: ~4% multi-stream cells
    common = ["--grid", 64, "--periodic", "--input", 105, "--padding", 15, "--lambda_th", 0.3, "--verbose", 1]
    _run_bin([ps_bin, d / "combined_050.hdf5", d / "ps_output", *common, "--avg-subsamples", 3, "--ps-volume-weighted",
              "--field", "density_a", "velocity_a", "divergence_a", "vweb_a"], d / "ps_output.runlog")
    _run_bin([dtfe_bin, d / "combined_050.hdf5", d / "output", *common, "--exact-average",
              "--field", "density_a", "velocity_a", "divergence_a", "vweb_a"], d / "output.runlog")
    ps, dt = FieldSet(d, method="ps"), FieldSet(d, method="dtfe")
    multi = float(np.mean(ps.load("streams") > 1.001))
    stored, mf = ps.load("divergence"), ps.load("divergence", mean_field=True)
    assert np.array_equal(mf, ps.mean_field("divergence")) and mf.dtype == real
    assert multi > 0.01 and float(stored.mean()) > 0.1 * float(stored.std()), (multi, stored.mean(), stored.std())
    assert abs(float(mf.mean())) < 1e-4 * float(mf.std()), (mf.mean(), mf.std())
    frac = lambda lab: float(np.mean(np.rint(lab) == 2))
    f_dt, f_st, f_mf = frac(dt.load("vweb")), frac(ps.load("vweb")), frac(ps.load("vweb", mean_field=True))
    print(f"    PS-DTFE 64^3 ({100 * multi:.1f}% multi-stream): divergence box mean stored {float(stored.mean()):+.2f}, "
          f"mean field {float(mf.mean()):+.2g} km/s/Mpc; V-web filaments standard DTFE {100 * f_dt:.2f}%, "
          f"PS stored {100 * f_st:.2f}%, PS mean field {100 * f_mf:.2f}%")
    assert abs(f_mf - f_dt) < abs(f_st - f_dt), (f_dt, f_st, f_mf)


def main():
    import matplotlib
    matplotlib.use("Agg")
    print("synthetic")
    check("analytic band-limited flow: gradient / divergence / shear / vorticity == analytic + DTFE.cpp's component loops (float64, float32)", t_analytic)
    check("V-web: eigenvalues of -sqrt(a)/(2*100h)(g + g^T), descending, labels above lambda_th, the a/h fallbacks", t_vweb_normalisation)
    check("symmetric_eigenvalues3 (the binary's Cardano) against numpy: random, isotropic, the pancake; classify_web", t_cardano)
    check("the spectra plan, the peak estimate, a memmap input, the shape refusals", t_plan_and_memmap)
    check("mean_field_plane (Explore): one plane in slabs == the cube's result cut there, six names, every "
          "axis, faces and slab boundaries, float32 / float64, even / odd size, a minimal slab reader", t_plane)
    check("FieldSet: PS mean field == analytic x sqrt(a), standard DTFE's stored grid kept, has(), the run log's lambda_th, refusals, the memory gate", t_fieldset_api)
    check("figure scripts: plot_web_streams / plot_cosmic_web / plot_shear_triaxial read PS's mean field by default, '--ps-gradient stored' the grids", t_scripts_default)
    print("binary (acceptance)")
    check("standard DTFE: the mean field of its own a_vel reproduces its a_velDiv / gradient / shear / vorticity / V-web (32^3, 64^3)", t_binary_standard)
    check("PS-DTFE: stored divergence box mean > 0, mean field ~0; the mean-field V-web's filaments closer to standard DTFE's", t_binary_ps)
    return summary(blank_line=True, show_zero_skips=False)


if __name__ == "__main__":
    sys.exit(main())
