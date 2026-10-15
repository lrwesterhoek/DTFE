#!/usr/bin/env python3
"""The Explore tab's void features of 2026-10-11, offscreen, on a synthetic snapshot (no T7, no binary):

  1. the void inspector: hovering an outline names the void (R_v, R_eff, delta, flags), clicking it fills the
     inspector (every R_v definition the catalogue has, R_eff, central delta, flags, centre), 'Go to its centre'
     moves the slice there, 'R_eff too' adds dashed outlines;
  2. the outlines carried into the saved PNG (matplotlib circles) with a CSV of the voids shown beside it
     (tools/void_outlines.write_csv), and into the figure grid (each panel's own catalogue, its tiles clipped);
  3. the void smoothing: 'production' or a sigma in Mpc, passed to the helper as --sigma-mpc, its own cache key and
     cache namespace;
  4. the memory guard: no question while no DTFE run is going; with one, a question (free memory said), and a
     'no' starts nothing (server, exact zoom, catalogue); a need that fits in the free memory less 4 GB starts
     without a question even then.
  5. (2026-10-13) the caustic (fold) map of a periodic PS run is the exact fold flag (dtfelib io.caustic_is_fold),
     not the stored per-sample grid of the runs made before then; the caustic-class map stays the stored mask; an
     exact zoom's fold and class maps read its class grid (they read the 0/1 '.caustic': no fold at all).
  6. (2026-10-14) the Save options (figsave.py): the default save is the old one pixel for pixel (and byte for byte);
     full resolution = the data's shape and the screen's pixels, with or without the outlines; no axes, title, colour
     bar, units (Mpc, Mpc/h from the header's h, cells), overlays, margins none, width, dpi, PDF/SVG text kept as
     text, TIFF, transparency, font size; the same for the figure grid (separate and merged panels); the presets,
     the dialog (no '--' in its texts) and the options remembered between sessions in QSettings.
  7. (2026-10-14) the slice kept across outputs and redshifts: another output or redshift of the same simulation
     shows the same axis and place in the box (the nearest cell of its grid: 64^3 and 32^3 outputs, the user's own
     place remembered, not re-read off the coarser grid), and the zoom window (from memory, through the query server,
     or not shown with the reason in the zoom and status lines); another simulation, a custom run or opened file in
     a box of another size start at the middle with the whole box, as the first open does; a 2D output on the way
     leaves the slice and the window to the next 3D output.

The pure parts (grids.void_at, void_circles_mpc, void_info, void_outlines.write_csv) run without Qt; the rest needs
PySide6 and reports SKIP (exit 0) without it. Usage: ~/.venvs/dtfe-gui/bin/python tests/py_gui_explore_test.py
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "gui"))
sys.path.insert(0, str(ROOT / "python" / "tools"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["DTFE_GUI_NO_NOTIFY"] = "1"
TMP = Path(tempfile.mkdtemp(prefix="gui_explore_"))
os.environ["DTFE_FIGURES_ROOT"] = str(TMP / "figures")     # BEFORE runspec is imported: no test figure on the T7

import grids as G  # noqa: E402
import void_outlines as VO  # noqa: E402

PASS, FAIL = [], []
SIM = "SYNEXP-1-Dark"
N, BOX = 64, 64.0


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"   [{detail}]"))


def make_snapdir(data: Path, snap: int = 50, z: float = 0.0, seed: int = 7, sim: str = SIM, box: float = BOX) -> Path:
    """data/SIM/snapdir_NNN: a combined header (box, redshift) and an N^3 PS-DTFE set ('ps_output': density,
    velocity, streams) of six broad Gaussian dips on 2% noise -- voids at the production smoothing (10 cells)."""
    import h5py
    d = data / sim / f"snapdir_{snap:03d}"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / f"combined_{snap:03d}.hdf5", "w") as f:
        h = f.create_group("Header").attrs
        h["BoxSize"], h["Redshift"], h["HubbleParam"], h["HFreeUnits"] = box * 1000.0, float(z), 0.6774, 1
        h["NumPart_ThisFile"], h["MassTable"] = np.array([0, N ** 3, 0, 0, 0, 0]), np.array([0, 1.0, 0, 0, 0, 0])
        h["NumPart_Total"] = np.array([0, N ** 3, 0, 0, 0, 0])
    rho = 1.0 + 0.02 * np.random.default_rng(seed).normal(size=(N, N, N))
    ii = np.indices((N, N, N))
    for c, w, depth in (((16, 16, 32), 6.0, 0.9), ((46, 20, 30), 7.0, 0.85), ((20, 48, 34), 5.0, 0.9),
                        ((50, 50, 28), 8.0, 0.8), ((8, 36, 54), 6.0, 0.85), ((36, 6, 10), 5.0, 0.8)):
        r2 = sum((((ii[k] - c[k] + N // 2) % N) - N // 2) ** 2 for k in range(3))
        rho -= depth * np.exp(-0.5 * r2 / w ** 2)
    rho.astype(np.float32).tofile(str(d / "ps_output.a_den"))
    np.zeros((N, N, N, 3), np.float32).tofile(str(d / "ps_output.a_vel"))
    np.ones((N, N, N), np.float32).tofile(str(d / "ps_output.a_streams"))
    return d


# ---------------------------------------------------------------------- pure parts (no Qt)
def pure():
    print("pure parts")
    circles = [(10.0, 10.0, 5.0, 5.0, 3), (11.0, 10.0, 2.0, 2.0, 7), (40.0, 40.0, 4.0, 2.0, 9)]
    check("void_at: the smallest outline containing the point (a small void inside a big one stays reachable), "
          "None outside, ellipses (non-square pixels) respected",
          G.void_at(circles, 11.5, 10.0) == 7 and G.void_at(circles, 6.0, 10.0) == 3 and G.void_at(circles, 30, 30) is None
          and G.void_at(circles, 43.5, 40.0) == 9 and G.void_at(circles, 40.0, 43.0) is None and G.void_at(None, 1, 1) is None)
    cen = np.array([[10.0, 20.0, 30.0], [62.5, 5.0, 31.0]])
    rad = np.array([4.0, 3.0])
    lo, hi = (0.0, 0.0), (64.0, 64.0)
    px = G.void_circles(cen, rad, 64.0, 2, 30.5, [0, 1], lo, hi, 128, 128)
    mp = G.void_circles_mpc(cen, rad, 64.0, 2, 30.5, [0, 1], lo, hi)
    same = len(px) == len(mp) and all(abs(u - x / 2.0) < 1e-9 and abs(v - (64.0 - y / 2.0)) < 1e-9 and abs(r - rx / 2.0) < 1e-9
                                      and k == kk for (x, y, rx, _ry, k), (u, v, r, kk) in zip(px, mp))
    check("void_circles_mpc = void_circles in the data frame (Mpc, y up), periodic images included",
          same and any(abs(u - 62.5) < 1e-9 for u, *_ in mp) and any(u < 0 for u, *_ in mp), f"{px} {mp}")
    res = {"sim": "S", "snap": 5, "method": "ps", "prefix": "", "box_mpc": 64.0, "n_grid": 64, "sigma_mpc": 2.0,
           "centres_mpc": [[1.5, 2.5, 3.5], [10.0, 11.0, 12.0]], "r_v_mpc": [4.0, None], "r_eff_mpc": [6.5, None],
           "delta": [-0.9, -0.5], "resolved": [True, False], "distinct": [True, False],
           "r_v_defs": {"crossing": [4.0, None], "delta08": [3.0, None], "ridge": [5.5, None]}}
    long, short = G.void_info(res, 0), G.void_info(res, 0, short=True)
    check("void_info: R_v of every definition the catalogue has, R_eff, central delta, flags, centre",
          all(t in long for t in ("4.00 Mpc", "3.00 Mpc", "5.50 Mpc", "6.50 Mpc", "-0.900", "resolved", "distinct",
                                  "1.50, 2.50, 3.50", "ridge")) and "R_v 4.00 Mpc" in short and "--" not in long,
          long)
    check("void_info: an undefined radius says 'none', the flags say 'not'",
          "R_v none" in G.void_info(res, 1, short=True) and "not resolved" in G.void_info(res, 1), G.void_info(res, 1))
    path = VO.write_csv(res, TMP / "pure.csv", rows=[1, 0])
    lines = Path(path).read_text().splitlines()
    head = lines[1].split(",")
    check("write_csv: a '#' provenance line, the header with every R_v definition, the rows asked for in order, "
          "empty fields where undefined",
          lines[0].startswith("# S snapshot 5 ps") and head == ["void", "x_mpc", "y_mpc", "z_mpc", "r_v_mpc",
                                                               "r_v_delta08_mpc", "r_v_ridge_mpc", "r_eff_mpc", "delta",
                                                               "resolved", "distinct"]
          and lines[2].startswith("1,10,11,12,,,,,-0.5,0,0") and lines[3].startswith("0,1.5,2.5,3.5,4,3,5.5,6.5,-0.9,1,1"),
          "\n".join(lines))
    # a cached catalogue whose radii voids() would re-measure is not 'cached' to the launcher, and the memory gate
    # applies (2026-10-11: 'r_v_cells' present was enough, whatever settings measured it): pipeline.catalog_r_v_ok
    from dtfelib import pipeline
    r3 = {k: np.ones(2) for k in ("r_v_cells", "r_v_delta08_cells", "r_v_ridge_cells")}
    old_set = pipeline.r_v_settings() + np.eye(5)[1]           # another R_V_MAX_SIGMA
    for name, arrays in (("today", dict(r3, r_v_settings=pipeline.r_v_settings())), ("older", dict(r3, r_v_settings=old_set)),
                         ("unrecorded", r3), ("no_rv", {})):
        np.savez(TMP / f"rv_{name}.npz", coords=np.zeros((2, 3)), **arrays)

    class Cached:                                               # a SnapshotProducts whose cache passed the grid stamp
        def __init__(self, path):
            self.path = path

        def _cached(self, _name):
            return self.path
    st = {n: VO.r_v_stale(TMP / f"rv_{n}.npz") for n in ("today", "older", "unrecorded", "no_rv")}
    check("r_v_stale / needs_build: the pipeline's own R_v test; older or unrecorded settings are stale and gated",
          st["today"] is None and all(st[n] for n in ("older", "unrecorded", "no_rv"))
          and "older settings" in st["older"] and not any("--" in str(v) for v in st.values())
          and not VO.needs_build(Cached(TMP / "rv_today.npz")) and VO.needs_build(Cached(TMP / "rv_older.npz"))
          and VO.needs_build(Cached(None)), st)
    custom_redshift()
    save_options_pure()


def custom_redshift():
    """2026-10-11: a custom snapshot's velocities in Explore get the sqrt(a) of its Gadget header's redshift, as a
    TNG-layout snapshot's do (custom outputs were always redshift None: u-units, 1/sqrt(a) too large); the Custom tab
    records the redshift in its sidecar, an older sidecar falls back to the input's header, none = no scaling."""
    import json
    import h5py
    import runspec as rs
    d = TMP / "custom_z"
    d.mkdir()
    n = 4
    vel = np.ones((n, n, n, 3), np.float32)
    snaps = {}
    for name, z in (("z1", 1.0), ("noz", None)):
        f = d / f"{name}.hdf5"
        with h5py.File(f, "w") as h:
            hd = h.create_group("Header").attrs
            hd["BoxSize"] = 10.0
            if z is not None:
                hd["Redshift"] = z
        snaps[name] = f
    side = rs.CustomSpec(input_file=str(snaps["z1"]), input_type=105).settings_sidecar()
    check("settings_sidecar records the header's redshift (None without one)",
          side.get("redshift") == 1.0
          and rs.CustomSpec(input_file=str(snaps["noz"]), input_type=105).settings_sidecar().get("redshift") is None,
          str(side.get("redshift")))
    cases = {"new": {"input_file": str(snaps["z1"]), "input_type": 105, "redshift": 1.0},
             "old": {"input_file": str(snaps["z1"]), "input_type": 105},             # a sidecar from before the key
             "noz": {"input_file": str(snaps["noz"]), "input_type": 105, "redshift": None},
             "ascii": {"input_file": str(snaps["z1"]), "input_type": 101}}            # not a Gadget HDF5 input
    for prefix, sc in cases.items():
        np.ones((n, n, n), np.float32).tofile(str(d / f"{prefix}.a_den"))
        vel.tofile(str(d / f"{prefix}.a_vel"))
        (d / f"{prefix}.gui.json").write_text(json.dumps(sc))
    outs = {o.prefix: o for _, o in G.find_outputs(TMP / "no_root", [d])}
    vmax = {k: float(np.nanmax(o.slice("velocity", 2, 1, "x"))) for k, o in outs.items()}
    want = 0.5 ** 0.5
    check("a custom output's redshift: the sidecar's, else its HDF5 input's header; velocities scaled by sqrt(a) "
          "like a TNG snapshot's (z = 1: 0.707); no Redshift, or no HDF5 input: unscaled",
          outs["new"].redshift == 1.0 and outs["old"].redshift == 1.0 and outs["noz"].redshift is None
          and outs["ascii"].redshift is None and abs(vmax["new"] - want) < 1e-6 and abs(vmax["old"] - want) < 1e-6
          and vmax["noz"] == 1.0 and vmax["ascii"] == 1.0,
          f"{ {k: o.redshift for k, o in outs.items()} } {vmax}")


# ---------------------------------------------------------------------- the Save options (2026-10-14)
def save_options_pure():
    """figsave's Qt-free parts: the options checked, the presets, the units, the formats, the dpi matching, the box's h."""
    import json
    import h5py
    import figsave as FS
    print("Save options (pure)")
    o = FS.options({"dpi": 5000, "format": "gif", "frame": "no", "width": -3, "font_scale": 1.5, "unit": "mpc_h",
                    "colour": 1, "title_text": "T", "title_for": "auto"})
    check("options(): out-of-range values clipped, unknown formats and wrong types back to the defaults, unknown keys "
          "dropped, the typed title kept for the session",
          o["dpi"] == FS.DPI_MAX and o["format"] == "png" and o["frame"] is True and o["width"] == 0.0
          and o["font_scale"] == 1.5 and o["unit"] == "mpc_h" and "colour" not in o and o["title_text"] == "T", str(o))
    check("the defaults are what was always written: 200 dpi, a tight crop, PNG, everything on",
          FS.options() == FS.DEFAULTS and FS.save_kwargs(FS.options(), 200) == {"dpi": 200, "bbox_inches": "tight"}
          and FS.rc(FS.options(), "png") == {})
    check("persistent()/dumps(): the typed title is never remembered between sessions; loads() of garbage = defaults",
          "title_text" not in json.loads(FS.dumps(o)) and FS.loads("{not json") == FS.options()
          and FS.loads(None) == FS.options() and FS.loads(json.dumps([1, 2])) == FS.options())
    pub, bare, slide = FS.preset("publication"), FS.preset("bare"), FS.preset("slide")
    check("presets: Publication = everything on, vector PDF or 300 dpi; Bare image = the map alone at full resolution, "
          "no frame, axes, title, bar or overlays; Slide = large fonts; each recognised by which_preset",
          pub["format"] == "pdf" and pub["dpi"] == 300 and all(pub[k] for k in ("title", "frame", "axis_names", "colorbar"))
          and bare["kind"] == "full" and bare["margins"] == "none"
          and not any(bare[k] for k in ("title", "frame", "ticks", "tick_labels", "axis_names", "colorbar", "arrows",
                                        "voids"))
          and slide["font_scale"] > 1.5 and [FS.which_preset(p) for p in (FS.options(), pub, bare, slide)]
          == ["standard", "publication", "bare", "slide"] and FS.which_preset(dict(pub, dpi=301)) is None)
    check("a preset keeps the typed title", FS.preset("bare", {"title_text": "X"})["title_text"] == "X")
    check("title_text(): the typed title only for the map it was typed for",
          FS.title_text({"title_text": "Mine", "title_for": "map A"}, "map A") == "Mine"
          and FS.title_text({"title_text": "Mine", "title_for": "map A"}, "map B") == "map B"
          and FS.title_text({}, "map A") == "map A")
    check("format_of / with_format: the suffix decides (.tif and .tiff are TIFF), a name without one gets the format's",
          [FS.format_of(p) for p in ("a.png", "a.PDF", "a.svg", "a.tif", "a.tiff", "a")] == ["png", "pdf", "svg", "tiff",
                                                                                           "tiff", "png"]
          and FS.with_format("/x/map", "svg").name == "map.svg" and FS.with_format("/x/m.pdf", "png").name == "m.pdf"
          and FS.with_format("/x/m.v2", "tiff").name == "m.v2.tif")

    class Out:
        def __init__(self, box, n=64):
            self.box, self.n = box, n
            self.lo = np.zeros(3) if box is None else np.asarray(box[0], float)
            self.length = np.full(3, float(n)) if box is None else np.asarray(box[1], float) - self.lo
    boxed = Out(((10.0, 20.0, 0.0), (42.0, 52.0, 32.0)))
    um_h, note_h = FS.unit_map(boxed, [0, 1], "mpc_h", 0.7)
    um_c, _ = FS.unit_map(boxed, [0, 1], "cells")
    um_u, note_u = FS.unit_map(boxed, [0, 1], "mpc_h", None)
    um_n, _ = FS.unit_map(Out(None), [0, 1], "mpc")
    check("unit_map: Mpc/h = Mpc x h; cells from the box's low corner (n / length); Mpc/h without a known h is Mpc, "
          "said; no box = cells; Mpc is the identity (the old numbers bit for bit)",
          um_h.name == "Mpc/h" and np.allclose(um_h.ext([10, 42, 20, 52]), [7.0, 29.4, 14.0, 36.4]) and not note_h
          and um_c.name == "cells" and np.allclose(um_c.ext([10, 42, 20, 52]), [0, 64, 0, 64])
          and um_u.name == "Mpc" and "h" in note_u and um_n.name == "cells"
          and FS.unit_map(boxed, [0, 1], "mpc")[0].identity and FS.unit_map(boxed, [0, 1], "mpc")[0].x(1.25) == 1.25)
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7, 6))
    ax = fig.add_axes([0.1, 0.1, 0.5, 0.5])
    check("match_dpi: the dpi at which an axes 3.5 in wide holds 700 data pixels is 200; clipped to 72 to 1200",
          abs(FS.match_dpi(fig, [(ax, 700)]) - 200.0) < 1e-6 and FS.match_dpi(fig, [(ax, 10)]) == FS.DPI_MIN
          and FS.match_dpi(fig, [(ax, 10 ** 6)]) == FS.DPI_MAX)
    # the box's h: a custom run's HDF5 input with the h-free mark gives it, one without the mark does not
    d = TMP / "hubble"
    d.mkdir()
    for name, mark in (("hfree", True), ("raw", False)):
        with h5py.File(d / f"{name}.hdf5", "w") as h:
            a = h.create_group("Header").attrs
            a["BoxSize"], a["HubbleParam"] = 10000.0, 0.7
            if mark:
                a["HFreeUnits"] = 0.7
        np.ones((4, 4, 4), np.float32).tofile(str(d / f"{name}.a_den"))
        (d / f"{name}.gui.json").write_text(json.dumps({"input_file": str(d / f"{name}.hdf5"), "input_type": 105,
                                                        "mpc_unit": 1000.0}))
    outs = {o.prefix: o for _, o in G.find_outputs(TMP / "no_root", [d])}
    check("grids.hubble: an h-free custom input gives its h (Mpc/h offered); a raw one (its box may be Mpc/h already) "
          "and an output without a box give None",
          G.hubble(outs["hfree"]) == 0.7 and G.hubble(outs["raw"]) is None
          and G.hubble(G.OutputSet(d / "raw", box=None)) is None, str({k: G.hubble(o) for k, o in outs.items()}))
    texts = ([t for _k, t in FS.PRESET_NAMES] + list(FS.PRESET_TIPS.values()) + [t for _k, t in FS.MARGINS]
             + [t for _k, t in FS.UNITS] + [t for t, _s in FS.FORMATS.values()] + [t for _f, t in FS.file_filters()]
             + [FS.describe(p) for p in (FS.options(), pub, bare, slide, dict(pub, match_data=True))])
    check("the Save options' texts carry no '--'", not [t for t in texts if "--" in t],
          str([t for t in texts if "--" in t]))


def _old_save_png(x, f):
    """ExploreControls.save_png as it was before the Save options (2026-10-13), verbatim but for 'self': the reference
    the default save must equal pixel for pixel."""
    import explore as E
    req = x._req
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    o = req["output"]
    data = x._plane.T
    if req["log"]:
        data = np.log10(np.where(data > 0, data, np.nan))
    _, _, rest, glo, ghi, _, _ = x._geometry()
    ext = [glo[0], ghi[0], glo[1], ghi[1]]
    fig, ax = plt.subplots(figsize=(7, 6))
    lo, hi = x.view.colorbar.vrange
    im = ax.imshow(data, origin="lower", extent=ext, cmap=req["cmap"], vmin=lo, vmax=hi)
    unit = "Mpc" if o.box is not None else "cells"
    ax.set_xlabel(f"{E.AXES[rest[0]]} [{unit}]")
    ax.set_ylabel(f"{E.AXES[rest[1]]} [{unit}]")
    ax.set_title(x.view.title.text().replace("<b>", "").replace("</b>", ""), fontsize=8)
    fig.colorbar(im, ax=ax, label=x.view.colorbar.label)
    arrows = getattr(x, "_arrows", None)
    if arrows is not None and arrows["vmax"] > 0:
        h = o.length / o.n
        X, Y = np.meshgrid(o.lo[rest[0]] + arrows["c0"] * h[rest[0]],
                           o.lo[rest[1]] + arrows["c1"] * h[rest[1]], indexing="ij")
        ax.quiver(X, Y, arrows["u"], arrows["v"], color="white", edgecolor="black", linewidth=0.3,
                  angles="xy", scale_units="xy", pivot="middle",
                  scale=arrows["vmax"] / (0.9 * arrows["spacing"] * h[rest[0]]), width=0.003)
    res = x._void_result()
    if res is not None and x.view.canvas.circles is not None:
        axis = x._geometry()[0]
        pos = x._slice_position()
        sel = np.nonzero(res["_shown"])[0]
        circ = [(u, v, r, int(sel[k])) for u, v, r, k in G.void_circles_mpc(
            res["_centres"][sel], res["_r_v"][sel], float(res["box_mpc"]), axis, pos, rest, glo, ghi)]
        dash = [] if not x.void_reff.isChecked() else [(u, v, r, int(sel[k])) for u, v, r, k in G.void_circles_mpc(
            res["_centres"][sel], res["_r_eff"][sel], float(res["box_mpc"]), axis, pos, rest, glo, ghi)]
        G.draw_void_circles_mpl(ax, circ, dash)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    fig.savefig(f, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _png(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im)


def _n_circles(fig):
    from matplotlib.patches import Circle
    return sum(isinstance(p, Circle) for ax in fig.axes for p in ax.patches)


def saving_map(qa, x, view):
    """The Save options on the single map (the void outlines and their R_eff shown, as part 2 leaves it): the default
    unchanged, full resolution, no axes, title, colour bar, units, overlays, margins, width, dpi, formats, transparency,
    font size."""
    import matplotlib
    import matplotlib.pyplot as plt
    import figsave as FS
    print("Save options: the map")
    d = TMP / "save map"
    d.mkdir(exist_ok=True)
    n0, n1 = x._plane.shape                          # plane rows = the horizontal axis
    _old_save_png(x, d / "old.png")
    w_new = x.save_png(d / "new.png")
    x.save_png(d / "explicit.png", FS.options())
    old, new = _png(d / "old.png"), _png(d / "new.png")
    check("the default save is the old one: the same pixels as save_png before the Save options (and the same bytes); "
          "the explicit defaults too; the voids' CSV beside it as before",
          old.shape == new.shape and np.array_equal(old, new) and np.array_equal(new, _png(d / "explicit.png"))
          and (d / "old.png").read_bytes() == (d / "new.png").read_bytes() and len(w_new) == 2,
          f"{old.shape} {new.shape} {w_new}")
    # full resolution
    bare = FS.preset("bare")
    w_full = x.save_png(d / "full.png", bare)
    full = _png(d / "full.png")
    check("Bare image: the map alone, one data pixel per image pixel: the image has the data's shape and the screen's "
          "very pixels; no CSV (no outlines drawn)", full.shape == (n1, n0, 3) and np.array_equal(full, view.canvas._buf)
          and w_full == [str(d / "full.png")], f"{full.shape} vs {(n1, n0)}")
    x.save_png(d / "full_voids.png", dict(bare, voids=True))
    fv = _png(d / "full_voids.png")
    changed = (fv[..., :3] != full).any(axis=-1).mean() if fv.shape[:2] == full.shape[:2] else -1
    check("... with the void outlines over it: the same size, the outlines drawn, the rest of the pixels untouched",
          fv.shape[:2] == full.shape[:2] and 0.0 < changed < 0.5 and (fv.shape[2] == 3 or fv[..., 3].min() == 255),
          f"{fv.shape} {changed}")
    x.save_png(d / "full.tif", bare)
    x.save_png(d / "full.pdf", bare)
    x.save_png(d / "full.svg", bare)
    pdf, svg = (d / "full.pdf").read_bytes(), (d / "full.svg").read_text(errors="replace")
    check("... as TIFF (the same pixels), PDF and SVG (the pixels embedded unresampled)",
          np.array_equal(_png(d / "full.tif")[..., :3], full) and pdf.startswith(b"%PDF") and f"/Width {n0}".encode() in pdf
          and "<image" in svg, f"{len(pdf)} {svg[:80]}")
    # the decorated figure, inspected before saving
    fig, _dpi = x.map_figure(FS.options())
    fig.canvas.draw()
    ax = fig.axes[0]
    base_labels = ax.get_xticklabels() + ax.get_yticklabels()
    base_circles, base_title = _n_circles(fig), ax.get_title()
    plt.close(fig)
    fig, _dpi = x.map_figure(FS.options({"frame": False, "ticks": False, "tick_labels": False, "axis_names": False}))
    fig.canvas.draw()
    ax = fig.axes[0]
    labels = [t for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_visible() and t.get_text()]
    check("no axes: no tick labels, no axis names, no frame, ticks of zero length (the default has them all); the "
          "title and the colour bar stay", base_labels and not labels and ax.get_xlabel() == ax.get_ylabel() == ""
          and not any(sp.get_visible() for sp in ax.spines.values())
          and all(t.tick1line.get_markersize() == 0 for t in ax.xaxis.get_major_ticks())
          and ax.get_title() == base_title and len(fig.axes) == 2, f"{labels} {ax.get_xlabel()}")
    plt.close(fig)
    auto = x._map_title()
    titles = []
    for o in ({"title": False}, {"title_text": "A void at z = 0", "title_for": auto},
              {"title_text": "stale", "title_for": "another map"}):
        fig, _ = x.map_figure(FS.options(o))
        titles.append(fig.axes[0].get_title())
        plt.close(fig)
    check("title off: none; a typed title for this map: that title; one typed for another map: the map's own",
          titles == ["", "A void at z = 0", auto] and auto == base_title, str(titles))
    fig, _ = x.map_figure(FS.options({"colorbar": False}))
    n_off = len(fig.axes)
    plt.close(fig)
    fig, _ = x.map_figure(FS.options({"colorbar_label": False}))
    bar_label = fig.axes[1].get_ylabel()
    plt.close(fig)
    check("colour bar off: no bar axes; its label off: the bar without a label", n_off == 1 and bar_label == "",
          f"{n_off} {bar_label!r}")
    _, _, rest, glo, ghi, _, _ = x._geometry()
    xn = "xyz"[rest[0]]
    got = {}
    for unit, extra in (("mpc", {}), ("mpc_h", {}), ("cells", {}), ("mpc", {"units": False})):
        fig, _ = x.map_figure(FS.options(dict(unit=unit, **extra)))
        got[unit + str(extra)] = (fig.axes[0].get_xlabel(), fig.axes[0].get_xlim())
        plt.close(fig)
    h = G.hubble(x.current)
    check("units: Mpc as before; Mpc/h (the snapshot header's h = 0.6774) scales the axes; cells count the grid's "
          "cells; the unit can be left out of the name",
          got["mpc{}"][0] == f"{xn} [Mpc]" and got["mpc_h{}"][0] == f"{xn} [Mpc/h]" and abs(h - 0.6774) < 1e-12
          and np.allclose(got["mpc_h{}"][1], (glo[0] * h, ghi[0] * h)) and got["cells{}"][0] == f"{xn} [cells]"
          and np.allclose(got["cells{}"][1], (0, n0)) and got["mpc{'units': False}"][0] == xn, str(got))
    fig, _ = x.map_figure(FS.options({"voids": False}))
    check("void outlines off: no circles (the default draws them)", base_circles > 0 and _n_circles(fig) == 0,
          f"{base_circles} {_n_circles(fig)}")
    plt.close(fig)
    w = x.save_png(d / "novoids.png", FS.options({"voids": False}))
    check("... and no voids' CSV beside a map saved without them", w == [str(d / "novoids.png")], str(w))
    # sizes: margins, width, dpi
    x.save_png(d / "borderless.png", FS.options({"margins": "none", "dpi": 100}))
    x.save_png(d / "borderless640.png", FS.options({"margins": "none", "width": 640, "width_unit": "px"}))
    x.save_png(d / "w5in.png", FS.options({"width": 5.0, "dpi": 100}))
    x.save_png(d / "w900px.png", FS.options({"width": 900, "width_unit": "px"}))
    x.save_png(d / "dpi100.png", FS.options({"dpi": 100}))
    bl, b640 = _png(d / "borderless.png"), _png(d / "borderless640.png")
    w5, w900, d100 = (_png(d / f).shape[1] for f in ("w5in.png", "w900px.png", "dpi100.png"))
    check("margins none: the map fills the file (7 in at 100 dpi = 700 x 700 for a square box; 640 px asked = 640 x "
          "640), no white edge", bl.shape[:2] == (700, 700) and b640.shape[:2] == (640, 640)
          and bl[0, :, :3].min() < 250, f"{bl.shape} {b640.shape}")
    check("width: the saved file as wide as asked, crop included (5 in at 100 dpi = 500 px, 900 px = 900 px, within "
          "2 px); dpi 100 = half the default's width", abs(w5 - 500) <= 2 and abs(w900 - 900) <= 2
          and abs(d100 - new.shape[1] / 2) <= 3, f"{w5} {w900} {d100} {new.shape}")
    fig, dpi = x.map_figure(FS.options({"match_data": True}))
    plt.close(fig)
    check("match the data: a 64 px map needs less than 72 dpi, so 72", dpi == FS.DPI_MIN, str(dpi))
    # formats, transparency, fonts
    x.save_png(d / "map.pdf", FS.preset("publication"))
    x.save_png(d / "map.svg", FS.options({"format": "svg"}))
    x.save_png(d / "map.tif", FS.options())
    x.save_png(d / "transparent.png", FS.options({"transparent": True}))
    pdf, svg = (d / "map.pdf").read_bytes(), (d / "map.svg").read_text(errors="replace")
    tr = _png(d / "transparent.png")
    check("PDF and SVG keep the text as text (TrueType embedded, SVG <text>); TIFF = the PNG's pixels; transparent: "
          "see-through around the figure", pdf.startswith(b"%PDF") and b"/FontFile2" in pdf and "<text" in svg
          and np.array_equal(_png(d / "map.tif")[..., :3], new[..., :3]) and tr.shape[2] == 4 and tr[0, 0, 3] == 0,
          f"{len(pdf)} {tr.shape}")
    s = FS.preset("slide")
    with matplotlib.rc_context(FS.rc(s)):
        fig, _ = x.map_figure(s)
        size = (fig.axes[0].title.get_fontsize(), fig.axes[0].xaxis.label.get_fontsize())
        wraps = fig.axes[0].title.get_wrap()
    plt.close(fig)
    check("Slide: every text 1.8 times larger (the title 8 -> 14.4 pt, the axis names 10 -> 18 pt); the long title "
          "wraps instead of running over the colour bar's top label (review)",
          abs(size[0] - 14.4) < 1e-9 and abs(size[1] - 18.0) < 1e-9 and wraps, f"{size} {wraps}")
    # (2026-10-14, review) a width or dpi far too large is refused BEFORE matplotlib allocates the picture (50 in at
    # 1200 dpi would be 60000 px across, 12 GB): no file, no figure left open, Save image says why
    import resource
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    refused = []
    for o in ({"width": 50, "width_unit": "in", "dpi": 1200}, {"width": 100000, "width_unit": "px"},
              {"width": 60000, "width_unit": "px", "margins": "none"}):
        try:
            x.save_png(d / "too_big.png", FS.options(o))
        except FS.TooLarge as e:
            refused.append(str(e))
    grew = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0     # bytes on macOS, kB on Linux: both small
    msgs = []

    def said(m):
        msgs.append(m)
    x.status.connect(said)
    real_ask, kept = x._ask_image_path, x.save_options
    x._ask_image_path = lambda *a, **k: (str(d / "too_big_dialog.png"), "png")
    x.save_options = FS.options({"width": 100000, "width_unit": "px"})
    try:
        x._save_image()
    finally:
        x._ask_image_path, x.save_options = real_ask, kept
        x.status.disconnect(said)
    check("a width far too large: refused before drawing (no file, no open figure, no memory taken), and Save image "
          "says so instead of writing the screen", len(refused) == 3 and not (d / "too_big.png").exists()
          and not (d / "too_big_dialog.png").exists() and not plt.get_fignums() and grew < 5e8
          and msgs and "could not save" in msgs[-1] and "pixels" in msgs[-1]
          and not [t for t in refused + msgs if "--" in t], f"{refused} {msgs[-1:]} {grew}")
    for f in ("old_voids.csv", "new_voids.csv", "explicit_voids.csv"):
        (d / f).unlink(missing_ok=True)


def saving_grid(qa, x, view):
    """The Save options on the figure grid (two redshifts side by side, each with its own voids, as part 2 leaves it):
    the default unchanged, full resolution, colour bars, panel titles, the title, margins none, merged panels."""
    import matplotlib.pyplot as plt  # noqa: F401  (the Agg backend)
    import explore as E
    import figsave as FS
    print("Save options: the figure grid")
    d = TMP / "save grid"
    d.mkdir(exist_ok=True)
    x.save_grid(d / "default.png")
    x.save_grid(d / "explicit.png", FS.options())
    fig0, _ = x.grid_figure()
    check("the grid's default save: the explicit defaults give the same pixels; the figure keeps its two panels and "
          "two colour bars", np.array_equal(_png(d / "default.png"), _png(d / "explicit.png")) and len(fig0.axes) == 4)
    req = x.grid_request()
    items = E.grid_items(req)
    T = max(it["img"].shape[0] for it in items if it is not None)
    tiles, _l, _b = E.grid_tiles(req, items, max_tile=T)
    x.save_grid(d / "full.png", FS.preset("bare"))
    x.save_grid(d / "full_voids.png", dict(FS.preset("bare"), voids=True))
    full, fv = _png(d / "full.png"), _png(d / "full_voids.png")
    check("full resolution: the panels at their own pixels with the screen's gap (1 x 2 of 64: 64 x 134), the "
          "screen's colours; with each panel's outlines over it", full.shape == (T, 2 * T + 6, 3)
          and np.array_equal(full, tiles) and fv.shape[:2] == full.shape[:2] and (fv[..., :3] != full).any(),
          f"{full.shape}")
    fig, _ = x.grid_figure(FS.options({"colorbar": False, "panel_titles": False, "title": False}))
    titles = [ax.get_title() for ax in fig.axes]
    check("no colour bars, no panel titles, no title: two panel axes, no titles, no suptitle",
          len(fig.axes) == 2 and titles == ["", ""] and (fig._suptitle is None or not fig._suptitle.get_text()),
          f"{len(fig.axes)} {titles}")
    fig, _ = x.grid_figure(FS.options({"title_text": "Two redshifts", "title_for": x._grid_title}))
    check("a title typed for this grid replaces its own (the worker's title)", fig._suptitle is not None
          and fig._suptitle.get_text() == "Two redshifts" and x._grid_title.startswith("slice normal"),
          f"{x._grid_title!r}")
    x.save_grid(d / "borderless.png", FS.options({"margins": "none", "dpi": 100}))
    x.save_grid(d / "grid.svg", FS.options())
    x.save_grid(d / "w1500px.png", FS.options({"width": 1500, "width_unit": "px", "font_scale": 1.3}))
    w1500 = _png(d / "w1500px.png").shape[1]
    check("the grid's width: 1500 px asked (larger fonts), 1500 px written within 2", abs(w1500 - 1500) <= 2, str(w1500))
    bl = _png(d / "borderless.png")
    check("margins none: the panels fill the file (2 x 4.2 in by 4.2 in at 100 dpi = 840 x 420); the SVG keeps its "
          "text", bl.shape[:2] == (420, 840) and "<text" in (d / "grid.svg").read_text(errors="replace"), f"{bl.shape}")
    x.grid_merge.setChecked(True)
    ok = wait_for(qa, lambda: x._grid_shown and not x._grid_busy and x.grid_request()["merged"], 30)
    fig, _ = x.grid_figure(FS.options({"colorbar": False, "panel_titles": False, "tick_labels": False}))
    fig.canvas.draw()
    labels = [t for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_visible() and t.get_text()]
    check("merged: without the colour bar column, the inside labels and the tick labels: two axes, no texts inside, "
          "no tick labels", ok and len(fig.axes) == 2 and not any(ax.texts for ax in fig.axes) and not labels,
          f"{len(fig.axes)} {labels[:3]}")
    try:                                                    # (2026-10-14, review) the grid's size guard
        x.save_grid(d / "too_big.png", FS.options({"width": 30000, "width_unit": "px"}))
        refused = ""
    except FS.TooLarge as e:
        refused = str(e)
    check("the grid: a width far too large is refused before drawing (no file)", "pixels" in refused
          and not (d / "too_big.png").exists(), refused)
    x.save_grid(d / "merged_full.png", FS.preset("bare"))
    check("merged at full resolution: no gap between the panels (64 x 128)", _png(d / "merged_full.png").shape == (T, 2 * T, 3))
    x.grid_merge.setChecked(False)
    wait_for(qa, lambda: x._grid_shown and not x._grid_busy and not x.grid_request()["merged"], 30)


def saving_dialog(qa, x):
    """The Save options dialog: presets fill it, an edit makes it 'Custom', 'Your last choice' goes back, Mpc/h only
    with a known h, no '--' in any text; open_save_options keeps what OK says."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractButton, QComboBox, QDialog, QGroupBox, QLabel, QLineEdit
    import figsave as FS
    print("Save options: the dialog")
    ctx = x.save_context()
    dlg = FS.make_dialog(x, x.save_options, ctx)
    first = dlg.preset.currentData()
    dlg.preset.setCurrentIndex(dlg.preset.findData("bare"))
    bare = dlg.options()
    dlg.frame.setChecked(not dlg.frame.isChecked())
    after_edit = dlg.preset.currentData()
    dlg.preset.setCurrentIndex(dlg.preset.findData("last"))
    back = dlg.options()
    mpc_h = dlg.unit.model().item(dlg.unit.findData("mpc_h")).isEnabled()
    check("dialog: opens on the preset the options are, a preset fills every option, an edit makes it Custom, 'Your "
          "last choice' brings the opened options back; Mpc/h offered (the header's h is known); the map's size in "
          "the full-resolution choice", first == "standard" and FS.persistent(bare) == FS.persistent(FS.preset("bare"))
          and after_edit == "custom" and FS.persistent(back) == FS.persistent(x.save_options) and mpc_h
          and f"{x._plane.shape[0]} × {x._plane.shape[1]} px" in dlg.kind_full.text(),
          f"{first} {after_edit} {mpc_h} {dlg.kind_full.text()}")
    dlg.kind_full.setChecked(True)
    greyed = not dlg.dpi.isEnabled() and not dlg.title.isEnabled() and not dlg.frame.isEnabled() and dlg.voids.isEnabled()
    dlg.kind_figure.setChecked(True)
    dlg.margins.setCurrentIndex(dlg.margins.findData("none"))
    greyed2 = not dlg.title.isEnabled() and not dlg.colorbar.isEnabled() and dlg.dpi.isEnabled()
    check("full resolution greys the dpi, the title and the axes (the overlays stay); no margins greys the title, axes "
          "and bar", greyed and greyed2)
    texts = [dlg.windowTitle()]
    for wd in dlg.findChildren(QLabel) + dlg.findChildren(QAbstractButton) + dlg.findChildren(QGroupBox) \
            + dlg.findChildren(QLineEdit) + dlg.findChildren(QComboBox):
        texts += [getattr(wd, "text", lambda: "")() if not isinstance(wd, QGroupBox) else wd.title(), wd.toolTip()]
        if isinstance(wd, QLineEdit):
            texts.append(wd.placeholderText())
        if isinstance(wd, QComboBox):
            texts += [wd.itemText(i) for i in range(wd.count())]
            texts += [str(wd.itemData(i, Qt.ToolTipRole) or "") for i in range(wd.count())]
    for wd in (x.view.save_btn, x.view.save_opts_btn, x.grid_save_btn, x.grid_opts_btn):
        texts += [wd.text(), wd.toolTip()]
    check("the dialog's and the save buttons' texts carry no '--'", not [t for t in texts if "--" in t],
          str([t for t in texts if "--" in t][:3]))
    dlg.deleteLater()
    nohub = FS.make_dialog(x, x.save_options, dict(ctx, hubble=None))
    check("without a known h, Mpc/h is not offered", not nohub.unit.model().item(nohub.unit.findData("mpc_h")).isEnabled())
    nohub.deleteLater()
    real = FS.make_dialog
    saved = []

    def fake(parent, opts, context=None, can_save=True):
        d_ = real(parent, opts, context, can_save)
        d_.exec = lambda: (d_.preset.setCurrentIndex(d_.preset.findData("publication")), d_.accept(),
                           QDialog.Accepted)[-1]
        return d_
    FS.make_dialog = fake
    real_save = x._save_image
    x._save_image = lambda: saved.append(1)
    try:
        ok = x.open_save_options()
    finally:
        FS.make_dialog = real
        x._save_image = real_save
    check("open_save_options: OK keeps the options chosen (Publication) for every save, without saving now; the save "
          "button's tooltip says so", ok and FS.which_preset(x.save_options) == "publication" and not saved
          and "Publication" in x.view.save_btn.toolTip(), x.view.save_btn.toolTip())
    x.set_save_options(FS.options(), remember=False)


def save_settings():
    """The Save options are remembered between sessions in the launcher's QSettings (without a typed title); none
    without settings (the tests' and remember=False windows)."""
    try:
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (save settings): PySide6 is not importable ({e})")
        return
    import json
    import explore as E
    import figsave as FS
    print("Save options: remembered")
    QApplication.instance() or QApplication([])
    ini = TMP / "launcher.ini"
    made = []

    def controls(settings):
        c = E.ExploreControls(E.ExploreView(), lambda: str(TMP / "no data"), lambda: [], settings=settings)
        made.append(c)
        return c
    try:
        x1 = controls(QSettings(str(ini), QSettings.IniFormat))
        fresh = dict(x1.save_options)
        want = dict(FS.preset("slide"), dpi=150, unit="cells", title_text="mine", title_for="t")
        x1.set_save_options(want)
        x1._settings.sync()
        x2 = controls(QSettings(str(ini), QSettings.IniFormat))
        raw = QSettings(str(ini), QSettings.IniFormat).value(E.ExploreControls.SAVE_KEY, "")
        check("nothing remembered: the defaults; set once, a new session (a new window) has them, the typed title not",
              fresh == FS.options() and x2.save_options == FS.persistent(want) and "title_text" not in x2.save_options
              and "mine" not in str(raw) and json.loads(raw)["dpi"] == 150, str(raw))
        st = QSettings(str(ini), QSettings.IniFormat)
        st.setValue(E.ExploreControls.SAVE_KEY, json.dumps({"dpi": 99999, "format": "gif", "frame": "no"}))
        st.sync()
        x3 = controls(QSettings(str(ini), QSettings.IniFormat))
        st.setValue(E.ExploreControls.SAVE_KEY, "{not json")
        st.sync()
        x4 = controls(QSettings(str(ini), QSettings.IniFormat))
        check("a remembered value out of range or of the wrong kind is checked (dpi 1200 at most, PNG, the frame on); "
              "unreadable settings give the defaults", x3.save_options["dpi"] == FS.DPI_MAX
              and x3.save_options["format"] == "png" and x3.save_options["frame"] is True and x4.save_options == FS.options())
        before = ini.read_bytes()
        x5 = controls(None)
        x5.set_save_options(FS.preset("bare"))
        check("without settings (remember off) the options live for the session only: nothing written",
              ini.read_bytes() == before and x5.save_options["kind"] == "full")
    finally:
        for c in made:
            c.shutdown()


# ---------------------------------------------------------------------- the GUI
def spin(qa, ms):
    end = time.time() + ms / 1000.0
    while time.time() < end:
        qa.processEvents()
        time.sleep(0.01)


def wait_for(qa, cond, limit):
    end = time.time() + limit
    while time.time() < end:
        qa.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def gui():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (GUI part): PySide6 is not importable ({e})")
        return
    import explore as E
    import runspec as rs
    print("GUI")
    qa = QApplication.instance() or QApplication([])
    data = TMP / "data"
    make_snapdir(data, 50, 0.0, 7)
    make_snapdir(data, 99, 0.0, 8)
    E.VOID_CACHE_DIR = TMP / "void cache"
    view = E.ExploreView()
    x = E.ExploreControls(view, lambda: str(data), lambda: [])
    view.resize(900, 800)
    asked = []
    x.confirm_heavy = lambda title, text: asked.append((title, text)) or False
    real_procs = rs.dtfe_processes
    rs.dtfe_processes = lambda max_age=2.0: {"runs": [], "servers": []}      # no DTFE run going
    try:
        x.refresh_outputs()
        ok = wait_for(qa, lambda: not x._busy and x.current is not None and x._plane is not None, 30)
        check("the synthetic snapshot is shown", ok and x.current.sim == SIM, str(x.current and x.current.root))
        x.voids.setChecked(True)
        ok = wait_for(qa, lambda: x._void_proc is None and bool(x._void_cache), 180)
        res = next(iter(x._void_cache.values()), {})
        check("voids ticked with no run going: the helper runs without a question; its result carries R_eff and "
              "the R_v definitions", ok and "error" not in res and not asked and "r_v_defs" in res
              and "crossing" in res["r_v_defs"] and list(x._void_cache)[0][-1] == 0.0,
              str(res.get("error", ""))[:300])
        c = view.canvas
        # the slice through a void's centre: there the outline is biggest
        sel = [k for k in np.nonzero(res.get("_shown", []))[0] if np.isfinite(res["_r_v"][k])]
        check("the fixture has distinct voids with R_v", len(sel) > 0, f"{len(sel)}")
        if not sel:
            return
        k0 = int(max(sel, key=lambda k: res["_r_v"][k]))
        axis = x.axis.currentIndex()
        i0 = int(res["_centres"][k0][axis] / (BOX / N))
        x.index.setValue((i0 + 5) % N)
        wait_for(qa, lambda: not x._busy and x._req["index"] == (i0 + 5) % N, 30)
        hit = next(((cx, cy, kk) for cx, cy, _rx, _ry, kk in c.circles or [] if kk == k0), None)
        check("an outline of the biggest void on a slice near its centre", hit is not None, f"{c.circles}")
        if hit is None:
            return
        r, col = int(hit[1]), int(hit[0])
        x._hover(r, col)
        check("1. hover over the outline: the readout names the void (R_v, R_eff, delta)",
              f"void {k0}: R_v" in view.readout.text() and "R_eff" in view.readout.text(), view.readout.text())
        x._click(r, col)
        info = x.void_info.text()
        check("1. click: the inspector shows R_v, R_eff, the central delta, the flags and the centre; Go to its "
              "centre enabled", f"void {k0}" in info and "R_eff" in info and "central δ" in info and "distinct" in info
              and "centre (" in info and x.void_goto.isEnabled(), info)
        x._void_goto()
        ok = wait_for(qa, lambda: not x._busy and x._req["index"] == i0, 30)
        check("1. Go to its centre: the slice through the void's centre cell", ok and x.index.value() == i0,
              f"{x.index.value()} vs {i0}")
        x.void_reff.setChecked(True)
        qa.processEvents()
        check("1. R_eff too: dashed outlines over the map, the same voids", bool(c.dashed)
              and c.dashed == x._void_circles("_r_eff") and {k for *_a, k in c.dashed} <= set(sel) | set(
                  np.nonzero(res["_shown"])[0]), f"{c.dashed}")
        c._redraw()                                         # draws both kinds without error
        # ---- 2. the saved PNG and its CSV
        png = TMP / "map.png"
        written = x.save_png(png)
        csvp = TMP / "map_voids.csv"
        rows = x.voids_shown()
        lines = csvp.read_text().splitlines() if csvp.is_file() else []
        ids = [int(ln.split(",")[0]) for ln in lines[2:]]
        check("2. Save image: the PNG and '<name>_voids.csv' with exactly the voids outlined (centre, R_v, R_eff, "
              "delta, flags)", png.is_file() and png.stat().st_size > 10000 and written == [str(png), str(csvp)]
              and ids == rows and k0 in ids and lines[1].startswith("void,x_mpc,y_mpc,z_mpc,r_v_mpc")
              and "r_eff_mpc,delta,resolved,distinct" in lines[1], f"{written} {ids} {rows}")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        orig_close = plt.close
        figs = []
        plt.close = lambda fig=None: figs.append(fig)       # keep the figure to count its circles
        try:
            x.save_png(TMP / "map2.png")
        finally:
            plt.close = orig_close
        fig = figs[0] if figs else None
        n_patch = sum(isinstance(p, Circle) for ax in (fig.axes if fig else []) for p in ax.patches)
        check("2. ... the PNG carries the outlines as on screen (two strokes per circle: R_v and R_eff)",
              fig is not None and n_patch == 2 * (len(c.circles) + len(c.dashed)), f"{n_patch} vs {len(c.circles)}+{len(c.dashed)}")
        if fig is not None:
            orig_close(fig)
        # ---- 2. the figure grid: two snapshots side by side, each with its own catalogue
        x.grid_set_layout(1, 2)
        x.grid_fill("redshifts")
        x.show_grid(True)
        ok = wait_for(qa, lambda: x._void_proc is None and len(x._void_cache) >= 2 and not x._grid_busy
                      and x._grid_pending is None and bool(c.circles), 240)
        keys = [k for k in x._void_cache if "error" not in x._void_cache[k]]
        clips = c.circle_clips or []
        check("2. the figure grid: each panel's own catalogue (found one after the other), its outlines clipped to "
              "its tile", ok and len(keys) == 2 and len(clips) == len(c.circles) + len(c.dashed or [])
              and len({cl[0] for cl in clips}) == 2, f"{len(x._void_cache)} {len(clips)}")
        fig, _n = x.grid_figure()
        n_patch = sum(isinstance(p, Circle) for ax in fig.axes for p in ax.patches)
        check("2. ... and the saved grid figure draws them (matplotlib circles in every panel)",
              n_patch > 0 and all(any(isinstance(p, Circle) for p in ax.patches) for ax in fig.axes[:2]), f"{n_patch}")
        saving_grid(qa, x, view)                            # 6. the Save options on the grid, then on the map
        x.show_grid(False)
        wait_for(qa, lambda: not x._busy and not x._grid_shown, 30)
        saving_map(qa, x, view)
        saving_dialog(qa, x)
        # ---- 3. the smoothing
        n_before = len(x._void_cache)
        x.void_sigma.setValue(6.0)
        ok = wait_for(qa, lambda: x._void_proc is None and len(x._void_cache) > n_before, 180)
        key = x._void_cache_key(x.current)
        r6 = x._void_cache.get(key, {})
        r0 = x._void_cache.get(x._void_cache_key(x.current, 0.0), {})
        check("3. a sigma in Mpc: its own cache key, the helper finds the voids at that sigma under its own namespace",
              ok and key[-1] == 6.0 and "error" not in r6 and abs(r6.get("sigma_mpc", 0) - 6.0) < 1e-6
              and r6.get("namespace") != r0.get("namespace") and r0.get("sigma_asked_mpc") == 0.0
              and any((TMP / "void cache").glob("*s6_*")), f"{key} {str(r6.get('error', ''))[:200]} "
              f"{r6.get('namespace')} {r0.get('namespace')}")
        st = x.state()
        check("3. the voids tick, R_eff and sigma are remembered", st.get("voids") is True and st.get("void_reff") is True
              and st.get("void_sigma") == 6.0, f"{st}")
        x.void_sigma.setValue(0.0)
        qa.processEvents()
        check("3. back to production: the cached catalogue, no new search", x._void_proc is None
              and x._void_cache_key(x.current) in x._void_cache)
        # ---- 4. the memory guard
        rs.dtfe_processes = lambda max_age=2.0: {"runs": [(4242, "PS-DTFE", 37.0)], "servers": []}
        asked.clear()
        x._voids_request(x._by_root[next(r for r in x._by_root if "snapdir_099" in r)])   # already cached: no question
        check("4. a cached catalogue is never asked about", not asked)
        os.environ["DTFE_FREE_GB"] = "2"                    # tight: the 21 MB build does not fit in 2 - 4 GB
        x.void_sigma.setValue(3.0)
        qa.processEvents()
        k3 = x._void_cache_key(x.current)
        check("4. a DTFE run going: before finding the voids the launcher asks (the run, its memory, what is free), "
              "and a 'no' starts nothing", len(asked) == 1 and "PS-DTFE pid 4242, 37.0 GB" in asked[0][1]
              and "free" in asked[0][1] and x._void_proc is None and "error" in x._void_cache.get(k3, {}),
              f"{asked} {x._void_cache.get(k3)}")
        asked.clear()
        os.environ["DTFE_FREE_GB"] = "30"
        check("4. a run going but the need fits in the free memory less the headroom: no question, started (said)",
              x.heavy_ok("Start the query server", "A query server", 12.0) and not asked)
        os.environ["DTFE_FREE_GB"] = "8"
        ok_heavy = x.heavy_ok("Start the query server", "A query server", 12.0)
        check("4. heavy_ok: asked with the need in GB, the answer obeyed", not ok_heavy and len(asked) == 1
              and "about 12 GB" in asked[0][1] and "--" not in asked[0][1], f"{asked}")
        x.confirm_heavy = lambda title, text: True
        check("4. ... a 'yes' lets it start", x.heavy_ok("Exact zoom", "it", None))
        rs.dtfe_processes = lambda max_age=2.0: {"runs": [], "servers": [(1, "PS-DTFE", 5.0)]}
        x.confirm_heavy = lambda title, text: asked.append(text) or False
        asked.clear()
        check("4. no run going (a server alone is not a run): never asks, never blocks",
              x.heavy_ok("Exact zoom", "it", 50.0) and not asked)
        line = x.memory_status()
        check("4. the memory status line: free GB and the server with its memory", "free" in line
              and "server pid 1, 5.0 GB" in line, line)
        check("launcher texts carry no '--'", all("--" not in t for t in (E.VOID_TIP, E.VOID_REFF_TIP, E.VOID_SIGMA_TIP,
                                                                           x.void_goto.toolTip())))
    finally:
        rs.dtfe_processes = real_procs
        os.environ.pop("DTFE_FREE_GB", None)
        x.shutdown() if hasattr(x, "shutdown") else None


def fold_map():
    """Integration item 9 (2026-10-13): Explore's 'caustic (fold) cells' map of a periodic PS run is the exact fold
    flag, dtfelib.io.caustic_is_fold(class mask, streams=, hidden=) -- the stored '.caustic' of the grids made before
    2026-10-13 was set only where a folded tetrahedron deposited a sample and missed every fold thinner than the
    sampling -- in the map, the click readout and its tooltip; the caustic-class map stays the stored mask."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (fold map): PySide6 is not importable ({e})")
        return
    import explore as E
    from dtfelib.io import caustic_is_fold
    print("fold map")
    qa = QApplication.instance() or QApplication([])
    data = TMP / "fold data"
    d = make_snapdir(data, 33, 0.0, 3)
    st = np.ones((N, N, N), np.float32)
    st[:, :, 20:22] = 3.0                           # a fold sheet the samples saw (three streams)
    hid = np.zeros((N, N, N), np.float32)
    hid[:, 10, :] = 1.0                             # a fold thinner than the sampling: hidden-streams bit 1 only
    hid[5, :, :] = 2.0                              # bit 2 alone (an unsampled tetrahedron's mass): no fold
    raw = np.zeros((N, N, N), np.float32)
    raw[:, :, 20] = 1.0                             # the pre-fix stored flag: half the sheet, none of the thin fold
    ccm = np.full((N, N, N), 1 + 4, np.float32)
    for sfx, arr in (("a_streams", st), ("a_hidden_streams", hid), ("caustic", raw), ("causticClass", ccm)):
        arr.tofile(str(d / f"ps_output.{sfx}"))
    want = caustic_is_fold(ccm, streams=st, hidden=hid)
    view = E.ExploreView()
    x = E.ExploreControls(view, lambda: str(data), lambda: [])
    view.resize(600, 600)
    try:
        x.refresh_outputs()
        wait_for(qa, lambda: not x._busy and x.current is not None and x._plane is not None, 30)
        x.field.setCurrentIndex(x.field.findData("caustic"))
        ok = wait_for(qa, lambda: not x._busy and x._req is not None and x._req["field"] == "caustic", 30)
        ax, k = (x._req["axis"], x._req["index"]) if ok else (2, 0)
        plane = np.asarray(x._plane) if ok else None
        exp = np.take(want, k, axis=ax).astype(np.float64)
        tip = x.field.itemData(x.field.findData("caustic"), Qt.ToolTipRole) or ""
        check("the caustic map of a periodic PS run is the exact fold flag (the thin fold and the whole sheet), not "
              "the stored per-sample grid; its tooltip says so",
              ok and plane is not None and plane.shape == exp.shape and np.array_equal(plane, exp)
              and not np.array_equal(plane, np.take(raw, k, axis=ax)) and "2026-10-13" in tip and "--" not in tip,
              f"{ok} {ax} {k} {None if plane is None else (plane.sum(), exp.sum())}")
        o = x.current
        check("... the click readout too (a thin-fold cell reads 1, a bit-2-only cell 0)",
              o.value("caustic", (3, 10, 40)) == 1.0 and o.value("caustic", (5, 30, 40)) == 0.0
              and o.value("caustic", (7, 30, 21)) == 1.0 and float(raw[7, 30, 21]) == 0.0)
        x.field.setCurrentIndex(x.field.findData("caustic_class"))
        ok = wait_for(qa, lambda: not x._busy and x._req is not None and x._req["field"] == "caustic_class", 30)
        check("... and the caustic-class map stays the stored mask",
              ok and np.array_equal(np.asarray(x._plane), np.take(ccm, x._req["index"], axis=x._req["axis"])))
        # the exact zoom (load_exact_planes): a new run's grids, one cell thick. Until 2026-10-13 its fold map masked
        # both parity bits of the 0/1 '.caustic' grid, so it showed no fold cell at all, and its class map showed 0/1
        import types
        zd = TMP / "fold zoom"
        zd.mkdir(parents=True, exist_ok=True)
        zg = (8, 8, 1)
        zc = np.zeros(zg, np.float32)
        zc[:3] = 1.0                                    # the fold cells (the fixed binary's exact flag)
        zcc = np.full(zg, 1 + 4, np.float32)
        zcc[:3] = 1 + 2 + 8
        for sfx, arr in (("a_den", np.ones(zg, np.float32)), ("caustic", zc), ("causticClass", zcc)):
            arr.tofile(str(zd / f"zoom.{sfx}"))

        def zoom_maps():
            stub = types.SimpleNamespace(_zoom_fields=E.load_exact_planes(zd / "zoom", zg),
                                         _zoom={"axis": 2, "output": types.SimpleNamespace(dim=3)})
            return (E.ExploreControls._zoom_plane(stub, "caustic", "norm"),
                    E.ExploreControls._zoom_plane(stub, "caustic_class", "norm"))
        zfold, zcls = zoom_maps()
        (zd / "zoom.causticClass").unlink()
        zfold1, _ = zoom_maps()
        check("the exact zoom's fold map is its '.caustic' and its class map the class mask (both read the 0/1 "
              "grid: no fold cell at all); a zoom without the class grid still shows its folds",
              np.array_equal(zfold, zc[:, :, 0]) and np.array_equal(zcls, zcc[:, :, 0])
              and np.array_equal(zfold1, zc[:, :, 0]), f"{zfold.sum()} {zcls.max()} {zfold1.sum()}")
    finally:
        x.shutdown() if hasattr(x, "shutdown") else None


def write_grids(d: Path, prefix: str, n: int, dim: int = 3, seed: int = 0):
    """A plain output 'd/prefix.*': density (noise around 1), velocity (zero) and stream counts on n^dim cells."""
    shape = (n,) * dim
    (1.0 + 0.1 * np.random.default_rng(seed).normal(size=shape)).astype(np.float32).tofile(str(d / f"{prefix}.a_den"))
    np.zeros(shape + (dim,), np.float32).tofile(str(d / f"{prefix}.a_vel"))
    np.ones(shape, np.float32).tofile(str(d / f"{prefix}.a_streams"))


def keep_slice():
    """7. (2026-10-14, Luuk's request) switching the output or the redshift keeps the slice (the axis and the place in
    the box, at the nearest cell of the new grid) and the zoom window; switching the simulation resets both."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (kept slice): PySide6 is not importable ({e})")
        return
    import json
    import types
    import explore as E
    print("the slice kept across outputs and redshifts")
    qa = QApplication.instance() or QApplication([])
    check("7. kept_index: a cell centre gives its own cell back, a cell face the upper cell, clamped to the grid",
          [E.kept_index((i + 0.5) / 64, 64) for i in (0, 10, 63)] == [0, 10, 63] and E.kept_index(7.5 / 32, 64) == 15
          and E.kept_index(10.5 / 64, 32) == 5 and E.kept_index(1.0, 8) == 7 and E.kept_index(-0.2, 8) == 0)
    data, custom, opened = TMP / "keep data", TMP / "keep custom", TMP / "keep opened"
    s50 = make_snapdir(data, 50, 1.0, 11)                 # z = 1: ps_output 64^3 and a standard 'output' 32^3
    write_grids(s50, "output", 32)
    make_snapdir(data, 99, 0.0, 12)                       # z = 0: ps_output 64^3
    sim2 = SIM.replace("-1-", "-2-")
    make_snapdir(data, 99, 0.0, 13, sim=sim2, box=32.0)   # another simulation, a 32 Mpc box

    def run(d_, name, n, dim=3, box=BOX, side=True):
        d_.mkdir(parents=True, exist_ok=True)
        write_grids(d_, name, n, dim)
        if side:
            (d_ / f"{name}.gui.json").write_text(json.dumps({"dim": dim, "box_mpc": [0.0, box] * dim, "redshift": None}))
        return d_ / name
    c3a, c2d, c3b, c3s = run(custom, "c3a", 64), run(custom, "c2d", 64, 2), run(custom, "c3b", 32), run(custom, "c3s", 32, box=32.0)
    o1, o2, o3 = run(opened, "o1", 64), run(opened, "o2", 32), run(opened, "o3", 16, side=False)
    view = E.ExploreView()
    x = E.ExploreControls(view, lambda: str(data), lambda: [str(custom)])
    view.resize(700, 600)
    said = []
    x.status.connect(said.append)
    lines = []                                            # every zoom line shown, for the '--' rule

    def shown(root=None):
        ok_ = wait_for(qa, lambda: not x._busy and x.current is not None and x._req is not None
                       and x._req["output"] is x.current and (root is None or str(x.current.root) == str(root)), 30)
        lines.append(x.zoom_state.text())
        return ok_

    def at():
        return x.axis.currentIndex(), x.index.value()

    def pick(combo, data_):
        combo.setCurrentIndex(combo.findData(data_))
        return shown()

    def fields(val, ax=None):
        shp = [8, 8, 8]
        shp[x.axis.currentIndex() if ax is None else ax] = 1      # the slice axis is squeezed out
        return types.SimpleNamespace(density=np.full(shp, float(val)), streams=None, caustic=None, velocity=None,
                                     dispersion=None, sigma=None, velocity_gradient=None, offsets=None)

    def zoom_req(o=None):
        o = o or x.current
        ax, i = x.axis.currentIndex(), x.index.value()
        return {"output": o, "axis": ax, "index": i, "rest": [a for a in range(3) if a != ax], "a0": (20.0, 30.0),
                "a1": (40.0, 50.0), "res": 8, "depth": 1, "stamp": x._file_stamp(o)}

    def memo(o, ax, i, f):                                # a zoom of 'o' made before (the window, slice ax/i)
        req = dict(zoom_req(o), axis=ax, index=i, rest=[a for a in range(3) if a != ax])
        x._memo_put(x._memo_key(o, ax, i, req["a0"], req["a1"], 8, False), req, f)

    try:
        x.refresh_outputs()
        ok = shown()
        check("7. the first open: the latest snapshot of the first simulation, at the middle of the box",
              ok and x.current.sim == SIM and x.current.snap == 99 and x.index.value() == N // 2, f"{at()}")
        x.axis.setCurrentIndex(0)
        x.index.setValue(10)
        shown()
        ok = pick(x.snap_combo, 50)
        check("7. another redshift: the same axis and the same place (x = 10.50 Mpc, the same grid)",
              ok and x.current.snap == 50 and x.current.prefix == "ps_output" and at() == (0, 10)
              and "x = 10.50 Mpc" in x.position.text(), f"{at()} {x.position.text()}")
        p50 = x.current
        ok = pick(x.prefix_combo, str(s50 / "output"))
        check("7. another output on a coarser grid (32³ in the same box): its nearest cell, x = 11.00 Mpc",
              ok and x.current.prefix == "output" and at() == (0, 5) and "x = 11.00 Mpc" in x.position.text(),
              f"{at()} {x.position.text()}")
        d50 = x.current
        ok = pick(x.prefix_combo, str(p50.root))
        check("7. ... and back: the place remembered, not re-read off the coarser grid (cell 10 again, not 11)",
              ok and at() == (0, 10), f"{at()}")
        pick(x.prefix_combo, str(d50.root))
        x.index.setValue(7)                               # its centre, 15 Mpc, is a face between two 64³ cells
        shown()
        ok = pick(x.snap_combo, 99)
        check("7. a place on a face of the finer grid's cells takes the upper cell (another redshift too)",
              ok and x.current.snap == 99 and at() == (0, 15), f"{at()}")
        ok = pick(x.sim_combo, sim2)
        check("7. another simulation: the middle of its box (the axis stays, as before)",
              ok and x.current.sim == sim2 and at() == (0, N // 2) and "x = 16.25 Mpc" in x.position.text(),
              f"{at()} {x.position.text()}")
        ok = pick(x.sim_combo, SIM)
        check("7. ... and back to the first simulation: the middle again, not the place left there",
              ok and x.current.sim == SIM and x.current.snap == 99 and at() == (0, N // 2), f"{at()}")

        # ---- the zoom window: kept across outputs and redshifts where it can be shown, said where it cannot
        o99 = x.current
        f99 = fields(3.0)
        x._zoomed(zoom_req(), f99)                        # a zoom of snapshot 099 lands (and is remembered)
        z99 = x.is_zoomed() and x._zoom_fields is f99
        said.clear()
        ok = pick(x.snap_combo, 50)                       # its 'output' (the prefix chosen last): no zoom of it yet
        st = x.zoom_state.text()
        check("7. zoom window kept, not shown where there is no zoom of it (no memory, no server): the slice map at "
              "the kept place, and the zoom and status lines say which window and why",
              z99 and ok and not x.is_zoomed() and x.current.prefix == "output" and at() == (0, N // 4)
              and x._plane is not None and x._plane.shape == (32, 32) and st.startswith("zoom window y 20.00–30.00, "
              "z 40.00–50.00 Mpc kept") and "query server" in st and st in said, f"{z99} {at()} {st!r} {said}")
        ok = pick(x.snap_combo, 99)
        check("7. ... back on the snapshot zoomed: its zoom of the window, from memory",
              ok and x.is_zoomed() and x._zoom_fields is f99 and x._zoom["a0"] == (20.0, 30.0)
              and x._zoom["a1"] == (40.0, 50.0) and x.zoom_time.text() == "from memory", x.zoom_state.text())
        # the query server running on the other output: the window is evaluated there (the request stood in for)
        sent = []
        spy = sent.append
        x._to_zoom.disconnect(x._server.zoom)
        x._to_zoom.connect(spy)
        x._server_running, x._server_output = True, x._by_root[str(d50.root)]
        try:
            ok = pick(x.snap_combo, 50)
            r = sent[0] if sent else {}
            check("7. ... with the query server on the other output: the same window evaluated there, on its kept "
                  "slice (the cell centre x = 33.00 Mpc of the 32³ grid)",
                  ok and len(sent) == 1 and r.get("output") is x.current and r.get("index") == N // 4 == x.index.value()
                  and r.get("a0") == (20.0, 30.0) and r.get("a1") == (40.0, 50.0) and r.get("res") == 8
                  and abs(float(r["coords"][0][0]) - 33.0) < 1e-9 and x._zoom_pending, f"{sent}")
            if r:
                f50 = fields(5.0)
                x._zoomed(r, f50)                         # its answer
                check("7. ... and shown when it arrives", x.is_zoomed() and x._zoom_fields is f50)
        finally:
            x._server_running, x._server_output = False, None
            x._to_zoom.disconnect(spy)
            x._to_zoom.connect(x._server.zoom)
        # 'exact' ticked: a switch never starts an exact run nor asks about one (review 2026-10-14: stepping through
        # redshifts would otherwise ask, or run PS-DTFE over a whole snapshot, at every step); a drag does
        calls, asked = [], []
        x._exact_zoom = lambda *a: calls.append(a)
        x.confirm_exact = lambda text: (asked.append(text), False)[1]
        x.zoom_exact.setChecked(True)
        try:
            ok = pick(x.prefix_combo, str(p50.root))      # ps_output of 050: no zoom of the window in memory
            st = x.zoom_state.text()
            check("7. ... 'exact' ticked: no exact run started or asked about by the switch, the window kept and the "
                  "zoom line says to drag it again",
                  ok and not calls and not asked and not x.is_zoomed() and at() == (0, N // 2)
                  and st.startswith(E.KEPT_ZOOM_TEXT) and "drag the window again" in st, f"{calls} {st!r}")
        finally:
            del x._exact_zoom
            x.confirm_exact = None
            x.zoom_exact.setChecked(False)
        ok = pick(x.prefix_combo, str(d50.root))
        check("7. ... flipping back to an output zoomed a moment ago: its zoom from memory at once",
              ok and x.is_zoomed() and x._zoom_fields is f50 and x.current.prefix == "output")
        pick(x.prefix_combo, str(p50.root))               # not shown here: said in the zoom line
        x._dragged(2, 2, 10, 10)                          # no server, 'exact' off: no zoom made
        check("7. a drag that makes no zoom (no server, 'exact' off) leaves the kept window and its line",
              bool(x._keep.get("zoom")) and x.zoom_state.text().startswith(E.KEPT_ZOOM_TEXT), x.zoom_state.text())
        x.index.setValue(x.index.value() + 1)             # a slice chosen by hand: the window goes with the old one
        shown()
        gone = x._keep.get("zoom") is None and x.zoom_state.text() == ""
        ok = pick(x.prefix_combo, str(d50.root))
        check("7. a slice chosen by hand ends the kept window (and its line): the next output shows its map",
              gone and ok and not x.is_zoomed() and x.zoom_state.text() == "", x.zoom_state.text())
        x._zoomed(zoom_req(), fields(6.0))
        x.zoom_out()
        ok = pick(x.prefix_combo, str(p50.root))
        check("7. Zoom out ends it too: the whole slice on the next output, nothing said",
              ok and not x.is_zoomed() and x._keep.get("zoom") is None and x.zoom_state.text() == "")
        x._zoomed(zoom_req(), fields(7.0))
        zoomed = x.is_zoomed()
        ok = pick(x.sim_combo, sim2)
        check("7. another simulation with a zoom on screen: the whole box at the middle, the window dropped",
              zoomed and ok and not x.is_zoomed() and x._keep.get("zoom") is None and at() == (0, N // 2)
              and x.zoom_state.text() == "", f"{at()} {x.zoom_state.text()!r}")

        # ---- the custom runs (and opened files): kept in a box of the same size, else the middle
        ok = pick(x.sim_combo, E.GROUP_CUSTOM)            # its last run: c3s (32³ in a 32 Mpc box)
        check("7. the custom snapshots (another entry of the first menu): the middle",
              ok and str(x.current.root) == str(c3s) and at() == (0, 16), f"{x.current.root} {at()}")
        ok = pick(x.snap_combo, str(c3a))
        check("7. a custom run in a box of another size (64 against 32 Mpc): the middle", ok and at() == (0, N // 2))
        x.axis.setCurrentIndex(1)
        x.index.setValue(20)
        shown()
        x._zoomed(zoom_req(), fields(8.0))
        zoomed = x.is_zoomed()
        ok = pick(x.snap_combo, str(c2d))
        st = x.zoom_state.text()
        check("7. a 2D run on the way: its one plane (no axis, no slice), the window not shown and why",
              zoomed and ok and x.current.dim == 2 and not x.axis.isEnabled() and at() == (2, 0)
              and not x.is_zoomed() and st.startswith("zoom window x 20.00–30.00, z 40.00–50.00 Mpc kept")
              and "2D output" in st, f"{at()} {st!r}")
        cb = x._by_root[str(c3b)]
        fb = fields(9.0, ax=1)
        memo(cb, 1, 10, fb)                               # (a zoom of c3b made before)
        ok = pick(x.snap_combo, str(c3b))
        check("7. ... the next 3D run in the same box gets the slice back (axis y, the nearest 32³ cell) and the window",
              ok and at() == (1, 10) and x.is_zoomed() and x._zoom_fields is fb, f"{at()} {x.zoom_state.text()!r}")
        ok = pick(x.snap_combo, str(c3s))
        check("7. ... and a run in a box of another size: the middle, the whole box",
              ok and at() == (1, 16) and not x.is_zoomed() and x._keep.get("zoom") is None)
        ok = x.open_file(str(o1) + ".a_den") and shown(o1)
        check("7. an opened file (another entry of the first menu): the middle", ok and at() == (1, N // 2))
        x.index.setValue(40)
        shown()
        ok = x.open_file(str(o2) + ".a_den") and shown(o2)
        check("7. ... another opened file in the same box: the same place (y = 40.50 Mpc: cell 20 of 32)",
              ok and at() == (1, 20), f"{at()}")
        ok = x.open_file(str(o3) + ".a_den") and shown(o3)
        check("7. ... one whose box is unknown (cells, another grid): the middle", ok and at() == (1, 8), f"{at()}")
        texts = said + lines + [E.KEPT_ZOOM_TEXT]
        check("7. the kept slice's texts carry no '--'", not [t for t in texts if "--" in t], f"{texts}")
    finally:
        x.shutdown()


def main():
    try:
        pure()
        gui()
        fold_map()
        keep_slice()
        save_settings()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nRESULT: {len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
