"""How the Explore tab saves its pictures.

  options(d)          the save options, checked and completed: the DEFAULTS draw exactly the reference pictures of
                      Save image… and Save figure… (a test compares the pixels), so nothing changes for whoever
                      never opens the dialog; PRESETS fill several in one go (Publication, Bare image, Slide)
  unit_map            the axes' unit: Mpc as stored (h-free), Mpc/h when the box's h is known, or cells
  position            a position along any axis in that unit (the slice position in a saved title)
  style_axes          frame, ticks, tick labels and axis names of one matplotlib Axes as the options say
  rc / save_kwargs    the matplotlib settings (font size, text kept as text in PDF and SVG) and savefig() arguments
                      (a TIFF compressed losslessly, LZW)
  fit_width           the figure scaled so the saved file has the width asked for
  match_dpi           the dpi that puts one data pixel on one image pixel inside the map's axes (a plane too small
                      for 72 dpi shrinks the figure instead)
  settle_wraps        wrapped titles given their drawn line breaks, so every matplotlib version measures them alike
  savefig             a matplotlib figure saved with the options (pixel guard, settled wraps, save_kwargs)
  writes_pixels       the options that save the data's own pixels: full resolution, or no margins + match the data
  write_pixels        the full-resolution file: one image pixel per data pixel, the screen's colours, overlays optional
  SaveOptionsDialog   the one dialog behind Save image…, Save figure… and Explore > Save Options…

Matplotlib only (no pyplot) and numpy; the dialog is the only Qt part.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# format -> (the file dialog's words, the suffixes it owns); vector formats keep their text as text
FORMATS = {"png": ("PNG image", ("png",)), "pdf": ("PDF, vector", ("pdf",)), "svg": ("SVG, vector", ("svg",)),
           "tiff": ("TIFF image", ("tif", "tiff"))}
VECTOR = ("pdf", "svg")
UNITS = (("mpc", "Mpc"), ("mpc_h", "Mpc/h"), ("cells", "cells"))
MARGINS = (("tight", "Tight crop (as before)"), ("pad", "Tight crop with this padding"),
           ("none", "None: the map fills the file"))
DPI_MIN, DPI_MAX = 72, 1200
FULL_DPI = 100.0                # the full-resolution file's nominal dpi: the outlines' line widths, a vector page size
# the largest side of a figure drawn for a save: matplotlib allocates the whole picture (4 bytes a
# pixel) already while laying it out, so a width meant in pixels but typed in inches (50 in at 1200 dpi: 24 GB) must
# stop before that; 16384 a side is 1 GB for the canvas alone (the image resampled into it takes more: a 1200 dpi map
# of 7 in peaks at about 2 GB)
MAX_SIDE_PX = 16384
# a TIFF's compression: LZW, lossless (the same pixels, a test reads them
# back) and read by every TIFF reader (the 'TIFF with LZW' journals ask for); Pillow's libtiff writes it, for the
# full-resolution file directly and for a matplotlib figure through savefig's pil_kwargs
TIFF_COMPRESSION = "tiff_lzw"

DEFAULTS = {
    "kind": "figure",           # "figure": title, axes and colour bar at a dpi; "full": the map alone at its own pixels
    "format": "png",
    "dpi": 200,
    "match_data": False,        # the dpi that puts one data pixel on one image pixel in the map (not above DPI_MAX)
    "width": 0.0,               # the saved file's width; 0 = the layout's own size (7 in for one map)
    "width_unit": "in",         # "in" or "px"
    "margins": "tight",         # "tight" (matplotlib's 0.1 in), "pad" (the padding below) or "none"
    "pad": 0.1,                 # inches, for "pad"
    "font_scale": 1.0,
    "transparent": False,
    "title": True,
    "panel_titles": True,       # the figure grid: each panel's title (or its inside label when merged)
    "frame": True,
    "ticks": True,
    "tick_labels": True,
    "axis_names": True,
    "units": True,              # the unit in the axis names: 'x [Mpc]' or 'x'
    "unit": "mpc",              # "mpc" (cells for a grid without a box), "mpc_h" or "cells"
    "colorbar": True,
    "colorbar_label": True,
    "arrows": True,             # as on screen: drawn when the map shows them
    "voids": True,              # as on screen: drawn when the map outlines them
}
# a typed title belongs to the map it was typed for ('title_for' = that map's own title): kept for the session,
# never remembered between sessions, and a new map (another field, slice, output) gets its own title again
SESSION_KEYS = ("title_text", "title_for")
_CHOICES = {"kind": ("figure", "full"), "format": tuple(FORMATS), "width_unit": ("in", "px"),
            "margins": tuple(m for m, _ in MARGINS), "unit": tuple(u for u, _ in UNITS)}
_RANGES = {"dpi": (DPI_MIN, DPI_MAX), "width": (0.0, 100000.0), "pad": (0.0, 2.0), "font_scale": (0.5, 3.0)}

_BARE = {"kind": "full", "margins": "none", "title": False, "panel_titles": False, "frame": False, "ticks": False,
         "tick_labels": False, "axis_names": False, "colorbar": False, "colorbar_label": False, "arrows": False,
         "voids": False}
PRESETS = {"standard": {}, "publication": {"format": "pdf", "dpi": 300}, "bare": _BARE,
           "slide": {"font_scale": 1.8}}
PRESET_NAMES = (("standard", "Standard (as before)"), ("publication", "Publication: vector PDF, 300 dpi"),
                ("bare", "Bare image: the map alone, full resolution"), ("slide", "Slide: large fonts"))
PRESET_SHORT = {"standard": "Standard", "publication": "Publication", "bare": "Bare image", "slide": "Slide"}
PRESET_TIPS = {
    "standard": "What Save image and Save figure always wrote: title, axes in Mpc, colour bar, arrows and void "
                "outlines as on screen, PNG at 200 dpi, cropped tight",
    "publication": "Everything on, as a vector PDF (the text stays text, editable); a PNG or TIFF chosen in the file "
                   "dialog gets 300 dpi",
    "bare": "Only the map: no frame, axes, title or colour bar, no arrows or outlines, one data pixel per image pixel "
            "(a 4096 by 4096 point plane gives a 4096 by 4096 image), the colours and range as on screen",
    "slide": "Everything on with fonts 1.8 times larger, for a projector",
    "last": "The options you used last (remembered between sessions)",
    "custom": "Your changes to a preset",
}


def options(d: dict | None = None) -> dict:
    """The save options from a (remembered, possibly older or hand-edited) dict: unknown keys dropped, wrong types
    and values outside their range replaced by the defaults, the session-only title kept as text."""
    out = dict(DEFAULTS)
    for k, v in (d or {}).items():
        if k in SESSION_KEYS:
            out[k] = str(v or "")
            continue
        if k not in DEFAULTS:
            continue
        dv = DEFAULTS[k]
        if isinstance(dv, bool):
            if isinstance(v, bool):
                out[k] = v
        elif isinstance(dv, (int, float)):
            if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                lo, hi = _RANGES.get(k, (-np.inf, np.inf))
                out[k] = type(dv)(min(max(v, lo), hi))
        elif isinstance(v, str) and v in _CHOICES.get(k, (v,)):
            out[k] = v
    return out


def persistent(opts: dict) -> dict:
    """What QSettings keeps: the options without the session-only title."""
    return {k: v for k, v in options(opts).items() if k not in SESSION_KEYS}


def dumps(opts: dict) -> str:
    return json.dumps(persistent(opts), sort_keys=True)


def loads(raw) -> dict:
    """options() of a remembered JSON text; the defaults for nothing or something unreadable."""
    try:
        d = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        d = {}
    return options(d if isinstance(d, dict) else {})


def preset(name: str, keep: dict | None = None) -> dict:
    """A preset's options (the defaults with its changes); the session-only title of 'keep' stays."""
    out = options({**DEFAULTS, **PRESETS[name]})
    for k in SESSION_KEYS:
        if keep and keep.get(k):
            out[k] = keep[k]
    return out


def which_preset(opts: dict) -> str | None:
    """The preset these options are, None for any other choice."""
    mine = persistent(opts)
    return next((n for n in PRESETS if persistent(preset(n)) == mine), None)


def describe(opts: dict) -> str:
    """A few words for a tooltip and the status line: the preset or the format, and the resolution."""
    o = options(opts)
    name = PRESET_SHORT.get(which_preset(o) or "", "your options")
    fmt = o["format"].upper()
    if writes_pixels(o):
        return f"{name}: {fmt} at the data's own resolution"
    res = "dpi matched to the data" if o["match_data"] else f"{o['dpi']} dpi"
    return f"{name}: {fmt}, {res}" if fmt not in ("PDF", "SVG") else f"{name}: {fmt} (vector), {res} for the map"


def title_text(opts: dict, auto: str, shown: str | None = None) -> str:
    """The title drawn: the typed one when it was typed for this map (its own title then was 'auto'), else the map's
    own title: 'shown' when given ('auto' with its positions in the axes' unit, Mpc/h or cells; 'auto'
    stays the key a typed title is tied to, so changing the unit never detaches it), else 'auto'."""
    t = (opts.get("title_text") or "").strip()
    return t if t and opts.get("title_for", auto) == auto else (auto if shown is None else shown)


def writes_pixels(opts: dict) -> bool:
    """The save writes the data's own pixels (write_pixels: one image pixel per data pixel, the screen's colours, no
    resampling): full resolution, or no margins with 'match the data' and no width asked (that borderless figure is
    the same file as full resolution: matplotlib's picture of the map at the matched dpi, 72 at least, would resample
    a plane under ~390 points across)."""
    o = options(opts)
    return o["kind"] == "full" or (o["margins"] == "none" and o["match_data"] and not o["width"] > 0)


# ---------------------------------------------------------------------- formats and file names
def format_of(path, fallback: str = "png") -> str:
    """The format a file name asks for by its suffix ('.tif' and '.tiff' are TIFF); 'fallback' without one."""
    suf = Path(path).suffix.lower().lstrip(".")
    return next((f for f, (_t, sfx) in FORMATS.items() if suf in sfx), fallback)


def with_format(path, fmt: str) -> Path:
    """The path with the format's suffix when it has none of its own (a name typed without one)."""
    p = Path(path)
    suf = p.suffix.lower().lstrip(".")
    if any(suf in sfx for _t, sfx in FORMATS.values()) or suf in ("npy", "csv"):
        return p
    return p.with_name(p.name + "." + FORMATS[fmt][1][0])


def file_filters() -> list[tuple[str, str]]:
    """(format, the file dialog's filter text) of every image format, in menu order."""
    return [(f, f"{t} (" + " ".join(f"*.{s}" for s in sfx) + ")") for f, (t, sfx) in FORMATS.items()]


# ---------------------------------------------------------------------- units
class UnitMap:
    """The axes' unit as one affine map per axis, x' = (x - off) k, of the positions as Explore keeps them (h-free
    Mpc of the box, or cells when the output has no box). The identity leaves the numbers untouched (bit for bit:
    the default figure keeps its exact numbers)."""

    def __init__(self, name: str = "Mpc", k=(1.0, 1.0), off=(0.0, 0.0)):
        self.name = name
        self.k = (float(k[0]), float(k[1]))
        self.off = (float(off[0]), float(off[1]))
        self.identity = self.k == (1.0, 1.0) and self.off == (0.0, 0.0)

    def x(self, v):
        return v if self.identity else (np.asarray(v, dtype=np.float64) - self.off[0]) * self.k[0]

    def y(self, v):
        return v if self.identity else (np.asarray(v, dtype=np.float64) - self.off[1]) * self.k[1]

    def ext(self, e) -> list:
        """[x0, x1, y0, y1] of an imshow extent."""
        return list(e) if self.identity else [float(self.x(e[0])), float(self.x(e[1])), float(self.y(e[2])),
                                               float(self.y(e[3]))]


def unit_map(o, rest, unit: str, hubble: float | None = None) -> tuple[UnitMap, str]:
    """(UnitMap, a note when the unit asked for cannot be used) for the in-plane axes 'rest' of output 'o'. 'mpc' is
    the stored unit (cells for an output without a box, as the map says); 'mpc_h' multiplies by the box's h
    ('hubble', grids.hubble: None when unknown, then Mpc and a note); 'cells' counts the grid's cells from the box's
    low corner."""
    if o.box is None:
        return UnitMap("cells"), ("" if unit == "cells" or unit == "mpc" else "no box known: the axes are in cells")
    if unit == "cells":
        k = [float(o.n) / float(o.length[a]) if o.length[a] > 0 else 1.0 for a in rest]
        return UnitMap("cells", k, [float(o.lo[a]) for a in rest]), ""
    if unit == "mpc_h":
        if hubble:
            return UnitMap("Mpc/h", (hubble, hubble)), ""
        return UnitMap("Mpc"), "the box's h is not known: the axes are in Mpc"
    return UnitMap("Mpc"), ""


def position(o, axis: int, v: float, unit: str, hubble: float | None = None) -> tuple[float, str]:
    """(value, unit name) of position 'v' along axis 'axis' of output 'o', as Explore keeps it (h-free Mpc, cells for
    an output without a box), in the unit the axes get from unit_map (Mpc/h without a known h: Mpc). A saved title's
    slice position and zoom window."""
    um, _note = unit_map(o, [axis, axis], unit, hubble)
    return float(um.x(v)), um.name


# ---------------------------------------------------------------------- axes, fonts, saving
def axis_label(name: str, unit: str, opts: dict) -> str:
    if not opts["axis_names"]:
        return ""
    return f"{name} [{unit}]" if opts["units"] else name


def style_axes(ax, opts: dict, names, unit: str, fontsize=None, ticksize=None) -> None:
    """The axis names, frame, ticks and tick labels of one Axes as the options say. With every option on it only
    sets the names, leaving matplotlib's own frame and ticks."""
    kw = {} if fontsize is None else {"fontsize": fontsize}
    ax.set_xlabel(axis_label(names[0], unit, opts), **kw)
    ax.set_ylabel(axis_label(names[1], unit, opts), **kw)
    if not opts["frame"]:
        for sp in ax.spines.values():
            sp.set_visible(False)
    if not opts["ticks"]:
        ax.tick_params(length=0)
    if not opts["tick_labels"]:
        ax.tick_params(labelbottom=False, labelleft=False)
    if ticksize is not None:
        ax.tick_params(labelsize=ticksize)


def rc(opts: dict, fmt: str = "png") -> dict:
    """matplotlib rcParams for the figure: the font size scaled, and in PDF and SVG the text kept as text (TrueType in
    PDF, real text in SVG: editable in Illustrator or Inkscape); empty for the defaults and a PNG."""
    import matplotlib
    d = {}
    s = float(opts.get("font_scale", 1.0))
    if s != 1.0:
        d["font.size"] = float(matplotlib.rcParams["font.size"]) * s
    if fmt == "pdf":
        d["pdf.fonttype"] = 42
    if fmt == "svg":
        d["svg.fonttype"] = "none"
    return d


def save_kwargs(opts: dict, dpi: float, fmt: str = "png") -> dict:
    """savefig()'s arguments: the dpi, the margins (the default: bbox_inches='tight'), transparency; a
    TIFF compressed losslessly (TIFF_COMPRESSION; matplotlib hands a TIFF to Pillow with pil_kwargs)."""
    kw = {"dpi": dpi}
    if opts["margins"] == "tight":
        kw["bbox_inches"] = "tight"
    elif opts["margins"] == "pad":
        kw.update(bbox_inches="tight", pad_inches=float(opts["pad"]))
    if opts["transparent"]:
        kw["transparent"] = True
    if fmt == "tiff":
        kw["pil_kwargs"] = {"compression": TIFF_COMPRESSION}
    return kw


def settle_wraps(fig, dpi: float) -> None:
    """Give every wrapped text of 'fig' the line breaks it gets at the figure's current size and 'dpi', as explicit
    newlines with wrapping off; the original text is kept on the artist, so a later call re-wraps it after a resize.
    matplotlib 3.6 (Ubuntu 24.04, the CI) wraps text only while drawing: its extents, get_tightbbox and savefig's
    tight crop measure a wrapped title unwrapped, so without this fit_width does not converge (a map asked 900 px wide
    comes out 907) and the crop takes the unwrapped title's width. matplotlib 3.10 lays the wrapped lines out
    everywhere, and the same breaks give the same file there."""
    from matplotlib.text import Text
    own = fig.dpi
    fig.set_dpi(dpi)
    try:
        renderer = fig.canvas.get_renderer()
        for t in fig.findobj(Text):
            src = getattr(t, "_figsave_wrap_src", None)
            if src is None:
                if not (t.get_wrap() and t.get_text()):
                    continue
                src = t._figsave_wrap_src = t.get_text()
            t.set_text(src)
            t.set_wrap(True)
            t.get_window_extent(renderer)          # the renderer the line-width measure uses (matplotlib 3.6)
            lines = t._get_wrapped_text()
            t.set_wrap(False)
            t.set_text(lines)
    finally:
        fig.set_dpi(own)


def savefig(fig, path, opts: dict, dpi: float, fmt: str = "png") -> None:
    """Save a matplotlib figure with the options: the pixel guard before Agg allocates it, the wrapped texts' line
    breaks for this size (settle_wraps), then savefig with save_kwargs."""
    check_pixels(*fig.get_size_inches(), dpi)
    settle_wraps(fig, dpi)
    fig.savefig(path, **save_kwargs(opts, dpi, fmt))


class TooLarge(ValueError):
    """A width or dpi that would make a picture of more than MAX_SIDE_PX pixels a side."""


def check_pixels(w_in: float, h_in: float, dpi: float) -> None:
    """Raise TooLarge, before matplotlib allocates it, for a figure of w_in by h_in inches at 'dpi' that is more than
    MAX_SIDE_PX pixels a side."""
    w, h = float(w_in) * float(dpi), float(h_in) * float(dpi)
    if max(w, h) > MAX_SIDE_PX:
        raise TooLarge(f"the figure would be {w:.0f} × {h:.0f} pixels ({4e-9 * w * h:.1f} GB in memory), more than "
                       f"{MAX_SIDE_PX} a side: choose a smaller width or dpi")


def target_width_in(opts: dict, dpi: float) -> float | None:
    """The width asked for in inches (None: the layout's own)."""
    w = float(opts.get("width") or 0.0)
    if w <= 0:
        return None
    return w if opts["width_unit"] == "in" else w / float(dpi)


def fit_width(fig, opts: dict, dpi: float, relayout=None) -> None:
    """Scale the figure so the SAVED file is as wide as asked: the whole figure when nothing is cropped, else the
    tight crop plus its padding (a few rounds: the text does not scale with the figure). 'relayout' re-runs a
    tight_layout after each resize."""
    target = target_width_in(opts, dpi)
    if target is None:
        return
    if fig._suptitle is not None:
        fig._suptitle.set_wrap(True)                # a long title wraps rather than widening the figure
    W, H = fig.get_size_inches()
    check_pixels(target, H * target / W, dpi)       # a width far too large stops here, before any drawing
    if opts["margins"] == "none":
        eps = 1e-3 / float(dpi)                     # int(w dpi) must not round a whole pixel down
        fig.set_size_inches(target + eps, H * target / W + eps)
        return
    pad = 0.1 if opts["margins"] == "tight" else float(opts["pad"])
    own = fig.dpi
    fig.set_dpi(dpi)                                # the text's extent at the dpi it is saved at (hinting differs)
    try:
        for _ in range(6):
            check_pixels(*fig.get_size_inches(), dpi)   # every renderer below allocates the whole figure
            if relayout is not None:
                relayout()
            settle_wraps(fig, dpi)                  # measure the title as it will be drawn at this size
            bb = fig.get_tightbbox(fig.canvas.get_renderer())
            have = bb.width + 2 * pad
            if abs(have - target) * dpi < 0.5:
                break
            W, H = fig.get_size_inches()
            f = max(0.1, (W + target - have) / W)
            fig.set_size_inches(W * f, H * f)
        check_pixels(*fig.get_size_inches(), dpi)
        if relayout is not None:
            relayout()
    finally:
        fig.set_dpi(own)


def match_dpi(fig, axes_px, relayout=None) -> float:
    """The dpi at which the widest data plane fills its axes pixel for pixel: max over (axes, data pixels across) of
    pixels / the axes' drawn width in inches, at most DPI_MAX. A plane too small for DPI_MIN (fewer than 72 points per
    inch of its axes: every plane under ~390 points across in the 7 in map, a 256³ grid's slice too) SHRINKS THE
    FIGURE instead: at DPI_MIN its axes then hold the plane pixel for pixel, so the saved map has the data's pixel
    count (stopping the dpi at 72 would resample a 64-cell map to 389 pixels), while the text keeps its size in
    points, readable at 72 dpi (a dpi below it draws 8 pt text one to three pixels high).
    The shrink is uniform with the layout's fractions kept, which is exact in one step (the aspect of every axes box
    is kept, so its width scales with the figure); 'relayout' (the separate panels' tight_layout) is re-run between
    such steps while it still fits the text, and a last step makes the size exact whatever it did. A tiny plane
    (64 cells) leaves little room for the text: legible, tightly packed. The caller's check_pixels guard stays (a
    shrunk figure only gets smaller)."""
    import warnings

    def need():                                     # (dpi wanted, data pixels of that axes)
        W = fig.get_size_inches()[0]
        best = (0.0, 0.0)
        for ax, px in axes_px:
            ax.apply_aspect()
            w_in = ax.get_position().width * W
            if w_in > 0 and float(px) / w_in > best[0]:
                best = (float(px) / w_in, float(px))
        return best

    def exact(b, p):                                # the widest-needing axes within 0.02 pixel of its plane
        return b > 0 and abs(p * (1.0 - DPI_MIN / b)) < 0.02
    best, px = need()
    if best >= DPI_MIN or best <= 0:
        return float(min(max(best or DPI_MIN, DPI_MIN), DPI_MAX))
    own = fig.dpi
    fig.set_dpi(DPI_MIN)                            # the text's extent at the dpi it is saved at (as fit_width)
    try:
        for _ in range(8):
            W, H = fig.get_size_inches()
            f = best / DPI_MIN
            fig.set_size_inches(W * f, H * f)       # the fractions kept: exact
            best, px = need()
            if relayout is None:
                break
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")     # 'tight layout not applied' (too small for the text): no output
                relayout()
            best, px = need()
            if caught or exact(best, px):
                break
        if not exact(best, px):                     # the last tight_layout moved the axes: one more uniform step
            W, H = fig.get_size_inches()
            f = best / DPI_MIN
            fig.set_size_inches(W * f, H * f)
        W, H = fig.get_size_inches()
        eps = 1e-3 / DPI_MIN                        # int(w dpi) must not round a whole pixel down (a borderless map)
        fig.set_size_inches(W + eps, H + eps)
    finally:
        fig.set_dpi(own)
    return float(DPI_MIN)


def write_pixels(path, rgb: np.ndarray, fmt: str, overlay=None, transparent: bool = False) -> None:
    """The full-resolution file: 'rgb' (rows from the top, uint8) one image pixel per element. 'overlay(ax)' draws
    on an invisible Axes spanning the image exactly (it sets its own limits: arrows, outlines); None writes the array
    as it is (PNG and TIFF straight through Pillow; a TIFF LZW-compressed, TIFF_COMPRESSION). A PDF or SVG embeds the
    pixels unresampled on a page of width / 100 inches."""
    rgb = np.ascontiguousarray(rgb)
    if overlay is None and fmt in ("png", "tiff"):
        from PIL import Image                       # matplotlib's own dependency: the RGB as it is
        extra = {"compression": TIFF_COMPRESSION} if fmt == "tiff" else {}
        Image.fromarray(rgb).save(str(path), format=fmt.upper(), dpi=(FULL_DPI, FULL_DPI), **extra)
        return
    if overlay is None:
        import matplotlib.image as mimage
        mimage.imsave(str(path), rgb, format=fmt, dpi=FULL_DPI)
        return
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    H, W = rgb.shape[:2]
    eps = 1e-3                                       # int(w dpi) must not round a whole pixel down
    fig = Figure(figsize=((W + eps) / FULL_DPI, (H + eps) / FULL_DPI), dpi=FULL_DPI, frameon=False)
    FigureCanvasAgg(fig)
    fig.figimage(rgb, origin="upper")               # unresampled: pixel (r, c) of the array is pixel (r, c) of the file
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_zorder(1)                                # over the figure image
    overlay(ax)
    extra = {"pil_kwargs": {"compression": TIFF_COMPRESSION}} if fmt == "tiff" else {}
    fig.savefig(str(path), dpi=FULL_DPI, format=fmt, transparent=transparent, **extra)


def draw_pixel_circles(ax, shape, circles, dashed=(), clips=()) -> int:
    """The figure grid's outlines over its full-resolution picture (grid_tile_circles: image pixels, y down), each
    clipped to its tile as on screen; white over a dark edge like grids.draw_void_circles_mpl. The patches drawn."""
    from matplotlib.patches import Ellipse, Rectangle
    H, W = shape[:2]
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    n = 0
    for k, ((x, y, rx, ry, _row), style) in enumerate([(c, "-") for c in circles] + [(c, (0, (4, 3))) for c in dashed]):
        clip = clips[k] if k < len(clips) else None
        for lw, col in ((1.6, (0, 0, 0, 0.65)), (0.7, "white")):
            e = Ellipse((x, y), 2 * rx, 2 * ry, fill=False, lw=lw, ec=col, ls=style)
            ax.add_patch(e)
            if clip is not None:
                e.set_clip_path(Rectangle((clip[0], clip[1]), clip[2], clip[3], transform=ax.transData))
        n += 1
    return n


# ---------------------------------------------------------------------- the dialog
def _qt():
    from PySide6.QtCore import QLocale, Qt
    from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
                                   QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton, QSpinBox,
                                   QVBoxLayout, QWidget)
    return locals()


def make_dialog(parent, opts: dict, context: dict | None = None, can_save: bool = True):
    """The Save options dialog (a SaveOptionsDialog; built on demand so this module imports without Qt)."""
    return _dialog_class()(parent, opts, context or {}, can_save)


_DIALOG = None


def _dialog_class():
    global _DIALOG
    if _DIALOG is not None:
        return _DIALOG
    q = _qt()
    QDialog, Qt = q["QDialog"], q["Qt"]

    class SaveOptionsDialog(QDialog):
        """One dialog for Save image…, Save figure… and the Explore menu's Save Options…: a preset, then every option.
        'context' says what is on screen: grid (bool), arrows / voids (shown now), hubble (h or None), box (bool),
        auto_title (the title the figure would get), pixels ((width, height) of the full-resolution image)."""

        def __init__(self, parent, opts: dict, context: dict, can_save: bool):
            super().__init__(parent)
            self.setLocale(q["QLocale"].c())             # a decimal point in the number fields, whatever the Mac's
            self.setWindowTitle("Save options")
            self.ctx = dict(context)
            self.opened = options(opts)                  # 'Your last choice'
            self.save_now = False
            self._filling = False
            v = q["QVBoxLayout"](self)
            top = q["QHBoxLayout"]()
            top.addWidget(q["QLabel"]("Preset:"))
            self.preset = q["QComboBox"]()
            for key, text in PRESET_NAMES + (("last", "Your last choice"), ("custom", "Custom (your changes)")):
                self.preset.addItem(text, key)
                self.preset.setItemData(self.preset.count() - 1, PRESET_TIPS[key], Qt.ToolTipRole)
            self.preset.setToolTip("Fill in every option at once; your choices are remembered between sessions")
            top.addWidget(self.preset, 1)
            v.addLayout(top)
            cols = q["QHBoxLayout"]()
            v.addLayout(cols)
            left, right = q["QVBoxLayout"](), q["QVBoxLayout"]()
            cols.addLayout(left, 1)
            cols.addLayout(right, 1)
            # ---- the picture
            pic = q["QGroupBox"]("Picture")
            pv = q["QVBoxLayout"](pic)
            self.kind_figure = q["QRadioButton"]("Figure: with title, axes and colour bar")
            px = self.ctx.get("pixels")
            size = f" ({px[0]} × {px[1]} px)" if px else ""
            what = "the panels" if self.ctx.get("grid") else "the map"
            self.kind_full = q["QRadioButton"](f"Full resolution: {what} alone" + size)
            self.kind_full.setToolTip("One data pixel is one image pixel: no interpolation, the colours and range as "
                                      "on screen, the arrows and void outlines when ticked below. A hi-res slice "
                                      "larger than 4096 points a side is written as shown, block-averaged to 4096.")
            self.kind_figure.setToolTip("The figure as matplotlib draws it, at the resolution below")
            grp = q["QButtonGroup"](self)
            grp.addButton(self.kind_figure)
            grp.addButton(self.kind_full)
            pv.addWidget(self.kind_figure)
            pv.addWidget(self.kind_full)
            form = q["QFormLayout"]()
            pv.addLayout(form)
            self.format = q["QComboBox"]()
            for f, (t, _s) in FORMATS.items():
                self.format.addItem(t, f)
            self.format.setToolTip("The file dialog offers this format first; the name you type there decides "
                                   "(.png, .pdf, .svg, .tif). PDF and SVG keep the text as text.")
            form.addRow("Format", self.format)
            drow = q["QHBoxLayout"]()
            self.dpi = q["QSpinBox"]()
            self.dpi.setRange(DPI_MIN, DPI_MAX)
            self.dpi.setSuffix(" dpi")
            self.dpi.setToolTip(f"Dots per inch of a PNG or TIFF, and of the map inside a PDF or SVG "
                                f"({DPI_MIN} to {DPI_MAX})")
            drow.addWidget(self.dpi)
            self.match = q["QCheckBox"]("match the data")
            self.match.setToolTip("The dpi at which the map's axes hold one image pixel per data pixel (at most "
                                  f"{DPI_MAX} dpi; a map too small for {DPI_MIN} dpi gets a smaller figure at "
                                  f"{DPI_MIN} dpi instead): the full resolution with the axes and colour bar around "
                                  "it. With no margins: the map's own pixels, unresampled")
            drow.addWidget(self.match)
            drow.addStretch(1)
            form.addRow("Resolution", drow)
            wrow = q["QHBoxLayout"]()
            self.width = q["QDoubleSpinBox"]()
            self.width.setRange(0.0, 100000.0)
            self.width.setDecimals(2)
            self.width.setSpecialValueText("as laid out")
            self.width.setToolTip("The saved file's width (the tight crop included); 'as laid out' keeps the layout's "
                                  "own size (7 inches for one map)")
            wrow.addWidget(self.width, 1)
            self.width_unit = q["QComboBox"]()
            self.width_unit.addItem("inches", "in")
            self.width_unit.addItem("pixels", "px")
            wrow.addWidget(self.width_unit)
            form.addRow("Width", wrow)
            mrow = q["QHBoxLayout"]()
            self.margins = q["QComboBox"]()
            for m, t in MARGINS:
                self.margins.addItem(t, m)
            self.margins.setToolTip("Tight crop: the white space around the figure cut away (as before); None: the "
                                    "map fills the whole file, without title, axes or colour bar")
            mrow.addWidget(self.margins, 1)
            self.pad = q["QDoubleSpinBox"]()
            self.pad.setRange(0.0, 2.0)
            self.pad.setSingleStep(0.05)
            self.pad.setSuffix(" in")
            self.pad.setToolTip("The white space kept around the crop, in inches")
            mrow.addWidget(self.pad)
            form.addRow("Margins", mrow)
            self.font_scale = q["QSpinBox"]()
            self.font_scale.setRange(50, 300)
            self.font_scale.setSingleStep(10)
            self.font_scale.setSuffix(" %")
            self.font_scale.setToolTip("Every text of the figure scaled by this")
            form.addRow("Font size", self.font_scale)
            self.transparent = q["QCheckBox"]("Transparent background")
            self.transparent.setToolTip("No white behind the axes and text (PNG, TIFF, PDF, SVG)")
            pv.addWidget(self.transparent)
            left.addWidget(pic)
            # ---- overlays
            ov = q["QGroupBox"]("Colour bar and overlays")
            ol = q["QVBoxLayout"](ov)
            self.colorbar = q["QCheckBox"]("Colour bar")
            self.colorbar_label = q["QCheckBox"]("its label")
            crow = q["QHBoxLayout"]()
            crow.addWidget(self.colorbar)
            crow.addWidget(self.colorbar_label)
            crow.addStretch(1)
            ol.addLayout(crow)
            self.arrows = q["QCheckBox"]("Arrows (when the map shows them)")
            self.voids = q["QCheckBox"]("Void outlines (when the map shows them)")
            self.voids.setToolTip("With the outlines drawn, a CSV of those voids is written beside a saved map")
            ol.addWidget(self.arrows)
            ol.addWidget(self.voids)
            left.addStretch(1)
            # ---- title
            tg = q["QGroupBox"]("Title")
            tl = q["QVBoxLayout"](tg)
            self.title = q["QCheckBox"]("Title")
            tl.addWidget(self.title)
            self.title_edit = q["QLineEdit"]()
            auto = self.ctx.get("auto_title", "")
            self.title_edit.setPlaceholderText(auto or "the map's own title")
            self.title_edit.setToolTip("Empty: the map's own title (shown greyed). A title typed here is used for this "
                                       "map only; a new field, slice or output gets its own title again.")
            if self.opened.get("title_text") and self.opened.get("title_for", auto) == auto:
                self.title_edit.setText(self.opened["title_text"])
            tl.addWidget(self.title_edit)
            self.panel_titles = q["QCheckBox"]("Panel titles (the figure grid)")
            self.panel_titles.setToolTip("Each panel's title, or its small label inside when the axes are merged")
            tl.addWidget(self.panel_titles)
            right.addWidget(tg)
            # ---- axes
            ag = q["QGroupBox"]("Axes")
            al = q["QVBoxLayout"](ag)
            arow = q["QHBoxLayout"]()
            self.frame = q["QCheckBox"]("Frame")
            self.frame.setToolTip("The border around the map (the axes' spines)")
            self.ticks = q["QCheckBox"]("Ticks")
            self.tick_labels = q["QCheckBox"]("Tick labels")
            for wdg in (self.frame, self.ticks, self.tick_labels):
                arow.addWidget(wdg)
            arow.addStretch(1)
            al.addLayout(arow)
            nrow = q["QHBoxLayout"]()
            self.axis_names = q["QCheckBox"]("Axis names")
            self.units = q["QCheckBox"]("with the unit")
            self.units.setToolTip("'x [Mpc]' rather than 'x'")
            nrow.addWidget(self.axis_names)
            nrow.addWidget(self.units)
            nrow.addStretch(1)
            al.addLayout(nrow)
            urow = q["QHBoxLayout"]()
            urow.addWidget(q["QLabel"]("Unit"))
            self.unit = q["QComboBox"]()
            h = self.ctx.get("hubble")
            for u, t in UNITS:
                self.unit.addItem(t, u)
            tips = {"mpc": "Comoving Mpc, h-free, as the code stores them" + ("" if self.ctx.get("box", True)
                                                                               else " (no box known: cells)"),
                    "mpc_h": (f"Comoving Mpc/h, h = {h:.4g} from the snapshot's header" if h else
                              "Not known for this output: its snapshot header gives no h for an h-free box"),
                    "cells": "The grid's cells, counted from the box's low corner"}
            for i in range(self.unit.count()):
                self.unit.setItemData(i, tips[self.unit.itemData(i)], Qt.ToolTipRole)
            if not h:
                self.unit.model().item(self.unit.findData("mpc_h")).setEnabled(False)
            self.unit.setToolTip("The unit of the tick values and of the axis names")
            urow.addWidget(self.unit, 1)
            al.addLayout(urow)
            right.addWidget(ag)
            right.addWidget(ov)
            right.addStretch(1)
            # ---- buttons
            self.note = q["QLabel"]("")
            self.note.setWordWrap(True)
            self.note.setStyleSheet("color: gray")
            v.addWidget(self.note)
            brow = q["QHBoxLayout"]()
            self.defaults_btn = q["QPushButton"]("Reset to Standard")
            self.defaults_btn.setToolTip("Back to what was always saved (the Standard preset)")
            self.defaults_btn.clicked.connect(lambda: self._choose("standard"))
            brow.addWidget(self.defaults_btn)
            brow.addStretch(1)
            self.cancel_btn = q["QPushButton"]("Cancel")
            self.cancel_btn.clicked.connect(self.reject)
            brow.addWidget(self.cancel_btn)
            self.ok_btn = q["QPushButton"]("OK")
            self.ok_btn.setToolTip("Use these options for every save from now on")
            self.ok_btn.clicked.connect(self.accept)
            brow.addWidget(self.ok_btn)
            self.save_btn = q["QPushButton"]("Save…")
            self.save_btn.setToolTip("Use these options and save the map (or the figure grid) now")
            self.save_btn.setDefault(True)
            self.save_btn.setEnabled(bool(can_save))
            self.save_btn.clicked.connect(self._save)
            brow.addWidget(self.save_btn)
            v.addLayout(brow)
            self._fill(self.opened)
            self._select_preset()
            self.preset.currentIndexChanged.connect(self._preset_picked)
            for wdg in (self.kind_figure, self.kind_full, self.match, self.transparent, self.colorbar,
                        self.colorbar_label, self.arrows, self.voids, self.title, self.panel_titles, self.frame,
                        self.ticks, self.tick_labels, self.axis_names, self.units):
                wdg.toggled.connect(self._edited)
            for wdg in (self.format, self.width_unit, self.margins, self.unit):
                wdg.currentIndexChanged.connect(self._edited)
            for wdg in (self.dpi, self.width, self.pad, self.font_scale):
                wdg.valueChanged.connect(self._edited)
            self._sync()

        # -------------------------------------------------------- widgets <-> options
        def _fill(self, o: dict):
            self._filling = True
            try:
                (self.kind_full if o["kind"] == "full" else self.kind_figure).setChecked(True)
                self.format.setCurrentIndex(max(0, self.format.findData(o["format"])))
                self.dpi.setValue(int(o["dpi"]))
                self.match.setChecked(o["match_data"])
                self.width.setValue(float(o["width"]))
                self.width_unit.setCurrentIndex(max(0, self.width_unit.findData(o["width_unit"])))
                self.margins.setCurrentIndex(max(0, self.margins.findData(o["margins"])))
                self.pad.setValue(float(o["pad"]))
                self.font_scale.setValue(int(round(100 * o["font_scale"])))
                self.transparent.setChecked(o["transparent"])
                for k in ("title", "panel_titles", "frame", "ticks", "tick_labels", "axis_names", "units", "colorbar",
                          "colorbar_label", "arrows", "voids"):
                    getattr(self, k).setChecked(bool(o[k]))
                unit = o["unit"] if (o["unit"] != "mpc_h" or self.ctx.get("hubble")) else "mpc"
                self.unit.setCurrentIndex(max(0, self.unit.findData(unit)))
            finally:
                self._filling = False

        def options(self) -> dict:
            """The options the widgets show (with the typed title, tied to this map's own title)."""
            o = options({
                "kind": "full" if self.kind_full.isChecked() else "figure",
                "format": self.format.currentData(), "dpi": self.dpi.value(), "match_data": self.match.isChecked(),
                "width": self.width.value(), "width_unit": self.width_unit.currentData(),
                "margins": self.margins.currentData(), "pad": self.pad.value(),
                "font_scale": self.font_scale.value() / 100.0, "transparent": self.transparent.isChecked(),
                **{k: getattr(self, k).isChecked() for k in ("title", "panel_titles", "frame", "ticks", "tick_labels",
                                                              "axis_names", "units", "colorbar", "colorbar_label",
                                                              "arrows", "voids")},
                "unit": self.unit.currentData()})
            if self.opened["unit"] == "mpc_h" and not self.ctx.get("hubble") and o["unit"] == "mpc":
                o["unit"] = "mpc_h"                     # not offered for this output: the remembered choice stays
            t = self.title_edit.text().strip()
            if t:
                o["title_text"], o["title_for"] = t, self.ctx.get("auto_title", "")
            return o

        # -------------------------------------------------------- presets
        def _select_preset(self):
            """The combo on the preset the widgets show, 'Your last choice' or 'Custom'."""
            cur = self.options()
            name = which_preset(cur)
            if name is None:
                name = "last" if persistent(cur) == persistent(self.opened) else "custom"
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(self.preset.findData(name))
            self.preset.blockSignals(False)

        def _choose(self, name: str):
            self.preset.setCurrentIndex(self.preset.findData(name))

        def _preset_picked(self, *_):
            name = self.preset.currentData()
            if name == "custom":
                return
            keep = {"title_text": self.title_edit.text()}
            self._fill(self.opened if name == "last" else preset(name, keep))
            self._sync()

        def _edited(self, *_):
            if self._filling:
                return
            self._sync()
            self._select_preset()

        # -------------------------------------------------------- what applies
        def _sync(self):
            full = self.kind_full.isChecked()
            borderless = self.margins.currentData() == "none"
            deco = not full and not borderless             # title, axes and colour bar can be drawn
            grid = bool(self.ctx.get("grid"))
            for wdg in (self.dpi, self.match, self.width, self.width_unit, self.margins, self.font_scale,
                        self.transparent):
                wdg.setEnabled(not full)
            self.dpi.setEnabled(not full and not self.match.isChecked())
            self.pad.setEnabled(not full and self.margins.currentData() == "pad")
            for wdg in (self.title, self.frame, self.ticks, self.tick_labels, self.axis_names, self.unit,
                        self.colorbar):
                wdg.setEnabled(deco)
            self.title_edit.setEnabled(deco and self.title.isChecked())
            self.units.setEnabled(deco and self.axis_names.isChecked())
            self.colorbar_label.setEnabled(deco and self.colorbar.isChecked())
            self.panel_titles.setEnabled(grid and not full)
            self.arrows.setEnabled(not grid and bool(self.ctx.get("arrows")))
            self.voids.setEnabled(bool(self.ctx.get("voids")))
            notes = []
            if full:
                px = self.ctx.get("pixels")
                notes.append("Full resolution: " + ("the panels" if grid else "the map") + " alone"
                             + (f", {px[0]} × {px[1]} pixels" if px else "") + "; the title, axes and colour bar are "
                             "not drawn.")
            elif borderless and self.match.isChecked() and not self.width.value() > 0:
                notes.append("No margins with 'match the data': the file is the map's own pixels, unresampled "
                             "(as Full resolution).")
            elif borderless:
                notes.append("No margins: the map fills the file, without title, axes or colour bar. Tick 'match the "
                             "data' (no width) for the map's own pixels, unresampled.")
            # a remembered Mpc/h where this output's h is not known: the axes fall back, the choice stays
            if deco and self.unit.currentData() != "cells" and self.opened["unit"] == "mpc_h" \
                    and not self.ctx.get("hubble"):
                notes.append(("No box is known for this output: its axes are in cells." if not self.ctx.get("box", True)
                              else "This output's h is not known: its axes are in Mpc.")
                             + " Mpc/h stays your choice for the outputs that know their h.")
            if grid:
                notes.append("The figure grid draws no arrows.")
            elif not self.ctx.get("arrows"):
                notes.append("The map shows no arrows now.")
            if not self.ctx.get("voids"):
                notes.append("No void outlines on screen now.")
            self.note.setText(" ".join(notes))

        def _save(self):
            self.save_now = True
            self.accept()

    _DIALOG = SaveOptionsDialog
    return _DIALOG
