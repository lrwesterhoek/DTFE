"""The Explore tab's figure grid (Compare): several maps composed as one figure -- a field at several redshifts,
several fields of one snapshot, or the outputs of one snapshot side by side. The pure parts (grid_items, grid_tiles,
grid_figure, ...: numpy and matplotlib on OutputSets and a request dict, run on the slice worker's thread for the
screen and on the GUI thread for a save; explore.py re-exports them) and GridMixin, ExploreControls' composer: the
table of panels, the Fill buttons, merged axes and shared colour ranges, the tiles on screen and Save figure…"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox

import explore as _ex
import figsave as FS
import grids as G


# ==================================================================== the figure grid (pure parts)
# grid_items / grid_tiles run in the slice worker's thread for the screen and on the GUI thread for a save:
# numpy on OutputSets and the request dict only, no widgets.

def grid_items(req: dict) -> list:
    """Per panel of the request: its plane on the slice and the colour range it gets (None for an empty
    panel). The range is colorize()'s (log10 where the panel is log, centred on 0 for a diverging field);
    with 'shared' or 'merged', panels of one (field, component) take the union of their ranges -- and of one family
    of velocity derivatives: a phase-space panel's stored stream average never shares a range, a colour
    bar or a name with a mean-field panel (grids.OutputSet.gradient_family); with a phase-space panel in the grid every
    velocity-derivative panel says its family ('qualify', grids.field_label)."""
    axis, rel, clip, smooth = req["axis"], req["rel"], req["clip"], req["smooth"]
    grad = req.get("ps_gradient", "stored")
    items = []
    for p in req["panels"]:
        if p is None:
            items.append(None)
            continue
        o, name, comp = p["output"], p["field"], p["component"]
        idx = 0 if o.dim == 2 else int(round(rel * (o.n - 1)))
        plane = o.slice(name, axis, idx, comp, grad)
        if smooth > 0:
            try:
                from scipy.ndimage import gaussian_filter
                plane = gaussian_filter(plane, smooth, mode="wrap")
            except ImportError:
                pass
        log, cmap = p["log"], p["cmap"]
        symmetric = name in G.DIVERGING and not log
        img = np.ascontiguousarray(plane.T[::-1])          # the map's orientation: horizontal = first rest axis
        lo, hi = G.value_range(img, log, (clip, 100 - clip), symmetric)
        finite = bool(np.isfinite(plane).any() and (not log or bool((plane > 0).any())))
        rest = [a for a in range(3) if a != axis] if o.dim == 3 else [0, 1]
        ext = [o.lo[rest[0]], o.lo[rest[0]] + o.length[rest[0]], o.lo[rest[1]], o.lo[rest[1]] + o.length[rest[1]]]
        items.append({"output": o, "field": name, "component": comp, "plane": plane, "img": img, "lo": lo, "hi": hi,
                      "log": log, "cmap": cmap, "symmetric": symmetric, "ext": ext, "rest": rest, "idx": idx,
                      "finite": finite, "voids": p.get("voids"), "family": o.gradient_family(name, grad)})
    qualify = any(it["family"] and G.estimator_label(it["output"]) == "PS-DTFE" for it in items if it is not None)
    for it in items:
        if it is not None:
            it["qualify"] = qualify
    if req["shared"] or req["merged"]:
        groups: dict = {}
        for it in items:
            if it is not None:
                groups.setdefault(quantity(it), []).append(it)
        for its in groups.values():
            full = [i for i in its if i["finite"]] or its        # an all-empty panel takes the others' range
            lo, hi = min(i["lo"] for i in full), max(i["hi"] for i in full)
            if its[0]["symmetric"]:
                m = max(abs(lo), abs(hi))
                lo, hi = -m, m
            for i in its:
                i["lo"], i["hi"] = lo, hi
    return items


def quantity(it: dict) -> tuple:
    """What a grid panel shows, for its colour range, colour bar and name: (field, component, family of velocity
    derivatives: '' / 'mean' / 'stream')."""
    return it["field"], it["component"], it.get("family", "")


def panel_names(it: dict) -> tuple[str, str, str, str]:
    """(field label, component text, who, when) of a panel; the label says the velocity derivatives' family
    (grids.field_label, the item's 'qualify')."""
    o, name, comp = it["output"], it["field"], it["component"]
    label = G.field_label(o, name, it.get("family", ""), it.get("qualify"))
    ctext = "" if comp in ("value", "") else f" [{comp}]"
    who = o.sim or o.dir.name
    when = f"z = {o.redshift:.2f}" if o.redshift is not None else (o.title.split(" · ")[1] if " · " in o.title else "")
    return label, ctext, who, when


def type_tag(o) -> str:
    """What tells two runs apart in a panel's label: the estimator with a non-default prefix
    (grids.type_label: 'DTFE', 'PS-DTFE', 'PS-DTFE · ps_mw' -- the Type column's words, not the raw
    prefix 'output' that says nothing to a reader of the saved figure); the prefix for a bare object."""
    if hasattr(o, "files") and getattr(o, "prefix", ""):
        return G.type_label(o)
    return getattr(o, "prefix", "") or ""


def grid_differences(items: list) -> dict:
    """What differs between the panels: the sets of outputs, simulations, quantities and (who, when);
    'by_prefix' = the run has to be named: outputs of one snapshot (the prefix tells them apart), or
    two estimators in the grid (a DTFE panel next to a PS-DTFE one at another redshift said nothing)."""
    live = [it for it in items if it is not None]
    outs = {str(it["output"].root) for it in live}
    moments = {panel_names(it)[2:] for it in live}
    types = {type_tag(it["output"]) for it in live}
    return {"outs": outs, "sims": {panel_names(it)[2] for it in live},
            "quantities": {quantity(it) for it in live},
            "by_prefix": len(outs) > len(moments) or len(types) > 1}


def panel_label(it: dict, diff: dict) -> str:
    """The short text inside a panel: only what differs between the panels (the redshift, the simulation
    when several, the field when several, the run -- its estimator and prefix -- when two runs of one
    snapshot or two estimators are shown)."""
    label, ctext, who, when = panel_names(it)
    o = it["output"]
    tag = f" · {type_tag(o)}" if diff["by_prefix"] and type_tag(o) else ""
    parts = []
    if len(diff["outs"]) > 1:
        parts.append((f"{who} · " if len(diff["sims"]) > 1 else "") + (when or who) + tag)
    if len(diff["quantities"]) > 1:
        parts.append(label + ctext)
    return "\n".join(parts) or (when or who)


def panel_void_circles(it: dict, axis: int, n0: float, n1: float, radius: str = "r_v") -> list:
    """A grid panel's void outlines (it['voids']: the panel's own catalogue, its shown rows) in the panel's pixels
    (n0 x n1 over its extent; void_circles' convention), [] without a catalogue or in 2D."""
    v = it.get("voids")
    o = it["output"]
    if not v or v.get(radius) is None or o.dim != 3:
        return []
    pos = float(o.lo[axis] + (it["idx"] + 0.5) * o.length[axis] / o.n)
    e = it["ext"]
    circles = G.void_circles(v["centres"], v[radius], v["box"], axis, pos, it["rest"], (e[0], e[2]), (e[1], e[3]),
                             n0, n1)
    return [(x, y, rx, ry, int(v["rows"][k])) for x, y, rx, ry, k in circles]


def grid_tile_circles(req: dict, items: list, max_tile: int = 1024) -> tuple[list, list, list]:
    """The outlines over grid_tiles' picture: (circles, dashed, clips) in its image pixels, each panel's offset by
    its tile and clipped to it (clips: one (x0, y0, w, h) per circle, the R_v ones first, then the R_eff ones)."""
    cols = req["cols"]
    sizes = [it["img"].shape[0] for it in items if it is not None]
    if not sizes:
        return [], [], []
    T = min(max(sizes), max_tile)                   # grid_tiles' tile size and gaps
    gap = 0 if req["merged"] else 6
    solid, dashed, clip_s, clip_d = [], [], [], []
    for k, it in enumerate(items):
        if it is None:
            continue
        r, c = divmod(k, cols)
        y0, x0 = r * (T + gap), c * (T + gap)
        for radius, out, clips in (("r_v", solid, clip_s), ("r_eff", dashed, clip_d)):
            for x, y, rx, ry, row in panel_void_circles(it, req["axis"], T, T, radius):
                out.append((x0 + x, y0 + y, rx, ry, row))
                clips.append((x0, y0, T, T))
    return solid, dashed, clip_s + clip_d


def grid_tiles(req: dict, items: list, max_tile: int = 1024) -> tuple[np.ndarray, list, tuple | None]:
    """The screen's picture: every panel coloured straight to pixels (colorize, the single map's way) and
    tiled, thin white gaps unless merged, resampled to one tile size (nearest). Returns (rgb, labels,
    bar): labels = [(x, y, text)] in image pixels, bar = (cmap, vrange, label, log) when every panel shows one
    quantity on one range (shown below the picture), else None."""
    rows, cols, clip = req["rows"], req["cols"], req["clip"]
    sizes = [it["img"].shape[0] for it in items if it is not None]
    T = min(max(sizes), max_tile)
    gap = 0 if req["merged"] else 6
    out = np.full((rows * T + gap * (rows - 1), cols * T + gap * (cols - 1), 3), 255, dtype=np.uint8)
    diff = grid_differences(items)
    labels = []
    for k, it in enumerate(items):
        r, c = divmod(k, cols)
        y0, x0 = r * (T + gap), c * (T + gap)
        if it is None:
            out[y0:y0 + T, x0:x0 + T] = 238
            continue
        img = it["img"]
        if img.shape[0] != T or img.shape[1] != T:    # resample first, colour T^2 pixels, not the whole plane
            ri = (np.arange(T) * img.shape[0]) // T
            ci = (np.arange(T) * img.shape[1]) // T
            sub = img[ri][:, ci]
            if it["log"]:                             # the full plane's floor for the empty cells (colorize's
                pos = img[img > 0]                    # own would be the sample's): the full plane's pixels
                sub = np.where(sub > 0, sub, pos.min() if pos.size else 1.0)
            img = sub
        rgb, _ = G.colorize(img, it["cmap"], log=it["log"], clip=(clip, 100 - clip),
                            symmetric=it["symmetric"], vrange=(it["lo"], it["hi"]))
        out[y0:y0 + T, x0:x0 + T] = rgb
        labels.append((x0 + 8, y0 + 8, panel_label(it, diff)))
    bar = None
    if len(diff["quantities"]) == 1 and (req["shared"] or req["merged"] or len(diff["outs"]) == 1):
        it0 = next(it for it in items if it is not None)
        label, ctext, _w, _t = panel_names(it0)
        bar = (it0["cmap"], (it0["lo"], it0["hi"]), ("log₁₀ " if it0["log"] else "") + label + ctext, it0["log"])
    return out, labels, bar


def grid_unit(items: list, unit: str) -> tuple[str, str]:
    """(the figure grid's axes unit, a note when it is not the one asked for): Mpc/h only when the h of EVERY panel's
    box is known, else Mpc in every panel, said (not one panel in Mpc alone, unsaid,
    beside panels in Mpc/h); the remembered choice stays Mpc/h for the outputs that know h."""
    if unit != "mpc_h" or all(G.hubble(it["output"]) for it in items if it is not None):
        return unit, ""
    return "mpc", "the h of a panel's box is not known: the axes are in Mpc"


def grid_suptitle(req: dict, items: list, position: str | None = None) -> str:
    """The figure grid's own title, what Save figure… writes above the panels: merged, what every panel shares (the
    quantity, the simulation) and the slice; else the slice alone ('' for 2D panels). 'position' in place of the
    request's (the map's 'z = 32.50 Mpc'): a saved figure's in its axes' unit."""
    diff = grid_differences(items)
    pdim = next((it["output"].dim for it in items if it), 3)
    where = f" ({req['position'] if position is None else position})" if req.get("position") else ""
    slice_text = f"slice normal to {_ex.AXES[req['axis']]} at {100 * req['rel']:.0f}% of the box{where}" if pdim == 3 else ""
    if not req["merged"]:
        return slice_text
    head = []
    if len(diff["quantities"]) == 1:
        label, ctext, _w, _t = panel_names(next(i for i in items if i))
        head.append(label + ctext)
    if len(diff["sims"]) == 1:
        head.append(next(iter(diff["sims"])))
    if slice_text:
        head.append(slice_text)
    return "   ·   ".join(head)


def grid_figure(req: dict, items: list, dpi: float = 200.0, opts: dict | None = None):
    """The full matplotlib figure (what Save figure… writes): axes in Mpc, per-panel titles and colour bars,
    or -- merged -- touching panels with inside labels and the bars in one column at the right, one per
    quantity. 'opts' = the Save options (figsave.options; None = the defaults):
    the title, the panel titles, frame, ticks, tick labels, axis names and their unit, the colour bars and their
    labels, the void outlines, the font size, the width; margins 'none' tiles the panels over the whole figure
    with nothing around them. Returns (Figure, panel count)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    o_ = FS.options(opts)
    o_["unit"], _note = grid_unit(items, o_["unit"])    # one unit in every panel
    rows, cols, merged, axis = req["rows"], req["cols"], req["merged"], req["axis"]
    fs = float(o_["font_scale"])
    bare = o_["margins"] == "none"          # the panels fill the figure: no title, axes or colour bars
    bars_on = o_["colorbar"] and not bare
    tile = merged or bare                   # panels touching on a gridspec of fixed margins (no tight_layout)
    if tile:                                # cells shaped like the data, no gap at wspace = hspace = 0
        e0 = next(i for i in items if i)["ext"]
        aspect = (e0[3] - e0[2]) / (e0[1] - e0[0]) if e0[1] > e0[0] and e0[3] > e0[2] else 1.0
        pw, ph = 4.2, 4.2 * float(np.clip(aspect, 0.25, 4.0))
        if bare:
            ml = mr = mb = mt = 0.0
        else:                                # margins in inches: left, the bar column, bottom, top
            ml, mr, mb, mt = 0.75, (1.3 if bars_on else 0.2), 0.6, (0.55 if o_["title"] else 0.2)
        size_in = (pw * cols + ml + mr, ph * rows + mb + mt)
    else:
        size_in = (4.8 * cols, 4.2 * rows)
    fig = Figure(figsize=size_in, dpi=dpi)
    FigureCanvasAgg(fig)
    W, H = size_in
    if tile:
        left, right_edge, bottom, top = ml / W, 1.0 - mr / W, mb / H, 1.0 - mt / H
        gs = fig.add_gridspec(rows, cols, left=left, right=right_edge, bottom=bottom, top=top, wspace=0.0, hspace=0.0)
        axes = gs.subplots(sharex=merged, sharey=merged, squeeze=False)
    else:
        axes = fig.subplots(rows, cols, squeeze=False)
    diff = grid_differences(items)
    shown = 0
    bars: dict = {}                         # merged: (field, component) -> the first panel's image, in order
    for k, it in enumerate(items):
        ax = axes[k // cols][k % cols]
        if it is None:
            ax.set_axis_off()
            continue
        shown += 1
        o = it["output"]
        um, _note = FS.unit_map(o, it["rest"], o_["unit"], G.hubble(o) if o_["unit"] == "mpc_h" else None)
        e = um.ext(it["ext"])
        data = it["plane"].T
        if it["log"]:
            data = np.log10(np.where(data > 0, data, np.nan))
        hi = it["hi"] if it["hi"] > it["lo"] else it["lo"] + 1.0
        im = ax.imshow(data, origin="lower", extent=e, cmap=it["cmap"], vmin=it["lo"], vmax=hi,
                       aspect="equal", interpolation="nearest")
        if it.get("voids") and o_["voids"]:  # the panel's own voids, as on screen
            to_mpc = lambda cs: [(e[0] + x, e[3] - y, rx, k) for x, y, rx, _ry, k in cs]   # noqa: E731
            G.draw_void_circles_mpl(ax, to_mpc(panel_void_circles(it, axis, e[1] - e[0], e[3] - e[2])),
                                    to_mpc(panel_void_circles(it, axis, e[1] - e[0], e[3] - e[2], "r_eff")))
        label, ctext, who, when = panel_names(it)
        if bare:
            ax.set_axis_off()
        else:
            FS.style_axes(ax, o_, (_ex.AXES[it["rest"][0]], _ex.AXES[it["rest"][1]]), um.name, fontsize=8 * fs,
                          ticksize=7 * fs)
        tag = f" · {type_tag(o)}" if diff["by_prefix"] and type_tag(o) else ""
        if tile:
            if merged and bars_on:
                bars.setdefault(quantity(it), (im, it))
            if o_["panel_titles"]:
                ax.text(0.03, 0.97, panel_label(it, diff), transform=ax.transAxes, va="top", ha="left",
                        fontsize=8 * fs, color="white", bbox=dict(facecolor="black", alpha=0.45, edgecolor="none",
                                                                  pad=2))
        else:
            if o_["panel_titles"]:
                ax.set_title(f"{label}{ctext}\n{who}" + (f" · {when}" if when else "") + tag, fontsize=9 * fs)
            if bars_on:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                if o_["colorbar_label"]:
                    cb.set_label(("log₁₀ " if it["log"] else "") + label + ctext, fontsize=8 * fs)
                cb.ax.tick_params(labelsize=7 * fs)
    auto = shown = grid_suptitle(req, items)
    first = next((it["output"] for it in items if it is not None), None)
    if o_["unit"] != "mpc" and req.get("position_at") and first is not None:
        # the slice position in the axes' unit, converted with the first panel's box (its h, its cells)
        a, v = req["position_at"]
        val, name = FS.position(first, a, v, o_["unit"], G.hubble(first) if o_["unit"] == "mpc_h" else None)
        shown = grid_suptitle(req, items, f"{_ex.AXES[a]} = {val:.2f} {name}")
    if bare:
        pass                                # nothing around the panels
    elif merged:
        # tick and axis labels on the outer edges only -- of the panels DRAWN: an empty panel in the bottom
        # row (three outputs in a 2 x 2) must not leave its column without an x axis
        drawn = [[axes[r][c].axison for c in range(cols)] for r in range(rows)]
        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                if not ax.axison:
                    continue
                lowest = not any(drawn[rr][c] for rr in range(r + 1, rows))
                leftmost = not any(drawn[r][cc] for cc in range(c))
                ax.tick_params(labelbottom=lowest and o_["tick_labels"], labelleft=leftmost and o_["tick_labels"])
                if not lowest:
                    ax.set_xlabel("")
                if not leftmost:
                    ax.set_ylabel("")
        n = len(bars)
        gap = (0.25 / H) if n > 1 else 0.0
        span = top - bottom
        h = (span - gap * (n - 1)) / max(n, 1)
        cb_x, cb_w = right_edge + 0.12 / W, 0.16 / W
        for i, (key, (im, it)) in enumerate(bars.items()):
            cax = fig.add_axes([cb_x, top - (i + 1) * h - i * gap, cb_w, h])
            cb = fig.colorbar(im, cax=cax)
            label, ctext, _w, _t = panel_names(it)
            text = ("log₁₀ " if it["log"] else "") + label + ("\n" + ctext.strip() if ctext else "")
            size = 8 * fs * float(np.clip(h * H / 1.6, 0.6, 1.0))   # a short bar (many of them): smaller text
            if o_["colorbar_label"]:
                cb.set_label(text, fontsize=size)
            cb.ax.tick_params(labelsize=min(7 * fs, size))
        if o_["title"]:
            fig.suptitle(FS.title_text(o_, auto, shown), fontsize=9 * fs, y=1.0 - 0.17 / H)
    else:
        title = FS.title_text(o_, auto, shown) if o_["title"] else ""
        if title:
            fig.suptitle(title, fontsize=9 * fs, y=0.995)
        fig.tight_layout()
    FS.fit_width(fig, o_, o_["dpi"], relayout=None if tile else fig.tight_layout)
    return fig, shown


def grid_dpi(fig, req: dict, items: list, opts: dict) -> float:
    """The dpi to save the grid figure at: the options', or with 'match the data' the one at which the widest
    panel holds its plane pixel for pixel (figsave.match_dpi: panels too small for 72 dpi shrink the figure, the
    separate panels' tight_layout re-run). The panels are the figure's first rows x cols axes."""
    if not opts["match_data"]:
        return float(opts["dpi"])
    panels = fig.axes[:req["rows"] * req["cols"]]
    tile = req["merged"] or opts["margins"] == "none"       # grid_figure's fixed margins: no tight_layout there
    return FS.match_dpi(fig, [(ax, it["img"].shape[1]) for ax, it in zip(panels, items) if it is not None],
                        relayout=None if tile else fig.tight_layout)


def grid_full_pixels(req: dict, items: list, opts: dict):
    """The grid at full resolution: (rgb, overlay or None) for figsave.write_pixels -- the screen's tiles (grid_tiles,
    the colours and ranges as shown) with each panel at the largest panel's own pixels instead of 1024 at most, and
    each panel's void outlines over it, clipped to its tile, when the options keep them."""
    sizes = [it["img"].shape[0] for it in items if it is not None]
    T = max(sizes)
    rgb, _labels, _bar = grid_tiles(req, items, max_tile=T)
    if not opts["voids"]:
        return rgb, None
    circles, dashed, clips = grid_tile_circles(req, items, max_tile=T)
    if not (circles or dashed):
        return rgb, None
    return rgb, lambda ax: FS.draw_pixel_circles(ax, rgb.shape, circles, dashed, clips)


class GridMixin:
    # ---------------------------------------------------------------- the figure grid (compare)
    def _grid_layout_changed(self, *_):
        self._grid_sync_rows()
        if self._grid_shown:
            self.show_grid(True)

    def grid_dims(self) -> tuple[int, int]:
        """The layout as (rows, columns)."""
        r, _x, c = (self.grid_layout.currentData() or "1x2").partition("x")
        return int(r), int(c)

    def grid_set_layout(self, rows: int, cols: int) -> bool:
        j = self.grid_layout.findData(f"{rows}x{cols}")
        if j >= 0:
            self.grid_layout.setCurrentIndex(j)
        return j >= 0

    def _grid_sync_rows(self):
        """One table row per panel of the layout (rows beyond it hidden, their choices kept); the output
        menus list every output known (a refresh keeps each row's choice by its root). Never recomposes
        by itself: the callers do, once."""
        rows, cols = self.grid_dims()
        want = rows * cols
        while self.grid_table.rowCount() < max(want, self.grid_table.rowCount()):
            self._grid_make_row(self.grid_table.rowCount())
        hold, self._grid_shown_hold = getattr(self, "_grid_shown_hold", False), True
        try:
            self._grid_sync_rows_inner(want)
        finally:
            self._grid_shown_hold = hold
        self._grid_update_merge()

    def _grid_sync_rows_inner(self, want: int):
        for i in range(self.grid_table.rowCount()):
            self.grid_table.setRowHidden(i, i >= want)
            self._grid_row_sims(i)

    def _grid_make_row(self, i: int):
        self.grid_table.insertRow(i)
        for c in range(len(_ex.GRID_COLS)):
            w = QComboBox()
            w.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(6 if c < _ex._C_FLD else 8)
            self.grid_table.setCellWidget(i, c, w)
        cw = self.grid_table.cellWidget
        cw(i, _ex._C_SIM).currentIndexChanged.connect(lambda _j, i=i: self._grid_row_snaps(i))
        cw(i, _ex._C_SNAP).currentIndexChanged.connect(lambda _j, i=i: self._grid_row_types(i))
        cw(i, _ex._C_TYPE).currentIndexChanged.connect(lambda _j, i=i: self._grid_type_picked(i))
        cw(i, _ex._C_FLD).currentIndexChanged.connect(lambda _j, i=i: self._grid_components_changed(i))
        cw(i, _ex._C_CMP).currentIndexChanged.connect(lambda _j, i=i: (self._grid_row_tips(i), self._grid_row_edited()))

    def _grid_type_picked(self, i: int):
        """The user chose a row's Type: remembered apart from the menu, so a rebuild's fallback (a snapshot
        without that run) never replaces the choice and stepping back to one that has it brings it back.
        Rebuilds block the menu's signals, so only a user's pick lands here."""
        typ = self.grid_table.cellWidget(i, _ex._C_TYPE)
        if typ.currentData():
            self._grid_type_pref[i] = (typ.currentData(Qt.UserRole + 1), typ.currentData(Qt.UserRole + 2), typ.currentData())
        self._grid_fields_changed(i, keep_field=True)

    def _grid_row_tips(self, i: int):
        """Each cell menu's tooltip = its full text: the five columns share the pane, and a closed Type menu
        clips 'PS-DTFE · ps_vwsh' to the estimator -- the part that tells two runs apart. Type adds the root."""
        for c in range(len(_ex.GRID_COLS)):
            w = self.grid_table.cellWidget(i, c)
            if w is None:
                continue
            tip = w.currentText()
            if c == _ex._C_TYPE and w.currentData():
                tip += f"\n{w.currentData()}.*"
            w.setToolTip(tip)

    @staticmethod
    def _grid_snap_key(o):
        """The Redshift menu's key of an output: the snapshot number in a simulation (a folder whose name
        gives none: the folder itself, so two renamed snapdirs never share one '(?)' entry); for a custom
        run the snapshot FILE it read (the sidecar's input_file: the DTFE and PS-DTFE runs of one snapshot
        meet in the Type menu whatever folder they wrote to, and two snapshots run into one folder --
        the Custom tab's default ~/DTFE-output -- stay apart); else the folder (a file opened by hand
        without a sidecar). Cached on the object (a refresh makes new objects)."""
        cached = getattr(o, "_grid_key", None)
        if cached is not None:
            return cached
        if o.sim:
            key = o.snap if o.snap is not None else str(o.dir)
        else:
            f = (G.sidecar(o.root) or {}).get("input_file")
            key = str(Path(f).expanduser().resolve()) if f else str(o.dir)
        o._grid_key = key
        return key

    def _grid_row_sims(self, i: int):
        """Row i's Simulation menu: the simulations in natural order, then the custom snapshots and the
        opened files; '—' leaves the panel blank. Keeps the row's choice, and (a refresh) its exact output."""
        sim, typ = self.grid_table.cellWidget(i, _ex._C_SIM), self.grid_table.cellWidget(i, _ex._C_TYPE)
        keep, keep_root = sim.currentData(), typ.currentData()
        kept = self._by_root.get(keep_root or "")
        if kept is not None:                    # a refresh re-lists an opened file under its simulation or the
            keep = self._group(kept)            # custom snapshots: the row follows its output, not its old group
        outs = [o for _, o in self.outputs]
        sim.blockSignals(True)
        sim.clear()
        sim.addItem("—", _ex.GRID_NONE)
        for name in sorted({o.sim for o in outs if o.sim}, key=G.sim_sort_key):
            sim.addItem(name, name)
        for group, text in ((_ex.GROUP_CUSTOM, "Custom snapshots"), (_ex.GROUP_OPENED, "Opened files")):
            if any(self._group(o) == group for o in outs):
                sim.addItem(text, group)
        j = sim.findData(keep) if keep else 0
        sim.setCurrentIndex(j if j >= 0 else 0)
        sim.blockSignals(False)
        self._grid_row_snaps(i, keep_root=keep_root)

    def _grid_row_snaps(self, i: int, keep_root: str | None = None):
        """Row i's Redshift menu for its simulation: the snapshots by redshift, earliest first (today
        last, the default), or the custom snapshots' and opened files' folders by name. Keeps the row's
        snapshot when the new simulation has it; 'keep_root' (a refresh, a fill) picks that output's."""
        sim, snap = self.grid_table.cellWidget(i, _ex._C_SIM), self.grid_table.cellWidget(i, _ex._C_SNAP)
        grp = sim.currentData()
        kept = self._by_root.get(keep_root or "")
        want = self._grid_snap_key(kept) if kept is not None else snap.currentData()
        outs = [o for _, o in self.outputs if grp and grp != _ex.GRID_NONE and self._group(o) == grp]
        snap.blockSignals(True)
        snap.clear()
        first: dict = {}
        for o in outs:
            first.setdefault(self._grid_snap_key(o), o)
        if grp in (_ex.GROUP_CUSTOM, _ex.GROUP_OPENED):
            keys = sorted(first, key=lambda k: (G.sim_sort_key(Path(k).name), k))
            labels = [Path(k).name for k in keys]
            depth = 1                           # two 'snapdir_099/combined_099.hdf5' read with as many parents as it takes
            while len(set(labels)) < len(labels) and depth < 8:
                depth += 1
                labels = [lab if Counter(labels)[lab] == 1 else "/".join(Path(k).parts[-depth:]) for lab, k in zip(labels, keys)]
            for key, lab in zip(keys, labels):
                snap.addItem(lab if len(set(labels)) == len(labels) else key, key)
                snap.setItemData(snap.count() - 1, key, Qt.ToolTipRole)
        else:                                   # snapshot numbers first, then the folders a number was not read from
            for key in sorted(first, key=lambda k: (isinstance(k, str), k if isinstance(k, int) else 0, str(k))):
                o = first[key]
                z = f"z = {o.redshift:.2f} " if o.redshift is not None else ""
                snap.addItem(z + (f"({key:03d})" if isinstance(key, int) else f"({Path(key).name})"), key)
                snap.setItemData(snap.count() - 1, f"snapshot {key:03d}" if isinstance(key, int) else str(key), Qt.ToolTipRole)
        j = snap.findData(want) if want is not None else -1
        if j < 0:                               # the default: the newest NUMBERED snapshot (today), not an unnumbered folder
            nums = [k for k in range(snap.count()) if isinstance(snap.itemData(k), int)]
            j = nums[-1] if nums else snap.count() - 1
        snap.setCurrentIndex(j)
        snap.setEnabled(snap.count() > 0)
        snap.blockSignals(False)
        self._grid_row_types(i, keep_root=keep_root)

    def _grid_row_types(self, i: int, keep_root: str | None = None):
        """Row i's Type menu: the outputs of its snapshot -- the standard DTFE, the phase-space run, other
        prefixes (grids.type_label; two runs with one label get their folder). Keeps the exact output
        ('keep_root': a refresh, a fill), else the row's remembered choice (_grid_type_pref: the user's
        pick or the fill's) by its root, else its prefix WITH its estimator, else its estimator, else its
        prefix, else the Output menus' preference (ps_output first): a row set to DTFE stays DTFE when its
        redshift changes, and a run of the same name as another estimator's (custom runs in two folders)
        comes back after a snapshot without it. Ends in the field menu, which recomposes once."""
        sim, snap, typ = (self.grid_table.cellWidget(i, c) for c in (_ex._C_SIM, _ex._C_SNAP, _ex._C_TYPE))
        grp, key = sim.currentData(), snap.currentData()
        prev_prefix, prev_kind, prev_root = self._grid_type_pref.get(
            i, (typ.currentData(Qt.UserRole + 1), typ.currentData(Qt.UserRole + 2), typ.currentData()))
        outs = [o for _, o in self.outputs if grp and grp != _ex.GRID_NONE and key is not None
                and self._group(o) == grp and self._grid_snap_key(o) == key]
        outs.sort(key=lambda o: G.type_order(o, self._pref_prefix))
        labels = [G.type_label(o) for o in outs]
        dup = Counter(labels)
        if any(c > 1 for c in dup.values()):            # two runs of one name and estimator: their folders tell
            labels = [lab if dup[lab] == 1 else f"{lab} ({o.dir.name})" for lab, o in zip(labels, outs)]
            dup2 = Counter(labels)
            labels = [lab if dup2[lab] == 1 else f"{G.type_label(o)} ({o.dir})" for lab, o in zip(labels, outs)]
        typ.blockSignals(True)
        typ.clear()
        for o, lab in zip(outs, labels):
            typ.addItem(lab, str(o.root))
            k = typ.count() - 1
            typ.setItemData(k, o.prefix, Qt.UserRole + 1)
            typ.setItemData(k, G.estimator_label(o), Qt.UserRole + 2)
            typ.setItemData(k, str(o.root) + ".*", Qt.ToolTipRole)
        j = typ.findData(keep_root) if keep_root else -1
        if j < 0 and prev_root:
            j = typ.findData(prev_root)
        if j < 0 and prev_prefix and prev_kind:
            j = next((k for k, o in enumerate(outs) if o.prefix == prev_prefix and G.estimator_label(o) == prev_kind), -1)
        if j < 0 and prev_kind:
            j = next((k for k, o in enumerate(outs) if G.estimator_label(o) == prev_kind), -1)
        if j < 0 and prev_prefix:
            j = next((k for k, o in enumerate(outs) if o.prefix == prev_prefix), -1)
        if j < 0 and outs:
            j = self._preferred_prefix([o.prefix for o in outs])
        typ.setCurrentIndex(j)
        typ.setEnabled(typ.count() > 1)
        typ.blockSignals(False)
        self._grid_fields_changed(i, keep_field=True)

    def _grid_fields_changed(self, i: int, keep_field: bool = False):
        out, fld = self.grid_table.cellWidget(i, _ex._C_TYPE), self.grid_table.cellWidget(i, _ex._C_FLD)
        o = self._by_root.get(out.currentData() or "")
        keep = fld.currentData() if keep_field else None
        fld.blockSignals(True)
        fld.clear()
        if o is not None:
            for name in o.fields():
                fld.addItem(G.FIELD_LABELS.get(name, name), name)
        j = fld.findData(keep) if keep else -1
        fld.setCurrentIndex(j if j >= 0 else 0)
        fld.setEnabled(o is not None)
        fld.blockSignals(False)
        self._grid_components_changed(i, keep_component=keep_field)

    def _grid_components_changed(self, i: int, keep_component: bool = False):
        out, fld, cmp = (self.grid_table.cellWidget(i, c) for c in (_ex._C_TYPE, _ex._C_FLD, _ex._C_CMP))
        o = self._by_root.get(out.currentData() or "")
        name = fld.currentData()
        keep = cmp.currentText() if keep_component else None
        cmp.blockSignals(True)
        cmp.clear()
        if o is not None and name in o.files:
            cmp.addItems(G.components(name, o.files[name].ncomp, o.dim))
        if keep and cmp.findText(keep) >= 0:
            cmp.setCurrentText(keep)
        cmp.setEnabled(cmp.count() > 1)
        cmp.blockSignals(False)
        self._grid_row_tips(i)
        self._grid_row_edited()

    def _grid_row_edited(self):
        self._grid_update_merge()
        if self._grid_shown:
            self.show_grid(True)

    def _grid_refresh_soon(self):
        """The shown grid again (a slice step, a colour change): a request to the worker, which serves
        only the latest one -- a held arrow key cannot pile up."""
        if self._grid_shown and not self._shut:
            self.show_grid(True)

    def _grid_merge_applicable(self) -> bool:
        """Every panel with an output comes from a box of one size along the shown axes (the merged axes and
        the shared colour bar need that); False when no panel has an output."""
        axis = self.axis.currentIndex()
        boxes = set()
        for pnl in self.grid_panels():
            if pnl is None:
                continue
            o = pnl[0]
            rest = [a for a in range(3) if a != axis] if o.dim == 3 else [0, 1]
            boxes.add((o.dim,) + tuple(round(float(v), 6) for v in np.r_[np.asarray(o.lo)[rest], np.asarray(o.length)[rest]]))
        return len(boxes) == 1

    def _grid_update_merge(self):
        """'merge the axes' is offered only for panels from boxes of one size; merged, the shared range is implied."""
        ok = self._grid_merge_applicable()
        self.grid_merge.setEnabled(ok)
        self.grid_merge.setToolTip(_ex.GRID_MERGE_TIP if ok else
                                   _ex.GRID_MERGE_TIP + "\n(greyed out: the panels come from boxes of different sizes)")
        merged = ok and self.grid_merge.isChecked()
        self.grid_shared.setEnabled(not merged)
        user = getattr(self, "_grid_shared_user", None)
        if merged and user is None:             # forced on while merged; the user's own choice comes back after
            self._grid_shared_user = self.grid_shared.isChecked()
            self.grid_shared.blockSignals(True)
            self.grid_shared.setChecked(True)
            self.grid_shared.blockSignals(False)
        elif not merged and user is not None:
            self._grid_shared_user = None
            self.grid_shared.blockSignals(True)
            self.grid_shared.setChecked(bool(user))
            self.grid_shared.blockSignals(False)

    def _grid_merge_toggled(self, _on):
        self._grid_update_merge()
        if self._grid_shown:
            self.show_grid(True)

    def grid_row(self, i: int):
        """Panel i as (OutputSet, field, component), or None for an empty panel."""
        out, fld, cmp = (self.grid_table.cellWidget(i, c) for c in (_ex._C_TYPE, _ex._C_FLD, _ex._C_CMP))
        o = self._by_root.get(out.currentData() or "")
        if o is None or fld.currentData() is None:
            return None
        return o, fld.currentData(), cmp.currentText() or "value"

    def grid_set_row(self, i: int, o, field: str | None = None, component: str | None = None):
        """Row i = the output o (None: blank), its simulation, snapshot and type menus set to it, then
        the field and component when given (else the row's own, kept when o has them)."""
        sim, fld, cmp = (self.grid_table.cellWidget(i, c) for c in (_ex._C_SIM, _ex._C_FLD, _ex._C_CMP))
        hold, self._grid_shown_hold = getattr(self, "_grid_shown_hold", False), True
        try:                                    # held: the cascade's recomposes (the old field, the first component) wait
            if o is not None:
                self._grid_type_pref[i] = (o.prefix, G.estimator_label(o), str(o.root))
            else:
                self._grid_type_pref.pop(i, None)
            sim.blockSignals(True)
            j = sim.findData(self._group(o)) if o is not None else 0
            sim.setCurrentIndex(j if j >= 0 else 0)
            sim.blockSignals(False)
            self._grid_row_snaps(i, keep_root=str(o.root) if o is not None else None)
            if field is not None and fld.findData(field) >= 0:
                fld.blockSignals(True)
                fld.setCurrentIndex(fld.findData(field))
                fld.blockSignals(False)
                self._grid_components_changed(i)
            if component is not None and cmp.findText(component) >= 0:
                cmp.blockSignals(True)
                cmp.setCurrentText(component)
                cmp.blockSignals(False)
                self._grid_row_tips(i)
        finally:
            self._grid_shown_hold = hold
        self._grid_row_edited()                 # once, from the final state (nothing under a fill's own hold)

    def grid_panels(self) -> list:
        rows, cols = self.grid_dims()
        return [self.grid_row(i) for i in range(rows * cols)]

    def grid_fill(self, mode: str):
        """Fill the panels from the map: 'map' = this output, field and component in every panel;
        'redshifts' = this field at this simulation's snapshots that have it, the latest ones, oldest
        first; 'fields' = this output's fields in the order of the field list, starting at the map's;
        'types' = this field from every output of this snapshot (DTFE, PS-DTFE, other prefixes)."""
        o, name, comp = self.current, self.field.currentData(), self.component.currentText()
        if o is None or name is None:
            return
        rows, cols = self.grid_dims()
        n = rows * cols
        plan: list = []
        if mode == "redshifts":
            sims = [x for _t, x in self.outputs if self._group(x) == self._group(o) and name in x.fields()]
            if o.sim:                           # one output per snapshot: the map's prefix AND estimator
                same = [x for x in sims if x.prefix == o.prefix and G.estimator_label(x) == G.estimator_label(o)]
            else:                               # custom/opened runs: the series run into THIS folder with this estimator
                same = [x for x in sims if str(x.dir) == str(o.dir) and G.estimator_label(x) == G.estimator_label(o)]
            if len(same) > 1:                   # (a custom run's prefix is its name: X in two folders is one snapshot twice)
                sims = same
            sims.sort(key=lambda x: (self._snap_key(x) if x.sim else 0, str(x.root)))
            plan = [(x, name, comp) for x in sims[-n:]]
        elif mode == "fields":
            fields = list(o.fields())
            k = fields.index(name) if name in fields else 0
            fields = fields[k:] + fields[:k]
            plan = [(o, f, None) for f in fields[:n]]
        elif mode == "types":                   # this snapshot's outputs: DTFE, PS-DTFE, other prefixes
            outs = [x for _t, x in self.outputs if self._group(x) == self._group(o)
                    and self._grid_snap_key(x) == self._grid_snap_key(o) and name in x.fields()]
            order = lambda x: G.type_order(x, self._pref_prefix)      # noqa: E731
            outs.sort(key=order)
            # the map's own run first, then the other estimator's best (its default prefix, else the preferred
            # one), then the rest: plain type order would show 'output' beside ps_mw (an A/B run) while the map
            # is on ps_output -- the production run and the map's own run both dropped
            pick = [x for x in outs if self._same(x, o)]
            pick += [x for x in outs if G.estimator_label(x) != G.estimator_label(o)][:1]
            roots = {str(x.root) for x in pick}
            pick += [x for x in outs if str(x.root) not in roots]
            plan = [(x, name, comp) for x in sorted(pick[:n], key=order)]
        else:
            plan = [(o, name, comp)] * n
        with_signals = self._grid_shown
        hold, self._grid_shown_hold = getattr(self, "_grid_shown_hold", False), True
        try:
            for i in range(n):
                if i < len(plan):
                    self.grid_set_row(i, *plan[i])
                else:
                    self.grid_set_row(i, None)
        finally:
            self._grid_shown_hold = hold
        self._grid_update_merge()
        if with_signals:
            self.show_grid(True)

    def _grid_cmap(self, name: str) -> str:
        if name == self.field.currentData():
            return self.cmap.currentText()
        return "RdBu_r" if name in G.DIVERGING else "tab10" if name in ("tweb", "vweb") else \
            "magma" if name in ("density", "dispersion") else "viridis"

    def _grid_log(self, name: str) -> bool:
        return self.log.isChecked() if name == self.field.currentData() else name in G.LOG_DEFAULT

    def grid_request(self) -> dict | None:
        """Everything the grid needs, read from the widgets on the GUI thread: the panels with their colour
        rules, the slice (this map's axis and relative position), the range clip, the smoothing, the
        options. None when no panel has an output."""
        panels = self.grid_panels()
        if not any(panels) or self.current is None:
            return None
        cur = self.current
        rows, cols = self.grid_dims()
        axis, index = self.axis.currentIndex(), self.index.value()
        return {"panels": [None if p is None else {"output": p[0], "field": p[1], "component": p[2],
                                                    "cmap": self._grid_cmap(p[1]), "log": self._grid_log(p[1]),
                                                    "voids": self._panel_voids(p[0])}
                           for p in panels],
                "rows": rows, "cols": cols, "axis": axis,
                "ps_gradient": self.ps_gradient.currentData() or G.PS_GRADIENT_DEFAULT,
                "rel": 0.5 if cur.n <= 1 or cur.dim == 2 else index / (cur.n - 1),
                "index": index, "clip": self.clip.value(), "smooth": self.smooth.value(),
                "shared": self.grid_shared.isChecked(),
                "merged": self.grid_merge.isChecked() and self._grid_merge_applicable(),
                "position": self.position.text() if cur.dim == 3 else "",
                # the same place as a number (Mpc, cells without a box): a saved title's in its unit
                "position_at": (axis, float(cur.lo[axis] + (index + 0.5) * cur.length[axis] / cur.n))
                if cur.dim == 3 else None}

    def show_grid(self, on: bool):
        """The toggle: the panels as tiles in place of the map (the worker colours them, like the single
        map's slice), or the map back. Shown already, a new request replaces the tiles when it arrives."""
        if getattr(self, "_grid_shown_hold", False):
            return
        if not on:
            if self._grid_shown:
                self._leave_grid()
                self.render()
            return
        if self.current is None:
            self.grid_toggle.blockSignals(True)
            self.grid_toggle.setChecked(False)
            self.grid_toggle.blockSignals(False)
            return
        if not self._grid_shown and not any(self.grid_panels()):
            self.grid_fill("map")               # turned on with nothing chosen: the map in every panel; shown and
        req = self.grid_request()               # every panel blanked by hand: the warning below, not a refill
        if req is None:
            self.grid_state.setText("⚠ no panel has an output (use a Fill button, or choose outputs in the table)")
            if self._grid_shown:
                self._leave_grid()
                self.render()
            else:
                self.grid_toggle.blockSignals(True)
                self.grid_toggle.setChecked(False)
                self.grid_toggle.blockSignals(False)
            return
        if not self._grid_shown:
            self._grid_shown = True
            self.grid_toggle.blockSignals(True)
            self.grid_toggle.setChecked(True)
            self.grid_toggle.blockSignals(False)
            self.grid_toggle.setText("Back to the map")
            self.view.streams.setVisible(False)     # the picture gets the height; a click lists no streams here
            self.view.readout.setText(" ")
            self._compare_sync()
        if self._grid_busy:
            self._grid_pending = req                # only the latest request is served
            return
        self._grid_busy = True
        self._threads()
        self._to_render_grid.emit(req)

    def _grid_rendered(self, res: dict):
        self._grid_busy = False
        if self._grid_pending is not None:
            req, self._grid_pending = self._grid_pending, None
            self._grid_busy = True
            self._to_render_grid.emit(req)
        if not self._grid_shown or self._shut:
            return
        req = res["request"]
        self._grid_title = res.get("title", "")
        self.view.canvas.setStyleSheet("")
        self.view.canvas.marker = None
        self.view.canvas.arrows = self.view.canvas.circles = self.view.canvas.dashed = None
        self.view.canvas.labels = res["labels"]
        self.view.canvas.circles = res.get("circles") or None       # each panel's own voids
        self.view.canvas.dashed = res.get("dashed") or None
        self.view.canvas.circle_clips = res.get("clips") or None
        self.view.canvas.axes = None                # tiles of several frames: the saved figure has the axes
        self.view.canvas.set_rgb(np.ascontiguousarray(res["rgb"]))
        bar = res["bar"]
        if bar is not None:                         # one quantity: its bar below, as the single map has
            cmap, vrange, label, log = bar
            self.view.colorbar.set(cmap, vrange, label, log)
            self.view.colorbar.setVisible(True)
        else:                                       # several: the saved figure carries one bar per quantity
            self.view.colorbar.setVisible(False)
        self._grid_index = req["index"]
        n, rows, cols = res["shown"], req["rows"], req["cols"]
        pos = req["position"]
        self._title_make = None                     # the map's title function: not this title's
        self.view.title.setText(f"<b>Figure grid</b> {rows} × {cols} — {n} panel{'s' if n != 1 else ''}"
                                + (f" — {pos}" if pos else ""))
        self.grid_state.setText(f"{n} panel{'s' if n != 1 else ''}" + (f" at {pos}" if pos else "")
                                + "; ← → step the slice for every panel, the slider and the colours apply too; "
                                  "Save figure… makes the full figure (" + FS.describe(self.save_options) + ")")
        self.changed.emit()

    def _grid_failed(self, msg: str):
        self.grid_state.setText(f"⚠ could not compose the grid: {msg}")
        if self._grid_pending is not None:          # a newer request waits (a panel was changed meanwhile): its
            req, self._grid_pending = self._grid_pending, None     # result decides whether the grid stays
            self._grid_busy = True
            self._to_render_grid.emit(req)
            return
        self._grid_busy = False
        if self._grid_shown:                        # a panel that cannot be read: the map, not a stale picture
            self._leave_grid()
            self.render()

    def _leave_grid(self):
        self._grid_shown = False
        self._grid_pending = None
        self.view.canvas.circle_clips = None
        self.grid_toggle.blockSignals(True)
        self.grid_toggle.setChecked(False)
        self.grid_toggle.blockSignals(False)
        self.grid_toggle.setText("Show the grid")
        self.view.canvas.labels = None
        self.view.colorbar.setVisible(True)
        self.view.streams.setVisible(True)
        self._compare_sync()
        self.changed.emit()                     # the Explore menu's Figure Grid text follows

    def grid_figure(self, options: dict | None = None):
        """The full matplotlib figure of the current panels (axes in Mpc, titles or inside labels, colour
        bars), composed on this thread at 200 dpi, as the Save options say ('options', else the ones chosen):
        what Save figure… writes. Returns (Figure, panel count)."""
        req = self.grid_request()
        if req is None:
            raise ValueError("no panel has an output (use a Fill button, or choose outputs in the table)")
        return grid_figure(req, grid_items(req), opts=self.save_options if options is None else options)

    def save_grid(self, path, options: dict | None = None) -> str:
        """Write the figure of the current panels as the Save options say ('options', else the ones chosen): PNG,
        PDF, SVG or TIFF by the file's suffix, the decorated figure or the panels at full resolution (no margins with
        'match the data' too: the panels' own pixels, touching as in that figure); the path written. The
        status line says when the axes are not in the unit asked for (grid_unit)."""
        opts = FS.options(self.save_options if options is None else options)
        fmt = FS.format_of(path, opts["format"])
        req = self.grid_request()
        if req is None:
            raise ValueError("no panel has an output (use a Fill button, or choose outputs in the table)")
        items = grid_items(req)
        note = ""
        if FS.writes_pixels(opts):
            # the borderless figure's panels touch: its full-resolution twin has no gaps (merged tiles)
            rgb, overlay = grid_full_pixels(req if opts["kind"] == "full" else dict(req, merged=True), items, opts)
            FS.write_pixels(path, rgb, fmt, overlay)
        else:
            import matplotlib
            _unit, note = grid_unit(items, opts["unit"])
            with matplotlib.rc_context(FS.rc(opts, fmt)):    # the font size and the text kept as text
                fig, _n = grid_figure(req, items, opts=opts)
                dpi = grid_dpi(fig, req, items, opts)
                FS.savefig(fig, str(path), opts, dpi, fmt)     # the pixel guard, the titles' line breaks, savefig
        self.status.emit(f"saved {path}" + (f"; {note}" if note else ""))
        return str(path)

    def _save_grid_dialog(self):
        if self.current is None:
            return
        f, _fmt = self._ask_image_path("Save the figure grid", "figure_grid")
        if not f:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.save_grid(f)
        except Exception as e:
            self.grid_state.setText(f"⚠ could not save the figure: {e}")
        finally:
            QApplication.restoreOverrideCursor()
