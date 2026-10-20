"""The Explore tab's Save image: SaveMixin, the part of ExploreControls that writes the map on screen -- the Save
options (remembered between sessions), the file dialog, the decorated matplotlib figure or the map's own pixels with
its arrows and void outlines, the table of the voids outlined, and the map's values as .npy or .csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog

import explore as _ex
import figsave as FS
import grids as G


class SaveMixin:
    # ---------------------------------------------------------------- saving: the Save options
    def set_save_options(self, opts: dict, remember: bool = True) -> None:
        """Use these Save options from now on; remembered between sessions (without a typed title) when the launcher
        remembers its settings."""
        self.save_options = FS.options(opts)
        if remember and self._settings is not None:
            self._settings.setValue(self.SAVE_KEY, FS.dumps(self.save_options))
        self._save_tips()

    def _save_tips(self):
        d = FS.describe(self.save_options)
        self.view.save_btn.setToolTip(f"Save the map ({d}; Save options… to change), or its values as .npy or .csv")
        self.grid_save_btn.setToolTip(f"Write the composed figure ({d}; Options… to change)")

    def _map_title(self, units=None) -> str:
        """The single map's own title, as the window shows it (the saved figure's title unless one was typed); with
        'units' ((axis, position) -> (value, unit name)) the same title with its slice position and zoom window in
        that unit, a saved figure's axes'."""
        html = self.view.title.text() if units is None or self._title_make is None else self._title_make(units)
        return html.replace("<b>", "").replace("</b>", "")

    def save_context(self) -> dict:
        """What the Save options dialog needs to know of the picture on screen: the grid or a map, arrows and outlines
        shown, the box's h (Mpc/h offered only when known), a box at all, the figure's own title, the full-resolution
        image size in pixels."""
        ctx = {"grid": bool(self._grid_shown), "arrows": False, "voids": False, "hubble": None, "box": True,
               "auto_title": "", "pixels": None}
        if self._grid_shown:
            req = self.grid_request()
            panels = [p for p in (req or {}).get("panels", []) if p]
            hs = [G.hubble(p["output"]) for p in panels]
            ctx["hubble"] = hs[0] if hs and all(hs) else None
            ctx["box"] = all(p["output"].box is not None for p in panels)
            ctx["voids"] = bool(self.view.canvas.circles or self.view.canvas.dashed)
            ctx["auto_title"] = self._grid_title
            if req is not None and panels:
                T = max(p["output"].n for p in panels)
                gap = 0 if req["merged"] else 6
                ctx["pixels"] = (req["cols"] * T + gap * (req["cols"] - 1), req["rows"] * T + gap * (req["rows"] - 1))
        elif self._req is not None and self._plane is not None:
            o = self._req["output"]
            ctx["arrows"] = self._arrows is not None and self._arrows.get("vmax", 0) > 0
            ctx["voids"] = self._void_result() is not None and self.view.canvas.circles is not None
            ctx["hubble"] = G.hubble(o)
            ctx["box"] = o.box is not None
            ctx["auto_title"] = self._map_title()
            ctx["pixels"] = (int(self._plane.shape[0]), int(self._plane.shape[1]))    # plane rows = horizontal axis
        return ctx

    def open_save_options(self) -> bool:
        """The Save options dialog: OK keeps the options for every save from now on (remembered between sessions),
        Save… keeps them and saves the map or the figure grid now. False when cancelled."""
        dlg = FS.make_dialog(self, self.save_options, self.save_context(), can_save=self.can_save())
        if dlg.exec() != QDialog.Accepted:
            return False
        self.set_save_options(dlg.options())
        if dlg.save_now:
            self._save_image()
        return True

    def _ask_image_path(self, title: str, stem: str, extra: list | None = None):
        """The file dialog of a save: the image formats (the options' first) and 'extra' filters. Returns (path,
        format or the extra filter chosen) or (None, None); the format is the name's suffix, else the filter's. A
        format chosen here is the options' format from now on."""
        fmt0 = self.save_options["format"]
        images = FS.file_filters()
        order = [t for f, t in images if f == fmt0] + [t for f, t in images if f != fmt0] + list(extra or [])
        f, chosen = QFileDialog.getSaveFileName(self, title, str(Path.home() / f"{stem}.{FS.FORMATS[fmt0][1][0]}"),
                                                ";;".join(order))
        if not f:
            return None, None
        if chosen in (extra or []) or Path(f).suffix.lower() in (".npy", ".csv"):
            return f, chosen
        fmt = FS.format_of(f, next((k for k, t in images if t == chosen), fmt0))
        if fmt != fmt0:
            self.set_save_options(dict(self.save_options, format=fmt))
        return str(FS.with_format(f, fmt)), fmt

    def _save_image(self):
        if self._grid_shown:
            self._save_grid_dialog()
            return
        if self._plane is None:
            return
        void_filter = "CSV table of the voids outlined (*.csv)"
        f, chosen = self._ask_image_path("Save the map", "explore", ["NumPy array of the values (*.npy)",
                                                                     "CSV table: x, y, value (*.csv)", void_filter])
        if not f:
            return
        if chosen == void_filter:
            try:
                self.status.emit(f"saved {self.save_voids_csv(f)}")
            except OSError as e:
                self.status.emit(f"could not save {f}: {e}")
            return
        if Path(f).suffix.lower() in (".npy", ".csv"):
            try:
                self.status.emit(f"saved {self.save_values(f)}")
            except OSError as e:
                self.status.emit(f"could not save {f}: {e}")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            written = self.save_png(f)
            self.status.emit("saved " + ", ".join(written) + f" ({FS.describe(self.save_options)})"
                             + (f"; {self._save_note}" if self._save_note else ""))
        except FS.TooLarge as e:                # not the screen instead: the options asked for too much
            self.status.emit(f"could not save {f}: {e}")
        except Exception as e:
            if FS.format_of(f) in ("png", "tiff"):
                self.view.canvas.grab().save(f)
                self.status.emit(f"saved {f} as the screen shows it (the figure failed: {e})")
            else:
                self.status.emit(f"could not save {f}: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def save_png(self, f, options: dict | None = None) -> list[str]:
        """The map as a file, as the Save options say ('options', else the ones chosen; until the dialog is used,
        the default figure: a matplotlib PNG at 200 dpi with title, axes, colour bar, arrows and the void
        outlines as on screen), in the format of the file's suffix (.png, .pdf, .svg, .tif). At full resolution,
        and with no margins and 'match the data' (figsave.writes_pixels): the map's own pixels
        (_write_full_map). With outlines drawn, '<name>_voids.csv' beside it: the table of those voids.
        The paths."""
        opts = FS.options(self.save_options if options is None else options)
        fmt = FS.format_of(f, opts["format"])
        written = [str(f)]
        self._save_note = ""
        if FS.writes_pixels(opts):
            self._write_full_map(f, fmt, opts)
        else:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            with matplotlib.rc_context(FS.rc(opts, fmt)):    # the font size and the text kept as text
                fig, dpi = self.map_figure(opts)
                try:
                    FS.savefig(fig, f, opts, dpi, fmt)     # the pixel guard, the titles' line breaks, savefig
                finally:
                    plt.close(fig)
        if opts["voids"] and self.voids_shown():
            written.append(self.save_voids_csv(Path(f).with_name(Path(f).stem + "_voids.csv")))
        return written

    def map_figure(self, options: dict | None = None):
        """The decorated map as a pyplot figure and the dpi to save it at (the options' dpi, or the one matching the
        data): title, axes (frame, ticks, tick labels, axis names in the unit chosen), colour bar and its label, the
        arrows and void outlines as on screen, each as the options say; margins 'none' = the map alone filling the
        figure. The caller closes it (plt.close) and applies figsave.rc for a font scale."""
        opts = FS.options(self.save_options if options is None else options)
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        req = self._req
        o = req["output"]
        data = self._plane.T
        if req["log"]:
            data = np.log10(np.where(data > 0, data, np.nan))
        _, _, rest, glo, ghi, _, _ = self._geometry()    # a hi-res slice's axes are its file's, not the menu's
        h = G.hubble(o) if opts["unit"] == "mpc_h" else None
        um, self._save_note = FS.unit_map(o, rest, opts["unit"], h)
        ext = um.ext([glo[0], ghi[0], glo[1], ghi[1]])
        s = float(opts["font_scale"])
        bare = opts["margins"] == "none"
        if bare:                                # the map alone: its axes are the whole figure
            aspect = (ext[3] - ext[2]) / (ext[1] - ext[0]) if ext[1] > ext[0] and ext[3] > ext[2] else 1.0
            fig = plt.figure(figsize=(7, 7 * aspect))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
        else:
            fig, ax = plt.subplots(figsize=(7, 6))
        lo, hi = self.view.colorbar.vrange
        kw = {"interpolation": "nearest"} if opts["match_data"] else {}
        if bare:
            kw["aspect"] = "auto"
        im = ax.imshow(data, origin="lower", extent=ext, cmap=req["cmap"], vmin=lo, vmax=hi, **kw)
        if not bare:
            FS.style_axes(ax, opts, (_ex.AXES[rest[0]], _ex.AXES[rest[1]]), um.name)
            if opts["title"]:
                # the slice position (and a zoom's window) in the axes' unit; in Mpc the screen's title
                auto = self._map_title()
                shown = auto if opts["unit"] == "mpc" else \
                    self._map_title(lambda a, v: FS.position(o, a, v, opts["unit"], h))
                t = ax.set_title(FS.title_text(opts, auto, shown), fontsize=8 * s)
                if opts["width"] > 0 or s != 1.0 or opts["match_data"]:
                    # a long title wraps rather than widening a figure of a set width or, in larger fonts (Slide),
                    # running over the colour bar's top tick label; the figure 'match the data'
                    # shrinks for a small map too
                    t.set_wrap(True)
            if opts["colorbar"]:
                if opts["colorbar_label"]:
                    fig.colorbar(im, ax=ax, label=self.view.colorbar.label)
                else:
                    fig.colorbar(im, ax=ax)
        self._map_overlays(ax, opts, um, rest, glo, ghi, ext)
        try:
            FS.fit_width(fig, opts, opts["dpi"])
        except FS.TooLarge:                     # a width far too large: no figure left open
            plt.close(fig)
            raise
        dpi = FS.match_dpi(fig, [(ax, data.shape[1])]) if opts["match_data"] else opts["dpi"]
        return fig, dpi

    def _map_overlays(self, ax, opts: dict, um, rest, glo, ghi, ext) -> None:
        """The arrows and the void outlines over a saved map as on screen, each when the options keep it, in the
        coordinates of the unit map 'um' (the axes' unit)."""
        o = self._req["output"]
        arrows = getattr(self, "_arrows", None)
        if opts["arrows"] and arrows is not None and arrows["vmax"] > 0:
            h = o.length / o.n                              # cell size per axis
            X, Y = np.meshgrid(um.x(o.lo[rest[0]] + arrows["c0"] * h[rest[0]]),
                               um.y(o.lo[rest[1]] + arrows["c1"] * h[rest[1]]), indexing="ij")
            ax.quiver(X, Y, arrows["u"], arrows["v"], color="white", edgecolor="black", linewidth=0.3,
                      angles="xy", scale_units="xy", pivot="middle",
                      scale=arrows["vmax"] / (0.9 * arrows["spacing"] * h[rest[0]] * um.k[0]), width=0.003)
        res = self._void_result()
        if opts["voids"] and res is not None and self.view.canvas.circles is not None:
            axis = self._geometry()[0]
            pos = self._slice_position()
            sel = np.nonzero(res["_shown"])[0]
            box = float(res["box_mpc"])
            conv = lambda cs: [(um.x(u), um.y(v), r * um.k[0], int(sel[k])) for u, v, r, k in cs]   # noqa: E731
            circ = conv(G.void_circles_mpc(res["_centres"][sel], res["_r_v"][sel], box, axis, pos, rest, glo, ghi))
            dash = [] if not self.void_reff.isChecked() else conv(G.void_circles_mpc(
                res["_centres"][sel], res["_r_eff"][sel], box, axis, pos, rest, glo, ghi))
            G.draw_void_circles_mpl(ax, circ, dash)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])

    def _write_full_map(self, f, fmt: str, opts: dict) -> None:
        """The map at its own resolution: the screen's pixels, one per cell, zoom point or hi-res point (the colours
        and range as shown; a hi-res slice over 4096 points a side as shown, block-averaged), the arrows and outlines
        over them when the options keep them."""
        req = self._req
        rgb = None if self._grid_shown else self.view.canvas._buf
        if rgb is None or rgb.shape[:2] != self._plane.T.shape:     # not the map's own pixels: colour them again
            img = np.ascontiguousarray(self._plane.T[::-1])
            rgb, _ = G.colorize(img, req["cmap"], log=req["log"], clip=(req["clip"], 100 - req["clip"]),
                                symmetric=req.get("symmetric", False), vrange=self.view.colorbar.vrange)
        _, _, rest, glo, ghi, _, _ = self._geometry()
        ext = [glo[0], ghi[0], glo[1], ghi[1]]
        arrows = self._arrows is not None and self._arrows["vmax"] > 0 and opts["arrows"]
        voids = opts["voids"] and self._void_result() is not None and self.view.canvas.circles is not None

        def overlay(ax):
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            self._map_overlays(ax, opts, FS.UnitMap(), rest, glo, ghi, ext)
        FS.write_pixels(f, rgb, fmt, overlay if (arrows or voids) else None)

    def save_voids_csv(self, path) -> str:
        """The voids outlined on the map now as a CSV table (tools/void_outlines.write_csv: centre, every R_v the
        catalogue has, R_eff, central delta, resolved and distinct)."""
        res = self._void_result()
        rows = self.voids_shown()
        if res is None or not rows:
            raise OSError("no void outlines on the map (tick 'voids')")
        return _ex._void_tool().write_csv(res, path, rows)

    def save_values(self, path) -> str:
        """The map's VALUES (not its colours): '.npy' = the 2D array as drawn, rows along the vertical axis from
        the bottom (values[iy, ix]); '.csv' = one row per cell, 'x,y,value' at the cell centres in Mpc (cells
        without a box). The field's own units -- not log10 -- and the difference for a difference map."""
        path = Path(path)
        g = self._geometry()
        if g is None or self._plane is None:
            raise OSError("no map on screen")
        _axis, _index, rest, lo, hi, n0, n1 = g
        values = np.asarray(self._plane, dtype=np.float64).T            # [iy, ix], origin at the bottom
        if path.suffix.lower() == ".npy":
            np.save(path, values)
            return str(path)
        xs = lo[0] + (np.arange(n0) + 0.5) / n0 * (hi[0] - lo[0])
        ys = lo[1] + (np.arange(n1) + 0.5) / n1 * (hi[1] - lo[1])
        X, Y = np.meshgrid(xs, ys)
        unit = "Mpc" if self._req["output"].box is not None else "cells"
        head = f"{_ex.AXES[rest[0]]}_{unit},{_ex.AXES[rest[1]]}_{unit},value"
        np.savetxt(path, np.column_stack([X.ravel(), Y.ravel(), values.ravel()]), delimiter=",", header=head,
                   comments="", fmt="%.8g")
        return str(path)
