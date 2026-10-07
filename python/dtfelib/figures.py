"""Figure styling and output for the plot scripts: colormaps, rcParams, colour norms,
and save_plot_to_multiple_paths (local figures dir + thesis mirror).

Import as `from dtfelib import figures as style` (the historical alias). Reads config
only at call time, so a test harness can patch config before running a script.
"""

from pathlib import Path

import matplotlib as mpl
import config

CMAP = {
    'density': 'plasma',
    'delta': 'coolwarm',
    'delta_raw': 'seismic',
    'divergence': 'RdBu_r',
    'potential': 'RdBu_r',
    'shear': 'plasma',
    'triaxiality': 'viridis',
    'velocity': 'viridis',
    'probability': 'hot',
    'sequential': 'plasma',
}

FIGSIZE = {
    'single': (8, 7),
    'wide': (12, 7),
    'large': (12, 9),
    'grid': (14, 11),
}


def apply():
    mpl.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.framealpha': 0.7,
        'savefig.dpi': config.DPI,
        'savefig.bbox': 'tight',
        'figure.constrained_layout.use': False,
        'axes.linewidth': 0.8,
        'grid.alpha': 0.3,
    })


def set_title(ax, *args, **kwargs):
    if not config.SHOW_TITLES:
        return
    kwargs.pop('fontsize', None)
    kwargs.pop('fontweight', None)
    ax.set_title(*args, **kwargs)


def set_suptitle(fig, *args, **kwargs):
    if not config.SHOW_TITLES:
        return
    kwargs.pop('fontsize', None)
    kwargs.pop('fontweight', None)
    fig.suptitle(*args, **kwargs)


import numpy as _np
from matplotlib import colors as _mcolors


def _finite(data):
    a = _np.asarray(data)
    return a[_np.isfinite(a)]


def norm_signed_log(data=None, field=None, vmax=None):
    if vmax is None:
        fixed = config.FIELD_LIMITS.get(field) if field else None
        if fixed is not None:
            vmax = float(fixed)
        else:
            a = _finite(data)
            vmax = float(_np.percentile(_np.abs(a), config.PERCENTILE_CLIP[1])) if a.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    override = config.SYMLOG_LINTHRESH_OVERRIDES.get(field)
    if override is not None:
        linthresh = min(override, vmax / 2)
    else:
        linthresh = config.SYMLOG_LINTHRESH_FRAC * vmax
    return _mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)


def norm_positive_log(data=None, field=None, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        fixed = config.FIELD_LIMITS.get(field) if field else None
        if fixed is not None:
            vmin, vmax = (float(fixed[0]), float(fixed[1]))
        else:
            a = _finite(data)
            a = a[a > 0]
            if a.size:
                lo, hi = config.PERCENTILE_CLIP
                vmin = float(_np.percentile(a, lo))
                vmax = float(_np.percentile(a, hi))
            else:
                vmin, vmax = 1e-6, 1.0
    if vmin <= 0:
        vmin = 1e-12
    if vmax <= vmin:
        vmax = vmin * 10
    return _mcolors.LogNorm(vmin=vmin, vmax=vmax)


def norm_bounded(lo=0.0, hi=1.0, field=None):
    fixed = config.FIELD_LIMITS.get(field) if field else None
    if fixed is not None:
        lo, hi = fixed
    return _mcolors.Normalize(vmin=lo, vmax=hi)


class SignedTwoSlopeLogNorm(_mcolors.Normalize):

    def __init__(self, vmin, vmax, linthresh):
        super().__init__(vmin=float(vmin), vmax=float(vmax))
        self.linthresh = max(float(linthresh), 1e-12)

    def _t(self, x):
        x = _np.asarray(x, dtype=float)
        return _np.sign(x) * _np.log10(1.0 + _np.abs(x) / self.linthresh)

    def _tinv(self, y):
        y = _np.asarray(y, dtype=float)
        return _np.sign(y) * self.linthresh * (10.0 ** _np.abs(y) - 1.0)

    def __call__(self, value, clip=None):
        t = self._t(_np.clip(value, self.vmin, self.vmax))
        tpos = max(float(self._t(self.vmax)), 1e-12) if self.vmax > 0 else 1.0
        tneg = max(float(-self._t(self.vmin)), 1e-12) if self.vmin < 0 else 1.0
        out = _np.where(t >= 0, 0.5 + 0.5 * t / tpos, 0.5 - 0.5 * (-t) / tneg)
        return _np.ma.masked_invalid(out)

    def inverse(self, value):
        value = _np.asarray(value, dtype=float)
        tpos = max(float(self._t(self.vmax)), 1e-12) if self.vmax > 0 else 1.0
        tneg = max(float(-self._t(self.vmin)), 1e-12) if self.vmin < 0 else 1.0
        t = _np.where(value >= 0.5, (value - 0.5) * 2.0 * tpos,
                      -(0.5 - value) * 2.0 * tneg)
        return self._tinv(t)


def norm_signed_centered(data=None, field=None, vmin=None, vmax=None):
    override = config.SYMLOG_LINTHRESH_OVERRIDES.get(field)
    if vmin is None or vmax is None:
        fixed = config.FIELD_LIMITS.get(field) if field else None
        if isinstance(fixed, (list, tuple)):
            vmin, vmax = float(fixed[0]), float(fixed[1])
        else:
            a = _finite(data)
            lo, hi = config.PERCENTILE_CLIP
            vmin = float(_np.percentile(a, lo)) if a.size else -1.0
            vmax = float(_np.percentile(a, hi)) if a.size else 1.0
    span = max(abs(vmin), abs(vmax), 1e-12)
    vmin = min(vmin, -1e-3 * span)
    vmax = max(vmax, 1e-3 * span)
    linthresh = override if override is not None else config.SYMLOG_LINTHRESH_FRAC * span
    return SignedTwoSlopeLogNorm(vmin=vmin, vmax=vmax, linthresh=linthresh)


def norm_signed_linear(data=None, field=None, vmax=None):
    if vmax is None:
        fixed = config.FIELD_LIMITS.get(field) if field else None
        if fixed is not None and not isinstance(fixed, (list, tuple)):
            vmax = float(fixed)
        else:
            vmax = robust_vmax(data) or 1.0
    return _mcolors.Normalize(vmin=-vmax, vmax=vmax)


def norm_ticks(norm):
    if not isinstance(norm, SignedTwoSlopeLogNorm):
        return None

    def decades(limit):
        out = []
        if limit <= 0:
            return out
        k0 = int(_np.ceil(_np.log10(norm.linthresh)))
        for k in range(k0, int(_np.floor(_np.log10(limit))) + 1):
            v = 10.0 ** k
            if norm.linthresh <= v <= limit:
                out.append(v)
        return out

    ticks = ([norm.vmin] + [-v for v in decades(-norm.vmin)] + [0.0]
             + decades(norm.vmax) + [norm.vmax])
    return sorted(set(ticks))


def field_norm(field, data=None):
    kind = config.FIELD_KINDS.get(field)
    if kind == 'signed':
        return norm_signed_log(data=data, field=field)
    if kind == 'signed_centered':
        return norm_signed_centered(data=data, field=field)
    if kind == 'signed_linear':
        return norm_signed_linear(data=data, field=field)
    if kind == 'positive':
        return norm_positive_log(data=data, field=field)
    if kind == 'bounded':
        return norm_bounded(field=field)
    raise KeyError(f"Unknown field type {field!r}; add it to config.FIELD_KINDS")


def robust_vmax(data):
    a = _finite(data)
    return float(_np.percentile(_np.abs(a), config.PERCENTILE_CLIP[1])) if a.size else 0.0


def symmetric_limits(data, pct=None):
    """(-m, m) for a diverging map centred on 0 (a divergence, an eigenvalue): m = the 'pct' (default
    config.PERCENTILE_CLIP[1]) percentile of the finite |values|; (-1, 1) when the slice is empty or all
    zero, so imshow never gets vmin == vmax. A percentile pair (1, 99) of the signed values put 0 off
    centre whenever the slice was skewed (plot_DTFE's divergence map, found 2026-10-05)."""
    a = _finite(data)
    m = float(_np.percentile(_np.abs(a), config.PERCENTILE_CLIP[1] if pct is None else pct)) if a.size else 0.0
    if not m > 0 and a.size:            # a MOSTLY-zero slice: its percentile is 0, its few values are not
        m = float(_np.abs(a).max())     # (sparse or zero-padded data; (-1, 1) would hide every pixel)
    if not m > 0:
        m = 1.0
    return -m, m


def log_limits(data, floor=1e-6):
    """(vmin, vmax) for a log colour scale: the smallest and largest finite POSITIVE values, vmin at least
    'floor'; None when there is none (an empty slice -- the caller skips the map instead of crashing on
    the min of an empty array, as the shear map did). vmax > vmin always."""
    a = _finite(data)
    a = a[a > 0]
    if not a.size:
        return None
    vmin, vmax = max(float(a.min()), floor), float(a.max())
    if not vmax > vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def delta_contour_levels(field_slice, norm, num_contours=20):
    vmin = float(_np.min(field_slice))
    vmax = float(_np.max(field_slice))
    if not isinstance(norm, SignedTwoSlopeLogNorm):
        return _np.linspace(vmin, vmax, num_contours)

    num_void = int(num_contours * 0.75)
    num_over = num_contours - num_void
    t, tinv = norm._t, norm._tinv
    levels = [0.0]
    if vmin < -0.01:
        levels.extend(tinv(_np.linspace(t(vmin), t(-0.01), num_void)))
    if vmax > 0.01:
        levels.extend(tinv(_np.linspace(t(0.01), t(vmax), num_over)))
    return _np.unique(_np.asarray(levels))


def mirror_relative_path(primary_path) -> str:
    """Where a figure goes below the thesis Figures/ tree: its path below config.LOCAL_FIGURES_ROOT (read at
    call time) when it lies under it, else below its first 'figures/' folder, else its bare name. The root
    comes first since 2026-10-07: DTFE_FIGURES_ROOT may name any folder, and under one without '/figures/' in
    its path (the T7 default was briefly 'DTFE figures') the folder test alone dropped every mirrored file flat
    into Figures/, simulations' names colliding."""
    p = Path(primary_path).expanduser()
    try:
        return p.resolve().relative_to(Path(str(config.LOCAL_FIGURES_ROOT)).expanduser().resolve()).as_posix()
    except (ValueError, OSError):
        pass
    s = str(p).replace('\\', '/')
    return s.split('/figures/', 1)[1] if '/figures/' in s else p.name


def save_plot_to_multiple_paths(fig, primary_path, dpi=300, mirror=True, **kwargs):
    """Save a figure to its primary path and mirror it into the thesis Figures/ tree
    (config.THESIS_FIGURES_DIR, read at call time), preserving its path below the figure root
    (mirror_relative_path). mirror=False saves the primary path only: a script run with --out (a path outside
    the figure root) would otherwise drop its files FLAT into the thesis tree's root, where simulations' names
    collide (review 2026-10-05)."""
    primary_path = Path(primary_path)
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(primary_path, dpi=dpi, **kwargs)
    if not mirror:
        return
    from . import pipeline as _pl                         # here: pipeline imports this module's package at load
    if _pl.smoothing_tag():
        return                                            # the thesis tree holds production-smoothing figures only

    additional_path = Path(str(config.THESIS_FIGURES_DIR)) / mirror_relative_path(primary_path)
    additional_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(additional_path, dpi=dpi, **kwargs)


# ---------------------------------------------------------------- slice maps in the house style (survey item 11, 2026-10-05)
# plot_DTFE.py, plot_PS_DTFE.py and plot_cosmic_web.py drew their maps by hand: matplotlib's default sans
# font, titles that ignored SHOW_TITLES, colour maps and ranges as literals, three private copies of the
# axis-label and DPI constants. They are thin calls of slice_map / slice_row now. The scripts keep their
# transposes (the data is passed rows-as-vertical, as imshow wants it), their file names and their choices
# of which maps to write; the look changed in four stated ways: serif text and no titles (SHOW_TITLES),
# lowercase axis letters from config.SLICE_PLANES, ONE rule per quantity (the web colours of
# plot_cosmic_web, the eigenvalue triptychs on norm_signed_log at the field's linthresh), and the
# density/streams colour floors (set_bad / set_under at the map's lowest colour) applied by both scripts.
from .io import STREAM_TOL as _STREAM_TOL

MAP_LABELS = {
    'density': r'$\rho / \bar\rho$',
    'divergence': r'$\nabla \cdot v$ [km/s/Mpc]',
    'shear': r'$|\sigma|$ [km/s/Mpc]',
    'streams': 'number of streams',
    'velDisp': r'Tr $\sigma^2$  [(km/s)$^2$]',
    'velDispTensor': r'$|\sigma^2|$  [(km/s)$^2$]',
    'velMag': r'$|v|$  [km/s]',
    'velocity': r'$|v|$  [km/s]',
    'eigenvalue': r'$\lambda$',
}
MAP_CMAPS = {'density': 'plasma', 'divergence': 'RdBu_r', 'shear': 'plasma', 'streams': 'inferno',
             'streams_single': 'viridis', 'velDisp': 'magma', 'velDispTensor': 'magma', 'velMag': 'magma',
             'velocity': 'viridis', 'eigenvalue': 'RdBu_r'}
WEB_NAMES = ('Void', 'Wall', 'Filament', 'Node')
WEB_COLORS = ['#1a1a2e', '#e0c97f', '#d4563e', '#f5f5dc']       # void, wall, filament, node (plot_cosmic_web's)
RESIDUAL_COLORS = ['#08306b', '#2171b5', '#6baed6', '#f0f0f0', '#fb6a4a', '#cb181d', '#67000d']   # T-web - V-web, -3..3


def web_cmap_norm():
    """The categorical colour map and norm of a web classification (0 void .. 3 node)."""
    return _mcolors.ListedColormap(WEB_COLORS), _mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)


def residual_cmap_norm():
    """The colour map and norm of a class difference (-3 .. 3)."""
    return _mcolors.ListedColormap(RESIDUAL_COLORS), _mcolors.BoundaryNorm(_np.arange(-3.5, 4.5, 1.0), 7)


def cmap_floored(name):
    """A copy of the named colour map whose 'bad' (NaN) and 'under' colours are its lowest colour: an empty
    or sub-range cell on a log map shows as the darkest, not white."""
    import matplotlib.pyplot as _plt
    cmap = _plt.get_cmap(name).copy()
    cmap.set_bad(cmap(0))
    cmap.set_under(cmap(0))
    return cmap


def norm_log_range(data, fixed=None, floor=1e-6):
    """A LogNorm for a positive field: 'fixed' = (vmin, vmax) with a None end taken from the slice's own
    log_limits; None only when an end must come from a slice with no positive value (a caller with a fixed
    range checks for an empty slice itself: plot_DTFE / plot_PS_DTFE.plot_density). The explicit range
    wins over config.FIELD_LIMITS (norm_positive_log's lookup): the map scripts pass DENSITY_MAP_RANGE."""
    own = log_limits(data, floor)
    lo, hi = (None, None) if fixed is None else fixed
    if (lo is None or hi is None) and own is None:
        return None
    vmin = float(lo) if lo is not None else own[0]
    vmax = float(hi) if hi is not None else own[1]
    if not vmax > vmin:
        vmax = vmin * 10.0
    return _mcolors.LogNorm(vmin=vmin, vmax=vmax)


def norm_symmetric(data, pct=None):
    """A linear norm centred on 0 (symmetric_limits): a divergence, a residual."""
    lo, hi = symmetric_limits(data, pct)
    return _mcolors.Normalize(vmin=lo, vmax=hi)


def norm_streams(data):
    """The stream-count map's rule: (cmap name, norm) -- a slice with no multi-stream cell (max <= 1 + tol)
    on a linear 0..1 viridis, else inferno on a log scale from 0.5 to max(max, 2); None for a slice
    without streams (max < tol). '.streams' is a float (STREAM_TOL), never compared as an integer."""
    a = _finite(data)
    smax = float(a.max()) if a.size else 0.0
    if smax < _STREAM_TOL:
        return None
    if smax <= 1.0 + _STREAM_TOL:
        return MAP_CMAPS['streams_single'], _mcolors.Normalize(vmin=0.0, vmax=1.0)
    return MAP_CMAPS['streams'], _mcolors.LogNorm(vmin=0.5, vmax=max(smax, 2.0))


def slice_axes(ax, plane, box_mpc, ylabel=True):
    """Axis labels from config.SLICE_PLANES[plane] in config.AXIS_UNITS, equal aspect, the box as limits."""
    labels = config.SLICE_PLANES[plane]['axis_labels']
    ax.set_xlabel(f"{labels[0]} [{config.AXIS_UNITS}]")
    if ylabel:
        ax.set_ylabel(f"{labels[1]} [{config.AXIS_UNITS}]")
    ax.set_xlim(0, box_mpc)
    ax.set_ylim(0, box_mpc)
    ax.set_aspect('equal')


def _draw_panel(fig, ax, panel, box_mpc, plane, ylabel=True):
    """One map into 'ax' from a panel dict: data (rows vertical), cmap (a name or a Colormap), norm or
    vmin/vmax, label (the colour bar's), title, interpolation, cbar_ticks / cbar_ticklabels, floor, text."""
    cmap = panel.get('cmap', 'viridis')
    if isinstance(cmap, str):
        cmap = cmap_floored(cmap) if panel.get('floor') else cmap
    kw = {'norm': panel['norm']} if panel.get('norm') is not None else {'vmin': panel.get('vmin'), 'vmax': panel.get('vmax')}
    im = ax.imshow(panel['data'], origin='lower', cmap=cmap, extent=[0, box_mpc, 0, box_mpc],
                   interpolation=panel.get('interpolation'), **kw)
    slice_axes(ax, plane, box_mpc, ylabel=ylabel)
    set_title(ax, panel.get('title', ''))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=panel.get('cbar_ticks'))
    if panel.get('cbar_ticklabels') is not None:
        cb.ax.set_yticklabels(panel['cbar_ticklabels'])
    if panel.get('label'):
        cb.set_label(panel['label'])
    if panel.get('text'):
        ax.text(0.02, 0.98, panel['text'], transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    return im


def _save(fig, path, mirror):
    import matplotlib.pyplot as _plt
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if mirror:
            save_plot_to_multiple_paths(fig, path, dpi=config.DPI, bbox_inches='tight')
        else:
            fig.savefig(path, dpi=config.DPI, bbox_inches='tight')
    _plt.close(fig)


def slice_map(data, box_mpc, plane, cmap, norm=None, label='', title='', path=None, vmin=None, vmax=None,
              interpolation=None, text=None, footnote=None, cbar_ticks=None, cbar_ticklabels=None, floor=False,
              figsize=(8, 7), mirror=False):
    """One slice map in the house style, saved to 'path' (config.DPI, tight; mirror=True also into the
    thesis Figures tree) and closed. 'data' is passed as imshow wants it (rows vertical: the scripts keep
    their .T); 'text' = a boxed note in the corner, 'footnote' = a line under the axes. Returns
    {'path', 'vmin', 'vmax'} -- the range drawn, for tests and logs."""
    import matplotlib.pyplot as _plt
    fig, ax = _plt.subplots(figsize=figsize)
    im = _draw_panel(fig, ax, dict(data=data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, label=label, title=title,
                                   interpolation=interpolation, text=text, cbar_ticks=cbar_ticks,
                                   cbar_ticklabels=cbar_ticklabels, floor=floor), box_mpc, plane)
    if footnote:
        ax.text(0.5, -0.12, footnote, transform=ax.transAxes, ha='center', fontsize=9, style='italic')
    fig.tight_layout()
    lo, hi = im.norm.vmin, im.norm.vmax
    _save(fig, path, mirror)
    return {'path': None if path is None else Path(path), 'vmin': lo, 'vmax': hi}


def slice_row(panels, box_mpc, plane, path=None, suptitle='', footnote=None, figsize=None, mirror=False):
    """A row of maps of one plane (a triptych of eigenvalues, the density next to the streams, the two webs
    and their residual): 'panels' = the dicts _draw_panel takes; only the first carries the y label.
    'footnote' goes under the row. Returns {'path', 'ranges': [(vmin, vmax), ...]}."""
    import matplotlib.pyplot as _plt
    n = len(panels)
    fig, axes = _plt.subplots(1, n, figsize=figsize or (6.6 * n + 1.0, 6.0), squeeze=False)
    ranges = []
    for i, (ax, panel) in enumerate(zip(axes[0], panels)):
        im = _draw_panel(fig, ax, panel, box_mpc, plane, ylabel=(i == 0))
        ranges.append((im.norm.vmin, im.norm.vmax))
    if footnote:
        fig.text(0.5, -0.02, footnote, ha='center', style='italic')
    set_suptitle(fig, suptitle, y=1.02)
    fig.tight_layout()
    _save(fig, path, mirror)
    return {'path': None if path is None else Path(path), 'ranges': ranges}
