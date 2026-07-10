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


def save_plot_to_multiple_paths(fig, primary_path, dpi=300, **kwargs):
    """Save a figure to its primary path and mirror it into the thesis Figures/ tree
    (config.THESIS_FIGURES_DIR, read at call time), preserving the path below 'figures/'."""
    primary_path = Path(primary_path)
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(primary_path, dpi=dpi, **kwargs)

    primary_str = str(primary_path)
    if '/figures/' in primary_str or '\\figures\\' in primary_str:
        parts = primary_str.replace('\\', '/').split('/figures/')
        relative_path = parts[1] if len(parts) > 1 else primary_path.name
    else:
        relative_path = primary_path.name

    additional_path = Path(str(config.THESIS_FIGURES_DIR)) / relative_path
    additional_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(additional_path, dpi=dpi, **kwargs)
