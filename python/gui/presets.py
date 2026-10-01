"""Presets and help text for the GUI's run settings. Pure Python (no Qt).

  PRESETS / preset_names / apply_preset   named starting points for a Grids / custom-snapshot run
                                          (built in, plus the user's own, kept in the GUI's settings)
  OPTION_TEXT                             plain-language label + the binary flag behind each option
  flag_help(binary)                       {flag: description} parsed from the binary's own
                                          '--full_help', so a tooltip can never drift from the code
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_SEVEN = ["density_a", "velocity_a", "gradient_a", "divergence_a", "shear_a", "vorticity_a",
          "dispersion_a"]

# name -> (one-line description, settings). The settings are RunSpec / CustomSpec fields; a
# preset leaves every field it does not name as it was. 'slice_plane': "largest" = the largest
# image plane found for the simulation (none: no slice).
BUILTIN_PRESETS: dict[str, tuple[str, dict]] = {
    "Quick look": ("128³ grid, density and velocity only: about a minute on a laptop-sized "
                   "snapshot, for checking that everything works",
                   {"grid": 128, "nsub": 1, "deposit": "sampled", "fields": ["density_a", "velocity_a"],
                    "vertex_mass": True, "volume_weighted": True, "caustics": False,
                    "caustic_cusps": False, "slice_plane": ""}),
    "Standard": ("512³ grid, the seven production fields with the literature-standard weighting "
                 "(the setting of the thesis grids)",
                 {"grid": 512, "nsub": 3, "deposit": "sampled", "fields": list(_SEVEN),
                  "vertex_mass": True, "volume_weighted": True, "caustics": False,
                  "caustic_cusps": False, "slice_plane": ""}),
    "Publication": ("Standard + caustic stratification + a hi-res point-evaluated slice (the "
                    "largest image plane of the simulation) for figures",
                    {"grid": 512, "nsub": 3, "deposit": "sampled", "fields": list(_SEVEN),
                     "vertex_mass": True, "volume_weighted": True, "caustics": True,
                     "caustic_cusps": False, "slice_plane": "largest", "slice_vel_grad": True}),
    "Exact deposit": ("256³ grid with the exact tetrahedron-cell clipping instead of sampling: "
                      "noise-free cell values, for accuracy checks (GPU strongly recommended)",
                      {"grid": 256, "deposit": "exact", "fields": ["density_a", "velocity_a"],
                       "vertex_mass": True, "volume_weighted": True, "gpu": True, "slice_plane": ""}),
}


def preset_names(user: dict) -> list[str]:
    return list(BUILTIN_PRESETS) + [n for n in user if n not in BUILTIN_PRESETS]


def preset(name: str, user: dict) -> tuple[str, dict] | None:
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name]
    if name in user:
        return "your own preset", dict(user[name])
    return None


PRESET_KEYS = ("grid", "nsub", "deposit", "fields", "gpu", "vertex_mass", "volume_weighted",
               "caustics", "caustic_cusps", "slice_plane", "slice_vel_grad", "estimator")


def capture(spec) -> dict:
    """The preset-able settings of a RunSpec / CustomSpec (for 'Save as preset')."""
    d = spec.to_dict()
    return {k: d[k] for k in PRESET_KEYS if k in d}


def apply_preset(spec, settings: dict, planes=()) -> list[str]:
    """Sets the named fields on spec (in place); returns what was changed, for the status line.
    planes: the simulation's image-plane files, for 'slice_plane': 'largest'."""
    changed = []
    known = spec.to_dict()
    for k, v in settings.items():
        if k not in known:
            continue
        if k == "slice_plane" and v == "largest":
            v = str(max(planes, key=lambda p: Path(p).stat().st_size)) if planes else ""
        if known[k] != v:
            setattr(spec, k, list(v) if isinstance(v, list) else v)
            changed.append(k)
    return changed


# ---------------------------------------------------------------------- labels
# option -> (plain-language label, flag). The flag's full description comes from the binary.
OPTION_TEXT = {
    "gpu":             ("Use the GPU for the grid deposit", "--ps-gpu"),
    "vertex_mass":     ("Tetrahedron masses from the particles (needed for TNG initial conditions)",
                        "--ps-vertex-mass"),
    "volume_weighted": ("Volume-weighted velocities (the literature convention)", "--ps-volume-weighted"),
    "caustics":        ("Mark caustics: folds and how many axes collapsed", "--ps-caustics"),
    "caustic_cusps":   ("Also mark cusps and swallowtails (slower, CPU)", "--ps-caustic-cusps"),
    "slice_vel_grad":  ("Velocity gradient at each slice point (divergence, shear, vorticity maps)",
                        "--pts-vel-grad"),
    "exact":           ("Exact: analytic tetrahedron-cell clipping (noise-free, slow)", "--ps-exact-deposit"),
    "sampled":         ("Sampled: nSub³ points per cell (fast, the default)", "--avg-subsamples"),
    "periodic":        ("The box is periodic (a cosmological simulation)", "--periodic"),
    "scalar":          ("Also interpolate a per-particle value", "--scalar-dataset"),
}


def label(key: str) -> str:
    return OPTION_TEXT[key][0]


# ---------------------------------------------------------------------- help from the binary
_OPT = re.compile(r"^  (?:-\w \[ )?--([A-Za-z0-9_\-]+)(?: \])?(?: arg(?: \(=[^)]*\))?)?\s{2,}(\S.*)$")
_OPT_NODESC = re.compile(r"^  (?:-\w \[ )?--([A-Za-z0-9_\-]+)(?: \])?(?: arg(?: \(=[^)]*\))?)?\s*$")
_HELP_CACHE: dict[str, dict[str, str]] = {}


def parse_help(text: str) -> dict[str, str]:
    """{flag (with '--'): description} from boost::program_options help output: an option line
    '  -g [ --grid ] arg      text' followed by continuation lines indented to the text column."""
    out: dict[str, str] = {}
    cur, col = None, 0
    for line in text.splitlines():
        m = _OPT.match(line)
        if m:
            cur, col = "--" + m.group(1), line.index(m.group(2))
            out[cur] = m.group(2).strip()
            continue
        m = _OPT_NODESC.match(line)
        if m:                           # a long option name pushes its text onto the next line
            cur, col = "--" + m.group(1), -1
            out[cur] = ""
            continue
        if cur and line.strip() and (col < 0 or line[:col].strip() == ""):
            if col < 0:
                col = len(line) - len(line.lstrip())
            out[cur] = (out[cur] + " " + line.strip()).strip()
        elif not line.strip() or not line.startswith(" "):
            cur = None
    return out


def flag_help(binary: str = "PS-DTFE") -> dict[str, str]:
    """The binary's '--full_help', parsed (cached per binary and modification time); {} when the
    binary is not built."""
    exe = REPO_ROOT / binary
    try:
        key = f"{exe}:{exe.stat().st_mtime_ns}"
    except OSError:
        return {}
    if key not in _HELP_CACHE:
        try:
            text = subprocess.run([str(exe), "--full_help"], capture_output=True, text=True,
                                  timeout=20, cwd=str(REPO_ROOT)).stdout
        except (OSError, subprocess.TimeoutExpired):
            text = ""
        _HELP_CACHE[key] = parse_help(text)
    return _HELP_CACHE[key]


def plain_help(text: str) -> str:
    """The binary's help text as prose for the window: option names without their leading dashes
    ('--partition' reads 'partition') and ' -- ' as a real dash."""
    text = re.sub(r"(?<![\w-])--(?=[A-Za-z])", "", text)
    text = re.sub(r"(?<=\w)--(?=\w)", "–", text)          # 'velocity-divergence--density': an en dash
    return text.replace(" -- ", " — ").replace("--", "—")


def tooltip(key: str, binary: str = "PS-DTFE", width: int = 90) -> str:
    """The binary's own description of the option behind 'key', wrapped for a tooltip."""
    if key not in OPTION_TEXT:
        return ""
    text = plain_help(flag_help(binary).get(OPTION_TEXT[key][1], ""))
    import textwrap
    return "\n".join(textwrap.wrap(text, width))
