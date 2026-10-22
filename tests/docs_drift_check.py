#!/usr/bin/env python3
"""Docs-drift check: what README.md and demo/COMMANDS.md tell a user to type must exist in the code,
and the knobs the drivers read must be documented somewhere a user looks.

  A) every '--flag' in README.md / demo/COMMANDS.md is an option of the binaries (src/user_options.cc's
     program_options names), of a script (scripts/*.sh) or of a python tool or test (argparse);
  B) every DTFE_* name in them is read somewhere in the code;
  C) the README knob table: each knob it names is a knob of scripts/config.sh / run_ps_dtfe.sh / run_dtfe.sh (or
     an environment variable the code reads), and where the table gives a plain value as the default it equals
     the scripts' 'VAR="${VAR:-default}"';
  D) the other way round: every knob of the drivers and the pipeline, and every DTFE_* variable that
     scripts/config.sh or python/config.py reads, is named in README.md, demo/COMMANDS.md or a script's --help
     header (the comment block a script's --help prints).
The synthetic half (S) feeds the same functions a made-up README with known drift, so a checker that stopped
finding anything fails too. Standard library only (runs under any python3 >= 3.10, no numpy).

    python3 tests/docs_drift_check.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ("README.md", "demo/COMMANDS.md")
DRIVERS = ("scripts/config.sh", "scripts/run_ps_dtfe.sh", "scripts/run_dtfe.sh")   # the README table's defaults
KNOB_SCRIPTS = DRIVERS + ("scripts/run_ps_pipeline.sh",)
SKIP_PARTS = {"tmp", "cache", "__pycache__", "figures", "reference"}

# a flag: '--' right after a non-word character and right before a letter. The README uses ' -- ' as a dash
# everywhere, so the letter after the dashes is what tells a flag from punctuation.
FLAG_RE = re.compile(r"(?<![\w-])--[A-Za-z][A-Za-z0-9_-]*[A-Za-z0-9]")
DTFE_RE = re.compile(r"\bDTFE_[A-Z0-9_]*[A-Z0-9]\b")

passed = failed = 0


def check(ok, msg):
    global passed, failed
    if ok:
        passed += 1
        print(f"   PASS  {msg}")
    else:
        failed += 1
        print(f"   FAIL  {msg}")


def read(rel):
    p = ROOT / rel
    return p.read_text(errors="replace") if p.exists() else ""


def code_files():
    pats = ["src/**/*.cc", "src/**/*.cpp", "src/**/*.h", "src/**/*.hpp", "src/**/*.cu", "src/**/*.metal",
            "scripts/*.sh", "python/**/*.py", "tests/**/*.py", "tests/**/*.sh", "Makefile", "metal/**/*.cpp"]
    seen = set()
    for pat in pats:
        for f in sorted(ROOT.glob(pat)):
            if f in seen or SKIP_PARTS & set(f.relative_to(ROOT).parts) or f.name == Path(__file__).name:
                continue
            seen.add(f)
            yield f


def code_flags(texts):
    """Every option name the code defines: literal '--x' strings (scripts, argparse) plus the program_options
    names of the binaries, which are declared without dashes: ("grid", ...) or ("config,c", ...)."""
    flags = set()
    for name, text in texts:
        flags.update(FLAG_RE.findall(text))
        if name.endswith(("user_options.cc", "user_options.h")):
            flags.update("--" + m for m in re.findall(r'\(\s*"([A-Za-z][\w-]*)(?:,\w)?"\s*,', text))
    return flags


def script_defaults(text):
    """{VAR: default} of every 'VAR="${VAR:-default}"' line; a nested '${A:-${B:-v}}' resolves to v."""
    out = {}
    for m in re.finditer(r'^\s*([A-Z][A-Z0-9_]*)="\$\{\1:?-', text, re.M):
        i, depth = m.end(), 1
        while i < len(text) and depth:            # the matching brace of '${VAR:-'
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        val = text[m.end():i - 1]
        while True:
            n = re.fullmatch(r"\$\{[A-Z][A-Z0-9_]*:?-(.*)\}", val, re.S)
            if not n:
                break
            val = n.group(1)
        out.setdefault(m.group(1), val)
    return out


def help_header(text):
    """The comment block a script's --help prints: line 2 up to the first blank line."""
    lines = text.splitlines()[1:]
    out = []
    for ln in lines:
        if not ln.strip():
            break
        out.append(ln)
    return "\n".join(out)


def knob_table(readme):
    """[(knob names, default cell)] of the README table whose header starts '| knob (environment)'."""
    rows, on = [], False
    for ln in readme.splitlines():
        if ln.startswith("| knob (environment)"):
            on = True
            continue
        if on:
            if not ln.startswith("|"):
                break
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) < 3 or set(cells[0]) <= set("-: "):
                continue
            names = [n for n in re.findall(r"`([A-Z][A-Z0-9_]*)`", cells[0])]
            names += [n for n in re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", re.sub(r"`[^`]*`", "", cells[0]))]
            rows.append((names, cells[-1]))
    return rows


def default_drift(rows, defaults):
    """[(knob, documented, script default)] where the table's plain value differs from the scripts'."""
    bad = []
    for names, cell in rows:
        items = [s.strip() for s in cell.split(",")] if len(names) > 1 else [cell]
        if len(items) != len(names):
            items = [cell] * len(names)
        for name, item in zip(names, items):
            if name not in defaults:
                continue
            want = defaults[name]
            if not want or re.search(r"\s|\$", want):       # empty or a list/expression: nothing to compare
                continue
            first = item.split()[0] if item.split() else ""
            plain = first.strip("`")
            if not (re.fullmatch(r"-?[0-9.]+", plain) or (first.startswith("`") and first.endswith("`"))):
                continue                                    # prose ('auto-tuned', 'off'): not a value
            if plain != want:
                bad.append((name, plain, want))
    return bad


def main():
    texts = [(str(f.relative_to(ROOT)), f.read_text(errors="replace")) for f in code_files()]
    code = "\n".join(t for _, t in texts)
    flags = code_flags(texts)
    docs = {d: read(d) for d in DOCS}
    headers = "\n".join(help_header(read(str(f.relative_to(ROOT)))) for f in sorted((ROOT / "scripts").glob("*.sh")))
    documented = "\n".join(docs.values()) + "\n" + headers

    print("(A) every --flag in the docs is an option the code defines")
    for d, text in docs.items():
        miss = sorted(f for f in set(FLAG_RE.findall(text)) if f not in flags)
        check(not miss and text, f"{d}: {len(set(FLAG_RE.findall(text)))} flags, none unknown"
              + (f" -- unknown: {' '.join(miss)}" if miss else ""))

    print("(B) every DTFE_* name in the docs is read by the code")
    for d, text in docs.items():
        names = set(DTFE_RE.findall(text))
        miss = sorted(n for n in names if not re.search(r"\b" + n + r"\b", code))
        check(not miss, f"{d}: {len(names)} DTFE_* names, all in the code" + (f" -- missing: {' '.join(miss)}" if miss else ""))

    print("(C) the README knob table against the drivers")
    defaults = {}
    for s in DRIVERS:
        for k, v in script_defaults(read(s)).items():
            defaults.setdefault(k, v)
    rows = knob_table(docs["README.md"])
    check(len(rows) >= 8, f"the knob table is found ({len(rows)} rows)")
    knobs_in_code = set(defaults)
    for s in KNOB_SCRIPTS:
        knobs_in_code |= set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", read(s)))
    unknown = sorted({n for names, _ in rows for n in names
                      if n not in knobs_in_code and not re.search(r"\b" + n + r"\b", code)})
    check(not unknown, "every knob the table names is read by the code" + (f" -- unknown: {' '.join(unknown)}" if unknown else ""))
    drift = default_drift(rows, defaults)
    check(not drift, "every plain default in the table equals the scripts' default"
          + "".join(f" -- {k}: README {d}, scripts {w}" for k, d, w in drift))

    print("(D) every driver knob and configuration variable is documented")
    knobs = set()
    for s in KNOB_SCRIPTS:
        knobs |= set(script_defaults(read(s)))
    miss = sorted(k for k in knobs if not re.search(r"\b" + k + r"\b", documented))
    check(not miss, f"{len(knobs)} driver knobs, each in the README, COMMANDS.md or a script's --help"
          + (f" -- undocumented: {' '.join(miss)}" if miss else ""))
    env = set(re.findall(r"\$\{(DTFE_[A-Z0-9_]+)", read("scripts/config.sh")))
    env |= set(re.findall(r"""(?:environ(?:\.get)?\(|environ\[|getenv\()\s*["'](DTFE_[A-Z0-9_]+)""", read("python/config.py")))
    miss = sorted(e for e in env if not re.search(r"\b" + e + r"\b", documented))
    check(not miss, f"{len(env)} DTFE_* variables of scripts/config.sh and python/config.py, each documented"
          + (f" -- undocumented: {' '.join(miss)}" if miss else ""))

    print("(S) the checker itself finds planted drift (synthetic)")
    fake = ("x\n| knob (environment) | meaning | default |\n|---|---|---|\n"
            "| `GRID_SIZE` (or `-g`) | cells | 512 (`config.sh`) |\n"
            "| `FIELDS`, `OUTPUT_PREFIX`, `LAMBDA_TH` | f | the production set, `ps_out`, 0.3 |\n"
            "| `PARTITION` | split | auto-tuned |\n\nafter\n")
    sd = script_defaults('GRID_SIZE="${GRID_SIZE:-${DTFE_GRID_SIZE:-1024}}"\nOUTPUT_PREFIX="${OUTPUT_PREFIX:-ps_output}"   # c\n'
                         'LAMBDA_TH="${LAMBDA_TH:-0.3}"\nPARTITION="${PARTITION:-}"\nFIELDS="${FIELDS:-a b}"\n')
    check(sd == {"GRID_SIZE": "1024", "OUTPUT_PREFIX": "ps_output", "LAMBDA_TH": "0.3", "PARTITION": "", "FIELDS": "a b"},
          f"script_defaults resolves nested and commented defaults ({sd})")
    got = sorted(default_drift(knob_table(fake), sd))
    check(got == [("GRID_SIZE", "512", "1024"), ("OUTPUT_PREFIX", "ps_out", "ps_output")],
          f"a wrong number and a wrong backticked value are caught, prose and lists are not compared ({got})")
    check(FLAG_RE.findall("a -- b --grid 8 (--ps-gpu) x--y ---z") == ["--grid", "--ps-gpu"],
          "the flag pattern takes '--grid' and '(--ps-gpu)', not the ' -- ' dash or 'x--y'")
    check(code_flags([("src/user_options.cc", '("config,c", po::value)\n("NGP", "nearest")')]) >= {"--config", "--NGP"},
          "program_options names count as the binaries' flags")

    print()
    print(f"RESULT: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
