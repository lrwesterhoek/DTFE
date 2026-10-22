"""The shared harness of the Python suites (tests/py_*_test.py): the preamble, the PASS / FAIL / SKIP lists, Skip,
check(name, fn), check_ok(name, ok, detail) and the closing RESULT line.

Import it before any python/ module and call setup() first: the figure root must point at a temp folder before
config is imported. Standard library only, so it never imports config itself.
"""

import atexit
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PASS, FAIL, SKIP = [], [], []


class Skip(Exception):
    """Raised by a check that cannot run here (its data or binary is not on disk): a SKIP line, not a failure."""


def setup(prefix, paths=("python", "python/plot"), mpl_agg=True, figures_sub=None):
    """The suites' preamble: each of 'paths' (repo-relative) inserted at the front of sys.path in turn (the last one
    ends up first), a fresh temp folder (mkdtemp 'prefix') as DTFE_FIGURES_ROOT -- or its subfolder 'figures_sub'
    -- removed at exit, so a suite never writes into the real figure root; mpl_agg defaults MPLBACKEND to Agg.
    Returns the temp folder."""
    for p in paths:
        sys.path.insert(0, str(REPO / p))
    tmp = tempfile.mkdtemp(prefix=prefix)
    os.environ["DTFE_FIGURES_ROOT"] = str(Path(tmp) / figures_sub) if figures_sub else tmp
    if mpl_agg:
        os.environ.setdefault("MPLBACKEND", "Agg")
    atexit.register(shutil.rmtree, tmp, True)
    return tmp


def data_root():
    """The folder the data-backed checks read: $DTFE_DATA_ROOT (~ expanded). Unset, the check is a SKIP: the data
    tiers never fail for want of data (dtfelib.cli.require_data_root has no default either)."""
    root = os.environ.get("DTFE_DATA_ROOT")
    if not root:
        raise Skip("no data root: DTFE_DATA_ROOT is unset")
    return Path(root).expanduser()


def check(name, fn, missing=(), data_backed=False, required=False, crash_traceback=True):
    """Run fn() and print one line: PASS when it returns; SKIP on Skip -- a FAIL instead when 'required' and
    DTFE_TEST_REQUIRED=1; FAIL with the message on an AssertionError. An exception of the 'missing' types, or
    dtfelib's NoDataRoot (DTFE_DATA_ROOT unset), is a SKIP ('data not on disk') when data_backed; a 'missing' one is
    a FAIL otherwise. Any other exception is this check's FAIL, not the end of the run, followed by the traceback's
    last six lines unless crash_traceback is False."""
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Skip as e:
        if required and os.environ.get("DTFE_TEST_REQUIRED", "0") == "1":
            FAIL.append(name)
            print(f"  FAIL  {name}: required but skipped: {e}")
        else:
            SKIP.append(name)
            print(f"  SKIP  {name}  ({e})")
    except AssertionError as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        no_data = isinstance(e, missing) or type(e).__name__ == "NoDataRoot"     # (by name: stdlib only here)
        if no_data and data_backed:
            SKIP.append(name)
            print(f"  SKIP  {name}  (data not on disk: {e})")
            return
        if isinstance(e, missing):  # a synthetic test builds its own inputs: a missing file or key is a FAILURE
            FAIL.append(name)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            return
        FAIL.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        if crash_traceback:
            print("".join("        " + line for line in traceback.format_exc().splitlines(True)[-6:]), end="")


def check_ok(name, ok, detail=""):
    """The launcher suites' check of a value: PASS / FAIL by 'ok', the detail shown on a failure only."""
    (PASS if ok else FAIL).append(name)
    print(f"   {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))


def summary(blank_line=False, show_zero_skips=True):
    """Print 'RESULT: <p> passed, <f> failed, <s> skipped' (after an empty line when blank_line; without ', 0
    skipped' when show_zero_skips is False) and return the exit code: 1 when a check failed."""
    line = f"RESULT: {len(PASS)} passed, {len(FAIL)} failed"
    if show_zero_skips or SKIP:
        line += f", {len(SKIP)} skipped"
    print(("\n" if blank_line else "") + line)
    return 1 if FAIL else 0
