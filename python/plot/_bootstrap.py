"""Puts python/ on sys.path so the plot scripts can be run directly from any cwd.

Every script in this directory starts with `import _bootstrap` (this file lives next to
them, so it is always importable); after that, `config` and `dtfelib` resolve normally.
"""

import sys
from pathlib import Path

_python_root = str(Path(__file__).resolve().parents[1])
if _python_root not in sys.path:
    sys.path.insert(0, _python_root)
