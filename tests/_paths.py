"""Make `data/` and `src/` importable from the tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("data", "src"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
