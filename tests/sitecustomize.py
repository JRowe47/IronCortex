"""Test suite path configuration.

Ensures `import ironcortex` works when tests are executed directly with
`python tests/test_*.py` by inserting the repository root into
``sys.path``. Python automatically imports this module if it is present on
``sys.path`` (e.g. the ``tests`` directory).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
