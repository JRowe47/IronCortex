"""Pytest configuration for IronCortex tests.

This file guarantees that the repository root is present on ``sys.path`` so
that ``import ironcortex`` works regardless of the working directory from
which tests are invoked.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
