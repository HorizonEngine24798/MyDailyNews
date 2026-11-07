from __future__ import annotations

import atexit
from pathlib import Path
import shutil

_TEMP_TEST_ROOT = Path(__file__).resolve().parents[1] / ".codex_tmp_test"
atexit.register(shutil.rmtree, _TEMP_TEST_ROOT, True)
