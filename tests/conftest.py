"""Suite-wide guards.

The default suite is TORCH-FREE by contract: futures_foundation parents
run xgboost in-process, and torch+xgboost libomp collide on macOS (the
segfault is at the first native xgboost call, far from the offending
import). Pytest imports every test module at COLLECTION time, so a single
module-top `import torch` poisons the whole run. Torch-needing tests gate
via CHRONOS_TORCH_TESTS=1 and import torch inside the test body.
"""
import os
import sys

# Make the repo root importable under a BARE `pytest` invocation (CI) — the
# package isn't pip-installed there and bare pytest doesn't put the repo root on
# sys.path, so `import futures_foundation` at a test module's top fails at
# collection. conftest loads before any test module, so this runs first.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def pytest_collection_finish(session):
    assert 'torch' not in sys.modules, (
        "a test module imported torch at module top — this segfaults the "
        "shared xgboost suite (libomp collision). Gate the test with "
        "CHRONOS_TORCH_TESTS and import torch inside the test body.")
