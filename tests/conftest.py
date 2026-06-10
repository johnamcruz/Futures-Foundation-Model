"""Suite-wide guards.

The default suite is TORCH-FREE by contract: futures_foundation parents
run xgboost in-process, and torch+xgboost libomp collide on macOS (the
segfault is at the first native xgboost call, far from the offending
import). Pytest imports every test module at COLLECTION time, so a single
module-top `import torch` poisons the whole run. Torch-needing tests gate
via CHRONOS_TORCH_TESTS=1 and import torch inside the test body.
"""
import sys


def pytest_collection_finish(session):
    assert 'torch' not in sys.modules, (
        "a test module imported torch at module top — this segfaults the "
        "shared xgboost suite (libomp collision). Gate the test with "
        "CHRONOS_TORCH_TESTS and import torch inside the test body.")
