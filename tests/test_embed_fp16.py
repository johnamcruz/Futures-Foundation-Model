"""fp16 embed contract (FFM_EMBED_FP16 in evaluate.py's batch-embed).

The walk-forward holds the flat price+volume embeds across ALL folds at once; at
fp32 that OOMs a 6-fold x 6-ticker run WITH volume (~17GB). Casting to fp16 halves
it; the per-fold SUBSET upcasts back to fp32 for XGBoost. These guard that fp16 is
(a) memory-halving and (b) precision-safe for tree splits on instance-normed embeds.
Full pipeline parity is validated by the actual walk-forward run (needs torch)."""
import numpy as np


def test_fp16_halves_memory():
    e = np.zeros((10000, 263), np.float32)        # price embed block
    assert np.asarray(e, np.float16).nbytes == e.nbytes // 2


def test_fp16_roundtrip_preserves_embed_precision():
    # Chronos embeds are instance-normed -> ~O(1) values; fp16 keeps ~3 sig digits,
    # so the fp16->fp32 upcast XGBoost trains on is negligibly different (trees
    # split on the same thresholds).
    rng = np.random.default_rng(0)
    e = rng.standard_normal((20000, 263)).astype(np.float32)
    up = np.asarray(np.asarray(e, np.float16), np.float32)   # hold fp16 -> upcast
    assert np.abs(up - e).max() < 0.01
    assert np.median(np.abs(up - e) / (np.abs(e) + 1e-6)) < 1e-3


def test_fp16_preserves_ranking_for_xgb():
    # what matters for trees is the ORDER of a feature's values, not exact magnitude.
    rng = np.random.default_rng(1)
    col = rng.standard_normal(50000).astype(np.float32)
    up = np.asarray(np.asarray(col, np.float16), np.float32)
    # near-perfect correlation -> tree split thresholds land in the same place
    assert np.corrcoef(col, up)[0, 1] > 0.9999
    # and sorting by the fp32 values leaves the fp16 values essentially monotonic
    up_sorted = up[np.argsort(col)]
    assert np.mean(np.diff(up_sorted) >= 0) > 0.99
