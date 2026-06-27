"""Unit tests for average-uniqueness sample weights (López de Prado).
Includes HAND-COMPUTED concurrency cases so the math is verified, not just smoke."""
import numpy as np

from futures_foundation.uniqueness import uniqueness_weights


def test_isolated_signals_full_uniqueness():
    # signals spaced > horizon apart -> never concurrent -> uniqueness 1.0
    keys = [('S', 0), ('S', 50), ('S', 100)]
    w = uniqueness_weights(keys, horizon=2, normalize=False)
    np.testing.assert_allclose(w, [1.0, 1.0, 1.0], atol=1e-6)


def test_overlap_downweights_and_exact_values():
    # horizon=2, signals at bars 0,1,2 -> spans [1,3],[2,4],[3,5]
    # concurrency per bar: 1:1  2:2  3:3  4:2  5:1
    #   s0 [1,3]: mean(1, 1/2, 1/3)        = 0.6111
    #   s1 [2,4]: mean(1/2, 1/3, 1/2)      = 0.4444
    #   s2 [3,5]: mean(1/3, 1/2, 1)        = 0.6111
    keys = [('S', 0), ('S', 1), ('S', 2)]
    w = uniqueness_weights(keys, horizon=2, normalize=False)
    np.testing.assert_allclose(w, [0.611111, 0.444444, 0.611111], atol=1e-5)
    assert w[1] < w[0] and w[1] < w[2]            # most-overlapping is lowest


def test_normalize_mean_one():
    keys = [('S', 0), ('S', 1), ('S', 2), ('S', 30)]
    w = uniqueness_weights(keys, horizon=5, normalize=True)
    assert abs(float(w.mean()) - 1.0) < 1e-5
    assert (w > 0).all() and np.isfinite(w).all()


def test_both_directions_share_concurrency():
    # two labels on the SAME bar (both directions) are separate overlapping
    # labels -> each spans [1,3], concurrency 2 over the whole span -> uniq 0.5;
    # an isolated bar-100 label -> uniq 1.0. Cluster must be down-weighted.
    keys = [('S', 0), ('S', 0), ('S', 100)]
    w = uniqueness_weights(keys, horizon=2, normalize=False)
    np.testing.assert_allclose(w[:2], [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(w[2], 1.0, atol=1e-6)
    assert w[0] < w[2]


def test_per_stream_independent():
    # overlap in stream A must not affect isolated stream B
    keys = [('A', 0), ('A', 1), ('A', 2), ('B', 0)]
    w = uniqueness_weights(keys, horizon=2, normalize=False)
    np.testing.assert_allclose(w[3], 1.0, atol=1e-6)     # B isolated -> 1.0
    assert w[1] < 1.0                                     # A middle overlapped


def test_shape_and_positivity():
    rng = np.random.default_rng(0)
    bars = np.sort(rng.integers(0, 5000, 2000))
    keys = [('S', int(b)) for b in bars]
    w = uniqueness_weights(keys, horizon=150, normalize=True)
    assert w.shape == (2000,)
    assert (w > 0).all() and np.isfinite(w).all()
    assert w.dtype == np.float32


def test_empty():
    w = uniqueness_weights([], horizon=10)
    assert w.shape == (0,)


def test_more_overlap_lower_weight_monotone():
    # denser cluster (horizon spanning many) -> lower average weight than sparse
    dense = [('S', i) for i in range(20)]            # heavy overlap, horizon 10
    w_dense = uniqueness_weights(dense, horizon=10, normalize=False)
    sparse = [('S', i * 100) for i in range(20)]     # no overlap
    w_sparse = uniqueness_weights(sparse, horizon=10, normalize=False)
    assert w_dense.mean() < w_sparse.mean()
    np.testing.assert_allclose(w_sparse, 1.0, atol=1e-6)
