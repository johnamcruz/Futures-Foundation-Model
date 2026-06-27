"""Unit tests for the fixed-percentile operating-point helper (_pct_threshold).
The operating point trades the top-X% of signals by proba — a fixed cutoff (no
val-meanR cherry-picking), which is the threshold-overfit fix."""
import numpy as np

from futures_foundation.pipeline import evaluate as ev


def test_top_pct_matches_quantile():
    p = np.arange(1000) / 1000.0                 # 0.000 .. 0.999
    # top 50% -> the median (50th percentile)
    np.testing.assert_allclose(ev._pct_threshold(p, 0.50), np.quantile(p, 0.50))
    # top 10% -> 90th percentile
    np.testing.assert_allclose(ev._pct_threshold(p, 0.10), np.quantile(p, 0.90))
    # top 25% -> 75th percentile
    np.testing.assert_allclose(ev._pct_threshold(p, 0.25), np.quantile(p, 0.75))


def test_top_pct_selects_right_fraction():
    rng = np.random.default_rng(0)
    p = rng.random(10000)
    for pct in (0.10, 0.25, 0.50, 0.75):
        thr = ev._pct_threshold(p, pct)
        frac = float((p >= thr).mean())
        assert abs(frac - pct) < 0.02            # ~top-pct fraction selected


def test_empty_returns_no_trade_cutoff():
    assert ev._pct_threshold(np.array([]), 0.50) == 1.0   # > any proba -> 0 trades


def test_full_includes_all():
    p = np.array([0.1, 0.4, 0.9])
    assert ev._pct_threshold(p, 1.0) <= p.min() + 1e-9    # top 100% -> everything


def test_default_percentile_is_usable():
    # the verdict operating point should be a usable volume (not a tiny sliver)
    assert ev.OP_PERCENTILE >= 0.25
