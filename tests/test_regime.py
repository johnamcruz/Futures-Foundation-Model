"""Unit tests for futures_foundation.regime — the market-regime HMM.

The two load-bearing properties are CAUSALITY (a bar's filtered posterior must
not change when FUTURE bars change — proves forward filtering, not smoothing)
and LEAK-SAFETY (fit uses only train-mask rows — test rows cannot move the
fitted params). Both are tested directly.
"""
import numpy as np
import pytest

from futures_foundation.regime import (
    RegimeHMM, context_observations, N_CONTEXT_OBS,
    select_regime_observations, REGIME_FEATURE_NAMES,
)


# ---- select_regime_observations (existing-feature column picker) -----------
def test_select_regime_observations_picks_existing_vol_cols():
    names = ['embed_0', 'adx', 'mom_5', 'std20atr', 'foo', 'range_vs_atr']
    feat = np.arange(6 * 3, dtype=np.float32).reshape(3, 6)
    obs, picked = select_regime_observations(feat, names)
    # only canonical vol names selected, in REGIME_FEATURE_NAMES order
    assert set(picked) == {'adx', 'std20atr', 'range_vs_atr'}
    assert picked == [n for n in REGIME_FEATURE_NAMES if n in picked]
    assert obs.shape == (3, 3)


def test_select_regime_observations_raises_when_none_present():
    import pytest as _pt
    with _pt.raises(ValueError):
        select_regime_observations(np.zeros((2, 2), np.float32), ['a', 'b'])


# ---- context_observations -------------------------------------------------
def test_context_observations_shape_and_finite():
    rng = np.random.default_rng(0)
    ctx = [rng.standard_normal(40).cumsum() for _ in range(25)]
    obs = context_observations(ctx)
    assert obs.shape == (25, N_CONTEXT_OBS)
    assert np.isfinite(obs).all()


def test_efficiency_ratio_trend_vs_chop():
    # pure up-trend -> efficiency ~1; pure zig-zag chop -> efficiency ~0
    trend = np.linspace(0, 1, 50)
    chop = np.tile([0.0, 0.1], 25)
    o = context_observations([trend, chop])
    eff_trend, eff_chop = o[0, 2], o[1, 2]
    assert eff_trend > 0.9
    assert eff_chop < 0.1
    assert eff_trend > eff_chop


# ---- helpers --------------------------------------------------------------
def _stream(n_bars, n_states_signal=2, seed=0):
    """One stream, n_bars unique bars, each bar = 2 signals (both directions)
    so the grouping logic is exercised. Returns keys, obs."""
    rng = np.random.default_rng(seed)
    obs, keys = [], []
    for i in range(n_bars):
        # two distinct vol regimes so the HMM has something to find
        mu = 0.0 if (i // 20) % 2 == 0 else 5.0
        v = rng.standard_normal(N_CONTEXT_OBS) + mu
        for d in (1, -1):                       # both directions share the bar
            obs.append(v); keys.append(('S', i, d))
    return keys, np.asarray(obs, np.float32)


# ---- RegimeHMM mechanics --------------------------------------------------
def test_fit_transform_shapes_and_simplex():
    keys, obs = _stream(60)
    m = np.ones(len(keys), bool)
    hmm = RegimeHMM(n_states=2, seed=0).fit(keys, obs, m)
    post = hmm.transform(keys, obs)
    assert post.shape == (len(keys), 2)
    assert (post >= 0).all() and (post <= 1).all()
    np.testing.assert_allclose(post.sum(1), 1.0, atol=1e-5)


def test_both_directions_of_a_bar_share_posterior():
    keys, obs = _stream(40)
    hmm = RegimeHMM(n_states=2, seed=0).fit(keys, obs, np.ones(len(keys), bool))
    post = hmm.transform(keys, obs)
    # rows 0 and 1 are the same bar (d=+1 / d=-1) -> identical posterior
    np.testing.assert_array_equal(post[0], post[1])
    np.testing.assert_array_equal(post[2], post[3])


def test_transform_is_causal_no_future_peek():
    """A bar's filtered posterior must NOT change when a LATER bar's obs change
    (forward filtering). Bars at/after the change MAY move."""
    keys, obs = _stream(60)
    hmm = RegimeHMM(n_states=2, seed=0).fit(keys, obs, np.ones(len(keys), bool))
    post1 = hmm.transform(keys, obs)
    obs2 = obs.copy()
    # bar index 40 -> signal rows 80,81; perturb hard
    obs2[80:82] += 50.0
    post2 = hmm.transform(keys, obs2)
    # everything BEFORE bar 40 (rows < 80) unchanged
    np.testing.assert_allclose(post1[:80], post2[:80], atol=1e-6)
    # the changed bar itself moves (otherwise the obs aren't being used)
    assert not np.allclose(post1[80], post2[80])


def test_fit_is_leak_safe_to_train_mask():
    """Changing TEST-mask rows must not change the fitted HMM params."""
    keys, obs = _stream(60)
    mask = np.array([k[1] < 40 for k in keys])      # first 40 bars = train
    a = RegimeHMM(n_states=2, seed=0).fit(keys, obs, mask)
    obs2 = obs.copy()
    obs2[mask == False] += 99.0                      # noqa: E712 — perturb test rows
    b = RegimeHMM(n_states=2, seed=0).fit(keys, obs2, mask)
    np.testing.assert_allclose(a._means, b._means, atol=1e-6)
    np.testing.assert_allclose(a._transmat, b._transmat, atol=1e-6)


def test_save_load_roundtrip(tmp_path):
    keys, obs = _stream(60)
    hmm = RegimeHMM(n_states=2, seed=0).fit(keys, obs, np.ones(len(keys), bool))
    p = tmp_path / 'regime.joblib'
    hmm.save(p)
    loaded = RegimeHMM.load(p)
    np.testing.assert_allclose(hmm.transform(keys, obs),
                               loaded.transform(keys, obs), atol=1e-6)


def test_params_dict_serializable():
    import json
    keys, obs = _stream(60)
    hmm = RegimeHMM(n_states=2, seed=0).fit(keys, obs, np.ones(len(keys), bool))
    d = hmm.params_dict()
    assert d['n_states'] == 2
    json.dumps(d)                                    # must be JSON-serializable


def test_transform_before_fit_raises():
    keys, obs = _stream(20)
    with pytest.raises(RuntimeError):
        RegimeHMM(n_states=2).transform(keys, obs)


def test_too_few_train_bars_raises():
    keys, obs = _stream(5)                            # 5 bars, n_states=2 -> <20
    with pytest.raises(ValueError):
        RegimeHMM(n_states=2).fit(keys, obs, np.ones(len(keys), bool))


def test_pca_path_for_highdim_obs():
    """obs dim > pca_dim triggers PCA before the HMM; posteriors still valid."""
    rng = np.random.default_rng(1)
    keys = [('S', i, d) for i in range(60) for d in (1, -1)]
    obs = rng.standard_normal((len(keys), 32)).astype(np.float32)
    hmm = RegimeHMM(n_states=2, pca_dim=8, seed=0).fit(
        keys, obs, np.ones(len(keys), bool))
    post = hmm.transform(keys, obs)
    np.testing.assert_allclose(post.sum(1), 1.0, atol=1e-5)
