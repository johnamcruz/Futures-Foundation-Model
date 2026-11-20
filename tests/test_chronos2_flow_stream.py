"""Optional order-flow stream (schema v2): NQ-only raw channels, causal, never filled."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune.classifiers.chronos2.multivariate import (
    FLOW_COLS,
    prepare_multivariate,
)
from futures_foundation.finetune.classifiers.chronos2.ssl_stages import _ticker_indices
from tests.test_chronos2_multivariate import TICKERS, _write_streams

KW = dict(tickers=("NQ",), holdout_start="2024-12-31T06:00:00Z", val_frac=0.2,
          context_length=16, prediction_length=4, validation_windows=4)


def _flow(path, minutes=620 * 3 + 30, seed=0, hole=None):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-12-30T00:00:00Z", periods=minutes, freq="1min")
    buy, sell = rng.integers(1, 50, minutes), rng.integers(1, 50, minutes)
    frame = pd.DataFrame({
        "timestamp": ts.tz_localize(None), "buy_vol": buy.astype(float),
        "sell_vol": sell.astype(float), "delta": (buy - sell).astype(float),
        "n_trades": (buy + sell).astype(float), "n_quotes": rng.integers(1, 9, minutes).astype(float),
        "spread_mean_ticks": 0.25 + 0.0 * buy})
    if hole is not None:
        frame = frame.drop(index=hole).reset_index(drop=True)
    frame.to_parquet(path)
    return frame


def _prepare(tmp_path, flow_file, **extra):
    _write_streams(tmp_path)
    return prepare_multivariate(tmp_path, flow_path=flow_file, **KW, **extra)


def test_default_has_no_flow_channels(tmp_path):
    _write_streams(tmp_path)
    result = prepare_multivariate(tmp_path, **KW)
    assert result.train.shape[0] == 5
    assert result.report["schema"] == "ffm_chronos2_multivariate_v1"


def test_flow_appended_after_nq_ohlcv_with_v2_identity(tmp_path):
    _flow(tmp_path / "flow.parquet")
    result = _prepare(tmp_path, tmp_path / "flow.parquet")
    assert result.channel_names[:5] == tuple(f"NQ.{c}" for c in ("open", "high", "low", "close", "volume"))
    assert result.channel_names[5:] == tuple(f"NQ.flow_{c}" for c in FLOW_COLS)
    assert result.report["schema"] == "ffm_chronos2_multivariate_v2_flow"
    assert len(result.report["flow"]["source_sha256"]) == 64
    assert result.train.shape[0] == 5 + len(FLOW_COLS)


def test_three_minute_bar_holds_only_its_own_three_minutes(tmp_path):
    frame = _flow(tmp_path / "flow.parquet")
    result = _prepare(tmp_path, tmp_path / "flow.parquet")
    closes = pd.DatetimeIndex(result.train_close_times)
    i = 40
    open_ts = (closes[i] - pd.Timedelta("3min")).tz_localize(None)
    want = frame[(frame.timestamp >= open_ts) & (frame.timestamp < open_ts + pd.Timedelta("3min"))]
    got = result.train[5 + FLOW_COLS.index("buy_vol"), i]
    assert got == pytest.approx(want.buy_vol.sum())
    assert result.train[5 + FLOW_COLS.index("spread_mean_ticks"), i] == pytest.approx(0.25)


def test_future_flow_never_changes_earlier_bars(tmp_path):
    frame = _flow(tmp_path / "a.parquet")
    a = _prepare(tmp_path, tmp_path / "a.parquet")
    changed = frame.copy()
    cut = pd.Timestamp("2024-12-30T20:00:00")
    changed.loc[changed.timestamp >= cut, ["buy_vol", "sell_vol", "delta"]] *= 7
    changed.to_parquet(tmp_path / "b.parquet")
    b = prepare_multivariate(tmp_path, flow_path=tmp_path / "b.parquet", **KW)
    closes = pd.DatetimeIndex(a.train_close_times)
    before = np.where(closes <= pd.Timestamp("2024-12-30T20:00:00Z"))[0]
    assert len(before)
    assert np.array_equal(a.train[:, before], b.train[:, before], equal_nan=True)


def test_missing_flow_minutes_stay_nan_not_zero(tmp_path):
    _flow(tmp_path / "flow.parquet", hole=list(range(300, 306)))
    result = _prepare(tmp_path, tmp_path / "flow.parquet")
    closes = pd.DatetimeIndex(result.train_close_times)
    gap_open = pd.Timestamp("2024-12-30T05:03:00Z")  # minutes 303..305 absent entirely
    row = np.where(closes == gap_open + pd.Timedelta("3min"))[0]
    assert len(row) and np.isnan(result.train[5:, row[0]]).all()


def test_flow_only_on_requested_ticker_and_rejects_bad_columns(tmp_path):
    _flow(tmp_path / "flow.parquet")
    _write_streams(tmp_path)
    with pytest.raises(ValueError, match="flow_ticker"):
        prepare_multivariate(tmp_path, flow_path=tmp_path / "flow.parquet",
                             flow_ticker="ES", **KW)
    with pytest.raises(ValueError, match="unsupported"):
        prepare_multivariate(tmp_path, flow_path=tmp_path / "flow.parquet",
                             flow_columns=("ofi",), **KW)
    both = dict(KW, tickers=TICKERS)
    r = prepare_multivariate(tmp_path, flow_path=tmp_path / "flow.parquet", **both)
    es = [n for n in r.channel_names if n.startswith("ES.")]
    assert len(es) == 5 and sum(n.startswith("NQ.flow_") for n in r.channel_names) == len(FLOW_COLS)


def test_ticker_indices_strict_by_default_flow_only_when_allowed(tmp_path):
    _flow(tmp_path / "flow.parquet")
    result = _prepare(tmp_path, tmp_path / "flow.parquet")
    with pytest.raises(RuntimeError, match="exactly five"):
        _ticker_indices(result)
    idx = _ticker_indices(result, allow_flow=True)
    assert len(idx["NQ"]) == 5 + len(FLOW_COLS)

    class Fake:  # a non-flow extra channel is still refused
        channel_names = ("NQ.open", "NQ.high", "NQ.low", "NQ.close", "NQ.volume", "NQ.extra")
    with pytest.raises(RuntimeError):
        _ticker_indices(Fake, allow_flow=True)


def _sessions_frame(days=28):
    """One flow value per minute encoding (session number, minute of session); sessions roll at 17:00 America/Chicago."""
    idx = pd.date_range("2025-01-05T23:00:00Z", periods=days * 1440, freq="1min")   # 17:00 CST
    sess = (np.arange(len(idx)) // 1440).astype(float)
    return pd.DataFrame({"flow_buy_vol": sess * 10000 + np.arange(len(idx)) % 1440}, index=idx)


def test_shuffle_control_moves_whole_exchange_sessions_with_same_weekday():
    from futures_foundation.finetune.classifiers.chronos2.multivariate import _shuffle_sessions
    frame = _sessions_frame()
    out = _shuffle_sessions(frame, 0)
    assert len(out) == len(frame) and out.index.equals(frame.index)
    assert sorted(out.flow_buy_vol) == sorted(frame.flow_buy_vol)
    assert (out.flow_buy_vol.values != frame.flow_buy_vol.values).mean() > 0.95      # no session keeps its own flow
    # every target session is built from exactly ONE source session, with the same minute-of-session
    tgt_sess = (np.arange(len(out)) // 1440)
    for t in (3, 10, 17):
        rows = out.flow_buy_vol.values[tgt_sess == t]
        assert len({int(v) // 10000 for v in rows}) == 1
        assert (rows % 10000 == np.arange(1440)).all()
        assert (int(rows[0]) // 10000 - t) % 7 == 0 and int(rows[0]) // 10000 != t      # never its own flow
    assert out.equals(_shuffle_sessions(frame, 0)) and not out.equals(_shuffle_sessions(frame, 1))


def test_shuffle_keeps_clock_alignment_across_the_daylight_saving_change():
    from futures_foundation.finetune.classifiers.chronos2.multivariate import _shuffle_sessions
    idx = pd.date_range("2025-03-02T23:00:00Z", periods=21 * 1440, freq="1min")   # spans 2025-03-09 US DST start
    frame = pd.DataFrame({"flow_buy_vol": np.ones(len(idx))}, index=idx)
    out = _shuffle_sessions(frame, 0)
    assert out.flow_buy_vol.notna().mean() > 0.99
    assert len(out) >= len(frame) - 2 * 60 * 3        # at most a few DST-gap hours go missing, not a whole shifted hour per session


def test_shuffle_never_uses_flow_at_or_after_the_holdout(tmp_path):
    frame = _flow(tmp_path / "flow.parquet")
    _write_streams(tmp_path)
    marker = frame.copy(); marker.loc[marker.timestamp >= pd.Timestamp("2024-12-31T12:00:00"), "buy_vol"] = 7e9
    marker.to_parquet(tmp_path / "late.parquet")
    r = prepare_multivariate(tmp_path, flow_path=tmp_path / "late.parquet", flow_control="shuf",
                             **dict(KW, holdout_start="2024-12-31T06:00:00Z"))
    assert not (r.train[5 + FLOW_COLS.index("buy_vol")] > 1e9).any()
    assert not (r.validation_matrix[5 + FLOW_COLS.index("buy_vol")] > 1e9).any()


def test_shuffle_control_wires_through_prepare_and_changes_the_flow_rows(tmp_path):
    long_kw = dict(KW, holdout_start="2025-01-25T00:00:00Z")
    _write_streams(tmp_path, rows=13440)
    _flow(tmp_path / "flow.parquet", minutes=13440 * 3 + 30)
    r = prepare_multivariate(tmp_path, flow_path=tmp_path / "flow.parquet", **long_kw)
    s = prepare_multivariate(tmp_path, flow_path=tmp_path / "flow.parquet", flow_control="shuf", **long_kw)
    assert np.array_equal(r.train[:5], s.train[:5])
    assert not np.array_equal(r.train[5:], s.train[5:], equal_nan=True)
    assert s.report["flow"]["control"] == "shuf"


def test_control_without_a_flow_file_is_refused(tmp_path):
    _write_streams(tmp_path)
    for control in ("shuf", "proxy"):
        with pytest.raises(ValueError, match="flow_path"):
            prepare_multivariate(tmp_path, flow_control=control, **KW)


def test_partial_source_minutes_give_the_partial_sum(tmp_path):
    frame = _flow(tmp_path / "flow.parquet", hole=[300])      # one minute of a 3-minute bar missing
    result = _prepare(tmp_path, tmp_path / "flow.parquet")
    closes = pd.DatetimeIndex(result.train_close_times)
    row = np.where(closes == pd.Timestamp("2024-12-30T05:00:00Z") + pd.Timedelta("3min"))[0]
    open_ts = pd.Timestamp("2024-12-30T05:00:00")
    want = frame[(frame.timestamp >= open_ts) & (frame.timestamp < open_ts + pd.Timedelta("3min"))].buy_vol.sum()
    assert result.train[5 + FLOW_COLS.index("buy_vol"), row[0]] == pytest.approx(want)


def test_proxy_control_is_a_function_of_price_only(tmp_path):
    _flow(tmp_path / "a.parquet", seed=1)
    _flow(tmp_path / "b.parquet", seed=2)
    a = _prepare(tmp_path, tmp_path / "a.parquet", flow_control="proxy")
    b = prepare_multivariate(tmp_path, flow_path=tmp_path / "b.parquet",
                             flow_control="proxy", **KW)
    assert np.array_equal(a.train, b.train, equal_nan=True)  # file content irrelevant
    o, c, v = a.train[0], a.train[3], a.train[4]
    assert np.allclose(a.train[5 + FLOW_COLS.index("buy_vol")], np.where(c > o, v, 0.0))
    assert np.allclose(a.train[5 + FLOW_COLS.index("n_trades")], v)


def test_unknown_control_rejected(tmp_path):
    _flow(tmp_path / "flow.parquet")
    with pytest.raises(ValueError, match="flow_control"):
        _prepare(tmp_path, tmp_path / "flow.parquet", flow_control="x")
