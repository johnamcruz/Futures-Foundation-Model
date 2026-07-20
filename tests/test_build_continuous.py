"""Contract-roll integrity tests for the Databento continuous builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


_PATH = Path(__file__).parents[1] / 'databento' / 'build_continuous.py'
_SPEC = importlib.util.spec_from_file_location('build_continuous_test', _PATH)
bc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bc)


def _roll_frame() -> pd.DataFrame:
    # Two CME sessions, with both contracts trading every minute.  Per-minute
    # volume alternates so the retired row-wise winner would manufacture bars
    # containing both the 100 and 200 price levels.
    first = pd.date_range('2026-03-08 22:00', periods=6, freq='1min', tz='UTC')
    second = pd.date_range('2026-03-09 22:00', periods=6, freq='1min', tz='UTC')
    rows = []
    for session_no, timestamps in enumerate((first, second)):
        for minute, ts in enumerate(timestamps):
            for symbol, base in (('NQH6', 100.0), ('NQM6', 200.0)):
                dominant = ('NQH6' if session_no == 0 else 'NQM6')
                volume = 100 if symbol == dominant else (150 if minute % 2 == 0 else 1)
                price = base + session_no + minute
                rows.append({
                    'datetime': ts, 'symbol': symbol, 'open': price,
                    'high': price + 1, 'low': price - 1, 'close': price + .5,
                    'volume': volume,
                })
    return pd.DataFrame(rows).set_index('datetime')


def test_selects_one_contract_for_entire_cme_session():
    clean = bc._clean_ohlcv(_roll_frame(), 'NQ', back_adjust=False)
    first = clean.loc['2026-03-08']
    second = clean.loc['2026-03-09']
    assert (first['close'] < 150).all()
    assert (second['close'] > 150).all()
    bars = bc.resample_ohlcv(clean, '3min')
    assert (bars['high'] - bars['low']).max() < 10


def test_back_adjust_removes_untradeable_contract_basis_gap():
    clean = bc._clean_ohlcv(_roll_frame(), 'NQ', back_adjust=True)
    seam = clean['open'].diff().abs().max()
    assert seam < 10
    assert clean.loc['2026-03-08', 'close'].median() > 190
    assert clean.loc['2026-03-09', 'close'].median() > 200


def test_spreads_never_enter_continuous_bars():
    raw = _roll_frame()
    spread = raw.iloc[[0]].copy()
    spread['symbol'] = 'NQH6-NQM6'
    spread[['open', 'high', 'low', 'close']] = -100.0
    clean = bc._clean_ohlcv(pd.concat([raw, spread]), 'NQ')
    assert (clean['close'] > 0).all()
