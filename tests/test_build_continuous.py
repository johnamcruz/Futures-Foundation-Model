"""Contract-roll integrity tests for the Databento continuous builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


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


def test_back_adjust_requires_simultaneous_contract_overlap():
    raw = _roll_frame()
    raw = raw[~(
        ((raw.index < pd.Timestamp('2026-03-09', tz='UTC')) &
         (raw['symbol'] == 'NQM6')) |
        ((raw.index >= pd.Timestamp('2026-03-09', tz='UTC')) &
         (raw['symbol'] == 'NQH6'))
    )]
    with pytest.raises(ValueError, match='fewer than 5 simultaneous'):
        bc._clean_ohlcv(raw, 'NQ', back_adjust=True)


def test_spreads_never_enter_continuous_bars():
    raw = _roll_frame()
    spread = raw.iloc[[0]].copy()
    spread['symbol'] = 'NQH6-NQM6'
    spread[['open', 'high', 'low', 'close']] = -100.0
    clean = bc._clean_ohlcv(pd.concat([raw, spread]), 'NQ')
    assert (clean['close'] > 0).all()


def test_merge_raw_sources_prefers_new_overlap_and_preserves_contracts(tmp_path):
    old_path = tmp_path / 'old.dbn.zst'
    new_path = tmp_path / 'new.dbn.zst'
    old_path.write_bytes(b'old')
    new_path.write_bytes(b'new')
    old = _roll_frame()
    new = old.loc[old.index >= pd.Timestamp('2026-03-09', tz='UTC')].copy()
    first_key = new.index[0]
    new.loc[first_key, 'close'] += 7.0

    with pytest.raises(ValueError, match='conflicting overlap bars'):
        bc.merge_raw_sources([
            (old_path, {'NQ': old}),
            (new_path, {'NQ': new}),
        ])

    merged = bc.merge_raw_sources([
        (old_path, {'NQ': old}),
        (new_path, {'NQ': new}),
    ], allow_source_corrections=True)['NQ']

    # Both outright contracts survive at a timestamp, but the newer archive's
    # duplicate row replaces the old archive's copy.
    rows = merged.loc[first_key]
    assert set(rows['symbol']) == {'NQH6', 'NQM6'}
    symbol = new.loc[first_key, 'symbol'].iloc[0]
    assert rows.loc[rows['symbol'] == symbol, 'close'].iloc[0] == \
        new.loc[first_key].loc[new.loc[first_key, 'symbol'] == symbol, 'close'].iloc[0]
    assert merged.attrs['source_paths'] == [str(old_path), str(new_path)]
    assert merged.attrs['source_conflicts_accepted'] == 2


def test_merge_raw_sources_accepts_identical_overlap_without_override(tmp_path):
    old_path = tmp_path / 'old.dbn.zst'
    new_path = tmp_path / 'new.dbn.zst'
    old_path.write_bytes(b'old')
    new_path.write_bytes(b'new')
    old = _roll_frame()
    new = old.loc[old.index >= pd.Timestamp('2026-03-09', tz='UTC')].copy()
    merged = bc.merge_raw_sources([
        (old_path, {'NQ': old}), (new_path, {'NQ': new}),
    ])['NQ']
    assert not merged.reset_index().duplicated(['datetime', 'symbol']).any()
    assert merged.attrs['source_conflicts_accepted'] == 0
    assert merged.attrs['overlap_duplicate_rows'] == len(new)


@pytest.mark.parametrize('column,value', [
    ('open', -1.0), ('high', 50.0), ('low', 250.0), ('volume', -1.0),
])
def test_invalid_outright_ohlcv_fails_closed(column, value):
    raw = _roll_frame()
    raw.iloc[0, raw.columns.get_loc(column)] = value
    with pytest.raises(ValueError, match='invalid outright OHLCV'):
        bc._clean_ohlcv(raw, 'NQ')


def test_contract_roll_flip_flop_fails_closed():
    raw = _roll_frame()
    third = raw.loc['2026-03-08'].copy()
    third.index = third.index + pd.Timedelta(days=2)
    with pytest.raises(ValueError, match='roll flip-flop'):
        bc._clean_ohlcv(pd.concat([raw, third]), 'NQ')


def test_save_is_hash_sealed_and_has_generation_id(tmp_path):
    clean = bc._clean_ohlcv(_roll_frame(), 'NQ', back_adjust=True)
    bc._save(clean, 'NQ', '1min', tmp_path, generation_id='generation-test')
    path = tmp_path / 'NQ_1min.csv'
    import json
    manifest = json.loads(path.with_suffix('.csv.manifest.json').read_text())
    assert manifest['generation_id'] == 'generation-test'
    assert manifest['output_sha256'] == bc._sha256(path)
    assert not list(tmp_path.glob('.*.tmp'))
