"""Regression tests for transactional Databento updates."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


au = _module('append_update_test', 'databento/append_update.py')


def _one_min() -> pd.DataFrame:
    index = pd.date_range('2026-01-05 14:30', periods=12, freq='1min', tz='UTC')
    values = pd.Series(range(100, 112), index=index, dtype=float)
    return pd.DataFrame({
        'open': values, 'high': values + 1, 'low': values - 1,
        'close': values + .5, 'volume': 10,
    }, index=index)


def _candidate(tmp_path: Path, generation: str = 'candidate') -> Path:
    stage = tmp_path / 'stage'
    stage.mkdir()
    one_min = _one_min()
    one_min.attrs['continuous_contract'] = {
        'schema': 'ffm_continuous_contract_v1',
        'selection': 'cme_session_total_volume', 'back_adjusted': True,
    }
    for period in au.MODEL_PERIODS:
        au.bc._save(one_min, 'NQ', period, stage, generation_id=generation)
    au.bc._write_generation_manifest(
        stage, {'NQ'}, list(au.MODEL_PERIODS), generation,
        allow_source_corrections=False)
    return stage


def test_candidate_requires_exact_cross_timeframe_resampling(tmp_path):
    stage = _candidate(tmp_path)
    au._validate_candidate(stage, {'NQ'}, list(au.MODEL_PERIODS), 'candidate')
    path = stage / 'NQ_3min.csv'
    frame = pd.read_csv(path)
    frame.loc[0, 'close'] += .1
    frame.to_csv(path, index=False)
    manifest_path = path.with_suffix('.csv.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    manifest['output_sha256'] = au.bc._sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='does not exactly resample'):
        au._validate_candidate(stage, {'NQ'}, list(au.MODEL_PERIODS), 'candidate')


def test_manifest_source_hash_drift_fails_closed(tmp_path):
    source = tmp_path / 'NQ-old.csv.zst'
    source.write_bytes(b'original')
    digest = au.bc._sha256(source)
    manifest = {
        'source_paths': [str(source)], 'source_sha256s': [digest],
    }
    (tmp_path / 'NQ_1min.csv.manifest.json').write_text(json.dumps(manifest))
    source.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='raw source hash changed'):
        au._manifest_sources('NQ', tmp_path)


def test_promote_moves_all_four_timeframes_and_preserves_backup(tmp_path):
    stage = _candidate(tmp_path)
    data = tmp_path / 'data'
    data.mkdir()
    old = data / 'NQ_1min.csv'
    old.write_text('old production')
    old.with_suffix('.csv.manifest.json').write_text('old manifest')

    backup = au._promote(
        stage, data, {'NQ'}, list(au.MODEL_PERIODS), 'candidate')

    assert (backup / 'NQ_1min.csv').read_text() == 'old production'
    assert (data / 'continuous_generation.json').is_file()
    assert not (data / '.continuous_update.lock').exists()
    for period in au.MODEL_PERIODS:
        assert (data / f'NQ_{period}.csv').is_file()
        assert (data / f'NQ_{period}.csv.manifest.json').is_file()


def test_promote_refuses_active_update_lock(tmp_path):
    stage = _candidate(tmp_path)
    data = tmp_path / 'data'
    data.mkdir()
    (data / '.continuous_update.lock').write_text('busy')
    with pytest.raises(RuntimeError, match='another data update'):
        au._promote(stage, data, {'NQ'}, list(au.MODEL_PERIODS), 'candidate')
