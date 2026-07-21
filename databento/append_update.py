"""Safely extend FFM continuous-contract bars from new Databento archives.

This command never splices derived CSVs. It recovers the immutable raw sources
recorded by the current 1-minute manifests, adds the new archive, deduplicates
at ``(timestamp, raw_symbol)``, and rebuilds the back-adjusted continuous series
and every model timeframe from the same raw history.

The complete candidate generation is written and validated in staging first.
``--commit`` backs up affected production streams and promotes their CSV and
manifest pairs under a dataset update lock. Without ``--commit`` it is a dry
run and prints the retained staging directory.

Usage:
    python3 databento/append_update.py NEW.dbn.zst
    python3 databento/append_update.py NEW.dbn.zst --commit
    python3 databento/append_update.py NEW.dbn.zst --tickers NQ,ES --commit

Conflicting vendor bars in an overlap fail closed. After inspecting a genuine
vendor correction, rerun with ``--allow-source-corrections`` to deliberately
prefer the newer archive and record that decision in provenance.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
DATA_DIR = _ROOT / 'data'
MODEL_PERIODS = ('1min', '3min', '5min', '15min')

_spec = importlib.util.spec_from_file_location(
    'ffm_build_continuous', _HERE / 'build_continuous.py')
bc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(bc)


def _resolve_source(value: str | Path) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, _HERE / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'raw source not found: {value}')


def _load_raw(path: Path) -> dict:
    if path.name.endswith('.dbn.zst'):
        return bc.load_dbn_zst_raw(path)
    if path.name.endswith('.csv.zst'):
        return {bc.ticker_from_path(path): bc.load_csv_zst_raw(path)}
    raise ValueError(f'unsupported source (need .dbn.zst or .csv.zst): {path}')


def _manifest_sources(ticker: str, data_dir: Path) -> list[Path]:
    """Recover and hash-check the raw lineage of the current production stream."""
    manifest_path = data_dir / f'{ticker}_1min.csv.manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError(
            f'{ticker}: current 1min provenance manifest missing: {manifest_path}; '
            'provide historical archives with --base-source')
    manifest = json.loads(manifest_path.read_text())
    paths = manifest.get('source_paths')
    hashes = manifest.get('source_sha256s')
    if not paths:
        one_path = manifest.get('source_path')
        one_hash = manifest.get('source_sha256')
        paths = [one_path] if one_path else []
        hashes = [one_hash] if one_hash else []
    if not paths or not hashes or len(paths) != len(hashes):
        raise RuntimeError(f'{ticker}: incomplete raw-source lineage in {manifest_path}')
    resolved = [_resolve_source(path) for path in paths]
    for path, expected in zip(resolved, hashes):
        actual = bc._sha256(path)
        if actual != expected:
            raise RuntimeError(
                f'{ticker}: raw source hash changed for {path}: '
                f'expected {expected}, got {actual}')
    return resolved


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _build_candidate(sources: list[Path], tickers: set[str], periods: list[str],
                     stage_dir: Path, *, allow_source_corrections: bool) -> str:
    source_maps = [(path, _load_raw(path)) for path in sources]
    merged = bc.merge_raw_sources(
        source_maps, tickers,
        allow_source_corrections=allow_source_corrections)
    missing = tickers - set(merged)
    if missing:
        raise RuntimeError(f'raw archives do not contain requested tickers: {sorted(missing)}')

    generation_id = str(uuid.uuid4())
    for ticker in sorted(tickers):
        raw = merged[ticker]
        one_min = bc._clean_ohlcv(raw, ticker, back_adjust=True)
        if one_min is None:
            raise RuntimeError(f'{ticker}: no valid bars after continuous selection')
        for key in (
                'source_paths', 'source_sha256s', 'overlap_duplicate_rows',
                'source_conflicts_accepted'):
            one_min.attrs[key] = raw.attrs[key]
        for period in periods:
            bc._save(one_min, ticker, period=period, output_dir=stage_dir,
                     generation_id=generation_id)
    bc._write_generation_manifest(
        stage_dir, tickers, periods, generation_id,
        allow_source_corrections=allow_source_corrections)
    _validate_candidate(stage_dir, tickers, periods, generation_id)
    return generation_id


def _validate_candidate(stage_dir: Path, tickers: set[str], periods: list[str],
                        generation_id: str) -> None:
    """Reload generated files and verify hashes and exact resampling parity."""
    import pandas as pd

    for ticker in sorted(tickers):
        frames = {}
        for period in periods:
            path = stage_dir / f'{ticker}_{period}.csv'
            manifest_path = path.with_suffix(path.suffix + '.manifest.json')
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('generation_id') != generation_id:
                raise RuntimeError(f'{path}: generation ID mismatch')
            if manifest.get('output_sha256') != bc._sha256(path):
                raise RuntimeError(f'{path}: output hash mismatch')
            frame = pd.read_csv(path, parse_dates=['datetime']).set_index('datetime')
            bc._validate_continuous_ohlcv(frame, str(path))
            frames[period] = frame

        base = frames['1min']
        for period in periods:
            expected = bc.resample_ohlcv(base, period)
            actual = frames[period]
            try:
                pd.testing.assert_frame_equal(
                    actual, expected, check_freq=False, check_dtype=False,
                    rtol=1e-12, atol=1e-12)
            except AssertionError as error:
                raise RuntimeError(
                    f'{ticker}@{period}: does not exactly resample from generated '
                    f'1min bars') from error


def _promote(stage_dir: Path, data_dir: Path, tickers: set[str],
             periods: list[str], generation_id: str) -> Path:
    """Back up and promote a fully validated candidate generation."""
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = data_dir / f'backup_{stamp}_{generation_id[:8]}'
    lock = data_dir / '.continuous_update.lock'
    if lock.exists():
        raise RuntimeError(f'another data update appears active: {lock}')
    data_dir.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({
        'generation_id': generation_id,
        'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    }) + '\n')
    targets = []
    for ticker in sorted(tickers):
        for period in periods:
            csv_path = data_dir / f'{ticker}_{period}.csv'
            targets.extend([csv_path, csv_path.with_suffix('.csv.manifest.json')])
    generation_target = data_dir / 'continuous_generation.json'
    targets.append(generation_target)

    backup_dir.mkdir(parents=True, exist_ok=False)
    existed = set()
    try:
        for target in targets:
            if target.exists():
                shutil.copy2(target, backup_dir / target.name)
                existed.add(target)
        for target in targets[:-1]:
            source = stage_dir / target.name
            os.replace(source, target)
        _write_full_generation_manifest(data_dir, generation_target, generation_id)
    except Exception:
        for target in targets:
            backup = backup_dir / target.name
            if backup.exists():
                shutil.copy2(backup, target)
            elif target not in existed and target.exists():
                target.unlink()
        raise
    finally:
        lock.unlink(missing_ok=True)
    return backup_dir


def _write_full_generation_manifest(data_dir: Path, target: Path,
                                    generation_id: str) -> None:
    """Write the final all-stream inventory while the update lock is held."""
    streams = {}
    for manifest_path in sorted(data_dir.glob('*_*.csv.manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        ticker = manifest.get('ticker')
        period = manifest.get('timeframe')
        output_hash = manifest.get('output_sha256')
        csv_path = data_dir / manifest_path.name.removesuffix('.manifest.json')
        if not ticker or not period or not output_hash or not csv_path.is_file():
            raise RuntimeError(f'incomplete stream pair during promotion: {manifest_path}')
        streams[f'{ticker}@{period}'] = {
            'path': csv_path.name,
            'sha256': output_hash,
            'manifest_sha256': bc._sha256(manifest_path),
        }
    payload = {
        'schema': 'ffm_continuous_generation_v1',
        'generation_id': generation_id,
        'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'streams': streams,
    }
    tmp = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', help='new Databento .dbn.zst or .csv.zst')
    parser.add_argument('--tickers', default='', help='comma list; default: all in new file')
    parser.add_argument('--periods', '--tfs', dest='periods',
                        default=','.join(MODEL_PERIODS))
    parser.add_argument('--base-source', action='append', default=[],
                        help='historical raw archive; repeatable. By default '
                             'sources are recovered from current manifests')
    parser.add_argument('--allow-source-corrections', action='store_true')
    parser.add_argument('--commit', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    new_source = _resolve_source(args.file)
    data_dir = args.data_dir.expanduser().resolve()
    periods = [item.strip() for item in args.periods.split(',') if item.strip()]
    if tuple(periods) != MODEL_PERIODS:
        parser.error(
            f'production updates must rebuild all model periods in order: '
            f'{",".join(MODEL_PERIODS)}')

    new_map = _load_raw(new_source)
    requested = {item.strip().upper() for item in args.tickers.split(',') if item.strip()}
    tickers = requested or set(new_map)
    missing_new = tickers - set(new_map)
    if missing_new:
        parser.error(f'new archive does not contain tickers: {sorted(missing_new)}')

    sources = [_resolve_source(path) for path in args.base_source]
    if not sources:
        for ticker in sorted(tickers):
            sources.extend(_manifest_sources(ticker, data_dir))
    sources = _unique_paths([*sources, new_source])

    stage_dir = Path(tempfile.mkdtemp(prefix='ffm-continuous-candidate-'))
    print(f'Tickers      : {", ".join(sorted(tickers))}')
    print(f'Raw sources  : {len(sources)}')
    print(f'Candidate dir: {stage_dir}')
    generation_id = _build_candidate(
        sources, tickers, periods, stage_dir,
        allow_source_corrections=args.allow_source_corrections)
    print(f'Validated generation {generation_id}')

    if not args.commit:
        print(f'DRY RUN complete; production unchanged. Candidate retained at {stage_dir}')
        return 0
    backup = _promote(stage_dir, data_dir, tickers, periods, generation_id)
    shutil.rmtree(stage_dir, ignore_errors=True)
    print(f'Committed generation {generation_id}')
    print(f'Previous files backed up at {backup}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
