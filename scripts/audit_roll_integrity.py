"""Audit roll contamination and downstream cache provenance across FFM + Algo.

The report compares legacy continuous CSVs with symbol-aware, session-dominant,
back-adjusted replay CSVs.  It also fails closed when Algo signal caches do not
pin the exact corrected bar hashes, because such caches cannot prove which
market-data revision trained/evaluated a policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TICKERS = ('CL', 'ES', 'GC', 'NQ', 'RTY', 'SI', 'YM', 'ZB', 'ZN')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=['datetime']).set_index('datetime')
    return frame[['open', 'high', 'low', 'close', 'volume']].sort_index()


def _outliers(frame: pd.DataFrame, threshold: float = 8.0) -> tuple[pd.Series, pd.Series]:
    previous = frame['close'].shift()
    true_range = pd.concat([
        frame['high'] - frame['low'],
        (frame['high'] - previous).abs(),
        (frame['low'] - previous).abs(),
    ], axis=1).max(axis=1)
    # Prior-only robust scale: the contaminated bar cannot inflate its own
    # denominator and hide the very mixed-contract range being audited.
    scale = true_range.shift().rolling(100, min_periods=20).median()
    mixed = (frame['high'] - frame['low']) > threshold * scale
    jump = frame['close'].diff().abs() > threshold * scale
    return mixed.fillna(False), jump.fillna(False)


def _dilated_count(mask: pd.Series, bars_after: int) -> int:
    values = mask.to_numpy(bool)
    tainted = np.zeros(len(values), dtype=bool)
    for index in np.flatnonzero(values):
        tainted[index:min(len(values), index + bars_after + 1)] = True
    return int(tainted.sum())


def audit_ticker(legacy: Path, clean: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    clean_hash_ok = manifest.get('output_sha256') == _sha256(clean)
    legacy_df, clean_df = _load(legacy), _load(clean)
    start = max(legacy_df.index.min(), clean_df.index.min())
    end = min(legacy_df.index.max(), clean_df.index.max())
    legacy_df = legacy_df.loc[start:end]
    clean_df = clean_df.loc[start:end]
    old_mixed, old_jump = _outliers(legacy_df)
    new_mixed, new_jump = _outliers(clean_df)

    roll_windows = []
    for adjustment in manifest.get('roll_adjustments', []):
        switch = pd.Timestamp(adjustment['switch_timestamp'])
        lo, hi = switch - pd.Timedelta(days=1), switch + pd.Timedelta(days=1)
        old_count = int((old_mixed | old_jump).loc[lo:hi].sum())
        new_count = int((new_mixed | new_jump).loc[lo:hi].sum())
        roll_windows.append({**adjustment,
                             'legacy_outlier_bars': old_count,
                             'clean_outlier_bars': new_count})

    aligned_old, aligned_new = legacy_df.align(clean_df, join='inner', axis=0)
    old_return = aligned_old['close'].diff()
    new_return = aligned_new['close'].diff()
    return_changed = int(((old_return - new_return).abs() > 1e-9).sum())
    old_any, new_any = old_mixed | old_jump, new_mixed | new_jump
    return {
        'legacy_path': str(legacy), 'clean_path': str(clean),
        'manifest_path': str(manifest_path),
        'manifest_hash_valid': clean_hash_ok,
        'rows_compared': int(len(aligned_old)),
        'legacy_mixed_range_bars': int(old_mixed.sum()),
        'legacy_jump_bars': int(old_jump.sum()),
        'clean_mixed_range_bars': int(new_mixed.sum()),
        'clean_jump_bars': int(new_jump.sum()),
        'legacy_encoder_windows_tainted_128_est': _dilated_count(old_any, 127),
        'clean_encoder_windows_tainted_128_est': _dilated_count(new_any, 127),
        'return_bars_changed': return_changed,
        'roll_windows': roll_windows,
        'roll_outliers_removed': int(sum(
            max(0, item['legacy_outlier_bars'] - item['clean_outlier_bars'])
            for item in roll_windows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parents[1]
    parser.add_argument('--legacy-dir', type=Path, default=root / 'data')
    parser.add_argument('--clean-dir', type=Path, default=root / 'data' / 'replay')
    parser.add_argument('--algo-root', type=Path,
                        default=root.parent / 'algoTraderAI')
    parser.add_argument('--tickers', default=','.join(TICKERS))
    parser.add_argument('--timeframe', default='3min')
    parser.add_argument('--output', type=Path,
                        default=root / 'temp' / 'roll_audit_3min' / 'audit.json')
    args = parser.parse_args()

    tickers = [item.strip().upper() for item in args.tickers.split(',')
               if item.strip()]
    report = {'schema': 'ffm_roll_integrity_audit_v1', 'tickers': {},
              'missing': [], 'downstream': {}}
    for ticker in tickers:
        legacy = args.legacy_dir / f'{ticker}_{args.timeframe}.csv'
        clean = args.clean_dir / f'{ticker}_{args.timeframe}.csv'
        manifest = clean.with_suffix(clean.suffix + '.manifest.json')
        missing = [str(path) for path in (legacy, clean, manifest) if not path.exists()]
        if missing:
            report['missing'].extend(missing)
            continue
        report['tickers'][ticker] = audit_ticker(legacy, clean, manifest)

    cache_manifest = (args.algo_root / 'data' / 'cache' /
                      'mantis_pivot_trend_v4_signals.joblib.manifest.json')
    if cache_manifest.exists():
        cache = json.loads(cache_manifest.read_text())
        pinned = cache.get('bar_source_sha256') or {}
        corrected_hashes = {
            ticker: item['manifest_hash_valid'] and _sha256(
                Path(item['clean_path']))
            for ticker, item in report['tickers'].items()
        }
        exact = bool(corrected_hashes) and all(
            pinned.get(ticker) == digest
            for ticker, digest in corrected_hashes.items())
        report['downstream']['algo_signal_cache'] = {
            'path': str(cache_manifest), 'pins_corrected_bars': exact,
            'bar_source_sha256': pinned,
            'rebuild_required': not exact,
        }
    else:
        report['downstream']['algo_signal_cache'] = {
            'path': str(cache_manifest), 'rebuild_required': True,
            'reason': 'manifest missing',
        }

    report['passed_data_integrity'] = (
        not report['missing']
        and len(report['tickers']) == len(tickers)
        and all(item['manifest_hash_valid'] for item in report['tickers'].values())
    )
    report['production_revalidation_required'] = (
        not report['passed_data_integrity']
        or report['downstream']['algo_signal_cache']['rebuild_required']
        or any(item['roll_outliers_removed'] > 0
               for item in report['tickers'].values())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    print(f"{'ticker':<6} {'legacy mix':>10} {'clean mix':>10} "
          f"{'roll fixed':>10} {'old win128':>12} {'new win128':>12}")
    for ticker, item in report['tickers'].items():
        print(f"{ticker:<6} {item['legacy_mixed_range_bars']:>10} "
              f"{item['clean_mixed_range_bars']:>10} "
              f"{item['roll_outliers_removed']:>10} "
              f"{item['legacy_encoder_windows_tainted_128_est']:>12} "
              f"{item['clean_encoder_windows_tainted_128_est']:>12}")
    if report['missing']:
        print('MISSING:', *report['missing'], sep='\n  ')
    print(f"report: {args.output}")
    if report['production_revalidation_required']:
        print('AUDIT: REVALIDATION REQUIRED (corrected bars/cache hashes not yet sealed)')
        return 2
    print('AUDIT: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
