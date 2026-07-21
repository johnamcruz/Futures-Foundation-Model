"""
Build continuous OHLCV CSVs from DataBento files.

Accepts two formats in the databento/ folder:
  - <TICKER>-glbx-mdp3-....ohlcv-1m.csv.zst   (one ticker per file)
  - <any-name>.dbn.zst                          (multi-ticker DBN binary)

Selects one liquid outright contract per CME session, resamples to
RESAMPLE_PERIOD, and writes to data/.  ``--back-adjust`` produces a replay-safe
continuous series whose historical contracts are translated onto the newest
contract's price basis.

When extending a back-adjusted history, pass the old and new archives together
with ``--merge-sources``.  The builder deduplicates their raw symbol bars before
contract selection and back-adjustment; splicing two already-adjusted outputs
would mix incompatible price bases.

Valid values: '1min', '3min', '5min', '15min', '30min', '1h', '4h', '1D'

DataBento roll handling:
  - Spread rows (symbol contains '-', e.g. NQM1-NQU1) carry a price diff,
    not a real bar — dropped.
  - Contract selection is made from TOTAL CME-session volume, never minute by
    minute.  Minute-level selection can alternate old/new contracts inside one
    candle and manufacture impossible 200+ point ranges.
  - Optional back-adjustment uses the median new-minus-old close spread over
    their real overlap.  This prevents a simulated position from booking the
    contract price-basis difference as P&L at the handoff.

Output format: datetime, open, high, low, close, volume
  Compatible with prepare_data.py and all Colab fine-tuning scripts.

Usage:
    pip install zstandard databento   # one-time
    python databento/build_continuous.py 1min
    python databento/build_continuous.py 5min
    python databento/build_continuous.py 3min --back-adjust \
        --output-dir data
    python databento/build_continuous.py --periods 1min,3min,5min,15min \
        --back-adjust --output-dir data
    python databento/build_continuous.py        # defaults to 5min
"""

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ───────────────────────────────────────────────────────────────

_DEFAULT_PERIOD = '5min'
RESAMPLE_PERIOD = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_PERIOD

DATABENTO_DIR = Path(__file__).parent
OUTPUT_DIR    = Path(__file__).parent.parent / 'data'

# DataBento uses 'SIL' for Silver; FFM / INSTRUMENT_MAP uses 'SI'.
# Add any other mismatches here.
TICKER_REMAP = {
    'SIL': 'SI',
    'MCL': 'CL',
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def ticker_from_path(path: Path) -> str:
    """Extract ticker from 'NQ-glbx-mdp3-...' filename."""
    raw = path.name.split('-')[0].upper()
    return TICKER_REMAP.get(raw, raw)


def root_from_symbol(symbol: str) -> str | None:
    """Extract root ticker from a futures symbol, e.g. 'RTYM1' → 'RTY', 'YMM1' → 'YM'.
    Returns None for OTC/strategy instruments like 'UD:ZB: TL ...'."""
    if ':' in symbol:  # OTC/inter-commodity spread instruments — not outright futures
        return None
    # Strip trailing month-code (single letter) + year digit(s)
    match = re.match(r'^([A-Z]+)(?=[FGHJKMNQUVXZ]\d)', symbol)
    root = match.group(1) if match else symbol
    return TICKER_REMAP.get(root, root)


_PRICE_COLUMNS = ('open', 'high', 'low', 'close')
_VALUE_COLUMNS = (*_PRICE_COLUMNS, 'volume')
_REQUIRED_RAW_COLUMNS = ('symbol', *_VALUE_COLUMNS)


def _validate_raw_ohlcv(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Validate vendor bars before contract selection; never repair silently."""
    missing = sorted(set(_REQUIRED_RAW_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f'{context}: missing required columns {missing}')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f'{context}: index must be a DatetimeIndex')
    if df.index.hasnans:
        raise ValueError(f'{context}: contains invalid timestamps')
    if df.index.tz is None:
        raise ValueError(f'{context}: timestamps must be timezone-aware UTC')

    out = df.copy()
    out.index = out.index.tz_convert('UTC')
    if out['symbol'].isna().any() or (out['symbol'].astype(str).str.len() == 0).any():
        raise ValueError(f'{context}: contains missing symbols')
    for column in _VALUE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors='coerce')
    finite = np.isfinite(out[list(_VALUE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    if not finite.all():
        raise ValueError(f'{context}: contains {int((~finite).sum())} non-finite OHLCV rows')
    return out.sort_index(kind='stable')


def _validate_continuous_ohlcv(df: pd.DataFrame, context: str) -> None:
    """Fail closed on malformed or ambiguous model-input bars."""
    if df.empty:
        raise ValueError(f'{context}: no bars')
    missing = sorted(set(_VALUE_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f'{context}: missing required columns {missing}')
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError(f'{context}: timestamps must be timezone-aware')
    if not df.index.is_monotonic_increasing:
        raise ValueError(f'{context}: timestamps are not sorted')
    if df.index.has_duplicates:
        raise ValueError(f'{context}: contains duplicate timestamps')
    values = df[list(_VALUE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'{context}: contains non-finite OHLCV values')
    invalid = (
        (df['open'] <= 0) | (df['high'] <= 0) |
        (df['low'] <= 0) | (df['close'] <= 0) |
        (df['volume'] < 0) |
        (df['high'] < df[['open', 'close', 'low']].max(axis=1)) |
        (df['low'] > df[['open', 'close']].min(axis=1))
    )
    if invalid.any():
        first = df.index[invalid][0]
        raise ValueError(
            f'{context}: {int(invalid.sum())} invalid OHLCV rows; first at {first}')


def _cme_session_keys(index: pd.DatetimeIndex) -> pd.Index:
    """Session key with the CME trading day rolling at 17:00 Chicago time."""
    if index.tz is None:
        index = index.tz_localize('UTC')
    local = index.tz_convert('America/Chicago')
    return pd.Index((local - pd.Timedelta(hours=17)).date, name='_session')


def _session_dominant_contract(df: pd.DataFrame) -> pd.Series:
    """Return ``session -> symbol`` using total outright session volume.

    Selecting the highest-volume row independently at each minute is invalid
    during a roll: sparse bars from both contracts can alternate and later
    resampling combines their two price levels into one impossible candle.
    """
    work = df[['symbol', 'volume']].copy()
    work['_session'] = _cme_session_keys(work.index).to_numpy()
    totals = work.groupby(['_session', 'symbol'], sort=True)['volume'].sum()
    dominant_pairs = totals.groupby(level=0, sort=True).idxmax()
    return dominant_pairs.map(lambda pair: pair[1]).rename('symbol')


def _overlap_spread(raw: pd.DataFrame, old_symbol: str, new_symbol: str,
                    switch_ts: pd.Timestamp,
                    lookback_days: int = 10) -> float:
    """Robust new-minus-old close spread around a contract handoff."""
    lo = switch_ts - pd.Timedelta(days=lookback_days)
    sample = raw.loc[(raw.index >= lo) & (raw.index <= switch_ts)]
    sample = sample[sample['symbol'].isin([old_symbol, new_symbol])]
    if not sample.empty:
        paired = (sample.reset_index()
                  .pivot_table(index=sample.index.name or 'datetime',
                               columns='symbol', values='close', aggfunc='last'))
        if old_symbol in paired and new_symbol in paired:
            spread = (paired[new_symbol] - paired[old_symbol]).dropna()
            if len(spread) >= 5:
                return float(spread.median())
    raise ValueError(
        f'cannot safely estimate roll spread {old_symbol}->{new_symbol} at '
        f'{switch_ts}: fewer than 5 simultaneous close observations in the '
        f'{lookback_days}-day overlap; refusing a last-old/first-new estimate '
        f'that could encode a real market move as a roll adjustment')


def _back_adjust_selected(selected: pd.DataFrame, raw: pd.DataFrame,
                          dominant: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    """Translate prior contract segments onto the newest contract basis."""
    out = selected.copy()
    adjustments = []
    sessions = dominant.sort_index()
    previous = None
    for session, symbol in sessions.items():
        if previous is None:
            previous = symbol
            continue
        if symbol == previous:
            continue
        switch_rows = out.loc[out['_session'] == session]
        if switch_rows.empty:
            previous = symbol
            continue
        switch_ts = switch_rows.index.min()
        spread = _overlap_spread(raw, previous, symbol, switch_ts)
        for column in _PRICE_COLUMNS:
            out.loc[out.index < switch_ts, column] += spread
        adjustments.append({
            'old_symbol': previous, 'new_symbol': symbol,
            'switch_timestamp': switch_ts.isoformat(), 'spread': spread,
        })
        print(f'    Back-adjusted {previous} -> {symbol} at {switch_ts} '
              f'by {spread:+.4f}')
        previous = symbol
    return out, adjustments


def _clean_ohlcv(df: pd.DataFrame, ticker: str,
                 back_adjust: bool = False) -> pd.DataFrame | None:
    """Drop invalid instruments and build one-contract-per-session 1m bars."""
    df = _validate_raw_ohlcv(df, f'{ticker} raw source')
    spread_mask = df['symbol'].str.contains('-', na=False)
    if spread_mask.sum():
        print(f'    Dropped {spread_mask.sum():,} spread rows')
    df = df[~spread_mask].copy()

    invalid = (
        (df['open'] <= 0) | (df['high'] <= 0) |
        (df['low'] <= 0) | (df['close'] <= 0) |
        (df['volume'] < 0) |
        (df['high'] < df[['open', 'close']].max(axis=1)) |
        (df['low'] > df[['open', 'close']].min(axis=1))
    )
    if invalid.any():
        first = df.index[invalid][0]
        raise ValueError(
            f'{ticker}: {int(invalid.sum())} invalid outright OHLCV rows; '
            f'first at {first}')

    if df.empty:
        print(f'    No valid bars remaining — skipping')
        return None

    raw = df.sort_index().copy()
    dominant = _session_dominant_contract(raw)
    sequence = dominant[dominant.ne(dominant.shift())]
    repeated = sequence[sequence.duplicated(keep=False)]
    if not repeated.empty:
        raise ValueError(
            f'{ticker}: dominant contract roll flip-flop detected: '
            f'{repeated.to_dict()}')
    df['_session'] = _cme_session_keys(df.index).to_numpy()
    df['_dominant'] = df['_session'].map(dominant)
    before = len(df)
    df = df[df['symbol'] == df['_dominant']].copy()
    removed = before - len(df)
    print(f'    Selected one session-dominant contract; dropped '
          f'{removed:,} overlap rows')

    # Defensive only: a symbol normally has one OHLCV row per minute.  If the
    # archive repeats one, retain its highest-volume record deterministically.
    before = len(df)
    df = df.sort_values('volume', ascending=False)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    if before != len(df):
        print(f'    Deduplicated {before - len(df):,} same-contract rows')

    adjustments = []
    if back_adjust:
        df, adjustments = _back_adjust_selected(df, raw, dominant)

    metadata = {
        'schema': 'ffm_continuous_contract_v1',
        'selection': 'cme_session_total_volume',
        'session_boundary': '17:00 America/Chicago',
        'back_adjusted': bool(back_adjust),
        'raw_outright_rows': int(len(raw)),
        'overlap_rows_dropped': int(removed),
        'dominant_sessions': int(len(dominant)),
        'roll_adjustments': adjustments,
    }
    df = df[['open', 'high', 'low', 'close', 'volume']]
    _validate_continuous_ohlcv(df, f'{ticker} selected 1min')
    df.attrs['continuous_contract'] = metadata
    print(f'    {len(df):,} 1-min bars  | {df.index[0]}  →  {df.index[-1]}')
    return df


def load_csv_zst_raw(path: Path) -> pd.DataFrame:
    """Decode a single-ticker CSV archive without selecting contracts."""
    print(f'  Reading {path.name} ...')
    df = pd.read_csv(path, compression='zstd')
    df['datetime'] = pd.to_datetime(df['ts_event'], utc=True)
    return df.set_index('datetime')


def load_csv_zst(path: Path, back_adjust: bool = False) -> pd.DataFrame:
    """Read a single-ticker DataBento .csv.zst file → clean 1-min DataFrame."""
    df = load_csv_zst_raw(path)
    clean = _clean_ohlcv(df, ticker_from_path(path), back_adjust=back_adjust)
    if clean is not None:
        clean.attrs['source_path'] = str(path.resolve())
        clean.attrs['source_sha256'] = _sha256(path)
    return clean


def load_dbn_zst_raw(path: Path) -> dict[str, pd.DataFrame]:
    """Decode a multi-ticker DBN archive without selecting contracts."""
    try:
        import databento as db
    except ImportError:
        print('ERROR: databento not installed.  Run:  pip install databento')
        sys.exit(1)

    print(f'  Reading {path.name} (DBN format) ...')
    if not hasattr(db, 'DBNStore'):
        raise RuntimeError(
            'the local databento/ directory shadowed the Databento SDK; run '
            'this script directly with an environment that has databento installed')
    store = db.DBNStore.from_file(str(path))
    metadata = store.metadata
    dataset = str(getattr(metadata, 'dataset', ''))
    schema = getattr(metadata, 'schema', None)
    schema_value = str(getattr(schema, 'value', schema)).lower()
    if dataset != 'GLBX.MDP3':
        raise ValueError(f'{path.name}: unexpected DBN dataset {dataset!r}')
    if schema_value != 'ohlcv-1m':
        raise ValueError(f'{path.name}: expected ohlcv-1m DBN, got {schema_value!r}')
    start = pd.Timestamp(metadata.start, unit='ns', tz='UTC')
    end_value = getattr(metadata, 'end', None)
    end = (pd.Timestamp(end_value, unit='ns', tz='UTC')
           if end_value is not None else 'open')
    print(f'    DBN metadata: dataset={dataset} schema={schema_value} '
          f'start={start} end={end}')
    df = store.to_df()

    # ts_event is the index after to_df(); reset so we can manipulate it
    df = df.reset_index()
    df = df.rename(columns={'ts_event': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime')

    # Split by instrument root
    if 'symbol' not in df:
        raise ValueError(f'{path.name}: DBN symbology mapping did not yield symbols')
    df['_root'] = df['symbol'].apply(
        lambda s: root_from_symbol(s) if '-' not in s else None
    )
    roots = [r for r in df['_root'].dropna().unique()]
    print(f'    Instruments found: {sorted(roots)}')

    result = {}
    for root in sorted(roots):
        sub = df[df['_root'] == root].copy()
        result[root] = sub
    return result


def load_dbn_zst(path: Path, back_adjust: bool = False) -> dict[str, pd.DataFrame]:
    """
    Read a multi-ticker DataBento .dbn.zst file.
    Returns {ticker: clean_1min_df} for every instrument root found.
    Requires: pip install databento
    """
    raw_map = load_dbn_zst_raw(path)
    result = {}
    for root, sub in raw_map.items():
        print(f'  Processing {root} ({len(sub):,} raw rows) ...')
        clean = _clean_ohlcv(sub, root, back_adjust=back_adjust)
        if clean is not None:
            clean.attrs['source_path'] = str(path.resolve())
            clean.attrs['source_sha256'] = _sha256(path)
            result[root] = clean

    return result


def merge_raw_sources(source_maps: list[tuple[Path, dict[str, pd.DataFrame]]],
                      wanted: set[str] | None = None,
                      allow_source_corrections: bool = False,
                      ) -> dict[str, pd.DataFrame]:
    """Merge overlapping raw archives, preferring the later source argument.

    Duplicate identity is ``(timestamp, symbol)``.  Keeping symbol in the key
    preserves simultaneous old/new contracts needed to estimate roll spreads.
    """
    wanted = wanted or set()
    by_ticker: dict[str, list[pd.DataFrame]] = {}
    provenance: dict[str, list[Path]] = {}
    for path, ticker_map in source_maps:
        for ticker, frame in ticker_map.items():
            if wanted and ticker not in wanted:
                continue
            by_ticker.setdefault(ticker, []).append(frame)
            provenance.setdefault(ticker, []).append(path)

    merged = {}
    for ticker, frames in by_ticker.items():
        checked = [
            _validate_raw_ohlcv(frame, f'{ticker} source {index + 1}')
            for index, frame in enumerate(frames)
        ]
        raw = pd.concat(checked, axis=0)
        before = len(raw)
        reset = raw.reset_index()
        keys = ['datetime', 'symbol']
        overlap = reset[reset.duplicated(keys, keep=False)]
        conflict_count = 0
        if not overlap.empty:
            distinct = overlap.groupby(keys, sort=False)[list(_VALUE_COLUMNS)].nunique(
                dropna=False)
            conflicts = distinct.gt(1).any(axis=1)
            conflict_count = int(conflicts.sum())
        if conflict_count and not allow_source_corrections:
            examples = [
                f'{timestamp.isoformat()} {symbol}'
                for timestamp, symbol in conflicts[conflicts].index[:5]
            ]
            raise ValueError(
                f'{ticker}: {conflict_count} conflicting overlap bars across raw '
                f'sources ({", ".join(examples)}); inspect the vendor correction '
                f'and rerun with --allow-source-corrections only if intentional')
        raw = (raw.reset_index()
               .drop_duplicates(subset=['datetime', 'symbol'], keep='last')
               .set_index('datetime').sort_index())
        print(f'  Merged {ticker}: {len(frames)} source(s), {len(raw):,} raw rows '
              f'({before - len(raw):,} overlap duplicates replaced)')
        paths = provenance[ticker]
        raw.attrs['source_paths'] = [str(path.resolve()) for path in paths]
        raw.attrs['source_sha256s'] = [_sha256(path) for path in paths]
        raw.attrs['overlap_duplicate_rows'] = before - len(raw)
        raw.attrs['source_conflicts_accepted'] = conflict_count
        merged[ticker] = raw
    return merged


def resample_ohlcv(df: pd.DataFrame, period: str) -> pd.DataFrame:
    resampled = df.resample(period).agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }).dropna()
    _validate_continuous_ohlcv(resampled, f'resample {period}')
    return resampled


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('period', nargs='?', default=_DEFAULT_PERIOD)
    parser.add_argument('--periods', default='',
                        help='comma-separated periods to build from one raw '
                             'load (for example 1min,3min,5min,15min); '
                             'overrides the positional period')
    parser.add_argument('--back-adjust', action='store_true',
                        help='anchor prior contracts to the newest price basis')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--source', action='append', type=Path, default=[],
                        help='process only this archive (repeatable); default '
                             'discovers every archive in databento/')
    parser.add_argument('--merge-sources', action='store_true',
                        help='merge repeated --source archives at raw '
                             '(timestamp, symbol) level before cleaning; '
                             'required for extending back-adjusted history')
    parser.add_argument('--allow-source-corrections', action='store_true',
                        help='accept conflicting bars in overlapping raw '
                             'archives and prefer the later --source; default '
                             'fails closed for manual review')
    parser.add_argument('--tickers', default='',
                        help='comma-separated roots to write (default: all)')
    args = parser.parse_args()
    periods = ([item.strip() for item in args.periods.split(',') if item.strip()]
               if args.periods else [args.period])
    if len(periods) != len(set(periods)):
        parser.error('--periods contains duplicates')
    output_dir = args.output_dir.expanduser().resolve()
    try:
        import zstandard  # noqa: F401
    except ImportError:
        print('ERROR: zstandard not installed.  Run:  pip install zstandard')
        sys.exit(1)

    if args.source:
        sources = [path.expanduser().resolve() for path in args.source]
        missing = [path for path in sources if not path.exists()]
        if missing:
            parser.error(f'source not found: {missing[0]}')
        csv_files = sorted(path for path in sources
                           if path.name.endswith('.csv.zst'))
        dbn_files = sorted(path for path in sources
                           if path.name.endswith('.dbn.zst'))
        unsupported = [path for path in sources
                       if path not in csv_files and path not in dbn_files]
        if unsupported:
            parser.error(f'unsupported source: {unsupported[0]}')
    else:
        csv_files = sorted(DATABENTO_DIR.glob('*.csv.zst'))
        dbn_files = sorted(DATABENTO_DIR.glob('*.dbn.zst'))
    source_count = len(csv_files) + len(dbn_files)
    if args.source and source_count > 1 and not args.merge_sources:
        parser.error('multiple --source archives require --merge-sources so '
                     'an output ticker can never be overwritten by source order')
    merge_sources = args.merge_sources or (not args.source and source_count > 1)
    wanted = {item.strip().upper() for item in args.tickers.split(',')
              if item.strip()}
    written: set[str] = set()
    had_error = False
    generation_id = str(uuid.uuid4())

    if not csv_files and not dbn_files:
        print(f'No .csv.zst or .dbn.zst files found in {DATABENTO_DIR}')
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'DataBento dir : {DATABENTO_DIR}')
    print(f'Output dir    : {output_dir}')
    print(f'Resample to   : {", ".join(periods)}')
    print(f'Back-adjust   : {args.back_adjust}')
    print(f'CSV files     : {len(csv_files)}')
    print(f'DBN files     : {len(dbn_files)}\n')

    if merge_sources:
        ordered_sources = sources if args.source else [*csv_files, *dbn_files]
        source_maps = []
        for path in ordered_sources:
            if path.name.endswith('.csv.zst'):
                source_maps.append((path, {ticker_from_path(path):
                                           load_csv_zst_raw(path)}))
            else:
                source_maps.append((path, load_dbn_zst_raw(path)))
        raw_map = merge_raw_sources(
            source_maps, wanted,
            allow_source_corrections=args.allow_source_corrections)
        for ticker, raw in sorted(raw_map.items()):
            print(f'{"="*55}\n  {ticker}  (merged raw sources)\n{"="*55}')
            try:
                df_1m = _clean_ohlcv(raw, ticker, back_adjust=args.back_adjust)
            except Exception as error:
                print(f'  ERROR cleaning merged {ticker}: {error}')
                had_error = True
                continue
            if df_1m is None:
                had_error = True
                continue
            df_1m.attrs['source_paths'] = raw.attrs['source_paths']
            df_1m.attrs['source_sha256s'] = raw.attrs['source_sha256s']
            df_1m.attrs['overlap_duplicate_rows'] = \
                raw.attrs['overlap_duplicate_rows']
            df_1m.attrs['source_conflicts_accepted'] = \
                raw.attrs['source_conflicts_accepted']
            for period in periods:
                _save(df_1m, ticker, period=period, output_dir=output_dir,
                      generation_id=generation_id)
            written.add(ticker)
        missing_tickers = wanted - written
        if missing_tickers:
            print(f'ERROR: requested tickers not written: {sorted(missing_tickers)}')
            had_error = True
        if had_error:
            print('Completed with errors.')
            raise SystemExit(1)
        _write_generation_manifest(
            output_dir, written, periods, generation_id,
            allow_source_corrections=args.allow_source_corrections)
        print('Done.')
        return

    # ── CSV files (one ticker per file) ──
    for path in csv_files:
        ticker = ticker_from_path(path)
        if wanted and ticker not in wanted:
            continue
        print(f'{"="*55}')
        print(f'  {ticker}  ({path.name})')
        print(f'{"="*55}')
        try:
            df_1m = load_csv_zst(path, back_adjust=args.back_adjust)
        except Exception as e:
            print(f'  ERROR loading {path.name}: {e}')
            had_error = True
            continue
        for period in periods:
            _save(df_1m, ticker, period=period, output_dir=output_dir,
                  generation_id=generation_id)
        written.add(ticker)

    # ── DBN files (may contain multiple tickers) ──
    for path in dbn_files:
        print(f'{"="*55}')
        print(f'  (DBN)  {path.name}')
        print(f'{"="*55}')
        try:
            ticker_map = load_dbn_zst(path, back_adjust=args.back_adjust)
        except Exception as e:
            print(f'  ERROR loading {path.name}: {e}')
            had_error = True
            continue
        for ticker, df_1m in ticker_map.items():
            if wanted and ticker not in wanted:
                continue
            for period in periods:
                _save(df_1m, ticker, period=period, output_dir=output_dir,
                      generation_id=generation_id)
            written.add(ticker)

    missing_tickers = wanted - written
    if missing_tickers:
        print(f'ERROR: requested tickers not written: {sorted(missing_tickers)}')
        had_error = True
    if had_error:
        print('Completed with errors.')
        raise SystemExit(1)
    _write_generation_manifest(
        output_dir, written, periods, generation_id,
        allow_source_corrections=args.allow_source_corrections)
    print('Done.')


def _save(df_1m: pd.DataFrame, ticker: str, period: str = RESAMPLE_PERIOD,
          output_dir: Path = OUTPUT_DIR,
          generation_id: str | None = None) -> None:
    df_out = resample_ohlcv(df_1m, period)
    out_path = output_dir / f'{ticker}_{period}.csv'
    df_out.index.name = 'datetime'
    csv_tmp = out_path.with_name(f'.{out_path.name}.{uuid.uuid4().hex}.tmp')
    df_out.to_csv(csv_tmp)
    contract_meta = dict(df_1m.attrs.get('continuous_contract', {}))
    manifest = {
        **contract_meta,
        'ticker': ticker,
        'timeframe': period,
        'source_path': df_1m.attrs.get('source_path'),
        'source_sha256': df_1m.attrs.get('source_sha256'),
        'source_paths': df_1m.attrs.get('source_paths'),
        'source_sha256s': df_1m.attrs.get('source_sha256s'),
        'overlap_duplicate_rows': df_1m.attrs.get('overlap_duplicate_rows', 0),
        'source_conflicts_accepted': df_1m.attrs.get(
            'source_conflicts_accepted', 0),
        'generation_id': generation_id,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'output_path': out_path.name,
        'output_sha256': _sha256(csv_tmp),
        'rows': int(len(df_out)),
        'start': df_out.index[0].isoformat(),
        'end': df_out.index[-1].isoformat(),
    }
    manifest_path = out_path.with_suffix(out_path.suffix + '.manifest.json')
    manifest_tmp = manifest_path.with_name(
        f'.{manifest_path.name}.{uuid.uuid4().hex}.tmp')
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    os.replace(csv_tmp, out_path)
    os.replace(manifest_tmp, manifest_path)
    print(f'  Saved: {out_path.name}  ({len(df_out):,} bars)')
    print(f'  Manifest: {manifest_path.name}')
    print(f'  Range: {df_out.index[0]}  →  {df_out.index[-1]}')
    print()


def _write_generation_manifest(output_dir: Path, tickers: set[str],
                               periods: list[str], generation_id: str,
                               *, allow_source_corrections: bool) -> Path:
    """Commit marker written only after every requested stream is complete."""
    streams = {}
    for ticker in sorted(tickers):
        for period in periods:
            path = output_dir / f'{ticker}_{period}.csv'
            manifest_path = path.with_suffix(path.suffix + '.manifest.json')
            if not path.is_file() or not manifest_path.is_file():
                raise RuntimeError(f'generation missing output pair: {path}')
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('output_sha256') != _sha256(path):
                raise RuntimeError(f'generation output hash mismatch: {path}')
            streams[f'{ticker}@{period}'] = {
                'path': path.name,
                'sha256': manifest['output_sha256'],
                'manifest_sha256': _sha256(manifest_path),
            }
    payload = {
        'schema': 'ffm_continuous_generation_v1',
        'generation_id': generation_id,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'allow_source_corrections': allow_source_corrections,
        'streams': streams,
    }
    path = output_dir / 'continuous_generation.json'
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)
    print(f'Generation manifest: {path} ({len(streams)} streams)')
    return path


if __name__ == '__main__':
    main()
