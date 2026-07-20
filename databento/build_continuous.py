"""
Build continuous OHLCV CSVs from DataBento files.

Accepts two formats in the databento/ folder:
  - <TICKER>-glbx-mdp3-....ohlcv-1m.csv.zst   (one ticker per file)
  - <any-name>.dbn.zst                          (multi-ticker DBN binary)

Selects one liquid outright contract per CME session, resamples to
RESAMPLE_PERIOD, and writes to data/.  ``--back-adjust`` produces a replay-safe
continuous series whose historical contracts are translated onto the newest
contract's price basis.

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
        --output-dir data/replay
    python databento/build_continuous.py --periods 1min,3min,5min,15min \
        --back-adjust --output-dir data/replay
    python databento/build_continuous.py        # defaults to 5min
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

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
    old = raw.loc[(raw.index < switch_ts) & (raw['symbol'] == old_symbol), 'close']
    new = raw.loc[(raw.index >= switch_ts) & (raw['symbol'] == new_symbol), 'close']
    if old.empty or new.empty:
        raise ValueError(
            f'cannot estimate roll spread {old_symbol}->{new_symbol} at {switch_ts}')
    return float(new.iloc[0] - old.iloc[-1])


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
    spread_mask = df['symbol'].str.contains('-', na=False)
    if spread_mask.sum():
        print(f'    Dropped {spread_mask.sum():,} spread rows')
    df = df[~spread_mask].copy()

    bad_price = df['close'] <= 0
    if bad_price.sum():
        print(f'    Dropped {bad_price.sum():,} non-positive price rows')
    df = df[~bad_price].copy()

    if df.empty:
        print(f'    No valid bars remaining — skipping')
        return None

    raw = df.sort_index().copy()
    dominant = _session_dominant_contract(raw)
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
    df.attrs['continuous_contract'] = metadata
    print(f'    {len(df):,} 1-min bars  | {df.index[0]}  →  {df.index[-1]}')
    return df


def load_csv_zst(path: Path, back_adjust: bool = False) -> pd.DataFrame:
    """Read a single-ticker DataBento .csv.zst file → clean 1-min DataFrame."""
    print(f'  Reading {path.name} ...')
    df = pd.read_csv(path, compression='zstd')
    df['datetime'] = pd.to_datetime(df['ts_event'], utc=True)
    df = df.set_index('datetime')
    clean = _clean_ohlcv(df, ticker_from_path(path), back_adjust=back_adjust)
    if clean is not None:
        clean.attrs['source_path'] = str(path.resolve())
        clean.attrs['source_sha256'] = _sha256(path)
    return clean


def load_dbn_zst(path: Path, back_adjust: bool = False) -> dict[str, pd.DataFrame]:
    """
    Read a multi-ticker DataBento .dbn.zst file.
    Returns {ticker: clean_1min_df} for every instrument root found.
    Requires: pip install databento
    """
    try:
        import databento as db
    except ImportError:
        print('ERROR: databento not installed.  Run:  pip install databento')
        sys.exit(1)

    print(f'  Reading {path.name} (DBN format) ...')
    store = db.DBNStore.from_file(str(path))
    df = store.to_df()

    # ts_event is the index after to_df(); reset so we can manipulate it
    df = df.reset_index()
    df = df.rename(columns={'ts_event': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime')

    # Split by instrument root
    df['_root'] = df['symbol'].apply(
        lambda s: root_from_symbol(s) if '-' not in s else None
    )
    roots = [r for r in df['_root'].dropna().unique()]
    print(f'    Instruments found: {sorted(roots)}')

    result = {}
    for root in sorted(roots):
        sub = df[df['_root'] == root].copy()
        print(f'  Processing {root} ({len(sub):,} raw rows) ...')
        clean = _clean_ohlcv(sub, root, back_adjust=back_adjust)
        if clean is not None:
            clean.attrs['source_path'] = str(path.resolve())
            clean.attrs['source_sha256'] = _sha256(path)
            result[root] = clean

    return result


def resample_ohlcv(df: pd.DataFrame, period: str) -> pd.DataFrame:
    resampled = df.resample(period).agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }).dropna()
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
    wanted = {item.strip().upper() for item in args.tickers.split(',')
              if item.strip()}
    written: set[str] = set()
    had_error = False

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
            _save(df_1m, ticker, period=period, output_dir=output_dir)
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
                _save(df_1m, ticker, period=period, output_dir=output_dir)
            written.add(ticker)

    missing_tickers = wanted - written
    if missing_tickers:
        print(f'ERROR: requested tickers not written: {sorted(missing_tickers)}')
        had_error = True
    if had_error:
        print('Completed with errors.')
        raise SystemExit(1)
    print('Done.')


def _save(df_1m: pd.DataFrame, ticker: str, period: str = RESAMPLE_PERIOD,
          output_dir: Path = OUTPUT_DIR) -> None:
    df_out = resample_ohlcv(df_1m, period)
    out_path = output_dir / f'{ticker}_{period}.csv'
    df_out.index.name = 'datetime'
    df_out.to_csv(out_path)
    contract_meta = dict(df_1m.attrs.get('continuous_contract', {}))
    manifest = {
        **contract_meta,
        'ticker': ticker,
        'timeframe': period,
        'source_path': df_1m.attrs.get('source_path'),
        'source_sha256': df_1m.attrs.get('source_sha256'),
        'output_path': str(out_path.resolve()),
        'output_sha256': _sha256(out_path),
        'rows': int(len(df_out)),
        'start': df_out.index[0].isoformat(),
        'end': df_out.index[-1].isoformat(),
    }
    manifest_path = out_path.with_suffix(out_path.suffix + '.manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'  Saved: {out_path.name}  ({len(df_out):,} bars)')
    print(f'  Manifest: {manifest_path.name}')
    print(f'  Range: {df_out.index[0]}  →  {df_out.index[-1]}')
    print()


if __name__ == '__main__':
    main()
