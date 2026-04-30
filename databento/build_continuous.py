"""
Build continuous OHLCV CSVs from DataBento files.

Accepts two formats in the databento/ folder:
  - <TICKER>-glbx-mdp3-....ohlcv-1m.csv.zst   (one ticker per file)
  - <any-name>.dbn.zst                          (multi-ticker DBN binary)

Deduplicates roll-overlap bars, resamples to RESAMPLE_PERIOD, writes to data/.

Valid values: '1min', '3min', '5min', '15min', '30min', '1h', '4h', '1D'

DataBento roll handling:
  - Spread rows (symbol contains '-', e.g. NQM1-NQU1) carry a price diff,
    not a real bar — dropped.
  - When two contracts trade the same minute during a roll, keep the row
    with the highest volume (front month).

Output format: datetime, open, high, low, close, volume
  Compatible with prepare_data.py and all Colab fine-tuning scripts.

Usage:
    pip install zstandard databento   # one-time
    python databento/build_continuous.py 1min
    python databento/build_continuous.py 5min
    python databento/build_continuous.py        # defaults to 5min
"""

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
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def ticker_from_path(path: Path) -> str:
    """Extract ticker from 'NQ-glbx-mdp3-...' filename."""
    raw = path.name.split('-')[0].upper()
    return TICKER_REMAP.get(raw, raw)


def root_from_symbol(symbol: str) -> str:
    """Extract root ticker from a futures symbol, e.g. 'RTYM1' → 'RTY', 'YMM1' → 'YM'."""
    # Strip trailing month-code (single letter) + year digit(s)
    match = re.match(r'^([A-Z]+)(?=[FGHJKMNQUVXZ]\d)', symbol)
    root = match.group(1) if match else symbol
    return TICKER_REMAP.get(root, root)


def _clean_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Drop spreads, bad prices, deduplicate, return indexed 1-min DataFrame."""
    spread_mask = df['symbol'].str.contains('-', na=False)
    if spread_mask.sum():
        print(f'    Dropped {spread_mask.sum():,} spread rows')
    df = df[~spread_mask].copy()

    bad_price = df['close'] <= 0
    if bad_price.sum():
        print(f'    Dropped {bad_price.sum():,} non-positive price rows')
    df = df[~bad_price].copy()

    before = len(df)
    df = df.sort_values('volume', ascending=False)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    dupes_removed = before - len(df)
    if dupes_removed:
        print(f'    Deduplicated {dupes_removed:,} back-month overlap rows')

    df = df[['open', 'high', 'low', 'close', 'volume']]
    print(f'    {len(df):,} 1-min bars  | {df.index[0]}  →  {df.index[-1]}')
    return df


def load_csv_zst(path: Path) -> pd.DataFrame:
    """Read a single-ticker DataBento .csv.zst file → clean 1-min DataFrame."""
    print(f'  Reading {path.name} ...')
    df = pd.read_csv(path, compression='zstd')
    df['datetime'] = pd.to_datetime(df['ts_event'], utc=True)
    df = df.set_index('datetime')
    return _clean_ohlcv(df, ticker_from_path(path))


def load_dbn_zst(path: Path) -> dict[str, pd.DataFrame]:
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
        clean = _clean_ohlcv(sub, root)
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    try:
        import zstandard  # noqa: F401
    except ImportError:
        print('ERROR: zstandard not installed.  Run:  pip install zstandard')
        sys.exit(1)

    csv_files = sorted(DATABENTO_DIR.glob('*.csv.zst'))
    dbn_files = sorted(DATABENTO_DIR.glob('*.dbn.zst'))

    if not csv_files and not dbn_files:
        print(f'No .csv.zst or .dbn.zst files found in {DATABENTO_DIR}')
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'DataBento dir : {DATABENTO_DIR}')
    print(f'Output dir    : {OUTPUT_DIR}')
    print(f'Resample to   : {RESAMPLE_PERIOD}')
    print(f'CSV files     : {len(csv_files)}')
    print(f'DBN files     : {len(dbn_files)}\n')

    # ── CSV files (one ticker per file) ──
    for path in csv_files:
        ticker = ticker_from_path(path)
        print(f'{"="*55}')
        print(f'  {ticker}  ({path.name})')
        print(f'{"="*55}')
        try:
            df_1m = load_csv_zst(path)
        except Exception as e:
            print(f'  ERROR loading {path.name}: {e}')
            continue
        _save(df_1m, ticker)

    # ── DBN files (may contain multiple tickers) ──
    for path in dbn_files:
        print(f'{"="*55}')
        print(f'  (DBN)  {path.name}')
        print(f'{"="*55}')
        try:
            ticker_map = load_dbn_zst(path)
        except Exception as e:
            print(f'  ERROR loading {path.name}: {e}')
            continue
        for ticker, df_1m in ticker_map.items():
            _save(df_1m, ticker)

    print('Done.')


def _save(df_1m: pd.DataFrame, ticker: str) -> None:
    df_out = resample_ohlcv(df_1m, RESAMPLE_PERIOD)
    out_path = OUTPUT_DIR / f'{ticker}_{RESAMPLE_PERIOD}.csv'
    df_out.index.name = 'datetime'
    df_out.to_csv(out_path)
    print(f'  Saved: {out_path.name}  ({len(df_out):,} bars)')
    print(f'  Range: {df_out.index[0]}  →  {df_out.index[-1]}')
    print()


if __name__ == '__main__':
    main()
