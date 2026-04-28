"""
Build continuous OHLCV CSVs from DataBento .csv.zst files.

Reads every .csv.zst file in this folder, deduplicates roll-overlap bars,
resamples to RESAMPLE_PERIOD, and writes to data/.

To change timeframe: edit RESAMPLE_PERIOD below — that's the only line you need.
Valid values: '1min', '3min', '5min', '15min', '30min', '1h', '4h', '1D'

DataBento roll handling:
  - Spread rows (symbol contains '-', e.g. NQM1-NQU1) carry a price diff,
    not a real bar — dropped.
  - When two contracts trade the same minute during a roll, keep the row
    with the highest volume (front month).

Output format: datetime, open, high, low, close, volume
  Compatible with prepare_data.py and all Colab fine-tuning scripts.

Usage:
    pip install zstandard          # one-time
    python databento/build_continuous.py
"""

import sys
from pathlib import Path

import pandas as pd

# ── Configuration — only edit this section ─────────────────────────────────────

RESAMPLE_PERIOD = '5min'   # ← change this to switch timeframe

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


def load_databento(path: Path) -> pd.DataFrame:
    """
    Read a DataBento OHLCV .csv.zst file and return a clean 1-min DataFrame.

    Steps:
      1. Read compressed CSV — pandas handles .zst natively with zstandard installed.
      2. Drop spread rows (symbol like 'NQM1-NQU1') — these carry a price diff,
         not a real bar.
      3. Drop rows with non-positive prices (a second guard against bad spread rows).
      4. For any remaining duplicate timestamps (back month also trading),
         keep the highest-volume row — that is always the front month.
      5. Parse ts_event → tz-aware UTC DatetimeIndex.
    """
    print(f'  Reading {path.name} ...')
    df = pd.read_csv(path, compression='zstd')

    # Drop roll-spread rows  (symbol contains a dash, e.g. NQM1-NQU1)
    spread_mask = df['symbol'].str.contains('-', na=False)
    if spread_mask.sum():
        print(f'    Dropped {spread_mask.sum():,} spread rows')
    df = df[~spread_mask].copy()

    # Drop rows with non-positive prices (safety net)
    bad_price = df['close'] <= 0
    if bad_price.sum():
        print(f'    Dropped {bad_price.sum():,} non-positive price rows')
    df = df[~bad_price].copy()

    # Deduplicate: keep highest-volume row per timestamp (front month)
    before = len(df)
    df = (df.sort_values('volume', ascending=False)
            .drop_duplicates(subset='ts_event', keep='first')
            .sort_values('ts_event'))
    dupes_removed = before - len(df)
    if dupes_removed:
        print(f'    Deduplicated {dupes_removed:,} back-month overlap rows')

    # Parse timestamp
    df['datetime'] = pd.to_datetime(df['ts_event'], utc=True)
    df = df.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']]

    print(f'    {len(df):,} 1-min bars  '
          f'| {df.index[0]}  →  {df.index[-1]}')
    return df


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

    zst_files = sorted(DATABENTO_DIR.glob('*.csv.zst'))
    if not zst_files:
        print(f'No .csv.zst files found in {DATABENTO_DIR}')
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'DataBento dir : {DATABENTO_DIR}')
    print(f'Output dir    : {OUTPUT_DIR}')
    print(f'Resample to   : {RESAMPLE_PERIOD}')
    print(f'Files found   : {len(zst_files)}\n')

    for path in zst_files:
        ticker = ticker_from_path(path)
        print(f'{"="*55}')
        print(f'  {ticker}  ({path.name})')
        print(f'{"="*55}')

        try:
            df_1m = load_databento(path)
        except Exception as e:
            print(f'  ERROR loading {path.name}: {e}')
            continue

        df_out = resample_ohlcv(df_1m, RESAMPLE_PERIOD)
        out_path = OUTPUT_DIR / f'{ticker}_{RESAMPLE_PERIOD}.csv'

        df_out.index.name = 'datetime'
        df_out.to_csv(out_path)

        print(f'  Saved: {out_path.name}  ({len(df_out):,} bars)')
        print(f'  Range: {df_out.index[0]}  →  {df_out.index[-1]}')
        print()

    print('Done.')


if __name__ == '__main__':
    main()
