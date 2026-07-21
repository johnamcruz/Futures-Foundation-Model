"""SSL data assembly — torch-free, raw OHLCV from the data/ CSVs.

Loads raw OHLCV bars for the 9 futures tickers x {1,3,5,15}min from a configurable
directory (local `data/` or a Colab Google-Drive mount), and yields the leak-safe
time split the contrastive pretraining trains on:

  * per-stream OHLCV arrays (float32 [N, 5] = open/high/low/close/volume) + timestamps
  * a strictly causal TRAIN / VAL split (VAL = the last `val_frac` of each stream's
    PRE-HOLDOUT timeline) so the val NT-Xent early-stop measures generalization
    forward in time — the SSL analogue of the WF VAL->TEST gap
  * the 2026 HOLDOUT is EXCLUDED entirely (never seen by SSL) so the downstream
    classifier's 2026 OOS stays uncontaminated by backbone adaptation

No torch here (testable without the GPU stack). The torch trainer (_ssl_torch)
consumes these arrays + the window-start indices.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

TICKERS_9 = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']
TFS_ALL = ['1min', '3min', '5min', '15min']
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']

SAMPLING_MODES = ('bar_proportional', 'uniform_stream')


@dataclass(frozen=True)
class WindowStartPool:
    """Training-window starts with their source-stream membership.

    ``uniform_stream`` follows Chronos's data-mixture semantics: choose a source
    using explicit source probabilities, then draw an example from that source.
    The compact group vector is retained instead of duplicating starts from short
    streams or materializing one floating-point weight per window on the accelerator.

    ``np.asarray(pool)`` intentionally returns the ordinary start array so existing
    leak guards and pretext-specific anchor filters remain backward compatible.
    Validation never uses this pool: it remains chronological and bar-proportional.
    """

    starts: np.ndarray
    group_ids: np.ndarray
    stream_ids: tuple[str, ...]
    sampling_mode: str = 'uniform_stream'

    def __post_init__(self):
        starts = np.asarray(self.starts, dtype=np.int64)
        groups = np.asarray(self.group_ids, dtype=np.int32)
        if starts.ndim != 1 or groups.ndim != 1 or len(starts) != len(groups):
            raise ValueError('window starts and group ids must be aligned 1-D arrays')
        if self.sampling_mode not in SAMPLING_MODES:
            raise ValueError(f'unsupported sampling_mode={self.sampling_mode!r}')
        if len(starts) and (groups.min() < 0 or groups.max() >= len(self.stream_ids)):
            raise ValueError('group ids must index stream_ids')
        object.__setattr__(self, 'starts', starts)
        object.__setattr__(self, 'group_ids', groups)

    def __len__(self):
        return len(self.starts)

    def __array__(self, dtype=None, copy=None):
        out = self.starts if dtype is None else self.starts.astype(dtype, copy=False)
        return out.copy() if copy else out

    def group_counts(self) -> dict[str, int]:
        """Number of legal training windows in each non-empty stream."""
        counts = np.bincount(self.group_ids, minlength=len(self.stream_ids))
        return {sid: int(counts[i]) for i, sid in enumerate(self.stream_ids)}

    def group_probabilities(self) -> dict[str, float]:
        """Configured source probability, independent of source length."""
        if not self.stream_ids:
            return {}
        p = 1.0 / len(self.stream_ids)
        return {sid: p for sid in self.stream_ids}

_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def load_ohlcv(data_dir=None, tickers=None, tfs=None, verbose=True):
    """Return a list of stream dicts {sid, ticker, tf, ohlcv[N,5], ts[N]} for every
    (ticker, tf) CSV found under `data_dir` (default repo data/). Missing files are
    skipped with a note (not all tickers have every TF historically)."""
    ddir = data_dir or _DATA
    tickers = tickers or TICKERS_9
    tfs = tfs or TFS_ALL
    streams = []
    for tk in tickers:
        for tf in tfs:
            path = os.path.join(ddir, f'{tk}_{tf}.csv')
            if not os.path.exists(path):
                if verbose:
                    print(f"  [ssl-data] skip (missing) {tk}_{tf}", flush=True)
                continue
            df = pd.read_csv(path, usecols=['datetime'] + OHLCV_COLS)
            df['ts'] = pd.to_datetime(df['datetime'], utc=True)
            df = df.sort_values('ts').reset_index(drop=True)
            ohlcv = df[OHLCV_COLS].to_numpy(np.float32)
            ts = df['ts'].to_numpy()
            streams.append({'sid': f'{tk}@{tf}', 'ticker': tk, 'tf': tf,
                            'ohlcv': ohlcv, 'ts': ts})
            if verbose:
                print(f"  [ssl-data] {tk}_{tf} bars={len(df)}", flush=True)
    if not streams:
        raise FileNotFoundError(f"no OHLCV CSVs found under {ddir} for "
                                f"tickers={tickers} tfs={tfs}")
    return streams


def time_split(ts, val_frac=0.1, holdout_start='2026-01-01'):
    """Strictly causal split of one stream's timestamps into (train_idx, val_idx).

    HOLDOUT (>= holdout_start, default 2026) is excluded from BOTH so the backbone
    never sees it. VAL = the last `val_frac` of the remaining (pre-holdout) bars;
    TRAIN = everything before VAL. Returns (train_idx, val_idx) int arrays into ts.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    n = len(ts)
    usable = np.arange(n)
    if holdout_start is not None:
        cut = pd.Timestamp(holdout_start, tz='UTC')
        usable = usable[ts[usable] < cut]
    if len(usable) == 0:
        return np.array([], int), np.array([], int)
    n_val = int(len(usable) * val_frac)
    if n_val == 0:
        return usable, np.array([], int)
    return usable[:-n_val], usable[-n_val:]


def window_starts(idx, seq_total, contiguous=True):
    """Valid window-start positions within an index range such that
    [start, start+seq_total) stays inside `idx`. With contiguous=True (default) the
    full window must be a run of consecutive bar indices (no split/holdout gap inside
    the window). Returns an int array of start positions (absolute bar indices)."""
    idx = np.asarray(idx, int)
    if len(idx) < seq_total:
        return np.array([], int)
    if not contiguous:
        return idx[:len(idx) - seq_total + 1]
    # keep starts whose next seq_total-1 indices are consecutive (idx[i+k] == idx[i]+k)
    starts = idx[:len(idx) - seq_total + 1]
    ahead = idx[seq_total - 1:]                       # idx shifted by seq_total-1
    return starts[(ahead - starts) == (seq_total - 1)]
