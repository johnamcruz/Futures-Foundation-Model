import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from futures_foundation.primitives import apply_rr_barriers, session_end_mask


class StrategyLabeler(ABC):
    """
    Implement this to plug a new strategy into the fine-tuning framework.

    Borrow #4 (locked): a strategy authors only **what is unique to it** —
    where the events are and what its features are — and the base class owns
    the universal, bug-prone label mechanics ONCE:

      - name:            short slug used in log output
      - feature_cols:    ordered list of strategy-specific column names
      - detect_events(): where/which-direction the strategy fires
      - compute_features(): the strategy-specific feature matrix

    The concrete `run()` is FINAL — do not override it. It applies a single
    session-calibrated, TP>=SL triple-barrier with entry on the *next bar's
    open* (the entry-after-signal rule), centralising the orientation /
    entry-timing bug class that historically sank labelers.

    Output contract (unchanged for HybridStrategyModel / Dataset / borrows
    #1-3): labels_df has signal_label (binary), max_rr (realized R for the
    risk head), sl_distance, and `direction` — the last makes borrow-#1
    realized-R / borrow-#3 econ selection available *for free* to every new
    strategy. The signal head stays pure binary; R lives in the risk head.
    """

    # Bars to allow a trade to resolve before forcing exit. None => run to
    # session end / data end (the triple barrier's time arm is the session).
    barrier_timeout: int = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in log output (e.g. 'cisd_ote', 'orb')."""
        ...

    @property
    @abstractmethod
    def feature_cols(self) -> list:
        """Ordered column names of the strategy-specific feature array."""
        ...

    def config_dict(self) -> dict:
        """
        JSON-serialisable dict of every parameter that affects labeling
        output.  The framework hashes this (plus tickers + feature_cols) to
        decide whether cached parquet files are still valid.  Override in
        every concrete labeler; include the labeling version + all
        thresholds.  Do NOT include paths — only logical parameters.
        """
        return {}

    @abstractmethod
    def detect_events(
        self,
        df_raw: pd.DataFrame,
        ffm_df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """
        Return the strategy's trade events — one row per signal bar.

        Required columns:
          bar_idx      int   positional index into ffm_df (the SIGNAL bar;
                              entry is the NEXT bar's open)
          direction    int   +1 = long, -1 = short
          sl_distance  float stop distance in price units (> 0)
          tp_rr        float take-profit as a multiple of sl_distance
                              (>= 1.0 enforced => TP >= SL orientation)
        Optional:
          timeout      int   bars to resolve before forced exit
                              (defaults to self.barrier_timeout)

        Return an empty DataFrame (with these columns) for no events.
        """
        ...

    @abstractmethod
    def compute_features(
        self,
        df_raw: pd.DataFrame,
        ffm_df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """
        Return the strategy-specific feature matrix: columns == feature_cols,
        one row per ffm_df row (aligned to ffm_df.index, len == len(ffm_df)).
        """
        ...

    def run(
        self,
        df_raw: pd.DataFrame,
        ffm_df: pd.DataFrame,
        ticker: str,
    ) -> tuple:
        """
        FINAL — do not override.  Orchestrates detect_events() +
        compute_features() into the (strategy_features, labels) contract via
        a session-calibrated TP>=SL triple barrier, entry = next-bar open.
        """
        n = len(ffm_df)

        feats = self.compute_features(df_raw, ffm_df, ticker)
        events = self.detect_events(df_raw, ffm_df, ticker)

        required = {'bar_idx', 'direction', 'sl_distance', 'tp_rr'}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(
                f"{self.name}.detect_events() missing columns: {sorted(missing)}")

        ohlc = df_raw[['open', 'high', 'low', 'close']].reindex(ffm_df.index)
        o = ohlc['open'].to_numpy(float)
        h = ohlc['high'].to_numpy(float)
        l = ohlc['low'].to_numpy(float)
        c = ohlc['close'].to_numpy(float)
        sess_end = session_end_mask(ffm_df.index)

        signal_label = np.zeros(n, dtype=np.int8)
        max_rr       = np.zeros(n, dtype=np.float32)
        sl_distance  = np.zeros(n, dtype=np.float32)
        direction    = np.zeros(n, dtype=np.int8)

        _clamped = False
        for ev in events.itertuples(index=False):
            bi  = int(ev.bar_idx)
            d   = int(ev.direction)
            sld = float(ev.sl_distance)
            rr  = float(ev.tp_rr)
            tmo = int(ev.timeout) if ('timeout' in events.columns
                                      and pd.notna(ev.timeout)) \
                else self.barrier_timeout

            if rr < 1.0:                       # enforce TP >= SL orientation
                rr = 1.0
                _clamped = True
            if bi < 0 or bi >= n or sld <= 0 or not np.isfinite(sld):
                continue
            entry_idx = bi + 1                 # entry = next-bar open
            if entry_idx >= n:
                continue
            entry = o[entry_idx]
            if not np.isfinite(entry):
                continue
            is_long  = d > 0
            sl_price = entry - sld if is_long else entry + sld

            res = apply_rr_barriers(
                h, l, c, entry_idx=entry_idx, is_long=is_long,
                entry_price=entry, sl_price=sl_price, rr_targets=[rr],
                lookahead=tmo, is_session_end=sess_end)[rr]

            signal_label[bi] = 1 if res['hit'] else 0
            max_rr[bi]       = float(res['realized_rr'])   # realized R
            sl_distance[bi]  = sld
            direction[bi]    = d

        if _clamped:
            warnings.warn(
                f"{self.name}: tp_rr < 1.0 clamped to 1.0 "
                f"(TP must be >= SL — triple-barrier orientation rule)",
                stacklevel=2)

        labels_df = pd.DataFrame({
            'signal_label': signal_label,
            'max_rr':       max_rr,
            'sl_distance':  sl_distance,
            'direction':    direction,
        })
        return feats, labels_df
