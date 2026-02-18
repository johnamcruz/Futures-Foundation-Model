"""
Hybrid ORB Dataset
==================

Sliding window dataset that returns both FFM features (seq_len, 42)
and ORB features (20,) at the last bar of each window.

Usage:
    from strategies.orb import HybridORBDataset

    ds = HybridORBDataset(features_df, orb_features_df, orb_labels_df, seq_len=64)
    sample = ds[0]
    # sample["features"]      → (64, 42) FFM features
    # sample["orb_features"]  → (20,)    ORB features at last bar
    # sample["signal_label"]  → scalar   0/1/2
    # sample["max_rr"]        → scalar   float
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from futures_foundation import get_model_feature_columns

from .features import ORB_FEATURE_COLS


class HybridORBDataset(Dataset):
    """
    Sliding window dataset for hybrid ORB fine-tuning.

    Each sample provides:
      - features: (seq_len, 42) FFM backbone input sequence
      - orb_features: (20,) ORB-specific features at the LAST bar
      - signal_label: 0=HOLD, 1=BUY, 2=SELL at last bar
      - max_rr: highest R:R target hit at last bar
      - metadata: instrument_ids, session_ids, time_of_day, day_of_week
    """

    def __init__(self, features_df, orb_features_df, orb_labels_df,
                 seq_len=64, stride=1):
        self.seq_len = seq_len
        self.feature_cols = get_model_feature_columns()

        valid = features_df[self.feature_cols].notna().all(axis=1)
        self.features = features_df[valid].reset_index(drop=True)
        self.orb_feats = orb_features_df[valid].reset_index(drop=True)
        self.labels = orb_labels_df[valid].reset_index(drop=True)

        self.window_starts = list(range(0, len(self.features) - seq_len + 1, stride))

        # Pre-convert to numpy for speed
        self._f = np.nan_to_num(
            self.features[self.feature_cols].values.astype(np.float32), nan=0.0)
        self._orb = np.nan_to_num(
            self.orb_feats[ORB_FEATURE_COLS].values.astype(np.float32), nan=0.0)
        self._inst = self.features.get(
            "_instrument_id", pd.Series(0, index=self.features.index)).values.astype(np.int64)
        self._sess = self.features.get(
            "sess_id", pd.Series(0, index=self.features.index)).values.astype(np.int64)
        self._tod = self.features.get(
            "sess_time_of_day", pd.Series(0.0, index=self.features.index)).values.astype(np.float32)
        self._dow = self.features.get(
            "tmp_day_of_week", pd.Series(0, index=self.features.index)).values.astype(np.int64)

        self._sig = self.labels['signal_label'].values.astype(np.int64)
        self._rr = self.labels['max_rr'].values.astype(np.float32)
        self.n_buy = (self._sig == 1).sum()
        self.n_sell = (self._sig == 2).sum()

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx):
        s = self.window_starts[idx]
        e = s + self.seq_len
        last = e - 1
        return {
            "features": torch.from_numpy(self._f[s:e]),
            "orb_features": torch.from_numpy(self._orb[last]),
            "instrument_ids": torch.tensor(self._inst[s], dtype=torch.long),
            "session_ids": torch.from_numpy(self._sess[s:e]),
            "time_of_day": torch.from_numpy(self._tod[s:e]),
            "day_of_week": torch.from_numpy(self._dow[s:e]),
            "signal_label": torch.tensor(self._sig[last], dtype=torch.long),
            "max_rr": torch.tensor(self._rr[last], dtype=torch.float32),
        }
