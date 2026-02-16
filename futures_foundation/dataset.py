"""
PyTorch Dataset for Futures Foundation Model.

Handles sliding window sequences, temporal train/val splits, and multi-instrument
dataset combination. All temporal integrity preserved — no data leakage.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple

from .features import get_model_feature_columns


class FFMDataset(Dataset):
    """
    Sliding-window dataset for pretraining / fine-tuning.

    Each sample is a contiguous window of seq_len bars with features,
    metadata, and labels at the LAST bar of the window.
    """

    def __init__(self, features_df, labels_df=None, seq_len=64, stride=1, feature_columns=None):
        self.seq_len = seq_len
        self.feature_columns = feature_columns or get_model_feature_columns()

        valid_mask = features_df[self.feature_columns].notna().all(axis=1)
        if labels_df is not None:
            valid_mask = valid_mask & labels_df.notna().all(axis=1)

        self.features_df = features_df[valid_mask].reset_index(drop=True)
        self.labels_df = labels_df[valid_mask].reset_index(drop=True) if labels_df is not None else None

        max_start = len(self.features_df) - seq_len
        self.window_starts = list(range(0, max_start + 1, stride))

        self._feature_matrix = np.nan_to_num(
            self.features_df[self.feature_columns].values.astype(np.float32), nan=0.0
        )

        self._instrument_ids = self.features_df.get("_instrument_id", pd.Series(0, index=self.features_df.index)).values.astype(np.int64)
        self._session_ids = self.features_df.get("sess_id", pd.Series(0, index=self.features_df.index)).values.astype(np.int64)
        self._time_of_day = self.features_df.get("sess_time_of_day", pd.Series(0.0, index=self.features_df.index)).values.astype(np.float32)
        self._day_of_week = self.features_df.get("tmp_day_of_week", pd.Series(0, index=self.features_df.index)).values.astype(np.int64)

        if self.labels_df is not None:
            self._label_arrays = {col: self.labels_df[col].values.astype(np.int64) for col in self.labels_df.columns}
        else:
            self._label_arrays = None

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx):
        start = self.window_starts[idx]
        end = start + self.seq_len

        sample = {
            "features": torch.from_numpy(self._feature_matrix[start:end]),
            "instrument_ids": torch.tensor(self._instrument_ids[start], dtype=torch.long),
            "session_ids": torch.from_numpy(self._session_ids[start:end]),
            "time_of_day": torch.from_numpy(self._time_of_day[start:end]),
            "day_of_week": torch.from_numpy(self._day_of_week[start:end]),
        }

        if self._label_arrays is not None:
            last_idx = end - 1
            for col, arr in self._label_arrays.items():
                sample[col] = torch.tensor(arr[last_idx], dtype=torch.long)

        return sample


class FFMMultiInstrumentDataset(Dataset):
    """Combines multiple single-instrument datasets for cross-instrument training."""

    def __init__(self, datasets: List[FFMDataset]):
        self.datasets = datasets
        self._cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self._cumulative_sizes.append(total)

    def __len__(self):
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, idx):
        for i, cum_size in enumerate(self._cumulative_sizes):
            if idx < cum_size:
                offset = self._cumulative_sizes[i - 1] if i > 0 else 0
                return self.datasets[i][idx - offset]
        raise IndexError(f"Index {idx} out of range")


def temporal_train_val_split(features_df, labels_df, val_ratio=0.15, seq_len=64, stride_train=1, stride_val=None):
    """Split by time — last val_ratio% is validation. Never random for time series."""
    stride_val = stride_val or seq_len // 2
    split_idx = int(len(features_df) * (1 - val_ratio))

    train_ds = FFMDataset(features_df.iloc[:split_idx].copy(), labels_df.iloc[:split_idx].copy(), seq_len=seq_len, stride=stride_train)
    val_ds = FFMDataset(features_df.iloc[split_idx:].copy(), labels_df.iloc[split_idx:].copy(), seq_len=seq_len, stride=stride_val)
    return train_ds, val_ds


def create_dataloaders(train_dataset, val_dataset, batch_size=256, num_workers=4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader