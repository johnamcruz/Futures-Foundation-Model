"""Isolated Chronos-2 embedding subprocess worker.

Input windows use the classifier seam's ordinary ``[N, C, T]`` convention.
Every window is one Chronos-2 multivariate group, so its C variates exchange
information through group attention.  The parent orchestration process remains
torch-free.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def _pool_embedding(value, mode: str) -> np.ndarray:
    """Pool one official ``[C, patches + 2, D]`` Chronos-2 embedding."""
    array = value.detach().float().cpu().numpy()
    if array.ndim != 3 or array.shape[1] < 3:
        raise RuntimeError(f"unexpected Chronos-2 embedding shape: {array.shape}")
    if mode == "reg":
        pooled = array[:, -2, :]  # official +2 tokens: [REG], masked output
    elif mode == "mean_context":
        pooled = array[:, :-2, :].mean(axis=1)
    else:
        raise ValueError(f"unsupported Chronos-2 pooling mode: {mode!r}")
    if not np.isfinite(pooled).all():
        raise RuntimeError("Chronos-2 produced non-finite pooled embeddings")
    return pooled.reshape(-1).astype(np.float32, copy=False)


def main(directory: str | Path) -> None:
    directory = Path(directory)
    cfg = json.loads((directory / "cfg.json").read_text())
    windows = np.load(cfg.pop("_windows"), mmap_mode="r")
    if windows.ndim != 3:
        raise ValueError(f"windows must have shape [N,C,T], got {windows.shape}")
    if windows.shape[1] != 5:
        raise ValueError(
            "serving windows must contain exactly one ticker in OHLCV order")

    from chronos import Chronos2Pipeline

    checkpoint = cfg.get("ckpt") or cfg.get("model_id", "autogluon/chronos-2-small")
    device = cfg.get("device", "cpu")
    batch_series = max(int(cfg.get("batch", 5)), int(windows.shape[1]))
    pool = cfg.get("pool", "reg")
    context_length = int(cfg.get("context_length") or windows.shape[-1])
    load_kwargs = {"device_map": device}
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir() and (checkpoint_path / "adapter_config.json").is_file():
        # PEFT 0.20 fails closed on adapter auto-mapping. Chronos's own model
        # class is the only dynamic import this worker permits.
        load_kwargs["import_allowlist"] = ["chronos.chronos2.model"]
    pipeline = Chronos2Pipeline.from_pretrained(checkpoint, **load_kwargs)

    # ``batch_size`` counts variates, not windows. Bound each call explicitly so
    # an M1 never materializes the full candidate corpus in accelerator memory.
    windows_per_call = max(1, batch_series // int(windows.shape[1]))
    output = []
    for start in range(0, len(windows), windows_per_call):
        chunk = np.asarray(windows[start:start + windows_per_call], np.float32)
        embeddings, _ = pipeline.embed(
            chunk,
            batch_size=batch_series,
            context_length=context_length,
        )
        output.extend(_pool_embedding(value, pool) for value in embeddings)
    if len(output) != len(windows):
        raise RuntimeError("Chronos-2 embedding count does not match input windows")
    np.save(directory / "emb.npy", np.stack(output).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1])
