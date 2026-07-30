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


def embed_window_chunks(
        chunks,
        *,
        checkpoint: str | Path | None = None,
        model_id: str = "autogluon/chronos-2-small",
        device: str = "cpu",
        batch: int = 5,
        pool: str = "reg",
        context_length: int | None = None,
):
    """Yield Chronos-2 embeddings while loading the frozen pipeline only once."""
    from chronos import Chronos2Pipeline

    source = checkpoint or model_id
    load_kwargs = {"device_map": device}
    checkpoint_path = Path(source)
    if (
        checkpoint_path.is_dir()
        and (checkpoint_path / "adapter_config.json").is_file()
    ):
        load_kwargs["import_allowlist"] = ["chronos.chronos2.model"]
    pipeline = Chronos2Pipeline.from_pretrained(source, **load_kwargs)

    for windows in chunks:
        windows = np.asarray(windows, np.float32)
        if windows.ndim != 3 or windows.shape[1] != 5:
            raise ValueError(
                "Chronos-2 chunks must have shape [N,5,T] in OHLCV order")
        if not np.isfinite(windows).all():
            raise ValueError("Chronos-2 chunks contain non-finite OHLCV")
        batch_series = max(int(batch), int(windows.shape[1]))
        windows_per_call = max(1, batch_series // int(windows.shape[1]))
        resolved_context = int(context_length or windows.shape[-1])
        output = []
        for start in range(0, len(windows), windows_per_call):
            chunk = windows[start:start + windows_per_call]
            embeddings, _ = pipeline.embed(
                chunk,
                batch_size=batch_series,
                context_length=resolved_context,
            )
            output.extend(_pool_embedding(value, pool) for value in embeddings)
        if len(output) != len(windows):
            raise RuntimeError(
                "Chronos-2 embedding count does not match input windows")
        yield (
            np.stack(output).astype(np.float32)
            if output else np.zeros((0, 0), np.float32)
        )


def main(directory: str | Path) -> None:
    directory = Path(directory)
    cfg = json.loads((directory / "cfg.json").read_text())
    windows = np.load(cfg.pop("_windows"), mmap_mode="r")
    if windows.ndim != 3:
        raise ValueError(f"windows must have shape [N,C,T], got {windows.shape}")
    if windows.shape[1] != 5:
        raise ValueError(
            "serving windows must contain exactly one ticker in OHLCV order")

    checkpoint = cfg.get("ckpt") or cfg.get("model_id", "autogluon/chronos-2-small")
    device = cfg.get("device", "cpu")
    batch_series = int(cfg.get("batch", 5))
    pool = cfg.get("pool", "reg")
    context_length = int(cfg.get("context_length") or windows.shape[-1])
    output = next(embed_window_chunks(
        (windows,),
        checkpoint=checkpoint,
        device=device,
        batch=batch_series,
        pool=pool,
        context_length=context_length,
    ))
    np.save(directory / "emb.npy", output)


if __name__ == "__main__":
    main(sys.argv[1])
