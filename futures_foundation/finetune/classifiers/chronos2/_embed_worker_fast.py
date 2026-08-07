"""Batched Chronos-2 REG embedding worker for dense cache generation.

The official ``Chronos2Pipeline.embed`` interface returns every grouped
embedding and its unused location/scale tensors to the host separately. Dense
stride-one caches need only the encoder's REG token. This worker preserves the
official dataset, grouping, model encode call, order, and output values while
performing one bounded device-to-host transfer per model batch.

The legacy ``_embed_worker`` remains unchanged so caches produced by its
authenticated module hash remain reusable.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def _embed_reg_windows(
        pipeline,
        windows: np.ndarray,
        *,
        batch_series: int,
        context_length: int,
) -> np.ndarray:
    import torch
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
    from torch.utils.data import DataLoader

    resolved_context = min(
        int(context_length), int(pipeline.model_context_length))
    dataset = Chronos2Dataset(
        windows,
        context_length=resolved_context,
        prediction_length=0,
        batch_size=int(batch_series),
        output_patch_size=int(pipeline.model_output_patch_size),
        mode=DatasetMode.TEST,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=pipeline.model.device.type == "cuda",
        shuffle=False,
        drop_last=False,
    )
    batches = []
    with torch.no_grad():
        for batch in loader:
            if batch["future_target"] is not None:
                raise RuntimeError(
                    "Chronos-2 embedding dataset exposed a future target")
            ranges = tuple(batch["target_idx_ranges"])
            encoder_outputs, _loc_scale, *_ = pipeline.model.encode(
                context=batch["context"].to(
                    device=pipeline.model.device,
                    dtype=torch.float32,
                ),
                group_ids=batch["group_ids"].to(pipeline.model.device),
            )
            encoded = encoder_outputs[0]
            expected_start = 0
            for start, end in ranges:
                if (
                    int(start) != expected_start
                    or int(end) - int(start) != windows.shape[1]
                ):
                    raise RuntimeError(
                        "Chronos-2 grouped target ranges are malformed")
                expected_start = int(end)
            if (
                not ranges
                or encoded.ndim != 3
                or encoded.shape[0] != expected_start
                or encoded.shape[1] < 3
            ):
                raise RuntimeError(
                    f"unexpected Chronos-2 encoder shape: {encoded.shape}")
            pooled = (
                encoded[:, -2, :]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            batches.append(pooled.reshape(len(ranges), -1))
    if not batches:
        raise RuntimeError("Chronos-2 embedding dataset returned no batches")
    output = np.concatenate(batches).astype(np.float32, copy=False)
    if output.shape[0] != len(windows) or not np.isfinite(output).all():
        raise RuntimeError("Chronos-2 REG embedding output is malformed")
    return output


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
    """Yield official-equivalent REG embeddings from bounded input chunks."""
    if pool != "reg":
        from ._embed_worker import embed_window_chunks as legacy_embed

        yield from legacy_embed(
            chunks,
            checkpoint=checkpoint,
            model_id=model_id,
            device=device,
            batch=batch,
            pool=pool,
            context_length=context_length,
        )
        return

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
        if not len(windows) or not np.isfinite(windows).all():
            raise ValueError(
                "Chronos-2 chunks must be nonempty and contain finite OHLCV")
        batch_series = max(int(batch), int(windows.shape[1]))
        resolved_context = int(context_length or windows.shape[-1])
        yield _embed_reg_windows(
            pipeline,
            windows,
            batch_series=batch_series,
            context_length=resolved_context,
        )


def main(directory: str | Path) -> None:
    directory = Path(directory)
    cfg = json.loads((directory / "cfg.json").read_text())
    windows = np.load(cfg.pop("_windows"), mmap_mode="r")
    if windows.ndim != 3:
        raise ValueError(f"windows must have shape [N,C,T], got {windows.shape}")
    output = next(embed_window_chunks(
        (windows,),
        checkpoint=cfg.get("ckpt") or cfg.get(
            "model_id", "autogluon/chronos-2-small"),
        device=cfg.get("device", "cpu"),
        batch=int(cfg.get("batch", 5)),
        pool=cfg.get("pool", "reg"),
        context_length=int(cfg.get("context_length") or windows.shape[-1]),
    ))
    np.save(directory / "emb.npy", output)


if __name__ == "__main__":
    main(sys.argv[1])
