"""Isolated frozen Chronos-2 worker for paired 3m/developing-15m inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .paired_embeddings import (
    Chronos2PairedEmbeddings,
    TOTAL_VARIATES,
    validate_paired_windows,
)


IMPORT_ALLOWLIST = ["chronos.chronos2.model"]


def group_ids_for_windows(batch_windows: int) -> np.ndarray:
    """Return one distinct group ID per ten-variate window."""
    if int(batch_windows) < 1:
        raise ValueError("batch_windows must be positive")
    return np.repeat(
        np.arange(int(batch_windows), dtype=np.int64), TOTAL_VARIATES
    )


def _hidden_state(encoder_outputs):
    try:
        hidden = encoder_outputs[0]
    except (TypeError, IndexError, KeyError) as error:
        raise RuntimeError("Chronos-2 encoder returned no hidden state") from error
    if getattr(hidden, "ndim", None) != 3:
        raise RuntimeError(
            "Chronos-2 hidden state must have shape [B*10,tokens,D], "
            f"got {getattr(hidden, 'shape', None)}"
        )
    return hidden


def _extract_named_reg(
    encoder_outputs,
    *,
    context_patches: int,
    batch_windows: int,
) -> Chronos2PairedEmbeddings:
    """Extract the official native REG token at ``-2`` into named halves."""
    hidden = _hidden_state(encoder_outputs)
    if hidden.shape[0] != int(batch_windows) * TOTAL_VARIATES:
        raise RuntimeError("Chronos-2 hidden-state row count drifted")
    # With one masked output patch, official Chronos-2 layout is:
    # [context patches..., REG, masked output].  Prove rather than assume that
    # the encoder-reported context-patch count points to token -2.
    if hidden.shape[1] < 2 or int(context_patches) != hidden.shape[1] - 2:
        raise RuntimeError("Chronos-2 REG token is no longer at position -2")
    reg = hidden[:, -2, :].detach().float().cpu().numpy()
    if reg.ndim != 2 or reg.shape[1] < 1 or not np.isfinite(reg).all():
        raise RuntimeError("Chronos-2 produced invalid native REG embeddings")
    grouped = reg.reshape(int(batch_windows), TOTAL_VARIATES, reg.shape[-1])
    return Chronos2PairedEmbeddings(
        grouped[:, :5].astype(np.float32, copy=False),
        grouped[:, 5:].astype(np.float32, copy=False),
    )


def encode_paired_chunk(model, windows: np.ndarray) -> Chronos2PairedEmbeddings:
    """Encode one finite ``[B,10,T]`` chunk using explicit group IDs."""
    import torch

    values = validate_paired_windows(windows)
    batch_windows, _, history = values.shape
    flat = torch.from_numpy(values.reshape(batch_windows * TOTAL_VARIATES, history))
    device = getattr(model, "device", torch.device("cpu"))
    context = flat.to(device=device, dtype=torch.float32)
    context_mask = torch.ones_like(context, dtype=torch.bool)
    group_ids = torch.from_numpy(group_ids_for_windows(batch_windows)).to(device)

    with torch.no_grad():
        encoder_outputs, _, _, context_patches = model.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            num_output_patches=1,
        )
    return _extract_named_reg(
        encoder_outputs,
        context_patches=int(context_patches),
        batch_windows=batch_windows,
    )


def embed_paired_window_chunks(
    chunks,
    *,
    checkpoint: str | Path | None = None,
    model_id: str = "autogluon/chronos-2-small",
    device: str = "cpu",
    batch_windows: int = 1,
    context_length: int | None = None,
):
    """Yield named paired embeddings while loading the frozen model once."""
    from chronos import Chronos2Pipeline

    if int(batch_windows) < 1:
        raise ValueError("batch_windows must be positive")
    source = checkpoint or model_id
    load_kwargs = {"device_map": device}
    checkpoint_path = Path(source)
    if (
        checkpoint_path.is_dir()
        and (checkpoint_path / "adapter_config.json").is_file()
    ):
        load_kwargs["import_allowlist"] = IMPORT_ALLOWLIST
    pipeline = Chronos2Pipeline.from_pretrained(source, **load_kwargs)
    model = pipeline.model
    model.eval()
    maximum_context = int(pipeline.model_context_length)

    for raw_windows in chunks:
        windows = validate_paired_windows(raw_windows)
        resolved_context = int(context_length or windows.shape[-1])
        if resolved_context < 1 or resolved_context > windows.shape[-1]:
            raise ValueError(
                "context_length must be positive and no larger than window history"
            )
        if resolved_context > maximum_context:
            raise ValueError(
                f"context_length {resolved_context} exceeds frozen model limit "
                f"{maximum_context}"
            )
        windows = windows[:, :, -resolved_context:]
        primary_parts, related_parts = [], []
        for start in range(0, len(windows), int(batch_windows)):
            output = encode_paired_chunk(
                model, windows[start:start + int(batch_windows)]
            )
            primary_parts.append(output.three_minute)
            related_parts.append(output.developing_fifteen_minute)
        yield Chronos2PairedEmbeddings(
            np.concatenate(primary_parts, axis=0),
            np.concatenate(related_parts, axis=0),
        )
