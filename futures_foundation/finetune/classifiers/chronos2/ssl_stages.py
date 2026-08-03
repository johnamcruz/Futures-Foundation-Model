"""Chronos-2 LoRA mask and temporal-contrastive SSL stages.

The objective heads are temporary.  Each stage saves only a PEFT adapter that
can be loaded by the ordinary Chronos-2 pipeline and classifier seam.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np


IMPORT_ALLOWLIST = ["chronos.chronos2.model"]
CHRONOS2_MODEL_ID = "autogluon/chronos-2-small"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def tree_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_torch(path: Path, payload) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@contextmanager
def _runtime_pinned_adapter(
        parent: str | Path,
        base_revision: str | None,
        base_snapshot: str | Path | None = None,
):
    """Load an adapter against the exact authenticated local base bytes."""
    parent = Path(parent).expanduser().resolve()
    base_snapshot = (
        None if base_snapshot is None
        else Path(base_snapshot).expanduser().resolve()
    )
    if base_snapshot is not None and (
        base_revision is None
        or base_snapshot.name != base_revision
        or not (base_snapshot / "config.json").is_file()
        or not (base_snapshot / "model.safetensors").is_file()
    ):
        raise RuntimeError(
            "pinned adapter base snapshot is incomplete or revision-mismatched")
    if base_revision is None:
        if base_snapshot is not None:
            raise RuntimeError("base snapshot requires an exact base revision")
        yield parent
        return
    config_path = parent / "adapter_config.json"
    config = _read_json_object(config_path)
    if config.get("revision") == base_revision and base_snapshot is None:
        yield parent
        return
    if config.get("revision") not in {None, base_revision}:
        raise RuntimeError("adapter base revision conflicts with pinned snapshot")
    if base_snapshot is None:
        config["revision"] = base_revision
    else:
        if config.get("base_model_name_or_path") != CHRONOS2_MODEL_ID:
            raise RuntimeError(
                "adapter base model does not match the authenticated snapshot")
        # PEFT consumes this temporary config when it instantiates the base.
        # A local path binds the loaded tensors to the exact files hashed by
        # _chronos_base_identity instead of merely trusting a Hub revision.
        config["base_model_name_or_path"] = str(base_snapshot)
        config["revision"] = None
    with tempfile.TemporaryDirectory(
        prefix="ffm-chronos2-pinned-adapter-",
    ) as temporary:
        runtime = Path(temporary)
        (runtime / "adapter_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n")
        weights = [
            path for name in ("adapter_model.safetensors", "adapter_model.bin")
            if (path := parent / name).is_file()
        ]
        if len(weights) != 1:
            raise RuntimeError("PEFT adapter must contain exactly one weight file")
        (runtime / weights[0].name).symlink_to(weights[0].resolve())
        yield runtime


def _load_trainable_adapter(
        parent: str | Path,
        device: str,
        *,
        base_revision: str | None = None,
        base_snapshot: str | Path | None = None,
):
    from peft import AutoPeftModel

    with _runtime_pinned_adapter(
        parent,
        base_revision,
        base_snapshot,
    ) as runtime_parent:
        model = AutoPeftModel.from_pretrained(
            runtime_parent,
            is_trainable=True,
            import_allowlist=IMPORT_ALLOWLIST,
        ).to(device)
    if base_snapshot is not None:
        peft_configs = getattr(model, "peft_config", None)
        if not isinstance(peft_configs, Mapping) or not peft_configs:
            raise RuntimeError("loaded adapter exposes no PEFT configuration")
        for adapter_config in peft_configs.values():
            adapter_config.base_model_name_or_path = CHRONOS2_MODEL_ID
            adapter_config.revision = base_revision
    model.train()
    base = model.base_model.model
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, parameter in model.named_parameters()
                            if parameter.requires_grad):
        raise RuntimeError("parent did not load as a trainable LoRA-only Chronos-2 adapter")
    return model, base


def _adapter_state(model) -> dict:
    from peft import get_peft_model_state_dict

    return {
        key: value.detach().cpu().clone()
        for key, value in get_peft_model_state_dict(model).items()
    }


def _restore_adapter(model, state: dict) -> None:
    from peft import set_peft_model_state_dict

    result = set_peft_model_state_dict(model, state)
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"unexpected LoRA keys while restoring: {result.unexpected_keys}")


def _standardize(window):
    """Per-window/per-variate z-score preserving the natural missing mask."""
    import torch

    finite = torch.isfinite(window)
    count = finite.sum(-1, keepdim=True).clamp_min(1)
    clean = torch.where(finite, window, torch.zeros_like(window))
    mean = clean.sum(-1, keepdim=True) / count
    centered = torch.where(finite, window - mean, torch.zeros_like(window))
    variance = centered.square().sum(-1, keepdim=True) / count
    scaled = centered / variance.sqrt().clamp_min(1e-5)
    return scaled.clamp(-10.0, 10.0), finite


def _sample_windows(matrix, starts, context_length, device):
    import torch

    windows = np.stack([
        matrix[:, int(start):int(start) + context_length] for start in starts
    ]).astype(np.float32)
    return torch.from_numpy(windows).to(device)


def _reg_embeddings(base, windows, finite):
    """Jointly encode B groups of C variates -> group-size-invariant [B,D]."""
    import torch

    batch, channels, _ = windows.shape
    flat = windows.reshape(batch * channels, -1)
    mask = finite.reshape(batch * channels, -1)
    groups = torch.arange(batch, device=windows.device).repeat_interleave(channels)
    outputs, _, _, context_patches = base.encode(
        context=flat,
        context_mask=mask,
        group_ids=groups,
        num_output_patches=1,
    )
    reg = outputs[0][:, context_patches, :]
    return reg.reshape(batch, channels, reg.shape[-1]).mean(dim=1)


def _ticker_indices(prepared) -> dict[str, np.ndarray]:
    tickers = tuple(dict.fromkeys(name.split(".", 1)[0] for name in prepared.channel_names))
    result = {
        ticker: np.asarray([
            index for index, name in enumerate(prepared.channel_names)
            if name.startswith(f"{ticker}.")
        ], dtype=np.int64)
        for ticker in tickers
    }
    if not result or any(len(indices) != 5 for indices in result.values()):
        raise RuntimeError("every ticker must have exactly five OHLCV channels")
    return result


def _as_corpus(prepared) -> dict[str, object]:
    if isinstance(prepared, Mapping):
        corpus = {str(key): value for key, value in prepared.items()}
    else:
        corpus = {str(prepared.report.get("timeframe", "3min")): prepared}
    if not corpus:
        raise ValueError("Chronos-2 corpus cannot be empty")
    reference_channels = None
    for timeframe, item in corpus.items():
        channels = tuple(item.channel_names)
        if reference_channels is None:
            reference_channels = channels
        elif channels != reference_channels:
            raise RuntimeError(
                f"OHLCV channel order differs for timeframe {timeframe}")
    return corpus


def _corpus_identity(corpus: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps({
        timeframe: item.report["identity_sha256"]
        for timeframe, item in corpus.items()
    }, sort_keys=True).encode()).hexdigest()


def _available_starts(length: int, context_length: int) -> np.ndarray:
    if length < context_length:
        raise RuntimeError("aligned training history is shorter than context_length")
    return np.arange(0, length - context_length + 1, dtype=np.int64)


def _observable_starts(
        matrix: np.ndarray,
        context_length: int,
        *,
        min_observations: int = 1,
) -> np.ndarray:
    """Starts where every selected variate has enough real observations."""
    starts = _available_starts(matrix.shape[1], context_length)
    valid = np.ones(len(starts), dtype=bool)
    for row in matrix:
        prefix = np.empty(row.shape[0] + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(np.isfinite(row), dtype=np.int64, out=prefix[1:])
        valid &= (prefix[context_length:] - prefix[:-context_length]) >= min_observations
    result = starts[valid]
    if not len(result):
        raise RuntimeError(
            "no training window has sufficient observations for every selected OHLCV channel")
    return result


def _validation_contexts(prepared, context_length: int, prediction_length: int):
    contexts = []
    for window in prepared.validation:
        context = window[:, -(context_length + prediction_length):-prediction_length]
        if context.shape[-1] != context_length:
            raise RuntimeError("validation context length mismatch")
        contexts.append(context)
    return tuple(contexts)


def _save_final(model, checkpoint: Path) -> None:
    if checkpoint.exists():
        raise RuntimeError(f"refusing to overwrite completed checkpoint: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)


def train_mask(
        prepared,
        *,
        parent: str | Path,
        out_dir: str | Path,
        device: str = "mps",
        context_length: int = 256,
        prediction_length: int = 32,
        mask_ratio: float = 0.40,
        epochs: int = 2,
        steps_per_epoch: int = 10,
        batch_windows: int = 1,
        gradient_accumulation: int = 8,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.05,
        patience: int = 3,
        seed: int = 0,
        single_ticker_probability: float = 0.25,
        resume: bool = False,
) -> dict:
    """Masked-patch reconstruction, saving only the best Chronos-2 adapter."""
    import torch
    import torch.nn as nn

    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    if not 0.0 <= single_ticker_probability <= 1.0:
        raise ValueError("single_ticker_probability must be between 0 and 1")
    corpus = _as_corpus(prepared)
    timeframes = tuple(corpus)
    out_dir, parent = Path(out_dir), Path(parent)
    checkpoint = out_dir / "checkpoint"
    state_path = out_dir / "trainer.pt"
    report_path = out_dir / "report.json"
    if checkpoint.exists():
        raise RuntimeError(f"completed mask checkpoint already exists: {checkpoint}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, base = _load_trainable_adapter(parent, device)
    patch_size = int(base.chronos_config.input_patch_size)
    if context_length % patch_size:
        raise ValueError("context_length must be divisible by Chronos-2 patch size")
    decoder = nn.Linear(int(base.model_dim), patch_size).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters += list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay)
    validation = {
        timeframe: _validation_contexts(
            item, context_length, prediction_length)
        for timeframe, item in corpus.items()
    }
    ticker_indices = {
        timeframe: _ticker_indices(item)
        for timeframe, item in corpus.items()
    }
    ticker_names = tuple(next(iter(ticker_indices.values())))
    # A union-aligned joint timeline can contain a window where one market is
    # entirely closed. Chronos group attention cannot safely consume an
    # all-missing variate, so sample only observable task windows.
    joint_starts = {
        timeframe: _observable_starts(
            item.train, context_length, min_observations=patch_size)
        for timeframe, item in corpus.items()
    }
    ticker_starts = {
        timeframe: {
            ticker: _observable_starts(
                corpus[timeframe].train[indices],
                context_length,
                min_observations=patch_size,
            )
            for ticker, indices in indices_by_ticker.items()
        }
        for timeframe, indices_by_ticker in ticker_indices.items()
    }
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    best_loss, best_adapter, best_decoder = math.inf, None, None
    history, start_epoch, bad = [], 0, 0

    if resume and state_path.is_file():
        saved = torch.load(state_path, map_location="cpu")
        _restore_adapter(model, saved["adapter"])
        decoder.load_state_dict(saved["decoder"])
        optimizer.load_state_dict(saved["optimizer"])
        best_loss = float(saved["best_loss"])
        best_adapter = saved["best_adapter"]
        best_decoder = saved["best_decoder"]
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1
        bad = int(saved["bad"])
        rng.bit_generator.state = saved["numpy_rng"]
        generator.set_state(saved["torch_generator"])

    def loss_for(raw, mask_generator):
        standardized, finite = _standardize(raw)
        time_mask = torch.rand(
            (raw.shape[0], raw.shape[-1]), device=device,
            generator=mask_generator) < mask_ratio
        time_mask[:, 0] = True
        selected = finite & time_mask[:, None, :]
        visible = finite & ~time_mask[:, None, :]
        flat = standardized.reshape(-1, context_length)
        visible_flat = visible.reshape(-1, context_length)
        batch, channels, _ = standardized.shape
        groups = torch.arange(batch, device=device).repeat_interleave(channels)
        outputs, _, _, n_patches = base.encode(
            context=flat,
            context_mask=visible_flat,
            group_ids=groups,
            num_output_patches=1,
        )
        reconstructed = decoder(outputs[0][:, :n_patches, :]).reshape(
            batch, channels, n_patches * patch_size)[..., :context_length]
        return (reconstructed[selected] - standardized[selected]).square().mean()

    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        model.train()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        for step in range(steps_per_epoch):
            timeframe = timeframes[int(rng.integers(len(timeframes)))]
            item = corpus[timeframe]
            if rng.random() < single_ticker_probability:
                ticker = ticker_names[int(rng.integers(len(ticker_names)))]
                train_matrix = item.train[ticker_indices[timeframe][ticker]]
                starts = ticker_starts[timeframe][ticker]
            else:
                train_matrix = item.train
                starts = joint_starts[timeframe]
            chosen = rng.choice(starts, size=batch_windows, replace=True)
            loss = loss_for(
                _sample_windows(train_matrix, chosen, context_length, device),
                generator,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "non-finite mask training loss; refusing to update the adapter")
            (loss / gradient_accumulation).backward()
            train_total += float(loss.detach())
            if ((step + 1) % gradient_accumulation == 0
                    or step + 1 == steps_per_epoch):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        decoder.eval()
        with torch.no_grad():
            validation_generator = torch.Generator(device=device)
            validation_generator.manual_seed(20260704)
            val_timeframe_losses = {}
            for timeframe in timeframes:
                joint_values = [
                    float(loss_for(
                        torch.from_numpy(value[None]).to(device),
                        validation_generator,
                    ))
                    for value in validation[timeframe]
                ]
                ticker_values = {}
                for ticker, indices in ticker_indices[timeframe].items():
                    ticker_generator = torch.Generator(device=device)
                    ticker_generator.manual_seed(20260704)
                    ticker_values[ticker] = float(np.mean([
                        float(loss_for(
                            torch.from_numpy(value[indices][None]).to(device),
                            ticker_generator,
                        ))
                        for value in validation[timeframe]
                    ]))
                val_timeframe_losses[timeframe] = {
                    "joint_loss": float(np.mean(joint_values)),
                    "single_ticker_losses": ticker_values,
                    "mean_single_ticker_loss": float(
                        np.mean(list(ticker_values.values()))),
                }
        val_joint_loss = float(np.mean([
            value["joint_loss"] for value in val_timeframe_losses.values()
        ]))
        val_single_loss = float(np.mean([
            value["mean_single_ticker_loss"]
            for value in val_timeframe_losses.values()
        ]))
        val_loss = 0.5 * (val_joint_loss + val_single_loss)
        improved = val_loss < best_loss - 1e-6
        if improved:
            best_loss, bad = val_loss, 0
            best_adapter = _adapter_state(model)
            best_decoder = deepcopy({
                key: value.detach().cpu() for key, value in decoder.state_dict().items()})
        else:
            bad += 1
        row = {
            "epoch": epoch,
            "train_loss": train_total / steps_per_epoch,
            "val_loss": val_loss,
            "val_joint_loss": val_joint_loss,
            "val_single_ticker_loss": val_single_loss,
            "val_timeframe_losses": val_timeframe_losses,
            "improved": improved,
        }
        history.append(row)
        print(
            f"[chronos2-mask] ep={epoch} train={row['train_loss']:.5f} "
            f"val={val_loss:.5f} joint={val_joint_loss:.5f} "
            f"single={val_single_loss:.5f}{' *' if improved else ''}",
            flush=True,
        )
        _atomic_torch(state_path, {
            "epoch": epoch,
            "adapter": _adapter_state(model),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_adapter": best_adapter,
            "best_decoder": best_decoder,
            "history": history,
            "bad": bad,
            "numpy_rng": rng.bit_generator.state,
            "torch_generator": generator.get_state(),
        })
        if bad >= patience:
            break
    if best_adapter is None:
        raise RuntimeError("mask stage never produced a finite validation checkpoint")
    _restore_adapter(model, best_adapter)
    _save_final(model, checkpoint)
    report = {
        "schema": "ffm_chronos2_mask_v1",
        "stage": "mask",
        "status": "complete",
        "parent": {"path": str(parent), "sha256": tree_sha256(parent)},
        "checkpoint": {"path": str(checkpoint), "sha256": tree_sha256(checkpoint)},
        "data_identity_sha256": _corpus_identity(corpus),
        "config": {
            "timeframes": list(timeframes),
            "context_length": context_length,
            "mask_ratio": mask_ratio,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "batch_windows": batch_windows,
            "gradient_accumulation": gradient_accumulation,
            "learning_rate": learning_rate,
            "single_ticker_probability": single_ticker_probability,
            "seed": seed,
        },
        "best_val_loss": best_loss,
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(report_path, report)
    return report


def _augment(values, finite, generator, *, noise, scale, time_mask):
    import torch

    output = values.clone()
    if scale:
        factor = 1.0 + scale * (
            2.0 * torch.rand(
                (*output.shape[:-1], 1), device=output.device, generator=generator) - 1.0)
        output = output * factor
    if noise:
        output = output + noise * torch.randn(
            output.shape, device=output.device, generator=generator)
    if time_mask:
        length = output.shape[-1]
        width = max(1, int(length * time_mask))
        start = int(torch.randint(
            0, length - width + 1, (1,), device=output.device,
            generator=generator).item())
        finite = finite.clone()
        finite[..., start:start + width] = False
    return output, finite


def _weighted_temporal_contrastive(
        embeddings, group, valid_positive, positions, temperature, far_min):
    import torch

    similarity = embeddings @ embeddings.T / temperature
    eye = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    same = (group[:, None] == group[None, :]) & ~eye
    near = (positions[:, None] - positions[None, :]).abs() < far_min
    excluded = eye | (near & ~same)
    similarity = similarity.masked_fill(excluded, -1e9)
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positive = same & valid_positive[None, :]
    count = positive.sum(1)
    valid = (count > 0) & valid_positive
    if not valid.any():
        raise RuntimeError("contrastive batch contains no valid anchors")
    row_loss = -(log_probability * positive).sum(1) / count.clamp_min(1)
    return row_loss[valid].mean()


def _kaufman_regime(raw, *, chop: float, trend: float, close_channel: int = 3):
    """Causal Kaufman ER labels matching the production Mantis pipeline."""
    import torch

    if not 0.0 <= chop < trend <= 1.0:
        raise ValueError("Kaufman thresholds must satisfy 0 <= chop < trend <= 1")
    close = raw[:, close_channel, :]
    displacement = close[:, -1] - close[:, 0]
    path = close.diff(dim=1).abs().sum(dim=1)
    efficiency = (displacement.abs() / path.clamp_min(1e-9)).clamp(0.0, 1.0)
    regime = torch.full_like(displacement, -1, dtype=torch.long)
    regime[efficiency <= chop] = 0
    regime[(efficiency >= trend) & (displacement > 0)] = 1
    regime[(efficiency >= trend) & (displacement < 0)] = 2
    return regime, efficiency


def _range_dynamics_scores(
        matrix: np.ndarray,
        starts: np.ndarray,
        *,
        context_length: int,
        dynamics_length: int,
) -> np.ndarray:
    """Causal log(second-half range / first-half range) from completed bars."""
    if not 4 <= dynamics_length <= context_length or dynamics_length % 2:
        raise ValueError(
            "dynamics_length must be even and in [4, context_length]")
    starts = np.asarray(starts, dtype=np.int64)
    offsets = (
        starts[:, None]
        + context_length - dynamics_length
        + np.arange(dynamics_length, dtype=np.int64)[None, :]
    )
    high = np.asarray(matrix[1, offsets], dtype=np.float64)
    low = np.asarray(matrix[2, offsets], dtype=np.float64)
    half = dynamics_length // 2
    first = high[:, :half].max(axis=1) - low[:, :half].min(axis=1)
    second = high[:, half:].max(axis=1) - low[:, half:].min(axis=1)
    scale = np.maximum(np.maximum(first, second), 1.0)
    epsilon = np.finfo(np.float64).eps * scale
    return np.log((second + epsilon) / (first + epsilon)).astype(np.float32)


def _fit_range_dynamics_thresholds(
        matrix: np.ndarray,
        starts: np.ndarray,
        *,
        context_length: int,
        dynamics_length: int,
        lower_quantile: float,
        upper_quantile: float,
        max_samples: int,
) -> dict[str, float | int]:
    """Fit scale-free SSL state cutoffs using training contexts only."""
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError(
            "range-dynamics quantiles must satisfy 0 < lower < upper < 1")
    if max_samples < 3:
        raise ValueError("threshold max_samples must be >=3")
    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) > max_samples:
        positions = np.linspace(
            0, len(starts) - 1, num=max_samples, dtype=np.int64)
        selected = starts[positions]
    else:
        selected = starts
    scores = _range_dynamics_scores(
        matrix,
        selected,
        context_length=context_length,
        dynamics_length=dynamics_length,
    )
    if not np.isfinite(scores).all():
        raise RuntimeError(
            "non-finite range-dynamics score in training-only threshold fit")
    lower, upper = np.quantile(
        scores, [lower_quantile, upper_quantile])
    if not float(lower) < float(upper):
        raise RuntimeError(
            "range-dynamics training distribution cannot define distinct states")
    return {
        "compression_max": float(lower),
        "expansion_min": float(upper),
        "fit_samples": int(len(scores)),
        "fit_score_mean": float(np.mean(scores)),
        "fit_score_std": float(np.std(scores)),
    }


def _range_dynamics_regime(
        raw,
        lower,
        upper,
        *,
        dynamics_length: int,
        high_channel: int = 1,
        low_channel: int = 2,
):
    """Self-supervised compression/stable/expansion state from input context."""
    import torch

    if not 4 <= dynamics_length <= raw.shape[-1] or dynamics_length % 2:
        raise ValueError(
            "dynamics_length must be even and no longer than the input context")
    high = raw[:, high_channel, -dynamics_length:]
    low = raw[:, low_channel, -dynamics_length:]
    half = dynamics_length // 2
    first = (
        high[:, :half].amax(dim=1) - low[:, :half].amin(dim=1))
    second = (
        high[:, half:].amax(dim=1) - low[:, half:].amin(dim=1))
    scale = torch.maximum(first, second).clamp_min(1.0)
    epsilon = torch.finfo(raw.dtype).eps * scale
    score = torch.log((second + epsilon) / (first + epsilon))
    lower = torch.as_tensor(lower, dtype=score.dtype, device=score.device)
    upper = torch.as_tensor(upper, dtype=score.dtype, device=score.device)
    regime = torch.ones_like(score, dtype=torch.long)
    regime[score <= lower] = 0
    regime[score >= upper] = 2
    return regime, score


def _regime_supcon(embeddings, instance, regime, temperature):
    """SupCon positives: same view-instance or same recognized SSL state."""
    import torch

    count = len(embeddings)
    eye = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    same_instance = instance[:, None] == instance[None, :]
    known = (regime[:, None] >= 0) & (regime[None, :] >= 0)
    same_regime = known & (regime[:, None] == regime[None, :])
    positive = (same_instance | same_regime) & ~eye
    similarity = embeddings @ embeddings.T / temperature
    similarity = similarity.masked_fill(eye, -1e9)
    log_probability = similarity - torch.logsumexp(
        similarity, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1)
    row_loss = -(log_probability * positive).sum(dim=1) / positive_count.clamp_min(1)

    # Match Mantis: inverse-frequency balance prevents common chop windows from
    # dominating; transition rows retain unit weight and only pair by instance.
    balanced = torch.ones(count, device=embeddings.device)
    recognized = regime >= 0
    if recognized.any():
        class_counts = torch.bincount(
            regime[recognized], minlength=3).float().clamp_min(1)
        present = class_counts > 1
        scales = recognized.sum().float() / (
            present.sum().clamp_min(1) * class_counts)
        balanced[recognized] = scales[regime[recognized]]
        balanced = balanced / balanced.mean().clamp_min(1e-9)
    valid = positive_count > 0
    return (row_loss[valid] * balanced[valid]).sum() / balanced[valid].sum().clamp_min(1e-9)


def train_contrastive(
        prepared,
        *,
        parent: str | Path,
        out_dir: str | Path,
        device: str = "mps",
        context_length: int = 256,
        prediction_length: int = 32,
        pos_deltas: tuple[int, ...] = (2, 16, 64),
        far_min: int = 512,
        temperature: float = 0.10,
        noise: float = 0.02,
        scale: float = 0.10,
        time_mask: float = 0.0,
        projection_dim: int = 128,
        epochs: int = 2,
        steps_per_epoch: int = 4,
        batch_windows: int = 2,
        gradient_accumulation: int = 2,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.05,
        patience: int = 3,
        seed: int = 0,
        regime_key: str = "kaufman",
        kaufman_chop: float = 0.25,
        kaufman_trend: float = 0.50,
        kaufman_length: int = 64,
        volatility_length: int = 64,
        volatility_lower_quantile: float = 0.25,
        volatility_upper_quantile: float = 0.75,
        volatility_threshold_samples: int = 50_000,
        kaufman_retention_weight: float = 1.0,
        single_ticker_probability: float = 0.25,
        resume: bool = False,
) -> dict:
    """Kaufman, volatility-state, or temporal contrastive adapter refinement."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if batch_windows < 2:
        raise ValueError("contrastive batch_windows must be >=2 to supply negatives")
    if regime_key not in {"kaufman", "volatility", "temporal"}:
        raise ValueError(
            "regime_key must be 'kaufman', 'volatility', or 'temporal'")
    if not 2 <= kaufman_length <= context_length:
        raise ValueError("kaufman_length must be in [2, context_length]")
    if regime_key == "kaufman":
        _kaufman_regime(
            torch.zeros((1, 5, 2), device=device),
            chop=kaufman_chop,
            trend=kaufman_trend,
        )
    if regime_key == "volatility":
        if not 4 <= volatility_length <= context_length or volatility_length % 2:
            raise ValueError(
                "volatility_length must be even and in [4, context_length]")
        if not (
            0.0 < volatility_lower_quantile
            < volatility_upper_quantile < 1.0
        ):
            raise ValueError(
                "volatility quantiles must satisfy 0 < lower < upper < 1")
        if volatility_threshold_samples < 3:
            raise ValueError("volatility_threshold_samples must be >=3")
        if kaufman_retention_weight < 0.0:
            raise ValueError("kaufman_retention_weight must be >=0")
    if not 0.0 <= single_ticker_probability <= 1.0:
        raise ValueError("single_ticker_probability must be between 0 and 1")
    corpus = _as_corpus(prepared)
    timeframes = tuple(corpus)
    out_dir, parent = Path(out_dir), Path(parent)
    checkpoint = out_dir / "checkpoint"
    state_path = out_dir / "trainer.pt"
    if checkpoint.exists():
        raise RuntimeError(f"completed contrastive checkpoint already exists: {checkpoint}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model, base = _load_trainable_adapter(parent, device)
    embedding_dim = int(base.model_dim)
    projection = nn.Sequential(
        nn.LayerNorm(embedding_dim),
        nn.Linear(embedding_dim, projection_dim),
        nn.GELU(),
        nn.Linear(projection_dim, projection_dim),
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters += list(projection.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay)
    validation_matrix = {
        timeframe: item.validation_matrix
        for timeframe, item in corpus.items()
    }
    ticker_indices = {
        timeframe: _ticker_indices(item)
        for timeframe, item in corpus.items()
    }
    ticker_names = tuple(next(iter(ticker_indices.values())))
    stream_keys = tuple(
        (timeframe, ticker)
        for timeframe in timeframes
        for ticker in ticker_names
    )

    def temporal_pool(matrix):
        max_delta = max(pos_deltas)
        observable = _observable_starts(matrix, context_length)
        flags = np.zeros(
            matrix.shape[1] - context_length + 1, dtype=bool)
        flags[observable] = True
        candidates = np.arange(
            0, len(flags) - max_delta, dtype=np.int64)
        valid = flags[candidates]
        for delta in pos_deltas:
            valid &= flags[candidates + delta]
        result = candidates[valid]
        if len(result) < batch_windows:
            raise RuntimeError(
                "insufficient observable temporal neighborhoods for contrastive training")
        return result

    def kaufman_pool(matrix):
        # Mantis Kaufman windows come from one uninterrupted stream. Requiring
        # every OHLCV value avoids assigning a pseudo-regime across exchange gaps
        # introduced by the union-aligned multivariate timeline.
        return _observable_starts(
            matrix,
            context_length,
            min_observations=context_length,
        )

    pool_for = temporal_pool if regime_key == "temporal" else kaufman_pool
    ticker_starts = {
        timeframe: {
            ticker: pool_for(corpus[timeframe].train[indices])
            for ticker, indices in indices_by_ticker.items()
        }
        for timeframe, indices_by_ticker in ticker_indices.items()
    }
    val_ticker_starts = {
        timeframe: {
            ticker: pool_for(validation_matrix[timeframe][indices])
            for ticker, indices in indices_by_ticker.items()
        }
        for timeframe, indices_by_ticker in ticker_indices.items()
    }
    if regime_key == "temporal":
        joint_starts = {
            timeframe: temporal_pool(item.train)
            for timeframe, item in corpus.items()
        }
        val_joint_starts = {
            timeframe: temporal_pool(matrix)
            for timeframe, matrix in validation_matrix.items()
        }
    volatility_thresholds = None
    if regime_key == "volatility":
        volatility_thresholds = {
            timeframe: {
                ticker: _fit_range_dynamics_thresholds(
                    corpus[timeframe].train[indices],
                    ticker_starts[timeframe][ticker],
                    context_length=context_length,
                    dynamics_length=volatility_length,
                    lower_quantile=volatility_lower_quantile,
                    upper_quantile=volatility_upper_quantile,
                    max_samples=volatility_threshold_samples,
                )
                for ticker, indices in indices_by_ticker.items()
            }
            for timeframe, indices_by_ticker in ticker_indices.items()
        }
    best_loss, best_adapter = math.inf, None
    history, start_epoch, bad = [], 0, 0
    if resume and state_path.is_file():
        saved = torch.load(state_path, map_location="cpu")
        _restore_adapter(model, saved["adapter"])
        projection.load_state_dict(saved["projection"])
        optimizer.load_state_dict(saved["optimizer"])
        best_loss, best_adapter = float(saved["best_loss"]), saved["best_adapter"]
        history, start_epoch, bad = list(saved["history"]), int(saved["epoch"]) + 1, int(saved["bad"])
        rng.bit_generator.state = saved["numpy_rng"]
        generator.set_state(saved["torch_generator"])

    def choose_anchors(pool, count, numpy_generator):
        for _ in range(100):
            chosen = np.sort(numpy_generator.choice(pool, size=count, replace=False))
            if np.max(np.abs(chosen[:, None] - chosen[None, :])) >= far_min:
                return chosen
        raise RuntimeError("unable to sample far-separated contrastive anchors")

    def temporal_loss_for(matrix, pool, count, numpy_generator, torch_generator):
        anchors = choose_anchors(pool, count, numpy_generator)
        positions = [anchors, anchors]
        all_starts = [anchors, anchors]
        for delta in pos_deltas:
            positive = anchors + delta
            all_starts.append(positive)
            positions.append(positive)
        groups = np.tile(np.arange(count), 2 + len(pos_deltas))
        raw = _sample_windows(
            matrix, np.concatenate(all_starts), context_length, device)
        standardized, finite = _standardize(raw)
        view1, mask1 = _augment(
            standardized[:count], finite[:count], torch_generator,
            noise=noise, scale=scale, time_mask=time_mask)
        view2, mask2 = _augment(
            standardized[count:2 * count], finite[count:2 * count], torch_generator,
            noise=noise, scale=scale, time_mask=time_mask)
        values = torch.cat([view1, view2, standardized[2 * count:]], 0)
        masks = torch.cat([mask1, mask2, finite[2 * count:]], 0)
        embedded = _reg_embeddings(base, values, masks)
        projected = F.normalize(projection(embedded), dim=1)
        return _weighted_temporal_contrastive(
            projected,
            torch.as_tensor(groups, device=device),
            torch.ones(len(groups), dtype=torch.bool, device=device),
            torch.as_tensor(np.concatenate(positions), device=device),
            temperature,
            far_min,
        )

    def state_loss_for(
            matrices,
            pools,
            count,
            numpy_generator,
            torch_generator,
            *,
            fixed_stream=None,
    ):
        chosen_streams = (
            [fixed_stream] * count
            if fixed_stream is not None
            else [
                stream_keys[int(numpy_generator.integers(len(stream_keys)))]
                for _ in range(count)
            ]
        )
        windows = []
        for timeframe, ticker in chosen_streams:
            starts = pools[timeframe][ticker]
            start = int(starts[int(numpy_generator.integers(len(starts)))])
            windows.append(
                matrices[timeframe][ticker][:, start:start + context_length])
        raw = torch.from_numpy(
            np.stack(windows).astype(np.float32)).to(device)
        if regime_key == "kaufman":
            regime, diagnostic = _kaufman_regime(
                raw[..., -kaufman_length:],
                chop=kaufman_chop,
                trend=kaufman_trend,
            )
        else:
            lower = [
                volatility_thresholds[timeframe][ticker]["compression_max"]
                for timeframe, ticker in chosen_streams
            ]
            upper = [
                volatility_thresholds[timeframe][ticker]["expansion_min"]
                for timeframe, ticker in chosen_streams
            ]
            regime, diagnostic = _range_dynamics_regime(
                raw,
                lower,
                upper,
                dynamics_length=volatility_length,
            )
            retention_regime, retention_efficiency = _kaufman_regime(
                raw[..., -kaufman_length:],
                chop=kaufman_chop,
                trend=kaufman_trend,
            )
        standardized, finite = _standardize(raw)
        view1, mask1 = _augment(
            standardized, finite, torch_generator,
            noise=noise, scale=scale, time_mask=time_mask)
        view2, mask2 = _augment(
            standardized, finite, torch_generator,
            noise=noise, scale=scale, time_mask=time_mask)
        embedded = _reg_embeddings(
            base,
            torch.cat([view1, view2], dim=0),
            torch.cat([mask1, mask2], dim=0),
        )
        projected = F.normalize(projection(embedded), dim=1)
        loss = _regime_supcon(
            projected,
            torch.arange(count, device=device).repeat(2),
            regime.repeat(2),
            temperature,
        )
        if regime_key == "volatility" and kaufman_retention_weight:
            retention_loss = _regime_supcon(
                projected,
                torch.arange(count, device=device).repeat(2),
                retention_regime.repeat(2),
                temperature,
            )
            loss = (
                loss + kaufman_retention_weight * retention_loss
            ) / (1.0 + kaufman_retention_weight)
        if regime_key == "kaufman":
            fractions = {
                "chop": float((regime == 0).float().mean()),
                "uptrend": float((regime == 1).float().mean()),
                "downtrend": float((regime == 2).float().mean()),
                "transition": float((regime == -1).float().mean()),
                "er_mean": float(diagnostic.mean()),
            }
        else:
            fractions = {
                "compression": float((regime == 0).float().mean()),
                "stable": float((regime == 1).float().mean()),
                "expansion": float((regime == 2).float().mean()),
                "range_dynamics_mean": float(diagnostic.mean()),
                "retention_chop": float(
                    (retention_regime == 0).float().mean()),
                "retention_uptrend": float(
                    (retention_regime == 1).float().mean()),
                "retention_downtrend": float(
                    (retention_regime == 2).float().mean()),
                "retention_transition": float(
                    (retention_regime == -1).float().mean()),
                "retention_er_mean": float(retention_efficiency.mean()),
            }
        return loss, fractions

    train_matrices = {
        timeframe: {
            ticker: corpus[timeframe].train[indices]
            for ticker, indices in ticker_indices[timeframe].items()
        }
        for timeframe in timeframes
    }
    validation_matrices = {
        timeframe: {
            ticker: validation_matrix[timeframe][indices]
            for ticker, indices in ticker_indices[timeframe].items()
        }
        for timeframe in timeframes
    }

    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        model.train()
        projection.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        if regime_key == "kaufman":
            train_regime_totals = {
                "chop": 0.0,
                "uptrend": 0.0,
                "downtrend": 0.0,
                "transition": 0.0,
                "er_mean": 0.0,
            }
        elif regime_key == "volatility":
            train_regime_totals = {
                "compression": 0.0,
                "stable": 0.0,
                "expansion": 0.0,
                "range_dynamics_mean": 0.0,
                "retention_chop": 0.0,
                "retention_uptrend": 0.0,
                "retention_downtrend": 0.0,
                "retention_transition": 0.0,
                "retention_er_mean": 0.0,
            }
        else:
            train_regime_totals = {}
        for step in range(steps_per_epoch):
            if regime_key in {"kaufman", "volatility"}:
                loss, train_fractions = state_loss_for(
                    train_matrices,
                    ticker_starts,
                    batch_windows,
                    rng,
                    generator,
                )
                for name in train_regime_totals:
                    train_regime_totals[name] += train_fractions[name]
            else:
                timeframe = timeframes[int(rng.integers(len(timeframes)))]
                item = corpus[timeframe]
                if rng.random() < single_ticker_probability:
                    ticker = ticker_names[int(rng.integers(len(ticker_names)))]
                    train_matrix = item.train[ticker_indices[timeframe][ticker]]
                    starts = ticker_starts[timeframe][ticker]
                else:
                    train_matrix = item.train
                    starts = joint_starts[timeframe]
                loss = temporal_loss_for(
                    train_matrix, starts, batch_windows, rng, generator)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "non-finite contrastive training loss; refusing to update the adapter")
            (loss / gradient_accumulation).backward()
            train_total += float(loss.detach())
            if ((step + 1) % gradient_accumulation == 0
                    or step + 1 == steps_per_epoch):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        projection.eval()
        with torch.no_grad():
            val_timeframe_losses = {}
            for timeframe in timeframes:
                timeframe_ticker_losses = {}
                timeframe_regime_fractions = {}
                for ticker, indices in ticker_indices[timeframe].items():
                    ticker_rng = np.random.default_rng(20260704)
                    ticker_generator = torch.Generator(device=device)
                    ticker_generator.manual_seed(20260704)
                    if regime_key in {"kaufman", "volatility"}:
                        ticker_loss, fractions = state_loss_for(
                            validation_matrices,
                            val_ticker_starts,
                            batch_windows,
                            ticker_rng,
                            ticker_generator,
                            fixed_stream=(timeframe, ticker),
                        )
                        timeframe_ticker_losses[ticker] = float(ticker_loss)
                        timeframe_regime_fractions[ticker] = fractions
                    else:
                        timeframe_ticker_losses[ticker] = float(
                            temporal_loss_for(
                                validation_matrix[timeframe][indices],
                                val_ticker_starts[timeframe][ticker],
                                batch_windows,
                                ticker_rng,
                                ticker_generator,
                            ))
                row = {
                    "single_ticker_losses": timeframe_ticker_losses,
                    "mean_single_ticker_loss": float(
                        np.mean(list(timeframe_ticker_losses.values()))),
                }
                if regime_key in {"kaufman", "volatility"}:
                    row["regime_fractions"] = timeframe_regime_fractions
                else:
                    joint_rng = np.random.default_rng(20260704)
                    joint_generator = torch.Generator(device=device)
                    joint_generator.manual_seed(20260704)
                    row["joint_loss"] = float(temporal_loss_for(
                        validation_matrix[timeframe],
                        val_joint_starts[timeframe],
                        batch_windows,
                        joint_rng,
                        joint_generator,
                    ))
                val_timeframe_losses[timeframe] = row
            val_single_loss = float(np.mean([
                value["mean_single_ticker_loss"]
                for value in val_timeframe_losses.values()
            ]))
            if regime_key in {"kaufman", "volatility"}:
                val_joint_loss = None
                val_loss = val_single_loss
            else:
                val_joint_loss = float(np.mean([
                    value["joint_loss"]
                    for value in val_timeframe_losses.values()
                ]))
                val_loss = 0.5 * (val_joint_loss + val_single_loss)
        improved = val_loss < best_loss - 1e-6
        if improved:
            best_loss, bad = val_loss, 0
            best_adapter = _adapter_state(model)
        else:
            bad += 1
        history_row = {
            "epoch": epoch,
            "train_loss": train_total / steps_per_epoch,
            "val_loss": val_loss,
            "val_joint_loss": val_joint_loss,
            "val_single_ticker_loss": val_single_loss,
            "val_timeframe_losses": val_timeframe_losses,
            "improved": improved,
        }
        if regime_key in {"kaufman", "volatility"}:
            history_row["train_regime_fractions"] = {
                name: value / steps_per_epoch
                for name, value in train_regime_totals.items()
            }
        history.append(history_row)
        print(
            f"[chronos2-contrastive] ep={epoch} "
            f"train={history[-1]['train_loss']:.5f} val={val_loss:.5f} "
            + (f"{regime_key}_single={val_single_loss:.5f}"
               if regime_key in {"kaufman", "volatility"}
               else f"joint={val_joint_loss:.5f} single={val_single_loss:.5f}")
            + f"{' *' if improved else ''}",
            flush=True,
        )
        _atomic_torch(state_path, {
            "epoch": epoch,
            "adapter": _adapter_state(model),
            "projection": projection.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_adapter": best_adapter,
            "history": history,
            "bad": bad,
            "numpy_rng": rng.bit_generator.state,
            "torch_generator": generator.get_state(),
        })
        if bad >= patience:
            break
    if best_adapter is None:
        raise RuntimeError("contrastive stage never produced a finite validation checkpoint")
    _restore_adapter(model, best_adapter)
    _save_final(model, checkpoint)
    report = {
        "schema": (
            "ffm_chronos2_volatility_contrastive_v1"
            if regime_key == "volatility"
            else "ffm_chronos2_contrastive_v2"),
        "stage": (
            "volatility_contrastive"
            if regime_key == "volatility"
            else "contrastive"),
        "status": "complete",
        "parent": {"path": str(parent), "sha256": tree_sha256(parent)},
        "checkpoint": {"path": str(checkpoint), "sha256": tree_sha256(checkpoint)},
        "data_identity_sha256": _corpus_identity(corpus),
        "config": {
            "timeframes": list(timeframes),
            "regime_key": regime_key,
            "kaufman_chop": kaufman_chop,
            "kaufman_trend": kaufman_trend,
            "kaufman_length": kaufman_length,
            "volatility_contract": (
                "self_supervised_completed_context_range_dynamics"
                if regime_key == "volatility" else None),
            "volatility_length": volatility_length,
            "volatility_lower_quantile": volatility_lower_quantile,
            "volatility_upper_quantile": volatility_upper_quantile,
            "volatility_threshold_samples": volatility_threshold_samples,
            "volatility_thresholds": volatility_thresholds,
            "kaufman_retention_weight": (
                kaufman_retention_weight
                if regime_key == "volatility" else None),
            "context_length": context_length,
            "pos_deltas": list(pos_deltas),
            "far_min": far_min,
            "temperature": temperature,
            "noise": noise,
            "scale": scale,
            "time_mask": time_mask,
            "projection_dim": projection_dim,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "batch_windows": batch_windows,
            "gradient_accumulation": gradient_accumulation,
            "learning_rate": learning_rate,
            "single_ticker_probability": (
                1.0 if regime_key in {"kaufman", "volatility"}
                else single_ticker_probability),
            "seed": seed,
        },
        "best_val_loss": best_loss,
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(out_dir / "report.json", report)
    return report


def train_volatility_contrastive(prepared, **kwargs) -> dict:
    """Train a separate causal, OHLCV-only range-dynamics SSL adapter."""
    if "regime_key" in kwargs:
        raise TypeError(
            "train_volatility_contrastive fixes regime_key='volatility'")
    return train_contrastive(
        prepared,
        regime_key="volatility",
        **kwargs,
    )


VOLUME_STRUCTURE_SUMMARY_SIZE = 15
VOLUME_STRUCTURE_TEMPORAL_SIZE = 12
VOLUME_STRUCTURE_TRAINER_SCHEMA = "ffm_chronos2_volume_structure_trainer_v4"
VOLUME_STRUCTURE_REPORT_SCHEMA = "ffm_chronos2_volume_structure_ssl_v3"
VOLUME_STRUCTURE_STAGED_CHECKPOINT = ".checkpoint.pending"
VOLUME_STRUCTURE_STAGED_REPORT = ".report.complete.pending.json"


def _read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unreadable JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _chronos_base_identity(
        parent: Path,
        base_snapshot: Path,
) -> dict[str, str]:
    """Authenticate the exact Chronos base used beneath a local PEFT adapter."""
    parent = Path(parent).expanduser().resolve()
    base_snapshot = Path(base_snapshot).expanduser().resolve()
    adapter_path = parent / "adapter_config.json"
    weights_path = base_snapshot / "model.safetensors"
    config_path = base_snapshot / "config.json"
    if (
        not parent.is_dir()
        or not adapter_path.is_file()
        or not base_snapshot.is_dir()
        or _HEX40.fullmatch(base_snapshot.name) is None
        or not weights_path.is_file()
        or not config_path.is_file()
    ):
        raise RuntimeError(
            "Volume-Structure SSL requires a complete pinned Chronos-2-small "
            "snapshot named by its 40-character revision")
    adapter = _read_json_object(adapter_path)
    if (
        adapter.get("base_model_name_or_path") != CHRONOS2_MODEL_ID
        or adapter.get("peft_type") != "LORA"
        or adapter.get("revision") not in {None, base_snapshot.name}
    ):
        raise RuntimeError(
            "Volume-Structure parent adapter does not match the pinned base")
    return {
        "model_id": CHRONOS2_MODEL_ID,
        "snapshot_path": str(base_snapshot),
        "revision": base_snapshot.name,
        "weights_sha256": _file_sha256(weights_path),
        "config_sha256": _file_sha256(config_path),
    }


def _authenticate_volume_structure_completion(
        report: Mapping,
        *,
        checkpoint: Path,
        run_identity_sha256: str,
        artifact_path: Path | None = None,
) -> None:
    """Fail closed unless a complete report proves native, head-free learning."""
    artifact_path = Path(checkpoint if artifact_path is None else artifact_path)
    saved_checkpoint = report.get("checkpoint")
    native = report.get("checkpoint_only_validation")
    artifact = report.get("final_artifact_contract")
    config = report.get("config")
    saved_path = (
        saved_checkpoint.get("path")
        if isinstance(saved_checkpoint, Mapping) else None
    )
    if (
        report.get("schema") != VOLUME_STRUCTURE_REPORT_SCHEMA
        or report.get("stage") != "volume_structure_ssl"
        or report.get("status") != "complete"
        or report.get("run_identity_sha256") != run_identity_sha256
        or not isinstance(saved_checkpoint, Mapping)
        or not isinstance(saved_path, str)
        or Path(saved_path).resolve() != Path(checkpoint).resolve()
        or saved_checkpoint.get("sha256") != tree_sha256(artifact_path)
        or not isinstance(native, Mapping)
        or native.get("status") != "pass"
        or native.get("contract")
        != "freshly_reloaded_lora_native_reg_without_temporary_heads"
        or not isinstance(artifact, Mapping)
        or not isinstance(config, Mapping)
    ):
        raise RuntimeError(
            "Volume-Structure complete report lacks authenticated native evidence")
    parent = native.get("parent")
    child = native.get("checkpoint")
    saved_lift = native.get("loss_lift_parent_minus_checkpoint")
    margin = native.get("required_margin")
    if (
        not isinstance(parent, Mapping)
        or not isinstance(parent.get("aggregate"), Mapping)
        or not isinstance(child, Mapping)
        or not isinstance(child.get("aggregate"), Mapping)
        or not isinstance(saved_lift, Mapping)
        or not isinstance(margin, (int, float))
        or not np.isfinite(float(margin))
        or float(margin) < 0.0
    ):
        raise RuntimeError(
            "Volume-Structure complete report has malformed native evidence")
    measured = _validate_native_volume_lift(
        parent["aggregate"], child["aggregate"], margin=float(margin))
    for objective, value in measured.items():
        saved = saved_lift.get(objective)
        if (
            not isinstance(saved, (int, float))
            or not np.isfinite(float(saved))
            or not math.isclose(value, float(saved), abs_tol=1e-9)
        ):
            raise RuntimeError(
                "Volume-Structure complete report native lift drifted")
    selection = config.get("checkpoint_selection")
    base = config.get("base_model")
    saved_files = artifact.get("checkpoint_files")
    checkpoint_files = sorted(
        str(path.relative_to(artifact_path))
        for path in artifact_path.rglob("*")
        if path.is_file()
    )
    if not (
        isinstance(selection, Mapping)
        and selection.get("contract")
        == "gate_feasible_weighted_native_reg_participation_concentration_v1"
        and selection.get("temporary_head_metrics_used") is False
        and isinstance(base, Mapping)
        and base.get("model_id") == CHRONOS2_MODEL_ID
        and isinstance(base.get("revision"), str)
        and _HEX40.fullmatch(base["revision"]) is not None
        and isinstance(base.get("weights_sha256"), str)
        and _HEX64.fullmatch(base["weights_sha256"]) is not None
        and isinstance(base.get("config_sha256"), str)
        and _HEX64.fullmatch(base["config_sha256"]) is not None
        and artifact.get("temporary_heads_in_checkpoint") is False
        and artifact.get("ssl_heads_required_for_inference") is False
        and artifact.get("trainer_state")
        == "discarded_after_successful_checkpoint"
        and artifact.get("inference_requires")
        == ["chronos_base_model", "lora_checkpoint"]
        and isinstance(saved_files, list)
        and sorted(saved_files) == checkpoint_files
        and checkpoint_files
        and not any(
            token in Path(name).name.lower()
            for name in checkpoint_files
            for token in ("head", "decoder", "projection", "trainer")
        )
    ):
        raise RuntimeError(
            "Volume-Structure complete report violates head-free artifact contract")


def _recover_volume_structure_finalization(
        out_dir: Path,
        *,
        run_identity_sha256: str,
        trainer_payload: Mapping | None = None,
) -> dict | None:
    """Complete a previously validated two-phase publish after a crash.

    The pending complete report is written while ``trainer.pt`` still exists.
    Therefore its presence authenticates that checkpoint-only validation passed
    before finalization began. Recovery publishes the exact staged checkpoint,
    removes only the run's resumable trainer state, and exposes that report.
    """
    out_dir = Path(out_dir)
    checkpoint = out_dir / "checkpoint"
    staged_checkpoint = out_dir / VOLUME_STRUCTURE_STAGED_CHECKPOINT
    state_path = out_dir / "trainer.pt"
    state_temporary = state_path.with_name(f".{state_path.name}.tmp")
    report_path = out_dir / "report.json"
    staged_report = out_dir / VOLUME_STRUCTURE_STAGED_REPORT
    if _HEX64.fullmatch(run_identity_sha256) is None:
        raise ValueError("expected Volume-Structure run identity must be SHA-256")
    if not staged_report.is_file():
        if checkpoint.exists() and staged_checkpoint.exists():
            raise RuntimeError(
                "published Volume-Structure run has an extra staged checkpoint")
        if (
            report_path.is_file()
            and checkpoint.is_dir()
            and not state_path.exists()
            and not staged_checkpoint.exists()
        ):
            published = _read_json_object(report_path)
            saved_checkpoint = published.get("checkpoint")
            if (
                published.get("schema") == VOLUME_STRUCTURE_REPORT_SCHEMA
                and published.get("status") == "complete"
                and published.get("run_identity_sha256")
                == run_identity_sha256
                and isinstance(saved_checkpoint, dict)
                and saved_checkpoint.get("sha256") == tree_sha256(checkpoint)
            ):
                _authenticate_volume_structure_completion(
                    published,
                    checkpoint=checkpoint,
                    run_identity_sha256=run_identity_sha256,
                )
                if state_temporary.exists():
                    if not state_temporary.is_file():
                        raise RuntimeError(
                            "Volume-Structure temporary trainer state is not a file")
                    state_temporary.unlink()
                return published
        return None
    if not report_path.is_file():
        raise RuntimeError(
            "Volume-Structure finalization marker exists without report.json")
    finalizing = _read_json_object(report_path)
    complete = _read_json_object(staged_report)
    expected_checkpoint = complete.get("checkpoint")
    expected_path = (
        expected_checkpoint.get("path")
        if isinstance(expected_checkpoint, dict) else None
    )
    expected_complete = deepcopy(finalizing)
    expected_complete["status"] = "complete"
    if (
        finalizing.get("schema") != VOLUME_STRUCTURE_REPORT_SCHEMA
        or complete.get("schema") != VOLUME_STRUCTURE_REPORT_SCHEMA
        or finalizing.get("status") != "finalizing"
        or complete.get("status") != "complete"
        or complete != expected_complete
        or complete.get("run_identity_sha256") != run_identity_sha256
        or not isinstance(expected_checkpoint, dict)
        or not isinstance(expected_path, str)
        or Path(expected_path).resolve() != checkpoint.resolve()
    ):
        raise RuntimeError(
            "Volume-Structure finalization artifacts do not share one identity")
    expected_sha256 = expected_checkpoint.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError(
            "Volume-Structure pending report has no checkpoint identity")
    if checkpoint.exists() and staged_checkpoint.exists():
        raise RuntimeError(
            "both published and staged Volume-Structure checkpoints exist")
    candidate = checkpoint if checkpoint.is_dir() else staged_checkpoint
    if not candidate.is_dir() or tree_sha256(candidate) != expected_sha256:
        raise RuntimeError(
            "Volume-Structure pending checkpoint identity is invalid")
    if candidate == staged_checkpoint and not state_path.is_file():
        raise RuntimeError(
            "staged Volume-Structure checkpoint cannot finalize without trainer state")
    _authenticate_volume_structure_completion(
        complete,
        checkpoint=checkpoint,
        artifact_path=candidate,
        run_identity_sha256=run_identity_sha256,
    )
    if state_path.exists():
        if not state_path.is_file():
            raise RuntimeError("Volume-Structure trainer state is not a file")
        trainer = trainer_payload
        if trainer is None:
            import torch

            try:
                trainer = torch.load(
                    state_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise RuntimeError(
                    "Volume-Structure trainer state is unreadable") from exc
        _validate_volume_structure_resume(
            trainer, run_identity_sha256=run_identity_sha256)
    if state_temporary.exists() and not state_temporary.is_file():
        raise RuntimeError(
            "Volume-Structure temporary trainer state is not a file")
    if candidate == staged_checkpoint:
        staged_checkpoint.replace(checkpoint)
    if state_path.exists():
        state_path.unlink()
    if state_temporary.exists():
        state_temporary.unlink()
    if state_path.exists() or state_temporary.exists():
        raise RuntimeError("Volume-Structure trainer state was not discarded")
    staged_report.replace(report_path)
    published = _read_json_object(report_path)
    if published.get("status") != "complete":
        raise RuntimeError("Volume-Structure completion report was not published")
    return published


def _release_accelerator_cache(device: str) -> None:
    import torch

    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif str(device) == "mps" and hasattr(torch, "mps"):
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()


def _evenly_spaced_starts(
        starts: np.ndarray,
        max_samples: int,
) -> np.ndarray:
    """Select deterministic starts spanning the full supplied chronology."""
    starts = np.asarray(starts, dtype=np.int64)
    if starts.ndim != 1 or not len(starts):
        raise ValueError("starts must be a non-empty one-dimensional array")
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1")
    if len(starts) <= max_samples:
        return starts.copy()
    indices = np.linspace(
        0, len(starts) - 1, num=max_samples, dtype=np.int64)
    selected = starts[indices]
    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("evenly spaced start selection produced duplicates")
    return selected


def _volume_structure_summary(
        window: np.ndarray,
        *,
        price_bins: int = 16,
) -> np.ndarray:
    """Return a scale-free causal OHLCV volume-structure description.

    This is an OHLCV proxy for volume-at-price structure, not an exchange
    volume-profile reconstruction: bars do not reveal where their volume
    traded intrabar.  The layout is four relative-volume quarters, four
    volume-weighted typical-price locations, four relative-range quarters,
    concentration, recent participation, and displacement-volume alignment.
    """
    values = np.asarray(window, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 5 or values.shape[1] < 8:
        raise ValueError("volume-structure windows must have shape [5,L], L>=8")
    if price_bins < 4:
        raise ValueError("price_bins must be >= 4")
    if not np.isfinite(values).all():
        return np.full(VOLUME_STRUCTURE_SUMMARY_SIZE, np.nan, dtype=np.float32)
    _, high, low, close, volume = values
    if (
        np.any(high < low)
        or np.any(volume < 0.0)
        or not np.isfinite(high - low).all()
    ):
        return np.full(VOLUME_STRUCTURE_SUMMARY_SIZE, np.nan, dtype=np.float32)

    quarters = np.array_split(np.arange(values.shape[1]), 4)
    mean_volume = float(np.mean(volume))
    volume_scale = mean_volume if mean_volume > 0.0 else 1.0
    volume_epsilon = volume_scale * 1e-12
    relative_volume = [
        float(np.log((float(np.mean(volume[index])) + volume_epsilon)
                     / (mean_volume + volume_epsilon)))
        for index in quarters
    ]

    price_low = float(np.min(low))
    price_high = float(np.max(high))
    price_width = price_high - price_low
    if not price_width > 0.0:
        price_width = max(abs(price_high), 1.0) * 1e-12
    typical = (high + low + close) / 3.0
    price_locations = []
    for index in quarters:
        weights = volume[index]
        if float(weights.sum()) > 0.0:
            location = float(np.average(typical[index], weights=weights))
        else:
            location = float(np.mean(typical[index]))
        price_locations.append((location - price_low) / price_width)

    ranges = high - low
    mean_range = float(np.mean(ranges))
    range_scale = mean_range if mean_range > 0.0 else 1.0
    range_epsilon = range_scale * 1e-12
    relative_range = [
        float(np.log((float(np.mean(ranges[index])) + range_epsilon)
                     / (mean_range + range_epsilon)))
        for index in quarters
    ]

    normalized_typical = np.clip(
        (typical - price_low) / price_width, 0.0, 1.0)
    histogram, _ = np.histogram(
        normalized_typical,
        bins=price_bins,
        range=(0.0, 1.0),
        weights=volume,
    )
    if float(histogram.sum()) <= 0.0:
        histogram, _ = np.histogram(
            normalized_typical, bins=price_bins, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64)
    probabilities /= probabilities.sum()
    positive = probabilities > 0.0
    entropy_concentration = 1.0 + float(
        np.sum(probabilities[positive] * np.log(probabilities[positive]))
        / np.log(price_bins))
    hhi = float(np.square(probabilities).sum())
    hhi_concentration = (hhi - 1.0 / price_bins) / (1.0 - 1.0 / price_bins)
    concentration = float(np.clip(
        0.5 * (entropy_concentration + hhi_concentration), 0.0, 1.0))

    recent = quarters[-1]
    prior = np.concatenate(quarters[:-1])
    prior_volume = float(np.mean(volume[prior]))
    recent_volume = float(np.mean(volume[recent]))
    participation_scale = (
        prior_volume if prior_volume > 0.0
        else mean_volume if mean_volume > 0.0
        else 1.0)
    participation_epsilon = participation_scale * 1e-12
    participation = float(np.log(
        (recent_volume + participation_epsilon)
        / (prior_volume + participation_epsilon)))

    prior_range = float(np.median(ranges[prior]))
    recent_range = float(np.quantile(ranges[recent], 0.75))
    prior_volume_median = float(np.median(volume[prior]))
    recent_volume_quantile = float(np.quantile(volume[recent], 0.75))
    safe_prior_range = prior_range if prior_range > 0.0 else range_scale
    safe_prior_volume = (
        prior_volume_median if prior_volume_median > 0.0 else volume_scale)
    displacement = float(np.log1p(
        (recent_range / max(safe_prior_range, 1e-12))
        * (recent_volume_quantile / max(safe_prior_volume, 1e-12))))

    result = np.asarray(
        relative_volume
        + price_locations
        + relative_range
        + [concentration, participation, displacement],
        dtype=np.float32,
    )
    if result.shape != (VOLUME_STRUCTURE_SUMMARY_SIZE,):
        raise RuntimeError("volume-structure summary shape drifted")
    return result


def _volume_structure_fit_arrays(
        matrix: np.ndarray,
        starts: np.ndarray,
        *,
        context_length: int,
        max_samples: int,
        price_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = _evenly_spaced_starts(starts, max_samples=max_samples)
    summaries = np.stack([
        _volume_structure_summary(
            matrix[:, int(start):int(start) + context_length],
            price_bins=price_bins,
        )
        for start in selected
    ])
    finite = np.isfinite(summaries).all(axis=1)
    if int(finite.sum()) < 16:
        raise RuntimeError("insufficient finite volume-structure fit samples")
    return selected[finite], summaries[finite]


def _fit_volume_structure_thresholds(
        matrix: np.ndarray,
        starts: np.ndarray,
        *,
        context_length: int,
        max_samples: int = 4096,
        price_bins: int = 16,
) -> dict:
    """Fit pseudo-state cutoffs from representative training windows only."""
    selected, summaries = _volume_structure_fit_arrays(
        matrix,
        starts,
        context_length=context_length,
        max_samples=max_samples,
        price_bins=price_bins,
    )
    return _volume_structure_thresholds_from_arrays(selected, summaries)


def _volume_structure_thresholds_from_arrays(
        selected: np.ndarray,
        summaries: np.ndarray,
) -> dict:
    temporal = summaries[:, :VOLUME_STRUCTURE_TEMPORAL_SIZE].astype(np.float64)
    temporal_feature_mean = temporal.mean(axis=0)
    temporal_feature_std = temporal.std(axis=0)
    temporal_feature_std = np.maximum(temporal_feature_std, 1e-6)
    participation_low, participation_high = np.quantile(
        summaries[:, 13], (0.33, 0.67))
    # Participation is the signed log ratio of recent to prior volume. Keep
    # the quantile-derived extremes, but require their class boundary to retain
    # its absolute falling (<0) versus rising (>0) meaning when a stream is
    # skewed toward one side. The downstream state-count gate still fails
    # closed when the training data contains no examples on either side.
    participation_zero_margin = 1e-6
    participation_low = min(
        float(participation_low), -participation_zero_margin)
    participation_high = max(
        float(participation_high), participation_zero_margin)
    result = {
        "fit_samples": int(len(summaries)),
        "fit_start_min": int(selected.min()),
        "fit_start_max": int(selected.max()),
        "participation_low": participation_low,
        "participation_high": participation_high,
        "participation_zero_margin": participation_zero_margin,
        "concentration_low": float(np.quantile(summaries[:, 12], 0.33)),
        "concentration_high": float(np.quantile(summaries[:, 12], 0.67)),
        "displacement_low": float(np.quantile(summaries[:, 14], 0.50)),
        "displacement_high": float(np.quantile(summaries[:, 14], 0.75)),
        "temporal_mean": float(temporal.mean()),
        "temporal_std": float(max(temporal.std(), 1e-6)),
        "temporal_feature_mean": temporal_feature_mean.tolist(),
        "temporal_feature_std": temporal_feature_std.tolist(),
    }
    for prefix in ("participation", "concentration", "displacement"):
        if not result[f"{prefix}_low"] < result[f"{prefix}_high"]:
            raise RuntimeError(
                f"degenerate {prefix} pseudo-state thresholds in training data")
    if not result["participation_low"] < 0.0 < result["participation_high"]:
        raise RuntimeError(
            "participation training states must span falling and rising volume")
    return result


def _volume_structure_states(summary: np.ndarray, thresholds: Mapping) -> np.ndarray:
    """Map one summary to low/high states, leaving the middle ambiguous."""
    summary = np.asarray(summary)
    if summary.shape != (VOLUME_STRUCTURE_SUMMARY_SIZE,):
        raise ValueError("volume-structure summary shape mismatch")
    states = []
    for name, index in (
        ("participation", 13),
        ("concentration", 12),
        ("displacement", 14),
    ):
        value = float(summary[index])
        states.append(
            0 if value <= float(thresholds[f"{name}_low"])
            else 2 if value >= float(thresholds[f"{name}_high"])
            else -1)
    return np.asarray(states, dtype=np.int64)


def _native_reg_volume_structure_losses(
        embeddings,
        instance,
        states,
        *,
        temperature: float,
        require_all: bool,
):
    """Parameter-free state losses applied directly to native Chronos REG."""
    import torch
    import torch.nn.functional as F

    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("volume-structure states must have shape [B,3]")
    if embeddings.ndim != 2 or len(embeddings) != 2 * len(states):
        raise ValueError("native REG embeddings must contain two views per state row")
    if instance.shape != (len(embeddings),):
        raise ValueError("native REG instance IDs have the wrong shape")
    losses, eligible = {}, {}
    normalized = F.normalize(embeddings, dim=1)
    for column, objective in enumerate(
            ("participation", "concentration", "displacement")):
        labels = states[:, column].repeat(2)
        valid = labels >= 0
        eligible[objective] = int(valid.sum().item() // 2)
        active = (
            int(valid.sum()) >= 4
            and int(torch.unique(labels[valid]).numel()) >= 2
        )
        if require_all and not active:
            raise RuntimeError(
                f"native REG batch lost the {objective} objective")
        losses[objective] = (
            _regime_supcon(
                normalized[valid],
                instance[valid],
                labels[valid],
                temperature,
            )
            if active else None
        )
    return losses, eligible


def _validate_native_volume_lift(
        parent: Mapping[str, float],
        child: Mapping[str, float],
        *,
        margin: float,
) -> dict[str, float]:
    """Fail closed unless native participation and concentration both improve."""
    if margin < 0.0:
        raise ValueError("native promotion margin must be nonnegative")
    lift = {}
    for objective in ("participation", "concentration"):
        before = float(parent[objective])
        after = float(child[objective])
        if not np.isfinite(before) or not np.isfinite(after):
            raise RuntimeError(
                f"non-finite native {objective} checkpoint evidence")
        lift[objective] = before - after
        if lift[objective] <= margin:
            raise RuntimeError(
                f"native {objective} did not improve beyond the parent checkpoint")
    return lift


def _volume_structure_run_identity(
        *,
        parent_sha256: str,
        data_identity_sha256: str,
        config: Mapping,
) -> str:
    """Bind resumable state to the exact parent, corpus, and objective."""
    payload = {
        "schema": VOLUME_STRUCTURE_TRAINER_SCHEMA,
        "parent_sha256": str(parent_sha256),
        "data_identity_sha256": str(data_identity_sha256),
        "config": dict(config),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_volume_structure_resume(
        saved: Mapping,
        *,
        run_identity_sha256: str,
) -> None:
    if (
        saved.get("schema") != VOLUME_STRUCTURE_TRAINER_SCHEMA
        or saved.get("run_identity_sha256") != run_identity_sha256
    ):
        raise RuntimeError(
            "volume-structure trainer identity does not match this run")


def _volume_structure_standardize(raw, visible_time=None):
    """Normalize OHLC together and volume separately using visible values only."""
    import torch

    if raw.ndim != 3 or raw.shape[1] != 5:
        raise ValueError("raw OHLCV batch must have shape [B,5,L]")
    finite = torch.isfinite(raw)
    if visible_time is None:
        visible_time = torch.ones(
            (raw.shape[0], raw.shape[-1]), dtype=torch.bool, device=raw.device)
    visible = finite & visible_time[:, None, :]
    if (visible.sum(dim=-1) == 0).any():
        raise RuntimeError("volume-structure corruption hid an entire variate")

    price_visible = visible[:, :4]
    price_count = price_visible.sum(dim=(1, 2), keepdim=True).clamp_min(1)
    price_clean = torch.where(
        price_visible, raw[:, :4], torch.zeros_like(raw[:, :4]))
    price_mean = price_clean.sum(dim=(1, 2), keepdim=True) / price_count
    price_centered = torch.where(
        price_visible, raw[:, :4] - price_mean, torch.zeros_like(raw[:, :4]))
    price_variance = (
        price_centered.square().sum(dim=(1, 2), keepdim=True) / price_count)
    price_scaled = (raw[:, :4] - price_mean) / price_variance.sqrt().clamp_min(1e-5)

    volume = raw[:, 4:5].clamp_min(0.0)
    volume_visible = visible[:, 4:5]
    volume_count = volume_visible.sum(dim=-1, keepdim=True).clamp_min(1)
    visible_volume = torch.where(
        volume_visible, volume, torch.zeros_like(volume))
    volume_scale = (
        visible_volume.sum(dim=-1, keepdim=True) / volume_count).clamp_min(1e-6)
    # Preserve the signed relative-volume path before Chronos performs its own
    # instance normalization. The objective distinguishes falling versus
    # rising participation; it does not claim to retain absolute volume units.
    volume_epsilon = volume_scale * 1e-6
    volume_scaled = torch.log(
        (volume + volume_epsilon) / (volume_scale + volume_epsilon))
    standardized = torch.cat([price_scaled, volume_scaled], dim=1).clamp(-10.0, 10.0)
    standardized = torch.where(finite, standardized, torch.zeros_like(standardized))
    return standardized, finite, visible


def _volume_structure_patch_mask(
        batch: int,
        n_patches: int,
        mask_ratio: float,
        generator,
        device: str,
):
    """Return exact-count patch masks with at least one hidden and visible patch."""
    import torch

    if n_patches < 2:
        raise ValueError("volume-structure reconstruction needs at least two patches")
    masked = min(n_patches - 1, max(1, int(round(n_patches * mask_ratio))))
    scores = torch.rand(
        (batch, n_patches), device=device, generator=generator)
    indices = scores.topk(masked, dim=1).indices
    result = torch.zeros(
        (batch, n_patches), dtype=torch.bool, device=device)
    result.scatter_(1, indices, True)
    return result


def _volume_structure_encoder_input(standardized, visible):
    """Hide artificial targets from Chronos InstanceNorm with actual NaNs."""
    import torch

    if standardized.shape != visible.shape or standardized.ndim != 3:
        raise ValueError("standardized values and visible mask must share [B,C,L]")
    flat = standardized.reshape(-1, standardized.shape[-1])
    visible_flat = visible.reshape(-1, visible.shape[-1])
    context = torch.where(
        visible_flat, flat, torch.full_like(flat, float("nan")))
    return context, visible_flat


def _volume_structure_augment(values, finite, generator, *, noise, scale):
    """Apply OHLC-safe common price perturbations and separate volume noise."""
    import torch

    output = values.clone()
    if scale:
        factor = 1.0 + scale * (
            2.0 * torch.rand(
                (values.shape[0], 1, 1), device=values.device,
                generator=generator) - 1.0)
        output = output * factor
    if noise:
        price_noise = noise * torch.randn(
            (values.shape[0], 1, values.shape[-1]),
            device=values.device,
            generator=generator,
        )
        volume_noise = noise * torch.randn(
            (values.shape[0], 1, values.shape[-1]),
            device=values.device,
            generator=generator,
        )
        output[:, :4] = output[:, :4] + price_noise
        output[:, 4:5] = output[:, 4:5] + volume_noise
    return output, finite


def _prepare_volume_structure_data(
        prepared,
        *,
        context_length: int,
        threshold_samples: int,
        validation_windows_per_stream: int,
        price_bins: int,
) -> dict:
    """Build train-only pseudo-state pools and fixed chronological validation."""
    corpus = _as_corpus(prepared)
    ticker_indices = {
        timeframe: _ticker_indices(item)
        for timeframe, item in corpus.items()
    }
    streams = {}
    thresholds = {}
    preflight_streams = {}
    for timeframe, indices_by_ticker in ticker_indices.items():
        thresholds[timeframe] = {}
        preflight_streams[timeframe] = {}
        for ticker, indices in indices_by_ticker.items():
            key = (timeframe, ticker)
            train_matrix = corpus[timeframe].train[indices]
            train_starts = _observable_starts(
                train_matrix,
                context_length,
                min_observations=context_length,
            )
            fit_starts, fit_summaries = _volume_structure_fit_arrays(
                train_matrix,
                train_starts,
                context_length=context_length,
                max_samples=threshold_samples,
                price_bins=price_bins,
            )
            fitted = _volume_structure_thresholds_from_arrays(
                fit_starts, fit_summaries)
            fit_states = np.stack([
                _volume_structure_states(summary, fitted)
                for summary in fit_summaries
            ])
            state_counts = {}
            for column, objective in enumerate(
                    ("participation", "concentration", "displacement")):
                counts = {
                    str(state): int((fit_states[:, column] == state).sum())
                    for state in (0, -1, 2)
                }
                if counts["0"] == 0 or counts["2"] == 0:
                    raise RuntimeError(
                        f"{timeframe}/{ticker} has a degenerate {objective} "
                        "training pool")
                state_counts[objective] = counts

            validation_matrix = corpus[timeframe].validation_matrix[indices]
            validation_starts = _observable_starts(
                validation_matrix,
                context_length,
                min_observations=context_length,
            )
            validation_starts = _evenly_spaced_starts(
                validation_starts,
                max_samples=validation_windows_per_stream,
            )
            validation_summaries = np.stack([
                _volume_structure_summary(
                    validation_matrix[:, int(start):int(start) + context_length],
                    price_bins=price_bins,
                )
                for start in validation_starts
            ])
            if not np.isfinite(validation_summaries).all():
                raise RuntimeError(
                    f"non-finite validation summaries for {timeframe}/{ticker}")
            validation_states = np.stack([
                _volume_structure_states(summary, fitted)
                for summary in validation_summaries
            ])
            temporal_mean = np.asarray(
                fitted["temporal_feature_mean"], dtype=np.float32)
            temporal_std = np.asarray(
                fitted["temporal_feature_std"], dtype=np.float32)
            train_temporal = (
                fit_summaries[:, :VOLUME_STRUCTURE_TEMPORAL_SIZE]
                - temporal_mean) / temporal_std
            validation_temporal = (
                validation_summaries[:, :VOLUME_STRUCTURE_TEMPORAL_SIZE]
                - temporal_mean) / temporal_std
            train_temporal = np.clip(train_temporal, -10.0, 10.0)
            validation_temporal = np.clip(validation_temporal, -10.0, 10.0)
            streams[key] = {
                "train_matrix": train_matrix,
                "train_starts": train_starts,
                "fit_starts": fit_starts,
                "fit_states": fit_states,
                "fit_temporal": train_temporal.astype(np.float32),
                "validation_matrix": validation_matrix,
                "validation_starts": validation_starts,
                "validation_states": validation_states,
                "validation_temporal": validation_temporal.astype(np.float32),
            }
            thresholds[timeframe][ticker] = fitted
            preflight_streams[timeframe][ticker] = {
                "train_available_windows": int(len(train_starts)),
                "train_fit_windows": int(len(fit_starts)),
                "train_fit_start_min": int(fit_starts.min()),
                "train_fit_start_max": int(fit_starts.max()),
                "validation_windows": int(len(validation_starts)),
                "validation_start_min": int(validation_starts.min()),
                "validation_start_max": int(validation_starts.max()),
                "train_state_counts": state_counts,
            }
    return {
        "corpus": corpus,
        "data_identity_sha256": _corpus_identity(corpus),
        "streams": streams,
        "thresholds": thresholds,
        "preflight_streams": preflight_streams,
    }


def preflight_volume_structure_ssl(
        prepared,
        *,
        context_length: int = 256,
        threshold_samples: int = 4096,
        validation_windows_per_stream: int = 16,
        price_bins: int = 16,
) -> dict:
    """Exercise all Volume-Structure data/target contracts without loading a model."""
    data = _prepare_volume_structure_data(
        prepared,
        context_length=context_length,
        threshold_samples=threshold_samples,
        validation_windows_per_stream=validation_windows_per_stream,
        price_bins=price_bins,
    )
    return {
        "schema": "ffm_chronos2_volume_structure_preflight_v1",
        "status": "pass",
        "data_identity_sha256": data["data_identity_sha256"],
        "data_contracts": {
            timeframe: item.report
            for timeframe, item in data["corpus"].items()
        },
        "context_length": context_length,
        "threshold_samples": threshold_samples,
        "validation_windows_per_stream": validation_windows_per_stream,
        "price_bins": price_bins,
        "streams": data["preflight_streams"],
        "thresholds": data["thresholds"],
    }


def train_volume_structure_ssl(
        prepared,
        *,
        parent: str | Path,
        base_snapshot: str | Path,
        out_dir: str | Path,
        device: str = "mps",
        context_length: int = 256,
        epochs: int = 60,
        steps_per_epoch: int = 100,
        batch_windows: int = 32,
        gradient_accumulation: int = 1,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.05,
        patience: int = 8,
        projection_dim: int = 128,
        head_auxiliary_weight: float = 0.25,
        temperature: float = 0.10,
        noise: float = 0.02,
        scale: float = 0.10,
        mask_ratio: float = 0.25,
        threshold_samples: int = 4096,
        validation_windows_per_stream: int = 16,
        price_bins: int = 16,
        reconstruction_weight: float = 1.0,
        participation_weight: float = 1.0,
        concentration_weight: float = 1.0,
        displacement_weight: float = 1.0,
        temporal_weight: float = 0.5,
        adapter_retention_weight: float = 0.1,
        native_promotion_margin: float = 1e-4,
        log_every_steps: int = 10,
        seed: int = 0,
        resume: bool = False,
) -> dict:
    """Train an isolated causal OHLCV Volume-Structure LoRA adapter.

    Participation and price-volume concentration are distinct self-supervised
    tasks trained directly on the native Chronos REG representation. Temporary
    projection heads and decoders are auxiliary optimization scaffolding only;
    best-checkpoint selection and the final parent/child lift gate never use
    them. Reconstruction predicts genuinely hidden Chronos patches using
    normalization fitted only from visible values. The temporal task
    reconstructs ordered quarter-level volume, price-location, and range
    descriptors; it does not claim to label a proprietary trading sequence or
    a literal intrabar exchange profile.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from importlib.metadata import PackageNotFoundError, version

    if context_length < 32:
        raise ValueError("context_length must be >= 32")
    if batch_windows < 6:
        raise ValueError(
            "batch_windows must be >= 6 to balance three contrastive objectives")
    if epochs < 1 or steps_per_epoch < 1 or patience < 1:
        raise ValueError("epochs, steps_per_epoch, and patience must be >= 1")
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be >= 1")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    if threshold_samples < 16:
        raise ValueError("threshold_samples must be >= 16")
    if validation_windows_per_stream < 4:
        raise ValueError("validation_windows_per_stream must be >= 4")
    if projection_dim < 2 or temperature <= 0.0:
        raise ValueError("projection_dim must be >= 2 and temperature must be > 0")
    if head_auxiliary_weight < 0.0:
        raise ValueError("head_auxiliary_weight must be nonnegative")
    if noise < 0.0 or scale < 0.0 or scale >= 1.0:
        raise ValueError("noise must be >= 0 and scale must be in [0,1)")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be nonnegative")
    if native_promotion_margin < 0.0:
        raise ValueError("native_promotion_margin must be nonnegative")
    objective_weights = {
        "reconstruction": float(reconstruction_weight),
        "participation": float(participation_weight),
        "concentration": float(concentration_weight),
        "displacement": float(displacement_weight),
        "temporal_structure": float(temporal_weight),
        "adapter_retention": float(adapter_retention_weight),
    }
    native_selection_weights = {
        "participation_native": float(participation_weight),
        "concentration_native": float(concentration_weight),
    }
    if any(value < 0.0 for value in objective_weights.values()):
        raise ValueError("volume-structure objective weights must be nonnegative")
    if not any(value > 0.0 for value in objective_weights.values()):
        raise ValueError("at least one volume-structure objective weight must be positive")
    if (
        objective_weights["participation"] <= 0.0
        or objective_weights["concentration"] <= 0.0
    ):
        raise ValueError(
            "native participation and concentration objectives must stay active")
    if log_every_steps < 1:
        raise ValueError("log_every_steps must be >= 1")

    out_dir, parent = Path(out_dir), Path(parent)
    base_snapshot = Path(base_snapshot)
    checkpoint = out_dir / "checkpoint"
    staged_checkpoint = out_dir / VOLUME_STRUCTURE_STAGED_CHECKPOINT
    state_path = out_dir / "trainer.pt"
    state_temporary = state_path.with_name(f".{state_path.name}.tmp")
    report_path = out_dir / "report.json"
    staged_report = out_dir / VOLUME_STRUCTURE_STAGED_REPORT
    preflight_path = out_dir / "preflight.json"
    if not parent.is_dir():
        raise RuntimeError(f"volume-structure parent adapter is missing: {parent}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _prepare_volume_structure_data(
        prepared,
        context_length=context_length,
        threshold_samples=threshold_samples,
        validation_windows_per_stream=validation_windows_per_stream,
        price_bins=price_bins,
    )
    corpus = data["corpus"]
    timeframes = tuple(corpus)
    parent_sha256 = tree_sha256(parent)
    data_identity_sha256 = data["data_identity_sha256"]
    base_identity = _chronos_base_identity(parent, base_snapshot)

    def package_version(distribution: str) -> str:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return "not-installed"

    run_config = {
        "objective_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "chronos_version": package_version("chronos-forecasting"),
        "peft_version": package_version("peft"),
        "base_model": base_identity,
        "timeframes": list(timeframes),
        "stream_sampler": "uniform_stream_balanced_extreme_states",
        "context_length": context_length,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_windows": batch_windows,
        "gradient_accumulation": gradient_accumulation,
        "learning_rate": learning_rate,
        "adapter_weight_decay": 0.0,
        "temporary_head_weight_decay": weight_decay,
        "patience": patience,
        "projection_dim": projection_dim,
        "head_auxiliary_weight": head_auxiliary_weight,
        "temperature": temperature,
        "noise": noise,
        "scale": scale,
        "mask_ratio": mask_ratio,
        "threshold_samples": threshold_samples,
        "threshold_sampling": "evenly_spaced_full_training_chronology",
        "validation_windows_per_stream": validation_windows_per_stream,
        "validation_sampling": "fixed_evenly_spaced_per_stream",
        "price_bins": price_bins,
        "objective_weights": objective_weights,
        "checkpoint_selection": {
            "contract": (
                "gate_feasible_weighted_native_reg_"
                "participation_concentration_v1"),
            "weights": native_selection_weights,
            "temporary_head_metrics_used": False,
        },
        "native_promotion_margin": native_promotion_margin,
        "log_every_steps": log_every_steps,
        "seed": seed,
        "objective_schema": {
            "summary": "scale_free_volume_price_range_quarters_v2",
            "reconstruction": "visible_normalized_masked_chronos_patches_v1",
            "participation": (
                "native_reg_plus_projection_recent_vs_prior_volume_v2"),
            "concentration": (
                "native_reg_plus_projection_typical_price_entropy_hhi_v2"),
            "displacement": (
                "native_reg_plus_projection_range_volume_cooccurrence_v2"),
            "temporal_structure": "ordered_quarter_descriptor_reconstruction_v1",
            "retention": "l2_sp_parent_adapter_anchor_v1",
        },
    }
    run_identity_sha256 = _volume_structure_run_identity(
        parent_sha256=parent_sha256,
        data_identity_sha256=data_identity_sha256,
        config=run_config,
    )
    resume_payload = None
    if resume and state_path.is_file():
        try:
            resume_payload = torch.load(
                state_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(
                "volume-structure trainer state is unreadable") from exc
        _validate_volume_structure_resume(
            resume_payload, run_identity_sha256=run_identity_sha256)
    if resume:
        recovered = _recover_volume_structure_finalization(
            out_dir,
            run_identity_sha256=run_identity_sha256,
            trainer_payload=resume_payload,
        )
        if recovered is not None:
            return recovered
    if checkpoint.exists():
        raise RuntimeError(
            f"completed volume-structure checkpoint already exists: {checkpoint}")
    if report_path.exists():
        pending = _read_json_object(report_path)
        can_rebuild_finalization = (
            resume_payload is not None
            and staged_checkpoint.is_dir()
            and not staged_report.exists()
            and pending.get("schema") == VOLUME_STRUCTURE_REPORT_SCHEMA
            and pending.get("status") == "finalizing"
            and pending.get("run_identity_sha256") == run_identity_sha256
        )
        if not can_rebuild_finalization:
            raise RuntimeError(
                f"completed volume-structure report already exists: {report_path}")
        # The complete marker was never serialized, so no success decision was
        # published. The authenticated trainer can deterministically rebuild it.
        report_path.unlink()
    if resume and resume_payload is None:
        raise RuntimeError(
            f"--resume requested but trainer state is missing: {state_path}")
    if state_path.exists() and not resume:
        raise RuntimeError(
            f"incomplete trainer state exists; pass --resume or use a new out-dir: {state_path}")
    if resume_payload is not None and state_temporary.exists():
        if not state_temporary.is_file():
            raise RuntimeError(
                "volume-structure temporary trainer state is not a file")
        state_temporary.unlink()
    if staged_report.exists():
        raise RuntimeError(
            f"unrecoverable Volume-Structure finalization report exists: {staged_report}")
    if staged_checkpoint.exists():
        if resume_payload is None or not staged_checkpoint.is_dir():
            raise RuntimeError(
                f"unrecoverable staged Volume-Structure checkpoint exists: {staged_checkpoint}")
        # A crash before the two-phase publish marker leaves a reconstructible
        # directory. Delete only this exact staging path after trainer identity
        # authentication, then rebuild it from the retained best adapter.
        shutil.rmtree(staged_checkpoint)
    _atomic_json(preflight_path, {
        "schema": "ffm_chronos2_volume_structure_preflight_v1",
        "status": "pass",
        "run_identity_sha256": run_identity_sha256,
        "parent": {"path": str(parent), "sha256": parent_sha256},
        "data_identity_sha256": data_identity_sha256,
        "data_contracts": {
            timeframe: item.report for timeframe, item in corpus.items()
        },
        "config": run_config,
        "streams": data["preflight_streams"],
        "thresholds": data["thresholds"],
    })

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model, base = _load_trainable_adapter(
        parent,
        device,
        base_revision=base_identity["revision"],
        base_snapshot=base_snapshot,
    )
    patch_size = int(base.chronos_config.input_patch_size)
    patch_stride = int(base.chronos_config.input_patch_stride)
    if patch_stride != patch_size:
        raise RuntimeError(
            "Volume-Structure reconstruction requires non-overlapping Chronos patches")
    if not bool(base.chronos_config.use_reg_token):
        raise RuntimeError("Volume-Structure SSL requires the Chronos REG token")
    if context_length % patch_size:
        raise ValueError("context_length must be divisible by Chronos-2 patch size")
    n_patches = context_length // patch_size
    if n_patches < 2:
        raise ValueError("context_length must contain at least two Chronos-2 patches")
    embedding_dim = int(base.model_dim)

    def projection_head():
        return nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    projection_heads = nn.ModuleDict({
        name: projection_head()
        for name in ("participation", "concentration", "displacement")
    }).to(device)
    patch_decoder = nn.Linear(embedding_dim, patch_size).to(device)
    temporal_decoder = nn.Sequential(
        nn.LayerNorm(embedding_dim),
        nn.Linear(embedding_dim, max(32, projection_dim)),
        nn.GELU(),
        nn.Linear(max(32, projection_dim), VOLUME_STRUCTURE_TEMPORAL_SIZE),
    ).to(device)

    adapter_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not adapter_parameters:
        raise RuntimeError("volume-structure stage found no trainable LoRA parameters")
    initial_adapter = {
        name: parameter.detach().clone()
        for name, parameter in adapter_parameters
    }
    parameters = [parameter for _, parameter in adapter_parameters]
    head_parameters = list(projection_heads.parameters())
    head_parameters += list(patch_decoder.parameters())
    head_parameters += list(temporal_decoder.parameters())
    parameters += head_parameters
    optimizer = torch.optim.AdamW([
        {
            "params": [parameter for _, parameter in adapter_parameters],
            "weight_decay": 0.0,
        },
        {"params": head_parameters, "weight_decay": weight_decay},
    ], lr=learning_rate)

    streams = data["streams"]
    stream_keys = tuple(streams)
    objective_names = ("participation", "concentration", "displacement")
    class_pools = {
        objective: {
            state: tuple(
                key for key in stream_keys
                if np.any(streams[key]["fit_states"][:, column] == state)
            )
            for state in (0, 2)
        }
        for column, objective in enumerate(objective_names)
    }
    if any(
        not keys
        for objective in class_pools.values()
        for keys in objective.values()
    ):
        raise RuntimeError("a balanced volume-structure state pool is empty")

    def balanced_record(key, row):
        stream = streams[key]
        start = int(stream["fit_starts"][row])
        raw = stream["train_matrix"][:, start:start + context_length]
        return (
            raw,
            stream["fit_states"][row],
            stream["fit_temporal"][row],
        )

    def uniform_record(key, row):
        stream = streams[key]
        start = int(stream["train_starts"][row])
        raw = stream["train_matrix"][:, start:start + context_length]
        target = _volume_structure_summary(raw, price_bins=price_bins)
        fitted = data["thresholds"][key[0]][key[1]]
        temporal_mean = np.asarray(
            fitted["temporal_feature_mean"], dtype=np.float32)
        temporal_std = np.asarray(
            fitted["temporal_feature_std"], dtype=np.float32)
        temporal = (
            target[:VOLUME_STRUCTURE_TEMPORAL_SIZE] - temporal_mean
        ) / temporal_std
        temporal = np.clip(temporal, -10.0, 10.0)
        return raw, _volume_structure_states(target, fitted), temporal

    def sample_training_batch():
        chosen = []
        for column, objective in enumerate(objective_names):
            for state in (0, 2):
                eligible = class_pools[objective][state]
                key = eligible[int(rng.integers(len(eligible)))]
                candidates = np.flatnonzero(
                    streams[key]["fit_states"][:, column] == state)
                row = int(candidates[int(rng.integers(len(candidates)))])
                chosen.append(balanced_record(key, row))
        while len(chosen) < batch_windows:
            key = stream_keys[int(rng.integers(len(stream_keys)))]
            row = int(rng.integers(len(streams[key]["train_starts"])))
            chosen.append(uniform_record(key, row))
        order = rng.permutation(len(chosen))
        chosen = [chosen[int(index)] for index in order]
        return (
            np.stack([item[0] for item in chosen]).astype(np.float32),
            np.stack([item[1] for item in chosen]).astype(np.int64),
            np.stack([item[2] for item in chosen]).astype(np.float32),
        )

    def validation_batch(key):
        stream = streams[key]
        raw = np.stack([
            stream["validation_matrix"][
                :, int(start):int(start) + context_length]
            for start in stream["validation_starts"]
        ]).astype(np.float32)
        return raw, stream["validation_states"], stream["validation_temporal"]

    def native_checkpoint_validation(validation_base):
        """Measure raw REG state geometry without any temporary SSL module."""
        validation_generator = torch.Generator(device=device)
        validation_generator.manual_seed(seed + 20_260_802)
        per_stream = {}
        with torch.no_grad():
            for timeframe, ticker in stream_keys:
                raw_np, states_np, _ = validation_batch((timeframe, ticker))
                raw = torch.from_numpy(raw_np).to(device)
                states = torch.from_numpy(
                    np.asarray(states_np, dtype=np.int64)).to(device)
                standardized, finite, _ = _volume_structure_standardize(raw)
                first, first_mask = _volume_structure_augment(
                    standardized,
                    finite,
                    validation_generator,
                    noise=noise,
                    scale=scale,
                )
                second, second_mask = _volume_structure_augment(
                    standardized,
                    finite,
                    validation_generator,
                    noise=noise,
                    scale=scale,
                )
                embeddings = _reg_embeddings(
                    validation_base,
                    torch.cat([first, second], dim=0),
                    torch.cat([first_mask, second_mask], dim=0),
                )
                instance = torch.arange(
                    len(raw), device=device).repeat(2)
                losses, eligible = _native_reg_volume_structure_losses(
                    embeddings,
                    instance,
                    states,
                    temperature=temperature,
                    require_all=False,
                )
                per_stream[f"{ticker}@{timeframe}"] = {
                    **{
                        name: (
                            None if value is None
                            else float(value.detach().cpu()))
                        for name, value in losses.items()
                    },
                    "eligible": eligible,
                }
        aggregate, worst_streams = {}, {}
        for objective in objective_names:
            eligible_losses = {
                name: metrics[objective]
                for name, metrics in per_stream.items()
                if metrics[objective] is not None
            }
            if not eligible_losses:
                raise RuntimeError(
                    f"native checkpoint validation cannot measure {objective}")
            aggregate[objective] = float(np.mean(list(eligible_losses.values())))
            worst = max(eligible_losses, key=eligible_losses.get)
            worst_streams[objective] = {
                "stream": worst,
                "loss": float(eligible_losses[worst]),
                "eligible_streams": int(len(eligible_losses)),
            }
        return {
            "contract": (
                "fixed_augmented_native_chronos_reg_without_ssl_heads"),
            "seed": seed + 20_260_802,
            "aggregate": aggregate,
            "worst_streams": worst_streams,
            "per_stream": per_stream,
        }

    model.eval()
    parent_native_validation = native_checkpoint_validation(base)
    model.train()

    def adapter_drift():
        values = [
            (parameter - initial_adapter[name]).square().mean()
            for name, parameter in adapter_parameters
        ]
        return torch.stack(values).mean()

    def loss_for(
            raw_np,
            states_np,
            temporal_np,
            *,
            loss_generator,
            native_generator=None,
            require_all_contrastive,
    ):
        raw = torch.from_numpy(raw_np).to(device)
        states = torch.from_numpy(np.asarray(states_np, dtype=np.int64)).to(device)
        temporal_target = torch.from_numpy(
            np.asarray(temporal_np, dtype=np.float32)).to(device)
        batch = raw.shape[0]

        patch_mask = _volume_structure_patch_mask(
            batch,
            n_patches,
            mask_ratio,
            loss_generator,
            device,
        )
        time_mask = patch_mask.repeat_interleave(patch_size, dim=1)
        standardized_masked, finite, visible = _volume_structure_standardize(
            raw, visible_time=~time_mask)
        encoder_context, visible_flat = _volume_structure_encoder_input(
            standardized_masked, visible)
        # Chronos-2 InstanceNorm runs before its context_mask is applied. Hide
        # artificial targets from that normalization with NaN; passing finite
        # values plus a False mask would leak target statistics into visible
        # patch tokens.
        groups = torch.arange(batch, device=device).repeat_interleave(5)
        outputs, loc_scale, _, context_patches = base.encode(
            context=encoder_context,
            context_mask=visible_flat,
            group_ids=groups,
            num_output_patches=1,
        )
        if context_patches != n_patches:
            raise RuntimeError(
                "Chronos-2 returned an unexpected context patch count")
        if not isinstance(loc_scale, tuple) or len(loc_scale) != 2:
            raise RuntimeError("Chronos-2 encode did not return visible loc/scale")
        loc, normalization_scale = loc_scale
        normalized_target = (
            standardized_masked.reshape(batch * 5, context_length) - loc
        ) / normalization_scale.clamp_min(1e-5)
        if bool(getattr(base.chronos_config, "use_arcsinh", False)):
            normalized_target = torch.arcsinh(normalized_target)
        normalized_target = normalized_target.reshape(
            batch, 5, context_length).clamp(-10.0, 10.0)
        reconstruction = patch_decoder(
            outputs[0][:, :context_patches, :]).reshape(
                batch, 5, context_patches * patch_size)[..., :context_length]
        selected = finite & time_mask[:, None, :]
        if not selected.any():
            raise RuntimeError("masked reconstruction selected no finite targets")
        reconstruction_loss = F.smooth_l1_loss(
            reconstruction[selected], normalized_target[selected])

        standardized, full_finite, _ = _volume_structure_standardize(raw)
        augmentation_generator = (
            loss_generator if native_generator is None else native_generator)
        view1, mask1 = _volume_structure_augment(
            standardized,
            full_finite,
            augmentation_generator,
            noise=noise,
            scale=scale,
        )
        view2, mask2 = _volume_structure_augment(
            standardized,
            full_finite,
            augmentation_generator,
            noise=noise,
            scale=scale,
        )
        embeddings = _reg_embeddings(
            base,
            torch.cat([view1, view2], dim=0),
            torch.cat([mask1, mask2], dim=0),
        )
        instance = torch.arange(batch, device=device).repeat(2)
        component_tensors = {"reconstruction": reconstruction_loss}
        native_losses, eligible = _native_reg_volume_structure_losses(
            embeddings,
            instance,
            states,
            temperature=temperature,
            require_all=require_all_contrastive,
        )
        for column, objective in enumerate(objective_names):
            labels = states[:, column].repeat(2)
            valid = labels >= 0
            native_loss = native_losses[objective]
            if native_loss is not None:
                projected = F.normalize(
                    projection_heads[objective](embeddings[valid]), dim=1)
                projected_loss = _regime_supcon(
                    projected,
                    instance[valid],
                    labels[valid],
                    temperature,
                )
                component_tensors[f"{objective}_native"] = native_loss
                component_tensors[f"{objective}_head"] = projected_loss
                # Native REG geometry is primary. The head can stabilize
                # gradients, but it cannot carry most of the optimized loss.
                component_tensors[objective] = (
                    native_loss + head_auxiliary_weight * projected_loss)
            else:
                component_tensors[objective] = None
                component_tensors[f"{objective}_native"] = None
                component_tensors[f"{objective}_head"] = None

        temporal_prediction = temporal_decoder(embeddings[:batch])
        component_tensors["temporal_structure"] = F.smooth_l1_loss(
            temporal_prediction, temporal_target)
        component_tensors["adapter_retention"] = adapter_drift()

        total = component_tensors["reconstruction"] * reconstruction_weight
        for objective, weight in (
            ("participation", participation_weight),
            ("concentration", concentration_weight),
            ("displacement", displacement_weight),
            ("temporal_structure", temporal_weight),
            ("adapter_retention", adapter_retention_weight),
        ):
            value = component_tensors[objective]
            if value is not None:
                total = total + value * weight
        components = {
            name: (None if value is None else float(value.detach()))
            for name, value in component_tensors.items()
        }
        components["embedding_std"] = float(
            embeddings[:batch].detach().std(dim=0, unbiased=False).mean())
        return total, components, eligible

    def capture_global_rng():
        state = {"cpu": torch.get_rng_state()}
        if str(device).startswith("cuda"):
            state["device"] = torch.cuda.get_rng_state(torch.device(device))
        elif (
            str(device) == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "get_rng_state")
        ):
            state["device"] = torch.mps.get_rng_state()
        return state

    def restore_global_rng(state):
        torch.set_rng_state(state["cpu"])
        if "device" not in state:
            return
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state(state["device"], torch.device(device))
        elif (
            str(device) == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "set_rng_state")
        ):
            torch.mps.set_rng_state(state["device"])

    best_loss, best_adapter, best_epoch = math.inf, None, None
    history, start_epoch, bad = [], 0, 0
    if resume:
        saved = resume_payload
        if saved is None:
            raise RuntimeError("validated volume-structure resume state is missing")
        _restore_adapter(model, saved["adapter"])
        projection_heads.load_state_dict(saved["projection_heads"])
        patch_decoder.load_state_dict(saved["patch_decoder"])
        temporal_decoder.load_state_dict(saved["temporal_decoder"])
        optimizer.load_state_dict(saved["optimizer"])
        best_loss = float(saved["best_loss"])
        best_adapter = saved["best_adapter"]
        best_epoch = (
            None if saved["best_epoch"] is None
            else int(saved["best_epoch"])
        )
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1
        bad = int(saved["bad"])
        rng.bit_generator.state = saved["numpy_rng"]
        generator.set_state(saved["torch_generator"])
        restore_global_rng(saved["global_torch_rng"])
        resume_payload = None
        del saved
        gc.collect()

    started = time.monotonic()
    component_keys = (
        "reconstruction",
        "participation",
        "participation_native",
        "participation_head",
        "concentration",
        "concentration_native",
        "concentration_head",
        "displacement",
        "displacement_native",
        "displacement_head",
        "temporal_structure",
        "adapter_retention",
        "embedding_std",
    )
    epoch_stop = start_epoch if bad >= patience else epochs
    for epoch in range(start_epoch, epoch_stop):
        model.train()
        projection_heads.train()
        patch_decoder.train()
        temporal_decoder.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, **{name: 0.0 for name in component_keys}}
        for step in range(steps_per_epoch):
            raw_np, states_np, temporal_np = sample_training_batch()
            loss, components, _ = loss_for(
                raw_np,
                states_np,
                temporal_np,
                loss_generator=generator,
                require_all_contrastive=True,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "non-finite volume-structure loss; refusing adapter update")
            (loss / gradient_accumulation).backward()
            totals["loss"] += float(loss.detach())
            for name in component_keys:
                value = components[name]
                if value is None:
                    raise RuntimeError(
                        f"training objective {name} became inactive")
                totals[name] += value
            if (
                (step + 1) % gradient_accumulation == 0
                or step + 1 == steps_per_epoch
            ):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % log_every_steps == 0 or step + 1 == steps_per_epoch:
                print(
                    f"[chronos2-volume-structure] ep={epoch} "
                    f"step={step + 1}/{steps_per_epoch} "
                    f"loss={totals['loss'] / (step + 1):.5f}",
                    flush=True,
                )

        model.eval()
        projection_heads.eval()
        patch_decoder.eval()
        temporal_decoder.eval()
        validation_generator = torch.Generator(device=device)
        validation_generator.manual_seed(seed + 20_260_801)
        native_validation_generator = torch.Generator(device=device)
        native_validation_generator.manual_seed(seed + 20_260_802)
        with torch.no_grad():
            val_per_stream = {}
            for timeframe, ticker in stream_keys:
                raw_np, states_np, temporal_np = validation_batch(
                    (timeframe, ticker))
                _, components, eligible = loss_for(
                    raw_np,
                    states_np,
                    temporal_np,
                    loss_generator=validation_generator,
                    native_generator=native_validation_generator,
                    require_all_contrastive=False,
                )
                val_per_stream[f"{ticker}@{timeframe}"] = {
                    **components,
                    "eligible": eligible,
                }

        val_components = {}
        for name in component_keys:
            values = [
                stream[name]
                for stream in val_per_stream.values()
                if stream[name] is not None
            ]
            if not values:
                raise RuntimeError(
                    f"fixed validation bank cannot measure {name}")
            val_components[name] = float(np.mean(values))
        # Checkpoint selection uses only native REG geometry. Temporary heads,
        # reconstruction decoders, and retention penalties remain diagnostics
        # and training auxiliaries; they cannot make an adapter look learned.
        val_loss = sum(
            weight * val_components[name]
            for name, weight in native_selection_weights.items()
        )
        worst_streams = {}
        for name in (
            "participation_native",
            "concentration_native",
            "displacement_native",
        ):
            eligible_values = {
                key: value[name]
                for key, value in val_per_stream.items()
                if value[name] is not None
            }
            worst_key = max(eligible_values, key=eligible_values.get)
            worst_streams[name] = {
                "stream": worst_key,
                "loss": float(eligible_values[worst_key]),
                "eligible_streams": int(len(eligible_values)),
            }

        try:
            _validate_native_volume_lift(
                parent_native_validation["aggregate"],
                {
                    objective: val_components[f"{objective}_native"]
                    for objective in ("participation", "concentration")
                },
                margin=native_promotion_margin,
            )
            native_gate_feasible = True
        except RuntimeError:
            native_gate_feasible = False
        improved = (
            native_gate_feasible and val_loss < best_loss - 1e-6)
        if improved:
            best_loss, bad = val_loss, 0
            best_adapter = _adapter_state(model)
            best_epoch = epoch
        elif best_adapter is not None:
            bad += 1
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / steps_per_epoch,
            "train_components": {
                name: totals[name] / steps_per_epoch
                for name in component_keys
            },
            "val_loss": val_loss,
            "val_components": val_components,
            "val_worst_streams": worst_streams,
            "val_per_stream": val_per_stream,
            "native_gate_feasible": native_gate_feasible,
            "improved": improved,
            "bad_epochs": bad,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        print(
            f"[chronos2-volume-structure] ep={epoch} "
            f"train={row['train_loss']:.5f} val={val_loss:.5f} "
            f"part_native={val_components['participation_native']:.5f} "
            f"conc_native={val_components['concentration_native']:.5f}"
            f" feasible={int(native_gate_feasible)}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        _atomic_torch(state_path, {
            "schema": VOLUME_STRUCTURE_TRAINER_SCHEMA,
            "run_identity_sha256": run_identity_sha256,
            "epoch": epoch,
            "adapter": _adapter_state(model),
            "projection_heads": projection_heads.state_dict(),
            "patch_decoder": patch_decoder.state_dict(),
            "temporal_decoder": temporal_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_adapter": best_adapter,
            "best_epoch": best_epoch,
            "history": history,
            "bad": bad,
            "numpy_rng": rng.bit_generator.state,
            "torch_generator": generator.get_state(),
            "global_torch_rng": capture_global_rng(),
        })
        if bad >= patience:
            break

    if best_adapter is None or best_epoch is None:
        raise RuntimeError(
            "volume-structure SSL never produced a finite validation checkpoint")
    if tree_sha256(parent) != parent_sha256:
        raise RuntimeError(
            "volume-structure parent adapter changed during training")
    if _chronos_base_identity(parent, base_snapshot) != base_identity:
        raise RuntimeError(
            "pinned Chronos base snapshot changed during training")
    if not state_path.is_file():
        raise RuntimeError(
            "volume-structure trainer state disappeared before finalization")
    _restore_adapter(model, best_adapter)
    _save_final(model, staged_checkpoint)
    forbidden_artifact_tokens = ("head", "decoder", "projection", "trainer")
    checkpoint_files = sorted(
        str(item.relative_to(staged_checkpoint))
        for item in staged_checkpoint.rglob("*")
        if item.is_file()
    )
    if not checkpoint_files or any(
        token in Path(name).name.lower()
        for name in checkpoint_files
        for token in forbidden_artifact_tokens
    ):
        raise RuntimeError(
            "final Volume-Structure checkpoint contains temporary training artifacts")
    adapter_config_path = staged_checkpoint / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise RuntimeError("final Volume-Structure LoRA config is missing")
    adapter_config = _read_json_object(adapter_config_path)
    base_model_name = adapter_config.get("base_model_name_or_path")
    if (
        base_model_name != CHRONOS2_MODEL_ID
        or adapter_config.get("revision") != base_identity["revision"]
    ):
        raise RuntimeError(
            "final Volume-Structure LoRA does not declare its pinned Chronos base")

    # Drop every temporary training module before reloading the staged LoRA.
    # The following validation therefore cannot accidentally consult a head or
    # decoder that will not exist in downstream inference.
    del loss_for
    del adapter_drift
    del optimizer
    del projection_heads
    del patch_decoder
    del temporal_decoder
    del parameters
    del head_parameters
    del adapter_parameters
    del base
    del model
    _release_accelerator_cache(device)

    checkpoint_model, checkpoint_base = _load_trainable_adapter(
        staged_checkpoint,
        device,
        base_revision=base_identity["revision"],
        base_snapshot=base_snapshot,
    )
    checkpoint_model.eval()
    child_native_validation = native_checkpoint_validation(checkpoint_base)
    del checkpoint_base
    del checkpoint_model
    _release_accelerator_cache(device)
    native_lift = _validate_native_volume_lift(
        parent_native_validation["aggregate"],
        child_native_validation["aggregate"],
        margin=native_promotion_margin,
    )
    staged_checkpoint_sha256 = tree_sha256(staged_checkpoint)
    report = {
        "schema": VOLUME_STRUCTURE_REPORT_SCHEMA,
        "stage": "volume_structure_ssl",
        "status": "finalizing",
        "run_identity_sha256": run_identity_sha256,
        "parent": {"path": str(parent), "sha256": parent_sha256},
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": staged_checkpoint_sha256,
        },
        "data_identity_sha256": data_identity_sha256,
        "data_contracts": {
            timeframe: item.report for timeframe, item in corpus.items()
        },
        "config": run_config,
        "thresholds": data["thresholds"],
        "preflight_streams": data["preflight_streams"],
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "checkpoint_only_validation": {
            "status": "pass",
            "contract": (
                "freshly_reloaded_lora_native_reg_without_temporary_heads"),
            "parent": parent_native_validation,
            "checkpoint": child_native_validation,
            "loss_lift_parent_minus_checkpoint": native_lift,
            "required_margin": native_promotion_margin,
        },
        "final_artifact_contract": {
            "checkpoint_files": checkpoint_files,
            "temporary_heads_in_checkpoint": False,
            "temporary_training_modules": [
                "projection_heads",
                "patch_decoder",
                "temporal_decoder",
                "optimizer",
            ],
            "ssl_heads_required_for_inference": False,
            "trainer_state": "discarded_after_successful_checkpoint",
            "base_model_name_or_path": base_model_name,
            "inference_requires": ["chronos_base_model", "lora_checkpoint"],
        },
        "history": history,
        "retention_gate": {
            "status": "required_before_promotion",
            "contract": (
                "matched parent-versus-child Probe Atlas on identical pools, "
                "per-stream metrics, and controls"
            ),
        },
        "limitations": [
            (
                "OHLCV bars support a causal volume-at-typical-price proxy, "
                "not literal intrabar exchange volume profile"
            ),
            (
                "temporary SSL heads are discarded; the native REG gate "
                "confirms direct learning, while Probe Atlas remains the "
                "required transfer and retention test"
            ),
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    complete_report = deepcopy(report)
    complete_report["status"] = "complete"

    # Two-phase publication keeps trainer.pt on every pre-success failure. The
    # completed report is fully serialized first, then the staged adapter is
    # published, exactly trainer.pt is removed, and only then is status=complete
    # exposed. _recover_volume_structure_finalization handles crashes between
    # those final atomic operations.
    _atomic_json(report_path, report)
    _atomic_json(staged_report, complete_report)
    staged_checkpoint.replace(checkpoint)
    if tree_sha256(checkpoint) != staged_checkpoint_sha256:
        raise RuntimeError(
            "published Volume-Structure checkpoint identity drifted")
    state_path.unlink()
    if state_temporary.exists():
        if not state_temporary.is_file():
            raise RuntimeError(
                "volume-structure temporary trainer state is not a file")
        state_temporary.unlink()
    if state_path.exists() or state_temporary.exists():
        raise RuntimeError(
            "volume-structure trainer state was not discarded")
    staged_report.replace(report_path)
    if staged_checkpoint.exists() or staged_report.exists():
        raise RuntimeError(
            "Volume-Structure finalization left a staged artifact behind")
    return complete_report


BALANCED_KAUFMAN_TRAINER_SCHEMA = (
    "ffm_chronos2_balanced_kaufman_trainer_v1")
BALANCED_KAUFMAN_REPORT_SCHEMA = "ffm_chronos2_balanced_kaufman_ssl_v1"
BALANCED_KAUFMAN_PREFLIGHT_SCHEMA = (
    "ffm_chronos2_balanced_kaufman_preflight_v1")
BALANCED_KAUFMAN_STAGED_CHECKPOINT = ".checkpoint.pending"
BALANCED_KAUFMAN_STAGED_REPORT = ".report.complete.pending.json"


def _kaufman_efficiency_scores(
        matrix: np.ndarray,
        starts: np.ndarray,
        *,
        context_length: int,
        kaufman_length: int,
        close_channel: int = 3,
) -> np.ndarray:
    """Vectorized causal ER from the final completed bars of each context."""
    matrix = np.asarray(matrix)
    starts = np.asarray(starts, dtype=np.int64)
    if matrix.ndim != 2 or not 0 <= close_channel < matrix.shape[0]:
        raise ValueError("Kaufman matrix or close channel is invalid")
    if starts.ndim != 1 or not len(starts):
        raise ValueError("Kaufman starts must be a non-empty vector")
    if not 2 <= kaufman_length <= context_length:
        raise ValueError("kaufman_length must be in [2, context_length]")
    if int(starts.min()) < 0 or int(starts.max()) + context_length > matrix.shape[1]:
        raise ValueError("Kaufman starts exceed the supplied matrix")
    close = np.asarray(matrix[close_channel], dtype=np.float64)
    first = starts + context_length - kaufman_length
    last = starts + context_length - 1
    if not (
        np.isfinite(close[first]).all()
        and np.isfinite(close[last]).all()
    ):
        raise RuntimeError("Kaufman endpoints contain non-finite values")
    differences = np.abs(np.diff(close))
    invalid_prefix = np.empty(len(close), dtype=np.int64)
    invalid_prefix[0] = 0
    np.cumsum(~np.isfinite(differences), dtype=np.int64,
              out=invalid_prefix[1:])
    if np.any((invalid_prefix[last] - invalid_prefix[first]) != 0):
        raise RuntimeError("Kaufman paths contain non-finite values")
    prefix = np.empty(len(close), dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(
        np.where(np.isfinite(differences), differences, 0.0),
        dtype=np.float64,
        out=prefix[1:],
    )
    path = prefix[last] - prefix[first]
    displacement = np.abs(close[last] - close[first])
    efficiency = np.divide(
        displacement,
        np.maximum(path, 1e-12),
        out=np.zeros_like(displacement),
    )
    efficiency = np.clip(efficiency, 0.0, 1.0).astype(np.float32)
    if not np.isfinite(efficiency).all():
        raise RuntimeError("Kaufman efficiency produced non-finite values")
    return efficiency


def _kaufman_binary_states(
        efficiency: np.ndarray,
        *,
        chop: float,
        trend: float,
) -> np.ndarray:
    """Direction-agnostic chop=0/trend=2 with an ambiguous middle state."""
    if not 0.0 <= chop < trend <= 1.0:
        raise ValueError("Kaufman thresholds must satisfy 0 <= chop < trend <= 1")
    efficiency = np.asarray(efficiency, dtype=np.float32)
    if not np.isfinite(efficiency).all():
        raise ValueError("Kaufman efficiency must be finite")
    states = np.full(efficiency.shape, -1, dtype=np.int64)
    states[efficiency <= chop] = 0
    states[efficiency >= trend] = 2
    return states


def _reg_embeddings_concat(base, windows, finite):
    """Return the exact ordered five-variate native REG concatenation [B,5D]."""
    import torch

    if windows.ndim != 3 or windows.shape[1] != 5 or finite.shape != windows.shape:
        raise ValueError("balanced Kaufman input must have shape [B,5,L]")
    batch, channels, _ = windows.shape
    flat = windows.reshape(batch * channels, -1)
    mask = finite.reshape(batch * channels, -1)
    groups = torch.arange(batch, device=windows.device).repeat_interleave(channels)
    outputs, _, _, context_patches = base.encode(
        context=flat,
        context_mask=mask,
        group_ids=groups,
        num_output_patches=1,
    )
    reg = outputs[0][:, context_patches, :]
    if reg.ndim != 2 or reg.shape[0] != batch * 5:
        raise RuntimeError("Chronos REG output shape drifted")
    return reg.reshape(batch, 5 * reg.shape[-1])


def _native_balanced_kaufman_metrics(
        embeddings,
        instance,
        states,
        *,
        temperature: float,
        require_both: bool,
):
    """Native direction-agnostic regime loss and projection-free geometry."""
    import torch
    import torch.nn.functional as F

    if embeddings.ndim != 2 or len(embeddings) % 2:
        raise ValueError("native embeddings must contain two views per instance")
    batch = len(embeddings) // 2
    if states.shape != (batch,) or instance.shape != (2 * batch,):
        raise ValueError("balanced Kaufman state or instance shape mismatch")
    labels = states.repeat(2)
    valid = labels >= 0
    classes = torch.unique(labels[valid])
    class_counts = {
        str(state): int((states == state).sum().item()) for state in (0, 2)
    }
    active = (
        int(valid.sum()) >= 8
        and int(classes.numel()) == 2
        and all(value >= 2 for value in class_counts.values())
    )
    if require_both and not active:
        raise RuntimeError("balanced Kaufman batch lost chop or trend support")
    if not active:
        return None
    normalized = F.normalize(embeddings[valid], dim=1)
    valid_instance = instance[valid]
    valid_labels = labels[valid]
    loss = _regime_supcon(
        normalized, valid_instance, valid_labels, temperature)
    similarity = normalized @ normalized.T
    eye = torch.eye(
        len(normalized), dtype=torch.bool, device=normalized.device)
    distinct_instance = valid_instance[:, None] != valid_instance[None, :]
    same = (
        (valid_labels[:, None] == valid_labels[None, :])
        & distinct_instance & ~eye)
    different = valid_labels[:, None] != valid_labels[None, :]
    if not same.any() or not different.any():
        if require_both:
            raise RuntimeError("balanced Kaufman geometry has no valid pairs")
        return None
    margin = similarity[same].mean() - similarity[different].mean()
    embedding_std = normalized[:batch].std(
        dim=0, unbiased=False).mean()
    return {
        "loss": loss,
        "margin": margin,
        "embedding_std": embedding_std,
        "class_counts": class_counts,
    }


def _validate_native_balanced_kaufman_lift(
        parent: Mapping[str, float],
        child: Mapping[str, float],
        *,
        margin: float,
) -> dict[str, float]:
    """Require lower native loss and a larger native regime margin."""
    if not np.isfinite(float(margin)) or margin < 0.0:
        raise ValueError("native promotion margin must be finite and nonnegative")
    values = {
        "loss": float(parent["loss"]) - float(child["loss"]),
        "margin": float(child["margin"]) - float(parent["margin"]),
    }
    for name, value in values.items():
        if not np.isfinite(value) or value <= margin:
            raise RuntimeError(
                f"native balanced Kaufman {name} did not improve beyond parent")
    for side, metrics in (("parent", parent), ("checkpoint", child)):
        std = float(metrics["embedding_std"])
        if not np.isfinite(std) or std <= 0.0:
            raise RuntimeError(f"{side} native balanced Kaufman embedding collapsed")
    return values


def _prepare_balanced_kaufman_data(
        prepared,
        *,
        context_length: int,
        kaufman_length: int,
        kaufman_chop: float,
        kaufman_trend: float,
        validation_windows_per_state: int,
) -> dict:
    """Build exact fixed-threshold, balanced train and validation state pools."""
    if validation_windows_per_state < 2:
        raise ValueError("validation_windows_per_state must be >= 2")
    if not 2 <= kaufman_length <= context_length:
        raise ValueError("kaufman_length must be in [2, context_length]")
    if not 0.0 <= kaufman_chop < kaufman_trend <= 1.0:
        raise ValueError("Kaufman thresholds must satisfy 0 <= chop < trend <= 1")
    corpus = _as_corpus(prepared)
    ticker_indices = {
        timeframe: _ticker_indices(item)
        for timeframe, item in corpus.items()
    }
    streams, preflight_streams = {}, {}
    for timeframe, indices_by_ticker in ticker_indices.items():
        preflight_streams[timeframe] = {}
        for ticker, indices in indices_by_ticker.items():
            key = (timeframe, ticker)
            train_matrix = corpus[timeframe].train[indices]
            train_starts = _observable_starts(
                train_matrix,
                context_length,
                min_observations=context_length,
            )
            train_efficiency = _kaufman_efficiency_scores(
                train_matrix,
                train_starts,
                context_length=context_length,
                kaufman_length=kaufman_length,
            )
            train_states = _kaufman_binary_states(
                train_efficiency,
                chop=kaufman_chop,
                trend=kaufman_trend,
            )
            train_state_starts = {
                state: train_starts[train_states == state] for state in (0, 2)
            }
            if any(len(values) < 2 for values in train_state_starts.values()):
                raise RuntimeError(
                    f"{timeframe}/{ticker} lacks fixed-threshold Kaufman train support")

            validation_matrix = corpus[timeframe].validation_matrix[indices]
            validation_starts = _observable_starts(
                validation_matrix,
                context_length,
                min_observations=context_length,
            )
            validation_efficiency = _kaufman_efficiency_scores(
                validation_matrix,
                validation_starts,
                context_length=context_length,
                kaufman_length=kaufman_length,
            )
            validation_states_all = _kaufman_binary_states(
                validation_efficiency,
                chop=kaufman_chop,
                trend=kaufman_trend,
            )
            validation_state_starts = {}
            for state in (0, 2):
                candidates = validation_starts[validation_states_all == state]
                if len(candidates) < validation_windows_per_state:
                    raise RuntimeError(
                        f"{timeframe}/{ticker} lacks fixed-threshold Kaufman "
                        "validation support")
                validation_state_starts[state] = _evenly_spaced_starts(
                    candidates,
                    max_samples=validation_windows_per_state,
                )
                if len(validation_state_starts[state]) != validation_windows_per_state:
                    raise RuntimeError(
                        "balanced Kaufman validation bank size drifted")
            selected_validation = np.concatenate([
                validation_state_starts[0], validation_state_starts[2]])
            order = np.argsort(selected_validation, kind="stable")
            selected_validation = selected_validation[order]
            selected_states = np.concatenate([
                np.zeros(len(validation_state_starts[0]), dtype=np.int64),
                np.full(len(validation_state_starts[2]), 2, dtype=np.int64),
            ])[order]
            streams[key] = {
                "train_matrix": train_matrix,
                "train_state_starts": train_state_starts,
                "validation_matrix": validation_matrix,
                "validation_starts": selected_validation,
                "validation_states": selected_states,
            }
            preflight_streams[timeframe][ticker] = {
                "train_available_windows": int(len(train_starts)),
                "train_state_counts": {
                    "chop": int(len(train_state_starts[0])),
                    "trend": int(len(train_state_starts[2])),
                    "transition": int((train_states == -1).sum()),
                },
                "validation_available_windows": int(len(validation_starts)),
                "validation_state_counts": {
                    "chop": int((validation_states_all == 0).sum()),
                    "trend": int((validation_states_all == 2).sum()),
                    "transition": int((validation_states_all == -1).sum()),
                },
                "validation_selected": {
                    "chop": int(len(validation_state_starts[0])),
                    "trend": int(len(validation_state_starts[2])),
                },
            }
    aggregate_state_counts = {
        "chop": int(sum(
            len(stream["train_state_starts"][0]) for stream in streams.values())),
        "trend": int(sum(
            len(stream["train_state_starts"][2]) for stream in streams.values())),
    }
    return {
        "corpus": corpus,
        "data_identity_sha256": _corpus_identity(corpus),
        "streams": streams,
        "preflight_streams": preflight_streams,
        "aggregate_state_counts": aggregate_state_counts,
    }


def _authenticate_balanced_kaufman_parent(
        *,
        parent: Path,
        parent_report: Path,
        base_snapshot: Path,
        data_identity_sha256: str,
) -> tuple[dict, dict, dict]:
    """Authenticate the completed Volume checkpoint, report, base, and corpus."""
    parent = Path(parent).expanduser().resolve()
    parent_report = Path(parent_report).expanduser().resolve()
    if not parent_report.is_file():
        raise RuntimeError("balanced Kaufman Volume parent report is missing")
    report = _read_json_object(parent_report)
    run_identity = report.get("run_identity_sha256")
    volume_source_parent = report.get("parent")
    volume_data_identity = report.get("data_identity_sha256")
    volume_config = report.get("config")
    if not isinstance(run_identity, str) or _HEX64.fullmatch(run_identity) is None:
        raise RuntimeError("Volume parent report has no valid run identity")
    if not (
        isinstance(volume_source_parent, Mapping)
        and isinstance(volume_source_parent.get("sha256"), str)
        and _HEX64.fullmatch(volume_source_parent["sha256"]) is not None
        and isinstance(volume_data_identity, str)
        and _HEX64.fullmatch(volume_data_identity) is not None
        and isinstance(volume_config, Mapping)
        and _volume_structure_run_identity(
            parent_sha256=volume_source_parent["sha256"],
            data_identity_sha256=volume_data_identity,
            config=volume_config,
        ) == run_identity
    ):
        raise RuntimeError("Volume parent report run identity is not reproducible")
    _authenticate_volume_structure_completion(
        report,
        checkpoint=parent,
        run_identity_sha256=run_identity,
    )
    base_identity = _chronos_base_identity(parent, Path(base_snapshot))
    config = report.get("config")
    if not isinstance(config, Mapping) or config.get("base_model") != base_identity:
        raise RuntimeError("Volume parent report base identity drifted")
    parent_identity = {"path": str(parent), "sha256": tree_sha256(parent)}
    report_identity = {
        "path": str(parent_report),
        "sha256": _file_sha256(parent_report),
        "schema": report["schema"],
        "stage": report["stage"],
        "run_identity_sha256": run_identity,
        "data_identity_sha256": volume_data_identity,
    }
    return parent_identity, report_identity, base_identity


def preflight_balanced_kaufman_ssl(
        prepared,
        *,
        parent: str | Path,
        parent_report: str | Path,
        base_snapshot: str | Path,
        context_length: int = 256,
        kaufman_length: int = 64,
        kaufman_chop: float = 0.25,
        kaufman_trend: float = 0.50,
        validation_windows_per_state: int = 16,
) -> dict:
    """Validate fixed-threshold class support and authenticated Volume lineage."""
    if (
        context_length != 256
        or kaufman_length != 64
        or not math.isclose(kaufman_chop, 0.25, abs_tol=1e-12)
        or not math.isclose(kaufman_trend, 0.50, abs_tol=1e-12)
    ):
        raise ValueError(
            "balanced Kaufman v1 fixes context=256, ER64, chop=.25, trend=.50")
    corpus = _as_corpus(prepared)
    data_identity_sha256 = _corpus_identity(corpus)
    parent_identity, report_identity, base_identity = (
        _authenticate_balanced_kaufman_parent(
            parent=Path(parent),
            parent_report=Path(parent_report),
            base_snapshot=Path(base_snapshot),
            data_identity_sha256=data_identity_sha256,
        )
    )
    data = _prepare_balanced_kaufman_data(
        corpus,
        context_length=context_length,
        kaufman_length=kaufman_length,
        kaufman_chop=kaufman_chop,
        kaufman_trend=kaufman_trend,
        validation_windows_per_state=validation_windows_per_state,
    )
    if data["data_identity_sha256"] != data_identity_sha256:
        raise RuntimeError("balanced Kaufman corpus identity drifted during preflight")
    if min(data["aggregate_state_counts"].values()) < 16:
        raise RuntimeError(
            "balanced Kaufman preflight lacks 16 distinct records per state")
    return {
        "schema": BALANCED_KAUFMAN_PREFLIGHT_SCHEMA,
        "status": "pass",
        "parent": parent_identity,
        "parent_report": report_identity,
        "base_model": base_identity,
        "data_identity_sha256": data["data_identity_sha256"],
        "data_contracts": {
            timeframe: item.report for timeframe, item in data["corpus"].items()
        },
        "config": {
            "context_length": context_length,
            "kaufman_length": kaufman_length,
            "kaufman_chop": kaufman_chop,
            "kaufman_trend": kaufman_trend,
            "direction_contract": "direction_agnostic",
            "validation_windows_per_state": validation_windows_per_state,
        },
        "streams": data["preflight_streams"],
        "aggregate_state_counts": data["aggregate_state_counts"],
    }


def _balanced_kaufman_run_identity(
        *,
        parent_sha256: str,
        parent_report_sha256: str,
        parent_report_run_identity_sha256: str,
        data_identity_sha256: str,
        config: Mapping,
) -> str:
    payload = {
        "schema": BALANCED_KAUFMAN_TRAINER_SCHEMA,
        "parent_sha256": parent_sha256,
        "parent_report_sha256": parent_report_sha256,
        "parent_report_run_identity_sha256": (
            parent_report_run_identity_sha256),
        "data_identity_sha256": data_identity_sha256,
        "config": dict(config),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_balanced_kaufman_resume(
        saved: Mapping,
        *,
        run_identity_sha256: str,
) -> None:
    if (
        saved.get("schema") != BALANCED_KAUFMAN_TRAINER_SCHEMA
        or saved.get("run_identity_sha256") != run_identity_sha256
    ):
        raise RuntimeError("balanced Kaufman trainer identity does not match this run")


def _sample_balanced_kaufman_batch(
        streams: Mapping,
        *,
        batch_windows: int,
        context_length: int,
        rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-stream batch with equal unique chop and trend records."""
    if batch_windows < 4 or batch_windows % 2:
        raise ValueError("balanced Kaufman batch_windows must be even and >= 4")
    stream_keys = tuple(streams)
    if not stream_keys:
        raise ValueError("balanced Kaufman streams cannot be empty")
    chosen, states, seen = [], [], set()
    per_state = batch_windows // 2
    for state in (0, 2):
        attempts = 0
        while len(states) < (per_state if state == 0 else batch_windows):
            attempts += 1
            if attempts > batch_windows * 100:
                raise RuntimeError("unable to sample distinct balanced Kaufman records")
            key = stream_keys[int(rng.integers(len(stream_keys)))]
            pool = streams[key]["train_state_starts"][state]
            start = int(pool[int(rng.integers(len(pool)))])
            identity = (key, start)
            if identity in seen:
                continue
            seen.add(identity)
            chosen.append(
                streams[key]["train_matrix"][:, start:start + context_length])
            states.append(state)
    order = rng.permutation(batch_windows)
    return (
        np.stack([chosen[int(index)] for index in order]).astype(np.float32),
        np.asarray([states[int(index)] for index in order], dtype=np.int64),
    )


def _balanced_kaufman_validation_batch(
        stream: Mapping,
        *,
        context_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.stack([
        stream["validation_matrix"][:, int(start):int(start) + context_length]
        for start in stream["validation_starts"]
    ]).astype(np.float32)
    states = np.asarray(stream["validation_states"], dtype=np.int64)
    return raw, states


def _balanced_kaufman_native_validation(
        base,
        data: Mapping,
        *,
        device: str,
        context_length: int,
        temperature: float,
        noise: float,
        scale: float,
        seed: int,
) -> dict:
    """Fixed projection-free validation on balanced per-stream banks."""
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    per_stream = {}
    with torch.no_grad():
        for timeframe, ticker in data["streams"]:
            raw_np, states_np = _balanced_kaufman_validation_batch(
                data["streams"][(timeframe, ticker)],
                context_length=context_length,
            )
            raw = torch.from_numpy(raw_np).to(device)
            states = torch.from_numpy(states_np).to(device)
            standardized, finite, _ = _volume_structure_standardize(raw)
            first, first_mask = _volume_structure_augment(
                standardized, finite, generator, noise=noise, scale=scale)
            second, second_mask = _volume_structure_augment(
                standardized, finite, generator, noise=noise, scale=scale)
            embeddings = _reg_embeddings_concat(
                base,
                torch.cat([first, second], dim=0),
                torch.cat([first_mask, second_mask], dim=0),
            )
            instance = torch.arange(len(raw), device=device).repeat(2)
            metrics = _native_balanced_kaufman_metrics(
                embeddings,
                instance,
                states,
                temperature=temperature,
                require_both=True,
            )
            if metrics is None:
                raise RuntimeError("balanced Kaufman native validation became inactive")
            per_stream[f"{ticker}@{timeframe}"] = {
                name: float(value.detach().cpu())
                for name, value in metrics.items()
                if name != "class_counts"
            }
            per_stream[f"{ticker}@{timeframe}"]["class_counts"] = (
                metrics["class_counts"])
    aggregate = {
        name: float(np.mean([row[name] for row in per_stream.values()]))
        for name in ("loss", "margin", "embedding_std")
    }
    worst_streams = {}
    for name, choose in (
        ("loss", max),
        ("margin", min),
        ("embedding_std", min),
    ):
        key = choose(per_stream, key=lambda value: per_stream[value][name])
        worst_streams[name] = {
            "stream": key,
            "value": float(per_stream[key][name]),
            "eligible_streams": int(len(per_stream)),
        }
    return {
        "contract": "fixed_balanced_native_chronos_reg_5d_concat_without_ssl_heads",
        "seed": seed,
        "aggregate": aggregate,
        "worst_streams": worst_streams,
        "per_stream": per_stream,
    }


def _validate_balanced_kaufman_native_receipt(
        receipt: Mapping,
        *,
        validation_windows_per_state: int,
) -> tuple[frozenset[str], int]:
    """Validate that a native receipt is internally self-consistent."""
    aggregate = receipt.get("aggregate")
    per_stream = receipt.get("per_stream")
    seed = receipt.get("seed")
    metric_names = ("loss", "margin", "embedding_std")
    if (
        receipt.get("contract")
        != "fixed_balanced_native_chronos_reg_5d_concat_without_ssl_heads"
        or not isinstance(aggregate, Mapping)
        or not isinstance(per_stream, Mapping)
        or not per_stream
        or type(seed) is not int
        or type(validation_windows_per_state) is not int
        or validation_windows_per_state < 2
    ):
        raise RuntimeError("balanced Kaufman native receipt is malformed")
    stream_names = frozenset(per_stream)
    if not all(isinstance(name, str) and name for name in stream_names):
        raise RuntimeError("balanced Kaufman native receipt has invalid streams")
    for stream, row in per_stream.items():
        if not isinstance(row, Mapping):
            raise RuntimeError(
                f"balanced Kaufman native receipt row is malformed for {stream}")
        for name in metric_names:
            value = row.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(float(value))
            ):
                raise RuntimeError(
                    f"balanced Kaufman native metric is invalid for {stream}")
        counts = row.get("class_counts")
        if (
            not isinstance(counts, Mapping)
            or set(counts) != {"0", "2"}
            or any(
                type(counts[state]) is not int
                or counts[state] != validation_windows_per_state
                for state in ("0", "2")
            )
        ):
            raise RuntimeError(
                f"balanced Kaufman native class counts drifted for {stream}")
    for name in metric_names:
        saved = aggregate.get(name)
        measured = float(np.mean([
            float(per_stream[stream][name]) for stream in sorted(stream_names)
        ]))
        if (
            not isinstance(saved, (int, float))
            or isinstance(saved, bool)
            or not np.isfinite(float(saved))
            or not math.isclose(
                float(saved), measured, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise RuntimeError(
                f"balanced Kaufman native aggregate drifted for {name}")
    return stream_names, seed


def _authenticate_balanced_kaufman_completion(
        report: Mapping,
        *,
        checkpoint: Path,
        run_identity_sha256: str,
        artifact_path: Path | None = None,
) -> None:
    """Authenticate a complete head-free balanced Kaufman artifact."""
    artifact_path = Path(checkpoint if artifact_path is None else artifact_path)
    saved_checkpoint = report.get("checkpoint")
    native = report.get("checkpoint_only_validation")
    artifact = report.get("final_artifact_contract")
    config = report.get("config")
    parent = report.get("parent")
    parent_report = report.get("parent_report")
    saved_path = (
        saved_checkpoint.get("path")
        if isinstance(saved_checkpoint, Mapping) else None)
    if (
        report.get("schema") != BALANCED_KAUFMAN_REPORT_SCHEMA
        or report.get("stage") != "balanced_kaufman_ssl"
        or report.get("status") != "complete"
        or report.get("run_identity_sha256") != run_identity_sha256
        or not isinstance(saved_checkpoint, Mapping)
        or not isinstance(saved_path, str)
        or Path(saved_path).resolve() != Path(checkpoint).resolve()
        or saved_checkpoint.get("sha256") != tree_sha256(artifact_path)
        or not isinstance(native, Mapping)
        or native.get("status") != "pass"
        or native.get("contract")
        != "freshly_reloaded_lora_native_reg_without_temporary_heads"
        or not isinstance(artifact, Mapping)
        or not isinstance(config, Mapping)
        or not isinstance(parent, Mapping)
        or not isinstance(parent_report, Mapping)
    ):
        raise RuntimeError("balanced Kaufman completion lacks authenticated evidence")
    parent_path = parent.get("path")
    report_path = parent_report.get("path")
    if (
        not isinstance(parent_path, str)
        or not Path(parent_path).is_dir()
        or parent.get("sha256") != tree_sha256(Path(parent_path))
        or not isinstance(report_path, str)
        or not Path(report_path).is_file()
        or parent_report.get("sha256") != _file_sha256(Path(report_path))
    ):
        raise RuntimeError("balanced Kaufman parent lineage drifted")
    volume_report = _read_json_object(Path(report_path))
    if (
        volume_report.get("schema") != parent_report.get("schema")
        or volume_report.get("stage") != parent_report.get("stage")
        or volume_report.get("run_identity_sha256")
        != parent_report.get("run_identity_sha256")
        or volume_report.get("data_identity_sha256")
        != parent_report.get("data_identity_sha256")
    ):
        raise RuntimeError("balanced Kaufman Volume report identity drifted")
    _authenticate_volume_structure_completion(
        volume_report,
        checkpoint=Path(parent_path),
        run_identity_sha256=parent_report["run_identity_sha256"],
    )
    parent_native = native.get("parent")
    child_native = native.get("checkpoint")
    saved_lift = native.get("loss_lift_parent_minus_checkpoint")
    required_margin = native.get("required_margin")
    if not (
        isinstance(parent_native, Mapping)
        and isinstance(parent_native.get("aggregate"), Mapping)
        and parent_native.get("contract")
        == "fixed_balanced_native_chronos_reg_5d_concat_without_ssl_heads"
        and isinstance(child_native, Mapping)
        and isinstance(child_native.get("aggregate"), Mapping)
        and child_native.get("contract")
        == "fixed_balanced_native_chronos_reg_5d_concat_without_ssl_heads"
        and isinstance(saved_lift, Mapping)
        and isinstance(required_margin, (int, float))
    ):
        raise RuntimeError("balanced Kaufman native evidence is malformed")
    validation_windows_per_state = config.get("validation_windows_per_state")
    parent_streams, parent_seed = _validate_balanced_kaufman_native_receipt(
        parent_native,
        validation_windows_per_state=validation_windows_per_state,
    )
    child_streams, child_seed = _validate_balanced_kaufman_native_receipt(
        child_native,
        validation_windows_per_state=validation_windows_per_state,
    )
    if parent_streams != child_streams or parent_seed != child_seed:
        raise RuntimeError(
            "balanced Kaufman parent/checkpoint validation contracts drifted")
    measured = _validate_native_balanced_kaufman_lift(
        parent_native["aggregate"],
        child_native["aggregate"],
        margin=float(required_margin),
    )
    for name, value in measured.items():
        saved = saved_lift.get(name)
        if (
            not isinstance(saved, (int, float))
            or not math.isclose(value, float(saved), abs_tol=1e-9)
        ):
            raise RuntimeError("balanced Kaufman native lift drifted")
    selection = config.get("checkpoint_selection")
    base = config.get("base_model")
    volume_base = volume_report.get("config", {}).get("base_model")
    saved_files = artifact.get("checkpoint_files")
    checkpoint_files = sorted(
        str(path.relative_to(artifact_path))
        for path in artifact_path.rglob("*") if path.is_file())
    if not (
        isinstance(selection, Mapping)
        and selection.get("contract")
        == "gate_feasible_native_reg_balanced_kaufman_v1"
        and selection.get("temporary_head_metrics_used") is False
        and isinstance(base, Mapping)
        and base == volume_base
        and base.get("model_id") == CHRONOS2_MODEL_ID
        and isinstance(base.get("revision"), str)
        and _HEX40.fullmatch(base["revision"]) is not None
        and isinstance(base.get("weights_sha256"), str)
        and _HEX64.fullmatch(base["weights_sha256"]) is not None
        and isinstance(base.get("config_sha256"), str)
        and _HEX64.fullmatch(base["config_sha256"]) is not None
        and config.get("context_length") == 256
        and config.get("kaufman_length") == 64
        and config.get("kaufman_chop") == 0.25
        and config.get("kaufman_trend") == 0.50
        and config.get("threshold_contract")
        == "fixed_causal_completed_context_er64_v1"
        and config.get("direction_contract")
        == "direction_agnostic_chop_vs_trend"
        and config.get("native_embedding_contract")
        == "ordered_ohlcv_reg_concat_5d"
        and config.get("stream_sampler")
        == "uniform_stream_equal_chop_trend_50_50"
        and config.get("validation_sampling")
        == "fixed_equal_state_per_stream"
        and isinstance(config.get("validation_windows_per_state"), int)
        and config["validation_windows_per_state"] >= 2
        and isinstance(config.get("native_promotion_margin"), (int, float))
        and np.isfinite(float(config["native_promotion_margin"]))
        and math.isclose(
            float(required_margin),
            float(config["native_promotion_margin"]),
            abs_tol=1e-12,
        )
        and artifact.get("temporary_heads_in_checkpoint") is False
        and artifact.get("ssl_heads_required_for_inference") is False
        and artifact.get("trainer_state")
        == "discarded_after_successful_checkpoint"
        and artifact.get("inference_requires")
        == ["chronos_base_model", "lora_checkpoint"]
        and isinstance(saved_files, list)
        and sorted(saved_files) == checkpoint_files
        and checkpoint_files
        and not any(
            token in Path(name).name.lower()
            for name in checkpoint_files
            for token in ("head", "decoder", "projection", "trainer"))
    ):
        raise RuntimeError("balanced Kaufman head-free artifact contract failed")


def _recover_balanced_kaufman_finalization(
        out_dir: Path,
        *,
        run_identity_sha256: str,
        trainer_payload: Mapping | None = None,
) -> dict | None:
    """Recover an authenticated balanced Kaufman two-phase publication."""
    out_dir = Path(out_dir)
    checkpoint = out_dir / "checkpoint"
    staged_checkpoint = out_dir / BALANCED_KAUFMAN_STAGED_CHECKPOINT
    state_path = out_dir / "trainer.pt"
    state_temporary = state_path.with_name(f".{state_path.name}.tmp")
    report_path = out_dir / "report.json"
    staged_report = out_dir / BALANCED_KAUFMAN_STAGED_REPORT
    if _HEX64.fullmatch(run_identity_sha256) is None:
        raise ValueError("balanced Kaufman run identity must be SHA-256")
    if not staged_report.is_file():
        if checkpoint.exists() and staged_checkpoint.exists():
            raise RuntimeError("published balanced Kaufman has a staged checkpoint")
        if (
            report_path.is_file() and checkpoint.is_dir()
            and not state_path.exists() and not staged_checkpoint.exists()
        ):
            published = _read_json_object(report_path)
            if (
                published.get("status") == "complete"
                and published.get("run_identity_sha256") == run_identity_sha256
            ):
                _authenticate_balanced_kaufman_completion(
                    published,
                    checkpoint=checkpoint,
                    run_identity_sha256=run_identity_sha256,
                )
                if state_temporary.exists():
                    if not state_temporary.is_file():
                        raise RuntimeError("temporary trainer state is not a file")
                    state_temporary.unlink()
                return published
        return None
    if not report_path.is_file():
        raise RuntimeError("balanced Kaufman finalization marker lacks report")
    finalizing = _read_json_object(report_path)
    complete = _read_json_object(staged_report)
    expected = deepcopy(finalizing)
    expected["status"] = "complete"
    expected_checkpoint = complete.get("checkpoint")
    expected_path = (
        expected_checkpoint.get("path")
        if isinstance(expected_checkpoint, Mapping) else None)
    if (
        finalizing.get("status") != "finalizing"
        or complete != expected
        or complete.get("schema") != BALANCED_KAUFMAN_REPORT_SCHEMA
        or complete.get("run_identity_sha256") != run_identity_sha256
        or not isinstance(expected_path, str)
        or Path(expected_path).resolve() != checkpoint.resolve()
    ):
        raise RuntimeError("balanced Kaufman finalization identities differ")
    expected_sha = expected_checkpoint.get("sha256")
    if checkpoint.exists() and staged_checkpoint.exists():
        raise RuntimeError("both published and staged Kaufman checkpoints exist")
    candidate = checkpoint if checkpoint.is_dir() else staged_checkpoint
    if (
        not candidate.is_dir()
        or not isinstance(expected_sha, str)
        or tree_sha256(candidate) != expected_sha
    ):
        raise RuntimeError("balanced Kaufman pending checkpoint is invalid")
    if candidate == staged_checkpoint and not state_path.is_file():
        raise RuntimeError("staged Kaufman checkpoint lacks trainer state")
    _authenticate_balanced_kaufman_completion(
        complete,
        checkpoint=checkpoint,
        artifact_path=candidate,
        run_identity_sha256=run_identity_sha256,
    )
    if state_path.exists():
        trainer = trainer_payload
        if trainer is None:
            import torch
            try:
                trainer = torch.load(
                    state_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise RuntimeError("balanced Kaufman trainer is unreadable") from exc
        _validate_balanced_kaufman_resume(
            trainer, run_identity_sha256=run_identity_sha256)
    if candidate == staged_checkpoint:
        staged_checkpoint.replace(checkpoint)
    if state_path.exists():
        state_path.unlink()
    if state_temporary.exists():
        if not state_temporary.is_file():
            raise RuntimeError("temporary trainer state is not a file")
        state_temporary.unlink()
    staged_report.replace(report_path)
    return _read_json_object(report_path)


def train_balanced_kaufman_ssl(
        prepared,
        *,
        parent: str | Path,
        parent_report: str | Path,
        base_snapshot: str | Path,
        out_dir: str | Path,
        device: str = "mps",
        context_length: int = 256,
        kaufman_length: int = 64,
        kaufman_chop: float = 0.25,
        kaufman_trend: float = 0.50,
        epochs: int = 60,
        steps_per_epoch: int = 100,
        batch_windows: int = 32,
        gradient_accumulation: int = 1,
        learning_rate: float = 5e-6,
        weight_decay: float = 0.05,
        patience: int = 8,
        projection_dim: int = 128,
        head_auxiliary_weight: float = 0.25,
        temperature: float = 0.10,
        noise: float = 0.02,
        scale: float = 0.10,
        validation_windows_per_state: int = 16,
        adapter_retention_weight: float = 0.1,
        native_promotion_margin: float = 1e-4,
        log_every_steps: int = 10,
        seed: int = 0,
        resume: bool = False,
) -> dict:
    """Direction-agnostic, class-balanced Kaufman refinement of Volume LoRA."""
    import torch
    import torch.nn as nn
    from importlib.metadata import PackageNotFoundError, version

    if (
        context_length != 256
        or kaufman_length != 64
        or not math.isclose(kaufman_chop, 0.25, abs_tol=1e-12)
        or not math.isclose(kaufman_trend, 0.50, abs_tol=1e-12)
    ):
        raise ValueError(
            "balanced Kaufman v1 fixes context=256, ER64, chop=.25, trend=.50")
    if batch_windows < 4 or batch_windows % 2:
        raise ValueError("balanced Kaufman batch_windows must be even and >= 4")
    if epochs < 1 or steps_per_epoch < 1 or patience < 1:
        raise ValueError("epochs, steps_per_epoch, and patience must be >= 1")
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be >= 1")
    numeric = (
        float(learning_rate), float(weight_decay), float(head_auxiliary_weight),
        float(temperature), float(noise), float(scale),
        float(adapter_retention_weight), float(native_promotion_margin),
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("balanced Kaufman floating configuration must be finite")
    if learning_rate <= 0.0:
        raise ValueError("balanced Kaufman learning_rate must be positive")
    if projection_dim < 2 or temperature <= 0.0:
        raise ValueError("projection_dim must be >= 2 and temperature must be > 0")
    if validation_windows_per_state < 2:
        raise ValueError("validation_windows_per_state must be >= 2")
    if any(value < 0.0 for value in (
            weight_decay, head_auxiliary_weight, adapter_retention_weight,
            native_promotion_margin, noise, scale)):
        raise ValueError("balanced Kaufman weights and augmentations must be nonnegative")
    if scale >= 1.0:
        raise ValueError("balanced Kaufman scale must be below 1")
    if log_every_steps < 1:
        raise ValueError("log_every_steps must be >= 1")

    parent = Path(parent).expanduser().resolve()
    parent_report = Path(parent_report).expanduser().resolve()
    base_snapshot = Path(base_snapshot).expanduser().resolve()
    out_dir = Path(out_dir)

    # Authenticate cheap immutable lineage before the full ER scan or any output
    # mutation. The child corpus is independently bound to its own run identity;
    # a bounded 3-minute smoke may validly inherit the full 36-stream parent.
    corpus = _as_corpus(prepared)
    data_identity_sha256 = _corpus_identity(corpus)
    parent_identity, parent_report_identity, base_identity = (
        _authenticate_balanced_kaufman_parent(
            parent=parent,
            parent_report=parent_report,
            base_snapshot=base_snapshot,
            data_identity_sha256=data_identity_sha256,
        )
    )
    data = _prepare_balanced_kaufman_data(
        corpus,
        context_length=context_length,
        kaufman_length=kaufman_length,
        kaufman_chop=kaufman_chop,
        kaufman_trend=kaufman_trend,
        validation_windows_per_state=validation_windows_per_state,
    )
    if data["data_identity_sha256"] != data_identity_sha256:
        raise RuntimeError("balanced Kaufman corpus identity drifted during preparation")
    if min(data["aggregate_state_counts"].values()) < batch_windows // 2:
        raise RuntimeError(
            "balanced Kaufman corpus cannot supply one distinct balanced batch")
    timeframes = tuple(data["corpus"])

    def package_version(distribution: str) -> str:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return "not-installed"

    run_config = {
        "objective_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "chronos_version": package_version("chronos-forecasting"),
        "peft_version": package_version("peft"),
        "base_model": base_identity,
        "timeframes": list(timeframes),
        "context_length": context_length,
        "kaufman_length": kaufman_length,
        "kaufman_chop": kaufman_chop,
        "kaufman_trend": kaufman_trend,
        "threshold_contract": "fixed_causal_completed_context_er64_v1",
        "direction_contract": "direction_agnostic_chop_vs_trend",
        "native_embedding_contract": "ordered_ohlcv_reg_concat_5d",
        "stream_sampler": "uniform_stream_equal_chop_trend_50_50",
        "validation_sampling": "fixed_equal_state_per_stream",
        "validation_windows_per_state": validation_windows_per_state,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_windows": batch_windows,
        "gradient_accumulation": gradient_accumulation,
        "learning_rate": learning_rate,
        "adapter_weight_decay": 0.0,
        "temporary_head_weight_decay": weight_decay,
        "patience": patience,
        "projection_dim": projection_dim,
        "head_auxiliary_weight": head_auxiliary_weight,
        "temperature": temperature,
        "noise": noise,
        "scale": scale,
        "adapter_retention_weight": adapter_retention_weight,
        "native_promotion_margin": native_promotion_margin,
        "log_every_steps": log_every_steps,
        "seed": seed,
        "checkpoint_selection": {
            "contract": "gate_feasible_native_reg_balanced_kaufman_v1",
            "temporary_head_metrics_used": False,
            "metric": "macro_stream_native_loss",
        },
        "objective_schema": {
            "regime": "native_reg_direction_agnostic_fixed_kaufman_er64_v1",
            "projection": "temporary_auxiliary_only_v1",
            "retention": "l2_sp_volume_parent_adapter_anchor_v1",
        },
    }
    run_identity_sha256 = _balanced_kaufman_run_identity(
        parent_sha256=parent_identity["sha256"],
        parent_report_sha256=parent_report_identity["sha256"],
        parent_report_run_identity_sha256=(
            parent_report_identity["run_identity_sha256"]),
        data_identity_sha256=data_identity_sha256,
        config=run_config,
    )

    checkpoint = out_dir / "checkpoint"
    staged_checkpoint = out_dir / BALANCED_KAUFMAN_STAGED_CHECKPOINT
    state_path = out_dir / "trainer.pt"
    state_temporary = state_path.with_name(f".{state_path.name}.tmp")
    report_path = out_dir / "report.json"
    staged_report = out_dir / BALANCED_KAUFMAN_STAGED_REPORT
    preflight_path = out_dir / "preflight.json"
    resume_payload = None
    if resume and state_path.is_file():
        try:
            resume_payload = torch.load(
                state_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError("balanced Kaufman trainer state is unreadable") from exc
        _validate_balanced_kaufman_resume(
            resume_payload, run_identity_sha256=run_identity_sha256)
    if resume:
        recovered = _recover_balanced_kaufman_finalization(
            out_dir,
            run_identity_sha256=run_identity_sha256,
            trainer_payload=resume_payload,
        )
        if recovered is not None:
            return recovered
    if checkpoint.exists():
        raise RuntimeError(f"completed balanced Kaufman checkpoint exists: {checkpoint}")
    if resume and resume_payload is None:
        raise RuntimeError(f"--resume requested but trainer state is missing: {state_path}")
    if state_path.exists() and not resume:
        raise RuntimeError(
            f"incomplete trainer state exists; pass --resume or use a new out-dir: {state_path}")
    if staged_report.exists():
        raise RuntimeError("unrecoverable balanced Kaufman staged report exists")
    if report_path.exists():
        pending = _read_json_object(report_path)
        rebuild = (
            resume_payload is not None
            and staged_checkpoint.is_dir()
            and pending.get("schema") == BALANCED_KAUFMAN_REPORT_SCHEMA
            and pending.get("status") == "finalizing"
            and pending.get("run_identity_sha256") == run_identity_sha256
        )
        if not rebuild:
            raise RuntimeError(f"balanced Kaufman report already exists: {report_path}")
        report_path.unlink()
    if resume_payload is not None and state_temporary.exists():
        if not state_temporary.is_file():
            raise RuntimeError("temporary trainer state is not a file")
        state_temporary.unlink()
    if staged_checkpoint.exists():
        if resume_payload is None or not staged_checkpoint.is_dir():
            raise RuntimeError("unrecoverable staged balanced Kaufman checkpoint exists")
        shutil.rmtree(staged_checkpoint)

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(preflight_path, {
        "schema": BALANCED_KAUFMAN_PREFLIGHT_SCHEMA,
        "status": "pass",
        "run_identity_sha256": run_identity_sha256,
        "parent": parent_identity,
        "parent_report": parent_report_identity,
        "data_identity_sha256": data_identity_sha256,
        "data_contracts": {
            timeframe: item.report for timeframe, item in data["corpus"].items()
        },
        "config": run_config,
        "streams": data["preflight_streams"],
        "aggregate_state_counts": data["aggregate_state_counts"],
    })

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model, base = _load_trainable_adapter(
        parent,
        device,
        base_revision=base_identity["revision"],
        base_snapshot=base_snapshot,
    )
    if not bool(base.chronos_config.use_reg_token):
        raise RuntimeError("balanced Kaufman SSL requires the Chronos REG token")
    embedding_dim = 5 * int(base.model_dim)
    projection_head = nn.Sequential(
        nn.LayerNorm(embedding_dim),
        nn.Linear(embedding_dim, projection_dim),
        nn.GELU(),
        nn.Linear(projection_dim, projection_dim),
    ).to(device)
    adapter_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not adapter_parameters:
        raise RuntimeError("balanced Kaufman stage found no trainable LoRA parameters")
    initial_adapter = {
        name: parameter.detach().clone()
        for name, parameter in adapter_parameters
    }
    head_parameters = list(projection_head.parameters())
    parameters = [parameter for _, parameter in adapter_parameters] + head_parameters
    optimizer = torch.optim.AdamW([
        {
            "params": [parameter for _, parameter in adapter_parameters],
            "weight_decay": 0.0,
        },
        {"params": head_parameters, "weight_decay": weight_decay},
    ], lr=learning_rate)

    def adapter_drift():
        return torch.stack([
            (parameter - initial_adapter[name]).square().mean()
            for name, parameter in adapter_parameters
        ]).mean()

    def loss_for(raw_np, states_np, *, loss_generator, require_both):
        raw = torch.from_numpy(raw_np).to(device)
        states = torch.from_numpy(states_np).to(device)
        standardized, finite, _ = _volume_structure_standardize(raw)
        first, first_mask = _volume_structure_augment(
            standardized, finite, loss_generator, noise=noise, scale=scale)
        second, second_mask = _volume_structure_augment(
            standardized, finite, loss_generator, noise=noise, scale=scale)
        embeddings = _reg_embeddings_concat(
            base,
            torch.cat([first, second], dim=0),
            torch.cat([first_mask, second_mask], dim=0),
        )
        instance = torch.arange(len(raw), device=device).repeat(2)
        native = _native_balanced_kaufman_metrics(
            embeddings,
            instance,
            states,
            temperature=temperature,
            require_both=require_both,
        )
        if native is None:
            raise RuntimeError("balanced Kaufman native objective became inactive")
        projected = torch.nn.functional.normalize(
            projection_head(embeddings), dim=1)
        projected_metrics = _native_balanced_kaufman_metrics(
            projected,
            instance,
            states,
            temperature=temperature,
            require_both=require_both,
        )
        if projected_metrics is None:
            raise RuntimeError("balanced Kaufman auxiliary objective became inactive")
        retention = adapter_drift()
        total = (
            native["loss"]
            + head_auxiliary_weight * projected_metrics["loss"]
            + adapter_retention_weight * retention)
        components = {
            "native_loss": float(native["loss"].detach()),
            "native_margin": float(native["margin"].detach()),
            "native_embedding_std": float(native["embedding_std"].detach()),
            "head_loss": float(projected_metrics["loss"].detach()),
            "adapter_retention": float(retention.detach()),
        }
        return total, components

    model.eval()
    parent_native_validation = _balanced_kaufman_native_validation(
        base,
        data,
        device=device,
        context_length=context_length,
        temperature=temperature,
        noise=noise,
        scale=scale,
        seed=seed + 20_260_803,
    )
    model.train()

    def capture_global_rng():
        state = {"cpu": torch.get_rng_state()}
        if str(device).startswith("cuda"):
            state["device"] = torch.cuda.get_rng_state(torch.device(device))
        elif (
            str(device) == "mps" and hasattr(torch, "mps")
            and hasattr(torch.mps, "get_rng_state")
        ):
            state["device"] = torch.mps.get_rng_state()
        return state

    def restore_global_rng(state):
        torch.set_rng_state(state["cpu"])
        if "device" not in state:
            return
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state(state["device"], torch.device(device))
        elif (
            str(device) == "mps" and hasattr(torch, "mps")
            and hasattr(torch.mps, "set_rng_state")
        ):
            torch.mps.set_rng_state(state["device"])

    best_loss, best_adapter, best_epoch = math.inf, None, None
    history, start_epoch, bad = [], 0, 0
    if resume_payload is not None:
        saved = resume_payload
        _restore_adapter(model, saved["adapter"])
        projection_head.load_state_dict(saved["projection_head"])
        optimizer.load_state_dict(saved["optimizer"])
        best_loss = float(saved["best_loss"])
        best_adapter = saved["best_adapter"]
        best_epoch = saved["best_epoch"]
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1
        bad = int(saved["bad"])
        rng.bit_generator.state = saved["numpy_rng"]
        generator.set_state(saved["torch_generator"])
        restore_global_rng(saved["global_torch_rng"])
        resume_payload = None
        del saved
        gc.collect()

    started = time.monotonic()
    component_names = (
        "native_loss", "native_margin", "native_embedding_std",
        "head_loss", "adapter_retention")
    epoch_stop = start_epoch if bad >= patience else epochs
    for epoch in range(start_epoch, epoch_stop):
        model.train()
        projection_head.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, **{name: 0.0 for name in component_names}}
        for step in range(steps_per_epoch):
            raw_np, states_np = _sample_balanced_kaufman_batch(
                data["streams"],
                batch_windows=batch_windows,
                context_length=context_length,
                rng=rng,
            )
            loss, components = loss_for(
                raw_np,
                states_np,
                loss_generator=generator,
                require_both=True,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite balanced Kaufman training loss")
            accumulation_group_start = (
                step // gradient_accumulation) * gradient_accumulation
            accumulation_group_size = min(
                gradient_accumulation,
                steps_per_epoch - accumulation_group_start,
            )
            (loss / accumulation_group_size).backward()
            totals["loss"] += float(loss.detach())
            for name in component_names:
                totals[name] += components[name]
            if (
                (step + 1) % gradient_accumulation == 0
                or step + 1 == steps_per_epoch
            ):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % log_every_steps == 0 or step + 1 == steps_per_epoch:
                print(
                    f"[chronos2-balanced-kaufman] ep={epoch} "
                    f"step={step + 1}/{steps_per_epoch} "
                    f"loss={totals['loss'] / (step + 1):.5f}",
                    flush=True,
                )

        model.eval()
        projection_head.eval()
        validation = _balanced_kaufman_native_validation(
            base,
            data,
            device=device,
            context_length=context_length,
            temperature=temperature,
            noise=noise,
            scale=scale,
            seed=seed + 20_260_803,
        )
        val_loss = float(validation["aggregate"]["loss"])
        try:
            _validate_native_balanced_kaufman_lift(
                parent_native_validation["aggregate"],
                validation["aggregate"],
                margin=native_promotion_margin,
            )
            native_gate_feasible = True
        except RuntimeError:
            native_gate_feasible = False
        improved = native_gate_feasible and val_loss < best_loss - 1e-6
        if improved:
            best_loss, bad = val_loss, 0
            best_adapter = _adapter_state(model)
            best_epoch = epoch
        elif best_adapter is not None:
            bad += 1
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / steps_per_epoch,
            "train_components": {
                name: totals[name] / steps_per_epoch for name in component_names
            },
            "val_loss": val_loss,
            "val_native": validation,
            "native_gate_feasible": native_gate_feasible,
            "improved": improved,
            "bad_epochs": bad,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        print(
            f"[chronos2-balanced-kaufman] ep={epoch} "
            f"train={row['train_loss']:.5f} val={val_loss:.5f} "
            f"margin={validation['aggregate']['margin']:.5f} "
            f"feasible={int(native_gate_feasible)}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        _atomic_torch(state_path, {
            "schema": BALANCED_KAUFMAN_TRAINER_SCHEMA,
            "run_identity_sha256": run_identity_sha256,
            "epoch": epoch,
            "adapter": _adapter_state(model),
            "projection_head": projection_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_adapter": best_adapter,
            "best_epoch": best_epoch,
            "history": history,
            "bad": bad,
            "numpy_rng": rng.bit_generator.state,
            "torch_generator": generator.get_state(),
            "global_torch_rng": capture_global_rng(),
        })
        if bad >= patience:
            break

    if best_adapter is None or best_epoch is None:
        raise RuntimeError(
            "balanced Kaufman SSL never produced native parent lift")
    current_parent, current_report, current_base = (
        _authenticate_balanced_kaufman_parent(
            parent=parent,
            parent_report=parent_report,
            base_snapshot=base_snapshot,
            data_identity_sha256=data_identity_sha256,
        )
    )
    if (
        current_parent != parent_identity
        or current_report != parent_report_identity
        or current_base != base_identity
    ):
        raise RuntimeError("balanced Kaufman parent lineage changed during training")
    if not state_path.is_file():
        raise RuntimeError("balanced Kaufman trainer disappeared before finalization")
    _restore_adapter(model, best_adapter)
    _save_final(model, staged_checkpoint)
    checkpoint_files = sorted(
        str(item.relative_to(staged_checkpoint))
        for item in staged_checkpoint.rglob("*") if item.is_file())
    if not checkpoint_files or any(
        token in Path(name).name.lower()
        for name in checkpoint_files
        for token in ("head", "decoder", "projection", "trainer")
    ):
        raise RuntimeError("balanced Kaufman checkpoint contains training artifacts")
    adapter_config_path = staged_checkpoint / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise RuntimeError("balanced Kaufman LoRA config is missing")
    adapter_config = _read_json_object(adapter_config_path)
    base_model_name = adapter_config.get("base_model_name_or_path")
    if (
        base_model_name != CHRONOS2_MODEL_ID
        or adapter_config.get("revision") != base_identity["revision"]
    ):
        raise RuntimeError("balanced Kaufman LoRA base identity drifted")

    del loss_for
    del adapter_drift
    del optimizer
    del projection_head
    del parameters
    del head_parameters
    del adapter_parameters
    del base
    del model
    _release_accelerator_cache(device)

    checkpoint_model, checkpoint_base = _load_trainable_adapter(
        staged_checkpoint,
        device,
        base_revision=base_identity["revision"],
        base_snapshot=base_snapshot,
    )
    checkpoint_model.eval()
    child_native_validation = _balanced_kaufman_native_validation(
        checkpoint_base,
        data,
        device=device,
        context_length=context_length,
        temperature=temperature,
        noise=noise,
        scale=scale,
        seed=seed + 20_260_803,
    )
    del checkpoint_base
    del checkpoint_model
    _release_accelerator_cache(device)
    native_lift = _validate_native_balanced_kaufman_lift(
        parent_native_validation["aggregate"],
        child_native_validation["aggregate"],
        margin=native_promotion_margin,
    )
    staged_checkpoint_sha256 = tree_sha256(staged_checkpoint)
    report = {
        "schema": BALANCED_KAUFMAN_REPORT_SCHEMA,
        "stage": "balanced_kaufman_ssl",
        "status": "finalizing",
        "run_identity_sha256": run_identity_sha256,
        "parent": parent_identity,
        "parent_report": parent_report_identity,
        "checkpoint": {
            "path": str(out_dir / "checkpoint"),
            "sha256": staged_checkpoint_sha256,
        },
        "data_identity_sha256": data_identity_sha256,
        "data_contracts": {
            timeframe: item.report for timeframe, item in data["corpus"].items()
        },
        "config": run_config,
        "preflight_streams": data["preflight_streams"],
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "checkpoint_only_validation": {
            "status": "pass",
            "contract": (
                "freshly_reloaded_lora_native_reg_without_temporary_heads"),
            "parent": parent_native_validation,
            "checkpoint": child_native_validation,
            "loss_lift_parent_minus_checkpoint": native_lift,
            "required_margin": native_promotion_margin,
        },
        "final_artifact_contract": {
            "checkpoint_files": checkpoint_files,
            "temporary_heads_in_checkpoint": False,
            "temporary_training_modules": ["projection_head", "optimizer"],
            "ssl_heads_required_for_inference": False,
            "trainer_state": "discarded_after_successful_checkpoint",
            "base_model_name_or_path": base_model_name,
            "inference_requires": ["chronos_base_model", "lora_checkpoint"],
        },
        "history": history,
        "retention_gate": {
            "status": "required_before_promotion",
            "contract": (
                "matched Volume-parent versus Kaufman-child Probe Atlas on "
                "identical pools, per-stream metrics, and controls"),
        },
        "limitations": [
            "Kaufman v1 separates current completed-context chop from trend; "
            "it does not claim to predict a future trend start",
            "trend direction is intentionally excluded from this SSL objective",
            "temporary projection heads are discarded and never required downstream",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    complete_report = deepcopy(report)
    complete_report["status"] = "complete"
    _authenticate_balanced_kaufman_completion(
        complete_report,
        checkpoint=checkpoint,
        artifact_path=staged_checkpoint,
        run_identity_sha256=run_identity_sha256,
    )
    _atomic_json(report_path, report)
    _atomic_json(staged_report, complete_report)
    staged_checkpoint.replace(checkpoint)
    if tree_sha256(checkpoint) != staged_checkpoint_sha256:
        raise RuntimeError("published balanced Kaufman checkpoint identity drifted")
    state_path.unlink()
    if state_temporary.exists():
        if not state_temporary.is_file():
            raise RuntimeError("temporary trainer state is not a file")
        state_temporary.unlink()
    if state_path.exists() or state_temporary.exists():
        raise RuntimeError("balanced Kaufman trainer state was not discarded")
    staged_report.replace(report_path)
    if staged_checkpoint.exists() or staged_report.exists():
        raise RuntimeError("balanced Kaufman finalization left staged artifacts")
    return complete_report


__all__ = [
    "train_mask",
    "train_contrastive",
    "train_volatility_contrastive",
    "preflight_volume_structure_ssl",
    "train_volume_structure_ssl",
    "preflight_balanced_kaufman_ssl",
    "train_balanced_kaufman_ssl",
    "tree_sha256",
]
