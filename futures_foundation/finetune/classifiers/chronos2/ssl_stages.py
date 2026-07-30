"""Chronos-2 LoRA mask and temporal-contrastive SSL stages.

The objective heads are temporary.  Each stage saves only a PEFT adapter that
can be loaded by the ordinary Chronos-2 pipeline and classifier seam.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


IMPORT_ALLOWLIST = ["chronos.chronos2.model"]


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


def _load_trainable_adapter(parent: str | Path, device: str):
    from peft import AutoPeftModel

    model = AutoPeftModel.from_pretrained(
        parent,
        is_trainable=True,
        import_allowlist=IMPORT_ALLOWLIST,
    ).to(device)
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
        learning_rate: float = 2e-5,
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


__all__ = [
    "train_mask",
    "train_contrastive",
    "train_volatility_contrastive",
    "tree_sha256",
]
