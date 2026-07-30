#!/usr/bin/env python3
"""LoRA-adapt Chronos-2-small on joint and single-ticker OHLCV tasks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.classifiers.chronos2.multivariate import (
    TIMEFRAME_MINUTES,
    prepare_multivariate,
)


MODEL_ID = "autogluon/chronos-2-small"
TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
LORA_TARGETS = (
    "self_attention.q",
    "self_attention.v",
    "self_attention.k",
    "self_attention.o",
    "output_patch_embedding.output_layer",
)
CHRONOS2_IMPORT_ALLOWLIST = ("chronos.chronos2.model",)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _device(requested: str) -> str:
    import torch

    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dependency_versions() -> dict[str, str]:
    required = ("chronos-forecasting", "transformers", "peft", "torch")
    versions = {}
    for package in required:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"{package} is required; install the repository requirements before training") from exc
    major_minor = tuple(int(part) for part in versions["chronos-forecasting"].split(".")[:2])
    if major_minor < (2, 2):
        raise RuntimeError("chronos-forecasting>=2.2 is required for Chronos-2 LoRA")
    return versions


def _load_pipeline(source, *, device: str):
    """Load base or local PEFT Chronos-2 with the narrow PEFT import allowlist."""
    from chronos import Chronos2Pipeline

    kwargs = {"device_map": device}
    path = Path(source)
    if path.is_dir() and (path / "adapter_config.json").is_file():
        kwargs["import_allowlist"] = list(CHRONOS2_IMPORT_ALLOWLIST)
    return Chronos2Pipeline.from_pretrained(source, **kwargs)


def _validation_loss(pipeline, inputs, *, context_length, prediction_length,
                     batch_size, device):
    """Exact Chronos-2 validation loss for one declared task arm."""
    import torch
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    dataset = Chronos2Dataset(
        inputs=inputs,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        output_patch_size=pipeline.model_output_patch_size,
        mode=DatasetMode.VALIDATION,
    )
    losses = []
    pipeline.model.eval()
    with torch.no_grad():
        for batch in dataset:
            batch = {
                key: (value.to(device) if hasattr(value, "to") else value)
                for key, value in batch.items()
            }
            losses.append(float(pipeline.model(**batch).loss))
    return float(np.mean(losses))


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timeframes = tuple(
        value.strip() for value in args.timeframes.split(",") if value.strip())
    if not timeframes or len(set(timeframes)) != len(timeframes):
        raise ValueError("timeframes must be a non-empty unique comma-separated list")
    unsupported = set(timeframes) - set(TIMEFRAME_MINUTES)
    if unsupported:
        raise ValueError(f"unsupported timeframes: {sorted(unsupported)}")
    prepared_by_timeframe = {
        timeframe: prepare_multivariate(
            args.data_dir,
            repo_root=ROOT,
            tickers=TICKERS,
            timeframe=timeframe,
            holdout_start=args.holdout_start,
            val_frac=args.val_frac,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            validation_windows=args.validation_windows,
            report_path=out_dir / f"data_preflight_{timeframe}.json",
        )
        for timeframe in timeframes
    }
    for timeframe, prepared in prepared_by_timeframe.items():
        print(
            f"[preflight:{timeframe}] {prepared.train.shape[0]} variates x "
            f"{prepared.train.shape[1]:,} train rows; "
            f"{len(prepared.validation)} chronological validation windows; "
            f"finite={prepared.report['finite_fraction']:.4f}",
            flush=True,
        )
    if args.preflight_only:
        return {
            "status": "preflight_pass",
            "data": {
                timeframe: prepared.report
                for timeframe, prepared in prepared_by_timeframe.items()
            },
        }

    versions = _dependency_versions()
    device = _device(args.device)
    reference = prepared_by_timeframe[timeframes[0]]
    ticker_channels = {
        ticker: np.asarray([
            index for index, name in enumerate(reference.channel_names)
            if name.startswith(f"{ticker}.")
        ], dtype=np.int64)
        for ticker in TICKERS
    }
    if any(len(indices) != 5 for indices in ticker_channels.values()):
        raise RuntimeError("every serve ticker must have exactly five OHLCV channels")
    if not 0.0 < args.single_ticker_probability <= 0.5:
        raise ValueError("single-ticker-probability must be in (0, 0.5]")
    joint_copies = max(
        1,
        round(len(TICKERS) * (1.0 - args.single_ticker_probability)
              / args.single_ticker_probability),
    )
    actual_single_probability = len(TICKERS) / (joint_copies + len(TICKERS))
    if abs(actual_single_probability - args.single_ticker_probability) > 1e-9:
        raise ValueError(
            "single-ticker-probability must map exactly to equally weighted "
            "individual ticker tasks and an integer number of joint task copies")
    train_inputs = []
    joint_validation = {}
    ticker_validation = {}
    validation_inputs = []
    for timeframe, prepared in prepared_by_timeframe.items():
        train_inputs.extend([prepared.train] * joint_copies)
        train_inputs.extend(
            prepared.train[indices] for indices in ticker_channels.values())
        joint_validation[timeframe] = list(prepared.validation)
        ticker_validation[timeframe] = {
            ticker: [value[indices] for value in prepared.validation]
            for ticker, indices in ticker_channels.items()
        }
        # Within every timeframe, joint and the collection of single-ticker
        # validation tasks receive equal aggregate checkpoint-selection weight.
        validation_inputs.extend(joint_validation[timeframe] * len(TICKERS))
        validation_inputs.extend(
            value
            for ticker in TICKERS
            for value in ticker_validation[timeframe][ticker]
        )
    print(
        f"[chronos2] model={MODEL_ID} device={device} steps={args.steps} "
        f"timeframes={','.join(timeframes)} "
        f"context={args.context_length} horizon={args.prediction_length} "
        f"LoRA r={args.lora_r} alpha={args.lora_alpha}",
        flush=True,
    )
    started = time.monotonic()
    pipeline = _load_pipeline(MODEL_ID, device=device)
    checkpoint_name = "finetuned-ckpt"
    fitted = pipeline.fit(
        inputs=train_inputs,
        validation_inputs=validation_inputs,
        prediction_length=args.prediction_length,
        finetune_mode="lora",
        lora_config={
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "target_modules": list(LORA_TARGETS),
        },
        context_length=args.context_length,
        learning_rate=args.learning_rate,
        num_steps=args.steps,
        # The minimum five-series task makes each draw one complete task:
        # either joint 45-variate context or one ticker's five OHLCV variates.
        batch_size=5,
        output_dir=out_dir / "trainer",
        finetuned_ckpt_name=checkpoint_name,
        optim="adamw_torch",
        logging_steps=1 if args.steps <= 20 else min(25, args.steps),
        eval_steps=max(1, min(args.eval_steps, args.steps)),
        save_steps=max(1, min(args.eval_steps, args.steps)),
        gradient_accumulation_steps=args.gradient_accumulation,
        dataloader_pin_memory=False,
        disable_tqdm=args.disable_tqdm,
        seed=args.seed,
        data_seed=args.seed,
    )
    checkpoint = out_dir / "trainer" / checkpoint_name
    if not checkpoint.is_dir():
        raise RuntimeError(f"Chronos-2 did not write the expected checkpoint: {checkpoint}")

    # Prove the saved artifact—not merely the in-memory model—can be consumed.
    del fitted, pipeline
    reloaded = _load_pipeline(checkpoint, device=device)
    embed_input = reference.validation[-1][:, :-args.prediction_length]
    embeddings, loc_scale = reloaded.embed(
        [embed_input], batch_size=len(reference.channel_names),
        context_length=args.context_length)
    embedding = embeddings[0].numpy()
    if embedding.shape[0] != len(reference.channel_names) or not np.isfinite(embedding).all():
        raise RuntimeError(
            f"reloaded embedding failed shape/finite contract: {embedding.shape}")
    if len(loc_scale) != 1:
        raise RuntimeError("reloaded embedding returned an unexpected scaling contract")
    validation_by_timeframe = {}
    for timeframe in timeframes:
        joint_val_loss = _validation_loss(
            reloaded, joint_validation[timeframe],
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            batch_size=len(reference.channel_names),
            device=device,
        )
        ticker_val_losses = {
            ticker: _validation_loss(
                reloaded, ticker_validation[timeframe][ticker],
                context_length=args.context_length,
                prediction_length=args.prediction_length,
                batch_size=5,
                device=device,
            )
            for ticker in TICKERS
        }
        mean_ticker_loss = float(np.mean(list(ticker_val_losses.values())))
        validation_by_timeframe[timeframe] = {
            "joint_loss": joint_val_loss,
            "single_ticker_losses": ticker_val_losses,
            "mean_single_ticker_loss": mean_ticker_loss,
            "equal_weight_mean": 0.5 * (joint_val_loss + mean_ticker_loss),
        }
    corpus_identity = hashlib.sha256(json.dumps({
        timeframe: prepared.report["identity_sha256"]
        for timeframe, prepared in prepared_by_timeframe.items()
    }, sort_keys=True).encode()).hexdigest()

    report = {
        "schema": "ffm_chronos2_lora_smoke_v2",
        "status": "smoke_pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "device": device,
        "versions": versions,
        "data_identity_sha256": corpus_identity,
        "checkpoint": str(checkpoint),
        "checkpoint_tree_sha256": _tree_sha256(checkpoint),
        "elapsed_seconds": time.monotonic() - started,
        "config": {
            "steps": args.steps,
            "timeframes": list(timeframes),
            "context_length": args.context_length,
            "prediction_length": args.prediction_length,
            "batch_size_series": 5,
            "training_tasks": ["joint_45", "single_ticker_5"],
            "single_ticker_probability": actual_single_probability,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "eval_steps": min(args.eval_steps, args.steps),
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "seed": args.seed,
        },
        "reload_embedding_shape": list(embedding.shape),
        "reload_embedding_finite": True,
        "validation": validation_by_timeframe,
        "validation_equal_weight_mean": float(np.mean([
            value["equal_weight_mean"]
            for value in validation_by_timeframe.values()
        ])),
        "data": {
            timeframe: prepared.report
            for timeframe, prepared in prepared_by_timeframe.items()
        },
    }
    _atomic_json(out_dir / "smoke_report.json", report)
    print(
        f"[chronos2] PASS checkpoint={checkpoint} embedding={embedding.shape} "
        f"elapsed={report['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "temp" / "chronos2_small_3min")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--prediction-length", type=int, default=32)
    parser.add_argument("--validation-windows", type=int, default=8)
    parser.add_argument("--timeframes", default="3min")
    parser.add_argument("--holdout-start", default="2026-01-01")
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--single-ticker-probability", type=float, default=0.25)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-tqdm", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (args.steps < 1 or args.lora_r < 1 or args.gradient_accumulation < 1
            or args.eval_steps < 1):
        raise SystemExit(
            "steps, lora-r, gradient-accumulation, and eval-steps must be positive")
    run(args)


if __name__ == "__main__":
    main()
