#!/usr/bin/env python3
"""Train the strategy-agnostic momentum-volatility transition SSL refinement.

The warm FFM encoder sees only raw OHLCV ending on the decision candle. Disposable linear heads
and a direct embedding contrastive loss teach the complete 2x2 momentum/volatility transition
matrix. The exported artifact is encoder-only and must beat its immediate parent with every task
head removed. The 2026 holdout is physically excluded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.data_provenance import seal_continuous_streams, sha256
from futures_foundation.finetune import ssl
from futures_foundation.finetune.pretext.momentum_volatility import (
    MOMENTUM_VOLATILITY_SCHEMA,
)


TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")
HOLDOUT_START = "2026-01-01"


def _csv(value):
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _default_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", str(ROOT / "data")))
    parser.add_argument("--warm-ckpt", default=os.environ.get(
        "WARM_CKPT", str(ROOT / "checkpoints" / "mantis_ssl_nextleg_race_v2.pt")))
    parser.add_argument("--out", default=os.environ.get(
        "OUT_PATH", str(ROOT / "temp" / "momentum_volatility" /
                        "mantis_ssl_mv_v2.pt")))
    parser.add_argument("--atlas-labels", default=os.environ.get("TREND_LABELS"))
    parser.add_argument("--tickers", default=",".join(TICKERS))
    parser.add_argument("--tfs", default=",".join(TIMEFRAMES))
    parser.add_argument("--sampling-mode", choices=("bar_proportional", "uniform_stream"),
                        default=os.environ.get("SAMPLING_MODE", "bar_proportional"))
    parser.add_argument("--controls", default=os.environ.get("SSL_CONTROLS", "shuffle,random"))
    parser.add_argument("--control-epochs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("MV_EPOCHS", "60")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("MV_STEPS", "50")))
    parser.add_argument("--batch", type=int)
    parser.add_argument("--lr", type=float, default=float(os.environ.get("MV_LR", "3e-5")))
    parser.add_argument("--head-lr", type=float,
                        default=float(os.environ.get("MV_HEAD_LR", "3e-5")))
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--scale-lookback", type=int, default=64)
    parser.add_argument("--momentum-lookback", type=int, default=20)
    parser.add_argument("--momentum-threshold", type=float, default=.5)
    parser.add_argument("--expansion-threshold", type=float, default=1.1)
    parser.add_argument("--candle-weight", type=float, default=.25)
    parser.add_argument("--momentum-weight", type=float, default=1.0)
    parser.add_argument("--volatility-weight", type=float, default=.5)
    parser.add_argument("--coupling-weight", type=float, default=.5)
    parser.add_argument("--transition-contrastive-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-temperature", type=float, default=.1)
    parser.add_argument("--encoder-transfer-margin", type=float, default=.002)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--freeze-encoder-layers", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-atlas", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _resolve(args):
    data = Path(args.data_dir).expanduser().resolve()
    parent = Path(args.warm_ckpt).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    labels = (Path(args.atlas_labels).expanduser().resolve()
              if args.atlas_labels else
              parent.parent / "probe_atlas" / "trend_lifecycle_labels_pre2026.npz")
    if not data.is_dir():
        raise FileNotFoundError(f"data directory not found: {data}")
    if not parent.is_file():
        raise FileNotFoundError(f"warm FFM checkpoint not found: {parent}")
    if output == parent:
        raise SystemExit("--out must differ from the immutable parent checkpoint")
    if output.exists() and not args.resume:
        raise SystemExit(f"output already exists: {output}; pass --resume to continue")
    if not args.skip_atlas and not labels.is_file():
        raise FileNotFoundError(
            f"Probe Atlas labels not found: {labels}; pass --atlas-labels or --skip-atlas")
    output.parent.mkdir(parents=True, exist_ok=True)
    return data, parent, output, labels


def _run_atlas(*, data, checkpoint, labels, device, batch, stem, artifact_dir):
    atlas_dir = Path(artifact_dir)
    atlas_dir.mkdir(parents=True, exist_ok=True)
    result = atlas_dir / f"{stem}.json"
    cache = atlas_dir / f"{stem}_emb.npy"
    env = os.environ.copy()
    env.update({
        "FFM_ROOT": str(ROOT), "DATA_DIR": str(data),
        "CKPT_NAME": checkpoint.name, "CKPT_PATH": str(checkpoint),
        "CKPT_SHA256": sha256(checkpoint), "TREND_LABELS": str(labels),
        "EMB_CACHE": str(cache), "ATLAS_OUT": str(result),
        "ATLAS_BATCH": str(batch), "DEVICE": device,
    })
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "probe_atlas.py")],
        cwd=ROOT, env=env, check=True)
    payload = json.loads(result.read_text())
    if payload.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("Probe Atlas result is not bound to the momentum-volatility checkpoint")
    return result


def _probe_auc(payload, name, stream=None):
    row = payload["probes"][name]
    return float(row["auc"] if stream is None else row["per_stream_auc"][stream])


def _assert_encoder_transfer(*, output, parent, margin, child_atlas=None,
                             parent_atlas=None):
    """Fail closed unless the exported encoder—not a disposable head—improves."""
    report = json.loads(Path(str(output) + ".report.json").read_text())
    probe = report.get("probe") or {}
    expected_parent = str(parent)
    core_delta = float(probe.get("mean_core_delta", float("-inf")))
    forward_delta = float(probe.get("forward_score", float("-inf")))
    failures = []
    if probe.get("comparison_baseline") != expected_parent:
        failures.append("probe baseline is not the immediate parent")
    if core_delta < float(margin):
        failures.append(
            f"parent-relative core {core_delta:+.6f} < {float(margin):+.6f}")
    if forward_delta <= 0:
        failures.append(f"parent-relative forward score {forward_delta:+.6f} <= 0")
    transfer = {
        "schema": "ffm_encoder_transfer_gate_v1",
        "child": {"path": str(output), "sha256": sha256(output)},
        "parent": {"path": str(parent), "sha256": sha256(parent)},
        "heads_available_to_gate": False,
        "parent_relative_probe": {
            "core_delta": core_delta,
            "forward_delta": forward_delta,
            "minimum_core_delta": float(margin),
        },
    }
    if child_atlas is not None and parent_atlas is not None:
        child = json.loads(Path(child_atlas).read_text())
        base = json.loads(Path(parent_atlas).read_text())
        predictive = (
            "pred_fwd_large_move",
            "pred_vol_expand",
            "pred_persistent_trend_start",
        )
        retention = (
            "ret_vol_regime",
            "ret_squeeze",
            "ret_vol_surge",
        )
        overall = {
            name: _probe_auc(child, name) - _probe_auc(base, name)
            for name in predictive}
        nq3 = {
            name: _probe_auc(child, name, "NQ@3min")
            - _probe_auc(base, name, "NQ@3min")
            for name in predictive}
        retained = {
            name: _probe_auc(child, name) - _probe_auc(base, name)
            for name in retention}
        transfer["probe_atlas_parent_delta"] = {
            "predictive": overall,
            "nq_3min": nq3,
            "retention": retained,
        }
        if (sum(overall.values()) / len(overall) < float(margin)
                or sum(nq3.values()) / len(nq3) <= 0
                or min(overall.values()) < -.005
                or min(retained.values()) < -.01):
            failures.append(
                "Probe Atlas child did not add transferable MV context over Race-v2")
    destination = Path(str(output) + ".encoder_transfer.json")
    transfer["passed"] = not failures
    transfer["failures"] = failures
    destination.write_text(json.dumps(transfer, indent=2) + "\n")
    if failures:
        raise RuntimeError(
            "MV-v2 encoder transfer failed: " + "; ".join(failures))
    return destination


def main():
    args = _parser().parse_args()
    data, parent, output, labels = _resolve(args)
    device = args.device or _default_device()
    batch = args.batch or {"cuda": 512, "mps": 128, "cpu": 32}[device]
    tickers, timeframes = _csv(args.tickers), _csv(args.tfs)
    expected = tuple((ticker, timeframe) for ticker in tickers for timeframe in timeframes)
    provenance = seal_continuous_streams(data, expected, repo_root=ROOT)
    controls = _csv(args.controls)
    if set(controls) - {"shuffle", "random"}:
        raise ValueError(f"unsupported controls: {controls}")
    if args.lora_r <= 0:
        raise ValueError("momentum-volatility refinement requires LoRA")
    if args.scale_lookback > 64 or args.momentum_lookback >= 64:
        raise ValueError("causal lookbacks must fit the shortest 64-bar context")

    contract = {
        **provenance,
        "stage": "momentum_volatility_v2",
        "schema": MOMENTUM_VOLATILITY_SCHEMA,
        "parent": {"path": str(parent), "sha256": sha256(parent)},
        "holdout_start": HOLDOUT_START,
        "sampling_mode": args.sampling_mode,
        "inputs": "raw_ohlcv_ending_on_decision_bar",
        "causal_scale": "median_completed_candle_high_minus_low",
        "uses_atr": False,
        "uses_strategy_labels": False,
        "promoted_artifact": "encoder_only",
        "task_heads": "disposable_training_supervision_only",
        "comparison_parent": {"path": str(parent), "sha256": sha256(parent)},
        "lora": {"rank": args.lora_r, "alpha": args.lora_alpha},
    }
    print("\nMOMENTUM-VOLATILITY SSL PREFLIGHT PASSED")
    print(f"  data       : {data}")
    print(f"  streams    : {len(tickers)} x {len(timeframes)} = {len(expected)}")
    print(f"  holdout    : >= {HOLDOUT_START} physically excluded")
    print(f"  parent     : {parent}")
    print("  targets    : path efficiency x volatility transition (complete 2x2 matrix)")
    print("  ATR/R/IP   : none")
    print(f"  tuning     : LoRA r={args.lora_r} alpha={args.lora_alpha:g} "
          f"freeze={args.freeze_encoder_layers}")
    print("  artifact   : encoder only; task heads are discarded")
    print(f"  gate       : child must beat Race-v2 parent by {args.encoder_transfer_margin:g}")
    print(f"  training   : {device} batch={batch} epochs={args.epochs} steps={args.steps}")
    print(f"  controls   : {controls}")
    print(f"  output     : {output}")
    if args.preflight_only:
        return
    Path(str(output) + ".data_provenance.json").write_text(
        json.dumps(contract, indent=2, default=list) + "\n")

    verdict = ssl.loop_ssl(
        data_dir=str(data), tickers=tickers, tfs=timeframes, out_path=str(output),
        holdout_start=HOLDOUT_START, val_frac=.1, pretext="momentum_volatility",
        backbone_ckpt=str(parent), sampling_mode=args.sampling_mode,
        probe_baseline_ckpt=str(parent),
        probe_margin=args.encoder_transfer_margin,
        controls=controls, control_epochs=args.control_epochs, probe=True,
        horizons=(5, 10, 20, 25), context_lengths=(64, 100, 150, 200),
        new_channels=3, batch=batch, epochs=args.epochs, steps_per_epoch=args.steps,
        lr=args.lr, head_lr=args.head_lr, weight_decay=0.0, patience=args.patience,
        scale_lookback=args.scale_lookback,
        momentum_lookback=args.momentum_lookback,
        momentum_threshold=args.momentum_threshold,
        expansion_threshold=args.expansion_threshold,
        candle_weight=args.candle_weight,
        momentum_weight=args.momentum_weight,
        volatility_weight=args.volatility_weight,
        coupling_weight=args.coupling_weight,
        transition_contrastive_weight=args.transition_contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        freeze_encoder_layers=args.freeze_encoder_layers,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        clamp=10.0, grad_clip=1.0, resume=args.resume, device=device, seed=args.seed)
    if not verdict.get("all_pass"):
        raise RuntimeError(f"momentum-volatility SSL/control gates failed: {verdict}")

    child_atlas = parent_atlas = None
    if not args.skip_atlas:
        child_atlas = _run_atlas(
            data=data, checkpoint=output, labels=labels, device=device, batch=batch,
            stem="momentum_volatility_v2",
            artifact_dir=output.parent / "probe_atlas")
        parent_atlas = _run_atlas(
            data=data, checkpoint=parent, labels=labels, device=device, batch=batch,
            stem="parent_race_v2",
            artifact_dir=output.parent / "probe_atlas")
    transfer = _assert_encoder_transfer(
        output=output, parent=parent, margin=args.encoder_transfer_margin,
        child_atlas=child_atlas, parent_atlas=parent_atlas)
    print("\nMOMENTUM-VOLATILITY SSL COMPLETE")
    print(f"  encoder : {output}")
    print("  heads   : discarded; no readout is part of the foundation contract")
    print(f"  report  : {output}.report.json")
    print(f"  transfer: {transfer}")
    print(f"  atlas   : {child_atlas or 'skipped'}")


if __name__ == "__main__":
    main()
