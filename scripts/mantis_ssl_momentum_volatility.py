#!/usr/bin/env python3
"""Train the strategy-agnostic momentum-volatility coupling SSL refinement.

The warm FFM encoder sees only raw OHLCV ending on the decision candle. It learns multi-horizon
signed displacement, completed-candle range expansion, and their chop/continuation/reversal/launch
coupling. The 2026 holdout is physically excluded and real input must beat shuffle/random controls.
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
                        "mantis_ssl_ma.pt")))
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
    parser.add_argument("--lr", type=float, default=float(os.environ.get("MV_LR", "1e-5")))
    parser.add_argument("--head-lr", type=float,
                        default=float(os.environ.get("MV_HEAD_LR", "1e-4")))
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--scale-lookback", type=int, default=64)
    parser.add_argument("--momentum-lookback", type=int, default=20)
    parser.add_argument("--momentum-threshold", type=float, default=.5)
    parser.add_argument("--expansion-threshold", type=float, default=1.1)
    parser.add_argument("--candle-weight", type=float, default=.25)
    parser.add_argument("--momentum-weight", type=float, default=1.0)
    parser.add_argument("--volatility-weight", type=float, default=.5)
    parser.add_argument("--coupling-weight", type=float, default=.5)
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


def _run_atlas(*, data, checkpoint, labels, device, batch):
    atlas_dir = checkpoint.parent / "probe_atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    result = atlas_dir / "momentum_volatility.json"
    cache = atlas_dir / "momentum_volatility_emb.npy"
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
        "stage": "momentum_volatility",
        "schema": MOMENTUM_VOLATILITY_SCHEMA,
        "parent": {"path": str(parent), "sha256": sha256(parent)},
        "holdout_start": HOLDOUT_START,
        "sampling_mode": args.sampling_mode,
        "inputs": "raw_ohlcv_ending_on_decision_bar",
        "causal_scale": "median_completed_candle_high_minus_low",
        "uses_atr": False,
        "uses_strategy_labels": False,
        "lora": {"rank": args.lora_r, "alpha": args.lora_alpha},
    }
    print("\nMOMENTUM-VOLATILITY SSL PREFLIGHT PASSED")
    print(f"  data       : {data}")
    print(f"  streams    : {len(tickers)} x {len(timeframes)} = {len(expected)}")
    print(f"  holdout    : >= {HOLDOUT_START} physically excluded")
    print(f"  parent     : {parent}")
    print("  targets    : displacement + range expansion + causal coupling")
    print("  ATR/R/IP   : none")
    print(f"  tuning     : LoRA r={args.lora_r} alpha={args.lora_alpha:g} "
          f"freeze={args.freeze_encoder_layers}")
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
        freeze_encoder_layers=args.freeze_encoder_layers,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        clamp=10.0, grad_clip=1.0, resume=args.resume, device=device, seed=args.seed)
    if not verdict.get("all_pass"):
        raise RuntimeError(f"momentum-volatility SSL/control gates failed: {verdict}")

    atlas = None if args.skip_atlas else _run_atlas(
        data=data, checkpoint=output, labels=labels, device=device, batch=batch)
    print("\nMOMENTUM-VOLATILITY SSL COMPLETE")
    print(f"  encoder : {output}")
    print(f"  report  : {output}.report.json")
    print(f"  atlas   : {atlas or 'skipped'}")


if __name__ == "__main__":
    main()
