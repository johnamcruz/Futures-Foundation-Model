#!/usr/bin/env python3
"""Train the opt-in Span-Structural NextLeg SSL objective on the clean 9x4 corpus.

This public, strategy-agnostic stage refines a NextLeg encoder with raw-candle targets only:

* reconstruct a contiguous span centered on the current, already-confirmed fractal formation;
* classify current and next HH/HL/LH/LL structure;
* predict both future leg durations and scale-free excursions;
* classify the first future close-confirmed BOS/CHOCH (or no break) and its delay; and
* retain the ordinary multi-horizon candle forecast anchor.

All targets are built independently per ticker/timeframe stream.  Inputs end at the current
pivot's confirmation candle, while the orchestrator reserves the complete future target horizon
inside the pre-2026 train/validation split.  The saved artifact is still an encoder checkpoint;
the task heads exist only in the crash-resume trainer sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.data_provenance import seal_continuous_streams
from futures_foundation.finetune import ssl


TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")
HOLDOUT_START = "2026-01-01"


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _csv(value))


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", str(ROOT / "data")))
    parser.add_argument("--warm-ckpt", default=os.environ.get(
        "BACKBONE_CKPT", str(ROOT / "checkpoints" / "mantis_ssl_nextleg.pt")))
    parser.add_argument("--warm-trainer", default=os.environ.get("WARM_TRAINER_CKPT"),
                        help="Optional parent NextLeg .trainer.pt for forecast/leg head reuse")
    parser.add_argument("--out", default=os.environ.get(
        "OUT_PATH", str(ROOT / "temp" / "structural_nextleg" /
                        "mantis_ssl_structural_nextleg.pt")))
    parser.add_argument("--tickers", default=os.environ.get("TICKERS", ",".join(TICKERS)))
    parser.add_argument("--tfs", default=os.environ.get("TFS", ",".join(TIMEFRAMES)))
    parser.add_argument("--sampling-mode", choices=("bar_proportional", "uniform_stream"),
                        default=os.environ.get("SAMPLING_MODE", "bar_proportional"))
    parser.add_argument("--controls", default=os.environ.get("CONTROLS", "shuffle,random"))
    parser.add_argument("--control-epochs", type=int,
                        default=int(os.environ.get("CONTROL_EPOCHS", "8")))
    # Sixty is a ceiling, not a requirement: validation patience restores the best epoch.
    # Structural heads were still improving near epoch 20 in the first 9x4 experiment, so a
    # shorter default could truncate useful learning before the early-stop gate could decide.
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "60")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "50")))
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-5")))
    parser.add_argument("--head-lr", type=float,
                        default=float(os.environ.get("HEAD_LR", "1e-4")))
    parser.add_argument("--patience", type=int, default=int(os.environ.get("PATIENCE", "8")))
    parser.add_argument("--context-lengths", default=os.environ.get(
        "CONTEXT_LENGTHS", "64,100,150,200"))
    parser.add_argument("--horizons", default=os.environ.get("HORIZONS", "5,10,20,25"))
    parser.add_argument("--leg-k", type=int, default=int(os.environ.get("LEG_K", "2")))
    parser.add_argument("--leg-cap", type=int, default=int(os.environ.get("LEG_CAP", "256")))
    parser.add_argument("--event-horizon", type=int,
                        default=int(os.environ.get("EVENT_HORIZON", "128")))
    parser.add_argument("--span-width", type=int,
                        default=int(os.environ.get("SPAN_WIDTH", "5")))
    parser.add_argument("--span-prob", type=float,
                        default=float(os.environ.get("SPAN_PROB", "0.5")))
    parser.add_argument("--freeze-encoder-layers", type=int,
                        default=int(os.environ.get("FREEZE_ENCODER_LAYERS", "2")))
    parser.add_argument(
        "--head-only", action="store_true",
        help="Freeze the complete parent encoder and refit only matched Structural task heads")
    parser.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "8")))
    parser.add_argument("--lora-alpha", type=float,
                        default=float(os.environ.get("LORA_ALPHA", "16")))
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"),
                        default=os.environ.get("DEVICE"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    parser.add_argument("--resume", action="store_true", default=os.environ.get("RESUME") == "1")
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _resolve(args):
    data_dir = Path(args.data_dir).expanduser().resolve()
    warm = Path(args.warm_ckpt).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    if not warm.is_file():
        raise FileNotFoundError(f"warm NextLeg encoder not found: {warm}")
    if warm == out:
        raise SystemExit("--out must differ from --warm-ckpt; the parent is immutable")
    if out.exists() and not args.resume:
        raise SystemExit(f"output already exists: {out}; pass --resume to continue it")
    warm_trainer = Path(args.warm_trainer).expanduser().resolve() if args.warm_trainer else None
    if warm_trainer is not None and not warm_trainer.is_file():
        raise FileNotFoundError(f"warm NextLeg trainer sidecar not found: {warm_trainer}")
    tickers, tfs = _csv(args.tickers), _csv(args.tfs)
    expected = tuple((ticker, timeframe) for ticker in tickers for timeframe in tfs)
    provenance = seal_continuous_streams(data_dir, expected, repo_root=ROOT)
    if len(provenance["streams"]) != len(expected):
        raise RuntimeError("data provenance did not seal the requested stream matrix")
    out.parent.mkdir(parents=True, exist_ok=True)
    return data_dir, warm, warm_trainer, out, tickers, tfs, provenance


def main() -> None:
    args = _parser().parse_args()
    device = args.device or _default_device()
    batch = args.batch or {"cuda": 512, "mps": 128, "cpu": 32}[device]
    data_dir, warm, warm_trainer, out, tickers, tfs, provenance = _resolve(args)
    controls = _csv(args.controls)

    run_contract = {
        **provenance,
        "stage": "nextleg_structural",
        "objective": "pivot_span_reconstruction+structure+bos_choch+nextleg+candle_anchor",
        "backbone_ckpt": str(warm),
        "warm_trainer_ckpt": str(warm_trainer) if warm_trainer else None,
        "holdout_start": HOLDOUT_START,
        "tickers": list(tickers), "timeframes": list(tfs),
        "sampling_mode": args.sampling_mode,
    }
    Path(str(out) + ".data_provenance.json").write_text(
        json.dumps(run_contract, indent=2) + "\n")

    print("\nSPAN-STRUCTURAL NEXTLEG PREFLIGHT PASSED")
    print(f"  data       : {data_dir}")
    print(f"  streams    : {len(tickers)} x {len(tfs)} = {len(tickers) * len(tfs)}")
    print(f"  holdout    : >= {HOLDOUT_START} physically excluded")
    print(f"  parent     : {warm}")
    print(f"  task heads : {warm_trainer or 'new (encoder-only parent)'}")
    print(f"  target     : k={args.leg_k} cap={args.leg_cap} "
          f"event_horizon={args.event_horizon} span={args.span_width}")
    print(f"  training   : {device} batch={batch} epochs={args.epochs} steps={args.steps} "
          f"encoder_lr={args.lr:g} head_lr={args.head_lr:g}")
    print(f"  head only  : {args.head_only}")
    print(f"  controls   : {controls or 'none'}")
    print(f"  output     : {out}")
    if args.preflight_only:
        return

    verdict = ssl.loop_ssl(
        data_dir=str(data_dir), tickers=tickers, tfs=tfs, out_path=str(out),
        holdout_start=HOLDOUT_START, val_frac=.1, pretext="nextleg_structural",
        backbone_ckpt=str(warm),
        warm_trainer_ckpt=str(warm_trainer) if warm_trainer else None,
        sampling_mode=args.sampling_mode, controls=controls,
        control_epochs=args.control_epochs, probe=not args.no_probe,
        context_lengths=_ints(args.context_lengths), horizons=_ints(args.horizons),
        leg_k=args.leg_k, leg_cap=args.leg_cap, leg_w=1.0, mse_weight=1.0,
        structure_current_w=.25, structure_next_w=.75, excursion_w=.25,
        structure_event_w=.75, structure_event_horizon=args.event_horizon,
        structure_span_w=.25, structure_span_width=args.span_width,
        structure_span_prob=args.span_prob,
        new_channels=3, batch=batch, epochs=args.epochs, steps_per_epoch=args.steps,
        lr=args.lr, head_lr=args.head_lr, weight_decay=0.0, patience=args.patience,
        freeze_encoder_layers=args.freeze_encoder_layers,
        freeze_encoder=args.head_only,
        lora_r=0 if args.head_only else args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=0.0,
        clamp=10.0, grad_clip=1.0, resume=args.resume, device=device, seed=args.seed)

    if args.head_only:
        import torch
        parent_state = torch.load(warm, map_location="cpu", weights_only=False)
        output_state = torch.load(out, map_location="cpu", weights_only=False)
        if (
            parent_state.keys() != output_state.keys()
            or any(not torch.equal(parent_state[key], output_state[key])
                   for key in parent_state)
        ):
            raise RuntimeError(
                "head-only Structural refit changed at least one frozen encoder tensor")
        # Preserve byte-identical artifact identity so the sidecar binding can be
        # checked cheaply by every downstream loader.
        shutil.copyfile(warm, out)
        parent_sha = _sha256(warm)
        trainer_path = Path(str(out) + ".trainer.pt")
        payload = torch.load(trainer_path, map_location="cpu", weights_only=False)
        payload["head_only"] = True
        payload["matched_encoder_sha256"] = parent_sha
        temporary = trainer_path.with_suffix(trainer_path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(trainer_path)
        Path(str(out) + ".matched_head.json").write_text(json.dumps({
            "schema": "ffm_structural_matched_head_v1",
            "head_only": True,
            "encoder_sha256": parent_sha,
            "encoder": str(warm),
            "trainer": str(trainer_path),
        }, indent=2, sort_keys=True) + "\n")

    print("\nSPAN-STRUCTURAL NEXTLEG COMPLETE")
    for key, value in verdict.items():
        if key not in ("history", "epochs"):
            print(f"  {key:>28}: {value}")
    print(f"  {'candidate':>28}: {out}")
    print("  promotion requires controls, Probe Atlas retention, and downstream Pivot WF")


if __name__ == "__main__":
    main()
