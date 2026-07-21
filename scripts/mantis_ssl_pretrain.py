#!/usr/bin/env python3
"""Stage 1: leakage-safe masked-OHLCV adaptation of public Mantis-8M.

This is the first command in the clean production lineage:

    mask -> contrastive -> seq2seq -> next-leg -> Pivot Trend

It always starts from ``paris-noah/Mantis-8M`` (no local warm checkpoint), requires
the complete roll-safe 9-ticker x 4-timeframe corpus, and physically removes every
bar on or after 2026-01-01 before the tensors used by training are assembled. The
entire year 2026 therefore remains untouched for downstream OOS validation.

Local defaults target Apple MPS and write to a new file under ``temp/``. All costly
training settings can be overridden with environment variables; see ``--help``.
"""
from __future__ import annotations

import argparse
import json
import os
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
PRODUCTION_HOLDOUT_START = "2026-01-01"
DEFAULT_OUT = ROOT / "temp" / "clean_ssl_pre2026" / "mantis_ssl_ohlcv.pt"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean Stage-1 masked-OHLCV pretraining from public Mantis-8M")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", str(ROOT / "data")))
    parser.add_argument("--out", default=os.environ.get("OUT_PATH", str(DEFAULT_OUT)))
    parser.add_argument("--holdout-start", default=os.environ.get(
        "HOLDOUT_START", PRODUCTION_HOLDOUT_START))
    parser.add_argument("--val-frac", type=float, default=float(os.environ.get("VAL_FRAC", "0.1")))
    parser.add_argument("--seq", type=int, default=int(os.environ.get("SEQ", "64")))
    parser.add_argument("--max-jitter", type=int, default=int(os.environ.get("MAX_JITTER", "16")))
    parser.add_argument("--new-channels", type=int, default=int(os.environ.get("NEW_C", "8")))
    parser.add_argument("--mask-ratio", type=float, default=float(os.environ.get("MASK_RATIO", "0.4")))
    parser.add_argument("--batch", type=int, default=None,
                        help="Default: 1024 CUDA, 256 MPS, 64 CPU (or BATCH env)")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "60")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "200")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-4")))
    parser.add_argument("--patience", type=int, default=int(os.environ.get("PATIENCE", "8")))
    parser.add_argument("--controls", default=os.environ.get("CONTROLS", "shuffle,random"))
    parser.add_argument("--probe-folds", type=int, default=int(os.environ.get("PROBE_FOLDS", "1")))
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"),
                        default=os.environ.get("DEVICE"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    parser.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "0")))
    parser.add_argument("--lora-alpha", type=float,
                        default=float(os.environ.get("LORA_ALPHA", "16")))
    parser.add_argument("--lora-dropout", type=float,
                        default=float(os.environ.get("LORA_DROPOUT", "0")))
    parser.add_argument("--log-every-steps", type=int,
                        default=int(os.environ.get("SSL_LOG_EVERY_STEPS", "25")))
    parser.add_argument("--resume", action="store_true", default=_env_bool("RESUME"))
    parser.add_argument("--compile", action="store_true", default=_env_bool("COMPILE"))
    parser.add_argument("--no-probe", action="store_true", default=_env_bool("NO_PROBE"))
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate the clean-data contract without starting training")
    return parser


def _preflight(args: argparse.Namespace) -> tuple[Path, Path, dict]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    # Production rebuilds have one immutable OOS boundary. Fail closed instead of
    # permitting an accidental environment override that trains on any 2026 bars.
    if args.holdout_start != PRODUCTION_HOLDOUT_START:
        raise SystemExit(
            f"HOLDOUT_START must be {PRODUCTION_HOLDOUT_START}; got {args.holdout_start!r}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir}")

    streams = tuple((ticker, timeframe) for ticker in TICKERS for timeframe in TIMEFRAMES)
    provenance = seal_continuous_streams(data_dir, streams, repo_root=ROOT)
    if len(provenance["streams"]) != len(streams):
        raise RuntimeError("provenance seal did not return the exact 9x4 stream universe")

    protected = {
        (ROOT / "checkpoints" / "mantis_ssl_ohlcv.pt").resolve(),
        (ROOT / "checkpoints" / "mantis_ssl_regime.pt").resolve(),
        (ROOT / "checkpoints" / "mantis_ssl_ctr_seq2seq.pt").resolve(),
        (ROOT / "checkpoints" / "mantis_ssl_nextleg.pt").resolve(),
    }
    if out_path in protected:
        raise SystemExit(f"refusing to overwrite an existing promoted checkpoint: {out_path}")
    if out_path.exists() and not args.resume:
        raise SystemExit(f"output already exists: {out_path}\nUse RESUME=1 only to resume this exact run.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return data_dir, out_path, provenance


def main() -> None:
    args = _parser().parse_args()
    device = args.device or _default_device()
    env_batch = os.environ.get("BATCH")
    batch = args.batch if args.batch is not None else (
        int(env_batch) if env_batch else {"cuda": 1024, "mps": 256, "cpu": 64}[device])
    data_dir, out_path, provenance = _preflight(args)

    provenance_path = Path(str(out_path) + ".data_provenance.json")
    provenance_report = {
        **provenance,
        "stage": "mask",
        "base_model": "paris-noah/Mantis-8M",
        "backbone_ckpt": None,
        "lora": {"rank": args.lora_r, "alpha": args.lora_alpha,
                 "dropout": args.lora_dropout},
        "holdout_start": PRODUCTION_HOLDOUT_START,
        "tickers": list(TICKERS),
        "timeframes": list(TIMEFRAMES),
    }
    provenance_path.write_text(json.dumps(provenance_report, indent=2) + "\n")

    print("\nCLEAN STAGE-1 PREFLIGHT PASSED")
    print(f"  data       : {data_dir}")
    print(f"  streams    : {len(TICKERS)} tickers x {len(TIMEFRAMES)} timeframes = 36")
    print(f"  base       : public paris-noah/Mantis-8M (no warm checkpoint)")
    print(f"  holdout    : >= {PRODUCTION_HOLDOUT_START} physically excluded")
    print(f"  device     : {device}")
    print(f"  adaptation : {'LoRA' if args.lora_r else 'full'}"
          + (f" r={args.lora_r} alpha={args.lora_alpha:g}" if args.lora_r else ""))
    print(f"  batch      : {batch}")
    print(f"  output     : {out_path}")
    print(f"  provenance : {provenance_path}", flush=True)
    if args.preflight_only:
        return

    controls = tuple(x.strip() for x in args.controls.split(",") if x.strip())
    verdict = ssl.loop_ssl(
        data_dir=str(data_dir), out_path=str(out_path),
        tickers=TICKERS, tfs=TIMEFRAMES,
        pretext="mask", backbone_ckpt=None,
        mask_ratio=args.mask_ratio, seq=args.seq, max_jitter=args.max_jitter,
        new_channels=args.new_channels, batch=batch, epochs=args.epochs,
        steps_per_epoch=args.steps, lr=args.lr, patience=args.patience,
        val_frac=args.val_frac, holdout_start=PRODUCTION_HOLDOUT_START,
        controls=controls, probe=not args.no_probe, probe_folds=args.probe_folds,
        resume=args.resume, device=device, compile_model=args.compile, seed=args.seed,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        log_every_steps=args.log_every_steps,
    )

    print("\n" + "=" * 60)
    print("CLEAN STAGE-1 VERDICT")
    print("=" * 60)
    for key, value in verdict.items():
        if key != "epochs":
            print(f"  {key:>22}: {value}")
    print(f"\nencoder   -> {out_path}")
    print(f"report    -> {out_path}.report.json")
    print(f"lineage   -> {provenance_path}")
    print("NEXT: pass this exact encoder as WARM_CKPT to mantis_ssl_contrastive.py")


if __name__ == "__main__":
    main()
