#!/usr/bin/env python3
"""Train causal-range NextLeg race v2 from a frozen Structural NextLeg parent.

This is an independent public SSL stage.  It consumes the clean 9x4 continuous-bar corpus and
exports two matched artifacts:

* ``*.pt``: merged LoRA encoder checkpoint;
* ``*.report.json`` plus Probe Atlas output: leakage/control/retention evidence.

The objective uses raw candle geometry only.  There is no ATR, entry, stop, R multiple, cost, or
strategy label.  Inputs end at a confirmed fractal pivot; every future target is fully contained
inside one source stream, one train/validation split, and the pre-2026 region.
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

SOURCE_PATH = (Path(__file__).resolve() if "__file__" in globals() else None)
ROOT = SOURCE_PATH.parents[1] if SOURCE_PATH is not None else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if (ROOT / "futures_foundation").is_dir():
    from futures_foundation.data_provenance import seal_continuous_streams, sha256
    from futures_foundation.finetune import ssl
    from futures_foundation.finetune.pretext.nextleg_race import RACE_SCHEMA

TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")
HOLDOUT_START = "2026-01-01"
GITHUB_REPOSITORY = "https://github.com/johnamcruz/Futures-Foundation-Model.git"
COLAB_CHECKOUT = Path("/content/ffm")


def _stream_command(command, *, cwd=None, log_path=None, env=None):
    """Run one visible command and optionally mirror its combined output to Drive."""
    print("[colab] $ " + " ".join(str(part) for part in command), flush=True)
    handle = None
    code = None
    try:
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            handle = open(log_path, "a", buffering=1)
        process = subprocess.Popen(
            [str(part) for part in command], cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if handle is not None:
                handle.write(line)
        code = process.wait()
    finally:
        if handle is not None:
            handle.close()
    if code:
        raise subprocess.CalledProcessError(code, command)


def _discover_probe_labels(ai_models):
    """Resolve the existing pre-2026 Probe Atlas labels from Google Drive."""
    ai_models = Path(ai_models)
    requested_labels = os.environ.get("TREND_LABELS")
    if requested_labels:
        labels = Path(requested_labels)
        if not labels.is_file():
            raise FileNotFoundError(f"TREND_LABELS does not exist: {labels}")
    else:
        label_candidates = sorted(
            ai_models.glob("**/trend_lifecycle_labels_pre2026.npz"),
            key=lambda path: (
                1 if "clean_ssl_pre2026_lora" in str(path).lower() else 0,
                path.stat().st_mtime),
            reverse=True)
        labels = label_candidates[0] if label_candidates else None
    return labels


def _require_materialized_checkpoint(path):
    """Reject a missing Git LFS download before torch reports an opaque unpickling error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {path}")
    with open(path, "rb") as handle:
        prefix = handle.read(64)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
        raise RuntimeError(
            f"checkpoint is still a Git LFS pointer: {path}; run `git lfs pull`")
    if path.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"checkpoint is unexpectedly small: {path}")


def _run_canonical_from_github_if_needed():
    """Bootstrap a blank Colab cell, then invoke this script from the GitHub checkout.

    The function returns ``True`` only for the outer pasted/uploaded Colab copy.  The canonical
    script inside ``/content/ffm`` sees a real repository root and proceeds directly to ``main``.
    """
    if not Path("/content").is_dir() or (ROOT / "futures_foundation").is_dir():
        return False

    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    branch = os.environ.get("FFM_BRANCH", "main")
    if COLAB_CHECKOUT.exists():
        if not (COLAB_CHECKOUT / ".git").is_dir():
            raise RuntimeError(
                f"{COLAB_CHECKOUT} exists but is not an FFM git checkout")
        status = subprocess.run(
            ["git", "-C", str(COLAB_CHECKOUT), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout.strip()
        if status:
            raise RuntimeError(
                f"{COLAB_CHECKOUT} has local changes; refusing to overwrite them:\n{status}")
        _stream_command(["git", "-C", str(COLAB_CHECKOUT), "fetch", "origin", branch])
        _stream_command(["git", "-C", str(COLAB_CHECKOUT), "checkout", branch])
        _stream_command([
            "git", "-C", str(COLAB_CHECKOUT), "pull", "--ff-only", "origin", branch])
    else:
        _stream_command([
            "git", "clone", "--branch", branch, "--single-branch",
            GITHUB_REPOSITORY, str(COLAB_CHECKOUT)])
    _stream_command([
        sys.executable, "-m", "pip", "install", "-q",
        "-e", str(COLAB_CHECKOUT), "mantis-tsfm"])
    _stream_command([
        sys.executable, "-c",
        "import safetensors; from mantis.architecture import Mantis8M; "
        "print('[colab] Mantis import OK; safetensors=' + safetensors.__version__)",
    ])

    canonical = COLAB_CHECKOUT / "scripts" / "mantis_ssl_nextleg_race.py"
    atlas = COLAB_CHECKOUT / "scripts" / "probe_atlas.py"
    parent = COLAB_CHECKOUT / "checkpoints" / "mantis_ssl_structural_nextleg.pt"
    missing = [
        path for path in (canonical, atlas, parent) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "GitHub checkout is missing required canonical scripts/checkpoints: "
            + ", ".join(str(path) for path in missing))
    _require_materialized_checkpoint(parent)
    output_dir = Path(os.environ.get(
        "RACE_OUTPUT_DIR", "/content/drive/MyDrive/AI_Models/nextleg_race_v2"))
    output = Path(os.environ.get(
        "OUT_PATH", str(output_dir / "mantis_ssl_nextleg_race_v2.pt")))
    data_dir = Path(os.environ.get(
        "DATA_DIR", "/content/drive/MyDrive/Futures Data"))
    labels = _discover_probe_labels("/content/drive/MyDrive/AI_Models")
    log = Path(os.environ.get(
        "LOG_PATH", str(output_dir / "logs" / "colab_pipeline.log")))
    completed, failed = output_dir / ".completed", output_dir / ".failed"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed.unlink(missing_ok=True)
    failed.unlink(missing_ok=True)
    # A pasted notebook cell inherits IPython's ``-f kernel.json`` arguments. Only forward
    # command-line arguments when this outer bootstrap was executed as an actual .py file.
    forwarded = sys.argv[1:] if SOURCE_PATH is not None else []
    print(f"[colab] repository Structural parent -> {parent}", flush=True)
    print(f"[colab] Probe Atlas labels -> {labels or 'not found'}", flush=True)
    command = [
        sys.executable, str(canonical),
        "--data-dir", str(data_dir),
        "--warm-ckpt", str(parent),
        "--out", str(output),
        "--device", "cuda",
        "--resume",
    ]
    if labels is not None:
        command += ["--atlas-labels", str(labels)]
    elif os.environ.get("SKIP_ATLAS") == "1":
        command.append("--skip-atlas")
    else:
        raise FileNotFoundError(
            "Probe Atlas labels were not found under MyDrive/AI_Models. "
            "Set TREND_LABELS to the existing trend_lifecycle_labels_pre2026.npz, "
            "or set SKIP_ATLAS=1 only for an intentional training-only run.")
    command += forwarded
    try:
        _stream_command(
            command,
            cwd=COLAB_CHECKOUT, log_path=log)
    except BaseException:
        failed.touch()
        print(f"[colab] FAILED; durable log: {log}", flush=True)
        raise
    completed.touch()
    print(f"[colab] COMPLETE; durable log: {log}", flush=True)
    return True


def _csv(value):
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _floats(value):
    return tuple(float(part) for part in _csv(value))


def _default_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get(
        "DATA_DIR", "/content/drive/MyDrive/Futures Data" if Path("/content").is_dir()
        else str(ROOT / "data")))
    parser.add_argument("--warm-ckpt", default=os.environ.get(
        "WARM_CKPT", "/content/drive/MyDrive/AI_Models/clean_ssl_pre2026_lora/"
        "mantis_ssl_structural_nextleg.pt" if Path("/content").is_dir()
        else str(ROOT / "checkpoints" / "mantis_ssl_structural_nextleg.pt")))
    parser.add_argument("--out", default=os.environ.get(
        "OUT_PATH", "/content/drive/MyDrive/AI_Models/nextleg_race_v2/"
        "mantis_ssl_nextleg_race_v2.pt" if Path("/content").is_dir()
        else str(ROOT / "temp" / "nextleg_race_v2" / "mantis_ssl_nextleg_race_v2.pt")))
    parser.add_argument("--atlas-labels", default=os.environ.get("TREND_LABELS"),
                        help="existing Probe Atlas lifecycle labels; defaults beside parent")
    parser.add_argument("--tickers", default=",".join(TICKERS))
    parser.add_argument("--tfs", default=",".join(TIMEFRAMES))
    parser.add_argument("--sampling-mode", choices=("bar_proportional", "uniform_stream"),
                        default=os.environ.get("SAMPLING_MODE", "bar_proportional"))
    parser.add_argument("--controls", default=os.environ.get("SSL_CONTROLS", "shuffle,random"))
    parser.add_argument("--control-epochs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("RACE_EPOCHS", "60")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("RACE_STEPS", "50")))
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=float(os.environ.get("RACE_LR", "1e-5")))
    parser.add_argument("--head-lr", type=float,
                        default=float(os.environ.get("RACE_HEAD_LR", "1e-4")))
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--leg-k", type=int, default=2)
    parser.add_argument("--leg-cap", type=int, default=256)
    parser.add_argument("--race-levels", default=os.environ.get("RACE_LEVELS", "1,2,3,4"))
    parser.add_argument("--race-scale-lookback", type=int, default=64)
    parser.add_argument("--race-cap", type=float, default=8.0)
    parser.add_argument("--race-w", type=float, default=.5)
    parser.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "8")))
    parser.add_argument("--lora-alpha", type=float,
                        default=float(os.environ.get("LORA_ALPHA", "16")))
    parser.add_argument("--freeze-encoder-layers", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=os.environ.get("RESUME") == "1")
    parser.add_argument("--skip-atlas", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _resolve(args):
    data = Path(args.data_dir).expanduser().resolve()
    parent = Path(args.warm_ckpt).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    labels = (Path(args.atlas_labels).expanduser().resolve() if args.atlas_labels else
              parent.parent / "probe_atlas" / "trend_lifecycle_labels_pre2026.npz")
    if not data.is_dir():
        raise FileNotFoundError(f"data directory not found: {data}")
    if not parent.is_file():
        raise FileNotFoundError(f"Structural NextLeg parent not found: {parent}")
    if output == parent:
        raise SystemExit("--out must differ from the immutable parent checkpoint")
    if output.exists() and not args.resume:
        raise SystemExit(f"output already exists: {output}; pass --resume to continue")
    if not args.skip_atlas and not labels.is_file():
        raise FileNotFoundError(
            f"existing Probe Atlas labels not found: {labels}; pass --atlas-labels or --skip-atlas")
    output.parent.mkdir(parents=True, exist_ok=True)
    return data, parent, output, labels


def _run_atlas(*, data, checkpoint, labels, output, device, batch):
    atlas_dir = output.parent / "probe_atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    result = atlas_dir / "nextleg_race_v2.json"
    cache = atlas_dir / "nextleg_race_v2_emb.npy"
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
        raise RuntimeError("Probe Atlas result is not bound to the race checkpoint")
    return result


def main():
    args = _parser().parse_args()
    data, parent, output, labels = _resolve(args)
    device = args.device or _default_device()
    batch = args.batch or {"cuda": 512, "mps": 128, "cpu": 32}[device]
    tickers, timeframes = _csv(args.tickers), _csv(args.tfs)
    expected = tuple((ticker, timeframe) for ticker in tickers for timeframe in timeframes)
    provenance = seal_continuous_streams(data, expected, repo_root=ROOT)
    if len(provenance["streams"]) != len(expected):
        raise RuntimeError("data provenance did not seal the requested stream matrix")
    levels = _floats(args.race_levels)
    controls = _csv(args.controls)
    if set(controls) - {"shuffle", "random"}:
        raise ValueError(f"unsupported controls: {controls}")
    if args.lora_r <= 0:
        raise ValueError("NextLeg race v2 requires LoRA; --lora-r must be positive")
    if args.race_scale_lookback > 64:
        raise ValueError("race scale lookback must fit the shortest 64-bar context")

    run_contract = {
        **provenance, "stage": "nextleg_race", "race_schema": RACE_SCHEMA,
        "parent": {"path": str(parent), "sha256": sha256(parent)},
        "holdout_start": HOLDOUT_START,
        "sampling_mode": args.sampling_mode, "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha, "race_levels": levels,
        "race_scale": "median_completed_candle_high_minus_low",
        "uses_atr": False,
    }
    print("\nNEXTLEG RACE V2 PREFLIGHT PASSED")
    print(f"  data       : {data}")
    print(f"  streams    : {len(tickers)} x {len(timeframes)} = {len(expected)}")
    print(f"  holdout    : >= {HOLDOUT_START} physically excluded")
    print(f"  parent     : {parent}")
    print("  task heads : disposable; not inherited or exported")
    print(f"  target     : reach/adverse/time @ {levels} median candle-range units")
    print("  ATR/R      : none (raw completed candle high-low scale)")
    print(f"  tuning     : LoRA r={args.lora_r} alpha={args.lora_alpha:g} "
          f"freeze={args.freeze_encoder_layers}")
    print(f"  training   : {device} batch={batch} epochs={args.epochs} steps={args.steps} "
          f"encoder_lr={args.lr:g} head_lr={args.head_lr:g}")
    print(f"  controls   : {controls}")
    print(f"  output     : {output}")
    if args.preflight_only:
        return
    Path(str(output) + ".data_provenance.json").write_text(
        json.dumps(run_contract, indent=2, default=list) + "\n")

    verdict = ssl.loop_ssl(
        data_dir=str(data), tickers=tickers, tfs=timeframes, out_path=str(output),
        holdout_start=HOLDOUT_START, val_frac=.1, pretext="nextleg_race",
        backbone_ckpt=str(parent),
        sampling_mode=args.sampling_mode, controls=controls,
        control_epochs=args.control_epochs, probe=True,
        horizons=(5, 10, 20, 25), context_lengths=(64, 100, 150, 200),
        leg_k=args.leg_k, leg_cap=args.leg_cap, leg_w=1.0, mse_weight=1.0,
        race_w=args.race_w, race_cap=args.race_cap, race_levels=levels,
        race_scale_lookback=args.race_scale_lookback,
        new_channels=3, batch=batch, epochs=args.epochs, steps_per_epoch=args.steps,
        lr=args.lr, head_lr=args.head_lr, weight_decay=0.0, patience=args.patience,
        freeze_encoder_layers=args.freeze_encoder_layers,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        clamp=10.0, grad_clip=1.0, resume=args.resume, device=device, seed=args.seed)

    if not verdict.get("all_pass"):
        raise RuntimeError(f"NextLeg race v2 failed SSL/control gates: {verdict}")

    atlas = None if args.skip_atlas else _run_atlas(
        data=data, checkpoint=output, labels=labels, output=output,
        device=device, batch=batch)
    print("\nNEXTLEG RACE V2 COMPLETE")
    print(f"  encoder : {output}")
    print("  heads   : discarded; foundation contract is encoder-only")
    print(f"  report  : {output}.report.json")
    print(f"  atlas   : {atlas or 'skipped'}")


if __name__ == "__main__":
    if not _run_canonical_from_github_if_needed():
        main()
