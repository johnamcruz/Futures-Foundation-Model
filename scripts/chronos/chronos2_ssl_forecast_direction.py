#!/usr/bin/env python3
"""Chronos-2 forecast-direction SSL (A0/A1/A2) on the Mask v5 LoRA.

Frozen contract: docs/chronos2_forecast_direction_ssl.md.

  preflight  authenticate streams and count anchors per period
  evaluate   score a checkpoint (``base`` or a PEFT adapter) per stream x horizon
  train      continue the parent LoRA: A1 native pinball, A2 + direction BCE,
             A3 = A2 + past-only bar-structure inputs
  select     choose λ from select-period reports and freeze the choice
  gate       forecast (A1 vs A0), direction (A2 vs A1) or retention (Atlas) gate

The outer period is one-shot: ``evaluate --periods outer`` requires
``--confirm-outer-frozen`` after every selection choice is recorded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.classifiers.chronos2 import forecast_ssl as fs


RUN_ROOT = ROOT / "temp/chronos2_small_36stream/forecast_direction"
PARENT = ROOT / "temp/chronos2_small_36stream/mask_full/checkpoint"
BASE_SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub/models--autogluon--chronos-2-small"
    / "snapshots/ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a")
SMOKE = {"tickers": ("NQ", "ES"), "timeframes": ("3min",), "epochs": 1, "steps": 5,
         "select_anchors": 20, "anchors_per_stream": 100, "baseline_train": 300}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = value.add_subparsers(dest="command", required=True)

    def common(command):
        command.add_argument("--data-dir", type=Path, default=ROOT / "data")
        command.add_argument("--out-root", type=Path, default=RUN_ROOT)
        command.add_argument("--base-snapshot", type=Path, default=BASE_SNAPSHOT)
        command.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
        command.add_argument("--smoke", action="store_true",
                             help="NQ/ES@3min, tiny counts, writes under <out-root>/smoke")
        command.add_argument("--context-length", type=int, default=fs.CONTEXT_LENGTH,
                             help="bars of context (multiple of 16)")

    common(commands.add_parser("preflight"))

    evaluate = commands.add_parser("evaluate")
    common(evaluate)
    evaluate.add_argument("--checkpoint", required=True, help="'base' or a PEFT adapter dir")
    evaluate.add_argument("--name", required=True)
    evaluate.add_argument("--periods", default="select")
    evaluate.add_argument("--anchors-per-stream", type=int, default=2000)
    evaluate.add_argument("--baseline-train", type=int, default=3000)
    evaluate.add_argument("--batch-windows", type=int, default=256)
    evaluate.add_argument("--confirm-outer-frozen", action="store_true")
    evaluate.add_argument("--bar-features", action="store_true",
                          help="append past-only bar-structure inputs (A3 checkpoints)")

    train = commands.add_parser("train")
    common(train)
    train.add_argument("--arm", choices=fs.ARMS, required=True)
    train.add_argument("--direction-weight", type=float, default=0.0)
    train.add_argument("--parent", type=Path, default=PARENT)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--batch-windows", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--select-anchors", type=int, default=100)
    train.add_argument("--anchors-per-stream", type=int, default=2000)
    train.add_argument("--baseline-train", type=int, default=3000)
    train.add_argument("--mask-ties", action="store_true",
                       help="ignore unchanged closes in the direction loss")

    select = commands.add_parser("select")
    select.add_argument("--a1", type=Path, required=True, help="A1 select-period eval JSON")
    select.add_argument("--a2", action="append", required=True, metavar="LAMBDA=PATH",
                        help="A2 select-period eval JSON per direction weight")
    select.add_argument("--out", type=Path, required=True)

    probe = commands.add_parser("probe", help="REG-embedding side probe (consumer view)")
    common(probe)
    probe.add_argument("--checkpoint", required=True, help="'base' or a PEFT adapter dir")
    probe.add_argument("--name", required=True)
    probe.add_argument("--bar-features", action="store_true")
    probe.add_argument("--train-per-stream", type=int, default=1500)
    probe.add_argument("--eval-per-stream", type=int, default=1500)

    rank = commands.add_parser("rank", help="rank arms by the pre-registered checks")
    rank.add_argument("--eval-dir", type=Path, default=RUN_ROOT / "eval")
    rank.add_argument("--prefix", default="rescore_")
    rank.add_argument("--reference", default="a1")
    rank.add_argument("--out", type=Path, default=RUN_ROOT / "ranking.json")

    gate = commands.add_parser("gate")
    gate.add_argument("--kind", choices=("forecast", "direction", "retention"), required=True)
    gate.add_argument("--candidate", type=Path, required=True)
    gate.add_argument("--reference", type=Path, required=True)
    gate.add_argument("--out", type=Path)
    return value


def parse(argv=None) -> argparse.Namespace:
    command = parser()
    args = command.parse_args(argv)
    if args.command == "evaluate":
        args.periods = tuple(item.strip() for item in args.periods.split(",") if item.strip())
        unknown = set(args.periods) - {"select", "outer"}
        if not args.periods or unknown:
            command.error(f"--periods must be select and/or outer, got {args.periods}")
        if "outer" in args.periods and not args.confirm_outer_frozen:
            command.error("outer is one-shot: pass --confirm-outer-frozen only after "
                          "every selection choice is recorded")
    return args


def run_settings(args: argparse.Namespace) -> dict:
    """Resolve streams, counts and output directory for a train/evaluate run."""
    smoke = getattr(args, "smoke", False)
    settings = {
        "tickers": SMOKE["tickers"] if smoke else fs.TICKERS,
        "timeframes": SMOKE["timeframes"] if smoke else fs.TIMEFRAMES,
        "epochs": SMOKE["epochs"] if smoke else getattr(args, "epochs", None),
        "steps": SMOKE["steps"] if smoke else getattr(args, "steps", None),
        "select_anchors": SMOKE["select_anchors"] if smoke else getattr(args, "select_anchors", None),
        "anchors_per_stream": (SMOKE["anchors_per_stream"] if smoke
                               else getattr(args, "anchors_per_stream", None)),
        "baseline_train": SMOKE["baseline_train"] if smoke else getattr(args, "baseline_train", None),
    }
    root = args.out_root / "smoke" if smoke else args.out_root
    if args.command == "train":
        weight = "" if args.arm == "a1" else f"_lam{args.direction_weight:g}"
        ties = "_ties-masked" if getattr(args, "mask_ties", False) else ""
        context = ("" if args.context_length == fs.CONTEXT_LENGTH
                   else f"_ctx{args.context_length}")
        settings["out_dir"] = root / f"{args.arm}{weight}{context}{ties}_seed{args.seed}"
    else:
        settings["out_dir"] = root
    return settings


def _load_model(checkpoint: str, base_snapshot: Path, device: str):
    if checkpoint == "base":
        from chronos import Chronos2Pipeline

        return Chronos2Pipeline.from_pretrained(str(base_snapshot), device_map=device).model
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _chronos_base_identity, _load_trainable_adapter)

    identity = _chronos_base_identity(Path(checkpoint), base_snapshot)
    _, model = _load_trainable_adapter(
        checkpoint, device, base_revision=identity["revision"], base_snapshot=base_snapshot)
    return model


def _checkpoint_identity(checkpoint: str, base_snapshot: Path) -> dict:
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import tree_sha256

    if checkpoint == "base":
        return {"checkpoint": "base", "base_snapshot": str(base_snapshot)}
    return {"checkpoint": checkpoint, "sha256": tree_sha256(Path(checkpoint))}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _markdown(report: dict, name: str) -> str:
    lines = [f"# {name}: {report['period']}", "",
             "| horizon | mean AUC | min AUC | streams > 0.5 | baseline AUC | "
             "shuffled AUC | scaled WQL | cov80 | big-move AUC | big-move baseline | "
             "breakout-side AUC | breakout-side baseline | liquidity-break AUC | "
             "liquidity-break baseline | stack gain over baseline |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, row in report["summary"].items():
        big = (f"{row['mean_expansion_auc']:.4f} | {row['mean_expansion_baseline_auc']:.4f}"
               if "mean_expansion_auc" in row else "n/a | n/a")
        event = (f"{row['mean_event_auc']:.4f} | {row['mean_event_baseline_auc']:.4f}"
                 if "mean_event_auc" in row else "n/a | n/a")
        stack = (f"{row['mean_stack_gain']:+.4f} ({row['streams_stack_gain_positive']}/{row['streams']})"
                 if "mean_stack_gain" in row else "n/a")
        side = (f"{row['mean_first_passage_side_auc']:.4f} | "
                f"{row['mean_first_passage_side_baseline_auc']:.4f}"
                if "mean_first_passage_side_auc" in row else "n/a | n/a")
        lines.append(
            f"| {key} | {row['mean_auc']:.4f} | {row['min_auc']:.4f} | "
            f"{row['streams_auc_above_half']}/{row['streams']} | "
            f"{row['mean_baseline_auc']:.4f} | {row['mean_shuffled_auc']:.4f} | "
            f"{row['mean_scaled_wql']:.4f} | {row['mean_coverage_80']:.3f} | {big} | {side} | {event} | {stack} |")
    return "\n".join(lines) + "\n"


def _evaluate(model, streams, name, periods, settings, args, identity) -> dict:
    reports = {}
    for period in periods:
        report = fs.evaluate(
            model, streams, period, device=args.device,
            anchors_per_stream=settings["anchors_per_stream"],
            baseline_train_per_stream=settings["baseline_train"],
            batch_windows=getattr(args, "batch_windows", 256) if args.command == "evaluate" else 256,
            bar_features=(args.bar_features if args.command == "evaluate"
                          else fs.uses_bar_features(args.arm)),
            context_length=args.context_length)
        rows = report.pop("_rows", {})
        report.update({"name": name, **identity})
        destination = settings["out_dir"] / "eval"
        if rows:
            import numpy as np

            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / f"{name}_{period}_rows.npz", **{
                f"{stream}|{field}": value
                for stream, arrays in rows.items() for field, value in arrays.items()})
        _write(destination / f"{name}_{period}.json", report)
        (destination / f"{name}_{period}.md").write_text(_markdown(report, name))
        print(_markdown(report, name), flush=True)
        reports[period] = report
    return reports


def main(argv=None) -> None:
    args = parse(argv)
    if args.command == "select":
        from datetime import datetime, timezone

        paths = {float(item.split("=", 1)[0]): Path(item.split("=", 1)[1]) for item in args.a2}
        choice = fs.select_direction_weight(
            json.loads(args.a1.read_text()),
            {weight: json.loads(path.read_text()) for weight, path in paths.items()})
        choice.update({
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {"a1": str(args.a1),
                       "a2": {str(weight): str(path) for weight, path in paths.items()}},
            "scores": {str(weight): row for weight, row in choice["scores"].items()},
        })
        _write(args.out, choice)
        print(json.dumps(choice, indent=2), flush=True)
        return
    if args.command == "rank":
        evals = {path.name[len(args.prefix):-len("_select.json")]: json.loads(path.read_text())
                 for path in sorted(args.eval_dir.glob(f"{args.prefix}*_select.json"))}
        probes = {path.name[:-len("_regprobe_select.json")]: json.loads(path.read_text())
                  for path in sorted(args.eval_dir.glob("*_regprobe_select.json"))}
        table = fs.rank_arms(evals, probes, reference=args.reference)
        _write(args.out, {"reference": args.reference, "ranking": table})
        for row in table:
            print(f"{row['arm']:>12} passes={row['passes']}/{len(row['checks'])} "
                  + " ".join(f"{k}={'Y' if v else '-'}" for k, v in row["checks"].items())
                  + f" dAUC={row['delta_vs_reference']}", flush=True)
        return
    if args.command == "gate":
        candidate = json.loads(args.candidate.read_text())
        reference = json.loads(args.reference.read_text())
        gate = {"forecast": fs.forecast_gate, "direction": fs.direction_gate,
                "retention": fs.retention_gate}[args.kind](candidate, reference)
        gate.update({"kind": args.kind, "candidate": str(args.candidate),
                     "reference": str(args.reference)})
        if args.out:
            _write(args.out, gate)
        print(json.dumps(gate, indent=2), flush=True)
        return

    settings = run_settings(args)
    streams, provenance = fs.load_streams(
        args.data_dir, tickers=settings["tickers"], timeframes=settings["timeframes"],
        repo_root=ROOT)
    if args.command == "preflight":
        counts = {
            name: {period: int(max(0, hi - lo)) for period, (lo, hi) in (
                (period, fs.period_bounds(stream.close_ns, *bounds))
                for period, bounds in fs.PERIODS.items())}
            for name, stream in sorted(streams.items())}
        payload = {"schema": "ffm_chronos2_forecast_direction_preflight_v1",
                   "periods": fs.PERIODS, "holdout_start": fs.HOLDOUT_START,
                   "anchor_counts": counts, "provenance": provenance}
        _write(settings["out_dir"] / "preflight.json", payload)
        low = {name: row for name, row in counts.items() if min(row.values()) < 1000}
        print(f"[forecast-preflight] PASS streams={len(counts)} "
              f"low_anchor_streams={sorted(low)}", flush=True)
        return

    if args.command == "probe":
        model = _load_model(args.checkpoint, args.base_snapshot, args.device)
        report = fs.reg_probe_report(
            model, streams, device=args.device, bar_features=args.bar_features,
            context_length=args.context_length,
            train_per_stream=300 if args.smoke else args.train_per_stream,
            eval_per_stream=300 if args.smoke else args.eval_per_stream)
        report.update({"name": args.name,
                       **_checkpoint_identity(args.checkpoint, args.base_snapshot)})
        _write(settings["out_dir"] / "eval" / f"{args.name}_regprobe_select.json", report)
        for target, sources in report["targets"].items():
            print(f"[reg-probe:{args.name}] {target}: " + " ".join(
                f"{source}={row['mean_auc']:.4f}({row['streams_above_half']}/{row['streams']}"
                f",shuf={row['mean_shuffled_auc']:.3f})" for source, row in sources.items()),
                flush=True)
        return

    if args.command == "evaluate":
        model = _load_model(args.checkpoint, args.base_snapshot, args.device)
        _evaluate(model, streams, args.name, args.periods, settings, args,
                  _checkpoint_identity(args.checkpoint, args.base_snapshot))
        return

    report = fs.train_forecast_direction(
        streams, parent=args.parent, base_snapshot=args.base_snapshot,
        out_dir=settings["out_dir"], provenance=provenance, arm=args.arm,
        direction_weight=args.direction_weight, device=args.device, seed=args.seed,
        epochs=settings["epochs"], steps_per_epoch=settings["steps"],
        batch_windows=args.batch_windows, learning_rate=args.lr,
        weight_decay=args.weight_decay, patience=args.patience,
        select_anchors_per_stream=settings["select_anchors"], repo_root=ROOT,
        mask_ties=args.mask_ties, context_length=args.context_length)
    checkpoint = report["checkpoint"]["path"]
    model = _load_model(checkpoint, args.base_snapshot, args.device)
    _evaluate(model, streams, settings["out_dir"].name, ("select",), settings, args,
              _checkpoint_identity(checkpoint, args.base_snapshot))


if __name__ == "__main__":
    main()
