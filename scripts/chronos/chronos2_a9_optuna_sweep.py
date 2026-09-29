#!/usr/bin/env python3
"""TPE sweep for A9: direction BCE on the consumer REG embedding.

Config-driven (configs/sweep/*.json), SQLite study (resumable, dashboard-ready),
TPE sampler, configurable pruner.  Every trial runs in this one process, one at
a time: train (reporting each epoch's select direction BCE to the pruner) ->
select evaluation -> REG-embedding probe -> objective score.

Objective (select period only):
    mean REG-probe direction AUC over the probe horizons (all bars, ties
    excluded) minus a penalty only when forecast WQL exceeds budget x A1.

Usage:
    python scripts/chronos/chronos2_a9_optuna_sweep.py --n-trials 8
    python scripts/chronos/chronos2_a9_optuna_sweep.py --n-trials 8 --resume
    python scripts/chronos/chronos2_a9_optuna_sweep.py --smoke      # 2 tiny trials, NQ/ES@3min
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.classifiers.chronos2 import forecast_ssl as fs


DEFAULT_CONFIG = ROOT / "configs" / "sweep" / "chronos2_a9_direction_v1.json"
BASE_SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--autogluon--chronos-2-small"
                 / "snapshots/ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a")
PRUNERS = ("percentile", "median", "hyperband", "none")
SMOKE = {"tickers": ("NQ", "ES"), "timeframes": ("3min",), "epochs": 1, "steps_per_epoch": 5,
         "select_anchors_per_stream": 20, "anchors_per_stream": 100,
         "baseline_train_per_stream": 300, "probe_train_per_stream": 300,
         "probe_eval_per_stream": 300}


def load_config(path: Path) -> dict:
    """Read and validate a sweep config."""
    config = json.loads(Path(path).read_text())
    for section in ("study", "search_space", "objective", "training", "evaluation"):
        if section not in config:
            raise ValueError(f"sweep config missing section {section!r}")
    pruner = config["study"].get("pruner", {}).get("type", "none")
    if pruner not in PRUNERS:
        raise ValueError(f"unsupported pruner {pruner!r}; expected one of {PRUNERS}")
    if config["study"].get("sampler", {}).get("type", "tpe") != "tpe":
        raise ValueError("only the TPE sampler is supported")
    for horizons in config["search_space"]["reg_horizons"]:
        values = [int(h) for h in horizons.split(",")]
        if min(values) < 1 or max(values) > fs.FORECAST_LENGTH:
            raise ValueError(f"reg_horizons {horizons!r} outside 1..{fs.FORECAST_LENGTH}")
    return config


def make_sampler(optuna, config: dict):
    sampler = config["study"].get("sampler", {})
    return optuna.samplers.TPESampler(seed=int(sampler.get("seed", 0)),
                                      n_startup_trials=int(sampler.get("n_startup_trials", 3)))


def make_pruner(optuna, config: dict):
    """study.pruner: {"type": "percentile" | "median" | "hyperband" | "none", ...}.

    ``percentile`` prunes only trials below the given percentile of earlier
    trials at the same epoch (25 = bottom quartile): gentler than the median,
    because the per-epoch value is a noisy stand-in for the final probe score.
    """
    pruner = config["study"].get("pruner", {})
    kind = pruner.get("type", "none")
    if kind == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=float(pruner.get("percentile", 25.0)),
            n_startup_trials=int(pruner.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner.get("n_warmup_steps", 5)),
            interval_steps=int(pruner.get("interval_steps", 1)))
    if kind == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner.get("n_startup_trials", 3)),
            n_warmup_steps=int(pruner.get("n_warmup_steps", 3)),
            interval_steps=int(pruner.get("interval_steps", 1)))
    if kind == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=int(pruner.get("min_resource", 3)),
            max_resource=int(config["training"]["epochs"]),
            reduction_factor=int(pruner.get("reduction_factor", 3)))
    return optuna.pruners.NopPruner()


def intermediate_value(metric: dict) -> float:
    """Value reported to the pruner each epoch: higher is better, like the objective."""
    return -float(metric["direction_bce"])


def trial_name(params: dict, seed: int) -> str:
    horizons = "-".join(params["reg_horizons"].split(","))
    return (f"a9_lam{params['direction_weight']:g}_focal{params['focal_gamma']:g}"
            f"_h{horizons}_lr{params['learning_rate']:g}_seed{seed}")


def top_k_trials(trials: list[dict], k: int) -> list[dict]:
    """Best ``k`` completed trials by score (pruned/failed trials excluded)."""
    completed = [t for t in trials if t.get("state") == "COMPLETE" and t.get("score") is not None]
    return sorted(completed, key=lambda t: t["score"], reverse=True)[:k]


def confirmation_winner(runs: dict[str, list[float]]) -> dict:
    """Pick the setup with the best mean score across seeds (not the luckiest seed)."""
    import statistics

    means = {key: statistics.fmean(values) for key, values in runs.items()}
    stdevs = {key: statistics.pstdev(values) for key, values in runs.items()}
    return {"winner": max(means, key=means.get), "means": means, "stdevs": stdevs, "runs": runs}


def wait_until_idle(poll_seconds: int = 60) -> None:
    """One job at a time: wait while another Chronos training/evaluation process runs."""
    me = str(os.getpid())
    while True:
        found = subprocess.run(
            ["pgrep", "-f", r"chronos2_ssl_forecast_direction.py (train|evaluate|probe)"],
            capture_output=True, text=True).stdout.split()
        if not [pid for pid in found if pid != me]:
            return
        time.sleep(poll_seconds)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    temporary.replace(path)


def _load_adapter(checkpoint: Path, device: str):
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _chronos_base_identity, _load_trainable_adapter)

    identity = _chronos_base_identity(checkpoint, BASE_SNAPSHOT)
    _, model = _load_trainable_adapter(checkpoint, device, base_revision=identity["revision"],
                                       base_snapshot=BASE_SNAPSHOT)
    return model


def _release(device: str) -> None:
    gc.collect()
    if device == "mps":
        import torch

        torch.mps.empty_cache()


def run(args: argparse.Namespace) -> dict:
    import optuna

    config = load_config(args.config)
    smoke = args.smoke
    training, evaluation, objective = config["training"], dict(config["evaluation"]), config["objective"]
    if smoke:
        training = {**training, **{k: SMOKE[k] for k in ("epochs", "steps_per_epoch",
                                                          "select_anchors_per_stream")}}
        evaluation.update({k: SMOKE[k] for k in ("anchors_per_stream", "baseline_train_per_stream",
                                                  "probe_train_per_stream", "probe_eval_per_stream")})
    study_name = config["study"]["name"] + ("_smoke" if smoke else "")
    storage_path = ROOT / config["study"]["storage"]
    if smoke:
        storage_path = storage_path.with_name("study_smoke.db")
    sweep_dir = storage_path.parent
    sweep_dir.mkdir(parents=True, exist_ok=True)
    if smoke and storage_path.exists():
        storage_path.unlink()
    storage = f"sqlite:///{storage_path}"
    if storage_path.exists() and not (args.resume or smoke or args.confirm):
        raise SystemExit(f"study exists at {storage_path}; pass --resume to continue it")

    reference = json.loads((ROOT / objective["reference_eval"]).read_text())
    probe_horizons = tuple(int(h) for h in objective["probe_horizons"])
    streams, provenance = fs.load_streams(
        ROOT / "data", tickers=SMOKE["tickers"] if smoke else fs.TICKERS,
        timeframes=SMOKE["timeframes"] if smoke else fs.TIMEFRAMES, repo_root=ROOT)

    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize",
                                sampler=make_sampler(optuna, config),
                                pruner=make_pruner(optuna, config), load_if_exists=True)
    distributions = {key: optuna.distributions.CategoricalDistribution(tuple(values))
                     for key, values in config["search_space"].items()}
    known = {tuple(sorted(t.params.items())) for t in study.trials}
    if not smoke:
        for seed_trial in config.get("seed_trials", []):
            params = seed_trial["params"]
            eval_path, probe_path = ROOT / seed_trial["evaluation"], ROOT / seed_trial["probe"]
            if tuple(sorted(params.items())) in known or not (eval_path.is_file()
                                                             and probe_path.is_file()):
                continue
            result = fs.objective_score(
                json.loads(probe_path.read_text()), json.loads(eval_path.read_text()), reference,
                horizons=probe_horizons, wql_budget=objective["wql_budget"],
                penalty_weight=objective["penalty_weight"])
            study.add_trial(optuna.trial.create_trial(
                params=params, distributions=distributions, value=result["score"],
                user_attrs={"seeded_from": str(eval_path), **{k: v for k, v in result.items()
                                                             if k != "score"}}))

    def train_score(params: dict, seed: int, out_dir: Path, report=None) -> dict:
        """Train one A9 setup, evaluate and probe it on select, return the objective."""
        wait_until_idle()
        try:
            fs.train_forecast_direction(
                streams, parent=ROOT / training["parent"], base_snapshot=BASE_SNAPSHOT,
                out_dir=out_dir, provenance=provenance, arm="a9",
                direction_weight=float(params["direction_weight"]), device=args.device,
                seed=int(seed), epochs=int(training["epochs"]),
                steps_per_epoch=int(training["steps_per_epoch"]),
                batch_windows=int(training["batch_windows"]),
                learning_rate=float(params["learning_rate"]),
                weight_decay=float(training["weight_decay"]), patience=int(training["patience"]),
                select_anchors_per_stream=int(training["select_anchors_per_stream"]),
                repo_root=ROOT, focal_gamma=float(params["focal_gamma"]),
                reg_horizons=tuple(int(h) for h in params["reg_horizons"].split(",")),
                epoch_callback=report)
        finally:
            _release(args.device)
        model = _load_adapter(out_dir / "checkpoint", args.device)
        try:
            evaluation_report = fs.evaluate(
                model, streams, "select", device=args.device,
                anchors_per_stream=int(evaluation["anchors_per_stream"]),
                baseline_train_per_stream=int(evaluation["baseline_train_per_stream"]))
            evaluation_report.pop("_rows", None)
            probe_report = fs.reg_probe_report(
                model, streams, device=args.device,
                train_per_stream=int(evaluation["probe_train_per_stream"]),
                eval_per_stream=int(evaluation["probe_eval_per_stream"]), horizons=probe_horizons)
        finally:
            del model
            _release(args.device)
        _write(out_dir / "eval_select.json", evaluation_report)
        _write(out_dir / "regprobe_select.json", probe_report)
        return fs.objective_score(probe_report, evaluation_report, reference,
                                  horizons=probe_horizons, wql_budget=objective["wql_budget"],
                                  penalty_weight=objective["penalty_weight"])

    if args.confirm:
        return _confirm(args, config, study, sweep_dir, train_score)

    def objective_fn(trial):
        params = {key: trial.suggest_categorical(key, tuple(values))
                  for key, values in config["search_space"].items()}
        name = trial_name(params, int(training["seed"]))
        out_dir = sweep_dir / "trials" / f"t{trial.number:03d}_{name}"
        trial.set_user_attr("out_dir", str(out_dir))

        def report(epoch, metric):
            trial.report(intermediate_value(metric), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch}: select direction "
                                         f"BCE {metric['direction_bce']:.5f}")

        result = train_score(params, int(training["seed"]), out_dir, report)
        for key, value in result.items():
            if key != "score":
                trial.set_user_attr(key, value)
        print(f"[a9-sweep] trial {trial.number} {name} score={result['score']:.4f} "
              f"direction={result['direction_auc']:.4f} wql_ratio={result['wql_ratio']:.4f}",
              flush=True)
        return result["score"]

    finished = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                       optuna.trial.TrialState.PRUNED)]
    remaining = max(0, args.n_trials - len(finished))
    study.optimize(objective_fn, n_trials=remaining, gc_after_trial=True)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    summary = {
        "study": study_name, "storage": storage, "config": str(args.config),
        "best_params": study.best_trial.params if completed else None,
        "best_score": study.best_trial.value if completed else None,
        "trials": [{"number": t.number, "state": t.state.name, "params": t.params,
                    "score": t.value, **t.user_attrs} for t in study.trials],
    }
    _write(sweep_dir / "summary.json", summary)
    print(json.dumps({"best_params": summary["best_params"], "best_score": summary["best_score"]},
                     indent=2), flush=True)
    print(f"  Visualize: optuna-dashboard {storage}", flush=True)
    return summary


def _confirm(args, config: dict, study, sweep_dir: Path, train_score) -> dict:
    """Re-run the top-k trials with extra seeds; winner = best mean across seeds."""
    confirmation = config.get("confirmation", {"top_k": 3, "seeds": [0, 1, 2]})
    trials = [{"number": t.number, "state": t.state.name, "params": t.params, "score": t.value,
               "out_dir": t.user_attrs.get("out_dir")} for t in study.trials]
    runs, details = {}, {}
    for trial in top_k_trials(trials, int(confirmation["top_k"])):
        key = trial_name(trial["params"], 0).rsplit("_seed", 1)[0]
        scores = []
        for seed in confirmation["seeds"]:
            if int(seed) == int(config["training"]["seed"]) and trial["score"] is not None:
                scores.append(float(trial["score"]))      # the sweep trial itself
                continue
            out_dir = sweep_dir / "confirm" / f"{key}_seed{seed}"
            result = train_score(trial["params"], int(seed), out_dir)
            details[f"{key}_seed{seed}"] = result
            scores.append(float(result["score"]))
            print(f"[a9-confirm] {key} seed={seed} score={result['score']:.4f}", flush=True)
        runs[key] = scores
    decision = confirmation_winner(runs)
    decision["details"] = details
    decision["params"] = {trial_name(t["params"], 0).rsplit("_seed", 1)[0]: t["params"]
                          for t in top_k_trials(trials, int(confirmation["top_k"]))}
    _write(sweep_dir / "confirmation.json", decision)
    print(json.dumps({"winner": decision["winner"], "means": decision["means"]}, indent=2),
          flush=True)
    return decision


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--n-trials", type=int, default=None,
                       help="defaults to the config's n_trials")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--smoke", action="store_true",
                       help="2 tiny trials on NQ/ES@3min into a throwaway study")
    value.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    value.add_argument("--confirm", action="store_true",
                       help="re-run the top-k trials with extra seeds and pick the best mean")
    return value


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.smoke:
        args.n_trials = 2
    elif args.n_trials is None:
        args.n_trials = int(load_config(args.config).get("n_trials", 30))
    run(args)


if __name__ == "__main__":
    main()
