"""Contracts for the A9 TPE sweep script (no training, no model download)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _sweep():
    path = ROOT / "scripts/chronos/chronos2_a9_optuna_sweep.py"
    spec = importlib.util.spec_from_file_location("chronos2_a9_optuna_sweep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_config_is_valid_and_searches_the_four_knobs():
    sweep = _sweep()
    config = sweep.load_config(sweep.DEFAULT_CONFIG)
    assert set(config["search_space"]) == {"direction_weight", "focal_gamma", "reg_horizons",
                                           "learning_rate"}
    assert 1e-05 in config["search_space"]["learning_rate"]
    assert config["study"]["sampler"]["type"] == "tpe"
    assert config["study"]["pruner"]["type"] == "percentile"
    assert config["study"]["pruner"]["percentile"] == 25.0
    assert config["study"]["pruner"]["n_startup_trials"] >= 5
    assert config["confirmation"]["top_k"] == 3 and len(config["confirmation"]["seeds"]) == 3
    assert config["n_trials"] == 30
    seeded = config["seed_trials"][0]["params"]
    assert set(seeded) == set(config["search_space"])
    for key, value in seeded.items():
        assert value in config["search_space"][key]


def test_config_validation_rejects_bad_pruner_and_horizons(tmp_path):
    sweep = _sweep()
    config = json.loads(sweep.DEFAULT_CONFIG.read_text())
    config["study"]["pruner"]["type"] = "random"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        sweep.load_config(path)
    config["study"]["pruner"]["type"] = "median"
    config["search_space"]["reg_horizons"] = ["5,99"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        sweep.load_config(path)


def test_pruner_factory_follows_config():
    optuna = pytest.importorskip("optuna")
    sweep = _sweep()
    config = sweep.load_config(sweep.DEFAULT_CONFIG)
    assert isinstance(sweep.make_pruner(optuna, config), optuna.pruners.PercentilePruner)
    config["study"]["pruner"] = {"type": "median"}
    assert isinstance(sweep.make_pruner(optuna, config), optuna.pruners.MedianPruner)
    config["study"]["pruner"] = {"type": "none"}
    assert isinstance(sweep.make_pruner(optuna, config), optuna.pruners.NopPruner)
    config["study"]["pruner"] = {"type": "hyperband", "min_resource": 3}
    assert isinstance(sweep.make_pruner(optuna, config), optuna.pruners.HyperbandPruner)
    assert isinstance(sweep.make_sampler(optuna, config), optuna.samplers.TPESampler)


def test_intermediate_value_rewards_lower_direction_bce():
    sweep = _sweep()
    better = sweep.intermediate_value({"direction_bce": 0.690})
    worse = sweep.intermediate_value({"direction_bce": 0.695})
    assert better > worse


def test_trial_name_encodes_every_knob():
    sweep = _sweep()
    name = sweep.trial_name({"direction_weight": 10.0, "focal_gamma": 2.0,
                             "reg_horizons": "5,10", "learning_rate": 2e-05}, seed=0)
    assert name == "a9_lam10_focal2_h5-10_lr2e-05_seed0"


def test_top_k_picks_the_best_completed_trials():
    sweep = _sweep()
    trials = [{"number": 0, "state": "COMPLETE", "score": 0.52, "params": {"a": 1}},
              {"number": 1, "state": "PRUNED", "score": None, "params": {"a": 2}},
              {"number": 2, "state": "COMPLETE", "score": 0.55, "params": {"a": 3}},
              {"number": 3, "state": "COMPLETE", "score": 0.53, "params": {"a": 4}},
              {"number": 4, "state": "COMPLETE", "score": 0.51, "params": {"a": 5}}]
    assert [t["number"] for t in sweep.top_k_trials(trials, 3)] == [2, 3, 0]


def test_confirmation_winner_is_best_mean_across_seeds():
    sweep = _sweep()
    runs = {"X": [0.56, 0.50, 0.51],      # lucky single seed
            "Y": [0.54, 0.54, 0.53]}      # consistently good
    decision = sweep.confirmation_winner(runs)
    assert decision["winner"] == "Y"
    assert decision["means"]["Y"] == pytest.approx(0.53667, abs=1e-4)
    assert decision["stdevs"]["X"] > decision["stdevs"]["Y"]


def test_confirm_flag_parses():
    args = _sweep().parser().parse_args(["--confirm"])
    assert args.confirm


def test_smoke_runs_use_their_own_folder():
    source = (ROOT / "scripts/chronos/chronos2_a9_optuna_sweep.py").read_text()
    assert 'storage_path.parent / "smoke" / "study.db"' in source
