"""Public Probe Atlas and lifecycle infrastructure must stay strategy-independent."""
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_atlas_is_a_standalone_public_implementation():
    source = (ROOT / "scripts" / "probe_atlas.py").read_text()
    assert 'ROOT / "colabs"' not in source
    assert "import pivot_trend_mantis" not in source
    assert "def _load_pool" in source
    assert "def main" in source
    assert "FIXED_TARGETS" not in source
    assert "STOP_BUFFER" not in source
    assert "pred_reach_" not in source


def test_clean_pipeline_has_no_colabs_runtime_dependency():
    source = (ROOT / "scripts" / "mantis" / "mantis_ssl_clean_pipeline.py").read_text()
    assert 'ROOT / "colabs"' not in source
    assert 'ROOT / "scripts" / "generate_trend_labels.py"' in source


def test_atlas_sampling_is_deterministic_and_preserves_time_coverage():
    atlas = _load("public_probe_atlas", ROOT / "scripts" / "probe_atlas.py")
    rows = np.arange(100)
    selected = atlas._even_sample(rows, 10)
    assert np.array_equal(selected, atlas._even_sample(rows, 10))
    assert len(selected) == 10
    assert selected[0] == 0 and selected[-1] == 99


def test_atlas_mv_targets_use_the_exact_future_window_and_causal_scale():
    atlas = _load("public_probe_atlas_mv", ROOT / "scripts" / "probe_atlas.py")
    rows = 100
    close = np.arange(rows, dtype=float)
    width = np.full(rows, 2.0)
    high, low = close + width / 2, close - width / 2

    strength, expansion, state = atlas._momentum_volatility_fields(
        high, low, close)
    assert strength[63] == 1.0
    assert expansion[63] == 1.0
    assert state[63] == 1

    changed_high, changed_low = high.copy(), low.copy()
    changed_high[84:] += 1000
    changed_low[84:] -= 1000
    changed = atlas._momentum_volatility_fields(
        changed_high, changed_low, close)
    assert changed[1][63] == expansion[63]
    assert changed[2][63] == state[63]


def test_atlas_multihorizon_targets_use_requested_completed_future_bars():
    atlas = _load("public_probe_atlas_horizons", ROOT / "scripts" / "probe_atlas.py")
    rows = 100
    close = np.arange(rows, dtype=float)
    width = np.full(rows, 2.0)
    high, low = close + width / 2, close - width / 2

    strength, expansion, state = atlas._momentum_volatility_fields(
        high, low, close, horizon=5)

    assert strength[63] == 1.0
    assert expansion[63] == 1.0
    assert state[63] == 1
    changed_high, changed_low = high.copy(), low.copy()
    changed_high[69:] += 1000
    changed_low[69:] -= 1000
    changed = atlas._momentum_volatility_fields(
        changed_high, changed_low, close, horizon=5)
    assert changed[1][63] == expansion[63]


def test_atlas_input_controls_are_deterministic_and_input_only(monkeypatch):
    monkeypatch.setenv("ATLAS_CONTROL", "shuffle")
    atlas = _load("public_probe_atlas_control", ROOT / "scripts" / "probe_atlas.py")
    window = np.arange(40, dtype=np.float32).reshape(5, 8)
    key = ("NQ", "3min", 42)
    first = atlas._controlled_window(window, key)
    second = atlas._controlled_window(window, key)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, window)
    for channel in range(5):
        np.testing.assert_array_equal(
            np.sort(first[channel]), np.sort(window[channel]))
    # One shared permutation preserves OHLCV alignment across each shuffled bar.
    np.testing.assert_array_equal(first[1] - first[0], np.full(8, 8))


def test_chronos2_probe_atlas_entrypoint_ports_public_atlas_contract(tmp_path):
    launcher = _load(
        "chronos2_probe_atlas",
        ROOT / "scripts" / "chronos" / "chronos2_probe_atlas.py",
    )
    adapter = tmp_path / "checkpoint"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    assert launcher._checkpoint_identity(str(adapter)) == launcher._tree_sha256(adapter)
    defaults = launcher.parser().parse_args([])
    assert defaults.window == 256
    assert defaults.horizons == "5,10,20,50"
    assert defaults.batch_series == 5
    assert defaults.control == "real"
    source = Path(launcher.__file__).read_text()
    assert 'ATLAS_BACKBONE": "chronos2"' in source
    assert 'ROOT / "scripts" / "probe_atlas.py"' in source


def test_atlas_pool_identity_seals_model_data_targets_and_control(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_BACKBONE", "chronos2")
    monkeypatch.setenv("ATLAS_CONTROL", "random")
    monkeypatch.setenv("ATLAS_WINDOW", "256")
    atlas = _load("public_probe_atlas_identity", ROOT / "scripts" / "probe_atlas.py")
    corpus = tmp_path / "labels.npz"
    np.savez(corpus, ticker=np.asarray(["NQ"]))
    monkeypatch.setattr(atlas, "CORPUS", corpus)
    bars = {
        ("NQ", "3min"): {"source_sha256": "source-hash"},
    }

    assert atlas.BACKBONE == "chronos2"
    assert atlas.CONTROL == "random"
    assert atlas.WINDOW == 256
    assert len(atlas._source_sha256(bars)) == 64
    assert len(atlas._file_sha256(corpus)) == 64


def test_public_lifecycle_marks_breaks_without_strategy_imports():
    lifecycle = _load("public_trend_lifecycle", ROOT / "scripts" / "trend_lifecycle.py")
    high = np.array([10.0, 0.0, 12.0, 0.0, 11.0])
    low = np.array([0.0, 5.0, 0.0, 6.0, 0.0])
    pivots = [
        {"origin": 0, "confirm": 0, "direction": -1},
        {"origin": 1, "confirm": 1, "direction": 1},
        {"origin": 2, "confirm": 2, "direction": -1},
        {"origin": 3, "confirm": 3, "direction": 1},
        {"origin": 4, "confirm": 4, "direction": -1},
    ]
    labels = lifecycle.label_trend_lifecycle(high, low, pivots)
    assert labels[2]["role_kind"] == "start"
    assert labels[3]["kind"] == "end"
    assert labels[4]["role_kind"] == "start"
