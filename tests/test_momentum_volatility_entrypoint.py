import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mantis_ssl_momentum_volatility.py"


def test_entrypoint_is_public_strategy_agnostic_and_calls_canonical_task():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    assert 'pretext="momentum_volatility"' in source
    assert 'holdout_start=HOLDOUT_START' in source
    assert 'backbone_ckpt=str(parent)' in source
    assert "shuffle,random" in source
    assert "--resume" in source
    assert "--lora-r" in source
    assert "uses_strategy_labels" in source
    assert "pivot" not in source.lower()
    assert "ffm-strategies" not in source.lower()
    assert all(
        not (isinstance(node, ast.Import) and
             any(alias.name == "torch" for alias in node.names))
        for node in tree.body)


def test_entrypoint_defaults_to_race_v2_parent_and_full_stream_matrix():
    source = SCRIPT.read_text()
    assert "mantis_ssl_nextleg_race_v2.pt" in source
    assert "mantis_ssl_mv.pt" in source
    assert 'TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")' in source
    assert 'TIMEFRAMES = ("1min", "3min", "5min", "15min")' in source
    assert 'HOLDOUT_START = "2026-01-01"' in source


def test_entrypoint_seals_data_and_runs_probe_atlas():
    source = SCRIPT.read_text()
    assert "seal_continuous_streams" in source
    assert "export_mv_readout" in source
    assert "optional diagnostics only" in source
    assert "probe_atlas.py" in source
    assert "checkpoint_sha256" in source
    assert ".data_provenance.json" in source
