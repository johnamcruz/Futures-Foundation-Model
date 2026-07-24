"""Static and CLI contracts for the public context-decoder trainer."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "fit_market_context.py"
SPEC = importlib.util.spec_from_file_location("fit_market_context", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_market_context_entrypoint_uses_promoted_mv3_and_public_atlas():
    args = MODULE.build_parser().parse_args([])

    assert args.checkpoint == ROOT / "checkpoints" / "mantis_ssl_mv_v3.pt"
    assert args.output == (
        ROOT / "checkpoints" / "mantis_ssl_mv_v3_context.npz")
    source = SCRIPT.read_text()
    assert "import probe_atlas" in source
    assert "fit_market_context_decoder" in source
    assert "ffm-strategies" not in source
    assert "pivot" not in source.lower()
    assert "strategy_inputs" in source


def test_market_context_entrypoint_keeps_2025_as_temporal_evaluation():
    args = MODULE.build_parser().parse_args([])

    assert args.fit_end == "2023-01-01"
    assert args.calibration_start == "2023-01-01"
    assert args.calibration_end == "2024-01-01"
    assert args.evaluation_start == "2025-01-01"
    assert args.evaluation_end == "2026-01-01"
