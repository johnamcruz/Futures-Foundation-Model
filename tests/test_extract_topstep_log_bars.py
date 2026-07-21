"""Tests for FFM-owned Topstep log replay extraction."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "scripts" / "extract_topstep_log_bars.py"
SPEC = importlib.util.spec_from_file_location("extract_topstep_log_bars", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_parse_bars_deduplicates_identical_logger_lines():
    line = "Bar: 2026-07-20 12:00 | O:10 H:12 L:9 C:11 V:25"
    frame = module.parse_bars(f"{line}\nINFO:root:{line}\n")
    assert len(frame) == 1
    assert frame.iloc[0]["close"] == 11


def test_parse_bars_rejects_conflicting_duplicates():
    text = "\n".join([
        "Bar: 2026-07-20 12:00 | O:10 H:12 L:9 C:11 V:25",
        "Bar: 2026-07-20 12:00 | O:10 H:12 L:9 C:10 V:25",
    ])
    with pytest.raises(ValueError, match="conflicting duplicate"):
        module.parse_bars(text)
