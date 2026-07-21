#!/usr/bin/env python3
"""Run the checkpoint capability atlas used by the Mantis pretraining pipeline.

This stable public entrypoint keeps orchestration under ``scripts/`` while the
strategy-specific probe definitions remain in the private ``colabs`` repository.
The clean SSL master sets these variables automatically:

    CKPT_PATH       exact merged encoder checkpoint
    CKPT_NAME       display/ledger name
    CKPT_SHA256     lineage identity
    EMB_CACHE       checkpoint-specific frozen embedding cache
    ATLAS_OUT       machine-readable capability JSON
    ATLAS_BATCH     embedding batch size
    DEVICE          cuda, mps, or cpu

It can also be called directly with the same environment contract.
"""
from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "colabs" / "mantis" / "probe_atlas.py"


def main() -> None:
    if not IMPLEMENTATION.is_file():
        raise FileNotFoundError(
            "private Probe Atlas implementation is unavailable; initialize the colabs submodule: "
            f"{IMPLEMENTATION}")
    runpy.run_path(str(IMPLEMENTATION), run_name="__main__")


if __name__ == "__main__":
    main()
