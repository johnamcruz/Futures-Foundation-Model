#!/usr/bin/env python3
"""Run or resume existing FFM SSL commands through ML Training Loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.orchestration import load_ssl_workflow, run_ssl_training


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--run-id", required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    state = run_ssl_training(
        load_ssl_workflow(args.config),
        run_id=args.run_id,
    )
    print(json.dumps({
        "run_id": state.run_id,
        "phase": state.phase.value,
        "stage_index": state.stage_index,
        "attempts": state.attempts,
        "message": state.message,
    }, indent=2, sort_keys=True))
    return 0 if state.phase.value == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
