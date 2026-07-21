"""Fail-closed provenance for FFM continuous-contract training bars."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_continuous_streams(
        data_dir: Path, streams: Iterable[tuple[str, str]],
        *, repo_root: Path | None = None) -> dict:
    """Validate manifests and return hashes for exact model inputs."""
    sealed = {}
    for ticker, timeframe in streams:
        bars = Path(data_dir) / f"{ticker}_{timeframe}.csv"
        manifest_path = bars.with_suffix(bars.suffix + ".manifest.json")
        if not bars.is_file() or not manifest_path.is_file():
            raise RuntimeError(
                f"roll-safe bars/manifest missing for {ticker}@{timeframe}: "
                f"{bars}, {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        bars_sha = sha256(bars)
        expected = {
            "ticker": ticker,
            "timeframe": timeframe,
            "schema": "ffm_continuous_contract_v1",
            "selection": "cme_session_total_volume",
            "back_adjusted": True,
            "output_sha256": bars_sha,
        }
        mismatches = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected.items() if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"invalid manifest for {ticker}@{timeframe}: {mismatches}")
        resolved = bars.resolve()
        if repo_root is not None:
            try:
                logical_path = str(resolved.relative_to(Path(repo_root).resolve()))
            except ValueError:
                logical_path = str(resolved)
        else:
            logical_path = str(resolved)
        sealed[f"{ticker}@{timeframe}"] = {
            "path": logical_path,
            "sha256": bars_sha,
            "manifest_sha256": sha256(manifest_path),
            "source_sha256": manifest.get("source_sha256"),
            "source_sha256s": manifest.get("source_sha256s"),
            "rows": manifest.get("rows"),
            "start": manifest.get("start"),
            "end": manifest.get("end"),
        }
    return {
        "schema": "ffm_training_data_provenance_v1",
        "continuous_contract": "session-dominant-back-adjusted",
        "streams": sealed,
    }
