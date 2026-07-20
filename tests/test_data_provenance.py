"""Training-data provenance must fail closed on stale continuous bars."""
from __future__ import annotations

import hashlib
import json

import pytest

from futures_foundation.data_provenance import seal_continuous_streams


def _write_stream(tmp_path, *, back_adjusted=True, output_hash=None):
    bars = tmp_path / "NQ_3min.csv"
    bars.write_text("datetime,open,high,low,close,volume\n"
                    "2026-01-01,1,2,0,1,10\n")
    digest = hashlib.sha256(bars.read_bytes()).hexdigest()
    manifest = {
        "ticker": "NQ", "timeframe": "3min",
        "schema": "ffm_continuous_contract_v1",
        "selection": "cme_session_total_volume",
        "back_adjusted": back_adjusted,
        "output_sha256": output_hash or digest,
        "source_sha256": "raw-hash", "rows": 1,
        "start": "2026-01-01", "end": "2026-01-01",
    }
    bars.with_suffix(".csv.manifest.json").write_text(json.dumps(manifest))
    return bars


def test_seals_exact_roll_safe_hash(tmp_path):
    bars = _write_stream(tmp_path)
    result = seal_continuous_streams(
        tmp_path, [("NQ", "3min")], repo_root=tmp_path)
    assert result["streams"]["NQ@3min"]["path"] == bars.name
    assert result["streams"]["NQ@3min"]["source_sha256"] == "raw-hash"


@pytest.mark.parametrize("kwargs", [
    {"back_adjusted": False},
    {"output_hash": "stale"},
])
def test_rejects_unsealed_or_modified_stream(tmp_path, kwargs):
    _write_stream(tmp_path, **kwargs)
    with pytest.raises(RuntimeError, match="invalid manifest"):
        seal_continuous_streams(tmp_path, [("NQ", "3min")])
