"""Optional MV diagnostic-readout contracts."""
import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("CHRONOS_TORCH_TESTS") != "1",
    reason="torch test - set CHRONOS_TORCH_TESTS=1")


def _payload(torch):
    return {"model_state": {
        "encoder.weight": torch.ones(2, 2),
        "adapter.transformation.weight": torch.ones(3, 5),
        "adapter.transformation.bias": torch.zeros(3),
        "decoder.0.weight": torch.ones(24, 24),
        "decoder.0.bias": torch.zeros(24),
        "decoder.2.weight": torch.ones(20, 24),
        "decoder.2.bias": torch.zeros(20),
        "mv_head.0.weight": torch.ones(6, 24),
        "mv_head.0.bias": torch.zeros(6),
        "mv_head.2.weight": torch.ones(24, 6),
        "mv_head.2.bias": torch.zeros(24),
    }}


def test_export_is_hash_bound_compact_and_diagnostic_only(tmp_path):
    import torch
    from futures_foundation.finetune.pretext._torch.momentum_volatility_inference import (
        export_mv_readout)

    encoder, trainer, output = (
        tmp_path / "encoder.pt", tmp_path / "trainer.pt", tmp_path / "readout.pt")
    torch.save({"weight": torch.ones(2, 2)}, encoder)
    torch.save(_payload(torch), trainer)
    export_mv_readout(output, trainer_ckpt=trainer, encoder_ckpt=encoder)
    result = torch.load(output, map_location="cpu", weights_only=False)
    assert result["usage"] == "optional_diagnostics_only"
    assert result["encoder_sha256"]
    assert not any(key.startswith("encoder.") for key in result["model_state"])


def test_feature_contract_contains_all_four_coupling_states():
    from futures_foundation.finetune.pretext._torch.momentum_volatility_inference import (
        mv_readout_feature_names)

    names = mv_readout_feature_names()
    assert len(names) == 44
    for state in ("chop", "continuation", "reversal", "launch"):
        assert f"mv_coupling_{state}_h25_prob" in names
