"""Compact Momentum-Volatility readout and frozen inference contracts."""
import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("CHRONOS_TORCH_TESTS") != "1",
    reason="torch test - set CHRONOS_TORCH_TESTS=1")


def _trainer_payload(torch, channels=5, adapted=3, hidden=8, horizons=4):
    embedding = adapted * hidden
    return {
        "model_state": {
            "encoder.weight": torch.ones(2, 2),
            "adapter.transformation.weight": torch.ones(adapted, channels),
            "adapter.transformation.bias": torch.zeros(adapted),
            "decoder.0.weight": torch.ones(embedding, embedding),
            "decoder.0.bias": torch.zeros(embedding),
            "decoder.2.weight": torch.ones(channels * horizons, embedding),
            "decoder.2.bias": torch.zeros(channels * horizons),
            "mv_head.0.weight": torch.ones(embedding // 4, embedding),
            "mv_head.0.bias": torch.zeros(embedding // 4),
            "mv_head.2.weight": torch.ones(horizons * 6, embedding // 4),
            "mv_head.2.bias": torch.zeros(horizons * 6),
        }
    }


def test_export_is_compact_hash_bound_and_excludes_encoder(tmp_path):
    import torch
    from futures_foundation.finetune.pretext._torch.momentum_volatility_inference import (
        READOUT_SCHEMA, export_mv_readout)

    encoder = tmp_path / "encoder.pt"
    trainer = tmp_path / "trainer.pt"
    output = tmp_path / "readout.pt"
    torch.save({"weight": torch.ones(2, 2)}, encoder)
    torch.save(_trainer_payload(torch), trainer)
    export_mv_readout(output, trainer_ckpt=trainer, encoder_ckpt=encoder)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["schema"] == READOUT_SCHEMA
    assert payload["encoder_sha256"]
    assert payload["horizons"] == [5, 10, 20, 25]
    assert not any(key.startswith("encoder.") for key in payload["model_state"])
    assert "mv_head.2.weight" in payload["model_state"]


def test_export_rejects_wrong_horizon_shape(tmp_path):
    import torch
    from futures_foundation.finetune.pretext._torch.momentum_volatility_inference import (
        export_mv_readout)

    encoder = tmp_path / "encoder.pt"
    trainer = tmp_path / "trainer.pt"
    torch.save({}, encoder)
    payload = _trainer_payload(torch)
    payload["model_state"]["mv_head.2.weight"] = torch.ones(18, 6)
    torch.save(payload, trainer)
    with pytest.raises(ValueError, match="metadata"):
        export_mv_readout(
            tmp_path / "readout.pt", trainer_ckpt=trainer, encoder_ckpt=encoder)


def test_feature_names_are_stable_and_include_all_coupling_probabilities():
    from futures_foundation.finetune.pretext._torch.momentum_volatility_inference import (
        mv_feature_names, mv_readout_feature_names)

    readout = mv_readout_feature_names()
    assert len(readout) == 44
    assert "mv_momentum_h20" in readout
    assert "mv_log_range_ratio_h25" in readout
    assert "mv_coupling_chop_h5_prob" in readout
    assert "mv_coupling_continuation_h10_prob" in readout
    assert "mv_coupling_reversal_h20_prob" in readout
    assert "mv_coupling_launch_h25_prob" in readout
    assert len(mv_feature_names()) == 768 + len(readout)


def test_readout_hash_mismatch_fails_before_model_load(tmp_path):
    import torch
    from futures_foundation.finetune.pretext._torch.momentum_volatility_inference import (
        export_mv_readout, load_mv_forecaster)

    encoder = tmp_path / "encoder.pt"
    wrong = tmp_path / "wrong.pt"
    trainer = tmp_path / "trainer.pt"
    readout = tmp_path / "readout.pt"
    torch.save({"weight": torch.ones(2, 2)}, encoder)
    torch.save({"weight": torch.zeros(2, 2)}, wrong)
    torch.save(_trainer_payload(torch), trainer)
    export_mv_readout(readout, trainer_ckpt=trainer, encoder_ckpt=encoder)
    with pytest.raises(ValueError, match="different encoder"):
        load_mv_forecaster(encoder_ckpt=wrong, readout_ckpt=readout)
