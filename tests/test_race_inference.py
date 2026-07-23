"""Compact NextLeg Race readout identity and inference contracts."""
import os

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("CHRONOS_TORCH_TESTS") != "1",
    reason="torch test — set CHRONOS_TORCH_TESTS=1")


def _tiny_net(torch, *, channels=5, adapted=3, horizons=(2, 7), levels=(.5, 1.5)):
    class TinyNet(torch.nn.Module):
        def __init__(self, C, new_channels, horizons, model_id, aux_dim, race_levels):
            super().__init__()
            self.C, self.new_c = C, new_channels
            self.horizons, self.race_levels = tuple(horizons), tuple(race_levels)
            self.nH = len(self.horizons)
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.adapter = torch.nn.Module()
            self.adapter.transformation = torch.nn.Linear(C, new_channels)
            embedding = 12
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(embedding, 8), torch.nn.GELU(),
                torch.nn.Linear(8, C * len(horizons)))
            self.leg_head = torch.nn.Sequential(
                torch.nn.Linear(embedding, 8), torch.nn.GELU(), torch.nn.Linear(8, 2))
            self.race_head = torch.nn.Sequential(
                torch.nn.Linear(embedding, 8), torch.nn.GELU(),
                torch.nn.Linear(8, 3 * len(race_levels)))

    return TinyNet(channels, adapted, horizons, "unused", 0, levels), TinyNet


def test_export_and_loader_bind_custom_metadata_to_exact_encoder(tmp_path, monkeypatch):
    import torch
    from futures_foundation.finetune.pretext._torch import race_inference as inference

    template, tiny_class = _tiny_net(torch)
    monkeypatch.setattr(inference, "NextLegRaceNet", tiny_class)
    with torch.no_grad():
        template.encoder.weight.fill_(9.0)  # sidecar encoder must never be deployed
        template.race_head[-1].bias.fill_(4.0)
    trainer = tmp_path / "race.trainer.pt"
    torch.save({"model_state": template.state_dict()}, trainer)
    encoder = tmp_path / "race.pt"
    torch.save({"weight": torch.full((2, 2), 2.0)}, encoder)
    readout = tmp_path / "race.readout.pt"

    inference.export_race_readout(
        readout, trainer_ckpt=trainer, encoder_ckpt=encoder,
        horizons=(2, 7), race_levels=(.5, 1.5))
    payload = torch.load(readout, map_location="cpu", weights_only=False)
    assert payload["horizons"] == [2, 7]
    assert payload["race_levels"] == [.5, 1.5]
    assert all(not key.startswith("encoder.") for key in payload["model_state"])

    loaded = inference.load_race_forecaster(
        encoder_ckpt=encoder, readout_ckpt=readout, model_id="unused")
    np.testing.assert_allclose(loaded.encoder.weight.detach(), 2.0)
    np.testing.assert_allclose(loaded.race_head[-1].bias.detach(), 4.0)

    torch.save({"weight": torch.full((2, 2), 3.0)}, encoder)
    with pytest.raises(ValueError, match="different encoder"):
        inference.load_race_forecaster(
            encoder_ckpt=encoder, readout_ckpt=readout, model_id="unused")


def test_export_rejects_metadata_that_disagrees_with_tensor_shapes(tmp_path):
    import torch
    from futures_foundation.finetune.pretext._torch import race_inference as inference

    template, _ = _tiny_net(torch)
    trainer = tmp_path / "race.trainer.pt"
    encoder = tmp_path / "race.pt"
    torch.save({"model_state": template.state_dict()}, trainer)
    torch.save(template.encoder.state_dict(), encoder)
    with pytest.raises(ValueError, match="metadata does not match"):
        inference.export_race_readout(
            tmp_path / "bad.pt", trainer_ckpt=trainer, encoder_ckpt=encoder,
            horizons=(2, 7, 12), race_levels=(.5, 1.5))


def test_feature_encoder_runs_encoder_once_and_preserves_monotone_probabilities():
    import torch
    from futures_foundation.finetune.pretext._torch.race_inference import (
        NextLegRaceFeatureEncoder)

    class FakeNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def embed(self, context):
            self.calls += 1
            return torch.ones(len(context), 3)

        def readouts(self, embedding):
            batch = len(embedding)
            return (
                torch.ones(batch, 1, 2), torch.ones(batch, 2),
                torch.tensor([[2.0, 1.0]]).repeat(batch, 1),
                torch.tensor([[.2, .8]]).repeat(batch, 1),
                torch.tensor([[.5, 1.0]]).repeat(batch, 1),
            )

    net = FakeNet()
    output = NextLegRaceFeatureEncoder(net)(torch.randn(4, 5, 64))
    assert net.calls == 1
    assert output.shape == (4, 13)
    reach = output[:, 7:9]
    assert torch.all(torch.diff(reach, dim=1) <= 0)
