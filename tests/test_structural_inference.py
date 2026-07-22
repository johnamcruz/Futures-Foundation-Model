"""Frozen Structural NextLeg task-head inference contract."""
import os

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test - set CHRONOS_TORCH_TESTS=1 (libomp isolation)')


def test_structural_feature_contract_has_embedding_and_all_task_readouts():
    from futures_foundation.finetune.pretext._torch.structural_inference import (
        structural_feature_names)

    names = structural_feature_names()
    assert len(names) == 806
    assert names[0] == 'struct_emb_0'
    assert 'next_uptrend_hh_hl_logit' in names
    assert 'break_bullish_bos_logit' in names
    assert names[-1] == 'break_delay_logbars'


def test_loader_uses_merged_encoder_and_sidecar_task_tensors(tmp_path, monkeypatch):
    import torch
    from futures_foundation.finetune.pretext._torch import structural_inference as inference

    class TinyNet(torch.nn.Module):
        def __init__(self, C, new_channels, horizons, model_id, aux_dim, span_width):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.adapter = torch.nn.Module()
            self.adapter.transformation = torch.nn.Linear(C, new_channels)
            emb = new_channels * 256
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(emb, emb), torch.nn.GELU(),
                torch.nn.Linear(emb, C * len(horizons)))
            self.leg_head = torch.nn.Sequential(
                torch.nn.Linear(emb, emb // 4), torch.nn.GELU(), torch.nn.Linear(emb // 4, 2))
            self.structure_head = torch.nn.Sequential(
                torch.nn.Linear(emb, emb // 4), torch.nn.GELU(), torch.nn.Linear(emb // 4, 8))
            self.excursion_head = torch.nn.Sequential(
                torch.nn.Linear(emb, emb // 4), torch.nn.GELU(), torch.nn.Linear(emb // 4, 2))
            self.break_head = torch.nn.Sequential(
                torch.nn.Linear(emb, emb // 4), torch.nn.GELU(), torch.nn.Linear(emb // 4, 5))
            self.break_delay_head = torch.nn.Sequential(
                torch.nn.Linear(emb, emb // 4), torch.nn.GELU(), torch.nn.Linear(emb // 4, 1))
            self.span_decoder = torch.nn.Sequential(
                torch.nn.Linear(emb, emb // 4), torch.nn.GELU(),
                torch.nn.Linear(emb // 4, C * span_width))

    monkeypatch.setattr(inference, 'StructuralNextLegNet', TinyNet)
    template = TinyNet(5, 3, inference.DEFAULT_HORIZONS, 'unused', 0, 5)
    with torch.no_grad():
        template.encoder.weight.fill_(9.0)  # sidecar encoder must be ignored
        template.break_head[-1].bias.fill_(4.0)
    sidecar = tmp_path / 'task.pt'
    torch.save({'model_state': template.state_dict()}, sidecar)
    merged = tmp_path / 'encoder.pt'
    torch.save({'weight': torch.full((2, 2), 2.0)}, merged)

    loaded = inference.load_structural_forecaster(
        encoder_ckpt=merged, trainer_ckpt=sidecar, model_id='unused')
    np.testing.assert_allclose(loaded.encoder.weight.detach(), 2.0)
    np.testing.assert_allclose(loaded.break_head[-1].bias.detach(), 4.0)


def test_loader_fails_closed_for_partial_task_sidecar(tmp_path, monkeypatch):
    import torch
    from futures_foundation.finetune.pretext._torch import structural_inference as inference

    sidecar = tmp_path / 'bad.pt'
    torch.save({'model_state': {'adapter.transformation.weight': torch.zeros(3, 5)}}, sidecar)
    with pytest.raises(ValueError, match='architecture-defining'):
        inference.load_structural_forecaster(
            encoder_ckpt=tmp_path / 'missing.pt', trainer_ckpt=sidecar, model_id='unused')
