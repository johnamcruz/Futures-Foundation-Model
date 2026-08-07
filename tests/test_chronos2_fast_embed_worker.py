from types import SimpleNamespace

import numpy as np
import pytest


def test_fast_reg_embedder_matches_official_pool_with_one_batch_transfer(
        monkeypatch):
    torch = pytest.importorskip("torch")
    import chronos
    from chronos import Chronos2Pipeline
    from futures_foundation.finetune.classifiers.chronos2._embed_worker_fast import (
        embed_window_chunks,
    )

    calls = {"cpu": 0, "encode": 0}

    class FakeModel:
        device = torch.device("cpu")
        chronos_config = SimpleNamespace(
            context_length=8,
            output_patch_size=2,
            max_output_patches=4,
        )

        def encode(self, *, context, group_ids):
            calls["encode"] += 1
            encoded = torch.arange(
                context.shape[0] * 4 * 3,
                dtype=torch.float32,
            ).reshape(context.shape[0], 4, 3)
            locs = torch.zeros(context.shape[0])
            scales = torch.ones(context.shape[0])
            return (encoded,), (locs, scales), None

    class FakePipeline:
        model_context_length = 8
        model_output_patch_size = 2
        embed = Chronos2Pipeline.embed

        def __init__(self):
            self.model = FakeModel()

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            assert source == "fake/model"
            assert kwargs == {"device_map": "cpu"}
            return cls()

    windows = np.ones((3, 5, 8), np.float32)
    reference_values, _ = FakePipeline().embed(
        windows,
        batch_size=15,
        context_length=8,
    )
    reference = np.stack([
        value.detach().float().numpy()[:, -2, :].reshape(-1)
        for value in reference_values
    ])

    original_cpu = torch.Tensor.cpu

    def counted_cpu(tensor, *args, **kwargs):
        calls["cpu"] += 1
        return original_cpu(tensor, *args, **kwargs)

    calls.update(cpu=0, encode=0)
    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    monkeypatch.setattr(chronos, "Chronos2Pipeline", FakePipeline)
    output = next(embed_window_chunks(
        (windows,),
        checkpoint="fake/model",
        device="cpu",
        batch=15,
        pool="reg",
        context_length=8,
    ))

    np.testing.assert_array_equal(output, reference)
    assert calls == {"cpu": 1, "encode": 1}
