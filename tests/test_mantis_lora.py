"""LoRA adapter and plain-checkpoint compatibility tests (Torch-isolated)."""
import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation)')


def _setup():
    import copy
    import torch
    from futures_foundation.finetune.pretext._torch.lora import (
        LoRALinear, inject_mantis_lora, load_plain_state_dict, merged_state_dict)

    class Fn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_qkv = torch.nn.Linear(4, 12)
            self.to_out = torch.nn.Sequential(torch.nn.Linear(12, 4))

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fn = Fn()

    class ToyMantis(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.ModuleList([
                torch.nn.ModuleList([Block()]) for _ in range(6)])
            self.unrelated = torch.nn.Linear(4, 4)

        def forward(self, x):
            for layer in self.transformer:
                block = layer[0].fn
                x = block.to_out(block.to_qkv(x))
            return self.unrelated(x)

    return (copy, torch, LoRALinear, inject_mantis_lora,
            load_plain_state_dict, merged_state_dict, ToyMantis)


def test_mantis_lora_injection_starts_at_exact_base_parity():
    copy, torch, LoRALinear, inject, _, _, Toy = _setup()
    model = Toy().eval()
    original = copy.deepcopy(model).eval()
    x = torch.randn(3, 4)
    expected = original(x)
    stats = inject(model, rank=2, alpha=4)
    assert stats['modules'] == 12 and stats['trainable'] > 0
    assert torch.equal(model(x), expected)
    assert all(not p.requires_grad for p in model.unrelated.parameters())
    assert sum(isinstance(m, LoRALinear) for m in model.modules()) == 12


def test_merged_lora_checkpoint_loads_into_plain_model_with_output_parity():
    copy, torch, LoRALinear, inject, _, merge, Toy = _setup()
    model = Toy().eval()
    plain = copy.deepcopy(model).eval()
    inject(model, rank=2, alpha=4)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.weight.normal_(0, 0.05)
    x = torch.randn(3, 4)
    adapted = model(x)
    state = merge(model)
    assert not any('lora_' in key or '.base.' in key for key in state)
    plain.load_state_dict(state)
    assert torch.allclose(plain(x), adapted, atol=1e-6)


def test_plain_checkpoint_can_resume_in_fresh_lora_wrappers():
    _, torch, LoRALinear, inject, load_plain, merge, Toy = _setup()
    source = Toy().eval()
    inject(source, rank=2, alpha=4)
    with torch.no_grad():
        for module in source.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.weight.normal_(0, 0.05)
    state = merge(source)
    resumed = Toy().eval()
    inject(resumed, rank=2, alpha=4)
    load_plain(resumed, state)
    x = torch.randn(3, 4)
    plain = Toy().eval(); plain.load_state_dict(state)
    assert torch.allclose(resumed(x), plain(x), atol=1e-6)
