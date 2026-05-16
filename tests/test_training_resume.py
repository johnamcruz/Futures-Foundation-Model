"""
Unit tests for the shared training-resume primitives.

These back the resume mechanics reused by BOTH frameworks:
  - pretrain (futures_foundation/pretrain/trainer.py)  — hash key "cfg_hash"
  - finetune (futures_foundation/finetune/trainer.py)  — hash key "config_hash"

So both consumers' contracts are exercised here (the configurable hash_key is
exactly what lets finetune reuse load_resume without changing its on-disk
checkpoint schema).
"""
import torch

from futures_foundation.training_resume import (
    resume_hash, atomic_save_resume, load_resume, clear_resume,
)


# ── resume_hash ───────────────────────────────────────────────────────────────

def _h(cfg=None, arch=None, data=None):
    return resume_hash(
        cfg or {'lr': 1e-4, 'epochs': 50},
        arch or {'h': 256, 'l': 6},
        data if data is not None else [('ES', 100), ('GC', 90)],
    )


def test_resume_hash_deterministic():
    assert _h() == _h()


def test_resume_hash_key_order_invariant():
    a = resume_hash({'a': 1, 'b': 2}, {'x': 1}, [1, 2])
    b = resume_hash({'b': 2, 'a': 1}, {'x': 1}, [1, 2])
    assert a == b


def test_resume_hash_data_sensitive():
    """The data term is the CRT stale-cache fix — a data change MUST bust it."""
    assert _h(data=[('ES', 100)]) != _h(data=[('ES', 101)])


def test_resume_hash_config_and_arch_sensitive():
    assert _h(cfg={'lr': 1e-4}) != _h(cfg={'lr': 5e-5})
    assert _h(arch={'h': 256}) != _h(arch={'h': 512})


def test_resume_hash_len_and_type():
    h = _h()
    assert isinstance(h, str) and len(h) == 10


# ── atomic_save_resume / load_resume roundtrip ───────────────────────────────

def test_save_load_roundtrip_pretrain_key(tmp_path):
    p = tmp_path / 'resume_abc.pt'
    blob = {'cfg_hash': 'abc', 'epoch': 7,
            'model': {'w': torch.tensor([1.0, 2.0])}}
    atomic_save_resume(p, blob)
    out = load_resume(p, 'abc')                       # default hash_key='cfg_hash'
    assert out is not None and out['epoch'] == 7
    assert torch.equal(out['model']['w'], torch.tensor([1.0, 2.0]))


def test_save_load_roundtrip_finetune_key(tmp_path):
    """Finetune stores its hash under 'config_hash' — reuse must work without
    finetune changing its on-disk schema."""
    p = tmp_path / 'F1_xyz_done.pt'
    atomic_save_resume(p, {'config_hash': 'xyz', 'next_fold_state': {'a': 1}})
    assert load_resume(p, 'xyz', hash_key='config_hash') is not None
    # Wrong key (default) must NOT match a finetune blob → None (start fresh).
    assert load_resume(p, 'xyz') is None


def test_load_missing_returns_none(tmp_path):
    assert load_resume(tmp_path / 'nope.pt', 'h') is None


def test_load_hash_mismatch_returns_none(tmp_path):
    p = tmp_path / 'r.pt'
    atomic_save_resume(p, {'cfg_hash': 'OLD', 'epoch': 3})
    assert load_resume(p, 'NEW') is None              # config/data changed → fresh


def test_atomic_save_leaves_no_tmp(tmp_path):
    p = tmp_path / 'r.pt'
    atomic_save_resume(p, {'cfg_hash': 'h', 'epoch': 1})
    assert p.exists()
    assert not (tmp_path / 'r.pt.tmp').exists()       # tmp renamed in, not left


def test_atomic_save_overwrite_is_clean(tmp_path):
    p = tmp_path / 'r.pt'
    atomic_save_resume(p, {'cfg_hash': 'h', 'epoch': 1})
    atomic_save_resume(p, {'cfg_hash': 'h', 'epoch': 2})
    assert load_resume(p, 'h')['epoch'] == 2


def test_save_creates_parent_dirs(tmp_path):
    p = tmp_path / 'deep' / 'nested' / 'r.pt'
    atomic_save_resume(p, {'cfg_hash': 'h', 'epoch': 1})
    assert p.exists()


# ── clear_resume ─────────────────────────────────────────────────────────────

def test_clear_resume_removes_file(tmp_path):
    p = tmp_path / 'r.pt'
    atomic_save_resume(p, {'cfg_hash': 'h', 'epoch': 1})
    clear_resume(p)
    assert not p.exists()


def test_clear_resume_idempotent(tmp_path):
    clear_resume(tmp_path / 'never_existed.pt')        # must not raise
