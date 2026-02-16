"""Unit tests — run: python tests/test_model.py"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, numpy as np, pandas as pd
from futures_foundation import (
    FFMConfig, FFMBackbone, FFMForPretraining, FFMForClassification,
    derive_features, generate_all_labels, get_model_feature_columns, FFMDataset,
)

def make_dummy_ohlcv(n=500, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    close = 5000 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "datetime": dates, "open": close + np.random.randn(n) * 1.5,
        "high": close + np.abs(np.random.randn(n)) * 3,
        "low": close - np.abs(np.random.randn(n)) * 3,
        "close": close, "volume": np.random.randint(100, 10000, n).astype(float),
    })

def small_config():
    return FFMConfig(num_features=len(get_model_feature_columns()), hidden_size=64,
                     num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, max_sequence_length=32)

passed = failed = 0
def test(name, fn):
    global passed, failed
    try:
        fn(); print(f"  ✓ {name}"); passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}"); failed += 1

# --- Config Tests ---
test("config_creation", lambda: setattr(c := FFMConfig(), '_', None) or None if c.hidden_size == 256 else (_ for _ in ()).throw(AssertionError()))

def test_config_save_load():
    c = FFMConfig(hidden_size=128)
    with tempfile.TemporaryDirectory() as d:
        c.save_pretrained(d)
        loaded = FFMConfig.from_pretrained(d)
        assert loaded.hidden_size == 128
test("config_save_load", test_config_save_load)

# --- Backbone Tests ---
def test_backbone_forward():
    c = small_config(); m = FFMBackbone(c)
    out = m(torch.randn(4, 32, c.num_features))
    assert out.shape == (4, c.hidden_size)
test("backbone_forward", test_backbone_forward)

def test_backbone_metadata():
    c = small_config(); m = FFMBackbone(c)
    out = m(torch.randn(4, 32, c.num_features), time_of_day=torch.rand(4, 32),
            day_of_week=torch.randint(0, 5, (4, 32)), instrument_ids=torch.randint(0, 4, (4,)),
            session_ids=torch.randint(0, 4, (4, 32)))
    assert out.shape == (4, c.hidden_size)
test("backbone_with_metadata", test_backbone_metadata)

def test_backbone_sequence():
    c = small_config(); m = FFMBackbone(c)
    out = m(torch.randn(4, 32, c.num_features), output_sequence=True)
    assert out.shape == (4, 33, c.hidden_size)
test("backbone_output_sequence", test_backbone_sequence)

# --- Pretraining Tests ---
def test_pretrain_no_labels():
    c = small_config(); m = FFMForPretraining(c)
    out = m(features=torch.randn(4, 32, c.num_features))
    assert "regime_logits" in out and "loss" not in out
test("pretraining_no_labels", test_pretrain_no_labels)

def test_pretrain_with_labels():
    c = small_config(); m = FFMForPretraining(c)
    out = m(features=torch.randn(4, 32, c.num_features), regime_labels=torch.randint(0, 4, (4,)),
            volatility_labels=torch.randint(0, 4, (4,)), structure_labels=torch.randint(0, 3, (4,)),
            range_labels=torch.randint(0, 5, (4,)))
    assert "loss" in out and out["loss"].item() > 0
test("pretraining_with_labels", test_pretrain_with_labels)

def test_pretrain_backward():
    c = small_config(); m = FFMForPretraining(c)
    out = m(features=torch.randn(4, 32, c.num_features), regime_labels=torch.randint(0, 4, (4,)),
            volatility_labels=torch.randint(0, 4, (4,)), structure_labels=torch.randint(0, 3, (4,)),
            range_labels=torch.randint(0, 5, (4,)))
    out["loss"].backward()
    for name, p in m.named_parameters():
        if p.requires_grad: assert p.grad is not None, f"No gradient for {name}"
test("pretraining_backward", test_pretrain_backward)

# --- Classification Tests ---
def test_classification():
    c = small_config(); m = FFMForClassification(c, num_labels=3)
    out = m(features=torch.randn(4, 32, c.num_features))
    assert out["logits"].shape == (4, 3)
test("classification_forward", test_classification)

def test_freeze():
    c = small_config(); m = FFMForClassification(c, num_labels=3)
    before = sum(p.requires_grad for p in m.parameters())
    m.freeze_backbone(freeze_ratio=0.66)
    after = sum(p.requires_grad for p in m.parameters())
    assert after < before
test("freeze_backbone", test_freeze)

# --- Features Tests ---
def test_features():
    df = make_dummy_ohlcv(500)
    features = derive_features(df, instrument="ES")
    for col in get_model_feature_columns():
        assert col in features.columns, f"Missing: {col}"
test("derive_features", test_features)

# --- Labels Tests ---
def test_labels():
    df = make_dummy_ohlcv(500)
    features = derive_features(df, instrument="ES")
    labels = generate_all_labels(features)
    assert all(c in labels.columns for c in ["regime_label", "volatility_label", "structure_label", "range_label"])
test("generate_labels", test_labels)

# --- Dataset Tests ---
def test_dataset():
    df = make_dummy_ohlcv(500)
    features = derive_features(df, instrument="ES")
    labels = generate_all_labels(features)
    ds = FFMDataset(features, labels, seq_len=32)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["features"].shape == (32, len(get_model_feature_columns()))
    assert not torch.isnan(sample["features"]).any()
test("dataset_creation", test_dataset)

print(f"\n{'='*40}\n  {passed} passed, {failed} failed\n{'='*40}")