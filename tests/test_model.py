"""Unit tests for Futures Foundation Model — pytest compatible."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from futures_foundation import (
    FFMConfig, FFMBackbone, FFMForPretraining, FFMForClassification,
    FFMForRegression, FFMForStrategyWithRisk,
    derive_features, generate_all_labels, get_model_feature_columns, FFMDataset,
)


# =============================================================================
# Helpers
# =============================================================================

def make_dummy_ohlcv(n=1000, seed=42):
    """Generate dummy OHLCV data. n>=1000 ensures enough valid rows after rolling warmup."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    close = 5000 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "datetime": dates,
        "open": close + np.random.randn(n) * 1.5,
        "high": close + np.abs(np.random.randn(n)) * 3,
        "low": close - np.abs(np.random.randn(n)) * 3,
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })


SEQ_LEN = 32

def small_config():
    return FFMConfig(
        num_features=len(get_model_feature_columns()),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_sequence_length=SEQ_LEN,
    )


# =============================================================================
# Config Tests
# =============================================================================

def test_config_creation():
    c = FFMConfig()
    assert c.hidden_size == 256
    assert c.num_features == 42


def test_config_save_load():
    c = FFMConfig(hidden_size=128)
    with tempfile.TemporaryDirectory() as d:
        c.save_pretrained(d)
        loaded = FFMConfig.from_pretrained(d)
        assert loaded.hidden_size == 128


# =============================================================================
# Backbone Tests
# =============================================================================

def test_backbone_forward():
    c = small_config()
    m = FFMBackbone(c)
    out = m(torch.randn(4, SEQ_LEN, c.num_features))
    assert out.shape == (4, c.hidden_size)


def test_backbone_metadata():
    c = small_config()
    m = FFMBackbone(c)
    out = m(
        torch.randn(4, SEQ_LEN, c.num_features),
        time_of_day=torch.rand(4, SEQ_LEN),
        day_of_week=torch.randint(0, 5, (4, SEQ_LEN)),
        instrument_ids=torch.randint(0, 4, (4,)),
        session_ids=torch.randint(0, 4, (4, SEQ_LEN)),
    )
    assert out.shape == (4, c.hidden_size)


def test_backbone_sequence():
    c = small_config()
    m = FFMBackbone(c)
    out = m(torch.randn(4, SEQ_LEN, c.num_features), output_sequence=True)
    assert out.shape == (4, SEQ_LEN + 1, c.hidden_size)  # +1 for CLS


# =============================================================================
# Pretraining Tests
# =============================================================================

def test_pretrain_no_labels():
    c = small_config()
    m = FFMForPretraining(c)
    out = m(features=torch.randn(4, SEQ_LEN, c.num_features))
    assert "regime_logits" in out
    assert "loss" not in out


def test_pretrain_with_labels():
    c = small_config()
    m = FFMForPretraining(c)
    out = m(
        features=torch.randn(4, SEQ_LEN, c.num_features),
        regime_labels=torch.randint(0, 4, (4,)),
        volatility_labels=torch.randint(0, 4, (4,)),
        structure_labels=torch.randint(0, 3, (4,)),
        range_labels=torch.randint(0, 5, (4,)),
    )
    assert "loss" in out
    assert out["loss"].item() > 0


def test_pretrain_backward():
    c = small_config()
    m = FFMForPretraining(c)
    out = m(
        features=torch.randn(4, SEQ_LEN, c.num_features),
        time_of_day=torch.rand(4, SEQ_LEN),
        day_of_week=torch.randint(0, 5, (4, SEQ_LEN)),
        instrument_ids=torch.randint(0, 4, (4,)),
        session_ids=torch.randint(0, 4, (4, SEQ_LEN)),
        regime_labels=torch.randint(0, 4, (4,)),
        volatility_labels=torch.randint(0, 4, (4,)),
        structure_labels=torch.randint(0, 3, (4,)),
        range_labels=torch.randint(0, 5, (4,)),
    )
    out["loss"].backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


# =============================================================================
# Classification Tests
# =============================================================================

def test_classification():
    c = small_config()
    m = FFMForClassification(c, num_labels=3)
    out = m(features=torch.randn(4, SEQ_LEN, c.num_features))
    assert out["logits"].shape == (4, 3)


def test_freeze():
    c = small_config()
    m = FFMForClassification(c, num_labels=3)
    before = sum(p.requires_grad for p in m.parameters())
    m.freeze_backbone(freeze_ratio=0.66)
    after = sum(p.requires_grad for p in m.parameters())
    assert after < before


# =============================================================================
# Regression Tests
# =============================================================================

def test_regression_forward():
    c = small_config()
    m = FFMForRegression(c, num_targets=2)
    out = m(features=torch.randn(4, SEQ_LEN, c.num_features))
    assert out["predictions"].shape == (4, 2)
    assert (out["predictions"] >= 0).all(), "Softplus should force positive outputs"


def test_regression_with_labels():
    c = small_config()
    m = FFMForRegression(c, num_targets=2)
    out = m(
        features=torch.randn(4, SEQ_LEN, c.num_features),
        labels=torch.rand(4, 2) * 3,
    )
    assert "loss" in out
    assert out["loss"].item() > 0


# =============================================================================
# Combined Strategy + Risk Tests
# =============================================================================

def test_strategy_with_risk_forward():
    c = small_config()
    m = FFMForStrategyWithRisk(c, num_labels=3, num_risk_targets=2)
    out = m(features=torch.randn(4, SEQ_LEN, c.num_features))
    assert out["signal_logits"].shape == (4, 3)
    assert out["risk_predictions"].shape == (4, 2)
    assert "loss" not in out


def test_strategy_with_risk_loss():
    c = small_config()
    m = FFMForStrategyWithRisk(c, num_labels=3, num_risk_targets=2)
    out = m(
        features=torch.randn(4, SEQ_LEN, c.num_features),
        signal_labels=torch.randint(0, 3, (4,)),
        risk_labels=torch.rand(4, 2) * 3,
    )
    assert "loss" in out
    assert "signal_loss" in out
    assert "risk_loss" in out


# =============================================================================
# Feature Tests
# =============================================================================

def test_features():
    df = make_dummy_ohlcv(1000)
    features = derive_features(df, instrument="ES")
    for col in get_model_feature_columns():
        assert col in features.columns, f"Missing: {col}"


def test_features_valid_ratio():
    """Verify that the NaN fix works — should have >90% valid rows for clean data."""
    df = make_dummy_ohlcv(1000)
    features = derive_features(df, instrument="ES")
    feature_cols = get_model_feature_columns()
    valid = features[feature_cols].notna().all(axis=1).sum()
    ratio = valid / len(features)
    assert ratio > 0.90, f"Only {ratio:.1%} valid rows — NaN propagation bug"


# =============================================================================
# Label Tests
# =============================================================================

def test_labels():
    df = make_dummy_ohlcv(1000)
    features = derive_features(df, instrument="ES")
    labels = generate_all_labels(features)
    expected_cols = ["regime_label", "volatility_label", "structure_label", "range_label"]
    assert all(c in labels.columns for c in expected_cols)


# =============================================================================
# Dataset Tests
# =============================================================================

def test_dataset():
    df = make_dummy_ohlcv(1000)
    features = derive_features(df, instrument="ES")
    labels = generate_all_labels(features)
    ds = FFMDataset(features, labels, seq_len=SEQ_LEN)
    assert len(ds) > 0, f"Dataset empty — need more data for rolling warmup"
    sample = ds[0]
    assert sample["features"].shape == (SEQ_LEN, len(get_model_feature_columns()))
    assert not torch.isnan(sample["features"]).any()