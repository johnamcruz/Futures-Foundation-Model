import numpy as np


def test_run_probe_uses_explicit_parent_checkpoint_as_baseline(monkeypatch):
    from futures_foundation.finetune import ssl_probe

    calls = []

    def fake_embed(_big, starts, _seq, *, ckpt, **_kwargs):
        calls.append(ckpt)
        value = 2.0 if ckpt == "child.pt" else 1.0
        return np.full((len(starts), 3), value, np.float32), np.asarray(starts)

    monkeypatch.setattr(
        "futures_foundation.finetune._ssl_torch.embed_encoder", fake_embed)
    monkeypatch.setattr(
        ssl_probe, "targets_from_windows",
        lambda _big, starts, _seq, fwd_k=16: {
            "vol": np.arange(len(starts), dtype=np.float32),
        })
    monkeypatch.setattr(
        ssl_probe, "compare",
        lambda child, parent, *_args, **_kwargs: {
            "per_target": {},
            "mean_core_delta": float(child.mean() - parent.mean()),
            "learns_regime_vol_structure": True,
            "per_stream": {},
            "stream_win_rate": 1.0,
            "average_target_win_rate": 1.0,
            "worst_stream_win_rate": 1.0,
        })

    starts = np.r_[np.arange(0, 100), np.arange(1000, 1100)]
    result = ssl_probe.run_probe(
        np.zeros((1200, 5), np.float32), starts, 8, "child.pt",
        baseline_ckpt="parent.pt", verbose=False)

    assert calls == ["child.pt", "parent.pt"]
    assert result["comparison_baseline"] == "parent.pt"
    assert result["mean_core_delta"] == 1.0
