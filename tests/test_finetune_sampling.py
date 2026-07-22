import numpy as np
import pytest

from futures_foundation.finetune.sampling import sample_epoch_rows


def test_bar_proportional_is_a_deterministic_without_replacement_shuffle():
    rows = np.asarray([1, 2, 4, 5, 7])
    streams = np.asarray(["x", "a", "a", "x", "b", "b", "x", "c"])
    first = sample_epoch_rows(rows, streams, seed=7, epoch=2)
    second = sample_epoch_rows(rows, streams, seed=7, epoch=2)
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == sorted(rows.tolist())


def test_uniform_stream_equalizes_counts_without_reading_outside_train_rows():
    streams = np.asarray(["one"] * 9 + ["five"] * 2 + ["fifteen"])
    train_rows = np.arange(len(streams), dtype=np.int64)
    sampled = sample_epoch_rows(
        train_rows, streams, mode="uniform_stream", seed=11, epoch=0)
    counts = {stream: int((streams[sampled] == stream).sum()) for stream in set(streams)}
    assert len(sampled) == len(train_rows)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert set(sampled).issubset(set(train_rows))
    assert len(np.unique(sampled[streams[sampled] == "fifteen"])) == 1


def test_uniform_stream_changes_epoch_draw_but_remains_reproducible():
    streams = np.asarray(["fast"] * 20 + ["slow"] * 4)
    rows = np.arange(len(streams), dtype=np.int64)
    epoch0 = sample_epoch_rows(rows, streams, mode="uniform_stream", seed=3, epoch=0)
    epoch1 = sample_epoch_rows(rows, streams, mode="uniform_stream", seed=3, epoch=1)
    np.testing.assert_array_equal(
        epoch1, sample_epoch_rows(rows, streams, mode="uniform_stream", seed=3, epoch=1))
    assert not np.array_equal(epoch0, epoch1)


def test_epoch_sampler_rejects_unknown_modes_and_out_of_range_rows():
    with pytest.raises(ValueError, match="unsupported sampling mode"):
        sample_epoch_rows([0], ["x"], mode="future_mode")
    with pytest.raises(ValueError, match="outside stream_ids"):
        sample_epoch_rows([1], ["x"], mode="uniform_stream")
