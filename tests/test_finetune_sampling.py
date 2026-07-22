import numpy as np
import pytest

from futures_foundation.finetune.sampling import (
    resolve_epoch_sampling_mode, sample_epoch_rows)


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


def test_two_stage_sampling_schedule_switches_at_declared_halfway_point():
    assert [resolve_epoch_sampling_mode("bar_then_uniform", ep, 6) for ep in range(6)] == [
        "bar_proportional", "bar_proportional", "bar_proportional",
        "uniform_stream", "uniform_stream", "uniform_stream",
    ]
    assert resolve_epoch_sampling_mode("uniform_stream", 0, 1) == "uniform_stream"
    with pytest.raises(ValueError, match="unsupported sampling curriculum"):
        resolve_epoch_sampling_mode("adaptive_oos", 0, 10)


def test_three_stage_schedule_reserves_final_epochs_for_primary_stream_refinement():
    assert [
        resolve_epoch_sampling_mode("bar_then_uniform_then_primary", ep, 10)
        for ep in range(10)
    ] == [
        "bar_proportional", "bar_proportional", "bar_proportional",
        "bar_proportional", "bar_proportional",
        "uniform_stream", "uniform_stream", "uniform_stream",
        "uniform_primary_stream", "uniform_primary_stream",
    ]


def test_primary_stream_refinement_is_train_only_and_balanced_within_primary_group():
    streams = np.asarray(
        ["NQ@1min"] * 8 + ["NQ@15min"] * 2 + ["GC@1min"] * 9)
    rows = np.arange(len(streams), dtype=np.int64)
    sampled = sample_epoch_rows(
        rows, streams, mode="uniform_primary_stream",
        primary_streams=("NQ@1min", "NQ@15min"), seed=7, epoch=8)
    assert len(sampled) == 10
    assert set(streams[sampled]) == {"NQ@1min", "NQ@15min"}
    counts = [int((streams[sampled] == stream).sum())
              for stream in ("NQ@1min", "NQ@15min")]
    assert counts == [5, 5]
    with pytest.raises(ValueError, match="no represented primary streams"):
        sample_epoch_rows(
            rows, streams, mode="uniform_primary_stream",
            primary_streams=("ES@3min",), seed=7, epoch=8)
