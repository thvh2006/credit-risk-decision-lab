import pandas as pd
import pytest

from src.temporal import TemporalSplitSpec, assign_temporal_partitions, partition_summary


def test_temporal_partitions_are_contiguous_and_keep_weeks_whole():
    weeks = pd.Series([week for week in range(20) for _ in range(3)])
    partition = assign_temporal_partitions(weeks)
    frame = pd.DataFrame({"week": weeks, "partition": partition})

    assert frame.groupby("week").partition.nunique().max() == 1
    boundaries = frame.groupby("partition").week.agg(["min", "max"])
    assert boundaries.loc["development", "max"] < boundaries.loc["calibration", "min"]
    assert boundaries.loc["calibration", "max"] < boundaries.loc["policy", "min"]
    assert boundaries.loc["policy", "max"] < boundaries.loc["oot", "min"]


def test_split_spec_rejects_invalid_shares():
    with pytest.raises(ValueError):
        TemporalSplitSpec(development=0.5)


def test_partition_summary_keeps_required_order():
    weeks = pd.Series([week for week in range(20) for _ in range(2)])
    target = pd.Series([0, 1] * 20)
    partition = assign_temporal_partitions(weeks)
    result = partition_summary(weeks, target, partition)
    assert result.partition.tolist() == ["development", "calibration", "policy", "oot"]
    assert result.rows.sum() == 40
