from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplitSpec:
    development: float = 0.60
    calibration: float = 0.15
    policy: float = 0.10
    oot: float = 0.15
    minimum_weeks_per_partition: int = 2

    def __post_init__(self):
        shares = (self.development, self.calibration, self.policy, self.oot)
        if any(share <= 0 for share in shares):
            raise ValueError("All split shares must be positive")
        if not np.isclose(sum(shares), 1.0):
            raise ValueError("Split shares must sum to one")
        if self.minimum_weeks_per_partition < 1:
            raise ValueError("minimum_weeks_per_partition must be positive")


def assign_temporal_partitions(week, spec: TemporalSplitSpec | None = None) -> pd.Series:
    """Assign entire ordered weeks to four contiguous, non-overlapping partitions."""
    spec = spec or TemporalSplitSpec()
    week_series = pd.Series(week)
    if week_series.isna().any():
        raise ValueError("Cohort week cannot be missing")
    ordered_weeks = np.asarray(sorted(week_series.unique()))
    minimum = spec.minimum_weeks_per_partition * 4
    if len(ordered_weeks) < minimum:
        raise ValueError(f"At least {minimum} unique weeks are required")

    raw_counts = np.asarray(
        [spec.development, spec.calibration, spec.policy, spec.oot], dtype=float
    ) * len(ordered_weeks)
    counts = np.floor(raw_counts).astype(int)
    counts = np.maximum(counts, spec.minimum_weeks_per_partition)

    while counts.sum() > len(ordered_weeks):
        candidates = np.where(counts > spec.minimum_weeks_per_partition)[0]
        if len(candidates) == 0:
            raise ValueError("Not enough weeks for the requested minimum partition size")
        index = candidates[np.argmax(counts[candidates] - raw_counts[candidates])]
        counts[index] -= 1
    while counts.sum() < len(ordered_weeks):
        index = int(np.argmax(raw_counts - counts))
        counts[index] += 1

    names = ("development", "calibration", "policy", "oot")
    mapping = {}
    cursor = 0
    for name, count in zip(names, counts, strict=True):
        for cohort in ordered_weeks[cursor : cursor + count]:
            mapping[cohort] = name
        cursor += count
    return week_series.map(mapping).rename("partition")


def partition_summary(week, target, partition) -> pd.DataFrame:
    frame = pd.DataFrame({"week": week, "target": target, "partition": partition})
    return (
        frame.groupby("partition", observed=True)
        .agg(
            rows=("target", "size"),
            events=("target", "sum"),
            target_rate=("target", "mean"),
            first_week=("week", "min"),
            last_week=("week", "max"),
            weeks=("week", "nunique"),
        )
        .reindex(["development", "calibration", "policy", "oot"])
        .reset_index()
    )
