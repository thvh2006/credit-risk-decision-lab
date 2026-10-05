"""Generate the immutable real-data profile used by the case study."""

from pathlib import Path

import polars as pl

from src.temporal import TemporalSplitSpec, assign_temporal_partitions, partition_summary

RAW = Path("data/raw")
OUT = Path("reports")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = pl.read_parquet(RAW / "train_base.parquet")
    partition = assign_temporal_partitions(base["WEEK_NUM"].to_numpy(), TemporalSplitSpec())
    summary = partition_summary(base["WEEK_NUM"], base["target"], partition)
    summary.to_csv(OUT / "temporal_partitions.csv", index=False)
    weekly = (
        base.group_by("WEEK_NUM")
        .agg(
            pl.len().alias("applications"),
            pl.col("target").sum().alias("defaults"),
            pl.col("target").mean().alias("default_rate"),
            pl.col("date_decision").min().alias("first_decision_date"),
            pl.col("date_decision").max().alias("last_decision_date"),
        )
        .sort("WEEK_NUM")
    )
    weekly.write_csv(OUT / "weekly_population.csv")


if __name__ == "__main__":
    main()
