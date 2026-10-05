"""Leakage-aware depth-0 feature preparation for Home Credit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

ID_COLUMNS = {"case_id", "target", "WEEK_NUM", "MONTH", "date_decision"}
SENSITIVE_TOKENS = ("birth", "marital", "education", "gender", "sex")


@dataclass(frozen=True)
class FeatureManifest:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    excluded_sensitive: tuple[str, ...]
    excluded_sparse: tuple[str, ...]
    excluded_high_cardinality: tuple[str, ...]

    @property
    def selected(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


def load_depth0(raw_dir: str | Path) -> pl.DataFrame:
    """Join base, vertically split internal static data, and external static data."""
    raw = Path(raw_dir)
    base = pl.read_parquet(raw / "train_base.parquet")
    internal = pl.concat(
        [
            pl.read_parquet(raw / "train_static_0_0.parquet"),
            pl.read_parquet(raw / "train_static_0_1.parquet"),
        ],
        how="vertical_relaxed",
        rechunk=True,
    )
    external = pl.read_parquet(raw / "train_static_cb_0.parquet")
    if internal["case_id"].n_unique() != internal.height:
        raise ValueError("Internal depth-0 table is not one row per case_id")
    if external["case_id"].n_unique() != external.height:
        raise ValueError("External depth-0 table is not one row per case_id")
    frame = base.join(internal, on="case_id", how="left", validate="1:1")
    frame = frame.join(external, on="case_id", how="left", validate="1:1")
    if frame.height != base.height:
        raise ValueError("Depth-0 joins changed base-table row count")
    return frame


def transform_dates(frame: pl.DataFrame) -> pl.DataFrame:
    """Replace absolute predictor dates with days relative to the decision date."""
    decision = pl.col("date_decision").cast(pl.Date)
    expressions = []
    for column, dtype in frame.schema.items():
        if column in ID_COLUMNS or not column.endswith("D"):
            continue
        value = (
            pl.col(column).str.to_date(strict=False)
            if dtype == pl.String
            else pl.col(column).cast(pl.Date, strict=False)
        )
        expressions.append((decision - value).dt.total_days().cast(pl.Float32).alias(column))
    return frame.with_columns(expressions)


def build_manifest(
    frame: pl.DataFrame,
    development_mask: np.ndarray,
    missing_limit: float = 0.98,
    category_limit: int = 200,
) -> FeatureManifest:
    """Fit feature eligibility rules on development data only."""
    development = frame.filter(pl.Series(development_mask))
    numeric: list[str] = []
    categorical: list[str] = []
    sensitive: list[str] = []
    sparse: list[str] = []
    high_cardinality: list[str] = []
    for column, dtype in frame.schema.items():
        if column in ID_COLUMNS:
            continue
        if any(token in column.lower() for token in SENSITIVE_TOKENS):
            sensitive.append(column)
            continue
        series = development[column]
        if series.null_count() / max(1, len(series)) > missing_limit:
            sparse.append(column)
            continue
        if dtype == pl.String:
            if series.n_unique() > category_limit:
                high_cardinality.append(column)
            elif series.n_unique() > 1:
                categorical.append(column)
        elif (dtype.is_numeric() or dtype == pl.Boolean) and series.n_unique() > 1:
            numeric.append(column)
    return FeatureManifest(
        numeric=tuple(numeric),
        categorical=tuple(categorical),
        excluded_sensitive=tuple(sensitive),
        excluded_sparse=tuple(sparse),
        excluded_high_cardinality=tuple(high_cardinality),
    )


def encode_features(
    frame: pl.DataFrame, manifest: FeatureManifest, development_mask: np.ndarray
) -> tuple[np.ndarray, list[int], dict[str, dict[str, float]]]:
    """Create a float32 matrix with development-only frequency encodings."""
    columns: list[np.ndarray] = []
    category_indices: list[int] = []
    vocabularies: dict[str, dict[str, float]] = {}
    for column in manifest.numeric:
        values = frame[column].cast(pl.Float32, strict=False).to_numpy()
        columns.append(values)
    for column in manifest.categorical:
        development_values = frame[column].filter(pl.Series(development_mask)).drop_nulls()
        counts = development_values.value_counts()
        denominator = max(1, len(development_values))
        vocabulary = {
            str(row[column]): float(row["count"] / denominator)
            for row in counts.iter_rows(named=True)
        }
        vocabularies[column] = vocabulary
        encoded = np.fromiter(
            (vocabulary.get(str(value), 0.0) if value is not None else 0.0 for value in frame[column]),
            dtype=np.float32,
            count=frame.height,
        )
        category_indices.append(len(columns))
        columns.append(encoded)
    return np.column_stack(columns).astype(np.float32, copy=False), category_indices, vocabularies
