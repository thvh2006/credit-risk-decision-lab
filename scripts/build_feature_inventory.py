"""Create a reviewable feature inventory and checksum from the fitted manifest."""

import hashlib
import json
from pathlib import Path

import polars as pl

from src.features import load_depth0, transform_dates
from src.temporal import assign_temporal_partitions

RAW = Path("data/raw")
REPORTS = Path("reports")


def main() -> None:
    frame = transform_dates(load_depth0(RAW))
    partition = assign_temporal_partitions(frame["WEEK_NUM"].to_numpy()).to_numpy()
    development = frame.filter(pl.Series(partition == "development"))
    oot = frame.filter(pl.Series(partition == "oot"))
    metadata = json.loads((REPORTS / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = metadata["feature_manifest"]
    definitions = dict(
        pl.read_csv(RAW / "feature_definitions.csv").select("Variable", "Description").iter_rows()
    )
    internal_columns = set(pl.read_parquet_schema(RAW / "train_static_0_0.parquet"))
    selected = set(manifest["numeric"]) | set(manifest["categorical"])
    sensitive = set(manifest["excluded_sensitive"])
    sparse = set(manifest["excluded_sparse"])
    high_cardinality = set(manifest["excluded_high_cardinality"])
    rows = []
    for column in frame.columns:
        if column in {"case_id", "target", "WEEK_NUM", "MONTH", "date_decision"}:
            continue
        if column in selected:
            status, reason = "model", "Eligible under fitted feature policy"
        elif column in sensitive:
            status, reason = "excluded", "Sensitive/protected or proxy-risk policy"
        elif column in sparse:
            status, reason = "excluded", "More than 98% missing in development"
        elif column in high_cardinality:
            status, reason = "excluded", "Categorical cardinality exceeds 200"
        else:
            status, reason = "excluded", "Constant or unsupported type in development"
        rows.append(
            {
                "feature": column,
                "description": definitions.get(column, "Definition unavailable in source file"),
                "source_table": (
                    "train_static_0" if column in internal_columns else "train_static_cb_0"
                ),
                "transformed_type": str(frame.schema[column]),
                "development_missing_rate": development[column].null_count() / development.height,
                "oot_missing_rate": oot[column].null_count() / oot.height,
                "status": status,
                "reason": reason,
                "date_treatment": (
                    "days before decision date" if column.endswith("D") else "not applicable"
                ),
            }
        )
    output = REPORTS / "feature_inventory.csv"
    pl.DataFrame(rows).sort("status", "source_table", "feature").write_csv(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    (REPORTS / "feature_inventory.sha256").write_text(
        f"{digest}  feature_inventory.csv\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
