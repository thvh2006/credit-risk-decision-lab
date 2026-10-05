from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataContract:
    key: str = "case_id"
    target: str = "target"
    cohort: str = "WEEK_NUM"
    decision_date: str = "date_decision"


FORBIDDEN_MODEL_FIELDS = frozenset(
    {
        "case_id",
        "target",
        "WEEK_NUM",
        "MONTH",
        "date_decision",
        "row_index",
        "source_file",
        "partition",
    }
)


def validate_application_frame(frame: pd.DataFrame, contract: DataContract | None = None) -> dict:
    contract = contract or DataContract()
    required = {contract.key, contract.target, contract.cohort, contract.decision_date}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[contract.key].isna().any():
        raise ValueError("Application key contains missing values")
    if frame[contract.key].duplicated().any():
        raise ValueError("Application key must be unique at model grain")
    labels = set(frame[contract.target].dropna().unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Target must be binary; found {sorted(labels)}")
    parsed_date = pd.to_datetime(frame[contract.decision_date], errors="coerce")
    if parsed_date.isna().any():
        raise ValueError("Decision date contains missing or unparsable values")
    if frame[contract.cohort].isna().any():
        raise ValueError("Cohort week contains missing values")
    return {
        "rows": len(frame),
        "columns": int(frame.shape[1]),
        "target_rate": float(frame[contract.target].mean()),
        "first_week": int(frame[contract.cohort].min()),
        "last_week": int(frame[contract.cohort].max()),
        "unique_weeks": int(frame[contract.cohort].nunique()),
        "first_decision_date": parsed_date.min().isoformat(),
        "last_decision_date": parsed_date.max().isoformat(),
    }


def model_feature_columns(frame: pd.DataFrame, extra_exclusions=()) -> list[str]:
    exclusions = FORBIDDEN_MODEL_FIELDS.union(extra_exclusions)
    return sorted(column for column in frame.columns if column not in exclusions)
