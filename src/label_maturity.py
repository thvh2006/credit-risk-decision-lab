"""Diagnostics that keep cohort drift separate from label maturity."""

from __future__ import annotations

import pandas as pd


def maturity_diagnostic(
    weekly: pd.DataFrame,
    reference_last_week: int = 54,
    late_first_week: int = 78,
) -> dict[str, float | int | bool | str]:
    required = {"WEEK_NUM", "applications", "defaults", "default_rate"}
    missing = required.difference(weekly.columns)
    if missing:
        raise ValueError(f"Missing weekly fields: {sorted(missing)}")
    reference = weekly.loc[weekly["WEEK_NUM"] <= reference_last_week]
    late = weekly.loc[weekly["WEEK_NUM"] >= late_first_week]
    if reference.empty or late.empty:
        raise ValueError("Reference and late cohorts must both contain rows")

    reference_rate = float(reference["defaults"].sum() / reference["applications"].sum())
    late_rate = float(late["defaults"].sum() / late["applications"].sum())
    reference_volume = float(reference["applications"].mean())
    late_volume = float(late["applications"].mean())
    return {
        "reference_last_week": reference_last_week,
        "late_first_week": late_first_week,
        "reference_default_rate": reference_rate,
        "late_default_rate": late_rate,
        "relative_default_rate_change": late_rate / reference_rate - 1,
        "reference_mean_weekly_applications": reference_volume,
        "late_mean_weekly_applications": late_volume,
        "relative_volume_change": late_volume / reference_volume - 1,
        "right_censoring_can_be_ruled_out": False,
        "reason": (
            "The public target has no documented outcome-window end date or per-row "
            "label-maturity timestamp. Cohort drift and right-censoring cannot be "
            "separated from this table alone."
        ),
    }
