from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class TriggerThresholds:
    auc_drop_warning: float = 0.03
    auc_drop_breach: float = 0.05
    calibration_gap_warning: float = 0.02
    calibration_gap_breach: float = 0.05


def weekly_performance(y_true, probability, week) -> pd.DataFrame:
    frame = pd.DataFrame({"target": y_true, "pd": probability, "week": week})
    rows = []
    for cohort, group in frame.groupby("week", sort=True):
        auc = np.nan
        if group.target.nunique() == 2:
            auc = float(roc_auc_score(group.target, group.pd))
        rows.append(
            {
                "week": cohort,
                "applications": len(group),
                "defaults": int(group.target.sum()),
                "observed_default_rate": float(group.target.mean()),
                "mean_predicted_pd": float(group.pd.mean()),
                "calibration_gap": float(group.pd.mean() - group.target.mean()),
                "roc_auc": auc,
                "gini": float(2 * auc - 1) if np.isfinite(auc) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def monitoring_status(
    reference_auc: float,
    current_auc: float,
    calibration_gap: float,
    thresholds: TriggerThresholds | None = None,
) -> dict:
    thresholds = thresholds or TriggerThresholds()
    auc_drop = reference_auc - current_auc
    absolute_gap = abs(calibration_gap)
    if auc_drop >= thresholds.auc_drop_breach or absolute_gap >= thresholds.calibration_gap_breach:
        status = "breach"
    elif (
        auc_drop >= thresholds.auc_drop_warning
        or absolute_gap >= thresholds.calibration_gap_warning
    ):
        status = "warning"
    else:
        status = "pass"
    return {
        "status": status,
        "auc_drop": float(auc_drop),
        "absolute_calibration_gap": float(absolute_gap),
    }
