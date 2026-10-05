from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class StabilityMetric:
    score: float
    mean_gini: float
    slope: float
    falling_rate: float
    residual_std: float
    eligible_weeks: int


def gini_stability(y_true, y_score, week) -> StabilityMetric:
    """Reproduce the competition's weekly Gini stability objective."""
    frame = pd.DataFrame({"target": y_true, "score": y_score, "week": week})
    weekly = []
    for week_num, group in frame.groupby("week", sort=True):
        if group.target.nunique() < 2:
            continue
        weekly.append((float(week_num), 2 * roc_auc_score(group.target, group.score) - 1))
    if len(weekly) < 2:
        raise ValueError("At least two weeks with both outcome classes are required")
    x = np.asarray([item[0] for item in weekly], dtype=float)
    gini = np.asarray([item[1] for item in weekly], dtype=float)
    slope, intercept = np.polyfit(x, gini, 1)
    residuals = gini - (slope * x + intercept)
    falling_rate = min(0.0, float(slope))
    residual_std = float(np.std(residuals))
    score = float(np.mean(gini) + 88.0 * falling_rate - 0.5 * residual_std)
    return StabilityMetric(
        score=score,
        mean_gini=float(np.mean(gini)),
        slope=float(slope),
        falling_rate=falling_rate,
        residual_std=residual_std,
        eligible_weeks=len(weekly),
    )


def expected_calibration_error(y_true, probability, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Outcome and probability must have equal non-zero length")
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return float(abs(y.mean() - p.mean()))
    assignments = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    ece = 0.0
    for index in np.unique(assignments):
        mask = assignments == index
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def probability_metrics(y_true, probability, week=None) -> dict:
    y = np.asarray(y_true)
    p = np.asarray(probability, dtype=float)
    result = {
        "roc_auc": float(roc_auc_score(y, p)),
        "gini": float(2 * roc_auc_score(y, p) - 1),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10": expected_calibration_error(y, p, bins=10),
        "observed_default_rate": float(y.mean()),
        "mean_predicted_pd": float(p.mean()),
    }
    if week is not None:
        result["stability"] = asdict(gini_stability(y, p, week))
    return result


def population_stability_index(reference, current, bins: int = 10, epsilon=1e-6) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference[np.isfinite(reference)], np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_share = np.histogram(current, bins=edges)[0] / len(current)
    ref_share = np.clip(ref_share, epsilon, None)
    cur_share = np.clip(cur_share, epsilon, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def calibration_intercept_offset(y_true, probability, epsilon: float = 1e-6) -> float:
    """Estimate a prevalence-only log-odds correction while preserving rank."""
    observed = float(np.asarray(y_true).mean())
    predicted = float(np.asarray(probability, dtype=float).mean())
    return float(
        logit(np.clip(observed, epsilon, 1 - epsilon))
        - logit(np.clip(predicted, epsilon, 1 - epsilon))
    )


def apply_calibration_intercept(probability, offset: float, epsilon: float = 1e-6):
    probability = np.asarray(probability, dtype=float)
    return expit(logit(np.clip(probability, epsilon, 1 - epsilon)) + offset)


def approval_policy_table(
    y_true,
    probability,
    exposure,
    approval_rates=(0.2, 0.4, 0.6, 0.8),
    lgd_assumption=0.45,
) -> pd.DataFrame:
    """Evaluate low-PD approvals; LGD is an explicit scenario, not an observed label."""
    frame = pd.DataFrame(
        {"target": y_true, "pd": probability, "exposure": np.asarray(exposure, dtype=float)}
    ).sort_values("pd", kind="stable")
    rows = []
    for rate in approval_rates:
        approved = frame.iloc[: max(1, int(len(frame) * rate))]
        expected_loss = approved.pd * approved.exposure * lgd_assumption
        realised_loss_proxy = approved.target * approved.exposure * lgd_assumption
        rows.append(
            {
                "approval_rate": float(len(approved) / len(frame)),
                "pd_cutoff": float(approved.pd.max()),
                "approved_count": len(approved),
                "observed_bad_rate": float(approved.target.mean()),
                "mean_predicted_pd": float(approved.pd.mean()),
                "approved_exposure": float(approved.exposure.sum()),
                "expected_loss_scenario": float(expected_loss.sum()),
                "realised_loss_proxy": float(realised_loss_proxy.sum()),
                "lgd_assumption": float(lgd_assumption),
            }
        )
    return pd.DataFrame(rows)
