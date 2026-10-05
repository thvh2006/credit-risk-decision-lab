"""Train chronological baseline/challenger models and emit auditable reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import build_manifest, encode_features, load_depth0, transform_dates
from src.metrics import (
    apply_calibration_intercept,
    approval_policy_table,
    calibration_intercept_offset,
    population_stability_index,
    probability_metrics,
)
from src.temporal import assign_temporal_partitions

RAW = Path("data/raw")
REPORTS = Path("reports")
SEED = 20261004


def deterministic_sample(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def fit_calibrator(y: np.ndarray, score: np.ndarray) -> LogisticRegression:
    calibration = LogisticRegression(random_state=SEED, max_iter=300)
    calibration.fit(score.reshape(-1, 1), y)
    return calibration


def calibrated(calibrator: LogisticRegression, score: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(score.reshape(-1, 1))[:, 1]


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    frame = transform_dates(load_depth0(RAW))
    y = frame["target"].to_numpy().astype(np.int8)
    week = frame["WEEK_NUM"].to_numpy()
    partitions = assign_temporal_partitions(week).to_numpy()
    development = partitions == "development"
    fit = development & (week <= 47)
    tune = development & (week > 47)
    calibration = partitions == "calibration"
    policy = partitions == "policy"
    oot = partitions == "oot"

    manifest = build_manifest(frame, development)
    X, categorical_indices, _ = encode_features(frame, manifest, development)
    feature_names = list(manifest.selected)
    fit_idx = deterministic_sample(np.flatnonzero(fit), 600_000, SEED)

    # Transparent linear baseline: numeric predictors only, median imputation and
    # missingness flags. SGD keeps the million-row baseline computationally honest.
    numeric_count = len(manifest.numeric)
    linear = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=150,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
            n_jobs=-1,
        ),
    )
    linear.fit(X[fit_idx, :numeric_count], y[fit_idx])
    linear_cal = fit_calibrator(
        y[calibration], linear.decision_function(X[calibration, :numeric_count])
    )

    challenger = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=1_200,
        learning_rate=0.035,
        max_depth=6,
        min_child_weight=250,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="auc",
        early_stopping_rounds=60,
        random_state=SEED,
        n_jobs=-1,
    )
    challenger.fit(
        X[fit_idx],
        y[fit_idx],
        eval_set=[(X[tune], y[tune])],
        verbose=False,
    )
    raw_calibration = challenger.predict_proba(X[calibration])[:, 1]
    clipped_calibration = np.clip(raw_calibration, 1e-6, 1 - 1e-6)
    challenger_cal = fit_calibrator(
        y[calibration], np.log(clipped_calibration / (1 - clipped_calibration))
    )

    rows = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, mask in (("calibration", calibration), ("policy", policy), ("oot", oot)):
        linear_pd = calibrated(
            linear_cal, linear.decision_function(X[mask, :numeric_count])
        )
        raw = challenger.predict_proba(X[mask])[:, 1]
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        challenger_pd = calibrated(challenger_cal, np.log(clipped / (1 - clipped)))
        predictions[name] = {"linear": linear_pd, "challenger": challenger_pd}
        for model_name, probability in predictions[name].items():
            result = probability_metrics(y[mask], probability, week[mask])
            stability = result.pop("stability")
            rows.append(
                {
                    "model": model_name,
                    "partition": name,
                    "rows": int(mask.sum()),
                    **result,
                    **{f"stability_{key}": value for key, value in stability.items()},
                }
            )
    pd.DataFrame(rows).to_csv(REPORTS / "model_metrics.csv", index=False)

    exposure_column = "credamount_770A"
    exposure = frame[exposure_column].fill_null(0).to_numpy()
    policies = []
    for lgd in (0.25, 0.45, 0.65):
        table = approval_policy_table(
            y[policy], predictions["policy"]["challenger"], exposure[policy], lgd_assumption=lgd
        )
        table.insert(0, "selection_partition", "policy")
        table["oot_score_psi"] = population_stability_index(
            predictions["policy"]["challenger"], predictions["oot"]["challenger"]
        )
        policies.append(table)
    pd.concat(policies, ignore_index=True).to_csv(REPORTS / "policy_sensitivity.csv", index=False)

    # Freeze cut-offs on policy weeks, then apply them unchanged to locked OOT weeks.
    oot_policy_rows = []
    policy_pd = predictions["policy"]["challenger"]
    oot_pd = predictions["oot"]["challenger"]
    for target_approval in (0.2, 0.4, 0.6, 0.8):
        cutoff = float(np.quantile(policy_pd, target_approval))
        approved = oot_pd <= cutoff
        for lgd in (0.25, 0.45, 0.65):
            oot_policy_rows.append(
                {
                    "target_policy_approval_rate": target_approval,
                    "frozen_pd_cutoff": cutoff,
                    "oot_approval_rate": float(approved.mean()),
                    "oot_approved_count": int(approved.sum()),
                    "oot_observed_bad_rate": float(y[oot][approved].mean()),
                    "oot_mean_predicted_pd": float(oot_pd[approved].mean()),
                    "oot_approved_exposure": float(exposure[oot][approved].sum()),
                    "oot_expected_loss_scenario": float(
                        (oot_pd[approved] * exposure[oot][approved] * lgd).sum()
                    ),
                    "oot_realised_loss_proxy": float(
                        (y[oot][approved] * exposure[oot][approved] * lgd).sum()
                    ),
                    "lgd_assumption": lgd,
                }
            )
    pd.DataFrame(oot_policy_rows).to_csv(REPORTS / "oot_policy_validation.csv", index=False)

    # Production-like delayed-label exercise: estimate only a calibration intercept
    # on early OOT weeks and assess it on strictly later OOT weeks. Ranking is fixed.
    early_oot = oot & (week <= 84)
    late_oot = oot & (week >= 85)
    early_pd = predictions["oot"]["challenger"][week[oot] <= 84]
    late_pd = predictions["oot"]["challenger"][week[oot] >= 85]
    offset = calibration_intercept_offset(y[early_oot], early_pd)
    refreshed_late_pd = apply_calibration_intercept(late_pd, offset)
    calibration_rows = []
    for version, probability in (
        ("original_calibration", late_pd),
        ("intercept_refresh", refreshed_late_pd),
    ):
        result = probability_metrics(y[late_oot], probability, week[late_oot])
        stability = result.pop("stability")
        calibration_rows.append(
            {
                "version": version,
                "fit_weeks": "78-84" if version == "intercept_refresh" else "55-68",
                "evaluation_weeks": "85-91",
                "rows": int(late_oot.sum()),
                "logit_intercept_offset": offset if version == "intercept_refresh" else 0.0,
                **result,
                **{f"stability_{key}": value for key, value in stability.items()},
            }
        )
    pd.DataFrame(calibration_rows).to_csv(REPORTS / "recalibration_backtest.csv", index=False)

    # Privacy-safe reason-code evidence: aggregate local positive-risk contributions,
    # never publish application-level explanations or identifiers.
    reason_idx = deterministic_sample(np.flatnonzero(oot), 10_000, SEED + 1)
    contributions = challenger.get_booster().predict(
        xgb.DMatrix(X[reason_idx]), pred_contribs=True
    )[:, :-1]
    positive = np.where(contributions > 0, contributions, -np.inf)
    top_reason = positive.argmax(axis=1)
    has_positive_reason = np.isfinite(positive.max(axis=1))
    top_reason = top_reason[has_positive_reason]
    definitions = dict(
        pd.read_csv(RAW / "feature_definitions.csv")[["Variable", "Description"]].itertuples(
            index=False, name=None
        )
    )
    reason_rows = []
    for index, count in zip(*np.unique(top_reason, return_counts=True), strict=True):
        mask = top_reason == index
        feature = feature_names[index]
        reason_rows.append(
            {
                "feature": feature,
                "description": definitions.get(feature, "Definition unavailable in source file"),
                "applications_as_top_reason": int(count),
                "share_as_top_reason": float(count / len(reason_idx)),
                "mean_positive_log_odds_contribution": float(
                    contributions[has_positive_reason][mask, index].mean()
                ),
            }
        )
    pd.DataFrame(reason_rows).sort_values(
        "applications_as_top_reason", ascending=False
    ).to_csv(REPORTS / "reason_code_summary.csv", index=False)

    importance = pd.DataFrame(
        {"feature": feature_names, "gain": challenger.feature_importances_}
    ).sort_values("gain", ascending=False)
    importance.to_csv(REPORTS / "feature_importance.csv", index=False)
    Path("models").mkdir(exist_ok=True)
    challenger.save_model("models/challenger.json")
    metadata = {
        "seed": SEED,
        "fit_rows": len(fit_idx),
        "fit_weeks": [int(week[fit].min()), int(week[fit].max())],
        "tune_weeks": [int(week[tune].min()), int(week[tune].max())],
        "best_iteration": int(challenger.best_iteration),
        "feature_manifest": asdict(manifest),
        "categorical_feature_count": len(categorical_indices),
        "exposure_proxy": exposure_column,
        "claim_boundary": "Source target is not represented as regulatory PD.",
    }
    (REPORTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
