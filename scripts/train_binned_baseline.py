"""Fit a quantile-binned logistic baseline on the frozen chronological splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.baseline import build_binned_logistic
from src.features import build_manifest, encode_features, load_depth0, transform_dates
from src.metrics import probability_metrics
from src.temporal import assign_temporal_partitions

RAW = Path("data/raw")
REPORTS = Path("reports")
SEED = 20261004


def deterministic_sample(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def main() -> None:
    frame = transform_dates(load_depth0(RAW))
    y = frame["target"].to_numpy().astype(np.int8)
    week = frame["WEEK_NUM"].to_numpy()
    partitions = assign_temporal_partitions(week).to_numpy()
    development = partitions == "development"
    fit = development & (week <= 47)
    calibration = partitions == "calibration"
    manifest = build_manifest(frame, development)
    matrix, _, _ = encode_features(frame, manifest, development)

    fit_idx = deterministic_sample(np.flatnonzero(fit), 600_000)
    model = build_binned_logistic(SEED)
    model.fit(matrix[fit_idx], y[fit_idx])

    calibration_score = model.decision_function(matrix[calibration])
    calibrator = LogisticRegression(random_state=SEED, max_iter=300)
    calibrator.fit(calibration_score.reshape(-1, 1), y[calibration])

    rows = []
    for partition in ("calibration", "policy", "oot"):
        mask = partitions == partition
        score = model.decision_function(matrix[mask])
        probability = calibrator.predict_proba(score.reshape(-1, 1))[:, 1]
        result = probability_metrics(y[mask], probability, week[mask])
        stability = result.pop("stability")
        rows.append(
            {
                "model": "binned_logistic",
                "partition": partition,
                "rows": int(mask.sum()),
                **result,
                **{f"stability_{key}": value for key, value in stability.items()},
            }
        )
    output = REPORTS / "binned_baseline_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
