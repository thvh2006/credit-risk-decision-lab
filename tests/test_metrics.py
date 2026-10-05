import numpy as np
import pytest

from src.metrics import (
    apply_calibration_intercept,
    approval_policy_table,
    calibration_intercept_offset,
    expected_calibration_error,
    gini_stability,
    population_stability_index,
)


def test_calibration_intercept_preserves_rank_and_moves_mean_toward_observed_rate():
    y = np.array([0, 0, 0, 1])
    probability = np.array([0.2, 0.3, 0.4, 0.5])
    offset = calibration_intercept_offset(y, probability)
    refreshed = apply_calibration_intercept(probability, offset)
    assert np.array_equal(np.argsort(probability), np.argsort(refreshed))
    assert abs(refreshed.mean() - y.mean()) < abs(probability.mean() - y.mean())


def test_stability_penalises_falling_weekly_gini():
    rng = np.random.default_rng(42)
    week = np.repeat(np.arange(6), 2_000)
    y = rng.binomial(1, 0.2, len(week))
    stable = y + rng.normal(0, 0.8, len(y))
    degrading = y + rng.normal(0, 0.45 + week * 0.25, len(y))
    assert gini_stability(y, stable, week).score > gini_stability(y, degrading, week).score


def test_ece_is_zero_for_exact_group_rates():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, bins=2) == pytest.approx(0.0)


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 30_000)
    same = rng.normal(0, 1, 30_000)
    shifted = rng.normal(1, 1, 30_000)
    assert population_stability_index(reference, shifted) > population_stability_index(
        reference, same
    )


def test_stricter_approval_policy_has_lower_bad_rate():
    y = np.array([0, 0, 0, 1, 1, 1])
    pd = np.array([0.01, 0.02, 0.05, 0.20, 0.40, 0.80])
    exposure = np.full(6, 1_000.0)
    table = approval_policy_table(y, pd, exposure, approval_rates=(0.5, 1.0))
    assert table.iloc[0].observed_bad_rate < table.iloc[1].observed_bad_rate
    assert table.iloc[0].approved_exposure < table.iloc[1].approved_exposure
