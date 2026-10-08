import numpy as np

from src.baseline import build_binned_logistic


def test_binned_logistic_handles_missing_values_and_nonlinearity():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2_000, 3))
    x[::17, 1] = np.nan
    y = ((x[:, 0] < -0.8) | (x[:, 0] > 0.8)).astype(int)
    model = build_binned_logistic(random_state=42)
    model.fit(x, y)
    probability = model.predict_proba(x[:25])[:, 1]
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
