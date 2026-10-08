"""Stronger, scorecard-like baseline components."""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer


def build_binned_logistic(random_state: int = 20261004):
    """Build a scalable nonlinear baseline with quantile bins and logit loss.

    This is deliberately more competitive than a linear model on raw numeric
    values: monotonicity is not assumed, missingness is explicit, and every
    predictor is represented by development-fitted quantile bins.
    """

    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        KBinsDiscretizer(
            n_bins=10,
            encode="onehot",
            strategy="quantile",
            subsample=200_000,
            random_state=random_state,
        ),
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=5e-5,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
            n_jobs=-1,
        ),
    )
