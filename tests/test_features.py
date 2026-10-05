import numpy as np
import polars as pl

from src.features import build_manifest, encode_features, transform_dates


def test_dates_become_days_before_decision_and_sensitive_fields_are_excluded():
    frame = pl.DataFrame(
        {
            "case_id": [1, 2, 3],
            "target": [0, 1, 0],
            "WEEK_NUM": [0, 1, 2],
            "MONTH": [201901, 201901, 201901],
            "date_decision": ["2019-01-10", "2019-01-12", "2019-01-15"],
            "eventdate_1D": ["2019-01-08", "2019-01-10", None],
            "birthdate_1D": ["1980-01-01", "1981-01-01", "1982-01-01"],
            "amount_1A": [1.0, 2.0, 3.0],
            "channel_1M": ["branch", "web", "web"],
        }
    ).with_columns(pl.col("date_decision").str.to_date())
    transformed = transform_dates(frame)
    assert transformed["eventdate_1D"].to_list() == [2.0, 2.0, None]
    manifest = build_manifest(transformed, np.array([True, True, True]))
    assert "birthdate_1D" in manifest.excluded_sensitive
    assert "amount_1A" in manifest.numeric
    assert "channel_1M" in manifest.categorical


def test_frequency_encoding_is_fitted_on_development_only():
    frame = pl.DataFrame(
        {
            "case_id": [1, 2, 3, 4],
            "target": [0, 0, 1, 1],
            "WEEK_NUM": [0, 0, 1, 2],
            "MONTH": [1, 1, 1, 1],
            "date_decision": [None, None, None, None],
            "amount_1A": [1.0, 2.0, 3.0, 4.0],
            "channel_1M": ["web", "web", "branch", "future-only"],
        }
    )
    development = np.array([True, True, True, False])
    manifest = build_manifest(frame, development)
    matrix, _, vocabularies = encode_features(frame, manifest, development)
    categorical_index = len(manifest.numeric)
    assert vocabularies["channel_1M"]["web"] == 2 / 3
    assert matrix[3, categorical_index] == 0.0
