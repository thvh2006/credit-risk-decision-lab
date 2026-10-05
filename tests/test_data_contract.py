import pandas as pd
import pytest

from src.data_contract import model_feature_columns, validate_application_frame


def valid_frame():
    return pd.DataFrame(
        {
            "case_id": [1, 2],
            "target": [0, 1],
            "WEEK_NUM": [3, 4],
            "MONTH": [202301, 202301],
            "date_decision": ["2023-01-01", "2023-01-08"],
            "income_mainA": [1000.0, 1500.0],
        }
    )


def test_data_contract_returns_profile_and_feature_allowlist():
    frame = valid_frame()
    profile = validate_application_frame(frame)
    assert profile["rows"] == 2
    assert model_feature_columns(frame) == ["income_mainA"]


def test_data_contract_rejects_duplicate_model_grain():
    frame = valid_frame()
    frame.loc[1, "case_id"] = 1
    with pytest.raises(ValueError, match="unique"):
        validate_application_frame(frame)
