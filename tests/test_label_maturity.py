import pandas as pd
import pytest

from src.label_maturity import maturity_diagnostic


def test_label_maturity_diagnostic_refuses_to_rule_out_censoring():
    weekly = pd.DataFrame(
        {
            "WEEK_NUM": [0, 1, 78, 79],
            "applications": [100, 100, 50, 50],
            "defaults": [5, 5, 1, 1],
            "default_rate": [0.05, 0.05, 0.02, 0.02],
        }
    )
    result = maturity_diagnostic(weekly, reference_last_week=1, late_first_week=78)
    assert result["relative_default_rate_change"] == pytest.approx(-0.6)
    assert result["relative_volume_change"] == -0.5
    assert result["right_censoring_can_be_ruled_out"] is False
