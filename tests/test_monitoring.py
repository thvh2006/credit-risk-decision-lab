from src.monitoring import monitoring_status


def test_monitoring_status_escalates_on_auc_drop():
    assert monitoring_status(0.75, 0.74, 0.001)["status"] == "pass"
    assert monitoring_status(0.75, 0.71, 0.001)["status"] == "warning"
    assert monitoring_status(0.75, 0.68, 0.001)["status"] == "breach"


def test_monitoring_status_escalates_on_calibration_gap():
    assert monitoring_status(0.75, 0.75, 0.03)["status"] == "warning"
    assert monitoring_status(0.75, 0.75, -0.06)["status"] == "breach"
