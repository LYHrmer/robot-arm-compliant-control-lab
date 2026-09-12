"""The top-level safety label must cover every gate it advertises."""

import pytest

from tests.test_measured_budget_robustness import _metric_payload
from tools import measured_budget_robustness as study


@pytest.mark.parametrize(
    "field,value", [("projection_pct", 0.1), ("minimum_reserved_torque_headroom_nm", 0.9)]
)
def test_summary_cannot_hide_projection_or_headroom_failure(field, value):
    overall, phases, _ = _metric_payload()
    overall[field] = value
    result = study._safety_summary(
        [
            {
                "scenario_id": "unit_case",
                "method": "adaptive6_8",
                "overall": overall,
                "phases": phases,
            }
        ]
    )
    assert result["observed_status"] == "FAIL"
    assert result["failed_checks"]
