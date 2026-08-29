from dataclasses import asdict

import pytest

from compliant_control_lab.franka_stress import (
    ResidualRlGate,
    failed_gate_checks,
    sample_stress_scenarios,
)


def test_stress_sampling_is_reproducible_and_bounded():
    first = sample_stress_scenarios(count=4, seed=8)
    second = sample_stress_scenarios(count=4, seed=8)
    assert [asdict(case) for case in first] == [asdict(case) for case in second]
    for case in first:
        assert 0.004 <= case.wall_time_constant <= 0.025
        assert 0.20 <= case.wall_sliding_friction <= 0.90
        assert -6.0 <= case.wall_yaw_deg <= 6.0
        assert 0 <= case.delay_steps <= 15
        assert 0.85 <= case.bias_compensation_scale <= 1.15


def test_residual_rl_gate_reports_exact_failures():
    metrics = {
        "force_rmse_n": 2.1,
        "contact_ratio_pct": 96.0,
        "peak_force_n": 34.0,
        "tangent_rmse_mm": 16.0,
        "saturation_pct": 0.0,
    }
    assert failed_gate_checks(metrics, ResidualRlGate()) == (
        "force_rmse",
        "tangent_rmse",
    )


def test_stress_sampling_rejects_nonpositive_case_count():
    with pytest.raises(ValueError, match="count must be positive"):
        sample_stress_scenarios(count=0)
