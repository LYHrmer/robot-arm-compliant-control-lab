import numpy as np

from compliant_control_lab.franka_simulation import FrankaTrialResult


def test_metrics_do_not_require_event_telemetry_for_legacy_results() -> None:
    result = FrankaTrialResult(
        controller="legacy",
        scenario="synthetic",
        time=np.array([0.0, 2.0]),
        q=np.zeros((2, 7)),
        position=np.zeros((2, 3)),
        desired_position=np.zeros((2, 3)),
        normal_force=np.array([0.0, 12.0]),
        raw_normal_force=np.array([0.0, 12.0]),
        desired_force=np.array([0.0, 12.0]),
        orientation_error_rad=np.zeros(2),
        torque=np.zeros((2, 7)),
        controller_time_us=np.ones(2),
        saturated=np.zeros(2, dtype=bool),
    )

    assert result.metrics()["force_rmse_n"] == 0.0
    assert result.linear_velocity.shape == (0, 3)
