import numpy as np
import pytest

from compliant_control_lab.franka_control import default_franka_controllers
from compliant_control_lab.franka_simulation import (
    FrankaSimulationConfig,
    FrankaTrialResult,
    run_franka_trial,
)


@pytest.mark.parametrize("controller_name", list(default_franka_controllers()))
def test_each_franka_controller_runs_in_mujoco(controller_name):
    result = run_franka_trial(
        default_franka_controllers()[controller_name],
        config=FrankaSimulationConfig(duration=0.06),
    )
    assert len(result.time) == 30
    assert result.q.shape == (30, 7)
    assert np.all(np.isfinite(result.position))
    assert np.all(np.isfinite(result.torque))


def test_franka_hybrid_controller_tracks_contact_force():
    result = run_franka_trial(
        default_franka_controllers()["hybrid"],
        config=FrankaSimulationConfig(duration=2.2),
    )
    metrics = result.metrics()
    assert metrics["force_rmse_n"] < 2.0
    assert metrics["contact_ratio_pct"] > 95.0
    assert metrics["saturation_pct"] < 1.0


def test_safety_metrics_include_the_approach_window():
    time = np.array([0.0, 1.0, 2.0])
    result = FrankaTrialResult(
        controller="test",
        scenario="test",
        time=time,
        q=np.zeros((3, 7)),
        position=np.zeros((3, 3)),
        desired_position=np.zeros((3, 3)),
        normal_force=np.array([0.0, 0.0, 12.0]),
        raw_normal_force=np.array([50.0, 0.0, 12.0]),
        desired_force=np.array([0.0, 0.0, 12.0]),
        orientation_error_rad=np.zeros(3),
        torque=np.zeros((3, 7)),
        controller_time_us=np.ones(3),
        saturated=np.array([True, False, False]),
    )
    metrics = result.metrics(evaluation_start=1.5)
    assert metrics["force_rmse_n"] == 0.0
    assert metrics["peak_force_n"] == 50.0
    assert metrics["saturation_pct"] == pytest.approx(100.0 / 3.0)
