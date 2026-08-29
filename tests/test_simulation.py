import numpy as np
import pytest

from compliant_control_lab.controllers import default_controllers
from compliant_control_lab.simulation import SimulationConfig, TrialResult, run_trial


@pytest.mark.parametrize("controller_name", list(default_controllers()))
def test_each_controller_runs_in_mujoco(controller_name):
    result = run_trial(
        default_controllers()[controller_name],
        config=SimulationConfig(duration=0.08),
    )
    assert len(result.time) == 40
    assert np.all(np.isfinite(result.position))
    assert np.all(np.isfinite(result.torque))


def test_hybrid_controller_tracks_contact_force():
    result = run_trial(
        default_controllers()["hybrid"],
        config=SimulationConfig(duration=2.0),
    )
    metrics = result.metrics()
    assert metrics["force_rmse_n"] < 3.0
    assert metrics["contact_ratio_pct"] > 95.0


def test_planar_safety_metrics_include_the_approach_window():
    result = TrialResult(
        controller="test",
        scenario="test",
        time=np.array([0.0, 1.0, 2.0]),
        q=np.zeros((3, 2)),
        position=np.zeros((3, 2)),
        desired_position=np.zeros((3, 2)),
        normal_force=np.array([0.0, 0.0, 12.0]),
        raw_normal_force=np.array([40.0, 0.0, 12.0]),
        desired_force=np.array([0.0, 0.0, 12.0]),
        torque=np.zeros((3, 2)),
        controller_time_us=np.ones(3),
        saturated=np.array([True, False, False]),
    )
    metrics = result.metrics(evaluation_start=1.5)
    assert metrics["force_rmse_n"] == 0.0
    assert metrics["peak_force_n"] == 40.0
    assert metrics["saturation_pct"] == pytest.approx(100.0 / 3.0)
