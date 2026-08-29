import numpy as np
import pytest

from compliant_control_lab.franka_control import default_franka_controllers
from compliant_control_lab.franka_simulation import (
    FrankaSimulationConfig,
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

