import numpy as np
import pytest

from compliant_control_lab.controllers import default_controllers
from compliant_control_lab.simulation import SimulationConfig, run_trial


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
