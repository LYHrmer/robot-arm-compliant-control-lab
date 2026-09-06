"""Check the newly integrated state even when no next control action will run."""

import mujoco
import numpy as np
import pytest

from compliant_control_lab.surface_env import SurfaceLearningEnv
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    yaw_frame,
)


@pytest.mark.parametrize("duration", (0.002, 0.004))
@pytest.mark.parametrize("delay", (0, 3))
def test_integrated_speed_violation_cannot_hide_at_horizon(monkeypatch, duration, delay):
    original = mujoco.mj_step2

    def inject_after_integration(model, data):
        original(model, data)
        jacobian, rotation = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian, rotation, model.site("ee_site").id)
        data.qvel[:7] = np.linalg.pinv(jacobian[:, :7]) @ [0.5, 0, 0]

    monkeypatch.setattr(mujoco, "mj_step2", inject_after_integration)
    env = SurfaceLearningEnv(
        yaw_frame(0),
        scenario=SurfaceScenario(delay_steps=delay),
        config=SurfaceSimulationConfig(duration=duration, contact_model="smooth"),
    )
    try:
        env.reset(seed=11)
        _, reward, terminated, truncated, info = env.step(np.zeros(3))
        assert terminated and not truncated, "the final integrated state escaped the safety gate"
        assert "speed_limit" in info["termination_reasons"]
        assert info["physics_substeps"] == 1 and reward <= -1
        trace = env.result().trace
        assert np.linalg.norm(trace["linear_velocity"][0]) < 0.3
        assert np.linalg.norm(trace["endpoint_linear_velocity"][0]) > 0.3
        assert trace["endpoint_time"][0] == pytest.approx(0.002)
        assert trace["endpoint_valid"][0]
    finally:
        env.close()


def test_final_refreshed_geometry_violation_is_not_a_successful_truncation(monkeypatch):
    original = mujoco.mj_step1

    def inject_refreshed_geometry(model, data):
        original(model, data)
        if data.time >= 0.002:
            data.site_xpos[model.site("ee_site").id, 0] = 0.378

    monkeypatch.setattr(mujoco, "mj_step1", inject_refreshed_geometry)
    env = SurfaceLearningEnv(
        yaw_frame(0),
        scenario=SurfaceScenario(delay_steps=3),
        config=SurfaceSimulationConfig(duration=0.002, contact_model="smooth"),
    )
    try:
        env.reset(seed=11)
        _, reward, terminated, truncated, info = env.step(np.zeros(3))
        assert terminated and not truncated and reward <= -1
        assert "penetration_limit" in info["termination_reasons"]
        trace = env.result().trace
        assert trace["true_contact_gap_m"][0] > 0
        assert trace["endpoint_contact_gap_m"][0] == pytest.approx(-0.003)
    finally:
        env.close()
