"""Independent real-trace checks for measured-force-only tangential additions."""

from dataclasses import replace

import numpy as np
import pytest

from compliant_control_lab import surface_simulation as sim
from compliant_control_lab.franka_control import FrankaActuationContext, FrankaState, FrankaTarget
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame
from compliant_control_lab.surface_replay import replay_surface_trace, save_surface_trace


@pytest.fixture(scope="module", params=["integral", "friction"])
def trial(request):
    return sim.run_surface_trial(
        sim.yaw_frame(15),
        sim.SurfaceScenario(wall_yaw_deg=15, wall_time_constant=0.005, delay_steps=2),
        sim.SurfaceSimulationConfig(duration=1.6, contact_model="smooth"),
        sim.SurfaceTask(yaw_deg=15),
        controller_kind=f"surface_{request.param}",
    )


def test_real_delayed_contact_and_wiping_trace_replays(trial, tmp_path):
    trace = trial.trace
    assert np.max(trace["true_normal_force"]) > 1
    assert np.max(np.linalg.norm(trace["requested_tangential_force_world"], axis=1)) > 1e-5
    assert np.any(trace["measured_kinematic_sample_time"] < trace["time"])
    replay = replay_surface_trace(save_surface_trace(tmp_path / "trace.npz", trace))
    assert replay.matches
    assert replay.sample_count == 800
    assert replay.controller_kind == str(trace["controller_kind"])
    assert replay.max_wrench_error <= 1e-10
    assert replay.max_torque_error <= 1e-10


def test_poisoned_live_evaluator_force_cannot_change_controller_or_replay(
    trial, monkeypatch, tmp_path
):
    monkeypatch.setattr(sim, "_normal_contact_force", lambda *_: 12345.0)
    poisoned = sim.run_surface_trial(
        sim.yaw_frame(15),
        trial.scenario,
        trial.config,
        trial.task,
        controller_kind=str(trial.trace["controller_kind"]),
    )
    np.testing.assert_array_equal(poisoned.trace["true_normal_force"], np.full(800, 12345.0))
    for name, original in trial.trace.items():
        if name != "true_normal_force":
            np.testing.assert_array_equal(poisoned.trace[name], original, err_msg=name)
    assert poisoned.metrics()["peak_force_n"] != trial.metrics()["peak_force_n"]
    assert replay_surface_trace(
        save_surface_trace(tmp_path / "poisoned.npz", poisoned.trace)
    ).matches


@pytest.mark.parametrize("mode", ["integral", "friction"])
def test_full_world_rotation_covariance_with_active_addition_and_projection(mode):
    frame = sim.yaw_frame(15)
    rotation = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    controller = SurfaceAdaptiveController(frame, tangential_mode=mode)
    rotated = SurfaceAdaptiveController(
        SurfaceFrame(rotation @ frame.rotation), tangential_mode=mode
    )
    state = FrankaState(
        np.zeros(3), np.eye(3), np.array([0.001, -0.01, 0.015]), np.array([0.01, 0.02, -0.01]), 12
    )
    target = FrankaTarget(
        frame.rotation @ np.array([0.01, 0.03, -0.04]),
        np.eye(3),
        frame.rotation @ np.array([0, 0.02, -0.01]),
        np.zeros(3),
        12,
    )
    rotated_target = replace(
        target,
        position=rotation @ target.position,
        rotation=rotation @ target.rotation,
        linear_velocity=rotation @ target.linear_velocity,
        angular_velocity=rotation @ target.angular_velocity,
    )
    saw_addition = saw_projection = False
    for step in range(40):
        limit = 100 if step < 20 else 1
        jacobian = np.column_stack((np.eye(6), np.zeros(6)))
        context = FrankaActuationContext(
            jacobian, np.zeros(7), np.full(7, -limit), np.full(7, limit)
        )
        current = replace(state, actuation=context)
        rotated_context = replace(
            context,
            cartesian_jacobian=np.concatenate(
                (rotation @ jacobian[:3], rotation @ jacobian[3:]),
                axis=0,
            ),
        )
        rotated_state = replace(
            current,
            position=rotation @ current.position,
            rotation=rotation @ current.rotation,
            linear_velocity=rotation @ current.linear_velocity,
            angular_velocity=rotation @ current.angular_velocity,
            actuation=rotated_context,
        )
        if step == 0:
            controller.reset(current)
            rotated.reset(rotated_state)
        wrench = controller.compute(current, target, 0.002)
        other = rotated.compute(rotated_state, rotated_target, 0.002)
        np.testing.assert_allclose(
            other.reshape(2, 3), wrench.reshape(2, 3) @ rotation.T, atol=2e-11
        )
        np.testing.assert_allclose(
            rotated_context.joint_torque(other), context.joint_torque(wrench), atol=2e-11
        )
        requested = controller.requested_tangential_force_world
        np.testing.assert_allclose(
            rotated.requested_tangential_force_world, rotation @ requested, atol=2e-11
        )
        assert abs(frame.rotation[:, 0] @ requested) < 1e-12
        assert np.max(np.abs(context.joint_torque(wrench))) <= 0.9 * limit + 1e-11
        saw_addition |= np.linalg.norm(requested) > 1e-5
        saw_projection |= controller.last_torque_projection_scale < 1
    assert saw_addition and saw_projection


def test_empty_mode_is_not_silently_treated_as_disabled():
    with pytest.raises(ValueError, match="mode"):
        SurfaceAdaptiveController(sim.yaw_frame(0), tangential_mode="")
