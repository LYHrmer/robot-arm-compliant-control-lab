"""Trajectory and causal sampling contracts, independent of benchmark scores."""

import mujoco
import numpy as np
import pytest

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.franka_simulation import (
    FrankaScenario,
    FrankaSimulationConfig,
    _normal_contact_force,
    _site_state,
    _target_at,
    franka_model_path,
    run_franka_trial,
)


@pytest.mark.parametrize("time", [0.0, 0.15, 0.55, 0.93, 1.05, 1.19, 1.21, 2.0])
def test_consistent_approach_velocity_is_the_position_derivative(time):
    config = FrankaSimulationConfig(approach_reference="consistent")
    position = np.array([0.30, -0.02, 0.45])
    epsilon = 1.0e-6
    target = _target_at(time, position, np.eye(3), config)
    before = _target_at(time - epsilon, position, np.eye(3), config)
    after = _target_at(time + epsilon, position, np.eye(3), config)
    assert np.allclose(
        target.linear_velocity,
        (after.position - before.position) / (2 * epsilon),
        atol=1.0e-9,
        rtol=0,
    )
    legacy = _target_at(time, position, np.eye(3), FrankaSimulationConfig())
    assert np.array_equal(target.position, legacy.position)
    assert target.normal_force == legacy.normal_force
    assert legacy.linear_velocity[0] == 0.0


@pytest.mark.parametrize("time", [0.10, 1.0])
def test_approach_velocity_is_continuous_at_its_endpoints(time):
    config = FrankaSimulationConfig(approach_reference="consistent")
    for offset in [-1.0e-8, 0.0, 1.0e-8]:
        target = _target_at(time + offset, np.array([0.30, 0, 0.45]), np.eye(3), config)
        assert np.max(np.abs(target.linear_velocity)) < 1.0e-8


@pytest.mark.parametrize(
    "options",
    [
        {"control_timing": "split"},
        {"approach_reference": "smooth"},
    ],
)
def test_unknown_simulation_modes_are_rejected(options):
    with pytest.raises(ValueError):
        FrankaSimulationConfig(**options)


def test_split_step_kinematics_and_actuation_use_the_logged_joint_state():
    class RecordingController(FrankaSafeAdaptiveController):
        def __init__(self):
            super().__init__()
            self.contexts = []

        def compute(self, state, target, dt):
            self.contexts.append(state.actuation)
            return super().compute(state, target, dt)

    controller = RecordingController()
    result = run_franka_trial(
        controller,
        config=FrankaSimulationConfig(duration=0.06, control_timing="split_step"),
    )
    model = mujoco.MjModel.from_xml_path(str(franka_model_path()))
    data = mujoco.MjData(model)
    site_id = model.site("ee_site").id
    for index, context in enumerate(controller.contexts):
        data.qpos[:7] = result.q[index]
        data.qvel[:7] = result.joint_velocity[index]
        mujoco.mj_forward(model, data)
        position, _, velocity, _, jacobian = _site_state(model, data, site_id)
        np.testing.assert_allclose(result.position[index], position, atol=1e-13, rtol=0)
        np.testing.assert_allclose(result.linear_velocity[index], velocity, atol=1e-13, rtol=0)
        np.testing.assert_allclose(context.cartesian_jacobian, jacobian, atol=1e-13, rtol=0)
    np.testing.assert_array_equal(result.kinematic_sample_time, result.time)
    np.testing.assert_array_equal(result.raw_force_sample_time, result.time)


def test_split_step_force_feedback_is_causal_and_scenario_delay_is_explicit():
    config = FrankaSimulationConfig(duration=1.8, control_timing="split_step")
    delay_steps = 3
    result = run_franka_trial(
        FrankaSafeAdaptiveController(),
        FrankaScenario(name="causality", force_noise_std=0, delay_steps=delay_steps),
        config,
    )
    assert np.max(result.raw_normal_force) > 1.0  # Exercise contact, not just zeros.
    alpha = config.timestep / (config.force_filter_time_constant + config.timestep)
    expected_feedback = np.zeros_like(result.time)
    for index in range(1, len(result.time)):
        expected_feedback[index] = expected_feedback[index - 1] + alpha * (
            result.raw_normal_force[index - 1] - expected_feedback[index - 1]
        )
    np.testing.assert_array_equal(result.normal_force, expected_feedback)
    delayed_indices = np.maximum(0, np.arange(len(result.time)) - delay_steps)
    force_indices = np.maximum(0, delayed_indices - 1)
    np.testing.assert_array_equal(result.measured_normal_force, expected_feedback[delayed_indices])
    np.testing.assert_array_equal(
        result.measured_kinematic_sample_time, result.time[delayed_indices]
    )
    np.testing.assert_array_equal(result.measured_force_sample_time, result.time[force_indices])
    np.testing.assert_array_equal(
        result.feedback_force_sample_time,
        result.time[np.maximum(0, np.arange(len(result.time)) - 1)],
    )


def test_step2_raw_force_belongs_to_preintegration_state_and_current_control(monkeypatch):
    original_step2 = mujoco.mj_step2
    expected_forces = []

    def checked_step2(model, data):
        scratch = mujoco.MjData(model)
        mujoco.mj_copyData(scratch, model, data)
        mujoco.mj_forward(model, scratch)
        expected_forces.append(
            _normal_contact_force(
                model,
                scratch,
                model.geom("tool_tip").id,
                model.geom("contact_wall").id,
            )
        )
        original_step2(model, data)

    monkeypatch.setattr(mujoco, "mj_step2", checked_step2)
    result = run_franka_trial(
        FrankaSafeAdaptiveController(),
        config=FrankaSimulationConfig(
            duration=1.5,
            control_timing="split_step",
        ),
    )
    assert np.max(expected_forces) > 1.0
    np.testing.assert_allclose(result.raw_normal_force, expected_forces, atol=1e-8, rtol=0)


def test_split_step_rejects_rk4_instead_of_silently_switching_integrator(monkeypatch):
    original = mujoco.MjModel.from_xml_path

    def rk4_model(path):
        model = original(path)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        return model

    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", rk4_model)
    with pytest.raises(ValueError, match="RK4"):
        run_franka_trial(
            FrankaSafeAdaptiveController(),
            config=FrankaSimulationConfig(control_timing="split_step"),
        )


def test_legacy_default_and_explicit_modes_are_identical():
    default = run_franka_trial(
        FrankaSafeAdaptiveController(),
        config=FrankaSimulationConfig(duration=0.08),
    )
    explicit = run_franka_trial(
        FrankaSafeAdaptiveController(),
        config=FrankaSimulationConfig(
            duration=0.08,
            control_timing="legacy",
            approach_reference="legacy",
        ),
    )
    for name in ["q", "position", "normal_force", "raw_normal_force", "torque"]:
        np.testing.assert_array_equal(getattr(default, name), getattr(explicit, name))
    assert default.control_timing == "legacy"
    np.testing.assert_array_equal(default.kinematic_sample_time[1:], default.time[:-1])


def test_split_step_aborts_if_simulator_resets_instead_of_publishing_false_timestamps(monkeypatch):
    original_step2 = mujoco.mj_step2

    def reset_time(model, data):
        original_step2(model, data)
        data.time = 0.0

    monkeypatch.setattr(mujoco, "mj_step2", reset_time)
    with pytest.raises(RuntimeError, match="timestamps"):
        run_franka_trial(
            FrankaSafeAdaptiveController(),
            config=FrankaSimulationConfig(duration=0.01, control_timing="split_step"),
        )
