"""The surface residual Interface is exercised without simulator or learning dependencies."""

from dataclasses import replace

import numpy as np
import pytest

from compliant_control_lab import surface_policy as policy
from compliant_control_lab.franka_control import FrankaActuationContext, FrankaState, FrankaTarget
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame


def frame():
    normal = np.array([0.8, 0.4, 0.3])
    return SurfaceFrame.from_normal(normal)


def inputs(surface=None, force=12.0, limit=87.0, offset=None):
    surface = surface or SurfaceFrame(np.eye(3))
    rotation = surface.rotation
    local_jacobian = np.eye(6, 7)
    jacobian = np.concatenate((rotation @ local_jacobian[:3], rotation @ local_jacobian[3:]))
    context = FrankaActuationContext(
        jacobian, np.zeros(7) if offset is None else offset, np.full(7, -limit), np.full(7, limit)
    )
    state = FrankaState(
        rotation @ [0.3, 0.0, 0.5], rotation.copy(), np.zeros(3), np.zeros(3), force, context
    )
    target = FrankaTarget(
        state.position + rotation @ [0.01, 0.004, -0.003],
        rotation.copy(),
        rotation @ [0.0, 0.05, -0.03],
        np.zeros(3),
        12.0,
    )
    return state, target


class Nominal:
    """Local test Adapter with a known feasible wrench and observable call count."""

    name = "test_nominal"
    contact_blend = 1.0
    corrected_force_n = 12.0
    filtered_force_rate_n_s = 0.0

    def __init__(self, frame, **kwargs):
        self.frame = frame
        self.calls = 0

    def reset(self, state):
        self.calls = 0

    def compute(self, state, target, dt):
        self.calls += 1
        self.corrected_force_n = state.normal_force
        return self.frame.wrench_to_world(np.array([12.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr(policy, "SurfaceAdaptiveController", Nominal)
    return policy.SurfaceResidualController(SurfaceFrame(np.eye(3)))


def run_steps(controller, state, target, action, count=100):
    output = []
    for _ in range(count):
        observation = controller.prepare(state, target, 0.002)
        wrench = controller.apply(action)
        output.append((observation, wrench, controller.last_command))
    return output


def test_two_phase_order_and_dt_reject_before_mutation(controller):
    state, target = inputs()
    with pytest.raises(RuntimeError, match="reset"):
        controller.prepare(state, target, 0.002)
    controller.reset(state)
    for dt in (0.0, -0.002, np.nan, np.inf):
        with pytest.raises(ValueError, match="dt"):
            controller.prepare(state, target, dt)
    assert controller.nominal.calls == 0
    with pytest.raises(RuntimeError, match="prepare"):
        controller.apply(np.zeros(3))
    controller.prepare(state, target, 0.002)
    assert controller.nominal.calls == 1
    with pytest.raises(RuntimeError, match="consume"):
        controller.prepare(state, target, 0.002)
    controller.apply(np.zeros(3))
    assert controller.nominal.calls == 1
    with pytest.raises(RuntimeError, match="prepare"):
        controller.apply(np.zeros(3))


@pytest.mark.parametrize("kind", ["adaptive", "friction"])
def test_zero_action_bitwise_matches_original_nominal(kind):
    surface = frame()
    state, target = inputs(surface)
    controller = policy.SurfaceResidualController(surface, nominal_kind=kind)
    nominal = SurfaceAdaptiveController(
        surface, tangential_mode="friction" if kind == "friction" else None
    )
    controller.reset(state)
    nominal.reset(state)
    for force in [0.0] * 10 + [12.0] * 350 + [20.0] * 5 + [0.0] * 10:
        current = replace(state, normal_force=force)
        controller.prepare(current, target, 0.002)
        np.testing.assert_array_equal(
            controller.apply(np.zeros(3)), nominal.compute(current, target, 0.002)
        )
        np.testing.assert_array_equal(controller.last_command.applied_force, np.zeros(3))


def test_surface_schema_scaling_copy_and_input_immutability(controller):
    state, target = inputs()
    saved_state, saved_target = state.position.copy(), target.position.copy()
    controller.reset(state)
    observation = controller.prepare(state, target, 0.002)
    assert policy.OBSERVATION_SCHEMA == "surface_v1"
    assert len(policy.OBSERVATION_NAMES) == len(set(policy.OBSERVATION_NAMES)) == 33
    assert len(policy.OBSERVATION_SCALES) == len(policy.OBSERVATION_UNITS) == 33
    assert observation.dtype == np.float64 and np.all(np.isfinite(observation))
    physical = observation * np.asarray(policy.OBSERVATION_SCALES)
    np.testing.assert_allclose(physical[:3], [12.0, 0.0, 0.0])
    np.testing.assert_allclose(physical[3:6], target.position - state.position)
    np.testing.assert_allclose(physical[9:12], target.linear_velocity)
    np.testing.assert_allclose(physical[17:23], 1.0)
    observation[:] = 100
    controller.apply(np.zeros(3))
    np.testing.assert_array_equal(state.position, saved_state)
    np.testing.assert_array_equal(target.position, saved_target)
    snapshot = controller.last_command
    snapshot.nominal_wrench[:] = -999
    snapshot.applied_force[:] = 999
    assert controller.last_command.nominal_wrench[0] == 12.0
    np.testing.assert_array_equal(controller.last_command.applied_force, 0.0)


def test_contact_confirmation_and_immediate_loss_clearing(controller):
    state, target = inputs()
    controller.reset(state)
    output = run_steps(controller, state, target, np.ones(3), 49)
    assert all(np.all(row[2].applied_force == 0) for row in output)
    controller.prepare(state, target, 0.002)
    controller.apply(np.ones(3))
    assert np.all(controller.last_command.applied_force > 0)
    previous_applied = controller.last_command.applied_force
    lost = replace(state, normal_force=1.0)
    observation = controller.prepare(lost, target, 0.002)
    np.testing.assert_allclose(observation[14:17] * [4.0, 6.0, 6.0], previous_applied)
    controller.apply(np.ones(3))
    assert "contact_lost" in controller.last_command.reasons
    np.testing.assert_array_equal(controller.last_command.applied_force, 0.0)
    run_steps(controller, state, target, np.ones(3), 49)
    np.testing.assert_array_equal(controller.last_command.applied_force, 0.0)


def test_clip_filter_slew_and_force_guard_in_local_normal(controller):
    state, target = inputs()
    controller.reset(state)
    output = run_steps(controller, state, target, [8.0, -9.0, 3.0], 200)
    applied = np.array([row[2].applied_force for row in output])
    assert np.all(np.abs(applied) <= [4.0, 6.0, 6.0])
    assert np.all(np.abs(np.diff(applied, axis=0)) <= np.array([40.0, 60.0, 60.0]) * 0.002 + 1e-12)
    assert "action_clipped" in controller.last_command.reasons
    controller.nominal.filtered_force_rate_n_s = 151.0
    controller.prepare(state, target, 0.002)
    controller.apply(np.ones(3))
    assert controller.last_command.applied_force[0] == 0
    assert np.any(controller.last_command.applied_force[1:] != 0)
    assert "force_guard" in controller.last_command.reasons
    controller.prepare(state, target, 0.002)
    controller.apply([-1.0, 0.0, 0.0])
    assert controller.last_command.applied_force[0] < 0


@pytest.mark.parametrize("bad", [[1.0, 2.0], [np.nan, 0, 0], [0, np.inf, 0], "invalid"])
def test_invalid_action_falls_back_immediately_and_consumes_pending(controller, bad):
    state, target = inputs()
    controller.reset(state)
    run_steps(controller, state, target, np.ones(3))
    assert np.linalg.norm(controller.last_command.applied_force) > 0
    controller.prepare(state, target, 0.002)
    wrench = controller.apply(bad)
    assert "invalid_action" in controller.last_command.reasons
    np.testing.assert_array_equal(controller.last_command.applied_force, 0.0)
    np.testing.assert_array_equal(wrench, controller.last_command.nominal_wrench)
    with pytest.raises(RuntimeError):
        controller.apply(np.zeros(3))


def test_missing_infeasible_and_changed_actuation_context(controller):
    state, target = inputs()
    controller.reset(state)
    run_steps(controller, state, target, np.ones(3))
    missing = replace(state, actuation=None)
    obs = controller.prepare(missing, target, 0.002)
    np.testing.assert_array_equal(obs[17:23], 0.0)
    controller.apply(np.ones(3))
    assert "torque_missing_context" in controller.last_command.reasons
    np.testing.assert_array_equal(controller.last_command.applied_force, 0.0)
    infeasible, _ = inputs(offset=np.full(7, 100.0))
    controller.prepare(infeasible, target, 0.002)
    controller.apply(np.ones(3))
    assert "torque_nominal_outside" in controller.last_command.reasons
    np.testing.assert_array_equal(controller.last_command.applied_force, 0.0)
    # Restore then tighten only the current torque limits: held action is reprojected now.
    run_steps(controller, state, target, np.ones(3))
    tight, _ = inputs(limit=13.5)
    controller.prepare(tight, target, 0.002)
    wrench = controller.apply(np.ones(3))
    assert controller.last_command.projection_scale < 1
    assert np.max(np.abs(tight.actuation.joint_torque(wrench))) <= 13.5 * 0.9 + 1e-12


def test_total_normal_interval_and_overshoot_guard(monkeypatch):
    monkeypatch.setattr(policy, "SurfaceAdaptiveController", Nominal)
    config = policy.SurfaceResidualConfig(
        min_total_normal_wrench=11.9, max_total_normal_wrench=12.1
    )
    controller = policy.SurfaceResidualController(SurfaceFrame(np.eye(3)), config=config)
    state, target = inputs()
    controller.reset(state)
    for action in ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]):
        output = run_steps(controller, state, target, action, 100)
        assert all(11.9 <= row[1][0] <= 12.1 for row in output)
        assert "normal_wrench_limit" in controller.last_command.reasons
    controller.prepare(replace(state, normal_force=16.0), target, 0.002)
    controller.apply(np.ones(3))
    assert "force_guard" in controller.last_command.reasons
    assert controller.last_command.applied_force[0] <= 0


def test_world_rotation_covariance_and_reset_sequences(monkeypatch):
    monkeypatch.setattr(policy, "SurfaceAdaptiveController", Nominal)
    surfaces = [SurfaceFrame(np.eye(3)), frame()]
    outputs = []
    for surface in surfaces:
        state, target = inputs(surface)
        controller = policy.SurfaceResidualController(surface)
        controller.reset(state)
        first = run_steps(controller, state, target, [0.4, -0.8, 0.3])
        controller.prepare(state, target, 0.002)
        controller.reset(state)  # Reset may cancel a pending sample.
        assert controller.last_command is None
        repeated = run_steps(controller, state, target, [0.4, -0.8, 0.3])
        for left, right in zip(first, repeated):
            np.testing.assert_array_equal(left[0], right[0])
            np.testing.assert_array_equal(left[1], right[1])
        outputs.append(first)
    for plain, rotated in zip(*outputs):
        np.testing.assert_allclose(plain[0], rotated[0], rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            surfaces[1].wrench_to_world(plain[1]), rotated[1], rtol=0, atol=1e-13
        )


def test_teacher_uses_only_corrected_force_and_explicit_target_velocity():
    observation = np.zeros(33)
    observation[0] = 12.0 / policy.OBSERVATION_SCALES[0]
    observation[10] = 0.05 / policy.OBSERVATION_SCALES[10]
    action = policy.friction_teacher_action(observation)
    expected = 0.45 * 12.0 * 0.05 / np.sqrt(0.05**2 + 0.005**2) / 6.0
    np.testing.assert_allclose(action, [0.0, expected, 0.0])
    altered = np.full(33, 2.5)
    altered[[0, 9, 10, 11]] = observation[[0, 9, 10, 11]]
    np.testing.assert_array_equal(policy.friction_teacher_action(altered), action)
    observation[0] = 3.0
    assert np.linalg.norm(policy.friction_teacher_action(observation) * [4, 6, 6]) <= 6.0
    for bad in (np.zeros(20), np.full(33, np.nan)):
        with pytest.raises(ValueError, match="surface_v1"):
            policy.friction_teacher_action(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_bounds_n": [4, 0, 6]},
        {"action_rate_limits_n_s": [1, 2]},
        {"filter_time_constant": 0},
        {"torque_reserve_fraction": 1},
        {"residual_enable_delay": -0.1},
        {"force_guard_rate": np.inf},
        {"teacher_nominal_mu": -0.1},
        {"min_total_normal_wrench": 25.0},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        policy.SurfaceResidualConfig(**kwargs)
