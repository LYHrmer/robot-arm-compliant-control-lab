from dataclasses import replace

import numpy as np
import pytest

from compliant_control_lab.franka_adaptive import FrankaAdaptiveHybridController
from compliant_control_lab.franka_control import (
    FrankaActuationContext,
    FrankaHybridController,
    FrankaState,
    FrankaTarget,
    capture_franka_controller_telemetry,
)
from compliant_control_lab.franka_reference import FrankaRateLimitedAdaptiveController


def _state() -> FrankaState:
    return FrankaState(
        position=np.array([0.36, 0.0, 0.45]),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        normal_force=0.0,
    )


def _target() -> FrankaTarget:
    return FrankaTarget(
        position=np.array([0.46, 0.04, 0.42]),
        rotation=np.eye(3),
        linear_velocity=np.array([1.0, 0.01, -0.02]),
        angular_velocity=np.array([0.01, -0.02, 0.03]),
        normal_force=12.0,
    )


def _capture_targets(controller: FrankaRateLimitedAdaptiveController) -> list[FrankaTarget]:
    captured: list[FrankaTarget] = []
    compute = controller.base.compute

    def capture(state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray:
        captured.append(target)
        return compute(state, target, dt)

    controller.base.compute = capture  # type: ignore[method-assign]
    return captured


def test_step_reference_converges_with_consistent_bounded_derivatives() -> None:
    state, target = _state(), _target()
    controller = FrankaRateLimitedAdaptiveController(max_normal_lead=0.20)
    captured = _capture_targets(controller)
    controller.reset(state)
    dt = 0.005
    positions = [state.position[0]]
    velocities = [0.0]

    for _ in range(1_200):
        controller.compute(state, target, dt)
        positions.append(captured[-1].position[0])
        velocities.append(captured[-1].linear_velocity[0])

    np.testing.assert_allclose(np.diff(positions), dt * np.asarray(velocities[1:]), atol=1e-15)
    assert np.max(np.abs(velocities)) <= controller.max_approach_velocity + 1e-14
    assert np.max(np.abs(np.diff(velocities) / dt)) <= controller.max_approach_acceleration + 1e-12
    assert np.min(np.diff(positions)) >= -1e-15
    assert positions[-1] == pytest.approx(target.position[0], abs=1e-10)
    assert velocities[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.max(positions) <= target.position[0] + 1e-14


def test_measured_forward_overspeed_requests_continuous_braking() -> None:
    state, target = _state(), _target()
    controller = FrankaRateLimitedAdaptiveController(max_normal_lead=0.20)
    dt = 0.01
    for _ in range(30):
        controller.compute(state, target, dt)
    previous_velocity = controller.reference_normal_velocity_m_s
    assert previous_velocity == pytest.approx(controller.max_approach_velocity)
    fast_state = replace(state, linear_velocity=np.array([0.10, 0.0, 0.0]))

    for _ in range(30):
        previous_position = controller.reference_normal_position_m
        controller.compute(fast_state, target, dt)
        velocity = controller.reference_normal_velocity_m_s
        assert velocity <= previous_velocity + 1e-14
        assert previous_velocity - velocity <= controller.max_approach_acceleration * dt + 1e-14
        assert controller.reference_normal_position_m - previous_position == pytest.approx(
            dt * velocity
        )
        previous_velocity = velocity

    assert controller.reference_normal_velocity_m_s == pytest.approx(0.0)
    controller.reset(fast_state)
    controller.compute(fast_state, target, dt)
    assert controller.reference_normal_velocity_m_s == 0.0


def test_retreat_and_changed_lead_goal_preserve_reference_continuity() -> None:
    state, target = _state(), _target()
    controller = FrankaRateLimitedAdaptiveController(max_normal_lead=0.002)
    dt = 0.002
    for _ in range(40):
        controller.compute(state, target, dt)
    retreat_state = replace(state, position=state.position - np.array([0.04, 0.0, 0.0]))
    retreat_target = replace(target, position=state.position - np.array([0.02, 0.0, 0.0]))
    initial_velocity = controller.reference_normal_velocity_m_s
    positions = [controller.reference_normal_position_m]
    velocities = [initial_velocity]

    for _ in range(300):
        controller.compute(retreat_state, retreat_target, dt)
        positions.append(controller.reference_normal_position_m)
        velocities.append(controller.reference_normal_velocity_m_s)

    np.testing.assert_allclose(np.diff(positions), dt * np.asarray(velocities[1:]), atol=1e-15)
    assert np.max(np.abs(np.diff(velocities) / dt)) <= controller.max_approach_acceleration + 1e-12
    assert velocities[1] > 0.0  # A reversal brakes first; position is not snapped backward.
    assert velocities[-1] < 0.0
    assert positions[1] - retreat_state.position[0] > controller.max_normal_lead


def test_prior_cycle_impact_guard_brakes_without_snapping_reference() -> None:
    state, target = _state(), _target()
    controller = FrankaRateLimitedAdaptiveController(max_normal_lead=0.20)
    dt = 0.01
    for _ in range(30):
        controller.compute(state, target, dt)
    impact_state = replace(state, normal_force=40.0)
    controller.compute(impact_state, target, dt)
    assert controller.corrected_force_n > target.normal_force + controller.impact_force_margin
    previous_position = controller.reference_normal_position_m
    previous_velocity = controller.reference_normal_velocity_m_s

    controller.compute(impact_state, target, dt)

    assert controller.reference_normal_velocity_m_s == pytest.approx(
        previous_velocity - controller.max_approach_acceleration * dt
    )
    assert controller.reference_normal_position_m - previous_position == pytest.approx(
        dt * controller.reference_normal_velocity_m_s
    )
    assert controller.last_governed_normal_lead_m > 0.0


def test_rotated_normal_preserves_tangential_and_other_target_fields_without_mutation() -> None:
    normal = np.array([1.0, 2.0, -1.0])
    direction = normal / np.linalg.norm(normal)
    controller = FrankaRateLimitedAdaptiveController(
        base=FrankaAdaptiveHybridController(base=FrankaHybridController(normal=normal)),
    )
    state, target = _state(), _target()
    original_state = {
        name: value.copy() for name, value in vars(state).items() if isinstance(value, np.ndarray)
    }
    original_target = {
        name: value.copy() for name, value in vars(target).items() if isinstance(value, np.ndarray)
    }
    captured = _capture_targets(controller)
    controller.compute(state, target, dt=0.002)
    governed = captured[-1]
    tangent = np.eye(3) - np.outer(direction, direction)

    np.testing.assert_allclose(tangent @ governed.position, tangent @ target.position, atol=1e-15)
    np.testing.assert_allclose(
        tangent @ governed.linear_velocity, tangent @ target.linear_velocity, atol=1e-15
    )
    assert direction @ governed.linear_velocity == pytest.approx(
        controller.reference_normal_velocity_m_s
    )
    assert direction @ governed.position == pytest.approx(controller.reference_normal_position_m)
    np.testing.assert_array_equal(governed.rotation, target.rotation)
    np.testing.assert_array_equal(governed.angular_velocity, target.angular_velocity)
    assert governed.normal_force == target.normal_force
    for name, value in original_state.items():
        np.testing.assert_array_equal(getattr(state, name), value)
    for name, value in original_target.items():
        np.testing.assert_array_equal(getattr(target, name), value)
    np.testing.assert_array_equal(normal, np.array([1.0, 2.0, -1.0]))


def test_reset_matches_fresh_controller_and_retains_torque_projection_telemetry() -> None:
    context = FrankaActuationContext(
        cartesian_jacobian=np.eye(6),
        joint_torque_offset=np.zeros(6),
        lower_torque_limit=np.full(6, -0.2),
        upper_torque_limit=np.full(6, 0.2),
    )
    state = replace(_state(), actuation=context)
    target = _target()
    used = FrankaRateLimitedAdaptiveController()
    fresh = FrankaRateLimitedAdaptiveController()
    assert fresh.reference_normal_position_m is None
    for _ in range(30):
        used.compute(state, target, dt=0.002)
    assert used.torque_projection_pct > 0.0

    used.reset(state)
    fresh.reset(state)
    assert used.reference_normal_position_m == fresh.reference_normal_position_m
    assert used.reference_normal_velocity_m_s == fresh.reference_normal_velocity_m_s == 0.0
    assert capture_franka_controller_telemetry(used) == capture_franka_controller_telemetry(fresh)
    assert used.torque_projection_pct == 0.0
    assert used.mean_torque_projection_scale == 1.0
    for _ in range(10):
        actual = used.compute(state, target, dt=0.002)
        np.testing.assert_array_equal(actual, fresh.compute(state, target, dt=0.002))
        assert used.reference_normal_position_m == fresh.reference_normal_position_m
        assert capture_franka_controller_telemetry(used) == capture_franka_controller_telemetry(
            fresh
        )
        assert np.max(np.abs(context.joint_torque(actual))) <= 0.18 + 1e-14


@pytest.mark.parametrize("dt", [0.0, -0.01, np.nan, np.inf, -np.inf])
def test_invalid_dt_is_rejected_before_state_changes(dt: float) -> None:
    controller = FrankaRateLimitedAdaptiveController()
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        controller.compute(_state(), _target(), dt)
    assert controller.reference_normal_position_m is None


@pytest.mark.parametrize(
    "name", ["max_normal_lead", "max_approach_velocity", "max_approach_acceleration"]
)
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_reference_limits_must_be_finite_positive(name: str, value: float) -> None:
    with pytest.raises(ValueError):
        FrankaRateLimitedAdaptiveController(**{name: value})


@pytest.mark.parametrize("name", ["impact_force_margin", "impact_force_rate"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_impact_thresholds_must_be_finite_nonnegative(name: str, value: float) -> None:
    with pytest.raises(ValueError):
        FrankaRateLimitedAdaptiveController(**{name: value})


@pytest.mark.parametrize("normal", [np.zeros(3), np.full(3, np.nan), np.ones(2)])
def test_invalid_normal_is_rejected(normal: np.ndarray) -> None:
    with pytest.raises(ValueError, match="normal"):
        FrankaRateLimitedAdaptiveController(
            base=FrankaAdaptiveHybridController(base=FrankaHybridController(normal=normal)),
        )
