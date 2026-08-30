import numpy as np
import pytest

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.franka_control import (
    FrankaActuationContext,
    FrankaState,
    FrankaTarget,
)
from compliant_control_lab.franka_simulation import (
    FrankaSimulationConfig,
    run_franka_trial,
)
from compliant_control_lab.franka_torque_safety import (
    project_residual_force,
    project_wrench_to_torque_limits,
    residual_torque_headroom,
)
from compliant_control_lab.residual_rl import TorqueProjectedResidualController


def _context(
    jacobian: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    offset: np.ndarray | None = None,
) -> FrankaActuationContext:
    dof = jacobian.shape[1]
    return FrankaActuationContext(
        cartesian_jacobian=jacobian,
        joint_torque_offset=np.zeros(dof) if offset is None else offset,
        lower_torque_limit=lower,
        upper_torque_limit=upper,
    )


def _state(context: FrankaActuationContext | None = None) -> FrankaState:
    return FrankaState(
        position=np.array([0.36, 0.0, 0.45]),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        normal_force=0.0,
        actuation=context,
    )


def _target() -> FrankaTarget:
    return FrankaTarget(
        position=np.array([0.38, 0.04, 0.42]),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        normal_force=12.0,
    )


def test_full_wrench_inside_torque_envelope_is_unchanged() -> None:
    jacobian = np.zeros((6, 2))
    jacobian[0, 0] = 1.0
    jacobian[4, 1] = 2.0
    context = _context(jacobian, np.array([-10.0, -10.0]), np.array([10.0, 10.0]))
    additive = np.array([3.0, 0.0, 0.0, 0.0, -2.0, 0.0])

    projection = project_wrench_to_torque_limits(
        context,
        nominal_wrench=np.zeros(6),
        additive_wrench=additive,
        reserve_fraction=0.0,
    )

    assert projection.status == "unchanged"
    assert projection.scale == 1.0
    np.testing.assert_array_equal(projection.additive_wrench, additive)


def test_full_wrench_ray_projection_has_hand_calculable_scale() -> None:
    jacobian = np.zeros((6, 1))
    jacobian[0, 0] = 2.0
    context = _context(jacobian, np.array([-10.0]), np.array([10.0]))
    nominal = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    additive = np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    projection = project_wrench_to_torque_limits(
        context,
        nominal_wrench=nominal,
        additive_wrench=additive,
        reserve_fraction=0.0,
    )

    assert projection.status == "scaled"
    assert projection.scale == pytest.approx(0.5)
    np.testing.assert_allclose(projection.additive_wrench, 0.5 * additive)
    assert context.joint_torque(nominal + projection.additive_wrench)[0] <= 10.0


def test_jointly_coupled_residual_respects_every_torque_limit() -> None:
    jacobian = np.zeros((6, 3))
    jacobian[:3] = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    context = _context(
        jacobian,
        lower=np.array([-5.0, -4.0, -3.0]),
        upper=np.array([5.0, 4.0, 3.0]),
        offset=np.array([0.5, -0.5, 0.25]),
    )
    residual = np.array([8.0, 5.0, -7.0])

    projection = project_residual_force(
        context,
        nominal_wrench=np.zeros(6),
        residual_force=residual,
        reserve_fraction=0.0,
    )

    assert 0.0 < projection.scale < 1.0
    np.testing.assert_allclose(projection.residual_force, projection.scale * residual)
    torque = context.joint_torque(projection.additive_wrench)
    assert np.all(torque >= context.lower_torque_limit - 1.0e-12)
    assert np.all(torque <= context.upper_torque_limit + 1.0e-12)


def test_directional_headroom_reports_asymmetric_positive_and_negative_space() -> None:
    jacobian = np.zeros((6, 1))
    jacobian[0, 0] = 1.0
    context = _context(
        jacobian,
        lower=np.array([-10.0]),
        upper=np.array([4.0]),
        offset=np.array([2.0]),
    )

    headroom = residual_torque_headroom(
        context,
        nominal_wrench=np.zeros(6),
        action_bounds=np.array([4.0, 6.0, 6.0]),
        reserve_fraction=0.0,
    )

    np.testing.assert_allclose(headroom, np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0]))


def test_zero_additive_wrench_is_exactly_unchanged() -> None:
    jacobian = np.zeros((6, 2))
    jacobian[0, 0] = 1.0
    jacobian[1, 1] = 1.0
    context = _context(jacobian, np.array([-5.0, -5.0]), np.array([5.0, 5.0]))

    projection = project_residual_force(
        context,
        nominal_wrench=np.ones(6),
        residual_force=np.zeros(3),
        reserve_fraction=0.0,
    )

    assert projection.status == "unchanged"
    assert projection.scale == 1.0
    np.testing.assert_array_equal(projection.additive_wrench, np.zeros(6))


def test_nominal_outside_reserved_envelope_disables_additive_wrench() -> None:
    jacobian = np.zeros((6, 1))
    jacobian[0, 0] = 1.0
    context = _context(jacobian, np.array([-10.0]), np.array([10.0]))

    projection = project_residual_force(
        context,
        nominal_wrench=np.array([9.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        residual_force=np.array([-2.0, 0.0, 0.0]),
        reserve_fraction=0.10,
    )

    assert projection.status == "nominal_outside"
    assert projection.scale == 0.0
    np.testing.assert_array_equal(projection.additive_wrench, np.zeros(6))


def test_nonzero_offset_nominal_and_residual_are_projected_in_two_layers() -> None:
    jacobian = np.zeros((6, 2))
    jacobian[0, 0] = 1.0
    jacobian[1, 1] = 1.0
    context = _context(
        jacobian,
        lower=np.array([-10.0, -10.0]),
        upper=np.array([10.0, 10.0]),
        offset=np.array([1.0, -1.0]),
    )
    raw_nominal = np.array([16.0, 4.0, 0.0, 0.0, 0.0, 0.0])

    nominal_projection = project_wrench_to_torque_limits(
        context,
        nominal_wrench=np.zeros(6),
        additive_wrench=raw_nominal,
        reserve_fraction=0.10,
    )
    residual_projection = project_residual_force(
        context,
        nominal_wrench=nominal_projection.additive_wrench,
        residual_force=np.array([0.0, 16.0, 0.0]),
        reserve_fraction=0.10,
    )

    assert nominal_projection.status == "scaled"
    assert nominal_projection.scale == pytest.approx(0.5)
    assert residual_projection.status == "scaled"
    assert residual_projection.scale == pytest.approx(0.5)
    np.testing.assert_allclose(nominal_projection.additive_wrench[:2], np.array([8.0, 2.0]))
    np.testing.assert_allclose(residual_projection.residual_force, np.array([0.0, 8.0, 0.0]))
    final_wrench = nominal_projection.additive_wrench + residual_projection.additive_wrench
    np.testing.assert_allclose(context.joint_torque(final_wrench), np.array([9.0, 9.0]))


def test_missing_context_and_nonfinite_input_fail_closed() -> None:
    missing = project_residual_force(
        None,
        nominal_wrench=np.zeros(6),
        residual_force=np.ones(3),
    )
    assert missing.status == "missing_context"
    assert missing.scale == 0.0
    np.testing.assert_array_equal(missing.additive_wrench, np.zeros(6))

    jacobian = np.zeros((6, 1))
    context = _context(jacobian, np.array([-10.0]), np.array([10.0]))
    nonfinite = project_residual_force(
        context,
        nominal_wrench=np.zeros(6),
        residual_force=np.array([np.nan, 0.0, 0.0]),
    )
    assert nonfinite.status == "nonfinite"
    assert nonfinite.scale == 0.0
    np.testing.assert_array_equal(nonfinite.additive_wrench, np.zeros(6))


def test_nonfinite_actuation_context_is_rejected_then_fails_closed_if_corrupted() -> None:
    jacobian = np.zeros((6, 1))
    with pytest.raises(ValueError, match="finite"):
        _context(
            np.full((6, 1), np.nan),
            lower=np.array([-10.0]),
            upper=np.array([10.0]),
        )

    context = _context(jacobian, np.array([-10.0]), np.array([10.0]))
    state = _state(context)
    controller = FrankaSafeAdaptiveController(torque_reserve_fraction=0.0)
    controller.reset(state)
    context.cartesian_jacobian[0, 0] = np.nan

    wrench = controller.compute(state, _target(), dt=0.002)

    np.testing.assert_array_equal(wrench, np.zeros(6))
    assert controller.last_torque_projection_scale == 0.0
    assert controller.torque_projection_fallback_count == 1


def test_nonfinite_safe_adaptive_wrench_is_disabled_before_actuation() -> None:
    jacobian = np.zeros((6, 1))
    jacobian[0, 0] = 1.0
    context = _context(jacobian, np.array([-10.0]), np.array([10.0]))
    state = _state(context)
    controller = FrankaSafeAdaptiveController(torque_reserve_fraction=0.0)
    controller.reset(state)
    controller.base.compute = lambda *_: np.full(6, np.nan)  # type: ignore[method-assign]

    wrench = controller.compute(state, _target(), dt=0.002)

    np.testing.assert_array_equal(wrench, np.zeros(6))
    assert controller.last_torque_projection_scale == 0.0
    assert controller.torque_projection_fallback_count == 1


@pytest.mark.parametrize("duration", [0.06, 2.2], ids=["short", "contact_and_slide"])
@pytest.mark.parametrize("controller_kind", ["safe_adaptive", "zero_residual"])
def test_real_franka_safe_rollouts_do_not_reach_actuator_clip(
    duration: float,
    controller_kind: str,
) -> None:
    controller = (
        FrankaSafeAdaptiveController()
        if controller_kind == "safe_adaptive"
        else TorqueProjectedResidualController()
    )

    result = run_franka_trial(
        controller,
        config=FrankaSimulationConfig(duration=duration, seed=11),
    )

    assert not np.any(result.saturated)
    assert result.metrics()["saturation_pct"] == 0.0
    if isinstance(controller, TorqueProjectedResidualController):
        np.testing.assert_array_equal(controller.last_residual, np.zeros(3))


@pytest.mark.parametrize("reserve_fraction", [-0.01, 1.0, np.nan])
def test_reserve_fraction_must_leave_a_nonempty_envelope(reserve_fraction: float) -> None:
    jacobian = np.zeros((6, 1))
    context = _context(jacobian, np.array([-10.0]), np.array([10.0]))

    with pytest.raises(ValueError, match="reserve_fraction"):
        project_residual_force(
            context,
            nominal_wrench=np.zeros(6),
            residual_force=np.zeros(3),
            reserve_fraction=reserve_fraction,
        )
