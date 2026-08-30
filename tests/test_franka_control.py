import numpy as np

from compliant_control_lab.franka_control import (
    FrankaActuationContext,
    FrankaAdmittanceController,
    FrankaHybridController,
    FrankaState,
    FrankaTarget,
    damped_nullspace_projector,
    orientation_error,
)


def _state(force=0.0):
    return FrankaState(
        position=np.array([0.36, 0.0, 0.45]),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        normal_force=force,
    )


def _target(force=12.0, normal_position=0.38):
    return FrankaTarget(
        position=np.array([normal_position, 0.04, 0.42]),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        normal_force=force,
    )


def test_orientation_error_has_expected_axis_and_sign():
    angle = 0.1
    desired = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    error = orientation_error(np.eye(3), desired)
    np.testing.assert_allclose(error[:2], 0.0, atol=1e-12)
    assert error[2] > 0.0


def test_damped_projector_removes_task_space_torque():
    rng = np.random.default_rng(4)
    jacobian = rng.normal(size=(6, 7))
    projector = damped_nullspace_projector(jacobian, damping=1e-6)
    np.testing.assert_allclose(jacobian @ projector, 0.0, atol=1e-8)


def test_actuation_context_maps_wrench_and_rejects_bad_shapes():
    jacobian = np.arange(42, dtype=float).reshape(6, 7) / 20.0
    offset = np.linspace(-1.0, 1.0, 7)
    context = FrankaActuationContext(
        cartesian_jacobian=jacobian,
        joint_torque_offset=offset,
        lower_torque_limit=-np.ones(7) * 87.0,
        upper_torque_limit=np.ones(7) * 87.0,
    )
    wrench = np.linspace(-2.0, 3.0, 6)
    np.testing.assert_allclose(context.joint_torque(wrench), jacobian.T @ wrench + offset)

    with np.testing.assert_raises_regex(ValueError, "shape"):
        FrankaActuationContext(
            cartesian_jacobian=np.zeros((3, 7)),
            joint_torque_offset=offset,
            lower_torque_limit=-np.ones(7),
            upper_torque_limit=np.ones(7),
        )


def test_franka_admittance_moves_toward_wall_when_force_is_low():
    controller = FrankaAdmittanceController()
    state = _state(force=0.0)
    controller.reset(state)
    wrench = controller.compute(state, _target(force=12.0), dt=0.01)
    assert wrench[0] > 0.0


def test_hybrid_normal_command_does_not_depend_on_normal_position_error():
    controller = FrankaHybridController()
    state = _state(force=6.0)
    controller.reset(state)
    first = controller.compute(state, _target(normal_position=0.37), dt=0.0)
    second = controller.compute(state, _target(normal_position=0.50), dt=0.0)
    np.testing.assert_allclose(first[0], second[0])
    assert first[1] > 0.0


def test_hybrid_uses_bounded_position_control_before_contact():
    controller = FrankaHybridController()
    state = _state(force=0.0)
    controller.reset(state)
    near = controller.compute(state, _target(normal_position=0.37), dt=0.0)

    controller.reset(state)
    far = controller.compute(state, _target(normal_position=0.50), dt=0.0)
    assert 0.0 < near[0] < far[0]
    assert far[0] == controller.max_approach_command
