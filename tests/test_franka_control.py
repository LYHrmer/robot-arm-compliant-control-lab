import numpy as np

from compliant_control_lab.franka_control import (
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

