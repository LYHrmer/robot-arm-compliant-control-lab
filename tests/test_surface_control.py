from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from compliant_control_lab.franka_adaptive import (
    FrankaAdaptiveHybridController,
    FrankaSafeAdaptiveController,
)
from compliant_control_lab.franka_control import (
    FrankaActuationContext,
    FrankaHybridController,
    FrankaState,
    FrankaTarget,
    capture_franka_controller_telemetry,
)
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame


def _rotation(seed: int) -> np.ndarray:
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    rotation[:, -1] *= np.linalg.det(rotation)
    return rotation


def _state(force: float = 0.0) -> FrankaState:
    return FrankaState(
        position=np.array([0.36, 0.02, 0.45]),
        rotation=_rotation(8),
        linear_velocity=np.array([0.01, -0.02, 0.03]),
        angular_velocity=np.array([-0.03, 0.02, 0.01]),
        normal_force=force,
    )


def _target() -> FrankaTarget:
    return FrankaTarget(
        position=np.array([0.38, 0.04, 0.42]),
        rotation=_rotation(9),
        linear_velocity=np.array([0.01, 0.03, -0.02]),
        angular_velocity=np.array([0.02, -0.01, 0.03]),
        normal_force=12.0,
    )


def _context() -> FrankaActuationContext:
    return FrankaActuationContext(
        cartesian_jacobian=np.random.default_rng(4).normal(size=(6, 7)),
        joint_torque_offset=np.linspace(-0.5, 0.5, 7),
        lower_torque_limit=np.full(7, -5.0),
        upper_torque_limit=np.full(7, 5.0),
    )


def _rotate_state(state: FrankaState, rotation: np.ndarray) -> FrankaState:
    actuation = state.actuation
    if actuation is not None:
        jacobian = actuation.cartesian_jacobian
        actuation = replace(
            actuation,
            cartesian_jacobian=np.concatenate(
                (rotation @ jacobian[:3], rotation @ jacobian[3:]), axis=0
            ),
        )
    return replace(
        state,
        position=rotation @ state.position,
        rotation=rotation @ state.rotation,
        linear_velocity=rotation @ state.linear_velocity,
        angular_velocity=rotation @ state.angular_velocity,
        actuation=actuation,
    )


def _rotate_target(target: FrankaTarget, rotation: np.ndarray) -> FrankaTarget:
    return replace(
        target,
        position=rotation @ target.position,
        rotation=rotation @ target.rotation,
        linear_velocity=rotation @ target.linear_velocity,
        angular_velocity=rotation @ target.angular_velocity,
    )


@pytest.mark.parametrize("seed", range(5))
def test_frame_roundtrips_points_vectors_and_same_origin_wrenches(seed: int) -> None:
    frame = SurfaceFrame(_rotation(seed))
    vector = np.array([0.7, -0.2, 0.4])
    wrench = np.arange(6, dtype=float) - 2.0
    np.testing.assert_allclose(frame.vector_to_world(frame.vector_to_local(vector)), vector)
    np.testing.assert_allclose(frame.point_to_world(frame.point_to_local(vector)), vector)
    np.testing.assert_allclose(
        frame.wrench_to_world(frame.wrench_to_local(wrench)), wrench, atol=1e-14
    )
    np.testing.assert_allclose(frame.wrench_to_local(wrench)[:3], frame.vector_to_local(wrench[:3]))
    np.testing.assert_allclose(frame.wrench_to_local(wrench)[3:], frame.vector_to_local(wrench[3:]))


def test_frame_from_normal_and_rotated_hint_define_a_covariant_proper_frame() -> None:
    normal = np.array([1.0, 2.0, -1.0])
    hint = np.array([0.0, 1.0, 1.0])
    frame = SurfaceFrame.from_normal(normal, hint)
    world_rotation = _rotation(12)
    rotated = SurfaceFrame.from_normal(world_rotation @ normal, world_rotation @ hint)

    np.testing.assert_allclose(frame.rotation[:, 0], normal / np.linalg.norm(normal))
    np.testing.assert_allclose(frame.rotation.T @ frame.rotation, np.eye(3), atol=1e-14)
    assert np.linalg.det(frame.rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(rotated.rotation, world_rotation @ frame.rotation, atol=1e-14)
    np.testing.assert_array_equal(
        SurfaceFrame.from_normal(np.array([1.0, 0.0, 0.0])).rotation, np.eye(3)
    )


@pytest.mark.parametrize("angle", [-2.4, -0.3, 0.0, 0.7, 2.9])
def test_default_yaw_frame_keeps_local_z_aligned_with_world_z(angle: float) -> None:
    frame = SurfaceFrame.from_normal(np.array([np.cos(angle), np.sin(angle), 0.0]))
    np.testing.assert_allclose(frame.rotation[:, 2], np.array([0.0, 0.0, 1.0]), atol=1e-14)
    np.testing.assert_allclose(
        frame.rotation[:, 1], np.array([-np.sin(angle), np.cos(angle), 0.0]), atol=1e-14
    )


def test_default_vertical_normal_uses_a_valid_fallback_up_direction() -> None:
    frame = SurfaceFrame.from_normal(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(frame.rotation.T @ frame.rotation, np.eye(3), atol=1e-14)
    np.testing.assert_array_equal(frame.rotation[:, 0], np.array([0.0, 0.0, 1.0]))


@pytest.mark.parametrize("with_actuation", [False, True])
def test_identity_frame_exactly_matches_old_safe_baseline(with_actuation: bool) -> None:
    baseline = FrankaSafeAdaptiveController()
    controller = SurfaceAdaptiveController(SurfaceFrame(np.eye(3)))
    state = replace(_state(), actuation=_context() if with_actuation else None)
    target = _target()
    controller.reset(state)
    baseline.reset(state)
    for step in range(60):
        current = replace(state, normal_force=0.0 if step < 20 else 14.0)
        expected = baseline.compute(current, target, 0.002)
        np.testing.assert_array_equal(controller.compute(current, target, 0.002), expected)
        assert capture_franka_controller_telemetry(
            controller
        ) == capture_franka_controller_telemetry(baseline)
        assert controller.corrected_force_n == baseline.corrected_force_n
        assert controller.filtered_force_rate_n_s == baseline.filtered_force_rate_n_s
        assert controller.torque_projection_pct == baseline.torque_projection_pct
        assert controller.mean_torque_projection_scale == baseline.mean_torque_projection_scale
        assert (
            controller.torque_projection_fallback_count == baseline.torque_projection_fallback_count
        )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("with_actuation", [False, True])
def test_world_rotation_covariance_including_torque_projection(
    seed: int, with_actuation: bool
) -> None:
    frame = SurfaceFrame(_rotation(17))
    world_rotation = _rotation(seed)
    rotated_frame = SurfaceFrame(world_rotation @ frame.rotation)
    controller = SurfaceAdaptiveController(frame)
    rotated_controller = SurfaceAdaptiveController(rotated_frame)
    state = replace(_state(), actuation=_context() if with_actuation else None)
    target = _target()
    controller.reset(state)
    rotated_controller.reset(_rotate_state(state, world_rotation))

    for step in range(50):
        current = replace(state, normal_force=0.0 if step < 10 else 14.0)
        rotated_state = _rotate_state(current, world_rotation)
        wrench = controller.compute(current, target, 0.002)
        rotated_wrench = rotated_controller.compute(
            rotated_state, _rotate_target(target, world_rotation), 0.002
        )
        expected = np.concatenate((world_rotation @ wrench[:3], world_rotation @ wrench[3:]))
        np.testing.assert_allclose(rotated_wrench, expected, atol=2e-12, rtol=2e-12)
        assert rotated_controller.last_torque_projection_scale == pytest.approx(
            controller.last_torque_projection_scale
        )
        if current.actuation is not None:
            assert rotated_state.actuation is not None
            np.testing.assert_allclose(
                rotated_state.actuation.joint_torque(rotated_wrench),
                current.actuation.joint_torque(wrench),
                atol=2e-12,
            )
            torque = current.actuation.joint_torque(wrench)
            assert np.max(np.abs(torque)) <= 4.5 + 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_tangential_impedance_has_no_normal_force_leak(seed: int) -> None:
    frame = SurfaceFrame(_rotation(seed))
    state = replace(
        _state(),
        position=np.zeros(3),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    target = replace(
        _target(),
        position=frame.vector_to_world(np.array([0.0, 0.01, -0.02])),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        normal_force=0.0,
    )
    controller = SurfaceAdaptiveController(frame)

    wrench = controller.compute(state, target, 0.002)

    assert frame.rotation[:, 0] @ wrench[:3] == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(wrench[:3]) > 1.0


def test_supplied_local_base_configuration_is_used_without_exposing_nested_state() -> None:
    nominal = FrankaHybridController(tangential_stiffness=np.array([0.0, 200.0, 600.0]))
    adaptive = FrankaAdaptiveHybridController(base=nominal)
    local_base = FrankaSafeAdaptiveController(base=adaptive, max_normal_lead=0.001)
    frame = SurfaceFrame(_rotation(11))
    local_state, local_target = _state(), _target()
    world_state = _rotate_state(local_state, frame.rotation)
    world_target = _rotate_target(local_target, frame.rotation)
    controller = SurfaceAdaptiveController(frame, base=local_base)
    controller.reset(world_state)
    expected_nominal = FrankaHybridController(tangential_stiffness=np.array([0.0, 200.0, 600.0]))
    expected_base = FrankaSafeAdaptiveController(
        base=FrankaAdaptiveHybridController(base=expected_nominal), max_normal_lead=0.001
    )
    expected_base.reset(local_state)

    expected = frame.wrench_to_world(expected_base.compute(local_state, local_target, 0.002))
    np.testing.assert_allclose(
        controller.compute(world_state, world_target, 0.002), expected, atol=1e-12
    )
    assert controller.frame is frame
    assert controller.last_governed_normal_lead_m == pytest.approx(0.001)


def test_reset_reproduces_fresh_sequence_and_clears_projection_history() -> None:
    frame = SurfaceFrame(_rotation(5))
    used = SurfaceAdaptiveController(frame)
    fresh = SurfaceAdaptiveController(frame)
    state = replace(_state(), actuation=_context())
    for _ in range(30):
        used.compute(state, _target(), 0.002)
    assert used.torque_projection_pct > 0.0
    used.reset(state)
    fresh.reset(state)
    assert used.torque_projection_pct == 0.0
    assert used.mean_torque_projection_scale == 1.0
    assert capture_franka_controller_telemetry(used) == capture_franka_controller_telemetry(fresh)
    for force in (0.0, 2.0, 5.0, 10.0, 20.0, 12.0):
        current = replace(state, normal_force=force)
        np.testing.assert_array_equal(
            used.compute(current, _target(), 0.002), fresh.compute(current, _target(), 0.002)
        )


def test_frame_defensively_copies_and_controller_does_not_mutate_inputs() -> None:
    rotation = _rotation(7)
    original = rotation.copy()
    frame = SurfaceFrame(rotation)
    rotation[:] = np.eye(3)
    np.testing.assert_array_equal(frame.rotation, original)
    with pytest.raises(ValueError, match="read-only"):
        frame.rotation[0, 0] = 2.0
    with pytest.raises(FrozenInstanceError):
        frame.rotation = np.eye(3)  # type: ignore[misc]

    state, target = replace(_state(), actuation=_context()), _target()
    assert state.actuation is not None
    records = (state, target, state.actuation)
    originals = [
        {
            name: value.copy()
            for name, value in vars(record).items()
            if isinstance(value, np.ndarray)
        }
        for record in records
    ]
    controller = SurfaceAdaptiveController(frame)
    controller.reset(state)
    controller.compute(state, target, 0.002)
    for record, saved in zip(records, originals):
        for name, value in saved.items():
            np.testing.assert_array_equal(getattr(record, name), value)


@pytest.mark.parametrize("dt", [0.0, -0.002, np.nan, np.inf, -np.inf])
def test_invalid_dt_is_rejected_without_mutating_state(dt: float) -> None:
    controller = SurfaceAdaptiveController(SurfaceFrame(np.eye(3)))
    before = capture_franka_controller_telemetry(controller)
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        controller.compute(_state(20.0), _target(), dt)
    assert capture_franka_controller_telemetry(controller) == before
    assert controller.corrected_force_n == 0.0
    fresh = SurfaceAdaptiveController(controller.frame)
    np.testing.assert_array_equal(
        controller.compute(_state(), _target(), 0.002), fresh.compute(_state(), _target(), 0.002)
    )


@pytest.mark.parametrize(
    "rotation", [np.eye(2), np.diag([-1.0, 1.0, 1.0]), 2.0 * np.eye(3), np.full((3, 3), np.nan)]
)
def test_invalid_surface_rotations_are_rejected(rotation: np.ndarray) -> None:
    with pytest.raises(ValueError, match="proper orthonormal"):
        SurfaceFrame(rotation)


@pytest.mark.parametrize("normal", [np.zeros(3), np.ones(2), np.array([1.0, np.inf, 0.0])])
def test_invalid_normals_are_rejected(normal: np.ndarray) -> None:
    with pytest.raises(ValueError):
        SurfaceFrame.from_normal(normal)


@pytest.mark.parametrize("hint", [np.zeros(3), np.array([2.0, 0.0, 0.0]), np.full(3, np.nan)])
def test_invalid_tangent_hints_are_rejected(hint: np.ndarray) -> None:
    with pytest.raises(ValueError):
        SurfaceFrame.from_normal(np.array([1.0, 0.0, 0.0]), hint)
