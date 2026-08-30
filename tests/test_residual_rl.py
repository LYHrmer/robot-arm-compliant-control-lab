import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.franka_adaptive import FrankaAdaptiveHybridController
from compliant_control_lab.franka_control import FrankaState, FrankaTarget
from compliant_control_lab.residual_rl import (
    ACTION_DIM,
    OBSERVATION_DIM,
    OBSERVATION_NAMES,
    TORQUE_AWARE_OBSERVATION_DIM,
    TORQUE_AWARE_OBSERVATION_NAMES,
    BoundedResidualController,
    LinearResidualPolicy,
    TorqueProjectedResidualController,
)


def _state(force: float = 0.0) -> FrankaState:
    return FrankaState(
        position=np.array([0.36, 0.0, 0.45]),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        normal_force=force,
    )


def _target(force: float = 12.0) -> FrankaTarget:
    return FrankaTarget(
        position=np.array([0.38, 0.04, 0.42]),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        normal_force=force,
    )


@dataclass
class ConstantPolicy:
    value: np.ndarray

    def action(self, observation: np.ndarray) -> np.ndarray:
        assert observation.shape == (OBSERVATION_DIM,)
        return self.value.copy()


class ZeroWrenchContactController:
    name = "zero_wrench_contact"
    contact_blend = 1.0
    corrected_force_n = 5.0
    filtered_force_rate_n_s = 0.0

    def reset(self, state: FrankaState) -> None:
        del state

    def compute(self, state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray:
        del state, target, dt
        return np.zeros(6)


class RecordingTorqueAwarePolicy:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value
        self.observations: list[np.ndarray] = []

    def action(self, observation: np.ndarray) -> np.ndarray:
        assert observation.shape == (TORQUE_AWARE_OBSERVATION_DIM,)
        self.observations.append(observation.copy())
        return self.value.copy()


def _actuation_state(force: float = 0.0) -> FrankaState:
    from compliant_control_lab.franka_control import FrankaActuationContext

    jacobian = np.zeros((6, 7))
    jacobian[0, 4] = 1.0
    jacobian[1, 5] = 1.0
    jacobian[2, 6] = 1.0
    state = _state(force)
    return FrankaState(
        position=state.position,
        rotation=state.rotation,
        linear_velocity=state.linear_velocity,
        angular_velocity=state.angular_velocity,
        normal_force=state.normal_force,
        actuation=FrankaActuationContext(
            cartesian_jacobian=jacobian,
            joint_torque_offset=np.zeros(7),
            lower_torque_limit=np.array([-87.0] * 4 + [-12.0] * 3),
            upper_torque_limit=np.array([87.0] * 4 + [12.0] * 3),
        ),
    )


def test_zero_residual_matches_adaptive_nominal_over_state_sequence() -> None:
    nominal = FrankaAdaptiveHybridController()
    wrapped_nominal = FrankaAdaptiveHybridController()
    controller = BoundedResidualController(
        policy=LinearResidualPolicy.zero(),
        nominal=wrapped_nominal,
        inference_deadline_us=1.0e9,
    )
    initial = _state()
    nominal.reset(initial)
    controller.reset(initial)

    for force in (0.0, 1.0, 3.5, 8.0, 14.0, 11.5):
        state = _state(force)
        expected = nominal.compute(state, _target(), dt=0.01)
        actual = controller.compute(state, _target(), dt=0.01)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_zero_policy_does_not_clamp_an_out_of_range_nominal_wrench() -> None:
    class LargeNominal(ZeroWrenchContactController):
        def compute(self, state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray:
            del state, target, dt
            return np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    controller = BoundedResidualController(
        policy=LinearResidualPolicy.zero(),
        nominal=LargeNominal(),
        inference_deadline_us=1.0e9,
    )
    controller.reset(_state())
    wrench = controller.compute(_state(), _target(), dt=0.02)
    np.testing.assert_array_equal(wrench, np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_residual_action_is_clipped_filtered_and_rate_limited() -> None:
    controller = BoundedResidualController(
        policy=ConstantPolicy(np.array([10.0, -10.0, 2.0])),
        nominal=ZeroWrenchContactController(),
        action_bounds=np.array([4.0, 6.0, 6.0]),
        action_rate_limits=np.array([1.0, 2.0, 3.0]),
        policy_period=0.01,
        filter_time_constant=0.01,
        residual_enable_delay=0.0,
        inference_deadline_us=1.0e9,
    )
    controller.reset(_state())
    controller.compute(_state(), _target(), dt=0.01)

    np.testing.assert_allclose(controller.last_residual, np.array([0.01, -0.02, 0.03]))
    for _ in range(1_000):
        controller.compute(_state(), _target(), dt=0.01)
    assert np.all(np.abs(controller.last_residual) <= controller.action_bounds + 1.0e-12)


def test_nonfinite_policy_action_immediately_falls_back_to_zero() -> None:
    policy = ConstantPolicy(np.ones(ACTION_DIM))
    controller = BoundedResidualController(
        policy=policy,
        nominal=ZeroWrenchContactController(),
        policy_period=0.01,
        residual_enable_delay=0.0,
        inference_deadline_us=1.0e9,
    )
    controller.reset(_state())
    controller.compute(_state(), _target(), dt=0.02)
    assert np.linalg.norm(controller.last_residual) > 0.0

    policy.value[:] = np.nan
    controller.compute(_state(), _target(), dt=0.02)
    np.testing.assert_array_equal(controller.last_residual, np.zeros(3))
    assert controller.fallback_count == 1


def test_residual_is_disabled_until_nominal_contact_is_confirmed() -> None:
    controller = BoundedResidualController(
        policy=ConstantPolicy(np.ones(ACTION_DIM)),
        policy_period=0.01,
        residual_enable_delay=0.0,
        inference_deadline_us=1.0e9,
    )
    controller.reset(_state())
    controller.compute(_state(), _target(), dt=0.02)

    np.testing.assert_array_equal(controller.last_residual, np.zeros(3))


def test_force_guard_removes_positive_normal_residual_on_overshoot() -> None:
    nominal = ZeroWrenchContactController()
    controller = BoundedResidualController(
        policy=ConstantPolicy(np.ones(ACTION_DIM)),
        nominal=nominal,
        policy_period=0.01,
        residual_enable_delay=0.0,
        inference_deadline_us=1.0e9,
    )
    controller.reset(_state())
    controller.compute(_state(), _target(), dt=0.02)
    assert controller.last_residual[0] > 0.0

    nominal.corrected_force_n = 20.0
    controller.compute(_state(force=20.0), _target(), dt=0.002)
    assert controller.last_residual[0] <= 0.0


def test_residual_waits_for_stable_contact_before_enabling() -> None:
    controller = BoundedResidualController(
        policy=ConstantPolicy(np.ones(ACTION_DIM)),
        nominal=ZeroWrenchContactController(),
        policy_period=0.01,
        residual_enable_delay=0.05,
        inference_deadline_us=1.0e9,
    )
    controller.reset(_state())
    controller.compute(_state(), _target(), dt=0.02)
    controller.compute(_state(), _target(), dt=0.02)
    np.testing.assert_array_equal(controller.last_residual, np.zeros(3))

    controller.compute(_state(), _target(), dt=0.02)
    assert np.linalg.norm(controller.last_residual) > 0.0


def test_torque_aware_policy_receives_six_directional_headroom_values() -> None:
    policy = RecordingTorqueAwarePolicy(np.zeros(ACTION_DIM))
    controller = TorqueProjectedResidualController(
        policy=policy,
        nominal=ZeroWrenchContactController(),
        policy_period=0.01,
        residual_enable_delay=0.0,
        inference_deadline_us=1.0e9,
    )
    state = _actuation_state()
    controller.reset(state)
    controller.compute(state, _target(), dt=0.02)

    assert len(policy.observations) == 1
    np.testing.assert_allclose(policy.observations[0][-6:], np.ones(6))


def test_torque_projection_keeps_residual_inside_reserved_joint_limits() -> None:
    policy = RecordingTorqueAwarePolicy(np.ones(ACTION_DIM))
    nominal = ZeroWrenchContactController()
    controller = TorqueProjectedResidualController(
        policy=policy,
        nominal=nominal,
        action_bounds=np.array([20.0, 20.0, 20.0]),
        action_rate_limits=np.array([1.0e6, 1.0e6, 1.0e6]),
        policy_period=0.01,
        filter_time_constant=1.0e-9,
        residual_enable_delay=0.0,
        torque_reserve_fraction=0.10,
        inference_deadline_us=1.0e9,
    )
    state = _actuation_state()
    controller.reset(state)
    wrench = controller.compute(state, _target(), dt=0.02)

    assert state.actuation is not None
    torque = state.actuation.joint_torque(wrench)
    assert np.all(np.abs(torque[4:]) <= 10.8 + 1.0e-12)
    assert controller.last_torque_projection_scale < 1.0


def test_losing_actuation_context_clears_an_active_residual_immediately() -> None:
    policy = RecordingTorqueAwarePolicy(np.ones(ACTION_DIM))
    controller = TorqueProjectedResidualController(
        policy=policy,
        nominal=ZeroWrenchContactController(),
        policy_period=0.01,
        filter_time_constant=0.01,
        residual_enable_delay=0.0,
        inference_deadline_us=1.0e9,
    )
    state = _actuation_state()
    controller.reset(state)
    controller.compute(state, _target(), dt=0.02)
    assert np.linalg.norm(controller.last_residual) > 0.0

    wrench = controller.compute(_state(), _target(), dt=0.002)
    np.testing.assert_array_equal(controller.last_residual, np.zeros(3))
    np.testing.assert_array_equal(wrench, np.zeros(6))
    assert controller.torque_context_fallback_count == 1


def test_linear_policy_checkpoint_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    policy = LinearResidualPolicy(
        rng.normal(size=(ACTION_DIM, OBSERVATION_DIM)),
        rng.normal(size=ACTION_DIM),
    )
    path = tmp_path / "policy.json"
    policy.save(path, metadata={"training_seed": 7})
    restored = LinearResidualPolicy.load(path)

    observation = rng.normal(size=OBSERVATION_DIM)
    np.testing.assert_allclose(restored.action(observation), policy.action(observation))
    np.testing.assert_allclose(restored.parameter_vector(), policy.parameter_vector())


def test_linear_policy_supports_a_versioned_observation_schema(tmp_path: Path) -> None:
    names = ("force", "headroom_positive", "headroom_negative")
    policy = LinearResidualPolicy.zero(names)
    path = tmp_path / "torque_aware_policy.json"
    policy.save(path)
    restored = LinearResidualPolicy.load(path)

    assert restored.observation_names == names
    assert restored.weights.shape == (ACTION_DIM, len(names))
    np.testing.assert_array_equal(restored.action(np.ones(len(names))), np.zeros(ACTION_DIM))


def test_published_v1_checkpoint_still_loads_into_bounded_controller(tmp_path: Path) -> None:
    path = tmp_path / "published_v1_policy.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "policy_type": "linear_tanh",
                "observation_names": list(OBSERVATION_NAMES),
                "weights": np.zeros((ACTION_DIM, OBSERVATION_DIM)).tolist(),
                "bias": np.zeros(ACTION_DIM).tolist(),
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    restored = LinearResidualPolicy.load(path)
    controller = BoundedResidualController(policy=restored)

    assert controller.policy is restored
    assert restored.observation_names == OBSERVATION_NAMES


def test_bounded_controller_rejects_a_reordered_linear_policy_schema() -> None:
    reordered = tuple(reversed(OBSERVATION_NAMES))
    policy = LinearResidualPolicy.zero(reordered)

    with pytest.raises(ValueError, match="bounded residual policy observation schema"):
        BoundedResidualController(policy=policy)


def test_torque_projected_controller_requires_the_torque_aware_schema() -> None:
    with pytest.raises(ValueError, match="torque-aware policy observation schema"):
        TorqueProjectedResidualController(policy=LinearResidualPolicy.zero())

    controller = TorqueProjectedResidualController(
        policy=LinearResidualPolicy.zero(TORQUE_AWARE_OBSERVATION_NAMES)
    )
    assert controller.policy.observation_names == TORQUE_AWARE_OBSERVATION_NAMES
