from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compliant_control_lab.franka_adaptive import FrankaAdaptiveHybridController
from compliant_control_lab.franka_control import FrankaState, FrankaTarget
from compliant_control_lab.residual_rl import (
    ACTION_DIM,
    OBSERVATION_DIM,
    BoundedResidualController,
    LinearResidualPolicy,
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
