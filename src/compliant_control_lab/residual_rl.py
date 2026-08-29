"""Bounded Cartesian Residual RL policy and safety wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Protocol

import numpy as np

from compliant_control_lab.franka_adaptive import FrankaAdaptiveHybridController
from compliant_control_lab.franka_control import FrankaController, FrankaState, FrankaTarget

OBSERVATION_NAMES = (
    "normal_force_error",
    "normal_force",
    "normal_force_rate",
    "normal_position_error",
    "normal_velocity_error",
    "tangent_y_position_error",
    "tangent_z_position_error",
    "tangent_y_velocity_error",
    "tangent_z_velocity_error",
    "target_normal_force",
    "contact_blend",
    "previous_normal_action",
    "previous_tangent_y_action",
    "previous_tangent_z_action",
)
OBSERVATION_DIM = len(OBSERVATION_NAMES)
ACTION_DIM = 3


class ResidualPolicy(Protocol):
    def action(self, observation: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class LinearResidualPolicy:
    """Inspectable linear policy with a bounded normalized output."""

    weights: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        bias = np.asarray(self.bias, dtype=float)
        if weights.shape != (ACTION_DIM, OBSERVATION_DIM):
            raise ValueError(
                f"weights must have shape {(ACTION_DIM, OBSERVATION_DIM)}, got {weights.shape}"
            )
        if bias.shape != (ACTION_DIM,):
            raise ValueError(f"bias must have shape {(ACTION_DIM,)}, got {bias.shape}")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
            raise ValueError("policy parameters must be finite")
        object.__setattr__(self, "weights", weights.copy())
        object.__setattr__(self, "bias", bias.copy())

    @classmethod
    def zero(cls) -> LinearResidualPolicy:
        return cls(np.zeros((ACTION_DIM, OBSERVATION_DIM)), np.zeros(ACTION_DIM))

    @classmethod
    def from_parameter_vector(cls, parameters: np.ndarray) -> LinearResidualPolicy:
        parameters = np.asarray(parameters, dtype=float)
        expected_size = ACTION_DIM * OBSERVATION_DIM + ACTION_DIM
        if parameters.shape != (expected_size,):
            raise ValueError(f"parameter vector must have shape {(expected_size,)}")
        split = ACTION_DIM * OBSERVATION_DIM
        return cls(parameters[:split].reshape(ACTION_DIM, OBSERVATION_DIM), parameters[split:])

    def parameter_vector(self) -> np.ndarray:
        return np.concatenate((self.weights.ravel(), self.bias))

    def action(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (OBSERVATION_DIM,):
            raise ValueError(f"observation must have shape {(OBSERVATION_DIM,)}")
        return np.tanh(self.weights @ observation + self.bias)

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "format_version": 1,
            "policy_type": "linear_tanh",
            "observation_names": list(OBSERVATION_NAMES),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "metadata": metadata or {},
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LinearResidualPolicy:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or payload.get("policy_type") != "linear_tanh":
            raise ValueError("unsupported residual policy format")
        if tuple(payload.get("observation_names", ())) != OBSERVATION_NAMES:
            raise ValueError("checkpoint observation schema does not match this version")
        return cls(np.asarray(payload["weights"], dtype=float), np.asarray(payload["bias"], dtype=float))


def encode_residual_observation(
    state: FrankaState,
    target: FrankaTarget,
    normal: np.ndarray,
    normal_force: float,
    normal_force_rate: float,
    contact_blend: float,
    previous_normalized_action: np.ndarray,
) -> np.ndarray:
    """Encode only feedback quantities that are available to a real controller."""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    position_error = target.position - state.position
    velocity_error = target.linear_velocity - state.linear_velocity
    previous_normalized_action = np.asarray(previous_normalized_action, dtype=float)
    if previous_normalized_action.shape != (ACTION_DIM,):
        raise ValueError("previous_normalized_action must have shape (3,)")
    observation = np.array(
        [
            (target.normal_force - normal_force) / 12.0,
            normal_force / 20.0,
            normal_force_rate / 250.0,
            float(normal @ position_error) / 0.02,
            float(normal @ velocity_error) / 0.10,
            position_error[1] / 0.05,
            position_error[2] / 0.05,
            velocity_error[1] / 0.10,
            velocity_error[2] / 0.10,
            target.normal_force / 12.0,
            contact_blend,
            *previous_normalized_action,
        ],
        dtype=float,
    )
    return np.clip(observation, -3.0, 3.0)


@dataclass
class BoundedResidualController:
    """Add a slow learned translational residual inside a hard safety envelope."""

    policy: ResidualPolicy = field(default_factory=LinearResidualPolicy.zero)
    nominal: FrankaController = field(default_factory=FrankaAdaptiveHybridController)
    normal: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    action_bounds: np.ndarray = field(default_factory=lambda: np.array([4.0, 6.0, 6.0]))
    action_rate_limits: np.ndarray = field(default_factory=lambda: np.array([40.0, 60.0, 60.0]))
    policy_period: float = 0.02
    filter_time_constant: float = 0.04
    residual_enable_delay: float = 0.10
    force_guard_margin: float = 3.0
    force_guard_rate: float = 150.0
    min_total_normal_wrench: float = -4.0
    max_total_normal_wrench: float = 25.0
    inference_deadline_us: float = 5_000.0
    name: str = "residual_rl"

    _residual: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _desired_residual: np.ndarray = field(
        default_factory=lambda: np.zeros(3), init=False, repr=False
    )
    _normalized_action: np.ndarray = field(
        default_factory=lambda: np.zeros(3), init=False, repr=False
    )
    _policy_elapsed: float = field(default=0.0, init=False, repr=False)
    _contact_ready_elapsed: float = field(default=0.0, init=False, repr=False)
    _previous_force: float | None = field(default=None, init=False, repr=False)
    _force_rate: float = field(default=0.0, init=False, repr=False)
    _fallback_count: int = field(default=0, init=False, repr=False)
    _policy_update_count: int = field(default=0, init=False, repr=False)
    _residual_squared_sum: float = field(default=0.0, init=False, repr=False)
    _residual_sample_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.normal = np.asarray(self.normal, dtype=float)
        self.action_bounds = np.asarray(self.action_bounds, dtype=float)
        self.action_rate_limits = np.asarray(self.action_rate_limits, dtype=float)
        if self.normal.shape != (3,) or np.linalg.norm(self.normal) <= 0.0:
            raise ValueError("normal must be a nonzero 3-vector")
        if self.action_bounds.shape != (3,) or np.any(self.action_bounds <= 0.0):
            raise ValueError("action_bounds must be a positive 3-vector")
        if self.action_rate_limits.shape != (3,) or np.any(self.action_rate_limits <= 0.0):
            raise ValueError("action_rate_limits must be a positive 3-vector")
        if self.policy_period <= 0.0 or self.filter_time_constant <= 0.0:
            raise ValueError("policy_period and filter_time_constant must be positive")
        if self.residual_enable_delay < 0.0:
            raise ValueError("residual_enable_delay must be nonnegative")
        if self.force_guard_margin < 0.0 or self.force_guard_rate < 0.0:
            raise ValueError("force guard thresholds must be nonnegative")
        if self.min_total_normal_wrench >= self.max_total_normal_wrench:
            raise ValueError("total normal wrench limits must be ordered")
        if self.inference_deadline_us <= 0.0:
            raise ValueError("inference_deadline_us must be positive")
        self.normal = self.normal / np.linalg.norm(self.normal)

    @property
    def last_residual(self) -> np.ndarray:
        return self._residual.copy()

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    @property
    def policy_update_count(self) -> int:
        return self._policy_update_count

    @property
    def residual_rms_n(self) -> float:
        if self._residual_sample_count == 0:
            return 0.0
        return float(np.sqrt(self._residual_squared_sum / self._residual_sample_count))

    def reset(self, state: FrankaState) -> None:
        self.nominal.reset(state)
        self._residual = np.zeros(3)
        self._desired_residual = np.zeros(3)
        self._normalized_action = np.zeros(3)
        self._policy_elapsed = self.policy_period
        self._contact_ready_elapsed = 0.0
        self._previous_force = None
        self._force_rate = 0.0
        self._fallback_count = 0
        self._policy_update_count = 0
        self._residual_squared_sum = 0.0
        self._residual_sample_count = 0

    def _nominal_feedback(self, state: FrankaState, dt: float) -> tuple[float, float, float]:
        normal_force = float(getattr(self.nominal, "corrected_force_n", state.normal_force))
        reported_rate = getattr(self.nominal, "filtered_force_rate_n_s", None)
        if reported_rate is not None:
            self._force_rate = float(reported_rate)
        elif dt > 0.0 and self._previous_force is not None:
            instantaneous_rate = (normal_force - self._previous_force) / dt
            alpha = dt / (0.03 + dt)
            self._force_rate += alpha * (instantaneous_rate - self._force_rate)
        self._previous_force = normal_force
        contact_blend = float(getattr(self.nominal, "contact_blend", normal_force >= 3.0))
        return normal_force, self._force_rate, contact_blend

    def _update_policy(
        self,
        state: FrankaState,
        target: FrankaTarget,
        normal_force: float,
        force_rate: float,
        contact_blend: float,
    ) -> bool:
        observation = encode_residual_observation(
            state,
            target,
            self.normal,
            normal_force,
            force_rate,
            contact_blend,
            self._normalized_action,
        )
        start_ns = perf_counter_ns()
        try:
            action = np.asarray(self.policy.action(observation), dtype=float)
        except Exception:  # noqa: BLE001 - policy failures must enter the safety fallback
            action = np.full(ACTION_DIM, np.nan)
        elapsed_us = (perf_counter_ns() - start_ns) / 1_000.0
        if (
            action.shape != (ACTION_DIM,)
            or not np.all(np.isfinite(action))
            or elapsed_us > self.inference_deadline_us
        ):
            self._normalized_action = np.zeros(3)
            self._desired_residual = np.zeros(3)
            self._residual = np.zeros(3)
            self._fallback_count += 1
            return False
        self._normalized_action = np.clip(action, -1.0, 1.0)
        contact_activation = float(
            contact_blend >= 0.99
            and self._contact_ready_elapsed >= self.residual_enable_delay
        )
        self._desired_residual = (
            contact_activation * self.action_bounds * self._normalized_action
        )
        self._policy_update_count += 1
        return True

    def compute(self, state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray:
        safe_dt = max(0.0, dt)
        nominal_wrench = self.nominal.compute(state, target, dt)
        normal_force, force_rate, contact_blend = self._nominal_feedback(state, safe_dt)
        if contact_blend >= 0.99:
            self._contact_ready_elapsed += safe_dt
        else:
            self._contact_ready_elapsed = 0.0
            self._desired_residual = np.zeros(3)
            self._residual = np.zeros(3)

        self._policy_elapsed += safe_dt
        policy_ok = True
        if self._policy_elapsed + 1.0e-12 >= self.policy_period:
            self._policy_elapsed %= self.policy_period
            policy_ok = self._update_policy(
                state,
                target,
                normal_force,
                force_rate,
                contact_blend,
            )

        force_guard_active = (
            normal_force > target.normal_force + self.force_guard_margin
            or force_rate > self.force_guard_rate
        )
        if force_guard_active:
            self._desired_residual[0] = min(0.0, self._desired_residual[0])
            self._residual[0] = min(0.0, self._residual[0])

        if policy_ok and safe_dt > 0.0:
            alpha = safe_dt / (self.filter_time_constant + safe_dt)
            filtered_target = self._residual + alpha * (
                self._desired_residual - self._residual
            )
            max_delta = self.action_rate_limits * safe_dt
            self._residual += np.clip(filtered_target - self._residual, -max_delta, max_delta)

        wrench = nominal_wrench.copy()
        wrench[:3] += self._residual
        total_normal = float(self.normal @ wrench[:3])
        safe_total_normal = float(
            np.clip(
                total_normal,
                self.min_total_normal_wrench,
                self.max_total_normal_wrench,
            )
        )
        wrench[:3] += self.normal * (safe_total_normal - total_normal)
        self._residual_squared_sum += float(self._residual @ self._residual)
        self._residual_sample_count += 1
        return wrench
