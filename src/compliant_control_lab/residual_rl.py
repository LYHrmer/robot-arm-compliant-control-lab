"""Bounded Cartesian Residual RL policy and safety wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Protocol

import numpy as np

from compliant_control_lab.franka_adaptive import (
    FrankaAdaptiveHybridController,
    FrankaSafeAdaptiveController,
)
from compliant_control_lab.franka_control import FrankaController, FrankaState, FrankaTarget
from compliant_control_lab.franka_torque_safety import (
    project_residual_force,
    residual_torque_headroom,
)

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
TORQUE_HEADROOM_NAMES = (
    "normal_positive_torque_headroom",
    "normal_negative_torque_headroom",
    "tangent_y_positive_torque_headroom",
    "tangent_y_negative_torque_headroom",
    "tangent_z_positive_torque_headroom",
    "tangent_z_negative_torque_headroom",
)
TORQUE_AWARE_OBSERVATION_NAMES = OBSERVATION_NAMES + TORQUE_HEADROOM_NAMES
TORQUE_AWARE_OBSERVATION_DIM = len(TORQUE_AWARE_OBSERVATION_NAMES)
ACTION_DIM = 3


class ResidualPolicy(Protocol):
    def action(self, observation: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class LinearResidualPolicy:
    """Inspectable linear policy with a bounded normalized output."""

    weights: np.ndarray
    bias: np.ndarray
    observation_names: tuple[str, ...] = OBSERVATION_NAMES

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        bias = np.asarray(self.bias, dtype=float)
        observation_names = tuple(self.observation_names)
        if not observation_names or len(set(observation_names)) != len(observation_names):
            raise ValueError("observation_names must be nonempty and unique")
        expected_shape = (ACTION_DIM, len(observation_names))
        if weights.shape != expected_shape:
            raise ValueError(
                f"weights must have shape {expected_shape}, got {weights.shape}"
            )
        if bias.shape != (ACTION_DIM,):
            raise ValueError(f"bias must have shape {(ACTION_DIM,)}, got {bias.shape}")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
            raise ValueError("policy parameters must be finite")
        object.__setattr__(self, "weights", weights.copy())
        object.__setattr__(self, "bias", bias.copy())
        object.__setattr__(self, "observation_names", observation_names)

    @classmethod
    def zero(
        cls,
        observation_names: tuple[str, ...] = OBSERVATION_NAMES,
    ) -> LinearResidualPolicy:
        return cls(
            np.zeros((ACTION_DIM, len(observation_names))),
            np.zeros(ACTION_DIM),
            observation_names,
        )

    @classmethod
    def from_parameter_vector(
        cls,
        parameters: np.ndarray,
        observation_names: tuple[str, ...] = OBSERVATION_NAMES,
    ) -> LinearResidualPolicy:
        parameters = np.asarray(parameters, dtype=float)
        observation_dim = len(observation_names)
        expected_size = ACTION_DIM * observation_dim + ACTION_DIM
        if parameters.shape != (expected_size,):
            raise ValueError(f"parameter vector must have shape {(expected_size,)}")
        split = ACTION_DIM * observation_dim
        return cls(
            parameters[:split].reshape(ACTION_DIM, observation_dim),
            parameters[split:],
            observation_names,
        )

    def parameter_vector(self) -> np.ndarray:
        return np.concatenate((self.weights.ravel(), self.bias))

    def action(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=float)
        expected_shape = (len(self.observation_names),)
        if observation.shape != expected_shape:
            raise ValueError(f"observation must have shape {expected_shape}")
        return np.tanh(self.weights @ observation + self.bias)

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "format_version": 2,
            "policy_type": "linear_tanh",
            "observation_names": list(self.observation_names),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "metadata": metadata or {},
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LinearResidualPolicy:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("format_version") not in (1, 2)
            or payload.get("policy_type") != "linear_tanh"
        ):
            raise ValueError("unsupported residual policy format")
        observation_names = tuple(payload.get("observation_names", ()))
        return cls(
            np.asarray(payload["weights"], dtype=float),
            np.asarray(payload["bias"], dtype=float),
            observation_names,
        )


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

    def _expected_policy_observation_names(self) -> tuple[str, ...]:
        return OBSERVATION_NAMES

    def _policy_schema_error(self) -> str:
        return "bounded residual policy observation schema does not match"

    def __post_init__(self) -> None:
        if (
            isinstance(self.policy, LinearResidualPolicy)
            and self.policy.observation_names != self._expected_policy_observation_names()
        ):
            raise ValueError(self._policy_schema_error())
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
        nominal_wrench: np.ndarray,
    ) -> bool:
        observation = self._encode_policy_observation(
            state,
            target,
            normal_force,
            force_rate,
            contact_blend,
            nominal_wrench,
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

    def _encode_policy_observation(
        self,
        state: FrankaState,
        target: FrankaTarget,
        normal_force: float,
        force_rate: float,
        contact_blend: float,
        nominal_wrench: np.ndarray,
    ) -> np.ndarray:
        del nominal_wrench
        return encode_residual_observation(
            state,
            target,
            self.normal,
            normal_force,
            force_rate,
            contact_blend,
            self._normalized_action,
        )

    def _prepare_policy_context(
        self,
        state: FrankaState,
        nominal_wrench: np.ndarray,
    ) -> bool:
        del state, nominal_wrench
        return True

    def _residual_context_available(self, state: FrankaState) -> bool:
        del state
        return True

    def _project_residual(
        self,
        state: FrankaState,
        nominal_wrench: np.ndarray,
    ) -> None:
        del state, nominal_wrench

    def _constrain_total_normal_residual(self, nominal_wrench: np.ndarray) -> None:
        nominal_normal = float(self.normal @ nominal_wrench[:3])
        residual_normal = float(self.normal @ self._residual)
        if self.min_total_normal_wrench <= nominal_normal <= self.max_total_normal_wrench:
            safe_residual_normal = float(
                np.clip(
                    residual_normal,
                    self.min_total_normal_wrench - nominal_normal,
                    self.max_total_normal_wrench - nominal_normal,
                )
            )
        else:
            safe_residual_normal = 0.0
        self._residual += self.normal * (safe_residual_normal - residual_normal)

    def compute(self, state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray:
        safe_dt = max(0.0, dt)
        nominal_wrench = self.nominal.compute(state, target, dt)
        normal_force, force_rate, contact_blend = self._nominal_feedback(state, safe_dt)
        context_ok = self._residual_context_available(state)
        if not context_ok:
            self._desired_residual = np.zeros(3)
            self._residual = np.zeros(3)
            self._normalized_action = np.zeros(3)
        if contact_blend >= 0.99:
            self._contact_ready_elapsed += safe_dt
        else:
            self._contact_ready_elapsed = 0.0
            self._desired_residual = np.zeros(3)
            self._residual = np.zeros(3)

        self._policy_elapsed += safe_dt
        policy_ok = True
        if context_ok and self._policy_elapsed + 1.0e-12 >= self.policy_period:
            self._policy_elapsed %= self.policy_period
            policy_ok = self._prepare_policy_context(state, nominal_wrench)
            if policy_ok:
                policy_ok = self._update_policy(
                    state,
                    target,
                    normal_force,
                    force_rate,
                    contact_blend,
                    nominal_wrench,
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

        self._constrain_total_normal_residual(nominal_wrench)
        self._project_residual(state, nominal_wrench)
        wrench = nominal_wrench.copy()
        wrench[:3] += self._residual
        self._residual_squared_sum += float(self._residual @ self._residual)
        self._residual_sample_count += 1
        return wrench


@dataclass
class TorqueProjectedResidualController(BoundedResidualController):
    """Bounded residual control with directional torque headroom and ray projection."""

    policy: ResidualPolicy = field(
        default_factory=lambda: LinearResidualPolicy.zero(TORQUE_AWARE_OBSERVATION_NAMES)
    )
    nominal: FrankaController = field(default_factory=FrankaSafeAdaptiveController)
    torque_reserve_fraction: float = 0.10
    name: str = "torque_projected_residual_rl"

    _torque_headroom: np.ndarray = field(
        default_factory=lambda: np.zeros(6), init=False, repr=False
    )
    _last_torque_projection_scale: float = field(default=1.0, init=False, repr=False)
    _torque_projection_count: int = field(default=0, init=False, repr=False)
    _torque_projection_samples: int = field(default=0, init=False, repr=False)
    _torque_projection_scale_sum: float = field(default=0.0, init=False, repr=False)
    _torque_context_fallback_count: int = field(default=0, init=False, repr=False)

    def _expected_policy_observation_names(self) -> tuple[str, ...]:
        return TORQUE_AWARE_OBSERVATION_NAMES

    def _policy_schema_error(self) -> str:
        return "torque-aware policy observation schema does not match"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.torque_reserve_fraction < 1.0:
            raise ValueError("torque_reserve_fraction must be in [0, 1)")

    @property
    def last_torque_headroom(self) -> np.ndarray:
        return self._torque_headroom.copy()

    @property
    def last_torque_projection_scale(self) -> float:
        return self._last_torque_projection_scale

    @property
    def torque_projection_pct(self) -> float:
        if self._torque_projection_samples == 0:
            return 0.0
        return 100.0 * self._torque_projection_count / self._torque_projection_samples

    @property
    def mean_torque_projection_scale(self) -> float:
        if self._torque_projection_samples == 0:
            return 1.0
        return self._torque_projection_scale_sum / self._torque_projection_samples

    @property
    def torque_context_fallback_count(self) -> int:
        return self._torque_context_fallback_count

    def reset(self, state: FrankaState) -> None:
        super().reset(state)
        self._torque_headroom = np.zeros(6)
        self._last_torque_projection_scale = 1.0
        self._torque_projection_count = 0
        self._torque_projection_samples = 0
        self._torque_projection_scale_sum = 0.0
        self._torque_context_fallback_count = 0

    def _prepare_policy_context(
        self,
        state: FrankaState,
        nominal_wrench: np.ndarray,
    ) -> bool:
        assert state.actuation is not None
        self._torque_headroom = residual_torque_headroom(
            state.actuation,
            nominal_wrench,
            self.action_bounds,
            self.torque_reserve_fraction,
        )
        return True

    def _residual_context_available(self, state: FrankaState) -> bool:
        if state.actuation is not None:
            return True
        self._torque_headroom = np.zeros(6)
        self._torque_context_fallback_count += 1
        return False

    def _encode_policy_observation(
        self,
        state: FrankaState,
        target: FrankaTarget,
        normal_force: float,
        force_rate: float,
        contact_blend: float,
        nominal_wrench: np.ndarray,
    ) -> np.ndarray:
        observation = super()._encode_policy_observation(
            state,
            target,
            normal_force,
            force_rate,
            contact_blend,
            nominal_wrench,
        )
        return np.concatenate((observation, self._torque_headroom))

    def _project_residual(
        self,
        state: FrankaState,
        nominal_wrench: np.ndarray,
    ) -> None:
        projection = project_residual_force(
            state.actuation,
            nominal_wrench,
            self._residual,
            self.torque_reserve_fraction,
        )
        candidate_was_active = bool(np.linalg.norm(self._residual) > 1.0e-12)
        self._residual = projection.residual_force
        self._last_torque_projection_scale = projection.scale
        if candidate_was_active:
            self._torque_projection_samples += 1
            self._torque_projection_scale_sum += projection.scale
            if projection.status == "scaled":
                self._torque_projection_count += 1
        self._normalized_action = np.divide(
            self._residual,
            self.action_bounds,
            out=np.zeros(3),
            where=self.action_bounds > 0.0,
        )
