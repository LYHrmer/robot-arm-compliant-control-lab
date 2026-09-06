"""Surface-local residual Interface; callers own the policy clock, never the safety clock."""

from dataclasses import dataclass, replace

import numpy as np

from compliant_control_lab.franka_control import (
    FrankaState,
    FrankaTarget,
    orientation_error,
)
from compliant_control_lab.franka_torque_safety import (
    project_residual_force,
    residual_torque_headroom,
)
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame
from compliant_control_lab.tangential_compensation import bounded_friction_force

OBSERVATION_SCHEMA = "surface_v1"
OBSERVATION_NAMES = (
    "corrected_normal_force",
    "normal_force_error",
    "normal_force_rate",
    "normal_position_error",
    "tangent1_position_error",
    "tangent2_position_error",
    "normal_velocity_error",
    "tangent1_velocity_error",
    "tangent2_velocity_error",
    "target_normal_velocity",
    "target_tangent1_velocity",
    "target_tangent2_velocity",
    "target_normal_force",
    "contact_blend",
    "previous_normal_residual",
    "previous_tangent1_residual",
    "previous_tangent2_residual",
    "normal_positive_headroom",
    "normal_negative_headroom",
    "tangent1_positive_headroom",
    "tangent1_negative_headroom",
    "tangent2_positive_headroom",
    "tangent2_negative_headroom",
    "normal_orientation_error",
    "tangent1_orientation_error",
    "tangent2_orientation_error",
    "normal_angular_velocity_error",
    "tangent1_angular_velocity_error",
    "tangent2_angular_velocity_error",
    "contact_ready_elapsed",
    "nominal_normal_force",
    "nominal_tangent1_force",
    "nominal_tangent2_force",
)
OBSERVATION_UNITS = (
    *("N", "N", "N/s"),
    *("m",) * 3,
    *("m/s",) * 6,
    "N",
    "1",
    *("N",) * 3,
    *("1",) * 6,
    *("rad",) * 3,
    *("rad/s",) * 3,
    "s",
    *("N",) * 3,
)
# Physical value / fixed scale, then clip. These do not change with configured bounds.
OBSERVATION_SCALES = (
    20.0,
    12.0,
    250.0,
    0.02,
    0.05,
    0.05,
    *(0.1,) * 6,
    12.0,
    1.0,
    4.0,
    6.0,
    6.0,
    *(1.0,) * 6,
    *(0.2,) * 3,
    *(1.0,) * 3,
    0.1,
    25.0,
    25.0,
    25.0,
)
OBSERVATION_CLIP = 3.0
OBSERVATION_DIM = len(OBSERVATION_NAMES)
ACTION_DIM = 3


@dataclass(frozen=True)
class SurfaceResidualConfig:
    action_bounds_n: tuple[float, float, float] = (4.0, 6.0, 6.0)
    action_rate_limits_n_s: tuple[float, float, float] = (40.0, 60.0, 60.0)
    filter_time_constant: float = 0.04
    torque_reserve_fraction: float = 0.1
    residual_enable_delay: float = 0.1
    force_guard_margin: float = 3.0
    force_guard_rate: float = 150.0
    min_total_normal_wrench: float = -4.0
    max_total_normal_wrench: float = 25.0
    teacher_nominal_mu: float = 0.45
    teacher_max_force_n: float = 6.0
    teacher_velocity_scale: float = 0.005

    def __post_init__(self):
        for name in ("action_bounds_n", "action_rate_limits_n_s"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"{name} must be a finite positive 3-vector")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        scalars = [v for name, v in vars(self).items() if not name.startswith("action_")]
        if not np.all(np.isfinite(scalars)):
            raise ValueError("residual parameters must be finite")
        if (
            min(self.filter_time_constant, self.teacher_max_force_n, self.teacher_velocity_scale)
            <= 0
        ):
            raise ValueError("filter and teacher bounds/scales must be positive")
        if (
            min(
                self.residual_enable_delay,
                self.force_guard_margin,
                self.force_guard_rate,
                self.teacher_nominal_mu,
            )
            < 0
        ):
            raise ValueError("delays, guards and nominal friction must be nonnegative")
        if (
            not 0 <= self.torque_reserve_fraction < 1
            or self.min_total_normal_wrench >= self.max_total_normal_wrench
        ):
            raise ValueError("invalid torque reserve or normal wrench interval")


DEFAULT_RESIDUAL_CONFIG = SurfaceResidualConfig()


@dataclass(frozen=True)
class SurfaceResidualCommand:
    """Force stages are LOCAL, nominal_wrench is WORLD; raw action may document invalid input.

    requested_force is after clipping/contact/force guards; filtered_force is
    after low-pass/slew but before normal-wrench and torque projection.
    Emergency clearing/projection can exceed the ordinary slew bound. This is a
    command envelope, not a bound on measured motion/contact or a passivity claim.
    """

    raw_action: np.ndarray
    requested_force: np.ndarray
    filtered_force: np.ndarray
    applied_force: np.ndarray
    projection_scale: float
    reasons: tuple[str, ...]
    nominal_wrench: np.ndarray

    def __post_init__(self):
        for name in (
            "raw_action",
            "requested_force",
            "filtered_force",
            "applied_force",
            "nominal_wrench",
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float).copy())


def friction_teacher_action(observation, config: SurfaceResidualConfig = DEFAULT_RESIDUAL_CONFIG):
    """Feedback-only teacher for adaptive nominal, not a truth-aware friction estimator."""
    observation = np.asarray(observation, dtype=np.float64)
    if observation.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(observation)):
        raise ValueError("teacher expects finite surface_v1 observation")
    physical = observation * np.asarray(OBSERVATION_SCALES)
    force = bounded_friction_force(
        np.array([1.0, 0.0, 0.0]),
        physical[9:12],
        physical[0],
        config.teacher_nominal_mu,
        config.teacher_max_force_n,
        config.teacher_velocity_scale,
    )
    return np.clip(force / np.asarray(config.action_bounds_n), -1.0, 1.0)


class SurfaceResidualController:
    """Two-phase 500 Hz control, with externally held normalized surface-local actions.

    reset -> prepare -> apply is mandatory. prepare advances nominal exactly once;
    apply consumes that prepared sample. No policy inference, timing, simulator,
    ideal force, or hidden scenario parameters enter this Module.
    """

    name = "surface_residual"

    def __init__(
        self, frame: SurfaceFrame, nominal_kind="friction", config=DEFAULT_RESIDUAL_CONFIG
    ):
        if nominal_kind not in {"friction", "adaptive"}:
            raise ValueError("nominal_kind must be friction or adaptive")
        self.frame, self.config, self.nominal_kind = frame, config, nominal_kind
        self._nominal = SurfaceAdaptiveController(
            frame, tangential_mode="friction" if nominal_kind == "friction" else None
        )
        self._pending = None
        self._initialized = False
        self._applied = np.zeros(3)
        self._ready_elapsed = 0.0
        self._last_command = None

    @property
    def nominal(self):
        return self._nominal

    @property
    def last_command(self):
        return replace(self._last_command) if self._last_command is not None else None

    def reset(self, state: FrankaState):
        self._nominal.reset(state)
        self._pending = None
        self._applied = np.zeros(3)
        self._ready_elapsed = 0.0
        self._last_command = None
        self._initialized = True

    def prepare(self, state: FrankaState, target: FrankaTarget, dt: float):
        if self._pending is not None:
            raise RuntimeError("apply must consume the previous prepared sample")
        if not self._initialized:
            raise RuntimeError("reset is required before prepare")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        rotation = self.frame.rotation
        error = self.frame.vector_to_local(target.position - state.position)
        velocity_error = self.frame.vector_to_local(target.linear_velocity - state.linear_velocity)
        target_velocity = self.frame.vector_to_local(target.linear_velocity)
        angular_error = self.frame.vector_to_local(target.angular_velocity - state.angular_velocity)
        for matrix in (state.rotation, target.rotation):
            SurfaceFrame(matrix)  # Validate before advancing nominal state.
        if not np.all(np.isfinite([state.normal_force, target.normal_force])):
            raise ValueError("normal forces must be finite")
        orientation = self.frame.vector_to_local(orientation_error(state.rotation, target.rotation))
        context = state.actuation
        if context is not None:
            context = replace(
                context,
                cartesian_jacobian=np.concatenate(
                    (
                        rotation.T @ context.cartesian_jacobian[:3],
                        rotation.T @ context.cartesian_jacobian[3:],
                    )
                ),
            )
        world_nominal = self._nominal.compute(state, target, dt)
        local_nominal = self.frame.wrench_to_local(world_nominal)
        force, rate, blend = (
            self._nominal.corrected_force_n,
            self._nominal.filtered_force_rate_n_s,
            self._nominal.contact_blend,
        )
        previous_applied = self._applied.copy()
        contact = bool(blend >= 0.99 and force > 1.0 and target.normal_force > 0.0)
        self._ready_elapsed = (
            min(self.config.residual_enable_delay, self._ready_elapsed + dt) if contact else 0.0
        )
        if not contact:
            self._applied[:] = 0.0
        headroom = residual_torque_headroom(
            context,
            local_nominal,
            np.asarray(self.config.action_bounds_n),
            self.config.torque_reserve_fraction,
        )
        values = np.array(
            [
                force,
                target.normal_force - force,
                rate,
                *error,
                *velocity_error,
                *target_velocity,
                target.normal_force,
                blend,
                *previous_applied,
                *headroom,
                *orientation,
                *angular_error,
                self._ready_elapsed,
                *local_nominal[:3],
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("nonfinite prepared observation")
        self._pending = (
            world_nominal.copy(),
            local_nominal,
            context,
            force,
            rate,
            target.normal_force,
            dt,
            contact,
        )
        return np.clip(values / np.asarray(OBSERVATION_SCALES), -OBSERVATION_CLIP, OBSERVATION_CLIP)

    def apply(self, action):
        if self._pending is None:
            raise RuntimeError("prepare is required before apply")
        world_nominal, local_nominal, context, force, rate, target_force, dt, contact = (
            self._pending
        )
        self._pending = None
        config, reasons = self.config, []
        try:
            raw = np.asarray(action, dtype=np.float64).copy()
        except (TypeError, ValueError):
            raw = np.full(3, np.nan)
        valid = raw.shape == (3,) and np.all(np.isfinite(raw))
        requested = np.zeros(3)
        if valid:
            clipped = np.clip(raw, -1.0, 1.0)
            requested = clipped * np.asarray(config.action_bounds_n)
            if not np.array_equal(raw, clipped):
                reasons.append("action_clipped")
        else:
            reasons.append("invalid_action")
        ready = contact and self._ready_elapsed + 1e-12 >= config.residual_enable_delay
        if not contact:
            reasons.append("contact_lost")
        elif not ready:
            reasons.append("contact_not_ready")
        if not valid or not ready:
            requested[:] = 0.0
            self._applied[:] = 0.0
        guard = force > target_force + config.force_guard_margin or rate > config.force_guard_rate
        if guard:
            requested[0] = min(0.0, requested[0])
            self._applied[0] = min(0.0, self._applied[0])
            reasons.append("force_guard")
        alpha = dt / (config.filter_time_constant + dt)
        delta = alpha * (requested - self._applied)
        limit = dt * np.asarray(config.action_rate_limits_n_s)
        filtered = self._applied + np.clip(delta, -limit, limit)
        if np.any(np.abs(delta) > limit):
            reasons.append("slew_limited")
        nominal_normal = local_nominal[0]
        if config.min_total_normal_wrench <= nominal_normal <= config.max_total_normal_wrench:
            safe_normal = np.clip(
                filtered[0],
                config.min_total_normal_wrench - nominal_normal,
                config.max_total_normal_wrench - nominal_normal,
            )
        else:
            safe_normal = 0.0
        if safe_normal != filtered[0]:
            reasons.append("normal_wrench_limit")
        candidate = filtered.copy()
        candidate[0] = safe_normal
        projection = project_residual_force(
            context, local_nominal, candidate, config.torque_reserve_fraction
        )
        self._applied = projection.residual_force
        if projection.status != "unchanged":
            reasons.append(f"torque_{projection.status}")
        self._last_command = SurfaceResidualCommand(
            raw, requested, filtered, self._applied, projection.scale, tuple(reasons), world_nominal
        )
        # Avoid rotation round trips of nominal: zero residual preserves every bit.
        wrench = world_nominal.copy()
        wrench[:3] += self.frame.vector_to_world(self._applied)
        return wrench
