"""Stateful normal-reference shaping for the classical Franka controller."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.franka_control import FrankaState, FrankaTarget


@dataclass
class FrankaRateLimitedAdaptiveController(FrankaSafeAdaptiveController):
    """Bound normal reference speed and acceleration before adaptive hybrid control.

    Reference position starts at the measured position on reset, with zero reference
    velocity. Each update uses ``p_next = p + dt * v_next`` and sends that next
    reference to the base controller during the same call. The position target drives
    normal motion; its normal velocity is replaced by this consistent discrete derivative.
    Tangential pose/velocity, orientation and force targets are preserved.

    The inherited lead limit and impact guard set a moving position goal, not a hard
    reference-position clamp. A sudden measurement change or retreat can therefore
    temporarily exceed the lead goal while the reference brakes continuously. Measured
    forward speed above the reference speed limit also requests braking. These limits
    constrain the reference, not actual robot speed, contact force or passivity.
    """

    max_approach_acceleration: float = 0.10
    name: str = "rate_limited_adaptive_hybrid"

    _reference_normal_position: float | None = field(default=None, init=False, repr=False)
    _reference_normal_velocity: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "max_normal_lead",
            "max_approach_velocity",
            "max_approach_acceleration",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("impact_force_margin", "impact_force_rate"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite and nonnegative")
        self._normal_direction()

    def _normal_direction(self) -> np.ndarray:
        normal = np.asarray(self.base.base.normal, dtype=float)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("normal must be a finite nonzero 3-vector")
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("normal must be a finite nonzero 3-vector")
        return normal / norm

    @property
    def reference_normal_position_m(self) -> float | None:
        return self._reference_normal_position

    @property
    def reference_normal_velocity_m_s(self) -> float:
        return self._reference_normal_velocity

    def reset(self, state: FrankaState) -> None:
        normal = self._normal_direction()
        super().reset(state)
        self._reference_normal_position = float(normal @ state.position)
        self._reference_normal_velocity = 0.0

    def _govern_target(self, state: FrankaState, target: FrankaTarget) -> FrankaTarget:
        # compute already prepared a dynamically consistent target. Applying the
        # parent's instantaneous position clamp here would break its derivative.
        normal = self._normal_direction()
        self._last_governed_normal_lead = float(normal @ (target.position - state.position))
        return target

    def compute(self, state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self._reference_normal_position is None:
            self.reset(state)
        assert self._reference_normal_position is not None

        normal = self._normal_direction()
        measured_position = float(normal @ state.position)
        target_position = float(normal @ target.position)
        goal = min(target_position, measured_position + self.max_normal_lead)
        # As in the original safe wrapper, these estimates precede this update.
        impact_guard = (
            self.corrected_force_n > target.normal_force + self.impact_force_margin
            or self.filtered_force_rate_n_s > self.impact_force_rate
        )
        if impact_guard and self.contact_blend < 0.99:
            goal = min(goal, measured_position)

        distance = goal - self._reference_normal_position
        acceleration_step = self.max_approach_acceleration * dt
        # Leave room for the next integration step and subsequent braking. This
        # conservative stopping envelope avoids snapping position at the goal.
        braking_term = 2.0 * self.max_approach_acceleration * abs(distance)
        stopping_speed = braking_term / (
            np.sqrt(acceleration_step**2 + braking_term) + acceleration_step
        )
        desired_velocity = float(
            np.sign(distance) * min(self.max_approach_velocity, stopping_speed)
        )
        if float(normal @ state.linear_velocity) > self.max_approach_velocity:
            desired_velocity = min(desired_velocity, 0.0)

        self._reference_normal_velocity += float(
            np.clip(
                desired_velocity - self._reference_normal_velocity,
                -acceleration_step,
                acceleration_step,
            )
        )
        self._reference_normal_position += dt * self._reference_normal_velocity
        governed_target = replace(
            target,
            position=target.position + normal * (self._reference_normal_position - target_position),
            linear_velocity=target.linear_velocity
            + normal * (self._reference_normal_velocity - float(normal @ target.linear_velocity)),
        )
        return super().compute(state, governed_target, dt)
