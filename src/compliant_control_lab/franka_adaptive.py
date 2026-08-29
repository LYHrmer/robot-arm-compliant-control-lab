"""Adaptive classical baseline for Franka compliant contact control."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from compliant_control_lab.franka_control import (
    FrankaHybridController,
    FrankaState,
    FrankaTarget,
)


@dataclass
class FrankaAdaptiveHybridController:
    """Bias-aware hybrid control with contact-stiffness and error gain scheduling.

    The module wraps the existing impact-aware hybrid controller. It estimates force-sensor bias
    while the target is still out of contact, estimates local normal contact stiffness from measured
    force/displacement increments, and schedules only classical gains. A force-rate guard provides
    extra damping during impact without changing the controller seam.
    """

    base: FrankaHybridController = field(
        default_factory=lambda: FrankaHybridController(force_transition_time=0.50)
    )
    bias_time_constant: float = 0.08
    force_rate_time_constant: float = 0.03
    stiffness_time_constant: float = 0.15
    reference_contact_stiffness: float = 4_000.0
    min_contact_stiffness: float = 500.0
    max_contact_stiffness: float = 30_000.0
    min_force_gain_scale: float = 0.65
    max_force_gain_scale: float = 1.10
    max_force_bias: float = 3.0
    max_retreat_command: float = 2.0
    name: str = "adaptive_hybrid"

    _force_bias: float = field(default=0.0, init=False, repr=False)
    _corrected_force: float = field(default=0.0, init=False, repr=False)
    _filtered_force_rate: float = field(default=0.0, init=False, repr=False)
    _contact_stiffness: float = field(default=4_000.0, init=False, repr=False)
    _force_gain_scale: float = field(default=1.0, init=False, repr=False)
    _previous_force: float | None = field(default=None, init=False, repr=False)
    _previous_normal_position: float | None = field(default=None, init=False, repr=False)
    _nominal_force_kp: float = field(init=False, repr=False)
    _nominal_force_ki: float = field(init=False, repr=False)
    _nominal_normal_damping: float = field(init=False, repr=False)
    _nominal_approach_stiffness: float = field(init=False, repr=False)
    _nominal_approach_damping: float = field(init=False, repr=False)
    _nominal_max_approach_command: float = field(init=False, repr=False)
    _nominal_tangential_stiffness: np.ndarray = field(init=False, repr=False)
    _nominal_tangential_damping: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bias_time_constant <= 0.0:
            raise ValueError("bias_time_constant must be positive")
        if self.force_rate_time_constant <= 0.0:
            raise ValueError("force_rate_time_constant must be positive")
        if self.stiffness_time_constant <= 0.0:
            raise ValueError("stiffness_time_constant must be positive")
        if not 0.0 < self.min_force_gain_scale <= self.max_force_gain_scale:
            raise ValueError("force gain scale bounds must be positive and ordered")
        if not 0.0 < self.min_contact_stiffness <= self.max_contact_stiffness:
            raise ValueError("contact stiffness bounds must be positive and ordered")
        self._nominal_force_kp = self.base.force_kp
        self._nominal_force_ki = self.base.force_ki
        self._nominal_normal_damping = self.base.normal_damping
        self._nominal_approach_stiffness = self.base.approach_stiffness
        self._nominal_approach_damping = self.base.approach_damping
        self._nominal_max_approach_command = self.base.max_approach_command
        self._nominal_tangential_stiffness = self.base.tangential_stiffness.copy()
        self._nominal_tangential_damping = self.base.tangential_damping.copy()

    @property
    def estimated_force_bias_n(self) -> float:
        return self._force_bias

    @property
    def corrected_force_n(self) -> float:
        return self._corrected_force

    @property
    def filtered_force_rate_n_s(self) -> float:
        return self._filtered_force_rate

    @property
    def estimated_contact_stiffness_n_m(self) -> float:
        return self._contact_stiffness

    @property
    def force_gain_scale(self) -> float:
        return self._force_gain_scale

    @property
    def contact_blend(self) -> float:
        return self.base.force_blend

    def reset(self, state: FrankaState) -> None:
        self.base.force_kp = self._nominal_force_kp
        self.base.force_ki = self._nominal_force_ki
        self.base.normal_damping = self._nominal_normal_damping
        self.base.approach_stiffness = self._nominal_approach_stiffness
        self.base.approach_damping = self._nominal_approach_damping
        self.base.max_approach_command = self._nominal_max_approach_command
        self.base.tangential_stiffness = self._nominal_tangential_stiffness.copy()
        self.base.tangential_damping = self._nominal_tangential_damping.copy()
        self.base.reset(state)
        self._force_bias = 0.0
        self._corrected_force = max(0.0, state.normal_force)
        self._filtered_force_rate = 0.0
        self._contact_stiffness = self.reference_contact_stiffness
        self._force_gain_scale = 1.0
        self._previous_force = None
        self._previous_normal_position = None

    def _update_estimates(self, state: FrankaState, target: FrankaTarget, dt: float) -> None:
        safe_dt = max(0.0, dt)
        if target.normal_force <= 2.0 and not self.base.in_contact:
            alpha_bias = safe_dt / (self.bias_time_constant + safe_dt)
            self._force_bias += alpha_bias * (state.normal_force - self._force_bias)
            self._force_bias = float(
                np.clip(self._force_bias, -self.max_force_bias, self.max_force_bias)
            )

        self._corrected_force = max(0.0, state.normal_force - self._force_bias)
        normal_position = float(self.base.normal @ state.position)
        if safe_dt > 0.0 and self._previous_force is not None:
            instantaneous_rate = (self._corrected_force - self._previous_force) / safe_dt
            instantaneous_rate = float(np.clip(instantaneous_rate, -2_000.0, 2_000.0))
            alpha_rate = safe_dt / (self.force_rate_time_constant + safe_dt)
            self._filtered_force_rate += alpha_rate * (
                instantaneous_rate - self._filtered_force_rate
            )

            assert self._previous_normal_position is not None
            displacement_delta = normal_position - self._previous_normal_position
            force_delta = self._corrected_force - self._previous_force
            if (
                self._corrected_force >= self.base.contact_threshold
                and abs(displacement_delta) >= 2.0e-6
                and force_delta * displacement_delta > 0.0
            ):
                stiffness_sample = abs(force_delta / displacement_delta)
                stiffness_sample = float(
                    np.clip(
                        stiffness_sample,
                        self.min_contact_stiffness,
                        self.max_contact_stiffness,
                    )
                )
                alpha_stiffness = safe_dt / (self.stiffness_time_constant + safe_dt)
                self._contact_stiffness += alpha_stiffness * (
                    stiffness_sample - self._contact_stiffness
                )

        self._previous_force = self._corrected_force
        self._previous_normal_position = normal_position

    def _schedule_gains(self, state: FrankaState, target: FrankaTarget) -> None:
        stiffness_scale = np.sqrt(
            self.reference_contact_stiffness / max(self._contact_stiffness, 1.0)
        )
        rate_scale = 1.0 / (1.0 + abs(self._filtered_force_rate) / 250.0)
        self._force_gain_scale = float(
            np.clip(
                stiffness_scale * rate_scale,
                self.min_force_gain_scale,
                self.max_force_gain_scale,
            )
        )
        self.base.force_kp = self._nominal_force_kp * self._force_gain_scale
        self.base.force_ki = self._nominal_force_ki * self._force_gain_scale
        self.base.normal_damping = self._nominal_normal_damping / np.sqrt(
            self._force_gain_scale
        )

        self.base.approach_stiffness = self._nominal_approach_stiffness
        self.base.approach_damping = self._nominal_approach_damping
        self.base.max_approach_command = self._nominal_max_approach_command

        tangent_projector = np.eye(3) - np.outer(self.base.normal, self.base.normal)
        tangent_error = tangent_projector @ (target.position - state.position)
        tangent_scale = 1.0 + 0.25 * float(np.clip(np.linalg.norm(tangent_error) / 0.02, 0, 1))
        self.base.tangential_stiffness = self._nominal_tangential_stiffness * tangent_scale
        self.base.tangential_damping = self._nominal_tangential_damping * np.sqrt(tangent_scale)

    def compute(self, state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray:
        self._update_estimates(state, target, dt)
        self._schedule_gains(state, target)
        corrected_state = replace(state, normal_force=self._corrected_force)
        wrench = self.base.compute(corrected_state, target, dt)

        normal_command = float(self.base.normal @ wrench[:3])
        force_overshoot = max(0.0, self._corrected_force - target.normal_force - 6.0)
        positive_force_rate = max(0.0, self._filtered_force_rate - 150.0)
        normal_command -= 0.35 * force_overshoot + 0.002 * positive_force_rate
        normal_command = float(
            np.clip(
                normal_command,
                -self.max_retreat_command,
                self.base.max_normal_command,
            )
        )
        wrench[:3] += self.base.normal * (normal_command - self.base.normal @ wrench[:3])
        return wrench
