"""Deterministic phase, direction and speed variants of the surface wiping task."""

from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np

from compliant_control_lab.surface_simulation import SurfaceTask, yaw_frame


@dataclass(frozen=True)
class LearningSurfaceTask(SurfaceTask):
    """Keep the original approach and smoothly start a configurable ellipse.

    Phase changes the ellipse center relative to the initial contact point, not
    just its timing. Arbitrary amplitudes/phases have no workspace safety claim.
    Randomization belongs to episode configuration, never target_at().
    """

    tangent_amplitude_m: float = 0.055
    vertical_amplitude_m: float = 0.040
    frequency_hz: float = 0.20
    phase_rad: float = 0.0
    direction: int = 1

    def __post_init__(self):
        super().__post_init__()
        for name in ("tangent_amplitude_m", "vertical_amplitude_m", "frequency_hz", "phase_rad"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite real number")
            if not np.isfinite(value) or (name != "phase_rad" and value <= 0):
                raise ValueError(f"{name} must be finite and amplitudes/frequency positive")
            object.__setattr__(self, name, float(value))
        if (
            isinstance(self.direction, (bool, np.bool_))
            or not isinstance(self.direction, Integral)
            or self.direction not in (-1, 1)
        ):
            raise ValueError("direction must be the integer -1 or +1")
        object.__setattr__(self, "direction", int(self.direction))

    def target_at(self, time, initial_position, initial_rotation, target_force):
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        target = super().target_at(min(time, 1.2), initial_position, initial_rotation, target_force)
        if time <= 1.2:
            return target
        elapsed, ramp = time - 1.2, 0.5
        u = min(elapsed / ramp, 1.0)
        clock = ramp * (u**3 - 0.5 * u**4) if elapsed < ramp else elapsed - ramp / 2
        omega = self.direction * 2.0 * np.pi * self.frequency_hz
        theta = self.phase_rad + omega * clock
        rate = omega * u**2 * (3.0 - 2.0 * u)
        _, tangent1, tangent2 = yaw_frame(self.yaw_deg).rotation.T
        a, b = self.tangent_amplitude_m, self.vertical_amplitude_m
        position = target.position.copy()
        position += a * (np.sin(theta) - np.sin(self.phase_rad)) * tangent1
        position += b * (np.cos(theta) - np.cos(self.phase_rad)) * tangent2
        velocity = target.linear_velocity + rate * (
            a * np.cos(theta) * tangent1 - b * np.sin(theta) * tangent2
        )
        return replace(target, position=position, linear_velocity=velocity)
