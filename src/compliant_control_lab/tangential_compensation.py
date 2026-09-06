"""Bounded classical force additions, evaluated before nominal torque projection."""

from dataclasses import dataclass, field

import numpy as np

from compliant_control_lab.franka_control import FrankaState, FrankaTarget


def _tangent(normal, vector):
    normal, vector = np.asarray(normal, dtype=float), np.asarray(vector, dtype=float)
    if (
        normal.shape != (3,)
        or vector.shape != (3,)
        or not np.all(np.isfinite(normal))
        or not np.all(np.isfinite(vector))
        or not np.isclose(normal @ normal, 1.0, rtol=0, atol=1e-10)
    ):
        raise ValueError("expected a finite unit normal and finite 3-vector")
    return vector - normal * (normal @ vector)


def bounded_friction_force(
    normal, target_velocity, normal_force, nominal_mu, max_force, velocity_scale
):
    """Smooth nominal Coulomb feedforward, not measured contact-friction cancellation."""
    values = np.asarray([normal_force, nominal_mu, max_force, velocity_scale], dtype=float)
    if not np.all(np.isfinite(values)) or nominal_mu < 0 or max_force <= 0 or velocity_scale <= 0:
        raise ValueError("invalid force, friction coefficient, bound or velocity scale")
    velocity = _tangent(normal, target_velocity)
    amplitude = min(nominal_mu * max(normal_force, 0.0), max_force)
    return amplitude * velocity / np.sqrt(velocity @ velocity + velocity_scale**2)


@dataclass
class TangentialCompensation:
    """Separate I or friction-FF experiments; no ideal plant force is accepted.

    force() uses current integral state. advance() commits the NEXT cycle state
    only after an unchanged torque projection. Saturated/fallback/no-context
    cycles freeze integration. Loss of measured contact clears stored force.
    """

    mode: str
    integral_gain: float = 800.0
    nominal_mu: float = 0.45
    max_force: float = 6.0
    velocity_scale: float = 0.005
    _integral_force: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _last_force: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.mode not in {"integral", "friction"}:
            raise ValueError("mode must be integral or friction")
        values = [self.integral_gain, self.nominal_mu, self.max_force, self.velocity_scale]
        if not np.all(np.isfinite(values)) or min(values[:2]) < 0 or min(values[2:]) <= 0:
            raise ValueError("invalid tangential compensation parameters")

    @property
    def integral_force(self):
        return self._integral_force.copy()

    @property
    def last_force(self):
        return self._last_force.copy()

    def reset(self):
        self._integral_force[:] = 0
        self._last_force[:] = 0
        self._active = False

    def force(
        self,
        state: FrankaState,
        target: FrankaTarget,
        normal,
        normal_force,
        contact_blend,
        in_contact,
    ):
        if not np.all(np.isfinite([normal_force, contact_blend])) or not 0 <= contact_blend <= 1:
            raise ValueError("force and contact blend must be finite and blend within [0, 1]")
        self._active = bool(in_contact and normal_force > 1.0 and target.normal_force > 0)
        if not self._active:
            self.reset()
        elif self.mode == "integral":
            self._last_force = contact_blend * _tangent(normal, self._integral_force)
        else:
            self._last_force = contact_blend * bounded_friction_force(
                normal,
                target.linear_velocity,
                normal_force,
                self.nominal_mu,
                self.max_force,
                self.velocity_scale,
            )
        return self.last_force

    def advance(self, state: FrankaState, target: FrankaTarget, normal, dt, allow_integration):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not (self._active and self.mode == "integral" and allow_integration):
            return
        error = _tangent(normal, target.position - state.position)
        candidate = self._integral_force + self.integral_gain * dt * error
        self._integral_force = candidate * min(
            1.0, self.max_force / max(np.linalg.norm(candidate), 1e-12)
        )
