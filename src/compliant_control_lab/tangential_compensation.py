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
    """Separate I, fixed FF or online equivalent-load experiments.

    force() uses current integral state. advance() commits the NEXT cycle state
    only after an unchanged torque projection. Saturated/fallback/no-context
    cycles freeze integration. Loss of measured contact clears stored force.
    """

    mode: str
    integral_gain: float = 800.0
    nominal_mu: float = 0.45
    max_force: float = 6.0
    velocity_scale: float = 0.005
    adaptation_gain: float = 800.0
    velocity_error_time: float = 0.05
    force_regularizer: float = 2.0
    max_equivalent_mu: float = 0.9
    min_update_speed: float = 0.005
    force_slew_rate: float = 20.0
    coefficient_rate_limit: float = 0.3
    motion_confirm_time: float = 0.05
    _integral_force: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _last_force: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)
    _equivalent_mu: float = field(default=0.45, init=False, repr=False)
    _normal_force: float = field(default=0.0, init=False, repr=False)
    _direction: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _previous_direction: np.ndarray = field(
        default_factory=lambda: np.zeros(3), init=False, repr=False
    )
    _update_ready: bool = field(default=False, init=False, repr=False)
    _motion_elapsed: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        if self.mode not in {"integral", "friction", "online"}:
            raise ValueError("mode must be integral, friction or online")
        values = [self.integral_gain, self.nominal_mu, self.max_force, self.velocity_scale]
        if not np.all(np.isfinite(values)) or min(values[:2]) < 0 or min(values[2:]) <= 0:
            raise ValueError("invalid tangential compensation parameters")
        online = [self.adaptation_gain, self.velocity_error_time, self.force_regularizer,
                  self.max_equivalent_mu, self.min_update_speed, self.force_slew_rate,
                  self.coefficient_rate_limit, self.motion_confirm_time]
        if not np.all(np.isfinite(online)) or min(online) <= 0:
            raise ValueError("online compensation parameters must be finite and positive")
        if self.mode == "online" and self.nominal_mu > self.max_equivalent_mu:
            raise ValueError("nominal_mu exceeds the equivalent coefficient bound")
        self._equivalent_mu = self.nominal_mu

    @property
    def equivalent_mu(self):
        """Error-driven load coefficient, not an identified material property."""
        return self._equivalent_mu

    @property
    def update_ready(self):
        """Kinematic/contact readiness; projection must additionally accept the request."""
        return self._update_ready

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
        self._equivalent_mu = self.nominal_mu
        self._normal_force = 0.0
        self._direction[:] = 0
        self._previous_direction[:] = 0
        self._update_ready = False
        self._motion_elapsed = 0.0

    def force(
        self,
        state: FrankaState,
        target: FrankaTarget,
        normal,
        normal_force,
        contact_blend,
        in_contact,
        *,
        dt=None,
    ):
        if not np.all(np.isfinite([normal_force, contact_blend])) or not 0 <= contact_blend <= 1:
            raise ValueError("force and contact blend must be finite and blend within [0, 1]")
        if self.mode == "online" and (dt is None or not np.isfinite(dt) or dt <= 0):
            raise ValueError("online force requires a finite positive dt")
        self._active = bool(in_contact and normal_force > 1.0 and target.normal_force > 0)
        if not self._active:
            self.reset()
        elif self.mode == "integral":
            self._last_force = contact_blend * _tangent(normal, self._integral_force)
        elif self.mode == "online":
            self._online_force(state, target, normal, normal_force, contact_blend, dt)
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

    def _online_force(self, state, target, normal, normal_force, blend, dt):
        velocity = _tangent(normal, target.linear_velocity)
        speed = float(np.linalg.norm(velocity))
        self._direction = velocity / np.sqrt(speed**2 + self.velocity_scale**2)
        self._normal_force = float(normal_force)
        amplitude = min(self._equivalent_mu * normal_force, self.max_force)
        requested = blend * amplitude * self._direction
        previous = _tangent(normal, self._last_force)
        delta = requested - previous
        delta_norm = float(np.linalg.norm(delta))
        slew_limited = delta_norm > self.force_slew_rate * dt
        self._last_force = previous + delta * min(
            1.0, self.force_slew_rate * dt / max(delta_norm, 1e-12)
        )
        reversal = float(self._direction @ self._previous_direction) < 0.0
        measured_velocity = _tangent(normal, state.linear_velocity)
        moving_together = (
            float(measured_velocity @ self._direction) >= self.min_update_speed / 2
        )
        eligible = bool(
            blend >= 0.99 and speed >= self.min_update_speed and moving_together
            and not reversal and not slew_limited
        )
        self._motion_elapsed = self._motion_elapsed + dt if eligible else 0.0
        self._update_ready = self._motion_elapsed >= self.motion_confirm_time
        if speed >= self.min_update_speed:
            self._previous_direction = self._direction.copy()

    def advance(self, state: FrankaState, target: FrankaTarget, normal, dt, allow_integration):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if self._active and self.mode == "online" and allow_integration and self._update_ready:
            error = _tangent(normal, target.position - state.position)
            velocity_error = _tangent(normal, target.linear_velocity - state.linear_velocity)
            drive = float(self._direction @ (error + self.velocity_error_time * velocity_error))
            force = self._normal_force
            increment = (
                dt * self.adaptation_gain * force / (force**2 + self.force_regularizer**2) * drive
            )
            increment = float(np.clip(
                increment, -self.coefficient_rate_limit * dt, self.coefficient_rate_limit * dt
            ))
            # Do not store an increasing coefficient behind the amplitude ceiling.
            if increment > 0 and self._equivalent_mu * force >= self.max_force:
                return
            self._equivalent_mu = float(np.clip(
                self._equivalent_mu + increment, 0.0, self.max_equivalent_mu
            ))
            return
        if not (self._active and self.mode == "integral" and allow_integration):
            return
        error = _tangent(normal, target.position - state.position)
        candidate = self._integral_force + self.integral_gain * dt * error
        self._integral_force = candidate * min(
            1.0, self.max_force / max(np.linalg.norm(candidate), 1e-12)
        )
