"""Experimental measured-load ceiling; the production controller is unchanged."""

from dataclasses import dataclass, field

import numpy as np

from compliant_control_lab.tangential_compensation import TangentialCompensation, _tangent


@dataclass
class LoadAwareCompensation(TangentialCompensation):
    """Schedule next-cycle capacity without replacing the error-driven update law.

    Force inputs are delayed/filtered robot-on-environment measurements in the
    controller frame. They are not true friction. A falling scheduled ceiling
    changes the request, not the existing slew-limited force instantaneously.
    """

    minimum_force: float = 6.0
    load_margin: float = 0.25
    load_time_constant: float = 0.20
    _load_estimate: float = field(default=0.0, init=False, repr=False)
    _next_budget: float = field(default=6.0, init=False, repr=False)
    _applied_budget: float = field(default=6.0, init=False, repr=False)
    _pending_measurement: object = field(default=None, init=False, repr=False)
    _cycle_measurement: object = field(default=None, init=False, repr=False)
    _measurement_used: bool = field(default=False, init=False, repr=False)
    _projected_load: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        values = [self.minimum_force, self.load_margin, self.load_time_constant]
        if (self.mode != "online" or not np.all(np.isfinite(values))
                or not 0 < self.minimum_force <= self.max_force
                or self.load_margin < 0 or self.load_time_constant <= 0):
            raise ValueError("invalid online load-budget parameters")
        self.reset()

    @property
    def load_estimate_n(self):
        return self._load_estimate

    @property
    def next_budget_n(self):
        return self._next_budget

    @property
    def applied_budget_n(self):
        return self._applied_budget

    @property
    def measurement_used_for_budget(self):
        return self._measurement_used

    @property
    def projected_load_n(self):
        return self._projected_load

    def reset(self):
        super().reset()
        self._load_estimate = 0.0
        self._next_budget = self.minimum_force
        self._applied_budget = self.minimum_force
        self._pending_measurement = None
        self._cycle_measurement = None
        self._measurement_used = False
        self._projected_load = 0.0

    def set_force_measurement(self, local_force):
        # Clear first: rejecting a new value must not leave a stale value queued.
        self._pending_measurement = None
        value = np.asarray(local_force, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("measured local force must be a finite 3-vector")
        self._pending_measurement = value.copy()

    def force(self, state, target, normal, normal_force, contact_blend, in_contact, *, dt=None):
        self._cycle_measurement = self._pending_measurement
        self._pending_measurement = None
        self._measurement_used = False
        self._projected_load = 0.0
        if self._cycle_measurement is None:
            self._load_estimate = 0.0
            self._next_budget = self.minimum_force
        self._applied_budget = self._next_budget
        global_ceiling = self.max_force
        try:
            self.max_force = self._applied_budget
            return super().force(
                state, target, normal, normal_force, contact_blend, in_contact, dt=dt,
            )
        finally:
            self.max_force = global_ceiling

    def advance(self, state, target, normal, dt, allow_integration):
        global_ceiling = self.max_force
        measurement = self._cycle_measurement
        self._cycle_measurement = None
        try:
            self.max_force = self._applied_budget
            super().advance(state, target, normal, dt, allow_integration)
        finally:
            self.max_force = global_ceiling
        if not (self._active and self._update_ready and allow_integration
                and measurement is not None):
            return
        velocity = _tangent(normal, target.linear_velocity)
        speed = float(np.linalg.norm(velocity))
        if speed < self.min_update_speed:
            return
        self._projected_load = float(np.clip(measurement @ (velocity / speed), 0, global_ceiling))
        alpha = -np.expm1(-dt / self.load_time_constant)
        self._load_estimate += float(alpha * (self._projected_load - self._load_estimate))
        self._next_budget = float(np.clip(
            self._load_estimate + self.load_margin, self.minimum_force, global_ceiling,
        ))
        self._measurement_used = True
