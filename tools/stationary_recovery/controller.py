"""Experimental force-cap tracking only at numerical zero tangential speed."""

import numpy as np

from compliant_control_lab.tangential_compensation import _tangent
from tools.load_aware_compensation import LoadAwareCompensation

# Numerical zero, not a tuned physical speed threshold (velocity units: m/s).
NUMERICAL_ZERO_TANGENTIAL_SPEED = 1e-12


class StationaryHoldCapTracking(LoadAwareCompensation):
    """Apply the existing hold release only to stationary tangential targets."""

    def advance(self, state, target, normal, dt, allow_integration):
        # The packet gate rejects stale/missing inputs before set_force_measurement.
        # Save availability now: the inherited advance consumes this packet.
        measured = self._cycle_measurement is not None
        super().advance(state, target, normal, dt, allow_integration)
        if not (self._active and measured and allow_integration and self._normal_force > 1.0):
            return
        if np.linalg.norm(_tangent(normal, target.linear_velocity)) > NUMERICAL_ZERO_TANGENTIAL_SPEED:
            return
        target_mu = min(self._equivalent_mu, self.applied_budget_n / self._normal_force)
        self._equivalent_mu -= min(
            self._equivalent_mu - target_mu, self.coefficient_rate_limit * dt,
        )
