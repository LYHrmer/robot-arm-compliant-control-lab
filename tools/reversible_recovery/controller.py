"""Experimental auxiliary-state cap tracking at numerical zero tangential speed."""

import numpy as np

from compliant_control_lab.tangential_compensation import _tangent
from tools.load_aware_compensation import LoadAwareCompensation

# Numerical zero, not a tuned physical speed threshold (velocity units: m/s).
NUMERICAL_ZERO_TANGENTIAL_SPEED = 1e-12


class ReversibleHoldCapTracking(LoadAwareCompensation):
    """Track the scheduled cap while holding, bounded by the coefficient at hold entry.

    The entry value is auxiliary bookkeeping for one stationary hold, not an
    identified friction property. It caps the coefficient from above and is
    dropped as soon as the hold ends, so motion cycles keep the inherited law.
    """

    def reset(self):
        super().reset()
        self._hold_entry_mu = None

    @property
    def hold_entry_mu(self):
        """Coefficient captured on the first stationary cycle of the current hold."""
        return self._hold_entry_mu

    def advance(self, state, target, normal, dt, allow_integration):
        # The packet gate rejects stale/missing inputs before set_force_measurement.
        # Save availability now: the inherited advance consumes this packet.
        measured = self._cycle_measurement is not None
        super().advance(state, target, normal, dt, allow_integration)
        stationary = bool(
            np.linalg.norm(_tangent(normal, target.linear_velocity))
            <= NUMERICAL_ZERO_TANGENTIAL_SPEED
        )
        if not (self._active and stationary):
            self._hold_entry_mu = None
            return
        # min_update_speed (0.005) keeps update_ready False while stationary, so the
        # inherited advance cannot have moved the coefficient: capturing after it is
        # causally the same value as capturing before it.
        if self._hold_entry_mu is None:
            self._hold_entry_mu = self._equivalent_mu
        entry = self._hold_entry_mu
        # Missing or rejected cycles keep the entry value and only freeze the update.
        if not (measured and allow_integration and self._normal_force > 1.0):
            return
        target_mu = min(entry, self.applied_budget_n / self._normal_force)
        increment = float(np.clip(
            target_mu - self._equivalent_mu,
            -self.coefficient_rate_limit * dt,
            self.coefficient_rate_limit * dt,
        ))
        self._equivalent_mu = float(np.clip(
            self._equivalent_mu + increment, 0.0, min(entry, self.max_equivalent_mu),
        ))
