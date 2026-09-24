"""Independent recorded-input properties; these are not robot rollouts."""

import numpy as np
import pytest

from tests.test_stationary_recovery_controller import DT, _cycle
from tools.reversible_recovery.controller import ReversibleHoldCapTracking
from tools.stationary_recovery.controller import StationaryHoldCapTracking


@pytest.mark.parametrize("seed", [11, 29, 103])
@pytest.mark.parametrize("budget", [6.0, 7.68])
@pytest.mark.parametrize("entry", [0.0, 0.45, 0.9])
def test_identical_hold_inputs_preserve_bounds_and_cannot_add_downward_memory(seed, budget, entry):
    """Restoration cannot imply closed-loop superiority: physical inputs are fixed."""
    options = {"nominal_mu": entry, "minimum_force": budget, "max_force": 8.0}
    candidate = ReversibleHoldCapTracking("online", **options)
    prior = StationaryHoldCapTracking("online", **options)
    rng = np.random.default_rng(seed)
    limit = candidate.coefficient_rate_limit * DT
    for _ in range(500):
        normal = float(rng.uniform(6.0, 18.0))
        measured, accepted = bool(rng.random() > 0.15), bool(rng.random() > 0.2)
        before = candidate.equivalent_mu
        expected = before
        if measured and accepted:
            expected += float(np.clip(min(entry, budget / normal) - before, -limit, limit))
        inputs = {"normal_force": normal, "measurement": measured, "accepted": accepted}
        actual_force = _cycle(candidate, **inputs)
        prior_force = _cycle(prior, **inputs)
        np.testing.assert_array_equal(actual_force, np.zeros(3))
        np.testing.assert_array_equal(actual_force, prior_force)
        assert candidate.equivalent_mu == pytest.approx(expected, abs=1e-14, rel=0)
        assert 0.0 <= candidate.equivalent_mu <= entry + 1e-14
        assert abs(candidate.equivalent_mu - before) <= limit + 1e-14
        assert candidate.equivalent_mu + 1e-14 >= prior.equivalent_mu
        assert candidate.applied_budget_n == prior.applied_budget_n == budget
        assert not candidate.update_ready


def test_restoration_is_internal_state_not_a_stationary_contact_force_command():
    candidate = ReversibleHoldCapTracking("online", nominal_mu=0.5, max_force=8.0)
    prior = StationaryHoldCapTracking("online", nominal_mu=0.5, max_force=8.0)
    candidate_states, prior_states = [], []
    for normal in (12.0, 14.0, 12.0):
        for controller, states in ((candidate, candidate_states), (prior, prior_states)):
            np.testing.assert_array_equal(_cycle(controller, normal_force=normal), np.zeros(3))
            states.append(controller.equivalent_mu)
    np.testing.assert_allclose(candidate_states, [0.5, 0.4994, 0.5], atol=1e-14, rtol=0)
    np.testing.assert_allclose(prior_states, [0.5, 0.4994, 0.4994], atol=1e-14, rtol=0)
