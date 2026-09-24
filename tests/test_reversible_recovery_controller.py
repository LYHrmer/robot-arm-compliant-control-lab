"""Controller-only regression checks; these tests do not integrate robot dynamics."""

from dataclasses import replace

import numpy as np
import pytest

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from tests.test_stationary_recovery_controller import DT, NORMAL, _cycle, _sample
from tests.test_tangential_safety import ControlledAdaptive, context
from tools.load_aware_compensation import LoadAwareCompensation
from tools.reversible_recovery.controller import ReversibleHoldCapTracking
from tools.stationary_recovery.controller import StationaryHoldCapTracking


def _new(mu=0.5, **kwargs):
    return ReversibleHoldCapTracking("online", nominal_mu=mu, max_force=8.0, **kwargs)


def test_normal_load_transient_recovers_only_the_entry_coefficient():
    candidate, old = _new(), StationaryHoldCapTracking("online", nominal_mu=0.5, max_force=8.0)
    actual, original = [], []
    for normal_force in (12.0, 14.0, 12.0, 8.0):
        np.testing.assert_array_equal(_cycle(candidate, normal_force=normal_force),
                                      _cycle(old, normal_force=normal_force))
        actual.append(candidate.equivalent_mu)
        original.append(old.equivalent_mu)
        assert candidate.equivalent_mu <= candidate.hold_entry_mu == 0.5
    np.testing.assert_allclose(actual, [0.5, 0.4994, 0.5, 0.5], atol=1e-14, rtol=0)
    np.testing.assert_allclose(original, [0.5, 0.4994, 0.4994, 0.4994], atol=1e-14, rtol=0)


def test_equal_peaks_at_different_hold_times_recover_the_same_boundary():
    final = []
    for peak_cycle in (6, 70):
        candidate = _new(0.66, minimum_force=7.68)
        for index in range(100):
            _cycle(candidate, normal_force=12.25 if index == peak_cycle else 12.0)
        final.append(candidate.equivalent_mu)
    np.testing.assert_array_equal(final, [0.64, 0.64])


@pytest.mark.parametrize("nominal_mu", [0.45, 0.5002, 0.62])
@pytest.mark.parametrize("normal_force", [10.0, 12.0, 14.0])
def test_constant_inputs_are_exactly_the_old_stationary_rule(nominal_mu, normal_force):
    candidate = _new(nominal_mu)
    old = StationaryHoldCapTracking("online", nominal_mu=nominal_mu, max_force=8.0)
    for _ in range(250):
        np.testing.assert_array_equal(_cycle(candidate, normal_force=normal_force),
                                      _cycle(old, normal_force=normal_force))
        for name in ("equivalent_mu", "update_ready", "applied_budget_n", "next_budget_n",
                     "load_estimate_n", "measurement_used_for_budget"):
            assert getattr(candidate, name) == getattr(old, name)


@pytest.mark.parametrize("velocity", [(0.0, 1.0001e-12, 0.0), (0.0, 0.001, 0.0),
                                      (0.0, -0.001, 0.0), (0.0, 0.05, 0.0)])
def test_nonstationary_motion_keeps_all_parent_state_transitions(velocity):
    candidate, parent = _new(0.62), LoadAwareCompensation("online", nominal_mu=0.62, max_force=8.0)
    for _ in range(250):
        np.testing.assert_array_equal(_cycle(candidate, velocity=velocity),
                                      _cycle(parent, velocity=velocity))
        assert candidate.hold_entry_mu is None
        for name in ("equivalent_mu", "update_ready", "applied_budget_n", "next_budget_n",
                     "load_estimate_n", "measurement_used_for_budget"):
            assert getattr(candidate, name) == getattr(parent, name)


@pytest.mark.parametrize("velocity", [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0),
                                      (0.0, 1e-12, 0.0), (0.0, -1e-12, 0.0)])
def test_entry_gate_uses_inclusive_numerical_zero_tangential_speed(velocity):
    candidate = _new(0.62)
    _cycle(candidate, velocity=velocity)
    assert candidate.hold_entry_mu == 0.62
    assert candidate.equivalent_mu == pytest.approx(0.6194, abs=1e-14)
    assert not candidate.update_ready


@pytest.mark.parametrize("interruption", [{"measurement": False}, {"accepted": False}])
def test_unavailable_or_rejected_first_hold_still_captures_learned_entry(interruption):
    candidate = _new(0.45)
    candidate._equivalent_mu = 0.52  # State acquired before this hold, not nominal_mu.
    _cycle(candidate, **interruption)
    assert candidate.hold_entry_mu == candidate.equivalent_mu == 0.52
    _cycle(candidate, normal_force=14.0)
    decreased = candidate.equivalent_mu
    assert decreased < 0.52
    for _ in range(5):
        _cycle(candidate, normal_force=8.0, **interruption)
        assert candidate.hold_entry_mu == 0.52
        assert candidate.equivalent_mu == decreased
    for _ in range(10):
        _cycle(candidate, normal_force=8.0)
    assert candidate.equivalent_mu == candidate.hold_entry_mu == 0.52


@pytest.mark.parametrize("boundary", [{"velocity": (0.0, 0.001, 0.0)}, {"contact": False},
                                     {"normal_force": 1.0}, {"target_force": 0.0}])
def test_hold_entry_clears_at_real_hold_or_contact_boundaries(boundary):
    candidate = _new(0.62)
    _cycle(candidate)
    assert candidate.hold_entry_mu == 0.62
    _cycle(candidate, **boundary)
    assert candidate.hold_entry_mu is None
    next_entry = candidate.equivalent_mu
    _cycle(candidate)
    assert candidate.hold_entry_mu == next_entry


def test_reset_clears_entry_and_packet_and_constructor_is_reset_safe():
    candidate = _new(0.62)
    assert candidate.hold_entry_mu is None
    _cycle(candidate)
    candidate.set_force_measurement([12.0, 7.8, 0.0])
    candidate.reset()
    assert candidate.hold_entry_mu is None
    assert candidate.equivalent_mu == 0.62
    _cycle(candidate, measurement=False)
    assert candidate.equivalent_mu == 0.62


def test_consumed_packet_cannot_authorize_a_second_update():
    candidate = _new()
    _cycle(candidate, normal_force=14.0)
    before, entry = candidate.equivalent_mu, candidate.hold_entry_mu
    state, target = _sample()
    candidate.advance(state, target, NORMAL, DT, True)
    assert candidate.equivalent_mu == before
    assert candidate.hold_entry_mu == entry


def test_release_and_recovery_are_rate_limited_and_do_not_change_current_output():
    candidate = _new(0.62)
    for normal_force in [20.0] * 20 + [8.0] * 25:
        before = candidate.equivalent_mu
        output = _cycle(candidate, normal_force=normal_force)
        assert abs(candidate.equivalent_mu - before) <= 0.3 * DT + 1e-14
        assert 0.0 <= candidate.equivalent_mu <= candidate.hold_entry_mu <= 0.9
        np.testing.assert_array_equal(output, np.zeros(3))
        np.testing.assert_array_equal(candidate.last_force, output)
    assert candidate.equivalent_mu == 0.62


def test_hold_update_uses_applied_not_next_budget():
    candidate = _new(0.5002)
    state, target = _sample()
    candidate.set_force_measurement([12.0, 7.8, 0.0])
    candidate.force(state, target, NORMAL, 12.0, 1.0, True, dt=DT)
    candidate._next_budget = 8.0
    candidate.advance(state, target, NORMAL, DT, True)
    assert candidate.equivalent_mu == 0.5
    assert candidate.applied_budget_n == 6.0
    assert candidate.next_budget_n == 8.0


@pytest.mark.parametrize("gate", ["accepted", "scaled", "fallback", "missing_context"])
def test_actual_projection_seam_guards_both_release_and_recovery(gate):
    candidate, base = _new(), ControlledAdaptive()
    safe = FrankaSafeAdaptiveController(base=base, tangential=candidate)
    state, target = _sample()
    actuation = {"accepted": context(), "scaled": context(1.0),
                 "fallback": context(1.0, 2.0), "missing_context": None}[gate]
    state = replace(state, actuation=actuation)
    safe.reset(state)
    _cycle(candidate, normal_force=14.0)
    before, entry = candidate.equivalent_mu, candidate.hold_entry_mu
    candidate.set_force_measurement([12.0, 7.8, 0.0])
    safe.compute(state, target, DT)
    assert candidate.hold_entry_mu == entry
    assert candidate.equivalent_mu == (entry if gate == "accepted" else before)
    assert base.calls == 1


@pytest.mark.parametrize("dt", [0.0, -DT, np.nan, np.inf])
def test_invalid_dt_is_rejected_by_the_inherited_advance(dt):
    candidate = _new()
    state, target = _sample()
    with pytest.raises(ValueError, match="dt"):
        candidate.advance(state, target, NORMAL, dt, True)
    assert candidate.hold_entry_mu is None


@pytest.mark.parametrize("measurement", [[12.0, np.nan, 0.0], [12.0, 7.8]])
def test_rejected_measurement_clears_pending_packet(measurement):
    candidate = _new()
    candidate.set_force_measurement([12.0, 7.8, 0.0])
    with pytest.raises(ValueError, match="finite 3-vector"):
        candidate.set_force_measurement(measurement)
    _cycle(candidate, normal_force=14.0, measurement=False)
    assert candidate.equivalent_mu == candidate.hold_entry_mu == 0.5
