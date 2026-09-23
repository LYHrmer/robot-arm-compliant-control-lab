"""Numerical controller checks; no new physics trajectories are generated."""

from dataclasses import replace

import numpy as np
import pytest

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.franka_control import FrankaState, FrankaTarget
from tests.test_tangential_safety import ControlledAdaptive, context
from tools.load_aware_compensation import LoadAwareCompensation
from tools.reversal_recovery.controller import HoldCapTracking
from tools.stationary_recovery.controller import StationaryHoldCapTracking

NORMAL = np.array([1.0, 0.0, 0.0])
DT = 0.002


def _sample(velocity=(0.0, 0.0, 0.0)):
    velocity = np.asarray(velocity, dtype=float)
    state = FrankaState(np.zeros(3), np.eye(3), velocity, np.zeros(3), 12.0)
    target = FrankaTarget(np.zeros(3), np.eye(3), velocity, np.zeros(3), 12.0)
    return state, target


def _cycle(compensation, *, velocity=(0.0, 0.0, 0.0), measurement=True,
           accepted=True, contact=True, normal_force=12.0, target_force=12.0, dt=DT):
    state, target = _sample(velocity)
    target = replace(target, normal_force=target_force)
    if measurement:
        compensation.set_force_measurement([normal_force, 7.8, 0.0])
    force = compensation.force(state, target, NORMAL, normal_force, 1.0, contact, dt=dt)
    compensation.advance(state, target, NORMAL, dt, accepted)
    return force


@pytest.mark.parametrize("velocity", [
    (0.0, 0.001, 0.0), (0.0, -0.001, 0.0),
    (0.0, 0.0, 0.001), (0.0, 0.0, -0.001),
    (0.0, 0.05, 0.0), (0.0, -0.05, 0.0),
    (0.0, 1.0001e-12, 0.0), (0.0, -1.0001e-12, 0.0),
])
def test_nonzero_tangential_motion_matches_inherited_cycles_exactly(velocity):
    candidate = StationaryHoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    parent = LoadAwareCompensation("online", nominal_mu=0.62, max_force=8.0)
    for _ in range(250):
        actual = _cycle(candidate, velocity=velocity)
        expected = _cycle(parent, velocity=velocity)
        np.testing.assert_array_equal(actual, expected)
        assert candidate.equivalent_mu == parent.equivalent_mu
        assert candidate.update_ready == parent.update_ready
        assert candidate.applied_budget_n == parent.applied_budget_n
        assert candidate.next_budget_n == parent.next_budget_n
        assert candidate.load_estimate_n == parent.load_estimate_n
        assert candidate.measurement_used_for_budget == parent.measurement_used_for_budget


@pytest.mark.parametrize("speed", [-0.001, 0.001])
def test_low_speed_motion_no_longer_activates_the_old_hold_release(speed):
    candidate = StationaryHoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    old = HoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    actual = _cycle(candidate, velocity=(0.0, speed, 0.0))
    expected = _cycle(old, velocity=(0.0, speed, 0.0))
    np.testing.assert_array_equal(actual, expected)
    assert candidate.equivalent_mu == 0.62
    assert old.equivalent_mu == pytest.approx(0.6194, abs=1e-14)


@pytest.mark.parametrize("velocity", [
    (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (-0.1, 0.0, 0.0),
    (0.0, 1e-12, 0.0), (0.0, -1e-12, 0.0),
])
def test_stationary_tangent_retains_original_release_and_next_cycle_timing(velocity):
    candidate = StationaryHoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    old = HoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    for _ in range(10):
        previous = candidate.equivalent_mu
        actual = _cycle(candidate, velocity=velocity)
        expected = _cycle(old, velocity=velocity)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(candidate.last_force, actual)
        assert candidate.equivalent_mu == old.equivalent_mu
        assert previous - candidate.equivalent_mu == pytest.approx(0.3 * DT, abs=1e-14)
        assert candidate.applied_budget_n == candidate.next_budget_n == 6.0
        assert candidate.max_force == 8.0
        assert not candidate.measurement_used_for_budget


@pytest.mark.parametrize("kwargs", [
    {"measurement": False}, {"accepted": False}, {"contact": False},
    {"normal_force": 1.0}, {"target_force": 0.0},
])
def test_existing_measurement_projection_and_contact_gates_still_reject(kwargs):
    candidate = StationaryHoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    old = HoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    np.testing.assert_array_equal(_cycle(candidate, **kwargs), _cycle(old, **kwargs))
    assert candidate.equivalent_mu == old.equivalent_mu == 0.62


def test_a_consumed_measurement_cannot_authorize_another_release():
    candidate = StationaryHoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    _cycle(candidate)
    before = candidate.equivalent_mu
    state, target = _sample()
    candidate.advance(state, target, NORMAL, DT, True)
    assert candidate.equivalent_mu == before


@pytest.mark.parametrize("nominal_mu", [0.45, 0.5002, 0.62])
@pytest.mark.parametrize("dt", [0.002, 0.01])
def test_release_never_increases_mu_and_obeys_existing_rate_until_current_cap(nominal_mu, dt):
    candidate = StationaryHoldCapTracking("online", nominal_mu=nominal_mu, max_force=8.0)
    assert candidate.coefficient_rate_limit == 0.3
    for _ in range(250):
        previous = candidate.equivalent_mu
        _cycle(candidate, dt=dt)
        assert 0.0 <= previous - candidate.equivalent_mu <= 0.3 * dt + 1e-14
    assert candidate.equivalent_mu == min(nominal_mu, 0.5)


def test_release_uses_current_applied_budget_without_changing_current_force():
    candidate = StationaryHoldCapTracking("online", nominal_mu=0.5002, max_force=8.0)
    state, target = _sample()
    candidate.set_force_measurement([12.0, 7.8, 0.0])
    output = candidate.force(state, target, NORMAL, 12.0, 1.0, True, dt=DT)
    assert candidate.equivalent_mu == 0.5002
    # Separate the two budget fields to check which causal boundary release reads.
    candidate._next_budget = 8.0
    candidate.advance(state, target, NORMAL, DT, True)
    assert candidate.applied_budget_n == 6.0
    assert candidate.next_budget_n == 8.0
    assert candidate.equivalent_mu == 0.5
    np.testing.assert_array_equal(output, np.zeros(3))
    np.testing.assert_array_equal(candidate.last_force, output)


@pytest.mark.parametrize("gate", ["accepted", "scaled", "fallback", "missing_context"])
@pytest.mark.parametrize("speed", [0.0, -0.001, 0.001])
def test_real_safe_controller_keeps_projection_before_stationary_release(gate, speed):
    candidate = StationaryHoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    base = ControlledAdaptive()
    safe = FrankaSafeAdaptiveController(base=base, tangential=candidate)
    actuation = {"accepted": context(), "scaled": context(1.0),
                 "fallback": context(1.0, 2.0), "missing_context": None}[gate]
    state, target = _sample((0.0, speed, 0.0))
    state = replace(state, actuation=actuation)
    safe.reset(state)
    candidate.set_force_measurement([12.0, 7.8, 0.0])
    output = safe.compute(state, target, DT)
    expected_mu = 0.6194 if gate == "accepted" and speed == 0.0 else 0.62
    assert candidate.equivalent_mu == pytest.approx(expected_mu, abs=1e-14)
    assert base.calls == 1
    if gate == "scaled":
        assert 0.0 < safe.last_torque_projection_scale < 1.0
        assert np.max(np.abs(actuation.joint_torque(output))) <= 0.9 + 1e-12
    elif gate == "fallback":
        assert safe.torque_projection_fallback_count == 1
        np.testing.assert_array_equal(output, np.zeros(6))
    elif speed == 0.0:
        np.testing.assert_array_equal(output, base.wrench)
