"""Cycle-level checks against the unchanged online compensation controller."""

from dataclasses import replace

import numpy as np
import pytest

from compliant_control_lab.tangential_compensation import TangentialCompensation
from tests.test_tangential_safety import setup
from tools.load_aware_compensation import LoadAwareCompensation

NORMAL = np.array([1., 0., 0.])
DT = .002


def inputs():
    _, _, _, state, target = setup("online")
    target = replace(target, position=np.array([.01, .004, 0.]))
    return replace(state, linear_velocity=target.linear_velocity.copy()), target


def cycle(c, state, target, force=(12., 7.2, 0.), allow=True, contact=True):
    if force is not None:
        c.set_force_measurement(force)
    output = c.force(state, target, NORMAL, 12., 1., contact, dt=DT)
    c.advance(state, target, NORMAL, DT, allow)
    return output


def settled():
    c = LoadAwareCompensation("online", max_force=8.)
    state, target = inputs()
    for _ in range(300):
        cycle(c, state, target, allow=False)
    assert c.update_ready
    return c, state, target


@pytest.mark.parametrize("cap", [6., 8.])
def test_fixed_ceiling_reproduces_old_cycles_exactly(cap):
    c = LoadAwareCompensation("online", max_force=cap, minimum_force=cap)
    old = TangentialCompensation("online", max_force=cap)
    state, target = inputs()
    for i in range(1500):
        speed = 0. if 450 <= i < 500 else (-.02 if 500 <= i < 1000 else .02)
        t = replace(target, linear_velocity=np.array([0., speed, 0.]))
        s = replace(state, linear_velocity=t.linear_velocity.copy())
        contact, allow = i != 1000, i % 9 != 0
        output = cycle(c, s, t, allow=allow, contact=contact)
        expected = old.force(s, t, NORMAL, 12., 1., contact, dt=DT)
        old.advance(s, t, NORMAL, DT, allow)
        np.testing.assert_array_equal(output, expected)
        assert c.equivalent_mu == old.equivalent_mu
        assert c.update_ready == old.update_ready
        assert c.max_force == cap


def test_budget_uses_exact_causal_recurrence_and_next_cycle_only():
    c, s, t = settled()
    for _ in range(400):
        before_load, before_budget = c.load_estimate_n, c.next_budget_n
        c.set_force_measurement([500., 7.2, 99.])
        c.force(s, t, NORMAL, 12., 1., True, dt=DT)
        assert c.next_budget_n == before_budget
        assert c.applied_budget_n == before_budget
        ready = c.update_ready
        c.advance(s, t, NORMAL, DT, True)
        expected = before_load + -np.expm1(-DT / .20) * (7.2 - before_load) if ready else before_load
        assert c.load_estimate_n == pytest.approx(expected, abs=1e-14)
        assert c.next_budget_n == pytest.approx(np.clip(expected + .25, 6., 8.))
        assert c.measurement_used_for_budget == ready
    assert 7.2 < c.next_budget_n < 7.45


@pytest.mark.parametrize("load,projected", [(5.7, 5.7), (-7.2, 0.), (99., 8.)])
def test_direction_sign_and_projected_clipping(load, projected):
    c, s, t = settled()
    cycle(c, s, t, [999., load, 123.])
    assert c.projected_load_n == projected
    if load <= 5.7:
        for _ in range(500):
            cycle(c, s, t, [999., load, 123.])
        assert c.next_budget_n == 6.


def test_missing_or_consumed_measurement_is_never_reused():
    c, s, t = settled()
    for _ in range(500):
        cycle(c, s, t)
    assert c.next_budget_n > 7.
    before = c.load_estimate_n
    c.advance(s, t, NORMAL, DT, True)
    assert c.load_estimate_n == before
    cycle(c, s, t, force=None)
    assert c.next_budget_n == c.applied_budget_n == 6.
    assert c.load_estimate_n == 0.
    assert not c.measurement_used_for_budget


@pytest.mark.parametrize("reason", ["projection", "stop", "reversal", "contact"])
def test_ineligible_cycles_freeze_or_clear_load(reason):
    c, s, t = settled()
    for _ in range(400):
        cycle(c, s, t)
    before = c.load_estimate_n
    if reason in {"stop", "reversal"}:
        t = replace(t, linear_velocity=t.linear_velocity * (0 if reason == "stop" else -1))
    cycle(c, s, t, allow=reason != "projection", contact=reason != "contact")
    assert c.load_estimate_n == (0. if reason == "contact" else before)
    assert not c.measurement_used_for_budget


def test_current_applied_ceiling_controls_parent_antiwindup():
    c, s, t = settled()
    c._equivalent_mu = .55  # 6.6 N exceeds current 6 N but not global 8 N.
    for _ in range(60):
        cycle(c, s, t, allow=False)
    assert c.update_ready
    cycle(c, s, t)
    assert c.equivalent_mu == .55
    assert c.max_force == 8.


def test_global_bound_and_slew_survive_falling_ceiling():
    c, s, t = settled()
    for _ in range(1200):
        cycle(c, s, t, [12., 8., 0.])
    assert np.linalg.norm(c.last_force) > 6.
    previous = c.last_force
    output = cycle(c, s, t, force=None)
    assert c.applied_budget_n == 6.
    assert np.linalg.norm(output) > 6.  # Target ceiling, not an instantaneous hard clip.
    assert np.linalg.norm(output - previous) <= 20 * DT + 1e-12
    for _ in range(200):
        previous = c.last_force
        output = cycle(c, s, t, [12., 5., 0.])
        assert np.linalg.norm(output) <= 8. + 1e-12
        assert np.linalg.norm(output - previous) <= 20 * DT + 1e-12


def test_temporary_parent_ceiling_restored_on_error():
    c, s, t = settled()
    with pytest.raises(ValueError):
        c.force(s, t, NORMAL, 12., 1., True, dt=-1)
    assert c.max_force == 8.
    with pytest.raises(ValueError):
        c.advance(s, t, NORMAL, -1, True)
    assert c.max_force == 8.
