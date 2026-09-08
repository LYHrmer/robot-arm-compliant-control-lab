"""Causal update and bounds of the optional error-driven load coefficient."""

from dataclasses import replace

import numpy as np
import pytest

from compliant_control_lab.tangential_compensation import TangentialCompensation
from tests.test_tangential_safety import context, setup


def ready_loop():
    safe, base, addition, state, target = setup("online")
    target = replace(target, position=np.array([0.01, 0.004, 0.0]))
    state = replace(state, linear_velocity=target.linear_velocity.copy())
    # Let the force request reach its FF value with updates disabled by projection.
    blocked = replace(state, actuation=None)
    for _ in range(200):
        safe.compute(blocked, target, 0.002)
    assert addition.update_ready
    assert addition.equivalent_mu == 0.45
    return safe, base, addition, state, target


def test_current_coefficient_generates_force_before_next_coefficient_update():
    safe, base, addition, state, target = ready_loop()
    mu_before = addition.equivalent_mu
    output = safe.compute(state, target, 0.002)
    direction = 0.02 / np.sqrt(0.02**2 + 0.005**2)
    expected_force = mu_before * 12 * direction
    np.testing.assert_allclose(output[:3] - base.wrench[:3], [0, expected_force, 0])
    expected_increment = 0.002 * 800 * 12 / (12**2 + 2**2) * direction * 0.004
    assert addition.equivalent_mu == pytest.approx(mu_before + expected_increment, abs=1e-15)


@pytest.mark.parametrize("reason", ["projection", "offset", "no_context", "blend", "slow"])
def test_ineligible_cycle_does_not_learn(reason):
    safe, base, addition, state, target = ready_loop()
    if reason == "projection":
        state = replace(state, actuation=context(1))
    elif reason == "offset":
        state = replace(state, actuation=context(1, 2))
    elif reason == "no_context":
        state = replace(state, actuation=None)
    elif reason == "blend":
        base.contact_blend = 0.5
    else:
        target = replace(target, linear_velocity=np.zeros(3))
    for _ in range(30):
        safe.compute(state, target, 0.002)
        assert addition.equivalent_mu == 0.45


def test_slew_and_reversal_freeze_then_direction_recovers():
    safe, _, addition, state, target = ready_loop()
    target = replace(target, linear_velocity=-target.linear_velocity,
                     position=np.array([0.01, -0.004, 0]))
    state = replace(state, linear_velocity=target.linear_velocity.copy())
    previous = addition.last_force
    safe.compute(state, target, 0.002)
    assert not addition.update_ready
    assert addition.equivalent_mu == 0.45
    assert np.linalg.norm(addition.last_force - previous) <= 20 * 0.002 + 1e-12
    for _ in range(300):
        previous = addition.last_force
        safe.compute(state, target, 0.002)
        assert np.linalg.norm(addition.last_force - previous) <= 20 * 0.002 + 1e-12
        assert np.linalg.norm(addition.last_force) <= 6 + 1e-12
        assert addition.last_force[0] == 0
    assert addition.last_force[1] < 0
    assert addition.equivalent_mu > 0.45


@pytest.mark.parametrize("loss", ["contact", "low_force", "target", "reset"])
def test_contact_loss_hard_clears_request_and_learned_state(loss):
    safe, base, addition, state, target = ready_loop()
    safe.compute(state, target, 0.002)
    assert addition.equivalent_mu > 0.45
    if loss == "reset":
        safe.reset(state)
    else:
        if loss == "contact":
            base.base.in_contact = False
        elif loss == "low_force":
            state = replace(state, normal_force=1)
        else:
            target = replace(target, normal_force=0)
        safe.compute(state, target, 0.002)
    assert addition.equivalent_mu == 0.45
    assert not addition.update_ready
    np.testing.assert_array_equal(addition.last_force, np.zeros(3))


def test_force_amplitude_ceiling_prevents_hidden_outward_adaptation_but_can_unwind():
    safe, _, addition, state, target = ready_loop()
    state = replace(state, normal_force=30)
    # 0.45 * 30 already exceeds the 6 N amplitude ceiling.
    for _ in range(150):
        safe.compute(state, target, 0.002)
    assert addition.equivalent_mu == 0.45
    target = replace(target, position=np.array([0.01, -0.004, 0]))
    safe.compute(state, target, 0.002)
    assert addition.equivalent_mu < 0.45


def test_coefficient_projection_upper_and_lower_bounds():
    safe, _, addition, state, target = ready_loop()
    addition.coefficient_rate_limit = 1e6  # Isolate projection from the separately tested rate cap.
    state = replace(state, normal_force=4)
    for sign, expected in [(1, 0.9), (-1, 0)]:
        target = replace(target, position=np.array([0.01, sign * 100.0, 0]))
        for _ in range(200):
            safe.compute(state, target, 0.002)
            assert 0 <= addition.equivalent_mu <= 0.9
        assert addition.equivalent_mu == expected


def test_parameter_rate_limit_and_motion_confirmation():
    safe, _, addition, state, target = ready_loop()
    target = replace(target, position=np.array([0.01, 10.0, 0]))
    before = addition.equivalent_mu
    safe.compute(state, target, 0.002)
    assert addition.equivalent_mu - before == pytest.approx(0.3 * 0.002)
    # Desired movement alone is insufficient when the measured tool is stuck.
    state = replace(state, linear_velocity=np.zeros(3))
    before = addition.equivalent_mu
    for _ in range(10):
        safe.compute(state, target, 0.002)
        assert not addition.update_ready
        assert addition.equivalent_mu == before
    state = replace(state, linear_velocity=target.linear_velocity.copy())
    for _ in range(24):
        safe.compute(state, target, 0.002)
        assert not addition.update_ready
        assert addition.equivalent_mu == before
    safe.compute(state, target, 0.002)
    assert addition.update_ready


@pytest.mark.parametrize("field,value", [
    ("adaptation_gain", np.nan), ("velocity_error_time", -1), ("force_regularizer", 0),
    ("max_equivalent_mu", 0.1), ("min_update_speed", np.inf), ("force_slew_rate", 0),
])
def test_invalid_online_parameters(field, value):
    with pytest.raises(ValueError):
        TangentialCompensation("online", **{field: value})


@pytest.mark.parametrize("dt", [None, 0, -1, np.nan, np.inf])
def test_online_requires_explicit_positive_timestep(dt):
    _, _, addition, state, target = setup("online")
    with pytest.raises(ValueError):
        addition.force(state, target, [1, 0, 0], 12, 1, True, dt=dt)
