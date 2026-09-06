"""Classical additions cross the existing safe-controller seam before projection."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.franka_control import FrankaActuationContext, FrankaState, FrankaTarget
from compliant_control_lab.franka_torque_safety import project_wrench_to_torque_limits
from compliant_control_lab.tangential_compensation import TangentialCompensation


class ControlledAdaptive:
    """Expose current contact estimates without hiding projection behind a mock."""

    def __init__(self):
        self.base = SimpleNamespace(normal=np.array([1.0, 0, 0]), in_contact=True)
        self.corrected_force_n = 12.0
        self.contact_blend = 1.0
        self.filtered_force_rate_n_s = 0.0
        self.calls = 0
        self.wrench = np.array([12.0, 0, 0, 0, 0, 0])

    def reset(self, state):
        self.corrected_force_n = max(0.0, state.normal_force)

    def compute(self, state, target, dt):
        self.calls += 1
        self.corrected_force_n = max(0.0, state.normal_force)
        return self.wrench.copy()


def context(limit=100.0, offset=0.0):
    return FrankaActuationContext(
        np.column_stack((np.eye(6), np.zeros(6))), np.full(7, offset),
        np.full(7, -limit), np.full(7, limit),
    )


def setup(mode="integral"):
    state = FrankaState(np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3), 12, context())
    target = FrankaTarget(np.array([0.01, 0.3, 0.4]), np.eye(3),
                         np.array([0.0, 0.02, 0]), np.zeros(3), 12)
    addition, base = TangentialCompensation(mode=mode), ControlledAdaptive()
    safe = FrankaSafeAdaptiveController(base=base, tangential=addition)
    safe.reset(state)
    return safe, base, addition, state, target


def test_integral_output_precedes_next_cycle_update_and_base_runs_once():
    safe, base, addition, state, target = setup()
    first = safe.compute(state, target, 0.002)
    np.testing.assert_array_equal(first, base.wrench)
    np.testing.assert_array_equal(addition.last_force, np.zeros(3))
    increment = np.array([0, 0.48, 0.64])
    np.testing.assert_allclose(addition.integral_force, increment, atol=1e-15)
    second = safe.compute(state, target, 0.002)
    np.testing.assert_allclose(second[:3] - base.wrench[:3], increment, atol=1e-15)
    np.testing.assert_allclose(addition.integral_force, 2 * increment, atol=1e-15)
    assert base.calls == 2


def test_integral_uses_vector_norm_cap_and_properties_are_copies():
    safe, _, addition, state, target = setup()
    for _ in range(100):
        wrench = safe.compute(state, target, 0.002)
        assert np.linalg.norm(addition.integral_force) <= 6 + 1e-12
        assert np.linalg.norm(wrench[1:3]) <= 6 + 1e-12
    np.testing.assert_allclose(addition.integral_force, [0, 3.6, 4.8], atol=1e-12)
    stored, requested = addition.integral_force, addition.last_force
    stored[:] = requested[:] = 99
    np.testing.assert_allclose(addition.integral_force, [0, 3.6, 4.8], atol=1e-12)
    np.testing.assert_allclose(addition.last_force, [0, 3.6, 4.8], atol=1e-12)


@pytest.mark.parametrize("loss", ["unconfirmed", "low_force", "zero_target", "reset"])
def test_contact_loss_and_reset_clear_immediately(loss):
    safe, base, addition, state, target = setup()
    safe.compute(state, target, 0.002)
    assert np.linalg.norm(addition.integral_force) > 0
    if loss == "reset":
        safe.reset(state)
    else:
        if loss == "unconfirmed":
            base.base.in_contact = False
        elif loss == "low_force":
            state = replace(state, normal_force=1.0)
        else:
            target = replace(target, normal_force=0)
        np.testing.assert_array_equal(safe.compute(state, target, 0.002), base.wrench)
    np.testing.assert_array_equal(addition.integral_force, np.zeros(3))
    np.testing.assert_array_equal(addition.last_force, np.zeros(3))


@pytest.mark.parametrize("reason", ["scaled", "offset_fallback", "missing_context"])
def test_projected_or_unknown_actuation_freezes_and_release_has_no_hidden_windup(reason):
    safe, base, addition, state, target = setup()
    safe.compute(state, target, 0.002)
    stored = addition.integral_force
    blocked_context = {"scaled": context(1), "offset_fallback": context(1, 2),
                       "missing_context": None}[reason]
    blocked_state = replace(state, actuation=blocked_context)
    for _ in range(50):
        output = safe.compute(blocked_state, target, 0.002)
        np.testing.assert_array_equal(addition.integral_force, stored)
    if reason == "scaled":
        assert 0 < safe.last_torque_projection_scale < 1
        assert np.max(np.abs(blocked_context.joint_torque(output))) <= 0.9 + 1e-12
    elif reason == "offset_fallback":
        assert safe.torque_projection_fallback_count == 50
        np.testing.assert_array_equal(output, np.zeros(6))
        # The infeasible offset itself is NOT repaired by returning zero wrench.
        assert np.max(np.abs(blocked_context.joint_torque(output))) > 1
    released = safe.compute(state, target, 0.002)
    np.testing.assert_allclose(released[:3] - base.wrench[:3], stored, atol=1e-15)
    np.testing.assert_allclose(addition.integral_force, 2 * stored, atol=1e-15)
    assert base.calls == 52


def test_friction_uses_current_corrected_measurement_blend_and_stops_on_negative_force():
    safe, base, addition, state, target = setup("friction")
    base.corrected_force_n, base.contact_blend = 3, 0.5
    state = replace(state, normal_force=10)
    wrench = safe.compute(state, target, 0.002)
    expected = 0.5 * 0.45 * 10 * 0.02 / np.sqrt(0.02**2 + 0.005**2)
    np.testing.assert_allclose(wrench[:3] - base.wrench[:3], [0, expected, 0])
    np.testing.assert_allclose(addition.last_force, [0, expected, 0])
    np.testing.assert_array_equal(
        safe.compute(replace(state, normal_force=-2), target, 0.002), base.wrench,
    )
    np.testing.assert_array_equal(addition.last_force, np.zeros(3))
    assert base.calls == 2


def test_friction_addition_is_inside_the_same_full_wrench_projection():
    safe, base, addition, state, target = setup("friction")
    state = replace(state, actuation=context(1))
    output = safe.compute(state, target, 0.002)
    requested = base.wrench.copy()
    requested[:3] += addition.last_force
    expected = project_wrench_to_torque_limits(state.actuation, np.zeros(6), requested)
    np.testing.assert_array_equal(output, expected.additive_wrench)
    assert np.linalg.norm(addition.last_force) > np.linalg.norm(output[1:3])
    assert base.calls == 1


@pytest.mark.parametrize("with_context", [False, True])
def test_none_option_matches_independent_original_compute_sequence_exactly(with_context):
    _, _, _, state, target = setup()
    if not with_context:
        state = replace(state, actuation=None)
    actual, legacy = FrankaSafeAdaptiveController(tangential=None), FrankaSafeAdaptiveController()
    assert legacy.tangential is None
    actual.reset(state)
    legacy.reset(state)
    for measured_force in [0.0, 0.5, 5.0, 12.0, 20.0, -1.0]:
        state = replace(state, normal_force=measured_force)
        governed = legacy._govern_target(state, target)
        expected = legacy.base.compute(state, governed, 0.002)
        if with_context:
            expected = project_wrench_to_torque_limits(
                state.actuation, np.zeros(6), expected, legacy.torque_reserve_fraction,
            ).additive_wrench
        np.testing.assert_array_equal(actual.compute(state, target, 0.002), expected)
