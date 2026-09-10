"""Synthetic seam tests for the onset observer; no simulator physics is run."""

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from compliant_control_lab.franka_control import FrankaState, FrankaTarget
from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.tangential_compensation import TangentialCompensation
from tools import onset_observer as observer

NORMAL = np.array([1.0, 0.0, 0.0])
DT = 0.002


def _state_target():
    state = FrankaState(
        np.zeros(3), np.eye(3), np.array([0.0, 0.02, 0.0]), np.zeros(3), 12.0
    )
    target = FrankaTarget(
        np.array([0.0, 0.1, 0.0]),
        np.eye(3),
        np.array([0.0, 0.02, 0.0]),
        np.zeros(3),
        12.0,
    )
    return state, target


def _force(compensation, state, target, *, contact=True, blend=1.0):
    return compensation.force(
        state, target, NORMAL, 12.0, blend, contact, dt=DT
    )


def _ready(**kwargs):
    compensation = observer.ObservedCompensation(
        "online", force_slew_rate=1e6, motion_confirm_time=2 * DT, **kwargs
    )
    state, target = _state_target()
    _force(compensation, state, target)
    _force(compensation, state, target)
    assert compensation.update_ready
    return compensation, state, target


def _assert_parent_state(actual, expected):
    for field in fields(TangentialCompensation):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=field.name)
        else:
            assert left == right, field.name


def test_observation_preserves_exact_parent_force_advance_sequence():
    parameters = {"force_slew_rate": 100.0, "motion_confirm_time": 0.004}
    actual = observer.ObservedCompensation("online", **parameters)
    expected = TangentialCompensation("online", **parameters)
    state, target = _state_target()

    for allow in (False, True, True, False):
        np.testing.assert_array_equal(
            _force(actual, state, target), _force(expected, state, target)
        )
        _assert_parent_state(actual, expected)
        actual.advance(state, target, NORMAL, DT, allow)
        expected.advance(state, target, NORMAL, DT, allow)
        _assert_parent_state(actual, expected)


def test_recorded_inputs_outputs_and_observation_property_do_not_alias():
    compensation = observer.ObservedCompensation("online")
    state, target = _state_target()
    normal = NORMAL.copy()
    request = compensation.force(state, target, normal, 12.0, 1.0, True, dt=DT)
    recorded = compensation.observation
    expected_normal = recorded["obs_normal_local"].copy()
    expected_request = recorded["obs_requested_force_local_n"].copy()

    normal[:] = 9.0
    state.linear_velocity[:] = 9.0
    target.linear_velocity[:] = 9.0
    request[:] = 9.0
    recorded["obs_normal_local"][:] = 8.0
    recorded["obs_requested_force_local_n"][:] = 8.0

    fresh = compensation.observation
    np.testing.assert_array_equal(fresh["obs_normal_local"], expected_normal)
    np.testing.assert_array_equal(fresh["obs_requested_force_local_n"], expected_request)
    np.testing.assert_array_equal(compensation.last_force, expected_request)


def test_inactive_force_records_prior_state_then_parent_reset():
    compensation, state, target = _ready()
    compensation._equivalent_mu = 0.6
    prior_force = compensation.last_force

    request = _force(compensation, state, target, contact=False)
    record = compensation.observation

    np.testing.assert_array_equal(request, np.zeros(3))
    np.testing.assert_array_equal(record["obs_previous_force_local_n"], prior_force)
    assert record["obs_coefficient_before_force"] == 0.6
    assert record["obs_coefficient_after_force"] == compensation.nominal_mu
    assert not record["obs_active"]
    assert not compensation.update_ready


def test_each_force_starts_a_fresh_record_without_stale_advance_flags():
    compensation, state, target = _ready()
    compensation.advance(state, target, NORMAL, DT, True)
    assert compensation.observation["obs_advance_called"]

    _force(compensation, state, target)
    record = compensation.observation

    assert not record["obs_advance_called"]
    assert not record["obs_allow_integration"]
    assert record["obs_coefficient_after_advance"] == record["obs_coefficient_after_force"]


def test_disallowed_advance_records_freeze_without_changing_coefficient():
    compensation, state, target = _ready()
    _force(compensation, state, target)
    before = compensation.equivalent_mu

    compensation.advance(state, target, NORMAL, DT, False)
    record = compensation.observation

    assert record["obs_advance_called"]
    assert not record["obs_allow_integration"]
    assert record["obs_coefficient_after_advance"] == before
    assert compensation.equivalent_mu == before


def test_ready_positive_drive_reports_rate_limit_and_committed_update():
    compensation, state, target = _ready()
    _force(compensation, state, target)
    record = compensation.observation

    assert record["obs_active"] and record["obs_update_ready"]
    assert not record["obs_amplitude_capped"]
    assert not record["obs_slew_limited"]
    assert record["obs_position_drive_m"] > 0
    assert record["obs_candidate_increment"] > record["obs_limited_increment"] > 0
    assert record["obs_limited_increment"] == pytest.approx(
        compensation.coefficient_rate_limit * DT
    )

    before = compensation.equivalent_mu
    compensation.advance(state, target, NORMAL, DT, True)
    assert compensation.observation["obs_coefficient_after_advance"] == pytest.approx(
        before + record["obs_limited_increment"]
    )


def test_cap_blocks_positive_update_but_allows_negative_unwind():
    compensation, state, target = _ready(max_force=4.0)
    _force(compensation, state, target)
    assert compensation.observation["obs_amplitude_capped"]
    before = compensation.equivalent_mu
    compensation.advance(state, target, NORMAL, DT, True)
    assert compensation.equivalent_mu == before

    negative = replace(target, position=np.array([0.0, -0.1, 0.0]))
    _force(compensation, state, negative)
    assert compensation.observation["obs_limited_increment"] < 0
    compensation.advance(state, negative, NORMAL, DT, True)
    assert compensation.equivalent_mu < before


def test_first_force_reports_slew_limited_request_and_signed_direction():
    compensation = observer.ObservedCompensation("online")
    state, target = _state_target()
    target = replace(
        target,
        position=np.array([0.0, -0.1, 0.0]),
        linear_velocity=np.array([0.0, -0.02, 0.0]),
    )

    request = _force(compensation, state, target)
    record = compensation.observation

    assert record["obs_slew_limited"]
    assert np.linalg.norm(request) == pytest.approx(compensation.force_slew_rate * DT)
    assert record["obs_direction_local"][1] < 0
    assert record["obs_position_drive_m"] > 0
    assert record["obs_velocity_drive_m"] > 0
    assert record["obs_candidate_increment"] > 0


@pytest.mark.parametrize("change", ("normal", "position", "velocity", "dt"))
def test_advance_rejects_arguments_that_differ_from_force(change):
    compensation = observer.ObservedCompensation("online")
    state, target = _state_target()
    _force(compensation, state, target)
    normal, dt = NORMAL, DT
    if change == "normal":
        normal = np.array([0.0, 1.0, 0.0])
    elif change == "position":
        state = replace(state, position=np.array([0.0, 0.01, 0.0]))
    elif change == "velocity":
        target = replace(target, linear_velocity=np.array([0.0, 0.03, 0.0]))
    else:
        dt = 2 * DT

    with pytest.raises(ValueError, match="advance .*disagrees"):
        compensation.advance(state, target, normal, dt, True)


def test_controller_factory_copies_every_parent_parameter(monkeypatch):
    parent = TangentialCompensation(
        "online",
        integral_gain=701.0,
        nominal_mu=0.3,
        max_force=6.0,
        velocity_scale=0.006,
        adaptation_gain=702.0,
        velocity_error_time=0.06,
        force_regularizer=2.2,
        max_equivalent_mu=0.8,
        min_update_speed=0.006,
        force_slew_rate=21.0,
        coefficient_rate_limit=0.2,
        motion_confirm_time=0.06,
    )
    control = SimpleNamespace(_base=SimpleNamespace(tangential=parent))
    monkeypatch.setattr(observer.rotation, "controller", lambda *args: control)

    actual = observer.controller(SurfaceFrame(np.eye(3)), 2.0, 8.0)._base.tangential

    assert isinstance(actual, observer.ObservedCompensation)
    for field in fields(TangentialCompensation):
        if field.init:
            expected = 8.0 if field.name == "max_force" else getattr(parent, field.name)
            assert getattr(actual, field.name) == expected


@pytest.mark.parametrize("scale,budget", ((True, 6.0), (1.0, 7.0)))
def test_controller_factory_rejects_invalid_frozen_choices(scale, budget):
    with pytest.raises((TypeError, ValueError)):
        observer.controller(SurfaceFrame(np.eye(3)), scale, budget)


def test_run_trial_mocked_single_step_orders_calls_and_adds_metadata(monkeypatch):
    calls = []
    observed = {
        name: np.array([1.0, 2.0, 3.0])
        if name.endswith(("_local", "_local_n", "_local_m"))
        else 1.0
        for name in observer.OBSERVATION_FIELDS
    }

    class FakeControl:
        def __init__(self):
            self._base = SimpleNamespace(
                tangential=SimpleNamespace(observation=observed)
            )

        def reset(self, state):
            calls.append(("reset", state))

        def compute(self, state, target, dt):
            calls.append(("compute", state, target, dt))
            return np.arange(6.0)

    class FakeSimulator:
        def __init__(self, frame, scenario, config, task, method):
            calls.append(("init", frame, scenario, config, task, method))
            self.config = config
            self.state = object()
            self.target = object()

        def sample(self):
            calls.append(("sample",))
            return SimpleNamespace(state=self.state, target=self.target)

        def step(self, wrench, control):
            calls.append(("step", wrench.copy(), control))

        def result(self):
            calls.append(("result",))
            return SimpleNamespace(trace={"time": np.array([0.0])})

    control = FakeControl()
    monkeypatch.setattr(observer, "controller", lambda *args: control)
    monkeypatch.setattr(observer, "SurfaceSimulator", FakeSimulator)
    case = {
        "case_index": 7,
        "task": SimpleNamespace(yaw_deg=0.0),
        "scenario": object(),
        "config": SimpleNamespace(duration=DT, timestep=DT),
    }

    result = observer.run_trial(case, 1.0, 8.0)

    assert [call[0] for call in calls] == [
        "init", "sample", "reset", "compute", "step", "result"
    ]
    np.testing.assert_array_equal(calls[4][1], np.arange(6.0))
    assert result.trace["rotation_gain_scale"] == 1.0
    assert result.trace["case_index"] == 7
    assert result.trace["method"] == "online"
    assert result.trace["max_force_n"] == 8.0
    for name in observer.OBSERVATION_FIELDS:
        assert result.trace[name].shape[0] == 1
    np.testing.assert_array_equal(result.trace["controller_frame_rotation"], np.eye(3))
