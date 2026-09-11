"""Unit tests for the measured-load trial adapter; no physics is executed."""

from types import SimpleNamespace

import numpy as np
import pytest

from compliant_control_lab.surface_simulation import yaw_frame
from tools import budget_gain_interaction as dynamic
from tools import budget_public24 as public
from tools import load_budget_trial as trial
from tools.load_aware_compensation import LoadAwareCompensation


def test_public_compaction_preserves_the_independent_validator_inputs():
    from tests.test_load_budget_validation import _trace
    from tools.load_budget_validation import validate_trace

    frame, trace = _trace()
    for field in (*public.COMPACT_FIELDS, *public.TRACE_METADATA):
        trace.setdefault(field, np.zeros(len(trace["time"])))
    trace["controller_update_ready_before_compute"] = np.zeros(len(trace["time"]), dtype=bool)
    compact = trial.compact_public(trace)
    assert "contact_blend" in compact
    assert validate_trace(compact, frame)["validated_cycles"] == len(trace["time"])


def test_load_aware_replacement_preserves_original_online_parameters():
    case = dynamic.cases()[0][1]
    controller = dynamic.controller(case, 1.0, 8.0)
    original = controller._base.tangential

    trial._load_aware(controller, 6.0, 8.0)
    replacement = controller._base.tangential

    assert isinstance(replacement, LoadAwareCompensation)
    assert replacement.minimum_force == 6.0
    assert replacement.max_force == 8.0
    for name in (
        "mode",
        "nominal_mu",
        "velocity_scale",
        "adaptation_gain",
        "velocity_error_time",
        "force_regularizer",
        "force_slew_rate",
        "coefficient_rate_limit",
    ):
        assert getattr(replacement, name) == getattr(original, name)


def test_observer_reconstructs_against_applied_not_global_ceiling():
    compensation = LoadAwareCompensation(mode="online", max_force=8.0, minimum_force=6.0)
    compensation._active = True
    compensation._applied_budget = 6.0
    compensation._last_force = np.array([0.0, 0.04, 0.0])
    base = SimpleNamespace(
        tangential=compensation,
        base=SimpleNamespace(base=SimpleNamespace(normal=np.array([1.0, 0.0, 0.0]))),
    )
    controller = SimpleNamespace(
        _base=base,
        frame=yaw_frame(0.0),
        equivalent_tangential_coefficient=0.6,
        corrected_force_n=12.0,
        contact_blend=1.0,
    )
    observer = trial.LoadBudgetObserver(controller)
    snapshot = {
        "coefficient": 0.6,
        "previous_force": np.zeros(3),
        "target_velocity": np.array([0.0, 0.02, 0.0]),
        "dt": 0.002,
    }

    diagnostics = observer.after_compute(snapshot)

    assert diagnostics["diagnostic_amplitude_capped"] is True
    assert observer.mismatch_cycles == 0
    assert compensation.max_force == 8.0


class _FakeCompensation:
    def __init__(self):
        self.next_budget_n = 6.0
        self.applied_budget_n = 6.0
        self.load_estimate_n = 0.0
        self.measurement_used_for_budget = False
        self.projected_load_n = 0.0
        self.last_force = np.zeros(3)
        self.pending = None

    def set_force_measurement(self, force):
        self.pending = np.asarray(force).copy()


class _FakeController:
    def __init__(self, events):
        self._base = SimpleNamespace(tangential=_FakeCompensation())
        self.equivalent_tangential_coefficient = 0.45
        self.tangential_update_ready = True
        self.last_torque_projection_scale = 1.0
        self.torque_projection_fallback_count = 0
        self.fallback_on_compute = False
        self.events = events

    def reset(self, _state):
        self.events.append("reset")

    def compute(self, _state, _target, _dt):
        compensation = self._base.tangential
        assert compensation.pending is not None
        self.events.append("compute")
        compensation.applied_budget_n = compensation.next_budget_n
        compensation.next_budget_n = 7.0
        compensation.load_estimate_n = 6.75
        compensation.projected_load_n = 6.5
        compensation.measurement_used_for_budget = True
        compensation.last_force = compensation.pending
        compensation.pending = None
        self.equivalent_tangential_coefficient += 0.01
        self.torque_projection_fallback_count += int(self.fallback_on_compute)
        return np.zeros(6)


class _FakeObserver:
    def __init__(self, _controller):
        self.max_reconstruction_error_n = 0.0
        self.mismatch_cycles = 0

    def before_compute(self, _target, _dt):
        return {}

    def after_compute(self, _snapshot):
        return {
            "diagnostic_corrected_force_n": 12.0,
            "diagnostic_compensation_active": True,
            "diagnostic_amplitude_capped": False,
            "diagnostic_slew_limited": False,
        }


class _FakeSimulator:
    def __init__(self, events, *, has_actuation=True):
        self._input_row = {}
        self.events = events
        self.index = 0
        self.has_actuation = has_actuation

    def sample(self):
        self._input_row = {"index": self.index}
        state = SimpleNamespace(actuation=object() if self.has_actuation else None)
        return SimpleNamespace(state=state, target=object(), time=self.index * 0.002)

    def step(self, _wrench, _controller, telemetry_before=None):
        self.events.append("step")
        self.index += 1

    def result(self):
        return SimpleNamespace(trace={})


def test_loop_uses_measurement_before_compute_and_records_cycle_budget(monkeypatch):
    events = []
    simulator = _FakeSimulator(events)
    controller = _FakeController(events)
    case = SimpleNamespace(config=SimpleNamespace(timestep=0.002, duration=0.004))
    monkeypatch.setattr(trial, "LoadBudgetObserver", _FakeObserver)

    def measurement(row, _sample, _frame):
        events.append("measurement")
        value = float(row["index"] + 1)
        return {
            "force_local": np.array([0.0, value, 0.0]),
            "measurement_time_s": value,
            "measurement_age_s": 0.0,
        }

    monkeypatch.setattr(trial, "measured_force_input", measurement)

    result = trial._run_loop(case, controller, simulator, object(), scheduled=True)

    assert events == [
        "reset", "measurement", "compute", "step",
        "measurement", "compute", "step",
    ]
    np.testing.assert_array_equal(result.trace["load_budget_applied_n"], [6.0, 7.0])
    np.testing.assert_array_equal(result.trace["load_budget_next_n"], [7.0, 7.0])
    np.testing.assert_array_equal(result.trace["load_budget_updated"], [True, True])
    np.testing.assert_array_equal(result.trace["load_projection_accepted"], [True, True])
    np.testing.assert_array_equal(
        result.trace["controller_coefficient_before_compute"], [0.45, 0.46]
    )
    np.testing.assert_allclose(
        result.trace["controller_coefficient_after_compute"], [0.46, 0.47]
    )


@pytest.mark.parametrize("blocker", ("no_actuation", "fallback", "scaled"))
def test_loop_records_projection_rejection_reason_independently(blocker, monkeypatch):
    events = []
    simulator = _FakeSimulator(events, has_actuation=blocker != "no_actuation")
    controller = _FakeController(events)
    controller.fallback_on_compute = blocker == "fallback"
    controller.last_torque_projection_scale = 0.9 if blocker == "scaled" else 1.0
    case = SimpleNamespace(config=SimpleNamespace(timestep=0.002, duration=0.002))
    monkeypatch.setattr(trial, "LoadBudgetObserver", _FakeObserver)
    monkeypatch.setattr(
        trial,
        "measured_force_input",
        lambda *_args: {
            "force_local": np.array([0.0, 1.0, 0.0]),
            "measurement_time_s": 0.0,
            "measurement_age_s": 0.0,
        },
    )

    result = trial._run_loop(case, controller, simulator, object(), scheduled=True)

    assert not result.trace["load_projection_accepted"][0]


@pytest.mark.parametrize("kind", ("dynamic", "public"))
def test_compact_retains_parent_and_scheduler_fields(kind, monkeypatch):
    parent = trial.dynamic if kind == "dynamic" else trial.public
    monkeypatch.setattr(parent, "compact", lambda _trace: {"parent": np.array(1.0)})
    trace = {name: np.array(1.0) for name in trial.TELEMETRY_FIELDS}
    trace["measured_wrench_world"] = np.ones((1, 6))

    compacted = getattr(trial, f"compact_{kind}")(trace)

    assert compacted["parent"] == 1.0
    assert set(trial.TELEMETRY_FIELDS) <= set(compacted)
    np.testing.assert_array_equal(compacted["measured_wrench_world"], np.ones((1, 6)))


def test_run_dynamic_sets_global_metadata_without_physics(monkeypatch):
    case = dynamic.cases()[0][1]
    captured = {}
    monkeypatch.setattr(trial, "ScheduledSurfaceSimulator", lambda *_args: object())

    def fake_loop(_case, controller, _simulator, _frame, *, scheduled):
        captured["compensation"] = controller._base.tangential
        captured["scheduled"] = scheduled
        return SimpleNamespace(trace={})

    monkeypatch.setattr(trial, "_run_loop", fake_loop)

    result = trial.run_dynamic(case, 2.0)

    assert isinstance(captured["compensation"], LoadAwareCompensation)
    assert captured["scheduled"] is True
    assert result.trace["rotation_gain_scale"] == 2.0
    assert result.trace["max_force_n"] == 8.0


def test_run_public_sets_case_method_and_global_metadata_without_physics(monkeypatch):
    case = public.cases()[0]
    captured = {}
    monkeypatch.setattr(trial, "SurfaceSimulator", lambda *_args: object())

    def fake_loop(_case, controller, _simulator, _frame, *, scheduled):
        captured["compensation"] = controller._base.tangential
        captured["scheduled"] = scheduled
        return SimpleNamespace(trace={})

    monkeypatch.setattr(trial, "_run_loop", fake_loop)

    result = trial.run_public(case, 1.0)

    assert isinstance(captured["compensation"], LoadAwareCompensation)
    assert captured["scheduled"] is False
    assert result.trace["case_index"] == case["case_index"]
    assert result.trace["method"] == "online"
    assert result.trace["max_force_n"] == 8.0
