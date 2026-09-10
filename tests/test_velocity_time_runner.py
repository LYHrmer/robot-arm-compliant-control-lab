"""Seam tests for the velocity-error time runner; no simulator physics is run."""

import json
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_simulation import yaw_frame
from compliant_control_lab.tangential_compensation import TangentialCompensation
from tools import onset_observer as observer
from tools import velocity_time_runner as runner

DT = 0.002
INIT_FIELDS = tuple(f.name for f in fields(TangentialCompensation) if f.init)


def _compensation(time_s, budget=6.0):
    return runner.controller(yaw_frame(0.0), 1.0, budget, time_s)._base.tangential


def test_factory_changes_only_the_velocity_error_time():
    original = observer.controller(yaw_frame(0.0), 1.0, 6.0)._base.tangential
    actual = _compensation(0.10)

    assert isinstance(actual, observer.ObservedCompensation)
    assert actual.velocity_error_time == 0.10
    for name in INIT_FIELDS:
        if name != "velocity_error_time":
            assert getattr(actual, name) == getattr(original, name), name


def test_default_time_leaves_the_fresh_observer_controller_unchanged():
    original = observer.controller(yaw_frame(0.0), 1.0, 6.0)._base.tangential
    assert original.velocity_error_time == runner.DEFAULT_VELOCITY_ERROR_TIME_S == 0.05

    actual = _compensation(0.05)
    for name in INIT_FIELDS:
        assert getattr(actual, name) == getattr(original, name), name


def test_parameters_report_json_native_executed_fields():
    values = runner.parameters(8.0, 0.10)

    assert sorted(values) == sorted(INIT_FIELDS)
    assert values == json.loads(json.dumps(values))
    assert values["velocity_error_time"] == 0.10
    assert values["max_force"] == 8.0
    assert values["mode"] == "online"


def test_controller_document_reports_only_the_changed_time():
    default = runner.controller_document(2.0, 8.0, 0.05)
    actual = runner.controller_document(2.0, 8.0, 0.10)

    assert default["tangential"]["velocity_error_time"] == 0.05
    assert actual["tangential"]["velocity_error_time"] == 0.10
    default["tangential"]["velocity_error_time"] = 0.10
    assert actual == default


@pytest.mark.parametrize("time_s", (True, 0.07, float("nan"), "0.05"))
def test_factory_rejects_invalid_velocity_error_times(time_s):
    with pytest.raises((TypeError, ValueError)):
        runner.controller(SurfaceFrame(np.eye(3)), 1.0, 6.0, time_s)


def _case(index=23, cycles=1):
    return {
        "case_index": index,
        "task": SimpleNamespace(yaw_deg=0.0),
        "scenario": object(),
        "config": SimpleNamespace(duration=cycles * DT, timestep=DT),
    }


@pytest.mark.parametrize("cycles", (1, 2))
def test_run_trial_mocked_loop_orders_calls_and_adds_metadata(monkeypatch, cycles):
    calls = []
    observed = {name: 1.0 for name in runner.OBSERVATION_FIELDS}

    class FakeControl:
        def __init__(self):
            self._base = SimpleNamespace(tangential=SimpleNamespace(observation=observed))

        def reset(self, state):
            calls.append(("reset",))

        def compute(self, state, target, dt):
            calls.append(("compute", dt))
            return np.arange(6.0)

    class FakeSimulator:
        def __init__(self, frame, scenario, config, task, method):
            calls.append(("init",))
            self.config = config

        def sample(self):
            calls.append(("sample",))
            return SimpleNamespace(state=object(), target=object())

        def step(self, wrench, control):
            calls.append(("step", wrench.copy()))

        def result(self):
            calls.append(("result",))
            return SimpleNamespace(trace={"time": np.zeros(cycles)})

    monkeypatch.setattr(runner, "controller", lambda *args: FakeControl())
    monkeypatch.setattr(runner, "SurfaceSimulator", FakeSimulator)

    result = runner.run_trial(_case(cycles=cycles), 1.0, 8.0, 0.10)

    expected = ["init", "sample", "reset", "compute", "step"]
    expected += ["sample", "compute", "step"] * (cycles - 1)
    assert [call[0] for call in calls] == [*expected, "result"]
    assert [call[1] for call in calls if call[0] == "compute"] == [DT] * cycles
    np.testing.assert_array_equal(calls[4][1], np.arange(6.0))
    assert result.trace["time"].shape == (cycles,)
    assert result.trace["velocity_error_time_s"] == 0.10
    assert result.trace["max_force_n"] == 8.0
    assert result.trace["rotation_gain_scale"] == 1.0
    assert result.trace["case_index"] == runner.CASE_INDEX
    assert result.trace["method"] == "online"
    for name in runner.OBSERVATION_FIELDS:
        assert result.trace[name].shape == (cycles,)
    np.testing.assert_array_equal(result.trace["controller_frame_rotation"], np.eye(3))


def test_run_trial_rejects_any_case_other_than_the_executed_one():
    with pytest.raises(ValueError, match="case 23"):
        runner.run_trial(_case(index=7), 1.0, 6.0, 0.05)
