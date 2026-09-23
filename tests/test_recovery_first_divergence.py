"""Arithmetic diagnostics preserve replay behavior and the caller's trace hook."""

import json
import sys

import numpy as np
import pytest

from tools.ci import recovery_first_divergence as probe


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("the first-divergence diagnostic must not run physics")

    monkeypatch.setattr(probe.study.runner, "_run_loop", forbidden)


def test_first_reports_earliest_row_and_column_with_exact_float_values():
    saved = np.arange(9, dtype=np.float64).reshape(3, 3)
    rebuilt = saved.copy()
    rebuilt[1, 1] = np.nextafter(saved[1, 1], np.inf)
    rebuilt[1, 2] += 100
    rebuilt[2, 0] -= 100
    result = probe._first(saved, rebuilt, ("applied", "load", "projected"), np.array([0, 0.002, 0.004]))
    assert result == {
        "index": 1, "column": "load", "time_s": 0.002,
        "saved_hex": float(saved[1, 1]).hex(),
        "rebuilt_hex": float(rebuilt[1, 1]).hex(),
        "absolute_difference": float(rebuilt[1, 1] - saved[1, 1]),
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_first_returns_none_when_reconstructed_values_match():
    saved = np.array([[1.0, -2.0], [0.0, 4.0]])
    assert probe._first(saved, saved.copy(), ("x", "y"), np.array([0, 0.002])) is None


def _sentinel_trace(_frame, _event, _argument):
    return None


@pytest.mark.parametrize("prior_hook", [None, _sentinel_trace])
def test_actual_trace_diagnostic_matches_independent_same_process_replay_and_restores_hook(prior_hook):
    study = probe.study
    protocol = study.protocol_document()
    spec = study.specifications(protocol)[0]
    _, parameters = study.make_controller(spec, protocol)
    path = study.ROOT / "results/franka_reversal_recovery" / spec["trace_path"]
    trace = study.previous._load_trace(path)
    expected = study.replay_compensation(trace, parameters, spec["method"], protocol["timestep_s"])
    previous = sys.gettrace()
    try:
        sys.settrace(prior_hook)
        report = probe.first_divergence()
        assert sys.gettrace() is prior_hook
    finally:
        sys.settrace(previous)
    # Recomputed residuals can vary across CPUs; no archived-float or zero assumption.
    assert report["audit"] == expected
    assert report["identity"] == {"scenario": "falling", "seed": 11, "method": "adaptive6_8"}
    assert report["trace_path"] == path.relative_to(study.ROOT).as_posix()
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    for field in ("first_scheduler_difference", "first_compensation_difference", "first_coefficient_difference"):
        difference = report[field]
        if difference is not None:
            assert 0 <= difference["index"] < len(trace["time"])
            assert difference["time_s"] == float(trace["time"][difference["index"]])
            assert difference["absolute_difference"] == abs(
                float.fromhex(difference["saved_hex"]) - float.fromhex(difference["rebuilt_hex"])
            )


@pytest.mark.parametrize("prior_hook", [None, _sentinel_trace])
def test_early_reconstruction_exception_is_not_masked_and_restores_hook(monkeypatch, prior_hook):
    class ReconstructionFailure(RuntimeError):
        pass

    failure = ReconstructionFailure("injected failure before replay locals exist")

    def reject(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(probe.study.previous, "_close", reject)
    previous = sys.gettrace()
    try:
        sys.settrace(prior_hook)
        with pytest.raises(ReconstructionFailure) as caught:
            probe.first_divergence()
        assert caught.value is failure
        assert sys.gettrace() is prior_hook
    finally:
        sys.settrace(previous)
