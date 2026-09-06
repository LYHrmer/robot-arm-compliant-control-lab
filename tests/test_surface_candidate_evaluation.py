import json
from dataclasses import asdict

import numpy as np
import pytest

import compliant_control_lab.surface_simulation as simulation
import tools.evaluate_surface_candidate as evaluation
from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_env import SurfaceLearningEnv
from compliant_control_lab.surface_policy_artifact import (
    freeze_evaluation_protocol,
    policy_contract,
    save_policy_artifact,
)
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    yaw_frame,
)
from compliant_control_lab.surface_splits import split_cases
from tools.surface_mlp_actor import mlp_payload, save_mlp_candidate


def _cases(duration=0.02):
    raw = [
        {
            "case_id": f"yaw{yaw}",
            "scenario": asdict(SurfaceScenario(wall_yaw_deg=yaw)),
            "config": asdict(SurfaceSimulationConfig(duration=duration, contact_model="smooth")),
            "task": asdict(LearningSurfaceTask(yaw_deg=yaw)),
            "controller_frame_rotation": yaw_frame(yaw).rotation.tolist(),
            "nominal_kind": "adaptive",
        }
        for yaw in (-15, 0, 15)
    ]
    return split_cases(raw, seed=20260906, train_fraction=0.7, validation_fraction=0.2)


def _frozen(tmp_path, duration=0.02):
    cases = _cases(duration)
    contract = policy_contract(cases, purpose="il_friction_teacher", nominal_kind="adaptive")
    candidate = tmp_path / "candidate.json"
    payload = {
        "kind": "linear_tanh",
        "weights": np.zeros((3, 49)).tolist(),
        "bias": [0, 0, 0],
    }
    save_policy_artifact(candidate, payload, contract=contract)
    protocol = tmp_path / "protocol.json"
    pinned = freeze_evaluation_protocol(
        protocol,
        candidate,
        expected_contract=contract,
        cases=cases,
        metrics=["force_rmse_n", "peak_force_n", "episode_return"],
        thresholds={
            "force_rmse_n": {"aggregation": "median", "operator": "<=", "value": 100.0},
            "peak_force_n": {"aggregation": "max", "operator": "<=", "value": 35.0},
            "episode_return": {"aggregation": "min", "operator": ">=", "value": -100.0},
        },
    )
    return candidate, protocol, pinned, cases


def _evaluate(tmp_path, monkeypatch):
    candidate, protocol, pinned, cases = _frozen(tmp_path)
    monkeypatch.setattr(evaluation, "_source_hashes", lambda: {"fixture_source": "fixed"})
    output = evaluation.evaluate_candidate(
        candidate,
        protocol,
        cases,
        pinned,
        tmp_path / "evaluation",
        purpose="il_friction_teacher",
        nominal_kind="adaptive",
    )
    return output, candidate, protocol, pinned, cases


def test_real_frozen_zero_fixture_writes_complete_selected_case_evidence(tmp_path, monkeypatch):
    output, candidate, _, pinned, cases = _evaluate(tmp_path, monkeypatch)
    report = json.loads((output / "report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    selected = [case for case in cases if case["split"] == "development_test"]
    assert report["evaluation_case_ids"] == [selected[0]["case_id"]]
    assert len(report["runs"]) == report["expected_case_count"] == 1
    row = report["runs"][0]
    assert row["truncated"] and not row["terminated"]
    assert row["complete_execution_log"] and row["trace_finite"] and row["endpoint_valid"]
    assert row["actor_failure_count"] == 0
    assert manifest["candidate_artifact_sha256"] == evaluation._sha256(candidate)
    assert manifest["protocol_sha256"] == pinned
    assert manifest["new_holdout"] is report["new_holdout"] is False
    assert manifest["scope"] == "complete_frozen_public_evaluation_selection"
    assert (output / "COMPLETE").read_text().strip() == evaluation._sha256(output / "manifest.json")
    assert {path.name for path in output.iterdir()} == {
        "case_000.npz",
        "case_000_events.json",
        "report.json",
        "summary.md",
        "manifest.json",
        "COMPLETE",
    }
    with np.load(output / row["trace_file"], allow_pickle=False) as stored:
        assert stored["decision_observation"].shape == (1, 49)
        assert stored["decision_action"].shape == (1, 3)
        assert stored["decision_terminal_observation_valid"].tolist() == [True]
        assert stored["decision_actor_succeeded"].tolist() == [True]
        assert stored["endpoint_valid"].shape == (10,)
        assert stored["intervention_active"].shape == (10,)
    events = json.loads((output / row["events_file"]).read_text())
    assert events["candidate_execution"]["format"] == "linear_tanh"
    assert events["decisions"][0]["info"]["actor_evaluation"]["reason"] is None


def test_no_tracking_window_is_unavailable_and_cannot_pass_via_other_metrics(tmp_path, monkeypatch):
    output, *_ = _evaluate(tmp_path, monkeypatch)
    report = json.loads((output / "report.json").read_text())
    row = report["runs"][0]
    assert not row["evaluation_observed"]
    assert row["force_rmse_n"] is None
    assert report["metrics"]["force_rmse_n"]["available"] is False
    assert report["metrics"]["peak_force_n"]["available"] is True
    assert report["gates"]["force_rmse_n"]["passed"] is False
    assert report["all_metrics_available"] is False
    assert report["all_episodes_succeeded"] is report["acceptance_met"] is False


def test_independent_physical_limits_cannot_be_omitted_by_protocol_metric_selection(
    tmp_path, monkeypatch
):
    candidate, protocol, pinned, cases = _frozen(tmp_path, duration=1.52)
    original = SurfaceLearningEnv.result

    def inconsistent_endpoint(env):
        result = original(env)
        result.trace["endpoint_linear_velocity"][-1] = [0.4, 0, 0]
        return result

    monkeypatch.setattr(SurfaceLearningEnv, "result", inconsistent_endpoint)
    output = evaluation.evaluate_candidate(
        candidate,
        protocol,
        cases,
        pinned,
        tmp_path / "endpoint-gate",
        purpose="il_friction_teacher",
        nominal_kind="adaptive",
    )
    report = json.loads((output / "report.json").read_text())
    row = report["runs"][0]
    assert row["truncated"] and not row["terminated"] and row["evaluation_observed"]
    assert row["max_speed_m_s"] == 0.4
    assert all(gate["passed"] for gate in report["gates"].values())
    assert not row["independent_physical_gates_met"] and not report["acceptance_met"]


@pytest.mark.parametrize("target", ["protocol", "candidate"])
def test_identity_tampering_is_rejected_before_environment_reset(tmp_path, monkeypatch, target):
    candidate, protocol, pinned, cases = _frozen(tmp_path)
    path = protocol if target == "protocol" else candidate
    path.write_text(path.read_text() + " \n")
    reset_calls = []

    def forbidden_reset(self, *args, **kwargs):
        reset_calls.append((args, kwargs))
        raise AssertionError("reset must not run")

    monkeypatch.setattr(SurfaceLearningEnv, "reset", forbidden_reset)
    with pytest.raises(ValueError):
        evaluation.evaluate_candidate(
            candidate,
            protocol,
            cases,
            pinned,
            tmp_path / "unpublished",
            purpose="il_friction_teacher",
            nominal_kind="adaptive",
        )
    assert reset_calls == []
    assert not (tmp_path / "unpublished").exists()


def test_safety_failure_is_retained_and_forces_failure_despite_return_gate(tmp_path, monkeypatch):
    candidate, protocol, pinned, cases = _frozen(tmp_path)
    monkeypatch.setattr(evaluation, "_source_hashes", lambda: {"fixture_source": "fixed"})
    monkeypatch.setattr(simulation, "_normal_contact_force", lambda *_: 36.0)
    output = evaluation.evaluate_candidate(
        candidate,
        protocol,
        cases,
        pinned,
        tmp_path / "failure",
        purpose="il_friction_teacher",
        nominal_kind="adaptive",
    )
    report = json.loads((output / "report.json").read_text())
    assert len(report["runs"]) == 1
    row = report["runs"][0]
    assert row["terminated"] and not row["truncated"]
    assert row["termination_reasons"]
    assert row["episode_return"] <= -1
    assert not row["episode_success"]
    assert report["all_episodes_succeeded"] is report["acceptance_met"] is False
    assert (output / row["trace_file"]).exists()
    assert (output / row["events_file"]).exists()


def test_restricted_mlp_candidate_runs_through_actual_short_evaluation(tmp_path, monkeypatch):
    cases = _cases()
    contract = policy_contract(cases, purpose="il_friction_teacher", nominal_kind="adaptive")
    candidate = tmp_path / "mlp.json"
    layers = [
        {"weights": np.zeros((2, 49)).tolist(), "bias": [0.2, -0.1]},
        {"weights": [[1, 0], [0, 1], [0.5, -0.5]], "bias": [0, 0, 0]},
    ]
    save_mlp_candidate(candidate, layers, contract=contract, activation="relu")
    protocol = tmp_path / "mlp_protocol.json"
    pinned = freeze_evaluation_protocol(
        protocol,
        candidate,
        expected_contract=contract,
        cases=cases,
        metrics=["peak_force_n"],
        thresholds={"peak_force_n": {"aggregation": "max", "operator": "<=", "value": 35}},
    )
    monkeypatch.setattr(evaluation, "_source_hashes", lambda: {"fixture_source": "fixed"})
    output = evaluation.evaluate_candidate(
        candidate,
        protocol,
        cases,
        pinned,
        tmp_path / "mlp_evaluation",
        purpose="il_friction_teacher",
        nominal_kind="adaptive",
    )
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert manifest["candidate_execution"]["format"] == "surface_numpy_mlp_v1"
    assert report["runs"][0]["actor_failure_count"] == 0
    with np.load(output / "case_000.npz", allow_pickle=False) as stored:
        assert np.any(stored["decision_action"] != 0)


def test_mlp_runner_identity_tampering_is_rejected_before_reset(tmp_path, monkeypatch):
    cases = _cases()
    contract = policy_contract(cases, purpose="il_friction_teacher", nominal_kind="adaptive")
    layers = [
        {"weights": np.zeros((2, 49)).tolist(), "bias": [0, 0]},
        {"weights": np.zeros((3, 2)).tolist(), "bias": [0, 0, 0]},
    ]
    payload = mlp_payload(layers)
    payload["runner_identity"]["runner_sha256"] = "0" * 64
    candidate = tmp_path / "tampered_runner.json"
    save_policy_artifact(candidate, payload, contract=contract)
    protocol = tmp_path / "tampered_runner_protocol.json"
    pinned = freeze_evaluation_protocol(
        protocol,
        candidate,
        expected_contract=contract,
        cases=cases,
        metrics=["peak_force_n"],
        thresholds={"peak_force_n": {"aggregation": "max", "operator": "<=", "value": 35}},
    )
    reset_calls = []

    def forbidden_reset(self, *args, **kwargs):
        reset_calls.append((args, kwargs))
        raise AssertionError("reset must not run")

    monkeypatch.setattr(SurfaceLearningEnv, "reset", forbidden_reset)
    with pytest.raises(ValueError, match="runner identity"):
        evaluation.evaluate_candidate(
            candidate,
            protocol,
            cases,
            pinned,
            tmp_path / "unpublished_mlp",
            purpose="il_friction_teacher",
            nominal_kind="adaptive",
        )
    assert reset_calls == []
