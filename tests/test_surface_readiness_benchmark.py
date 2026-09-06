import json
from copy import deepcopy

import numpy as np
import pytest

import compliant_control_lab.surface_readiness_benchmark as benchmark
from compliant_control_lab.surface_readiness_cases import preparation_cases
from compliant_control_lab.surface_simulation import SurfaceSimulator
from compliant_control_lab.surface_splits import case_group_id


def _short_case(index=0, *, duration=0.04):
    case = deepcopy(preparation_cases()[index])
    case["config"]["duration"] = duration
    return case


@pytest.fixture(scope="module")
def completed_run():
    case = _short_case()
    row, trace, evidence = benchmark.run_case(case, "zero_friction")
    return case, row, trace, evidence


def test_short_real_case_has_complete_independently_auditable_evidence(completed_run):
    case, row, trace, evidence = completed_run
    assert row["physics_steps"] == 20 and row["policy_steps"] == 2
    assert row["truncated"] and not row["terminated"]
    assert row["complete_execution_log"] and row["all_actions_legal"]
    assert row["hard_safe_completion"] and not row["engineering_targets_met"]
    assert row["endpoint_valid"] and evidence["illegal_action_attempted"] is False
    assert trace["decision_observation"].shape == (3, 49)
    assert trace["decision_action"].shape == (2, 3)
    audited = benchmark.audit_trace(trace, case)
    assert all(row[key] == value for key, value in audited.items())


def test_endpoint_geometry_and_speed_are_included_in_independent_maxima(completed_run):
    case, _, original, _ = completed_run
    trace = deepcopy(original)
    angle = np.deg2rad(case["scenario"]["wall_yaw_deg"])
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    trace["endpoint_position"][-1] = np.array([0.4, 0.0, 0.0]) - 0.022 * normal
    trace["endpoint_contact_gap_m"][-1] = (
        np.array([0.4, 0.0, 0.0]) - trace["endpoint_position"][-1]
    ) @ normal - 0.025
    trace["endpoint_linear_velocity"][-1] = [0.3001, 0.0, 0.0]
    metrics = benchmark.audit_trace(trace, case)
    assert metrics["max_penetration_mm"] == pytest.approx(3.0)
    assert metrics["max_speed_m_s"] == pytest.approx(0.3001)


@pytest.mark.parametrize("fault", ("time", "gap", "shape"))
def test_endpoint_evidence_must_be_complete_and_geometrically_consistent(completed_run, fault):
    case, _, original, _ = completed_run
    trace = deepcopy(original)
    if fault == "time":
        trace["endpoint_time"][-1] += 0.001
    elif fault == "gap":
        trace["endpoint_contact_gap_m"][-1] += 0.001
    else:
        trace["endpoint_position"] = trace["endpoint_position"][:-1]
    with pytest.raises(ValueError, match="endpoint"):
        benchmark.audit_trace(trace, case)


def test_false_endpoint_validity_is_retained_as_a_failed_run(monkeypatch):
    original = benchmark.SurfaceLearningEnv.result

    def invalidate_last_endpoint(env):
        result = original(env)
        result.trace["endpoint_valid"][-1] = False
        return result

    monkeypatch.setattr(benchmark.SurfaceLearningEnv, "result", invalidate_last_endpoint)
    row, trace, _ = benchmark.run_case(_short_case(), "zero_friction")
    assert not trace["endpoint_valid"][-1]
    assert not row["endpoint_valid"]
    assert not row["hard_safe_completion"]
    assert "invalid_endpoint_evidence" in row["termination_reasons"]


def test_normal_truncation_cannot_hide_an_endpoint_speed_violation(monkeypatch):
    original = benchmark.SurfaceLearningEnv.result

    def inject_endpoint_speed(env):
        result = original(env)
        result.trace["endpoint_linear_velocity"][-1] = [0.3001, 0.0, 0.0]
        return result

    monkeypatch.setattr(benchmark.SurfaceLearningEnv, "result", inject_endpoint_speed)
    row, _, _ = benchmark.run_case(_short_case(), "zero_friction")
    assert row["truncated"] and not row["terminated"]
    assert row["max_speed_m_s"] == pytest.approx(0.3001)
    assert not row["hard_safe_completion"]


def test_generated_short_report_is_atomic_hashed_and_reauditable(tmp_path):
    output = benchmark.generate_readiness_benchmark(
        tmp_path / "short", cases=[_short_case()], methods=["zero_friction"]
    )
    assert {path.name for path in output.iterdir()} == {
        "COMPLETE",
        "case_000_zero_friction.json",
        "case_000_zero_friction.npz",
        "comparison.csv",
        "manifest.json",
        "report.json",
        "summary.md",
    }
    assert benchmark.audit_benchmark(output) == {
        "matches": True,
        "run_count": 1,
        "physics_steps": 20,
    }
    with (output / "comparison.csv").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    with pytest.raises(ValueError, match="artifact mismatch"):
        benchmark.audit_benchmark(output)


def test_illegal_policy_failure_still_publishes_and_audits_all_available_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(benchmark, "_action", lambda *_args: np.array([np.nan, 0.0, 0.0]))
    output = benchmark.generate_readiness_benchmark(
        tmp_path / "failed", cases=[_short_case()], methods=["zero_friction"]
    )
    with np.load(output / "case_000_zero_friction.npz", allow_pickle=False) as trace:
        assert trace["decision_action"].shape == (0, 3)
        assert trace["decision_observation"].shape == (1, 49)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    row = report["runs"][0]
    assert not row["all_actions_legal"] and not row["hard_safe_completion"]
    assert row["physics_steps"] == 0 and "benchmark_exception" in row["termination_reasons"]
    assert benchmark.audit_benchmark(output)["run_count"] == 1


def test_three_observation_failures_publish_raw_nan_and_json_null_evidence(tmp_path, monkeypatch):
    original = SurfaceSimulator.evaluator_kinematics

    def nonfinite_endpoint(simulator):
        endpoint = original(simulator)
        endpoint["endpoint_position"][0] = np.nan
        endpoint["endpoint_valid"] = False
        return endpoint

    monkeypatch.setattr(SurfaceSimulator, "evaluator_kinematics", nonfinite_endpoint)
    cases = [_short_case(index) for index in (0, 2, 4)]
    output = benchmark.generate_readiness_benchmark(
        tmp_path / "observation-failures", cases=cases, methods=["zero_friction"]
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert len(report["runs"]) == 3 and not report["acceptance_met"]
    for index, row in enumerate(report["runs"]):
        assert row["physics_steps"] == 1 and row["terminated"]
        assert not row["trace_finite"] and not row["endpoint_valid"]
        assert "observation_failure" in row["termination_reasons"]
        events = json.loads(
            (output / f"case_{index:03d}_zero_friction.json").read_text(encoding="utf-8")
        )
        stage = events["decisions"][0]["action_stages"][0]
        assert stage["endpoint_position"] == [None, None, None]
        assert stage["endpoint_time"] is None and not stage["endpoint_valid"]
        with np.load(output / f"case_{index:03d}_zero_friction.npz", allow_pickle=False) as trace:
            assert np.isnan(trace["endpoint_position"][0]).all()
            assert np.isnan(trace["endpoint_time"][0])
            assert not trace["endpoint_valid"][0]
    assert benchmark.audit_benchmark(output)["run_count"] == 3


def test_generation_refuses_existing_destination_and_cleans_failed_staging(tmp_path, monkeypatch):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="new directory"):
        benchmark.generate_readiness_benchmark(
            existing, cases=[_short_case()], methods=["zero_friction"]
        )

    def fail_run(*_args):
        raise RuntimeError("injected benchmark failure")

    monkeypatch.setattr(benchmark, "run_case", fail_run)
    destination = tmp_path / "never-published"
    with pytest.raises(RuntimeError, match="injected"):
        benchmark.generate_readiness_benchmark(
            destination, cases=[_short_case()], methods=["zero_friction"]
        )
    assert not destination.exists()


def test_stress_mode_uses_exactly_three_anchors_and_four_declared_actions(
    tmp_path, monkeypatch, completed_run
):
    _, template_row, template_trace, template_evidence = completed_run
    seen = []

    def record(case, method):
        seen.append((case["scenario"]["name"], case["config"]["seed"], method))
        row = deepcopy(template_row)
        row.update(
            case_id=case["case_id"],
            group_id=case_group_id(case),
            method=method,
            split=case["split"],
        )
        return row, deepcopy(template_trace), deepcopy(template_evidence)

    monkeypatch.setattr(benchmark, "run_case", record)
    cases = [_short_case(index, duration=0.02) for index in range(6)]
    output = benchmark.generate_readiness_benchmark(tmp_path / "stress", cases=cases, stress=True)
    assert len(seen) == 12
    assert {name for name, _, _ in seen} == {
        "preparation_000",
        "preparation_001",
        "preparation_002",
    }
    assert {method for _, _, method in seen} == set(benchmark.STRESS_METHODS)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stress"] and manifest["methods"] == list(benchmark.STRESS_METHODS)


def test_public_thresholds_and_method_interfaces_are_fixed_before_results():
    assert benchmark.METHODS == (
        "zero_adaptive",
        "zero_friction",
        "friction_teacher_50hz",
    )
    assert benchmark.ENGINEERING_TARGETS == {
        "contact_ratio_pct_min": 99.0,
        "tangent_rmse_mm_max": 10.0,
        "force_rmse_n_max": 2.0,
    }
