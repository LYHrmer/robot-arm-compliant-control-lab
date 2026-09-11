"""Fast structural tests for the measured-load pilot publisher; no physics is run."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import load_budget_study as study

SCHEDULE_FIELDS = (
    "time",
    "target_position",
    "target_linear_velocity",
    "target_normal_force",
    "applied_wall_friction",
    "applied_tool_friction",
    "applied_raw_wrench_bias_world",
    "feedback_raw_wrench_bias_world",
    "controller_yaw_error_deg",
    "trajectory_rate_scale",
)


def _schedule():
    return {
        "time": np.array([0.0, 0.002]),
        "target_position": np.zeros((2, 3)),
        "target_linear_velocity": np.ones((2, 3)),
        "target_normal_force": np.array([0.0, 12.0]),
        "applied_wall_friction": np.array([0.25, 0.65]),
        "applied_tool_friction": np.array([0.25, 0.65]),
        "applied_raw_wrench_bias_world": np.zeros((2, 6)),
        "feedback_raw_wrench_bias_world": np.zeros((2, 6)),
        "controller_yaw_error_deg": np.array([5.0, 5.0]),
        "trajectory_rate_scale": np.array([1.0, 1.0]),
    }


def test_protocol_thresholds_exactly_match_screening_constants():
    protocol = study.protocol_document()
    criteria = study.screening.CRITERIA

    assert protocol["stage_a"]["gates"] == {
        "post_8_12_s_tangent_rmse_max_mm": criteria["maximum_post_tangent_rmse_mm"],
        "post_reduction_min_pct": criteria["minimum_post_tangent_reduction_pct"],
        "late_10_12_s_tangent_rmse_max_mm": criteria["maximum_late_tangent_rmse_mm"],
        "worst_phase_force_rmse_increase_max_n": criteria[
            "maximum_phase_force_rmse_increase_n"
        ],
        "worst_phase_orientation_rmse_increase_max_deg": criteria[
            "maximum_phase_orientation_rmse_increase_deg"
        ],
        "full_peak_force_increase_max_n": criteria["maximum_full_raw_peak_increase_n"],
        "additional_budget_actually_exercised": True,
    }
    assert protocol["stage_a"]["candidate_gain2_minus_gain1_limits"] == dict(
        study.screening.GAIN_LIMITS
    )
    assert protocol["stage_b"]["candidate_minus_same_case_fixed6_limits"] == dict(
        study.screening.PUBLIC_LIMITS
    )
    assert protocol["absolute_checks_on_candidate_and_reference"] == {
        "contact_ratio_min_pct": study.screening.ABSOLUTE["contact_ratio_pct"],
        "full_raw_peak_force_max_n": study.screening.ABSOLUTE["peak_force_n"],
        "saturation_pct": study.screening.ABSOLUTE["saturation_pct"],
        "projection_pct": study.screening.ABSOLUTE["projection_pct"],
        "minimum_reserved_torque_headroom_nm": study.screening.ABSOLUTE[
            "minimum_reserved_torque_headroom_nm"
        ],
    }


def test_protocol_parent_pins_match_the_modules_used_for_reference_paths():
    parents = study.protocol_document()["parent_manifest_sha256"]

    assert parents[study.dynamic.REFERENCE["directory"]] == study.dynamic.REFERENCE[
        "manifest_sha256"
    ]
    assert parents[study.public.REFERENCE["directory"]] == study.public.REFERENCE[
        "manifest_sha256"
    ]
    for directory, digest in parents.items():
        assert study._sha256(study.ROOT / directory / "manifest.json") == digest


def test_dynamic_reference_selects_the_exact_scale_specific_parent_paths():
    cases = [case for variant, case in study.dynamic.cases() if variant == "combined"]

    for case in cases:
        scale1 = study.dynamic_reference(case, 1.0)
        assert scale1 == (
            study.ROOT
            / study.dynamic.REFERENCE["directory"]
            / study.dynamic.reference_trace("combined", case, 6.0)
        )
        yaw = int(case.scenario.wall_yaw_deg)
        assert study.dynamic_reference(case, 2.0) == (
            study.ROOT
            / f"results/franka_budget_transfer/traces/dynamic__yaw{yaw}__combined__s2__f6n.npz"
        )
        assert scale1.is_file()
        assert study.dynamic_reference(case, 2.0).is_file()


def test_exact_reference_check_accepts_and_reports_all_frozen_fields():
    reference = {"time": np.array([0.0, 0.002]), "value": np.array([1.0, 2.0])}

    result = study.exact_reference_check(deepcopy(reference), reference)

    assert result == {
        "checked_fields": ["time", "value"],
        "cycles": 2,
        "bit_exact": True,
    }


@pytest.mark.parametrize("tamper", ("missing", "value", "dtype"))
def test_exact_reference_check_rejects_any_non_bit_exact_field(tamper):
    reference = {"time": np.array([0.0, 0.002]), "value": np.array([1.0, 2.0])}
    trace = deepcopy(reference)
    if tamper == "missing":
        del trace["value"]
    elif tamper == "value":
        trace["value"][0] += 1.0
    else:
        trace["value"] = trace["value"].astype(np.float32)

    with pytest.raises(ValueError, match="fixed-budget physics reproduction differs"):
        study.exact_reference_check(trace, reference)


@pytest.mark.parametrize("field", SCHEDULE_FIELDS)
def test_paired_schedule_rejects_every_frozen_exogenous_field_mismatch(field):
    reference = _schedule()
    trace = deepcopy(reference)
    trace[field].flat[0] += 1.0

    with pytest.raises(ValueError, match=f"paired schedule differs: {field}"):
        study.paired_schedule(trace, reference)


def test_collect_stage_a_failure_does_not_run_stage_b(tmp_path, monkeypatch):
    cases = [
        SimpleNamespace(
            scenario=SimpleNamespace(wall_yaw_deg=yaw),
            task=SimpleNamespace(yaw_deg=yaw),
            controller_yaw_error_deg=0.0,
        )
        for yaw in (-15.0, 0.0, 15.0)
    ]
    public_case = {
        "case_index": 23,
        "task": SimpleNamespace(yaw_deg=0.0),
        "scenario": SimpleNamespace(wall_yaw_deg=0.0),
    }
    trace = {
        **_schedule(),
        "load_budget_applied_n": np.array([6.0, 6.1]),
        "controller_coefficient_before_compute": np.array([0.45, 0.9]),
        "diagnostic_corrected_force_n": np.array([12.0, 12.0]),
    }
    calls = []

    def fake_trace(directory, name, execute, trial, compact):
        calls.append(name)
        return deepcopy(trace)

    def fail_stage_a(rows):
        assert len(rows) == 6
        assert all(row["budget_exercised"] for row in rows)
        return {"status": "FAIL"}

    monkeypatch.setattr(study, "references", dict)
    monkeypatch.setattr(study.dynamic, "cases", lambda: [("combined", case) for case in cases])
    monkeypatch.setattr(study.public, "cases", lambda: [public_case])
    monkeypatch.setattr(study.public, "reference_trace", lambda *args: "reference.npz")
    monkeypatch.setattr(study, "dynamic_reference", lambda *args: Path("reference.npz"))
    monkeypatch.setattr(study, "_trace", fake_trace)
    monkeypatch.setattr(study.transfer, "load_trace", lambda path: deepcopy(trace))
    monkeypatch.setattr(study, "exact_reference_check", lambda *args: {"bit_exact": True})
    monkeypatch.setattr(study.validation, "validate_trace", lambda *args, **kwargs: {})
    monkeypatch.setattr(study.dynamic, "metrics", lambda *args: ({}, [], []))
    monkeypatch.setattr(study.screening, "dynamic_screen", fail_stage_a)
    monkeypatch.setattr(
        study.screening,
        "public_screen",
        lambda rows: pytest.fail("stage B screen must not run after stage A failure"),
    )

    result = study.collect(tmp_path, execute=True)

    assert len(calls) == 8
    assert not any(name.startswith("public__case") for name in calls)
    assert result["public"] == []
    assert result["screening"]["stage_b"] == {
        "status": "NOT_RUN",
        "reason": "stage_a_failed",
    }
    assert not result["screening"]["eligible_for_expansion"]


def test_collect_stage_a_pass_runs_exact_public_grid_with_unit_wall_normals(
    tmp_path, monkeypatch
):
    dynamic_cases = [
        SimpleNamespace(
            scenario=SimpleNamespace(wall_yaw_deg=yaw),
            task=SimpleNamespace(yaw_deg=yaw),
            controller_yaw_error_deg=0.0,
        )
        for yaw in (-15.0, 0.0, 15.0)
    ]
    indices = study.protocol_document()["stage_b"]["public_case_indices"]
    public_cases = [
        {
            "case_index": index,
            "task": SimpleNamespace(yaw_deg=yaw),
            "scenario": SimpleNamespace(wall_yaw_deg=yaw),
        }
        for index, yaw in zip(indices, (-15.0, -15.0, 0.0, 0.0, 15.0, 15.0), strict=True)
    ]
    trace = {
        **_schedule(),
        "load_budget_applied_n": np.array([6.0, 6.1]),
        "controller_coefficient_before_compute": np.array([0.45, 0.9]),
        "diagnostic_corrected_force_n": np.array([12.0, 12.0]),
    }
    calls, normals = [], []

    def fake_trace(directory, name, execute, trial, compact):
        calls.append(name)
        return deepcopy(trace)

    def catchup(data, normal):
        normal = np.asarray(normal)
        assert normal.shape == (3,)
        assert np.all(np.isfinite(normal))
        assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-12)
        normals.append(normal.copy())
        return {"samples": len(data["time"])}

    def public_screen(rows):
        assert len(rows) == 12
        assert {(row["case_index"], row["rotation_gain_scale"]) for row in rows} == {
            (index, scale) for index in indices for scale in (1.0, 2.0)
        }
        assert all(row["catchup_candidate"] == {"samples": 2} for row in rows)
        assert all(row["catchup_reference"] == {"samples": 2} for row in rows)
        return {"status": "PASS"}

    monkeypatch.setattr(study, "references", dict)
    monkeypatch.setattr(
        study.dynamic,
        "cases",
        lambda: [("combined", case) for case in dynamic_cases],
    )
    monkeypatch.setattr(study.public, "cases", lambda: public_cases)
    monkeypatch.setattr(study.public, "reference_trace", lambda *args: "reference.npz")
    monkeypatch.setattr(study, "dynamic_reference", lambda *args: Path("reference.npz"))
    monkeypatch.setattr(study, "_trace", fake_trace)
    monkeypatch.setattr(study.transfer, "load_trace", lambda path: deepcopy(trace))
    monkeypatch.setattr(study, "exact_reference_check", lambda *args: {"bit_exact": True})
    monkeypatch.setattr(study.validation, "validate_trace", lambda *args, **kwargs: {})
    monkeypatch.setattr(study.dynamic, "metrics", lambda *args: ({}, [], []))
    monkeypatch.setattr(study.public, "metrics", lambda data, case: {"case": case["case_index"]})
    monkeypatch.setattr(study, "catchup_metrics", catchup)
    monkeypatch.setattr(study.screening, "dynamic_screen", lambda rows: {"status": "PASS"})
    monkeypatch.setattr(study.screening, "public_screen", public_screen)
    monkeypatch.setattr(
        study.runner,
        "run_dynamic",
        lambda *args, **kwargs: pytest.fail("mock collect must not run dynamic physics"),
    )
    monkeypatch.setattr(
        study.runner,
        "run_public",
        lambda *args, **kwargs: pytest.fail("mock collect must not run public physics"),
    )

    result = study.collect(tmp_path, execute=True)

    assert len(calls) == 20
    assert len([name for name in calls if name.startswith("public__case")]) == 12
    assert len(normals) == 24
    expected_normals = {
        tuple(np.round(study.yaw_frame(yaw).rotation[:, 0], 12))
        for yaw in (-15.0, 0.0, 15.0)
    }
    assert {tuple(np.round(normal, 12)) for normal in normals} == expected_normals
    assert len(result["public"]) == 12
    assert result["screening"]["stage_a"]["status"] == "PASS"
    assert result["screening"]["stage_b"]["status"] == "PASS"
    assert result["screening"]["eligible_for_expansion"]


def test_run_failure_leaves_no_partial_publication(tmp_path, monkeypatch):
    output = tmp_path / "load-budget"
    monkeypatch.setattr(study, "source_identity", lambda: {"source.py": "a" * 64})
    monkeypatch.setattr(
        study,
        "collect",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        study.run(output)

    assert not output.exists()
