"""Protocol, provenance, collection and archive tests for the robustness study."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from compliant_control_lab.online_compensation_experiment import ProtocolPhase
from compliant_control_lab.surface_simulation import yaw_frame
from tools import measured_budget_robustness as study


def test_protocol_completely_freezes_the_twenty_seven_run_design():
    protocol = study.protocol_document()
    specs = study.specifications()

    assert protocol["identity"] == study.IDENTITY
    assert protocol["new_simulations"] == len(specs) == 27
    assert protocol["scenario_count"] == 9
    assert protocol["seed"] == 11
    assert protocol["gain_scale"] == 1.0
    assert protocol["duration_s"] == 12.0
    assert protocol["timestep_s"] == 0.002
    assert protocol["maximum_packet_age_s"] == 0.020
    assert list(protocol["methods"]) == list(study.METHODS)
    assert len(protocol["scenarios"]) == 9
    assert protocol["full_trace_fields"] == sorted(study.FULL_TRACE_FIELDS)
    assert len(study.FULL_TRACE_FIELDS) == 88
    assert json.loads(json.dumps(protocol)) == protocol


def test_every_fault_has_both_fixed_controls_and_the_cross_direction_case_is_fresh():
    specs = study.specifications()
    identities = [(spec["scenario_id"], spec["method"]) for spec in specs]

    assert len(set(identities)) == 27
    for fault in study.FAULTS:
        methods = {
            spec["method"] for spec in specs if spec["scenario_id"] == f"yaw0_combined_{fault}"
        }
        assert methods == set(study.METHODS)
    cross = [spec for spec in specs if spec["scenario_id"] == "yaw15_combined_fresh"]
    assert len(cross) == 3
    assert {spec["fault"].kind for spec in cross} == {"fresh"}


def test_falling_friction_case_freezes_overlap_with_stop_hold_reverse():
    case = next(
        scenario["case"]
        for scenario in study.scenarios()
        if scenario["case_kind"] == "falling_friction_stop_reverse"
    )

    assert case.trajectory.name == "stop_hold_reverse"
    assert case.config.seed == 11
    assert case.scenario.wall_yaw_deg == 0
    assert case.scenario.wall_sliding_friction == 0.65
    assert case.scenario.tool_sliding_friction == 0.65
    assert [case.friction.at(time) for time in (0.0, 4.0, 4.5, 5.5, 6.0, 8.0)] == [
        0.65,
        0.65,
        0.5625,
        0.3875,
        0.30,
        0.30,
    ]
    assert [(phase.name, phase.start_s, phase.end_s) for phase in case.phases] == [
        ("high_forward", 1.5, 4.0),
        ("falling_forward", 4.0, 4.5),
        ("falling_stopping", 4.5, 5.5),
        ("falling_hold", 5.5, 6.0),
        ("low_hold", 6.0, 7.0),
        ("low_reverse_ramp", 7.0, 8.0),
        ("low_reverse", 8.0, 12.0),
    ]
    assert case.trajectory.rate_scale_at(5.75) == 0.0
    assert case.trajectory.rate_scale_at(8.0) == -1.0


def _provenance_fixture(fault_name):
    case = next(
        scenario["case"]
        for scenario in study.scenarios()
        if scenario["scenario_id"] == "yaw0_combined_fresh"
    )
    fault = study.runner.fault_config(fault_name)
    time = np.array([5.998, 6.0, 7.998, 8.0, 8.1, 8.3, 10.0])
    local_force = np.column_stack(
        (
            np.full(len(time), 12.0),
            np.arange(len(time), dtype=float) + 1.0,
            np.arange(len(time), dtype=float) - 2.0,
        )
    )
    rotation = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg).rotation
    measured = np.column_stack((local_force @ rotation.T, np.zeros((len(time), 3))))
    target_local = np.broadcast_to([0.0, 0.03, 0.04], (len(time), 3)).copy()
    target_world = target_local @ rotation.T
    stamps = time - 0.002
    expected_present = np.ones(len(time), dtype=bool)
    expected_force = local_force.copy()
    expected_stamp = stamps.copy()
    last_force = None
    last_stamp = None
    for index, sample_time in enumerate(time):
        active = fault.start_s <= sample_time < fault.end_s
        if fault.kind == "stale" and active:
            expected_force[index] = last_force
            expected_stamp[index] = last_stamp
            continue
        last_force, last_stamp = local_force[index].copy(), stamps[index]
        if fault.kind == "fresh" or not active:
            continue
        if fault.kind == "missing":
            expected_present[index] = False
            expected_force[index] = 0.0
            expected_stamp[index] = 0.0
        elif fault.kind == "scale":
            expected_force[index, 1:] *= fault.value
        elif fault.kind == "bias":
            expected_force[index, 1:] += fault.value * np.array([0.6, 0.8])
    trace = {
        "time": time,
        "measured_wrench_world": measured,
        "measured_wrench_sample_time": stamps,
        "target_linear_velocity": target_world,
        "raw_load_packet_present": expected_present,
        "raw_load_packet_force": expected_force,
        "raw_load_packet_stamp": expected_stamp,
    }
    return {"case": case, "fault": fault}, trace


@pytest.mark.parametrize("fault_name", study.FAULTS)
def test_fault_provenance_reconstructs_each_declared_transform(fault_name):
    spec, trace = _provenance_fixture(fault_name)

    result = study.validate_fault_provenance(spec, trace)

    assert result["fault_provenance_checked"] is True
    assert result["max_raw_packet_force_reconstruction_error_n"] <= 1e-12
    assert result["max_raw_packet_stamp_reconstruction_error_s"] <= 1e-12


@pytest.mark.parametrize("fault_name", study.FAULTS)
def test_fault_provenance_rejects_raw_packet_corruption(fault_name):
    spec, trace = _provenance_fixture(fault_name)
    trace["raw_load_packet_force"][-1, 1] += 1e-4

    with pytest.raises(ValueError, match="declared raw packet force"):
        study.validate_fault_provenance(spec, trace)


def _short_spec():
    spec = next(
        spec
        for spec in study.specifications()
        if spec["scenario_id"] == "yaw0_combined_fresh" and spec["method"] == "adaptive6_8"
    )
    short = replace(
        spec["case"],
        config=replace(spec["case"].config, duration=0.006),
        phases=(ProtocolPhase("short", 0.0, 0.006),),
        recovery_start_s=None,
    )
    return {**spec, "case": short}


def test_candidate_audit_accepts_a_full_short_trace_and_replays_state():
    spec = _short_spec()
    trace = study.runner.run_dynamic(
        spec["case"], study.GAIN_SCALE, spec["method"], spec["fault"]
    ).trace

    result = study.validate_candidate(spec, trace)

    assert set(trace) == study.FULL_TRACE_FIELDS
    assert result["validated_cycles"] == 3
    assert result["coefficient_transition_checked"] is True
    assert result["readiness_transition_checked"] is True


def test_candidate_audit_rejects_resealed_scheduler_state_corruption():
    spec = _short_spec()
    trace = study.runner.run_dynamic(
        spec["case"], study.GAIN_SCALE, spec["method"], spec["fault"]
    ).trace
    trace["load_budget_next_n"][1] += 1e-4

    with pytest.raises(ValueError, match="load-budget scheduler"):
        study.validate_candidate(spec, trace)


def test_candidate_audit_refuses_compact_substitution():
    spec = _short_spec()
    trace = study.runner.run_dynamic(
        spec["case"], study.GAIN_SCALE, spec["method"], spec["fault"]
    ).trace
    del trace["measured_position"]

    with pytest.raises(ValueError, match="full trace schema"):
        study.validate_candidate(spec, trace)


def test_pre_fault_prefix_is_bit_exact_and_rejects_early_changes():
    fresh = {
        "time": np.array([5.998, 6.0, 6.002]),
        "signal": np.array([1.0, 2.0, 3.0]),
        "metadata": np.array("fresh"),
    }
    faulted = deepcopy(fresh)
    faulted["metadata"] = np.array("bias_plus")
    faulted["signal"][1:] += 1.0

    result = study.unchanged_prefix(fresh, faulted, 6.0)

    assert result["bit_exact"] is True
    assert result["samples"] == 1
    faulted["signal"][0] += 1.0
    with pytest.raises(ValueError, match="pre-fault prefix differs: signal"):
        study.unchanged_prefix(fresh, faulted, 6.0)


def _metric_payload():
    overall = {name: 1.0 for name in study.OVERALL_DELTA_METRICS}
    overall.update(
        contact_ratio_pct=100.0,
        peak_force_n=10.0,
        saturation_pct=0.0,
        projection_pct=0.0,
        minimum_reserved_torque_headroom_nm=2.0,
        failed_absolute_checks="",
        absolute_checks_pass="yes",
    )
    phases = []
    for name in study.screening.PHASES:
        phase = {metric: 1.0 for metric in study.PHASE_DELTA_METRICS}
        phase.update(phase=name, failed_absolute_checks="", absolute_checks_pass="yes")
        phases.append(phase)
    windows = []
    for name in study.screening.WINDOWS:
        window = {metric: 1.0 for metric in study.WINDOW_DELTA_METRICS}
        window["window"] = name
        windows.append(window)
    return overall, phases, windows


def test_collect_executes_all_fixed_per_fault_controls_and_saves_full_trace(tmp_path, monkeypatch):
    (tmp_path / "traces").mkdir()
    calls = []
    saved = []
    fake = {
        "time": np.array([0.0]),
        "measured_position": np.zeros((1, 3)),
        "load_budget_applied_n": np.array([6.0]),
        "controller_coefficient_before_compute": np.array([0.45]),
        "diagnostic_corrected_force_n": np.array([12.0]),
    }
    metrics = _metric_payload()

    def run_dynamic(case, scale, method, fault):
        calls.append((method, fault.name))
        return SimpleNamespace(trace=deepcopy(fake))

    monkeypatch.setattr(study, "input_archive", lambda: tmp_path)
    monkeypatch.setattr(study.runner, "run_dynamic", run_dynamic)
    monkeypatch.setattr(
        study, "_save_trace", lambda path, trace: saved.append((path.name, set(trace)))
    )
    monkeypatch.setattr(study, "validate_candidate", lambda *args: {"validated_cycles": 1})
    monkeypatch.setattr(study.dynamic, "metrics", lambda *args: deepcopy(metrics))
    monkeypatch.setattr(study, "_recovery_metrics", lambda *args: [])

    result = study.collect(tmp_path, execute=True)

    assert len(calls) == len(saved) == len(result["runs"]) == 27
    assert len(result["comparisons"]) == 18
    assert all(fields == set(fake) for _, fields in saved)
    for fault in study.FAULTS:
        assert {(method, name) for method, name in calls if name == fault} == {
            (method, fault) for method in study.METHODS
        }


def test_comparison_retains_all_old_safety_failures_without_forcing_pass():
    metrics = _metric_payload()
    runs = []
    for scenario in study.scenarios():
        for method in study.METHODS:
            overall, phases, windows = deepcopy(metrics)
            if scenario["scenario_id"] == "yaw0_combined_missing" and method == "adaptive6_8":
                overall["peak_force_n"] = 36.0
                phases[0]["failed_absolute_checks"] = "contact_ratio;saturation"
            runs.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "method": method,
                    "overall": overall,
                    "phases": phases,
                    "diagnostics": windows,
                    "recovery": [],
                    "budget_exercised": method == "adaptive6_8",
                }
            )

    comparisons = study.compare_runs(runs)
    affected = next(
        row
        for row in comparisons
        if row["scenario_id"] == "yaw0_combined_missing" and row["reference_method"] == "fixed6"
    )

    assert affected["existing_preset_criteria_applicable"] is True
    assert affected["candidate_safety_failures"] == [
        {"scope": "overall", "phase": None, "checks": "peak"},
        {"scope": "phase", "phase": "pre_change", "checks": "contact_ratio;saturation"},
    ]


def test_common_cost_screen_accepts_the_actual_seven_falling_phase_identities():
    overall, _, windows = _metric_payload()
    case = study._falling_friction_case()
    phases = []
    for phase in case.phases:
        row = {metric: 1.0 for metric in study.PHASE_DELTA_METRICS}
        row.update(phase=phase.name, failed_absolute_checks="", absolute_checks_pass="yes")
        phases.append(row)
    candidate = {"overall": deepcopy(overall), "phases": deepcopy(phases), "diagnostics": windows}
    reference = {"overall": deepcopy(overall), "phases": deepcopy(phases), "diagnostics": windows}

    result = study._common_cost_screen(candidate, reference)

    assert result["status"] == "PASS"
    assert result["worst_phase_force_rmse_increase_n"] == 0.0
    assert result["worst_phase_orientation_rmse_increase_deg"] == 0.0


def _archive_fixture(root, monkeypatch):
    protocol = {"identity": study.IDENTITY, "frozen": True}
    sources = {"source.py": "a" * 64}
    comparison = {"safety": {"observed_status": "PASS"}, "runs": [1, 2, 3]}
    monkeypatch.setattr(study, "protocol_document", lambda: deepcopy(protocol))
    monkeypatch.setattr(study, "source_identity", lambda: deepcopy(sources))
    monkeypatch.setattr(study, "collect", lambda *args, **kwargs: deepcopy(comparison))
    for name in study.expected_artifacts():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "protocol.json":
            study._write_json(path, protocol)
        elif name == "source_hashes.json":
            study._write_json(path, sources)
        elif name == "comparison.json":
            study._write_json(path, comparison)
        else:
            path.write_bytes(b"synthetic full trace")
    artifacts = {name: study._sha256(root / name) for name in study.expected_artifacts()}
    study._write_json(
        root / "manifest.json",
        {
            "identity": study.IDENTITY,
            "reference": study.REFERENCE,
            "new_simulations": 27,
            "artifact_sha256": artifacts,
        },
    )
    (root / "COMPLETE").write_text(study._sha256(root / "manifest.json") + "\n")
    return comparison


def _reseal(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"] = {
        name: study._sha256(root / name) for name in manifest["artifact_sha256"]
    }
    study._write_json(manifest_path, manifest)
    (root / "COMPLETE").write_text(study._sha256(manifest_path) + "\n")


def test_audit_rejects_resealed_comparison_corruption(tmp_path, monkeypatch):
    comparison = _archive_fixture(tmp_path, monkeypatch)
    altered = deepcopy(comparison)
    altered["runs"][0] = 999
    study._write_json(tmp_path / "comparison.json", altered)
    _reseal(tmp_path)

    with pytest.raises(ValueError, match="recomputed robustness comparison"):
        study.audit_archive(tmp_path)


@pytest.mark.parametrize("occupied", ("directory", "symlink"))
def test_run_rejects_existing_or_symlink_output(tmp_path, monkeypatch, occupied):
    output = tmp_path / "published"
    if occupied == "directory":
        output.mkdir()
    else:
        target = tmp_path / "target"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(study, "collect", lambda *args, **kwargs: pytest.fail("must fail early"))

    with pytest.raises(ValueError, match="must not exist|symlink"):
        study.run(output)
