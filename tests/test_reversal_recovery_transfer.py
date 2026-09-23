from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from compliant_control_lab.surface_simulation import yaw_frame
from tools.reversal_recovery import study as original
from tools.reversal_recovery_transfer import study


def test_matrix_has_72_runs_and_reuses_only_the_eight_original_traces():
    protocol = study.protocol_document()
    specs = study.specifications(protocol)
    assert len(specs) == 72
    assert len({study.run_key(spec) for spec in specs}) == 72
    reused = [spec for spec in specs if spec["origin"] == "reference"]
    assert len(reused) == 8
    assert {(spec["surface_yaw_deg"], spec["error_profile"]) for spec in reused} == {(0, "clean")}
    assert len({spec["trace_path"] for spec in specs if spec["origin"] == "new"}) == 64
    for spec in specs:
        case = spec["case"]
        yaw, profile = spec["surface_yaw_deg"], spec["error_profile"]
        assert case.task.yaw_deg == case.scenario.wall_yaw_deg == yaw
        assert case.config.seed == spec["seed"]
        assert case.config.duration == 12.0 and case.config.timestep == 0.002
        assert case.trajectory.name == "stop_hold_reverse"
        assert case.controller_yaw_error_deg == (0.0 if profile == "clean" else 3.0)
        np.testing.assert_array_equal(case.wrench_bias_world.at(5.999), np.zeros(6))
        expected = 0.0 if profile == "clean" else 0.75
        np.testing.assert_allclose(case.wrench_bias_world.at(6.0)[:3],
                                   expected * yaw_frame(yaw).rotation[:, 0], atol=1e-15)
        assert spec["fault"].name == ("scale_0p8" if profile == "combined_scale_0p8" else "fresh")
        controller, parameters = original.make_controller(spec, protocol)
        assert parameters["velocity_error_time"] == 0.05
        assert parameters["minimum_force"] == 6.0 and parameters["max_force"] == 8.0
        assert type(controller._base.tangential).__name__ == (
            "HoldCapTracking" if spec["method"] == "hold_cap_tracking" else "LoadAwareCompensation"
        )


@pytest.mark.parametrize("field,value", [
    ("methods", ["fixed6", "hold_cap_tracking"]), ("seeds", [11, 30]),
    ("scenario_names", ["falling", "invented"]), ("duration_s", 13),
    ("timestep_s", 0.004), ("rotation_gain_scale", 2), ("velocity_error_time_s", 0.10),
    ("reused_runs", 9), ("paired_comparisons", 35), ("physical_surface_load_combinations", 7),
])
def test_protocol_cannot_claim_unexecuted_configuration(field, value):
    protocol = study.protocol_document()
    protocol[field] = value
    with pytest.raises(ValueError):
        study.specifications(protocol)


@pytest.mark.parametrize("corruption", ["reuse_yaw", "reuse_error", "fault", "bias_time", "prefix", "gate"])
def test_protocol_checks_reuse_identity_fault_onset_prefix_and_original_gate(corruption):
    protocol = study.protocol_document()
    if corruption == "reuse_yaw":
        protocol["reference"]["reuse_surface_yaw_deg"] = 15
    elif corruption == "reuse_error":
        protocol["reference"]["reuse_error_profile"] = "combined"
    elif corruption == "fault":
        protocol["auxiliary_scale_fault"]["value"] = 1.2
    elif corruption == "bias_time":
        protocol["normal_bias_start_s"] = 5.0
    elif corruption == "prefix":
        protocol["prefix_checks"][1]["before_s"] = 5.0
    else:
        protocol["acceptance"]["maximum_window_tangent_velocity_rmse_increase_mm_s"] = 0.6
    with pytest.raises(ValueError):
        study.specifications(protocol)


@pytest.fixture
def reference_trace():
    spec = next(spec for spec in study.specifications(study.protocol_document())
                if spec["origin"] == "reference" and spec["scenario"] == "falling"
                and spec["seed"] == 11 and spec["method"] == "adaptive6_8")
    path = study.ROOT / study.protocol_document()["reference"]["directory"] / spec["trace_path"]
    return spec, original.previous._load_trace(path)


def test_original_full_trace_passes_new_schema_schedule_packet_and_state_audits(reference_trace):
    spec, trace = reference_trace
    protocol = study.protocol_document()
    _, parameters = original.make_controller(spec, protocol)
    checked = study.validate_trace(spec, trace, parameters, protocol)
    assert checked["validated_cycles"] == 6000
    assert checked["fault_provenance_checked"] is True
    assert checked["hold_release_cycles"] == 0


@pytest.mark.parametrize("field,value", [
    ("fault_kind", "scale"), ("fault_value", 0.8), ("method", "hold_cap_tracking"),
    ("trace_schema_version", 2), ("load_max_measurement_age_s", 0.2),
    ("rotation_gain_scale", 2.0),
])
def test_trace_metadata_cannot_be_relabelled(reference_trace, field, value):
    spec, trace = reference_trace
    trace[field] = np.array(value)
    protocol = study.protocol_document()
    _, parameters = original.make_controller(spec, protocol)
    with pytest.raises(ValueError, match="metadata"):
        study.validate_trace(spec, trace, parameters, protocol)


def test_fault_provenance_rejects_unchanged_packet_labelled_as_scale(reference_trace):
    spec, trace = reference_trace
    spec = {**spec, "fault": original.runner.FAULTS["scale_0p8"]}
    for field in ("name", "kind", "start_s", "end_s", "value"):
        trace[f"fault_{field}"] = np.array(getattr(spec["fault"], field))
    protocol = study.protocol_document()
    _, parameters = original.make_controller(spec, protocol)
    with pytest.raises(ValueError, match="raw packet force"):
        study.validate_trace(spec, trace, parameters, protocol)


@pytest.mark.parametrize("field", ["controller_frame_rotation", "applied_raw_wrench_bias_world"])
def test_full_trace_rejects_wrong_coordinate_or_bias(reference_trace, field):
    spec, trace = reference_trace
    trace[field].flat[0] += 0.001
    protocol = study.protocol_document()
    _, parameters = original.make_controller(spec, protocol)
    with pytest.raises(ValueError, match="mismatch"):
        study.validate_trace(spec, trace, parameters, protocol)


def test_window_errors_use_real_surface_frame_not_controller_error():
    protocol = study.protocol_document()
    spec = next(spec for spec in study.specifications(protocol)
                if spec["surface_yaw_deg"] == 15 and spec["error_profile"] == "combined")
    time = np.arange(6000) * 0.002
    rotation = yaw_frame(15).rotation
    trace = {"time": time, "position": np.tile(np.array([0.2, 0.003, 0.004]) @ rotation.T, (6000, 1)),
             "linear_velocity": np.tile(np.array([0.4, 0.006, 0.008]) @ rotation.T, (6000, 1)),
             "target_position": np.zeros((6000, 3)), "target_linear_velocity": np.zeros((6000, 3))}
    windows = study.window_metrics(trace, spec["case"], protocol)
    assert all(row["tangent_rmse_mm"] == pytest.approx(5.0) for row in windows)
    assert all(row["tangent_velocity_rmse_mm_s"] == pytest.approx(10.0) for row in windows)


def passing_rows(protocol):
    rows = []
    for spec in study.specifications(protocol):
        metrics = {"force_rmse_n": 1.0, "orientation_rmse_deg": 0.1,
                   "contact_ratio_pct": 100.0, "peak_force_n": 20.0, "saturation_pct": 0.0}
        windows = []
        for window in protocol["windows"]:
            improved = (spec["scenario"] == "falling" and spec["method"] == "hold_cap_tracking"
                        and window["name"] in {"early", "post"})
            windows.append({**window, "tangent_rmse_mm": 1.7 if improved else 2.0,
                            "tangent_velocity_rmse_mm_s": 10.0})
        rows.append({key: spec[key] for key in study.RUN_KEYS} | {
            "overall": {**metrics, "projection_pct": 0.0, "minimum_reserved_torque_headroom_nm": 2.0},
            "phases": [{"phase": phase.name, **metrics} for phase in spec["case"].phases],
            "windows": windows,
        })
    return rows


def test_all_36_pairs_use_original_gates_without_averaging_away_one_failure():
    protocol = study.protocol_document()
    rows = passing_rows(protocol)
    assert len(study.compare_runs(rows, protocol)) == 36
    assert all(row["status"] == "PASS" for row in study.compare_runs(rows, protocol))
    candidate = next(row for row in rows if row["method"] == "hold_cap_tracking")
    candidate["windows"][1]["tangent_rmse_mm"] = 1.95
    compared = study.compare_runs(rows, protocol)
    assert sum(row["status"] == "FAIL" for row in compared) == 1
    failed = next(row for row in compared if row["status"] == "FAIL")
    assert failed["missed_benefit_checks"] == ["early:position_reduction"]
    assert failed["absolute_failures"] == failed["paired_cost_failures"] == []
    candidate["overall"]["projection_pct"] = 0.1
    failed = next(row for row in study.compare_runs(rows, protocol) if row["status"] == "FAIL")
    assert failed["absolute_failures"] == ["hold_cap_tracking:overall:projection_pct"]


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unknown_yaw", "unknown_error", "nan", "empty_windows"])
def test_comparison_rejects_incomplete_or_misidentified_pairs(corruption):
    protocol = study.protocol_document()
    rows = passing_rows(protocol)
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif corruption == "unknown_yaw":
        rows[0]["surface_yaw_deg"] = 30
    elif corruption == "unknown_error":
        rows[0]["error_profile"] = "unregistered"
    elif corruption == "nan":
        rows[0]["windows"][0]["tangent_rmse_mm"] = float("nan")
    else:
        rows[0]["windows"] = []
    with pytest.raises(ValueError):
        study.compare_runs(rows, protocol)


def test_prefix_check_reports_real_cycle_fields_and_retains_failure(reference_trace):
    _, trace = reference_trace
    altered = {name: value.copy() for name, value in trace.items()}
    report = study.prefix_check(trace, altered, 4.5)
    assert report["bit_exact"] is True and report["samples"] == 2250
    assert "measured_position" in report["fields"]
    altered["measured_position"][2249, 0] += 1e-8
    report = study.prefix_check(trace, altered, 4.5)
    assert report["bit_exact"] is False
    assert report["different_fields"] == ["measured_position"]


def test_prefix_check_rejects_signed_zero_difference():
    left = {"time": np.array([0.0, 0.002]), "force": np.array([0.0, 1.0])}
    right = {"time": left["time"].copy(), "force": np.array([-0.0, 1.0])}
    assert np.array_equal(left["force"], right["force"])
    report = study.prefix_check(left, right, 0.002)
    assert report["bit_exact"] is False
    assert report["different_fields"] == ["force"]


def test_new_source_identity_preserves_all_existing_source_hashes():
    before, after = original.source_identity(), study.source_identity()
    assert all(after[path] == digest for path, digest in before.items())
    assert "tools/reversal_recovery_transfer/study.py" in after
    assert "tools/reversal_recovery_transfer/protocol.json" in after


def test_cli_requires_explicit_operation_and_never_starts_physics(tmp_path, monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "invalid-inherited-backend")
    result = subprocess.run([sys.executable, "-m", "tools.reversal_recovery_transfer.study"],
                            cwd=study.ROOT, env={**os.environ, "MUJOCO_GL": "disable"},
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 2
    assert "required" in result.stderr
    assert not list(tmp_path.iterdir())


def test_original_comparison_json_values_are_not_relaxed():
    with pytest.raises(ValueError):
        original.verify_comparison({"default_changed": False}, {"default_changed": 0})
    original.verify_comparison({"case": [1, 2]}, {"case": (1, 2)})


def test_comparison_replay_accepts_observed_cross_blas_roundoff():
    # Same archived yaw15/combined trace, NumPy 2.2.6: Haswell versus Prescott.
    saved = {"runs": [{"audit": {"max_scheduler_reconstruction_error_n": 1.7763568394002505e-15}}]}
    rebuilt = {"runs": [{"audit": {"max_scheduler_reconstruction_error_n": 0.0}}]}
    study.verify_comparison(saved, rebuilt, study.protocol_document())


def _comparison_payload():
    return {"runs": [{
        "case": {"yaw_deg": 0.0}, "parameters": {"velocity_error_time": 0.05},
        "trace_sha256": "a" * 64,
        "overall": {"tangent_rmse_mm": 1.0, "sample_count": 6000,
                    "slew_reconstruction_mismatch_cycles": 0.0, "force_recovered": True},
        "phases": [{"phase": "hold", "force_rmse_n": 1.0}],
        "windows": [{"name": "ramp", "start_s": 7.0, "end_s": 8.0, "tangent_rmse_mm": 1.0}],
        "audit": {"max_scheduler_reconstruction_error_n": 0.0, "validated_cycles": 6000},
    }], "comparisons": [{"status": "PASS", "failed_checks": [],
                          "window_deltas": [{"name": "ramp", "tangent_rmse_mm": 1.0,
                                             "tangent_reduction_pct": None}]}],
        "prefix_checks": [{"bit_exact": True, "samples": 2250}], "default_changed": False}


def _nested(container, path):
    for key in path:
        container = container[key]
    return container


CONTINUOUS_PATHS = [
    ("runs", 0, "overall", "tangent_rmse_mm"),
    ("runs", 0, "phases", 0, "force_rmse_n"),
    ("runs", 0, "windows", 0, "tangent_rmse_mm"),
    ("comparisons", 0, "window_deltas", 0, "tangent_rmse_mm"),
    ("runs", 0, "audit", "max_scheduler_reconstruction_error_n"),
]


@pytest.mark.parametrize("path", CONTINUOUS_PATHS)
def test_named_continuous_report_fields_accept_one_ulp_and_declared_absolute_roundoff(path):
    saved = _comparison_payload()
    rebuilt = copy.deepcopy(saved)
    value = _nested(saved, path)
    _nested(rebuilt, path[:-1])[path[-1]] = float(np.nextafter(value, float("inf")))
    study.verify_comparison(saved, rebuilt, study.protocol_document())
    tolerance = 1e-12 if path[2] == "audit" else 1e-10
    _nested(rebuilt, path[:-1])[path[-1]] = value + tolerance / 2
    study.verify_comparison(saved, rebuilt, study.protocol_document())
    _nested(rebuilt, path[:-1])[path[-1]] = value + 2 * tolerance
    with pytest.raises(ValueError, match="differs"):
        study.verify_comparison(saved, rebuilt, study.protocol_document())


@pytest.mark.parametrize("value", [1, True, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("path", CONTINUOUS_PATHS)
def test_numeric_report_tolerance_never_accepts_type_changes_or_nonfinite_values(path, value):
    saved, rebuilt = _comparison_payload(), _comparison_payload()
    _nested(rebuilt, path[:-1])[path[-1]] = value
    with pytest.raises(ValueError):
        study.verify_comparison(saved, rebuilt, study.protocol_document())
    _nested(saved, path[:-1])[path[-1]] = value
    if isinstance(value, float):
        with pytest.raises(ValueError, match="nonfinite"):
            study.verify_comparison(saved, rebuilt, study.protocol_document())


@pytest.mark.parametrize("path,value", [
    (("runs", 0, "case", "yaw_deg"), 1e-15),
    (("runs", 0, "parameters", "velocity_error_time"), 0.05 + 1e-15),
    (("runs", 0, "windows", 0, "start_s"), 7.0 + 1e-15),
    (("runs", 0, "overall", "slew_reconstruction_mismatch_cycles"), 1e-15),
    (("runs", 0, "overall", "sample_count"), 6000.0),
    (("runs", 0, "audit", "validated_cycles"), 6001),
    (("runs", 0, "overall", "force_recovered"), 1),
    (("comparisons", 0, "status"), "FAIL"),
    (("comparisons", 0, "failed_checks"), ["ramp:velocity_cost"]),
    (("comparisons", 0, "window_deltas", 0, "tangent_reduction_pct"), 0.0),
    (("prefix_checks", 0, "bit_exact"), False),
    (("runs", 0, "trace_sha256"), "b" * 64),
    (("default_changed",), 0),
])
def test_configuration_counts_flags_status_failures_prefixes_and_hashes_stay_exact(path, value):
    saved, rebuilt = _comparison_payload(), _comparison_payload()
    _nested(rebuilt, path[:-1])[path[-1]] = value
    with pytest.raises(ValueError):
        study.verify_comparison(saved, rebuilt, study.protocol_document())


@pytest.mark.parametrize("corruption", ["missing_key", "new_key", "short_list", "order"])
def test_comparison_replay_preserves_structure_and_order(corruption):
    saved = _comparison_payload()
    saved["ordered"] = ["first", "second"]
    rebuilt = copy.deepcopy(saved)
    if corruption == "missing_key":
        rebuilt.pop("default_changed")
    elif corruption == "new_key":
        rebuilt["extra"] = None
    elif corruption == "short_list":
        rebuilt["runs"] = []
    else:
        rebuilt["ordered"].reverse()
    with pytest.raises(ValueError):
        study.verify_comparison(saved, rebuilt, study.protocol_document())


def test_comparison_replay_has_no_relative_tolerance():
    saved, rebuilt = _comparison_payload(), _comparison_payload()
    saved["runs"][0]["overall"]["tangent_rmse_mm"] = 1e9
    rebuilt["runs"][0]["overall"]["tangent_rmse_mm"] = 1e9 + 1e-6
    with pytest.raises(ValueError):
        study.verify_comparison(saved, rebuilt, study.protocol_document())
    protocol = study.protocol_document()
    protocol["comparison_replay"]["relative_tolerance"] = 1e-12
    with pytest.raises(ValueError, match="contract"):
        study.verify_comparison(saved, rebuilt, protocol)


def _metadata_only_archive(directory, **overrides):
    """Invalid-artifact fixture: metadata tests must reject before reading traces."""
    protocol = study.protocol_document()
    artifacts = {"protocol.json": json.dumps(protocol),
                 "source_hashes.json": json.dumps(study.source_identity()), "comparison.json": "{}"}
    artifacts.update({spec["trace_path"]: "metadata-only test fixture, not a simulated trace"
                      for spec in study.specifications(protocol) if spec["origin"] == "new"})
    for name, content in artifacts.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    manifest = {
        "identity": protocol["identity"], "new_simulations": 64, "evaluated_runs": 72,
        "source_commit": "a" * 40, "default_changed": False, "new_holdout": False,
        "public_development": True,
        "artifact_sha256": {name: study._sha256(directory / name) for name in artifacts},
        **overrides,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "COMPLETE").write_text(study._sha256(directory / "manifest.json") + "\n")


@pytest.mark.parametrize("field,value", [
    ("identity", "other-study"), ("new_simulations", 72), ("evaluated_runs", 64),
    ("default_changed", True), ("default_changed", 0), ("new_holdout", True),
    ("new_holdout", 0), ("public_development", False), ("public_development", 1),
    ("source_commit", None), ("source_commit", "a" * 39), ("source_commit", "z" * 40),
])
def test_resealed_archive_rejects_material_scope_or_execution_identity(tmp_path, monkeypatch, field, value):
    _metadata_only_archive(tmp_path, **{field: value})

    def must_not_replay(*args, **kwargs):
        pytest.fail("invalid archive metadata reached trace replay")

    monkeypatch.setattr(study, "collect", must_not_replay)
    with pytest.raises(ValueError):
        study.audit_archive(tmp_path)


def test_resealed_source_identity_change_is_rejected_before_trace_replay(tmp_path, monkeypatch):
    _metadata_only_archive(tmp_path)
    source_file = tmp_path / "source_hashes.json"
    sources = json.loads(source_file.read_text())
    sources["tools/reversal_recovery/controller.py"] = "0" * 64
    source_file.write_text(json.dumps(sources))
    manifest_file = tmp_path / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["artifact_sha256"]["source_hashes.json"] = study._sha256(source_file)
    manifest_file.write_text(json.dumps(manifest))
    (tmp_path / "COMPLETE").write_text(study._sha256(manifest_file) + "\n")
    monkeypatch.setattr(study, "collect", lambda *args, **kwargs: pytest.fail("unexpected replay"))
    with pytest.raises(ValueError, match="source identity"):
        study.audit_archive(tmp_path)


def test_output_must_not_overwrite_existing_results(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="new"):
        study.run(tmp_path)
    assert marker.read_text() == "keep"


def test_finalization_rejects_symlink_staging_before_audit(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(study, "audit_archive", lambda *_: pytest.fail("unexpected audit"))
    with pytest.raises(ValueError, match="symlink"):
        study.finalize(linked, tmp_path / "published")


def test_complete_failed_experiment_can_be_finalized_without_rerunning_physics(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    staging.mkdir()
    original_bytes = b"preserve FAIL evidence exactly\n"
    (staging / "evidence.txt").write_bytes(original_bytes)
    calls = []

    def verified_failure(directory):
        calls.append(directory)
        return {"archive_integrity": "PASS", "comparison_status": "FAIL"}

    monkeypatch.setattr(study, "audit_archive", verified_failure)
    monkeypatch.setattr(study.original.runner, "_run_loop", lambda *_: pytest.fail("unexpected physics"))
    destination = tmp_path / "published"
    result = study.finalize(staging, destination)
    assert result["comparison_status"] == "FAIL"
    assert calls == [staging]
    assert not staging.exists()
    assert (destination / "evidence.txt").read_bytes() == original_bytes


def test_execution_commit_is_captured_before_collection_not_at_publication(tmp_path, monkeypatch):
    events = []
    real_check_output = study.subprocess.check_output

    def head(*args, **kwargs):
        if args[0] != ["git", "rev-parse", "HEAD"]:
            return real_check_output(*args, **kwargs)
        events.append("HEAD")
        return "a" * 40 + "\n"

    def collect(directory, protocol, *, execute):
        assert execute is True and events == ["HEAD"]
        events.append("collect")
        return {"status": "FAIL"}

    def finalize(staging, destination):
        assert events == ["HEAD", "collect"]
        manifest = json.loads((staging / "manifest.json").read_text())
        assert manifest["source_commit"] == "a" * 40
        assert manifest["source_commit_role"].startswith("execution-start HEAD")
        return {"comparison_status": "FAIL"}

    monkeypatch.setattr(study.subprocess, "check_output", head)
    monkeypatch.setattr(study, "collect", collect)
    monkeypatch.setattr(study, "finalize", finalize)
    assert study.run(tmp_path / "new_archive")["comparison_status"] == "FAIL"
