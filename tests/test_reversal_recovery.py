from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.franka_control import FrankaState, FrankaTarget
from tools.reversal_recovery import study
from tools.reversal_recovery.controller import HoldCapTracking

NORMAL = np.array([1.0, 0.0, 0.0])
DT = 0.002


def _sample(speed=0.0):
    velocity = np.array([0.0, speed, 0.0])
    state = FrankaState(np.zeros(3), np.eye(3), velocity, np.zeros(3), 12.0)
    target = FrankaTarget(np.zeros(3), np.eye(3), velocity, np.zeros(3), 12.0)
    return state, target


def _cycle(compensation, *, speed=0.0, measurement=True, accepted=True, normal_force=12.0):
    state, target = _sample(speed)
    if measurement:
        compensation.set_force_measurement(np.array([normal_force, 7.8, 0.0]))
    force = compensation.force(state, target, NORMAL, normal_force, 1.0, True, dt=DT)
    compensation.advance(state, target, NORMAL, DT, accepted)
    return force


def test_hold_releases_only_next_cycle_coefficient_at_the_existing_rate():
    compensation = HoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    force = _cycle(compensation)
    assert compensation.equivalent_mu == pytest.approx(0.6194, abs=1e-14)
    np.testing.assert_array_equal(force, np.zeros(3))
    np.testing.assert_array_equal(compensation.last_force, force)


@pytest.mark.parametrize("kwargs", [
    {"measurement": False}, {"accepted": False}, {"speed": 0.005}, {"speed": 0.05},
    {"normal_force": 1.0},
])
def test_missing_rejected_moving_or_inactive_cycle_does_not_release(kwargs):
    compensation = HoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    _cycle(compensation, **kwargs)
    assert compensation.equivalent_mu == 0.62


def test_hold_never_increases_a_coefficient_below_the_current_cap():
    compensation = HoldCapTracking("online", nominal_mu=0.45, max_force=8.0)
    for _ in range(500):
        _cycle(compensation)
    assert compensation.equivalent_mu == 0.45


def test_hold_stops_exactly_at_the_cap_without_resetting_to_nominal():
    compensation = HoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    previous = compensation.equivalent_mu
    for _ in range(500):
        _cycle(compensation)
        assert 0.0 <= previous - compensation.equivalent_mu <= 0.0006 + 1e-14
        previous = compensation.equivalent_mu
    assert compensation.equivalent_mu == 0.5
    assert compensation.equivalent_mu != compensation.nominal_mu


def test_high_load_budget_retains_a_coefficient_below_its_own_cap():
    compensation = HoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    for _ in range(1500):
        _cycle(compensation, speed=0.05)
    assert compensation.next_budget_n == 8.0
    for _ in range(50):
        _cycle(compensation)
    assert compensation.equivalent_mu == 0.62


def test_reset_clears_the_new_hold_state_using_existing_reset_semantics():
    compensation = HoldCapTracking("online", nominal_mu=0.62, max_force=8.0)
    _cycle(compensation)
    compensation.reset()
    assert compensation.equivalent_mu == 0.62
    assert compensation.applied_budget_n == compensation.minimum_force
    assert compensation.load_estimate_n == 0.0
    assert not compensation.update_ready
    np.testing.assert_array_equal(compensation.last_force, np.zeros(3))


def test_protocol_builds_only_four_paired_cases_without_changing_seed_or_schedule():
    protocol = study.protocol_document()
    specs = study.specifications(protocol)
    assert len(specs) == 8
    assert {(row["scenario"], row["seed"]) for row in specs} == {
        ("falling", 11), ("falling", 29), ("constant_high", 11), ("constant_high", 29),
    }
    for spec in specs:
        case = spec["case"]
        assert case.config.seed == spec["seed"]
        assert case.scenario.name.endswith(f"seed_{spec['seed']}")
        assert case.friction.at(8.0) == (0.3 if spec["scenario"] == "falling" else 0.65)
        if spec["scenario"] == "constant_high":
            assert all("low" not in phase.name and "falling" not in phase.name for phase in case.phases)
        controller, parameters = study.make_controller(spec, protocol)
        assert parameters["velocity_error_time"] == 0.05
        assert parameters["force_slew_rate"] == 20.0
        assert parameters["coefficient_rate_limit"] == 0.3
        assert isinstance(controller._base.tangential, HoldCapTracking) == (
            spec["method"] == "hold_cap_tracking"
        )


@pytest.fixture
def old_falling_trace():
    path = study.ROOT / (
        "results/franka_measured_budget_robustness/traces/"
        "yaw0_falling_friction_stop_reverse_fresh__adaptive6_8.npz"
    )
    with np.load(path, allow_pickle=False) as saved:
        trace = {name: saved[name] for name in saved.files}
    # Only the metadata schema label changes in this in-memory audit fixture.
    trace["trace_schema_version"] = np.array(study.SCHEMA)
    return trace


def test_independent_replay_agrees_with_the_frozen_parent_rule(old_falling_trace):
    protocol = study.protocol_document()
    spec = study.specifications(protocol)[0]
    _, parameters = study.make_controller(spec, protocol)
    report = study.validate_trace(spec, old_falling_trace, parameters, protocol)
    assert report["validated_cycles"] == 6000
    assert report["hold_release_cycles"] == 0
    assert report["max_coefficient_transition_error"] < 1e-12
    overall, phases, _ = study.dynamic.metrics(old_falling_trace, spec["case"], 8.0)
    assert overall["peak_force_n"] < 35.0
    assert len(phases) == 7


def test_independent_hold_replay_matches_three_hand_calculated_steps(old_falling_trace):
    trace = {name: value[2750:2753].copy() if value.ndim and value.shape[0] == 6000 else value.copy()
             for name, value in old_falling_trace.items()}
    time = np.arange(3) * DT
    force = np.tile([12.0, 7.8, 0.0], (3, 1))
    trace.update(time=time, raw_load_packet_force=force.copy(), load_force_local=force.copy(),
                 raw_load_packet_stamp=time.copy(), load_measurement_time_s=time.copy(),
                 controller_coefficient_before_compute=np.array([0.62, 0.6194, 0.6188]),
                 controller_coefficient_after_compute=np.array([0.6194, 0.6188, 0.6182]))
    for name in ("target_linear_velocity", "measured_linear_velocity", "target_position",
                 "measured_position", "load_compensation_force_local", "requested_tangential_force_world"):
        trace[name] = np.zeros((3, 3))
    for name in ("raw_load_packet_present", "load_measurement_available", "measured_in_contact",
                 "diagnostic_compensation_active", "load_projection_accepted", "diagnostic_amplitude_capped"):
        trace[name] = np.ones(3, dtype=bool)
    for name in ("controller_update_ready_after_compute", "controller_update_ready_before_compute",
                 "diagnostic_slew_limited", "load_budget_updated"):
        trace[name] = np.zeros(3, dtype=bool)
    for name in ("load_estimate_n", "load_projected_n", "load_measurement_age_s"):
        trace[name] = np.zeros(3)
    for name in ("diagnostic_corrected_force_n", "target_normal_force"):
        trace[name] = np.full(3, 12.0)
    for name in ("load_budget_applied_n", "load_budget_next_n"):
        trace[name] = np.full(3, 6.0)
    trace["contact_blend"] = trace["torque_projection_scale"] = np.ones(3)
    trace["load_packet_status"] = np.ones(3, dtype=np.uint8)
    protocol = study.protocol_document()
    _, parameters = study.make_controller(study.specifications(protocol)[1], protocol)
    parameters["nominal_mu"] = 0.62
    report = study.replay_compensation(trace, parameters, "hold_cap_tracking", DT)
    assert report["hold_release_cycles"] == 3
    assert report["max_coefficient_transition_error"] < 1e-12


@pytest.mark.parametrize("field,index,change", [
    ("controller_coefficient_after_compute", 3000, 1e-4),
    ("load_compensation_force_local", (3000, 1), 0.001),
    ("load_budget_applied_n", 3000, 0.1),
    ("time", 3000, 0.0001),
])
def test_independent_replay_rejects_changed_states(old_falling_trace, field, index, change):
    protocol = study.protocol_document()
    spec = study.specifications(protocol)[0]
    _, parameters = study.make_controller(spec, protocol)
    old_falling_trace[field][index] += change
    with pytest.raises(ValueError, match="mismatch"):
        study.validate_trace(spec, old_falling_trace, parameters, protocol)


def test_readonly_audit_does_not_label_old_hold_dynamics_as_the_new_rule(old_falling_trace):
    protocol = study.protocol_document()
    spec = study.specifications(protocol)[1]
    _, parameters = study.make_controller(spec, protocol)
    old_falling_trace["method"] = np.array("hold_cap_tracking")
    with pytest.raises(ValueError, match="coefficient"):
        study.validate_trace(spec, old_falling_trace, parameters, protocol)


def test_replay_rejects_a_suppressed_projection_gate_during_a_hold(old_falling_trace):
    protocol = study.protocol_document()
    spec = study.specifications(protocol)[0]
    _, parameters = study.make_controller(spec, protocol)
    assert old_falling_trace["torque_projection_scale"][3000] == 1.0
    old_falling_trace["load_projection_accepted"][3000] = False
    with pytest.raises(ValueError, match="projection acceptance"):
        study.validate_trace(spec, old_falling_trace, parameters, protocol)


def test_consistent_scaled_gate_is_auditable_but_projection_cost_fails_acceptance(old_falling_trace):
    protocol = study.protocol_document()
    spec = study.specifications(protocol)[0]
    _, parameters = study.make_controller(spec, protocol)
    # A local compensation-only fixture; no claim that the edited command replays dynamics.
    old_falling_trace["torque_projection_scale"][3000] = 0.5
    old_falling_trace["load_projection_accepted"][3000] = False
    audit = study.replay_compensation(old_falling_trace, parameters, "adaptive6_8", DT)
    assert audit["projection_scaled_or_fallback_cycles"] == 1
    rows = _passing_rows(protocol)
    rows[1]["overall"]["projection_pct"] = 100.0 / 6000
    comparison = study.compare_runs(rows, protocol)[0]
    assert comparison["status"] == "FAIL"
    assert "hold_cap_tracking:overall:projection_pct" in comparison["failed_checks"]


def _passing_rows(protocol):
    rows = []
    for spec in study.specifications(protocol):
        metrics = {"force_rmse_n": 1.0, "orientation_rmse_deg": 0.1,
                   "contact_ratio_pct": 100.0, "peak_force_n": 20.0, "saturation_pct": 0.0}
        windows = []
        for window in protocol["windows"]:
            improved = (spec["method"] == "hold_cap_tracking" and spec["scenario"] == "falling"
                        and window["name"] in {"early", "post"})
            windows.append({**window, "tangent_rmse_mm": 1.7 if improved else 2.0,
                            "tangent_velocity_rmse_mm_s": 10.0})
        rows.append({key: spec[key] for key in ("scenario", "seed", "method")} | {
            "overall": {**metrics, "projection_pct": 0.0, "minimum_reserved_torque_headroom_nm": 2.0},
            "phases": [{"phase": phase.name, **metrics} for phase in spec["case"].phases],
            "windows": windows,
        })
    return rows


def test_comparison_requires_each_seed_and_all_windows():
    protocol = study.protocol_document()
    rows = _passing_rows(protocol)
    assert all(row["status"] == "PASS" for row in study.compare_runs(rows, protocol))
    rows[1]["windows"][1]["tangent_rmse_mm"] = 1.95
    comparisons = study.compare_runs(rows, protocol)
    assert comparisons[0]["status"] == "FAIL"
    assert "early:position_reduction" in comparisons[0]["failed_checks"]


@pytest.mark.parametrize("corruption", [
    "missing_window", "duplicate_window", "nan", "phase_nan", "duplicate_phase", "missing_phase",
])
def test_comparison_rejects_missing_or_nonfinite_metrics(corruption):
    protocol = study.protocol_document()
    rows = _passing_rows(protocol)
    if corruption == "missing_window":
        rows[1]["windows"].pop()
    elif corruption == "duplicate_window":
        rows[1]["windows"][1] = rows[1]["windows"][0]
    elif corruption == "nan":
        rows[1]["windows"][0]["tangent_rmse_mm"] = float("nan")
    elif corruption == "phase_nan":
        rows[1]["phases"][0]["tangent_rmse_mm"] = float("nan")
    elif corruption == "duplicate_phase":
        rows[1]["phases"][1] = rows[1]["phases"][0]
    else:
        rows[1]["phases"].pop()
    with pytest.raises(ValueError):
        study.compare_runs(rows, protocol)


def test_archive_audit_rejects_modified_summary_before_replaying(tmp_path):
    summary = tmp_path / "comparison.json"
    summary.write_text('{"status":"FAIL"}')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifact_sha256": {"comparison.json": study._sha256(summary)}}))
    (tmp_path / "COMPLETE").write_text(study._sha256(manifest))
    summary.write_text('{"status":"PASS"}')
    with pytest.raises(ValueError, match="artifact hash differs: comparison.json"):
        study.audit_archive(tmp_path)


def _sealed_manifest_with_change(directory, key, value):
    """Hash-consistent inventory; metadata must be rejected before loading its dummy traces."""
    protocol = study.protocol_document()
    payloads = {"protocol.json": protocol, "source_hashes.json": study.source_identity(),
                "comparison.json": {}}
    for relative, payload in payloads.items():
        (directory / relative).write_text(json.dumps(payload))
    for spec in study.specifications(protocol):
        path = directory / spec["trace_path"]
        path.parent.mkdir(exist_ok=True)
        path.write_text("trace parsing must not be reached in this metadata rejection test")
    files = {path.relative_to(directory).as_posix(): study._sha256(path)
             for path in directory.rglob("*") if path.is_file()}
    manifest = {"identity": protocol["identity"], "new_simulations": 8,
                "artifact_sha256": files, "default_changed": False, "new_holdout": False,
                "source_commit": "a" * 40}
    manifest[key] = value
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    (directory / "COMPLETE").write_text(study._sha256(path))


@pytest.mark.parametrize("commit", [None, 123, "main", "g" * 40, "a" * 39, ["a" * 40]])
def test_resealed_manifest_rejects_non_commit_source_identity(tmp_path, commit):
    _sealed_manifest_with_change(tmp_path, "source_commit", commit)
    with pytest.raises(ValueError, match="source commit"):
        study.audit_archive(tmp_path)


@pytest.mark.parametrize("key,value", [
    ("default_changed", True), ("default_changed", 0), ("default_changed", None),
    ("new_holdout", True), ("new_holdout", "False"), ("new_holdout", 0),
])
def test_resealed_manifest_rejects_changed_scope_flags(tmp_path, key, value):
    _sealed_manifest_with_change(tmp_path, key, value)
    with pytest.raises(ValueError, match="archive identity/count/inventory differs"):
        study.audit_archive(tmp_path)


def test_json_roundtrip_of_real_case_metadata_does_not_invalidate_the_summary():
    spec = study.specifications(study.protocol_document())[0]
    rebuilt = {"case": study._case_document(spec["case"]), "metric": 1.375}
    saved = json.loads(json.dumps(rebuilt))
    assert saved != rebuilt  # Real schedule/configuration tuples become JSON lists.
    study.verify_comparison(saved, rebuilt)


@pytest.mark.parametrize("changed", [1.375 + 1e-12, True, float("nan")])
def test_json_normalization_does_not_relax_values(changed):
    with pytest.raises(ValueError):
        study.verify_comparison({"metric": changed}, {"metric": 1.375})
    with pytest.raises(ValueError, match="comparison differs"):
        study.verify_comparison({"default_changed": 0}, {"default_changed": False})


def test_source_check_accepts_exact_inputs_or_the_one_recorded_serialization_fix():
    sources = study.source_identity()
    assert study.verify_sources(sources, "a" * 40) == "exact"
    sources[study.STUDY_SOURCE_PATH] = study.SERIALIZATION_FIX_STUDY_SHA256
    assert study.verify_sources(sources, study.SERIALIZATION_FIX_EXECUTION_COMMIT) == (
        "serialization-finalization-only compatibility"
    )


@pytest.mark.parametrize("change", ["commit", "runner_hash", "other_source", "extra_source"])
def test_legacy_source_exception_rejects_every_other_change(change):
    sources = study.source_identity()
    sources[study.STUDY_SOURCE_PATH] = study.SERIALIZATION_FIX_STUDY_SHA256
    commit = study.SERIALIZATION_FIX_EXECUTION_COMMIT
    if change == "commit":
        commit = "b" * 40
    elif change == "runner_hash":
        sources[study.STUDY_SOURCE_PATH] = "c" * 64
    elif change == "other_source":
        sources["tools/reversal_recovery/controller.py"] = "d" * 64
    else:
        sources["tools/not_part_of_the_experiment.py"] = "e" * 64
    with pytest.raises(ValueError, match="source identity differs"):
        study.verify_sources(sources, commit)


@pytest.fixture
def copied_recovery_archive(tmp_path):
    # During pre-publication testing this points at the complete, read-only staging
    # archive. CI uses the published directory. Neither source is modified by tests.
    source = Path(os.environ.get(
        "REVERSAL_RECOVERY_TEST_ARCHIVE", study.ROOT / "results/franka_reversal_recovery",
    ))
    if not source.is_dir():
        raise AssertionError("the real eight-trace archive is required for finalization tests")
    destination = tmp_path / "staging"
    shutil.copytree(source, destination)
    return destination


def _artifact_hashes(directory):
    return {path.relative_to(directory).as_posix(): study._sha256(path)
            for path in directory.rglob("*") if path.is_file()}


def test_finalize_cli_only_audits_and_preserves_every_original_artifact(copied_recovery_archive, tmp_path):
    staging = copied_recovery_archive
    manifest_before = (staging / "manifest.json").read_bytes()
    hashes_before = _artifact_hashes(staging)
    destination = tmp_path / "published"
    output = subprocess.run(
        [sys.executable, "-m", "tools.reversal_recovery.study", "--output", str(destination),
         "--finalize", str(staging)], cwd=study.ROOT, capture_output=True, text=True, check=True,
    )
    report = json.loads(output.stdout)
    assert report["archive_integrity"] == "PASS"
    assert report["comparison_status"] == "PASS"
    assert "executed " not in output.stdout
    assert not staging.exists()
    assert (destination / "manifest.json").read_bytes() == manifest_before
    assert _artifact_hashes(destination) == hashes_before


def test_finalize_refuses_an_existing_destination_without_touching_staging(copied_recovery_archive, tmp_path):
    staging = copied_recovery_archive
    before = _artifact_hashes(staging)
    destination = tmp_path / "already_exists"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("user content")
    with pytest.raises(ValueError, match="output must be new"):
        study.finalize(staging, destination)
    assert marker.read_text() == "user content"
    assert _artifact_hashes(staging) == before


def test_finalize_failed_audit_preserves_original_files(copied_recovery_archive, tmp_path):
    staging = copied_recovery_archive
    comparison = staging / "comparison.json"
    comparison.write_text('{"status":"forged"}')
    before = _artifact_hashes(staging)
    destination = tmp_path / "not_created"
    with pytest.raises(ValueError, match="artifact hash differs"):
        study.finalize(staging, destination)
    assert staging.exists() and not destination.exists()
    assert _artifact_hashes(staging) == before


def test_finalize_rejects_a_symlink_source_before_reading_or_moving_it(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    link = tmp_path / "staging-link"
    link.symlink_to(original, target_is_directory=True)
    destination = tmp_path / "published"
    with pytest.raises(ValueError, match="staging.*symlink"):
        study.finalize(link, destination)
    assert link.is_symlink() and original.is_dir()
    assert not destination.exists()
