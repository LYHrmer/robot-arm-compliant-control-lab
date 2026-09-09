import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import compliant_control_lab.online_compensation_experiment as experiment

SCRIPT = Path(__file__).parents[1] / "tools" / "audit_online_compensation_errors.py"
HISTORICAL_ARCHIVE = Path(__file__).parents[1] / "results" / "franka_online_compensation_errors"
SPEC = importlib.util.spec_from_file_location("audit_online_compensation_errors", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _fake_trial(case, arm, rotation_gain_scale=1.0):
    time = np.arange(0.0, case.config.duration, case.config.timestep)
    count = len(time)
    tangent = experiment.yaw_frame(case.task.yaw_deg).rotation[:, 1]
    force_offset = {"baseline": 0.15, "friction": 0.0, "online": 0.1}[arm]
    after = (
        np.linspace(0.45, 0.50, count)
        if arm == "online"
        else np.full(count, 0.45 if arm == "friction" else 0.0)
    )
    before = np.concatenate((after[:1], after[:-1]))
    friction = np.array([case.friction.at(t) for t in time])
    bias = np.array([case.wrench_bias_world.at(t) for t in time])
    position_error = np.tile(0.001 * tangent, (count, 1))
    target_position = np.tile(np.array([0.4, 0.0, 0.5]), (count, 1))
    return SimpleNamespace(
        trace={
            "schema_version": np.array(1),
            "dt": np.array(case.config.timestep),
            "rotation_gain_scale": np.array(rotation_gain_scale),
            "controller_kind": np.array(
                "surface_adaptive" if arm == "baseline" else f"surface_{arm}"
            ),
            "controller_frame_rotation": experiment.yaw_frame(
                case.task.yaw_deg + case.controller_yaw_error_deg
            ).rotation,
            "time": time,
            "position": target_position + position_error,
            "linear_velocity": np.tile(0.018 * tangent, (count, 1)),
            "target_position": target_position,
            "target_linear_velocity": np.tile(0.02 * tangent, (count, 1)),
            "target_normal_force": np.full(count, 12.0),
            "measured_normal_force": np.full(count, 12.0),
            "true_normal_force": np.full(count, 12.0 + force_offset),
            "orientation_error_rad": np.full(count, np.deg2rad(0.05)),
            "commanded_torque": np.zeros((count, 7)),
            "applied_torque": np.zeros((count, 7)),
            "contact_blend": np.ones(count),
            "torque_projection_scale": np.ones(count),
            "requested_tangential_force_world": np.zeros((count, 3)),
            "applied_wall_friction": friction,
            "applied_tool_friction": friction,
            "applied_raw_wrench_bias_world": bias,
            "feedback_raw_wrench_bias_world": np.concatenate((bias[:1], bias[:-1])),
            "controller_yaw_error_deg": np.full(count, case.controller_yaw_error_deg),
            "controller_coefficient_before_compute": before,
            "controller_coefficient_after_compute": after,
            "controller_update_ready_before_compute": np.full(count, arm == "online"),
            "controller_update_ready_after_compute": np.full(count, arm == "online"),
            "trajectory_rate_scale": np.array(
                [case.trajectory.rate_scale_at(t) for t in time]
            ),
        }
    )


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "run_protocol_trial", _fake_trial)
    return experiment.generate_online_compensation_experiment(
        tmp_path / "archive",
        case_names=["friction_step_up"],
        arms=["baseline", "friction", "online"],
        seeds=[11],
    )


@pytest.fixture
def scaled_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "run_protocol_trial", _fake_trial)
    return experiment.generate_online_compensation_experiment(
        tmp_path / "scaled_archive",
        case_names=["friction_step_up"],
        arms=["baseline", "friction", "online"],
        seeds=[11],
        full_traces=True,
        rotation_gain_scale=2.0,
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(directory, changed_artifact):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = _sha(directory / changed_artifact)
    manifest["artifact_sha256"][changed_artifact] = digest
    if changed_artifact in manifest["input_sha256"]:
        manifest["input_sha256"][changed_artifact] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (directory / "COMPLETE").write_text(_sha(manifest_path) + "\n")


def _rewrite_json(path, mutate):
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _rewrite_trace(path, mutate):
    with np.load(path, allow_pickle=False) as loaded:
        trace = {name: loaded[name] for name in loaded.files}
    mutate(trace)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **trace)


def test_valid_subset_audit_reconstructs_every_row_phase_gate_and_schedule(archive):
    result = audit.audit_archive(archive)
    assert result == {
        "archive": str(archive.resolve()),
        "identity": experiment.PROTOCOL_ID,
        "current_source_matches_archive": True,
        "source_identity_status": "current",
        "is_subset": True,
        "comparison_rows": 3,
        "phase_rows": 6,
        "compact_traces": 3,
        "full_traces": 0,
        "status": "PASS",
    }


def test_host_package_drift_is_reported_without_invalidating_offline_audit(
    archive, monkeypatch
):
    monkeypatch.setattr(
        audit,
        "KNOWN_ARCHIVE_SOURCE_HASHES_SHA256",
        frozenset({_sha(archive / "source_hashes.json")}),
    )
    monkeypatch.setattr(audit, "_live_source_hashes", lambda: {"future.py": "0" * 64})
    result = audit.audit_archive(archive)
    assert result["status"] == "PASS"
    assert result["current_source_matches_archive"] is False
    assert result["source_identity_status"] == "known_archive"


def test_real_historical_archive_uses_fixed_known_source_identity():
    result = audit.audit_archive(HISTORICAL_ARCHIVE)
    assert result["current_source_matches_archive"] is False
    assert result["source_identity_status"] == "known_archive"


def test_unknown_host_source_drift_is_rejected(archive, monkeypatch):
    monkeypatch.setattr(audit, "KNOWN_ARCHIVE_SOURCE_HASHES_SHA256", frozenset())
    monkeypatch.setattr(audit, "_live_source_hashes", lambda: {"future.py": "0" * 64})
    with pytest.raises(ValueError, match="neither current nor a known archive"):
        audit.audit_archive(archive)


def test_resealed_arbitrary_source_map_is_rejected(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    _rewrite_json(
        changed / "source_hashes.json",
        lambda payload: payload.__setitem__("invented.py", "0" * 64),
    )
    _reseal(changed, "source_hashes.json")
    with pytest.raises(ValueError, match="neither current nor a known archive"):
        audit.audit_archive(changed)


def test_legacy_archive_without_rotation_scale_metadata_defaults_to_one(tmp_path, archive):
    changed = tmp_path / "legacy"
    shutil.copytree(archive, changed)
    for name in ("protocol.json", "configurations.json"):
        _rewrite_json(changed / name, lambda payload: payload.pop("rotation_gain_scale"))
        _reseal(changed, name)
    _rewrite_json(changed / "manifest.json", lambda payload: payload.pop("rotation_gain_scale"))
    (changed / "COMPLETE").write_text(_sha(changed / "manifest.json") + "\n")
    assert audit.audit_archive(changed)["status"] == "PASS"


def test_scale_two_archive_with_matching_gains_and_full_metadata_passes(scaled_archive):
    result = audit.audit_archive(scaled_archive)
    assert result["status"] == "PASS"
    assert result["full_traces"] == 3


def test_resealed_rotation_variant_metadata_mismatch_is_rejected(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    _rewrite_json(
        changed / "configurations.json",
        lambda payload: payload.__setitem__("rotation_gain_scale", 2.0),
    )
    _reseal(changed, "configurations.json")
    with pytest.raises(ValueError, match="metadata differs across frozen JSON files"):
        audit.audit_archive(changed)


def test_resealed_controller_gain_mismatch_is_rejected(tmp_path, scaled_archive):
    changed = tmp_path / "changed"
    shutil.copytree(scaled_archive, changed)

    def mutate(payload):
        payload["controller_constructor_configurations"]["online"]["safe_adaptive_base"][
            "base"
        ]["base"]["rotational_stiffness"][0] = 20.0

    _rewrite_json(changed / "protocol.json", mutate)
    _reseal(changed, "protocol.json")
    with pytest.raises(ValueError, match="online controller rotational stiffness"):
        audit.audit_archive(changed)


def test_resealed_full_trace_rotation_scale_mismatch_is_rejected(tmp_path, scaled_archive):
    changed = tmp_path / "changed"
    shutil.copytree(scaled_archive, changed)
    trace_path = next((changed / "traces").glob("*online*full.npz"))
    _rewrite_trace(
        trace_path,
        lambda trace: trace.__setitem__("rotation_gain_scale", np.array(1.0)),
    )
    _reseal(changed, str(trace_path.relative_to(changed)))
    with pytest.raises(ValueError, match="full trace rotation_gain_scale differs"):
        audit.audit_archive(changed)


def test_bytes_corruption_is_rejected_before_metrics(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    trace = next((changed / "traces").glob("*online*compact.npz"))
    with trace.open("ab") as handle:
        handle.write(b"damage")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        audit.audit_archive(changed)


def test_archived_source_hash_file_corruption_is_still_rejected(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    source_hashes = changed / "source_hashes.json"
    source_hashes.write_bytes(source_hashes.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact hash mismatch: source_hashes.json"):
        audit.audit_archive(changed)


def test_resealed_metric_change_is_rejected_by_trace_reconstruction(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    csv_path = changed / "comparison.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[-1]["force_rmse_n"] = "9.0"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _reseal(changed, "comparison.csv")
    with pytest.raises(ValueError, match="reconstructed metric mismatch"):
        audit.audit_archive(changed)


def test_resealed_missing_row_is_rejected_by_exact_grid(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    csv_path = changed / "comparison.csv"
    lines = csv_path.read_text().splitlines()
    csv_path.write_text("\n".join(lines[:-1]) + "\n")
    _reseal(changed, "comparison.csv")
    with pytest.raises(ValueError, match="comparison grid"):
        audit.audit_archive(changed)


def test_resealed_trace_schedule_change_is_rejected(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    trace_path = next((changed / "traces").glob("*online*compact.npz"))
    _rewrite_trace(trace_path, lambda trace: trace["applied_wall_friction"].__setitem__(0, 0.99))
    relative = str(trace_path.relative_to(changed))
    _reseal(changed, relative)
    with pytest.raises(ValueError, match="friction schedule"):
        audit.audit_archive(changed)


def test_resealed_active_coefficient_jump_cannot_masquerade_as_reset(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    trace_path = next((changed / "traces").glob("*online*compact.npz"))

    def mutate(trace):
        index = len(trace["time"]) // 2
        trace["controller_coefficient_after_compute"][index:] = 0.45
        trace["controller_coefficient_before_compute"][index + 1 :] = 0.45

    _rewrite_trace(trace_path, mutate)
    _reseal(changed, str(trace_path.relative_to(changed)))
    with pytest.raises(ValueError, match="illegitimate coefficient reset"):
        audit.audit_archive(changed)


def test_resealed_active_force_jump_cannot_masquerade_as_reset(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    trace_path = next((changed / "traces").glob("*online*compact.npz"))

    def mutate(trace):
        index = len(trace["time"]) // 2
        trace["requested_tangential_force_delta_norm_n"][index] = 1.0

    _rewrite_trace(trace_path, mutate)
    _reseal(changed, str(trace_path.relative_to(changed)))
    with pytest.raises(ValueError, match="illegitimate force reset"):
        audit.audit_archive(changed)
