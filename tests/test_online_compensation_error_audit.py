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
SPEC = importlib.util.spec_from_file_location("audit_online_compensation_errors", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _fake_trial(case, arm):
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


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(directory, changed_artifact):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"][changed_artifact] = _sha(directory / changed_artifact)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (directory / "COMPLETE").write_text(_sha(manifest_path) + "\n")


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
        "is_subset": True,
        "comparison_rows": 3,
        "phase_rows": 6,
        "compact_traces": 3,
        "full_traces": 0,
        "status": "PASS",
    }


def test_bytes_corruption_is_rejected_before_metrics(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    trace = next((changed / "traces").glob("*online*compact.npz"))
    with trace.open("ab") as handle:
        handle.write(b"damage")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
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
