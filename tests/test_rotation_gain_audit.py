import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.surface_simulation import SurfaceTrialResult
from tools import rotation_gain_regression as regression

SCRIPT = Path(__file__).parents[1] / "tools" / "audit_rotation_gain_regression.py"
SPEC = importlib.util.spec_from_file_location("audit_rotation_gain_regression", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _fake_trial(case, method, scale):
    config, scenario, task = (case[name] for name in ("config", "scenario", "task"))
    time = np.arange(round(config.duration / config.timestep)) * config.timestep
    count = len(time)
    angle = np.deg2rad(scenario.wall_yaw_deg)
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    target = np.tile(np.array([0.375, 0.0, 0.5]), (count, 1))
    gap = (np.array([0.4, 0.0, 0.0]) - target) @ normal - 0.025
    trace = {
        "time": time,
        "true_normal_force": np.full(count, 12.0),
        "target_normal_force": np.full(count, 12.0),
        "position": target.copy(),
        "target_position": target,
        "linear_velocity": np.zeros((count, 3)),
        "target_linear_velocity": np.zeros((count, 3)),
        "orientation_error_rad": np.zeros(count),
        "measured_normal_force": np.full(count, 12.0),
        "commanded_torque": np.zeros((count, 7)),
        "applied_torque": np.zeros((count, 7)),
        "lower_torque_limit": np.full((count, 7), -10.0),
        "upper_torque_limit": np.full((count, 7), 10.0),
        "torque_projection_scale": np.ones(count),
        "requested_tangential_force_world": np.zeros((count, 3)),
        "true_contact_gap_m": gap,
        "rotation_gain_scale": np.array(scale),
        "case_index": np.array(case["case_index"]),
        "method": np.array(method),
    }
    return SurfaceTrialResult(trace, scenario, config, task)


def _fake_save_trace(path, trace):
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, **trace)
    return Path(path)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr(regression, "run_trial", _fake_trial)
    monkeypatch.setattr(regression, "save_surface_trace", _fake_save_trace)
    monkeypatch.setattr(regression, "_frozen_metric_error", lambda *_: 0.0)
    output = regression.generate(tmp_path / "archive", case_indices=[0])
    _, reference_configs, reference = audit._reference()
    with (output / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        published = list(csv.DictReader(handle))
    synthetic_reference = {
        (row["method"], int(row["case_index"])): row
        for row in published
        if float(row["scale"]) == 1.0
    }
    monkeypatch.setattr(
        audit,
        "_reference",
        lambda: (synthetic_reference, reference_configs, reference),
    )
    return output


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(directory, relative):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"][relative] = _sha256(directory / relative)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (directory / "COMPLETE").write_text(_sha256(manifest_path) + "\n")


def _rewrite_npz(path, mutate):
    with np.load(path, allow_pickle=False) as loaded:
        values = {name: loaded[name] for name in loaded.files}
    mutate(values)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **values)


def test_valid_subset_recomputes_comparison_pairs_summary_and_traces(archive):
    assert audit.audit_archive(archive) == {
        "archive": str(archive.resolve()),
        "identity": regression.IDENTITY,
        "current_source_matches_archive": True,
        "source_identity_status": "current",
        "is_subset": True,
        "comparison_rows": 8,
        "paired_rows": 4,
        "compact_traces": 8,
        "full_traces": 4,
        "status": "PASS",
    }


def test_unsealed_trace_corruption_is_rejected_before_arithmetic(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    trace = next((changed / "traces").glob("*compact.npz"))
    with trace.open("ab") as handle:
        handle.write(b"damage")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        audit.audit_archive(changed)


def test_resealed_compact_geometry_change_is_rejected(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    trace = next((changed / "traces").glob("*compact.npz"))
    _rewrite_npz(trace, lambda values: values["position"].__setitem__((0, 0), 0.5))
    relative = str(trace.relative_to(changed))
    _reseal(changed, relative)
    with pytest.raises(ValueError, match="contact gap differs"):
        audit.audit_archive(changed)


def test_resealed_full_trace_identity_change_is_rejected(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    trace = next((changed / "traces").glob("s2*full.npz"))
    _rewrite_npz(
        trace,
        lambda values: values.__setitem__("rotation_gain_scale", np.array(1.0)),
    )
    relative = str(trace.relative_to(changed))
    _reseal(changed, relative)
    with pytest.raises(ValueError, match="trace scalar identity mismatch"):
        audit.audit_archive(changed)


def test_resealed_full_trace_compact_field_change_is_rejected(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    trace = next((changed / "traces").glob("*full.npz"))
    _rewrite_npz(trace, lambda values: values["position"].__setitem__((0, 0), 0.5))
    relative = str(trace.relative_to(changed))
    _reseal(changed, relative)
    with pytest.raises(ValueError, match="full trace differs from compact field position"):
        audit.audit_archive(changed)


def test_scale_one_must_reproduce_pinned_reference_within_tolerance(archive, monkeypatch):
    reference_rows, reference_configs, reference = audit._reference()
    changed_rows = {key: dict(row) for key, row in reference_rows.items()}
    changed_rows["baseline", 0]["force_rmse_n"] = "0.001"
    monkeypatch.setattr(
        audit,
        "_reference",
        lambda: (changed_rows, reference_configs, reference),
    )
    with pytest.raises(ValueError, match="does not reproduce the pinned public96 metrics"):
        audit.audit_archive(archive)


def test_resealed_paired_metric_change_is_rejected(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    path = changed / "paired.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["force_rmse_n"] = "1.0"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _reseal(changed, "paired.csv")
    with pytest.raises(ValueError, match="paired.force_rmse_n mismatch"):
        audit.audit_archive(changed)


def test_resealed_summary_change_is_rejected(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    path = changed / "summary.json"
    summary = json.loads(path.read_text())
    summary["default_changed"] = True
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _reseal(changed, "summary.json")
    with pytest.raises(ValueError, match="summary.default_changed mismatch"):
        audit.audit_archive(changed)


def test_resealed_arbitrary_source_identity_is_rejected(tmp_path, archive):
    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    path = changed / "source_hashes.json"
    sources = json.loads(path.read_text())
    sources["invented.py"] = "0" * 64
    path.write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
    _reseal(changed, "source_hashes.json")
    with pytest.raises(ValueError, match="neither current nor a known archive"):
        audit.audit_archive(changed)


def test_registered_archive_source_remains_auditable_after_host_drift(archive, monkeypatch):
    monkeypatch.setattr(audit, "_live_source_identity", lambda: {"future.py": "0" * 64})
    result = audit.audit_archive(archive)
    assert result["current_source_matches_archive"] is False
    assert result["source_identity_status"] == "known_archive"


def test_unknown_archive_source_is_rejected_after_host_drift(archive, monkeypatch):
    monkeypatch.setattr(audit, "_live_source_identity", lambda: {"future.py": "0" * 64})
    monkeypatch.setattr(audit, "KNOWN_ARCHIVE_SOURCE_HASHES_SHA256", frozenset())
    with pytest.raises(ValueError, match="neither current nor a known archive"):
        audit.audit_archive(archive)
