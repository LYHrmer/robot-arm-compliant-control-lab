import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools.audit_cross_surface_regression import audit_archive

ARCHIVE = Path(__file__).resolve().parents[1] / "results/franka_cross_surface_dynamic"


@pytest.fixture
def archive(tmp_path):
    result = tmp_path / "archive"
    shutil.copytree(ARCHIVE, result)
    return result


def reseal(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name in manifest["artifact_sha256"]:
        manifest["artifact_sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in manifest["input_sha256"]:
        manifest["input_sha256"][name] = manifest["artifact_sha256"][name]
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    (root / "COMPLETE").write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def mutate_npz(path, field, update):
    with np.load(path, allow_pickle=False) as data:
        values = {k: data[k] for k in data.files}
    update(values[field])
    np.savez_compressed(path, **values)


def mutate_csv(path, mutate):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    mutate(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def test_complete_archive_recomputes_all_rows_and_pinned_references():
    result = audit_archive(ARCHIVE)
    assert result["status"] == "PASS"
    assert (result["comparison_rows"], result["phase_rows"], result["gain_pairs"]) == (36, 144, 90)
    assert all(value == 0 for group in result["reference_max_abs_error"].values() for value in group.values())


def test_resealed_raw_position_tamper_rejects(archive):
    path = archive / "traces/yaw_0__stop_hold_reverse__seed_11__online__s2__inputs.npz"
    mutate_npz(path, "position", lambda a: a.__setitem__((2500, 1), a[2500, 1] + .01))
    reseal(archive)
    with pytest.raises(ValueError, match="tangent_position"):
        audit_archive(archive)


def test_resealed_actuator_limit_expansion_rejects(archive):
    path = archive / "traces/yaw_0__stop_hold_reverse__seed_11__online__s2__inputs.npz"
    mutate_npz(path, "upper_torque_limit", lambda a: a.__setitem__((slice(None), 6), 120.0))
    reseal(archive)
    with pytest.raises(ValueError, match="fixed upper actuator"):
        audit_archive(archive)


def test_resealed_source_map_is_not_a_trusted_source(archive):
    path = archive / "source_hashes.json"
    content = json.loads(path.read_text())
    content["tools/cross_surface_regression.py"] = "0" * 64
    path.write_text(json.dumps(content))
    reseal(archive)
    with pytest.raises(ValueError, match="source"):
        audit_archive(archive)


def test_resealed_gain_pair_sign_change_rejects(archive):
    mutate_csv(archive / "gain_paired.csv", lambda rows: rows[0].__setitem__("delta_orientation_rmse_deg", "1.0"))
    reseal(archive)
    with pytest.raises(ValueError, match="gain pair"):
        audit_archive(archive)


def test_resealed_hidden_known_failure_rejects(archive):
    def hide(rows):
        failed = next(r for r in rows if r["failed_gates"])
        failed.update(failed_gates="", all_gates_pass="yes", all_checks_pass="yes")
    mutate_csv(archive / "phase_metrics.csv", hide)
    reseal(archive)
    with pytest.raises(ValueError, match="phase gates"):
        audit_archive(archive)


def test_resealed_full_trace_force_disagrees_with_compact(archive):
    path = archive / "traces/yaw_-15__stop_hold_reverse__seed_11__online__s1__full.npz"
    mutate_npz(path, "true_normal_force", lambda a: a.__setitem__(2500, a[2500] + .1))
    reseal(archive)
    with pytest.raises(ValueError, match="full compact"):
        audit_archive(archive)


def test_resealed_wrong_controller_frame_rejects(archive):
    path = archive / "configurations.json"
    configurations = json.loads(path.read_text())
    configurations["executions"][0]["controller"]["surface_frame_rotation"] = np.eye(3).tolist()
    path.write_text(json.dumps(configurations))
    reseal(archive)
    with pytest.raises(ValueError, match="execution matrix"):
        audit_archive(archive)
