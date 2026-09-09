import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools import combined_residual_ablation as study
from tools.audit_combined_residual_ablation import audit_archive
from tools.combined_residual_validation import check_unchanged_prefix

ARCHIVE = Path(__file__).parents[1] / "results/franka_combined_residual_ablation"


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
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    update(values[field])
    np.savez_compressed(path, **values)


def mutate_csv(path, update):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    update(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def trace_path(root, yaw, variant):
    case = next(
        case
        for label, case in study.ablation_cases()
        if label == variant and int(case.scenario.wall_yaw_deg) == yaw
    )
    return root / "traces" / f"{study.stem(variant, case)}__diagnostic.npz"


def load_trace(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def test_complete_archive_recomputes_tables_references_and_prefixes():
    result = audit_archive(ARCHIVE)
    assert result["status"] == "PASS"
    assert tuple(result[name] for name in (
        "runs", "phase_rows", "diagnostic_rows", "paired_rows"
    )) == (12, 36, 36, 27)
    assert result["original_combined_exact_runs"] == 3
    assert result["pre_onset_exact_pairs"] == 6
    assert result["comparator_gates_evaluated"] is False


def test_resealed_cap_flag_flip_rejects(archive):
    path = trace_path(archive, 0, "combined")
    mutate_npz(path, "diagnostic_amplitude_capped", lambda a: a.__setitem__(4500, ~a[4500]))
    reseal(archive)
    with pytest.raises(ValueError, match="amplitude cap flag"):
        audit_archive(archive)


def test_resealed_actuator_limit_expansion_rejects(archive):
    path = trace_path(archive, 0, "combined")
    mutate_npz(
        path,
        "upper_torque_limit",
        lambda a: a.__setitem__((slice(None), 6), 120.0),
    )
    reseal(archive)
    with pytest.raises(ValueError, match="fixed upper torque limits"):
        audit_archive(archive)


def test_resealed_source_map_rejects(archive):
    path = archive / "source_hashes.json"
    content = json.loads(path.read_text())
    content[next(iter(content))] = "0" * 64
    path.write_text(json.dumps(content))
    reseal(archive)
    with pytest.raises(ValueError, match="source identity"):
        audit_archive(archive)


def test_resealed_paired_diagnostic_delta_rejects(archive):
    field = f"delta_{study.PAIRED_DIAGNOSTICS[0]}"
    mutate_csv(archive / "paired_diagnostics.csv", lambda rows: rows[0].__setitem__(field, "1e9"))
    reseal(archive)
    with pytest.raises(ValueError, match="diagnostic pair"):
        audit_archive(archive)


def test_no_bias_prefix_check_observes_intervention_boundary():
    combined = load_trace(trace_path(ARCHIVE, -15, "combined"))
    no_bias = load_trace(trace_path(ARCHIVE, -15, "no_bias"))
    check_unchanged_prefix(combined, no_bias, 6.0)

    before = {name: values.copy() for name, values in no_bias.items()}
    before["position"][np.searchsorted(before["time"], 5.0), 0] += 1e-6
    with pytest.raises(ValueError, match="pre-onset.*position"):
        check_unchanged_prefix(combined, before, 6.0)

    after = {name: values.copy() for name, values in no_bias.items()}
    after["position"][np.searchsorted(after["time"], 7.0), 0] += 1e-6
    check_unchanged_prefix(combined, after, 6.0)
