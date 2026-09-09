import csv
import errno
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools.audit_compensation_budget_study import audit_archive

ARCHIVE = Path(__file__).parents[1] / "results/franka_compensation_budget"


@pytest.fixture
def archive(tmp_path):
    """Use hard links; mutation helpers detach only the files a test changes."""
    result = tmp_path / "archive"
    shutil.copytree(ARCHIVE, result, copy_function=link_or_copy)
    return result


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, destination)
    return destination


def detach(path):
    copy = path.with_name(f".{path.name}.detached")
    shutil.copyfile(path, copy)
    os.replace(copy, path)


def reseal(root):
    manifest_path = root / "manifest.json"
    detach(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for name in manifest["artifact_sha256"]:
        manifest["artifact_sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in manifest["input_sha256"]:
        manifest["input_sha256"][name] = manifest["artifact_sha256"][name]
    manifest_path.write_text(json.dumps(manifest))
    complete = root / "COMPLETE"
    detach(complete)
    complete.write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest())


def mutate_npz(path, field, update):
    detach(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    update(values[field])
    np.savez_compressed(path, **values)


def mutate_csv(path, update):
    detach(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    update(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def trace_for_budget(root, budget):
    for path in sorted((root / "traces").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if float(archive["max_force_n"]) == budget:
                return path
    raise AssertionError(f"archive has no {budget:g} N trace")


def test_complete_archive_recomputes_integrity_independently_of_screen_result():
    result = audit_archive(ARCHIVE)
    assert result["status"] == "PASS"
    assert (result["runs"], result["diagnostic_rows"], result["paired_rows"]) == (12, 36, 18)
    assert result["exact_reference_runs"] == 6
    assert result["pre_onset_exact_pairs"] == 6
    assert result["screening_status"] in {"PASS", "FAIL"}
    assert 0 <= result["screening_passed_pairs"] <= 6
    assert result["default_changed"] is False
    assert result["comparator_gates_evaluated"] is False


def test_resealed_eight_newton_trace_budget_tamper_rejects(archive):
    path = trace_for_budget(archive, 8.0)
    mutate_npz(path, "max_force_n", lambda value: value.__setitem__((), 6.0))
    reseal(archive)
    with pytest.raises(ValueError, match="max_force_n"):
        audit_archive(archive)


def test_resealed_suppressed_cap_telemetry_rejects(archive):
    path = trace_for_budget(archive, 6.0)

    def suppress_one(values):
        capped = np.flatnonzero(values)
        assert capped.size
        values[capped[0]] = False

    mutate_npz(path, "diagnostic_amplitude_capped", suppress_one)
    reseal(archive)
    with pytest.raises(ValueError, match="amplitude cap flag"):
        audit_archive(archive)


@pytest.mark.parametrize(
    ("field", "update", "message"),
    (
        ("time", lambda value: value.__setitem__(100, value[100] + 0.001), "time grid"),
        (
            "requested_tangential_force_world",
            lambda value: value.__setitem__((4500, 1), value[4500, 1] + 0.1),
            "slew observer flag|observed compensation request",
        ),
    ),
)
def test_resealed_time_or_force_command_tamper_rejects(archive, field, update, message):
    mutate_npz(trace_for_budget(archive, 8.0), field, update)
    reseal(archive)
    with pytest.raises(ValueError, match=message):
        audit_archive(archive)


def test_resealed_source_hash_deletion_rejects(archive):
    path = archive / "source_hashes.json"
    detach(path)
    content = json.loads(path.read_text())
    del content[next(iter(content))]
    path.write_text(json.dumps(content))
    reseal(archive)
    with pytest.raises(ValueError, match="source identity"):
        audit_archive(archive)


def test_resealed_manifest_budget_tamper_rejects(archive):
    path = archive / "manifest.json"
    detach(path)
    content = json.loads(path.read_text())
    content["budgets_n"] = [6.0, 9.0]
    path.write_text(json.dumps(content))
    reseal(archive)
    with pytest.raises(ValueError, match="wrong budget identity"):
        audit_archive(archive)


def test_resealed_reversed_paired_delta_rejects(archive):
    path = archive / "paired_diagnostics.csv"

    def reverse_delta(rows):
        fields = [
            name for name in rows[0] if name.startswith("delta_") and name != "delta_direction"
        ]
        for row in rows:
            for field in fields:
                value = float(row[field])
                if value != 0.0:
                    row[field] = str(-value)
                    return
        raise AssertionError("paired table has no nonzero delta")

    mutate_csv(path, reverse_delta)
    reseal(archive)
    with pytest.raises(ValueError, match="pair"):
        audit_archive(archive)


def test_resealed_screening_status_lie_rejects_even_if_real_screen_failed(archive):
    path = archive / "screening.json"
    detach(path)
    content = json.loads(path.read_text())
    content["status"] = "FAIL" if content["status"] == "PASS" else "PASS"
    path.write_text(json.dumps(content))
    reseal(archive)
    with pytest.raises(ValueError, match="screening result"):
        audit_archive(archive)


def test_unlisted_extra_file_rejects_with_an_unchanged_valid_manifest(archive):
    (archive / "unlisted.txt").write_text("not declared by the manifest\n")
    with pytest.raises(ValueError, match="unexpected archive files"):
        audit_archive(archive)
