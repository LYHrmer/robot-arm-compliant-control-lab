from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from compliant_control_lab.published_results_audit import audit_published_results

REPOSITORY_ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = REPOSITORY_ROOT / "results/franka_safety_preholdout/protocol.json"
RESULT_DIR = REPOSITORY_ROOT / "results/franka_safety_blind"


def _copy_result(tmp_path: Path) -> Path:
    copied = tmp_path / "blind-result"
    shutil.copytree(RESULT_DIR, copied)
    return copied


def _copy_protocol(tmp_path: Path) -> Path:
    copied = tmp_path / "preholdout"
    shutil.copytree(PROTOCOL_PATH.parent, copied)
    return copied / "protocol.json"


def _refresh_manifest_hash(result_dir: Path, filename: str) -> None:
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"][filename] = hashlib.sha256(
        (result_dir / filename).read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _edit_csv(
    result_dir: Path,
    edit_row: Callable[[list[dict[str, str]]], None],
) -> None:
    csv_path = result_dir / "comparison.csv"
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    edit_row(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _refresh_manifest_hash(result_dir, "comparison.csv")


def test_published_v05_result_passes_offline_audit() -> None:
    report = audit_published_results(PROTOCOL_PATH, RESULT_DIR)

    assert report.row_count == 384
    assert report.method_count == 8
    assert report.case_count == 48
    assert report.pass_counts == {
        "fixed_hybrid": 17,
        "adaptive_hybrid": 23,
        "safe_adaptive_hybrid": 24,
        "torque_residual_run_00": 22,
        "torque_residual_run_01": 25,
        "torque_residual_run_02": 26,
        "torque_residual_run_03": 24,
        "torque_residual_run_04": 25,
    }
    assert report.primary_rule_passed is False


def test_audit_rejects_an_artifact_that_no_longer_matches_manifest(tmp_path: Path) -> None:
    result_dir = _copy_result(tmp_path)
    (result_dir / "summary.md").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact checksum mismatch: summary.md"):
        audit_published_results(PROTOCOL_PATH, result_dir)


def test_audit_rejects_a_changed_protocol_without_a_new_checksum(tmp_path: Path) -> None:
    protocol_path = _copy_protocol(tmp_path)
    protocol_path.write_bytes(protocol_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="protocol checksum mismatch"):
        audit_published_results(protocol_path, RESULT_DIR)


def test_audit_rejects_a_different_protocol_even_with_a_new_checksum(tmp_path: Path) -> None:
    protocol_path = _copy_protocol(tmp_path)
    protocol_path.write_bytes(protocol_path.read_bytes() + b" ")
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    protocol_path.with_name("protocol.sha256").write_text(
        f"{protocol_hash}  protocol.json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not the published v0.5 contract"):
        audit_published_results(protocol_path, RESULT_DIR)


def test_audit_rejects_a_changed_frozen_policy_file(tmp_path: Path) -> None:
    protocol_path = _copy_protocol(tmp_path)
    policy_path = protocol_path.parent / "policy_00.json"
    policy_path.write_bytes(policy_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="artifact checksum mismatch: policy_00.json"):
        audit_published_results(protocol_path, RESULT_DIR)


def test_audit_rejects_a_case_seed_changed_after_reveal(tmp_path: Path) -> None:
    result_dir = _copy_result(tmp_path)

    def edit(rows: list[dict[str, str]]) -> None:
        rows[0]["simulation_seed"] = "999"

    _edit_csv(result_dir, edit)

    with pytest.raises(ValueError, match="seeds do not match reveal: blind_00"):
        audit_published_results(PROTOCOL_PATH, result_dir)


def test_audit_rederives_seeds_even_if_reveal_and_csv_are_changed(tmp_path: Path) -> None:
    result_dir = _copy_result(tmp_path)
    reveal_path = result_dir / "reveal.json"
    reveal = json.loads(reveal_path.read_text(encoding="utf-8"))
    reveal["scenario_seeds"][0] = 999
    reveal_path.write_text(
        json.dumps(reveal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_manifest_hash(result_dir, "reveal.json")

    def edit(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["case"] == "blind_00":
                row["scenario_seed"] = "999"

    _edit_csv(result_dir, edit)

    with pytest.raises(ValueError, match="scenario seeds do not follow"):
        audit_published_results(PROTOCOL_PATH, result_dir)


def test_audit_rejects_replacing_the_saved_beacon_evidence(tmp_path: Path) -> None:
    result_dir = _copy_result(tmp_path)
    reveal_path = result_dir / "reveal.json"
    reveal = json.loads(reveal_path.read_text(encoding="utf-8"))
    forged_signature = "00" * 96
    forged_randomness = hashlib.sha256(bytes.fromhex(forged_signature)).hexdigest()
    verifier = reveal["beacon_verification_audit"]["verifier_output"]
    for response in verifier["responses"]:
        response["beacon"]["signature"] = forged_signature
        response["beacon"]["randomness"] = forged_randomness
    reveal["beacon"]["signature"] = forged_signature
    reveal["beacon"]["randomness"] = forged_randomness
    reveal_path.write_text(
        json.dumps(reveal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_manifest_hash(result_dir, "reveal.json")

    with pytest.raises(ValueError, match="not the published v0.5 value"):
        audit_published_results(PROTOCOL_PATH, result_dir)


def test_audit_rejects_a_residual_policy_hash_changed_in_csv(tmp_path: Path) -> None:
    result_dir = _copy_result(tmp_path)

    def edit(rows: list[dict[str, str]]) -> None:
        row = next(row for row in rows if row["method"] == "torque_residual_run_00")
        row["policy_sha256"] = "f" * 64

    _edit_csv(result_dir, edit)

    with pytest.raises(ValueError, match="policy hash mismatch: torque_residual_run_00"):
        audit_published_results(PROTOCOL_PATH, result_dir)


def test_audit_recomputes_each_rows_gate_result(tmp_path: Path) -> None:
    result_dir = _copy_result(tmp_path)

    def edit(rows: list[dict[str, str]]) -> None:
        rows[0]["gate_pass"] = "yes"
        rows[0]["failed_checks"] = ""

    _edit_csv(result_dir, edit)

    with pytest.raises(ValueError, match="stored gate result mismatch"):
        audit_published_results(PROTOCOL_PATH, result_dir)


def test_audit_recomputes_summary_pass_counts(tmp_path: Path) -> None:
    result_dir = _copy_result(tmp_path)
    summary_path = result_dir / "summary.md"
    summary = summary_path.read_text(encoding="utf-8").replace(
        "| fixed_hybrid | 17/48 |",
        "| fixed_hybrid | 18/48 |",
    )
    summary_path.write_text(summary, encoding="utf-8")
    _refresh_manifest_hash(result_dir, "summary.md")

    with pytest.raises(ValueError, match="summary pass count mismatch: fixed_hybrid"):
        audit_published_results(PROTOCOL_PATH, result_dir)


def test_audit_rejects_a_refreshed_manifest_for_otherwise_valid_tamper(
    tmp_path: Path,
) -> None:
    result_dir = _copy_result(tmp_path)

    def edit(rows: list[dict[str, str]]) -> None:
        rows[0]["controller_p95_us"] = "0"

    _edit_csv(result_dir, edit)

    with pytest.raises(ValueError, match="published v0.5 manifest checksum mismatch"):
        audit_published_results(PROTOCOL_PATH, result_dir)


def test_audit_requires_the_published_completion_marker(tmp_path: Path) -> None:
    result_dir = _copy_result(tmp_path)
    (result_dir / "COMPLETE").write_text("incomplete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="completion marker mismatch"):
        audit_published_results(PROTOCOL_PATH, result_dir)
