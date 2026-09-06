"""Check compact claims, partition provenance and the three included raw episodes.

This is not a full 84-run physics replay: the compact archive deliberately omits
those large traces. Reproduction commands create and audit the complete corpus.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.surface_dataset import audit_dataset
from compliant_control_lab.surface_readiness_benchmark import audit_trace
from compliant_control_lab.surface_splits import split_cases

ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "results/franka_surface_learning_preparation"
EXAMPLES = ROOT / "results/franka_surface_learning_examples"
PARTITION = ROOT / "results/franka_surface_learning_partition"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compact_archive_integrity_and_claimed_scope():
    manifest = read(EVIDENCE / "manifest.json")
    assert (EVIDENCE / "COMPLETE").read_text().strip() == sha(EVIDENCE / "manifest.json")
    assert manifest["full_archive"] is False and manifest["new_holdout"] is False
    for filename, digest in manifest["artifacts_sha256"].items():
        assert sha(EVIDENCE / filename) == digest
    checks = read(EVIDENCE / "checks.json")
    assert checks["benchmark_audit"]["run_count"] == 72
    assert checks["stress_audit"]["run_count"] == 12
    assert checks["teacher_collection_equivalence"]["cases"] == 24
    assert checks["dataset_audit"]["physics_steps"] == 144000
    assert checks["dataset_audit"]["transitions"] == 14400
    assert not checks["compact_archive_contains_raw_npz_or_events"]


def test_tracking_numbers_include_all_methods_and_noise_replicas():
    report = read(EVIDENCE / "benchmark_report.json")
    assert len(report["runs"]) == 72 and report["acceptance_met"]
    expected = {
        "zero_adaptive": (11.8388, 4),
        "zero_friction": (3.46597, 24),
        "friction_teacher_50hz": (3.60601, 24),
    }
    for method, (tangent, passed) in expected.items():
        rows = [row for row in report["runs"] if row["method"] == method]
        assert len(rows) == 24 and all(row["hard_safe_completion"] for row in rows)
        assert sum(row["engineering_targets_met"] for row in rows) == passed
        assert set(Counter(row["group_id"] for row in rows).values()) == {2}
        assert min(row["contact_ratio_pct"] for row in rows) == 100
        assert np.median([row["tangent_rmse_mm"] for row in rows]) == pytest.approx(
            tangent, abs=5e-5
        )
    stress = read(EVIDENCE / "stress_report.json")
    assert len(stress["runs"]) == 12 and stress["acceptance_met"]
    assert all(row["all_actions_legal"] and row["complete_execution_log"] for row in stress["runs"])


def test_repartition_preserves_parent_evidence_and_separates_configuration_types():
    partition = read(PARTITION / "partition.json")
    assert partition["selection_uses_performance"] is False and not partition["new_holdout"]
    assert [item["seed"] for item in partition["candidates"]] == [20260906, 20260907, 20260908]
    for kind in ("benchmark", "stress", "dataset"):
        parent = PARTITION / f"v1_{kind}_manifest.json"
        child = EVIDENCE / (
            "dataset_manifest.json" if kind == "dataset" else f"{kind}_manifest.json"
        )
        assert sha(parent) == partition["parent_manifest_sha256"][kind]
        old, new = read(parent), read(child)
        assert old["source_and_assets_sha256"] == new["source_and_assets_sha256"]
        assert new["partition_provenance"]["parent_manifest_sha256"][kind] == sha(parent)
        assert [{k: v for k, v in case.items() if k != "split"} for case in old["cases"]] == [
            {k: v for k, v in case.items() if k != "split"} for case in new["cases"]
        ]
        if kind == "dataset":
            assert [item["sha256"] for item in old["episodes"]] == [
                item["sha256"] for item in new["episodes"]
            ]
        else:
            for name, digest in old["artifact_sha256"].items():
                if name.endswith(".npz") or (name.startswith("case_") and name.endswith(".json")):
                    assert new["artifact_sha256"][name] == digest
    manifest = read(EVIDENCE / "dataset_manifest.json")
    cases = manifest["cases"]
    assert split_cases(cases, **manifest["group_split"]) == cases
    groups = {case["group_id"]: case for case in cases}
    assert Counter(case["split"] for case in groups.values()) == {
        "train": 8,
        "validation": 2,
        "development_test": 2,
    }
    anchors = [
        case
        for case in groups.values()
        if case["scenario"]["name"] in {"preparation_000", "preparation_001", "preparation_002"}
    ]
    assert Counter(case["split"] for case in anchors) == {
        "train": 1,
        "validation": 1,
        "development_test": 1,
    }


def test_included_episodes_are_a_small_disclosed_subset_with_recomputed_metrics():
    audit = audit_dataset(EXAMPLES)
    assert audit["corpus_scope"] == "example_subset" and audit["episodes"] == 3
    assert audit["planned_episodes"] == 24 and audit["physics_steps"] == 18000
    manifest = read(EXAMPLES / "manifest.json")
    assert manifest["parent_manifest_sha256"] == sha(EVIDENCE / "dataset_manifest.json")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    rows = {
        row["case_id"]: row
        for row in read(EVIDENCE / "benchmark_report.json")["runs"]
        if row["method"] == "friction_teacher_50hz"
    }
    assert {entry["case_id"] for entry in manifest["episodes"]} == {
        "g000_seed11",
        "g005_seed11",
        "g007_seed11",
    }
    for entry in manifest["episodes"]:
        with np.load(EXAMPLES / entry["file"], allow_pickle=False) as stored:
            trace = {key[7:]: stored[key] for key in stored.files if key.startswith("trace__")}
        trace["intervention_active"] = np.array(
            [
                bool(set(str(reason).split("|")) - {"", "contact_lost", "contact_not_ready"})
                for reason in trace["residual_reasons"]
            ]
        )
        for name, value in audit_trace(trace, cases[entry["case_id"]]).items():
            assert rows[entry["case_id"]][name] == pytest.approx(value)
