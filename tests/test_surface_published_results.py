"""Check the committed development report without rerunning its 96 simulations."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.surface_experiment import ARMS, METRICS, paired_deltas
from compliant_control_lab.surface_replay import replay_surface_trace
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    SurfaceTask,
    SurfaceTrialResult,
)

REPORT = Path(__file__).resolve().parents[1] / "results/franka_surface_development"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_rows():
    with (REPORT / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["case_index"] = int(row["case_index"])
        assert row["has_raw_contact"] in {"True", "False"}
        row["has_raw_contact"] = row["has_raw_contact"] == "True"
        for field in METRICS:
            row[field] = float(row[field]) if row[field] else None
    return rows


def test_published_manifest_artifact_bytes_and_complete_grid():
    manifest_path = REPORT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (REPORT / "COMPLETE").read_text().strip() == sha256(manifest_path)
    assert manifest["experiment_identity"] == "surface-task-development-v1"
    assert manifest["new_holdout"] is False
    assert manifest["is_subset"] is False
    assert manifest["selected_case_indices"] == list(range(24))
    assert manifest["full_trace_case_indices"] == [16]
    assert len(manifest["all_development_cases"]) == 24
    assert set(manifest["arms"]) == set(ARMS)
    expected = {"comparison.csv", "paired_deltas.csv", "summary.md", "representative_case_16.png"}
    expected.update(f"representative_case_16_{arm}.npz" for arm in ARMS)
    assert set(manifest["artifact_sha256"]) == expected
    for name, checksum in manifest["artifact_sha256"].items():
        assert sha256(REPORT / name) == checksum
    for values in manifest["replay_checks"].values():
        assert values["matches"] is True
        assert values["sample_count"] == 2250
    rows = comparison_rows()
    assert len(rows) == 96
    assert {(row["arm"], row["case_index"]) for row in rows} == {
        (arm, case) for arm in ARMS for case in range(24)
    }


@pytest.mark.parametrize("arm", ARMS)
def test_published_representative_inputs_replay_and_reproduce_metrics(arm):
    trace_path = REPORT / f"representative_case_16_{arm}.npz"
    replay = replay_surface_trace(trace_path)
    assert replay.matches
    assert replay.sample_count == 2250
    manifest = json.loads((REPORT / "manifest.json").read_text(encoding="utf-8"))
    case = manifest["all_development_cases"][16]
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    result = SurfaceTrialResult(
        trace,
        SurfaceScenario(**case["scenario"]),
        SurfaceSimulationConfig(**case["config"]),
        SurfaceTask(**case["task"]),
    )
    measured = result.metrics()
    published = next(
        row for row in comparison_rows() if row["arm"] == arm and row["case_index"] == 16
    )
    assert measured["has_raw_contact"] == published["has_raw_contact"]
    for field in METRICS:
        assert measured[field] == pytest.approx(published[field], rel=0, abs=1e-10)
    np.testing.assert_array_equal(
        trace["feedback_raw_wrench_world"][1:], trace["raw_wrench_world"][:-1]
    )


def test_published_pair_differences_match_complete_comparison():
    calculated = paired_deltas(comparison_rows())
    with (REPORT / "paired_deltas.csv").open(newline="", encoding="utf-8") as handle:
        published = list(csv.DictReader(handle))
    assert len(calculated) == len(published) == 72
    for actual, row in zip(calculated, published):
        assert (actual["arm"], actual["case_index"]) == (row["arm"], int(row["case_index"]))
        assert str(actual["contact_in_both_arms"]) == row["contact_in_both_arms"]
        for field in METRICS:
            if actual[field] is None:
                assert row[field] == ""
            else:
                assert actual[field] == pytest.approx(float(row[field]), rel=0, abs=1e-12)
