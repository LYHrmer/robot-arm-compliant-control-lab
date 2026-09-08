import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.online_compensation_regression as regression


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_frozen_archive_maps_three_methods_to_unchanged_public24_cases():
    rows, configurations, identity = regression._load_frozen_archive()
    expected = {
        (method, index) for method in regression.FROZEN_METHODS for index in range(24)
    }
    assert set(rows) == set(configurations) == expected
    assert identity["directory"] == "results/franka_tangential_development"
    cases = regression._fixed_cases(configurations, list(range(24)))
    assert [case["case_index"] for case in cases] == list(range(24))
    assert all(case["config"].duration == 4.5 for case in cases)
    assert all(case["config"].contact_model == "smooth" for case in cases)
    assert all(case["scenario"].wall_yaw_deg == case["task"].yaw_deg for case in cases)


@pytest.fixture
def fake_trials(monkeypatch):
    frozen, _, _ = regression._load_frozen_archive()
    calls = []

    def trial(frame, *, scenario, config, task, controller_kind):
        calls.append((frame, scenario, config, task, controller_kind))
        method = {kind: name for name, kind in regression.METHODS.items()}[controller_kind]
        index = int(scenario.name.rsplit("_", 1)[1])
        reference = frozen[("friction" if method == "online" else method), index]
        metrics = {
            name: float(reference[name]) if reference[name] else None
            for name in regression.METRICS
        }
        metrics["has_raw_contact"] = reference["has_raw_contact"] == "True"
        if method == "online":
            metrics["tangent_rmse_mm"] -= 0.1
        return SimpleNamespace(metrics=lambda: metrics, trace={"method": method, "index": index})

    def auxiliary(result, _case):
        method, index = result.trace["method"], result.trace["index"]
        if method == "online":
            method = "friction"
        reference = frozen[method, index]
        return {name: float(reference[name]) for name in regression.AUXILIARY_METRICS}

    monkeypatch.setattr(regression, "run_surface_trial", trial)
    monkeypatch.setattr(regression, "_auxiliary_metrics", auxiliary)
    monkeypatch.setattr(regression, "_source_identity", lambda: {"source.py": "a" * 64})
    return calls


def test_subset_writes_atomic_evidence_and_exactly_reproduces_frozen_methods(
    tmp_path, fake_trials
):
    output = regression.generate_online_compensation_regression(
        tmp_path / "report", case_indices=[0]
    )
    assert [call[-1] for call in fake_trials] == list(regression.METHODS.values())
    for call in fake_trials:
        np.testing.assert_array_equal(call[0].rotation, fake_trials[0][0].rotation)
        assert call[1:4] == fake_trials[0][1:4]

    rows = _rows(output / "comparison.csv")
    pairs = _rows(output / "paired_online_vs_friction.csv")
    assert len(rows) == 4 and len(pairs) == 1
    assert all(float(row["frozen_metric_max_abs_error"]) == 0 for row in rows[:3])
    assert rows[-1]["online_gate_pass"] == "yes"
    assert float(pairs[0]["tangent_rmse_mm"]) == pytest.approx(-0.1)

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["new_holdout"] is False and summary["is_subset"] is True
    assert summary["online_outcome"]["all_cases_pass"] is True
    assert summary["representative_trace"]["retained"] is False
    assert not list(output.glob("*.npz"))

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frozen_archive"]["comparison_sha256"]
    for name, digest in manifest["artifact_sha256"].items():
        assert regression._sha256(output / name) == digest
    assert (output / "COMPLETE").read_text().strip() == regression._sha256(
        output / "manifest.json"
    )


def test_changed_development_case_is_rejected_before_trials(tmp_path, fake_trials, monkeypatch):
    cases = regression.development_cases()
    cases[0]["scenario"] = replace(cases[0]["scenario"], position_noise_std_m=0.0)
    monkeypatch.setattr(regression, "development_cases", lambda: cases)
    with pytest.raises(ValueError, match="development case differs"):
        regression.generate_online_compensation_regression(
            tmp_path / "bad", case_indices=[0]
        )
    assert not fake_trials and not (tmp_path / "bad").exists()


def test_frozen_metric_drift_above_tolerance_stops_before_new_method(
    tmp_path, fake_trials, monkeypatch
):
    original = regression.run_surface_trial

    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        supplied = result.metrics()
        supplied["force_rmse_n"] += 2e-10
        result.metrics = lambda: supplied
        return result

    monkeypatch.setattr(regression, "run_surface_trial", drift)
    with pytest.raises(ValueError, match="frozen metric mismatch"):
        regression.generate_online_compensation_regression(
            tmp_path / "bad", case_indices=[0]
        )
    assert len(fake_trials) == 1 and not (tmp_path / "bad").exists()


def test_online_gate_boundaries_pass_and_all_failures_are_retained():
    friction = {"force_rmse_n": 1.0, "orientation_rmse_deg": 2.0}
    passing = {
        "has_raw_contact": True,
        "contact_ratio_pct": 99.0,
        "peak_force_n": 35.0,
        "saturation_pct": 0.0,
        "force_rmse_n": 1.2,
        "orientation_rmse_deg": 2.2,
    }
    assert regression.online_failures(passing, friction) == ()
    failing = {
        **passing,
        "has_raw_contact": False,
        "contact_ratio_pct": 98.9,
        "peak_force_n": 35.1,
        "saturation_pct": 0.01,
        "force_rmse_n": 1.200001,
        "orientation_rmse_deg": 2.200001,
    }
    assert regression.online_failures(failing, friction) == (
        "contact_ratio",
        "raw_peak_force",
        "saturation",
        "paired_force_rmse",
        "paired_orientation_rmse",
    )


def test_existing_output_is_refused_without_changing_it(tmp_path, fake_trials):
    output = tmp_path / "occupied"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_text("user data", encoding="utf-8")
    with pytest.raises((FileExistsError, ValueError)):
        regression.generate_online_compensation_regression(output, case_indices=[0])
    assert keep.read_text(encoding="utf-8") == "user data"
    assert not fake_trials


def test_complete_mismatch_is_rejected(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "manifest.json").write_text("{}\n", encoding="utf-8")
    (archive / "COMPLETE").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="COMPLETE"):
        regression._load_frozen_archive(archive)
