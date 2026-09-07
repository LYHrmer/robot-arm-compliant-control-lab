"""Selection, pairing and resume guards do not require optional PyTorch."""

from copy import deepcopy

import pytest

from tools import surface_learning_pilot as pilot

IDS = ["val_a", "val_b", "val_c", "val_d"]


def _report(tangent=3.0, force=0.2):
    return {
        "evaluation_split": "validation",
        "evaluation_case_ids": IDS,
        "expected_case_count": 4,
        "acceptance_met": True,
        "all_metrics_available": True,
        "all_episodes_succeeded": True,
        "runs": [
            {
                "case_id": identifier,
                "group_id": identifier,
                "split": "validation",
                "simulation_seed": 123,
                "episode_success": True,
                "independent_physical_gates_met": True,
                "tangent_rmse_mm": tangent,
                "force_rmse_n": force,
                "contact_ratio_pct": 100.0,
                "peak_force_n": 12.0,
                "max_penetration_mm": 0.9,
                "max_speed_m_s": 0.08,
                "saturation_pct": 0.0,
                "max_contact_loss_s": 0.0,
                "episode_return": 11.9,
                "intervention_pct": 0.0,
            }
            for identifier in IDS
        ],
    }


def _choose(reports):
    return pilot.select_ppo_checkpoint(
        [{"episode": episode, "report": report} for episode, report in reports],
        expected_case_ids=IDS,
    )


def test_selection_uses_tracking_then_force_then_earlier_checkpoint_not_return():
    assert _choose([(0, _report(4)), (16, _report(3)), (32, _report(3))])["episode"] == 16
    assert _choose([(0, _report(3, 0.1)), (16, _report(3, 0.2))])["episode"] == 0
    assert _choose([(0, _report(3)), (32, _report(3))])["episode"] == 0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "split", "count"])
def test_selection_rejects_incomplete_or_nonvalidation_reports(mutation):
    report = _report()
    if mutation == "missing":
        report["runs"].pop()
    elif mutation == "duplicate":
        report["runs"][1] = report["runs"][0]
    elif mutation == "extra":
        report["runs"].append(report["runs"][0])
    elif mutation == "split":
        report["evaluation_split"] = "development_test"
    else:
        report["expected_case_count"] = 3
    with pytest.raises(ValueError, match="complete frozen validation"):
        _choose([(0, report)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("tangent_rmse_mm", 11.0),
        ("force_rmse_n", None),
        ("contact_ratio_pct", float("nan")),
        ("peak_force_n", 35.01),
        ("max_contact_loss_s", 0.1),
        ("saturation_pct", 0.001),
        ("episode_success", False),
        ("independent_physical_gates_met", False),
    ],
)
def test_claimed_acceptance_cannot_hide_failure_or_nonfinite_metrics(field, value):
    bad = _report(0.001)
    bad["runs"][0][field] = value
    assert _choose([(0, _report()), (32, bad)])["episode"] == 0
    fallback = _choose([(0, bad)])
    assert fallback["episode"] == 0 and not fallback["eligible"]


def test_selection_requires_unique_checkpoints_and_zero():
    with pytest.raises(ValueError, match="include zero without duplicates"):
        _choose([(16, _report())])
    with pytest.raises(ValueError, match="include zero without duplicates"):
        _choose([(0, _report()), (0, _report())])


def _baseline():
    rows = []
    for row in _report()["runs"]:
        for method in ("zero_adaptive", "zero_friction", "friction_teacher_50hz"):
            rows.append(
                {
                    **row,
                    "method": method,
                    "hard_safe_completion": True,
                    "nominal_kind": "friction" if method == "zero_friction" else "adaptive",
                }
            )
    return {"runs": rows}


def test_baselines_are_paired_with_exact_simulator_seed_and_nominal():
    pairs = pilot.paired_baselines(_report(), _baseline())
    assert len(pairs) == 12
    assert all(pair["delta_candidate_minus_baseline"]["tangent_rmse_mm"] == 0 for pair in pairs)
    for key, value in (("simulation_seed", 321), ("nominal_kind", "wrong"), ("group_id", "wrong")):
        bad = _baseline()
        bad["runs"][0][key] = value
        with pytest.raises(ValueError, match="baseline pair"):
            pilot.paired_baselines(_report(), bad)


def test_baseline_duplicates_are_not_silently_overwritten():
    bad = _baseline()
    bad["runs"].append(bad["runs"][0])
    with pytest.raises(ValueError, match="duplicate baseline"):
        pilot.paired_baselines(_report(), bad)


def test_manifest_detects_modified_artifacts_and_refuses_parent_paths(tmp_path):
    (tmp_path / "candidate.json").write_text("original")
    manifest = {"artifact_sha256": {"candidate.json": pilot._sha256(tmp_path / "candidate.json")}}
    pilot._write_new(tmp_path / "manifest.json", manifest)
    (tmp_path / "COMPLETE").write_text(pilot._sha256(tmp_path / "manifest.json"))
    assert pilot._verified_manifest(tmp_path) == manifest
    (tmp_path / "candidate.json").write_text("modified")
    with pytest.raises(ValueError, match="artifact changed"):
        pilot._verified_manifest(tmp_path)


def test_bc_resume_rejects_seed_budget_or_data_mismatch(monkeypatch, tmp_path):
    manifest = {
        "schema": "surface_behavior_clone_run_v1",
        "hyperparameters": {"seed": 11, **pilot.BC_CONFIG},
        "dataset": {"manifest_sha256": "data"},
        "trainer_source_sha256": {"trainer": "code"},
        "development_test_used": False,
    }
    plan = {
        "bc": pilot.BC_CONFIG,
        "dataset_manifest_sha256": "data",
        "trainer_sha256": {"tools/train_surface_bc.py": "code"},
    }
    monkeypatch.setattr(pilot, "_verified_manifest", lambda _: manifest)
    assert pilot._training_manifest("bc", tmp_path, 11, plan) is manifest
    with pytest.raises(ValueError, match="frozen seed/data/budget/source"):
        pilot._training_manifest("bc", tmp_path, 29, plan)
    changed = deepcopy(plan)
    changed["bc"]["epochs"] = 1
    with pytest.raises(ValueError):
        pilot._training_manifest("bc", tmp_path, 11, changed)
    changed = deepcopy(plan)
    changed["dataset_manifest_sha256"] = "other"
    with pytest.raises(ValueError):
        pilot._training_manifest("bc", tmp_path, 11, changed)


def test_pilot_cases_change_nominal_without_mutating_frozen_cases():
    plan = {"cases": [{"case_id": "case", "nominal_kind": "adaptive"}]}
    assert pilot.pilot_cases(plan, "friction")[0]["nominal_kind"] == "friction"
    assert plan["cases"][0]["nominal_kind"] == "adaptive"


def test_completed_stage_never_reruns_development(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "load_plan", lambda *args: {})
    (tmp_path / "bc_summary.json").write_text("{}")
    with pytest.raises(ValueError, match="do not rerun development"):
        pilot.run_stage("bc", tmp_path, "sha", tmp_path, tmp_path)
