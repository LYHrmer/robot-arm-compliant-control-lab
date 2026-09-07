import json
from copy import deepcopy
from pathlib import Path

import pytest

from tools import publish_surface_bc_transfer as publication
from tools import surface_bc_transfer as transfer
from tools import surface_learning_pilot as pilot


def _plan():
    return {
        "arms": list(transfer.ARMS),
        "seeds": [11, 29, 47],
        "architecture": [49, 32, 32, 3],
        "bc": {
            "epochs": 2,
            "batch_size": 256,
            "learning_rate": 0.001,
            "overfit_steps": 300,
        },
        "cases": [
            {"case_id": f"{split}{index}", "split": split}
            for split in ("validation", "development_test")
            for index in range(2)
        ],
        "thresholds": pilot.THRESHOLDS,
        "dataset_manifest_sha256": "dataset",
        "transfer_trainer_sources": {"transfer_trainer": "new", "trainer": "old"},
        "training_runtime": {
            "torch_version": "test",
            "device": "cpu",
            "dtype": "float64",
            "torch_num_threads": 1,
        },
        "input_interventions": {
            "drop_previous_residual": {"zero_indices": [14, 15, 16]},
            "teacher_inputs": {"keep_indices": [0, 10, 11]},
        },
    }


def _report(split, tangent):
    case_ids = [f"{split}{index}" for index in range(2)]
    return {
        "evaluation_split": split,
        "evaluation_case_ids": case_ids,
        "expected_case_count": len(case_ids),
        "acceptance_met": True,
        "all_metrics_available": True,
        "all_episodes_succeeded": True,
        "runs": [
            {
                "case_id": case_id,
                "split": split,
                "episode_success": True,
                "independent_physical_gates_met": True,
                **{name: rule["value"] for name, rule in pilot.THRESHOLDS.items()},
                "tangent_rmse_mm": tangent,
                "force_rmse_n": 0.2,
                "max_contact_loss_s": 0.0,
            }
            for case_id in case_ids
        ],
    }


def _summary_files(tmp_path):
    plan = _plan()
    pilot._write_new(tmp_path / "plan.json", plan)
    candidates = []
    validation = []
    development = []
    for arm in plan["arms"]:
        for seed in plan["seeds"]:
            candidate = tmp_path / f"{arm}_seed{seed}" / "selected_model.json"
            candidate.parent.mkdir()
            candidate.write_text(f"{arm}:{seed}\n", encoding="utf-8")
            candidates.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "candidate": str(candidate.relative_to(tmp_path)),
                    "candidate_sha256": publication._sha256(candidate),
                }
            )
            tangent = 3.0 if arm == "drop_previous_residual" else 4.0
            validation.append({"arm": arm, "seed": seed, "report": _report("validation", tangent)})
            development.append(
                {"arm": arm, "seed": seed, "report": _report("development_test", tangent)}
            )
    selected = transfer.select_arm(validation, plan)
    record = {
        "plan_sha256": publication._sha256(tmp_path / "plan.json"),
        "candidates": candidates,
        "validation": validation,
        "selection": selected,
    }
    pilot._write_new(tmp_path / "selection.json", record)
    summary = {
        "schema": "surface_bc_transfer_summary_v1",
        "new_holdout": False,
        **record,
        "selection_sha256": publication._sha256(tmp_path / "selection.json"),
        "development_test": development,
    }
    return plan, summary


def _rewrite_selection(tmp_path, summary):
    record = {
        key: summary[key] for key in ("plan_sha256", "candidates", "validation", "selection")
    }
    (tmp_path / "selection.json").write_text(json.dumps(record), encoding="utf-8")
    summary["selection_sha256"] = publication._sha256(tmp_path / "selection.json")


def test_summary_requires_all_six_candidates_and_evaluation_cohorts(tmp_path):
    plan, summary = _summary_files(tmp_path)
    publication._validate_summary(summary, tmp_path / "selection.json", plan)
    readme = publication._readme(plan, summary, summary)
    assert "selected_arm=drop_previous_residual" in readme
    assert "6/6" not in readme
    summary["validation"].pop()
    _rewrite_selection(tmp_path, summary)
    with pytest.raises(ValueError, match="cohort omits an arm or seed"):
        publication._validate_summary(summary, tmp_path / "selection.json", plan)


def test_summary_recomputes_selection_instead_of_trusting_record(tmp_path):
    plan, summary = _summary_files(tmp_path)
    summary["selection"] = deepcopy(summary["selection"])
    summary["selection"]["selected_arm"] = "teacher_inputs"
    _rewrite_selection(tmp_path, summary)
    with pytest.raises(ValueError, match="selection differs"):
        publication._validate_summary(summary, tmp_path / "selection.json", plan)


def test_summary_candidate_hash_and_evaluation_report_are_bound(tmp_path):
    plan, summary = _summary_files(tmp_path)
    candidate = tmp_path / summary["candidates"][0]["candidate"]
    candidate.write_text("altered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate path or hash"):
        publication._validate_summary(summary, tmp_path / "selection.json", plan)

    report = deepcopy(summary["validation"][0]["report"])
    report["runs"][0]["tangent_rmse_mm"] += 1.0
    with pytest.raises(ValueError, match="summary and evaluation report differ"):
        publication._require_summary_report(
            summary, transfer.ARMS[0], 11, "validation", report
        )


def _training_copy(tmp_path, plan, arm="teacher_inputs", seed=11):
    root = tmp_path / f"{arm}_seed{seed}"
    root.mkdir()
    history = [
        {"epoch": 1, "validation_action_mse": 0.2},
        {"epoch": 2, "validation_action_mse": 0.1},
    ]
    files = {
        "report.json": {
            "arm": arm,
            "input_mask": publication._expected_mask(plan, arm),
            "selection": {"selected_epoch": 2},
        },
        "epoch_losses.json": history,
        "selected_model.json": {"layers": []},
    }
    for name, value in files.items():
        pilot._write_new(root / name, value)
    (root / "epoch_losses.csv").write_text("epoch,validation_action_mse\n", encoding="utf-8")
    manifest = {
        "schema": "surface_behavior_clone_transfer_run_v1",
        "arm": arm,
        "input_mask": publication._expected_mask(plan, arm),
        "hyperparameters": {"seed": seed, **plan["bc"]},
        "dataset": {"manifest_sha256": plan["dataset_manifest_sha256"]},
        "trainer_source_sha256": deepcopy(plan["transfer_trainer_sources"]),
        "runtime": deepcopy(plan["training_runtime"]),
        "development_test_used": False,
        "artifact_sha256": {
            name: publication._sha256(root / name)
            for name in (*files, "epoch_losses.csv")
        },
    }
    pilot._write_new(root / "source_manifest.json", manifest)
    (root / "source_COMPLETE").write_text(
        publication._sha256(root / "source_manifest.json"), encoding="utf-8"
    )
    return root


@pytest.mark.parametrize("alter", ["runtime", "budget", "mask", "epoch", "nonfinite"])
def test_training_manifest_audit_recomputes_frozen_identity_and_epoch_choice(tmp_path, alter):
    plan = _plan()
    root = _training_copy(tmp_path, plan)
    publication._audit_training_copy(tmp_path, plan, "teacher_inputs", 11)
    if alter in {"runtime", "budget", "mask"}:
        manifest = pilot._read(root / "source_manifest.json")
        if alter == "runtime":
            manifest["runtime"]["dtype"] = "float32"
        elif alter == "budget":
            manifest["hyperparameters"]["batch_size"] = 64
        else:
            manifest["input_mask"][1] = 1
        (root / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "source_COMPLETE").write_text(
            publication._sha256(root / "source_manifest.json"), encoding="utf-8"
        )
    else:
        history = pilot._read(root / "epoch_losses.json")
        if alter == "epoch":
            history[1]["epoch"] = 3
        else:
            history[1]["validation_action_mse"] = float("nan")
        (root / "epoch_losses.json").write_text(json.dumps(history), encoding="utf-8")
        manifest = pilot._read(root / "source_manifest.json")
        manifest["artifact_sha256"]["epoch_losses.json"] = publication._sha256(
            root / "epoch_losses.json"
        )
        (root / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "source_COMPLETE").write_text(
            publication._sha256(root / "source_manifest.json"), encoding="utf-8"
        )
    with pytest.raises(ValueError):
        publication._audit_training_copy(tmp_path, plan, "teacher_inputs", 11)


def test_root_manifest_audit_rejects_altered_distributed_file(tmp_path):
    (tmp_path / "README.md").write_text("original\n", encoding="utf-8")
    manifest = {"artifact_sha256": {"README.md": publication._sha256(tmp_path / "README.md")}}
    pilot._write_new(tmp_path / "manifest.json", manifest)
    (tmp_path / "COMPLETE").write_text(
        publication._sha256(tmp_path / "manifest.json"), encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("altered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        publication.audit(tmp_path)


def test_real_public_archive_audits_when_present():
    root = Path(__file__).resolve().parents[1]
    candidates = (
        root / "results" / "franka_surface_bc_transfer",
        root / "results" / "franka_surface_bc_transfer_public",
    )
    archive = next((path for path in candidates if path.exists()), None)
    if archive is None:
        pytest.skip("public transfer archive is generated after these unit tests")
    result = publication.audit(archive)
    assert result["hashes_match"] and result["selection_recomputed"]
    assert result["representative_physics_traces_recomputed"] == 2
