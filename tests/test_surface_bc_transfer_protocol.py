from copy import deepcopy
from pathlib import Path

import pytest

from tools import surface_bc_transfer as transfer
from tools import surface_learning_pilot as pilot


def _plan():
    return {
        "arms": list(transfer.ARMS),
        "seeds": [11, 29, 47],
        "cases": [
            {"case_id": f"{split}{i}", "split": split}
            for split in ("validation", "development_test")
            for i in range(4)
        ],
        "thresholds": pilot.THRESHOLDS,
    }


def _report(split="validation", tangent=3.0):
    identifiers = [f"{split}{i}" for i in range(4)]
    return {
        "evaluation_split": split,
        "evaluation_case_ids": identifiers,
        "expected_case_count": 4,
        "acceptance_met": True,
        "all_metrics_available": True,
        "all_episodes_succeeded": True,
        "runs": [
            {
                "case_id": name,
                "split": split,
                "episode_success": True,
                "independent_physical_gates_met": True,
                **{key: value["value"] for key, value in pilot.THRESHOLDS.items()},
                "tangent_rmse_mm": tangent,
                "force_rmse_n": 0.2,
                "max_contact_loss_s": 0.0,
            }
            for name in identifiers
        ],
    }


def _validation():
    return [
        {
            "arm": arm,
            "seed": seed,
            "report": _report(tangent=3.0 if arm == transfer.ARMS[0] else 2.0),
        }
        for arm in transfer.ARMS
        for seed in (11, 29, 47)
    ]


def test_selection_uses_entire_seed_cohort_and_fixed_validation_ordering():
    cohort = _validation()
    selected = transfer.select_arm(cohort, _plan())
    assert selected["selected_arm"] == "teacher_inputs"
    assert selected["seed_selection"] is False
    cohort[-1]["report"]["runs"][0]["tangent_rmse_mm"] = 11.0
    # One failed seed rejects the whole arm, even if its average is still better.
    assert transfer.select_arm(cohort, _plan())["selected_arm"] == "drop_previous_residual"


@pytest.mark.parametrize("alter", ["missing_seed", "duplicate_seed", "wrong_split", "missing_case"])
def test_selection_rejects_missing_or_wrong_evidence(alter):
    cohort = _validation()
    if alter == "missing_seed":
        cohort.pop()
    elif alter == "duplicate_seed":
        cohort[-1] = deepcopy(cohort[-2])
    elif alter == "wrong_split":
        cohort[0]["report"]["evaluation_split"] = "development_test"
    else:
        cohort[0]["report"]["runs"].pop()
    with pytest.raises(ValueError):
        transfer.select_arm(cohort, _plan())


@pytest.mark.parametrize(
    "alter", ["failed_episode", "missing_metric", "nonfinite", "contact_timeout"]
)
def test_no_eligible_arm_does_not_silently_promote_failed_bc(alter):
    cohort = _validation()
    for item in cohort:
        row = item["report"]["runs"][0]
        if alter == "failed_episode":
            row["episode_success"] = False
        elif alter == "missing_metric":
            row.pop("force_rmse_n")
        elif alter == "nonfinite":
            row["force_rmse_n"] = float("nan")
        else:
            row["max_contact_loss_s"] = 0.1
    selected = transfer.select_arm(cohort, _plan())
    assert selected["selected_arm"] is None
    assert selected["eligible"] is False


def test_run_freezes_every_candidate_before_any_development_and_refuses_repeat(
    tmp_path, monkeypatch
):
    plan = _plan()
    monkeypatch.setattr(transfer, "load", lambda *args: plan)
    monkeypatch.setattr(transfer, "_student_errors", lambda *args: {"runs": []})
    monkeypatch.setattr(pilot, "paired_baselines", lambda *args: [])
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    pilot._write_new(benchmark / "report.json", {"runs": []})
    output = tmp_path / "output"
    output.mkdir()
    observed = []

    def train(root, plan, dataset, arm, seed):
        path = Path(root) / f"{arm}_seed{seed}" / "selected_model.json"
        path.parent.mkdir()
        path.write_text(f"{arm} {seed}\n")
        return path

    def evaluate(root, plan, candidate, name, split, nominal):
        assert nominal == "adaptive"
        if split == "development_test":
            record = pilot._read(Path(root) / "selection.json")
            assert len(record["candidates"]) == 6
            assert len(record["validation"]) == 6
            assert observed.count("validation") == 6
            assert record["selection"]["seed_selection"] is False
        observed.append(split)
        return _report(split)

    monkeypatch.setattr(transfer, "train_one", train)
    monkeypatch.setattr(pilot, "_evaluate", evaluate)
    transfer.run(output, "unused", "unused", benchmark, "prefit-sha")
    assert observed == ["validation"] * 6 + ["development_test"] * 6
    assert (output / "COMPLETE").read_text().strip() == pilot._sha256(output / "summary.json")
    with pytest.raises(ValueError, match="already completed"):
        transfer.run(output, "unused", "unused", benchmark, "prefit-sha")


def test_changed_frozen_selection_prevents_development(tmp_path, monkeypatch):
    plan = _plan()
    monkeypatch.setattr(transfer, "load", lambda *args: plan)
    candidate = tmp_path / "candidate.json"
    candidate.write_text("candidate")
    monkeypatch.setattr(transfer, "train_one", lambda *args: candidate)
    output = tmp_path
    pilot._write_new(output / "selection.json", {"plan_sha256": "different"})
    observed = []

    def evaluate(root, plan, candidate, name, split, nominal):
        observed.append(split)
        assert split == "validation"
        return _report()

    monkeypatch.setattr(pilot, "_evaluate", evaluate)
    with pytest.raises(ValueError, match="changed after freeze"):
        transfer.run(output, "unused", "unused", "unused", "prefit-sha")
    assert observed == ["validation"] * 6


@pytest.mark.parametrize(
    "change", ["fitter", "torch_version", "dtype", "device", "torch_num_threads"]
)
def test_resumed_training_rejects_wrong_inherited_fitter_or_runtime(tmp_path, monkeypatch, change):
    from tools.train_surface_bc_transfer import _input_mask

    plan = _plan() | {
        "bc": {"epochs": 1},
        "dataset_manifest_sha256": "dataset",
        "source_sha256": {"tools/train_surface_bc_transfer.py": "transfer"},
        "transfer_trainer_sources": {"transfer_trainer": "transfer", "trainer": "original"},
        "training_runtime": {
            "torch_version": "frozen",
            "device": "cpu",
            "dtype": "float64",
            "torch_num_threads": 1,
        },
    }
    manifest = {
        "schema": "surface_behavior_clone_transfer_run_v1",
        "arm": "teacher_inputs",
        "input_mask": _input_mask("teacher_inputs").astype(int).tolist(),
        "hyperparameters": {"seed": 11, "epochs": 1},
        "dataset": {"manifest_sha256": "dataset"},
        "trainer_source_sha256": deepcopy(plan["transfer_trainer_sources"]),
        "runtime": deepcopy(plan["training_runtime"]),
        "development_test_used": False,
    }
    if change == "fitter":
        manifest["trainer_source_sha256"]["trainer"] = "transient-other-fitter"
    else:
        manifest["runtime"][change] = "wrong-runtime"
    monkeypatch.setattr(pilot, "_verified_manifest", lambda *args: manifest)
    with pytest.raises(ValueError, match="differs from frozen"):
        transfer._training_manifest(tmp_path, plan, "teacher_inputs", 11)
