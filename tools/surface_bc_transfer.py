"""Controlled BC input ablations; preserve the first pilot and all evaluation gates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_experiment import _sha256
from compliant_control_lab.surface_policy import friction_teacher_action
from tools import surface_learning_pilot as pilot

ARMS = ("drop_previous_residual", "teacher_inputs")
SOURCE_FILES = (
    "tools/train_surface_bc_transfer.py",
    "tools/train_surface_bc.py",
    "tools/surface_bc_transfer.py",
)


def _sources():
    root = Path(__file__).resolve().parents[1]
    return {name: _sha256(root / name) for name in SOURCE_FILES}


def _parent_records(parent, parent_plan):
    parent = Path(parent)
    summary = pilot._read(parent / "bc_summary.json")
    if (parent / "bc_COMPLETE").read_text().strip() != _sha256(parent / "bc_summary.json"):
        raise ValueError("parent BC stage is incomplete")
    if summary["plan_sha256"] != _sha256(parent / "plan.json"):
        raise ValueError("parent BC summary belongs to another plan")
    seeds = parent_plan["seeds"]
    for field in ("validation", "development_test", "selections"):
        if [item["seed"] for item in summary[field]] != seeds:
            raise ValueError("parent BC cohort is incomplete")
    if summary["selection_sha256"] != _sha256(parent / "bc_selection.json"):
        raise ValueError("parent selection changed")
    files = ["bc_summary.json", "bc_selection.json", "bc_COMPLETE"]
    for seed in seeds:
        directory = parent / f"bc_seed{seed}"
        pilot._training_manifest("bc", directory, seed, parent_plan)
        files.extend(
            f"bc_seed{seed}/{name}"
            for name in ("manifest.json", "COMPLETE", "report.json", "selected_model.json")
        )
    return {name: _sha256(parent / name) for name in files}


def freeze(output, parent, dataset, benchmark, expected_parent_sha256):
    from tools.train_surface_bc_transfer import _source_identity

    output = Path(output)
    if output.exists():
        raise ValueError("transfer output must be new")
    parent_plan = pilot.load_plan(parent, expected_parent_sha256, dataset, benchmark)
    plan = {
        "schema": "surface_bc_transfer_plan_v1",
        "parent_plan_sha256": expected_parent_sha256,
        "parent_bc_records_sha256": _parent_records(parent, parent_plan),
        "source_sha256": _sources(),
        "transfer_trainer_sources": _source_identity(),
        "training_runtime": {
            key: parent_plan[key]
            for key in ("torch_version", "device", "dtype", "torch_num_threads")
        },
        "new_holdout": False,
        "scope": "public-development input ablation; previous development results are known",
        "arms": list(ARMS),
        "seeds": parent_plan["seeds"],
        "bc": parent_plan["bc"],
        "architecture": parent_plan["architecture"],
        "cases": parent_plan["cases"],
        "metrics": parent_plan["metrics"],
        "thresholds": parent_plan["thresholds"],
        "dataset_manifest_sha256": parent_plan["dataset_manifest_sha256"],
        "input_interventions": {
            "drop_previous_residual": {"zero_indices": [14, 15, 16]},
            "teacher_inputs": {"keep_indices": [0, 10, 11]},
        },
        "training_data": "original train only; no student-state augmentation in this comparison",
        "control": "reuse frozen full49 pilot with identical seeds, data, optimizer and budget",
        "checkpoint_selection": "minimum full offline validation MSE, exact ties earlier epoch",
        "arm_selection": {
            "eligibility": "all three seeds pass all four validation cases and original gates",
            "ordering": ["pooled mean tangent_rmse_mm", "pooled mean force_rmse_n", "arm name"],
            "if_none": "no accepted BC arm; continue BC diagnosis before new RL training",
            "seed_selection": "none; report all three seeds",
        },
        "development_policy": "freeze all six candidates and arm selection before any new development",
        "next_stage": "RL remains separate and requires a subsequently frozen protocol",
    }
    output.mkdir(parents=True)
    pilot._write_new(output / "plan.json", plan)
    digest = _sha256(output / "plan.json")
    with (output / "PLAN_SHA256").open("x") as stream:
        stream.write(digest + "\n")
    return digest


def load(output, parent, dataset, benchmark, expected_sha256):
    from tools.train_surface_bc_transfer import _source_identity

    output = Path(output)
    if _sha256(output / "plan.json") != expected_sha256:
        raise ValueError("transfer plan differs from the pre-fit SHA")
    if (output / "PLAN_SHA256").read_text().strip() != expected_sha256:
        raise ValueError("transfer plan marker differs")
    plan = pilot._read(output / "plan.json")
    parent_plan = pilot.load_plan(parent, plan["parent_plan_sha256"], dataset, benchmark)
    if plan["source_sha256"] != _sources():
        raise ValueError("transfer source changed after freezing")
    if plan["transfer_trainer_sources"] != _source_identity():
        raise ValueError("inherited transfer implementation changed")
    if any(parent_plan[key] != value for key, value in plan["training_runtime"].items()):
        raise ValueError("transfer training runtime differs from parent")
    if plan["parent_bc_records_sha256"] != _parent_records(parent, parent_plan):
        raise ValueError("parent control changed")
    for key in (
        "seeds",
        "bc",
        "architecture",
        "cases",
        "metrics",
        "thresholds",
        "dataset_manifest_sha256",
    ):
        if plan[key] != parent_plan[key]:
            raise ValueError(f"transfer changed the matched control: {key}")
    if plan["arms"] != list(ARMS):
        raise ValueError("transfer arm cohort differs")
    return plan


def _training_manifest(output, plan, arm, seed):
    from tools.train_surface_bc_transfer import _input_mask

    directory = Path(output) / f"{arm}_seed{seed}"
    manifest = pilot._verified_manifest(directory)
    parameters = manifest["hyperparameters"]
    if (
        manifest["schema"] != "surface_behavior_clone_transfer_run_v1"
        or manifest["arm"] != arm
        or manifest["input_mask"] != _input_mask(arm).astype(int).tolist()
        or parameters["seed"] != seed
        or any(parameters[key] != value for key, value in plan["bc"].items())
        or manifest["dataset"]["manifest_sha256"] != plan["dataset_manifest_sha256"]
        or manifest["trainer_source_sha256"]["transfer_trainer"]
        != plan["source_sha256"]["tools/train_surface_bc_transfer.py"]
        or manifest["trainer_source_sha256"] != plan["transfer_trainer_sources"]
        or any(
            manifest["runtime"].get(key) != value for key, value in plan["training_runtime"].items()
        )
        or manifest["development_test_used"] is not False
    ):
        raise ValueError("transfer training differs from frozen inputs/arm/budget/source")
    report = pilot._read(directory / "report.json")
    history = pilot._read(directory / "epoch_losses.json")
    if len(history) != plan["bc"]["epochs"] or [row["epoch"] for row in history] != list(
        range(1, plan["bc"]["epochs"] + 1)
    ):
        raise ValueError("transfer training epoch cohort is incomplete")
    losses = [row["validation_action_mse"] for row in history]
    if not all(np.isfinite(value) for value in losses):
        raise ValueError("nonfinite transfer checkpoint selection")
    selected = min(history, key=lambda row: (row["validation_action_mse"], row["epoch"]))
    if report["selection"]["selected_epoch"] != selected["epoch"]:
        raise ValueError("transfer checkpoint selection differs from validation MSE")
    return manifest


def train_one(output, plan, dataset, arm, seed):
    from tools.train_surface_bc_transfer import train_bc_transfer

    if arm not in plan["arms"] or seed not in plan["seeds"]:
        raise ValueError("undeclared transfer arm or seed")
    directory = Path(output) / f"{arm}_seed{seed}"
    if not directory.exists():
        print(f"train {arm} seed={seed}", flush=True)
        train_bc_transfer(dataset, directory, arm=arm, seed=seed, **plan["bc"])
    _training_manifest(output, plan, arm, seed)
    return directory / "selected_model.json"


def validate_one(output, plan, arm, seed):
    if arm not in plan["arms"] or seed not in plan["seeds"]:
        raise ValueError("undeclared transfer arm or seed")
    _training_manifest(output, plan, arm, seed)
    candidate = Path(output) / f"{arm}_seed{seed}" / "selected_model.json"
    return pilot._evaluate(output, plan, candidate, f"{arm}_seed{seed}", "validation", "adaptive")


def select_arm(validation, plan):
    expected = [(arm, seed) for arm in plan["arms"] for seed in plan["seeds"]]
    if [(item["arm"], item["seed"]) for item in validation] != expected:
        raise ValueError("arm selection requires every declared seed and arm exactly once")
    case_ids = [case["case_id"] for case in plan["cases"] if case["split"] == "validation"]
    choices, details = [], []
    for arm in plan["arms"]:
        cohort = [item for item in validation if item["arm"] == arm]
        gates = [
            pilot._selection_eligible(item["report"], case_ids, plan["thresholds"])
            for item in cohort
        ]
        rows = [row for item in cohort for row in item["report"]["runs"]]
        eligible = all(gates)
        ordering = None
        if eligible:
            ordering = [
                float(np.mean([row[key] for row in rows]))
                for key in ("tangent_rmse_mm", "force_rmse_n")
            ]
            choices.append((*ordering, arm))
        details.append(
            {"arm": arm, "eligible": eligible, "seed_gates": gates, "ordering": ordering}
        )
    return {
        "selected_arm": min(choices)[-1] if choices else None,
        "eligible": bool(choices),
        "arms": details,
        "selection_split": "validation",
        "seed_selection": False,
    }


def _student_errors(output, validation):
    rows = []
    for item in validation:
        directory = Path(output) / "evaluations" / f"{item['arm']}_seed{item['seed']}_validation"
        for run in item["report"]["runs"]:
            with np.load(directory / run["trace_file"], allow_pickle=False) as trace:
                x, actions = trace["decision_observation"], trace["decision_action"]
                teacher = np.asarray([friction_teacher_action(obs[:33]) for obs in x])
                mse = float(np.mean((actions - teacher) ** 2)) if len(actions) else None
            rows.append(
                {
                    "arm": item["arm"],
                    "seed": item["seed"],
                    "case_id": run["case_id"],
                    "split": "validation",
                    "action_mse": mse,
                }
            )
    return {"use": "post-fit diagnostic; not training or checkpoint/arm selection", "runs": rows}


def run(output, parent, dataset, benchmark, expected_sha256):
    output = Path(output)
    plan = load(output, parent, dataset, benchmark, expected_sha256)
    if (output / "summary.json").exists():
        raise ValueError("transfer already completed; do not rerun development")
    candidates, validation = [], []
    for arm in plan["arms"]:
        for seed in plan["seeds"]:
            candidate = train_one(output, plan, dataset, arm, seed)
            report = pilot._evaluate(
                output, plan, candidate, f"{arm}_seed{seed}", "validation", "adaptive"
            )
            validation.append({"arm": arm, "seed": seed, "report": report})
            candidates.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "candidate": str(candidate.relative_to(output)),
                    "candidate_sha256": _sha256(candidate),
                }
            )
    record = {
        "plan_sha256": expected_sha256,
        "candidates": candidates,
        "validation": validation,
        "selection": select_arm(validation, plan),
    }
    selection_path = output / "selection.json"
    if selection_path.exists():
        if pilot._read(selection_path) != record:
            raise ValueError("BC arm/candidate selection changed after freeze")
    else:
        pilot._write_new(selection_path, record)
    diagnostic_path = output / "student_state_errors.json"
    diagnostic = _student_errors(output, validation)
    if diagnostic_path.exists():
        if pilot._read(diagnostic_path) != diagnostic:
            raise ValueError("saved student-state diagnostic changed")
    else:
        pilot._write_new(diagnostic_path, diagnostic)
    development, baseline = [], pilot._read(Path(benchmark) / "report.json")
    for item in candidates:
        report = pilot._evaluate(
            output,
            plan,
            output / item["candidate"],
            f"{item['arm']}_seed{item['seed']}",
            "development_test",
            "adaptive",
        )
        development.append(
            {
                "arm": item["arm"],
                "seed": item["seed"],
                "report": report,
                "paired_baselines": pilot.paired_baselines(report, baseline),
            }
        )
    load(output, parent, dataset, benchmark, expected_sha256)
    summary = {
        "schema": "surface_bc_transfer_summary_v1",
        "new_holdout": False,
        "plan_sha256": expected_sha256,
        "selection_sha256": _sha256(selection_path),
        **record,
        "development_test": development,
    }
    pilot._write_new(output / "summary.json", summary)
    with (output / "COMPLETE").open("x") as stream:
        stream.write(_sha256(output / "summary.json") + "\n")
    print(f"BC transfer complete: {record['selection']}", flush=True)
    return output / "summary.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "train", "validate", "run"))
    for name in ("output", "parent", "dataset", "benchmark"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-parent-plan-sha256")
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.stage == "freeze":
        if not args.expected_parent_plan_sha256:
            parser.error("freeze requires the original parent plan SHA")
        print(
            freeze(
                args.output,
                args.parent,
                args.dataset,
                args.benchmark,
                args.expected_parent_plan_sha256,
            )
        )
        return
    if not args.expected_plan_sha256:
        parser.error("training requires the transfer plan SHA preserved before fits")
    if args.stage in ("train", "validate"):
        if args.arm is None or args.seed is None:
            parser.error("train/validate requires a declared --arm and --seed")
        plan = load(
            args.output, args.parent, args.dataset, args.benchmark, args.expected_plan_sha256
        )
        if args.stage == "train":
            train_one(args.output, plan, args.dataset, args.arm, args.seed)
        else:
            validate_one(args.output, plan, args.arm, args.seed)
        load(args.output, args.parent, args.dataset, args.benchmark, args.expected_plan_sha256)
    else:
        run(args.output, args.parent, args.dataset, args.benchmark, args.expected_plan_sha256)


if __name__ == "__main__":
    main()
