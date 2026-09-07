"""Freeze and run a small BC-then-PPO pilot without using development cases to select.

Training remains optional. The existing NumPy evaluator, controller and preparation
archives are reused unchanged. Hashes establish identity, not independent preregistration.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_experiment import _sha256, _source_hashes
from compliant_control_lab.surface_policy_artifact import (
    freeze_evaluation_protocol,
    policy_contract,
    verify_evaluation_protocol,
)
from compliant_control_lab.surface_readiness_benchmark import HARD_SAFETY_LIMITS, METRICS
from tools.evaluate_surface_candidate import evaluate_candidate
from tools.surface_mlp_actor import current_runner_identity

SEEDS = (11, 29, 47)
TRAINER_FILES = (
    "tools/train_surface_bc.py",
    "tools/train_surface_ppo.py",
    "tools/surface_learning_pilot.py",
)
BC_CONFIG = {"epochs": 100, "batch_size": 256, "learning_rate": 0.001, "overfit_steps": 300}
PPO_CONFIG = {
    "episodes": 32,
    "episodes_per_update": 4,
    "checkpoint_episodes": [0, 16, 32],
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "learning_rate": 0.0003,
    "update_epochs": 4,
    "batch_size": 128,
    "max_grad_norm": 0.5,
    "std": 0.05,
    "target_kl": 0.02,
}
THRESHOLDS = {
    "contact_ratio_pct": {"aggregation": "min", "operator": ">=", "value": 99.0},
    "tangent_rmse_mm": {"aggregation": "max", "operator": "<=", "value": 10.0},
    "force_rmse_n": {"aggregation": "max", "operator": "<=", "value": 2.0},
    **{
        name: {"aggregation": "max", "operator": "<=", "value": value}
        for name, value in HARD_SAFETY_LIMITS.items()
        if name != "contact_loss_timeout_s"
    },
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path, value):
    """Never replace a plan, selection record, or already completed stage."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _trainer_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: _sha256(root / name) for name in TRAINER_FILES}


def _verified_manifest(directory):
    directory = Path(directory)
    manifest = _read(directory / "manifest.json")
    if (directory / "COMPLETE").read_text().strip() != _sha256(directory / "manifest.json"):
        raise ValueError("COMPLETE does not bind manifest")
    artifacts = manifest.get("artifact_sha256", {})
    if not artifacts:
        artifacts = {item["file"]: item["sha256"] for item in manifest.get("files", [])}
    for name, digest in artifacts.items():
        target = directory / name
        if target.is_symlink() or not target.resolve().is_relative_to(directory.resolve()):
            raise ValueError("unsafe artifact path")
        if _sha256(target) != digest:
            raise ValueError(f"artifact changed: {name}")
    return manifest


def pilot_cases(plan, nominal_kind):
    cases = deepcopy(plan["cases"])
    for case in cases:
        case["nominal_kind"] = nominal_kind
    return cases


def freeze_plan(dataset, benchmark, output):
    """Freeze the already public v2 partition and every pilot choice before fitting."""
    dataset, benchmark, output = Path(dataset), Path(benchmark), Path(output)
    if output.exists():
        raise ValueError("pilot output must be new")
    data = _verified_manifest(dataset)
    baseline = _verified_manifest(benchmark)
    if data["source_and_assets_sha256"] != _source_hashes():
        raise ValueError("dataset source identity differs from current runtime")
    if baseline["source_and_assets_sha256"] != data["source_and_assets_sha256"]:
        raise ValueError("baseline and teacher data source identities differ")
    cases = data["cases"]
    if baseline["cases"] != cases:
        raise ValueError("baseline and teacher case plans differ")
    counts = {
        split: sum(case["split"] == split for case in cases)
        for split in ("train", "validation", "development_test")
    }
    if counts != {"train": 16, "validation": 4, "development_test": 4}:
        raise ValueError("pilot requires the complete 16/4/4 public v2 plan")
    reference_rows = _read(benchmark / "report.json")["runs"]
    expected_pairs = {
        (method, case["case_id"])
        for method in ("zero_adaptive", "zero_friction", "friction_teacher_50hz")
        for case in cases
    }
    if (
        len(reference_rows) != len(expected_pairs)
        or {(row["method"], row["case_id"]) for row in reference_rows} != expected_pairs
    ):
        raise ValueError("baseline report does not contain the complete paired 72-run matrix")
    episodes = {episode["case_id"]: episode for episode in data["episodes"]}
    if any(
        row["env_seed"] != episodes[row["case_id"]]["env_seed"]
        or row["simulation_seed"] != episodes[row["case_id"]]["simulation_seed"]
        for row in reference_rows
    ):
        raise ValueError("baseline and teacher simulator/reset seeds differ")
    contract = policy_contract(cases, purpose="il_friction_teacher", nominal_kind="adaptive")
    import torch

    plan = {
        "schema": "surface_learning_pilot_v1",
        "new_holdout": False,
        "scope": "small_fixed_budget_public_development_pilot_not_convergence_evidence",
        "seeds": list(SEEDS),
        "architecture": [49, 32, 32, 3],
        "dtype": "float64",
        "device": "cpu",
        "torch_num_threads": 1,
        "deterministic_algorithms": True,
        "torch_version": torch.__version__,
        "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
        "benchmark_manifest_sha256": _sha256(benchmark / "manifest.json"),
        "benchmark_report_sha256": _sha256(benchmark / "report.json"),
        "cases": cases,
        "split_counts": counts,
        "split_binding": contract["split_binding"],
        "trainer_sha256": _trainer_hashes(),
        "runner_identity": current_runner_identity(),
        "bc": BC_CONFIG,
        "ppo": PPO_CONFIG,
        "metrics": list(METRICS) + ["episode_return"],
        "thresholds": THRESHOLDS,
        "bc_selection": "minimum full validation action MSE; exact ties choose earlier epoch",
        "bc_to_ppo_gate": "all BC training and evaluation artifacts complete; no superiority gate",
        "ppo_selection": {
            "eligibility": "all four validation episodes succeed and pass every frozen gate",
            "ordering": ["mean tangent_rmse_mm", "mean force_rmse_n", "earlier episode"],
            "nonfinite": "ineligible",
            "if_none_eligible": "zero checkpoint, explicitly not accepted",
            "zero_checkpoint_eligible": True,
            "final_trained_checkpoint": "always retain and report its validation result",
        },
        "development_policy": "freeze every seed selection before any development evaluation",
        "bc_nominal": "adaptive",
        "ppo_nominal": "friction",
        "ppo_initialization": "fresh actor with zero final head; never BC weights",
        "references": ["https://arxiv.org/abs/1707.06347", "https://arxiv.org/abs/1506.02438"],
    }
    output.mkdir(parents=True)
    _write_new(output / "plan.json", plan)
    digest = _sha256(output / "plan.json")
    (output / "PLAN_SHA256").write_text(digest + "\n")
    return digest


def load_plan(output, expected_sha256, dataset, benchmark):
    output, dataset, benchmark = Path(output), Path(dataset), Path(benchmark)
    if _sha256(output / "plan.json") != expected_sha256:
        raise ValueError("pilot plan differs from externally supplied SHA")
    plan = _read(output / "plan.json")
    if (
        plan["trainer_sha256"] != _trainer_hashes()
        or plan["runner_identity"] != current_runner_identity()
    ):
        raise ValueError("pilot implementation changed since freeze")
    for path, digest in (
        (dataset / "manifest.json", plan["dataset_manifest_sha256"]),
        (benchmark / "manifest.json", plan["benchmark_manifest_sha256"]),
        (benchmark / "report.json", plan["benchmark_report_sha256"]),
    ):
        if _sha256(path) != digest:
            raise ValueError("pilot input identity changed")
    import torch

    if torch.__version__ != plan["torch_version"]:
        raise ValueError("training runtime differs from frozen plan")
    return plan


def _selection_eligible(report, expected_case_ids, thresholds):
    identifiers = sorted(expected_case_ids)
    if (
        report.get("evaluation_split") != "validation"
        or sorted(report.get("evaluation_case_ids", [])) != identifiers
        or report.get("expected_case_count") != len(identifiers)
        or sorted(row["case_id"] for row in report["runs"]) != identifiers
        or len(identifiers) != len(set(identifiers))
        or not identifiers
        or any(row["split"] != "validation" for row in report["runs"])
    ):
        raise ValueError("selection requires the complete frozen validation case set")
    if not all(
        report.get(key) is True
        for key in ("acceptance_met", "all_metrics_available", "all_episodes_succeeded")
    ):
        return False
    for row in report["runs"]:
        if not row["episode_success"] or not row["independent_physical_gates_met"]:
            return False
        values = [row.get(metric) for metric in (*thresholds, "max_contact_loss_s")]
        if not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)
            for x in values
        ):
            return False
        if row["max_contact_loss_s"] >= HARD_SAFETY_LIMITS["contact_loss_timeout_s"] - 1e-12:
            return False
        for metric, rule in thresholds.items():
            if (rule["operator"] == "<=" and row[metric] > rule["value"]) or (
                rule["operator"] == ">=" and row[metric] < rule["value"]
            ):
                return False
    return True


def select_ppo_checkpoint(evaluations, *, expected_case_ids, thresholds=THRESHOLDS):
    """Use validation only; failures/NaNs cannot win by disappearing from averages."""
    eligible = []
    episodes = [item["episode"] for item in evaluations]
    if len(episodes) != len(set(episodes)) or 0 not in episodes:
        raise ValueError("checkpoint set must include zero without duplicates")
    for item in evaluations:
        report = item["report"]
        values = [
            [row.get(metric) for row in report["runs"]]
            for metric in ("tangent_rmse_mm", "force_rmse_n")
        ]
        if _selection_eligible(report, expected_case_ids, thresholds):
            eligible.append((float(np.mean(values[0])), float(np.mean(values[1])), item["episode"]))
    if eligible:
        best = min(eligible)
        return {"episode": best[2], "eligible": True, "validation_ordering_values": list(best)}
    return {"episode": 0, "eligible": False, "reason": "no eligible validation checkpoint"}


def _evaluate(output, plan, candidate, name, split, nominal):
    destination = Path(output) / "evaluations" / f"{name}_{split}"
    purpose = "il_friction_teacher" if nominal == "adaptive" else "rl_residual"
    cases = pilot_cases(plan, nominal)
    destination.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = destination.with_suffix(".protocol.json")
    pin_path = destination.with_suffix(".protocol.sha256")
    if not protocol_path.exists():
        digest = freeze_evaluation_protocol(
            protocol_path,
            candidate,
            expected_contract=policy_contract(cases, purpose=purpose, nominal_kind=nominal),
            cases=cases,
            metrics=plan["metrics"],
            thresholds=plan["thresholds"],
            evaluation_split=split,
        )
        with pin_path.open("x") as stream:
            stream.write(digest + "\n")
    digest = pin_path.read_text().strip()
    frozen = verify_evaluation_protocol(
        protocol_path,
        candidate,
        expected_protocol_sha256=digest,
        expected_contract=policy_contract(cases, purpose=purpose, nominal_kind=nominal),
        cases=cases,
    )
    if (
        frozen["evaluation_split"] != split
        or frozen["metrics"] != plan["metrics"]
        or frozen["thresholds"] != plan["thresholds"]
    ):
        raise ValueError("evaluation protocol differs from pilot plan")
    if not destination.exists():
        print(f"evaluate {name} {split}", flush=True)
        evaluate_candidate(
            candidate,
            protocol_path,
            cases,
            digest,
            destination,
            purpose=purpose,
            nominal_kind=nominal,
        )
    manifest = _verified_manifest(destination)
    if (
        manifest["candidate_artifact_sha256"] != _sha256(Path(candidate))
        or manifest["protocol_sha256"] != digest
        or _sha256(protocol_path) != digest
        or manifest["evaluation_split"] != split
        or manifest["purpose"] != purpose
        or manifest["nominal_kind"] != nominal
        or manifest["source_and_assets_sha256"] != _source_hashes()
        or manifest["evaluation_case_ids"] != frozen["evaluation_case_ids"]
    ):
        raise ValueError("resumed evaluation identity differs")
    report = _read(destination / "report.json")
    if (
        report["evaluation_split"] != split
        or report["evaluation_case_ids"] != frozen["evaluation_case_ids"]
        or report["expected_case_count"] != len(frozen["evaluation_case_ids"])
        or sorted(row["case_id"] for row in report["runs"]) != frozen["evaluation_case_ids"]
    ):
        raise ValueError("evaluation report has incomplete or different cases")
    return report


def paired_baselines(report, baseline):
    by_key = {
        (row["method"], row["case_id"], row["simulation_seed"]): row for row in baseline["runs"]
    }
    if len(by_key) != len(baseline["runs"]):
        raise ValueError("duplicate baseline case/seed")
    pairs = []
    for row in report["runs"]:
        for method in ("zero_adaptive", "friction_teacher_50hz", "zero_friction"):
            reference = by_key.get((method, row["case_id"], row["simulation_seed"]))
            nominal = "friction" if method == "zero_friction" else "adaptive"
            if (
                reference is None
                or reference["nominal_kind"] != nominal
                or any(row[key] != reference[key] for key in ("group_id", "split"))
            ):
                raise ValueError("baseline pair does not match case/group/simulator seed")
            pairs.append(
                {
                    "case_id": row["case_id"],
                    "baseline": method,
                    "candidate_success": row["episode_success"],
                    "baseline_success": reference["hard_safe_completion"],
                    "delta_candidate_minus_baseline": {
                        metric: None if row.get(metric) is None else row[metric] - reference[metric]
                        for metric in (
                            "tangent_rmse_mm",
                            "force_rmse_n",
                            "contact_ratio_pct",
                            "episode_return",
                            "intervention_pct",
                        )
                    },
                }
            )
    return pairs


def _training_manifest(stage, directory, seed, plan):
    manifest = _verified_manifest(directory)
    parameters = manifest["hyperparameters"]
    if stage == "bc":
        valid = (
            manifest["schema"] == "surface_behavior_clone_run_v1"
            and parameters["seed"] == seed
            and all(parameters[key] == value for key, value in plan["bc"].items())
            and manifest["dataset"]["manifest_sha256"] == plan["dataset_manifest_sha256"]
            and manifest["trainer_source_sha256"]["trainer"]
            == plan["trainer_sha256"]["tools/train_surface_bc.py"]
            and manifest["development_test_used"] is False
        )
    else:
        valid = (
            manifest["schema"] == "surface_ppo_training_v1"
            and manifest["seed"] == seed
            and manifest["case_plan"] == pilot_cases(plan, "friction")
            and all(
                parameters["fixed_std" if key == "std" else key] == value
                for key, value in plan["ppo"].items()
            )
            and manifest["script_sha256"]["train_surface_ppo.py"]
            == plan["trainer_sha256"]["tools/train_surface_ppo.py"]
            and manifest["nominal_kind"] == "friction"
            and manifest["purpose"] == "rl_residual"
        )
    if not valid:
        raise ValueError("completed training run differs from frozen seed/data/budget/source")
    return manifest


def run_stage(stage, output, expected_sha256, dataset, benchmark):
    output = Path(output)
    plan = load_plan(output, expected_sha256, dataset, benchmark)
    summary_path = output / f"{stage}_summary.json"
    if summary_path.exists():
        raise ValueError(
            "stage already completed; inspect existing evidence, do not rerun development"
        )
    if stage == "ppo":
        bc_summary = _read(output / "bc_summary.json")
        if bc_summary["plan_sha256"] != expected_sha256 or (
            output / "bc_COMPLETE"
        ).read_text().strip() != _sha256(output / "bc_summary.json"):
            raise ValueError("BC stage must complete under the same frozen plan first")
    validation, selections, candidates = [], [], {}
    for seed in plan["seeds"]:
        train_output = output / f"{stage}_seed{seed}"
        if not train_output.exists():
            print(f"train {stage} seed={seed}", flush=True)
            if stage == "bc":
                from tools.train_surface_bc import train_bc

                train_bc(dataset, train_output, seed=seed, **plan["bc"])
            else:
                from tools.train_surface_ppo import train_ppo

                train_ppo(pilot_cases(plan, "friction"), train_output, seed=seed, **plan["ppo"])
        train_manifest = _training_manifest(stage, train_output, seed, plan)
        if stage == "bc":
            candidate = train_output / "selected_model.json"
            report = _evaluate(output, plan, candidate, f"bc_seed{seed}", "validation", "adaptive")
            validation.append({"seed": seed, "report": report})
            selected = {
                "seed": seed,
                "candidate": str(candidate.relative_to(output)),
                "candidate_sha256": _sha256(candidate),
                "selection": "offline validation action MSE; closed-loop is not reselection",
            }
        else:
            evaluated = []
            for episode in plan["ppo"]["checkpoint_episodes"]:
                candidate = train_output / train_manifest["checkpoints"][str(episode)]
                report = _evaluate(
                    output, plan, candidate, f"ppo_seed{seed}_ep{episode}", "validation", "friction"
                )
                evaluated.append({"episode": episode, "report": report})
            validation.append({"seed": seed, "checkpoints": evaluated})
            selected = {
                "seed": seed,
                **select_ppo_checkpoint(
                    evaluated,
                    expected_case_ids=[
                        case["case_id"] for case in plan["cases"] if case["split"] == "validation"
                    ],
                    thresholds=plan["thresholds"],
                ),
            }
            candidate = train_output / train_manifest["checkpoints"][str(selected["episode"])]
            selected.update(
                candidate=str(candidate.relative_to(output)), candidate_sha256=_sha256(candidate)
            )
        selections.append(selected)
        candidates[seed] = candidate
    selection_path = output / f"{stage}_selection.json"
    selection_record = {
        "plan_sha256": expected_sha256,
        "selections": selections,
        "validation": validation,
    }
    if selection_path.exists():
        if _read(selection_path) != selection_record:
            raise ValueError("selection changed after it was frozen")
    else:
        _write_new(selection_path, selection_record)
    baseline, development = _read(Path(benchmark) / "report.json"), []
    for seed in plan["seeds"]:
        report = _evaluate(
            output,
            plan,
            candidates[seed],
            f"{stage}_seed{seed}_selected",
            "development_test",
            "adaptive" if stage == "bc" else "friction",
        )
        development.append(
            {"seed": seed, "report": report, "paired_baselines": paired_baselines(report, baseline)}
        )
    load_plan(output, expected_sha256, dataset, benchmark)
    _write_new(
        summary_path,
        {
            "schema": "surface_learning_pilot_stage_v1",
            "stage": stage,
            "new_holdout": False,
            "plan_sha256": expected_sha256,
            "selection_sha256": _sha256(selection_path),
            "selections": selections,
            "validation": validation,
            "development_test": development,
            "all_selected_development_gates_met": all(
                x["report"]["acceptance_met"] for x in development
            ),
        },
    )
    with (output / f"{stage}_COMPLETE").open("x") as stream:
        stream.write(_sha256(summary_path) + "\n")
    print(f"complete {stage}: {summary_path}", flush=True)
    return summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "bc", "ppo"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args(argv)
    if args.stage == "freeze":
        print(freeze_plan(args.dataset, args.benchmark, args.output))
    else:
        if not args.expected_plan_sha256:
            parser.error("training requires --expected-plan-sha256 preserved before fitting")
        run_stage(args.stage, args.output, args.expected_plan_sha256, args.dataset, args.benchmark)


if __name__ == "__main__":
    main()
