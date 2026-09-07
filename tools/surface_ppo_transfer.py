"""Freeze and run the final-checkpoint BC-trunk PPO comparison.

The old fresh PPO runs are immutable controls.  Only episode 32 is compared;
validation may veto development but never selects an earlier checkpoint.
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_experiment import _sha256
from compliant_control_lab.surface_policy_artifact import load_policy_artifact, policy_contract
from tools import surface_bc_transfer as bc_protocol
from tools import surface_learning_pilot as pilot
from tools import surface_mlp_actor

SEEDS = (11, 29, 47)
SELECTED_BC_ARM = "drop_previous_residual"
FINAL_EPISODE = 32
SOURCE_FILES = (
    "tools/surface_ppo_transfer.py",
    "tools/train_surface_ppo_transfer.py",
    "tools/train_surface_ppo.py",
    "tools/surface_learning_pilot.py",
    "tools/surface_mlp_actor.py",
)


def _sources():
    root = Path(__file__).resolve().parents[1]
    return {name: _sha256(root / name) for name in SOURCE_FILES}


def _file_hashes(root, names):
    root = Path(root)
    return {name: _sha256(root / name) for name in names}


def _gate(report, split, case_ids, thresholds):
    """Reuse the original validation gate without changing its implementation."""
    normalized = deepcopy(report)
    if normalized.get("evaluation_split") != split:
        raise ValueError(f"expected {split} BC evidence")
    normalized["evaluation_split"] = "validation"
    for row in normalized.get("runs", []):
        if row.get("split") != split:
            raise ValueError(f"BC row is not from {split}")
        row["split"] = "validation"
    return pilot._selection_eligible(normalized, case_ids, thresholds)


def _bc_training_manifest(bc_output, bc_plan, arm, seed):
    """Accept the raw BC run or its verified publication layout."""
    directory = Path(bc_output) / f"{arm}_seed{seed}"
    if (directory / "manifest.json").exists():
        return bc_protocol._training_manifest(bc_output, bc_plan, arm, seed), (
            "manifest.json", "COMPLETE"
        )
    manifest_path, marker = directory / "source_manifest.json", directory / "source_COMPLETE"
    manifest = pilot._read(manifest_path)
    if marker.read_text().strip() != _sha256(manifest_path):
        raise ValueError("published BC source manifest is incomplete")
    for name, digest in manifest.get("artifact_sha256", {}).items():
        if _sha256(directory / name) != digest:
            raise ValueError(f"published BC source artifact changed: {name}")
    from tools.train_surface_bc_transfer import _input_mask

    parameters = manifest.get("hyperparameters", {})
    if (
        manifest.get("schema") != "surface_behavior_clone_transfer_run_v1"
        or manifest.get("arm") != arm
        or manifest.get("input_mask") != _input_mask(arm).astype(int).tolist()
        or parameters.get("seed") != seed
        or any(parameters.get(key) != value for key, value in bc_plan["bc"].items())
        or manifest.get("dataset", {}).get("manifest_sha256")
        != bc_plan["dataset_manifest_sha256"]
        or manifest.get("trainer_source_sha256") != bc_plan["transfer_trainer_sources"]
        or any(
            manifest.get("runtime", {}).get(key) != value
            for key, value in bc_plan["training_runtime"].items()
        )
        or manifest.get("development_test_used") is not False
    ):
        raise ValueError("published BC training differs from frozen inputs/arm/source")
    history = pilot._read(directory / "epoch_losses.json")
    losses = [row.get("validation_action_mse") for row in history]
    if (
        len(history) != bc_plan["bc"]["epochs"]
        or [row.get("epoch") for row in history] != list(range(1, len(history) + 1))
        or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in losses)
    ):
        raise ValueError("published BC checkpoint-selection history is incomplete")
    selected = min(history, key=lambda row: (row["validation_action_mse"], row["epoch"]))
    report = pilot._read(directory / "report.json")
    if report.get("selection", {}).get("selected_epoch") != selected["epoch"]:
        raise ValueError("published BC selected epoch differs from validation MSE")
    return manifest, ("source_manifest.json", "source_COMPLETE")


def _evaluation_evidence(
    evaluation, embedded_report, candidate_sha256, cases, split, nominal, source_sha256
):
    evaluation = Path(evaluation)
    published = not (evaluation / "manifest.json").exists()
    if published:
        manifest_name, marker_name = "source_manifest.json", "source_COMPLETE"
        manifest_path = evaluation / manifest_name
        manifest = pilot._read(manifest_path)
        if (evaluation / marker_name).read_text().strip() != _sha256(manifest_path):
            raise ValueError("published evaluation source manifest is incomplete")
        if manifest.get("artifact_sha256", {}).get("report.json") != _sha256(
            evaluation / "report.json"
        ):
            raise ValueError("published evaluation report differs from source manifest")
    else:
        manifest_name, marker_name = "manifest.json", "COMPLETE"
        manifest = pilot._verified_manifest(evaluation)
    report = pilot._read(evaluation / "report.json")
    protocol_path = evaluation.with_suffix(".protocol.json")
    pin_path = evaluation.with_suffix(".protocol.sha256")
    expected_contract = policy_contract(
        cases,
        purpose="il_friction_teacher" if nominal == "adaptive" else "rl_residual",
        nominal_kind=nominal,
    )
    case_ids = sorted(case["case_id"] for case in cases if case["split"] == split)
    if (
        report != embedded_report
        or manifest.get("candidate_artifact_sha256") != candidate_sha256
        or manifest.get("evaluation_split") != split
        or manifest.get("nominal_kind") != nominal
        or manifest.get("purpose") != expected_contract["purpose"]
        or manifest.get("policy_contract") != expected_contract
        or manifest.get("source_and_assets_sha256") != source_sha256
        or sorted(manifest.get("evaluation_case_ids", [])) != case_ids
        or manifest.get("protocol_sha256") != _sha256(protocol_path)
        or pin_path.read_text().strip() != _sha256(protocol_path)
    ):
        raise ValueError("evaluation report/protocol/candidate/source identity differs")
    return {
        "published_layout": published,
        **_file_hashes(evaluation, (manifest_name, marker_name, "report.json")),
        "protocol_json": _sha256(protocol_path),
        "protocol_marker": _sha256(pin_path),
    }


def _bc_records(bc_output, bc_plan):
    bc_output = Path(bc_output)
    summary_path, selection_path = bc_output / "summary.json", bc_output / "selection.json"
    stage_marker = "transfer_COMPLETE" if (bc_output / "transfer_COMPLETE").exists() else "COMPLETE"
    if (bc_output / stage_marker).read_text().strip() != _sha256(summary_path):
        raise ValueError("BC transfer summary is incomplete")
    publication = None
    if stage_marker == "transfer_COMPLETE":
        publication = pilot._verified_manifest(bc_output)
    summary, selection = pilot._read(summary_path), pilot._read(selection_path)
    plan_sha = _sha256(bc_output / "plan.json")
    if (
        summary.get("schema") != "surface_bc_transfer_summary_v1"
        or summary.get("plan_sha256") != plan_sha
        or summary.get("selection_sha256") != _sha256(selection_path)
        or summary.get("selection") != selection.get("selection")
        or summary.get("candidates") != selection.get("candidates")
        or summary.get("validation") != selection.get("validation")
        or summary["selection"].get("selected_arm") != SELECTED_BC_ARM
        or summary["selection"].get("eligible") is not True
        or summary["selection"].get("seed_selection") is not False
    ):
        raise ValueError("BC transfer selection/summary identity is invalid")
    expected = {(arm, seed) for arm in bc_plan["arms"] for seed in bc_plan["seeds"]}
    recomputed = bc_protocol.select_arm(summary["validation"], bc_plan)
    if recomputed != summary["selection"]:
        raise ValueError("BC selected arm does not reproduce from frozen validation")
    candidates = {(row["arm"], row["seed"]): row for row in summary["candidates"]}
    if set(candidates) != expected or len(summary["candidates"]) != len(expected):
        raise ValueError("BC candidate cohort is incomplete or duplicated")
    validation_keys = [(row["arm"], row["seed"]) for row in summary["validation"]]
    development_keys = [(row["arm"], row["seed"]) for row in summary["development_test"]]
    if (
        len(validation_keys) != len(expected)
        or set(validation_keys) != expected
        or len(development_keys) != len(expected)
        or set(development_keys) != expected
    ):
        raise ValueError("BC validation/development cohort is incomplete or duplicated")
    validation = [x for x in summary["validation"] if x["arm"] == SELECTED_BC_ARM]
    development = [x for x in summary["development_test"] if x["arm"] == SELECTED_BC_ARM]
    if [x["seed"] for x in validation] != list(SEEDS) or [x["seed"] for x in development] != list(SEEDS):
        raise ValueError("selected BC arm lacks the complete ordered seed cohort")
    case_ids = {
        split: [case["case_id"] for case in bc_plan["cases"] if case["split"] == split]
        for split in ("validation", "development_test")
    }
    records, selected, training_sources = {}, {}, {}
    for arm, seed in sorted(expected):
        row = candidates[(arm, seed)]
        candidate = bc_output / row["candidate"]
        if row["candidate_sha256"] != _sha256(candidate):
            raise ValueError("BC candidate changed after arm selection")
        manifest, source_names = _bc_training_manifest(bc_output, bc_plan, arm, seed)
        training_sources[(arm, seed)] = manifest["dataset"]["source_and_assets_sha256"]
        directory = candidate.parent
        records[f"{arm}_seed{seed}"] = {
            "candidate": row["candidate"],
            "candidate_sha256": row["candidate_sha256"],
            "manifest_file": source_names[0],
            "manifest_sha256": _sha256(directory / source_names[0]),
            "complete_file": source_names[1],
            "complete_sha256": _sha256(directory / source_names[1]),
            "report_sha256": _sha256(directory / "report.json"),
            "trainer_source_sha256": manifest["trainer_source_sha256"],
        }
        if arm == SELECTED_BC_ARM:
            report = pilot._read(directory / "report.json")
            folded = report["diagnostics"]["folded_export"]
            offline = report["diagnostics"]["validation"]
            parity_values = [value for key, value in folded.items() if key.endswith("max_abs")]
            tolerance = folded.get("absolute_tolerance", np.nan)
            if (
                folded.get("passed") is not True
                or not parity_values
                or not all(np.isfinite(value) for value in parity_values)
                or not np.isfinite(tolerance)
                or tolerance < 0
                or any(value > tolerance for value in parity_values)
                or not np.isfinite(offline.get("model_mse", np.nan))
                or not np.isfinite(offline.get("constant_train_label_mean_mse", np.nan))
                or offline["model_mse"] >= offline["constant_train_label_mean_mse"]
            ):
                raise ValueError("selected BC export/offline validation evidence failed")
            selected[seed] = {
                "candidate": row["candidate"],
                "candidate_sha256": row["candidate_sha256"],
                "folded_export": folded,
                "validation_model_mse": offline["model_mse"],
                "validation_constant_mse": offline["constant_train_label_mean_mse"],
            }
    evaluations = {}
    for item in summary["validation"]:
        key = item["arm"], item["seed"]
        row = candidates[key]
        evaluations[f"{key[0]}_seed{key[1]}_validation"] = _evaluation_evidence(
            bc_output / "evaluations" / f"{key[0]}_seed{key[1]}_validation",
            item["report"], row["candidate_sha256"], pilot.pilot_cases(bc_plan, "adaptive"),
            "validation", "adaptive", training_sources[key],
        )
    for item in summary["development_test"]:
        key = item["arm"], item["seed"]
        row = candidates[key]
        evaluations[f"{key[0]}_seed{key[1]}_development_test"] = _evaluation_evidence(
            bc_output / "evaluations" / f"{key[0]}_seed{key[1]}_development_test",
            item["report"], row["candidate_sha256"], pilot.pilot_cases(bc_plan, "adaptive"),
            "development_test", "adaptive", training_sources[key],
        )
    for item in validation:
        if not _gate(
            item["report"], "validation", case_ids["validation"], bc_plan["thresholds"]
        ):
            raise ValueError("selected BC arm failed an original validation gate")
    for item in development:
        if not _gate(
            item["report"], "development_test", case_ids["development_test"],
            bc_plan["thresholds"],
        ):
            raise ValueError("selected BC arm failed an original development gate")
    names = ["plan.json", "PLAN_SHA256", "selection.json", "summary.json", stage_marker]
    if publication is not None:
        names.extend(("manifest.json", "COMPLETE"))
    top = _file_hashes(bc_output, names)
    return {
        "top_level_sha256": top, "candidate_records": records,
        "evaluation_records": evaluations,
        "published_package_verified": publication is not None,
    }, selected


def _fresh_controls(parent, parent_plan):
    parent = Path(parent)
    summary_path = parent / "ppo_summary.json"
    if (parent / "ppo_COMPLETE").read_text().strip() != _sha256(summary_path):
        raise ValueError("fresh PPO stage is incomplete")
    summary = pilot._read(summary_path)
    if summary.get("plan_sha256") != _sha256(parent / "plan.json"):
        raise ValueError("fresh PPO summary belongs to another plan")
    validation_by_seed = {item["seed"]: item for item in summary["validation"]}
    if set(validation_by_seed) != set(SEEDS) or len(summary["validation"]) != len(SEEDS):
        raise ValueError("fresh PPO validation cohort is incomplete")
    expected_validation = [
        case["case_id"] for case in parent_plan["cases"] if case["split"] == "validation"
    ]
    controls = []
    for seed in SEEDS:
        directory = parent / f"ppo_seed{seed}"
        manifest = pilot._training_manifest("ppo", directory, seed, parent_plan)
        runs = manifest.get("runs", [])
        if (
            manifest.get("checkpoints")
            != {"0": "checkpoint_ep000.json", "16": "checkpoint_ep016.json",
                "32": "checkpoint_ep032.json"}
            or len(runs) != FINAL_EPISODE
            or len(manifest.get("updates", [])) != 8
            or [row.get("after_episode") for row in manifest.get("updates", [])]
            != list(range(4, 33, 4))
            or manifest.get("failed_episodes") != 0
            or [row.get("episode") for row in runs] != list(range(1, 33))
            or [row.get("case_id") for row in runs] != manifest.get("schedule_case_ids")
            or [row.get("requested_reset_seed") for row in runs] != manifest.get("reset_seeds")
            or {path.name for path in directory.glob("checkpoint_ep*.json")}
            != {"checkpoint_ep000.json", "checkpoint_ep016.json", "checkpoint_ep032.json"}
        ):
            raise ValueError("fresh PPO control does not have the complete fixed budget")
        candidate_name = manifest["checkpoints"].get(str(FINAL_EPISODE))
        if candidate_name != "checkpoint_ep032.json":
            raise ValueError("fresh control lacks the fixed final checkpoint")
        candidate = directory / candidate_name
        checkpoint_rows = validation_by_seed[seed]["checkpoints"]
        reports = {item["episode"]: item["report"] for item in checkpoint_rows}
        if (
            len(checkpoint_rows) != len(reports)
            or set(reports) != set(parent_plan["ppo"]["checkpoint_episodes"])
        ):
            raise ValueError("fresh validation checkpoint record is incomplete")
        report = reports[FINAL_EPISODE]
        if not _gate(report, "validation", expected_validation, parent_plan["thresholds"]):
            raise ValueError("fixed fresh episode-32 validation control is ineligible")
        evaluation = parent / "evaluations" / f"ppo_seed{seed}_ep32_validation"
        evaluation_evidence = _evaluation_evidence(
            evaluation, report, _sha256(candidate), pilot.pilot_cases(parent_plan, "friction"),
            "validation", "friction", manifest["source_and_assets_sha256"],
        )
        controls.append(
            {
                "seed": seed,
                "candidate": str(candidate.relative_to(parent)),
                "candidate_sha256": _sha256(candidate),
                "training_manifest_sha256": _sha256(directory / "manifest.json"),
                "training_complete_sha256": _sha256(directory / "COMPLETE"),
                "training_source_sha256": manifest["script_sha256"],
                "source_and_assets_sha256": manifest["source_and_assets_sha256"],
                "runtime": manifest["runtime"],
                "versions": manifest["versions"],
                "torch_seed": manifest["torch_seed"],
                "schedule_case_ids": manifest["schedule_case_ids"],
                "reset_seeds": manifest["reset_seeds"],
                "actual_simulation_seeds": [row["actual_simulation_seed"] for row in runs],
                "validation_report": report,
                "validation_evidence_sha256": evaluation_evidence,
            }
        )
    return controls


def _runtime():
    import platform

    import torch

    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "mujoco_version": version("mujoco"),
        "gymnasium_version": version("gymnasium"),
        "device": "cpu",
        "dtype": "float64",
        "torch_num_threads": 1,
        "deterministic_algorithms": True,
    }


def freeze(
    output, parent, bc_transfer, dataset, benchmark, expected_parent_sha256,
    expected_bc_sha256,
):
    """Freeze reliable BC inputs and immutable fresh episode-32 controls."""
    output = Path(output)
    if output.exists():
        raise ValueError("PPO transfer output must be new")
    parent_plan = pilot.load_plan(parent, expected_parent_sha256, dataset, benchmark)
    bc_plan = bc_protocol.load(
        bc_transfer, parent, dataset, benchmark, expected_bc_sha256
    )
    if tuple(parent_plan["seeds"]) != SEEDS or bc_plan["seeds"] != list(SEEDS):
        raise ValueError("PPO transfer requires the frozen three-seed cohort")
    expected_ppo = {
        "episodes": 32, "episodes_per_update": 4, "checkpoint_episodes": [0, 16, 32],
        "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2,
        "learning_rate": 0.0003, "update_epochs": 4, "batch_size": 128,
        "max_grad_norm": 0.5, "std": 0.05, "target_kl": 0.02,
    }
    if parent_plan["ppo"] != expected_ppo:
        raise ValueError("parent PPO budget differs from the fixed comparison")
    bc_records, selected = _bc_records(bc_transfer, bc_plan)
    fresh, runtime = _fresh_controls(parent, parent_plan), _runtime()
    old_versions = {
        "python": runtime["python_version"], "numpy": runtime["numpy_version"],
        "torch": runtime["torch_version"], "mujoco": runtime["mujoco_version"],
        "gymnasium": runtime["gymnasium_version"],
    }
    old_runtime = {
        key: runtime[key]
        for key in ("device", "dtype", "torch_num_threads", "deterministic_algorithms")
    }
    if any(item["runtime"] != old_runtime or item["versions"] != old_versions for item in fresh):
        raise ValueError("fresh control runtime differs from declared transfer runtime")
    plan = {
        "schema": "surface_ppo_bc_hidden_transfer_plan_v1",
        "new_holdout": False,
        "scope": "fixed public validation/development representation-transfer comparison",
        "parent_plan_sha256": expected_parent_sha256,
        "bc_transfer_plan_sha256": expected_bc_sha256,
        "input_roots_relative_to_output": {
            "parent": os.path.relpath(Path(parent).resolve(), output.resolve()),
            "bc_transfer": os.path.relpath(Path(bc_transfer).resolve(), output.resolve()),
        },
        "source_sha256": _sources(),
        "runtime": runtime,
        "dataset_manifest_sha256": parent_plan["dataset_manifest_sha256"],
        "benchmark_sha256": _file_hashes(
            benchmark, ("manifest.json", "report.json", "COMPLETE")
        ),
        "seeds": list(SEEDS),
        "cases": parent_plan["cases"],
        "metrics": parent_plan["metrics"],
        "thresholds": parent_plan["thresholds"],
        "ppo": parent_plan["ppo"],
        "selected_bc_arm": SELECTED_BC_ARM,
        "selected_bc_candidates": {str(key): value for key, value in selected.items()},
        "bc_records": bc_records,
        "fresh_controls": fresh,
        "initialization": {
            "copied": "BC actor hidden layers 0 and 1 only",
            "not_copied": "BC action head; BC actions are not converted to friction residuals",
            "episode0": "fresh exactly-zero residual head",
            "rl_inputs": "all 49 measured observations",
            "optimization": "transferred trunk remains unfrozen; all actor/critic parameters train",
            "nominal_kind": "friction",
        },
        "comparison": {
            "checkpoint_episode": FINAL_EPISODE,
            "checkpoint_selection": False,
            "validation_control": "immutable original-pilot fresh PPO episode 32",
            "acceptance": (
                "all transfer seeds pass unchanged validation gates, no failed training episodes, "
                "and paired mean tangent_rmse_mm transfer-minus-fresh is negative for every seed"
            ),
            "minimum_improvement": None,
            "if_not_consistent": "retain fresh control; do not claim RL is invalid",
            "development": (
                "freeze all candidate/comparison hashes first, then report both fixed episode-32 "
                "arms; fresh is a repeated public comparison, not an independent case"
            ),
        },
    }
    output.mkdir(parents=True)
    pilot._write_new(output / "plan.json", plan)
    digest = _sha256(output / "plan.json")
    with (output / "PLAN_SHA256").open("x") as stream:
        stream.write(digest + "\n")
    return digest


def load(
    output, parent, bc_transfer, dataset, benchmark, expected_sha256,
):
    output = Path(output)
    if _sha256(output / "plan.json") != expected_sha256:
        raise ValueError("PPO transfer plan differs from externally preserved SHA")
    if (output / "PLAN_SHA256").read_text().strip() != expected_sha256:
        raise ValueError("PPO transfer plan marker differs")
    plan = pilot._read(output / "plan.json")
    expected_roots = {
        "parent": os.path.relpath(Path(parent).resolve(), output.resolve()),
        "bc_transfer": os.path.relpath(Path(bc_transfer).resolve(), output.resolve()),
    }
    if plan.get("input_roots_relative_to_output") != expected_roots:
        raise ValueError("supplied parent/BC roots differ from frozen relative locations")
    parent_plan = pilot.load_plan(parent, plan["parent_plan_sha256"], dataset, benchmark)
    bc_plan = bc_protocol.load(
        bc_transfer, parent, dataset, benchmark, plan["bc_transfer_plan_sha256"]
    )
    bc_records, selected = _bc_records(bc_transfer, bc_plan)
    current = {
        "source_sha256": _sources(), "runtime": _runtime(),
        "dataset_manifest_sha256": parent_plan["dataset_manifest_sha256"],
        "benchmark_sha256": _file_hashes(
            benchmark, ("manifest.json", "report.json", "COMPLETE")
        ),
        "cases": parent_plan["cases"], "metrics": parent_plan["metrics"],
        "thresholds": parent_plan["thresholds"], "ppo": parent_plan["ppo"],
        "bc_records": bc_records,
        "selected_bc_candidates": {str(key): value for key, value in selected.items()},
        "fresh_controls": _fresh_controls(parent, parent_plan),
    }
    for key, value in current.items():
        if plan.get(key) != value:
            raise ValueError(f"frozen PPO transfer input changed: {key}")
    if plan.get("seeds") != list(SEEDS) or plan.get("selected_bc_arm") != SELECTED_BC_ARM:
        raise ValueError("frozen PPO transfer cohort/BC arm changed")
    return plan


def _expected_artifacts():
    names = {f"checkpoint_ep{episode:03d}.json" for episode in (0, 16, 32)}
    for episode in range(1, FINAL_EPISODE + 1):
        names.update((f"episode_{episode:03d}.json", f"episode_{episode:03d}.npz"))
    return names


def _input_root(output, plan, name):
    return (Path(output) / plan["input_roots_relative_to_output"][name]).resolve()


def _zero_checkpoint_audit(candidate, bc_candidate, plan, seed):
    cases = pilot.pilot_cases(plan, "friction")
    contract = policy_contract(cases, purpose="rl_residual", nominal_kind="friction")
    artifact = load_policy_artifact(candidate, expected_contract=contract)
    actor, _ = surface_mlp_actor.actor_from_artifact(artifact)
    bc_cases = pilot.pilot_cases(plan, "adaptive")
    bc_contract = policy_contract(
        bc_cases, purpose="il_friction_teacher", nominal_kind="adaptive"
    )
    bc_artifact = load_policy_artifact(bc_candidate, expected_contract=bc_contract)
    surface_mlp_actor.actor_from_artifact(bc_artifact)
    episode0_layers = surface_mlp_actor._dense_layers(artifact.payload["layers"])
    bc_layers = surface_mlp_actor._dense_layers(bc_artifact.payload["layers"])
    trunk_equal = [
        bool(np.array_equal(left[0], right[0]) and np.array_equal(left[1], right[1]))
        for left, right in zip(episode0_layers[:2], bc_layers[:2])
    ]
    if trunk_equal != [True, True]:
        raise ValueError("episode-0 trunk differs from pinned BC candidate")
    probes = np.vstack(
        (np.zeros((1, 49)), np.linspace(-3, 3, 49)[None],
         np.random.default_rng(seed + 3200).uniform(-3, 3, (16, 49)))
    )
    outputs = np.asarray([actor(row) for row in probes])
    if not np.array_equal(outputs, np.zeros_like(outputs)):
        raise ValueError("episode-0 checkpoint is not exact zero residual")
    return {
        "candidate_sha256": _sha256(candidate), "probe_count": len(probes),
        "max_abs_action": float(np.max(np.abs(outputs))), "exact_zero": True,
        "pinned_bc_candidate_sha256": _sha256(bc_candidate),
        "hidden_layer_exact_equal_to_pinned_bc": trunk_equal,
        "bc_output_head_copied": False,
    }


def _training_manifest(output, plan, seed):
    directory = Path(output) / f"transfer_seed{seed}"
    if (directory / "FAILED").exists():
        raise ValueError("failed transfer training evidence is preserved; do not overwrite")
    manifest = pilot._verified_manifest(directory)
    control = next(item for item in plan["fresh_controls"] if item["seed"] == seed)
    selected = plan["selected_bc_candidates"][str(seed)]
    parameters = manifest.get("hyperparameters", {})
    semantics = manifest.get("transfer_semantics", {})
    expected_scripts = {
        "train_surface_ppo_transfer.py": plan["source_sha256"][
            "tools/train_surface_ppo_transfer.py"
        ],
        "train_surface_ppo.py": plan["source_sha256"]["tools/train_surface_ppo.py"],
        "surface_mlp_actor.py": plan["source_sha256"]["tools/surface_mlp_actor.py"],
    }
    expected_semantics = {
        "bc_action_head_copied": False,
        "bc_actions_converted_to_friction_residuals": False,
        "runtime_observation_inputs": "all 49 measured inputs; no BC input mask",
        "actor_hidden_trunk_trainable_during_rl": True,
        "optimizer_scope": "all actor and critic parameters",
    }
    ppo_match = all(
        parameters.get("fixed_std" if key == "std" else key) == value
        for key, value in plan["ppo"].items()
    )
    runs, updates = manifest.get("runs", []), manifest.get("updates", [])
    valid = (
        manifest.get("schema") == "surface_ppo_bc_hidden_transfer_v1"
        and manifest.get("seed") == seed
        and manifest.get("nominal_kind") == "friction"
        and manifest.get("purpose") == "rl_residual"
        and manifest.get("case_plan") == pilot.pilot_cases(plan, "friction")
        and ppo_match
        and manifest.get("source_and_assets_sha256")
        == control["source_and_assets_sha256"]
    )
    # Keep the long identity checks outside a conditional expression for auditability.
    valid = valid and (
        manifest.get("script_sha256") == expected_scripts
        and manifest.get("runtime") == {"before": plan["runtime"], "after": plan["runtime"]}
        and manifest.get("versions")
        == {key: plan["runtime"][key] for key in (
            "python_version", "numpy_version", "torch_version", "mujoco_version",
            "gymnasium_version",
        )}
        and manifest.get("initialization", {}).get("inherited_bc_candidate_sha256")
        == selected["candidate_sha256"]
        and manifest.get("initialization", {}).get("fresh_critic_exactly_unchanged") is True
        and manifest.get("initialization", {}).get("episode0_zero_probe", {}).get(
            "max_abs_action"
        ) == 0.0
        and manifest.get("architecture")
        == "BC 49-32-32 trunk; fresh zero 3 head; fresh value 49-32-32-1"
        and semantics == expected_semantics
        and manifest.get("torch_seed") == control["torch_seed"]
        and manifest.get("schedule_case_ids") == control["schedule_case_ids"]
        and manifest.get("reset_seeds") == control["reset_seeds"]
        and manifest.get("data_use") == {
            "optimization_split": "train", "validation_loaded": False,
            "development_test_loaded": False, "selection_performed": False,
        }
        and manifest.get("checkpoints")
        == {"0": "checkpoint_ep000.json", "16": "checkpoint_ep016.json",
            "32": "checkpoint_ep032.json"}
        and set(manifest.get("artifact_sha256", {})) == _expected_artifacts()
        and len(runs) == FINAL_EPISODE
        and [row.get("episode") for row in runs] == list(range(1, FINAL_EPISODE + 1))
        and [row.get("case_id") for row in runs] == control["schedule_case_ids"]
        and [row.get("requested_reset_seed") for row in runs] == control["reset_seeds"]
        and [row.get("actual_simulation_seed") for row in runs]
        == control["actual_simulation_seeds"]
        and len(updates) == 8
        and [row.get("after_episode") for row in updates] == list(range(4, 33, 4))
        and manifest.get("failed_episodes") == sum(bool(row.get("failed")) for row in runs)
        and manifest.get("actual_physics_steps")
        == sum(row.get("physics_steps", -1) for row in runs)
        and manifest.get("paired_fresh_control", {}).get(
            "shared_rng_consumption_may_diverge_after_kl_stop"
        ) is True
        and manifest.get("paired_fresh_control", {}).get(
            "same_case_schedule_reset_seed_and_initial_rng_derivation"
        ) is True
        and manifest.get("policy_contract")
        == policy_contract(
            pilot.pilot_cases(plan, "friction"), purpose="rl_residual", nominal_kind="friction"
        )
        and {path.name for path in directory.glob("checkpoint_ep*.json")}
        == {"checkpoint_ep000.json", "checkpoint_ep016.json", "checkpoint_ep032.json"}
    )
    if not valid:
        raise ValueError("transfer training differs from frozen source/runtime/init/schedule/budget")
    bc_candidate = _input_root(output, plan, "bc_transfer") / selected["candidate"]
    if _sha256(bc_candidate) != selected["candidate_sha256"]:
        raise ValueError("pinned BC candidate changed before episode-0 audit")
    zero = _zero_checkpoint_audit(
        directory / "checkpoint_ep000.json", bc_candidate, plan, seed
    )
    return manifest, zero


def train_one(output, plan, seed):
    from tools.train_surface_ppo_transfer import train_ppo_transfer

    if seed not in plan["seeds"]:
        raise ValueError("undeclared transfer seed")
    directory = Path(output) / f"transfer_seed{seed}"
    if directory.exists() and not (directory / "COMPLETE").exists():
        raise ValueError("incomplete/failed transfer evidence is preserved; do not overwrite")
    if not directory.exists():
        selected = plan["selected_bc_candidates"][str(seed)]
        train_ppo_transfer(
            pilot.pilot_cases(plan, "friction"), directory,
            bc_candidate=_input_root(output, plan, "bc_transfer") / selected["candidate"],
            bc_cases=pilot.pilot_cases(plan, "adaptive"), seed=seed, **plan["ppo"],
        )
    manifest, _ = _training_manifest(output, plan, seed)
    return directory / manifest["checkpoints"][str(FINAL_EPISODE)]


def validate_one(output, plan, seed):
    if seed not in plan["seeds"]:
        raise ValueError("undeclared transfer seed")
    manifest, _ = _training_manifest(output, plan, seed)
    candidate = Path(output) / f"transfer_seed{seed}" / manifest["checkpoints"]["32"]
    return pilot._evaluate(
        output, plan, candidate, f"transfer_seed{seed}_ep32", "validation", "friction"
    )


def _paired_tangent_delta(transfer_report, fresh_report):
    fresh = {
        (row["case_id"], row["simulation_seed"]): row for row in fresh_report["runs"]
    }
    if len(fresh) != len(fresh_report["runs"]):
        raise ValueError("duplicate fresh comparison row")
    deltas, observed, unavailable = [], set(), []
    for row in transfer_report["runs"]:
        key = row["case_id"], row["simulation_seed"]
        if key in observed:
            raise ValueError("duplicate transfer comparison row")
        observed.add(key)
        reference = fresh.get(key)
        if reference is None or any(
            row[field] != reference[field] for field in ("group_id", "split")
        ):
            raise ValueError("transfer/fresh comparison is not simulator-paired")
        values = row.get("tangent_rmse_mm"), reference.get("tangent_rmse_mm")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(value) for value in values
        ):
            # A failed rollout is an outcome, not a row to drop from the mean.
            deltas.append(None)
            unavailable.append(row["case_id"])
        else:
            deltas.append(float(values[0] - values[1]))
    if observed != set(fresh) or not observed:
        raise ValueError("transfer/fresh comparison case cohort differs")
    return {
        "per_case": deltas,
        "mean": None if unavailable else float(np.mean(deltas)),
        "all_metrics_available": not unavailable,
        "unavailable_case_ids": unavailable,
    }


def _comparison(output, parent, plan, validation):
    expected_ids = [case["case_id"] for case in plan["cases"] if case["split"] == "validation"]
    if [item["seed"] for item in validation] != list(SEEDS):
        raise ValueError("comparison requires all transfer seeds exactly once")
    rows, validation_safe = [], True
    for item in validation:
        seed, report = item["seed"], item["report"]
        manifest, zero = _training_manifest(output, plan, seed)
        control = next(row for row in plan["fresh_controls"] if row["seed"] == seed)
        transfer_candidate = Path(output) / f"transfer_seed{seed}" / "checkpoint_ep032.json"
        fresh_candidate = Path(parent) / control["candidate"]
        eligible = pilot._selection_eligible(report, expected_ids, plan["thresholds"])
        no_failed = manifest["failed_episodes"] == 0
        paired = _paired_tangent_delta(report, control["validation_report"])
        validation_safe = (
            validation_safe and eligible and no_failed and paired["all_metrics_available"]
        )
        rows.append(
            {
                "seed": seed, "transfer_validation": report,
                "fresh_validation": control["validation_report"],
                "transfer_validation_eligible": eligible,
                "no_failed_training_episodes": no_failed,
                "paired_tangent_delta_transfer_minus_fresh_mm": paired,
                "transfer_candidate_sha256": _sha256(transfer_candidate),
                "fresh_candidate_sha256": _sha256(fresh_candidate),
                "episode0_zero_checkpoint_audit": zero,
            }
        )
    consistent = validation_safe and all(
        row["paired_tangent_delta_transfer_minus_fresh_mm"]["mean"] < 0 for row in rows
    )
    return {
        "schema": "surface_ppo_bc_hidden_transfer_comparison_v1",
        "plan_sha256": _sha256(Path(output) / "plan.json"),
        "fixed_checkpoint_episode": FINAL_EPISODE,
        "checkpoint_selected_on_validation": False,
        "seeds": rows,
        "validation_safe_for_development": validation_safe,
        "consistent_transfer_gain": consistent,
        "retained_arm": "bc_hidden_transfer_ep32" if consistent else "fresh_ep32",
        "interpretation_if_not_consistent": "retain fresh control; this does not invalidate RL",
    }


def _freeze_comparison(output, record):
    path = Path(output) / "comparison.json"
    if path.exists():
        if pilot._read(path) != record:
            raise ValueError("candidate/validation comparison changed after freeze")
    else:
        pilot._write_new(path, record)
    return path


def run(output, parent, bc_transfer, dataset, benchmark, expected_sha256):
    output = Path(output)
    plan = load(output, parent, bc_transfer, dataset, benchmark, expected_sha256)
    if (output / "summary.json").exists():
        state = "completed" if (output / "COMPLETE").exists() else "incomplete"
        raise ValueError(f"PPO transfer summary is {state}; preserve it and do not rerun")
    if not (output / "comparison.json").exists():
        premature = list((output / "evaluations").glob("*_seed*_ep32_development_test"))
        if premature:
            raise ValueError("development evidence exists before comparison freeze")
    validation = []
    for seed in SEEDS:
        train_one(output, plan, seed)
        validation.append({"seed": seed, "report": validate_one(output, plan, seed)})
    comparison = _comparison(output, parent, plan, validation)
    comparison_path = _freeze_comparison(output, comparison)
    if not comparison["validation_safe_for_development"]:
        veto = output / "VALIDATION_VETO"
        if not veto.exists():
            with veto.open("x") as stream:
                stream.write(_sha256(comparison_path) + "\n")
        return comparison_path
    baseline, development = pilot._read(Path(benchmark) / "report.json"), []
    for arm in ("fresh", "transfer"):
        for seed in SEEDS:
            if arm == "fresh":
                control = next(row for row in plan["fresh_controls"] if row["seed"] == seed)
                candidate = Path(parent) / control["candidate"]
            else:
                candidate = output / f"transfer_seed{seed}" / "checkpoint_ep032.json"
            report = pilot._evaluate(
                output, plan, candidate, f"{arm}_seed{seed}_ep32",
                "development_test", "friction",
            )
            development.append(
                {
                    "arm": arm, "seed": seed, "report": report,
                    "paired_baselines": pilot.paired_baselines(report, baseline),
                    "public_comparison": True,
                    "independent_new_case": False,
                }
            )
    load(output, parent, bc_transfer, dataset, benchmark, expected_sha256)
    if pilot._read(comparison_path) != comparison:
        raise ValueError("comparison changed before publication")
    summary = {
        "schema": "surface_ppo_bc_hidden_transfer_summary_v1",
        "new_holdout": False, "plan_sha256": expected_sha256,
        "comparison_sha256": _sha256(comparison_path), "comparison": comparison,
        "development_test": development,
        "development_use": "repeated public comparison; not independent evidence or selection",
    }
    pilot._write_new(output / "summary.json", summary)
    with (output / "COMPLETE").open("x") as stream:
        stream.write(_sha256(output / "summary.json") + "\n")
    return output / "summary.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "train", "validate", "run"))
    for name in ("output", "parent", "bc-transfer", "dataset", "benchmark"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-parent-plan-sha256")
    parser.add_argument("--expected-bc-plan-sha256")
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.stage == "freeze":
        if not args.expected_parent_plan_sha256 or not args.expected_bc_plan_sha256:
            parser.error("freeze requires externally preserved parent and BC plan SHAs")
        print(freeze(
            args.output, args.parent, args.bc_transfer, args.dataset, args.benchmark,
            args.expected_parent_plan_sha256, args.expected_bc_plan_sha256,
        ))
        return
    if not args.expected_plan_sha256:
        parser.error("train/validate/run requires the preserved PPO-transfer plan SHA")
    plan = load(
        args.output, args.parent, args.bc_transfer, args.dataset, args.benchmark,
        args.expected_plan_sha256,
    )
    if args.stage in ("train", "validate"):
        if args.seed is None:
            parser.error("train/validate requires --seed")
        result = (
            train_one(args.output, plan, args.seed)
            if args.stage == "train" else validate_one(args.output, plan, args.seed)
        )
        load(
            args.output, args.parent, args.bc_transfer, args.dataset, args.benchmark,
            args.expected_plan_sha256,
        )
        print(result)
    else:
        print(run(
            args.output, args.parent, args.bc_transfer, args.dataset, args.benchmark,
            args.expected_plan_sha256,
        ))


if __name__ == "__main__":
    main()
