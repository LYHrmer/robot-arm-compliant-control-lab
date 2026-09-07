"""Publish and audit compact evidence for the BC-trunk PPO comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_experiment import _sha256
from compliant_control_lab.surface_policy_artifact import (
    load_policy_artifact,
    policy_contract,
    verify_evaluation_protocol,
)
from compliant_control_lab.surface_readiness_benchmark import METRICS, audit_trace
from tools import surface_learning_pilot as pilot
from tools import surface_mlp_actor
from tools import surface_ppo_transfer as protocol


def _copy(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink():
        raise ValueError("publication refuses source symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _write(path, value):
    pilot._write_new(path, value)


def _case_ids(plan, split):
    return sorted(case["case_id"] for case in plan["cases"] if case["split"] == split)


def _validate_report(report, plan, split):
    expected = _case_ids(plan, split)
    if (
        report.get("evaluation_split") != split
        or report.get("evaluation_case_ids") != expected
        or report.get("expected_case_count") != len(expected)
        or sorted(row["case_id"] for row in report.get("runs", [])) != expected
        or any(row.get("split") != split for row in report.get("runs", []))
    ):
        raise ValueError(f"incomplete or different {split} evaluation cohort")


def _source_manifest(directory, *, public=False):
    directory = Path(directory)
    if not public:
        return pilot._verified_manifest(directory)
    manifest_name = "source_manifest.json" if public else "manifest.json"
    marker_name = "source_COMPLETE" if public else "COMPLETE"
    manifest_path = directory / manifest_name
    manifest = pilot._read(manifest_path)
    if (directory / marker_name).read_text().strip() != _sha256(manifest_path):
        raise ValueError("source COMPLETE does not bind manifest")
    return manifest


def _validate_training(directory, plan, seed, *, public=False):
    directory = Path(directory)
    manifest = _source_manifest(directory, public=public)
    copied = {f"checkpoint_ep{episode:03d}.json" for episode in (0, 16, 32)}
    if any(
        _sha256(directory / name) != manifest.get("artifact_sha256", {}).get(name)
        for name in copied
    ):
        raise ValueError("PPO checkpoint differs from training manifest")
    parameters = manifest.get("hyperparameters", {})
    expected_parameters = {
        "fixed_std" if key == "std" else key: value for key, value in plan["ppo"].items()
    }
    control = next(row for row in plan["fresh_controls"] if row["seed"] == seed)
    selected = plan["selected_bc_candidates"][str(seed)]
    runs, updates = manifest.get("runs", []), manifest.get("updates", [])
    expected_semantics = {
        "bc_action_head_copied": False,
        "bc_actions_converted_to_friction_residuals": False,
        "runtime_observation_inputs": "all 49 measured inputs; no BC input mask",
        "actor_hidden_trunk_trainable_during_rl": True,
        "optimizer_scope": "all actor and critic parameters",
    }
    if (
        manifest.get("schema") != "surface_ppo_bc_hidden_transfer_v1"
        or manifest.get("seed") != seed
        or manifest.get("nominal_kind") != "friction"
        or manifest.get("purpose") != "rl_residual"
        or any(parameters.get(key) != value for key, value in expected_parameters.items())
        or manifest.get("checkpoints")
        != {"0": "checkpoint_ep000.json", "16": "checkpoint_ep016.json", "32": "checkpoint_ep032.json"}
        or len(runs) != 32
        or [row.get("episode") for row in runs] != list(range(1, 33))
        or [row.get("case_id") for row in runs] != control["schedule_case_ids"]
        or [row.get("requested_reset_seed") for row in runs] != control["reset_seeds"]
        or [row.get("actual_simulation_seed") for row in runs]
        != control["actual_simulation_seeds"]
        or len(updates) != 8
        or [row.get("after_episode") for row in updates] != list(range(4, 33, 4))
        or manifest.get("failed_episodes") != sum(bool(row.get("failed")) for row in runs)
        or manifest.get("actual_physics_steps")
        != sum(row.get("physics_steps", -1) for row in runs)
        or manifest.get("torch_seed") != control["torch_seed"]
        or manifest.get("source_and_assets_sha256") != control["source_and_assets_sha256"]
        or manifest.get("initialization", {}).get("inherited_bc_candidate_sha256")
        != selected["candidate_sha256"]
        or manifest.get("initialization", {}).get("fresh_critic_exactly_unchanged") is not True
        or manifest.get("transfer_semantics") != expected_semantics
        or manifest.get("data_use")
        != {
            "optimization_split": "train",
            "validation_loaded": False,
            "development_test_loaded": False,
            "selection_performed": False,
        }
        or manifest.get("runtime") != {"before": plan["runtime"], "after": plan["runtime"]}
        or manifest.get("script_sha256")
        != {
            "train_surface_ppo_transfer.py": plan["source_sha256"][
                "tools/train_surface_ppo_transfer.py"
            ],
            "train_surface_ppo.py": plan["source_sha256"]["tools/train_surface_ppo.py"],
            "surface_mlp_actor.py": plan["source_sha256"]["tools/surface_mlp_actor.py"],
        }
    ):
        raise ValueError("PPO transfer training differs from frozen budget/pairing/runtime")
    if public and pilot._read(directory / "training_curve.json") != {
        "seed": seed,
        "runs": runs,
        "updates": updates,
    }:
        raise ValueError("public training curve differs from source manifest")
    return manifest


def _validate_fresh(directory, plan, seed, *, public=False):
    directory = Path(directory)
    manifest = _source_manifest(directory, public=public)
    control = next(row for row in plan["fresh_controls"] if row["seed"] == seed)
    candidate = directory / "checkpoint_ep032.json"
    if (
        _sha256(candidate) != control["candidate_sha256"]
        or manifest.get("artifact_sha256", {}).get("checkpoint_ep032.json")
        != _sha256(candidate)
        or _sha256(directory / ("source_manifest.json" if public else "manifest.json"))
        != control["training_manifest_sha256"]
        or _sha256(directory / ("source_COMPLETE" if public else "COMPLETE"))
        != control["training_complete_sha256"]
    ):
        raise ValueError("fresh episode-32 control identity differs")
    return manifest


def _bc_candidate_audit(directory, plan, seed, *, public=False):
    directory = Path(directory)
    manifest = _source_manifest(directory, public=public)
    candidate = directory / "selected_model.json"
    selected = plan["selected_bc_candidates"][str(seed)]
    record = plan["bc_records"]["candidate_records"][f"{plan['selected_bc_arm']}_seed{seed}"]
    manifest_name = "source_manifest.json" if public else record["manifest_file"]
    marker_name = "source_COMPLETE" if public else record["complete_file"]
    if (
        _sha256(candidate) != selected["candidate_sha256"]
        or manifest.get("artifact_sha256", {}).get("selected_model.json") != _sha256(candidate)
        or _sha256(directory / manifest_name) != record["manifest_sha256"]
        or _sha256(directory / marker_name) != record["complete_sha256"]
        or _sha256(directory / "report.json") != record["report_sha256"]
    ):
        raise ValueError("inherited BC candidate identity differs")
    return candidate


def _zero_and_trunk_audit(checkpoint, bc_candidate, plan, seed):
    rl_contract = policy_contract(
        pilot.pilot_cases(plan, "friction"), purpose="rl_residual", nominal_kind="friction"
    )
    bc_contract = policy_contract(
        pilot.pilot_cases(plan, "adaptive"),
        purpose="il_friction_teacher",
        nominal_kind="adaptive",
    )
    artifact = load_policy_artifact(checkpoint, expected_contract=rl_contract)
    bc_artifact = load_policy_artifact(bc_candidate, expected_contract=bc_contract)
    expected_runner = {
        "schema": surface_mlp_actor.RUNNER_SCHEMA,
        "runner_sha256": plan["source_sha256"]["tools/surface_mlp_actor.py"],
        "evaluator_sha256": _sha256(Path(surface_mlp_actor.__file__).with_name(
            "evaluate_surface_candidate.py"
        )),
        "package_source_and_assets_sha256": next(
            row["source_and_assets_sha256"]
            for row in plan["fresh_controls"]
            if row["seed"] == seed
        ),
        "python_version": plan["runtime"]["python_version"],
        "numpy_version": plan["runtime"]["numpy_version"],
        "mujoco_version": plan["runtime"]["mujoco_version"],
        "gymnasium_version": plan["runtime"]["gymnasium_version"],
    }

    def checked_layers(candidate):
        payload = candidate.payload
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {"kind", "activation", "output_activation", "layers", "runner_identity"}
            or payload.get("kind") != surface_mlp_actor.FORMAT
            or payload.get("activation") != "tanh"
            or payload.get("output_activation") != "tanh"
            or payload.get("runner_identity") != expected_runner
        ):
            raise ValueError("candidate MLP or recorded runner identity differs from frozen plan")
        dense = surface_mlp_actor._dense_layers(payload["layers"])
        expected_shapes = (((32, 49), (32,)), ((32, 32), (32,)), ((3, 32), (3,)))
        if [
            (weights.shape, bias.shape) for weights, bias in dense
        ] != list(expected_shapes):
            raise ValueError("candidate is not a finite 49-32-32-3 MLP")
        return dense

    current_runner = surface_mlp_actor.current_runner_identity()
    stable_identity = (
        "schema",
        "runner_sha256",
        "evaluator_sha256",
        "package_source_and_assets_sha256",
    )
    if any(expected_runner[key] != current_runner[key] for key in stable_identity):
        raise ValueError("frozen runner/source identity differs from the audit code")
    layers, bc_layers = checked_layers(artifact), checked_layers(bc_artifact)
    equal = [
        bool(np.array_equal(left[0], right[0]) and np.array_equal(left[1], right[1]))
        for left, right in zip(layers[:2], bc_layers[:2])
    ]
    probes = np.vstack(
        (
            np.zeros((1, 49)),
            np.linspace(-3, 3, 49)[None],
            np.random.default_rng(seed + 3200).uniform(-3, 3, (16, 49)),
        )
    )
    values = probes
    with np.errstate(over="raise", invalid="raise"):
        for weights, bias in layers[:-1]:
            values = np.tanh(values @ weights.T + bias)
        outputs = np.tanh(values @ layers[-1][0].T + layers[-1][1])
    zero_head = bool(
        np.array_equal(layers[-1][0], np.zeros_like(layers[-1][0]))
        and np.array_equal(layers[-1][1], np.zeros_like(layers[-1][1]))
    )
    if (
        equal != [True, True]
        or not zero_head
        or not np.array_equal(outputs, np.zeros_like(outputs))
    ):
        raise ValueError("episode-0 checkpoint is not the pinned BC trunk with a zero head")
    return {
        "candidate_sha256": _sha256(checkpoint),
        "probe_count": len(probes),
        "max_abs_action": float(np.max(np.abs(outputs))),
        "exact_zero": True,
        "pinned_bc_candidate_sha256": _sha256(bc_candidate),
        "hidden_layer_exact_equal_to_pinned_bc": equal,
        "bc_output_head_copied": False,
    }


def _evaluation_source(directory, candidate, plan, split, *, public=False):
    directory = Path(directory)
    manifest = _source_manifest(directory, public=public)
    if manifest.get("artifact_sha256", {}).get("report.json") != _sha256(
        directory / "report.json"
    ):
        raise ValueError("evaluation report differs from source manifest")
    protocol_path = directory.with_suffix(".protocol.json")
    protocol_pin = directory.with_suffix(".protocol.sha256")
    digest = _sha256(protocol_path)
    contract = policy_contract(
        pilot.pilot_cases(plan, "friction"), purpose="rl_residual", nominal_kind="friction"
    )
    frozen = verify_evaluation_protocol(
        protocol_path,
        candidate,
        expected_protocol_sha256=digest,
        expected_contract=contract,
        cases=pilot.pilot_cases(plan, "friction"),
    )
    if (
        manifest.get("protocol_sha256") != digest
        or protocol_pin.read_text().strip() != digest
        or frozen.get("evaluation_split") != split
        or frozen.get("metrics") != plan["metrics"]
        or frozen.get("thresholds") != plan["thresholds"]
    ):
        raise ValueError("evaluation protocol pin differs")
    return manifest, pilot._read(directory / "report.json")


def _copy_evaluation(source, destination, plan, split, candidate, source_sha256):
    source, destination = Path(source), Path(destination)
    manifest, report = _evaluation_source(source, candidate, plan, split)
    _validate_report(report, plan, split)
    if (
        manifest.get("candidate_artifact_sha256") != _sha256(candidate)
        or manifest.get("evaluation_split") != split
        or manifest.get("nominal_kind") != "friction"
        or manifest.get("purpose") != "rl_residual"
        or manifest.get("source_and_assets_sha256") != source_sha256
    ):
        raise ValueError("evaluation candidate/split identity differs")
    for source_name, public_name in (
        ("manifest.json", "source_manifest.json"),
        ("COMPLETE", "source_COMPLETE"),
        ("report.json", "report.json"),
    ):
        _copy(source / source_name, destination / public_name)
    _copy(source.with_suffix(".protocol.json"), destination.with_suffix(".protocol.json"))
    _copy(source.with_suffix(".protocol.sha256"), destination.with_suffix(".protocol.sha256"))
    return report


def _public_evaluation(directory, plan, split, candidate, source_sha256):
    manifest, report = _evaluation_source(
        directory, candidate, plan, split, public=True
    )
    _validate_report(report, plan, split)
    if (
        manifest.get("candidate_artifact_sha256") != _sha256(candidate)
        or manifest.get("evaluation_split") != split
        or manifest.get("nominal_kind") != "friction"
        or manifest.get("purpose") != "rl_residual"
        or manifest.get("source_and_assets_sha256") != source_sha256
    ):
        raise ValueError("public evaluation candidate/split identity differs")
    return report


def _evaluation_name(arm, seed, split):
    return f"{arm}_seed{seed}_ep32_{split}"


def _reports(directory, plan, split):
    result = {}
    for arm in ("fresh", "transfer"):
        for seed in plan["seeds"]:
            candidate = (
                Path(directory) / "fresh_control" / f"seed{seed}" / "checkpoint_ep032.json"
                if arm == "fresh"
                else Path(directory) / f"transfer_seed{seed}" / "checkpoint_ep032.json"
            )
            evaluation = Path(directory) / "evaluations" / _evaluation_name(arm, seed, split)
            control = next(row for row in plan["fresh_controls"] if row["seed"] == seed)
            result[(arm, seed)] = _public_evaluation(
                evaluation, plan, split, candidate, control["source_and_assets_sha256"]
            )
    return result


def _recompute_comparison(directory, plan, comparison, reports, training, zero_audits):
    expected_ids = _case_ids(plan, "validation")
    rows, validation_safe = [], True
    for seed in plan["seeds"]:
        control = next(row for row in plan["fresh_controls"] if row["seed"] == seed)
        transfer_report, fresh_report = reports[("transfer", seed)], reports[("fresh", seed)]
        control = next(row for row in plan["fresh_controls"] if row["seed"] == seed)
        if fresh_report != control["validation_report"]:
            raise ValueError("public fresh validation report differs from frozen control")
        eligible = pilot._selection_eligible(transfer_report, expected_ids, plan["thresholds"])
        no_failed = training[seed]["failed_episodes"] == 0
        paired = protocol._paired_tangent_delta(transfer_report, fresh_report)
        validation_safe = validation_safe and eligible and no_failed and paired[
            "all_metrics_available"
        ]
        rows.append(
            {
                "seed": seed,
                "transfer_validation": transfer_report,
                "fresh_validation": fresh_report,
                "transfer_validation_eligible": eligible,
                "no_failed_training_episodes": no_failed,
                "paired_tangent_delta_transfer_minus_fresh_mm": paired,
                "transfer_candidate_sha256": _sha256(
                    Path(directory) / f"transfer_seed{seed}" / "checkpoint_ep032.json"
                ),
                "fresh_candidate_sha256": _sha256(
                    Path(directory) / "fresh_control" / f"seed{seed}" / "checkpoint_ep032.json"
                ),
                "episode0_zero_checkpoint_audit": zero_audits[seed],
            }
        )
    consistent = validation_safe and all(
        row["paired_tangent_delta_transfer_minus_fresh_mm"]["mean"] < 0 for row in rows
    )
    expected = {
        "schema": "surface_ppo_bc_hidden_transfer_comparison_v1",
        "plan_sha256": _sha256(Path(directory) / "plan.json"),
        "fixed_checkpoint_episode": 32,
        "checkpoint_selected_on_validation": False,
        "seeds": rows,
        "validation_safe_for_development": validation_safe,
        "consistent_transfer_gain": consistent,
        "retained_arm": "bc_hidden_transfer_ep32" if consistent else "fresh_ep32",
        "interpretation_if_not_consistent": "retain fresh control; this does not invalidate RL",
    }
    if comparison != expected:
        raise ValueError("frozen PPO comparison does not recompute from bound reports")
    return expected


def _terminal_state(run):
    run = Path(run)
    failed = sorted(path for path in run.glob("transfer_seed*/FAILED") if path.is_file())
    if failed:
        if any((run / name).exists() for name in ("COMPLETE", "VALIDATION_VETO", "summary.json")):
            raise ValueError("failed PPO evidence conflicts with a later terminal state")
        return "training_failed"
    if (run / "COMPLETE").exists():
        if (run / "VALIDATION_VETO").exists():
            raise ValueError("complete PPO evidence also contains a validation veto")
        return "complete"
    if (run / "VALIDATION_VETO").exists():
        if not (run / "comparison.json").exists() or (run / "summary.json").exists():
            raise ValueError("validation veto has inconsistent artifacts")
        if (run / "VALIDATION_VETO").read_text().strip() != _sha256(
            run / "comparison.json"
        ):
            raise ValueError("validation veto does not bind comparison")
        return "validation_veto"
    raise ValueError("PPO transfer has no publishable terminal state")


def _copy_training_source(run, staging, plan, seed):
    source, destination = Path(run) / f"transfer_seed{seed}", staging / f"transfer_seed{seed}"
    manifest = _validate_training(source, plan, seed)
    for name in ("manifest.json", "COMPLETE", "checkpoint_ep000.json", "checkpoint_ep016.json", "checkpoint_ep032.json"):
        public_name = {"manifest.json": "source_manifest.json", "COMPLETE": "source_COMPLETE"}.get(name, name)
        _copy(source / name, destination / public_name)
    _write(
        destination / "training_curve.json",
        {"seed": seed, "runs": manifest["runs"], "updates": manifest["updates"]},
    )
    return manifest


def _copy_failed_source(source, destination):
    source, destination = Path(source), Path(destination)
    failure = pilot._read(source / "failure.json")
    if (source / "FAILED").read_text().strip() != _sha256(source / "failure.json"):
        raise ValueError("FAILED marker does not bind failure record")
    for name, digest in failure.get("existing_artifact_sha256", {}).items():
        if _sha256(source / name) != digest:
            raise ValueError("partial failure artifact changed")
    for path in sorted(source.iterdir()):
        if path.is_file():
            _copy(path, destination / path.name)


def _copy_controls(parent, bc_transfer, staging, plan):
    parent, bc_transfer = Path(parent), Path(bc_transfer)
    baseline = bc_transfer / "baseline_report.json"
    if _sha256(baseline) != plan["benchmark_sha256"]["report.json"]:
        raise ValueError("BC publication baseline differs from frozen PPO benchmark")
    _copy(baseline, staging / "baseline_report.json")
    for seed in plan["seeds"]:
        fresh_source = parent / Path(
            next(row for row in plan["fresh_controls"] if row["seed"] == seed)["candidate"]
        ).parent
        _validate_fresh(fresh_source, plan, seed)
        fresh_destination = staging / "fresh_control" / f"seed{seed}"
        for name, public_name in (
            ("checkpoint_ep032.json", "checkpoint_ep032.json"),
            ("manifest.json", "source_manifest.json"),
            ("COMPLETE", "source_COMPLETE"),
        ):
            _copy(fresh_source / name, fresh_destination / public_name)

        selected = plan["selected_bc_candidates"][str(seed)]
        bc_source = bc_transfer / Path(selected["candidate"]).parent
        public_bc = (bc_source / "source_manifest.json").exists()
        _bc_candidate_audit(bc_source, plan, seed, public=public_bc)
        bc_destination = staging / "bc_initialization" / f"seed{seed}"
        manifest_name = "source_manifest.json" if public_bc else "manifest.json"
        marker_name = "source_COMPLETE" if public_bc else "COMPLETE"
        for name, public_name in (
            ("selected_model.json", "selected_model.json"),
            ("report.json", "report.json"),
            (manifest_name, "source_manifest.json"),
            (marker_name, "source_COMPLETE"),
        ):
            _copy(bc_source / name, bc_destination / public_name)


def _copy_evaluations(run, parent, staging, plan, state):
    run, parent = Path(run), Path(parent)
    reports = {}
    for seed in plan["seeds"]:
        control = next(row for row in plan["fresh_controls"] if row["seed"] == seed)
        fresh_source = parent / "evaluations" / f"ppo_seed{seed}_ep32_validation"
        transfer_source = run / "evaluations" / _evaluation_name("transfer", seed, "validation")
        for arm, source in (("fresh", fresh_source), ("transfer", transfer_source)):
            destination = staging / "evaluations" / _evaluation_name(arm, seed, "validation")
            candidate = (
                staging / "fresh_control" / f"seed{seed}" / "checkpoint_ep032.json"
                if arm == "fresh"
                else staging / f"transfer_seed{seed}" / "checkpoint_ep032.json"
            )
            reports[(arm, seed, "validation")] = _copy_evaluation(
                source,
                destination,
                plan,
                "validation",
                candidate,
                control["source_and_assets_sha256"],
            )
    if state == "complete":
        for arm in ("fresh", "transfer"):
            for seed in plan["seeds"]:
                source = run / "evaluations" / _evaluation_name(
                    arm, seed, "development_test"
                )
                destination = staging / "evaluations" / source.name
                candidate = (
                    staging / "fresh_control" / f"seed{seed}" / "checkpoint_ep032.json"
                    if arm == "fresh"
                    else staging / f"transfer_seed{seed}" / "checkpoint_ep032.json"
                )
                reports[(arm, seed, "development_test")] = _copy_evaluation(
                    source,
                    destination,
                    plan,
                    "development_test",
                    candidate,
                    next(
                        row["source_and_assets_sha256"]
                        for row in plan["fresh_controls"]
                        if row["seed"] == seed
                    ),
                )
    return reports


def _copy_partial_evaluations(run, staging, plan):
    copied = []
    for seed in plan["seeds"]:
        source = Path(run) / "evaluations" / _evaluation_name("transfer", seed, "validation")
        if not (source / "COMPLETE").exists():
            continue
        destination = staging / "evaluations" / source.name
        candidate = staging / f"transfer_seed{seed}" / "checkpoint_ep032.json"
        control = next(row for row in plan["fresh_controls"] if row["seed"] == seed)
        _copy_evaluation(
            source,
            destination,
            plan,
            "validation",
            candidate,
            control["source_and_assets_sha256"],
        )
        copied.append(seed)
    return copied


def _representatives(directory, plan):
    directory = Path(directory)
    index = pilot._read(directory / "representative_traces.json")
    case_id = _case_ids(plan, "development_test")[0]
    expected = [(arm, plan["seeds"][0], case_id) for arm in ("fresh", "transfer")]
    if [(row["arm"], row["seed"], row["case_id"]) for row in index["records"]] != expected:
        raise ValueError("representative trace choice differs")
    results = []
    for record in index["records"]:
        trace = directory / record["file"]
        manifest = pilot._read(trace.parent / "source_manifest.json")
        if _sha256(trace) != manifest["artifact_sha256"].get(trace.name):
            raise ValueError("representative trace differs from source manifest")
        report = pilot._read(trace.parent / "report.json")
        expected_row = next(row for row in report["runs"] if row["case_id"] == case_id)
        case = next(case for case in plan["cases"] if case["case_id"] == case_id)
        with np.load(trace, allow_pickle=False) as stored:
            observed = audit_trace({key: stored[key] for key in stored.files}, case)
            physics_steps = len(stored["time"])
        differences = []
        for metric in METRICS:
            left, right = observed[metric], expected_row[metric]
            if left is None or right is None:
                if left != right:
                    raise ValueError(f"representative metric availability differs: {metric}")
            elif not np.isclose(left, right, rtol=0, atol=1e-10):
                raise ValueError(f"representative metric differs: {metric}")
            else:
                differences.append(abs(left - right))
        if physics_steps != expected_row["physics_steps"]:
            raise ValueError("representative physics step count differs")
        results.append(
            {
                "arm": record["arm"],
                "seed": record["seed"],
                "case_id": case_id,
                "physics_steps": physics_steps,
                "maximum_absolute_metric_difference": max(differences, default=0.0),
            }
        )
    return results


def _readme(state, comparison):
    def number(value):
        return "不可用" if value is None or not np.isfinite(value) else f"{value:.3f}"

    lines = [
        "# BC 隐藏层初始化的 PPO 对照",
        "",
        "该归档比较固定第 32 回合的历史 fresh PPO 与 BC 隐藏层初始化 PPO。",
        "validation/development_test 都是已公开开发域，不是新盲测；结果不能推出真机表现。",
        f"终态：`{state}`。只比较第 32 回合，没有从 0/16/32 中挑选检查点。",
        "",
    ]
    if comparison is not None:
        lines += [
            f"验证允许进入开发集：`{comparison['validation_safe_for_development']}`；",
            f"三种子一致改善：`{comparison['consistent_transfer_gain']}`；",
            f"按冻结规则保留：`{comparison['retained_arm']}`。",
            "",
            "| seed | fresh 验证切向 RMSE [mm] | transfer 验证切向 RMSE [mm] | 配对差值 [mm] |",
            "|---:|---:|---:|---:|",
        ]
        for row in comparison["seeds"]:
            fresh = [item.get("tangent_rmse_mm") for item in row["fresh_validation"]["runs"]]
            transferred = [
                item.get("tangent_rmse_mm") for item in row["transfer_validation"]["runs"]
            ]
            fresh_mean = None if any(value is None for value in fresh) else float(np.mean(fresh))
            transfer_mean = (
                None if any(value is None for value in transferred) else float(np.mean(transferred))
            )
            delta = row["paired_tangent_delta_transfer_minus_fresh_mm"]["mean"]
            lines.append(
                f"| {row['seed']} | {number(fresh_mean)} | {number(transfer_mean)} | "
                f"{number(delta)} |"
            )
        lines.append("")
    lines += [
        "训练 manifest 保存完整曲线；公开副本仅保留每个 transfer 种子的 0/16/32 权重。",
        "成功完成时附 fresh/transfer 各一份首种子、首开发 case 的完整物理 NPZ；",
        "其他训练和评价原始轨迹保留在本地。因此 `all_raw_traces_distributed=false`。",
        "",
        "实验语义和后续步骤见 [BC 到残差 RL](../../docs/bc_to_residual_rl.md)。",
        "",
        "```bash",
        "python -m tools.publish_surface_ppo_transfer --audit results/franka_surface_ppo_transfer",
        "```",
        "",
    ]
    return "\n".join(lines)


def audit(directory):
    """Audit a compact archive without Torch or dynamics reruns."""
    directory = Path(directory)
    root_manifest = pilot._verified_manifest(directory)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    if actual != set(root_manifest["artifact_sha256"]) | {"manifest.json", "COMPLETE"}:
        raise ValueError("public archive contains missing or unlisted files")
    plan = pilot._read(directory / "plan.json")
    state = root_manifest.get("outcome")
    if (
        root_manifest.get("schema") != "surface_ppo_transfer_public_v1"
        or root_manifest.get("plan_sha256") != _sha256(directory / "plan.json")
        or (directory / "PLAN_SHA256").read_text().strip() != root_manifest["plan_sha256"]
        or root_manifest.get("new_holdout") is not False
        or root_manifest.get("all_raw_traces_distributed") is not False
        or state not in {"complete", "validation_veto", "training_failed"}
        or plan.get("schema") != "surface_ppo_bc_hidden_transfer_plan_v1"
        or plan.get("seeds") != [11, 29, 47]
        or plan.get("comparison", {}).get("checkpoint_episode") != 32
        or plan.get("comparison", {}).get("checkpoint_selection") is not False
    ):
        raise ValueError("public PPO transfer manifest/plan identity differs")
    if _sha256(directory / "baseline_report.json") != plan["benchmark_sha256"]["report.json"]:
        raise ValueError("public baseline report differs from frozen benchmark")
    for seed in plan["seeds"]:
        _validate_fresh(directory / "fresh_control" / f"seed{seed}", plan, seed, public=True)
        bc_candidate = _bc_candidate_audit(
            directory / "bc_initialization" / f"seed{seed}", plan, seed, public=True
        )
        transfer_dir = directory / f"transfer_seed{seed}"
        if (transfer_dir / "source_COMPLETE").exists():
            _validate_training(transfer_dir, plan, seed, public=True)
            _zero_and_trunk_audit(
                transfer_dir / "checkpoint_ep000.json", bc_candidate, plan, seed
            )
    if state == "training_failed":
        failed = list(directory.glob("transfer_seed*/FAILED"))
        if not failed:
            raise ValueError("training_failed publication lacks FAILED evidence")
        for marker in failed:
            root = marker.parent
            failure = pilot._read(root / "failure.json")
            if (
                failure.get("schema") != "surface_ppo_bc_hidden_transfer_failure_v1"
                or marker.read_text().strip() != _sha256(root / "failure.json")
                or any(
                    _sha256(root / name) != digest
                    for name, digest in failure.get("existing_artifact_sha256", {}).items()
                )
            ):
                raise ValueError("public failed-training evidence differs")
        observed_partial = []
        for seed in plan["seeds"]:
            evaluation = directory / "evaluations" / _evaluation_name(
                "transfer", seed, "validation"
            )
            if not evaluation.exists():
                continue
            candidate = directory / f"transfer_seed{seed}" / "checkpoint_ep032.json"
            control = next(row for row in plan["fresh_controls"] if row["seed"] == seed)
            _public_evaluation(
                evaluation,
                plan,
                "validation",
                candidate,
                control["source_and_assets_sha256"],
            )
            observed_partial.append(seed)
        if observed_partial != root_manifest.get("partial_validation_seeds"):
            raise ValueError("partial validation evidence list differs")
        evaluation_root = directory / "evaluations"
        observed_directories = (
            {path.name for path in evaluation_root.iterdir() if path.is_dir()}
            if evaluation_root.exists()
            else set()
        )
        expected_directories = {
            _evaluation_name("transfer", seed, "validation") for seed in observed_partial
        }
        if observed_directories != expected_directories:
            raise ValueError("failed-run partial evaluation cohort differs")
        return {
            "hashes_match": True,
            "outcome": state,
            "comparison_recomputed": False,
            "representative_physics_traces_recomputed": 0,
        }

    training, zeros = {}, {}
    for seed in plan["seeds"]:
        transfer_dir = directory / f"transfer_seed{seed}"
        training[seed] = _validate_training(transfer_dir, plan, seed, public=True)
        zeros[seed] = _zero_and_trunk_audit(
            transfer_dir / "checkpoint_ep000.json",
            directory / "bc_initialization" / f"seed{seed}" / "selected_model.json",
            plan,
            seed,
        )
    validation = _reports(directory, plan, "validation")
    comparison = pilot._read(directory / "comparison.json")
    _recompute_comparison(directory, plan, comparison, validation, training, zeros)
    if state == "validation_veto":
        if comparison["validation_safe_for_development"] is not False:
            raise ValueError("validation-veto archive contains a safe comparison")
        if (directory / "VALIDATION_VETO").read_text().strip() != _sha256(
            directory / "comparison.json"
        ):
            raise ValueError("public validation veto differs from comparison")
        if (directory / "summary.json").exists() or list(
            (directory / "evaluations").glob("*_development_test")
        ):
            raise ValueError("validation-veto archive contains phantom development evidence")
        expected_directories = {
            _evaluation_name(arm, seed, "validation")
            for arm in ("fresh", "transfer")
            for seed in plan["seeds"]
        }
        observed_directories = {
            path.name for path in (directory / "evaluations").iterdir() if path.is_dir()
        }
        if observed_directories != expected_directories:
            raise ValueError("validation-veto evaluation directory cohort differs")
        return {
            "hashes_match": True,
            "outcome": state,
            "comparison_recomputed": True,
            "representative_physics_traces_recomputed": 0,
        }

    summary = pilot._read(directory / "summary.json")
    if (
        (directory / "run_COMPLETE").read_text().strip() != _sha256(directory / "summary.json")
        or summary.get("schema") != "surface_ppo_bc_hidden_transfer_summary_v1"
        or summary.get("new_holdout") is not False
        or summary.get("plan_sha256") != _sha256(directory / "plan.json")
        or summary.get("comparison_sha256") != _sha256(directory / "comparison.json")
        or summary.get("comparison") != comparison
        or comparison.get("validation_safe_for_development") is not True
        or summary.get("development_use")
        != "repeated public comparison; not independent evidence or selection"
    ):
        raise ValueError("completed PPO transfer summary identity differs")
    development = _reports(directory, plan, "development_test")
    baseline = pilot._read(directory / "baseline_report.json")
    expected = [(arm, seed) for arm in ("fresh", "transfer") for seed in plan["seeds"]]
    items = summary.get("development_test", [])
    expected_directories = {
        _evaluation_name(arm, seed, split)
        for split in ("validation", "development_test")
        for arm in ("fresh", "transfer")
        for seed in plan["seeds"]
    }
    observed_directories = {
        path.name for path in (directory / "evaluations").iterdir() if path.is_dir()
    }
    if observed_directories != expected_directories:
        raise ValueError("completed evaluation directory cohort differs")
    if [(item["arm"], item["seed"]) for item in items] != expected:
        raise ValueError("development cohort is incomplete or reordered")
    for item in items:
        if (
            item["report"] != development[(item["arm"], item["seed"])]
            or item.get("paired_baselines") != pilot.paired_baselines(item["report"], baseline)
            or item.get("public_comparison") is not True
            or item.get("independent_new_case") is not False
        ):
            raise ValueError("development summary differs from bound evaluation")
    representatives = _representatives(directory, plan)
    if pilot._read(directory / "representative_physics_audit.json") != representatives:
        raise ValueError("stored representative audit differs from recomputation")
    return {
        "hashes_match": True,
        "outcome": state,
        "comparison_recomputed": True,
        "representative_physics_traces_recomputed": len(representatives),
    }


def publish(run, parent, bc_transfer, output, *, expected_plan_sha256):
    run, parent, bc_transfer, output = map(Path, (run, parent, bc_transfer, output))
    publisher_sha256 = _sha256(Path(__file__))
    if output.exists():
        raise ValueError("publication output must be new")
    plan_path = run / "plan.json"
    if (
        _sha256(plan_path) != expected_plan_sha256
        or (run / "PLAN_SHA256").read_text().strip() != expected_plan_sha256
    ):
        raise ValueError("PPO transfer plan differs from preserved pre-fit SHA")
    plan, state = pilot._read(plan_path), _terminal_state(run)
    repository = Path(__file__).resolve().parents[1]
    if (
        plan.get("schema") != "surface_ppo_bc_hidden_transfer_plan_v1"
        or plan.get("seeds") != [11, 29, 47]
        or plan.get("comparison", {}).get("checkpoint_episode") != 32
        or plan.get("comparison", {}).get("checkpoint_selection") is not False
        or any(_sha256(repository / name) != digest for name, digest in plan["source_sha256"].items())
    ):
        raise ValueError("PPO transfer plan does not declare the fixed three-seed episode-32 study")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-ppo-transfer-public-", dir=output.parent) as temp:
        staging = Path(temp) / "public"
        staging.mkdir()
        _copy(run / "plan.json", staging / "plan.json")
        _copy(run / "PLAN_SHA256", staging / "PLAN_SHA256")
        _copy_controls(parent, bc_transfer, staging, plan)

        training = {}
        for seed in plan["seeds"]:
            source = run / f"transfer_seed{seed}"
            if (source / "COMPLETE").exists():
                training[seed] = _copy_training_source(run, staging, plan, seed)
            elif (source / "FAILED").exists():
                _copy_failed_source(source, staging / source.name)
            elif state != "training_failed":
                raise ValueError("terminal run lacks a declared training seed")

        comparison, partial_validation_seeds = None, []
        if state != "training_failed":
            reports = _copy_evaluations(run, parent, staging, plan, state)
            _copy(run / "comparison.json", staging / "comparison.json")
            comparison = pilot._read(staging / "comparison.json")
            zeros = {
                seed: _zero_and_trunk_audit(
                    staging / f"transfer_seed{seed}" / "checkpoint_ep000.json",
                    staging / "bc_initialization" / f"seed{seed}" / "selected_model.json",
                    plan,
                    seed,
                )
                for seed in plan["seeds"]
            }
            _recompute_comparison(
                staging,
                plan,
                comparison,
                {(arm, seed): reports[(arm, seed, "validation")] for arm in ("fresh", "transfer") for seed in plan["seeds"]},
                training,
                zeros,
            )
            if state == "validation_veto":
                _copy(run / "VALIDATION_VETO", staging / "VALIDATION_VETO")
            else:
                _copy(run / "summary.json", staging / "summary.json")
                _copy(run / "COMPLETE", staging / "run_COMPLETE")
                case_id, seed = _case_ids(plan, "development_test")[0], plan["seeds"][0]
                records = []
                for arm in ("fresh", "transfer"):
                    evaluation = staging / "evaluations" / _evaluation_name(
                        arm, seed, "development_test"
                    )
                    report = reports[(arm, seed, "development_test")]
                    row = next(item for item in report["runs"] if item["case_id"] == case_id)
                    source = run / "evaluations" / evaluation.name / row["trace_file"]
                    _copy(source, evaluation / row["trace_file"])
                    records.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "case_id": case_id,
                            "file": str(
                                Path("evaluations") / evaluation.name / row["trace_file"]
                            ),
                        }
                    )
                _write(
                    staging / "representative_traces.json",
                    {"selection": "each arm; first seed; first development case", "records": records},
                )
                _write(
                    staging / "representative_physics_audit.json",
                    _representatives(staging, plan),
                )
        else:
            partial_validation_seeds = _copy_partial_evaluations(run, staging, plan)

        (staging / "README.md").write_text(_readme(state, comparison), encoding="utf-8")
        root_manifest = {
            "schema": "surface_ppo_transfer_public_v1",
            "plan_sha256": expected_plan_sha256,
            "outcome": state,
            "new_holdout": False,
            "all_raw_traces_distributed": False,
            "partial_validation_seeds": partial_validation_seeds,
            "publisher_sha256": publisher_sha256,
            "artifact_sha256": {
                str(path.relative_to(staging)): _sha256(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            },
        }
        _write(staging / "manifest.json", root_manifest)
        (staging / "COMPLETE").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="utf-8"
        )
        audit(staging)
        if publisher_sha256 != _sha256(Path(__file__)):
            raise RuntimeError("publisher source changed during publication")
        if output.exists():
            raise FileExistsError("publication destination appeared")
        os.rename(staging, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--bc-transfer", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args(argv)
    if args.audit:
        print(json.dumps(audit(args.audit), sort_keys=True))
        return
    if not all((args.run, args.parent, args.bc_transfer, args.output, args.expected_plan_sha256)):
        parser.error(
            "publication needs --run, --parent, --bc-transfer, --output and --expected-plan-sha256"
        )
    result = publish(
        args.run,
        args.parent,
        args.bc_transfer,
        args.output,
        expected_plan_sha256=args.expected_plan_sha256,
    )
    print(json.dumps({"output": str(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
