import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import publish_surface_ppo_transfer as publication
from tools import surface_learning_pilot as pilot


def _cases():
    return [
        {"case_id": f"{split}{index}", "split": split, "group_id": f"g{index}"}
        for split in ("train", "validation", "development_test")
        for index in range(2)
    ]


def _ppo():
    return {
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


def _plan():
    runtime = {
        "python_version": "3",
        "numpy_version": "2",
        "torch_version": "2",
        "torch_git_version": "git",
        "mujoco_version": "3",
        "gymnasium_version": "1",
        "device": "cpu",
        "dtype": "float64",
        "torch_num_threads": 1,
        "deterministic_algorithms": True,
    }
    fresh = [
        {
            "seed": seed,
            "candidate": f"ppo_seed{seed}/checkpoint_ep032.json",
            "candidate_sha256": f"fresh-{seed}",
            "training_manifest_sha256": f"manifest-{seed}",
            "training_complete_sha256": f"complete-{seed}",
            "torch_seed": seed + 100,
            "schedule_case_ids": ["train0"] * 32,
            "reset_seeds": list(range(32)),
            "actual_simulation_seeds": list(range(100, 132)),
            "source_and_assets_sha256": {"source": "same"},
            "validation_report": _report(4.0),
        }
        for seed in (11, 29, 47)
    ]
    return {
        "schema": "surface_ppo_bc_hidden_transfer_plan_v1",
        "seeds": [11, 29, 47],
        "cases": _cases(),
        "thresholds": pilot.THRESHOLDS,
        "metrics": ["tangent_rmse_mm"],
        "ppo": _ppo(),
        "runtime": runtime,
        "source_sha256": {
            "tools/train_surface_ppo_transfer.py": "transfer",
            "tools/train_surface_ppo.py": "fresh",
            "tools/surface_mlp_actor.py": "runner",
        },
        "selected_bc_arm": "drop_previous_residual",
        "selected_bc_candidates": {
            str(seed): {
                "candidate": f"drop_previous_residual_seed{seed}/selected_model.json",
                "candidate_sha256": f"bc-{seed}",
            }
            for seed in (11, 29, 47)
        },
        "bc_records": {"candidate_records": {}},
        "fresh_controls": fresh,
        "comparison": {"checkpoint_episode": 32, "checkpoint_selection": False},
    }


def _report(tangent, split="validation", *, missing=False):
    case_ids = [f"{split}{index}" for index in range(2)]
    runs = []
    for index, case_id in enumerate(case_ids):
        row = {
            "case_id": case_id,
            "group_id": f"g{index}",
            "split": split,
            "simulation_seed": 100 + index,
            "episode_success": not missing,
            "independent_physical_gates_met": not missing,
            "max_contact_loss_s": 0.0,
            **{name: rule["value"] for name, rule in pilot.THRESHOLDS.items()},
        }
        row["tangent_rmse_mm"] = None if missing and index == 0 else tangent
        row["force_rmse_n"] = None if missing and index == 0 else 0.2
        runs.append(row)
    return {
        "evaluation_split": split,
        "evaluation_case_ids": case_ids,
        "expected_case_count": 2,
        "acceptance_met": not missing,
        "all_metrics_available": not missing,
        "all_episodes_succeeded": not missing,
        "runs": runs,
    }


def _training_manifest(plan, seed=11):
    control = next(row for row in plan["fresh_controls"] if row["seed"] == seed)
    runs = [
        {
            "episode": episode,
            "case_id": "train0",
            "requested_reset_seed": episode - 1,
            "actual_simulation_seed": 99 + episode,
            "failed": False,
            "physics_steps": 10,
        }
        for episode in range(1, 33)
    ]
    parameters = deepcopy(plan["ppo"])
    parameters["fixed_std"] = parameters.pop("std")
    return {
        "schema": "surface_ppo_bc_hidden_transfer_v1",
        "seed": seed,
        "nominal_kind": "friction",
        "purpose": "rl_residual",
        "hyperparameters": parameters,
        "checkpoints": {
            "0": "checkpoint_ep000.json",
            "16": "checkpoint_ep016.json",
            "32": "checkpoint_ep032.json",
        },
        "artifact_sha256": {},
        "runs": runs,
        "updates": [{"after_episode": episode} for episode in range(4, 33, 4)],
        "failed_episodes": 0,
        "actual_physics_steps": 320,
        "torch_seed": control["torch_seed"],
        "runtime": {"before": plan["runtime"], "after": plan["runtime"]},
        "script_sha256": {
            "train_surface_ppo_transfer.py": "transfer",
            "train_surface_ppo.py": "fresh",
            "surface_mlp_actor.py": "runner",
        },
        "source_and_assets_sha256": control["source_and_assets_sha256"],
        "initialization": {
            "inherited_bc_candidate_sha256": plan["selected_bc_candidates"][str(seed)][
                "candidate_sha256"
            ],
            "fresh_critic_exactly_unchanged": True,
        },
        "transfer_semantics": {
            "bc_action_head_copied": False,
            "bc_actions_converted_to_friction_residuals": False,
            "runtime_observation_inputs": "all 49 measured inputs; no BC input mask",
            "actor_hidden_trunk_trainable_during_rl": True,
            "optimizer_scope": "all actor and critic parameters",
        },
        "data_use": {
            "optimization_split": "train",
            "validation_loaded": False,
            "development_test_loaded": False,
            "selection_performed": False,
        },
    }


def _write_public_training(tmp_path, plan, manifest):
    root = tmp_path / "transfer_seed11"
    root.mkdir()
    for episode in (0, 16, 32):
        path = root / f"checkpoint_ep{episode:03d}.json"
        path.write_text(f"checkpoint {episode}\n", encoding="utf-8")
        manifest["artifact_sha256"][path.name] = publication._sha256(path)
    pilot._write_new(root / "source_manifest.json", manifest)
    (root / "source_COMPLETE").write_text(
        publication._sha256(root / "source_manifest.json"), encoding="utf-8"
    )
    pilot._write_new(
        root / "training_curve.json",
        {"seed": 11, "runs": manifest["runs"], "updates": manifest["updates"]},
    )
    return root


@pytest.mark.parametrize("alter", ["episode", "reset", "runtime", "curve"])
def test_training_audit_enforces_budget_pairing_runtime_and_curve(tmp_path, alter):
    plan = _plan()
    manifest = _training_manifest(plan)
    root = _write_public_training(tmp_path, plan, manifest)
    publication._validate_training(root, plan, 11, public=True)
    if alter == "curve":
        curve = pilot._read(root / "training_curve.json")
        curve["runs"].pop()
        (root / "training_curve.json").write_text(json.dumps(curve), encoding="utf-8")
    else:
        changed = pilot._read(root / "source_manifest.json")
        if alter == "episode":
            changed["runs"][0]["episode"] = 9
        elif alter == "reset":
            changed["runs"][0]["requested_reset_seed"] = 999
        else:
            changed["runtime"]["after"]["dtype"] = "float32"
        (root / "source_manifest.json").write_text(json.dumps(changed), encoding="utf-8")
        (root / "source_COMPLETE").write_text(
            publication._sha256(root / "source_manifest.json"), encoding="utf-8"
        )
    with pytest.raises(ValueError):
        publication._validate_training(root, plan, 11, public=True)


def test_zero_audit_requires_exact_bc_hidden_layers(monkeypatch, tmp_path):
    plan = _plan()
    runner = publication.surface_mlp_actor.current_runner_identity()
    plan["source_sha256"]["tools/surface_mlp_actor.py"] = runner["runner_sha256"]
    for control in plan["fresh_controls"]:
        control["source_and_assets_sha256"] = runner["package_source_and_assets_sha256"]
    recorded_runner = {
        **runner,
        "python_version": plan["runtime"]["python_version"],
        "numpy_version": plan["runtime"]["numpy_version"],
        "mujoco_version": plan["runtime"]["mujoco_version"],
        "gymnasium_version": plan["runtime"]["gymnasium_version"],
    }
    rng = np.random.default_rng(4)
    bc_layers = [
        (rng.normal(size=(32, 49)), rng.normal(size=32)),
        (rng.normal(size=(32, 32)), rng.normal(size=32)),
        (rng.normal(size=(3, 32)), rng.normal(size=3)),
    ]
    episode_layers = [(w.copy(), b.copy()) for w, b in bc_layers]
    episode_layers[-1] = (np.zeros((3, 32)), np.zeros(3))
    artifacts = {
        "checkpoint": SimpleNamespace(
            payload={
                "kind": publication.surface_mlp_actor.FORMAT,
                "activation": "tanh",
                "output_activation": "tanh",
                "layers": episode_layers,
                "runner_identity": recorded_runner,
            }
        ),
        "bc": SimpleNamespace(
            payload={
                "kind": publication.surface_mlp_actor.FORMAT,
                "activation": "tanh",
                "output_activation": "tanh",
                "layers": bc_layers,
                "runner_identity": recorded_runner,
            }
        ),
    }
    monkeypatch.setattr(publication, "policy_contract", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        publication,
        "load_policy_artifact",
        lambda path, **kwargs: artifacts[Path(path).name],
    )
    monkeypatch.setattr(
        publication.surface_mlp_actor,
        "_dense_layers",
        lambda layers: layers,
    )
    (tmp_path / "checkpoint").write_text("rl", encoding="utf-8")
    (tmp_path / "bc").write_text("bc", encoding="utf-8")
    result = publication._zero_and_trunk_audit(
        tmp_path / "checkpoint", tmp_path / "bc", plan, 11
    )
    assert result["exact_zero"] and result["hidden_layer_exact_equal_to_pinned_bc"] == [True, True]
    episode_layers[0][0][0, 0] += 1
    with pytest.raises(ValueError, match="pinned BC trunk"):
        publication._zero_and_trunk_audit(
            tmp_path / "checkpoint", tmp_path / "bc", plan, 11
        )


def _comparison_fixture(tmp_path, plan, reports, training, zeros):
    for seed in plan["seeds"]:
        for root, text in (
            (tmp_path / f"transfer_seed{seed}", "transfer"),
            (tmp_path / "fresh_control" / f"seed{seed}", "fresh"),
        ):
            root.mkdir(parents=True)
            (root / "checkpoint_ep032.json").write_text(f"{text} {seed}\n", encoding="utf-8")
    rows = []
    for seed in plan["seeds"]:
        paired = publication.protocol._paired_tangent_delta(
            reports[("transfer", seed)], reports[("fresh", seed)]
        )
        rows.append(
            {
                "seed": seed,
                "transfer_validation": reports[("transfer", seed)],
                "fresh_validation": reports[("fresh", seed)],
                "transfer_validation_eligible": True,
                "no_failed_training_episodes": True,
                "paired_tangent_delta_transfer_minus_fresh_mm": paired,
                "transfer_candidate_sha256": publication._sha256(
                    tmp_path / f"transfer_seed{seed}" / "checkpoint_ep032.json"
                ),
                "fresh_candidate_sha256": publication._sha256(
                    tmp_path / "fresh_control" / f"seed{seed}" / "checkpoint_ep032.json"
                ),
                "episode0_zero_checkpoint_audit": zeros[seed],
            }
        )
    pilot._write_new(tmp_path / "plan.json", plan)
    return {
        "schema": "surface_ppo_bc_hidden_transfer_comparison_v1",
        "plan_sha256": publication._sha256(tmp_path / "plan.json"),
        "fixed_checkpoint_episode": 32,
        "checkpoint_selected_on_validation": False,
        "seeds": rows,
        "validation_safe_for_development": True,
        "consistent_transfer_gain": True,
        "retained_arm": "bc_hidden_transfer_ep32",
        "interpretation_if_not_consistent": "retain fresh control; this does not invalidate RL",
    }


def test_comparison_is_recomputed_from_all_bound_reports(tmp_path):
    plan = _plan()
    reports = {
        (arm, seed): _report(3.0 if arm == "transfer" else 4.0)
        for arm in ("fresh", "transfer")
        for seed in plan["seeds"]
    }
    training = {seed: {"failed_episodes": 0} for seed in plan["seeds"]}
    zeros = {seed: {"exact_zero": True} for seed in plan["seeds"]}
    comparison = _comparison_fixture(tmp_path, plan, reports, training, zeros)
    publication._recompute_comparison(tmp_path, plan, comparison, reports, training, zeros)
    comparison["seeds"][0]["paired_tangent_delta_transfer_minus_fresh_mm"]["mean"] = -9
    with pytest.raises(ValueError, match="does not recompute"):
        publication._recompute_comparison(
            tmp_path, plan, comparison, reports, training, zeros
        )


def test_missing_validation_metric_produces_veto_not_a_filtered_mean():
    paired = publication.protocol._paired_tangent_delta(_report(3, missing=True), _report(4))
    assert paired["mean"] is None
    assert paired["all_metrics_available"] is False
    assert paired["per_case"][0] is None


def test_terminal_veto_rejects_phantom_summary(tmp_path):
    (tmp_path / "comparison.json").write_text("{}", encoding="utf-8")
    (tmp_path / "VALIDATION_VETO").write_text(
        publication._sha256(tmp_path / "comparison.json"), encoding="utf-8"
    )
    assert publication._terminal_state(tmp_path) == "validation_veto"
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        publication._terminal_state(tmp_path)


def test_root_manifest_rejects_altered_distributed_file(tmp_path):
    (tmp_path / "README.md").write_text("original", encoding="utf-8")
    pilot._write_new(
        tmp_path / "manifest.json",
        {"artifact_sha256": {"README.md": publication._sha256(tmp_path / "README.md")}},
    )
    (tmp_path / "COMPLETE").write_text(
        publication._sha256(tmp_path / "manifest.json"), encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("altered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        publication.audit(tmp_path)


def test_real_public_archive_audits_when_present():
    archive = Path(__file__).resolve().parents[1] / "results" / "franka_surface_ppo_transfer"
    if not archive.exists():
        pytest.skip("public PPO transfer archive is generated after formal completion")
    result = publication.audit(archive)
    assert result["hashes_match"] and result["comparison_recomputed"]
