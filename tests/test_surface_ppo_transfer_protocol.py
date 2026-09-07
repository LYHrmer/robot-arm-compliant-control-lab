from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import surface_learning_pilot as pilot
from tools import surface_ppo_transfer as protocol


def _cases():
    return [
        {"case_id": f"{split}{index}", "split": split, "group_id": f"g{index}"}
        for split in ("train", "validation", "development_test")
        for index in range(4)
    ]


def _report(split="validation", tangent=3.0):
    identifiers = [f"{split}{index}" for index in range(4)]
    rows = []
    for index, case_id in enumerate(identifiers):
        row = {
            "case_id": case_id, "group_id": f"g{index}", "split": split,
            "simulation_seed": 100 + index, "episode_success": True,
            "independent_physical_gates_met": True, "max_contact_loss_s": 0.0,
            "tangent_rmse_mm": tangent, "force_rmse_n": 0.2,
        }
        row.update({key: rule["value"] for key, rule in pilot.THRESHOLDS.items()})
        row["tangent_rmse_mm"], row["force_rmse_n"] = tangent, 0.2
        rows.append(row)
    return {
        "evaluation_split": split, "evaluation_case_ids": identifiers,
        "expected_case_count": 4, "acceptance_met": True,
        "all_metrics_available": True, "all_episodes_succeeded": True, "runs": rows,
    }


def _ppo():
    return {
        "episodes": 32, "episodes_per_update": 4, "checkpoint_episodes": [0, 16, 32],
        "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2,
        "learning_rate": 0.0003, "update_epochs": 4, "batch_size": 128,
        "max_grad_norm": 0.5, "std": 0.05, "target_kl": 0.02,
    }


def _plan(tmp_path):
    fresh, selected = [], {}
    for seed in protocol.SEEDS:
        candidate = tmp_path / f"fresh{seed}.json"
        candidate.write_text(f"fresh {seed}\n", encoding="utf-8")
        fresh.append({
            "seed": seed, "candidate": candidate.name,
            "candidate_sha256": pilot._sha256(candidate), "torch_seed": seed * 10,
            "schedule_case_ids": ["train0"] * 32, "reset_seeds": list(range(32)),
            "actual_simulation_seeds": list(range(100, 132)),
            "source_and_assets_sha256": {"source": "same"},
            "validation_report": _report(tangent=4.0),
        })
        bc = tmp_path / f"bc{seed}.json"
        bc.write_text(f"bc {seed}\n", encoding="utf-8")
        selected[str(seed)] = {
            "candidate": bc.name, "candidate_sha256": pilot._sha256(bc)
        }
    return {
        "seeds": list(protocol.SEEDS), "cases": _cases(), "thresholds": pilot.THRESHOLDS,
        "input_roots_relative_to_output": {"parent": ".", "bc_transfer": "."},
        "ppo": _ppo(), "fresh_controls": fresh,
        "selected_bc_candidates": selected,
        "source_sha256": {
            "tools/train_surface_ppo_transfer.py": "transfer",
            "tools/train_surface_ppo.py": "fresh",
            "tools/surface_mlp_actor.py": "runner",
        },
        "runtime": {
            "python_version": "3", "numpy_version": "2", "torch_version": "2",
            "torch_git_version": "git", "mujoco_version": "3",
            "gymnasium_version": "1", "device": "cpu", "dtype": "float64",
            "torch_num_threads": 1, "deterministic_algorithms": True,
        },
    }


def _complete_manifest(plan, seed=11):
    control = next(row for row in plan["fresh_controls"] if row["seed"] == seed)
    runs = [
        {
            "episode": episode, "case_id": case_id, "requested_reset_seed": reset,
            "actual_simulation_seed": simulation_seed, "physics_steps": 10,
            "failed": False,
        }
        for episode, (case_id, reset, simulation_seed) in enumerate(
            zip(
                control["schedule_case_ids"], control["reset_seeds"],
                control["actual_simulation_seeds"],
            ),
            start=1,
        )
    ]
    parameters = deepcopy(plan["ppo"])
    parameters["fixed_std"] = parameters.pop("std")
    return {
        "schema": "surface_ppo_bc_hidden_transfer_v1", "seed": seed,
        "nominal_kind": "friction", "purpose": "rl_residual",
        "case_plan": pilot.pilot_cases(plan, "friction"), "hyperparameters": parameters,
        "architecture": "BC 49-32-32 trunk; fresh zero 3 head; fresh value 49-32-32-1",
        "policy_contract": {"contract": True},
        "source_and_assets_sha256": control["source_and_assets_sha256"],
        "script_sha256": {
            "train_surface_ppo_transfer.py": "transfer",
            "train_surface_ppo.py": "fresh", "surface_mlp_actor.py": "runner",
        },
        "runtime": {"before": plan["runtime"], "after": plan["runtime"]},
        "versions": {key: plan["runtime"][key] for key in (
            "python_version", "numpy_version", "torch_version", "mujoco_version",
            "gymnasium_version",
        )},
        "initialization": {
            "inherited_bc_candidate_sha256": plan["selected_bc_candidates"][str(seed)][
                "candidate_sha256"
            ],
            "fresh_critic_exactly_unchanged": True,
            "episode0_zero_probe": {"max_abs_action": 0.0},
        },
        "transfer_semantics": {
            "bc_action_head_copied": False,
            "bc_actions_converted_to_friction_residuals": False,
            "runtime_observation_inputs": "all 49 measured inputs; no BC input mask",
            "actor_hidden_trunk_trainable_during_rl": True,
            "optimizer_scope": "all actor and critic parameters",
        },
        "torch_seed": control["torch_seed"],
        "schedule_case_ids": control["schedule_case_ids"],
        "reset_seeds": control["reset_seeds"],
        "data_use": {
            "optimization_split": "train", "validation_loaded": False,
            "development_test_loaded": False, "selection_performed": False,
        },
        "checkpoints": {
            "0": "checkpoint_ep000.json", "16": "checkpoint_ep016.json",
            "32": "checkpoint_ep032.json",
        },
        "artifact_sha256": {name: "hash" for name in protocol._expected_artifacts()},
        "runs": runs,
        "updates": [{"after_episode": episode} for episode in range(4, 33, 4)],
        "failed_episodes": 0, "actual_physics_steps": 320,
        "paired_fresh_control": {
            "shared_rng_consumption_may_diverge_after_kl_stop": True,
            "same_case_schedule_reset_seed_and_initial_rng_derivation": True,
        },
    }


def test_training_manifest_requires_full_source_runtime_init_schedule_and_budget(
    tmp_path, monkeypatch
):
    plan, manifest = _plan(tmp_path), None
    manifest = _complete_manifest(plan)
    directory = tmp_path / "transfer_seed11"
    directory.mkdir()
    for episode in (0, 16, 32):
        (directory / f"checkpoint_ep{episode:03d}.json").write_text("checkpoint\n")
    monkeypatch.setattr(pilot, "_verified_manifest", lambda *args: manifest)
    monkeypatch.setattr(protocol, "policy_contract", lambda *args, **kwargs: {"contract": True})
    monkeypatch.setattr(
        protocol, "_zero_checkpoint_audit", lambda *args: {"exact_zero": True}
    )
    accepted, zero = protocol._training_manifest(tmp_path, plan, 11)
    assert accepted is manifest and zero["exact_zero"]

    mutations = [
        lambda value: value["schedule_case_ids"].__setitem__(0, "wrong"),
        lambda value: value["reset_seeds"].__setitem__(0, 999),
        lambda value: value.__setitem__("torch_seed", 999),
        lambda value: value["runtime"]["after"].__setitem__("dtype", "float32"),
        lambda value: value["initialization"].__setitem__(
            "inherited_bc_candidate_sha256", "wrong"
        ),
        lambda value: value["artifact_sha256"].pop("checkpoint_ep016.json"),
        lambda value: value["updates"].pop(),
        lambda value: value["transfer_semantics"].__setitem__(
            "actor_hidden_trunk_trainable_during_rl", False
        ),
    ]
    for mutate in mutations:
        changed = deepcopy(_complete_manifest(plan))
        mutate(changed)
        monkeypatch.setattr(pilot, "_verified_manifest", lambda *args, value=changed: value)
        with pytest.raises(ValueError, match="frozen source/runtime/init/schedule/budget"):
            protocol._training_manifest(tmp_path, plan, 11)


def test_failed_training_directory_is_preserved_and_never_retried(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    directory = tmp_path / "transfer_seed11"
    directory.mkdir()
    (directory / "FAILED").write_text("proof\n", encoding="utf-8")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("tools.train_surface_ppo_transfer.train_ppo_transfer", forbidden)
    with pytest.raises(ValueError, match="preserved"):
        protocol.train_one(tmp_path, plan, 11)
    assert not called and (directory / "FAILED").read_text() == "proof\n"


def test_comparison_requires_negative_paired_mean_for_every_seed(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    pilot._write_new(output / "plan.json", {"frozen": True})
    parent = tmp_path
    for seed in protocol.SEEDS:
        directory = output / f"transfer_seed{seed}"
        directory.mkdir()
        (directory / "checkpoint_ep032.json").write_text(f"transfer {seed}\n")
    monkeypatch.setattr(
        protocol, "_training_manifest",
        lambda output, plan, seed: ({"failed_episodes": 0}, {"exact_zero": True}),
    )
    validation = [
        {"seed": seed, "report": _report(tangent=3.0 if seed != 47 else 4.1)}
        for seed in protocol.SEEDS
    ]
    record = protocol._comparison(output, parent, plan, validation)
    assert not record["consistent_transfer_gain"]
    assert record["retained_arm"] == "fresh_ep32"
    validation[-1]["report"] = _report(tangent=3.9)
    assert protocol._comparison(output, parent, plan, validation)["consistent_transfer_gain"]


def test_pairing_rejects_same_case_with_different_simulation_seed():
    transfer, fresh = _report(tangent=3), _report(tangent=4)
    transfer["runs"][0]["simulation_seed"] += 1
    with pytest.raises(ValueError, match="not simulator-paired"):
        protocol._paired_tangent_delta(transfer, fresh)


def test_episode_zero_audit_directly_matches_both_pinned_bc_hidden_layers(
    tmp_path, monkeypatch
):
    def layers(offset=0.0):
        return [
            {"weights": np.full((32, 49), 0.1 + offset), "bias": np.full(32, 0.2)},
            {"weights": np.full((32, 32), 0.3), "bias": np.full(32, 0.4)},
            {"weights": np.zeros((3, 32)), "bias": np.zeros(3)},
        ]

    episode0, bc = tmp_path / "episode0.json", tmp_path / "bc.json"
    episode0.write_text("episode0\n", encoding="utf-8")
    bc.write_text("bc\n", encoding="utf-8")
    artifacts = {
        episode0: SimpleNamespace(payload={"layers": layers()}),
        bc: SimpleNamespace(payload={"layers": layers()}),
    }
    monkeypatch.setattr(protocol, "policy_contract", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        protocol, "load_policy_artifact", lambda path, **kwargs: artifacts[Path(path)]
    )
    monkeypatch.setattr(
        protocol.surface_mlp_actor, "actor_from_artifact",
        lambda artifact: (lambda observation: np.zeros(3), {}),
    )
    audit = protocol._zero_checkpoint_audit(episode0, bc, _plan(tmp_path), 11)
    assert audit["hidden_layer_exact_equal_to_pinned_bc"] == [True, True]
    assert audit["exact_zero"] and not audit["bc_output_head_copied"]

    artifacts[episode0] = SimpleNamespace(payload={"layers": layers(offset=0.01)})
    with pytest.raises(ValueError, match="trunk differs"):
        protocol._zero_checkpoint_audit(episode0, bc, _plan(tmp_path), 11)


def test_run_freezes_comparison_before_exactly_six_public_development_evaluations(
    tmp_path, monkeypatch
):
    output, parent, benchmark = tmp_path / "output", tmp_path / "parent", tmp_path / "bench"
    output.mkdir()
    parent.mkdir()
    benchmark.mkdir()
    pilot._write_new(benchmark / "report.json", {"runs": []})
    plan = _plan(parent)
    monkeypatch.setattr(protocol, "load", lambda *args: plan)

    def train(root, plan, seed):
        directory = Path(root) / f"transfer_seed{seed}"
        directory.mkdir(exist_ok=True)
        candidate = directory / "checkpoint_ep032.json"
        candidate.write_text(f"transfer {seed}\n", encoding="utf-8")
        return candidate

    monkeypatch.setattr(protocol, "train_one", train)
    monkeypatch.setattr(protocol, "validate_one", lambda *args: _report())
    monkeypatch.setattr(protocol, "_comparison", lambda *args: {
        "validation_safe_for_development": True, "consistent_transfer_gain": False,
        "retained_arm": "fresh_ep32",
    })
    observed = []

    def evaluate(root, plan, candidate, name, split, nominal):
        assert (Path(root) / "comparison.json").is_file()
        assert name.endswith("_ep32") and split == "development_test" and nominal == "friction"
        observed.append(name)
        return _report("development_test")

    monkeypatch.setattr(pilot, "_evaluate", evaluate)
    monkeypatch.setattr(pilot, "paired_baselines", lambda *args: [])
    result = protocol.run(output, parent, "bc", "data", benchmark, "plan-sha")
    assert result == output / "summary.json"
    assert observed == [
        *(f"fresh_seed{seed}_ep32" for seed in protocol.SEEDS),
        *(f"transfer_seed{seed}_ep32" for seed in protocol.SEEDS),
    ]
    summary = pilot._read(result)
    assert len(summary["development_test"]) == 6
    assert all(not row["independent_new_case"] for row in summary["development_test"])


def test_validation_veto_is_frozen_and_never_opens_development(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(protocol, "load", lambda *args: {"seeds": list(protocol.SEEDS)})
    monkeypatch.setattr(protocol, "train_one", lambda *args: None)
    monkeypatch.setattr(protocol, "validate_one", lambda *args: _report())
    monkeypatch.setattr(protocol, "_comparison", lambda *args: {
        "validation_safe_for_development": False, "consistent_transfer_gain": False,
        "retained_arm": "fresh_ep32",
    })
    monkeypatch.setattr(
        pilot, "_evaluate", lambda *args: pytest.fail("development must stay closed")
    )
    result = protocol.run(output, tmp_path, "bc", "data", "bench", "plan-sha")
    assert result == output / "comparison.json"
    assert (output / "VALIDATION_VETO").read_text().strip() == pilot._sha256(result)
    assert not (output / "COMPLETE").exists()


def test_declared_training_runtime_is_independent_of_ambient_torch_flags():
    torch = pytest.importorskip("torch")
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(max(2, previous_threads))
        torch.use_deterministic_algorithms(False)
        runtime = protocol._runtime()
        assert runtime["torch_num_threads"] == 1
        assert runtime["deterministic_algorithms"] is True
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)
