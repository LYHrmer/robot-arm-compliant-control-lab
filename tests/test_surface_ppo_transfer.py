import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_policy_artifact import load_policy_artifact, policy_contract
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    yaw_frame,
)
from compliant_control_lab.surface_splits import split_cases
from tools import surface_mlp_actor
from tools import train_surface_ppo as base_ppo
from tools import train_surface_ppo_transfer as transfer


def _torch():
    return pytest.importorskip("torch")


def _case_plans(duration=0.02):
    cases = []
    for group, yaw in enumerate((-15.0, 0.0, 15.0)):
        cases.append(
            {
                "case_id": f"g{group}_seed11",
                "scenario": asdict(
                    SurfaceScenario(name=f"transfer_{group}", wall_yaw_deg=yaw)
                ),
                "config": asdict(
                    SurfaceSimulationConfig(
                        duration=duration, seed=11, contact_model="smooth"
                    )
                ),
                "task": asdict(LearningSurfaceTask(yaw_deg=yaw)),
                "controller_frame_rotation": yaw_frame(yaw).rotation.tolist(),
                "nominal_kind": "adaptive",
            }
        )
    bc_cases = split_cases(cases, seed=31, train_fraction=0.34, validation_fraction=0.33)
    rl_cases = [{**deepcopy(case), "nominal_kind": "friction"} for case in bc_cases]
    return bc_cases, rl_cases


def _layers(width=32):
    rng = np.random.default_rng(19 + width)
    return [
        {
            "weights": rng.normal(0, 0.08, size=(width, 49)).tolist(),
            "bias": rng.normal(0, 0.03, size=width).tolist(),
        },
        {
            "weights": rng.normal(0, 0.08, size=(width, width)).tolist(),
            "bias": rng.normal(0, 0.03, size=width).tolist(),
        },
        {
            "weights": rng.normal(0, 0.08, size=(3, width)).tolist(),
            "bias": [0.2, -0.1, 0.05],
        },
    ]


def _candidate(path, bc_cases, *, layers=None, contract=None):
    contract = contract or policy_contract(
        bc_cases, purpose="il_friction_teacher", nominal_kind="adaptive"
    )
    surface_mlp_actor.save_mlp_candidate(
        path, layers or _layers(), contract=contract, activation="tanh"
    )
    return path


def _torch_seed(seed):
    sequence = np.random.SeedSequence(seed).spawn(3)[2]
    return int(sequence.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def test_nonzero_bc_trunk_is_copied_but_full_head_is_never_copied(tmp_path):
    torch = _torch()
    bc_cases, rl_cases = _case_plans()
    source_layers = _layers()
    candidate = _candidate(tmp_path / "bc.json", bc_cases, layers=source_layers)
    copied, provenance = transfer._load_bc_trunk(candidate, bc_cases, rl_cases)
    model, evidence = transfer._initialize_model(_torch_seed(11), copied, probe_seed=9)

    for module, source in zip((model.actor_hidden[0], model.actor_hidden[2]), source_layers[:2]):
        np.testing.assert_array_equal(module.weight.detach().numpy(), source["weights"])
        np.testing.assert_array_equal(module.bias.detach().numpy(), source["bias"])
    assert np.any(np.asarray(source_layers[2]["weights"]) != 0)
    assert torch.count_nonzero(model.actor_mean.weight) == 0
    assert torch.count_nonzero(model.actor_mean.bias) == 0
    assert provenance["excluded_bc_output_layer_sha256"] != {
        "weights": transfer._array_sha256(model.actor_mean.weight.detach().numpy()),
        "bias": transfer._array_sha256(model.actor_mean.bias.detach().numpy()),
    }
    assert evidence["actor_final_head_forced_zero"]
    assert evidence["episode0_zero_probe"]["max_abs_action"] == 0.0


def test_transfer_leaves_same_seed_fresh_critic_bit_exact_and_e0_zero(tmp_path):
    torch = _torch()
    bc_cases, rl_cases = _case_plans()
    copied, _ = transfer._load_bc_trunk(
        _candidate(tmp_path / "bc.json", bc_cases), bc_cases, rl_cases
    )
    seed = _torch_seed(29)
    model, evidence = transfer._initialize_model(seed, copied, probe_seed=12)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fresh = base_ppo.ActorCritic().double().cpu()
    for transferred, reference in zip(model.value_hidden.parameters(), fresh.value_hidden.parameters()):
        torch.testing.assert_close(transferred, reference, rtol=0, atol=0)
    for transferred, reference in zip(model.value_head.parameters(), fresh.value_head.parameters()):
        torch.testing.assert_close(transferred, reference, rtol=0, atol=0)
    observations = torch.as_tensor(
        np.random.default_rng(5).uniform(-3, 3, size=(32, 49)), dtype=torch.float64
    )
    with torch.no_grad():
        torch.testing.assert_close(
            model.deterministic_action(observations), torch.zeros((32, 3), dtype=torch.float64),
            rtol=0, atol=0,
        )
    assert evidence["fresh_critic_exactly_unchanged"]


def test_nonzero_updated_head_export_matches_restricted_numpy_actor(tmp_path):
    torch = _torch()
    bc_cases, rl_cases = _case_plans()
    copied, _ = transfer._load_bc_trunk(
        _candidate(tmp_path / "bc.json", bc_cases), bc_cases, rl_cases
    )
    model, _ = transfer._initialize_model(_torch_seed(47), copied, probe_seed=2)
    with torch.no_grad():
        model.actor_mean.weight.copy_(
            torch.linspace(-0.2, 0.3, 96, dtype=torch.float64).reshape(3, 32)
        )
        model.actor_mean.bias.copy_(torch.tensor([0.03, -0.07, 0.11], dtype=torch.float64))
    contract = policy_contract(rl_cases, purpose="rl_residual", nominal_kind="friction")
    exported = tmp_path / "rl.json"
    surface_mlp_actor.save_mlp_candidate(
        exported, base_ppo.export_actor_layers(model), contract=contract, activation="tanh"
    )
    actor, _ = surface_mlp_actor.actor_from_artifact(
        load_policy_artifact(exported, expected_contract=contract)
    )
    observations = np.random.default_rng(8).uniform(-3, 3, size=(9, 49))
    with torch.no_grad():
        expected = model.deterministic_action(
            torch.as_tensor(observations, dtype=torch.float64)
        ).numpy()
    assert np.any(expected != 0)
    np.testing.assert_allclose(
        np.stack([actor(observation) for observation in observations]), expected,
        rtol=0, atol=1e-12,
    )


def test_wrong_bc_contract_is_rejected(tmp_path):
    _torch()
    bc_cases, rl_cases = _case_plans()
    wrong_contract = policy_contract(rl_cases, purpose="rl_residual", nominal_kind="friction")
    candidate = _candidate(tmp_path / "wrong_contract.json", bc_cases, contract=wrong_contract)
    with pytest.raises(ValueError, match="contract differs"):
        transfer._load_bc_trunk(candidate, bc_cases, rl_cases)


def test_wrong_bc_architecture_is_rejected(tmp_path):
    _torch()
    bc_cases, rl_cases = _case_plans()
    candidate = _candidate(tmp_path / "wrong_width.json", bc_cases, layers=_layers(16))
    with pytest.raises(ValueError, match="49-32-32-3"):
        transfer._load_bc_trunk(candidate, bc_cases, rl_cases)


def test_bc_artifact_from_different_runner_source_is_rejected(tmp_path):
    _torch()
    bc_cases, rl_cases = _case_plans()
    candidate = _candidate(tmp_path / "wrong_source.json", bc_cases)
    document = json.loads(candidate.read_text(encoding="utf-8"))
    document["body"]["payload"]["runner_identity"]["runner_sha256"] = "0" * 64
    canonical = json.dumps(
        document["body"], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    document["body_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    candidate.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runner identity"):
        transfer._load_bc_trunk(candidate, bc_cases, rl_cases)


def test_bc_and_rl_case_plan_mismatch_is_rejected(tmp_path):
    _torch()
    bc_cases, rl_cases = _case_plans()
    candidate = _candidate(tmp_path / "bc.json", bc_cases)
    changed = deepcopy(rl_cases)
    changed[0]["config"]["duration"] += 0.02
    with pytest.raises(ValueError, match="plans differ"):
        transfer._load_bc_trunk(candidate, bc_cases, changed)


def _fake_episode(*args, **kwargs):
    data = {
        "observation": np.zeros((1, 49)),
        "action": np.zeros((1, 3)),
        "latent": np.zeros((1, 3)),
        "old_log_prob": np.zeros(1),
        "reward": np.ones(1),
        "next_observation": np.zeros((1, 49)),
        "terminated": np.array([False]),
        "truncated": np.array([True]),
        "terminal_observation_valid": np.array([True]),
        "physics_substeps": np.array([10]),
    }
    decisions = [{"termination_reasons": ()}]
    return data, {"time": np.arange(10) * 0.002}, decisions, {
        "simulation_seed": 123
    }, None, {"force_rmse_n": 0.1}, None


def _fake_update(*args, **kwargs):
    return {
        "transitions": 1, "minibatches": 1, "early_stop": False,
        "approx_kl": 0.001, "policy_loss_mean": -0.1, "value_loss_mean": 0.2,
        "gradient_norm_mean": 0.3, "gradient_norm_max": 0.3,
    }


def test_tiny_mocked_training_records_pairing_and_transfer_provenance(tmp_path, monkeypatch):
    _torch()
    bc_cases, rl_cases = _case_plans()
    candidate = _candidate(tmp_path / "bc.json", bc_cases)
    monkeypatch.setattr(transfer, "_run_episode", _fake_episode)
    monkeypatch.setattr(transfer, "_ppo_update", _fake_update)
    output = transfer.train_ppo_transfer(
        rl_cases, tmp_path / "run", bc_candidate=candidate, bc_cases=bc_cases,
        seed=11, episodes=1, episodes_per_update=1, checkpoint_episodes=(0, 1),
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (output / "COMPLETE").read_text().strip() == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    initialization = manifest["initialization"]
    assert initialization["inherited_bc_candidate_sha256"] == hashlib.sha256(
        candidate.read_bytes()
    ).hexdigest()
    assert len(initialization["transferred_layer_sha256"]) == 2
    assert initialization["fresh_critic_exactly_unchanged"]
    assert initialization["episode0_zero_probe"]["max_abs_action"] == 0.0
    assert manifest["data_use"] == {
        "optimization_split": "train", "validation_loaded": False,
        "development_test_loaded": False, "selection_performed": False,
    }
    expected_schedule = np.random.default_rng(np.random.SeedSequence(11).spawn(3)[0])
    train_count = sum(case["split"] == "train" for case in rl_cases)
    expected_case = [case for case in rl_cases if case["split"] == "train"][
        expected_schedule.permutation(train_count)[0]
    ]["case_id"]
    assert manifest["schedule_case_ids"] == [expected_case]
    assert set(manifest["script_sha256"]) == {
        "train_surface_ppo_transfer.py", "train_surface_ppo.py", "surface_mlp_actor.py"
    }
    assert manifest["runtime"]["before"] == manifest["runtime"]["after"]
    assert set(manifest["versions"]) == {
        "python_version", "numpy_version", "torch_version",
        "mujoco_version", "gymnasium_version",
    }
    paired = manifest["paired_fresh_control"]
    assert paired["same_case_schedule_reset_seed_and_initial_rng_derivation"]
    assert paired["shared_rng_consumption_may_diverge_after_kl_stop"]


def test_shared_update_rng_consumption_can_change_later_exploration_samples():
    torch = _torch()
    uninterrupted = torch.Generator(device="cpu").manual_seed(71)
    update_consumed = torch.Generator(device="cpu").manual_seed(71)
    torch.testing.assert_close(
        torch.randn(3, generator=uninterrupted, dtype=torch.float64),
        torch.randn(3, generator=update_consumed, dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.randperm(128, generator=update_consumed)
    assert not torch.equal(
        torch.randn(3, generator=uninterrupted, dtype=torch.float64),
        torch.randn(3, generator=update_consumed, dtype=torch.float64),
    )


def test_tiny_mocked_training_failure_preserves_partial_artifacts(tmp_path, monkeypatch):
    _torch()
    bc_cases, rl_cases = _case_plans()
    candidate = _candidate(tmp_path / "bc.json", bc_cases)
    monkeypatch.setattr(transfer, "_run_episode", _fake_episode)

    def fail_update(*args, **kwargs):
        raise FloatingPointError("injected optimizer failure")

    monkeypatch.setattr(transfer, "_ppo_update", fail_update)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="partial record published"):
        transfer.train_ppo_transfer(
            rl_cases, output, bc_candidate=candidate, bc_cases=bc_cases,
            seed=29, episodes=1, episodes_per_update=1, checkpoint_episodes=(0,),
        )
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_kind"] == "FloatingPointError"
    assert failure["completed_attempted_episodes"] == 1
    assert failure["initialization"]["inherited_bc_candidate_sha256"] == hashlib.sha256(
        candidate.read_bytes()
    ).hexdigest()
    assert "episode_001.npz" in failure["existing_artifact_sha256"]
    assert (output / "FAILED").is_file()
    assert not (output / "COMPLETE").exists()


def test_late_source_change_preserves_completed_episode_and_no_complete(tmp_path, monkeypatch):
    _torch()
    bc_cases, rl_cases = _case_plans()
    candidate = _candidate(tmp_path / "bc.json", bc_cases)
    monkeypatch.setattr(transfer, "_run_episode", _fake_episode)
    monkeypatch.setattr(transfer, "_ppo_update", _fake_update)
    monkeypatch.setattr(transfer, "_unchanged", lambda *args: False)
    output = tmp_path / "late_source_failure"
    with pytest.raises(RuntimeError, match="partial record published"):
        transfer.train_ppo_transfer(
            rl_cases, output, bc_candidate=candidate, bc_cases=bc_cases,
            seed=47, episodes=1, episodes_per_update=1, checkpoint_episodes=(0, 1),
        )
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_kind"] == "RuntimeError"
    assert failure["completed_attempted_episodes"] == 1
    assert "episode_001.npz" in failure["existing_artifact_sha256"]
    assert failure["runtime"]["before"] == failure["runtime"]["after_failure"]
    assert "source_and_assets_sha256" in failure["observed_after_failure"]
    assert (output / "FAILED").is_file()
    assert not (output / "COMPLETE").exists()


def test_deleted_bc_during_late_check_still_publishes_failure(tmp_path, monkeypatch):
    _torch()
    bc_cases, rl_cases = _case_plans()
    candidate = _candidate(tmp_path / "bc.json", bc_cases)
    monkeypatch.setattr(transfer, "_run_episode", _fake_episode)
    monkeypatch.setattr(transfer, "_ppo_update", _fake_update)

    def delete_then_reject(*args):
        candidate.unlink()
        return False

    monkeypatch.setattr(transfer, "_unchanged", delete_then_reject)
    output = tmp_path / "missing_bc_failure"
    with pytest.raises(RuntimeError, match="partial record published"):
        transfer.train_ppo_transfer(
            rl_cases, output, bc_candidate=candidate, bc_cases=bc_cases,
            seed=47, episodes=1, episodes_per_update=1, checkpoint_episodes=(0, 1),
        )
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    observed = failure["observed_after_failure"]["inherited_bc_candidate_sha256"]
    assert observed["value"] is None
    assert observed["inspection_error"]["kind"] == "FileNotFoundError"
    assert failure["completed_attempted_episodes"] == 1
    assert (output / "FAILED").is_file()
