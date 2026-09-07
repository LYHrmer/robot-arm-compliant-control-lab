"""Bounded PPO with a BC hidden representation and a fresh zero residual head.

This is a representation-transfer experiment, not a conversion of adaptive
teacher actions into friction-nominal residual actions.  It never evaluates or
selects candidates; callers own the frozen validation/development protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from copy import deepcopy
from importlib.metadata import version
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_experiment import _output_path, _source_hashes
from compliant_control_lab.surface_policy_artifact import load_policy_artifact, policy_contract
from tools import surface_mlp_actor
from tools.train_surface_ppo import (
    ActorCritic,
    _action_diagnostics,
    _deterministic_cpu,
    _finish_episode,
    _publish,
    _require_torch,
    _run_episode,
    _sha256,
    _write_json,
    export_actor_layers,
)
from tools.train_surface_ppo import _update as _ppo_update

SCHEMA = "surface_ppo_bc_hidden_transfer_v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_json(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _array_sha256(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(_canonical(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_sha256(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        array = parameter.detach().cpu().numpy()
        digest.update(_array_sha256(array).encode())
    return digest.hexdigest()


def _paired_plans(cases, bc_cases):
    def without_nominal(case):
        return {key: value for key, value in case.items() if key != "nominal_kind"}

    rl = sorted((without_nominal(case) for case in cases), key=lambda case: case["case_id"])
    bc = sorted((without_nominal(case) for case in bc_cases), key=lambda case: case["case_id"])
    if _canonical(rl) != _canonical(bc):
        raise ValueError("BC and RL case plans differ beyond nominal_kind")
    return _digest_json(rl)


def _load_bc_trunk(bc_candidate, bc_cases, cases):
    """Verify the adaptive artifact and return exactly its first two dense layers."""
    paired_sha256 = _paired_plans(cases, bc_cases)
    contract = policy_contract(
        bc_cases, purpose="il_friction_teacher", nominal_kind="adaptive"
    )
    artifact = load_policy_artifact(bc_candidate, expected_contract=contract)
    _, descriptor = surface_mlp_actor.actor_from_artifact(artifact)
    payload = artifact.payload
    if (
        payload.get("kind") != surface_mlp_actor.FORMAT
        or payload.get("activation") != "tanh"
        or payload.get("output_activation") != "tanh"
        or len(payload.get("layers", ())) != 3
    ):
        raise ValueError("BC transfer requires exactly a 49-32-32-3 tanh MLP")
    dense = surface_mlp_actor._dense_layers(payload["layers"])
    expected = (((32, 49), (32,)), ((32, 32), (32,)), ((3, 32), (3,)))
    if any(
        weights.shape != weight_shape or bias.shape != bias_shape
        for (weights, bias), (weight_shape, bias_shape) in zip(dense, expected)
    ):
        raise ValueError("BC transfer requires exactly a 49-32-32-3 tanh MLP")
    layers = [(weights.copy(), bias.copy()) for weights, bias in dense[:2]]
    return layers, {
        "mode": "bc_hidden_layers_only_fresh_zero_actor_head",
        "inherited_bc_candidate_sha256": artifact.artifact_sha256,
        "bc_contract_sha256": _digest_json(contract),
        "paired_case_plan_without_nominal_sha256": paired_sha256,
        "transferred_layer_sha256": [
            {"weights": _array_sha256(weights), "bias": _array_sha256(bias)}
            for weights, bias in layers
        ],
        "excluded_bc_output_layer_sha256": {
            "weights": _array_sha256(dense[2][0]), "bias": _array_sha256(dense[2][1])
        },
        "bc_runner_identity": descriptor["runner_identity"],
    }


def _initialize_model(torch_seed, layers, probe_seed):
    """Copy the trunk without touching the same-seed fresh critic or zero head."""
    torch = _require_torch()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(torch_seed)
        model = ActorCritic().double().cpu()
    critic_before = _state_sha256(model.value_hidden.named_parameters())
    critic_head_before = _state_sha256(model.value_head.named_parameters())
    with torch.no_grad():
        for module, (weights, bias) in zip(
            (model.actor_hidden[0], model.actor_hidden[2]), layers
        ):
            module.weight.copy_(torch.as_tensor(weights, dtype=torch.float64))
            module.bias.copy_(torch.as_tensor(bias, dtype=torch.float64))
        model.actor_mean.weight.zero_()
        model.actor_mean.bias.zero_()
    critic_after = _state_sha256(model.value_hidden.named_parameters())
    critic_head_after = _state_sha256(model.value_head.named_parameters())
    probes = np.vstack(
        (
            np.zeros((1, 49)),
            np.linspace(-3.0, 3.0, 49).reshape(1, 49),
            np.random.default_rng(probe_seed).uniform(-3.0, 3.0, size=(16, 49)),
        )
    )
    with torch.no_grad():
        output = model.deterministic_action(torch.as_tensor(probes, dtype=torch.float64)).numpy()
    if not np.array_equal(output, np.zeros_like(output)):
        raise RuntimeError("transferred actor does not start at exact zero residual")
    evidence = {
        "actor_final_head_forced_zero": True,
        "actor_final_head_sha256": _state_sha256(model.actor_mean.named_parameters()),
        "fresh_critic_before_transfer_sha256": {
            "hidden": critic_before, "head": critic_head_before
        },
        "fresh_critic_after_transfer_sha256": {
            "hidden": critic_after, "head": critic_head_after
        },
        "fresh_critic_exactly_unchanged": bool(
            critic_before == critic_after and critic_head_before == critic_head_after
        ),
        "episode0_zero_probe": {
            "seed": probe_seed,
            "count": len(probes),
            "input_sha256": _array_sha256(probes),
            "output_sha256": _array_sha256(output),
            "max_abs_action": float(np.max(np.abs(output))),
        },
    }
    if not evidence["fresh_critic_exactly_unchanged"]:
        raise RuntimeError("BC transfer changed the fresh critic")
    return model, evidence


def _validate(seed, episodes, episodes_per_update, checkpoint_episodes, gamma, gae_lambda,
              clip_epsilon, learning_rate, update_epochs, batch_size, max_grad_norm, std,
              target_kl):
    integer_values = (seed, episodes, episodes_per_update, update_epochs, batch_size)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           for value in integer_values):
        raise ValueError("seed, episode, epoch and batch values must be integers")
    if seed < 0 or episodes <= 0 or episodes_per_update <= 0 or episodes % episodes_per_update:
        raise ValueError("invalid seed or episode/update budget")
    checkpoints = tuple(checkpoint_episodes)
    if (
        any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in checkpoints)
        or 0 not in checkpoints
        or any(value < 0 or value > episodes for value in checkpoints)
        or any(value not in (0, episodes) and value % episodes_per_update for value in checkpoints)
    ):
        raise ValueError("checkpoints must include zero and follow update boundaries or final")
    real_values = (gamma, gae_lambda, clip_epsilon, learning_rate, max_grad_norm, std)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
           or not np.isfinite(value) for value in real_values):
        raise ValueError("PPO scalar hyperparameters must be finite real numbers")
    if (
        not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1 or not 0 < clip_epsilon < 1
        or learning_rate <= 0 or max_grad_norm <= 0 or not 0 < std < 1
        or update_epochs <= 0 or batch_size <= 0
        or (target_kl is not None and (
            isinstance(target_kl, (bool, np.bool_)) or not isinstance(target_kl, Real)
            or not np.isfinite(target_kl) or target_kl <= 0
        ))
    ):
        raise ValueError("invalid PPO hyperparameters")
    return int(seed), int(episodes), int(episodes_per_update), tuple(map(int, checkpoints))


def _script_hashes():
    here = Path(__file__).resolve()
    return {
        "train_surface_ppo_transfer.py": _sha256(here),
        "train_surface_ppo.py": _sha256(here.with_name("train_surface_ppo.py")),
        "surface_mlp_actor.py": _sha256(here.with_name("surface_mlp_actor.py")),
    }


def _runtime_identity(torch):
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "mujoco_version": version("mujoco"),
        "gymnasium_version": version("gymnasium"),
        "device": "cpu",
        "dtype": "float64",
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _save_checkpoint(staging, episode_count, model, contract):
    name = f"checkpoint_ep{episode_count:03d}.json"
    surface_mlp_actor.save_mlp_candidate(
        staging / name, export_actor_layers(model), contract=contract, activation="tanh"
    )
    return name


def _unchanged(sources, scripts, bc_candidate, bc_sha256):
    return (
        _source_hashes() == sources
        and _script_hashes() == scripts
        and _sha256(bc_candidate) == bc_sha256
    )


def _inspect_after_failure(label, function):
    """Failure evidence must survive even when the identity target disappeared."""
    try:
        return {"value": function(), "inspection_error": None}
    except Exception as error:  # noqa: BLE001 - diagnostic capture must not mask failure
        return {
            "value": None,
            "inspection_error": {"kind": type(error).__name__, "detail": str(error)},
            "label": label,
        }


@_deterministic_cpu
def train_ppo_transfer(
    cases,
    output,
    *,
    bc_candidate,
    bc_cases,
    seed,
    episodes=32,
    episodes_per_update=4,
    checkpoint_episodes=(0, 16, 32),
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    learning_rate=3e-4,
    update_epochs=4,
    batch_size=128,
    max_grad_norm=0.5,
    std=0.05,
    target_kl=0.02,
):
    """Run friction-nominal PPO after copying only an audited BC hidden trunk."""
    seed, episodes, episodes_per_update, checkpoint_episodes = _validate(
        seed, episodes, episodes_per_update, checkpoint_episodes, gamma, gae_lambda,
        clip_epsilon, learning_rate, update_epochs, batch_size, max_grad_norm, std, target_kl,
    )
    torch = _require_torch()
    cases, bc_cases = deepcopy(list(cases)), deepcopy(list(bc_cases))
    output, bc_candidate = _output_path(output), Path(bc_candidate).absolute()
    if output.exists():
        raise ValueError("PPO transfer output must be a new directory")
    contract = policy_contract(cases, purpose="rl_residual", nominal_kind="friction")
    train_cases = [case for case in cases if case["split"] == "train"]
    if not train_cases:
        raise ValueError("case plan contains no training cases")
    layers, transfer = _load_bc_trunk(bc_candidate, bc_cases, cases)
    bc_sha256 = transfer["inherited_bc_candidate_sha256"]
    seed_sequences = np.random.SeedSequence(seed).spawn(3)
    schedule_rng, reset_rng = (np.random.default_rng(item) for item in seed_sequences[:2])
    torch_seed = int(seed_sequences[2].generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))
    schedule = []
    while len(schedule) < episodes:
        schedule.extend(schedule_rng.permutation(len(train_cases)).tolist())
    schedule = schedule[:episodes]
    reset_seeds = reset_rng.integers(0, 2**31 - 1, size=episodes).tolist()
    probe_seed = int(np.random.SeedSequence([seed, 0xBC]).generate_state(1)[0])
    model, initialization = _initialize_model(torch_seed, layers, probe_seed)
    transfer.update(initialization)
    torch_generator = torch.Generator(device="cpu").manual_seed(torch_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sources, scripts = _source_hashes(), _script_hashes()
    runtime_before = _runtime_identity(torch)
    hyperparameters = {
        "episodes": episodes, "episodes_per_update": episodes_per_update,
        "checkpoint_episodes": sorted(set(checkpoint_episodes)), "gamma": gamma,
        "gae_lambda": gae_lambda, "clip_epsilon": clip_epsilon,
        "learning_rate": learning_rate, "update_epochs": update_epochs,
        "batch_size": batch_size, "max_grad_norm": max_grad_norm,
        "fixed_std": std, "target_kl": target_kl,
        "physics_discount": "gamma**(physics_substeps/10)",
        "gae_trace": "lambda**(physics_substeps/10)",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-ppo-transfer-", dir=output.parent) as temp:
        staging = Path(temp) / "training"
        staging.mkdir()
        rows, updates, checkpoints, pending = [], [], {}, []
        try:
            if 0 in checkpoint_episodes:
                checkpoints[0] = _save_checkpoint(staging, 0, model, contract)
            for episode_index, (case_index, reset_seed) in enumerate(
                zip(schedule, reset_seeds), start=1
            ):
                case = train_cases[case_index]
                data, trace, decisions, reset_info, failure, metrics, fatal = _run_episode(
                    case, int(reset_seed), model, std, torch_generator
                )
                if fatal is None:
                    _finish_episode(data, model, gamma, gae_lambda)
                trace.update({f"decision_{key.rstrip('_')}": value for key, value in data.items()})
                stem = f"episode_{episode_index:03d}"
                np.savez_compressed(staging / f"{stem}.npz", **trace)
                diagnostics = _action_diagnostics(data)
                _write_json(staging / f"{stem}.json", {
                    "case_id": case["case_id"], "group_id": case["group_id"],
                    "split": case["split"], "requested_reset_seed": int(reset_seed),
                    "actual_simulation_seed": reset_info.get("simulation_seed"),
                    "exception_detail": failure, "action_diagnostics": diagnostics,
                    "physical_metrics": metrics, "decisions": decisions,
                })
                row = {
                    "episode": episode_index, "case_id": case["case_id"],
                    "group_id": case["group_id"], "requested_reset_seed": int(reset_seed),
                    "actual_simulation_seed": reset_info.get("simulation_seed"),
                    "policy_steps": len(data["reward"]),
                    "physics_steps": int(data["physics_substeps"].sum()),
                    "episode_return": float(data["reward"].sum()),
                    "terminated": bool(data["terminated"][-1]) if len(data["reward"]) else True,
                    "truncated": bool(data["truncated"][-1]) if len(data["reward"]) else False,
                    "failed": bool(failure or (len(data["reward"]) and data["terminated"][-1])),
                    "failure_reasons": list(dict.fromkeys(
                        reason for decision in decisions for reason in decision["termination_reasons"]
                    )),
                    "exception_detail": failure, "trace_file": f"{stem}.npz",
                    "events_file": f"{stem}.json", "action_diagnostics": diagnostics,
                    "physical_metrics": metrics,
                }
                rows.append(row)
                if fatal is not None:
                    raise fatal
                pending.append(data)
                if episode_index % episodes_per_update == 0:
                    updates.append({"after_episode": episode_index, **_ppo_update(
                        model, optimizer, pending, std=std, clip_epsilon=clip_epsilon,
                        update_epochs=update_epochs, batch_size=batch_size,
                        max_grad_norm=max_grad_norm, target_kl=target_kl,
                        torch_generator=torch_generator,
                    )})
                    pending = []
                if episode_index in checkpoint_episodes:
                    checkpoints[episode_index] = _save_checkpoint(
                        staging, episode_index, model, contract
                    )
            runtime_after = _runtime_identity(torch)
            if not _unchanged(sources, scripts, bc_candidate, bc_sha256):
                raise RuntimeError("source, trainer, or inherited BC artifact changed during training")
            if runtime_after != runtime_before:
                raise RuntimeError("runtime identity changed during PPO transfer")
            artifact_hashes = {
                path.name: _sha256(path) for path in sorted(staging.iterdir()) if path.is_file()
            }
            manifest = {
                "schema": SCHEMA, "purpose": "rl_residual", "nominal_kind": "friction",
                "scope": "bounded representation-transfer pilot; no performance claim or evaluation",
                "seed": seed, "torch_seed": torch_seed, "hyperparameters": hyperparameters,
                "architecture": "BC 49-32-32 trunk; fresh zero 3 head; fresh value 49-32-32-1",
                "initialization": transfer, "case_plan": cases,
                "bc_case_plan_sha256": _digest_json(bc_cases),
                "paired_fresh_control": {
                    "trainer": "tools/train_surface_ppo.py",
                    "trainer_sha256": scripts["train_surface_ppo.py"],
                    "same_case_schedule_reset_seed_and_initial_rng_derivation": True,
                    "shared_rng_consumption_may_diverge_after_kl_stop": True,
                    "same_seed_fresh_critic": transfer["fresh_critic_exactly_unchanged"],
                    "only_intended_model_difference": "actor hidden layers 0 and 1",
                },
                "transfer_semantics": {
                    "bc_action_head_copied": False,
                    "bc_actions_converted_to_friction_residuals": False,
                    "runtime_observation_inputs": "all 49 measured inputs; no BC input mask",
                    "actor_hidden_trunk_trainable_during_rl": True,
                    "optimizer_scope": "all actor and critic parameters",
                },
                "training_case_ids": [case["case_id"] for case in train_cases],
                "training_group_ids": sorted({case["group_id"] for case in train_cases}),
                "schedule_case_ids": [train_cases[index]["case_id"] for index in schedule],
                "reset_seeds": [int(value) for value in reset_seeds], "runs": rows,
                "updates": updates, "failed_episodes": sum(row["failed"] for row in rows),
                "actual_physics_steps": sum(row["physics_steps"] for row in rows),
                "checkpoints": {str(key): value for key, value in sorted(checkpoints.items())},
                "policy_contract": contract, "source_and_assets_sha256": sources,
                "script_sha256": scripts,
                "runtime": {"before": runtime_before, "after": runtime_after},
                "versions": {
                    key: runtime_before[key] for key in (
                        "python_version", "numpy_version", "torch_version",
                        "mujoco_version", "gymnasium_version",
                    )
                },
                "data_use": {
                    "optimization_split": "train", "validation_loaded": False,
                    "development_test_loaded": False, "selection_performed": False,
                },
                "artifact_sha256": artifact_hashes,
            }
            _write_json(staging / "manifest.json", manifest)
            if not _unchanged(sources, scripts, bc_candidate, bc_sha256):
                raise RuntimeError(
                    "source, trainer, or inherited BC artifact changed before publication"
                )
            if _runtime_identity(torch) != runtime_before:
                raise RuntimeError("runtime identity changed before PPO transfer publication")
            (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
            _publish(staging, output)
        except Exception as error:
            complete = staging / "COMPLETE"
            if complete.exists():
                complete.unlink()
            existing = {
                path.name: _sha256(path) for path in sorted(staging.iterdir()) if path.is_file()
            }
            failed = {
                "schema": "surface_ppo_bc_hidden_transfer_failure_v1",
                "error": str(error), "failure_kind": type(error).__name__,
                "completed_attempted_episodes": len(rows), "runs": rows,
                "initialization": transfer, "source_and_assets_sha256": sources,
                "script_sha256": scripts, "runtime": {
                    "before": runtime_before, "after_failure": _runtime_identity(torch)
                },
                "observed_after_failure": {
                    "source_and_assets_sha256": _inspect_after_failure(
                        "source_and_assets", _source_hashes
                    ),
                    "script_sha256": _inspect_after_failure("scripts", _script_hashes),
                    "inherited_bc_candidate_sha256": _inspect_after_failure(
                        "inherited_bc_candidate", lambda: _sha256(bc_candidate)
                    ),
                },
                "existing_artifact_sha256": existing,
            }
            _write_json(staging / "failure.json", failed)
            (staging / "FAILED").write_text(_sha256(staging / "failure.json") + "\n")
            _publish(staging, output)
            raise RuntimeError(
                f"PPO transfer failure; partial record published at {output}"
            ) from error
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--bc-cases", type=Path, required=True)
    parser.add_argument("--bc-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    args = parser.parse_args(argv)
    result = train_ppo_transfer(
        json.loads(args.cases.read_text(encoding="utf-8")), args.output,
        bc_candidate=args.bc_candidate,
        bc_cases=json.loads(args.bc_cases.read_text(encoding="utf-8")),
        seed=args.seed, episodes=args.episodes,
    )
    print(json.dumps({"output": str(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
