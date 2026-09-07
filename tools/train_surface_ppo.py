"""Bounded CPU/float64 PPO pilot for the friction-nominal surface residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from copy import deepcopy
from functools import wraps
from importlib.metadata import version
from pathlib import Path

import numpy as np

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_env import POLICY_SUBSTEPS, SurfaceLearningEnv
from compliant_control_lab.surface_experiment import _output_path, _source_hashes
from compliant_control_lab.surface_policy import SurfaceResidualConfig
from compliant_control_lab.surface_policy_artifact import policy_contract
from compliant_control_lab.surface_simulation import SurfaceScenario, SurfaceSimulationConfig

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # The base package intentionally does not depend on torch.
    torch = None
    nn = None

try:
    from tools.surface_mlp_actor import save_mlp_candidate
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    from surface_mlp_actor import save_mlp_candidate


SCHEMA = "surface_ppo_training_v1"


class NumericalTrainingError(RuntimeError):
    """A nonfinite actor, loss, gradient, or parameter invalidates the run."""


class EpisodeExecutionError(RuntimeError):
    """An unexpected environment/runtime exception invalidates the run."""


if nn is not None:

    class ActorCritic(nn.Module):
        """Separate 49-32-32 actor and value MLPs; actor starts at zero residual."""

        def __init__(self):
            super().__init__()
            self.actor_hidden = nn.Sequential(
                nn.Linear(49, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
            )
            self.actor_mean = nn.Linear(32, 3)
            self.value_hidden = nn.Sequential(
                nn.Linear(49, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
            )
            self.value_head = nn.Linear(32, 1)
            nn.init.zeros_(self.actor_mean.weight)
            nn.init.zeros_(self.actor_mean.bias)

        def policy_mean(self, observation):
            return self.actor_mean(self.actor_hidden(observation))

        def value(self, observation):
            return self.value_head(self.value_hidden(observation)).squeeze(-1)

        def deterministic_action(self, observation):
            return torch.tanh(self.policy_mean(observation))

else:

    class ActorCritic:  # pragma: no cover - exercised only in installations without torch
        def __init__(self):
            raise ModuleNotFoundError("train_surface_ppo requires optional dependency torch")


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError("train_surface_ppo requires optional dependency torch")
    return torch


def _deterministic_cpu(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        library = _require_torch()
        previous_threads = library.get_num_threads()
        previous_deterministic = library.are_deterministic_algorithms_enabled()
        library.set_num_threads(1)
        library.use_deterministic_algorithms(True)
        try:
            return function(*args, **kwargs)
        finally:
            library.use_deterministic_algorithms(previous_deterministic)
            library.set_num_threads(previous_threads)

    return wrapped


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def tanh_normal_log_prob(latent, mean, std):
    """Stable log p(tanh(latent)); `latent` is retained rather than inverted."""
    library = _require_torch()
    std_tensor = library.as_tensor(std, dtype=latent.dtype, device=latent.device)
    gaussian = -0.5 * (((latent - mean) / std_tensor) ** 2)
    gaussian -= library.log(std_tensor) + 0.5 * np.log(2.0 * np.pi)
    log_jacobian = 2.0 * (np.log(2.0) - latent - library.nn.functional.softplus(-2.0 * latent))
    return (gaussian - log_jacobian).sum(dim=-1)


def ppo_clipped_loss(new_log_prob, old_log_prob, advantage, clip_epsilon=0.2):
    library = _require_torch()
    ratio = (new_log_prob - old_log_prob).exp()
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return -library.minimum(ratio * advantage, clipped * advantage).mean()


def compute_gae(
    rewards,
    values,
    next_values,
    terminated,
    truncated,
    terminal_observation_valid,
    physics_substeps,
    *,
    gamma=0.99,
    gae_lambda=0.95,
):
    """Variable-step GAE, with bootstrap and cross-reset continuation kept distinct."""
    arrays = [
        np.asarray(value)
        for value in (
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            terminal_observation_valid,
            physics_substeps,
        )
    ]
    if not arrays or any(value.shape != arrays[0].shape or value.ndim != 1 for value in arrays):
        raise ValueError("GAE inputs must be equal-length one-dimensional arrays")
    if not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and GAE lambda are outside their probability ranges")
    if (
        np.any(arrays[-1] < 0)
        or not np.all(np.isfinite(arrays[-1]))
        or not np.all(arrays[-1] == np.floor(arrays[-1]))
    ):
        raise ValueError("physics_substeps must be finite nonnegative integers")
    rewards, values, next_values = (value.astype(np.float64) for value in arrays[:3])
    terminated, truncated, valid = (value.astype(bool) for value in arrays[3:6])
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(values)):
        raise ValueError("rewards and current values must be finite")
    bootstrap = (~terminated) & valid
    if not np.all(np.isfinite(next_values[bootstrap])):
        raise ValueError("every bootstrap-eligible next value must be finite")
    durations = arrays[6].astype(np.float64) / POLICY_SUBSTEPS
    discounts = gamma**durations
    traces = gae_lambda**durations
    continuation = (~terminated) & (~truncated) & valid
    delta = rewards + discounts * np.where(bootstrap, next_values, 0.0) - values
    advantage = np.zeros_like(delta)
    carried = 0.0
    for index in range(len(delta) - 1, -1, -1):
        carried = delta[index] + discounts[index] * traces[index] * continuation[index] * carried
        advantage[index] = carried
    return advantage, advantage + values


def export_actor_layers(model):
    """Return the restricted runner's out-by-in dense representation."""
    dense = (model.actor_hidden[0], model.actor_hidden[2], model.actor_mean)
    return [
        {
            "weights": layer.weight.detach().cpu().numpy().astype(np.float64).tolist(),
            "bias": layer.bias.detach().cpu().numpy().astype(np.float64).tolist(),
        }
        for layer in dense
    ]


def _environment(case):
    return SurfaceLearningEnv(
        SurfaceFrame(case["controller_frame_rotation"]),
        scenario=SurfaceScenario(**case["scenario"]),
        config=SurfaceSimulationConfig(**case["config"]),
        task=LearningSurfaceTask(**case["task"]),
        nominal_kind="friction",
        residual_config=SurfaceResidualConfig(),
    )


def _empty_episode():
    return {
        "observation": [],
        "action": [],
        "latent": [],
        "old_log_prob": [],
        "reward": [],
        "next_observation": [],
        "terminated": [],
        "truncated": [],
        "terminal_observation_valid": [],
        "physics_substeps": [],
    }


def _run_episode(case, reset_seed, model, std, torch_generator):
    library = _require_torch()
    env, data, decisions, execution_reasons = None, _empty_episode(), [], []
    reset_info, failure, fatal, trace, physical_metrics = {}, None, None, {}, {}
    try:
        env = _environment(case)
        observation, reset_info = env.reset(seed=reset_seed)
        done = False
        while not done:
            obs_tensor = library.as_tensor(observation, dtype=library.float64)
            with library.no_grad():
                mean = model.policy_mean(obs_tensor)
                latent = mean + std * library.randn(3, generator=torch_generator, dtype=library.float64)
                action_tensor = library.tanh(latent)
                log_prob = tanh_normal_log_prob(latent, mean, std)
            if not all(
                library.isfinite(value).all().item() for value in (mean, latent, action_tensor, log_prob)
            ):
                raise NumericalTrainingError("nonfinite policy sample")
            action = action_tensor.numpy().copy()
            next_observation, reward, terminated, truncated, info = env.step(action)
            data["observation"].append(np.asarray(observation, dtype=np.float64).copy())
            data["action"].append(action)
            data["latent"].append(latent.numpy().copy())
            data["old_log_prob"].append(float(log_prob))
            data["reward"].append(float(reward))
            data["next_observation"].append(np.asarray(next_observation, dtype=np.float64).copy())
            data["terminated"].append(bool(terminated))
            data["truncated"].append(bool(truncated))
            data["terminal_observation_valid"].append(bool(info["terminal_observation_valid"]))
            data["physics_substeps"].append(int(info["physics_substeps"]))
            for stage in info["action_stages"]:
                if stage["execution_status"] == "recorded":
                    execution_reasons.append(";".join(stage["reasons"]))
            decisions.append(
                {
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "terminal_observation_valid": info["terminal_observation_valid"],
                    "physics_substeps": info["physics_substeps"],
                    "termination_reasons": info["termination_reasons"],
                    "input_action_issue": info["input_action_issue"],
                    "exception_detail": info["exception_detail"],
                    "reward_cost_integrals": info["reward_cost_integrals"],
                    "termination_penalty": info["termination_penalty"],
                }
            )
            observation, done = next_observation, bool(terminated or truncated)
        result = env.result()
        trace, physical_metrics = result.trace, result.metrics()
    except NumericalTrainingError as error:
        failure, fatal = f"{type(error).__name__}: {error}", error
        if env is not None:
            try:
                result = env.result()
                trace, physical_metrics = result.trace, result.metrics()
            except (RuntimeError, ValueError, IndexError):
                trace = {}
    except Exception as error:  # noqa: BLE001 - preserve every attempted episode visibly
        failure = f"{type(error).__name__}: {error}"
        fatal = EpisodeExecutionError(failure)
        if env is not None:
            try:
                result = env.result()
                trace, physical_metrics = result.trace, result.metrics()
            except (RuntimeError, ValueError, IndexError):
                trace = {}
    finally:
        if env is not None:
            env.close()
    shaped = {
        "observation": np.asarray(data["observation"], dtype=np.float64).reshape((-1, 49)),
        "action": np.asarray(data["action"], dtype=np.float64).reshape((-1, 3)),
        "latent": np.asarray(data["latent"], dtype=np.float64).reshape((-1, 3)),
        "old_log_prob": np.asarray(data["old_log_prob"], dtype=np.float64),
        "reward": np.asarray(data["reward"], dtype=np.float64),
        "next_observation": np.asarray(data["next_observation"], dtype=np.float64).reshape((-1, 49)),
        "terminated": np.asarray(data["terminated"], dtype=bool),
        "truncated": np.asarray(data["truncated"], dtype=bool),
        "terminal_observation_valid": np.asarray(
            data["terminal_observation_valid"], dtype=bool
        ),
        "physics_substeps": np.asarray(data["physics_substeps"], dtype=np.int64),
    }
    finite_keys = {
        "observation", "action", "latent", "old_log_prob", "reward", "physics_substeps"
    }
    if not all(np.all(np.isfinite(shaped[key])) for key in finite_keys):
        fatal = NumericalTrainingError("nonfinite rollout data")
        failure = f"{type(fatal).__name__}: {fatal}"
    valid_next = shaped["terminal_observation_valid"]
    if not np.all(np.isfinite(shaped["next_observation"][valid_next])):
        fatal = NumericalTrainingError("nonfinite valid next observation")
        failure = f"{type(fatal).__name__}: {fatal}"
    trace = dict(trace)
    trace["execution_safety_reasons"] = np.asarray(execution_reasons, dtype=str)
    return shaped, trace, decisions, reset_info, failure, physical_metrics, fatal


def _finish_episode(data, model, gamma, gae_lambda):
    library = _require_torch()
    count = len(data["reward"])
    if not count:
        data.update(value=np.empty(0), next_value=np.empty(0), advantage=np.empty(0), return_=np.empty(0))
        return
    bootstrap = (~data["terminated"]) & data["terminal_observation_valid"]
    with library.no_grad():
        values = model.value(library.as_tensor(data["observation"], dtype=library.float64)).numpy()
        next_values = np.zeros(count, dtype=np.float64)
        if np.any(bootstrap):
            next_values[bootstrap] = model.value(
                library.as_tensor(data["next_observation"][bootstrap], dtype=library.float64)
            ).numpy()
    advantages, returns = compute_gae(
        data["reward"], values, next_values, data["terminated"], data["truncated"],
        data["terminal_observation_valid"], data["physics_substeps"], gamma=gamma,
        gae_lambda=gae_lambda,
    )
    if not all(np.all(np.isfinite(value)) for value in (values, next_values, advantages, returns)):
        raise NumericalTrainingError("nonfinite value target or advantage")
    data.update(value=values, next_value=next_values, advantage=advantages, return_=returns)


def _update(model, optimizer, episodes, *, std, clip_epsilon, update_epochs, batch_size,
            max_grad_norm, target_kl, torch_generator):
    library = _require_torch()
    usable = [episode for episode in episodes if len(episode["reward"])]
    transitions = sum(len(episode["reward"]) for episode in usable)
    if not transitions:
        return {
            "transitions": 0, "minibatches": 0, "early_stop": False,
            "approx_kl": None, "policy_loss_mean": None, "value_loss_mean": None,
            "gradient_norm_mean": None, "gradient_norm_max": None,
        }
    joined = {key: np.concatenate([episode[key] for episode in usable]) for key in (
        "observation", "latent", "old_log_prob", "advantage", "return_"
    )}
    advantage = joined["advantage"]
    advantage = (advantage - advantage.mean()) / max(advantage.std(), 1e-8)
    tensors = {
        key: library.as_tensor(value, dtype=library.float64)
        for key, value in {**joined, "advantage": advantage}.items()
    }
    minibatches, last_kl, early_stop = 0, None, False
    policy_losses, value_losses, gradient_norms = [], [], []
    for _ in range(update_epochs):
        permutation = library.randperm(transitions, generator=torch_generator)
        for start in range(0, transitions, batch_size):
            index = permutation[start : start + batch_size]
            mean = model.policy_mean(tensors["observation"][index])
            new_log_prob = tanh_normal_log_prob(tensors["latent"][index], mean, std)
            policy_loss = ppo_clipped_loss(
                new_log_prob, tensors["old_log_prob"][index], tensors["advantage"][index],
                clip_epsilon,
            )
            value_loss = 0.5 * (model.value(tensors["observation"][index]) - tensors["return_"][index]).square().mean()
            loss = policy_loss + value_loss
            if not library.isfinite(loss).item():
                raise NumericalTrainingError("nonfinite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = library.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not library.isfinite(norm).item():
                raise NumericalTrainingError("nonfinite PPO gradient")
            optimizer.step()
            if not all(library.isfinite(parameter).all().item() for parameter in model.parameters()):
                raise NumericalTrainingError("nonfinite PPO parameter")
            minibatches += 1
            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            gradient_norms.append(float(norm.detach()))
            with library.no_grad():
                updated_mean = model.policy_mean(tensors["observation"][index])
                updated_log_prob = tanh_normal_log_prob(
                    tensors["latent"][index], updated_mean, std
                )
                last_kl = float(
                    (tensors["old_log_prob"][index] - updated_log_prob).mean()
                )
            if target_kl is not None and last_kl > target_kl:
                early_stop = True
                break
        if early_stop:
            break
    return {
        "transitions": transitions, "minibatches": minibatches,
        "early_stop": early_stop, "approx_kl": last_kl,
        "policy_loss_mean": float(np.mean(policy_losses)),
        "value_loss_mean": float(np.mean(value_losses)),
        "gradient_norm_mean": float(np.mean(gradient_norms)),
        "gradient_norm_max": float(np.max(gradient_norms)),
    }


def _save_checkpoint(staging, episode_count, model, contract):
    name = f"checkpoint_ep{episode_count:03d}.json"
    save_mlp_candidate(staging / name, export_actor_layers(model), contract=contract, activation="tanh")
    return name


def _action_diagnostics(data):
    if not len(data["reward"]):
        return {
            "sampled_policy_entropy_nats": None, "latent_mean": None,
            "latent_std": None, "latent_abs_max": None, "action_mean": None,
            "action_std": None, "action_abs_max": None,
            "near_zero_fraction_abs_lt_0_01": None,
            "saturation_fraction_abs_gt_0_95": None,
        }
    latent, action = data["latent"], data["action"]
    return {
        "sampled_policy_entropy_nats": float(-data["old_log_prob"].mean()),
        "latent_mean": latent.mean(axis=0), "latent_std": latent.std(axis=0),
        "latent_abs_max": float(np.max(np.abs(latent))),
        "action_mean": action.mean(axis=0), "action_std": action.std(axis=0),
        "action_abs_max": float(np.max(np.abs(action))),
        "near_zero_fraction_abs_lt_0_01": float(np.mean(np.abs(action) < 0.01)),
        "saturation_fraction_abs_gt_0_95": float(np.mean(np.abs(action) > 0.95)),
    }


def _publish(staging, output):
    _output_path(output)
    if output.exists():
        raise FileExistsError("PPO destination appeared during training")
    os.rename(staging, output)


@_deterministic_cpu
def train_ppo(
    cases, output, *, seed, episodes=32, episodes_per_update=4,
    checkpoint_episodes=(0, 16, 32), gamma=0.99, gae_lambda=0.95,
    clip_epsilon=0.2, learning_rate=3e-4, update_epochs=4, batch_size=128,
    max_grad_norm=0.5, std=0.05, target_kl=0.02,
):
    """Train only on declared train cases and atomically publish auditable evidence."""
    library = _require_torch()
    cases, output = deepcopy(list(cases)), _output_path(output)
    if output.exists():
        raise ValueError("PPO output must be a new directory")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    integer_arguments = (episodes, episodes_per_update, batch_size, update_epochs)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_arguments):
        raise ValueError("episode, update, epoch and batch counts must be integers")
    if episodes <= 0 or episodes_per_update <= 0 or episodes % episodes_per_update:
        raise ValueError("episodes must be positive and divisible by episodes_per_update")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in checkpoint_episodes)
        or any(value < 0 or value > episodes for value in checkpoint_episodes)
        or 0 not in checkpoint_episodes
        or any(value not in (0, episodes) and value % episodes_per_update for value in checkpoint_episodes)
    ):
        raise ValueError("checkpoints must include zero and follow update boundaries or the final episode")
    finite_hyperparameters = (
        gamma, gae_lambda, clip_epsilon, learning_rate, max_grad_norm, std
    )
    if (
        not all(np.isfinite(value) for value in finite_hyperparameters)
        or not 0 < gamma <= 1
        or not 0 <= gae_lambda <= 1
        or not 0 < std < 1
        or not 0 < learning_rate
        or not 0 < clip_epsilon < 1
        or not max_grad_norm > 0
        or batch_size <= 0
        or update_epochs <= 0
        or (target_kl is not None and (not np.isfinite(target_kl) or target_kl <= 0))
    ):
        raise ValueError("invalid PPO hyperparameters")
    contract = policy_contract(cases, purpose="rl_residual", nominal_kind="friction")
    train_cases = [case for case in cases if case["split"] == "train"]
    if not train_cases:
        raise ValueError("case plan contains no training cases")
    seed_sequences = np.random.SeedSequence(seed).spawn(3)
    schedule_rng, reset_rng = (np.random.default_rng(item) for item in seed_sequences[:2])
    torch_seed = int(seed_sequences[2].generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))
    schedule = []
    while len(schedule) < episodes:
        schedule.extend(schedule_rng.permutation(len(train_cases)).tolist())
    schedule = schedule[:episodes]
    reset_seeds = reset_rng.integers(0, 2**31 - 1, size=episodes).tolist()
    with library.random.fork_rng(devices=[]):
        library.manual_seed(torch_seed)
        model = ActorCritic().double().cpu()
    torch_generator = library.Generator(device="cpu").manual_seed(torch_seed)
    optimizer = library.optim.Adam(model.parameters(), lr=learning_rate)
    sources, trainer_path = _source_hashes(), Path(__file__).resolve()
    scripts = {
        "train_surface_ppo.py": _sha256(trainer_path),
        "surface_mlp_actor.py": _sha256(trainer_path.with_name("surface_mlp_actor.py")),
    }
    hyperparameters = {
        "episodes": episodes, "episodes_per_update": episodes_per_update,
        "checkpoint_episodes": sorted(set(checkpoint_episodes)), "gamma": gamma,
        "gae_lambda": gae_lambda, "clip_epsilon": clip_epsilon,
        "learning_rate": learning_rate, "update_epochs": update_epochs,
        "batch_size": batch_size, "max_grad_norm": max_grad_norm, "fixed_std": std,
        "target_kl": target_kl, "physics_discount": "gamma**(physics_substeps/10)",
        "gae_trace": "lambda**(physics_substeps/10)",
        "reward_normalization": None, "extra_failure_penalty": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-ppo-", dir=output.parent) as temporary:
        staging = Path(temporary) / "training"
        staging.mkdir()
        rows, update_rows, checkpoints, pending = [], [], {}, []
        try:
            if 0 in checkpoint_episodes:
                checkpoints[0] = _save_checkpoint(staging, 0, model, contract)
            for episode_index, (case_index, reset_seed) in enumerate(
                zip(schedule, reset_seeds), start=1
            ):
                case = train_cases[case_index]
                (
                    data, trace, decisions, reset_info, failure, physical_metrics, fatal
                ) = _run_episode(
                    case, int(reset_seed), model, std, torch_generator
                )
                if fatal is None:
                    _finish_episode(data, model, gamma, gae_lambda)
                trace.update({f"decision_{key.rstrip('_')}": value for key, value in data.items()})
                stem = f"episode_{episode_index:03d}"
                np.savez_compressed(staging / f"{stem}.npz", **trace)
                _write_json(staging / f"{stem}.json", {
                    "case_id": case["case_id"], "group_id": case["group_id"],
                    "split": case["split"], "requested_reset_seed": int(reset_seed),
                    "actual_simulation_seed": reset_info.get("simulation_seed"),
                    "exception_detail": failure, "action_diagnostics": _action_diagnostics(data),
                    "physical_metrics": physical_metrics, "decisions": decisions,
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
                        reason for decision in decisions
                        for reason in decision["termination_reasons"]
                    )),
                    "exception_detail": failure, "trace_file": f"{stem}.npz",
                    "events_file": f"{stem}.json", "action_diagnostics": _action_diagnostics(data),
                    "physical_metrics": physical_metrics,
                }
                rows.append(row)
                if fatal is not None:
                    raise fatal
                pending.append(data)
                if episode_index % episodes_per_update == 0:
                    update_rows.append({"after_episode": episode_index, **_update(
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
        except Exception as error:
            failure = {
                "schema": "surface_ppo_failed_training_v1", "error": str(error),
                "failure_kind": type(error).__name__,
                "completed_attempted_episodes": len(rows), "runs": rows,
                "source_and_assets_sha256": sources, "script_sha256": scripts,
            }
            _write_json(staging / "failure.json", failure)
            (staging / "FAILED").write_text(_sha256(staging / "failure.json") + "\n")
            _publish(staging, output)
            raise RuntimeError(f"PPO failure; partial record published at {output}") from error
        if _source_hashes() != sources or any(
            _sha256(trainer_path.with_name(name)) != digest for name, digest in scripts.items()
        ):
            raise RuntimeError("source or trainer scripts changed during PPO training")
        artifact_hashes = {
            path.name: _sha256(path) for path in sorted(staging.iterdir()) if path.is_file()
        }
        manifest = {
            "schema": SCHEMA, "purpose": "rl_residual", "nominal_kind": "friction",
            "scope": "bounded_public-training pilot; not convergence or validation evidence",
            "seed": seed, "torch_seed": torch_seed, "hyperparameters": hyperparameters,
            "architecture": "49-32-tanh-32-tanh-3-tanh; separate same-width value MLP",
            "action_distribution": "fixed-std pre-tanh Gaussian with exact stored-latent Jacobian",
            "case_plan": cases, "training_case_ids": [case["case_id"] for case in train_cases],
            "training_case_count": len(train_cases),
            "training_group_ids": sorted({case["group_id"] for case in train_cases}),
            "schedule_case_ids": [train_cases[index]["case_id"] for index in schedule],
            "reset_seeds": [int(value) for value in reset_seeds], "runs": rows,
            "updates": update_rows, "failed_episodes": sum(row["failed"] for row in rows),
            "actual_physics_steps": sum(row["physics_steps"] for row in rows),
            "checkpoints": {str(key): value for key, value in sorted(checkpoints.items())},
            "policy_contract": contract, "source_and_assets_sha256": sources,
            "script_sha256": scripts,
            "runtime": {
                "device": "cpu", "dtype": "float64",
                "torch_num_threads": library.get_num_threads(),
                "deterministic_algorithms": library.are_deterministic_algorithms_enabled(),
            },
            "versions": {
                "python": platform.python_version(), "numpy": np.__version__,
                "torch": library.__version__, "mujoco": version("mujoco"),
                "gymnasium": version("gymnasium"),
            },
            "artifact_sha256": artifact_hashes,
        }
        _write_json(staging / "manifest.json", manifest)
        if _source_hashes() != sources or any(
            _sha256(trainer_path.with_name(name)) != digest for name, digest in scripts.items()
        ):
            raise RuntimeError("source or trainer scripts changed before PPO publication")
        (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
        _publish(staging, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    args = parser.parse_args(argv)
    result = train_ppo(
        json.loads(args.cases.read_text(encoding="utf-8")), args.output,
        seed=args.seed, episodes=args.episodes,
    )
    print(json.dumps({"output": str(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
