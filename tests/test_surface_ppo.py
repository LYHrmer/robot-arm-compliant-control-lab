import hashlib
import json
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
from tools import train_surface_ppo as ppo
from tools.surface_mlp_actor import actor_from_artifact, save_mlp_candidate


def _torch():
    return pytest.importorskip("torch")


def _cases(duration=0.02):
    cases = []
    for group, yaw in enumerate((-15.0, 0.0, 15.0)):
        for noise_seed in (11, 29):
            cases.append(
                {
                    "case_id": f"g{group}_seed{noise_seed}",
                    "scenario": asdict(
                        SurfaceScenario(name=f"ppo_test_{group}", wall_yaw_deg=yaw)
                    ),
                    "config": asdict(
                        SurfaceSimulationConfig(
                            duration=duration, seed=noise_seed, contact_model="smooth"
                        )
                    ),
                    "task": asdict(LearningSurfaceTask(yaw_deg=yaw)),
                    "controller_frame_rotation": yaw_frame(yaw).rotation.tolist(),
                    "nominal_kind": "friction",
                }
            )
    return split_cases(cases, seed=37, train_fraction=0.34, validation_fraction=0.33)


def test_gae_terminated_truncated_partial_steps_and_episode_boundaries():
    gamma, lam = 0.99, 0.95
    rewards = np.array([1.0, 2.0, 3.0, 4.0])
    values = np.array([0.5, 0.2, 1.0, 0.4])
    next_values = np.array([0.2, 9.0, 0.4, 0.7])
    terminated = np.array([False, True, False, False])
    truncated = np.array([False, False, False, True])
    valid = np.ones(4, dtype=bool)
    substeps = np.array([10, 5, 3, 10])

    advantage, returns = ppo.compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        valid,
        substeps,
        gamma=gamma,
        gae_lambda=lam,
    )

    delta_1 = 2.0 - 0.2  # Termination forbids the deliberately huge next value.
    delta_0 = 1.0 + gamma * 0.2 - 0.5
    delta_3 = 4.0 + gamma * 0.7 - 0.4  # Valid time truncation bootstraps.
    fractional_discount = gamma ** (3 / 10)
    fractional_trace = lam ** (3 / 10)
    delta_2 = 3.0 + fractional_discount * 0.4 - 1.0
    expected = np.array(
        [
            delta_0 + gamma * lam * delta_1,
            delta_1,
            delta_2 + fractional_discount * fractional_trace * delta_3,
            delta_3,
        ]
    )
    np.testing.assert_allclose(advantage, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(returns, expected + values, rtol=0, atol=1e-14)


@pytest.mark.parametrize("terminated,truncated", [(False, True), (True, False)])
def test_gae_invalid_next_observation_never_bootstraps(terminated, truncated):
    advantage, returns = ppo.compute_gae(
        [2.0],
        [0.25],
        [np.nan],
        [terminated],
        [truncated],
        [False],
        [2],
    )
    np.testing.assert_array_equal(advantage, [1.75])
    np.testing.assert_array_equal(returns, [2.0])


def test_tanh_log_prob_matches_torch_transformed_distribution():
    torch = _torch()
    mean = torch.tensor([0.1, -0.3, 0.25], dtype=torch.float64)
    latent = torch.tensor([0.7, -1.1, 0.02], dtype=torch.float64)
    std = 0.05
    actual = ppo.tanh_normal_log_prob(latent, mean, std)
    transformed = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(mean, torch.full_like(mean, std)),
        [torch.distributions.TanhTransform(cache_size=1)],
    )
    expected = torch.distributions.Independent(transformed, 1).log_prob(torch.tanh(latent))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_clipped_surrogate_has_finite_gradient():
    torch = _torch()
    new = torch.tensor([-0.9, -1.2, -0.3], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([-1.0, -1.0, -0.5], dtype=torch.float64)
    advantage = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    loss = ppo.ppo_clipped_loss(new, old, advantage)
    loss.backward()
    assert torch.isfinite(loss)
    assert new.grad is not None and torch.isfinite(new.grad).all()


def test_zero_initial_actor_and_nonzero_restricted_export_have_exact_parity(tmp_path):
    torch = _torch()
    torch.manual_seed(7)
    model = ppo.ActorCritic().double()
    observations = np.random.default_rng(9).uniform(-3, 3, size=(5, 49))
    tensor = torch.as_tensor(observations, dtype=torch.float64)
    with torch.no_grad():
        model_actions = model.deterministic_action(tensor).numpy()
    np.testing.assert_array_equal(model_actions, np.zeros((5, 3)))

    with torch.no_grad():
        model.actor_mean.weight.copy_(
            torch.linspace(-0.3, 0.2, 3 * 32, dtype=torch.float64).reshape(3, 32)
        )
        model.actor_mean.bias.copy_(torch.tensor([0.1, -0.05, 0.2], dtype=torch.float64))
        model_actions = model.deterministic_action(tensor).numpy()
    assert np.any(model_actions != 0)

    cases = _cases()
    contract = policy_contract(cases, purpose="rl_residual", nominal_kind="friction")
    candidate = tmp_path / "initial.json"
    save_mlp_candidate(candidate, ppo.export_actor_layers(model), contract=contract)
    actor, _ = actor_from_artifact(load_policy_artifact(candidate, expected_contract=contract))
    exported_actions = np.stack([actor(observation) for observation in observations])
    np.testing.assert_allclose(exported_actions, model_actions, rtol=0, atol=1e-12)


def test_update_reports_post_optimizer_step_kl_for_one_minibatch():
    torch = _torch()
    torch.manual_seed(5)
    model = ppo.ActorCritic().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    observations = torch.as_tensor(
        np.random.default_rng(4).normal(size=(6, 49)), dtype=torch.float64
    )
    latent = torch.tensor(
        [[0.03, -0.02, 0.01], [0.07, 0.01, -0.04], [-0.06, 0.02, 0.05],
         [0.04, -0.08, 0.03], [-0.02, 0.06, -0.01], [0.08, -0.03, 0.02]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        old_log_prob = ppo.tanh_normal_log_prob(latent, model.policy_mean(observations), 0.05)
        value = model.value(observations)
    episode = {
        "reward": np.ones(6),
        "observation": observations.numpy(),
        "latent": latent.numpy(),
        "old_log_prob": old_log_prob.numpy(),
        "advantage": np.array([1.0, -0.5, 0.7, -1.2, 0.2, -0.2]),
        "return_": (value + torch.linspace(-0.2, 0.3, 6)).numpy(),
    }
    report = ppo._update(
        model,
        optimizer,
        [episode],
        std=0.05,
        clip_epsilon=0.2,
        update_epochs=1,
        batch_size=6,
        max_grad_norm=0.5,
        target_kl=None,
        torch_generator=torch.Generator().manual_seed(3),
    )
    with torch.no_grad():
        updated = ppo.tanh_normal_log_prob(latent, model.policy_mean(observations), 0.05)
        expected = float((old_log_prob - updated).mean())
    assert report["minibatches"] == 1
    assert report["approx_kl"] != 0.0
    assert report["approx_kl"] == pytest.approx(expected, abs=1e-14)


def test_two_episode_genuine_environment_run_is_balanced_auditable_and_atomic(tmp_path):
    torch = _torch()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    cases = _cases()
    output = ppo.train_ppo(
        cases,
        tmp_path / "ppo",
        seed=47,
        episodes=2,
        episodes_per_update=1,
        checkpoint_episodes=(0, 2),
        update_epochs=1,
        batch_size=8,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (output / "COMPLETE").read_text().strip() == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    train_ids = [case["case_id"] for case in cases if case["split"] == "train"]
    assert manifest["training_case_count"] == 2
    assert sorted(manifest["schedule_case_ids"]) == sorted(train_ids)
    assert len(manifest["runs"]) == 2 and len(manifest["updates"]) == 2
    assert manifest["actual_physics_steps"] == 20
    assert set(manifest["checkpoints"]) == {"0", "2"}
    assert manifest["nominal_kind"] == "friction"
    assert all(row["case_id"] in train_ids for row in manifest["runs"])
    assert all(row["physical_metrics"] for row in manifest["runs"])
    assert all(row["action_diagnostics"]["latent_abs_max"] is not None for row in manifest["runs"])
    assert not list(output.glob("*.pt")) and not list(output.glob("*.pkl"))
    for row in manifest["runs"]:
        with np.load(output / row["trace_file"], allow_pickle=False) as trace:
            assert trace["decision_observation"].shape == (1, 49)
            assert trace["decision_latent"].shape == (1, 3)
            assert trace["decision_physics_substeps"].tolist() == [10]
            assert len(trace["execution_safety_reasons"]) == 10
    assert torch.get_num_threads() == previous_threads
    assert torch.are_deterministic_algorithms_enabled() == previous_deterministic


def test_numerical_failure_publishes_visible_partial_record_without_complete(
    tmp_path, monkeypatch
):
    _torch()

    def fail_update(*args, **kwargs):
        raise ppo.NumericalTrainingError("injected nonfinite gradient")

    monkeypatch.setattr(ppo, "_update", fail_update)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="partial record published"):
        ppo.train_ppo(
            _cases(),
            output,
            seed=11,
            episodes=1,
            episodes_per_update=1,
            checkpoint_episodes=(0,),
        )
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["completed_attempted_episodes"] == 1
    assert (output / "FAILED").is_file()
    assert not (output / "COMPLETE").exists()


def test_unexpected_environment_exception_aborts_instead_of_training_prefix(
    tmp_path, monkeypatch
):
    _torch()

    def fail_constructor(case):
        raise RuntimeError("injected environment constructor fault")

    monkeypatch.setattr(ppo, "_environment", fail_constructor)
    output = tmp_path / "broken"
    with pytest.raises(RuntimeError, match="partial record published"):
        ppo.train_ppo(
            _cases(), output, seed=47, episodes=1, episodes_per_update=1,
            checkpoint_episodes=(0,),
        )
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_kind"] == "EpisodeExecutionError"
    assert failure["completed_attempted_episodes"] == 1
    assert failure["runs"][0]["policy_steps"] == 0
    assert not failure.get("updates")
