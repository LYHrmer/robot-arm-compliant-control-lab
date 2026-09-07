import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_policy_artifact import load_policy_artifact, policy_contract
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    yaw_frame,
)
from compliant_control_lab.surface_splits import split_cases
from tools import train_surface_bc
from tools.surface_mlp_actor import actor_from_artifact, save_mlp_candidate


def _arrays(seed=7, train_count=96, validation_count=48):
    rng = np.random.default_rng(seed)
    teacher_weights = rng.normal(scale=0.12, size=(3, 49))
    train_x = rng.uniform(-1, 1, size=(train_count, 49))
    validation_x = rng.uniform(-1, 1, size=(validation_count, 49))
    train_y = np.tanh(train_x @ teacher_weights.T)
    validation_y = np.tanh(validation_x @ teacher_weights.T)
    return train_x, train_y, validation_x, validation_y


def _cases():
    raw = [
        {
            "case_id": f"yaw{yaw}",
            "scenario": asdict(SurfaceScenario(wall_yaw_deg=yaw)),
            "config": asdict(SurfaceSimulationConfig(contact_model="smooth")),
            "task": asdict(LearningSurfaceTask(yaw_deg=yaw)),
            "controller_frame_rotation": yaw_frame(yaw).rotation.tolist(),
            "nominal_kind": "adaptive",
        }
        for yaw in (-15, 0, 15)
    ]
    return split_cases(raw, seed=20260906, train_fraction=0.7, validation_fraction=0.2)


def _fit(train_x, train_y, validation_x, validation_y, seed=11, epochs=8):
    return train_surface_bc._fit_behavior_clone(
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=seed,
        epochs=epochs,
        batch_size=32,
        learning_rate=0.003,
    )


def test_autograd_matches_numerical_gradient_and_training_loss_decreases():
    train_x, train_y, validation_x, validation_y = _arrays()
    model = train_surface_bc._build_model(torch, 13)
    x = torch.from_numpy(train_x[:8])
    y = torch.from_numpy(train_y[:8])
    parameter = model[0].weight
    loss = (model(x) - y).square().mean()
    loss.backward()
    analytic = float(parameter.grad[0, 0])
    original = float(parameter[0, 0].detach())
    epsilon = 1e-6
    with torch.no_grad():
        parameter[0, 0] = original + epsilon
        plus = float((model(x) - y).square().mean())
        parameter[0, 0] = original - epsilon
        minus = float((model(x) - y).square().mean())
        parameter[0, 0] = original
    numerical = (plus - minus) / (2 * epsilon)
    assert analytic == pytest.approx(numerical, rel=2e-5, abs=1e-8)

    _, _, history = _fit(train_x, train_y, validation_x, validation_y)
    assert history[-1]["train_action_mse"] < history[0]["train_action_mse"]


def test_same_seed_produces_bitwise_reproducible_selected_weights():
    arrays = _arrays(seed=9)
    first, first_epoch, first_history = _fit(*arrays, seed=29)
    second, second_epoch, second_history = _fit(*arrays, seed=29)
    assert first_epoch == second_epoch
    assert first_history == second_history
    for left, right in zip(first.parameters(), second.parameters()):
        np.testing.assert_array_equal(left.detach().numpy(), right.detach().numpy())


def test_exported_numpy_actor_matches_selected_torch_model(tmp_path):
    arrays = _arrays(seed=3)
    model, _, _ = _fit(*arrays, seed=47)
    contract = policy_contract(
        _cases(), purpose="il_friction_teacher", nominal_kind="adaptive"
    )
    candidate = tmp_path / "candidate.json"
    save_mlp_candidate(
        candidate,
        train_surface_bc._model_layers(model),
        contract=contract,
        activation="tanh",
    )
    artifact = load_policy_artifact(candidate, expected_contract=contract)
    actor, _ = actor_from_artifact(artifact)
    observation = arrays[2][5]
    with torch.no_grad():
        expected = model(torch.from_numpy(observation)).numpy()
    np.testing.assert_allclose(actor(observation), expected, rtol=0, atol=2e-15)


def test_checkpoint_selection_uses_validation_loss_and_keeps_earliest_minimum():
    train_x = np.zeros((64, 49))
    validation_x = np.zeros((32, 49))
    train_y = np.full((64, 3), 0.8)
    validation_y = np.full((32, 3), -0.8)
    _, selected_epoch, history = _fit(
        train_x, train_y, validation_x, validation_y, seed=5, epochs=6
    )
    expected = min(history, key=lambda row: (row["validation_action_mse"], row["epoch"]))
    assert selected_epoch == expected["epoch"]
    assert history[-1]["train_action_mse"] < history[0]["train_action_mse"]
    assert selected_epoch != min(history, key=lambda row: row["train_action_mse"])["epoch"]


def test_train_bc_loads_exact_splits_without_mutating_inputs_and_publishes_run(
    tmp_path, monkeypatch
):
    train_x, train_y, validation_x, validation_y = _arrays(train_count=80)
    original = tuple(value.copy() for value in (train_x, train_y, validation_x, validation_y))
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    episode = dataset / "episode.bin"
    episode.write_bytes(b"identity only")
    manifest = {
        "corpus_scope": "full",
        "nominal_kind": "adaptive",
        "action_stage": "normalized_local_residual",
        "cases": _cases(),
        "episodes": [
            {
                "file": episode.name,
                "sha256": hashlib.sha256(episode.read_bytes()).hexdigest(),
            }
        ],
        "source_and_assets_sha256": {"fixture": hashlib.sha256(b"fixture").hexdigest()},
    }
    encoded = json.dumps(manifest, sort_keys=True) + "\n"
    (dataset / "manifest.json").write_text(encoded)
    (dataset / "COMPLETE").write_text(hashlib.sha256(encoded.encode()).hexdigest() + "\n")
    calls = []

    def load(_, split):
        calls.append(split)
        if split == "train":
            observations, actions = train_x, train_y
            count, groups = len(train_x), ["train-group"]
        elif split == "validation":
            observations, actions = validation_x, validation_y
            count, groups = len(validation_x), ["validation-group"]
        else:
            raise AssertionError("development_test must not be loaded")
        return {
            "observations": observations,
            "actions": actions,
            "episode_id": np.repeat([f"{split}-a", f"{split}-b"], [count // 2, count - count // 2]),
            "metadata": {"split": split, "group_ids": groups},
        }

    monkeypatch.setattr(train_surface_bc.surface_transitions, "load_transition_batch", load)
    previous_threads = torch.get_num_threads()
    output = train_surface_bc.train_bc(
        dataset,
        tmp_path / "run",
        seed=11,
        epochs=2,
        batch_size=32,
        learning_rate=0.003,
        overfit_steps=2,
    )
    assert torch.get_num_threads() == previous_threads
    assert calls == ["train", "validation"]
    for actual, expected in zip((train_x, train_y, validation_x, validation_y), original):
        np.testing.assert_array_equal(actual, expected)
    assert output == tmp_path / "run"
    assert {"selected_model.json", "report.json", "epoch_losses.json", "epoch_losses.csv"} < {
        path.name for path in output.iterdir()
    }
    assert (output / "COMPLETE").read_text().strip() == hashlib.sha256(
        (output / "manifest.json").read_bytes()
    ).hexdigest()
    report = json.loads((output / "report.json").read_text())
    run_manifest = json.loads((output / "manifest.json").read_text())
    assert run_manifest["runtime"]["torch_num_threads"] == 1
    assert run_manifest["dataset"]["logical_name"] == "dataset"
    assert "path" not in run_manifest["dataset"]
    assert run_manifest["artifact_sha256"]["selected_model.json"] == hashlib.sha256(
        (output / "selected_model.json").read_bytes()
    ).hexdigest()
    assert report["data_use"]["development_test_loaded_for_training_or_selection"] is False
    assert report["selection"]["split"] == "validation"
    assert len(report["diagnostics"]["train"]["per_episode"]) == 2
    assert len(report["diagnostics"]["validation"]["per_episode"]) == 2
    assert set(report["diagnostics"]["validation"]["model_mse_by_axis"]) == {
        "normal",
        "tangent1",
        "tangent2",
    }
    assert report["diagnostics"]["train_only_64_sample_overfit"]["sample_count"] == 64
    tiny = report["diagnostics"]["train_only_64_sample_overfit"]
    assert tiny["loss_decreased"]
    assert tiny["final_to_initial_loss_ratio"] == pytest.approx(
        tiny["final_action_mse"] / tiny["initial_action_mse"]
    )
