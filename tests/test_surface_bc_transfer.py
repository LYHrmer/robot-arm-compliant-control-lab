import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_policy import friction_teacher_action
from compliant_control_lab.surface_policy_artifact import load_policy_artifact, policy_contract
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    yaw_frame,
)
from compliant_control_lab.surface_splits import split_cases
from tools import train_surface_bc as frozen_bc
from tools import train_surface_bc_transfer as transfer
from tools.surface_mlp_actor import actor_from_artifact, save_mlp_candidate


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


def _arrays(seed=3, train_count=80, validation_count=40):
    rng = np.random.default_rng(seed)
    train_x = rng.uniform(-1, 1, size=(train_count, 49))
    validation_x = rng.uniform(-1, 1, size=(validation_count, 49))
    train_y = np.stack([friction_teacher_action(row[:33]) for row in train_x])
    validation_y = np.stack([friction_teacher_action(row[:33]) for row in validation_x])
    return train_x, train_y, validation_x, validation_y


def _dataset(tmp_path):
    path = tmp_path / "dataset"
    path.mkdir()
    episode = path / "episode.bin"
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
    (path / "manifest.json").write_text(encoded)
    (path / "COMPLETE").write_text(hashlib.sha256(encoded.encode()).hexdigest() + "\n")
    return path


def _loader(arrays, calls, *, shared_group=False):
    train_x, train_y, validation_x, validation_y = arrays

    def load(_, split):
        calls.append(split)
        if split == "train":
            observations, actions, groups = train_x, train_y, ["same" if shared_group else "train"]
        elif split == "validation":
            observations, actions, groups = (
                validation_x,
                validation_y,
                ["same" if shared_group else "validation"],
            )
        else:
            raise AssertionError("development_test must not be loaded")
        count = len(observations)
        return {
            "observations": observations,
            "actions": actions,
            "episode_id": np.repeat(
                [f"{split}-a", f"{split}-b"], [count // 2, count - count // 2]
            ),
            "metadata": {"split": split, "group_ids": groups},
        }

    return load


def test_arm_indices_are_exact_and_both_masks_preserve_teacher_equation():
    assert transfer.ARMS["teacher_inputs"] == (0, 10, 11)
    assert set(range(49)) - set(transfer.ARMS["drop_previous_residual"]) == {14, 15, 16}
    observations = np.random.default_rng(2).uniform(-1, 1, size=(12, 49))
    expected = np.stack([friction_teacher_action(row[:33]) for row in observations])
    for arm in transfer.ARMS:
        masked = transfer._mask_observations(observations, transfer._input_mask(arm))
        actual = np.stack([friction_teacher_action(row[:33]) for row in masked])
        np.testing.assert_array_equal(actual, expected)


def test_masked_training_does_not_mutate_arrays_or_update_inactive_input_weights():
    train_x, train_y, validation_x, validation_y = _arrays()
    originals = tuple(value.copy() for value in (train_x, train_y, validation_x, validation_y))
    mask = transfer._input_mask("teacher_inputs")
    masked_train = transfer._mask_observations(train_x, mask)
    masked_validation = transfer._mask_observations(validation_x, mask)
    initial = frozen_bc._build_model(torch, 11)
    model, _, _ = frozen_bc._fit_behavior_clone(
        masked_train,
        train_y,
        masked_validation,
        validation_y,
        seed=11,
        epochs=3,
        batch_size=32,
        learning_rate=0.003,
    )
    for actual, expected in zip((train_x, train_y, validation_x, validation_y), originals):
        np.testing.assert_array_equal(actual, expected)
    assert not masked_train[:, ~mask].any()
    np.testing.assert_array_equal(
        model[0].weight.detach().numpy()[:, ~mask],
        initial[0].weight.detach().numpy()[:, ~mask],
    )


def test_folded_nonzero_export_matches_masked_torch_and_ignores_inactive_noise(tmp_path):
    torch.manual_seed(19)
    model = frozen_bc._build_model(torch, 19)
    observations = np.random.default_rng(7).uniform(-1, 1, size=(9, 49))
    mask = transfer._input_mask("teacher_inputs")
    masked = transfer._mask_observations(observations, mask)
    layers = transfer._fold_input_mask(frozen_bc._model_layers(model), mask)
    candidate = tmp_path / "candidate.json"
    contract = policy_contract(
        _cases(), purpose="il_friction_teacher", nominal_kind="adaptive"
    )
    save_mlp_candidate(candidate, layers, contract=contract, activation="tanh")
    actor, _ = actor_from_artifact(
        load_policy_artifact(candidate, expected_contract=contract)
    )
    with torch.no_grad():
        expected = model(torch.from_numpy(masked)).numpy()
    actual = np.stack([actor(row) for row in observations])
    assert np.any(np.abs(actual) > 1e-6)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    noisy = observations.copy()
    noisy[:, ~mask] = np.random.default_rng(8).uniform(-3, 3, size=(len(noisy), (~mask).sum()))
    np.testing.assert_array_equal(np.stack([actor(row) for row in noisy]), actual)


def test_masked_fit_is_deterministic():
    train_x, train_y, validation_x, validation_y = _arrays(seed=13)
    mask = transfer._input_mask("drop_previous_residual")
    arguments = (
        transfer._mask_observations(train_x, mask),
        train_y,
        transfer._mask_observations(validation_x, mask),
        validation_y,
    )
    keyword = {"seed": 29, "epochs": 3, "batch_size": 32, "learning_rate": 0.003}
    first, first_epoch, first_history = frozen_bc._fit_behavior_clone(*arguments, **keyword)
    second, second_epoch, second_history = frozen_bc._fit_behavior_clone(*arguments, **keyword)
    assert first_epoch == second_epoch and first_history == second_history
    for left, right in zip(first.parameters(), second.parameters()):
        np.testing.assert_array_equal(left.detach().numpy(), right.detach().numpy())


def test_transfer_loads_only_train_validation_and_publishes_masked_candidate(
    tmp_path, monkeypatch
):
    arrays = _arrays()
    originals = tuple(value.copy() for value in arrays)
    calls = []
    monkeypatch.setattr(
        transfer.surface_transitions, "load_transition_batch", _loader(arrays, calls)
    )
    output = transfer.train_bc_transfer(
        _dataset(tmp_path),
        tmp_path / "run",
        arm="teacher_inputs",
        seed=11,
        epochs=2,
        batch_size=32,
        learning_rate=0.003,
        overfit_steps=2,
    )
    assert calls == ["train", "validation"]
    for actual, expected in zip(arrays, originals):
        np.testing.assert_array_equal(actual, expected)
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert manifest["schema"] == "surface_behavior_clone_transfer_run_v1"
    assert manifest["arm"] == report["arm"] == "teacher_inputs"
    assert manifest["input_mask"] == report["input_mask"]
    assert manifest["dataset"]["manifest_sha256"]
    assert set(manifest["trainer_source_sha256"]) >= {"transfer_trainer", "trainer"}
    assert manifest["development_test_used"] is False
    assert report["selection"]["selected_epoch"] in (1, 2)
    assert report["diagnostics"]["folded_export"]["passed"]
    assert (output / "COMPLETE").read_text().strip() == hashlib.sha256(
        (output / "manifest.json").read_bytes()
    ).hexdigest()


def test_group_overlap_and_invalid_arm_fail_before_publication(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path)
    with pytest.raises(ValueError, match="arm must be exactly"):
        transfer.train_bc_transfer(dataset, tmp_path / "invalid", arm="full49", seed=11)
    assert not (tmp_path / "invalid").exists()

    calls = []
    monkeypatch.setattr(
        transfer.surface_transitions,
        "load_transition_batch",
        _loader(_arrays(), calls, shared_group=True),
    )
    with pytest.raises(ValueError, match="overlap"):
        transfer.train_bc_transfer(
            dataset,
            tmp_path / "overlap",
            arm="drop_previous_residual",
            seed=11,
        )
    assert calls == ["train", "validation"]
    assert not (tmp_path / "overlap").exists()


def test_runtime_failure_publishes_failed_directory(tmp_path, monkeypatch):
    arrays = _arrays()
    monkeypatch.setattr(
        transfer.surface_transitions, "load_transition_batch", _loader(arrays, [])
    )
    monkeypatch.setattr(
        transfer.frozen_bc,
        "_overfit_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="partial record published"):
        transfer.train_bc_transfer(
            _dataset(tmp_path),
            output,
            arm="teacher_inputs",
            seed=47,
            overfit_steps=2,
        )
    assert (output / "FAILED").is_file()
    assert not (output / "COMPLETE").exists()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["arm"] == "teacher_inputs"
    assert failure["error"] == "injected"
