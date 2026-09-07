"""Deterministic CPU behavior cloning for the audited surface teacher dataset.

``train_bc`` returns the newly published run directory.  Its deployable policy is
``selected_model.json`` and ``report.json`` records why that epoch was selected.
The development-test split is never loaded or used for optimization/selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import tempfile
from copy import deepcopy
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from compliant_control_lab import surface_policy_artifact, surface_transitions
from compliant_control_lab.surface_experiment import _output_path
from tools import surface_mlp_actor

ARCHITECTURE = (49, 32, 32, 3)
AXES = ("normal", "tangent1", "tangent2")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "behavior-cloning training requires the optional CPU PyTorch dependency"
        ) from error
    return torch


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_inputs(seed, epochs, batch_size, learning_rate, overfit_steps):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    epochs = _validate_positive_integer("epochs", epochs)
    batch_size = _validate_positive_integer("batch_size", batch_size)
    overfit_steps = _validate_positive_integer("overfit_steps", overfit_steps)
    if (
        isinstance(learning_rate, (bool, np.bool_))
        or not isinstance(learning_rate, Real)
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    return int(seed), epochs, batch_size, float(learning_rate), overfit_steps


def _source_identity() -> dict[str, str]:
    paths = {
        "trainer": Path(__file__).resolve(),
        "transition_loader": Path(surface_transitions.__file__).resolve(),
        "policy_artifact": Path(surface_policy_artifact.__file__).resolve(),
        "numpy_mlp_exporter": Path(surface_mlp_actor.__file__).resolve(),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _runtime_identity(torch) -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": "cpu",
        "dtype": "float64",
        "torch_num_threads": torch.get_num_threads(),
    }


def _dataset_snapshot(path: Path) -> dict:
    manifest_path = path / "manifest.json"
    complete_path = path / "COMPLETE"
    manifest_sha256 = _sha256(manifest_path)
    if complete_path.read_text(encoding="utf-8").strip() != manifest_sha256:
        raise ValueError("dataset COMPLETE does not match manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {}
    for entry in manifest.get("episodes", []):
        filename = entry["file"]
        episode_path = path / filename
        if Path(filename).name != filename or episode_path.is_symlink():
            raise ValueError("unsafe dataset episode path")
        files[filename] = _sha256(episode_path)
        if files[filename] != entry["sha256"]:
            raise ValueError("dataset episode hash differs from manifest")
    return {
        "manifest_sha256": manifest_sha256,
        "complete_sha256": _sha256(complete_path),
        "episode_sha256": files,
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _training_arrays(batch: dict, expected_split: str):
    metadata = batch.get("metadata", {})
    if metadata.get("split") != expected_split:
        raise ValueError(f"loader returned the wrong split for {expected_split}")
    observations = np.array(batch["observations"], dtype=np.float64, copy=True)
    actions = np.array(batch["actions"], dtype=np.float64, copy=True)
    episode_ids = np.array(batch["episode_id"], dtype=str, copy=True)
    if (
        observations.ndim != 2
        or observations.shape[1:] != (49,)
        or actions.shape != (len(observations), 3)
        or episode_ids.shape != (len(observations),)
        or not len(observations)
    ):
        raise ValueError(f"invalid {expected_split} actor arrays")
    if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(actions)):
        raise ValueError(f"nonfinite {expected_split} actor arrays")
    if np.max(np.abs(observations)) > 3 or np.max(np.abs(actions)) > 1:
        raise ValueError(f"out-of-contract {expected_split} actor arrays")
    groups = metadata.get("group_ids")
    if not isinstance(groups, list) or not groups or len(set(groups)) != len(groups):
        raise ValueError(f"invalid {expected_split} group identities")
    return observations, actions, episode_ids, set(groups)


def _build_model(torch, seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(49, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 3),
            torch.nn.Tanh(),
        )
    return model.to(device="cpu", dtype=torch.float64)


def _mse_values(torch, prediction, target) -> tuple[float, list[float]]:
    error = (prediction - target).square()
    overall = float(error.mean().item())
    by_axis = [float(value) for value in error.mean(dim=0).tolist()]
    if not np.isfinite(overall) or not np.all(np.isfinite(by_axis)):
        raise RuntimeError("training produced a nonfinite loss")
    return overall, by_axis


def _evaluate_model(torch, model, observations, actions):
    model.eval()
    with torch.no_grad():
        return _mse_values(torch, model(observations), actions)


def _fit_behavior_clone(
    train_observations,
    train_actions,
    validation_observations,
    validation_actions,
    *,
    seed,
    epochs,
    batch_size,
    learning_rate,
):
    """Fit all training rows and select strictly by finite validation action MSE."""
    torch = _require_torch()
    train_x = torch.from_numpy(np.ascontiguousarray(train_observations))
    train_y = torch.from_numpy(np.ascontiguousarray(train_actions))
    validation_x = torch.from_numpy(np.ascontiguousarray(validation_observations))
    validation_y = torch.from_numpy(np.ascontiguousarray(validation_actions))
    model = _build_model(torch, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    shuffle = torch.Generator(device="cpu")
    shuffle.manual_seed(seed + 1)
    history = []
    selected_epoch, selected_validation, selected_state = None, float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=shuffle)
        for start in range(0, len(train_x), batch_size):
            index = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[index])
            loss = (prediction - train_y[index]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("training produced a nonfinite loss")
            loss.backward()
            optimizer.step()
        train_mse, train_axis = _evaluate_model(torch, model, train_x, train_y)
        validation_mse, validation_axis = _evaluate_model(
            torch, model, validation_x, validation_y
        )
        row = {
            "epoch": epoch,
            "train_action_mse": train_mse,
            "train_action_mse_by_axis": dict(zip(AXES, train_axis)),
            "validation_action_mse": validation_mse,
            "validation_action_mse_by_axis": dict(zip(AXES, validation_axis)),
        }
        history.append(row)
        if validation_mse < selected_validation:
            selected_epoch = epoch
            selected_validation = validation_mse
            selected_state = deepcopy(model.state_dict())

    if selected_state is None:
        raise RuntimeError("no finite validation checkpoint was produced")
    model.load_state_dict(selected_state)
    return model, selected_epoch, history


def _overfit_diagnostic(
    train_observations, train_actions, *, seed, learning_rate, steps, sample_count=64
):
    """Overfit one fixed train-only subset with a fresh model; never touches validation/dev."""
    if len(train_observations) < sample_count:
        raise ValueError(f"train split needs at least {sample_count} rows for overfit diagnostic")
    torch = _require_torch()
    sample_rng = torch.Generator(device="cpu")
    sample_rng.manual_seed(seed + 10)
    index = torch.randperm(len(train_observations), generator=sample_rng)[:sample_count]
    x = torch.from_numpy(np.ascontiguousarray(train_observations))[index]
    y = torch.from_numpy(np.ascontiguousarray(train_actions))[index]
    model = _build_model(torch, seed + 11)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial, _ = _evaluate_model(torch, model, x, y)
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - y).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("overfit diagnostic produced a nonfinite loss")
        loss.backward()
        optimizer.step()
    final, final_axis = _evaluate_model(torch, model, x, y)
    if not final < initial:
        raise RuntimeError("train-only overfit diagnostic did not decrease its loss")
    return {
        "source_split": "train",
        "fresh_model": True,
        "sample_count": sample_count,
        "steps": steps,
        "sample_indices_sha256": _array_sha256(index.numpy()),
        "initial_action_mse": initial,
        "final_action_mse": final,
        "final_to_initial_loss_ratio": final / initial,
        "final_action_mse_by_axis": dict(zip(AXES, final_axis)),
        "loss_decreased": True,
    }


def _model_layers(model) -> list[dict]:
    linear = [layer for layer in model if layer.__class__.__name__ == "Linear"]
    if [tuple(layer.weight.shape) for layer in linear] != [(32, 49), (32, 32), (3, 32)]:
        raise RuntimeError("trained model architecture changed before export")
    return [
        {
            "weights": layer.weight.detach().cpu().numpy().tolist(),
            "bias": layer.bias.detach().cpu().numpy().tolist(),
        }
        for layer in linear
    ]


def _split_diagnostics(model, observations, actions, episode_ids, constant_action):
    torch = _require_torch()
    with torch.no_grad():
        prediction = model(torch.from_numpy(np.ascontiguousarray(observations))).numpy()
    errors = {
        "model": (prediction - actions) ** 2,
        "zero_action": actions**2,
        "constant_train_label_mean": (constant_action[None, :] - actions) ** 2,
    }

    def metrics(rows):
        return {
            f"{name}_mse": float(value[rows].mean())
            for name, value in errors.items()
        } | {
            f"{name}_mse_by_axis": dict(zip(AXES, map(float, value[rows].mean(axis=0))))
            for name, value in errors.items()
        }

    ordered_ids = list(dict.fromkeys(episode_ids.tolist()))
    return {
        **metrics(slice(None)),
        "per_episode": [
            {"episode_id": identity, "sample_count": int(np.sum(episode_ids == identity))}
            | metrics(episode_ids == identity)
            for identity in ordered_ids
        ],
    }


def _write_history(staging: Path, history: list[dict]) -> None:
    _json(staging / "epoch_losses.json", history)
    fieldnames = [
        "epoch",
        "train_action_mse",
        *(f"train_{axis}_mse" for axis in AXES),
        "validation_action_mse",
        *(f"validation_{axis}_mse" for axis in AXES),
    ]
    with (staging / "epoch_losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_action_mse": row["train_action_mse"],
                    **{
                        f"train_{axis}_mse": row["train_action_mse_by_axis"][axis]
                        for axis in AXES
                    },
                    "validation_action_mse": row["validation_action_mse"],
                    **{
                        f"validation_{axis}_mse": row["validation_action_mse_by_axis"][axis]
                        for axis in AXES
                    },
                }
            )


def _train_bc_impl(
    dataset,
    output,
    *,
    seed,
    epochs=100,
    batch_size=256,
    learning_rate=0.001,
    overfit_steps=300,
) -> Path:
    """Train one seed and atomically publish a finite-JSON surface BC run."""
    seed, epochs, batch_size, learning_rate, overfit_steps = _validate_inputs(
        seed, epochs, batch_size, learning_rate, overfit_steps
    )
    torch = _require_torch()
    dataset = Path(dataset).absolute()
    output = _output_path(output)
    if output.exists():
        raise ValueError("BC output must be a new directory")

    source_before = _source_identity()
    runtime_before = _runtime_identity(torch)
    dataset_before = _dataset_snapshot(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("corpus_scope") != "full":
        raise ValueError("behavior cloning requires a full audited dataset")
    if manifest.get("nominal_kind") != "adaptive" or manifest.get("action_stage") != (
        "normalized_local_residual"
    ):
        raise ValueError("dataset is not the normalized adaptive friction-teacher corpus")

    train_batch = surface_transitions.load_transition_batch(dataset, "train")
    validation_batch = surface_transitions.load_transition_batch(dataset, "validation")
    train_x, train_y, train_episode, train_groups = _training_arrays(train_batch, "train")
    validation_x, validation_y, validation_episode, validation_groups = _training_arrays(
        validation_batch, "validation"
    )
    if not train_groups.isdisjoint(validation_groups):
        raise ValueError("train and validation physical/task groups overlap")
    input_hashes = {
        "train_observations": _array_sha256(train_x),
        "train_actions": _array_sha256(train_y),
        "validation_observations": _array_sha256(validation_x),
        "validation_actions": _array_sha256(validation_y),
    }

    overfit = _overfit_diagnostic(
        train_x,
        train_y,
        seed=seed,
        learning_rate=learning_rate,
        steps=overfit_steps,
    )
    model, selected_epoch, history = _fit_behavior_clone(
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    if input_hashes != {
        "train_observations": _array_sha256(train_x),
        "train_actions": _array_sha256(train_y),
        "validation_observations": _array_sha256(validation_x),
        "validation_actions": _array_sha256(validation_y),
    }:
        raise RuntimeError("in-memory training inputs changed during optimization")
    constant_action = train_y.mean(axis=0)
    diagnostics = {
        "train": _split_diagnostics(model, train_x, train_y, train_episode, constant_action),
        "validation": _split_diagnostics(
            model, validation_x, validation_y, validation_episode, constant_action
        ),
        "train_only_64_sample_overfit": overfit,
    }

    contract = surface_policy_artifact.policy_contract(
        manifest["cases"], purpose="il_friction_teacher", nominal_kind="adaptive"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-bc-", dir=output.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        candidate_file = "selected_model.json"
        surface_mlp_actor.save_mlp_candidate(
            staging / candidate_file,
            _model_layers(model),
            contract=contract,
            activation="tanh",
        )
        _write_history(staging, history)
        report = {
            "schema": "surface_behavior_clone_report_v1",
            "candidate_file": candidate_file,
            "selection": {
                "split": "validation",
                "metric": "action_mse",
                "rule": "minimum finite value; exact ties keep earlier epoch",
                "selected_epoch": selected_epoch,
                "selected_validation_action_mse": history[selected_epoch - 1][
                    "validation_action_mse"
                ],
            },
            "architecture": list(ARCHITECTURE),
            "hidden_activation": "tanh",
            "output_activation": "tanh",
            "constant_train_label_mean": constant_action.tolist(),
            "diagnostics": diagnostics,
            "data_use": {
                "optimization_split": "train",
                "selection_split": "validation",
                "development_test_loaded_for_training_or_selection": False,
                "train_sample_count": len(train_x),
                "validation_sample_count": len(validation_x),
                "train_group_count": len(train_groups),
                "validation_group_count": len(validation_groups),
                "groups_disjoint": True,
                "input_array_sha256": input_hashes,
            },
        }
        _json(staging / "report.json", report)

        if (
            source_before != _source_identity()
            or runtime_before != _runtime_identity(torch)
            or dataset_before != _dataset_snapshot(dataset)
        ):
            raise RuntimeError("training input, trainer code, or runtime changed during the run")
        files = sorted(path for path in staging.iterdir() if path.is_file())
        run_manifest = {
            "schema": "surface_behavior_clone_run_v1",
            "dataset": {
                "logical_name": dataset.name,
                **dataset_before,
                "source_and_assets_sha256": manifest["source_and_assets_sha256"],
            },
            "trainer_source_sha256": source_before,
            "runtime": runtime_before,
            "hyperparameters": {
                "architecture": list(ARCHITECTURE),
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "overfit_steps": overfit_steps,
                "dtype": "float64",
                "device": "cpu",
                "seeds": {
                    "model_initialization": seed,
                    "batch_shuffle": seed + 1,
                    "overfit_subset": seed + 10,
                    "overfit_model_initialization": seed + 11,
                },
            },
            "candidate_file": candidate_file,
            "selected_epoch": selected_epoch,
            "artifact_sha256": {path.name: _sha256(path) for path in files},
            "development_test_used": False,
        }
        _json(staging / "manifest.json", run_manifest)
        (staging / "COMPLETE").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="utf-8"
        )
        if output.exists():
            raise ValueError("BC output appeared during training")
        os.rename(staging, output)
    return output


def train_bc(
    dataset,
    output,
    *,
    seed,
    epochs=100,
    batch_size=256,
    learning_rate=0.001,
    overfit_steps=300,
) -> Path:
    """Train one seed using one deterministic CPU thread, restoring Torch state afterward."""
    torch = _require_torch()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        return _train_bc_impl(
            dataset,
            output,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            overfit_steps=overfit_steps,
        )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--overfit-steps", type=int, default=300)
    args = parser.parse_args(argv)
    result = train_bc(
        args.dataset,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        overfit_steps=args.overfit_steps,
    )
    print(
        json.dumps(
            {"output": str(result), "candidate": str(result / "selected_model.json")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
