"""Controlled BC input ablations built on the frozen surface BC fitter."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

from compliant_control_lab import surface_policy_artifact, surface_transitions
from compliant_control_lab.surface_env import OBSERVATION_NAMES
from tools import surface_mlp_actor
from tools import train_surface_bc as frozen_bc

ARMS = {
    "drop_previous_residual": tuple(index for index in range(49) if index not in (14, 15, 16)),
    "teacher_inputs": (0, 10, 11),
}


def _input_mask(arm: str) -> np.ndarray:
    if arm not in ARMS:
        raise ValueError(f"arm must be exactly one of {tuple(ARMS)}")
    mask = np.zeros(49, dtype=bool)
    mask[list(ARMS[arm])] = True
    return mask


def _mask_observations(observations, mask: np.ndarray) -> np.ndarray:
    values = np.array(observations, dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[1:] != (49,):
        raise ValueError("input mask expects an N-by-49 observation array")
    values[:, ~mask] = 0.0
    return values


def _fold_input_mask(layers, mask: np.ndarray) -> list[dict]:
    folded = deepcopy(layers)
    first = np.asarray(folded[0]["weights"], dtype=np.float64)
    if first.shape != (32, 49):
        raise ValueError("input mask requires the frozen 49-32-32-3 architecture")
    first[:, ~mask] = 0.0
    folded[0]["weights"] = first.tolist()
    return folded


def _source_identity() -> dict[str, str]:
    return {
        "transfer_trainer": frozen_bc._sha256(Path(__file__).resolve()),
        **frozen_bc._source_identity(),
    }


def _numpy_predictions(candidate, contract, observations) -> np.ndarray:
    artifact = surface_policy_artifact.load_policy_artifact(
        candidate, expected_contract=contract
    )
    actor, _ = surface_mlp_actor.actor_from_artifact(artifact)
    return np.stack([actor(observation) for observation in observations])


def _export_checks(candidate, contract, model, raw_observations, masked_observations, mask):
    torch = frozen_bc._require_torch()
    with torch.no_grad():
        expected = model(torch.from_numpy(np.ascontiguousarray(masked_observations))).numpy()
    observed = _numpy_predictions(candidate, contract, raw_observations)
    parity = float(np.max(np.abs(observed - expected), initial=0.0))

    noisy = np.array(raw_observations, copy=True)
    inactive = np.flatnonzero(~mask)
    if len(inactive):
        noisy[:, inactive] = np.where(inactive % 2, -3.0, 3.0)
    perturbed = _numpy_predictions(candidate, contract, noisy)
    invariance = float(np.max(np.abs(perturbed - observed), initial=0.0))
    if parity > 2e-14 or invariance > 2e-14:
        raise RuntimeError("folded NumPy export differs from masked Torch policy")
    return {
        "sample_count": len(raw_observations),
        "torch_masked_vs_numpy_raw_max_abs": parity,
        "inactive_extremes_vs_raw_numpy_max_abs": invariance,
        "absolute_tolerance": 2e-14,
        "passed": True,
    }


def _publish_failure(staging, output, error, *, arm, seed, dataset_snapshot, sources, runtime):
    partial = {
        path.name: frozen_bc._sha256(path)
        for path in sorted(staging.iterdir())
        if path.is_file()
    }
    failure = {
        "schema": "surface_behavior_clone_transfer_failure_v1",
        "arm": arm,
        "seed": seed,
        "error_type": type(error).__name__,
        "error": str(error),
        "dataset": dataset_snapshot,
        "trainer_source_sha256": sources,
        "runtime": runtime,
        "partial_artifact_sha256": partial,
    }
    frozen_bc._json(staging / "failure.json", failure)
    (staging / "FAILED").write_text(
        frozen_bc._sha256(staging / "failure.json") + "\n", encoding="utf-8"
    )
    if output.exists():
        raise FileExistsError("BC transfer destination appeared during failed training")
    os.rename(staging, output)


def _train_bc_transfer_impl(
    dataset,
    output,
    *,
    arm,
    seed,
    epochs,
    batch_size,
    learning_rate,
    overfit_steps,
) -> Path:
    seed, epochs, batch_size, learning_rate, overfit_steps = frozen_bc._validate_inputs(
        seed, epochs, batch_size, learning_rate, overfit_steps
    )
    mask = _input_mask(arm)
    torch = frozen_bc._require_torch()
    dataset = Path(dataset).absolute()
    output = frozen_bc._output_path(output)
    if output.exists():
        raise ValueError("BC transfer output must be a new directory")

    sources = _source_identity()
    runtime = frozen_bc._runtime_identity(torch)
    dataset_snapshot = frozen_bc._dataset_snapshot(dataset)
    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if dataset_manifest.get("corpus_scope") != "full":
        raise ValueError("BC transfer requires a full audited dataset")
    if (
        dataset_manifest.get("nominal_kind") != "adaptive"
        or dataset_manifest.get("action_stage") != "normalized_local_residual"
    ):
        raise ValueError("dataset is not the normalized adaptive friction-teacher corpus")

    train_batch = surface_transitions.load_transition_batch(dataset, "train")
    validation_batch = surface_transitions.load_transition_batch(dataset, "validation")
    train_raw, train_y, train_episode, train_groups = frozen_bc._training_arrays(
        train_batch, "train"
    )
    validation_raw, validation_y, validation_episode, validation_groups = (
        frozen_bc._training_arrays(validation_batch, "validation")
    )
    if not train_groups.isdisjoint(validation_groups):
        raise ValueError("train and validation physical/task groups overlap")
    train_x = _mask_observations(train_raw, mask)
    validation_x = _mask_observations(validation_raw, mask)
    input_hashes = {
        "train_raw_observations": frozen_bc._array_sha256(train_raw),
        "train_masked_observations": frozen_bc._array_sha256(train_x),
        "train_actions": frozen_bc._array_sha256(train_y),
        "validation_raw_observations": frozen_bc._array_sha256(validation_raw),
        "validation_masked_observations": frozen_bc._array_sha256(validation_x),
        "validation_actions": frozen_bc._array_sha256(validation_y),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-bc-transfer-", dir=output.parent) as temp:
        staging = Path(temp) / "run"
        staging.mkdir()
        try:
            overfit = frozen_bc._overfit_diagnostic(
                train_x,
                train_y,
                seed=seed,
                learning_rate=learning_rate,
                steps=overfit_steps,
            )
            model, selected_epoch, history = frozen_bc._fit_behavior_clone(
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
                "train_raw_observations": frozen_bc._array_sha256(train_raw),
                "train_masked_observations": frozen_bc._array_sha256(train_x),
                "train_actions": frozen_bc._array_sha256(train_y),
                "validation_raw_observations": frozen_bc._array_sha256(validation_raw),
                "validation_masked_observations": frozen_bc._array_sha256(validation_x),
                "validation_actions": frozen_bc._array_sha256(validation_y),
            }:
                raise RuntimeError("in-memory transfer inputs changed during optimization")

            contract = surface_policy_artifact.policy_contract(
                dataset_manifest["cases"],
                purpose="il_friction_teacher",
                nominal_kind="adaptive",
            )
            candidate_file = "selected_model.json"
            layers = _fold_input_mask(frozen_bc._model_layers(model), mask)
            surface_mlp_actor.save_mlp_candidate(
                staging / candidate_file, layers, contract=contract, activation="tanh"
            )
            raw_all = np.concatenate((train_raw, validation_raw))
            masked_all = np.concatenate((train_x, validation_x))
            export_checks = _export_checks(
                staging / candidate_file, contract, model, raw_all, masked_all, mask
            )
            frozen_bc._write_history(staging, history)
            constant_action = train_y.mean(axis=0)
            report = {
                "schema": "surface_behavior_clone_transfer_report_v1",
                "arm": arm,
                "input_mask": mask.astype(int).tolist(),
                "active_indices": np.flatnonzero(mask).tolist(),
                "active_names": [OBSERVATION_NAMES[index] for index in np.flatnonzero(mask)],
                "inactive_indices": np.flatnonzero(~mask).tolist(),
                "inactive_names": [OBSERVATION_NAMES[index] for index in np.flatnonzero(~mask)],
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
                "architecture": list(frozen_bc.ARCHITECTURE),
                "constant_train_label_mean": constant_action.tolist(),
                "diagnostics": {
                    "train": frozen_bc._split_diagnostics(
                        model, train_x, train_y, train_episode, constant_action
                    ),
                    "validation": frozen_bc._split_diagnostics(
                        model,
                        validation_x,
                        validation_y,
                        validation_episode,
                        constant_action,
                    ),
                    "train_only_64_sample_overfit": overfit,
                    "folded_export": export_checks,
                },
                "data_use": {
                    "optimization_split": "train",
                    "selection_split": "validation",
                    "development_test_loaded_for_training_or_selection": False,
                    "train_sample_count": len(train_x),
                    "validation_sample_count": len(validation_x),
                    "train_group_count": len(train_groups),
                    "validation_group_count": len(validation_groups),
                    "groups_disjoint": True,
                    "teacher_labels_altered": False,
                    "input_array_sha256": input_hashes,
                },
            }
            frozen_bc._json(staging / "report.json", report)

            if (
                sources != _source_identity()
                or runtime != frozen_bc._runtime_identity(torch)
                or dataset_snapshot != frozen_bc._dataset_snapshot(dataset)
            ):
                raise RuntimeError("training input, trainer code, or runtime changed during run")
            artifacts = {
                path.name: frozen_bc._sha256(path)
                for path in sorted(staging.iterdir())
                if path.is_file()
            }
            manifest = {
                "schema": "surface_behavior_clone_transfer_run_v1",
                "arm": arm,
                "input_mask": mask.astype(int).tolist(),
                "dataset": {
                    "logical_name": dataset.name,
                    **dataset_snapshot,
                    "source_and_assets_sha256": dataset_manifest[
                        "source_and_assets_sha256"
                    ],
                },
                "trainer_source_sha256": sources,
                "runtime": runtime,
                "hyperparameters": {
                    "seed": seed,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "overfit_steps": overfit_steps,
                    "architecture": list(frozen_bc.ARCHITECTURE),
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
                "artifact_sha256": artifacts,
                "development_test_used": False,
            }
            frozen_bc._json(staging / "manifest.json", manifest)
            (staging / "COMPLETE").write_text(
                frozen_bc._sha256(staging / "manifest.json") + "\n", encoding="utf-8"
            )
            if output.exists():
                raise FileExistsError("BC transfer destination appeared during training")
            os.rename(staging, output)
        except Exception as error:
            if staging.exists():
                _publish_failure(
                    staging,
                    output,
                    error,
                    arm=arm,
                    seed=seed,
                    dataset_snapshot=dataset_snapshot,
                    sources=sources,
                    runtime=runtime,
                )
            raise RuntimeError(
                f"BC transfer failure; partial record published at {output}"
            ) from error
    return output


def train_bc_transfer(
    dataset,
    output,
    *,
    arm,
    seed,
    epochs=100,
    batch_size=256,
    learning_rate=0.001,
    overfit_steps=300,
) -> Path:
    """Train one fixed input-ablation arm and publish a 49-input masked candidate."""
    torch = frozen_bc._require_torch()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        return _train_bc_transfer_impl(
            dataset,
            output,
            arm=arm,
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
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--overfit-steps", type=int, default=300)
    args = parser.parse_args(argv)
    output = train_bc_transfer(
        args.dataset,
        args.output,
        arm=args.arm,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        overfit_steps=args.overfit_steps,
    )
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
