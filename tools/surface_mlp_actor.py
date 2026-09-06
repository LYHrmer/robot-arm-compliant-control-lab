"""Restricted finite-JSON NumPy MLP export and inference for surface candidates."""

import hashlib
import platform
from importlib.metadata import version
from numbers import Real
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_experiment import _source_hashes
from compliant_control_lab.surface_policy_artifact import save_policy_artifact

FORMAT = "surface_numpy_mlp_v1"
RUNNER_SCHEMA = "surface_numpy_mlp_runner_identity_v1"
MAX_HIDDEN_WIDTH = 128
MAX_ABS_PARAMETER = 1_000_000.0


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_runner_identity():
    """Identity pinned into candidate bytes and checked before environment reset."""
    runner = Path(__file__).resolve()
    evaluator = runner.with_name("evaluate_surface_candidate.py")
    return {
        "schema": RUNNER_SCHEMA,
        "runner_sha256": _sha256(runner),
        "evaluator_sha256": _sha256(evaluator),
        "package_source_and_assets_sha256": _source_hashes(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "mujoco_version": version("mujoco"),
        "gymnasium_version": version("gymnasium"),
    }


def _dense_layers(layers):
    if not isinstance(layers, (list, tuple)) or not 2 <= len(layers) <= 4:
        raise ValueError("MLP requires one to three hidden layers plus one output layer")
    checked, expected_input = [], 49
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != {"weights", "bias"}:
            raise ValueError("each dense layer must contain only weights and bias")
        try:
            raw_weights = np.asarray(layer["weights"], dtype=object)
            raw_bias = np.asarray(layer["bias"], dtype=object)
        except (TypeError, ValueError) as error:
            raise ValueError("dense weights and bias must be rectangular numeric arrays") from error
        raw_values = (*raw_weights.flat, *raw_bias.flat)
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in raw_values
        ):
            raise ValueError("dense weights and bias must contain real numbers, not booleans")
        weights, bias = raw_weights.astype(np.float64), raw_bias.astype(np.float64)
        output_width = 3 if index == len(layers) - 1 else len(bias)
        if index < len(layers) - 1 and not 1 <= output_width <= MAX_HIDDEN_WIDTH:
            raise ValueError("hidden widths must be between 1 and 128")
        if weights.shape != (output_width, expected_input) or bias.shape != (output_width,):
            raise ValueError("dense layer shape does not match the 49-hidden-3 architecture")
        if (
            not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(bias))
            or np.max(np.abs(weights), initial=0) > MAX_ABS_PARAMETER
            or np.max(np.abs(bias), initial=0) > MAX_ABS_PARAMETER
        ):
            raise ValueError("MLP parameters must be finite and bounded")
        checked.append((weights.copy(), bias.copy()))
        expected_input = output_width
    return checked


def mlp_payload(layers, *, activation="tanh"):
    if activation not in {"tanh", "relu"}:
        raise ValueError("hidden activation must be exactly tanh or relu")
    checked = _dense_layers(layers)
    return {
        "kind": FORMAT,
        "activation": activation,
        "output_activation": "tanh",
        "layers": [
            {"weights": weights.tolist(), "bias": bias.tolist()}
            for weights, bias in checked
        ],
        "runner_identity": current_runner_identity(),
    }


def save_mlp_candidate(path, layers, *, contract, activation="tanh"):
    """Export trainer-independent out-by-in dense weights as finite JSON."""
    return save_policy_artifact(
        path, mlp_payload(layers, activation=activation), contract=contract
    )


def actor_from_artifact(artifact):
    """Return a restricted actor and evidence descriptor; never import candidate code."""
    payload = artifact.payload
    if payload.get("kind") == "linear_tanh":
        return None, {"format": "linear_tanh", "runner_identity": None}
    if not isinstance(payload, dict) or set(payload) != {
        "kind",
        "activation",
        "output_activation",
        "layers",
        "runner_identity",
    }:
        raise ValueError("unsupported or malformed candidate payload")
    if payload["kind"] != FORMAT or payload["output_activation"] != "tanh":
        raise ValueError("unsupported MLP format or output activation")
    if payload["activation"] not in {"tanh", "relu"}:
        raise ValueError("hidden activation must be exactly tanh or relu")
    if payload["runner_identity"] != current_runner_identity():
        raise ValueError("candidate runner identity differs from this evaluator installation")
    layers = _dense_layers(payload["layers"])
    activation = payload["activation"]

    def actor(observation):
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (49,) or not np.all(np.isfinite(values)):
            raise ValueError("MLP expects one finite 49-vector")
        with np.errstate(over="raise", invalid="raise"):
            for weights, bias in layers[:-1]:
                values = weights @ values + bias
                values = np.tanh(values) if activation == "tanh" else np.maximum(values, 0)
            weights, bias = layers[-1]
            return np.tanh(weights @ values + bias)

    return actor, {"format": FORMAT, "runner_identity": payload["runner_identity"]}
