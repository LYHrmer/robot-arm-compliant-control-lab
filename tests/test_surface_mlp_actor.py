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
from tools.surface_mlp_actor import actor_from_artifact, save_mlp_candidate


def _contract():
    cases = [
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
    cases = split_cases(cases, seed=20260906, train_fraction=0.7, validation_fraction=0.2)
    return policy_contract(cases, purpose="il_friction_teacher", nominal_kind="adaptive")


def _layers(hidden=2):
    first = np.zeros((hidden, 49))
    first[0, 0] = 1.0
    if hidden > 1:
        first[1, 1] = -2.0
    return [
        {"weights": first.tolist(), "bias": np.linspace(0.1, 0.2, hidden).tolist()},
        {
            "weights": np.resize(np.array([[1.0, 2.0], [-1.0, 0.5], [0.2, -0.3]]), (3, hidden)).tolist(),
            "bias": [0.3, -0.2, 0.1],
        },
    ]


def test_nonzero_relu_forward_matches_explicit_dense_math(tmp_path):
    contract = _contract()
    layers = _layers()
    path = tmp_path / "mlp.json"
    save_mlp_candidate(path, layers, contract=contract, activation="relu")
    artifact = load_policy_artifact(path, expected_contract=contract)
    actor, descriptor = actor_from_artifact(artifact)
    observation = np.zeros(49)
    observation[:2] = [0.4, -0.25]
    hidden = np.maximum(np.asarray(layers[0]["weights"]) @ observation + layers[0]["bias"], 0)
    expected = np.tanh(np.asarray(layers[1]["weights"]) @ hidden + layers[1]["bias"])
    np.testing.assert_allclose(actor(observation), expected, rtol=0, atol=1e-15)
    assert descriptor["format"] == "surface_numpy_mlp_v1"
    identity = descriptor["runner_identity"]
    assert set(identity) == {
        "schema",
        "runner_sha256",
        "evaluator_sha256",
        "package_source_and_assets_sha256",
        "python_version",
        "numpy_version",
        "mujoco_version",
        "gymnasium_version",
    }


def test_three_hidden_tanh_layers_are_supported(tmp_path):
    widths = (3, 4, 2)
    layers, incoming = [], 49
    for width in widths:
        layers.append(
            {
                "weights": np.full((width, incoming), 0.01).tolist(),
                "bias": np.linspace(-0.1, 0.1, width).tolist(),
            }
        )
        incoming = width
    layers.append({"weights": np.full((3, incoming), 0.2).tolist(), "bias": [0, 0.1, -0.1]})
    contract = _contract()
    path = tmp_path / "deep.json"
    save_mlp_candidate(path, layers, contract=contract, activation="tanh")
    actor, _ = actor_from_artifact(load_policy_artifact(path, expected_contract=contract))
    output = actor(np.linspace(-1, 1, 49))
    assert output.shape == (3,) and np.all(np.isfinite(output))
    assert np.all(np.abs(output) <= 1)


@pytest.mark.parametrize(
    ("layers", "activation"),
    [
        ([{"weights": np.zeros((3, 49)).tolist(), "bias": [0, 0, 0]}], "tanh"),
        (_layers(129), "tanh"),
        (_layers() + [_layers()[-1]] * 3, "tanh"),
        ([{"weights": np.zeros((2, 48)).tolist(), "bias": [0, 0]}, _layers()[-1]], "tanh"),
        ([*_layers()[:-1], {"weights": np.zeros((4, 2)).tolist(), "bias": [0] * 4}], "tanh"),
        ([{"weights": [[False] * 49] * 2, "bias": [0, 0]}, _layers()[-1]], "tanh"),
        ([{"weights": np.zeros((2, 49)).tolist(), "bias": [0, np.nan]}, _layers()[-1]], "tanh"),
        ([{"weights": np.zeros((2, 49)).tolist(), "bias": [0, 0], "module": "evil"}, _layers()[-1]], "tanh"),
        (_layers(), "sigmoid"),
    ],
)
def test_invalid_architecture_parameters_and_unknown_fields_are_rejected(
    tmp_path, layers, activation
):
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        save_mlp_candidate(path, layers, contract=_contract(), activation=activation)
    assert not path.exists()
