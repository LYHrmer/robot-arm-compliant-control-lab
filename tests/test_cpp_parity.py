import csv
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.franka_control import (
    FrankaAdmittanceController,
    FrankaHybridController,
    FrankaImpedanceController,
    FrankaState,
    FrankaTarget,
    orientation_error,
)


def _probe_path() -> Path:
    configured = os.environ.get("COMPLIANT_CONTROL_CPP_PROBE")
    if configured:
        return Path(configured)
    return Path(__file__).parents[1] / "build" / "compliant_control_probe"


def _reference_inputs() -> tuple[FrankaState, FrankaTarget]:
    angle = 0.1
    target_rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    state = FrankaState(
        position=np.array([0.36, -0.01, 0.45]),
        rotation=np.eye(3),
        linear_velocity=np.array([0.02, -0.01, 0.0]),
        angular_velocity=np.array([0.01, -0.02, 0.03]),
        normal_force=6.0,
    )
    target = FrankaTarget(
        position=np.array([0.38, 0.04, 0.42]),
        rotation=target_rotation,
        linear_velocity=np.array([0.01, -0.02, 0.03]),
        angular_velocity=np.array([-0.02, 0.01, 0.0]),
        normal_force=12.0,
    )
    return state, target


def test_cpp_controllers_match_python_reference():
    probe = _probe_path()
    if not probe.is_file():
        pytest.skip("C++ probe is not built; run cmake -S . -B build && cmake --build build")

    completed = subprocess.run(
        [str(probe)],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = {
        row[0]: np.asarray(row[1:], dtype=float)
        for row in csv.reader(completed.stdout.splitlines())
    }

    state, target = _reference_inputs()
    expected = {"orientation_error": orientation_error(state.rotation, target.rotation)}
    controllers = {
        "impedance": FrankaImpedanceController(),
        "admittance": FrankaAdmittanceController(),
        "hybrid": FrankaHybridController(),
    }
    for name, controller in controllers.items():
        controller.reset(state)
        expected[name] = controller.compute(state, target, 0.01)

    approach_state = replace(state, normal_force=0.0)
    approach_hybrid = FrankaHybridController()
    approach_hybrid.reset(approach_state)
    expected["hybrid_approach"] = approach_hybrid.compute(approach_state, target, 0.01)

    transition_hybrid = FrankaHybridController()
    transition_hybrid.reset(approach_state)
    for _ in range(12):
        transition_wrench = transition_hybrid.compute(state, target, 0.002)
    expected["hybrid_transition"] = transition_wrench

    assert rows.keys() == expected.keys()
    for name, values in expected.items():
        np.testing.assert_allclose(rows[name], values, rtol=1e-12, atol=1e-12)
