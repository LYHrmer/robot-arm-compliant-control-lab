import csv
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.franka_control import (
    FrankaActuationContext,
    FrankaAdmittanceController,
    FrankaHybridController,
    FrankaImpedanceController,
    FrankaState,
    FrankaTarget,
    orientation_error,
)
from compliant_control_lab.franka_torque_safety import (
    project_residual_force,
    project_wrench_to_torque_limits,
    residual_torque_headroom,
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


def _fixed_random_torque_cases() -> list[dict[str, np.ndarray | float]]:
    rng = np.random.default_rng(20260904)
    cases: list[dict[str, np.ndarray | float]] = []
    for case_index in range(160):
        jacobian = rng.normal(0.0, 0.45, size=(6, 7))
        lower = -rng.uniform(25.0, 90.0, size=7)
        upper = rng.uniform(20.0, 80.0, size=7)
        offset = rng.uniform(-2.0, 2.0, size=7)
        nominal = np.zeros(6)
        if case_index % 4 == 3:
            nominal = rng.normal(0.0, 80.0, size=6)
        additive_scale = (0.0, 6.0, 120.0, 15.0)[case_index % 4]
        additive = rng.normal(0.0, additive_scale, size=6)
        residual = rng.normal(0.0, max(4.0, additive_scale), size=3)
        cases.append(
            {
                "jacobian": jacobian,
                "offset": offset,
                "lower": lower,
                "upper": upper,
                "nominal": nominal,
                "additive": additive,
                "residual": residual,
                "action_bounds": rng.uniform(2.0, 25.0, size=3),
                "reserve": float(rng.uniform(0.0, 0.25)),
            }
        )
    return cases


def _encode_torque_case(case_index: int, case: dict[str, np.ndarray | float]) -> str:
    values: list[float | int] = [case_index, float(case["reserve"])]
    for name in (
        "jacobian",
        "offset",
        "lower",
        "upper",
        "nominal",
        "additive",
        "residual",
        "action_bounds",
    ):
        values.extend(np.asarray(case[name], dtype=float).reshape(-1))
    return " ".join(format(value, ".17g") for value in values)


def test_cpp_torque_safety_matches_python_for_160_fixed_random_cases() -> None:
    probe = _probe_path()
    if not probe.is_file():
        pytest.skip("C++ probe is not built; run cmake -S . -B build && cmake --build build")

    cases = _fixed_random_torque_cases()
    completed = subprocess.run(
        [str(probe), "--torque-safety"],
        input="\n".join(
            _encode_torque_case(case_index, case)
            for case_index, case in enumerate(cases)
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.reader(completed.stdout.splitlines()))
    assert len(rows) == len(cases)

    observed_statuses: set[str] = set()
    for case_index, (case, row) in enumerate(zip(cases, rows, strict=True)):
        assert row[0] == "torque_case"
        assert int(row[1]) == case_index
        context = FrankaActuationContext(
            cartesian_jacobian=np.asarray(case["jacobian"]),
            joint_torque_offset=np.asarray(case["offset"]),
            lower_torque_limit=np.asarray(case["lower"]),
            upper_torque_limit=np.asarray(case["upper"]),
        )
        nominal = np.asarray(case["nominal"])
        reserve = float(case["reserve"])
        full = project_wrench_to_torque_limits(
            context,
            nominal,
            np.asarray(case["additive"]),
            reserve,
        )
        residual = project_residual_force(
            context,
            nominal,
            np.asarray(case["residual"]),
            reserve,
        )
        headroom = residual_torque_headroom(
            context,
            nominal,
            np.asarray(case["action_bounds"]),
            reserve,
        )

        assert row[2] == full.status
        np.testing.assert_allclose(float(row[3]), full.scale, rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(
            np.asarray(row[4:10], dtype=float),
            full.additive_wrench,
            rtol=2e-13,
            atol=2e-13,
        )
        assert row[10] == residual.status
        np.testing.assert_allclose(float(row[11]), residual.scale, rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(
            np.asarray(row[12:18], dtype=float),
            residual.additive_wrench,
            rtol=2e-13,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            np.asarray(row[18:24], dtype=float),
            headroom,
            rtol=2e-13,
            atol=2e-13,
        )
        observed_statuses.update((full.status, residual.status))

    assert {"unchanged", "scaled", "nominal_outside"} <= observed_statuses
