"""Check the hand-worked tutorial answers independently of its implementation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.tutorials.wrench_to_torque import JACOBIAN, compute_example


def test_default_wrench_and_task_torque_match_hand_calculation():
    wrench, torque = compute_example()
    np.testing.assert_allclose(wrench, [2.6, -3.7, 2.0, 0.2, -0.6, 0.2], atol=1e-12)
    np.testing.assert_allclose(torque, [-0.17, -0.28, -0.91, 0.34, 0.14, 0.40, 0.46], atol=1e-12)


def test_twenty_mm_variant_changes_only_x_force_and_its_joint_contributions():
    wrench, torque = compute_example(20.0)
    np.testing.assert_allclose(wrench, [5.6, -3.7, 2.0, 0.2, -0.6, 0.2], atol=1e-12)
    np.testing.assert_allclose(torque, [-0.17, 0.32, -0.91, 0.04, 0.14, 0.40, 0.76], atol=1e-12)
    base_wrench, base_torque = compute_example()
    np.testing.assert_allclose(wrench - base_wrench, [3.0, 0, 0, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(torque - base_torque, 3.0 * JACOBIAN[0], atol=1e-12)


def test_transpose_mapping_preserves_instantaneous_power():
    wrench, torque = compute_example()
    joint_velocity = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.4])
    cartesian_velocity = JACOBIAN @ joint_velocity
    assert float(torque @ joint_velocity) == pytest.approx(float(wrench @ cartesian_velocity))


@pytest.mark.parametrize("x_error_mm", [0.0, float("nan"), float("inf")])
def test_lab_accepts_only_the_two_documented_inputs(x_error_mm):
    with pytest.raises(ValueError, match="10 or 20"):
        compute_example(x_error_mm)


@pytest.mark.parametrize(
    ("args", "expected_torque"),
    [
        ([], "[-0.170, -0.280, -0.910, 0.340, 0.140, 0.400, 0.460]"),
        (["--x-error-mm", "20"], "[-0.170, 0.320, -0.910, 0.040, 0.140, 0.400, 0.760]"),
    ],
)
def test_cli_prints_answer_without_creating_output_files(tmp_path, args, expected_torque):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    inherited_paths = [
        str(Path(entry).resolve())
        for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry
    ]
    # Keep explicit caller paths, but never inject a legacy dependency directory.
    env["PYTHONPATH"] = os.pathsep.join(
        (str(root), str(root / "src"), *inherited_paths)
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "tools.tutorials.wrench_to_torque", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert expected_torque in result.stdout
    assert "not a hardware command" in result.stdout
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []
