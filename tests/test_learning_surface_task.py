from dataclasses import asdict

import numpy as np
import pytest

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_simulation import SurfaceTask, yaw_frame


def test_default_task_reproduces_original_targets():
    position, rotation = np.array([0.32, 0.03, 0.55]), np.eye(3)
    old, new = SurfaceTask(yaw_deg=15), LearningSurfaceTask(yaw_deg=15)
    for time in (0, 0.1, 0.4, 1, 1.2, 1.20001, 1.5, 1.7, 3, 8, 12):
        reference, candidate = (task.target_at(time, position, rotation, 12) for task in (old, new))
        for key, values in asdict(reference).items():
            np.testing.assert_allclose(asdict(candidate)[key], values, rtol=0, atol=1e-12)


@pytest.mark.parametrize("yaw,phase,direction", [(0, 0.7, 1), (15, 2.1, -1), (-15, 5.0, 1)])
def test_analytic_velocity_and_no_normal_motion_after_contact(yaw, phase, direction):
    task = LearningSurfaceTask(
        yaw_deg=yaw,
        phase_rad=phase,
        direction=direction,
        frequency_hz=0.23,
        tangent_amplitude_m=0.06,
    )
    position, rotation = np.array([0.32, 0.03, 0.55]), np.eye(3)
    normal = yaw_frame(yaw).rotation[:, 0]
    anchor = task.target_at(1.2, position, rotation, 12).position
    epsilon = 1e-6
    for time in (0.4, 1.2, 1.2001, 1.5, 1.7, 3, 11.8):
        target = task.target_at(time, position, rotation, 12)
        before = task.target_at(time - epsilon, position, rotation, 12)
        after = task.target_at(time + epsilon, position, rotation, 12)
        np.testing.assert_allclose(
            target.linear_velocity,
            (after.position - before.position) / (2 * epsilon),
            rtol=0,
            atol=3e-7,
        )
        if time >= 1.2:
            assert abs(normal @ (target.position - anchor)) < 1e-12
            assert abs(normal @ target.linear_velocity) < 1e-12


@pytest.mark.parametrize(
    "changes",
    [
        {"tangent_amplitude_m": 0},
        {"vertical_amplitude_m": -0.1},
        {"frequency_hz": 0},
        {"phase_rad": np.inf},
        {"frequency_hz": np.nan},
        {"frequency_hz": True},
        {"direction": True},
        {"direction": 1.0},
        {"direction": 0},
        {"direction": 2},
    ],
)
def test_bad_parameters_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        LearningSurfaceTask(**changes)


def test_inputs_are_owned_and_approach_does_not_depend_on_ellipse_parameters():
    position, rotation = np.array([0.32, 0.03, 0.55]), np.eye(3)
    copy = position.copy()
    task = LearningSurfaceTask(phase_rad=2.4, direction=np.int64(-1), frequency_hz=0.13)
    for time in (0, 0.6, 1.2):
        before = SurfaceTask().target_at(time, position, rotation, 12)
        after = task.target_at(time, position, rotation, 12)
        np.testing.assert_array_equal(after.position, before.position)
        np.testing.assert_array_equal(after.linear_velocity, before.linear_velocity)
    target = task.target_at(3, position, rotation, 12)
    target.position[:] = 99
    target.rotation[:] = 99
    np.testing.assert_array_equal(position, copy)
    np.testing.assert_array_equal(rotation, np.eye(3))
    with pytest.raises(ValueError, match="time"):
        task.target_at(np.nan, position, rotation, 12)
