"""Solved contact-force caching, filter input and delayed sensor contracts."""

import mujoco
import numpy as np
import pytest

import compliant_control_lab.franka_simulation as simulation


@pytest.mark.parametrize(
    ("reset_force", "expected_feedback", "expected_measurement"),
    [
        (0.0, [0, 5, 12.5, 21.25, 30.625], [2, 2, 2, 7, 14.5]),
        (4.0, [2, 6, 13, 21.5, 30.75], [4, 4, 4, 8, 15]),
    ],
)
def test_split_step_caches_solved_force_before_contact_rebuild(
    monkeypatch, reset_force, expected_feedback, expected_measurement
):
    class RecordingController:
        name = "force_cache_probe"

        def __init__(self):
            self.forces = []

        def reset(self, state):
            pass

        def compute(self, state, target, dt):
            self.forces.append(state.normal_force)
            return np.zeros(6)

    original_forward = mujoco.mj_forward
    original_step1 = mujoco.mj_step1
    original_step2 = mujoco.mj_step2
    phase = "uninitialized"
    read_phases = []
    samples = iter([reset_force, 10.0, 20.0, 30.0, 40.0, 50.0])

    def reset_forward(model, data):
        nonlocal phase
        original_forward(model, data)
        phase = "reset_solved"

    def step1(model, data):
        nonlocal phase
        original_step1(model, data)
        phase = "unsolved"

    def step2(model, data):
        nonlocal phase
        original_step2(model, data)
        phase = "solved"

    def solved_force(model, data, tool_id, wall_id):
        assert phase in {"reset_solved", "solved"}, "force read from unsolved contact map"
        read_phases.append(phase)
        return next(samples)

    monkeypatch.setattr(mujoco, "mj_forward", reset_forward)
    monkeypatch.setattr(mujoco, "mj_step1", step1)
    monkeypatch.setattr(mujoco, "mj_step2", step2)
    monkeypatch.setattr(simulation, "_normal_contact_force", solved_force)
    controller = RecordingController()
    result = simulation.run_franka_trial(
        controller,
        scenario=simulation.FrankaScenario(
            name="force_cache_probe",
            delay_steps=2,
            force_bias_n=2.0,
            force_noise_std=0.0,
        ),
        config=simulation.FrankaSimulationConfig(
            duration=0.01,
            timestep=0.002,
            force_filter_time_constant=0.002,
            control_timing="split_step",
        ),
    )

    assert read_phases == ["reset_solved"] + ["solved"] * 5
    np.testing.assert_array_equal(result.raw_normal_force, [10, 20, 30, 40, 50])
    np.testing.assert_array_equal(result.normal_force, expected_feedback)
    np.testing.assert_array_equal(result.measured_normal_force, expected_measurement)
    np.testing.assert_array_equal(controller.forces, result.measured_normal_force)
    np.testing.assert_array_equal(result.raw_force_sample_time, result.time)
    np.testing.assert_array_equal(
        result.feedback_force_sample_time, [0, 0, 0.002, 0.004, 0.006]
    )
    np.testing.assert_array_equal(
        result.measured_kinematic_sample_time, [0, 0, 0, 0.002, 0.004]
    )
    np.testing.assert_array_equal(result.measured_force_sample_time, [0, 0, 0, 0, 0.002])
