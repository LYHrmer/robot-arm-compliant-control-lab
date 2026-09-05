"""Physical sampling and task contracts for the opt-in surface runner."""

from dataclasses import replace

import numpy as np
import pytest

from compliant_control_lab import surface_simulation as sim
from compliant_control_lab.surface_control import SurfaceFrame


def short_trial(**kwargs):
    return sim.run_surface_trial(
        kwargs.pop("controller_frame", sim.yaw_frame(15)),
        config=kwargs.pop("config", sim.SurfaceSimulationConfig(duration=0.04)),
        **kwargs,
    )


@pytest.mark.parametrize("delay", [0, 3])
def test_causal_sensor_and_delayed_observation_timestamps(delay):
    result = short_trial(scenario=sim.SurfaceScenario(delay_steps=delay))
    log = result.trace
    np.testing.assert_array_equal(
        log["feedback_raw_wrench_world"][1:], log["raw_wrench_world"][:-1]
    )
    steps = np.arange(len(log["time"]))
    np.testing.assert_array_equal(log["raw_wrench_sample_time"], log["time"])
    np.testing.assert_array_equal(log["kinematic_sample_time"], log["time"])
    np.testing.assert_array_equal(
        log["measured_kinematic_sample_time"], np.maximum(0, steps - delay) * result.config.timestep
    )
    np.testing.assert_array_equal(
        log["measured_wrench_sample_time"],
        np.maximum(0, steps - delay - 1) * result.config.timestep,
    )
    filtered = np.zeros(6)
    for index, sample in enumerate(log["feedback_raw_wrench_world"]):
        filtered += float(log["force_filter_alpha"]) * (sample - filtered)
        np.testing.assert_array_equal(log["filtered_wrench_world"][index], filtered)
    np.testing.assert_array_equal(
        log["measured_wrench_world"], log["filtered_wrench_world"][np.maximum(0, steps - delay)]
    )
    np.testing.assert_allclose(
        log["measured_normal_force"],
        log["measured_wrench_world"][:, :3] @ log["controller_frame_rotation"][:, 0],
        rtol=0,
        atol=1e-15,
    )


def test_ideal_contact_force_is_never_used_as_feedback(monkeypatch):
    original = short_trial()
    monkeypatch.setattr(sim, "_normal_contact_force", lambda *args: 1234.0)
    changed = short_trial()
    for field in (
        "measured_normal_force",
        "measured_wrench_world",
        "commanded_wrench",
        "applied_torque",
        "position",
    ):
        np.testing.assert_array_equal(original.trace[field], changed.trace[field])
    np.testing.assert_array_equal(changed.trace["true_normal_force"], 1234.0)
    assert changed.metrics()["peak_force_n"] == 1234.0


def test_identity_frame_matches_world_baseline_and_repeats():
    first = short_trial(controller_frame=SurfaceFrame(np.eye(3)))
    world = short_trial(
        controller_frame=SurfaceFrame(np.eye(3)), controller_kind="world_safe_adaptive"
    )
    repeated = short_trial(controller_frame=SurfaceFrame(np.eye(3)))
    for field in ("commanded_wrench", "applied_torque", "measured_position", "raw_wrench_world"):
        np.testing.assert_array_equal(first.trace[field], world.trace[field])
        np.testing.assert_array_equal(first.trace[field], repeated.trace[field])


def test_horizontal_normal_does_not_observe_gravity_calibration_residual():
    scenario = sim.SurfaceScenario(tool_mass_kg=0.13, wall_yaw_deg=15)
    first = short_trial(scenario=scenario)
    changed = short_trial(scenario=replace(scenario, nominal_tool_mass_kg=0.20))
    # Payload is the same in both plants. Changing only compensation mass cannot
    # establish normal-loop robustness: its residual is vertical in this protocol.
    for field in ("measured_normal_force", "commanded_wrench", "applied_torque", "position"):
        np.testing.assert_array_equal(first.trace[field], changed.trace[field])
    np.testing.assert_allclose(
        changed.trace["raw_wrench_world"][:, 2] - first.trace["raw_wrench_world"][:, 2],
        -0.981,
        rtol=0,
        atol=1e-14,
    )


def test_all_arms_share_world_targets_and_record_current_torque_context():
    task = sim.SurfaceTask(yaw_deg=15)
    exact = short_trial(task=task)
    wrong = short_trial(task=task, controller_frame=sim.yaw_frame(10))
    for name in (
        "target_position",
        "target_rotation",
        "target_linear_velocity",
        "target_angular_velocity",
        "target_normal_force",
    ):
        np.testing.assert_array_equal(exact.trace[name], wrong.trace[name])
    log = exact.trace
    mapped = (
        np.einsum("nij,ni->nj", log["cartesian_jacobian"], log["commanded_wrench"])
        + log["joint_torque_offset"]
    )
    np.testing.assert_allclose(mapped, log["commanded_torque"], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(
        log["applied_torque"],
        np.clip(log["commanded_torque"], log["lower_torque_limit"], log["upper_torque_limit"]),
    )


@pytest.mark.parametrize("time", [0.05, 0.1, 0.4, 1.0, 1.2, 1.4, 1.7, 2.0])
def test_task_has_consistent_analytic_velocity(time):
    task = sim.SurfaceTask(yaw_deg=21)
    initial = np.array([0.32, 0.03, 0.55])
    epsilon = 1e-6
    target = task.target_at(time, initial, np.eye(3), 12)
    before = task.target_at(time - epsilon, initial, np.eye(3), 12)
    after = task.target_at(time + epsilon, initial, np.eye(3), 12)
    np.testing.assert_allclose(
        target.linear_velocity,
        (after.position - before.position) / (2 * epsilon),
        atol=3e-7,
        rtol=0,
    )


def test_metrics_use_true_surface_projection_and_do_not_hide_missing_evaluation():
    result = short_trial(scenario=sim.SurfaceScenario(wall_yaw_deg=30))
    assert result.metrics()["force_rmse_n"] is None
    assert result.metrics()["contact_ratio_pct"] is None
    assert result.metrics()["evaluation_observed"] is False
    result.config = replace(result.config, evaluation_start=0)
    normal = sim.yaw_frame(30).rotation[:, 0]
    tangent = sim.yaw_frame(30).rotation[:, 1]
    result.trace["position"] = result.trace["target_position"] + 0.003 * normal + 0.004 * tangent
    result.trace["true_normal_force"][:] = 12
    result.trace["target_normal_force"][:] = 12
    assert result.metrics()["tangent_rmse_mm"] == pytest.approx(4)
    assert result.metrics()["force_rmse_n"] == 0
    assert result.metrics()["contact_ratio_pct"] == 100


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (sim.SurfaceSimulationConfig, {"duration": 0}),
        (sim.SurfaceSimulationConfig, {"duration": 0.003}),
        (sim.SurfaceSimulationConfig, {"timestep": float("nan")}),
        (sim.SurfaceSimulationConfig, {"evaluation_start": -1}),
        (sim.SurfaceSimulationConfig, {"seed": True}),
        (sim.SurfaceScenario, {"delay_steps": 1.5}),
        (sim.SurfaceScenario, {"force_noise_std_n": -1}),
        (sim.SurfaceScenario, {"tool_mass_kg": 0}),
        (sim.SurfaceScenario, {"force_bias_sensor_n": (1, 2)}),
        (sim.SurfaceTask, {"yaw_deg": float("inf")}),
    ],
)
def test_invalid_configuration_fails_early(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_bad_controller_contract_fails_before_simulation():
    with pytest.raises(ValueError, match="identity"):
        short_trial(controller_kind="world_safe_adaptive")
    with pytest.raises(ValueError, match="unknown"):
        short_trial(controller_kind="mystery")


def test_sensor_reads_solved_step_before_next_kinematic_refresh(monkeypatch):
    phase = {"stage": "reset", "rotation": None, "reads": 0}
    step1, step2, read = sim.mujoco.mj_step1, sim.mujoco.mj_step2, sim.ToolWrenchSensor.read_world

    def first(model, data):
        step1(model, data)
        phase["stage"] = "step1"
        phase["rotation"] = data.site_xmat[model.site("ee_site").id].copy()

    def second(model, data):
        step2(model, data)
        phase["stage"] = "step2"

    def measured(sensor, data):
        assert phase["stage"] in {"reset", "step2"}
        if phase["stage"] == "step2":
            # The just-solved sample uses pre-integration site rotation.
            np.testing.assert_array_equal(data.site_xmat[sensor._site_id], phase["rotation"])
        phase["reads"] += 1
        return read(sensor, data)

    monkeypatch.setattr(sim.mujoco, "mj_step1", first)
    monkeypatch.setattr(sim.mujoco, "mj_step2", second)
    monkeypatch.setattr(sim.ToolWrenchSensor, "read_world", measured)
    result = short_trial()
    assert phase["reads"] == len(result.trace["time"]) + 1


def test_delayed_cartesian_state_uses_current_jacobian(monkeypatch):
    observed = []
    original = sim._site_state

    def site_state(*args):
        value = original(*args)
        observed.append(value)
        return value

    monkeypatch.setattr(sim, "_site_state", site_state)
    result = short_trial(scenario=sim.SurfaceScenario(delay_steps=3, position_noise_std_m=0))
    log = result.trace
    # First observation belongs to the initial forward solve, not a control row.
    for index, actual in enumerate(observed[1:]):
        np.testing.assert_array_equal(log["cartesian_jacobian"][index], actual[-1])
        delayed = observed[max(0, index - 3) + 1]
        np.testing.assert_array_equal(log["measured_position"][index], delayed[0])
    assert not np.array_equal(log["cartesian_jacobian"][-1], log["cartesian_jacobian"][-4])


def test_engine_auto_reset_cannot_produce_plausible_trace(monkeypatch):
    original = sim.mujoco.mj_step2

    def bad_step(model, data):
        original(model, data)
        data.time = 10

    monkeypatch.setattr(sim.mujoco, "mj_step2", bad_step)
    with pytest.raises(RuntimeError, match="time drift"):
        short_trial()
