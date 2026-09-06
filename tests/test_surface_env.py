from dataclasses import replace

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import compliant_control_lab.surface_env as env_module
from compliant_control_lab.surface_env import SurfaceLearningEnv
from compliant_control_lab.surface_experiment import development_cases
from compliant_control_lab.surface_simulation import (
    SurfaceSimulationConfig,
    run_surface_trial,
    yaw_frame,
)


def test_gym_contract_fixed_policy_clock_and_terminal_next_observation():
    env = SurfaceLearningEnv(
        yaw_frame(15),
        config=SurfaceSimulationConfig(
            duration=0.04,
            evaluation_start=1.5,
            contact_model="smooth",
        ),
    )
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=17)
    assert observation.shape == (49,) and observation.dtype == np.float64
    observation, reward, terminated, truncated, info = env.step(np.zeros(3))
    assert not terminated and not truncated and 0 <= reward <= 0.02
    assert info["physics_substeps"] == 10 and info["observation_time_s"] == 0.02
    assert len(env.result().trace["time"]) == 10
    observation, _, terminated, truncated, info = env.step(np.zeros(3))
    assert not terminated and truncated
    assert info["observation_time_s"] == 0.04 and env.observation_space.contains(observation)
    env.close()


def test_zero_action_complete_trace_exactly_matches_independent_friction_trial():
    case = development_cases()[16]
    config = replace(case["config"], contact_model="smooth")
    frame = yaw_frame(case["task"].yaw_deg)
    env = SurfaceLearningEnv(frame, scenario=case["scenario"], config=config, task=case["task"])
    _, initial = env.reset(seed=31)
    total = 0
    for policy_step in range(225):
        _, reward, terminated, truncated, info = env.step(np.zeros(3))
        assert not terminated
        assert truncated == (policy_step == 224)
        total += reward
    assert 0 <= total <= 4.5
    reference = run_surface_trial(
        frame,
        case["scenario"],
        replace(config, seed=initial["simulation_seed"]),
        case["task"],
        controller_kind="surface_friction",
    )
    result = env.result()
    assert str(result.trace["controller_kind"]) == "surface_residual"
    for key, value in reference.trace.items():
        if key != "controller_kind":
            np.testing.assert_array_equal(result.trace[key], value, err_msg=key)
    np.testing.assert_array_equal(result.trace["policy_step_index"], np.repeat(np.arange(225), 10))
    assert info["observation_time_s"] == 4.5
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(3))


def test_seed_reset_and_instances_do_not_share_history():
    config = SurfaceSimulationConfig(duration=0.04, contact_model="smooth")
    first, second = (SurfaceLearningEnv(yaw_frame(0), config=config) for _ in range(2))
    initial, first_info = first.reset(seed=8)
    other, second_info = second.reset(seed=8)
    np.testing.assert_array_equal(initial, other)
    assert first_info["simulation_seed"] == second_info["simulation_seed"]
    a = first.step(np.array([1, -1, 0.2]))
    b = second.step(np.array([1, -1, 0.2]))
    np.testing.assert_array_equal(a[0], b[0])
    first.close()
    second.step(np.zeros(3))
    repeated, repeated_info = first.reset(seed=8)
    np.testing.assert_array_equal(repeated, initial)
    assert repeated_info == first_info
    _, fresh = first.reset()
    assert fresh["simulation_seed"] != first_info["simulation_seed"]


@pytest.mark.parametrize(
    "action", (np.array([np.nan, 0, 0]), np.array([np.inf, 0, 0]), np.zeros(2))
)
def test_invalid_action_executes_nominal_fail_closed_then_terminates(action):
    env = SurfaceLearningEnv(
        yaw_frame(0), config=SurfaceSimulationConfig(duration=0.2, contact_model="smooth")
    )
    env.reset(seed=1)
    observation, reward, terminated, truncated, info = env.step(action)
    assert terminated and not truncated
    assert info["termination_reasons"] == ("invalid_action",)
    assert info["physics_substeps"] == 1 and info["terminal_observation_valid"]
    assert reward <= -1
    assert env.observation_space.contains(observation)
    trace = env.result().trace
    np.testing.assert_array_equal(trace["applied_residual_force_local"], np.zeros((1, 3)))
    assert not trace["policy_action_valid"][0]


def test_no_contact_before_evaluation_is_allowed_but_persistent_loss_terminates():
    env = SurfaceLearningEnv(
        yaw_frame(0),
        config=SurfaceSimulationConfig(
            duration=0.2,
            evaluation_start=0.04,
            contact_model="smooth",
        ),
    )
    env.reset(seed=1)
    total = 0
    for index in range(7):
        _, reward, terminated, truncated, info = env.step(np.zeros(3))
        total += reward
        assert not truncated
        assert terminated == (index == 6)
    assert info["termination_reasons"] == ("contact_lost_timeout",)
    assert info["elapsed_s"] == pytest.approx(0.14)
    assert len(env.result().trace["time"]) == 70
    assert info["termination_penalty"] == 1.2 and total <= -1


def test_actor_has_no_scenario_truth_and_result_is_not_live_state():
    case = development_cases()[16]
    config = replace(case["config"], duration=0.04, contact_model="smooth")
    frames = yaw_frame(15)
    first = SurfaceLearningEnv(frames, scenario=case["scenario"], config=config, task=case["task"])
    changed = replace(case["scenario"], tool_sliding_friction=0.8, wall_sliding_friction=0.8)
    second = SurfaceLearningEnv(frames, scenario=changed, config=config, task=case["task"])
    np.testing.assert_array_equal(first.reset(seed=7)[0], second.reset(seed=7)[0])
    a, b = first.step(np.zeros(3)), second.step(np.zeros(3))
    np.testing.assert_array_equal(a[0], b[0])  # Before contact, only hidden friction differs.
    assert not any("scenario" in key or key.startswith("true_") for key in a[4])
    detached = first.result().trace
    detached["true_normal_force"][:] = 1e9
    detached["position"][:] = np.nan
    np.testing.assert_array_equal(first.step(np.zeros(3))[0], second.step(np.zeros(3))[0])


@pytest.mark.parametrize("value", (1.0001, -1.0001, 1e300))
def test_out_of_space_actions_fail_closed_with_explicit_original_input(value):
    env = SurfaceLearningEnv(yaw_frame(0), config=SurfaceSimulationConfig(duration=0.04))
    env.reset(seed=2)
    _, _, terminated, _, info = env.step([value, 0, 0])
    assert terminated and info["termination_reasons"] == ("invalid_action", "out_of_space")
    assert info["input_action_issue"] == "out_of_space" and info["input_action_shape"] == (3,)
    assert info["invalid_input_repr"] is not None
    assert not env.result().trace["policy_action_valid"][0]
    np.testing.assert_array_equal(env.result().trace["applied_residual_force_local"], [[0, 0, 0]])


def test_legal_action_corners_are_not_declared_invalid():
    env = SurfaceLearningEnv(yaw_frame(0), config=SurfaceSimulationConfig(duration=0.04))
    env.reset(seed=2)
    _, _, terminated, _, info = env.step([1, -1, 1])
    assert not terminated and info["input_action_issue"] is None
    assert info["physics_substeps"] == 10
    assert np.all(env.result().trace["policy_action_valid"])


@pytest.mark.parametrize(
    "reason",
    (
        "torque_missing_context",
        "torque_nominal_outside",
        "torque_nonfinite",
        "torque_verification_failed",
    ),
)
def test_reported_controller_fail_closed_status_is_terminal(monkeypatch, reason):
    # Fault injection at the explicitly contracted controller status Interface.
    class FaultStatus(env_module.SurfaceResidualController):
        @property
        def last_command(self):
            return replace(super().last_command, reasons=(reason,))

    monkeypatch.setattr(env_module, "SurfaceResidualController", FaultStatus)
    env = SurfaceLearningEnv(yaw_frame(0), config=SurfaceSimulationConfig(duration=0.04))
    env.reset(seed=3)
    _, reward, terminated, truncated, info = env.step(np.zeros(3))
    assert terminated and not truncated and info["termination_reasons"] == (reason,)
    assert info["physics_substeps"] == 1 and reward <= -1
    np.testing.assert_array_equal(env.result().trace["applied_residual_force_local"], [[0, 0, 0]])


@pytest.mark.parametrize(
    "fault,reason",
    (("speed", "speed_limit"), ("penetration", "penetration_limit"), ("bias", "actuator_clip")),
)
def test_engine_state_faults_stop_at_the_first_observed_physics_sample(monkeypatch, fault, reason):
    step1 = mujoco.mj_step1

    def injected(model, data):
        step1(model, data)
        if data.time < 0.002:
            return
        if fault == "penetration":
            data.site_xpos[model.site("ee_site").id, 0] = 0.3771
        elif fault == "bias":
            data.qfrc_bias[:7] = 1000
        else:
            jacobian, rotation = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacobian, rotation, model.site("ee_site").id)
            data.qvel[:7] = np.linalg.pinv(jacobian[:, :7]) @ [0.3001, 0, 0]

    monkeypatch.setattr(mujoco, "mj_step1", injected)
    env = SurfaceLearningEnv(yaw_frame(0), config=SurfaceSimulationConfig(duration=0.04))
    env.reset(seed=3)
    _, reward, terminated, _, info = env.step(np.zeros(3))
    assert terminated and reason in info["termination_reasons"]
    # Geometry and speed are observed at x[1], before another action is applied.
    # A changed bias affects torque only when the second action is executed.
    expected_steps = 2 if fault == "bias" else 1
    assert info["physics_substeps"] == expected_steps and reward <= -1
    assert len(env.result().trace["time"]) == expected_steps


@pytest.mark.parametrize("nonfinite", (False, True))
def test_raw_force_fault_and_nonfinite_truth_are_retained_not_actor_inputs(monkeypatch, nonfinite):
    contact_force = mujoco.mj_contactForce

    def injected(model, data, index, force):
        contact_force(model, data, index, force)
        if data.time >= 0.55:
            force[1 if nonfinite else 0] = np.nan if nonfinite else 35.0001

    monkeypatch.setattr(mujoco, "mj_contactForce", injected)
    env = SurfaceLearningEnv(
        yaw_frame(0), config=SurfaceSimulationConfig(duration=0.8, contact_model="smooth")
    )
    env.reset(seed=3)
    total = 0
    for _ in range(40):
        observation, reward, terminated, truncated, info = env.step(np.zeros(3))
        total += reward
        if terminated:
            break
    assert terminated and not truncated and total <= -1
    assert env.observation_space.contains(observation)
    trace = env.result().trace
    if nonfinite:
        assert {"simulator_failure", "nonfinite"} <= set(info["termination_reasons"])
        assert not info["terminal_observation_valid"]
        assert np.isnan(trace["true_tangent_force_n"][-1])
    else:
        assert "raw_force_limit" in info["termination_reasons"]
        assert info["terminal_observation_valid"]
        assert trace["true_normal_force"][-1] > 35
    assert np.all(np.isfinite(trace["true_normal_force"][:-1]))


def test_live_engine_failure_returns_flagged_last_finite_observation(monkeypatch):
    env = SurfaceLearningEnv(yaw_frame(0), config=SurfaceSimulationConfig(duration=0.04))
    initial, _ = env.reset(seed=3)

    def failure(*_args):
        raise RuntimeError("injected engine failure before integration")

    monkeypatch.setattr(mujoco, "mj_step2", failure)
    observation, reward, terminated, _, info = env.step(np.zeros(3))
    assert terminated and info["termination_reasons"] == ("simulator_failure",)
    assert not info["terminal_observation_valid"] and info["physics_substeps"] == 0
    assert info["attempted_substeps"] == 1
    assert info["action_stages"][0]["execution_status"] == "unconfirmed"
    assert reward == -1.04
    np.testing.assert_array_equal(initial, observation)
    with pytest.raises(RuntimeError, match="no executed"):
        env.result()


def test_nonzero_legal_policy_runs_at_50hz_and_action_info_is_owned():
    case = development_cases()[16]
    env = SurfaceLearningEnv(
        yaw_frame(15),
        scenario=case["scenario"],
        task=case["task"],
        config=replace(case["config"], duration=1.4, contact_model="smooth"),
    )
    env.reset(seed=4)
    action = np.array([0.1, 0.2, -0.2])
    for _ in range(70):
        _, _, terminated, _, info = env.step(action)
        assert not terminated
    trace = env.result().trace
    assert np.any(np.abs(trace["applied_residual_force_local"]) > 0)
    np.testing.assert_array_equal(trace["policy_action"], np.tile(action, (700, 1)))
    info["action_stages"][-1]["policy_action"][:] = 99
    np.testing.assert_array_equal(env.result().trace["policy_action"][-1], action)


def test_encoder_observation_is_current_and_age_fields_include_sensor_delay():
    case = development_cases()[16]
    env = SurfaceLearningEnv(
        yaw_frame(15),
        scenario=replace(case["scenario"], delay_steps=3),
        task=case["task"],
        config=replace(case["config"], duration=0.04),
    )
    env.reset(seed=5)
    observation = env.step(np.zeros(3))[0]
    np.testing.assert_allclose(observation[-2:], [0.06, 0.08], rtol=0, atol=1e-12)
    env.step(np.zeros(3))
    trace = env.result().trace
    np.testing.assert_allclose(observation[33:40] * np.pi, trace["q"][10], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        observation[40:47] * 2, trace["joint_velocity"][10], rtol=0, atol=1e-12
    )
    assert (
        len(env_module.OBSERVATION_NAMES)
        == len(env_module.OBSERVATION_SCALES)
        == len(env_module.OBSERVATION_UNITS)
        == 49
    )
