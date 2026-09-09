import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import compliant_control_lab.online_compensation_experiment as experiment
from compliant_control_lab import surface_simulation


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fake_trace(case, arm):
    time = np.arange(0.0, 12.0, case.config.timestep)
    count = len(time)
    tangent = experiment.yaw_frame(15).rotation[:, 1]
    target_position = np.tile(np.array([0.4, 0.0, 0.5]), (count, 1))
    position = target_position + 0.001 * tangent
    target_velocity = np.tile(0.02 * tangent, (count, 1))
    coefficient = np.linspace(0.45, 0.5 if arm == "online" else 0.45, count)
    friction = np.array([case.friction.at(t) for t in time])
    bias = np.array([case.wrench_bias_world.at(t) for t in time])
    feedback_bias = np.concatenate((bias[:1], bias[:-1]))
    force_offset = {"baseline": 1.0, "friction": 0.0, "online": 0.3}.get(arm, 0.0)
    trace = {
        "time": time,
        "position": position,
        "linear_velocity": np.tile(0.018 * tangent, (count, 1)),
        "target_position": target_position,
        "target_linear_velocity": target_velocity,
        "target_normal_force": np.full(count, 12.0),
        "measured_normal_force": np.full(count, 12.0),
        "true_normal_force": np.full(count, 12.0 + force_offset),
        "orientation_error_rad": np.full(count, np.deg2rad(0.05)),
        "commanded_torque": np.zeros((count, 7)),
        "applied_torque": np.zeros((count, 7)),
        "contact_blend": np.ones(count),
        "torque_projection_scale": np.ones(count),
        "requested_tangential_force_world": np.zeros((count, 3)),
        "applied_wall_friction": friction,
        "applied_tool_friction": friction,
        "applied_raw_wrench_bias_world": bias,
        "feedback_raw_wrench_bias_world": feedback_bias,
        "controller_yaw_error_deg": np.full(count, case.controller_yaw_error_deg),
        "controller_coefficient_before_compute": coefficient,
        "controller_coefficient_after_compute": coefficient,
        "controller_update_ready_before_compute": np.ones(count, dtype=bool),
        "controller_update_ready_after_compute": np.ones(count, dtype=bool),
        "trajectory_rate_scale": np.array([case.trajectory.rate_scale_at(t) for t in time]),
    }
    return SimpleNamespace(trace=trace)


def test_protocol_is_fixed_case16_public_development_matrix():
    cases = experiment.protocol_cases()
    assert list(dict.fromkeys(case.name for case in cases)) == [
        "static_nominal",
        "static_low_friction",
        "static_high_friction",
        "friction_step_up",
        "friction_step_down",
        "friction_ramp_up",
        "stop_hold_reverse",
        "wrench_bias_step",
        "normal_calibration_error",
        "modest_combined",
    ]
    assert len(cases) == 20
    assert experiment.ARMS == {
        "baseline": None,
        "integral": "integral",
        "friction": "friction",
        "online": "online",
    }
    assert {case.config.seed for case in cases} == {11, 29}
    for case in cases:
        assert case.config.duration == 12.0
        assert case.config.contact_model == "smooth"
        assert case.scenario.wall_yaw_deg == case.task.yaw_deg == 15.0
        assert case.scenario.wall_time_constant == 0.005
        assert case.scenario.tool_mass_kg == 0.10
    by_name = {case.name: case for case in cases if case.config.seed == 11}
    assert by_name["friction_step_up"].friction.at(5.999) == 0.25
    assert by_name["friction_step_up"].friction.at(6.0) == 0.65
    assert by_name["friction_step_down"].friction.at(6.0) == 0.25
    assert by_name["friction_ramp_up"].friction.at(6.0) == pytest.approx(0.45)
    assert by_name["stop_hold_reverse"].trajectory.rate_scale_at(6.0) == 0.0
    assert by_name["stop_hold_reverse"].trajectory.rate_scale_at(9.0) == -1.0
    normal = experiment.yaw_frame(15).rotation[:, 0]
    np.testing.assert_allclose(
        by_name["wrench_bias_step"].wrench_bias_world.at(6.0)[:3], 1.5 * normal
    )
    assert by_name["normal_calibration_error"].controller_yaw_error_deg == 5.0


def test_scheduled_simulator_applies_before_prepare_once_and_keeps_causal_bias():
    case = next(
        c
        for c in experiment.protocol_cases()
        if c.name == "friction_step_up" and c.config.seed == 11
    )
    case = replace(
        case,
        config=replace(case.config, duration=0.006, evaluation_start=0.0),
        friction=experiment.DeterministicSchedule(
            ((0.0, 0.25), (0.002, 0.65)), interpolation="step"
        ),
        wrench_bias_world=experiment.DeterministicSchedule(
            ((0.0, (0.0,) * 6), (0.002, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0))),
            interpolation="step",
        ),
        phases=(experiment.ProtocolPhase("short", 0.0, 0.006),),
        recovery_start_s=0.002,
    )
    simulator = experiment.ScheduledSurfaceSimulator(case, experiment.yaw_frame(15), "baseline")
    first = simulator.sample()
    repeated = simulator.sample()
    np.testing.assert_array_equal(first.state.position, repeated.state.position)
    assert simulator._input_row["applied_wall_friction"] == 0.25
    simulator.step(np.zeros(6))
    second = simulator.sample()
    # Bias generated by the k=0 solve is zero; the k=.002 raw bias is not visible
    # to the controller until the following causal sample.
    assert second.state.normal_force == pytest.approx(
        simulator._input_row["measured_normal_force"]
    )
    assert simulator._input_row["applied_wall_friction"] == 0.65
    assert simulator._input_row["feedback_raw_wrench_bias_world"][0] == 0.0
    simulator.step(np.zeros(6))
    third = simulator.sample()
    assert simulator._input_row["feedback_raw_wrench_bias_world"][0] == 1.0
    assert third.measured_wrench_sample_time == pytest.approx(0.002)


def test_ideal_force_is_evaluator_only_in_scheduled_runner(monkeypatch):
    case = replace(
        experiment.protocol_cases()[0],
        config=replace(
            experiment.protocol_cases()[0].config, duration=0.02, evaluation_start=0.0
        ),
        phases=(experiment.ProtocolPhase("short", 0.0, 0.02),),
    )
    original = experiment.run_protocol_trial(case, "baseline")
    monkeypatch.setattr(
        surface_simulation, "_normal_contact_force", lambda *args: 1234.0
    )
    changed = experiment.run_protocol_trial(case, "baseline")
    for field in ("measured_normal_force", "commanded_wrench", "applied_torque", "position"):
        np.testing.assert_array_equal(original.trace[field], changed.trace[field])
    np.testing.assert_array_equal(changed.trace["true_normal_force"], 1234.0)


def test_phase_recovery_and_online_telemetry_metrics_are_explicit():
    case = next(
        c
        for c in experiment.protocol_cases()
        if c.name == "friction_step_up" and c.config.seed == 11
    )
    trace = _fake_trace(case, "online")
    summary, phases = experiment.summarize_trial(trace, case)
    assert {row["phase"] for row in phases} == {"pre_step", "post_step"}
    assert summary["worst_force_phase"] in {"pre_step", "post_step"}
    assert summary["force_recovered"] is True
    assert summary["force_recovery_time_s"] == 0.0
    assert summary["tangent_recovery_status"] == "recovered"
    assert summary["tangent_recovery_time_s"] == 0.0
    assert summary["tangential_update_ready_pct"] == 100.0
    assert summary["coefficient_min"] == 0.45
    assert summary["coefficient_max"] == 0.5
    assert summary["max_normal_operation_coefficient_rate_s"] > 0
    assert summary["max_coefficient_reset_jump"] == 0
    assert summary["tangent_velocity_error_rms_m_s"] == pytest.approx(0.002)


@pytest.mark.parametrize("time_s", [4.7, 5.8, 7.4, 8.5])
def test_stop_hold_reverse_target_velocity_is_the_path_derivative(time_s):
    case = next(
        c
        for c in experiment.protocol_cases()
        if c.name == "stop_hold_reverse" and c.config.seed == 11
    )
    epsilon = 1e-6
    initial_position = np.array([0.32, 0.03, 0.55])
    target = case.task.target_at(time_s, initial_position, np.eye(3), 12.0)
    before = case.task.target_at(time_s - epsilon, initial_position, np.eye(3), 12.0)
    after = case.task.target_at(time_s + epsilon, initial_position, np.eye(3), 12.0)
    np.testing.assert_allclose(
        target.linear_velocity,
        (after.position - before.position) / (2.0 * epsilon),
        rtol=0,
        atol=2e-8,
    )


@pytest.fixture
def fake_runs(monkeypatch):
    calls = []

    def run(case, arm, *, rotation_gain_scale=1.0):
        calls.append((case, arm))
        return _fake_trace(case, arm)

    monkeypatch.setattr(experiment, "run_protocol_trial", run)
    monkeypatch.setattr(experiment, "_source_hashes", lambda: {"source.py": "a" * 64})
    return calls


def test_subset_runner_writes_reproducible_atomic_evidence(tmp_path, fake_runs):
    output = experiment.generate_online_compensation_experiment(
        tmp_path / "report",
        case_names=["friction_step_up"],
        arms=["baseline", "friction", "online"],
        seeds=[11],
    )
    assert [(case.name, arm) for case, arm in fake_runs] == [
        ("friction_step_up", "baseline"),
        ("friction_step_up", "friction"),
        ("friction_step_up", "online"),
    ]
    rows = _rows(output / "comparison.csv")
    phases = _rows(output / "phase_metrics.csv")
    assert len(rows) == 3 and len(phases) == 6
    assert all(row["paired_baseline_seed"] == "11" for row in rows)
    online = next(row for row in rows if row["arm"] == "online")
    assert online["all_gates_pass"] == "no"
    assert "paired_friction_force_rmse" in online["failed_gates"]
    assert all(
        "paired_friction_force_rmse" in row["failed_gates"]
        for row in phases
        if row["arm"] == "online"
    )
    assert len(list((output / "traces").glob("*.npz"))) == 3
    protocol = json.loads((output / "protocol.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert protocol["identity"] == experiment.PROTOCOL_ID
    assert protocol["public_development"] is True
    assert protocol["comparison_defined_before_execution"] is True
    assert manifest["is_subset"] is True and manifest["new_holdout"] is False
    assert protocol["controller_constructor_configurations"]["online"][
        "safe_adaptive_base"
    ]["tangential"]["coefficient_rate_limit"] == 0.3
    for name in ("protocol.json", "configurations.json", "source_hashes.json"):
        assert experiment._sha256(output / name) == manifest["input_sha256"][name]
    for name, digest in manifest["artifact_sha256"].items():
        assert experiment._sha256(output / name) == digest
    assert (output / "COMPLETE").read_text().strip() == experiment._sha256(
        output / "manifest.json"
    )


def test_rotational_preset_is_declared_and_applied_to_every_arm(tmp_path, monkeypatch):
    scales = []

    def run(case, arm, *, rotation_gain_scale):
        scales.append(rotation_gain_scale)
        return _fake_trace(case, arm)

    monkeypatch.setattr(experiment, "run_protocol_trial", run)
    output = experiment.generate_online_compensation_experiment(
        tmp_path / "scaled", case_names=["static_nominal"],
        arms=["baseline", "friction", "online"], seeds=[11], rotation_gain_scale=2.0,
    )
    assert scales == [2.0, 2.0, 2.0]
    for name in ("protocol.json", "configurations.json", "manifest.json"):
        assert json.loads((output / name).read_text())["rotation_gain_scale"] == 2.0
    protocol = json.loads((output / "protocol.json").read_text())
    for configuration in protocol["controller_constructor_configurations"].values():
        hybrid = configuration["safe_adaptive_base"]["base"]["base"]
        np.testing.assert_array_equal(hybrid["rotational_stiffness"], [40.0] * 3)
        np.testing.assert_array_equal(hybrid["rotational_damping"], [5 * np.sqrt(2)] * 3)


def test_runner_refuses_overwrite_and_invalid_unpaired_selection(tmp_path, fake_runs):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        experiment.generate_online_compensation_experiment(
            occupied, case_names=["static_nominal"], arms=["baseline"]
        )
    with pytest.raises(ValueError, match="baseline"):
        experiment.generate_online_compensation_experiment(
            tmp_path / "unpaired", case_names=["static_nominal"], arms=["online"]
        )
    assert not fake_runs


def test_default_trace_retention_is_predeclared_and_compact_for_every_run(tmp_path, fake_runs):
    output = experiment.generate_online_compensation_experiment(
        tmp_path / "traces",
        case_names=["stop_hold_reverse"],
        arms=["baseline", "friction", "online"],
        seeds=[11],
    )
    names = {path.name for path in (output / "traces").glob("*.npz")}
    assert len([name for name in names if name.endswith("__compact.npz")]) == 3
    assert {name for name in names if name.endswith("__full.npz")} == {
        "stop_hold_reverse__seed_11__friction__full.npz",
        "stop_hold_reverse__seed_11__online__full.npz",
    }


def test_actual_future_perturbations_do_not_leak_before_their_schedule_time():
    source = next(
        c
        for c in experiment.protocol_cases()
        if c.name == "static_low_friction" and c.config.seed == 11
    )
    common = {
        "config": replace(source.config, duration=0.008, evaluation_start=0.0),
        "phases": (experiment.ProtocolPhase("short", 0.0, 0.008),),
    }
    static = replace(source, **common)
    changed = replace(
        source,
        **common,
        friction=experiment.DeterministicSchedule(((0.0, 0.25), (0.004, 0.65))),
        wrench_bias_world=experiment.DeterministicSchedule(
            ((0.0, (0.0,) * 6), (0.004, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        ),
    )
    first = experiment.run_protocol_trial(static, "baseline").trace
    second = experiment.run_protocol_trial(changed, "baseline").trace
    for field in ("measured_normal_force", "commanded_wrench", "applied_torque", "position"):
        np.testing.assert_array_equal(first[field][:2], second[field][:2])
    np.testing.assert_array_equal(second["applied_wall_friction"], [0.25, 0.25, 0.65, 0.65])


def test_source_mutation_leaves_no_complete_or_partial_output(tmp_path, fake_runs, monkeypatch):
    calls = {"count": 0}

    def hashes():
        calls["count"] += 1
        return {"source.py": ("a" if calls["count"] == 1 else "b") * 64}

    monkeypatch.setattr(experiment, "_source_hashes", hashes)
    output = tmp_path / "mutated"
    with pytest.raises(ValueError, match="changed during execution"):
        experiment.generate_online_compensation_experiment(
            output, case_names=["static_nominal"], arms=["baseline"], seeds=[11]
        )
    assert not output.exists()


def test_failure_never_publishes_complete_or_partial_output(tmp_path, fake_runs, monkeypatch):
    monkeypatch.setattr(
        experiment,
        "summarize_trial",
        lambda *_: (_ for _ in ()).throw(RuntimeError("metric failure")),
    )
    output = tmp_path / "bad"
    with pytest.raises(RuntimeError, match="metric failure"):
        experiment.generate_online_compensation_experiment(
            output, case_names=["static_nominal"], arms=["baseline"]
        )
    assert not output.exists()
