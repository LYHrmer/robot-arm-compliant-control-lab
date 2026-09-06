"""Independent archived trajectories and ownership contracts for the surface stepper."""

import hashlib
import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab import surface_simulation as sim
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame
from compliant_control_lab.surface_replay import REPLAY_ABSOLUTE_TOLERANCE

ROOT = Path(__file__).parents[1]


def assert_same(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_same(left[name], right[name])
    else:
        np.testing.assert_array_equal(left, right)


def assert_archived_field(actual, expected, field):
    """Across hosts, allow roundoff only; same-process parity above stays exact."""
    assert actual.shape == expected.shape, field
    assert actual.dtype == expected.dtype, field
    if actual.dtype.kind == "f" and field not in {"time", "dt"}:
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), field
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=REPLAY_ABSOLUTE_TOLERANCE, err_msg=field
        )
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=field)


def test_archive_comparison_accepts_roundoff_but_not_regressions():
    tolerance = REPLAY_ABSOLUTE_TOLERANCE
    assert_archived_field(np.array([np.nextafter(1.0, 2.0)]), np.array([1.0]), "q")
    assert_archived_field(np.array([np.nextafter(tolerance, 0)]), np.zeros(1), "q")
    for actual, expected, field in (
        (np.array([np.nextafter(tolerance, np.inf)]), np.zeros(1), "q"),
        (np.array([np.nan]), np.array([np.nan]), "q"),
        (np.array([np.inf]), np.array([np.inf]), "q"),
        (np.array([np.nextafter(1.0, 2.0)]), np.array([1.0]), "time"),
        (np.array([True]), np.array([False]), "contact_confirmed"),
        (np.ones((1, 1)), np.ones(1), "q"),
        (np.ones(1, dtype=np.float32), np.ones(1), "q"),
    ):
        with pytest.raises(AssertionError):
            assert_archived_field(actual, expected, field)


def short_simulator(**kwargs):
    return sim.SurfaceSimulator(
        sim.yaw_frame(15),
        scenario=kwargs.pop("scenario", sim.SurfaceScenario(delay_steps=2)),
        config=kwargs.pop(
            "config", sim.SurfaceSimulationConfig(duration=0.008, contact_model="smooth")
        ),
        **kwargs,
    )


def execute(simulator, controller):
    for index in range(round(simulator.config.duration / simulator.config.timestep)):
        sample = simulator.sample()
        if index == 0:
            controller.reset(sample.state)
        simulator.step(
            controller.compute(sample.state, sample.target, simulator.config.timestep), controller
        )
    return simulator.result()


@pytest.mark.parametrize("method", ["baseline", "integral", "friction"])
def test_full_smooth_case16_numerically_matches_independent_published_trace(method):
    directory = ROOT / "results/franka_tangential_development"
    manifest = json.loads((directory / "manifest.json").read_text())
    entry = next(
        row
        for row in manifest["case_configurations"]
        if row["case_index"] == 16 and row["category"] == "main" and row["method"] == method
    )
    name = f"representative_case_16_{method}.npz"
    assert (
        hashlib.sha256((directory / name).read_bytes()).hexdigest()
        == manifest["artifact_sha256"][name]
    )
    frame = SurfaceFrame(np.asarray(entry["controller_frame_rotation"]))
    simulator = sim.SurfaceSimulator(
        frame,
        sim.SurfaceScenario(**entry["scenario"]),
        sim.SurfaceSimulationConfig(**entry["config"]),
        sim.SurfaceTask(**entry["task"]),
        entry["controller_kind"],
    )
    controller = SurfaceAdaptiveController(
        frame, tangential_mode=None if method == "baseline" else method
    )
    result = execute(simulator, controller)
    with np.load(directory / name, allow_pickle=False) as archive:
        assert result.trace.keys() == set(archive.files)
        for field in archive.files:
            assert_archived_field(result.trace[field], archive[field], field)


def test_full_legacy_case16_preserves_original_frame_trace_fields():
    manifest = json.loads((ROOT / "results/franka_surface_contact_fix/manifest.json").read_text())
    reference = manifest["legacy_trace_references"]["surface_exact"]
    path = ROOT / reference["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    frame = sim.yaw_frame(15)
    simulator = sim.SurfaceSimulator(
        frame,
        sim.SurfaceScenario(name="surface_dev_16", wall_yaw_deg=15, wall_time_constant=0.005),
        sim.SurfaceSimulationConfig(contact_model="legacy"),
        sim.SurfaceTask(yaw_deg=15),
    )
    result = execute(simulator, SurfaceAdaptiveController(frame))
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) <= result.trace.keys()
        for field in archive.files:
            assert_archived_field(result.trace[field], archive[field], field)


def test_repeated_sample_does_not_repeat_engine_refresh_filter_or_noise(monkeypatch):
    subject, reference = short_simulator(), short_simulator()
    original, refreshes = sim.mujoco.mj_step1, []

    def refresh(model, data):
        refreshes.append(float(data.time))
        original(model, data)

    monkeypatch.setattr(sim.mujoco, "mj_step1", refresh)
    action = np.array([1.0, -0.5, 0.2, 0, 0, 0])
    for _ in range(4):
        expected, first = reference.sample(), subject.sample()
        count = len(refreshes)
        assert_same(asdict(first), asdict(expected))
        for _ in range(5):
            assert_same(asdict(subject.sample()), asdict(first))
        assert len(refreshes) == count
        assert_same(subject.step(action), reference.step(action))
    assert len(refreshes) == 8
    assert_same(subject.result().trace, reference.result().trace)


def test_owned_sample_row_and_result_cannot_mutate_simulator():
    subject, reference = short_simulator(), short_simulator()

    def corrupt_arrays(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind in "biuf":
                value[...] = 42
        elif isinstance(value, dict) or is_dataclass(value):
            for child in (value if isinstance(value, dict) else vars(value)).values():
                corrupt_arrays(child)

    sample = subject.sample()
    expected = asdict(reference.sample())
    corrupt_arrays(sample)
    assert_same(asdict(subject.sample()), expected)

    class TelemetryOnly:
        def compute(self, *args):
            pytest.fail("the simulator must not compute a supplied controller")

    action = np.zeros(6)
    row = subject.step(action, TelemetryOnly())
    expected_row = reference.step(np.zeros(6))
    assert_same(row, expected_row)
    corrupt_arrays(row)
    action[:] = 99
    assert_same(subject.result().trace, reference.result().trace)
    result = subject.result()
    corrupt_arrays(result.trace)
    assert_same(subject.result().trace, reference.result().trace)
    assert_same(asdict(subject.sample()), asdict(reference.sample()))


def test_fresh_instances_reset_seeds_and_interleaving_has_no_shared_state():
    first_config = sim.SurfaceSimulationConfig(duration=0.008, contact_model="smooth", seed=11)
    second_config = replace(first_config, seed=29)
    actions = [np.array([2.0, -1, 0.5, 0, 0, 0]) * index for index in range(4)]

    def sequential(config, values):
        simulator = short_simulator(config=config)
        for action in values:
            simulator.step(action)
        return simulator.result().trace

    expected_first = sequential(first_config, actions)
    expected_second = sequential(second_config, actions[::-1])
    first = short_simulator(config=first_config)
    second = short_simulator(config=second_config)
    for left, right in zip(actions, actions[::-1], strict=True):
        first.step(left)
        second.step(right)
    assert_same(first.result().trace, expected_first)
    assert_same(second.result().trace, expected_second)
    assert_same(sequential(first_config, actions), expected_first)
    assert not np.array_equal(
        expected_first["measured_position"][0], expected_second["measured_position"][0]
    )


def test_horizon_exposes_one_final_causal_sample_but_rejects_another_action(monkeypatch):
    simulator = short_simulator()
    original, solves = sim.mujoco.mj_step2, []

    def solve(model, data):
        solves.append(float(data.time))
        original(model, data)

    monkeypatch.setattr(sim.mujoco, "mj_step2", solve)
    with pytest.raises(RuntimeError, match="no executed"):
        simulator.result()
    for _ in range(4):
        simulator.step(np.ones(6))
    before = simulator.result().trace
    final = simulator.sample()
    assert final.time == 0.008
    assert final.measured_kinematic_sample_time == 0.004
    assert final.measured_wrench_sample_time == 0.002
    assert not np.array_equal(final.q, before["q"][-1])
    assert_same(asdict(simulator.sample()), asdict(final))
    with pytest.raises(RuntimeError, match="horizon"):
        simulator.step(np.zeros(6))
    assert len(solves) == 4
    assert_same(simulator.result().trace, before)


@pytest.mark.parametrize(
    "bad",
    [np.full(6, np.nan), np.full(6, np.inf), np.full(6, -np.inf), np.zeros(3), np.zeros((6, 1))],
)
@pytest.mark.parametrize("prepared", [False, True])
def test_invalid_action_is_rejected_before_solve_without_consuming_input(
    bad, prepared, monkeypatch
):
    subject, reference = short_simulator(), short_simulator()
    if prepared:
        subject.sample()

    def forbidden(*args):
        pytest.fail("invalid wrench reached physics integration")

    with monkeypatch.context() as patch:
        patch.setattr(sim.mujoco, "mj_step2", forbidden)
        with pytest.raises(ValueError, match="finite 6-vector"):
            subject.step(bad)
    with pytest.raises(RuntimeError, match="no executed"):
        subject.result()
    assert_same(asdict(subject.sample()), asdict(reference.sample()))
    assert_same(subject.step(np.zeros(6)), reference.step(np.zeros(6)))


def test_executed_nonfinite_evaluator_row_is_reported_and_not_silently_dropped(monkeypatch):
    simulator = short_simulator()
    simulator.step(np.zeros(6))
    with monkeypatch.context() as patch:
        patch.setattr(sim, "_normal_contact_force", lambda *_: float("nan"))
        with pytest.raises(RuntimeError, match="nonfinite"):
            simulator.step(np.zeros(6))
    # Physics ran twice: result must not pretend that only the successful row exists.
    trace = simulator.result().trace
    np.testing.assert_array_equal(trace["time"], [0.0, 0.002])
    assert np.isfinite(trace["true_normal_force"][0])
    assert np.isnan(trace["true_normal_force"][1])
    assert simulator.sample().time == 0.004
