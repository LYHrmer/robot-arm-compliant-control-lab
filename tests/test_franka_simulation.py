import numpy as np
import pytest

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.franka_control import FrankaActuationContext, default_franka_controllers
from compliant_control_lab.franka_simulation import (
    FrankaSimulationConfig,
    FrankaTrialResult,
    run_franka_trial,
)


@pytest.mark.parametrize("controller_name", list(default_franka_controllers()))
def test_each_franka_controller_runs_in_mujoco(controller_name):
    result = run_franka_trial(
        default_franka_controllers()[controller_name],
        config=FrankaSimulationConfig(duration=0.06),
    )
    assert len(result.time) == 30
    assert result.q.shape == (30, 7)
    assert np.all(np.isfinite(result.position))
    assert np.all(np.isfinite(result.torque))


class RecordingController:
    name = "recording"

    def __init__(self) -> None:
        self.contexts: list[FrankaActuationContext] = []

    def reset(self, state) -> None:
        del state

    def compute(self, state, target, dt):
        del target, dt
        assert state.actuation is not None
        self.contexts.append(state.actuation)
        return np.zeros(6)


class TelemetryRecordingController(RecordingController):
    @property
    def contact_blend(self) -> float:
        return 0.25

    @property
    def last_governed_normal_lead_m(self) -> float:
        return 0.002

    @property
    def last_torque_projection_scale(self) -> float:
        return 0.75


def test_simulation_adapter_supplies_actuation_context_each_cycle():
    controller = RecordingController()
    result = run_franka_trial(
        controller,
        config=FrankaSimulationConfig(duration=0.01),
    )

    assert len(controller.contexts) == len(result.time) == 5
    assert all(context.cartesian_jacobian.shape == (6, 7) for context in controller.contexts)
    assert all(context.joint_torque(np.zeros(6)).shape == (7,) for context in controller.contexts)


def test_simulation_records_event_level_telemetry_each_cycle():
    controller = RecordingController()
    result = run_franka_trial(
        controller,
        config=FrankaSimulationConfig(duration=0.01),
    )

    assert result.linear_velocity.shape == (5, 3)
    assert result.target_linear_velocity.shape == (5, 3)
    assert result.commanded_wrench.shape == (5, 6)
    assert result.minimum_torque_headroom_nm.shape == (5,)
    assert len(result.controller_snapshots) == 5
    assert np.all(np.isfinite(result.linear_velocity))
    assert np.allclose(result.target_linear_velocity, 0.0)
    assert np.allclose(result.commanded_wrench, 0.0)

    for context, headroom in zip(
        controller.contexts,
        result.minimum_torque_headroom_nm,
        strict=True,
    ):
        torque = context.joint_torque(np.zeros(6))
        expected = np.min(
            np.minimum(
                torque - context.lower_torque_limit,
                context.upper_torque_limit - torque,
            )
        )
        assert headroom == pytest.approx(expected)


def test_safe_adaptive_public_snapshot_is_recorded_after_each_update():
    result = run_franka_trial(
        FrankaSafeAdaptiveController(),
        config=FrankaSimulationConfig(duration=0.01),
    )

    assert all(snapshot.contact_blend is not None for snapshot in result.controller_snapshots)
    assert all(
        snapshot.governed_normal_lead_m is not None
        for snapshot in result.controller_snapshots
    )
    assert all(
        snapshot.torque_projection_scale is not None
        for snapshot in result.controller_snapshots
    )


def test_reading_public_controller_telemetry_does_not_change_the_rollout():
    config = FrankaSimulationConfig(duration=0.04, seed=29)
    plain = run_franka_trial(RecordingController(), config=config)
    instrumented = run_franka_trial(TelemetryRecordingController(), config=config)

    assert np.array_equal(instrumented.q, plain.q)
    assert np.array_equal(instrumented.position, plain.position)
    assert np.array_equal(instrumented.raw_normal_force, plain.raw_normal_force)
    assert np.array_equal(instrumented.torque, plain.torque)
    assert np.array_equal(instrumented.commanded_wrench, plain.commanded_wrench)
    deterministic_metrics = set(plain.metrics()) - {"controller_p95_us"}
    assert {key: plain.metrics()[key] for key in deterministic_metrics} == {
        key: instrumented.metrics()[key] for key in deterministic_metrics
    }
    assert all(
        snapshot.contact_blend == 0.25
        and snapshot.governed_normal_lead_m == 0.002
        and snapshot.torque_projection_scale == 0.75
        for snapshot in instrumented.controller_snapshots
    )


def test_franka_hybrid_controller_tracks_contact_force():
    result = run_franka_trial(
        default_franka_controllers()["hybrid"],
        config=FrankaSimulationConfig(duration=2.2),
    )
    metrics = result.metrics()
    assert metrics["force_rmse_n"] < 2.0
    assert metrics["contact_ratio_pct"] > 95.0
    assert metrics["saturation_pct"] < 1.0


def test_safety_metrics_include_the_approach_window():
    time = np.array([0.0, 1.0, 2.0])
    result = FrankaTrialResult(
        controller="test",
        scenario="test",
        time=time,
        q=np.zeros((3, 7)),
        position=np.zeros((3, 3)),
        desired_position=np.zeros((3, 3)),
        normal_force=np.array([0.0, 0.0, 12.0]),
        raw_normal_force=np.array([50.0, 0.0, 12.0]),
        desired_force=np.array([0.0, 0.0, 12.0]),
        orientation_error_rad=np.zeros(3),
        torque=np.zeros((3, 7)),
        controller_time_us=np.ones(3),
        saturated=np.array([True, False, False]),
        linear_velocity=np.zeros((3, 3)),
        target_linear_velocity=np.zeros((3, 3)),
        commanded_wrench=np.zeros((3, 6)),
        minimum_torque_headroom_nm=np.ones(3),
        controller_snapshots=(),
    )
    metrics = result.metrics(evaluation_start=1.5)
    assert metrics["force_rmse_n"] == 0.0
    assert metrics["peak_force_n"] == 50.0
    assert metrics["saturation_pct"] == pytest.approx(100.0 / 3.0)
