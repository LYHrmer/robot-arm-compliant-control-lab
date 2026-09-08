import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.franka_control import (
    FrankaActuationContext,
    FrankaState,
    FrankaTarget,
)
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame


def _probe_path() -> Path:
    configured = os.environ.get("COMPLIANT_CONTROL_CPP_SURFACE_PROBE")
    if configured:
        return Path(configured)
    return Path(__file__).parents[1] / "build" / "compliant_control_surface_probe"


@dataclass
class TraceStep:
    state: FrankaState
    target: FrankaTarget
    dt: float = 0.002
    timestamp: float = 0.0
    now: float = 0.0
    reset: bool = False


def _rotation_z(angle: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _context(limit: float = 100.0, offset: float = 0.0) -> FrankaActuationContext:
    jacobian = np.zeros((6, 7))
    jacobian[:, :6] = np.eye(6)
    jacobian[:, 6] = np.array([0.2, -0.1, 0.15, 0.05, -0.08, 0.12])
    return FrankaActuationContext(
        cartesian_jacobian=jacobian,
        joint_torque_offset=np.full(7, offset),
        lower_torque_limit=np.full(7, -limit),
        upper_torque_limit=np.full(7, limit),
    )


def _world_step(
    frame: SurfaceFrame,
    index: int,
    *,
    force: float,
    target_velocity: float,
    measured_velocity: float,
    context: FrankaActuationContext | None,
    reset: bool = False,
) -> TraceStep:
    local_position = np.array([0.36 + 3e-6 * index, 0.0002 * index, -0.003])
    local_target_position = np.array([0.38, 0.0002 * index + 0.007, 0.002])
    local_velocity = np.array([0.001, measured_velocity, -0.001])
    local_target_velocity = np.array([0.04, target_velocity, 0.002])
    state = FrankaState(
        position=frame.point_to_world(local_position),
        rotation=frame.rotation @ _rotation_z(-0.015),
        linear_velocity=frame.vector_to_world(local_velocity),
        angular_velocity=frame.vector_to_world(np.array([0.01, -0.02, 0.015])),
        normal_force=force,
        actuation=context,
    )
    target = FrankaTarget(
        position=frame.point_to_world(local_target_position),
        rotation=frame.rotation @ _rotation_z(0.025),
        linear_velocity=frame.vector_to_world(local_target_velocity),
        angular_velocity=frame.vector_to_world(np.array([-0.01, 0.015, -0.005])),
        normal_force=12.0,
    )
    timestamp = 2.0 + 0.002 * index
    return TraceStep(state, target, timestamp=timestamp, now=timestamp, reset=reset)


def _parity_trace(frame: SurfaceFrame, *, exercise_online_bounds: bool = False) -> list[TraceStep]:
    wide = _context()
    trace: list[TraceStep] = []
    # Approach, contact confirmation, full 0.50 s transition, a no-context freeze,
    # tangential reversal, release hysteresis, and explicit reset.
    for index in range(8):
        trace.append(
            _world_step(
                frame,
                index,
                force=0.4,
                target_velocity=0.018,
                measured_velocity=0.012,
                context=wide,
                reset=index == 0,
            )
        )
    contact_end = 1140 if exercise_online_bounds else 290
    for index in range(8, contact_end):
        force = 4.0 + min(7.0, 0.035 * (index - 8))
        if exercise_online_bounds:
            force = 20.0 if 1050 <= index < 1135 else 4.0
            context = None if index in {1000, 1001, 1002} else wide
        else:
            context = None if index in {180, 181} else wide
        trace.append(
            _world_step(
                frame,
                index,
                force=force,
                target_velocity=0.018,
                measured_velocity=0.012,
                context=context,
            )
        )
    reversal_start = contact_end
    reversal_end = 3250 if exercise_online_bounds else 298
    for index in range(reversal_start, reversal_end):
        trace.append(
            _world_step(
                frame,
                index,
                force=11.0,
                target_velocity=-0.018,
                measured_velocity=-0.012,
                context=wide,
            )
        )
    release_start = reversal_end
    release_end = reversal_end + 32
    for index in range(release_start, release_end):
        trace.append(
            _world_step(
                frame,
                index,
                force=0.2,
                target_velocity=-0.018,
                measured_velocity=-0.012,
                context=wide,
            )
        )
    trace.append(
        _world_step(
            frame,
            release_end,
            force=0.0,
            target_velocity=0.0,
            measured_velocity=0.0,
            context=wide,
            reset=True,
        )
    )
    return trace


def _numbers(values) -> list[str]:
    return [format(float(value), ".17g") for value in np.asarray(values).reshape(-1)]


def _encode_trace(frame: SurfaceFrame, trace: list[TraceStep]) -> str:
    lines = [" ".join(_numbers(frame.rotation))]
    for index, step in enumerate(trace):
        context = step.state.actuation
        fields = [
            str(index),
            str(int(step.reset)),
            format(step.timestamp, ".17g"),
            format(step.now, ".17g"),
            format(step.dt, ".17g"),
            str(int(context is not None)),
        ]
        fields += _numbers(step.state.position)
        fields += _numbers(step.state.rotation)
        fields += _numbers(step.state.linear_velocity)
        fields += _numbers(step.state.angular_velocity)
        fields.append(format(step.state.normal_force, ".17g"))
        fields += _numbers(step.target.position)
        fields += _numbers(step.target.rotation)
        fields += _numbers(step.target.linear_velocity)
        fields += _numbers(step.target.angular_velocity)
        fields.append(format(step.target.normal_force, ".17g"))
        if context is not None:
            fields += _numbers(context.cartesian_jacobian)
            fields += _numbers(context.joint_torque_offset)
            fields += _numbers(context.lower_torque_limit)
            fields += _numbers(context.upper_torque_limit)
        lines.append(" ".join(fields))
    return "\n".join(lines)


def _run_probe(mode: str, frame: SurfaceFrame, trace: list[TraceStep]) -> list[list[str]]:
    probe = _probe_path()
    if not probe.is_file():
        pytest.skip("C++ surface probe is not built")
    completed = subprocess.run(
        [str(probe), "--mode", mode],
        input=_encode_trace(frame, trace),
        check=True,
        capture_output=True,
        text=True,
    )
    return list(csv.reader(completed.stdout.splitlines()))


@pytest.mark.parametrize("mode", [None, "friction", "integral", "online"])
def test_cpp_surface_loop_matches_python_over_stateful_rotated_trace(mode: str | None):
    frame = SurfaceFrame(_rotation_z(0.47) @ _rotation_z(-0.08).T)
    trace = _parity_trace(frame, exercise_online_bounds=mode == "online")
    observed = _run_probe("none" if mode is None else mode, frame, trace)
    controller = SurfaceAdaptiveController(frame, tangential_mode=mode)

    assert len(observed) == len(trace)
    equivalent_mu: list[float] = []
    tangential_norm: list[float] = []
    gain_scale: list[float] = []
    update_ready: list[bool] = []
    for index, (row, step) in enumerate(zip(observed, trace, strict=True)):
        if step.reset:
            controller.reset(step.state)
        expected_wrench = controller.compute(step.state, step.target, step.dt)
        assert row[:3] == ["surface_case", str(index), "accepted"]
        np.testing.assert_allclose(
            np.asarray(row[6:12], dtype=float), expected_wrench, rtol=3e-12, atol=3e-11
        )
        expected_mu = controller.equivalent_tangential_coefficient
        expected_tangent = controller.requested_tangential_force_world
        np.testing.assert_allclose(float(row[12]), controller.contact_blend, atol=2e-13)
        np.testing.assert_allclose(float(row[13]), controller.corrected_force_n, atol=2e-12)
        np.testing.assert_allclose(
            float(row[14]), controller.filtered_force_rate_n_s, rtol=2e-12, atol=2e-11
        )
        np.testing.assert_allclose(float(row[15]), expected_mu, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(
            np.asarray(row[16:19], dtype=float), expected_tangent, rtol=3e-12, atol=3e-12
        )
        np.testing.assert_allclose(
            float(row[19]), controller.last_governed_normal_lead_m, atol=2e-13
        )
        np.testing.assert_allclose(
            float(row[20]), controller.last_torque_projection_scale, rtol=2e-12, atol=2e-13
        )
        assert bool(int(row[23])) == controller.tangential_update_ready
        equivalent_mu.append(float(row[15]))
        tangential_norm.append(float(np.linalg.norm(np.asarray(row[16:19], dtype=float))))
        gain_scale.append(float(row[22]))
        update_ready.append(bool(int(row[23])))

    if mode == "online":
        assert any(update_ready)
        assert max(equivalent_mu) == pytest.approx(0.9, abs=2e-12)
        assert min(equivalent_mu[:-1]) == pytest.approx(0.0, abs=2e-12)
        assert equivalent_mu[-1] == pytest.approx(0.45, abs=2e-12)
        assert max(tangential_norm) <= 6.0 + 2e-12
        speed = np.hypot(0.018, 0.002)
        smoothed_six_newtons = 6.0 * speed / np.hypot(speed, 0.005)
        assert max(tangential_norm) == pytest.approx(smoothed_six_newtons, abs=2e-12)
        assert min(gain_scale[:-1]) == pytest.approx(0.65, abs=2e-12)
        assert equivalent_mu[1000:1003] == pytest.approx(
            [equivalent_mu[999]] * 3, abs=2e-12
        )
        assert update_ready[1140] is False


def test_cpp_surface_loop_watchdog_boundaries_reset_and_recover():
    frame = SurfaceFrame(np.eye(3))
    wide = _context()
    trace = [
        _world_step(frame, 0, force=4.0, target_velocity=0.01,
                    measured_velocity=0.01, context=wide, reset=True),
        _world_step(frame, 1, force=4.0, target_velocity=0.01,
                    measured_velocity=0.01, context=wide),
        _world_step(frame, 2, force=4.0, target_velocity=0.01,
                    measured_velocity=0.01, context=wide),
        _world_step(frame, 3, force=0.0, target_velocity=0.01,
                    measured_velocity=0.01, context=wide),
        _world_step(frame, 4, force=0.0, target_velocity=0.01,
                    measured_velocity=0.01, context=wide),
        _world_step(frame, 5, force=0.0, target_velocity=0.01,
                    measured_velocity=0.01, context=wide),
        _world_step(frame, 6, force=0.0, target_velocity=0.01,
                    measured_velocity=0.01, context=wide),
        _world_step(frame, 7, force=0.0, target_velocity=0.01,
                    measured_velocity=0.01, context=None),
    ]
    trace[0].now = trace[0].timestamp
    trace[1].timestamp = trace[0].timestamp
    trace[1].now = trace[0].now + 0.001  # duplicate timestamp is stale
    trace[2].timestamp = trace[0].timestamp - 0.0005
    trace[2].now = trace[1].now + 0.001  # watermark survives stale-sample reset
    trace[3].now = trace[2].now + 0.02
    trace[4].now = trace[3].now + 0.001
    trace[4].timestamp = trace[4].now + 0.001
    trace[5].now = trace[3].now - 0.001
    trace[6].timestamp = trace[4].now + 0.001
    trace[6].now = trace[6].timestamp
    trace[6].state = FrankaState(
        position=np.array([np.nan, 0.0, 0.0]),
        rotation=trace[6].state.rotation,
        linear_velocity=trace[6].state.linear_velocity,
        angular_velocity=trace[6].state.angular_velocity,
        normal_force=trace[6].state.normal_force,
        actuation=trace[6].state.actuation,
    )
    trace[7].timestamp = trace[3].now + 0.01
    trace[7].now = trace[7].timestamp + 0.010  # inclusive age boundary
    # Invalid samples clear timestamp/state, so the next fresh sample recovers.
    rows = _run_probe("online", frame, trace)
    assert [row[2] for row in rows] == [
        "accepted",
        "stale_timestamp",
        "stale_timestamp",
        "expired",
        "future_timestamp",
        "nonmonotonic_now",
        "nonfinite_input",
        "accepted",
    ]
    for row in rows[1:7]:
        np.testing.assert_array_equal(np.asarray(row[6:12], dtype=float), np.zeros(6))
        assert row[4] == "1"
    assert rows[7][4] == "0"
    assert rows[7][5] == "0"  # no actuation context means feasibility is unknown/false
    assert float(rows[7][12]) == 0.0


def test_cpp_watchdog_fault_restarts_contact_confirmation_without_blend_jump():
    frame = SurfaceFrame(np.eye(3))
    wide = _context()
    trace = [
        _world_step(
            frame, index, force=0.0 if index == 0 else 4.0,
            target_velocity=0.01, measured_velocity=0.01,
            context=wide, reset=index == 0,
        )
        for index in range(13)
    ]
    trace[11].timestamp = trace[10].timestamp
    trace[11].now = trace[10].now + trace[11].dt
    rows = _run_probe("none", frame, trace)
    assert 0.0 < float(rows[10][12]) < 0.1
    assert rows[11][2] == "stale_timestamp"
    assert float(rows[11][12]) == 0.0
    assert rows[12][2] == "accepted"
    assert float(rows[12][12]) == 0.0


def test_cpp_surface_loop_reports_scaling_and_infeasible_offset_separately():
    frame = SurfaceFrame(np.eye(3))
    scaled = _world_step(
        frame, 0, force=10.0, target_velocity=0.02,
        measured_velocity=0.01, context=_context(limit=2.0), reset=True
    )
    infeasible = _world_step(
        frame, 1, force=10.0, target_velocity=0.02,
        measured_velocity=0.01, context=_context(limit=2.0, offset=5.0), reset=True
    )
    rows = _run_probe("friction", frame, [scaled, infeasible])
    assert rows[0][3:6] == ["scaled", "0", "1"]
    assert 0.0 < float(rows[0][20]) < 1.0
    assert rows[1][3:6] == ["nominal_outside", "1", "0"]
    np.testing.assert_array_equal(np.asarray(rows[1][6:12], dtype=float), np.zeros(6))


def test_cpp_surface_benchmark_reports_whole_loop_latency_distribution():
    benchmark = _probe_path().with_name("compliant_control_surface_benchmark")
    if not benchmark.is_file():
        pytest.skip("C++ surface benchmark is not built")
    completed = subprocess.run([str(benchmark)], check=True, capture_output=True, text=True)
    metrics = {
        row[0]: float(row[1])
        for row in csv.reader(completed.stdout.splitlines())
    }
    assert metrics.keys() == {"samples", "p50_us", "p99_us", "max_us", "overruns_2ms"}
    assert metrics["samples"] == 20000
    assert 0.0 < metrics["p50_us"] <= metrics["p99_us"] <= metrics["max_us"]
    assert 0.0 <= metrics["overruns_2ms"] <= metrics["samples"]
