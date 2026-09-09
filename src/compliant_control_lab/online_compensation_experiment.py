"""Causal public-development comparison for online tangential compensation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np

from compliant_control_lab.franka_control import FrankaTarget
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame
from compliant_control_lab.surface_experiment import development_cases
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    SurfaceSimulator,
    SurfaceTask,
    SurfaceTrialResult,
    yaw_frame,
)
from compliant_control_lab.tangential_compensation import TangentialCompensation

PROTOCOL_ID = "online-compensation-error-comparison-v1"
ARMS = {
    "baseline": None,
    "integral": "integral",
    "friction": "friction",
    "online": "online",
}
GATES = {
    "minimum_contact_ratio_pct": 99.0,
    "maximum_raw_peak_force_n": 35.0,
    "required_saturation_pct": 0.0,
    "maximum_paired_force_rmse_increase_n": 0.2,
    "maximum_paired_orientation_rmse_increase_deg": 0.1,
}
RECOVERY_WINDOW_S = 0.25
RECOVERY_ABSOLUTE_FORCE_ERROR_N = 1.0
RECOVERY_TANGENT_RMSE_MM = 3.0
REPRESENTATIVE_CASE_INDEX = 16
PROTOCOL_SEEDS = (11, 29)
FULL_TRACE_SELECTION = {
    ("friction_step_down", 11, "friction"),
    ("friction_step_down", 11, "online"),
    ("stop_hold_reverse", 11, "friction"),
    ("stop_hold_reverse", 11, "online"),
}


@dataclass(frozen=True)
class DeterministicSchedule:
    """Finite step or piecewise-linear values on the simulation clock."""

    knots: tuple[tuple[float, object], ...]
    interpolation: str = "step"

    def __post_init__(self) -> None:
        if self.interpolation not in {"step", "linear"}:
            raise ValueError("schedule interpolation must be step or linear")
        if not self.knots:
            raise ValueError("schedule needs at least one knot")
        times = np.asarray([item[0] for item in self.knots], dtype=float)
        values = [np.asarray(item[1], dtype=float) for item in self.knots]
        if (
            not np.all(np.isfinite(times))
            or times[0] != 0.0
            or np.any(np.diff(times) <= 0.0)
            or any(value.shape != values[0].shape or not np.all(np.isfinite(value)) for value in values)
        ):
            raise ValueError("schedule knots must start at zero, increase, and have finite equal shapes")
        canonical = tuple(
            (float(time), float(value) if value.ndim == 0 else tuple(float(x) for x in value))
            for time, value in zip(times, values)
        )
        object.__setattr__(self, "knots", canonical)

    def at(self, time_s: float):
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("schedule time must be finite and nonnegative")
        times = np.asarray([item[0] for item in self.knots])
        values = [np.asarray(item[1], dtype=float) for item in self.knots]
        right = int(np.searchsorted(times, time_s, side="right"))
        if self.interpolation == "step" or right == 0 or right == len(times):
            result = values[max(0, right - 1)]
        else:
            left = right - 1
            weight = (time_s - times[left]) / (times[right] - times[left])
            result = values[left] + weight * (values[right] - values[left])
        return float(result) if result.ndim == 0 else result.copy()


@dataclass(frozen=True)
class TrajectoryProfile:
    """Original ellipse or a smooth stop/hold/reverse of its path clock."""

    name: str = "standard"

    def __post_init__(self) -> None:
        if self.name not in {"standard", "stop_hold_reverse"}:
            raise ValueError("unknown trajectory profile")

    @staticmethod
    def _smooth(u: float) -> float:
        return u * u * (3.0 - 2.0 * u)

    def clock_and_rate_at(self, time_s: float) -> tuple[float, float]:
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("trajectory time must be finite and nonnegative")
        if time_s <= 1.2:
            return 0.0, 0.0
        elapsed = time_s - 1.2
        if elapsed < 0.5:
            u = elapsed / 0.5
            return 0.5 * (u**3 - 0.5 * u**4), self._smooth(u)
        clock = elapsed - 0.25
        if self.name == "standard" or time_s < 4.5:
            return clock, 1.0
        clock_at_brake = 4.5 - 1.2 - 0.25
        if time_s < 5.5:
            u = time_s - 4.5
            return clock_at_brake + u - u**3 + 0.5 * u**4, 1.0 - self._smooth(u)
        clock_at_hold = clock_at_brake + 0.5
        if time_s < 7.0:
            return clock_at_hold, 0.0
        if time_s < 8.0:
            u = time_s - 7.0
            return clock_at_hold - (u**3 - 0.5 * u**4), -self._smooth(u)
        return clock_at_brake - (time_s - 8.0), -1.0

    def rate_scale_at(self, time_s: float) -> float:
        return self.clock_and_rate_at(time_s)[1]


@dataclass(frozen=True)
class ProtocolSurfaceTask(SurfaceTask):
    trajectory: TrajectoryProfile = TrajectoryProfile()

    def target_at(
        self,
        time: float,
        initial_position: np.ndarray,
        initial_rotation: np.ndarray,
        target_force: float,
    ) -> FrankaTarget:
        if self.trajectory.name == "standard":
            return super().target_at(time, initial_position, initial_rotation, target_force)
        target = super().target_at(min(time, 1.2), initial_position, initial_rotation, target_force)
        if time <= 1.2:
            return target
        clock, rate_scale = self.trajectory.clock_and_rate_at(time)
        omega = 2.0 * np.pi * 0.20
        angle = omega * clock
        _, tangent1, tangent2 = yaw_frame(self.yaw_deg).rotation.T
        position = target.position.copy()
        position += 0.055 * np.sin(angle) * tangent1
        position += 0.040 * (np.cos(angle) - 1.0) * tangent2
        velocity = rate_scale * omega * (
            0.055 * np.cos(angle) * tangent1 - 0.040 * np.sin(angle) * tangent2
        )
        return replace(target, position=position, linear_velocity=velocity)


@dataclass(frozen=True)
class ProtocolPhase:
    name: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not self.name or not np.all(np.isfinite([self.start_s, self.end_s])):
            raise ValueError("phase must have a name and finite bounds")
        if self.start_s < 0.0 or self.end_s <= self.start_s:
            raise ValueError("phase end must follow its nonnegative start")


@dataclass(frozen=True)
class ProtocolCase:
    name: str
    scenario: SurfaceScenario
    config: SurfaceSimulationConfig
    task: ProtocolSurfaceTask
    friction: DeterministicSchedule
    wrench_bias_world: DeterministicSchedule
    controller_yaw_error_deg: float
    phases: tuple[ProtocolPhase, ...]
    recovery_start_s: float | None = None

    @property
    def trajectory(self) -> TrajectoryProfile:
        return self.task.trajectory

    def __post_init__(self) -> None:
        if not self.name or not np.isfinite(self.controller_yaw_error_deg):
            raise ValueError("case name and controller calibration must be valid")
        if np.asarray(self.friction.at(0.0)).shape != () or self.friction.at(0.0) < 0.0:
            raise ValueError("friction schedule must be scalar and nonnegative")
        if np.asarray(self.wrench_bias_world.at(0.0)).shape != (6,):
            raise ValueError("wrench-bias schedule must contain world six-vectors")
        if self.recovery_start_s is not None and not (
            self.config.evaluation_start <= self.recovery_start_s < self.config.duration
        ):
            raise ValueError("recovery start must lie in the evaluation interval")
        if not self.phases or any(phase.end_s > self.config.duration for phase in self.phases):
            raise ValueError("case phases must be nonempty and lie within the trial")


def _constant(value: float) -> DeterministicSchedule:
    return DeterministicSchedule(((0.0, value),))


def _zero_bias() -> DeterministicSchedule:
    return DeterministicSchedule(((0.0, (0.0,) * 6),))


def protocol_cases() -> tuple[ProtocolCase, ...]:
    """Return two predetermined noise seeds for each fixed case-16 profile."""
    source = development_cases()[REPRESENTATIVE_CASE_INDEX]
    base_config = replace(source["config"], duration=12.0, contact_model="smooth")
    base_scenario = source["scenario"]
    yaw = source["task"].yaw_deg
    normal = yaw_frame(yaw).rotation[:, 0]
    zero = _zero_bias()

    def build(
        name: str,
        friction: DeterministicSchedule,
        phases: tuple[ProtocolPhase, ...],
        *,
        bias: DeterministicSchedule = zero,
        yaw_error: float = 0.0,
        trajectory: str = "standard",
        recovery: float | None = None,
    ) -> ProtocolCase:
        initial_friction = friction.at(0.0)
        return ProtocolCase(
            name=name,
            scenario=replace(
                base_scenario,
                name=f"online_error_{name}",
                wall_sliding_friction=initial_friction,
                tool_sliding_friction=initial_friction,
            ),
            config=base_config,
            task=ProtocolSurfaceTask(yaw_deg=yaw, trajectory=TrajectoryProfile(trajectory)),
            friction=friction,
            wrench_bias_world=bias,
            controller_yaw_error_deg=yaw_error,
            phases=phases,
            recovery_start_s=recovery,
        )

    steady = (ProtocolPhase("steady", 1.5, 12.0),)
    step_phases = (ProtocolPhase("pre_step", 1.5, 6.0), ProtocolPhase("post_step", 6.0, 12.0))
    ramp_phases = (
        ProtocolPhase("pre_ramp", 1.5, 4.0),
        ProtocolPhase("ramp", 4.0, 8.0),
        ProtocolPhase("post_ramp", 8.0, 12.0),
    )
    motion_phases = (
        ProtocolPhase("forward", 1.5, 4.5),
        ProtocolPhase("stopping", 4.5, 5.5),
        ProtocolPhase("hold", 5.5, 7.0),
        ProtocolPhase("reverse_ramp", 7.0, 8.0),
        ProtocolPhase("reverse", 8.0, 12.0),
    )
    profiles = (
        build("static_nominal", _constant(0.45), steady),
        build("static_low_friction", _constant(0.25), steady),
        build("static_high_friction", _constant(0.65), steady),
        build(
            "friction_step_up",
            DeterministicSchedule(((0.0, 0.25), (6.0, 0.65))),
            step_phases,
            recovery=6.0,
        ),
        build(
            "friction_step_down",
            DeterministicSchedule(((0.0, 0.65), (6.0, 0.25))),
            step_phases,
            recovery=6.0,
        ),
        build(
            "friction_ramp_up",
            DeterministicSchedule(((0.0, 0.25), (4.0, 0.25), (8.0, 0.65)), "linear"),
            ramp_phases,
            recovery=8.0,
        ),
        build(
            "stop_hold_reverse",
            _constant(0.45),
            motion_phases,
            trajectory="stop_hold_reverse",
            recovery=8.0,
        ),
        build(
            "wrench_bias_step",
            _constant(0.45),
            step_phases,
            bias=DeterministicSchedule(
                ((0.0, (0.0,) * 6), (6.0, tuple(1.5 * normal) + (0.0, 0.0, 0.0)))
            ),
            recovery=6.0,
        ),
        build("normal_calibration_error", _constant(0.45), steady, yaw_error=5.0),
        build(
            "modest_combined",
            DeterministicSchedule(((0.0, 0.30), (4.0, 0.30), (8.0, 0.60)), "linear"),
            (
                ProtocolPhase("pre_change", 1.5, 4.0),
                ProtocolPhase("combined_transition", 4.0, 8.0),
                ProtocolPhase("combined_post", 8.0, 12.0),
            ),
            bias=DeterministicSchedule(
                ((0.0, (0.0,) * 6), (6.0, tuple(0.75 * normal) + (0.0, 0.0, 0.0)))
            ),
            yaw_error=3.0,
            trajectory="stop_hold_reverse",
            recovery=8.0,
        ),
    )
    return tuple(
        replace(
            profile,
            scenario=replace(profile.scenario, name=f"{profile.scenario.name}_seed_{seed}"),
            config=replace(profile.config, seed=seed),
        )
        for profile in profiles
        for seed in PROTOCOL_SEEDS
    )


class _ScheduledBiasSensor:
    """Add deterministic world bias after exactly one delegate sensor read."""

    def __init__(self, delegate, bias: np.ndarray) -> None:
        self._delegate = delegate
        self.bias = np.asarray(bias, dtype=float)

    def read_world(self, data) -> np.ndarray:
        return self._delegate.read_world(data) + self.bias


class ScheduledSurfaceSimulator(SurfaceSimulator):
    """Diagnostic simulator that changes public-development inputs causally."""

    def __init__(self, case: ProtocolCase, controller_frame: SurfaceFrame, arm: str) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        kind = "surface_adaptive" if arm == "baseline" else f"surface_{arm}"
        self.protocol_case = case
        self.arm = arm
        super().__init__(controller_frame, case.scenario, case.config, case.task, kind)
        initial_bias = np.asarray(case.wrench_bias_world.at(0.0), dtype=float)
        self._sensor = _ScheduledBiasSensor(self._sensor, initial_bias)
        self._previous_wrench += initial_bias
        self._last_raw_bias = initial_bias.copy()
        self._current_raw_bias = initial_bias.copy()
        self._current_friction = float(case.friction.at(0.0))

    def _prepare(self) -> None:
        if self._pending is not None:
            return
        time_s = self._step * self.config.timestep
        self._current_friction = float(self.protocol_case.friction.at(time_s))
        self._current_raw_bias = np.asarray(
            self.protocol_case.wrench_bias_world.at(time_s), dtype=float
        )
        self._model.geom_friction[self._wall_id, 0] = self._current_friction
        self._model.geom_friction[self._tool_id, 0] = self._current_friction
        self._sensor.bias = self._current_raw_bias
        super()._prepare()
        self._input_row.update(
            applied_wall_friction=self._current_friction,
            applied_tool_friction=self._current_friction,
            applied_raw_wrench_bias_world=self._current_raw_bias.copy(),
            feedback_raw_wrench_bias_world=self._last_raw_bias.copy(),
            controller_yaw_error_deg=self.protocol_case.controller_yaw_error_deg,
            trajectory_rate_scale=self.protocol_case.trajectory.rate_scale_at(time_s),
        )

    def step(self, wrench, controller=None, *, telemetry_before=None) -> dict:
        if telemetry_before is None:
            telemetry_before = (
                float(getattr(controller, "equivalent_tangential_coefficient", 0.0)),
                bool(getattr(controller, "tangential_update_ready", False)),
            )
        telemetry_after = (
            float(getattr(controller, "equivalent_tangential_coefficient", 0.0)),
            bool(getattr(controller, "tangential_update_ready", False)),
        )
        row = super().step(wrench, controller)
        extra = {
            "controller_coefficient_before_compute": telemetry_before[0],
            "controller_coefficient_after_compute": telemetry_after[0],
            "controller_update_ready_before_compute": telemetry_before[1],
            "controller_update_ready_after_compute": telemetry_after[1],
        }
        self._rows[-1].update(extra)
        row.update(extra)
        self._last_raw_bias = self._current_raw_bias.copy()
        return row


def run_protocol_trial(
    case: ProtocolCase, arm: str, *, rotation_gain_scale: float = 1.0,
) -> SurfaceTrialResult:
    """Run one measured-input-only arm/case pair on the common simulation seed."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    frame = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg)
    controller = SurfaceAdaptiveController(
        frame, tangential_mode=ARMS[arm], rotation_gain_scale=rotation_gain_scale,
    )
    simulator = ScheduledSurfaceSimulator(case, frame, arm)
    steps = round(case.config.duration / case.config.timestep)
    for step in range(steps):
        sample = simulator.sample()
        if step == 0:
            controller.reset(sample.state)
        before = (
            controller.equivalent_tangential_coefficient,
            controller.tangential_update_ready,
        )
        wrench = controller.compute(sample.state, sample.target, case.config.timestep)
        simulator.step(wrench, controller, telemetry_before=before)
    result = simulator.result()
    result.trace["rotation_gain_scale"] = np.array(float(rotation_gain_scale))
    return result


def _window_metrics(trace: dict[str, np.ndarray], mask: np.ndarray, normal: np.ndarray) -> dict:
    force_error = trace["true_normal_force"][mask] - trace["target_normal_force"][mask]
    position_error = trace["position"][mask] - trace["target_position"][mask]
    tangent_position = position_error - np.outer(position_error @ normal, normal)
    velocity_error = trace["linear_velocity"][mask] - trace["target_linear_velocity"][mask]
    tangent_velocity = velocity_error - np.outer(velocity_error @ normal, normal)
    saturated = np.any(
        np.abs(trace["commanded_torque"][mask] - trace["applied_torque"][mask]) > 1e-9,
        axis=1,
    )
    return {
        "sample_count": int(np.count_nonzero(mask)),
        "force_rmse_n": float(np.sqrt(np.mean(force_error**2))),
        "peak_force_n": float(np.max(trace["true_normal_force"][mask])),
        "contact_ratio_pct": float(100.0 * np.mean(trace["true_normal_force"][mask] > 0.5)),
        "tangent_rmse_mm": float(1000.0 * np.sqrt(np.mean(np.sum(tangent_position**2, axis=1)))),
        "tangent_velocity_error_rms_m_s": float(
            np.sqrt(np.mean(np.sum(tangent_velocity**2, axis=1)))
        ),
        "orientation_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(trace["orientation_error_rad"][mask] ** 2)))
        ),
        "saturation_pct": float(100.0 * np.mean(saturated)),
    }


def _recovery_metrics(trace: dict[str, np.ndarray], case: ProtocolCase, normal: np.ndarray) -> dict:
    if case.recovery_start_s is None:
        return {
            "force_recovery_status": "not_applicable",
            "force_recovered": None,
            "force_recovery_time_s": None,
            "tangent_recovery_status": "not_applicable",
            "tangent_recovered": None,
            "tangent_recovery_time_s": None,
        }
    time = trace["time"]
    force_error = trace["true_normal_force"] - trace["target_normal_force"]
    position_error = trace["position"] - trace["target_position"]
    tangent_error = position_error - np.outer(position_error @ normal, normal)
    window_steps = max(1, round(RECOVERY_WINDOW_S / case.config.timestep))
    candidates = np.flatnonzero(time >= case.recovery_start_s)
    force_time = None
    tangent_time = None
    for start in candidates:
        stop = start + window_steps
        if stop > len(time):
            break
        elapsed = float(time[start] - case.recovery_start_s)
        if force_time is None and np.sqrt(np.mean(force_error[start:stop] ** 2)) <= (
            RECOVERY_ABSOLUTE_FORCE_ERROR_N
        ):
            force_time = elapsed
        if tangent_time is None and 1000.0 * np.sqrt(
            np.mean(np.sum(tangent_error[start:stop] ** 2, axis=1))
        ) <= RECOVERY_TANGENT_RMSE_MM:
            tangent_time = elapsed
        if force_time is not None and tangent_time is not None:
            break
    return {
        "force_recovery_status": "recovered" if force_time is not None else "not_recovered",
        "force_recovered": force_time is not None,
        "force_recovery_time_s": force_time,
        "tangent_recovery_status": "recovered"
        if tangent_time is not None
        else "not_recovered",
        "tangent_recovered": tangent_time is not None,
        "tangent_recovery_time_s": tangent_time,
    }


def summarize_trial(result, case: ProtocolCase) -> tuple[dict, list[dict]]:
    """Compute fixed overall, phase, worst-phase and recovery diagnostics."""
    trace = result.trace
    time = np.asarray(trace["time"])
    if time.ndim != 1 or len(time) == 0 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("trace time must be a finite increasing vector")
    normal = yaw_frame(case.scenario.wall_yaw_deg).rotation[:, 0]
    evaluation = time >= case.config.evaluation_start
    if not np.any(evaluation):
        raise ValueError("trial does not observe the evaluation interval")
    phase_rows = []
    for phase in case.phases:
        mask = (time >= phase.start_s) & (time < phase.end_s)
        if not np.any(mask):
            raise ValueError(f"phase has no samples: {phase.name}")
        phase_rows.append({"phase": phase.name, **_window_metrics(trace, mask, normal)})
    overall = _window_metrics(trace, evaluation, normal)
    full_peak = float(np.max(trace["true_normal_force"]))
    full_saturation = float(
        100.0
        * np.mean(
            np.any(
                np.abs(trace["commanded_torque"] - trace["applied_torque"]) > 1e-9,
                axis=1,
            )
        )
    )
    coefficient = np.asarray(trace["controller_coefficient_after_compute"], dtype=float)
    dt = case.config.timestep
    rates = np.diff(coefficient) / dt
    ready = np.asarray(trace["controller_update_ready_after_compute"], dtype=bool)
    requested = np.asarray(trace["requested_tangential_force_world"], dtype=float)
    recovery = _recovery_metrics(trace, case, normal)
    worst_force = max(phase_rows, key=lambda row: row["force_rmse_n"])
    worst_tangent = max(phase_rows, key=lambda row: row["tangent_rmse_mm"])
    worst_orientation = max(phase_rows, key=lambda row: row["orientation_rmse_deg"])
    online_config = TangentialCompensation("online")
    overall.update(
        peak_force_n=full_peak,
        saturation_pct=full_saturation,
        worst_force_phase=worst_force["phase"],
        worst_phase_force_rmse_n=worst_force["force_rmse_n"],
        worst_tangent_phase=worst_tangent["phase"],
        worst_phase_tangent_rmse_mm=worst_tangent["tangent_rmse_mm"],
        worst_orientation_phase=worst_orientation["phase"],
        worst_phase_orientation_rmse_deg=worst_orientation["orientation_rmse_deg"],
        **recovery,
        tangential_update_ready_pct=float(100.0 * np.mean(ready[evaluation])),
        coefficient_min=float(np.min(coefficient)),
        coefficient_max=float(np.max(coefficient)),
        max_requested_tangential_force_n=float(np.max(np.linalg.norm(requested, axis=1))),
    )
    overall["compensation_limit_observed"] = bool(
        overall["coefficient_max"] >= online_config.max_equivalent_mu - 1e-12
        or overall["max_requested_tangential_force_n"] >= online_config.max_force - 1e-12
    )
    if (
        overall["tangent_recovery_status"] == "not_recovered"
        and overall["compensation_limit_observed"]
    ):
        overall["tangent_recovery_status"] = "not_recovered_with_limit_active"
    blend = np.asarray(trace.get("contact_blend", np.ones(len(time))), dtype=float)
    measured_force = np.asarray(trace["measured_normal_force"], dtype=float)
    active = (measured_force > 1.0) & (trace["target_normal_force"] > 0.0)
    force_normal_transition = (
        (blend[1:] >= 0.99) & (blend[:-1] >= 0.99) & active[1:] & active[:-1]
    )
    requested_delta = np.linalg.norm(np.diff(requested, axis=0), axis=1)
    controller_kind = str(np.asarray(trace.get("controller_kind", "")).item())
    online_trace = controller_kind == "surface_online" or np.ptp(coefficient) > 1e-12
    coefficient_reset = (
        np.abs(rates) > online_config.coefficient_rate_limit + 1e-10
        if online_trace
        else np.zeros_like(rates, dtype=bool)
    )
    force_reset = (
        requested_delta > online_config.force_slew_rate * dt + 1e-10
        if online_trace
        else np.zeros_like(requested_delta, dtype=bool)
    )
    coefficient_normal_transition = (
        force_normal_transition & ready[1:] & ~coefficient_reset
    )
    force_slew_transition = force_normal_transition & ~force_reset
    overall["max_normal_operation_coefficient_rate_s"] = (
        float(np.max(np.abs(rates[coefficient_normal_transition])))
        if np.any(coefficient_normal_transition)
        else 0.0
    )
    overall["max_coefficient_reset_jump"] = (
        float(np.max(np.abs(np.diff(coefficient)[coefficient_reset])))
        if np.any(coefficient_reset)
        else 0.0
    )
    overall["max_normal_operation_force_slew_n_s"] = (
        float(np.max(requested_delta[force_slew_transition] / dt))
        if np.any(force_slew_transition)
        else 0.0
    )
    overall["max_force_reset_jump_n"] = (
        float(np.max(requested_delta[force_reset]))
        if np.any(force_reset)
        else 0.0
    )
    return overall, phase_rows


def _gate_failures(row: dict, baseline: dict, friction: dict | None, arm: str) -> tuple[str, ...]:
    failures = [
        label
        for passed, label in (
            (row["contact_ratio_pct"] >= GATES["minimum_contact_ratio_pct"], "contact_ratio"),
            (row["peak_force_n"] <= GATES["maximum_raw_peak_force_n"], "raw_peak_force"),
            (row["saturation_pct"] == GATES["required_saturation_pct"], "saturation"),
            (
                row["force_rmse_n"] - baseline["force_rmse_n"]
                <= GATES["maximum_paired_force_rmse_increase_n"],
                "paired_force_rmse",
            ),
            (
                row["orientation_rmse_deg"] - baseline["orientation_rmse_deg"]
                <= GATES["maximum_paired_orientation_rmse_increase_deg"],
                "paired_orientation_rmse",
            ),
        )
        if not passed
    ]
    if arm == "online":
        if friction is None:
            raise ValueError("online gates require the fixed-friction comparator")
        if row["force_rmse_n"] - friction["force_rmse_n"] > GATES[
            "maximum_paired_force_rmse_increase_n"
        ]:
            failures.append("paired_friction_force_rmse")
        if row["orientation_rmse_deg"] - friction["orientation_rmse_deg"] > GATES[
            "maximum_paired_orientation_rmse_increase_deg"
        ]:
            failures.append("paired_friction_orientation_rmse")
    return tuple(failures)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    files = sorted(set(package.rglob("*.py")) | set((package / "assets").rglob("*")))
    return {str(path.relative_to(package)): _sha256(path) for path in files if path.is_file()}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _case_document(case: ProtocolCase) -> dict:
    return {
        "name": case.name,
        "scenario": asdict(case.scenario),
        "config": asdict(case.config),
        "task": {
            "yaw_deg": case.task.yaw_deg,
            "nominal_plane_x_m": case.task.nominal_plane_x_m,
            "trajectory": case.trajectory.name,
        },
        "friction_schedule": asdict(case.friction),
        "additional_world_wrench_bias_schedule": asdict(case.wrench_bias_world),
        "controller_yaw_error_deg": case.controller_yaw_error_deg,
        "phases": [asdict(phase) for phase in case.phases],
        "recovery_start_s": case.recovery_start_s,
    }


def _constructor_config(value):
    if is_dataclass(value):
        return {
            field.name: _constructor_config(getattr(value, field.name))
            for field in fields(value)
            if field.init
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _controller_configs(rotation_gain_scale: float = 1.0) -> dict:
    frame = yaw_frame(15.0)
    configs = {}
    for arm, mode in ARMS.items():
        controller = SurfaceAdaptiveController(
            frame, tangential_mode=mode, rotation_gain_scale=rotation_gain_scale,
        )
        configs[arm] = {
            "surface_frame_rotation": frame.rotation.tolist(),
            "safe_adaptive_base": _constructor_config(controller._base),
        }
    return configs


def _compact_trace(trace: dict[str, np.ndarray], case: ProtocolCase) -> dict[str, np.ndarray]:
    normal = yaw_frame(case.scenario.wall_yaw_deg).rotation[:, 0]
    position_error = trace["position"] - trace["target_position"]
    velocity_error = trace["linear_velocity"] - trace["target_linear_velocity"]
    tangent_position = position_error - np.outer(position_error @ normal, normal)
    tangent_velocity = velocity_error - np.outer(velocity_error @ normal, normal)
    compact = {
        name: trace[name]
        for name in (
            "time",
            "true_normal_force",
            "target_normal_force",
            "measured_normal_force",
            "orientation_error_rad",
            "torque_projection_scale",
            "contact_blend",
            "controller_coefficient_before_compute",
            "controller_coefficient_after_compute",
            "controller_update_ready_before_compute",
            "controller_update_ready_after_compute",
            "applied_wall_friction",
            "applied_tool_friction",
            "applied_raw_wrench_bias_world",
            "feedback_raw_wrench_bias_world",
            "controller_yaw_error_deg",
            "trajectory_rate_scale",
        )
    }
    compact.update(
        tangent_position_error_world=tangent_position,
        tangent_position_error_norm_m=np.linalg.norm(tangent_position, axis=1),
        tangent_velocity_error_world=tangent_velocity,
        tangent_velocity_error_norm_m_s=np.linalg.norm(tangent_velocity, axis=1),
        actuator_clipped=np.any(
            np.abs(trace["commanded_torque"] - trace["applied_torque"]) > 1e-9, axis=1
        ),
        requested_tangential_force_norm_n=np.linalg.norm(
            trace["requested_tangential_force_world"], axis=1
        ),
        requested_tangential_force_delta_norm_n=np.concatenate(
            (
                np.zeros(1),
                np.linalg.norm(np.diff(trace["requested_tangential_force_world"], axis=0), axis=1),
            )
        ),
    )
    return compact


def _save_trace(path: Path, trace: dict[str, np.ndarray]) -> None:
    required = {
        "time",
        "applied_wall_friction",
        "applied_tool_friction",
        "applied_raw_wrench_bias_world",
        "feedback_raw_wrench_bias_world",
        "controller_coefficient_before_compute",
        "controller_coefficient_after_compute",
        "controller_update_ready_after_compute",
        "trajectory_rate_scale",
    }
    if required - trace.keys():
        raise ValueError(f"trace missing protocol fields: {sorted(required - trace.keys())}")
    payload = {name: np.asarray(value) for name, value in trace.items()}
    for name, value in payload.items():
        if value.dtype.kind not in "biufUS" or (value.dtype.kind in "biuf" and not np.all(np.isfinite(value))):
            raise ValueError(f"trace field is not finite numeric/string data: {name}")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **payload)


def _select(values, available: tuple[str, ...], label: str) -> list[str]:
    selected = list(available) if values is None else list(values)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"{label} must be nonempty and unique")
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"unknown {label}: {sorted(unknown)}")
    return selected


def generate_online_compensation_experiment(
    output_dir: Path | str,
    *,
    case_names=None,
    arms=None,
    seeds=None,
    full_traces: bool = False,
    rotation_gain_scale: float = 1.0,
) -> Path:
    """Run a selected fixed matrix and atomically publish complete evidence."""
    output = Path(output_dir).absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("output path must not contain symlinks")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    controller_configs = _controller_configs(rotation_gain_scale)
    rotation_gain_scale = float(rotation_gain_scale)
    cases = protocol_cases()
    profile_names = tuple(dict.fromkeys(case.name for case in cases))
    selected_names = _select(case_names, profile_names, "case names")
    selected_arms = _select(arms, tuple(ARMS), "arms")
    if selected_arms != ["baseline"] and "baseline" not in selected_arms:
        raise ValueError("non-baseline arms require the paired baseline arm")
    if "online" in selected_arms and "friction" not in selected_arms:
        raise ValueError("the online arm requires the fixed-friction comparator")
    selected_arms = [arm for arm in ARMS if arm in selected_arms]
    raw_seeds = PROTOCOL_SEEDS if seeds is None else tuple(seeds)
    if (
        not raw_seeds
        or any(
            isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            for seed in raw_seeds
        )
    ):
        raise ValueError(f"seeds must be a nonempty unique subset of {PROTOCOL_SEEDS}")
    selected_seeds = tuple(int(seed) for seed in raw_seeds)
    if len(set(selected_seeds)) != len(selected_seeds) or any(
        seed not in PROTOCOL_SEEDS for seed in selected_seeds
    ):
        raise ValueError(f"seeds must be a nonempty unique subset of {PROTOCOL_SEEDS}")
    selected_cases = [
        case
        for name in selected_names
        for case in cases
        if case.name == name and case.config.seed in selected_seeds
    ]

    # These comparison definitions and their digest exist before the first trial.
    protocol = {
        "identity": PROTOCOL_ID,
        "public_development": True,
        "new_holdout": False,
        "comparison_defined_before_execution": True,
        "representative_original_case_index": REPRESENTATIVE_CASE_INDEX,
        "predetermined_noise_seeds": PROTOCOL_SEEDS,
        "arms": ARMS,
        "controller_constructor_configurations": controller_configs,
        "rotation_gain_scale": rotation_gain_scale,
        "gates": GATES,
        "recovery": {
            "window_s": RECOVERY_WINDOW_S,
            "absolute_force_error_n": RECOVERY_ABSOLUTE_FORCE_ERROR_N,
            "tangent_rmse_mm": RECOVERY_TANGENT_RMSE_MM,
        },
        "metrics_note": "Tangent velocity-error RMS is a time-domain diagnostic, not spectral or stability evidence.",
        "limitations": [
            "Fixed known fixture and surface frame only; no unknown or dynamic surface claim.",
            "The online coefficient is error-driven and is not identified material friction.",
            "Simulation and command bounds do not establish passivity or hardware safety.",
            "All failures are retained; no case or seed is selected from observed results.",
            "Seeds 11 and 29 vary sensor/position noise on one fixed geometry; they are not independent physical surface groups.",
        ],
        "all_cases": [_case_document(case) for case in cases],
    }
    configurations = {
        "rotation_gain_scale": rotation_gain_scale,
        "selected_case_names": selected_names,
        "selected_seeds": selected_seeds,
        "selected_arms": selected_arms,
        "is_subset": len(selected_names) != len(profile_names)
        or len(selected_seeds) != len(PROTOCOL_SEEDS)
        or len(selected_arms) != len(ARMS),
        "trace_retention": {
            "all_runs": "compact metric-recomputable NPZ",
            "default_full_trace_selection": sorted(FULL_TRACE_SELECTION),
            "full_traces_override": bool(full_traces),
        },
        "executions": [
            {"case": _case_document(case), "arm": arm, "paired_seed": case.config.seed}
            for case in selected_cases
            for arm in selected_arms
        ],
    }
    source_hashes = _source_hashes()

    rows: list[dict] = []
    phase_rows: list[dict] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".online-compensation-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        traces = staging / "traces"
        traces.mkdir()
        _write_json(staging / "protocol.json", protocol)
        _write_json(staging / "configurations.json", configurations)
        _write_json(staging / "source_hashes.json", source_hashes)
        for case in selected_cases:
            case_summaries = {}
            case_phases = {}
            for arm in selected_arms:
                result = run_protocol_trial(case, arm, rotation_gain_scale=rotation_gain_scale)
                summary, phases = summarize_trial(result, case)
                case_summaries[arm] = summary
                case_phases[arm] = phases
                stem = f"{case.name}__seed_{case.config.seed}__{arm}"
                _save_trace(traces / f"{stem}__compact.npz", _compact_trace(result.trace, case))
                if full_traces or (case.name, case.config.seed, arm) in FULL_TRACE_SELECTION:
                    _save_trace(traces / f"{stem}__full.npz", result.trace)
            baseline = case_summaries["baseline"]
            friction = case_summaries.get("friction")
            for arm in selected_arms:
                summary = case_summaries[arm]
                failures = _gate_failures(summary, baseline, friction, arm)
                rows.append(
                    {
                        "case": case.name,
                        "arm": arm,
                        "simulation_seed": case.config.seed,
                        "paired_baseline_seed": case.config.seed,
                        "controller_yaw_error_deg": case.controller_yaw_error_deg,
                        **summary,
                        "paired_force_rmse_increase_n": summary["force_rmse_n"]
                        - baseline["force_rmse_n"],
                        "paired_orientation_rmse_increase_deg": summary["orientation_rmse_deg"]
                        - baseline["orientation_rmse_deg"],
                        "paired_friction_force_rmse_increase_n": None
                        if friction is None
                        else summary["force_rmse_n"] - friction["force_rmse_n"],
                        "paired_friction_orientation_rmse_increase_deg": None
                        if friction is None
                        else summary["orientation_rmse_deg"] - friction["orientation_rmse_deg"],
                        "failed_gates": ";".join(failures),
                        "all_gates_pass": "yes" if not failures else "no",
                    }
                )
                baseline_phases = {phase["phase"]: phase for phase in case_phases["baseline"]}
                friction_phases = (
                    {phase["phase"]: phase for phase in case_phases["friction"]}
                    if friction is not None
                    else {}
                )
                for phase in case_phases[arm]:
                    reference = baseline_phases[phase["phase"]]
                    friction_reference = friction_phases.get(phase["phase"])
                    phase_failures = _gate_failures(phase, reference, friction_reference, arm)
                    phase_rows.append(
                        {
                            "case": case.name,
                            "simulation_seed": case.config.seed,
                            "arm": arm,
                            **phase,
                            "paired_baseline_force_rmse_increase_n": phase["force_rmse_n"]
                            - reference["force_rmse_n"],
                            "paired_baseline_orientation_rmse_increase_deg": phase[
                                "orientation_rmse_deg"
                            ]
                            - reference["orientation_rmse_deg"],
                            "paired_friction_force_rmse_increase_n": None
                            if friction_reference is None
                            else phase["force_rmse_n"] - friction_reference["force_rmse_n"],
                            "paired_friction_orientation_rmse_increase_deg": None
                            if friction_reference is None
                            else phase["orientation_rmse_deg"]
                            - friction_reference["orientation_rmse_deg"],
                            "failed_gates": ";".join(phase_failures),
                            "all_gates_pass": "yes" if not phase_failures else "no",
                        }
                    )
        if _source_hashes() != source_hashes:
            raise ValueError("source/assets changed during execution; refusing mixed-version output")
        _write_csv(staging / "comparison.csv", rows)
        _write_csv(staging / "phase_metrics.csv", phase_rows)
        input_names = ("protocol.json", "configurations.json", "source_hashes.json")
        manifest = {
            "schema_version": 1,
            "experiment_identity": PROTOCOL_ID,
            "rotation_gain_scale": rotation_gain_scale,
            "new_holdout": False,
            "is_subset": configurations["is_subset"],
            "selected_case_names": selected_names,
            "selected_seeds": selected_seeds,
            "selected_arms": selected_arms,
            "versions": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "mujoco", "compliant-control-lab")},
            },
            "input_sha256": {name: _sha256(staging / name) for name in input_names},
            "artifact_sha256": {
                str(path.relative_to(staging)): _sha256(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            },
        }
        _write_json(staging / "manifest.json", manifest)
        (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"output appeared during execution: {output}")
        os.rename(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-names", nargs="+")
    parser.add_argument("--arms", choices=tuple(ARMS), nargs="+")
    parser.add_argument("--seeds", choices=PROTOCOL_SEEDS, type=int, nargs="+")
    parser.add_argument("--full-traces", action="store_true")
    parser.add_argument("--rotation-gain-scale", type=float, default=1.0,
                        help="Constant Krot multiplier; Drot is multiplied by its square root.")
    arguments = parser.parse_args()
    output = generate_online_compensation_experiment(
        arguments.output,
        case_names=arguments.case_names,
        arms=arguments.arms,
        seeds=arguments.seeds,
        full_traces=arguments.full_traces,
        rotation_gain_scale=arguments.rotation_gain_scale,
    )
    print(f"Public-development error comparison complete: {output}")


if __name__ == "__main__":
    main()
