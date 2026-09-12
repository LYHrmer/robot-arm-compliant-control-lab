"""Full-trace trials for auxiliary measured-load packet faults.

The injected packet is used only by ``LoadAwareCompensation``.  The controller's
existing normal-force feedback continues to use the simulator measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from compliant_control_lab.online_compensation_experiment import ScheduledSurfaceSimulator
from compliant_control_lab.surface_simulation import yaw_frame
from tools import budget_gain_interaction as dynamic
from tools import combined_residual_ablation as old
from tools import load_budget_trial as base
from tools.load_budget_inputs import measured_force_input

MAX_MEASUREMENT_AGE_S = 0.020
METHOD_BOUNDS = {
    "adaptive6_8": (6.0, 8.0),
    "fixed6": (6.0, 6.0),
    "fixed8": (8.0, 8.0),
}


class PacketStatus(IntEnum):
    MISSING = 0
    ACCEPTED = 1
    STALE = 2
    FUTURE = 3
    REORDERED = 4
    NONFINITE = 5


@dataclass(frozen=True)
class RawLoadPacket:
    present: bool
    force_local: np.ndarray
    stamp_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.present, (bool, np.bool_)):
            raise TypeError("packet present flag must be boolean")
        force = np.asarray(self.force_local, dtype=float)
        if force.shape != (3,):
            raise ValueError("packet force must be a 3-vector")
        force = force.copy()
        force.setflags(write=False)
        object.__setattr__(self, "present", bool(self.present))
        object.__setattr__(self, "force_local", force)
        object.__setattr__(self, "stamp_s", float(self.stamp_s))


@dataclass(frozen=True)
class PacketDecision:
    status: PacketStatus
    available: bool
    force_local: np.ndarray
    stamp_s: float
    age_s: float


class LoadPacketGate:
    """Accept fresh, finite, non-reordered packets without inventing samples."""

    def __init__(self, max_age_s: float = MAX_MEASUREMENT_AGE_S) -> None:
        if not np.isfinite(max_age_s) or max_age_s < 0.0:
            raise ValueError("maximum measurement age must be finite and nonnegative")
        self.max_age_s = float(max_age_s)
        self.last_accepted_stamp_s: float | None = None

    def evaluate(self, packet: RawLoadPacket, sample_time_s: float) -> PacketDecision:
        sample_time_s = float(sample_time_s)
        if not np.isfinite(sample_time_s):
            raise ValueError("sample time must be finite")
        if not packet.present:
            status = PacketStatus.MISSING
        elif not np.isfinite(packet.stamp_s) or not np.all(np.isfinite(packet.force_local)):
            status = PacketStatus.NONFINITE
        else:
            age_s = sample_time_s - packet.stamp_s
            if age_s < 0.0:
                status = PacketStatus.FUTURE
            elif packet.stamp_s < 0.0 or age_s > self.max_age_s + 1e-12:
                status = PacketStatus.STALE
            elif (
                self.last_accepted_stamp_s is not None
                and packet.stamp_s < self.last_accepted_stamp_s
            ):
                status = PacketStatus.REORDERED
            else:
                status = PacketStatus.ACCEPTED
                self.last_accepted_stamp_s = packet.stamp_s
                return PacketDecision(
                    status, True, packet.force_local.copy(), packet.stamp_s, age_s
                )
        return PacketDecision(status, False, np.zeros(3), 0.0, 0.0)


@dataclass(frozen=True)
class FaultConfig:
    name: str
    kind: str
    start_s: float
    end_s: float
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"fresh", "bias", "scale", "missing", "stale"}:
            raise ValueError(f"unknown measurement fault kind: {self.kind}")
        values = np.asarray([self.start_s, self.end_s, self.value], dtype=float)
        if not self.name or not np.all(np.isfinite(values)) or self.end_s < self.start_s:
            raise ValueError("fault configuration must have a name and finite ordered bounds")
        if self.kind == "scale" and self.value <= 0.0:
            raise ValueError("measurement scale must be positive")


FAULTS = {
    "fresh": FaultConfig("fresh", "fresh", 0.0, 0.0),
    "bias_plus": FaultConfig("bias_plus", "bias", 6.0, 12.0, 0.75),
    "bias_minus": FaultConfig("bias_minus", "bias", 6.0, 12.0, -0.75),
    "scale_0p8": FaultConfig("scale_0p8", "scale", 6.0, 12.0, 0.8),
    "scale_1p2": FaultConfig("scale_1p2", "scale", 6.0, 12.0, 1.2),
    "missing": FaultConfig("missing", "missing", 8.0, 8.3),
    "stale": FaultConfig("stale", "stale", 8.0, 8.3),
}


def fault_config(value: str | FaultConfig) -> FaultConfig:
    if isinstance(value, FaultConfig):
        return value
    try:
        return FAULTS[value]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown measurement fault: {value!r}") from error


class DeterministicFaultAdapter:
    """Transform the auxiliary packet while retaining its explicit provenance."""

    def __init__(self, config: str | FaultConfig) -> None:
        self.config = fault_config(config)
        self._last_source: RawLoadPacket | None = None

    def apply(self, source: RawLoadPacket, sample_time_s: float, target, frame) -> RawLoadPacket:
        config = self.config
        active = config.start_s <= sample_time_s < config.end_s
        if config.kind == "stale" and active:
            if self._last_source is None:
                return RawLoadPacket(False, np.zeros(3), 0.0)
            return self._last_source

        self._last_source = source
        if config.kind == "fresh" or not active:
            return source
        if config.kind == "missing":
            return RawLoadPacket(False, np.zeros(3), 0.0)

        force = source.force_local.copy()
        if config.kind == "scale":
            force[1:] *= config.value
        else:
            velocity_local = frame.vector_to_local(target.linear_velocity)
            tangent = velocity_local[1:]
            speed = float(np.linalg.norm(tangent))
            if speed > 0.0:
                force[1:] += config.value * tangent / speed
        return RawLoadPacket(source.present, force, source.stamp_s)


def _source_packet(input_row, sample, frame) -> RawLoadPacket:
    measurement = measured_force_input(input_row, sample, frame)
    return RawLoadPacket(True, measurement["force_local"], measurement["measurement_time_s"])


def _run_loop(case, controller, simulator, frame, fault: str | FaultConfig):
    compensation = controller._base.tangential
    observer = base.LoadBudgetObserver(controller)
    adapter = DeterministicFaultAdapter(fault)
    gate = LoadPacketGate()
    rows = {name: [] for name in (
        *base.SCHEDULER_FIELDS,
        *old.DIAGNOSTIC_FIELDS,
        "raw_load_packet_present",
        "raw_load_packet_force",
        "raw_load_packet_stamp",
        "load_packet_status",
        "load_measurement_available",
        "measured_in_contact",
    )}
    coefficient_before, coefficient_after = [], []
    dt, duration = case.config.timestep, case.config.duration
    for step in range(round(duration / dt)):
        sample = simulator.sample()
        if step == 0:
            controller.reset(sample.state)
        raw_packet = adapter.apply(
            _source_packet(simulator._input_row, sample, frame), sample.time, sample.target, frame
        )
        decision = gate.evaluate(raw_packet, sample.time)
        if decision.available:
            compensation.set_force_measurement(decision.force_local)
        before = (
            controller.equivalent_tangential_coefficient,
            controller.tangential_update_ready,
        )
        snapshot = observer.before_compute(sample.target, dt)
        projection_before = controller.torque_projection_fallback_count
        wrench = controller.compute(sample.state, sample.target, dt)
        diagnostics = observer.after_compute(snapshot)
        after = controller.equivalent_tangential_coefficient
        simulator.step(wrench, controller, telemetry_before=before)
        values = {
            "load_budget_applied_n": compensation.applied_budget_n,
            "load_budget_next_n": compensation.next_budget_n,
            "load_estimate_n": compensation.load_estimate_n,
            "load_budget_updated": compensation.measurement_used_for_budget,
            "load_projected_n": compensation.projected_load_n,
            "load_force_local": decision.force_local,
            "load_measurement_time_s": decision.stamp_s,
            "load_measurement_age_s": decision.age_s,
            "load_compensation_force_local": compensation.last_force,
            "load_projection_accepted": bool(
                sample.state.actuation is not None
                and controller.last_torque_projection_scale == 1.0
                and controller.torque_projection_fallback_count == projection_before
            ),
            "controller_update_ready_before_compute": before[1],
            "controller_update_ready_after_compute": controller.tangential_update_ready,
            "raw_load_packet_present": raw_packet.present,
            "raw_load_packet_force": raw_packet.force_local,
            "raw_load_packet_stamp": raw_packet.stamp_s,
            "load_packet_status": decision.status,
            "load_measurement_available": decision.available,
            "measured_in_contact": controller._base.base.base.in_contact,
            **diagnostics,
        }
        for name, value in values.items():
            rows[name].append(value)
        coefficient_before.append(before[0])
        coefficient_after.append(after)

    result = simulator.result()
    result.trace.update({name: np.asarray(values) for name, values in rows.items()})
    result.trace["load_packet_status"] = np.asarray(rows["load_packet_status"], dtype=np.uint8)
    result.trace["controller_coefficient_before_compute"] = np.asarray(coefficient_before)
    result.trace["controller_coefficient_after_compute"] = np.asarray(coefficient_after)
    result.trace[old.AUDIT_FIELDS[0]] = np.array(observer.max_reconstruction_error_n)
    result.trace[old.AUDIT_FIELDS[1]] = np.array(observer.mismatch_cycles)
    return result


def run_dynamic(case, scale, method: str, fault: str | FaultConfig):
    try:
        minimum_force, maximum_force = METHOD_BOUNDS[method]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown load-budget method: {method!r}") from error
    config = fault_config(fault)
    frame = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg)
    controller = base._load_aware(
        dynamic.controller(case, scale, maximum_force), minimum_force, maximum_force
    )
    simulator = ScheduledSurfaceSimulator(case, frame, old.ARM)
    result = _run_loop(case, controller, simulator, frame, config)
    result.trace.update(
        rotation_gain_scale=np.array(float(scale)),
        minimum_force_n=np.array(minimum_force),
        max_force_n=np.array(maximum_force),
        method=np.array(method),
        fault_name=np.array(config.name),
        fault_kind=np.array(config.kind),
        fault_start_s=np.array(config.start_s),
        fault_end_s=np.array(config.end_s),
        fault_value=np.array(config.value),
        load_max_measurement_age_s=np.array(MAX_MEASUREMENT_AGE_S),
        trace_schema_version=np.array(2),
    )
    return result
