"""Measured-input-only trial adapters for the experimental load-aware ceiling."""

from dataclasses import fields

import numpy as np

from compliant_control_lab.online_compensation_experiment import ScheduledSurfaceSimulator
from compliant_control_lab.surface_simulation import SurfaceFrame, SurfaceSimulator, yaw_frame
from compliant_control_lab.tangential_compensation import TangentialCompensation
from tools import budget_gain_interaction as dynamic
from tools import budget_public24 as public
from tools import combined_residual_ablation as old
from tools.load_aware_compensation import LoadAwareCompensation
from tools.load_budget_inputs import measured_force_input

MINIMUM_FORCE_N = 6.0
MAXIMUM_FORCE_N = 8.0
SCHEDULER_FIELDS = (
    "load_budget_applied_n",
    "load_budget_next_n",
    "load_estimate_n",
    "load_budget_updated",
    "load_projected_n",
    "load_force_local",
    "load_measurement_time_s",
    "load_measurement_age_s",
    "load_compensation_force_local",
    "load_projection_accepted",
    "controller_update_ready_before_compute",
    "controller_update_ready_after_compute",
)
TELEMETRY_FIELDS = (
    *SCHEDULER_FIELDS,
    "contact_blend",
    "controller_coefficient_before_compute",
    "controller_coefficient_after_compute",
    *old.DIAGNOSTIC_FIELDS,
    *old.AUDIT_FIELDS,
)


def _load_aware(controller, minimum_force: float, maximum_force: float):
    original = controller._base.tangential
    if original is None or original.mode != old.ARM:
        raise ValueError("load-aware trial requires the online compensation arm")
    parameters = {
        item.name: getattr(original, item.name)
        for item in fields(TangentialCompensation)
        if item.init
    }
    parameters["max_force"] = float(maximum_force)
    controller._base.tangential = LoadAwareCompensation(
        **parameters, minimum_force=float(minimum_force)
    )
    return controller


class LoadBudgetObserver(old.CompensationObserver):
    """Reconstruct the request against the ceiling actually used this cycle."""

    def after_compute(self, snapshot: dict) -> dict:
        global_ceiling = self._compensation.max_force
        try:
            self._compensation.max_force = self._compensation.applied_budget_n
            return super().after_compute(snapshot)
        finally:
            self._compensation.max_force = global_ceiling


def _run_loop(case, controller, simulator, frame, *, scheduled: bool):
    compensation = controller._base.tangential
    observer = LoadBudgetObserver(controller)
    rows = {name: [] for name in (*SCHEDULER_FIELDS, *old.DIAGNOSTIC_FIELDS)}
    coefficient_before, coefficient_after = [], []
    dt = case.config.timestep if scheduled else case["config"].timestep
    duration = case.config.duration if scheduled else case["config"].duration
    for step in range(round(duration / dt)):
        sample = simulator.sample()
        if step == 0:
            controller.reset(sample.state)
        measurement = measured_force_input(simulator._input_row, sample, frame)
        compensation.set_force_measurement(measurement["force_local"])
        before = (
            controller.equivalent_tangential_coefficient,
            controller.tangential_update_ready,
        )
        snapshot = observer.before_compute(sample.target, dt)
        projection_before = controller.torque_projection_fallback_count
        wrench = controller.compute(sample.state, sample.target, dt)
        diagnostics = observer.after_compute(snapshot)
        after = controller.equivalent_tangential_coefficient
        if scheduled:
            simulator.step(wrench, controller, telemetry_before=before)
        else:
            simulator.step(wrench, controller)
        values = {
            "load_budget_applied_n": compensation.applied_budget_n,
            "load_budget_next_n": compensation.next_budget_n,
            "load_estimate_n": compensation.load_estimate_n,
            "load_budget_updated": compensation.measurement_used_for_budget,
            "load_projected_n": compensation.projected_load_n,
            "load_force_local": measurement["force_local"],
            "load_measurement_time_s": measurement["measurement_time_s"],
            "load_measurement_age_s": measurement["measurement_age_s"],
            "load_compensation_force_local": compensation.last_force,
            "load_projection_accepted": bool(
                sample.state.actuation is not None
                and controller.last_torque_projection_scale == 1.
                and controller.torque_projection_fallback_count == projection_before
            ),
            "controller_update_ready_before_compute": before[1],
            "controller_update_ready_after_compute": controller.tangential_update_ready,
            **diagnostics,
        }
        for name, value in values.items():
            rows[name].append(value)
        coefficient_before.append(before[0])
        coefficient_after.append(after)
    result = simulator.result()
    result.trace.update({name: np.asarray(values) for name, values in rows.items()})
    result.trace["controller_coefficient_before_compute"] = np.asarray(coefficient_before)
    result.trace["controller_coefficient_after_compute"] = np.asarray(coefficient_after)
    result.trace[old.AUDIT_FIELDS[0]] = np.array(observer.max_reconstruction_error_n)
    result.trace[old.AUDIT_FIELDS[1]] = np.array(observer.mismatch_cycles)
    return result


def run_dynamic(case, scale, minimum_force=MINIMUM_FORCE_N, max_force=MAXIMUM_FORCE_N):
    frame = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg)
    controller = _load_aware(
        dynamic.controller(case, scale, max_force), minimum_force, max_force
    )
    simulator = ScheduledSurfaceSimulator(case, frame, old.ARM)
    result = _run_loop(case, controller, simulator, frame, scheduled=True)
    result.trace["rotation_gain_scale"] = np.array(float(scale))
    result.trace["max_force_n"] = np.array(float(max_force))
    return result


def run_public(case, scale, minimum_force=MINIMUM_FORCE_N, max_force=MAXIMUM_FORCE_N):
    angle = np.deg2rad(case["task"].yaw_deg)
    frame = SurfaceFrame.from_normal(np.array([np.cos(angle), np.sin(angle), 0.0]))
    controller = _load_aware(
        public._with_budget(public.rotation.controller(frame, public.METHOD, scale), max_force),
        minimum_force,
        max_force,
    )
    simulator = SurfaceSimulator(
        frame, case["scenario"], case["config"], case["task"], public.rotation.METHODS[public.METHOD]
    )
    result = _run_loop(case, controller, simulator, frame, scheduled=False)
    result.trace.update(
        rotation_gain_scale=np.array(float(scale)),
        case_index=np.array(case["case_index"]),
        method=np.array(public.METHOD),
        max_force_n=np.array(float(max_force)),
    )
    return result


def compact_dynamic(trace):
    compacted = dynamic.compact(trace)
    compacted.update({name: np.asarray(trace[name]) for name in TELEMETRY_FIELDS})
    compacted["measured_wrench_world"] = np.asarray(trace["measured_wrench_world"])
    return compacted


def compact_public(trace):
    compacted = public.compact(trace)
    compacted.update({name: np.asarray(trace[name]) for name in TELEMETRY_FIELDS})
    compacted["measured_wrench_world"] = np.asarray(trace["measured_wrench_world"])
    return compacted
