"""Independent checks for the frozen 6/8 N compensation-budget study."""

import numpy as np

from compliant_control_lab.surface_simulation import yaw_frame

FORCE_BUDGETS_N = (6.0, 8.0)


def close(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"invalid shape/nonfinite: {label}")
    if not np.allclose(a, b, rtol=0, atol=1e-10):
        raise ValueError(f"numeric mismatch: {label}")


def _force_budget(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a predeclared numeric force budget")
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iuf" or not np.isfinite(array):
        raise ValueError(f"{label} must be a finite scalar")
    budget = float(array)
    if budget not in FORCE_BUDGETS_N:
        raise ValueError(f"{label} must be one of {FORCE_BUDGETS_N}")
    return budget


def validate_trace(trace, case, max_force_n):
    """Check raw telemetry against one predeclared force budget.

    This validates schedules and the recorded command path. It does not attest that
    the dynamics were executed.
    """
    budget = _force_budget(max_force_n, "expected max_force_n")
    if "max_force_n" not in trace:
        raise ValueError("trace missing max_force_n")
    recorded_budget = _force_budget(trace["max_force_n"], "trace max_force_n")
    if recorded_budget != budget:
        raise ValueError("trace max_force_n differs from expected budget")

    dt = case.config.timestep
    count = round(case.config.duration / dt)
    time = np.arange(count) * dt
    close(trace["time"], time, "time grid")
    for name, values in trace.items():
        values = np.asarray(values)
        if values.dtype.kind not in "biufUS":
            raise ValueError(f"unsupported array: {name}")
        if values.dtype.kind in "biuf" and not np.all(np.isfinite(values)):
            raise ValueError(f"nonfinite trace: {name}")
    for name, expected in (("rotation_gain_scale", 1.0), ("controller_kind", "surface_online")):
        if np.asarray(trace[name]).shape != () or np.asarray(trace[name]).item() != expected:
            raise ValueError(f"wrong controller identity: {name}")
    for name in ("applied_wall_friction", "applied_tool_friction"):
        close(trace[name], np.array([case.friction.at(t) for t in time]), name)
    bias = np.array([case.wrench_bias_world.at(t) for t in time])
    close(trace["applied_raw_wrench_bias_world"], bias, "applied bias schedule")
    close(
        trace["feedback_raw_wrench_bias_world"],
        np.concatenate((bias[:1], bias[:-1])),
        "one-cycle feedback bias",
    )
    close(
        trace["controller_yaw_error_deg"],
        np.full(count, case.controller_yaw_error_deg),
        "controller yaw error",
    )
    close(
        trace["trajectory_rate_scale"],
        np.array([case.trajectory.rate_scale_at(t) for t in time]),
        "trajectory schedule",
    )
    limits = np.broadcast_to([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], (count, 7))
    close(trace["lower_torque_limit"], -limits, "fixed lower torque limits")
    close(trace["upper_torque_limit"], limits, "fixed upper torque limits")
    close(
        trace["applied_torque"],
        np.clip(trace["commanded_torque"], -limits, limits),
        "actuator clipping",
    )
    for name in ("contact_blend", "torque_projection_scale"):
        values = np.asarray(trace[name])
        if values.shape != (count,) or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"invalid fraction: {name}")
    flags = (
        "diagnostic_compensation_active",
        "diagnostic_amplitude_capped",
        "diagnostic_slew_limited",
        "controller_update_ready_before_compute",
        "controller_update_ready_after_compute",
    )
    for name in flags:
        values = np.asarray(trace[name])
        if values.shape != (count,) or values.dtype.kind != "b":
            raise ValueError(f"invalid boolean telemetry: {name}")
    before = np.asarray(trace["controller_coefficient_before_compute"])
    after = np.asarray(trace["controller_coefficient_after_compute"])
    close(before[1:], after[:-1], "coefficient cycle alignment")
    if before.shape != (count,) or after.shape != (count,):
        raise ValueError("invalid coefficient shape")
    if np.any(before < 0) or np.any(before > 0.9) or np.any(after < 0) or np.any(after > 0.9):
        raise ValueError("coefficient exceeds frozen bound")
    force = np.asarray(trace["diagnostic_corrected_force_n"])
    active = trace["diagnostic_compensation_active"]
    if force.shape != (count,) or np.any(force < 0):
        raise ValueError("invalid corrected force")
    if np.any(active & ((force <= 1) | (trace["target_normal_force"] <= 0))):
        raise ValueError("invalid active compensation state")
    cap = active & (before * force >= budget)
    if not np.array_equal(trace["diagnostic_amplitude_capped"], cap):
        raise ValueError("amplitude cap flag disagrees with force budget")
    normal = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg).rotation[:, 0]
    velocity = np.asarray(trace["target_linear_velocity"])
    tangent_velocity = velocity - np.outer(velocity @ normal, normal)
    speed = np.linalg.norm(tangent_velocity, axis=1)
    direction = tangent_velocity / np.sqrt(speed[:, None] ** 2 + 0.005**2)
    desired = (trace["contact_blend"] * np.minimum(before * force, budget))[:, None] * direction
    requested = np.asarray(trace["requested_tangential_force_world"])
    previous = np.concatenate((np.zeros((1, 3)), requested[:-1]))
    previous -= np.outer(previous @ normal, normal)
    delta = desired - previous
    distance = np.linalg.norm(delta, axis=1)
    limited = distance > 20.0 * dt
    certain = np.abs(distance - 20.0 * dt) > 1e-10
    if not np.array_equal(trace["diagnostic_slew_limited"][certain], (active & limited)[certain]):
        raise ValueError("slew observer flag disagrees with bounded request")
    expected = previous + delta * np.minimum(1.0, 20.0 * dt / np.maximum(distance, 1e-12))[:, None]
    expected[~active] = 0.0
    close(requested, expected, "observed compensation request")
