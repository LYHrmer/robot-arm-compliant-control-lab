"""Independent trace checks for the fixed budget-transfer studies."""

import numpy as np

import tools.combined_residual_ablation as combined
import tools.rotation_gain_regression as rotation
from compliant_control_lab.surface_simulation import yaw_frame

BUDGETS_N = (6.0, 8.0)
ROTATION_GAIN_SCALES = (1.0, 2.0)


def _choice(value, choices, label):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a predeclared numeric value")
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iuf" or not np.isfinite(array):
        raise ValueError(f"{label} must be a finite scalar")
    number = float(array)
    if number not in choices:
        raise ValueError(f"{label} must be one of {choices}")
    return number


def _close(actual, expected, label, *, atol=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.shape != expected.shape
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(expected))
    ):
        raise ValueError(f"invalid shape/nonfinite: {label}")
    if not np.allclose(actual, expected, rtol=0, atol=atol):
        raise ValueError(f"numeric mismatch: {label}")


def _metadata(trace, case_index, scale, budget):
    expected_scale = _choice(scale, ROTATION_GAIN_SCALES, "expected rotation gain scale")
    expected_budget = _choice(budget, BUDGETS_N, "expected max_force_n")

    recorded_scale = _choice(
        trace["rotation_gain_scale"], ROTATION_GAIN_SCALES, "trace rotation_gain_scale"
    )
    recorded_budget = _choice(trace["max_force_n"], BUDGETS_N, "trace max_force_n")
    if recorded_scale != expected_scale:
        raise ValueError("trace rotation_gain_scale differs from expected scale")
    if recorded_budget != expected_budget:
        raise ValueError("trace max_force_n differs from expected budget")

    recorded_index = np.asarray(trace["case_index"])
    if (
        recorded_index.shape != ()
        or recorded_index.dtype.kind not in "iu"
        or isinstance(recorded_index.item(), (bool, np.bool_))
        or int(recorded_index) != case_index
    ):
        raise ValueError("trace case_index differs from expected case")
    method = np.asarray(trace["method"])
    if method.shape != () or method.dtype.kind not in "US" or method.item() != "online":
        raise ValueError("trace method must be scalar 'online'")
    return expected_scale, expected_budget


def validate_public(trace, case, scale, budget):
    """Validate one compact original-public24 online trace.

    The compact schema can validate chronology, actuator clipping, and geometry,
    but it contains no compensation observers and therefore cannot attest that a
    particular force bound affected the command path.
    """
    expected_fields = {
        *rotation.COMPACT_FIELDS,
        "rotation_gain_scale",
        "case_index",
        "method",
        "max_force_n",
    }
    if set(trace) != expected_fields:
        raise ValueError("public compact trace schema mismatch")
    _metadata(trace, case["case_index"], scale, budget)

    config = case["config"]
    count = round(config.duration / config.timestep)
    scalar_fields = (
        "time",
        "true_normal_force",
        "target_normal_force",
        "orientation_error_rad",
        "measured_normal_force",
        "torque_projection_scale",
        "true_contact_gap_m",
    )
    vector_fields = (
        "position",
        "target_position",
        "linear_velocity",
        "target_linear_velocity",
        "requested_tangential_force_world",
    )
    torque_fields = (
        "commanded_torque",
        "applied_torque",
        "lower_torque_limit",
        "upper_torque_limit",
    )
    shapes = {
        **dict.fromkeys(scalar_fields, (count,)),
        **dict.fromkeys(vector_fields, (count, 3)),
        **dict.fromkeys(torque_fields, (count, 7)),
    }
    for name, shape in shapes.items():
        values = np.asarray(trace[name])
        if values.shape != shape or values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid public trace field {name}: expected finite numeric {shape}")

    _close(trace["time"], np.arange(count) * config.timestep, "time grid")
    limits = np.broadcast_to(
        [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], (count, 7)
    )
    _close(trace["lower_torque_limit"], -limits, "fixed lower torque limits")
    _close(trace["upper_torque_limit"], limits, "fixed upper torque limits")
    _close(
        trace["applied_torque"],
        np.clip(trace["commanded_torque"], -limits, limits),
        "actuator clipping",
    )
    projection = np.asarray(trace["torque_projection_scale"])
    if np.any(projection < 0) or np.any(projection > 1):
        raise ValueError("invalid torque projection fraction")

    angle = np.deg2rad(case["scenario"].wall_yaw_deg)
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    gap = (np.array([0.4, 0.0, 0.0]) - trace["position"]) @ normal - 0.025
    _close(trace["true_contact_gap_m"], gap, "true contact gap geometry", atol=1e-12)


def validate_dynamic(trace, case, scale, budget):
    """Validate schedules and the bounded request in one dynamic-case trace."""
    expected_fields = {*combined.TRACE_FIELDS, "max_force_n"}
    if set(trace) != expected_fields:
        raise ValueError("dynamic trace schema mismatch")
    expected_scale = _choice(scale, ROTATION_GAIN_SCALES, "expected rotation gain scale")
    expected_budget = _choice(budget, BUDGETS_N, "expected max_force_n")
    recorded_scale = _choice(
        trace["rotation_gain_scale"], ROTATION_GAIN_SCALES, "trace rotation_gain_scale"
    )
    recorded_budget = _choice(trace["max_force_n"], BUDGETS_N, "trace max_force_n")
    if recorded_scale != expected_scale:
        raise ValueError("trace rotation_gain_scale differs from expected scale")
    if recorded_budget != expected_budget:
        raise ValueError("trace max_force_n differs from expected budget")

    dt = case.config.timestep
    count = round(case.config.duration / dt)
    time = np.arange(count) * dt
    _close(trace["time"], time, "time grid")
    for name, values in trace.items():
        values = np.asarray(values)
        if values.dtype.kind not in "biufUS":
            raise ValueError(f"unsupported array: {name}")
        if values.dtype.kind in "biuf" and not np.all(np.isfinite(values)):
            raise ValueError(f"nonfinite trace: {name}")
    kind = np.asarray(trace["controller_kind"])
    if kind.shape != () or kind.dtype.kind not in "US" or kind.item() != "surface_online":
        raise ValueError("wrong controller identity: controller_kind")
    for name in ("applied_wall_friction", "applied_tool_friction"):
        _close(trace[name], np.array([case.friction.at(t) for t in time]), name)
    bias = np.array([case.wrench_bias_world.at(t) for t in time])
    _close(trace["applied_raw_wrench_bias_world"], bias, "applied bias schedule")
    _close(
        trace["feedback_raw_wrench_bias_world"],
        np.concatenate((bias[:1], bias[:-1])),
        "one-cycle feedback bias",
    )
    _close(
        trace["controller_yaw_error_deg"],
        np.full(count, case.controller_yaw_error_deg),
        "controller yaw error",
    )
    _close(
        trace["trajectory_rate_scale"],
        np.array([case.trajectory.rate_scale_at(t) for t in time]),
        "trajectory schedule",
    )
    limits = np.broadcast_to(
        [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], (count, 7)
    )
    _close(trace["lower_torque_limit"], -limits, "fixed lower torque limits")
    _close(trace["upper_torque_limit"], limits, "fixed upper torque limits")
    _close(
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
    if before.shape != (count,) or after.shape != (count,):
        raise ValueError("invalid coefficient shape")
    _close(before[1:], after[:-1], "coefficient cycle alignment")
    if np.any(before < 0) or np.any(before > 0.9) or np.any(after < 0) or np.any(after > 0.9):
        raise ValueError("coefficient exceeds frozen bound")
    force = np.asarray(trace["diagnostic_corrected_force_n"])
    active = np.asarray(trace["diagnostic_compensation_active"])
    if force.shape != (count,) or np.any(force < 0):
        raise ValueError("invalid corrected force")
    if np.any(active & ((force <= 1) | (trace["target_normal_force"] <= 0))):
        raise ValueError("invalid active compensation state")
    cap = active & (before * force >= expected_budget)
    if not np.array_equal(trace["diagnostic_amplitude_capped"], cap):
        raise ValueError("amplitude cap flag disagrees with force budget")

    normal = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg).rotation[:, 0]
    velocity = np.asarray(trace["target_linear_velocity"])
    tangent_velocity = velocity - np.outer(velocity @ normal, normal)
    speed = np.linalg.norm(tangent_velocity, axis=1)
    direction = tangent_velocity / np.sqrt(speed[:, None] ** 2 + 0.005**2)
    desired = (
        trace["contact_blend"] * np.minimum(before * force, expected_budget)
    )[:, None] * direction
    requested = np.asarray(trace["requested_tangential_force_world"])
    previous = np.concatenate((np.zeros((1, 3)), requested[:-1]))
    previous -= np.outer(previous @ normal, normal)
    delta = desired - previous
    distance = np.linalg.norm(delta, axis=1)
    limited = distance > 20.0 * dt
    certain = np.abs(distance - 20.0 * dt) > 1e-10
    if not np.array_equal(
        trace["diagnostic_slew_limited"][certain], (active & limited)[certain]
    ):
        raise ValueError("slew observer flag disagrees with bounded request")
    expected = previous + delta * np.minimum(
        1.0, 20.0 * dt / np.maximum(distance, 1e-12)
    )[:, None]
    expected[~active] = 0.0
    _close(requested, expected, "observed compensation request")
