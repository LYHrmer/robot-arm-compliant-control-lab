"""Independent numeric validation for compensation-onset observations."""

import numbers

import numpy as np

DT = 0.002
REQUEST_ATOL = 1e-10
TRANSITION_ATOL = 1e-12
PARAMETERS = (
    "mode", "integral_gain", "nominal_mu", "max_force", "velocity_scale",
    "adaptation_gain", "velocity_error_time", "force_regularizer",
    "max_equivalent_mu", "min_update_speed", "force_slew_rate",
    "coefficient_rate_limit", "motion_confirm_time",
)
VECTORS = (
    "normal_local", "position_error_local_m", "velocity_error_local_m_s",
    "target_velocity_local_m_s", "measured_velocity_local_m_s", "direction_local",
    "previous_direction_local", "previous_force_local_n", "requested_force_local_n",
)
SCALARS = (
    "corrected_force_n", "contact_blend", "target_normal_force_n",
    "coefficient_before_force", "coefficient_after_force", "coefficient_after_advance",
    "motion_elapsed_before_s", "motion_elapsed_after_s", "position_drive_m",
    "velocity_drive_m", "drive_m", "candidate_increment", "limited_increment", "dt_s",
)
FLAGS = (
    "in_contact", "active", "update_ready", "amplitude_capped", "slew_limited",
    "advance_called", "allow_integration",
)


def _parameters(values):
    if set(values) != set(PARAMETERS) or values["mode"] != "online":
        raise ValueError("expected every online TangentialCompensation init parameter")
    result = {"mode": "online"}
    for name in PARAMETERS[1:]:
        value = values[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise TypeError(f"compensation parameter must be a numeric scalar: {name}")
        result[name] = float(value)
    numeric = np.array(list(result.values())[1:])
    if not np.all(np.isfinite(numeric)):
        raise ValueError("compensation parameters must be finite")
    if result["integral_gain"] < 0 or result["nominal_mu"] < 0:
        raise ValueError("integral gain and nominal coefficient must be nonnegative")
    if any(result[name] <= 0 for name in PARAMETERS[3:]):
        raise ValueError("online compensation bounds and rates must be positive")
    if result["nominal_mu"] > result["max_equivalent_mu"]:
        raise ValueError("nominal coefficient exceeds its bound")
    return result


def _numeric(trace, name, shape):
    if name not in trace:
        raise ValueError(f"trace missing field: {name}")
    value = np.asarray(trace[name])
    if value.shape != shape or value.dtype.kind not in "iuf" or not np.all(np.isfinite(value)):
        raise ValueError(f"trace field must be finite numeric {shape}: {name}")
    return value.astype(float, copy=False)


def _close(actual, expected, label, atol=REQUEST_ATOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=atol):
        raise ValueError(f"numeric mismatch: {label}")


def _tangent(normal, value):
    return value - normal * float(normal @ value)


def validate_observation(trace: dict, parameters: dict) -> dict:
    """Validate recorded online-compensation transitions without running a controller."""
    parameter = _parameters(parameters)
    time = np.asarray(trace.get("time"))
    if time.ndim != 1 or not len(time) or time.dtype.kind not in "iuf" or not np.all(np.isfinite(time)):
        raise ValueError("time must be a nonempty finite numeric vector")
    count = len(time)
    _close(time, np.arange(count) * DT, "fixed 2 ms time grid", TRANSITION_ATOL)
    if "dt" in trace:
        _close(np.asarray(trace["dt"]), np.asarray(DT), "trace dt", TRANSITION_ATOL)

    vectors = {name: _numeric(trace, f"obs_{name}", (count, 3)) for name in VECTORS}
    scalars = {name: _numeric(trace, f"obs_{name}", (count,)) for name in SCALARS}
    flags = {}
    for name in FLAGS:
        value = np.asarray(trace.get(f"obs_{name}"))
        if value.shape != (count,) or value.dtype.kind != "b":
            raise ValueError(f"observation flag must be boolean ({count},): {name}")
        flags[name] = value
    _close(scalars["dt_s"], np.full(count, DT), "observation dt", TRANSITION_ATOL)

    rotation = _numeric(trace, "controller_frame_rotation", (3, 3))
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-10) or not np.isclose(
        np.linalg.det(rotation), 1.0, rtol=0, atol=1e-10
    ):
        raise ValueError("controller frame must be a proper rotation")
    expected_normal = rotation.T @ rotation[:, 0]
    measured_position = _numeric(trace, "measured_position", (count, 3))
    measured_velocity = _numeric(trace, "measured_linear_velocity", (count, 3))
    target_position = _numeric(trace, "target_position", (count, 3))
    target_velocity = _numeric(trace, "target_linear_velocity", (count, 3))
    measured_wrench = _numeric(trace, "measured_wrench_world", (count, 6))
    measured_force = _numeric(trace, "measured_normal_force", (count,))
    target_force = _numeric(trace, "target_normal_force", (count,))
    blend = _numeric(trace, "contact_blend", (count,))
    _numeric(trace, "governed_normal_lead_m", (count,))
    projection = _numeric(trace, "torque_projection_scale", (count,))
    request_world = _numeric(trace, "requested_tangential_force_world", (count, 3))
    raw_force = measured_wrench[:, :3] @ rotation[:, 0]
    _close(measured_force, raw_force, "measured normal force projection")

    bias = 0.0
    previous_contact = bool(raw_force[0] >= 3.0)
    corrected = np.empty(count)
    for i in range(count):
        if target_force[i] <= 2.0 and not previous_contact:
            alpha = DT / (0.08 + DT)
            bias = float(np.clip(bias + alpha * (raw_force[i] - bias), -3.0, 3.0))
        corrected[i] = max(0.0, raw_force[i] - bias)
        previous_contact = bool(flags["in_contact"][i])
    _close(scalars["corrected_force_n"], corrected, "corrected force argument")
    _close(scalars["contact_blend"], blend, "contact blend argument")
    _close(scalars["target_normal_force_n"], target_force, "target force argument")

    local_measured_velocity = measured_velocity @ rotation
    local_target_velocity = target_velocity @ rotation
    local_position_error = (target_position - measured_position) @ rotation
    _close(vectors["normal_local"], np.broadcast_to(expected_normal, (count, 3)), "local normal")
    expected_measured_velocity = np.asarray([
        _tangent(normal, value)
        for normal, value in zip(vectors["normal_local"], local_measured_velocity)
    ])
    expected_target_velocity = np.asarray([
        _tangent(normal, value)
        for normal, value in zip(vectors["normal_local"], local_target_velocity)
    ])
    expected_position_error = np.asarray([
        _tangent(normal, value)
        for normal, value in zip(vectors["normal_local"], local_position_error)
    ])
    expected_velocity_error = np.asarray([
        _tangent(normal, target - measured)
        for normal, target, measured in zip(
            vectors["normal_local"], local_target_velocity, local_measured_velocity
        )
    ])
    _close(vectors["measured_velocity_local_m_s"], expected_measured_velocity,
           "projected local measured velocity")
    _close(vectors["target_velocity_local_m_s"], expected_target_velocity,
           "projected local target velocity")
    _close(vectors["position_error_local_m"], expected_position_error,
           "projected local position error")
    _close(vectors["velocity_error_local_m_s"], expected_velocity_error,
           "projected local velocity error")
    _close(request_world, vectors["requested_force_local_n"] @ rotation.T,
           "world/local requested force")

    expected_coefficient = parameter["nominal_mu"]
    expected_motion = 0.0
    expected_previous_force = np.zeros(3)
    expected_previous_direction = np.zeros(3)
    request_errors, transition_errors = [], []
    for i in range(count):
        normal = vectors["normal_local"][i]
        before = scalars["coefficient_before_force"][i]
        transition_errors.append(abs(before - expected_coefficient))
        _close(vectors["previous_force_local_n"][i], expected_previous_force,
               "previous force chain", TRANSITION_ATOL)
        _close(vectors["previous_direction_local"][i], expected_previous_direction,
               "previous direction chain", TRANSITION_ATOL)
        if abs(scalars["motion_elapsed_before_s"][i] - expected_motion) > TRANSITION_ATOL:
            raise ValueError("motion elapsed chain differs")

        active = bool(flags["in_contact"][i] and corrected[i] > 1.0 and target_force[i] > 0.0)
        if flags["active"][i] != active:
            raise ValueError("active flag disagrees with force arguments")
        after_force = before if active else parameter["nominal_mu"]
        transition_errors.append(abs(scalars["coefficient_after_force"][i] - after_force))
        velocity = _tangent(normal, vectors["target_velocity_local_m_s"][i])
        speed = float(np.linalg.norm(velocity))
        direction = velocity / np.sqrt(speed**2 + parameter["velocity_scale"]**2) if active else np.zeros(3)
        _close(vectors["direction_local"][i], direction, "soft direction")
        previous = _tangent(normal, vectors["previous_force_local_n"][i]) if active else np.zeros(3)
        desired = blend[i] * min(after_force * corrected[i], parameter["max_force"]) * direction
        distance = float(np.linalg.norm(desired - previous))
        slew = bool(active and distance > parameter["force_slew_rate"] * DT)
        capped = bool(active and after_force * corrected[i] >= parameter["max_force"])
        if flags["slew_limited"][i] != slew or flags["amplitude_capped"][i] != capped:
            raise ValueError("cap or slew flag disagrees with bounded request")
        factor = min(1.0, parameter["force_slew_rate"] * DT / max(distance, 1e-12))
        expected_request = previous + (desired - previous) * factor if active else np.zeros(3)
        request_errors.append(float(np.linalg.norm(vectors["requested_force_local_n"][i] - expected_request)))

        measured_tangent = _tangent(normal, vectors["measured_velocity_local_m_s"][i])
        reversal = float(direction @ vectors["previous_direction_local"][i]) < 0.0
        eligible = bool(active and blend[i] >= 0.99 and speed >= parameter["min_update_speed"]
                        and measured_tangent @ direction >= parameter["min_update_speed"] / 2
                        and not reversal and not slew)
        motion_after = expected_motion + DT if eligible else 0.0
        ready = motion_after >= parameter["motion_confirm_time"]
        if abs(scalars["motion_elapsed_after_s"][i] - motion_after) > TRANSITION_ATOL or flags["update_ready"][i] != ready:
            raise ValueError("motion confirmation or ready flag disagrees")

        position_drive = float(direction @ vectors["position_error_local_m"][i])
        velocity_drive = float(parameter["velocity_error_time"] * direction @ vectors["velocity_error_local_m_s"][i])
        drive = position_drive + velocity_drive
        candidate = (DT * parameter["adaptation_gain"] * corrected[i]
                     / (corrected[i] ** 2 + parameter["force_regularizer"] ** 2) * drive) if active else 0.0
        limited = float(np.clip(candidate, -parameter["coefficient_rate_limit"] * DT,
                                parameter["coefficient_rate_limit"] * DT))
        for name, expected in (("position_drive_m", position_drive), ("velocity_drive_m", velocity_drive),
                               ("drive_m", drive), ("candidate_increment", candidate),
                               ("limited_increment", limited)):
            if abs(scalars[name][i] - expected) > REQUEST_ATOL:
                raise ValueError(f"numeric mismatch: {name}")
        if not flags["advance_called"][i] and flags["allow_integration"][i]:
            raise ValueError("integration cannot be allowed without advance")
        if flags["allow_integration"][i] and abs(projection[i] - 1.0) > REQUEST_ATOL:
            raise ValueError("allowed integration requires unchanged projection")
        after_advance = after_force
        if (flags["advance_called"][i] and active and flags["allow_integration"][i] and ready
                and not (limited > 0.0 and after_force * corrected[i] >= parameter["max_force"])):
            after_advance = float(np.clip(
                after_force + limited, 0.0, parameter["max_equivalent_mu"]
            ))
        transition_errors.append(abs(scalars["coefficient_after_advance"][i] - after_advance))
        expected_coefficient = after_advance
        expected_motion = motion_after
        expected_previous_force = expected_request
        if not active:
            expected_previous_direction = np.zeros(3)
        elif speed >= parameter["min_update_speed"]:
            expected_previous_direction = direction

    max_request = max(request_errors)
    max_transition = max(transition_errors)
    if max_request > REQUEST_ATOL:
        raise ValueError("requested force reconstruction exceeds tolerance")
    if max_transition > TRANSITION_ATOL:
        raise ValueError("coefficient transition exceeds tolerance")
    return {
        "max_request_reconstruction_error_n": max_request,
        "max_coefficient_transition_error": max_transition,
        "validated_cycles": count,
    }
