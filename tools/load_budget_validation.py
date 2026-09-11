"""Independent numeric audit of one measured-load budget trace."""

import numpy as np

DT = 0.002
ATOL = 1e-10
SLEW_RATE_N_S = 20.0
NOMINAL_MU = 0.45
VELOCITY_SCALE_M_S = 0.005
MIN_UPDATE_SPEED_M_S = 0.005
ADAPTATION_GAIN = 800.0
VELOCITY_ERROR_TIME_S = 0.05
FORCE_REGULARIZER_N = 2.0
COEFFICIENT_RATE_LIMIT_S = 0.3
MAX_EQUIVALENT_MU = 0.9


def _numeric(trace, name, shape):
    value = np.asarray(trace.get(name))
    if value.shape != shape or value.dtype.kind not in "iuf" or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid finite numeric field: {name}")
    return value.astype(float, copy=False)


def _flag(trace, name, count):
    value = np.asarray(trace.get(name))
    if value.shape != (count,) or value.dtype.kind != "b":
        raise ValueError(f"invalid boolean field: {name}")
    return value


def _close(actual, expected, label, atol=ATOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.all(np.isfinite(actual)):
        raise ValueError(f"invalid shape/nonfinite: {label}")
    error = float(np.max(np.abs(actual - expected), initial=0.0))
    if error > atol:
        raise ValueError(f"numeric mismatch: {label}")
    return error


def _tangent(normal, value):
    return value - np.outer(value @ normal, normal)


def validate_trace(
    trace,
    frame,
    dt=DT,
    minimum_force=6.0,
    max_force=8.0,
    load_margin=0.25,
    load_time_constant=0.20,
):
    """Reconstruct measured input, scheduler, request and online coefficient state."""
    parameters = np.asarray(
        [dt, minimum_force, max_force, load_margin, load_time_constant], dtype=float
    )
    if (
        not np.all(np.isfinite(parameters))
        or dt != DT
        or not 0 < minimum_force <= max_force
        or load_margin < 0
        or load_time_constant <= 0
    ):
        raise ValueError("invalid or non-protocol load-budget parameters")
    time = np.asarray(trace.get("time"))
    if time.ndim != 1 or not len(time) or time.dtype.kind not in "iuf":
        raise ValueError("time must be a nonempty numeric vector")
    count = len(time)
    _close(time, np.arange(count) * dt, "fixed time grid", 1e-12)
    if "dt" in trace:
        _close(np.asarray(trace["dt"]), np.asarray(dt), "trace dt", 1e-12)

    rotation = np.asarray(frame.rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0, atol=ATOL
    ):
        raise ValueError("frame must contain an orthonormal rotation")
    wrench = _numeric(trace, "measured_wrench_world", (count, 6))
    force_local = _numeric(trace, "load_force_local", (count, 3))
    expected_force = np.asarray([frame.wrench_to_local(value)[:3] for value in wrench])
    force_error = _close(force_local, expected_force, "measured local force")
    measured_normal = _numeric(trace, "measured_normal_force", (count,))
    force_error = max(
        force_error,
        _close(force_local[:, 0], measured_normal, "measured normal component"),
    )
    measurement_time = _numeric(trace, "load_measurement_time_s", (count,))
    measurement_age = _numeric(trace, "load_measurement_age_s", (count,))
    if np.any(measurement_time < 0) or np.any(measurement_time > time):
        raise ValueError("measurement timestamp is noncausal")
    _close(measurement_age, time - measurement_time, "measurement age", 1e-12)

    active = _flag(trace, "diagnostic_compensation_active", count)
    updated = _flag(trace, "load_budget_updated", count)
    ready = _flag(trace, "controller_update_ready_after_compute", count)
    projection_accepted = _flag(trace, "load_projection_accepted", count)
    projection = _numeric(trace, "torque_projection_scale", (count,))
    if np.any(projection_accepted & (projection != 1.0)):
        raise ValueError("accepted projection has non-unit scale")
    expected_updated = active & ready & projection_accepted
    if not np.array_equal(updated, expected_updated):
        raise ValueError("budget update disagrees with active/ready/projection gate")
    applied = _numeric(trace, "load_budget_applied_n", (count,))
    next_budget = _numeric(trace, "load_budget_next_n", (count,))
    estimate = _numeric(trace, "load_estimate_n", (count,))
    projected = _numeric(trace, "load_projected_n", (count,))
    target_velocity = _numeric(trace, "target_linear_velocity", (count, 3)) @ rotation
    local_normal = np.array([1.0, 0.0, 0.0])
    tangent_velocity = _tangent(local_normal, target_velocity)
    speed = np.linalg.norm(tangent_velocity, axis=1)

    expected_estimate, expected_next = 0.0, float(minimum_force)
    scheduler_error = 0.0
    alpha = -np.expm1(-dt / load_time_constant)
    for index in range(count):
        if not active[index]:
            expected_applied = minimum_force
            expected_estimate, expected_next, expected_projected = 0.0, minimum_force, 0.0
            if updated[index]:
                raise ValueError("inactive cycle updates load budget")
        else:
            expected_applied = expected_next
            expected_projected = 0.0
            if updated[index]:
                if speed[index] < MIN_UPDATE_SPEED_M_S:
                    raise ValueError("budget update occurs below minimum target speed")
                unit = tangent_velocity[index] / speed[index]
                expected_projected = float(
                    np.clip(force_local[index] @ unit, 0.0, max_force)
                )
                expected_estimate += alpha * (expected_projected - expected_estimate)
                expected_next = float(
                    np.clip(expected_estimate + load_margin, minimum_force, max_force)
                )
        scheduler_error = max(
            scheduler_error,
            abs(applied[index] - expected_applied),
            abs(estimate[index] - expected_estimate),
            abs(next_budget[index] - expected_next),
            abs(projected[index] - expected_projected),
        )
    if scheduler_error > ATOL:
        raise ValueError("numeric mismatch: load-budget scheduler")

    coefficient_before = _numeric(trace, "controller_coefficient_before_compute", (count,))
    coefficient_after = _numeric(trace, "controller_coefficient_after_compute", (count,))
    corrected_force = _numeric(trace, "diagnostic_corrected_force_n", (count,))
    blend = _numeric(trace, "contact_blend", (count,))
    if np.any(corrected_force < 0) or np.any((blend < 0) | (blend > 1)):
        raise ValueError("corrected force and contact blend are outside protocol bounds")
    request_local = _numeric(trace, "load_compensation_force_local", (count, 3))
    request_world = _numeric(trace, "requested_tangential_force_world", (count, 3))
    _close(request_world, request_local @ rotation.T, "world compensation force")
    direction = tangent_velocity / np.sqrt(speed[:, None] ** 2 + VELOCITY_SCALE_M_S**2)
    previous = np.zeros(3)
    request_error = 0.0
    cycle_request_errors = np.zeros(count)
    expected_cap = np.zeros(count, dtype=bool)
    expected_slew = np.zeros(count, dtype=bool)
    for index in range(count):
        if not active[index]:
            expected = np.zeros(3)
            previous = expected
        else:
            amplitude = min(
                coefficient_before[index] * corrected_force[index], applied[index]
            )
            desired = blend[index] * amplitude * direction[index]
            delta = desired - previous
            distance = float(np.linalg.norm(delta))
            expected_slew[index] = distance > SLEW_RATE_N_S * dt
            expected = previous + delta * min(
                1.0, SLEW_RATE_N_S * dt / max(distance, 1e-12)
            )
            previous = expected
            expected_cap[index] = (
                coefficient_before[index] * corrected_force[index] >= applied[index]
            )
        cycle_request_errors[index] = float(np.max(np.abs(request_local[index] - expected)))
        request_error = max(request_error, cycle_request_errors[index])
    if request_error > ATOL:
        raise ValueError("numeric mismatch: compensation request")
    if np.max(np.abs(request_local[:, 0]), initial=0.0) > ATOL:
        raise ValueError("compensation force is not tangent")
    if np.any(np.linalg.norm(request_local, axis=1) > applied + ATOL):
        raise ValueError("compensation force exceeds applied budget")
    if not np.array_equal(
        _flag(trace, "diagnostic_amplitude_capped", count), active & expected_cap
    ):
        raise ValueError("amplitude cap flag mismatch")
    if not np.array_equal(
        _flag(trace, "diagnostic_slew_limited", count), active & expected_slew
    ):
        raise ValueError("slew flag mismatch")

    coefficient_error = 0.0
    if "measured_position" in trace and "measured_linear_velocity" in trace:
        measured_position = _numeric(trace, "measured_position", (count, 3)) @ rotation
        measured_velocity = _numeric(trace, "measured_linear_velocity", (count, 3)) @ rotation
        target_position = _numeric(trace, "target_position", (count, 3)) @ rotation
        expected_after = coefficient_before.copy()
        for index in range(count):
            if not active[index]:
                expected_after[index] = NOMINAL_MU
            elif ready[index] and projection_accepted[index]:
                position_error = _tangent(local_normal, target_position[index:index + 1]
                                          - measured_position[index:index + 1])[0]
                velocity_error = tangent_velocity[index] - _tangent(
                    local_normal, measured_velocity[index:index + 1]
                )[0]
                drive = float(direction[index] @ (
                    position_error + VELOCITY_ERROR_TIME_S * velocity_error
                ))
                force = corrected_force[index]
                increment = dt * ADAPTATION_GAIN * force / (
                    force**2 + FORCE_REGULARIZER_N**2
                ) * drive
                increment = float(np.clip(
                    increment, -COEFFICIENT_RATE_LIMIT_S * dt,
                    COEFFICIENT_RATE_LIMIT_S * dt,
                ))
                if not (increment > 0 and coefficient_before[index] * force >= applied[index]):
                    expected_after[index] = np.clip(
                        coefficient_before[index] + increment, 0.0, MAX_EQUIVALENT_MU
                    )
        coefficient_error = _close(
            coefficient_after, expected_after, "coefficient transition", 1e-12
        )
    _close(coefficient_before[1:], coefficient_after[:-1], "coefficient cycle alignment", 1e-12)

    recorded_error = np.asarray(trace.get("slew_reconstruction_max_error_n"))
    recorded_mismatches = np.asarray(trace.get("slew_reconstruction_mismatch_cycles"))
    if recorded_error.shape != () or not np.isfinite(recorded_error):
        raise ValueError("invalid observer reconstruction error")
    if recorded_mismatches.shape != () or recorded_mismatches.dtype.kind not in "iu":
        raise ValueError("invalid observer mismatch count")
    if not np.isclose(float(recorded_error), request_error, rtol=0, atol=ATOL):
        raise ValueError("observer reconstruction scalar mismatch")
    expected_mismatches = int(np.count_nonzero(cycle_request_errors > 1e-9))
    if int(recorded_mismatches) != expected_mismatches:
        raise ValueError("observer mismatch count disagrees with reconstruction")
    return {
        "validated_cycles": count,
        "load_budget_update_count": int(np.count_nonzero(updated)),
        "max_force_input_reconstruction_error_n": force_error,
        "max_scheduler_reconstruction_error_n": scheduler_error,
        "max_compensation_reconstruction_error_n": request_error,
        "max_coefficient_transition_error": coefficient_error,
        "coefficient_transition_checked": (
            "measured_position" in trace and "measured_linear_velocity" in trace
        ),
    }
