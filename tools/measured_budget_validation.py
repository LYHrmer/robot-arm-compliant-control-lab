"""Version-2 full-state audit; historical compact-trace audits remain unchanged.

The scheduled budget caps the desired amplitude, not the slew-limited output.
The output must remain below the global bound. Packet rejection resets only the
load scheduler; it does not reset the error-driven coefficient or stored force.
"""

import numpy as np

from tools.load_budget_validation import (
    ADAPTATION_GAIN,
    ATOL,
    COEFFICIENT_RATE_LIMIT_S,
    DT,
    FORCE_REGULARIZER_N,
    MAX_EQUIVALENT_MU,
    MIN_UPDATE_SPEED_M_S,
    NOMINAL_MU,
    SLEW_RATE_N_S,
    VELOCITY_ERROR_TIME_S,
    VELOCITY_SCALE_M_S,
    _close,
    _flag,
    _numeric,
)

MAX_MEASUREMENT_AGE_S = 0.020
MOTION_CONFIRM_TIME_S = 0.05


def _packets(trace, time):
    """Independently reconstruct acceptance; never trust a recorded status flag."""
    count = len(time)
    _close(_numeric(trace, "load_max_measurement_age_s", ()),
           np.asarray(MAX_MEASUREMENT_AGE_S), "packet age limit", 0.0)
    present = _flag(trace, "raw_load_packet_present", count)
    raw_force = np.asarray(trace.get("raw_load_packet_force"))
    raw_stamp = np.asarray(trace.get("raw_load_packet_stamp"))
    if (raw_force.shape != (count, 3) or raw_stamp.shape != (count,)
            or raw_force.dtype.kind not in "iuf" or raw_stamp.dtype.kind not in "iuf"):
        raise ValueError("invalid raw packet shape/type")
    status = np.asarray(trace.get("load_packet_status"))
    if status.shape != (count,) or status.dtype.kind not in "iu":
        raise ValueError("invalid integer packet status")
    expected_status = np.zeros(count, dtype=np.uint8)
    expected_force = np.zeros((count, 3))
    expected_stamp, expected_age = np.zeros(count), np.zeros(count)
    last_stamp = -np.inf
    for index in range(count):
        stamp = raw_stamp[index]
        if not present[index]:
            code = 0
        elif not np.isfinite(stamp) or not np.all(np.isfinite(raw_force[index])):
            code = 5
        elif stamp > time[index]:
            code = 3
        elif stamp < 0.0 or time[index] - stamp > MAX_MEASUREMENT_AGE_S + 1e-12:
            code = 2
        elif stamp < last_stamp:
            code = 4
        else:
            code = 1
            last_stamp = stamp
            expected_force[index] = raw_force[index]
            expected_stamp[index] = stamp
            expected_age[index] = time[index] - stamp
        expected_status[index] = code
    if not np.array_equal(status, expected_status):
        raise ValueError("packet status disagrees with raw packet history")
    available = _flag(trace, "load_measurement_available", count)
    if not np.array_equal(available, expected_status == 1):
        raise ValueError("packet availability disagrees with acceptance")
    force = _numeric(trace, "load_force_local", (count, 3))
    error = _close(force, expected_force, "delivered packet force")
    _close(_numeric(trace, "load_measurement_time_s", (count,)), expected_stamp,
           "delivered packet timestamp", 1e-12)
    _close(_numeric(trace, "load_measurement_age_s", (count,)), expected_age,
           "delivered packet age", 1e-12)
    return force, available, status, error


def validate_trace(trace, frame, dt=DT, minimum_force=6.0, max_force=8.0,
                   load_margin=0.25, load_time_constant=0.20):
    """Require measured state and replay all coefficient and scheduler transitions.

    This audits the auxiliary load channel only. Normal force control retains its
    original sensor stream; whole force/torque-sensor fault tolerance is not tested.
    """
    parameters = np.asarray([dt, minimum_force, max_force, load_margin, load_time_constant])
    if (not np.all(np.isfinite(parameters)) or dt != DT
            or not 0 < minimum_force <= max_force or load_margin < 0
            or load_time_constant <= 0):
        raise ValueError("invalid or non-protocol load-budget parameters")
    _close(_numeric(trace, "trace_schema_version", ()), np.asarray(2), "trace schema", 0.0)
    time = np.asarray(trace.get("time"))
    if time.ndim != 1 or not len(time) or time.dtype.kind not in "iuf":
        raise ValueError("time must be a nonempty numeric vector")
    count = len(time)
    _close(time, np.arange(count) * dt, "fixed time grid", 1e-12)
    _close(_numeric(trace, "dt", ()), np.asarray(dt), "trace dt", 1e-12)
    rotation = np.asarray(frame.rotation)
    if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=ATOL)
            or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=ATOL)):
        raise ValueError("frame must contain a proper rotation")
    wrench = _numeric(trace, "measured_wrench_world", (count, 6))
    normal = _numeric(trace, "measured_normal_force", (count,))
    force_error = _close((wrench[:, :3] @ rotation)[:, 0], normal,
                         "original measured normal component")
    force, available, status, packet_error = _packets(trace, time)
    force_error = max(force_error, packet_error)

    # Never substitute true position/velocity when recorded measured state is absent.
    measured_position = _numeric(trace, "measured_position", (count, 3)) @ rotation
    measured_velocity = _numeric(trace, "measured_linear_velocity", (count, 3)) @ rotation
    target_position = _numeric(trace, "target_position", (count, 3)) @ rotation
    velocity = _numeric(trace, "target_linear_velocity", (count, 3)) @ rotation
    velocity[:, 0] = 0.0
    measured_velocity[:, 0] = 0.0
    error = target_position - measured_position
    error[:, 0] = 0.0
    speed = np.linalg.norm(velocity, axis=1)
    direction = velocity / np.sqrt(speed[:, None]**2 + VELOCITY_SCALE_M_S**2)
    active = _flag(trace, "diagnostic_compensation_active", count)
    contact = _flag(trace, "measured_in_contact", count)
    corrected = _numeric(trace, "diagnostic_corrected_force_n", (count,))
    target_force = _numeric(trace, "target_normal_force", (count,))
    blend = _numeric(trace, "contact_blend", (count,))
    if np.any(corrected < 0) or np.any((blend < 0) | (blend > 1)):
        raise ValueError("corrected force and blend outside protocol bounds")
    if not np.array_equal(active, contact & (corrected > 1.0) & (target_force > 0.0)):
        raise ValueError("active flag disagrees with measured contact/force")
    ready = _flag(trace, "controller_update_ready_after_compute", count)
    before_ready = _flag(trace, "controller_update_ready_before_compute", count)
    if before_ready[0] or not np.array_equal(before_ready[1:], ready[:-1]):
        raise ValueError("readiness cycle alignment mismatch")
    accepted = _flag(trace, "load_projection_accepted", count)
    scale = _numeric(trace, "torque_projection_scale", (count,))
    if np.any((scale < 0) | (scale > 1)):
        raise ValueError("projection scale outside protocol bounds")
    if np.any(accepted & (scale != 1.0)):
        raise ValueError("accepted projection has non-unit scale")
    updated = _flag(trace, "load_budget_updated", count)
    if not np.array_equal(updated, active & ready & accepted & available):
        raise ValueError("budget update disagrees with packet/control gate")

    applied = _numeric(trace, "load_budget_applied_n", (count,))
    next_budget = _numeric(trace, "load_budget_next_n", (count,))
    estimate = _numeric(trace, "load_estimate_n", (count,))
    projected = _numeric(trace, "load_projected_n", (count,))
    mu_before = _numeric(trace, "controller_coefficient_before_compute", (count,))
    mu_after = _numeric(trace, "controller_coefficient_after_compute", (count,))
    if np.any((mu_before < 0) | (mu_before > MAX_EQUIVALENT_MU)):
        raise ValueError("coefficient outside protocol bounds")
    request = _numeric(trace, "load_compensation_force_local", (count, 3))
    _close(_numeric(trace, "requested_tangential_force_world", (count, 3)),
           request @ rotation.T, "world compensation force")
    cap_flags = np.zeros(count, dtype=bool)
    slew_flags = np.zeros(count, dtype=bool)
    replay_ready = np.zeros(count, dtype=bool)
    replay_before, replay_after = np.zeros(count), np.zeros(count)
    replay_scheduler = np.zeros((count, 4))
    replay_force = np.zeros_like(request)
    previous, previous_direction = np.zeros(3), np.zeros(3)
    load, budget, mu, motion_elapsed = 0.0, minimum_force, NOMINAL_MU, 0.0
    alpha = -np.expm1(-dt / load_time_constant)
    for index in range(count):
        replay_before[index] = mu
        if not available[index]:
            load, budget = 0.0, minimum_force
        current_budget = budget
        projected_load = 0.0
        if not active[index]:
            load, budget, current_budget = 0.0, minimum_force, minimum_force
            mu, motion_elapsed = NOMINAL_MU, 0.0
            previous, previous_direction = np.zeros(3), np.zeros(3)
        else:
            desired = blend[index] * min(mu * corrected[index], current_budget) * direction[index]
            delta = desired - previous
            distance = float(np.linalg.norm(delta))
            slew_flags[index] = distance > SLEW_RATE_N_S * dt
            previous = previous + delta * min(1.0, SLEW_RATE_N_S * dt / max(distance, 1e-12))
            cap_flags[index] = mu * corrected[index] >= current_budget
            eligible = (blend[index] >= 0.99 and speed[index] >= MIN_UPDATE_SPEED_M_S
                        and float(measured_velocity[index] @ direction[index])
                        >= MIN_UPDATE_SPEED_M_S / 2
                        and float(direction[index] @ previous_direction) >= 0.0
                        and not slew_flags[index])
            motion_elapsed = motion_elapsed + dt if eligible else 0.0
            replay_ready[index] = motion_elapsed >= MOTION_CONFIRM_TIME_S
            if speed[index] >= MIN_UPDATE_SPEED_M_S:
                previous_direction = direction[index].copy()
            if replay_ready[index] and accepted[index]:
                drive = float(direction[index] @ (error[index] + VELOCITY_ERROR_TIME_S
                                                   * (velocity[index] - measured_velocity[index])))
                normal_force = corrected[index]
                increment = float(np.clip(
                    dt * ADAPTATION_GAIN * normal_force
                    / (normal_force**2 + FORCE_REGULARIZER_N**2) * drive,
                    -COEFFICIENT_RATE_LIMIT_S * dt, COEFFICIENT_RATE_LIMIT_S * dt))
                if not (increment > 0 and mu * normal_force >= current_budget):
                    mu = float(np.clip(mu + increment, 0.0, MAX_EQUIVALENT_MU))
                if available[index]:
                    projected_load = float(np.clip(
                        force[index] @ (velocity[index] / speed[index]), 0.0, max_force))
                    load += float(alpha * (projected_load - load))
                    budget = float(np.clip(load + load_margin, minimum_force, max_force))
        replay_after[index] = mu
        replay_force[index] = previous
        replay_scheduler[index] = (current_budget, budget, load, projected_load)

    scheduler_error = _close(np.column_stack((applied, next_budget, estimate, projected)),
                             replay_scheduler, "load-budget scheduler")
    coefficient_error = max(_close(mu_before, replay_before, "coefficient before", 1e-12),
                            _close(mu_after, replay_after, "coefficient transition", 1e-12))
    request_error = _close(request, replay_force, "compensation request")
    if not np.array_equal(ready, replay_ready):
        raise ValueError("readiness state transition mismatch")
    for name, expected in (("diagnostic_amplitude_capped", cap_flags),
                           ("diagnostic_slew_limited", slew_flags)):
        if not np.array_equal(_flag(trace, name, count), expected):
            raise ValueError(f"{name} flag mismatch")
    magnitudes = np.linalg.norm(request, axis=1)
    if np.any(magnitudes > max_force + ATOL):
        raise ValueError("compensation force exceeds global hard bound")
    steps = np.linalg.norm(np.diff(np.vstack((np.zeros(3), request)), axis=0), axis=1)
    # Contact loss is the existing instantaneous reset exception, not a slew claim.
    if np.any(steps[active] > SLEW_RATE_N_S * dt + ATOL):
        raise ValueError("compensation force exceeds active-cycle slew bound")
    _close(_numeric(trace, "slew_reconstruction_max_error_n", ()), np.asarray(request_error),
           "observer reconstruction scalar")
    mismatch = np.asarray(trace.get("slew_reconstruction_mismatch_cycles"))
    if mismatch.shape != () or mismatch.dtype.kind not in "iu" or int(mismatch) != 0:
        raise ValueError("invalid observer mismatch count")
    return {
        "trace_schema_version": 2,
        "validated_cycles": count,
        "load_budget_update_count": int(np.count_nonzero(updated)),
        "packet_status_counts": {str(code): int(np.count_nonzero(status == code))
                                 for code in range(6)},
        "max_force_input_reconstruction_error_n": force_error,
        "max_scheduler_reconstruction_error_n": scheduler_error,
        "max_compensation_reconstruction_error_n": request_error,
        "max_coefficient_transition_error": coefficient_error,
        "coefficient_transition_checked": True,
        "readiness_transition_checked": True,
        "above_scheduled_budget_cycles": int(np.count_nonzero(magnitudes > applied + ATOL)),
        "max_active_slew_n_s": float(np.max(steps[active], initial=0.0) / dt),
        "peak_requested_force_n": float(np.max(magnitudes)),
    }
