"""Independent auxiliary-state replay for the reversible hold-cap pilot.

Copied from tools.stationary_recovery.audit so every published auditor stays
frozen: adaptive6_8, hold_cap_tracking and stationary_hold_cap keep their
existing equations and their reports. reversible_hold_cap adds its own
equations here, rebuilt from the recorded trace alone: a hold-entry coefficient
captured on the first stationary cycle, then rate-limited tracking of the
applied cap bounded above by that entry. This module never instantiates or calls
a controller and never integrates dynamics; its caller must also check
full schema, case schedules, metadata, packet source provenance and archive
identity.
"""

import numpy as np

from tools.reversal_recovery.study import SCHEMA, old, packet_audit, previous

STATIONARY_SPEED_TOLERANCE = 1e-12
METHODS = {"adaptive6_8", "hold_cap_tracking", "stationary_hold_cap", "reversible_hold_cap"}


def replay_compensation(trace, parameters, method, dt):
    """Reconstruct coefficients, scheduler, readiness and requests independently."""
    if method not in METHODS:
        raise ValueError(f"unknown compensation replay method: {method!r}")
    p = parameters
    time = np.asarray(trace["time"])
    count = len(time)
    close = previous._close
    close(time, np.arange(count) * dt, "time grid", 1e-12)
    close(trace["dt"], np.asarray(dt), "dt", 1e-12)
    force, available, _, packet_error = packet_audit._packets(trace, time)
    rotation = trace["controller_frame_rotation"]
    velocity = trace["target_linear_velocity"] @ rotation
    measured_velocity = trace["measured_linear_velocity"] @ rotation
    error = (trace["target_position"] - trace["measured_position"]) @ rotation
    velocity[:, 0], measured_velocity[:, 0], error[:, 0] = 0.0, 0.0, 0.0
    speed = np.linalg.norm(velocity, axis=1)
    direction = velocity / np.sqrt(speed[:, None]**2 + p["velocity_scale"]**2)
    corrected, blend = trace["diagnostic_corrected_force_n"], trace["contact_blend"]
    active = trace["measured_in_contact"] & (corrected > 1.0) & (trace["target_normal_force"] > 0)
    if not np.array_equal(active, trace["diagnostic_compensation_active"]):
        raise ValueError("active state differs from measured contact")
    if np.any(corrected < 0) or np.any((blend < 0) | (blend > 1)):
        raise ValueError("invalid corrected force or contact blend")
    # This protocol uses ScheduledSurfaceSimulator, which always supplies actuation.
    # The frozen projector emits scale=0 for every fallback; with present context,
    # scale=1 therefore means "unchanged". We do not replay the nominal wrench here.
    for name, shape in (("cartesian_jacobian", (count, 6, 7)),
                        ("joint_torque_offset", (count, 7)),
                        ("lower_torque_limit", (count, 7)), ("upper_torque_limit", (count, 7))):
        packet_audit._numeric(trace, name, shape)
    if np.any(trace["lower_torque_limit"] >= trace["upper_torque_limit"]):
        raise ValueError("invalid recorded actuation limits")
    scale = packet_audit._numeric(trace, "torque_projection_scale", (count,))
    if np.any((scale < 0.0) | (scale > 1.0)):
        raise ValueError("projection scale outside [0, 1]")
    accepted = scale == 1.0
    if not np.array_equal(packet_audit._flag(trace, "load_projection_accepted", count), accepted):
        raise ValueError("projection acceptance disagrees with scale")
    mu = p["nominal_mu"]
    load, budget, elapsed = 0.0, p["minimum_force"], 0.0
    prior_force, prior_direction = np.zeros(3), np.zeros(3)
    before, after, outputs = np.zeros(count), np.zeros(count), np.zeros((count, 3))
    scheduler = np.zeros((count, 4))
    ready, capped, slew, updated = (np.zeros(count, dtype=bool) for _ in range(4))
    released, recovered, hold_entry = 0, 0, None
    alpha = -np.expm1(-dt / p["load_time_constant"])
    for index in range(count):
        before[index] = mu
        if not active[index] or speed[index] > STATIONARY_SPEED_TOLERANCE:
            hold_entry = None
        if not available[index]:
            load, budget = 0.0, p["minimum_force"]
        applied, projected = budget, 0.0
        if not active[index]:
            mu, elapsed = p["nominal_mu"], 0.0
            prior_force, prior_direction = np.zeros(3), np.zeros(3)
            load, budget, applied = 0.0, p["minimum_force"], p["minimum_force"]
        else:
            desired = blend[index] * min(mu * corrected[index], applied) * direction[index]
            delta = desired - prior_force
            distance = float(np.linalg.norm(delta))
            slew[index] = distance > p["force_slew_rate"] * dt
            prior_force += delta * min(1.0, p["force_slew_rate"] * dt / max(distance, 1e-12))
            capped[index] = mu * corrected[index] >= applied
            eligible = (blend[index] >= 0.99 and speed[index] >= p["min_update_speed"]
                        and measured_velocity[index] @ direction[index] >= p["min_update_speed"] / 2
                        and direction[index] @ prior_direction >= 0.0 and not slew[index])
            elapsed = elapsed + dt if eligible else 0.0
            ready[index] = elapsed >= p["motion_confirm_time"]
            if speed[index] >= p["min_update_speed"]:
                prior_direction = direction[index].copy()
            if ready[index] and accepted[index]:
                drive = direction[index] @ (
                    error[index] + p["velocity_error_time"] * (velocity[index] - measured_velocity[index])
                )
                increment = float(np.clip(
                    dt * p["adaptation_gain"] * corrected[index]
                    / (corrected[index]**2 + p["force_regularizer"]**2) * drive,
                    -p["coefficient_rate_limit"] * dt, p["coefficient_rate_limit"] * dt,
                ))
                if not (increment > 0 and mu * corrected[index] >= applied):
                    mu = float(np.clip(mu + increment, 0.0, p["max_equivalent_mu"]))
                if available[index]:
                    projected = float(np.clip(force[index] @ (velocity[index] / speed[index]),
                                              0.0, p["max_force"]))
                    load += float(alpha * (projected - load))
                    budget = float(np.clip(load + p["load_margin"],
                                           p["minimum_force"], p["max_force"]))
                    updated[index] = True
            release_allowed = (
                (method == "hold_cap_tracking" and speed[index] < p["min_update_speed"])
                or (method == "stationary_hold_cap" and speed[index] <= STATIONARY_SPEED_TOLERANCE)
            )
            if release_allowed and available[index] and accepted[index]:
                decrease = min(max(0.0, mu - applied / corrected[index]),
                               p["coefficient_rate_limit"] * dt)
                mu -= decrease
                released += int(decrease > 0)
            if method == "reversible_hold_cap" and speed[index] <= STATIONARY_SPEED_TOLERANCE:
                # Entry is captured after the inherited update on the first stationary
                # cycle; a missing or rejected gate freezes the update, not the entry.
                if hold_entry is None:
                    hold_entry = mu
                if available[index] and accepted[index]:
                    target_mu = min(hold_entry, applied / corrected[index])
                    increment = float(np.clip(
                        target_mu - mu,
                        -p["coefficient_rate_limit"] * dt, p["coefficient_rate_limit"] * dt,
                    ))
                    tracked = float(np.clip(
                        mu + increment, 0.0, min(hold_entry, p["max_equivalent_mu"]),
                    ))
                    released += int(tracked < mu)
                    recovered += int(tracked > mu)
                    mu = tracked
        after[index], outputs[index] = mu, prior_force
        scheduler[index] = (applied, budget, load, projected)
    coefficient_error = max(
        close(trace["controller_coefficient_before_compute"], before, "coefficient before", 1e-12),
        close(trace["controller_coefficient_after_compute"], after, "coefficient transition", 1e-12),
    )
    request_error = close(trace["load_compensation_force_local"], outputs, "force reconstruction")
    close(trace["requested_tangential_force_world"], outputs @ rotation.T, "world force")
    close(trace[old.AUDIT_FIELDS[0]], np.asarray(request_error), "observer reconstruction")
    mismatch = np.asarray(trace[old.AUDIT_FIELDS[1]])
    if mismatch.shape != () or mismatch.dtype.kind not in "iu" or int(mismatch) != 0:
        raise ValueError("observer mismatch count is not zero")
    scheduler_error = close(np.column_stack([trace[name] for name in (
        "load_budget_applied_n", "load_budget_next_n", "load_estimate_n", "load_projected_n",
    )]), scheduler, "budget reconstruction")
    for name, expected in (
        ("controller_update_ready_after_compute", ready),
        ("controller_update_ready_before_compute", np.r_[False, ready[:-1]]),
        ("diagnostic_amplitude_capped", capped), ("diagnostic_slew_limited", slew),
        ("load_budget_updated", updated),
    ):
        if not np.array_equal(trace[name], expected):
            raise ValueError(f"state/flag mismatch: {name}")
    steps = np.linalg.norm(np.diff(np.vstack((np.zeros(3), outputs)), axis=0), axis=1)
    if np.any(np.linalg.norm(outputs, axis=1) > p["max_force"] + 1e-10):
        raise ValueError("global force bound exceeded")
    if np.any(steps[active] > p["force_slew_rate"] * dt + 1e-10):
        raise ValueError("active force slew exceeded")
    recovery = {"hold_recovery_cycles": recovered} if method == "reversible_hold_cap" else {}
    return {"scope": "coefficient/slew/readiness/scheduler replay; not independent dynamics",
            "trace_schema_version": SCHEMA, "validated_cycles": count, "hold_release_cycles": released,
            "max_coefficient_transition_error": coefficient_error,
            "max_compensation_reconstruction_error_n": request_error,
            "max_scheduler_reconstruction_error_n": scheduler_error,
            "max_packet_reconstruction_error_n": packet_error,
            "projection_gate_basis": "recorded scale; actuation-present simulator; fallback scale is zero",
            "projection_scaled_or_fallback_cycles": int(np.count_nonzero(~accepted)), **recovery}
