"""Pure-array corruption tests for the load-budget trace validator."""

from copy import deepcopy

import numpy as np
import pytest

from compliant_control_lab.surface_simulation import yaw_frame
from tools import load_budget_validation as validation


def _trace():
    count, dt = 300, 0.002
    frame = yaw_frame(0.0)
    rotation = frame.rotation
    time = np.arange(count) * dt
    active = np.arange(count) >= 5
    ready = np.arange(count) >= 165
    updated = ready.copy()
    target_local = np.broadcast_to([0.0, 0.02, 0.0], (count, 3)).copy()
    target_world = target_local @ rotation.T
    local_wrench = np.broadcast_to([12.0, 8.0, 0.0, 0.0, 0.0, 0.0], (count, 6))
    wrench_world = np.asarray([frame.wrench_to_world(value) for value in local_wrench])

    applied = np.empty(count)
    next_budget = np.empty(count)
    estimate = np.empty(count)
    projected = np.zeros(count)
    current_estimate, current_next = 0.0, 6.0
    alpha = -np.expm1(-dt / 0.20)
    for index in range(count):
        if not active[index]:
            current_estimate, current_next = 0.0, 6.0
            applied[index] = 6.0
        else:
            applied[index] = current_next
            if updated[index]:
                projected[index] = 8.0
                current_estimate += alpha * (8.0 - current_estimate)
                current_next = np.clip(current_estimate + 0.25, 6.0, 8.0)
        estimate[index], next_budget[index] = current_estimate, current_next

    direction = target_local / np.sqrt(0.02**2 + 0.005**2)
    request = np.zeros((count, 3))
    slew = np.zeros(count, dtype=bool)
    previous = np.zeros(3)
    for index in range(count):
        if not active[index]:
            previous = np.zeros(3)
        else:
            desired = 0.45 * 12.0 * direction[index]
            delta = desired - previous
            distance = np.linalg.norm(delta)
            slew[index] = distance > 20.0 * dt
            previous = previous + delta * min(1.0, 20.0 * dt / max(distance, 1e-12))
        request[index] = previous

    zeros = np.zeros((count, 3))
    return frame, {
        "time": time,
        "dt": np.array(dt),
        "measured_wrench_world": wrench_world,
        "load_force_local": local_wrench[:, :3].copy(),
        "measured_normal_force": np.full(count, 12.0),
        "load_measurement_time_s": time.copy(),
        "load_measurement_age_s": np.zeros(count),
        "diagnostic_compensation_active": active,
        "load_budget_updated": updated,
        "controller_update_ready_after_compute": ready,
        "load_projection_accepted": ready.copy(),
        "torque_projection_scale": np.ones(count),
        "load_budget_applied_n": applied,
        "load_budget_next_n": next_budget,
        "load_estimate_n": estimate,
        "load_projected_n": projected,
        "target_linear_velocity": target_world,
        "controller_coefficient_before_compute": np.full(count, 0.45),
        "controller_coefficient_after_compute": np.full(count, 0.45),
        "diagnostic_corrected_force_n": np.full(count, 12.0),
        "contact_blend": active.astype(float),
        "load_compensation_force_local": request,
        "requested_tangential_force_world": request @ rotation.T,
        "diagnostic_amplitude_capped": np.zeros(count, dtype=bool),
        "diagnostic_slew_limited": active & slew,
        "measured_position": zeros.copy(),
        "target_position": zeros.copy(),
        "measured_linear_velocity": target_world.copy(),
        "slew_reconstruction_max_error_n": np.array(0.0),
        "slew_reconstruction_mismatch_cycles": np.array(0),
    }


def test_valid_trace_reconstructs_all_numeric_paths():
    frame, trace = _trace()

    result = validation.validate_trace(trace, frame)

    assert result["validated_cycles"] == 300
    assert result["load_budget_update_count"] == 135
    assert result["max_force_input_reconstruction_error_n"] == 0.0
    assert result["max_scheduler_reconstruction_error_n"] < 1e-12
    assert result["max_compensation_reconstruction_error_n"] < 1e-12
    assert result["max_coefficient_transition_error"] == 0.0
    assert result["coefficient_transition_checked"] is True


def test_compact_trace_explicitly_reports_skipped_coefficient_audit():
    frame, trace = _trace()
    del trace["measured_position"]
    del trace["measured_linear_velocity"]

    result = validation.validate_trace(trace, frame)

    assert result["coefficient_transition_checked"] is False


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("load_force_local", "measured local force"),
        ("load_budget_applied_n", "scheduler"),
        ("load_compensation_force_local", "compensation force"),
        ("controller_coefficient_after_compute", "coefficient transition"),
    ),
)
def test_numeric_corruption_is_rejected(field, message):
    frame, trace = _trace()
    trace[field] = trace[field].copy()
    trace[field][200] += 1e-4

    with pytest.raises(ValueError, match=message):
        validation.validate_trace(trace, frame)


def test_normal_component_must_match_actor_state():
    frame, trace = _trace()
    trace["measured_normal_force"] = trace["measured_normal_force"].copy()
    trace["measured_normal_force"][10] += 1e-3

    with pytest.raises(ValueError, match="normal component"):
        validation.validate_trace(trace, frame)


@pytest.mark.parametrize("tamper", ("future", "age"))
def test_measurement_time_must_be_causal_and_age_consistent(tamper):
    frame, trace = _trace()
    if tamper == "future":
        trace["load_measurement_time_s"] = trace["load_measurement_time_s"].copy()
        trace["load_measurement_time_s"][10] += 0.002
        message = "noncausal"
    else:
        trace["load_measurement_age_s"] = trace["load_measurement_age_s"].copy()
        trace["load_measurement_age_s"][10] += 0.002
        message = "measurement age"

    with pytest.raises(ValueError, match=message):
        validation.validate_trace(trace, frame)


@pytest.mark.parametrize("gate", ("active", "ready", "projection"))
def test_budget_update_cannot_bypass_control_gate(gate):
    frame, trace = _trace()
    index = 200
    if gate == "active":
        trace["diagnostic_compensation_active"] = trace[
            "diagnostic_compensation_active"
        ].copy()
        trace["diagnostic_compensation_active"][index] = False
    elif gate == "ready":
        trace["controller_update_ready_after_compute"] = trace[
            "controller_update_ready_after_compute"
        ].copy()
        trace["controller_update_ready_after_compute"][index] = False
    else:
        trace["load_projection_accepted"] = trace["load_projection_accepted"].copy()
        trace["load_projection_accepted"][index] = False

    with pytest.raises(ValueError, match="disagrees"):
        validation.validate_trace(trace, frame)


def test_projection_acceptance_requires_unit_scale():
    frame, trace = _trace()
    trace["torque_projection_scale"] = trace["torque_projection_scale"].copy()
    trace["torque_projection_scale"][200] = 0.99

    with pytest.raises(ValueError, match="non-unit scale"):
        validation.validate_trace(trace, frame)


@pytest.mark.parametrize("flag", ("diagnostic_amplitude_capped", "diagnostic_slew_limited"))
def test_cap_and_slew_flags_are_reconstructed(flag):
    frame, trace = _trace()
    trace[flag] = trace[flag].copy()
    trace[flag][200] = ~trace[flag][200]

    with pytest.raises(ValueError, match="cap flag|slew flag"):
        validation.validate_trace(trace, frame)


def test_protocol_dt_and_finite_shapes_are_strict():
    frame, trace = _trace()
    with pytest.raises(ValueError, match="non-protocol"):
        validation.validate_trace(trace, frame, dt=0.001)

    broken = deepcopy(trace)
    broken["load_estimate_n"] = broken["load_estimate_n"][:-1]
    with pytest.raises(ValueError, match="load_estimate_n"):
        validation.validate_trace(broken, frame)

    broken = deepcopy(trace)
    broken["load_projected_n"][20] = np.nan
    with pytest.raises(ValueError, match="load_projected_n"):
        validation.validate_trace(broken, frame)


def test_contact_blend_protocol_bounds_are_enforced():
    frame, trace = _trace()
    trace["contact_blend"] = trace["contact_blend"].copy()
    trace["contact_blend"][20] = 1.01

    with pytest.raises(ValueError, match="protocol bounds"):
        validation.validate_trace(trace, frame)
