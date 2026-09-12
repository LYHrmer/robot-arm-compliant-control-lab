"""Full-state replay, including legal downward-budget transients."""

from dataclasses import replace

import numpy as np
import pytest

from compliant_control_lab.surface_simulation import yaw_frame
from tests.test_load_aware_compensation import DT, NORMAL, inputs
from tools import load_budget_validation as legacy
from tools import measured_budget_validation as validation
from tools.load_aware_compensation import LoadAwareCompensation


def controller_trace(event="falling", count=1750):
    """Record the real stateful controller from reset, without simulator truth."""
    control = LoadAwareCompensation("online", max_force=8.0)
    state, target = inputs()
    frame = yaw_frame(0.0)
    rows = []
    max_error = 0.0
    for step in range(count):
        time = step * DT
        speed = 0.2
        if event == "stop_reverse" and step >= 1400:
            speed = 0.0 if step < 1450 else -0.2
        current_target = replace(target, linear_velocity=np.array([0.0, speed, 0.0]))
        current_state = replace(state, linear_velocity=current_target.linear_velocity.copy())
        packet = np.array([12.0, 8.0 if speed >= 0 else -8.0, 0.0])
        if event == "falling" and step >= 1400:
            packet[1] = 0.0
        available = not (event == "missing" and 1400 <= step < 1450)
        allow = not (event == "projection" and 1400 <= step < 1450)
        contact = not (event == "contact" and 1400 <= step < 1450)
        before, previous = control.equivalent_mu, control.last_force
        ready_before = control.update_ready
        if available:
            control.set_force_measurement(packet)
        force = control.force(current_state, current_target, NORMAL, 12.0, 1.0, contact, dt=DT)
        control.advance(current_state, current_target, NORMAL, DT, allow)
        active = control._active
        direction = current_target.linear_velocity / np.sqrt(speed**2 + 0.005**2)
        desired = min(before * 12.0, control.applied_budget_n) * direction
        distance = float(np.linalg.norm(desired - previous))
        expected = previous + (desired - previous) * min(1.0, 20.0 * DT / max(distance, 1e-12))
        if not active:
            expected = np.zeros(3)
        max_error = max(max_error, float(np.max(np.abs(force - expected))))
        rows.append({
            "time": time,
            "measured_wrench_world": frame.wrench_to_world(np.r_[packet, np.zeros(3)]),
            "measured_normal_force": 12.0,
            "measured_in_contact": contact,
            "target_normal_force": current_target.normal_force,
            "load_force_local": packet if available else np.zeros(3),
            "load_measurement_time_s": time if available else 0.0,
            "load_measurement_age_s": 0.0,
            "raw_load_packet_present": available,
            "raw_load_packet_force": packet if available else np.zeros(3),
            "raw_load_packet_stamp": time if available else 0.0,
            "load_packet_status": 1 if available else 0,
            "load_measurement_available": available,
            "diagnostic_compensation_active": active,
            "diagnostic_corrected_force_n": 12.0,
            "contact_blend": 1.0,
            "load_budget_updated": control.measurement_used_for_budget,
            "load_budget_applied_n": control.applied_budget_n,
            "load_budget_next_n": control.next_budget_n,
            "load_estimate_n": control.load_estimate_n,
            "load_projected_n": control.projected_load_n,
            "load_projection_accepted": allow,
            "torque_projection_scale": 1.0 if allow else 0.9,
            "load_compensation_force_local": force,
            "requested_tangential_force_world": frame.vector_to_world(force),
            "controller_update_ready_before_compute": ready_before,
            "controller_update_ready_after_compute": control.update_ready,
            "controller_coefficient_before_compute": before,
            "controller_coefficient_after_compute": control.equivalent_mu,
            "diagnostic_amplitude_capped": active and before * 12.0 >= control.applied_budget_n,
            "diagnostic_slew_limited": active and distance > 20.0 * DT,
            "measured_position": frame.vector_to_world(current_state.position),
            "target_position": frame.vector_to_world(current_target.position),
            "measured_linear_velocity": frame.vector_to_world(current_state.linear_velocity),
            "target_linear_velocity": frame.vector_to_world(current_target.linear_velocity),
        })
    trace = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    trace.update(dt=np.array(DT), trace_schema_version=np.array(2),
                 load_max_measurement_age_s=np.array(0.020),
                 slew_reconstruction_max_error_n=np.array(max_error),
                 slew_reconstruction_mismatch_cycles=np.array(0))
    return frame, trace


def test_legal_falling_budget_replays_full_state():
    frame, trace = controller_trace()
    magnitude = np.linalg.norm(trace["load_compensation_force_local"], axis=1)
    assert np.any(magnitude > trace["load_budget_applied_n"] + 1e-10)
    assert np.max(magnitude) <= 8.0
    result = validation.validate_trace(trace, frame)
    assert result["coefficient_transition_checked"] is True
    assert result["above_scheduled_budget_cycles"] > 0


def test_historical_validator_rejects_the_legal_transient():
    frame, trace = controller_trace()
    with pytest.raises(ValueError, match="exceeds applied budget"):
        legacy.validate_trace(trace, frame)


@pytest.mark.parametrize("event", ["steady", "missing", "stop_reverse", "projection", "contact"])
def test_complete_controller_event_history(event):
    frame, trace = controller_trace(event)
    result = validation.validate_trace(trace, frame)
    assert result["validated_cycles"] == 1750
    assert result["readiness_transition_checked"]


@pytest.mark.parametrize("field", ["measured_position", "measured_linear_velocity"])
def test_full_replay_refuses_compact_trace(field):
    frame, trace = controller_trace(count=350)
    del trace[field]
    with pytest.raises(ValueError, match=field):
        validation.validate_trace(trace, frame)


@pytest.mark.parametrize("field", ["load_budget_applied_n", "load_budget_next_n",
                                  "load_estimate_n", "load_projected_n",
                                  "controller_coefficient_after_compute",
                                  "controller_coefficient_before_compute",
                                  "load_compensation_force_local"])
def test_state_corruption_fails(field):
    frame, trace = controller_trace(count=350)
    trace[field][320] += 1e-4
    with pytest.raises(ValueError, match="numeric mismatch"):
        validation.validate_trace(trace, frame)


@pytest.mark.parametrize("field", ["load_measurement_available", "load_budget_updated",
                                  "controller_update_ready_after_compute",
                                  "diagnostic_compensation_active"])
def test_gate_corruption_fails(field):
    frame, trace = controller_trace(count=350)
    trace[field][320] = ~trace[field][320]
    with pytest.raises(ValueError, match="disagrees|alignment|transition"):
        validation.validate_trace(trace, frame)


def test_projection_outside_unit_interval_fails_even_when_not_accepted():
    frame, trace = controller_trace("projection")
    trace["torque_projection_scale"][1410] = -0.1
    with pytest.raises(ValueError, match="projection scale"):
        validation.validate_trace(trace, frame)


@pytest.mark.parametrize("kind,code", [("missing", 0), ("repeat", 1), ("stale", 2),
                                      ("negative", 2), ("future", 3),
                                      ("reordered", 4), ("nonfinite", 5)])
def test_raw_packet_gate_is_reconstructed_independently(kind, code):
    _, trace = controller_trace(count=350)
    index = 320
    time = trace["time"]
    if kind == "missing":
        trace["raw_load_packet_present"][index] = False
    elif kind == "repeat":
        trace["raw_load_packet_stamp"][index] = time[index - 1]
    elif kind == "stale":
        trace["raw_load_packet_stamp"][index] = time[index] - 0.03
    elif kind == "negative":
        trace["raw_load_packet_stamp"][index] = -0.001
    elif kind == "future":
        trace["raw_load_packet_stamp"][index] = time[index] + DT
    elif kind == "reordered":
        trace["raw_load_packet_stamp"][index] = time[index - 2]
    else:
        trace["raw_load_packet_force"][index, 1] = np.nan
    trace["load_packet_status"][index] = code
    trace["load_measurement_available"][index] = code == 1
    trace["load_force_local"][index] = trace["raw_load_packet_force"][index] if code == 1 else 0
    trace["load_measurement_time_s"][index] = trace["raw_load_packet_stamp"][index] if code == 1 else 0
    trace["load_measurement_age_s"][index] = time[index] - trace["raw_load_packet_stamp"][index] if code == 1 else 0
    _, available, status, _ = validation._packets(trace, time)
    assert status[index] == code
    assert available[index] == (code == 1)
    trace["load_packet_status"][index] = (code + 1) % 6
    with pytest.raises(ValueError, match="packet status"):
        validation._packets(trace, time)
