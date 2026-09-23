"""No-physics checks of the pilot's independent auxiliary-state equations."""

import json

import numpy as np
import pytest

from tools.reversal_recovery import study as original
from tools.stationary_recovery.audit import replay_compensation

DT = 0.002
ARCHIVE = original.ROOT / "results/franka_reversal_recovery_transfer"


@pytest.fixture(scope="module")
def archived_high_pairs():
    document = json.loads((ARCHIVE / "comparison.json").read_text())
    selected = [row for row in document["runs"]
                if row["surface_yaw_deg"] == -15 and row["error_profile"] == "clean"
                and row["scenario"] == "constant_high" and row["seed"] == 11]
    pairs = {}
    for row in selected:
        with np.load(ARCHIVE / row["trace_path"], allow_pickle=False) as source:
            trace = {name: source[name] for name in source.files}
        pairs[row["method"]] = (trace, row["parameters"])
    return pairs


@pytest.mark.parametrize("method", ["adaptive6_8", "hold_cap_tracking"])
def test_archived_parent_and_old_candidate_match_frozen_auditor(archived_high_pairs, method):
    trace, parameters = archived_high_pairs[method]
    report = replay_compensation(trace, parameters, method, DT)
    assert report == original.replay_compensation(trace, parameters, method, DT)
    assert report["validated_cycles"] == 6000
    assert report["hold_release_cycles"] == (59 if method == "hold_cap_tracking" else 0)
    assert report["max_coefficient_transition_error"] < 1e-12
    assert report["max_compensation_reconstruction_error_n"] < 1e-10


def test_stationary_rule_rejects_archived_low_speed_release(archived_high_pairs):
    trace, parameters = archived_high_pairs["hold_cap_tracking"]
    with pytest.raises(ValueError, match="coefficient"):
        replay_compensation(trace, parameters, "stationary_hold_cap", DT)


def test_unknown_method_is_rejected_before_reading_any_trace():
    with pytest.raises(ValueError, match="unknown compensation replay method"):
        replay_compensation({}, {}, "stationary_hold_cap_typo", DT)


@pytest.fixture
def stationary_trace(archived_high_pairs):
    """Five hand-calculated steps; existing arrays provide shapes, not dynamics.

    A nonzero slow target must not release. Zero and the inclusive 1e-12 m/s
    boundary release 0.0006 each, while 2e-12 does not. None of these speeds
    reaches the unchanged 0.005 m/s motion-confirmation threshold.
    """
    reference, archived_parameters = archived_high_pairs["adaptive6_8"]
    count = 5
    trace = {
        name: value[2750:2750 + count].copy()
        if value.ndim and value.shape[0] == 6000 else value.copy()
        for name, value in reference.items()
    }
    time = np.arange(count) * DT
    packet = np.tile([12., 7.8, 0.], (count, 1))
    velocity = np.zeros((count, 3))
    velocity[:, 1] = [0.0001, 0., 1e-12, 2e-12, 0.0001]
    output = np.zeros((count, 3))
    output[:, 1] = [0.04, 0., 1.2e-9, 2.4e-9, 0.04 + 2.4e-9]
    trace.update(
        time=time, dt=np.array(DT), controller_frame_rotation=np.eye(3),
        target_linear_velocity=velocity, measured_linear_velocity=velocity.copy(),
        target_position=np.zeros((count, 3)), measured_position=np.zeros((count, 3)),
        raw_load_packet_force=packet.copy(), load_force_local=packet.copy(),
        raw_load_packet_stamp=time.copy(), load_measurement_time_s=time.copy(),
        controller_coefficient_before_compute=np.array([0.62, 0.62, 0.6194, 0.6188, 0.6188]),
        controller_coefficient_after_compute=np.array([0.62, 0.6194, 0.6188, 0.6188, 0.6188]),
        load_compensation_force_local=output, requested_tangential_force_world=output.copy(),
        diagnostic_slew_limited=np.array([True, False, False, False, True]),
        method=np.array("stationary_hold_cap"),
    )
    for name in ("raw_load_packet_present", "load_measurement_available", "measured_in_contact",
                 "diagnostic_compensation_active", "load_projection_accepted", "diagnostic_amplitude_capped"):
        trace[name] = np.ones(count, dtype=bool)
    for name in ("controller_update_ready_after_compute", "controller_update_ready_before_compute",
                 "load_budget_updated"):
        trace[name] = np.zeros(count, dtype=bool)
    for name in ("load_estimate_n", "load_projected_n", "load_measurement_age_s"):
        trace[name] = np.zeros(count)
    for name in ("diagnostic_corrected_force_n", "target_normal_force"):
        trace[name] = np.full(count, 12.)
    for name in ("load_budget_applied_n", "load_budget_next_n"):
        trace[name] = np.full(count, 6.)
    trace["contact_blend"] = np.ones(count)
    trace["torque_projection_scale"] = np.ones(count)
    trace["load_packet_status"] = np.ones(count, dtype=np.uint8)
    trace[original.old.AUDIT_FIELDS[0]] = np.array(0.)
    trace[original.old.AUDIT_FIELDS[1]] = np.array(0)
    parameters = {**archived_parameters, "nominal_mu": 0.62}
    return trace, parameters


def test_stationary_replay_matches_hand_calculated_boundary_and_slew_steps(stationary_trace):
    trace, parameters = stationary_trace
    report = replay_compensation(trace, parameters, "stationary_hold_cap", DT)
    assert report["validated_cycles"] == 5
    assert report["hold_release_cycles"] == 2
    assert report["max_coefficient_transition_error"] < 1e-12
    assert report["max_compensation_reconstruction_error_n"] < 1e-12


@pytest.mark.parametrize("method", ["adaptive6_8", "hold_cap_tracking"])
def test_old_rules_reject_the_new_boundary_transition(stationary_trace, method):
    trace, parameters = stationary_trace
    with pytest.raises(ValueError, match="coefficient"):
        replay_compensation(trace, parameters, method, DT)


@pytest.mark.parametrize("field,index,change", [
    ("controller_coefficient_after_compute", 1, 1e-4),
    ("load_compensation_force_local", (1, 1), 0.001),
    ("requested_tangential_force_world", (1, 1), 0.001),
    ("load_budget_applied_n", 1, 0.1),
    ("load_budget_next_n", 1, 0.1),
    ("raw_load_packet_force", (1, 1), 0.1),
    ("time", 1, 0.0001),
])
def test_replay_rejects_corrupted_state_or_packet(stationary_trace, field, index, change):
    trace, parameters = stationary_trace
    trace[field][index] += change
    with pytest.raises(ValueError, match="mismatch"):
        replay_compensation(trace, parameters, "stationary_hold_cap", DT)


@pytest.mark.parametrize("field", [
    "controller_update_ready_after_compute", "controller_update_ready_before_compute",
    "diagnostic_amplitude_capped", "diagnostic_slew_limited", "load_budget_updated",
    "load_projection_accepted", "load_measurement_available",
])
def test_replay_rejects_corrupted_gates(stationary_trace, field):
    trace, parameters = stationary_trace
    trace[field][1] = not trace[field][1]
    with pytest.raises(ValueError, match="mismatch|disagrees"):
        replay_compensation(trace, parameters, "stationary_hold_cap", DT)
