"""Hand-calculated state transitions and archive parity, without new dynamics."""

import json

import numpy as np
import pytest

from tools.reversal_recovery import study as original
from tools.reversible_recovery.audit import replay_compensation
from tools.stationary_recovery import audit as stationary

DT = 0.002


@pytest.fixture(scope="module")
def archived_high_pairs():
    archive = original.ROOT / "results/franka_reversal_recovery_transfer"
    document = json.loads((archive / "comparison.json").read_text())
    selected = [row for row in document["runs"] if row["surface_yaw_deg"] == -15
                and row["error_profile"] == "clean" and row["scenario"] == "constant_high"
                and row["seed"] == 11]
    pairs = {}
    for row in selected:
        with np.load(archive / row["trace_path"], allow_pickle=False) as source:
            trace = {name: source[name] for name in source.files}
        pairs[row["method"]] = (trace, row["parameters"])
    return pairs


@pytest.mark.parametrize("method", ["adaptive6_8", "hold_cap_tracking"])
def test_existing_archive_replay_is_unchanged(archived_high_pairs, method):
    trace, parameters = archived_high_pairs[method]
    assert replay_compensation(trace, parameters, method, DT) == original.replay_compensation(
        trace, parameters, method, DT,
    )


def test_stationary_archive_replay_is_unchanged():
    archive = original.ROOT / "results/franka_stationary_recovery_pilot"
    document = json.loads((archive / "comparison.json").read_text())
    rows = [row for row in document["runs"] if row["method"] == "stationary_hold_cap"]
    assert len(rows) == 4
    for row in rows:
        with np.load(archive / row["trace_path"], allow_pickle=False) as source:
            trace = {name: source[name] for name in source.files}
        assert replay_compensation(trace, row["parameters"], row["method"], DT) == (
            stationary.replay_compensation(trace, row["parameters"], row["method"], DT)
        )


def _trace(archived_high_pairs, normal, after):
    """Reuse array shapes only; coefficient expectations are specified by each test."""
    reference, parameters = archived_high_pairs["adaptive6_8"]
    count = len(after)
    trace = {name: value[2750:2750 + count].copy()
             if value.ndim and value.shape[0] == 6000 else value.copy()
             for name, value in reference.items()}
    time = np.arange(count) * DT
    normal = np.asarray(normal, dtype=float)
    packet = np.column_stack([normal, np.full(count, 7.8), np.zeros(count)])
    before = np.r_[0.5, after[:-1]]
    trace.update(
        time=time, dt=np.array(DT), controller_frame_rotation=np.eye(3),
        target_linear_velocity=np.zeros((count, 3)), measured_linear_velocity=np.zeros((count, 3)),
        target_position=np.zeros((count, 3)), measured_position=np.zeros((count, 3)),
        raw_load_packet_force=packet.copy(), load_force_local=packet.copy(),
        raw_load_packet_stamp=time.copy(), load_measurement_time_s=time.copy(),
        controller_coefficient_before_compute=before,
        controller_coefficient_after_compute=np.asarray(after),
        load_compensation_force_local=np.zeros((count, 3)),
        requested_tangential_force_world=np.zeros((count, 3)),
        diagnostic_corrected_force_n=normal, target_normal_force=np.full(count, 12.0),
        diagnostic_amplitude_capped=before * normal >= 6.0,
        method=np.array("reversible_hold_cap"),
    )
    for name in ("raw_load_packet_present", "load_measurement_available", "measured_in_contact",
                 "diagnostic_compensation_active", "load_projection_accepted"):
        trace[name] = np.ones(count, dtype=bool)
    for name in ("controller_update_ready_after_compute", "controller_update_ready_before_compute",
                 "load_budget_updated", "diagnostic_slew_limited"):
        trace[name] = np.zeros(count, dtype=bool)
    for name in ("load_estimate_n", "load_projected_n", "load_measurement_age_s"):
        trace[name] = np.zeros(count)
    for name in ("load_budget_applied_n", "load_budget_next_n"):
        trace[name] = np.full(count, 6.0)
    trace["contact_blend"] = np.ones(count)
    trace["torque_projection_scale"] = np.ones(count)
    trace["load_packet_status"] = np.ones(count, dtype=np.uint8)
    trace[original.old.AUDIT_FIELDS[0]] = np.array(0.0)
    trace[original.old.AUDIT_FIELDS[1]] = np.array(0)
    return trace, {**parameters, "nominal_mu": 0.5}


@pytest.fixture
def reversible_trace(archived_high_pairs):
    return _trace(archived_high_pairs, [12., 14., 12., 8.], [.5, .4994, .5, .5])


def test_replay_recovers_to_entry_after_force_transient(reversible_trace):
    trace, parameters = reversible_trace
    report = replay_compensation(trace, parameters, "reversible_hold_cap", DT)
    assert report["validated_cycles"] == 4
    assert report["hold_release_cycles"] == report["hold_recovery_cycles"] == 1
    assert report["max_coefficient_transition_error"] < 1e-12


@pytest.mark.parametrize("method", ["adaptive6_8", "hold_cap_tracking", "stationary_hold_cap"])
def test_old_rules_reject_the_reversible_transition(reversible_trace, method):
    trace, parameters = reversible_trace
    with pytest.raises(ValueError, match="coefficient"):
        replay_compensation(trace, parameters, method, DT)


@pytest.mark.parametrize("packet_kind", ["missing", "stale"])
def test_entry_survives_packet_and_projection_interruptions(archived_high_pairs, packet_kind):
    trace, parameters = _trace(archived_high_pairs, [12., 14., 12., 12., 12.],
                               [.5, .4994, .4994, .4994, .5])
    index = 2
    trace["load_measurement_available"][index] = False
    trace["load_force_local"][index] = 0.0
    trace["load_measurement_time_s"][index] = trace["load_measurement_age_s"][index] = 0.0
    if packet_kind == "missing":
        trace["raw_load_packet_present"][index] = False
        trace["load_packet_status"][index] = 0
    else:
        trace["raw_load_packet_stamp"][index] = -DT
        trace["load_packet_status"][index] = 2
    trace["torque_projection_scale"][3] = 0.5
    trace["load_projection_accepted"][3] = False
    report = replay_compensation(trace, parameters, "reversible_hold_cap", DT)
    assert report["hold_release_cycles"] == report["hold_recovery_cycles"] == 1
    assert report["projection_scaled_or_fallback_cycles"] == 1


def test_motion_clears_entry_before_the_next_hold(archived_high_pairs):
    trace, parameters = _trace(archived_high_pairs, [12., 14., 12., 12.],
                               [.5, .4994, .4994, .4994])
    trace["target_linear_velocity"][2, 1] = trace["measured_linear_velocity"][2, 1] = 0.001
    trace["load_compensation_force_local"][2, 1] = 0.04
    trace["requested_tangential_force_world"][2, 1] = 0.04
    trace["diagnostic_slew_limited"][2] = True
    report = replay_compensation(trace, parameters, "reversible_hold_cap", DT)
    assert report["hold_release_cycles"] == 1
    assert report["hold_recovery_cycles"] == 0


def test_contact_loss_resets_coefficient_and_the_next_hold_entry(archived_high_pairs):
    trace, parameters = _trace(archived_high_pairs, [12., 14., 12., 8.],
                               [.5, .4994, .5, .5])
    trace["measured_in_contact"][2] = trace["diagnostic_compensation_active"][2] = False
    trace["diagnostic_amplitude_capped"][2] = False
    report = replay_compensation(trace, parameters, "reversible_hold_cap", DT)
    assert report["hold_release_cycles"] == 1
    assert report["hold_recovery_cycles"] == 0


@pytest.mark.parametrize("field,index,change", [
    ("controller_coefficient_after_compute", 2, 1e-4),
    ("controller_coefficient_before_compute", 2, 1e-4),
    ("load_compensation_force_local", (1, 1), 0.001),
    ("requested_tangential_force_world", (1, 1), 0.001),
    ("load_budget_applied_n", 1, 0.1), ("load_budget_next_n", 1, 0.1),
    ("raw_load_packet_force", (1, 1), 0.1), ("time", 1, 0.0001),
])
def test_replay_rejects_numeric_or_packet_corruption(reversible_trace, field, index, change):
    trace, parameters = reversible_trace
    trace[field][index] += change
    with pytest.raises(ValueError, match="mismatch"):
        replay_compensation(trace, parameters, "reversible_hold_cap", DT)


@pytest.mark.parametrize("field", [
    "controller_update_ready_after_compute", "controller_update_ready_before_compute",
    "diagnostic_amplitude_capped", "diagnostic_slew_limited", "load_budget_updated",
    "load_projection_accepted", "load_measurement_available", "diagnostic_compensation_active",
])
def test_replay_rejects_gate_corruption(reversible_trace, field):
    trace, parameters = reversible_trace
    trace[field][1] = not trace[field][1]
    with pytest.raises(ValueError, match="mismatch|disagrees|differs"):
        replay_compensation(trace, parameters, "reversible_hold_cap", DT)


def test_unknown_method_is_rejected_without_reading_trace():
    with pytest.raises(ValueError, match="unknown compensation replay method"):
        replay_compensation({}, {}, "reversible_hold_cap_typo", DT)
