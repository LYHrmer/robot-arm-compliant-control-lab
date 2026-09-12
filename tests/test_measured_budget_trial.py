"""Packet-policy and short-run tests for the measured-state fault trials."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from compliant_control_lab.online_compensation_experiment import ProtocolPhase
from compliant_control_lab.surface_simulation import yaw_frame
from tools import budget_gain_interaction as dynamic
from tools import load_budget_trial as old_trial
from tools import measured_budget_trial as trial


def packet(stamp, force=(12.0, 3.0, -2.0), present=True):
    return trial.RawLoadPacket(present, np.asarray(force), stamp)


def short_case(duration):
    case = dynamic.cases()[0][1]
    config = replace(case.config, duration=duration)
    return replace(
        case,
        config=config,
        phases=(ProtocolPhase("short", 0.0, duration),),
        recovery_start_s=None,
    )


def test_packet_gate_status_contract_and_repeated_stamp_boundary():
    gate = trial.LoadPacketGate()

    first = gate.evaluate(packet(1.0), 1.01)
    repeated = gate.evaluate(packet(1.0), 1.02)
    assert first.status is trial.PacketStatus.ACCEPTED
    assert repeated.status is trial.PacketStatus.ACCEPTED
    assert repeated.age_s == pytest.approx(0.02)
    assert gate.evaluate(packet(1.0), 1.020001).status is trial.PacketStatus.STALE
    assert gate.evaluate(packet(-0.001), 0.0).status is trial.PacketStatus.STALE
    assert gate.evaluate(packet(1.02), 1.019).status is trial.PacketStatus.FUTURE
    assert gate.evaluate(packet(0.999), 1.0).status is trial.PacketStatus.REORDERED
    assert gate.evaluate(packet(np.nan), 1.0).status is trial.PacketStatus.NONFINITE
    nonfinite = gate.evaluate(packet(1.0, force=(1.0, np.inf, 0.0)), 1.0)
    assert nonfinite.status is trial.PacketStatus.NONFINITE
    assert gate.evaluate(packet(0.0, present=False), 1.0).status is trial.PacketStatus.MISSING


@pytest.mark.parametrize(
    "status",
    [
        trial.PacketStatus.MISSING,
        trial.PacketStatus.STALE,
        trial.PacketStatus.FUTURE,
        trial.PacketStatus.REORDERED,
        trial.PacketStatus.NONFINITE,
    ],
)
def test_rejected_packet_has_explicit_zero_delivery(status):
    gate = trial.LoadPacketGate()
    gate.last_accepted_stamp_s = 1.0
    candidates = {
        trial.PacketStatus.MISSING: packet(0.0, present=False),
        trial.PacketStatus.STALE: packet(0.0),
        trial.PacketStatus.FUTURE: packet(2.0),
        trial.PacketStatus.REORDERED: packet(0.999),
        trial.PacketStatus.NONFINITE: packet(1.0, force=(np.nan, 0.0, 0.0)),
    }
    decision = gate.evaluate(candidates[status], 1.0)
    assert decision.status is status
    assert not decision.available
    np.testing.assert_array_equal(decision.force_local, np.zeros(3))
    assert decision.stamp_s == decision.age_s == 0.0


def test_fault_adapter_changes_only_tangential_packet_channels():
    frame = yaw_frame(0.0)
    target = SimpleNamespace(linear_velocity=frame.vector_to_world(np.array([0.0, 3.0, 4.0])))
    source = packet(6.5)

    biased = trial.DeterministicFaultAdapter("bias_plus").apply(source, 6.5, target, frame)
    scaled = trial.DeterministicFaultAdapter("scale_0p8").apply(source, 6.5, target, frame)

    assert biased.force_local[0] == source.force_local[0]
    np.testing.assert_allclose(biased.force_local[1:], source.force_local[1:] + [0.45, 0.60])
    assert scaled.force_local[0] == source.force_local[0]
    np.testing.assert_allclose(scaled.force_local[1:], 0.8 * source.force_local[1:])


def test_stale_fault_holds_pre_window_packet_without_hiding_raw_age():
    adapter = trial.DeterministicFaultAdapter("stale")
    target = SimpleNamespace(linear_velocity=np.array([0.0, 0.02, 0.0]))
    frame = yaw_frame(0.0)
    before = adapter.apply(packet(7.996), 7.998, target, frame)
    held = adapter.apply(packet(7.998, force=(99.0, 99.0, 99.0)), 8.0, target, frame)

    assert held is before
    np.testing.assert_array_equal(held.force_local, [12.0, 3.0, -2.0])
    assert held.stamp_s == 7.996
    assert trial.LoadPacketGate().evaluate(held, 8.018).status is trial.PacketStatus.STALE


def test_fault_profiles_and_method_bounds_are_frozen():
    assert trial.METHOD_BOUNDS == {
        "adaptive6_8": (6.0, 8.0),
        "fixed6": (6.0, 6.0),
        "fixed8": (8.0, 8.0),
    }
    assert (trial.FAULTS["bias_plus"].start_s, trial.FAULTS["bias_plus"].end_s) == (6.0, 12.0)
    assert (trial.FAULTS["scale_1p2"].start_s, trial.FAULTS["scale_1p2"].end_s) == (6.0, 12.0)
    assert (trial.FAULTS["missing"].start_s, trial.FAULTS["missing"].end_s) == (8.0, 8.3)
    assert (trial.FAULTS["stale"].start_s, trial.FAULTS["stale"].end_s) == (8.0, 8.3)


def test_short_fresh_simulation_is_old_runner_equivalent_and_full_trace():
    short = short_case(0.006)

    expected = old_trial.run_dynamic(short, 1.0).trace
    actual = trial.run_dynamic(short, 1.0, "adaptive6_8", "fresh").trace

    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values, err_msg=name)
    count = len(actual["time"])
    vector_fields = (
        "measured_position",
        "measured_linear_velocity",
        "target_position",
        "target_linear_velocity",
        "raw_load_packet_force",
        "load_force_local",
    )
    assert all(actual[name].shape == (count, 3) for name in vector_fields)
    for name in (
        "raw_load_packet_present",
        "raw_load_packet_stamp",
        "load_packet_status",
        "load_measurement_available",
        "measured_in_contact",
        "controller_coefficient_before_compute",
        "controller_coefficient_after_compute",
    ):
        assert actual[name].shape == (count,)
    np.testing.assert_array_equal(actual["load_packet_status"], trial.PacketStatus.ACCEPTED)
    np.testing.assert_array_equal(actual["raw_load_packet_force"], actual["load_force_local"])
    assert actual["trace_schema_version"] == 2


@pytest.mark.parametrize("method,bound", [("fixed6", 6.0), ("fixed8", 8.0)])
def test_short_fixed_methods_run_the_same_fault_adapter(method, bound):
    short = short_case(0.004)

    missing_now = trial.FaultConfig("test_missing", "missing", 0.0, 0.004)
    result = trial.run_dynamic(short, 1.0, method, missing_now).trace

    assert result["minimum_force_n"] == result["max_force_n"] == bound
    np.testing.assert_array_equal(result["load_packet_status"], trial.PacketStatus.MISSING)
    assert not np.any(result["load_measurement_available"])
