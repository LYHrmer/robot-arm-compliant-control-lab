import csv
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


def _probe() -> Path:
    configured = os.environ.get("COMPLIANT_CONTROL_CPP_SURFACE_PROBE")
    if configured:
        return Path(configured)
    return Path(__file__).parents[1] / "build" / "compliant_control_surface_probe"


def _numbers(values):
    return [format(float(value), ".17g") for value in np.asarray(values).reshape(-1)]


def _row(index, time_s, *, reset=False, packet_present=True,
         packet_force=(0.0, 7.0, 0.0), packet_stamp=None, torque_offset=0.0):
    rotation = np.eye(3)
    state_position = np.array([0.36, 0.002 * index, -0.003])
    state_velocity = np.array([0.0, 0.018, 0.0])
    target_position = state_position + np.array([0.002, 0.007, 0.0])
    target_velocity = np.array([0.0, 0.020, 0.0])
    jacobian = np.zeros((6, 7))
    jacobian[:, :6] = np.eye(6)
    fields = [str(index), str(int(reset)), format(time_s, ".17g"),
              format(time_s, ".17g"), ".002", "1"]
    fields += _numbers(state_position) + _numbers(rotation)
    fields += _numbers(state_velocity) + _numbers(np.zeros(3)) + ["12"]
    fields += _numbers(target_position) + _numbers(rotation)
    fields += _numbers(target_velocity) + _numbers(np.zeros(3)) + ["12"]
    fields += _numbers(jacobian) + _numbers(np.full(7, torque_offset))
    fields += _numbers(np.full(7, -100.0)) + _numbers(np.full(7, 100.0))
    fields += [str(int(packet_present)), *_numbers(packet_force),
               format(time_s if packet_stamp is None else packet_stamp, ".17g")]
    return " ".join(fields)


def _run(rows, minimum=6.0, maximum=8.0):
    probe = _probe()
    if not probe.is_file():
        pytest.skip("C++ surface probe is not built")
    encoded = " ".join(_numbers(np.eye(3))) + "\n" + "\n".join(rows)
    completed = subprocess.run(
        [str(probe), "--mode", "online", "--load-budget", str(minimum),
         str(maximum), "--maximum-packet-age", ".02"],
        input=encoded, capture_output=True, text=True, check=True,
    )
    return list(csv.reader(completed.stdout.splitlines()))


def test_packet_gate_status_boundary_watermark_and_reset():
    rows = [
        _row(0, 0.000, reset=True, packet_stamp=0.000),
        _row(1, 0.002, packet_present=False, packet_force=(np.nan, 0, 0),
             packet_stamp=np.nan),
        _row(2, 0.004, packet_force=(np.nan, 0, 0), packet_stamp=0.004),
        _row(3, 0.006, packet_stamp=0.007),
        _row(4, 0.008, packet_stamp=-0.001),
        _row(5, 0.010, packet_stamp=0.010),
        _row(6, 0.012, packet_stamp=0.009),
        _row(7, 0.032, packet_stamp=0.012),
        _row(8, 0.054, packet_stamp=0.033),
        _row(9, 0.000, reset=True, packet_stamp=0.000),
    ]
    observed = _run(rows)
    assert all(len(row) == 53 for row in observed)
    assert [int(row[24]) for row in observed] == [1, 0, 5, 3, 2, 1, 4, 1, 2, 1]
    assert [int(row[25]) for row in observed] == [1, 0, 0, 0, 0, 1, 0, 1, 0, 1]
    assert float(observed[7][30]) == pytest.approx(0.020, abs=1e-15)
    for row in (observed[index] for index in (1, 2, 3, 4, 6, 8)):
        np.testing.assert_array_equal(np.asarray(row[26:29], dtype=float), np.zeros(3))
        assert float(row[29]) == float(row[30]) == 0.0
        assert row[2] == "accepted"
        assert row[4:6] == ["0", "1"]


def test_scheduler_learns_then_missing_packet_resets_budget_without_fallback():
    rows = [_row(index, index * 0.002, reset=index == 0) for index in range(420)]
    rows.append(_row(420, 0.840, packet_present=False))
    observed = _run(rows)

    updated = np.asarray([int(row[34]) for row in observed], dtype=bool)
    next_budget = np.asarray([float(row[32]) for row in observed])
    applied_budget = np.asarray([float(row[31]) for row in observed])
    assert updated.any()
    assert next_budget.max() > 6.0
    assert next_budget.max() <= 8.0
    update_index = int(np.flatnonzero(updated)[0])
    assert applied_budget[update_index] == pytest.approx(6.0)
    assert applied_budget[update_index + 1] == pytest.approx(next_budget[update_index])

    missing = observed[-1]
    assert int(missing[24]) == 0
    assert float(missing[31]) == float(missing[32]) == 6.0
    assert float(missing[33]) == 0.0
    assert int(missing[34]) == 0
    assert missing[2:6] == ["accepted", "unchanged", "0", "1"]


def test_watchdog_and_projection_fallback_preserve_watermark_until_explicit_reset():
    observed = _run([
        _row(0, 0.010, reset=True, packet_stamp=0.010),
        _row(1, 0.012, packet_stamp=0.012, torque_offset=200.0),
        _row(2, 0.012, packet_stamp=0.012),
        _row(3, 0.014, packet_stamp=0.011),
        _row(4, 0.016, reset=True, packet_stamp=0.011),
    ])

    assert observed[0][2] == "accepted" and int(observed[0][24]) == 1
    assert observed[1][3:6] == ["nominal_outside", "1", "0"]
    assert int(observed[1][24]) == 1
    assert observed[2][2] == "stale_timestamp"
    assert int(observed[2][24]) == 0
    assert float(observed[2][31]) == float(observed[2][32]) == 6.0
    assert observed[3][2] == "accepted" and int(observed[3][24]) == 4
    assert observed[4][2] == "accepted" and int(observed[4][24]) == 1


def test_load_budget_flag_is_opt_in_and_validated():
    probe = _probe()
    if not probe.is_file():
        pytest.skip("C++ surface probe is not built")
    frame_only = " ".join(_numbers(np.eye(3))) + "\n"
    legacy = subprocess.run(
        [str(probe), "--mode", "online"], input=frame_only,
        capture_output=True, text=True, check=False,
    )
    invalid = subprocess.run(
        [str(probe), "--mode", "online", "--load-budget", "8", "6"],
        input=frame_only, capture_output=True, text=True, check=False,
    )
    wrong_mode = subprocess.run(
        [str(probe), "--mode", "friction", "--load-budget", "6", "8"],
        input=frame_only, capture_output=True, text=True, check=False,
    )
    assert legacy.returncode == 0 and legacy.stdout == ""
    assert invalid.returncode == 2
    assert wrong_mode.returncode != 0
