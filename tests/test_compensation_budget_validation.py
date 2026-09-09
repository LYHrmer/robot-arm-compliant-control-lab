from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.franka_control import FrankaState, FrankaTarget
from compliant_control_lab.surface_simulation import yaw_frame
from compliant_control_lab.tangential_compensation import TangentialCompensation
from tools import combined_residual_ablation as ablation
from tools.compensation_budget_validation import validate_trace

ARCHIVE = Path(__file__).parents[1] / "results/franka_combined_residual_ablation"


def _case(yaw=0, variant="combined"):
    return next(
        case
        for label, case in ablation.ablation_cases()
        if label == variant and int(case.scenario.wall_yaw_deg) == yaw
    )


def _trace(case, variant="combined"):
    path = ARCHIVE / "traces" / f"{ablation.stem(variant, case)}__diagnostic.npz"
    with np.load(path, allow_pickle=False) as archive:
        trace = {name: archive[name].copy() for name in archive.files}
    trace["max_force_n"] = np.array(6.0)
    return trace


def _synthetic_budget_trace(case, budget):
    trace = _trace(case)
    trace["max_force_n"] = np.array(float(budget))
    normal = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg).rotation[:, 0]
    velocity = trace["target_linear_velocity"]
    tangent_velocity = velocity - np.outer(velocity @ normal, normal)
    speed = np.linalg.norm(tangent_velocity, axis=1)
    direction = tangent_velocity / np.sqrt(speed[:, None] ** 2 + 0.005**2)
    active = trace["diagnostic_compensation_active"]
    load = trace["controller_coefficient_before_compute"] * trace["diagnostic_corrected_force_n"]
    desired = (trace["contact_blend"] * np.minimum(load, budget))[:, None] * direction
    requested = np.zeros_like(desired)
    limited = np.zeros(len(desired), dtype=bool)
    compensation = TangentialCompensation("online", max_force=budget)
    step = 20.0 * case.config.timestep
    for index in range(len(desired)):
        previous = compensation.last_force
        previous = previous - normal * (normal @ previous)
        delta = desired[index] - previous
        distance = np.linalg.norm(delta)
        limited[index] = bool(active[index] and distance > step)
        compensation._equivalent_mu = trace["controller_coefficient_before_compute"][index]
        state = FrankaState(
            np.zeros(3),
            np.eye(3),
            trace["linear_velocity"][index],
            np.zeros(3),
            trace["diagnostic_corrected_force_n"][index],
        )
        target = FrankaTarget(
            np.zeros(3),
            np.eye(3),
            velocity[index],
            np.zeros(3),
            trace["target_normal_force"][index],
        )
        requested[index] = compensation.force(
            state,
            target,
            normal,
            trace["diagnostic_corrected_force_n"][index],
            trace["contact_blend"][index],
            bool(active[index]),
            dt=case.config.timestep,
        )
    trace["requested_tangential_force_world"] = requested
    trace["diagnostic_amplitude_capped"] = active & (load >= budget)
    trace["diagnostic_slew_limited"] = limited
    return trace


@pytest.mark.parametrize("yaw", (-15, 0, 15))
@pytest.mark.parametrize("variant", ("combined", "no_bias"))
def test_archived_six_newton_traces_validate(yaw, variant):
    case = _case(yaw, variant)
    validate_trace(_trace(case, variant), case, 6.0)


def test_synthetic_eight_newton_request_reconstructs_with_the_budget():
    case = _case()
    original = _trace(case)
    trace = _synthetic_budget_trace(case, 8.0)

    validate_trace(trace, case, 8.0)
    assert np.any(
        trace["controller_coefficient_before_compute"]
        * trace["diagnostic_corrected_force_n"]
        > 6.0
    )
    assert not np.array_equal(
        trace["requested_tangential_force_world"],
        original["requested_tangential_force_world"],
    )


def test_rejects_tampered_cap_and_request_under_eight_newtons():
    case = _case()
    trace = _synthetic_budget_trace(case, 8.0)
    trace["diagnostic_amplitude_capped"][4500] ^= True
    with pytest.raises(ValueError, match="amplitude cap flag"):
        validate_trace(trace, case, 8.0)

    trace = _synthetic_budget_trace(case, 8.0)
    trace["requested_tangential_force_world"][4500, 1] += 0.1
    with pytest.raises(ValueError, match="slew observer flag|observed compensation request"):
        validate_trace(trace, case, 8.0)


def test_rejects_tampered_actuator_limit_time_and_coefficient_alignment():
    case = _case()
    trace = _trace(case)
    trace["upper_torque_limit"][:, 6] = 120.0
    with pytest.raises(ValueError, match="fixed upper torque limits"):
        validate_trace(trace, case, 6.0)

    trace = _trace(case)
    trace["time"][100] += 0.001
    with pytest.raises(ValueError, match="time grid"):
        validate_trace(trace, case, 6.0)

    trace = _trace(case)
    trace["controller_coefficient_before_compute"][100] += 0.01
    with pytest.raises(ValueError, match="coefficient cycle alignment"):
        validate_trace(trace, case, 6.0)


@pytest.mark.parametrize("expected", (True, np.bool_(False), np.nan, 7.0))
def test_rejects_invalid_expected_budget(expected):
    case = _case()
    with pytest.raises((TypeError, ValueError), match="expected max_force_n"):
        validate_trace(_trace(case), case, expected)


@pytest.mark.parametrize("recorded", (True, np.bool_(False), np.nan, 7.0, 8.0))
def test_rejects_invalid_or_mismatched_recorded_budget(recorded):
    case = _case()
    trace = _trace(case)
    trace["max_force_n"] = np.array(recorded)
    with pytest.raises((TypeError, ValueError), match="trace max_force_n"):
        validate_trace(trace, case, 6.0)


def test_rejects_non_boolean_observer_flag_and_nonfinite_trace_value():
    case = _case()
    trace = _trace(case)
    trace["diagnostic_amplitude_capped"] = trace["diagnostic_amplitude_capped"].astype(float)
    with pytest.raises(ValueError, match="invalid boolean telemetry"):
        validate_trace(trace, case, 6.0)

    trace = _trace(case)
    trace["requested_tangential_force_world"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite trace"):
        validate_trace(trace, case, 6.0)
