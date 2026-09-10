from pathlib import Path

import numpy as np
import pytest

import tools.compensation_budget_study as budget_study
import tools.rotation_gain_regression as rotation
from tools.budget_transfer_validation import validate_dynamic, validate_public

ROOT = Path(__file__).parents[1]
PUBLIC_ARCHIVE = ROOT / "results/franka_rotation_gain_public24"
DYNAMIC_ARCHIVE = ROOT / "results/franka_compensation_budget"


@pytest.fixture(scope="module")
def public_case():
    _, configurations, _ = rotation.load_reference()
    _, cases = rotation.make_protocol([0], configurations)
    return cases[0]


@pytest.fixture(scope="module")
def dynamic_case():
    return next(
        case
        for variant, case in budget_study.study_cases()
        if variant == "combined" and int(case.scenario.wall_yaw_deg) == 0
    )


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _public_trace(*, scale=1.0, budget=6.0):
    trace = _load(PUBLIC_ARCHIVE / "traces/s1__online__case_00__compact.npz")
    trace["rotation_gain_scale"] = np.array(scale)
    trace["max_force_n"] = np.array(budget)
    return trace


def _dynamic_trace(case, *, scale=1.0, budget=6.0):
    label = budget_study.stem("combined", case, budget)
    trace = _load(DYNAMIC_ARCHIVE / "traces" / f"{label}__diagnostic.npz")
    trace["rotation_gain_scale"] = np.array(scale)
    return trace


def test_archived_public_six_newton_trace_validates(public_case):
    validate_public(_public_trace(), public_case, 1.0, 6.0)


@pytest.mark.parametrize("scale", (1.0, 2.0))
@pytest.mark.parametrize("force_budget", (6.0, 8.0))
def test_public_validator_accepts_predeclared_scalar_metadata(
    public_case, scale, force_budget
):
    trace = _public_trace(scale=scale, budget=force_budget)
    validate_public(trace, public_case, scale, force_budget)


def test_public_validator_rejects_time_limit_clipping_and_geometry_tampering(public_case):
    trace = _public_trace()
    trace["time"][100] += 0.001
    with pytest.raises(ValueError, match="time grid"):
        validate_public(trace, public_case, 1.0, 6.0)

    trace = _public_trace()
    trace["upper_torque_limit"][:, 6] = 120.0
    with pytest.raises(ValueError, match="fixed upper torque limits"):
        validate_public(trace, public_case, 1.0, 6.0)

    trace = _public_trace()
    trace["applied_torque"][100, 0] += 0.1
    with pytest.raises(ValueError, match="actuator clipping"):
        validate_public(trace, public_case, 1.0, 6.0)

    trace = _public_trace()
    trace["true_contact_gap_m"][100] += 1e-6
    with pytest.raises(ValueError, match="true contact gap geometry"):
        validate_public(trace, public_case, 1.0, 6.0)


def test_public_validator_rejects_bad_schema_shape_nonfinite_and_projection(public_case):
    trace = _public_trace()
    trace["unlisted"] = np.array(0.0)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_public(trace, public_case, 1.0, 6.0)

    trace = _public_trace()
    trace["position"] = trace["position"][:-1]
    with pytest.raises(ValueError, match="invalid public trace field position"):
        validate_public(trace, public_case, 1.0, 6.0)

    trace = _public_trace()
    trace["true_normal_force"][10] = np.nan
    with pytest.raises(ValueError, match="invalid public trace field true_normal_force"):
        validate_public(trace, public_case, 1.0, 6.0)

    trace = _public_trace()
    trace["torque_projection_scale"][10] = 1.1
    with pytest.raises(ValueError, match="projection fraction"):
        validate_public(trace, public_case, 1.0, 6.0)


def test_public_validator_rejects_wrong_scalar_identity(public_case):
    trace = _public_trace()
    trace["case_index"] = np.array(1)
    with pytest.raises(ValueError, match="case_index"):
        validate_public(trace, public_case, 1.0, 6.0)

    trace = _public_trace()
    trace["method"] = np.array("friction")
    with pytest.raises(ValueError, match="method"):
        validate_public(trace, public_case, 1.0, 6.0)

    trace = _public_trace()
    trace["rotation_gain_scale"] = np.array(2.0)
    with pytest.raises(ValueError, match="differs from expected scale"):
        validate_public(trace, public_case, 1.0, 6.0)

    trace = _public_trace()
    trace["max_force_n"] = np.array(8.0)
    with pytest.raises(ValueError, match="differs from expected budget"):
        validate_public(trace, public_case, 1.0, 6.0)


@pytest.mark.parametrize("force_budget", (6.0, 8.0))
def test_archived_dynamic_gain_one_traces_validate(dynamic_case, force_budget):
    validate_dynamic(
        _dynamic_trace(dynamic_case, budget=force_budget),
        dynamic_case,
        1.0,
        force_budget,
    )


def test_dynamic_validator_accepts_synthetic_gain_two_identity(dynamic_case):
    trace = _dynamic_trace(dynamic_case, scale=2.0, budget=8.0)
    validate_dynamic(trace, dynamic_case, 2.0, 8.0)


def test_dynamic_validator_rejects_tampered_cap_request_and_coefficient(dynamic_case):
    trace = _dynamic_trace(dynamic_case, budget=8.0)
    trace["diagnostic_amplitude_capped"][4500] ^= True
    with pytest.raises(ValueError, match="amplitude cap flag"):
        validate_dynamic(trace, dynamic_case, 1.0, 8.0)

    trace = _dynamic_trace(dynamic_case, budget=8.0)
    trace["requested_tangential_force_world"][4500, 1] += 0.1
    with pytest.raises(ValueError, match="slew observer flag|observed compensation request"):
        validate_dynamic(trace, dynamic_case, 1.0, 8.0)

    trace = _dynamic_trace(dynamic_case, budget=8.0)
    trace["controller_coefficient_before_compute"][100] += 0.01
    with pytest.raises(ValueError, match="coefficient cycle alignment"):
        validate_dynamic(trace, dynamic_case, 1.0, 8.0)


def test_dynamic_validator_rejects_tampered_time_limit_and_input_schedule(dynamic_case):
    trace = _dynamic_trace(dynamic_case)
    trace["time"][100] += 0.001
    with pytest.raises(ValueError, match="time grid"):
        validate_dynamic(trace, dynamic_case, 1.0, 6.0)

    trace = _dynamic_trace(dynamic_case)
    trace["lower_torque_limit"][:, 6] = -120.0
    with pytest.raises(ValueError, match="fixed lower torque limits"):
        validate_dynamic(trace, dynamic_case, 1.0, 6.0)

    trace = _dynamic_trace(dynamic_case)
    trace["applied_wall_friction"][100] += 0.01
    with pytest.raises(ValueError, match="applied_wall_friction"):
        validate_dynamic(trace, dynamic_case, 1.0, 6.0)

    trace = _dynamic_trace(dynamic_case)
    trace["feedback_raw_wrench_bias_world"][100, 0] += 0.01
    with pytest.raises(ValueError, match="one-cycle feedback bias"):
        validate_dynamic(trace, dynamic_case, 1.0, 6.0)


@pytest.mark.parametrize("value", (True, np.bool_(False), np.nan, 0.0, 3.0))
def test_rejects_invalid_expected_rotation_scale(dynamic_case, value):
    with pytest.raises((TypeError, ValueError), match="expected rotation gain scale"):
        validate_dynamic(_dynamic_trace(dynamic_case), dynamic_case, value, 6.0)


@pytest.mark.parametrize("value", (True, np.bool_(False), np.nan, 7.0))
def test_rejects_invalid_expected_budget(dynamic_case, value):
    with pytest.raises((TypeError, ValueError), match="expected max_force_n"):
        validate_dynamic(_dynamic_trace(dynamic_case), dynamic_case, 1.0, value)


def test_dynamic_validator_rejects_wrong_recorded_metadata_and_schema(dynamic_case):
    trace = _dynamic_trace(dynamic_case)
    trace["rotation_gain_scale"] = np.array(2.0)
    with pytest.raises(ValueError, match="differs from expected scale"):
        validate_dynamic(trace, dynamic_case, 1.0, 6.0)

    trace = _dynamic_trace(dynamic_case)
    trace["max_force_n"] = np.array(8.0)
    with pytest.raises(ValueError, match="differs from expected budget"):
        validate_dynamic(trace, dynamic_case, 1.0, 6.0)

    trace = _dynamic_trace(dynamic_case)
    trace.pop("controller_kind")
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_dynamic(trace, dynamic_case, 1.0, 6.0)
