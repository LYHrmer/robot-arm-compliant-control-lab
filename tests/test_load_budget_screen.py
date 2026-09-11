"""Synthetic boundary tests for the measured-load budget screen."""

from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from tools import load_budget_screen as screen


def _overall(peak=20.0):
    return {
        "contact_ratio_pct": 99.0,
        "peak_force_n": peak,
        "saturation_pct": 0.0,
        "projection_pct": 0.0,
        "minimum_reserved_torque_headroom_nm": 1.0,
    }


def _arm(*, post, late, force, pose, peak):
    return {
        "overall": {**_overall(peak), "first_100ms_within_1mm_s": None},
        "phases": [
            {"phase": phase, "force_rmse_n": force, "orientation_rmse_deg": pose}
            for phase in screen.PHASES
        ],
        "windows": [
            {
                "window": window,
                "tangent_rmse_mm": late if window == "late_post" else post,
            }
            for window in screen.WINDOWS
        ],
    }


def _dynamic_rows():
    rows = []
    for yaw, scale in product(screen.YAWS, screen.SCALES):
        rows.append(
            {
                "surface_yaw_deg": yaw,
                "rotation_gain_scale": scale,
                "variant": "combined",
                "candidate": _arm(post=2.0, late=3.0, force=0.2, pose=0.1, peak=21.0),
                "reference": _arm(post=2.5, late=3.5, force=0.0, pose=0.0, peak=20.0),
                "budget_exercised": True,
            }
        )
    return rows


def _public_metrics(**updates):
    metrics = {
        "tangent_rmse_mm": 0.0,
        "tangent_velocity_error_rms_m_s": 0.0,
        "force_rmse_n": 0.0,
        "orientation_rmse_deg": 0.0,
        "peak_force_n": 20.0,
        "contact_ratio_pct": 99.0,
        "saturation_pct": 0.0,
        "projection_pct": 0.0,
        "minimum_reserved_torque_headroom_nm": 1.0,
    }
    metrics.update(updates)
    return metrics


def _public_rows():
    return [
        {
            "case_index": case,
            "rotation_gain_scale": scale,
            "candidate": _public_metrics(
                tangent_rmse_mm=screen.PUBLIC_LIMITS["tangent_rmse_mm"],
                tangent_velocity_error_rms_m_s=screen.PUBLIC_LIMITS[
                    "tangent_velocity_error_rms_m_s"
                ],
                force_rmse_n=screen.PUBLIC_LIMITS["force_rmse_n"],
                orientation_rmse_deg=screen.PUBLIC_LIMITS["orientation_rmse_deg"],
                peak_force_n=20.0 + screen.PUBLIC_LIMITS["peak_force_n"],
            ),
            "reference": _public_metrics(),
        }
        for case, scale in product(screen.PUBLIC_CASES, screen.SCALES)
    ]


def test_dynamic_inclusive_boundaries_pass_all_six_pairs_and_three_gains():
    result = screen.dynamic_screen(_dynamic_rows())

    assert result["status"] == "PASS"
    assert result["passed_pairs"] == 6
    assert len(result["pairs"]) == 6
    assert len(result["gain_pairs"]) == 3
    assert all(
        row["post_tangent_reduction_pct"] == pytest.approx(20.0)
        for row in result["pairs"]
    )
    assert all(row["status"] == "PASS" for row in result["gain_pairs"])
    assert result["default_changed"] is result["new_holdout"] is False


def test_thresholds_are_copies_not_mutable_constant_aliases():
    dynamic = screen.dynamic_screen(_dynamic_rows())
    public = screen.public_screen(_public_rows())

    dynamic["criteria"]["maximum_post_tangent_rmse_mm"] = -1
    dynamic["gain_limits"]["post_tangent_rmse_mm"] = -1
    public["public_limits"]["tangent_rmse_mm"] = -1
    assert screen.CRITERIA["maximum_post_tangent_rmse_mm"] == 3.0
    assert screen.GAIN_LIMITS["post_tangent_rmse_mm"] == 0.1
    assert screen.PUBLIC_LIMITS["tangent_rmse_mm"] == 0.1


@pytest.mark.parametrize("kind", ("missing", "duplicate", "unexpected", "boolean"))
def test_dynamic_rejects_incomplete_duplicate_or_unexpected_identity(kind):
    rows = _dynamic_rows()
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif kind == "unexpected":
        rows[0]["variant"] = "no_bias"
    else:
        rows[0]["rotation_gain_scale"] = True

    with pytest.raises(ValueError, match="dynamic"):
        screen.dynamic_screen(rows)


@pytest.mark.parametrize("kind", ("missing", "duplicate", "unexpected"))
def test_dynamic_rejects_bad_nested_phase_or_window_identity(kind):
    rows = _dynamic_rows()
    windows = rows[0]["candidate"]["windows"]
    if kind == "missing":
        windows.pop()
    elif kind == "duplicate":
        windows[-1] = deepcopy(windows[0])
    else:
        windows[0]["window"] = "catchup"

    with pytest.raises(ValueError, match="window"):
        screen.dynamic_screen(rows)


@pytest.mark.parametrize(
    "mutation,failure",
    (
        (lambda row: row["candidate"]["windows"][0].update(tangent_rmse_mm=3.01), "post_accuracy"),
        (lambda row: row["candidate"]["windows"][0].update(tangent_rmse_mm=2.01), "post_reduction"),
        (lambda row: row["candidate"]["windows"][2].update(tangent_rmse_mm=3.01), "late_accuracy"),
        (lambda row: row["candidate"]["phases"][0].update(force_rmse_n=0.201), "phase_force_cost"),
        (lambda row: row["candidate"]["phases"][0].update(orientation_rmse_deg=0.101), "phase_pose_cost"),
        (lambda row: row["candidate"]["overall"].update(peak_force_n=21.01), "full_peak_cost"),
        (lambda row: row.update(budget_exercised=False), "budget_exercised"),
    ),
)
def test_each_dynamic_gate_fails_closed(mutation, failure):
    rows = _dynamic_rows()
    mutation(rows[0])

    result = screen.dynamic_screen(rows)

    assert result["status"] == "FAIL"
    assert failure in result["pairs"][0]["failed_checks"]


def test_dynamic_gain_interaction_uses_candidate_gain2_minus_gain1():
    rows = _dynamic_rows()
    gain2 = next(
        row
        for row in rows
        if row["surface_yaw_deg"] == -15.0 and row["rotation_gain_scale"] == 2.0
    )
    gain2["candidate"]["windows"][0]["tangent_rmse_mm"] = 2.11
    gain2["reference"]["windows"][0]["tangent_rmse_mm"] = 2.6375

    result = screen.dynamic_screen(rows)
    decision = result["gain_pairs"][0]

    assert decision["post_tangent_rmse_mm"] == pytest.approx(0.11)
    assert "post_tangent_rmse_mm" in decision["failed_checks"]
    assert decision["status"] == result["status"] == "FAIL"


def test_dynamic_rejects_missing_or_nonfinite_required_metric():
    rows = _dynamic_rows()
    rows[0]["candidate"]["phases"][0]["force_rmse_n"] = np.nan

    with pytest.raises(ValueError, match="nonfinite"):
        screen.dynamic_screen(rows)


def test_descriptive_catchup_metric_is_not_an_acceptance_gate():
    rows = _dynamic_rows()
    rows[0]["candidate"]["overall"]["first_100ms_within_1mm_s"] = np.nan

    assert screen.dynamic_screen(rows)["status"] == "PASS"


def test_public_inclusive_delta_and_absolute_boundaries_pass():
    result = screen.public_screen(_public_rows())

    assert result["status"] == "PASS"
    assert result["passed_pairs"] == 12
    assert len(result["pairs"]) == 12


@pytest.mark.parametrize("kind", ("missing", "duplicate", "unexpected", "boolean"))
def test_public_rejects_incomplete_duplicate_or_unexpected_identity(kind):
    rows = _public_rows()
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif kind == "unexpected":
        rows[0]["case_index"] = 5
    else:
        rows[0]["rotation_gain_scale"] = True

    with pytest.raises(ValueError, match="public"):
        screen.public_screen(rows)


def test_public_delta_above_limit_fails():
    rows = _public_rows()
    rows[0]["candidate"]["tangent_rmse_mm"] = np.nextafter(
        screen.PUBLIC_LIMITS["tangent_rmse_mm"], np.inf
    )

    result = screen.public_screen(rows)

    assert "tangent_rmse_mm" in result["pairs"][0]["failed_checks"]
    assert result["status"] == "FAIL"


@pytest.mark.parametrize("arm", ("candidate", "reference"))
def test_public_applies_absolute_limits_to_both_arms(arm):
    rows = _public_rows()
    rows[0][arm]["contact_ratio_pct"] = 98.9

    result = screen.public_screen(rows)

    assert f"{arm}_contact" in result["pairs"][0]["failed_checks"]
    assert result["status"] == "FAIL"


def test_public_rejects_nonfinite_required_metric():
    rows = _public_rows()
    rows[0]["reference"]["orientation_rmse_deg"] = np.inf

    with pytest.raises(ValueError, match="nonfinite"):
        screen.public_screen(rows)
