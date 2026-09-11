"""Synthetic full-regression tests for the measured-load adoption screen."""

from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from tools import measured_budget_screen as screen


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
        "overall": _overall(peak),
        "phases": [
            {"phase": phase, "force_rmse_n": force, "orientation_rmse_deg": pose}
            for phase in screen.pilot.PHASES
        ],
        "windows": [
            {
                "window": window,
                "tangent_rmse_mm": late if window == "late_post" else post,
            }
            for window in screen.pilot.WINDOWS
        ],
    }


def _dynamic_rows():
    return [
        {
            "surface_yaw_deg": yaw,
            "variant": variant,
            "rotation_gain_scale": scale,
            "candidate": _arm(post=2.0, late=3.0, force=0.2, pose=0.1, peak=21.0),
            "reference": _arm(post=2.5, late=3.5, force=0.0, pose=0.0, peak=20.0),
            "budget_exercised": True,
        }
        for yaw, variant, scale in product(screen.YAWS, screen.VARIANTS, screen.SCALES)
    ]


def _metrics(**updates):
    result = {
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
    result.update(updates)
    return result


def _public_rows():
    return [
        {
            "case_index": case,
            "rotation_gain_scale": scale,
            "candidate": _metrics(
                tangent_rmse_mm=screen.PUBLIC_LIMITS["tangent_rmse_mm"],
                tangent_velocity_error_rms_m_s=screen.PUBLIC_LIMITS[
                    "tangent_velocity_error_rms_m_s"
                ],
                force_rmse_n=screen.PUBLIC_LIMITS["force_rmse_n"],
                orientation_rmse_deg=screen.PUBLIC_LIMITS["orientation_rmse_deg"],
                peak_force_n=20.0 + screen.PUBLIC_LIMITS["peak_force_n"],
            ),
            "reference": _metrics(),
            "catchup_candidate": {"first_100ms_within_1mm_s": None},
            "catchup_reference": {"first_100ms_within_1mm_s": None},
        }
        for case, scale in product(screen.PUBLIC_CASES, screen.SCALES)
    ]


def test_complete_boundary_grid_is_eligible_only_for_scoped_adoption():
    result = screen.screen(_dynamic_rows(), _public_rows())

    assert result["status"] == "PASS"
    assert result["eligible_for_scoped_adoption"]
    assert result["global_default_changed"] is result["new_holdout"] is False
    assert result["dynamic"]["passed_pairs"] == 12
    assert result["dynamic"]["passed_gain_pairs"] == 6
    assert len(result["dynamic"]["gain_pairs"]) == 6
    assert result["public24"]["passed_pairs"] == 48
    assert all(
        row["post_tangent_reduction_pct"] == pytest.approx(20.0)
        for row in result["dynamic"]["pairs"]
    )


def test_dynamic_preserves_actual_variants_without_mutating_inputs():
    rows = _dynamic_rows()
    before = deepcopy(rows)

    result = screen.dynamic_screen(rows)

    assert {row["variant"] for row in result["pairs"]} == set(screen.VARIANTS)
    assert {row["variant"] for row in result["gain_pairs"]} == set(screen.VARIANTS)
    assert rows == before


def test_public_screen_ignores_descriptive_catchup_fields_and_preserves_inputs():
    rows = _public_rows()
    rows[0]["catchup_candidate"]["first_100ms_within_1mm_s"] = np.nan
    before = deepcopy(rows)

    result = screen.public_screen(rows)

    assert result["status"] == "PASS"
    for index, (actual, expected) in enumerate(zip(rows, before, strict=True)):
        assert actual.keys() == expected.keys()
        assert actual["candidate"] == expected["candidate"]
        assert actual["reference"] == expected["reference"]
        value = actual["catchup_candidate"]["first_100ms_within_1mm_s"]
        assert np.isnan(value) if index == 0 else value is None


@pytest.mark.parametrize("kind", ("missing", "duplicate", "variant", "scale"))
def test_dynamic_rejects_incomplete_duplicate_or_unexpected_identity(kind):
    rows = _dynamic_rows()
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif kind == "variant":
        rows[0]["variant"] = "other"
    else:
        rows[0]["rotation_gain_scale"] = 3.0

    with pytest.raises(ValueError, match="dynamic identity|dynamic identities"):
        screen.dynamic_screen(rows)


@pytest.mark.parametrize("kind", ("missing", "duplicate", "case", "boolean"))
def test_public_rejects_incomplete_duplicate_or_unexpected_identity(kind):
    rows = _public_rows()
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif kind == "case":
        rows[0]["case_index"] = 24
    else:
        rows[0]["rotation_gain_scale"] = True

    with pytest.raises(ValueError, match="public identity|public identities"):
        screen.public_screen(rows)


def test_no_bias_high_gain_failure_blocks_adoption_without_renaming_variant():
    rows = _dynamic_rows()
    row = next(
        item
        for item in rows
        if item["surface_yaw_deg"] == 0.0
        and item["variant"] == "no_bias"
        and item["rotation_gain_scale"] == 2.0
    )
    row["candidate"]["windows"][0]["tangent_rmse_mm"] = 2.11
    row["reference"]["windows"][0]["tangent_rmse_mm"] = 2.6375

    result = screen.screen(rows, _public_rows())
    failed = next(
        item
        for item in result["dynamic"]["gain_pairs"]
        if item["surface_yaw_deg"] == 0.0 and item["variant"] == "no_bias"
    )

    assert failed["status"] == "FAIL"
    assert "post_tangent_rmse_mm" in failed["failed_checks"]
    assert result["status"] == "FAIL"
    assert not result["eligible_for_scoped_adoption"]


@pytest.mark.parametrize("suite", ("dynamic", "public"))
def test_nonfinite_required_metric_fails_closed(suite):
    if suite == "dynamic":
        rows = _dynamic_rows()
        rows[0]["candidate"]["phases"][0]["force_rmse_n"] = np.nan
        function = screen.dynamic_screen
    else:
        rows = _public_rows()
        rows[0]["reference"]["force_rmse_n"] = np.inf
        function = screen.public_screen

    with pytest.raises(ValueError, match="nonfinite"):
        function(rows)


@pytest.mark.parametrize("arm", ("candidate", "reference"))
def test_public_absolute_failure_on_either_arm_blocks_full_screen(arm):
    rows = _public_rows()
    rows[0][arm]["contact_ratio_pct"] = 98.9

    result = screen.screen(_dynamic_rows(), rows)

    assert f"{arm}_contact" in result["public24"]["pairs"][0]["failed_checks"]
    assert result["status"] == "FAIL"
    assert not result["eligible_for_scoped_adoption"]


def test_returned_thresholds_are_copies_of_unchanged_pilot_thresholds():
    result = screen.screen(_dynamic_rows(), _public_rows())

    assert result["dynamic"]["criteria"] == screen.pilot.CRITERIA
    assert result["dynamic"]["gain_limits"] == screen.pilot.GAIN_LIMITS
    assert result["dynamic"]["absolute_limits"] == screen.pilot.ABSOLUTE
    assert result["public24"]["public_limits"] == screen.pilot.PUBLIC_LIMITS
    result["dynamic"]["criteria"]["maximum_post_tangent_rmse_mm"] = -1.0
    result["public24"]["public_limits"]["tangent_rmse_mm"] = -1.0
    assert screen.pilot.CRITERIA["maximum_post_tangent_rmse_mm"] == 3.0
    assert screen.pilot.PUBLIC_LIMITS["tangent_rmse_mm"] == 0.1
