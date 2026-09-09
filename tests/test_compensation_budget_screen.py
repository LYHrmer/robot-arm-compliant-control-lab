from copy import deepcopy
from itertools import product

import pytest

from tools.compensation_budget_screen import PHASES, VARIANTS, WINDOWS, YAWS, screen


def tables():
    rows, phases, windows = [], [], []
    for yaw, variant, budget in product(YAWS, VARIANTS, (6., 8.)):
        key = {"surface_yaw_deg": yaw, "variant": variant, "max_force_n": budget}
        rows.append({**key, "contact_ratio_pct": 100., "peak_force_n": 13.,
                     "saturation_pct": 0., "projection_pct": 0.,
                     "minimum_reserved_torque_headroom_nm": 4.,
                     "max_uncapped_tangential_amplitude_n": budget,
                     "max_matched_request_difference_n": 1. if budget == 8 else 0.})
        phases.extend({**key, "phase": p, "force_rmse_n": .5, "orientation_rmse_deg": .3}
                      for p in PHASES)
        windows.extend({**key, "window": w, "tangent_rmse_mm": 4. if budget == 6 else 2.}
                       for w in WINDOWS)
    return rows, phases, windows


def test_all_pairs_retained_and_default_never_promoted():
    result = screen(*tables())
    assert result["status"] == "PASS"
    assert result["passed_pairs"] == 6
    assert len(result["pairs"]) == 6
    assert not result["default_changed"]
    assert all(p["post_tangent_reduction_pct"] == 50 for p in result["pairs"])


@pytest.mark.parametrize(("table", "field", "value", "reason"), [
    (0, "contact_ratio_pct", 98., "8n_contact"),
    (0, "minimum_reserved_torque_headroom_nm", .99, "8n_reserved_headroom"),
    (0, "projection_pct", .01, "8n_projection"),
    (0, "max_uncapped_tangential_amplitude_n", 6., "budget_exercised"),
    (0, "max_matched_request_difference_n", 0., "budget_exercised"),
    (1, "force_rmse_n", .71, "phase_force_cost"),
    (1, "orientation_rmse_deg", .41, "phase_pose_cost"),
    (2, "tangent_rmse_mm", 3.1, "post_accuracy"),
])
def test_single_bad_cost_cannot_hide_in_average(table, field, value, reason):
    data = tables()
    row = next(row for row in data[table] if row["max_force_n"] == 8.)
    row[field] = value
    result = screen(*data)
    assert result["status"] == "FAIL"
    assert reason in result["pairs"][0]["failed_checks"]


def test_late_regression_fails_even_when_primary_passes():
    data = tables()
    next(r for r in data[2] if r["max_force_n"] == 8. and r["window"] == "late_post")[
        "tangent_rmse_mm"] = 3.1
    assert "late_accuracy" in screen(*data)["pairs"][0]["failed_checks"]


@pytest.mark.parametrize("table", [0, 1, 2])
def test_missing_or_duplicate_identities_reject(table):
    data = tables()
    data[table].pop()
    with pytest.raises(ValueError, match="identities"):
        screen(*data)
    data = tables()
    data[table].append(deepcopy(data[table][-1]))
    with pytest.raises(ValueError, match="duplicate"):
        screen(*data)


def test_nonfinite_and_zero_reference_reject():
    data = tables()
    data[0][0]["peak_force_n"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        screen(*data)
    data = tables()
    data[2][0]["tangent_rmse_mm"] = 0.
    with pytest.raises(ValueError, match="positive reference"):
        screen(*data)
