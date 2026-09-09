from copy import deepcopy

import pytest

from tools.cross_surface_pairs import METRICS, pair_gains


def row(scale, value, **context):
    return {"case": "stop_hold_reverse", "surface_yaw_deg": -15,
            "simulation_seed": 11, "arm": "online", "scale": scale,
            **dict.fromkeys(METRICS, value), **context}


def test_direction_and_identity_are_independent_of_order_and_do_not_mutate():
    rows = [row(2, 5), row(1, 2), row(1, 10, phase="hold"), row(2, 9, phase="hold")]
    before = deepcopy(rows)
    pairs = pair_gains(rows)
    assert rows == before
    assert {r["phase"]: r["delta_tangent_rmse_mm"] for r in pairs} == {"overall": 3, "hold": -1}
    assert pair_gains(list(reversed(rows))) == pairs


@pytest.mark.parametrize("key,value", [("surface_yaw_deg", 0), ("simulation_seed", 29),
                                      ("arm", "friction"), ("case", "modest_combined")])
def test_different_context_does_not_pair(key, value):
    with pytest.raises(ValueError, match="missing"):
        pair_gains([row(1, 1), row(2, 2, **{key: value})])


@pytest.mark.parametrize("scale", [True, 0, 3, "1", float("nan")])
def test_invalid_scales_reject(scale):
    with pytest.raises(ValueError, match="scale"):
        pair_gains([row(scale, 1)])


def test_duplicate_counterparts_reject():
    with pytest.raises(ValueError, match="duplicate"):
        pair_gains([row(1, 1), row(1, 2), row(2, 3)])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, True])
def test_nonfinite_or_missing_required_metric_rejects(value):
    with pytest.raises(ValueError, match="metric"):
        pair_gains([row(1, value), row(2, 2)])


def test_recovery_not_recovered_is_not_zero_seconds():
    pair = pair_gains([row(1, 1, force_recovery_time_s=None),
                       row(2, 2, force_recovery_time_s=0.0)])[0]
    assert pair["delta_force_recovery_time_s"] is None
