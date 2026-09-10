"""Unit tests for the pure velocity-time trajectory metrics."""

from copy import deepcopy

import numpy as np
import pytest

from tools.velocity_time_metrics import catchup_metrics

DT = 0.002
NORMAL = np.array([1.0, 0.0, 0.0])


def _trace(samples=100, lag_mm=2.0):
    time = np.arange(samples, dtype=float) * DT
    target_velocity = np.tile([0.0, 0.1, 0.0], (samples, 1))
    target_position = np.zeros((samples, 3))
    position = target_position.copy()
    position[:, 1] = -lag_mm / 1_000.0
    return {
        "time": time,
        "position": position,
        "target_position": target_position,
        "linear_velocity": target_velocity.copy(),
        "target_linear_velocity": target_velocity,
    }


def test_signed_lag_units_area_and_full_vector_rms():
    trace = _trace(lag_mm=3.0)
    trace["position"][:, 2] = -4.0 / 1_000.0
    trace["linear_velocity"] += np.array([9.0, 0.03, 0.04])

    metrics = catchup_metrics(trace, NORMAL, 0.0, 0.2)

    assert metrics["samples"] == 100
    assert metrics["tangent_rmse_mm"] == pytest.approx(5.0)
    assert metrics["tangent_velocity_error_rms_m_s"] == pytest.approx(0.05)
    assert metrics["positive_along_lag_area_mm_s"] == pytest.approx(0.6)
    assert metrics["max_positive_along_lag_mm"] == pytest.approx(3.0)
    assert metrics["first_100ms_within_1mm_s"] is None


def test_negative_along_error_has_no_positive_lag_area():
    metrics = catchup_metrics(_trace(lag_mm=-2.0), NORMAL, 0.0, 0.2)

    assert metrics["positive_along_lag_area_mm_s"] == 0.0
    assert metrics["max_positive_along_lag_mm"] == 0.0


def test_direction_follows_signed_target_tangent():
    trace = _trace(lag_mm=2.0)
    trace["target_linear_velocity"][:, 1] = -0.1

    metrics = catchup_metrics(trace, NORMAL, 0.0, 0.2)

    assert metrics["positive_along_lag_area_mm_s"] == 0.0
    assert metrics["max_positive_along_lag_mm"] == 0.0


def test_first_entry_is_inclusive_and_need_not_remain_settled():
    trace = _trace(samples=80, lag_mm=2.0)
    trace["position"][5:55, 1] = -0.001
    trace["position"][55:, 1] = -0.003

    metrics = catchup_metrics(trace, NORMAL, 0.0, 0.16)

    assert metrics["first_100ms_within_1mm_s"] == pytest.approx(5 * DT)


@pytest.mark.parametrize("samples", (49, 60))
def test_entry_is_null_without_fifty_consecutive_samples(samples):
    trace = _trace(samples=samples, lag_mm=2.0)
    trace["position"][:49, 1] = -0.0005

    assert catchup_metrics(trace, NORMAL, 0.0, samples * DT)[
        "first_100ms_within_1mm_s"
    ] is None


def test_window_includes_start_and_excludes_end():
    trace = _trace(samples=10, lag_mm=0.0)
    trace["position"][:, 1] = -np.arange(10) / 1_000.0

    metrics = catchup_metrics(trace, NORMAL, 2 * DT, 5 * DT)

    assert metrics["start_s"] == 2 * DT
    assert metrics["end_s"] == 5 * DT
    assert metrics["samples"] == 3
    assert metrics["max_positive_along_lag_mm"] == pytest.approx(4.0)


def test_real_grid_onset_and_whole_windows_have_frozen_sample_counts():
    trace = _trace(samples=2_250, lag_mm=0.0)

    onset = catchup_metrics(trace, NORMAL, 1.5, 2.0)
    whole = catchup_metrics(trace, NORMAL)

    assert onset["samples"] == 250
    assert whole["samples"] == 1_500
    assert onset["first_100ms_within_1mm_s"] == 1.5
    assert whole["first_100ms_within_1mm_s"] == 1.5


def test_normal_position_and_velocity_components_are_ignored():
    trace = _trace(lag_mm=0.0)
    trace["position"][:, 0] = 100.0
    trace["linear_velocity"][:, 0] = 200.0

    metrics = catchup_metrics(trace, NORMAL, 0.0, 0.2)

    assert metrics["tangent_rmse_mm"] == 0.0
    assert metrics["tangent_velocity_error_rms_m_s"] == 0.0


def test_inputs_are_not_mutated():
    trace = _trace()
    original = deepcopy(trace)
    normal = NORMAL.copy()

    catchup_metrics(trace, normal, 0.0, 0.2)

    np.testing.assert_array_equal(normal, NORMAL)
    for name in trace:
        np.testing.assert_array_equal(trace[name], original[name])


@pytest.mark.parametrize(
    "normal",
    (np.array([1.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([np.nan, 0.0, 0.0])),
)
def test_rejects_invalid_normal(normal):
    with pytest.raises(ValueError, match="finite unit"):
        catchup_metrics(_trace(), normal, 0.0, 0.2)


@pytest.mark.parametrize("kind", ("start", "step", "shape", "nonfinite"))
def test_rejects_time_off_the_fixed_grid(kind):
    trace = _trace()
    if kind == "start":
        trace["time"] += DT
    elif kind == "step":
        trace["time"][10] += DT / 2
    elif kind == "shape":
        trace["time"] = trace["time"][:, None]
    else:
        trace["time"][10] = np.nan

    with pytest.raises(ValueError, match="fixed 0.002 s grid"):
        catchup_metrics(trace, NORMAL, 0.0, 0.2)


@pytest.mark.parametrize(
    "field,transform",
    (
        ("position", lambda values: values[:-1]),
        ("target_position", lambda values: values[:, :2]),
        ("linear_velocity", lambda values: np.full_like(values, np.nan)),
        ("target_linear_velocity", lambda values: values.reshape(-1)),
    ),
)
def test_rejects_invalid_state_or_target_arrays(field, transform):
    trace = _trace()
    trace[field] = transform(trace[field])

    with pytest.raises(ValueError, match="finite with shape"):
        catchup_metrics(trace, NORMAL, 0.0, 0.2)


@pytest.mark.parametrize("start,end", ((0.2, 0.1), (0.2, 0.3), (np.nan, 0.1)))
def test_rejects_invalid_or_empty_window(start, end):
    with pytest.raises(ValueError, match="window"):
        catchup_metrics(_trace(), NORMAL, start, end)


@pytest.mark.parametrize("speed", (0.0, 1e-12))
def test_target_tangential_speed_must_strictly_exceed_threshold(speed):
    trace = _trace()
    trace["target_linear_velocity"][:, 1] = speed

    with pytest.raises(ValueError, match="exceed 1e-12"):
        catchup_metrics(trace, NORMAL, 0.0, 0.2)
