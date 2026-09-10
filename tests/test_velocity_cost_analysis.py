import numpy as np
import pytest

from tools.velocity_cost_analysis import analyze_pair

DT = 0.002
SAMPLES = 2250
PER_WINDOW = 250
TOTAL = 1500
FIRST = 750
NORMAL = np.array([0.0, 0.0, 1.0])
ALONG = np.array([1.0, 0.0, 0.0])
CROSS = np.array([0.0, 1.0, 0.0])
TARGET_SPEED = 0.05
ORIGIN = np.array([0.4, 0.0, 0.35])


def _fill(value) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, dtype=float), (SAMPLES,)).astype(float)


def _alternate(even: float, odd: float) -> np.ndarray:
    values = np.full(SAMPLES, float(even))
    values[1::2] = float(odd)
    return values


def _after(index: int, value: float) -> np.ndarray:
    values = np.zeros(SAMPLES)
    values[index:] = float(value)
    return values


def _trace(
    *,
    velocity_along=0.0,
    velocity_cross=0.0,
    velocity_normal=0.0,
    position_along_mm=0.0,
    position_cross_mm=0.0,
    request_along=0.0,
    request_cross=0.0,
    request_normal=0.0,
    target_direction=ALONG,
) -> dict[str, np.ndarray]:
    """Uniform 0-4.498 s synthetic trace with prescribed in-frame errors and requests.

    The last timestamp is end - dt, which is the conventional sampling the analysis allows.
    """
    time = DT * np.arange(SAMPLES)
    target_velocity = np.outer(_fill(TARGET_SPEED), np.asarray(target_direction, dtype=float))
    target_position = ORIGIN + np.cumsum(target_velocity * DT, axis=0)
    velocity_offset = (
        np.outer(_fill(velocity_along), ALONG)
        + np.outer(_fill(velocity_cross), CROSS)
        + np.outer(_fill(velocity_normal), NORMAL)
    )
    position_offset = (
        np.outer(_fill(position_along_mm), ALONG) + np.outer(_fill(position_cross_mm), CROSS)
    ) / 1000.0
    request = (
        np.outer(_fill(request_along), ALONG)
        + np.outer(_fill(request_cross), CROSS)
        + np.outer(_fill(request_normal), NORMAL)
    )
    return {
        "time": time,
        "position": target_position + position_offset,
        "target_position": target_position,
        "linear_velocity": target_velocity + velocity_offset,
        "target_linear_velocity": target_velocity,
        "requested_tangential_force_world": request,
        "unused_extra_field": np.zeros(SAMPLES),
    }


def _rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _rotate(trace: dict, rotation: np.ndarray) -> dict:
    rotated = {}
    for name, value in trace.items():
        rotated[name] = value @ rotation.T if value.ndim == 2 else value.copy()
    return rotated


def test_components_means_and_variances_are_hand_derived():
    reference = _trace(
        velocity_along=_alternate(0.03, -0.01),
        velocity_cross=0.004,
        position_along_mm=_alternate(3.0, 1.0),
    )
    candidate = _trace(
        velocity_along=0.02,
        velocity_cross=_alternate(0.01, -0.01),
        velocity_normal=0.2,
        position_along_mm=1.0,
        position_cross_mm=1.0,
    )

    result = analyze_pair(reference, candidate, NORMAL)

    assert set(result) == {"overall", "windows", "timing"}
    overall = result["overall"]
    assert overall["start_s"] == 1.5
    assert overall["end_s"] == 4.5
    assert overall["samples"] == TOTAL
    assert overall["along_mse_6_m2_s2"] == pytest.approx(0.0005)
    assert overall["cross_mse_6_m2_s2"] == pytest.approx(1.6e-5)
    assert overall["along_mse_8_m2_s2"] == pytest.approx(0.0004)
    assert overall["cross_mse_8_m2_s2"] == pytest.approx(0.0001)
    assert overall["along_mean_6_m_s"] == pytest.approx(0.01)
    assert overall["along_mean_8_m_s"] == pytest.approx(0.02)
    assert overall["along_mean_6_m_s"] != pytest.approx(overall["along_mean_8_m_s"])
    assert overall["cross_mean_6_m_s"] == pytest.approx(0.004)
    assert overall["cross_mean_8_m_s"] == pytest.approx(0.0, abs=1e-15)
    assert overall["along_variance_6_m2_s2"] == pytest.approx(0.0004)
    assert overall["along_variance_8_m2_s2"] == pytest.approx(0.0, abs=1e-18)
    assert overall["cross_variance_6_m2_s2"] == pytest.approx(0.0, abs=1e-18)
    assert overall["cross_variance_8_m2_s2"] == pytest.approx(0.0001)
    assert overall["velocity_rms_6_m_s"] == pytest.approx(np.sqrt(0.000516))
    assert overall["velocity_rms_8_m_s"] == pytest.approx(np.sqrt(0.0005))
    assert overall["velocity_rms_delta_m_s"] == pytest.approx(
        np.sqrt(0.0005) - np.sqrt(0.000516)
    )
    assert overall["position_rms_6_mm"] == pytest.approx(np.sqrt(5.0))
    assert overall["position_rms_8_mm"] == pytest.approx(np.sqrt(2.0))
    assert overall["position_rms_delta_mm"] == pytest.approx(np.sqrt(2.0) - np.sqrt(5.0))
    assert overall["position_along_mean_6_mm"] == pytest.approx(2.0)
    assert overall["position_along_mean_8_mm"] == pytest.approx(1.0)
    assert overall["mse_delta_m2_s2"] == pytest.approx(-1.6e-5)
    assert overall["weighted_mse_delta_m2_s2"] == pytest.approx(overall["mse_delta_m2_s2"])


def test_stationary_statistics_repeat_in_every_window():
    reference = _trace(velocity_along=_alternate(0.03, -0.01), velocity_cross=0.004)
    candidate = _trace(velocity_along=0.02, velocity_cross=_alternate(0.01, -0.01))

    result = analyze_pair(reference, candidate, NORMAL)

    assert len(result["windows"]) == 6
    edges = [(row["start_s"], row["end_s"], row["samples"]) for row in result["windows"]]
    assert edges == [
        (1.5, 2.0, PER_WINDOW),
        (2.0, 2.5, PER_WINDOW),
        (2.5, 3.0, PER_WINDOW),
        (3.0, 3.5, PER_WINDOW),
        (3.5, 4.0, PER_WINDOW),
        (4.0, 4.5, PER_WINDOW),
    ]
    assert sum(row["samples"] for row in result["windows"]) == result["overall"]["samples"]
    for row in result["windows"]:
        assert row["along_mse_6_m2_s2"] == pytest.approx(0.0005)
        assert row["cross_mse_8_m2_s2"] == pytest.approx(0.0001)
        assert row["weighted_mse_delta_m2_s2"] == pytest.approx(row["mse_delta_m2_s2"] / 6.0)


def test_weighted_windows_decompose_overall_delta_across_opposite_signs():
    time = DT * np.arange(SAMPLES)
    reference = _trace(velocity_along=0.03)
    candidate = _trace(velocity_along=np.where(time < 3.0, 0.01, 0.05))

    result = analyze_pair(reference, candidate, NORMAL)
    windows = result["windows"]

    deltas = [row["mse_delta_m2_s2"] for row in windows]
    assert deltas[:3] == pytest.approx([-8e-4] * 3)
    assert deltas[3:] == pytest.approx([1.6e-3] * 3)
    assert result["overall"]["mse_delta_m2_s2"] == pytest.approx(4e-4)
    assert sum(row["weighted_mse_delta_m2_s2"] for row in windows) == pytest.approx(
        result["overall"]["mse_delta_m2_s2"], abs=1e-15
    )
    averaged_rms_delta = float(np.mean([row["velocity_rms_delta_m_s"] for row in windows]))
    assert averaged_rms_delta == pytest.approx(0.0, abs=1e-15)
    assert result["overall"]["velocity_rms_delta_m_s"] == pytest.approx(
        np.sqrt(0.0013) - 0.03
    )
    assert result["overall"]["velocity_rms_delta_m_s"] != pytest.approx(averaged_rms_delta)


def test_identical_traces_report_zero_deltas_and_no_difference_times():
    reference = _trace(velocity_along=0.02, velocity_cross=0.01, position_along_mm=1.5)
    candidate = _trace(velocity_along=0.02, velocity_cross=0.01, position_along_mm=1.5)

    result = analyze_pair(reference, candidate, NORMAL)

    for row in [result["overall"], *result["windows"]]:
        assert row["velocity_rms_delta_m_s"] == pytest.approx(0.0, abs=1e-18)
        assert row["position_rms_delta_mm"] == pytest.approx(0.0, abs=1e-18)
        assert row["mse_delta_m2_s2"] == pytest.approx(0.0, abs=1e-18)
        assert row["weighted_mse_delta_m2_s2"] == pytest.approx(0.0, abs=1e-18)
        assert row["request_delta_vector_rms_n"] == pytest.approx(0.0, abs=1e-18)
        assert row["along_mse_6_m2_s2"] == pytest.approx(row["along_mse_8_m2_s2"])
        assert row["cross_variance_6_m2_s2"] == pytest.approx(row["cross_variance_8_m2_s2"])
    assert result["timing"] == {
        "first_request_difference_s": None,
        "first_velocity_difference_s": None,
        "first_position_difference_s": None,
    }


def test_results_are_invariant_under_3d_rotation():
    reference = _trace(
        velocity_along=_alternate(0.03, -0.01),
        velocity_cross=0.004,
        position_along_mm=2.0,
        position_cross_mm=-1.0,
        request_along=5.0,
        request_cross=0.5,
    )
    candidate = _trace(
        velocity_along=0.02,
        velocity_cross=_alternate(0.01, -0.01),
        position_along_mm=1.0,
        position_cross_mm=1.0,
        request_along=_alternate(6.0, 4.0),
        request_cross=0.5,
    )
    rotation = _rotation([0.3, -0.7, 0.5], 0.9)

    plain = analyze_pair(reference, candidate, NORMAL)
    rotated = analyze_pair(
        _rotate(reference, rotation), _rotate(candidate, rotation), rotation @ NORMAL
    )

    assert rotated["timing"] == plain["timing"]
    for expected, actual in zip([plain["overall"], *plain["windows"]],
                                [rotated["overall"], *rotated["windows"]]):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key


def test_genuine_cross_track_error_lands_in_the_cross_channel():
    reference = _trace()
    candidate = _trace(velocity_cross=0.02, position_cross_mm=4.0)

    overall = analyze_pair(reference, candidate, NORMAL)["overall"]

    assert overall["along_mse_8_m2_s2"] == pytest.approx(0.0, abs=1e-18)
    assert overall["cross_mse_8_m2_s2"] == pytest.approx(4e-4)
    assert overall["cross_mean_8_m_s"] == pytest.approx(0.02)
    assert overall["velocity_rms_8_m_s"] == pytest.approx(0.02)
    assert overall["velocity_rms_6_m_s"] == pytest.approx(0.0, abs=1e-18)
    assert overall["mse_delta_m2_s2"] == pytest.approx(4e-4)
    assert overall["position_along_mean_8_mm"] == pytest.approx(0.0, abs=1e-12)
    assert overall["position_rms_8_mm"] == pytest.approx(4.0)


def test_request_statistics_and_first_difference_times():
    reference = _trace(request_along=5.0, request_cross=0.5)
    candidate = _trace(
        request_along=np.where(np.arange(SAMPLES) < 100, 5.0, _alternate(6.0, 4.0)),
        request_cross=0.5,
        request_normal=_after(100, 7.0),
        velocity_along=_after(200, 0.01),
        position_along_mm=_after(300, 1.0),
    )

    result = analyze_pair(reference, candidate, NORMAL)
    overall = result["overall"]

    assert overall["request_along_mean_6_n"] == pytest.approx(5.0)
    assert overall["request_along_std_6_n"] == pytest.approx(0.0, abs=1e-15)
    assert overall["request_along_mean_8_n"] == pytest.approx(5.0)
    assert overall["request_along_std_8_n"] == pytest.approx(1.0)
    assert overall["request_delta_vector_rms_n"] == pytest.approx(1.0)
    assert result["timing"] == {
        "first_request_difference_s": pytest.approx(0.2),
        "first_velocity_difference_s": pytest.approx(0.4),
        "first_position_difference_s": pytest.approx(0.6),
    }


def test_missing_field_is_rejected():
    reference = _trace()
    candidate = _trace()
    del candidate["target_linear_velocity"]
    with pytest.raises(ValueError, match="missing required fields"):
        analyze_pair(reference, candidate, NORMAL)


def test_non_mapping_trace_is_rejected():
    with pytest.raises(TypeError, match="must be a mapping"):
        analyze_pair([], _trace(), NORMAL)


def test_wrong_vector_shape_is_rejected():
    reference = _trace()
    reference["position"] = reference["position"][:, :2]
    with pytest.raises(ValueError, match="expected"):
        analyze_pair(reference, _trace(), NORMAL)


def test_nonfinite_sample_is_rejected():
    candidate = _trace()
    candidate["linear_velocity"][1200, 1] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        analyze_pair(_trace(), candidate, NORMAL)


def test_nonuniform_time_grid_is_rejected():
    reference = _trace()
    reference["time"] = reference["time"] + np.where(np.arange(SAMPLES) > 500, 0.001, 0.0)
    with pytest.raises(ValueError, match="uniformly sampled"):
        analyze_pair(reference, _trace(), NORMAL)


def test_non_increasing_time_grid_is_rejected():
    reference = _trace()
    reference["time"] = reference["time"][::-1].copy()
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze_pair(reference, _trace(), NORMAL)


def test_mismatched_paired_time_is_rejected():
    candidate = _trace()
    candidate["time"] = candidate["time"] + 0.5
    with pytest.raises(ValueError, match="identical time"):
        analyze_pair(_trace(), candidate, NORMAL)


def test_mismatched_sample_count_is_rejected():
    candidate = _trace()
    for name, value in list(candidate.items()):
        candidate[name] = value[:-1]
    with pytest.raises(ValueError, match="same sample count"):
        analyze_pair(_trace(), candidate, NORMAL)


def test_mismatched_target_is_rejected():
    candidate = _trace()
    candidate["target_position"] = candidate["target_position"] + 0.001
    with pytest.raises(ValueError, match="identical target_position"):
        analyze_pair(_trace(), candidate, NORMAL)


@pytest.mark.parametrize(
    "normal", [np.array([0.0, 0.0, 2.0]), np.zeros(2), np.array([0.0, np.nan, 1.0])]
)
def test_invalid_normal_is_rejected(normal):
    with pytest.raises(ValueError):
        analyze_pair(_trace(), _trace(), normal)


def test_zero_evaluated_target_tangent_speed_is_rejected():
    reference = _trace(target_direction=NORMAL)
    candidate = _trace(target_direction=NORMAL)
    with pytest.raises(ValueError, match="at rest"):
        analyze_pair(reference, candidate, NORMAL)


def test_window_not_dividing_the_interval_is_rejected():
    with pytest.raises(ValueError, match="windows per evaluation interval"):
        analyze_pair(_trace(), _trace(), NORMAL, window_s=0.7)


def test_window_without_whole_samples_is_rejected():
    with pytest.raises(ValueError, match="samples per window"):
        analyze_pair(_trace(), _trace(), NORMAL, window_s=0.003)


def test_interval_beyond_the_trace_is_rejected():
    with pytest.raises(ValueError, match="does not cover"):
        analyze_pair(_trace(), _trace(), NORMAL, start=1.5, end=5.0)


def test_interval_before_the_trace_is_rejected():
    with pytest.raises(ValueError, match="land on a recorded sample"):
        analyze_pair(_trace(), _trace(), NORMAL, start=-0.5, end=1.5)


def test_unaligned_interval_start_is_rejected():
    with pytest.raises(ValueError, match="land on a recorded sample"):
        analyze_pair(_trace(), _trace(), NORMAL, start=1.501, end=4.501)


def test_empty_or_reversed_interval_is_rejected():
    with pytest.raises(ValueError, match="must be greater than start"):
        analyze_pair(_trace(), _trace(), NORMAL, start=4.5, end=1.5)


@pytest.mark.parametrize("bounds", [{"start": float("nan")}, {"end": np.inf}])
def test_nonfinite_interval_bound_is_rejected(bounds):
    with pytest.raises(ValueError, match="finite scalar"):
        analyze_pair(_trace(), _trace(), NORMAL, **bounds)


def test_last_timestamp_at_end_minus_dt_is_accepted():
    reference = _trace()
    assert reference["time"][-1] == pytest.approx(4.5 - DT)
    overall = analyze_pair(reference, _trace(), NORMAL)["overall"]
    assert overall["samples"] == TOTAL
    assert reference["time"][FIRST] == pytest.approx(1.5)
