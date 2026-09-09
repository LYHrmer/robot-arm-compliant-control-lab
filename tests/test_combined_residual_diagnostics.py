import numpy as np
import pytest

from tools.combined_residual_diagnostics import analyze_trace

DT = 0.002
SAMPLES = 6000
WINDOW_SAMPLES = 2000
LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


def _frame(yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = np.deg2rad(yaw_deg)
    normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    along = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    return normal, along, np.cross(normal, along)


def _trace(yaw_deg: float = 0.0) -> dict[str, np.ndarray]:
    """Uniform 0-12 s synthetic trace with zero tracking error and steady motion."""
    _, along, _ = _frame(yaw_deg)
    target_velocity = np.tile(0.05 * along, (SAMPLES, 1))
    target_position = np.array([0.4, 0.0, 0.35]) + np.cumsum(target_velocity * DT, axis=0)
    return {
        "time": DT * np.arange(SAMPLES),
        "position": target_position.copy(),
        "target_position": target_position,
        "linear_velocity": target_velocity.copy(),
        "target_linear_velocity": target_velocity,
        "requested_tangential_force_world": np.tile(5.2 * along, (SAMPLES, 1)),
        "true_normal_force": np.full(SAMPLES, 12.0),
        "target_normal_force": np.full(SAMPLES, 12.0),
        "measured_normal_force": np.full(SAMPLES, 12.75),
        "contact_blend": np.ones(SAMPLES),
        "controller_coefficient_before_compute": np.full(SAMPLES, 0.6),
        "controller_coefficient_after_compute": np.full(SAMPLES, 0.6),
        "controller_update_ready_after_compute": np.ones(SAMPLES, dtype=bool),
        "torque_projection_scale": np.ones(SAMPLES),
        "commanded_torque": np.zeros((SAMPLES, 7)),
        "applied_torque": np.zeros((SAMPLES, 7)),
        "lower_torque_limit": np.tile(-LIMITS, (SAMPLES, 1)),
        "upper_torque_limit": np.tile(LIMITS, (SAMPLES, 1)),
        "orientation_error_rad": np.full(SAMPLES, np.deg2rad(0.2)),
    }


def _fill(value) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, dtype=float), (SAMPLES,)).copy()


def _set_error(trace: dict, yaw_deg: float, along_mm, cross_mm, normal_mm=0.0) -> None:
    normal, along, cross = _frame(yaw_deg)
    offset = (
        _fill(along_mm)[:, None] * along
        + _fill(cross_mm)[:, None] * cross
        + _fill(normal_mm)[:, None] * normal
    ) / 1000.0
    trace["position"] = trace["target_position"] + offset


def _stop(trace: dict, start: int, stop: int) -> None:
    trace["target_linear_velocity"] = trace["target_linear_velocity"].copy()
    trace["target_linear_velocity"][start:stop] = 0.0


def test_rotated_geometry_splits_along_cross_and_drops_the_normal_component() -> None:
    trace = _trace(15.0)
    _set_error(trace, 15.0, along_mm=2.0, cross_mm=-1.5, normal_mm=4.0)
    _, control_along, _ = _frame(18.0)
    trace["requested_tangential_force_world"] = np.tile(5.2 * control_along, (SAMPLES, 1))

    result = analyze_trace(trace, surface_yaw_deg=15.0, controller_yaw_error_deg=3.0)

    assert result["sample_count"] == WINDOW_SAMPLES
    assert result["moving_sample_count"] == WINDOW_SAMPLES
    assert result["undefined_direction_sample_count"] == 0
    assert result["tangent_rmse_mm"] == pytest.approx(2.5)
    assert result["along_track_rmse_mm"] == pytest.approx(2.0)
    assert result["cross_track_rmse_mm"] == pytest.approx(1.5)
    assert result["along_track_mean_mm"] == pytest.approx(2.0)
    assert result["window_peak_tangent_error_mm"] == pytest.approx(2.5)
    assert result["tangent_p95_error_mm"] == pytest.approx(2.5)
    assert result["tangent_error_above_threshold_pct"] == pytest.approx(0.0)
    # The requested force lives in the 3 deg misaligned control plane, so a fixed share
    # of it points along the true normal instead of the true tangent plane.
    assert result["window_peak_requested_tangential_force_n"] == pytest.approx(5.2)
    assert result["window_peak_requested_force_true_normal_component_n"] == pytest.approx(
        5.2 * np.sin(np.deg2rad(3.0))
    )


def test_signed_along_track_mean_follows_actual_minus_target() -> None:
    trace = _trace(-15.0)
    _set_error(trace, -15.0, along_mm=-2.5, cross_mm=0.0)

    result = analyze_trace(trace, surface_yaw_deg=-15.0, controller_yaw_error_deg=0.0)

    assert result["along_track_mean_mm"] == pytest.approx(-2.5)
    assert result["along_track_rmse_mm"] == pytest.approx(2.5)
    assert result["cross_track_rmse_mm"] == pytest.approx(0.0)


def test_along_and_cross_energy_identity_on_moving_samples() -> None:
    trace = _trace(-15.0)
    phase = 2.0 * np.pi * DT * np.arange(SAMPLES)
    _set_error(
        trace,
        -15.0,
        along_mm=3.1 * np.sin(7.0 * phase),
        cross_mm=2.3 * np.cos(3.0 * phase) - 0.4,
        normal_mm=5.0 * np.sin(2.0 * phase),
    )
    trace["linear_velocity"] = trace["target_linear_velocity"] + 0.01

    result = analyze_trace(trace, surface_yaw_deg=-15.0, controller_yaw_error_deg=3.0)

    assert result["moving_sample_count"] == WINDOW_SAMPLES
    assert result["moving_tangent_rmse_mm"] == pytest.approx(result["tangent_rmse_mm"], rel=1e-12)
    assert result["along_track_rmse_mm"] ** 2 + result["cross_track_rmse_mm"] ** 2 == pytest.approx(
        result["tangent_rmse_mm"] ** 2, rel=1e-12
    )


def test_stationary_samples_leave_the_split_but_stay_in_the_total() -> None:
    trace = _trace()
    along = np.zeros(SAMPLES)
    cross = np.zeros(SAMPLES)
    along[4000:4500] = 6.0
    cross[4500:6000] = 2.0
    _set_error(trace, 0.0, along_mm=along, cross_mm=cross)
    _stop(trace, 4000, 4500)

    result = analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    assert result["moving_sample_count"] == 1500
    assert result["undefined_direction_sample_count"] == 500
    assert result["sample_count"] == WINDOW_SAMPLES
    assert result["tangent_rmse_mm"] == pytest.approx(np.sqrt((500 * 36.0 + 1500 * 4.0) / 2000))
    assert result["moving_tangent_rmse_mm"] == pytest.approx(2.0)
    assert result["cross_track_rmse_mm"] == pytest.approx(2.0)
    assert result["along_track_mean_mm"] == pytest.approx(0.0)


def test_fully_stationary_window_reports_the_split_as_unavailable() -> None:
    trace = _trace()
    _set_error(trace, 0.0, along_mm=0.0, cross_mm=2.0)
    _stop(trace, 0, SAMPLES)

    result = analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    assert result["moving_sample_count"] == 0
    assert result["undefined_direction_sample_count"] == WINDOW_SAMPLES
    assert result["tangent_rmse_mm"] == pytest.approx(2.0)
    assert result["moving_tangent_rmse_mm"] is None
    assert result["along_track_rmse_mm"] is None
    assert result["cross_track_rmse_mm"] is None
    assert result["along_track_mean_mm"] is None


def test_rolling_windows_stay_inside_the_window_and_do_not_imply_recovery() -> None:
    trace = _trace()
    cross = np.full(SAMPLES, 10.0)
    cross[4050:4175] = 0.0
    cross[5870:6000] = 0.0
    _set_error(trace, 0.0, along_mm=0.0, cross_mm=cross)

    result = analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    # 2000 window samples and a 125-sample rolling window leave 1876 candidate starts;
    # a start at local index 1876 would need a sample at 12.0 s and must not be counted.
    assert result["rolling_window_samples"] == 125
    assert result["rolling_window_count"] == 1876
    # Early qualifying starts 39..61 and trailing qualifying starts 1859..1875.
    assert result["rolling_window_pass_pct"] == pytest.approx(100.0 * 40 / 1876)
    assert result["first_qualifying_window_start_s"] == pytest.approx(39 * DT)
    assert result["longest_qualifying_run_window_count"] == 23
    assert result["longest_qualifying_run_start_span_s"] == pytest.approx(22 * DT)
    # The first qualifying window is not a persistent recovery: most of the window is
    # still above the 3 mm reference level.
    assert result["tangent_error_above_threshold_pct"] == pytest.approx(87.25)
    assert result["tangent_error_above_threshold_fraction"] == pytest.approx(0.8725)
    assert result["tangent_rmse_mm"] == pytest.approx(np.sqrt(1745 * 100.0 / 2000))


def test_window_without_any_qualifying_rolling_window() -> None:
    trace = _trace()
    _set_error(trace, 0.0, along_mm=0.0, cross_mm=9.0)

    result = analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    assert result["rolling_window_pass_pct"] == 0.0
    assert result["first_qualifying_window_start_s"] is None
    assert result["longest_qualifying_run_window_count"] == 0
    assert result["longest_qualifying_run_start_span_s"] == 0.0


def test_optional_observer_telemetry_is_reported_as_unavailable() -> None:
    trace = _trace()
    _set_error(trace, 0.0, along_mm=1.0, cross_mm=0.0)

    result = analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    for key in (
        "compensation_active_pct",
        "amplitude_capped_pct",
        "slew_limited_pct",
        "window_peak_corrected_force_n",
        "corrected_force_mean_n",
    ):
        assert key in result
        assert result[key] is None


def test_amplitude_cap_comes_from_the_flag_not_from_the_requested_force_norm() -> None:
    trace = _trace()
    _set_error(trace, 0.0, along_mm=3.4, cross_mm=0.0)
    trace["diagnostic_compensation_active"] = np.ones(SAMPLES, dtype=bool)
    trace["diagnostic_amplitude_capped"] = np.ones(SAMPLES, dtype=bool)
    trace["diagnostic_slew_limited"] = np.zeros(SAMPLES, dtype=bool)
    trace["diagnostic_corrected_force_n"] = np.full(SAMPLES, 7.4)

    result = analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    # Direction smoothing keeps the request below the 6 N ceiling while the amplitude is
    # capped, so the flag and the norm must be read independently.
    assert result["window_peak_requested_tangential_force_n"] == pytest.approx(5.2)
    assert result["amplitude_capped_pct"] == 100.0
    assert result["compensation_active_pct"] == 100.0
    assert result["slew_limited_pct"] == 0.0
    assert result["window_peak_corrected_force_n"] == pytest.approx(7.4)
    assert result["corrected_force_mean_n"] == pytest.approx(7.4)

    trace["diagnostic_amplitude_capped"] = np.zeros(SAMPLES, dtype=bool)
    lowered = analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)
    assert lowered["amplitude_capped_pct"] == 0.0
    assert lowered["window_peak_requested_tangential_force_n"] == pytest.approx(5.2)


def test_window_physical_contact_and_actuator_metrics() -> None:
    trace = _trace()
    _set_error(trace, 0.0, along_mm=1.0, cross_mm=0.0)
    trace["true_normal_force"][4000:6000] = 12.3
    trace["contact_blend"][4200:4300] = 0.98
    trace["torque_projection_scale"][4000:4004] = 0.97
    trace["commanded_torque"][4000:6000, 4] = 11.5
    trace["commanded_torque"][4000:4010, 0] = 90.0
    trace["applied_torque"] = np.clip(
        trace["commanded_torque"], trace["lower_torque_limit"], trace["upper_torque_limit"]
    )
    trace["controller_coefficient_after_compute"][4000:4100] = 0.6004
    trace["controller_update_ready_after_compute"][5000:6000] = False

    result = analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    assert result["window_peak_true_normal_force_n"] == pytest.approx(12.3)
    assert result["normal_force_error_mean_n"] == pytest.approx(0.3)
    assert result["normal_force_error_rms_n"] == pytest.approx(0.3)
    assert result["measured_minus_true_normal_force_mean_n"] == pytest.approx(0.45)
    assert result["contact_blend_min"] == pytest.approx(0.98)
    assert result["contact_blend_full_pct"] == pytest.approx(95.0)
    assert result["torque_projection_scale_min"] == pytest.approx(0.97)
    assert result["torque_projection_pct"] == pytest.approx(0.2)
    assert result["actuator_clipped_pct"] == pytest.approx(0.5)
    # Joint 0 requests 90 N.m against an 87 N.m limit; joint 4 stays inside the physical
    # envelope but inside the 5% reserve on each side.
    assert result["minimum_torque_headroom_nm"] == pytest.approx(-3.0)
    assert result["minimum_reserved_torque_headroom_nm"] == pytest.approx(-11.7)
    assert result["torque_reserve_fraction_each_side"] == 0.05
    assert result["orientation_error_rms_deg"] == pytest.approx(0.2)
    assert result["window_peak_orientation_error_deg"] == pytest.approx(0.2)
    assert result["update_ready_after_compute_pct"] == pytest.approx(50.0)
    assert result["coefficient_before_compute_max"] == pytest.approx(0.6)
    assert result["coefficient_after_compute_max"] == pytest.approx(0.6004)
    assert result["coefficient_rate_max_abs_per_s"] == pytest.approx(0.2)
    assert result["coefficient_rate_nonzero_pct"] == pytest.approx(5.0)
    assert result["coefficient_rate_mean_per_s"] == pytest.approx(0.01)


def test_metrics_are_plain_scalars_and_echo_the_window() -> None:
    trace = _trace(15.0)
    _set_error(trace, 15.0, along_mm=1.0, cross_mm=1.0)

    result = analyze_trace(
        trace, surface_yaw_deg=15.0, controller_yaw_error_deg=3.0, start_s=8.0, end_s=10.0
    )

    assert result["window_start_s"] == 8.0
    assert result["window_end_s"] == 10.0
    assert result["dt_s"] == pytest.approx(DT)
    assert result["sample_count"] == 1000
    assert result["surface_yaw_deg"] == 15.0
    assert result["controller_yaw_error_deg"] == 3.0
    assert result["tangent_threshold_mm"] == 3.0
    assert result["rolling_window_s"] == 0.25
    for key, value in result.items():
        assert value is None or isinstance(value, (int, float)), key
        assert not isinstance(value, np.generic), key


def test_missing_required_field_is_rejected() -> None:
    trace = _trace()
    del trace["contact_blend"]

    with pytest.raises(ValueError, match="missing required fields"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)


def test_nonfinite_and_misshaped_fields_are_rejected() -> None:
    trace = _trace()
    trace["position"][5000, 1] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    trace = _trace()
    trace["linear_velocity"] = trace["linear_velocity"][:, :2]
    with pytest.raises(ValueError, match="expected"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    trace = _trace()
    trace["contact_blend"] = trace["contact_blend"][:-1]
    with pytest.raises(ValueError, match="expected"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    with pytest.raises(TypeError, match="mapping"):
        analyze_trace([1.0, 2.0], surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)


def test_nonuniform_time_and_uncovered_windows_are_rejected() -> None:
    trace = _trace()
    trace["time"][5000:] += 0.001
    with pytest.raises(ValueError, match="uniform"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    trace = _trace()
    with pytest.raises(ValueError, match="does not cover"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0, end_s=14.0)

    with pytest.raises(ValueError, match="rolling window"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0, end_s=8.1)

    with pytest.raises(ValueError, match="finite scalar"):
        analyze_trace(trace, surface_yaw_deg=np.nan, controller_yaw_error_deg=3.0)


def test_flag_telemetry_must_be_boolean_or_zero_one() -> None:
    trace = _trace()
    trace["diagnostic_amplitude_capped"] = np.ones(SAMPLES)
    with pytest.raises(ValueError, match="boolean telemetry"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    trace = _trace()
    trace["controller_update_ready_after_compute"] = np.full(SAMPLES, 0.5)
    with pytest.raises(ValueError, match="0/1 flag"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)


def test_degenerate_actuator_envelope_is_rejected() -> None:
    trace = _trace()
    trace["upper_torque_limit"] = trace["lower_torque_limit"].copy()

    with pytest.raises(ValueError, match="envelope"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)


def test_contact_ratio_counts_samples_above_the_contact_force_floor() -> None:
    trace = _trace()
    _set_error(trace, 0.0, along_mm=1.0, cross_mm=0.0)
    # Lift off for a quarter of the window; 0.5 N itself is not contact.
    trace["true_normal_force"][4000:4400] = 0.0
    trace["true_normal_force"][4400:4500] = 0.5

    result = analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    assert result["contact_force_floor_n"] == 0.5
    assert result["contact_ratio_pct"] == pytest.approx(75.0)
    assert analyze_trace(_trace(), surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)[
        "contact_ratio_pct"
    ] == pytest.approx(100.0)


def test_fraction_fields_outside_the_unit_interval_are_rejected() -> None:
    trace = _trace()
    trace["contact_blend"][4100] = 1.2
    with pytest.raises(ValueError, match=r"fraction in \[0, 1\]"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    trace = _trace()
    trace["contact_blend"][4100] = -1e-6
    with pytest.raises(ValueError, match=r"fraction in \[0, 1\]"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)

    trace = _trace()
    trace["torque_projection_scale"][10] = 1.5
    with pytest.raises(ValueError, match=r"fraction in \[0, 1\]"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)


def test_negative_optional_corrected_force_is_rejected() -> None:
    trace = _trace()
    trace["diagnostic_corrected_force_n"] = np.full(SAMPLES, 7.4)
    trace["diagnostic_corrected_force_n"][4500] = -0.1

    with pytest.raises(ValueError, match="nonnegative"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)


def test_backward_time_steps_are_rejected_even_when_the_mean_step_is_positive() -> None:
    trace = _trace()
    # Steps alternate +5e-10 and -1e-10: the mean is positive and every step sits within
    # the uniformity tolerance of it, so only an explicit sign check rejects this grid.
    time = 2e-10 * np.arange(SAMPLES)
    time[1::2] += 3e-10
    trace["time"] = time
    assert np.mean(np.diff(time)) > 0.0

    with pytest.raises(ValueError, match="uniform"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0)


def test_invalid_window_bounds_are_rejected() -> None:
    trace = _trace()

    with pytest.raises(ValueError, match="nonnegative"):
        analyze_trace(trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0, start_s=-1.0)

    with pytest.raises(ValueError, match="after window start"):
        analyze_trace(
            trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0, start_s=8.0, end_s=8.0
        )

    with pytest.raises(ValueError, match="after window start"):
        analyze_trace(
            trace, surface_yaw_deg=0.0, controller_yaw_error_deg=3.0, start_s=10.0, end_s=9.0
        )
