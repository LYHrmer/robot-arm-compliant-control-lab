"""Explain the modest_combined 8-12 s tangential residual from one recorded trace.

Every returned value is a diagnostic for the requested window. None of them is a new
gate, none is compared against the frozen acceptance thresholds, and nothing here
changes a controller, a default or a pass count. The module only reads a trace.
"""

from collections.abc import Mapping

import numpy as np

ROLLING_WINDOW_S = 0.25
TANGENT_THRESHOLD_MM = 3.0
SPEED_FLOOR_M_S = 1e-9
TORQUE_RESERVE_FRACTION = 0.05
CONTACT_FORCE_FLOOR_N = 0.5
_TOLERANCE = 1e-9

_TIME_FIELD = "time"
_SCALAR_FIELDS = (
    "true_normal_force",
    "target_normal_force",
    "measured_normal_force",
    "contact_blend",
    "controller_coefficient_before_compute",
    "controller_coefficient_after_compute",
    "torque_projection_scale",
    "orientation_error_rad",
)
_VECTOR3_FIELDS = (
    "position",
    "target_position",
    "linear_velocity",
    "target_linear_velocity",
    "requested_tangential_force_world",
)
_VECTOR7_FIELDS = (
    "commanded_torque",
    "applied_torque",
    "lower_torque_limit",
    "upper_torque_limit",
)
_UNIT_INTERVAL_FIELDS = ("contact_blend", "torque_projection_scale")
_READY_FIELD = "controller_update_ready_after_compute"
REQUIRED_FIELDS = (
    (_TIME_FIELD,) + _SCALAR_FIELDS + _VECTOR3_FIELDS + _VECTOR7_FIELDS + (_READY_FIELD,)
)
OPTIONAL_FLAG_FIELDS = (
    "diagnostic_compensation_active",
    "diagnostic_amplitude_capped",
    "diagnostic_slew_limited",
)
OPTIONAL_FORCE_FIELD = "diagnostic_corrected_force_n"


def _finite(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"trace field is not numeric: {name}")
    array = array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"trace field is not finite: {name}")
    return array


def _shaped(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = _finite(value, name)
    if array.shape != shape:
        raise ValueError(f"trace field has shape {array.shape}, expected {shape}: {name}")
    return array


def _binary(value, name: str, length: int) -> np.ndarray:
    """Accept archived 0/1 telemetry for the required readiness flag."""
    array = _shaped(value, name, (length,))
    if not np.all(np.isin(array, (0.0, 1.0))):
        raise ValueError(f"trace field is not a 0/1 flag: {name}")
    return array.astype(bool)


def _boolean(value, name: str, length: int) -> np.ndarray:
    """Require real bool telemetry for the optional observer flags."""
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise ValueError(f"optional diagnostic field must be boolean telemetry: {name}")
    if array.shape != (length,):
        raise ValueError(f"trace field has shape {array.shape}, expected {(length,)}: {name}")
    return array


def _scalar(value, name: str) -> float:
    number = np.asarray(value, dtype=float)
    if number.shape != () or not np.isfinite(number):
        raise ValueError(f"argument must be a finite scalar: {name}")
    return float(number)


def _validate(trace) -> dict[str, np.ndarray]:
    if not isinstance(trace, Mapping):
        raise TypeError("trace must be a mapping of field name to array")
    missing = sorted(name for name in REQUIRED_FIELDS if name not in trace)
    if missing:
        raise ValueError(f"trace is missing required fields: {', '.join(missing)}")
    time = _finite(trace[_TIME_FIELD], _TIME_FIELD)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("trace time must be a vector with at least two samples")
    length = int(time.size)
    fields = {_TIME_FIELD: time}
    for name in _SCALAR_FIELDS:
        fields[name] = _shaped(trace[name], name, (length,))
    for name in _UNIT_INTERVAL_FIELDS:
        if np.any(fields[name] < 0.0) or np.any(fields[name] > 1.0):
            raise ValueError(f"trace field must be a fraction in [0, 1]: {name}")
    for name in _VECTOR3_FIELDS:
        fields[name] = _shaped(trace[name], name, (length, 3))
    for name in _VECTOR7_FIELDS:
        fields[name] = _shaped(trace[name], name, (length, 7))
    fields[_READY_FIELD] = _binary(trace[_READY_FIELD], _READY_FIELD, length)
    for name in OPTIONAL_FLAG_FIELDS:
        if name in trace:
            fields[name] = _boolean(trace[name], name, length)
    if OPTIONAL_FORCE_FIELD in trace:
        fields[OPTIONAL_FORCE_FIELD] = _shaped(
            trace[OPTIONAL_FORCE_FIELD], OPTIONAL_FORCE_FIELD, (length,)
        )
        if np.any(fields[OPTIONAL_FORCE_FIELD] < 0.0):
            raise ValueError(
                f"optional diagnostic force must be nonnegative: {OPTIONAL_FORCE_FIELD}"
            )
    return fields


def _uniform_dt(time: np.ndarray) -> float:
    steps = np.diff(time)
    if float(np.min(steps)) <= 0.0:
        raise ValueError("trace time must be an increasing uniform grid")
    dt = float(np.mean(steps))
    if dt <= 0 or float(np.max(np.abs(steps - dt))) > _TOLERANCE:
        raise ValueError("trace time must be an increasing uniform grid")
    return dt


def _window_indices(time: np.ndarray, dt: float, start: float, end: float) -> np.ndarray:
    if start < 0.0:
        raise ValueError("window start must be nonnegative")
    if end <= start:
        raise ValueError("window end must be after window start")
    if end - start < ROLLING_WINDOW_S - _TOLERANCE:
        raise ValueError("window must be at least one rolling window long")
    if time[0] > start + _TOLERANCE or time[-1] + dt < end - _TOLERANCE:
        raise ValueError("trace does not cover the requested window")
    indices = np.flatnonzero((time >= start - _TOLERANCE) & (time < end - _TOLERANCE))
    if indices.size == 0:
        raise ValueError("window contains no samples")
    return indices


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def _pct(flags: np.ndarray) -> float:
    return float(100.0 * np.mean(flags))


def _longest_run(flags: np.ndarray) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def analyze_trace(
    trace,
    *,
    surface_yaw_deg,
    controller_yaw_error_deg,
    start_s: float = 8.0,
    end_s: float = 12.0,
) -> dict:
    """Describe the tangential residual on [start_s, end_s) of an in-memory trace.

    Geometry uses the true surface normal from ``surface_yaw_deg``; the along-track
    unit vector is the target velocity projected into that true tangent plane, and
    cross-track is ``true_normal x along``. Position error is actual minus target, so a
    positive ``along_track_mean_mm`` means the tool sits ahead of the target along its
    own direction of travel. Samples whose in-plane target speed is at or below
    ``SPEED_FLOOR_M_S`` have no defined along/cross split and are excluded from that
    split only; the total tangent metrics keep every window sample.

    ``controller_coefficient_before_compute`` produced this cycle's requested force,
    while ``controller_coefficient_after_compute`` is the state carried into the next
    cycle, hence the per-cycle rate ``(after - before) / dt``. Readiness is not next-cycle
    state in the same way: it is computed inside this cycle's ``force()`` and gates this
    cycle's ``advance()``, so ``update_ready_after_compute_pct`` reports how often the
    coefficient update actually ran within the window itself.

    Rolling-window state: a candidate start is a window sample whose ``ROLLING_WINDOW_S``
    block of samples lies wholly inside ``[start_s, end_s)``, so there are
    ``sample_count - rolling_window_samples + 1`` candidates. A candidate qualifies when
    the tangent RMSE of its own block is at or below ``TANGENT_THRESHOLD_MM``, and a run
    is a maximal set of consecutive qualifying candidate starts one ``dt`` apart. Because
    consecutive blocks overlap, neither the first qualifying start nor the longest run
    states that the residual stayed below the threshold afterwards.

    Optional observer fields are reported as ``None`` when the trace does not carry
    them. Amplitude capping and slew limiting are never inferred from the requested
    force norm: direction smoothing keeps that norm below ``max_force`` even when the
    amplitude ceiling is active, and no corrected force is reconstructed from measured
    force.
    """
    fields = _validate(trace)
    yaw = _scalar(surface_yaw_deg, "surface_yaw_deg")
    yaw_error = _scalar(controller_yaw_error_deg, "controller_yaw_error_deg")
    start = _scalar(start_s, "start_s")
    end = _scalar(end_s, "end_s")
    time = fields[_TIME_FIELD]
    dt = _uniform_dt(time)
    window = _window_indices(time, dt, start, end)
    count = int(window.size)
    normal = np.array([np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw)), 0.0])

    error = fields["position"][window] - fields["target_position"][window]
    tangent_error = error - np.outer(error @ normal, normal)
    tangent_norm = np.linalg.norm(tangent_error, axis=1)
    velocity_error = fields["linear_velocity"][window] - fields["target_linear_velocity"][window]
    tangent_velocity = velocity_error - np.outer(velocity_error @ normal, normal)
    target_velocity = fields["target_linear_velocity"][window]
    plane_velocity = target_velocity - np.outer(target_velocity @ normal, normal)
    speed = np.linalg.norm(plane_velocity, axis=1)
    moving = speed > SPEED_FLOOR_M_S
    if np.any(moving):
        along_hat = plane_velocity[moving] / speed[moving, None]
        cross_hat = np.cross(np.broadcast_to(normal, along_hat.shape), along_hat)
        along = np.sum(tangent_error[moving] * along_hat, axis=1)
        cross = np.sum(tangent_error[moving] * cross_hat, axis=1)
        moving_tangent = tangent_norm[moving]
        split = {
            "moving_tangent_rmse_mm": 1000.0 * _rms(moving_tangent),
            "along_track_rmse_mm": 1000.0 * _rms(along),
            "cross_track_rmse_mm": 1000.0 * _rms(cross),
            "along_track_mean_mm": 1000.0 * float(np.mean(along)),
        }
    else:
        split = dict.fromkeys(
            ("moving_tangent_rmse_mm", "along_track_rmse_mm", "cross_track_rmse_mm",
             "along_track_mean_mm")
        )

    span = round(ROLLING_WINDOW_S / dt)
    if span < 1 or span > count:
        raise ValueError("rolling window does not fit inside the requested window")
    cumulative = np.concatenate(([0.0], np.cumsum(tangent_norm**2)))
    rolling_mm = 1000.0 * np.sqrt((cumulative[span:] - cumulative[:-span]) / span)
    qualifying = rolling_mm <= TANGENT_THRESHOLD_MM
    first = np.flatnonzero(qualifying)
    longest = _longest_run(qualifying)

    requested = fields["requested_tangential_force_world"][window]
    requested_norm = np.linalg.norm(requested, axis=1)
    leakage = np.abs(requested @ normal)
    before = fields["controller_coefficient_before_compute"][window]
    after = fields["controller_coefficient_after_compute"][window]
    rate = (after - before) / dt

    true_force = fields["true_normal_force"][window]
    target_force = fields["target_normal_force"][window]
    measured_force = fields["measured_normal_force"][window]
    blend = fields["contact_blend"][window]
    projection = fields["torque_projection_scale"][window]
    commanded = fields["commanded_torque"][window]
    applied = fields["applied_torque"][window]
    lower = fields["lower_torque_limit"][window]
    upper = fields["upper_torque_limit"][window]
    if np.any(upper <= lower):
        raise ValueError("actuator torque envelope is not positive")
    reserve = TORQUE_RESERVE_FRACTION * (upper - lower)
    headroom = np.minimum(commanded - lower, upper - commanded)
    reserved = np.minimum(commanded - lower - reserve, upper - reserve - commanded)
    clipped = np.any(np.abs(commanded - applied) > 1e-9, axis=1)

    metrics = {
        "window_start_s": start,
        "window_end_s": end,
        "dt_s": dt,
        "sample_count": count,
        "moving_sample_count": int(np.count_nonzero(moving)),
        "undefined_direction_sample_count": int(np.count_nonzero(~moving)),
        "surface_yaw_deg": yaw,
        "controller_yaw_error_deg": yaw_error,
        "tangent_rmse_mm": 1000.0 * _rms(tangent_norm),
        "tangent_velocity_error_rms_m_s": _rms(np.linalg.norm(tangent_velocity, axis=1)),
        "tangent_p95_error_mm": 1000.0 * float(np.percentile(tangent_norm, 95.0)),
        "window_peak_tangent_error_mm": 1000.0 * float(np.max(tangent_norm)),
        "tangent_error_above_threshold_fraction": float(
            np.mean(1000.0 * tangent_norm > TANGENT_THRESHOLD_MM)
        ),
        "tangent_error_above_threshold_pct": _pct(1000.0 * tangent_norm > TANGENT_THRESHOLD_MM),
        "tangent_threshold_mm": TANGENT_THRESHOLD_MM,
        **split,
        "rolling_window_s": ROLLING_WINDOW_S,
        "rolling_window_samples": span,
        "rolling_window_count": int(rolling_mm.size),
        "rolling_window_pass_pct": _pct(qualifying),
        "first_qualifying_window_start_s": (
            float(time[window[int(first[0])]] - start) if first.size else None
        ),
        "longest_qualifying_run_window_count": longest,
        "longest_qualifying_run_start_span_s": (longest - 1) * dt if longest else 0.0,
        "update_ready_after_compute_pct": _pct(fields[_READY_FIELD][window]),
        "coefficient_before_compute_min": float(np.min(before)),
        "coefficient_before_compute_max": float(np.max(before)),
        "coefficient_after_compute_min": float(np.min(after)),
        "coefficient_after_compute_max": float(np.max(after)),
        "coefficient_rate_max_abs_per_s": float(np.max(np.abs(rate))),
        "coefficient_rate_mean_per_s": float(np.mean(rate)),
        "coefficient_rate_nonzero_pct": _pct(np.abs(rate) > 1e-12),
        "window_peak_requested_tangential_force_n": float(np.max(requested_norm)),
        "requested_tangential_force_mean_n": float(np.mean(requested_norm)),
        "window_peak_requested_force_true_normal_component_n": float(np.max(leakage)),
        "requested_force_true_normal_component_mean_abs_n": float(np.mean(leakage)),
        "window_peak_true_normal_force_n": float(np.max(true_force)),
        "normal_force_error_rms_n": _rms(true_force - target_force),
        "normal_force_error_mean_n": float(np.mean(true_force - target_force)),
        "measured_minus_true_normal_force_mean_n": float(np.mean(measured_force - true_force)),
        "contact_ratio_pct": _pct(true_force > CONTACT_FORCE_FLOOR_N),
        "contact_force_floor_n": CONTACT_FORCE_FLOOR_N,
        "contact_blend_min": float(np.min(blend)),
        "contact_blend_full_pct": _pct(blend >= 0.99),
        "orientation_error_rms_deg": float(np.rad2deg(_rms(fields["orientation_error_rad"][window]))),
        "window_peak_orientation_error_deg": float(
            np.rad2deg(np.max(fields["orientation_error_rad"][window]))
        ),
        "torque_projection_scale_min": float(np.min(projection)),
        "torque_projection_pct": _pct(projection < 1.0 - 1e-12),
        "actuator_clipped_pct": _pct(clipped),
        "minimum_torque_headroom_nm": float(np.min(headroom)),
        "minimum_reserved_torque_headroom_nm": float(np.min(reserved)),
        "torque_reserve_fraction_each_side": TORQUE_RESERVE_FRACTION,
    }
    for name, key in zip(
        OPTIONAL_FLAG_FIELDS,
        ("compensation_active_pct", "amplitude_capped_pct", "slew_limited_pct"),
    ):
        metrics[key] = _pct(fields[name][window]) if name in fields else None
    corrected = fields.get(OPTIONAL_FORCE_FIELD)
    metrics["window_peak_corrected_force_n"] = (
        float(np.max(corrected[window])) if corrected is not None else None
    )
    metrics["corrected_force_mean_n"] = (
        float(np.mean(corrected[window])) if corrected is not None else None
    )
    return metrics
