"""Pure trajectory metrics for the velocity-time-constant follow-up."""

import numpy as np

TIMESTEP_S = 0.002
ENTRY_SAMPLES = 50


def _trace_array(trace, name, shape):
    try:
        values = np.asarray(trace[name], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid or missing trace field: {name}") from error
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise ValueError(f"trace field must be finite with shape {shape}: {name}")
    return values


def catchup_metrics(trace, normal, start_s=1.5, end_s=4.5):
    """Summarize tangential tracking and first 100 ms catch-up entry in one window."""
    normal = np.asarray(normal, dtype=float)
    if (
        normal.shape != (3,)
        or not np.all(np.isfinite(normal))
        or not np.isclose(normal @ normal, 1.0, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("normal must be a finite unit 3-vector")
    try:
        time = np.asarray(trace["time"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid or missing trace field: time") from error
    if (
        time.ndim != 1
        or not len(time)
        or not np.all(np.isfinite(time))
        or not np.array_equal(time, np.arange(len(time), dtype=float) * TIMESTEP_S)
    ):
        raise ValueError("time must start at zero on the fixed 0.002 s grid")

    shape = (len(time), 3)
    position = _trace_array(trace, "position", shape)
    target_position = _trace_array(trace, "target_position", shape)
    velocity = _trace_array(trace, "linear_velocity", shape)
    target_velocity = _trace_array(trace, "target_linear_velocity", shape)
    try:
        start, end = float(start_s), float(end_s)
    except (TypeError, ValueError) as error:
        raise ValueError("window bounds must be finite") from error
    if not np.all(np.isfinite([start, end])) or start >= end:
        raise ValueError("window bounds must be finite and increasing")
    mask = (time >= start) & (time < end)
    if not np.any(mask):
        raise ValueError("analysis window must contain samples")

    normal_projection = np.outer(target_velocity[mask] @ normal, normal)
    target_tangent = target_velocity[mask] - normal_projection
    target_speed = np.linalg.norm(target_tangent, axis=1)
    if np.any(target_speed <= 1e-12):
        raise ValueError("target tangential speed must exceed 1e-12 m/s")
    direction = target_tangent / target_speed[:, None]

    position_error = target_position[mask] - position[mask]
    position_tangent = position_error - np.outer(position_error @ normal, normal)
    velocity_error = velocity[mask] - target_velocity[mask]
    velocity_tangent = velocity_error - np.outer(velocity_error @ normal, normal)
    lag_mm = 1_000.0 * np.sum(position_error * direction, axis=1)
    within = np.abs(lag_mm) <= 1.0
    entry = None
    if len(within) >= ENTRY_SAMPLES:
        runs = np.convolve(within.astype(int), np.ones(ENTRY_SAMPLES, dtype=int), mode="valid")
        indices = np.flatnonzero(runs == ENTRY_SAMPLES)
        if len(indices):
            entry = float(time[mask][indices[0]])

    return {
        "start_s": start,
        "end_s": end,
        "samples": int(np.count_nonzero(mask)),
        "tangent_rmse_mm": float(
            1_000.0 * np.sqrt(np.mean(np.sum(position_tangent**2, axis=1)))
        ),
        "tangent_velocity_error_rms_m_s": float(
            np.sqrt(np.mean(np.sum(velocity_tangent**2, axis=1)))
        ),
        "positive_along_lag_area_mm_s": float(
            np.sum(np.maximum(lag_mm, 0.0)) * TIMESTEP_S
        ),
        "max_positive_along_lag_mm": float(np.max(np.maximum(lag_mm, 0.0))),
        "first_100ms_within_1mm_s": entry,
    }
