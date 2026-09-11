"""Extract the causal measured wrench input for the load-aware budget study."""

import numpy as np


def measured_force_input(input_row, sample, frame):
    """Copy and rotate the controller-visible wrench without reading evaluator data."""
    if input_row is None:
        raise ValueError("input row is required")
    names = ("time", "measured_wrench_sample_time", "measured_wrench_world")
    try:
        row_time, measurement_time, measured_wrench = (input_row[name] for name in names)
    except (KeyError, TypeError) as error:
        missing = error.args[0] if isinstance(error, KeyError) and error.args else "required field"
        raise ValueError(f"input row missing required field: {missing}") from error
    try:
        sample_time = float(sample.time)
        sample_measurement_time = float(sample.measured_wrench_sample_time)
        measured_normal_force = float(sample.state.normal_force)
        row_time = float(row_time)
        measurement_time = float(measurement_time)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("sample and row times and normal force must be scalar values") from error
    if not np.all(
        np.isfinite(
            [row_time, measurement_time, sample_time, sample_measurement_time,
             measured_normal_force]
        )
    ):
        raise ValueError("sample and row times and normal force must be finite")
    if not np.isclose(row_time, sample_time, rtol=0.0, atol=1e-12):
        raise ValueError("input row time differs from sample time")
    if not np.isclose(
        measurement_time, sample_measurement_time, rtol=0.0, atol=1e-12
    ):
        raise ValueError("measured wrench time differs from sample timestamp")
    if measurement_time < 0.0 or measurement_time > sample_time:
        raise ValueError("measured wrench time must be within [0, sample time]")

    measured_wrench = np.asarray(measured_wrench, dtype=float)
    if measured_wrench.shape != (6,) or not np.all(np.isfinite(measured_wrench)):
        raise ValueError("measured wrench must be a finite 6-vector")
    wrench_world = measured_wrench.copy()
    wrench_local = frame.wrench_to_local(wrench_world)
    if not np.isclose(
        wrench_local[0], measured_normal_force, rtol=0.0, atol=1e-10
    ):
        raise ValueError("measured wrench normal component differs from sample state")
    return {
        "force_local": wrench_local[:3].copy(),
        "wrench_world": wrench_world,
        "measurement_time_s": measurement_time,
        "measurement_age_s": sample_time - measurement_time,
    }
