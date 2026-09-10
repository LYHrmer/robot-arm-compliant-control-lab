"""Compare tangential velocity tracking cost between one reference and one candidate trace.

Every returned value is a descriptive statistic of the two supplied traces over the
requested evaluation windows. Nothing here is a gate, nothing is compared against the
frozen acceptance thresholds, and nothing changes a controller, a default or a pass
count. The module only reads traces.

`position` and `linear_velocity` are the pre-control state x[k]; the recorded request is
the actuation u[k] chosen from that state. The reported first-difference timestamps are
therefore descriptions of when the two archived traces first disagree in each field. They
are not a control latency, they do not order cause and effect between the arms, and no
missing cap or observer state is reconstructed from them.
"""

from collections.abc import Mapping

import numpy as np

SPEED_FLOOR_M_S = 1e-12
DIFFERENCE_FLOOR = 1e-12
_TOLERANCE = 1e-9

_TIME_FIELD = "time"
_VECTOR3_FIELDS = (
    "position",
    "target_position",
    "linear_velocity",
    "target_linear_velocity",
    "requested_tangential_force_world",
)
_PAIRED_FIELDS = (_TIME_FIELD, "target_position", "target_linear_velocity")
REQUIRED_FIELDS = (_TIME_FIELD,) + _VECTOR3_FIELDS

_REFERENCE_ARM = "6"
_CANDIDATE_ARM = "8"


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


def _scalar(value, name: str) -> float:
    number = np.asarray(value, dtype=float)
    if number.shape != () or not np.isfinite(number):
        raise ValueError(f"argument must be a finite scalar: {name}")
    return float(number)


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= _TOLERANCE * max(1.0, abs(left), abs(right))


def _validate_trace(trace, label: str) -> tuple[dict[str, np.ndarray], float]:
    if not isinstance(trace, Mapping):
        raise TypeError(f"{label} trace must be a mapping of field name to array")
    missing = sorted(name for name in REQUIRED_FIELDS if name not in trace)
    if missing:
        raise ValueError(f"{label} trace is missing required fields: {', '.join(missing)}")
    time = _finite(trace[_TIME_FIELD], f"{label}.time")
    if time.ndim != 1 or time.size < 2:
        raise ValueError(f"{label} trace time must be a vector with at least two samples")
    steps = np.diff(time)
    dt = float(steps[0])
    if dt <= 0.0 or np.any(steps <= 0.0):
        raise ValueError(f"{label} trace time must be strictly increasing")
    if not np.all(np.abs(steps - dt) <= _TOLERANCE * max(1.0, dt)):
        raise ValueError(f"{label} trace time must be uniformly sampled")
    fields = {_TIME_FIELD: time}
    for name in _VECTOR3_FIELDS:
        fields[name] = _shaped(trace[name], f"{label}.{name}", (int(time.size), 3))
    return fields, dt


def _validate_pair(reference: dict, candidate: dict) -> None:
    if reference[_TIME_FIELD].size != candidate[_TIME_FIELD].size:
        raise ValueError("reference and candidate traces must have the same sample count")
    for name in _PAIRED_FIELDS:
        if not np.array_equal(reference[name], candidate[name]):
            raise ValueError(f"reference and candidate traces must share identical {name}")


def _validate_normal(normal) -> np.ndarray:
    vector = _shaped(normal, "normal", (3,))
    norm = float(np.linalg.norm(vector))
    if not _close(norm, 1.0):
        raise ValueError(f"normal must be a unit vector, got norm {norm}")
    return vector


def _integer_count(value: float, name: str) -> int:
    count = round(value)
    if count < 1 or not _close(value, float(count)):
        raise ValueError(f"{name} must be a positive whole number, got {value}")
    return count


def _evaluation_layout(time: np.ndarray, dt: float, start: float, end: float, window_s: float):
    """Resolve [start, end) into aligned, nonoverlapping, equally sized sample windows."""
    if not end > start:
        raise ValueError(f"evaluation end {end} must be greater than start {start}")
    if window_s <= 0.0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    windows = _integer_count((end - start) / window_s, "windows per evaluation interval")
    per_window = _integer_count(window_s / dt, "samples per window")
    offset = (start - float(time[0])) / dt
    first = round(offset)
    if first < 0 or not _close(offset, float(first)):
        raise ValueError("evaluation start must land on a recorded sample inside the trace")
    total = windows * per_window
    if first + total > int(time.size):
        raise ValueError("trace does not cover the requested evaluation interval")
    return first, per_window, windows, total


def _project(vectors: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vectors - np.outer(vectors @ normal, normal)


def _frame(target_velocity: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample in-plane frame taken from the target velocity shared by both arms."""
    projected = _project(target_velocity, normal)
    speed = np.linalg.norm(projected, axis=1)
    if np.any(speed <= SPEED_FLOOR_M_S):
        raise ValueError(
            "evaluated target tangential speed is at rest; the tangent direction is undefined"
        )
    along = projected / speed[:, None]
    cross = np.cross(np.broadcast_to(normal, along.shape), along)
    return along, cross


def _rms(vectors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(vectors * vectors, axis=1))))


def _components(
    trace: dict[str, np.ndarray],
    window: slice,
    normal: np.ndarray,
    along: np.ndarray,
    cross: np.ndarray,
) -> dict[str, np.ndarray]:
    velocity_error = _project(
        trace["linear_velocity"][window] - trace["target_linear_velocity"][window], normal
    )
    position_error = _project(
        trace["position"][window] - trace["target_position"][window], normal
    )
    request = _project(trace["requested_tangential_force_world"][window], normal)
    return {
        "velocity_error": velocity_error,
        "along_error": np.sum(velocity_error * along, axis=1),
        "cross_error": np.sum(velocity_error * cross, axis=1),
        "position_error": position_error,
        "position_along": np.sum(position_error * along, axis=1),
        "request": request,
        "request_along": np.sum(request * along, axis=1),
    }


def _row(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    span: slice,
    start_s: float,
    end_s: float,
    total: int,
) -> dict:
    arms = {_REFERENCE_ARM: reference, _CANDIDATE_ARM: candidate}
    velocity_rms = {arm: _rms(parts["velocity_error"][span]) for arm, parts in arms.items()}
    along_mse = {
        arm: float(np.mean(parts["along_error"][span] ** 2)) for arm, parts in arms.items()
    }
    cross_mse = {
        arm: float(np.mean(parts["cross_error"][span] ** 2)) for arm, parts in arms.items()
    }
    position_rms = {arm: _rms(parts["position_error"][span]) for arm, parts in arms.items()}
    count = int(span.stop - span.start)
    mse_delta = (along_mse[_CANDIDATE_ARM] + cross_mse[_CANDIDATE_ARM]) - (
        along_mse[_REFERENCE_ARM] + cross_mse[_REFERENCE_ARM]
    )
    row = {
        "start_s": float(start_s),
        "end_s": float(end_s),
        "samples": count,
        "velocity_rms_6_m_s": velocity_rms[_REFERENCE_ARM],
        "velocity_rms_8_m_s": velocity_rms[_CANDIDATE_ARM],
        "velocity_rms_delta_m_s": velocity_rms[_CANDIDATE_ARM] - velocity_rms[_REFERENCE_ARM],
        "along_mse_6_m2_s2": along_mse[_REFERENCE_ARM],
        "along_mse_8_m2_s2": along_mse[_CANDIDATE_ARM],
        "cross_mse_6_m2_s2": cross_mse[_REFERENCE_ARM],
        "cross_mse_8_m2_s2": cross_mse[_CANDIDATE_ARM],
    }
    for key, field in (("along", "along_error"), ("cross", "cross_error")):
        for arm, parts in arms.items():
            row[f"{key}_mean_{arm}_m_s"] = float(np.mean(parts[field][span]))
    for key, field in (("along", "along_error"), ("cross", "cross_error")):
        for arm, parts in arms.items():
            row[f"{key}_variance_{arm}_m2_s2"] = float(np.var(parts[field][span]))
    row["position_rms_6_mm"] = 1000.0 * position_rms[_REFERENCE_ARM]
    row["position_rms_8_mm"] = 1000.0 * position_rms[_CANDIDATE_ARM]
    row["position_rms_delta_mm"] = 1000.0 * (
        position_rms[_CANDIDATE_ARM] - position_rms[_REFERENCE_ARM]
    )
    for arm, parts in arms.items():
        mean_m = float(np.mean(parts["position_along"][span]))
        row[f"position_along_mean_{arm}_mm"] = 1000.0 * mean_m
    for arm, parts in arms.items():
        row[f"request_along_mean_{arm}_n"] = float(np.mean(parts["request_along"][span]))
    for arm, parts in arms.items():
        row[f"request_along_std_{arm}_n"] = float(np.std(parts["request_along"][span]))
    row["request_delta_vector_rms_n"] = _rms(
        candidate["request"][span] - reference["request"][span]
    )
    row["mse_delta_m2_s2"] = mse_delta
    row["weighted_mse_delta_m2_s2"] = (count / total) * mse_delta
    return row


def _first_difference_time(time: np.ndarray, left: np.ndarray, right: np.ndarray):
    """First trace timestamp whose paired vectors differ; descriptive only, not a lag."""
    difference = np.linalg.norm(right - left, axis=1)
    hits = np.flatnonzero(difference > DIFFERENCE_FLOOR)
    if hits.size == 0:
        return None
    return float(time[hits[0]])


def analyze_pair(
    reference: dict,
    candidate: dict,
    normal: np.ndarray,
    *,
    start: float = 1.5,
    end: float = 4.5,
    window_s: float = 0.5,
) -> dict:
    """Report windowed tangential tracking cost for a reference/candidate trace pair.

    Error sign is actual minus target throughout. Both arms are scored in the identical
    per-sample in-plane frame built from the shared target velocity.
    """
    reference_fields, dt = _validate_trace(reference, "reference")
    candidate_fields, candidate_dt = _validate_trace(candidate, "candidate")
    if not _close(dt, candidate_dt):
        raise ValueError("reference and candidate traces must share one sample interval")
    _validate_pair(reference_fields, candidate_fields)
    unit_normal = _validate_normal(normal)
    start = _scalar(start, "start")
    end = _scalar(end, "end")
    window_s = _scalar(window_s, "window_s")
    time = reference_fields[_TIME_FIELD]
    first, per_window, windows, total = _evaluation_layout(time, dt, start, end, window_s)
    evaluated = slice(first, first + total)

    along, cross = _frame(candidate_fields["target_linear_velocity"][evaluated], unit_normal)
    reference_parts = _components(reference_fields, evaluated, unit_normal, along, cross)
    candidate_parts = _components(candidate_fields, evaluated, unit_normal, along, cross)

    overall = _row(reference_parts, candidate_parts, slice(0, total), start, end, total)
    rows = []
    for index in range(windows):
        span = slice(index * per_window, (index + 1) * per_window)
        rows.append(
            _row(
                reference_parts,
                candidate_parts,
                span,
                start + index * window_s,
                start + (index + 1) * window_s,
                total,
            )
        )
    timing = {
        "first_request_difference_s": _first_difference_time(
            time,
            reference_fields["requested_tangential_force_world"],
            candidate_fields["requested_tangential_force_world"],
        ),
        "first_velocity_difference_s": _first_difference_time(
            time, reference_fields["linear_velocity"], candidate_fields["linear_velocity"]
        ),
        "first_position_difference_s": _first_difference_time(
            time, reference_fields["position"], candidate_fields["position"]
        ),
    }
    return {"overall": overall, "windows": rows, "timing": timing}
