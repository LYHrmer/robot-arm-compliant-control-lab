"""Post-reveal event diagnostics for the published Franka v0.5 evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from operator import index as integer_index
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.franka_control import FrankaControllerTelemetrySnapshot
from compliant_control_lab.franka_safety_learning import (
    derive_blind_root,
    sample_blind_scenarios,
)
from compliant_control_lab.franka_simulation import (
    WIPING_START_TIME_S,
    FrankaSimulationConfig,
    FrankaTrialResult,
    run_franka_trial,
)
from compliant_control_lab.published_results_audit import audit_published_results

REPLAY_IDENTITY = "post_reveal_replay:v0.5"
REPLAYED_METRICS = (
    "force_rmse_n",
    "peak_force_n",
    "tangent_rmse_mm",
    "orientation_rmse_deg",
    "contact_ratio_pct",
    "torque_rms_nm",
    "saturation_pct",
)
METRIC_ERROR_FIELDS = tuple(
    f"verified_{metric_name}_abs_error" for metric_name in REPLAYED_METRICS
)
PUBLISHED_CASE_COUNT = 48
PUBLISHED_PEAK_FAILURE_COUNT = 18
PUBLISHED_EARLY_PEAK_FAILURE_COUNT = 7
PUBLISHED_LATE_PEAK_FAILURE_COUNT = 11
PUBLISHED_MINIMUM_PEAK_HEADROOM_NM = 5.08
PEAK_FORCE_GATE_N = 35.0
EARLY_CONTACT_LIMIT_S = 0.5
LATE_CONTACT_LIMIT_S = 1.0
ReplayValue = bool | float | int | str | None
ContactPhase = Literal[
    "pre_contact",
    "raw_contact",
    "confirmed_transition",
    "blended_contact",
]
MotionPhase = Literal["pre_wiping", "wiping"]
CONTACT_PHASES: tuple[ContactPhase, ...] = (
    "pre_contact",
    "raw_contact",
    "confirmed_transition",
    "blended_contact",
)
MOTION_PHASES: tuple[MotionPhase, ...] = ("pre_wiping", "wiping")

EVENT_CSV_FIELDS = (
    "analysis_identity",
    "case_index",
    "case",
    "scenario_seed",
    "simulation_seed",
    "verified_metric_max_abs_error",
    *METRIC_ERROR_FIELDS,
    "frozen_peak_force_n",
    "replay_peak_force_n",
    "frozen_gate_pass",
    "frozen_failed_checks",
    "peak_gate_failed",
    "first_raw_contact_time_s",
    "contact_confirm_time_s",
    "contact_blend_99_time_s",
    "global_peak_time_s",
    "global_peak_after_first_contact_s",
    "global_peak_force_n",
    "global_peak_contact_phase",
    "global_peak_motion_phase",
    "post_contact_peak_time_s",
    "post_contact_peak_force_n",
    "post_contact_peak_contact_phase",
    "post_contact_peak_motion_phase",
    "peak_minimum_torque_headroom_nm",
    "peak_contact_blend",
    "peak_governed_normal_lead_m",
    "peak_torque_projection_scale",
    "peak_linear_velocity_x_m_s",
    "peak_linear_velocity_y_m_s",
    "peak_linear_velocity_z_m_s",
    "peak_target_linear_velocity_x_m_s",
    "peak_target_linear_velocity_y_m_s",
    "peak_target_linear_velocity_z_m_s",
    "peak_commanded_force_x_n",
    "peak_commanded_force_y_n",
    "peak_commanded_force_z_n",
    "peak_commanded_torque_x_nm",
    "peak_commanded_torque_y_nm",
    "peak_commanded_torque_z_nm",
)
PEAK_CONTEXT_CSV_FIELDS = (
    "cohort",
    "n",
    "actual_normal_velocity_median_m_s",
    "actual_normal_velocity_minimum_m_s",
    "actual_normal_velocity_maximum_m_s",
    "target_normal_velocity_max_abs_m_s",
    "commanded_normal_force_median_n",
    "commanded_normal_force_minimum_n",
    "commanded_normal_force_maximum_n",
    "minimum_torque_headroom_minimum_nm",
    "minimum_torque_headroom_median_nm",
    "torque_projection_scale_minimum",
)


def motion_phase_at(time_s: float) -> MotionPhase:
    """Match the frozen target generator, whose 1.2 s boundary sample precedes wiping."""
    return "wiping" if time_s > WIPING_START_TIME_S else "pre_wiping"


@dataclass(frozen=True)
class ContactEventSample:
    """One log row from the frozen simulation loop, without temporal resampling.

    MuJoCo contact and Jacobian caches come from the previous forward evaluation;
    qvel has already been integrated. The returned wrench is this row's next
    command and has not yet been applied. These values are not physically
    simultaneous measurements.
    """

    sample_index: int
    time_s: float
    raw_force_n: float
    contact_phase: ContactPhase
    motion_phase: MotionPhase
    linear_velocity_m_s: tuple[float, float, float]
    target_linear_velocity_m_s: tuple[float, float, float]
    commanded_wrench: tuple[float, float, float, float, float, float]
    minimum_torque_headroom_nm: float
    controller_snapshot: FrankaControllerTelemetrySnapshot


@dataclass(frozen=True)
class ContactEventAnalysis:
    """Contact landmarks and peaks extracted from a single trial.

    ``contact_confirm`` is inferred from the first controller sample whose force blend is
    positive. ``first_raw_contact`` remains the separate physical-contact landmark.
    """

    first_raw_contact: ContactEventSample | None
    contact_confirm: ContactEventSample | None
    contact_blend_99: ContactEventSample | None
    global_peak: ContactEventSample
    peak_after_first_contact: ContactEventSample | None


@dataclass(frozen=True)
class DelayStatistics:
    """Five-number timing summary in seconds."""

    count: int
    minimum_s: float
    p25_s: float
    median_s: float
    p75_s: float
    maximum_s: float


@dataclass(frozen=True)
class EventCohortSummary:
    """Peak timing and phase counts for one explicitly named cohort."""

    name: str
    case_count: int
    delay: DelayStatistics
    early_count: int
    late_count: int
    contact_phase_counts: tuple[tuple[str, int], ...]
    motion_phase_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PeakSampleContext:
    """Controller and motion values logged with failed cases' raw-force peaks.

    Historical ``normal`` field names denote world-x, the controller approach axis.
    They are not projections onto each scenario's yawed wall normal.
    """

    name: str
    case_count: int
    actual_normal_velocity_median_m_s: float
    actual_normal_velocity_minimum_m_s: float
    actual_normal_velocity_maximum_m_s: float
    target_normal_velocity_max_abs_m_s: float
    commanded_normal_force_median_n: float
    commanded_normal_force_minimum_n: float
    commanded_normal_force_maximum_n: float
    minimum_torque_headroom_minimum_nm: float
    minimum_torque_headroom_median_nm: float
    torque_projection_scale_minimum: float


@dataclass(frozen=True)
class ContactEventReplaySummary:
    """Validated facts derived from the 48-case post-reveal replay."""

    case_count: int
    peak_failure_count: int
    all_cases: EventCohortSummary
    peak_failures: EventCohortSummary
    early_peak_failure_context: PeakSampleContext
    late_peak_failure_context: PeakSampleContext
    metric_max_abs_errors: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ContactEventReportPaths:
    """Files written by :func:`generate_contact_event_report`."""

    csv_path: Path
    peak_context_csv_path: Path
    summary_path: Path
    plot_path: Path


def _first_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if indices.size else None


def _contact_phase_at(
    raw_force_n: float,
    contact_blend: float | None,
) -> ContactPhase:
    if raw_force_n <= 0.0:
        return "pre_contact"
    if contact_blend is not None and contact_blend >= 0.99:
        return "blended_contact"
    if contact_blend is not None and contact_blend > 0.0:
        return "confirmed_transition"
    return "raw_contact"


def _validate_event_telemetry(result: FrankaTrialResult) -> int:
    sample_count = len(result.time)
    expected_shapes = {
        "time": (sample_count,),
        "raw_normal_force": (sample_count,),
        "linear_velocity": (sample_count, 3),
        "target_linear_velocity": (sample_count, 3),
        "commanded_wrench": (sample_count, 6),
        "minimum_torque_headroom_nm": (sample_count,),
    }
    for field_name, expected_shape in expected_shapes.items():
        values = np.asarray(getattr(result, field_name))
        if values.shape != expected_shape:
            raise ValueError(f"{field_name} must have shape {expected_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{field_name} must contain only finite values")
    if np.any(np.diff(result.time) < 0.0):
        raise ValueError("trial time must be nondecreasing")
    if len(result.controller_snapshots) != sample_count:
        raise ValueError("controller snapshots must align with trial samples")
    for snapshot in result.controller_snapshots:
        if not isinstance(snapshot, FrankaControllerTelemetrySnapshot):
            raise TypeError("controller snapshots must use FrankaControllerTelemetrySnapshot")
        values = (
            snapshot.contact_blend,
            snapshot.governed_normal_lead_m,
            snapshot.torque_projection_scale,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("controller snapshots must contain finite values when available")
        for field_name, value in (
            ("contact_blend", snapshot.contact_blend),
            ("torque_projection_scale", snapshot.torque_projection_scale),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"snapshot {field_name} must be in [0, 1]")
    return sample_count


def analyze_contact_events(result: FrankaTrialResult) -> ContactEventAnalysis:
    """Extract contact landmarks and peak context from one telemetry-enabled trial."""
    sample_count = _validate_event_telemetry(result)
    if sample_count == 0:
        raise ValueError("cannot analyze an empty Franka trial")

    first_raw_contact_index = _first_index(result.raw_normal_force > 0.0)
    contact_blends = tuple(snapshot.contact_blend for snapshot in result.controller_snapshots)
    contact_confirm_index = next(
        (
            index
            for index, blend in enumerate(contact_blends)
            if blend is not None and blend > 0.0
        ),
        None,
    )
    contact_blend_99_index = next(
        (
            index
            for index, blend in enumerate(contact_blends)
            if blend is not None and blend >= 0.99
        ),
        None,
    )

    def event_at(index: int) -> ContactEventSample:
        return ContactEventSample(
            sample_index=index,
            time_s=float(result.time[index]),
            raw_force_n=float(result.raw_normal_force[index]),
            contact_phase=_contact_phase_at(
                float(result.raw_normal_force[index]),
                result.controller_snapshots[index].contact_blend,
            ),
            motion_phase=motion_phase_at(float(result.time[index])),
            linear_velocity_m_s=tuple(float(value) for value in result.linear_velocity[index]),
            target_linear_velocity_m_s=tuple(
                float(value) for value in result.target_linear_velocity[index]
            ),
            commanded_wrench=tuple(float(value) for value in result.commanded_wrench[index]),
            minimum_torque_headroom_nm=float(result.minimum_torque_headroom_nm[index]),
            controller_snapshot=result.controller_snapshots[index],
        )

    global_peak_index = int(np.argmax(result.raw_normal_force))
    peak_after_first_contact_index = (
        None
        if first_raw_contact_index is None
        else first_raw_contact_index
        + int(np.argmax(result.raw_normal_force[first_raw_contact_index:]))
    )
    return ContactEventAnalysis(
        first_raw_contact=(
            None if first_raw_contact_index is None else event_at(first_raw_contact_index)
        ),
        contact_confirm=None if contact_confirm_index is None else event_at(contact_confirm_index),
        contact_blend_99=(
            None if contact_blend_99_index is None else event_at(contact_blend_99_index)
        ),
        global_peak=event_at(global_peak_index),
        peak_after_first_contact=(
            None
            if peak_after_first_contact_index is None
            else event_at(peak_after_first_contact_index)
        ),
    )


def _read_safe_adaptive_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["method"] == "safe_adaptive_hybrid"
        ]
    return {row["case"]: row for row in rows}


def _verified_metric_error(
    case_name: str,
    replayed_metrics: dict[str, float | str],
    frozen_row: dict[str, str],
) -> dict[str, float]:
    errors = {}
    for metric_name in REPLAYED_METRICS:
        replayed = float(replayed_metrics[metric_name])
        frozen = float(frozen_row[metric_name])
        error = abs(replayed - frozen)
        if not math.isclose(replayed, frozen, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(
                f"post-reveal replay mismatch for {case_name} {metric_name}: "
                f"frozen={frozen}, replayed={replayed}"
            )
        errors[metric_name] = error
    return errors


def _optional_event_time(event: ContactEventSample | None) -> float | None:
    return None if event is None else event.time_s


def _replay_row(
    case_index: int,
    scenario_seed: int,
    simulation_seed: int,
    metrics: dict[str, float | str],
    frozen_row: dict[str, str],
    events: ContactEventAnalysis,
    metric_errors: Mapping[str, float],
) -> dict[str, ReplayValue]:
    peak = events.global_peak
    post_contact_peak = events.peak_after_first_contact
    first_contact_time = _optional_event_time(events.first_raw_contact)
    row: dict[str, ReplayValue] = {
        "analysis_identity": REPLAY_IDENTITY,
        "case_index": case_index,
        "case": str(metrics["scenario"]),
        "scenario_seed": scenario_seed,
        "simulation_seed": simulation_seed,
        "verified_metric_max_abs_error": max(metric_errors.values(), default=0.0),
        **{
            f"verified_{metric_name}_abs_error": metric_errors[metric_name]
            for metric_name in REPLAYED_METRICS
        },
        "frozen_peak_force_n": float(frozen_row["peak_force_n"]),
        "replay_peak_force_n": float(metrics["peak_force_n"]),
        "frozen_gate_pass": frozen_row["gate_pass"],
        "frozen_failed_checks": frozen_row["failed_checks"],
        "peak_gate_failed": "peak_force" in frozen_row["failed_checks"].split(";"),
        "first_raw_contact_time_s": first_contact_time,
        "contact_confirm_time_s": _optional_event_time(events.contact_confirm),
        "contact_blend_99_time_s": _optional_event_time(events.contact_blend_99),
        "global_peak_time_s": peak.time_s,
        "global_peak_after_first_contact_s": (
            None if first_contact_time is None else peak.time_s - first_contact_time
        ),
        "global_peak_force_n": peak.raw_force_n,
        "global_peak_contact_phase": peak.contact_phase,
        "global_peak_motion_phase": peak.motion_phase,
        "post_contact_peak_time_s": _optional_event_time(post_contact_peak),
        "post_contact_peak_force_n": (
            None if post_contact_peak is None else post_contact_peak.raw_force_n
        ),
        "post_contact_peak_contact_phase": (
            None if post_contact_peak is None else post_contact_peak.contact_phase
        ),
        "post_contact_peak_motion_phase": (
            None if post_contact_peak is None else post_contact_peak.motion_phase
        ),
        "peak_minimum_torque_headroom_nm": peak.minimum_torque_headroom_nm,
        "peak_contact_blend": peak.controller_snapshot.contact_blend,
        "peak_governed_normal_lead_m": peak.controller_snapshot.governed_normal_lead_m,
        "peak_torque_projection_scale": peak.controller_snapshot.torque_projection_scale,
    }
    for axis, value in zip("xyz", peak.linear_velocity_m_s, strict=True):
        row[f"peak_linear_velocity_{axis}_m_s"] = value
    for axis, value in zip("xyz", peak.target_linear_velocity_m_s, strict=True):
        row[f"peak_target_linear_velocity_{axis}_m_s"] = value
    for axis, value in zip("xyz", peak.commanded_wrench[:3], strict=True):
        row[f"peak_commanded_force_{axis}_n"] = value
    for axis, value in zip("xyz", peak.commanded_wrench[3:], strict=True):
        row[f"peak_commanded_torque_{axis}_nm"] = value
    return row


def replay_published_safe_adaptive_cases(
    protocol_path: Path | str,
    result_dir: Path | str,
    *,
    case_indices: Iterable[int] | None = None,
) -> list[dict[str, ReplayValue]]:
    """Replay published safe-adaptive cases and return verified event rows in memory.

    With ``case_indices=None`` the function always replays all 48 v0.5 cases. A subset is
    accepted for focused diagnostics and tests, while the complete frozen archive is still
    audited before any simulation runs.
    """
    requested_indices: tuple[int, ...] | None = None
    if case_indices is not None:
        normalized_indices = []
        for value in case_indices:
            if isinstance(value, bool):
                raise TypeError("case indices must be integers, not booleans")
            try:
                normalized_indices.append(integer_index(value))
            except TypeError as error:
                raise TypeError("case indices must be integers") from error
        requested_indices = tuple(normalized_indices)
        if len(set(requested_indices)) != len(requested_indices):
            raise ValueError("case indices must be unique")

    protocol_path = Path(protocol_path)
    result_dir = Path(result_dir)
    audit = audit_published_results(protocol_path, result_dir)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    reveal = json.loads((result_dir / "reveal.json").read_text(encoding="utf-8"))
    blind_root = derive_blind_root(
        str(reveal["protocol_sha256"]),
        str(reveal["beacon"]["randomness"]),
    )
    scenarios, scenario_seeds, simulation_seeds = sample_blind_scenarios(
        blind_root,
        audit.case_count,
    )
    selected_indices: tuple[int, ...]
    if requested_indices is None:
        selected_indices = tuple(range(audit.case_count))
    else:
        selected_indices = requested_indices
    if any(index < 0 or index >= audit.case_count for index in selected_indices):
        raise IndexError("case index is outside the published v0.5 set")

    frozen_rows = _read_safe_adaptive_rows(result_dir / "comparison.csv")
    if len(frozen_rows) != audit.case_count:
        raise ValueError("published comparison must contain one safe-adaptive row per case")
    duration = float(protocol["blind_contract"]["duration_s"])
    rows: list[dict[str, ReplayValue]] = []
    for case_index in selected_indices:
        scenario = scenarios[case_index]
        frozen_row = frozen_rows[scenario.name]
        result = run_franka_trial(
            FrankaSafeAdaptiveController(),
            scenario=scenario,
            config=FrankaSimulationConfig(
                duration=duration,
                seed=simulation_seeds[case_index],
            ),
        )
        metrics = result.metrics()
        metric_errors = _verified_metric_error(
            scenario.name,
            metrics,
            frozen_row,
        )
        rows.append(
            _replay_row(
                case_index,
                scenario_seeds[case_index],
                simulation_seeds[case_index],
                metrics,
                frozen_row,
                analyze_contact_events(result),
                metric_errors,
            )
        )
    return rows


_REQUIRED_FINITE_FIELDS = (
    "verified_metric_max_abs_error",
    *METRIC_ERROR_FIELDS,
    "frozen_peak_force_n",
    "replay_peak_force_n",
    "first_raw_contact_time_s",
    "global_peak_time_s",
    "global_peak_after_first_contact_s",
    "global_peak_force_n",
    "peak_minimum_torque_headroom_nm",
    "peak_linear_velocity_x_m_s",
    "peak_linear_velocity_y_m_s",
    "peak_linear_velocity_z_m_s",
    "peak_target_linear_velocity_x_m_s",
    "peak_target_linear_velocity_y_m_s",
    "peak_target_linear_velocity_z_m_s",
    "peak_commanded_force_x_n",
    "peak_commanded_force_y_n",
    "peak_commanded_force_z_n",
    "peak_commanded_torque_x_nm",
    "peak_commanded_torque_y_nm",
    "peak_commanded_torque_z_nm",
)
_OPTIONAL_FINITE_FIELDS = (
    "contact_confirm_time_s",
    "contact_blend_99_time_s",
    "post_contact_peak_time_s",
    "post_contact_peak_force_n",
    "peak_contact_blend",
    "peak_governed_normal_lead_m",
    "peak_torque_projection_scale",
)
_KNOWN_FAILURES = {
    "force_rmse",
    "contact_ratio",
    "peak_force",
    "tangent_rmse",
    "saturation",
}


def _finite_report_value(row: Mapping[str, ReplayValue], field_name: str) -> float:
    value = row[field_name]
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    return number


def _optional_finite_report_value(
    row: Mapping[str, ReplayValue], field_name: str
) -> float | None:
    if row[field_name] is None:
        return None
    return _finite_report_value(row, field_name)


def _report_integer(row: Mapping[str, ReplayValue], field_name: str) -> int:
    value = row[field_name]
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    try:
        return integer_index(value)
    except TypeError as error:
        raise ValueError(f"{field_name} must be an integer") from error


def _validate_expected_count(value: int, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a positive integer")
    try:
        normalized = integer_index(value)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a positive integer") from error
    if normalized <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return normalized


def _validated_report_rows(
    rows: Iterable[Mapping[str, ReplayValue]],
    *,
    expected_case_count: int,
    expected_peak_failure_count: int,
) -> list[dict[str, ReplayValue]]:
    expected_case_count = _validate_expected_count(
        expected_case_count, "expected_case_count"
    )
    if isinstance(expected_peak_failure_count, bool):
        raise TypeError("expected_peak_failure_count must be an integer")
    try:
        expected_peak_failure_count = integer_index(expected_peak_failure_count)
    except TypeError as error:
        raise ValueError("expected_peak_failure_count must be an integer") from error
    if not 0 < expected_peak_failure_count <= expected_case_count:
        raise ValueError(
            "expected_peak_failure_count must be between one and the case count"
        )

    normalized_rows: list[dict[str, ReplayValue]] = []
    seen_cases: set[str] = set()
    seen_indices: set[int] = set()
    expected_fields = set(EVENT_CSV_FIELDS)
    for row_number, source_row in enumerate(rows, start=1):
        if not isinstance(source_row, Mapping):
            raise TypeError(f"row {row_number} must be a mapping")
        missing = expected_fields - set(source_row)
        unexpected = set(source_row) - expected_fields
        if missing:
            raise ValueError(f"row {row_number} is missing fields: {sorted(missing)}")
        if unexpected:
            raise ValueError(
                f"row {row_number} has unexpected fields: {sorted(unexpected)}"
            )
        row = dict(source_row)
        if row["analysis_identity"] != REPLAY_IDENTITY:
            raise ValueError(f"row {row_number} has the wrong analysis_identity")

        case_index = _report_integer(row, "case_index")
        case = row["case"]
        if not isinstance(case, str) or not case:
            raise ValueError(f"row {row_number} has an invalid case name")
        if case_index in seen_indices:
            raise ValueError(f"duplicate case_index: {case_index}")
        if case in seen_cases:
            raise ValueError(f"duplicate case: {case}")
        seen_indices.add(case_index)
        seen_cases.add(case)
        for seed_field in ("scenario_seed", "simulation_seed"):
            if _report_integer(row, seed_field) < 0:
                raise ValueError(f"{seed_field} must be non-negative")

        for field_name in _REQUIRED_FINITE_FIELDS:
            _finite_report_value(row, field_name)
        for field_name in _OPTIONAL_FINITE_FIELDS:
            _optional_finite_report_value(row, field_name)
        if any(_finite_report_value(row, field) < 0.0 for field in METRIC_ERROR_FIELDS):
            raise ValueError("verified metric errors must be non-negative")
        if any(_finite_report_value(row, field) > 1.0e-12 for field in METRIC_ERROR_FIELDS):
            raise ValueError("verified metric errors exceed the 1e-12 absolute tolerance")
        if _finite_report_value(row, "peak_minimum_torque_headroom_nm") < 0.0:
            raise ValueError("peak_minimum_torque_headroom_nm must be non-negative")
        for field_name in ("peak_contact_blend", "peak_torque_projection_scale"):
            value = _optional_finite_report_value(row, field_name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")

        contact_phase = row["global_peak_contact_phase"]
        motion_phase = row["global_peak_motion_phase"]
        if contact_phase not in CONTACT_PHASES:
            raise ValueError(f"invalid global_peak_contact_phase: {contact_phase!r}")
        if motion_phase not in MOTION_PHASES:
            raise ValueError(f"invalid global_peak_motion_phase: {motion_phase!r}")
        for field_name, allowed_values in (
            ("post_contact_peak_contact_phase", CONTACT_PHASES),
            ("post_contact_peak_motion_phase", MOTION_PHASES),
        ):
            value = row[field_name]
            if value is not None and value not in allowed_values:
                raise ValueError(f"invalid {field_name}: {value!r}")

        if row["frozen_gate_pass"] not in {"yes", "no"}:
            raise ValueError("frozen_gate_pass must be 'yes' or 'no'")
        failures_value = row["frozen_failed_checks"]
        if not isinstance(failures_value, str):
            raise TypeError("frozen_failed_checks must be a semicolon-delimited string")
        failures = [item for item in failures_value.split(";") if item]
        if len(set(failures)) != len(failures) or not set(failures) <= _KNOWN_FAILURES:
            raise ValueError("frozen_failed_checks contains duplicate or unknown checks")
        peak_gate_failed = row["peak_gate_failed"]
        if not isinstance(peak_gate_failed, bool):
            raise TypeError("peak_gate_failed must be a boolean")
        if peak_gate_failed != ("peak_force" in failures):
            raise ValueError("peak_gate_failed disagrees with frozen_failed_checks")
        if (row["frozen_gate_pass"] == "yes") != (not failures):
            raise ValueError("frozen_gate_pass disagrees with frozen_failed_checks")

        frozen_peak = _finite_report_value(row, "frozen_peak_force_n")
        replay_peak = _finite_report_value(row, "replay_peak_force_n")
        global_peak = _finite_report_value(row, "global_peak_force_n")
        peak_error = _finite_report_value(row, "verified_peak_force_n_abs_error")
        if not math.isclose(abs(replay_peak - frozen_peak), peak_error, abs_tol=1.0e-12):
            raise ValueError("peak-force replay error is inconsistent")
        if abs(replay_peak - frozen_peak) > 1.0e-12:
            raise ValueError("peak-force replay exceeds the 1e-12 absolute tolerance")
        if not math.isclose(global_peak, replay_peak, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("global peak does not match the replayed peak metric")
        if peak_gate_failed != (frozen_peak > PEAK_FORCE_GATE_N):
            raise ValueError("peak-force failure flag disagrees with the frozen 35 N gate")

        error_values = [_finite_report_value(row, field) for field in METRIC_ERROR_FIELDS]
        maximum_error = _finite_report_value(row, "verified_metric_max_abs_error")
        if not math.isclose(max(error_values), maximum_error, abs_tol=1.0e-15):
            raise ValueError("verified_metric_max_abs_error is inconsistent")
        first_contact = _finite_report_value(row, "first_raw_contact_time_s")
        peak_time = _finite_report_value(row, "global_peak_time_s")
        peak_delay = _finite_report_value(row, "global_peak_after_first_contact_s")
        if peak_delay < 0.0:
            raise ValueError("global peak cannot precede first raw contact")
        if not math.isclose(peak_time - first_contact, peak_delay, abs_tol=1.0e-12):
            raise ValueError("peak delay is inconsistent with the event times")
        if motion_phase != motion_phase_at(peak_time):
            raise ValueError("global peak motion phase disagrees with its timestamp")
        if contact_phase != _contact_phase_at(
            global_peak, _optional_finite_report_value(row, "peak_contact_blend")
        ):
            raise ValueError("global peak contact phase disagrees with force and blend")
        for field_name, expected_value in (
            ("post_contact_peak_time_s", peak_time),
            ("post_contact_peak_force_n", global_peak),
        ):
            if not math.isclose(
                _finite_report_value(row, field_name), expected_value,
                rel_tol=1.0e-12, abs_tol=1.0e-12,
            ):
                raise ValueError("post-contact peak disagrees with the global contact peak")
        if (
            row["post_contact_peak_contact_phase"] != contact_phase
            or row["post_contact_peak_motion_phase"] != motion_phase
        ):
            raise ValueError("post-contact peak phases disagree with the global contact peak")
        normalized_rows.append(row)

    if len(normalized_rows) != expected_case_count:
        raise ValueError(
            f"expected {expected_case_count} event rows, got {len(normalized_rows)}"
        )
    expected_indices = set(range(expected_case_count))
    if seen_indices != expected_indices:
        missing_indices = sorted(expected_indices - seen_indices)
        extra_indices = sorted(seen_indices - expected_indices)
        raise ValueError(
            "case indices must form one complete zero-based sequence "
            f"(missing={missing_indices}, extra={extra_indices})"
        )
    peak_failure_count = sum(bool(row["peak_gate_failed"]) for row in normalized_rows)
    if peak_failure_count != expected_peak_failure_count:
        raise ValueError(
            f"expected {expected_peak_failure_count} frozen peak-force failures, "
            f"got {peak_failure_count}"
        )
    return sorted(normalized_rows, key=lambda row: _report_integer(row, "case_index"))


def _delay_statistics(rows: list[dict[str, ReplayValue]]) -> DelayStatistics:
    values = np.asarray(
        [_finite_report_value(row, "global_peak_after_first_contact_s") for row in rows],
        dtype=float,
    )
    p25, median, p75 = np.quantile(values, (0.25, 0.5, 0.75))
    return DelayStatistics(
        count=len(rows),
        minimum_s=float(np.min(values)),
        p25_s=float(p25),
        median_s=float(median),
        p75_s=float(p75),
        maximum_s=float(np.max(values)),
    )


def _summarize_cohort(
    name: str, rows: list[dict[str, ReplayValue]]
) -> EventCohortSummary:
    contact_counts = Counter(str(row["global_peak_contact_phase"]) for row in rows)
    motion_counts = Counter(str(row["global_peak_motion_phase"]) for row in rows)
    delays = [
        _finite_report_value(row, "global_peak_after_first_contact_s") for row in rows
    ]
    return EventCohortSummary(
        name=name,
        case_count=len(rows),
        delay=_delay_statistics(rows),
        early_count=sum(delay <= EARLY_CONTACT_LIMIT_S for delay in delays),
        late_count=sum(delay >= LATE_CONTACT_LIMIT_S for delay in delays),
        contact_phase_counts=tuple(
            (phase, contact_counts.get(phase, 0)) for phase in CONTACT_PHASES
        ),
        motion_phase_counts=tuple(
            (phase, motion_counts.get(phase, 0)) for phase in MOTION_PHASES
        ),
    )


def _summarize_peak_sample_context(
    name: str, rows: list[dict[str, ReplayValue]]
) -> PeakSampleContext:
    if not rows:
        raise ValueError(f"{name} peak-failure cohort is empty")
    actual_velocity = np.asarray(
        [_finite_report_value(row, "peak_linear_velocity_x_m_s") for row in rows],
        dtype=float,
    )
    target_velocity = np.asarray(
        [
            _finite_report_value(row, "peak_target_linear_velocity_x_m_s")
            for row in rows
        ],
        dtype=float,
    )
    commanded_force = np.asarray(
        [_finite_report_value(row, "peak_commanded_force_x_n") for row in rows],
        dtype=float,
    )
    torque_headroom = np.asarray(
        [_finite_report_value(row, "peak_minimum_torque_headroom_nm") for row in rows],
        dtype=float,
    )
    projection_scale = np.asarray(
        [_finite_report_value(row, "peak_torque_projection_scale") for row in rows],
        dtype=float,
    )
    return PeakSampleContext(
        name=name,
        case_count=len(rows),
        actual_normal_velocity_median_m_s=float(np.median(actual_velocity)),
        actual_normal_velocity_minimum_m_s=float(np.min(actual_velocity)),
        actual_normal_velocity_maximum_m_s=float(np.max(actual_velocity)),
        target_normal_velocity_max_abs_m_s=float(np.max(np.abs(target_velocity))),
        commanded_normal_force_median_n=float(np.median(commanded_force)),
        commanded_normal_force_minimum_n=float(np.min(commanded_force)),
        commanded_normal_force_maximum_n=float(np.max(commanded_force)),
        minimum_torque_headroom_minimum_nm=float(np.min(torque_headroom)),
        minimum_torque_headroom_median_nm=float(np.median(torque_headroom)),
        torque_projection_scale_minimum=float(np.min(projection_scale)),
    )


def _summarize_validated_rows(
    rows: list[dict[str, ReplayValue]],
) -> ContactEventReplaySummary:
    failures = [row for row in rows if row["peak_gate_failed"] is True]
    early_failures = [
        row
        for row in failures
        if _finite_report_value(row, "global_peak_after_first_contact_s")
        <= EARLY_CONTACT_LIMIT_S
    ]
    late_failures = [
        row
        for row in failures
        if _finite_report_value(row, "global_peak_after_first_contact_s")
        >= LATE_CONTACT_LIMIT_S
    ]
    metric_errors = tuple(
        (
            metric_name,
            max(
                _finite_report_value(
                    row, f"verified_{metric_name}_abs_error"
                )
                for row in rows
            ),
        )
        for metric_name in REPLAYED_METRICS
    )
    return ContactEventReplaySummary(
        case_count=len(rows),
        peak_failure_count=len(failures),
        all_cases=_summarize_cohort("all replayed cases", rows),
        peak_failures=_summarize_cohort("frozen peak-force failures", failures),
        early_peak_failure_context=_summarize_peak_sample_context(
            f"early (<= {EARLY_CONTACT_LIMIT_S:.1f} s)", early_failures
        ),
        late_peak_failure_context=_summarize_peak_sample_context(
            f"late (>= {LATE_CONTACT_LIMIT_S:.1f} s)", late_failures
        ),
        metric_max_abs_errors=metric_errors,
    )


def summarize_contact_event_rows(
    rows: Iterable[Mapping[str, ReplayValue]],
    *,
    expected_case_count: int = PUBLISHED_CASE_COUNT,
    expected_peak_failure_count: int = PUBLISHED_PEAK_FAILURE_COUNT,
) -> ContactEventReplaySummary:
    """Validate replay rows and summarize peak timing for two fixed cohorts."""
    validated = _validated_report_rows(
        rows,
        expected_case_count=expected_case_count,
        expected_peak_failure_count=expected_peak_failure_count,
    )
    return _summarize_validated_rows(validated)


def _assert_published_peak_context(summary: ContactEventReplaySummary) -> None:
    early = summary.early_peak_failure_context
    late = summary.late_peak_failure_context
    if early.case_count != PUBLISHED_EARLY_PEAK_FAILURE_COUNT:
        raise ValueError(
            "published replay must contain exactly 7 early peak-force failures"
        )
    if late.case_count != PUBLISHED_LATE_PEAK_FAILURE_COUNT:
        raise ValueError(
            "published replay must contain exactly 11 late peak-force failures"
        )
    contexts = (early, late)
    if max(context.target_normal_velocity_max_abs_m_s for context in contexts) != 0.0:
        raise ValueError("published peak target world-x velocity must remain zero")
    if min(context.torque_projection_scale_minimum for context in contexts) != 1.0:
        raise ValueError("published peak torque-projection scale must remain one")
    if (
        min(context.minimum_torque_headroom_minimum_nm for context in contexts)
        < PUBLISHED_MINIMUM_PEAK_HEADROOM_NM
    ):
        raise ValueError("published peak torque headroom fell below 5.08 Nm")


def _format_phase_counts(counts: tuple[tuple[str, int], ...]) -> str:
    return ", ".join(f"{name}={count}" for name, count in counts if count)


def _write_event_csv(rows: list[dict[str, ReplayValue]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_peak_context_csv(
    summary: ContactEventReplaySummary, path: Path
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PEAK_CONTEXT_CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for context in (
            summary.early_peak_failure_context,
            summary.late_peak_failure_context,
        ):
            writer.writerow(
                {
                    "cohort": context.name,
                    "n": context.case_count,
                    "actual_normal_velocity_median_m_s": (
                        context.actual_normal_velocity_median_m_s
                    ),
                    "actual_normal_velocity_minimum_m_s": (
                        context.actual_normal_velocity_minimum_m_s
                    ),
                    "actual_normal_velocity_maximum_m_s": (
                        context.actual_normal_velocity_maximum_m_s
                    ),
                    "target_normal_velocity_max_abs_m_s": (
                        context.target_normal_velocity_max_abs_m_s
                    ),
                    "commanded_normal_force_median_n": (
                        context.commanded_normal_force_median_n
                    ),
                    "commanded_normal_force_minimum_n": (
                        context.commanded_normal_force_minimum_n
                    ),
                    "commanded_normal_force_maximum_n": (
                        context.commanded_normal_force_maximum_n
                    ),
                    "minimum_torque_headroom_minimum_nm": (
                        context.minimum_torque_headroom_minimum_nm
                    ),
                    "minimum_torque_headroom_median_nm": (
                        context.minimum_torque_headroom_median_nm
                    ),
                    "torque_projection_scale_minimum": (
                        context.torque_projection_scale_minimum
                    ),
                }
            )


def _write_event_summary(
    summary: ContactEventReplaySummary,
    path: Path,
    *,
    frozen_source_link: str | None,
) -> None:
    lines = [
        "# Safe-adaptive contact-event diagnosis",
        "",
        (
            "This is a post-reveal descriptive replay of the published v0.5 "
            "torque-safe adaptive baseline. It does not alter the frozen first-reveal "
            "result or create new holdout evidence."
        ),
        "",
        "## Replay check",
        "",
        (
            f"All {summary.case_count} published cases were replayed. The table reports "
            "the largest absolute difference from the seven frozen metrics. "
            "The report writer requires each absolute replay error to be <= 1e-12."
        ),
        "",
        "| Frozen metric | Maximum absolute replay error |",
        "|---|---:|",
    ]
    for metric_name, error in summary.metric_max_abs_errors:
        lines.append(f"| `{metric_name}` | {error:.3e} |")

    lines.extend(
        [
            "",
            "## Where the raw-force peak occurred",
            "",
            (
                f"The frozen 35 N gate failed in {summary.peak_failure_count}/"
                f"{summary.case_count} safe-adaptive cases. Delay is measured from the "
                "first sample with positive raw normal force to the trial's global raw-force peak."
            ),
            "",
            (
                "| Cohort | Cases | Delay median [P25, P75] (s) | Delay range (s) | "
                f"Early <= {EARLY_CONTACT_LIMIT_S:.1f} s | Late >= "
                f"{LATE_CONTACT_LIMIT_S:.1f} s |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cohort in (summary.all_cases, summary.peak_failures):
        delay = cohort.delay
        lines.append(
            f"| {cohort.name} | {cohort.case_count} | {delay.median_s:.3f} "
            f"[{delay.p25_s:.3f}, {delay.p75_s:.3f}] | "
            f"{delay.minimum_s:.3f} to {delay.maximum_s:.3f} | "
            f"{cohort.early_count} | {cohort.late_count} |"
        )

    lines.extend(
        [
            "",
            "Peak contact phase:",
            "",
            f"- All cases: {_format_phase_counts(summary.all_cases.contact_phase_counts)}.",
            (
                "- Frozen peak-force failures: "
                f"{_format_phase_counts(summary.peak_failures.contact_phase_counts)}."
            ),
            "",
            "Peak motion phase:",
            "",
            f"- All cases: {_format_phase_counts(summary.all_cases.motion_phase_counts)}.",
            (
                "- Frozen peak-force failures: "
                f"{_format_phase_counts(summary.peak_failures.motion_phase_counts)}."
            ),
            "",
            "## Peak-sample context",
            "",
            (
                "These values share the log row of each failed case's global raw-force peak. "
                "They describe the recorded motion and torque context; they are not "
                "evidence that any listed variable caused the peak."
            ),
            "",
            (
                "The frozen loop reads contact and Jacobian caches from the previous "
                "MuJoCo forward evaluation together with already integrated qvel. "
                "The wrench is the current controller return, before its next mj_step. "
                "The row does not represent strictly synchronized physical measurements."
            ),
            "",
            (
                "Velocity and commanded-force components below use world-x, the "
                "controller approach axis. Yawed walls have a different surface normal. "
                "The `normal` names in peak_context.csv are retained for compatibility "
                "and refer to these world-x components."
            ),
            "",
            (
                f"| Metric | Early <= {EARLY_CONTACT_LIMIT_S:.1f} s "
                f"(n={summary.early_peak_failure_context.case_count}) | "
                f"Late >= {LATE_CONTACT_LIMIT_S:.1f} s "
                f"(n={summary.late_peak_failure_context.case_count}) |"
            ),
            "|---|---:|---:|",
        ]
    )
    context_columns = []
    for context in (
        summary.early_peak_failure_context,
        summary.late_peak_failure_context,
    ):
        context_columns.append(
            (
                (
                    f"{context.actual_normal_velocity_median_m_s:.4f} "
                    f"[{context.actual_normal_velocity_minimum_m_s:.4f}, "
                    f"{context.actual_normal_velocity_maximum_m_s:.4f}]"
                ),
                f"{context.target_normal_velocity_max_abs_m_s:.4f}",
                (
                    f"{context.commanded_normal_force_median_n:.3f} "
                    f"[{context.commanded_normal_force_minimum_n:.3f}, "
                    f"{context.commanded_normal_force_maximum_n:.3f}]"
                ),
                (
                    f"{context.minimum_torque_headroom_minimum_nm:.3f} / "
                    f"{context.minimum_torque_headroom_median_nm:.3f}"
                ),
                f"{context.torque_projection_scale_minimum:.4f}",
            )
        )
    context_labels = (
        "Actual world-x velocity median [range] (m/s)",
        "Target world-x velocity max abs (m/s)",
        "Commanded world-x force median [range] (N)",
        "Minimum torque headroom min / median (Nm)",
        "Torque-projection scale min",
    )
    for label, early_value, late_value in zip(context_labels, *context_columns, strict=True):
        lines.append(f"| {label} | {early_value} | {late_value} |")
    lines.extend(
        [
            "",
            (
                "The per-case event table is [`safe_adaptive_contact_events.csv`]"
                "(safe_adaptive_contact_events.csv). The two-row derived table is "
                "[`peak_context.csv`](peak_context.csv); the plot uses the same "
                "validated case rows."
            ),
        ]
    )
    if frozen_source_link is not None:
        lines.extend(
            [
                "",
                f"Frozen source: [`{frozen_source_link}`]({frozen_source_link}).",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_peak_timing(rows: list[dict[str, ReplayValue]], path: Path) -> None:
    motion_colors = {"pre_wiping": "#2f6690", "wiping": "#e07a3f"}
    contact_markers = {
        "pre_contact": "x",
        "raw_contact": "o",
        "confirmed_transition": "s",
        "blended_contact": "^",
    }
    fig, axis = plt.subplots(figsize=(9.4, 5.8))
    axis.axvspan(
        0.0,
        EARLY_CONTACT_LIMIT_S,
        color="#6baed6",
        alpha=0.09,
        label=f"early <= {EARLY_CONTACT_LIMIT_S:.1f} s",
    )
    axis.axvline(EARLY_CONTACT_LIMIT_S, color="#6c757d", linestyle=":", linewidth=1.0)
    axis.axvline(LATE_CONTACT_LIMIT_S, color="#6c757d", linestyle=":", linewidth=1.0)
    axis.axhline(
        PEAK_FORCE_GATE_N,
        color="#9b2226",
        linestyle="--",
        linewidth=1.4,
        label=f"frozen peak gate: {PEAK_FORCE_GATE_N:.0f} N",
    )
    for row in rows:
        motion_phase = str(row["global_peak_motion_phase"])
        contact_phase = str(row["global_peak_contact_phase"])
        axis.scatter(
            _finite_report_value(row, "global_peak_after_first_contact_s"),
            _finite_report_value(row, "global_peak_force_n"),
            s=58,
            marker=contact_markers[contact_phase],
            facecolor=motion_colors[motion_phase],
            edgecolor="black" if row["peak_gate_failed"] else "white",
            linewidth=1.0 if row["peak_gate_failed"] else 0.55,
            alpha=0.9,
            zorder=3,
        )

    axis.set_xlabel("global peak delay after first raw contact (s)")
    axis.set_ylabel("global raw normal-force peak (N)")
    axis.set_title("Post-reveal peak timing: torque-safe adaptive baseline")
    axis.set_xlim(left=0.0)
    axis.grid(alpha=0.2)
    motion_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=phase.replace("_", " "),
        )
        for phase, color in motion_colors.items()
    ]
    contact_handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            color="#4a4a4a",
            markersize=7,
            label=phase.replace("_", " "),
        )
        for phase, marker in contact_markers.items()
        if any(row["global_peak_contact_phase"] == phase for row in rows)
    ]
    gate_handles, gate_labels = axis.get_legend_handles_labels()
    axis.legend(
        gate_handles + motion_handles + contact_handles,
        gate_labels
        + [handle.get_label() for handle in motion_handles + contact_handles],
        loc="upper right",
        fontsize=8,
        ncol=2,
        framealpha=0.95,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _check_report_destination(
    output_dir: Path, frozen_csv_path: Path | str | None
) -> None:
    source_dirs = {
        Path("results/franka_safety_blind").resolve(),
        (Path(__file__).resolve().parents[2] / "results/franka_safety_blind").resolve(),
    }
    if frozen_csv_path is not None:
        source_dirs.add(Path(frozen_csv_path).resolve().parent)
    targets = [output_dir.resolve()]
    targets.extend(
        (output_dir / name).resolve()
        for name in (
            "safe_adaptive_contact_events.csv",
            "peak_context.csv",
            "summary.md",
            "contact_peak_timing.png",
        )
    )
    if any(
        target == source or source in target.parents
        for target in targets
        for source in source_dirs
    ):
        raise ValueError("post-reveal output must stay outside the first-reveal directory")


def write_contact_event_report(
    rows: Iterable[Mapping[str, ReplayValue]],
    output_dir: Path | str,
    *,
    expected_case_count: int = PUBLISHED_CASE_COUNT,
    expected_peak_failure_count: int = PUBLISHED_PEAK_FAILURE_COUNT,
    frozen_csv_path: Path | str | None = None,
) -> ContactEventReportPaths:
    """Validate replay rows and write outside the frozen archive.

    Every absolute metric error must be <= 1e-12. This conservative absolute-only
    check also covers metrics whose source scales are absent from event rows.
    """
    output_dir = Path(output_dir)
    _check_report_destination(output_dir, frozen_csv_path)
    validated = _validated_report_rows(
        rows,
        expected_case_count=expected_case_count,
        expected_peak_failure_count=expected_peak_failure_count,
    )
    summary = _summarize_validated_rows(validated)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ContactEventReportPaths(
        csv_path=output_dir / "safe_adaptive_contact_events.csv",
        peak_context_csv_path=output_dir / "peak_context.csv",
        summary_path=output_dir / "summary.md",
        plot_path=output_dir / "contact_peak_timing.png",
    )
    _write_event_csv(validated, paths.csv_path)
    _write_peak_context_csv(summary, paths.peak_context_csv_path)
    frozen_source_link = None
    if frozen_csv_path is not None:
        frozen_source_link = Path(
            os.path.relpath(Path(frozen_csv_path).resolve(), output_dir.resolve())
        ).as_posix()
    _write_event_summary(
        summary,
        paths.summary_path,
        frozen_source_link=frozen_source_link,
    )
    _plot_peak_timing(validated, paths.plot_path)
    return paths


def generate_contact_event_report(
    protocol_path: Path | str = Path("results/franka_safety_preholdout/protocol.json"),
    result_dir: Path | str = Path("results/franka_safety_blind"),
    output_dir: Path | str = Path(
        "results/franka_safety_postreveal/contact_events"
    ),
) -> ContactEventReportPaths:
    """Replay all 48 frozen safe-adaptive cases and write post-reveal evidence."""
    protocol_path = Path(protocol_path)
    result_dir = Path(result_dir)
    output_dir = Path(output_dir)
    _check_report_destination(output_dir, result_dir / "comparison.csv")
    rows = replay_published_safe_adaptive_cases(protocol_path, result_dir)
    summary = summarize_contact_event_rows(rows)
    _assert_published_peak_context(summary)
    return write_contact_event_report(
        rows,
        output_dir,
        frozen_csv_path=result_dir / "comparison.csv",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/franka_safety_preholdout/protocol.json"),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/franka_safety_blind"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/franka_safety_postreveal/contact_events"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = generate_contact_event_report(
        protocol_path=args.protocol,
        result_dir=args.result_dir,
        output_dir=args.output,
    )
    print(f"wrote {paths.csv_path}")
    print(f"wrote {paths.peak_context_csv_path}")
    print(f"wrote {paths.summary_path}")
    print(f"wrote {paths.plot_path}")


if __name__ == "__main__":
    main()
