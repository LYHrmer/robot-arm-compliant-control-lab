"""Run and independently audit the predeclared measured-budget robustness study.

The 27 public-development trials compare the adaptive 6--8 N scheduler with
fixed 6 N and fixed 8 N controls under the same nine physical/packet scenarios.
Packet faults affect only the scheduler's auxiliary force input.  The original
normal-force feedback channel is unchanged, so this is not a whole-sensor fault
tolerance claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from compliant_control_lab.online_compensation_experiment import (
    DeterministicSchedule,
    ProtocolPhase,
    _case_document,
    _sha256,
    _write_json,
)
from compliant_control_lab.surface_simulation import yaw_frame
from tools import budget_gain_interaction as dynamic
from tools import budget_transfer as archive
from tools import combined_residual_ablation as old
from tools import combined_residual_diagnostics as diagnostics
from tools import cross_surface_regression
from tools import load_budget_screen as screening
from tools import measured_budget_study as measured_study
from tools import measured_budget_trial as runner
from tools import measured_budget_validation as validation

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "measured-load-budget-robustness-v1"
REFERENCE = {
    "directory": "results/franka_measured_budget_full",
    "manifest_sha256": "e4a93740b952707a50e2132dfcf931298e97132edc5fb3a8b0a8cb147b3c8496",
}
METHODS = ("adaptive6_8", "fixed6", "fixed8")
FAULTS = ("fresh", "bias_plus", "bias_minus", "scale_0p8", "scale_1p2", "missing", "stale")
GAIN_SCALE = 1.0
SEED = 11
RECOVERY_WINDOWS = (("packet_recovery", 8.3, 9.0), ("late_recovery", 10.0, 12.0))
OVERALL_DELTA_METRICS = (
    "tangent_rmse_mm",
    "tangent_velocity_error_rms_m_s",
    "force_rmse_n",
    "orientation_rmse_deg",
    "peak_force_n",
    "contact_ratio_pct",
    "saturation_pct",
    "projection_pct",
    "minimum_reserved_torque_headroom_nm",
)
PHASE_DELTA_METRICS = (
    "tangent_rmse_mm",
    "tangent_velocity_error_rms_m_s",
    "force_rmse_n",
    "orientation_rmse_deg",
    "peak_force_n",
    "contact_ratio_pct",
    "saturation_pct",
)
WINDOW_DELTA_METRICS = old.PAIRED_DIAGNOSTICS

# This exact schema distinguishes the full simulator record from the historical
# compact archive.  In particular, all measured kinematic states are retained.
FULL_TRACE_FIELDS = frozenset(
    [
        "applied_raw_wrench_bias_world",
        "applied_tool_friction",
        "applied_torque",
        "applied_wall_friction",
        "cartesian_jacobian",
        "commanded_torque",
        "commanded_wrench",
        "contact_blend",
        "contact_model",
        "controller_coefficient_after_compute",
        "controller_coefficient_before_compute",
        "controller_frame_rotation",
        "controller_kind",
        "controller_update_ready_after_compute",
        "controller_update_ready_before_compute",
        "controller_yaw_error_deg",
        "diagnostic_amplitude_capped",
        "diagnostic_compensation_active",
        "diagnostic_corrected_force_n",
        "diagnostic_slew_limited",
        "dt",
        "fault_end_s",
        "fault_kind",
        "fault_name",
        "fault_start_s",
        "fault_value",
        "feedback_raw_wrench_bias_world",
        "feedback_raw_wrench_world",
        "feedback_wrench_sample_time",
        "filtered_wrench_world",
        "force_filter_alpha",
        "governed_normal_lead_m",
        "joint_torque_offset",
        "joint_velocity",
        "kinematic_sample_time",
        "linear_velocity",
        "load_budget_applied_n",
        "load_budget_next_n",
        "load_budget_updated",
        "load_compensation_force_local",
        "load_estimate_n",
        "load_force_local",
        "load_max_measurement_age_s",
        "load_measurement_age_s",
        "load_measurement_available",
        "load_measurement_time_s",
        "load_packet_status",
        "load_projected_n",
        "load_projection_accepted",
        "lower_torque_limit",
        "max_force_n",
        "measured_angular_velocity",
        "measured_in_contact",
        "measured_kinematic_sample_time",
        "measured_linear_velocity",
        "measured_normal_force",
        "measured_position",
        "measured_rotation",
        "measured_wrench_sample_time",
        "measured_wrench_world",
        "method",
        "minimum_force_n",
        "orientation_error_rad",
        "position",
        "q",
        "raw_load_packet_force",
        "raw_load_packet_present",
        "raw_load_packet_stamp",
        "raw_wrench_world",
        "raw_wrench_sample_time",
        "rotation_gain_scale",
        "schema_version",
        "target_angular_velocity",
        "requested_tangential_force_world",
        "slew_reconstruction_max_error_n",
        "slew_reconstruction_mismatch_cycles",
        "target_linear_velocity",
        "target_normal_force",
        "target_position",
        "target_rotation",
        "time",
        "torque_projection_scale",
        "trace_schema_version",
        "trajectory_rate_scale",
        "true_contact_gap_m",
        "true_normal_force",
        "true_tangent_force_n",
        "upper_torque_limit",
    ]
)
STATIC_TRACE_FIELDS = {
    "contact_model",
    "controller_frame_rotation",
    "controller_kind",
    "dt",
    "fault_end_s",
    "fault_kind",
    "fault_name",
    "fault_start_s",
    "fault_value",
    "force_filter_alpha",
    "load_max_measurement_age_s",
    "max_force_n",
    "method",
    "minimum_force_n",
    "rotation_gain_scale",
    "schema_version",
    "slew_reconstruction_max_error_n",
    "slew_reconstruction_mismatch_cycles",
    "trace_schema_version",
}


def _falling_friction_case():
    originals = [
        case
        for case in cross_surface_regression.make_cases()
        if (
            case.name == "stop_hold_reverse"
            and case.config.seed == SEED
            and case.scenario.wall_yaw_deg == 0
        )
    ]
    if len(originals) != 1:
        raise ValueError("expected one seed-11 stop_hold_reverse case")
    original = originals[0]
    if original.task.yaw_deg != 0 or original.trajectory.name != "stop_hold_reverse":
        raise ValueError("unexpected source stop/reverse case")
    friction = DeterministicSchedule(((0.0, 0.65), (4.0, 0.65), (6.0, 0.30)), "linear")
    phases = (
        ProtocolPhase("high_forward", 1.5, 4.0),
        ProtocolPhase("falling_forward", 4.0, 4.5),
        ProtocolPhase("falling_stopping", 4.5, 5.5),
        ProtocolPhase("falling_hold", 5.5, 6.0),
        ProtocolPhase("low_hold", 6.0, 7.0),
        ProtocolPhase("low_reverse_ramp", 7.0, 8.0),
        ProtocolPhase("low_reverse", 8.0, 12.0),
    )
    return replace(
        original,
        name="falling_friction_stop_hold_reverse",
        scenario=replace(
            original.scenario,
            name="online_error_falling_friction_stop_hold_reverse_seed_11",
            wall_sliding_friction=0.65,
            tool_sliding_friction=0.65,
        ),
        friction=friction,
        phases=phases,
        recovery_start_s=8.0,
    )


def scenarios():
    """Return the frozen nine scenario definitions in execution order."""
    combined = {
        int(case.scenario.wall_yaw_deg): case
        for variant, case in dynamic.cases()
        if variant == "combined"
    }
    if set(combined) != {-15, 0, 15}:
        raise ValueError("combined source cases changed")
    result = [
        {
            "scenario_id": f"yaw0_combined_{fault}",
            "case_kind": "combined",
            "case": combined[0],
            "fault": runner.fault_config(fault),
        }
        for fault in FAULTS
    ]
    result.extend(
        (
            {
                "scenario_id": "yaw15_combined_fresh",
                "case_kind": "combined",
                "case": combined[15],
                "fault": runner.fault_config("fresh"),
            },
            {
                "scenario_id": "yaw0_falling_friction_stop_reverse_fresh",
                "case_kind": "falling_friction_stop_reverse",
                "case": _falling_friction_case(),
                "fault": runner.fault_config("fresh"),
            },
        )
    )
    return result


def specifications():
    return [
        {**scenario, "method": method, "trace_name": f"{scenario['scenario_id']}__{method}.npz"}
        for scenario in scenarios()
        for method in METHODS
    ]


def source_identity():
    sources = measured_study.source_identity()
    for filename in (__file__, runner.__file__, validation.__file__):
        path = Path(filename).resolve()
        sources[path.relative_to(ROOT).as_posix()] = _sha256(path)
    return dict(sorted(sources.items()))


def input_archive():
    root = ROOT / REFERENCE["directory"]
    archive.verify_archive(root, REFERENCE["manifest_sha256"])
    measured_study.audit_archive(root)
    return root


def protocol_document():
    specs = specifications()
    first = specs[0]["case"]
    scenario_documents = []
    for scenario in scenarios():
        scenario_documents.append(
            {
                "scenario_id": scenario["scenario_id"],
                "case_kind": scenario["case_kind"],
                "fault": asdict(scenario["fault"]),
                "case": _case_document(scenario["case"]),
            }
        )
    document = {
        "identity": IDENTITY,
        "reference": REFERENCE,
        "public_development": True,
        "new_holdout": False,
        "default_changed": False,
        "comparison_defined_before_execution": True,
        "new_simulations": 27,
        "scenario_count": 9,
        "methods": {
            name: {
                "minimum_force_n": runner.METHOD_BOUNDS[name][0],
                "maximum_force_n": runner.METHOD_BOUNDS[name][1],
            }
            for name in METHODS
        },
        "gain_scale": GAIN_SCALE,
        "seed": SEED,
        "duration_s": first.config.duration,
        "timestep_s": first.config.timestep,
        "evaluation_start_s": first.config.evaluation_start,
        "maximum_packet_age_s": runner.MAX_MEASUREMENT_AGE_S,
        "scenarios": scenario_documents,
        "recovery_windows": [
            {"name": name, "start_s": start, "end_s": end} for name, start, end in RECOVERY_WINDOWS
        ],
        "overall_delta_metrics": list(OVERALL_DELTA_METRICS),
        "phase_delta_metrics": list(PHASE_DELTA_METRICS),
        "diagnostic_delta_metrics": list(WINDOW_DELTA_METRICS),
        "existing_preset_criteria": dict(screening.CRITERIA),
        "old_absolute_safety_gates": dict(screening.ABSOLUTE),
        "common_cost_limits": {
            "maximum_phase_force_rmse_increase_n": screening.CRITERIA[
                "maximum_phase_force_rmse_increase_n"
            ],
            "maximum_phase_orientation_rmse_increase_deg": screening.CRITERIA[
                "maximum_phase_orientation_rmse_increase_deg"
            ],
            "maximum_full_raw_peak_increase_n": screening.CRITERIA[
                "maximum_full_raw_peak_increase_n"
            ],
        },
        "comparison_rule": (
            "For the eight original combined scenarios, adaptive versus fixed 6 N uses the "
            "unchanged measured-load dynamic criteria. The new falling-friction scenario has "
            "no 20% improvement requirement. Every pair reports unchanged common force, pose "
            "and raw-peak costs; fixed-8 tracking and velocity deltas are descriptive."
        ),
        "paired_prefix_rule": (
            "For each method, every yaw-0 combined packet-fault trace must be bit-exact "
            "with its fresh control on all cycle arrays before the declared fault onset."
        ),
        "full_trace_fields": sorted(FULL_TRACE_FIELDS),
        "limitations": [
            (
                "Auxiliary scheduler-force packet faults only; the normal-control channel is "
                "unchanged."
            ),
            "Public development study, not an unseen holdout and not a default-change decision.",
            "One seed, gain scale 1 and velocity-error time constant 0.05 are covered.",
            (
                "The falling-friction case is a predeclared counterfactual built from the original "
                "stop_hold_reverse trajectory, not a historical archived case."
            ),
            (
                "During the zero-velocity hold, readiness and load updates freeze; the load "
                "filter's time constant advances on eligible updates rather than wall-clock time."
            ),
        ],
    }
    if len(specs) != document["new_simulations"]:
        raise ValueError("protocol/specification count differs")
    return json.loads(json.dumps(document))


def _close(actual, expected, label, atol=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.shape != expected.shape
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(expected))
    ):
        raise ValueError(f"invalid shape/nonfinite: {label}")
    error = float(np.max(np.abs(actual - expected), initial=0.0))
    if error > atol:
        raise ValueError(f"numeric mismatch: {label}")
    return error


def validate_fault_provenance(spec, trace):
    """Rebuild the declared raw packet from the recorded controller-visible wrench."""
    case, config = spec["case"], spec["fault"]
    time = np.asarray(trace.get("time"))
    count = len(time)
    rotation = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg).rotation
    measured = np.asarray(trace.get("measured_wrench_world"))
    stamps = np.asarray(trace.get("measured_wrench_sample_time"))
    target_velocity = np.asarray(trace.get("target_linear_velocity"))
    if (
        measured.shape != (count, 6)
        or stamps.shape != (count,)
        or target_velocity.shape != (count, 3)
    ):
        raise ValueError("invalid source measurement fields for packet provenance")
    source_force = measured[:, :3] @ rotation
    local_velocity = target_velocity @ rotation
    expected_present = np.ones(count, dtype=bool)
    expected_force = source_force.copy()
    expected_stamp = stamps.copy()
    last_force = None
    last_stamp = None
    for index, sample_time in enumerate(time):
        active = config.start_s <= sample_time < config.end_s
        if config.kind == "stale" and active:
            if last_force is None:
                expected_present[index] = False
                expected_force[index] = 0.0
                expected_stamp[index] = 0.0
            else:
                expected_force[index] = last_force
                expected_stamp[index] = last_stamp
            continue
        last_force = source_force[index].copy()
        last_stamp = stamps[index]
        if config.kind == "fresh" or not active:
            continue
        if config.kind == "missing":
            expected_present[index] = False
            expected_force[index] = 0.0
            expected_stamp[index] = 0.0
        elif config.kind == "scale":
            expected_force[index, 1:] *= config.value
        elif config.kind == "bias":
            tangent = local_velocity[index, 1:]
            speed = float(np.linalg.norm(tangent))
            if speed > 0.0:
                expected_force[index, 1:] += config.value * tangent / speed
        else:
            raise ValueError(f"unsupported declared fault kind: {config.kind}")
    present = np.asarray(trace.get("raw_load_packet_present"))
    if (
        present.shape != (count,)
        or present.dtype.kind != "b"
        or not np.array_equal(present, expected_present)
    ):
        raise ValueError("raw packet presence disagrees with declared fault")
    force_error = _close(
        trace.get("raw_load_packet_force"), expected_force, "declared raw packet force", 1e-12
    )
    stamp_error = _close(
        trace.get("raw_load_packet_stamp"), expected_stamp, "declared raw packet stamp", 1e-12
    )
    return {
        "fault_provenance_checked": True,
        "fault_active_cycles": int(
            np.count_nonzero((time >= config.start_s) & (time < config.end_s))
        )
        if config.kind != "fresh"
        else 0,
        "max_raw_packet_force_reconstruction_error_n": force_error,
        "max_raw_packet_stamp_reconstruction_error_s": stamp_error,
    }


def _metadata(spec, trace):
    minimum, maximum = runner.METHOD_BOUNDS[spec["method"]]
    expected = {
        "rotation_gain_scale": GAIN_SCALE,
        "minimum_force_n": minimum,
        "max_force_n": maximum,
        "method": spec["method"],
        "fault_name": spec["fault"].name,
        "fault_kind": spec["fault"].kind,
        "fault_start_s": spec["fault"].start_s,
        "fault_end_s": spec["fault"].end_s,
        "fault_value": spec["fault"].value,
        "load_max_measurement_age_s": runner.MAX_MEASUREMENT_AGE_S,
        "trace_schema_version": 2,
    }
    for name, value in expected.items():
        actual = np.asarray(trace.get(name))
        if actual.shape != () or actual.item() != value:
            raise ValueError(f"trace metadata mismatch: {name}")
    return minimum, maximum


def _validate_full_schema(trace, case):
    if set(trace) != FULL_TRACE_FIELDS:
        raise ValueError("full trace schema mismatch")
    count = round(case.config.duration / case.config.timestep)
    for name, raw in trace.items():
        value = np.asarray(raw)
        if value.dtype.kind not in "biufUS":
            raise ValueError(f"unsupported full trace field: {name}")
        if value.dtype.kind in "biuf" and not np.all(np.isfinite(value)):
            raise ValueError(f"nonfinite full trace field: {name}")
        if value.dtype.kind == "f" and value.dtype != np.dtype("float64"):
            raise ValueError(f"continuous full trace field must be float64: {name}")
        if name not in STATIC_TRACE_FIELDS and (value.ndim == 0 or value.shape[0] != count):
            raise ValueError(f"full trace cycle count mismatch: {name}")
    shapes = {
        "measured_position": (count, 3),
        "measured_linear_velocity": (count, 3),
        "measured_angular_velocity": (count, 3),
        "measured_rotation": (count, 3, 3),
        "position": (count, 3),
        "linear_velocity": (count, 3),
        "q": (count, 7),
        "joint_velocity": (count, 7),
        "cartesian_jacobian": (count, 6, 7),
        "controller_frame_rotation": (3, 3),
    }
    for name, shape in shapes.items():
        if np.asarray(trace[name]).shape != shape:
            raise ValueError(f"invalid full-state shape: {name}")


def _validate_case_schedule(trace, case):
    count = len(trace["time"])
    time = np.arange(count) * case.config.timestep
    _close(trace["time"], time, "time grid", 1e-12)
    for name in ("applied_wall_friction", "applied_tool_friction"):
        _close(trace[name], np.array([case.friction.at(t) for t in time]), name)
    bias = np.array([case.wrench_bias_world.at(t) for t in time])
    _close(trace["applied_raw_wrench_bias_world"], bias, "applied wrench bias")
    _close(
        trace["feedback_raw_wrench_bias_world"],
        np.concatenate((bias[:1], bias[:-1])),
        "feedback wrench bias",
    )
    _close(
        trace["controller_yaw_error_deg"],
        np.full(count, case.controller_yaw_error_deg),
        "controller yaw error",
    )
    _close(
        trace["trajectory_rate_scale"],
        np.array([case.trajectory.rate_scale_at(t) for t in time]),
        "trajectory schedule",
    )
    frame = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg)
    _close(trace["controller_frame_rotation"], frame.rotation, "controller frame", 1e-12)
    limits = np.broadcast_to([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], (count, 7))
    _close(trace["lower_torque_limit"], -limits, "lower torque limits")
    _close(trace["upper_torque_limit"], limits, "upper torque limits")
    _close(
        trace["applied_torque"],
        np.clip(trace["commanded_torque"], -limits, limits),
        "applied torque",
    )
    normal = yaw_frame(case.scenario.wall_yaw_deg).rotation[:, 0]
    gap = (np.array([0.4, 0.0, 0.0]) - trace["position"]) @ normal - 0.025
    _close(trace["true_contact_gap_m"], gap, "contact geometry", 1e-12)


def validate_candidate(spec, trace):
    _validate_full_schema(trace, spec["case"])
    minimum, maximum = _metadata(spec, trace)
    _validate_case_schedule(trace, spec["case"])
    provenance = validate_fault_provenance(spec, trace)
    frame = yaw_frame(spec["case"].task.yaw_deg + spec["case"].controller_yaw_error_deg)
    replay = validation.validate_trace(trace, frame, minimum_force=minimum, max_force=maximum)
    return {**provenance, **replay}


def _recovery_metrics(spec, trace):
    if spec["fault"].kind not in {"missing", "stale"}:
        return []
    case = spec["case"]
    return [
        {
            "window": name,
            **diagnostics.analyze_trace(
                trace,
                surface_yaw_deg=case.scenario.wall_yaw_deg,
                controller_yaw_error_deg=case.controller_yaw_error_deg,
                start_s=start,
                end_s=end,
            ),
        }
        for name, start, end in RECOVERY_WINDOWS
    ]


def _deltas(candidate, reference, names):
    result = {}
    for name in names:
        try:
            first, second = float(candidate[name]), float(reference[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"missing numeric comparison metric: {name}") from error
        if not np.isfinite(first) or not np.isfinite(second):
            raise ValueError(f"nonfinite comparison metric: {name}")
        result[f"delta_{name}"] = first - second
    return result


def _matched_deltas(candidate, reference, identity, names):
    indexed = {row[identity]: row for row in reference}
    if len(indexed) != len(reference):
        raise ValueError(f"duplicate reference {identity}")
    result = []
    for row in candidate:
        key = row[identity]
        if key not in indexed:
            raise ValueError(f"missing matched reference {identity}: {key}")
        result.append({identity: key, **_deltas(row, indexed[key], names)})
    if len(result) != len(reference):
        raise ValueError(f"unmatched {identity} rows")
    return result


def _safety_failures(run):
    failures = []
    overall = screening._absolute(run["overall"], "run")
    if overall:
        failures.append(
            {
                "scope": "overall",
                "phase": None,
                "checks": ";".join(label.removeprefix("run_") for label in overall),
            }
        )
    for row in run["phases"]:
        labels = row["failed_absolute_checks"]
        if labels:
            failures.append({"scope": "phase", "phase": row["phase"], "checks": labels})
    return failures


def _screen_payload(run):
    return {
        "overall": run["overall"],
        "phases": {row["phase"]: row for row in run["phases"]},
        "windows": {row["window"]: row for row in run["diagnostics"]},
    }


def _common_cost_screen(candidate, reference):
    candidate_phases = {row["phase"]: row for row in candidate["phases"]}
    reference_phases = {row["phase"]: row for row in reference["phases"]}
    if set(candidate_phases) != set(reference_phases) or not candidate_phases:
        raise ValueError("common-cost phase identities differ")
    force_cost = max(
        screening._number(candidate_phases[name], "force_rmse_n")
        - screening._number(reference_phases[name], "force_rmse_n")
        for name in candidate_phases
    )
    pose_cost = max(
        screening._number(candidate_phases[name], "orientation_rmse_deg")
        - screening._number(reference_phases[name], "orientation_rmse_deg")
        for name in candidate_phases
    )
    peak_cost = screening._number(candidate["overall"], "peak_force_n") - screening._number(
        reference["overall"], "peak_force_n"
    )
    checks = {
        "phase_force_cost": force_cost <= screening.CRITERIA["maximum_phase_force_rmse_increase_n"],
        "phase_pose_cost": pose_cost
        <= screening.CRITERIA["maximum_phase_orientation_rmse_increase_deg"],
        "full_peak_cost": peak_cost <= screening.CRITERIA["maximum_full_raw_peak_increase_n"],
    }
    failures = [
        *screening._absolute(reference["overall"], "reference"),
        *screening._absolute(candidate["overall"], "candidate"),
        *(name for name, passed in checks.items() if not passed),
    ]
    return {
        "worst_phase_force_rmse_increase_n": force_cost,
        "worst_phase_orientation_rmse_increase_deg": pose_cost,
        "full_raw_peak_increase_n": peak_cost,
        "failed_checks": failures,
        "status": "FAIL" if failures else "PASS",
    }


def _existing_preset_screen(candidate, reference):
    candidate_payload, reference_payload = _screen_payload(candidate), _screen_payload(reference)
    costs = screening._costs(candidate_payload, reference_payload)
    minimum = screening.CRITERIA["minimum_post_tangent_reduction_pct"]
    checks = {
        "post_accuracy": costs["candidate_post_tangent_rmse_mm"]
        <= screening.CRITERIA["maximum_post_tangent_rmse_mm"],
        "post_reduction": 100.0 * costs["candidate_post_tangent_rmse_mm"]
        <= (100.0 - minimum) * costs["reference_post_tangent_rmse_mm"],
        "late_accuracy": costs["candidate_late_tangent_rmse_mm"]
        <= screening.CRITERIA["maximum_late_tangent_rmse_mm"],
        "phase_force_cost": costs["worst_phase_force_rmse_increase_n"]
        <= screening.CRITERIA["maximum_phase_force_rmse_increase_n"],
        "phase_pose_cost": costs["worst_phase_orientation_rmse_increase_deg"]
        <= screening.CRITERIA["maximum_phase_orientation_rmse_increase_deg"],
        "full_peak_cost": costs["full_raw_peak_increase_n"]
        <= screening.CRITERIA["maximum_full_raw_peak_increase_n"],
        "budget_exercised": bool(candidate["budget_exercised"]),
    }
    failures = [
        *screening._absolute(reference["overall"], "reference"),
        *screening._absolute(candidate["overall"], "candidate"),
        *(name for name, passed in checks.items() if not passed),
    ]
    return {
        **costs,
        "failed_checks": failures,
        "status": "FAIL" if failures else "PASS",
    }


def compare_runs(runs):
    indexed = {}
    for row in runs:
        key = (row["scenario_id"], row["method"])
        if key in indexed:
            raise ValueError("duplicate scenario/method result")
        indexed[key] = row
    expected = {(scenario["scenario_id"], method) for scenario in scenarios() for method in METHODS}
    if set(indexed) != expected:
        raise ValueError("incomplete scenario/method results")
    comparisons = []
    for scenario in scenarios():
        scenario_id = scenario["scenario_id"]
        candidate = indexed[scenario_id, "adaptive6_8"]
        for reference_method in ("fixed6", "fixed8"):
            reference = indexed[scenario_id, reference_method]
            common_cost = _common_cost_screen(candidate, reference)
            existing = (
                _existing_preset_screen(candidate, reference)
                if scenario["case_kind"] == "combined" and reference_method == "fixed6"
                else None
            )
            comparisons.append(
                {
                    "scenario_id": scenario_id,
                    "candidate_method": "adaptive6_8",
                    "reference_method": reference_method,
                    "delta_direction": "adaptive6_8_minus_reference",
                    "overall": _deltas(
                        candidate["overall"], reference["overall"], OVERALL_DELTA_METRICS
                    ),
                    "phases": _matched_deltas(
                        candidate["phases"], reference["phases"], "phase", PHASE_DELTA_METRICS
                    ),
                    "diagnostics": _matched_deltas(
                        candidate["diagnostics"],
                        reference["diagnostics"],
                        "window",
                        WINDOW_DELTA_METRICS,
                    ),
                    "recovery": _matched_deltas(
                        candidate["recovery"], reference["recovery"], "window", WINDOW_DELTA_METRICS
                    ),
                    "candidate_safety_failures": _safety_failures(candidate),
                    "reference_safety_failures": _safety_failures(reference),
                    "common_cost_checks": common_cost,
                    "existing_preset_criteria": existing,
                    "existing_preset_criteria_applicable": existing is not None,
                    "eligible_for_existing_preset_criteria": bool(
                        existing is not None and existing["status"] == "PASS"
                    ),
                    "fixed8_tracking_velocity_gate_status": (
                        "descriptive" if reference_method == "fixed8" else "not_applicable"
                    ),
                }
            )
    return comparisons


def _safety_summary(runs):
    failures = []
    for run in runs:
        for failure in _safety_failures(run):
            failures.append({"scenario_id": run["scenario_id"], "method": run["method"], **failure})
    return {
        "old_absolute_gates": protocol_document()["old_absolute_safety_gates"],
        "failed_checks": failures,
        "observed_status": "FAIL" if failures else "PASS",
        "existing_preset_criteria": dict(screening.CRITERIA),
        "eligible_for_default_change": False,
    }


def _load_trace(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def _save_trace(path, trace):
    with path.open("xb") as handle:
        np.savez_compressed(handle, **{name: np.asarray(value) for name, value in trace.items()})


def unchanged_prefix(fresh, faulted, before_s):
    if set(fresh) != set(faulted):
        raise ValueError("paired prefix trace schemas differ")
    count = len(fresh["time"])
    mask = np.asarray(fresh["time"]) < before_s
    fields = []
    for name, expected in fresh.items():
        expected, actual = np.asarray(expected), np.asarray(faulted[name])
        if expected.ndim and expected.shape[0] == count:
            if (
                actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or not np.array_equal(actual[mask], expected[mask])
            ):
                raise ValueError(f"pre-fault prefix differs: {name}")
            fields.append(name)
    return {
        "before_s": before_s,
        "samples": int(np.count_nonzero(mask)),
        "fields": sorted(fields),
        "bit_exact": True,
    }


def collect(directory, *, execute):
    input_archive()
    runs = []
    fresh_controls = {}
    prefixes = []
    for index, spec in enumerate(specifications(), start=1):
        path = Path(directory) / "traces" / spec["trace_name"]
        if execute:
            trace = runner.run_dynamic(
                spec["case"], GAIN_SCALE, spec["method"], spec["fault"]
            ).trace
            _save_trace(path, trace)
            print(f"executed {index}/27 {spec['trace_name']}", flush=True)
        else:
            trace = _load_trace(path)
        checks = validate_candidate(spec, trace)
        if spec["scenario_id"] == "yaw0_combined_fresh":
            fresh_controls[spec["method"]] = trace
        elif spec["case_kind"] == "combined" and spec["case"].scenario.wall_yaw_deg == 0:
            prefix = unchanged_prefix(fresh_controls[spec["method"]], trace, spec["fault"].start_s)
            prefixes.append(
                {
                    "scenario_id": spec["scenario_id"],
                    "method": spec["method"],
                    **prefix,
                }
            )
        minimum, maximum = runner.METHOD_BOUNDS[spec["method"]]
        overall, phases, windows = dynamic.metrics(trace, spec["case"], maximum)
        runs.append(
            {
                "scenario_id": spec["scenario_id"],
                "case_kind": spec["case_kind"],
                "surface_yaw_deg": int(spec["case"].scenario.wall_yaw_deg),
                "fault": spec["fault"].name,
                "method": spec["method"],
                "minimum_force_n": minimum,
                "maximum_force_n": maximum,
                "trace_path": f"traces/{spec['trace_name']}",
                "overall": overall,
                "phases": phases,
                "diagnostics": windows,
                "recovery": _recovery_metrics(spec, trace),
                "audit": checks,
                "budget_exercised": bool(
                    np.any(
                        (trace["load_budget_applied_n"] > 6.0 + 1e-9)
                        & (
                            trace["controller_coefficient_before_compute"]
                            * trace["diagnostic_corrected_force_n"]
                            > 6.0 + 1e-9
                        )
                    )
                ),
            }
        )
    if len(prefixes) != 18:
        raise ValueError("incomplete pre-fault prefix checks")
    return {
        "runs": runs,
        "comparisons": compare_runs(runs),
        "prefixes": prefixes,
        "safety": _safety_summary(runs),
    }


def expected_artifacts():
    return {"protocol.json", "source_hashes.json", "comparison.json"} | {
        f"traces/{spec['trace_name']}" for spec in specifications()
    }


def audit_archive(directory):
    directory = Path(directory).absolute()
    manifest = archive.verify_archive(directory)
    if (
        manifest.get("identity") != IDENTITY
        or manifest.get("reference") != REFERENCE
        or manifest.get("new_simulations") != 27
        or set(manifest.get("artifact_sha256", {})) != expected_artifacts()
    ):
        raise ValueError("robustness archive identity, counts or inventory differs")
    for name, expected in (
        ("protocol.json", protocol_document()),
        ("source_hashes.json", source_identity()),
    ):
        if json.loads((directory / name).read_text()) != expected:
            raise ValueError(f"frozen input differs: {name}")
    regenerated = collect(directory, execute=False)
    if json.loads((directory / "comparison.json").read_text()) != regenerated:
        raise ValueError("recomputed robustness comparison differs")
    return regenerated["safety"]


def run(directory):
    destination = Path(directory).absolute()
    if destination.exists() or any(
        path.is_symlink() for path in (destination, *destination.parents)
    ):
        raise ValueError("output must not exist or traverse a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    print(f"staging {staging}; incomplete runs are retained here", flush=True)
    (staging / "traces").mkdir()
    sources = source_identity()
    protocol = protocol_document()
    input_archive()
    _write_json(staging / "protocol.json", protocol)
    _write_json(staging / "source_hashes.json", sources)
    result = collect(staging, execute=True)
    if source_identity() != sources or protocol_document() != protocol:
        raise ValueError("source or protocol changed during execution")
    input_archive()
    _write_json(staging / "comparison.json", result)
    artifacts = {name: _sha256(staging / name) for name in sorted(expected_artifacts())}
    _write_json(
        staging / "manifest.json",
        {
            "identity": IDENTITY,
            "reference": REFERENCE,
            "new_simulations": 27,
            "artifact_sha256": artifacts,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "mujoco": mujoco.__version__,
            },
            "default_changed": False,
            "new_holdout": False,
        },
    )
    (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
    audited = audit_archive(staging)
    if destination.exists():
        raise ValueError("output appeared during execution; staging retained")
    staging.rename(destination)
    return audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "audit"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = run(args.directory) if args.action == "run" else audit_archive(args.directory)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
