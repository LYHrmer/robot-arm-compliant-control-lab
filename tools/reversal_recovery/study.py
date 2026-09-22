"""Eight predeclared public-development trials of a next-cycle hold-cap rule.

This experiment deliberately uses frozen runner construction seams. Its numerical
audit replays compensation state, not MuJoCo dynamics or the whole normal controller.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import tempfile
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from compliant_control_lab.online_compensation_experiment import (
    DeterministicSchedule,
    ProtocolPhase,
    ScheduledSurfaceSimulator,
    _case_document,
    _sha256,
    _write_json,
)
from compliant_control_lab.surface_simulation import yaw_frame
from tools import budget_gain_interaction as dynamic
from tools import combined_residual_ablation as old
from tools import load_budget_trial as base
from tools import measured_budget_robustness as previous
from tools import measured_budget_trial as runner
from tools import measured_budget_validation as packet_audit
from tools.load_aware_compensation import LoadAwareCompensation
from tools.reversal_recovery.controller import HoldCapTracking

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")
SCHEMA = 3
# The eight original traces completed at this exact revision. That runner compared
# tuple-containing case metadata directly with JSON lists and stopped before rename.
# This sole compatibility path changes only summary serialization/finalization;
# the recorded execution identity and every other input hash remain untouched.
SERIALIZATION_FIX_EXECUTION_COMMIT = "64176beffd1a21b0d86b14a92254cc424f7a2674"
SERIALIZATION_FIX_STUDY_SHA256 = "9954d696d107886a23fa01f5605d54264abcb364b5a39be1234dd4601562f54e"
STUDY_SOURCE_PATH = "tools/reversal_recovery/study.py"


def protocol_document():
    return json.loads(PROTOCOL.read_text())


def specifications(protocol):
    original = previous._falling_friction_case()
    specs = []
    high_names = ("high_forward", "high_forward_before_stop", "high_stopping",
                  "high_hold_initial", "high_hold", "high_reverse_ramp", "high_reverse")
    for scenario in protocol["scenario_names"]:
        phases = original.phases if scenario == "falling" else tuple(
            ProtocolPhase(name, phase.start_s, phase.end_s)
            for name, phase in zip(high_names, original.phases, strict=True)
        )
        friction = DeterministicSchedule(
            tuple(tuple(point) for point in protocol["friction_schedules"][scenario]), "linear",
        )
        for seed in protocol["seeds"]:
            case = replace(
                original, name=f"{scenario}_stop_hold_reverse",
                config=replace(original.config, seed=seed, duration=protocol["duration_s"],
                               timestep=protocol["timestep_s"]),
                scenario=replace(original.scenario, name=f"{scenario}_stop_hold_reverse_seed_{seed}"),
                phases=phases, friction=friction,
            )
            if case.scenario.wall_yaw_deg != protocol["surface_yaw_deg"]:
                raise ValueError("source case yaw differs from protocol")
            for method in protocol["methods"]:
                if method not in {"adaptive6_8", "hold_cap_tracking"}:
                    raise ValueError("unknown recovery method")
                specs.append({"scenario": scenario, "seed": seed, "method": method, "case": case,
                              "trace_path": f"traces/{scenario}_seed{seed}__{method}.npz"})
    if len(specs) != protocol["maximum_new_simulations"]:
        raise ValueError("simulation count differs from frozen protocol")
    return specs


def make_controller(spec, protocol):
    low, high = runner.METHOD_BOUNDS["adaptive6_8"]
    controller = base._load_aware(
        dynamic.controller(spec["case"], protocol["rotation_gain_scale"], high), low, high,
    )
    original = controller._base.tangential
    parameters = {field.name: getattr(original, field.name)
                  for field in fields(LoadAwareCompensation) if field.init}
    if parameters["velocity_error_time"] != protocol["velocity_error_time_s"]:
        raise ValueError("source controller velocity-error time differs from protocol")
    if spec["method"] == "hold_cap_tracking":
        controller._base.tangential = HoldCapTracking(**parameters)
    return controller, parameters


def replay_compensation(trace, parameters, method, dt):
    """Independent equations: never call either compensation implementation."""
    p = parameters
    time = np.asarray(trace["time"])
    count = len(time)
    close = previous._close
    close(time, np.arange(count) * dt, "time grid", 1e-12)
    close(trace["dt"], np.asarray(dt), "dt", 1e-12)
    force, available, _, packet_error = packet_audit._packets(trace, time)
    rotation = trace["controller_frame_rotation"]
    velocity = trace["target_linear_velocity"] @ rotation
    measured_velocity = trace["measured_linear_velocity"] @ rotation
    error = (trace["target_position"] - trace["measured_position"]) @ rotation
    velocity[:, 0], measured_velocity[:, 0], error[:, 0] = 0.0, 0.0, 0.0
    speed = np.linalg.norm(velocity, axis=1)
    direction = velocity / np.sqrt(speed[:, None]**2 + p["velocity_scale"]**2)
    corrected, blend = trace["diagnostic_corrected_force_n"], trace["contact_blend"]
    active = trace["measured_in_contact"] & (corrected > 1.0) & (trace["target_normal_force"] > 0)
    if not np.array_equal(active, trace["diagnostic_compensation_active"]):
        raise ValueError("active state differs from measured contact")
    if np.any(corrected < 0) or np.any((blend < 0) | (blend > 1)):
        raise ValueError("invalid corrected force or contact blend")
    # This protocol uses ScheduledSurfaceSimulator, which always supplies actuation.
    # The frozen projector emits scale=0 for every fallback; with present context,
    # scale=1 therefore means "unchanged". We do not replay the nominal wrench here.
    for name, shape in (("cartesian_jacobian", (count, 6, 7)),
                        ("joint_torque_offset", (count, 7)),
                        ("lower_torque_limit", (count, 7)), ("upper_torque_limit", (count, 7))):
        packet_audit._numeric(trace, name, shape)
    if np.any(trace["lower_torque_limit"] >= trace["upper_torque_limit"]):
        raise ValueError("invalid recorded actuation limits")
    scale = packet_audit._numeric(trace, "torque_projection_scale", (count,))
    if np.any((scale < 0.0) | (scale > 1.0)):
        raise ValueError("projection scale outside [0, 1]")
    accepted = scale == 1.0
    if not np.array_equal(packet_audit._flag(trace, "load_projection_accepted", count), accepted):
        raise ValueError("projection acceptance disagrees with scale")
    mu = p["nominal_mu"]
    load, budget, elapsed = 0.0, p["minimum_force"], 0.0
    prior_force, prior_direction = np.zeros(3), np.zeros(3)
    before, after, outputs = np.zeros(count), np.zeros(count), np.zeros((count, 3))
    scheduler = np.zeros((count, 4))
    ready, capped, slew, updated = (np.zeros(count, dtype=bool) for _ in range(4))
    released = 0
    alpha = -np.expm1(-dt / p["load_time_constant"])
    for index in range(count):
        before[index] = mu
        if not available[index]:
            load, budget = 0.0, p["minimum_force"]
        applied, projected = budget, 0.0
        if not active[index]:
            mu, elapsed = p["nominal_mu"], 0.0
            prior_force, prior_direction = np.zeros(3), np.zeros(3)
            load, budget, applied = 0.0, p["minimum_force"], p["minimum_force"]
        else:
            desired = blend[index] * min(mu * corrected[index], applied) * direction[index]
            delta = desired - prior_force
            distance = float(np.linalg.norm(delta))
            slew[index] = distance > p["force_slew_rate"] * dt
            prior_force += delta * min(1.0, p["force_slew_rate"] * dt / max(distance, 1e-12))
            capped[index] = mu * corrected[index] >= applied
            eligible = (blend[index] >= 0.99 and speed[index] >= p["min_update_speed"]
                        and measured_velocity[index] @ direction[index] >= p["min_update_speed"] / 2
                        and direction[index] @ prior_direction >= 0.0 and not slew[index])
            elapsed = elapsed + dt if eligible else 0.0
            ready[index] = elapsed >= p["motion_confirm_time"]
            if speed[index] >= p["min_update_speed"]:
                prior_direction = direction[index].copy()
            if ready[index] and accepted[index]:
                drive = direction[index] @ (
                    error[index] + p["velocity_error_time"] * (velocity[index] - measured_velocity[index])
                )
                increment = float(np.clip(
                    dt * p["adaptation_gain"] * corrected[index]
                    / (corrected[index]**2 + p["force_regularizer"]**2) * drive,
                    -p["coefficient_rate_limit"] * dt, p["coefficient_rate_limit"] * dt,
                ))
                if not (increment > 0 and mu * corrected[index] >= applied):
                    mu = float(np.clip(mu + increment, 0.0, p["max_equivalent_mu"]))
                if available[index]:
                    projected = float(np.clip(force[index] @ (velocity[index] / speed[index]),
                                              0.0, p["max_force"]))
                    load += float(alpha * (projected - load))
                    budget = float(np.clip(load + p["load_margin"],
                                           p["minimum_force"], p["max_force"]))
                    updated[index] = True
            if (method == "hold_cap_tracking" and available[index] and accepted[index]
                    and speed[index] < p["min_update_speed"]):
                decrease = min(max(0.0, mu - applied / corrected[index]),
                               p["coefficient_rate_limit"] * dt)
                mu -= decrease
                released += int(decrease > 0)
        after[index], outputs[index] = mu, prior_force
        scheduler[index] = (applied, budget, load, projected)
    coefficient_error = max(
        close(trace["controller_coefficient_before_compute"], before, "coefficient before", 1e-12),
        close(trace["controller_coefficient_after_compute"], after, "coefficient transition", 1e-12),
    )
    request_error = close(trace["load_compensation_force_local"], outputs, "force reconstruction")
    close(trace["requested_tangential_force_world"], outputs @ rotation.T, "world force")
    close(trace[old.AUDIT_FIELDS[0]], np.asarray(request_error), "observer reconstruction")
    mismatch = np.asarray(trace[old.AUDIT_FIELDS[1]])
    if mismatch.shape != () or mismatch.dtype.kind not in "iu" or int(mismatch) != 0:
        raise ValueError("observer mismatch count is not zero")
    scheduler_error = close(np.column_stack([trace[name] for name in (
        "load_budget_applied_n", "load_budget_next_n", "load_estimate_n", "load_projected_n",
    )]), scheduler, "budget reconstruction")
    for name, expected in (
        ("controller_update_ready_after_compute", ready),
        ("controller_update_ready_before_compute", np.r_[False, ready[:-1]]),
        ("diagnostic_amplitude_capped", capped), ("diagnostic_slew_limited", slew),
        ("load_budget_updated", updated),
    ):
        if not np.array_equal(trace[name], expected):
            raise ValueError(f"state/flag mismatch: {name}")
    steps = np.linalg.norm(np.diff(np.vstack((np.zeros(3), outputs)), axis=0), axis=1)
    if np.any(np.linalg.norm(outputs, axis=1) > p["max_force"] + 1e-10):
        raise ValueError("global force bound exceeded")
    if np.any(steps[active] > p["force_slew_rate"] * dt + 1e-10):
        raise ValueError("active force slew exceeded")
    return {"scope": "coefficient/slew/readiness/scheduler replay; not independent dynamics",
            "trace_schema_version": SCHEMA, "validated_cycles": count, "hold_release_cycles": released,
            "max_coefficient_transition_error": coefficient_error,
            "max_compensation_reconstruction_error_n": request_error,
            "max_scheduler_reconstruction_error_n": scheduler_error,
            "max_packet_reconstruction_error_n": packet_error,
            "projection_gate_basis": "recorded scale; actuation-present simulator; fallback scale is zero",
            "projection_scaled_or_fallback_cycles": int(np.count_nonzero(~accepted))}


def validate_trace(spec, trace, parameters, protocol):
    previous._validate_full_schema(trace, spec["case"])
    previous._validate_case_schedule(trace, spec["case"])
    metadata = {"method": spec["method"], "trace_schema_version": SCHEMA,
                "rotation_gain_scale": protocol["rotation_gain_scale"],
                "minimum_force_n": parameters["minimum_force"], "max_force_n": parameters["max_force"],
                "fault_name": "fresh", "fault_kind": "fresh", "fault_start_s": 0.0,
                "fault_end_s": 0.0, "fault_value": 0.0}
    for name, expected in metadata.items():
        if np.asarray(trace[name]).shape != () or np.asarray(trace[name]).item() != expected:
            raise ValueError(f"trace metadata mismatch: {name}")
    previous.validate_fault_provenance({"case": spec["case"], "fault": runner.FAULTS["fresh"]}, trace)
    return replay_compensation(trace, parameters, spec["method"], protocol["timestep_s"])


def window_metrics(trace, protocol):
    rotation = yaw_frame(protocol["surface_yaw_deg"]).rotation
    position = (trace["position"] - trace["target_position"]) @ rotation
    velocity = (trace["linear_velocity"] - trace["target_linear_velocity"]) @ rotation
    result = []
    for window in protocol["windows"]:
        mask = (trace["time"] >= window["start_s"]) & (trace["time"] < window["end_s"])
        if not np.any(mask):
            raise ValueError("empty comparison window")
        result.append({**window,
                       "tangent_rmse_mm": float(1000 * np.sqrt(np.mean(np.sum(position[mask, 1:]**2, axis=1)))),
                       "tangent_velocity_rmse_mm_s": float(1000 * np.sqrt(np.mean(np.sum(velocity[mask, 1:]**2, axis=1))))})
    return result


def compare_runs(runs, protocol):
    indexed = {(row["scenario"], row["seed"], row["method"]): row for row in runs}
    specs = {(s["scenario"], s["seed"], s["method"]): s for s in specifications(protocol)}
    if len(indexed) != len(runs) or set(indexed) != set(specs):
        raise ValueError("duplicate/missing paired run")
    limits, absolute = protocol["acceptance"], protocol["absolute_gates"]
    for key, run in indexed.items():
        for row in (run["overall"], *run["phases"], *run["windows"]):
            for name, value in row.items():
                if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
                    raise ValueError(f"nonfinite comparison metric: {name}")
        windows = run["windows"]
        if [row.get("name") for row in windows] != [row["name"] for row in protocol["windows"]]:
            raise ValueError("missing/duplicate/unordered comparison window")
        for row, expected in zip(windows, protocol["windows"], strict=True):
            for field in ("start_s", "end_s"):
                if row.get(field) != expected[field]:
                    raise ValueError("window bounds differ from protocol")
            for metric in ("tangent_rmse_mm", "tangent_velocity_rmse_mm_s"):
                previous.screening._number(row, metric)
        if [row.get("phase") for row in run["phases"]] != [
            phase.name for phase in specs[key]["case"].phases
        ]:
            raise ValueError("missing/duplicate/unordered comparison phase")
        for metric in absolute:
            previous.screening._number(run["overall"], metric)
        for row in run["phases"]:
            for metric in ("force_rmse_n", "orientation_rmse_deg", "contact_ratio_pct",
                           "peak_force_n", "saturation_pct"):
                previous.screening._number(row, metric)
    comparisons = []
    for scenario in protocol["scenario_names"]:
        for seed in protocol["seeds"]:
            baseline, candidate = (indexed[scenario, seed, method] for method in protocol["methods"])
            failures, deltas = [], []
            for left, right in zip(candidate["windows"], baseline["windows"], strict=True):
                name = left["name"]
                if name != right["name"]:
                    raise ValueError("unmatched comparison windows")
                position_delta = left["tangent_rmse_mm"] - right["tangent_rmse_mm"]
                speed_delta = left["tangent_velocity_rmse_mm_s"] - right["tangent_velocity_rmse_mm_s"]
                reduction = (100 * (1 - left["tangent_rmse_mm"] / right["tangent_rmse_mm"])
                             if right["tangent_rmse_mm"] > 0 else None)
                if scenario == "falling" and name in {"early", "post"}:
                    threshold = limits[f"minimum_falling_{name}_tangent_reduction_pct_each_seed"]
                    if reduction is None or reduction < threshold:
                        failures.append(f"{name}:position_reduction")
                elif position_delta > limits["maximum_other_window_tangent_increase_mm"]:
                    failures.append(f"{name}:position_cost")
                if speed_delta > limits["maximum_window_tangent_velocity_rmse_increase_mm_s"]:
                    failures.append(f"{name}:velocity_cost")
                deltas.append({"name": name, "tangent_rmse_mm": position_delta,
                               "tangent_velocity_rmse_mm_s": speed_delta, "tangent_reduction_pct": reduction})
            for left, right in zip(candidate["phases"], baseline["phases"], strict=True):
                if left["phase"] != right["phase"]:
                    raise ValueError("unmatched phases")
                for key, limit in (("force_rmse_n", "maximum_phase_force_rmse_increase_n"),
                                   ("orientation_rmse_deg", "maximum_phase_orientation_rmse_increase_deg")):
                    if left[key] - right[key] > limits[limit]:
                        failures.append(f"{left['phase']}:{key}")
            if candidate["overall"]["peak_force_n"] - baseline["overall"]["peak_force_n"] > limits["maximum_full_raw_peak_increase_n"]:
                failures.append("full_peak_cost")
            for run in (baseline, candidate):
                for row in (run["overall"], *run["phases"]):
                    for key, bound in absolute.items():
                        if key not in row:
                            if row is not run["overall"] and key in {
                                "projection_pct", "minimum_reserved_torque_headroom_nm",
                            }:
                                continue  # These are whole-run metrics only.
                            raise ValueError(f"missing absolute metric: {key}")
                        passed = (row[key] >= bound if key in {"contact_ratio_pct", "minimum_reserved_torque_headroom_nm"}
                                  else row[key] <= bound)
                        if not passed:
                            failures.append(f"{run['method']}:{row.get('phase', 'overall')}:{key}")
            comparisons.append({"scenario": scenario, "seed": seed, "window_deltas": deltas,
                                "failed_checks": failures, "status": "FAIL" if failures else "PASS"})
    return comparisons


def source_identity():
    sources = previous.source_identity()
    for path in (*Path(__file__).parent.glob("*.py"), PROTOCOL, ROOT / "environment/core.lock"):
        sources[path.relative_to(ROOT).as_posix()] = _sha256(path)
    return dict(sorted(sources.items()))


def collect(directory, protocol, *, execute):
    runs = []
    for number, spec in enumerate(specifications(protocol), 1):
        controller, parameters = make_controller(spec, protocol)
        path = directory / spec["trace_path"]
        if execute:
            frame = yaw_frame(spec["case"].task.yaw_deg + spec["case"].controller_yaw_error_deg)
            simulator = ScheduledSurfaceSimulator(spec["case"], frame, old.ARM)
            trace = runner._run_loop(spec["case"], controller, simulator, frame, "fresh").trace
            trace.update(rotation_gain_scale=np.array(protocol["rotation_gain_scale"]),
                         minimum_force_n=np.array(parameters["minimum_force"]),
                         max_force_n=np.array(parameters["max_force"]), method=np.array(spec["method"]),
                         fault_name=np.array("fresh"), fault_kind=np.array("fresh"),
                         fault_start_s=np.array(0.0), fault_end_s=np.array(0.0), fault_value=np.array(0.0),
                         load_max_measurement_age_s=np.array(runner.MAX_MEASUREMENT_AGE_S),
                         trace_schema_version=np.array(SCHEMA))
            previous._save_trace(path, trace)
            print(f"executed {number}/{protocol['maximum_new_simulations']} {spec['trace_path']}", flush=True)
        else:
            trace = previous._load_trace(path)
        audit = validate_trace(spec, trace, parameters, protocol)
        overall, phases, _ = dynamic.metrics(trace, spec["case"], parameters["max_force"])
        runs.append({key: spec[key] for key in ("scenario", "seed", "method", "trace_path")} | {
            "case": _case_document(spec["case"]), "parameters": parameters, "overall": overall,
            "phases": phases, "windows": window_metrics(trace, protocol), "audit": audit,
        })
    comparisons = compare_runs(runs, protocol)
    return {"runs": runs, "comparisons": comparisons,
            "status": "PASS" if all(row["status"] == "PASS" for row in comparisons) else "FAIL",
            "default_changed": False, "eligible_for_default_change": False}


def verify_comparison(saved, rebuilt):
    # JSON gives tuples and lists the same representation; no numeric tolerance.
    # Comparing serialized values also keeps False distinct from a forged numeric 0.
    if json.dumps(saved, sort_keys=True, allow_nan=False) != json.dumps(
        rebuilt, sort_keys=True, allow_nan=False,
    ):
        raise ValueError("recomputed comparison differs")


def verify_sources(saved, commit):
    current = source_identity()
    if saved == current:
        return "exact"
    compatible = {**current, STUDY_SOURCE_PATH: SERIALIZATION_FIX_STUDY_SHA256}
    if commit == SERIALIZATION_FIX_EXECUTION_COMMIT and saved == compatible:
        return "serialization-finalization-only compatibility"
    raise ValueError("source identity differs")


def audit_archive(directory):
    directory = Path(directory)
    manifest = previous.archive.verify_archive(directory)
    protocol = protocol_document()
    expected = {"protocol.json", "source_hashes.json", "comparison.json"} | {
        spec["trace_path"] for spec in specifications(protocol)
    }
    if (manifest["identity"] != protocol["identity"] or manifest["new_simulations"] != len(expected) - 3
            or manifest.get("default_changed") is not False or manifest.get("new_holdout") is not False
            or set(manifest["artifact_sha256"]) != expected):
        raise ValueError("archive identity/count/inventory differs")
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("source commit must be a 40-character hexadecimal Git commit")
    if json.loads((directory / "protocol.json").read_text()) != protocol:
        raise ValueError("protocol differs")
    source_check = verify_sources(json.loads((directory / "source_hashes.json").read_text()), commit)
    rebuilt = collect(directory, protocol, execute=False)
    verify_comparison(json.loads((directory / "comparison.json").read_text()), rebuilt)
    return {"archive_integrity": "PASS", "comparison_status": rebuilt["status"],
            "new_simulations": manifest["new_simulations"], "default_changed": False,
            "source_identity_check": source_check}


def finalize(staging, directory):
    """Audit a completed archive and rename it; never regenerate or rewrite artifacts."""
    staging, destination = Path(staging).absolute(), Path(directory).absolute()
    if not staging.is_dir() or any(path.is_symlink() for path in (staging, *staging.parents)):
        raise ValueError("staging must be a directory and must not traverse a symlink")
    if (destination.exists() or staging == destination or staging in destination.parents
            or any(path.is_symlink() for path in (destination, *destination.parents))):
        raise ValueError("output must be new, outside staging, and must not traverse a symlink")
    audited = audit_archive(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination appeared during audit; staging preserved")
    staging.rename(destination)
    return audited


def run(directory):
    destination = Path(directory).absolute()
    if destination.exists() or any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("output must be new and must not traverse a symlink")
    protocol, sources = protocol_document(), source_identity()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    (staging / "traces").mkdir()
    print(f"staging {staging}; failures retain their partial traces", flush=True)
    _write_json(staging / "protocol.json", protocol)
    _write_json(staging / "source_hashes.json", sources)
    result = collect(staging, protocol, execute=True)
    if sources != source_identity() or protocol != protocol_document():
        raise ValueError("source/protocol changed during execution")
    _write_json(staging / "comparison.json", result)
    artifacts = {path.relative_to(staging).as_posix(): _sha256(path)
                 for path in sorted(staging.rglob("*")) if path.is_file()}
    _write_json(staging / "manifest.json", {
        "identity": protocol["identity"], "new_simulations": protocol["maximum_new_simulations"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_commit_role": "execution HEAD; working inputs pinned in source_hashes.json",
        "artifact_sha256": artifacts, "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "mujoco", "matplotlib", "gymnasium")},
        }, "default_changed": False, "new_holdout": False,
    })
    (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
    return finalize(staging, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--audit", type=Path)
    parser.add_argument("--finalize", type=Path, help="audit and rename an existing complete staging archive")
    args = parser.parse_args()
    if args.finalize and not args.output:
        parser.error("--finalize requires --output and cannot be combined with --audit")
    if args.audit:
        report = audit_archive(args.audit)
    elif args.finalize:
        report = finalize(args.finalize, args.output)
    else:
        report = run(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
