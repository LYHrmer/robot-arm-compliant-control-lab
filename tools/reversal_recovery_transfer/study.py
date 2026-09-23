"""Public transfer regression; the old controller and replay equations stay frozen.

The private runner seams deliberately match the original experiment. Audits cover
recorded schedules, packet provenance and auxiliary state, not independent physics.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from compliant_control_lab.online_compensation_experiment import (
    DeterministicSchedule,
    ScheduledSurfaceSimulator,
    _case_document,
    _sha256,
    _write_json,
)
from compliant_control_lab.surface_simulation import yaw_frame
from tools.reversal_recovery import study as original

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")
RUN_KEYS = ("surface_yaw_deg", "error_profile", "scenario", "seed", "method")


def protocol_document():
    return json.loads(PROTOCOL.read_text())


def run_key(row):
    return tuple(row[name] for name in RUN_KEYS)


def specifications(protocol):
    # This also preserves the original phase names for unchanged gate evaluation.
    old_protocol = original.protocol_document()
    for name in ("methods", "scenario_names", "seeds", "duration_s", "timestep_s",
                 "friction_schedules", "rotation_gain_scale", "velocity_error_time_s", "windows",
                 "absolute_gates"):
        if isinstance(protocol[name], bool) or protocol[name] != old_protocol[name]:
            raise ValueError(f"frozen original protocol input differs: {name}")
    for name, threshold in old_protocol["acceptance"].items():
        if name.startswith(("minimum_", "maximum_")) and protocol["acceptance"].get(name) != threshold:
            raise ValueError(f"original acceptance threshold differs: {name}")
    reference = protocol["reference"]
    reuse_yaw, reuse_profile = reference["reuse_surface_yaw_deg"], reference["reuse_error_profile"]
    if (reuse_yaw != old_protocol["surface_yaw_deg"] or reuse_profile not in protocol["error_profiles"]
            or protocol["error_profiles"][reuse_profile] != {
                "controller_yaw_error_deg": 0, "normal_bias_n": 0, "fault": "fresh",
            }):
        raise ValueError("reference reuse does not describe the original clean yaw-0 traces")
    scale = original.runner.FAULTS["scale_0p8"]
    if any(protocol["auxiliary_scale_fault"][name] != getattr(scale, name)
           for name in ("start_s", "end_s", "value")):
        raise ValueError("auxiliary fault declaration differs from executed fault")
    if (protocol["normal_bias_start_s"] != scale.start_s
            or [check["before_s"] for check in protocol["prefix_checks"]] != [4.5, scale.start_s]
            or not all(check["require_bit_exact"] is True for check in protocol["prefix_checks"])):
        raise ValueError("prefix or normal-bias timing differs from the fixed experiment")
    old_specs = original.specifications(old_protocol)
    specs = []
    for yaw in protocol["surface_yaws_deg"]:
        normal = yaw_frame(yaw).rotation[:, 0]
        for profile, errors in protocol["error_profiles"].items():
            fault = original.runner.fault_config(errors["fault"])
            for old_spec in old_specs:
                scenario, seed, method = (old_spec[name] for name in ("scenario", "seed", "method"))
                case = old_spec["case"]
                reuse = yaw == reuse_yaw and profile == reuse_profile
                label = f"yaw{int(yaw)}__{scenario}__{profile}__seed{seed}"
                if not reuse:
                    bias = DeterministicSchedule((
                        (0.0, (0.0,) * 6),
                        (protocol["normal_bias_start_s"],
                         tuple(errors["normal_bias_n"] * normal) + (0.0,) * 3),
                    ))
                    case = replace(
                        case, name=label,
                        scenario=replace(case.scenario, wall_yaw_deg=yaw, name=label),
                        task=replace(case.task, yaw_deg=yaw), wrench_bias_world=bias,
                        controller_yaw_error_deg=errors["controller_yaw_error_deg"],
                    )
                specs.append({
                    "surface_yaw_deg": yaw, "error_profile": profile,
                    "scenario": scenario, "seed": seed, "method": method, "case": case,
                    "fault": fault, "origin": "reference" if reuse else "new",
                    "trace_path": old_spec["trace_path"] if reuse else f"traces/{label}__{method}.npz",
                })
    reused = sum(spec["origin"] == "reference" for spec in specs)
    if (len(specs) != protocol["evaluated_runs"] or reused != protocol["reused_runs"]
            or sum(spec["origin"] == "new" for spec in specs) != protocol["maximum_new_simulations"]
            or len(specs) != 2 * protocol["paired_comparisons"]
            or len(protocol["surface_yaws_deg"]) * len(protocol["scenario_names"])
            != protocol["physical_surface_load_combinations"]
            or len({run_key(spec) for spec in specs}) != len(specs)):
        raise ValueError("duplicate specifications or simulation counts differ from protocol")
    return specs


def validate_trace(spec, trace, parameters, protocol):
    previous = original.previous
    previous._validate_full_schema(trace, spec["case"])
    previous._validate_case_schedule(trace, spec["case"])
    expected = {
        "method": spec["method"], "trace_schema_version": original.SCHEMA,
        "rotation_gain_scale": protocol["rotation_gain_scale"],
        "minimum_force_n": parameters["minimum_force"], "max_force_n": parameters["max_force"],
        "load_max_measurement_age_s": original.runner.MAX_MEASUREMENT_AGE_S,
        **{f"fault_{key}": value for key, value in asdict(spec["fault"]).items()},
    }
    for name, value in expected.items():
        actual = np.asarray(trace.get(name))
        if actual.shape != () or actual.item() != value:
            raise ValueError(f"trace metadata mismatch: {name}")
    provenance = previous.validate_fault_provenance(spec, trace)
    replay = original.replay_compensation(trace, parameters, spec["method"], protocol["timestep_s"])
    return {**provenance, **replay}


def window_metrics(trace, case, protocol):
    # The physical surface frame, not the intentionally miscalibrated controller.
    return original.window_metrics(trace, {**protocol, "surface_yaw_deg": case.scenario.wall_yaw_deg})


def compare_runs(runs, protocol):
    keys = [run_key(run) for run in runs]
    if len(set(keys)) != len(keys) or set(keys) != {run_key(spec) for spec in specifications(protocol)}:
        raise ValueError("duplicate/missing/misidentified transfer run")
    comparisons = []
    # The original gate's specifications enforce yaw=0. This adapter passes only
    # scenario/seed/method identities to that gate; metrics already use actual yaw.
    gate_protocol = {**protocol, "surface_yaw_deg": 0.0, "maximum_new_simulations": 8}
    for yaw in protocol["surface_yaws_deg"]:
        for profile in protocol["error_profiles"]:
            group = [run for run in runs if run["surface_yaw_deg"] == yaw
                     and run["error_profile"] == profile]
            for row in original.compare_runs(group, gate_protocol):
                absolute = [check for check in row["failed_checks"]
                            if check.startswith(tuple(f"{method}:" for method in protocol["methods"]))]
                benefit = [check for check in row["failed_checks"] if check.endswith(":position_reduction")]
                costs = [check for check in row["failed_checks"] if check not in absolute + benefit]
                comparisons.append({"surface_yaw_deg": yaw, "error_profile": profile, **row,
                                    "absolute_failures": absolute, "missed_benefit_checks": benefit,
                                    "paired_cost_failures": costs})
    return comparisons


def source_identity():
    sources = original.source_identity()
    for path in (*Path(__file__).parent.glob("*.py"), PROTOCOL):
        sources[path.relative_to(ROOT).as_posix()] = _sha256(path)
    return dict(sorted(sources.items()))


def verify_reference(protocol):
    reference = protocol["reference"]
    directory = ROOT / reference["directory"]
    original.previous.archive.verify_archive(directory, reference["manifest_sha256"])
    original.audit_archive(directory)
    return directory


def trace_location(directory, spec, protocol):
    root = ROOT / protocol["reference"]["directory"] if spec["origin"] == "reference" else directory
    return root / spec["trace_path"]


def prefix_check(left, right, before_s):
    if set(left) != set(right):
        raise ValueError("paired prefix schemas differ")
    count = len(left["time"])
    mask = left["time"] < before_s
    checked, different = [], []
    for name, expected in left.items():
        actual = right[name]
        if expected.ndim and expected.shape[0] == count:
            checked.append(name)
            if (actual.shape != expected.shape or actual.dtype != expected.dtype
                    or actual[mask].tobytes() != expected[mask].tobytes()):
                different.append(name)
    return {"before_s": before_s, "samples": int(np.count_nonzero(mask)),
            "fields": sorted(checked), "different_fields": sorted(different), "bit_exact": not different}


def prefix_comparisons(directory, specs, protocol):
    indexed = {run_key(spec): spec for spec in specs}
    reports = []
    for spec in specs:
        key = run_key(spec)
        targets = []
        if spec["method"] == "hold_cap_tracking":
            targets.append(((*key[:-1], "adaptive6_8"), "before_candidate_divergence",
                            protocol["prefix_checks"][0]["before_s"]))
        if spec["error_profile"] == "combined_scale_0p8":
            targets.append(((key[0], "combined", *key[2:]), "before_auxiliary_scale_fault",
                            protocol["prefix_checks"][1]["before_s"]))
        if not targets:
            continue
        trace = original.previous._load_trace(trace_location(directory, spec, protocol))
        for reference_key, kind, before_s in targets:
            reference = indexed[reference_key]
            control = original.previous._load_trace(trace_location(directory, reference, protocol))
            reports.append({**{name: spec[name] for name in RUN_KEYS}, "kind": kind,
                            **prefix_check(control, trace, before_s)})
    return reports


def collect(directory, protocol, *, execute):
    verify_reference(protocol)
    specs, runs, executed = specifications(protocol), [], 0
    for spec in specs:
        controller, parameters = original.make_controller(spec, protocol)
        path = trace_location(directory, spec, protocol)
        if execute and spec["origin"] == "new":
            frame = yaw_frame(spec["case"].task.yaw_deg + spec["case"].controller_yaw_error_deg)
            simulator = ScheduledSurfaceSimulator(spec["case"], frame, original.old.ARM)
            trace = original.runner._run_loop(spec["case"], controller, simulator, frame, spec["fault"]).trace
            trace.update({name: np.array(value) for name, value in {
                "rotation_gain_scale": protocol["rotation_gain_scale"],
                "minimum_force_n": parameters["minimum_force"], "max_force_n": parameters["max_force"],
                "method": spec["method"], "trace_schema_version": original.SCHEMA,
                "load_max_measurement_age_s": original.runner.MAX_MEASUREMENT_AGE_S,
                **{f"fault_{key}": value for key, value in asdict(spec["fault"]).items()},
            }.items()})
            original.previous._save_trace(path, trace)
            executed += 1
            print(f"executed {executed}/{protocol['maximum_new_simulations']} {spec['trace_path']}", flush=True)
        else:
            trace = original.previous._load_trace(path)
        audit = validate_trace(spec, trace, parameters, protocol)
        overall, phases, _ = original.dynamic.metrics(trace, spec["case"], parameters["max_force"])
        runs.append({name: spec[name] for name in (*RUN_KEYS, "origin", "trace_path")} | {
            "trace_sha256": _sha256(path), "case": _case_document(spec["case"]),
            "fault": asdict(spec["fault"]), "parameters": parameters, "overall": overall,
            "phases": phases, "windows": window_metrics(trace, spec["case"], protocol), "audit": audit,
        })
    comparisons = compare_runs(runs, protocol)
    prefixes = prefix_comparisons(directory, specs, protocol)
    passed = all(row["status"] == "PASS" for row in comparisons) and all(row["bit_exact"] for row in prefixes)
    return {"runs": runs, "comparisons": comparisons, "prefix_checks": prefixes,
            "status": "PASS" if passed else "FAIL", "default_changed": False,
            "eligible_for_default_change": False, "new_holdout": False}


def verify_comparison(saved, rebuilt, protocol):
    """Allow declared absolute roundoff only in named continuous result fields."""
    contract = protocol["comparison_replay"]
    metric_tol, audit_tol = (contract[name] for name in (
        "metrics_absolute_tolerance", "audit_absolute_tolerance",
    ))
    if (contract["relative_tolerance"] != 0 or isinstance(contract["relative_tolerance"], bool)
            or any(isinstance(value, bool) or not isinstance(value, (float, int))
                   or not np.isfinite(value) or value < 0 for value in (metric_tol, audit_tol))):
        raise ValueError("invalid comparison replay tolerance contract")
    metrics = {
        "coefficient_max", "coefficient_min", "contact_ratio_pct",
        "diagnostic_amplitude_capped_pct", "diagnostic_compensation_active_pct",
        "diagnostic_slew_limited_pct", "force_recovery_time_s", "force_rmse_n",
        "max_coefficient_reset_jump", "max_force_reset_jump_n",
        "max_normal_operation_coefficient_rate_s", "max_normal_operation_force_slew_n_s",
        "max_requested_tangential_force_n", "max_uncapped_tangential_amplitude_n",
        "minimum_reserved_torque_headroom_nm", "minimum_torque_headroom_nm",
        "orientation_rmse_deg", "peak_force_n", "projection_pct", "saturation_pct",
        "slew_reconstruction_max_error_n", "tangent_recovery_time_s", "tangent_rmse_mm",
        "tangent_velocity_error_rms_m_s", "tangential_update_ready_pct",
        "worst_phase_force_rmse_n", "worst_phase_orientation_rmse_deg", "worst_phase_tangent_rmse_mm",
    }
    audits = {
        "max_coefficient_transition_error", "max_compensation_reconstruction_error_n",
        "max_packet_reconstruction_error_n", "max_scheduler_reconstruction_error_n",
        "max_raw_packet_force_reconstruction_error_n", "max_raw_packet_stamp_reconstruction_error_s",
    }

    def tolerance(path):
        if len(path) >= 4 and path[0] == "runs" and isinstance(path[1], int):
            if len(path) == 4 and path[2] == "audit" and path[3] in audits:
                return audit_tol
            if len(path) == 4 and path[2] == "overall" and path[3] in metrics:
                return metric_tol
            if len(path) == 5 and isinstance(path[3], int):
                if path[2] == "phases" and path[4] in metrics:
                    return metric_tol
                if path[2] == "windows" and path[4] in {"tangent_rmse_mm", "tangent_velocity_rmse_mm_s"}:
                    return metric_tol
        if (len(path) == 5 and path[0] == "comparisons" and isinstance(path[1], int)
                and path[2] == "window_deltas" and isinstance(path[3], int)
                and path[4] in {"tangent_rmse_mm", "tangent_velocity_rmse_mm_s", "tangent_reduction_pct"}):
            return metric_tol
        return None

    def compare(left, right, path=()):
        label = "/".join(map(str, path)) or "root"
        if isinstance(left, dict) and isinstance(right, dict):
            if (not all(isinstance(key, str) for key in (*left, *right))
                    or left.keys() != right.keys()):
                raise ValueError(f"comparison keys differ: {label}")
            for key in left:
                compare(left[key], right[key], (*path, key))
        elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            # Case dataclass tuples become lists in JSON, as in the original audit.
            if len(left) != len(right):
                raise ValueError(f"comparison lengths differ: {label}")
            for index, (one, other) in enumerate(zip(left, right, strict=True)):
                compare(one, other, (*path, index))
        elif type(left) is not type(right):
            raise ValueError(f"comparison types differ: {label}")
        elif isinstance(left, float):
            if not np.isfinite(left) or not np.isfinite(right):
                raise ValueError(f"nonfinite comparison value: {label}")
            allowed = tolerance(path)
            if ((allowed is None and left.hex() != right.hex())
                    or (allowed is not None and abs(left - right) > allowed)):
                raise ValueError(f"recomputed comparison differs: {label}")
        elif left is None or isinstance(left, (str, int, bool)):
            if left != right:
                raise ValueError(f"recomputed comparison differs: {label}")
        else:
            raise ValueError(f"unsupported comparison value: {label}")

    compare(saved, rebuilt)


def audit_archive(directory):
    directory = Path(directory)
    protocol = protocol_document()
    manifest = original.previous.archive.verify_archive(directory)
    expected = {"protocol.json", "source_hashes.json", "comparison.json"} | {
        spec["trace_path"] for spec in specifications(protocol) if spec["origin"] == "new"
    }
    if (manifest["identity"] != protocol["identity"]
            or manifest["new_simulations"] != protocol["maximum_new_simulations"]
            or manifest["evaluated_runs"] != protocol["evaluated_runs"]
            or manifest.get("default_changed") is not False or manifest.get("new_holdout") is not False
            or manifest.get("public_development") is not True
            or set(manifest["artifact_sha256"]) != expected):
        raise ValueError("archive identity/count/scope/inventory differs")
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("source commit must be a 40-character hexadecimal Git commit")
    original.verify_comparison(json.loads((directory / "protocol.json").read_text()), protocol)
    if json.loads((directory / "source_hashes.json").read_text()) != source_identity():
        raise ValueError("source identity differs")
    rebuilt = collect(directory, protocol, execute=False)
    verify_comparison(json.loads((directory / "comparison.json").read_text()), rebuilt, protocol)
    return {"archive_integrity": "PASS", "comparison_status": rebuilt["status"],
            "evaluated_runs": protocol["evaluated_runs"], "new_simulations": manifest["new_simulations"],
            "reused_runs": protocol["evaluated_runs"] - manifest["new_simulations"],
            "default_changed": False, "new_holdout": False}


def finalize(staging, directory):
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
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    protocol, sources = protocol_document(), source_identity()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    (staging / "traces").mkdir()
    print(f"staging {staging}; failures retain their traces", flush=True)
    _write_json(staging / "protocol.json", protocol)
    _write_json(staging / "source_hashes.json", sources)
    result = collect(staging, protocol, execute=True)
    if sources != source_identity() or protocol != protocol_document():
        raise ValueError("source/protocol changed during execution")
    _write_json(staging / "comparison.json", result)
    _write_json(staging / "manifest.json", {
        "identity": protocol["identity"], "new_simulations": protocol["maximum_new_simulations"],
        "evaluated_runs": protocol["evaluated_runs"],
        "source_commit": commit,
        "source_commit_role": "execution-start HEAD; working inputs pinned in source_hashes.json",
        "artifact_sha256": {path.relative_to(staging).as_posix(): _sha256(path)
                            for path in sorted(staging.rglob("*")) if path.is_file()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name)
                                     for name in ("numpy", "mujoco", "matplotlib", "gymnasium")}},
        "default_changed": False, "new_holdout": False, "public_development": True,
    })
    (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
    return finalize(staging, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--audit", type=Path)
    parser.add_argument("--finalize", type=Path, help="audit and rename a complete existing staging archive")
    args = parser.parse_args()
    if args.finalize and not args.output:
        parser.error("--finalize requires --output and cannot be combined with --audit")
    report = (audit_archive(args.audit) if args.audit else
              finalize(args.finalize, args.output) if args.finalize else run(args.output))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
