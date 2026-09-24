"""Frozen reversible-release pilot, followed only on success by public transfer."""

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from compliant_control_lab.online_compensation_experiment import (
    ScheduledSurfaceSimulator,
    _case_document,
    _sha256,
    _write_json,
)
from compliant_control_lab.surface_simulation import yaw_frame
from tools.budget_transfer import safe_path, verify_archive
from tools.ci import check_replay_kernel, diagnose_recovery_replay, recovery_runtime_matrix
from tools.reversal_recovery import study as original
from tools.reversal_recovery_transfer import study as transfer
from tools.reversible_recovery.audit import replay_compensation
from tools.reversible_recovery.controller import ReversibleHoldCapTracking
from tools.stationary_recovery import study as stationary

ROOT = original.ROOT
METHOD = "reversible_hold_cap"
STATIONARY = "stationary_hold_cap"
REFERENCE = ROOT / "results/franka_reversal_recovery_transfer"
REFERENCE_SHA256 = "eef733d9b228267f2987a0e4c743020c73a27541010dbeedd8f571a94681c44a"
STATIONARY_REFERENCE = ROOT / "results/franka_stationary_recovery_pilot"
STATIONARY_SHA256 = "ae864f1a86658ecc308aa0e671cecf9f1dd0e3bf13dba0ebd1e46c6ed0c5a88f"
COUNTS = {"pilot": (4, 12, 16, 4), "transfer": (32, 76, 108, 36)}

RUN_KEYS = transfer.RUN_KEYS
CASE_KEYS = ("surface_yaw_deg", "error_profile", "scenario", "seed")
ADAPTIVE = "adaptive6_8"
HOLD = "hold_cap_tracking"
PILOT_YAW_DEG, PILOT_PROFILE = -15, "clean"
HIGH_SEEDS = (11, 29)
CANDIDATE_PREFIX_BEFORE_S = 5.5  # the stationary pilot's prefix bound, unchanged
AUXILIARY_PREFIX_BEFORE_S = 6.0  # the transfer protocol's auxiliary scale-fault start
SCOPE_COUNTS = {
    "pilot": {"total": 16, "transfer": 8, "stationary": 4, "new": 4, "cases": 4},
    "transfer": {"total": 108, "transfer": 72, "pilot": 4, "new": 32, "cases": 36},
}


def case_identity(row):
    """Paired case identity including seed; methods never add paired cases."""
    return tuple(row[name] for name in CASE_KEYS)


def _is_pilot_case(spec):
    return spec["surface_yaw_deg"] == PILOT_YAW_DEG and spec["error_profile"] == PILOT_PROFILE


def specifications(scope="pilot"):
    if scope not in SCOPE_COUNTS:
        raise ValueError(f"unknown scope: {scope}")
    if STATIONARY != stationary.METHOD:
        raise ValueError("stationary control method differs from the frozen pilot")
    old = transfer.specifications(transfer.protocol_document())
    if scope == "pilot":
        old = [spec for spec in old if _is_pilot_case(spec)]
    # The transfer and original archives keep these traces in place; reference_spec
    # retains the origin and path that transfer.trace_location needs to find each one,
    # including the yaw-zero clean runs that still live in the original archive.
    reused = [dict(spec, origin="transfer", reference_spec=spec) for spec in old]
    derived = []
    for spec in old:
        if spec["method"] != ADAPTIVE:
            continue
        methods = [STATIONARY] if scope == "pilot" else []
        for method in (*methods, METHOD):
            if f"__{ADAPTIVE}" not in spec["trace_path"]:
                raise ValueError("unexpected baseline trace path")
            origin = ("stationary" if method == STATIONARY else
                      "pilot" if scope == "transfer" and _is_pilot_case(spec) else "new")
            derived.append(dict(
                spec, method=method, origin=origin,
                trace_path=spec["trace_path"].replace(f"__{ADAPTIVE}", f"__{method}")))
    specs = reused + derived
    counted = {"total": len(specs), "cases": len({case_identity(spec) for spec in specs})}
    for spec in specs:
        counted[spec["origin"]] = counted.get(spec["origin"], 0) + 1
    if (counted != SCOPE_COUNTS[scope]
            or len({transfer.run_key(spec) for spec in specs}) != len(specs)):
        raise ValueError(f"{scope} specification counts or identities differ")
    return specs


def compared_methods(scope):
    """The new candidate is primary; the old and stationary pairings are diagnostic."""
    return (METHOD, HOLD, STATIONARY) if scope == "pilot" else (METHOD, HOLD)


def _window(run, name):
    return next(row for row in run["windows"] if row["name"] == name)


def _high_load_check(indexed, seed):
    """Both ramp errors must be strictly below the stationary-only control; ties fail."""
    key = (PILOT_YAW_DEG, PILOT_PROFILE, "constant_high", seed)
    control, candidate = (_window(indexed[(*key, method)], "ramp")
                          for method in (STATIONARY, METHOD))
    return {
        "seed": seed,
        "position_improved": bool(candidate["tangent_rmse_mm"] < control["tangent_rmse_mm"]),
        "velocity_improved": bool(candidate["tangent_velocity_rmse_mm_s"]
                                  < control["tangent_velocity_rmse_mm_s"]),
    }


def _reaudited_high_checks(rows):
    """Transfer inherits these only from a full pilot re-audit, never from a file."""
    if rows is None:
        raise ValueError("transfer scope requires the re-audited pilot high-load checks")
    rows = [dict(row) for row in rows]
    seeds = [row.get("seed") for row in rows]
    if (any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds) or sorted(seeds) != sorted(HIGH_SEEDS)):
        raise ValueError("pilot high-load checks must cover each high-load seed exactly once")
    for row in rows:
        for name in ("position_improved", "velocity_improved"):
            if type(row.get(name)) is not bool:
                raise ValueError(f"pilot high-load check flag is not boolean: {name}")
    return rows


def compare_runs(runs, scope="pilot", pilot_high_checks=None):
    specs = specifications(scope)
    keys = [transfer.run_key(run) for run in runs]
    if len(set(keys)) != len(keys) or set(keys) != {transfer.run_key(spec) for spec in specs}:
        raise ValueError("duplicate/missing/unknown paired run")
    indexed = dict(zip(keys, runs, strict=True))
    gate_protocol = original.protocol_document()
    comparisons = []
    for yaw, profile in sorted({(spec["surface_yaw_deg"], spec["error_profile"]) for spec in specs}):
        for method in compared_methods(scope):
            # Every metric already uses the run's real surface yaw. This adapter passes
            # only scenario/seed/method identities to the frozen yaw-zero gate, which is
            # also the only place where the candidate wears the old hold-cap alias.
            group = [dict(run, method=HOLD if run["method"] == method else ADAPTIVE)
                     for run in runs
                     if run["method"] in {ADAPTIVE, method}
                     and run["surface_yaw_deg"] == yaw and run["error_profile"] == profile]
            for row in original.compare_runs(group, gate_protocol):
                comparisons.append({
                    "surface_yaw_deg": yaw, "error_profile": profile, **row,
                    "compared_method": method,
                    "failed_checks": [check.replace(f"{HOLD}:", f"{method}:")
                                      for check in row["failed_checks"]],
                })
    if scope == "pilot":
        if pilot_high_checks is not None:
            raise ValueError("the pilot recomputes its own high-load checks")
        high_checks = [_high_load_check(indexed, seed) for seed in HIGH_SEEDS]
    else:
        high_checks = _reaudited_high_checks(pilot_high_checks)
    primary = [row for row in comparisons if row["compared_method"] == METHOD]
    # Acceptance is per case, never an average; diagnostic pairings only report.
    passed = (len(primary) == SCOPE_COUNTS[scope]["cases"]
              and all(row["status"] == "PASS" for row in primary)
              and all(row["position_improved"] and row["velocity_improved"] for row in high_checks))
    return comparisons, high_checks, passed


def prefix_checks(traces, scope="pilot"):
    specs = specifications(scope)
    pairs = []
    for spec in specs:
        if spec["method"] != METHOD:
            continue
        key = transfer.run_key(spec)
        # prefix_check reads per-cycle fields only, so the differing scalar method
        # label is never mistaken for a divergence in the shared prefix.
        pairs.append(((*key[:-1], ADAPTIVE), key,
                      "before_stationary_change", CANDIDATE_PREFIX_BEFORE_S))
        if scope == "transfer" and spec["error_profile"] == "combined_scale_0p8":
            pairs.append(((key[0], "combined", *key[2:]), key,
                          "before_auxiliary_scale_fault", AUXILIARY_PREFIX_BEFORE_S))
    needed = {key for control, candidate, _, _ in pairs for key in (control, candidate)}
    if set(traces) - {transfer.run_key(spec) for spec in specs} or not needed <= set(traces):
        raise ValueError("missing or unknown prefix trace")
    reports = [{**dict(zip(RUN_KEYS, candidate, strict=True)), "kind": kind,
                **transfer.prefix_check(traces[control], traces[candidate], before_s)}
               for control, candidate, kind, before_s in pairs]
    if len(reports) != (4 if scope == "pilot" else 48):
        raise ValueError(f"{scope} prefix comparison count differs")
    return reports


def protocol_document(scope="pilot"):
    if scope not in COUNTS:
        raise ValueError("unknown scope")
    old = original.protocol_document()
    design = transfer.protocol_document()
    candidate = dict(transfer.specifications(design)[0], method=METHOD)
    _, parameters = make_controller(candidate)
    return {
        "identity": f"reversible-release-{scope}-v1", "scope": scope,
        "public_development": True, "new_holdout": False, "default_changed": False,
        "comparison_defined_before_execution": True, "candidate_method": METHOD,
        "changed_factor": "stationary coefficient tracks min(fixed hold-entry coefficient, budget / normal load) in both directions",
        "release_speed_max_m_s": 1e-12, "coefficient_rate_s": 0.3,
        "controller_parameters": parameters,
        "default_force_n": 6, "rotation_gain_scale": 1, "velocity_error_time_s": 0.05,
        "acceptance": old["acceptance"], "absolute_gates": old["absolute_gates"],
        "windows": old["windows"], "comparison_replay": design["comparison_replay"],
        "reference_directory": REFERENCE.relative_to(ROOT).as_posix(),
        "reference_manifest_sha256": REFERENCE_SHA256,
        "stationary_directory": STATIONARY_REFERENCE.relative_to(ROOT).as_posix(),
        "stationary_manifest_sha256": STATIONARY_SHA256,
        "candidate_prefix_before_s": 5.5, "auxiliary_prefix_before_s": 6.0,
        "additional_gate": "both high-load pilot seeds: ramp position AND velocity RMSE strictly lower than stationary candidate",
        "transfer_precondition": "complete recomputed PASS pilot with identical source identity; retain its four traces in place",
        "gate_adapter": "method alias only inside original comparison; real-yaw metrics",
        "scopes": {name: dict(zip(
            ("maximum_new_simulations", "reused_runs", "evaluated_runs", "paired_cases"), counts, strict=True,
        )) for name, counts in COUNTS.items()},
        "frozen_transfer_cases": [
            {**{key: spec[key] for key in transfer.RUN_KEYS[:-1]}, "case": _case_document(spec["case"])}
            for spec in transfer.specifications(design) if spec["method"] == "adaptive6_8"
        ],
        "pilot_selection": {"surface_yaw_deg": -15, "error_profile": "clean"},
        "scope_boundary": "public development only; no default promotion, hardware or C++ parity claim",
    }


def source_identity():
    sources = stationary.source_identity()
    for path in (*Path(__file__).parent.glob("*.py"), Path(check_replay_kernel.__file__),
                 Path(recovery_runtime_matrix.__file__), Path(diagnose_recovery_replay.__file__)):
        sources[path.relative_to(ROOT).as_posix()] = _sha256(path)
    return dict(sorted(sources.items()))


def make_controller(spec):
    controller, parameters = original.make_controller(
        {**spec, "method": "adaptive6_8"}, original.protocol_document(),
    )
    if spec["method"] == METHOD:
        controller._base.tangential = ReversibleHoldCapTracking(**parameters)
    return controller, parameters


def validate_trace(spec, trace, parameters):
    original.previous._validate_full_schema(trace, spec["case"])
    original.previous._validate_case_schedule(trace, spec["case"])
    for key, expected in {
        "method": spec["method"], "trace_schema_version": original.SCHEMA,
        "rotation_gain_scale": 1, "minimum_force_n": 6.0, "max_force_n": 8.0,
        "load_max_measurement_age_s": original.runner.MAX_MEASUREMENT_AGE_S,
        **{f"fault_{name}": value for name, value in asdict(spec["fault"]).items()},
    }.items():
        actual = np.asarray(trace.get(key))
        if actual.shape != () or actual.item() != expected:
            raise ValueError(f"trace metadata mismatch: {key}")
    return {**original.previous.validate_fault_provenance(spec, trace),
            **replay_compensation(trace, parameters, spec["method"], 0.002)}


def verify_references():
    verify_archive(REFERENCE, REFERENCE_SHA256)
    verify_archive(STATIONARY_REFERENCE, STATIONARY_SHA256)
    for directory, expected in ((REFERENCE, transfer.source_identity()),
                                (STATIONARY_REFERENCE, stationary.source_identity())):
        if json.loads((directory / "source_hashes.json").read_text()) != expected:
            raise ValueError("reference source identity differs")
    nested = transfer.protocol_document()["reference"]
    verify_archive(ROOT / nested["directory"], nested["manifest_sha256"])


def trace_location(directory, spec, pilot_reference=None):
    origin = spec["origin"]
    if origin == "transfer":
        return transfer.trace_location(REFERENCE, spec["reference_spec"], transfer.protocol_document())
    if origin == "stationary":
        return STATIONARY_REFERENCE / spec["trace_path"]
    if origin == "pilot":
        if pilot_reference is None:
            raise ValueError("transfer requires pilot reference")
        return Path(pilot_reference) / spec["trace_path"]
    if origin != "new":
        raise ValueError("unknown trace origin")
    return Path(directory) / spec["trace_path"]


def verified_pilot(directory, *, expected_digest=None):
    if directory is None:
        raise ValueError("transfer requires pilot reference")
    directory = Path(directory).absolute()
    manifest = verify_archive(directory, expected_digest)
    if manifest.get("scope") != "pilot":
        raise ValueError("transfer reference must be a pilot")
    audited = audit_archive(directory, scope="pilot")
    if audited["comparison_status"] != "PASS":
        raise ValueError("transfer requires a recomputed successful pilot")
    # This summary is read only after audit_archive has recomputed every field.
    return directory, json.loads((directory / "comparison.json").read_text())["high_load_checks"]


def collect(directory, *, scope="pilot", execute=False, pilot_reference=None):
    verify_references()
    expected_parameters = protocol_document(scope)["controller_parameters"]
    high_checks = None
    if scope == "transfer":
        pilot_reference, high_checks = verified_pilot(pilot_reference)
    elif pilot_reference is not None:
        raise ValueError("pilot scope cannot reference another pilot")
    runs, traces = [], {}
    for spec in specifications(scope):
        path = trace_location(directory, spec, pilot_reference)
        controller, parameters = make_controller(spec)
        if parameters != expected_parameters:
            raise ValueError("constructed controller parameters differ from protocol")
        if execute and spec["origin"] == "new":
            if path.exists() or path.is_symlink():
                raise ValueError("refuse to overwrite a trial")
            frame = yaw_frame(spec["case"].task.yaw_deg + spec["case"].controller_yaw_error_deg)
            simulator = ScheduledSurfaceSimulator(spec["case"], frame, original.old.ARM)
            trace = original.runner._run_loop(spec["case"], controller, simulator, frame, spec["fault"]).trace
            trace.update({name: np.array(value) for name, value in {
                "method": METHOD, "trace_schema_version": original.SCHEMA,
                "rotation_gain_scale": 1, "minimum_force_n": 6.0, "max_force_n": 8.0,
                "load_max_measurement_age_s": original.runner.MAX_MEASUREMENT_AGE_S,
                **{f"fault_{name}": value for name, value in asdict(spec["fault"]).items()},
            }.items()})
            original.previous._save_trace(path, trace)
            print(f"executed {spec['trace_path']}", flush=True)
        else:
            trace = original.previous._load_trace(path)
        audit = validate_trace(spec, trace, parameters)
        overall, phases, _ = original.dynamic.metrics(trace, spec["case"], parameters["max_force"])
        runs.append({key: spec[key] for key in (*transfer.RUN_KEYS, "origin", "trace_path")} | {
            "trace_sha256": _sha256(path), "case": _case_document(spec["case"]),
            "parameters": parameters, "overall": overall, "phases": phases,
            "windows": transfer.window_metrics(trace, spec["case"], original.protocol_document()),
            "audit": audit,
        })
        traces[transfer.run_key(spec)] = trace
    comparisons, high_checks, passed = compare_runs(runs, scope, high_checks)
    prefixes = prefix_checks(traces, scope)
    return {"runs": runs, "comparisons": comparisons, "high_load_checks": high_checks,
            "prefix_checks": prefixes,
            "status": "PASS" if passed and all(row["bit_exact"] for row in prefixes) else "FAIL",
            "default_changed": False, "eligible_for_default_change": False, "new_holdout": False}


def audit_archive(directory, *, scope=None, pilot_reference=None):
    directory = Path(directory)
    manifest = verify_archive(directory)
    actual_scope = manifest.get("scope")
    if actual_scope not in COUNTS or scope is not None and scope != actual_scope:
        raise ValueError("archive scope differs")
    scope = actual_scope
    protocol = protocol_document(scope)
    counts = protocol["scopes"][scope]
    expected = {"protocol.json", "source_hashes.json", "comparison.json"} | {
        spec["trace_path"] for spec in specifications(scope) if spec["origin"] == "new"
    }
    if (manifest.get("identity") != protocol["identity"]
            or any(type(manifest.get(key)) is not int or manifest[key] != counts[key] for key in counts)
            or manifest.get("default_changed") is not False or manifest.get("new_holdout") is not False
            or manifest.get("public_development") is not True
            or set(manifest["artifact_sha256"]) != expected):
        raise ValueError("archive identity/count/scope/inventory differs")
    if not isinstance(manifest.get("source_commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"]):
        raise ValueError("invalid execution commit")
    original.verify_comparison(json.loads((directory / "protocol.json").read_text()), protocol)
    if json.loads((directory / "source_hashes.json").read_text()) != source_identity():
        raise ValueError("source identity differs")
    if scope == "transfer":
        reference = manifest.get("pilot_reference")
        if not isinstance(reference, dict) or set(reference) != {"directory", "manifest_sha256"}:
            raise ValueError("missing pilot provenance")
        digest = reference["manifest_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid pilot manifest digest")
        declared = safe_path(ROOT, reference["directory"])
        pilot_reference, _ = verified_pilot(pilot_reference or declared, expected_digest=digest)
    elif "pilot_reference" in manifest or pilot_reference is not None:
        raise ValueError("pilot scope cannot reference another pilot")
    rebuilt = collect(directory, scope=scope, execute=False, pilot_reference=pilot_reference)
    transfer.verify_comparison(json.loads((directory / "comparison.json").read_text()), rebuilt, protocol)
    return {"archive_integrity": "PASS", "comparison_status": rebuilt["status"], "scope": scope,
            "evaluated_runs": counts["evaluated_runs"], "new_simulations": counts["maximum_new_simulations"],
            "reused_runs": counts["reused_runs"], "default_changed": False, "new_holdout": False}


def run(directory, *, scope="pilot", pilot_reference=None):
    destination = Path(directory).absolute()
    if destination.exists() or any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("output must be new and must not traverse a symlink")
    protocol, sources = protocol_document(scope), source_identity()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT):
        raise ValueError("commit tracked changes before execution")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.check_output(["git", "ls-files", "--error-unmatch", "--", *sources], cwd=ROOT)
    check_replay_kernel.main()
    reference = None
    if scope == "transfer":
        pilot_reference, _ = verified_pilot(pilot_reference)
        try:
            relative = pilot_reference.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise ValueError("execution pilot reference must be inside repository") from exc
        safe_path(ROOT, relative)
        reference = {"directory": relative, "manifest_sha256": _sha256(pilot_reference / "manifest.json")}
    elif pilot_reference is not None:
        raise ValueError("pilot scope cannot reference another pilot")
    verify_references()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    (staging / "traces").mkdir()
    print(f"staging {staging}; failures retain every partial trace", flush=True)
    _write_json(staging / "protocol.json", protocol)
    _write_json(staging / "source_hashes.json", sources)
    result = collect(staging, scope=scope, execute=True, pilot_reference=pilot_reference)
    if sources != source_identity() or protocol != protocol_document(scope):
        raise ValueError("source/protocol changed during execution")
    _write_json(staging / "comparison.json", result)
    manifest = {
        "identity": protocol["identity"], "scope": scope, "source_commit": commit,
        "source_commit_role": "clean tracked execution-start HEAD; exact inputs in source_hashes.json",
        **protocol["scopes"][scope], "default_changed": False, "new_holdout": False,
        "public_development": True,
        "artifact_sha256": {path.relative_to(staging).as_posix(): _sha256(path)
                            for path in sorted(staging.rglob("*")) if path.is_file()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "mujoco")}},
    }
    if reference is not None:
        manifest["pilot_reference"] = reference
    _write_json(staging / "manifest.json", manifest)
    (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
    audited = audit_archive(staging, scope=scope, pilot_reference=pilot_reference)
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination appeared during audit; staging preserved")
    staging.rename(destination)
    return audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--audit", type=Path)
    parser.add_argument("--scope", choices=tuple(COUNTS))
    parser.add_argument("--pilot-reference", type=Path)
    args = parser.parse_args()
    if args.output and (args.scope or "pilot") == "transfer" and args.pilot_reference is None:
        parser.error("--scope transfer --output requires --pilot-reference")
    if args.pilot_reference is not None and (args.scope == "pilot" or args.output and args.scope is None):
        parser.error("pilot scope cannot use --pilot-reference")
    report = (audit_archive(args.audit, scope=args.scope, pilot_reference=args.pilot_reference) if args.audit
              else run(args.output, scope=args.scope or "pilot", pilot_reference=args.pilot_reference))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
