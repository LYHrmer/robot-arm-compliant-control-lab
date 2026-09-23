"""Four frozen public-development trials of stationary-only coefficient release.

Reuse eight archived controls, retain every outcome, and never replace an archive.
The original gate uses method aliases only inside its comparison adapter.
"""

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
from tools.budget_transfer import verify_archive
from tools.ci.check_replay_kernel import main as check_replay_kernel
from tools.reversal_recovery import study as original
from tools.reversal_recovery_transfer import study as transfer
from tools.stationary_recovery.audit import replay_compensation
from tools.stationary_recovery.controller import StationaryHoldCapTracking

ROOT = original.ROOT
METHOD = "stationary_hold_cap"
REFERENCE = ROOT / "results/franka_reversal_recovery_transfer"
REFERENCE_SHA256 = "eef733d9b228267f2987a0e4c743020c73a27541010dbeedd8f571a94681c44a"


def specifications():
    selected = [dict(spec, origin="reference")
                for spec in transfer.specifications(transfer.protocol_document())
                if spec["surface_yaw_deg"] == -15 and spec["error_profile"] == "clean"]
    candidates = [dict(spec, method=METHOD, origin="new",
                       trace_path=spec["trace_path"].replace("__adaptive6_8", f"__{METHOD}"))
                  for spec in selected if spec["method"] == "adaptive6_8"]
    return selected + candidates


def protocol_document():
    old = original.protocol_document()
    return {
        "identity": "stationary-release-public-pilot-v1",
        "public_development": True, "new_holdout": False, "default_changed": False,
        "comparison_defined_before_execution": True, "maximum_new_simulations": 4,
        "evaluated_runs": 12, "reused_runs": 8, "physical_scenarios": 2,
        "surface_yaw_deg": -15, "error_profile": "clean", "seeds": [11, 29],
        "candidate_method": METHOD, "release_speed_max_m_s": 1e-12,
        "changed_factor": "only release speed gate; no clock/phase-label input",
        "reference_directory": REFERENCE.relative_to(ROOT).as_posix(),
        "reference_manifest_sha256": REFERENCE_SHA256,
        "acceptance": old["acceptance"], "absolute_gates": old["absolute_gates"],
        "windows": old["windows"], "prefix_before_s": 5.5,
        "comparison_replay": transfer.protocol_document()["comparison_replay"],
        "additional_gate": "each high-load seed: both ramp position and velocity RMSE lower than prior hold-cap",
        "gate_adapter": "reuse original gate; stationary method aliases hold_cap_tracking only inside comparison",
        "scope": "four public paired cases only; no default promotion, transfer, public24 or C++ claim",
        "cases": [{"scenario": s["scenario"], "seed": s["seed"], "case": _case_document(s["case"])}
                  for s in specifications() if s["origin"] == "new"],
    }


def source_identity():
    sources = transfer.source_identity()
    for path in Path(__file__).parent.glob("*.py"):
        sources[path.relative_to(ROOT).as_posix()] = _sha256(path)
    return dict(sorted(sources.items()))


def make_controller(spec):
    controller, parameters = original.make_controller(
        {**spec, "method": "adaptive6_8"}, original.protocol_document())
    if spec["method"] == METHOD:
        controller._base.tangential = StationaryHoldCapTracking(**parameters)
    return controller, parameters


def validate_trace(spec, trace, parameters):
    original.previous._validate_full_schema(trace, spec["case"])
    original.previous._validate_case_schedule(trace, spec["case"])
    expected = {
        "method": spec["method"], "trace_schema_version": original.SCHEMA,
        "rotation_gain_scale": 1, "minimum_force_n": 6.0, "max_force_n": 8.0,
        "load_max_measurement_age_s": original.runner.MAX_MEASUREMENT_AGE_S,
        **{f"fault_{key}": value for key, value in asdict(spec["fault"]).items()},
    }
    for name, value in expected.items():
        actual = np.asarray(trace.get(name))
        if actual.shape != () or actual.item() != value:
            raise ValueError(f"trace metadata mismatch: {name}")
    provenance = original.previous.validate_fault_provenance(spec, trace)
    return {**provenance, **replay_compensation(trace, parameters, spec["method"], 0.002)}


def compare_runs(runs):
    expected = {(s["scenario"], s["seed"], s["method"]) for s in specifications()}
    indexed = {(s["scenario"], s["seed"], s["method"]): s for s in runs}
    if len(indexed) != len(runs) or set(indexed) != expected:
        raise ValueError("missing/duplicate/misidentified paired run")
    comparisons = []
    for method in ("hold_cap_tracking", METHOD):
        # All metrics were already projected onto the real -15 degree surface.
        # The old gate's yaw-zero specification is used only for phase/seed roles.
        group = [dict(row, method="hold_cap_tracking" if row["method"] == method else "adaptive6_8")
                 for row in runs if row["method"] in {"adaptive6_8", method}]
        for row in original.compare_runs(group, original.protocol_document()):
            comparisons.append({**row, "compared_method": method,
                                "failed_checks": [s.replace("hold_cap_tracking:", f"{method}:")
                                                  for s in row["failed_checks"]]})
    high_checks = []
    for seed in (11, 29):
        old, new = [next(w for w in indexed["constant_high", seed, method]["windows"]
                         if w["name"] == "ramp") for method in ("hold_cap_tracking", METHOD)]
        high_checks.append({"seed": seed, "position_improved": new["tangent_rmse_mm"] < old["tangent_rmse_mm"],
                            "velocity_improved": new["tangent_velocity_rmse_mm_s"] < old["tangent_velocity_rmse_mm_s"]})
    passed = (all(row["status"] == "PASS" for row in comparisons if row["compared_method"] == METHOD)
              and all(row["position_improved"] and row["velocity_improved"] for row in high_checks))
    return comparisons, high_checks, passed


def collect(directory, *, execute):
    verify_archive(REFERENCE, REFERENCE_SHA256)
    if json.loads((REFERENCE / "source_hashes.json").read_text()) != transfer.source_identity():
        raise ValueError("reference sources differ")
    runs, traces = [], {}
    for spec in specifications():
        path = (directory if spec["origin"] == "new" else REFERENCE) / spec["trace_path"]
        controller, parameters = make_controller(spec)
        if execute and spec["origin"] == "new":
            if path.exists():
                raise ValueError("refuse to overwrite a trial")
            frame = yaw_frame(-15)
            simulator = ScheduledSurfaceSimulator(spec["case"], frame, original.old.ARM)
            trace = original.runner._run_loop(spec["case"], controller, simulator, frame, spec["fault"]).trace
            trace.update({k: np.array(v) for k, v in {
                "rotation_gain_scale": 1, "minimum_force_n": 6.0, "max_force_n": 8.0,
                "method": METHOD, "trace_schema_version": original.SCHEMA,
                "load_max_measurement_age_s": original.runner.MAX_MEASUREMENT_AGE_S,
                **{f"fault_{k}": v for k, v in asdict(spec["fault"]).items()},
            }.items()})
            original.previous._save_trace(path, trace)
            print(f"executed {spec['scenario']} seed {spec['seed']}", flush=True)
        else:
            trace = original.previous._load_trace(path)
        audit = validate_trace(spec, trace, parameters)
        overall, phases, _ = original.dynamic.metrics(trace, spec["case"], parameters["max_force"])
        runs.append({name: spec[name] for name in (*transfer.RUN_KEYS, "origin", "trace_path")} | {
            "trace_sha256": _sha256(path), "case": _case_document(spec["case"]),
            "parameters": parameters, "overall": overall, "phases": phases,
            "windows": transfer.window_metrics(trace, spec["case"], original.protocol_document()),
            "audit": audit,
        })
        traces[spec["scenario"], spec["seed"], spec["method"]] = trace
    prefixes = [{"scenario": scenario, "seed": seed, **transfer.prefix_check(
        traces[scenario, seed, "adaptive6_8"], traces[scenario, seed, METHOD], 5.5)}
        for scenario in ("falling", "constant_high") for seed in (11, 29)]
    comparisons, high_checks, passed = compare_runs(runs)
    return {"runs": runs, "comparisons": comparisons, "high_load_checks": high_checks,
            "prefix_checks": prefixes, "status": "PASS" if passed and all(p["bit_exact"] for p in prefixes) else "FAIL",
            "default_changed": False, "eligible_for_default_change": False, "new_holdout": False}


def audit_archive(directory):
    directory = Path(directory)
    manifest = verify_archive(directory)
    protocol = protocol_document()
    expected = {"protocol.json", "source_hashes.json", "comparison.json"} | {
        s["trace_path"] for s in specifications() if s["origin"] == "new"}
    if (manifest.get("identity") != protocol["identity"]
            or manifest.get("new_simulations") != 4 or manifest.get("reused_runs") != 8
            or manifest.get("default_changed") is not False or manifest.get("new_holdout") is not False
            or set(manifest["artifact_sha256"]) != expected):
        raise ValueError("archive identity/scope/inventory differs")
    if not isinstance(manifest.get("source_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", manifest["source_commit"]
    ):
        raise ValueError("invalid execution commit")
    original.verify_comparison(json.loads((directory / "protocol.json").read_text()), protocol)
    if json.loads((directory / "source_hashes.json").read_text()) != source_identity():
        raise ValueError("source identity differs")
    rebuilt = collect(directory, execute=False)
    transfer.verify_comparison(json.loads((directory / "comparison.json").read_text()), rebuilt, protocol)
    return {"archive_integrity": "PASS", "comparison_status": rebuilt["status"],
            "evaluated_runs": 12, "new_simulations": 4, "reused_runs": 8,
            "default_changed": False, "new_holdout": False}


def run(directory):
    destination = Path(directory).absolute()
    if destination.exists() or any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError("output must be new and must not traverse a symlink")
    # Freeze the tracked implementation before integrating any new dynamics.
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT):
        raise ValueError("commit tracked changes before execution")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    protocol, sources = protocol_document(), source_identity()
    subprocess.check_output(["git", "ls-files", "--error-unmatch", "--", *sources], cwd=ROOT)
    check_replay_kernel()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    (staging / "traces").mkdir()
    print(f"staging {staging}; failures retain every partial trace", flush=True)
    _write_json(staging / "protocol.json", protocol)
    _write_json(staging / "source_hashes.json", sources)
    result = collect(staging, execute=True)
    if sources != source_identity() or protocol != protocol_document():
        raise ValueError("source/protocol changed during execution")
    _write_json(staging / "comparison.json", result)
    _write_json(staging / "manifest.json", {
        "identity": protocol["identity"], "source_commit": commit,
        "source_commit_role": "clean tracked execution-start HEAD; exact inputs in source_hashes.json",
        "new_simulations": 4,
        "reused_runs": 8, "default_changed": False, "new_holdout": False,
        "artifact_sha256": {p.relative_to(staging).as_posix(): _sha256(p)
                            for p in sorted(staging.rglob("*")) if p.is_file()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "mujoco")}},
    })
    (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
    audited = audit_archive(staging)
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination appeared during audit; staging preserved")
    staging.rename(destination)
    return audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--audit", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_archive(args.audit) if args.audit else run(args.output), indent=2))


if __name__ == "__main__":
    main()
