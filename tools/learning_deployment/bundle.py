"""Portable, hash-pinned inference bundles; no training framework is imported."""

import os
import shutil
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_policy_artifact import (
    freeze_evaluation_protocol,
    load_policy_artifact,
    policy_contract,
    save_policy_artifact,
    verify_evaluation_protocol,
)
from compliant_control_lab.surface_readiness_benchmark import METRICS
from tools.evaluate_surface_candidate import evaluate_candidate
from tools.learning_deployment.common import (
    output_path,
    read_json,
    runtime_identity,
    seal,
    verify_sealed,
    write_new,
)
from tools.surface_learning_pilot import THRESHOLDS
from tools.surface_mlp_actor import actor_from_artifact

ALGORITHMS = {"bc": ("il_friction_teacher", "adaptive"), "ppo": ("rl_residual", "friction")}
SCHEMA = "surface_simulation_deployment_v1"


def cases_for(cases, algorithm):
    assigned = deepcopy(cases)
    for case in assigned:
        case["nominal_kind"] = ALGORITHMS[algorithm][1]
    return assigned


def contract_for(cases, algorithm):
    purpose, nominal = ALGORITHMS[algorithm]
    return policy_contract(cases_for(cases, algorithm), purpose=purpose, nominal_kind=nominal)


def build_bundle(output, *, plan, candidates, training_hashes):
    output = output_path(output)
    if output.exists():
        raise FileExistsError("bundle destination must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".learning-bundle-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        write_new(staging / "cases.json", plan["cases"])
        protocols = {}
        for algorithm in ALGORITHMS:
            contract = contract_for(plan["cases"], algorithm)
            shutil.copyfile(candidates[algorithm], staging / f"{algorithm}.json")
            artifact = load_policy_artifact(staging / f"{algorithm}.json", expected_contract=contract)
            actor, _ = actor_from_artifact(artifact)
            if actor is None:
                raise ValueError("deployment requires a trained MLP, not a linear fixture")
            save_policy_artifact(
                staging / f"{algorithm}_nominal.json",
                {"kind": "linear_tanh", "weights": np.zeros((3, 49)).tolist(),
                 "bias": [0.0, 0.0, 0.0]}, contract=contract,
            )
            for label in (algorithm, f"{algorithm}_nominal"):
                protocols[label] = freeze_evaluation_protocol(
                    staging / f"{label}_protocol.json", staging / f"{label}.json",
                    expected_contract=contract, cases=cases_for(plan["cases"], algorithm),
                    metrics=METRICS, thresholds=plan["thresholds"],
                )
        if runtime_identity() != plan["runtime_identity"]:
            raise RuntimeError("source/runtime changed during bundle creation")
        digest = seal(staging, {
            "schema": SCHEMA, "target": "mujoco_simulation_only",
            "source_commit": plan["source_commit"], "new_holdout": False,
            "runtime_identity": plan["runtime_identity"],
            "protocol_sha256": protocols, "training_manifest_sha256": training_hashes,
            "policy_selection": plan["policy_selection"],
        })
        verify_bundle(staging, digest)
        if output.exists():
            raise FileExistsError("bundle destination appeared during creation")
        os.rename(staging, output)
    return digest


def verify_bundle(directory, expected_sha256):
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("an externally recorded bundle manifest SHA256 is required")
    directory = Path(directory)
    manifest = verify_sealed(directory, expected_sha256=expected_sha256)
    if manifest.get("schema") != SCHEMA or manifest.get("target") != "mujoco_simulation_only":
        raise ValueError("unsupported deployment target/schema")
    if manifest.get("runtime_identity") != runtime_identity():
        raise ValueError("deployment runtime/source differs; install the matching release")
    cases = read_json(directory / "cases.json")
    expected_files = {"cases.json"}
    for algorithm in ALGORITHMS:
        contract = contract_for(cases, algorithm)
        for label in (algorithm, f"{algorithm}_nominal"):
            expected_files.update((f"{label}.json", f"{label}_protocol.json"))
            artifact = load_policy_artifact(directory / f"{label}.json", expected_contract=contract)
            actor, _ = actor_from_artifact(artifact)
            if (actor is None) != label.endswith("_nominal"):
                raise ValueError("model and baseline payload roles differ")
            protocol = verify_evaluation_protocol(
                directory / f"{label}_protocol.json", directory / f"{label}.json",
                expected_protocol_sha256=manifest["protocol_sha256"][label],
                expected_contract=contract, cases=cases_for(cases, algorithm),
            )
            if (protocol["evaluation_split"] != "development_test"
                    or protocol["thresholds"] != THRESHOLDS
                    or protocol["metrics"] != list(METRICS)):
                raise ValueError("deployment evaluation scope or gates differ")
    if set(manifest["artifact_sha256"]) != expected_files:
        raise ValueError("deployment bundle inventory differs")
    return manifest


def profile_actor(bundle, algorithm):
    """Measure cold-load and bounded NumPy call latency on this host only."""
    start = time.perf_counter()
    cases = read_json(Path(bundle) / "cases.json")
    artifact = load_policy_artifact(
        Path(bundle) / f"{algorithm}.json", expected_contract=contract_for(cases, algorithm),
    )
    actor, _ = actor_from_artifact(artifact)
    cold_load = time.perf_counter() - start
    observations = np.random.default_rng(20261008).uniform(-3, 3, (512, 49))
    latencies = []
    for observation in observations:
        start = time.perf_counter()
        action = actor(observation)
        latencies.append(time.perf_counter() - start)
        if action.shape != (3,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1):
            raise ValueError("exported model violated the action contract")
    return {"samples": len(latencies), "cold_load_s": cold_load,
            "p99_s": float(np.percentile(latencies, 99)), "max_s": max(latencies),
            "budget_s": artifact.contract["policy_period_s"],
            "within_budget": max(latencies) <= artifact.contract["policy_period_s"],
            "scope": "this_host_synchronous_calls_not_hard_realtime"}


def run_bundle(bundle, expected_sha256, output, *, algorithm, nominal=False):
    manifest = verify_bundle(bundle, expected_sha256)
    if algorithm not in ALGORITHMS:
        raise ValueError("algorithm must be bc or ppo")
    directory = Path(bundle)
    label = f"{algorithm}_nominal" if nominal else algorithm
    purpose, kind = ALGORITHMS[algorithm]
    result = evaluate_candidate(
        directory / f"{label}.json", directory / f"{label}_protocol.json",
        cases_for(read_json(directory / "cases.json"), algorithm),
        manifest["protocol_sha256"][label], output, purpose=purpose, nominal_kind=kind,
    )
    verify_bundle(bundle, expected_sha256)
    return read_json(result / "report.json")


def summarize_evaluations(reports, profiles):
    """Keep software checks, absolute policy gates and relative benefits separate."""
    if set(reports) != {"bc", "bc_nominal", "ppo", "ppo_nominal"}:
        raise ValueError("both policies and both nominal baselines must be evaluated")
    if set(profiles) != set(ALGORITHMS):
        raise ValueError("both policies must have an inference profile")
    paired = {}
    for algorithm in ALGORITHMS:
        candidate = {r["case_id"]: r for r in reports[algorithm]["runs"]}
        nominal = {r["case_id"]: r for r in reports[f"{algorithm}_nominal"]["runs"]}
        if (len(candidate) != len(reports[algorithm]["runs"])
                or len(nominal) != len(reports[f"{algorithm}_nominal"]["runs"])):
            raise ValueError("candidate/nominal case grids contain duplicate case IDs")
        if not candidate or set(candidate) != set(nominal):
            raise ValueError("candidate/nominal case grids differ")
        paired[algorithm] = [{
            "case_id": case,
            **{f"{metric}_delta_vs_nominal":
               candidate[case][metric] - nominal[case][metric]
               if candidate[case][metric] is not None and nominal[case][metric] is not None else None
               for metric in ("tangent_rmse_mm", "force_rmse_n")},
        } for case in sorted(candidate)]
    engineering = all(r["all_episodes_succeeded"] for r in reports.values())
    engineering = engineering and all(p["within_budget"] for p in profiles.values())
    return {
        "engineering_status": "PASS" if engineering else "FAIL",
        "policy_acceptance": {a: reports[a]["acceptance_met"] for a in ALGORITHMS},
        "paired_comparisons": paired, "inference_profile": profiles,
        "new_holdout": False, "hardware_deployment": False,
        "scope": "single_seed_deployment_acceptance_not_policy_superiority_or_convergence",
    }


def accept_bundle(bundle, expected_sha256, output):
    """One installed command reruns the complete paired scope without Torch."""
    manifest = verify_bundle(bundle, expected_sha256)
    output = output_path(output)
    if output.exists():
        raise FileExistsError("acceptance output must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".learning-acceptance-", dir=output.parent) as temporary:
        staging = Path(temporary) / "acceptance"
        staging.mkdir()
        profiles = {a: profile_actor(bundle, a) for a in ALGORITHMS}
        reports = {}
        for algorithm in ALGORITHMS:
            for nominal in (False, True):
                label = f"{algorithm}_nominal" if nominal else algorithm
                print(f"stage: {label}", flush=True)
                reports[label] = run_bundle(bundle, expected_sha256, staging / label,
                                             algorithm=algorithm, nominal=nominal)
        summary = summarize_evaluations(reports, profiles)
        summary.update(bundle_manifest_sha256=expected_sha256,
                       source_commit=manifest["source_commit"])
        write_new(staging / "report.json", summary)
        seal(staging, {"schema": "surface_installed_acceptance_v1",
                       "bundle_manifest_sha256": expected_sha256})
        if output.exists():
            raise FileExistsError("acceptance output appeared during execution")
        os.rename(staging, output)
    return summary
