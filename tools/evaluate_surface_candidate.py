"""Evaluate one frozen finite-JSON surface-policy candidate; never train or load pickle."""

import argparse
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_env import SurfaceLearningEnv
from compliant_control_lab.surface_experiment import _output_path, _sha256, _source_hashes
from compliant_control_lab.surface_policy import SurfaceResidualConfig
from compliant_control_lab.surface_policy_artifact import (
    ActorEvaluator,
    load_policy_artifact,
    policy_contract,
    verify_evaluation_protocol,
)
from compliant_control_lab.surface_readiness_benchmark import (
    HARD_SAFETY_LIMITS,
    METRICS,
    audit_trace,
)
from compliant_control_lab.surface_simulation import SurfaceScenario, SurfaceSimulationConfig

try:
    from tools.surface_mlp_actor import actor_from_artifact
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    from surface_mlp_actor import actor_from_artifact


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path, value):
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _unavailable_metrics():
    return {
        **dict.fromkeys(METRICS),
        "trace_finite": False,
        "endpoint_valid": False,
        "evaluation_observed": False,
    }


def _run_case(case, artifact, actor, actor_descriptor):
    config = SurfaceSimulationConfig(**case["config"])
    env = SurfaceLearningEnv(
        SurfaceFrame(case["controller_frame_rotation"]),
        scenario=SurfaceScenario(**case["scenario"]),
        config=config,
        task=LearningSurfaceTask(**case["task"]),
        nominal_kind=artifact.contract["nominal_kind"],
        residual_config=SurfaceResidualConfig(**artifact.contract["residual_config"]),
    )
    evaluator = ActorEvaluator(artifact, actor=actor) if actor is not None else ActorEvaluator(artifact)
    observations, next_observations, actions, rewards = [], [], [], []
    terminated_flags, truncated_flags, events, stages, latencies = [], [], [], [], []
    terminal_validity, actor_reasons = [], []
    failure, trace, reset_info = None, {}, {}
    terminated = truncated = False
    try:
        observation, reset_info = env.reset(seed=config.seed)
        while not (terminated or truncated):
            observations.append(observation.copy())
            observation, reward, terminated, truncated, info = evaluator.step(env, observation)
            recorded = [
                stage for stage in info["action_stages"] if stage["execution_status"] == "recorded"
            ]
            stages.extend(recorded)
            actor_info = info["actor_evaluation"]
            latencies.append(float(actor_info["latency_s"]))
            terminal_validity.append(bool(info["terminal_observation_valid"]))
            actor_reasons.append(actor_info["reason"] or "")
            action = (
                np.asarray(info["action_stages"][0]["policy_action"], dtype=np.float64)
                if info["action_stages"]
                else np.zeros(3)
            )
            actions.append(action.copy())
            next_observations.append(observation.copy())
            rewards.append(float(reward))
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            events.append(
                {
                    "decision_index": len(events),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": info,
                }
            )
        trace = env.result().trace
    except Exception as error:  # noqa: BLE001 - retain the failed case and available evidence
        failure = f"{type(error).__name__}: {error}"
        terminated, truncated = True, False
        try:
            trace = env.result().trace
        except (RuntimeError, ValueError, IndexError):
            trace = {}
    finally:
        env.close()

    count = len(trace.get("time", []))
    reported = sum(event["info"].get("physics_substeps", 0) for event in events)
    complete_log = count == len(stages) == reported
    if trace:
        trace["intervention_active"] = np.asarray(
            [
                any(
                    reason not in {"contact_lost", "contact_not_ready"}
                    for reason in stage["reasons"]
                )
                for stage in stages
            ],
            dtype=bool,
        )
        trace["intervention_reasons"] = np.asarray(
            [";".join(stage["reasons"]) for stage in stages], dtype=str
        )
    try:
        metrics = audit_trace(trace, case) if complete_log else _unavailable_metrics()
    except (KeyError, TypeError, ValueError) as error:
        metrics = _unavailable_metrics()
        failure = f"trace audit: {type(error).__name__}: {error}"
    actor_failures = sum(
        event["info"]["actor_evaluation"]["reason"] is not None for event in events
    )
    physical_safe = (
        metrics["trace_finite"]
        and metrics["endpoint_valid"]
        and all(
            metrics[name] <= limit
            for name, limit in HARD_SAFETY_LIMITS.items()
            if name != "contact_loss_timeout_s"
        )
        and metrics["max_contact_loss_s"] < HARD_SAFETY_LIMITS["contact_loss_timeout_s"] - 1e-12
    )
    episode_success = bool(
        truncated
        and not terminated
        and failure is None
        and actor_failures == 0
        and complete_log
        and metrics["trace_finite"]
        and metrics["endpoint_valid"]
        and metrics["evaluation_observed"]
        and physical_safe
        and count == round(config.duration / config.timestep)
    )
    row = {
        "case_id": case["case_id"],
        "group_id": case["group_id"],
        "split": case["split"],
        "simulation_seed": reset_info.get("simulation_seed"),
        "physics_steps": count,
        "policy_steps": len(events),
        "episode_return": float(np.sum(rewards)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reasons": list(
            dict.fromkeys(
                reason
                for event in events
                for reason in event["info"].get("termination_reasons", ())
            )
        ),
        "complete_execution_log": complete_log,
        "actor_failure_count": actor_failures,
        "max_actor_latency_s": max(latencies, default=None),
        "exception_detail": failure,
        "episode_success": episode_success,
        "independent_physical_gates_met": bool(physical_safe),
        **metrics,
    }
    trace.update(
        decision_observation=np.asarray(observations, dtype=np.float64).reshape((-1, 49)),
        decision_action=np.asarray(actions, dtype=np.float64).reshape((-1, 3)),
        decision_reward=np.asarray(rewards, dtype=np.float64),
        decision_next_observation=np.asarray(next_observations, dtype=np.float64).reshape((-1, 49)),
        decision_terminated=np.asarray(terminated_flags, dtype=bool),
        decision_truncated=np.asarray(truncated_flags, dtype=bool),
        decision_terminal_observation_valid=np.asarray(terminal_validity, dtype=bool),
        decision_actor_latency_s=np.asarray(latencies, dtype=np.float64),
        decision_actor_succeeded=np.asarray([not reason for reason in actor_reasons], dtype=bool),
        decision_actor_failure_reason=np.asarray(actor_reasons, dtype=str),
    )
    evidence = {
        "reset": reset_info,
        "decisions": events,
        "exception_detail": failure,
        "candidate_execution": actor_descriptor,
    }
    return row, trace, evidence


def _aggregate(rows, protocol):
    aggregates, gates = {}, {}
    for metric in protocol["metrics"]:
        values = [row.get(metric) for row in rows]
        available = all(
            value is not None and not isinstance(value, bool) and np.isfinite(value)
            for value in values
        )
        aggregates[metric] = {"available": available, "values": values}
        if metric not in protocol["thresholds"]:
            continue
        rule = protocol["thresholds"][metric]
        aggregate = None
        if available:
            aggregate = float(
                {
                    "max": np.max,
                    "min": np.min,
                    "median": np.median,
                    "p95": lambda items: np.percentile(items, 95),
                }[rule["aggregation"]](values)
            )
        passed = False
        if available:
            if rule["operator"] == "<=":
                passed = aggregate <= rule["value"]
            elif rule["operator"] == ">=":
                passed = aggregate >= rule["value"]
            else:
                passed = aggregate == rule["value"]
        aggregates[metric].update(aggregation=rule["aggregation"], value=aggregate)
        gates[metric] = {"rule": rule, "actual": aggregate, "passed": passed}
    all_metrics_available = all(item["available"] for item in aggregates.values())
    all_episodes_succeeded = all(row["episode_success"] for row in rows)
    return {
        "metrics": aggregates,
        "gates": gates,
        "all_metrics_available": all_metrics_available,
        "all_episodes_succeeded": all_episodes_succeeded,
        "acceptance_met": bool(
            all_metrics_available
            and all_episodes_succeeded
            and gates
            and all(gate["passed"] for gate in gates.values())
        ),
    }


def _summary(report):
    lines = [
        "# Frozen surface candidate evaluation",
        "",
        "Public validation/development evidence; new_holdout=false. No training occurred.",
        "Executable formats are finite-JSON linear_tanh or the restricted NumPy MLP runner.",
        "No pickle, candidate imports or implicit executable deserialization are permitted.",
        "Actor timeout remains a check after the bounded synchronous NumPy call returns.",
        "",
        f"Acceptance: {'PASS' if report['acceptance_met'] else 'FAIL'}",
        f"Complete selected scope: {len(report['runs'])}/{report['expected_case_count']} cases.",
        "",
        "| Case | Success | Terminated | Tracking observed | Actor failures | Return |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        lines.append(
            f"| {row['case_id']} | {row['episode_success']} | {row['terminated']} | "
            f"{row['evaluation_observed']} | {row['actor_failure_count']} | "
            f"{row['episode_return']:.6g} |"
        )
    lines += [
        "",
        "Missing metrics and every failed episode force FAIL; reward cannot override safety.",
        "",
    ]
    return "\n".join(lines)


def evaluate_candidate(
    candidate,
    protocol,
    cases,
    expected_protocol_sha256,
    output,
    *,
    purpose,
    nominal_kind,
):
    """Verify frozen identities, execute exactly the selected cases and atomically publish evidence."""
    cases = deepcopy(list(cases))
    contract = policy_contract(cases, purpose=purpose, nominal_kind=nominal_kind)
    frozen = verify_evaluation_protocol(
        protocol,
        candidate,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_contract=contract,
        cases=cases,
    )
    artifact = load_policy_artifact(candidate, expected_contract=contract)
    actor, actor_descriptor = actor_from_artifact(artifact)
    selected_by_id = {case["case_id"]: case for case in cases}
    selected = [selected_by_id[case_id] for case_id in frozen["evaluation_case_ids"]]
    if len(selected) != len(frozen["evaluation_case_ids"]):
        raise ValueError("protocol evaluation case matrix is incomplete")

    output = _output_path(output)
    if output.exists():
        raise ValueError("candidate evaluation output must be a new directory")
    sources = _source_hashes()
    script_path = Path(__file__).resolve()
    script_sha256 = _sha256(script_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-candidate-", dir=output.parent) as temporary:
        staging = Path(temporary) / "evaluation"
        staging.mkdir()
        rows = []
        for index, case in enumerate(selected):
            row, trace, evidence = _run_case(case, artifact, actor, actor_descriptor)
            stem = f"case_{index:03d}"
            np.savez_compressed(staging / f"{stem}.npz", **trace)
            _write_json(staging / f"{stem}_events.json", evidence)
            rows.append(
                {
                    **row,
                    "trace_file": f"{stem}.npz",
                    "events_file": f"{stem}_events.json",
                }
            )
        aggregate = _aggregate(rows, frozen)
        report = {
            "schema": "surface_candidate_evaluation_v1",
            "new_holdout": False,
            "evaluation_split": frozen["evaluation_split"],
            "evaluation_case_ids": frozen["evaluation_case_ids"],
            "expected_case_count": len(frozen["evaluation_case_ids"]),
            "runs": rows,
            **aggregate,
        }
        _write_json(staging / "report.json", report)
        (staging / "summary.md").write_text(_summary(report), encoding="utf-8")
        if (
            _source_hashes() != sources
            or _sha256(script_path) != script_sha256
            or _sha256(Path(candidate)) != artifact.artifact_sha256
            or _sha256(Path(protocol)) != expected_protocol_sha256
        ):
            raise RuntimeError(
                "source, script, candidate or protocol changed during candidate evaluation"
            )
        actor_from_artifact(artifact)  # Recheck runner/package identity before publication.
        manifest = {
            "schema": "surface_candidate_evaluation_manifest_v1",
            "new_holdout": False,
            "scope": "complete_frozen_public_evaluation_selection",
            "purpose": purpose,
            "nominal_kind": nominal_kind,
            "evaluation_split": frozen["evaluation_split"],
            "evaluation_case_ids": frozen["evaluation_case_ids"],
            "candidate_artifact_sha256": artifact.artifact_sha256,
            "candidate_execution": actor_descriptor,
            "protocol_sha256": expected_protocol_sha256,
            "policy_contract": contract,
            "source_and_assets_sha256": sources,
            "evaluation_script_sha256": script_sha256,
            "artifact_sha256": {path.name: _sha256(path) for path in sorted(staging.iterdir())},
        }
        _write_json(staging / "manifest.json", manifest)
        (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
        _output_path(output)
        if output.exists():
            raise FileExistsError("candidate evaluation destination appeared during execution")
        os.rename(staging, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=("rl_residual", "il_friction_teacher"), required=True)
    parser.add_argument("--nominal-kind", choices=("friction", "adaptive"), required=True)
    args = parser.parse_args(argv)
    output = evaluate_candidate(
        args.candidate,
        args.protocol,
        json.loads(args.cases.read_text()),
        args.expected_protocol_sha256,
        args.output,
        purpose=args.purpose,
        nominal_kind=args.nominal_kind,
    )
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
