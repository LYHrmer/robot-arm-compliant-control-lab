"""Public preparation checks, with complete failure evidence; never training or a blind test."""

import argparse
import csv
import json
import os
import platform
import tempfile
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_env import PHYSICS_DT, POLICY_SUBSTEPS, SurfaceLearningEnv
from compliant_control_lab.surface_experiment import _output_path, _sha256, _source_hashes
from compliant_control_lab.surface_policy import SurfaceResidualConfig, friction_teacher_action
from compliant_control_lab.surface_readiness_cases import preparation_cases
from compliant_control_lab.surface_simulation import SurfaceScenario, SurfaceSimulationConfig
from compliant_control_lab.surface_splits import case_group_id

METHODS = ("zero_adaptive", "zero_friction", "friction_teacher_50hz")
STRESS_METHODS = ("plus_one", "minus_one", "alternating", "seeded_random")
ENGINEERING_TARGETS = {
    "contact_ratio_pct_min": 99.0,
    "tangent_rmse_mm_max": 10.0,
    "force_rmse_n_max": 2.0,
}
HARD_SAFETY_LIMITS = {
    "peak_force_n": 35.0,
    "max_penetration_mm": 2.0,
    "max_speed_m_s": 0.3,
    "saturation_pct": 0.0,
    "contact_loss_timeout_s": 0.1,
}
METRICS = (
    "force_rmse_n",
    "tangent_rmse_mm",
    "contact_ratio_pct",
    "peak_force_n",
    "seconds_over_35_n",
    "max_penetration_mm",
    "orientation_rmse_deg",
    "saturation_pct",
    "nominal_projection_pct",
    "residual_projection_pct",
    "intervention_pct",
    "max_speed_m_s",
    "max_kinematic_age_s",
    "max_force_age_s",
    "torque_mapping_max_error_nm",
    "max_contact_loss_s",
)


def audit_trace(trace: dict, case: dict) -> dict:
    """Recompute from executed physics rows, never environment episode metrics.

    RMSE/contact use evaluation_start; peak, penetration, clipping and intervention
    use the full executed episode. Nonfinite evidence is retained, not averaged away.
    """
    empty = dict.fromkeys(METRICS)
    if not trace or len(trace.get("time", [])) == 0:
        return {
            **empty,
            "trace_finite": False,
            "endpoint_valid": False,
            "evaluation_observed": False,
        }
    finite = all(
        np.all(np.isfinite(value))
        for value in trace.values()
        if np.asarray(value).dtype.kind in "biuf"
    )
    if not finite:
        return {
            **empty,
            "trace_finite": False,
            "endpoint_valid": False,
            "evaluation_observed": False,
        }
    time, dt = trace["time"], case["config"]["timestep"]
    if not np.allclose(time, np.arange(len(time)) * dt, rtol=0, atol=1e-12):
        raise ValueError("executed timestamps are not contiguous")
    count = len(time)
    endpoint_shapes = {
        "endpoint_time": (count,),
        "endpoint_position": (count, 3),
        "endpoint_linear_velocity": (count, 3),
        "endpoint_contact_gap_m": (count,),
        "endpoint_valid": (count,),
    }
    if any(np.asarray(trace[name]).shape != shape for name, shape in endpoint_shapes.items()):
        raise ValueError("endpoint evidence is missing or has the wrong shape")
    if not np.allclose(trace["endpoint_time"], time + dt, rtol=0, atol=1e-12):
        raise ValueError("endpoint timestamps do not match executed intervals")
    window = time >= case["config"]["evaluation_start"]
    angle = np.deg2rad(case["scenario"]["wall_yaw_deg"])
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    error = trace["position"] - trace["target_position"]
    tangent = error - np.outer(error @ normal, normal)
    gap = (np.array([0.4, 0.0, 0.0]) - trace["position"]) @ normal - 0.025
    endpoint_gap = (np.array([0.4, 0.0, 0.0]) - trace["endpoint_position"]) @ normal - 0.025
    if not np.allclose(gap, trace["true_contact_gap_m"], rtol=0, atol=1e-12):
        raise ValueError("stored gap differs from independent true-wall geometry")
    if not np.allclose(endpoint_gap, trace["endpoint_contact_gap_m"], rtol=0, atol=1e-12):
        raise ValueError("stored endpoint gap differs from independent true-wall geometry")
    mapped = (
        np.einsum("nij,ni->nj", trace["cartesian_jacobian"], trace["commanded_wrench"])
        + trace["joint_torque_offset"]
    )
    mapping_error = float(np.max(np.abs(mapped - trace["commanded_torque"])))
    if mapping_error > 1e-9 or not np.allclose(
        trace["applied_torque"],
        np.clip(
            trace["commanded_torque"], trace["lower_torque_limit"], trace["upper_torque_limit"]
        ),
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("recorded torque mapping/clipping is inconsistent")
    force = trace["true_normal_force"]
    lost = np.r_[False, window & (force <= 0.5), False].astype(int)
    loss_lengths = np.flatnonzero(np.diff(lost) == -1) - np.flatnonzero(np.diff(lost) == 1)
    output = {
        **empty,
        "trace_finite": True,
        "endpoint_valid": bool(np.all(trace["endpoint_valid"])),
        "evaluation_observed": bool(np.any(window)),
        "peak_force_n": float(np.max(force)),
        "seconds_over_35_n": float(np.count_nonzero(force > 35) * dt),
        "max_penetration_mm": float(
            1000 * max(np.maximum(-gap, 0).max(), np.maximum(-endpoint_gap, 0).max())
        ),
        "saturation_pct": float(
            100
            * np.mean(
                np.any(np.abs(trace["commanded_torque"] - trace["applied_torque"]) > 1e-9, axis=1)
            )
        ),
        "nominal_projection_pct": float(
            100 * np.mean(trace["torque_projection_scale"] < 1 - 1e-12)
        ),
        "residual_projection_pct": float(
            100 * np.mean(trace["residual_projection_scale"] < 1 - 1e-12)
        ),
        "intervention_pct": float(100 * np.mean(trace["intervention_active"])),
        "max_speed_m_s": float(
            max(
                np.linalg.norm(trace["linear_velocity"], axis=1).max(),
                np.linalg.norm(trace["endpoint_linear_velocity"], axis=1).max(),
            )
        ),
        "max_kinematic_age_s": float(np.max(time - trace["measured_kinematic_sample_time"])),
        "max_force_age_s": float(np.max(time - trace["measured_wrench_sample_time"])),
        "torque_mapping_max_error_nm": mapping_error,
        "max_contact_loss_s": float(loss_lengths.max(initial=0) * dt),
    }
    if np.any(window):
        output.update(
            force_rmse_n=float(
                np.sqrt(np.mean((force[window] - trace["target_normal_force"][window]) ** 2))
            ),
            tangent_rmse_mm=float(1000 * np.sqrt(np.mean(np.sum(tangent[window] ** 2, axis=1)))),
            contact_ratio_pct=float(100 * np.mean(force[window] > 0.5)),
            orientation_rmse_deg=float(
                np.rad2deg(np.sqrt(np.mean(trace["orientation_error_rad"][window] ** 2)))
            ),
        )
    return output


def _action(method, observation, step, rng):
    if method == "friction_teacher_50hz":
        return friction_teacher_action(observation[:33])
    if method == "plus_one":
        return np.ones(3)
    if method == "minus_one":
        return -np.ones(3)
    if method == "alternating":
        return np.full(3, 1.0 if step % 2 == 0 else -1.0)
    if method == "seeded_random":
        return rng.uniform(-1.0, 1.0, 3)
    return np.zeros(3)


def run_case(case: dict, method: str) -> tuple[dict, dict, dict]:
    """Return aggregate row, complete available trace and decision/event evidence."""
    if method not in (*METHODS, *STRESS_METHODS):
        raise ValueError("unsupported method")
    nominal = "adaptive" if method in {"zero_adaptive", "friction_teacher_50hz"} else "friction"
    config = SurfaceSimulationConfig(**case["config"])
    env = SurfaceLearningEnv(
        SurfaceFrame(case["controller_frame_rotation"]),
        scenario=SurfaceScenario(**case["scenario"]),
        config=config,
        task=LearningSurfaceTask(**case["task"]),
        nominal_kind=nominal,
    )
    started, events, actions, observations, rewards, latencies = perf_counter(), [], [], [], [], []
    terminal = truncated = illegal_action = False
    trace, failure, reset_info = {}, None, {}
    try:
        observation, reset_info = env.reset(seed=config.seed)
        observations.append(observation.copy())
        rng = np.random.default_rng(np.random.SeedSequence([20260906, config.seed]))
        while not (terminal or truncated):
            tick = perf_counter()
            action = _action(method, observation, len(actions), rng)
            if (
                action.shape != (3,)
                or not np.all(np.isfinite(action))
                or np.any(np.abs(action) > 1)
            ):
                illegal_action = True
                raise ValueError("benchmark produced an illegal action")
            observation, reward, terminal, truncated, info = env.step(action)
            latencies.append(perf_counter() - tick)
            actions.append(action.copy())
            observations.append(observation.copy())
            rewards.append(float(reward))
            events.append(
                {
                    **{
                        key: value
                        for key, value in info.items()
                        if not key.startswith("evaluator_")
                    },
                    "terminated": bool(terminal),
                    "truncated": bool(truncated),
                    "reward": float(reward),
                }
            )
        trace = env.result().trace
    except Exception as error:  # noqa: BLE001 - preserve failed episodes, never discard their evidence
        failure = f"{type(error).__name__}: {error}"
        terminal, truncated = True, False
        try:
            trace = env.result().trace
        except (RuntimeError, ValueError, IndexError):
            trace = {}
    finally:
        env.close()
    elapsed = perf_counter() - started
    stages = [
        stage
        for event in events
        for stage in event.get("action_stages", [])
        if stage["execution_status"] == "recorded"
    ]
    count = len(trace.get("time", []))
    reported = sum(event["physics_substeps"] for event in events)
    complete_log = count == len(stages) == reported
    if trace:
        trace["intervention_active"] = np.array(
            [
                any(
                    reason not in {"contact_lost", "contact_not_ready"}
                    for reason in stage["reasons"]
                )
                for stage in stages
            ]
        )
        trace["intervention_reasons"] = np.array(
            [";".join(stage["reasons"]) for stage in stages], dtype=str
        )
    try:
        audited = (
            audit_trace(trace, case)
            if complete_log
            else {
                **dict.fromkeys(METRICS),
                "trace_finite": False,
                "endpoint_valid": False,
                "evaluation_observed": False,
            }
        )
    except (KeyError, ValueError, TypeError) as error:
        audited = {
            **dict.fromkeys(METRICS),
            "trace_finite": False,
            "endpoint_valid": False,
            "evaluation_observed": False,
        }
        failure = f"trace audit: {type(error).__name__}: {error}"
    reasons = list(
        dict.fromkeys(reason for event in events for reason in event["termination_reasons"])
    )
    if failure:
        reasons.append("benchmark_exception")
    if not complete_log:
        reasons.append("incomplete_execution_log")
    if not audited["endpoint_valid"]:
        reasons.append("invalid_endpoint_evidence")
    physical_safe = (
        audited["trace_finite"]
        and audited["endpoint_valid"]
        and all(
            audited[name] <= limit
            for name, limit in HARD_SAFETY_LIMITS.items()
            if name != "contact_loss_timeout_s"
        )
        and audited["max_contact_loss_s"] < HARD_SAFETY_LIMITS["contact_loss_timeout_s"] - 1e-12
    )
    hard_complete = bool(
        truncated
        and not terminal
        and not reasons
        and complete_log
        and physical_safe
        and count == round(config.duration / config.timestep)
    )
    engineering = bool(
        hard_complete
        and audited["evaluation_observed"]
        and audited["contact_ratio_pct"] >= ENGINEERING_TARGETS["contact_ratio_pct_min"]
        and audited["force_rmse_n"] <= ENGINEERING_TARGETS["force_rmse_n_max"]
        and audited["tangent_rmse_mm"] <= ENGINEERING_TARGETS["tangent_rmse_mm_max"]
    )
    row = {
        "case_id": case.get("case_id", case_group_id(case)),
        "group_id": case_group_id(case),
        "split": case.get("split", "unspecified_public_development"),
        "method": method,
        "nominal_kind": nominal,
        "env_seed": config.seed,
        "simulation_seed": reset_info.get("simulation_seed"),
        "physics_steps": count,
        "policy_steps": len(actions),
        "episode_return": float(sum(rewards)),
        "terminated": terminal,
        "truncated": truncated,
        "termination_reasons": ";".join(reasons),
        "exception_detail": failure,
        "complete_execution_log": complete_log,
        "all_actions_legal": bool(
            not illegal_action
            and all(
                np.asarray(action).shape == (3,)
                and np.all(np.isfinite(action))
                and np.all(np.abs(action) <= 1)
                for action in actions
            )
        ),
        "hard_safe_completion": hard_complete,
        "engineering_targets_met": engineering,
        **audited,
        "wall_time_s": elapsed,
        "simulated_seconds_per_wall_second": count * config.timestep / elapsed,
        "physics_steps_per_wall_second": count / elapsed,
        "mean_decision_wall_ms": float(1000 * np.mean(latencies)) if latencies else None,
        "p95_decision_wall_ms": float(1000 * np.percentile(latencies, 95)) if latencies else None,
    }
    trace.update(
        decision_observation=np.asarray(observations),
        decision_action=np.asarray(actions).reshape((-1, 3)),
        decision_reward=np.asarray(rewards),
    )
    return (
        row,
        trace,
        {
            "reset": reset_info,
            "decisions": events,
            "illegal_action_attempted": illegal_action,
            "exception_detail": failure,
        },
    )


def _json_safe(value):
    """Use JSON null for unavailable numeric evidence; raw NPZ remains untouched."""
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


def _write_json(path, payload):
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _report(rows, stress):
    methods = {}
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        metrics = {}
        for name in METRICS:
            values = [row[name] for row in selected if row[name] is not None]
            metrics[name] = {
                "sample_count": len(values),
                "median": float(np.median(values)) if values else None,
                "worst": float(min(values) if name == "contact_ratio_pct" else max(values))
                if values
                else None,
            }
        methods[method] = {
            "run_count": len(selected),
            "hard_safe_completions": sum(row["hard_safe_completion"] for row in selected),
            "engineering_targets_met": sum(row["engineering_targets_met"] for row in selected),
            "metrics": metrics,
        }
    return {
        "scope": "Public preparation; complete local physics NPZ and decision/event JSON accompany these aggregates. Not a blind test.",
        "stress": stress,
        "methods": methods,
        "runs": rows,
        "acceptance_met": all(
            row["complete_execution_log"]
            and row["all_actions_legal"]
            and row["trace_finite"]
            and row["endpoint_valid"]
            and row["exception_detail"] is None
            for row in rows
        )
        if stress
        else all(row["hard_safe_completion"] for row in rows),
        "failed_cases": [
            {
                key: row[key]
                for key in (
                    "case_id",
                    "method",
                    "hard_safe_completion",
                    "engineering_targets_met",
                    "termination_reasons",
                    "force_rmse_n",
                    "tangent_rmse_mm",
                    "contact_ratio_pct",
                )
            }
            for row in rows
            if not row["hard_safe_completion"] or not row["engineering_targets_met"]
        ],
    }


def _summary(rows, stress):
    lines = [
        "# Surface learning preparation checks",
        "",
        "Public development checks, NOT a new holdout. No training or parameter tuning.",
        "Regular-run hard safety target: 100% normal horizon completion, no safety termination or missing logs.",
        "Separate engineering targets per case: wiping contact >=99%, tangent RMSE <=10 mm, force RMSE <=2 N.",
        "Stress runs may terminate safely; terminations and all available raw evidence remain published."
        if stress
        else "Failures are retained, including episodes without an observed wiping window.",
        "All runs use the same 50 Hz / 500 Hz environment. Teacher: adaptive nominal + feedback-only friction action.",
        "Metrics are independently reconstructed from full physics traces; no environment episode metrics are copied.",
        "",
        "| Method | Runs | Hard completions | Engineering targets | Force median [N] | Tangent median [mm] | Contact min [%] | Peak max [N] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        values = []
        for key, function in (
            ("force_rmse_n", np.median),
            ("tangent_rmse_mm", np.median),
            ("contact_ratio_pct", np.min),
            ("peak_force_n", np.max),
        ):
            finite = [row[key] for row in selected if row[key] is not None]
            values.append(f"{function(finite):.6g}" if finite else "NA")
        lines.append(
            f"| {method} | {len(selected)} | {sum(r['hard_safe_completion'] for r in selected)}/{len(selected)} | {sum(r['engineering_targets_met'] for r in selected)}/{len(selected)} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "Unavailable metrics are excluded only from the displayed aggregate, never from run/pass counts.",
        "Throughput and decision wall time include simulation, not policy-inference latency or real-time guarantees.",
        "Complete local physics NPZ and decision/event JSON accompany this report.",
        "",
        "## Cases missing completion or engineering targets",
        "",
        *[
            f"- {row['case_id']} / {row['method']}: hard={row['hard_safe_completion']}, engineering={row['engineering_targets_met']}; {row['termination_reasons'] or 'engineering/trace target not met'}."
            for row in rows
            if not row["hard_safe_completion"] or not row["engineering_targets_met"]
        ],
        "",
    ]
    return "\n".join(lines)


def generate_readiness_benchmark(output_dir, *, cases=None, methods=None, stress=False) -> Path:
    output = _output_path(output_dir)
    if output.exists():
        raise ValueError("benchmark output must be a new directory")
    cases = deepcopy(preparation_cases() if cases is None else list(cases))
    if not cases:
        raise ValueError("cases must be nonempty")
    for case in cases:
        actual = case_group_id(case)
        if case.get("group_id", actual) != actual:
            raise ValueError("case group_id does not match its physical/task configuration")
    methods = tuple(methods if methods is not None else STRESS_METHODS if stress else METHODS)
    if (
        not methods
        or len(methods) != len(set(methods))
        or not set(methods) <= set(STRESS_METHODS if stress else METHODS)
    ):
        raise ValueError("methods must be unique and belong to the selected mode")
    if stress:
        anchors = []
        for index in range(3):
            matching = [
                case for case in cases if case["scenario"]["name"] == f"preparation_{index:03d}"
            ]
            if not matching:
                raise ValueError("stress requires all three declared preparation anchors")
            anchors.append(min(matching, key=lambda case: case["config"]["seed"]))
        cases = anchors
    sources, started, rows = _source_hashes(), perf_counter(), []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".readiness-", dir=output.parent) as temporary:
        staged = Path(temporary) / "report"
        staged.mkdir()
        for case_index, case in enumerate(cases):
            for method in methods:
                row, trace, evidence = run_case(case, method)
                stem = f"case_{case_index:03d}_{method}"
                np.savez_compressed(staged / f"{stem}.npz", **trace)
                _write_json(staged / f"{stem}.json", evidence)
                rows.append(
                    {
                        **row,
                        "case_index": case_index,
                        "trace_file": f"{stem}.npz",
                        "events_file": f"{stem}.json",
                    }
                )
                print(
                    f"readiness {len(rows)}/{len(cases) * len(methods)}: {method} case {case_index} completed",
                    flush=True,
                )
        with (staged / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (staged / "summary.md").write_text(_summary(rows, stress), encoding="utf-8")
        _write_json(staged / "report.json", _report(rows, stress))
        if _source_hashes() != sources:
            raise RuntimeError("source/assets changed during benchmark; refusing publication")
        manifest = {
            "schema_version": 1,
            "identity": "surface-readiness-stress-v1"
            if stress
            else "surface-readiness-benchmark-v1",
            "new_holdout": False,
            "stress": stress,
            "methods": methods,
            "cases": cases,
            "engineering_targets": ENGINEERING_TARGETS,
            "hard_safe_completion_target_pct": None if stress else 100.0,
            "hard_safety_limits": HARD_SAFETY_LIMITS,
            "policy_period_s": PHYSICS_DT * POLICY_SUBSTEPS,
            "physics_timestep_s": PHYSICS_DT,
            "residual_config": vars(SurfaceResidualConfig()),
            "source_and_assets_sha256": sources,
            "versions": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in ("numpy", "mujoco", "gymnasium", "compliant-control-lab")
                },
            },
            "wall_time_s": perf_counter() - started,
            "artifact_sha256": {path.name: _sha256(path) for path in sorted(staged.iterdir())},
        }
        _write_json(staged / "manifest.json", manifest)
        (staged / "COMPLETE").write_text(_sha256(staged / "manifest.json") + "\n", encoding="utf-8")
        _output_path(output)
        if output.exists():
            raise FileExistsError("benchmark destination appeared during execution")
        os.rename(staged, output)
    return output


def audit_benchmark(path) -> dict:
    """Audit the COMPLETE local evidence set; compact exports cannot satisfy this check."""
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (directory / "COMPLETE").read_text().strip() != _sha256(directory / "manifest.json"):
        raise ValueError("COMPLETE/manifest mismatch")
    for name, digest in manifest["artifact_sha256"].items():
        if Path(name).name != name or _sha256(directory / name) != digest:
            raise ValueError(f"artifact mismatch: {name}")
    report = json.loads((directory / "report.json").read_text())
    rows = report["runs"]
    with (directory / "comparison.csv").open(newline="") as handle:
        stored = list(csv.DictReader(handle))
    if len(stored) != len(rows) or len(rows) != len(manifest["cases"]) * len(manifest["methods"]):
        raise ValueError("run matrix is incomplete")
    if {(row["case_index"], row["method"]) for row in rows} != {
        (i, method) for i in range(len(manifest["cases"])) for method in manifest["methods"]
    }:
        raise ValueError("run matrix duplicates or missing methods")
    for row, csv_row in zip(rows, stored):
        if csv_row != {key: "" if value is None else str(value) for key, value in row.items()}:
            raise ValueError("CSV/report row mismatch")
        case = manifest["cases"][row["case_index"]]
        with np.load(directory / row["trace_file"], allow_pickle=False) as archive:
            trace = {key: archive[key] for key in archive.files}
        evidence = json.loads((directory / row["events_file"]).read_text())
        events = evidence["decisions"]
        count = len(trace.get("time", []))
        stages = [
            stage
            for event in events
            for stage in event["action_stages"]
            if stage["execution_status"] == "recorded"
        ]
        complete = count == len(stages) == sum(event["physics_substeps"] for event in events)
        if (
            row["complete_execution_log"] != complete
            or row["physics_steps"] != count
            or row["group_id"] != case_group_id(case)
        ):
            raise ValueError("execution count/group mismatch")
        if not complete:
            raise ValueError("incomplete executed evidence")
        intervention = np.array(
            [
                any(
                    reason not in {"contact_lost", "contact_not_ready"}
                    for reason in stage["reasons"]
                )
                for stage in stages
            ]
        )
        if count and not np.array_equal(trace["intervention_active"], intervention):
            raise ValueError("intervention/event mismatch")
        for index, stage in enumerate(stages):
            for key in (
                "endpoint_time",
                "endpoint_position",
                "endpoint_linear_velocity",
                "endpoint_contact_gap_m",
                "endpoint_valid",
            ):
                if key not in stage or _json_safe(trace[key][index]) != stage[key]:
                    raise ValueError("endpoint NPZ/event evidence mismatch")
        metrics = audit_trace(trace, case)
        if any(row[key] != value for key, value in metrics.items()):
            raise ValueError("trace metric mismatch")
        if (
            row["policy_steps"] != len(events)
            or row["episode_return"] != sum(event["reward"] for event in events)
            or not np.array_equal(trace["decision_reward"], [event["reward"] for event in events])
        ):
            raise ValueError("decision/reward evidence mismatch")
        legal = (
            not evidence.get("illegal_action_attempted", False)
            and trace["decision_action"].shape == (len(events), 3)
            and np.all(np.isfinite(trace["decision_action"]))
            and np.all(np.abs(trace["decision_action"]) <= 1)
        )
        if row["all_actions_legal"] != bool(legal):
            raise ValueError("illegal or missing decision actions")
        terminal = bool(evidence["exception_detail"]) or (events and events[-1]["terminated"])
        truncated = not terminal and bool(events and events[-1]["truncated"])
        safe = (
            metrics["trace_finite"]
            and metrics["endpoint_valid"]
            and all(
                metrics[name] <= limit
                for name, limit in HARD_SAFETY_LIMITS.items()
                if name != "contact_loss_timeout_s"
            )
            and metrics["max_contact_loss_s"] < HARD_SAFETY_LIMITS["contact_loss_timeout_s"] - 1e-12
        )
        hard = bool(
            truncated
            and not terminal
            and safe
            and count == round(case["config"]["duration"] / case["config"]["timestep"])
        )
        engineering = bool(
            hard
            and metrics["evaluation_observed"]
            and metrics["contact_ratio_pct"] >= ENGINEERING_TARGETS["contact_ratio_pct_min"]
            and metrics["tangent_rmse_mm"] <= ENGINEERING_TARGETS["tangent_rmse_mm_max"]
            and metrics["force_rmse_n"] <= ENGINEERING_TARGETS["force_rmse_n_max"]
        )
        if (row["hard_safe_completion"], row["engineering_targets_met"]) != (hard, engineering):
            raise ValueError("readiness checks differ from trace/termination evidence")
    if report != _report(rows, manifest["stress"]) or (
        directory / "summary.md"
    ).read_text() != _summary(rows, manifest["stress"]):
        raise ValueError("aggregate report/summary mismatch")
    return {
        "matches": True,
        "run_count": len(rows),
        "physics_steps": sum(row["physics_steps"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=(*METHODS, *STRESS_METHODS))
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8")) if args.cases else None
    generate_readiness_benchmark(args.output, cases=cases, methods=args.methods, stress=args.stress)


if __name__ == "__main__":
    main()
