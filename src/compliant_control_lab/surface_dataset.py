"""Real 50 Hz teacher episodes and independent audits; no training or blind-test claim."""

import argparse
import json
import os
import platform
import tempfile
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

import gymnasium
import mujoco
import numpy as np

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_env import (
    OBSERVATION_NAMES,
    OBSERVATION_SCALES,
    OBSERVATION_SCHEMA,
    OBSERVATION_UNITS,
    PHYSICS_DT,
    POLICY_SUBSTEPS,
    REWARD_WEIGHTS,
    SurfaceLearningEnv,
)
from compliant_control_lab.surface_experiment import _output_path, _sha256, _source_hashes
from compliant_control_lab.surface_policy import (
    OBSERVATION_NAMES as TEACHER_NAMES,
)
from compliant_control_lab.surface_policy import (
    OBSERVATION_SCALES as TEACHER_SCALES,
)
from compliant_control_lab.surface_policy import SurfaceResidualConfig, friction_teacher_action
from compliant_control_lab.surface_readiness_cases import PREPARATION_SPLIT
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    yaw_frame,
)
from compliant_control_lab.surface_splits import SPLIT_NAMES, case_group_id, split_cases

DATASET_SCHEMA = "surface_teacher_dataset_v1"
ACTION_STAGE = "normalized_local_residual"
SPLIT_SEED = 20260906
TRANSITION_CONTRACT = {
    "policy_period_s": PHYSICS_DT * POLICY_SUBSTEPS,
    "reward_semantics": "sum_over_executed_physics_substeps_with_terminal_penalty",
    "bootstrap_mask_definition": "not terminated and terminal_observation_valid",
    "discount_semantics": "gamma_per_policy_step ** (physics_substeps / policy_substeps)",
    "teacher_input_names": list(TEACHER_NAMES),
    "teacher_input_indices": list(range(len(TEACHER_NAMES))),
}


def _json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _finite_metadata(value):
    if isinstance(value, dict):
        return {key: _finite_metadata(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _collect_episode(case):
    config, residual = SurfaceSimulationConfig(**case["config"]), SurfaceResidualConfig()
    task, scenario = LearningSurfaceTask(**case["task"]), SurfaceScenario(**case["scenario"])
    env = SurfaceLearningEnv(
        SurfaceFrame(case["controller_frame_rotation"]),
        scenario=scenario,
        config=config,
        task=task,
        nominal_kind="adaptive",
        residual_config=residual,
    )
    rows, reasons = [], []
    try:
        observation, initial = env.reset(seed=config.seed)
        simulation_seed = initial["simulation_seed"]
        before = 0.0
        while True:
            if not np.all(np.isfinite(observation)):
                raise RuntimeError("nonfinite actor observation; dataset not published")
            action = friction_teacher_action(observation[:33], residual)
            next_observation, reward, terminated, truncated, info = env.step(action)
            valid = bool(info["terminal_observation_valid"])
            rows.append(
                {
                    "observations": observation.copy(),
                    "actions": action.copy(),
                    "rewards": reward,
                    "next_observations": next_observation.copy() if valid else np.full(49, np.nan),
                    "terminated": terminated,
                    "truncated": truncated,
                    "terminal_observation_valid": valid,
                    "time_before_s": before,
                    "time_after_s": info["elapsed_s"],
                    "physics_substeps": info["physics_substeps"],
                    "attempted_substeps": info["attempted_substeps"],
                    "attempted_execution_status": "|".join(
                        stage["execution_status"] for stage in info["action_stages"]
                    ),
                    "termination_reasons": "|".join(info["termination_reasons"]),
                }
            )
            reasons.extend(
                "|".join(stage["reasons"])
                for stage in info["action_stages"]
                if stage["execution_status"] == "recorded"
            )
            if terminated or truncated:
                break
            if not valid or info["physics_substeps"] <= 0:
                raise RuntimeError("nonterminal transition cannot lack a valid next observation")
            observation, before = next_observation, info["elapsed_s"]
        try:
            result = env.result()
            trace, metrics = result.trace, _finite_metadata(result.metrics())
        except RuntimeError:
            if info["elapsed_s"] != 0:
                raise
            trace, metrics = {}, {}
        arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
        arrays.update({f"trace__{key}": value for key, value in trace.items()})
        arrays["trace__residual_reasons"] = np.asarray(reasons, dtype=str)
        nonfinite = sorted(
            key
            for key, value in trace.items()
            if value.dtype.kind in "biuf" and not np.all(np.isfinite(value))
        )
        metadata = {
            "case_id": str(case["case_id"]),
            "group_id": case["group_id"],
            "split": case["split"],
            "env_seed": config.seed,
            "simulation_seed": simulation_seed,
            "actual_config": asdict(replace(config, seed=simulation_seed)),
            "actual_scenario": asdict(scenario),
            "actual_task": asdict(task),
            "transition_count": len(rows),
            "physics_count": len(reasons),
            "return": float(np.sum(arrays["rewards"])),
            "termination_reasons": list(info["termination_reasons"]),
            "exception_detail": info["exception_detail"],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "nonfinite_trace_fields": nonfinite,
            "metrics": metrics,
            "gates": {
                "environment_termination_free": not terminated,
                "evaluation_observed": bool(metrics.get("evaluation_observed", False)),
            },
        }
        return arrays, metadata
    finally:
        env.close()


def collect_demonstrations(output_dir, cases: list[dict], *, split_seed=SPLIT_SEED) -> Path:
    """Publish all planned teacher episodes atomically, including terminated failures.

    Configuration/reset errors abort publication rather than silently omitting a
    planned case. Invalid terminal next observations are NaN with an explicit mask;
    they are never passed to the teacher. No data-dependent normalization is fitted.
    """
    output = _output_path(output_dir)
    if output.exists():
        raise ValueError("dataset output must be a new directory")
    if tuple(OBSERVATION_NAMES[: len(TEACHER_NAMES)]) != tuple(TEACHER_NAMES):
        raise ValueError("teacher input prefix does not match the environment schema")
    assigned = split_cases(cases, seed=split_seed, **PREPARATION_SPLIT)
    for original, labeled in zip(cases, assigned):
        for key in ("group_id", "split"):
            if key in original and original[key] != labeled[key]:
                raise ValueError(f"supplied {key} conflicts with declared split seed")
    cases = assigned
    identifiers = [case.get("case_id") for case in cases]
    if not all(isinstance(value, str) and value.strip() for value in identifiers):
        raise ValueError("case_id values must be nonempty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("case_id values must be unique")
    for case in cases:
        if case.get("nominal_kind", "adaptive") != "adaptive":
            raise ValueError("friction teacher demonstrations require adaptive nominal")
        SurfaceFrame(case["controller_frame_rotation"])
        LearningSurfaceTask(**case["task"])
    sources = _source_hashes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-dataset-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        episodes = []
        for index, case in enumerate(cases):
            arrays, metadata = _collect_episode(case)
            filename = f"episode_{index:05d}.npz"
            np.savez_compressed(staging / filename, **arrays)
            episodes.append({**metadata, "file": filename, "sha256": _sha256(staging / filename)})
        if sources != _source_hashes():
            raise RuntimeError("source changed during demonstration collection")
        manifest = {
            "schema": DATASET_SCHEMA,
            "new_holdout": False,
            "corpus_scope": "full",
            "observation_schema": OBSERVATION_SCHEMA,
            "observation_names": list(OBSERVATION_NAMES),
            "observation_scales": list(OBSERVATION_SCALES),
            "observation_units": list(OBSERVATION_UNITS),
            "action_stage": ACTION_STAGE,
            "action_names": ["normal", "tangent1", "tangent2"],
            "action_units": "normalized",
            "nominal_kind": "adaptive",
            "residual_config": asdict(SurfaceResidualConfig()),
            "physics_period_s": PHYSICS_DT,
            "policy_substeps": POLICY_SUBSTEPS,
            "reward_weights": REWARD_WEIGHTS,
            "transition_contract": TRANSITION_CONTRACT,
            "invalid_next_observation": "NaN placeholder; terminal_observation_valid=False; do not bootstrap",
            "group_split": {"seed": split_seed, **PREPARATION_SPLIT},
            "quality_scope": "Environment termination gates, not teacher tracking acceptance or a new holdout.",
            "cases": cases,
            "episodes": episodes,
            "source_and_assets_sha256": sources,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "mujoco": mujoco.__version__,
                "gymnasium": gymnasium.__version__,
            },
        }
        _json(staging / "manifest.json", manifest)
        (staging / "COMPLETE").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="utf-8"
        )
        audit_dataset(staging)
        if _output_path(output).exists():
            raise ValueError("dataset output appeared during collection")
        os.rename(staging, output)
    return output


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _episode_audit(arrays, entry, case, residual):
    obs, action, nxt = (arrays[name] for name in ("observations", "actions", "next_observations"))
    n = len(obs)
    _require(
        n > 0 and obs.shape == nxt.shape == (n, 49) and action.shape == (n, 3),
        "actor shape mismatch",
    )
    _require(np.all(np.isfinite(obs)) and np.max(np.abs(obs)) <= 3, "invalid actor observation")
    _require(np.all(np.isfinite(action)) and np.max(np.abs(action)) <= 1, "invalid actor action")
    for key in (
        "rewards",
        "terminated",
        "truncated",
        "terminal_observation_valid",
        "time_before_s",
        "time_after_s",
        "physics_substeps",
        "attempted_substeps",
        "attempted_execution_status",
        "termination_reasons",
    ):
        _require(arrays[key].shape == (n,), f"invalid transition field {key}")
    for key in ("terminated", "truncated", "terminal_observation_valid"):
        _require(arrays[key].dtype.kind == "b", f"{key} must be boolean")
    term, trunc, valid = (
        arrays[key] for key in ("terminated", "truncated", "terminal_observation_valid")
    )
    _require(
        not np.any((term | trunc)[:-1]) and bool(term[-1]) != bool(trunc[-1]),
        "done must occur exactly at episode end",
    )
    _require(
        np.all(valid[:-1]) and (valid[-1] or term[-1]), "invalid next observation outside failure"
    )
    _require(
        np.all(np.isfinite(nxt[valid])) and np.all(np.abs(nxt[valid]) <= 3),
        "invalid next observation",
    )
    _require(np.all(np.isnan(nxt[~valid])), "invalid terminal next observation must be masked NaN")
    _require(np.array_equal(obs[1:], nxt[:-1]), "observation chain crosses or skips transitions")
    # Independent teacher formula: do not call friction_teacher_action here.
    physical = obs[:, :33] * np.asarray(TEACHER_SCALES)
    velocity = physical[:, 9:12].copy()
    velocity[:, 0] = 0
    amplitude = np.minimum(
        residual.teacher_nominal_mu * np.maximum(physical[:, 0], 0), residual.teacher_max_force_n
    )
    expected = (
        amplitude[:, None]
        * velocity
        / np.sqrt(np.sum(velocity**2, axis=1) + residual.teacher_velocity_scale**2)[:, None]
    )
    expected = np.clip(expected / np.asarray(residual.action_bounds_n), -1, 1)
    _require(np.allclose(action, expected, rtol=0, atol=1e-12), "teacher label mismatch")
    counts, before, after = (
        arrays[key] for key in ("physics_substeps", "time_before_s", "time_after_s")
    )
    _require(
        counts.dtype.kind in "iu" and np.all((counts >= 0) & (counts <= 10)),
        "invalid substep count",
    )
    _require(np.all(counts[:-1] == 10), "nonterminal action was not held for 10 steps")
    for count, attempted, status in zip(
        counts, arrays["attempted_substeps"], arrays["attempted_execution_status"]
    ):
        statuses = str(status).split("|") if str(status) else []
        _require(
            len(statuses) == attempted
            and statuses.count("recorded") == count
            and all(item in {"recorded", "unconfirmed"} for item in statuses),
            "attempted/executed evidence mismatch",
        )
    expected_after = np.cumsum(counts) * 0.002
    _require(
        np.allclose(after, expected_after, atol=1e-12, rtol=0)
        and np.allclose(before, np.r_[0, expected_after[:-1]], atol=1e-12, rtol=0),
        "transition clock mismatch",
    )
    duration = case["config"]["duration"]
    _require(
        after[-1] <= duration + 1e-12 and (not trunc[-1] or abs(after[-1] - duration) < 1e-12),
        "invalid horizon truncation",
    )
    trace = {key[7:]: value for key, value in arrays.items() if key.startswith("trace__")}
    size = int(np.sum(counts))
    _require(len(trace.get("time", [])) == size, "executed trace lost or added rows")
    nonfinite = sorted(
        key
        for key, value in trace.items()
        if value.dtype.kind in "biuf" and not np.all(np.isfinite(value))
    )
    _require(
        nonfinite == entry["nonfinite_trace_fields"] and (not nonfinite or term[-1]),
        "nonfinite failure evidence mismatch",
    )
    reward_rows = np.zeros(size)
    if size:
        _require(
            all(
                np.ndim(value) == 0 or key == "controller_frame_rotation" or len(value) == size
                for key, value in trace.items()
            ),
            "incomplete trace column",
        )
        _require(
            np.allclose(trace["time"], np.arange(size) * 0.002, atol=1e-12, rtol=0),
            "physics clock mismatch",
        )
        _require(
            np.array_equal(trace["policy_action"], np.repeat(action, counts, axis=0)),
            "held policy action mismatch",
        )
        _require(
            np.array_equal(trace["policy_step_index"], np.repeat(np.arange(n), counts)),
            "policy step index mismatch",
        )
        _require(np.all(trace["policy_action_valid"]), "teacher action marked invalid")
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        present = starts < size
        index = starts[present]
        rotation = np.asarray(case["controller_frame_rotation"])
        measured = np.c_[
            (trace["target_position"][index] - trace["measured_position"][index])
            @ rotation
            / [0.02, 0.05, 0.05],
            (trace["target_linear_velocity"][index] - trace["measured_linear_velocity"][index])
            @ rotation
            / 0.1,
            trace["target_linear_velocity"][index] @ rotation / 0.1,
            trace["target_normal_force"][index] / 12,
        ]
        _require(
            np.allclose(obs[present, 3:13], np.clip(measured, -3, 3), atol=1e-12, rtol=0),
            "observation/trace input mismatch",
        )
        ages = np.c_[
            trace["time"] - trace["measured_kinematic_sample_time"],
            trace["time"] - trace["measured_wrench_sample_time"],
        ]
        encoders = np.c_[
            trace["q"][index] / np.pi, trace["joint_velocity"][index] / 2, ages[index] / 0.1
        ]
        _require(
            np.all(ages >= -1e-12)
            and np.allclose(obs[present, 33:], np.clip(encoders, -3, 3), atol=1e-12, rtol=0),
            "future or incorrect encoder/sensor age",
        )
        normal = yaw_frame(case["scenario"]["wall_yaw_deg"]).rotation[:, 0]
        endpoint_valid = trace["endpoint_valid"]
        _require(
            endpoint_valid.dtype.kind == "b" and endpoint_valid.shape == (size,),
            "invalid endpoint validity mask",
        )
        _require(
            np.all(endpoint_valid[:-1]) and (endpoint_valid[-1] or (term[-1] and not valid[-1])),
            "unobserved endpoint outside explicit terminal observation failure",
        )
        endpoint_fields = (
            "endpoint_time",
            "endpoint_position",
            "endpoint_linear_velocity",
            "endpoint_contact_gap_m",
        )
        _require(
            all(np.all(np.isfinite(trace[key][endpoint_valid])) for key in endpoint_fields),
            "valid endpoint is nonfinite",
        )
        _require(
            np.allclose(
                trace["endpoint_time"][endpoint_valid],
                (trace["time"] + 0.002)[endpoint_valid],
                rtol=0,
                atol=1e-12,
            ),
            "endpoint clock mismatch",
        )
        endpoint_gap = (np.array([0.4, 0, 0]) - trace["endpoint_position"]) @ normal - 0.025
        _require(
            np.allclose(
                endpoint_gap[endpoint_valid],
                trace["endpoint_contact_gap_m"][endpoint_valid],
                rtol=0,
                atol=1e-12,
            ),
            "endpoint geometry mismatch",
        )
        endpoint_speed = np.linalg.norm(trace["endpoint_linear_velocity"], axis=1)
        for violations, reason in (
            (endpoint_speed > 0.3, "speed_limit"),
            (-endpoint_gap > 0.002, "penetration_limit"),
        ):
            _require(
                not np.any(violations[:-1])
                and (not violations[-1] or (term[-1] and reason in entry["termination_reasons"])),
                "integrated endpoint violation did not terminate immediately",
            )
        finite_rows = np.logical_and.reduce(
            [
                np.all(np.isfinite(value).reshape(size, -1), axis=1)
                for key, value in trace.items()
                if value.dtype.kind in "biuf"
                and value.ndim
                and key != "controller_frame_rotation"
                and not key.startswith("endpoint_")
            ]
        )
        error = trace["position"] - trace["target_position"]
        tangent = error - (error @ normal)[:, None] * normal
        costs = {
            "force": np.clip(
                ((trace["true_normal_force"] - trace["target_normal_force"]) / 12) ** 2, 0, 1
            ),
            "tangent": np.clip(np.sum(tangent**2, axis=1) / 0.02**2, 0, 1),
            "contact": (trace["time"] >= case["config"]["evaluation_start"])
            & (trace["true_normal_force"] <= 0.5),
            "orientation": np.clip((trace["orientation_error_rad"] / 0.2) ** 2, 0, 1),
            "action": np.mean(trace["policy_action"] ** 2, axis=1),
            "intervention": np.array(
                [
                    bool(set(str(value).split("|")) - {"", "contact_lost", "contact_not_ready"})
                    for value in trace["residual_reasons"]
                ]
            ),
        }
        earned = 0.002 * np.maximum(
            0, 1 - sum(REWARD_WEIGHTS[key] * value for key, value in costs.items())
        )
        # Nonfinite executed rows cost the entire 1/s alive reward, not NaN reward.
        reward_rows = np.where(finite_rows, earned, 0.0)
    boundaries = np.r_[0, np.cumsum(counts)]
    expected_rewards = np.array([np.sum(reward_rows[a:b]) for a, b in pairwise(boundaries)])
    expected_rewards[-1] -= duration + 1 if term[-1] else 0
    _require(
        np.all(np.isfinite(arrays["rewards"]))
        and np.allclose(arrays["rewards"], expected_rewards, atol=1e-12, rtol=0),
        "reward mismatch",
    )
    _require(
        entry["transition_count"] == n and entry["physics_count"] == size, "episode count mismatch"
    )
    _require(
        entry["terminated"] == bool(term[-1]) and entry["truncated"] == bool(trunc[-1]),
        "termination metadata mismatch",
    )
    _require(
        np.isclose(entry["return"], np.sum(arrays["rewards"]), atol=1e-12, rtol=0),
        "return mismatch",
    )
    _require(not term[-1] or entry["return"] <= -1 + 1e-12, "failed episode rewarded")
    _require(
        entry["gates"]
        == {
            "environment_termination_free": not bool(term[-1]),
            "evaluation_observed": bool(
                size and np.any(trace["time"] >= case["config"]["evaluation_start"])
            ),
        },
        "quality gate metadata mismatch",
    )
    reasons = str(arrays["termination_reasons"][-1]).split("|") if term[-1] else []
    _require(
        np.all(arrays["termination_reasons"][:-1] == "")
        and bool(str(arrays["termination_reasons"][-1])) == bool(term[-1])
        and entry["termination_reasons"] == reasons,
        "termination reasons mismatch",
    )
    raw_nonfinite = [key for key in nonfinite if not key.startswith("endpoint_")]
    _require(not raw_nonfinite or "nonfinite" in reasons, "nonfinite failure reason omitted")
    return n, size, bool(term[-1])


def audit_dataset(path) -> dict:
    """Audit stored arrays independently; actor labels never consume evaluator truth."""
    path = Path(path).absolute()
    _require(
        not any(item.is_symlink() for item in (path, *path.parents)),
        "dataset path contains symlinks",
    )
    _require(
        not any(item.is_symlink() for item in path.iterdir()), "dataset artifact contains symlinks"
    )
    _require(
        (path / "COMPLETE").read_text().strip() == _sha256(path / "manifest.json"),
        "COMPLETE hash mismatch",
    )
    manifest = json.loads((path / "manifest.json").read_text())
    _require(
        manifest["schema"] == DATASET_SCHEMA
        and manifest["observation_schema"] == OBSERVATION_SCHEMA,
        "unsupported dataset schema",
    )
    _require(
        manifest["observation_names"] == list(OBSERVATION_NAMES)
        and manifest["observation_scales"] == list(OBSERVATION_SCALES)
        and manifest["observation_units"] == list(OBSERVATION_UNITS),
        "observation schema mismatch",
    )
    _require(
        manifest["nominal_kind"] == "adaptive" and manifest["action_stage"] == ACTION_STAGE,
        "wrong teacher/action contract",
    )
    _require(
        manifest["new_holdout"] is False
        and manifest["physics_period_s"] == 0.002
        and manifest["policy_substeps"] == 10
        and manifest["action_names"] == ["normal", "tangent1", "tangent2"]
        and manifest["action_units"] == "normalized",
        "sampling/action schema mismatch",
    )
    _require(manifest["reward_weights"] == REWARD_WEIGHTS, "reward weights mismatch")
    _require(
        manifest["transition_contract"] == TRANSITION_CONTRACT,
        "transition/discount/teacher input contract mismatch",
    )
    residual = SurfaceResidualConfig(**manifest["residual_config"])
    _require(asdict(residual) == asdict(SurfaceResidualConfig()), "teacher parameters changed")
    _require(
        all(
            isinstance(case["case_id"], str) and case["case_id"].strip()
            for case in manifest["cases"]
        ),
        "case_id values must be nonempty strings",
    )
    cases = {case["case_id"]: case for case in manifest["cases"]}
    entries = manifest["episodes"]
    assigned = split_cases(manifest["cases"], **manifest["group_split"])
    _require(
        all(
            original["group_id"] == labeled["group_id"] and original["split"] == labeled["split"]
            for original, labeled in zip(manifest["cases"], assigned)
        ),
        "declared group split mismatch",
    )
    _require(
        len(cases) == len(manifest["cases"])
        and len({e["case_id"] for e in entries}) == len(entries),
        "duplicate planned episode",
    )
    selected = {entry["case_id"] for entry in entries}
    scope = manifest["corpus_scope"]
    if scope == "example_subset":
        parent = manifest.get("parent_manifest_sha256", "")
        _require(
            len(parent) == 64
            and all(c in "0123456789abcdef" for c in parent)
            and selected < set(cases),
            "invalid explicit example subset",
        )
    else:
        _require(scope == "full" and selected == set(cases), "missing planned episode")
    _require(len({entry["file"] for entry in entries}) == len(entries), "duplicate episode file")
    _require(
        {p.name for p in path.iterdir()}
        == {"manifest.json", "COMPLETE", *(e["file"] for e in entries)},
        "untracked dataset artifact",
    )
    groups, totals, splits = {}, [0, 0, 0], dict.fromkeys(SPLIT_NAMES, 0)
    for entry in entries:
        case, filename = cases[entry["case_id"]], entry["file"]
        group = case_group_id(case)
        split = case["split"]
        _require(
            split in SPLIT_NAMES
            and group == case["group_id"] == entry["group_id"]
            and split == entry["split"],
            "group metadata mismatch",
        )
        _require(group not in groups or groups[group] == split, "group leakage across splits")
        groups[group], splits[split] = split, splits[split] + 1
        _require(
            Path(filename).name == filename and not (path / filename).is_symlink(),
            "unsafe episode path",
        )
        _require(_sha256(path / filename) == entry["sha256"], "episode hash mismatch")
        _require(
            entry["env_seed"] == case["config"]["seed"]
            and entry["actual_config"] == {**case["config"], "seed": entry["simulation_seed"]},
            "actual seed/config mismatch",
        )
        rng, _ = gymnasium.utils.seeding.np_random(entry["env_seed"])
        _require(
            entry["simulation_seed"] == int(rng.integers(0, 2**31 - 1)),
            "derived simulation seed mismatch",
        )
        _require(
            entry["actual_scenario"]
            == json.loads(json.dumps(asdict(SurfaceScenario(**case["scenario"]))))
            and entry["actual_task"] == asdict(LearningSurfaceTask(**case["task"]))
            and case.get("nominal_kind", "adaptive") == "adaptive",
            "actual task/scenario mismatch",
        )
        with np.load(path / filename, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files}
        _require(
            all(value.dtype.kind in "biufUS" for value in arrays.values()), "unsafe array dtype"
        )
        counts = _episode_audit(arrays, entry, case, residual)
        totals = [left + right for left, right in zip(totals, counts)]
    _require(len(groups) >= 3 and all(splits.values()), "three disjoint nonempty splits required")
    return {
        "valid": True,
        "episodes": len(entries),
        "transitions": totals[0],
        "physics_steps": totals[1],
        "failed_episodes": totals[2],
        "groups": len(groups),
        "split_episode_counts": splits,
        "new_holdout": False,
        "corpus_scope": scope,
        "planned_episodes": len(cases),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", type=Path)
    mode.add_argument("--output", type=Path)
    parser.add_argument("--cases", type=Path)
    args = parser.parse_args(argv)
    if args.audit:
        if args.cases:
            parser.error("--cases is only valid with --output")
        result = audit_dataset(args.audit)
    else:
        if args.cases is None:
            parser.error("--output requires --cases JSON")
        result = {
            "output": str(collect_demonstrations(args.output, json.loads(args.cases.read_text())))
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
