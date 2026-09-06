"""Finite-JSON policy contracts, isolated actor calls and externally pinned evaluation plans.

Hashes detect changes, not authorship or a genuinely unseen test. Preserve the
returned protocol SHA outside the protocol before evaluation; never recompute
that trusted value from a potentially changed file at evaluation time.
"""

import hashlib
import json
import os
import tempfile
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path

import numpy as np

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_env import (
    OBSERVATION_NAMES,
    OBSERVATION_SCALES,
    OBSERVATION_SCHEMA,
    OBSERVATION_UNITS,
    PHYSICS_DT,
    POLICY_SUBSTEPS,
    SurfaceLearningEnv,
)
from compliant_control_lab.surface_experiment import _output_path, _sha256
from compliant_control_lab.surface_policy import SurfaceResidualConfig
from compliant_control_lab.surface_readiness_benchmark import METRICS
from compliant_control_lab.surface_readiness_cases import DOMAIN_BOUNDS, DOMAIN_SCHEMA
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    yaw_frame,
)
from compliant_control_lab.surface_splits import (
    GROUP_SCHEMA,
    GROUP_SCHEMA_VERSION,
    SPLIT_NAMES,
    _json_value,
    case_group_id,
)

PURPOSES = {"rl_residual": "friction", "il_friction_teacher": "adaptive"}
_EXTRA_METRICS = {
    "episode_return",
}
_DOMAIN_SCENARIO_DEFAULTS = SurfaceScenario()
_DOMAIN_CONFIG_DEFAULTS = SurfaceSimulationConfig(contact_model="smooth")


def _canonical(value):
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _domain_check(case):
    scenario, config, task = (
        SurfaceScenario(**case["scenario"]),
        SurfaceSimulationConfig(**case["config"]),
        LearningSurfaceTask(**case["task"]),
    )
    frame = np.asarray(case["controller_frame_rotation"])
    yaw = float(np.rad2deg(np.arctan2(frame[1, 0], frame[0, 0])))
    values = {
        "wall_yaw_deg": scenario.wall_yaw_deg,
        "sliding_friction": scenario.wall_sliding_friction,
        "normal_calibration_error_deg": yaw - scenario.wall_yaw_deg,
        "wall_time_constant_s": scenario.wall_time_constant,
        "tool_mass_kg": scenario.tool_mass_kg,
        "bias_compensation_scale": scenario.bias_compensation_scale,
        "force_bias_component_n": scenario.force_bias_sensor_n,
        "position_noise_std_m": scenario.position_noise_std_m,
        "force_noise_std_n": scenario.force_noise_std_n,
        "delay_steps": scenario.delay_steps,
        "target_force_n": config.target_force,
        **{
            name: getattr(task, name)
            for name in ("frequency_hz", "tangent_amplitude_m", "vertical_amplitude_m", "phase_rad")
        },
    }
    for name, value in values.items():
        low, high = DOMAIN_BOUNDS[name]
        if np.any(np.asarray(value) < low - 1e-12) or np.any(np.asarray(value) > high + 1e-12):
            raise ValueError(f"case outside declared task domain: {name}")
    if (
        scenario.tool_sliding_friction != scenario.wall_sliding_friction
        or scenario.nominal_tool_mass_kg != _DOMAIN_SCENARIO_DEFAULTS.nominal_tool_mass_kg
        or scenario.torque_noise_std_nm != _DOMAIN_SCENARIO_DEFAULTS.torque_noise_std_nm
        or scenario.torque_bias_sensor_nm != _DOMAIN_SCENARIO_DEFAULTS.torque_bias_sensor_nm
        or config.contact_model != "smooth"
        or config.timestep != PHYSICS_DT
        or config.force_filter_time_constant != _DOMAIN_CONFIG_DEFAULTS.force_filter_time_constant
        or config.evaluation_start != _DOMAIN_CONFIG_DEFAULTS.evaluation_start
        or task.yaw_deg != scenario.wall_yaw_deg
        or task.nominal_plane_x_m != 0.4
        or not np.allclose(frame, yaw_frame(yaw).rotation, rtol=0, atol=1e-10)
    ):
        raise ValueError("case violates fixed smooth/known-plane/yaw-frame domain assumptions")


def _split_binding(cases, nominal_kind):
    cases = json.loads(_canonical(list(cases)))
    assignments, identifiers = {}, set()
    for case in cases:
        group = case_group_id(case)
        if case.get("group_id") != group or case.get("split") not in SPLIT_NAMES:
            raise ValueError("incorrect group identity or unsupported split")
        if group in assignments and assignments[group] != case["split"]:
            raise ValueError("physical/task group leaks across splits")
        identifier = case.get("case_id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("case_id must be a unique nonempty string")
        if case.get("nominal_kind") != nominal_kind:
            raise ValueError("case nominal_kind differs from artifact purpose")
        _domain_check(case)
        identifiers.add(identifier)
        assignments[group] = case["split"]
    if set(assignments.values()) != set(SPLIT_NAMES):
        raise ValueError("train, validation and development_test must all be represented")
    ordered = sorted(cases, key=lambda case: case["case_id"])
    manifest = {
        "group_schema": GROUP_SCHEMA,
        "group_schema_version": GROUP_SCHEMA_VERSION,
        "cases": ordered,
        "group_assignments": assignments,
    }
    return {
        "cases_sha256": _digest(ordered),
        "group_assignments_sha256": _digest(assignments),
        "split_manifest_sha256": _digest(manifest),
        "case_count": len(cases),
        "group_count": len(assignments),
        "group_schema": GROUP_SCHEMA,
        "group_schema_version": GROUP_SCHEMA_VERSION,
        "test_identity": "public_development_test_not_blind_holdout",
    }


def _interface(purpose, nominal_kind, residual_config, binding):
    if purpose not in PURPOSES or nominal_kind != PURPOSES[purpose]:
        raise ValueError(
            "purpose and nominal_kind mismatch: RL uses friction; IL teacher uses adaptive"
        )
    return json.loads(
        _canonical(
            {
                "schema": "surface_policy_contract_v1",
                "purpose": purpose,
                "nominal_kind": nominal_kind,
                "observation": {
                    "schema": OBSERVATION_SCHEMA,
                    "names": OBSERVATION_NAMES,
                    "scales": OBSERVATION_SCALES,
                    "units": OBSERVATION_UNITS,
                    "dtype": "float64",
                    "shape": [49],
                    "clip": [-3, 3],
                    "normalization": "physical_divided_by_fixed_scales_then_clipped_no_running_statistics",
                },
                "action": {
                    "shape": [3],
                    "dtype": "float64",
                    "bounds": [-1, 1],
                    "frame": "controller_surface_normal_tangent1_tangent2",
                    "units": "normalized",
                    "stage": "before_contact_gating_filter_slew_normal_limit_and_residual_torque_projection",
                },
                "physics_period_s": PHYSICS_DT,
                "policy_period_s": PHYSICS_DT * POLICY_SUBSTEPS,
                "residual_config": asdict(residual_config),
                "teacher_labels": "normalized_friction_teacher_request_not_projected_wrench_difference"
                if purpose == "il_friction_teacher"
                else None,
                "task_domain": {
                    "schema": DOMAIN_SCHEMA,
                    "bounds": DOMAIN_BOUNDS,
                    "assumptions": [
                        "fixed_smooth_contact_model",
                        "known_task_plane",
                        "yaw_calibration_error",
                        "Coulomb_friction",
                        "simulation_only",
                    ],
                },
                "split_binding": binding,
            }
        )
    )


def policy_contract(cases, *, purpose, nominal_kind, residual_config=None):
    """Build a fixed-interface contract from explicit, independently validated grouped cases."""
    residual_config = SurfaceResidualConfig() if residual_config is None else residual_config
    return _interface(purpose, nominal_kind, residual_config, _split_binding(cases, nominal_kind))


def _validate_contract(contract):
    try:
        expected = _interface(
            contract["purpose"],
            contract["nominal_kind"],
            SurfaceResidualConfig(**contract["residual_config"]),
            contract["split_binding"],
        )
        binding = contract["split_binding"]
        if any(
            len(binding[key]) != 64 or any(c not in "0123456789abcdef" for c in binding[key])
            for key in ("cases_sha256", "group_assignments_sha256", "split_manifest_sha256")
        ):
            raise ValueError("invalid split digest")
        if _canonical(contract) != _canonical(expected):
            raise ValueError("policy interface schema/normalization/action contract mismatch")
    except (KeyError, TypeError) as error:
        raise ValueError("missing or invalid surface policy contract") from error


def _write_new(path, body):
    path = _output_path(path)
    if path.exists():
        raise ValueError("artifact/protocol destination must be a new file")
    encoded = (_canonical({"body": body, "body_sha256": _digest(body)}) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-policy-", dir=path.parent) as directory:
        temporary = Path(directory) / "document.json"
        temporary.write_bytes(encoded)
        _output_path(path)
        os.link(temporary, path)  # Atomic exclusive publication, never replace existing data.
    return _sha256(path)


def _read(path):
    document = json.loads(Path(path).read_text())
    if (
        set(document) != {"body", "body_sha256"}
        or _digest(document["body"]) != document["body_sha256"]
    ):
        raise ValueError("artifact/protocol body hash mismatch")
    return document["body"]


@dataclass(frozen=True)
class PolicyArtifact:
    artifact_sha256: str
    contract: dict
    payload: dict


def _validate_payload(payload):
    if not isinstance(payload, dict):
        raise TypeError("payload must be a finite JSON mapping")
    try:
        _json_value(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("payload must be a finite JSON mapping") from error
    if payload.get("kind") != "linear_tanh":
        return
    if set(payload) != {"kind", "weights", "bias"}:
        raise ValueError("linear_tanh payload must contain only kind, weights and bias")
    try:
        raw_weights = np.asarray(payload["weights"], dtype=object)
        raw_bias = np.asarray(payload["bias"], dtype=object)
        weights = raw_weights.astype(np.float64)
        bias = raw_bias.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid linear_tanh fixture dimensions or values") from error
    if (
        weights.shape != (3, 49)
        or bias.shape != (3,)
        or any(isinstance(value, (bool, np.bool_)) for value in raw_weights.flat)
        or any(isinstance(value, (bool, np.bool_)) for value in raw_bias.flat)
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(bias))
    ):
        raise ValueError("invalid linear_tanh fixture dimensions or values")


def save_policy_artifact(path, payload, *, contract):
    """Store finite JSON only: no pickle, executable loader, fitted normalizer or torch dependency."""
    _validate_contract(contract)
    _validate_payload(payload)
    return _write_new(
        path, {"schema": "surface_policy_artifact_v1", "contract": contract, "payload": payload}
    )


def load_policy_artifact(path, *, expected_contract):
    _validate_contract(expected_contract)
    body = _read(path)
    if (
        set(body) != {"schema", "contract", "payload"}
        or body["schema"] != "surface_policy_artifact_v1"
    ):
        raise ValueError("unsupported policy artifact schema")
    _validate_contract(body["contract"])
    if _canonical(body["contract"]) != _canonical(expected_contract):
        raise ValueError("policy contract differs from the explicitly expected deployment contract")
    _validate_payload(body["payload"])
    return PolicyArtifact(
        _sha256(Path(path)), deepcopy(body["contract"]), deepcopy(body["payload"])
    )


def linear_fixture_actor(artifact):
    """A deterministic, actually executable fixture; not a trained-policy claim."""
    payload = artifact.payload
    if set(payload) != {"kind", "weights", "bias"} or payload["kind"] != "linear_tanh":
        raise ValueError("supply an explicit actor callback for non-fixture JSON payloads")
    weights, bias = (
        np.array(payload["weights"], dtype=np.float64),
        np.array(payload["bias"], dtype=np.float64),
    )
    if (
        weights.shape != (3, 49)
        or bias.shape != (3,)
        or not all(np.all(np.isfinite(v)) for v in (weights, bias))
    ):
        raise ValueError("invalid linear_tanh fixture dimensions or values")
    return lambda observation: np.tanh(weights @ observation + bias)


@dataclass(frozen=True)
class ActorDecision:
    action: np.ndarray | None
    reason: str | None
    latency_s: float
    exception_type: str | None = None


class ActorEvaluator:
    """Only an owned observation is passed to the callback, never env/info/truth.

    This is an API boundary, not a sandbox against callback closures. Timeouts
    are detected AFTER the synchronous callback returns; no realtime preemption
    or watchdog is provided. Failure returns None, deliberately invalid for Env,
    which executes its zero-residual nominal fallback and terminates.
    """

    def __init__(self, artifact, actor=None, *, latency_budget_s=0.02):
        _validate_contract(artifact.contract)
        if not np.isfinite(latency_budget_s) or latency_budget_s <= 0:
            raise ValueError("latency budget must be finite and positive")
        self._contract = deepcopy(artifact.contract)
        self._actor = linear_fixture_actor(artifact) if actor is None else actor
        if not callable(self._actor):
            raise TypeError("actor must be callable")
        self.latency_budget_s = float(latency_budget_s)

    def infer(self, observation):
        try:
            owned = np.array(observation, dtype=np.float64, copy=True)
        except (TypeError, ValueError):
            return ActorDecision(None, "invalid_observation", 0.0)
        if owned.shape != (49,) or not np.all(np.isfinite(owned)) or np.any(np.abs(owned) > 3):
            return ActorDecision(None, "invalid_observation", 0.0)
        started = time.perf_counter()
        try:
            output = self._actor(owned)
        except Exception as error:  # noqa: BLE001 - arbitrary policy callbacks must fail closed
            latency = time.perf_counter() - started
            return ActorDecision(None, "actor_exception", latency, type(error).__name__)
        latency = time.perf_counter() - started
        if latency > self.latency_budget_s:
            return ActorDecision(None, "actor_timeout_after_return", latency)
        try:
            action = np.array(output, dtype=np.float64, copy=True)
        except (TypeError, ValueError):
            return ActorDecision(None, "actor_invalid_action", latency)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            return ActorDecision(None, "actor_invalid_action", latency)
        if np.any(np.abs(action) > 1):
            return ActorDecision(None, "actor_out_of_space", latency)
        return ActorDecision(action, None, latency)

    def step(self, env, observation):
        if (
            not isinstance(env, SurfaceLearningEnv)
            or env.nominal_kind != self._contract["nominal_kind"]
        ):
            raise ValueError("actor environment nominal/interface mismatch")
        if _canonical(asdict(env.residual_config)) != _canonical(self._contract["residual_config"]):
            raise ValueError("actor environment residual configuration mismatch")
        _domain_check(
            {
                "scenario": asdict(env.scenario),
                "config": asdict(env.config),
                "task": asdict(env.task),
                "controller_frame_rotation": env.frame.rotation.tolist(),
            }
        )
        decision = self.infer(observation)
        next_observation, reward, terminated, truncated, info = env.step(decision.action)
        info = {
            **info,
            "actor_evaluation": {
                "reason": decision.reason,
                "latency_s": decision.latency_s,
                "exception_type": decision.exception_type,
                "latency_budget_s": self.latency_budget_s,
                "timeout_semantics": "checked_after_synchronous_return_not_preemptive",
            },
        }
        return next_observation, reward, terminated, truncated, info


def _protocol(candidate_path, contract, cases, evaluation_split, metrics, thresholds):
    artifact = load_policy_artifact(candidate_path, expected_contract=contract)
    binding = _split_binding(cases, contract["nominal_kind"])
    if binding != contract["split_binding"]:
        raise ValueError("evaluation cases/group split differs from the frozen policy binding")
    if evaluation_split not in {"validation", "development_test"}:
        raise ValueError(
            "evaluation must use validation or development_test, never claim blind holdout"
        )
    metrics = list(metrics)
    if (
        not metrics
        or len(set(metrics)) != len(metrics)
        or not set(metrics) <= set(METRICS) | _EXTRA_METRICS
    ):
        raise ValueError("metrics must be unique supported evaluator metrics")
    if not isinstance(thresholds, dict) or not thresholds or not thresholds.keys() <= set(metrics):
        raise ValueError("thresholds must refer to declared metrics")
    for rule in thresholds.values():
        if (
            not isinstance(rule, dict)
            or set(rule) != {"aggregation", "operator", "value"}
            or rule["aggregation"] not in {"max", "min", "median", "p95"}
            or rule["operator"] not in {"<=", ">=", "=="}
            or isinstance(rule["value"], bool)
            or not isinstance(rule["value"], Real)
            or not np.isfinite(rule["value"])
        ):
            raise ValueError(
                "thresholds need explicit aggregation, operator and finite numeric value"
            )
    selected = sorted(
        (case for case in cases if case["split"] == evaluation_split),
        key=lambda case: case["case_id"],
    )
    return {
        "schema": "surface_policy_evaluation_protocol_v1",
        "candidate_artifact_sha256": artifact.artifact_sha256,
        "policy_contract_sha256": _digest(contract),
        "split_binding": binding,
        "evaluation_split": evaluation_split,
        "new_blind_holdout": False,
        "evaluation_cases_sha256": _digest(selected),
        "evaluation_case_ids": [case["case_id"] for case in selected],
        "metrics": metrics,
        "thresholds": thresholds,
        "reset_seed_semantics": "case config.seed is Gym reset seed; generated simulator seed must be recorded",
    }


def freeze_evaluation_protocol(
    path,
    candidate_path,
    *,
    expected_contract,
    cases,
    metrics,
    thresholds,
    evaluation_split="development_test",
):
    """Return a SHA that the caller must preserve externally BEFORE evaluation."""
    return _write_new(
        path,
        _protocol(
            candidate_path, expected_contract, list(cases), evaluation_split, metrics, thresholds
        ),
    )


def verify_evaluation_protocol(
    path, candidate_path, *, expected_protocol_sha256, expected_contract, cases
):
    """Verify against an external freeze digest, candidate bytes and independently supplied cases."""
    if _sha256(Path(path)) != expected_protocol_sha256:
        raise ValueError("evaluation protocol differs from the externally frozen SHA")
    protocol = _read(path)
    expected = _protocol(
        candidate_path,
        expected_contract,
        list(cases),
        protocol["evaluation_split"],
        protocol["metrics"],
        protocol["thresholds"],
    )
    if _canonical(protocol) != _canonical(expected):
        raise ValueError("candidate or evaluation protocol identity mismatch")
    return deepcopy(protocol)
