import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_env import (
    OBSERVATION_NAMES,
    OBSERVATION_SCALES,
    OBSERVATION_SCHEMA,
    OBSERVATION_UNITS,
    PHYSICS_DT,
    POLICY_SUBSTEPS,
    SurfaceLearningEnv,
)
from compliant_control_lab.surface_policy import SurfaceResidualConfig
from compliant_control_lab.surface_policy_artifact import (
    ActorEvaluator,
    freeze_evaluation_protocol,
    load_policy_artifact,
    policy_contract,
    save_policy_artifact,
    verify_evaluation_protocol,
)
from compliant_control_lab.surface_readiness_cases import (
    DOMAIN_BOUNDS,
    DOMAIN_SCHEMA,
    preparation_cases,
)
from compliant_control_lab.surface_simulation import SurfaceScenario, SurfaceSimulationConfig


def _fixture_payload(value=0.0):
    return {
        "kind": "linear_tanh",
        "weights": np.full((3, 49), value).tolist(),
        "bias": [0.1, 0.0, -0.1],
    }


def _contract_and_cases():
    cases = preparation_cases(6)
    return policy_contract(cases, purpose="il_friction_teacher", nominal_kind="adaptive"), cases


def _save_candidate(tmp_path):
    contract, cases = _contract_and_cases()
    path = tmp_path / "candidate.json"
    save_policy_artifact(path, _fixture_payload(), contract=contract)
    return path, contract, cases


def _set_path(mapping, path, value):
    changed = deepcopy(mapping)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return changed


def test_contract_fixes_observation_action_clock_domain_split_and_nominal():
    contract, cases = _contract_and_cases()
    assert contract["observation"] == {
        "schema": OBSERVATION_SCHEMA,
        "names": list(OBSERVATION_NAMES),
        "scales": list(OBSERVATION_SCALES),
        "units": list(OBSERVATION_UNITS),
        "dtype": "float64",
        "shape": [49],
        "clip": [-3, 3],
        "normalization": "physical_divided_by_fixed_scales_then_clipped_no_running_statistics",
    }
    assert contract["action"]["shape"] == [3]
    assert contract["action"]["bounds"] == [-1, 1]
    assert contract["action"]["units"] == "normalized"
    assert contract["action"]["frame"] == "controller_surface_normal_tangent1_tangent2"
    assert contract["physics_period_s"] == PHYSICS_DT
    assert contract["policy_period_s"] == PHYSICS_DT * POLICY_SUBSTEPS
    assert contract["task_domain"] == {
        "schema": DOMAIN_SCHEMA,
        "bounds": {name: list(bounds) for name, bounds in DOMAIN_BOUNDS.items()},
        "assumptions": [
            "fixed_smooth_contact_model",
            "known_task_plane",
            "yaw_calibration_error",
            "Coulomb_friction",
            "simulation_only",
        ],
    }
    binding = contract["split_binding"]
    assert binding["case_count"] == len(cases)
    assert binding["group_count"] == 6
    assert binding["test_identity"] == "public_development_test_not_blind_holdout"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("observation", "schema"), "old_axis_world_v0"),
        (("observation", "names"), ["world_axis"] * 49),
        (("observation", "scales"), [1.0] * 49),
        (("observation", "units"), ["unknown"] * 49),
        (("observation", "shape"), [33]),
        (("action", "frame"), "world_xyz"),
        (("action", "bounds"), [-2, 2]),
        (("physics_period_s",), 0.004),
        (("policy_period_s",), 0.04),
        (("nominal_kind",), "friction"),
        (("task_domain", "schema"), "other_domain"),
    ],
)
def test_save_rejects_cross_schema_nominal_config_and_old_world_axis_contract(
    tmp_path, path, value
):
    contract, _ = _contract_and_cases()
    with pytest.raises(ValueError):
        save_policy_artifact(
            tmp_path / "candidate.json",
            _fixture_payload(),
            contract=_set_path(contract, path, value),
        )


def test_rl_and_il_nominal_kinds_cannot_be_mixed():
    _, cases = _contract_and_cases()
    rl_cases = [{**case, "nominal_kind": "friction"} for case in cases]
    rl = policy_contract(rl_cases, purpose="rl_residual", nominal_kind="friction")
    assert rl["teacher_labels"] is None
    with pytest.raises(ValueError, match="purpose and nominal_kind mismatch"):
        policy_contract(cases, purpose="rl_residual", nominal_kind="adaptive")
    with pytest.raises(ValueError, match="purpose and nominal_kind mismatch"):
        policy_contract(rl_cases, purpose="il_friction_teacher", nominal_kind="friction")


def test_case_domain_and_group_identity_tampering_is_rejected():
    contract, cases = _contract_and_cases()
    outside = deepcopy(cases)
    outside[0]["scenario"]["wall_yaw_deg"] = 90.0
    with pytest.raises(ValueError):
        policy_contract(outside, purpose="il_friction_teacher", nominal_kind="adaptive")
    leaked = deepcopy(cases)
    peer = next(case for case in leaked[1:] if case["group_id"] == leaked[0]["group_id"])
    peer["split"] = next(
        name for name in ("train", "validation", "development_test") if name != leaked[0]["split"]
    )
    with pytest.raises(ValueError, match="leaks across splits"):
        policy_contract(leaked, purpose="il_friction_teacher", nominal_kind="adaptive")
    assert contract["split_binding"]["group_count"] == 6


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("scenario", "nominal_tool_mass_kg"), 0.11),
        (("scenario", "torque_noise_std_nm"), 0.003),
        (("scenario", "torque_bias_sensor_nm"), [0.0, 0.0, 0.01]),
        (("config", "force_filter_time_constant"), 0.01),
        (("config", "evaluation_start"), 1.0),
        (("task", "direction"), 1.0),
    ],
)
def test_contract_rejects_fixed_domain_fields_and_invalid_learning_task_types(path, value):
    _, cases = _contract_and_cases()
    with pytest.raises((TypeError, ValueError)):
        policy_contract(
            _set_path(cases, path, value), purpose="il_friction_teacher", nominal_kind="adaptive"
        )


def test_contract_allows_variable_case_name_duration_and_seed_for_smoke_runs():
    _, cases = _contract_and_cases()
    smoke = deepcopy(cases)
    smoke[0]["scenario"]["name"] = "short_smoke"
    smoke[0]["config"]["duration"] = 0.02
    smoke[0]["config"]["seed"] = 97
    contract = policy_contract(smoke, purpose="il_friction_teacher", nominal_kind="adaptive")
    assert contract["split_binding"]["case_count"] == len(smoke)


def test_finite_json_fixture_round_trip_is_stable_for_identical_observations(tmp_path):
    path, contract, _ = _save_candidate(tmp_path)
    loaded = load_policy_artifact(path, expected_contract=contract)
    actor = ActorEvaluator(loaded)
    first, repeated = actor.infer(np.zeros(49)), actor.infer(np.zeros(49))
    np.testing.assert_array_equal(first.action, repeated.action)
    np.testing.assert_allclose(first.action, [0.09966799462495582, 0, -0.09966799462495582])
    assert first.reason is None and first.latency_s >= 0


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "linear_tanh", "weights": [[0.0]], "bias": [0.0] * 3},
        {
            "kind": "linear_tanh",
            "weights": np.zeros((3, 49)).tolist(),
            "bias": [0, 0, np.nan],
        },
        {
            "kind": "linear_tanh",
            "weights": np.zeros((3, 49)).tolist(),
            "bias": [0, 0, 0],
            "code": "exec",
        },
        {"kind": "linear_tanh", "weights": [[False] * 49] * 3, "bias": [0, 0, 0]},
        {"kind": "custom", "value": np.inf},
        {"kind": "custom", "value": object()},
    ],
)
def test_save_rejects_nonfinite_non_json_or_malformed_fixture_payload(tmp_path, payload):
    contract, _ = _contract_and_cases()
    with pytest.raises(ValueError):
        save_policy_artifact(tmp_path / "bad.json", payload, contract=contract)


def test_load_rejects_contract_and_payload_tampering(tmp_path):
    path, contract, _ = _save_candidate(tmp_path)
    original = path.read_text()
    document = json.loads(path.read_text())
    document["body"]["payload"]["bias"][0] = 0.2
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="body hash mismatch"):
        load_policy_artifact(path, expected_contract=contract)

    document = json.loads(original)
    document["body"]["payload"]["weights"] = [[0]]
    canonical = json.dumps(document["body"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["body_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="linear_tanh"):
        load_policy_artifact(path, expected_contract=contract)

    document = json.loads(original)
    document["body"]["contract"]["action"]["frame"] = "world_xyz"
    canonical = json.dumps(document["body"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["body_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="interface schema"):
        load_policy_artifact(path, expected_contract=contract)

    changed = _set_path(contract, ("observation", "schema"), "old_axis_world_v0")
    with pytest.raises(ValueError):
        load_policy_artifact(path, expected_contract=changed)


def test_load_rejects_a_valid_but_different_residual_configuration(tmp_path):
    path, expected, cases = _save_candidate(tmp_path)
    changed = policy_contract(
        cases,
        purpose="il_friction_teacher",
        nominal_kind="adaptive",
        residual_config=SurfaceResidualConfig(action_bounds_n=(4.0, 6.0, 5.0)),
    )
    with pytest.raises(ValueError, match="explicitly expected deployment contract"):
        load_policy_artifact(path, expected_contract=changed)
    assert expected["residual_config"] != changed["residual_config"]


@pytest.mark.parametrize(
    ("callback", "reason"),
    [
        (lambda observation: (_ for _ in ()).throw(RuntimeError("boom")), "actor_exception"),
        (lambda observation: [0.0, np.nan, 0.0], "actor_invalid_action"),
        (lambda observation: [0.0, 1.01, 0.0], "actor_out_of_space"),
        (lambda observation: [0.0, 0.0], "actor_invalid_action"),
    ],
)
def test_actor_callback_failures_are_fail_closed(tmp_path, callback, reason):
    path, contract, _ = _save_candidate(tmp_path)
    decision = ActorEvaluator(
        load_policy_artifact(path, expected_contract=contract), callback
    ).infer(np.zeros(49))
    assert decision.action is None
    assert decision.reason == reason


def test_actor_receives_only_an_owned_observation_copy(tmp_path):
    path, contract, _ = _save_candidate(tmp_path)
    original = np.zeros(49)
    seen = {}

    def actor(observation):
        seen["shares"] = np.shares_memory(observation, original)
        observation[:] = 2.0
        return observation[:3] / 2

    decision = ActorEvaluator(load_policy_artifact(path, expected_contract=contract), actor).infer(
        original
    )
    assert not seen["shares"]
    np.testing.assert_array_equal(original, np.zeros(49))
    np.testing.assert_array_equal(decision.action, np.ones(3))


def test_timeout_is_reported_only_after_synchronous_callback_returns(tmp_path, monkeypatch):
    path, contract, _ = _save_candidate(tmp_path)
    times = iter((10.0, 10.03))
    monkeypatch.setattr(
        "compliant_control_lab.surface_policy_artifact.time.perf_counter", lambda: next(times)
    )
    called = []

    def actor(observation):
        called.append(True)
        return np.zeros(3)

    decision = ActorEvaluator(
        load_policy_artifact(path, expected_contract=contract), actor, latency_budget_s=0.02
    ).infer(np.zeros(49))
    assert called == [True]
    assert decision.action is None
    assert decision.reason == "actor_timeout_after_return"
    assert decision.latency_s == pytest.approx(0.03)


def test_callback_exception_steps_none_and_terminates_nominal_fallback(tmp_path):
    path, contract, cases = _save_candidate(tmp_path)
    case = cases[0]
    env = SurfaceLearningEnv(
        SurfaceFrame(case["controller_frame_rotation"]),
        scenario=SurfaceScenario(**case["scenario"]),
        config=SurfaceSimulationConfig(**case["config"]),
        task=LearningSurfaceTask(**case["task"]),
        nominal_kind="adaptive",
        residual_config=SurfaceResidualConfig(**contract["residual_config"]),
    )
    observation, _ = env.reset(seed=case["config"]["seed"])
    evaluator = ActorEvaluator(
        load_policy_artifact(path, expected_contract=contract),
        lambda observation: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    _, _, terminated, truncated, info = evaluator.step(env, observation)
    assert terminated and not truncated
    assert info["input_action_issue"] == "shape"
    assert info["invalid_input_repr"] == "None"
    assert "invalid_action" in info["termination_reasons"]
    assert info["action_stages"][0]["requested_residual_force_local"].tolist() == [0, 0, 0]
    assert info["actor_evaluation"]["reason"] == "actor_exception"
    env.close()


def test_protocol_freeze_binds_artifact_contract_cases_metrics_and_gates(tmp_path):
    candidate, contract, cases = _save_candidate(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    metrics = ["force_rmse_n", "episode_return"]
    thresholds = {
        "force_rmse_n": {"aggregation": "median", "operator": "<=", "value": 5.0},
        "episode_return": {"aggregation": "min", "operator": ">=", "value": -1.0},
    }
    pinned = freeze_evaluation_protocol(
        protocol_path,
        candidate,
        expected_contract=contract,
        cases=cases,
        metrics=metrics,
        thresholds=thresholds,
    )
    protocol = verify_evaluation_protocol(
        protocol_path,
        candidate,
        expected_protocol_sha256=pinned,
        expected_contract=contract,
        cases=cases,
    )
    assert protocol["metrics"] == metrics
    assert protocol["thresholds"] == thresholds
    assert protocol["evaluation_split"] == "development_test"
    assert protocol["new_blind_holdout"] is False
    assert protocol["split_binding"]["test_identity"] == (
        "public_development_test_not_blind_holdout"
    )
    with pytest.raises(ValueError, match="absent or empty|new file"):
        freeze_evaluation_protocol(
            protocol_path,
            candidate,
            expected_contract=contract,
            cases=cases,
            metrics=metrics,
            thresholds=thresholds,
        )


def test_protocol_accepts_readiness_metrics_and_rejects_unemitted_metric_names(tmp_path):
    candidate, contract, cases = _save_candidate(tmp_path)
    metrics = [
        "max_speed_m_s",
        "residual_projection_pct",
        "intervention_pct",
        "max_contact_loss_s",
    ]
    thresholds = {
        metric: {"aggregation": "max", "operator": "<=", "value": 1.0}
        for metric in metrics
    }
    freeze_evaluation_protocol(
        tmp_path / "readiness-protocol.json",
        candidate,
        expected_contract=contract,
        cases=cases,
        metrics=metrics,
        thresholds=thresholds,
    )
    for metric in ("projection_pct", "max_actor_latency_s", "actor_failure_count"):
        with pytest.raises(ValueError, match="supported evaluator metrics"):
            freeze_evaluation_protocol(
                tmp_path / f"{metric}.json",
                candidate,
                expected_contract=contract,
                cases=cases,
                metrics=[metric],
                thresholds={metric: {"aggregation": "max", "operator": "<=", "value": 1.0}},
            )


def test_protocol_external_pin_rejects_rehashed_protocol_and_candidate_or_case_changes(tmp_path):
    candidate, contract, cases = _save_candidate(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    kwargs = {
        "expected_contract": contract,
        "cases": cases,
        "metrics": ["force_rmse_n"],
        "thresholds": {"force_rmse_n": {"aggregation": "median", "operator": "<=", "value": 5.0}},
    }
    pinned = freeze_evaluation_protocol(protocol_path, candidate, **kwargs)
    original_protocol = protocol_path.read_text()
    document = json.loads(original_protocol)
    document["body"]["thresholds"]["force_rmse_n"]["value"] = 50.0
    canonical = json.dumps(document["body"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["body_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    protocol_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="externally frozen SHA"):
        verify_evaluation_protocol(
            protocol_path,
            candidate,
            expected_protocol_sha256=pinned,
            expected_contract=contract,
            cases=cases,
        )

    protocol_path.write_text(original_protocol)
    original_candidate = candidate.read_text()
    candidate.write_text(original_candidate + " \n")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_evaluation_protocol(
            protocol_path,
            candidate,
            expected_protocol_sha256=pinned,
            expected_contract=contract,
            cases=cases,
        )
    candidate.write_text(original_candidate)
    changed_cases = deepcopy(cases)
    changed_cases[0]["config"]["target_force"] += 0.1
    with pytest.raises(ValueError):
        verify_evaluation_protocol(
            protocol_path,
            candidate,
            expected_protocol_sha256=pinned,
            expected_contract=contract,
            cases=changed_cases,
        )
