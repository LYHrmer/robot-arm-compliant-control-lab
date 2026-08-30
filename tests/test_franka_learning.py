import numpy as np
import pytest

from compliant_control_lab import franka_learning
from compliant_control_lab.franka_learning import (
    ArsTrainingConfig,
    evaluate_frozen_holdout,
    evaluate_policy_cost,
    physical_rollout_cost,
    sample_training_scenarios,
    train_residual_policy,
)
from compliant_control_lab.residual_rl import (
    TORQUE_AWARE_OBSERVATION_NAMES,
    BoundedResidualController,
    LinearResidualPolicy,
    TorqueProjectedResidualController,
)


def test_training_scenarios_are_separate_and_reproducible() -> None:
    first = sample_training_scenarios(count=3, seed=101)
    second = sample_training_scenarios(count=3, seed=101)
    assert first == second
    assert all(scenario.name.startswith("train_") for scenario in first)
    with pytest.raises(ValueError, match="reserved"):
        sample_training_scenarios(count=1, seed=29)


def test_physical_cost_penalizes_gate_regression() -> None:
    good = {
        "force_rmse_n": 1.0,
        "peak_force_n": 25.0,
        "tangent_rmse_mm": 8.0,
        "saturation_pct": 0.0,
        "contact_ratio_pct": 100.0,
    }
    unsafe = {**good, "peak_force_n": 60.0, "saturation_pct": 5.0}
    assert physical_rollout_cost(unsafe) > physical_rollout_cost(good)


def test_ars_smoke_run_is_deterministic() -> None:
    config = ArsTrainingConfig(
        iterations=1,
        directions=1,
        top_directions=1,
        training_cases=1,
        duration=0.04,
        policy_seed=5,
    )
    first, first_records, _ = train_residual_policy(config)
    second, second_records, _ = train_residual_policy(config)

    np.testing.assert_allclose(first.parameter_vector(), second.parameter_vector())
    assert first_records == second_records
    assert len(first_records) == 2


def test_same_holdout_case_and_seed_are_used_for_all_methods() -> None:
    rows = evaluate_frozen_holdout(
        LinearResidualPolicy.zero(),
        count=1,
        duration=0.04,
        seed=29,
    )
    assert {row["method"] for row in rows} == {
        "fixed_hybrid",
        "adaptive_hybrid",
        "bounded_residual_rl",
    }
    assert {row["case"] for row in rows} == {"holdout_00"}
    assert {row["simulation_seed"] for row in rows} == {29}


def test_trainer_supports_torque_aware_schema_and_validation_selection() -> None:
    config = ArsTrainingConfig(
        iterations=1,
        directions=1,
        top_directions=1,
        training_cases=1,
        duration=0.04,
        policy_seed=13,
    )
    validation = sample_training_scenarios(count=1, seed=211)
    policy, records, _ = train_residual_policy(
        config,
        controller_factory=lambda candidate: TorqueProjectedResidualController(
            policy=candidate,
            inference_deadline_us=1.0e9,
        ),
        observation_names=TORQUE_AWARE_OBSERVATION_NAMES,
        validation_scenarios=validation,
        validation_simulation_seed=3_001,
        resample_simulation_noise=True,
    )

    assert policy.observation_names == TORQUE_AWARE_OBSERVATION_NAMES
    assert all(record.validation_cost is not None for record in records)


def test_validation_noise_is_namespaced_by_run_and_resampled_with_common_random_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool]] = []

    def fake_policy_cost(
        policy,
        scenarios,
        duration,
        simulation_seed,
        controller_factory=None,
        enforce_inference_deadline=True,
    ):
        del scenarios, duration, controller_factory
        calls.append((simulation_seed, enforce_inference_deadline))
        return float(np.mean(policy.parameter_vector() ** 2))

    monkeypatch.setattr(franka_learning, "evaluate_policy_cost", fake_policy_cost)
    validation = sample_training_scenarios(count=1, seed=211)
    config = ArsTrainingConfig(
        iterations=2,
        directions=1,
        top_directions=1,
        training_cases=1,
        simulation_seed=1_001,
        policy_seed=13,
        enforce_inference_deadline_during_training=False,
    )

    _, records, _ = train_residual_policy(
        config,
        validation_scenarios=validation,
        validation_simulation_seed=70_001,
        resample_simulation_noise=True,
    )

    assert [seed for seed, _ in calls] == [
        1_001,
        71_002,
        101_001,
        101_001,
        101_001,
        171_002,
        201_001,
        201_001,
        201_001,
        271_002,
    ]
    assert all(not deadline_enabled for _, deadline_enabled in calls)
    assert [record.training_simulation_seed for record in records] == [
        1_001,
        101_001,
        201_001,
    ]
    assert [record.validation_simulation_seed for record in records] == [
        71_002,
        171_002,
        271_002,
    ]

    calls.clear()
    second_run = ArsTrainingConfig(
        iterations=1,
        directions=1,
        top_directions=1,
        training_cases=1,
        simulation_seed=2_001,
        policy_seed=13,
        enforce_inference_deadline_during_training=False,
    )
    _, second_records, _ = train_residual_policy(
        second_run,
        validation_scenarios=validation,
        validation_simulation_seed=70_001,
        resample_simulation_noise=True,
    )
    assert [record.validation_simulation_seed for record in second_records] == [
        72_002,
        172_002,
    ]


def test_policy_cost_disables_deadline_only_when_requested() -> None:
    scenarios = sample_training_scenarios(count=1, seed=101)
    policy = LinearResidualPolicy.zero()
    training_controllers: list[BoundedResidualController] = []

    def training_factory(candidate: LinearResidualPolicy) -> BoundedResidualController:
        controller = BoundedResidualController(policy=candidate)
        training_controllers.append(controller)
        return controller

    evaluate_policy_cost(
        policy,
        scenarios,
        duration=0.04,
        simulation_seed=1_001,
        controller_factory=training_factory,
        enforce_inference_deadline=False,
    )
    assert all(np.isinf(controller.inference_deadline_us) for controller in training_controllers)

    evaluation_controllers: list[BoundedResidualController] = []

    def evaluation_factory(candidate: LinearResidualPolicy) -> BoundedResidualController:
        controller = BoundedResidualController(policy=candidate)
        evaluation_controllers.append(controller)
        return controller

    evaluate_policy_cost(
        policy,
        scenarios,
        duration=0.04,
        simulation_seed=1_001,
        controller_factory=evaluation_factory,
    )
    assert all(
        controller.inference_deadline_us == 5_000.0
        for controller in evaluation_controllers
    )
