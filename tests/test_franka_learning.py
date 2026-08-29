import numpy as np
import pytest

from compliant_control_lab.franka_learning import (
    ArsTrainingConfig,
    evaluate_frozen_holdout,
    physical_rollout_cost,
    sample_training_scenarios,
    train_residual_policy,
)
from compliant_control_lab.residual_rl import LinearResidualPolicy


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
