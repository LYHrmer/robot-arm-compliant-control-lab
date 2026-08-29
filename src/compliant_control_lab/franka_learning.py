"""Train and compare a bounded Residual RL policy for Franka contact control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compliant_control_lab.franka_adaptive import FrankaAdaptiveHybridController
from compliant_control_lab.franka_control import (
    FrankaController,
    FrankaHybridController,
)
from compliant_control_lab.franka_simulation import (
    FrankaScenario,
    FrankaSimulationConfig,
    run_franka_trial,
)
from compliant_control_lab.franka_stress import (
    ResidualRlGate,
    failed_gate_checks,
    sample_stress_scenarios,
    scenario_values,
)
from compliant_control_lab.residual_rl import (
    BoundedResidualController,
    LinearResidualPolicy,
)


@dataclass(frozen=True)
class ArsTrainingConfig:
    """Small deterministic ARS configuration for an inspectable residual policy."""

    iterations: int = 8
    directions: int = 6
    top_directions: int = 3
    noise_std: float = 0.06
    step_size: float = 0.04
    training_cases: int = 6
    duration: float = 3.2
    scenario_seed: int = 101
    simulation_seed: int = 1_001
    policy_seed: int = 17

    def validate(self) -> None:
        if self.iterations <= 0 or self.directions <= 0:
            raise ValueError("iterations and directions must be positive")
        if not 0 < self.top_directions <= self.directions:
            raise ValueError("top_directions must be in [1, directions]")
        if self.noise_std <= 0.0 or self.step_size <= 0.0:
            raise ValueError("noise_std and step_size must be positive")
        if self.training_cases <= 0 or self.duration <= 0.0:
            raise ValueError("training_cases and duration must be positive")


@dataclass(frozen=True)
class TrainingRecord:
    iteration: int
    mean_cost: float
    best_cost: float


def residual_safety_manifest() -> dict[str, float | list[float]]:
    controller = BoundedResidualController()
    return {
        "action_bounds_n": controller.action_bounds.tolist(),
        "action_rate_limits_n_s": controller.action_rate_limits.tolist(),
        "policy_period_s": controller.policy_period,
        "filter_time_constant_s": controller.filter_time_constant,
        "residual_enable_delay_s": controller.residual_enable_delay,
        "force_guard_margin_n": controller.force_guard_margin,
        "force_guard_rate_n_s": controller.force_guard_rate,
        "min_total_normal_wrench_n": controller.min_total_normal_wrench,
        "max_total_normal_wrench_n": controller.max_total_normal_wrench,
        "inference_deadline_us": controller.inference_deadline_us,
    }


def sample_training_scenarios(
    count: int = 6,
    seed: int = 101,
) -> tuple[FrankaScenario, ...]:
    """Use the stress distribution with a namespace separate from the frozen holdout."""
    if seed == 29:
        raise ValueError("seed 29 is reserved for the frozen holdout")
    return tuple(
        replace(scenario, name=f"train_{index:02d}")
        for index, scenario in enumerate(sample_stress_scenarios(count=count, seed=seed))
    )


def physical_rollout_cost(
    metrics: dict[str, float | str],
    residual_rms_n: float = 0.0,
) -> float:
    """Return a smooth dimensionless objective while preserving the physical final gate."""
    force_ratio = float(metrics["force_rmse_n"]) / 2.0
    peak_ratio = float(metrics["peak_force_n"]) / 35.0
    tangent_ratio = float(metrics["tangent_rmse_mm"]) / 15.0
    saturation_ratio = float(metrics["saturation_pct"]) / 1.0
    contact_shortfall = max(0.0, 95.0 - float(metrics["contact_ratio_pct"])) / 5.0
    residual_ratio = residual_rms_n / 5.0
    return float(
        force_ratio**2
        + 0.75 * peak_ratio**2
        + tangent_ratio**2
        + 2.0 * saturation_ratio**2
        + 4.0 * contact_shortfall**2
        + 0.05 * residual_ratio**2
    )


def evaluate_policy_cost(
    policy: LinearResidualPolicy,
    scenarios: Sequence[FrankaScenario],
    duration: float,
    simulation_seed: int,
) -> float:
    costs = []
    for index, scenario in enumerate(scenarios):
        controller = BoundedResidualController(
            policy=policy,
            nominal=FrankaAdaptiveHybridController(),
        )
        result = run_franka_trial(
            controller,
            scenario=scenario,
            config=FrankaSimulationConfig(
                duration=duration,
                seed=simulation_seed + index,
            ),
        )
        costs.append(physical_rollout_cost(result.metrics(), controller.residual_rms_n))
    return float(np.mean(costs))


def train_residual_policy(
    config: ArsTrainingConfig | None = None,
) -> tuple[LinearResidualPolicy, list[TrainingRecord], tuple[FrankaScenario, ...]]:
    """Train a bounded linear residual policy with Augmented Random Search."""
    config = config or ArsTrainingConfig()
    config.validate()
    scenarios = sample_training_scenarios(config.training_cases, config.scenario_seed)
    rng = np.random.default_rng(config.policy_seed)
    parameters = LinearResidualPolicy.zero().parameter_vector()
    current_cost = evaluate_policy_cost(
        LinearResidualPolicy.from_parameter_vector(parameters),
        scenarios,
        config.duration,
        config.simulation_seed,
    )
    best_parameters = parameters.copy()
    best_cost = current_cost
    records = [TrainingRecord(iteration=0, mean_cost=current_cost, best_cost=best_cost)]

    for iteration in range(1, config.iterations + 1):
        directions = rng.normal(size=(config.directions, parameters.size))
        rollouts = []
        for direction in directions:
            plus_policy = LinearResidualPolicy.from_parameter_vector(
                parameters + config.noise_std * direction
            )
            minus_policy = LinearResidualPolicy.from_parameter_vector(
                parameters - config.noise_std * direction
            )
            plus_score = -evaluate_policy_cost(
                plus_policy,
                scenarios,
                config.duration,
                config.simulation_seed,
            )
            minus_score = -evaluate_policy_cost(
                minus_policy,
                scenarios,
                config.duration,
                config.simulation_seed,
            )
            rollouts.append((plus_score, minus_score, direction))

        rollouts.sort(key=lambda item: max(item[0], item[1]), reverse=True)
        selected = rollouts[: config.top_directions]
        reward_std = float(np.std([score for pair in selected for score in pair[:2]]))
        if reward_std > 1.0e-12:
            update = sum(
                ((plus_score - minus_score) * direction for plus_score, minus_score, direction in selected),
                start=np.zeros_like(parameters),
            )
            parameters += (
                config.step_size / (config.top_directions * reward_std)
            ) * update

        current_policy = LinearResidualPolicy.from_parameter_vector(parameters)
        current_cost = evaluate_policy_cost(
            current_policy,
            scenarios,
            config.duration,
            config.simulation_seed,
        )
        if current_cost < best_cost:
            best_cost = current_cost
            best_parameters = parameters.copy()
        records.append(
            TrainingRecord(
                iteration=iteration,
                mean_cost=current_cost,
                best_cost=best_cost,
            )
        )
        print(
            f"ARS iteration {iteration:02d}/{config.iterations}: "
            f"cost={current_cost:.4f} best={best_cost:.4f}",
            flush=True,
        )

    return (
        LinearResidualPolicy.from_parameter_vector(best_parameters),
        records,
        scenarios,
    )


def _write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _controller_factories(
    policy: LinearResidualPolicy,
) -> dict[str, Callable[[], FrankaController]]:
    return {
        "fixed_hybrid": FrankaHybridController,
        "adaptive_hybrid": FrankaAdaptiveHybridController,
        "bounded_residual_rl": lambda: BoundedResidualController(
            policy=policy,
            nominal=FrankaAdaptiveHybridController(),
        ),
    }


def evaluate_frozen_holdout(
    policy: LinearResidualPolicy,
    count: int = 24,
    duration: float = 4.5,
    seed: int = 29,
    gate: ResidualRlGate | None = None,
) -> list[dict[str, float | int | str]]:
    """Evaluate all methods on identical scenario and simulation seeds."""
    gate = gate or ResidualRlGate()
    rows: list[dict[str, float | int | str]] = []
    factories = _controller_factories(policy)
    for index, scenario in enumerate(sample_stress_scenarios(count=count, seed=seed)):
        for method, factory in factories.items():
            result = run_franka_trial(
                factory(),
                scenario=scenario,
                config=FrankaSimulationConfig(duration=duration, seed=seed + index),
            )
            metrics = result.metrics()
            failures = failed_gate_checks(metrics, gate)
            rows.append(
                {
                    "method": method,
                    "case": scenario.name,
                    "simulation_seed": seed + index,
                    **scenario_values(scenario),
                    **metrics,
                    "gate_pass": "yes" if not failures else "no",
                    "failed_checks": ";".join(failures),
                }
            )
        print(
            f"evaluated {scenario.name} with {len(factories)} controllers",
            flush=True,
        )
    return rows


def _methods(rows: Sequence[dict[str, float | int | str]]) -> list[str]:
    return list(dict.fromkeys(str(row["method"]) for row in rows))


def _method_rows(
    rows: Sequence[dict[str, float | int | str]],
    method: str,
) -> list[dict[str, float | int | str]]:
    return [row for row in rows if row["method"] == method]


def _aggregate(
    rows: Sequence[dict[str, float | int | str]],
    method: str,
) -> dict[str, float]:
    selected = _method_rows(rows, method)
    return {
        "pass_rate_pct": 100.0 * np.mean([row["gate_pass"] == "yes" for row in selected]),
        "force_p95_n": float(np.percentile([row["force_rmse_n"] for row in selected], 95)),
        "force_worst_n": max(float(row["force_rmse_n"]) for row in selected),
        "peak_p95_n": float(np.percentile([row["peak_force_n"] for row in selected], 95)),
        "peak_worst_n": max(float(row["peak_force_n"]) for row in selected),
        "tangent_p95_mm": float(
            np.percentile([row["tangent_rmse_mm"] for row in selected], 95)
        ),
        "tangent_worst_mm": max(float(row["tangent_rmse_mm"]) for row in selected),
        "contact_worst_pct": min(float(row["contact_ratio_pct"]) for row in selected),
        "saturation_worst_pct": max(float(row["saturation_pct"]) for row in selected),
    }


def _write_comparison_summary(
    rows: list[dict[str, float | int | str]],
    gate: ResidualRlGate,
    path: Path,
) -> None:
    lines = [
        "# Franka adaptive control and bounded Residual RL comparison",
        "",
        "All methods use the same frozen 24 cases and the same per-case simulation seeds.",
        "The seed-29 set is retained for continuity with v0.3 and is not described as a blind test.",
        "",
        (
            "| Method | Pass rate | Force RMSE P95 / worst [N] | Raw peak P95 / worst [N] | "
            "Tangent P95 / worst [mm] | Contact worst | Saturation worst |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    aggregates = {}
    for method in _methods(rows):
        aggregate = _aggregate(rows, method)
        aggregates[method] = aggregate
        lines.append(
            f"| {method} | {aggregate['pass_rate_pct']:.1f}% | "
            f"{aggregate['force_p95_n']:.2f} / {aggregate['force_worst_n']:.2f} | "
            f"{aggregate['peak_p95_n']:.2f} / {aggregate['peak_worst_n']:.2f} | "
            f"{aggregate['tangent_p95_mm']:.2f} / {aggregate['tangent_worst_mm']:.2f} | "
            f"{aggregate['contact_worst_pct']:.1f}% | "
            f"{aggregate['saturation_worst_pct']:.2f}% |"
        )

    adaptive = aggregates["adaptive_hybrid"]
    residual = aggregates["bounded_residual_rl"]
    residual_meets_gate = residual["pass_rate_pct"] >= gate.min_case_pass_rate_pct
    improves_adaptive = residual["pass_rate_pct"] > adaptive["pass_rate_pct"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The bounded residual policy meets the pre-declared 90% case-pass target."
                if residual_meets_gate
                else "The bounded residual policy does not meet the pre-declared 90% case-pass target."
            ),
            (
                "It improves case pass rate over the adaptive classical baseline."
                if improves_adaptive
                else "It does not improve case pass rate over the adaptive classical baseline."
            ),
            "Training return is therefore not used as evidence of deployment readiness.",
            "See `comparison.csv` for every physical metric and failure label.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_training(records: Sequence[TrainingRecord], path: Path) -> None:
    iterations = [record.iteration for record in records]
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.plot(iterations, [record.mean_cost for record in records], marker="o", label="current")
    axis.plot(iterations, [record.best_cost for record in records], label="best so far")
    axis.set_xlabel("ARS iteration")
    axis.set_ylabel("Mean physical rollout cost")
    axis.set_title("Bounded residual policy training")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_comparison(
    rows: list[dict[str, float | int | str]],
    gate: ResidualRlGate,
    path: Path,
) -> None:
    methods = _methods(rows)
    cases = list(dict.fromkeys(str(row["case"]) for row in rows))
    x = np.arange(len(cases))
    width = 0.24
    plots = (
        ("force_rmse_n", gate.max_force_rmse_n, "Force RMSE [N]"),
        ("peak_force_n", gate.max_peak_force_n, "Raw peak force [N]"),
        ("tangent_rmse_mm", gate.max_tangent_rmse_mm, "Tangential RMSE [mm]"),
        ("saturation_pct", gate.max_saturation_pct, "Torque saturation [%]"),
    )
    fig, axes = plt.subplots(4, 1, figsize=(12.0, 11.0), sharex=True)
    colors = ("#6c757d", "#277da1", "#e76f51")
    for axis, (metric, limit, label) in zip(axes, plots, strict=True):
        for method_index, (method, color) in enumerate(zip(methods, colors, strict=True)):
            selected = _method_rows(rows, method)
            values = [float(row[metric]) for row in selected]
            offset = (method_index - (len(methods) - 1) / 2) * width
            axis.bar(x + offset, values, width=width, label=method, color=color)
        axis.axhline(limit, color="black", linestyle="--", linewidth=1.0)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3, fontsize=9)
    axes[-1].set_xticks(x, cases, rotation=55, ha="right", fontsize=8)
    fig.suptitle("Same-case comparison: fixed vs adaptive vs bounded Residual RL")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_frozen_holdout_report(
    policy: LinearResidualPolicy,
    output_dir: Path,
    holdout_cases: int = 24,
    holdout_duration: float = 4.5,
    holdout_seed: int = 29,
) -> list[dict[str, float | int | str]]:
    """Run and persist the same-case physical comparison for a frozen policy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = ResidualRlGate()
    rows = evaluate_frozen_holdout(
        policy,
        count=holdout_cases,
        duration=holdout_duration,
        seed=holdout_seed,
        gate=gate,
    )
    _write_csv(rows, output_dir / "comparison.csv")
    _write_comparison_summary(rows, gate, output_dir / "summary.md")
    _plot_comparison(rows, gate, output_dir / "comparison.png")
    return rows


def run_learning_experiment(
    output_dir: Path,
    training_config: ArsTrainingConfig | None = None,
    holdout_cases: int = 24,
    holdout_duration: float = 4.5,
    holdout_seed: int = 29,
) -> list[dict[str, float | int | str]]:
    training_config = training_config or ArsTrainingConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy, records, training_scenarios = train_residual_policy(training_config)
    policy_path = output_dir / "policy.json"
    policy.save(
        policy_path,
        metadata={
            "algorithm": "augmented_random_search",
            **asdict(training_config),
            "residual_safety": residual_safety_manifest(),
        },
    )
    training_rows = [asdict(record) for record in records]
    _write_csv(training_rows, output_dir / "training_curve.csv")
    _plot_training(records, output_dir / "training_curve.png")

    rows = write_frozen_holdout_report(
        policy,
        output_dir,
        holdout_cases=holdout_cases,
        holdout_duration=holdout_duration,
        holdout_seed=holdout_seed,
    )
    gate = ResidualRlGate()
    manifest = {
        "training_config": asdict(training_config),
        "training_cases": [asdict(scenario) for scenario in training_scenarios],
        "holdout": {
            "scenario_seed": holdout_seed,
            "simulation_seeds": [holdout_seed + index for index in range(holdout_cases)],
            "cases": holdout_cases,
            "duration": holdout_duration,
            "status": "frozen_public_holdout",
        },
        "gate": asdict(gate),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "residual_safety": residual_safety_manifest(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/franka_learning"))
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--directions", type=int, default=6)
    parser.add_argument("--top-directions", type=int, default=3)
    parser.add_argument("--training-cases", type=int, default=6)
    parser.add_argument("--training-duration", type=float, default=3.2)
    parser.add_argument("--policy-seed", type=int, default=17)
    parser.add_argument("--holdout-cases", type=int, default=24)
    parser.add_argument("--holdout-duration", type=float, default=4.5)
    parser.add_argument("--holdout-seed", type=int, default=29)
    parser.add_argument(
        "--policy",
        type=Path,
        help="load a frozen policy and skip training",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.policy is not None:
        write_frozen_holdout_report(
            LinearResidualPolicy.load(args.policy),
            args.output,
            holdout_cases=args.holdout_cases,
            holdout_duration=args.holdout_duration,
            holdout_seed=args.holdout_seed,
        )
        return
    config = ArsTrainingConfig(
        iterations=args.iterations,
        directions=args.directions,
        top_directions=args.top_directions,
        training_cases=args.training_cases,
        duration=args.training_duration,
        policy_seed=args.policy_seed,
    )
    run_learning_experiment(
        args.output,
        training_config=config,
        holdout_cases=args.holdout_cases,
        holdout_duration=args.holdout_duration,
        holdout_seed=args.holdout_seed,
    )


if __name__ == "__main__":
    main()
