"""Randomized holdout benchmark used to gate a Residual RL extension."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from compliant_control_lab.franka_control import FrankaHybridController
from compliant_control_lab.franka_simulation import (
    FrankaScenario,
    FrankaSimulationConfig,
    run_franka_trial,
)


@dataclass(frozen=True)
class ResidualRlGate:
    """Pre-declared engineering limits for deciding whether learning is justified."""

    max_force_rmse_n: float = 2.0
    min_contact_ratio_pct: float = 95.0
    max_peak_force_n: float = 35.0
    max_tangent_rmse_mm: float = 15.0
    max_saturation_pct: float = 1.0
    min_case_pass_rate_pct: float = 90.0


def sample_stress_scenarios(count: int = 24, seed: int = 29) -> tuple[FrankaScenario, ...]:
    """Sample deterministic holdout cases over contact and sensing mismatch."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    scenarios = []
    for index in range(count):
        scenarios.append(
            FrankaScenario(
                name=f"holdout_{index:02d}",
                wall_time_constant=float(np.exp(rng.uniform(np.log(0.004), np.log(0.025)))),
                wall_sliding_friction=float(rng.uniform(0.20, 0.90)),
                wall_yaw_deg=float(rng.uniform(-6.0, 6.0)),
                position_noise_std=float(rng.uniform(0.0, 0.0008)),
                force_noise_std=float(rng.uniform(0.10, 0.80)),
                force_bias_n=float(rng.uniform(-1.5, 1.5)),
                delay_steps=int(rng.integers(0, 16)),
                bias_compensation_scale=float(rng.uniform(0.85, 1.15)),
            )
        )
    return tuple(scenarios)


def failed_gate_checks(
    metrics: dict[str, float | str],
    gate: ResidualRlGate,
) -> tuple[str, ...]:
    checks = (
        (float(metrics["force_rmse_n"]) <= gate.max_force_rmse_n, "force_rmse"),
        (float(metrics["contact_ratio_pct"]) >= gate.min_contact_ratio_pct, "contact_ratio"),
        (float(metrics["peak_force_n"]) <= gate.max_peak_force_n, "peak_force"),
        (float(metrics["tangent_rmse_mm"]) <= gate.max_tangent_rmse_mm, "tangent_rmse"),
        (float(metrics["saturation_pct"]) <= gate.max_saturation_pct, "saturation"),
    )
    return tuple(name for passed, name in checks if not passed)


def scenario_values(scenario: FrankaScenario) -> dict[str, float | int]:
    return {
        "wall_time_constant_s": scenario.wall_time_constant,
        "wall_sliding_friction": scenario.wall_sliding_friction,
        "wall_yaw_deg": scenario.wall_yaw_deg,
        "position_noise_std_m": scenario.position_noise_std,
        "force_noise_std_n": scenario.force_noise_std,
        "force_bias_n": scenario.force_bias_n,
        "measurement_delay_ms": 2 * scenario.delay_steps,
        "bias_compensation_scale": scenario.bias_compensation_scale,
    }


def _write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _percentile(rows: list[dict[str, float | int | str]], key: str, q: float) -> float:
    return float(np.percentile([float(row[key]) for row in rows], q))


def _write_summary(
    rows: list[dict[str, float | int | str]],
    gate: ResidualRlGate,
    output_path: Path,
) -> None:
    pass_rate = 100.0 * np.mean([row["gate_pass"] == "yes" for row in rows])
    residual_rl_triggered = pass_rate < gate.min_case_pass_rate_pct
    decision = (
        "GO TO EXPERIMENT: the impact-aware fixed-gain baseline misses the pre-declared "
        "robustness target, so evaluating a bounded residual policy is justified. This result "
        "does not by itself prove that RL will outperform adaptive classical control."
        if residual_rl_triggered
        else "NO-GO for the current wiping task: the impact-aware fixed-gain baseline already "
        "meets the pre-declared robustness target. Add Residual RL only after introducing a task with a "
        "measured performance gap, such as uncertain assembly or curved-surface normal estimation."
    )
    lines = [
        "# Franka hybrid controller — randomized holdout stress test",
        "",
        "## Pre-declared Residual RL gate",
        "",
        f"- Force RMSE: <= {gate.max_force_rmse_n:.1f} N",
        f"- Contact ratio: >= {gate.min_contact_ratio_pct:.1f}%",
        f"- Raw peak force: <= {gate.max_peak_force_n:.1f} N",
        f"- Tangential RMSE: <= {gate.max_tangent_rmse_mm:.1f} mm",
        f"- Torque saturation: <= {gate.max_saturation_pct:.1f}%",
        f"- Required case pass rate: >= {gate.min_case_pass_rate_pct:.1f}%",
        "",
        "## Holdout result",
        "",
        f"- Cases: {len(rows)}",
        f"- Case pass rate: {pass_rate:.1f}%",
        (
            f"- Force RMSE P50 / P95 / worst: {_percentile(rows, 'force_rmse_n', 50):.2f} / "
            f"{_percentile(rows, 'force_rmse_n', 95):.2f} / "
            f"{max(float(row['force_rmse_n']) for row in rows):.2f} N"
        ),
        (
            f"- Contact ratio P05 / worst: "
            f"{_percentile(rows, 'contact_ratio_pct', 5):.1f} / "
            f"{min(float(row['contact_ratio_pct']) for row in rows):.1f}%"
        ),
        (
            f"- Raw peak force P95 / worst: {_percentile(rows, 'peak_force_n', 95):.2f} / "
            f"{max(float(row['peak_force_n']) for row in rows):.2f} N"
        ),
        (
            f"- Tangential RMSE P95 / worst: "
            f"{_percentile(rows, 'tangent_rmse_mm', 95):.2f} / "
            f"{max(float(row['tangent_rmse_mm']) for row in rows):.2f} mm"
        ),
        f"- Saturation worst: {max(float(row['saturation_pct']) for row in rows):.2f}%",
        "",
        "## Decision",
        "",
        decision,
        "",
        "The randomized variables and every per-case metric are preserved in `metrics.csv`.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(rows: list[dict[str, float | int | str]], gate: ResidualRlGate, path: Path) -> None:
    indices = np.arange(len(rows))
    force_rmse = np.asarray([float(row["force_rmse_n"]) for row in rows])
    peak_force = np.asarray([float(row["peak_force_n"]) for row in rows])
    tangent_rmse = np.asarray([float(row["tangent_rmse_mm"]) for row in rows])
    passed = np.asarray([row["gate_pass"] == "yes" for row in rows])
    colors = np.where(passed, "#277da1", "#d1495b")

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True)
    axes[0].bar(indices, force_rmse, color=colors)
    axes[0].axhline(gate.max_force_rmse_n, color="black", linestyle="--")
    axes[0].set_ylabel("Force RMSE [N]")
    axes[1].bar(indices, peak_force, color=colors)
    axes[1].axhline(gate.max_peak_force_n, color="black", linestyle="--")
    axes[1].set_ylabel("Raw peak force [N]")
    axes[2].bar(indices, tangent_rmse, color=colors)
    axes[2].axhline(gate.max_tangent_rmse_mm, color="black", linestyle="--")
    axes[2].set_ylabel("Tangential RMSE [mm]")
    axes[2].set_xlabel("Randomized holdout case")
    axes[0].legend(
        handles=[
            Patch(color="#277da1", label="all gates pass"),
            Patch(color="#d1495b", label="one or more gates fail"),
            Line2D([0], [0], color="black", linestyle="--", label="metric gate"),
        ],
        ncol=3,
        fontsize=8,
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Franka hybrid force-position controller — robustness gate")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_stress_benchmark(
    output_dir: Path,
    count: int = 24,
    duration: float = 4.5,
    seed: int = 29,
    gate: ResidualRlGate | None = None,
) -> list[dict[str, float | int | str]]:
    gate = gate or ResidualRlGate()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for index, scenario in enumerate(sample_stress_scenarios(count=count, seed=seed)):
        result = run_franka_trial(
            FrankaHybridController(),
            scenario=scenario,
            config=FrankaSimulationConfig(duration=duration, seed=seed + index),
        )
        metrics = result.metrics()
        failures = failed_gate_checks(metrics, gate)
        row: dict[str, float | int | str] = {
            "case": scenario.name,
            "simulation_seed": seed + index,
            **scenario_values(scenario),
            **metrics,
            "gate_pass": "yes" if not failures else "no",
            "failed_checks": ";".join(failures),
        }
        rows.append(row)
        print(
            f"finished {scenario.name} force_rmse={float(metrics['force_rmse_n']):.2f} N "
            f"gate={'pass' if not failures else 'fail:' + ','.join(failures)}"
        )

    _write_csv(rows, output_dir / "metrics.csv")
    _write_summary(rows, gate, output_dir / "summary.md")
    _plot(rows, gate, output_dir / "stress_summary.png")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/franka_stress"))
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--duration", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_stress_benchmark(
        args.output,
        count=args.cases,
        duration=args.duration,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
