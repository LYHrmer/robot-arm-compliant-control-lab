"""CLI for the Franka 7-DOF compliant-control benchmark."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

from compliant_control_lab.franka_control import default_franka_controllers
from compliant_control_lab.franka_plotting import plot_franka_scenario, save_franka_gif
from compliant_control_lab.franka_simulation import (
    DEFAULT_FRANKA_SCENARIOS,
    FrankaScenario,
    FrankaSimulationConfig,
    FrankaTrialResult,
    run_franka_trial,
)


def _write_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, float | str]], output_path: Path) -> None:
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_franka_benchmark(
    output_dir: Path,
    duration: float = 4.5,
    controller_names: Iterable[str] | None = None,
    scenarios: Iterable[FrankaScenario] = DEFAULT_FRANKA_SCENARIOS,
    create_gif: bool = False,
) -> list[dict[str, float | str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_names = list(controller_names or default_franka_controllers())
    metrics: list[dict[str, float | str]] = []
    all_results: list[FrankaTrialResult] = []

    for scenario in scenarios:
        scenario_results = []
        for controller_name in selected_names:
            controllers = default_franka_controllers()
            if controller_name not in controllers:
                raise ValueError(f"Unknown controller: {controller_name}")
            result = run_franka_trial(
                controllers[controller_name],
                scenario=scenario,
                config=FrankaSimulationConfig(duration=duration),
            )
            scenario_results.append(result)
            all_results.append(result)
            row = result.metrics()
            metrics.append(row)
            print(
                f"finished controller={controller_name:<10} scenario={scenario.name:<12} "
                f"force_rmse={row['force_rmse_n']:.2f} N"
            )
        plot_franka_scenario(scenario_results, output_dir / f"{scenario.name}.png")

    _write_csv(metrics, output_dir / "metrics.csv")
    _write_markdown(metrics, output_dir / "metrics.md")
    if create_gif:
        demo = next(
            result
            for result in all_results
            if result.controller == "hybrid" and result.scenario == "nominal"
        )
        save_franka_gif(demo, output_dir / "hybrid_demo.gif")
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/franka"))
    parser.add_argument("--duration", type=float, default=4.5)
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=list(default_franka_controllers()),
        default=list(default_franka_controllers()),
    )
    parser.add_argument("--gif", action="store_true", help="also create a rendered GIF")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_franka_benchmark(
        args.output,
        duration=args.duration,
        controller_names=args.controllers,
        create_gif=args.gif,
    )


if __name__ == "__main__":
    main()

