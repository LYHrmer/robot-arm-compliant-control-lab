"""One-command check of the published archive and the main simulation path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.franka_simulation import (
    DEFAULT_FRANKA_SCENARIOS,
    FrankaSimulationConfig,
    FrankaTrialResult,
    run_franka_trial,
)
from compliant_control_lab.published_results_audit import (
    PublishedResultAudit,
    audit_published_results,
)

DEFAULT_PROTOCOL = Path("results/franka_safety_preholdout/protocol.json")
DEFAULT_RESULT_DIR = Path("results/franka_safety_blind")


@dataclass(frozen=True)
class SmokeReport:
    archive: PublishedResultAudit
    simulation_steps: int
    force_rmse_n: float
    peak_force_n: float
    saturation_pct: float

    def render(self) -> str:
        frozen_decision = "PASS" if self.archive.primary_rule_passed else "FAIL"
        return "\n".join(
            (
                (
                    f"archive: PASS ({self.archive.row_count} rows, "
                    f"frozen_decision={frozen_decision})"
                ),
                (
                    "simulation: PASS "
                    f"(safe_adaptive_hybrid/nominal, steps={self.simulation_steps}, "
                    f"force_rmse={self.force_rmse_n:.2f} N, "
                    f"raw_peak={self.peak_force_n:.2f} N, "
                    f"saturation={self.saturation_pct:.2f}%)"
                ),
                "smoke: PASS",
            )
        )


def _validate_trial(result: FrankaTrialResult) -> dict[str, float | str]:
    sample_count = len(result.time)
    if sample_count == 0:
        raise ValueError("simulation returned no samples")

    signals = {
        "time": result.time,
        "position": result.position,
        "normal_force": result.normal_force,
        "raw_normal_force": result.raw_normal_force,
        "torque": result.torque,
        "controller_time_us": result.controller_time_us,
    }
    for name, values in signals.items():
        if len(values) != sample_count:
            raise ValueError(f"simulation signal length mismatch: {name}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"simulation returned non-finite values: {name}")

    metrics = result.metrics()
    numeric_metrics = (
        "force_rmse_n",
        "peak_force_n",
        "tangent_rmse_mm",
        "saturation_pct",
        "controller_p95_us",
    )
    if not all(np.isfinite(float(metrics[name])) for name in numeric_metrics):
        raise ValueError("simulation returned non-finite metrics")
    saturation_pct = float(metrics["saturation_pct"])
    if not 0.0 <= saturation_pct <= 100.0:
        raise ValueError("simulation returned an invalid saturation percentage")
    return metrics


def run_smoke(
    protocol_path: Path = DEFAULT_PROTOCOL,
    result_dir: Path = DEFAULT_RESULT_DIR,
    duration: float = 2.0,
) -> SmokeReport:
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be positive and finite")

    archive = audit_published_results(protocol_path, result_dir)
    trial = run_franka_trial(
        FrankaSafeAdaptiveController(),
        scenario=DEFAULT_FRANKA_SCENARIOS[0],
        config=FrankaSimulationConfig(duration=duration),
    )
    metrics = _validate_trial(trial)
    return SmokeReport(
        archive=archive,
        simulation_steps=len(trial.time),
        force_rmse_n=float(metrics["force_rmse_n"]),
        peak_force_n=float(metrics["peak_force_n"]),
        saturation_pct=float(metrics["saturation_pct"]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--duration", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        report = run_smoke(args.protocol, args.result, args.duration)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"smoke: FAIL: {error}") from error
    print(report.render())


if __name__ == "__main__":
    main()
