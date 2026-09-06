"""Declared public preparation domain, with grouped seeds and varied task trajectories."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.reference_ablation import validate_output_path
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    yaw_frame,
)
from compliant_control_lab.surface_splits import split_cases

DOMAIN_SCHEMA = "surface_learning_preparation_v1"
DOMAIN_SEED = 20260906
PREPARATION_SPLIT = {"train_fraction": 0.7, "validation_fraction": 0.2}
# These bounds are fixed before the preparation rollouts, not fitted to pass rates.
DOMAIN_BOUNDS = {
    "wall_yaw_deg": (-15.0, 15.0),
    "sliding_friction": (0.25, 0.65),
    "normal_calibration_error_deg": (-5.0, 5.0),
    "wall_time_constant_s": (0.005, 0.012),
    "tool_mass_kg": (0.10, 0.13),
    "bias_compensation_scale": (0.95, 1.05),
    "force_bias_component_n": (-0.4, 0.4),
    "position_noise_std_m": (0.0002, 0.0005),
    "force_noise_std_n": (0.10, 0.25),
    "delay_steps": (0, 3),
    "target_force_n": (8.0, 16.0),
    "frequency_hz": (0.12, 0.25),
    "tangent_amplitude_m": (0.040, 0.065),
    "vertical_amplitude_m": (0.030, 0.045),
    "phase_rad": (0.0, 2 * np.pi),
}


def preparation_cases(group_count=12, *, seed=DOMAIN_SEED):
    """Three nominal anchors plus independent stratified parameter draws; two noise seeds/group.

    This is public development coverage, not exhaustive corners or a blind test.
    Task plane is known; controller normal has calibration error. Moving/unknown
    surfaces, non-Coulomb friction and hardware are outside this declared domain.
    """
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count < 6:
        raise ValueError("at least six physical/task groups are required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    count = group_count - 3
    draws = {}
    for name, (low, high) in DOMAIN_BOUNDS.items():
        # Independent strata per dimension reduce systematic parameter correlations.
        fractions = (rng.permutation(count) + rng.uniform(size=count)) / count
        draws[name] = low + (high - low) * fractions
    delays = rng.integers(0, 4, count)
    directions = rng.choice([-1, 1], count)
    cases = []
    for group in range(group_count):
        if group < 3:
            yaw, calibration_error = (-15.0, 0.0, 15.0)[group], 0.0
            scenario = SurfaceScenario(name=f"preparation_{group:03d}", wall_yaw_deg=yaw)
            task = LearningSurfaceTask(yaw_deg=yaw)
            force = 12.0
        else:
            index = group - 3
            values = {name: float(array[index]) for name, array in draws.items()}
            yaw = values["wall_yaw_deg"]
            calibration_error = values["normal_calibration_error_deg"]
            mu = values["sliding_friction"]
            scenario = SurfaceScenario(
                name=f"preparation_{group:03d}",
                wall_yaw_deg=yaw,
                wall_time_constant=values["wall_time_constant_s"],
                wall_sliding_friction=mu,
                tool_sliding_friction=mu,
                tool_mass_kg=values["tool_mass_kg"],
                bias_compensation_scale=values["bias_compensation_scale"],
                force_bias_sensor_n=tuple(rng.uniform(-0.4, 0.4, 3)),
                position_noise_std_m=values["position_noise_std_m"],
                force_noise_std_n=values["force_noise_std_n"],
                delay_steps=int(delays[index]),
            )
            task = LearningSurfaceTask(
                yaw_deg=yaw,
                frequency_hz=values["frequency_hz"],
                tangent_amplitude_m=values["tangent_amplitude_m"],
                vertical_amplitude_m=values["vertical_amplitude_m"],
                phase_rad=values["phase_rad"],
                direction=int(directions[index]),
            )
            force = values["target_force_n"]
        for noise_seed in (11, 29):
            cases.append(
                {
                    "case_id": f"g{group:03d}_seed{noise_seed}",
                    "scenario": asdict(scenario),
                    "config": asdict(
                        SurfaceSimulationConfig(
                            duration=12.0,
                            target_force=force,
                            seed=noise_seed,
                            contact_model="smooth",
                        )
                    ),
                    "task": asdict(task),
                    "controller_frame_rotation": yaw_frame(
                        yaw + calibration_error
                    ).rotation.tolist(),
                    "nominal_kind": "adaptive",
                }
            )
    return split_cases(cases, seed=seed, **PREPARATION_SPLIT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=12)
    args = parser.parse_args()
    output = validate_output_path(args.output)
    if output.exists():
        raise ValueError("case output must be a new file")
    cases = preparation_cases(args.groups)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation refuses a destination created by another caller meanwhile.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(cases, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"{len(cases)} public cases / {args.groups} physical-task groups: {output}")


if __name__ == "__main__":
    main()
