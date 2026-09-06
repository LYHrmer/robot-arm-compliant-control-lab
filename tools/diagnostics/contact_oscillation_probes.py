"""One-factor contact diagnosis; temporary model overrides never edit scene assets.

Run from the installed repository. Each probe starts a fresh plant/controller and
changes one stated factor of the same 3 s noiseless yaw-zero surface task.
"""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from compliant_control_lab import surface_simulation as sim
from compliant_control_lab.franka_adaptive import (
    FrankaAdaptiveHybridController,
    FrankaSafeAdaptiveController,
)
from compliant_control_lab.franka_control import FrankaHybridController
from compliant_control_lab.surface_control import SurfaceAdaptiveController
from compliant_control_lab.surface_experiment import _source_hashes

PROBES = (
    "baseline",
    "hold_position",
    "zero_p",
    "zero_i",
    "normal_damping_60",
    "no_filter",
    "filter_5ms",
    "friction_zero",
    "friction_01",
    "impedance_half",
    "impratio_10",
    "smooth",
)


class HoldTask(sim.SurfaceTask):
    def target_at(self, time, initial_position, initial_rotation, target_force):
        return super().target_at(min(time, 1.2), initial_position, initial_rotation, target_force)


def run_probe(name: str, *, duration: float = 3.0) -> dict:
    if name not in PROBES:
        raise ValueError("unknown contact probe")
    scenario = sim.SurfaceScenario(
        name="noiseless_wiping_probe",
        wall_time_constant=0.012,
        position_noise_std_m=0,
        force_noise_std_n=0,
        torque_noise_std_nm=0,
        force_bias_sensor_n=(0, 0, 0),
    )
    config = sim.SurfaceSimulationConfig(duration=duration)
    if name in {"no_filter", "filter_5ms"}:
        config = replace(config, force_filter_time_constant=0 if name == "no_filter" else 0.005)
    if name == "smooth":
        config = replace(config, contact_model="smooth")
    task = HoldTask() if name == "hold_position" else sim.SurfaceTask()
    original_const = mujoco.mj_setConst

    def configure_model(model, data):
        ids = [model.geom(item).id for item in ("tool_tip", "contact_wall")]
        if name in {"friction_zero", "friction_01"}:
            # Same-priority geoms mix friction by MAX, so both must change.
            model.geom_friction[ids, 0] = 0 if name == "friction_zero" else 0.1
        if name == "impedance_half":
            model.geom_solimp[ids, 0] = 0.5
        if name == "impratio_10":
            model.opt.impratio = 10
        original_const(model, data)

    def controller(frame):
        parameters = {
            "zero_p": {"force_kp": 0},
            "zero_i": {"force_ki": 0},
            "normal_damping_60": {"normal_damping": 60},
        }.get(name, {})
        hybrid = FrankaHybridController(force_transition_time=0.5, **parameters)
        adaptive = FrankaAdaptiveHybridController(base=hybrid)
        return SurfaceAdaptiveController(frame, base=FrankaSafeAdaptiveController(base=adaptive))

    with (
        patch.object(mujoco, "mj_setConst", configure_model),
        patch.object(sim, "SurfaceAdaptiveController", controller),
    ):
        trial = sim.run_surface_trial(sim.yaw_frame(0), scenario, config, task)
    trace = trial.trace
    window = trace["time"] >= config.evaluation_start
    contact = window & (trace["true_normal_force"] > 0.5)
    if not np.any(window):
        raise ValueError("probe must observe the task evaluation window")
    ratios = trace["true_tangent_force_n"][contact] / trace["true_normal_force"][contact]
    return {
        "probe": name,
        "scenario": asdict(scenario),
        "config": asdict(config),
        "task": asdict(task),
        "metrics": trial.metrics(),
        "overrides": {
            "hold_after_s": 1.2 if name == "hold_position" else None,
            "force_kp": 0 if name == "zero_p" else None,
            "force_ki": 0 if name == "zero_i" else None,
            "normal_damping": 60 if name == "normal_damping_60" else None,
            "both_geom_sliding_friction": {"friction_zero": 0, "friction_01": 0.1}.get(name),
            "both_geom_initial_impedance": 0.5 if name == "impedance_half" else None,
            "impratio": 10 if name == "impratio_10" else None,
        },
        "geometry_separation_pct": float(100 * np.mean(trace["true_contact_gap_m"][window] > 0)),
        "maximum_gap_um": float(1e6 * np.max(trace["true_contact_gap_m"][window])),
        "median_penetration_um": float(-1e6 * np.median(trace["true_contact_gap_m"][window])),
        "median_loaded_friction_ratio": float(np.median(ratios)) if ratios.size else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", choices=PROBES, nargs="+", default=list(PROBES))
    args = parser.parse_args()
    output = args.output.absolute()
    if any(path.is_symlink() for path in (output, *output.parents)) or output.exists():
        raise ValueError("output must be new and contain no symlink components")
    if len(set(args.probes)) != len(args.probes):
        raise ValueError("probes must be unique")
    sources = _source_hashes()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = []
    for name in args.probes:
        rows.append(run_probe(name))
        print(f"completed contact probe: {name}", flush=True)
    if (
        _source_hashes() != sources
        or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_hash
    ):
        raise RuntimeError("source changed during probes")
    payload = {
        "identity": "wiping-contact-one-factor-diagnosis-v1",
        "scope": "Public diagnostic data, not holdout evaluation or controller tuning.",
        "control_period_s": 0.002,
        "engine_versions": {
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "rows": rows,
        "source_and_assets_sha256": sources,
        "script_sha256": script_hash,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


if __name__ == "__main__":
    main()
