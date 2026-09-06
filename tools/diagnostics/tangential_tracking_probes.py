"""Fixed tangential probes and closed-loop force-depth characterization, not material ID."""

import argparse
import hashlib
import json
import os
import platform
import tempfile
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
from compliant_control_lab.surface_contact_validation import _path_label
from compliant_control_lab.surface_control import SurfaceAdaptiveController
from compliant_control_lab.surface_experiment import _source_hashes

PROBES = ("baseline", "friction_zero", "double_k", "slow", "load_6", "load_12", "load_18")
REFERENCE = (
    Path(__file__).resolve().parents[2]
    / "results/franka_surface_contact_fix/representative_case_16_smooth_surface_exact.npz"
)


class HoldTask(sim.SurfaceTask):
    def target_at(self, time, initial_position, initial_rotation, target_force):
        return super().target_at(min(time, 1.2), initial_position, initial_rotation, target_force)


class SlowTask(sim.SurfaceTask):
    def target_at(self, time, initial_position, initial_rotation, target_force):
        if time <= 1.2:
            return super().target_at(time, initial_position, initial_rotation, target_force)
        target = super().target_at(
            1.2 + 0.5 * (time - 1.2), initial_position, initial_rotation, target_force
        )
        return replace(target, linear_velocity=0.5 * target.linear_velocity)


def run_probe(name: str) -> dict:
    if name not in PROBES:
        raise ValueError("unknown tangential probe")
    static = name.startswith("load_")
    scenario = sim.SurfaceScenario(
        name="tangential_diagnostic",
        wall_yaw_deg=0,
        wall_time_constant=0.012,
        position_noise_std_m=0,
        force_noise_std_n=0,
        torque_noise_std_nm=0,
        force_bias_sensor_n=(0, 0, 0),
        torque_bias_sensor_nm=(0, 0, 0),
    )
    config = sim.SurfaceSimulationConfig(
        duration=3.0,
        timestep=0.002,
        seed=11,
        contact_model="smooth",
        target_force=float(name.split("_")[1]) if static else 12.0,
        evaluation_start=2.0 if static else 1.5,
    )
    task = HoldTask() if static else SlowTask() if name == "slow" else sim.SurfaceTask()
    original_const = mujoco.mj_setConst

    def configure_model(model, data):
        if name == "friction_zero":
            ids = [model.geom(item).id for item in ("tool_tip", "contact_wall")]
            model.geom_friction[ids, 0] = 0
        original_const(model, data)

    def controller(frame):
        hybrid = FrankaHybridController(
            force_transition_time=0.5,
            tangential_stiffness=np.array([0.0, 450.0, 450.0]) * (2 if name == "double_k" else 1),
        )
        adaptive = FrankaAdaptiveHybridController(base=hybrid)
        return SurfaceAdaptiveController(frame, base=FrankaSafeAdaptiveController(base=adaptive))

    with (
        patch.object(mujoco, "mj_setConst", configure_model),
        patch.object(sim, "SurfaceAdaptiveController", controller),
    ):
        result = sim.run_surface_trial(sim.yaw_frame(0), scenario, config, task)
    trace = result.trace
    window = trace["time"] >= config.evaluation_start
    if not np.any(window):
        raise ValueError("probe must observe the configured evaluation window")
    penetration = np.maximum(-trace["true_contact_gap_m"][window], 0.0)
    return {
        "probe": name,
        "scenario": asdict(scenario),
        "config": asdict(config),
        "task": asdict(task),
        "task_variant": type(task).__name__,
        "metrics": result.metrics(),
        "overrides": {
            "friction_zero": {"both_geom_sliding_friction": 0},
            "double_k": {"tangent_stiffness_multiplier": 2},
            "slow": {"tangent_clock_rate": 0.5},
        }.get(name, {"hold_after_s": 1.2} if static else {}),
        "actual_normal_force_median_n": float(np.median(trace["true_normal_force"][window])),
        "median_penetration_mm": float(1000 * np.median(penetration)),
        "max_penetration_mm": float(1000 * np.max(penetration)),
    }


def analyze_reference_trace(path: Path = REFERENCE) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        trace = {name: stored[name] for name in stored.files}
    window = (trace["time"] >= 1.5) & (trace["time"] < 4.5)
    rotation = trace["controller_frame_rotation"]
    local = lambda name: (trace[name][window] @ rotation)[:, 1:]
    error = local("target_position") - local("position")
    measured_error = local("target_position") - local("measured_position")
    velocity = local("target_linear_velocity")
    speed = np.linalg.norm(velocity, axis=1)
    direction = velocity / speed[:, None]
    lag = np.sum(error * direction, axis=1)
    scale = 1 + 0.25 * np.clip(np.linalg.norm(measured_error, axis=1) / 0.02, 0, 1)
    spring = 450 * scale[:, None] * measured_error
    damping = 38 * np.sqrt(scale[:, None]) * (velocity - local("measured_linear_velocity"))
    commanded = (trace["commanded_wrench"][window, :3] @ rotation)[:, 1:]
    reconstructed = (spring + damping) * trace["torque_projection_scale"][window, None]
    cross_error = error - lag[:, None] * direction
    return {
        "reference_path": _path_label(path),
        "reference_sha256": _hash(path),
        "window_s": [1.5, 4.5],
        "samples": int(np.count_nonzero(window)),
        "mean_signed_lag_mm": float(1000 * np.mean(lag)),
        "behind_target_pct": float(100 * np.mean(lag > 0)),
        "lag_direction_cosine_median": float(np.median(lag / np.linalg.norm(error, axis=1))),
        "cross_motion_error_rmse_mm": float(
            1000 * np.sqrt(np.mean(np.sum(cross_error**2, axis=1)))
        ),
        "target_speed_mean_m_s": float(np.mean(speed)),
        "measured_speed_mean_m_s": float(
            np.mean(np.linalg.norm(local("measured_linear_velocity"), axis=1))
        ),
        "scheduled_stiffness_median_n_m": float(np.median(450 * scale)),
        "command_force_mean_n": float(np.mean(np.linalg.norm(commanded, axis=1))),
        "spring_force_mean_n": float(np.mean(np.linalg.norm(spring, axis=1))),
        "damping_force_mean_n": float(np.mean(np.linalg.norm(damping, axis=1))),
        "true_contact_tangent_force_mean_n": float(np.mean(trace["true_tangent_force_n"][window])),
        "pd_reconstruction_max_error_n": float(np.max(np.abs(commanded - reconstructed))),
        "interpretation": "Descriptive friction-compatible offset plus dynamic tracking; force magnitudes do not establish vector force balance or causality.",
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _new_output(path: Path) -> Path:
    output = path.absolute()
    if output.exists() or any(item.is_symlink() for item in (output, *output.parents)):
        raise ValueError("output must be new and contain no symlink components")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _new_output(args.output)
    sources, script_hash, reference_hash = _source_hashes(), _hash(Path(__file__)), _hash(REFERENCE)
    rows = []
    for name in PROBES:
        rows.append(run_probe(name))
        print(f"completed tangential diagnostic {len(rows)}/7: {name}", flush=True)
    reference = analyze_reference_trace()
    if (
        _source_hashes() != sources
        or _hash(Path(__file__)) != script_hash
        or _hash(REFERENCE) != reference_hash
    ):
        raise RuntimeError("source/assets/script/reference changed during diagnostic")
    payload = {
        "identity": "tangential-tracking-diagnosis-v1",
        "scope": "Public diagnostic aggregates only, not full raw traces; no policy tuning or new holdout.",
        "force_depth_scope": "Closed-loop simulation characterization with fixed smooth contact; not real-material identification.",
        "tracking_probes": rows[:4],
        "static_force_depth": rows[4:],
        "reference_trace_analysis": reference,
        "source_and_assets_sha256": sources,
        "script_sha256": script_hash,
        "engine_versions": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tangent-", dir=output.parent) as temporary:
        staged = Path(temporary) / "result.json"
        staged.write_text(encoded, encoding="utf-8")
        _new_output(output)
        os.link(staged, output)


if __name__ == "__main__":
    main()
