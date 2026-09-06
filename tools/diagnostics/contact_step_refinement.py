"""Refine physics only for public noisy case 16; keep control and F/T at 500 Hz.

F/T is sampled at the FIRST solved substep, rotated immediately, and cached for
the next control cycle. Extra substeps hold torque and never consume sensor RNG.
This process-local patch is a diagnostic, not a production runner or a hardware
validation. Smooth versus legacy changes contact compliance, not the controller.
Run from the repository with PYTHONPATH=.local-deps:src.
"""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from compliant_control_lab import surface_simulation as sim
from compliant_control_lab.surface_experiment import _source_hashes


def _grid_metrics(samples, evaluation_start, timestep):
    """Columns: time, normal force, tangent load, signed gap, tangent error, target force."""
    time, force, tangent, gap, error, target = np.asarray(samples).T
    window = time >= evaluation_start
    contact = np.flatnonzero(force > 0)
    result = {
        "sample_count": len(time),
        "first_raw_contact_time_s": float(time[contact[0]]) if contact.size else None,
        "peak_force_n": float(np.max(force)),
        "seconds_over_35_n": float(np.count_nonzero(force > 35) * timestep),
        "evaluation_sample_count": int(np.count_nonzero(window)),
    }
    result.update(dict.fromkeys((
        "force_rmse_n", "contact_ratio_pct", "mean_normal_force_n", "tangent_rmse_mm",
        "mean_tangent_force_n", "geometry_separation_pct", "maximum_signed_gap_um",
        "median_loaded_penetration_um", "median_loaded_friction_ratio",
    )))
    if np.any(window):
        loaded = window & (force > 0.5)
        result.update(
            force_rmse_n=float(np.sqrt(np.mean((force[window] - target[window]) ** 2))),
            contact_ratio_pct=float(100 * np.mean(force[window] > 0.5)),
            mean_normal_force_n=float(np.mean(force[window])),
            tangent_rmse_mm=float(1000 * np.sqrt(np.mean(error[window] ** 2))),
            mean_tangent_force_n=float(np.mean(tangent[window])),
            geometry_separation_pct=float(100 * np.mean(gap[window] > 0)),
            maximum_signed_gap_um=float(1e6 * np.max(gap[window])),
            median_loaded_penetration_um=float(-1e6 * np.median(gap[loaded]))
            if np.any(loaded) else None,
            median_loaded_friction_ratio=float(np.median(tangent[loaded] / force[loaded]))
            if np.any(loaded) else None,
        )
    return result


def run_refinement(substeps, *, config=None):
    """Return (ordinary replayable trial, JSON report, all-physics sample matrix).

    Config permits short test durations; the CLI fixes the original 4.5 s case.
    Patches are process-global: do not call concurrently within the same process.
    """
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    config = config or sim.SurfaceSimulationConfig(contact_model="smooth")
    if config.timestep != 0.002:
        raise ValueError("control period must remain 0.002 s (500 Hz)")
    scenario = sim.SurfaceScenario(name="case16", wall_yaw_deg=15, wall_time_constant=0.005)
    task, frame = sim.SurfaceTask(yaw_deg=15), sim.yaw_frame(15)
    normal, h = frame.rotation[:, 0], config.timestep / substeps
    original_const, original_step1, original_step2 = (
        mujoco.mj_setConst, mujoco.mj_step1, mujoco.mj_step2
    )
    original_read, original_force = sim.ToolWrenchSensor.read_world, sim._normal_contact_force
    samples, pair_parameters, engine_options = [], {}, {}
    cache = {"sensor": None, "pending": False, "reads": 0}
    initial_pose = None

    def configure_model(model, data):
        model.opt.timestep = h
        engine_options.update({name: float(getattr(model.opt, name)) for name in (
            "integrator", "cone", "solver", "iterations", "tolerance", "impratio",
            "noslip_iterations", "ls_iterations", "ls_tolerance", "disableflags",
        )})
        original_const(model, data)

    def read_sensor(sensor, data):
        if cache["sensor"] is None:
            cache["sensor"], cache["reads"] = sensor, 1
            return original_read(sensor, data)  # Explicit reset/forward observation.
        if not cache["pending"]:
            raise RuntimeError("unexpected extra F/T read would change sensor sampling")
        cache["pending"] = False
        return cache["wrench"].copy()

    def step_control_interval(model, data):
        nonlocal initial_pose
        tool, wall, site = model.geom("tool_tip").id, model.geom("contact_wall").id, model.site("ee_site").id
        if initial_pose is None:
            initial_pose = (data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy())
        held_torque = data.ctrl.copy()
        for index in range(substeps):
            if index:
                original_step1(model, data)
            time = len(samples) * h
            if abs(data.time - time) > 1e-9 or not np.array_equal(data.ctrl, held_torque):
                raise RuntimeError("physics timestamp or held torque changed")
            original_step2(model, data)
            # Pose/contact/sensordata still belong to the just-solved preintegration state.
            force, tangent_world, contact_wrench = original_force(model, data, tool, wall), np.zeros(3), np.zeros(6)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                if {contact.geom1, contact.geom2} == {tool, wall}:
                    mujoco.mj_contactForce(model, data, contact_index, contact_wrench)
                    tangent_world += contact.frame.reshape(3, 3)[1:].T @ contact_wrench[1:3]
                    if not pair_parameters:
                        pair_parameters.update(dim=int(contact.dim), solref=contact.solref.tolist(),
                                               solimp=contact.solimp.tolist(), friction=contact.friction.tolist())
            position = data.site_xpos[site]
            target = task.target_at(time, *initial_pose, config.target_force)
            error = position - target.position
            gap = normal @ (np.array([0.4, 0, 0]) - position) - 0.025
            samples.append((time, force, np.linalg.norm(tangent_world), gap,
                            np.linalg.norm(error - normal * (normal @ error)), target.normal_force))
            if index == 0:
                cache.update(wrench=original_read(cache["sensor"], data), force=force, pending=True)
                cache["reads"] += 1

    with (
        patch.object(mujoco, "mj_setConst", configure_model),
        patch.object(mujoco, "mj_step2", step_control_interval),
        patch.object(sim.ToolWrenchSensor, "read_world", read_sensor),
        patch.object(sim, "_normal_contact_force", lambda *_: cache["force"]),
    ):
        trial = sim.run_surface_trial(frame, scenario, config, task)
    samples = np.asarray(samples)
    first = samples[::substeps]
    np.testing.assert_allclose(first[:, 0], trial.trace["time"], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(first[:, 1], trial.trace["true_normal_force"])
    np.testing.assert_allclose(first[:, 3], trial.trace["true_contact_gap_m"], atol=1e-12, rtol=0)
    # The production loop otherwise reads this evaluator-only field at the LAST substep.
    trial.trace["true_tangent_force_n"] = first[:, 2].copy()
    if cache["reads"] != len(first) + 1 or cache["pending"]:
        raise RuntimeError("sensor noise cadence changed")
    report = {
        "substeps": int(substeps), "physics_period_s": h, "control_period_s": config.timestep,
        "scenario": asdict(scenario), "config": asdict(config), "task": asdict(task),
        "controller_frame_rotation": frame.rotation.tolist(), "controller_kind": "surface_adaptive",
        "sensor_reads_including_reset": cache["reads"], "engine_options": engine_options,
        "resolved_contact_pair": pair_parameters, "ordinary_trial_metrics": trial.metrics(),
        "control_grid": _grid_metrics(first, config.evaluation_start, config.timestep),
        "all_physics": _grid_metrics(samples, config.evaluation_start, h),
    }
    return trial, report, samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--substeps", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--contact-model", choices=("smooth", "legacy"), default="smooth")
    args = parser.parse_args(argv)
    output = args.output.absolute()
    if output.exists() or any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("output must be new and contain no symlink components")
    if len(set(args.substeps)) != len(args.substeps) or any(n < 1 for n in args.substeps):
        raise ValueError("substeps must be unique positive integers")
    sources = _source_hashes()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = []
    for count in args.substeps:
        _, report, _ = run_refinement(count, config=sim.SurfaceSimulationConfig(contact_model=args.contact_model))
        rows.append(report)
        print(json.dumps({"substeps": count, "control_grid": report["control_grid"],
                          "all_physics": report["all_physics"]}), flush=True)
    if sources != _source_hashes() or script_hash != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("source changed during refinement")
    payload = {
        "identity": "surface-contact-phase-strict-refinement-v1",
        "scope": "Public case16 diagnostic, not holdout or hardware validation. Smooth changes compliance.",
        "sampling": {
            "control_and_filter_hz": 500, "torque": "zero-order hold over N physics substeps",
            "feedback": "reset forward solve, then first solved substep cached for next control cycle",
            "wrench_rotation": "immediately after first step2 at solved TCP orientation; no origin shift",
            "noise": "one F/T draw per control cycle plus reset; no substep noise draws",
            "time": "preintegration solve time; control grid is every Nth all-physics sample",
            "evaluation": "fixed [1.5, 4.5) seconds; contact > 0.5 N; gap > 0 means separated",
        },
        "engine_versions": {"mujoco": mujoco.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "source_and_assets_sha256": sources, "script_sha256": script_hash, "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
