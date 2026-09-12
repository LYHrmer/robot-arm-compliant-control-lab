"""Run and render one measured-load scheduling case on the Franka simulator."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
import mujoco
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from compliant_control_lab.surface_experiment import _output_path, _sha256
from compliant_control_lab.surface_sensing import franka_surface_model_path
from compliant_control_lab.surface_simulation import yaw_frame
from tools import budget_gain_interaction as interaction
from tools import measured_budget_trial, measured_budget_validation

FPS = 25
FRAME_STRIDE = 20
MINIMUM_FORCE_N = 6.0
MAXIMUM_FORCE_N = 8.0
ROTATION_GAIN_SCALE = 1.0
VELOCITY_ERROR_TIME_S = 0.05
DISCLAIMER = "MUJOCO SIMULATION | one fresh run; no hardware claim"
REPOSITORY = Path(__file__).resolve().parents[2]
LEGACY_ARCHIVE = REPOSITORY / "results/franka_measured_budget_full"


def representative_case():
    """Return the predeclared yaw-0 combined-error case, independent of list order."""
    matches = [
        (variant, case)
        for variant, case in interaction.cases()
        if variant == "combined" and case.scenario.wall_yaw_deg == 0
    ]
    if len(matches) != 1:
        raise ValueError("expected one yaw-0 combined measured-load case")
    variant, case = matches[0]
    if (
        case.config.duration != 12.0
        or case.config.timestep != 0.002
        or case.config.seed != 11
    ):
        raise ValueError("representative case must remain the 12 s / 500 Hz / seed-11 case")
    controller = interaction.controller(case, ROTATION_GAIN_SCALE, MAXIMUM_FORCE_N)
    if controller._base.tangential.velocity_error_time != VELOCITY_ERROR_TIME_S:
        raise ValueError("representative preset must keep velocity_error_time at 0.05 s")
    return variant, case


def tangent_error_mm(trace: dict, wall_yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(wall_yaw_deg)
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    error = np.asarray(trace["position"]) - np.asarray(trace["target_position"])
    return 1000.0 * np.linalg.norm(error - np.outer(error @ normal, normal), axis=1)


def display_signals(trace: dict, wall_yaw_deg: float) -> dict[str, np.ndarray]:
    return {
        "tangent_error_mm": tangent_error_mm(trace, wall_yaw_deg),
        "tangent_compensation_n": np.linalg.norm(
            np.asarray(trace["load_compensation_force_local"])[:, 1:], axis=1
        ),
        "budget_applied_n": np.asarray(trace["load_budget_applied_n"], dtype=float),
        "budget_next_n": np.asarray(trace["load_budget_next_n"], dtype=float),
    }


def validate_trace(trace: dict, case) -> dict[str, np.ndarray]:
    count = round(case.config.duration / case.config.timestep)
    required = {
        "time": (count,),
        "q": (count, 7),
        "position": (count, 3),
        "target_position": (count, 3),
        "true_normal_force": (count,),
        "target_normal_force": (count,),
        "load_compensation_force_local": (count, 3),
        "load_budget_applied_n": (count,),
        "load_budget_next_n": (count,),
    }
    for name, shape in required.items():
        value = np.asarray(trace.get(name))
        if value.shape != shape or not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"invalid trace field: {name}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"non-finite trace field: {name}")
    expected_time = np.arange(count) * case.config.timestep
    if not np.allclose(trace["time"], expected_time, rtol=0.0, atol=1e-12):
        raise ValueError("trace does not use the case's 500 Hz clock")
    for name in ("load_budget_applied_n", "load_budget_next_n"):
        values = np.asarray(trace[name])
        if np.any(values < MINIMUM_FORCE_N - 1e-12) or np.any(
            values > MAXIMUM_FORCE_N + 1e-12
        ):
            raise ValueError(f"{name} leaves the 6-8 N preset")
    return display_signals(trace, case.scenario.wall_yaw_deg)


def source_identity() -> dict:
    """Bind this run to its v2 code and the verified 153-source baseline."""
    legacy_manifest_path = LEGACY_ARCHIVE / "manifest.json"
    legacy_manifest = json.loads(legacy_manifest_path.read_text())
    if (LEGACY_ARCHIVE / "COMPLETE").read_text().strip() != _sha256(legacy_manifest_path):
        raise ValueError("legacy measured-budget COMPLETE mismatch")
    source_hashes_path = LEGACY_ARCHIVE / "source_hashes.json"
    source_hashes = json.loads(source_hashes_path.read_text())
    if len(source_hashes) != 153:
        raise ValueError("legacy measured-budget identity must contain 153 sources")
    if legacy_manifest["artifact_sha256"]["source_hashes.json"] != _sha256(source_hashes_path):
        raise ValueError("legacy measured-budget source identity hash mismatch")
    for name, digest in source_hashes.items():
        path = REPOSITORY / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"current baseline source differs from the legacy identity: {name}")
    runtime_paths = (
        REPOSITORY / "tools/measured_budget_trial.py",
        REPOSITORY / "tools/measured_budget_validation.py",
    )
    return {
        "legacy_measured_budget_v1": {
            "directory": "results/franka_measured_budget_full",
            "manifest_sha256": _sha256(legacy_manifest_path),
            "source_count": len(source_hashes),
            "source_hashes_sha256": _sha256(source_hashes_path),
            "sources": source_hashes,
        },
        "v2_runtime": {
            path.relative_to(REPOSITORY).as_posix(): _sha256(path) for path in runtime_paths
        },
    }


def run_case():
    variant, case = representative_case()
    result = measured_budget_trial.run_dynamic(
        case,
        ROTATION_GAIN_SCALE,
        "adaptive6_8",
        "fresh",
    )
    validate_trace(result.trace, case)
    frame = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg)
    validation = measured_budget_validation.validate_trace(
        result.trace,
        frame,
        minimum_force=MINIMUM_FORCE_N,
        max_force=MAXIMUM_FORCE_N,
    )
    return result.trace, variant, case, validation


def _configure_figure(trace: dict, case):
    signals = display_signals(trace, case.scenario.wall_yaw_deg)
    figure = Figure(figsize=(14.0, 7.2), dpi=100)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=(1.25, 1.0),
        left=0.015,
        right=0.985,
        top=0.86,
        bottom=0.09,
        wspace=0.20,
        hspace=0.46,
    )
    scene = figure.add_subplot(grid[:, 0])
    scene.axis("off")
    picture = scene.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    axes = [figure.add_subplot(grid[row, 1]) for row in range(3)]
    time = np.asarray(trace["time"])
    with matplotlib.rc_context({"path.simplify": False}):
        axes[0].plot(time, signals["tangent_error_mm"], color="#2d6a9f", linewidth=0.8)
        axes[1].plot(
            time, trace["true_normal_force"], color="#272727", linewidth=0.8, label="contact"
        )
        axes[1].plot(
            time,
            trace["target_normal_force"],
            "--",
            color="#c14d32",
            linewidth=0.8,
            label="target",
        )
        axes[2].plot(
            time,
            signals["tangent_compensation_n"],
            color="#19745d",
            linewidth=0.8,
            label="compensation",
        )
        axes[2].plot(
            time,
            signals["budget_applied_n"],
            color="#d17a00",
            linewidth=0.9,
            label="applied budget",
        )
        axes[2].plot(
            time,
            signals["budget_next_n"],
            ":",
            color="#8a3f9d",
            linewidth=0.9,
            label="next budget",
        )
    labels = ("Tangential error [mm]", "Normal force [N]", "Tangential force / budget [N]")
    cursors = []
    for axis, label in zip(axes, labels, strict=True):
        axis.set(xlim=(0.0, case.config.duration), ylabel=label)
        axis.grid(alpha=0.2)
        cursors.append(axis.axvline(0.0, color="#111111", linewidth=0.9))
    axes[1].legend(fontsize=8, loc="upper right")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].set_xlabel("Shared simulation time [s]")
    figure.suptitle(DISCLAIMER, fontsize=16, weight="bold", color="#9d250d")
    figure.text(
        0.5,
        0.905,
        "Franka 7-DOF | combined error, yaw 0° | adaptive measured-load budget 6–8 N",
        ha="center",
        fontsize=10,
    )
    title = scene.set_title("", fontsize=10)
    return figure, picture, title, cursors


def _render(staging: Path, trace: dict, case, ffmpeg: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(franka_surface_model_path()))
    wall_id = model.geom("contact_wall").id
    yaw = np.deg2rad(case.scenario.wall_yaw_deg)
    normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    model.geom_pos[wall_id, :2] = [0.4, 0.0] + model.geom_size[wall_id, 0] * normal[:2]
    model.geom_quat[wall_id] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    finger_qpos = data.qpos[7:].copy()
    renderer = mujoco.Renderer(model, height=480, width=640)
    figure, picture, title, cursors = _configure_figure(trace, case)
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1400x720",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(staging / "demo.mp4"),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    preview_index = round(8.5 / case.config.timestep / FRAME_STRIDE) * FRAME_STRIDE
    try:
        for index in range(0, len(trace["time"]), FRAME_STRIDE):
            data.qpos[:7], data.qpos[7:] = trace["q"][index], finger_qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="demo")
            picture.set_data(renderer.render())
            current_time = float(trace["time"][index])
            title.set_text(f"t = {current_time:.2f} s | logged joints from this run")
            for cursor in cursors:
                cursor.set_xdata([current_time, current_time])
            figure.canvas.draw()
            rgb = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
            if index == preview_index:
                Image.fromarray(rgb).save(staging / "overview.png")
            process.stdin.write(rgb.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error}")
    finally:
        renderer.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()
        if not process.stdin.closed:
            process.stdin.close()


def generate_demo(output_dir: str | Path) -> Path:
    output = _output_path(output_dir)
    if output.exists():
        raise ValueError("demo output must be a fresh directory")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the synchronized demo")
    script_hash = _sha256(Path(__file__))
    sources = source_identity()
    trace, variant, case, validation = run_case()
    signals = validate_trace(trace, case)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".measured-budget-demo-", dir=output.parent) as temp:
        staging = Path(temp) / "demo"
        staging.mkdir()
        np.savez_compressed(staging / "trace.npz", **trace)
        (staging / "validation_report.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n"
        )
        _render(staging, trace, case, ffmpeg)
        if not all((staging / name).is_file() for name in ("demo.mp4", "overview.png")):
            raise ValueError("renderer did not produce both video and overview")
        if _sha256(Path(__file__)) != script_hash or source_identity() != sources:
            raise ValueError("demo sources changed while the demo was running")
        manifest = {
            "identity": "franka-measured-budget-representative-demo-v2",
            "disclaimer": DISCLAIMER,
            "case": interaction.case_document(variant, case),
            "preset": {
                "minimum_force_n": MINIMUM_FORCE_N,
                "maximum_force_n": MAXIMUM_FORCE_N,
                "rotation_gain_scale": ROTATION_GAIN_SCALE,
                "velocity_error_time_s": VELOCITY_ERROR_TIME_S,
            },
            "run": {
                "duration_s": case.config.duration,
                "timestep_s": case.config.timestep,
                "sample_count": len(trace["time"]),
                "camera_fps": FPS,
                "camera_frame_stride": FRAME_STRIDE,
                "max_tangent_error_mm": float(np.max(signals["tangent_error_mm"])),
                "max_tangent_compensation_n": float(
                    np.max(signals["tangent_compensation_n"])
                ),
                "budget_range_n": [
                    float(np.min(signals["budget_applied_n"])),
                    float(np.max(signals["budget_applied_n"])),
                ],
            },
            "rendering": (
                "fresh MuJoCo run; qpos[:7] from its trace; mj_forward for camera playback; "
                "all curves use the same 500 Hz trace clock"
            ),
            "script_sha256": script_hash,
            "source_identity": sources,
            "model_xml_sha256": _sha256(franka_surface_model_path()),
            "versions": {
                "mujoco": mujoco.__version__,
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
                "python": platform.python_version(),
                "ffmpeg": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
            },
            "artifact_sha256": {
                path.name: _sha256(path) for path in sorted(staging.iterdir())
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
        if output.exists():
            raise ValueError("demo destination appeared while rendering")
        os.rename(staging, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(generate_demo(arguments.output))
