"""Render recorded seven-joint motion and raw telemetry, without plant integration."""

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

from compliant_control_lab.surface_contact_validation import _path_label
from compliant_control_lab.surface_experiment import _output_path, _sha256
from compliant_control_lab.surface_sensing import franka_surface_model_path

TRACE_NAME = "long_case_16_mu_0.45_friction.npz"
FPS, STRIDE = 25, 20
DISCLAIMER = "SIMULATION | logged joints, not plant replay"


def load_source(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (directory / "COMPLETE").read_text().strip() != _sha256(manifest_path):
        raise ValueError("source COMPLETE mismatch")
    if manifest["experiment_identity"] != "tangential-compensation-public24-v1":
        raise ValueError("wrong experiment identity")
    if _sha256(directory / TRACE_NAME) != manifest["artifact_sha256"][TRACE_NAME]:
        raise ValueError("source trace hash mismatch")
    cases = [
        case
        for case in manifest["case_configurations"]
        if (
            case["category"] == "long"
            and case["method"] == "friction"
            and case["case_index"] == 16
            and case["scenario"]["wall_sliding_friction"] == 0.45
        )
    ]
    if len(cases) != 1 or cases[0]["scenario"]["tool_sliding_friction"] != 0.45:
        raise ValueError("expected unique long case 16 with both friction coefficients 0.45")
    case = cases[0]
    with np.load(directory / TRACE_NAME, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    if case["config"]["duration"] != 12 or case["config"]["timestep"] != 0.002:
        raise ValueError("demo requires the original 12 s / 500 Hz trace")
    if (
        str(trace["controller_kind"]) != "surface_friction"
        or str(trace["contact_model"]) != "smooth"
    ):
        raise ValueError("trace controller/contact identity mismatch")
    for key, shape in {
        "time": (6000,),
        "q": (6000, 7),
        "position": (6000, 3),
        "target_position": (6000, 3),
        "true_normal_force": (6000,),
        "target_normal_force": (6000,),
        "kinematic_sample_time": (6000,),
        "raw_wrench_sample_time": (6000,),
    }.items():
        if trace[key].shape != shape or not np.all(np.isfinite(trace[key])):
            raise ValueError(f"invalid trace field: {key}")
    if not np.allclose(trace["time"], np.arange(6000) * 0.002, rtol=0, atol=1e-12):
        raise ValueError("invalid trace time grid")
    for key in ("kinematic_sample_time", "raw_wrench_sample_time"):
        if not np.array_equal(trace[key], trace["time"]):
            raise ValueError("joint/force timestamps must align")
    package = franka_surface_model_path().parent.parent
    assets = {
        name: digest
        for name, digest in manifest["source_and_assets_sha256"].items()
        if name.startswith("assets/")
    }
    if not assets or any(_sha256(package / name) != digest for name, digest in assets.items()):
        raise ValueError("current rendering assets differ from the source archive")
    identity = {
        "directory": _path_label(directory),
        "manifest_sha256": _sha256(manifest_path),
        "trace": TRACE_NAME,
        "trace_sha256": _sha256(directory / TRACE_NAME),
        "asset_sha256": assets,
        "experiment_versions": manifest["versions"],
    }
    return trace, case, identity


def tangent_error_mm(trace, yaw_deg):
    angle = np.deg2rad(yaw_deg)
    normal = np.array([np.cos(angle), np.sin(angle), 0])
    error = trace["position"] - trace["target_position"]
    return 1000 * np.linalg.norm(error - np.outer(error @ normal, normal), axis=1)


def _render(staging, trace, case, ffmpeg):
    model = mujoco.MjModel.from_xml_path(str(franka_surface_model_path()))
    wall = model.geom("contact_wall").id
    yaw = np.deg2rad(case["scenario"]["wall_yaw_deg"])
    normal = np.array([np.cos(yaw), np.sin(yaw), 0])
    model.geom_pos[wall, :2] = [0.4, 0] + model.geom_size[wall, 0] * normal[:2]
    model.geom_quat[wall] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    fingers = data.qpos[7:].copy()
    renderer = mujoco.Renderer(model, height=480, width=640)
    figure = Figure(figsize=(12.8, 5.4), dpi=100)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.3, 1),
        left=0.01,
        right=0.98,
        top=0.83,
        bottom=0.12,
        wspace=0.19,
        hspace=0.48,
    )
    scene = figure.add_subplot(grid[:, 0])
    scene.axis("off")
    picture = scene.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    force_axis, error_axis = (figure.add_subplot(grid[index, 1]) for index in (0, 1))
    # Keep every 500 Hz force/error sample. Only camera frames are subsampled.
    with matplotlib.rc_context({"path.simplify": False}):
        force_axis.plot(
            trace["time"], trace["true_normal_force"], linewidth=0.8, label="raw true Fn"
        )
        force_axis.plot(
            trace["time"], trace["target_normal_force"], "--", linewidth=0.7, label="target"
        )
        error_axis.plot(
            trace["time"], tangent_error_mm(trace, case["scenario"]["wall_yaw_deg"]), linewidth=0.8
        )
    cursors = []
    for axis, label in ((force_axis, "Normal force [N]"), (error_axis, "Tangent error [mm]")):
        axis.set(xlim=(0, 12), ylabel=label)
        axis.grid(alpha=0.2)
        cursors.append(axis.axvline(0, color="black", linewidth=1))
    force_axis.legend(fontsize=8, loc="upper right")
    error_axis.set_xlabel("Original logged time [s]")
    figure.suptitle(DISCLAIMER, fontsize=16, weight="bold", color="#9d250d")
    figure.text(
        0.5,
        0.9,
        "Case 16 | fixed smooth model | friction feedforward | raw curves: 500 Hz",
        ha="center",
        fontsize=10,
    )
    title = scene.set_title("", fontsize=10)
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
        "1280x540",
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
    try:
        for index in range(0, len(trace["time"]), STRIDE):
            data.qpos[:7], data.qpos[7:] = trace["q"][index], fingers
            mujoco.mj_forward(model, data)  # Kinematics only: never mj_step or mj_step2.
            renderer.update_scene(data, camera="demo")
            picture.set_data(renderer.render())
            time = trace["time"][index]
            title.set_text(f"t = {time:.2f} s | seven logged joints; fingers fixed at home")
            for cursor in cursors:
                cursor.set_xdata([time, time])
            figure.canvas.draw()
            rgb = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
            if index == 3000:
                Image.fromarray(rgb).save(staging / "preview.png")
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


def render_demo(source_dir, output_dir):
    source_dir = Path(source_dir)
    output = _output_path(output_dir)
    if output.exists():
        raise ValueError("demo output must be a fresh directory")
    if output.is_relative_to(source_dir.resolve()) or source_dir.resolve().is_relative_to(output):
        raise ValueError("demo output must be outside the source archive and its ancestors")
    trace, case, identity = load_source(source_dir)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg unavailable; use the existing representative plot")
    script_hash = _sha256(Path(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tangential-demo-", dir=output.parent) as temporary:
        staging = Path(temporary) / "demo"
        staging.mkdir()
        _render(staging, trace, case, ffmpeg)
        if load_source(source_dir)[2] != identity or _sha256(Path(__file__)) != script_hash:
            raise ValueError("source or render script changed during rendering")
        manifest = {
            "identity": "tangential-logged-motion-demo-v1",
            "disclaimer": DISCLAIMER,
            "source": identity,
            "case_configuration": case,
            "script_sha256": script_hash,
            "fps": FPS,
            "frame_stride": STRIDE,
            "frame_count": 300,
            "duration_s": 12,
            "raw_curve_sample_count": 6000,
            "raw_curve_hz": 500,
            "rendering": "surface XML; archive wall pose; qpos[:7] from log; home fingers; mj_forward only; demo camera; no recomputed force curves",
            "versions": {
                "mujoco": mujoco.__version__,
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
                "python": platform.python_version(),
                "ffmpeg": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
            },
            "artifact_sha256": {path.name: _sha256(path) for path in staging.iterdir()},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
        _output_path(output)
        if output.exists():
            raise ValueError("demo destination appeared during rendering")
        os.rename(staging, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("results/franka_tangential_development")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render_demo(args.source, args.output))
