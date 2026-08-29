"""Plots and MuJoCo rendering for the Franka benchmark."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

from compliant_control_lab.franka_simulation import FrankaTrialResult, franka_model_path

COLORS = {
    "impedance": "#1b9e77",
    "admittance": "#7570b3",
    "hybrid": "#277da1",
}


def plot_franka_scenario(results: Sequence[FrankaTrialResult], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), sharex=True)
    for result in results:
        color = COLORS[result.controller]
        axes[0].plot(
            result.time,
            result.normal_force,
            color=color,
            linewidth=1.4,
            label=result.controller,
        )
        tangent_error = np.linalg.norm(
            result.position[:, 1:] - result.desired_position[:, 1:], axis=1
        )
        axes[1].plot(
            result.time,
            1_000.0 * tangent_error,
            color=color,
            linewidth=1.2,
            label=result.controller,
        )
        axes[2].plot(
            result.time,
            np.rad2deg(result.orientation_error_rad),
            color=color,
            linewidth=1.2,
            label=result.controller,
        )

    reference = results[0]
    axes[0].plot(
        reference.time,
        reference.desired_force,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="target",
    )
    axes[0].set_ylabel("Normal force [N]")
    axes[1].set_ylabel("Tangential error [mm]")
    axes[2].set_ylabel("Orientation error [deg]")
    axes[2].set_xlabel("Time [s]")
    axes[0].set_title(f"Franka 7-DOF — scenario: {reference.scenario}")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_franka_gif(
    result: FrankaTrialResult,
    output_path: Path,
    fps: int = 25,
    width: int = 640,
    height: int = 480,
) -> None:
    """Replay logged joint positions through the MuJoCo renderer."""
    model = mujoco.MjModel.from_xml_path(str(franka_model_path()))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    timestep = float(np.mean(np.diff(result.time)))
    sample_period = max(1, round(1.0 / (fps * timestep)))
    images: list[Image.Image] = []
    try:
        for index in range(0, len(result.time), sample_period):
            data.qpos[:7] = result.q[index]
            data.qpos[7:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="demo")
            images.append(Image.fromarray(renderer.render().copy()))
    finally:
        renderer.close()

    if not images:
        raise ValueError("No frames were generated")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=round(1_000 / fps),
        loop=0,
        optimize=True,
    )

