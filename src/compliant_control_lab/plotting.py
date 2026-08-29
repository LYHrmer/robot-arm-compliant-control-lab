"""Benchmark plots and a lightweight GitHub-ready animation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from compliant_control_lab.kinematics import LINK_LENGTHS
from compliant_control_lab.simulation import WALL_SURFACE_X, TrialResult

COLORS = {
    "position": "#d95f02",
    "impedance": "#1b9e77",
    "admittance": "#7570b3",
    "hybrid": "#277da1",
}


def plot_scenario(results: Sequence[TrialResult], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    for result in results:
        color = COLORS[result.controller]
        axes[0].plot(
            result.time,
            result.normal_force,
            color=color,
            linewidth=1.4,
            label=result.controller,
        )
        axes[1].plot(
            result.time,
            1_000.0 * (result.position[:, 1] - result.desired_position[:, 1]),
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
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title(f"Scenario: {reference.scenario}")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_demo_gif(result: TrialResult, output_path: Path, fps: int = 25) -> None:
    """Render a compact top-view arm animation without requiring an OpenGL window."""
    sample_period = max(1, round(1.0 / (fps * np.mean(np.diff(result.time)))))
    frames = np.arange(0, len(result.time), sample_period)
    l1, l2 = LINK_LENGTHS

    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.set_xlim(-0.10, 0.80)
    axis.set_ylim(-0.35, 0.45)
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.grid(alpha=0.2)
    axis.axvspan(WALL_SURFACE_X, WALL_SURFACE_X + 0.05, color="#e76f51", alpha=0.8)
    arm_line, = axis.plot([], [], "o-", color="#277da1", linewidth=8, markersize=9)
    desired_point, = axis.plot([], [], "x", color="#2a9d8f", markersize=9, mew=2)
    status = axis.text(0.02, 0.96, "", transform=axis.transAxes, va="top", family="monospace")

    def update(frame_index: int):
        index = int(frames[frame_index])
        q1, q2 = result.q[index]
        joint = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
        tip = joint + np.array([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)])
        arm_line.set_data([0.0, joint[0], tip[0]], [0.0, joint[1], tip[1]])
        desired_point.set_data(
            [result.desired_position[index, 0]], [result.desired_position[index, 1]]
        )
        status.set_text(
            f"{result.controller} | t={result.time[index]:.2f}s | "
            f"F={result.normal_force[index]:.1f}N"
        )
        return arm_line, desired_point, status

    movie = animation.FuncAnimation(fig, update, frames=len(frames), interval=1_000 / fps, blit=True)
    movie.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
