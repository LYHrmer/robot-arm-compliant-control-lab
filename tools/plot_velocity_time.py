"""Plot the velocity-error time coefficient tradeoff from frozen true-state traces."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure

from tools.budget_transfer import verify_archive

ROOT = Path(__file__).resolve().parents[1]
NEW_REFERENCE = (
    "results/franka_velocity_time",
    "803376ba2042d5ca8550fa6d2608676096cedd9678cabe8317d09220e4252f7d",
)
BASELINE_REFERENCE = (
    "results/franka_onset_observer",
    "e01db28864f04057d5272c4f0e469e2261c5166e5afe877afe2f00ef2d66a2ce",
)
CASE_INDEX = 23
MAX_FORCE_N = 8.0
NEW_VELOCITY_TIME_S = 0.10
SURFACE_YAW_DEG = 15.0
NORMAL = np.array(
    [np.cos(np.deg2rad(SURFACE_YAW_DEG)), np.sin(np.deg2rad(SURFACE_YAW_DEG)), 0.0]
)
WINDOW_S = (1.5, 2.2)
GAINS = (("s1", 1.0), ("s2", 2.0))
COLORS = {"005": "#1F3D7A", "010": "#D2762A"}
LABELS = {"005": "Velocity-error time 0.05 s", "010": "Velocity-error time 0.10 s"}
RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "svg.hashsalt": "velocity-time-tradeoff",
}


def _field(trace, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        values = np.array(trace[name], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid or missing trace field: {name}") from error
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise ValueError(f"trace field must be finite with shape {shape}: {name}")
    return values


def _along(trace):
    """Project one trace onto the target tangent inside the plotted window."""
    try:
        time = np.array(trace["time"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid or missing trace field: time") from error
    if time.ndim != 1 or not len(time) or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be a finite increasing one-dimensional grid")
    shape = (len(time), 3)
    position = _field(trace, "position", shape)
    target_position = _field(trace, "target_position", shape)
    velocity = _field(trace, "linear_velocity", shape)
    target_velocity = _field(trace, "target_linear_velocity", shape)
    mask = (time >= WINDOW_S[0]) & (time < WINDOW_S[1])
    if not np.any(mask):
        raise ValueError(f"plotted window {WINDOW_S} must contain samples")
    tangent = target_velocity[mask] - np.outer(target_velocity[mask] @ NORMAL, NORMAL)
    speed = np.linalg.norm(tangent, axis=1)
    if np.any(speed <= 1e-12):
        raise ValueError("target tangential speed must stay positive across the window")
    direction = tangent / speed[:, None]
    lag_mm = 1_000.0 * np.sum((target_position[mask] - position[mask]) * direction, axis=1)
    velocity_mm_s = 1_000.0 * np.sum((velocity[mask] - target_velocity[mask]) * direction, axis=1)
    return time, mask, (target_position, target_velocity), lag_mm, velocity_mm_s


def paired_series(before, after) -> dict[str, np.ndarray]:
    """Pair a 0.05 s baseline run with its 0.10 s follow-up on one shared target."""
    time, mask, targets, lag_005, velocity_005 = _along(before)
    other_time, other_mask, other_targets, lag_010, velocity_010 = _along(after)
    if not np.array_equal(time, other_time) or not np.array_equal(mask, other_mask):
        raise ValueError("paired traces must share one time grid")
    if not all(np.array_equal(a, b) for a, b in zip(targets, other_targets)):
        raise ValueError("paired traces must share one target position and velocity")
    return {
        "time": time[mask],
        "lag_005_mm": lag_005,
        "lag_010_mm": lag_010,
        "velocity_005_mm_s": velocity_005,
        "velocity_010_mm_s": velocity_010,
    }


def _load_trace(archive: Path, name: str, gain: float, velocity_time_s=None):
    with np.load(archive / "traces" / name, allow_pickle=False) as loaded:
        trace = {field: loaded[field] for field in loaded.files}
    expected = {
        "case_index": CASE_INDEX,
        "method": "online",
        "max_force_n": MAX_FORCE_N,
        "rotation_gain_scale": gain,
    }
    if velocity_time_s is not None:
        expected["velocity_error_time_s"] = velocity_time_s
    for field, value in expected.items():
        if field not in trace:
            raise ValueError(f"trace lacks metadata field {field}: {name}")
        actual = trace[field][()]
        if np.ndim(actual) or actual != value:
            raise ValueError(f"trace metadata mismatch for {field}: {name}")
    return trace


def _draw(series) -> Figure:
    figure = Figure(figsize=(7.2, 4.8), layout="constrained")
    axes = figure.subplots(2, 2, sharex=True)
    for column, (_, gain) in enumerate(GAINS):
        paired = series[gain]
        top, bottom = axes[0, column], axes[1, column]
        bottom.axhspan(-1.0, 1.0, color="#7F8FA6", alpha=0.14, linewidth=0.0,
                       label="Position lag within 1 mm")
        for key in ("005", "010"):
            top.plot(paired["time"], paired[f"velocity_{key}_mm_s"], color=COLORS[key],
                     linewidth=1.05, label=LABELS[key])
            bottom.plot(paired["time"], paired[f"lag_{key}_mm"], color=COLORS[key], linewidth=1.05)
        for axis in (top, bottom):
            axis.axhline(0.0, color="#33383D", linestyle="--", linewidth=0.7, zorder=1)
            axis.grid(color="#DDE2E8", linewidth=0.45, alpha=0.8)
            axis.set_xlim(*WINDOW_S)
        top.set_title(f"Rotation gain scale {gain:g} (true evaluator states)", fontsize=8)
        bottom.set_title("True evaluator position lag", fontsize=7.5, color="#4B5157")
        bottom.set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Along-path velocity error [mm/s]\n(positive = faster than target)")
    axes[1, 0].set_ylabel("Along-path position lag [mm]\n(positive = behind target)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    band, band_label = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles + band, labels + band_label, loc="outside lower center", ncol=3)
    figure.suptitle(
        f"Case {CASE_INDEX} ({SURFACE_YAW_DEG:g} deg yaw wall), 8 N budget: "
        "velocity-error time 0.05 s vs 0.10 s",
        fontsize=9,
    )
    return figure


def render(root: Path, output: Path) -> Path:
    """Verify both frozen archives and write the 2x2 tradeoff figure as SVG."""
    root = Path(root).resolve()
    output = Path(output).absolute()
    if output.suffix.lower() != ".svg":
        raise ValueError("output must be a .svg file")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"figure output already exists: {output}")
    output = output.resolve()
    archives = {}
    for key, (directory, pin) in (("new", NEW_REFERENCE), ("baseline", BASELINE_REFERENCE)):
        archive = (root / directory).resolve()
        if archive == output or archive in output.parents:
            raise ValueError("output must not be written inside a frozen archive")
        verify_archive(archive, pin)
        archives[key] = archive

    series = {}
    for suffix, gain in GAINS:
        before = _load_trace(archives["baseline"], f"case{CASE_INDEX}__{suffix}__f8n.npz", gain)
        after = _load_trace(
            archives["new"],
            f"case{CASE_INDEX}__{suffix}__f8n__tv100ms.npz",
            gain,
            NEW_VELOCITY_TIME_S,
        )
        series[gain] = paired_series(before, after)

    description = json.dumps(
        {
            "case_index": CASE_INDEX,
            "source_manifest_sha256": {
                NEW_REFERENCE[0]: NEW_REFERENCE[1],
                BASELINE_REFERENCE[0]: BASELINE_REFERENCE[1],
            },
            "states": "true evaluator only",
            "window_s": list(WINDOW_S),
        },
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(RC_PARAMS):
        figure = _draw(series)
        buffer = io.StringIO()
        FigureCanvasSVG(figure).print_svg(
            buffer, metadata={"Date": None, "Description": description}
        )
    # Matplotlib leaves trailing blanks in SVG paths; normalize for source control.
    svg = "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(svg)
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args(argv)
    print(f"figure: {render(arguments.root, arguments.output)}")


if __name__ == "__main__":
    main()
