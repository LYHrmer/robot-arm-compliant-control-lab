"""Plot two predeclared online-compensation diagnostics from a finalized archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.audit_online_compensation_errors import audit_archive

CASES = ("friction_step_down", "stop_hold_reverse")
ARMS = ("friction", "online")
SEED = 11
COLORS = {"friction": "#5B6573", "online": "#1976A3"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace_path(archive: Path, case: str, arm: str) -> Path:
    return archive / "traces" / f"{case}__seed_{SEED}__{arm}__compact.npz"


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        trace = {name: loaded[name] for name in loaded.files}
    required = {
        "time",
        "tangent_position_error_norm_m",
        "true_normal_force",
        "controller_coefficient_after_compute",
        "applied_wall_friction",
    }
    if required - trace.keys():
        raise ValueError(f"compact trace lacks plotting fields: {path.name}")
    count = len(trace["time"])
    if count == 0 or any(np.asarray(trace[name]).shape != (count,) for name in required):
        raise ValueError(f"compact trace plotting fields are not aligned: {path.name}")
    if not all(np.all(np.isfinite(trace[name])) for name in required):
        raise ValueError(f"compact trace plotting fields are not finite: {path.name}")
    return trace


def _shade(ax, case: str) -> None:
    if case == "friction_step_down":
        ax.axvline(6.0, color="#B55233", linestyle=":", linewidth=0.9, zorder=1)
        ax.axvspan(5.9, 6.1, color="#E8A87C", alpha=0.24, linewidth=0)
    else:
        ax.axvline(7.0, color="#B55233", linestyle=":", linewidth=0.9, zorder=1)
        ax.axvspan(5.5, 7.0, color="#A8B0B8", alpha=0.20, linewidth=0)


def _draw(traces: dict[tuple[str, str], dict[str, np.ndarray]]):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.4), sharex="col", constrained_layout=True)
    titles = ("Friction step-down", "Stop, hold, reverse")
    for column, (case, title) in enumerate(zip(CASES, titles)):
        for arm in ARMS:
            trace = traces[case, arm]
            axes[0, column].plot(
                trace["time"],
                1000.0 * trace["tangent_position_error_norm_m"],
                color=COLORS[arm],
                linewidth=1.05,
                label="Fixed friction" if arm == "friction" else "Online",
            )
            axes[1, column].plot(
                trace["time"],
                trace["true_normal_force"],
                color=COLORS[arm],
                linewidth=0.9,
            )
        online = traces[case, "online"]
        axes[2, column].plot(
            online["time"],
            online["controller_coefficient_after_compute"],
            color=COLORS["online"],
            linewidth=1.05,
            label="Online equivalent coefficient",
        )
        axes[2, column].plot(
            online["time"],
            online["applied_wall_friction"],
            color="#C47A2C",
            linestyle="--",
            linewidth=1.0,
            label="Applied friction (environment diagnostic)",
        )
        axes[1, column].axhline(
            12.0, color="#262B30", linestyle="--", linewidth=0.8, label="12 N reference"
        )
        for row in range(3):
            _shade(axes[row, column], case)
            axes[row, column].grid(color="#D7DCE0", linewidth=0.45, alpha=0.7)
            axes[row, column].set_xlim(0.0, 12.0)
        axes[0, column].set_title(title, fontweight="bold", pad=5)
        axes[0, column].text(
            6.08 if case == "friction_step_down" else 7.08,
            0.98,
            "6 s friction step" if case == "friction_step_down" else "7 s reverse ramp",
            transform=axes[0, column].get_xaxis_transform(),
            color="#8D402A",
            fontsize=6.2,
            va="top",
        )
        if case == "stop_hold_reverse":
            axes[0, column].text(
                6.25, 0.80, "Hold", transform=axes[0, column].get_xaxis_transform(),
                ha="center", va="top", color="#5B6573", fontsize=6.2,
            )
        axes[2, column].set_xlabel("Time [s]")

    axes[0, 0].set_ylabel("Tangential error [mm]")
    axes[1, 0].set_ylabel("True normal force [N]")
    axes[2, 0].set_ylabel("Coefficient [–]")
    axes[0, 0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), ncol=2, handlelength=2.2)
    axes[1, 0].legend(loc="upper right", handlelength=2.2)
    handles, labels = axes[2, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.015),
               ncol=2, fontsize=6.2, handlelength=2.2)
    axes[0, 0].text(-0.16, 1.08, "a", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=8)
    axes[0, 1].text(-0.16, 1.08, "b", transform=axes[0, 1].transAxes, fontweight="bold", fontsize=8)
    fig.suptitle("Online compensation under friction and motion changes", fontsize=9)
    fig.text(
        0.5,
        -0.065,
        "Applied friction is an environmental diagnostic, not the estimator target.",
        ha="center",
        fontsize=6.5,
        color="#4B5157",
    )
    return fig


def plot_archive(archive_dir: Path | str, output_path: Path | str) -> tuple[Path, Path]:
    archive = Path(archive_dir).resolve()
    output = Path(output_path).absolute()
    sidecar = output.with_name(output.name + ".provenance.json")
    if output.suffix.lower() != ".png":
        raise ValueError("output must be a .png file")
    if any(path.is_symlink() for path in (output, sidecar, *output.parents)):
        raise ValueError("output path must not contain symlinks")
    if output.exists() or sidecar.exists():
        raise FileExistsError("figure output or provenance sidecar already exists")

    audit = audit_archive(archive)
    if audit.get("status") != "PASS" or audit.get("is_subset") is not False:
        raise ValueError("plot requires a complete finalized full archive")
    paths = {(case, arm): _trace_path(archive, case, arm) for case in CASES for arm in ARMS}
    traces = {key: _load_trace(path) for key, path in paths.items()}
    fig = _draw(traces)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".online-error-plot-", dir=output.parent) as temporary:
        staged_figure = Path(temporary) / output.name
        staged_sidecar = Path(temporary) / sidecar.name
        fig.savefig(staged_figure, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        provenance = {
            "schema_version": 1,
            "archive_identity": audit["identity"],
            "archive_manifest_sha256": _sha256(archive / "manifest.json"),
            "archive_complete_sha256": _sha256(archive / "COMPLETE"),
            "input_trace_sha256": {
                str(path.relative_to(archive)): _sha256(path) for path in paths.values()
            },
            "source_sha256": {
                "tools/plot_online_compensation_errors.py": _sha256(Path(__file__).resolve()),
                "tools/audit_online_compensation_errors.py": _sha256(
                    Path(__file__).with_name("audit_online_compensation_errors.py")
                ),
            },
            "figure_sha256": _sha256(staged_figure),
            "seed": SEED,
            "cases": CASES,
            "arms": ARMS,
            "applied_friction_note": "Environmental diagnostic only; not an estimator target.",
        }
        staged_sidecar.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists() or sidecar.exists():
            raise FileExistsError("figure output or provenance sidecar appeared during rendering")
        os.link(staged_figure, output)
        try:
            os.link(staged_sidecar, sidecar)
        except Exception:
            output.unlink()
            raise
    return output, sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    figure, provenance = plot_archive(arguments.archive, arguments.output)
    print(f"figure: {figure}\nprovenance: {provenance}")


if __name__ == "__main__":
    main()
