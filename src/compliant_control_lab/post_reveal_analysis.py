"""Generate a diagnostic report from the published v0.5 first-reveal CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAILURE_ORDER = (
    "force_rmse",
    "contact_ratio",
    "peak_force",
    "tangent_rmse",
    "saturation",
)
FAILURE_LABELS = {
    "force_rmse": "force RMSE",
    "contact_ratio": "contact ratio",
    "peak_force": "raw peak force",
    "tangent_rmse": "tangent RMSE",
    "saturation": "torque saturation",
}
DISPLAY_NAMES = {
    "fixed_hybrid": "fixed hybrid",
    "adaptive_hybrid": "adaptive hybrid",
    "safe_adaptive_hybrid": "torque-safe adaptive",
}


@dataclass(frozen=True)
class MethodGateSummary:
    method: str
    case_count: int
    pass_count: int
    failure_counts: dict[str, int]


def summarize_gate_results(csv_path: Path) -> tuple[MethodGateSummary, ...]:
    """Read per-case rows and count passes and failure flags for each method."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"method", "gate_pass", "failed_checks"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{csv_path} is empty or missing {sorted(required)}")

    method_order = list(dict.fromkeys(row["method"] for row in rows))
    summaries = []
    for method in method_order:
        method_rows = [row for row in rows if row["method"] == method]
        failure_counts: Counter[str] = Counter()
        for row in method_rows:
            failure_counts.update(failure for failure in row["failed_checks"].split(";") if failure)
        summaries.append(
            MethodGateSummary(
                method=method,
                case_count=len(method_rows),
                pass_count=sum(row["gate_pass"] == "yes" for row in method_rows),
                failure_counts=dict(failure_counts),
            )
        )
    case_counts = {summary.case_count for summary in summaries}
    if len(case_counts) != 1:
        raise ValueError("all methods must contain the same number of cases")
    return tuple(summaries)


def _display_name(method: str) -> str:
    if method in DISPLAY_NAMES:
        return DISPLAY_NAMES[method]
    return method.replace("torque_residual_run_", "residual ")


def _write_summary(
    summaries: tuple[MethodGateSummary, ...],
    required_passes: int,
    source_link: str,
    path: Path,
) -> None:
    case_count = summaries[0].case_count
    lines = [
        "# Post-reveal gate diagnosis",
        "",
        (
            f"This report is derived from the published first-reveal CSV. The {case_count} "
            "cases are now public validation data; this is diagnostic work, not new blind "
            "evidence."
        ),
        f"The frozen primary threshold was {required_passes}/{case_count} passes per policy.",
        "Failure columns are non-exclusive; one case may trigger more than one failure flag.",
        "",
        "| Method | Pass | Force RMSE failures | Peak-force failures | Tangent failures | Saturation failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        failures = summary.failure_counts
        lines.append(
            f"| {_display_name(summary.method)} | {summary.pass_count}/{summary.case_count} | "
            f"{failures.get('force_rmse', 0)} | {failures.get('peak_force', 0)} | "
            f"{failures.get('tangent_rmse', 0)} | {failures.get('saturation', 0)} |"
        )
    residual = [
        summary for summary in summaries if summary.method.startswith("torque_residual_run_")
    ]
    lines.append("")
    if residual:
        peak_failures = [summary.failure_counts.get("peak_force", 0) for summary in residual]
        tangent_failures = [summary.failure_counts.get("tangent_rmse", 0) for summary in residual]
        saturation_failures = sum(
            summary.failure_counts.get("saturation", 0) for summary in residual
        )
        lines.append(
            f"Across {len(residual)} residual policies, peak force failed in "
            f"{min(peak_failures)}–{max(peak_failures)} cases and tangential tracking failed "
            f"in {min(tangent_failures)}–{max(tangent_failures)} cases. The combined "
            f"saturation-failure count was {saturation_failures}. The next experiment "
            "instruments nominal approach and contact transition before changing the policy "
            "class."
        )
    else:
        lines.append("This input contains no torque-residual policy rows.")
    lines.extend(["", f"Source: [`{source_link}`]({source_link})."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(
    summaries: tuple[MethodGateSummary, ...],
    required_passes: int,
    path: Path,
) -> None:
    labels = [_display_name(summary.method) for summary in summaries]
    pass_counts = [summary.pass_count for summary in summaries]
    y = np.arange(len(summaries))
    method_colors = ["#6c757d", "#277da1", "#577590"] + ["#e76f51"] * max(0, len(summaries) - 3)

    fig, (pass_axis, failure_axis) = plt.subplots(1, 2, figsize=(14.0, 6.2))
    pass_axis.barh(y, pass_counts, color=method_colors)
    case_count = summaries[0].case_count
    pass_axis.axvline(
        required_passes,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"required: {required_passes}/{case_count}",
    )
    pass_axis.set_yticks(y, labels)
    pass_axis.invert_yaxis()
    pass_axis.set_xlim(0, case_count)
    pass_axis.set_xlabel("passing cases")
    pass_axis.set_title("Primary gate")
    pass_axis.grid(axis="x", alpha=0.25)
    pass_axis.legend(loc="lower right")
    for index, value in enumerate(pass_counts):
        pass_axis.text(value + 0.5, index, str(value), va="center", fontsize=9)

    bottoms = np.zeros(len(summaries))
    colors = ("#4c78a8", "#72b7b2", "#e45756", "#f2cf5b", "#b279a2")
    hatches = ("//", "..", "xx", "\\\\", "++")
    for reason, color, hatch in zip(FAILURE_ORDER, colors, hatches, strict=True):
        values = np.array(
            [summary.failure_counts.get(reason, 0) for summary in summaries], dtype=float
        )
        if not values.any():
            continue
        failure_axis.barh(
            y,
            values,
            left=bottoms,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.5,
            label=FAILURE_LABELS[reason],
        )
        bottoms += values
    failure_axis.set_yticks(y, labels)
    failure_axis.invert_yaxis()
    failure_axis.set_xlabel("failure flags (one case may have several)")
    failure_axis.set_title("Why cases failed")
    failure_axis.grid(axis="x", alpha=0.25)
    failure_axis.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Post-reveal diagnosis: {case_count} cases, now public validation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_post_reveal_analysis(
    csv_path: Path,
    output_dir: Path,
    *,
    required_passes: int,
) -> Path:
    """Write a readable failure summary without changing first-reveal artifacts."""
    source_dir = csv_path.resolve().parent
    resolved_output = output_dir.resolve()
    if resolved_output == source_dir or source_dir in resolved_output.parents:
        raise ValueError("post-reveal output must stay outside the first-reveal directory")
    summaries = summarize_gate_results(csv_path)
    case_count = summaries[0].case_count
    if not 0 < required_passes <= case_count:
        raise ValueError("required passes must be between one and the case count")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_link = Path(os.path.relpath(csv_path.resolve(), resolved_output)).as_posix()
    _write_summary(summaries, required_passes, source_link, output_dir / "summary.md")
    plot_path = output_dir / "failure_analysis.png"
    _plot(summaries, required_passes, plot_path)
    return plot_path


def _required_passes_from_protocol(protocol_path: Path) -> int:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    try:
        return int(protocol["blind_contract"]["primary_rule"]["minimum_passes_per_policy"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("protocol is missing the primary pass threshold") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/franka_safety_blind/comparison.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/franka_safety_postreveal"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/franka_safety_preholdout/protocol.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_post_reveal_analysis(
        args.input,
        args.output,
        required_passes=_required_passes_from_protocol(args.protocol),
    )


if __name__ == "__main__":
    main()
