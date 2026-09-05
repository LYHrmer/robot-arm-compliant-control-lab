"""Generate a diagnostic report from the published v0.5 first-reveal CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
PAIRED_METRICS = ("force_rmse_n", "peak_force_n", "tangent_rmse_mm")
PAIRED_METRIC_LABELS = {
    "force_rmse_n": "force RMSE (N)",
    "peak_force_n": "raw peak force (N)",
    "tangent_rmse_mm": "tangent RMSE (mm)",
}
BOOTSTRAP_SAMPLES = 20_000


@dataclass(frozen=True)
class MethodGateSummary:
    method: str
    case_count: int
    pass_count: int
    failure_counts: dict[str, int]


@dataclass(frozen=True)
class PairedEffect:
    method: str
    metric: str
    pair_count: int
    median_delta: float
    ci_low: float
    ci_high: float
    win_count: int
    tie_count: int
    loss_count: int


@dataclass(frozen=True)
class LeaveOneGateOutSummary:
    method: str
    case_count: int
    pass_count: int
    pass_count_without: dict[str, int]


def _read_rows(csv_path: Path, required: set[str]) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{csv_path} is missing {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    return rows


def _method_sort_key(method: str) -> tuple[int, str]:
    canonical = {
        "fixed_hybrid": 0,
        "adaptive_hybrid": 1,
        "safe_adaptive_hybrid": 2,
    }
    if method in canonical:
        return canonical[method], method
    if method.startswith("torque_residual_run_"):
        return 3, method
    return 4, method


def _index_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    by_method: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        method = row["method"]
        case = row["case"]
        method_rows = by_method.setdefault(method, {})
        if case in method_rows:
            raise ValueError(f"duplicate method/case pair: {method}/{case}")
        method_rows[case] = row
    return by_method


def _finite_metric(row: dict[str, str], metric: str) -> float:
    try:
        value = float(row[metric])
    except ValueError as error:
        raise ValueError(
            f"invalid {metric} for {row['method']}/{row['case']}: {row[metric]!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {metric} for {row['method']}/{row['case']}")
    return value


def compute_paired_effects(csv_path: Path) -> tuple[PairedEffect, ...]:
    """Compare every residual policy with the torque-safe adaptive baseline by case."""
    required = {"method", "case", *PAIRED_METRICS}
    rows = _read_rows(csv_path, required)
    by_method = _index_rows(rows)

    baseline = by_method.get("safe_adaptive_hybrid")
    if baseline is None:
        raise ValueError("safe_adaptive_hybrid baseline is missing")
    residual_methods = sorted(
        (method for method in by_method if method.startswith("torque_residual_run_")),
        key=_method_sort_key,
    )
    if not residual_methods:
        raise ValueError("no torque-residual policy rows found")

    effects = []
    ordered_cases = sorted(baseline)
    for method in residual_methods:
        method_rows = by_method[method]
        if set(method_rows) != set(baseline):
            missing = sorted(set(baseline) - set(method_rows))
            extra = sorted(set(method_rows) - set(baseline))
            raise ValueError(
                f"{method} case set does not match baseline "
                f"(missing={missing}, extra={extra})"
            )
        for metric in PAIRED_METRICS:
            deltas = np.asarray(
                [
                    _finite_metric(method_rows[case], metric)
                    - _finite_metric(baseline[case], metric)
                    for case in ordered_cases
                ],
                dtype=float,
            )
            seed_material = f"v0.5/paired-bootstrap/{method}/{metric}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            generator = np.random.default_rng(seed)
            samples = generator.integers(
                0,
                len(deltas),
                size=(BOOTSTRAP_SAMPLES, len(deltas)),
            )
            bootstrap_medians = np.median(deltas[samples], axis=1)
            ci_low, ci_high = np.quantile(bootstrap_medians, [0.025, 0.975])
            effects.append(
                PairedEffect(
                    method=method,
                    metric=metric,
                    pair_count=len(deltas),
                    median_delta=float(np.median(deltas)),
                    ci_low=float(ci_low),
                    ci_high=float(ci_high),
                    win_count=int(np.sum(deltas < 0.0)),
                    tie_count=int(np.sum(deltas == 0.0)),
                    loss_count=int(np.sum(deltas > 0.0)),
                )
            )
    return tuple(effects)


def summarize_leave_one_gate_out(csv_path: Path) -> tuple[LeaveOneGateOutSummary, ...]:
    """Count post-hoc passes after omitting each gate in turn."""
    required = {"method", "case", "gate_pass", "failed_checks"}
    rows = _read_rows(csv_path, required)
    by_method = _index_rows(rows)
    methods = sorted(by_method, key=_method_sort_key)
    summaries = []
    for method in methods:
        method_rows = list(by_method[method].values())
        failed_sets = [set(filter(None, row["failed_checks"].split(";"))) for row in method_rows]
        summaries.append(
            LeaveOneGateOutSummary(
                method=method,
                case_count=len(method_rows),
                pass_count=sum(row["gate_pass"] == "yes" for row in method_rows),
                pass_count_without={
                    gate: sum(not (failures - {gate}) for failures in failed_sets)
                    for gate in FAILURE_ORDER
                },
            )
        )
    return tuple(summaries)


def summarize_gate_results(csv_path: Path) -> tuple[MethodGateSummary, ...]:
    """Read per-case rows and count passes and failure flags for each method."""
    rows = _read_rows(csv_path, {"method", "case", "gate_pass", "failed_checks"})
    by_method = _index_rows(rows)
    summaries = []
    for method in sorted(by_method, key=_method_sort_key):
        method_rows = list(by_method[method].values())
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
    effects: tuple[PairedEffect, ...],
    gate_sensitivity: tuple[LeaveOneGateOutSummary, ...],
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
            f"saturation-failure count was {saturation_failures}. Peak timing and "
            "controller-state telemetry are needed to distinguish entry and later "
            "in-contact failures."
        )
    else:
        lines.append("This input contains no torque-residual policy rows.")

    lines.extend(
        [
            "",
            "## Paired residual effect",
            "",
            (
                "Each difference is residual minus torque-safe adaptive on the same case; "
                "negative values favor the residual policy. The interval is a deterministic "
                f"{BOOTSTRAP_SAMPLES:,}-sample percentile bootstrap of the paired median. "
                "Win/tie/loss uses the stored values without a tolerance."
            ),
            "These are post-reveal descriptive estimates, not confirmatory intervals.",
            "",
            "| Policy | Metric | Median difference | Bootstrap 95% CI | Win/tie/loss |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for effect in effects:
        lines.append(
            f"| {_display_name(effect.method)} | {PAIRED_METRIC_LABELS[effect.metric]} | "
            f"{effect.median_delta:.3f} | {effect.ci_low:.3f} to {effect.ci_high:.3f} | "
            f"{effect.win_count}/{effect.tie_count}/{effect.loss_count} |"
        )

    lines.extend(
        [
            "",
            "## Exploratory gate sensitivity (post-hoc)",
            "",
            (
                "Each column recounts passes after omitting only the named gate and keeping "
                "the other four. This was computed after reveal and does not alter the frozen "
                "primary result."
            ),
            "",
            "| Method | Original | No force RMSE | No contact ratio | No peak force | No tangent RMSE | No saturation |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sensitivity in gate_sensitivity:
        omitted = sensitivity.pass_count_without
        lines.append(
            f"| {_display_name(sensitivity.method)} | {sensitivity.pass_count} | "
            f"{omitted['force_rmse']} | {omitted['contact_ratio']} | "
            f"{omitted['peak_force']} | {omitted['tangent_rmse']} | "
            f"{omitted['saturation']} |"
        )
    residual_sensitivity = [
        summary
        for summary in gate_sensitivity
        if summary.method.startswith("torque_residual_run_")
    ]
    if residual_sensitivity:
        without_peak = [
            summary.pass_count_without["peak_force"] for summary in residual_sensitivity
        ]
        threshold_comparison = (
            f"still below the frozen {required_passes}/{case_count} threshold"
            if max(without_peak) < required_passes
            else f"reaching the frozen {required_passes}/{case_count} threshold"
        )
        lines.extend(
            [
                "",
                (
                    "Removing only the peak-force gate raises the residual-policy counts to "
                    f"{min(without_peak)}–{max(without_peak)}/{case_count}, "
                    f"{threshold_comparison}. The raw-peak gate is the largest single source "
                    "of gate failures; event timing is needed before assigning a physical cause."
                ),
            ]
        )
    lines.extend(["", f"Source: [`{source_link}`]({source_link})."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(
    summaries: tuple[MethodGateSummary, ...],
    gate_sensitivity: tuple[LeaveOneGateOutSummary, ...],
    required_passes: int,
    path: Path,
) -> None:
    labels = [_display_name(summary.method) for summary in summaries]
    pass_counts = [summary.pass_count for summary in summaries]
    y = np.arange(len(summaries))
    method_colors = ["#6c757d", "#277da1", "#577590"] + ["#e76f51"] * max(0, len(summaries) - 3)

    fig, (pass_axis, failure_axis, sensitivity_axis) = plt.subplots(
        1,
        3,
        figsize=(19.0, 6.2),
        gridspec_kw={"width_ratios": (1.0, 1.15, 1.2)},
    )
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
    pass_axis.set_title("Frozen primary gate")
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

    sensitivity_by_method = {summary.method: summary for summary in gate_sensitivity}
    sensitivity_values = np.asarray(
        [
            [sensitivity_by_method[summary.method].pass_count_without[gate] for gate in FAILURE_ORDER]
            for summary in summaries
        ],
        dtype=float,
    )
    sensitivity_axis.imshow(
        sensitivity_values,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=case_count,
    )
    sensitivity_axis.set_xticks(
        np.arange(len(FAILURE_ORDER)),
        ("force\nRMSE", "contact\nratio", "peak\nforce", "tangent\nRMSE", "saturation"),
        rotation=25,
        ha="right",
    )
    sensitivity_axis.set_yticks(y, labels)
    sensitivity_axis.set_title("Post-hoc: omit one gate")
    sensitivity_axis.set_xlabel("passing cases; other four gates retained")
    for row_index, row in enumerate(sensitivity_values):
        for column_index, value in enumerate(row):
            sensitivity_axis.text(
                column_index,
                row_index,
                f"{int(value)}",
                ha="center",
                va="center",
                color="white" if value > 0.55 * case_count else "black",
                fontsize=8,
            )

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
    effects = compute_paired_effects(csv_path)
    gate_sensitivity = summarize_leave_one_gate_out(csv_path)
    case_count = summaries[0].case_count
    if not 0 < required_passes <= case_count:
        raise ValueError("required passes must be between one and the case count")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_link = Path(os.path.relpath(csv_path.resolve(), resolved_output)).as_posix()
    _write_summary(
        summaries,
        effects,
        gate_sensitivity,
        required_passes,
        source_link,
        output_dir / "summary.md",
    )
    plot_path = output_dir / "failure_analysis.png"
    _plot(summaries, gate_sensitivity, required_passes, plot_path)
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
