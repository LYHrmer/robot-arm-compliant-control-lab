"""Fixed paired development study for the new surface task and wrench-sensor model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, fields, is_dataclass
from importlib.metadata import version
from itertools import product
from operator import index as integer_index
from pathlib import Path

import numpy as np

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.reference_ablation import validate_output_path
from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_replay import replay_surface_trace, save_surface_trace
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    SurfaceTask,
    run_surface_trial,
)

ARMS = {
    "world_safe_adaptive": ("world_safe_adaptive", None),
    "surface_exact": ("surface_adaptive", 0.0),
    "surface_minus5": ("surface_adaptive", -5.0),
    "surface_plus5": ("surface_adaptive", 5.0),
}
METRICS = (
    "force_rmse_n",
    "peak_force_n",
    "tangent_rmse_mm",
    "contact_ratio_pct",
    "saturation_pct",
    "orientation_rmse_deg",
    "measurement_rmse_n",
    "seconds_over_35_n",
    "first_raw_contact_time_s",
)
CONTACT_METRICS = {"peak_force_n", "seconds_over_35_n", "first_raw_contact_time_s"}
REPRESENTATIVE_CASE_INDEX = 16


def development_cases() -> list[dict]:
    """Return the fixed 24 cases in yaw, wall time, tool mass, noise-seed order."""
    return [
        {
            "case_index": index,
            "scenario": SurfaceScenario(
                name=f"surface_dev_{index:02d}",
                wall_yaw_deg=yaw,
                wall_time_constant=wall_time,
                tool_mass_kg=mass,
            ),
            "config": SurfaceSimulationConfig(seed=seed),
            "task": SurfaceTask(yaw_deg=yaw),
        }
        for index, (yaw, wall_time, mass, seed) in enumerate(
            product((-15.0, 0.0, 15.0), (0.005, 0.012), (0.10, 0.13), (11, 29))
        )
    ]


def _constructor_config(value):
    if is_dataclass(value):
        return {
            field.name: _constructor_config(getattr(value, field.name))
            for field in fields(value)
            if field.init
        }
    return value.tolist() if isinstance(value, np.ndarray) else value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    paths = sorted(set(package.rglob("*.py")) | set((package / "assets").rglob("*")))
    return {str(path.relative_to(package)): _sha256(path) for path in paths if path.is_file()}


def _output_path(output_dir: Path | str) -> Path:
    output = validate_output_path(output_dir)
    results = Path(__file__).resolve().parents[2] / "results"
    archives = (
        tuple(path for path in results.iterdir() if path.is_dir() and path.resolve() != output)
        if results.exists()
        else ()
    )
    return validate_output_path(output, *archives)


def _metrics(result) -> dict:
    supplied = result.metrics()
    if not isinstance(supplied.get("has_raw_contact"), (bool, np.bool_)):
        raise TypeError("has_raw_contact must be boolean")
    values = {"has_raw_contact": bool(supplied["has_raw_contact"])}
    for name in METRICS:
        value = supplied[name]
        if name == "first_raw_contact_time_s" and value is None:
            values[name] = None
        elif isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        else:
            values[name] = float(value)
    if values["has_raw_contact"] != (values["first_raw_contact_time_s"] is not None):
        raise ValueError("first contact time must be present exactly when raw contact exists")
    return values


def paired_deltas(rows: list[dict]) -> list[dict]:
    """Pair by case; contact-dependent differences are absent if either arm misses contact."""
    baseline = {row["case_index"]: row for row in rows if row["arm"] == "world_safe_adaptive"}
    pairs = []
    for row in rows:
        if row["arm"] == "world_safe_adaptive":
            continue
        reference = baseline[row["case_index"]]
        contact = row["has_raw_contact"] and reference["has_raw_contact"]
        pairs.append(
            {
                "arm": row["arm"],
                "case_index": row["case_index"],
                "contact_in_both_arms": contact,
                **{
                    name: row[name] - reference[name]
                    if row[name] is not None
                    and reference[name] is not None
                    and (name not in CONTACT_METRICS or contact)
                    else None
                    for name in METRICS
                },
            }
        )
    return pairs


def render_summary(rows: list[dict], pairs: list[dict], scope: str, traces: list[str]) -> str:
    lines = [
        "# Surface-task development study",
        "",
        scope,
        "NEW task and sensor/noise definitions: neither the old 24 holdout cases nor the 48 blind cases.",
        "This is public development validation, NOT a new holdout. No fitting or tuning occurs.",
        "All arms share the task, physical case, seed and sensor bias/noise definitions.",
        "The task yaw is supplied explicitly; controller frames use yaw, yaw−5°, yaw+5°, or world x.",
        "True wall normals are restricted to physics/scoring. No pass/fail gate is defined here.",
        "surface_exact assumes ideal frame calibration. An accurate fixture description supplies",
        "the task plane; the study does not test finding an unknown surface.",
        "Payload mass changes are known to the robot dynamics model. Nominal sensor gravity",
        "compensation can be mismatched, but its residual lies along world z and is not consumed",
        "by these horizontal normal-force projections. This is not an unknown-payload test.",
        "All six wrench channels are recorded; only the normal force drives force control.",
        "Moment channels are diagnostic. Measurement RMSE mixes sensor error, filtering and timing lag.",
        "",
        "Force RMSE uses current raw true force after evaluation_start; peak uses the full trial.",
        "Tangent RMSE is the Euclidean position-error norm projected onto the true wall plane.",
        "Measurement RMSE compares the actual scalar controller feedback to current true force.",
        "Contact ratio uses raw true force > 0.5 N after evaluation_start. Saturation uses all steps.",
        "Seconds over 35 N = count(raw true force > 35 N) × timestep.",
        "",
        "| Arm | Cases | Cases with raw contact | Median contact ratio [%] |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        lines.append(
            f"| {arm} | {len(selected)} | {sum(row['has_raw_contact'] for row in selected)} | "
            f"{np.median([row['contact_ratio_pct'] for row in selected]):.6g} |"
        )
    lines += [
        "",
        "## Absolute metrics across executed cases",
        "",
        "| Arm | Median force RMSE [N] | Peak P95 [N] | Median tangent RMSE [mm] | Median contact [%] | Worst saturation [%] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        values = [
            np.percentile([row[name] for row in selected], percentile)
            for name, percentile in (
                ("force_rmse_n", 50),
                ("peak_force_n", 95),
                ("tangent_rmse_mm", 50),
                ("contact_ratio_pct", 50),
                ("saturation_pct", 100),
            )
        ]
        lines.append(f"| {arm} | " + " | ".join(f"{value:.6g}" for value in values) + " |")
    lines += [
        "",
        "## Paired differences from world_safe_adaptive",
        "",
        "Differences are arm − baseline. Missing-contact peak/time differences are NA;",
        "contact counts and tracking costs must accompany any peak comparison.",
        "",
        "| Arm | Metric | Paired n | Median difference |",
        "|---|---|---:|---:|",
    ]
    for arm in tuple(ARMS)[1:]:
        for metric in METRICS:
            values = [row[metric] for row in pairs if row["arm"] == arm and row[metric] is not None]
            median = f"{np.median(values):.6g}" if values else "NA"
            lines.append(f"| {arm} | {metric} | {len(values)} | {median} |")
    lines += [
        "",
        f"Full canonical traces retained: {', '.join(traces) if traces else 'none in this subset'}.",
        "Only representative case 16 retains full traces; other cases retain full parameters",
        "and summary metrics, not full traces. comparison.csv includes every executed arm/case.",
        "The representative case was fixed before execution: yaw +15°, wall time 0.005 s,",
        "tool mass 0.10 kg, noise seed 11. Existing v0.5 archives and conclusions are unchanged.",
        "",
    ]
    return "\n".join(lines)


def _plot_representative(
    path: Path, traces: dict[str, dict], wall_yaw_deg: float, target_force: float
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 7))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)
    angle = np.deg2rad(wall_yaw_deg)
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    for arm, trace in traces.items():
        (line,) = axes[0].plot(trace["time"], trace["true_normal_force"], label=arm, linewidth=1.0)
        error = trace["position"] - trace["target_position"]
        tangent = error - np.outer(error @ normal, normal)
        axes[1].plot(
            trace["time"],
            1000 * np.linalg.norm(tangent, axis=1),
            color=line.get_color(),
            linewidth=1.0,
        )
    axes[0].axhline(
        target_force, color="0.35", linestyle="--", linewidth=1, label="12 N final target"
    )
    axes[0].axhline(35.0, color="0.55", linestyle=":", linewidth=1, label="35 N reference")
    axes[0].set_ylabel("Raw true contact force [N]")
    axes[1].set_ylabel("True wall-tangent error [mm]")
    axes[1].set_xlabel("Control time [s]")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    figure.text(
        0.5,
        0.015,
        "Preselected case 16: yaw +15°, wall time 0.005 s, tool 0.10 kg, seed 11; not overall performance.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.90))
    figure.savefig(path, dpi=150)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_surface_experiment(
    output_dir: Path | str,
    *,
    case_indices: Iterable[int] | None = None,
) -> Path:
    """Run the fixed paired development grid and atomically publish complete artifacts."""
    output = _output_path(output_dir)
    cases = development_cases()
    indices = list(range(len(cases))) if case_indices is None else list(case_indices)
    if any(isinstance(item, (bool, np.bool_)) for item in indices):
        raise TypeError("case indices must be integers, not booleans")
    indices = sorted(integer_index(item) for item in indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("case indices must be nonempty and unique")
    if indices[0] < 0 or indices[-1] >= len(cases):
        raise IndexError("case index outside the new development grid")
    scope = (
        f"Full development grid: {len(cases)} cases, {len(cases) * len(ARMS)} rows."
        if len(indices) == len(cases)
        else f"Development SUBSET: {len(indices)}/{len(cases)} cases; indices={indices}."
    )
    source_hashes = _source_hashes()
    rows, traces = [], []
    representative, replay_checks = {}, {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-experiment-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        for arm, (kind, offset) in ARMS.items():
            for index in indices:
                case = cases[index]
                yaw = 0.0 if offset is None else case["task"].yaw_deg + offset
                radians = np.deg2rad(yaw)
                frame = SurfaceFrame.from_normal(np.array([np.cos(radians), np.sin(radians), 0.0]))
                result = run_surface_trial(
                    frame,
                    scenario=case["scenario"],
                    config=case["config"],
                    task=case["task"],
                    controller_kind=kind,
                )
                rows.append(
                    {
                        "arm": arm,
                        "case_index": index,
                        "scenario": case["scenario"].name,
                        "simulation_seed": case["config"].seed,
                        "controller_yaw_deg": yaw,
                        **_metrics(result),
                    }
                )
                if index == REPRESENTATIVE_CASE_INDEX:
                    filename = f"representative_case_16_{arm}.npz"
                    save_surface_trace(staging / filename, result.trace)
                    replay = replay_surface_trace(staging / filename)
                    if not replay.matches:
                        raise ValueError(f"representative replay mismatch: {arm}")
                    replay_checks[arm] = asdict(replay)
                    representative[arm] = result.trace
                    traces.append(filename)
                print(
                    f"completed {len(rows)}/{len(indices) * len(ARMS)}: {arm} case {index}",
                    flush=True,
                )
        if _source_hashes() != source_hashes:
            raise ValueError(
                "source/assets changed during experiment; refusing mixed-version report"
            )
        pairs = paired_deltas(rows)
        _write_csv(staging / "comparison.csv", rows)
        _write_csv(staging / "paired_deltas.csv", pairs)
        (staging / "summary.md").write_text(
            render_summary(rows, pairs, scope, traces), encoding="utf-8"
        )
        if representative:
            case = cases[REPRESENTATIVE_CASE_INDEX]
            _plot_representative(
                staging / "representative_case_16.png",
                representative,
                case["scenario"].wall_yaw_deg,
                case["config"].target_force,
            )
        manifest = {
            "schema_version": 1,
            "experiment_identity": "surface-task-development-v1",
            "scope": scope,
            "new_holdout": False,
            "selected_case_indices": indices,
            "is_subset": len(indices) != len(cases),
            "arms": {
                arm: {"controller_kind": kind, "yaw_offset_deg": offset}
                for arm, (kind, offset) in ARMS.items()
            },
            "local_safe_controller_parameters": _constructor_config(FrankaSafeAdaptiveController()),
            "all_development_cases": [
                {
                    "case_index": case["case_index"],
                    **{name: asdict(case[name]) for name in ("scenario", "config", "task")},
                }
                for case in cases
            ],
            "versions": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "mujoco", "compliant-control-lab")},
            },
            "source_and_assets_sha256": source_hashes,
            "full_trace_case_indices": [REPRESENTATIVE_CASE_INDEX] if traces else [],
            "replay_checks": replay_checks,
            "artifact_sha256": {path.name: _sha256(path) for path in sorted(staging.iterdir())},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        (staging / "COMPLETE").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="utf-8"
        )
        # Existing archives were checked before creating staging or new parents;
        # recheck target/symlinks without treating our new parent as an archive.
        validate_output_path(output)
        os.rename(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-indices", type=int, nargs="+")
    arguments = parser.parse_args()
    output = generate_surface_experiment(arguments.output, case_indices=arguments.case_indices)
    print(f"Surface development report complete (NOT a new holdout): {output}")


if __name__ == "__main__":
    main()
