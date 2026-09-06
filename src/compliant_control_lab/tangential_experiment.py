"""Fixed classical tangential-control comparison on public smooth-contact cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, replace
from importlib.metadata import version
from operator import index as integer_index
from pathlib import Path

import numpy as np

from compliant_control_lab.reference_ablation import validate_output_path
from compliant_control_lab.surface_contact_validation import _geometry_gap, _path_label
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame
from compliant_control_lab.surface_experiment import (
    ARMS as SURFACE_ARMS,
)
from compliant_control_lab.surface_experiment import (
    CONTACT_METRICS,
    METRICS,
    _constructor_config,
    _metrics,
    _output_path,
    _sha256,
    _source_hashes,
    _write_csv,
    development_cases,
)
from compliant_control_lab.surface_replay import replay_surface_trace, save_surface_trace
from compliant_control_lab.surface_simulation import run_surface_trial

METHODS = {
    "baseline": "surface_adaptive",
    "integral": "surface_integral",
    "friction": "surface_friction",
}
AUXILIARY_METRICS = (
    "max_penetration_mm",
    "evaluation_median_penetration_mm",
    "projection_pct",
    "max_compensation_force_n",
)
CHECKS = {
    "maximum_main_median_tangent_ratio": 0.5,
    "minimum_contact_ratio_pct": 99.0,
    "maximum_raw_peak_force_n": 35.0,
    "required_saturation_pct": 0.0,
    "maximum_paired_force_rmse_increase_n": 0.2,
}


def _load_baseline(directory: Path) -> tuple[dict, dict, dict]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (directory / "COMPLETE").read_text().strip() != _sha256(manifest_path):
        raise ValueError("baseline COMPLETE does not match manifest")
    if manifest["experiment_identity"] != "surface-contact-model-repair-v1":
        raise ValueError("expected the published smooth-contact repair baseline")
    for name, digest in manifest["artifact_sha256"].items():
        if Path(name).name != name or _sha256(directory / name) != digest:
            raise ValueError(f"baseline artifact hash mismatch: {name}")
    if "comparison.csv" not in manifest["artifact_sha256"]:
        raise ValueError("baseline manifest must cover comparison.csv")
    with (directory / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(p, a, i) for p in ("legacy", "smooth") for a in SURFACE_ARMS for i in range(24)}
    if (
        len(rows) != 192
        or {(r["profile"], r["arm"], int(r["case_index"])) for r in rows} != expected
    ):
        raise ValueError("baseline must be the complete unique 192-row repair grid")
    selected = {
        int(r["case_index"]): r
        for r in rows
        if (r["profile"], r["arm"]) == ("smooth", "surface_exact")
    }
    configs = {
        r["case_index"]: r
        for r in manifest["case_configurations"]
        if (r["profile"], r["arm"]) == ("smooth", "surface_exact")
    }
    identity = {
        "directory": _path_label(directory),
        "manifest_sha256": _sha256(manifest_path),
        "complete_sha256": _sha256(directory / "COMPLETE"),
        "artifact_sha256": manifest["artifact_sha256"],
    }
    return selected, configs, identity


def _verify_baseline(row: dict, frozen: dict) -> float:
    if (
        row["scenario"] != frozen["scenario"]
        or row["simulation_seed"] != int(frozen["simulation_seed"])
        or row["has_raw_contact"] != (frozen["has_raw_contact"] == "True")
    ):
        raise ValueError("baseline case identity or contact status differs")
    errors = []
    for name in METRICS:
        expected = float(frozen[name]) if frozen[name] else None
        actual = row[name]
        error = (
            0.0
            if actual is expected
            else (
                abs(actual - expected)
                if actual is not None and expected is not None
                else float("inf")
            )
        )
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError(f"baseline metric mismatch: {name}, error={error}")
        errors.append(error)
    return max(errors)


def _auxiliary_metrics(result, case: dict) -> dict:
    trace = result.trace
    gap = _geometry_gap(trace, case["scenario"].wall_yaw_deg)
    if not np.allclose(gap, trace["true_contact_gap_m"], rtol=0, atol=1e-12):
        raise ValueError("geometric gap disagrees with true wall-plane reconstruction")
    correction = np.asarray(trace["requested_tangential_force_world"])
    projection = np.asarray(trace["torque_projection_scale"])
    if (
        correction.shape != (len(gap), 3)
        or projection.shape != gap.shape
        or not all(np.all(np.isfinite(x)) for x in (gap, correction, projection))
    ):
        raise ValueError("invalid compensation or geometry telemetry")
    window = trace["time"] >= case["config"].evaluation_start
    if not np.any(window):
        raise ValueError("trial must observe the evaluation window")
    penetration = np.maximum(-gap, 0) * 1000
    return {
        "max_penetration_mm": float(np.max(penetration)),
        "evaluation_median_penetration_mm": float(np.median(penetration[window])),
        "projection_pct": float(100 * np.mean(projection < 1 - 1e-12)),
        "max_compensation_force_n": float(np.max(np.linalg.norm(correction, axis=1))),
    }


def case_failures(row: dict, baseline: dict) -> tuple[str, ...]:
    return tuple(
        name
        for passed, name in (
            (row["has_raw_contact"] and row["contact_ratio_pct"] >= 99, "contact_ratio"),
            (row["peak_force_n"] <= 35, "raw_peak_force"),
            (row["saturation_pct"] == 0, "saturation"),
            (row["force_rmse_n"] - baseline["force_rmse_n"] <= 0.2, "paired_force_rmse"),
        )
        if not passed
    )


def paired_deltas(rows: list[dict]) -> list[dict]:
    baseline = {
        (r["case_index"], r["true_sliding_friction"]): r for r in rows if r["method"] == "baseline"
    }
    pairs = []
    for row in rows:
        if row["method"] == "baseline":
            continue
        reference = baseline[row["case_index"], row["true_sliding_friction"]]
        contact = row["has_raw_contact"] and reference["has_raw_contact"]
        pairs.append(
            {
                "method": row["method"],
                "case_index": row["case_index"],
                "true_sliding_friction": row["true_sliding_friction"],
                "contact_in_both_methods": contact,
                **{
                    name: row[name] - reference[name]
                    if row[name] is not None
                    and reference[name] is not None
                    and (name not in CONTACT_METRICS or contact)
                    else None
                    for name in (*METRICS, *AUXILIARY_METRICS)
                },
            }
        )
    return pairs


def main_outcomes(rows: list[dict]) -> dict:
    baseline = float(np.median([r["tangent_rmse_mm"] for r in rows if r["method"] == "baseline"]))
    outcomes = {}
    for method in tuple(METHODS)[1:]:
        selected = [r for r in rows if r["method"] == method]
        median = float(np.median([r["tangent_rmse_mm"] for r in selected]))
        outcomes[method] = {
            "median_tangent_ratio": median / baseline if baseline > 0 else None,
            "tangent_criterion_pass": median <= 0.5 * baseline,
            "all_case_checks_pass": all(r["case_checks_pass"] == "yes" for r in selected),
        }
        outcomes[method]["overall_pass"] = (
            outcomes[method]["tangent_criterion_pass"] and outcomes[method]["all_case_checks_pass"]
        )
    return outcomes


def render_summary(rows: list[dict], long_rows: list[dict], scope: str) -> str:
    lines = [
        "# Classical tangential-compensation comparison",
        "",
        scope,
        "Public development data, NOT a new holdout. The smooth contact model is fixed;",
        "this does not identify physical contact parameters or establish real-robot safety.",
        "No tuning: integral gain 800, correction cap 6 N, nominal friction 0.45, velocity scale 0.005 m/s.",
        "Engineering criteria: main median tangent RMSE <= half baseline; every case contact >=99%,",
        "raw peak <=35 N, saturation exactly 0%, and paired force-RMSE increase <=0.2 N.",
        "These are development checks, not preregistered holdout gates. Failures are retained.",
        "Force/tangent RMSE and contact use evaluation_start; peak, projection and saturation use the full trial.",
        "Penetration is max(0, -signed gap); maximum uses the full trial, median uses evaluation_start.",
        "Requested correction is before torque projection; input replay is controller mathematics, not plant replay.",
        "",
        "## Main fixed 24-case grid",
        "",
        "| Method | Cases | Tangent median [mm] | Force RMSE median [N] | Peak max [N] | Contact min [%] | Sat max [%] | Penetration max [mm] | Projection max [%] | Case passes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        values = [
            np.percentile([r[key] for r in selected], q)
            for key, q in (
                ("tangent_rmse_mm", 50),
                ("force_rmse_n", 50),
                ("peak_force_n", 100),
                ("contact_ratio_pct", 0),
                ("saturation_pct", 100),
                ("max_penetration_mm", 100),
                ("projection_pct", 100),
            )
        ]
        lines.append(
            f"| {method} | {len(selected)} | "
            + " | ".join(f"{v:.6g}" for v in values)
            + f" | {sum(r['case_checks_pass'] == 'yes' for r in selected)}/{len(selected)} |"
        )
    for method, outcome in main_outcomes(rows).items():
        ratio = outcome["median_tangent_ratio"]
        ratio_label = f"{ratio:.6g}" if ratio is not None else "NA"
        lines.append(
            f"\n{method}: tangent median ratio={ratio_label}; tangent criterion={outcome['tangent_criterion_pass']}; all case checks={outcome['all_case_checks_pass']}; overall={outcome['overall_pass']}."
        )
    lines += [
        "",
        "## Separate long-duration friction-mismatch diagnostic",
        "",
        "Original case 16, duration 12 s; both actual tool/wall friction coefficients change together.",
        "Controller nominal friction remains 0.45. These 9 runs are NOT pooled with the main grid.",
        "| True friction | Method | Tangent [mm] | Force RMSE [N] | Contact [%] | Peak [N] | Sat [%] | Penetration max [mm] | Failed checks |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in long_rows:
        values = [
            row[key]
            for key in (
                "tangent_rmse_mm",
                "force_rmse_n",
                "contact_ratio_pct",
                "peak_force_n",
                "saturation_pct",
                "max_penetration_mm",
            )
        ]
        lines.append(
            f"| {row['true_sliding_friction']:.2f} | {row['method']} | "
            + " | ".join(f"{v:.6g}" for v in values)
            + f" | {row['failed_case_checks'] or 'none'} |"
        )
    if not long_rows:
        lines.append("Long diagnostic omitted for this explicitly requested subset smoke run.")
    lines += [
        "",
        "Both paired CSVs use method minus matching baseline; do not reward a missing-contact peak.",
        "Only main case 16 (three methods) and long case 16 / friction 0.45 / friction method retain full traces.",
        "Other rows retain configurations and metrics only. The representative image is not an aggregate result.",
        "",
    ]
    return "\n".join(lines)


def _plot_representative(path: Path, traces: dict[str, dict], yaw: float) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 8))
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1, sharex=True)
    normal = np.array([np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw)), 0])
    for method, trace in traces.items():
        time = trace["time"]
        axes[0].plot(time, trace["true_normal_force"], label=method, linewidth=0.8)
        error = trace["position"] - trace["target_position"]
        tangent = error - np.outer(error @ normal, normal)
        window = time >= 1.5
        axes[1].plot(time[window], 1000 * np.linalg.norm(tangent[window], axis=1), linewidth=0.8)
        axes[2].plot(
            time, np.linalg.norm(trace["requested_tangential_force_world"], axis=1), linewidth=0.8
        )
    for axis, label in zip(
        axes, ("Raw normal force [N]", "Tangent error [mm]", "Requested correction [N]")
    ):
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[2].set_xlabel("Time [s]")
    figure.suptitle("Preselected main case 16: representative, not aggregate performance")
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def generate_tangential_experiment(
    output_dir: Path | str,
    *,
    baseline_dir: Path | str = Path("results/franka_surface_contact_fix"),
    case_indices: Iterable[int] | None = None,
    skip_long: bool = False,
) -> Path:
    baseline_dir = Path(baseline_dir)
    output = validate_output_path(_output_path(output_dir), baseline_dir)
    frozen, frozen_configs, identity = _load_baseline(baseline_dir)
    indices = list(range(24)) if case_indices is None else list(case_indices)
    if any(isinstance(i, (bool, np.bool_)) for i in indices):
        raise TypeError("case indices must be integers, not booleans")
    indices = sorted(integer_index(i) for i in indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("case indices must be nonempty and unique")
    if indices[0] < 0 or indices[-1] >= 24:
        raise IndexError("case index outside fixed development grid")
    if skip_long and len(indices) == 24:
        raise ValueError("the full report requires the long-duration diagnostic")
    cases = development_cases()
    for index in indices:
        cases[index] = {
            **cases[index],
            "config": replace(cases[index]["config"], contact_model="smooth"),
        }
        expected = frozen_configs[index]
        for key in ("scenario", "config", "task"):
            recorded = dict(expected[key])
            if key == "scenario":
                recorded.setdefault("tool_sliding_friction", 0.45)
            if json.loads(json.dumps(asdict(cases[index][key]))) != recorded:
                raise ValueError(f"baseline constructor configuration mismatch: {index}/{key}")
    jobs = [("main", method, cases[index]) for method in METHODS for index in indices]
    if not skip_long:
        original = development_cases()[16]
        for mu in (0.25, 0.45, 0.65):
            case = {
                **original,
                "config": replace(original["config"], contact_model="smooth", duration=12.0),
                "scenario": replace(
                    original["scenario"], wall_sliding_friction=mu, tool_sliding_friction=mu
                ),
            }
            jobs.extend(("long", method, case) for method in METHODS)
    scope = f"{'Full public grid' if len(indices) == 24 else 'Public SUBSET'}: {len(indices)}/24 cases x 3 methods; selected indices={indices}."
    sources = _source_hashes()
    rows, long_rows, configurations, replay_checks, traces = [], [], [], {}, {}
    baselines = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".tangential-experiment-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        for count, (category, method, case) in enumerate(jobs, 1):
            scenario, config, task = (case[key] for key in ("scenario", "config", "task"))
            angle = np.deg2rad(task.yaw_deg)
            frame = SurfaceFrame.from_normal(np.array([np.cos(angle), np.sin(angle), 0]))
            result = run_surface_trial(
                frame, scenario=scenario, config=config, task=task, controller_kind=METHODS[method]
            )
            row = {
                "category": category,
                "method": method,
                "case_index": case["case_index"],
                "scenario": scenario.name,
                "simulation_seed": config.seed,
                "duration_s": config.duration,
                "controller_yaw_deg": task.yaw_deg,
                "wall_yaw_deg": scenario.wall_yaw_deg,
                "wall_time_constant_s": scenario.wall_time_constant,
                "tool_mass_kg": scenario.tool_mass_kg,
                "true_sliding_friction": scenario.wall_sliding_friction,
                **_metrics(result),
                **_auxiliary_metrics(result, case),
                "baseline_metric_max_abs_error": None,
            }
            key = category, case["case_index"], scenario.wall_sliding_friction
            if method == "baseline":
                if category == "main":
                    row["baseline_metric_max_abs_error"] = _verify_baseline(
                        row, frozen[case["case_index"]]
                    )
                baselines[key] = row
            failures = case_failures(row, baselines[key])
            row.update(
                case_checks_pass="no" if failures else "yes", failed_case_checks=";".join(failures)
            )
            (rows if category == "main" else long_rows).append(row)
            configurations.append(
                {
                    "category": category,
                    "method": method,
                    "case_index": case["case_index"],
                    "controller_kind": METHODS[method],
                    "controller_frame_rotation": frame.rotation.tolist(),
                    **{name: asdict(case[name]) for name in ("scenario", "config", "task")},
                }
            )
            save_name = None
            if category == "main" and case["case_index"] == 16:
                traces[method] = result.trace
                save_name = f"representative_case_16_{method}.npz"
            elif (
                category == "long"
                and scenario.wall_sliding_friction == 0.45
                and method == "friction"
            ):
                save_name = "long_case_16_mu_0.45_friction.npz"
            if save_name:
                save_surface_trace(staging / save_name, result.trace)
                replay = replay_surface_trace(staging / save_name)
                if not replay.matches:
                    raise ValueError(f"controller input replay mismatch: {save_name}")
                replay_checks[save_name] = asdict(replay)
            print(
                f"completed {count}/{len(jobs)}: {category} {method} case {case['case_index']} mu={scenario.wall_sliding_friction}",
                flush=True,
            )
        _write_csv(staging / "comparison.csv", rows)
        _write_csv(staging / "paired_deltas.csv", paired_deltas(rows))
        if long_rows:
            _write_csv(staging / "long_comparison.csv", long_rows)
            _write_csv(staging / "long_paired_deltas.csv", paired_deltas(long_rows))
        (staging / "summary.md").write_text(
            render_summary(rows, long_rows, scope), encoding="utf-8"
        )
        if traces:
            _plot_representative(
                staging / "representative_case_16.png", traces, cases[16]["task"].yaw_deg
            )
        if _source_hashes() != sources or _load_baseline(baseline_dir)[2] != identity:
            raise ValueError("source/assets or baseline archive changed during experiment")
        parameters = {
            method: _constructor_config(
                SurfaceAdaptiveController(
                    SurfaceFrame(np.eye(3)),
                    tangential_mode=None if method == "baseline" else method,
                )._base
            )
            for method in METHODS
        }
        manifest = {
            "schema_version": 1,
            "experiment_identity": "tangential-compensation-public24-v1",
            "long_experiment_identity": "tangential-compensation-long-friction-v1"
            if long_rows
            else None,
            "scope": scope,
            "new_holdout": False,
            "is_subset": len(indices) != 24,
            "selected_case_indices": indices,
            "methods": METHODS,
            "engineering_checks": CHECKS,
            "main_outcomes": main_outcomes(rows),
            "case_configurations": configurations,
            "controller_parameters": parameters,
            "baseline_archive": identity,
            "baseline_trace_reference": {
                "path": _path_label(
                    baseline_dir / "representative_case_16_smooth_surface_exact.npz"
                ),
                "sha256": identity["artifact_sha256"][
                    "representative_case_16_smooth_surface_exact.npz"
                ],
            },
            "replay_checks": replay_checks,
            "source_and_assets_sha256": sources,
            "versions": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "mujoco", "compliant-control-lab")},
            },
            "artifact_sha256": {path.name: _sha256(path) for path in sorted(staging.iterdir())},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        (staging / "COMPLETE").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="utf-8"
        )
        validate_output_path(_output_path(output), baseline_dir)
        os.rename(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("results/franka_surface_contact_fix"))
    parser.add_argument("--case-indices", type=int, nargs="+")
    parser.add_argument("--skip-long", action="store_true")
    args = parser.parse_args()
    print(
        generate_tangential_experiment(
            args.output,
            baseline_dir=args.baseline,
            case_indices=args.case_indices,
            skip_long=args.skip_long,
        )
    )


if __name__ == "__main__":
    main()
