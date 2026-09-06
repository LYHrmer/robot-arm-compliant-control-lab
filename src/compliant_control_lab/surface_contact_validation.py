"""Paired public-development validation of a smooth contact-model repair."""

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

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.reference_ablation import validate_output_path
from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_experiment import (
    ARMS,
    CONTACT_METRICS,
    METRICS,
    REPRESENTATIVE_CASE_INDEX,
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

PROFILES = ("legacy", "smooth")
AUXILIARY_METRICS = (
    "max_geometric_separation_um",
    "mean_true_tangent_force_n",
    "median_true_friction_ratio",
)
REPAIR_CHECKS = {
    "minimum_contact_ratio_pct": 99.0,
    "maximum_raw_force_rmse_n": 2.0,
    "maximum_raw_peak_force_n": 35.0,
    "required_saturation_pct": 0.0,
}


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    repository = Path(__file__).resolve().parents[2]
    return (
        str(resolved.relative_to(repository))
        if resolved.is_relative_to(repository)
        else str(resolved)
    )


def _load_baseline(directory: Path) -> tuple[dict, dict]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (directory / "COMPLETE").read_text().strip() != _sha256(manifest_path):
        raise ValueError("baseline COMPLETE does not match its manifest")
    hashes = manifest["artifact_sha256"]
    for name, digest in hashes.items():
        if Path(name).name != name or _sha256(directory / name) != digest:
            raise ValueError(f"baseline artifact hash mismatch: {name}")
    if "comparison.csv" not in hashes:
        raise ValueError("baseline manifest must cover comparison.csv")
    with (directory / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["arm"], int(row["case_index"])): row for row in rows}
    expected = {(arm, index) for arm in ARMS for index in range(24)}
    if len(rows) != 96 or set(indexed) != expected:
        raise ValueError("baseline must contain exactly the complete unique 24 x 4 grid")
    identity = {
        "directory": _path_label(directory),
        "manifest_sha256": _sha256(manifest_path),
        "complete_sha256": _sha256(directory / "COMPLETE"),
        "artifact_sha256": hashes,
    }
    return indexed, identity


def repair_failures(metrics: dict) -> tuple[str, ...]:
    """Engineering repair checks, not preregistered holdout or controller-ranking gates."""
    return tuple(
        name
        for passed, name in (
            (metrics["contact_ratio_pct"] >= 99.0, "contact_ratio"),
            (metrics["force_rmse_n"] <= 2.0, "raw_force_rmse"),
            (metrics["peak_force_n"] <= 35.0, "raw_peak_force"),
            (metrics["saturation_pct"] == 0.0, "saturation"),
        )
        if not passed
    )


def _geometry_gap(trace: dict, yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    # Positive means separation; negative means penetration, matching the runner.
    return (np.array([0.4, 0.0, 0.0]) - trace["position"]) @ normal - 0.025


def _auxiliary_metrics(result, case: dict) -> dict:
    mask = result.trace["time"] >= case["config"].evaluation_start
    gap = _geometry_gap(result.trace, case["scenario"].wall_yaw_deg)
    if "true_contact_gap_m" in result.trace and not np.allclose(
        result.trace["true_contact_gap_m"], gap, rtol=0, atol=1e-12
    ):
        raise ValueError("recorded geometric separation disagrees with wall-plane reconstruction")
    tangent = result.trace.get("true_tangent_force_n")
    if tangent is not None:
        tangent = np.asarray(tangent)
        tangent = np.linalg.norm(tangent, axis=1) if tangent.ndim == 2 else np.abs(tangent)
    loaded = mask & (result.trace["true_normal_force"] > 0.5)
    return {
        "max_geometric_separation_um": float(max(0, gap[mask].max()) * 1e6)
        if np.any(mask)
        else None,
        "mean_true_tangent_force_n": float(np.mean(tangent[mask]))
        if tangent is not None and np.any(mask)
        else None,
        "median_true_friction_ratio": float(
            np.median(tangent[loaded] / result.trace["true_normal_force"][loaded])
        )
        if tangent is not None and np.any(loaded)
        else None,
    }


def _verify_legacy(row: dict, frozen: dict) -> float:
    if (
        row["scenario"] != frozen["scenario"]
        or row["simulation_seed"] != int(frozen["simulation_seed"])
        or row["controller_yaw_deg"] != float(frozen["controller_yaw_deg"])
        or row["has_raw_contact"] != (frozen["has_raw_contact"] == "True")
    ):
        raise ValueError("legacy case identity or contact status differs from baseline")
    errors = []
    for name in METRICS:
        expected = float(frozen[name]) if frozen[name] else None
        actual = row[name]
        if expected is None or actual is None:
            if expected is not actual:
                raise ValueError(f"legacy replay mismatch: {name}")
            errors.append(0.0)
        else:
            error = abs(actual - expected)
            if not np.isfinite(expected) or error > 1e-10:
                raise ValueError(f"legacy replay absolute metric mismatch: {name}, error={error}")
            errors.append(error)
    return max(errors)


def paired_deltas(rows: list[dict]) -> list[dict]:
    baseline = {(row["arm"], row["case_index"]): row for row in rows if row["profile"] == "legacy"}
    pairs = []
    for row in rows:
        if row["profile"] != "smooth":
            continue
        reference = baseline[row["arm"], row["case_index"]]
        contact = row["has_raw_contact"] and reference["has_raw_contact"]
        pairs.append(
            {
                "arm": row["arm"],
                "case_index": row["case_index"],
                "contact_in_both_profiles": contact,
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


def render_summary(rows: list[dict], scope: str) -> str:
    lines = [
        "# Smooth-contact repair validation",
        "",
        scope,
        "Public development cases, NOT a new holdout. This compares contact-model profiles,",
        "not controller superiority. No policy training, gain fitting or controller changes.",
        "Setting both geoms' solimp d0 to zero changes the force-distance response as well as",
        "continuity at contact onset. The model is not physically identified or hardware-validated.",
        "Engineering repair checks: contact >=99%, raw-force RMSE <=2 N, full raw peak <=35 N,",
        "and actuator saturation exactly 0%. These were not preregistered holdout gates.",
        "There is no tangent gate; tangent tracking remains visible as a possible cost.",
        "Contact and RMSE use evaluation_start; peaks and saturation use the full trial.",
        "",
        "| Profile / arm | Contact min [%] | Contact median [%] | Raw RMSE median [N] | Raw RMSE P95 [N] | Peak max [N] | Peak P95 [N] | Tangent median [mm] | Worst saturation [%] | Repair passes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for arm in ARMS:
            selected = [row for row in rows if row["profile"] == profile and row["arm"] == arm]
            values = [
                np.percentile([row[key] for row in selected], percentile)
                for key, percentile in (
                    ("contact_ratio_pct", 0),
                    ("contact_ratio_pct", 50),
                    ("force_rmse_n", 50),
                    ("force_rmse_n", 95),
                    ("peak_force_n", 100),
                    ("peak_force_n", 95),
                    ("tangent_rmse_mm", 50),
                    ("saturation_pct", 100),
                )
            ]
            passed = sum(row["repair_pass"] == "yes" for row in selected)
            lines.append(
                f"| {profile} / {arm} | "
                + " | ".join(f"{value:.6g}" for value in values)
                + f" | {passed}/{len(selected)} |"
            )
    lines += [
        "",
        "paired_deltas.csv records smooth minus legacy for each identical case and frame.",
        "All selected legacy runs first reproduce the archived metrics within absolute 1e-10.",
        "The representative image and four smooth input traces use preselected case 16 only;",
        "they are not the aggregate result. Legacy trace references are retained by hash.",
        "Input replay reproduces controller mathematics, not closed-loop robot dynamics.",
        "Geometry uses the true wall plane: gap >0 means separation, gap <0 penetration.",
        "max_geometric_separation_um measures task-period separation; optional mean true",
        "tangent force and median ||Ft||/Fn (Fn >0.5 N) use recorded contact loads.",
        "Empty auxiliary values mean NA, not zero force or verified absence of separation.",
        "Changing this contact model does not establish physical collision safety.",
        "",
    ]
    return "\n".join(lines)


def _plot_representative(path: Path, legacy: dict, smooth: dict, yaw: float) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 6))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)
    for label, trace in (("legacy", legacy), ("smooth", smooth)):
        axes[0].plot(trace["time"], trace["true_normal_force"], label=label, linewidth=0.8)
        axes[1].plot(trace["time"], 1e6 * _geometry_gap(trace, yaw), label=label, linewidth=0.8)
    axes[0].axhline(12, color="0.5", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Raw true force [N]")
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="0.5", linewidth=0.8)
    axes[1].set_ylabel("Signed geometry gap [um]\npositive = separated")
    axes[1].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Preselected case 16 / exact frame: contact-model profiles, not overall performance"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def generate_surface_contact_validation(
    output_dir: Path | str,
    *,
    baseline_dir: Path | str = Path("results/franka_surface_development"),
    case_indices: Iterable[int] | None = None,
) -> Path:
    baseline_dir = Path(baseline_dir)
    output = validate_output_path(_output_path(output_dir), baseline_dir)
    archived, baseline_identity = _load_baseline(baseline_dir)
    cases = development_cases()
    indices = list(range(len(cases))) if case_indices is None else list(case_indices)
    if any(isinstance(item, (bool, np.bool_)) for item in indices):
        raise TypeError("case indices must be integers, not booleans")
    indices = sorted(integer_index(item) for item in indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("case indices must be nonempty and unique")
    if indices[0] < 0 or indices[-1] >= len(cases):
        raise IndexError("case index outside development grid")
    scope = (
        "Full public development grid: 24 cases x 4 frames x 2 profiles = 192 rows."
        if len(indices) == 24
        else f"Public development SUBSET: {len(indices)}/24 cases; indices={indices}."
    )
    source_hashes = _source_hashes()
    rows, configurations, replay_checks, legacy_references = [], [], {}, {}
    smooth_exact = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".surface-contact-validation-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        for profile in PROFILES:
            for arm, (kind, offset) in ARMS.items():
                for index in indices:
                    case = cases[index]
                    config = replace(case["config"], contact_model=profile)
                    yaw = 0.0 if offset is None else case["task"].yaw_deg + offset
                    angle = np.deg2rad(yaw)
                    frame = SurfaceFrame.from_normal(np.array([np.cos(angle), np.sin(angle), 0.0]))
                    result = run_surface_trial(
                        frame,
                        scenario=case["scenario"],
                        config=config,
                        task=case["task"],
                        controller_kind=kind,
                    )
                    values = _metrics(result)
                    row = {
                        "profile": profile,
                        "arm": arm,
                        "case_index": index,
                        "scenario": case["scenario"].name,
                        "simulation_seed": config.seed,
                        "controller_yaw_deg": yaw,
                        "wall_yaw_deg": case["scenario"].wall_yaw_deg,
                        "wall_time_constant_s": case["scenario"].wall_time_constant,
                        "tool_mass_kg": case["scenario"].tool_mass_kg,
                        **values,
                        **_auxiliary_metrics(result, case),
                        "legacy_metric_max_abs_error": None,
                    }
                    if profile == "legacy":
                        row["legacy_metric_max_abs_error"] = _verify_legacy(
                            row, archived[arm, index]
                        )
                    failures = repair_failures(values)
                    row.update(
                        repair_pass="no" if failures else "yes",
                        failed_repair_checks=";".join(failures),
                    )
                    rows.append(row)
                    configurations.append(
                        {
                            "profile": profile,
                            "arm": arm,
                            "case_index": index,
                            "controller_kind": kind,
                            "controller_frame_rotation": frame.rotation.tolist(),
                            "scenario": asdict(case["scenario"]),
                            "config": asdict(config),
                            "task": asdict(case["task"]),
                        }
                    )
                    if profile == "smooth" and index == REPRESENTATIVE_CASE_INDEX:
                        name = f"representative_case_16_smooth_{arm}.npz"
                        save_surface_trace(staging / name, result.trace)
                        replay = replay_surface_trace(staging / name)
                        if not replay.matches:
                            raise ValueError(f"smooth representative replay mismatch: {arm}")
                        replay_checks[arm] = asdict(replay)
                        old_name = f"representative_case_16_{arm}.npz"
                        if old_name not in baseline_identity["artifact_sha256"]:
                            raise ValueError("baseline manifest must cover representative traces")
                        legacy_references[arm] = {
                            "path": _path_label(baseline_dir / old_name),
                            "sha256": baseline_identity["artifact_sha256"][old_name],
                        }
                        if arm == "surface_exact":
                            smooth_exact = result.trace
                    print(
                        f"completed {len(rows)}/{8 * len(indices)}: {profile} {arm} case {index}",
                        flush=True,
                    )
        if _source_hashes() != source_hashes:
            raise ValueError("source/assets changed during validation")
        if _load_baseline(baseline_dir)[1] != baseline_identity:
            raise ValueError("baseline archive changed during validation")
        _write_csv(staging / "comparison.csv", rows)
        _write_csv(staging / "paired_deltas.csv", paired_deltas(rows))
        (staging / "summary.md").write_text(render_summary(rows, scope), encoding="utf-8")
        if smooth_exact is not None:
            with np.load(
                baseline_dir / "representative_case_16_surface_exact.npz", allow_pickle=False
            ) as archive:
                legacy = {name: archive[name] for name in archive.files}
            _plot_representative(
                staging / "representative_case_16_exact.png",
                legacy,
                smooth_exact,
                cases[REPRESENTATIVE_CASE_INDEX]["scenario"].wall_yaw_deg,
            )
        manifest = {
            "schema_version": 1,
            "experiment_identity": "surface-contact-model-repair-v1",
            "scope": scope,
            "new_holdout": False,
            "is_subset": len(indices) != 24,
            "selected_case_indices": indices,
            "engineering_repair_checks": REPAIR_CHECKS,
            "case_configurations": configurations,
            "baseline_archive": baseline_identity,
            "legacy_trace_references": legacy_references,
            "replay_checks": replay_checks,
            "local_safe_controller_parameters": _constructor_config(FrankaSafeAdaptiveController()),
            "contact_model_description": {
                "legacy_solimp": [0.925, 0.97, 0.0015, 0.5, 2.0],
                "smooth_solimp": [0.0, 0.97, 0.0015, 0.5, 2.0],
                "mixed_solref_by_wall_time_constant": {
                    "0.005": [0.0125, 1.0],
                    "0.012": [0.016, 1.0],
                },
                "sliding_friction": 0.45,
                "interpretation": "Model settings and inferred equal-weight tool/wall mixing, not per-sample observed contact parameters or physical identification. d0 changes force-distance response.",
            },
            "versions": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "mujoco", "compliant-control-lab")},
            },
            "source_and_assets_sha256": source_hashes,
            "artifact_sha256": {path.name: _sha256(path) for path in sorted(staging.iterdir())},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        (staging / "COMPLETE").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="utf-8"
        )
        validate_output_path(output, baseline_dir)
        os.rename(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("results/franka_surface_development"))
    parser.add_argument("--case-indices", type=int, nargs="+")
    args = parser.parse_args()
    print(
        generate_surface_contact_validation(
            args.output, baseline_dir=args.baseline, case_indices=args.case_indices
        )
    )


if __name__ == "__main__":
    main()
