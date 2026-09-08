"""Paired public-24 regression for fixed classical and online compensation."""

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

from compliant_control_lab.surface_contact_validation import _path_label
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame
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
from compliant_control_lab.surface_simulation import run_surface_trial
from compliant_control_lab.tangential_experiment import AUXILIARY_METRICS, _auxiliary_metrics

ARCHIVE = Path("results/franka_tangential_development")
METHODS = {
    "baseline": "surface_adaptive",
    "integral": "surface_integral",
    "friction": "surface_friction",
    "online": "surface_online",
}
FROZEN_METHODS = tuple(METHODS)[:3]
EXACT_TOLERANCE = 1e-10
CHECKS = {
    "minimum_contact_ratio_pct": 99.0,
    "maximum_raw_peak_force_n": 35.0,
    "required_saturation_pct": 0.0,
    "maximum_paired_force_rmse_increase_n": 0.2,
    "maximum_paired_orientation_rmse_increase_deg": 0.2,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_frozen_archive(directory: Path = ARCHIVE) -> tuple[dict, dict, dict]:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    complete_path = directory / "COMPLETE"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if complete_path.read_text(encoding="utf-8").strip() != _sha256(manifest_path):
        raise ValueError("frozen archive COMPLETE does not match manifest")
    if (
        manifest.get("experiment_identity") != "tangential-compensation-public24-v1"
        or manifest.get("is_subset") is not False
        or manifest.get("selected_case_indices") != list(range(24))
    ):
        raise ValueError("expected the complete public-24 tangential archive")
    artifacts = manifest.get("artifact_sha256", {})
    if "comparison.csv" not in artifacts:
        raise ValueError("frozen manifest must cover comparison.csv")
    for name, digest in artifacts.items():
        if Path(name).name != name or _sha256(directory / name) != digest:
            raise ValueError(f"frozen artifact hash mismatch: {name}")

    rows = [row for row in _read_csv(directory / "comparison.csv") if row["category"] == "main"]
    expected = {(method, index) for method in FROZEN_METHODS for index in range(24)}
    mapping = {(row["method"], int(row["case_index"])): row for row in rows}
    if len(rows) != len(mapping) or set(mapping) != expected:
        raise ValueError("frozen comparison must contain one row per method and public case")

    configurations = {
        (entry["method"], int(entry["case_index"])): entry
        for entry in manifest.get("case_configurations", [])
        if entry.get("category") == "main" and entry.get("method") in FROZEN_METHODS
    }
    if set(configurations) != expected:
        raise ValueError("frozen manifest has an incomplete public-24 configuration mapping")
    identity = {
        "directory": _path_label(directory),
        "manifest_sha256": _sha256(manifest_path),
        "complete_sha256": _sha256(complete_path),
        "comparison_sha256": artifacts["comparison.csv"],
    }
    return mapping, configurations, identity


def _normalized(value) -> object:
    return json.loads(json.dumps(asdict(value)))


def _fixed_cases(configurations: dict, indices: list[int]) -> list[dict]:
    cases = development_cases()
    if len(cases) != 24 or [case["case_index"] for case in cases] != list(range(24)):
        raise ValueError("development_cases must remain the unchanged ordered 24-case grid")
    for index in indices:
        cases[index] = {
            **cases[index],
            "config": replace(cases[index]["config"], contact_model="smooth", duration=4.5),
        }
        for method in FROZEN_METHODS:
            recorded = configurations[method, index]
            for key in ("scenario", "config", "task"):
                if _normalized(cases[index][key]) != recorded[key]:
                    raise ValueError(
                        f"development case differs from frozen archive: {method}/{index}/{key}"
                    )
    return cases


def _frozen_metric_error(row: dict, frozen: dict) -> float:
    if (
        row["scenario"] != frozen["scenario"]
        or row["simulation_seed"] != int(frozen["simulation_seed"])
        or row["has_raw_contact"] != (frozen["has_raw_contact"] == "True")
    ):
        raise ValueError("frozen case identity or contact status differs")
    errors = []
    for name in (*METRICS, *AUXILIARY_METRICS):
        expected = float(frozen[name]) if frozen[name] else None
        actual = row[name]
        error = (
            0.0
            if actual is expected
            else abs(actual - expected)
            if actual is not None and expected is not None
            else float("inf")
        )
        if not np.isfinite(error) or error > EXACT_TOLERANCE:
            raise ValueError(f"frozen metric mismatch: {name}, error={error}")
        errors.append(error)
    return max(errors)


def online_failures(row: dict, friction: dict) -> tuple[str, ...]:
    """Return every failed online-vs-friction development check."""
    return tuple(
        name
        for passed, name in (
            (
                row["has_raw_contact"]
                and row["contact_ratio_pct"] >= CHECKS["minimum_contact_ratio_pct"],
                "contact_ratio",
            ),
            (row["peak_force_n"] <= CHECKS["maximum_raw_peak_force_n"], "raw_peak_force"),
            (row["saturation_pct"] == CHECKS["required_saturation_pct"], "saturation"),
            (
                row["force_rmse_n"] - friction["force_rmse_n"]
                <= CHECKS["maximum_paired_force_rmse_increase_n"] + 1e-12,
                "paired_force_rmse",
            ),
            (
                row["orientation_rmse_deg"] - friction["orientation_rmse_deg"]
                <= CHECKS["maximum_paired_orientation_rmse_increase_deg"] + 1e-12,
                "paired_orientation_rmse",
            ),
        )
        if not passed
    )


def online_pairs(rows: list[dict]) -> list[dict]:
    friction = {row["case_index"]: row for row in rows if row["method"] == "friction"}
    pairs = []
    for row in (item for item in rows if item["method"] == "online"):
        reference = friction[row["case_index"]]
        contact = row["has_raw_contact"] and reference["has_raw_contact"]
        pairs.append(
            {
                "case_index": row["case_index"],
                "scenario": row["scenario"],
                "simulation_seed": row["simulation_seed"],
                "contact_in_both_methods": contact,
                **{
                    name: row[name] - reference[name]
                    if row[name] is not None
                    and reference[name] is not None
                    and (name not in CONTACT_METRICS or contact)
                    else None
                    for name in (*METRICS, *AUXILIARY_METRICS)
                },
                "online_gate_pass": row["online_gate_pass"],
                "failed_online_gates": row["failed_online_gates"],
            }
        )
    return pairs


def _source_identity() -> dict[str, str]:
    sources = {f"src/compliant_control_lab/{name}": digest for name, digest in _source_hashes().items()}
    sources["tools/online_compensation_regression.py"] = _sha256(Path(__file__).resolve())
    return sources


def _summary(rows: list[dict], pairs: list[dict], indices: list[int]) -> dict:
    online = [row for row in rows if row["method"] == "online"]
    tangent = [row["tangent_rmse_mm"] for row in online]
    tangent_delta = [row["tangent_rmse_mm"] for row in pairs]
    return {
        "schema_version": 1,
        "experiment_identity": "online-compensation-public24-regression-v1",
        "scope": "full public 24-case development grid" if len(indices) == 24 else "public subset smoke run",
        "new_holdout": False,
        "is_subset": len(indices) != 24,
        "selected_case_indices": indices,
        "reference_method": "friction",
        "engineering_checks": CHECKS,
        "online_outcome": {
            "case_count": len(online),
            "case_pass_count": sum(row["online_gate_pass"] == "yes" for row in online),
            "all_cases_pass": all(row["online_gate_pass"] == "yes" for row in online),
            "median_tangent_rmse_mm": float(np.median(tangent)),
            "worst_tangent_rmse_mm": float(np.max(tangent)),
            "median_paired_tangent_delta_mm": float(np.median(tangent_delta)),
            "worst_paired_tangent_delta_mm": float(np.max(tangent_delta)),
            "failed_cases": [
                {
                    "case_index": row["case_index"],
                    "failed_online_gates": row["failed_online_gates"],
                }
                for row in online
                if row["online_gate_pass"] != "yes"
            ],
        },
        "paired_online_vs_friction": pairs,
        "representative_trace": {
            "retained": False,
            "case_index": 16 if 16 in indices else None,
            "reason": "surface_online is not supported by the frozen surface trace replay schema",
        },
    }


def generate_online_compensation_regression(
    output_dir: Path | str,
    *,
    case_indices: Iterable[int] | None = None,
    archive_dir: Path | str = ARCHIVE,
) -> Path:
    """Run four paired methods without changing the frozen public-24 protocol."""
    archive_dir = Path(archive_dir)
    output = _output_path(output_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    indices = list(range(24)) if case_indices is None else list(case_indices)
    if any(isinstance(item, (bool, np.bool_)) for item in indices):
        raise TypeError("case indices must be integers, not booleans")
    indices = sorted(integer_index(item) for item in indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("case indices must be nonempty and unique")
    if indices[0] < 0 or indices[-1] >= 24:
        raise IndexError("case index outside fixed development grid")

    frozen, frozen_configurations, archive_identity = _load_frozen_archive(archive_dir)
    cases = _fixed_cases(frozen_configurations, indices)
    sources = _source_identity()
    rows: list[dict] = []
    configurations: list[dict] = []
    friction_rows: dict[int, dict] = {}

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".online-regression-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        jobs = [(method, cases[index]) for method in METHODS for index in indices]
        for count, (method, case) in enumerate(jobs, 1):
            scenario, config, task = (case[key] for key in ("scenario", "config", "task"))
            angle = np.deg2rad(task.yaw_deg)
            frame = SurfaceFrame.from_normal(np.array([np.cos(angle), np.sin(angle), 0.0]))
            result = run_surface_trial(
                frame,
                scenario=scenario,
                config=config,
                task=task,
                controller_kind=METHODS[method],
            )
            row = {
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
                "frozen_metric_max_abs_error": None,
                "online_gate_pass": "",
                "failed_online_gates": "",
            }
            if method in FROZEN_METHODS:
                row["frozen_metric_max_abs_error"] = _frozen_metric_error(
                    row, frozen[method, case["case_index"]]
                )
            if method == "friction":
                friction_rows[case["case_index"]] = row
            elif method == "online":
                failures = online_failures(row, friction_rows[case["case_index"]])
                row["online_gate_pass"] = "no" if failures else "yes"
                row["failed_online_gates"] = ";".join(failures)
            rows.append(row)
            configurations.append(
                {
                    "method": method,
                    "case_index": case["case_index"],
                    "controller_kind": METHODS[method],
                    "controller_frame_rotation": frame.rotation.tolist(),
                    **{name: asdict(case[name]) for name in ("scenario", "config", "task")},
                }
            )
            print(f"completed {count}/{len(jobs)}: {method} case {case['case_index']}", flush=True)

        pairs = online_pairs(rows)
        summary = _summary(rows, pairs, indices)
        _write_csv(staging / "comparison.csv", rows)
        _write_csv(staging / "paired_online_vs_friction.csv", pairs)
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "configurations.json").write_text(
            json.dumps(configurations, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "source_hashes.json").write_text(
            json.dumps(sources, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if _source_identity() != sources or _load_frozen_archive(archive_dir)[2] != archive_identity:
            raise ValueError("source or frozen archive changed during regression")
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
            **{key: summary[key] for key in ("schema_version", "experiment_identity", "scope", "new_holdout", "is_subset", "selected_case_indices")},
            "methods": METHODS,
            "engineering_checks": CHECKS,
            "frozen_archive": archive_identity,
            "controller_parameters": parameters,
            "configuration_sha256": _sha256(staging / "configurations.json"),
            "source_hashes_sha256": _sha256(staging / "source_hashes.json"),
            "versions": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "mujoco", "compliant-control-lab")},
            },
            "artifact_sha256": {
                path.name: _sha256(path) for path in sorted(staging.iterdir())
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "COMPLETE").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        os.rename(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-indices", type=int, nargs="+")
    args = parser.parse_args()
    print(
        generate_online_compensation_regression(
            args.output,
            case_indices=args.case_indices,
        )
    )


if __name__ == "__main__":
    main()
