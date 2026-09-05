"""Four fixed reference ablations on public v0.5 development cases, never a new holdout."""

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
from operator import index as integer_index
from pathlib import Path

import numpy as np

from compliant_control_lab.contact_event_analysis import REPLAYED_METRICS
from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.franka_reference import FrankaRateLimitedAdaptiveController
from compliant_control_lab.franka_safety_learning import derive_blind_root, sample_blind_scenarios
from compliant_control_lab.franka_simulation import (
    FrankaScenario,
    FrankaSimulationConfig,
    FrankaTrialResult,
    run_franka_trial,
)
from compliant_control_lab.franka_stress import ResidualRlGate, failed_gate_checks
from compliant_control_lab.published_results_audit import audit_published_results

ARMS = {
    "legacy_replay": ("legacy", "legacy", FrankaSafeAdaptiveController),
    "timing_only": ("split_step", "legacy", FrankaSafeAdaptiveController),
    "consistent_reference": ("split_step", "consistent", FrankaSafeAdaptiveController),
    "rate_limited_reference": ("split_step", "consistent", FrankaRateLimitedAdaptiveController),
}
EVENT_FIELDS = (
    "first_raw_contact_time_s",
    "first_contact_wall_normal_velocity_m_s",
    "early_raw_peak_n",
    "late_raw_peak_n",
    "seconds_over_35_n",
    "peak_wall_normal_velocity_m_s",
    "peak_commanded_wall_normal_force_n",
)
PAIRED_FIELDS = (
    "first_raw_contact_time_s",
    "early_raw_peak_n",
    "late_raw_peak_n",
    "seconds_over_35_n",
    "peak_force_n",
    "force_rmse_n",
    "tangent_rmse_mm",
    "contact_ratio_pct",
)


def analyze_trial(
    result: FrankaTrialResult,
    scenario: FrankaScenario,
    config: FrankaSimulationConfig,
) -> dict:
    """Use the true wall normal only for diagnostics, never as a controller input.

    Force windows include their 0.5/1.0 s boundaries. Missing contact or an unobserved
    window is None; zero force cannot masquerade as a contact improvement. Legacy
    rows intentionally omit the physically aligned event diagnostics.
    """
    output = dict.fromkeys(EVENT_FIELDS)
    output["event_timing"] = (
        "aligned_split_step" if config.control_timing == "split_step" else "legacy_unaligned"
    )
    if result.control_timing != config.control_timing:
        raise ValueError("result/config control timing must match")
    force = np.asarray(result.raw_normal_force)
    time = np.asarray(result.time)
    if time.ndim != 1 or force.shape != time.shape or not np.all(np.isfinite(force)):
        raise ValueError("raw force must be finite and aligned with time")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("time must be finite and strictly increasing")
    if len(time) > 1 and not np.allclose(np.diff(time), config.timestep, rtol=0, atol=1e-12):
        raise ValueError("samples must use the configured timestep")
    if config.control_timing == "split_step":
        for name in ("raw_force_sample_time", "kinematic_sample_time"):
            stamps = np.asarray(getattr(result, name))
            if stamps.shape != time.shape or not np.array_equal(stamps, time):
                raise ValueError(f"{name} must align exactly with time for split-step diagnostics")
    contact = np.flatnonzero(force > 0)
    output["has_raw_contact"] = bool(contact.size)
    if not contact.size or config.control_timing == "legacy":
        return output
    for name, shape in (("linear_velocity", (len(time), 3)), ("commanded_wrench", (len(time), 6))):
        values = np.asarray(getattr(result, name))
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite aligned samples")
    yaw = np.deg2rad(scenario.wall_yaw_deg)
    normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    first, peak = int(contact[0]), int(np.argmax(force))
    delay = time - time[first]
    early = (delay >= 0) & (delay <= 0.5 + 1e-12)
    late = delay >= 1.0 - 1e-12
    output.update(
        first_raw_contact_time_s=float(time[first]),
        first_contact_wall_normal_velocity_m_s=float(normal @ result.linear_velocity[first]),
        early_raw_peak_n=float(np.max(force[early])),
        late_raw_peak_n=float(np.max(force[late])) if np.any(force[late] > 0) else None,
        seconds_over_35_n=float(np.count_nonzero(force > 35.0) * config.timestep),
        peak_wall_normal_velocity_m_s=float(normal @ result.linear_velocity[peak]),
        peak_commanded_wall_normal_force_n=float(normal @ result.commanded_wrench[peak, :3]),
    )
    return output


def validate_output_path(output_dir: Path | str, *protected_dirs: Path) -> Path:
    """Reject symlinks, frozen trees and their ancestors before doing any work."""
    candidate = Path(output_dir).absolute()
    if any(path.is_symlink() for path in (candidate, *candidate.parents)):
        raise ValueError("output path must not contain symlinks")
    output = candidate.resolve()
    repository = Path(__file__).resolve().parents[2]
    protected = (
        repository / "results/franka_safety_blind",
        repository / "results/franka_safety_preholdout",
        *protected_dirs,
    )
    for path in protected:
        frozen = Path(path).resolve()
        if output.is_relative_to(frozen) or frozen.is_relative_to(output):
            raise ValueError("output must be outside frozen directories and their ancestors")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be absent or empty")
    return output


def _constructor_config(value):
    if is_dataclass(value):
        return {
            field.name: _constructor_config(getattr(value, field.name))
            for field in fields(value)
            if field.init
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    paths = sorted(package.glob("*.py")) + sorted((package / "assets").rglob("*"))
    return {str(path.relative_to(package)): _sha256(path) for path in paths if path.is_file()}


def _stat(rows: list[dict], field: str, percentile: float = 50) -> str:
    values = [row[field] for row in rows if row[field] is not None]
    return f"{np.percentile(values, percentile):.5g}" if values else "NA"


def render_summary(rows: list[dict], scope: str, gate: dict) -> str:
    lines = [
        "# Public-development reference ablation",
        "",
        scope,
        "These are already public v0.5 cases, NOT a new holdout. No fitting or tuning occurs.",
        "The frozen v0.5 FAIL is unchanged. All original per-case gates remain active.",
        f"Gate values: `{json.dumps(gate, sort_keys=True)}`",
        "",
        "Legacy event diagnostics are NA because its log fields are not physically aligned.",
        "Split-step raw force is solved at time[k] with action[k]; filtered force feedback is causal",
        "and lagged. The wall normal is diagnostic ground truth only; controllers use world-x.",
        "Normal/tangent task metrics retain their historical definitions, including world-y/z error.",
        "Empty contact windows are NA. Contact count and contact ratio must accompany peak gains.",
        "Seconds above 35 N = count(raw force > 35 N) × timestep; no interpolation is used.",
        "",
        "| Metric | " + " | ".join(ARMS) + " |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline = {row["case_index"]: row for row in rows if row["arm"] == "timing_only"}
    overview = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        overview[arm] = {
            "Cases": str(len(selected)),
            "Cases with raw contact": str(sum(row["has_raw_contact"] for row in selected)),
            "Gate passes": str(sum(row["gate_pass"] == "yes" for row in selected)),
            "First contact median [s]": _stat(selected, "first_raw_contact_time_s"),
            "Contact ratio median [%]": _stat(selected, "contact_ratio_pct"),
            "Global peak P95 [N]": _stat(selected, "peak_force_n", 95),
            "Early peak P95 [N]": _stat(selected, "early_raw_peak_n", 95),
            "Late peak P95 [N]": _stat(selected, "late_raw_peak_n", 95),
            "Time over 35 N median [s]": _stat(selected, "seconds_over_35_n"),
        }
    for metric in overview["timing_only"]:
        lines.append(f"| {metric} | " + " | ".join(overview[arm][metric] for arm in ARMS) + " |")
    lines += [
        "",
        "## Paired differences from timing_only",
        "",
        "Values are arm − timing_only, each case paired by index and noise seed. Peak/time",
        "differences require raw contact in both arms; n is reported for each metric.",
        "Legacy differences change engine timing, so they are not controller improvements.",
        "",
        "| Arm | Metric | Paired n | Median difference |",
        "|---|---|---:|---:|",
    ]
    for arm in ARMS:
        if arm == "timing_only":
            continue
        for field in PAIRED_FIELDS:
            pairs = [
                {"delta": row[field] - baseline[row["case_index"]][field]}
                for row in rows
                if row["arm"] == arm
                and row[field] is not None
                and baseline[row["case_index"]][field] is not None
                and (
                    field in {"force_rmse_n", "tangent_rmse_mm", "contact_ratio_pct"}
                    or (row["has_raw_contact"] and baseline[row["case_index"]]["has_raw_contact"])
                )
            ]
            lines.append(f"| {arm} | {field} | {len(pairs)} | {_stat(pairs, 'delta')} |")
    lines += [
        "",
        "Assess early-peak reductions alongside later peaks, contact delay/loss and",
        "force/tangent tracking costs. The paired ablations isolate component changes;",
        "one peak sample does not identify its physical cause. Deployment readiness is untested.",
        "A final performance claim requires a frozen new protocol",
        "and unseen scenarios. Controller wall-clock latency is descriptive, not deterministic.",
        "",
    ]
    return "\n".join(lines)


def generate_reference_ablation(
    protocol_path: Path | str,
    result_dir: Path | str,
    output_dir: Path | str,
    *,
    case_indices: Iterable[int] | None = None,
) -> Path:
    """Verify the historical arm, run fixed paired arms, then publish one complete report."""
    protocol_path, result_dir = Path(protocol_path), Path(result_dir)
    output = validate_output_path(output_dir, result_dir, protocol_path.parent)
    audit = audit_published_results(protocol_path, result_dir)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    reveal = json.loads((result_dir / "reveal.json").read_text(encoding="utf-8"))
    root = derive_blind_root(reveal["protocol_sha256"], reveal["beacon"]["randomness"])
    scenarios, scenario_seeds, noise_seeds = sample_blind_scenarios(root, audit.case_count)
    indices = list(range(audit.case_count)) if case_indices is None else list(case_indices)
    if any(isinstance(item, bool) for item in indices):
        raise TypeError("case indices must be integers, not booleans")
    indices = [integer_index(item) for item in indices]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("case indices must be nonempty and unique")
    if any(item < 0 or item >= audit.case_count for item in indices):
        raise IndexError("case index outside published set")
    indices.sort()
    with (result_dir / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        archived = {
            row["case"]: row
            for row in csv.DictReader(handle)
            if row["method"] == "safe_adaptive_hybrid"
        }
    gate = ResidualRlGate(**protocol["gate"])
    duration = float(protocol["blind_contract"]["duration_s"])
    scope = (
        f"All {audit.case_count} public cases."
        if len(indices) == audit.case_count
        else f"Public subset: {len(indices)}/{audit.case_count} cases; indices={indices}."
    )
    source_hashes = _source_hashes()
    rows, arm_configs = [], {}
    for arm, (timing, reference, factory) in ARMS.items():
        arm_configs[arm] = {
            "controller_class": factory.__name__,
            "controller_parameters": _constructor_config(factory()),
            "cases": [],
        }
        for case_index in indices:
            config = FrankaSimulationConfig(
                duration=duration,
                seed=noise_seeds[case_index],
                control_timing=timing,
                approach_reference=reference,
            )
            arm_configs[arm]["cases"].append({"case_index": case_index, **asdict(config)})
            result = run_franka_trial(factory(), scenario=scenarios[case_index], config=config)
            metrics = result.metrics()
            if not all(
                np.isfinite(float(metrics[key])) for key in (*REPLAYED_METRICS, "controller_p95_us")
            ):
                raise ValueError(f"non-finite trial metrics: {arm}, case {case_index}")
            errors = dict.fromkeys(REPLAYED_METRICS)
            if arm == "legacy_replay":
                frozen = archived[scenarios[case_index].name]
                errors = {
                    key: abs(float(metrics[key]) - float(frozen[key])) for key in REPLAYED_METRICS
                }
                if max(errors.values()) > 1e-12:
                    raise ValueError(
                        f"legacy replay absolute metric mismatch: case {case_index}: {errors}"
                    )
            failures = failed_gate_checks(metrics, gate)
            rows.append(
                {
                    "arm": arm,
                    "case_index": case_index,
                    "scenario_seed": scenario_seeds[case_index],
                    "simulation_seed": config.seed,
                    **metrics,
                    "gate_pass": "no" if failures else "yes",
                    "failed_checks": ";".join(failures),
                    **{f"verified_{key}_abs_error": value for key, value in errors.items()},
                    **analyze_trial(result, scenarios[case_index], config),
                }
            )
    if _source_hashes() != source_hashes:
        raise ValueError("source/assets changed during ablation; refusing mixed-version report")
    audit_published_results(protocol_path, result_dir)
    manifest = {
        "analysis_identity": "public-development-reference-ablation-v1",
        "scope": scope,
        "new_holdout": False,
        "selected_case_indices": indices,
        "arms": arm_configs,
        "gate": protocol["gate"],
        "frozen_primary_rule": protocol["blind_contract"]["primary_rule"],
        "archive_audit": asdict(audit),
        "archive_files_sha256": {
            str(path): _sha256(path)
            for path in (
                protocol_path,
                result_dir / "manifest.json",
                result_dir / "reveal.json",
                result_dir / "comparison.csv",
            )
        },
        "source_and_assets_sha256": source_hashes,
        "versions": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ("numpy", "mujoco", "compliant-control-lab")
            },
        },
        "all_public_cases": [
            {
                "case_index": index,
                "scenario": asdict(scenario),
                "scenario_seed": scenario_seeds[index],
                "simulation_seed": noise_seeds[index],
            }
            for index, scenario in enumerate(scenarios)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-ablation-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        with (staging / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (staging / "summary.md").write_text(
            render_summary(rows, scope, protocol["gate"]), encoding="utf-8"
        )
        manifest["artifact_sha256"] = {
            name: _sha256(staging / name) for name in ("comparison.csv", "summary.md")
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        (staging / "COMPLETE").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="utf-8"
        )
        validate_output_path(output, result_dir, protocol_path.parent)
        os.rename(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("results/franka_safety_preholdout/protocol.json")
    )
    parser.add_argument("--result", type=Path, default=Path("results/franka_safety_blind"))
    parser.add_argument("--output", type=Path, default=Path("results/franka_reference_ablation"))
    parser.add_argument("--case-indices", type=int, nargs="+")
    arguments = parser.parse_args()
    output = generate_reference_ablation(
        arguments.protocol, arguments.result, arguments.output, case_indices=arguments.case_indices
    )
    print(f"Public-development ablation complete (NOT a new holdout): {output}")


if __name__ == "__main__":
    main()
