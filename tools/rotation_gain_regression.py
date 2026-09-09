"""Matched constant-rotation-gain study on the unchanged public 24-case grid."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame
from compliant_control_lab.surface_experiment import (
    METRICS,
    _constructor_config,
    _metrics,
    _output_path,
    _sha256,
    _source_hashes,
    _write_csv,
)
from compliant_control_lab.surface_replay import save_surface_trace
from compliant_control_lab.surface_simulation import SurfaceSimulator, SurfaceTrialResult
from compliant_control_lab.tangential_experiment import AUXILIARY_METRICS, _auxiliary_metrics
from tools.online_compensation_regression import (
    CHECKS,
    METHODS,
    _fixed_cases,
    _frozen_metric_error,
    _read_csv,
    online_failures,
)

IDENTITY = "rotation-gain-public24-v1"
SCALES = (1.0, 2.0)
REFERENCE = Path(__file__).resolve().parents[1] / "results/franka_online_compensation_regression"
REFERENCE_MANIFEST_SHA256 = "ddd65fd48fd078603c399aa1a9f7485c57ab018610139800e233c3c7764ef543"
FULL_CASES = (0, 16, 23)
EXTRA_METRICS = (
    "tangent_velocity_error_rms_m_s",
    "minimum_torque_headroom_nm",
    "minimum_reserved_torque_headroom_nm",
)
ALL_METRICS = (*METRICS, *AUXILIARY_METRICS, *EXTRA_METRICS)
COMPACT_FIELDS = (
    "time", "true_normal_force", "target_normal_force", "position", "target_position",
    "linear_velocity", "target_linear_velocity", "orientation_error_rad",
    "measured_normal_force", "commanded_torque", "applied_torque", "lower_torque_limit",
    "upper_torque_limit", "torque_projection_scale", "requested_tangential_force_world",
    "true_contact_gap_m",
)


def load_reference() -> tuple[dict, dict, dict]:
    manifest_path = REFERENCE / "manifest.json"
    if _sha256(manifest_path) != REFERENCE_MANIFEST_SHA256:
        raise ValueError("reference does not match the pinned public96 manifest")
    if (REFERENCE / "COMPLETE").read_text().strip() != REFERENCE_MANIFEST_SHA256:
        raise ValueError("reference COMPLETE does not bind manifest")
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["artifact_sha256"].items():
        path = REFERENCE / name
        if Path(name).name != name or path.is_symlink() or _sha256(path) != digest:
            raise ValueError(f"reference artifact mismatch: {name}")
    rows = _read_csv(REFERENCE / "comparison.csv")
    entries = json.loads((REFERENCE / "configurations.json").read_text())
    keys = {(method, index) for method in METHODS for index in range(24)}
    mapping = {(row["method"], int(row["case_index"])): row for row in rows}
    configs = {(row["method"], row["case_index"]): row for row in entries}
    if len(rows) != 96 or len(entries) != 96 or set(mapping) != keys or set(configs) != keys:
        raise ValueError("reference must contain the unique four-method public24 grid")
    return mapping, configs, {
        "directory": "results/franka_online_compensation_regression",
        "manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "artifact_sha256": manifest["artifact_sha256"],
    }


def source_identity() -> dict:
    sources = {f"src/compliant_control_lab/{k}": v for k, v in _source_hashes().items()}
    for name in ("rotation_gain_regression.py", "online_compensation_regression.py"):
        sources[f"tools/{name}"] = _sha256(Path(__file__).with_name(name))
    return sources


def controller(frame: SurfaceFrame, method: str, scale: float) -> SurfaceAdaptiveController:
    if method not in METHODS or scale not in SCALES or isinstance(scale, (bool, np.bool_)):
        raise ValueError("expected one of four methods and fixed scale 1 or 2")
    return SurfaceAdaptiveController(
        frame, tangential_mode=None if method == "baseline" else method,
        rotation_gain_scale=scale,
    )


def run_trial(case: dict, method: str, scale: float) -> SurfaceTrialResult:
    angle = np.deg2rad(case["task"].yaw_deg)
    frame = SurfaceFrame.from_normal(np.array([np.cos(angle), np.sin(angle), 0.0]))
    control = controller(frame, method, scale)
    sim = SurfaceSimulator(frame, case["scenario"], case["config"], case["task"], METHODS[method])
    for step in range(round(case["config"].duration / case["config"].timestep)):
        sample = sim.sample()
        if step == 0:
            control.reset(sample.state)
        sim.step(control.compute(sample.state, sample.target, sim.config.timestep), control)
    result = sim.result()
    result.trace.update(rotation_gain_scale=np.array(scale), case_index=np.array(case["case_index"]),
                        method=np.array(method))
    return result


def trial_metrics(result: SurfaceTrialResult, case: dict) -> dict:
    trace = result.trace
    mask = trace["time"] >= case["config"].evaluation_start
    angle = np.deg2rad(case["scenario"].wall_yaw_deg)
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    error = trace["linear_velocity"][mask] - trace["target_linear_velocity"][mask]
    error -= np.outer(error @ normal, normal)
    torque = trace["commanded_torque"]
    low, high = trace["lower_torque_limit"], trace["upper_torque_limit"]
    reserve = 0.05 * (high - low)  # Existing controller reserve_fraction = 0.10.
    return {
        **_metrics(result), **_auxiliary_metrics(result, case),
        "tangent_velocity_error_rms_m_s": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "minimum_torque_headroom_nm": float(np.min(np.minimum(torque - low, high - torque))),
        "minimum_reserved_torque_headroom_nm": float(np.min(
            np.minimum(torque - low - reserve, high - reserve - torque)
        )),
    }


def failures(row: dict, friction: dict) -> tuple[str, ...]:
    extra = []
    if row["projection_pct"] > 1e-10:
        extra.append("torque_projection")
    if row["minimum_reserved_torque_headroom_nm"] < -1e-10:
        extra.append("reserved_torque_headroom")
    return (*online_failures(row, friction), *extra)


def pair_rows(rows: list[dict]) -> list[dict]:
    old = {(r["method"], r["case_index"]): r for r in rows if r["scale"] == 1}
    pairs = []
    for row in (r for r in rows if r["scale"] == 2):
        baseline = old[row["method"], row["case_index"]]
        context = ("method", "case_index", "simulation_seed", "wall_yaw_deg",
                   "wall_time_constant_s", "tool_mass_kg")
        if any(row[key] != baseline[key] for key in context):
            raise ValueError("gain pair case/seed identity differs")
        pairs.append({
            **{key: row[key] for key in context}, "from_scale": 1.0, "to_scale": 2.0,
            **{key: row[key] - baseline[key]
               if row[key] is not None and baseline[key] is not None else None
               for key in ALL_METRICS},
        })
    return pairs


def summary(rows: list[dict], pairs: list[dict], indices: list[int]) -> dict:
    effects = {}
    for method in METHODS:
        group = [r for r in pairs if r["method"] == method]
        effects[method] = {}
        for name in ALL_METRICS:
            values = [r[name] for r in group if r[name] is not None]
            effects[method][name] = {
                "count": len(values), "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "minimum": min(values) if values else None, "maximum": max(values) if values else None,
                "negative_count": sum(v < -1e-10 for v in values),
                "positive_count": sum(v > 1e-10 for v in values),
            }
    group_keys = ("method", "wall_yaw_deg", "wall_time_constant_s", "tool_mass_kg")
    grouped = []
    for key in sorted({tuple(r[k] for k in group_keys) for r in pairs}):
        group = [r for r in pairs if tuple(r[k] for k in group_keys) == key]
        grouped.append({
            **dict(zip(group_keys, key, strict=True)),
            "noise_seeds": sorted(r["simulation_seed"] for r in group),
            "mean_deltas": {name: float(np.mean([r[name] for r in group]))
                            if all(r[name] is not None for r in group) else None
                            for name in ALL_METRICS},
        })
    return {
        "identity": IDENTITY, "is_subset": len(indices) != 24, "new_holdout": False,
        "comparison_rows": len(rows), "paired_rows": len(pairs),
        "gate_counts": {str(int(s)): {m: sum(r["all_gates_pass"] == "yes" for r in rows
            if r["scale"] == s and r["method"] == m) for m in METHODS} for s in SCALES},
        "failures": [{k: r[k] for k in ("scale", "method", "case_index", "failed_gates")}
                     for r in rows if r["all_gates_pass"] != "yes"],
        "frozen_metric_max_abs_error": max(r["frozen_metric_max_abs_error"] for r in rows
                                           if r["scale"] == 1),
        "delta_direction": "scale2_minus_scale1_same_method_case_seed",
        "paired_effects": effects, "physical_groups": grouped,
        "default_changed": False,
        "decision_scope": "Engineering checks and tradeoffs; no automatic global-default promotion.",
    }


def make_protocol(indices: list[int], configurations: dict) -> tuple[dict, list[dict]]:
    if (not indices or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                           or i < 0 or i >= 24 for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("case indices must be unique integers in [0,24)")
    cases = _fixed_cases(configurations, indices)
    for i in indices:
        for method in METHODS:
            for key in ("scenario", "config", "task"):
                if json.loads(json.dumps(asdict(cases[i][key]))) != configurations[method, i][key]:
                    raise ValueError("reference case differs across methods")
    protocol = {
        "identity": IDENTITY, "scales": list(SCALES), "methods": METHODS,
        "selected_case_indices": indices, "is_subset": len(indices) != 24, "new_holdout": False,
        "checks": {**CHECKS, "maximum_projection_pct": 0.0,
                   "minimum_reserved_torque_headroom_nm": 0.0, "numeric_tolerance": 1e-10},
        "gate_scope": "Same-scale friction reference for every method; new descriptive engineering checks.",
        "configurations": [{"case_index": i, **{k: asdict(cases[i][k])
                            for k in ("scenario", "config", "task")}} for i in indices],
        "controller_parameters": {str(int(s)): {m: _constructor_config(
            controller(SurfaceFrame(np.eye(3)), m, s)._base) for m in METHODS} for s in SCALES},
        "full_trace_case_indices": list(FULL_CASES), "full_trace_methods": ["friction", "online"],
        "compact_fields": list(COMPACT_FIELDS),
        "pair_metrics": list(ALL_METRICS), "torque_reserve_fraction": 0.10,
        "decision_policy": "No new tuning; report all paired and physical-group costs before adoption.",
    }
    return protocol, cases


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def generate(output_dir: Path | str, *, case_indices=None) -> Path:
    output = _output_path(output_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    indices = list(range(24)) if case_indices is None else list(case_indices)
    reference, configurations, reference_identity = load_reference()
    protocol, cases = make_protocol(indices, configurations)
    sources = source_identity()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rotation-public24-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        (staging / "traces").mkdir(parents=True)
        _write_json(staging / "protocol.json", protocol)
        _write_json(staging / "source_hashes.json", sources)
        rows = []
        for scale in SCALES:
            for index in indices:
                case = cases[index]
                for method in METHODS:
                    result = run_trial(case, method, scale)
                    row = {
                        "scale": scale, "method": method, "case_index": index,
                        "scenario": case["scenario"].name, "simulation_seed": case["config"].seed,
                        "wall_yaw_deg": case["scenario"].wall_yaw_deg,
                        "wall_time_constant_s": case["scenario"].wall_time_constant,
                        "tool_mass_kg": case["scenario"].tool_mass_kg,
                        "duration_s": case["config"].duration, **trial_metrics(result, case),
                        "frozen_metric_max_abs_error": None,
                    }
                    if scale == 1:
                        row["frozen_metric_max_abs_error"] = _frozen_metric_error(row, reference[method, index])
                    rows.append(row)
                    stem = f"s{int(scale)}__{method}__case_{index:02d}"
                    compact = {k: result.trace[k] for k in COMPACT_FIELDS}
                    compact.update(rotation_gain_scale=np.array(scale), case_index=np.array(index),
                                   method=np.array(method))
                    np.savez_compressed(staging / "traces" / f"{stem}__compact.npz", **compact)
                    if index in FULL_CASES and method in ("friction", "online"):
                        save_surface_trace(staging / "traces" / f"{stem}__full.npz", result.trace)
                    print(f"completed {len(rows)}/{8 * len(indices)}: {stem}", flush=True)
        friction = {(r["scale"], r["case_index"]): r for r in rows if r["method"] == "friction"}
        for row in rows:
            failed = failures(row, friction[row["scale"], row["case_index"]])
            row.update(failed_gates=";".join(failed), all_gates_pass="no" if failed else "yes")
        pairs = pair_rows(rows)
        _write_csv(staging / "comparison.csv", rows)
        _write_csv(staging / "paired.csv", pairs)
        _write_json(staging / "summary.json", summary(rows, pairs, indices))
        if source_identity() != sources or load_reference()[2] != reference_identity:
            raise ValueError("source or reference changed during regression")
        manifest = {
            "identity": IDENTITY, "is_subset": protocol["is_subset"], "new_holdout": False,
            "reference_archive": reference_identity,
            "versions": {"python": platform.python_version(), **{k: version(k) for k in ("numpy", "mujoco")}},
            "artifact_sha256": {str(p.relative_to(staging)): _sha256(p)
                                for p in sorted(staging.rglob("*")) if p.is_file()},
        }
        _write_json(staging / "manifest.json", manifest)
        (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
        if output.exists():
            raise FileExistsError(f"output appeared during regression: {output}")
        os.rename(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-indices", type=int, nargs="+")
    args = parser.parse_args()
    print(generate(args.output, case_indices=args.case_indices))


if __name__ == "__main__":
    main()
