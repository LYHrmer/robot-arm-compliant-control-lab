"""Fixed 36-run transfer of two existing error profiles across surface yaw."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import tempfile
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np

from compliant_control_lab.online_compensation_experiment import (
    ARMS,
    GATES,
    DeterministicSchedule,
    _case_document,
    _compact_trace,
    _constructor_config,
    _gate_failures,
    _save_trace,
    _sha256,
    _source_hashes,
    _write_csv,
    _write_json,
    protocol_cases,
    run_protocol_trial,
    summarize_trial,
)
from compliant_control_lab.surface_control import SurfaceAdaptiveController
from compliant_control_lab.surface_simulation import yaw_frame
from tools.cross_surface_pairs import METRICS, RECOVERY_METRICS, pair_gains

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "cross-surface-dynamic-v1"
YAWS = (-15, 0, 15)
PROFILES = ("stop_hold_reverse", "modest_combined")
METHODS = ("baseline", "friction", "online")
SCALES = (1, 2)
SEED = 11
REFERENCES = {
    "1": {"directory": "results/franka_online_compensation_errors",
          "manifest_sha256": "36b1190b2afc6c4e622896a9bf540dd938cdcc55fe5e371b118cf668946ce75d"},
    "2": {"directory": "results/franka_rotation_gain_comparison",
          "manifest_sha256": "664da8de9daea7fc5eaa256cfaef1b76e845d0cc3b97f71b6ef018b18afe6f8a"},
}
INPUT_FIELDS = (
    "time", "position", "target_position", "linear_velocity", "target_linear_velocity",
    "commanded_torque", "applied_torque", "lower_torque_limit", "upper_torque_limit",
    "requested_tangential_force_world", "true_contact_gap_m",
)
KEYS = ("surface_yaw_deg", "scale", "case", "arm", "simulation_seed")


def make_cases():
    originals = {c.name: c for c in protocol_cases() if c.config.seed == SEED}
    cases = []
    for yaw in YAWS:
        for name in PROFILES:
            original = originals[name]
            if yaw == 15:
                cases.append(original)
                continue
            normal = yaw_frame(yaw).rotation[:, 0]
            bias = original.wrench_bias_world
            if name == "modest_combined":
                bias = DeterministicSchedule((
                    (0.0, (0.0,) * 6), (6.0, tuple(0.75 * normal) + (0.0,) * 3),
                ))
            cases.append(replace(
                original,
                scenario=replace(original.scenario, wall_yaw_deg=yaw,
                                 name=f"{original.scenario.name}_yaw_{yaw}"),
                task=replace(original.task, yaw_deg=yaw), wrench_bias_world=bias,
            ))
    return cases


def stem(case, arm, scale):
    return f"yaw_{int(case.scenario.wall_yaw_deg)}__{case.name}__seed_{SEED}__{arm}__s{scale}"


def make_protocol():
    return {
        "identity": IDENTITY, "public_development": True, "new_holdout": False,
        "default_changed": False, "comparison_defined_before_execution": True,
        "yaws": list(YAWS), "profiles": list(PROFILES), "arms": list(METHODS),
        "scales": list(SCALES), "seed": SEED, "gates": GATES,
        "recovery": {"window_s": 0.25, "absolute_force_error_n": 1.0, "tangent_rmse_mm": 3.0},
        "supplemental_checks": {"maximum_projection_pct": 0.0,
            "minimum_reserved_torque_headroom_nm": 0.0, "numeric_tolerance": 1e-10,
            "projection_scale_threshold": 1.0 - 1e-12, "reserve_fraction": 0.10},
        "cases": [_case_document(c) for c in make_cases()],
        "full_trace_selection": {"surface_yaw_deg": -15, "arm": "online",
                                 "profiles": list(PROFILES), "scales": list(SCALES)},
        "input_fields": list(INPUT_FIELDS), "references": REFERENCES,
        "reference_absolute_tolerance": 1e-10,
        "limitations": ["Single preselected noise seed; not statistical robustness or new holdout.",
            "Known mass, fixed home target rotation; no new pitch/roll or unknown payload.",
            "Original trajectory-clock reversal; no new 90-degree path turn.",
            "Engineering gates and time-domain velocity RMS do not prove stability or hardware safety."],
    }


def executions(cases):
    entries = []
    for scale in SCALES:
        for case in cases:
            frame = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg)
            for arm in METHODS:
                control = SurfaceAdaptiveController(
                    frame, tangential_mode=ARMS[arm], rotation_gain_scale=scale,
                )
                entries.append({
                    **identity(case, arm, scale), "stem": stem(case, arm, scale),
                    "controller": {"surface_frame_rotation": frame.rotation.tolist(),
                                   "safe_adaptive_base": _constructor_config(control._base)},
                })
    return entries


def identity(case, arm, scale):
    return dict(zip(KEYS, (case.scenario.wall_yaw_deg, scale, case.name, arm, case.config.seed)))


def source_identity():
    sources = {f"src/compliant_control_lab/{k}": v for k, v in _source_hashes().items()}
    for name in ("cross_surface_regression.py", "cross_surface_pairs.py"):
        sources[f"tools/{name}"] = _sha256(Path(__file__).with_name(name))
    return sources


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_references():
    references = {}
    for scale, specification in REFERENCES.items():
        root = ROOT / specification["directory"]
        digest = specification["manifest_sha256"]
        if _sha256(root / "manifest.json") != digest or (root / "COMPLETE").read_text().strip() != digest:
            raise ValueError("pinned reference manifest differs")
        manifest = json.loads((root / "manifest.json").read_text())
        for name, expected in manifest["artifact_sha256"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or (root / path).is_symlink():
                raise ValueError("unsafe reference artifact")
            if _sha256(root / path) != expected:
                raise ValueError(f"reference artifact mismatch: {name}")
        tables = {}
        for name in ("comparison", "phase_metrics"):
            rows = read_csv(root / f"{name}.csv")
            tables[name] = {(r["case"], r["arm"], int(r["simulation_seed"]),
                             r.get("phase", "overall")): r for r in rows}
        documents = json.loads((root / "protocol.json").read_text())["all_cases"]
        for case in (c for c in make_cases() if c.scenario.wall_yaw_deg == 15):
            matches = [d for d in documents if d["name"] == case.name and d["config"]["seed"] == SEED]
            if matches != [json.loads(json.dumps(_case_document(case)))]:
                raise ValueError("original +15 case no longer matches reference protocol")
        references[int(scale)] = (root, tables)
    return references


def compare_reference(row, reference):
    error = 0.0
    for name, old in reference.items():
        new = row[name]
        if new is None or isinstance(new, (str, bool, np.bool_)):
            if old != ("" if new is None else str(new)):
                raise ValueError(f"reference category differs: {name}")
        else:
            difference = abs(float(old) - float(new))
            if not np.isfinite(difference) or difference > 1e-10:
                raise ValueError(f"reference metric differs: {name} ({difference})")
            error = max(error, difference)
    return error


def extra_metrics(trace, mask):
    torque = trace["commanded_torque"][mask]
    low, high = trace["lower_torque_limit"][mask], trace["upper_torque_limit"][mask]
    reserve = 0.05 * (high - low)
    return {
        "projection_pct": float(100 * np.mean(trace["torque_projection_scale"][mask] < 1 - 1e-12)),
        "minimum_torque_headroom_nm": float(np.min(np.minimum(torque - low, high - torque))),
        "minimum_reserved_torque_headroom_nm": float(np.min(
            np.minimum(torque - low - reserve, high - reserve - torque))),
    }


def annotate(row, baseline, friction, arm, *, phase=False):
    prefix = "paired_baseline" if phase else "paired"
    failures = _gate_failures(row, baseline, friction, arm)
    supplemental = []
    if row["projection_pct"] > 1e-10:
        supplemental.append("torque_projection")
    if row["minimum_reserved_torque_headroom_nm"] < -1e-10:
        supplemental.append("reserved_torque_headroom")
    return {
        **row,
        **{f"{prefix}_{name}_increase_{unit}": row[f"{name}_{unit}"] - baseline[f"{name}_{unit}"]
           for name, unit in (("force_rmse", "n"), ("orientation_rmse", "deg"))},
        **{f"paired_friction_{name}_increase_{unit}": row[f"{name}_{unit}"] - friction[f"{name}_{unit}"]
           for name, unit in (("force_rmse", "n"), ("orientation_rmse", "deg"))},
        "failed_gates": ";".join(failures), "all_gates_pass": "no" if failures else "yes",
        "supplemental_failures": ";".join(supplemental),
        "all_checks_pass": "no" if failures or supplemental else "yes",
    }


def summarize(rows, phases, pairs, reference_errors):
    groups = []
    for yaw in YAWS:
        for scale in SCALES:
            for arm in METHODS:
                selections = [[r for r in table if (r["surface_yaw_deg"], r["scale"], r["arm"])
                               == (yaw, scale, arm)] for table in (rows, phases)]
                groups.append({"surface_yaw_deg": yaw, "scale": scale, "arm": arm,
                    **{f"{label}_{kind}": len(table) if kind == "total" else
                       sum(r["all_checks_pass"] == "yes" for r in table)
                       for label, table in zip(("overall", "phase"), selections)
                       for kind in ("passed", "total")}})
    failures = [{"phase": r.get("phase", "overall"),
                 **{k: r[k] for k in (*KEYS, "failed_gates", "supplemental_failures", *METRICS)}}
                for r in (*rows, *phases) if r["all_checks_pass"] != "yes"]
    return {"identity": IDENTITY, "new_holdout": False, "default_changed": False,
            "comparison_rows": len(rows), "phase_rows": len(phases), "gain_pairs": len(pairs),
            "groups": groups, "failures": failures, "reference_max_abs_error": reference_errors,
            "gain_delta_direction": "scale2_minus_scale1_same_yaw_profile_arm_seed_phase",
            "overall_gain_delta_ranges": {
                arm: {name: {"min": min(values) if values else None,
                             "max": max(values) if values else None}
                      for name in (*METRICS, *RECOVERY_METRICS)
                      for values in [[r[f"delta_{name}"] for r in pairs
                                      if r["arm"] == arm and r["phase"] == "overall"
                                      and r[f"delta_{name}"] is not None]]}
                for arm in METHODS}}


def generate(output_dir):
    output = Path(output_dir).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("output path must not contain symlinks")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    cases, protocol = make_cases(), make_protocol()
    references, sources = load_references(), source_identity()
    entries = executions(cases)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = {str(s): {"comparison": 0.0, "phase_metrics": 0.0, "compact": 0.0} for s in SCALES}
    rows, phases = [], []
    with tempfile.TemporaryDirectory(prefix=".cross-surface-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        traces = staging / "traces"
        traces.mkdir(parents=True)
        _write_json(staging / "protocol.json", protocol)
        _write_json(staging / "configurations.json", {"executions": entries})
        _write_json(staging / "source_hashes.json", sources)
        for scale in SCALES:
            for case in cases:
                overall, windows = {}, {}
                for arm in METHODS:
                    result = run_protocol_trial(case, arm, rotation_gain_scale=scale)
                    whole, partial = summarize_trial(result, case)
                    whole.update(extra_metrics(result.trace, slice(None)))
                    for row, phase in zip(partial, case.phases, strict=True):
                        mask = (result.trace["time"] >= phase.start_s) & (result.trace["time"] < phase.end_s)
                        row.update(extra_metrics(result.trace, mask))
                    overall[arm], windows[arm] = whole, partial
                    label = stem(case, arm, scale)
                    compact = _compact_trace(result.trace, case)
                    _save_trace(traces / f"{label}__compact.npz", compact)
                    inputs = {k: result.trace[k] for k in INPUT_FIELDS}
                    inputs.update({k: np.asarray(v) for k, v in identity(case, arm, scale).items()})
                    with (traces / f"{label}__inputs.npz").open("xb") as handle:
                        np.savez_compressed(handle, **inputs)
                    if case.scenario.wall_yaw_deg == -15 and arm == "online":
                        result.trace.update({k: np.asarray(v) for k, v in identity(case, arm, scale).items()})
                        _save_trace(traces / f"{label}__full.npz", result.trace)
                    if case.scenario.wall_yaw_deg == 15:
                        original = references[scale][0] / "traces" / f"{case.name}__seed_{SEED}__{arm}__compact.npz"
                        with np.load(original, allow_pickle=False) as old:
                            if set(old.files) != set(compact):
                                raise ValueError("reference compact schema differs")
                            for name, values in compact.items():
                                difference = float(np.max(np.abs(values.astype(float) - old[name].astype(float))))
                                if not np.isfinite(difference) or difference > 1e-10:
                                    raise ValueError(f"reference compact differs: {name}")
                                errors[str(scale)]["compact"] = max(errors[str(scale)]["compact"], difference)
                    print(f"{len(rows) + len(overall):02d}/36 {label}", flush=True)
                for arm in METHODS:
                    context = identity(case, arm, scale)
                    row = {**context, "paired_baseline_seed": SEED,
                           "controller_yaw_error_deg": case.controller_yaw_error_deg,
                           **annotate(overall[arm], overall["baseline"], overall["friction"], arm)}
                    rows.append(row)
                    case_rows = []
                    for index, window in enumerate(windows[arm]):
                        case_rows.append({**context, **annotate(window, windows["baseline"][index],
                                                               windows["friction"][index], arm, phase=True)})
                    phases.extend(case_rows)
                    if case.scenario.wall_yaw_deg == 15:
                        for table, selected in (("comparison", [row]), ("phase_metrics", case_rows)):
                            for current in selected:
                                old = references[scale][1][table][case.name, arm, SEED, current.get("phase", "overall")]
                                difference = compare_reference(current, old)
                                errors[str(scale)][table] = max(errors[str(scale)][table], difference)
        if sources != source_identity():
            raise ValueError("source/assets changed during execution")
        load_references()
        pairs = pair_gains([*rows, *phases])
        for name, table in (("comparison", rows), ("phase_metrics", phases), ("gain_paired", pairs)):
            _write_csv(staging / f"{name}.csv", table)
        _write_json(staging / "summary.json", summarize(rows, phases, pairs, errors))
        manifest = {"identity": IDENTITY, "new_holdout": False,
            "versions": {"python": platform.python_version(),
                         **{name: version(name) for name in ("numpy", "mujoco")}},
            "input_sha256": {name: _sha256(staging / name)
                             for name in ("protocol.json", "configurations.json", "source_hashes.json")},
            "artifact_sha256": {str(p.relative_to(staging)): _sha256(p)
                                for p in sorted(staging.rglob("*")) if p.is_file()}}
        _write_json(staging / "manifest.json", manifest)
        (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
        if output.exists():
            raise FileExistsError(f"output appeared during execution: {output}")
        os.rename(staging, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.output))


if __name__ == "__main__":
    main()
