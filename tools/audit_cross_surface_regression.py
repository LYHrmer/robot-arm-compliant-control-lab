"""Recompute cross-surface evidence without rerunning robot dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools import audit_online_compensation_errors as legacy
from tools import cross_surface_regression as study


def close(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"shape/nonfinite mismatch: {label}")
    if not np.allclose(a, b, rtol=0, atol=1e-10):
        raise ValueError(f"numeric mismatch: {label}")


def numeric_rows(rows):
    # Only numeric columns used by gain pairing; categories stay textual.
    numeric = {*study.METRICS, *study.RECOVERY_METRICS, "surface_yaw_deg", "scale", "simulation_seed"}
    return [{k: (None if v == "" else float(v)) if k in numeric else v for k, v in r.items()}
            for r in rows]


def row_key(row, *, gain=False):
    names = ("surface_yaw_deg", "case", "arm", "simulation_seed")
    return tuple(float(row[k]) if k in ("surface_yaw_deg", "simulation_seed") else row[k]
                 for k in names) + (row.get("phase", "overall"),) + (() if gain else (float(row["scale"]),))


def unique_rows(rows, *, gain=False):
    mapping = {row_key(r, gain=gain): r for r in rows}
    if len(mapping) != len(rows):
        raise ValueError("duplicate comparison identity")
    return mapping


def inspect_inputs(path, compact, case, execution):
    with np.load(path, allow_pickle=False) as loaded:
        inputs = {k: loaded[k] for k in loaded.files}
    if set(inputs) != set(study.INPUT_FIELDS) | set(study.KEYS):
        raise ValueError("raw input schema differs")
    for name in study.KEYS:
        if inputs[name].shape != () or inputs[name].item() != execution[name]:
            raise ValueError(f"input identity differs: {name}")
    size = len(compact["time"])
    vector3 = {"position", "target_position", "linear_velocity", "target_linear_velocity",
               "requested_tangential_force_world"}
    vector7 = {"commanded_torque", "applied_torque", "lower_torque_limit", "upper_torque_limit"}
    for name in study.INPUT_FIELDS:
        expected = (size, 3) if name in vector3 else (size, 7) if name in vector7 else (size,)
        if inputs[name].shape != expected or inputs[name].dtype.kind not in "biuf" or not np.all(np.isfinite(inputs[name])):
            raise ValueError(f"input shape/type differs: {name}")
    close(inputs["time"], compact["time"], "input time")
    normal = np.array([np.cos(np.deg2rad(case["scenario"]["wall_yaw_deg"])),
                       np.sin(np.deg2rad(case["scenario"]["wall_yaw_deg"])), 0.0])
    for name, target, field in (("position", "target_position", "tangent_position_error_world"),
                                ("linear_velocity", "target_linear_velocity", "tangent_velocity_error_world")):
        delta = inputs[name] - inputs[target]
        close(delta - np.outer(delta @ normal, normal), compact[field], field)
    low, high = inputs["lower_torque_limit"], inputs["upper_torque_limit"]
    if np.any(high <= low):
        raise ValueError("invalid actuator envelope")
    # The source-bound surface XML has these fixed actuator limits; do not let an
    # archive invent a wider envelope and then reseal its improved headroom.
    limits = np.array([87., 87., 87., 87., 12., 12., 12.])
    close(low, np.broadcast_to(-limits, low.shape), "fixed lower actuator limits")
    close(high, np.broadcast_to(limits, high.shape), "fixed upper actuator limits")
    torque, applied = inputs["commanded_torque"], inputs["applied_torque"]
    close(np.clip(torque, low, high), applied, "physical actuator clipping")
    clipped = np.any(np.abs(torque - applied) > 1e-9, axis=1)
    if not np.array_equal(clipped, compact["actuator_clipped"]):
        raise ValueError("clipping flags differ")
    requested = inputs["requested_tangential_force_world"]
    close(np.linalg.norm(requested, axis=1), compact["requested_tangential_force_norm_n"], "request norm")
    close(np.r_[0.0, np.linalg.norm(np.diff(requested, axis=0), axis=1)],
          compact["requested_tangential_force_delta_norm_n"], "request slew")
    return inputs


def extras(compact, inputs, mask):
    low, high = inputs["lower_torque_limit"][mask], inputs["upper_torque_limit"][mask]
    requested = inputs["commanded_torque"][mask]
    return {"projection_pct": float(100 * np.mean(compact["torque_projection_scale"][mask] < 1 - 1e-12)),
            "minimum_torque_headroom_nm": float(min(np.min(requested - low), np.min(high - requested))),
            "minimum_reserved_torque_headroom_nm": float(min(
                np.min(requested - low - (high - low) / 20),
                np.min(high - (high - low) / 20 - requested)))}


def check_row(published, reconstructed, baseline, friction, arm, *, phase=False):
    for key, value in reconstructed.items():
        legacy._cell(published, key, value, "cross-surface")
    failed = legacy._failures(reconstructed, baseline, friction, arm, legacy.EXPECTED_GATES)
    legacy._cell(published, "failed_gates", ";".join(failed), "phase gates")
    legacy._cell(published, "all_gates_pass", "no" if failed else "yes", "phase gates")
    extra = []
    if reconstructed["projection_pct"] > 1e-10:
        extra.append("torque_projection")
    if reconstructed["minimum_reserved_torque_headroom_nm"] < -1e-10:
        extra.append("reserved_torque_headroom")
    legacy._cell(published, "supplemental_failures", ";".join(extra), "supplemental")
    legacy._cell(published, "all_checks_pass", "no" if failed or extra else "yes", "supplemental")
    prefix = "paired_baseline" if phase else "paired"
    for name, unit in (("force_rmse", "n"), ("orientation_rmse", "deg")):
        metric = f"{name}_{unit}"
        legacy._cell(published, f"{prefix}_{name}_increase_{unit}", reconstructed[metric] - baseline[metric], "baseline pair")
        legacy._cell(published, f"paired_friction_{name}_increase_{unit}", reconstructed[metric] - friction[metric], "friction pair")


def audit_archive(directory):
    root = Path(directory)
    manifest, protocol, configurations, sources = legacy._verify_hashes(root)
    if manifest.get("identity") != study.IDENTITY or manifest.get("new_holdout") is not False:
        raise ValueError("experiment identity differs")
    if protocol != json.loads(json.dumps(study.make_protocol())):
        raise ValueError("fixed cross-surface protocol differs")
    if sources != study.source_identity():
        raise ValueError("archive source is not the current frozen generator/source")
    expected_entries = study.executions(study.make_cases())
    if configurations != json.loads(json.dumps({"executions": expected_entries})):
        raise ValueError("execution matrix or controller configuration differs")
    references = study.load_references()
    _, rows = legacy._read_csv(root / "comparison.csv")
    _, phases = legacy._read_csv(root / "phase_metrics.csv")
    _, paired = legacy._read_csv(root / "gain_paired.csv")
    if (len(rows), len(phases), len(paired)) != (36, 144, 90):
        raise ValueError("incomplete cross-surface matrix")
    lookup, phase_lookup = unique_rows(rows), unique_rows(phases)
    paired_lookup = unique_rows(paired, gain=True)
    documents = {(c["scenario"]["wall_yaw_deg"], c["name"]): c for c in protocol["cases"]}
    rebuilt, rebuilt_phases = {}, {}
    expected_traces = set()
    errors = {str(s): {"comparison": 0.0, "phase_metrics": 0.0, "compact": 0.0} for s in study.SCALES}
    for entry in expected_entries:
        key = row_key(entry)
        if key not in lookup:
            raise ValueError("missing execution row")
        case = documents[entry["surface_yaw_deg"], entry["case"]]
        for name in study.KEYS:
            legacy._cell(lookup[key], name, entry[name], "execution identity")
        legacy._cell(lookup[key], "paired_baseline_seed", 11, "paired seed")
        legacy._cell(lookup[key], "controller_yaw_error_deg", case["controller_yaw_error_deg"], "frame error")
        label = entry["stem"]
        compact = legacy._load_compact(root / "traces" / f"{label}__compact.npz", case, entry["arm"], entry["controller"])
        for name, values in compact.items():
            width = 6 if name in ("applied_raw_wrench_bias_world", "feedback_raw_wrench_bias_world") else (
                3 if name in ("tangent_position_error_world", "tangent_velocity_error_world") else None)
            shape = (6000, width) if width else (6000,)
            if values.shape != shape:
                raise ValueError(f"compact field shape differs: {name}")
        for name in ("actuator_clipped", "controller_update_ready_before_compute",
                     "controller_update_ready_after_compute"):
            if not np.all(np.isin(compact[name], [0, 1])):
                raise ValueError(f"invalid Boolean telemetry: {name}")
        for name in ("contact_blend", "torque_projection_scale"):
            if np.any(compact[name] < 0) or np.any(compact[name] > 1):
                raise ValueError(f"telemetry outside [0,1]: {name}")
        inputs = inspect_inputs(root / "traces" / f"{label}__inputs.npz", compact, case, entry)
        online_entry = next(e for e in expected_entries if e["surface_yaw_deg"] == entry["surface_yaw_deg"]
                            and e["scale"] == entry["scale"] and e["case"] == entry["case"] and e["arm"] == "online")
        old_protocol = {"recovery": protocol["recovery"],
                        "controller_constructor_configurations": {"online": online_entry["controller"]}}
        whole, partial = legacy._summarize(compact, case, entry["arm"], old_protocol)
        whole.update(extras(compact, inputs, slice(None)))
        rebuilt[key] = whole
        for window, definition in zip(partial, case["phases"], strict=True):
            mask = (compact["time"] >= definition["start_s"]) & (compact["time"] < definition["end_s"])
            window.update(extras(compact, inputs, mask))
            rebuilt_phases[row_key({**entry, "phase": window["phase"]})] = window
        expected_traces.update(f"traces/{label}__{suffix}.npz" for suffix in ("compact", "inputs"))
        if entry["surface_yaw_deg"] == -15 and entry["arm"] == "online":
            expected_traces.add(f"traces/{label}__full.npz")
            with np.load(root / "traces" / f"{label}__full.npz", allow_pickle=False) as full:
                for name, values in inputs.items():
                    if values.dtype.kind in "US":
                        if not np.array_equal(full[name], values):
                            raise ValueError("full trace identity differs")
                    else:
                        close(full[name], values, f"full inputs: {name}")
                if full["rotation_gain_scale"].shape != () or full["rotation_gain_scale"].item() != entry["scale"]:
                    raise ValueError("full gain differs")
                recomputed = study._compact_trace({k: full[k] for k in full.files},
                    next(c for c in study.make_cases() if c.name == entry["case"] and c.scenario.wall_yaw_deg == -15))
                for name, values in compact.items():
                    close(values, recomputed[name], f"full compact: {name}")
        if entry["surface_yaw_deg"] == 15:
            reference = references[entry["scale"]][0] / "traces" / f"{entry['case']}__seed_11__{entry['arm']}__compact.npz"
            with np.load(reference, allow_pickle=False) as old:
                if set(old.files) != set(compact):
                    raise ValueError("reference compact schema differs")
                for name, values in compact.items():
                    close(values, old[name], f"reference compact {name}")
                    error = float(np.max(np.abs(values.astype(float) - old[name].astype(float))))
                    errors[str(entry["scale"])]["compact"] = max(errors[str(entry["scale"])]["compact"], error)
    if set(phase_lookup) != set(rebuilt_phases):
        raise ValueError("phase matrix differs")
    actual_traces = {str(p.relative_to(root)) for p in (root / "traces").rglob("*") if p.is_file()}
    if actual_traces != expected_traces:
        raise ValueError("trace retention selection differs")
    for label, published, computed in (("comparison", lookup, rebuilt), ("phase_metrics", phase_lookup, rebuilt_phases)):
        for key, current in computed.items():
            yaw, case, arm, seed, phase, scale = key
            baseline = computed[yaw, case, "baseline", seed, phase, scale]
            friction = computed[yaw, case, "friction", seed, phase, scale]
            check_row(published[key], current, baseline, friction, arm, phase=label == "phase_metrics")
            if yaw == 15:
                original = references[int(scale)][1][label][case, arm, int(seed), phase]
                # Categories are checked verbatim, numeric absolute tolerance is pinned.
                typed = dict(published[key])
                for name, value in current.items():
                    if value is None:
                        typed[name] = None
                    elif isinstance(value, (bool, np.bool_)):
                        typed[name] = published[key][name] == "True"
                    elif not isinstance(value, str):
                        typed[name] = float(published[key][name])
                for name in original:
                    if name not in current and name not in {"case", "arm", "phase", "failed_gates", "all_gates_pass"}:
                        typed[name] = float(typed[name]) if typed[name] != "" else None
                error = study.compare_reference(typed, original)
                errors[str(int(scale))][label] = max(errors[str(int(scale))][label], error)
    typed_rows, typed_phases = numeric_rows(rows), numeric_rows(phases)
    expected_pairs = study.pair_gains([*typed_rows, *typed_phases])
    if set(paired_lookup) != {row_key(r, gain=True) for r in expected_pairs}:
        raise ValueError("gain pairing identity differs")
    for row in expected_pairs:
        current = paired_lookup[row_key(row, gain=True)]
        for name, value in row.items():
            legacy._cell(current, name, value, "gain pair")
    expected_summary = study.summarize(typed_rows, typed_phases, expected_pairs, errors)
    if json.loads((root / "summary.json").read_text()) != expected_summary:
        raise ValueError("summary differs from recomputed rows")
    return {"status": "PASS", "comparison_rows": 36, "phase_rows": 144, "gain_pairs": 90,
            "compact_traces": 36, "raw_input_traces": 36, "full_traces": 4,
            "source_identity_status": "current", "reference_max_abs_error": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_archive(args.audit), sort_keys=True))


if __name__ == "__main__":
    main()
