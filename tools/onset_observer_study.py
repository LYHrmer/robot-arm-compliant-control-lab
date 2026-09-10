"""Instrument selected public24 reruns without changing their controller or inputs."""

import argparse
import json
import os
import platform
import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np

from compliant_control_lab.online_compensation_experiment import _sha256, _write_csv, _write_json
from compliant_control_lab.tangential_compensation import TangentialCompensation
from tools import budget_public24 as public
from tools import budget_transfer as transfer
from tools import onset_observer as observer
from tools import onset_observer_validation as validation
from tools import velocity_cost_study as velocity

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "public24-onset-observer-v1"
SELECTED = ((7, 1.), (15, 1.), (23, 1.), (23, 2.))
WINDOWS = (("motion_ramp", 1.2, 1.5), ("onset", 1.5, 2.),
           ("middle", 2., 3.5), ("late", 3.5, 4.5))
PARENTS = {**velocity.PARENTS, "velocity": {
    "directory": "results/franka_velocity_cost",
    "manifest_sha256": "1179f95aff09ca82174abccdf7cd65b643050ba682f36d4f59509c6f249ab488",
}}


def parameters(budget):
    compensation = TangentialCompensation("online", max_force=budget)
    return {f.name: getattr(compensation, f.name) for f in fields(compensation) if f.init}


def specifications():
    cases = {c["case_index"]: c for c in public.cases()}
    for index, scale in SELECTED:
        for budget in (6., 8.):
            yield cases[index], scale, budget, f"traces/case{index:02d}__s{int(scale)}__f{int(budget)}n.npz"


def protocol_document():
    return {
        "identity": IDENTITY, "parents": PARENTS, "new_simulations": 8,
        "selected_pairs": [list(p) for p in SELECTED], "default_changed": False,
        "new_holdout": False, "controller_changed": False,
        "selection_reason": "same tau .012s, mass .13kg, seed29 across yaw; add +15deg gain2 worst pair",
        "duration_s": 4.5, "timestep_s": .002, "evaluation_start_s": 1.5,
        "primary_window": "onset", "windows": [list(w) for w in WINDOWS],
        "cases": [public.case_document(c) for c, _, _, _ in specifications()],
        "parameters": {str(int(b)): parameters(b) for b in (6., 8.)},
        "controllers": [{"case_index": c["case_index"], "rotation_gain_scale": s,
                         "max_force_n": b, "constructor": public.controller_document(s, b)}
                        for c, s, b, _ in specifications()],
        "observation": "copies of actual local governed/measured arguments around one parent force/advance call",
        "error_sign": "target_minus_measured_state; the direction is soft-normalized",
        "positive_update_threshold": 1e-15, "first_difference_threshold": 1e-12,
        "reference_match": "all common public compact arrays exact; stored old arrays are never overwritten",
        "limitations": [
            "Eight new simulations recovering missing telemetry, not eight new independent conditions.",
            "Selected public-development cases cannot update the original 24-case pass count.",
            "Hypothetical raw increments need not be accepted by cap, readiness or projection gates.",
            "Ahead velocity while position lags can be necessary catch-up; this alone is not excess force.",
            "No controller tuning, learning, causal performance claim, passivity or hardware guarantee.",
        ],
    }


def source_identity():
    result = velocity.source_identity()
    for filename in (__file__, observer.__file__, validation.__file__):
        path = Path(filename).resolve()
        result[path.relative_to(ROOT).as_posix()] = _sha256(path)
    return dict(sorted(result.items()))


def input_archives():
    roots = {}
    for name, record in PARENTS.items():
        roots[name] = transfer.safe_path(ROOT, record["directory"])
        transfer.verify_archive(roots[name], record["manifest_sha256"])
    archived = json.loads((roots["transfer"] / "protocol.json").read_text())
    if archived["public24"]["cases"] != [public.case_document(c) for c in public.cases()]:
        raise ValueError("parent public24 cases differ")
    for row in archived["controllers"]:
        if row["public24"] != public.controller_document(row["rotation_gain_scale"], row["max_force_n"]):
            raise ValueError("parent controller constructor differs")
    current = source_identity()
    for root in roots.values():
        for name, digest in json.loads((root / "source_hashes.json").read_text()).items():
            if name.startswith("src/") and current.get(name) != digest:
                raise ValueError(f"parent simulation/controller source differs: {name}")
    return roots


def parent_trace(roots, case, scale, budget):
    if budget == 6.:
        name = public.reference_trace(case, scale)
        root = roots["public24"]
    else:
        name = f"traces/public24__case{case['case_index']:02d}__s{int(scale)}__f8n.npz"
        root = roots["transfer"]
    return transfer.load_trace(transfer.safe_path(root, name)), f"{root.relative_to(ROOT).as_posix()}/{name}"


def window_rows(trace):
    rows = []
    time = trace["time"]
    delta = trace["obs_coefficient_after_advance"] - trace["obs_coefficient_after_force"]
    growth = delta > 1e-15
    lag = trace["obs_position_drive_m"] > 0
    faster = trace["obs_velocity_drive_m"] < 0
    ready = trace["obs_update_ready"]
    active = trace["obs_active"]
    allowed = trace["obs_advance_called"] & trace["obs_allow_integration"]
    cap_blocked = active & ready & allowed & trace["obs_amplitude_capped"] & (trace["obs_limited_increment"] > 0)
    for label, start, end in WINDOWS:
        mask = (time >= start) & (time < end)
        row = {"window": label, "start_s": start, "end_s": end, "samples": int(np.sum(mask))}
        for name, values in {
            "active": active, "ready": ready, "projection_allowed": allowed,
            "amplitude_capped": trace["obs_amplitude_capped"], "slew_limited": trace["obs_slew_limited"],
            "positive_update": growth, "cap_blocks_positive_update": cap_blocked,
            "position_lag_velocity_ahead": lag & faster,
            "positive_update_lag_ahead": growth & lag & faster,
            "rate_limited_candidate": np.abs(trace["obs_candidate_increment"] - trace["obs_limited_increment"]) > 1e-15,
        }.items():
            row[f"{name}_pct"] = float(100 * np.mean(values[mask]))
        row.update({
            "position_drive_mean_mm": float(1000 * np.mean(trace["obs_position_drive_m"][mask])),
            "velocity_drive_mean_mm": float(1000 * np.mean(trace["obs_velocity_drive_m"][mask])),
            "drive_mean_mm": float(1000 * np.mean(trace["obs_drive_m"][mask])),
            "accepted_positive_coefficient_sum": float(np.sum(np.maximum(delta[mask], 0))),
            "accepted_negative_coefficient_sum": float(np.sum(np.minimum(delta[mask], 0))),
            "coefficient_start": float(trace["obs_coefficient_before_force"][mask][0]),
            "coefficient_end": float(trace["obs_coefficient_after_advance"][mask][-1]),
            "corrected_force_mean_n": float(np.mean(trace["obs_corrected_force_n"][mask])),
        })
        rows.append(row)
    return rows


def first_difference(time, left, right):
    delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    magnitude = np.linalg.norm(delta, axis=1) if delta.ndim == 2 else np.abs(delta)
    indices = np.flatnonzero(magnitude > 1e-12)
    return float(time[indices[0]]) if len(indices) else None


def pair_events(before, after):
    """Distinguish internal budget effects from slew-limited requests and later inputs."""
    time = before["time"]
    events = {f"first_{label}_difference_s": first_difference(time, before[field], after[field])
              for label, field in (("cap", "obs_amplitude_capped"),
                                   ("coefficient", "obs_coefficient_after_advance"),
                                   ("request", "requested_tangential_force_world"),
                                   ("measured_velocity", "measured_linear_velocity"),
                                   ("measured_position", "measured_position"),
                                   ("velocity", "linear_velocity"), ("position", "position"))}
    indices = np.flatnonzero(before["obs_amplitude_capped"])
    events["first_6n_cap_s"] = float(time[indices[0]]) if len(indices) else None
    blocked = (before["obs_active"] & before["obs_update_ready"] & before["obs_advance_called"]
               & before["obs_allow_integration"] & before["obs_amplitude_capped"]
               & (before["obs_limited_increment"] > 0))
    indices = np.flatnonzero(blocked)
    events["first_6n_cap_blocks_positive_update_s"] = float(time[indices[0]]) if len(indices) else None
    request = events["first_request_difference_s"]
    if request is None and any(events[f"first_{label}_difference_s"] is not None
                               for label in ("measured_velocity", "measured_position", "velocity", "position")):
        raise ValueError("actor/state differs without a changed request")
    if request is not None:
        for label in ("measured_velocity", "measured_position", "velocity", "position"):
            first = events[f"first_{label}_difference_s"]
            if first is not None and first <= request:
                raise ValueError("actor/state input differs before changed request can act")
        prefix = time <= request
        for field in ("measured_position", "measured_linear_velocity", "target_position", "target_linear_velocity"):
            if not np.array_equal(before[field][prefix], after[field][prefix]):
                raise ValueError(f"actor prefix differs before request takes effect: {field}")
    return events


def collect(directory, *, execute):
    roots = input_archives()
    tables = {name: [] for name in ("comparison", "windows", "events")}
    pending = {}
    for ordinal, (case, scale, budget, name) in enumerate(specifications(), 1):
        expected, parent_name = parent_trace(roots, case, scale, budget)
        trace = observer.run_trial(case, scale, budget).trace if execute else transfer.load_trace(directory / name)
        compact = public.compact(trace)
        for field, values in expected.items():
            if (field not in compact or values.dtype != compact[field].dtype
                    or not np.array_equal(values, compact[field])):
                raise ValueError(f"instrumented run differs from pinned parent: {name}/{field}")
        transfer.validation.validate_public(compact, case, scale, budget)
        checks = validation.validate_observation(trace, parameters(budget))
        context = {"case_index": case["case_index"], "rotation_gain_scale": scale, "max_force_n": budget,
                   "surface_yaw_deg": case["scenario"].wall_yaw_deg, "trace_path": name,
                   "reference_trace": parent_name, "new_simulation": True}
        tables["comparison"].append({**context, **public.metrics(trace, case), **checks})
        tables["windows"].extend({**context, **row} for row in window_rows(trace))
        identity = (case["case_index"], scale)
        if budget == 6.:
            pending[identity] = trace
        else:
            before = pending.pop(identity)
            tables["events"].append({
                "case_index": identity[0], "rotation_gain_scale": scale,
                **pair_events(before, trace),
            })
        if execute:
            np.savez_compressed(directory / name, **trace)
            print(f"{ordinal}/8 instrumented and parent-matched: {name}", flush=True)
    return tables


def audit_archive(directory):
    root = Path(directory).absolute()
    manifest = transfer.verify_archive(root)
    expected_files = {name for _, _, _, name in specifications()} | {
        "protocol.json", "source_hashes.json", "comparison.csv", "windows.csv", "events.csv"}
    if manifest.get("identity") != IDENTITY or manifest.get("new_simulations") != 8:
        raise ValueError("onset archive identity/count differs")
    if manifest.get("parents") != PARENTS or set(manifest["artifact_sha256"]) != expected_files:
        raise ValueError("onset archive parents/inventory differs")
    for name, value in (("protocol.json", protocol_document()), ("source_hashes.json", source_identity())):
        if json.loads((root / name).read_text()) != value:
            raise ValueError(f"frozen input differs: {name}")
    tables = collect(root, execute=False)
    for name, rows in tables.items():
        transfer.check_table(root / f"{name}.csv", rows)
    return {"audit_status": "PASS", "new_simulations": 8, "parent_matched_traces": 8,
            "observer_cycles": sum(r["validated_cycles"] for r in tables["comparison"]),
            "windows": len(tables["windows"]), "pairs": len(tables["events"]), "default_changed": False}


def generate(directory):
    output = transfer.rotation._output_path(directory)
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("output must not contain symlinks")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    input_archives()
    sources, protocol = source_identity(), protocol_document()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".onset-observer-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        (staging / "traces").mkdir(parents=True)
        _write_json(staging / "protocol.json", protocol)
        _write_json(staging / "source_hashes.json", sources)
        tables = collect(staging, execute=True)
        if source_identity() != sources:
            raise ValueError("sources changed during instrumented execution")
        for name, rows in tables.items():
            _write_csv(staging / f"{name}.csv", rows)
        manifest = {"identity": IDENTITY, "new_simulations": 8, "parents": PARENTS,
                    "versions": {"python": platform.python_version(), "numpy": np.__version__},
                    "artifact_sha256": {p.relative_to(staging).as_posix(): _sha256(p)
                                        for p in sorted(staging.rglob("*")) if p.is_file()}}
        _write_json(staging / "manifest.json", manifest)
        (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
        audit_archive(staging)
        if output.exists():
            raise FileExistsError("output appeared during execution")
        os.rename(staging, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--audit", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_archive(args.audit)) if args.audit else generate(args.output))


if __name__ == "__main__":
    main()
