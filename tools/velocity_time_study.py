"""Single-factor 0.05/0.10 s velocity-error-time experiment on four matched arms."""

import argparse
import json
import os
import platform
import tempfile
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

from compliant_control_lab.online_compensation_experiment import _sha256, _write_csv, _write_json
from tools import budget_public24 as public
from tools import budget_transfer as transfer
from tools import onset_observer_study as onset
from tools import velocity_time_metrics as metrics
from tools import velocity_time_runner as runner

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "public24-velocity-error-time-v1"
REFERENCE = {"directory": "results/franka_onset_observer",
             "manifest_sha256": "e01db28864f04057d5272c4f0e469e2261c5166e5afe877afe2f00ef2d66a2ce"}
NONWORSE_ATOL = 1e-12
WINDOWS = (("onset", 1.5, 2.), ("whole", 1.5, 4.5))


def specifications():
    case = next(c for c in public.cases() if c["case_index"] == 23)
    for scale in (1., 2.):
        for budget in (6., 8.):
            for time in (.05, .10):
                yield case, scale, budget, time, f"traces/case23__s{int(scale)}__f{int(budget)}n__tv{int(1000*time)}ms.npz"


def source_identity():
    result = onset.source_identity()
    for filename in (__file__, runner.__file__, metrics.__file__):
        path = Path(filename).resolve()
        result[path.relative_to(ROOT).as_posix()] = _sha256(path)
    return dict(sorted(result.items()))


def protocol_document():
    case = next(specifications())[0]
    return {
        "identity": IDENTITY, "reference": REFERENCE,
        "new_simulations": 4, "reused_traces": 4, "evaluated_rows": 8,
        "case": public.case_document(case), "gains": [1., 2.], "budgets_n": [6., 8.],
        "velocity_error_times_s": [.05, .10], "only_changed_parameter": "velocity_error_time",
        "default_changed": False, "new_holdout": False,
        "windows": [list(w) for w in WINDOWS], "observer_windows": [list(w) for w in onset.WINDOWS],
        "catchup": {"direction": "unit projected target velocity; true evaluator, target-minus-actual position",
                    "positive_lag_area": "sum(max(signed_along_lag_mm, 0)) * .002 s",
                    "first_entry": "first 50 consecutive samples with abs(along lag) <= 1 mm; start timestamp",
                    "no_entry": None, "interpretation": "100ms sampled first entry, not permanent settling"},
        "constructors": [{"rotation_gain_scale": s, "max_force_n": b, "velocity_error_time_s": t,
                          "constructor": runner.controller_document(s, b, t)} for _, s, b, t, _ in specifications()],
        "decision": {
            "purpose": "eligibility for larger original-grid regression, never direct default promotion",
            "nonworse_tolerance_numeric_only": NONWORSE_ATOL,
            "per_arm_nonworse": ["tangent_velocity_error_rms_m_s"],
            "position_regression_limit_mm": .10,
            "catchup_statistics_are_diagnostic": True,
            "both_gains": "8-minus-6 velocity RMS cost strictly decreases by more than numerical tolerance",
            "original_relative_limits": transfer.screening.PUBLIC_LIMITS,
            "original_absolute_limits": transfer.screening.ABSOLUTE,
            "candidate_budget_compatibility_required": True,
        },
        "limitations": ["One preselected physical case and one seed; not new independent or blind samples.",
                        "No leakage, new gate, gain change, learning or controller-source change.",
                        "Along-position first entry is not geometric path distance or a safety/stability proof."],
    }


def input_archive():
    root = transfer.safe_path(ROOT, REFERENCE["directory"])
    transfer.verify_archive(root, REFERENCE["manifest_sha256"])
    onset.input_archives()
    current = source_identity()
    for name, digest in json.loads((root / "source_hashes.json").read_text()).items():
        if current.get(name) != digest:
            raise ValueError(f"parent source changed: {name}")
    parent = json.loads((root / "protocol.json").read_text())
    for case, scale, budget, time, _ in specifications():
        document = runner.controller_document(scale, budget, time)
        original = public.controller_document(scale, budget)
        expected = deepcopy(original)
        expected["tangential"]["velocity_error_time"] = time
        if document != expected:
            raise ValueError("constructor changes more than velocity_error_time")
        if public.case_document(case) not in parent["cases"]:
            raise ValueError("case differs from pinned parent")
        records = [r for r in parent["controllers"] if r["case_index"] == 23
                   and r["rotation_gain_scale"] == scale and r["max_force_n"] == budget]
        if len(records) != 1 or records[0]["constructor"] != original:
            raise ValueError("parent constructor differs")
    return root


def screen(rows):
    expected = {(s, b, t) for s in (1., 2.) for b in (6., 8.) for t in (.05, .10)}
    lookup = transfer.screening.index(rows, ("rotation_gain_scale", "max_force_n", "velocity_error_time_s"), expected)
    arm_pairs, budget_pairs = [], []
    names = protocol_document()["decision"]["per_arm_nonworse"]
    for scale in (1., 2.):
        for budget in (6., 8.):
            old, new = (lookup[scale, budget, t] for t in (.05, .10))
            deltas = {key: transfer.screening.number(new, key) - transfer.screening.number(old, key)
                      for key in set(names) | set(transfer.screening.PUBLIC_LIMITS)}
            failed = [key for key in names if deltas[key] > NONWORSE_ATOL]
            failed += ["relative_" + key for key, bound in transfer.screening.PUBLIC_LIMITS.items()
                       if deltas[key] > bound]
            failed += transfer.screening.absolute_failures(new)
            arm_pairs.append({"rotation_gain_scale": scale, "max_force_n": budget,
                              "delta_direction": "0.10_minus_0.05", "deltas": dict(sorted(deltas.items())),
                              "failed_checks": failed, "status": "FAIL" if failed else "PASS"})
        costs, decisions = {}, {}
        for time in (.05, .10):
            before, after = (lookup[scale, b, time] for b in (6., 8.))
            deltas = {key: transfer.screening.number(after, key) - transfer.screening.number(before, key)
                      for key in transfer.screening.PUBLIC_LIMITS}
            costs[time] = deltas["tangent_velocity_error_rms_m_s"]
            decisions[str(time)] = transfer.screening.decision({}, deltas, transfer.screening.PUBLIC_LIMITS,
                                                               transfer.screening.absolute_failures(after))
        improved = costs[.10] < costs[.05] - NONWORSE_ATOL
        budget_pairs.append({"rotation_gain_scale": scale, "cost_005_m_s": costs[.05],
                             "cost_010_m_s": costs[.10], "cost_change_m_s": costs[.10] - costs[.05],
                             "velocity_cost_reduced": improved, "compatibility": decisions})
    eligible = (all(p["status"] == "PASS" for p in arm_pairs)
                and all(p["velocity_cost_reduced"] and p["compatibility"]["0.1"]["status"] == "PASS"
                        for p in budget_pairs))
    return {"arm_pairs": arm_pairs, "budget_pairs": budget_pairs, "eligible_for_full_grid": eligible,
            "decision": "expand_original_grid" if eligible else "do_not_expand",
            "default_changed": False, "new_simulations": 4, "reused_traces": 4}


def collect(directory, *, execute):
    parent = input_archive()
    tables = {name: [] for name in ("comparison", "catchup", "observer_windows", "events")}
    previous = {}
    for case, scale, budget, time, filename in specifications():
        relative = f"traces/case23__s{int(scale)}__f{int(budget)}n.npz"
        original = transfer.load_trace(transfer.safe_path(parent, relative))
        reused = time == .05
        if reused:
            trace = original
            path = f"{REFERENCE['directory']}/{relative}"
        else:
            trace = runner.run_trial(case, scale, budget, time).trace if execute else transfer.load_trace(directory / filename)
            path = filename
            if np.asarray(trace.get("velocity_error_time_s")).shape != () or float(trace["velocity_error_time_s"]) != time:
                raise ValueError("velocity-time trace metadata differs")
        compact = public.compact(trace)
        transfer.validation.validate_public(compact, case, scale, budget)
        checks = onset.validation.validate_observation(trace, runner.parameters(budget, time))
        for field in ("time", "target_position", "target_linear_velocity", "target_normal_force"):
            if not np.array_equal(trace[field], original[field]):
                raise ValueError(f"paired schedule differs: {field}")
        context = {"case_index": 23, "rotation_gain_scale": scale, "max_force_n": budget,
                   "velocity_error_time_s": time, "trace_path": path,
                   "row_origin": "pinned_reference" if reused else "executed"}
        angle = np.deg2rad(case["scenario"].wall_yaw_deg)
        normal = np.array([np.cos(angle), np.sin(angle), 0.])
        catchup = [{"window": label, **metrics.catchup_metrics(trace, normal, start, end)}
                   for label, start, end in WINDOWS]
        public_metrics = public.metrics(trace, case)
        for key in ("tangent_rmse_mm", "tangent_velocity_error_rms_m_s"):
            if not np.isclose(public_metrics[key], catchup[1][key], rtol=0, atol=1e-12):
                raise ValueError(f"independent full metric differs: {key}")
        tables["comparison"].append({**context, **public_metrics, **checks,
                                     **{k: v for k, v in catchup[1].items() if k not in {"window", "start_s", "end_s", "samples"}},
                                     "onset_positive_along_lag_area_mm_s": catchup[0]["positive_along_lag_area_mm_s"]})
        tables["catchup"].extend({**context, **row} for row in catchup)
        tables["observer_windows"].extend({**context, **row} for row in onset.window_rows(trace))
        if reused:
            previous[scale, budget] = trace
        else:
            events = onset.pair_events(previous[scale, budget], trace)
            events["first_reference_cap_s"] = events.pop("first_6n_cap_s")
            events["first_reference_cap_blocks_positive_update_s"] = events.pop("first_6n_cap_blocks_positive_update_s")
            tables["events"].append({**context, **events})
            if execute:
                np.savez_compressed(directory / filename, **trace)
                print(f"executed case23 s{scale:g} {budget:g}N Tv={time:g}s", flush=True)
    return tables


def audit_archive(directory):
    root = Path(directory).absolute()
    manifest = transfer.verify_archive(root)
    files = {name for _, _, _, t, name in specifications() if t == .10} | {
        "protocol.json", "source_hashes.json", "screening.json", "comparison.csv", "catchup.csv", "observer_windows.csv", "events.csv"}
    if (manifest.get("identity") != IDENTITY or manifest.get("new_simulations") != 4
            or manifest.get("reference") != REFERENCE or set(manifest["artifact_sha256"]) != files):
        raise ValueError("velocity-time archive identity, reference or inventory differs")
    for name, value in (("protocol.json", protocol_document()), ("source_hashes.json", source_identity())):
        if json.loads((root / name).read_text()) != value:
            raise ValueError(f"frozen input differs: {name}")
    tables = collect(root, execute=False)
    for name, rows in tables.items():
        transfer.check_table(root / f"{name}.csv", rows)
    rebuilt = screen(tables["comparison"])
    onset.velocity.check_summary(json.loads((root / "screening.json").read_text()), rebuilt)
    return {"audit_status": "PASS", "new_simulations": 4, "reused_traces": 4,
            "evaluated_rows": 8, "decision": rebuilt["decision"], "default_changed": False}


def generate(directory):
    output = transfer.rotation._output_path(directory)
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("output must not contain symlinks")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    input_archive()
    sources, protocol = source_identity(), protocol_document()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".velocity-time-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        (staging / "traces").mkdir(parents=True)
        _write_json(staging / "protocol.json", protocol)
        _write_json(staging / "source_hashes.json", sources)
        tables = collect(staging, execute=True)
        if source_identity() != sources:
            raise ValueError("sources changed during execution")
        for name, rows in tables.items():
            _write_csv(staging / f"{name}.csv", rows)
        _write_json(staging / "screening.json", screen(tables["comparison"]))
        manifest = {"identity": IDENTITY, "reference": REFERENCE, "new_simulations": 4,
                    "versions": {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__},
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
