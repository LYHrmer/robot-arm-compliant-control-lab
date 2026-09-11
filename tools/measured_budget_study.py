"""Complete the frozen 42-run measured-load expansion and audit all 60 candidates."""

import argparse
import json
import platform
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from compliant_control_lab.online_compensation_experiment import _sha256, _write_json
from compliant_control_lab.surface_simulation import yaw_frame
from tools import budget_gain_interaction as dynamic
from tools import budget_public24 as public
from tools import budget_transfer as transfer
from tools import load_budget_study as pilot
from tools import load_budget_trial as runner
from tools import load_budget_validation as validation
from tools import measured_budget_screen as screening
from tools.velocity_time_metrics import catchup_metrics

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "measured-load-budget-full-regression-v1"
PILOT = {"directory": "results/franka_load_budget_pilot",
         "manifest_sha256": "cfa79734700c42585be364a3016efc5290b8761db60f25631231efb9de68bf4f"}


def specifications():
    result = []
    for variant, case in dynamic.cases():
        yaw = int(case.scenario.wall_yaw_deg)
        for scale in (1., 2.):
            result.append({"suite": "dynamic", "case": case, "scale": scale,
                           "surface_yaw_deg": yaw, "variant": variant, "reused": variant == "combined",
                           "trace_name": f"dynamic__yaw{yaw}__{variant}__s{int(scale)}.npz"})
    for case in public.cases():
        index = case["case_index"]
        for scale in (1., 2.):
            result.append({"suite": "public", "case": case, "scale": scale, "case_index": index,
                           "reused": index in (6, 7, 14, 15, 22, 23),
                           "trace_name": f"public__case{index}__s{int(scale)}.npz"})
    return result


def reference_path(spec):
    if spec["suite"] == "public":
        return ROOT / public.REFERENCE["directory"] / public.reference_trace(spec["case"], spec["scale"])
    if spec["scale"] == 1.:
        return ROOT / dynamic.REFERENCE["directory"] / dynamic.reference_trace(spec["variant"], spec["case"], 6.)
    name = f"dynamic__yaw{spec['surface_yaw_deg']}__{spec['variant']}__s2__f6n.npz"
    return ROOT / "results/franka_budget_transfer/traces" / name


def source_identity():
    sources = pilot.source_identity()
    for filename in (__file__, screening.__file__):
        path = Path(filename).resolve()
        sources[path.relative_to(ROOT).as_posix()] = _sha256(path)
    return dict(sorted(sources.items()))


def protocol_document():
    configurations = []
    for spec in specifications():
        document = (public.case_document(spec["case"]) if spec["suite"] == "public"
                    else dynamic.case_document(spec["variant"], spec["case"]))
        configurations.append({**{k: v for k, v in spec.items() if k != "case"},
                               "configuration": document,
                               "reference_trace": reference_path(spec).relative_to(ROOT).as_posix()})
    document = {
        "identity": IDENTITY, "pilot": PILOT,
        "reference_manifests": pilot.protocol_document()["parent_manifest_sha256"],
        "parameters": pilot.protocol_document()["parameters"],
        "new_simulations": 42, "reused_pilot_candidates": 18, "candidate_rows": 60,
        "new_public_runs": 36, "new_no_bias_runs": 6,
        "public_limits": dict(screening.PUBLIC_LIMITS), "dynamic_limits": dict(screening.CRITERIA),
        "gain_limits": dict(screening.GAIN_LIMITS), "absolute_limits": dict(screening.ABSOLUTE),
        "configurations": configurations,
        "adoption_rule": "All 48 public pairs, 12 dynamic pairs and 6 gain interactions must pass unchanged gates.",
        "if_pass": "Recommend the unchanged measured-load preset for the covered public development configurations.",
        "if_fail": "Do not promote; preserve all failures and both variants without retuning this round.",
        "global_default_changed": False, "new_holdout": False,
        "catchup_statistics": "Descriptive versus fixed 6 N only; no posthoc catchup acceptance gate.",
        "paired_prefix": "Combined and no_bias must have bit-exact cycle arrays before the first bias change at 6 s.",
        "limitations": [
            "Candidate adds measured tangential-force input; not an equal-input single-parameter comparison.",
            "Public configurations already examined; no unseen-data claim.",
            "Combined/no_bias uses seed 11; not all historical dynamic profiles or C++/BC/PPO baselines.",
            "No global-default replacement, physical hardware safety, passivity or identified-friction claim.",
            "Compact traces do not support full measured-state coefficient replay; report skipped audit explicitly.",
        ],
    }
    return json.loads(json.dumps(document))


def input_archive():
    path = ROOT / PILOT["directory"]
    for directory, digest in protocol_document()["reference_manifests"].items():
        transfer.verify_archive(ROOT / directory, digest)
    transfer.verify_archive(path, PILOT["manifest_sha256"])
    if not pilot.audit_archive(path)["eligible_for_expansion"]:
        raise ValueError("pilot is not eligible for expansion")
    return path


def validate_candidate(spec, trace, reference):
    """Bind stored metadata, schedules and physical limits before scheduler replay."""
    case, scale = spec["case"], spec["scale"]
    for name, expected in (("rotation_gain_scale", scale), ("max_force_n", 8.)):
        value = np.asarray(trace.get(name))
        if value.shape != () or value.dtype.kind not in "iuf" or float(value) != expected:
            raise ValueError(f"candidate identity mismatch: {name}")
    pilot.paired_schedule(trace, reference)
    if spec["suite"] == "public":
        compact = runner.compact_public(trace)
        transfer.validation.validate_public(public.compact(trace), case, scale, 8.)
        frame = yaw_frame(case["task"].yaw_deg)
        count = round(case["config"].duration / case["config"].timestep)
    else:
        compact = runner.compact_dynamic(trace)
        kind = np.asarray(trace.get("controller_kind"))
        if kind.shape != () or kind.dtype.kind not in "US" or kind.item() != "surface_online":
            raise ValueError("wrong dynamic controller identity")
        frame = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg)
        count = round(case.config.duration / case.config.timestep)
        limits = np.broadcast_to([87., 87., 87., 87., 12., 12., 12.], (count, 7))
        for name, expected in (("lower_torque_limit", -limits), ("upper_torque_limit", limits),
                               ("applied_torque", np.clip(trace["commanded_torque"], -limits, limits))):
            transfer.validation._close(trace[name], expected, name)
        normal = yaw_frame(case.scenario.wall_yaw_deg).rotation[:, 0]
        gap = (np.array([.4, 0., 0.]) - trace["position"]) @ normal - .025
        transfer.validation._close(trace["true_contact_gap_m"], gap, "contact geometry", atol=1e-12)
    if set(compact) != set(trace) or np.asarray(trace["time"]).shape != (count,):
        raise ValueError("unexpected compact schema or cycle count")
    for value in trace.values():
        array = np.asarray(value)
        if array.dtype.kind not in "biufUS" or (array.dtype.kind in "biuf" and not np.all(np.isfinite(array))):
            raise ValueError("nonfinite or unsupported candidate array")
        if array.dtype.kind == "f" and array.dtype != np.dtype("float64"):
            raise ValueError("continuous candidate arrays must be float64")
    for name in set(trace) - {"case_index", "slew_reconstruction_mismatch_cycles"}:
        if np.asarray(trace[name]).dtype.kind in "iu":
            raise ValueError(f"unexpected integer candidate field: {name}")
    return validation.validate_trace(trace, frame)


def unchanged_prefix(combined, no_bias):
    count = len(combined["time"])
    mask = np.asarray(combined["time"]) < 6.
    fields = []
    if set(combined) != set(no_bias):
        raise ValueError("paired dynamic schemas differ")
    for name, before in combined.items():
        before, after = np.asarray(before), np.asarray(no_bias[name])
        if before.ndim and before.shape[0] == count:
            if (after.shape != before.shape or after.dtype != before.dtype
                    or not np.array_equal(before[mask], after[mask])):
                raise ValueError(f"pre-bias prefix differs: {name}")
            fields.append(name)
    return {"samples": int(mask.sum()), "fields": sorted(fields), "bit_exact": True}


def collect(directory, *, execute):
    parent = input_archive()
    rows = {"dynamic": [], "public": []}
    audits, prefixes, executed = [], [], 0
    for spec in specifications():
        suite, case, scale = spec["suite"], spec["case"], spec["scale"]
        path = (parent if spec["reused"] else directory) / "traces" / spec["trace_name"]
        if execute and not spec["reused"]:
            trial = runner.run_public if suite == "public" else runner.run_dynamic
            compact = runner.compact_public if suite == "public" else runner.compact_dynamic
            trace = compact(trial(case, scale).trace)
            np.savez_compressed(path, **trace)
            executed += 1
            print(f"executed {executed}/42 {spec['trace_name']}", flush=True)
        else:
            trace = transfer.load_trace(path)
        reference = transfer.load_trace(reference_path(spec))
        checks = validate_candidate(spec, trace, reference)
        if suite == "dynamic" and spec["variant"] == "no_bias":
            name = f"dynamic__yaw{spec['surface_yaw_deg']}__combined__s{int(scale)}.npz"
            combined = transfer.load_trace(parent / "traces" / name)
            prefixes.append({"surface_yaw_deg": spec["surface_yaw_deg"],
                             "rotation_gain_scale": scale, **unchanged_prefix(combined, trace)})
        identity = {"rotation_gain_scale": scale,
                    **({"case_index": spec["case_index"]} if suite == "public" else {
                        "surface_yaw_deg": spec["surface_yaw_deg"], "variant": spec["variant"]})}
        origin = {"trace_path": (f"{PILOT['directory']}/traces/{spec['trace_name']}" if spec["reused"]
                                 else f"traces/{spec['trace_name']}"),
                  "reference_path": reference_path(spec).relative_to(ROOT).as_posix(),
                  "row_origin": "pinned_pilot" if spec["reused"] else "executed"}
        audits.append({**identity, **origin, **checks,
                       "maximum_applied_budget_n": float(np.max(trace["load_budget_applied_n"])),
                       "extra_budget_cycles": int(np.count_nonzero(trace["load_budget_applied_n"] > 6. + 1e-9))})
        if suite == "public":
            normal = yaw_frame(case["scenario"].wall_yaw_deg).rotation[:, 0]
            rows[suite].append({**identity, **origin,
                                "candidate": public.metrics(trace, case), "reference": public.metrics(reference, case),
                                "catchup_candidate": catchup_metrics(trace, normal),
                                "catchup_reference": catchup_metrics(reference, normal)})
        else:
            entries = {}
            for label, data, budget in (("candidate", trace, 8.), ("reference", reference, 6.)):
                whole, phases, windows = dynamic.metrics(data, case, budget)
                entries[label] = {"overall": whole, "phases": phases, "windows": windows}
            exercised = bool(np.any((trace["load_budget_applied_n"] > 6. + 1e-9)
                                    & (trace["controller_coefficient_before_compute"]
                                       * trace["diagnostic_corrected_force_n"] > 6. + 1e-9)))
            rows[suite].append({**identity, **origin, **entries, "budget_exercised": exercised})
    if execute and executed != 42:
        raise ValueError("wrong number of new simulations")
    return {**rows, "audits": audits, "prefixes": prefixes,
            "screening": screening.screen(rows["dynamic"], rows["public"])}


def expected_artifacts():
    return {"protocol.json", "source_hashes.json", "comparison.json"} | {
        f"traces/{spec['trace_name']}" for spec in specifications() if not spec["reused"]}


def audit_archive(directory):
    directory = Path(directory).absolute()
    manifest = transfer.verify_archive(directory)
    if (manifest.get("identity") != IDENTITY or manifest.get("pilot") != PILOT
            or manifest.get("new_simulations") != 42 or manifest.get("reused_candidates") != 18
            or set(manifest["artifact_sha256"]) != expected_artifacts()):
        raise ValueError("full-regression archive identity, counts or inventory differs")
    for name, expected in (("protocol.json", protocol_document()), ("source_hashes.json", source_identity())):
        if json.loads((directory / name).read_text()) != expected:
            raise ValueError(f"frozen input differs: {name}")
    result = collect(directory, execute=False)
    if json.loads((directory / "comparison.json").read_text()) != result:
        raise ValueError("recomputed full-regression comparison differs")
    return result["screening"]


def run(directory):
    destination = Path(directory).absolute()
    if destination.exists() or any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError("output must not exist or traverse a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    print(f"staging {staging}; incomplete runs are retained here", flush=True)
    (staging / "traces").mkdir()
    source = source_identity()
    _write_json(staging / "protocol.json", protocol_document())
    _write_json(staging / "source_hashes.json", source)
    result = collect(staging, execute=True)
    if source_identity() != source:
        raise ValueError("source changed during execution")
    _write_json(staging / "comparison.json", result)
    artifacts = {name: _sha256(staging / name) for name in sorted(expected_artifacts())}
    _write_json(staging / "manifest.json", {
        "identity": IDENTITY, "pilot": PILOT, "new_simulations": 42, "reused_candidates": 18,
        "artifact_sha256": artifacts,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__},
    })
    (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
    audited = audit_archive(staging)
    if destination.exists():
        raise ValueError("output appeared during execution; staging retained")
    staging.rename(destination)
    return audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "audit"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = run(args.directory) if args.action == "run" else audit_archive(args.directory)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
