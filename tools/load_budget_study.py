"""Run or independently audit the predeclared measured-load budget pilot."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from compliant_control_lab.online_compensation_experiment import _sha256, _write_json
from compliant_control_lab.surface_simulation import yaw_frame
from tools import budget_gain_interaction as dynamic
from tools import budget_public24 as public
from tools import budget_transfer as transfer
from tools import load_budget_screen as screening
from tools import load_budget_trial as runner
from tools import load_budget_validation as validation
from tools.velocity_time_metrics import catchup_metrics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("load_budget_protocol.json")


def protocol_document():
    return json.loads(PROTOCOL.read_text())


def source_identity():
    sources = transfer.source_identity()
    for path in sorted((ROOT / "tools").glob("load_*.*")):
        if path.suffix in {".py", ".json"}:
            sources[path.relative_to(ROOT).as_posix()] = _sha256(path)
    sources["tools/velocity_time_metrics.py"] = _sha256(ROOT / "tools/velocity_time_metrics.py")
    return dict(sorted(sources.items()))


def references():
    for directory, digest in protocol_document()["parent_manifest_sha256"].items():
        root = ROOT / directory
        transfer.verify_archive(root, digest)
        for name, expected in json.loads((root / "source_hashes.json").read_text()).items():
            if _sha256(ROOT / name) != expected:
                raise ValueError(f"parent source changed: {name}")
    return transfer.references()


def dynamic_reference(case, scale):
    if scale == 1:
        return ROOT / dynamic.REFERENCE["directory"] / dynamic.reference_trace("combined", case, 6.)
    yaw = int(case.scenario.wall_yaw_deg)
    return ROOT / f"results/franka_budget_transfer/traces/dynamic__yaw{yaw}__combined__s2__f6n.npz"


def exact_reference_check(trace, reference):
    # The extra sensor/observer arrays do not change any original archived field.
    checked = []
    for name, expected in reference.items():
        if (name not in trace or np.asarray(trace[name]).dtype != expected.dtype
                or not np.array_equal(trace[name], expected)):
            raise ValueError(f"fixed-budget physics reproduction differs: {name}")
        checked.append(name)
    return {"checked_fields": sorted(checked), "cycles": len(trace["time"]), "bit_exact": True}


def paired_schedule(trace, reference):
    names = ("time", "target_position", "target_linear_velocity", "target_normal_force")
    names += tuple(name for name in (
        "applied_wall_friction", "applied_tool_friction", "applied_raw_wrench_bias_world",
        "feedback_raw_wrench_bias_world", "controller_yaw_error_deg", "trajectory_rate_scale",
    ) if name in reference)
    for name in names:
        if name not in trace:
            raise ValueError(f"missing paired schedule: {name}")
        if not np.array_equal(trace[name], reference[name]):
            raise ValueError(f"paired schedule differs: {name}")


def _trace(directory, name, execute, trial, compact):
    path = directory / "traces" / f"{name}.npz"
    if execute:
        trace = compact(trial().trace)
        np.savez_compressed(path, **trace)
        print(f"executed {name}", flush=True)
        return trace
    return transfer.load_trace(path)


def collect(directory, *, execute):
    references()
    cases = [c for variant, c in dynamic.cases() if variant == "combined"]
    public_cases = {c["case_index"]: c for c in public.cases()}
    controls, rows, public_rows, audits = [], [], [], []
    case = next(c for c in cases if c.scenario.wall_yaw_deg == 0)
    for suite, scale, item, path, trial, compact, frame in (
        ("dynamic", 1., case, dynamic_reference(case, 1.),
         lambda: runner.run_dynamic(case, 1., 6., 6.), runner.compact_dynamic,
         yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg)),
        ("public", 2., public_cases[23],
         ROOT / public.REFERENCE["directory"] / public.reference_trace(public_cases[23], 2.),
         lambda: runner.run_public(public_cases[23], 2., 6., 6.), runner.compact_public,
         yaw_frame(public_cases[23]["task"].yaw_deg)),
    ):
        trace = _trace(directory, f"control__{suite}__s{int(scale)}__fixed6", execute, trial, compact)
        checks = exact_reference_check(trace, transfer.load_trace(path))
        checks.update(validation.validate_trace(trace, frame, minimum_force=6., max_force=6.))
        controls.append({"suite": suite, "rotation_gain_scale": scale, **checks})
    for case in cases:
        yaw = int(case.scenario.wall_yaw_deg)
        for scale in (1., 2.):
            trace = _trace(directory, f"dynamic__yaw{yaw}__combined__s{int(scale)}", execute,
                           lambda c=case, s=scale: runner.run_dynamic(c, s), runner.compact_dynamic)
            original = transfer.load_trace(dynamic_reference(case, scale))
            paired_schedule(trace, original)
            checks = validation.validate_trace(trace, yaw_frame(
                case.task.yaw_deg + case.controller_yaw_error_deg))
            identity = {"surface_yaw_deg": yaw, "rotation_gain_scale": scale, "variant": "combined"}
            audits.append({**identity, **checks})
            entries = {}
            for label, data, budget in (("candidate", trace, 8.), ("reference", original, 6.)):
                whole, phases, windows = dynamic.metrics(data, case, budget)
                entries[label] = {"overall": whole, "phases": phases, "windows": windows}
            exercised = bool(np.any(
                (trace["load_budget_applied_n"] > 6. + 1e-9)
                & (trace["controller_coefficient_before_compute"]
                   * trace["diagnostic_corrected_force_n"] > 6. + 1e-9)))
            rows.append({**identity, **entries, "budget_exercised": exercised})
    stage_a = screening.dynamic_screen(rows)
    if stage_a["status"] == "PASS":
        for index in protocol_document()["stage_b"]["public_case_indices"]:
            for scale in (1., 2.):
                case = public_cases[index]
                trace = _trace(directory, f"public__case{index}__s{int(scale)}", execute,
                               lambda c=case, s=scale: runner.run_public(c, s), runner.compact_public)
                original = transfer.load_trace(ROOT / public.REFERENCE["directory"]
                                               / public.reference_trace(case, scale))
                paired_schedule(trace, original)
                frame = yaw_frame(case["task"].yaw_deg)
                audits.append({"case_index": index, "rotation_gain_scale": scale,
                               **validation.validate_trace(trace, frame)})
                normal = yaw_frame(case["scenario"].wall_yaw_deg).rotation[:, 0]
                public_rows.append({
                    "case_index": index, "rotation_gain_scale": scale,
                    "candidate": public.metrics(trace, case), "reference": public.metrics(original, case),
                    "catchup_candidate": catchup_metrics(trace, normal),
                    "catchup_reference": catchup_metrics(original, normal),
                })
        stage_b = screening.public_screen(public_rows)
    else:
        stage_b = {"status": "NOT_RUN", "reason": "stage_a_failed"}
    return {"controls": controls, "dynamic": rows, "public": public_rows, "audits": audits,
            "screening": {"stage_a": stage_a, "stage_b": stage_b,
                          "eligible_for_expansion": stage_a["status"] == stage_b["status"] == "PASS",
                          "default_changed": False, "new_holdout": False}}


def audit_archive(directory):
    directory = Path(directory).absolute()
    manifest = transfer.verify_archive(directory)
    if manifest.get("identity") != protocol_document()["identity"]:
        raise ValueError("archive identity differs")
    for name, expected in (("protocol.json", protocol_document()), ("source_hashes.json", source_identity())):
        if json.loads((directory / name).read_text()) != expected:
            raise ValueError(f"frozen input differs: {name}")
    regenerated = collect(directory, execute=False)
    if json.loads((directory / "comparison.json").read_text()) != regenerated:
        raise ValueError("recomputed comparison differs")
    expected = {"protocol.json", "source_hashes.json", "comparison.json"}
    expected |= {"traces/control__dynamic__s1__fixed6.npz", "traces/control__public__s2__fixed6.npz"}
    expected |= {f"traces/dynamic__yaw{yaw}__combined__s{s}.npz" for yaw in (-15, 0, 15) for s in (1, 2)}
    if regenerated["screening"]["stage_a"]["status"] == "PASS":
        expected |= {f"traces/public__case{i}__s{s}.npz"
                     for i in protocol_document()["stage_b"]["public_case_indices"] for s in (1, 2)}
    if set(manifest["artifact_sha256"]) != expected:
        raise ValueError("unexpected or missing pilot artifact")
    return regenerated["screening"]


def run(directory):
    destination = Path(directory).absolute()
    if destination.exists() or any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError("output must not exist or traverse a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    print(f"staging {directory}; incomplete runs remain here for diagnosis", flush=True)
    (directory / "traces").mkdir()
    source = source_identity()
    _write_json(directory / "protocol.json", protocol_document())
    _write_json(directory / "source_hashes.json", source)
    result = collect(directory, execute=True)
    if source_identity() != source:
        raise ValueError("source changed during execution")
    _write_json(directory / "comparison.json", result)
    artifacts = {p.relative_to(directory).as_posix(): _sha256(p)
                 for p in sorted(directory.rglob("*")) if p.is_file()}
    _write_json(directory / "manifest.json", {
        "identity": protocol_document()["identity"], "artifact_sha256": artifacts,
        "default_changed": False, "new_holdout": False,
    })
    (directory / "COMPLETE").write_text(_sha256(directory / "manifest.json") + "\n")
    audited = audit_archive(directory)
    if destination.exists():
        raise ValueError("output appeared during execution; retained staging archive")
    directory.rename(destination)
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
