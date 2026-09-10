"""Fixed budget-transfer regression, with explicit pinned-reference row origins.

Execute 60 missing configurations; evaluate 120 rows without duplicating archived
reference trajectories. The audit recomputes tables from raw arrays and checks
geometry and telemetry independently of the execution adapters; it does not replay dynamics.
"""

import argparse
import csv
import json
import os
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from compliant_control_lab.online_compensation_experiment import (
    _sha256,
    _write_csv,
    _write_json,
)
from tools import budget_gain_interaction as dynamic
from tools import budget_public24 as public
from tools import budget_transfer_screen as screening
from tools import budget_transfer_validation as validation
from tools import combined_residual_ablation as old
from tools import compensation_budget_study as prior
from tools import rotation_gain_regression as rotation
from tools.audit_combined_residual_ablation import independent_diagnostics
from tools.audit_online_compensation_errors import _cell
from tools.combined_residual_validation import check_unchanged_prefix

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "compensation-budget-transfer-v1"
TABLES = ("public24", "dynamic", "phases", "diagnostics")


def safe_path(root, relative):
    name = Path(relative)
    if name.is_absolute() or not name.parts or ".." in name.parts or name.as_posix() != relative:
        raise ValueError("unsafe archive artifact")
    path = root / name
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("archive must not contain symlinks")
    return path


def verify_archive(directory, expected_digest=None):
    root = Path(directory).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("archive must not contain symlinks")
    digest = _sha256(root / "manifest.json")
    if expected_digest is not None and digest != expected_digest:
        raise ValueError("pinned reference manifest differs")
    if (root / "COMPLETE").read_text().strip() != digest:
        raise ValueError("incomplete manifest")
    manifest = json.loads((root / "manifest.json").read_text())
    artifacts = manifest["artifact_sha256"]
    if not artifacts:
        raise ValueError("empty artifact inventory")
    for name, expected in artifacts.items():
        if _sha256(safe_path(root, name)) != expected:
            raise ValueError(f"artifact hash differs: {name}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("archive must not contain symlinks")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(artifacts) | {"manifest.json", "COMPLETE"}:
        raise ValueError("unexpected/missing archive files")
    return manifest


def references():
    result = {}
    for name, module in (("public24", public), ("dynamic", dynamic)):
        ref = module.REFERENCE
        path = ROOT / ref["directory"]
        verify_archive(path, ref["manifest_sha256"])
        result[name] = path
    public_protocol = json.loads((result["public24"] / "protocol.json").read_text())
    if public_protocol["configurations"] != [public.case_document(c) for c in public.cases()]:
        raise ValueError("public reference cases differ")
    for scale in (1., 2.):
        if public_protocol["controller_parameters"][str(int(scale))]["online"] != public.controller_document(scale, 6.):
            raise ValueError("public reference controller constructor differs")
    dynamic_protocol = json.loads((result["dynamic"] / "protocol.json").read_text())
    for variant, case in dynamic.cases():
        document = json.loads(json.dumps(dynamic.case_document(variant, case)))
        matched = [c for c in dynamic_protocol["cases"] if c["variant"] == variant
                   and c["surface_yaw_deg"] == case.scenario.wall_yaw_deg]
        if len(matched) != 2 or any({k: c[k] for k in document} != document for c in matched):
            raise ValueError("dynamic reference cases differ")
    for archived in dynamic_protocol["controllers"]:
        if archived["safe_adaptive_base"] != dynamic.controller_document(1., archived["max_force_n"]):
            raise ValueError("dynamic reference controller constructor differs")
    current_sources = source_identity()
    for path in result.values():
        archived_sources = json.loads((path / "source_hashes.json").read_text())
        for name, digest in archived_sources.items():
            if name.startswith("src/") and current_sources.get(name) != digest:
                raise ValueError(f"reference simulation/controller source differs: {name}")
    return result


def source_identity():
    sources = {**rotation.source_identity(), **prior.source_identity()}
    for module in (__file__, public.__file__, dynamic.__file__, screening.__file__, validation.__file__,
                   ROOT / "tools/audit_rotation_gain_regression.py",
                   ROOT / "tools/audit_online_compensation_errors.py",
                   ROOT / "tools/audit_combined_residual_ablation.py"):
        path = Path(module).resolve()
        sources[path.relative_to(ROOT).as_posix()] = _sha256(path)
    return dict(sorted(sources.items()))


def protocol_document():
    return {
        "identity": IDENTITY, "public_development": True, "new_holdout": False,
        "default_changed": False, "comparison_defined_before_execution": True,
        "evaluated_rows": 120, "new_executions": 60, "pinned_reference_rows": 60,
        "references": {"public24": public.REFERENCE, "dynamic": dynamic.REFERENCE},
        "public24": {"rows": 96, "executed": "48 budget-8 N rows", "reused": "48 budget-6 N rows",
                     "physical_configurations": 12, "noise_seeds": [11, 29],
                     "cases": [public.case_document(c) for c in public.cases()],
                     "same_gain_8_minus_6_limits": screening.PUBLIC_LIMITS,
                     "absolute_candidate_limits": screening.ABSOLUTE,
                     "budget_exercised_required": False},
        "dynamic": {"rows": 24, "executed": "12 gain-2 rows", "reused": "12 gain-1 rows",
                    "cases": [dynamic.case_document(v, c) for v, c in dynamic.cases()],
                    "budget_criteria": screening.budget_screen.CRITERIA,
                    "reduction_boundary_comparison": "100*candidate <= (100-minimum_pct)*reference",
                    "same_budget_gain2_minus_gain1_limits": screening.GAIN_LIMITS,
                    "diagnostic_windows": list(old.DIAGNOSTIC_WINDOWS),
                    "same_budget_no_bias_prefix_end_s": 6.,
                    "interaction_effect": "descriptive difference-in-differences; no effect-size gate"},
        "controllers": [{"rotation_gain_scale": s, "max_force_n": b,
                         "public24": public.controller_document(s, b),
                         "dynamic": dynamic.controller_document(s, b)}
                        for s in (1., 2.) for b in (6., 8.)],
        "limitations": ["Compatibility on fixed public-development cases, not a new blind holdout.",
                        "Dynamic cases share seed 11 and one payload; no statistical robustness claim.",
                        "No baseline/friction comparator reruns or original method-gate promotion.",
                        "No hardware safety, passivity proof, BC/RL training or 8 N C++ full-trace parity."],
    }


def specifications():
    for case in public.cases():
        for scale in (1., 2.):
            for budget in (6., 8.):
                label = f"public24__case{case['case_index']:02d}__s{int(scale)}__f{int(budget)}n"
                yield "public24", None, case, scale, budget, budget == 8., f"traces/{label}.npz"
    for variant, case in dynamic.cases():
        for scale in (1., 2.):
            for budget in (6., 8.):
                label = (f"dynamic__yaw{int(case.scenario.wall_yaw_deg)}__{variant}"
                         f"__s{int(scale)}__f{int(budget)}n")
                yield "dynamic", variant, case, scale, budget, scale == 2., f"traces/{label}.npz"


def load_trace(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def pinned_metric_preflight(refs):
    """Reconstruct all 60 reference rows before spending any new physics execution."""
    def read(path):
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    public_rows = {(int(r["case_index"]), float(r["scale"])): r
                   for r in read(refs["public24"] / "comparison.csv") if r["method"] == "online"}
    dynamic_rows = read(refs["dynamic"] / "comparison.csv")
    phase_rows = read(refs["dynamic"] / "phase_metrics.csv")
    diagnostic_rows = read(refs["dynamic"] / "diagnostics.csv")
    for suite, variant, case, scale, budget, is_new, _ in specifications():
        if is_new:
            continue
        if suite == "public24":
            trace = load_trace(refs[suite] / public.reference_trace(case, scale))
            trace["max_force_n"] = np.array(6.)
            validation.validate_public(trace, case, scale, budget)
            for key, value in public.metrics(trace, case).items():
                _cell(public_rows[case["case_index"], scale], key, value, "pinned public24")
        else:
            trace = load_trace(refs[suite] / dynamic.reference_trace(variant, case, budget))
            validation.validate_dynamic(trace, case, scale, budget)
            expected = dynamic.metrics(trace, case, budget)
            for rebuilt, published, suffix in zip(([expected[0]], expected[1], expected[2]),
                                                  (dynamic_rows, phase_rows, diagnostic_rows),
                                                  (None, "phase", "window"), strict=True):
                candidates = [r for r in published if float(r["surface_yaw_deg"]) == case.scenario.wall_yaw_deg
                              and r["variant"] == variant and float(r["max_force_n"]) == budget]
                for row in rebuilt:
                    matched = [r for r in candidates if suffix is None or r[suffix] == row[suffix]]
                    if len(matched) != 1:
                        raise ValueError("pinned metric identity differs")
                    for key, value in row.items():
                        _cell(matched[0], key, value, "pinned dynamic")


def collect(directory, refs, *, execute):
    tables = {name: [] for name in TABLES}
    previous_requests, combined = {}, {}
    files, new_count = set(), 0
    for suite, variant, case, scale, budget, is_new, name in specifications():
        adapter = public if suite == "public24" else dynamic
        if is_new:
            files.add(name)
            if execute:
                trace = adapter.compact(adapter.run_trial(case, scale, budget).trace)
                # Both compact schemas are validated below. The legacy protocol
                # writer requires dynamic-only fields absent from public24.
                # Files remain private until the complete archive passes audit.
                np.savez_compressed(directory / name, **trace)
                new_count += 1
                print(f"{new_count:02d}/60 executed {name}", flush=True)
            else:
                trace = load_trace(directory / name)
            trace_path = name
        else:
            relative = (public.reference_trace(case, scale) if suite == "public24"
                        else dynamic.reference_trace(variant, case, budget))
            trace = load_trace(safe_path(refs[suite], relative))
            trace_path = f"{adapter.REFERENCE['directory']}/{relative}"
            if suite == "public24":
                # This scalar comes from the pinned 6 N constructor, not an archived observer.
                trace["max_force_n"] = np.array(6.)
        context = {"rotation_gain_scale": scale, "max_force_n": budget,
                   "row_origin": "executed" if is_new else "pinned_reference",
                   "trace_path": trace_path}
        if suite == "public24":
            validation.validate_public(trace, case, scale, budget)
            context.update(case_index=case["case_index"], wall_yaw_deg=case["scenario"].wall_yaw_deg,
                           wall_time_constant_s=case["scenario"].wall_time_constant,
                           tool_mass_kg=case["scenario"].tool_mass_kg,
                           simulation_seed=case["config"].seed)
            tables[suite].append({**context, **public.metrics(trace, case)})
        else:
            validation.validate_dynamic(trace, case, scale, budget)
            yaw = case.scenario.wall_yaw_deg
            context.update(surface_yaw_deg=yaw, variant=variant, case=case.name,
                           simulation_seed=case.config.seed, arm="online")
            whole, phases, diagnostics = dynamic.metrics(trace, case, budget)
            identity = (yaw, variant, scale)
            requested = trace["requested_tangential_force_world"]
            if budget == 6.:
                previous_requests[identity] = requested.copy()
            whole["max_matched_request_difference_n"] = float(np.max(np.linalg.norm(
                requested - previous_requests[identity], axis=1)))
            tables[suite].append({**context, **whole})
            tables["phases"].extend({**context, **row} for row in phases)
            for row, (_, start, end) in zip(diagnostics, old.DIAGNOSTIC_WINDOWS, strict=True):
                independent_diagnostics(trace, case, row, start, end)
                tables["diagnostics"].append({**context, **row})
            if variant == "combined":
                combined[yaw, scale, budget] = trace
            else:
                check_unchanged_prefix(combined[yaw, scale, budget], trace, 6.)
    return tables, files


def decision(tables):
    return screening.screen(*(tables[name] for name in TABLES))


def check_table(path, expected):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if not expected or reader.fieldnames != list(expected[0]):
            raise ValueError(f"table schema differs: {path.name}")
    if len(rows) != len(expected):
        raise ValueError(f"table row count differs: {path.name}")
    for row, target in zip(rows, expected, strict=True):
        for name, value in target.items():
            _cell(row, name, value, path.name)


def audit_archive(directory):
    root = Path(directory).absolute()
    manifest = verify_archive(root)
    for key, value in {"identity": IDENTITY, "evaluated_rows": 120, "new_executions": 60,
                       "pinned_reference_rows": 60, "default_changed": False, "new_holdout": False}.items():
        if manifest.get(key) != value:
            raise ValueError(f"wrong manifest scope/count: {key}")
    if json.loads((root / "protocol.json").read_text()) != json.loads(json.dumps(protocol_document())):
        raise ValueError("frozen protocol differs")
    if json.loads((root / "source_hashes.json").read_text()) != source_identity():
        raise ValueError("source identity differs")
    refs = references()
    pinned_metric_preflight(refs)
    tables, traces = collect(root, refs, execute=False)
    for name in TABLES:
        check_table(root / f"{name}.csv", tables[name])
    rebuilt = decision(tables)
    if json.loads((root / "screening.json").read_text()) != rebuilt:
        raise ValueError("screening differs from trace-derived decisions")
    expected = traces | {f"{n}.csv" for n in TABLES} | {"protocol.json", "source_hashes.json", "screening.json"}
    if set(manifest["artifact_sha256"]) != expected:
        raise ValueError("unexpected artifact inventory")
    inputs = {n: manifest["artifact_sha256"][n] for n in ("protocol.json", "source_hashes.json")}
    if manifest["input_sha256"] != inputs:
        raise ValueError("input hashes not bound to artifacts")
    return {"audit_status": "PASS", "evaluated_rows": 120, "new_executions": 60,
            "pinned_reference_rows": 60, "pre_onset_exact_pairs": 12,
            "public24_passed_pairs": rebuilt["public24"]["passed_pairs"],
            "dynamic_gain_passed_pairs": rebuilt["dynamic"]["passed_gain_pairs"],
            "screening_status": rebuilt["status"], "default_changed": False}


def generate(directory):
    output = rotation._output_path(directory)
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("output must not contain symlinks")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    refs, protocol, sources = references(), protocol_document(), source_identity()
    pinned_metric_preflight(refs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".budget-transfer-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        (staging / "traces").mkdir(parents=True)
        _write_json(staging / "protocol.json", protocol)
        _write_json(staging / "source_hashes.json", sources)
        tables, _ = collect(staging, refs, execute=True)
        if sources != source_identity():
            raise ValueError("sources changed during execution")
        references()
        for name in TABLES:
            _write_csv(staging / f"{name}.csv", tables[name])
        _write_json(staging / "screening.json", decision(tables))
        manifest = {"schema_version": 1, "identity": IDENTITY,
                    "evaluated_rows": 120, "new_executions": 60, "pinned_reference_rows": 60,
                    "default_changed": False, "new_holdout": False,
                    "versions": {"python": platform.python_version(),
                                 **{n: version(n) for n in ("numpy", "mujoco")}},
                    "input_sha256": {n: _sha256(staging / n) for n in ("protocol.json", "source_hashes.json")},
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
