"""Derive configuration-stratified v2 preparation archives without new rollouts."""

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

from compliant_control_lab.surface_dataset import audit_dataset
from compliant_control_lab.surface_experiment import _output_path, _sha256
from compliant_control_lab.surface_readiness_benchmark import (
    _report,
    _summary,
    audit_benchmark,
)
from compliant_control_lab.surface_readiness_cases import PREPARATION_SPLIT
from compliant_control_lab.surface_splits import SPLIT_NAMES, case_group_id, split_cases

START_SEED = 20260906
IDENTITY = "no new rollout; public post-benchmark, pre-learning partition"


def _json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _group_roles(cases):
    roles = {}
    for case in cases:
        group = case_group_id(case)
        anchor = case["scenario"]["name"] in {
            "preparation_000",
            "preparation_001",
            "preparation_002",
        }
        role = "nominal_anchor" if anchor else "joint_configuration"
        if group in roles and roles[group]["role"] != role:
            raise ValueError("physical group mixes anchor and joint roles")
        roles[group] = {"role": role, "split": case["split"]}
    return roles


def choose_partition(cases, start_seed=START_SEED):
    """Search using configuration identities only; no report or metric is accepted."""
    anchor_names = {f"preparation_{index:03d}" for index in range(3)}
    found_names = {case["scenario"]["name"] for case in cases} & anchor_names
    if found_names != anchor_names:
        raise ValueError("all three declared nominal anchors are required")
    candidates = []
    seed = start_seed
    while True:
        assigned = split_cases(cases, seed=seed, **PREPARATION_SPLIT)
        anchor_splits = {
            case["scenario"]["name"]: case["split"]
            for case in assigned
            if case["scenario"]["name"] in anchor_names
        }
        valid = set(anchor_splits.values()) == set(SPLIT_NAMES)
        candidate = {"seed": seed, "anchor_splits": anchor_splits, "condition_met": valid}
        candidates.append(candidate)
        print(json.dumps(candidate, sort_keys=True), flush=True)
        if valid:
            return assigned, candidates
        seed += 1


def _rewrite_dataset(source, destination, assigned, provenance):
    shutil.copytree(source, destination)
    manifest = _load(destination / "manifest.json")
    by_id = {case["case_id"]: case for case in assigned}
    manifest["cases"] = assigned
    manifest["group_split"]["seed"] = provenance["selected_seed"]
    for episode in manifest["episodes"]:
        episode["split"] = by_id[episode["case_id"]]["split"]
    manifest["partition_provenance"] = provenance
    _rewrite_manifest(destination, manifest)


def _rewrite_benchmark(source, destination, assigned, provenance):
    shutil.copytree(source, destination)
    manifest = _load(destination / "manifest.json")
    by_id = {case["case_id"]: case for case in assigned}
    manifest["cases"] = [by_id[case["case_id"]] for case in manifest["cases"]]
    report = _load(destination / "report.json")
    rows = report["runs"]
    for row in rows:
        row["split"] = by_id[row["case_id"]]["split"]
    _write_csv(destination / "comparison.csv", rows)
    (destination / "summary.md").write_text(_summary(rows, manifest["stress"]), encoding="utf-8")
    _write_json(destination / "report.json", _report(rows, manifest["stress"]))
    manifest["partition_provenance"] = provenance
    _rewrite_manifest(destination, manifest)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rewrite_manifest(directory, manifest):
    if "artifact_sha256" in manifest:
        manifest["artifact_sha256"] = {
            path.name: _sha256(path)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name not in {"manifest.json", "COMPLETE"}
        }
    _write_json(directory / "manifest.json", manifest)
    (directory / "COMPLETE").write_text(
        _sha256(directory / "manifest.json") + "\n", encoding="utf-8"
    )


def _evidence_hashes(directory, names):
    return {name: _sha256(directory / name) for name in names}


def repartition_surface_preparation(output_root, *, benchmark, stress, dataset):
    inputs = {
        name: Path(value).absolute()
        for name, value in {"benchmark": benchmark, "stress": stress, "dataset": dataset}.items()
    }
    for name, path in inputs.items():
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise ValueError(f"{name} v1 archive path contains a symlink")
        if not path.is_dir() or any(item.is_symlink() for item in path.iterdir()):
            raise ValueError(f"{name} v1 archive is missing or contains symlink artifacts")
    audits = {
        "benchmark": audit_benchmark(inputs["benchmark"]),
        "stress": audit_benchmark(inputs["stress"]),
        "dataset": audit_dataset(inputs["dataset"]),
    }
    manifests = {name: _load(path / "manifest.json") for name, path in inputs.items()}
    sources = {
        json.dumps(manifest["source_and_assets_sha256"], sort_keys=True)
        for manifest in manifests.values()
    }
    if len(sources) != 1:
        raise ValueError("v1 archives have different source/assets identities")
    cases = manifests["benchmark"]["cases"]
    if manifests["benchmark"]["stress"] or not manifests["stress"]["stress"]:
        raise ValueError("benchmark/stress archive modes are reversed")
    if cases != manifests["dataset"]["cases"]:
        raise ValueError("benchmark and dataset v1 case plans differ")
    if not all(case in cases for case in manifests["stress"]["cases"]):
        raise ValueError("stress cases are not from the full v1 plan")
    assigned, candidates = choose_partition(cases)
    if any(
        {key: value for key, value in original.items() if key != "split"}
        != {key: value for key, value in derived.items() if key != "split"}
        for original, derived in zip(cases, assigned)
    ):
        raise RuntimeError("partition search changed non-split case metadata")
    selected_seed = candidates[-1]["seed"]
    parent_hashes = {name: _sha256(path / "manifest.json") for name, path in inputs.items()}
    provenance = {
        "identity": IDENTITY,
        "metadata_only": True,
        "performance_metrics_used": False,
        "selection_rule": "first seed >= 20260906 placing one named nominal anchor in each split",
        "selected_seed": selected_seed,
        "publisher_script_sha256": _sha256(Path(__file__)),
        "parent_manifest_sha256": parent_hashes,
    }
    output = _output_path(output_root)
    if output.exists():
        raise ValueError("derived v2 root must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark_files = [
        row[key]
        for row in _load(inputs["benchmark"] / "report.json")["runs"]
        for key in ("trace_file", "events_file")
    ]
    stress_files = [
        row[key]
        for row in _load(inputs["stress"] / "report.json")["runs"]
        for key in ("trace_file", "events_file")
    ]
    dataset_files = [entry["file"] for entry in manifests["dataset"]["episodes"]]
    parent_evidence = {
        "benchmark": _evidence_hashes(inputs["benchmark"], benchmark_files),
        "stress": _evidence_hashes(inputs["stress"], stress_files),
        "dataset": _evidence_hashes(inputs["dataset"], dataset_files),
    }
    with tempfile.TemporaryDirectory(prefix=".surface-repartition-", dir=output.parent) as temp:
        staging = Path(temp) / "v2"
        staging.mkdir()
        _rewrite_dataset(inputs["dataset"], staging / "dataset", assigned, provenance)
        _rewrite_benchmark(inputs["benchmark"], staging / "benchmark", assigned, provenance)
        _rewrite_benchmark(inputs["stress"], staging / "stress", assigned, provenance)
        derived_audits = {
            "benchmark": audit_benchmark(staging / "benchmark"),
            "stress": audit_benchmark(staging / "stress"),
            "dataset": audit_dataset(staging / "dataset"),
        }
        for name, hashes in parent_evidence.items():
            if _evidence_hashes(staging / name, hashes) != hashes:
                raise RuntimeError(f"{name} raw evidence changed during metadata derivation")
        partition = {
            "identity": IDENTITY,
            "new_holdout": False,
            "selection_uses_performance": False,
            "start_seed": START_SEED,
            "selected_seed": selected_seed,
            "candidates": candidates,
            "group_roles": _group_roles(assigned),
            "parent_manifest_sha256": parent_hashes,
            "input_audits": audits,
            "derived_audits": derived_audits,
        }
        _write_json(staging / "partition.json", partition)
        if _output_path(output).exists():
            raise ValueError("derived v2 root appeared during repartition")
        os.rename(staging, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = repartition_surface_preparation(
        args.output, benchmark=args.benchmark, stress=args.stress, dataset=args.dataset
    )
    print(json.dumps({"output": str(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
