"""Publish a compact, audited index of complete surface-preparation archives."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_dataset import audit_dataset
from compliant_control_lab.surface_experiment import _output_path, _sha256
from compliant_control_lab.surface_readiness_benchmark import audit_benchmark

_BENCHMARK_FILES = ("manifest.json", "report.json", "comparison.csv", "summary.md")


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _same_array(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    if actual.dtype.kind in "fc":
        return np.array_equal(actual, expected, equal_nan=True)
    return np.array_equal(actual, expected)


def _teacher_trace_checks(benchmark, dataset, benchmark_manifest, dataset_manifest):
    rows = _load_json(benchmark / "report.json")["runs"]
    teacher_rows = [row for row in rows if row["method"] == "friction_teacher_50hz"]
    if len(teacher_rows) != len(benchmark_manifest["cases"]):
        raise ValueError("full benchmark lacks exactly one friction teacher run per case")
    entries = {entry["case_id"]: entry for entry in dataset_manifest["episodes"]}
    compared_fields = None
    for row in teacher_rows:
        case = benchmark_manifest["cases"][row["case_index"]]
        entry = entries.get(case["case_id"])
        if entry is None or row["case_id"] != case["case_id"]:
            raise ValueError("teacher benchmark/dataset case mismatch")
        if row["simulation_seed"] != entry["simulation_seed"]:
            raise ValueError("teacher benchmark/dataset actual simulation seed mismatch")
        with np.load(benchmark / row["trace_file"], allow_pickle=False) as benchmark_npz:
            benchmark_trace = {name: benchmark_npz[name] for name in benchmark_npz.files}
        with np.load(dataset / entry["file"], allow_pickle=False) as dataset_npz:
            dataset_trace = {
                name[7:]: dataset_npz[name]
                for name in dataset_npz.files
                if name.startswith("trace__")
            }
        shared = sorted(benchmark_trace.keys() & dataset_trace.keys())
        if not shared:
            raise ValueError("teacher benchmark/dataset have no shared trace fields")
        if compared_fields is None:
            compared_fields = shared
        elif shared != compared_fields:
            raise ValueError("shared teacher trace schema varies by case")
        for name in shared:
            if not _same_array(benchmark_trace[name], dataset_trace[name]):
                raise ValueError(
                    f"teacher benchmark/dataset trace mismatch: {case['case_id']}:{name}"
                )
    return {"cases": len(teacher_rows), "shared_trace_fields": compared_fields}


def _group_checks(rows, cases):
    """Noise replicas are not independent physical/task groups."""
    grouped = []
    for method, group in sorted({(row["method"], row["group_id"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method and row["group_id"] == group]
        means = {}
        for name in ("force_rmse_n", "tangent_rmse_mm", "contact_ratio_pct", "intervention_pct"):
            values = [row[name] for row in selected if row[name] is not None]
            means[name] = {
                "valid_replicas": len(values),
                "mean": float(np.mean(values)) if values else None,
            }
        grouped.append(
            {
                "method": method,
                "group_id": group,
                "split": selected[0]["split"],
                "case_ids": [row["case_id"] for row in selected],
                "replicas": len(selected),
                "all_replicas_complete": all(row["hard_safe_completion"] for row in selected),
                "all_replicas_meet_tracking_targets": all(
                    row["engineering_targets_met"] for row in selected
                ),
                "replica_means": means,
            }
        )
    coverage = {}
    for split in ("train", "validation", "development_test"):
        selected = {case["group_id"]: case for case in cases if case["split"] == split}
        parameters = {
            "friction": [case["scenario"]["wall_sliding_friction"] for case in selected.values()],
            "wall_yaw_deg": [case["scenario"]["wall_yaw_deg"] for case in selected.values()],
            "target_force_n": [case["config"]["target_force"] for case in selected.values()],
            "frequency_hz": [case["task"]["frequency_hz"] for case in selected.values()],
        }
        coverage[split] = {
            "physical_task_groups": len(selected),
            "parameter_ranges": {
                key: [min(values), max(values)] for key, values in parameters.items() if values
            },
        }
    return {
        "scope": "Descriptive group summaries, not confidence intervals or independent-transition statistics.",
        "per_group": grouped,
        "split_coverage": coverage,
    }


def publish_surface_preparation_evidence(output_dir, *, benchmark, stress, dataset, examples):
    """Audit four full archives, cross-check them, then publish compact metadata copies."""
    inputs = {
        "benchmark": Path(benchmark).absolute(),
        "stress": Path(stress).absolute(),
        "dataset": Path(dataset).absolute(),
        "examples": Path(examples).absolute(),
    }
    for name, path in inputs.items():
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise ValueError(f"{name} archive path contains a symlink")
        if not path.is_dir() or any(item.is_symlink() for item in path.iterdir()):
            raise ValueError(f"{name} archive is missing or contains symlink artifacts")
    benchmark_audit = audit_benchmark(inputs["benchmark"])
    stress_audit = audit_benchmark(inputs["stress"])
    dataset_audit = audit_dataset(inputs["dataset"])
    examples_audit = audit_dataset(inputs["examples"])
    manifests = {name: _load_json(path / "manifest.json") for name, path in inputs.items()}
    source_manifest_sha256 = {
        name: _sha256(path / "manifest.json") for name, path in inputs.items()
    }
    full_cases = manifests["benchmark"]["cases"]
    if manifests["benchmark"]["stress"] or not manifests["stress"]["stress"]:
        raise ValueError("benchmark/stress archive modes are reversed")
    if full_cases != manifests["dataset"]["cases"]:
        raise ValueError("full benchmark and dataset cases differ")
    if not all(case in full_cases for case in manifests["stress"]["cases"]):
        raise ValueError("stress cases are not a subset of full benchmark cases")
    if manifests["examples"]["cases"] != manifests["dataset"]["cases"]:
        raise ValueError("examples do not preserve the full dataset case plan")
    if manifests["examples"]["corpus_scope"] != "example_subset" or manifests["examples"].get(
        "parent_manifest_sha256"
    ) != _sha256(inputs["dataset"] / "manifest.json"):
        raise ValueError("examples are not linked to the supplied full dataset")
    source_hashes = {
        json.dumps(manifest["source_and_assets_sha256"], sort_keys=True)
        for manifest in manifests.values()
    }
    if len(source_hashes) != 1:
        raise ValueError("source/assets hashes differ across preparation archives")
    teacher_checks = _teacher_trace_checks(
        inputs["benchmark"],
        inputs["dataset"],
        manifests["benchmark"],
        manifests["dataset"],
    )

    output = _output_path(output_dir)
    if output.exists():
        raise ValueError("compact evidence output must be a new directory")
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in inputs.values()):
        raise ValueError("compact output must be separate from all source archives")
    checks = {
        "valid": True,
        "full_archive_required": True,
        "compact_archive_contains_raw_npz_or_events": False,
        "benchmark_audit": benchmark_audit,
        "stress_audit": stress_audit,
        "dataset_audit": dataset_audit,
        "examples_audit": examples_audit,
        "full_case_count": len(full_cases),
        "stress_case_count": len(manifests["stress"]["cases"]),
        "teacher_collection_equivalence": teacher_checks,
        "grouped_tracking": _group_checks(
            _load_json(inputs["benchmark"] / "report.json")["runs"], full_cases
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-evidence-", dir=output.parent) as temporary:
        staging = Path(temporary) / "evidence"
        staging.mkdir()
        for prefix in ("benchmark", "stress"):
            for name in _BENCHMARK_FILES:
                shutil.copyfile(inputs[prefix] / name, staging / f"{prefix}_{name}")
        shutil.copyfile(inputs["dataset"] / "manifest.json", staging / "dataset_manifest.json")
        (staging / "checks.json").write_text(
            json.dumps(checks, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        examples_relative = os.path.relpath(inputs["examples"], output)
        (staging / "README.md").write_text(
            "# Surface preparation evidence index\n\n"
            "This is a compact, reproducible index of independently audited full archives. "
            "It contains no physics NPZ or decision/event JSON and must not be represented as "
            "the full benchmark, stress, or teacher dataset archive.\n\n"
            f"Public example episodes remain in [{examples_relative}]({examples_relative}).\n",
            encoding="utf-8",
        )
        artifacts = {
            path.name: _sha256(path) for path in sorted(staging.iterdir()) if path.is_file()
        }
        publication_manifest = {
            "schema": "surface_preparation_compact_evidence_v1",
            "full_archive": False,
            "new_holdout": False,
            "artifacts_sha256": artifacts,
            "publisher_script_sha256": _sha256(Path(__file__)),
            "source_archive_manifest_sha256": source_manifest_sha256,
            "source_and_assets_sha256": manifests["benchmark"]["source_and_assets_sha256"],
        }
        (staging / "manifest.json").write_text(
            json.dumps(publication_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "COMPLETE").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="utf-8"
        )
        if any(
            _sha256(path / "manifest.json") != source_manifest_sha256[name]
            for name, path in inputs.items()
        ):
            raise RuntimeError("source archive manifest changed during publication")
        if any(
            _sha256(staging / name) != digest
            for name, digest in publication_manifest["artifacts_sha256"].items()
        ):
            raise RuntimeError("compact evidence changed during publication")
        if _output_path(output).exists():
            raise ValueError("compact evidence output appeared during publication")
        os.rename(staging, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = publish_surface_preparation_evidence(
        args.output,
        benchmark=args.benchmark,
        stress=args.stress,
        dataset=args.dataset,
        examples=args.examples,
    )
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
