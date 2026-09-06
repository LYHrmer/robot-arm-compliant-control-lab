import hashlib
import json
from copy import deepcopy

from compliant_control_lab.surface_dataset import collect_demonstrations
from compliant_control_lab.surface_dataset_replay import export_examples
from compliant_control_lab.surface_readiness_benchmark import generate_readiness_benchmark
from compliant_control_lab.surface_readiness_cases import PREPARATION_SPLIT, preparation_cases
from compliant_control_lab.surface_splits import split_cases
from tools.publish_surface_preparation_evidence import (
    main,
    publish_surface_preparation_evidence,
)


def _short_three_group_cases():
    cases = deepcopy(preparation_cases()[:6])
    for case in cases:
        case["config"]["duration"] = 0.02
        case.pop("group_id")
        case.pop("split")
    return split_cases(cases, seed=20260906, **PREPARATION_SPLIT)


def test_real_compact_evidence_export_cross_checks_both_collection_paths(tmp_path, capsys):
    cases = _short_three_group_cases()
    benchmark = generate_readiness_benchmark(
        tmp_path / "benchmark",
        cases=cases,
        methods=["friction_teacher_50hz"],
    )
    stress = generate_readiness_benchmark(tmp_path / "stress", cases=cases, stress=True)
    dataset = collect_demonstrations(tmp_path / "dataset", cases)
    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    example_ids = []
    for split in ("train", "validation", "development_test"):
        example_ids.append(
            next(
                entry["case_id"]
                for entry in dataset_manifest["episodes"]
                if entry["split"] == split
            )
        )
    examples = export_examples(dataset, tmp_path / "examples", case_ids=example_ids)

    output = publish_surface_preparation_evidence(
        tmp_path / "compact",
        benchmark=benchmark,
        stress=stress,
        dataset=dataset,
        examples=examples,
    )
    expected = {
        "COMPLETE",
        "README.md",
        "benchmark_comparison.csv",
        "benchmark_manifest.json",
        "benchmark_report.json",
        "benchmark_summary.md",
        "checks.json",
        "dataset_manifest.json",
        "manifest.json",
        "stress_comparison.csv",
        "stress_manifest.json",
        "stress_report.json",
        "stress_summary.md",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    checks = json.loads((output / "checks.json").read_text(encoding="utf-8"))
    assert (output / "COMPLETE").read_text().strip() == _sha256(output / "manifest.json")
    assert manifest["full_archive"] is False
    assert checks["valid"] and checks["full_case_count"] == 6
    assert checks["stress_case_count"] == 3
    assert checks["teacher_collection_equivalence"]["cases"] == 6
    assert len(checks["grouped_tracking"]["per_group"]) == 3
    assert {row["replicas"] for row in checks["grouped_tracking"]["per_group"]} == {2}
    assert all(
        row["physical_task_groups"] == 1
        for row in checks["grouped_tracking"]["split_coverage"].values()
    )
    assert {
        "time",
        "position",
        "endpoint_time",
        "endpoint_position",
        "true_normal_force",
        "policy_action",
    } <= set(checks["teacher_collection_equivalence"]["shared_trace_fields"])
    assert checks["compact_archive_contains_raw_npz_or_events"] is False
    assert not any(path.suffix == ".npz" for path in output.iterdir())
    for name, digest in manifest["artifacts_sha256"].items():
        assert _sha256(output / name) == digest
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "must not be represented as the full" in readme
    assert "../examples" in readme

    cli_output = tmp_path / "compact-cli"
    main(
        [
            "--benchmark",
            str(benchmark),
            "--stress",
            str(stress),
            "--dataset",
            str(dataset),
            "--examples",
            str(examples),
            "--output",
            str(cli_output),
        ]
    )
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"output": str(cli_output)}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
