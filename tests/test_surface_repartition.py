import json
import shutil
from copy import deepcopy

import pytest

from compliant_control_lab.surface_dataset import audit_dataset, collect_demonstrations
from compliant_control_lab.surface_experiment import _sha256
from compliant_control_lab.surface_readiness_benchmark import (
    audit_benchmark,
    generate_readiness_benchmark,
)
from compliant_control_lab.surface_readiness_cases import preparation_cases
from tools.repartition_surface_preparation import repartition_surface_preparation


def _short_full_cases():
    cases = deepcopy(preparation_cases())
    for case in cases:
        case["config"]["duration"] = 0.002
    return cases


def _manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _report(path):
    return json.loads((path / "report.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v1_archives(tmp_path_factory):
    root = tmp_path_factory.mktemp("surface-v1")
    cases = _short_full_cases()
    benchmark = generate_readiness_benchmark(
        root / "benchmark", cases=cases, methods=["friction_teacher_50hz"]
    )
    stress = generate_readiness_benchmark(root / "stress", cases=cases, stress=True)
    dataset = collect_demonstrations(root / "dataset", cases)
    return root, benchmark, stress, dataset


def test_metadata_only_repartition_stratifies_anchors_and_preserves_evidence(tmp_path, v1_archives):
    _, benchmark, stress, dataset = v1_archives
    output = repartition_surface_preparation(
        tmp_path / "v2", benchmark=benchmark, stress=stress, dataset=dataset
    )
    assert {path.name for path in output.iterdir()} == {
        "benchmark",
        "stress",
        "dataset",
        "partition.json",
    }
    assert audit_benchmark(output / "benchmark")["run_count"] == 24
    assert audit_benchmark(output / "stress")["run_count"] == 12
    assert audit_dataset(output / "dataset")["episodes"] == 24
    partition = json.loads((output / "partition.json").read_text(encoding="utf-8"))
    assert partition["identity"] == (
        "no new rollout; public post-benchmark, pre-learning partition"
    )
    assert not partition["selection_uses_performance"]
    assert partition["candidates"][-1]["condition_met"]
    assert all(not item["condition_met"] for item in partition["candidates"][:-1])
    anchor_roles = [
        value for value in partition["group_roles"].values() if value["role"] == "nominal_anchor"
    ]
    assert {value["split"] for value in anchor_roles} == {
        "train",
        "validation",
        "development_test",
    }
    role_counts = {
        split: sum(value["split"] == split for value in partition["group_roles"].values())
        for split in ("train", "validation", "development_test")
    }
    assert role_counts == {"train": 8, "validation": 2, "development_test": 2}

    parent_rows = _report(benchmark)["runs"]
    derived_rows = _report(output / "benchmark")["runs"]
    for parent, derived in zip(parent_rows, derived_rows):
        assert {key: value for key, value in parent.items() if key != "split"} == {
            key: value for key, value in derived.items() if key != "split"
        }
        for key in ("trace_file", "events_file"):
            assert _sha256(benchmark / parent[key]) == _sha256(output / "benchmark" / derived[key])
    parent_dataset = _manifest(dataset)
    derived_dataset = _manifest(output / "dataset")
    for parent, derived in zip(parent_dataset["episodes"], derived_dataset["episodes"]):
        assert {key: value for key, value in parent.items() if key != "split"} == {
            key: value for key, value in derived.items() if key != "split"
        }
        assert _sha256(dataset / parent["file"]) == _sha256(output / "dataset" / derived["file"])
    assert (
        _manifest(benchmark)["source_and_assets_sha256"]
        == _manifest(output / "benchmark")["source_and_assets_sha256"]
    )


def test_repartition_rejects_mixed_source_identity(tmp_path, v1_archives):
    _, benchmark, stress, dataset = v1_archives
    changed = tmp_path / "changed-stress"
    shutil.copytree(stress, changed)
    manifest = _manifest(changed)
    manifest["source_and_assets_sha256"]["surface_env.py"] = "0" * 64
    (changed / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (changed / "COMPLETE").write_text(_sha256(changed / "manifest.json") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source/assets"):
        repartition_surface_preparation(
            tmp_path / "rejected", benchmark=benchmark, stress=changed, dataset=dataset
        )
