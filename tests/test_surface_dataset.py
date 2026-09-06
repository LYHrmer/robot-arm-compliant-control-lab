import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import asdict

import mujoco
import numpy as np
import pytest

import compliant_control_lab.surface_dataset as dataset
import compliant_control_lab.surface_simulation as simulation
from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    yaw_frame,
)
from compliant_control_lab.surface_splits import split_cases


def cases(duration=0.04):
    return [
        {
            "case_id": f"yaw{yaw}",
            "scenario": asdict(SurfaceScenario(wall_yaw_deg=yaw)),
            "config": asdict(SurfaceSimulationConfig(duration=duration, contact_model="smooth")),
            "task": asdict(LearningSurfaceTask(yaw_deg=yaw)),
            "controller_frame_rotation": yaw_frame(yaw).rotation.tolist(),
            "nominal_kind": "adaptive",
        }
        for yaw in (-15, 0, 15)
    ]


def manifest(path):
    return json.loads((path / "manifest.json").read_text())


def arrays(path, entry):
    with np.load(path / entry["file"], allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def reseal(path, document, changed=None):
    """Tamper with arrays AND all integrity hashes; semantic audit must still fail."""
    if changed is not None:
        filename = document["episodes"][0]["file"]
        np.savez_compressed(path / filename, **changed)
        document["episodes"][0]["sha256"] = hashlib.sha256(
            (path / filename).read_bytes()
        ).hexdigest()
    encoded = json.dumps(document, sort_keys=True).encode()
    (path / "manifest.json").write_bytes(encoded)
    (path / "COMPLETE").write_text(hashlib.sha256(encoded).hexdigest() + "\n")


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    path = tmp_path_factory.mktemp("teacher") / "episodes"
    return dataset.collect_demonstrations(path, cases())


def test_real_three_group_collection_keeps_raw_labels_and_both_clocks(collected):
    audit = dataset.audit_dataset(collected)
    assert audit["episodes"] == audit["groups"] == 3
    assert audit["transitions"] == 6 and audit["physics_steps"] == 60
    assert audit["failed_episodes"] == 0 and not audit["new_holdout"]
    assert set(audit["split_episode_counts"].values()) == {1}
    document = manifest(collected)
    assert document["group_split"]["seed"] == 20260906
    assert document["action_stage"] == "normalized_local_residual"
    assert "surface_dataset.py" in document["source_and_assets_sha256"]
    for entry in document["episodes"]:
        stored = arrays(collected, entry)
        assert stored["observations"].shape == (2, 49)
        np.testing.assert_array_equal(
            stored["trace__policy_action"], np.repeat(stored["actions"], 10, axis=0)
        )
        np.testing.assert_array_equal(stored["time_after_s"], [0.02, 0.04])
        assert stored["truncated"].tolist() == [False, True]
        assert entry["simulation_seed"] != entry["env_seed"]
        assert not entry["gates"]["evaluation_observed"]


def test_partial_last_action_and_seeded_repeat_are_exact(tmp_path):
    first = dataset.collect_demonstrations(tmp_path / "first", cases(0.03))
    second = dataset.collect_demonstrations(tmp_path / "second", cases(0.03))
    for left, right in zip(manifest(first)["episodes"], manifest(second)["episodes"]):
        a, b = arrays(first, left), arrays(second, right)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        np.testing.assert_array_equal(a["physics_substeps"], [10, 5])
    assert dataset.audit_dataset(first)["physics_steps"] == 45


@pytest.mark.parametrize("nonfinite", (False, True))
def test_failed_episodes_retained_with_truth_outside_actor_and_terminal_mask(
    tmp_path, monkeypatch, nonfinite
):
    monkeypatch.setattr(
        simulation, "_normal_contact_force", lambda *_: np.nan if nonfinite else 36.0
    )
    path = dataset.collect_demonstrations(tmp_path / "failure", cases())
    audit = dataset.audit_dataset(path)
    assert audit["failed_episodes"] == audit["episodes"] == audit["physics_steps"] == 3
    for entry in manifest(path)["episodes"]:
        stored = arrays(path, entry)
        assert np.all(np.isfinite(stored["observations"]))
        assert stored["terminal_observation_valid"].tolist() == [not nonfinite]
        assert stored["terminated"].tolist() == [True]
        assert not stored["truncated"].any() and entry["return"] <= -1
        if nonfinite:
            assert np.isnan(stored["trace__true_normal_force"][-1])
            assert np.isnan(stored["next_observations"][-1]).all()
            assert "true_normal_force" in entry["nonfinite_trace_fields"]
        else:
            assert stored["trace__true_normal_force"][-1] == 36
            assert np.isfinite(stored["next_observations"][-1]).all()


def test_preintegration_failure_keeps_attempt_but_never_fabricates_trace_row(tmp_path, monkeypatch):
    def fail(*_):
        raise RuntimeError("injected before integration")

    monkeypatch.setattr(mujoco, "mj_step2", fail)
    path = dataset.collect_demonstrations(tmp_path / "failure", cases())
    assert dataset.audit_dataset(path)["physics_steps"] == 0
    for entry in manifest(path)["episodes"]:
        stored = arrays(path, entry)
        assert stored["attempted_substeps"].tolist() == [1]
        assert stored["attempted_execution_status"].tolist() == ["unconfirmed"]
        assert stored["physics_substeps"].tolist() == [0]
        assert np.isnan(stored["next_observations"]).all()
        assert entry["exception_detail"] == "injected before integration"


@pytest.mark.parametrize(
    "field,index,value,error",
    [
        ("actions", (0, 1), 0.2, "teacher label"),
        ("observations", (1, 4), 0.2, "observation chain"),
        ("time_after_s", 0, 0.018, "clock mismatch"),
        ("trace__policy_action", (1, 1), 0.2, "held policy action"),
        ("rewards", 0, 0.01, "reward mismatch"),
        ("terminated", 0, True, "done must"),
        ("terminal_observation_valid", -1, False, "outside failure"),
        ("trace__measured_wrench_sample_time", 0, 1.0, "future or incorrect"),
    ],
)
def test_resealed_semantic_tampering_is_rejected(collected, tmp_path, field, index, value, error):
    path = tmp_path / "tampered"
    shutil.copytree(collected, path)
    document = manifest(path)
    changed = arrays(path, document["episodes"][0])
    changed[field][index] = value
    reseal(path, document, changed)
    with pytest.raises(ValueError, match=error):
        dataset.audit_dataset(path)


def test_hash_and_failure_omission_are_independently_rejected(collected, tmp_path):
    path = tmp_path / "tampered"
    shutil.copytree(collected, path)
    document = manifest(path)
    episode = path / document["episodes"][0]["file"]
    episode.write_bytes(episode.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="episode hash"):
        dataset.audit_dataset(path)
    document["episodes"].pop()
    reseal(path, document)
    with pytest.raises(ValueError, match="missing planned"):
        dataset.audit_dataset(path)


def test_existing_labels_never_silently_reassigned_and_output_never_overwritten(
    tmp_path, collected
):
    declared = split_cases(cases(), seed=dataset.SPLIT_SEED)
    declared[0]["split"] = "invalid"
    with pytest.raises(ValueError, match="supplied split"):
        dataset.collect_demonstrations(tmp_path / "bad", declared)
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="absent or empty|new directory"):
        dataset.collect_demonstrations(collected, cases())
    link = tmp_path / "link"
    link.symlink_to(collected, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        dataset.collect_demonstrations(link, cases())
    with pytest.raises(ValueError, match="three distinct"):
        dataset.collect_demonstrations(tmp_path / "two", cases()[:2])


def test_group_assignment_derived_seed_and_quality_gate_are_audited(collected, tmp_path):
    for fault in ("group", "seed", "gate"):
        path = tmp_path / fault
        shutil.copytree(collected, path)
        document = manifest(path)
        if fault == "group":
            document["cases"][0]["split"] = document["cases"][1]["split"]
        elif fault == "seed":
            document["episodes"][0]["simulation_seed"] += 1
            document["episodes"][0]["actual_config"]["seed"] += 1
        else:
            document["episodes"][0]["gates"]["evaluation_observed"] = True
        reseal(path, document)
        with pytest.raises(
            ValueError, match="split mismatch|simulation seed mismatch|quality gate"
        ):
            dataset.audit_dataset(path)


def test_source_mutation_aborts_atomic_publication(tmp_path, monkeypatch):
    snapshots = iter(({"surface_dataset.py": "old"}, {"surface_dataset.py": "new"}))
    monkeypatch.setattr(dataset, "_source_hashes", lambda: next(snapshots))
    with pytest.raises(RuntimeError, match="source changed"):
        dataset.collect_demonstrations(tmp_path / "unpublished", cases())
    assert not (tmp_path / "unpublished").exists()
    assert list(tmp_path.iterdir()) == []


def test_real_contact_teacher_labels_precede_filter_and_are_held_at_policy_rate(tmp_path):
    path = dataset.collect_demonstrations(tmp_path / "contact", cases(1.6))
    assert dataset.audit_dataset(path)["failed_episodes"] == 0
    for entry in manifest(path)["episodes"]:
        assert entry["gates"]["evaluation_observed"]
        stored = arrays(path, entry)
        assert np.max(np.abs(stored["actions"])) > 0.1
        held_force = np.repeat(stored["actions"], 10, axis=0) * [4, 6, 6]
        assert np.max(np.abs(held_force - stored["trace__filtered_residual_force_local"])) > 0.1


def test_example_subset_is_explicit_and_keeps_full_grouping_plan(collected, tmp_path):
    path = tmp_path / "example"
    shutil.copytree(collected, path)
    document = manifest(path)
    extra = deepcopy(document["cases"][0])
    extra["case_id"], extra["config"]["seed"] = "another_seed", 29
    document["cases"].append(extra)
    document["corpus_scope"] = "example_subset"
    document["parent_manifest_sha256"] = hashlib.sha256(b"parent corpus").hexdigest()
    reseal(path, document)
    audit = dataset.audit_dataset(path)
    assert audit["episodes"] == 3 and audit["planned_episodes"] == 4
    assert audit["corpus_scope"] == "example_subset" and not audit["new_holdout"]
    document["corpus_scope"] = "full"
    reseal(path, document)
    with pytest.raises(ValueError, match="missing planned"):
        dataset.audit_dataset(path)
