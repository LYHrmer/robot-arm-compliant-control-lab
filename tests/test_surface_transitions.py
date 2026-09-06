import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import asdict

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
from compliant_control_lab.surface_transitions import load_transition_batch


def _cases(duration=0.04):
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


def _many_cases(duration=0.002):
    return [
        {
            "case_id": f"yaw{yaw}",
            "scenario": asdict(SurfaceScenario(wall_yaw_deg=yaw)),
            "config": asdict(SurfaceSimulationConfig(duration=duration, contact_model="smooth")),
            "task": asdict(LearningSurfaceTask(yaw_deg=yaw)),
            "controller_frame_rotation": yaw_frame(yaw).rotation.tolist(),
            "nominal_kind": "adaptive",
        }
        for yaw in (-15, -10, -5, 0, 5, 15)
    ]


def _collect(tmp_path, monkeypatch, *, duration=0.04):
    monkeypatch.setattr(dataset, "_source_hashes", lambda: {"test_fixture": "fixed"})
    return dataset.collect_demonstrations(tmp_path / "dataset", _cases(duration))


def _manifest(path):
    return json.loads((path / "manifest.json").read_text())


def _reseal_manifest(path, manifest):
    encoded = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (path / "manifest.json").write_text(encoded)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    (path / "COMPLETE").write_text(digest + "\n")


def test_normal_split_batch_uses_only_actor_arrays_and_explicit_episode_boundaries(
    tmp_path, monkeypatch
):
    path = _collect(tmp_path, monkeypatch)
    batch = load_transition_batch(path, "train")
    assert batch["observations"].shape == batch["next_observations"].shape == (2, 49)
    assert batch["actions"].shape == (2, 3)
    assert batch["rewards"].shape == (2,)
    assert batch["episode_start"].tolist() == [True, False]
    assert len(set(batch["episode_id"])) == 1
    assert not any(key.startswith("trace__") or "true_" in key for key in batch)
    assert batch["metadata"]["split"] == "train"
    assert batch["metadata"]["corpus_scope"] == "full"
    assert batch["metadata"]["example_smoke_test_only"] is False


def test_episode_ids_and_starts_mark_every_concatenation_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "_source_hashes", lambda: {"test_fixture": "fixed"})
    path = dataset.collect_demonstrations(tmp_path / "many", _many_cases())
    batch = load_transition_batch(path, "train")
    starts = np.flatnonzero(batch["episode_start"])
    assert len(starts) == len(batch["metadata"]["case_ids"]) > 1
    assert batch["episode_id"][starts].tolist() == batch["metadata"]["case_ids"]
    continuation = ~batch["episode_start"][1:]
    assert np.all(batch["episode_id"][1:][continuation] == batch["episode_id"][:-1][continuation])


def test_valid_time_truncation_bootstraps_and_partial_block_uses_fractional_discount(
    tmp_path, monkeypatch
):
    path = _collect(tmp_path, monkeypatch, duration=0.03)
    batch = load_transition_batch(path, "validation", gamma_per_policy_step=0.81)
    assert batch["terminated"].tolist() == [False, False]
    assert batch["truncated"].tolist() == [False, True]
    assert batch["terminal_observation_valid"].all()
    assert batch["bootstrap_mask"].all()
    assert batch["physics_substeps"].tolist() == [10, 5]
    np.testing.assert_allclose(batch["discounts"], [0.81, 0.81**0.5])


@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_gamma_boundaries_are_supported(tmp_path, monkeypatch, gamma):
    path = _collect(tmp_path, monkeypatch)
    batch = load_transition_batch(path, "development_test", gamma_per_policy_step=gamma)
    np.testing.assert_array_equal(batch["discounts"], np.full(2, gamma))


@pytest.mark.parametrize("gamma", [-0.1, 1.01, np.nan, np.inf, True, "0.99"])
def test_invalid_gamma_is_rejected_after_dataset_audit(tmp_path, monkeypatch, gamma):
    path = _collect(tmp_path, monkeypatch)
    calls = []
    original = dataset.audit_dataset

    def audited(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr("compliant_control_lab.surface_transitions.audit_dataset", audited)
    with pytest.raises(ValueError):
        load_transition_batch(path, "train", gamma_per_policy_step=gamma)
    assert calls == [path]


def test_finite_safety_failure_is_retained_but_never_bootstraps(tmp_path, monkeypatch):
    monkeypatch.setattr(simulation, "_normal_contact_force", lambda *_: 36.0)
    path = _collect(tmp_path, monkeypatch)
    batch = load_transition_batch(path, "train")
    assert batch["observations"].shape == (1, 49)
    assert batch["terminated"].tolist() == [True]
    assert batch["truncated"].tolist() == [False]
    assert batch["terminal_observation_valid"].tolist() == [True]
    assert batch["bootstrap_mask"].tolist() == [False]
    assert batch["discounts"].tolist() == [0.0]
    assert batch["rewards"][0] <= -1
    assert np.any(batch["next_observations"][0] != 0)


def test_invalid_terminal_nan_is_zero_only_in_loader_view(tmp_path, monkeypatch):
    monkeypatch.setattr(simulation, "_normal_contact_force", lambda *_: np.nan)
    path = _collect(tmp_path, monkeypatch)
    manifest = _manifest(path)
    entry = next(item for item in manifest["episodes"] if item["split"] == "validation")
    with np.load(path / entry["file"], allow_pickle=False) as stored:
        assert np.isnan(stored["next_observations"][-1]).all()
    batch = load_transition_batch(path, "validation")
    assert batch["terminated"].tolist() == [True]
    assert batch["terminal_observation_valid"].tolist() == [False]
    assert batch["bootstrap_mask"].tolist() == [False]
    assert batch["discounts"].tolist() == [0.0]
    assert not batch["next_observations"].any()
    with np.load(path / entry["file"], allow_pickle=False) as stored:
        assert np.isnan(stored["next_observations"][-1]).all()


def test_split_batches_have_disjoint_declared_groups(tmp_path, monkeypatch):
    path = _collect(tmp_path, monkeypatch)
    batches = {name: load_transition_batch(path, name) for name in dataset.SPLIT_NAMES}
    groups = [set(batch["metadata"]["group_ids"]) for batch in batches.values()]
    assert all(groups)
    assert all(left.isdisjoint(right) for i, left in enumerate(groups) for right in groups[i + 1 :])


def test_example_subset_requires_explicit_smoke_test_opt_in(tmp_path, monkeypatch):
    source = _collect(tmp_path, monkeypatch)
    path = tmp_path / "example"
    shutil.copytree(source, path)
    manifest = _manifest(path)
    extra = deepcopy(manifest["cases"][0])
    extra["case_id"], extra["config"]["seed"] = "planned_but_absent", 29
    manifest["cases"].append(extra)
    manifest["corpus_scope"] = "example_subset"
    manifest["parent_manifest_sha256"] = hashlib.sha256(b"parent").hexdigest()
    _reseal_manifest(path, manifest)
    with pytest.raises(ValueError, match="smoke-test material"):
        load_transition_batch(path, "train")
    batch = load_transition_batch(path, "train", allow_examples=True)
    assert batch["metadata"]["corpus_scope"] == "example_subset"
    assert batch["metadata"]["example_smoke_test_only"] is True
