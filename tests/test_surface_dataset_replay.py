import json

import numpy as np
import pytest

import compliant_control_lab.surface_dataset as dataset
import compliant_control_lab.surface_dataset_replay as replay
from compliant_control_lab.surface_readiness_cases import preparation_cases


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    # This test isolates replay/export, not a package-wide source-freeze test.
    monkeypatch.setattr(dataset, "_source_hashes", dict)
    monkeypatch.setattr(replay, "_source_hashes", dict)
    cases = preparation_cases(6)
    for case in cases:
        case["config"]["duration"] = 0.04
    path = dataset.collect_demonstrations(tmp_path / "corpus", cases)
    return path, cases


def test_exact_causal_reintegration_and_explicit_partial_scope(corpus):
    path, cases = corpus
    selected = [case["case_id"] for case in cases[:2]]
    report = replay.replay_dataset(path, case_ids=selected)
    assert report["valid"] and report["source_matches_collection"]
    assert report["replayed_case_ids"] == selected
    assert report["replayed_episodes"] == 2 and report["available_episodes"] == 12
    assert report["dataset_audit"]["physics_steps"] == 240


def test_reintegration_detects_an_actor_observation_change(corpus, monkeypatch):
    path, cases = corpus
    original = replay.SurfaceLearningEnv

    class AlteredObservation(original):
        def reset(self, **kwargs):
            observation, info = super().reset(**kwargs)
            observation[0] += 0.0001
            return observation, info

    monkeypatch.setattr(replay, "SurfaceLearningEnv", AlteredObservation)
    with pytest.raises(ValueError, match="causal replay mismatch.*observation"):
        replay.replay_dataset(path, case_ids=[cases[0]["case_id"]])


def test_examples_preserve_parent_plan_and_need_three_splits(corpus, tmp_path):
    path, cases = corpus
    selected = [
        next(case["case_id"] for case in cases if case["split"] == split)
        for split in ("train", "validation", "development_test")
    ]
    output = replay.export_examples(path, tmp_path / "examples", case_ids=selected)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["corpus_scope"] == "example_subset"
    assert len(manifest["cases"]) == 12 and len(manifest["episodes"]) == 3
    assert manifest["parent_episode_count"] == 12
    assert manifest["parent_manifest_sha256"] == replay._sha256(path / "manifest.json")
    assert dataset.audit_dataset(output)["episodes"] == 3
    assert replay.replay_dataset(output)["replayed_episodes"] == 3
    with pytest.raises(ValueError, match="new directory|absent or empty"):
        replay.export_examples(path, output, case_ids=selected)
    with pytest.raises(ValueError, match="three disjoint nonempty splits"):
        replay.export_examples(path, tmp_path / "one_split", case_ids=[selected[0]])
    assert not (tmp_path / "one_split").exists()


@pytest.mark.parametrize("selection", ([], ["unknown"]))
def test_unknown_or_empty_replay_selection_is_rejected(corpus, selection):
    with pytest.raises(ValueError, match="existing episodes"):
        replay.replay_dataset(corpus[0], case_ids=selection)


def test_comparator_preserves_nan_failure_evidence():
    replay._same([np.nan, 1.0], [np.nan, 1.0], "masked terminal")
    with pytest.raises(ValueError, match="causal replay mismatch"):
        replay._same([0.0, 1.0], [np.nan, 1.0], "sanitized failure")


@pytest.mark.parametrize("identity", (1, None, " "))
def test_collector_rejects_case_ids_that_cannot_be_replayed(tmp_path, identity):
    cases = preparation_cases(6)
    cases[0]["case_id"] = identity
    with pytest.raises(ValueError, match="case_id values must be nonempty strings"):
        dataset.collect_demonstrations(tmp_path / "bad_identity", cases)
    assert not (tmp_path / "bad_identity").exists()
