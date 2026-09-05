import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import compliant_control_lab.surface_experiment as experiment
from compliant_control_lab.surface_replay import SurfaceReplayResult, replay_surface_trace


@pytest.fixture
def fake_trials(monkeypatch):
    calls, saved = [], []

    def fake_trial(frame, *, scenario, config, task, controller_kind):
        calls.append((frame, scenario, config, task, controller_kind))
        metrics = {name: 1.0 for name in experiment.METRICS}
        metrics.update(has_raw_contact=True, first_raw_contact_time_s=0.4, contact_ratio_pct=75.0)
        return SimpleNamespace(metrics=lambda: metrics, trace={"sample": np.array([config.seed])})

    def fake_save(path, arrays):
        saved.append((path.name, arrays))
        np.savez_compressed(path, **arrays)
        return path

    monkeypatch.setattr(experiment, "run_surface_trial", fake_trial)
    monkeypatch.setattr(experiment, "save_surface_trace", fake_save)
    monkeypatch.setattr(
        experiment,
        "replay_surface_trace",
        lambda _path: SurfaceReplayResult(
            sample_count=1,
            max_wrench_error=0.0,
            max_torque_error=0.0,
            matches=True,
            controller_kind="surface_adaptive",
            controller_name="synthetic",
            controller_supplied=False,
        ),
    )
    monkeypatch.setattr(
        experiment, "_plot_representative", lambda path, *_args: path.write_bytes(b"synthetic plot")
    )
    monkeypatch.setattr(experiment, "_source_hashes", lambda: {"surface_simulation.py": "a" * 64})
    return calls, saved


def test_grid_is_fixed_and_representative_case_is_prespecified():
    cases = experiment.development_cases()
    assert len(cases) == 24
    coordinates = [
        (
            case["scenario"].wall_yaw_deg,
            case["scenario"].wall_time_constant,
            case["scenario"].tool_mass_kg,
            case["config"].seed,
        )
        for case in cases
    ]
    assert len(set(coordinates)) == 24
    assert coordinates[16] == (15.0, 0.005, 0.10, 11)
    for case in cases:
        assert case["config"].duration == 4.5
        assert case["config"].timestep == 0.002
        assert case["task"].yaw_deg == case["scenario"].wall_yaw_deg
        assert case["scenario"].nominal_tool_mass_kg == 0.10
        assert tuple(case["scenario"].force_bias_sensor_n) == (0.2, 0.0, 0.0)


def test_all_96_rows_share_paired_case_inputs_and_only_save_four_representatives(
    tmp_path, fake_trials
):
    calls, saved = fake_trials
    output = experiment.generate_surface_experiment(tmp_path / "development")
    assert len(calls) == 96
    assert len(saved) == 4
    for arm_index, (arm, (kind, offset)) in enumerate(experiment.ARMS.items()):
        for case_index in range(24):
            frame, scenario, config, task, actual_kind = calls[24 * arm_index + case_index]
            assert (scenario, config, task) == calls[case_index][1:4]
            assert actual_kind == kind
            yaw = 0.0 if offset is None else task.yaw_deg + offset
            np.testing.assert_allclose(
                frame.rotation[:, 0], [np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw)), 0.0]
            )
        assert f"representative_case_16_{arm}.npz" in {name for name, _ in saved}
    with (output / "comparison.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 96
    assert "gate_pass" not in rows[0]
    with (output / "paired_deltas.csv").open(newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 72
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["is_subset"] is False
    assert manifest["new_holdout"] is False
    assert len(manifest["all_development_cases"]) == 24
    assert manifest["full_trace_case_indices"] == [16]
    assert manifest["versions"]["mujoco"]
    assert "base" in manifest["local_safe_controller_parameters"]
    assert "manifest.json" not in manifest["artifact_sha256"]
    assert len(manifest["artifact_sha256"]) == 8
    assert set(manifest["replay_checks"]) == set(experiment.ARMS)
    assert all(check["matches"] for check in manifest["replay_checks"].values())
    for filename, expected in manifest["artifact_sha256"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    assert (output / "COMPLETE").read_text().strip() == hashlib.sha256(
        (output / "manifest.json").read_bytes()
    ).hexdigest()
    summary = (output / "summary.md").read_text()
    assert "24 cases, 96 rows" in summary
    assert "neither the old 24 holdout cases nor the 48 blind cases" in summary
    assert "not full traces" in summary
    assert "Median force RMSE" in summary
    assert "ideal frame calibration" in summary
    assert "filtering and timing lag" in summary


@pytest.mark.parametrize("indices, expected_traces", [([16], 4), ([0], 0)])
def test_subset_is_labeled_and_preserves_default_protocol(
    tmp_path, fake_trials, indices, expected_traces
):
    calls, saved = fake_trials
    output = experiment.generate_surface_experiment(tmp_path / "subset", case_indices=indices)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["is_subset"] is True
    assert manifest["selected_case_indices"] == indices
    assert "Development SUBSET: 1/24" in (output / "summary.md").read_text()
    assert len(saved) == expected_traces
    assert len(calls) == 4
    assert all(call[2].duration == 4.5 for call in calls)


def test_missing_contact_cannot_be_counted_as_paired_peak_improvement():
    base = {
        "arm": "world_safe_adaptive",
        "case_index": 0,
        "has_raw_contact": True,
        **{name: 10.0 for name in experiment.METRICS},
    }
    missed = {
        **base,
        "arm": "surface_exact",
        "has_raw_contact": False,
        "peak_force_n": 0.0,
        "first_raw_contact_time_s": None,
        "contact_ratio_pct": 0.0,
    }
    pairs = experiment.paired_deltas([base, missed])
    assert len(pairs) == 1
    assert pairs[0]["contact_in_both_arms"] is False
    assert all(pairs[0][name] is None for name in experiment.CONTACT_METRICS)
    assert pairs[0]["contact_ratio_pct"] == -10.0
    # Fill the other arms to exercise the public summary's zero eligible sample count.
    rows = [base, missed, {**base, "arm": "surface_minus5"}, {**base, "arm": "surface_plus5"}]
    summary = experiment.render_summary(rows, experiment.paired_deltas(rows), "subset", [])
    assert "| surface_exact | peak_force_n | 0 | NA |" in summary


@pytest.mark.parametrize(
    "indices, error",
    [
        ([], ValueError),
        ([0, 0], ValueError),
        ([24], IndexError),
        ([-1], IndexError),
        ([True], TypeError),
        ([0.5], TypeError),
    ],
)
def test_invalid_indices_stop_before_running(tmp_path, fake_trials, indices, error):
    calls, _ = fake_trials
    with pytest.raises(error):
        experiment.generate_surface_experiment(tmp_path / "invalid", case_indices=indices)
    assert not calls


@pytest.mark.parametrize("kind", ["nonempty", "symlink", "symlink_parent", "file"])
def test_output_guard_preserves_existing_data(tmp_path, fake_trials, kind):
    calls, _ = fake_trials
    sentinel = tmp_path / "retained"
    sentinel.mkdir()
    (sentinel / "existing.txt").write_text("keep")
    output = tmp_path / "output"
    if kind == "nonempty":
        output = sentinel
    elif kind == "file":
        output.write_text("keep")
    else:
        output.symlink_to(sentinel, target_is_directory=True)
        if kind == "symlink_parent":
            output /= "child"
    with pytest.raises(ValueError):
        experiment.generate_surface_experiment(output, case_indices=[0])
    assert (sentinel / "existing.txt").read_text() == "keep"
    assert not calls


@pytest.mark.parametrize(
    "path",
    ["results/franka_safety_blind/new_surface", "results/franka_safety_preholdout", "results", "."],
)
def test_frozen_directories_and_ancestors_are_rejected(path, fake_trials):
    calls, _ = fake_trials
    with pytest.raises(ValueError):
        experiment.generate_surface_experiment(Path(path), case_indices=[0])
    assert not calls


def test_empty_output_can_be_published_atomically(tmp_path, fake_trials):
    output = tmp_path / "empty"
    output.mkdir()
    experiment.generate_surface_experiment(output, case_indices=[0])
    assert (output / "COMPLETE").exists()


def test_missing_parent_can_be_created_for_a_new_report(tmp_path, fake_trials):
    output = tmp_path / "new_parent" / "report"
    experiment.generate_surface_experiment(output, case_indices=[0])
    assert (output / "COMPLETE").exists()


def test_short_real_runner_saves_replayable_representative_traces(tmp_path, monkeypatch):
    cases = experiment.development_cases()
    for case in cases:
        case["config"] = replace(case["config"], duration=0.02, evaluation_start=0.0)
    monkeypatch.setattr(experiment, "development_cases", lambda: cases)
    # Other agents may edit source concurrently; source-drift rejection is tested separately.
    monkeypatch.setattr(experiment, "_source_hashes", lambda: {"test_fixture": "short_duration"})
    output = experiment.generate_surface_experiment(tmp_path / "real_short", case_indices=[16])
    for arm in experiment.ARMS:
        replay = replay_surface_trace(output / f"representative_case_16_{arm}.npz")
        assert replay.matches
        assert replay.sample_count == 10
    assert (output / "representative_case_16.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "Development SUBSET" in (output / "summary.md").read_text()


@pytest.mark.parametrize("failure", ["source_drift", "write_failure", "nonfinite_metrics"])
def test_failure_does_not_publish_partial_artifacts(tmp_path, monkeypatch, fake_trials, failure):
    calls, _ = fake_trials
    if failure == "source_drift":
        monkeypatch.setattr(
            experiment, "_source_hashes", lambda: {"source": "after" if calls else "before"}
        )
    elif failure == "write_failure":

        def fail(*_args):
            raise RuntimeError("write failed")

        monkeypatch.setattr(experiment, "render_summary", fail)
    else:
        previous = experiment.run_surface_trial

        def invalid(*args, **kwargs):
            trial = previous(*args, **kwargs)
            values = trial.metrics()
            values["force_rmse_n"] = np.nan
            trial.metrics = lambda: values
            return trial

        monkeypatch.setattr(experiment, "run_surface_trial", invalid)
    output = tmp_path / "failed"
    with pytest.raises((ValueError, RuntimeError)):
        experiment.generate_surface_experiment(output, case_indices=[0])
    assert not output.exists()
    assert not list(tmp_path.glob(".surface-experiment-*"))


def test_representative_replay_mismatch_refuses_publication(tmp_path, monkeypatch, fake_trials):
    calls, saved = fake_trials
    monkeypatch.setattr(
        experiment,
        "replay_surface_trace",
        lambda _path: SurfaceReplayResult(
            sample_count=1,
            max_wrench_error=1.0,
            max_torque_error=1.0,
            matches=False,
            controller_kind="surface_adaptive",
            controller_name="synthetic",
            controller_supplied=False,
        ),
    )
    output = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="representative replay mismatch"):
        experiment.generate_surface_experiment(output, case_indices=[16])
    assert len(calls) == len(saved) == 1
    assert not output.exists()
    assert not list(tmp_path.glob(".surface-experiment-*"))
