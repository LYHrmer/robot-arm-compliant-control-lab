import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import compliant_control_lab.reference_ablation as ablation
from compliant_control_lab.franka_simulation import (
    FrankaScenario,
    FrankaSimulationConfig,
    FrankaTrialResult,
)


def _trial(force, *, timestep=0.002, timing="split_step"):
    force = np.asarray(force, dtype=float)
    count = len(force)
    return FrankaTrialResult(
        controller="synthetic",
        scenario="synthetic",
        time=np.arange(count) * timestep,
        q=np.zeros((count, 7)),
        position=np.zeros((count, 3)),
        desired_position=np.zeros((count, 3)),
        normal_force=force.copy(),
        raw_normal_force=force,
        desired_force=np.full(count, 12.0),
        orientation_error_rad=np.zeros(count),
        torque=np.zeros((count, 7)),
        controller_time_us=np.ones(count),
        saturated=np.zeros(count, dtype=bool),
        linear_velocity=np.tile([2.0, 3.0, 5.0], (count, 1)),
        commanded_wrench=np.tile([7.0, 11.0, 13.0, 0, 0, 0], (count, 1)),
        control_timing=timing,
        raw_force_sample_time=np.arange(count) * timestep,
        kinematic_sample_time=np.arange(count) * timestep,
    )


def _aligned_config(**values):
    return FrankaSimulationConfig(control_timing="split_step", **values)


def test_contact_window_boundaries_and_true_wall_normal():
    force = np.zeros(650)
    force[100] = 1.0  # First contact at 0.2 s.
    force[350] = 36.0  # Exactly +0.5 s, included early.
    force[351] = 70.0  # +0.502 s, excluded early.
    force[599] = 80.0  # +0.998 s, excluded late.
    force[600] = 40.0  # Exactly +1.0 s, included late.
    force[601] = 35.0  # Equality does not count as over 35 N.
    result = _trial(force)

    row = ablation.analyze_trial(result, FrankaScenario("yaw", wall_yaw_deg=90), _aligned_config())

    assert row["has_raw_contact"] is True
    assert row["first_raw_contact_time_s"] == pytest.approx(0.2)
    assert row["early_raw_peak_n"] == 36.0
    assert row["late_raw_peak_n"] == 40.0
    assert row["seconds_over_35_n"] == pytest.approx(4 * 0.002)
    assert row["first_contact_wall_normal_velocity_m_s"] == pytest.approx(3.0)
    assert row["peak_wall_normal_velocity_m_s"] == pytest.approx(3.0)
    assert row["peak_commanded_wall_normal_force_n"] == pytest.approx(11.0)


@pytest.mark.parametrize("force", ([], [0, 0, 0]))
def test_empty_or_no_contact_has_no_invented_event_values(force):
    row = ablation.analyze_trial(_trial(force), FrankaScenario("none"), _aligned_config())

    assert row["has_raw_contact"] is False
    assert all(row[field] is None for field in ablation.EVENT_FIELDS)


@pytest.mark.parametrize("count", (10, 600))
def test_unobserved_or_contact_free_late_window_is_none(count):
    force = np.zeros(count)
    force[0] = 12
    row = ablation.analyze_trial(_trial(force), FrankaScenario("brief"), _aligned_config())

    assert row["early_raw_peak_n"] == 12
    assert row["late_raw_peak_n"] is None


def test_legacy_event_values_are_explicitly_unaligned():
    row = ablation.analyze_trial(
        _trial([0, 12, 40], timing="legacy"), FrankaScenario("legacy"), FrankaSimulationConfig()
    )

    assert row["has_raw_contact"] is True
    assert row["event_timing"] == "legacy_unaligned"
    assert all(row[field] is None for field in ablation.EVENT_FIELDS)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda trial: setattr(trial, "time", np.array([0.0, 0.004])),
        lambda trial: setattr(trial, "raw_normal_force", np.array([0.0, np.nan])),
        lambda trial: setattr(trial, "linear_velocity", np.zeros((1, 3))),
        lambda trial: setattr(trial, "control_timing", "legacy"),
        lambda trial: setattr(trial, "raw_force_sample_time", trial.time + 0.002),
        lambda trial: setattr(trial, "kinematic_sample_time", np.zeros(0)),
    ),
)
def test_invalid_telemetry_is_rejected(mutate):
    trial = _trial([0, 12])
    mutate(trial)
    with pytest.raises(ValueError):
        ablation.analyze_trial(trial, FrankaScenario("invalid"), _aligned_config())


@pytest.mark.parametrize(
    "destination", ("frozen", "child", "ancestor", "symlink", "symlink_parent", "nonempty")
)
def test_path_guard_preserves_existing_data(tmp_path, destination):
    frozen = tmp_path / "archive"
    frozen.mkdir()
    sentinel = frozen / "frozen.txt"
    sentinel.write_text("retain", encoding="utf-8")
    output = {"frozen": frozen, "child": frozen / "child", "ancestor": tmp_path}.get(
        destination, tmp_path / "output"
    )
    if destination == "symlink":
        output.symlink_to(frozen, target_is_directory=True)
    elif destination == "symlink_parent":
        output.symlink_to(frozen, target_is_directory=True)
        output /= "child"
    elif destination == "nonempty":
        output.mkdir()
        (output / "partial.csv").write_text("retain partial", encoding="utf-8")

    with pytest.raises(ValueError):
        ablation.validate_output_path(output, frozen)
    assert sentinel.read_text(encoding="utf-8") == "retain"
    if destination == "nonempty":
        assert (output / "partial.csv").read_text(encoding="utf-8") == "retain partial"


def test_path_guard_allows_empty_unrelated_destination(tmp_path):
    output = tmp_path / "empty"
    output.mkdir()
    assert ablation.validate_output_path(output, tmp_path / "archive") == output


@pytest.fixture
def mocked_trials(monkeypatch):
    """Use the real audited archive and seed derivation, but never run a simulation."""
    archive = Path("results/franka_safety_blind")
    with (archive / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        frozen = {
            row["case"]: row
            for row in csv.DictReader(handle)
            if row["method"] == "safe_adaptive_hybrid"
        }
    calls = []

    def fake_trial(controller, *, scenario, config):
        calls.append((controller, scenario, config))
        trial = _trial(np.full(650, 12.0), timing=config.control_timing)
        metrics = {key: float(frozen[scenario.name][key]) for key in ablation.REPLAYED_METRICS}
        metrics.update(controller=controller.name, scenario=scenario.name, controller_p95_us=1.0)
        trial.metrics = lambda: metrics
        return trial

    monkeypatch.setattr(ablation, "run_franka_trial", fake_trial)
    return calls, frozen


def test_four_arms_share_scenarios_seeds_duration_and_preserve_gates(tmp_path, mocked_trials):
    calls, frozen = mocked_trials
    output = ablation.generate_reference_ablation(
        "results/franka_safety_preholdout/protocol.json",
        "results/franka_safety_blind",
        tmp_path / "ablation",
        case_indices=[3, 0],
    )

    assert len(calls) == 8
    for arm_index, (_name, (timing, reference, factory)) in enumerate(ablation.ARMS.items()):
        for case_offset in range(2):
            controller, scenario, config = calls[2 * arm_index + case_offset]
            assert type(controller) is factory
            assert scenario == calls[case_offset][1]
            assert config.seed == calls[case_offset][2].seed
            assert config.duration == 4.5
            assert config.control_timing == timing
            assert config.approach_reference == reference
            assert (
                replace(config, control_timing="legacy", approach_reference="legacy")
                == calls[case_offset][2]
            )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["new_holdout"] is False
    assert manifest["selected_case_indices"] == [0, 3]
    assert len(manifest["all_public_cases"]) == 48
    assert manifest["gate"]["max_peak_force_n"] == 35.0
    assert "franka_reference.py" in manifest["source_and_assets_sha256"]
    assert "assets/franka_scene.xml" in manifest["source_and_assets_sha256"]
    assert manifest["versions"]["mujoco"]
    assert (output / "COMPLETE").read_text().strip() == hashlib.sha256(
        (output / "manifest.json").read_bytes()
    ).hexdigest()
    assert set(manifest["artifact_sha256"]) == {"comparison.csv", "summary.md"}
    for filename, expected_hash in manifest["artifact_sha256"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected_hash
    with (output / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        assert row["gate_pass"] == frozen[row["scenario"]]["gate_pass"]
        assert row["failed_checks"] == frozen[row["scenario"]]["failed_checks"]
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "Public subset: 2/48" in summary
    assert "NOT a new holdout" in summary
    assert "Paired differences from timing_only" in summary
    assert "Legacy event diagnostics are NA" in summary
    with pytest.raises(ValueError, match="empty"):
        ablation.generate_reference_ablation(
            "results/franka_safety_preholdout/protocol.json",
            "results/franka_safety_blind",
            output,
            case_indices=[0],
        )
    assert len(calls) == 8


def test_legacy_absolute_drift_stops_before_other_arms_or_writes(
    tmp_path, mocked_trials, monkeypatch
):
    calls, _ = mocked_trials
    fake_trial = ablation.run_franka_trial

    def drifting_trial(*args, **kwargs):
        trial = fake_trial(*args, **kwargs)
        metrics = trial.metrics()
        metrics["peak_force_n"] += (
            2e-12  # Below relative tolerance at 35 N, above absolute tolerance.
        )
        trial.metrics = lambda: metrics
        return trial

    monkeypatch.setattr(ablation, "run_franka_trial", drifting_trial)
    output = tmp_path / "ablation"
    with pytest.raises(ValueError, match="absolute metric mismatch"):
        ablation.generate_reference_ablation(
            "results/franka_safety_preholdout/protocol.json",
            "results/franka_safety_blind",
            output,
            case_indices=[0],
        )
    assert len(calls) == 1
    assert not output.exists()


@pytest.mark.parametrize("changed_file", ("franka_reference.py", "assets/franka_scene.xml"))
def test_source_or_asset_drift_during_trials_rejects_report_before_writing(
    tmp_path, mocked_trials, monkeypatch, changed_file
):
    calls, _ = mocked_trials
    original = ablation._source_hashes()

    def simulated_hashes():
        current = original.copy()
        if calls:
            current[changed_file] = "0" * 64
        return current

    monkeypatch.setattr(ablation, "_source_hashes", simulated_hashes)
    output = tmp_path / "ablation"
    with pytest.raises(ValueError, match="source/assets changed during ablation"):
        ablation.generate_reference_ablation(
            "results/franka_safety_preholdout/protocol.json",
            "results/franka_safety_blind",
            output,
            case_indices=[0],
        )
    assert len(calls) == len(ablation.ARMS)
    assert not output.exists()
    assert not list(tmp_path.glob(".reference-ablation-*"))


@pytest.mark.parametrize(
    "indices, error",
    (
        ([], ValueError),
        ([0, 0], ValueError),
        ([48], IndexError),
        ([True], TypeError),
        ([0.5], TypeError),
    ),
)
def test_invalid_case_selection_never_runs_trials(tmp_path, mocked_trials, indices, error):
    calls, _ = mocked_trials
    with pytest.raises(error):
        ablation.generate_reference_ablation(
            "results/franka_safety_preholdout/protocol.json",
            "results/franka_safety_blind",
            tmp_path / "ablation",
            case_indices=indices,
        )
    assert not calls


def test_failed_report_write_does_not_publish_partial_destination(
    tmp_path, mocked_trials, monkeypatch
):
    def failing_summary(*_args):
        raise RuntimeError("render failure")

    monkeypatch.setattr(ablation, "render_summary", failing_summary)
    output = tmp_path / "ablation"
    with pytest.raises(RuntimeError, match="render failure"):
        ablation.generate_reference_ablation(
            "results/franka_safety_preholdout/protocol.json",
            "results/franka_safety_blind",
            output,
            case_indices=[0],
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".reference-ablation-*"))
