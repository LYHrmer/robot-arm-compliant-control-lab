import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import compliant_control_lab.surface_contact_validation as validation
from compliant_control_lab.surface_replay import SurfaceReplayResult, replay_surface_trace


@pytest.fixture
def fake_trials(monkeypatch):
    archived, _ = validation._load_baseline(Path("results/franka_surface_development"))
    calls, saved = [], []

    def fake_trial(frame, *, scenario, config, task, controller_kind):
        calls.append((frame, scenario, config, task, controller_kind))
        index = int(scenario.name.rsplit("_", 1)[1])
        yaw = np.rad2deg(np.arctan2(frame.rotation[1, 0], frame.rotation[0, 0]))
        arm = next(
            name
            for name, (kind, offset) in validation.ARMS.items()
            if kind == controller_kind
            and np.isclose(yaw, 0 if offset is None else task.yaw_deg + offset)
        )
        frozen = archived[arm, index]
        metrics = {
            name: float(frozen[name]) if frozen[name] else None for name in validation.METRICS
        }
        metrics["has_raw_contact"] = True
        if config.contact_model == "smooth":
            metrics.update(
                force_rmse_n=0.5,
                peak_force_n=14.0,
                contact_ratio_pct=100.0,
                saturation_pct=0.0,
                first_raw_contact_time_s=0.6,
                seconds_over_35_n=0.0,
            )
        angle = np.deg2rad(scenario.wall_yaw_deg)
        normal = np.array([np.cos(angle), np.sin(angle), 0])
        position = np.array([0.4, 0, 0]) - 0.024 * normal
        trace = {
            "time": np.array([1.5, 2.0, 4.0]),
            "position": np.tile(position, (3, 1)),
            "true_contact_gap_m": np.full(3, -0.001),
            "true_normal_force": np.full(3, 12.0),
            "true_tangent_force_n": np.full(3, 5.4),
        }
        return SimpleNamespace(metrics=lambda: metrics, trace=trace)

    def fake_save(path, arrays):
        saved.append(path.name)
        np.savez_compressed(path, **arrays)
        return path

    monkeypatch.setattr(validation, "run_surface_trial", fake_trial)
    monkeypatch.setattr(validation, "save_surface_trace", fake_save)
    monkeypatch.setattr(
        validation,
        "replay_surface_trace",
        lambda _path: SurfaceReplayResult(
            sample_count=3,
            max_wrench_error=0.0,
            max_torque_error=0.0,
            matches=True,
            controller_kind="surface_adaptive",
            controller_name="synthetic",
            controller_supplied=False,
        ),
    )
    monkeypatch.setattr(
        validation, "_plot_representative", lambda path, *_args: path.write_bytes(b"plot")
    )
    monkeypatch.setattr(validation, "_source_hashes", lambda: {"test_source": "unchanged"})
    return calls, saved


def test_complete_paired_grid_only_changes_contact_model_and_preserves_baseline(
    tmp_path, fake_trials
):
    calls, saved = fake_trials
    output = validation.generate_surface_contact_validation(tmp_path / "full")

    assert len(calls) == 192
    for index in range(96):
        first, second = calls[index], calls[index + 96]
        np.testing.assert_array_equal(first[0].rotation, second[0].rotation)
        assert first[1] == second[1]
        assert first[3:] == second[3:]
        assert first[2].contact_model == "legacy"
        assert second[2].contact_model == "smooth"
        assert first[2] == replace(second[2], contact_model="legacy")
        assert first[2].duration == 4.5
    with (output / "comparison.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert (
        len(rows) == len({(row["profile"], row["arm"], row["case_index"]) for row in rows}) == 192
    )
    assert all(float(row["legacy_metric_max_abs_error"]) == 0 for row in rows[:96])
    assert all(row["repair_pass"] == "yes" for row in rows[96:])
    assert all(float(row["median_true_friction_ratio"]) == pytest.approx(0.45) for row in rows)
    with (output / "paired_deltas.csv").open(newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 96
    lookup = {(row["profile"], row["arm"], row["case_index"]): row for row in rows}
    for pair in pairs:
        key = pair["arm"], pair["case_index"]
        expected = float(lookup[("smooth", *key)]["force_rmse_n"]) - float(
            lookup[("legacy", *key)]["force_rmse_n"]
        )
        assert float(pair["force_rmse_n"]) == pytest.approx(expected)
    assert len(saved) == 4
    assert all("smooth" in name for name in saved)
    assert len(list(output.glob("*.npz"))) == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["case_configurations"]) == 192
    assert manifest["selected_case_indices"] == list(range(24))
    assert manifest["new_holdout"] is False
    assert manifest["is_subset"] is False
    assert set(manifest["legacy_trace_references"]) == set(validation.ARMS)
    assert manifest["baseline_archive"]["directory"] == "results/franka_surface_development"
    assert all(
        reference["path"].startswith("results/franka_surface_development/")
        for reference in manifest["legacy_trace_references"].values()
    )
    assert all(check["matches"] for check in manifest["replay_checks"].values())
    assert manifest["contact_model_description"]["smooth_solimp"][0] == 0.0
    assert manifest["contact_model_description"]["legacy_solimp"][0] == 0.925
    for name, expected in manifest["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert (output / "COMPLETE").read_text().strip() == hashlib.sha256(
        (output / "manifest.json").read_bytes()
    ).hexdigest()
    summary = (output / "summary.md").read_text()
    assert "NOT a new holdout" in summary
    assert "not controller superiority" in summary
    assert "changes the force-distance response" in summary
    assert "no tangent gate" in summary
    assert "Worst saturation [%]" in summary
    assert "gap >0 means separation, gap <0 penetration" in summary


@pytest.mark.parametrize("yaw", (0.0, 15.0))
def test_geometry_sign_and_separation_metric_match_runner(yaw):
    normal = np.array([np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw)), 0.0])
    expected = np.array([0.001, 0.0, -0.001])
    trace = {
        "time": np.array([1.5, 2.0, 2.5]),
        "position": np.array([0.4, 0.0, 0.0]) - (0.025 + expected[:, None]) * normal,
        "true_contact_gap_m": expected,
        "true_normal_force": np.array([0.0, 0.0, 12.0]),
    }
    np.testing.assert_allclose(validation._geometry_gap(trace, yaw), expected, atol=1e-15)
    case = validation.development_cases()[0]
    case["scenario"] = replace(case["scenario"], wall_yaw_deg=yaw)
    metrics = validation._auxiliary_metrics(SimpleNamespace(trace=trace), case)
    assert metrics["max_geometric_separation_um"] == pytest.approx(1000.0)
    trace["true_contact_gap_m"] = -expected
    with pytest.raises(ValueError, match="geometric separation disagrees"):
        validation._auxiliary_metrics(SimpleNamespace(trace=trace), case)


def test_subset_preserves_settings_and_does_not_require_representative(tmp_path, fake_trials):
    calls, saved = fake_trials
    output = validation.generate_surface_contact_validation(tmp_path / "subset", case_indices=[0])
    assert len(calls) == 8
    assert not saved
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["is_subset"] is True
    assert "SUBSET: 1/24" in (output / "summary.md").read_text()


def test_damaged_baseline_artifact_stops_before_trials(tmp_path, fake_trials):
    calls, _ = fake_trials
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "comparison.csv").write_text("damaged archive\n")
    manifest = baseline / "manifest.json"
    manifest.write_text(json.dumps({"artifact_sha256": {"comparison.csv": "0" * 64}}))
    (baseline / "COMPLETE").write_text(hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n")
    with pytest.raises(ValueError, match="baseline artifact hash mismatch"):
        validation.generate_surface_contact_validation(
            tmp_path / "output", baseline_dir=baseline, case_indices=[0]
        )
    assert not calls
    assert not (tmp_path / "output").exists()


def test_changed_baseline_identity_during_run_refuses_publication(
    tmp_path, fake_trials, monkeypatch
):
    calls, _ = fake_trials
    original = validation._load_baseline

    def changed_identity(path):
        rows, identity = original(path)
        if calls:
            identity["manifest_sha256"] = "0" * 64
        return rows, identity

    monkeypatch.setattr(validation, "_load_baseline", changed_identity)
    with pytest.raises(ValueError, match="baseline archive changed during validation"):
        validation.generate_surface_contact_validation(tmp_path / "output", case_indices=[0])
    assert len(calls) == 8
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".surface-contact-validation-*"))


@pytest.mark.parametrize(
    "indices, error",
    [
        ([], ValueError),
        ([0, 0], ValueError),
        ([24], IndexError),
        ([True], TypeError),
        ([0.5], TypeError),
    ],
)
def test_invalid_case_indices_do_not_run(tmp_path, fake_trials, indices, error):
    calls, _ = fake_trials
    with pytest.raises(error):
        validation.generate_surface_contact_validation(tmp_path / "bad", case_indices=indices)
    assert not calls


def test_legacy_drift_rejects_before_smooth_and_before_publication(
    tmp_path, fake_trials, monkeypatch
):
    calls, _ = fake_trials
    original = validation.run_surface_trial

    def drift(*args, **kwargs):
        trial = original(*args, **kwargs)
        metrics = trial.metrics()
        metrics["force_rmse_n"] += 2e-10
        trial.metrics = lambda: metrics
        return trial

    monkeypatch.setattr(validation, "run_surface_trial", drift)
    output = tmp_path / "drift"
    with pytest.raises(ValueError, match="legacy replay absolute metric mismatch"):
        validation.generate_surface_contact_validation(output, case_indices=[0])
    assert len(calls) == 1
    assert not output.exists()
    assert not list(tmp_path.glob(".surface-contact-validation-*"))


@pytest.mark.parametrize("kind", ("source_drift", "write_failure", "replay_mismatch"))
def test_failures_never_publish_partial_reports(tmp_path, fake_trials, monkeypatch, kind):
    calls, _ = fake_trials
    if kind == "source_drift":
        monkeypatch.setattr(validation, "_source_hashes", lambda: {"source": str(len(calls))})
    elif kind == "write_failure":

        def failure(*_args):
            raise RuntimeError("render failed")

        monkeypatch.setattr(validation, "render_summary", failure)
    else:
        monkeypatch.setattr(
            validation, "replay_surface_trace", lambda _path: SimpleNamespace(matches=False)
        )
    output = tmp_path / "failed"
    with pytest.raises((ValueError, RuntimeError)):
        validation.generate_surface_contact_validation(output, case_indices=[16])
    assert not output.exists()
    assert not list(tmp_path.glob(".surface-contact-validation-*"))


@pytest.mark.parametrize("kind", ("nonempty", "symlink", "archive", "ancestor"))
def test_output_guard_rejects_before_running(tmp_path, fake_trials, kind):
    calls, _ = fake_trials
    output = tmp_path / "output"
    sentinel = tmp_path / "existing"
    sentinel.mkdir()
    (sentinel / "keep").write_text("retain")
    if kind == "nonempty":
        output = sentinel
    elif kind == "symlink":
        output.symlink_to(sentinel)
    elif kind == "archive":
        output = Path("results/franka_surface_development/derived")
    else:
        output = Path("results")
    with pytest.raises(ValueError):
        validation.generate_surface_contact_validation(output, case_indices=[0])
    assert not calls
    assert (sentinel / "keep").read_text() == "retain"


def test_repair_checks_use_exact_limits_and_have_no_tangent_gate():
    values = {
        "contact_ratio_pct": 99.0,
        "force_rmse_n": 2.0,
        "peak_force_n": 35.0,
        "saturation_pct": 0.0,
        "tangent_rmse_mm": 1000.0,
    }
    assert validation.repair_failures(values) == ()
    assert validation.repair_failures({**values, "saturation_pct": 1e-15}) == ("saturation",)
    assert validation.repair_failures({**values, "contact_ratio_pct": 98.999}) == ("contact_ratio",)


def test_missing_contact_is_not_a_paired_peak_improvement():
    base = {
        "profile": "legacy",
        "arm": "surface_exact",
        "case_index": 0,
        "has_raw_contact": True,
        **dict.fromkeys((*validation.METRICS, *validation.AUXILIARY_METRICS), 10.0),
    }
    smooth = {
        **base,
        "profile": "smooth",
        "has_raw_contact": False,
        "peak_force_n": 0.0,
        "first_raw_contact_time_s": None,
    }
    pair = validation.paired_deltas([base, smooth])[0]
    assert not pair["contact_in_both_profiles"]
    assert all(pair[key] is None for key in validation.CONTACT_METRICS)


def test_four_short_real_smooth_trials_feed_metrics_and_replay(tmp_path):
    case = validation.development_cases()[16]
    case["config"] = replace(
        case["config"], duration=0.02, evaluation_start=0.0, contact_model="smooth"
    )
    for arm, (kind, offset) in validation.ARMS.items():
        yaw = 0.0 if offset is None else case["task"].yaw_deg + offset
        angle = np.deg2rad(yaw)
        frame = validation.SurfaceFrame.from_normal(np.array([np.cos(angle), np.sin(angle), 0]))
        result = validation.run_surface_trial(
            frame,
            scenario=case["scenario"],
            config=case["config"],
            task=case["task"],
            controller_kind=kind,
        )
        metrics = validation._metrics(result)
        auxiliary = validation._auxiliary_metrics(result, case)
        assert np.isfinite(metrics["force_rmse_n"])
        assert auxiliary["max_geometric_separation_um"] >= 0
        path = validation.save_surface_trace(tmp_path / f"{arm}.npz", result.trace)
        replay = replay_surface_trace(path)
        assert replay.sample_count == 10
        assert replay.matches
