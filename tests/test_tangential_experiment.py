import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import compliant_control_lab.tangential_experiment as experiment
from compliant_control_lab.surface_replay import SurfaceReplayResult


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def fake_trials(monkeypatch):
    archived, _, _ = experiment._load_baseline(Path("results/franka_surface_contact_fix"))
    calls, saved = [], []

    def trial(frame, *, scenario, config, task, controller_kind):
        calls.append((frame, scenario, config, task, controller_kind))
        index = int(scenario.name.rsplit("_", 1)[1])
        frozen = archived[index]
        metrics = {
            name: float(frozen[name]) if frozen[name] else None for name in experiment.METRICS
        }
        metrics["has_raw_contact"] = True
        if controller_kind != "surface_adaptive":
            metrics["tangent_rmse_mm"] *= 0.4 if controller_kind == "surface_friction" else 0.8
            metrics["force_rmse_n"] += 0.1
        normal = frame.rotation[:, 0]
        trace = {
            "time": np.array([1.5, 2.0, 3.0]),
            "position": np.tile(np.array([0.4, 0, 0]) - 0.024 * normal, (3, 1)),
            "true_contact_gap_m": np.full(3, -0.001),
            "torque_projection_scale": np.array([1.0, 0.8, 1.0]),
            "requested_tangential_force_world": np.tile(2 * frame.rotation[:, 1], (3, 1)),
        }
        return SimpleNamespace(trace=trace, metrics=lambda: metrics)

    def save(path, trace):
        saved.append(path.name)
        np.savez_compressed(path, **trace)

    monkeypatch.setattr(experiment, "run_surface_trial", trial)
    monkeypatch.setattr(experiment, "save_surface_trace", save)
    monkeypatch.setattr(
        experiment,
        "replay_surface_trace",
        lambda _: SurfaceReplayResult(
            sample_count=3,
            max_wrench_error=0,
            max_torque_error=0,
            matches=True,
            controller_kind="surface_adaptive",
            controller_name="fake",
            controller_supplied=False,
        ),
    )
    monkeypatch.setattr(
        experiment, "_plot_representative", lambda path, *_: path.write_bytes(b"plot")
    )
    monkeypatch.setattr(experiment, "_source_hashes", lambda: {"test": "unchanged"})
    return calls, saved


def test_complete_protocol_fixed_pairs_and_atomic_artifact_integrity(tmp_path, fake_trials):
    calls, saved = fake_trials
    output = experiment.generate_tangential_experiment(tmp_path / "full")
    assert len(calls) == 81
    assert all(call[-1] == "surface_adaptive" for call in calls[:24])
    for index in range(24):
        baseline = calls[index]
        assert baseline[2].contact_model == "smooth" and baseline[2].duration == 4.5
        for method_index, kind in enumerate(experiment.METHODS.values()):
            paired = calls[method_index * 24 + index]
            np.testing.assert_array_equal(baseline[0].rotation, paired[0].rotation)
            assert paired[1:4] == baseline[1:4] and paired[-1] == kind
    for group, mu in enumerate((0.25, 0.45, 0.65)):
        for call in calls[72 + 3 * group : 75 + 3 * group]:
            assert call[1].wall_sliding_friction == call[1].tool_sliding_friction == mu
            assert call[2] == replace(calls[16][2], duration=12)
            assert call[3] == calls[16][3]
    rows = read_rows(output / "comparison.csv")
    long_rows = read_rows(output / "long_comparison.csv")
    pairs = read_rows(output / "paired_deltas.csv")
    assert len(rows) == len({(r["method"], r["case_index"]) for r in rows}) == 72
    assert (
        len(long_rows) == len({(r["method"], r["true_sliding_friction"]) for r in long_rows}) == 9
    )
    assert len(pairs) == 48 and len(read_rows(output / "long_paired_deltas.csv")) == 6
    assert all(float(row["baseline_metric_max_abs_error"]) == 0 for row in rows[:24])
    assert all(float(pair["force_rmse_n"]) == pytest.approx(0.1) for pair in pairs)
    assert len(saved) == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["new_holdout"] is manifest["is_subset"] is False
    assert len(manifest["case_configurations"]) == 81
    assert manifest["long_experiment_identity"] != manifest["experiment_identity"]
    assert manifest["main_outcomes"]["friction"]["overall_pass"] is True
    assert manifest["main_outcomes"]["integral"]["overall_pass"] is False
    assert manifest["main_outcomes"]["integral"]["all_case_checks_pass"] is True
    for method in ("integral", "friction"):
        parameters = manifest["controller_parameters"][method]["tangential"]
        assert parameters == {
            "mode": method,
            "integral_gain": 800,
            "nominal_mu": 0.45,
            "max_force": 6,
            "velocity_scale": 0.005,
            # Recorded constructor defaults include the opt-in online fields;
            # these are inactive for both legacy modes in this experiment.
            "adaptation_gain": 800.0,
            "velocity_error_time": 0.05,
            "force_regularizer": 2.0,
            "max_equivalent_mu": 0.9,
            "min_update_speed": 0.005,
            "force_slew_rate": 20.0,
            "coefficient_rate_limit": 0.3,
            "motion_confirm_time": 0.05,
        }
    assert manifest["baseline_archive"]["directory"] == "results/franka_surface_contact_fix"
    for name, digest in manifest["artifact_sha256"].items():
        assert experiment._sha256(output / name) == digest
    assert (output / "COMPLETE").read_text().strip() == experiment._sha256(output / "manifest.json")
    assert set(manifest["replay_checks"]) == set(saved)
    summary = (output / "summary.md").read_text()
    assert "NOT a new holdout" in summary and "NOT pooled" in summary
    assert "integral: tangent median ratio=0.8" in summary


def test_subset_can_skip_long_and_omit_representative(tmp_path, fake_trials):
    calls, saved = fake_trials
    output = experiment.generate_tangential_experiment(
        tmp_path / "subset", case_indices=[0], skip_long=True
    )
    assert len(calls) == 3 and not saved
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["is_subset"] and manifest["long_experiment_identity"] is None
    assert not (output / "long_comparison.csv").exists()


@pytest.mark.parametrize(
    "indices,error",
    [
        ([], ValueError),
        ([0, 0], ValueError),
        ([True], TypeError),
        ([0.5], TypeError),
        ([24], IndexError),
    ],
)
def test_invalid_indices_never_run(tmp_path, fake_trials, indices, error):
    with pytest.raises(error):
        experiment.generate_tangential_experiment(tmp_path / "bad", case_indices=indices)
    assert not fake_trials[0]


def test_full_cannot_omit_long(tmp_path, fake_trials):
    with pytest.raises(ValueError, match="requires the long"):
        experiment.generate_tangential_experiment(tmp_path / "bad", skip_long=True)
    assert not fake_trials[0]


def test_corrupt_baseline_bytes_rejected_before_simulation(tmp_path, fake_trials):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "comparison.csv").write_text("damaged")
    (baseline / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_identity": "surface-contact-model-repair-v1",
                "artifact_sha256": {"comparison.csv": "0" * 64},
            }
        )
    )
    (baseline / "COMPLETE").write_text(experiment._sha256(baseline / "manifest.json"))
    with pytest.raises(ValueError, match="baseline artifact hash mismatch"):
        experiment.generate_tangential_experiment(
            tmp_path / "bad",
            baseline_dir=baseline,
            case_indices=[0],
            skip_long=True,
        )
    assert not fake_trials[0] and not (tmp_path / "bad").exists()


def test_changed_case_settings_rejected_before_simulation(tmp_path, fake_trials, monkeypatch):
    cases = experiment.development_cases()
    cases[0]["scenario"] = replace(cases[0]["scenario"], position_noise_std_m=0)
    monkeypatch.setattr(experiment, "development_cases", lambda: cases)
    with pytest.raises(ValueError, match="baseline constructor configuration mismatch"):
        experiment.generate_tangential_experiment(
            tmp_path / "bad", case_indices=[0], skip_long=True
        )
    assert not fake_trials[0]


def test_baseline_metric_drift_stops_before_any_new_method(tmp_path, fake_trials, monkeypatch):
    original = experiment.run_surface_trial

    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        metrics = result.metrics()
        metrics["force_rmse_n"] += 2e-10
        result.metrics = lambda: metrics
        return result

    monkeypatch.setattr(experiment, "run_surface_trial", drift)
    with pytest.raises(ValueError, match="baseline metric mismatch"):
        experiment.generate_tangential_experiment(
            tmp_path / "bad", case_indices=[0], skip_long=True
        )
    assert len(fake_trials[0]) == 1
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("failure", ("source", "baseline", "replay", "write"))
def test_failures_do_not_publish_partial_output(tmp_path, fake_trials, monkeypatch, failure):
    calls, _ = fake_trials
    if failure == "source":
        monkeypatch.setattr(experiment, "_source_hashes", lambda: {"source": str(len(calls))})
    elif failure == "baseline":
        original = experiment._load_baseline

        def changed(path):
            rows, configs, identity = original(path)
            if calls:
                identity["manifest_sha256"] = "changed"
            return rows, configs, identity

        monkeypatch.setattr(experiment, "_load_baseline", changed)
    elif failure == "replay":
        monkeypatch.setattr(
            experiment, "replay_surface_trace", lambda _: SimpleNamespace(matches=False)
        )
    else:
        monkeypatch.setattr(
            experiment,
            "render_summary",
            lambda *_: (_ for _ in ()).throw(RuntimeError("write failure")),
        )
    with pytest.raises((ValueError, RuntimeError)):
        experiment.generate_tangential_experiment(
            tmp_path / "bad", case_indices=[16], skip_long=True
        )
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".tangential-experiment-*"))


def test_output_guard_preserves_nonempty_archives_and_symlinks(tmp_path, fake_trials):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep").write_text("user data")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "new")
    for target in (occupied, link, Path("results/franka_surface_contact_fix"), Path("results")):
        with pytest.raises(ValueError):
            experiment.generate_tangential_experiment(target, case_indices=[0], skip_long=True)
    assert not fake_trials[0] and (occupied / "keep").read_text() == "user data"


def test_auxiliary_window_units_and_check_boundaries():
    case = experiment.development_cases()[8]
    trace = {
        "time": np.array([0, 1.5, 2]),
        "position": np.array([[0.378, 0, 0], [0.376, 0, 0], [0.374, 0, 0]]),
        "true_contact_gap_m": np.array([-0.003, -0.001, 0.001]),
        "torque_projection_scale": np.array([1, 1, 0.5]),
        "requested_tangential_force_world": np.array([[0, 0, 0], [0, 3, 4], [0, 0, 0]]),
    }
    values = experiment._auxiliary_metrics(SimpleNamespace(trace=trace), case)
    assert values == pytest.approx(
        {
            "max_penetration_mm": 3,
            "evaluation_median_penetration_mm": 0.5,
            "projection_pct": 100 / 3,
            "max_compensation_force_n": 5,
        }
    )
    baseline = {"force_rmse_n": 0.0}
    row = {
        "has_raw_contact": True,
        "contact_ratio_pct": 99,
        "peak_force_n": 35,
        "saturation_pct": 0,
        "force_rmse_n": 0.2,
    }
    assert experiment.case_failures(row, baseline) == ()
    assert experiment.case_failures({**row, "force_rmse_n": 0.200001}, baseline) == (
        "paired_force_rmse",
    )
    assert "contact_ratio" in experiment.case_failures({**row, "has_raw_contact": False}, baseline)


def test_missing_contact_cannot_earn_peak_or_time_improvement():
    baseline = {
        "method": "baseline",
        "case_index": 0,
        "true_sliding_friction": 0.45,
        "has_raw_contact": True,
        **{key: 1.0 for key in (*experiment.METRICS, *experiment.AUXILIARY_METRICS)},
    }
    missing = {
        **baseline,
        "method": "friction",
        "has_raw_contact": False,
        "peak_force_n": 0,
        "seconds_over_35_n": 0,
        "first_raw_contact_time_s": None,
    }
    pair = experiment.paired_deltas([baseline, missing])[0]
    assert not pair["contact_in_both_methods"]
    assert all(pair[key] is None for key in experiment.CONTACT_METRICS)
