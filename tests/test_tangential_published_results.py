"""Audit published tangential aggregates and retained inputs without plant reruns."""

import csv
import hashlib
import json
import re
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.surface_experiment import CONTACT_METRICS, METRICS
from compliant_control_lab.surface_replay import replay_surface_trace
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    SurfaceTask,
    SurfaceTrialResult,
)
from compliant_control_lab.tangential_experiment import AUXILIARY_METRICS, render_summary

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "results/franka_tangential_development"
BASELINE = REPO / "results/franka_surface_contact_fix"
DIAGNOSIS = REPO / "results/franka_tangential_diagnostics/diagnosis.json"
METHODS = {
    "baseline": "surface_adaptive",
    "integral": "surface_integral",
    "friction": "surface_friction",
}
NUMERIC_FIELDS = (*METRICS, *AUXILIARY_METRICS)
# Historical identity, deliberately not compared with a future working-tree script.
DIAGNOSTIC_SCRIPT_SHA = "3e83576c5b6eda8a1321f3bc14203c75008b10c66cc2c326b8afd2db0bf7b234"
TRACES = (
    ("representative_case_16_baseline.npz", "main", "baseline", 2250),
    ("representative_case_16_integral.npz", "main", "integral", 2250),
    ("representative_case_16_friction.npz", "main", "friction", 2250),
    ("long_case_16_mu_0.45_friction.npz", "long", "friction", 6000),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def typed_rows(path):
    records = csv_rows(path)
    for row in records:
        row["case_index"] = int(row["case_index"])
        row["true_sliding_friction"] = float(row["true_sliding_friction"])
        assert row["has_raw_contact"] in {"True", "False"}
        row["has_raw_contact"] = row["has_raw_contact"] == "True"
        for field in NUMERIC_FIELDS:
            row[field] = float(row[field]) if row[field] else None
    return records


@pytest.fixture(scope="module")
def manifest():
    assert (REPORT / "COMPLETE").is_file(), "published archive is required, not skipped"
    return read_json(REPORT / "manifest.json")


@pytest.fixture(scope="module")
def rows():
    return typed_rows(REPORT / "comparison.csv")


@pytest.fixture(scope="module")
def long_rows():
    return typed_rows(REPORT / "long_comparison.csv")


def test_complete_artifacts_and_separate_unique_matrices(manifest, rows, long_rows):
    assert (REPORT / "COMPLETE").read_text().strip() == sha256(REPORT / "manifest.json")
    assert manifest["schema_version"] == 1
    assert manifest["experiment_identity"] == "tangential-compensation-public24-v1"
    assert manifest["long_experiment_identity"] == "tangential-compensation-long-friction-v1"
    assert manifest["new_holdout"] is manifest["is_subset"] is False
    assert manifest["selected_case_indices"] == list(range(24))
    assert manifest["methods"] == METHODS
    expected = {
        "comparison.csv",
        "paired_deltas.csv",
        "long_comparison.csv",
        "long_paired_deltas.csv",
        "summary.md",
        "representative_case_16.png",
        *(name for name, *_ in TRACES),
    }
    assert set(manifest["artifact_sha256"]) == expected
    assert {p.name for p in REPORT.iterdir()} == expected | {"manifest.json", "COMPLETE"}
    for name, checksum in manifest["artifact_sha256"].items():
        assert not (REPORT / name).is_symlink()
        assert sha256(REPORT / name) == checksum
    assert (REPORT / "representative_case_16.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for category, records, grid in (
        ("main", rows, {(m, i, 0.45) for m in METHODS for i in range(24)}),
        ("long", long_rows, {(m, 16, mu) for m in METHODS for mu in (0.25, 0.45, 0.65)}),
    ):
        assert len(records) == len(grid)
        assert {r["category"] for r in records} == {category}
        assert {(r["method"], r["case_index"], r["true_sliding_friction"]) for r in records} == grid
        configs = [c for c in manifest["case_configurations"] if c["category"] == category]
        assert len(configs) == len(grid)
        assert {
            (c["method"], c["case_index"], c["scenario"]["wall_sliding_friction"]) for c in configs
        } == grid
    assert len(manifest["case_configurations"]) == 81
    historical = manifest["source_and_assets_sha256"]
    assert {
        "tangential_experiment.py",
        "tangential_compensation.py",
        "surface_simulation.py",
    } <= historical.keys()
    assert any(name.startswith("assets/") for name in historical)
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in historical.values())
    assert {"python", "numpy", "mujoco", "compliant-control-lab"} == manifest["versions"].keys()


def test_baseline_all_24_metrics_exact_and_historical_archive_identity(manifest, rows):
    identity = manifest["baseline_archive"]
    archived = read_json(BASELINE / "manifest.json")
    assert identity["directory"] == str(BASELINE.relative_to(REPO))
    assert identity["manifest_sha256"] == sha256(BASELINE / "manifest.json")
    assert (BASELINE / "COMPLETE").read_text().strip() == identity["manifest_sha256"]
    assert identity["complete_sha256"] == sha256(BASELINE / "COMPLETE")
    assert identity["artifact_sha256"] == archived["artifact_sha256"]
    for name, checksum in identity["artifact_sha256"].items():
        assert sha256(BASELINE / name) == checksum
    frozen = {
        int(r["case_index"]): r
        for r in csv_rows(BASELINE / "comparison.csv")
        if (r["profile"], r["arm"]) == ("smooth", "surface_exact")
    }
    baseline = [r for r in rows if r["method"] == "baseline"]
    assert len(frozen) == len(baseline) == 24
    for row in baseline:
        old = frozen[row["case_index"]]
        for field in ("scenario", "simulation_seed", "controller_yaw_deg"):
            assert row[field] == old[field]
        assert str(row["has_raw_contact"]) == old["has_raw_contact"]
        for field in METRICS:
            assert row[field] == (float(old[field]) if old[field] else None)
        assert float(row["baseline_metric_max_abs_error"]) == 0
    reference = manifest["baseline_trace_reference"]
    name = "representative_case_16_smooth_surface_exact.npz"
    assert reference == {
        "path": str((BASELINE / name).relative_to(REPO)),
        "sha256": identity["artifact_sha256"][name],
    }


def test_fixed_paired_constructors_and_both_actual_geometry_friction(manifest, rows, long_rows):
    old = {
        c["case_index"]: c
        for c in read_json(BASELINE / "manifest.json")["case_configurations"]
        if (c["profile"], c["arm"]) == ("smooth", "surface_exact")
    }
    grid = list(product((-15.0, 0.0, 15.0), (0.005, 0.012), (0.10, 0.13), (11, 29)))
    indexed = {
        (r["category"], r["method"], r["case_index"], r["true_sliding_friction"]): r
        for r in rows + long_rows
    }
    for case in manifest["case_configurations"]:
        category, method, index = (case[k] for k in ("category", "method", "case_index"))
        scenario, config = case["scenario"], case["config"]
        yaw, tau, mass, seed = grid[index]
        mu = scenario["wall_sliding_friction"]
        assert scenario["tool_sliding_friction"] == mu
        expected_scenario = {**old[index]["scenario"], "tool_sliding_friction": 0.45}
        if category == "long":
            expected_scenario.update(tool_sliding_friction=mu, wall_sliding_friction=mu)
        assert scenario == expected_scenario
        assert (
            scenario["wall_yaw_deg"],
            scenario["wall_time_constant"],
            scenario["tool_mass_kg"],
            config["seed"],
        ) == (yaw, tau, mass, seed)
        assert config == {**old[index]["config"], "duration": 12.0 if category == "long" else 4.5}
        assert case["task"] == old[index]["task"] == {"nominal_plane_x_m": 0.4, "yaw_deg": yaw}
        assert config["contact_model"] == "smooth" and config["timestep"] == 0.002
        assert case["controller_kind"] == METHODS[method]
        angle = np.deg2rad(yaw)
        expected_rotation = [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
        np.testing.assert_allclose(case["controller_frame_rotation"], expected_rotation, atol=1e-15)
        row = indexed[category, method, index, mu]
        for field, value in {
            "duration_s": config["duration"],
            "simulation_seed": seed,
            "controller_yaw_deg": yaw,
            "wall_yaw_deg": yaw,
            "wall_time_constant_s": tau,
            "tool_mass_kg": mass,
        }.items():
            assert float(row[field]) == value
        assert row["scenario"] == scenario["name"]
    parameters = manifest["controller_parameters"]
    baseline = parameters["baseline"]
    assert baseline["tangential"] is None
    assert baseline["base"]["base"]["tangential_stiffness"] == [0, 450, 450]
    assert baseline["base"]["base"]["tangential_damping"] == [0, 38, 38]
    for method in ("integral", "friction"):
        assert parameters[method] == {
            **baseline,
            "tangential": {
                "mode": method,
                "integral_gain": 800.0,
                "nominal_mu": 0.45,
                "max_force": 6.0,
                "velocity_scale": 0.005,
            },
        }


@pytest.mark.parametrize("filename,category,method,samples", TRACES)
def test_retained_trace_metrics_geometry_compensation_and_replay(
    manifest, rows, long_rows, filename, category, method, samples
):
    path = REPORT / filename
    replay = replay_surface_trace(path)
    assert asdict(replay) == manifest["replay_checks"][filename]
    assert replay.matches and replay.sample_count == samples
    assert replay.max_wrench_error == replay.max_torque_error == 0
    case = next(
        c
        for c in manifest["case_configurations"]
        if (c["category"], c["method"], c["case_index"], c["scenario"]["wall_sliding_friction"])
        == (category, method, 16, 0.45)
    )
    with np.load(path, allow_pickle=False) as stored:
        trace = {name: stored[name] for name in stored.files}
    result = SurfaceTrialResult(
        trace,
        SurfaceScenario(**case["scenario"]),
        SurfaceSimulationConfig(**case["config"]),
        SurfaceTask(**case["task"]),
    )
    row = next(
        r
        for r in rows + long_rows
        if (r["category"], r["method"], r["case_index"], r["true_sliding_friction"])
        == (category, method, 16, 0.45)
    )
    assert str(trace["controller_kind"]) == METHODS[method]
    assert str(trace["contact_model"]) == "smooth"
    np.testing.assert_array_equal(
        trace["controller_frame_rotation"], case["controller_frame_rotation"]
    )
    measured = result.metrics()
    assert measured["has_raw_contact"] is row["has_raw_contact"]
    for field in METRICS:
        assert measured[field] == pytest.approx(row[field], rel=0, abs=1e-10)
    normal = np.array([np.cos(np.deg2rad(15)), np.sin(np.deg2rad(15)), 0])
    gap = (np.array([0.4, 0, 0]) - trace["position"]) @ normal - 0.025
    np.testing.assert_allclose(trace["true_contact_gap_m"], gap, rtol=0, atol=1e-12)
    window = trace["time"] >= result.config.evaluation_start
    penetration = np.maximum(-gap, 0) * 1000
    correction = trace["requested_tangential_force_world"]
    force_norm = np.linalg.norm(correction, axis=1)
    auxiliary = (
        penetration.max(),
        np.median(penetration[window]),
        100 * np.mean(trace["torque_projection_scale"] < 1 - 1e-12),
        force_norm.max(),
    )
    for field, value in zip(AUXILIARY_METRICS, auxiliary):
        assert row[field] == pytest.approx(value, rel=0, abs=1e-10)
    np.testing.assert_allclose(correction @ normal, 0, rtol=0, atol=1e-12)
    assert force_norm.max() <= 6 + 1e-12
    if method == "baseline":
        np.testing.assert_array_equal(correction, np.zeros_like(correction))
    loaded = window & (trace["true_normal_force"] > 0.5)
    assert np.mean(trace["true_tangent_force_n"][loaded]) > 1
    assert np.median(
        trace["true_tangent_force_n"][loaded] / trace["true_normal_force"][loaded]
    ) == pytest.approx(0.45, abs=1e-10)


@pytest.mark.parametrize(
    "category,filename,count",
    [("main", "paired_deltas.csv", 48), ("long", "long_paired_deltas.csv", 6)],
)
def test_paired_deltas_and_missing_contact_policy(rows, long_rows, category, filename, count):
    records = rows if category == "main" else long_rows
    indexed = {(r["method"], r["case_index"], r["true_sliding_friction"]): r for r in records}
    pairs = csv_rows(REPORT / filename)
    assert len(pairs) == count
    assert {
        (p["method"], int(p["case_index"]), float(p["true_sliding_friction"])) for p in pairs
    } == {key for key in indexed if key[0] != "baseline"}
    for pair in pairs:
        index, mu = int(pair["case_index"]), float(pair["true_sliding_friction"])
        row, baseline = indexed[pair["method"], index, mu], indexed["baseline", index, mu]
        contact = row["has_raw_contact"] and baseline["has_raw_contact"]
        assert pair["contact_in_both_methods"] == str(contact)
        for field in NUMERIC_FIELDS:
            if (
                row[field] is None
                or baseline[field] is None
                or (field in CONTACT_METRICS and not contact)
            ):
                assert pair[field] == ""
            else:
                assert float(pair[field]) == pytest.approx(
                    row[field] - baseline[field], rel=0, abs=1e-12
                )


def test_failures_outcomes_and_summary_reconstructed(manifest, rows, long_rows):
    assert manifest["engineering_checks"] == {
        "maximum_main_median_tangent_ratio": 0.5,
        "minimum_contact_ratio_pct": 99.0,
        "maximum_raw_peak_force_n": 35.0,
        "required_saturation_pct": 0.0,
        "maximum_paired_force_rmse_increase_n": 0.2,
    }
    for records in (rows, long_rows):
        baseline = {
            (r["case_index"], r["true_sliding_friction"]): r
            for r in records
            if r["method"] == "baseline"
        }
        for row in records:
            reference = baseline[row["case_index"], row["true_sliding_friction"]]
            failed = [
                name
                for name, passed in (
                    ("contact_ratio", row["has_raw_contact"] and row["contact_ratio_pct"] >= 99),
                    ("raw_peak_force", row["peak_force_n"] <= 35),
                    ("saturation", row["saturation_pct"] == 0),
                    ("paired_force_rmse", row["force_rmse_n"] - reference["force_rmse_n"] <= 0.2),
                )
                if not passed
            ]
            assert row["failed_case_checks"] == ";".join(failed)
            assert row["case_checks_pass"] == ("no" if failed else "yes")
        assert sum(r["case_checks_pass"] == "no" for r in records) == 0
    baseline_median = np.median([r["tangent_rmse_mm"] for r in rows if r["method"] == "baseline"])
    for method in ("integral", "friction"):
        selected = [r for r in rows if r["method"] == method]
        ratio = np.median([r["tangent_rmse_mm"] for r in selected]) / baseline_median
        assert manifest["main_outcomes"][method] == {
            "median_tangent_ratio": ratio,
            "tangent_criterion_pass": bool(ratio <= 0.5),
            "all_case_checks_pass": True,
            "overall_pass": bool(ratio <= 0.5),
        }
    assert manifest["main_outcomes"]["integral"]["overall_pass"] is False
    assert manifest["main_outcomes"]["friction"]["overall_pass"] is True
    summary = (REPORT / "summary.md").read_text(encoding="utf-8")
    assert summary == render_summary(rows, long_rows, manifest["scope"])
    assert "NOT pooled with the main grid" in summary and "NOT a new holdout" in summary
    assert "Failures are retained" in summary and "metrics only" in summary


def test_diagnostic_historical_identity_fixed_four_plus_three_probes(manifest):
    diagnosis = read_json(DIAGNOSIS)
    assert diagnosis["identity"] == "tangential-tracking-diagnosis-v1"
    assert diagnosis["script_sha256"] == DIAGNOSTIC_SCRIPT_SHA
    assert diagnosis["source_and_assets_sha256"] == manifest["source_and_assets_sha256"]
    assert diagnosis["engine_versions"] == {
        key: manifest["versions"][key] for key in ("mujoco", "numpy", "python")
    }
    assert "aggregates only, not full raw traces" in diagnosis["scope"]
    assert "not real-material identification" in diagnosis["force_depth_scope"]
    tracking, static = diagnosis["tracking_probes"], diagnosis["static_force_depth"]
    assert [r["probe"] for r in tracking] == ["baseline", "friction_zero", "double_k", "slow"]
    assert [r["probe"] for r in static] == ["load_6", "load_12", "load_18"]
    overrides = [
        {},
        {"both_geom_sliding_friction": 0},
        {"tangent_stiffness_multiplier": 2},
        {"tangent_clock_rate": 0.5},
    ]
    historical_rmse = [11.651333023785067, 1.9593284173032022, 6.447088712150435, 9.541975370618388]
    for row, override, rmse in zip(tracking, overrides, historical_rmse):
        assert row["overrides"] == override
        assert row["metrics"]["tangent_rmse_mm"] == pytest.approx(rmse, rel=0, abs=1e-12)
        assert row["task_variant"] == ("SlowTask" if row["probe"] == "slow" else "SurfaceTask")
    for row in tracking + static:
        is_static = row in static
        target = float(row["probe"].split("_")[1]) if is_static else 12.0
        assert row["config"] == {
            "duration": 3.0,
            "timestep": 0.002,
            "seed": 11,
            "contact_model": "smooth",
            "target_force": target,
            "evaluation_start": 2.0 if is_static else 1.5,
            "force_filter_time_constant": 0.02,
        }
        assert row["task"] == {"yaw_deg": 0.0, "nominal_plane_x_m": 0.4}
        assert row["scenario"] == {
            "name": "tangential_diagnostic",
            "wall_yaw_deg": 0,
            "wall_time_constant": 0.012,
            "wall_sliding_friction": 0.45,
            "tool_sliding_friction": 0.45,
            "tool_mass_kg": 0.1,
            "nominal_tool_mass_kg": 0.1,
            "position_noise_std_m": 0,
            "force_noise_std_n": 0,
            "torque_noise_std_nm": 0,
            "force_bias_sensor_n": [0, 0, 0],
            "torque_bias_sensor_nm": [0, 0, 0],
            "delay_steps": 0,
            "bias_compensation_scale": 1.0,
        }
        assert row["metrics"]["contact_ratio_pct"] == 100
        assert row["metrics"]["saturation_pct"] == 0
        if is_static:
            assert row["task_variant"] == "HoldTask" and row["overrides"] == {"hold_after_s": 1.2}
            assert 0 < row["median_penetration_mm"] <= row["max_penetration_mm"]
            assert abs(row["actual_normal_force_median_n"] - target) < 0.5
    assert np.all(np.diff([r["median_penetration_mm"] for r in static]) > 0)


def test_diagnostic_directional_lag_and_pd_decomposition_recomputed(manifest):
    recorded = read_json(DIAGNOSIS)["reference_trace_analysis"]
    reference = manifest["baseline_trace_reference"]
    assert recorded["reference_path"] == reference["path"]
    assert recorded["reference_sha256"] == reference["sha256"] == sha256(REPO / reference["path"])
    assert recorded["window_s"] == [1.5, 4.5]
    with np.load(REPO / reference["path"], allow_pickle=False) as stored:
        trace = {name: stored[name] for name in stored.files}
    window = (trace["time"] >= 1.5) & (trace["time"] < 4.5)
    assert recorded["samples"] == np.count_nonzero(window) == 1500
    rotation = trace["controller_frame_rotation"]
    local = {
        key: (trace[key][window] @ rotation)[:, 1:]
        for key in (
            "target_position",
            "position",
            "measured_position",
            "target_linear_velocity",
            "measured_linear_velocity",
        )
    }
    error = local["target_position"] - local["position"]
    measured_error = local["target_position"] - local["measured_position"]
    velocity = local["target_linear_velocity"]
    speed = np.linalg.norm(velocity, axis=1)
    assert np.all(speed > 0)
    direction = velocity / speed[:, None]
    lag = np.sum(error * direction, axis=1)
    scale = 1 + 0.25 * np.clip(np.linalg.norm(measured_error, axis=1) / 0.02, 0, 1)
    spring = 450 * scale[:, None] * measured_error
    damping = 38 * np.sqrt(scale[:, None]) * (velocity - local["measured_linear_velocity"])
    commanded = (trace["commanded_wrench"][window, :3] @ rotation)[:, 1:]
    reconstructed = (spring + damping) * trace["torque_projection_scale"][window, None]
    measured = {
        "mean_signed_lag_mm": 1000 * np.mean(lag),
        "behind_target_pct": 100 * np.mean(lag > 0),
        "lag_direction_cosine_median": np.median(lag / np.linalg.norm(error, axis=1)),
        "cross_motion_error_rmse_mm": 1000
        * np.sqrt(np.mean(np.sum((error - lag[:, None] * direction) ** 2, axis=1))),
        "target_speed_mean_m_s": np.mean(speed),
        "measured_speed_mean_m_s": np.mean(
            np.linalg.norm(local["measured_linear_velocity"], axis=1)
        ),
        "scheduled_stiffness_median_n_m": np.median(450 * scale),
        "command_force_mean_n": np.mean(np.linalg.norm(commanded, axis=1)),
        "spring_force_mean_n": np.mean(np.linalg.norm(spring, axis=1)),
        "damping_force_mean_n": np.mean(np.linalg.norm(damping, axis=1)),
        "true_contact_tangent_force_mean_n": np.mean(trace["true_tangent_force_n"][window]),
        "pd_reconstruction_max_error_n": np.max(np.abs(commanded - reconstructed)),
    }
    for key, value in measured.items():
        assert recorded[key] == pytest.approx(value, rel=0, abs=1e-12)
    assert measured["pd_reconstruction_max_error_n"] < 1e-10
    assert "do not establish vector force balance or causality" in recorded["interpretation"]
