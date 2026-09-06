"""Audit public repair artifacts and diagnostics without rerunning the plant grid."""

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.surface_contact_validation import AUXILIARY_METRICS, PROFILES
from compliant_control_lab.surface_experiment import ARMS, METRICS
from compliant_control_lab.surface_replay import replay_surface_trace
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    SurfaceTask,
    SurfaceTrialResult,
)

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "results/franka_surface_contact_fix"
BASELINE = REPO / "results/franka_surface_development"
DIAGNOSTICS = REPO / "results/franka_surface_contact_diagnostics"
NUMERIC_FIELDS = (*METRICS, *AUXILIARY_METRICS)
# Historical script identities: future source edits must not invalidate this archive.
SCRIPT_HASHES = {
    "one_factor.json": "e03a8eed947ddc5dc1df6a1e9f2b761423a38d53587b77f23299b8466c823276",
    "step_refinement.json": "407ef3f6f61e700dad84a634dc434f9c4bff48bf7b8362190ddb496496d5d73d",
    "legacy_step_refinement.json": "407ef3f6f61e700dad84a634dc434f9c4bff48bf7b8362190ddb496496d5d73d",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def manifest():
    return read_json(REPORT / "manifest.json")


@pytest.fixture(scope="module")
def rows():
    records = csv_rows(REPORT / "comparison.csv")
    for row in records:
        row["case_index"] = int(row["case_index"])
        assert row["has_raw_contact"] in {"True", "False"}
        row["has_raw_contact"] = row["has_raw_contact"] == "True"
        for field in NUMERIC_FIELDS:
            row[field] = float(row[field]) if row[field] else None
    return records


def case_config(manifest, profile, arm="surface_exact", index=16):
    return next(
        case
        for case in manifest["case_configurations"]
        if (case["profile"], case["arm"], case["case_index"]) == (profile, arm, index)
    )


def representative(manifest, profile, arm="surface_exact"):
    path = (
        REPORT / f"representative_case_16_smooth_{arm}.npz"
        if profile == "smooth"
        else REPO / manifest["legacy_trace_references"][arm]["path"]
    )
    with np.load(path, allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    case = case_config(manifest, profile, arm)
    return SurfaceTrialResult(
        trace,
        SurfaceScenario(**case["scenario"]),
        SurfaceSimulationConfig(**case["config"]),
        SurfaceTask(**case["task"]),
    )


def test_complete_hashes_and_unique_full_grid(manifest, rows):
    assert (REPORT / "COMPLETE").read_text().strip() == sha256(REPORT / "manifest.json")
    assert manifest["schema_version"] == 1
    assert manifest["experiment_identity"] == "surface-contact-model-repair-v1"
    assert manifest["new_holdout"] is manifest["is_subset"] is False
    assert manifest["selected_case_indices"] == list(range(24))
    expected = {
        "comparison.csv",
        "paired_deltas.csv",
        "summary.md",
        "representative_case_16_exact.png",
    }
    expected.update(f"representative_case_16_smooth_{arm}.npz" for arm in ARMS)
    assert set(manifest["artifact_sha256"]) == expected
    assert {path.name for path in REPORT.iterdir()} == expected | {"manifest.json", "COMPLETE"}
    for name, checksum in manifest["artifact_sha256"].items():
        assert sha256(REPORT / name) == checksum
    grid = {(profile, arm, case) for profile in PROFILES for arm in ARMS for case in range(24)}
    for records in (rows, manifest["case_configurations"]):
        assert len(records) == 192
        assert {(r["profile"], r["arm"], r["case_index"]) for r in records} == grid
    historical = manifest["source_and_assets_sha256"]
    assert {
        "surface_simulation.py",
        "surface_contact_validation.py",
        "surface_replay.py",
    } <= historical.keys()
    assert any(name.startswith("assets/") for name in historical)
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in historical.values())


def test_baseline_identity_unchanged_rows_and_contact_model_only_pairs(manifest, rows):
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
        (row["arm"], int(row["case_index"])): row for row in csv_rows(BASELINE / "comparison.csv")
    }
    assert len(frozen) == 96
    for row in rows:
        if row["profile"] != "legacy":
            continue
        old = frozen[row["arm"], row["case_index"]]
        for field in ("scenario", "simulation_seed", "controller_yaw_deg"):
            assert row[field] == old[field]
        assert str(row["has_raw_contact"]) == old["has_raw_contact"]
        for field in METRICS:
            assert row[field] == (float(old[field]) if old[field] else None)
        assert float(row["legacy_metric_max_abs_error"]) == 0
    for arm in ARMS:
        reference = manifest["legacy_trace_references"][arm]
        name = f"representative_case_16_{arm}.npz"
        assert reference["path"] == str((BASELINE / name).relative_to(REPO))
        assert reference["sha256"] == identity["artifact_sha256"][name]
        for index in range(24):
            legacy = case_config(manifest, "legacy", arm, index)
            smooth = case_config(manifest, "smooth", arm, index)
            assert legacy["config"]["contact_model"] == "legacy"
            assert smooth == {
                **legacy,
                "profile": "smooth",
                "config": {**legacy["config"], "contact_model": "smooth"},
            }
            assert legacy["config"] == {
                **archived["all_development_cases"][index]["config"],
                "contact_model": "legacy",
            }
            for field in ("scenario", "task"):
                assert legacy[field] == archived["all_development_cases"][index][field]


@pytest.mark.parametrize("arm", ARMS)
def test_representative_replay_metrics_and_actual_contact_loads(manifest, rows, arm):
    replay = replay_surface_trace(REPORT / f"representative_case_16_smooth_{arm}.npz")
    assert replay.matches and replay.sample_count == 2250
    recorded = manifest["replay_checks"][arm]
    assert recorded["matches"] and recorded["sample_count"] == replay.sample_count
    assert replay.max_wrench_error == recorded["max_wrench_error"] == 0
    assert replay.max_torque_error == recorded["max_torque_error"] == 0
    result = representative(manifest, "smooth", arm)
    trace = result.trace
    assert str(trace["contact_model"]) == "smooth"
    measured = result.metrics()
    row = next(r for r in rows if (r["profile"], r["arm"], r["case_index"]) == ("smooth", arm, 16))
    for field in METRICS:
        assert measured[field] == pytest.approx(row[field], rel=0, abs=1e-10)
    normal = np.array([np.cos(np.deg2rad(15)), np.sin(np.deg2rad(15)), 0])
    gap = (np.array([0.4, 0, 0]) - trace["position"]) @ normal - 0.025
    np.testing.assert_allclose(trace["true_contact_gap_m"], gap, rtol=0, atol=1e-12)
    window = trace["time"] >= result.config.evaluation_start
    loaded = window & (trace["true_normal_force"] > 0.5)
    auxiliary = (
        max(0, gap[window].max()) * 1e6,
        np.mean(trace["true_tangent_force_n"][window]),
        np.median(trace["true_tangent_force_n"][loaded] / trace["true_normal_force"][loaded]),
    )
    for field, value in zip(AUXILIARY_METRICS, auxiliary):
        assert row[field] == pytest.approx(value, rel=0, abs=1e-10)
    assert auxiliary[1] > 1  # Actual nonzero load, not merely a configured friction coefficient.
    assert auxiliary[2] == pytest.approx(0.45, abs=1e-10)
    np.testing.assert_array_equal(
        trace["feedback_raw_wrench_world"][1:], trace["raw_wrench_world"][:-1]
    )


def test_paired_differences_repair_checks_and_summary_reconstructed(manifest, rows):
    indexed = {(r["profile"], r["arm"], r["case_index"]): r for r in rows}
    pairs = csv_rows(REPORT / "paired_deltas.csv")
    assert len(pairs) == 96
    assert {(p["arm"], int(p["case_index"])) for p in pairs} == {
        (a, i) for a in ARMS for i in range(24)
    }
    for pair in pairs:
        legacy, smooth = (indexed[p, pair["arm"], int(pair["case_index"])] for p in PROFILES)
        assert pair["contact_in_both_profiles"] == str(
            legacy["has_raw_contact"] and smooth["has_raw_contact"]
        )
        for field in NUMERIC_FIELDS:
            assert float(pair[field]) == pytest.approx(
                smooth[field] - legacy[field], rel=0, abs=1e-12
            )
    assert manifest["engineering_repair_checks"] == {
        "minimum_contact_ratio_pct": 99.0,
        "maximum_raw_force_rmse_n": 2.0,
        "maximum_raw_peak_force_n": 35.0,
        "required_saturation_pct": 0.0,
    }
    for row in rows:
        failed = [
            name
            for ok, name in (
                (row["contact_ratio_pct"] >= 99, "contact_ratio"),
                (row["force_rmse_n"] <= 2, "raw_force_rmse"),
                (row["peak_force_n"] <= 35, "raw_peak_force"),
                (row["saturation_pct"] == 0, "saturation"),
            )
            if not ok
        ]
        assert row["failed_repair_checks"] == ";".join(failed)
        assert row["repair_pass"] == ("no" if failed else "yes")
        if row["profile"] == "smooth":
            assert row["contact_ratio_pct"] == 100 and not failed
    summary = (REPORT / "summary.md").read_text()
    for profile in PROFILES:
        for arm in ARMS:
            selected = [r for r in rows if (r["profile"], r["arm"]) == (profile, arm)]
            values = [
                np.percentile([r[field] for r in selected], q)
                for field, q in (
                    ("contact_ratio_pct", 0),
                    ("contact_ratio_pct", 50),
                    ("force_rmse_n", 50),
                    ("force_rmse_n", 95),
                    ("peak_force_n", 100),
                    ("peak_force_n", 95),
                    ("tangent_rmse_mm", 50),
                    ("saturation_pct", 100),
                )
            ]
            passes = sum(r["repair_pass"] == "yes" for r in selected)
            line = (
                f"| {profile} / {arm} | "
                + " | ".join(f"{v:.6g}" for v in values)
                + f" | {passes}/24 |"
            )
            assert line in summary.splitlines()


@pytest.mark.parametrize("name", SCRIPT_HASHES)
def test_diagnostic_historical_provenance(manifest, name):
    diagnostic = read_json(DIAGNOSTICS / name)
    assert diagnostic["source_and_assets_sha256"] == manifest["source_and_assets_sha256"]
    assert diagnostic["engine_versions"] == {
        key: manifest["versions"][key] for key in ("mujoco", "numpy", "python")
    }
    assert diagnostic["script_sha256"] == SCRIPT_HASHES[name]
    assert "not holdout" in diagnostic["scope"]


def test_one_factor_recorded_identities_and_friction_are_not_placeholder_values():
    """These JSON rows are aggregates, not retained per-probe raw trajectories."""
    diagnostic = read_json(DIAGNOSTICS / "one_factor.json")
    assert diagnostic["identity"] == "wiping-contact-one-factor-diagnosis-v1"
    assert diagnostic["control_period_s"] == 0.002
    probes = {row["probe"]: row for row in diagnostic["rows"]}
    changes = {
        "baseline": {},
        "hold_position": {"hold_after_s": 1.2},
        "zero_p": {"force_kp": 0},
        "zero_i": {"force_ki": 0},
        "normal_damping_60": {"normal_damping": 60},
        "no_filter": {},
        "filter_5ms": {},
        "friction_zero": {"both_geom_sliding_friction": 0},
        "friction_01": {"both_geom_sliding_friction": 0.1},
        "impedance_half": {"both_geom_initial_impedance": 0.5},
        "impratio_10": {"impratio": 10},
        "smooth": {},
    }
    assert len(diagnostic["rows"]) == 12 and probes.keys() == changes.keys()
    baseline = probes["baseline"]
    for name, row in probes.items():
        assert row["scenario"] == baseline["scenario"] and row["task"] == baseline["task"]
        assert {
            key: value for key, value in row["overrides"].items() if value is not None
        } == changes[name]
        config = {**baseline["config"]}
        if name == "smooth":
            config["contact_model"] = "smooth"
        if name in {"no_filter", "filter_5ms"}:
            config["force_filter_time_constant"] = 0 if name == "no_filter" else 0.005
        assert row["config"] == config
        metrics = row["metrics"]
        assert metrics["evaluation_observed"] and metrics["has_raw_contact"]
        assert all(np.isfinite(metrics[field]) for field in METRICS)
        assert metrics["saturation_pct"] == 0 and metrics["peak_force_n"] > 12
        assert 0 <= row["geometry_separation_pct"] <= 100
        assert (row["maximum_gap_um"] > 0) == (row["geometry_separation_pct"] > 0)
    assert baseline["config"]["contact_model"] == "legacy"
    assert baseline["metrics"]["contact_ratio_pct"] < 70
    assert probes["smooth"]["metrics"]["contact_ratio_pct"] == 100
    assert baseline["median_loaded_friction_ratio"] == pytest.approx(0.45)
    assert probes["smooth"]["median_loaded_friction_ratio"] == pytest.approx(0.45)
    assert probes["friction_01"]["median_loaded_friction_ratio"] == pytest.approx(0.1)
    assert probes["friction_zero"]["median_loaded_friction_ratio"] == pytest.approx(1e-5, abs=1e-10)
    assert probes["hold_position"]["median_loaded_friction_ratio"] < 0.01


@pytest.mark.parametrize(
    "profile,name", [("smooth", "step_refinement.json"), ("legacy", "legacy_step_refinement.json")]
)
def test_step_refinement_sampling_identity_and_n1_reconstructed(manifest, profile, name):
    diagnostic = read_json(DIAGNOSTICS / name)
    assert diagnostic["identity"] == "surface-contact-phase-strict-refinement-v1"
    assert diagnostic["sampling"]["control_and_filter_hz"] == 500
    records = diagnostic["rows"]
    assert [row["substeps"] for row in records] == [1, 4, 8]
    case = case_config(manifest, profile)
    description = manifest["contact_model_description"]
    for row in records:
        n = row["substeps"]
        assert row["config"] == case["config"]
        assert row["scenario"] == {**case["scenario"], "name": "case16"}
        assert row["task"] == case["task"]
        assert row["controller_kind"] == case["controller_kind"]
        assert row["controller_frame_rotation"] == case["controller_frame_rotation"]
        assert row["control_period_s"] == 0.002 and row["physics_period_s"] == 0.002 / n
        assert row["sensor_reads_including_reset"] == 2251
        assert row["engine_options"] == records[0]["engine_options"]
        assert row["resolved_contact_pair"] == {
            "dim": 3,
            "friction": [0.45, 0.45, 0.05, 0.005, 0.005],
            "solimp": description[f"{profile}_solimp"],
            "solref": description["mixed_solref_by_wall_time_constant"]["0.005"],
        }
        for grid, multiplier in (("control_grid", 1), ("all_physics", n)):
            assert row[grid]["sample_count"] == 2250 * multiplier
            assert row[grid]["evaluation_sample_count"] == 1500 * multiplier
            assert all(np.isfinite(value) for value in row[grid].values())
            assert row[grid]["median_loaded_friction_ratio"] == pytest.approx(0.45)
        ordinary = row["ordinary_trial_metrics"]
        for field in ordinary.keys() & row["control_grid"].keys():
            assert ordinary[field] == row["control_grid"][field]
    result = representative(manifest, profile)
    for field, value in result.metrics().items():
        assert records[0]["ordinary_trial_metrics"][field] == pytest.approx(value, rel=0, abs=1e-10)
    assert records[0]["control_grid"] == records[0]["all_physics"]
    trace = result.trace
    window = trace["time"] >= 1.5
    loaded = window & (trace["true_normal_force"] > 0.5)
    normal = np.array([np.cos(np.deg2rad(15)), np.sin(np.deg2rad(15)), 0])
    gap = (np.array([0.4, 0, 0]) - trace["position"]) @ normal - 0.025
    calculated = {
        "mean_normal_force_n": np.mean(trace["true_normal_force"][window]),
        "geometry_separation_pct": 100 * np.mean(gap[window] > 0),
        "maximum_signed_gap_um": 1e6 * gap[window].max(),
        "median_loaded_penetration_um": -1e6 * np.median(gap[loaded]),
    }
    # The historical legacy trace predates true tangent-load telemetry.
    if "true_tangent_force_n" in trace:
        calculated["mean_tangent_force_n"] = np.mean(trace["true_tangent_force_n"][window])
        calculated["median_loaded_friction_ratio"] = np.median(
            trace["true_tangent_force_n"][loaded] / trace["true_normal_force"][loaded]
        )
    for field, value in calculated.items():
        assert records[0]["control_grid"][field] == pytest.approx(value, rel=0, abs=1e-8)
