"""Read-only checks of the published transfer matrix; never rerun the plant."""

import json
from collections import Counter
from itertools import product
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.reversal_recovery_transfer import study

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/franka_reversal_recovery_transfer"
REFERENCE = ROOT / "results/franka_reversal_recovery"
EXECUTION_COMMIT = "03cd2a53d88871e9805712a123b7ce45be422ad4"
MANIFEST_SHA256 = "eef733d9b228267f2987a0e4c743020c73a27541010dbeedd8f571a94681c44a"
REFERENCE_MANIFEST_SHA256 = "e8310dca906626267a8d8884aea4a3c6f619e66cc79a8627acbff220604746e5"
YAWS = (-15, 0, 15)
PROFILES = ("clean", "combined", "combined_scale_0p8")
SCENARIOS = ("falling", "constant_high")
SEEDS = (11, 29)
METHODS = ("adaptive6_8", "hold_cap_tracking")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_state():
    return {path: (path.stat().st_size, path.stat().st_mtime_ns)
            for root in (ARCHIVE, REFERENCE) for path in root.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def published():
    assert (ARCHIVE / "COMPLETE").is_file(), "published archive is required, not skipped"
    before = file_state()
    # One full replay for this module: it checks hashes, all 72 summaries and all
    # 60 byte prefixes. The remaining tests reuse JSON, not another full replay.
    with patch.object(study.original.runner, "_run_loop", side_effect=AssertionError("unexpected physics")):
        audited = study.audit_archive(ARCHIVE)
    assert file_state() == before, "read-only audit changed a published artifact"
    return {"audit": audited, "manifest": read_json(ARCHIVE / "manifest.json"),
            "protocol": read_json(ARCHIVE / "protocol.json"),
            "report": read_json(ARCHIVE / "comparison.json"),
            "reference_manifest": read_json(REFERENCE / "manifest.json")}


def test_published_execution_identity_hashes_and_no_promotion(published):
    manifest, protocol, report, audited = (published[key] for key in ("manifest", "protocol", "report", "audit"))
    assert manifest["identity"] == protocol["identity"] == "reversal-recovery-cross-direction-errors-v1"
    assert manifest["source_commit"] == EXECUTION_COMMIT
    assert study._sha256(ARCHIVE / "manifest.json") == MANIFEST_SHA256
    assert (ARCHIVE / "COMPLETE").read_text().strip() == MANIFEST_SHA256
    assert manifest["source_commit_role"].startswith("execution-start HEAD")
    assert audited["archive_integrity"] == "PASS"
    assert audited["comparison_status"] == report["status"]
    assert manifest["new_simulations"] == audited["new_simulations"] == 64
    assert manifest["evaluated_runs"] == audited["evaluated_runs"] == 72
    assert audited["reused_runs"] == 8
    assert manifest["public_development"] is protocol["public_development"] is True
    for row in (manifest, protocol, report, audited):
        assert row["default_changed"] is False
        assert row["new_holdout"] is False
    assert report["eligible_for_default_change"] is False
    sources = read_json(ARCHIVE / "source_hashes.json")
    assert sources == study.source_identity()
    assert sources["tools/reversal_recovery_transfer/protocol.json"] == study._sha256(study.PROTOCOL)
    assert manifest["artifact_sha256"]["protocol.json"] == study._sha256(ARCHIVE / "protocol.json")
    study.original.verify_comparison(protocol, study.protocol_document())


def test_matrix_has_72_unique_runs_and_36_unique_matched_pairs(published):
    runs, pairs = (published["report"][key] for key in ("runs", "comparisons"))
    expected = set(product(YAWS, PROFILES, SCENARIOS, SEEDS, METHODS))
    assert len(runs) == len(expected) == 72
    assert {study.run_key(row) for row in runs} == expected
    assert len(pairs) == 36
    assert {tuple(row[key] for key in study.RUN_KEYS[:-1]) for row in pairs} == {
        key[:-1] for key in expected
    }
    assert sum(row["audit"]["validated_cycles"] for row in runs) == 432_000
    for row in runs:
        case, parameters = row["case"], row["parameters"]
        assert case["scenario"]["wall_yaw_deg"] == case["task"]["yaw_deg"] == row["surface_yaw_deg"]
        assert case["config"]["seed"] == row["seed"]
        assert case["config"]["timestep"] == 0.002 and case["config"]["duration"] == 12.0
        assert case["controller_yaw_error_deg"] == (0 if row["error_profile"] == "clean" else 3)
        assert case["task"]["trajectory"] == "stop_hold_reverse"
        assert parameters["minimum_force"] == 6.0 and parameters["max_force"] == 8.0
        assert parameters["velocity_error_time"] == 0.05
        assert row["audit"]["trace_schema_version"] == 3
        assert row["audit"]["fault_provenance_checked"] is True
        assert row["fault"]["name"] == ("scale_0p8" if row["error_profile"] == "combined_scale_0p8" else "fresh")


def test_eight_original_traces_are_pinned_references_not_copied_or_reexecuted(published):
    protocol, report, manifest = (published[key] for key in ("protocol", "report", "manifest"))
    assert protocol["reference"]["directory"] == "results/franka_reversal_recovery"
    assert protocol["reference"]["manifest_sha256"] == REFERENCE_MANIFEST_SHA256
    assert study._sha256(REFERENCE / "manifest.json") == REFERENCE_MANIFEST_SHA256
    assert (REFERENCE / "COMPLETE").read_text().strip() == REFERENCE_MANIFEST_SHA256
    reused = [row for row in report["runs"] if row["origin"] == "reference"]
    fresh = [row for row in report["runs"] if row["origin"] == "new"]
    assert len(reused) == 8 and len(fresh) == 64
    assert {(row["surface_yaw_deg"], row["error_profile"]) for row in reused} == {(0, "clean")}
    original_rows = {
        (row["scenario"], row["seed"], row["method"]): row
        for row in read_json(REFERENCE / "comparison.json")["runs"]
    }
    for row in reused:
        assert row["trace_path"] == original_rows[row["scenario"], row["seed"], row["method"]]["trace_path"]
        assert row["trace_sha256"] == published["reference_manifest"]["artifact_sha256"][row["trace_path"]]
        assert not (ARCHIVE / row["trace_path"]).exists()
    for row in fresh:
        assert row["trace_sha256"] == manifest["artifact_sha256"][row["trace_path"]]
    assert set(manifest["artifact_sha256"]) == {
        "protocol.json", "source_hashes.json", "comparison.json", *(row["trace_path"] for row in fresh),
    }


def test_all_sixty_prefix_checks_cover_every_recorded_cycle_field(published):
    prefixes = published["report"]["prefix_checks"]
    expected = {
        (*key, "before_candidate_divergence")
        for key in product(YAWS, PROFILES, SCENARIOS, SEEDS, ("hold_cap_tracking",))
    } | {
        (*key, "before_auxiliary_scale_fault")
        for key in product(YAWS, ("combined_scale_0p8",), SCENARIOS, SEEDS, METHODS)
    }
    assert len(prefixes) == len(expected) == 60
    assert {(*study.run_key(row), row["kind"]) for row in prefixes} == expected
    fields = study.original.previous.FULL_TRACE_FIELDS - study.original.previous.STATIC_TRACE_FIELDS
    assert len(fields) == 69
    for row in prefixes:
        assert set(row["fields"]) == fields and len(row["fields"]) == 69
        assert set(row["different_fields"]) <= fields
        assert row["bit_exact"] is (not row["different_fields"])
        assert row["bit_exact"] is True
        before = 4.5 if row["kind"] == "before_candidate_divergence" else 6.0
        assert row["before_s"] == before
        assert row["samples"] == round(before / 0.002)


def test_rebuilt_summary_keeps_benefit_cost_and_absolute_failures_separate(published):
    report, protocol = published["report"], published["protocol"]
    runs = {study.run_key(row): row for row in report["runs"]}
    for pair in report["comparisons"]:
        categories = [pair[name] for name in (
            "absolute_failures", "missed_benefit_checks", "paired_cost_failures",
        )]
        assert Counter(pair["failed_checks"]) == Counter(item for group in categories for item in group)
        assert pair["status"] == ("FAIL" if pair["failed_checks"] else "PASS")
        identity = tuple(pair[key] for key in study.RUN_KEYS[:-1])
        baseline, candidate = (runs[(*identity, method)] for method in METHODS)
        absolute, benefit, costs = [], [], []
        for run in (baseline, candidate):
            for row in (run["overall"], *run["phases"]):
                for metric, bound in protocol["absolute_gates"].items():
                    if metric not in row:
                        assert metric in {"projection_pct", "minimum_reserved_torque_headroom_nm"}
                        continue
                    passed = (row[metric] >= bound if metric in {"contact_ratio_pct", "minimum_reserved_torque_headroom_nm"}
                              else row[metric] <= bound)
                    if not passed:
                        absolute.append(f"{run['method']}:{row.get('phase', 'overall')}:{metric}")
        limits = protocol["acceptance"]
        for left, right in zip(candidate["windows"], baseline["windows"], strict=True):
            name = left["name"]
            if pair["scenario"] == "falling" and name in {"early", "post"}:
                reduction = (100 * (1 - left["tangent_rmse_mm"] / right["tangent_rmse_mm"])
                             if right["tangent_rmse_mm"] > 0 else None)
                if reduction is None or reduction < limits[f"minimum_falling_{name}_tangent_reduction_pct_each_seed"]:
                    benefit.append(f"{name}:position_reduction")
            elif left["tangent_rmse_mm"] - right["tangent_rmse_mm"] > limits["maximum_other_window_tangent_increase_mm"]:
                costs.append(f"{name}:position_cost")
            if left["tangent_velocity_rmse_mm_s"] - right["tangent_velocity_rmse_mm_s"] > limits["maximum_window_tangent_velocity_rmse_increase_mm_s"]:
                costs.append(f"{name}:velocity_cost")
        for left, right in zip(candidate["phases"], baseline["phases"], strict=True):
            for metric, limit in (("force_rmse_n", "maximum_phase_force_rmse_increase_n"),
                                  ("orientation_rmse_deg", "maximum_phase_orientation_rmse_increase_deg")):
                if left[metric] - right[metric] > limits[limit]:
                    costs.append(f"{left['phase']}:{metric}")
        if candidate["overall"]["peak_force_n"] - baseline["overall"]["peak_force_n"] > limits["maximum_full_raw_peak_increase_n"]:
            costs.append("full_peak_cost")
        assert pair["absolute_failures"] == absolute
        assert pair["missed_benefit_checks"] == benefit
        assert pair["paired_cost_failures"] == costs
    passed = all(row["status"] == "PASS" for row in report["comparisons"])
    passed = passed and all(row["bit_exact"] for row in report["prefix_checks"])
    assert report["status"] == ("PASS" if passed else "FAIL")


def test_published_failures_are_two_high_load_costs_not_absolute_or_benefit_failures(published):
    report = published["report"]
    pairs = report["comparisons"]
    assert report["status"] == "FAIL"
    assert sum(row["status"] == "PASS" for row in pairs) == 34
    assert sum(row["status"] == "PASS" and row["scenario"] == "falling" for row in pairs) == 18
    assert sum(row["status"] == "PASS" and row["scenario"] == "constant_high" for row in pairs) == 16
    failed = [row for row in pairs if row["status"] == "FAIL"]
    assert {(row["surface_yaw_deg"], row["error_profile"], row["scenario"], row["seed"])
            for row in failed} == {(-15, "clean", "constant_high", seed) for seed in SEEDS}
    assert all(not row["absolute_failures"] and not row["missed_benefit_checks"] for row in pairs)
    assert sum(bool(row["paired_cost_failures"]) for row in pairs) == 2
    for row in failed:
        assert row["paired_cost_failures"] == ["ramp:position_cost", "ramp:velocity_cost"]


def test_guide_tables_and_tradeoffs_match_all_real_pairs(published):
    report = published["report"]
    guide = (ROOT / "docs/reversal_recovery_transfer.md").read_text(encoding="utf-8")
    pairs = report["comparisons"]
    runs = {study.run_key(row): row for row in report["runs"]}
    for scenario in (None, *SCENARIOS):
        selected = [row for row in pairs if scenario is None or row["scenario"] == scenario]
        assert f"{sum(row['status'] == 'PASS' for row in selected)}/{len(selected)}" in guide
    tables = [[cell.strip() for cell in line.strip().strip("|").split("|")]
              for line in guide.splitlines() if line.strip().startswith("|")]
    yaw_labels = {"−15°": -15, "0°": 0, "+15°": 15}
    matrix_rows = [row for row in tables if row[0] in yaw_labels]
    assert len(matrix_rows) == 9
    seen = set()
    for cells in matrix_rows:
        yaw = yaw_labels[cells[0]]
        if cells[1].startswith("无额外误差"):
            profile = "clean"
        elif cells[1] == "组合误差":
            profile = "combined"
        else:
            assert "组合误差" in cells[1] and "0.8" in cells[1]
            profile = "combined_scale_0p8"
        assert (yaw, profile) not in seen
        seen.add((yaw, profile))
        for index, scenario in enumerate(SCENARIOS, start=2):
            selected = [row for row in pairs if (row["surface_yaw_deg"], row["error_profile"], row["scenario"])
                        == (yaw, profile, scenario)]
            assert len(selected) == 2
            assert cells[index] == f"{sum(row['status'] == 'PASS' for row in selected)}/2"
    assert seen == set(product(YAWS, PROFILES))
    failure_rows = {int(row[0]): row for row in tables if row[0] in {str(seed) for seed in SEEDS}}
    assert set(failure_rows) == set(SEEDS)
    for pair in (row for row in pairs if row["status"] == "FAIL"):
        identity = tuple(pair[key] for key in study.RUN_KEYS[:-1])
        baseline, candidate = (runs[(*identity, method)] for method in METHODS)
        left, right = (next(row for row in run["windows"] if row["name"] == "ramp")
                       for run in (baseline, candidate))
        delta = next(row for row in pair["window_deltas"] if row["name"] == "ramp")
        cells = failure_rows[pair["seed"]]
        for metric, column in (("tangent_rmse_mm", 1), ("tangent_velocity_rmse_mm_s", 3)):
            assert cells[column] == f"{left[metric]:.3f} → {right[metric]:.3f}"
            assert cells[column + 1] == f"{delta[metric]:+.6f}"
    falling = [row for row in pairs if row["scenario"] == "falling"]
    for window, metric, digits in (("early", "tangent_reduction_pct", 2),
                                   ("post", "tangent_reduction_pct", 2),
                                   ("late", "tangent_velocity_rmse_mm_s", 3)):
        values = [next(row for row in pair["window_deltas"] if row["name"] == window)[metric]
                  for pair in falling]
        assert min(values) > 0
        assert f"{min(values):.{digits}f}" in guide and f"{max(values):.{digits}f}" in guide
    assert f"{max(row['overall']['peak_force_n'] for row in runs.values()):.3f}" in guide
    assert f"{min(row['overall']['minimum_reserved_torque_headroom_nm'] for row in runs.values()):.3f}" in guide
    for metric in ("force_rmse_n", "orientation_rmse_deg"):
        deltas = []
        for pair in pairs:
            identity = tuple(pair[key] for key in study.RUN_KEYS[:-1])
            baseline, candidate = (runs[(*identity, method)] for method in METHODS)
            deltas.extend(right[metric] - left[metric]
                          for left, right in zip(baseline["phases"], candidate["phases"], strict=True))
        assert f"{max(deltas):.6f}" in guide
    assert EXECUTION_COMMIT in guide
