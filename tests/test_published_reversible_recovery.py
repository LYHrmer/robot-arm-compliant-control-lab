"""Read-only checks of frozen recovery evidence; no new plant trajectories."""

import json
from collections import Counter
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest

from tools.reversible_recovery import study

EXECUTION_COMMIT = "648d3e280263664bd45ee04d388e0e32d5834daa"
PILOT = study.ROOT / "results/franka_reversible_recovery_pilot"
PILOT_SHA256 = "f76cd7bbd6c9ae971872cbbb4d4e55a9eca2d12d5cae0506f443f57590f4f5ba"
TRANSFER = study.ROOT / "results/franka_reversible_recovery_transfer"
TRANSFER_SHA256 = "e1869a3145eb559f96e81e8769263cb61968a7b432009351439b7ecb4890bd07"
SCENARIOS, SEEDS = ("falling", "constant_high"), (11, 29)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _file_state(roots):
    return {path: (path.stat().st_size, path.stat().st_mtime_ns)
            for root in roots for path in root.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def published_pilot():
    assert (PILOT / "COMPLETE").is_file(), "published pilot is required, not skipped"
    roots = (PILOT, study.REFERENCE, study.STATIONARY_REFERENCE,
             study.ROOT / study.transfer.protocol_document()["reference"]["directory"])
    before = _file_state(roots)
    calls = Counter()
    pure_replay = study.replay_compensation

    def replay(trace, parameters, method, dt):
        calls[method] += 1
        return pure_replay(trace, parameters, method, dt)

    # Exactly one full replay per module; the other checks reuse its sealed JSON.
    with (
        patch.object(study.original.runner, "_run_loop", side_effect=AssertionError("unexpected physics")),
        patch.object(study, "ScheduledSurfaceSimulator", side_effect=AssertionError("unexpected plant")),
        patch.object(study.original.previous, "_save_trace", side_effect=AssertionError("unexpected trace write")),
        patch.object(study, "replay_compensation", side_effect=replay),
    ):
        audited = study.audit_archive(PILOT, scope="pilot")
    assert _file_state(roots) == before, "audit changed a published artifact"
    assert pure_replay.__module__ == "tools.reversible_recovery.audit"
    assert calls == Counter({method: 4 for method in (study.ADAPTIVE, study.HOLD, study.STATIONARY, study.METHOD)})
    return {"audit": audited, "manifest": _json(PILOT / "manifest.json"),
            "protocol": _json(PILOT / "protocol.json"), "report": _json(PILOT / "comparison.json")}


def test_pilot_identity_protocol_sources_and_nonpromotion(published_pilot):
    manifest, protocol, report = (published_pilot[key] for key in ("manifest", "protocol", "report"))
    assert manifest["source_commit"] == EXECUTION_COMMIT
    assert study._sha256(PILOT / "manifest.json") == PILOT_SHA256
    assert (PILOT / "COMPLETE").read_text().strip() == PILOT_SHA256
    assert manifest["identity"] == protocol["identity"] == "reversible-release-pilot-v1"
    assert manifest["scope"] == protocol["scope"] == "pilot"
    assert manifest["public_development"] is protocol["public_development"] is True
    assert published_pilot["audit"] == {
        "archive_integrity": "PASS", "comparison_status": "PASS", "scope": "pilot",
        "evaluated_runs": 16, "new_simulations": 4, "reused_runs": 12,
        "default_changed": False, "new_holdout": False,
    }
    for row in (manifest, protocol, report, published_pilot["audit"]):
        assert row["default_changed"] is row["new_holdout"] is False
    assert report["eligible_for_default_change"] is False
    study.original.verify_comparison(protocol, study.protocol_document("pilot"))
    sources = _json(PILOT / "source_hashes.json")
    assert len(sources) == 175
    assert sources == study.source_identity()
    assert _json(study.STATIONARY_REFERENCE / "source_hashes.json").items() <= sources.items()
    assert protocol["reference_manifest_sha256"] == study.REFERENCE_SHA256
    assert protocol["stationary_manifest_sha256"] == study.STATIONARY_SHA256
    assert manifest["artifact_sha256"]["protocol.json"] == study._sha256(PILOT / "protocol.json")


def test_four_new_traces_and_twelve_in_place_references_cover_sixteen_runs(published_pilot):
    report, manifest = (published_pilot[key] for key in ("report", "manifest"))
    runs = report["runs"]
    expected = set(product((-15,), ("clean",), SCENARIOS, SEEDS,
                           (study.ADAPTIVE, study.HOLD, study.STATIONARY, study.METHOD)))
    assert len(runs) == len(expected) == 16
    assert {study.transfer.run_key(row) for row in runs} == expected
    assert Counter(row["origin"] for row in runs) == {"transfer": 8, "stationary": 4, "new": 4}
    specs = {study.transfer.run_key(spec): spec for spec in study.specifications("pilot")}
    for row in runs:
        spec = specs[study.transfer.run_key(row)]
        path = study.trace_location(PILOT, spec)
        assert row["trace_sha256"] == study._sha256(path)
        assert row["parameters"] == published_pilot["protocol"]["controller_parameters"]
        assert row["audit"]["validated_cycles"] == 6000
        assert row["audit"]["fault_provenance_checked"] is True
        if row["origin"] == "new":
            assert row["trace_sha256"] == manifest["artifact_sha256"][row["trace_path"]]
            assert row["audit"]["hold_recovery_cycles"] > 0
        else:
            assert not (PILOT / row["trace_path"]).exists()
            assert path.parent.parent in (study.REFERENCE, study.STATIONARY_REFERENCE)
    assert set(manifest["artifact_sha256"]) == {
        "protocol.json", "source_hashes.json", "comparison.json",
        *(row["trace_path"] for row in runs if row["origin"] == "new"),
    }
    assert sum(row["audit"]["validated_cycles"] for row in runs) == 96_000


def test_all_four_prefixes_cover_every_cycle_field(published_pilot):
    rows = published_pilot["report"]["prefix_checks"]
    fields = study.original.previous.FULL_TRACE_FIELDS - study.original.previous.STATIC_TRACE_FIELDS
    assert len(rows) == 4 and len(fields) == 69
    assert {(row["scenario"], row["seed"]) for row in rows} == set(product(SCENARIOS, SEEDS))
    for row in rows:
        assert row["method"] == study.METHOD and row["kind"] == "before_stationary_change"
        assert row["surface_yaw_deg"] == -15 and row["error_profile"] == "clean"
        assert set(row["fields"]) == fields
        assert row["before_s"] == 5.5 and row["samples"] == 2750
        assert row["bit_exact"] is True and row["different_fields"] == []


def test_pilot_pass_preserves_old_failures_and_positive_high_load_costs(published_pilot):
    report = published_pilot["report"]
    assert report["status"] == "PASS"
    expected_passes = {study.METHOD: 4, study.HOLD: 2, study.STATIONARY: 3}
    for method, count in expected_passes.items():
        rows = [row for row in report["comparisons"] if row["compared_method"] == method]
        assert len(rows) == 4
        assert sum(row["status"] == "PASS" for row in rows) == count
    assert report["high_load_checks"] == [
        {"seed": seed, "position_improved": True, "velocity_improved": True} for seed in SEEDS
    ]
    expected_costs = {11: (0.020050998278365206, 0.09804960529982631),
                      29: (0.03375299325473424, 0.1637234048750531)}
    for row in report["comparisons"]:
        if row["compared_method"] != study.METHOD:
            continue
        assert row["status"] == "PASS" and row["failed_checks"] == []
        if row["scenario"] == "constant_high":
            ramp = next(window for window in row["window_deltas"] if window["name"] == "ramp")
            actual = (ramp["tangent_rmse_mm"], ramp["tangent_velocity_rmse_mm_s"])
            np.testing.assert_allclose(actual, expected_costs[row["seed"]], rtol=0, atol=1e-8)
            assert 0 < actual[0] < 0.1 and 0 < actual[1] < 0.5


def test_candidate_window_errors_are_recomputed_from_physical_trace_coordinates(published_pilot):
    for run in published_pilot["report"]["runs"]:
        if run["method"] != study.METHOD:
            continue
        angle = np.deg2rad(run["surface_yaw_deg"])
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
        with np.load(PILOT / run["trace_path"], allow_pickle=False) as trace:
            for window in run["windows"]:
                mask = (trace["time"] >= window["start_s"]) & (trace["time"] < window["end_s"])
                assert np.any(mask)
                for actual, target, metric in (
                    ("position", "target_position", "tangent_rmse_mm"),
                    ("linear_velocity", "target_linear_velocity", "tangent_velocity_rmse_mm_s"),
                ):
                    error = (trace[actual] - trace[target])[mask]
                    squared_norm = (error @ tangent)**2 + error[:, 2]**2
                    value = 1000 * np.sqrt(np.mean(squared_norm))
                    assert window[metric] == pytest.approx(value, abs=1e-10, rel=0)


@pytest.fixture(scope="module")
def published_transfer():
    assert (TRANSFER / "COMPLETE").is_file(), "published transfer is required, not skipped"
    roots = (TRANSFER, PILOT, study.REFERENCE, study.STATIONARY_REFERENCE,
             study.ROOT / study.transfer.protocol_document()["reference"]["directory"])
    before = _file_state(roots)
    with (
        patch.object(study.original.runner, "_run_loop", side_effect=AssertionError("unexpected physics")),
        patch.object(study, "ScheduledSurfaceSimulator", side_effect=AssertionError("unexpected plant")),
        patch.object(study.original.previous, "_save_trace", side_effect=AssertionError("unexpected trace write")),
    ):
        audited = study.audit_archive(TRANSFER, scope="transfer")
    assert _file_state(roots) == before, "audit changed a published artifact"
    return {"audit": audited, "manifest": _json(TRANSFER / "manifest.json"),
            "protocol": _json(TRANSFER / "protocol.json"),
            "report": _json(TRANSFER / "comparison.json")}


def test_transfer_integrity_pass_is_not_experimental_pass(published_transfer, published_pilot):
    manifest, protocol, report = (published_transfer[key] for key in ("manifest", "protocol", "report"))
    assert study._sha256(TRANSFER / "manifest.json") == TRANSFER_SHA256
    assert (TRANSFER / "COMPLETE").read_text().strip() == TRANSFER_SHA256
    assert manifest["source_commit"] == EXECUTION_COMMIT
    assert manifest["identity"] == protocol["identity"] == "reversible-release-transfer-v1"
    assert manifest["scope"] == protocol["scope"] == "transfer"
    assert manifest["pilot_reference"] == {
        "directory": "results/franka_reversible_recovery_pilot", "manifest_sha256": PILOT_SHA256,
    }
    assert _json(TRANSFER / "source_hashes.json") == _json(PILOT / "source_hashes.json")
    study.original.verify_comparison(protocol, study.protocol_document("transfer"))
    assert published_transfer["audit"] == {
        "archive_integrity": "PASS", "comparison_status": "FAIL", "scope": "transfer",
        "evaluated_runs": 108, "new_simulations": 32, "reused_runs": 76,
        "default_changed": False, "new_holdout": False,
    }
    assert report["status"] == "FAIL"
    assert report["high_load_checks"] == published_pilot["report"]["high_load_checks"]
    for row in (manifest, protocol, report, published_transfer["audit"]):
        assert row["default_changed"] is row["new_holdout"] is False
    assert report["eligible_for_default_change"] is False


def test_transfer_reuses_pilot_and_nested_yaw_zero_traces_without_copying(published_transfer):
    report, manifest = (published_transfer[key] for key in ("report", "manifest"))
    runs = report["runs"]
    expected = set(product((-15, 0, 15), ("clean", "combined", "combined_scale_0p8"),
                           SCENARIOS, SEEDS, (study.ADAPTIVE, study.HOLD, study.METHOD)))
    assert len(runs) == len(expected) == 108
    assert {study.transfer.run_key(row) for row in runs} == expected
    assert Counter(row["origin"] for row in runs) == {"transfer": 72, "pilot": 4, "new": 32}
    specs = {study.transfer.run_key(spec): spec for spec in study.specifications("transfer")}
    nested = study.ROOT / study.transfer.protocol_document()["reference"]["directory"]
    yaw_zero_old = 0
    for row in runs:
        path = study.trace_location(TRANSFER, specs[study.transfer.run_key(row)], PILOT)
        assert row["trace_sha256"] == study._sha256(path)
        assert row["parameters"] == published_transfer["protocol"]["controller_parameters"]
        assert row["audit"]["validated_cycles"] == 6000
        if row["origin"] == "new":
            assert row["trace_sha256"] == manifest["artifact_sha256"][row["trace_path"]]
        else:
            assert not (TRANSFER / row["trace_path"]).exists()
            if row["origin"] == "pilot":
                assert path == PILOT / row["trace_path"]
            elif (row["surface_yaw_deg"], row["error_profile"]) == (0, "clean"):
                assert path == nested / row["trace_path"]
                yaw_zero_old += 1
    assert yaw_zero_old == 8
    assert sum(row["audit"]["validated_cycles"] for row in runs) == 648_000
    assert set(manifest["artifact_sha256"]) == {
        "protocol.json", "source_hashes.json", "comparison.json",
        *(row["trace_path"] for row in runs if row["origin"] == "new"),
    }


def test_transfer_has_thirty_six_candidate_and_twelve_fault_prefix_checks(published_transfer):
    rows = published_transfer["report"]["prefix_checks"]
    fields = study.original.previous.FULL_TRACE_FIELDS - study.original.previous.STATIC_TRACE_FIELDS
    assert len(rows) == 48
    expected = {
        (*key, "before_stationary_change")
        for key in product((-15, 0, 15), ("clean", "combined", "combined_scale_0p8"),
                           SCENARIOS, SEEDS, (study.METHOD,))
    } | {
        (*key, "before_auxiliary_scale_fault")
        for key in product((-15, 0, 15), ("combined_scale_0p8",), SCENARIOS, SEEDS, (study.METHOD,))
    }
    assert {(*study.transfer.run_key(row), row["kind"]) for row in rows} == expected
    for row in rows:
        assert set(row["fields"]) == fields and len(fields) == 69
        assert row["bit_exact"] is True and row["different_fields"] == []
        before = 5.5 if row["kind"] == "before_stationary_change" else 6.0
        assert row["before_s"] == before and row["samples"] == round(before / 0.002)


def test_transfer_preserves_four_falling_velocity_failures_despite_pilot_success(published_transfer):
    comparisons = published_transfer["report"]["comparisons"]
    primary = [row for row in comparisons if row["compared_method"] == study.METHOD]
    old = [row for row in comparisons if row["compared_method"] == study.HOLD]
    assert len(primary) == len(old) == 36
    assert sum(row["status"] == "PASS" for row in primary) == 32
    assert sum(row["status"] == "PASS" for row in old) == 34
    failed = [row for row in primary if row["status"] == "FAIL"]
    assert {study.case_identity(row) for row in failed} == set(product(
        (15,), ("combined", "combined_scale_0p8"), ("falling",), SEEDS,
    ))
    expected = {("combined", 11): 0.6353323165282205, ("combined", 29): 0.6433271895361159,
                ("combined_scale_0p8", 11): 0.6354589929684575,
                ("combined_scale_0p8", 29): 0.6431797672440069}
    specs = {study.transfer.run_key(spec): spec for spec in study.specifications("transfer")}
    for row in failed:
        assert row["failed_checks"] == ["ramp:velocity_cost"]
        ramp = next(window for window in row["window_deltas"] if window["name"] == "ramp")
        assert ramp["tangent_velocity_rmse_mm_s"] == pytest.approx(
            expected[row["error_profile"], row["seed"]], abs=1e-8, rel=0,
        )
        assert ramp["tangent_velocity_rmse_mm_s"] > 0.5
        # Derive the failing extra cost from world-frame velocity arrays, not JSON
        # deltas or the runner's metric helper. The physical tangent is at +15 deg.
        angle = np.deg2rad(15.0)
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
        values = []
        for method in (study.ADAPTIVE, study.METHOD):
            spec = specs[(*study.case_identity(row), method)]
            with np.load(study.trace_location(TRANSFER, spec, PILOT), allow_pickle=False) as trace:
                mask = (trace["time"] >= 7.0) & (trace["time"] < 8.0)
                error = (trace["linear_velocity"] - trace["target_linear_velocity"])[mask]
                values.append(1000 * np.sqrt(np.mean((error @ tangent)**2 + error[:, 2]**2)))
        assert values[1] - values[0] == pytest.approx(ramp["tangent_velocity_rmse_mm_s"], abs=1e-10, rel=0)
