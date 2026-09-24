"""Frozen four-case pilot identities and gates, without running new physics."""

import copy
import json
import sys

import pytest

from tools.stationary_recovery import study
from tools.stationary_recovery.controller import StationaryHoldCapTracking


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("study unit tests must not start physics")

    monkeypatch.setattr(study.original.runner, "_run_loop", forbidden)
    monkeypatch.setattr(study, "ScheduledSurfaceSimulator", forbidden)
    monkeypatch.setattr(study.original.previous, "_save_trace", forbidden)


def test_exact_twelve_identities_reuse_eight_controls_and_add_four_candidates():
    specs = study.specifications()
    keys = {(s["scenario"], s["seed"], s["method"]) for s in specs}
    assert len(specs) == len(keys) == 12
    assert keys == {(scenario, seed, method)
                    for scenario in ("falling", "constant_high") for seed in (11, 29)
                    for method in ("adaptive6_8", "hold_cap_tracking", study.METHOD)}
    references = [s for s in specs if s["origin"] == "reference"]
    candidates = [s for s in specs if s["origin"] == "new"]
    assert len(references) == 8 and len(candidates) == 4
    assert len({s["trace_path"] for s in specs}) == 12
    old_specs = study.transfer.specifications(study.transfer.protocol_document())
    expected_references = [dict(s, origin="reference") for s in old_specs
                           if s["surface_yaw_deg"] == -15 and s["error_profile"] == "clean"]
    assert references == expected_references
    for spec in specs:
        assert (spec["surface_yaw_deg"], spec["error_profile"], spec["fault"].name) == (
            -15, "clean", "fresh",
        )
        assert spec["case"].task.yaw_deg == spec["case"].scenario.wall_yaw_deg == -15
        assert spec["case"].controller_yaw_error_deg == 0
        assert spec["case"].config.seed == spec["seed"]
        assert spec["case"].config.duration == 12 and spec["case"].config.timestep == 0.002
    for candidate in candidates:
        baseline = next(s for s in references if s["scenario"] == candidate["scenario"]
                        and s["seed"] == candidate["seed"] and s["method"] == "adaptive6_8")
        assert candidate["method"] == study.METHOD
        assert candidate["case"] == baseline["case"]
        assert candidate["trace_path"] == baseline["trace_path"].replace(
            "__adaptive6_8", f"__{study.METHOD}",
        )


def test_candidate_class_keeps_original_controller_parameters():
    for spec in study.specifications():
        controller, parameters = study.make_controller(spec)
        _, expected = study.original.make_controller(
            {**spec, "method": "adaptive6_8"}, study.original.protocol_document(),
        )
        assert parameters == expected
        assert parameters["minimum_force"] == 6 and parameters["max_force"] == 8
        assert parameters["velocity_error_time"] == 0.05
        if spec["origin"] == "new":
            assert type(controller._base.tangential) is StationaryHoldCapTracking


def test_source_identity_keeps_all_164_pinned_reference_hashes():
    pinned = json.loads((study.REFERENCE / "source_hashes.json").read_text())
    assert len(pinned) == 164
    assert study.transfer.source_identity() == pinned
    sources = study.source_identity()
    assert pinned.items() <= sources.items()
    assert {"tools/stationary_recovery/study.py", "tools/stationary_recovery/controller.py",
            "tools/stationary_recovery/audit.py"} <= sources.keys()


def test_frozen_protocol_preserves_original_gates_and_declares_only_four_new_runs():
    protocol = study.protocol_document()
    original = study.original.protocol_document()
    for key in ("acceptance", "absolute_gates", "windows"):
        assert protocol[key] == original[key]
    assert protocol["comparison_replay"] == study.transfer.protocol_document()["comparison_replay"]
    assert (protocol["maximum_new_simulations"], protocol["reused_runs"],
            protocol["evaluated_runs"], protocol["physical_scenarios"]) == (4, 8, 12, 2)
    assert protocol["comparison_defined_before_execution"] is True
    assert protocol["public_development"] is True
    assert protocol["default_changed"] is protocol["new_holdout"] is False
    assert protocol["release_speed_max_m_s"] == 1e-12
    assert protocol["prefix_before_s"] == 5.5
    assert protocol["reference_manifest_sha256"] == study.REFERENCE_SHA256
    assert study.ROOT / protocol["reference_directory"] == study.REFERENCE
    assert {(s["scenario"], s["seed"]) for s in protocol["cases"]} == {
        (scenario, seed) for scenario in ("falling", "constant_high") for seed in (11, 29)
    }
    # A caller cannot mutate future protocol instances through shared gate dictionaries.
    protocol["acceptance"]["maximum_other_window_tangent_increase_mm"] = 99
    assert study.protocol_document()["acceptance"] == original["acceptance"]


@pytest.fixture
def passing_rows():
    rows = []
    protocol = study.original.protocol_document()
    for spec in study.specifications():
        metrics = {"force_rmse_n": 1.0, "orientation_rmse_deg": 0.1,
                   "contact_ratio_pct": 100.0, "peak_force_n": 20.0, "saturation_pct": 0.0}
        windows = []
        for window in protocol["windows"]:
            position, velocity = 2.0, 10.0
            if (spec["scenario"] == "falling" and spec["method"] != "adaptive6_8"
                    and window["name"] in {"early", "post"}):
                position = 1.7
            if spec["scenario"] == "constant_high" and window["name"] == "ramp":
                if spec["method"] == "hold_cap_tracking":
                    position, velocity = 2.2, 10.8
                elif spec["method"] == study.METHOD:
                    position, velocity = 2.05, 10.25
            windows.append({**window, "tangent_rmse_mm": position,
                            "tangent_velocity_rmse_mm_s": velocity})
        rows.append({key: spec[key] for key in study.transfer.RUN_KEYS} | {
            "overall": {**metrics, "projection_pct": 0.0,
                        "minimum_reserved_torque_headroom_nm": 2.0},
            "phases": [{"phase": phase.name, **metrics} for phase in spec["case"].phases],
            "windows": windows,
        })
    return rows


def test_primary_can_pass_while_prior_reference_failures_are_retained(passing_rows):
    before = copy.deepcopy(passing_rows)
    comparisons, high_checks, passed = study.compare_runs(passing_rows)
    assert passed is True
    assert len(comparisons) == 8
    primary = [row for row in comparisons if row["compared_method"] == study.METHOD]
    prior = [row for row in comparisons if row["compared_method"] == "hold_cap_tracking"]
    assert len(primary) == len(prior) == 4
    assert all(row["status"] == "PASS" for row in primary)
    failed = [row for row in prior if row["status"] == "FAIL"]
    assert {(row["scenario"], row["seed"]) for row in failed} == {
        ("constant_high", 11), ("constant_high", 29),
    }
    assert all(row["failed_checks"] == ["ramp:position_cost", "ramp:velocity_cost"] for row in failed)
    assert high_checks == [{"seed": seed, "position_improved": True, "velocity_improved": True}
                           for seed in (11, 29)]
    assert passing_rows == before


@pytest.mark.parametrize("scenario", ["falling", "constant_high"])
@pytest.mark.parametrize("seed", [11, 29])
def test_each_of_four_primary_pairs_must_pass_without_averaging(passing_rows, scenario, seed):
    candidate = next(row for row in passing_rows if (row["scenario"], row["seed"], row["method"])
                     == (scenario, seed, study.METHOD))
    candidate["overall"]["projection_pct"] = 0.1
    comparisons, high_checks, passed = study.compare_runs(passing_rows)
    assert passed is False
    primary = [row for row in comparisons if row["compared_method"] == study.METHOD]
    assert sum(row["status"] == "PASS" for row in primary) == 3
    failed = next(row for row in primary if row["status"] == "FAIL")
    assert (failed["scenario"], failed["seed"]) == (scenario, seed)
    assert failed["failed_checks"] == [f"{study.METHOD}:overall:projection_pct"]
    assert all(row["position_improved"] and row["velocity_improved"] for row in high_checks)


@pytest.mark.parametrize("seed", [11, 29])
@pytest.mark.parametrize("metric,flag", [
    ("tangent_rmse_mm", "position_improved"),
    ("tangent_velocity_rmse_mm_s", "velocity_improved"),
])
@pytest.mark.parametrize("prior_offset", [0.0, -0.01], ids=["equal", "prior_better"])
def test_each_high_load_seed_requires_both_strict_improvements(
    passing_rows, seed, metric, flag, prior_offset,
):
    ramps = {row["method"]: next(w for w in row["windows"] if w["name"] == "ramp")
             for row in passing_rows if row["scenario"] == "constant_high" and row["seed"] == seed}
    # Alter only the old reference: all four primary comparisons still pass.
    ramps["hold_cap_tracking"][metric] = ramps[study.METHOD][metric] + prior_offset
    comparisons, high_checks, passed = study.compare_runs(passing_rows)
    assert all(row["status"] == "PASS" for row in comparisons if row["compared_method"] == study.METHOD)
    assert passed is False
    for row in high_checks:
        for name in ("position_improved", "velocity_improved"):
            assert row[name] is not (row["seed"] == seed and name == flag)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unknown_seed", "unknown_method"])
def test_missing_duplicate_or_misidentified_run_rejected(passing_rows, corruption):
    if corruption == "missing":
        passing_rows.pop()
    elif corruption == "duplicate":
        passing_rows[-1] = copy.deepcopy(passing_rows[0])
    elif corruption == "unknown_seed":
        passing_rows[0]["seed"] = 999
    else:
        passing_rows[0]["method"] = "unregistered"
    with pytest.raises(ValueError, match="missing/duplicate/misidentified paired run"):
        study.compare_runs(passing_rows)


@pytest.mark.parametrize("arguments", [[], ["--output"], ["--audit"],
                                       ["--output", "new", "--audit", "old"]])
def test_cli_requires_one_complete_operation_before_any_work(arguments, monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid CLI arguments must not execute or audit")

    monkeypatch.setattr(study, "run", forbidden)
    monkeypatch.setattr(study, "audit_archive", forbidden)
    monkeypatch.setattr(sys, "argv", ["stationary_recovery.study", *arguments])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as caught:
        study.main()
    assert caught.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", ["file", "directory"])
def test_existing_output_rejected_before_git_or_collection(existing, monkeypatch, tmp_path):
    destination = tmp_path / "preserve"
    if existing == "directory":
        destination.mkdir()
        sentinel = destination / "sentinel"
    else:
        sentinel = destination
    sentinel.write_text("keep me")

    def forbidden(*_args, **_kwargs):
        pytest.fail("existing output must be rejected before git or collection")

    monkeypatch.setattr(study.subprocess, "check_output", forbidden)
    monkeypatch.setattr(study, "collect", forbidden)
    with pytest.raises(ValueError, match="output must be new"):
        study.run(destination)
    assert sentinel.read_text() == "keep me"
    assert list(tmp_path.iterdir()) == [destination]


def _seal_metadata_archive(directory, manifest):
    manifest["artifact_sha256"] = {
        name: study._sha256(directory / name) for name in manifest["artifact_sha256"]
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "COMPLETE").write_text(study._sha256(directory / "manifest.json") + "\n")


@pytest.fixture
def metadata_archive(tmp_path, monkeypatch):
    """Tiny, fully sealed artifacts that must be rejected before trace replay."""
    artifacts = {
        "protocol.json": json.dumps(study.protocol_document()),
        "source_hashes.json": json.dumps(study.source_identity()),
        "comparison.json": "{}",
        **{spec["trace_path"]: "metadata-only fixture; not a simulated trace"
           for spec in study.specifications() if spec["origin"] == "new"},
    }
    for name, content in artifacts.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    manifest = {
        "identity": study.protocol_document()["identity"],
        "source_commit": "a" * 40, "new_simulations": 4, "reused_runs": 8,
        "default_changed": False, "new_holdout": False,
        "artifact_sha256": dict.fromkeys(artifacts),
    }
    _seal_metadata_archive(tmp_path, manifest)

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid archive metadata reached trace replay")

    monkeypatch.setattr(study, "collect", forbidden)
    return tmp_path, manifest


@pytest.mark.parametrize("field,value", [
    ("identity", "different-pilot"), ("new_simulations", 5), ("reused_runs", 7),
    ("default_changed", True), ("default_changed", 0),
    ("new_holdout", True), ("new_holdout", 0),
])
def test_resealed_scope_changes_rejected_before_replay(metadata_archive, field, value):
    directory, manifest = metadata_archive
    manifest[field] = value
    _seal_metadata_archive(directory, manifest)
    with pytest.raises(ValueError, match="identity/scope/inventory"):
        study.audit_archive(directory)


@pytest.mark.parametrize("commit", [
    None, 123, True, ["a" * 40], "a" * 39, "a" * 41, "g" * 40, "A" * 40,
])
def test_resealed_invalid_execution_commit_rejected_before_replay(metadata_archive, commit):
    directory, manifest = metadata_archive
    manifest["source_commit"] = commit
    _seal_metadata_archive(directory, manifest)
    with pytest.raises(ValueError, match="invalid execution commit"):
        study.audit_archive(directory)


@pytest.mark.parametrize("source", [
    "tools/reversal_recovery/controller.py", "tools/stationary_recovery/controller.py",
    "tools/stationary_recovery/audit.py", "tools/stationary_recovery/study.py",
])
def test_resealed_expected_source_hash_change_rejected_before_replay(metadata_archive, source):
    directory, manifest = metadata_archive
    source_file = directory / "source_hashes.json"
    sources = json.loads(source_file.read_text())
    assert source in sources
    sources[source] = "0" * 64
    source_file.write_text(json.dumps(sources))
    _seal_metadata_archive(directory, manifest)
    with pytest.raises(ValueError, match="source identity differs"):
        study.audit_archive(directory)


def test_resealed_protocol_cannot_replace_the_pinned_reference_manifest(metadata_archive):
    directory, manifest = metadata_archive
    protocol_file = directory / "protocol.json"
    protocol = json.loads(protocol_file.read_text())
    protocol["reference_manifest_sha256"] = "0" * 64
    protocol_file.write_text(json.dumps(protocol))
    _seal_metadata_archive(directory, manifest)
    with pytest.raises(ValueError, match="differs"):
        study.audit_archive(directory)


def test_collection_rejects_a_changed_reference_manifest_before_loading_traces(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "COMPLETE").write_text(study._sha256(tmp_path / "manifest.json") + "\n")
    monkeypatch.setattr(study, "REFERENCE", tmp_path)
    monkeypatch.setattr(study.original.previous, "_load_trace",
                        lambda *_: pytest.fail("invalid reference reached trace loading"))
    with pytest.raises(ValueError, match="pinned reference manifest differs"):
        study.collect(tmp_path / "unused", execute=False)


def test_published_archive_audits_without_physics_and_preserves_three_of_four_failure():
    directory = study.ROOT / "results/franka_stationary_recovery_pilot"
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns)
              for path in directory.rglob("*") if path.is_file()}
    manifest = study.verify_archive(
        directory, "ae864f1a86658ecc308aa0e671cecf9f1dd0e3bf13dba0ebd1e46c6ed0c5a88f",
    )
    assert manifest["source_commit"] == "6eff326cea8765dee5ca422dd59e259c5a8e9b77"
    checked = study.audit_archive(directory)
    assert checked == {
        "archive_integrity": "PASS", "comparison_status": "FAIL", "evaluated_runs": 12,
        "new_simulations": 4, "reused_runs": 8, "default_changed": False, "new_holdout": False,
    }
    report = json.loads((directory / "comparison.json").read_text())
    assert report["status"] == "FAIL"
    assert report["default_changed"] is report["eligible_for_default_change"] is False
    assert len(report["runs"]) == 12
    primary = [row for row in report["comparisons"] if row["compared_method"] == study.METHOD]
    assert len(primary) == 4 and sum(row["status"] == "PASS" for row in primary) == 3
    failed = [row for row in primary if row["status"] == "FAIL"]
    assert len(failed) == 1
    assert (failed[0]["scenario"], failed[0]["seed"]) == ("constant_high", 29)
    assert failed[0]["failed_checks"] == ["ramp:position_cost", "ramp:velocity_cost"]
    ramp = next(row for row in failed[0]["window_deltas"] if row["name"] == "ramp")
    assert ramp["tangent_rmse_mm"] == pytest.approx(0.10695571007888849, rel=0, abs=1e-10)
    assert ramp["tangent_velocity_rmse_mm_s"] == pytest.approx(0.5457319472178757, rel=0, abs=1e-10)
    limits = study.original.protocol_document()["acceptance"]
    assert ramp["tangent_rmse_mm"] > limits["maximum_other_window_tangent_increase_mm"]
    assert ramp["tangent_velocity_rmse_mm_s"] > limits["maximum_window_tangent_velocity_rmse_increase_mm_s"]
    assert report["high_load_checks"] == [
        {"seed": seed, "position_improved": True, "velocity_improved": True} for seed in (11, 29)
    ]
    prefixes = report["prefix_checks"]
    assert len(prefixes) == 4
    assert {(row["scenario"], row["seed"]) for row in prefixes} == {
        (scenario, seed) for scenario in ("falling", "constant_high") for seed in (11, 29)
    }
    assert all(row["bit_exact"] and not row["different_fields"] for row in prefixes)
    assert all(row["before_s"] == 5.5 and row["samples"] == 2750 for row in prefixes)
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns)
             for path in directory.rglob("*") if path.is_file()}
    assert after == before
