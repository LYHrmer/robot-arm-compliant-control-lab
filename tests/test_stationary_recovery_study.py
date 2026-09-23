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
