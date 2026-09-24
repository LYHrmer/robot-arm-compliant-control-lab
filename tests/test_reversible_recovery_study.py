"""Gate and archive tests; synthetic fixtures are not physics evidence."""

import copy
import json
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from tools.reversible_recovery import study
from tools.reversible_recovery.controller import ReversibleHoldCapTracking


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("unit tests must not start physics")
    monkeypatch.setattr(study.original.runner, "_run_loop", forbidden)
    monkeypatch.setattr(study, "ScheduledSurfaceSimulator", forbidden)
    monkeypatch.setattr(study.original.previous, "_save_trace", forbidden)


@pytest.mark.parametrize("scope,counts,total", [
    ("pilot", {"transfer": 8, "stationary": 4, "new": 4}, 16),
    ("transfer", {"transfer": 72, "pilot": 4, "new": 32}, 108),
])
def test_grid(scope, counts, total):
    specs = study.specifications(scope)
    assert len(specs) == len({study.transfer.run_key(s) for s in specs}) == total
    assert Counter(s["origin"] for s in specs) == counts
    for spec in specs:
        baseline = next(s for s in specs if study.transfer.run_key(s) == (
            *study.transfer.run_key(spec)[:-1], "adaptive6_8",
        ))
        assert spec["case"] == baseline["case"] and spec["fault"] == baseline["fault"]
    with pytest.raises(ValueError):
        study.specifications("unknown")


def test_both_scope_design_and_reference_sources():
    for scope in study.COUNTS:
        protocol = study.protocol_document(scope)
        for field in ("acceptance", "absolute_gates", "windows"):
            assert protocol[field] == study.original.protocol_document()[field]
        assert len(protocol["frozen_transfer_cases"]) == 36
        assert protocol["scopes"]["pilot"]["maximum_new_simulations"] == 4
        assert protocol["scopes"]["transfer"]["maximum_new_simulations"] == 32
        assert protocol["default_changed"] is protocol["new_holdout"] is False
        candidate = next(s for s in study.specifications(scope) if s["method"] == study.METHOD)
        assert protocol["controller_parameters"] == study.make_controller(candidate)[1]
        assert protocol["controller_parameters"]["min_update_speed"] > protocol["release_speed_max_m_s"]
    old = json.loads((study.STATIONARY_REFERENCE / "source_hashes.json").read_text())
    assert len(old) == 168 and study.stationary.source_identity() == old
    current = study.source_identity()
    assert old.items() <= current.items()
    assert {"tools/reversible_recovery/study.py", "tools/reversible_recovery/controller.py",
            "tools/reversible_recovery/audit.py", "tools/reversible_recovery/__init__.py",
            "tools/ci/check_replay_kernel.py", "tools/ci/recovery_runtime_matrix.py"} <= current.keys()


def test_constructor():
    for spec in study.specifications():
        controller, parameters = study.make_controller(spec)
        _, old = study.original.make_controller({**spec, "method": "adaptive6_8"}, study.original.protocol_document())
        assert parameters == old and parameters["velocity_error_time"] == 0.05
        if spec["method"] == study.METHOD:
            assert type(controller._base.tangential) is ReversibleHoldCapTracking


def rows_for(scope):
    rows = []
    for spec in study.specifications(scope):
        metrics = {"force_rmse_n": 1.0, "orientation_rmse_deg": 0.1,
                   "contact_ratio_pct": 100.0, "peak_force_n": 20.0, "saturation_pct": 0.0}
        windows = []
        for window in study.original.protocol_document()["windows"]:
            position, velocity = 2.0, 10.0
            if spec["scenario"] == "falling" and spec["method"] != "adaptive6_8" and window["name"] in {"early", "post"}:
                position = 1.5
            if spec["scenario"] == "constant_high" and window["name"] == "ramp":
                if spec["method"] == study.METHOD:
                    position, velocity = 2.02, 10.2
                elif spec["method"] in {study.STATIONARY, "hold_cap_tracking"}:
                    position, velocity = 2.08, 10.4
            windows.append({**window, "tangent_rmse_mm": position, "tangent_velocity_rmse_mm_s": velocity})
        rows.append({**{name: spec[name] for name in study.transfer.RUN_KEYS},
                     "overall": {**metrics, "projection_pct": 0.0, "minimum_reserved_torque_headroom_nm": 4.0},
                     "phases": [{"phase": phase.name, **metrics} for phase in spec["case"].phases], "windows": windows})
    return rows


def high_checks():
    return [{"seed": seed, "position_improved": True, "velocity_improved": True} for seed in (11, 29)]


@pytest.mark.parametrize("scope", ["pilot", "transfer"])
def test_every_primary_pair_must_pass(scope):
    rows = rows_for(scope)
    checks = high_checks() if scope == "transfer" else None
    comparisons, _, passed = study.compare_runs(rows, scope, checks)
    assert passed is True
    assert sum(row["compared_method"] == study.METHOD for row in comparisons) == (4 if scope == "pilot" else 36)
    for candidate in [row for row in rows if row["method"] == study.METHOD]:
        candidate["overall"]["projection_pct"] = 0.1
        comparisons, _, passed = study.compare_runs(rows, scope, checks)
        assert passed is False
        assert sum(row["status"] == "FAIL" for row in comparisons if row["compared_method"] == study.METHOD) == 1
        candidate["overall"]["projection_pct"] = 0.0


@pytest.mark.parametrize("seed", [11, 29])
@pytest.mark.parametrize("metric", ["tangent_rmse_mm", "tangent_velocity_rmse_mm_s"])
@pytest.mark.parametrize("offset", [0.0, -0.01])
def test_strict_improvement_against_stationary(seed, metric, offset):
    rows = rows_for("pilot")
    ramps = {row["method"]: next(w for w in row["windows"] if w["name"] == "ramp")
             for row in rows if row["scenario"] == "constant_high" and row["seed"] == seed}
    ramps[study.STATIONARY][metric] = ramps[study.METHOD][metric] + offset
    comparisons, _, passed = study.compare_runs(rows)
    assert passed is False
    assert all(row["status"] == "PASS" for row in comparisons if row["compared_method"] == study.METHOD)


@pytest.mark.parametrize("scope", ["pilot", "transfer"])
@pytest.mark.parametrize("corruption", ["missing", "duplicate", "method", "yaw", "profile"])
def test_bad_grids(scope, corruption):
    rows = rows_for(scope)
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    else:
        rows[0][{"method": "method", "yaw": "surface_yaw_deg", "profile": "error_profile"}[corruption]] = "unknown"
    with pytest.raises(ValueError):
        study.compare_runs(rows, scope, high_checks() if scope == "transfer" else None)


@pytest.mark.parametrize("checks", [None, [], [{"seed": 11, "position_improved": True, "velocity_improved": True}] * 2,
                                     [{"seed": seed, "position_improved": 1, "velocity_improved": True} for seed in (11, 29)]])
def test_transfer_high_gate_identity_and_types(checks):
    with pytest.raises(ValueError):
        study.compare_runs(rows_for("transfer"), "transfer", checks)


@pytest.mark.parametrize("scope,expected", [("pilot", 4), ("transfer", 48)])
def test_prefix_counts_and_auxiliary_pairs(scope, expected):
    traces = {study.transfer.run_key(spec): {"time": np.array([0., 5., 5.75, 6.0]), "signal": np.zeros(4)}
              for spec in study.specifications(scope)}
    prefixes = study.prefix_checks(traces, scope)
    assert len(prefixes) == expected and all(row["bit_exact"] for row in prefixes)
    if scope == "transfer":
        traces[-15, "combined_scale_0p8", "constant_high", 29, study.METHOD]["signal"][2] = 1.
        failed = [row for row in study.prefix_checks(traces, scope) if not row["bit_exact"]]
        assert len(failed) == 1 and failed[0]["kind"] == "before_auxiliary_scale_fault"


def test_yaw_zero_old_reference_location():
    spec = next(s for s in study.specifications("transfer")
                if s["surface_yaw_deg"] == 0 and s["error_profile"] == "clean" and s["origin"] == "transfer")
    path = study.trace_location(study.ROOT / "unused", spec)
    assert path.is_file()
    assert path.parent.parent == study.ROOT / study.transfer.protocol_document()["reference"]["directory"]


@pytest.mark.parametrize("arguments", [[], ["--output"], ["--audit"], ["--scope", "invalid", "--output", "x"],
    ["--output", "x", "--audit", "y"], ["--scope", "transfer", "--output", "x"], ["--output", "x", "--pilot-reference", "y"]])
def test_invalid_cli(arguments, monkeypatch):
    monkeypatch.setattr(study, "run", lambda *_a, **_k: pytest.fail("run called"))
    monkeypatch.setattr(study, "audit_archive", lambda *_a, **_k: pytest.fail("audit called"))
    monkeypatch.setattr(sys, "argv", ["study", *arguments])
    with pytest.raises(SystemExit) as caught:
        study.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "parent_symlink"])
def test_existing_output_and_symlinks(tmp_path, monkeypatch, kind):
    destination = tmp_path / "out"
    if kind == "file":
        destination.write_text("preserve")
    elif kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / "absent", target_is_directory=kind == "parent_symlink")
        if kind == "parent_symlink":
            destination = destination / "out"
    monkeypatch.setattr(study.subprocess, "check_output", lambda *_a, **_k: pytest.fail("git called"))
    with pytest.raises(ValueError, match="output must be new"):
        study.run(destination)


@pytest.mark.parametrize("failure", ["dirty", "untracked", "kernel"])
def test_run_guards(tmp_path, monkeypatch, failure):
    def git(args, **_kwargs):
        if args[1] == "status":
            return b" M tool.py" if failure == "dirty" else b""
        if args[1] == "rev-parse":
            return "a" * 40
        if failure == "untracked":
            raise subprocess.CalledProcessError(1, args)
        return b""
    monkeypatch.setattr(study.subprocess, "check_output", git)
    monkeypatch.setattr(study.check_replay_kernel, "main", lambda: (_ for _ in ()).throw(RuntimeError("wrong kernel")))
    with pytest.raises((ValueError, subprocess.CalledProcessError, RuntimeError)):
        study.run(tmp_path / "not_created")
    assert not (tmp_path / "not_created").exists()


def seal(directory, manifest):
    manifest["artifact_sha256"] = {name: study._sha256(directory / name) for name in manifest["artifact_sha256"]}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "COMPLETE").write_text(study._sha256(directory / "manifest.json") + "\n")


@pytest.fixture
def metadata_archive(tmp_path, monkeypatch):
    protocol = study.protocol_document()
    artifacts = {"protocol.json": json.dumps(protocol), "source_hashes.json": json.dumps(study.source_identity()),
                 "comparison.json": "{}", **{spec["trace_path"]: "not a physics trace"
                 for spec in study.specifications() if spec["origin"] == "new"}}
    for name, content in artifacts.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    manifest = {"identity": protocol["identity"], "scope": "pilot", "source_commit": "a" * 40,
                **protocol["scopes"]["pilot"], "default_changed": False, "new_holdout": False,
                "public_development": True, "artifact_sha256": dict.fromkeys(artifacts)}
    seal(tmp_path, manifest)
    monkeypatch.setattr(study, "collect", lambda *_a, **_k: pytest.fail("invalid archive reached replay"))
    return tmp_path, manifest


@pytest.mark.parametrize("field,value", [("scope", "invalid"), ("evaluated_runs", 15), ("reused_runs", 11),
    ("maximum_new_simulations", 5), ("paired_cases", 4.0), ("default_changed", 0), ("new_holdout", True),
    ("public_development", False), ("source_commit", "g" * 40), ("source_commit", True)])
def test_resealed_metadata_tampering(metadata_archive, field, value):
    directory, manifest = metadata_archive
    manifest[field] = value
    seal(directory, manifest)
    with pytest.raises(ValueError):
        study.audit_archive(directory)


@pytest.mark.parametrize("name", ["source_hashes.json", "protocol.json"])
def test_resealed_frozen_inputs(metadata_archive, name):
    directory, manifest = metadata_archive
    path = directory / name
    value = json.loads(path.read_text())
    value[next(iter(value))] = "changed"
    path.write_text(json.dumps(value))
    seal(directory, manifest)
    with pytest.raises(ValueError):
        study.audit_archive(directory)


def test_failed_pilot_cannot_unlock_transfer(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "verify_archive", lambda *_a, **_k: {"scope": "pilot"})
    monkeypatch.setattr(study, "audit_archive", lambda *_a, **_k: {"comparison_status": "FAIL"})
    with pytest.raises(ValueError, match="successful pilot"):
        study.verified_pilot(tmp_path)


def test_non_pilot_cannot_unlock_transfer(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "verify_archive", lambda *_a, **_k: {"scope": "transfer"})
    monkeypatch.setattr(study, "audit_archive", lambda *_a, **_k: pytest.fail("should reject before recursion"))
    with pytest.raises(ValueError, match="must be a pilot"):
        study.verified_pilot(tmp_path)


def test_pilot_digest_verified_before_audit(monkeypatch, tmp_path):
    seen = []
    def verify(directory, expected_digest=None):
        seen.append(expected_digest)
        raise ValueError("pinned reference manifest differs")
    monkeypatch.setattr(study, "verify_archive", verify)
    monkeypatch.setattr(study, "audit_archive", lambda *_a, **_k: pytest.fail("invalid hash reached audit"))
    with pytest.raises(ValueError, match="pinned reference"):
        study.verified_pilot(tmp_path, expected_digest="0" * 64)
    assert seen == ["0" * 64]


def test_unsealed_trace_tampering_rejected_before_replay(metadata_archive):
    directory, _ = metadata_archive
    path = next(directory.glob("traces/*.npz"))
    path.write_text("modified")
    with pytest.raises(ValueError, match="artifact hash differs"):
        study.audit_archive(directory)


def test_resealed_failed_summary_cannot_claim_pass(metadata_archive, monkeypatch):
    directory, manifest = metadata_archive
    (directory / "comparison.json").write_text('{"status": "PASS"}')
    seal(directory, manifest)
    monkeypatch.setattr(study, "collect", lambda *_a, **_k: {"status": "FAIL"})
    with pytest.raises(ValueError, match="differs"):
        study.audit_archive(directory)


def test_archive_integrity_pass_does_not_promote_failed_experiment(metadata_archive, monkeypatch):
    directory, manifest = metadata_archive
    (directory / "comparison.json").write_text('{"status": "FAIL"}')
    seal(directory, manifest)
    monkeypatch.setattr(study, "collect", lambda *_a, **_k: {"status": "FAIL"})
    result = study.audit_archive(directory)
    assert result["archive_integrity"] == "PASS"
    assert result["comparison_status"] == "FAIL"
    assert result["default_changed"] is False


def test_changed_source_pilot_cannot_unlock_transfer(metadata_archive):
    directory, manifest = metadata_archive
    path = directory / "source_hashes.json"
    sources = json.loads(path.read_text())
    sources["tools/reversible_recovery/controller.py"] = "0" * 64
    path.write_text(json.dumps(sources))
    seal(directory, manifest)
    with pytest.raises(ValueError, match="source identity differs"):
        study.verified_pilot(directory)


@pytest.mark.parametrize("declared", ["/tmp/not-portable", "../outside"])
def test_transfer_rejects_unsafe_stored_reference_before_pilot_loading(tmp_path, monkeypatch, declared):
    protocol = study.protocol_document("transfer")
    artifacts = {"protocol.json": json.dumps(protocol), "source_hashes.json": json.dumps(study.source_identity()),
                 "comparison.json": "{}", **{s["trace_path"]: "metadata-only, not simulation"
                 for s in study.specifications("transfer") if s["origin"] == "new"}}
    for name, content in artifacts.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    manifest = {"identity": protocol["identity"], "scope": "transfer", "source_commit": "a" * 40,
                **protocol["scopes"]["transfer"], "default_changed": False, "new_holdout": False,
                "public_development": True, "artifact_sha256": dict.fromkeys(artifacts),
                "pilot_reference": {"directory": declared, "manifest_sha256": "0" * 64}}
    seal(tmp_path, manifest)
    monkeypatch.setattr(study, "verified_pilot", lambda *_a, **_k: pytest.fail("unsafe path reached pilot audit"))
    with pytest.raises(ValueError, match="unsafe archive artifact"):
        study.audit_archive(tmp_path)


def test_prefix_failure_overrides_successful_metric_gates(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "verify_references", lambda: None)
    monkeypatch.setattr(study, "specifications", lambda _scope: [])
    monkeypatch.setattr(study, "compare_runs", lambda *_a: ([], high_checks(), True))
    monkeypatch.setattr(study, "prefix_checks", lambda *_a: [{"bit_exact": False}])
    result = study.collect(tmp_path, execute=False)
    assert result["status"] == "FAIL"
    assert result["eligible_for_default_change"] is False


def test_constructed_parameter_drift_rejected_before_loading_or_physics(monkeypatch, tmp_path):
    protocol = study.protocol_document()
    actual_constructor = study.make_controller
    def changed(spec):
        controller, parameters = actual_constructor(spec)
        return controller, {**parameters, "min_update_speed": 1e-13}
    monkeypatch.setattr(study, "verify_references", lambda: None)
    monkeypatch.setattr(study, "protocol_document", lambda _scope="pilot": protocol)
    monkeypatch.setattr(study, "make_controller", changed)
    monkeypatch.setattr(study.original.previous, "_load_trace", lambda *_a: pytest.fail("drift reached trace loading"))
    with pytest.raises(ValueError, match="constructed controller parameters"):
        study.collect(tmp_path, execute=False)
