"""Fast publisher checks for the 60-run measured-budget regression."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import measured_budget_study as study


def _reused_spec(suite):
    return next(
        spec
        for spec in study.specifications()
        if spec["suite"] == suite and spec["reused"]
    )


def _pilot_trace(spec):
    path = study.ROOT / study.PILOT["directory"] / "traces" / spec["trace_name"]
    return study.transfer.load_trace(path)


def test_protocol_freezes_counts_parameters_and_screening_thresholds():
    protocol = study.protocol_document()
    pilot_protocol = study.pilot.protocol_document()

    assert protocol["new_simulations"] == 42
    assert protocol["reused_pilot_candidates"] == 18
    assert protocol["candidate_rows"] == 60
    assert protocol["new_public_runs"] == 36
    assert protocol["new_no_bias_runs"] == 6
    assert protocol["parameters"] == pilot_protocol["parameters"]
    assert protocol["public_limits"] == dict(study.screening.PUBLIC_LIMITS)
    assert protocol["dynamic_limits"] == dict(study.screening.CRITERIA)
    assert protocol["gain_limits"] == dict(study.screening.GAIN_LIMITS)
    assert protocol["absolute_limits"] == dict(study.screening.ABSOLUTE)


def test_protocol_configuration_counts_agree_with_execution_contract():
    specs = study.specifications()
    protocol = study.protocol_document()

    assert len(specs) == protocol["candidate_rows"]
    assert sum(spec["reused"] for spec in specs) == protocol["reused_pilot_candidates"]
    assert sum(not spec["reused"] for spec in specs) == protocol["new_simulations"]
    assert len(protocol["configurations"]) == len(specs)


def test_protocol_is_stable_across_json_publication_roundtrip():
    protocol = study.protocol_document()

    assert json.loads(json.dumps(protocol)) == protocol


@pytest.mark.parametrize("suite", ("public", "dynamic"))
def test_validate_candidate_accepts_freshly_loaded_pilot_schema(suite):
    spec = _reused_spec(suite)
    trace = _pilot_trace(spec)
    reference = study.transfer.load_trace(study.reference_path(spec))

    result = study.validate_candidate(spec, trace, reference)

    assert result["validated_cycles"] == len(trace["time"])
    assert result["max_scheduler_reconstruction_error_n"] <= 1e-10


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda trace: trace.__setitem__("rotation_gain_scale", np.array(9.0)), "identity"),
        (lambda trace: trace.__setitem__("unexpected", np.array(1.0)), "schema"),
        (lambda trace: trace.__setitem__("time", trace["time"][:-1]), "schedule"),
        (
            lambda trace: trace.__setitem__(
                "load_force_local", trace["load_force_local"].astype(np.float32)
            ),
            "float64",
        ),
    ),
    ids=("wrong-metadata", "extra-field", "truncated", "wrong-dtype"),
)
def test_validate_candidate_rejects_public_trace_tampering(mutation, match):
    spec = _reused_spec("public")
    trace = _pilot_trace(spec)
    reference = study.transfer.load_trace(study.reference_path(spec))
    mutation(trace)

    with pytest.raises(ValueError, match=match):
        study.validate_candidate(spec, trace, reference)


@pytest.mark.parametrize("field", ("lower_torque_limit", "upper_torque_limit"))
def test_validate_candidate_rejects_changed_dynamic_physical_limits(field):
    spec = _reused_spec("dynamic")
    trace = _pilot_trace(spec)
    reference = study.transfer.load_trace(study.reference_path(spec))
    trace[field][0, 0] += 1.0

    with pytest.raises(ValueError, match=field):
        study.validate_candidate(spec, trace, reference)


def test_validate_candidate_does_not_mutate_loaded_arrays():
    spec = _reused_spec("public")
    trace = _pilot_trace(spec)
    reference = study.transfer.load_trace(study.reference_path(spec))
    before = {name: value.copy() for name, value in trace.items()}

    study.validate_candidate(spec, trace, reference)

    assert trace.keys() == before.keys()
    assert all(np.array_equal(trace[name], value) for name, value in before.items())


def test_unchanged_prefix_rejects_a_difference_before_six_seconds():
    combined = {
        "time": np.array([5.998, 6.0, 6.002]),
        "signal": np.array([1.0, 2.0, 3.0]),
        "metadata": np.array(1.0),
    }
    no_bias = deepcopy(combined)
    no_bias["signal"][0] += 1.0

    with pytest.raises(ValueError, match="pre-bias prefix differs: signal"):
        study.unchanged_prefix(combined, no_bias)


def test_unchanged_prefix_allows_the_first_difference_at_six_seconds():
    combined = {
        "time": np.array([5.998, 6.0, 6.002]),
        "signal": np.array([1.0, 2.0, 3.0]),
        "metadata": np.array(1.0),
    }
    no_bias = deepcopy(combined)
    no_bias["signal"][1:] += 1.0

    result = study.unchanged_prefix(combined, no_bias)

    assert result == {
        "samples": 1,
        "fields": ["signal", "time"],
        "bit_exact": True,
    }


def test_collect_executes_only_42_new_rows_and_reuses_exactly_18(tmp_path, monkeypatch):
    (tmp_path / "traces").mkdir()
    pilot_root = tmp_path / "pinned-pilot"
    fake = {
        "time": np.array([0.0]),
        "load_budget_applied_n": np.array([6.5]),
        "controller_coefficient_before_compute": np.array([0.9]),
        "diagnostic_corrected_force_n": np.array([8.0]),
    }
    calls = {"dynamic": 0, "public": 0, "saved": [], "loaded": []}

    def trial(suite):
        calls[suite] += 1
        return SimpleNamespace(trace=deepcopy(fake))

    def save(path, **trace):
        calls["saved"].append(Path(path).name)
        assert trace.keys() == fake.keys()

    def load(path):
        calls["loaded"].append(Path(path))
        return deepcopy(fake)

    monkeypatch.setattr(study, "input_archive", lambda: pilot_root)
    monkeypatch.setattr(study.runner, "run_dynamic", lambda *args: trial("dynamic"))
    monkeypatch.setattr(study.runner, "run_public", lambda *args: trial("public"))
    monkeypatch.setattr(study.runner, "compact_dynamic", deepcopy)
    monkeypatch.setattr(study.runner, "compact_public", deepcopy)
    monkeypatch.setattr(study.np, "savez_compressed", save)
    monkeypatch.setattr(study.transfer, "load_trace", load)
    monkeypatch.setattr(study, "validate_candidate", lambda *args: {})
    monkeypatch.setattr(study, "unchanged_prefix", lambda *args: {"bit_exact": True})
    monkeypatch.setattr(study.dynamic, "metrics", lambda *args: ({}, [], []))
    monkeypatch.setattr(study.public, "metrics", lambda *args: {})
    monkeypatch.setattr(study, "catchup_metrics", lambda *args: {})
    monkeypatch.setattr(
        study.screening,
        "screen",
        lambda dynamic_rows, public_rows: {
            "status": "PASS",
            "dynamic_rows": len(dynamic_rows),
            "public_rows": len(public_rows),
        },
    )

    result = study.collect(tmp_path, execute=True)

    assert calls["dynamic"] == 6
    assert calls["public"] == 36
    assert len(calls["saved"]) == 42
    assert len(set(calls["saved"])) == 42
    reused_names = [spec["trace_name"] for spec in study.specifications() if spec["reused"]]
    prefix_names = [
        f"dynamic__yaw{yaw}__combined__s{scale}.npz"
        for yaw in (-15, 0, 15)
        for scale in (1, 2)
    ]
    loaded_from_pilot = [
        path.name for path in calls["loaded"] if pilot_root in path.parents
    ]
    assert sorted(loaded_from_pilot) == sorted(reused_names + prefix_names)
    assert len(result["audits"]) == 60
    assert sum(row["row_origin"] == "pinned_pilot" for row in result["audits"]) == 18
    assert sum(row["row_origin"] == "executed" for row in result["audits"]) == 42
    assert len(result["dynamic"]) == 12
    assert len(result["public"]) == 48


@pytest.mark.parametrize("occupied", ("directory", "symlink"))
def test_run_rejects_occupied_or_symlink_output_before_collection(
    tmp_path, monkeypatch, occupied
):
    output = tmp_path / "published"
    if occupied == "directory":
        output.mkdir()
    else:
        target = tmp_path / "target"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        study,
        "collect",
        lambda *args, **kwargs: pytest.fail("invalid output must fail before collection"),
    )

    with pytest.raises(ValueError, match="must not exist|symlink"):
        study.run(output)


def test_audit_rejects_nonexistent_archive(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        study.audit_archive(tmp_path / "missing")


def _archive_fixture(root, monkeypatch):
    protocol = {"identity": study.IDENTITY, "frozen": True}
    sources = {"source.py": "a" * 64}
    comparison = {"screening": {"status": "PASS"}, "rows": [1, 2, 3]}
    monkeypatch.setattr(study, "protocol_document", lambda: deepcopy(protocol))
    monkeypatch.setattr(study, "source_identity", lambda: deepcopy(sources))
    monkeypatch.setattr(study, "collect", lambda *args, **kwargs: deepcopy(comparison))

    for name in study.expected_artifacts():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "protocol.json":
            study._write_json(path, protocol)
        elif name == "source_hashes.json":
            study._write_json(path, sources)
        elif name == "comparison.json":
            study._write_json(path, comparison)
        else:
            path.write_bytes(b"synthetic trace; never used as physics evidence")
    artifacts = {
        name: study._sha256(root / name) for name in sorted(study.expected_artifacts())
    }
    manifest = {
        "identity": study.IDENTITY,
        "pilot": study.PILOT,
        "new_simulations": 42,
        "reused_candidates": 18,
        "artifact_sha256": artifacts,
    }
    study._write_json(root / "manifest.json", manifest)
    (root / "COMPLETE").write_text(study._sha256(root / "manifest.json") + "\n")
    return comparison


def _reseal(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"] = {
        name: study._sha256(root / name) for name in manifest["artifact_sha256"]
    }
    study._write_json(manifest_path, manifest)
    (root / "COMPLETE").write_text(study._sha256(manifest_path) + "\n")


def test_audit_rejects_resealed_comparison_tamper(tmp_path, monkeypatch):
    comparison = _archive_fixture(tmp_path, monkeypatch)
    altered = deepcopy(comparison)
    altered["rows"][0] = 999
    study._write_json(tmp_path / "comparison.json", altered)
    _reseal(tmp_path)

    with pytest.raises(ValueError, match="recomputed.*comparison"):
        study.audit_archive(tmp_path)


def test_audit_rejects_resealed_missing_inventory_entry(tmp_path, monkeypatch):
    _archive_fixture(tmp_path, monkeypatch)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"].pop(next(iter(study.expected_artifacts() - {
        "protocol.json", "source_hashes.json", "comparison.json"
    })))
    study._write_json(manifest_path, manifest)
    (tmp_path / "COMPLETE").write_text(study._sha256(manifest_path) + "\n")

    with pytest.raises(ValueError, match="inventory|archive files"):
        study.audit_archive(tmp_path)
