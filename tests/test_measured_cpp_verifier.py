import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_measured_budget_cpp as verifier

DEMO = Path(__file__).parents[1] / "results/franka_measured_budget_demo/trace.npz"


@pytest.fixture(scope="module")
def demo_arrays():
    if not DEMO.is_file():
        pytest.skip("measured-budget demo trace is unavailable")
    with np.load(DEMO, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def changed(arrays, name, value):
    result = dict(arrays)
    result[name] = np.asarray(value)
    return result


def probe_row():
    return [
        "surface_case",
        "0",
        "accepted",
        "unchanged",
        "0",
        "1",
        *["0"] * 6,
        "1",
        "12",
        "0",
        "0.45",
        *["0"] * 3,
        "0",
        "1",
        "4000",
        "1",
        "1",
        "1",
        "1",
        *["0"] * 3,
        "0",
        "0",
        "6",
        "6.1",
        "5.8",
        "1",
        "5.9",
        *["0"] * 3,
        "1",
        "0.45",
        "1",
        "1",
        "0",
        "1",
        *["0"] * 7,
        "1",
    ]


def test_measured_probe_parser_requires_exact_count_and_shape():
    row = probe_row()
    assert len(row) == 53
    parsed = verifier._parse_probe_output(",".join(row), 1)
    assert parsed["commanded_wrench"].shape == (1, 6)
    assert parsed["commanded_torque"].shape == (1, 7)
    assert parsed["load_packet_status"].tolist() == [1]
    assert parsed["load_budget_updated"].tolist() == [True]
    with pytest.raises(ValueError, match="returned"):
        verifier._parse_probe_output(",".join(row), 2)
    with pytest.raises(ValueError, match="malformed"):
        verifier._parse_probe_output(",".join(row[:-1]), 1)


@pytest.mark.parametrize(
    ("index", "value"),
    [
        (1, "1"),
        (2, "expired"),
        (3, "fake"),
        (4, "2"),
        (24, "6"),
        (25, "-1"),
        (33, "nan"),
        (45, "inf"),
    ],
)
def test_measured_probe_parser_rejects_corruption(index, value):
    row = probe_row()
    row[index] = value
    with pytest.raises(ValueError):
        verifier._parse_probe_output(",".join(row), 1)


def test_schema2_full_trace_metadata_is_accepted(demo_arrays):
    count, metadata = verifier._validate_measured_trace(demo_arrays)
    assert count == 6000
    assert metadata == {
        "trace_schema_version": 2,
        "controller_kind": "surface_online",
        "method": "adaptive6_8",
        "minimum_force_n": 6.0,
        "maximum_force_n": 8.0,
        "rotation_gain_scale": 1.0,
        "maximum_packet_age_s": 0.02,
        "fault": {
            "name": "fresh",
            "kind": "fresh",
            "start_s": 0.0,
            "end_s": 0.0,
            "value": 0.0,
        },
    }


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("trace_schema_version", 1, "trace_schema_version"),
        ("controller_kind", "surface_friction", "surface_online"),
        ("method", "invented", "method"),
        ("rotation_gain_scale", 0.0, "rotation_gain_scale"),
        ("rotation_gain_scale", np.nan, "finite real scalar"),
        ("rotation_gain_scale", True, "finite real scalar"),
        ("rotation_gain_scale", [1.0], "finite real scalar"),
        ("minimum_force_n", 5.0, "budget metadata"),
        ("max_force_n", 7.0, "budget metadata"),
        ("load_max_measurement_age_s", 0.03, "frozen runner"),
        ("fault_kind", "invented", "fault metadata"),
        ("fault_end_s", -1.0, "fault metadata"),
    ],
)
def test_metadata_corruption_is_rejected(demo_arrays, name, value, message):
    with pytest.raises(ValueError, match=message):
        verifier._validate_measured_trace(changed(demo_arrays, name, value))


def test_missing_full_field_and_packet_shape_are_rejected(demo_arrays):
    missing = dict(demo_arrays)
    del missing["measured_angular_velocity"]
    with pytest.raises(ValueError, match="full trace schema mismatch"):
        verifier._validate_measured_trace(missing)
    malformed = changed(
        demo_arrays,
        "raw_load_packet_force",
        demo_arrays["raw_load_packet_force"][:, :2],
    )
    with pytest.raises(ValueError, match="invalid full-state shape"):
        verifier._validate_measured_trace(malformed)


def test_nonfinite_full_trace_value_is_rejected(demo_arrays):
    force = demo_arrays["raw_load_packet_force"].copy()
    force[10, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        verifier._validate_measured_trace(changed(demo_arrays, "raw_load_packet_force", force))


def test_python_full_chain_replays_recorded_prefix_exactly(demo_arrays):
    count = 350
    arrays = {
        name: value[:count].copy() if name not in verifier.STATIC_TRACE_FIELDS else value.copy()
        for name, value in demo_arrays.items()
    }
    actual_count, metadata = verifier._validate_measured_trace(arrays)
    replay = verifier._python_replay(arrays, actual_count, metadata)
    comparison = verifier._compare(replay, arrays)
    assert comparison["matches"] is True
    assert max(comparison["max_abs_errors"].values()) == 0.0
    assert not any(comparison["exact_mismatch_counts"].values())


def test_comparison_uses_tolerance_for_numbers_and_exact_flags():
    expected = {}
    for name in verifier.NUMERIC_FIELDS:
        expected[name] = np.zeros((2, 6)) if name == "commanded_wrench" else np.zeros(2)
    expected["commanded_torque"] = np.zeros((2, 7))
    expected["requested_tangential_force_world"] = np.zeros((2, 3))
    expected["load_force_local"] = np.zeros((2, 3))
    expected["load_compensation_force_local"] = np.zeros((2, 3))
    for name in verifier.BOOLEAN_FIELDS:
        expected[name] = np.zeros(2, dtype=bool)
    expected["load_packet_status"] = np.zeros(2, dtype=np.uint8)
    actual = {name: value.copy() for name, value in expected.items()}
    actual["load_estimate_n"][0] = verifier.TOLERANCE
    assert verifier._compare(actual, expected)["matches"] is True
    actual["load_estimate_n"][0] = np.nextafter(verifier.TOLERANCE, np.inf)
    assert verifier._compare(actual, expected)["matches"] is False
    actual["load_estimate_n"][0] = 0.0
    actual["load_budget_updated"][0] = True
    assert verifier._compare(actual, expected)["matches"] is False


def test_encoder_uses_trace_time_for_both_watchdog_timestamps(demo_arrays):
    encoded = verifier._encode_probe_input(demo_arrays, 2).splitlines()
    first = encoded[1].split()
    second = encoded[2].split()
    assert first[2:4] == ["0", "0"]
    assert second[2] == second[3] == format(float(demo_arrays["time"][1]), ".17g")
    assert first[-5:] == [
        "1",
        *verifier._numbers(demo_arrays["raw_load_packet_force"][0]),
        *verifier._numbers([demo_arrays["raw_load_packet_stamp"][0]]),
    ]


def test_run_trace_uses_agreed_measured_budget_probe_flags(monkeypatch):
    replay = {}
    for name in verifier.NUMERIC_FIELDS:
        replay[name] = np.zeros((1, 6)) if name == "commanded_wrench" else np.zeros(1)
    replay["commanded_torque"] = np.zeros((1, 7))
    for name in (
        "requested_tangential_force_world",
        "load_force_local",
        "load_compensation_force_local",
    ):
        replay[name] = np.zeros((1, 3))
    for name in verifier.BOOLEAN_FIELDS:
        replay[name] = np.zeros(1, dtype=bool)
    replay["load_packet_status"] = np.ones(1, dtype=np.uint8)
    metadata = {
        "rotation_gain_scale": 2.0,
        "minimum_force_n": 6.0,
        "maximum_force_n": 8.0,
        "maximum_packet_age_s": 0.02,
    }
    commands = []
    monkeypatch.setattr(verifier, "_load_trace", lambda path: (replay, 1, metadata))
    monkeypatch.setattr(verifier, "_python_replay", lambda *args: replay)
    monkeypatch.setattr(verifier, "_encode_probe_input", lambda arrays, count: "input")
    monkeypatch.setattr(verifier, "_parse_probe_output", lambda output, count: replay)

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="output", stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", fake_run)
    summary, _ = verifier._run_trace(Path("trace.npz"), Path("probe"))
    assert summary["matches"] is True
    assert commands == [
        [
            "probe",
            "--mode",
            "online",
            "--rotation-gain-scale",
            "2",
            "--load-budget",
            "6",
            "8",
            "--maximum-packet-age",
            "0.02",
        ]
    ]


def make_archive(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "source.py"
    source.write_text("source\n", encoding="utf-8")
    trace = repository / "trace.npz"
    trace.write_bytes(b"trace")
    probe = repository / "probe"
    probe.write_bytes(b"probe")
    monkeypatch.setattr(verifier, "SOURCE_PATHS", (source,))
    monkeypatch.setattr(verifier, "REPOSITORY", repository)
    monkeypatch.setattr(verifier, "baseline_source_identity", dict)
    frozen = {
        "traces": {"trace.npz": verifier._sha256(trace)},
        "sources": {"source.py": verifier._sha256(source)},
        "probe_run_path": str(probe),
        "probe_binary": verifier._sha256(probe),
    }
    comparison = {
        "max_abs_errors": {name: 0.0 for name in verifier.NUMERIC_FIELDS},
        "exact_mismatch_counts": {
            name: 0 for name in (*verifier.BOOLEAN_FIELDS, *verifier.INTEGER_FIELDS)
        },
        "matches": True,
    }
    report = {
        "identity": "measured-budget-cpp-replay-v1",
        "tolerance": verifier.TOLERANCE,
        "protocol": {
            "identity": verifier.PROBE_PROTOCOL,
            "numeric_tolerance_fields": list(verifier.NUMERIC_FIELDS),
            "exact_fields": [*verifier.BOOLEAN_FIELDS, *verifier.INTEGER_FIELDS],
        },
        "trace_count": 1,
        "independent_trace_count": 1,
        "demo_trace_count": 0,
        "sample_count": 2,
        "matches": True,
        "traces": [
            {
                "trace": "trace.npz",
                "sample_count": 2,
                "matches": True,
                "python_vs_trace": comparison,
                "cpp_vs_python": comparison,
                "cpp_vs_trace": comparison,
                "gate_effect": {
                    "accepted_samples": 2,
                    "rejected_samples": 0,
                    "budget_update_samples": 2,
                    "status_counts": {
                        name: 2 if name == "accepted" else 0
                        for name in verifier.PACKET_STATUS_NAMES.values()
                    },
                },
            }
        ],
        "aggregate_results": {},
        "frozen_sha256": frozen,
    }
    report["aggregate_results"] = verifier._aggregate_results(report["traces"])
    output = verifier._write_archive(tmp_path / "archive", report, frozen)
    return output, probe


def reseal_archive(output):
    report_path = output / "report.json"
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = {"report.json": verifier._sha256(report_path)}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "COMPLETE").write_text(verifier._sha256(manifest_path) + "\n", encoding="utf-8")


def test_archive_audit_rejects_report_corruption(tmp_path, monkeypatch):
    output, _ = make_archive(tmp_path, monkeypatch)
    audited = verifier.audit_archive(output)
    assert audited["matches"] is True
    assert audited["audit"]["binary_verification"] == "not_checked"
    payload = json.loads((output / "report.json").read_text(encoding="utf-8"))
    payload["traces"][0]["cpp_vs_python"]["max_abs_errors"]["commanded_wrench"] = 1.0
    (output / "report.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hashes"):
        verifier.audit_archive(output)


def test_archive_audit_is_portable_and_binary_check_is_opt_in(tmp_path, monkeypatch):
    output, original_probe = make_archive(tmp_path, monkeypatch)
    matching_probe = tmp_path / "matching-probe"
    shutil.copyfile(original_probe, matching_probe)
    moved = tmp_path / "another-machine" / "archive"
    shutil.copytree(output, moved)
    original_probe.rename(tmp_path / "original-probe-unavailable")
    assert verifier.audit_archive(moved)["audit"]["binary_verification"] == "not_checked"
    assert verifier.audit_archive(moved, matching_probe)["audit"]["binary_verification"] == (
        "verified"
    )
    wrong_probe = tmp_path / "wrong-probe"
    wrong_probe.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="specified replay binary differs"):
        verifier.audit_archive(moved, wrong_probe)


def test_archive_audit_will_not_accept_a_resealed_larger_tolerance(tmp_path, monkeypatch):
    output, _ = make_archive(tmp_path, monkeypatch)
    report_path = output / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["tolerance"] = 1.0
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    reseal_archive(output)
    with pytest.raises(ValueError, match="counts or result summary"):
        verifier.audit_archive(output)


def test_archive_audit_rejects_resealed_trace_inventory_tamper(tmp_path, monkeypatch):
    output, _ = make_archive(tmp_path, monkeypatch)
    report_path = output / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["traces"][0]["trace"] = "unfrozen.npz"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    reseal_archive(output)
    with pytest.raises(ValueError, match="trace inventory"):
        verifier.audit_archive(output)


def test_archive_audit_rejects_balanced_negative_gate_counts(tmp_path, monkeypatch):
    output, _ = make_archive(tmp_path, monkeypatch)
    report_path = output / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    gate = payload["traces"][0]["gate_effect"]
    gate["status_counts"]["missing"] = -1
    gate["status_counts"]["accepted"] = 3
    gate["accepted_samples"] = 3
    gate["rejected_samples"] = -1
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    reseal_archive(output)
    with pytest.raises(ValueError, match="gate-effect"):
        verifier.audit_archive(output)


def test_full_demo_cpp_replay_when_measured_probe_is_built(tmp_path):
    probe = verifier.DEFAULT_PROBE
    if not probe.is_file():
        pytest.skip("C++ surface probe is not built")
    help_result = verifier.subprocess.run(
        [str(probe), "--help"], capture_output=True, text=True, check=False
    )
    if "--load-budget" not in (help_result.stdout + help_result.stderr):
        pytest.skip("C++ surface probe does not yet expose measured-budget mode")
    summary, _ = verifier._run_trace(DEMO, probe)
    assert summary["matches"] is True
