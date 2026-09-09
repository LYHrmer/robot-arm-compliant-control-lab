"""Evidence-tool format checks separate from the C++ numeric parity tests."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_cpp_surface_replay as verifier
from tools.verify_cpp_surface_replay import (
    _mode,
    _parse_probe_output,
    _rotation_gain_scale,
    verify,
)


def valid_row():
    return ["surface_case", "0", "accepted", "unchanged", "0", "1",
            *["0"] * 6, "1", "12", "0", "0.45", "0", "5", "0",
            "0.01", "1", "4000", "1", "1"]


def test_probe_parser_requires_exact_count_and_shape():
    row = valid_row()
    parsed = _parse_probe_output(",".join(row), 1)
    assert parsed[0]["equivalent_mu"] == 0.45
    assert parsed[0]["update_ready"] is True
    with pytest.raises(ValueError, match="rows"):
        _parse_probe_output(",".join(row), 2)
    with pytest.raises(ValueError, match="malformed"):
        _parse_probe_output(",".join(row[:-1]), 1)


@pytest.mark.parametrize("index,value", [(1, "1"), (2, "expired"), (3, "fake_pass"),
                                        (4, "2"), (5, "-1"), (23, "2"), (6, "nan")])
def test_probe_parser_rejects_invalid_status_or_numeric_value(index, value):
    row = valid_row()
    row[index] = value
    with pytest.raises(ValueError):
        _parse_probe_output(",".join(row), 1)


def test_mode_mapping_and_unknown_rejection():
    assert _mode("surface_online") == "online"
    assert _mode("surface_adaptive") == "none"
    with pytest.raises(ValueError, match="unsupported"):
        _mode("online_but_unidentified")


def test_rotation_gain_scale_defaults_to_one_and_accepts_positive_scalar():
    assert _rotation_gain_scale({}) == 1.0
    assert _rotation_gain_scale({"rotation_gain_scale": 2}) == 2.0
    assert _rotation_gain_scale({"rotation_gain_scale": np.array(0.25)}) == 0.25


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), float("inf"), 1e308, True, "2", [1.0]],
)
def test_rotation_gain_scale_rejects_invalid_metadata(value):
    with pytest.raises(ValueError, match="finite positive real scalar"):
        _rotation_gain_scale({"rotation_gain_scale": value})


@pytest.mark.parametrize(
    ("metadata", "expected_argument"),
    [({}, "1"), ({"rotation_gain_scale": np.array(4.0)}, "4")],
)
def test_run_trace_passes_rotation_gain_scale_to_probe(
    monkeypatch, metadata, expected_argument
):
    arrays = {
        "controller_kind": np.array("surface_adaptive"),
        "cartesian_jacobian": np.zeros((1, 6, 7)),
        "joint_torque_offset": np.zeros((1, 7)),
        "commanded_wrench": np.zeros((1, 6)),
        "commanded_torque": np.zeros((1, 7)),
        **metadata,
    }
    observed_command = []

    def fake_run(command, **kwargs):
        observed_command.extend(command)
        return SimpleNamespace(returncode=0, stdout="ignored", stderr="")

    monkeypatch.setattr(verifier, "_load_trace", lambda path: (arrays, 1))
    monkeypatch.setattr(verifier, "_encode_probe_input", lambda arrays, count: "")
    monkeypatch.setattr(
        verifier,
        "replay_surface_trace",
        lambda path: SimpleNamespace(
            matches=True,
            max_wrench_error=0.0,
            max_torque_error=0.0,
        ),
    )
    monkeypatch.setattr(
        verifier,
        "_parse_probe_output",
        lambda output, count: [
            {
                "projection_status": "unchanged",
                "fallback": False,
                "feasible": True,
                "wrench": np.zeros(6),
                "contact_blend": 0.0,
                "corrected_force": 0.0,
                "force_rate": 0.0,
                "equivalent_mu": 0.45,
                "tangential_force": np.zeros(3),
                "governed_lead": 0.0,
                "projection_scale": 1.0,
                "contact_stiffness": 4000.0,
                "gain_scale": 1.0,
                "update_ready": False,
            }
        ],
    )
    monkeypatch.setattr(verifier.subprocess, "run", fake_run)

    verifier._run_trace(Path("trace.npz"), Path("probe"))

    assert observed_command[-2:] == ["--rotation-gain-scale", expected_argument]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "1e308", "not-a-number"])
def test_cpp_probe_rejects_invalid_rotation_gain_scale(value):
    probe = Path(__file__).parents[1] / "build" / "compliant_control_surface_probe"
    if not probe.is_file():
        pytest.skip("C++ surface probe is not built")
    completed = subprocess.run(
        [str(probe), "--mode", "none", "--rotation-gain-scale", value],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "finite positive" in completed.stderr


def test_cpp_probe_rotation_gain_scale_is_optional_but_requires_a_value():
    probe = Path(__file__).parents[1] / "build" / "compliant_control_surface_probe"
    if not probe.is_file():
        pytest.skip("C++ surface probe is not built")
    frame = "1 0 0 0 1 0 0 0 1\n"
    default = subprocess.run(
        [str(probe), "--mode", "none"],
        input=frame,
        capture_output=True,
        text=True,
        check=False,
    )
    explicit = subprocess.run(
        [str(probe), "--mode", "none", "--rotation-gain-scale", "1"],
        input=frame,
        capture_output=True,
        text=True,
        check=False,
    )
    missing = subprocess.run(
        [str(probe), "--mode", "none", "--rotation-gain-scale"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert default.returncode == explicit.returncode == 0
    assert default.stdout == explicit.stdout == ""
    assert missing.returncode == 2
    assert "usage:" in missing.stderr


def test_verifier_refuses_existing_or_aliased_output_before_execution(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "new")
    with pytest.raises(FileExistsError):
        verify([Path("missing.npz")], Path("missing_probe"), existing)
    with pytest.raises(ValueError, match="symlink"):
        verify([Path("missing.npz")], Path("missing_probe"), alias)
