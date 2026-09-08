"""Evidence-tool format checks separate from the C++ numeric parity tests."""

from pathlib import Path

import pytest

from tools.verify_cpp_surface_replay import _mode, _parse_probe_output, verify


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


def test_verifier_refuses_existing_or_aliased_output_before_execution(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "new")
    with pytest.raises(FileExistsError):
        verify([Path("missing.npz")], Path("missing_probe"), existing)
    with pytest.raises(ValueError, match="symlink"):
        verify([Path("missing.npz")], Path("missing_probe"), alias)
