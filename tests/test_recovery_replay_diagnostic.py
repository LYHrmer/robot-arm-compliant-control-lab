"""The failure diagnostic exposes exact differences, not an alternate pass gate."""

from pathlib import Path

import pytest

from tools.ci.diagnose_recovery_replay import differences


@pytest.mark.parametrize("saved,rebuilt", [(0.0, -0.0), (1, 1.0), (False, 0), (1.0, 1.0 + 2e-16)])
def test_exact_numeric_and_type_differences_are_reported(saved, rebuilt):
    rows = list(differences({"runs": [saved]}, {"runs": [rebuilt]}))
    assert len(rows) == 1 and rows[0]["path"] == "root/runs/0"
    assert rows[0]["saved_type"] == type(saved).__name__
    assert rows[0]["rebuilt_type"] == type(rebuilt).__name__


def test_structure_differences_and_unchanged_values_are_reported():
    rows = list(differences({"a": [1], "b": None}, {"a": [1, 2], "c": None}))
    assert [(row["path"], row["kind"]) for row in rows] == [
        ("root/a", "length"), ("root/b", "missing_key"), ("root/c", "missing_key"),
    ]
    assert list(differences({"a": [1, False, None]}, {"a": [1, False, None]})) == []


def test_nonfinite_values_do_not_turn_into_a_success():
    with pytest.raises(ValueError):
        list(differences(float("nan"), float("nan")))


def test_ci_cpu_probe_is_failure_only_and_does_not_change_test_environment():
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/tests.yml").read_text()
    name = "      - name: Probe recovery replay without NumPy AVX512\n        if: failure()\n"
    assert workflow.count(name) == 2
    assert "NPY_DISABLE_CPU_FEATURES" not in workflow.split("jobs:", 1)[0]
    assert "continue-on-error" not in workflow
