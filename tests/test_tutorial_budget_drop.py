"""Worked answers use fixed published samples, not a second controller rollout."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.tutorials import budget_drop as lab


@pytest.mark.parametrize(
    "event,index,rejected,previous_norm,current_norm,first_below",
    [
        ("missing", 4000, 150, 6.21074781, 6.17081729, 4005),
        ("stale", 4009, 141, 6.22465441, 6.18471479, 4014),
    ],
)
def test_published_fault_answers(event, index, rejected, previous_norm, current_norm, first_below):
    result = lab.analyze(lab.load_trace(event), event)
    assert result["index"] == index
    assert result["rejected_packets"] == rejected
    assert result["budget_n"] == 6.0
    assert result["load_n"] == 0.0
    assert result["mu_before"] == result["mu_after"]
    assert result["mu_after"] > 0.45
    assert result["magnitude_before_n"] == pytest.approx(previous_norm, abs=1e-8)
    assert result["magnitude_n"] == pytest.approx(current_norm, abs=1e-8)
    assert result["step_n"] == pytest.approx(0.04, abs=1e-12)
    assert result["first_below_index"] == first_below
    assert result["recovery_s"] == pytest.approx(8.3)
    assert result["validation"]["validated_cycles"] == 6000
    assert result["validation"]["above_scheduled_budget_cycles"] == 5
    assert result["validation"]["coefficient_transition_checked"]
    assert result["validation"]["readiness_transition_checked"]


def test_first_missing_vector_matches_rounded_hand_calculation():
    result = lab.analyze(lab.load_trace())
    np.testing.assert_allclose(
        result["force_n"], [0.0, 5.279640, -3.194431], rtol=0, atol=5e-7
    )
    # The norm decrease is slightly less than the norm of the vector difference.
    assert result["magnitude_before_n"] - result["magnitude_n"] < result["step_n"]


def test_documented_unlimited_vector_change_matches_trace():
    result = lab.analyze(lab.load_trace())
    delta_norm = np.linalg.norm(result["desired_force_n"] - result["previous_force_n"])
    document = (lab.ROOT / "docs/tutorial/labs/02_budget_drop.md").read_text()
    assert f"它的长度约为 {delta_norm:.6f} N" in document


@pytest.mark.parametrize("event", ["missing", "stale"])
def test_tamper_check_does_not_change_original_arrays(event):
    trace = lab.load_trace(event)
    original = trace["controller_coefficient_after_compute"].copy()
    lab.check_tamper(trace, 4000 if event == "missing" else 4009)
    np.testing.assert_array_equal(trace["controller_coefficient_after_compute"], original)


def test_missing_state_is_not_silently_replaced():
    trace = lab.load_trace()
    del trace["measured_position"]
    with pytest.raises(ValueError, match="measured_position"):
        lab.analyze(trace)


def test_unknown_event_rejected():
    with pytest.raises(ValueError, match="event"):
        lab.load_trace("unknown")


def test_documented_stdout_is_the_actual_command_output():
    root = Path(__file__).resolve().parents[1]
    document = (root / "docs/tutorial/labs/02_budget_drop.md").read_text()
    expected = re.search(r"```text\n(.*?)\n```", document, re.DOTALL).group(1)
    completed = subprocess.run(
        [sys.executable, "-m", "tools.tutorials.budget_drop"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert completed.stdout.strip() == expected


def test_stale_cli_and_tamper_answer():
    completed = subprocess.run(
        [sys.executable, "-m", "tools.tutorials.budget_drop", "--event", "stale", "--check-tamper"],
        cwd=lab.ROOT, capture_output=True, text=True, check=True,
    )
    assert "first rejection: k=4009, t=8.018 s" in completed.stdout
    assert "REJECTED (expected)" in completed.stdout
