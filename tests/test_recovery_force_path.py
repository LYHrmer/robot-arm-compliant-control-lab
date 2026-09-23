"""Independent force arithmetic and read-only archived recovery diagnostics."""

import json
import sys

import numpy as np
import pytest

from tools.tutorials import recovery_force_path as lab


@pytest.fixture
def hand_trace():
    # Tangential direction is 3 / sqrt(3**2 + 4**2) = 0.6;
    # the independent per-cycle slew allowance is 4 * 0.5 = 2 N.
    trace = {
        "time": np.arange(6) * 0.5,
        "dt": np.array(0.5),
        "controller_frame_rotation": np.eye(3),
        "target_linear_velocity": np.array([[100.0, 3.0, 0.0]] * 6),
        "controller_coefficient_before_compute": np.array([0.2, 2, 2, 2, 2, 2]),
        "load_budget_applied_n": np.array([20.0, 3, 20, 20, 20, 20]),
        "diagnostic_corrected_force_n": np.full(6, 5.0),
        "contact_blend": np.array([0.5, 1, 1, 1, 1, 1]),
        "diagnostic_compensation_active": np.array([True, True, True, False, True, True]),
        "load_compensation_force_local": np.array([
            [0.0, 0.3, 0.0], [0.0, 1.8, 0.0], [0.0, 3.8, 0.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0],
        ]),
        "diagnostic_amplitude_capped": np.array([False, True, False, False, False, False]),
        "diagnostic_slew_limited": np.array([False, False, True, False, False, True]),
    }
    trace["target_linear_velocity"][4] = 0
    return trace, {"velocity_scale": 4.0, "force_slew_rate": 4.0}


def test_hand_calculated_uncapped_capped_slew_inactive_reset_and_zero_velocity(hand_trace):
    trace, parameters = hand_trace
    saved = {key: value.copy() for key, value in trace.items()}
    result = lab.force_path(trace, parameters)
    np.testing.assert_allclose(result["raw_amplitude_n"], [1, 10, 10, 10, 10, 10])
    np.testing.assert_allclose(result["capped_amplitude_n"], [1, 3, 10, 10, 10, 10])
    np.testing.assert_allclose(result["desired_n"], [
        [0, 0.3, 0], [0, 1.8, 0], [0, 6, 0], [0, 6, 0], [0, 0, 0], [0, 6, 0],
    ])
    np.testing.assert_allclose(result["output_n"], trace["load_compensation_force_local"])
    np.testing.assert_array_equal(result["cap_active"], trace["diagnostic_amplitude_capped"])
    np.testing.assert_array_equal(result["slew_active"], trace["diagnostic_slew_limited"])
    for key, value in saved.items():
        np.testing.assert_array_equal(trace[key], value)


@pytest.mark.parametrize("override", ["coefficient", "budget"])
def test_one_step_override_uses_recorded_previous_output_not_recursive_rollout(hand_trace, override):
    trace, parameters = hand_trace
    for key, value in trace.items():
        if value.shape and value.shape[0] == 6:
            trace[key] = value[:3].copy()
    trace["controller_coefficient_before_compute"][:] = 1
    trace["diagnostic_corrected_force_n"][:] = 10
    trace["load_budget_applied_n"][:] = 100
    trace["contact_blend"][:] = 1
    trace["load_compensation_force_local"][:, 1] = [2, 4, 6]
    result = lab.force_path(trace, parameters, **{override: np.zeros(3)})
    np.testing.assert_array_equal(result["desired_n"], np.zeros((3, 3)))
    # All three desired values are zero, but the last recorded previous output
    # is 4 N. One 2 N slew step leaves 2 N; a recursive rollout would leave zero.
    np.testing.assert_array_equal(result["output_n"], [[0, 0, 0], [0, 0, 0], [0, 2, 0]])
    np.testing.assert_array_equal(result["slew_active"], [False, False, True])
    np.testing.assert_array_equal(trace["load_compensation_force_local"][:, 1], [2, 4, 6])


@pytest.mark.parametrize("override", ["coefficient", "budget"])
@pytest.mark.parametrize("value", [
    1.0, np.ones(5), np.ones((6, 1)),
    [1, 1, 1, 1, 1, np.nan], [1, 1, 1, 1, 1, np.inf], [1, 1, 1, 1, 1, -0.01],
], ids=["scalar", "short", "matrix", "nan", "infinite", "negative"])
def test_invalid_override_rejected(hand_trace, override, value):
    trace, parameters = hand_trace
    with pytest.raises(ValueError, match="finite nonnegative cycle arrays"):
        lab.force_path(trace, parameters, **{override: value})


@pytest.mark.parametrize("field", [
    "time", "dt", "controller_frame_rotation", "target_linear_velocity",
])
def test_unmatched_pair_rejected_before_analysis(hand_trace, field):
    trace, parameters = hand_trace
    changed = {key: value.copy() for key, value in trace.items()}
    changed[field].flat[0] += 0.01
    with pytest.raises(ValueError, match=f"unmatched paired input: {field}"):
        lab.analyze_pair(trace, changed, parameters, 0)


@pytest.mark.parametrize("field,message", [
    ("load_compensation_force_local", "force reconstruction differs"),
    ("diagnostic_amplitude_capped", "flag reconstruction differs: cap_active"),
    ("diagnostic_slew_limited", "flag reconstruction differs: slew_active"),
])
def test_inconsistent_reconstructed_output_or_flags_rejected(hand_trace, field, message):
    trace, parameters = hand_trace
    changed = {key: value.copy() for key, value in trace.items()}
    if changed[field].dtype == bool:
        changed[field][0] = not changed[field][0]
    else:
        changed[field][0, 1] += 0.01
    with pytest.raises(ValueError, match=message):
        lab.analyze_pair(trace, changed, parameters, 0)


@pytest.fixture(scope="module")
def archived_report():
    protocol = lab.study.protocol_document()
    archives = (lab.ARCHIVE, lab.study.ROOT / protocol["reference"]["directory"])

    def archive_state():
        return {path: (path.stat().st_size, path.stat().st_mtime_ns)
                for archive in archives for path in archive.rglob("*") if path.is_file()}

    def forbidden(*_args, **_kwargs):
        pytest.fail("force-path diagnosis must not run physics or save traces")

    before = archive_state()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(lab.original.runner, "_run_loop", forbidden)
        monkeypatch.setattr(lab.study, "ScheduledSurfaceSimulator", forbidden)
        monkeypatch.setattr(lab.original.previous, "_save_trace", forbidden)
        result = lab.diagnose()
    assert archive_state() == before
    return result


def test_six_real_pairs_preserve_the_original_failed_gates(archived_report):
    report = archived_report
    assert report["validated_traces"] == 12
    assert report["new_simulations"] == 0
    assert report["default_changed"] is False
    assert report["windows_are_post_hoc"] is True
    assert report["archived_comparison_status"] == "FAIL"
    assert (report["archived_passing_pairs"], report["archived_total_pairs"]) == (34, 36)
    assert "not closed-loop evidence" in report["analysis"]
    assert len(report["pairs"]) == 6
    assert {(pair["surface_yaw_deg"], pair["seed"]) for pair in report["pairs"]} == {
        (yaw, seed) for yaw in (-15, 0, 15) for seed in (11, 29)
    }
    published = json.loads((lab.ARCHIVE / "comparison.json").read_text())
    comparisons = published["comparisons"]
    assert len(comparisons) == 36
    assert sum(pair["status"] == "PASS" for pair in comparisons) == 34
    failures = [pair for pair in comparisons if pair["status"] == "FAIL"]
    assert {(pair["surface_yaw_deg"], pair["error_profile"], pair["scenario"], pair["seed"])
            for pair in failures} == {(-15, "clean", "constant_high", seed) for seed in (11, 29)}
    limits = lab.study.protocol_document()["acceptance"]
    for pair in report["pairs"]:
        expected_lead = 0.004 if pair["seed"] == 11 else 0.006
        assert pair["baseline_first_ready_s"] - pair["candidate_first_ready_s"] == pytest.approx(
            expected_lead, abs=1e-12
        )
        assert pair["hold_output_equal"] and pair["hold_budget_equal"]
        assert pair["hold_max_position_difference_m"] < 2e-9
        assert pair["candidate_first_low_speed_release_s"] == 5.308
        assert 5.406 <= pair["candidate_last_low_speed_release_s"] <= 5.428
        assert 46 <= pair["candidate_low_speed_release_cycles"] <= 59
        if pair["surface_yaw_deg"] == -15:
            assert pair["baseline_first_coefficient_increase_s"] == (
                7.610 if pair["seed"] == 11 else 7.556
            )
            assert pair["candidate_first_coefficient_increase_s"] == 7.438
            assert pair["ramp_position_delta_mm"] > limits["maximum_other_window_tangent_increase_mm"]
            assert pair["ramp_velocity_delta_mm_s"] > (
                limits["maximum_window_tangent_velocity_rmse_increase_mm_s"]
            )


def test_real_sensitivities_keep_slew_and_budget_window_distinctions(archived_report):
    for pair in archived_report["pairs"]:
        early, late = pair["windows"]
        assert (early["start_s"], early["end_s"], early["samples"]) == (7.35, 7.65, 150)
        assert (late["start_s"], late["end_s"], late["samples"]) == (7.5, 8.0, 250)
        assert late["baseline_slew_cycles"] == late["candidate_slew_cycles"] == 0
        for row in (early, late):
            gap = row["observed_request_gap_mean_n"]
            change = row["mu_only_request_change_mean_n"]
            remaining = row["mu_only_remaining_gap_mean_n"]
            assert gap < remaining < 0
            assert remaining == pytest.approx(gap + change, abs=1e-12)
            assert row["mu_only_slew_cycles"] > row["candidate_slew_cycles"]
            assert row["mu_only_desired_change_mean_n"] > change > 0
            assert abs(row["mu_only_desired_remaining_gap_mean_n"]) < 0.004
        if pair["surface_yaw_deg"] == -15:
            assert early["candidate_cap_cycles"] == 0
            assert early["budget_only_request_change_mean_n"] == 0
            assert -0.121 < early["observed_request_gap_mean_n"] < -0.119
            assert 0.031 < early["mu_only_request_change_mean_n"] < 0.032
            # The later window contains capped cycles: budget-only is small,
            # but genuinely nonzero, so it cannot be called ineffective there.
            assert 0.0013 < late["budget_only_request_change_mean_n"] < 0.0016


def test_json_cli_serializes_the_real_report_without_creating_files(
    archived_report, monkeypatch, capsys, tmp_path,
):
    monkeypatch.setattr(lab, "diagnose", lambda: archived_report)
    monkeypatch.setattr(sys, "argv", ["recovery_force_path", "--json"])
    monkeypatch.chdir(tmp_path)
    lab.main()
    assert json.loads(capsys.readouterr().out) == archived_report
    assert list(tmp_path.iterdir()) == []
