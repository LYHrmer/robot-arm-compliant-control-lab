import csv
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import compliant_control_lab.contact_event_analysis as event_analysis
from compliant_control_lab.contact_event_analysis import (
    EVENT_CSV_FIELDS,
    PEAK_CONTEXT_CSV_FIELDS,
    REPLAYED_METRICS,
    analyze_contact_events,
    generate_contact_event_report,
    motion_phase_at,
    replay_published_safe_adaptive_cases,
    summarize_contact_event_rows,
    write_contact_event_report,
)
from compliant_control_lab.franka_control import FrankaControllerTelemetrySnapshot
from compliant_control_lab.franka_simulation import FrankaTrialResult


def test_wiping_motion_phase_matches_the_frozen_open_boundary():
    assert motion_phase_at(1.2) == "pre_wiping"
    assert motion_phase_at(1.2 + 1.0e-12) == "wiping"


def _trial(
    time: np.ndarray,
    raw_force: np.ndarray,
    blends: tuple[float | None, ...],
) -> FrankaTrialResult:
    samples = len(time)
    linear_velocity = np.arange(samples * 3, dtype=float).reshape(samples, 3)
    target_linear_velocity = linear_velocity + 0.5
    commanded_wrench = np.arange(samples * 6, dtype=float).reshape(samples, 6)
    return FrankaTrialResult(
        controller="safe_adaptive_hybrid",
        scenario="synthetic",
        time=time,
        q=np.zeros((samples, 7)),
        position=np.zeros((samples, 3)),
        desired_position=np.zeros((samples, 3)),
        normal_force=raw_force.copy(),
        raw_normal_force=raw_force,
        desired_force=np.full(samples, 12.0),
        orientation_error_rad=np.zeros(samples),
        torque=np.zeros((samples, 7)),
        controller_time_us=np.ones(samples),
        saturated=np.zeros(samples, dtype=bool),
        linear_velocity=linear_velocity,
        target_linear_velocity=target_linear_velocity,
        commanded_wrench=commanded_wrench,
        minimum_torque_headroom_nm=np.arange(samples, dtype=float) + 10.0,
        controller_snapshots=tuple(
            FrankaControllerTelemetrySnapshot(
                contact_blend=blend,
                governed_normal_lead_m=None if blend is None else 0.001 * index,
                torque_projection_scale=None if blend is None else 1.0 - 0.1 * index,
            )
            for index, blend in enumerate(blends)
        ),
    )


def test_event_extraction_preserves_contact_and_motion_as_separate_phases():
    result = _trial(
        time=np.array([0.0, 1.0, 1.2, 1.4, 1.6]),
        raw_force=np.array([0.0, 2.0, 8.0, 5.0, 7.0]),
        blends=(0.0, 0.0, 0.1, 0.99, 1.0),
    )

    events = analyze_contact_events(result)

    assert events.first_raw_contact is not None
    assert events.first_raw_contact.time_s == 1.0
    assert events.first_raw_contact.contact_phase == "raw_contact"
    assert events.first_raw_contact.motion_phase == "pre_wiping"
    assert events.contact_confirm is not None
    assert events.contact_confirm.time_s == 1.2
    assert events.contact_confirm.contact_phase == "confirmed_transition"
    assert events.contact_confirm.motion_phase == "pre_wiping"
    assert events.contact_blend_99 is not None
    assert events.contact_blend_99.time_s == 1.4
    assert events.contact_blend_99.contact_phase == "blended_contact"

    assert events.global_peak.time_s == 1.2
    assert events.global_peak.raw_force_n == 8.0
    assert events.global_peak.motion_phase == "pre_wiping"
    assert events.global_peak.linear_velocity_m_s == (6.0, 7.0, 8.0)
    assert events.global_peak.target_linear_velocity_m_s == (6.5, 7.5, 8.5)
    assert events.global_peak.commanded_wrench == tuple(range(12, 18))
    assert events.global_peak.minimum_torque_headroom_nm == 12.0
    assert events.global_peak.controller_snapshot.contact_blend == 0.1
    assert events.peak_after_first_contact == events.global_peak


def test_event_extraction_reports_no_contact_without_inventing_landmarks():
    result = _trial(
        time=np.array([0.0, 1.2, 1.4]),
        raw_force=np.zeros(3),
        blends=(None, None, None),
    )

    events = analyze_contact_events(result)

    assert events.first_raw_contact is None
    assert events.contact_confirm is None
    assert events.contact_blend_99 is None
    assert events.peak_after_first_contact is None
    assert events.global_peak.time_s == 0.0
    assert events.global_peak.raw_force_n == 0.0
    assert events.global_peak.contact_phase == "pre_contact"


def test_event_extraction_rejects_misaligned_telemetry():
    result = _trial(
        time=np.array([0.0, 0.1, 0.2]),
        raw_force=np.array([0.0, 1.0, 2.0]),
        blends=(0.0, 0.1, 0.2),
    )
    result.commanded_wrench = np.zeros((2, 6))

    with pytest.raises(ValueError, match="commanded_wrench must have shape"):
        analyze_contact_events(result)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("contact_blend", -0.01),
        ("contact_blend", 1.01),
        ("torque_projection_scale", -0.01),
        ("torque_projection_scale", 1.01),
    ),
)
def test_event_extraction_rejects_invalid_snapshot_ratios(field_name, invalid_value):
    result = _trial(
        time=np.array([0.0, 0.1]),
        raw_force=np.array([0.0, 1.0]),
        blends=(0.0, 0.1),
    )
    snapshots = list(result.controller_snapshots)
    snapshots[1] = replace(snapshots[1], **{field_name: invalid_value})
    result.controller_snapshots = tuple(snapshots)

    with pytest.raises(ValueError, match=field_name):
        analyze_contact_events(result)


def test_contact_phase_uses_the_current_sample_after_contact_release():
    result = _trial(
        time=np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        raw_force=np.array([1.0, 2.0, 3.0, 0.0, 5.0]),
        blends=(0.0, 0.1, 1.0, 0.5, 0.0),
    )

    events = analyze_contact_events(result)

    assert events.contact_blend_99 is not None
    assert events.contact_blend_99.time_s == 0.2
    assert events.global_peak.time_s == 0.4
    assert events.global_peak.contact_phase == "raw_contact"


def test_post_reveal_runner_replays_and_verifies_one_synthetic_case(
    tmp_path, monkeypatch
):
    protocol_path = tmp_path / "protocol.json"
    result_dir = tmp_path / "frozen"
    result_dir.mkdir()
    protocol_path.write_text(
        '{"blind_contract": {"duration_s": 2.0}}', encoding="utf-8"
    )
    (result_dir / "reveal.json").write_text(
        '{"protocol_sha256": "sha", "beacon": {"randomness": "random"}}',
        encoding="utf-8",
    )
    frozen_metrics = {
        "force_rmse_n": 1.0,
        "peak_force_n": 40.0,
        "tangent_rmse_mm": 2.0,
        "orientation_rmse_deg": 3.0,
        "contact_ratio_pct": 100.0,
        "torque_rms_nm": 4.0,
        "saturation_pct": 0.0,
    }
    with (result_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "case",
                *REPLAYED_METRICS,
                "gate_pass",
                "failed_checks",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "safe_adaptive_hybrid",
                "case": "blind_00",
                **frozen_metrics,
                "gate_pass": "no",
                "failed_checks": "peak_force",
            }
        )
    trial = _trial(
        time=np.array([0.0, 0.2, 0.4]),
        raw_force=np.array([0.0, 2.0, 40.0]),
        blends=(0.0, 0.1, 1.0),
    )
    trial.metrics = lambda: {"scenario": "blind_00", **frozen_metrics}
    monkeypatch.setattr(
        event_analysis,
        "audit_published_results",
        lambda *_args: SimpleNamespace(case_count=1),
    )
    monkeypatch.setattr(event_analysis, "derive_blind_root", lambda *_args: b"root")
    monkeypatch.setattr(
        event_analysis,
        "sample_blind_scenarios",
        lambda *_args: ([SimpleNamespace(name="blind_00")], [11], [22]),
    )
    monkeypatch.setattr(event_analysis, "run_franka_trial", lambda *_args, **_kwargs: trial)

    rows = replay_published_safe_adaptive_cases(
        protocol_path=protocol_path,
        result_dir=result_dir,
        case_indices=(0,),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["analysis_identity"] == "post_reveal_replay:v0.5"
    assert row["case"] == "blind_00"
    assert row["frozen_peak_force_n"] == row["replay_peak_force_n"]
    assert row["replay_peak_force_n"] == 40.0
    assert row["verified_metric_max_abs_error"] <= 1.0e-12
    assert all(row[f"verified_{metric}_abs_error"] == 0.0 for metric in REPLAYED_METRICS)
    assert row["peak_gate_failed"] is True
    assert "peak_force" in row["frozen_failed_checks"]
    assert 0.0 < row["first_raw_contact_time_s"] <= row["global_peak_time_s"]
    assert row["global_peak_after_first_contact_s"] == (
        row["global_peak_time_s"] - row["first_raw_contact_time_s"]
    )
    assert row["contact_confirm_time_s"] is not None
    assert row["contact_blend_99_time_s"] is not None
    assert row["global_peak_contact_phase"] in {
        "raw_contact",
        "confirmed_transition",
        "blended_contact",
    }
    assert row["global_peak_motion_phase"] in {"pre_wiping", "wiping"}
    assert row["peak_minimum_torque_headroom_nm"] >= 0.0


@pytest.mark.parametrize("case_index", (True, 1.9, "2"))
def test_post_reveal_runner_rejects_non_integer_case_indices(case_index):
    with pytest.raises(TypeError, match="case indices must be integers"):
        replay_published_safe_adaptive_cases(
            protocol_path="results/franka_safety_preholdout/protocol.json",
            result_dir="results/franka_safety_blind",
            case_indices=(case_index,),
        )


def _report_row(
    case_index: int,
    *,
    peak_force: float,
    delay: float,
    contact_phase: str,
    motion_phase: str,
    peak_failure: bool,
    actual_normal_velocity: float = 0.01,
    target_normal_velocity: float = 0.0,
    commanded_normal_force: float = 10.0,
    minimum_torque_headroom: float = 6.0,
    torque_projection_scale: float = 1.0,
) -> dict:
    first_contact = 0.1
    row = dict.fromkeys(EVENT_CSV_FIELDS)
    row.update(
        {
            "analysis_identity": "post_reveal_replay:v0.5",
            "case_index": case_index,
            "case": f"blind_{case_index:02d}",
            "scenario_seed": 1000 + case_index,
            "simulation_seed": 2000 + case_index,
            "verified_metric_max_abs_error": 0.0,
            "frozen_peak_force_n": peak_force,
            "replay_peak_force_n": peak_force,
            "frozen_gate_pass": "no" if peak_failure else "yes",
            "frozen_failed_checks": "peak_force" if peak_failure else "",
            "peak_gate_failed": peak_failure,
            "first_raw_contact_time_s": first_contact,
            "contact_confirm_time_s": first_contact + 0.02,
            "contact_blend_99_time_s": first_contact + 0.04,
            "global_peak_time_s": first_contact + delay,
            "global_peak_after_first_contact_s": delay,
            "global_peak_force_n": peak_force,
            "global_peak_contact_phase": contact_phase,
            "global_peak_motion_phase": motion_phase,
            "post_contact_peak_time_s": first_contact + delay,
            "post_contact_peak_force_n": peak_force,
            "post_contact_peak_contact_phase": contact_phase,
            "post_contact_peak_motion_phase": motion_phase,
            "peak_minimum_torque_headroom_nm": minimum_torque_headroom,
            "peak_contact_blend": {
                "raw_contact": 0.0,
                "confirmed_transition": 0.5,
                "blended_contact": 1.0,
            }[contact_phase],
            "peak_governed_normal_lead_m": 0.001,
            "peak_torque_projection_scale": torque_projection_scale,
            "peak_linear_velocity_x_m_s": actual_normal_velocity,
            "peak_linear_velocity_y_m_s": 0.02,
            "peak_linear_velocity_z_m_s": 0.03,
            "peak_target_linear_velocity_x_m_s": target_normal_velocity,
            "peak_target_linear_velocity_y_m_s": 0.0,
            "peak_target_linear_velocity_z_m_s": 0.0,
            "peak_commanded_force_x_n": commanded_normal_force,
            "peak_commanded_force_y_n": 2.0,
            "peak_commanded_force_z_n": 3.0,
            "peak_commanded_torque_x_nm": 0.1,
            "peak_commanded_torque_y_nm": 0.2,
            "peak_commanded_torque_z_nm": 0.3,
        }
    )
    row.update({f"verified_{metric}_abs_error": 0.0 for metric in REPLAYED_METRICS})
    return row


def _four_report_rows() -> list[dict]:
    return [
        _report_row(
            0,
            peak_force=40.0,
            delay=0.2,
            contact_phase="raw_contact",
            motion_phase="pre_wiping",
            peak_failure=True,
        ),
        _report_row(
            1,
            peak_force=36.0,
            delay=1.0,
            contact_phase="confirmed_transition",
            motion_phase="pre_wiping",
            peak_failure=True,
        ),
        _report_row(
            2,
            peak_force=30.0,
            delay=1.0,
            contact_phase="blended_contact",
            motion_phase="pre_wiping",
            peak_failure=False,
        ),
        _report_row(
            3,
            peak_force=25.0,
            delay=1.4,
            contact_phase="blended_contact",
            motion_phase="wiping",
            peak_failure=False,
        ),
    ]


def test_summary_separates_all_cases_from_frozen_peak_failures():
    summary = summarize_contact_event_rows(
        reversed(_four_report_rows()),
        expected_case_count=4,
        expected_peak_failure_count=2,
    )

    assert summary.case_count == 4
    assert summary.peak_failure_count == 2
    assert summary.all_cases.delay.median_s == pytest.approx(1.0)
    assert summary.all_cases.delay.p25_s == pytest.approx(0.8)
    assert summary.all_cases.delay.p75_s == pytest.approx(1.1)
    assert summary.all_cases.delay.minimum_s == 0.2
    assert summary.all_cases.delay.maximum_s == 1.4
    assert summary.all_cases.early_count == 1
    assert summary.all_cases.late_count == 3
    assert summary.peak_failures.delay.median_s == pytest.approx(0.6)
    assert summary.peak_failures.early_count == 1
    assert summary.peak_failures.late_count == 1
    assert dict(summary.peak_failures.contact_phase_counts) == {
        "pre_contact": 0,
        "raw_contact": 1,
        "confirmed_transition": 1,
        "blended_contact": 0,
    }
    assert dict(summary.peak_failures.motion_phase_counts) == {
        "pre_wiping": 2,
        "wiping": 0,
    }
    assert dict(summary.metric_max_abs_errors) == {
        metric: 0.0 for metric in REPLAYED_METRICS
    }


def test_peak_context_is_derived_from_early_and_late_failure_samples():
    rows = [
        _report_row(
            0,
            peak_force=40.0,
            delay=0.1,
            contact_phase="raw_contact",
            motion_phase="pre_wiping",
            peak_failure=True,
            actual_normal_velocity=0.1,
            target_normal_velocity=-0.01,
            commanded_normal_force=8.0,
            minimum_torque_headroom=6.0,
            torque_projection_scale=0.9,
        ),
        _report_row(
            1,
            peak_force=39.0,
            delay=0.5,
            contact_phase="confirmed_transition",
            motion_phase="pre_wiping",
            peak_failure=True,
            actual_normal_velocity=0.3,
            target_normal_velocity=0.02,
            commanded_normal_force=12.0,
            minimum_torque_headroom=5.5,
            torque_projection_scale=0.8,
        ),
        _report_row(
            2,
            peak_force=38.0,
            delay=1.0,
            contact_phase="blended_contact",
            motion_phase="pre_wiping",
            peak_failure=True,
            actual_normal_velocity=0.01,
            target_normal_velocity=0.0,
            commanded_normal_force=10.0,
            minimum_torque_headroom=8.0,
            torque_projection_scale=1.0,
        ),
        _report_row(
            3,
            peak_force=37.0,
            delay=1.5,
            contact_phase="blended_contact",
            motion_phase="wiping",
            peak_failure=True,
            actual_normal_velocity=0.03,
            target_normal_velocity=-0.005,
            commanded_normal_force=14.0,
            minimum_torque_headroom=7.0,
            torque_projection_scale=0.95,
        ),
        _report_row(
            4,
            peak_force=30.0,
            delay=0.7,
            contact_phase="blended_contact",
            motion_phase="pre_wiping",
            peak_failure=False,
        ),
        _report_row(
            5,
            peak_force=29.0,
            delay=0.8,
            contact_phase="blended_contact",
            motion_phase="pre_wiping",
            peak_failure=False,
        ),
    ]

    summary = summarize_contact_event_rows(
        rows,
        expected_case_count=6,
        expected_peak_failure_count=4,
    )
    early = summary.early_peak_failure_context
    late = summary.late_peak_failure_context

    assert early.case_count == 2
    assert early.actual_normal_velocity_median_m_s == pytest.approx(0.2)
    assert early.actual_normal_velocity_minimum_m_s == 0.1
    assert early.actual_normal_velocity_maximum_m_s == 0.3
    assert early.target_normal_velocity_max_abs_m_s == 0.02
    assert early.commanded_normal_force_median_n == 10.0
    assert early.commanded_normal_force_minimum_n == 8.0
    assert early.commanded_normal_force_maximum_n == 12.0
    assert early.minimum_torque_headroom_minimum_nm == 5.5
    assert early.minimum_torque_headroom_median_nm == 5.75
    assert early.torque_projection_scale_minimum == 0.8
    assert late.case_count == 2
    assert late.actual_normal_velocity_median_m_s == pytest.approx(0.02)
    assert late.target_normal_velocity_max_abs_m_s == 0.005
    assert late.commanded_normal_force_median_n == 12.0
    assert late.minimum_torque_headroom_minimum_nm == 7.0
    assert late.minimum_torque_headroom_median_nm == 7.5
    assert late.torque_projection_scale_minimum == 0.95


def test_writer_emits_stable_csv_readable_summary_and_png(tmp_path):
    paths = write_contact_event_report(
        reversed(_four_report_rows()),
        tmp_path / "post_reveal",
        expected_case_count=4,
        expected_peak_failure_count=2,
        frozen_csv_path=tmp_path / "first_reveal" / "comparison.csv",
    )

    with paths.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert tuple(reader.fieldnames or ()) == EVENT_CSV_FIELDS
    assert [row["case"] for row in rows] == [
        "blind_00",
        "blind_01",
        "blind_02",
        "blind_03",
    ]
    with paths.peak_context_csv_path.open(newline="", encoding="utf-8") as handle:
        context_reader = csv.DictReader(handle)
        contexts = list(context_reader)
        assert tuple(context_reader.fieldnames or ()) == PEAK_CONTEXT_CSV_FIELDS
    assert [row["cohort"] for row in contexts] == [
        "early (<= 0.5 s)",
        "late (>= 1.0 s)",
    ]
    assert [row["n"] for row in contexts] == ["1", "1"]
    summary_text = paths.summary_path.read_text(encoding="utf-8")
    assert "post-reveal descriptive replay" in summary_text
    assert "does not alter the frozen first-reveal result" in summary_text
    assert "failed in 2/4" in summary_text
    assert "frozen peak-force failures | 2 | 0.600 [0.400, 0.800]" in summary_text
    assert "## Peak-sample context" in summary_text
    assert "they are not evidence that any listed variable caused the peak" in summary_text
    assert "Actual world-x velocity" in summary_text
    assert "controller approach axis" in summary_text
    assert "strictly synchronized physical measurements" in summary_text
    assert all(f"`{metric}`" in summary_text for metric in REPLAYED_METRICS)
    assert paths.plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda rows: rows[0].pop("case"), "missing fields"),
        (lambda rows: rows[1].update(case=rows[0]["case"]), "duplicate case"),
        (
            lambda rows: rows[0].update(global_peak_force_n=float("nan")),
            "global_peak_force_n must be a finite number",
        ),
        (
            lambda rows: rows[0].update(global_peak_after_first_contact_s=0.3),
            "peak delay is inconsistent",
        ),
    ),
)
def test_summary_rejects_malformed_rows(mutate, message):
    rows = _four_report_rows()
    mutate(rows)

    with pytest.raises(ValueError, match=message):
        summarize_contact_event_rows(
            rows,
            expected_case_count=4,
            expected_peak_failure_count=2,
        )


@pytest.mark.parametrize("metric", REPLAYED_METRICS)
def test_writer_rejects_claimed_verified_metric_drift_before_writing(tmp_path, metric):
    rows = _four_report_rows()
    rows[0][f"verified_{metric}_abs_error"] = 1.0
    rows[0]["verified_metric_max_abs_error"] = 1.0
    if metric == "peak_force_n":
        rows[0]["replay_peak_force_n"] += 1.0
        rows[0]["global_peak_force_n"] += 1.0
        rows[0]["post_contact_peak_force_n"] += 1.0
    output = tmp_path / "post_reveal"

    with pytest.raises(ValueError, match="absolute tolerance"):
        write_contact_event_report(
            rows, output, expected_case_count=4, expected_peak_failure_count=2
        )

    assert not output.exists()


@pytest.mark.parametrize("error", (1.0e-15, 1.0e-12))
def test_non_peak_error_within_absolute_tolerance_is_accepted(error):
    rows = _four_report_rows()
    rows[0]["verified_force_rmse_n_abs_error"] = error
    rows[0]["verified_metric_max_abs_error"] = error

    summary = summarize_contact_event_rows(
        rows, expected_case_count=4, expected_peak_failure_count=2
    )

    assert dict(summary.metric_max_abs_errors)["force_rmse_n"] == error


def test_peak_error_within_absolute_tolerance_is_accepted():
    rows = _four_report_rows()
    replay_peak = rows[0]["frozen_peak_force_n"] + 1.0e-13
    error = abs(replay_peak - rows[0]["frozen_peak_force_n"])
    rows[0].update(
        replay_peak_force_n=replay_peak,
        global_peak_force_n=replay_peak,
        post_contact_peak_force_n=replay_peak,
        verified_peak_force_n_abs_error=error,
        verified_metric_max_abs_error=error,
    )

    summary = summarize_contact_event_rows(
        rows, expected_case_count=4, expected_peak_failure_count=2
    )

    assert dict(summary.metric_max_abs_errors)["peak_force_n"] == error


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("global_peak_motion_phase", "wiping", "motion phase disagrees"),
        ("global_peak_contact_phase", "blended_contact", "contact phase disagrees"),
        ("post_contact_peak_motion_phase", "wiping", "peak phases disagree"),
        ("post_contact_peak_time_s", 1.4, "post-contact peak disagrees"),
    ),
)
def test_report_rejects_phase_and_peak_inconsistency(field, value, message):
    rows = _four_report_rows()
    rows[0][field] = value

    with pytest.raises(ValueError, match=message):
        summarize_contact_event_rows(
            rows, expected_case_count=4, expected_peak_failure_count=2
        )


@pytest.mark.parametrize("destination", ("source", "nested", "symlink"))
def test_writer_protects_explicit_frozen_source_and_preserves_sentinel(tmp_path, destination):
    frozen = tmp_path / "first_reveal"
    frozen.mkdir()
    sentinel = frozen / "summary.md"
    sentinel.write_text("frozen first-reveal summary", encoding="utf-8")
    if destination == "source":
        output = frozen
    elif destination == "nested":
        output = frozen / "derived"
    else:
        output = tmp_path / "post_reveal"
        output.mkdir()
        (output / "summary.md").symlink_to(sentinel)

    with pytest.raises(ValueError, match="outside the first-reveal directory"):
        write_contact_event_report(
            _four_report_rows(), output,
            expected_case_count=4, expected_peak_failure_count=2,
            frozen_csv_path=frozen / "comparison.csv",
        )

    assert sentinel.read_text(encoding="utf-8") == "frozen first-reveal summary"
    assert not (output / "safe_adaptive_contact_events.csv").exists()


def test_writer_protects_default_frozen_directory_without_source_argument(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frozen = tmp_path / "results" / "franka_safety_blind"
    frozen.mkdir(parents=True)
    sentinel = frozen / "summary.md"
    sentinel.write_text("frozen first-reveal summary", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the first-reveal directory"):
        write_contact_event_report(
            _four_report_rows(), frozen,
            expected_case_count=4, expected_peak_failure_count=2,
        )

    assert sentinel.read_text(encoding="utf-8") == "frozen first-reveal summary"


def _published_report_rows() -> list[dict]:
    rows = [
        _report_row(
            case_index,
            peak_force=40.0 if case_index < 18 else 30.0,
            delay=(
                0.1 + 0.05 * case_index
                if case_index < 7
                else 1.0 + 0.05 * (case_index - 7)
                if case_index < 18
                else 1.5 + 0.01 * case_index
            ),
            contact_phase="raw_contact" if case_index < 7 else "blended_contact",
            motion_phase="wiping" if case_index >= 24 else "pre_wiping",
            peak_failure=case_index < 18,
        )
        for case_index in range(48)
    ]
    for row in rows:
        phase = motion_phase_at(row["global_peak_time_s"])
        row["global_peak_motion_phase"] = phase
        row["post_contact_peak_motion_phase"] = phase
    return rows


def test_generator_replays_the_complete_set_without_a_subset(tmp_path, monkeypatch):
    rows = _published_report_rows()
    replay_calls = []

    def fake_replay(protocol_path, result_dir, *, case_indices=None):
        replay_calls.append((protocol_path, result_dir, case_indices))
        return rows

    monkeypatch.setattr(
        event_analysis, "replay_published_safe_adaptive_cases", fake_replay
    )
    output_dir = tmp_path / "post_reveal"
    paths = generate_contact_event_report(
        tmp_path / "protocol.json",
        tmp_path / "first_reveal",
        output_dir,
    )

    assert replay_calls == [
        (tmp_path / "protocol.json", tmp_path / "first_reveal", None)
    ]
    assert paths.csv_path.exists()
    assert paths.peak_context_csv_path.exists()
    summary_text = paths.summary_path.read_text(encoding="utf-8")
    assert "failed in 18/48" in summary_text
    assert "Early <= 0.5 s (n=7)" in summary_text
    assert "Late >= 1.0 s (n=11)" in summary_text
    assert "| 0.0000 |" in summary_text


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda rows: rows[0].update(
                global_peak_time_s=0.85,
                global_peak_after_first_contact_s=0.75,
                post_contact_peak_time_s=0.85,
            ),
            "exactly 7 early",
        ),
        (
            lambda rows: rows[0].update(peak_target_linear_velocity_x_m_s=0.001),
            "target world-x velocity",
        ),
        (
            lambda rows: rows[0].update(peak_torque_projection_scale=0.99),
            "torque-projection scale",
        ),
        (
            lambda rows: rows[0].update(peak_minimum_torque_headroom_nm=5.0),
            "torque headroom fell below 5.08",
        ),
    ),
)
def test_generator_locks_the_published_peak_context(
    tmp_path, monkeypatch, mutate, message
):
    rows = _published_report_rows()
    mutate(rows)
    monkeypatch.setattr(
        event_analysis,
        "replay_published_safe_adaptive_cases",
        lambda *_args, **_kwargs: rows,
    )

    with pytest.raises(ValueError, match=message):
        generate_contact_event_report(
            tmp_path / "protocol.json",
            tmp_path / "first_reveal",
            tmp_path / "post_reveal",
        )


def test_generator_refuses_to_write_inside_the_first_reveal_directory(tmp_path):
    result_dir = tmp_path / "first_reveal"

    with pytest.raises(ValueError, match="outside the first-reveal directory"):
        generate_contact_event_report(
            tmp_path / "protocol.json",
            result_dir,
            result_dir / "derived",
        )
