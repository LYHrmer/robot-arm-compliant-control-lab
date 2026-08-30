from pathlib import Path

import pytest

from compliant_control_lab.post_reveal_analysis import (
    generate_post_reveal_analysis,
    summarize_gate_results,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_CSV = REPOSITORY_ROOT / "results/franka_safety_blind/comparison.csv"


def test_published_failure_breakdown_matches_first_reveal() -> None:
    summaries = summarize_gate_results(PUBLISHED_CSV)

    assert len(summaries) == 8
    assert {summary.case_count for summary in summaries} == {48}
    residual = [summary for summary in summaries if summary.method.startswith("torque_residual")]
    assert [summary.pass_count for summary in residual] == [22, 25, 26, 24, 25]
    assert [summary.failure_counts["peak_force"] for summary in residual] == [24, 20, 20, 22, 21]
    assert [summary.failure_counts["tangent_rmse"] for summary in residual] == [4, 5, 4, 5, 5]
    assert all(summary.failure_counts.get("saturation", 0) == 0 for summary in residual)


def test_report_generation_keeps_analysis_outside_first_reveal(tmp_path: Path) -> None:
    plot_path = generate_post_reveal_analysis(
        PUBLISHED_CSV,
        tmp_path / "post_reveal",
        required_passes=44,
    )

    assert plot_path.is_file()
    summary = (plot_path.parent / "summary.md").read_text(encoding="utf-8")
    assert "not new blind evidence" in summary
    assert "residual 00 | 22/48" in summary
    assert "20–24 cases" in summary


def test_report_refuses_to_write_inside_first_reveal() -> None:
    with pytest.raises(ValueError, match="outside the first-reveal directory"):
        generate_post_reveal_analysis(
            PUBLISHED_CSV,
            PUBLISHED_CSV.parent,
            required_passes=44,
        )


def test_report_text_is_derived_from_the_supplied_csv(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    csv_path = input_dir / "small.csv"
    csv_path.write_text(
        "method,gate_pass,failed_checks\n"
        "fixed_hybrid,yes,\n"
        "fixed_hybrid,no,force_rmse\n"
        "torque_residual_run_00,yes,\n"
        'torque_residual_run_00,no,"peak_force;saturation"\n',
        encoding="utf-8",
    )

    plot_path = generate_post_reveal_analysis(
        csv_path,
        tmp_path / "analysis",
        required_passes=2,
    )

    summary = (plot_path.parent / "summary.md").read_text(encoding="utf-8")
    assert "Across 1 residual policies" in summary
    assert "peak force failed in 1–1 cases" in summary
    assert "combined saturation-failure count was 1" in summary
