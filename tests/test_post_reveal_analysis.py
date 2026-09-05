import csv
from pathlib import Path

import pytest

from compliant_control_lab.post_reveal_analysis import (
    compute_paired_effects,
    generate_post_reveal_analysis,
    summarize_gate_results,
    summarize_leave_one_gate_out,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_CSV = REPOSITORY_ROOT / "results/franka_safety_blind/comparison.csv"


def _published_rows() -> tuple[list[dict[str, str]], list[str]]:
    with PUBLISHED_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or ())


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_published_failure_breakdown_matches_first_reveal() -> None:
    summaries = summarize_gate_results(PUBLISHED_CSV)

    assert len(summaries) == 8
    assert {summary.case_count for summary in summaries} == {48}
    residual = [summary for summary in summaries if summary.method.startswith("torque_residual")]
    assert [summary.pass_count for summary in residual] == [22, 25, 26, 24, 25]
    assert [summary.failure_counts["peak_force"] for summary in residual] == [24, 20, 20, 22, 21]
    assert [summary.failure_counts["tangent_rmse"] for summary in residual] == [4, 5, 4, 5, 5]
    assert all(summary.failure_counts.get("saturation", 0) == 0 for summary in residual)


def test_published_paired_effects_match_known_casewise_differences() -> None:
    effects = compute_paired_effects(PUBLISHED_CSV)

    tangent = [effect for effect in effects if effect.metric == "tangent_rmse_mm"]
    assert [effect.win_count for effect in tangent] == [35, 35, 35, 35, 34]
    assert all(effect.tie_count == 0 for effect in tangent)
    assert [effect.loss_count for effect in tangent] == [13, 13, 13, 13, 14]
    assert tangent[0].median_delta == pytest.approx(-1.92515251456)
    assert tangent[0].ci_low == pytest.approx(-2.34297379443)
    assert tangent[0].ci_high == pytest.approx(-0.842154421224)

    first_force = effects[0]
    assert first_force.method == "torque_residual_run_00"
    assert first_force.metric == "force_rmse_n"
    assert first_force.median_delta == pytest.approx(0.110700170887)
    assert (first_force.win_count, first_force.tie_count, first_force.loss_count) == (7, 0, 41)


def test_published_leave_one_gate_out_counts_expose_peak_bottleneck() -> None:
    summaries = summarize_leave_one_gate_out(PUBLISHED_CSV)

    residual = [summary for summary in summaries if summary.method.startswith("torque_residual")]
    assert [summary.pass_count for summary in residual] == [22, 25, 26, 24, 25]
    assert [summary.pass_count_without["peak_force"] for summary in residual] == [
        41,
        40,
        41,
        40,
        40,
    ]
    assert [summary.pass_count_without["tangent_rmse"] for summary in residual] == [
        24,
        28,
        28,
        26,
        27,
    ]


def test_paired_effects_reject_duplicate_method_case_pair(tmp_path: Path) -> None:
    rows, fieldnames = _published_rows()
    duplicate = next(row for row in rows if row["method"] == "torque_residual_run_00")
    csv_path = tmp_path / "duplicate.csv"
    _write_rows(csv_path, [*rows, duplicate], fieldnames)

    with pytest.raises(ValueError, match="duplicate method/case pair"):
        compute_paired_effects(csv_path)


def test_paired_effects_reject_missing_residual_case(tmp_path: Path) -> None:
    rows, fieldnames = _published_rows()
    rows = [
        row
        for row in rows
        if not (row["method"] == "torque_residual_run_00" and row["case"] == "blind_00")
    ]
    csv_path = tmp_path / "missing.csv"
    _write_rows(csv_path, rows, fieldnames)

    with pytest.raises(ValueError, match="case set does not match baseline"):
        compute_paired_effects(csv_path)


def test_paired_effects_reject_non_finite_metric(tmp_path: Path) -> None:
    rows, fieldnames = _published_rows()
    target = next(row for row in rows if row["method"] == "torque_residual_run_00")
    target["force_rmse_n"] = "nan"
    csv_path = tmp_path / "non_finite.csv"
    _write_rows(csv_path, rows, fieldnames)

    with pytest.raises(ValueError, match="non-finite force_rmse_n"):
        compute_paired_effects(csv_path)


def test_analysis_is_independent_of_csv_row_order(tmp_path: Path) -> None:
    rows, fieldnames = _published_rows()
    csv_path = tmp_path / "reversed.csv"
    _write_rows(csv_path, list(reversed(rows)), fieldnames)

    assert summarize_gate_results(csv_path) == summarize_gate_results(PUBLISHED_CSV)
    assert compute_paired_effects(csv_path) == compute_paired_effects(PUBLISHED_CSV)
    assert summarize_leave_one_gate_out(csv_path) == summarize_leave_one_gate_out(PUBLISHED_CSV)


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
    assert "## Paired residual effect" in summary
    assert "negative values favor the residual policy" in summary
    assert "| residual 00 | tangent RMSE (mm) | -1.925 | -2.343 to -0.842 | 35/0/13 |" in summary
    assert "## Exploratory gate sensitivity (post-hoc)" in summary
    assert "| residual 00 | 22 | 22 | 22 | 41 | 24 | 22 |" in summary
    assert "40–41/48" in summary
    assert "still below the frozen 44/48 threshold" in summary


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
        "method,case,gate_pass,failed_checks,force_rmse_n,peak_force_n,tangent_rmse_mm\n"
        "safe_adaptive_hybrid,case_00,yes,,1.0,10.0,5.0\n"
        "safe_adaptive_hybrid,case_01,no,force_rmse,2.0,20.0,6.0\n"
        "torque_residual_run_00,case_00,yes,,0.9,11.0,4.0\n"
        'torque_residual_run_00,case_01,no,"peak_force;saturation",2.1,21.0,5.0\n',
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
