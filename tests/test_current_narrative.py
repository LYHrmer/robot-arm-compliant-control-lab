"""Keep the current project narrative tied to its published evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def test_architecture_separates_current_control_learning_and_history() -> None:
    text = (ROOT / "docs/architecture.md").read_text(encoding="utf-8")
    current, history = text.split("## 历史 v0.5：20 维残差与首次揭盲", 1)
    assert "500 Hz" in current and "在线切向补偿" in current
    assert "预算默认固定 6 N" in current and "6–8 N 调度需显式启用" in current
    assert "surface_env_v1" in current and "49 维" in current and "16/4/4" in current
    assert "24 个公开 case 按物理任务组划分" in current
    assert "不是训练期间累计采样的 rollout episode 数" in current
    assert "BC/PPO 尚未跨种子一致超过解析前馈" in current
    assert "8 train cases" in history and "8 development cases" in history
    assert "20-D observation" in history and "48 scenario seeds" in history
    assert "44/48" in history and "`FAIL`" in history
    assert "六轴机械臂在独立项目开发" in current


def test_architecture_preserves_legacy_heading_text() -> None:
    text = (ROOT / "docs/architecture.md").read_text(encoding="utf-8")
    headings = {
        line.lstrip("#").strip()
        for line in text.splitlines()
        if line.startswith("#")
    }
    assert {
        "RL 改了哪里", "实时控制", "训练、冻结与揭盲", "模块接口与边界（seam）"
    } <= headings


def test_auxiliary_audit_is_not_described_as_full_control_replay() -> None:
    text = (ROOT / "docs/architecture.md").read_text(encoding="utf-8")
    rows = text.splitlines()
    auxiliary = next(row for row in rows if row.startswith("| 辅助负载通道审计 |"))
    complete = next(row for row in rows if row.startswith("| Python/C++ 完整表面控制回放 |"))
    assert "measured_budget_validation.py" in auxiliary
    assert "不重算完整名义控制器或七轴力矩" in auxiliary
    assert "verify_measured_budget_cpp.py" in complete
    assert "完整 wrench 与七轴力矩" in complete and "不重新积分动力学" in complete


def test_resume_tracking_numbers_match_public_paired_archive() -> None:
    text = (ROOT / "docs/tutorial/06_exercises_and_interview.md").read_text(encoding="utf-8")
    with (ROOT / "results/franka_online_compensation_regression/comparison.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    methods = {
        method: {
            int(row["case_index"]): float(row["tangent_rmse_mm"])
            for row in rows if row["method"] == method
        }
        for method in ("friction", "online")
    }
    assert methods["friction"].keys() == methods["online"].keys()
    assert len(methods["online"]) == 24
    assert all(methods["online"][case] < value for case, value in methods["friction"].items())
    for values in methods.values():
        assert f"{median(values.values()):.3f}" in text
    assert "公开 24-case 配对回归" in text
    assert "在线补偿的 24/24 改善是解析更新律的结果" in text


def test_cpp_replay_counts_and_bound_match_current_report() -> None:
    report = json.loads(
        (ROOT / "results/franka_measured_budget_cpp_replay/report.json").read_text()
    )
    assert report["trace_count"] == 28
    assert report["sample_count"] == 168_000
    assert report["demo_trace_count"] == 1
    errors = report["aggregate_results"]["cpp_vs_trace"]["max_abs_errors"]
    assert max(errors.values()) < 3.56e-15
    for relative in ("docs/architecture.md", "docs/tutorial/06_exercises_and_interview.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "28 条" in text and "168,000" in text and "3.56e-15" in text
        assert "重复演示" in text and "真机" in text


def test_exercises_label_scope_and_retain_measured_failure() -> None:
    text = (ROOT / "docs/tutorial/06_exercises_and_interview.md").read_text(encoding="utf-8")
    for label in ("已有实现，可复现", "新增练习", "尚未实现的扩展", "已有小规模实验，可复现"):
        assert label in text
    assert "labs/01_wrench_to_torque.md" in text and "labs/02_budget_drop.md" in text
    assert "高负载下短暂停顿后恢复" in text
    assert "BC/PPO 保持研究支线" in text
    report = json.loads(
        (ROOT / "results/franka_measured_budget_robustness/comparison.json").read_text()
    )
    checks = [
        row for row in report["comparisons"] if row["existing_preset_criteria_applicable"]
    ]
    failed = [row for row in checks if row["existing_preset_criteria"]["status"] == "FAIL"]
    assert len(report["runs"]) == 27 and len(checks) == 8 and len(failed) == 1
    assert "scale_0p8" in failed[0]["scenario_id"]
    assert not report["safety"]["eligible_for_default_change"]
    assert "7/8" in text and "×0.8" in text and "`FAIL`" in text
