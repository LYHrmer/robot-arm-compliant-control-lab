"""Keep the result tables and stated tradeoffs tied to the published summaries."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs/reversible_recovery.md").read_text()
PILOT = json.loads((ROOT / "results/franka_reversible_recovery_pilot/comparison.json").read_text())
TRANSFER = json.loads((ROOT / "results/franka_reversible_recovery_transfer/comparison.json").read_text())
METHOD = "reversible_hold_cap"
PRIMARY = [row for row in TRANSFER["comparisons"] if row["compared_method"] == METHOD]
PROFILES = {"clean": "无额外误差", "combined": "组合误差", "combined_scale_0p8": "组合误差＋辅助力幅值 ×0.8"}
YAWS = {-15: "−15°", 0: "0°", 15: "+15°"}


def table(header):
    lines = DOC.split(header, 1)[1].strip().splitlines()[1:]
    rows = []
    for line in lines:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def window(row, name="ramp", field="window_deltas"):
    return next(value for value in row[field] if value["name"] == name)


def test_nine_direction_error_rows_match_each_scenario():
    expected = []
    for yaw, label in YAWS.items():
        for profile, description in PROFILES.items():
            counts = [sum(r["status"] == "PASS" for r in PRIMARY if r["surface_yaw_deg"] == yaw
                          and r["error_profile"] == profile and r["scenario"] == scenario)
                      for scenario in ("falling", "constant_high")]
            expected.append([label, description, *(f"{count}/2" for count in counts)])
    assert table("| 表面方向 | 误差配置 | 降载 | 恒定高负载 |") == expected
    assert f'{sum(r["status"] == "PASS" for r in PRIMARY)}/{len(PRIMARY)}' in DOC


def test_high_load_table_contains_relative_not_absolute_errors():
    expected = []
    for seed in (11, 29):
        selected = {r["compared_method"]: window(r) for r in PILOT["comparisons"]
                    if r["scenario"] == "constant_high" and r["seed"] == seed}
        values = [selected[method][metric] for metric in ("tangent_rmse_mm", "tangent_velocity_rmse_mm_s")
                  for method in ("stationary_hold_cap", METHOD)]
        expected.append([str(seed), *(f"{value:+.6f}" for value in values)])
    assert table("| 种子 | 静止回退位置增量 mm | 可逆回退位置增量 mm | 静止回退速度增量 mm/s | 可逆回退速度增量 mm/s |") == expected


def test_four_failure_values_and_their_gate():
    failed = [row for row in PRIMARY if row["status"] == "FAIL"]
    assert len(failed) == 4 and all(r["failed_checks"] == ["ramp:velocity_cost"] for r in failed)
    expected = []
    for profile in ("combined", "combined_scale_0p8"):
        rows = [next(r for r in failed if r["error_profile"] == profile and r["seed"] == seed) for seed in (11, 29)]
        assert all(r["surface_yaw_deg"] == 15 and r["scenario"] == "falling" for r in rows)
        expected.append([PROFILES[profile], *(f'{window(r)["tangent_velocity_rmse_mm_s"]:+.6f}' for r in rows)])
    assert table("| +15° 降载误差配置 | seed 11 速度增量 mm/s | seed 29 速度增量 mm/s |") == expected


def test_falling_benefit_and_old_hold_cost_ranges():
    falling = [row for row in PRIMARY if row["scenario"] == "falling"]
    assert len(falling) == 18
    for name, label in (("early", "8–10 s 位置收益"), ("post", "8–12 s 为")):
        values = [window(r, name)["tangent_reduction_pct"] for r in falling]
        span = f"{min(values):.2f}%–{max(values):.2f}%"
        assert re.search(re.escape(label) + r"[^，。]*" + re.escape(span), DOC)
    indexed = {(r["surface_yaw_deg"], r["error_profile"], r["scenario"], r["seed"], r["method"]): r for r in TRANSFER["runs"]}
    for metric, unit in (("tangent_rmse_mm", "mm"), ("tangent_velocity_rmse_mm_s", "mm/s")):
        costs = [window(indexed[(*key[:-1], METHOD)], field="windows")[metric] - window(row, field="windows")[metric]
                 for key, row in indexed.items() if key[-1] == "hold_cap_tracking" and key[2] == "falling"]
        assert len(costs) == 18 and min(costs) > 0
        assert f"{min(costs):.6f}–{max(costs):.6f} {unit}" in DOC


def test_smallest_margin_among_passed_pairs_and_execution_commit():
    protocol = json.loads((ROOT / "results/franka_reversible_recovery_transfer/protocol.json").read_text())
    limit = protocol["acceptance"]["maximum_window_tangent_velocity_rmse_increase_mm_s"]
    margins = [(limit - w["tangent_velocity_rmse_mm_s"], row) for row in PRIMARY
               if row["status"] == "PASS" for w in row["window_deltas"]]
    margin, row = min(margins, key=lambda item: item[0])
    described = re.search(r"速度余量也仅有 ([0-9.]+) mm/s", DOC).group(1)
    assert described == f"{margin:.6f}"
    assert row["scenario"] == "falling"
    assert f'{YAWS[row["surface_yaw_deg"]]} {PROFILES[row["error_profile"]]}、降载 seed {row["seed"]} 的速度余量' in DOC
    commits = {json.loads((ROOT / f"results/franka_reversible_recovery_{scope}/manifest.json").read_text())["source_commit"]
               for scope in ("pilot", "transfer")}
    assert len(commits) == 1 and f"`{commits.pop()}`" in DOC
