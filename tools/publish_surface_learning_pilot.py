"""Publish compact, hash-checked BC/PPO evidence without claiming to include every trace."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_experiment import _sha256
from compliant_control_lab.surface_policy import friction_teacher_action
from compliant_control_lab.surface_policy_artifact import load_policy_artifact, policy_contract
from compliant_control_lab.surface_readiness_benchmark import METRICS, audit_trace
from compliant_control_lab.surface_transitions import load_transition_batch
from tools.surface_learning_pilot import (
    _read,
    _training_manifest,
    _verified_manifest,
    _write_new,
    select_ppo_checkpoint,
)
from tools.surface_mlp_actor import actor_from_artifact


def audit_public_pilot(directory):
    """Check the distributed subset and selection records; not a raw-trajectory re-audit."""
    directory = Path(directory)
    manifest = _verified_manifest(directory)
    actual_files = {
        str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()
    }
    if actual_files != set(manifest["artifact_sha256"]) | {"manifest.json", "COMPLETE"}:
        raise ValueError("public archive contains missing or unlisted files")
    plan = _read(directory / "plan.json")
    if (
        manifest["plan_sha256"] != _sha256(directory / "plan.json")
        or (directory / "PLAN_SHA256").read_text().strip() != manifest["plan_sha256"]
    ):
        raise ValueError("published plan SHA differs")
    validation_ids = [case["case_id"] for case in plan["cases"] if case["split"] == "validation"]
    for stage in ("bc", "ppo"):
        summary = _read(directory / f"{stage}_summary.json")
        selection = directory / f"{stage}_selection.json"
        record = _read(selection)
        if (
            summary["plan_sha256"] != manifest["plan_sha256"]
            or summary["selection_sha256"] != _sha256(selection)
            or record != {key: summary[key] for key in ("plan_sha256", "selections", "validation")}
        ):
            raise ValueError("stage/selection identity differs")
        if [item["seed"] for item in summary["selections"]] != plan["seeds"]:
            raise ValueError("published cohort omits a seed")
        for split in ("validation", "development_test"):
            cohort = summary[split]
            if [item["seed"] for item in cohort] != plan["seeds"]:
                raise ValueError("published evaluation cohort omits a seed")
            expected_ids = sorted(
                case["case_id"] for case in plan["cases"] if case["split"] == split
            )
            for item in cohort:
                reports = (
                    [value["report"] for value in item["checkpoints"]]
                    if stage == "ppo" and split == "validation"
                    else [item["report"]]
                )
                if (
                    stage == "ppo"
                    and split == "validation"
                    and [value["episode"] for value in item["checkpoints"]]
                    != plan["ppo"]["checkpoint_episodes"]
                ):
                    raise ValueError("published validation omits a checkpoint")
                for report in reports:
                    if (
                        report["evaluation_split"] != split
                        or report["evaluation_case_ids"] != expected_ids
                        or report["expected_case_count"] != len(expected_ids)
                        or sorted(row["case_id"] for row in report["runs"]) != expected_ids
                        or any(row["split"] != split for row in report["runs"])
                    ):
                        raise ValueError("published evaluation has incomplete or different cases")
        for item in summary["selections"]:
            candidate = directory / item["candidate"]
            if (
                candidate.is_symlink()
                or not candidate.resolve().is_relative_to(directory.resolve())
                or _sha256(candidate) != item["candidate_sha256"]
            ):
                raise ValueError("selected candidate identity differs")
        if stage == "ppo":
            for selected, validated in zip(summary["selections"], summary["validation"]):
                recomputed = select_ppo_checkpoint(
                    validated["checkpoints"],
                    expected_case_ids=validation_ids,
                    thresholds=plan["thresholds"],
                )
                if selected["seed"] != validated["seed"] or any(
                    selected[key] != recomputed[key] for key in ("episode", "eligible")
                ):
                    raise ValueError("PPO selection differs from frozen validation ordering")
    representatives = _audit_representatives(directory, plan)
    return {
        "hashes_match": True,
        "selection_recomputed": True,
        "distributed_file_count": len(manifest["artifact_sha256"]),
        "all_physics_traces_distributed": False,
        "representative_physics_traces_recomputed": len(representatives),
        "new_holdout": False,
    }


def _audit_representatives(directory, plan):
    index = _read(directory / "representative_traces.json")
    expected_case = min(
        case["case_id"] for case in plan["cases"] if case["split"] == "development_test"
    )
    if [record["stage"] for record in index["records"]] != ["bc", "ppo"]:
        raise ValueError("representative trace cohort is incomplete")
    results = []
    for record in index["records"]:
        if record["seed"] != plan["seeds"][0] or record["case_id"] != expected_case:
            raise ValueError("representative selection rule differs")
        path = directory / record["file"]
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("unsafe representative path")
        source = _read(path.parent / "source_manifest.json")
        if _sha256(path) != source["artifact_sha256"][path.name]:
            raise ValueError("representative differs from original trace hash")
        case = next(case for case in plan["cases"] if case["case_id"] == expected_case)
        expected = next(
            row
            for row in _read(path.parent / "report.json")["runs"]
            if row["case_id"] == expected_case
        )
        with np.load(path, allow_pickle=False) as archive:
            trace = {key: archive[key] for key in archive.files}
        observed = audit_trace(trace, case)
        differences = []
        for metric in METRICS:
            if observed[metric] is None or expected[metric] is None:
                if observed[metric] != expected[metric]:
                    raise ValueError("representative metric availability differs")
            elif not np.isclose(observed[metric], expected[metric], rtol=0, atol=1e-10):
                raise ValueError(f"representative metric differs: {metric}")
            else:
                differences.append(abs(observed[metric] - expected[metric]))
        if len(trace["time"]) != expected["physics_steps"]:
            raise ValueError("representative physics step count differs")
        results.append(
            {
                "stage": record["stage"],
                "case_id": expected_case,
                "physics_steps": len(trace["time"]),
                "maximum_absolute_metric_difference": max(differences, default=0.0),
            }
        )
    return results


def _copy(source, destination):
    if Path(source).is_symlink():
        raise ValueError("publication refuses source symlinks")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _comparison_rows(root, benchmark):
    rows = []
    for stage in ("bc", "ppo"):
        for item in _read(root / f"{stage}_summary.json")["development_test"]:
            for row in item["report"]["runs"]:
                rows.append({"method": f"{stage}_seed{item['seed']}", **row})
    for row in benchmark["runs"]:
        if row["split"] == "development_test":
            rows.append({**row, "episode_success": row["hard_safe_completion"]})
    return rows


def _write_comparison(staging, rows):
    columns = [
        "method",
        "case_id",
        "group_id",
        "simulation_seed",
        "episode_success",
        "tangent_rmse_mm",
        "force_rmse_n",
        "contact_ratio_pct",
        "peak_force_n",
        "max_penetration_mm",
        "max_speed_m_s",
        "saturation_pct",
        "intervention_pct",
        "episode_return",
    ]
    with (staging / "development_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summaries = {}
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]

        def mean(metric, cohort=selected):
            values = [row.get(metric) for row in cohort]
            return None if any(value is None for value in values) else float(np.mean(values))

        summaries[method] = {
            "case_count": len(selected),
            "safe_completions": sum(row["episode_success"] for row in selected),
            "tracking_passes": sum(
                row["episode_success"]
                and all(
                    row.get(metric) is not None
                    for metric in ("contact_ratio_pct", "tangent_rmse_mm", "force_rmse_n")
                )
                and row["contact_ratio_pct"] >= 99
                and row["tangent_rmse_mm"] <= 10
                and row["force_rmse_n"] <= 2
                for row in selected
            ),
            "mean_tangent_rmse_mm": mean("tangent_rmse_mm"),
            "mean_force_rmse_n": mean("force_rmse_n"),
            "minimum_contact_ratio_pct": (
                None
                if any(row.get("contact_ratio_pct") is None for row in selected)
                else min(row["contact_ratio_pct"] for row in selected)
            ),
        }
    _write_new(staging / "method_summary.json", summaries)
    return summaries


def _bc_closed_loop_errors(root, plan):
    """Relabel already recorded validation states; no new fit or dynamics rollout."""
    runs = []
    for seed in plan["seeds"]:
        source = root / "evaluations" / f"bc_seed{seed}_validation"
        report = _read(source / "report.json")
        for row in report["runs"]:
            with np.load(source / row["trace_file"], allow_pickle=False) as trace:
                observations, actions = trace["decision_observation"], trace["decision_action"]
                teacher = np.asarray([friction_teacher_action(obs[:33]) for obs in observations])
                error = (actions - teacher) ** 2
                runs.append(
                    {
                        "seed": seed,
                        "case_id": row["case_id"],
                        "split": "validation",
                        "policy_steps": len(actions),
                        "on_policy_teacher_action_mse": float(np.mean(error))
                        if len(actions)
                        else None,
                        "on_policy_teacher_action_mse_by_axis": error.mean(axis=0).tolist()
                        if len(actions)
                        else [None] * 3,
                        "actor_failure_count": row["actor_failure_count"],
                    }
                )
    return {
        "scope": "post-fit diagnostic on existing BC validation rollouts; no reselection",
        "interpretation": "action discrepancy on student states; not a causal proof of its source",
        "runs": runs,
    }


def _bc_export_check(root, plan, dataset):
    if _sha256(Path(dataset) / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("offline export check dataset differs from frozen input")
    batch = load_transition_batch(dataset, "validation")
    contract = policy_contract(
        plan["cases"], purpose="il_friction_teacher", nominal_kind="adaptive"
    )
    rows = []
    for seed in plan["seeds"]:
        source = root / f"bc_seed{seed}"
        actor, _ = actor_from_artifact(
            load_policy_artifact(source / "selected_model.json", expected_contract=contract)
        )
        prediction = np.stack([actor(observation) for observation in batch["observations"]])
        observed = float(np.mean((prediction - batch["actions"]) ** 2))
        expected = _read(source / "report.json")["selection"]["selected_validation_action_mse"]
        if not np.isclose(observed, expected, rtol=0, atol=1e-12):
            raise ValueError("exported BC no longer reproduces its selected validation MSE")
        rows.append(
            {
                "seed": seed,
                "numpy_validation_mse": observed,
                "training_report_validation_mse": expected,
                "absolute_difference": abs(observed - expected),
            }
        )
    return {
        "scope": "saved NumPy candidates on unchanged offline validation arrays",
        "absolute_tolerance": 1e-12,
        "passed": True,
        "runs": rows,
    }


def _plot_learning(staging, plan, benchmark):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    ppo = _read(staging / "ppo_summary.json")
    baseline = float(
        np.mean(
            [
                row["tangent_rmse_mm"]
                for row in benchmark["runs"]
                if row["split"] == "validation" and row["method"] == "zero_friction"
            ]
        )
    )
    deltas = []
    colors = ("#0072B2", "#D55E00", "#009E73")
    for seed, color in zip(plan["seeds"], colors):
        losses = _read(staging / f"bc_seed{seed}" / "epoch_losses.json")
        axes[0].plot(
            [row["epoch"] for row in losses],
            [row["validation_action_mse"] for row in losses],
            label=f"seed {seed}",
            color=color,
            linewidth=1.6,
        )
        run = next(item for item in ppo["validation"] if item["seed"] == seed)
        changes = [
            np.mean(
                [
                    np.nan if row["tangent_rmse_mm"] is None else row["tangent_rmse_mm"]
                    for row in item["report"]["runs"]
                ]
            )
            - baseline
            for item in run["checkpoints"]
        ]
        deltas.extend(value for value in changes if np.isfinite(value))
        axes[1].plot(
            [item["episode"] for item in run["checkpoints"]],
            changes,
            marker="o",
            label=f"seed {seed}",
            color=color,
            linewidth=1.6,
        )
    axes[1].axhline(
        0.0,
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="friction baseline",
    )
    axes[0].set(title="BC: offline validation only", xlabel="Epoch", ylabel="Action MSE")
    axes[0].set_yscale("log")
    axes[1].set(
        title="PPO: four fixed validation cases",
        xlabel="Training episodes",
        ylabel="Mean tangent RMSE change [mm]",
        xticks=[0, 16, 32],
    )
    extent = max(0.25, max(map(abs, deltas), default=0.0) * 1.2)
    axes[1].set_ylim(-extent, extent)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(staging / "learning_curves.png", dpi=180)
    plt.close(figure)


def _summary_text(staging, plan, summaries):
    def number(value):
        return "unavailable" if value is None else f"{value:.3f}"

    lines = [
        "# 小规模 BC / bounded Residual PPO 结果",
        "",
        "训练种子为 11、29、47。24 个公开 case 按组分为 16 train、4 validation、4 development_test。",
        "下表仅统计四个开发测试 case；重复训练种子不增加独立场景数，也不构成新盲测。",
        "",
        "| 方法 | 安全完成 | 跟踪门槛 | 平均切向 RMSE [mm] | 平均法向力 RMSE [N] | 最低接触率 [%] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, row in summaries.items():
        lines.append(
            f"| {method} | {row['safe_completions']}/{row['case_count']} | "
            f"{row['tracking_passes']}/{row['case_count']} | "
            f"{number(row['mean_tangent_rmse_mm'])} | {number(row['mean_force_rmse_n'])} | "
            f"{number(row['minimum_contact_ratio_pct'])} |"
        )
    lines += [
        "",
        "## 模型选择",
        "",
        "BC 每种子完整训练 100 epoch，按完整验证动作 MSE 选择。",
        "",
        "| BC seed | 选中 epoch | 验证动作 MSE | 零动作 MSE |",
        "|---:|---:|---:|---:|",
    ]
    for seed in plan["seeds"]:
        report = _read(staging / f"bc_seed{seed}" / "report.json")
        selected = report["selection"]
        lines.append(
            f"| {seed} | {selected['selected_epoch']} | "
            f"{selected['selected_validation_action_mse']:.6g} | "
            f"{report['diagnostics']['validation']['zero_action_mse']:.6g} |"
        )
    lines += [
        "",
        "PPO 每种子固定尝试 32 个训练回合，只比较第 0、16、32 回合检查点。",
        "第 0 回合是未训练的零残差；若被选中，不能称为 RL 优于强基线。",
        "",
        "| PPO seed | 选中回合 | 验证资格 | 开发测试门槛 |",
        "|---:|---:|---|---|",
    ]
    ppo = _read(staging / "ppo_summary.json")
    for selected, development in zip(ppo["selections"], ppo["development_test"]):
        lines.append(
            f"| {selected['seed']} | {selected['episode']} | {selected['eligible']} | "
            f"{'PASS' if development['report']['acceptance_met'] else 'FAIL'} |"
        )
    lines += [
        "",
        "完整验证检查点如下。数值是四个 case 的均值，资格仍要求每个 case 分别过门槛。",
        "",
        "| PPO seed | 回合 | 平均切向 RMSE [mm] | 平均法向力 RMSE [N] | 合格 |",
        "|---:|---:|---:|---:|---|",
    ]
    for seed_record in ppo["validation"]:
        for checkpoint in seed_record["checkpoints"]:
            rows = checkpoint["report"]["runs"]
            tangent = [row.get("tangent_rmse_mm") for row in rows]
            force = [row.get("force_rmse_n") for row in rows]
            tangent_mean = None if None in tangent else float(np.mean(tangent))
            force_mean = None if None in force else float(np.mean(force))
            lines.append(
                f"| {seed_record['seed']} | {checkpoint['episode']} | {number(tangent_mean)} | "
                f"{number(force_mean)} | {checkpoint['report']['acceptance_met']} |"
            )
    lines += [
        "",
        "![BC 验证损失和 PPO 固定验证集误差](learning_curves.png)",
        "",
        "图中曲线用于模型选择，均来自 validation，未使用 development_test。",
        "PPO 图绘制相对摩擦强基线的误差差值，负值表示降低，单位是 mm。",
        "每种子只有 8 次 PPO 采样更新，不能据此声称训练收敛。",
        "",
        "## 复核范围",
        "",
        (
            "[逐 case 对照](development_comparison.csv)保留所有种子，差值详见 "
            "[BC](bc_summary.json) 与 [PPO](ppo_summary.json) 的 paired_baselines。"
        ),
        (
            "[plan.json](plan.json)记录训练前固定的预算和源码身份；"
            "[选择记录](ppo_selection.json)保留全部 PPO 检查点的验证结果。"
        ),
        (
            "[闭环动作复核](bc_closed_loop_errors.json)在已经保存的验证状态上重新计算教师动作，"
            "用于比较离线 MSE 与学生实际访问状态上的误差，不重新训练或挑选权重。"
        ),
        "[导出复核](bc_export_check.json)用保存的 NumPy 候选重新计算离线验证 MSE，并与训练记录比较。",
        "",
        "本目录是小体积公开副本：包含全部评测候选权重、训练统计与评价报告。",
        (
            "附带 BC 与 PPO 各一份完整物理 NPZ，按首个训练种子、首个开发 case 选择，"
            "不是挑选最好表现。路径见[代表轨迹索引](representative_traces.json)，"
            "[重算记录](representative_physics_audit.json)核对全部物理指标。其余轨迹与逐决策事件日志留在本地。"
        ),
        (
            "各子目录的 source_manifest.json/source_COMPLETE 是原完整产物的哈希记录，"
            "不能当作本目录含有全部原始文件的证明。根 manifest.json 只校验实际分发的文件。"
        ),
        "",
        "```bash",
        (
            "python -m tools.publish_surface_learning_pilot --audit "
            "results/franka_surface_learning_pilot"
        ),
        "```",
        "",
        (
            "该命令检查公开文件哈希、重算 PPO 选择及两份轨迹指标，不会重新积分动力学；"
            "完整重训与闭环复现见[算法和命令](../../docs/surface_learning_pilot.md)。"
        ),
        "旧 v0.5 的 FAIL 不变。全部候选仅用于仿真，禁止直接用于真机。",
        "",
    ]
    return "\n".join(lines)


def publish_pilot(root, output, benchmark, *, expected_plan_sha256, dataset):
    root, output, benchmark = Path(root), Path(output), Path(benchmark)
    if output.exists():
        raise ValueError("publication output must be new")
    if _sha256(root / "plan.json") != expected_plan_sha256:
        raise ValueError("plan differs from preserved pre-fit SHA")
    if (root / "PLAN_SHA256").read_text().strip() != expected_plan_sha256:
        raise ValueError("stored plan marker differs from preserved pre-fit SHA")
    plan = _read(root / "plan.json")
    if _sha256(benchmark / "report.json") != plan["benchmark_report_sha256"]:
        raise ValueError("baseline report differs from frozen identity")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".surface-pilot-public-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "public"
        staging.mkdir()
        for filename in (
            "plan.json",
            "PLAN_SHA256",
            "bc_summary.json",
            "ppo_summary.json",
            "bc_selection.json",
            "ppo_selection.json",
            "bc_COMPLETE",
            "ppo_COMPLETE",
        ):
            _copy(root / filename, staging / filename)
        for stage in ("bc", "ppo"):
            if (root / f"{stage}_COMPLETE").read_text().strip() != _sha256(
                root / f"{stage}_summary.json"
            ):
                raise ValueError("stage is incomplete")
            for seed in plan["seeds"]:
                source = root / f"{stage}_seed{seed}"
                manifest = _training_manifest(stage, source, seed, plan)
                names = (
                    ["report.json", "epoch_losses.csv", "epoch_losses.json", "selected_model.json"]
                    if stage == "bc"
                    else list(manifest["checkpoints"].values())
                )
                for name in names:
                    _copy(source / name, staging / source.name / name)
                _copy(source / "manifest.json", staging / source.name / "source_manifest.json")
                _copy(source / "COMPLETE", staging / source.name / "source_COMPLETE")
        for source in sorted((root / "evaluations").iterdir()):
            if source.is_file():
                _copy(source, staging / "evaluations" / source.name)
            elif source.is_dir():
                _verified_manifest(source)
                destination = staging / "evaluations" / source.name
                _copy(source / "report.json", destination / "report.json")
                _copy(source / "manifest.json", destination / "source_manifest.json")
                _copy(source / "COMPLETE", destination / "source_COMPLETE")
        representative_case = min(
            case["case_id"] for case in plan["cases"] if case["split"] == "development_test"
        )
        representatives = []
        for stage in ("bc", "ppo"):
            seed = plan["seeds"][0]
            source = root / "evaluations" / f"{stage}_seed{seed}_selected_development_test"
            row = next(
                row
                for row in _read(source / "report.json")["runs"]
                if row["case_id"] == representative_case
            )
            relative = Path("evaluations") / source.name / row["trace_file"]
            _copy(root / relative, staging / relative)
            representatives.append(
                {
                    "stage": stage,
                    "seed": seed,
                    "case_id": representative_case,
                    "file": str(relative),
                }
            )
        _write_new(
            staging / "representative_traces.json",
            {
                "selection": "first training seed and lexicographically first development case; reporting sample",
                "records": representatives,
            },
        )
        _write_new(
            staging / "representative_physics_audit.json", _audit_representatives(staging, plan)
        )
        baseline = _read(benchmark / "report.json")
        _copy(benchmark / "report.json", staging / "baseline_report.json")
        summaries = _write_comparison(staging, _comparison_rows(root, baseline))
        _write_new(staging / "bc_closed_loop_errors.json", _bc_closed_loop_errors(root, plan))
        _write_new(staging / "bc_export_check.json", _bc_export_check(root, plan, dataset))
        _plot_learning(staging, plan, baseline)
        (staging / "README.md").write_text(
            _summary_text(staging, plan, summaries), encoding="utf-8"
        )
        manifest = {
            "schema": "surface_learning_pilot_public_v1",
            "plan_sha256": expected_plan_sha256,
            "new_holdout": False,
            "all_physics_traces_distributed": False,
            "publisher_sha256": _sha256(Path(__file__)),
            "artifact_sha256": {
                str(path.relative_to(staging)): _sha256(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            },
        }
        _write_new(staging / "manifest.json", manifest)
        (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
        audit_public_pilot(staging)
        if output.exists():
            raise FileExistsError("publication destination appeared")
        os.rename(staging, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args(argv)
    if args.audit:
        print(json.dumps(audit_public_pilot(args.audit), sort_keys=True))
    else:
        if not all(
            (args.pilot, args.output, args.benchmark, args.dataset, args.expected_plan_sha256)
        ):
            parser.error(
                "publication needs --pilot, --output, --benchmark, --dataset and --expected-plan-sha256"
            )
        print(
            publish_pilot(
                args.pilot,
                args.output,
                args.benchmark,
                expected_plan_sha256=args.expected_plan_sha256,
                dataset=args.dataset,
            )
        )


if __name__ == "__main__":
    main()
