"""Publish and audit compact evidence for the controlled surface BC transfer study."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_experiment import _sha256
from compliant_control_lab.surface_readiness_benchmark import METRICS, audit_trace
from tools import surface_bc_transfer as transfer
from tools import surface_learning_pilot as parent_pilot


def _copy(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink():
        raise ValueError("publication refuses source symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _pairs(plan):
    return [(arm, seed) for arm in plan["arms"] for seed in plan["seeds"]]


def _case_ids(plan, split):
    return sorted(case["case_id"] for case in plan["cases"] if case["split"] == split)


def _expected_mask(plan, arm):
    width = plan["architecture"][0]
    intervention = plan["input_interventions"][arm]
    if "zero_indices" in intervention:
        active = set(range(width)) - set(intervention["zero_indices"])
    else:
        active = set(intervention["keep_indices"])
    return [int(index in active) for index in range(width)]


def _validate_report(report, plan, split):
    expected = _case_ids(plan, split)
    if (
        report.get("evaluation_split") != split
        or report.get("evaluation_case_ids") != expected
        or report.get("expected_case_count") != len(expected)
        or sorted(row["case_id"] for row in report.get("runs", [])) != expected
        or any(row.get("split") != split for row in report.get("runs", []))
    ):
        raise ValueError(f"incomplete or different {split} case cohort")


def _validate_summary(summary, selection, plan):
    expected = _pairs(plan)
    if (
        summary.get("schema") != "surface_bc_transfer_summary_v1"
        or summary.get("new_holdout") is not False
        or summary.get("plan_sha256") != _sha256(Path(selection).parent / "plan.json")
        or summary.get("selection_sha256") != _sha256(selection)
    ):
        raise ValueError("transfer summary identity differs")
    record = parent_pilot._read(selection)
    expected_record = {
        key: summary[key] for key in ("plan_sha256", "candidates", "validation", "selection")
    }
    if record != expected_record:
        raise ValueError("selection record differs from completed summary")
    if [(item["arm"], item["seed"]) for item in summary["candidates"]] != expected:
        raise ValueError("candidate cohort omits an arm or seed")
    for item in summary["candidates"]:
        candidate = Path(selection).parent / item["candidate"]
        if (
            item["candidate"] != f"{item['arm']}_seed{item['seed']}/selected_model.json"
            or item["candidate_sha256"] != _sha256(candidate)
        ):
            raise ValueError("selected candidate path or hash differs")
    for split in ("validation", "development_test"):
        cohort = summary[split]
        if [(item["arm"], item["seed"]) for item in cohort] != expected:
            raise ValueError(f"{split} cohort omits an arm or seed")
        for item in cohort:
            _validate_report(item["report"], plan, split)
    recomputed = transfer.select_arm(summary["validation"], plan)
    if summary["selection"] != recomputed:
        raise ValueError("BC arm selection differs from complete validation cohort")


def _require_summary_report(summary, arm, seed, split, report):
    matches = [
        item["report"]
        for item in summary[split]
        if item["arm"] == arm and item["seed"] == seed
    ]
    if matches != [report]:
        raise ValueError("transfer summary and evaluation report differ")


def _validate_source(pilot, parent, benchmark, expected_plan_sha256):
    pilot, parent, benchmark = Path(pilot), Path(parent), Path(benchmark)
    plan_path = pilot / "plan.json"
    if (
        _sha256(plan_path) != expected_plan_sha256
        or (pilot / "PLAN_SHA256").read_text().strip() != expected_plan_sha256
    ):
        raise ValueError("transfer plan differs from preserved pre-fit SHA")
    plan = parent_pilot._read(plan_path)
    parent_plan = parent_pilot._read(parent / "plan.json")
    if (
        plan["parent_plan_sha256"] != _sha256(parent / "plan.json")
        or (parent / "PLAN_SHA256").read_text().strip() != plan["parent_plan_sha256"]
        or plan["parent_bc_records_sha256"] != transfer._parent_records(parent, parent_plan)
    ):
        raise ValueError("full49 parent control identity differs")
    parent_pilot._verified_manifest(benchmark)
    if (
        _sha256(benchmark / "manifest.json") != parent_plan["benchmark_manifest_sha256"]
        or _sha256(benchmark / "report.json") != parent_plan["benchmark_report_sha256"]
    ):
        raise ValueError("baseline report differs from parent frozen plan")
    if plan["source_sha256"] != transfer._sources():
        raise ValueError("transfer orchestration source changed after freezing")
    for arm, seed in _pairs(plan):
        transfer._training_manifest(pilot, plan, arm, seed)
    summary_path, selection_path = pilot / "summary.json", pilot / "selection.json"
    if (pilot / "COMPLETE").read_text().strip() != _sha256(summary_path):
        raise ValueError("transfer summary is incomplete")
    summary = parent_pilot._read(summary_path)
    _validate_summary(summary, selection_path, plan)
    for arm, seed in _pairs(plan):
        for split in ("validation", "development_test"):
            name = _evaluation_name(arm, seed, split)
            _require_summary_report(
                summary,
                arm,
                seed,
                split,
                parent_pilot._read(pilot / "evaluations" / name / "report.json"),
            )
    _validate_student_errors(parent_pilot._read(pilot / "student_state_errors.json"), plan)
    return plan, parent_plan, summary


def _validate_student_errors(diagnostic, plan):
    expected = {
        (arm, seed, case_id)
        for arm, seed in _pairs(plan)
        for case_id in _case_ids(plan, "validation")
    }
    runs = diagnostic.get("runs", [])
    observed = {(row["arm"], row["seed"], row["case_id"]) for row in runs}
    if (
        diagnostic.get("use")
        != "post-fit diagnostic; not training or checkpoint/arm selection"
        or observed != expected
        or len(runs) != len(expected)
        or any(row.get("split") != "validation" for row in runs)
    ):
        raise ValueError("student-state diagnostic cohort is incomplete")


def _evaluation_name(arm, seed, split):
    return f"{arm}_seed{seed}_{split}"


def _copy_evaluation(pilot, staging, plan, arm, seed, split):
    name = _evaluation_name(arm, seed, split)
    source = Path(pilot) / "evaluations" / name
    source_manifest = parent_pilot._verified_manifest(source)
    report = parent_pilot._read(source / "report.json")
    _validate_report(report, plan, split)
    destination = staging / "evaluations" / name
    for name_in, name_out in (
        ("report.json", "report.json"),
        ("manifest.json", "source_manifest.json"),
        ("COMPLETE", "source_COMPLETE"),
    ):
        _copy(source / name_in, destination / name_out)
    protocol = Path(pilot) / "evaluations" / f"{name}.protocol.json"
    pin = protocol.with_suffix(".sha256")
    if pin.read_text().strip() != _sha256(protocol):
        raise ValueError("evaluation protocol pin differs")
    _copy(protocol, staging / "evaluations" / protocol.name)
    _copy(pin, staging / "evaluations" / pin.name)
    candidate = Path(pilot) / f"{arm}_seed{seed}" / "selected_model.json"
    if (
        source_manifest["candidate_artifact_sha256"] != _sha256(candidate)
        or source_manifest["protocol_sha256"] != _sha256(protocol)
        or source_manifest["evaluation_split"] != split
    ):
        raise ValueError("evaluation source identity differs")
    return source, destination, report


def _audit_representatives(directory, plan):
    directory = Path(directory)
    index = parent_pilot._read(directory / "representative_traces.json")
    expected_case = _case_ids(plan, "development_test")[0]
    expected = [(arm, plan["seeds"][0], expected_case) for arm in plan["arms"]]
    if [
        (item["arm"], item["seed"], item["case_id"]) for item in index.get("records", [])
    ] != expected:
        raise ValueError("representative trace selection differs")
    results = []
    for record in index["records"]:
        path = directory / record["file"]
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("unsafe representative trace path")
        source_manifest = parent_pilot._read(path.parent / "source_manifest.json")
        if _sha256(path) != source_manifest["artifact_sha256"].get(path.name):
            raise ValueError("representative trace differs from source manifest")
        report = parent_pilot._read(path.parent / "report.json")
        expected_row = next(row for row in report["runs"] if row["case_id"] == expected_case)
        case = next(case for case in plan["cases"] if case["case_id"] == expected_case)
        with np.load(path, allow_pickle=False) as stored:
            observed = audit_trace({key: stored[key] for key in stored.files}, case)
            physics_steps = len(stored["time"])
        differences = []
        for metric in METRICS:
            left, right = observed[metric], expected_row[metric]
            if left is None or right is None:
                if left != right:
                    raise ValueError(f"representative metric availability differs: {metric}")
            elif not np.isclose(left, right, rtol=0, atol=1e-10):
                raise ValueError(f"representative metric differs: {metric}")
            else:
                differences.append(abs(left - right))
        if physics_steps != expected_row["physics_steps"]:
            raise ValueError("representative physics step count differs")
        results.append(
            {
                "arm": record["arm"],
                "seed": record["seed"],
                "case_id": expected_case,
                "physics_steps": physics_steps,
                "maximum_absolute_metric_difference": max(differences, default=0.0),
            }
        )
    return results


def _audit_training_copy(directory, plan, arm, seed):
    root = Path(directory) / f"{arm}_seed{seed}"
    source = parent_pilot._read(root / "source_manifest.json")
    if (root / "source_COMPLETE").read_text().strip() != _sha256(
        root / "source_manifest.json"
    ):
        raise ValueError("copied training source manifest is incomplete")
    copied = ("report.json", "epoch_losses.csv", "epoch_losses.json", "selected_model.json")
    if any(_sha256(root / name) != source["artifact_sha256"].get(name) for name in copied):
        raise ValueError("copied training artifact differs from source manifest")
    if (
        source["schema"] != "surface_behavior_clone_transfer_run_v1"
        or source["arm"] != arm
        or source["hyperparameters"]["seed"] != seed
        or any(source["hyperparameters"].get(key) != value for key, value in plan["bc"].items())
        or source["input_mask"] != _expected_mask(plan, arm)
        or source["dataset"]["manifest_sha256"] != plan["dataset_manifest_sha256"]
        or source["trainer_source_sha256"] != plan["transfer_trainer_sources"]
        or any(
            source["runtime"].get(key) != value
            for key, value in plan["training_runtime"].items()
        )
        or source["development_test_used"] is not False
    ):
        raise ValueError("copied training identity differs from transfer plan")
    history = parent_pilot._read(root / "epoch_losses.json")
    if [row["epoch"] for row in history] != list(range(1, plan["bc"]["epochs"] + 1)):
        raise ValueError("copied training epoch cohort is incomplete")
    if not all(np.isfinite(row["validation_action_mse"]) for row in history):
        raise ValueError("copied checkpoint losses are nonfinite")
    selected = min(history, key=lambda row: (row["validation_action_mse"], row["epoch"]))
    report = parent_pilot._read(root / "report.json")
    if (
        len(history) != plan["bc"]["epochs"]
        or report["selection"]["selected_epoch"] != selected["epoch"]
        or report["arm"] != arm
        or report["input_mask"] != _expected_mask(plan, arm)
    ):
        raise ValueError("copied selected epoch or input mask differs")


def _audit_evaluation_copy(directory, plan, arm, seed, split):
    name = _evaluation_name(arm, seed, split)
    root = Path(directory) / "evaluations" / name
    source = parent_pilot._read(root / "source_manifest.json")
    if (
        (root / "source_COMPLETE").read_text().strip()
        != _sha256(root / "source_manifest.json")
        or _sha256(root / "report.json") != source["artifact_sha256"].get("report.json")
    ):
        raise ValueError("copied evaluation differs from source manifest")
    report = parent_pilot._read(root / "report.json")
    _validate_report(report, plan, split)
    protocol = Path(directory) / "evaluations" / f"{name}.protocol.json"
    pin = protocol.with_suffix(".sha256")
    candidate = Path(directory) / f"{arm}_seed{seed}" / "selected_model.json"
    if (
        pin.read_text().strip() != _sha256(protocol)
        or source["protocol_sha256"] != _sha256(protocol)
        or source["candidate_artifact_sha256"] != _sha256(candidate)
        or source["evaluation_split"] != split
    ):
        raise ValueError("copied evaluation protocol/candidate identity differs")
    return report


def _method_rows(transfer_summary, control_summary):
    rows = []
    for split in ("validation", "development_test"):
        for item in control_summary[split]:
            rows.append(("full49", item["seed"], split, item["report"]))
        for item in transfer_summary[split]:
            rows.append((item["arm"], item["seed"], split, item["report"]))
    return rows


def _readme(plan, transfer_summary, control_summary):
    def number(values):
        return (
            "不可用"
            if any(value is None or not np.isfinite(value) for value in values)
            else f"{np.mean(values):.3f}"
        )

    def all_gates(row):
        values = [row.get(name) for name in (*plan["thresholds"], "max_contact_loss_s")]
        if not row.get("episode_success") or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
            for value in values
        ):
            return False
        if row["max_contact_loss_s"] >= (
            parent_pilot.HARD_SAFETY_LIMITS["contact_loss_timeout_s"] - 1e-12
        ):
            return False
        return all(
            row[name] <= rule["value"]
            if rule["operator"] == "<="
            else row[name] >= rule["value"]
            for name, rule in plan["thresholds"].items()
        )

    lines = [
        "# 表面行为克隆输入消融",
        "",
        "该归档比较历史 full49 控制与两个预先固定的输入掩码。所有结果来自同一公开 16/4/4 分组、",
        "同一训练预算和种子 11/29/47；development_test 已经是公开开发集，不是新盲测。",
        "这是输入组干预比较，但不能据此断言某个单独通道是唯一原因，也不能推出真机泛化能力。",
        "",
        "| 输入 | seed | 集合 | 全部门槛 | 安全完成 | 跟踪通过 | 平均切向 RMSE [mm] | 平均力 RMSE [N] |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for method, seed, split, report in _method_rows(transfer_summary, control_summary):
        rows = report["runs"]
        lines.append(
            f"| {method} | {seed} | {split} | {'PASS' if report['acceptance_met'] else 'FAIL'} | "
            f"{sum(row['episode_success'] for row in rows)}/{len(rows)} | "
            f"{sum(all_gates(row) for row in rows)}/{len(rows)} | "
            f"{number([row.get('tangent_rmse_mm') for row in rows])} | "
            f"{number([row.get('force_rmse_n') for row in rows])} |"
        )
    lines += [
        "",
        (
            "验证集预先选择："
            f"`selected_arm={transfer_summary['selection']['selected_arm']}`，"
            f"`eligible={transfer_summary['selection']['eligible']}`；完整均值和排序见 "
            "[selection.json](selection.json)。该选择不使用 development_test。"
        ),
        "",
        "每个掩码和种子的权重、完整 epoch 曲线、训练报告、12 份闭环评价报告及其协议均已复制。",
        "归档仅附每个掩码各一份代表性完整物理 NPZ：首个种子和按字典序首个开发 case。",
        "其他物理轨迹和逐决策事件只保留在本地源产物中；source_manifest/source_COMPLETE 只是哈希身份记录。",
        "根 manifest.json 仅绑定实际分发文件。所有候选仅限仿真，不能直接部署到真机。",
        "",
        "```bash",
        "python -m tools.publish_surface_bc_transfer --audit results/franka_surface_bc_transfer",
        "```",
        "",
    ]
    return "\n".join(lines)


def audit(directory):
    """Audit only the compact public archive; no Torch or dynamics rerun is required."""
    directory = Path(directory)
    manifest = parent_pilot._verified_manifest(directory)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    if actual != set(manifest["artifact_sha256"]) | {"manifest.json", "COMPLETE"}:
        raise ValueError("public archive contains missing or unlisted files")
    plan = parent_pilot._read(directory / "plan.json")
    if (
        manifest.get("schema") != "surface_bc_transfer_public_v1"
        or manifest.get("plan_sha256") != _sha256(directory / "plan.json")
        or (directory / "PLAN_SHA256").read_text().strip() != manifest["plan_sha256"]
        or manifest.get("new_holdout") is not False
        or manifest.get("all_physics_traces_distributed") is not False
    ):
        raise ValueError("public transfer manifest/plan identity differs")
    summary_path, selection_path = directory / "summary.json", directory / "selection.json"
    if (directory / "transfer_COMPLETE").read_text().strip() != _sha256(summary_path):
        raise ValueError("copied transfer summary is incomplete")
    summary = parent_pilot._read(summary_path)
    _validate_summary(summary, selection_path, plan)
    for arm, seed in _pairs(plan):
        _audit_training_copy(directory, plan, arm, seed)
        for split in ("validation", "development_test"):
            _require_summary_report(
                summary,
                arm,
                seed,
                split,
                _audit_evaluation_copy(directory, plan, arm, seed, split),
            )
    _validate_student_errors(parent_pilot._read(directory / "student_state_errors.json"), plan)
    parent_plan = parent_pilot._read(directory / "control_full49" / "plan.json")
    if (
        _sha256(directory / "control_full49" / "plan.json") != plan["parent_plan_sha256"]
        or _sha256(directory / "baseline_report.json")
        != parent_plan["benchmark_report_sha256"]
    ):
        raise ValueError("copied parent plan or baseline differs")
    for name, digest in plan["parent_bc_records_sha256"].items():
        public_name = (
            Path("control_full49") / name.replace("manifest.json", "source_manifest.json")
        )
        public_name = Path(str(public_name).replace("/COMPLETE", "/source_COMPLETE"))
        if _sha256(directory / public_name) != digest:
            raise ValueError("copied full49 control differs")
    representatives = _audit_representatives(directory, plan)
    if parent_pilot._read(directory / "representative_physics_audit.json") != representatives:
        raise ValueError("stored representative audit differs from recomputation")
    return {
        "hashes_match": True,
        "selection_recomputed": True,
        "representative_physics_traces_recomputed": len(representatives),
        "distributed_file_count": len(manifest["artifact_sha256"]),
        "all_physics_traces_distributed": False,
        "new_holdout": False,
    }


def publish(pilot, parent, benchmark, output, *, expected_plan_sha256):
    pilot, parent, benchmark, output = map(Path, (pilot, parent, benchmark, output))
    publisher_sha256 = _sha256(Path(__file__))
    if output.exists():
        raise ValueError("publication output must be new")
    plan, _parent_plan, summary = _validate_source(
        pilot, parent, benchmark, expected_plan_sha256
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-bc-transfer-public-", dir=output.parent) as temp:
        staging = Path(temp) / "public"
        staging.mkdir()
        for name_in, name_out in (
            ("plan.json", "plan.json"),
            ("PLAN_SHA256", "PLAN_SHA256"),
            ("selection.json", "selection.json"),
            ("summary.json", "summary.json"),
            ("COMPLETE", "transfer_COMPLETE"),
            ("student_state_errors.json", "student_state_errors.json"),
        ):
            _copy(pilot / name_in, staging / name_out)
        for arm, seed in _pairs(plan):
            source = pilot / f"{arm}_seed{seed}"
            destination = staging / source.name
            for name in ("report.json", "epoch_losses.csv", "epoch_losses.json", "selected_model.json"):
                _copy(source / name, destination / name)
            _copy(source / "manifest.json", destination / "source_manifest.json")
            _copy(source / "COMPLETE", destination / "source_COMPLETE")
        evaluations = {}
        for arm, seed in _pairs(plan):
            for split in ("validation", "development_test"):
                evaluations[(arm, seed, split)] = _copy_evaluation(
                    pilot, staging, plan, arm, seed, split
                )

        control = staging / "control_full49"
        for name in ("plan.json", "PLAN_SHA256", "bc_summary.json", "bc_selection.json", "bc_COMPLETE"):
            _copy(parent / name, control / name)
        for seed in plan["seeds"]:
            source = parent / f"bc_seed{seed}"
            destination = control / source.name
            _copy(source / "report.json", destination / "report.json")
            _copy(source / "selected_model.json", destination / "selected_model.json")
            _copy(source / "manifest.json", destination / "source_manifest.json")
            _copy(source / "COMPLETE", destination / "source_COMPLETE")
        _copy(benchmark / "report.json", staging / "baseline_report.json")

        representative_case = _case_ids(plan, "development_test")[0]
        representatives = []
        for arm in plan["arms"]:
            seed = plan["seeds"][0]
            source, destination, report = evaluations[(arm, seed, "development_test")]
            row = next(row for row in report["runs"] if row["case_id"] == representative_case)
            _copy(source / row["trace_file"], destination / row["trace_file"])
            representatives.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "case_id": representative_case,
                    "file": str(
                        Path("evaluations") / source.name / row["trace_file"]
                    ),
                }
            )
        parent_pilot._write_new(
            staging / "representative_traces.json",
            {
                "selection": "each arm; first seed; lexicographically first development case",
                "records": representatives,
            },
        )
        parent_pilot._write_new(
            staging / "representative_physics_audit.json",
            _audit_representatives(staging, plan),
        )
        control_summary = parent_pilot._read(parent / "bc_summary.json")
        (staging / "README.md").write_text(
            _readme(plan, summary, control_summary), encoding="utf-8"
        )
        manifest = {
            "schema": "surface_bc_transfer_public_v1",
            "plan_sha256": expected_plan_sha256,
            "new_holdout": False,
            "all_physics_traces_distributed": False,
            "representative_trace_count": 2,
            "publisher_sha256": publisher_sha256,
            "artifact_sha256": {
                str(path.relative_to(staging)): _sha256(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            },
        }
        parent_pilot._write_new(staging / "manifest.json", manifest)
        (staging / "COMPLETE").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="utf-8"
        )
        audit(staging)
        if publisher_sha256 != _sha256(Path(__file__)):
            raise RuntimeError("publisher source changed during publication")
        if output.exists():
            raise FileExistsError("publication destination appeared")
        os.rename(staging, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args(argv)
    if args.audit:
        print(json.dumps(audit(args.audit), sort_keys=True))
    else:
        if not all((args.pilot, args.parent, args.benchmark, args.output, args.expected_plan_sha256)):
            parser.error(
                "publication needs --pilot, --parent, --benchmark, --output and --expected-plan-sha256"
            )
        result = publish(
            args.pilot,
            args.parent,
            args.benchmark,
            args.output,
            expected_plan_sha256=args.expected_plan_sha256,
        )
        print(json.dumps({"output": str(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
