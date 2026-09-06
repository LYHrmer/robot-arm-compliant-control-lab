"""Causal reruns and explicitly incomplete, independently auditable dataset examples."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_dataset import audit_dataset
from compliant_control_lab.surface_env import SurfaceLearningEnv
from compliant_control_lab.surface_experiment import _output_path, _sha256, _source_hashes
from compliant_control_lab.surface_policy import SurfaceResidualConfig
from compliant_control_lab.surface_simulation import SurfaceScenario, SurfaceSimulationConfig


def _same(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    equal = (
        np.array_equal(actual, expected, equal_nan=True)
        if actual.dtype.kind in "f" and expected.dtype.kind in "f"
        else np.array_equal(actual, expected)
    )
    if not equal:
        raise ValueError(f"causal replay mismatch: {label}")


def replay_dataset(path, *, case_ids=None):
    """Reintegrate stored actions, not recompute labels or copy a saved trace.

    Exact comparison is intended for the recorded software/physics version.
    A different version may legitimately diverge; such a run does not pass.
    The returned scope always names the episodes actually replayed.
    """
    path = Path(path)
    audit = audit_dataset(path)
    manifest = json.loads((path / "manifest.json").read_text())
    cases = {case["case_id"]: case for case in manifest["cases"]}
    available = {entry["case_id"] for entry in manifest["episodes"]}
    selected = available if case_ids is None else set(case_ids)
    if not selected or not selected <= available:
        raise ValueError("replay selection must name existing episodes")
    source_before = _source_hashes()
    completed = []
    for entry in manifest["episodes"]:
        identity = entry["case_id"]
        if identity not in selected:
            continue
        case = cases[identity]
        with np.load(path / entry["file"], allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files}
        env = SurfaceLearningEnv(
            SurfaceFrame(case["controller_frame_rotation"]),
            scenario=SurfaceScenario(**case["scenario"]),
            config=SurfaceSimulationConfig(**case["config"]),
            task=LearningSurfaceTask(**case["task"]),
            nominal_kind=manifest["nominal_kind"],
            residual_config=SurfaceResidualConfig(**manifest["residual_config"]),
        )
        try:
            obs, initial = env.reset(seed=entry["env_seed"])
            _same(initial["simulation_seed"], entry["simulation_seed"], "reset seed")
            reasons = []
            for index, action in enumerate(arrays["actions"]):
                _same(obs, arrays["observations"][index], f"{identity}: observation {index}")
                obs, reward, terminated, truncated, info = env.step(action)
                valid = info["terminal_observation_valid"]
                for key, value in {
                    "next_observations": obs if valid else np.full(49, np.nan),
                    "rewards": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "terminal_observation_valid": valid,
                    "physics_substeps": info["physics_substeps"],
                    "attempted_substeps": info["attempted_substeps"],
                    "time_after_s": info["elapsed_s"],
                    "termination_reasons": "|".join(info["termination_reasons"]),
                    "attempted_execution_status": "|".join(
                        stage["execution_status"] for stage in info["action_stages"]
                    ),
                }.items():
                    _same(value, arrays[key][index], f"{identity}: {key} {index}")
                reasons.extend(
                    "|".join(stage["reasons"])
                    for stage in info["action_stages"]
                    if stage["execution_status"] == "recorded"
                )
            trace = env.result().trace if entry["physics_count"] else {}
            trace["residual_reasons"] = np.asarray(reasons, dtype=str)
            expected = {
                name[7:]: value for name, value in arrays.items() if name.startswith("trace__")
            }
            if trace.keys() != expected.keys():
                raise ValueError(f"causal replay trace schema mismatch: {identity}")
            for key, value in trace.items():
                _same(value, expected[key], f"{identity}: trace {key}")
            completed.append(identity)
        finally:
            env.close()
    if _source_hashes() != source_before:
        raise RuntimeError("source changed during causal replay")
    return {
        "valid": True,
        "comparison": "exact including nonfinite failure evidence",
        "dataset_audit": audit,
        "replayed_case_ids": completed,
        "replayed_episodes": len(completed),
        "available_episodes": len(available),
        "manifest_sha256": _sha256(path / "manifest.json"),
        "source_matches_collection": source_before == manifest["source_and_assets_sha256"],
        "replay_source_and_assets_sha256": source_before,
    }


def export_examples(source, output, *, case_ids):
    """Copy selected full episodes, preserving the entire parent split plan."""
    source, output = Path(source), _output_path(output)
    audit_dataset(source)
    if output.exists():
        raise ValueError("example output must be a new directory")
    manifest = json.loads((source / "manifest.json").read_text())
    selected = set(case_ids)
    available = {entry["case_id"] for entry in manifest["episodes"]}
    if not selected or not selected < available:
        raise ValueError("examples must be a nonempty proper subset of available episodes")
    manifest["episodes"] = [entry for entry in manifest["episodes"] if entry["case_id"] in selected]
    manifest["corpus_scope"] = "example_subset"
    manifest["parent_manifest_sha256"] = _sha256(source / "manifest.json")
    manifest["parent_episode_count"] = len(available)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-examples-", dir=output.parent) as temporary:
        staging = Path(temporary) / "examples"
        staging.mkdir()
        for entry in manifest["episodes"]:
            shutil.copyfile(source / entry["file"], staging / entry["file"])
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        (staging / "COMPLETE").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="utf-8"
        )
        audit_dataset(staging)
        if _output_path(output).exists():
            raise ValueError("example output appeared during export")
        os.rename(staging, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--export-examples", type=Path)
    args = parser.parse_args(argv)
    if args.export_examples:
        if not args.case_ids:
            parser.error("--export-examples requires explicit --case-ids")
        result = {
            "output": str(
                export_examples(args.dataset, args.export_examples, case_ids=args.case_ids)
            )
        }
    else:
        result = replay_dataset(args.dataset, case_ids=args.case_ids)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
