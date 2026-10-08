"""Prepare, verify and run BC/PPO in local MuJoCo with explicit artifact pins."""

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

from tools.learning_deployment.bundle import accept_bundle, profile_actor, run_bundle, verify_bundle
from tools.learning_deployment.pipeline import BC_INPUT_MODES, prepare, preview


def doctor(*, training=False):
    packages = ["numpy", "mujoco", "gymnasium", "matplotlib"]
    if training:
        packages.append("torch")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    result = {"status": "FAIL" if missing else "PASS", "missing_dependencies": missing,
              "python": platform.python_version(), "platform": platform.platform(),
              "architecture": platform.machine(),
              "versions": {name: version(name) for name in packages if name not in missing},
              "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
              "training_requested": training, "target": "local_mujoco_simulation",
              "inference_requires_torch": False,
              "network_required_at_inference": False,
              "display_required": False}
    if hasattr(os, "sysconf"):
        result["physical_memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("doctor", help="Report local dependencies and resources")
    health.add_argument("--training", action="store_true")
    build = commands.add_parser("prepare", help="Dataset → BC/PPO → bundle → full local acceptance")
    build.add_argument("--workspace", type=Path, required=True)
    build.add_argument("--seed", type=int, default=11)
    build.add_argument("--bc-epochs", type=int, default=100)
    build.add_argument("--bc-input-mode", choices=BC_INPUT_MODES,
                       default="drop_previous_residual")
    build.add_argument("--ppo-episodes", type=int, default=32)
    build.add_argument("--resume", action="store_true")
    build.add_argument("--dry-run", action="store_true")
    package = commands.add_parser("package", help="Build an offline release from the bundle's Git commit")
    package.add_argument("--bundle", type=Path, required=True)
    package.add_argument("--expected-manifest-sha256", required=True)
    package.add_argument("--wheelhouse", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    for name in ("verify", "run", "accept"):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--expected-manifest-sha256", required=True)
        if name in ("run", "accept"):
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--dry-run", action="store_true")
            if name == "run":
                command.add_argument("--algorithm", choices=("bc", "ppo"), required=True)
                command.add_argument("--nominal", action="store_true")
        else:
            command.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(training=args.training)
            code = 0 if result["status"] == "PASS" else 1
        elif args.command == "prepare":
            options = {name: getattr(args, name)
                       for name in ("seed", "bc_epochs", "ppo_episodes", "bc_input_mode")}
            if args.dry_run:
                result = preview(**options)
                code = 0
            else:
                result = prepare(args.workspace, resume=args.resume, **options)
                code = 0 if (result["engineering_status"] == "PASS"
                             and all(result["policy_acceptance"].values())) else 2
        elif args.command == "package":
            from tools.learning_deployment.release import package_release

            result = package_release(args.bundle, args.expected_manifest_sha256,
                                     args.wheelhouse, args.output)
            code = 0
        elif args.command == "verify" or args.dry_run:
            manifest = verify_bundle(args.bundle, args.expected_manifest_sha256)
            result = {"status": "PASS", "source_commit": manifest["source_commit"],
                      "bundle_manifest_sha256": args.expected_manifest_sha256,
                      "target": manifest["target"], "new_simulations": 0}
            if args.command == "verify" and args.profile:
                result["profiles"] = {a: profile_actor(args.bundle, a) for a in ("bc", "ppo")}
            code = 0
            if "profiles" in result and not all(p["within_budget"] for p in result["profiles"].values()):
                code = 2
        elif args.command == "accept":
            result = accept_bundle(args.bundle, args.expected_manifest_sha256, args.output)
            code = 0 if (result["engineering_status"] == "PASS"
                         and all(result["policy_acceptance"].values())) else 2
        else:
            report = run_bundle(args.bundle, args.expected_manifest_sha256, args.output,
                                algorithm=args.algorithm, nominal=args.nominal)
            result = {"output": str(args.output), "acceptance_met": report["acceptance_met"],
                      "episodes_succeeded": report["all_episodes_succeeded"],
                      "case_count": report["expected_case_count"]}
            code = 0 if report["acceptance_met"] else 2
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return code
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(json.dumps({"status": "FAIL", "error_type": type(error).__name__,
                          "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
