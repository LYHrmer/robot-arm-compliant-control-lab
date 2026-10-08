"""Restartable preparation using the existing dataset, BC, PPO and evaluation code."""

import inspect
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_dataset import audit_dataset, collect_demonstrations
from compliant_control_lab.surface_policy_artifact import load_policy_artifact
from compliant_control_lab.surface_readiness_cases import preparation_cases
from tools.learning_deployment.bundle import (
    ALGORITHMS,
    build_bundle,
    cases_for,
    contract_for,
    profile_actor,
    run_bundle,
    summarize_evaluations,
    verify_bundle,
)
from tools.learning_deployment.common import (
    committed_source,
    output_path,
    read_json,
    runtime_identity,
    seal,
    sha256,
    verify_sealed,
    write_new,
)
from tools.surface_learning_pilot import THRESHOLDS
from tools.surface_mlp_actor import actor_from_artifact

BC_INPUT_MODES = ("full49", "drop_previous_residual")


def settings(*, seed=11, bc_epochs=100, ppo_episodes=32,
             bc_input_mode="drop_previous_residual"):
    for name, value in (("seed", seed), ("bc_epochs", bc_epochs), ("ppo_episodes", ppo_episodes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 1):
            raise ValueError(f"invalid {name}")
    if ppo_episodes % 4:
        raise ValueError("ppo_episodes must be divisible by four")
    if bc_input_mode not in BC_INPUT_MODES:
        raise ValueError(f"bc_input_mode must be one of {BC_INPUT_MODES}")
    return {"seed": seed, "bc_epochs": bc_epochs, "ppo_episodes": ppo_episodes,
            "bc_input_mode": bc_input_mode}


def preview(**options):
    configuration = settings(**options)
    cases = preparation_cases()
    return {"target": "mujoco_simulation_only", "settings": configuration,
            "cases": len(cases), "groups": len({c["group_id"] for c in cases}),
            "splits": {split: sum(c["split"] == split for c in cases)
                       for split in ("train", "validation", "development_test")},
            "stages": ["teacher_dataset", "bc_training", "ppo_training", "inference_bundle",
                       "export_parity", "bc", "bc_nominal", "ppo", "ppo_nominal", "acceptance"],
            "writes": False, "training_device": "cpu", "inference_requires_torch": False}


def _training_parameters(function, **options):
    """Resolve the trainer's own defaults, then call it with every value explicit."""
    signature = inspect.signature(function)
    bound = signature.bind(None, None, **options)
    bound.apply_defaults()
    return {name: bound.arguments[name] for name, parameter in signature.parameters.items()
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY}


def _bc_trainer(input_mode):
    if input_mode == "full49":
        from tools import train_surface_bc

        return train_surface_bc, train_surface_bc.train_bc
    from tools import train_surface_bc_transfer

    return train_surface_bc_transfer, train_surface_bc_transfer.train_bc_transfer


def _training_configuration(options, identity):
    from tools import train_surface_bc, train_surface_ppo

    bc_runtime = train_surface_bc._runtime_identity(train_surface_bc._require_torch())
    # Both trainers enter their deterministic CPU wrapper with one Torch thread.
    bc_runtime["torch_num_threads"] = 1
    runner = identity["runner"]
    bc_module, bc_train = _bc_trainer(options["bc_input_mode"])
    bc_options = {"seed": options["seed"], "epochs": options["bc_epochs"]}
    bc_manifest = {"schema": "surface_behavior_clone_run_v1",
                   "trainer_source_sha256": bc_module._source_identity(),
                   "runtime": bc_runtime}
    if options["bc_input_mode"] == "drop_previous_residual":
        bc_options["arm"] = options["bc_input_mode"]
        bc_manifest.update(schema="surface_behavior_clone_transfer_run_v1",
                           arm=options["bc_input_mode"],
                           input_mask=bc_module._input_mask(options["bc_input_mode"]).astype(int).tolist())
    return {
        "bc": {
            "parameters": _training_parameters(bc_train, **bc_options),
            "manifest": bc_manifest,
        },
        "ppo": {
            "parameters": _training_parameters(
                train_surface_ppo.train_ppo, seed=options["seed"],
                episodes=options["ppo_episodes"],
                checkpoint_episodes=(0, options["ppo_episodes"])),
            "manifest": {
                "source_and_assets_sha256": runner["package_source_and_assets_sha256"],
                "script_sha256": {
                    "train_surface_ppo.py": identity["deployment_sources"]["train_surface_ppo.py"],
                    "surface_mlp_actor.py": runner["runner_sha256"],
                },
                "runtime": {"device": "cpu", "dtype": "float64", "torch_num_threads": 1,
                            "deterministic_algorithms": True},
                "versions": {
                    "python": runner["python_version"], "numpy": runner["numpy_version"],
                    "torch": train_surface_ppo._require_torch().__version__,
                    "mujoco": runner["mujoco_version"], "gymnasium": runner["gymnasium_version"],
                },
            },
        },
    }


def build_plan(**options):
    """Describe this checkout's complete training invocation without writing or training."""
    options = settings(**options)
    identity = runtime_identity()
    plan = {"schema": "surface_simulation_deployment_plan_v1", "settings": options,
            "source_commit": committed_source(), "runtime_identity": identity,
            "cases": preparation_cases(), "thresholds": deepcopy(THRESHOLDS),
            "training": _training_configuration(options, identity),
            "policy_selection": {"bc": "minimum_validation_action_mse_earliest_tie",
                                 "ppo": f"fixed_final_episode_{options['ppo_episodes']}_no_dev_selection"}}
    # Persisted cases and checkpoint tuples use JSON lists, including on resumes.
    return json.loads(json.dumps(plan, allow_nan=False))


def _verify_training(manifest, algorithm, configuration):
    if any(manifest.get(name) != value for name, value in configuration["manifest"].items()):
        raise ValueError(f"{algorithm.upper()} training source or runtime differs from the frozen plan")
    for name, value in configuration["parameters"].items():
        if algorithm == "ppo" and name == "seed":
            actual = manifest.get("seed")
        elif algorithm == "bc" and name == "arm":
            actual = manifest.get("arm")
        else:
            field = "fixed_std" if algorithm == "ppo" and name == "std" else name
            actual = manifest["hyperparameters"].get(field)
        if actual != value:
            raise ValueError(f"{algorithm.upper()} training parameter {name} differs from the frozen plan")


def _export_parity(bundle, dataset):
    """Compare the exported NumPy actor with Torch using real measured observations."""
    import torch

    data = read_json(dataset / "manifest.json")
    first = data["episodes"][0]["file"]
    with np.load(dataset / first, allow_pickle=False) as trace:
        observations = np.asarray(trace["observations"][:256], dtype=np.float64)
    reports = {}
    for algorithm in ALGORITHMS:
        artifact = load_policy_artifact(
            bundle / f"{algorithm}.json",
            expected_contract=contract_for(data["cases"], algorithm),
        )
        actor, _ = actor_from_artifact(artifact)
        with torch.no_grad():
            values = torch.from_numpy(observations.copy())
            for index, layer in enumerate(artifact.payload["layers"]):
                values = torch.nn.functional.linear(
                    values, torch.tensor(layer["weights"], dtype=torch.float64),
                    torch.tensor(layer["bias"], dtype=torch.float64),
                )
                if index == len(artifact.payload["layers"]) - 1 or artifact.payload["activation"] == "tanh":
                    values = torch.tanh(values)
                else:
                    values = torch.relu(values)
        error = float(np.max(np.abs(np.stack([actor(x) for x in observations]) - values.numpy())))
        reports[algorithm] = {"observations": len(observations), "max_abs_error": error,
                              "tolerance": 1e-12, "passed": error <= 1e-12,
                              "reference": "Torch_float64_reconstruction_of_exported_weights"}
    if not all(row["passed"] for row in reports.values()):
        raise ValueError("exported NumPy/Torch inference parity failed")
    return reports


def prepare(workspace, *, resume=False, **options):
    workspace = output_path(workspace, allow_existing=True)
    plan = build_plan(**options)
    options, source, identity = plan["settings"], plan["source_commit"], plan["runtime_identity"]
    if workspace.exists():
        if not resume:
            raise FileExistsError("workspace exists; use --resume to verify and reuse completed stages")
        if read_json(workspace / "plan.json") != plan:
            raise ValueError("resume plan, source commit, configuration or runtime differs")
    else:
        workspace.mkdir(parents=True)
        write_new(workspace / "plan.json", plan)
    plan_hash = sha256(workspace / "plan.json")

    def stage(name):
        if runtime_identity() != identity or sha256(workspace / "plan.json") != plan_hash:
            raise RuntimeError("code/runtime/plan changed during preparation")
        print(f"stage: {name}", flush=True)
        return workspace / name

    dataset = stage("dataset")
    if not dataset.exists():
        collect_demonstrations(dataset, plan["cases"])
    audit_dataset(dataset)
    data = read_json(dataset / "manifest.json")
    if data["cases"] != plan["cases"] or data["source_and_assets_sha256"] != identity["runner"]["package_source_and_assets_sha256"]:
        raise ValueError("dataset cases/source differ from the deployment plan")

    from tools.train_surface_ppo import train_ppo

    _, train_bc = _bc_trainer(options["bc_input_mode"])
    bc = stage("bc_training")
    if not bc.exists():
        train_bc(dataset, bc, **plan["training"]["bc"]["parameters"])
    bc_manifest = verify_sealed(bc)
    _verify_training(bc_manifest, "bc", plan["training"]["bc"])
    if bc_manifest["dataset"]["manifest_sha256"] != sha256(dataset / "manifest.json"):
        raise ValueError("BC training does not match the frozen plan/dataset")

    ppo = stage("ppo_training")
    if not ppo.exists():
        train_ppo(cases_for(plan["cases"], "ppo"), ppo, **plan["training"]["ppo"]["parameters"])
    ppo_manifest = verify_sealed(ppo)
    _verify_training(ppo_manifest, "ppo", plan["training"]["ppo"])
    if ppo_manifest["case_plan"] != cases_for(plan["cases"], "ppo"):
        raise ValueError("PPO training does not match the frozen plan")
    candidates = {"bc": bc / bc_manifest["candidate_file"],
                  "ppo": ppo / ppo_manifest["checkpoints"][str(options["ppo_episodes"])]}
    training_hashes = {"bc": sha256(bc / "manifest.json"), "ppo": sha256(ppo / "manifest.json")}
    bundle = stage("bundle")
    if not bundle.exists():
        build_bundle(bundle, plan=plan, candidates=candidates, training_hashes=training_hashes)
    bundle_hash = sha256(bundle / "manifest.json")
    manifest = verify_bundle(bundle, bundle_hash)
    if manifest["training_manifest_sha256"] != training_hashes:
        raise ValueError("bundle training origin differs")
    for algorithm, candidate in candidates.items():
        if sha256(candidate) != manifest["artifact_sha256"][f"{algorithm}.json"]:
            raise ValueError("bundle model differs from the selected trained checkpoint")

    parity = _export_parity(bundle, dataset)
    profiles = {algorithm: profile_actor(bundle, algorithm) for algorithm in ALGORITHMS}
    reports, evaluation_hashes = {}, {}
    for algorithm in ALGORITHMS:
        for nominal in (False, True):
            label = f"{algorithm}_nominal" if nominal else algorithm
            output = stage(f"evaluation_{label}")
            if not output.exists():
                run_bundle(bundle, bundle_hash, output, algorithm=algorithm, nominal=nominal)
            evaluated = verify_sealed(output)
            if (evaluated["protocol_sha256"] != manifest["protocol_sha256"][label]
                    or evaluated["candidate_artifact_sha256"] != manifest["artifact_sha256"][f"{label}.json"]):
                raise ValueError("completed evaluation belongs to a different candidate/protocol")
            reports[label] = read_json(output / "report.json")
            evaluation_hashes[label] = sha256(output / "manifest.json")
    summary = summarize_evaluations(reports, profiles)
    summary.update(bundle_manifest_sha256=bundle_hash, plan_sha256=plan_hash, export_parity=parity,
                   evaluation_manifest_sha256=evaluation_hashes, source_commit=source)
    summary["training"] = {"bc_selected_epoch": bc_manifest["selected_epoch"],
                           "bc_input_mode": options["bc_input_mode"],
                           "ppo_episodes": options["ppo_episodes"],
                           "ppo_failed_training_episodes": ppo_manifest["failed_episodes"]}
    acceptance = stage("acceptance")
    if acceptance.exists():
        recorded = verify_sealed(acceptance)
        if recorded["bundle_manifest_sha256"] != bundle_hash:
            raise ValueError("acceptance belongs to a different bundle")
        # Keep the original latency measurements immutable on a resume.
        summary = read_json(acceptance / "report.json")
        if summary["evaluation_manifest_sha256"] != evaluation_hashes:
            raise ValueError("acceptance evaluation identities differ")
    else:
        with tempfile.TemporaryDirectory(prefix=".deployment-acceptance-", dir=workspace) as temporary:
            staging = Path(temporary) / "acceptance"
            staging.mkdir()
            write_new(staging / "report.json", summary)
            seal(staging, {"schema": "surface_deployment_acceptance_v1",
                           "bundle_manifest_sha256": bundle_hash, "plan_sha256": plan_hash})
            if acceptance.exists():
                raise FileExistsError("acceptance destination appeared during preparation")
            os.rename(staging, acceptance)
    return summary
