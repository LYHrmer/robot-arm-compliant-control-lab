"""Deployment boundaries: artifact identity, relocation, fail-closed CLI and resume."""

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from compliant_control_lab.surface_readiness_cases import preparation_cases
from tools.learning_deployment import bundle, cli, common, install_release, pipeline
from tools.surface_learning_pilot import THRESHOLDS
from tools.surface_mlp_actor import save_mlp_candidate


@pytest.fixture
def deployment(tmp_path):
    cases = preparation_cases()
    candidates = {}
    for algorithm in bundle.ALGORITHMS:
        candidate = tmp_path / f"input_{algorithm}.json"
        layers = [{"weights": np.full((4, 49), 0.001).tolist(), "bias": [0.01] * 4},
                  {"weights": np.full((3, 4), 0.002).tolist(), "bias": [0.0] * 3}]
        save_mlp_candidate(candidate, layers, contract=bundle.contract_for(cases, algorithm))
        candidates[algorithm] = candidate
    plan = {"cases": cases, "thresholds": THRESHOLDS, "source_commit": "a" * 40,
            "runtime_identity": common.runtime_identity(), "policy_selection": {"test_fixture": True}}
    output = tmp_path / "bundle"
    digest = bundle.build_bundle(output, plan=plan, candidates=candidates,
                                 training_hashes={"bc": "b" * 64, "ppo": "c" * 64})
    return output, digest


def test_bundle_relocates_without_repository_or_torch(deployment, tmp_path):
    source, digest = deployment
    target = tmp_path / "relocated"
    shutil.copytree(source, target)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    script = """
import importlib.abc, sys
class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('inference imported torch')
sys.meta_path.insert(0, RejectTorch())
from tools.learning_deployment.bundle import verify_bundle, profile_actor
from tools.learning_deployment.cli import main
verify_bundle(sys.argv[1], sys.argv[2])
assert main(['verify', '--bundle', sys.argv[1], '--expected-manifest-sha256', sys.argv[2]]) == 0
for algorithm in ('bc', 'ppo'):
    assert profile_actor(sys.argv[1], algorithm)['samples'] == 512
assert 'torch' not in sys.modules
print('PASS')
"""
    result = subprocess.run([sys.executable, "-c", script, str(target), digest],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines()[-1] == "PASS"


def test_external_pin_rejects_resealed_model(deployment):
    directory, digest = deployment
    path = directory / "bc.json"
    path.write_text(path.read_text() + " ")
    manifest = common.read_json(directory / "manifest.json")
    manifest["artifact_sha256"]["bc.json"] = common.sha256(path)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "COMPLETE").write_text(common.sha256(directory / "manifest.json"))
    with pytest.raises(ValueError, match="externally pinned"):
        bundle.verify_bundle(directory, digest)


def test_changed_model_rejected_before_any_simulation(deployment, tmp_path, monkeypatch):
    directory, digest = deployment
    (directory / "ppo.json").write_text("{}")
    monkeypatch.setattr(bundle, "evaluate_candidate", lambda *a, **kw: pytest.fail("ran simulation"))
    with pytest.raises(ValueError, match="artifact changed"):
        bundle.run_bundle(directory, digest, tmp_path / "run", algorithm="ppo")


def test_extra_file_or_symlink_is_rejected(deployment, tmp_path):
    directory, digest = deployment
    extra = directory / "unexpected"
    extra.write_text("x")
    with pytest.raises(ValueError, match="unexpected or missing"):
        bundle.verify_bundle(directory, digest)
    extra.unlink()
    extra.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        bundle.verify_bundle(directory, digest)


def test_runtime_drift_is_rejected(deployment, monkeypatch):
    directory, digest = deployment
    monkeypatch.setattr(bundle, "runtime_identity", lambda: {"changed": True})
    with pytest.raises(ValueError, match="runtime/source differs"):
        bundle.verify_bundle(directory, digest)


def test_bc_and_ppo_contracts_are_not_interchangeable(deployment):
    directory, _ = deployment
    from compliant_control_lab.surface_policy_artifact import load_policy_artifact

    with pytest.raises(ValueError, match="policy contract differs"):
        load_policy_artifact(directory / "bc.json", expected_contract=bundle.contract_for(
            common.read_json(directory / "cases.json"), "ppo"))


def test_prepare_dry_run_writes_nothing(tmp_path, capsys):
    target = tmp_path / "workspace"
    assert cli.main(["prepare", "--workspace", str(target), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["splits"] == {"train": 16, "validation": 4, "development_test": 4}
    assert report["settings"]["bc_input_mode"] == "drop_previous_residual"
    assert not target.exists()


def test_prepare_dry_run_preserves_explicit_full49(tmp_path, capsys):
    target = tmp_path / "workspace"
    assert cli.main(["prepare", "--workspace", str(target), "--bc-input-mode", "full49",
                     "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["settings"]["bc_input_mode"] == "full49"
    assert not target.exists()


def test_run_dry_run_does_not_create_output(deployment, tmp_path, capsys):
    directory, digest = deployment
    output = tmp_path / "run"
    assert cli.main(["run", "--bundle", str(directory), "--expected-manifest-sha256", digest,
                     "--algorithm", "bc", "--output", str(output), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["new_simulations"] == 0
    assert not output.exists()


def test_cli_reports_policy_failure_with_nonzero_exit(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "prepare", lambda *a, **kw: {
        "engineering_status": "PASS", "policy_acceptance": {"bc": False, "ppo": True},
    })
    assert cli.main(["prepare", "--workspace", str(tmp_path / "run")]) == 2
    assert json.loads(capsys.readouterr().out)["policy_acceptance"]["bc"] is False


@pytest.mark.parametrize("options", ({"seed": -1}, {"bc_epochs": 0}, {"ppo_episodes": 3},
                                    {"bc_input_mode": "teacher_inputs"}))
def test_invalid_training_configuration_fails_before_writes(options):
    with pytest.raises(ValueError):
        pipeline.preview(**options)


@pytest.fixture
def training_runtime(monkeypatch):
    """Stage orchestration tests use fake runtime metadata, never optional Torch."""
    from tools import train_surface_bc, train_surface_ppo

    torch = SimpleNamespace(__version__="test-fixture")
    monkeypatch.setattr(train_surface_bc, "_require_torch", lambda: torch)
    monkeypatch.setattr(train_surface_ppo, "_require_torch", lambda: torch)
    monkeypatch.setattr(train_surface_bc, "_runtime_identity", lambda _: {
        "torch_version": torch.__version__, "torch_num_threads": 1,
    })


def test_resume_refuses_changed_plan_before_training(tmp_path, monkeypatch, training_runtime):
    monkeypatch.setattr(pipeline, "committed_source", lambda: "a" * 40)
    common.write_new(tmp_path / "plan.json", {"tampered": True})
    monkeypatch.setattr(pipeline, "collect_demonstrations", lambda *a: pytest.fail("started training"))
    with pytest.raises(ValueError, match="resume plan"):
        pipeline.prepare(tmp_path, resume=True)


def test_summary_keeps_engineering_and_scientific_decisions_separate():
    base = {"all_episodes_succeeded": True, "acceptance_met": True,
            "runs": [{"case_id": "case1", "tangent_rmse_mm": 2.0, "force_rmse_n": 1.0}]}
    reports = {name: deepcopy(base) for name in ("bc", "bc_nominal", "ppo", "ppo_nominal")}
    reports["bc"]["acceptance_met"] = False
    reports["bc"]["runs"][0]["tangent_rmse_mm"] = 2.5
    summary = bundle.summarize_evaluations(reports, {name: {"within_budget": True} for name in ("bc", "ppo")})
    assert summary["engineering_status"] == "PASS"
    assert not summary["policy_acceptance"]["bc"]
    assert summary["paired_comparisons"]["bc"][0]["tangent_rmse_mm_delta_vs_nominal"] == 0.5
    reports["ppo"]["all_episodes_succeeded"] = False
    profiles = {name: {"within_budget": True} for name in ("bc", "ppo")}
    assert bundle.summarize_evaluations(reports, profiles)["engineering_status"] == "FAIL"


@pytest.mark.parametrize("profiles", ({}, {"bc": {"within_budget": True}},
                                    {"ppo": {"within_budget": True}}))
def test_summary_refuses_missing_inference_profiles(profiles):
    report = {"all_episodes_succeeded": True, "acceptance_met": True,
              "runs": [{"case_id": "case1", "tangent_rmse_mm": 2.0, "force_rmse_n": 1.0}]}
    reports = {name: deepcopy(report) for name in ("bc", "bc_nominal", "ppo", "ppo_nominal")}
    with pytest.raises(ValueError, match="inference profile"):
        bundle.summarize_evaluations(reports, profiles)


@pytest.mark.parametrize("label", ("bc", "bc_nominal", "ppo", "ppo_nominal"))
def test_summary_refuses_duplicate_cases(label):
    report = {"all_episodes_succeeded": True, "acceptance_met": True,
              "runs": [{"case_id": "case1", "tangent_rmse_mm": 2.0, "force_rmse_n": 1.0}]}
    reports = {name: deepcopy(report) for name in ("bc", "bc_nominal", "ppo", "ppo_nominal")}
    reports[label]["runs"].append(deepcopy(reports[label]["runs"][0]))
    with pytest.raises(ValueError, match="duplicate case IDs"):
        bundle.summarize_evaluations(reports, {a: {"within_budget": True} for a in ("bc", "ppo")})


@pytest.fixture
def preparation(tmp_path, monkeypatch, training_runtime):
    from tools import train_surface_bc, train_surface_bc_transfer, train_surface_ppo

    monkeypatch.setattr(pipeline, "committed_source", lambda: "a" * 40)
    training = pipeline.build_plan()["training"]
    calls = []

    def collect(path, cases):
        calls.append("dataset")
        path.mkdir()
        common.write_new(path / "manifest.json", {"cases": cases,
            "source_and_assets_sha256": common.runtime_identity()["runner"]["package_source_and_assets_sha256"]})

    def candidate(path, cases, algorithm):
        save_mlp_candidate(path, [
            {"weights": np.zeros((4, 49)).tolist(), "bias": [0.0] * 4},
            {"weights": np.zeros((3, 4)).tolist(), "bias": [0.0] * 3},
        ], contract=bundle.contract_for(cases, algorithm))

    def publish_bc(dataset, output, parameters, input_mode):
        calls.append(f"bc_{input_mode}")
        output.mkdir()
        candidate(output / "selected_model.json", common.read_json(dataset / "manifest.json")["cases"], "bc")
        configuration = pipeline.build_plan(bc_input_mode=input_mode)["training"]["bc"]
        common.seal(output, {**configuration["manifest"],
            "hyperparameters": {k: v for k, v in parameters.items() if k != "arm"},
            "dataset": {"manifest_sha256": common.sha256(dataset / "manifest.json")},
            "candidate_file": "selected_model.json", "selected_epoch": 1})

    @wraps(train_surface_bc.train_bc)
    def train_bc(dataset, output, **parameters):
        publish_bc(dataset, output, parameters, "full49")

    @wraps(train_surface_bc_transfer.train_bc_transfer)
    def train_bc_transfer(dataset, output, **parameters):
        publish_bc(dataset, output, parameters, "drop_previous_residual")

    @wraps(train_surface_ppo.train_ppo)
    def train_ppo(cases, output, **parameters):
        calls.append("ppo")
        output.mkdir()
        candidate(output / "checkpoint.json", cases, "ppo")
        hyperparameters = {**parameters, "fixed_std": parameters["std"]}
        del hyperparameters["seed"], hyperparameters["std"]
        common.seal(output, {**training["ppo"]["manifest"], "seed": parameters["seed"],
            "hyperparameters": hyperparameters, "case_plan": cases,
            "checkpoints": {str(parameters["episodes"]): "checkpoint.json"}, "failed_episodes": 0})

    def evaluate(bundle_path, digest, output, *, algorithm, nominal):
        calls.append(f"evaluate_{algorithm}_{nominal}")
        output.mkdir()
        report = {"all_episodes_succeeded": True, "acceptance_met": True,
                  "runs": [{"case_id": "case1", "tangent_rmse_mm": 2.0, "force_rmse_n": 1.0}]}
        common.write_new(output / "report.json", report)
        manifest = common.read_json(bundle_path / "manifest.json")
        label = f"{algorithm}_nominal" if nominal else algorithm
        common.seal(output, {"protocol_sha256": manifest["protocol_sha256"][label],
            "candidate_artifact_sha256": manifest["artifact_sha256"][f"{label}.json"]})
        return report

    monkeypatch.setattr(pipeline, "collect_demonstrations", collect)
    monkeypatch.setattr(pipeline, "audit_dataset", lambda *a: None)
    monkeypatch.setattr(train_surface_bc, "train_bc", train_bc)
    monkeypatch.setattr(train_surface_bc_transfer, "train_bc_transfer", train_bc_transfer)
    monkeypatch.setattr(train_surface_ppo, "train_ppo", train_ppo)
    monkeypatch.setattr(pipeline, "_export_parity", lambda *a: {"test": True})
    monkeypatch.setattr(pipeline, "profile_actor", lambda *a: {"within_budget": True})
    monkeypatch.setattr(pipeline, "run_bundle", evaluate)
    return tmp_path / "workspace", calls


@pytest.mark.parametrize("input_mode", pipeline.BC_INPUT_MODES)
def test_prepare_resume_checks_real_stage_contracts_without_retraining(preparation, input_mode):
    work, calls = preparation
    original = pipeline.prepare(work, bc_input_mode=input_mode)
    assert calls == ["dataset", f"bc_{input_mode}", "ppo", "evaluate_bc_False", "evaluate_bc_True",
                     "evaluate_ppo_False", "evaluate_ppo_True"]
    calls.clear()
    assert pipeline.prepare(work, resume=True, bc_input_mode=input_mode) == original
    assert not calls


def test_resume_refuses_a_different_bc_input_mode(preparation):
    work, calls = preparation
    pipeline.prepare(work)
    calls.clear()
    with pytest.raises(ValueError, match="resume plan"):
        pipeline.prepare(work, resume=True, bc_input_mode="full49")
    assert not calls


@pytest.mark.parametrize("field,value", (
    ("schema", "surface_behavior_clone_run_v1"),
    ("arm", "teacher_inputs"),
    ("input_mask", [1] * 49),
))
def test_resume_refuses_resealed_bc_input_identity(preparation, field, value):
    work, calls = preparation
    pipeline.prepare(work)
    calls.clear()
    directory = work / "bc_training"
    manifest = common.read_json(directory / "manifest.json")
    manifest[field] = value
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "COMPLETE").write_text(common.sha256(directory / "manifest.json"))
    with pytest.raises(ValueError, match="training source or runtime differs"):
        pipeline.prepare(work, resume=True)
    assert not calls


@pytest.mark.parametrize("algorithm,section,field", (
    ("bc", "trainer_source_sha256", "trainer"),
    ("bc", "trainer_source_sha256", "transfer_trainer"),
    ("bc", "runtime", "torch_version"),
    ("ppo", "script_sha256", "train_surface_ppo.py"),
    ("ppo", "versions", "torch"),
    ("ppo", "runtime", "deterministic_algorithms"),
))
def test_resume_refuses_resealed_training_source_or_runtime(preparation, algorithm, section, field):
    work, calls = preparation
    pipeline.prepare(work)
    calls.clear()
    directory = work / f"{algorithm}_training"
    manifest = common.read_json(directory / "manifest.json")
    manifest[section][field] = "different"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "COMPLETE").write_text(common.sha256(directory / "manifest.json"))
    with pytest.raises(ValueError, match="training source or runtime differs"):
        pipeline.prepare(work, resume=True)
    assert not calls


@pytest.mark.parametrize("algorithm,field", (
    ("bc", "batch_size"), ("bc", "learning_rate"), ("bc", "overfit_steps"),
    ("ppo", "episodes_per_update"), ("ppo", "checkpoint_episodes"), ("ppo", "gamma"),
    ("ppo", "gae_lambda"), ("ppo", "clip_epsilon"), ("ppo", "learning_rate"),
    ("ppo", "update_epochs"), ("ppo", "batch_size"), ("ppo", "max_grad_norm"),
    ("ppo", "fixed_std"), ("ppo", "target_kl"),
))
def test_resume_refuses_resealed_training_parameters(preparation, algorithm, field):
    work, calls = preparation
    pipeline.prepare(work)
    calls.clear()
    directory = work / f"{algorithm}_training"
    manifest = common.read_json(directory / "manifest.json")
    value = manifest["hyperparameters"][field]
    manifest["hyperparameters"][field] = [] if isinstance(value, list) else value * 2
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "COMPLETE").write_text(common.sha256(directory / "manifest.json"))
    with pytest.raises(ValueError, match="training parameter .* differs"):
        pipeline.prepare(work, resume=True)
    assert not calls


@pytest.mark.parametrize("input_mode", pipeline.BC_INPUT_MODES)
def test_plan_records_all_actual_training_parameters(preparation, input_mode):
    work, _ = preparation
    pipeline.prepare(work, bc_input_mode=input_mode)
    plan = common.read_json(work / "plan.json")
    parameters = {"seed", "epochs", "batch_size", "learning_rate", "overfit_steps"}
    if input_mode == "drop_previous_residual":
        parameters.add("arm")
        bc = plan["training"]["bc"]
        assert bc["parameters"]["arm"] == bc["manifest"]["arm"] == input_mode
        assert [i for i, active in enumerate(bc["manifest"]["input_mask"]) if not active] == [14, 15, 16]
    assert set(plan["training"]["bc"]["parameters"]) == parameters
    assert plan["settings"]["bc_input_mode"] == input_mode
    sources = plan["runtime_identity"]["deployment_sources"]
    trainer = Path(__file__).resolve().parents[1] / "tools/train_surface_bc_transfer.py"
    assert sources[trainer.name] == common.sha256(trainer)
    assert set(plan["training"]["ppo"]["parameters"]) == {
        "seed", "episodes", "episodes_per_update", "checkpoint_episodes", "gamma", "gae_lambda",
        "clip_epsilon", "learning_rate", "update_epochs", "batch_size", "max_grad_norm", "std", "target_kl",
    }


def test_acceptance_interruption_can_resume_without_partial_final_directory(preparation, monkeypatch):
    work, calls = preparation
    original_seal = pipeline.seal

    def interrupted_seal(directory, metadata):
        raise RuntimeError("interrupted before acceptance publication")

    monkeypatch.setattr(pipeline, "seal", interrupted_seal)
    with pytest.raises(RuntimeError, match="interrupted before acceptance"):
        pipeline.prepare(work)
    assert not (work / "acceptance").exists()
    assert not list(work.glob(".deployment-acceptance-*"))
    calls.clear()
    monkeypatch.setattr(pipeline, "seal", original_seal)
    result = pipeline.prepare(work, resume=True)
    assert result["engineering_status"] == "PASS"
    common.verify_sealed(work / "acceptance")
    assert not calls


def test_installer_dry_run_is_offline_and_preserves_existing_environments(tmp_path, monkeypatch, capsys):
    import platform

    release = tmp_path / "release"
    release.mkdir()
    (release / "install.py").write_text("# test fixture\n")
    (release / "runtime.lock").write_text("# test fixture\n")
    digest = common.seal(release, {"schema": "surface_offline_release_v1",
                                  "python_version": platform.python_version(), "source_commit": "a" * 40})
    monkeypatch.setattr(install_release, "__file__", str(release / "install.py"))
    monkeypatch.setattr(install_release.subprocess, "run", lambda *a, **kw: pytest.fail("executed subprocess"))
    target = tmp_path / "env"
    arguments = ["--venv", str(target), "--expected-release-sha256", digest, "--dry-run"]
    assert install_release.main(arguments) == 0
    assert not target.exists()
    assert json.loads(capsys.readouterr().out)["writes"] is False
    target.mkdir()
    (target / "existing-user-data").write_text("keep")
    assert install_release.main(arguments) == 1
    assert (target / "existing-user-data").read_text() == "keep"
    assert "another release" in capsys.readouterr().err


def test_runtime_lock_omits_training_and_dev_packages_and_uses_core_hashes():
    import re

    root = Path(__file__).resolve().parents[1]
    runtime = (root / "environment/deployment-runtime.lock").read_text()
    core = (root / "environment/core.lock").read_text()
    assert all(digest in core for digest in re.findall(r"sha256:[0-9a-f]{64}", runtime))
    for unwanted in ("torch", "pytest", "ruff", "setuptools", "wheel", "pip"):
        assert re.search(rf"^{unwanted}==", runtime, re.MULTILINE) is None
