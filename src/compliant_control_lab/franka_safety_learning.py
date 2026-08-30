"""Freeze, train and evaluate torque-safe Residual RL without peeking at the blind set."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import platform
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from compliant_control_lab.franka_adaptive import (
    FrankaAdaptiveHybridController,
    FrankaSafeAdaptiveController,
)
from compliant_control_lab.franka_control import FrankaController, FrankaHybridController
from compliant_control_lab.franka_learning import (
    ArsTrainingConfig,
    TrainingRecord,
    _plot_training,
    _write_csv,
    train_residual_policy,
)
from compliant_control_lab.franka_simulation import (
    FrankaScenario,
    FrankaSimulationConfig,
    franka_model_path,
    run_franka_trial,
)
from compliant_control_lab.franka_stress import (
    ResidualRlGate,
    failed_gate_checks,
    sample_stress_scenarios,
    scenario_values,
)
from compliant_control_lab.residual_rl import (
    TORQUE_AWARE_OBSERVATION_NAMES,
    LinearResidualPolicy,
    TorqueProjectedResidualController,
)

EXPERIMENT_ID = "franka-torque-safe-residual-v0.5"
FREEZE_TAG = "v0.5-preholdout"
PROTOCOL_FORMAT_VERSION = 3
TRAINING_POLICY_SEEDS = (17, 23, 31, 43, 59)
TRAINING_SIMULATION_SEEDS = (10_001, 20_001, 30_001, 40_001, 50_001)
TRAINING_SCENARIO_SEED = 101
DEVELOPMENT_SCENARIO_SEED = 211
DEVELOPMENT_SIMULATION_SEED = 70_001
TRAINING_CASES = 8
DEVELOPMENT_CASES = 8
BLIND_CASES = 48
BLIND_DURATION = 4.5
BLIND_REQUIRED_PASSES = 44
SEED_DERIVATION = "hmac-sha256-v1"
MIN_BEACON_LEAD_SECONDS = 600
MAX_REFERENCE_ROUND_SKEW = 1
QUICKNET_CHAIN_INFO = {
    "public_key": (
        "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c8"
        "c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb5"
        "ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
    ),
    "period": 3,
    "genesis_time": 1_692_803_367,
    "hash": "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971",
    "groupHash": "f477d5c89f21a17c863a7f937c6a6d15859414d2be09cd448d4279af331c5d3e",
    "schemeID": "bls-unchained-g1-rfc9380",
    "metadata": {"beaconID": "quicknet"},
}
BEACON_RELAYS = ("https://api.drand.sh", "https://drand.cloudflare.com")
BEACON_VERIFIER = Path("tools/verify_drand_beacon.mjs")
BEACON_LOCKFILE = Path("package-lock.json")
BEACON_VERIFIER_NAME = "drand-client-js"
BEACON_VERIFIER_VERSION = "1.4.2"
BLIND_CONTROLLER_CONTRACT = (
    "fixed_hybrid",
    "adaptive_hybrid",
    "safe_adaptive_hybrid",
    "five_frozen_torque_projected_residual_policies",
)


@dataclass(frozen=True)
class PolicyArtifact:
    run_index: int
    policy_seed: int
    simulation_seed: int
    policy_file: str
    policy_sha256: str
    training_curve_file: str
    training_curve_sha256: str
    training_plot_file: str
    training_plot_sha256: str


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def derive_seed(root: bytes, namespace: str, index: int) -> int:
    """Derive a stable uint64 seed without reusing another namespace's stream."""
    if len(root) < 16:
        raise ValueError("seed root must contain at least 16 bytes")
    if not namespace or index < 0:
        raise ValueError("namespace must be nonempty and index must be nonnegative")
    message = f"v0.5/{namespace}/{index}".encode()
    digest = hmac.new(root, message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _scenario_digest(scenario: FrankaScenario) -> str:
    values = asdict(scenario)
    values.pop("name")
    return _json_sha256(values)


def _assert_disjoint_scenarios(*groups: tuple[FrankaScenario, ...]) -> None:
    digest_sets = [{_scenario_digest(scenario) for scenario in group} for group in groups]
    for left_index, left in enumerate(digest_sets):
        for right in digest_sets[left_index + 1 :]:
            if left & right:
                raise ValueError("training, development and blind scenarios must be disjoint")


def _development_scenarios() -> tuple[FrankaScenario, ...]:
    return tuple(
        FrankaScenario(name=f"development_{index:02d}", **{
            key: value
            for key, value in asdict(scenario).items()
            if key != "name"
        })
        for index, scenario in enumerate(
            sample_stress_scenarios(DEVELOPMENT_CASES, DEVELOPMENT_SCENARIO_SEED)
        )
    )


def _training_scenarios() -> tuple[FrankaScenario, ...]:
    return tuple(
        FrankaScenario(name=f"training_{index:02d}", **{
            key: value
            for key, value in asdict(scenario).items()
            if key != "name"
        })
        for index, scenario in enumerate(
            sample_stress_scenarios(TRAINING_CASES, TRAINING_SCENARIO_SEED)
        )
    )


def torque_safe_controller_factory(
    policy: LinearResidualPolicy,
) -> TorqueProjectedResidualController:
    return TorqueProjectedResidualController(
        policy=policy,
        nominal=FrankaSafeAdaptiveController(),
    )


def torque_safety_manifest() -> dict[str, Any]:
    nominal = FrankaSafeAdaptiveController()
    residual = TorqueProjectedResidualController()
    return {
        "reference_governor": {
            "max_normal_lead_m": nominal.max_normal_lead,
            "max_approach_velocity_m_s": nominal.max_approach_velocity,
            "impact_force_margin_n": nominal.impact_force_margin,
            "impact_force_rate_n_s": nominal.impact_force_rate,
        },
        "nominal_torque_reserve_fraction": nominal.torque_reserve_fraction,
        "residual": {
            "action_bounds_n": residual.action_bounds.tolist(),
            "action_rate_limits_n_s": residual.action_rate_limits.tolist(),
            "policy_period_s": residual.policy_period,
            "filter_time_constant_s": residual.filter_time_constant,
            "residual_enable_delay_s": residual.residual_enable_delay,
            "force_guard_margin_n": residual.force_guard_margin,
            "force_guard_rate_n_s": residual.force_guard_rate,
            "torque_reserve_fraction": residual.torque_reserve_fraction,
            "inference_deadline_us": residual.inference_deadline_us,
        },
        "observation_names": list(TORQUE_AWARE_OBSERVATION_NAMES),
    }


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _require_clean_repository() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("freeze/evaluate requires a clean Git worktree")


def _require_repository_unchanged(
    implementation_commit: str,
    model_sha256: str,
    torque_safety_sha256: str,
) -> None:
    if _git("rev-parse", "HEAD") != implementation_commit:
        raise RuntimeError("repository HEAD changed while policies were training")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked files changed while policies were training")
    if _sha256(franka_model_path()) != model_sha256:
        raise RuntimeError("Franka model changed while policies were training")
    if _json_sha256(torque_safety_manifest()) != torque_safety_sha256:
        raise RuntimeError("torque-safety settings changed while policies were training")


def beacon_round_time(beacon_round: int) -> int:
    if beacon_round <= 0:
        raise ValueError("beacon round must be positive")
    return int(QUICKNET_CHAIN_INFO["genesis_time"]) + int(
        QUICKNET_CHAIN_INFO["period"]
    ) * (beacon_round - 1)


def _utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _require_future_beacon(beacon_round: int, *, reference_round: int) -> int:
    if reference_round <= 0:
        raise ValueError("reference_round must be positive")
    scheduled_time = beacon_round_time(beacon_round)
    guaranteed_lead_seconds = (beacon_round - reference_round - 1) * int(
        QUICKNET_CHAIN_INFO["period"]
    )
    if guaranteed_lead_seconds < MIN_BEACON_LEAD_SECONDS:
        raise ValueError(
            "beacon round must remain unpublished for at least "
            f"{MIN_BEACON_LEAD_SECONDS} seconds"
        )
    return scheduled_time


def _repository_relative_path(path: Path) -> str:
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact is outside the repository: {path}") from error


def _require_file_at_tag(path: Path, tag: str) -> None:
    relative_path = _repository_relative_path(path)
    try:
        frozen_bytes = _git_bytes("show", f"{tag}:{relative_path}")
    except subprocess.CalledProcessError as error:
        raise ValueError(f"{relative_path} is not present in freeze tag {tag}") from error
    if frozen_bytes != path.read_bytes():
        raise ValueError(f"{relative_path} differs from freeze tag {tag}")


def _training_config(run_index: int) -> ArsTrainingConfig:
    return ArsTrainingConfig(
        iterations=8,
        directions=6,
        top_directions=3,
        training_cases=TRAINING_CASES,
        duration=3.2,
        scenario_seed=TRAINING_SCENARIO_SEED,
        simulation_seed=TRAINING_SIMULATION_SEEDS[run_index],
        policy_seed=TRAINING_POLICY_SEEDS[run_index],
    )


def _save_training_curve(records: list[TrainingRecord], csv_path: Path, plot_path: Path) -> None:
    _write_csv([asdict(record) for record in records], csv_path)
    _plot_training(records, plot_path)


def train_policy_run(run_index: int, output_dir: Path) -> PolicyArtifact:
    if not 0 <= run_index < len(TRAINING_POLICY_SEEDS):
        raise ValueError("run_index is outside the frozen training seed list")
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = _development_scenarios()
    config = _training_config(run_index)
    policy, records, scenarios = train_residual_policy(
        config,
        controller_factory=torque_safe_controller_factory,
        observation_names=TORQUE_AWARE_OBSERVATION_NAMES,
        validation_scenarios=validation,
        validation_simulation_seed=DEVELOPMENT_SIMULATION_SEED,
        resample_simulation_noise=True,
    )
    _assert_disjoint_scenarios(scenarios, validation)

    policy_path = output_dir / f"policy_{run_index:02d}.json"
    curve_path = output_dir / f"training_curve_{run_index:02d}.csv"
    plot_path = output_dir / f"training_curve_{run_index:02d}.png"
    policy.save(
        policy_path,
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "algorithm": "augmented_random_search",
            "training_run_index": run_index,
            "training_config": asdict(config),
            "development_scenario_seed": DEVELOPMENT_SCENARIO_SEED,
            "development_simulation_seed_base": DEVELOPMENT_SIMULATION_SEED,
            "resample_simulation_noise_each_iteration": True,
            "torque_safety": torque_safety_manifest(),
        },
    )
    _save_training_curve(records, curve_path, plot_path)
    return PolicyArtifact(
        run_index=run_index,
        policy_seed=config.policy_seed,
        simulation_seed=config.simulation_seed,
        policy_file=policy_path.name,
        policy_sha256=_sha256(policy_path),
        training_curve_file=curve_path.name,
        training_curve_sha256=_sha256(curve_path),
        training_plot_file=plot_path.name,
        training_plot_sha256=_sha256(plot_path),
    )


def _train_worker(arguments: tuple[int, str]) -> PolicyArtifact:
    run_index, output_dir = arguments
    return train_policy_run(run_index, Path(output_dir))


def _dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "node": subprocess.run(
            ["node", "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "drand-client": BEACON_VERIFIER_VERSION,
    }


def prepare_preholdout(
    output_dir: Path,
    beacon_chain_hash: str,
    beacon_round: int,
    jobs: int = 1,
) -> Path:
    """Train five policies and write the immutable pre-holdout protocol."""
    _require_clean_repository()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if beacon_chain_hash != QUICKNET_CHAIN_INFO["hash"]:
        raise ValueError("beacon_chain_hash must be the pinned drand Quicknet chain")
    if not 1 <= jobs <= len(TRAINING_POLICY_SEEDS):
        raise ValueError("jobs must be between one and five")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    repository_root = Path(_git("rev-parse", "--show-toplevel"))
    implementation_commit = _git("rev-parse", "HEAD")
    model_sha256 = _sha256(franka_model_path())
    torque_safety_sha256 = _json_sha256(torque_safety_manifest())
    verifier_path = repository_root / BEACON_VERIFIER
    lockfile_path = repository_root / BEACON_LOCKFILE
    verifier_sha256 = _sha256(verifier_path)
    lockfile_sha256 = _sha256(lockfile_path)
    starting_reference, _ = fetch_verified_reference_beacon(verifier_path)
    _require_future_beacon(
        beacon_round,
        reference_round=int(starting_reference["round"]),
    )
    with tempfile.TemporaryDirectory(prefix="franka-v05-prepare-", dir=output_dir.parent) as tmp:
        temporary_dir = Path(tmp)
        arguments = [(index, str(temporary_dir)) for index in range(len(TRAINING_POLICY_SEEDS))]
        if jobs == 1:
            artifacts = [_train_worker(item) for item in arguments]
        else:
            with ProcessPoolExecutor(max_workers=min(jobs, len(arguments))) as executor:
                artifacts = list(executor.map(_train_worker, arguments))
        artifacts.sort(key=lambda artifact: artifact.run_index)

        _require_repository_unchanged(
            implementation_commit,
            model_sha256,
            torque_safety_sha256,
        )
        if _sha256(verifier_path) != verifier_sha256 or _sha256(lockfile_path) != lockfile_sha256:
            raise RuntimeError("beacon verifier files changed while policies were training")
        final_reference, reference_audit = fetch_verified_reference_beacon(verifier_path)
        scheduled_time = _require_future_beacon(
            beacon_round,
            reference_round=int(final_reference["round"]),
        )
        reference_time = beacon_round_time(int(final_reference["round"]))

        training = _training_scenarios()
        development = _development_scenarios()
        _assert_disjoint_scenarios(training, development)
        protocol = {
            "format_version": PROTOCOL_FORMAT_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "preholdout_frozen",
            "prepared_at_utc": _utc_iso(reference_time),
            "implementation_commit": implementation_commit,
            "freeze_tag": FREEZE_TAG,
            "source_integrity": {
                "model_file": _repository_relative_path(franka_model_path()),
                "model_sha256": model_sha256,
                "torque_safety_sha256": torque_safety_sha256,
                "beacon_verifier_file": BEACON_VERIFIER.as_posix(),
                "beacon_verifier_sha256": verifier_sha256,
                "beacon_lockfile": BEACON_LOCKFILE.as_posix(),
                "beacon_lockfile_sha256": lockfile_sha256,
            },
            "dependencies": _dependency_versions(),
            "gate": asdict(ResidualRlGate()),
            "torque_safety": torque_safety_manifest(),
            "training": {
                "runs": [asdict(artifact) for artifact in artifacts],
                "jobs": jobs,
                "scenario_seed": TRAINING_SCENARIO_SEED,
                "scenarios": [asdict(scenario) for scenario in training],
                "resample_simulation_noise_each_iteration": True,
            },
            "development_validation": {
                "scenario_seed": DEVELOPMENT_SCENARIO_SEED,
                "simulation_seed_base": DEVELOPMENT_SIMULATION_SEED,
                "simulation_seed_derivation": (
                    "base + run simulation seed + 100000 * iteration"
                ),
                "scenarios": [asdict(scenario) for scenario in development],
                "checkpoint_selection_metric": "mean_physical_rollout_cost",
            },
            "blind_contract": {
                "cases": BLIND_CASES,
                "duration_s": BLIND_DURATION,
                "controllers": list(BLIND_CONTROLLER_CONTRACT),
                "primary_rule": {
                    "policy_count": len(TRAINING_POLICY_SEEDS),
                    "minimum_passes_per_policy": BLIND_REQUIRED_PASSES,
                    "case_count": BLIND_CASES,
                },
                "seed_derivation": SEED_DERIVATION,
                "beacon": {
                    "network": "drand-quicknet",
                    "chain_hash": beacon_chain_hash,
                    "round": beacon_round,
                    "scheduled_unix": scheduled_time,
                    "scheduled_utc": _utc_iso(scheduled_time),
                    "minimum_unpublished_lead_seconds_at_freeze": (
                        MIN_BEACON_LEAD_SECONDS
                    ),
                    "freshness_reference": {
                        "round": int(final_reference["round"]),
                        "scheduled_unix": reference_time,
                        "scheduled_utc": _utc_iso(reference_time),
                        "guaranteed_unpublished_lead_seconds": (
                            beacon_round - int(final_reference["round"]) - 1
                        )
                        * int(QUICKNET_CHAIN_INFO["period"]),
                        "beacon": final_reference,
                        "verification_audit": reference_audit,
                    },
                    "chain_info": QUICKNET_CHAIN_INFO,
                    "verification": {
                        "mode": "two-relay-cryptographic",
                        "client": BEACON_VERIFIER_NAME,
                        "client_version": BEACON_VERIFIER_VERSION,
                        "relays": list(BEACON_RELAYS),
                        "signature_verification_required": True,
                    },
                },
            },
        }
        protocol_path = temporary_dir / "protocol.json"
        protocol_path.write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary_dir / "protocol.sha256").write_text(
            _sha256(protocol_path) + "  protocol.json\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, output_dir)
    return output_dir / "protocol.json"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_preholdout_protocol(
    protocol_path: Path,
    *,
    require_tag_binding: bool = False,
) -> tuple[dict[str, Any], str]:
    protocol_hash = _sha256(protocol_path)
    checksum_path = protocol_path.with_name("protocol.sha256")
    expected_checksum = f"{protocol_hash}  protocol.json\n"
    if checksum_path.read_text(encoding="utf-8") != expected_checksum:
        raise ValueError("protocol checksum mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    fixed_fields = {
        "format_version": PROTOCOL_FORMAT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "preholdout_frozen",
        "freeze_tag": FREEZE_TAG,
        "gate": asdict(ResidualRlGate()),
        "torque_safety": torque_safety_manifest(),
    }
    for field, expected in fixed_fields.items():
        if protocol.get(field) != expected:
            raise ValueError(f"protocol field does not match frozen contract: {field}")

    source_integrity = protocol.get("source_integrity", {})
    expected_source_files = {
        "model_file": _repository_relative_path(franka_model_path()),
        "beacon_verifier_file": BEACON_VERIFIER.as_posix(),
        "beacon_lockfile": BEACON_LOCKFILE.as_posix(),
    }
    for field, expected in expected_source_files.items():
        if source_integrity.get(field) != expected:
            raise ValueError(f"unexpected source-integrity path: {field}")
    for field in (
        "model_sha256",
        "torque_safety_sha256",
        "beacon_verifier_sha256",
        "beacon_lockfile_sha256",
    ):
        if not _is_sha256(source_integrity.get(field)):
            raise ValueError(f"malformed source-integrity hash: {field}")
    if source_integrity["torque_safety_sha256"] != _json_sha256(
        torque_safety_manifest()
    ):
        raise ValueError("torque-safety manifest hash mismatch")

    training = protocol.get("training", {})
    if training.get("scenario_seed") != TRAINING_SCENARIO_SEED:
        raise ValueError("training scenario seed changed")
    if training.get("scenarios") != [asdict(value) for value in _training_scenarios()]:
        raise ValueError("training scenarios changed")
    if training.get("resample_simulation_noise_each_iteration") is not True:
        raise ValueError("training noise-resampling contract changed")
    if not isinstance(training.get("jobs"), int) or not 1 <= training["jobs"] <= 5:
        raise ValueError("training jobs must be recorded in [1, 5]")
    artifacts = training.get("runs", [])
    if len(artifacts) != len(TRAINING_POLICY_SEEDS):
        raise ValueError("protocol must contain exactly five policy runs")
    seen_files: set[str] = set()
    seen_hashes: set[str] = set()
    artifact_paths: list[Path] = []
    for run_index, artifact in enumerate(artifacts):
        expected_run_fields = {
            "run_index": run_index,
            "policy_seed": TRAINING_POLICY_SEEDS[run_index],
            "simulation_seed": TRAINING_SIMULATION_SEEDS[run_index],
            "policy_file": f"policy_{run_index:02d}.json",
            "training_curve_file": f"training_curve_{run_index:02d}.csv",
            "training_plot_file": f"training_curve_{run_index:02d}.png",
        }
        for field, expected in expected_run_fields.items():
            if artifact.get(field) != expected:
                raise ValueError(f"run {run_index} violates frozen field {field}")
        for file_field, hash_field in (
            ("policy_file", "policy_sha256"),
            ("training_curve_file", "training_curve_sha256"),
            ("training_plot_file", "training_plot_sha256"),
        ):
            filename = str(artifact[file_field])
            digest = artifact.get(hash_field)
            if filename in seen_files or digest in seen_hashes or not _is_sha256(digest):
                raise ValueError("artifact filenames and hashes must be valid and unique")
            seen_files.add(filename)
            seen_hashes.add(str(digest))
            artifact_path = protocol_path.parent / filename
            if _sha256(artifact_path) != digest:
                raise ValueError(f"artifact checksum mismatch: {filename}")
            artifact_paths.append(artifact_path)

        policy_path = protocol_path.parent / artifact["policy_file"]
        policy = LinearResidualPolicy.load(policy_path)
        if policy.observation_names != TORQUE_AWARE_OBSERVATION_NAMES:
            raise ValueError(f"policy schema mismatch: {policy_path.name}")
        policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "experiment_id": EXPERIMENT_ID,
            "algorithm": "augmented_random_search",
            "training_run_index": run_index,
            "training_config": asdict(_training_config(run_index)),
            "development_scenario_seed": DEVELOPMENT_SCENARIO_SEED,
            "development_simulation_seed_base": DEVELOPMENT_SIMULATION_SEED,
            "resample_simulation_noise_each_iteration": True,
            "torque_safety": torque_safety_manifest(),
        }
        if policy_payload.get("metadata") != expected_metadata:
            raise ValueError(f"policy metadata mismatch: {policy_path.name}")

    development = protocol.get("development_validation", {})
    expected_development = {
        "scenario_seed": DEVELOPMENT_SCENARIO_SEED,
        "simulation_seed_base": DEVELOPMENT_SIMULATION_SEED,
        "simulation_seed_derivation": "base + run simulation seed + 100000 * iteration",
        "scenarios": [asdict(value) for value in _development_scenarios()],
        "checkpoint_selection_metric": "mean_physical_rollout_cost",
    }
    if development != expected_development:
        raise ValueError("development-validation contract changed")

    blind_contract = protocol.get("blind_contract", {})
    expected_primary_rule = {
        "policy_count": len(TRAINING_POLICY_SEEDS),
        "minimum_passes_per_policy": BLIND_REQUIRED_PASSES,
        "case_count": BLIND_CASES,
    }
    if (
        blind_contract.get("cases") != BLIND_CASES
        or blind_contract.get("duration_s") != BLIND_DURATION
        or blind_contract.get("controllers") != list(BLIND_CONTROLLER_CONTRACT)
        or blind_contract.get("primary_rule") != expected_primary_rule
        or blind_contract.get("seed_derivation") != SEED_DERIVATION
    ):
        raise ValueError("blind-evaluation contract changed")

    beacon_contract = blind_contract.get("beacon", {})
    beacon_round = beacon_contract.get("round")
    if not isinstance(beacon_round, int) or beacon_round <= 0:
        raise ValueError("beacon round must be a positive integer")
    scheduled_time = beacon_round_time(beacon_round)
    expected_beacon_fields = {
        "network": "drand-quicknet",
        "chain_hash": QUICKNET_CHAIN_INFO["hash"],
        "scheduled_unix": scheduled_time,
        "scheduled_utc": _utc_iso(scheduled_time),
        "minimum_unpublished_lead_seconds_at_freeze": MIN_BEACON_LEAD_SECONDS,
        "chain_info": QUICKNET_CHAIN_INFO,
        "verification": {
            "mode": "two-relay-cryptographic",
            "client": BEACON_VERIFIER_NAME,
            "client_version": BEACON_VERIFIER_VERSION,
            "relays": list(BEACON_RELAYS),
            "signature_verification_required": True,
        },
    }
    for field, expected in expected_beacon_fields.items():
        if beacon_contract.get(field) != expected:
            raise ValueError(f"beacon contract changed: {field}")

    freshness = beacon_contract.get("freshness_reference", {})
    reference_round = freshness.get("round")
    if type(reference_round) is not int or reference_round <= 0:
        raise ValueError("freshness reference round must be a positive integer")
    reference_time = beacon_round_time(reference_round)
    expected_freshness_fields = {
        "scheduled_unix": reference_time,
        "scheduled_utc": _utc_iso(reference_time),
        "guaranteed_unpublished_lead_seconds": (
            beacon_round - reference_round - 1
        )
        * int(QUICKNET_CHAIN_INFO["period"]),
    }
    for field, expected in expected_freshness_fields.items():
        if freshness.get(field) != expected:
            raise ValueError(f"freshness reference changed: {field}")
    reference_audit = freshness.get("verification_audit", {})
    if not isinstance(reference_audit, dict):
        raise TypeError("freshness verification audit must be an object")
    verifier_output = reference_audit.get("verifier_output", {})
    if not isinstance(verifier_output, dict):
        raise TypeError("freshness verifier output must be an object")
    verified_reference = verify_latest_reference(verifier_output)
    if verified_reference != freshness.get("beacon"):
        raise ValueError("freshness beacon does not match its verification evidence")
    if verified_reference["round"] != reference_round:
        raise ValueError("freshness beacon round changed")
    _require_future_beacon(beacon_round, reference_round=reference_round)

    try:
        prepared_time = datetime.fromisoformat(
            str(protocol["prepared_at_utc"]).replace("Z", "+00:00")
        ).timestamp()
    except (KeyError, ValueError) as error:
        raise ValueError("prepared_at_utc is malformed") from error
    if prepared_time != reference_time:
        raise ValueError("prepared_at_utc must match the verified freshness round")

    if require_tag_binding:
        tag = str(protocol["freeze_tag"])
        tag_commit = _git("rev-list", "-n", "1", tag)
        if _git("rev-parse", "HEAD") != tag_commit:
            raise RuntimeError("HEAD must match the frozen pre-holdout tag")
        parents = _git("rev-list", "--parents", "-n", "1", tag_commit).split()
        if len(parents) != 2 or parents[1] != protocol.get("implementation_commit"):
            raise ValueError("freeze tag must be a single artifact commit on implementation_commit")
        for path in (protocol_path, checksum_path, *artifact_paths):
            _require_file_at_tag(path, tag)
        repository_root = Path(_git("rev-parse", "--show-toplevel"))
        source_files = (
            (repository_root / source_integrity["model_file"], "model_sha256"),
            (
                repository_root / source_integrity["beacon_verifier_file"],
                "beacon_verifier_sha256",
            ),
            (
                repository_root / source_integrity["beacon_lockfile"],
                "beacon_lockfile_sha256",
            ),
        )
        for path, hash_field in source_files:
            if _sha256(path) != source_integrity[hash_field]:
                raise ValueError(f"source-integrity checksum mismatch: {path.name}")
            _require_file_at_tag(path, tag)
    return protocol, protocol_hash


def verify_beacon(protocol: dict[str, Any], verifier_output: dict[str, Any]) -> dict[str, Any]:
    if (
        verifier_output.get("verifier") != BEACON_VERIFIER_NAME
        or verifier_output.get("verifier_version") != BEACON_VERIFIER_VERSION
        or verifier_output.get("cryptographic_signature_verified") is not True
    ):
        raise ValueError("beacon was not cryptographically verified by the frozen client")
    expected_chain = str(protocol["blind_contract"]["beacon"]["chain_hash"])
    expected_round = int(protocol["blind_contract"]["beacon"]["round"])
    if verifier_output.get("round") != expected_round:
        raise ValueError("beacon round does not match the frozen protocol")
    responses = verifier_output.get("responses")
    if not isinstance(responses, list) or len(responses) != len(BEACON_RELAYS):
        raise ValueError("beacon verifier must return both frozen relays")
    canonical_beacon: dict[str, Any] | None = None
    for relay, response in zip(BEACON_RELAYS, responses, strict=True):
        if response.get("base_url") != relay:
            raise ValueError("beacon relay order or identity changed")
        if response.get("chain_url") != f"{relay}/{expected_chain}":
            raise ValueError("beacon request did not use the pinned chain URL")
        if response.get("chain_info") != QUICKNET_CHAIN_INFO:
            raise ValueError("beacon chain info does not match the pinned root of trust")
        beacon = response.get("beacon", {})
        if beacon.get("round") != expected_round or "previous_signature" in beacon:
            raise ValueError("beacon response has the wrong round or scheme")
        randomness = str(beacon.get("randomness", ""))
        signature = str(beacon.get("signature", ""))
        if not _is_sha256(randomness) or len(signature) < 64:
            raise ValueError("beacon randomness or signature is malformed")
        try:
            derived_randomness = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
        except ValueError as error:
            raise ValueError("beacon signature must be hexadecimal") from error
        if derived_randomness != randomness:
            raise ValueError("beacon randomness does not match its signature hash")
        if canonical_beacon is None:
            canonical_beacon = beacon
        elif beacon != canonical_beacon:
            raise ValueError("official drand relays returned different beacons")
    assert canonical_beacon is not None
    return {
        "chain_hash": expected_chain,
        "round": expected_round,
        "signature": str(canonical_beacon["signature"]),
        "randomness": str(canonical_beacon["randomness"]),
    }


def verify_latest_reference(verifier_output: dict[str, Any]) -> dict[str, Any]:
    if verifier_output.get("mode") != "latest-reference":
        raise ValueError("beacon verifier did not use latest-reference mode")
    observed = verifier_output.get("observed_latest_rounds")
    if (
        not isinstance(observed, list)
        or len(observed) != len(BEACON_RELAYS)
        or any(type(round_number) is not int or round_number <= 0 for round_number in observed)
    ):
        raise ValueError("latest-reference evidence must contain one round per relay")
    reference_round = max(observed)
    if max(observed) - min(observed) > MAX_REFERENCE_ROUND_SKEW:
        raise ValueError("official drand relays are too far apart")
    if verifier_output.get("round") != reference_round:
        raise ValueError("latest-reference round must be the freshest observed relay round")
    protocol = {
        "blind_contract": {
            "beacon": {
                "chain_hash": QUICKNET_CHAIN_INFO["hash"],
                "round": reference_round,
            }
        }
    }
    return verify_beacon(protocol, verifier_output)


def _invoke_beacon_verifier(
    verifier_path: Path,
    requested_round: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = ["node", str(verifier_path), requested_round]
    request_started_at = datetime.now(tz=timezone.utc).timestamp()
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    retrieved_at = datetime.now(tz=timezone.utc).timestamp()
    try:
        verifier_output = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("beacon verifier returned malformed JSON") from error
    audit = {
        "host_request_started_at_utc": _utc_iso(request_started_at),
        "host_retrieved_at_utc": _utc_iso(retrieved_at),
        "command": command,
        "exit_code": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr": result.stderr,
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "verifier_output": verifier_output,
        "relay_equality_verified": True,
        "signature_hash_verified": True,
    }
    return verifier_output, audit


def fetch_verified_reference_beacon(
    verifier_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verifier_output, audit = _invoke_beacon_verifier(verifier_path, "latest")
    return verify_latest_reference(verifier_output), audit


def fetch_verified_beacon(
    protocol: dict[str, Any],
    verifier_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_round = int(protocol["blind_contract"]["beacon"]["round"])
    verifier_output, audit = _invoke_beacon_verifier(verifier_path, str(expected_round))
    if verifier_output.get("mode") != "exact-round":
        raise ValueError("beacon verifier did not use exact-round mode")
    beacon = verify_beacon(protocol, verifier_output)
    return beacon, audit


def derive_blind_root(protocol_hash: str, beacon_randomness: str) -> bytes:
    try:
        material = bytes.fromhex(protocol_hash) + bytes.fromhex(beacon_randomness)
    except ValueError as error:
        raise ValueError("protocol hash and beacon randomness must be hexadecimal") from error
    return hashlib.sha256(material).digest()


def sample_blind_scenarios(
    root: bytes,
    count: int,
) -> tuple[tuple[FrankaScenario, ...], tuple[int, ...], tuple[int, ...]]:
    scenarios = []
    scenario_seeds = []
    simulation_seeds = []
    for index in range(count):
        scenario_seed = derive_seed(root, "blind/scenario", index)
        simulation_seed = derive_seed(root, "blind/noise", index)
        sampled = sample_stress_scenarios(1, scenario_seed)[0]
        values = asdict(sampled)
        values["name"] = f"blind_{index:02d}"
        scenarios.append(FrankaScenario(**values))
        scenario_seeds.append(scenario_seed)
        simulation_seeds.append(simulation_seed)
    return tuple(scenarios), tuple(scenario_seeds), tuple(simulation_seeds)


def _controller_diagnostics(controller: FrankaController) -> dict[str, float | int]:
    is_residual = hasattr(controller, "residual_rms_n")
    nominal = getattr(controller, "nominal", controller)
    return {
        "fallback_count": int(getattr(controller, "fallback_count", 0)),
        "policy_update_count": int(getattr(controller, "policy_update_count", 0)),
        "torque_context_fallback_count": int(
            getattr(controller, "torque_context_fallback_count", 0)
        ),
        "residual_rms_n": float(getattr(controller, "residual_rms_n", 0.0)),
        "residual_torque_projection_pct": float(
            getattr(controller, "torque_projection_pct", 0.0) if is_residual else 0.0
        ),
        "residual_mean_projection_scale": float(
            getattr(controller, "mean_torque_projection_scale", 1.0)
            if is_residual
            else 1.0
        ),
        "nominal_torque_projection_pct": float(
            getattr(nominal, "torque_projection_pct", 0.0)
        ),
        "nominal_mean_projection_scale": float(
            getattr(nominal, "mean_torque_projection_scale", 1.0)
        ),
        "nominal_projection_fallback_count": int(
            getattr(nominal, "torque_projection_fallback_count", 0)
        ),
    }


def evaluate_blind_cases(
    policies: tuple[LinearResidualPolicy, ...],
    policy_hashes: tuple[str, ...],
    scenarios: tuple[FrankaScenario, ...],
    scenario_seeds: tuple[int, ...],
    simulation_seeds: tuple[int, ...],
    duration: float,
) -> list[dict[str, float | int | str]]:
    if not (
        len(scenarios) == len(scenario_seeds) == len(simulation_seeds)
        and len(policies) == len(policy_hashes)
    ):
        raise ValueError("scenario, seed and policy inputs must have matching lengths")
    gate = ResidualRlGate()
    rows: list[dict[str, float | int | str]] = []
    for case_index, scenario in enumerate(scenarios):
        controller_specs: list[
            tuple[str, str, str, FrankaController]
        ] = [
            ("fixed_hybrid", "", "", FrankaHybridController()),
            ("adaptive_hybrid", "", "", FrankaAdaptiveHybridController()),
            ("safe_adaptive_hybrid", "", "", FrankaSafeAdaptiveController()),
        ]
        controller_specs.extend(
            (
                f"torque_residual_run_{run_index:02d}",
                f"run_{run_index:02d}",
                policy_hashes[run_index],
                TorqueProjectedResidualController(
                    policy=policy,
                    nominal=FrankaSafeAdaptiveController(),
                    name=f"torque_residual_run_{run_index:02d}",
                ),
            )
            for run_index, policy in enumerate(policies)
        )
        for method, training_run_id, policy_hash, controller in controller_specs:
            result = run_franka_trial(
                controller,
                scenario=scenario,
                config=FrankaSimulationConfig(
                    duration=duration,
                    seed=simulation_seeds[case_index],
                ),
            )
            metrics = result.metrics()
            failures = failed_gate_checks(metrics, gate)
            rows.append(
                {
                    "method": method,
                    "training_run_id": training_run_id,
                    "policy_sha256": policy_hash,
                    "case": scenario.name,
                    "scenario_seed": scenario_seeds[case_index],
                    "simulation_seed": simulation_seeds[case_index],
                    **scenario_values(scenario),
                    **metrics,
                    **_controller_diagnostics(controller),
                    "gate_pass": "yes" if not failures else "no",
                    "failed_checks": ";".join(failures),
                }
            )
        print(f"blind case {case_index + 1:02d}/{len(scenarios):02d} complete", flush=True)
    return rows


def _method_order(rows: list[dict[str, float | int | str]]) -> list[str]:
    return list(dict.fromkeys(str(row["method"]) for row in rows))


def _method_rows(
    rows: list[dict[str, float | int | str]], method: str
) -> list[dict[str, float | int | str]]:
    return [row for row in rows if row["method"] == method]


def _aggregate(rows: list[dict[str, float | int | str]], method: str) -> dict[str, float]:
    selected = _method_rows(rows, method)
    return {
        "pass_count": float(sum(row["gate_pass"] == "yes" for row in selected)),
        "pass_rate_pct": 100.0 * float(np.mean([row["gate_pass"] == "yes" for row in selected])),
        "force_p95_n": float(np.percentile([row["force_rmse_n"] for row in selected], 95)),
        "peak_p95_n": float(np.percentile([row["peak_force_n"] for row in selected], 95)),
        "tangent_p95_mm": float(
            np.percentile([row["tangent_rmse_mm"] for row in selected], 95)
        ),
        "contact_worst_pct": min(float(row["contact_ratio_pct"]) for row in selected),
        "saturation_worst_pct": max(float(row["saturation_pct"]) for row in selected),
    }


def _hierarchical_pass_interval(
    rows: list[dict[str, float | int | str]], seed: int, samples: int = 2_000
) -> tuple[float, float]:
    methods = [method for method in _method_order(rows) if method.startswith("torque_residual")]
    cases = list(dict.fromkeys(str(row["case"]) for row in rows))
    matrix = np.asarray(
        [
            [
                next(
                    row["gate_pass"] == "yes"
                    for row in rows
                    if row["method"] == method and row["case"] == case
                )
                for case in cases
            ]
            for method in methods
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        run_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        case_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        estimates[index] = 100.0 * np.mean(matrix[run_indices][:, case_indices])
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def _write_summary(
    rows: list[dict[str, float | int | str]],
    reporting_seed: int,
    path: Path,
) -> None:
    case_count = len({row["case"] for row in rows})
    required_passes = math.ceil(0.90 * case_count)
    methods = _method_order(rows)
    aggregates = {method: _aggregate(rows, method) for method in methods}
    residual_methods = [method for method in methods if method.startswith("torque_residual")]
    residual_pass_counts = [aggregates[method]["pass_count"] for method in residual_methods]
    all_pass = all(count >= required_passes for count in residual_pass_counts)
    lower, upper = _hierarchical_pass_interval(rows, reporting_seed)
    lines = [
        "# v0.5 torque-safe Residual RL blind result",
        "",
        f"The first reveal evaluated {case_count} cases. Each method used the same case and noise seed.",
        f"The frozen primary rule requires every policy to pass at least {required_passes}/{case_count} cases.",
        "",
        (
            "| Method | Pass | Force P95 [N] | Raw peak P95 [N] | "
            "Tangent P95 [mm] | Contact worst | Saturation worst |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        aggregate = aggregates[method]
        lines.append(
            f"| {method} | {int(aggregate['pass_count'])}/{case_count} | "
            f"{aggregate['force_p95_n']:.2f} | {aggregate['peak_p95_n']:.2f} | "
            f"{aggregate['tangent_p95_mm']:.2f} | {aggregate['contact_worst_pct']:.1f}% | "
            f"{aggregate['saturation_worst_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Mean residual pass rate hierarchical-bootstrap 95% interval: {lower:.1f}% to {upper:.1f}%.",
            f"Frozen primary rule: {'PASS' if all_pass else 'FAIL'}.",
            "This directory is the first reveal. Later tuning against these cases is validation work, not blind evaluation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_summary(rows: list[dict[str, float | int | str]], path: Path) -> None:
    methods = _method_order(rows)
    labels = [method.replace("torque_residual_", "residual_") for method in methods]
    aggregates = [_aggregate(rows, method) for method in methods]
    panels = (
        ("pass_rate_pct", "Case pass rate [%]", 90.0),
        ("force_p95_n", "Force RMSE P95 [N]", 2.0),
        ("peak_p95_n", "Raw peak force P95 [N]", 35.0),
        ("tangent_p95_mm", "Tangential RMSE P95 [mm]", 15.0),
        ("saturation_worst_pct", "Worst torque saturation [%]", 1.0),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0))
    colors = ["#6c757d", "#277da1", "#577590"] + ["#e76f51"] * 5
    for axis, (metric, label, gate) in zip(axes.flat, panels, strict=False):
        values = [aggregate[metric] for aggregate in aggregates]
        axis.bar(np.arange(len(methods)), values, color=colors)
        axis.axhline(gate, color="black", linestyle="--", linewidth=1.0)
        axis.set_ylabel(label)
        axis.set_xticks(np.arange(len(methods)), labels, rotation=45, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle("v0.5 first-reveal comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_blind_evaluation(
    protocol_path: Path,
    output_dir: Path,
    verifier_path: Path | None = None,
) -> Path:
    """Verify the freeze, derive the hidden cases and run the first reveal once."""
    _require_clean_repository()
    protocol, protocol_hash = verify_preholdout_protocol(
        protocol_path,
        require_tag_binding=True,
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    repository_root = Path(_git("rev-parse", "--show-toplevel"))
    verifier_path = verifier_path or repository_root / BEACON_VERIFIER
    if verifier_path.resolve() != (repository_root / BEACON_VERIFIER).resolve():
        raise ValueError("blind evaluation must use the verifier frozen in the tag")
    beacon, beacon_audit = fetch_verified_beacon(protocol, verifier_path)
    blind_root = derive_blind_root(protocol_hash, beacon["randomness"])
    case_count = int(protocol["blind_contract"]["cases"])
    scenarios, scenario_seeds, simulation_seeds = sample_blind_scenarios(
        blind_root, case_count
    )
    training = tuple(FrankaScenario(**values) for values in protocol["training"]["scenarios"])
    development = tuple(
        FrankaScenario(**values) for values in protocol["development_validation"]["scenarios"]
    )
    _assert_disjoint_scenarios(training, development, scenarios)

    artifacts = protocol["training"]["runs"]
    policies = tuple(
        LinearResidualPolicy.load(protocol_path.parent / artifact["policy_file"])
        for artifact in artifacts
    )
    policy_hashes = tuple(str(artifact["policy_sha256"]) for artifact in artifacts)
    rows = evaluate_blind_cases(
        policies,
        policy_hashes,
        scenarios,
        scenario_seeds,
        simulation_seeds,
        float(protocol["blind_contract"]["duration_s"]),
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="franka-v05-blind-", dir=output_dir.parent) as tmp:
        temporary_dir = Path(tmp)
        _write_csv(rows, temporary_dir / "comparison.csv")
        reporting_seed = derive_seed(blind_root, "report/bootstrap", 0)
        _write_summary(rows, reporting_seed, temporary_dir / "summary.md")
        _plot_summary(rows, temporary_dir / "comparison.png")
        reveal = {
            "experiment_id": EXPERIMENT_ID,
            "protocol_sha256": protocol_hash,
            "beacon": beacon,
            "beacon_verification_audit": beacon_audit,
            "blind_root_sha256": blind_root.hex(),
            "scenario_seeds": list(scenario_seeds),
            "simulation_seeds": list(simulation_seeds),
            "reporting_seed": reporting_seed,
        }
        (temporary_dir / "reveal.json").write_text(
            json.dumps(reveal, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_hashes = {
            path.name: _sha256(path)
            for path in sorted(temporary_dir.iterdir())
            if path.is_file()
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "freeze_tag": protocol["freeze_tag"],
                    "protocol_sha256": protocol_hash,
                    "artifact_sha256": artifact_hashes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary_dir / "COMPLETE").write_text("first reveal complete\n", encoding="utf-8")
        os.replace(temporary_dir, output_dir)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="train policies and freeze the protocol")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--beacon-chain-hash",
        default=QUICKNET_CHAIN_INFO["hash"],
        help="pinned drand Quicknet chain hash",
    )
    prepare.add_argument("--beacon-round", type=int, required=True)
    prepare.add_argument("--jobs", type=int, default=1)

    evaluate = subparsers.add_parser("evaluate", help="run the frozen first reveal")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument(
        "--verifier",
        type=Path,
        help="path to the frozen drand verifier (defaults to tools/verify_drand_beacon.mjs)",
    )
    evaluate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prepare":
        prepare_preholdout(
            args.output,
            beacon_chain_hash=args.beacon_chain_hash,
            beacon_round=args.beacon_round,
            jobs=args.jobs,
        )
    else:
        run_blind_evaluation(args.protocol, args.output, args.verifier)


if __name__ == "__main__":
    main()
