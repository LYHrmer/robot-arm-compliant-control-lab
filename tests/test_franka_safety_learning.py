from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

import compliant_control_lab.franka_safety_learning as safety_learning
from compliant_control_lab.franka_safety_learning import (
    EXPERIMENT_ID,
    _write_summary,
    derive_seed,
    evaluate_blind_cases,
    sample_blind_scenarios,
    verify_beacon,
    verify_preholdout_protocol,
)
from compliant_control_lab.residual_rl import (
    TORQUE_AWARE_OBSERVATION_NAMES,
    LinearResidualPolicy,
)

HISTORICAL_QUICKNET_ROUND = 1_000
HISTORICAL_QUICKNET_SIGNATURE = (
    "b44679b9a59af2ec876b1a6b1ad52ea9b1615fc3982b19576350f93447cb1125"
    "e342b73a8dd2bacbe47e4b6b63ed5e39"
)
HISTORICAL_QUICKNET_RANDOMNESS = "fe290beca10872ef2fb164d2aa4442de4566183ec51c56ff3cd603d930e54fdd"


def _write_protocol(protocol_path: Path, protocol: dict[str, Any]) -> str:
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    protocol_path.with_name("protocol.sha256").write_text(
        f"{protocol_hash}  protocol.json\n",
        encoding="utf-8",
    )
    return protocol_hash


def _historical_beacon_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    chain_hash = safety_learning.QUICKNET_CHAIN_INFO["hash"]
    protocol = {
        "blind_contract": {
            "beacon": {
                "chain_hash": chain_hash,
                "round": HISTORICAL_QUICKNET_ROUND,
            }
        }
    }
    beacon = {
        "round": HISTORICAL_QUICKNET_ROUND,
        "randomness": HISTORICAL_QUICKNET_RANDOMNESS,
        "signature": HISTORICAL_QUICKNET_SIGNATURE,
    }
    verifier_output = {
        "verifier": safety_learning.BEACON_VERIFIER_NAME,
        "verifier_version": safety_learning.BEACON_VERIFIER_VERSION,
        "cryptographic_signature_verified": True,
        "mode": "exact-round",
        "round": HISTORICAL_QUICKNET_ROUND,
        "responses": [
            {
                "base_url": relay,
                "chain_url": f"{relay}/{chain_hash}",
                "chain_info": deepcopy(safety_learning.QUICKNET_CHAIN_INFO),
                "beacon": deepcopy(beacon),
            }
            for relay in safety_learning.BEACON_RELAYS
        ],
    }
    return protocol, verifier_output


def _latest_reference_fixture(reference_round: int) -> tuple[dict[str, Any], dict[str, Any]]:
    chain_hash = safety_learning.QUICKNET_CHAIN_INFO["hash"]
    signature = (f"verified latest round {reference_round}".encode() * 3).hex()
    raw_beacon = {
        "round": reference_round,
        "randomness": hashlib.sha256(bytes.fromhex(signature)).hexdigest(),
        "signature": signature,
    }
    verifier_output = {
        "verifier": safety_learning.BEACON_VERIFIER_NAME,
        "verifier_version": safety_learning.BEACON_VERIFIER_VERSION,
        "cryptographic_signature_verified": True,
        "mode": "latest-reference",
        "round": reference_round,
        "observed_latest_rounds": [reference_round - 1, reference_round],
        "responses": [
            {
                "base_url": relay,
                "chain_url": f"{relay}/{chain_hash}",
                "chain_info": deepcopy(safety_learning.QUICKNET_CHAIN_INFO),
                "beacon": deepcopy(raw_beacon),
            }
            for relay in safety_learning.BEACON_RELAYS
        ],
    }
    beacon = {
        "chain_hash": chain_hash,
        "round": reference_round,
        "signature": signature,
        "randomness": raw_beacon["randomness"],
    }
    return beacon, verifier_output


@pytest.fixture
def strict_protocol(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    artifacts = []
    for run_index, (policy_seed, simulation_seed) in enumerate(
        zip(
            safety_learning.TRAINING_POLICY_SEEDS,
            safety_learning.TRAINING_SIMULATION_SEEDS,
            strict=True,
        )
    ):
        policy_path = tmp_path / f"policy_{run_index:02d}.json"
        curve_path = tmp_path / f"training_curve_{run_index:02d}.csv"
        plot_path = tmp_path / f"training_curve_{run_index:02d}.png"
        LinearResidualPolicy.zero(TORQUE_AWARE_OBSERVATION_NAMES).save(
            policy_path,
            metadata={
                "experiment_id": EXPERIMENT_ID,
                "algorithm": "augmented_random_search",
                "training_run_index": run_index,
                "training_config": asdict(safety_learning._training_config(run_index)),
                "development_scenario_seed": safety_learning.DEVELOPMENT_SCENARIO_SEED,
                "development_simulation_seed_base": (safety_learning.DEVELOPMENT_SIMULATION_SEED),
                "resample_simulation_noise_each_iteration": True,
                "torque_safety": safety_learning.torque_safety_manifest(),
            },
        )
        curve_path.write_text(f"iteration,cost\n{run_index},{run_index + 0.5}\n", encoding="utf-8")
        plot_path.write_bytes(f"frozen plot {run_index}\n".encode())
        artifacts.append(
            {
                "run_index": run_index,
                "policy_seed": policy_seed,
                "simulation_seed": simulation_seed,
                "policy_file": policy_path.name,
                "policy_sha256": safety_learning._sha256(policy_path),
                "training_curve_file": curve_path.name,
                "training_curve_sha256": safety_learning._sha256(curve_path),
                "training_plot_file": plot_path.name,
                "training_plot_sha256": safety_learning._sha256(plot_path),
            }
        )

    repository_root = Path(safety_learning._git("rev-parse", "--show-toplevel"))
    model_path = safety_learning.franka_model_path()
    verifier_path = repository_root / safety_learning.BEACON_VERIFIER
    lockfile_path = repository_root / safety_learning.BEACON_LOCKFILE
    scheduled_time = safety_learning.beacon_round_time(HISTORICAL_QUICKNET_ROUND)
    reference_round = HISTORICAL_QUICKNET_ROUND - 201
    reference_time = safety_learning.beacon_round_time(reference_round)
    reference_beacon, reference_output = _latest_reference_fixture(reference_round)
    reference_audit = {
        "host_request_started_at_utc": safety_learning._utc_iso(reference_time),
        "host_retrieved_at_utc": safety_learning._utc_iso(reference_time),
        "command": ["node", str(verifier_path), "latest"],
        "exit_code": 0,
        "stdout_sha256": "0" * 64,
        "stderr": "",
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "verifier_output": reference_output,
        "relay_equality_verified": True,
        "signature_hash_verified": True,
    }
    protocol = {
        "format_version": safety_learning.PROTOCOL_FORMAT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "preholdout_frozen",
        "prepared_at_utc": safety_learning._utc_iso(reference_time),
        "implementation_commit": "implementation-commit",
        "freeze_tag": safety_learning.FREEZE_TAG,
        "source_integrity": {
            "model_file": safety_learning._repository_relative_path(model_path),
            "model_sha256": safety_learning._sha256(model_path),
            "torque_safety_sha256": safety_learning._json_sha256(
                safety_learning.torque_safety_manifest()
            ),
            "beacon_verifier_file": safety_learning.BEACON_VERIFIER.as_posix(),
            "beacon_verifier_sha256": safety_learning._sha256(verifier_path),
            "beacon_lockfile": safety_learning.BEACON_LOCKFILE.as_posix(),
            "beacon_lockfile_sha256": safety_learning._sha256(lockfile_path),
        },
        "gate": asdict(safety_learning.ResidualRlGate()),
        "torque_safety": safety_learning.torque_safety_manifest(),
        "training": {
            "runs": artifacts,
            "jobs": 1,
            "scenario_seed": safety_learning.TRAINING_SCENARIO_SEED,
            "scenarios": [asdict(scenario) for scenario in safety_learning._training_scenarios()],
            "resample_simulation_noise_each_iteration": True,
        },
        "development_validation": {
            "scenario_seed": safety_learning.DEVELOPMENT_SCENARIO_SEED,
            "simulation_seed_base": safety_learning.DEVELOPMENT_SIMULATION_SEED,
            "simulation_seed_derivation": "base + run simulation seed + 100000 * iteration",
            "scenarios": [
                asdict(scenario) for scenario in safety_learning._development_scenarios()
            ],
            "checkpoint_selection_metric": "mean_physical_rollout_cost",
        },
        "blind_contract": {
            "cases": safety_learning.BLIND_CASES,
            "duration_s": safety_learning.BLIND_DURATION,
            "controllers": list(safety_learning.BLIND_CONTROLLER_CONTRACT),
            "primary_rule": {
                "policy_count": len(safety_learning.TRAINING_POLICY_SEEDS),
                "minimum_passes_per_policy": safety_learning.BLIND_REQUIRED_PASSES,
                "case_count": safety_learning.BLIND_CASES,
            },
            "seed_derivation": safety_learning.SEED_DERIVATION,
            "beacon": {
                "network": "drand-quicknet",
                "chain_hash": safety_learning.QUICKNET_CHAIN_INFO["hash"],
                "round": HISTORICAL_QUICKNET_ROUND,
                "scheduled_unix": scheduled_time,
                "scheduled_utc": safety_learning._utc_iso(scheduled_time),
                "minimum_unpublished_lead_seconds_at_freeze": (
                    safety_learning.MIN_BEACON_LEAD_SECONDS
                ),
                "freshness_reference": {
                    "round": reference_round,
                    "scheduled_unix": reference_time,
                    "scheduled_utc": safety_learning._utc_iso(reference_time),
                    "guaranteed_unpublished_lead_seconds": (
                        safety_learning.MIN_BEACON_LEAD_SECONDS
                    ),
                    "beacon": reference_beacon,
                    "verification_audit": reference_audit,
                },
                "chain_info": deepcopy(safety_learning.QUICKNET_CHAIN_INFO),
                "verification": {
                    "mode": "two-relay-cryptographic",
                    "client": safety_learning.BEACON_VERIFIER_NAME,
                    "client_version": safety_learning.BEACON_VERIFIER_VERSION,
                    "relays": list(safety_learning.BEACON_RELAYS),
                    "signature_verification_required": True,
                },
            },
        },
    }
    protocol_path = tmp_path / "protocol.json"
    _write_protocol(protocol_path, protocol)
    return protocol_path, protocol


def _patch_frozen_git(
    monkeypatch: pytest.MonkeyPatch,
    *,
    repository_root: Path,
    scheduled_time: int,
) -> None:
    tag_commit = "freeze-tag-commit"
    replies = {
        ("rev-parse", "--show-toplevel"): str(repository_root),
        ("rev-list", "-n", "1", safety_learning.FREEZE_TAG): tag_commit,
        ("rev-parse", "HEAD"): tag_commit,
        ("rev-list", "--parents", "-n", "1", tag_commit): (f"{tag_commit} implementation-commit"),
        ("show", "-s", "--format=%ct", tag_commit): str(
            scheduled_time - safety_learning.MIN_BEACON_LEAD_SECONDS
        ),
    }

    def fake_git(*arguments: str) -> str:
        return replies[arguments]

    monkeypatch.setattr(safety_learning, "_git", fake_git)


def test_seed_derivation_is_stable_and_namespaces_do_not_collide() -> None:
    root = bytes(range(32))
    assert derive_seed(root, "blind/scenario", 3) == derive_seed(root, "blind/scenario", 3)
    assert derive_seed(root, "blind/scenario", 3) != derive_seed(root, "blind/noise", 3)
    assert derive_seed(root, "blind/scenario", 3) != derive_seed(root, "blind/scenario", 4)


def test_blind_scenarios_and_noise_seeds_are_reproducible() -> None:
    root = hashlib.sha256(b"unseen-root").digest()
    first = sample_blind_scenarios(root, 3)
    second = sample_blind_scenarios(root, 3)

    assert first == second
    assert len(set(first[1])) == len(set(first[2])) == 3
    assert set(first[1]).isdisjoint(first[2])


def test_historical_quicknet_beacon_with_verified_bls_structure_passes() -> None:
    protocol, verifier_output = _historical_beacon_fixture()

    beacon = verify_beacon(protocol, verifier_output)

    assert beacon == {
        "chain_hash": safety_learning.QUICKNET_CHAIN_INFO["hash"],
        "round": HISTORICAL_QUICKNET_ROUND,
        "signature": HISTORICAL_QUICKNET_SIGNATURE,
        "randomness": HISTORICAL_QUICKNET_RANDOMNESS,
    }
    assert hashlib.sha256(bytes.fromhex(beacon["signature"])).hexdigest() == beacon["randomness"]


def test_forged_beacon_verifier_identity_is_rejected() -> None:
    protocol, verifier_output = _historical_beacon_fixture()
    verifier_output["verifier"] = "lookalike-verifier"

    with pytest.raises(ValueError, match="not cryptographically verified"):
        verify_beacon(protocol, verifier_output)


@pytest.mark.parametrize("verified_value", [False, None])
def test_beacon_without_explicit_bls_verification_is_rejected(
    verified_value: bool | None,
) -> None:
    protocol, verifier_output = _historical_beacon_fixture()
    if verified_value is None:
        verifier_output.pop("cryptographic_signature_verified")
    else:
        verifier_output["cryptographic_signature_verified"] = verified_value

    with pytest.raises(ValueError, match="not cryptographically verified"):
        verify_beacon(protocol, verifier_output)


def test_two_official_relays_must_return_the_same_beacon() -> None:
    protocol, verifier_output = _historical_beacon_fixture()
    other_signature = (b"different but internally consistent beacon" * 2).hex()
    other_beacon = verifier_output["responses"][1]["beacon"]
    other_beacon["signature"] = other_signature
    other_beacon["randomness"] = hashlib.sha256(bytes.fromhex(other_signature)).hexdigest()

    with pytest.raises(ValueError, match="relays returned different beacons"):
        verify_beacon(protocol, verifier_output)


def test_latest_reference_accepts_adjacent_verified_relay_rounds() -> None:
    expected, verifier_output = _latest_reference_fixture(HISTORICAL_QUICKNET_ROUND)

    assert safety_learning.verify_latest_reference(verifier_output) == expected


def test_latest_reference_rejects_relay_skew_above_one_round() -> None:
    _, verifier_output = _latest_reference_fixture(HISTORICAL_QUICKNET_ROUND)
    verifier_output["observed_latest_rounds"] = [
        HISTORICAL_QUICKNET_ROUND - 2,
        HISTORICAL_QUICKNET_ROUND,
    ]

    with pytest.raises(ValueError, match="too far apart"):
        safety_learning.verify_latest_reference(verifier_output)


def test_prepare_future_beacon_check_uses_verified_round_not_host_clock() -> None:
    minimum_round_gap = (
        safety_learning.MIN_BEACON_LEAD_SECONDS
        + safety_learning.QUICKNET_CHAIN_INFO["period"]
        - 1
    ) // safety_learning.QUICKNET_CHAIN_INFO["period"] + 1
    target_round = HISTORICAL_QUICKNET_ROUND + minimum_round_gap

    assert safety_learning._require_future_beacon(
        target_round,
        reference_round=HISTORICAL_QUICKNET_ROUND,
    ) == safety_learning.beacon_round_time(target_round)


def test_prepare_future_beacon_check_rejects_short_verified_round_gap() -> None:
    with pytest.raises(ValueError, match="remain unpublished"):
        safety_learning._require_future_beacon(
            HISTORICAL_QUICKNET_ROUND + 200,
            reference_round=HISTORICAL_QUICKNET_ROUND,
        )


def test_strict_protocol_accepts_exactly_five_runs_and_48_cases(
    strict_protocol: tuple[Path, dict[str, Any]],
) -> None:
    protocol_path, _ = strict_protocol

    restored, restored_hash = verify_preholdout_protocol(protocol_path)

    assert len(restored["training"]["runs"]) == 5
    assert restored["blind_contract"]["cases"] == 48
    assert restored_hash == hashlib.sha256(protocol_path.read_bytes()).hexdigest()


def test_strict_protocol_rejects_tampered_freshness_evidence(
    strict_protocol: tuple[Path, dict[str, Any]],
) -> None:
    protocol_path, original = strict_protocol
    protocol = deepcopy(original)
    protocol["blind_contract"]["beacon"]["freshness_reference"][
        "verification_audit"
    ]["verifier_output"]["observed_latest_rounds"][0] -= 2
    _write_protocol(protocol_path, protocol)

    with pytest.raises(ValueError, match="too far apart"):
        verify_preholdout_protocol(protocol_path)


def test_strict_protocol_rejects_any_run_count_other_than_five(
    strict_protocol: tuple[Path, dict[str, Any]],
) -> None:
    protocol_path, original = strict_protocol
    protocol = deepcopy(original)
    protocol["training"]["runs"] = protocol["training"]["runs"][:4]
    _write_protocol(protocol_path, protocol)

    with pytest.raises(ValueError, match="exactly five policy runs"):
        verify_preholdout_protocol(protocol_path)


def test_strict_protocol_rejects_any_blind_case_count_other_than_48(
    strict_protocol: tuple[Path, dict[str, Any]],
) -> None:
    protocol_path, original = strict_protocol
    protocol = deepcopy(original)
    protocol["blind_contract"]["cases"] = 47
    _write_protocol(protocol_path, protocol)

    with pytest.raises(ValueError, match="blind-evaluation contract changed"):
        verify_preholdout_protocol(protocol_path)


def test_tag_binding_rejects_a_protocol_outside_the_repository(
    strict_protocol: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path, protocol = strict_protocol
    repository_root = Path(safety_learning._git("rev-parse", "--show-toplevel"))
    scheduled_time = protocol["blind_contract"]["beacon"]["scheduled_unix"]
    _patch_frozen_git(
        monkeypatch,
        repository_root=repository_root,
        scheduled_time=scheduled_time,
    )

    with pytest.raises(ValueError, match="outside the repository"):
        verify_preholdout_protocol(protocol_path, require_tag_binding=True)


def test_tag_binding_rejects_bytes_that_differ_from_the_freeze_tag(
    strict_protocol: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path, protocol = strict_protocol
    repository_root = Path(safety_learning._git("rev-parse", "--show-toplevel"))
    scheduled_time = protocol["blind_contract"]["beacon"]["scheduled_unix"]
    original_relative_path = safety_learning._repository_relative_path
    fixture_root = protocol_path.parent.resolve()
    _patch_frozen_git(
        monkeypatch,
        repository_root=repository_root,
        scheduled_time=scheduled_time,
    )

    def fixture_relative_path(path: Path) -> str:
        try:
            relative_path = path.resolve().relative_to(fixture_root)
        except ValueError:
            return original_relative_path(path)
        return f"results/v0.5/{relative_path.as_posix()}"

    monkeypatch.setattr(safety_learning, "_repository_relative_path", fixture_relative_path)
    monkeypatch.setattr(safety_learning, "_git_bytes", lambda *_: b"not the working bytes")

    with pytest.raises(ValueError, match="differs from freeze tag"):
        verify_preholdout_protocol(protocol_path, require_tag_binding=True)


def test_post_training_repository_recheck_reads_git_state_and_both_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_calls: list[tuple[str, ...]] = []

    def fake_git(*arguments: str) -> str:
        git_calls.append(arguments)
        if arguments == ("rev-parse", "HEAD"):
            return "implementation-commit"
        if arguments == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        raise AssertionError(f"unexpected git arguments: {arguments}")

    monkeypatch.setattr(safety_learning, "_git", fake_git)
    monkeypatch.setattr(safety_learning, "_sha256", lambda _: "model-hash")
    monkeypatch.setattr(safety_learning, "_json_sha256", lambda _: "safety-hash")

    safety_learning._require_repository_unchanged(
        "implementation-commit",
        "model-hash",
        "safety-hash",
    )

    assert git_calls == [
        ("rev-parse", "HEAD"),
        ("status", "--porcelain", "--untracked-files=no"),
    ]


@pytest.mark.parametrize(
    ("head", "tracked_status", "model_hash", "safety_hash", "message"),
    [
        ("changed", "", "model-hash", "safety-hash", "HEAD changed"),
        ("implementation-commit", " M source.py", "model-hash", "safety-hash", "tracked files"),
        ("implementation-commit", "", "changed", "safety-hash", "Franka model"),
        ("implementation-commit", "", "model-hash", "changed", "torque-safety settings"),
    ],
)
def test_post_training_repository_recheck_rejects_each_changed_input(
    monkeypatch: pytest.MonkeyPatch,
    head: str,
    tracked_status: str,
    model_hash: str,
    safety_hash: str,
    message: str,
) -> None:
    def fake_git(*arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return head
        if arguments == ("status", "--porcelain", "--untracked-files=no"):
            return tracked_status
        raise AssertionError(f"unexpected git arguments: {arguments}")

    monkeypatch.setattr(safety_learning, "_git", fake_git)
    monkeypatch.setattr(safety_learning, "_sha256", lambda _: model_hash)
    monkeypatch.setattr(safety_learning, "_json_sha256", lambda _: safety_hash)

    with pytest.raises(RuntimeError, match=message):
        safety_learning._require_repository_unchanged(
            "implementation-commit",
            "model-hash",
            "safety-hash",
        )


def test_smoke_comparison_reuses_each_case_and_noise_seed(tmp_path: Path) -> None:
    root = hashlib.sha256(b"smoke-comparison").digest()
    scenarios, scenario_seeds, simulation_seeds = sample_blind_scenarios(root, 2)
    policies = (
        LinearResidualPolicy.zero(TORQUE_AWARE_OBSERVATION_NAMES),
        LinearResidualPolicy.zero(TORQUE_AWARE_OBSERVATION_NAMES),
    )
    rows = evaluate_blind_cases(
        policies,
        ("a" * 64, "b" * 64),
        scenarios,
        scenario_seeds,
        simulation_seeds,
        duration=0.04,
    )

    assert len(rows) == 2 * (3 + len(policies))
    for case in ("blind_00", "blind_01"):
        selected = [row for row in rows if row["case"] == case]
        assert len({row["scenario_seed"] for row in selected}) == 1
        assert len({row["simulation_seed"] for row in selected}) == 1
    summary_path = tmp_path / "summary.md"
    _write_summary(rows, reporting_seed=7, path=summary_path)
    summary = summary_path.read_text(encoding="utf-8")
    assert "2 cases" in summary
    assert "2/2 cases" in summary
