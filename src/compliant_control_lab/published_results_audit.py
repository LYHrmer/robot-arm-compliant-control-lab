"""Offline integrity and semantic audit for the published v0.5 first reveal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASELINE_METHODS = (
    "fixed_hybrid",
    "adaptive_hybrid",
    "safe_adaptive_hybrid",
)
PUBLISHED_PROTOCOL_FORMAT_VERSION = 3
PUBLISHED_EXPERIMENT_ID = "franka-torque-safe-residual-v0.5"
PUBLISHED_PROTOCOL_SHA256 = "781839a62da0725bbe1ba8e321812dea2271ac7c8e646cdc9cf76732e4f02395"
PUBLISHED_BEACON_RANDOMNESS = "8312ee298ba9c9682876b19e5d00fb768c0eba4c5f0091323dd00d895491d02a"
PUBLISHED_MANIFEST_SHA256 = "9fc40ea11bb11cfbbe0fd291f46e69cc9cccc6fd0db208c21a24a96dd08e193c"
PUBLISHED_COMPLETE_SHA256 = "ea49ff658c1c20498c737af49548241b5fb663e84b2c5e76046b20a0c47d7baa"
PUBLISHED_ARTIFACT_SHA256 = {
    "comparison.csv": "4438970269075d25dfcf3361bccc7256cac96a48c44f12572d35c5f5530b2fb0",
    "comparison.png": "99ab410a35d6dd89b317b21e42a50c96fc17eea02a730e2ea1a7ad75e1585544",
    "reveal.json": "0e9a931769db42f687f549b2a8fac31520003d7f424fa23f629c6fee4b628ab7",
    "summary.md": "32f1a8f39861f68b8a2aa5b9e3d7329ec640809d2c06ca16e7273d6101785f9f",
}
REQUIRED_ARTIFACTS = frozenset({"comparison.csv", "reveal.json", "summary.md"})
GATE_METRICS = (
    "force_rmse_n",
    "contact_ratio_pct",
    "peak_force_n",
    "tangent_rmse_mm",
    "saturation_pct",
)
SUMMARY_PASS_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(\d+)/(\d+)\s*\|")


@dataclass(frozen=True)
class PublishedResultAudit:
    protocol_sha256: str
    artifact_count: int
    row_count: int
    method_count: int
    case_count: int
    pass_counts: dict[str, int]
    primary_rule_passed: bool

    def render(self) -> str:
        decision = "PASS" if self.primary_rule_passed else "FAIL"
        counts = ", ".join(
            f"{method}={count}/{self.case_count}" for method, count in self.pass_counts.items()
        )
        return "\n".join(
            (
                "audit: PASS",
                f"protocol_sha256: {self.protocol_sha256}",
                f"artifacts: {self.artifact_count}",
                (
                    f"comparison: {self.row_count} rows, "
                    f"{self.method_count} methods x {self.case_count} cases"
                ),
                f"pass_counts: {counts}",
                f"frozen_primary_rule: {decision}",
            )
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path.name}")
    return value


def _verify_v05_protocol(protocol_path: Path) -> tuple[dict[str, Any], str]:
    """Load the exact published v0.5 protocol, independent of later experiments."""
    protocol_hash = _sha256(protocol_path)
    checksum_path = protocol_path.with_name("protocol.sha256")
    expected_checksum = f"{protocol_hash}  protocol.json\n"
    if checksum_path.read_text(encoding="utf-8") != expected_checksum:
        raise ValueError("protocol checksum mismatch")
    if protocol_hash != PUBLISHED_PROTOCOL_SHA256:
        raise ValueError("protocol is not the published v0.5 contract")

    protocol = _read_json_object(protocol_path)
    if protocol.get("format_version") != PUBLISHED_PROTOCOL_FORMAT_VERSION:
        raise ValueError("unsupported published protocol format")
    if protocol.get("experiment_id") != PUBLISHED_EXPERIMENT_ID:
        raise ValueError("unexpected published experiment id")

    runs = protocol.get("training", {}).get("runs")
    if not isinstance(runs, list):
        raise TypeError("published protocol training runs must be a list")
    resolved_protocol_dir = protocol_path.resolve().parent
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("published protocol training run must be an object")
        for file_field, hash_field in (
            ("policy_file", "policy_sha256"),
            ("training_curve_file", "training_curve_sha256"),
            ("training_plot_file", "training_plot_sha256"),
        ):
            filename = run.get(file_field)
            expected_hash = run.get(hash_field)
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError(f"unsafe published artifact path: {filename}")
            artifact_path = protocol_path.parent / filename
            if artifact_path.resolve().parent != resolved_protocol_dir:
                raise ValueError(f"published artifact escapes protocol directory: {filename}")
            if not artifact_path.is_file() or _sha256(artifact_path) != expected_hash:
                raise ValueError(f"artifact checksum mismatch: {filename}")
    return protocol, protocol_hash


def _verify_manifest(result_dir: Path, protocol_hash: str) -> tuple[dict[str, Any], int]:
    manifest_path = result_dir / "manifest.json"
    manifest = _read_json_object(manifest_path)
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("manifest protocol hash mismatch")
    artifacts = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS.issubset(artifacts):
        raise ValueError("manifest does not cover the required result artifacts")

    resolved_result_dir = result_dir.resolve()
    for filename, expected_hash in artifacts.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"unsafe manifest artifact path: {filename}")
        path = result_dir / filename
        if path.resolve().parent != resolved_result_dir or not path.is_file():
            raise ValueError(f"missing manifest artifact: {filename}")
        if _sha256(path) != expected_hash:
            raise ValueError(f"artifact checksum mismatch: {filename}")
    return manifest, len(artifacts)


def _verify_published_archive(result_dir: Path, manifest: dict[str, Any]) -> None:
    if _sha256(result_dir / "manifest.json") != PUBLISHED_MANIFEST_SHA256:
        raise ValueError("published v0.5 manifest checksum mismatch")
    if manifest.get("artifact_sha256") != PUBLISHED_ARTIFACT_SHA256:
        raise ValueError("published v0.5 artifact map mismatch")
    complete_path = result_dir / "COMPLETE"
    if not complete_path.is_file() or _sha256(complete_path) != PUBLISHED_COMPLETE_SHA256:
        raise ValueError("published v0.5 completion marker mismatch")


def _read_comparison(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = set(reader.fieldnames or ())
    required_columns = {
        "method",
        "training_run_id",
        "policy_sha256",
        "case",
        "scenario_seed",
        "simulation_seed",
        "gate_pass",
        "failed_checks",
        *GATE_METRICS,
    }
    missing = sorted(required_columns - columns)
    if missing:
        raise ValueError(f"comparison.csv is missing columns: {', '.join(missing)}")
    return rows


def _expected_methods(protocol: dict[str, Any]) -> tuple[str, ...]:
    run_count = len(protocol["training"]["runs"])
    residual = tuple(f"torque_residual_run_{index:02d}" for index in range(run_count))
    return BASELINE_METHODS + residual


def _verify_shape(
    rows: list[dict[str, str]],
    methods: tuple[str, ...],
    case_count: int,
) -> tuple[str, ...]:
    expected_rows = len(methods) * case_count
    if len(rows) != expected_rows:
        raise ValueError(
            f"comparison row count mismatch: expected {expected_rows}, got {len(rows)}"
        )
    cases = tuple(f"blind_{index:02d}" for index in range(case_count))
    observed_pairs = [(row["case"], row["method"]) for row in rows]
    if len(set(observed_pairs)) != len(observed_pairs):
        raise ValueError("comparison contains duplicate case/method rows")
    if set(observed_pairs) != {(case, method) for case in cases for method in methods}:
        raise ValueError("comparison must contain every frozen case/method pair exactly once")
    return cases


def _integer(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"comparison has a malformed {field}: {value}") from error


def _verify_beacon_evidence(
    protocol: dict[str, Any],
    reveal: dict[str, Any],
) -> None:
    """Rebuild the v0.5 blind seeds from the stored, first-reveal beacon evidence."""
    contract = protocol["blind_contract"]
    beacon_contract = contract["beacon"]
    audit = reveal.get("beacon_verification_audit")
    if not isinstance(audit, dict):
        raise TypeError("reveal beacon verification audit must be an object")
    if (
        audit.get("exit_code") != 0
        or audit.get("relay_equality_verified") is not True
        or audit.get("signature_hash_verified") is not True
    ):
        raise ValueError("stored beacon verification audit did not pass")

    verifier = audit.get("verifier_output")
    verification_contract = beacon_contract["verification"]
    if not isinstance(verifier, dict):
        raise TypeError("stored beacon verifier output must be an object")
    if (
        verifier.get("verifier") != verification_contract["client"]
        or verifier.get("verifier_version") != verification_contract["client_version"]
        or verifier.get("mode") != "exact-round"
        or verifier.get("cryptographic_signature_verified") is not True
        or verifier.get("round") != beacon_contract["round"]
    ):
        raise ValueError("stored beacon verifier output violates the frozen contract")

    relays = verification_contract["relays"]
    responses = verifier.get("responses")
    if not isinstance(responses, list) or len(responses) != len(relays):
        raise ValueError("stored beacon evidence must contain both frozen relays")
    canonical_beacon: dict[str, Any] | None = None
    for relay, response in zip(relays, responses, strict=True):
        if not isinstance(response, dict):
            raise TypeError("stored beacon relay response must be an object")
        if response.get("base_url") != relay:
            raise ValueError("stored beacon relay identity or order changed")
        if response.get("chain_url") != f"{relay}/{beacon_contract['chain_hash']}":
            raise ValueError("stored beacon request did not use the pinned chain")
        if response.get("chain_info") != beacon_contract["chain_info"]:
            raise ValueError("stored beacon chain info changed")
        beacon = response.get("beacon")
        if not isinstance(beacon, dict) or beacon.get("round") != beacon_contract["round"]:
            raise ValueError("stored beacon response has the wrong round")
        randomness = beacon.get("randomness")
        signature = beacon.get("signature")
        if not isinstance(randomness, str) or not isinstance(signature, str):
            raise TypeError("stored beacon signature and randomness must be strings")
        try:
            derived_randomness = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
        except ValueError as error:
            raise ValueError("stored beacon signature must be hexadecimal") from error
        if derived_randomness != randomness:
            raise ValueError("stored beacon randomness does not match the signature hash")
        if canonical_beacon is None:
            canonical_beacon = beacon
        elif beacon != canonical_beacon:
            raise ValueError("stored relay beacons do not agree")
    assert canonical_beacon is not None
    if canonical_beacon.get("randomness") != PUBLISHED_BEACON_RANDOMNESS:
        raise ValueError("stored beacon randomness is not the published v0.5 value")

    normalized_beacon = {
        "chain_hash": beacon_contract["chain_hash"],
        "round": beacon_contract["round"],
        "signature": canonical_beacon["signature"],
        "randomness": canonical_beacon["randomness"],
    }
    if reveal.get("beacon") != normalized_beacon:
        raise ValueError("reveal beacon does not match its stored verification evidence")

    root = hashlib.sha256(
        bytes.fromhex(PUBLISHED_PROTOCOL_SHA256)
        + bytes.fromhex(str(canonical_beacon["randomness"]))
    ).digest()
    if reveal.get("blind_root_sha256") != root.hex():
        raise ValueError("reveal blind root does not match protocol and beacon")

    case_count = int(contract["cases"])

    def derive(namespace: str, index: int) -> int:
        message = f"v0.5/{namespace}/{index}".encode()
        digest = hmac.new(root, message, hashlib.sha256).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    scenario_seeds = tuple(derive("blind/scenario", index) for index in range(case_count))
    simulation_seeds = tuple(derive("blind/noise", index) for index in range(case_count))
    if reveal.get("scenario_seeds") != list(scenario_seeds):
        raise ValueError("reveal scenario seeds do not follow the frozen derivation")
    if reveal.get("simulation_seeds") != list(simulation_seeds):
        raise ValueError("reveal simulation seeds do not follow the frozen derivation")
    if reveal.get("reporting_seed") != derive("report/bootstrap", 0):
        raise ValueError("reveal reporting seed does not follow the frozen derivation")


def _verify_case_seeds(
    rows: list[dict[str, str]],
    cases: tuple[str, ...],
    reveal: dict[str, Any],
) -> None:
    scenario_seeds = reveal.get("scenario_seeds")
    simulation_seeds = reveal.get("simulation_seeds")
    if not (
        isinstance(scenario_seeds, list)
        and isinstance(simulation_seeds, list)
        and len(scenario_seeds) == len(simulation_seeds) == len(cases)
        and all(type(value) is int for value in scenario_seeds + simulation_seeds)
    ):
        raise ValueError("reveal must contain one integer scenario/noise seed per case")

    for index, case in enumerate(cases):
        selected = [row for row in rows if row["case"] == case]
        observed = {
            (
                _integer(row["scenario_seed"], "scenario_seed"),
                _integer(row["simulation_seed"], "simulation_seed"),
            )
            for row in selected
        }
        expected = {(scenario_seeds[index], simulation_seeds[index])}
        if observed != expected:
            raise ValueError(f"case seeds do not match reveal: {case}")


def _verify_policy_hashes(
    rows: list[dict[str, str]],
    protocol: dict[str, Any],
) -> None:
    for method in BASELINE_METHODS:
        selected = [row for row in rows if row["method"] == method]
        if any(row["policy_sha256"] or row["training_run_id"] for row in selected):
            raise ValueError(f"baseline row unexpectedly names a policy: {method}")

    for run_index, artifact in enumerate(protocol["training"]["runs"]):
        method = f"torque_residual_run_{run_index:02d}"
        selected = [row for row in rows if row["method"] == method]
        expected_hash = artifact["policy_sha256"]
        expected_run_id = f"run_{run_index:02d}"
        if any(row["policy_sha256"] != expected_hash for row in selected):
            raise ValueError(f"policy hash mismatch: {method}")
        if any(row["training_run_id"] != expected_run_id for row in selected):
            raise ValueError(f"training run id mismatch: {method}")


def _recompute_pass_counts(
    rows: list[dict[str, str]],
    methods: tuple[str, ...],
    protocol: dict[str, Any],
) -> dict[str, int]:
    gate = protocol["gate"]
    counts = dict.fromkeys(methods, 0)
    for row in rows:
        try:
            metrics = {name: float(row[name]) for name in GATE_METRICS}
        except ValueError as error:
            raise ValueError("comparison contains a malformed gate metric") from error
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("comparison contains a non-finite gate metric")
        checks = (
            (metrics["force_rmse_n"] <= float(gate["max_force_rmse_n"]), "force_rmse"),
            (
                metrics["contact_ratio_pct"] >= float(gate["min_contact_ratio_pct"]),
                "contact_ratio",
            ),
            (metrics["peak_force_n"] <= float(gate["max_peak_force_n"]), "peak_force"),
            (
                metrics["tangent_rmse_mm"] <= float(gate["max_tangent_rmse_mm"]),
                "tangent_rmse",
            ),
            (metrics["saturation_pct"] <= float(gate["max_saturation_pct"]), "saturation"),
        )
        failures = tuple(name for passed, name in checks if not passed)
        expected_pass = "yes" if not failures else "no"
        expected_failures = ";".join(failures)
        if row["gate_pass"] != expected_pass or row["failed_checks"] != expected_failures:
            raise ValueError(f"stored gate result mismatch: {row['case']}/{row['method']}")
        if not failures:
            counts[row["method"]] += 1
    return counts


def _verify_summary(
    summary_path: Path,
    methods: tuple[str, ...],
    case_count: int,
    pass_counts: dict[str, int],
    primary_rule_passed: bool,
) -> None:
    summary = summary_path.read_text(encoding="utf-8")
    summary_counts: dict[str, tuple[int, int]] = {}
    for line in summary.splitlines():
        match = SUMMARY_PASS_ROW.match(line)
        if match and match.group(1) in methods:
            summary_counts[match.group(1)] = (int(match.group(2)), int(match.group(3)))
    if set(summary_counts) != set(methods):
        raise ValueError("summary does not contain every method pass count")
    for method in methods:
        count, denominator = summary_counts[method]
        if count != pass_counts[method] or denominator != case_count:
            raise ValueError(f"summary pass count mismatch: {method}")
    expected_decision = "PASS" if primary_rule_passed else "FAIL"
    if f"Frozen primary rule: {expected_decision}." not in summary:
        raise ValueError("summary primary-rule decision mismatch")


def audit_published_results(
    protocol_path: Path,
    result_dir: Path,
) -> PublishedResultAudit:
    """Audit frozen inputs and published outputs without network access or simulation."""
    protocol, protocol_hash = _verify_v05_protocol(Path(protocol_path))
    result_dir = Path(result_dir)
    manifest, artifact_count = _verify_manifest(result_dir, protocol_hash)
    reveal = _read_json_object(result_dir / "reveal.json")
    if reveal.get("protocol_sha256") != protocol_hash:
        raise ValueError("reveal protocol hash mismatch")
    if reveal.get("experiment_id") != protocol["experiment_id"]:
        raise ValueError("reveal experiment id mismatch")
    if manifest.get("experiment_id") != protocol["experiment_id"]:
        raise ValueError("manifest experiment id mismatch")
    if manifest.get("freeze_tag") != protocol["freeze_tag"]:
        raise ValueError("manifest freeze tag mismatch")
    _verify_beacon_evidence(protocol, reveal)

    rows = _read_comparison(result_dir / "comparison.csv")
    methods = _expected_methods(protocol)
    case_count = int(protocol["blind_contract"]["cases"])
    cases = _verify_shape(rows, methods, case_count)
    _verify_case_seeds(rows, cases, reveal)
    _verify_policy_hashes(rows, protocol)
    pass_counts = _recompute_pass_counts(rows, methods, protocol)

    primary_rule = protocol["blind_contract"]["primary_rule"]
    required_passes = int(primary_rule["minimum_passes_per_policy"])
    residual_methods = methods[len(BASELINE_METHODS) :]
    primary_rule_passed = all(pass_counts[method] >= required_passes for method in residual_methods)
    _verify_summary(
        result_dir / "summary.md",
        methods,
        case_count,
        pass_counts,
        primary_rule_passed,
    )
    _verify_published_archive(result_dir, manifest)
    return PublishedResultAudit(
        protocol_sha256=protocol_hash,
        artifact_count=artifact_count,
        row_count=len(rows),
        method_count=len(methods),
        case_count=case_count,
        pass_counts=pass_counts,
        primary_rule_passed=primary_rule_passed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/franka_safety_preholdout/protocol.json"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results/franka_safety_blind"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        report = audit_published_results(args.protocol, args.result)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"audit: FAIL: {error}") from error
    print(report.render())


if __name__ == "__main__":
    main()
